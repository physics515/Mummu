//! The recovery's production wiring, driven through the REAL engine.
//!
//! Every test here sends chats through the handlers the router serves
//! (`POST /api/chat`, the shim's `/api/chat`), which run `engine::run_chat`
//! → `plan_fit` → `drive` → the slot's load closure → mummu's Qwen2 loader →
//! a real generation on the CPU backend, and injects the incident's failures
//! at the points `fault` documents. The model is a two-layer Qwen2 written
//! for each test (a few kilobytes: `config.json`, `model.safetensors`, a
//! word-level `tokenizer.json`), so the suite needs no weights on disk and no
//! GPU, and cleans up after itself.
//!
//! These exist because unit tests of the helpers were not enough: in v0.3.2's
//! first cut, one-line mutations of the wiring — the stale-model predicate,
//! the load check's mark, the call that resets the failure count, the
//! eviction, the refusals, the in-flight hold, the drain — all left the
//! whole suite green. Each test below names the invariant it pins, and the
//! commit that added them lists the mutation each one fails against.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::response::Response;
use futures::StreamExt;
use mummu_num::f32_from_usize;
use serde_json::{Value, json};

use super::{Arm, arm, last_load, loads};
use crate::engine::{self, BackendChoice};
use crate::recovery::{self, ChatError, DeviceKey, Recovery};
use crate::test_seams::{self, Scratch};

/// A catalog name, so the handlers find the spec and its directory. Its
/// files are the tiny fixture, never the real 0.5B.
const MODEL: &str = "qwen2.5-0.5b-instruct";

const CUDA0: DeviceKey = DeviceKey::Cubecl {
    type_id: 0,
    index: 0,
};

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

/// A little-endian safetensors file of F32 tensors.
fn safetensors(tensors: &[(String, Vec<usize>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (i, (name, shape)) in tensors.iter().enumerate() {
        let begin = data.len();
        let n: usize = shape.iter().product();
        let is_norm = name.ends_with("norm.weight");
        for k in 0..n {
            // Norms at one; everything else small and deterministic, so the
            // forward is real arithmetic and never a NaN.
            let v = if is_norm {
                1.0f32
            } else {
                (f32_from_usize((k * 7 + i * 13) % 11) - 5.0) * 0.02
            };
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let mut head = serde_json::to_vec(&Value::Object(header)).expect("header");
    while !head.len().is_multiple_of(8) {
        head.push(b' ');
    }
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend(head);
    out.extend(data);
    out
}

/// A two-layer Qwen2 (hidden 16, vocab 64) with a word-level tokenizer.
/// `eos_token_id` is outside the vocab, so a generation always runs to its
/// `max_tokens` and always produces a first token.
fn write_tiny_qwen2(dir: &Path) {
    std::fs::create_dir_all(dir).expect("model dir");
    let (hidden, inter, layers, heads, kv) = (16usize, 32usize, 2usize, 4usize, 2usize);
    let head_dim = hidden / heads;
    let vocab = 64usize;
    std::fs::write(
        dir.join("config.json"),
        json!({
            "architectures": ["Qwen2ForCausalLM"],
            "vocab_size": vocab, "hidden_size": hidden, "intermediate_size": inter,
            "num_hidden_layers": layers, "num_attention_heads": heads,
            "num_key_value_heads": kv, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": true, "eos_token_id": 999,
            "max_position_embeddings": 512
        })
        .to_string(),
    )
    .expect("config");
    let mut vocab_map = serde_json::Map::new();
    vocab_map.insert("<unk>".into(), json!(0));
    for id in 1..vocab {
        vocab_map.insert(format!("w{id}"), json!(id));
    }
    std::fs::write(
        dir.join("tokenizer.json"),
        json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab_map, "unk_token": "<unk>"}
        })
        .to_string(),
    )
    .expect("tokenizer");
    let mut tensors = vec![
        ("model.embed_tokens.weight".to_owned(), vec![vocab, hidden]),
        ("model.norm.weight".to_owned(), vec![hidden]),
    ];
    for l in 0..layers {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        tensors.extend([
            (p("input_layernorm.weight"), vec![hidden]),
            (p("post_attention_layernorm.weight"), vec![hidden]),
            (p("self_attn.q_proj.weight"), vec![heads * head_dim, hidden]),
            (p("self_attn.q_proj.bias"), vec![heads * head_dim]),
            (p("self_attn.k_proj.weight"), vec![kv * head_dim, hidden]),
            (p("self_attn.k_proj.bias"), vec![kv * head_dim]),
            (p("self_attn.v_proj.weight"), vec![kv * head_dim, hidden]),
            (p("self_attn.v_proj.bias"), vec![kv * head_dim]),
            (p("self_attn.o_proj.weight"), vec![hidden, heads * head_dim]),
            (p("mlp.gate_proj.weight"), vec![inter, hidden]),
            (p("mlp.up_proj.weight"), vec![inter, hidden]),
            (p("mlp.down_proj.weight"), vec![hidden, inter]),
        ]);
    }
    std::fs::write(dir.join("model.safetensors"), safetensors(&tensors)).expect("weights");
}

/// One test's world: the fixture model under a scratch models root, the CPU
/// backend, clean recovery and engine state — and all of it put back, and the
/// directories removed, when it drops. Holds `progress_serial` for its whole
/// life (declared last, so it is released after the cleanup).
struct Fixture {
    root: Scratch,
    local: Scratch,
    _serial: tokio::sync::MutexGuard<'static, ()>,
}

impl Fixture {
    async fn new(name: &str) -> Self {
        let serial = crate::progress_serial().await;
        recovery::reset_for_tests();
        recovery::install_panic_hook();
        engine::register_devices();
        arm(Arm::default());
        engine::test_support::reset();
        mummu::progress::idle();
        let root = Scratch::new(name);
        write_tiny_qwen2(&root.path().join(MODEL));
        test_seams::set_models_root(Some(root.path().to_path_buf()));
        test_seams::set_backend(Some(BackendChoice::Cpu));
        let local = Scratch::new(&format!("{name}-local"));
        Self {
            root,
            local,
            _serial: serial,
        }
    }

    fn dir(&self) -> PathBuf {
        self.root.path().join(MODEL)
    }

    /// Run as the binary does: under a supervisor, whose exit is `exit`.
    fn supervise(&self, exit: fn(i32)) {
        recovery::supervise_for_tests(
            self.root.path(),
            self.local.path(),
            exit,
            never_called,
            Duration::from_secs(3600),
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        arm(Arm::default());
        engine::test_support::reset();
        recovery::reset_for_tests();
        test_seams::set_models_root(None);
        test_seams::set_backend(None);
        mummu::progress::idle();
    }
}

fn never_called(_: i32) {
    panic!("the exit watchdog fired in an engine test");
}

fn wait_for(n: &AtomicU32, at_least: u32, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    while n.load(SeqCst) < at_least && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    n.load(SeqCst) >= at_least
}

fn request() -> Bytes {
    Bytes::from(
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there"}],
            "options": {"temperature": 0, "max_tokens": 3},
        })
        .to_string(),
    )
}

/// An SSE body as the frames a client parses out of it.
async fn frames(response: Response) -> Vec<Value> {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    String::from_utf8_lossy(&body)
        .split("\n\n")
        .filter_map(|f| f.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).expect("each frame is JSON"))
        .collect()
}

/// One `POST /api/chat`, read to the end.
async fn chat() -> Vec<Value> {
    frames(crate::chat(request()).await).await
}

fn last(frames: &[Value]) -> &Value {
    frames
        .last()
        .expect("a stream with no frames: the incident's empty 200")
}

#[track_caller]
fn assert_answered(frames: &[Value]) {
    let last = last(frames);
    assert_eq!(last["type"], json!("done"), "{frames:?}");
    assert!(last["tokens"].as_u64().is_some_and(|t| t >= 1), "{last}");
}

#[track_caller]
fn assert_failed(frames: &[Value], recovery: Recovery) -> String {
    let last = last(frames);
    assert_eq!(last["type"], json!("error"), "{frames:?}");
    assert_eq!(last["recovery"], json!(recovery.as_str()), "{last}");
    last["error"].as_str().expect("a message").to_owned()
}

async fn health() -> u16 {
    crate::health().await.status().as_u16()
}

fn poisoned_on(key: DeviceKey) -> bool {
    recovery::snapshot()
        .iter()
        .any(|d| d.key == key && d.poisoned)
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

/// The fixture is a real model on the real path: it loads once, answers, and
/// a second chat is a hit on the slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_fixture_loads_once_and_answers_through_the_real_engine() {
    let fx = Fixture::new("happy").await;
    let before = loads();
    assert_answered(&chat().await);
    assert_eq!(loads(), before + 1);
    assert_answered(&chat().await);
    assert_eq!(loads(), before + 1, "a resident model is a hit");
    assert_eq!(engine::resident_dirs(), vec![fx.dir()]);
    assert_eq!(health().await, 200);
}

/// The incident's load, on the real load path: cubecl's device thread
/// panics and swallows an OOM while the loader returns `Ok`. The load check
/// (the epoch mark taken BEFORE the load, compared after the sync) must fail
/// it AS A LOAD: an error frame, a 503, nothing in the slot, and the
/// residency note `plan_fit` wrote before the load taken back. The next chat
/// loads fresh — and the CPU load that answers it does NOT clear the card's
/// poison, because it proves nothing about the card.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_oom_during_the_load_fails_the_load_and_nothing_half_placed_is_kept() {
    let fx = Fixture::new("load-oom").await;
    // As plan_fit's GGUF ladder notes a model it plans, before the load.
    engine::test_support::note(BackendChoice::Cpu, &fx.dir());
    arm(Arm {
        load_oom: 1,
        ..Arm::default()
    });
    let (before, epoch) = (loads(), recovery::fault_epoch());
    let message = assert_failed(&chat().await, Recovery::Reload);
    assert!(message.contains("out of device memory"), "{message}");
    assert_eq!(loads(), before + 1, "the load ran");
    assert!(recovery::fault_epoch() > epoch);
    assert!(
        engine::resident_dirs().is_empty(),
        "a load the device failed under left its model in the slot"
    );
    assert!(
        !engine::test_support::noted(&fx.dir()),
        "a load that did not land left the model noted as resident"
    );
    assert!(
        poisoned_on(CUDA0),
        "charged to the device whose thread failed"
    );
    assert_eq!(health().await, 503);

    assert_answered(&chat().await);
    assert_eq!(loads(), before + 2, "the next request loads it fresh");
    assert_eq!(engine::resident_dirs(), vec![fx.dir()]);
    assert_eq!(
        health().await,
        503,
        "a clean CPU load cleared the GPU's poison (Device::sync on flex is a no-op)"
    );
    assert!(poisoned_on(CUDA0));

    // A load that DID land keeps its residency note through a hit.
    engine::test_support::note(BackendChoice::Cpu, &fx.dir());
    assert_answered(&chat().await);
    assert!(
        engine::test_support::noted(&fx.dir()),
        "a model that landed lost its residency note"
    );
    // The fixture holds `progress_serial` for the whole test: released
    // here, at the end, and not at its last mention above.
    drop(fx);
}

/// A device failure no request caught — the hook saw it on a device thread
/// while nothing ran — leaves the resident model in the slot; only its fault
/// stamp, checked under the slot lock, stops it being served. The next chat
/// must reload.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_device_failure_nobody_caught_makes_the_resident_model_stale() {
    let fx = Fixture::new("stale").await;
    assert_answered(&chat().await);
    let loaded = loads();
    arm(Arm {
        device_oom_now: true,
        ..Arm::default()
    });
    assert_eq!(
        health().await,
        503,
        "the failure is reported the moment it happens"
    );
    assert_eq!(engine::resident_dirs(), vec![fx.dir()], "still resident");

    assert_answered(&chat().await);
    assert_eq!(
        loads(),
        loaded + 1,
        "a model loaded before a device failure was served instead of reloaded"
    );
    assert_eq!(
        health().await,
        503,
        "the CPU reload cleared a failure on the card"
    );
}

/// MAJOR 1: the failure arriving as an `Err` — an eager readback's
/// `ServerUnhealthy` — moves the fault epoch exactly as a panic does, and the
/// model it failed on is gone from the slot the moment the request ends.
/// The next chat loads fresh, and that load (on the model's own device)
/// clears the poison.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_error_path_device_failure_moves_the_epoch_and_the_model_is_never_served_again() {
    let fx = Fixture::new("err-path").await;
    assert_answered(&chat().await);
    let (loaded, epoch) = (loads(), recovery::fault_epoch());
    arm(Arm {
        read_invalid_err: 1,
        ..Arm::default()
    });
    let message = assert_failed(&chat().await, Recovery::Reload);
    assert!(message.contains("invalid state"), "{message}");
    assert!(
        recovery::fault_epoch() > epoch,
        "an Err-path device failure did not move the fault epoch"
    );
    assert!(
        engine::resident_dirs().is_empty(),
        "the model the device failed under is still in the slot"
    );
    assert_eq!(health().await, 503);

    assert_answered(&chat().await);
    assert_eq!(loads(), loaded + 1);
    assert_eq!(engine::resident_dirs(), vec![fx.dir()]);
    assert_eq!(
        health().await,
        200,
        "a clean load on the failed device clears it"
    );
}

/// MAJOR 1's scenario: a request QUEUED on the slot behind an Err-path
/// failure is answered by a fresh model, never handed the one that failed.
/// (Two layers hold this — the eviction through the slot guard and the
/// epoch — and each has its own test above; this one fails only if both
/// go, and it is here because it is the scenario the review described.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_queued_behind_an_error_path_failure_gets_a_fresh_model() {
    let _fx = Fixture::new("err-queued").await;
    assert_answered(&chat().await);
    let loaded = loads();
    arm(Arm {
        stall_ms: 400,
        read_invalid_err: 1,
        ..Arm::default()
    });
    let a = tokio::spawn(chat());
    tokio::time::sleep(Duration::from_millis(100)).await;
    let b = tokio::spawn(chat());
    let (a, b) = (a.await.expect("a"), b.await.expect("b"));
    assert_failed(&a, Recovery::Reload);
    assert_answered(&b);
    assert_eq!(
        loads(),
        loaded + 1,
        "the queued request was handed the model the device had just failed under"
    );
}

static COUNT_EXITS: AtomicU32 = AtomicU32::new(0);
fn count_exit(_: i32) {
    COUNT_EXITS.fetch_add(1, SeqCst);
}

/// The escalation, end to end, per device. A token between two failures on
/// the model's own device resets that device's count: two reloads, no exit.
/// But two failures on the CARD with a CPU token between them are still two
/// in a row for the card — the sticky fault a restart is for — and the
/// second one restarts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_token_resets_the_failure_count_only_on_the_device_that_produced_it() {
    let fx = Fixture::new("count").await;
    fx.supervise(count_exit);
    let exits = COUNT_EXITS.load(SeqCst);

    arm(Arm {
        read_invalid: 1,
        ..Arm::default()
    });
    assert_failed(&chat().await, Recovery::Reload);
    assert_answered(&chat().await);
    arm(Arm {
        read_invalid: 1,
        ..Arm::default()
    });
    assert_failed(&chat().await, Recovery::Reload);
    assert!(
        !recovery::restarting(),
        "a token between two failures did not reset the count"
    );

    arm(Arm {
        load_oom: 1,
        ..Arm::default()
    });
    assert_failed(&chat().await, Recovery::Reload);
    assert_answered(&chat().await);
    // The CPU model is resident and fine; the next request must load again
    // for the card to be tried again (POST /api/unload, the real path).
    assert_eq!(crate::unload().await.status(), 200);
    arm(Arm {
        load_oom: 1,
        ..Arm::default()
    });
    assert_failed(&chat().await, Recovery::Restart);
    assert!(recovery::restarting());
    assert!(
        wait_for(&COUNT_EXITS, exits + 1, Duration::from_secs(5)),
        "the restart was decided and never taken"
    );
    // The fixture holds `progress_serial` for the whole test: released
    // here, at the end, and not at its last mention above.
    drop(fx);
}

static LATCH_EXITS: AtomicU32 = AtomicU32::new(0);
fn latch_exit(_: i32) {
    LATCH_EXITS.fetch_add(1, SeqCst);
}

/// MINOR 9: the decision to restart is latched while the failing request
/// still holds the slot, so the request queued behind it — which got past
/// every entry check before the decision existed — is refused at the load
/// closure, the first point it would touch the device. It must not load.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_queued_behind_a_restart_decision_is_refused_not_loaded() {
    let fx = Fixture::new("latch").await;
    fx.supervise(latch_exit);
    let exits = LATCH_EXITS.load(SeqCst);
    arm(Arm {
        read_invalid: 1,
        ..Arm::default()
    });
    assert_failed(&chat().await, Recovery::Reload);

    let loaded = loads();
    arm(Arm {
        stall_ms: 400,
        read_invalid: 1,
        ..Arm::default()
    });
    let a = tokio::spawn(chat());
    tokio::time::sleep(Duration::from_millis(100)).await;
    let b = tokio::spawn(chat());
    let (a, b) = (a.await.expect("a"), b.await.expect("b"));
    assert_failed(&a, Recovery::Restart);
    let refusal = assert_failed(&b, Recovery::Restart);
    assert_eq!(refusal, recovery::RESTARTING_MESSAGE, "{b:?}");
    assert!(!b.iter().any(|f| f["type"] == json!("delta")), "{b:?}");
    assert_eq!(
        loads(),
        loaded + 1,
        "the queued request started a load on a device the process had decided to leave"
    );
    assert!(wait_for(&LATCH_EXITS, exits + 1, Duration::from_secs(5)));
    // The fixture holds `progress_serial` for the whole test: released
    // here, at the end, and not at its last mention above.
    drop(fx);
}

static REFUSE_EXITS: AtomicU32 = AtomicU32::new(0);
fn refuse_exit(_: i32) {
    REFUSE_EXITS.fetch_add(1, SeqCst);
}

/// While the process is exiting, a NEW chat is refused at the door with a
/// real status on both surfaces — a 503, not a 200 whose stream then says
/// so — and nothing is loaded.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_entry_point_refuses_a_new_chat_while_restarting() {
    let fx = Fixture::new("refuse").await;
    fx.supervise(refuse_exit);
    let exits = REFUSE_EXITS.load(SeqCst);
    let _ = recovery::record_failure(MODEL, &[CUDA0], "CUDA_ERROR_ILLEGAL_ADDRESS");
    let _ = recovery::record_failure(MODEL, &[CUDA0], "CUDA_ERROR_ILLEGAL_ADDRESS");
    assert!(recovery::restarting());
    let before = loads();

    let native = crate::chat(request()).await;
    assert_eq!(native.status(), 503, "POST /api/chat during a restart");
    let shim = Box::pin(crate::shim::chat(Bytes::from(
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there"}],
            "stream": true,
            "options": {"temperature": 0, "num_predict": 3},
        })
        .to_string(),
    )))
    .await;
    assert_eq!(
        shim.status(),
        503,
        "the shim's streamed /api/chat during a restart"
    );
    assert_eq!(loads(), before, "a refused chat loaded a model");
    assert!(wait_for(&REFUSE_EXITS, exits + 1, Duration::from_secs(5)));
    // The fixture holds `progress_serial` for the whole test: released
    // here, at the end, and not at its last mention above.
    drop(fx);
}

/// A chat response still being written counts as in flight — from the moment
/// it is handed out until its last frame — so an exiting process waits for
/// the error frame it owes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_open_chat_response_counts_as_in_flight_until_its_last_frame() {
    let _fx = Fixture::new("in-flight").await;
    assert_answered(&chat().await);
    assert_eq!(recovery::in_flight(), 0);
    arm(Arm {
        stall_ms: 400,
        ..Arm::default()
    });
    let mut body = crate::chat(request()).await.into_body().into_data_stream();
    // Start the stream (a response that let go of its claim when first
    // polled would show it here) while the generation is still stalled.
    let first = tokio::time::timeout(Duration::from_millis(150), body.next()).await;
    assert!(first.is_err(), "no frame yet: the generation is stalled");
    assert_eq!(
        recovery::in_flight(),
        1,
        "an open chat response is not counted — an exit would not wait for it"
    );
    while body.next().await.is_some() {}
    assert_eq!(recovery::in_flight(), 0, "and it is released at the end");
}

static DRAIN_EXITS: AtomicU32 = AtomicU32::new(0);
fn drain_exit(_: i32) {
    DRAIN_EXITS.fetch_add(1, SeqCst);
}

/// The exit waits for open responses before it goes (up to its budget): a
/// client still reading its error frame holds the process up; when it is
/// done, the exit is taken.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_exit_waits_for_open_responses_before_it_goes() {
    let fx = Fixture::new("drain").await;
    fx.supervise(drain_exit);
    let exits = DRAIN_EXITS.load(SeqCst);
    arm(Arm {
        read_invalid: 1,
        ..Arm::default()
    });
    assert_failed(&chat().await, Recovery::Reload);
    // A client still reading the frames it is owed.
    let open = recovery::InFlight::enter();
    arm(Arm {
        read_invalid: 1,
        ..Arm::default()
    });
    assert_failed(&chat().await, Recovery::Restart);
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(
        DRAIN_EXITS.load(SeqCst),
        exits,
        "the process exited with a response still open"
    );
    drop(open);
    assert!(
        wait_for(&DRAIN_EXITS, exits + 1, Duration::from_secs(3)),
        "and it did not exit once the response closed"
    );
    // The fixture holds `progress_serial` for the whole test: released
    // here, at the end, and not at its last mention above.
    drop(fx);
}

/// MINOR 6: mummu catches some device-thread panics on purpose — the kernel
/// gap `run_readback_with_fallback` retries around — and one of those must
/// not poison a healthy server: the chat completes, nothing is reloaded,
/// health stays 200. An OOM in the very same window still poisons the
/// device and makes the model stale.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_handled_kernel_gap_does_not_poison_but_an_oom_in_the_same_window_does() {
    let _fx = Fixture::new("kernel-gap").await;
    assert_answered(&chat().await);
    let (loaded, epoch) = (loads(), recovery::fault_epoch());
    arm(Arm {
        kernel_gap: 1,
        ..Arm::default()
    });
    assert_answered(&chat().await);
    assert_eq!(
        recovery::fault_epoch(),
        epoch,
        "a panic mummu handled was counted as a device failure"
    );
    assert_eq!(health().await, 200, "a healthy server was marked poisoned");
    assert_answered(&chat().await);
    assert_eq!(
        loads(),
        loaded,
        "the model was reloaded for a handled panic"
    );

    arm(Arm {
        kernel_gap_oom: 1,
        ..Arm::default()
    });
    assert_answered(&chat().await);
    assert!(recovery::fault_epoch() > epoch);
    assert!(
        poisoned_on(CUDA0),
        "an OOM inside a retry window went unseen"
    );
    assert_eq!(health().await, 503);
    assert_answered(&chat().await);
    assert_eq!(
        loads(),
        loaded + 1,
        "the model the OOM happened under was served again"
    );
}

/// MINOR 10: a request planned by `plan_fit`'s "already resident" shortcut
/// (no fit check, placeholder policy) that finds the resident copy stale
/// must plan its load afresh — never load under the shortcut's plan.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_load_planned_against_a_resident_copy_that_went_stale_is_planned_afresh() {
    let fx = Fixture::new("replan").await;
    assert_answered(&chat().await);
    let loaded = loads();
    // As plan_fit's GGUF ladder notes a model it plans.
    engine::test_support::note(BackendChoice::Cpu, &fx.dir());
    arm(Arm {
        device_oom_now: true,
        ..Arm::default()
    });
    assert_answered(&chat().await);
    assert_eq!(loads(), loaded + 1);
    let info = last_load().expect("a load ran");
    assert!(info.found_stale, "{info:?}");
    assert!(
        !info.from_resident,
        "the reload ran under the resident shortcut's plan — no fit check: {info:?}"
    );
    // The fixture holds `progress_serial` for the whole test: released
    // here, at the end, and not at its last mention above.
    drop(fx);
}

/// MINOR 7: the eviction outside the engine says exactly what it found — an
/// empty slot is not a dropped model, and a slot another request holds is
/// neither dropped nor claimed to hold a model from before the failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_eviction_says_only_what_happened() {
    use mummu::cache::Cleared;
    let fx = Fixture::new("evict-says").await;
    let empty = engine::evict_after_device_failure();
    assert_eq!(empty, Cleared::Empty);
    let line = engine::eviction_line(&empty);
    assert!(
        !line.contains("dropped the resident model") && line.contains("nothing to drop"),
        "{line}"
    );

    assert_answered(&chat().await);
    let dropped = engine::evict_after_device_failure();
    assert_eq!(dropped, Cleared::Dropped(fx.dir()));
    assert!(
        engine::eviction_line(&dropped).contains("dropped the resident model"),
        "{dropped:?}"
    );
    assert_eq!(engine::resident_dirs(), [] as [std::path::PathBuf; 0]);

    arm(Arm {
        stall_ms: 400,
        ..Arm::default()
    });
    let held = tokio::spawn(chat());
    tokio::time::sleep(Duration::from_millis(150)).await;
    let busy = engine::evict_after_device_failure();
    assert_eq!(busy, Cleared::Busy);
    let line = engine::eviction_line(&busy);
    assert!(
        !line.contains("dropped the resident model") && line.contains("held by another request"),
        "{line}"
    );
    assert_answered(&held.await.expect("held"));
}

/// A device failure decided outside the engine — a generation that is not
/// the engine's, reported through `recovery::contain` — still evicts the
/// resident model rather than leaving it for the fault stamp alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_device_failure_decided_outside_the_engine_still_evicts_the_resident_model() {
    let _fx = Fixture::new("fallback-evict").await;
    assert_answered(&chat().await);
    let stream = crate::spawn_chat(MODEL.into(), false, |_| async {
        Err::<engine::ChatResult, _>(ChatError::request(super::INVALID_READ_ERR))
    });
    let f = frames(crate::sse_response(stream.rx, Some(stream.inflight))).await;
    assert_failed(&f, Recovery::Reload);
    assert!(
        engine::resident_dirs().is_empty(),
        "the resident model outlived a device failure"
    );
}
