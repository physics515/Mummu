//! Records — and sanity-checks — the qwen4exp (Qwen3.8-Flash-Next) parity
//! fixture from a llama.cpp reference that is ALREADY RUNNING.
//!
//! The reference is too big to spawn per test (111 GB mmap under a container
//! memory cap, minutes to load; see `tests/qwen4exp_fixture`), so this test
//! attaches to it by URL. Prompts are tokenized by mummu's own GGUF tokenizer
//! and sent as id arrays, so the replay feeds our model exactly the ids the
//! reference evaluated. Recording (ignored; skips with a message when either
//! variable is unset):
//!
//! ```text
//! docker run -d --name flashnext-ref-server --memory 48g --memory-swap 48g \
//!   -p 127.0.0.1:18650:8080 -v "$MUMMU_QWEN4EXP_DIR":/models:ro \
//!   --entrypoint /app/llama-server ghcr.io/ggml-org/llama.cpp:full \
//!   -m /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
//!   --host 0.0.0.0 --port 8080 -c 2048 -t 16 -ngl 0 -np 1 --no-webui
//! MUMMU_QWEN4EXP_DIR=/path/to/qwen3.8-flash-next \
//! MUMMU_QWEN4EXP_REF_URL=http://127.0.0.1:18650 \
//! MUMMU_QWEN4EXP_REF_IMAGE="ghcr.io/ggml-org/llama.cpp:full@sha256:…" \
//! MUMMU_QWEN4EXP_REF_ARGS="…the server command line…" \
//!   cargo test -p mummu --test parity_qwen4exp_record -- \
//!     --ignored record_the_qwen4exp_fixture --nocapture
//! ```
//!
//! Recorded 2026-09-16 that way from ghcr.io/ggml-org/llama.cpp:full (b10991)
//! on the `NVMe` copy: ~9.5 min for the first leg, ~5 min for the second.
//!
//! Point `MUMMU_QWEN4EXP_DIR` at an `NVMe` copy: the tokenizer read is small,
//! but the reference's random expert reads from spinning disks are ~1000x
//! slower.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use mummu_testkit::gguf_compare;
use mummu_testkit::llama_ref;
use mummu_testkit::qwen4exp_fixture;

use std::path::PathBuf;

use gguf_compare::{MAX_TOKENS, PROMPT, TOP_K};
use llama_ref::{LlamaServer, parse_completion};
use mummu::gguf::GgufFile;
use mummu_num::narrow;
use qwen4exp_fixture::{
    FIRST_SHARD, FIXTURE_PATH, FORMAT, Fixture, LEGS, LONG_FIXTURE_PATH, LONG_MAX_TOKENS,
    LONG_PROMPT, Leg, ModelInfo, N_PROBS, ReferenceInfo, Step, TopEntry, compare_leg,
    render_prompt_ids,
};

/// Shard 1 of the split set, when `MUMMU_QWEN4EXP_DIR` names a directory
/// holding it.
fn first_shard() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN4EXP_DIR")?);
    let p = dir.join(FIRST_SHARD);
    p.is_file().then_some(p)
}

fn tokenizer(first: &std::path::Path) -> tokenizers::Tokenizer {
    let f = GgufFile::open_sharded(first).expect("split set opens");
    mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf")
}

/// A `/completion` entry's text. The server sends `""` for control tokens it
/// does not render (`<|im_end|>` arrives empty), so absence is not an error.
fn token_text(v: &serde_json::Value) -> String {
    v["token"].as_str().unwrap_or_default().to_string()
}

#[test]
#[ignore = "needs a running qwen4exp llama-server (MUMMU_QWEN4EXP_REF_URL) + the shards (MUMMU_QWEN4EXP_DIR)"]
fn record_the_qwen4exp_fixture_from_a_running_llama_server() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let Some(url) = std::env::var_os("MUMMU_QWEN4EXP_REF_URL") else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_REF_URL to a running qwen4exp llama-server");
        return;
    };
    let url = url.to_string_lossy().into_owned();
    let env_or =
        |var: &str| std::env::var(var).unwrap_or_else(|_| format!("unrecorded (set {var})"));

    let tok = tokenizer(&first);
    let server = LlamaServer::attach(&url).expect("reference server is healthy");
    let props = server.get_json("/props").expect("GET /props");
    let build_info = props["build_info"]
        .as_str()
        .expect("/props carries build_info")
        .to_string();
    let ftype = props["model_ftype"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    eprintln!("[record/qwen4exp] reference build {build_info}, ftype {ftype:?}");

    let mut legs = Vec::new();
    for (name, prompt) in LEGS {
        legs.push(record_leg(&server, &tok, name, prompt, MAX_TOKENS));
    }

    let fixture = Fixture {
        format: FORMAT,
        model: ModelInfo {
            first_shard: FIRST_SHARD.to_string(),
            ftype,
        },
        reference: ReferenceInfo {
            image: env_or("MUMMU_QWEN4EXP_REF_IMAGE"),
            build_info,
            server_args: env_or("MUMMU_QWEN4EXP_REF_ARGS"),
        },
        legs,
    };
    let mut json = serde_json::to_string_pretty(&fixture).expect("fixture serializes");
    json.push('\n');
    std::fs::write(FIXTURE_PATH, json).expect("write fixture");
    eprintln!("[record/qwen4exp] wrote {FIXTURE_PATH}");
}

/// The long-prompt reference (`LONG_FIXTURE_PATH`), recorded from the same
/// running server as the short legs: one leg named `long`, [`LONG_PROMPT`]
/// trimmed, [`LONG_MAX_TOKENS`] greedy tokens. Same variables as
/// [`record_the_qwen4exp_fixture_from_a_running_llama_server`]; run it with
/// `--ignored record_the_qwen4exp_long_prompt_reference`.
#[test]
#[ignore = "needs a running qwen4exp llama-server (MUMMU_QWEN4EXP_REF_URL) + the shards (MUMMU_QWEN4EXP_DIR)"]
fn record_the_qwen4exp_long_prompt_reference_from_a_running_llama_server() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let Some(url) = std::env::var_os("MUMMU_QWEN4EXP_REF_URL") else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_REF_URL to a running qwen4exp llama-server");
        return;
    };
    let env_or =
        |var: &str| std::env::var(var).unwrap_or_else(|_| format!("unrecorded (set {var})"));
    let tok = tokenizer(&first);
    let server = LlamaServer::attach(&url.to_string_lossy()).expect("reference server is healthy");
    let props = server.get_json("/props").expect("GET /props");
    let leg = record_leg(&server, &tok, "long", LONG_PROMPT.trim(), LONG_MAX_TOKENS);
    let fixture = Fixture {
        format: FORMAT,
        model: ModelInfo {
            first_shard: FIRST_SHARD.to_string(),
            ftype: props["model_ftype"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        },
        reference: ReferenceInfo {
            image: env_or("MUMMU_QWEN4EXP_REF_IMAGE"),
            build_info: props["build_info"]
                .as_str()
                .expect("/props carries build_info")
                .to_string(),
            server_args: env_or("MUMMU_QWEN4EXP_REF_ARGS"),
        },
        legs: vec![leg],
    };
    let mut json = serde_json::to_string_pretty(&fixture).expect("fixture serializes");
    json.push('\n');
    std::fs::write(LONG_FIXTURE_PATH, json).expect("write long fixture");
    eprintln!("[record/qwen4exp] wrote {LONG_FIXTURE_PATH}");
}

/// Ask the attached reference for one greedy leg (`max_tokens` generated,
/// [`N_PROBS`] logprobs per position) on mummu's own ids for `prompt`, and
/// return it checked for self-consistency.
fn record_leg(
    server: &LlamaServer,
    tok: &tokenizers::Tokenizer,
    name: &str,
    prompt: &str,
    max_tokens: usize,
) -> Leg {
    let (rendered, ids) = render_prompt_ids(tok, prompt);
    let request = serde_json::json!({
        "prompt": ids,
        "n_predict": max_tokens,
        "n_probs": N_PROBS,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": false,
        "return_tokens": true,
    });
    eprintln!(
        "[record/qwen4exp/{name}] {} prompt ids, requesting {max_tokens} tokens",
        ids.len()
    );
    let started = std::time::Instant::now();
    let v = server
        .raw_completion(&request)
        .expect("reference completion");
    eprintln!(
        "[record/qwen4exp/{name}] answered in {:.1} s",
        started.elapsed().as_secs_f64()
    );

    // The shared parser is the transport contract every live gate relies
    // on; running it over this response re-checks its field names and
    // the no-BOS-injection invariant against the build being recorded.
    let parsed = parse_completion(&v, ids.len()).expect("response parses like a live gate's");
    let greedy_ids: Vec<u32> = v["tokens"]
        .as_array()
        .expect("return_tokens yields a tokens array")
        .iter()
        .map(|t| u32::try_from(t.as_u64().expect("token id")).expect("token id fits u32"))
        .collect();
    assert_eq!(
        parsed.chosen, greedy_ids,
        "per-position sampled ids disagree with the returned tokens"
    );

    let positions = v["completion_probabilities"]
        .as_array()
        .expect("n_probs yields completion_probabilities");
    let steps: Vec<Step> = positions
        .iter()
        .map(|pos| Step {
            id: u32::try_from(pos["id"].as_u64().expect("position id"))
                .expect("position id fits u32"),
            token: token_text(pos),
            logprob: pos["logprob"].as_f64().expect("position logprob"),
            top: pos["top_logprobs"]
                .as_array()
                .expect("top_logprobs")
                .iter()
                .map(|e| TopEntry {
                    id: u32::try_from(e["id"].as_u64().expect("entry id"))
                        .expect("entry id fits u32"),
                    token: token_text(e),
                    logprob: e["logprob"].as_f64().expect("entry logprob"),
                })
                .collect(),
        })
        .collect();

    let greedy_text_mummu = tok.decode(&greedy_ids, true).expect("decode");
    let content = parsed.content;
    if greedy_text_mummu != content {
        eprintln!(
            "[record/qwen4exp/{name}] NOTE server content differs from mummu's decode \
             of the same ids:\n  server: {content:?}\n  mummu : {greedy_text_mummu:?}"
        );
    }
    eprintln!("[record/qwen4exp/{name}] greedy: {content:?}");
    eprintln!("[record/qwen4exp/{name}] timings: {}", v["timings"]);

    let leg = Leg {
        name: name.to_string(),
        prompt: prompt.to_string(),
        rendered,
        prompt_ids: ids,
        tokens_evaluated: usize::try_from(
            v["tokens_evaluated"].as_u64().expect("tokens_evaluated"),
        )
        .expect("tokens_evaluated fits usize"),
        request,
        steps,
        greedy_ids,
        content,
        greedy_text_mummu,
        stop_type: v["stop_type"].as_str().unwrap_or_default().to_string(),
        timings: v["timings"].clone(),
    };
    assert_leg_is_self_consistent(&leg, max_tokens);
    leg
}

/// Invariants a replayable leg must satisfy, checked when recording AND on
/// the committed file (so a hand edit or a partial re-record fails here, not
/// as a confusing parity divergence later).
fn assert_leg_is_self_consistent(leg: &Leg, max_tokens: usize) {
    let name = &leg.name;
    assert_eq!(
        leg.tokens_evaluated,
        leg.prompt_ids.len(),
        "[{name}] the server evaluated a different number of tokens than we sent"
    );
    assert!(!leg.steps.is_empty(), "[{name}] no generated positions");
    assert!(
        leg.steps.len() <= max_tokens,
        "[{name}] more positions than requested"
    );
    assert_eq!(
        leg.steps.len(),
        leg.greedy_ids.len(),
        "[{name}] one logprob step per generated id"
    );
    for (i, (step, &id)) in leg.steps.iter().zip(&leg.greedy_ids).enumerate() {
        assert_eq!(step.id, id, "[{name}] step {i} id vs greedy id");
        assert_eq!(
            step.top.len(),
            N_PROBS,
            "[{name}] step {i} has {} logprobs, want {N_PROBS}",
            step.top.len()
        );
        assert!(
            step.top.windows(2).all(|w| w[0].logprob >= w[1].logprob),
            "[{name}] step {i} top list is not best-first"
        );
        assert!(
            step.top
                .iter()
                .all(|e| e.logprob <= 0.0 && e.logprob.is_finite()),
            "[{name}] step {i} has a non-logprob value"
        );
        // A greedy reference that did not take its own top-1 is not a
        // greedy reference.
        assert_eq!(
            step.top[0].id, id,
            "[{name}] step {i} sampled {id} but its top-1 is {}",
            step.top[0].id
        );
    }
    assert!(
        leg.steps[0].top.len() >= TOP_K,
        "[{name}] first forward shorter than top-{TOP_K}"
    );
}

/// The committed fixture parses, has both legs in order, and every leg is
/// internally consistent. Runs in the default test pass: it needs no model.
#[test]
fn the_committed_qwen4exp_fixture_is_self_consistent() {
    let fixture = Fixture::load();
    let names: Vec<&str> = fixture.legs.iter().map(|l| l.name.as_str()).collect();
    let want: Vec<&str> = LEGS.iter().map(|&(n, _)| n).collect();
    assert_eq!(names, want, "legs recorded in LEGS order");
    assert_eq!(
        fixture.leg("primes").prompt,
        PROMPT,
        "leg 1 is the prompt every GGUF gate shares"
    );
    assert_eq!(fixture.model.first_shard, FIRST_SHARD);
    for leg in &fixture.legs {
        assert_leg_is_self_consistent(leg, MAX_TOKENS);
        assert!(
            leg.rendered.ends_with("<|im_start|>assistant\n"),
            "[{}] the rendering opens the assistant turn",
            leg.name
        );
        assert!(
            leg.prompt_ids.len() >= 8,
            "[{}] recorded prompt suspiciously short",
            leg.name
        );
    }
}

/// The replay verdict accepts the reference's OWN numbers: logits built from
/// the recorded top-10 logprobs plus the recorded greedy ids must pass
/// `compare_leg`. Proves the replay plumbing (id re-derivation, rank/id
/// alignment, text decode) before any model output is judged by it.
#[test]
#[ignore = "needs the Flash-Next shards for the tokenizer (MUMMU_QWEN4EXP_DIR)"]
fn the_reference_replayed_against_itself_passes_the_gate() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let tok = tokenizer(&first);
    let fixture = Fixture::load();
    for leg in &fixture.legs {
        // Unrecorded ids sit far below the recorded tail, so the softmax over
        // this row renormalizes the top-10 mass and nothing else.
        let vocab = tok.get_vocab_size(true);
        let mut logits = vec![-1.0e4_f32; vocab];
        for e in &leg.steps[0].top {
            logits[e.id as usize] = narrow(e.logprob);
        }
        compare_leg(leg, &logits, &leg.greedy_ids, &tok);
    }
}

/// The committed long-prompt reference parses and is internally consistent,
/// and its prompt is still [`LONG_PROMPT`]. Default test pass, no model.
#[test]
fn the_committed_long_prompt_reference_is_self_consistent() {
    let fixture = Fixture::load_from(LONG_FIXTURE_PATH);
    assert_eq!(fixture.legs.len(), 1, "one long leg");
    let leg = fixture.leg("long");
    assert_eq!(
        leg.prompt,
        LONG_PROMPT.trim(),
        "the recorded prompt is LONG_PROMPT"
    );
    assert_eq!(
        leg.greedy_ids.len(),
        LONG_MAX_TOKENS,
        "no early stop recorded"
    );
    assert!(
        leg.prompt_ids.len() > 512,
        "the long leg must cross several 64-token GDN chunks: {} ids",
        leg.prompt_ids.len()
    );
    assert_leg_is_self_consistent(leg, LONG_MAX_TOKENS);
    assert_eq!(fixture.model.first_shard, FIRST_SHARD);
}
