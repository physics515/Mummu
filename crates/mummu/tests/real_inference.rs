//! Real-model smoke tests: load an actual Qwen2 checkpoint and decode on the
//! machine's default device. Ignored by default (multi-GB weights); run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-x.xb cargo test -p mummu --test real_inference -- --ignored --nocapture
//! ```
//!
//! where the dir holds `config.json`, `tokenizer.json`, `model.safetensors`.
//!
//! All GPU tests share one [`mummu::cache::ModelSlot`] static — the suite pays
//! the multi-GB load once, and the slot's mutex serializes GPU access, so the
//! whole file stays within a single model's VRAM even with parallel test
//! threads (two concurrent 6 GB loads would blow the 16 GB reference card).

use std::path::PathBuf;

use mummu::backend::{Cpu, Gpu, use_gpu};
use mummu::models::CausalLm;
use mummu::models::qwen2::{self, LoadedQwen2};
use tokenizers::Tokenizer;

/// One model for the whole suite (see the module docs).
static QWEN2_SLOT: mummu::cache::ModelSlot<LoadedQwen2<Gpu>> = mummu::cache::ModelSlot::new();

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

/// Run `f` with the shared GPU model (loading it on first use).
fn with_gpu_model<R>(dir: &std::path::Path, f: impl FnOnce(&LoadedQwen2<Gpu>) -> R) -> R {
    let device = burn::tensor::Device::<Gpu>::default();
    QWEN2_SLOT
        .with(dir, |d| qwen2::load_from_dir::<Gpu>(d, &device), f)
        .expect("weights load checked")
}

/// ChatML prompt for the Qwen2.5 instruct checkpoints.
fn chatml(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn encode(dir: &std::path::Path, text: &str) -> Vec<u32> {
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    tok.encode(text, true)
        .expect("prompt encodes")
        .get_ids()
        .to_vec()
}

fn decode(dir: &std::path::Path, ids: &[u32]) -> String {
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    tok.decode(ids, true).expect("ids decode")
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR)"]
fn qwen2_greedy_decodes_coherent_text_on_default_device() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let prompt = encode(
        &dir,
        &chatml(
            "You are a concise assistant.",
            "What is 2+2? Answer in one short sentence.",
        ),
    );

    let (ids, label) = if use_gpu() {
        let device = burn::tensor::Device::<Gpu>::default();
        (
            with_gpu_model(&dir, |m| {
                m.greedy_generate(&prompt, 32, &device).expect("decode")
            }),
            "GPU",
        )
    } else {
        let device = burn::tensor::Device::<Cpu>::default();
        let loaded = qwen2::load_from_dir::<Cpu>(&dir, &device).expect("weights load checked");
        (
            loaded
                .greedy_generate(&prompt, 32, &device)
                .expect("decode"),
            "CPU",
        )
    };

    assert!(
        !ids.is_empty(),
        "greedy decode produced no tokens before EOS"
    );
    let text = decode(&dir, &ids);
    eprintln!("[real_inference/{label}] {} tokens: {text:?}", ids.len());
    // Weakest useful coherence check that stays deterministic under greedy:
    // the answer to 2+2 must mention 4.
    assert!(
        text.contains('4'),
        "expected the answer to mention 4, got: {text:?}"
    );
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR)"]
fn qwen2_first_token_probe_reports_top5() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let prompt = encode(&dir, &chatml("You are a helpful assistant.", "Say hello."));
    let device = burn::tensor::Device::<Gpu>::default();
    let top5 = with_gpu_model(&dir, |m| m.first_token(&prompt, 5, &device).expect("probe"));
    assert_eq!(top5.len(), 5);
    eprintln!(
        "[real_inference] top-5 next-token ids: {top5:?} → {:?}",
        decode(&dir, &top5[..1])
    );
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR)"]
fn qwen2_sampled_streaming_is_seeded_deterministic_and_cancellable() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let prompt = encode(
        &dir,
        &chatml("You are a poet.", "Write one line about the sea."),
    );
    let device = burn::tensor::Device::<Gpu>::default();

    let opts = mummu::decode::SamplerOptions {
        temperature: 0.7,
        top_p: 0.9,
        seed: 42,
        ..mummu::decode::SamplerOptions::default()
    };

    let (cancelled, streamed, replay) = with_gpu_model(&dir, |loaded| {
        // Leg 1: streaming + cooperative cancellation after 8 tokens.
        let mut streamed = Vec::new();
        let cancelled = loaded
            .generate(&prompt, 64, &opts, &device, |id| {
                streamed.push(id);
                if streamed.len() == 8 {
                    std::ops::ControlFlow::Break(())
                } else {
                    std::ops::ControlFlow::Continue(())
                }
            })
            .expect("sampled decode");

        // Leg 2: the same seed replays the same sampled prefix.
        let replay = loaded
            .generate(&prompt, 8, &opts, &device, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .expect("replay decode");
        (cancelled, streamed, replay)
    });

    assert_eq!(cancelled.len(), 8, "cancel after 8 streamed tokens");
    assert_eq!(cancelled, streamed, "returned ids == streamed ids");
    assert_eq!(
        replay, cancelled,
        "same (prompt, options, seed) must resample the same tokens"
    );

    let text = decode(&dir, &cancelled);
    eprintln!("[real_inference/sampled] 8 tokens @ T=0.7 p=0.9 seed=42: {text:?}");
    assert!(!text.trim().is_empty(), "sampled text should be non-empty");
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR)"]
fn qwen2_model_slot_reuses_the_loaded_model() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let device = burn::tensor::Device::<Gpu>::default();
    let prompt = encode(
        &dir,
        "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n",
    );

    // Another test may already have populated the shared slot, so the
    // invariant is "no reload between consecutive same-key calls", observed
    // via the load closure never firing once the slot is warm.
    let mut loads_after_warm = 0;
    for round in 0..2 {
        let ids = QWEN2_SLOT
            .with(
                &dir,
                |d| {
                    if round > 0 {
                        loads_after_warm += 1;
                    }
                    qwen2::load_from_dir::<Gpu>(d, &device)
                },
                |m| m.greedy_generate(&prompt, 4, &device).expect("decode"),
            )
            .expect("slot load");
        assert!(!ids.is_empty(), "cached model must still decode");
    }
    assert_eq!(loads_after_warm, 0, "a warm slot must never reload");
    assert_eq!(QWEN2_SLOT.loaded_key().as_deref(), Some(dir.as_path()));
}
