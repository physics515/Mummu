//! Real-model smoke tests: load an actual Qwen2 checkpoint and decode on the
//! machine's default device. Ignored by default (multi-GB weights); run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-x.xb cargo test -p mummu --test real_inference -- --ignored --nocapture
//! ```
//!
//! where the dir holds `config.json`, `tokenizer.json`, `model.safetensors`.

use std::path::PathBuf;

use mummu::backend::{Cpu, Gpu, use_gpu};
use mummu::models::qwen2;
use tokenizers::Tokenizer;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
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
        let loaded = qwen2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");
        (
            loaded
                .greedy_generate(&prompt, 32, &device)
                .expect("decode"),
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
    let loaded = qwen2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");
    let top5 = loaded.first_token(&prompt, 5, &device).expect("probe");
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
    let loaded = qwen2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");

    let opts = mummu::decode::SamplerOptions {
        temperature: 0.7,
        top_p: 0.9,
        seed: 42,
        ..mummu::decode::SamplerOptions::default()
    };

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
    assert_eq!(cancelled.len(), 8, "cancel after 8 streamed tokens");
    assert_eq!(cancelled, streamed, "returned ids == streamed ids");

    // Leg 2: the same seed replays the same sampled prefix.
    let replay = loaded
        .generate(&prompt, 8, &opts, &device, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("replay decode");
    assert_eq!(
        replay, cancelled,
        "same (prompt, options, seed) must resample the same tokens"
    );

    let text = decode(&dir, &cancelled);
    eprintln!("[real_inference/sampled] 8 tokens @ T=0.7 p=0.9 seed=42: {text:?}");
    assert!(!text.trim().is_empty(), "sampled text should be non-empty");
}

/// The process-lifetime slot the cached-decode test drives (a static, as
/// consumers would hold it).
static QWEN2_SLOT: mummu::cache::ModelSlot<qwen2::LoadedQwen2<Gpu>> =
    mummu::cache::ModelSlot::new();

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR)"]
fn qwen2_model_slot_loads_once_across_calls() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let device = burn::tensor::Device::<Gpu>::default();
    let prompt = encode(
        &dir,
        "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n",
    );

    let mut loads = 0;
    for _ in 0..2 {
        let ids = QWEN2_SLOT
            .with(
                &dir,
                |d| {
                    loads += 1;
                    qwen2::load_from_dir::<Gpu>(d, &device)
                },
                |m| m.greedy_generate(&prompt, 4, &device).expect("decode"),
            )
            .expect("slot load");
        assert!(!ids.is_empty(), "cached model must still decode");
    }
    assert_eq!(loads, 1, "the multi-GB load must happen exactly once");
    assert_eq!(QWEN2_SLOT.loaded_key().as_deref(), Some(dir.as_path()));
}
