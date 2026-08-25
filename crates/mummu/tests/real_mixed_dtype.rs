//! In-process mixed precision (the P6 dtype-policy hazard, pinned): run an
//! f16 (`GpuF16`) model and an f32 (`Gpu`) model in the SAME process, in the
//! order that used to crash — f16 first, so Burn's per-DEVICE default-dtype
//! policy (a first-touch-locked global registry both aliases share) locks to
//! F16 before the f32 model ever runs.
//!
//! With every runtime tensor-creation site dtype-pinned to the backend TYPE
//! (`backend::{float_dtype, int_dtype}`), the ambient policy no longer
//! matters: both models must load, sanity-check, produce logits in THEIR
//! OWN dtype, and decode coherently. Before the pinning, the f32 leg died
//! `TypeMismatch("expected F16, got F32")` (observed 2026-07-23).
//!
//! This binary deliberately breaks the one-alias-per-process test convention
//! — that convention is the *workaround* this test proves unnecessary for
//! pinned code paths. Keep it its own binary so its policy pollution can't
//! leak into other suites. Ignored by default; run with:
//!
//! ```text
//! MUMMU_QWEN3_DIR=path/to/qwen3-0.6b \
//!   cargo test -p mummu --release --test real_mixed_dtype -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use burn::tensor::DType;
use mummu::backend::inventory;
use mummu::models::{CausalLm, qwen3};
use tokenizers::Tokenizer;

fn qwen3_dir() -> Option<PathBuf> {
    std::env::var_os("MUMMU_QWEN3_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("model.safetensors").is_file())
}

#[tokio::test]
#[ignore = "needs the Qwen3 safetensors dir (MUMMU_QWEN3_DIR) + a SHADER_F16 GPU"]
async fn f32_model_survives_an_f16_model_having_locked_the_device_policy() {
    let dir = qwen3_dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 safetensors dir");
    assert!(
        inventory().any_shader_f16(),
        "no adapter advertises SHADER_F16 — cannot validate the mixed-dtype path here"
    );

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let raw = mummu::chat::ChatMl::qwen3().render(&[
        mummu::chat::Turn::system("You are a concise assistant. Do not think, answer directly."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    let prompt = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    // Leg 1 — f16 FIRST: its first tensor touch locks the shared device
    // policy to F16 (the historically poisonous order).
    let f16_top = {
        let device =
            mummu::backend::gpu_device_f16().expect("f16 device settings lock once per process");
        let model = qwen3::load_from_dir(&dir, &device).expect("f16 weights load checked");
        let mut cache = model.new_cache();
        let logits = model.forward(&prompt, 0, &mut cache, &device);
        assert_eq!(
            logits.dtype(),
            DType::F16,
            "the f16 model's logits must be f16"
        );
        let smoke = model
            .sanity_check(&prompt, model.config.vocab_size, &device)
            .await
            .expect("f16 forward is finite and non-degenerate");
        eprintln!(
            "[mixed_dtype/f16] sanity smoke: top_id {} · spread {:.3}",
            smoke.top_id, smoke.spread
        );
        smoke.top_id
        // model + VRAM dropped here; the policy lock persists — that is the point.
    };

    // Leg 2 — f32 in the SAME process, with the policy locked to F16: every
    // pinned creation site must still produce f32 tensors, and the strict
    // f32 readback that used to die TypeMismatch must succeed.
    let device = mummu::backend::gpu_device();
    let model = qwen3::load_from_dir(&dir, &device).expect("f32 weights load checked");
    let mut cache = model.new_cache();
    let logits = model.forward(&prompt, 0, &mut cache, &device);
    assert_eq!(
        logits.dtype(),
        DType::F32,
        "the f32 model's logits must be f32 even with the device policy locked to f16"
    );
    let strict: Vec<f32> = logits
        .into_data()
        .to_vec()
        .expect("strict f32 readback (the 2026-07-23 TypeMismatch site)");
    assert_eq!(strict.len(), model.config.vocab_size, "full logit row");
    let smoke = model
        .sanity_check(&prompt, model.config.vocab_size, &device)
        .await
        .expect("f32 forward is finite and non-degenerate");
    eprintln!(
        "[mixed_dtype/f32] sanity smoke: top_id {} · spread {:.3}",
        smoke.top_id, smoke.spread
    );
    // Same weights, same prompt: the two precisions should agree on the
    // greedy first token (they do everywhere else in the zoo's f16 gates).
    assert_eq!(
        smoke.top_id, f16_top,
        "f32 and f16 disagree on the first greedy token — a dtype leak, not rounding"
    );

    // And the f32 model still decodes coherently end-to-end. Generous budget:
    // Qwen3-0.6B spends tokens on a <think> block before the answer.
    let ids = model
        .greedy_generate(&prompt, 128, &device)
        .await
        .expect("f32 greedy decode with a poisoned device policy");
    assert!(!ids.is_empty(), "decode produced no tokens before EOS");
    let text = tok.decode(&ids, true).expect("decode");
    eprintln!("[mixed_dtype/f32] decoded: {text:?}");
    assert!(
        text.contains('4'),
        "expected the f32 answer to mention 4, got: {text:?}"
    );
}
