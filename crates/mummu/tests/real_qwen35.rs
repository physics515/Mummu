//! Real-model smoke for the qwen35 hybrid: load an actual Qwen3.5 GGUF and
//! greedy-decode on the machine's default device. Ignored by default
//! (multi-GB weights); run with
//!
//! ```text
//! MUMMU_QWEN35_GGUF=path/to/Qwen3.5-2B-BF16.gguf \
//!   cargo test -p mummu --release --test real_qwen35 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::use_gpu;
use mummu::models::CausalLm;
use mummu::models::qwen35;

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MUMMU_QWEN35_GGUF")?);
    p.is_file().then_some(p)
}

/// P9 quality gate: the keep-quantized (Q8, block-32) model must agree with
/// the f32 model on the first token's argmax and still answer correctly
/// under greedy decoding. Runs on CPU — the backend where the quantized
/// path's dequantize fallback executes — so this also proves the fallback.
#[tokio::test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN35_GGUF)"]
async fn qwen35_q8_agrees_with_f32_and_stays_correct() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_QWEN35_GGUF to a qwen35 GGUF file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(
        "What is 2+2? Answer in one short sentence.",
    )]);
    let prompt = tok
        .encode(rendered.as_str(), false)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();
    let device = mummu::backend::cpu_device();

    let f32_top = {
        let loaded = qwen35::load_from_gguf(&path, &device).expect("f32 load");
        loaded
            .first_token(&prompt, 1, &device)
            .await
            .expect("probe")[0]
    };

    let q8 = qwen35::load_from_gguf_quantized(&path, &device, mummu::quant::QuantPolicy::Q8)
        .expect("q8 load");
    let q8_top = q8.first_token(&prompt, 1, &device).await.expect("probe")[0];
    assert_eq!(q8_top, f32_top, "Q8's first-token argmax diverges from f32");

    // Q8's slightly shifted logits legitimately pick a longer think stream
    // than f32's before answering — give the answer room to land.
    let ids = q8
        .generate(
            &prompt,
            192,
            &mummu::decode::SamplerOptions::greedy(),
            &device,
            |_| std::ops::ControlFlow::Continue(()),
        )
        .await
        .expect("q8 decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_qwen35/q8] {} tokens: {text:?}", ids.len());
    assert!(
        text.contains('4'),
        "expected the Q8 answer to mention 4, got: {text:?}"
    );
}

#[tokio::test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN35_GGUF)"]
async fn qwen35_greedy_decodes_coherent_text_on_default_device() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_QWEN35_GGUF to a qwen35 GGUF file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(
        "What is 2+2? Answer in one short sentence.",
    )]);
    let prompt = tok
        .encode(rendered.as_str(), false)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();

    let (ids, label) = if use_gpu() {
        let device = mummu::backend::gpu_device();
        let loaded = qwen35::load_from_gguf(&path, &device).expect("weights load checked");
        (
            loaded
                .generate(
                    &prompt,
                    96,
                    &mummu::decode::SamplerOptions::greedy(),
                    &device,
                    |_| std::ops::ControlFlow::Continue(()),
                )
                .await
                .expect("decode"),
            "GPU",
        )
    } else {
        let device = mummu::backend::cpu_device();
        let loaded = qwen35::load_from_gguf(&path, &device).expect("weights load checked");
        (
            loaded
                .generate(
                    &prompt,
                    96,
                    &mummu::decode::SamplerOptions::greedy(),
                    &device,
                    |_| std::ops::ControlFlow::Continue(()),
                )
                .await
                .expect("decode"),
            "CPU",
        )
    };

    assert!(
        !ids.is_empty(),
        "greedy decode produced no tokens before EOS"
    );
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_qwen35/{label}] {} tokens: {text:?}", ids.len());
    // The family thinks by default; the `4` must appear in the think stream
    // or the answer either way under greedy decoding.
    assert!(
        text.contains('4'),
        "expected the answer to mention 4, got: {text:?}"
    );
}
