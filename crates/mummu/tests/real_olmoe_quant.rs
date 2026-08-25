//! P9 MoE gate: OLMoE with **per-expert quantized** experts (routed compute)
//! must agree with the fused f32 model on the first token's argmax and
//! answer correctly under greedy decoding. CPU — where flex's dequantize
//! fallback executes the quantized matmuls. Ignored by default; run with
//!
//! ```text
//! MUMMU_OLMOE_GGUF_PATH=path/to/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
//!   cargo test -p mummu --release --test real_olmoe_quant -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::models::CausalLm;
use mummu::models::olmoe;
use mummu::quant::QuantPolicy;

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MUMMU_OLMOE_GGUF_PATH")?);
    p.is_file().then_some(p)
}

#[tokio::test]
#[ignore = "needs the local OLMoE GGUF (MUMMU_OLMOE_GGUF_PATH)"]
async fn olmoe_per_expert_q8_agrees_with_f32_and_answers() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_OLMOE_GGUF_PATH to an OLMoE GGUF file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);
    // The Tulu template this family speaks (the real_olmoe.rs pattern).
    let raw =
        "<|endoftext|><|user|>\nWhat is 2 + 2? Answer in one short sentence.\n<|assistant|>\n";
    let prompt = tok
        .encode(raw, false)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();
    let device = mummu::backend::cpu_device();

    let f32_top = {
        let loaded = olmoe::load_from_gguf(&path, &device).expect("f32 load");
        loaded
            .first_token(&prompt, 1, &device)
            .await
            .expect("probe")[0]
    };

    let q = olmoe::load_from_gguf_quantized(&path, &device, QuantPolicy::Q8).expect("q8 load");
    let q_top = q.first_token(&prompt, 1, &device).await.expect("probe")[0];
    assert_eq!(
        q_top, f32_top,
        "per-expert Q8 first-token argmax diverges from f32"
    );

    let ids = q
        .generate(
            &prompt,
            48,
            &mummu::decode::SamplerOptions::greedy(),
            &device,
            |_| std::ops::ControlFlow::Continue(()),
        )
        .await
        .expect("q8 decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_olmoe_quant/q8] {} tokens: {text:?}", ids.len());
    assert!(
        text.contains('4'),
        "expected the answer to mention 4, got: {text:?}"
    );
}
