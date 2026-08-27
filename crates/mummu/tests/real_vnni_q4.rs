//! Real-weights gate for the packed-nibble VNNI host GEMV (SPEC 1): OLMoE
//! keep-quantized at **Q4** on the CPU backend — every expert projection
//! routes `qlinear → try_q4s_gemv → flex` and, with the registry enabled,
//! through the packed twin. The twin is its OWN quantization grid (groups
//! along K; a second rounding when built lazily from the i8 slab), so this
//! is a behavior gate, not a bit-parity gate: the model must stay coherent
//! and the two paths' timings are printed for the record. Ignored by
//! default; run with
//!
//! ```text
//! MUMMU_OLMOE_GGUF_PATH=path/to/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
//!   cargo test -p mummu --release --test real_vnni_q4 -- --ignored --nocapture
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
async fn olmoe_q4_stays_coherent_and_faster_on_the_vnni_twin() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_OLMOE_GGUF_PATH to an OLMoE GGUF file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);
    let raw =
        "<|endoftext|><|user|>\nWhat is 2 + 2? Answer in one short sentence.\n<|assistant|>\n";
    let prompt = tok
        .encode(raw, false)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();
    let device = mummu::backend::cpu_device();

    let q = olmoe::load_from_gguf_quantized(&path, &device, QuantPolicy::Q4).expect("q4 load");

    // Baseline first so the twin run includes its own lazy-repack warmup in
    // the printed wall time (honest ordering: cold cost lands on the new
    // path, not the incumbent).
    let mut results: Vec<(String, f64)> = Vec::new();
    for (label, twin) in [("i8-slab", false), ("vnni-twin", true)] {
        mummu::flex::registry::force_disable(!twin);
        let started = std::time::Instant::now();
        let ids = q
            .generate(
                &prompt,
                32,
                &mummu::decode::SamplerOptions::greedy(),
                &device,
                |_| std::ops::ControlFlow::Continue(()),
            )
            .await
            .expect("decode");
        let secs = started.elapsed().as_secs_f64();
        let text = tok.decode(&ids, true).expect("ids decode");
        eprintln!(
            "[real_vnni_q4/{label}] {} tokens in {secs:.1}s ({:.2} s/tok): {text:?}",
            ids.len(),
            secs / ids.len().max(1) as f64,
        );
        results.push((text, secs / ids.len().max(1) as f64));
    }
    mummu::flex::registry::force_disable(false);
    let (base_text, base_spt) = results.remove(0);
    let (twin_text, twin_spt) = results.remove(0);

    assert!(
        twin_text.contains('4'),
        "the vnni-twin decode must still answer correctly, got: {twin_text:?}"
    );
    assert!(
        base_text.contains('4'),
        "the baseline decode must answer correctly, got: {base_text:?}"
    );
    eprintln!(
        "[real_vnni_q4] twin/i8 speed ratio: {:.2}x (i8 {base_spt:.2} s/tok, twin {twin_spt:.2})",
        base_spt / twin_spt.max(1e-9),
    );
}
