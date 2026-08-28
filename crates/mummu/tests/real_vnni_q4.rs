//! Real-weights gate for the packed-nibble VNNI host GEMV (SPEC 1): a real
//! checkpoint tensor (llama.cpp-quantized Qwen2.5-1.5B's widest FFN
//! projection), dequantized and re-quantized onto the device grid exactly
//! the way a keep-quantized load does, then driven through `try_q4s_gemv`
//! on the flex backend with the packed twin off and on.
//!
//! What this pins that the synthetic kernel tests cannot: real weight
//! STATISTICS (outlier structure, per-block dynamic range) through the
//! whole dispatch stack — twin build, registry hit, error bounds — plus
//! the timing of both paths on a production-shaped tensor. The full-model
//! behavior gate for the twin is the 27B through mummu-serve (qwen35's
//! `qlinear` is the production caller); this is the largest real-weights
//! gate that runs from the local fixture cache. Ignored by default; run
//! with
//!
//! ```text
//! MUMMU_GGUF_PATH=path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test -p mummu --release --test real_vnni_q4 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use burn::tensor::{Tensor, TensorData};

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MUMMU_GGUF_PATH")?);
    p.is_file().then_some(p)
}

#[test]
#[ignore = "needs a local GGUF model file (MUMMU_GGUF_PATH)"]
fn real_weight_tensor_through_the_twin_stays_inside_its_bound_and_is_faster() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_GGUF_PATH to a local .gguf model file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    // The widest projection of the model — the shape class that dominates a
    // host layer's decode cost.
    let name = "blk.0.ffn_gate.weight";
    let info = f.tensor(name).expect("gate tensor exists");
    // ggml dims are fastest-varying first: [in, out] reversed.
    let (out_w, in_w) = (info.dims[1] as usize, info.dims[0] as usize);
    let vals = f.read_tensor_f32(name).expect("dequantizes"); // row-major [out, in]
    drop(f);

    // The keep-quantized load path: transpose to Linear's [in, out] and
    // quantize onto the device grid (Q4S, blocks along out).
    let mut linear = vec![0.0f32; vals.len()];
    for o in 0..out_w {
        for i in 0..in_w {
            linear[i * out_w + o] = vals[o * in_w + i];
        }
    }
    let device = mummu::backend::cpu_device();
    let w = Tensor::<2>::from_data(TensorData::new(linear, [in_w, out_w]), &device);
    let wq = mummu::quant::quantize_weight(mummu::quant::QuantPolicy::Q4, w);

    // A realistic activation: unit-scale noise with a mild heavy tail, the
    // regime the quality gate must NOT trip on.
    let x_host: Vec<f32> = (0..in_w)
        .map(|i| {
            let t = (i as f32) * 0.7311;
            t.sin() + 0.1 * (t * 13.7).sin().powi(3)
        })
        .collect();
    let x = Tensor::<2>::from_data(TensorData::new(x_host.clone(), [1, in_w]), &device);

    let time_path = |label: &str, twin: bool| -> (Vec<f32>, f64) {
        mummu::flex::registry::force_disable(!twin);
        let y = mummu::nn::try_q4s_gemv(&x, &wq).expect("packed path engages");
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            let t0 = std::time::Instant::now();
            let _ = mummu::nn::try_q4s_gemv(&x, &wq).expect("packed path engages");
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        let v = y.into_data().try_to_vec::<f32>().expect("readback");
        eprintln!("[real_vnni_q4/{label}] [{in_w} x {out_w}] {best:.3} ms/call");
        (v, best)
    };

    let (base, base_ms) = time_path("i8-slab", false);
    let (twin, twin_ms) = time_path("vnni-twin", true);
    mummu::flex::registry::force_disable(false);

    // The twin's answer differs from the device grid's by exactly two
    // bounded terms, and both are computable here — so the budget is the
    // real one, not a guess:
    //  (1) requantization along K: per output n, Σ_k |W_dev − W_twin|·|x_k|
    //      with both grids rebuilt from the same bytes the paths read;
    //  (2) activation quantization: the per-row ε bound from the kernel.
    let dev_grid = wq
        .dequantize()
        .into_data()
        .try_to_vec::<f32>()
        .expect("device grid");
    let (qi8, qscales) = mummu::pack::quantize_blocks(&dev_grid, out_w, mummu::pack::Precision::Q4);
    let twin_pack = mummu::flex::kernels::PackedQ4::from_q4s_slab(&qi8, &qscales, in_w, out_w);
    let twin_grid = twin_pack.dequantize();
    let acts = mummu::flex::kernels::Q8Acts::quantize(&x_host);
    let act_bound = mummu::flex::kernels::act_error_bound(&twin_pack, &acts);
    let scale = base.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    let mut worst_rel_to_budget = 0.0f32;
    let mut worst_abs = 0.0f32;
    for n in 0..out_w {
        let mut req = 0.0f32;
        for k in 0..in_w {
            req += (dev_grid[k * out_w + n] - twin_grid[k * out_w + n]).abs() * x_host[k].abs();
        }
        let budget = req + act_bound[n] + 1e-5 * scale;
        let err = (base[n] - twin[n]).abs();
        worst_abs = worst_abs.max(err);
        worst_rel_to_budget = worst_rel_to_budget.max(err / budget);
    }
    eprintln!(
        "[real_vnni_q4] max |twin - i8| = {worst_abs:.5} against output scale {scale:.3} \
         ({:.2}% relative); worst error/budget = {worst_rel_to_budget:.3}; \
         twin {twin_ms:.3} ms vs i8 {base_ms:.3} ms ({:.2}x)",
        100.0 * worst_abs / scale,
        base_ms / twin_ms,
    );
    assert!(
        worst_rel_to_budget <= 1.0,
        "twin diverges beyond its mathematical requant+activation budget \
         ({worst_rel_to_budget:.3}x) — the kernel is reading the wrong grid"
    );
    assert!(
        twin_ms < base_ms,
        "the twin must beat the i8 slab on a production-shaped real tensor \
         (twin {twin_ms:.3} ms vs i8 {base_ms:.3} ms)"
    );
}
