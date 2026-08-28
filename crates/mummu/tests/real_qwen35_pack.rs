//! P9 stage-3 gate: the `.mummu` pack round-trips a real Qwen3.5 GGUF.
//! Import once at every level, then prove (on CPU, the backend every pack
//! precision can execute on):
//!
//! - pack **F32** reproduces the GGUF f32 model's logits (a copy, not a
//!   re-derivation);
//! - pack **Q8** / **Q4** agree with the streaming re-quantizing loader on
//!   the first token (same quantizer, stored once vs. derived each load).
//!
//! Ignored by default (multi-GB weights); run with
//!
//! ```text
//! MUMMU_QWEN35_GGUF=path/to/Qwen3.5-2B-BF16.gguf \
//!   cargo test -p mummu --release --test real_qwen35_pack -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use burn::tensor::Tensor;
use mummu::models::CausalLm;
use mummu::models::qwen35;
use mummu::pack::{Pack, Precision};
use mummu::quant::QuantPolicy;

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MUMMU_QWEN35_GGUF")?);
    p.is_file().then_some(p)
}

fn max_abs_diff(a: &Tensor<2>, b: &Tensor<2>) -> f32 {
    a.clone().sub(b.clone()).abs().max().into_scalar()
}

fn argmax(t: &Tensor<2>) -> u32 {
    let v = t
        .clone()
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |m, (i, &x)| {
            if x > m.1 { (i, x) } else { m }
        })
        .0 as u32
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN35_GGUF)"]
fn qwen35_pack_round_trips_every_level() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_QWEN35_GGUF to a qwen35 GGUF file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    let cfg = qwen35::Qwen35Config::from_gguf(&f).expect("config");
    let trunk = cfg.num_layers;
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

    // Import beside the fixture (reused across runs when already complete).
    let pack_dir = path.parent().unwrap().join("pack-gate");
    if !Pack::is_pack(&pack_dir) {
        let _ = std::fs::remove_dir_all(&pack_dir);
        let t = std::time::Instant::now();
        let manifest = mummu::pack::import_gguf(
            &path,
            &pack_dir,
            &Precision::ALL,
            &|info| qwen35::pack_actions(info, trunk),
            |i, n, name| {
                if i % 50 == 0 {
                    eprintln!("[pack-gate] import {i}/{n} {name}");
                }
            },
        )
        .expect("pack import");
        eprintln!(
            "[pack-gate] imported {} tensors in {:.1}s",
            manifest.tensors.len(),
            t.elapsed().as_secs_f32()
        );
    }
    let pack = Pack::open(&pack_dir).expect("pack opens");
    let sizes: Vec<(Precision, u64)> = Precision::ALL
        .iter()
        .map(|&p| {
            (
                p,
                std::fs::metadata(pack_dir.join(p.blob_name())).map_or(0, |m| m.len()),
            )
        })
        .collect();
    eprintln!("[pack-gate] blobs: {sizes:?}");
    assert_eq!(pack.manifest.precisions, Precision::ALL.to_vec());
    drop(pack);

    let logits_of = |m: &qwen35::LoadedQwen35| {
        let mut cache = m.new_cache();
        m.forward(&prompt, 0, &mut cache, &device)
    };

    // 1. F32: the pack is a copy of the source.
    let ref_logits = {
        let m = qwen35::load_from_gguf(&path, &device).expect("gguf f32 load");
        logits_of(&m)
    };
    let ref_top = argmax(&ref_logits);
    let f32_logits = {
        let m =
            qwen35::load_from_pack(&pack_dir, &device, &|_| Precision::F32).expect("pack f32 load");
        logits_of(&m)
    };
    let d = max_abs_diff(&ref_logits, &f32_logits);
    eprintln!("[pack-gate] F32 pack vs GGUF f32: max |Δlogit| = {d:.3e}, top {ref_top}");
    assert!(
        d <= 1e-3,
        "pack F32 must reproduce the GGUF f32 logits (Δ={d})"
    );

    // F16: every linear stored at half precision; must still pick the same token.
    let f16_logits = {
        let m =
            qwen35::load_from_pack(&pack_dir, &device, &|_| Precision::F16).expect("pack f16 load");
        logits_of(&m)
    };
    let d = max_abs_diff(&ref_logits, &f16_logits);
    eprintln!(
        "[pack-gate] F16 pack vs f32: max |Δlogit| = {d:.3e}, top {}",
        argmax(&f16_logits)
    );
    assert_eq!(
        argmax(&f16_logits),
        ref_top,
        "F16 pack changes the first token"
    );

    // 2. Q8 / Q4: stored quantization ≡ streaming re-quantization.
    for (precision, policy) in [
        (Precision::Q8, QuantPolicy::Q8),
        (Precision::Q4, QuantPolicy::Q4),
    ] {
        let stream = {
            let m = qwen35::load_from_gguf_quantized(&path, &device, policy)
                .expect("streaming quantized load");
            logits_of(&m)
        };
        let packed = {
            let m = qwen35::load_from_pack(&pack_dir, &device, &|_| precision)
                .expect("pack quantized load");
            logits_of(&m)
        };
        let d = max_abs_diff(&stream, &packed);
        eprintln!(
            "[pack-gate] {precision:?}: pack vs streaming max |Δlogit| = {d:.3e}; tops {} / {} (f32 {ref_top})",
            argmax(&packed),
            argmax(&stream)
        );
        assert_eq!(
            argmax(&packed),
            argmax(&stream),
            "{precision:?} pack diverges from the streaming loader on the first token"
        );
        assert!(
            d <= 5e-2,
            "{precision:?} pack vs streaming Δ={d} — same quantizer expected"
        );
    }
}
