//! Moving layers of a resident model is exactly a load of the target
//! placement: `qwen35::relocate_layers` re-reads each moved layer from the
//! pack onto its new device at its new precision, and the logits must then
//! match a model loaded that way from scratch — both after moving layers
//! onto the second device at a different precision, and after moving them
//! back. This is the correctness gate for live placement in mummu-serve.
//!
//! Ignored by default (multi-GB weights); run with
//!
//! ```text
//! MUMMU_QWEN35_GGUF=path/to/Qwen3.5-2B-BF16.gguf \
//!   cargo test -p mummu --release --test real_qwen35_relocate -- --ignored --nocapture
//! ```
//!
//! With `--features cuda` and `MUMMU_RELOCATE_DEVICE=cuda` the second device
//! is the card (four 2B layers, a few hundred MB); otherwise both are the
//! host and the test covers the precision change and the replacement.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::path::PathBuf;

use burn::tensor::Tensor;
use mummu::models::CausalLm;
use mummu::models::qwen35;
use mummu::pack::{Pack, Precision, Role, TensorEntry};

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MUMMU_QWEN35_GGUF")?);
    p.is_file().then_some(p)
}

fn second_device() -> burn::tensor::Device {
    #[cfg(feature = "cuda")]
    if std::env::var("MUMMU_RELOCATE_DEVICE").is_ok_and(|v| v.eq_ignore_ascii_case("cuda")) {
        return mummu::backend::cuda_device();
    }
    mummu::backend::cpu_device()
}

fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

fn choosable(e: &TensorEntry) -> bool {
    matches!(e.role, Role::Linear | Role::Expert { .. })
        && e.precisions.contains_key(&Precision::Q4)
        && e.precisions.contains_key(&Precision::Q8)
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN35_GGUF)"]
fn relocating_layers_equals_loading_the_target_placement() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_QWEN35_GGUF to a qwen35 GGUF file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let cfg = qwen35::Qwen35Config::from_gguf(&f).expect("config");
    let trunk = cfg.num_layers;
    drop(f);

    // Shared with `real_qwen35_pack`: import once, reuse after.
    let pack_dir = path.parent().unwrap().join("pack-gate");
    if !Pack::is_pack(&pack_dir) {
        let _ = std::fs::remove_dir_all(&pack_dir);
        mummu::pack::import_gguf(
            &path,
            &pack_dir,
            &Precision::ALL,
            &|info| qwen35::pack_actions(info, trunk),
            |_, _, _| {},
        )
        .expect("pack import");
    }

    let host = mummu::backend::cpu_device();
    let second = second_device();
    let moved: Vec<usize> = (0..4).collect();

    // A: every layer on the host, linears at Q8.
    // B: layers 0..4 on the second device at Q4 (the card's level in
    // production), the rest as A.
    let prec_a = |e: &TensorEntry| {
        if choosable(e) {
            Precision::Q8
        } else {
            *e.precisions.keys().max().unwrap()
        }
    };
    let prec_b = |e: &TensorEntry| {
        if choosable(e) && layer_of(&e.name).is_some_and(|l| l < 4) {
            Precision::Q4
        } else {
            prec_a(e)
        }
    };
    let dev_a = |_l: usize| host.clone();
    let dev_b = |l: usize| if l < 4 { second.clone() } else { host.clone() };

    let prompt: Vec<u32> = vec![9707, 11, 1246, 525, 498, 30];
    let logits_of = |m: &qwen35::LoadedQwen35| -> Tensor<2> {
        let mut cache = m.new_cache();
        m.forward(&prompt, 0, &mut cache, &host).to_device(&host)
    };
    let diff = |a: &Tensor<2>, b: &Tensor<2>| -> f32 {
        a.clone().sub(b.clone()).abs().max().into_scalar()
    };

    let fresh_a = {
        let m = qwen35::load_from_pack_layered(&pack_dir, &dev_a, &host, &host, &prec_a)
            .expect("load A");
        logits_of(&m)
    };
    let fresh_b = {
        let m = qwen35::load_from_pack_layered(&pack_dir, &dev_b, &host, &host, &prec_b)
            .expect("load B");
        logits_of(&m)
    };
    // The two placements really differ (Q4 vs Q8 on four layers), so the
    // equality checks below cannot pass by accident.
    assert!(diff(&fresh_a, &fresh_b) > 0.0, "A and B must differ");

    let mut m = qwen35::load_from_pack_layered(&pack_dir, &dev_a, &host, &host, &prec_a)
        .expect("load A for moving");
    let read =
        qwen35::relocate_layers(&mut m, &pack_dir, &moved, &second, &prec_b).expect("move A -> B");
    assert!(read > 0);
    for l in 0..trunk {
        let want = if l < 4 { &second } else { &host };
        assert_eq!(&qwen35::layer_device(&m, l).unwrap(), want, "layer {l}");
    }
    let d = diff(&logits_of(&m), &fresh_b);
    eprintln!("[relocate] A->B vs fresh B: max |Δlogit| = {d:.3e}");
    assert!(
        d <= 1e-4,
        "moved model must equal a fresh load of B (Δ={d})"
    );

    qwen35::relocate_layers(&mut m, &pack_dir, &moved, &host, &prec_a).expect("move B -> A");
    let d = diff(&logits_of(&m), &fresh_a);
    eprintln!("[relocate] B->A vs fresh A: max |Δlogit| = {d:.3e}");
    assert!(d <= 1e-4, "moved back must equal a fresh load of A (Δ={d})");

    // A layer index past the trunk is refused, not silently ignored.
    assert!(qwen35::relocate_layers(&mut m, &pack_dir, &[trunk], &host, &prec_a).is_err());
}
