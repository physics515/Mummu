//! P9 stage-3(c) gate on real weights: a dense Qwen3.5 pack with its FFNs
//! **partitioned** into neuron clusters must (1) stay the same model when
//! loaded densely (the permutation is invisible), (2) stay the same model
//! when half of every layer's clusters run *remotely* through the expert
//! pool at f32 (exact tiering), (3) keep the first token and answer when the
//! remote half runs at int8 (exact placement, quantized tier), and (4) report
//! an honest skip trade-off at a small tau.
//!
//! Ignored by default; needs the 2B fixture and its `pack-gate` import (the
//! `real_qwen35_pack` gate creates it; this gate partitions a **copy**):
//!
//! ```text
//! MUMMU_QWEN35_GGUF=path/to/Qwen3.5-2B-BF16.gguf \
//!   cargo test -p mummu --release --test real_qwen35_partition -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::tensor::Tensor;
use mummu::backend::Cpu;
use mummu::models::CausalLm;
use mummu::models::qwen35;
use mummu::nn::{DeviceExpert, ExpertExec, ExpertPool};
use mummu::pack::{Pack, Precision};
use mummu::partition::{FfnNames, partition_pack};
use mummu::tier::Tier;

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MUMMU_QWEN35_GGUF")?);
    p.is_file().then_some(p)
}

fn argmax(t: &Tensor<Cpu, 2>) -> u32 {
    let v = t.clone().into_data().convert::<f32>().to_vec::<f32>().unwrap();
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
        .0 as u32
}

fn max_abs(a: &Tensor<Cpu, 2>, b: &Tensor<Cpu, 2>) -> f32 {
    a.clone().sub(b.clone()).abs().max().into_scalar()
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dst.join(e.file_name())).unwrap();
    }
}

/// Remote half: one executor per cluster on the CPU at `precision`.
fn remote_pool(pack: &Pack, layers: usize, local: usize, precision: Precision, device: &burn::tensor::Device<Cpu>) -> Arc<ExpertPool> {
    let part = pack.manifest.ffn_partition.as_ref().unwrap();
    let rows: Vec<Vec<Arc<dyn ExpertExec>>> = (0..layers)
        .map(|l| {
            (local..part.layers[l].len())
                .map(|c| {
                    let w = qwen35::load_ffn_clusters::<Cpu>(pack, l, &[c], precision, device).expect("cluster");
                    Arc::new(DeviceExpert::<Cpu> {
                        weights: w,
                        device: *device,
                        tier: Tier { device: 0, precision },
                        bytes: 0,
                    }) as Arc<dyn ExpertExec>
                })
                .collect()
        })
        .collect();
    Arc::new(ExpertPool::new(rows))
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN35_GGUF) and the pack-gate import"]
fn qwen35_partitioned_ffn_is_exact_when_every_cluster_runs() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_QWEN35_GGUF to a qwen35 GGUF file");
    };
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    let cfg = qwen35::Qwen35Config::from_gguf(&f).expect("config");
    drop(f);
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(
        "What is 2+2? Answer in one short sentence.",
    )]);
    let prompt = tok.encode(rendered.as_str(), false).unwrap().get_ids().to_vec();
    let device = burn::tensor::Device::<Cpu>::default();

    let src = path.parent().unwrap().join("pack-gate");
    assert!(Pack::is_pack(&src), "run real_qwen35_pack first (creates pack-gate)");
    let dst = path.parent().unwrap().join("pack-gate-part");
    if !Pack::is_pack(&dst) || Pack::open(&dst).unwrap().manifest.ffn_partition.is_none() {
        let _ = std::fs::remove_dir_all(&dst);
        copy_dir(&src, &dst);
        let mut pack = Pack::open(&dst).unwrap();
        let names: Vec<FfnNames> = (0..cfg.num_layers)
            .map(|l| FfnNames {
                gate: format!("blk.{l}.ffn_gate.weight"),
                up: format!("blk.{l}.ffn_up.weight"),
                down: format!("blk.{l}.ffn_down.weight"),
            })
            .collect();
        let t = std::time::Instant::now();
        partition_pack(&mut pack, &names, mummu::partition::DEFAULT_CLUSTERS, |i, n| {
            if i % 8 == 0 {
                eprintln!("[partition-gate] layer {i}/{n}");
            }
        })
        .expect("partition");
        eprintln!("[partition-gate] partitioned in {:.0}s", t.elapsed().as_secs_f32());
    }
    let pack = Pack::open(&dst).unwrap();
    let part = pack.manifest.ffn_partition.as_ref().unwrap();
    let clusters = part.layers[0].len();
    eprintln!("[partition-gate] {} layers × {clusters} clusters of {}", part.layers.len(), part.layers[0][0].len);

    let logits_of = |m: &qwen35::LoadedQwen35<Cpu>| {
        let mut cache = m.new_cache();
        m.forward(&prompt, 0, &mut cache, &device)
    };

    // Reference: the unpartitioned pack, dense f32.
    let reference = {
        let m = qwen35::load_from_pack::<Cpu>(&src, &device, &|_| Precision::F32).unwrap();
        logits_of(&m)
    };
    let ref_top = argmax(&reference);

    // 1. Dense load of the partitioned pack: same model.
    let dense_part = {
        let m = qwen35::load_from_pack::<Cpu>(&dst, &device, &|_| Precision::F32).unwrap();
        logits_of(&m)
    };
    let d = max_abs(&reference, &dense_part);
    eprintln!("[partition-gate] dense(partitioned) vs dense(original): Δ = {d:.3e}");
    assert!(d <= 2e-3, "permutation must not change the model (Δ={d})");

    // 2. Half local / half remote at f32 through the pool: exact.
    let local = clusters / 2;
    let choose_local = |_l: usize| (0..local).collect::<Vec<_>>();
    let m = qwen35::load_from_pack_partitioned::<Cpu>(&dst, &device, &|_| Precision::F32, &choose_local)
        .unwrap()
        .with_ffn_pool(remote_pool(&pack, cfg.num_layers, local, Precision::F32, &device));
    let tiered = logits_of(&m);
    let d = max_abs(&reference, &tiered);
    eprintln!("[partition-gate] tiered f32 (local {local}/{clusters}) vs dense: Δ = {d:.3e}, tops {} / {ref_top}", argmax(&tiered));
    assert!(d <= 2e-3, "exact tiering must reproduce the dense model (Δ={d})");
    drop(m);

    // 3. Remote half at int8: placement-exact, quantization-noisy.
    let m = qwen35::load_from_pack_partitioned::<Cpu>(&dst, &device, &|_| Precision::F32, &choose_local)
        .unwrap()
        .with_ffn_pool(remote_pool(&pack, cfg.num_layers, local, Precision::Q8, &device));
    let mixed = logits_of(&m);
    let d = max_abs(&reference, &mixed);
    eprintln!("[partition-gate] tiered f32-local/int8-remote vs dense: Δ = {d:.3e}, tops {} / {ref_top}", argmax(&mixed));
    assert_eq!(argmax(&mixed), ref_top, "int8 remote tier changed the first token");
    let ids = m
        .generate(&prompt, 64, &mummu::decode::SamplerOptions::greedy(), &device, |_| std::ops::ControlFlow::Continue(()))
        .unwrap();
    let text = tok.decode(&ids, true).unwrap();
    eprintln!("[partition-gate] mixed answer ({} tokens): {text:?}", ids.len());
    assert!(text.contains('4'), "expected the answer to mention 4: {text:?}");

    // 4. Skipping at a small tau: report the trade honestly (no hard bound —
    // the calibrate tool stores the measured table the planner reads).
    let m = m.with_ffn_skip(0.02);
    let skipped = logits_of(&m);
    let d = max_abs(&reference, &skipped);
    eprintln!("[partition-gate] skip tau=0.02: Δ = {d:.3e}, top {} (ref {ref_top})", argmax(&skipped));
}
