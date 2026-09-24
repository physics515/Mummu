//! Calibrate a partitioned qwen35 pack (P9 stage 3c): measure per-cluster
//! activation energy (the **hotness prior** the tier planner places by) and
//! the **skip table** — how far the model strays from exact when clusters
//! below an energy threshold `tau` are skipped per token — and write both
//! into the pack manifest. The serve planner only ever picks a `tau` that
//! appears in this table, under a user-set tolerance.
//!
//! ```text
//! cargo run --release --example pack-calibrate -- <pack dir> [tau,tau,...]
//! ```
//!
//! Runs on the CPU with one local cluster per layer and every other cluster
//! behind the expert pool at f32, so the skip decision covers nearly the
//! whole FFN and the measurement is about skipping, not quantization.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use burn::tensor::{Device, Tensor};
use mummu::models::CausalLm;
use mummu::models::qwen35::{self, LoadedQwen35};
use mummu::nn::{DeviceExpert, ExpertExec, ExpertPool};
use mummu::pack::{Pack, Precision, SkipPoint};
use mummu::tier::Tier;
use mummu_num::{f32_from_u64, f32_from_usize, f64_from_usize, narrow};

const PROMPTS: &[&str] = &[
    "What is 2+2? Answer in one short sentence.",
    "Write a haiku about the sea.",
    "Explain what a hash map is to a beginner, in two sentences.",
    "Translate 'good morning' into French and German.",
    "List three prime numbers greater than 50.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "What is the capital of Australia?",
    "Give me a Python one-liner that reverses a string.",
];

fn log_softmax(t: &Tensor<2>) -> Vec<f32> {
    let v = t
        .clone()
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
    v.iter().map(|x| x - lse).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |m, (i, &x)| {
            if x > m.1 { (i, x) } else { m }
        })
        .0
}

/// Largest |Δ| between two log-prob rows.
fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Every calibration prompt rendered as a Qwen3 user turn and tokenized.
fn encode_prompts(tok: &tokenizers::Tokenizer) -> Vec<Vec<u32>> {
    PROMPTS
        .iter()
        .map(|p| {
            let r = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(*p)]);
            tok.encode(r.as_str(), false).unwrap().get_ids().to_vec()
        })
        .collect()
}

/// The dense, exact reference: every prompt's first-forward log-probs from
/// the unpartitioned f32 model.
fn dense_reference(dir: &Path, device: &Device, prompts: &[Vec<u32>]) -> Vec<Vec<f32>> {
    let dense = qwen35::load_from_pack(dir, device, &|_| Precision::F32).expect("dense load");
    prompts
        .iter()
        .map(|p| {
            let mut c = dense.new_cache();
            log_softmax(&dense.forward(p, 0, &mut c, device))
        })
        .collect()
}

/// The measurement model's expert pool: cluster 0 of every layer stays
/// local, every other cluster runs remotely at f32.
fn measurement_pool(pack: &Pack, layers: usize, epl: usize, device: &Device) -> Arc<ExpertPool> {
    let rows: Vec<Vec<Arc<dyn ExpertExec>>> = (0..layers)
        .map(|l| {
            (1..epl)
                .map(|c| {
                    let w = qwen35::load_ffn_clusters(pack, l, &[c], Precision::F32, device)
                        .expect("cluster");
                    Arc::new(DeviceExpert {
                        weights: w,
                        device: device.clone(),
                        tier: Tier {
                            device: 0,
                            precision: Precision::F32,
                        },
                        bytes: 0,
                        native_ok: std::sync::atomic::AtomicBool::new(true),
                    }) as Arc<dyn ExpertExec>
                })
                .collect()
        })
        .collect();
    Arc::new(ExpertPool::new(rows))
}

/// Per-layer hotness rows from the pool's activation energy (flat: per
/// layer, clusters 1..epl), each normalized to sum to one. Cluster 0 ran
/// locally (unmeasured) and is given the mean share.
fn hotness_rows(energy: &[f64], layers: usize, epl: usize) -> Vec<Vec<f32>> {
    let mut hotness: Vec<Vec<f32>> = Vec::with_capacity(layers);
    for l in 0..layers {
        let remote = &energy[l * (epl - 1)..(l + 1) * (epl - 1)];
        let mean = remote.iter().sum::<f64>() / f64_from_usize(remote.len().max(1));
        let mut row: Vec<f64> = std::iter::once(mean)
            .chain(remote.iter().copied())
            .collect();
        let total: f64 = row.iter().sum::<f64>().max(1e-12);
        for v in &mut row {
            *v /= total;
        }
        hotness.push(row.iter().map(|&v| narrow(v)).collect());
    }
    hotness
}

/// One skip-table point at the model's current `tau`: every prompt against
/// its dense reference, plus the share of offered clusters the pool kept.
fn skip_point(
    model: &LoadedQwen35,
    pool: &ExpertPool,
    prompts: &[Vec<u32>],
    refs: &[Vec<f32>],
    device: &Device,
    tau: f32,
) -> SkipPoint {
    let _ = pool.take_dense_rows();
    let mut max_delta = 0f32;
    let mut agree = 0usize;
    for (p, r) in prompts.iter().zip(refs) {
        let mut c = model.new_cache();
        let out = log_softmax(&model.forward(p, 0, &mut c, device));
        max_delta = max_delta.max(max_abs_delta(&out, r));
        agree += usize::from(argmax(&out) == argmax(r));
    }
    let (kept, offered) = pool.take_dense_rows();
    SkipPoint {
        tau,
        max_delta_logprob: max_delta,
        argmax_agreement: f32_from_usize(agree) / f32_from_usize(prompts.len()),
        kept_fraction: if offered > 0 {
            f32_from_u64(kept) / f32_from_u64(offered)
        } else {
            1.0
        },
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        eprintln!("usage: pack-calibrate <pack dir> [tau,tau,...]");
        std::process::exit(2);
    };
    let dir = PathBuf::from(dir);
    let taus: Vec<f32> = args.get(1).map_or_else(
        || vec![0.001, 0.003, 0.01, 0.02, 0.05],
        |s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
    );
    let mut pack = Pack::open(&dir).unwrap_or_else(|e| {
        eprintln!("{}: {e}", dir.display());
        std::process::exit(1);
    });
    let Some(part) = pack.manifest.ffn_partition.clone() else {
        eprintln!(
            "{} is not partitioned (run pack-partition first)",
            dir.display()
        );
        std::process::exit(1);
    };
    let header = pack.header().expect("header");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&header).expect("tokenizer");
    let cfg = qwen35::Qwen35Config::from_gguf(&header).expect("config");
    drop(header);
    let layers = part.layers.len();
    let epl = part.layers[0].len();
    let device = mummu::backend::cpu_device();
    let prompts = encode_prompts(&tok);

    let started = Instant::now();
    // Reference: dense, exact.
    let refs = dense_reference(&dir, &device, &prompts);
    eprintln!(
        "[calibrate] dense reference over {} prompts in {:.0}s",
        prompts.len(),
        started.elapsed().as_secs_f32()
    );

    // Measurement model: cluster 0 local, every other cluster remote at f32.
    let pool = measurement_pool(&pack, layers, epl, &device);
    let mut model =
        qwen35::load_from_pack_partitioned(&dir, &device, &|_| Precision::F32, &|_| vec![0])
            .expect("partitioned load")
            .with_ffn_pool(pool.clone());
    eprintln!(
        "[calibrate] measurement model loaded ({:.0}s)",
        started.elapsed().as_secs_f32()
    );

    // Exact pass (tau = 0): sanity + hotness.
    let _ = pool.take_energy();
    let _ = pool.take_dense_rows();
    let mut worst_exact = 0f32;
    for (p, r) in prompts.iter().zip(&refs) {
        let mut c = model.new_cache();
        let out = log_softmax(&model.forward(p, 0, &mut c, &device));
        worst_exact = worst_exact.max(max_abs_delta(&out, r));
    }
    eprintln!("[calibrate] exact tiered vs dense: max |Δlogprob| = {worst_exact:.3e}");
    let hotness = hotness_rows(&pool.take_energy(), layers, epl);

    // Skip table.
    let mut table = Vec::with_capacity(taus.len());
    for &tau in &taus {
        model = model.with_ffn_skip(tau);
        let point = skip_point(&model, &pool, &prompts, &refs, &device, tau);
        eprintln!(
            "[calibrate] tau={tau}: max |Δlogprob| {:.3}, argmax agreement {:.0}%, clusters kept {:.1}%",
            point.max_delta_logprob,
            point.argmax_agreement * 100.0,
            point.kept_fraction * 100.0
        );
        table.push(point);
    }

    let mut part = part;
    part.hotness = hotness;
    part.skip_table = table;
    pack.manifest.ffn_partition = Some(part);
    pack.save_manifest().expect("save manifest");
    eprintln!(
        "[calibrate] wrote hotness ({layers}×{epl}) and {} skip points into {} in {:.0}s",
        taus.len(),
        dir.join("manifest.json").display(),
        started.elapsed().as_secs_f32()
    );
    let _ = cfg;
}
