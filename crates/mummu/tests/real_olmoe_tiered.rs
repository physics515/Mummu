//! P9 stage-3(b) gate on real weights: OLMoE served from a `.mummu` pack
//! with its experts **tiered** — spread across devices and precisions at
//! once behind `ExpertPool` — must (1) reproduce the same-backend pack
//! model when every tier is Q8 (the pooled execution path itself is
//! exact), (2) still answer correctly with a genuinely mixed plan (f32 on
//! the GPU when one is present, int8 and int4 on the CPU), and (3) keep
//! answering after a hot-swap re-tier driven by the routing hits.
//!
//! Ignored by default; run with
//!
//! ```text
//! MUMMU_OLMOE_GGUF_PATH=path/to/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
//!   cargo test -p mummu --release --test real_olmoe_tiered -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use burn::tensor::Tensor;

use mummu::models::CausalLm;
use mummu::models::olmoe;
use mummu::nn::{ExpertExec, ExpertPool};
use mummu::pack::{Pack, Precision};
use mummu::tier::{DeviceClass, Tier, TierDevice, plan_tiers, smooth_hotness};

fn gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MUMMU_OLMOE_GGUF_PATH")?);
    p.is_file().then_some(p)
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

/// Load one expert onto the backend a tier device denotes (0 = CPU, 1 = wgpu).
fn load_expert(pack: &Pack, layer: usize, index: usize, tier: Tier) -> Arc<dyn ExpertExec> {
    match tier.device {
        0 => Arc::new(
            olmoe::load_expert_from_pack(pack, layer, index, tier, &Default::default())
                .expect("cpu expert"),
        ),
        1 => Arc::new(
            olmoe::load_expert_from_pack(pack, layer, index, tier, &Default::default())
                .expect("gpu expert"),
        ),
        other => panic!("no device {other}"),
    }
}

#[tokio::test]
#[ignore = "needs the local OLMoE GGUF (MUMMU_OLMOE_GGUF_PATH)"]
async fn olmoe_tiered_experts_match_then_answer_across_devices() {
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

    // Import once (reused across runs).
    let pack_dir = path.parent().unwrap().join("pack-gate");
    if !Pack::is_pack(&pack_dir) {
        let _ = std::fs::remove_dir_all(&pack_dir);
        let t = std::time::Instant::now();
        mummu::pack::import_gguf(
            &path,
            &pack_dir,
            &Precision::ALL,
            &olmoe::pack_actions,
            |i, n, name| {
                if i % 100 == 0 {
                    eprintln!("[tiered-gate] import {i}/{n} {name}");
                }
            },
        )
        .expect("pack import");
        eprintln!(
            "[tiered-gate] imported in {:.0}s",
            t.elapsed().as_secs_f32()
        );
    }
    let pack = Pack::open(&pack_dir).expect("pack opens");
    let costs = olmoe::pack_expert_costs(&pack).expect("expert costs");
    let header = pack.header().unwrap();
    let cfg = olmoe::OlmoeConfig::from_gguf(&header).unwrap();
    drop(header);
    let (layers, epl) = (cfg.num_hidden_layers, cfg.num_experts);
    assert_eq!(costs.len(), layers * epl);
    eprintln!(
        "[tiered-gate] {} experts; per-expert bytes {:?}",
        costs.len(),
        costs[0].bytes
    );

    let logits_of = |m: &olmoe::LoadedOlmoeQ| {
        let mut cache = m.new_cache();
        m.forward(&prompt, 0, &mut cache, &device)
    };

    // 1. Pooled execution is exact: all-Q8 pool ≡ same-backend all-Q8 model.
    let ref_q8 = {
        let m =
            olmoe::load_from_pack(&pack_dir, &device, &|_| Precision::Q8).expect("q8 pack load");
        logits_of(&m)
    };
    let q8_tier = Tier {
        device: 0,
        precision: Precision::Q8,
    };
    let pool_q8 = Arc::new(ExpertPool::new(
        (0..layers)
            .map(|l| {
                (0..epl)
                    .map(|e| load_expert(&pack, l, e, q8_tier))
                    .collect()
            })
            .collect(),
    ));
    let pooled = olmoe::load_trunk_from_pack(&pack_dir, &device)
        .expect("trunk load")
        .with_pool(pool_q8.clone());
    let pooled_logits = logits_of(&pooled);
    let d: f32 = ref_q8
        .clone()
        .sub(pooled_logits.clone())
        .abs()
        .max()
        .into_scalar();
    eprintln!(
        "[tiered-gate] all-Q8 pool vs same-backend Q8: max |Δlogit| = {d:.3e}; tops {} / {}",
        argmax(&pooled_logits),
        argmax(&ref_q8)
    );
    assert_eq!(argmax(&pooled_logits), argmax(&ref_q8));
    assert!(
        d <= 2e-3,
        "pooled path must be exact up to summation order (Δ={d})"
    );
    let hits = pool_q8.take_hits();
    assert_eq!(
        hits.iter().sum::<u64>() as usize,
        prompt.len() * cfg.num_experts_per_tok * layers,
        "every routed row is counted"
    );
    drop(pooled);
    drop(pool_q8);

    // 2. A genuinely mixed plan: f32 on the GPU (if present), int8/int4 on the CPU.
    let gib = u64::from(1u32 << 30);
    let mut devices = vec![TierDevice {
        name: "cpu".into(),
        class: DeviceClass::Cpu,
        ladder: vec![Precision::Q8, Precision::Q4],
        budget_bytes: 5 * gib,
        speed: 1,
        preload_units: 0,
    }];
    if mummu::backend::use_gpu() {
        devices.push(TierDevice {
            name: "wgpu".into(),
            class: DeviceClass::DiscreteGpu,
            ladder: vec![Precision::F32, Precision::Q8],
            budget_bytes: 3 * gib,
            speed: 10,
            preload_units: 0,
        });
    }
    let plan = plan_tiers(&devices, &costs, &[]).expect("plan");
    eprintln!("[tiered-gate] plan: {:?}", plan.histogram());
    assert!(
        plan.histogram().len() >= 2,
        "the plan must actually mix tiers: {:?}",
        plan.histogram()
    );
    let t0 = std::time::Instant::now();
    let pool = Arc::new(ExpertPool::new(
        (0..layers)
            .map(|l| {
                (0..epl)
                    .map(|e| load_expert(&pack, l, e, plan.tiers[l * epl + e]))
                    .collect()
            })
            .collect(),
    ));
    eprintln!(
        "[tiered-gate] mixed pool loaded in {:.0}s; used {:?}",
        t0.elapsed().as_secs_f32(),
        pool.used_bytes(devices.len())
    );
    let mixed = olmoe::load_trunk_from_pack(&pack_dir, &device)
        .expect("trunk load")
        .with_pool(pool.clone());
    let f32_top = {
        let m =
            olmoe::load_from_pack(&pack_dir, &device, &|_| Precision::F32).expect("f32 pack load");
        argmax(&logits_of(&m))
    };
    let mixed_top = argmax(&logits_of(&mixed));
    eprintln!("[tiered-gate] mixed first token {mixed_top} (f32 {f32_top})");
    assert_eq!(
        mixed_top, f32_top,
        "mixed-tier first token diverges from f32"
    );
    let t1 = std::time::Instant::now();
    let ids = mixed
        .generate(
            &prompt,
            48,
            &mummu::decode::SamplerOptions::greedy(),
            &device,
            |_| std::ops::ControlFlow::Continue(()),
        )
        .await
        .expect("mixed decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!(
        "[tiered-gate] mixed: {} tokens in {:.1}s: {text:?}",
        ids.len(),
        t1.elapsed().as_secs_f32()
    );
    assert!(
        text.contains('4'),
        "expected the answer to mention 4, got: {text:?}"
    );

    // 3. Hot-swap from routing hits: re-plan with the smoothed hotness,
    // apply the moves live, generate again.
    let hits = pool.take_hits();
    let mut hotness = Vec::new();
    smooth_hotness(&mut hotness, &hits, 1.0);
    let next = plan_tiers(&devices, &costs, &hotness).expect("re-plan");
    let moves = plan.diff(&next);
    eprintln!(
        "[tiered-gate] re-tier: {} moves; next {:?}",
        moves.len(),
        next.histogram()
    );
    assert!(
        !moves.is_empty(),
        "hot experts should be promoted after a request"
    );
    let t2 = std::time::Instant::now();
    for &(flat, tier) in moves.iter().take(64) {
        let (l, e) = (flat / epl, flat % epl);
        let old = pool.swap(l, e, load_expert(&pack, l, e, tier));
        drop(old);
    }
    eprintln!(
        "[tiered-gate] swapped {} experts in {:.1}s",
        moves.len().min(64),
        t2.elapsed().as_secs_f32()
    );
    let ids = mixed
        .generate(
            &prompt,
            48,
            &mummu::decode::SamplerOptions::greedy(),
            &device,
            |_| std::ops::ControlFlow::Continue(()),
        )
        .await
        .expect("post-swap decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[tiered-gate] after swap: {text:?}");
    assert!(
        text.contains('4'),
        "post-swap answer should still mention 4, got: {text:?}"
    );
}
