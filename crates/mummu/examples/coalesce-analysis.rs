//! Would coalescing adjacent clusters actually collapse the dispatch count?
//!
//! Placement is per cluster, but a *matmul* does not have to be. Clusters that
//! land on the same device at the same precision and are adjacent in the
//! intermediate dimension can be multiplied as one wider matmul: for gate/up
//! they are a contiguous column slice of `[hidden, inter]`, for down they are
//! contiguous rows whose partial products sum. Merging is exact.
//!
//! That only pays if the scheduler's assignment comes out in RUNS. If it
//! alternates device by device, every cluster is its own matmul and nothing
//! is saved. This runs the real planner over the real pack's cluster costs and
//! counts maximal runs — no weights loaded, no GPU needed, so it answers the
//! question in seconds instead of an eight-minute model load.
use mummu::pack::{Pack, Precision};
use mummu::tier::{DeviceClass, TierDevice, plan_tiers};
use std::path::PathBuf;

fn main() {
    let dir = std::env::var("PACK_DIR").unwrap_or_else(|_| {
        r"D:\Docker Containers\mummu\models\qwen3.8-27b-ud-q4ks\pack".to_string()
    });
    let pack = Pack::open(&PathBuf::from(&dir)).expect("open pack");
    let part = pack
        .manifest
        .ffn_partition
        .as_ref()
        .expect("pack has no FFN partition");
    let layers = part.layers.len();

    // The devices the serving planner builds, with the speeds it now measures
    // and the budgets the 27B run reported.
    let devices = vec![
        TierDevice {
            name: "cpu".into(),
            class: DeviceClass::Cpu,
            ladder: vec![Precision::F16, Precision::Q8, Precision::Q4],
            budget_bytes: 43 << 30,
            speed: 72,
            preload_units: 1407, // the trunk, charged as cluster equivalents
        },
        TierDevice {
            name: "wgpu".into(),
            class: DeviceClass::DiscreteGpu,
            ladder: vec![Precision::F16, Precision::Q8, Precision::Q4],
            budget_bytes: 9_663_676_416, // 9.0 GiB
            speed: 629,
            preload_units: 0,
        },
        TierDevice {
            name: "igpu".into(),
            class: DeviceClass::IntegratedGpu,
            ladder: vec![Precision::F16, Precision::Q8, Precision::Q4],
            budget_bytes: 5 << 30,
            speed: 71,
            preload_units: 0,
        },
    ];

    let mut costs = Vec::new();
    for l in 0..layers {
        costs.extend(mummu::partition::cluster_costs(&pack, l).expect("cluster costs"));
    }
    let per_layer = costs.len() / layers;
    println!("{} layers x {per_layer} clusters = {} total\n", layers, costs.len());

    let plan = plan_tiers(&devices, &costs, &[]).expect("plan");

    // Maximal runs of adjacent clusters sharing a (device, precision), counted
    // per layer — this is exactly how many matmuls a coalescing executor would
    // issue where today it issues `per_layer`.
    let mut runs_total = 0usize;
    let mut hist = std::collections::BTreeMap::<usize, usize>::new();
    for l in 0..layers {
        let row = &plan.tiers[l * per_layer..(l + 1) * per_layer];
        let mut runs = 1usize;
        for w in row.windows(2) {
            if w[0] != w[1] {
                runs += 1;
            }
        }
        runs_total += runs;
        *hist.entry(runs).or_default() += 1;
    }
    let before = costs.len();
    let after = runs_total;
    println!("matmuls per token, one FFN projection:");
    println!("  today (one per cluster)      {before}");
    println!("  coalesced (one per run)      {after}");
    println!("  reduction                    {:.1}x\n", before as f64 / after as f64);

    println!("runs per layer (how fragmented each layer's assignment is):");
    for (runs, layers_with) in &hist {
        println!("  {runs:3} runs  x {layers_with:3} layers");
    }

    // Where the clusters ended up, for context.
    let mut by_slot = std::collections::BTreeMap::<String, usize>::new();
    for t in &plan.tiers {
        *by_slot
            .entry(format!("{} @ {:?}", devices[t.device].name, t.precision))
            .or_default() += 1;
    }
    println!("\nplacement:");
    for (slot, n) in &by_slot {
        println!("  {slot:16} {n}");
    }
}
