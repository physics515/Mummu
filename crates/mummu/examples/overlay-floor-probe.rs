//! Overlay-floor probe (SPEC 5): measure REAL host->GPU staging bandwidth at
//! layer-slab size, then print the three-state planner's decision table for
//! the 27B under today's host FFN (~36 ms) and the post-VNNI host FFN (~6 ms)
//! — showing the crossover the planner respects from MEASURED inputs.
//!
//! The slab: one 27B layer's Q4-packed weights are ~134 MB ([5120 x 17408]
//! at ~4.5 bits/weight across three projections). We stage the same BYTE
//! COUNT as f32 rows scaled down — [1925 x 17408] f32 = 134.05 MB — because
//! the bus does not care what the bytes mean, only how many there are.
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};
use mummu::overlay::{min_vram_bytes, plan, LayerAction, LayerCost, OverlayModel};

fn main() {
    // ---- staging bandwidth, measured (or skipped gracefully) -------------
    let rows = 1925usize; // 1925 * 17408 * 4 B = 134.05 MB, one Q4 layer's worth
    let cols = 17408usize;
    let slab_bytes = (rows * cols * 4) as u64;

    let (tx_bytes_per_ms, label) = if mummu::backend::use_gpu() {
        let cpu = mummu::backend::cpu_device();
        let gpu = mummu::backend::gpu_device();
        let w = Tensor::<2>::random([rows, cols], Distribution::Default, &cpu);
        // Warm: first transfer pays allocator + staging-ring setup and must
        // not pollute the number (warm every mummu perf probe).
        let _ = w.clone().to_device(&gpu).into_data();
        let reps = 3u32;
        let t = Instant::now();
        for _ in 0..reps {
            // Same pattern as stage-probe: to_device + into_data forces
            // completion (round trip included — conservative for planning).
            let _ = w.clone().to_device(&gpu).into_data();
        }
        let per_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        let per_byte_ms = slab_bytes as f64 / per_ms;
        println!(
            "staging [{}x{}] f32 = {:.1} MB (Q4 layer-slab equivalent): {:.2} ms/rep over {reps} warm reps -> {:.1} GB/s",
            rows,
            cols,
            slab_bytes as f64 / 1e6,
            per_ms,
            per_byte_ms * 1e3 / 1e9,
        );
        (per_byte_ms, "measured")
    } else {
        println!(
            "no GPU adapter: skipping the staging measurement; \
             decision table below uses the roadmap's 25 GB/s ASSUMED figure"
        );
        (25e9 / 1e3, "ASSUMED 25 GB/s")
    };

    // ---- the planner's decision table for the 27B ------------------------
    // 64 layers; per-layer Q4 bytes = the slab above; dGPU per-layer compute
    // 0.72 ms (the measured paired figure). The budget models the 16 GB
    // card's real situation: 42 of 64 layers resident today.
    let num_layers = 64usize;
    let resident_today = 42u64;
    let gpu_ms: f64 = 0.72;
    let m = OverlayModel {
        tx_bytes_per_ms,
        slot_latency_ms: 0.25, // nominal submit+fence handoff
        ring_slots: 2,
        crossing_ms: 0.4, // nominal host<->device activation hop
    };
    let budget = resident_today * slab_bytes + m.ring_slots as u64 * slab_bytes;
    let tx_layer_ms = slab_bytes as f64 / tx_bytes_per_ms;

    println!(
        "\n27B decision table: {num_layers} layers x {:.1} MB Q4, gpu {gpu_ms} ms/layer, \
         tx {tx_layer_ms:.2} ms/layer ({label}), budget {:.2} GB ({resident_today} layers + ring)",
        slab_bytes as f64 / 1e6,
        budget as f64 / 1e9,
    );
    println!(
        "stream slot cost = max(gpu, tx) + lat/slots = {:.2} ms/layer -- the number host_ms must beat",
        gpu_ms.max(tx_layer_ms) + m.slot_latency_ms / m.ring_slots as f64
    );
    println!("\n  host_ms   resident  stream  host   predicted_ms   ring_MB   tail decision");
    for host_ms in [36.0, 6.0] {
        let layers: Vec<LayerCost> = (0..num_layers)
            .map(|_| LayerCost {
                bytes: slab_bytes,
                gpu_ms,
                host_ms,
            })
            .collect();
        let p = plan(&layers, budget, &m);
        let count = |want: LayerAction| p.actions.iter().filter(|a| **a == want).count();
        let tail = p
            .actions
            .iter()
            .rev()
            .find(|a| **a != LayerAction::Resident)
            .map_or("(all resident)", |a| match a {
                LayerAction::Stream => "STREAM the overflow",
                LayerAction::Host => "HOST the overflow",
                LayerAction::Resident => unreachable!(),
            });
        println!(
            "  {host_ms:>7.1}   {:>8}  {:>6}  {:>4}   {:>12.1}   {:>7.1}   {tail}",
            count(LayerAction::Resident),
            count(LayerAction::Stream),
            count(LayerAction::Host),
            p.predicted_token_ms,
            p.ring_bytes as f64 / 1e6,
        );
    }

    // ---- the capacity theorem, in numbers --------------------------------
    let layers: Vec<LayerCost> = (0..num_layers)
        .map(|_| LayerCost {
            bytes: slab_bytes,
            gpu_ms,
            host_ms: 36.0,
        })
        .collect();
    let floor = min_vram_bytes(&layers, &m, 512 << 20, 1 << 30);
    println!(
        "\ncapacity theorem: ring ({} x {:.1} MB) + 512 MB activations + 1 GB KV = {:.2} GB \
         streams ALL {num_layers} layers -- independent of model depth",
        m.ring_slots,
        slab_bytes as f64 / 1e6,
        floor as f64 / 1e9,
    );
}
