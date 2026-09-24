//! SPEC P1.1's diagnostic instrument: the (threads x priority x memory
//! contender) ANOVA over the SAME packed host GEMV serve runs, so the live
//! ~2-3x inflation over the quiet microbench decomposes into named,
//! measured effects instead of suspicion.
//!
//! Run on the machine under test (quiet for the baseline cells; a busy
//! desktop is itself one of the effects being measured):
//!
//! ```text
//! cargo run --release -p mummu --example insitu-anova --no-default-features --features vulkan-spirv
//! ```
//!
//! Grid: rayon width {4, 8, 16} x worker priority {below-normal, normal} x
//! memory contender {off, on}. Each cell runs R = 30 reps of a
//! DRAM-regime GEMV (round-robin over enough packed tensors to defeat the
//! L3) and reports median + 5/95 bands; the three main-effect contrasts
//! follow, plus SPEC P1.1's decision rule: the largest |contrast| names
//! the first code change, and the kernel is innocent iff its quiet cell
//! reaches >= 85% of the measured roofline. The `wgpu-live` axis from the
//! spec is approximated by the memory contender (a thread streaming a
//! buffer, the bus-level effect wgpu staging has); replace it with a real
//! GPU loop when hunting a wgpu-specific interaction.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use mummu::flex::insitu::{CellStats, cell_stats, kernel_innocent, main_effect};
use mummu::flex::kernels::{self, PackedQ4};
use mummu_num::{f32_from_usize, f64_from_usize};
use rayon::prelude::*;

/// One ANOVA cell: `(shape, batch, fusion, autotune, stats)`.
type Cell = (String, usize, bool, bool, CellStats);

/// Timed reps per cell.
const REPS: usize = 30;

fn wave(len: usize, f: f32) -> Vec<f32> {
    (0..len).map(|i| (f32_from_usize(i) * f).sin()).collect()
}

#[cfg(windows)]
fn set_current_thread_priority(below_normal: bool) {
    #[link(name = "kernel32.dll", kind = "raw-dylib", modifiers = "+verbatim")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(handle: isize, priority: i32) -> i32;
    }
    // SAFETY: plain kernel32 calls on the current thread's pseudo handle;
    // -1 = BELOW_NORMAL, 0 = NORMAL.
    unsafe {
        SetThreadPriority(GetCurrentThread(), if below_normal { -1 } else { 0 });
    }
}

#[cfg(not(windows))]
const fn set_current_thread_priority(_below_normal: bool) {}

/// The DRAM read roofline for this box (the innocence denominator): a
/// threaded sum over 1 GiB, best of 3, in GB/s.
fn dram_roofline_gbps() -> f64 {
    let words = (1usize << 30) / 8;
    let buf: Vec<u64> = (0..words as u64).collect();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let s: u64 = buf
            .par_chunks(1 << 16)
            .map(|c| c.iter().fold(0u64, |a, &b| a.wrapping_add(b)))
            .reduce(|| 0, u64::wrapping_add);
        std::hint::black_box(s);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    let gbps = f64_from_usize(words * 8) / (best * 1e6);
    println!("dram read roofline: {gbps:.1} GB/s\n");
    gbps
}

/// `REPS` timed GEMVs on `pool`, round-robin over `tensors` so the stream
/// defeats the L3, after one warm call outside the clock.
fn time_cell(pool: &rayon::ThreadPool, tensors: &[PackedQ4], x: &[f32], n: usize) -> CellStats {
    let mut out = vec![0.0f32; n];
    // Warm the pool + pages outside the clock.
    pool.install(|| kernels::gemv_q4n_auto(&tensors[0], x, &mut out));
    let mut samples = Vec::with_capacity(REPS);
    for r in 0..REPS {
        let w = &tensors[r % tensors.len()];
        let t0 = std::time::Instant::now();
        pool.install(|| kernels::gemv_q4n_auto(w, x, &mut out));
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    cell_stats(&samples)
}

/// One main-effect contrast line.
fn print_contrast(label: &str, delta_ms: f64, clear: bool) {
    println!(
        "{label} {delta_ms:+.2} ms {}",
        if clear { "(clear)" } else { "(within noise)" }
    );
}

/// SPEC P1.1's decision rule. Quiet = widest pool, normal priority, no
/// contender; live proxy = same pool with the contender on.
fn print_verdict(cells: &[Cell], bytes_per_call: usize, roofline_gbps: f64) {
    let quiet = cells
        .iter()
        .find(|c| c.1 == 16 && !c.2 && !c.3)
        .expect("quiet cell");
    let live = cells
        .iter()
        .find(|c| c.1 == 16 && !c.2 && c.3)
        .expect("live cell");
    let quiet_gbps = f64_from_usize(bytes_per_call) / (quiet.4.median_ms * 1e6);
    let live_gbps = f64_from_usize(bytes_per_call) / (live.4.median_ms * 1e6);
    println!(
        "\nquiet {quiet_gbps:.1} GB/s vs roofline {roofline_gbps:.1}: kernel {} \
         (live-proxy {live_gbps:.1} GB/s, ratio {:.2})",
        if quiet_gbps >= 0.85 * roofline_gbps {
            "INNOCENT — chase the environment"
        } else {
            "NOT at the roofline — look at the kernel first"
        },
        live_gbps / quiet_gbps,
    );
    let _ = kernel_innocent(quiet_gbps, roofline_gbps, live_gbps);
    println!("largest |contrast| is the first code change (SPEC P1.1's decision rule).");
}

fn main() {
    println!("vnni available: {}", kernels::vnni_available());
    let roofline_gbps = dram_roofline_gbps();

    // Enough production-shaped tensors that round-robin defeats the 128 MB
    // L3: 6 x [5120, 17408] packed ~= 300 MB of stream.
    let (k, n) = (5120usize, 17408usize);
    println!("packing 6 x [{k}, {n}] twins (~300 MB stream)…");
    let tensors: Vec<PackedQ4> = (0..6usize)
        .map(|i| {
            let vals = wave(k * n, f32_from_usize(i).mul_add(0.013, 0.11));
            PackedQ4::from_f32(&vals, k, n)
        })
        .collect();
    let bytes_per_call = tensors[0].streamed_bytes();
    let x = wave(k, 0.007);

    // The memory contender: a thread streaming 512 MiB in a loop — the
    // bus-level shape of wgpu staging traffic and a busy desktop.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spawn_contender = |on: bool| -> Option<std::thread::JoinHandle<()>> {
        if !on {
            return None;
        }
        let stop = std::sync::Arc::clone(&stop);
        Some(std::thread::spawn(move || {
            let buf: Vec<u64> = (0..(512usize << 20) / 8).map(|i| i as u64).collect();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let s = buf.iter().fold(0u64, |a, &b| a.wrapping_add(b));
                std::hint::black_box(s);
            }
        }))
    };

    let mut cells: Vec<Cell> = Vec::new();
    for &threads in &[4usize, 8, 16] {
        for &below in &[true, false] {
            for &contend in &[false, true] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .start_handler(move |_| set_current_thread_priority(below))
                    .build()
                    .expect("cell pool");
                stop.store(false, std::sync::atomic::Ordering::Relaxed);
                let contender = spawn_contender(contend);
                let stats = time_cell(&pool, &tensors, &x, n);
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                if let Some(h) = contender {
                    let _ = h.join();
                }
                let label = format!(
                    "threads {threads:>2} | {} | contender {}",
                    if below { "below " } else { "normal" },
                    if contend { "on " } else { "off" },
                );
                let gbps = f64_from_usize(bytes_per_call) / (stats.median_ms * 1e6);
                println!(
                    "{label}: median {:>6.2} ms [{:>6.2}, {:>6.2}] = {gbps:.1} GB/s",
                    stats.median_ms, stats.q05_ms, stats.q95_ms
                );
                cells.push((label, threads, below, contend, stats));
            }
        }
    }

    // Main-effect contrasts: pool each factor's levels (medians of cell
    // medians would hide reps; pool the raw medians per level instead).
    let pooled = |pick: &dyn Fn(&Cell) -> bool| -> CellStats {
        let ms: Vec<f64> = cells
            .iter()
            .filter(|c| pick(c))
            .map(|c| c.4.median_ms)
            .collect();
        cell_stats(&ms)
    };
    println!();
    let c_threads = main_effect(&pooled(&|c| c.1 == 4), &pooled(&|c| c.1 == 16));
    print_contrast(
        "contrast threads (4 vs 16):       ",
        c_threads.delta_ms,
        c_threads.clear,
    );
    let c_prio = main_effect(&pooled(&|c| c.2), &pooled(&|c| !c.2));
    print_contrast(
        "contrast priority (below vs norm):",
        c_prio.delta_ms,
        c_prio.clear,
    );
    let c_cont = main_effect(&pooled(&|c| c.3), &pooled(&|c| !c.3));
    print_contrast(
        "contrast contender (on vs off):   ",
        c_cont.delta_ms,
        c_cont.clear,
    );

    print_verdict(&cells, bytes_per_call, roofline_gbps);
}
