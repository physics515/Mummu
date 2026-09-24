//! Does burn 0.22's graph capture cut wgpu's per-dispatch CPU cost?
//!
//! The roadmap has named graph capture "the lever" for dispatch-bound decode
//! since 2026-08-06 without ever measuring it. Two *different* mechanisms ship
//! under that name in 0.22.0-pre.3 and only one of them is the lever:
//!
//! - `Device::capture()` / `Device::capture_scope` (feature `capture`) records
//!   a `CapturedGraph`/`GraphIr` **without executing** — an op-graph recorder
//!   whose only executing consumer is `burn-remote`'s server (`Graph::replay`
//!   re-registers ops one at a time). It saves *transport*, not launches.
//! - `burn::tensor::capture(device, closure)` -> `Graph::replay()` is the real
//!   one: on wgpu it builds a **software graph** (`cubecl-wgpu`'s `WgpuGraph`)
//!   that resolves pipeline lookup, binding resolution, info-uniform upload and
//!   bind-group creation ONCE at record time, then re-encodes prebuilt state.
//!
//! Replay stays O(n) in recorded dispatches under WebGPU — only the per-launch
//! constant shrinks — so the win is confined to sizes where launch overhead is
//! a large share of the pass. cubecl's own module docs say to measure rather
//! than assume. This is that measurement, on the decode-shaped end of the
//! sweep: a dependent chain of matmuls (matmuls do not fuse into each other,
//! so the chain stays N dispatches even with the `fusion` feature on).
//!
//! Run: `cargo run --release -p mummu --example graph-capture-probe`

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::cell::Cell;
use std::rc::Rc;
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};
use mummu_num::f64_from_usize;

/// Square sizes swept, smallest first: small = launch-overhead dominated (the
/// decode regime), large = GPU-bound (where any win must vanish, which is the
/// control that says the harness is measuring what it claims).
const SIZES: [usize; 5] = [64, 128, 256, 512, 1024];

/// Dependent matmuls per captured closure. Bounded, and deliberately in the
/// same order of magnitude as one transformer layer's dispatch count so the
/// per-launch constant is read at a realistic graph length.
const CHAIN: usize = 32;

/// Timed iterations per arm. Bounded; enough to average out scheduler noise
/// without letting one size dominate the probe's wall-clock.
const ITERS: usize = 50;

/// Untimed iterations run before the uncaptured arm is timed. Without these the
/// probe compares a COLD uncaptured arm against a replay that `capture` already
/// warmed three times, and autotune's per-size search lands entirely in arm A —
/// which is exactly the artifact that made this sweep non-monotonic on its
/// first run (64 us/dispatch reading slower than 256).
const WARMUP: usize = 10;

/// Largest square side accepted, so a typo cannot ask for a multi-GiB
/// allocation on a card that must also hold a display.
const MAX_SIDE: usize = 4096;

/// Time one arm: `run` is invoked `ITERS` times with a device sync after each,
/// which is the honest decode shape (every token ends in a readback).
/// Returns microseconds per *dispatch*.
fn time_per_dispatch(device: &burn::tensor::Device, mut run: impl FnMut()) -> f64 {
    let started = Instant::now();
    for _ in 0..ITERS {
        run();
        let _ = burn::tensor::Device::sync(device);
    }
    let elapsed = started.elapsed().as_secs_f64();
    assert!(elapsed > 0.0, "timer returned a non-positive interval");
    let dispatches = f64_from_usize(ITERS * CHAIN);
    assert!(dispatches > 0.0, "dispatch count must be positive");
    elapsed * 1e6 / dispatches
}

/// Sweep one square size: build a dependent matmul chain, time it uncaptured,
/// capture it, then time the replay. Returns
/// `(uncaptured_us, replayed_us, hardware_graph)` where `hardware_graph` is
/// false when burn fell back to re-running the closure.
fn sweep_size(device: &burn::tensor::Device, side: usize) -> (f64, f64, bool) {
    assert!(side > 0, "square side must be positive");
    assert!(
        side <= MAX_SIDE,
        "square side {side} exceeds the {MAX_SIDE} bound"
    );

    // Scale the operand so a 32-long chain neither overflows nor denormalizes:
    // entries ~N(0, 1/side) keep each product's magnitude near the input's.
    let scale = 1.0 / f64_from_usize(side).sqrt();
    let x = Tensor::<2>::random([side, side], Distribution::Normal(0.0, scale), device);
    let w = Tensor::<2>::random([side, side], Distribution::Normal(0.0, scale), device);

    // `replay()` re-runs the closure ONLY on the fallback path (no hardware or
    // software graph). Counting invocations is therefore a direct, dependency
    // free test of which path burn took — no private field to inspect.
    let calls = Rc::new(Cell::new(0usize));
    let counter = Rc::clone(&calls);
    let x_captured = x;
    let w_captured = w;
    let chain = move || {
        counter.set(counter.get() + 1);
        let mut acc = x_captured.clone();
        for _ in 0..CHAIN {
            acc = acc.matmul(w_captured.clone());
        }
        acc
    };

    // Warm up BEFORE timing anything: trigger autotune's per-size kernel search
    // and populate the pools, so arm A measures steady-state dispatch cost
    // rather than a one-off tuning sweep.
    for _ in 0..WARMUP {
        let held = chain();
        let _ = burn::tensor::Device::sync(device);
        drop(held);
    }

    // Arm A: the ordinary path, closure re-executed every iteration.
    let uncaptured = time_per_dispatch(device, || {
        let _held = chain();
    });

    // Arm B: capture once, then replay. `capture` warms up (3 runs) before
    // recording, so autotune and every pool allocation happen before the
    // window opens.
    let mut graph = burn::tensor::capture(device, chain);
    let before_replays = calls.get();
    let replayed = time_per_dispatch(device, || {
        // SAFETY: the captured closure owns clones of `x`/`w` and its output
        // is owned by `graph`, so every buffer the recording touched is alive
        // for as long as `graph` is. Nothing else on this thread touches those
        // tensors, and this probe is single-threaded on one device, so the
        // replay is ordered against the sync that follows it on the same
        // stream.
        let _ = unsafe { graph.replay() };
    });
    let replay_calls = calls.get() - before_replays;
    assert!(
        replay_calls == 0 || replay_calls == ITERS,
        "replay must either always re-run the closure (fallback) or never (graph); saw {replay_calls} of {ITERS}"
    );

    (uncaptured, replayed, replay_calls == 0)
}

fn main() {
    let device = mummu::backend::gpu_device();
    println!("device: {device:?}");
    println!("chain={CHAIN} dependent matmuls, {ITERS} iters/arm, sync after each iter\n");
    println!(
        "{:>6}  {:>14}  {:>14}  {:>9}  {:>8}",
        "side", "uncaptured us", "replayed us", "saved us", "path"
    );

    for side in SIZES {
        let (uncaptured, replayed, hardware) = sweep_size(&device, side);
        let saved = uncaptured - replayed;
        let path = if hardware { "graph" } else { "fallback" };
        println!("{side:>6}  {uncaptured:>14.2}  {replayed:>14.2}  {saved:>9.2}  {path:>8}");
    }

    println!(
        "\nus are per dispatch. A real software graph shows a roughly constant\n\
         per-dispatch saving that is a large share of the small-side rows and\n\
         vanishes into the noise on the GPU-bound large-side rows."
    );
}
