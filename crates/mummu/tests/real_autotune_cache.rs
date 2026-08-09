//! Real-GPU proof for `mummu::tune`: CubeCL really does persist autotune
//! picks to the directory we report, and clearing it really does remove them.
//!
//! This test deliberately deletes the cache — its own crate's, never the
//! benchmark crate's. CubeCL resolves the cache root by walking up from the
//! process CWD for a `Cargo.toml`, and cargo runs these tests with CWD =
//! `crates/mummu`, so the tree touched here is `crates/mummu/target/autotune`
//! while the recorded benchmark numbers keep tuning out of
//! `crates/mummu-bench/target/autotune`.
//!
//! ```text
//! cargo test -p mummu --release --test real_autotune_cache -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use burn::tensor::Tensor;
use mummu::backend::{Gpu, use_gpu};
use mummu::tune::{autotune_cache_report, clear_autotune_cache};

/// Square matmul side. Big enough that CubeCL autotunes it (and small enough
/// to stay well inside any GPU's memory).
const N: usize = 512;
/// Matmuls run to provoke tuning.
const ROUNDS: usize = 4;
/// Autotune commits its winner from a worker, so the file may land shortly
/// after the work does. Bounded wait, not a sleep-and-hope.
const WRITE_TRIES: u32 = 40;
const WRITE_INTERVAL: Duration = Duration::from_millis(250);

#[test]
#[ignore = "needs a real GPU, and deletes this crate's autotune cache"]
fn autotune_picks_are_persisted_where_we_report_and_clearing_removes_them() {
    if !use_gpu() {
        eprintln!("[tune] no GPU adapter — skipping");
        return;
    }

    // Start from a known-empty cache so "files appeared" means this run.
    let cleared = clear_autotune_cache().expect("clearing succeeds");
    eprintln!(
        "[tune] cache dir {:?} — cleared {} files / {} bytes",
        cleared.dir, cleared.files, cleared.bytes
    );
    let empty = autotune_cache_report().expect("report after clear");
    assert!(
        empty.is_empty(),
        "cache must be empty right after a clear, found {} files",
        empty.files
    );
    assert_eq!(empty.dir, cleared.dir, "the reported dir must be stable");

    // Provoke autotuning with real GPU work: a few matmuls, each result read
    // back so the work is actually executed rather than queued.
    let device = burn::tensor::Device::<Gpu>::default();
    for round in 0..ROUNDS {
        let a = Tensor::<Gpu, 2>::ones([N, N], &device).mul_scalar(1.0 + round as f32);
        let b = Tensor::<Gpu, 2>::ones([N, N], &device);
        let sum = a
            .matmul(b)
            .sum()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("matmul readback");
        assert_eq!(sum.len(), 1, "sum reduces to one element");
        assert!(sum[0].is_finite(), "matmul produced a non-finite sum");
    }

    // The winner is committed from a worker thread, so poll (bounded).
    let start = Instant::now();
    let mut report = autotune_cache_report().expect("report after work");
    for _ in 0..WRITE_TRIES {
        if !report.is_empty() {
            break;
        }
        std::thread::sleep(WRITE_INTERVAL);
        report = autotune_cache_report().expect("report while waiting");
    }
    eprintln!(
        "[tune] after {ROUNDS} matmuls: {} files / {} bytes in {:?} (waited {:?})",
        report.files,
        report.bytes,
        report.dir,
        start.elapsed()
    );
    assert!(
        !report.is_empty(),
        "GPU work must leave autotune picks in {:?}",
        report.dir
    );
    assert!(report.bytes > 0, "persisted picks cannot be zero bytes");

    // And clearing takes them away again — the "re-tune on next launch" action.
    let cleared = clear_autotune_cache().expect("second clear succeeds");
    assert_eq!(
        cleared.files, report.files,
        "the clear must report exactly what it removed"
    );
    let after = autotune_cache_report().expect("report after second clear");
    assert!(after.is_empty(), "cache must be empty after clearing");
}
