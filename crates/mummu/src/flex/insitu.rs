//! **In-situ bandwidth accounting for the host weight stream (SPEC P1.1).**
//!
//! The open finding this instrument exists for: the packed host GEMV
//! measures 46-48 GB/s in a quiet microbench and ~10 GB/s effective inside
//! the serving process (5.2 ms/call vs 1.1-1.6 quiet on `mlp.down`) — a
//! 2-3x inflation with no attributed cause. `prof.rs` records *time*;
//! attribution needs *bytes over time*, per call, on the production path —
//! not a probe beside it (the probe-the-production-path lesson, again).
//!
//! [`record`] is called by the packed GEMV/GEMM with the bytes it actually
//! streamed and the wall time it took; the ledger keeps per-shape
//! statistics (count, bytes, a bounded ring of per-call times for
//! quantiles, and the best call ever seen — the in-process ceiling proxy).
//! [`report`] renders the achieved-GB/s table that turns "live is slower"
//! into "THESE shapes at THESE times", and the identity
//! `beta_hat = bytes / dt` per shape is exactly SPEC P1.1's estimator.
//!
//! The second half of the spec — the (threads x priority x background)
//! ANOVA grid — is the statistics here ([`cell_stats`], [`main_effect`])
//! plus `examples/insitu-anova.rs`, which runs the SAME kernel the serve
//! path runs under each cell and prints the three contrasts. The decision
//! rule from the spec: the largest |contrast| is the first code change,
//! and the kernel is innocent iff the quiet cell reaches ~85% of the
//! roofline while the live ratio stays under ~0.4.
//!
//! Overhead: one `Instant::now` pair and one mutex insert per projection
//! call (~microseconds against multi-ms calls). `MUMMU_INSITU=0` turns
//! recording off entirely.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

/// Recording on? `MUMMU_INSITU`, default on (`0`/`off`/`false` disables).
#[must_use]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MUMMU_INSITU").map_or(true, |v| {
            !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        })
    })
}

/// Per-call samples kept per shape for quantiles (bounded ring).
const RING: usize = 512;

#[derive(Debug, Default, Clone)]
struct ShapeStats {
    calls: u64,
    bytes: u128,
    total_ns: u128,
    /// Fastest single call — the in-process ceiling proxy.
    best_ns: u64,
    /// Bounded ring of recent per-call ns.
    recent_ns: Vec<u64>,
    cursor: usize,
}

fn ledger() -> &'static Mutex<HashMap<(usize, usize, usize), ShapeStats>> {
    static MAP: OnceLock<Mutex<HashMap<(usize, usize, usize), ShapeStats>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record one call on the production path: shape `[k, n]` at batch `m`,
/// `bytes` actually streamed, `dur` wall time. No-op when disabled.
pub fn record(k: usize, n: usize, m: usize, bytes: usize, dur: Duration) {
    if !enabled() {
        return;
    }
    let ns = u64::try_from(dur.as_nanos()).unwrap_or(u64::MAX);
    let mut map = ledger().lock().unwrap_or_else(PoisonError::into_inner);
    let e = map.entry((k, n, m)).or_default();
    e.calls += 1;
    e.bytes += bytes as u128;
    e.total_ns += u128::from(ns);
    e.best_ns = if e.best_ns == 0 {
        ns
    } else {
        e.best_ns.min(ns)
    };
    if e.recent_ns.len() < RING {
        e.recent_ns.push(ns);
    } else {
        e.recent_ns[e.cursor] = ns;
        e.cursor = (e.cursor + 1) % RING;
    }
}

/// Forget everything (per-request reporting resets between requests).
pub fn reset() {
    ledger()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
}

/// One shape's rendered row.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeReport {
    pub k: usize,
    pub n: usize,
    pub m: usize,
    pub calls: u64,
    pub gib_moved: f64,
    /// `bytes / total_time` — the mean achieved bandwidth.
    pub mean_gbps: f64,
    /// Bandwidth of the median recent call.
    pub p50_gbps: f64,
    /// Bandwidth of the fastest call ever — the in-process ceiling proxy.
    pub best_gbps: f64,
    /// Mean per-call milliseconds.
    pub mean_ms: f64,
}

/// The report, sorted by total bytes moved descending (the shapes that own
/// the token come first).
#[must_use]
pub fn shape_reports() -> Vec<ShapeReport> {
    let map = ledger().lock().unwrap_or_else(PoisonError::into_inner);
    let mut rows: Vec<ShapeReport> = map
        .iter()
        .map(|(&(k, n, m), s)| {
            let bytes_per_call = if s.calls > 0 {
                s.bytes as f64 / s.calls as f64
            } else {
                0.0
            };
            let mut recent = s.recent_ns.clone();
            recent.sort_unstable();
            let p50_ns = recent.get(recent.len() / 2).copied().unwrap_or(0);
            let gbps = |ns: f64, bytes: f64| {
                if ns > 0.0 { bytes / ns } else { 0.0 } // bytes/ns == GB/s
            };
            ShapeReport {
                k,
                n,
                m,
                calls: s.calls,
                gib_moved: s.bytes as f64 / f64::from(1u32 << 30),
                mean_gbps: gbps(s.total_ns as f64 / s.calls.max(1) as f64, bytes_per_call),
                p50_gbps: gbps(p50_ns as f64, bytes_per_call),
                best_gbps: gbps(s.best_ns as f64, bytes_per_call),
                mean_ms: s.total_ns as f64 / s.calls.max(1) as f64 / 1e6,
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        (b.gib_moved)
            .partial_cmp(&a.gib_moved)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

/// Render the table plus the aggregate line. Empty ledger renders a
/// one-line note rather than nothing, so a silent misconfiguration is
/// visible.
#[must_use]
pub fn report() -> String {
    let rows = shape_reports();
    if rows.is_empty() {
        return "[insitu] no packed host GEMV calls recorded".to_string();
    }
    let mut out = String::from(
        "[insitu] host weight-stream bandwidth (beta_hat = bytes/dt, per call)\n\
         [insitu]      shape          m    calls    GiB    mean GB/s   p50 GB/s   best GB/s   mean ms\n",
    );
    let (mut bytes, mut ns) = (0f64, 0f64);
    for r in &rows {
        out.push_str(&format!(
            "[insitu] {:>6} x {:<6} {:>4} {:>7} {:>7.2} {:>10.1} {:>10.1} {:>10.1} {:>9.2}\n",
            r.k, r.n, r.m, r.calls, r.gib_moved, r.mean_gbps, r.p50_gbps, r.best_gbps, r.mean_ms
        ));
        bytes += r.gib_moved * f64::from(1u32 << 30);
        ns += r.mean_ms * 1e6 * r.calls as f64;
    }
    out.push_str(&format!(
        "[insitu] aggregate: {:.2} GiB at {:.1} GB/s effective\n",
        bytes / f64::from(1u32 << 30),
        if ns > 0.0 { bytes / ns } else { 0.0 }
    ));
    out
}

// ---------------------------------------------------------------------------
// The ANOVA-grid statistics (SPEC P1.1's estimator + contrasts)
// ---------------------------------------------------------------------------

/// Robust summary of one grid cell's repeated measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct CellStats {
    pub n: usize,
    pub median_ms: f64,
    pub q05_ms: f64,
    pub q95_ms: f64,
}

/// Median and 5/95 quantiles of one cell (nearest-rank). Panics on empty
/// input — a cell with no measurements is a harness bug.
#[must_use]
pub fn cell_stats(samples_ms: &[f64]) -> CellStats {
    assert!(!samples_ms.is_empty(), "cell_stats: empty cell");
    let mut v: Vec<f64> = samples_ms.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |q: f64| {
        let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
        v[idx.min(v.len() - 1)]
    };
    CellStats {
        n: v.len(),
        median_ms: at(0.5),
        q05_ms: at(0.05),
        q95_ms: at(0.95),
    }
}

/// A main-effect contrast: median(a) − median(b), with a crude significance
/// note — the difference is "clear" when the two cells' 5–95% bands do not
/// overlap. That is deliberately conservative: a 30-rep cell on a live
/// machine has heavy tails, and the spec's decision rule acts on the
/// LARGEST contrast, where overlap ambiguity matters least.
#[derive(Debug, Clone, PartialEq)]
pub struct Contrast {
    pub delta_ms: f64,
    pub clear: bool,
}

#[must_use]
pub fn main_effect(a: &CellStats, b: &CellStats) -> Contrast {
    Contrast {
        delta_ms: a.median_ms - b.median_ms,
        clear: a.q05_ms > b.q95_ms || b.q05_ms > a.q95_ms,
    }
}

/// SPEC P1.1's acceptance rule for the kernel itself: quiet-cell bandwidth
/// at ≥ 85% of the measured roofline says the kernel is innocent; a live
/// ratio under 0.4 says the environment owes the rest.
#[must_use]
pub fn kernel_innocent(quiet_gbps: f64, roofline_gbps: f64, live_gbps: f64) -> bool {
    roofline_gbps > 0.0 && quiet_gbps >= 0.85 * roofline_gbps && live_gbps / quiet_gbps < 0.4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_accumulates_and_reports() {
        reset();
        // 1 GiB in 0.1 s = 10.74 GB/s (decimal GB).
        record(5120, 17408, 1, 1 << 30, Duration::from_millis(100));
        record(5120, 17408, 1, 1 << 30, Duration::from_millis(50));
        let rows = shape_reports();
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.calls, 2);
        assert!((r.gib_moved - 2.0).abs() < 1e-9);
        // Mean: 2 GiB over 150 ms = 14.3 GB/s; best: 1 GiB / 50 ms = 21.5.
        assert!((r.mean_gbps - (2.0 * (1u64 << 30) as f64 / 150e6)).abs() < 0.1);
        assert!((r.best_gbps - ((1u64 << 30) as f64 / 50e6)).abs() < 0.1);
        assert!(report().contains("5120"));
        reset();
        assert!(report().contains("no packed host GEMV"));
    }

    #[test]
    fn quantiles_are_nearest_rank() {
        let s = cell_stats(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(s.median_ms, 3.0);
        assert_eq!(s.q05_ms, 1.0);
        assert_eq!(s.q95_ms, 5.0);
        assert_eq!(s.n, 5);
    }

    #[test]
    fn contrasts_flag_only_separated_cells() {
        let slow = cell_stats(&[10.0, 11.0, 12.0]);
        let fast = cell_stats(&[1.0, 1.1, 1.2]);
        let c = main_effect(&slow, &fast);
        assert!(c.clear && c.delta_ms > 8.0);
        let noisy_a = cell_stats(&[1.0, 5.0, 9.0]);
        let noisy_b = cell_stats(&[2.0, 5.5, 8.0]);
        assert!(!main_effect(&noisy_a, &noisy_b).clear);
    }

    #[test]
    fn innocence_rule_matches_the_spec() {
        // The measured numbers: roofline 42.2, quiet 46-48, live ~10.
        assert!(kernel_innocent(46.4, 42.2, 10.0));
        // A kernel that cannot reach the roofline even quiet is not innocent.
        assert!(!kernel_innocent(20.0, 42.2, 10.0));
        // Live near quiet: no environmental inflation to explain.
        assert!(!kernel_innocent(46.4, 42.2, 40.0));
    }
}
