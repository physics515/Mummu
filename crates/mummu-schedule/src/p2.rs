//! **P² — one running quantile in five numbers (Jain & Chlamtac 1985).**
//!
//! The chance constraint in [`crate::watermark`] needs a high quantile of
//! ambient VRAM consumption — the memory *other* processes take on the same
//! card — observed one NVML poll at a time over a session that can run for
//! days. Keeping every sample and sorting would be exact, but the stream is
//! unbounded and the estimate is consulted on every placement decision, so
//! the right shape is O(1) memory and O(1) per observation.
//!
//! P² maintains five markers: the minimum, the target quantile, the maximum,
//! and two intermediates halfway (in probability) to either side. Each
//! marker has a height (a sample value), an actual position (how many
//! samples sit at or below it), and a desired position (where an ideal
//! order statistic for its probability would sit after n samples). When a
//! marker drifts a full step from its desired position, its height is moved
//! by fitting a parabola through it and its neighbors — the piecewise-
//! parabolic (P²) update — falling back to linear interpolation whenever the
//! parabola would carry the height past a neighbor and break monotonicity.
//!
//! Accuracy is more than the guard needs: after a few thousand samples the
//! estimate sits within a few percent of the exact quantile (the tests
//! below measure this against exact sorts), and the watermark carries its
//! own fragmentation slack on top, so estimator error is not the binding
//! term.

/// Online estimator for a single quantile `q` in (0, 1).
///
/// Contract: `observe` accepts any finite `f64`. NaN inputs are **ignored**
/// rather than poisoning the markers — a NaN here would come from a broken
/// telemetry read, and one bad poll must not corrupt days of state. Ignored
/// NaNs are counted separately in [`P2Quantile::nan_ignored`] so the caller
/// can notice a telemetry source going bad. Infinities are accepted (they
/// sort), but a caller feeding them should expect the extreme markers to
/// pin there.
#[derive(Debug, Clone)]
pub struct P2Quantile {
    /// The target quantile, fixed at construction.
    q: f64,
    /// Marker heights q_1..q_5: min, lower mid, target, upper mid, max.
    heights: [f64; 5],
    /// Actual marker positions n_1..n_5 (1-based sample counts). Kept as
    /// f64 for the update arithmetic but always integral, and exact: they
    /// grow by 1 per sample, far below f64's 2^53 integer horizon.
    positions: [f64; 5],
    /// Desired positions n'_1..n'_5.
    desired: [f64; 5],
    /// Per-sample increments of the desired positions.
    increments: [f64; 5],
    /// Samples accepted so far (NaNs excluded).
    count: u64,
    /// NaN inputs ignored so far.
    nan_count: u64,
}

impl P2Quantile {
    /// Build an estimator for quantile `q`.
    ///
    /// # Panics
    /// If `q` is not strictly inside (0, 1) — a quantile of 0 or 1 is just
    /// the running min/max and does not need this machinery.
    #[must_use]
    pub fn new(q: f64) -> Self {
        assert!(
            q > 0.0 && q < 1.0,
            "P2Quantile: q must be in (0,1), got {q}"
        );
        P2Quantile {
            q,
            heights: [0.0; 5],
            positions: [1.0, 2.0, 3.0, 4.0, 5.0],
            desired: [1.0, 1.0 + 2.0 * q, 1.0 + 4.0 * q, 3.0 + 2.0 * q, 5.0],
            increments: [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0],
            count: 0,
            nan_count: 0,
        }
    }

    /// The quantile this estimator tracks.
    #[must_use]
    pub fn q(&self) -> f64 {
        self.q
    }

    /// Samples accepted so far (NaNs not included).
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// NaN inputs ignored so far. Non-zero means the telemetry source fed
    /// garbage at least once; the estimate itself is unaffected.
    #[must_use]
    pub fn nan_ignored(&self) -> u64 {
        self.nan_count
    }

    /// Current estimate of the `q`-quantile, or `None` before 5 samples —
    /// with fewer than five points the marker structure does not exist yet,
    /// and pretending otherwise would hand the caller a number with no
    /// distributional meaning.
    #[must_use]
    pub fn estimate(&self) -> Option<f64> {
        if self.count < 5 {
            None
        } else {
            Some(self.heights[2])
        }
    }

    /// Feed one observation. NaN is ignored (see the type-level contract).
    pub fn observe(&mut self, x: f64) {
        if x.is_nan() {
            self.nan_count += 1;
            return;
        }

        // The first five samples ARE the markers: store, and once all five
        // exist, sort them into initial heights. (`heights` doubles as the
        // holding buffer — positions/desired stay at their constructed
        // values, which are exactly the paper's initial values.)
        if self.count < 5 {
            self.heights[self.count as usize] = x;
            self.count += 1;
            if self.count == 5 {
                self.heights.sort_by(f64::total_cmp);
            }
            return;
        }
        self.count += 1;

        // 1. Find the cell k (0-indexed: heights[k] <= x < heights[k+1]),
        //    extending the extreme markers when x falls outside them.
        let k = if x < self.heights[0] {
            self.heights[0] = x;
            0
        } else if x >= self.heights[4] {
            self.heights[4] = x;
            3
        } else {
            // x is inside [heights[0], heights[4]); walk to its cell.
            let mut k = 0;
            for i in (0..4).rev() {
                if self.heights[i] <= x {
                    k = i;
                    break;
                }
            }
            k
        };

        // 2. Every marker above the cell has one more sample at or below
        //    it now; 3. every desired position advances by its increment.
        for i in (k + 1)..5 {
            self.positions[i] += 1.0;
        }
        for i in 0..5 {
            self.desired[i] += self.increments[i];
        }

        // 4. Nudge the three interior markers toward their desired
        //    positions, at most one step each, keeping heights monotone.
        for i in 1..=3 {
            let d = self.desired[i] - self.positions[i];
            let room_up = self.positions[i + 1] - self.positions[i] > 1.0;
            let room_down = self.positions[i - 1] - self.positions[i] < -1.0;
            if (d >= 1.0 && room_up) || (d <= -1.0 && room_down) {
                let d = d.signum(); // +-1.0

                // Piecewise-parabolic: fit a parabola through the marker
                // and its neighbors, evaluate one position over.
                let parabolic = self.heights[i]
                    + d / (self.positions[i + 1] - self.positions[i - 1])
                        * ((self.positions[i] - self.positions[i - 1] + d)
                            * (self.heights[i + 1] - self.heights[i])
                            / (self.positions[i + 1] - self.positions[i])
                            + (self.positions[i + 1] - self.positions[i] - d)
                                * (self.heights[i] - self.heights[i - 1])
                                / (self.positions[i] - self.positions[i - 1]));

                // The parabola may overshoot a neighbor when the local
                // density is lopsided; heights must stay strictly ordered
                // or the estimator stops being a quantile at all, so fall
                // back to linear interpolation toward the neighbor in the
                // direction of travel.
                if self.heights[i - 1] < parabolic && parabolic < self.heights[i + 1] {
                    self.heights[i] = parabolic;
                } else {
                    let j = if d > 0.0 { i + 1 } else { i - 1 };
                    self.heights[i] += d * (self.heights[j] - self.heights[i])
                        / (self.positions[j] - self.positions[i]);
                }
                self.positions[i] += d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal PCG32 (O'Neill) — this crate is deliberately dependency-free,
    /// so the tests carry their own 10-line generator. Deterministic by
    /// construction: same seed, same stream, no flake surface.
    struct Pcg(u64);
    impl Pcg {
        fn new(seed: u64) -> Self {
            Pcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xDA3E_39CB_94B9_5BDB))
        }
        fn next_u32(&mut self) -> u32 {
            let old = self.0;
            self.0 = old
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
            let rot = (old >> 59) as u32;
            xorshifted.rotate_right(rot)
        }
        /// Uniform in [0, 1).
        fn uniform(&mut self) -> f64 {
            f64::from(self.next_u32()) / (f64::from(u32::MAX) + 1.0)
        }
        /// Roughly standard normal: Irwin-Hall, sum of 12 uniforms minus 6
        /// (mean 0, variance 1, symmetric, thin tails past +-6).
        fn normalish(&mut self) -> f64 {
            (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
        }
    }

    /// Exact empirical quantile by nearest rank on a sorted copy.
    fn exact_quantile(sorted: &[f64], q: f64) -> f64 {
        let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[idx]
    }

    /// Feed a stream to P2 and compare against the exact sorted quantile.
    ///
    /// Tolerance: 5% of the IQR plus 2% of |exact|. The IQR term is the
    /// scale-free "body" tolerance — published P2 results and these fixed
    /// streams both land well inside a few percent of the distribution's
    /// spread at n = 10_000. The relative term matters only for far-tail
    /// quantiles of the heavy-tailed stream, where the local sample spacing
    /// is many IQRs wide and expecting IQR-scale accuracy from ~10 tail
    /// samples per marker step would test the sample, not the estimator.
    /// The streams are seeded, so this bound is checked, not hoped for.
    fn assert_close(stream: &[f64], q: f64) {
        let mut est = P2Quantile::new(q);
        for &x in stream {
            est.observe(x);
        }
        let mut sorted = stream.to_vec();
        sorted.sort_by(f64::total_cmp);
        let exact = exact_quantile(&sorted, q);
        let iqr = exact_quantile(&sorted, 0.75) - exact_quantile(&sorted, 0.25);
        let tol = 0.05 * iqr + 0.02 * exact.abs();
        let got = est.estimate().expect("10k samples in, estimate must exist");
        assert!(
            (got - exact).abs() <= tol,
            "q={q}: p2={got}, exact={exact}, tol={tol}"
        );
    }

    #[test]
    fn tracks_uniform_quantiles() {
        let mut r = Pcg::new(1);
        let stream: Vec<f64> = (0..10_000).map(|_| r.uniform()).collect();
        for q in [0.1, 0.5, 0.9, 0.99] {
            assert_close(&stream, q);
        }
    }

    #[test]
    fn tracks_normalish_quantiles() {
        let mut r = Pcg::new(2);
        let stream: Vec<f64> = (0..10_000).map(|_| r.normalish()).collect();
        for q in [0.1, 0.5, 0.9, 0.99] {
            assert_close(&stream, q);
        }
    }

    /// Heavy tail — exp of a normal-ish variate (lognormal-ish). This is
    /// the shape ambient VRAM actually has: a body around the desktop's
    /// steady state and rare large spikes, and it is where a naive
    /// mean+k*sigma guard fails hardest.
    #[test]
    fn tracks_heavy_tail_quantiles() {
        let mut r = Pcg::new(3);
        let stream: Vec<f64> = (0..10_000).map(|_| r.normalish().exp()).collect();
        for q in [0.5, 0.9, 0.99] {
            assert_close(&stream, q);
        }
    }

    /// On a symmetric stream the median estimator must settle near the
    /// center of symmetry — the one quantile with an obvious ground truth.
    #[test]
    fn median_of_symmetric_stream_converges_to_center() {
        let mut r = Pcg::new(4);
        let mut est = P2Quantile::new(0.5);
        for _ in 0..10_000 {
            est.observe(r.normalish());
        }
        let m = est.estimate().unwrap();
        assert!(m.abs() < 0.05, "median of symmetric stream drifted: {m}");
    }

    /// Two estimators on the SAME stream must order by their quantile —
    /// the watermark's whole premise is that the 0.999 line sits above the
    /// median, and an estimator that inverted them would invert the guard.
    #[test]
    fn estimates_are_monotone_in_q() {
        let mut r = Pcg::new(5);
        let mut lo = P2Quantile::new(0.5);
        let mut hi = P2Quantile::new(0.9);
        for _ in 0..10_000 {
            let x = r.uniform();
            lo.observe(x);
            hi.observe(x);
        }
        let (l, h) = (lo.estimate().unwrap(), hi.estimate().unwrap());
        assert!(h >= l, "q=0.9 estimate {h} fell below q=0.5 estimate {l}");
    }

    /// Before five samples there is no marker structure and no estimate;
    /// at exactly five, the estimate is the middle of the sorted five.
    #[test]
    fn no_estimate_before_five_samples_then_exact_median_of_five() {
        let mut est = P2Quantile::new(0.5);
        for x in [5.0, 1.0, 4.0, 2.0] {
            est.observe(x);
            assert_eq!(est.estimate(), None);
        }
        est.observe(3.0);
        assert_eq!(est.count(), 5);
        assert_eq!(est.estimate(), Some(3.0));
    }

    /// A NaN poll is ignored and counted, never mixed into the markers.
    #[test]
    fn nan_is_ignored_and_counted() {
        let mut est = P2Quantile::new(0.9);
        for i in 0..100 {
            est.observe(f64::from(i));
        }
        let before = est.estimate().unwrap();
        est.observe(f64::NAN);
        assert_eq!(est.nan_ignored(), 1);
        assert_eq!(est.count(), 100, "NaN must not count as a sample");
        assert_eq!(est.estimate().unwrap(), before, "NaN must not move markers");
    }

    /// A constant stream is a legal (if dull) distribution: every quantile
    /// is the constant, and the divided differences must not blow up.
    #[test]
    fn constant_stream_is_stable() {
        let mut est = P2Quantile::new(0.999);
        for _ in 0..1000 {
            est.observe(42.0);
        }
        assert_eq!(est.estimate(), Some(42.0));
    }
}
