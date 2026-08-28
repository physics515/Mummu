//! **The chance-constrained VRAM guard: Pr[usage + ambient > VRAM] <= alpha.**
//!
//! What this replaces: placement took ONE NVML free-VRAM sample at model
//! load, subtracted a fixed 2 GiB desktop reserve, and trusted the result
//! for the life of the process. Ambient consumption — everything else on
//! the card: compositor, browsers, capture tools — then drifted from 1.9 to
//! 8.9 GiB inside a single session, and the next load silently moved the
//! plan from 42 GPU layers to 20. One sample is not a budget. A fixed
//! reserve is not a budget either: it is simultaneously too big on a quiet
//! box (layers left on the host for nothing) and too small under a spike
//! (allocation failure mid-decode).
//!
//! The budget is a *distribution*. The guard tracks the (1 - alpha)
//! quantile of observed ambient bytes with a [`crate::p2`] estimator, adds
//! a fragmentation slack (a quantile bounds what the driver reports, not
//! what a contiguous allocation can actually get), and clamps to a floor.
//! Reserving that many bytes makes the residency plan satisfy
//! Pr[usage + ambient > VRAM] <= alpha by construction, up to estimator
//! error — which is why the slack and the breach correction below exist.
//!
//! Asymmetric dynamics, on purpose:
//! - **Up immediately.** A single sample above the guard is proof the guard
//!   under-covers *right now*, so the guard jumps to cover it (an envelope,
//!   not a mean). Reacting slowly in this direction is how the 8.9 GiB
//!   session ended in allocation failures.
//! - **Down slowly.** The guard relaxes to quantile + slack only after
//!   `hysteresis_window` consecutive quiet samples. Shrinking eagerly would
//!   let one calm second between spikes re-promote layers that are about to
//!   be evicted again — plan churn costs seconds per re-placement, while
//!   holding a too-high guard costs only a few layers of residency.
//! - **Breaches are posterior evidence.** An actual allocation failure
//!   means the polled distribution missed real peaks (NVML sampling is
//!   sparse; spikes between polls are invisible). Each breach permanently
//!   multiplies the quantile term by [`BREACH_BOOST`] — the estimator said
//!   alpha, reality said more, so correct the estimate, don't just retry.

use crate::p2::P2Quantile;

/// Multiplier applied per recorded breach (allocation failure): the
/// posterior correction. 1.5 is deliberately coarse — a breach is a rare,
/// high-information event, and under-reacting to one repeats it, while
/// over-reacting costs a few layers of residency until the session ends.
/// Two breaches more than double the quantile term; the boost never decays
/// within a session because the evidence never un-happens.
pub const BREACH_BOOST: f64 = 1.5;

/// Tuning for [`Watermark`]. `Default` gives the drop-in replacement for
/// the guards it removes: the old fixed 2 GiB desktop reserve is demoted to
/// `floor_bytes` (a lower bound, no longer the whole story), and the slack
/// covers allocator fragmentation between "bytes reported free" and "bytes
/// a real allocation can get".
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkConfig {
    /// Acceptable probability of ambient exceeding the guard. The tracked
    /// quantile is 1 - alpha. Must be strictly inside (0, 1).
    pub alpha: f64,
    /// Consecutive samples below the guard required before it may shrink.
    pub hysteresis_window: u32,
    /// The guard never reports below this, no matter how quiet the box —
    /// the desktop can always wake up.
    pub floor_bytes: u64,
    /// Added on top of the quantile estimate: fragmentation and estimator
    /// slack. This is also what makes the guard strictly exceed the most
    /// recent spike rather than merely equal it.
    pub frag_slack_bytes: u64,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        WatermarkConfig {
            alpha: 1e-3,
            hysteresis_window: 32,
            floor_bytes: 2 * 1024 * 1024 * 1024,
            frag_slack_bytes: 128 * 1024 * 1024,
        }
    }
}

/// The guard itself. Feed it every ambient-VRAM poll via
/// [`Watermark::observe_ambient`]; read the bytes to reserve via
/// [`Watermark::guard_bytes`]; report real allocation failures via
/// [`Watermark::breach`].
///
/// "Ambient" means bytes used on the device by everything that is not this
/// process: total VRAM minus free minus our own tracked usage, from
/// whatever telemetry the serve crate polls.
#[derive(Debug, Clone)]
pub struct Watermark {
    cfg: WatermarkConfig,
    /// Online (1 - alpha) quantile of ambient bytes.
    quantile: P2Quantile,
    /// Largest ambient ever seen — the conservative stand-in while the
    /// quantile estimator has fewer than five samples. For alpha = 1e-3
    /// the 0.999 quantile of a five-sample history IS essentially the max,
    /// so this is the same estimate, just honest about its sample size.
    max_seen: u64,
    /// The published guard.
    guard: u64,
    /// Consecutive samples below the guard since it last moved up.
    quiet: u32,
    /// Allocation failures reported so far.
    breaches: u32,
}

impl Watermark {
    /// # Panics
    /// If `cfg.alpha` is not strictly inside (0, 1) — the tracked quantile
    /// 1 - alpha must itself be a proper quantile.
    #[must_use]
    pub fn new(cfg: WatermarkConfig) -> Self {
        let quantile = P2Quantile::new(1.0 - cfg.alpha);
        let guard = cfg.floor_bytes.max(cfg.frag_slack_bytes);
        Watermark {
            cfg,
            quantile,
            max_seen: 0,
            guard,
            quiet: 0,
            breaches: 0,
        }
    }

    /// Bytes to keep free for everyone else. Monotone responses only: this
    /// number moved up the instant evidence demanded it, and down only
    /// after a full hysteresis window of quiet.
    #[must_use]
    pub fn guard_bytes(&self) -> u64 {
        self.guard
    }

    /// Allocation failures reported via [`Watermark::breach`] so far.
    #[must_use]
    pub fn breaches(&self) -> u32 {
        self.breaches
    }

    /// Ambient samples observed so far.
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.quantile.count()
    }

    /// Feed one ambient-usage poll (bytes used by everything that is not
    /// this process).
    pub fn observe_ambient(&mut self, bytes: u64) {
        self.max_seen = self.max_seen.max(bytes);
        self.quantile.observe(bytes as f64);

        // Up immediately: the guard must cover both the steady-state
        // quantile and the sample we are looking at right now.
        let up = self
            .settle_target()
            .max(bytes.saturating_add(self.cfg.frag_slack_bytes))
            .max(self.cfg.floor_bytes);
        if up > self.guard {
            self.guard = up;
            self.quiet = 0;
        } else if bytes < self.guard {
            // Down slowly: only a full window of consecutive quiet samples
            // earns a shrink, and the shrink lands on the quantile target,
            // never below the floor.
            self.quiet += 1;
            if self.quiet >= self.cfg.hysteresis_window {
                let down = self.settle_target();
                if down < self.guard {
                    self.guard = down;
                }
                self.quiet = 0;
            }
        } else {
            // At-or-above the guard without forcing a rise (possible only
            // with zero slack): not quiet, so the countdown restarts.
            self.quiet = 0;
        }
    }

    /// Record a real allocation failure: the guard was charged and reality
    /// still said no. Bumps the breach counter and forces the guard up by
    /// at least [`BREACH_BOOST`] — see the module doc for why this is a
    /// posterior correction and not a retry policy.
    ///
    /// (Degenerate corner: with a zero guard — zero floor, zero slack, no
    /// samples — multiplying zero cannot rise; callers get sane behavior
    /// here by setting a floor, which the default config does.)
    pub fn breach(&mut self) {
        self.breaches = self.breaches.saturating_add(1);
        let scaled = to_bytes_saturating(self.guard as f64 * BREACH_BOOST);
        self.guard = self.guard.max(self.settle_target()).max(scaled);
        self.quiet = 0;
    }

    /// The steady-state target the guard relaxes to: boosted quantile
    /// estimate + fragmentation slack, floor-clamped.
    fn settle_target(&self) -> u64 {
        to_bytes_saturating(self.q_term())
            .saturating_add(self.cfg.frag_slack_bytes)
            .max(self.cfg.floor_bytes)
    }

    /// The (1 - alpha) quantile term with the breach boost applied. Falls
    /// back to the running max while the estimator is still warming up.
    fn q_term(&self) -> f64 {
        let q = self
            .quantile
            .estimate()
            .unwrap_or(self.max_seen as f64)
            .max(0.0);
        // powi capped so the boost stays finite; the guard saturates at
        // u64::MAX long before 1.5^64 matters.
        q * BREACH_BOOST.powi(self.breaches.min(64) as i32)
    }
}

/// f64 -> bytes, rounding up (a guard rounds against itself) and
/// saturating at the ends instead of invoking float-cast UB corners.
fn to_bytes_saturating(x: f64) -> u64 {
    if x >= u64::MAX as f64 {
        u64::MAX
    } else if x <= 0.0 {
        0
    } else {
        x.ceil() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    /// Same 10-line PCG32 as in `p2::tests` — duplicated on purpose so each
    /// module's tests stay self-contained in a dependency-free crate.
    struct Pcg(u64);
    impl Pcg {
        fn new(seed: u64) -> Self {
            Pcg(seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0xDA3E_39CB_94B9_5BDB))
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
        fn uniform(&mut self) -> f64 {
            f64::from(self.next_u32()) / (f64::from(u32::MAX) + 1.0)
        }
    }

    /// A config where the quantile term is visible in tests: alpha = 0.5
    /// tracks the MEDIAN, so a single spike moves the settle target barely
    /// at all and the shrink after the hysteresis window is observable.
    /// (With the production alpha = 1e-3, the 0.999 quantile of a short
    /// stream IS the spike, and there is nothing to shrink back to.)
    fn median_cfg(window: u32) -> WatermarkConfig {
        WatermarkConfig {
            alpha: 0.5,
            hysteresis_window: window,
            floor_bytes: 0,
            frag_slack_bytes: 0,
        }
    }

    /// The failure this module exists for: ambient spikes, the guard must
    /// cover it on THAT sample, not after the estimator catches up.
    #[test]
    fn guard_rises_immediately_on_a_spike() {
        let mut wm = Watermark::new(median_cfg(8));
        for _ in 0..100 {
            wm.observe_ambient(1000);
        }
        assert!(wm.guard_bytes() <= 1100, "settled near the base level");
        wm.observe_ambient(5000);
        assert!(
            wm.guard_bytes() >= 5000,
            "one spike sample must lift the guard past it, got {}",
            wm.guard_bytes()
        );
    }

    /// The guard shrinks only after W consecutive quiet samples — W-1 calm
    /// samples must leave it exactly where the spike put it.
    #[test]
    fn guard_shrinks_only_after_a_full_quiet_window() {
        let w = 8;
        let mut wm = Watermark::new(median_cfg(w));
        for _ in 0..100 {
            wm.observe_ambient(1000);
        }
        wm.observe_ambient(5000);
        let spiked = wm.guard_bytes();
        assert!(spiked >= 5000);

        for i in 0..(w - 1) {
            wm.observe_ambient(900);
            assert_eq!(
                wm.guard_bytes(),
                spiked,
                "guard moved after only {} quiet samples",
                i + 1
            );
        }
        wm.observe_ambient(900);
        assert!(
            wm.guard_bytes() < spiked,
            "a full quiet window must let the guard settle"
        );
        assert!(
            wm.guard_bytes() < 2000,
            "settled guard should be near the median, got {}",
            wm.guard_bytes()
        );
    }

    /// The floor holds regardless of how quiet the stream is.
    #[test]
    fn floor_is_respected() {
        let cfg = WatermarkConfig {
            alpha: 0.5,
            hysteresis_window: 4,
            floor_bytes: 10_000,
            frag_slack_bytes: 0,
        };
        let mut wm = Watermark::new(cfg);
        assert_eq!(wm.guard_bytes(), 10_000, "floor from the very start");
        for _ in 0..200 {
            wm.observe_ambient(50);
            assert!(wm.guard_bytes() >= 10_000);
        }
        assert_eq!(wm.guard_bytes(), 10_000);
    }

    /// A breach is posterior evidence: the guard must rise by the boost
    /// even though no observed sample justified it, and the boosted
    /// quantile term must persist — after a quiet window the guard settles
    /// ABOVE the pre-breach level, not back onto it.
    #[test]
    fn breach_forces_a_rise_and_the_boost_persists() {
        let w = 8;
        let mut wm = Watermark::new(median_cfg(w));
        for _ in 0..100 {
            wm.observe_ambient(1000);
        }
        let before = wm.guard_bytes();
        wm.breach();
        assert_eq!(wm.breaches(), 1);
        assert!(
            wm.guard_bytes() as f64 >= before as f64 * BREACH_BOOST,
            "breach must scale the guard: {} -> {}",
            before,
            wm.guard_bytes()
        );

        // Let it settle: the boosted quantile term keeps it elevated.
        for _ in 0..(4 * w) {
            wm.observe_ambient(1000);
        }
        assert!(
            wm.guard_bytes() > before,
            "the posterior correction must not wash out: settled at {}, pre-breach {}",
            wm.guard_bytes(),
            before
        );
    }

    /// The contract itself, empirically: with alpha = 1e-3 over a spiky
    /// synthetic ambient stream (the 1.9 -> 8.9 GiB session, statistically),
    /// the violation rate of `ambient > guard` after warmup stays within a
    /// small factor of alpha. The bound is 10x alpha — generous on purpose
    /// (the estimator, the envelope, and the slack all push the true rate
    /// far below it) and the stream is seeded, so this never flakes.
    #[test]
    fn violation_rate_stays_near_alpha() {
        let cfg = WatermarkConfig {
            alpha: 1e-3,
            hysteresis_window: 32,
            floor_bytes: 2 * GIB,
            frag_slack_bytes: 128 * MIB,
        };
        let mut wm = Watermark::new(cfg);
        let mut r = Pcg::new(7);

        // Base desktop load ~2 GiB, jitter up to 256 MiB, and with 2%
        // probability a spike of up to 4 GiB on top (browser tab, game
        // launcher, capture tool) — the observed drift pattern.
        let sample = |r: &mut Pcg| -> u64 {
            let base = 2 * GIB;
            let noise = (r.uniform() * 256.0 * MIB as f64) as u64;
            let spike = if r.uniform() < 0.02 {
                (r.uniform() * 4.0 * GIB as f64) as u64
            } else {
                0
            };
            base + noise + spike
        };

        for _ in 0..3000 {
            wm.observe_ambient(sample(&mut r));
        }

        let n = 10_000;
        let mut violations = 0u32;
        for _ in 0..n {
            let x = sample(&mut r);
            // Judge the guard that stood BEFORE this sample arrived — the
            // number an allocation in flight would actually have trusted.
            if x > wm.guard_bytes() {
                violations += 1;
            }
            wm.observe_ambient(x);
            assert!(wm.guard_bytes() >= 2 * GIB, "floor must hold throughout");
        }

        let rate = f64::from(violations) / f64::from(n);
        assert!(
            rate <= 10.0 * 1e-3,
            "violation rate {rate} exceeds 10x alpha ({violations}/{n})"
        );
    }
}
