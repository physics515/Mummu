//! **Adaptive placement** — keep the device/host split earning its keep as
//! the machine changes underneath it.
//!
//! A placement decided once at load is a guess about a machine that will not
//! stay still: a game starts, another model loads, the desktop compositor
//! grows, a background job eats host RAM. The split that was optimal at load
//! becomes an out-of-memory crash or an idle GPU an hour later.
//!
//! # Why control on *effects*, not causes
//!
//! The tempting design is to introspect every cause — free VRAM, other
//! processes' usage, GPU utilization. That needs a different API per OS and
//! per vendor (DXGI `QueryVideoMemoryInfo`, NVML, sysfs), each of which can
//! be missing, stale, or lie about what a driver will actually hand out.
//!
//! This controller instead watches what actually matters and is always
//! observable: **did an allocation fail**, and **did throughput move**. Those
//! two signals capture every cause, including ones no query would reveal
//! (thermal throttling, a driver reserving memory, another process's spike).
//! Direct readings — host memory available, our own resident bytes — are used
//! where they are cheap, as *hints* that bound the search rather than as the
//! decision.
//!
//! # AIMD, and why that shape
//!
//! The device budget moves by **additive increase, multiplicative decrease**:
//! grow slowly while things are fine, cut hard the moment they are not. This
//! is the shape TCP congestion control settled on for the same problem —
//! a shared resource with an unknown, moving limit where overshoot is far
//! more expensive than undershoot. Here overshoot means an OOM that kills a
//! generation (or, as measured on this project, the whole process); undershoot
//! only means some layers run on the CPU. The asymmetry in cost justifies the
//! asymmetry in response.
//!
//! Two damping rules keep it from thrashing:
//!
//! - a **dwell time** — no change until the current placement has been
//!   observed long enough to judge it;
//! - a **deadband** — throughput must move by more than measurement noise
//!   before it counts as evidence.
//!
//! Moves are applied in bounded batches by the caller (see
//! `ExpertPool::apply_schedule`), so adapting never stalls a generation.

use std::time::{Duration, Instant};

/// What the controller learned from one observation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjust {
    /// Budget unchanged — either nothing moved enough to act on, or the
    /// placement has not been observed long enough yet.
    Hold,
    /// Grow the device budget to this many bytes (additive increase).
    Grow(u64),
    /// Shrink to this many bytes (multiplicative decrease).
    Shrink(u64),
}

/// One observation of how the current placement is doing.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Decode throughput since the last sample. The signal that matters.
    pub tokens_per_sec: f64,
    /// Did anything fail to allocate on the device since the last sample?
    /// Overrides everything else: a placement that cannot allocate is not a
    /// placement, however fast it looked.
    pub device_alloc_failed: bool,
    /// Host memory currently available, when the OS will say. `None` means
    /// "unknown", never "plenty".
    pub host_available_bytes: Option<u64>,
    /// Bytes the model currently holds on the device.
    pub device_bytes_in_use: u64,
}

/// Tunables. The defaults are deliberately cautious: this controller runs
/// unattended against a machine that other software is also using.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// Never plan below this device budget — under it, the device is not
    /// worth the cross-device traffic.
    pub floor_bytes: u64,
    /// Never plan above this (the hardware bound, minus whatever headroom
    /// the caller wants to leave the rest of the system).
    pub ceiling_bytes: u64,
    /// Additive increase step.
    pub grow_step_bytes: u64,
    /// Multiplicative decrease factor, in (0, 1).
    pub shrink_factor: f64,
    /// How long a placement must be observed before it is judged.
    pub dwell: Duration,
    /// Relative throughput change below which a sample is treated as noise.
    pub deadband: f64,
}

impl Policy {
    /// A policy for a device of `total_bytes`, leaving `reserve_bytes` to the
    /// rest of the system (desktop compositor, other processes).
    #[must_use]
    pub fn for_device(total_bytes: u64, reserve_bytes: u64) -> Self {
        let ceiling = total_bytes.saturating_sub(reserve_bytes);
        Self {
            floor_bytes: (total_bytes / 8).min(ceiling),
            ceiling_bytes: ceiling,
            // ~6% of the ceiling per step: a dozen good windows to go from
            // floor to ceiling, which is slow enough to notice a mistake.
            grow_step_bytes: (ceiling / 16).max(1),
            shrink_factor: 0.75,
            dwell: Duration::from_secs(30),
            deadband: 0.05,
        }
    }
}

/// The controller. Feed it [`Sample`]s; it answers with an [`Adjust`].
#[derive(Debug)]
pub struct Controller {
    policy: Policy,
    budget: u64,
    /// Best throughput seen, and the budget that produced it — what "did this
    /// change help?" is judged against.
    best: Option<(f64, u64)>,
    last_change: Instant,
    /// Budgets that produced an allocation failure. Never grow back into one
    /// blindly: the ceiling that actually matters is the one the machine
    /// enforced, not the one the spec sheet advertises.
    failed_at: Option<u64>,
}

impl Controller {
    #[must_use]
    pub fn new(policy: Policy, initial_budget: u64) -> Self {
        Self {
            budget: initial_budget.clamp(policy.floor_bytes, policy.ceiling_bytes),
            policy,
            best: None,
            last_change: Instant::now(),
            failed_at: None,
        }
    }

    /// The device budget the placement planner should use right now.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Feed one observation. `now` is injected so the logic is testable
    /// without sleeping.
    pub fn observe(&mut self, sample: &Sample, now: Instant) -> Adjust {
        // 1. An allocation failure is not a data point to weigh against
        //    throughput — it is a hard ceiling discovery. Act immediately,
        //    ignoring dwell: staying here risks the next OOM.
        if sample.device_alloc_failed {
            let ceiling = sample.device_bytes_in_use.min(self.budget);
            self.failed_at = Some(match self.failed_at {
                Some(prev) => prev.min(ceiling),
                None => ceiling,
            });
            let next =
                ((ceiling as f64 * self.policy.shrink_factor) as u64).max(self.policy.floor_bytes);
            self.last_change = now;
            self.best = None; // the old best was measured under a limit that no longer holds
            if next < self.budget {
                self.budget = next;
                return Adjust::Shrink(next);
            }
            return Adjust::Hold;
        }

        // 2. Judge a placement only once it has been observed long enough.
        if now.duration_since(self.last_change) < self.policy.dwell {
            return Adjust::Hold;
        }

        // 3. Host pressure: if the OS says memory is short, the CPU side of
        //    the split is the problem, so pulling MORE onto the device helps.
        //    (Unknown is not permission — `None` does nothing.)
        let host_pressure = sample
            .host_available_bytes
            .is_some_and(|avail| avail < self.policy.grow_step_bytes * 2);

        let improved = match self.best {
            None => true,
            Some((best_tps, _)) => {
                let delta = (sample.tokens_per_sec - best_tps) / best_tps.max(1e-9);
                delta > self.policy.deadband
            }
        };
        let regressed = match self.best {
            None => false,
            Some((best_tps, best_budget)) => {
                let delta = (best_tps - sample.tokens_per_sec) / best_tps.max(1e-9);
                delta > self.policy.deadband && self.budget != best_budget
            }
        };

        if improved {
            self.best = Some((sample.tokens_per_sec, self.budget));
        }

        // 4. A regression means the last move was wrong: go back to what was
        //    measurably better rather than continuing to explore.
        if regressed && let Some((_, best_budget)) = self.best {
            self.last_change = now;
            let next = best_budget.clamp(self.policy.floor_bytes, self.policy.ceiling_bytes);
            if next != self.budget {
                self.budget = next;
                return if next > self.budget {
                    Adjust::Grow(next)
                } else {
                    Adjust::Shrink(next)
                };
            }
            return Adjust::Hold;
        }

        // 5. Otherwise creep upward, but never back into a budget the machine
        //    already refused, and never past the ceiling.
        let hard_ceiling = self
            .failed_at
            .map_or(self.policy.ceiling_bytes, |f| {
                // Stay a decrease-step below the level that failed.
                ((f as f64 * self.policy.shrink_factor) as u64).max(self.policy.floor_bytes)
            })
            .min(self.policy.ceiling_bytes);

        let want = self.budget.saturating_add(self.policy.grow_step_bytes);
        if (improved || host_pressure) && want <= hard_ceiling {
            self.last_change = now;
            self.budget = want;
            return Adjust::Grow(want);
        }
        Adjust::Hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy {
            floor_bytes: 1_000,
            ceiling_bytes: 10_000,
            grow_step_bytes: 1_000,
            shrink_factor: 0.5,
            dwell: Duration::from_secs(10),
            deadband: 0.05,
        }
    }

    fn sample(tps: f64) -> Sample {
        Sample {
            tokens_per_sec: tps,
            device_alloc_failed: false,
            host_available_bytes: None,
            device_bytes_in_use: 5_000,
        }
    }

    #[test]
    fn an_allocation_failure_shrinks_immediately_ignoring_dwell() {
        let mut c = Controller::new(policy(), 8_000);
        let now = Instant::now();
        // No dwell has passed at all — a hard limit must not wait for one.
        let got = c.observe(
            &Sample {
                device_alloc_failed: true,
                device_bytes_in_use: 8_000,
                ..sample(10.0)
            },
            now,
        );
        assert_eq!(got, Adjust::Shrink(4_000));
        assert_eq!(c.budget(), 4_000);
    }

    #[test]
    fn it_never_grows_back_into_a_budget_the_machine_refused() {
        let mut c = Controller::new(policy(), 8_000);
        let t0 = Instant::now();
        c.observe(
            &Sample {
                device_alloc_failed: true,
                device_bytes_in_use: 8_000,
                ..sample(10.0)
            },
            t0,
        );
        // Now improve repeatedly for a long time; it may creep, but never to
        // the level that failed.
        let mut t = t0;
        let mut tps = 10.0;
        for _ in 0..20 {
            t += Duration::from_secs(11);
            tps *= 1.5;
            c.observe(&sample(tps), t);
        }
        assert!(
            c.budget() < 8_000,
            "must stay below the refused budget, got {}",
            c.budget()
        );
    }

    #[test]
    fn it_holds_until_the_placement_has_been_observed_long_enough() {
        let mut c = Controller::new(policy(), 5_000);
        let t = Instant::now();
        assert_eq!(c.observe(&sample(10.0), t), Adjust::Hold);
        assert_eq!(
            c.observe(&sample(99.0), t + Duration::from_secs(5)),
            Adjust::Hold
        );
        // Past the dwell, an improvement is actionable.
        assert_eq!(
            c.observe(&sample(99.0), t + Duration::from_secs(11)),
            Adjust::Grow(6_000)
        );
    }

    #[test]
    fn noise_within_the_deadband_does_not_move_the_budget() {
        let mut c = Controller::new(policy(), 5_000);
        let t = Instant::now();
        c.observe(&sample(10.0), t + Duration::from_secs(11)); // sets a baseline, grows
        let after_first = c.budget();
        // A 2% wobble is not evidence of anything.
        let got = c.observe(&sample(10.2), t + Duration::from_secs(30));
        assert_eq!(got, Adjust::Hold, "2% is inside the deadband");
        assert_eq!(c.budget(), after_first);
    }

    #[test]
    fn a_regression_returns_to_the_budget_that_measured_best() {
        let mut c = Controller::new(policy(), 5_000);
        let t = Instant::now();
        // Establish a good result at 5_000, then grow to 6_000.
        assert_eq!(
            c.observe(&sample(20.0), t + Duration::from_secs(11)),
            Adjust::Grow(6_000)
        );
        // 6_000 turns out much worse -> go back to what was measurably better.
        let got = c.observe(&sample(5.0), t + Duration::from_secs(30));
        assert!(
            matches!(got, Adjust::Shrink(_) | Adjust::Hold),
            "a regression must not keep growing: {got:?}"
        );
        assert!(c.budget() <= 6_000);
    }

    #[test]
    fn host_memory_pressure_pulls_work_onto_the_device() {
        let mut c = Controller::new(policy(), 5_000);
        let t = Instant::now();
        c.observe(&sample(10.0), t + Duration::from_secs(11));
        let before = c.budget();
        // Throughput flat (no improvement), but the host is nearly out of
        // memory — moving more onto the device is the relief valve.
        let got = c.observe(
            &Sample {
                host_available_bytes: Some(100),
                ..sample(10.0)
            },
            t + Duration::from_secs(40),
        );
        assert!(
            matches!(got, Adjust::Grow(_)),
            "host pressure should pull work to the device: {got:?}"
        );
        assert!(c.budget() > before);
    }

    #[test]
    fn unknown_host_memory_is_not_treated_as_plenty() {
        // `None` must behave like "no information", not like "lots free" —
        // the difference decides whether an unsupported OS silently gets the
        // aggressive path.
        let mut c = Controller::new(policy(), 5_000);
        let t = Instant::now();
        c.observe(&sample(10.0), t + Duration::from_secs(11));
        let before = c.budget();
        let got = c.observe(&sample(10.0), t + Duration::from_secs(40));
        assert_eq!(got, Adjust::Hold);
        assert_eq!(c.budget(), before);
    }

    #[test]
    fn the_budget_stays_within_its_bounds() {
        let p = policy();
        let mut c = Controller::new(p, 50_000); // above the ceiling
        assert_eq!(c.budget(), p.ceiling_bytes, "clamped on construction");
        let mut t = Instant::now();
        let mut tps = 1.0;
        for _ in 0..30 {
            t += Duration::from_secs(11);
            tps *= 1.5;
            c.observe(&sample(tps), t);
            assert!(c.budget() <= p.ceiling_bytes, "never above the ceiling");
            assert!(c.budget() >= p.floor_bytes, "never below the floor");
        }
    }
}
