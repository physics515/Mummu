//! **Tier planning** — P9 stage 3(b): which device runs which MoE expert at
//! which stored precision, all at once.
//!
//! A `.mummu` pack holds every expert at every level ([`crate::pack`]), so
//! the runtime can keep *different* experts resident on *different*
//! devices at *different* precisions simultaneously — int4 on the CPU,
//! int8 on an integrated GPU, f32 on the discrete card — and move them as
//! routing statistics shift ("hot-swapping"). This module decides the
//! assignment; [`crate::nn::ExpertPool`] executes it.
//!
//! The plan is two-phase and deterministic:
//!
//! 1. **Admission** — every expert gets the *cheapest* slot that exists
//!    (smallest bytes on the slowest device), so a plan either exists for
//!    all experts or fails loudly with the shortfall. Nothing is silently
//!    dropped.
//! 2. **Promotion** — experts are visited hottest-first and each moves to
//!    the most desirable tier that still fits: faster device first, then
//!    higher precision within it. Capacity is checked against the bytes
//!    already promised, so the result never oversubscribes a device.
//!
//! Hotness is the caller's (routing hit counts smoothed over requests);
//! uniform hotness degrades to "fill the best device in expert order".
//! Re-planning with new hotness and diffing against the live plan gives
//! the swap list — the pool applies exactly those moves.

use std::collections::BTreeMap;

pub use crate::pack::Precision;

/// What kind of device a tier lives on — the planner's speed ordering
/// follows `speed`, not this tag; it exists for labels and defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceClass {
    Cpu,
    IntegratedGpu,
    DiscreteGpu,
}

/// One device the planner may place experts on.
#[derive(Debug, Clone, PartialEq)]
pub struct TierDevice {
    pub name: String,
    pub class: DeviceClass,
    /// The precision ladder this device runs experts at, **best first**
    /// (e.g. a discrete GPU `[F32, Q8]`, a CPU `[Q8, Q4]`). A level absent
    /// here is never placed on this device, whatever the pack stores.
    pub ladder: Vec<Precision>,
    /// Work this device is already committed to every step, in expert
    /// equivalents. The trunk lives on exactly one device and runs on every
    /// token; without it that device looks idle to the scheduler.
    pub preload_units: usize,
    /// Bytes of expert weights this device may hold (after whatever else
    /// — the trunk, caches, the desktop — already lives there).
    pub budget_bytes: u64,
    /// Relative throughput rank; higher runs hotter experts. Ties break
    /// toward the earlier device in the slice.
    pub speed: u32,
}

/// One expert's placement: device index into the planner's device slice
/// and the stored level it is loaded at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Tier {
    pub device: usize,
    pub precision: Precision,
}

/// Resident bytes of one expert at each level the pack stores for it
/// (values + scales), as a device would hold them. Levels a device's
/// backend widens (f16 on an f32 backend) are the caller's to cost.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertCost {
    pub bytes: BTreeMap<Precision, u64>,
}

/// A complete assignment: one tier per expert (planner input order) and the
/// bytes each device ends up holding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierPlan {
    pub tiers: Vec<Tier>,
    pub used_bytes: Vec<u64>,
}

impl TierPlan {
    /// Experts whose tier differs between `self` (live) and `next`: the
    /// swap list, in expert order.
    #[must_use]
    pub fn diff(&self, next: &TierPlan) -> Vec<(usize, Tier)> {
        self.tiers
            .iter()
            .zip(&next.tiers)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (_, b))| (i, *b))
            .collect()
    }

    /// Experts per (device, precision), for logs and `/api/ps`-style summaries.
    #[must_use]
    pub fn histogram(&self) -> BTreeMap<(usize, Precision), usize> {
        let mut h = BTreeMap::new();
        for t in &self.tiers {
            *h.entry((t.device, t.precision)).or_insert(0) += 1;
        }
        h
    }
}

/// Plan tiers for `costs.len()` experts over `devices`, hottest-first.
/// `hotness` is per expert (any non-negative scale; empty = uniform).
///
/// Errors when some expert fits nowhere even at its cheapest level — the
/// message names the expert and the shortfall, never a silent drop.
pub fn plan_tiers(
    devices: &[TierDevice],
    costs: &[ExpertCost],
    hotness: &[f64],
) -> Result<TierPlan, String> {
    if devices.is_empty() {
        return Err("tier plan: no devices".into());
    }
    if !hotness.is_empty() && hotness.len() != costs.len() {
        return Err(format!(
            "tier plan: {} hotness values for {} experts",
            hotness.len(),
            costs.len()
        ));
    }
    // Device visiting order: fastest first (stable on ties).
    let mut by_speed: Vec<usize> = (0..devices.len()).collect();
    by_speed.sort_by(|&a, &b| devices[b].speed.cmp(&devices[a].speed));

    // Desirability order of every (device, precision) slot: faster device
    // first, then that device's ladder best-first.
    let slots: Vec<Tier> = by_speed
        .iter()
        .flat_map(|&d| {
            devices[d].ladder.iter().map(move |&p| Tier {
                device: d,
                precision: p,
            })
        })
        .collect();
    // The admission order is the reverse: cheapest bytes on the slowest device.
    // Host residency differs from stored size for Q4: flex unpacks nibbles
    // to one i8 per element at load (`q_from_data`), so a Q4 blob occupies
    // 1.125 B/elem resident against 0.625 stored — exactly 9/5. Without
    // this the planner overpacks the host by 1.8x and the commit charge,
    // not the budget, becomes the limit.
    let cost_of = |tier: Tier, e: usize| -> Option<u64> {
        let stored = costs[e].bytes.get(&tier.precision).copied()?;
        let host_q4 = devices[tier.device].class == DeviceClass::Cpu
            && tier.precision == Precision::Q4;
        Some(if host_q4 { stored * 9 / 5 } else { stored })
    };

    let mut used = vec![0u64; devices.len()];
    let mut tiers: Vec<Tier> = Vec::with_capacity(costs.len());

    // 1. Admission: every expert at its cheapest available slot.
    for (e, cost) in costs.iter().enumerate() {
        let mut best: Option<(u64, Tier)> = None;
        for &slot in slots.iter().rev() {
            let Some(bytes) = cost_of(slot, e) else { continue };
            if used[slot.device] + bytes <= devices[slot.device].budget_bytes
                && best.is_none_or(|(b, _)| bytes < b)
            {
                best = Some((bytes, slot));
            }
        }
        let Some((bytes, slot)) = best else {
            let cheapest = cost.bytes.values().copied().min().unwrap_or(0);
            let free: u64 = devices
                .iter()
                .zip(&used)
                .map(|(d, &u)| d.budget_bytes.saturating_sub(u))
                .max()
                .unwrap_or(0);
            return Err(format!(
                "tier plan: expert {e} ({cheapest} bytes at its cheapest level) fits no device — \
                 largest free budget is {free} bytes after admitting experts 0..{e}"
            ));
        };
        used[slot.device] += bytes;
        tiers.push(slot);
    }

    // 2a. Scheduler A: how many experts each device SHOULD hold.
    //
    // Admission below put everything on the slowest device and promotion
    // pulls it up, fastest-slot-first — which fills the quick device and
    // dumps the remainder on the slow ones. That is the right shape when
    // devices run one after another, and the wrong one now that they run
    // concurrently: the layer then costs the slowest device's share, so the
    // objective is for every device to FINISH TOGETHER, not for the fast one
    // to be busiest. Measured, fill-first put 996 experts on a device that
    // takes 1.59 ms each and 1052 on devices taking ~14 ms — the slow side
    // ran ~9x longer and decided the layer.
    //
    // `schedule::divide` gives the makespan-minimizing split; promotion
    // treats it as a quota rather than a target, so a device is never filled
    // past its share while a slower one still has work it could have taken.
    let quota = {
        let sched: Vec<crate::schedule::Device> = devices
            .iter()
            .map(|dev| {
                // Cheapest an expert can be on this device, over the whole
                // set — what its budget divides into.
                let cheapest = costs
                    .iter()
                    .filter_map(|c| {
                        dev.ladder.iter().filter_map(|p| c.bytes.get(p).copied()).min()
                    })
                    .max()
                    .unwrap_or(u64::MAX);
                crate::schedule::Device {
                    name: dev.name.clone(),
                    throughput: f64::from(dev.speed),
                    capacity_units: if cheapest == 0 || cheapest == u64::MAX {
                        0
                    } else {
                        (dev.budget_bytes / cheapest) as usize
                    },
                    // Work this device already owes every token, in cluster
                    // equivalents — the trunk, for whichever device holds it.
                    // Counting it is what stops a host that is already
                    // saturated from being handed a "fair share" on top.
                    preload_units: dev.preload_units,
                }
            })
            .collect();
        crate::schedule::divide(&sched, costs.len()).units
    };
    let mut held = vec![0usize; devices.len()];
    for t in &tiers {
        held[t.device] += 1;
    }

    // 2b. Promotion, hottest first.
    let mut order: Vec<usize> = (0..costs.len()).collect();
    if !hotness.is_empty() {
        order.sort_by(|&a, &b| {
            hotness[b]
                .partial_cmp(&hotness[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
    }
    // Two phases, because placement and precision optimize different things
    // and one pass let the wrong one win.
    //
    // Moving an expert from a 14.15 ms device to a 1.59 ms one saves 12.6 ms
    // EVERY token. Upgrading the precision of an expert already on the fast
    // device saves nothing — for a quantized checkpoint the extra bytes buy
    // no accuracy at all (measured 0.0000 relative error for f16 against a
    // 4.55 bits/param source) — and those bytes are capacity another expert
    // could have used. Interleaved, precision won: the fast device ended up
    // holding 610 experts at mixed rungs while the slow one held 1374.
    //
    // So: fill the fast devices first, at their CHEAPEST rung, and only then
    // spend whatever is left over on precision.

    // Phase 1 — placement. Hottest first, onto the fastest device with room.
    for &e in &order {
        let cur = tiers[e];
        let cur_bytes = cost_of(cur, e).expect("admitted tier has a cost");
        for &d in &by_speed {
            if d == cur.device {
                continue;
            }
            if held[d] >= quota[d] {
                continue; // scheduler A's balanced share for this device
            }
            // Move when the destination is faster — or when the CURRENT
            // holder is past its own balanced share, even at equal-or-lower
            // speed. The quota is the arbiter, not raw speed: it already
            // prices in preloaded work, so a device level with the CPU on
            // throughput is still a win while the CPU carries the trunk.
            // The old strict "only strictly faster" gate made an
            // equal-speed device UNREACHABLE — admission parks everything
            // on the cheapest host slot, and with the integrated GPU rated
            // 71 against the host's 72 it received zero clusters in every
            // default plan (found by adversarial review, verified against
            // this function line by line).
            if devices[d].speed <= devices[cur.device].speed && held[cur.device] <= quota[cur.device]
            {
                continue; // no makespan gain from this move
            }
            // Cheapest rung this device accepts — capacity beats precision.
            let Some((bytes, slot)) = devices[d]
                .ladder
                .iter()
                .filter_map(|&p| {
                    let slot = Tier { device: d, precision: p };
                    cost_of(slot, e).map(|b| (b, slot))
                })
                .min_by_key(|&(b, _)| b)
            else {
                continue;
            };
            if used[d] + bytes <= devices[d].budget_bytes {
                used[cur.device] -= cur_bytes;
                used[d] += bytes;
                held[cur.device] -= 1;
                held[d] += 1;
                tiers[e] = slot;
                break;
            }
        }
    }

    // Phase 2 — precision, with what is left, and without moving anything.
    for &e in &order {
        let cur = tiers[e];
        let cur_bytes = cost_of(cur, e).expect("admitted tier has a cost");
        for &p in &devices[cur.device].ladder {
            let slot = Tier { device: cur.device, precision: p };
            if slot == cur {
                break; // the ladder is best-first: nothing finer remains
            }
            let Some(bytes) = cost_of(slot, e) else { continue };
            if used[cur.device] - cur_bytes + bytes <= devices[cur.device].budget_bytes {
                used[cur.device] = used[cur.device] - cur_bytes + bytes;
                tiers[e] = slot;
                break;
            }
        }
    }

    Ok(TierPlan {
        tiers,
        used_bytes: used,
    })
}

/// Hotness smoothing across requests: an exponential moving average of
/// per-expert routing hits, normalized per request so a long prompt does
/// not drown a short one. `alpha` in (0, 1]: 1 = this request only.
pub fn smooth_hotness(prev: &mut Vec<f64>, hits: &[u64], alpha: f64) {
    if prev.len() != hits.len() {
        *prev = vec![0.0; hits.len()];
    }
    let total: u64 = hits.iter().sum();
    if total == 0 {
        return;
    }
    let inv = 1.0 / total as f64;
    for (p, &h) in prev.iter_mut().zip(hits) {
        *p = (1.0 - alpha) * *p + alpha * (h as f64 * inv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(q4: u64, q8: u64, f32: u64) -> ExpertCost {
        ExpertCost {
            bytes: [(Precision::Q4, q4), (Precision::Q8, q8), (Precision::F32, f32)]
                .into_iter()
                .collect(),
        }
    }

    fn devices(cpu_budget: u64, gpu_budget: u64) -> Vec<TierDevice> {
        vec![
            TierDevice {
                name: "cpu".into(),
                class: DeviceClass::Cpu,
                ladder: vec![Precision::Q8, Precision::Q4],
                budget_bytes: cpu_budget,
                speed: 1,
                preload_units: 0,
            },
            TierDevice {
                name: "gpu".into(),
                class: DeviceClass::DiscreteGpu,
                ladder: vec![Precision::F32, Precision::Q8],
                budget_bytes: gpu_budget,
                speed: 10,
                preload_units: 0,
            },
        ]
    }

    /// The fast device is filled with EXPERTS before any of its bytes go on
    /// precision.
    ///
    /// This reverses the earlier policy, deliberately and on measurement. Here
    /// the GPU's 8 bytes hold either one f32 expert or four q8 ones, and it is
    /// 10x the CPU's speed: four experts on it beats one on it and three on
    /// the CPU, by a wide margin. The old ordering interleaved placement and
    /// precision, so precision won — on the real 27B that left the 1.59 ms
    /// device holding 610 clusters while the 14.15 ms device held 1374.
    ///
    /// Precision above the source buys nothing anyway (measured 0.0000
    /// relative error for f16 against a 4.55 bits/param checkpoint), while a
    /// byte spent on it is capacity another expert could have used.
    #[test]
    fn the_fast_device_is_filled_with_experts_before_precision() {
        let costs = vec![cost(1, 2, 8); 4];
        let plan = plan_tiers(&devices(100, 8), &costs, &[0.1, 0.5, 0.3, 0.1]).unwrap();
        assert!(
            plan.tiers.iter().all(|t| t.device == 1),
            "every expert should be on the fast device: {:?}",
            plan.tiers
        );
        assert!(
            plan.tiers.iter().all(|t| t.precision == Precision::Q8),
            "at its cheapest rung, not its best: {:?}",
            plan.tiers
        );
        assert_eq!(plan.used_bytes, vec![0, 8]);
    }

    /// An equal-speed device must still receive work when the current
    /// holder is over its balanced share. The trunk-preloaded host and the
    /// integrated GPU are level on measured throughput; the old strictly-
    /// faster gate made the iGPU unreachable — every cluster admitted to
    /// the cheap host slot and stayed there, while scheduler A's quota
    /// assumed the iGPU would carry ~a fifth of the slow-tier work.
    #[test]
    fn an_equal_speed_idle_device_relieves_an_overloaded_one() {
        let devices = vec![
            TierDevice {
                name: "cpu".into(),
                class: DeviceClass::Cpu,
                ladder: vec![Precision::Q8, Precision::Q4],
                budget_bytes: 1_000,
                speed: 72,
                // Busy with the trunk: its balanced share of extra work is
                // near zero, so held > quota from the first admission.
                preload_units: 1_000,
            },
            TierDevice {
                name: "igpu".into(),
                class: DeviceClass::IntegratedGpu,
                // The expensive rung only — the kernel-safety shape of the
                // real integrated GPU after the source cap.
                ladder: vec![Precision::F32],
                budget_bytes: 1_000,
                speed: 71,
                preload_units: 0,
            },
        ];
        let costs = vec![cost(1, 2, 8); 8];
        let plan = plan_tiers(&devices, &costs, &[]).unwrap();
        let on_igpu = plan.tiers.iter().filter(|t| t.device == 1).count();
        assert!(
            on_igpu > 0,
            "the idle equal-speed device took nothing: {:?}",
            plan.tiers
        );
        assert!(
            plan.tiers.iter().filter(|t| t.device == 1).all(|t| t.precision == Precision::F32),
            "only its own ladder's rung may be used: {:?}",
            plan.tiers
        );
    }

    /// ...and when the fast device is genuinely full, hotness still decides
    /// who gets in.
    #[test]
    fn the_hottest_expert_wins_the_last_slot_on_the_fast_device() {
        let costs = vec![cost(1, 2, 8); 4];
        // GPU budget 2 = exactly one expert at its cheapest rung (q8).
        let plan = plan_tiers(&devices(100, 2), &costs, &[0.1, 0.5, 0.3, 0.1]).unwrap();
        assert_eq!(plan.tiers[1], Tier { device: 1, precision: Precision::Q8 });
        assert!(
            plan.tiers.iter().enumerate().all(|(e, t)| e == 1 || t.device == 0),
            "only the hottest gets the fast device: {:?}",
            plan.tiers
        );
    }

    #[test]
    fn admission_spills_to_int4_when_budgets_are_tight() {
        // CPU holds 4 bytes: four q4 experts exactly; GPU off (0 budget).
        let costs = vec![cost(1, 2, 8); 4];
        let plan = plan_tiers(&devices(4, 0), &costs, &[]).unwrap();
        assert!(plan.tiers.iter().all(|t| *t == Tier { device: 0, precision: Precision::Q4 }));
        assert_eq!(plan.used_bytes, vec![4, 0]);
    }

    #[test]
    fn promotion_upgrades_precision_within_a_device_when_room_allows() {
        // CPU holds 5: admission puts four q4 (4 bytes); promotion lifts the
        // hottest to q8 (frees 1, costs 2 → 5). The next can't (6 > 5).
        let costs = vec![cost(1, 2, 8); 4];
        let plan = plan_tiers(&devices(5, 0), &costs, &[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(plan.tiers[2].precision, Precision::Q8);
        assert_eq!(plan.tiers.iter().filter(|t| t.precision == Precision::Q4).count(), 3);
        assert_eq!(plan.used_bytes[0], 5);
    }

    #[test]
    fn no_fit_is_a_loud_error_naming_the_expert() {
        let costs = vec![cost(1, 2, 8); 4];
        let err = plan_tiers(&devices(3, 0), &costs, &[]).unwrap_err();
        assert!(err.contains("expert 3"), "{err}");
    }

    #[test]
    fn diff_lists_only_moved_experts() {
        let costs = vec![cost(1, 2, 8); 3];
        // A GPU budget of 2 holds exactly ONE expert at its cheapest rung, so
        // hotness decides which — without that contention every expert lands
        // on the fast device in both plans and there is no diff to observe.
        let a = plan_tiers(&devices(100, 2), &costs, &[1.0, 0.0, 0.0]).unwrap();
        let b = plan_tiers(&devices(100, 2), &costs, &[0.0, 1.0, 0.0]).unwrap();
        let moves = a.diff(&b);
        assert_eq!(moves.len(), 2, "{moves:?}");
        assert!(moves.contains(&(1, Tier { device: 1, precision: Precision::Q8 })), "{moves:?}");
        assert!(moves.contains(&(0, Tier { device: 0, precision: Precision::Q8 })), "{moves:?}");
    }

    #[test]
    fn smoothing_normalizes_per_request() {
        let mut h = Vec::new();
        smooth_hotness(&mut h, &[3, 1], 1.0);
        assert_eq!(h, vec![0.75, 0.25]);
        smooth_hotness(&mut h, &[0, 4], 0.5);
        assert_eq!(h, vec![0.375, 0.625]);
        smooth_hotness(&mut h, &[0, 0], 0.5); // empty request changes nothing
        assert_eq!(h, vec![0.375, 0.625]);
    }
}
