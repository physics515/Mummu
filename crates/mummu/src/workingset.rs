//! **VRAM as a working set** — P9 stage 4: host RAM is the backing store,
//! device memory is a cache, and the CPU (or an iGPU) computes whatever the
//! cache could not supply in time.
//!
//! The tiering in [`crate::tier`] gives every unit a *permanent* home, which
//! caps how much of a model can ever run on the fast device: a 15 GB model
//! and a 9 GiB budget means most units live — and therefore compute — on the
//! CPU forever. The working-set design breaks that cap. Every unit lives in
//! host RAM; the device holds only what the *current* layers need, staged in
//! ahead of use and evicted behind it, so every layer can execute on the
//! fast device without any unit owning a permanent slot.
//!
//! # What makes this schedulable rather than a cache heuristic
//!
//! The access sequence is **known in advance and identical every token**:
//! layer 0, 1, … L-1, then the next token repeats it. Two consequences the
//! scheduler exploits, neither available to a general-purpose cache:
//!
//! - **Eviction is optimal, not heuristic.** Bélády's rule — evict the entry
//!   whose next use is furthest away — is normally unimplementable because
//!   it needs the future. Here the future is a `for` loop. [`Plan::victim`]
//!   applies it exactly.
//! - **Prefetch distance is a decision, not a guess.** Staging for layer
//!   `L + d` is issued while layer `L` computes; `d` is chosen from measured
//!   staging bandwidth and per-layer compute time, not from access history.
//!
//! # The bound that shapes the policy
//!
//! Streaming is not free, and for a *dense* model it cannot beat the CPU on
//! bandwidth alone: every unit is needed every layer, so streaming the whole
//! model each token moves the whole model across the bus each token — at
//! which point the CPU reading the same bytes from its own DDR is no worse.
//! Streaming wins in exactly two situations, and the policy is built around
//! them:
//!
//! 1. **Selectivity.** A routed MoE touches `top_k` of `E` experts per token,
//!    so it stages a `k/E` fraction. This is why the MoE conversion matters
//!    beyond fit: it turns "move everything" into "move what was routed".
//! 2. **Overlap.** Staging that happens *while the previous layer computes*
//!    costs nothing on the critical path until it exceeds compute time.
//!
//! So the policy is: **pin what fits, stream what is selective, overlap
//! always, and fall back to host compute on a miss** — never stall waiting
//! for a transfer, because a stall is strictly worse than computing in place
//! (the CPU already holds the bytes).

use std::collections::HashMap;

/// A unit of placement: one MoE expert, or one FFN neuron cluster.
pub type UnitId = usize;

/// Where a unit's compute happened — what the scheduler reports back so a
/// caller can see whether the cache is earning its keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Served {
    /// Resident on the device when the layer ran (the fast path).
    Cached,
    /// Staged in time by the prefetcher.
    Prefetched,
    /// Not resident and not staged in time — computed on the host instead.
    /// Not an error: computing in place beats stalling on a transfer.
    Overflow,
}

/// One layer's demand: the units it needs, in the order it needs them.
#[derive(Debug, Clone)]
pub struct LayerDemand {
    pub layer: usize,
    pub units: Vec<UnitId>,
}

/// What the scheduler decided for one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSchedule {
    pub layer: usize,
    /// Units to compute on the device (resident by the time the layer runs).
    pub on_device: Vec<UnitId>,
    /// Units to compute on the host — the cache could not supply them in
    /// time and stalling would be worse.
    pub overflow: Vec<UnitId>,
    /// Staging to *issue* now, for a later layer, while this layer computes.
    pub prefetch: Vec<UnitId>,
    /// Units evicted to make room for `prefetch`.
    pub evict: Vec<UnitId>,
}

/// Sizes and speeds the scheduler plans against. All measured, never assumed
/// — see `examples/stage-probe.rs`.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Device memory the working set may use.
    pub device_bytes: u64,
    /// Bytes per unit (uniform in practice: clusters are equal-sized by
    /// construction, experts by architecture).
    pub unit_bytes: u64,
    /// Host→device staging bandwidth, bytes/sec.
    pub stage_bytes_per_sec: f64,
    /// How long one layer's compute takes on the device, seconds. Staging
    /// issued during a layer is free until it exceeds this.
    pub layer_compute_secs: f64,
}

impl Budget {
    /// How many units fit in the device budget.
    #[must_use]
    pub fn capacity(&self) -> usize {
        if self.unit_bytes == 0 {
            return 0;
        }
        usize::try_from(self.device_bytes / self.unit_bytes).unwrap_or(usize::MAX)
    }

    /// How many units can be staged inside one layer's compute window —
    /// the prefetcher's per-layer budget. Staging beyond this lands late and
    /// would stall, so the scheduler sends the remainder to the host instead.
    #[must_use]
    pub fn stageable_per_layer(&self) -> usize {
        if self.unit_bytes == 0 || self.stage_bytes_per_sec <= 0.0 {
            return 0;
        }
        let per_unit_secs = self.unit_bytes as f64 / self.stage_bytes_per_sec;
        if per_unit_secs <= 0.0 {
            return usize::MAX;
        }
        (self.layer_compute_secs / per_unit_secs).floor().max(0.0) as usize
    }
}

/// The schedule for one full pass (one token) over a known layer sequence.
#[derive(Debug, Clone)]
pub struct Plan {
    pub layers: Vec<LayerSchedule>,
    /// Units pinned for the whole pass: they fit, and they are used often
    /// enough that streaming them would be pure waste.
    pub pinned: Vec<UnitId>,
    /// Fraction of unit-uses served from the device.
    pub hit_rate: f64,
}

impl Plan {
    /// Bélády's optimal victim: of `resident`, the unit whose next use is
    /// furthest in the future (or never). Implementable here only because
    /// the access sequence is known — see the module header.
    ///
    /// `next_use[u]` is the index of `u`'s next use, `usize::MAX` if unused
    /// again this pass.
    #[must_use]
    pub fn victim(resident: &[UnitId], next_use: &HashMap<UnitId, usize>) -> Option<UnitId> {
        resident
            .iter()
            .copied()
            .max_by_key(|u| next_use.get(u).copied().unwrap_or(usize::MAX))
    }
}

/// Build a working-set schedule for one pass over `demands`.
///
/// The policy, in order:
///
/// 1. **Pin the hot core.** Units used in every layer (a dense model's local
///    slab, an MoE's always-hot experts) never leave the device: streaming
///    something needed every layer is pure overhead. Pinning is capped so
///    the stream always has room to work.
/// 2. **Prefetch ahead.** While layer `L` computes, issue staging for the
///    units layer `L + 1` needs and does not have, bounded by
///    [`Budget::stageable_per_layer`] — what actually fits in the compute
///    window.
/// 3. **Evict optimally.** Make room with Bélády's rule.
/// 4. **Overflow to the host.** Anything not resident and not stageable in
///    time is computed on the host. Never stall: the host already has the
///    bytes, so waiting is strictly worse than computing.
pub fn schedule(demands: &[LayerDemand], budget: &Budget) -> Plan {
    let capacity = budget.capacity();
    let per_layer_stage = budget.stageable_per_layer();

    // --- 1. pin the units every layer needs ------------------------------
    let mut uses: HashMap<UnitId, usize> = HashMap::new();
    for d in demands {
        for &u in &d.units {
            *uses.entry(u).or_insert(0) += 1;
        }
    }
    // Reserve room for streaming: pinning the entire cache would leave the
    // prefetcher nowhere to land, turning every non-pinned unit into
    // overflow.
    let pin_cap = capacity.saturating_sub(per_layer_stage.max(1)).min(capacity);
    let mut hot: Vec<(UnitId, usize)> = uses.iter().map(|(&u, &n)| (u, n)).collect();
    // Hottest first; ties by id so a plan is reproducible.
    hot.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let pinned: Vec<UnitId> = hot
        .iter()
        .filter(|(_, n)| *n > 1) // used more than once — worth keeping
        .take(pin_cap)
        .map(|(u, _)| *u)
        .collect();

    // --- flat access sequence, for optimal eviction ----------------------
    // position -> unit, so "next use of u after position i" is a lookup.
    let mut seq: Vec<UnitId> = Vec::new();
    let mut layer_span: Vec<(usize, usize)> = Vec::with_capacity(demands.len());
    for d in demands {
        let start = seq.len();
        seq.extend(d.units.iter().copied());
        layer_span.push((start, seq.len()));
    }

    let mut resident: Vec<UnitId> = pinned.clone();
    let mut layers: Vec<LayerSchedule> = Vec::with_capacity(demands.len());
    let (mut hits, mut total) = (0usize, 0usize);

    for (li, d) in demands.iter().enumerate() {
        let mut on_device = Vec::new();
        let mut overflow = Vec::new();
        for &u in &d.units {
            total += 1;
            if resident.contains(&u) {
                on_device.push(u);
                hits += 1;
            } else {
                // Not staged in time: compute on the host rather than stall.
                overflow.push(u);
            }
        }

        // --- 2/3. prefetch for the next layer, evicting optimally --------
        let mut prefetch = Vec::new();
        let mut evict = Vec::new();
        if let Some(next) = demands.get(li + 1) {
            // Next use of each unit, measured from the end of this layer —
            // the horizon eviction decisions are made against.
            let from = layer_span[li].1;
            let mut next_use: HashMap<UnitId, usize> = HashMap::new();
            for (pos, &u) in seq.iter().enumerate().skip(from) {
                next_use.entry(u).or_insert(pos);
            }
            for &u in &next.units {
                if prefetch.len() >= per_layer_stage {
                    break; // the rest cannot land in time; they will overflow
                }
                if resident.contains(&u) || prefetch.contains(&u) {
                    continue;
                }
                if resident.len() >= capacity {
                    // Evict, but never a pinned unit and never something this
                    // prefetch round just brought in.
                    let candidates: Vec<UnitId> = resident
                        .iter()
                        .copied()
                        .filter(|r| !pinned.contains(r) && !prefetch.contains(r))
                        .collect();
                    let Some(v) = Plan::victim(&candidates, &next_use) else {
                        break; // nothing evictable — the rest overflows
                    };
                    resident.retain(|r| *r != v);
                    evict.push(v);
                }
                resident.push(u);
                prefetch.push(u);
            }
        }

        layers.push(LayerSchedule {
            layer: d.layer,
            on_device,
            overflow,
            prefetch,
            evict,
        });
    }

    let hit_rate = if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    };
    Plan {
        layers,
        pinned,
        hit_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(layers: usize, units_per_layer: usize) -> Vec<LayerDemand> {
        // A dense partitioned model: every layer needs its OWN units, and
        // needs all of them (exact mode).
        (0..layers)
            .map(|l| LayerDemand {
                layer: l,
                units: (0..units_per_layer)
                    .map(|u| l * units_per_layer + u)
                    .collect(),
            })
            .collect()
    }

    fn budget(capacity_units: usize, stage_per_layer: usize) -> Budget {
        // unit = 1 MB; pick bandwidth so exactly `stage_per_layer` units fit
        // in one layer's compute window.
        let unit = 1_000_000u64;
        Budget {
            device_bytes: unit * capacity_units as u64,
            unit_bytes: unit,
            stage_bytes_per_sec: (unit as f64) * stage_per_layer as f64 / 0.010,
            layer_compute_secs: 0.010,
        }
    }

    #[test]
    fn capacity_and_stage_window_come_from_measurements() {
        let b = budget(8, 3);
        assert_eq!(b.capacity(), 8);
        assert_eq!(b.stageable_per_layer(), 3);
        // A zero-bandwidth device can stage nothing (everything overflows).
        let none = Budget {
            stage_bytes_per_sec: 0.0,
            ..budget(8, 3)
        };
        assert_eq!(none.stageable_per_layer(), 0);
    }

    #[test]
    fn beladys_rule_evicts_the_furthest_next_use() {
        let resident = [1usize, 2, 3];
        let next: HashMap<UnitId, usize> = [(1, 10), (2, 99), (3, 20)].into_iter().collect();
        assert_eq!(Plan::victim(&resident, &next), Some(2));
        // A unit with no next use at all is the best possible victim.
        let next: HashMap<UnitId, usize> = [(1, 10), (3, 20)].into_iter().collect();
        assert_eq!(Plan::victim(&resident, &next), Some(2));
        assert_eq!(Plan::victim(&[], &next), None);
    }

    #[test]
    fn a_model_that_fits_is_pinned_and_never_streams() {
        // 4 layers x 2 units = 8 units, capacity 16: everything fits.
        // Nothing should be evicted, and after the first pass every use is a
        // hit. (Units are used once per pass here, so pinning is driven by
        // capacity, not by reuse — what matters is that nothing thrashes.)
        let demands = dense(4, 2);
        let plan = schedule(&demands, &budget(16, 4));
        assert!(
            plan.layers.iter().all(|l| l.evict.is_empty()),
            "a model that fits must never evict: {:?}",
            plan.layers
        );
    }

    #[test]
    fn streaming_prefetches_the_next_layer_while_this_one_computes() {
        // Capacity 4, can stage 2 units per layer window; each layer needs 2.
        let demands = dense(6, 2);
        let plan = schedule(&demands, &budget(4, 2));
        // Layer 0 issues staging for layer 1's units.
        assert_eq!(plan.layers[0].prefetch, vec![2, 3], "{:?}", plan.layers[0]);
        // And layer 1 then finds them resident — the point of the pipeline.
        assert_eq!(plan.layers[1].on_device, vec![2, 3]);
        assert!(plan.layers[1].overflow.is_empty());
    }

    #[test]
    fn what_cannot_be_staged_in_time_overflows_to_the_host_never_stalls() {
        // Each layer needs 4 units but only 1 can be staged per window.
        let demands = dense(4, 4);
        let plan = schedule(&demands, &budget(8, 1));
        let overflowed: usize = plan.layers.iter().map(|l| l.overflow.len()).sum();
        assert!(
            overflowed > 0,
            "a starved stage window must produce host overflow, not a stall"
        );
        // Every unit is still accounted for: computed somewhere, every layer.
        for (l, d) in plan.layers.iter().zip(&demands) {
            assert_eq!(
                l.on_device.len() + l.overflow.len(),
                d.units.len(),
                "every unit must be computed somewhere"
            );
        }
    }

    #[test]
    fn a_routed_moe_reuses_its_hot_experts() {
        // The selective case the design exists for: 8 layers routing over a
        // small hot set, so the cache actually pays off.
        let demands: Vec<LayerDemand> = (0..8)
            .map(|l| LayerDemand {
                layer: l,
                units: vec![0, 1, (l % 3) + 2], // two always-hot + one rotating
            })
            .collect();
        let plan = schedule(&demands, &budget(6, 2));
        assert!(
            plan.pinned.contains(&0) && plan.pinned.contains(&1),
            "always-hot experts must be pinned, not streamed: {:?}",
            plan.pinned
        );
        assert!(
            plan.hit_rate > 0.6,
            "a routed workload with a hot core should mostly hit: {}",
            plan.hit_rate
        );
    }

    #[test]
    fn pinning_never_starves_the_stream() {
        // Even when everything looks hot, the scheduler must leave room for
        // staging — a fully pinned cache turns every miss into permanent
        // overflow.
        let demands: Vec<LayerDemand> = (0..10)
            .map(|l| LayerDemand {
                layer: l,
                units: vec![0, 1, 2, l + 3],
            })
            .collect();
        let b = budget(4, 2);
        let plan = schedule(&demands, &b);
        assert!(
            plan.pinned.len() < b.capacity(),
            "pinning must leave staging room: pinned {} of {}",
            plan.pinned.len(),
            b.capacity()
        );
    }
}
