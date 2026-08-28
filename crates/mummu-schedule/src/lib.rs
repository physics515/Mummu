//! **Scheduler A — divide the work across devices.**
//!
//! Distinct from the `mummu-mix` crate, which is scheduler B: this one decides *how
//! much* work each device gets, `mummu-mix` decides *how many bits* the weights on
//! a device are stored in. They are separate problems with separate
//! objectives — makespan here, accuracy there — and they were conflated for
//! a long time by a placement that simply filled the fastest device until it
//! was full and pushed the remainder onto the next one.
//!
//! Fill-until-full is wrong once devices run concurrently, because it
//! optimizes the wrong thing. Measured on this box: filling the discrete GPU
//! and spilling the rest put 996 clusters on a device that takes 1.59 ms each
//! and 1052 on devices that take ~14 ms each, so the slow devices ran ~9x
//! longer than the fast one and decided the whole layer. The objective is not
//! "keep the fast device busy", it is **make every device finish at the same
//! time** — minimize the maximum, not the sum.
//!
//! The algorithm is water-filling: hand out work in proportion to throughput,
//! clamp any device that runs out of memory, and redistribute what it could
//! not take among those still open. That terminates in at most one pass per
//! device and gives the makespan-optimal split whenever memory does not bind;
//! where memory does bind, it gives the best split subject to those caps.
//!
//! The residency modules extend the same philosophy — measure, don't
//! guess — to VRAM itself. [`p2`] keeps an online quantile of ambient VRAM
//! consumption in five numbers; [`watermark`] turns it into a
//! chance-constrained reserve `Pr[usage + ambient > VRAM] <= alpha` with
//! hysteresis, replacing the one-NVML-sample-at-load guess that once
//! drifted 1.9 -> 8.9 GiB mid-session; [`placement`] chooses which
//! contiguous run of layers lives on the GPU under that budget (a knapsack
//! on a chain, solved exactly); and [`prefill`] picks the prefill chunk
//! size that removes the unchunked ~855 MB activation peak from the
//! reserve entirely.

pub mod p2;
pub mod placement;
pub mod prefill;
pub mod watermark;

/// One device the scheduler may assign work to.
#[derive(Debug, Clone)]
pub struct Device {
    pub name: String,
    /// Work units per second, in any consistent unit — only ratios matter.
    /// Measured, not guessed: see `examples/device-throughput.rs`.
    pub throughput: f64,
    /// Most units this device can hold, from its memory budget and the bytes
    /// a unit costs at the least precise rung it will accept. This is where
    /// scheduler B feeds in.
    pub capacity_units: usize,
    /// Work this device is ALREADY committed to every step, in the same
    /// units — not schedulable, but it delays everything scheduled behind it.
    ///
    /// Without this, `throughput` silently means "speed when idle", and the
    /// device carrying the model trunk looks as free as one doing nothing.
    /// Measured consequence: the host, already running a trunk that saturates
    /// it, was handed 191 further clusters as its "fair share".
    pub preload_units: usize,
}

/// How the work came out.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Units assigned, parallel to the `devices` slice.
    pub units: Vec<usize>,
    /// Units that fit nowhere. Non-zero means the caller must quantize
    /// further (scheduler B), spill to a device not offered here, or refuse.
    pub unplaced: usize,
}

impl Plan {
    /// Time the slowest device takes, in the reciprocal of `throughput`'s
    /// unit — including whatever it was already committed to. This is what a
    /// layer actually costs when devices run concurrently, and the quantity
    /// the split minimizes.
    #[must_use]
    pub fn makespan(&self, devices: &[Device]) -> f64 {
        devices
            .iter()
            .zip(&self.units)
            .filter(|(d, _)| d.throughput > 0.0)
            .map(|(d, &n)| (d.preload_units + n) as f64 / d.throughput)
            .fold(0.0, f64::max)
    }
}

/// Split `total_units` across `devices` to minimize the makespan.
///
/// Units are treated as interchangeable and equal-cost, which is what an FFN
/// cluster is. A device with zero throughput or zero capacity is skipped
/// rather than being handed work it cannot do.
#[must_use]
pub fn divide(devices: &[Device], total_units: usize) -> Plan {
    let mut units = vec![0usize; devices.len()];
    let mut remaining = total_units;

    // Greedy on the RESULTING finish time, one unit at a time.
    //
    // This replaces a proportional pass plus a remainder pass. Proportional
    // shares cannot express preloaded work — a device that is already busy
    // deserves a smaller share, not the same fraction — and the two passes
    // could disagree at the boundary. Assigning to whichever device would
    // finish earliest WITH the unit handles both, and is the textbook
    // approximation for makespan on identical tasks.
    //
    // O(units x devices): a few thousand comparisons for a whole model, which
    // is nothing beside the work being placed.
    while remaining > 0 {
        let best = (0..devices.len())
            .filter(|&i| devices[i].throughput > 0.0 && units[i] < devices[i].capacity_units)
            .min_by(|&a, &b| {
                let finish_with = |i: usize| {
                    (devices[i].preload_units + units[i] + 1) as f64 / devices[i].throughput
                };
                finish_with(a)
                    .partial_cmp(&finish_with(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        match best {
            Some(i) => {
                units[i] += 1;
                remaining -= 1;
            }
            None => break, // every device is full or dead
        }
    }

    Plan {
        units,
        unplaced: remaining,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(name: &str, throughput: f64, capacity_units: usize) -> Device {
        Device { name: name.into(), throughput, capacity_units, preload_units: 0 }
    }

    fn busy(name: &str, throughput: f64, capacity_units: usize, preload_units: usize) -> Device {
        Device { name: name.into(), throughput, capacity_units, preload_units }
    }

    /// With memory to spare, work splits in proportion to speed — so every
    /// device finishes at the same time, which is the whole objective.
    #[test]
    fn work_splits_in_proportion_to_throughput() {
        // 1.59 ms / 14.15 ms / 13.82 ms inverted: the measured ratios.
        let devices = vec![
            dev("dgpu", 1.0 / 1.59, 100_000),
            dev("igpu", 1.0 / 14.15, 100_000),
            dev("cpu", 1.0 / 13.82, 100_000),
        ];
        let plan = divide(&devices, 2048);
        assert_eq!(plan.unplaced, 0);
        assert_eq!(plan.units.iter().sum::<usize>(), 2048);
        // The fast device should carry roughly 9x either slow one.
        assert!(plan.units[0] > 8 * plan.units[1], "got {:?}", plan.units);
        // And the two slow devices, being near-equal, get near-equal shares.
        let (a, b) = (plan.units[1] as f64, plan.units[2] as f64);
        assert!((a - b).abs() / a.max(b) < 0.1, "got {:?}", plan.units);
    }

    /// The split beats filling the fastest device first — which is the
    /// placement this replaces, and the reason it was wrong.
    #[test]
    fn beats_filling_the_fastest_device_first() {
        let devices = vec![
            dev("dgpu", 1.0 / 1.59, 996),
            dev("igpu", 1.0 / 14.15, 885),
            dev("cpu", 1.0 / 13.82, 100_000),
        ];
        let balanced = divide(&devices, 2048);

        // Fill-in-order: 996 on the GPU, 885 on the iGPU, the rest to the CPU.
        let greedy = Plan {
            units: vec![996, 885, 2048 - 996 - 885],
            unplaced: 0,
        };
        assert!(
            balanced.makespan(&devices) < greedy.makespan(&devices),
            "balanced {:?} ({:.0}) should beat fill-first {:?} ({:.0})",
            balanced.units,
            balanced.makespan(&devices),
            greedy.units,
            greedy.makespan(&devices)
        );
    }

    /// A device already busy gets a SMALLER share, not an equal one. The host
    /// carrying a model trunk is not idle just because no clusters are on it
    /// yet, and treating it as idle handed it 191 clusters it had no time for.
    #[test]
    fn a_preloaded_device_is_given_less_work() {
        let idle = vec![dev("a", 1.0, 1000), dev("b", 1.0, 1000)];
        assert_eq!(divide(&idle, 100).units, vec![50, 50], "equal when both idle");

        // Same speeds, but `b` already owes 40 units of work.
        let loaded = vec![dev("a", 1.0, 1000), busy("b", 1.0, 1000, 40)];
        let plan = divide(&loaded, 100);
        assert!(plan.units[0] > plan.units[1], "the busy device takes less: {plan:?}");
        // ...and they still finish together, which is the point.
        let finish = |i: usize| (loaded[i].preload_units + plan.units[i]) as f64;
        assert!((finish(0) - finish(1)).abs() <= 1.0, "{plan:?}");
    }

    /// The last unit goes where it finishes soonest, not where the device is
    /// idlest. Ranking by current load put it on an empty slow device and
    /// tripled the makespan.
    #[test]
    fn the_remainder_goes_where_it_finishes_soonest() {
        let devices = vec![dev("cpu", 1.0, 100), dev("gpu", 10.0, 100)];
        let plan = divide(&devices, 4);
        assert_eq!(plan.units, vec![0, 4], "all four belong on the fast device");
        assert!((plan.makespan(&devices) - 0.4).abs() < 1e-9);
    }

    /// A device that cannot hold its proportional share is clamped, and what
    /// it could not take goes to the others rather than vanishing.
    #[test]
    fn a_device_short_of_memory_is_clamped_and_the_rest_redistributed() {
        let devices = vec![
            dev("dgpu", 10.0, 50), // fast but tiny
            dev("cpu", 1.0, 100_000),
        ];
        let plan = divide(&devices, 500);
        assert_eq!(plan.units[0], 50, "the fast device fills to its cap");
        assert_eq!(plan.units[1], 450, "and the remainder lands on the host");
        assert_eq!(plan.unplaced, 0);
    }

    /// When nothing can hold the work, say so rather than silently dropping
    /// it — the caller has to quantize further or refuse the model.
    #[test]
    fn work_that_fits_nowhere_is_reported() {
        let devices = vec![dev("dgpu", 10.0, 10), dev("igpu", 1.0, 5)];
        let plan = divide(&devices, 100);
        assert_eq!(plan.units, vec![10, 5]);
        assert_eq!(plan.unplaced, 85);
    }

    /// Devices that cannot take work at all are skipped, not handed zero-speed
    /// assignments that would make the makespan infinite.
    #[test]
    fn dead_devices_are_skipped() {
        let devices = vec![
            dev("broken", 0.0, 1_000),
            dev("full", 5.0, 0),
            dev("cpu", 1.0, 1_000),
        ];
        let plan = divide(&devices, 40);
        assert_eq!(plan.units, vec![0, 0, 40]);
        assert!(plan.makespan(&devices).is_finite());
    }

    /// Every unit is placed exactly once — no double-counting from the
    /// proportional pass and the one-at-a-time remainder pass overlapping.
    #[test]
    fn every_unit_is_placed_exactly_once() {
        let devices = vec![
            dev("a", 1.0 / 1.59, 700),
            dev("b", 1.0 / 14.15, 900),
            dev("c", 1.0 / 13.82, 900),
        ];
        for total in [0usize, 1, 7, 64, 999, 2048, 2500] {
            let plan = divide(&devices, total);
            assert_eq!(
                plan.units.iter().sum::<usize>() + plan.unplaced,
                total,
                "total {total} lost or duplicated units: {plan:?}"
            );
            for (d, &n) in devices.iter().zip(&plan.units) {
                assert!(n <= d.capacity_units, "{} over capacity: {plan:?}", d.name);
            }
        }
    }

    /// A single device gets everything it can hold — the degenerate case must
    /// not need the parallel machinery to behave.
    #[test]
    fn one_device_takes_what_it_can() {
        let devices = vec![dev("only", 3.0, 30)];
        assert_eq!(divide(&devices, 20).units, vec![20]);
        let over = divide(&devices, 45);
        assert_eq!(over.units, vec![30]);
        assert_eq!(over.unplaced, 15);
    }
}
