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
    /// unit. This is what a layer actually costs when devices run
    /// concurrently, and the quantity the split minimizes.
    #[must_use]
    pub fn makespan(&self, devices: &[Device]) -> f64 {
        devices
            .iter()
            .zip(&self.units)
            .filter(|(d, _)| d.throughput > 0.0)
            .map(|(d, &n)| n as f64 / d.throughput)
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
    // Open devices are those that can still take more work.
    let mut open: Vec<usize> = (0..devices.len())
        .filter(|&i| devices[i].throughput > 0.0 && devices[i].capacity_units > 0)
        .collect();
    let mut remaining = total_units;

    while remaining > 0 && !open.is_empty() {
        let total_throughput: f64 = open.iter().map(|&i| devices[i].throughput).sum();
        if total_throughput <= 0.0 {
            break;
        }

        // Proportional share of what is left. Rounding is deliberately down
        // so the loop never over-assigns; the remainder is placed one unit at
        // a time below, which also handles shares smaller than a whole unit.
        let mut handed_out = 0usize;
        let mut share: Vec<(usize, usize)> = Vec::with_capacity(open.len());
        for &i in &open {
            let want = (remaining as f64 * devices[i].throughput / total_throughput) as usize;
            let want = want.min(devices[i].capacity_units - units[i]);
            share.push((i, want));
            handed_out += want;
        }
        for (i, want) in &share {
            units[*i] += want;
        }
        remaining -= handed_out;

        // Anything left after rounding goes to whichever open device is
        // currently *earliest* to finish — the same greedy that the
        // proportional step approximates, applied one unit at a time.
        let mut progress = handed_out > 0;
        while remaining > 0 {
            let Some(&best) = open
                .iter()
                .filter(|&&i| units[i] < devices[i].capacity_units)
                .min_by(|&&a, &&b| {
                    let finish = |i: usize| units[i] as f64 / devices[i].throughput;
                    finish(a).partial_cmp(&finish(b)).unwrap_or(std::cmp::Ordering::Equal)
                })
            else {
                break;
            };
            units[best] += 1;
            remaining -= 1;
            progress = true;
        }

        // Close anything that is now full; if a whole pass placed nothing,
        // every open device is full and the rest is unplaceable.
        open.retain(|&i| units[i] < devices[i].capacity_units);
        if !progress {
            break;
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
        Device {
            name: name.into(),
            throughput,
            capacity_units,
        }
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
