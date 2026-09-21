//! **Joint placement: which device every layer runs on, and at what
//! precision each of its weights is stored there — re-solved whenever the
//! inputs move.**
//!
//! [`crate::plan`] answers half of this (the precision of every tensor on
//! ONE device under ONE byte budget), and the serve used to answer the other
//! half with a fixed rule (a GPU prefix sized by a configured budget, the
//! rest on the host at a fixed precision). Both halves are the same
//! decision: a byte spent on the card is a byte that cannot hold another
//! layer, and a layer on the host runs at the host's speed for its level.
//! So they are solved together, over measured inputs, with no configured
//! budget anywhere.
//!
//! # The model
//!
//! Decode is memory-bandwidth bound: a token streams every weight once. So
//! the time a part (a layer's attention or FFN weights) costs on device `d`
//! at level `p` is its bytes times that device's measured seconds-per-byte
//! at that level,
//!
//! ```text
//!   t(part, d, p) = bytes(params, p) · s[d][p]                         (1)
//!   T(x)          = Σ_l Σ_part t(part_l, d_l, p_part) + κ · crossings(x) (2)
//! ```
//!
//! where `κ` is the measured cost of moving the activations between
//! devices once and `crossings` counts adjacent layers on different devices.
//! `s[d][p]` is not bytes/bandwidth from a spec sheet: the serve times a real
//! projection on each device at each level, and the host's rate carries the
//! machine's CPU contention because it is measured under it.
//!
//! Memory on each device is a hard constraint:
//!
//! ```text
//!   Σ_{l on d} ( Σ_part bytes(part_l, p) + state_l ) + fixed_d ≤ C_d     (3)
//! ```
//!
//! `state_l` is the layer's per-generation state (KV at the context being
//! served, or the recurrent state), which lives wherever the layer does;
//! `fixed_d` is the device's working set (activations, allocator residual —
//! measured, see the serve's `placement` module); `C_d` is the capacity the
//! serve derives from the live card (`C = total − guard(ambient)`).
//!
//! Precision enters twice: through (1), where fewer bits read faster, and
//! through quality, `E(x) = Σ sensitivity · rel_error(p) · params / Σ params`
//! (the measured error model of [`crate::rel_error`]). Throughput is the
//! objective and quality the tie-break: a precision upgrade is taken only
//! with capacity that can no longer buy speed, and only while the total time
//! it costs stays within `tolerance · T` — so the card spends its leftover
//! bytes on the tensors whose error costs the most, and the host is never
//! slowed down for accuracy it did not need.
//!
//! # The solver
//!
//! A multiple-choice knapsack across devices; greedy by ratio, which is the
//! standard approximation (the exact answer is not worth its time inside a
//! rebalance that runs while a model serves, and greedy's error is bounded
//! by one step):
//!
//! 1. Every layer starts on the fallback device (the host) at the level each
//!    part runs FASTEST there — measured, so the host can pick a float level
//!    for one part and an integer level for another if that is what its
//!    kernels say.
//! 2. **Place**: repeatedly move the layer whose move saves the most time per
//!    byte of the destination's capacity, at the destination's fastest
//!    levels, while it fits and saves time.
//! 3. **Spend**: fill what capacity is left with precision upgrades, most
//!    error removed per byte first, within the time tolerance.
//!
//! # Changing a live placement
//!
//! Moving a layer is not free: it is re-read from the pack (so each level
//! keeps its single rounding) at the disk's measured rate. [`replan`]
//! therefore separates two cases:
//!
//! * **Repair** — the current placement violates (3) because ambient grew or
//!   this request needs more state. It must change now, and minimally: the
//!   cheapest relief per byte freed, which is a precision demotion or a move
//!   to the host, whichever loses less time.
//! * **Improve** — the placement is feasible but a better one exists. It is
//!   adopted only if the time it saves over the expected remaining tokens
//!   pays for the bytes it re-reads:
//!
//! ```text
//!   (T(x_now) − T(x*)) · H  >  Σ_{changed parts} bytes · s_disk            (4)
//! ```
//!
//!   which is the hysteresis: a few layers of gain on a quiet server whose
//!   horizon `H` is short do not justify re-reading gigabytes, and the same
//!   gain on a busy one does.

use crate::{Kind, QuantPolicy, rel_error};

/// One block of a layer that picks its precision as a unit: the attention
/// projections, or the FFN. Norms and other `Fixed` weights are not parts;
/// their bytes go into [`Layer::fixed_bytes`] at the level they always use.
#[derive(Debug, Clone)]
pub struct Part {
    /// Element count — the quality term's weight.
    pub params: usize,
    pub kind: Kind,
    /// The levels the pack stores for every tensor of this part, with the
    /// part's bytes in the pack at each — what a move re-reads, and what a
    /// device's measured rate is per. What it OCCUPIES once loaded is a
    /// property of the device ([`Device::resident`]).
    pub levels: Vec<(QuantPolicy, u64)>,
}

impl Part {
    fn bytes(&self, p: QuantPolicy) -> Option<u64> {
        self.levels.iter().find(|(q, _)| *q == p).map(|&(_, b)| b)
    }

    fn stores(&self, p: QuantPolicy) -> bool {
        self.bytes(p).is_some()
    }
}

/// One layer of the stack.
#[derive(Debug, Clone)]
pub struct Layer {
    pub parts: Vec<Part>,
    /// Weights that never change precision (norms, conv kernels, small
    /// projections): their bytes wherever the layer lives.
    pub fixed_bytes: u64,
    /// Per-generation state at the context being served (KV for an
    /// attention layer, the recurrent state for a linear-attention one).
    pub state_bytes: u64,
}

/// A device weights can live on.
#[derive(Debug, Clone)]
pub struct Device {
    /// Bytes this placement may occupy there — layers, their state and
    /// [`Self::fixed`]. Derived from the live device by the caller.
    pub capacity: u64,
    /// The device's working set, charged once it hosts any layer.
    pub fixed: u64,
    /// Measured seconds per pack byte at each level this device can
    /// execute. A level absent here is one the device cannot run.
    pub rate: Vec<(QuantPolicy, f64)>,
    /// Bytes resident per pack byte at each level (absent = 1.0). Not
    /// cosmetic: the host keeps a Q4 weight as an i8 slab plus a packed twin
    /// (3x the pack's nibbles), and the host and wgpu widen an f16 blob to
    /// f32 — so the same level costs different memory on different devices.
    pub resident: Vec<(QuantPolicy, f64)>,
}

impl Device {
    fn rate_of(&self, p: QuantPolicy) -> Option<f64> {
        self.rate.iter().find(|(q, _)| *q == p).map(|&(_, s)| s)
    }

    fn resident_of(&self, p: QuantPolicy) -> f64 {
        self.resident
            .iter()
            .find(|(q, _)| *q == p)
            .map_or(1.0, |&(_, m)| m)
    }

    fn occupies(&self, part: &Part, p: QuantPolicy) -> u64 {
        (part_bytes(part, p) as f64 * self.resident_of(p)).ceil() as u64
    }
}

/// Everything the solver reads.
#[derive(Debug, Clone)]
pub struct Problem {
    pub layers: Vec<Layer>,
    /// Device 0 is the fallback: every layer can live there, and a layer
    /// that fits nowhere else stays there. On this server it is the host.
    pub devices: Vec<Device>,
    /// Seconds one activation crossing between devices costs per token.
    pub crossing_s: f64,
    /// Seconds per byte re-read from the pack when a part changes.
    pub disk_s_per_byte: f64,
    /// Tokens the current placement is expected to serve — what a time
    /// saving per token is multiplied by in (4).
    pub horizon_tokens: f64,
    /// Fraction of total time precision upgrades may cost.
    pub tolerance: f64,
    /// The lowest level any part may take (Q4 on this server: Q2 is a
    /// quality cliff, and spilling to the host is the better relief).
    pub floor: QuantPolicy,
}

/// Where a layer is and what precision each of its parts is at.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub device: usize,
    /// Parallel to [`Layer::parts`].
    pub levels: Vec<QuantPolicy>,
}

/// A full placement.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub layers: Vec<Choice>,
}

/// What [`solve`] / [`replan`] decided.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub assignment: Assignment,
    /// Predicted seconds per token, (2).
    pub time_s: f64,
    /// Bytes charged against each device, (3).
    pub used: Vec<u64>,
    /// False when even the fallback device is over its capacity: the caller
    /// is out of memory everywhere and must say so rather than place.
    pub feasible: bool,
}

fn level_ok(p: QuantPolicy, floor: QuantPolicy) -> bool {
    p.bits() >= floor.bits()
}

fn part_time(part: &Part, dev: &Device, p: QuantPolicy) -> Option<f64> {
    Some(part.bytes(p)? as f64 * dev.rate_of(p)?)
}

fn part_bytes(part: &Part, p: QuantPolicy) -> u64 {
    part.bytes(p).unwrap_or(u64::MAX / 1024)
}

/// The level at which `part` runs fastest on `dev`, among the stored levels
/// the device can execute at or above the floor; ties go to the more
/// precise level (same time, less error).
fn fastest(part: &Part, dev: &Device, floor: QuantPolicy) -> Option<QuantPolicy> {
    part.levels
        .iter()
        .map(|&(p, _)| p)
        .filter(|&p| level_ok(p, floor))
        .filter_map(|p| part_time(part, dev, p).map(|t| (p, t)))
        .min_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.bits().cmp(&a.0.bits()))
        })
        .map(|(p, _)| p)
}

/// The layer at its fastest levels on `dev`, if every part can run there.
fn fastest_choice(
    layer: &Layer,
    device: usize,
    dev: &Device,
    floor: QuantPolicy,
) -> Option<Choice> {
    let levels = layer
        .parts
        .iter()
        .map(|p| fastest(p, dev, floor))
        .collect::<Option<Vec<_>>>()?;
    Some(Choice { device, levels })
}

fn layer_bytes(pb: &Problem, layer: &Layer, c: &Choice) -> u64 {
    let dev = &pb.devices[c.device];
    layer
        .parts
        .iter()
        .zip(&c.levels)
        .map(|(p, &q)| dev.occupies(p, q))
        .sum::<u64>()
        + layer.fixed_bytes
        + layer.state_bytes
}

fn layer_time(pb: &Problem, layer: &Layer, c: &Choice) -> f64 {
    let dev = &pb.devices[c.device];
    layer
        .parts
        .iter()
        .zip(&c.levels)
        .map(|(p, &q)| part_time(p, dev, q).unwrap_or(f64::INFINITY))
        .sum()
}

/// Predicted seconds per token, (2).
#[must_use]
pub fn time_of(pb: &Problem, a: &Assignment) -> f64 {
    let compute: f64 = pb
        .layers
        .iter()
        .zip(&a.layers)
        .map(|(l, c)| layer_time(pb, l, c))
        .sum();
    compute + pb.crossing_s * crossings(a) as f64
}

fn crossings(a: &Assignment) -> usize {
    a.layers
        .windows(2)
        .filter(|w| w[0].device != w[1].device)
        .count()
}

/// Bytes charged against each device, (3).
#[must_use]
pub fn used_of(pb: &Problem, a: &Assignment) -> Vec<u64> {
    let mut used = vec![0u64; pb.devices.len()];
    let mut hosts = vec![false; pb.devices.len()];
    for (l, c) in pb.layers.iter().zip(&a.layers) {
        used[c.device] += layer_bytes(pb, l, c);
        hosts[c.device] = true;
    }
    for (d, dev) in pb.devices.iter().enumerate() {
        if hosts[d] {
            used[d] += dev.fixed;
        }
    }
    used
}

fn over(pb: &Problem, used: &[u64]) -> Option<usize> {
    // Accelerators first: the fallback being over is not something a move
    // can fix, and it must not mask a card that is.
    (1..pb.devices.len())
        .chain(std::iter::once(0))
        .find(|&d| used[d] > pb.devices[d].capacity)
}

fn outcome(pb: &Problem, assignment: Assignment) -> Outcome {
    let used = used_of(pb, &assignment);
    let feasible = over(pb, &used).is_none();
    Outcome {
        time_s: time_of(pb, &assignment),
        used,
        assignment,
        feasible,
    }
}

/// Mean sensitivity-weighted error, the quality term.
#[must_use]
pub fn error_of(pb: &Problem, a: &Assignment) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (l, c) in pb.layers.iter().zip(&a.layers) {
        for (p, &q) in l.parts.iter().zip(&c.levels) {
            let w = sensitivity(p.kind) * p.params as f64;
            num += w * rel_error(q);
            den += w;
        }
    }
    if den == 0.0 { 0.0 } else { num / den }
}

fn sensitivity(kind: Kind) -> f64 {
    match kind {
        Kind::Attention => 2.0,
        Kind::Ffn | Kind::Fixed => 1.0,
    }
}

/// The best placement from scratch — used at load, and as the target a live
/// placement is compared against in [`replan`].
#[must_use]
pub fn solve(pb: &Problem) -> Outcome {
    let n = pb.layers.len();
    let Some(start) = pb
        .layers
        .iter()
        .map(|l| fastest_choice(l, 0, &pb.devices[0], pb.floor))
        .collect::<Option<Vec<_>>>()
    else {
        // A part the fallback cannot run at any stored level: nothing sane
        // can be placed, and the caller must say so.
        let assignment = Assignment {
            layers: pb
                .layers
                .iter()
                .map(|l| Choice {
                    device: 0,
                    levels: l.parts.iter().map(|p| p.levels[0].0).collect(),
                })
                .collect(),
        };
        let mut o = outcome(pb, assignment);
        o.feasible = false;
        return o;
    };
    let mut a = Assignment { layers: start };

    // 2. Place: the move that saves the most time per destination byte.
    loop {
        let used = used_of(pb, &a);
        let base = time_of(pb, &a);
        let mut best: Option<(f64, usize, Choice)> = None;
        for l in 0..n {
            for d in 1..pb.devices.len() {
                if a.layers[l].device == d {
                    continue;
                }
                let Some(c) = fastest_choice(&pb.layers[l], d, &pb.devices[d], pb.floor) else {
                    continue;
                };
                let add = layer_bytes(pb, &pb.layers[l], &c)
                    + if used_of_device_hosts(&a, d) {
                        0
                    } else {
                        pb.devices[d].fixed
                    };
                if used[d] + add > pb.devices[d].capacity {
                    continue;
                }
                let mut trial = a.clone();
                trial.layers[l] = c.clone();
                let saved = base - time_of(pb, &trial);
                if saved <= 0.0 {
                    continue;
                }
                let ratio = saved / add.max(1) as f64;
                // Ties go to the lower layer: a prefix keeps crossings at one.
                if best.as_ref().is_none_or(|(r, bl, _)| {
                    ratio > *r * (1.0 + 1e-9) || (ratio >= *r * (1.0 - 1e-9) && l < *bl)
                }) {
                    best = Some((ratio, l, c));
                }
            }
        }
        match best {
            Some((_, l, c)) => a.layers[l] = c,
            None => break,
        }
    }

    // 3. Spend: leftover capacity on precision, within the time tolerance.
    spend(pb, &mut a);
    outcome(pb, a)
}

fn used_of_device_hosts(a: &Assignment, d: usize) -> bool {
    a.layers.iter().any(|c| c.device == d)
}

/// Precision upgrades with what no layer move could use: most error removed
/// per byte first, while every device stays within capacity and the total
/// time stays within `tolerance` of where placement left it.
fn spend(pb: &Problem, a: &mut Assignment) {
    let budget_s = time_of(pb, a) * (1.0 + pb.tolerance.max(0.0));
    loop {
        let used = used_of(pb, a);
        let now = time_of(pb, a);
        let mut best: Option<(f64, usize, usize, QuantPolicy)> = None;
        for (li, (layer, c)) in pb.layers.iter().zip(&a.layers).enumerate() {
            let dev = &pb.devices[c.device];
            for (pi, (part, &q)) in layer.parts.iter().zip(&c.levels).enumerate() {
                let Some(up) = q.promote() else { continue };
                if !part.stores(up) || dev.rate_of(up).is_none() {
                    continue;
                }
                let add = dev.occupies(part, up).saturating_sub(dev.occupies(part, q));
                if used[c.device] + add > dev.capacity {
                    continue;
                }
                let dt = part_time(part, dev, up).unwrap_or(f64::INFINITY)
                    - part_time(part, dev, q).unwrap_or(f64::INFINITY);
                if now + dt.max(0.0) > budget_s {
                    continue;
                }
                let gain =
                    sensitivity(part.kind) * (rel_error(q) - rel_error(up)) * part.params as f64;
                if gain <= 0.0 {
                    continue;
                }
                let ratio = gain / add.max(1) as f64;
                if best.as_ref().is_none_or(|(r, ..)| ratio > *r) {
                    best = Some((ratio, li, pi, up));
                }
            }
        }
        match best {
            Some((_, li, pi, up)) => a.layers[li].levels[pi] = up,
            None => break,
        }
    }
}

/// Bytes that change between two placements — what (4) charges for.
#[must_use]
pub fn changed_bytes(pb: &Problem, from: &Assignment, to: &Assignment) -> u64 {
    pb.layers
        .iter()
        .zip(from.layers.iter().zip(&to.layers))
        .map(|(l, (f, t))| {
            if f.device != t.device {
                // The whole layer is re-read on its new device.
                l.parts
                    .iter()
                    .zip(&t.levels)
                    .map(|(p, &q)| part_bytes(p, q))
                    .sum::<u64>()
                    + l.fixed_bytes
            } else {
                l.parts
                    .iter()
                    .zip(f.levels.iter().zip(&t.levels))
                    .filter(|(_, (a, b))| a != b)
                    .map(|(p, (_, &q))| part_bytes(p, q))
                    .sum()
            }
        })
        .sum()
}

/// Why [`replan`] returned what it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The current placement is feasible and nothing better pays for itself.
    Hold,
    /// The current placement no longer fits; this is the least costly fix.
    Repair,
    /// A better placement pays for its re-reads, (4).
    Improve,
}

/// Re-decide a live placement. See the module docs: repair when (3) is
/// violated, improve when (4) holds, hold otherwise.
#[must_use]
pub fn replan(pb: &Problem, current: &Assignment) -> (Verdict, Outcome) {
    let now = outcome(pb, current.clone());
    if !now.feasible {
        return (Verdict::Repair, repair(pb, current.clone()));
    }
    let target = solve(pb);
    if !target.feasible {
        return (Verdict::Hold, now);
    }
    let saved = now.time_s - target.time_s;
    let cost = changed_bytes(pb, current, &target.assignment) as f64 * pb.disk_s_per_byte;
    if saved > 0.0 && saved * pb.horizon_tokens > cost {
        (Verdict::Improve, target)
    } else {
        (Verdict::Hold, now)
    }
}

/// Bring an over-capacity placement back inside (3) with the least time lost
/// per byte freed: demote a part a rung on its device, or move a layer to the
/// fallback at its fastest levels there.
fn repair(pb: &Problem, mut a: Assignment) -> Outcome {
    loop {
        let used = used_of(pb, &a);
        let Some(d) = over(pb, &used) else { break };
        if d == 0 {
            break; // nothing can move off the fallback
        }
        let base = time_of(pb, &a);
        let mut best: Option<(f64, Assignment)> = None;
        for (li, c) in a.layers.iter().enumerate() {
            if c.device != d {
                continue;
            }
            // (a) evict the layer to the fallback.
            if let Some(to) = fastest_choice(&pb.layers[li], 0, &pb.devices[0], pb.floor) {
                let mut trial = a.clone();
                trial.layers[li] = to;
                let freed = used[d] - used_of(pb, &trial)[d];
                let lost = time_of(pb, &trial) - base;
                if freed > 0 {
                    let r = lost / freed as f64;
                    if best.as_ref().is_none_or(|(b, _)| r < *b) {
                        best = Some((r, trial));
                    }
                }
            }
            // (b) demote one part a rung in place.
            for (pi, &q) in c.levels.iter().enumerate() {
                let Some(down) = q.demote() else { continue };
                let part = &pb.layers[li].parts[pi];
                if !level_ok(down, pb.floor)
                    || !part.stores(down)
                    || pb.devices[d].rate_of(down).is_none()
                {
                    continue;
                }
                let mut trial = a.clone();
                trial.layers[li].levels[pi] = down;
                let freed = used[d] - used_of(pb, &trial)[d];
                // Faster at fewer bits is common on a bandwidth-bound device;
                // then only the quality is lost, and it is charged as a tiny
                // time so the cheapest precision goes first.
                let lost = (time_of(pb, &trial) - base).max(0.0)
                    + 1e-12 * sensitivity(part.kind) * (rel_error(down) - rel_error(q));
                if freed > 0 {
                    let r = lost / freed as f64;
                    if best.as_ref().is_none_or(|(b, _)| r < *b) {
                        best = Some((r, trial));
                    }
                }
            }
        }
        match best {
            Some((_, next)) => a = next,
            None => break,
        }
    }
    outcome(pb, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes_at;
    use QuantPolicy::{F16, Off, Q2, Q4, Q8};

    fn stored(params: usize, levels: &[QuantPolicy]) -> Vec<(QuantPolicy, u64)> {
        levels.iter().map(|&q| (q, bytes_at(params, q))).collect()
    }

    const M: usize = 1 << 20;

    fn layer(ffn: usize, attn: usize, state: u64) -> Layer {
        Layer {
            parts: vec![
                Part {
                    params: attn,
                    kind: Kind::Attention,
                    levels: stored(attn, &[Off, F16, Q8, Q4]),
                },
                Part {
                    params: ffn,
                    kind: Kind::Ffn,
                    levels: stored(ffn, &[Off, F16, Q8, Q4]),
                },
            ],
            fixed_bytes: 4096,
            state_bytes: state,
        }
    }

    /// A host that streams every level at DRAM speed, and a card 13x faster
    /// (the ratio production measured: 2.74 ms host vs 0.21 ms card).
    fn host(capacity: u64) -> Device {
        Device {
            capacity,
            fixed: 0,
            rate: vec![(Off, 1e-10), (F16, 1e-10), (Q8, 1e-10), (Q4, 1e-10)],
            resident: Vec::new(),
        }
    }
    fn card(capacity: u64, fixed: u64) -> Device {
        Device {
            capacity,
            fixed,
            rate: vec![(F16, 7.7e-12), (Q8, 7.7e-12), (Q4, 7.7e-12)],
            resident: Vec::new(),
        }
    }

    fn problem(layers: usize, card_cap: u64) -> Problem {
        Problem {
            layers: (0..layers).map(|_| layer(8 * M, 2 * M, 1 << 20)).collect(),
            devices: vec![host(u64::MAX / 4), card(card_cap, 64 << 20)],
            crossing_s: 1e-5,
            disk_s_per_byte: 1.0 / 150e6,
            horizon_tokens: 1000.0,
            tolerance: 0.01,
            floor: Q4,
        }
    }

    fn on_card(a: &Assignment) -> usize {
        a.layers.iter().filter(|c| c.device == 1).count()
    }

    /// With no card room everything runs on the host, at the level the host
    /// is FASTEST at — Q4 here, because it streams the fewest bytes.
    #[test]
    fn everything_on_the_host_when_the_card_is_full() {
        let o = solve(&problem(8, 0));
        assert!(o.feasible);
        assert_eq!(on_card(&o.assignment), 0);
        assert!(o.assignment.layers.iter().all(|c| c.levels == vec![Q4, Q4]));
    }

    /// The host picks per part by its own measured rates: a host whose f16
    /// kernel streams 4x faster per byte than its Q4 kernel keeps f16 for
    /// that part (same time as Q4 at half the bits' error), and a slow f16
    /// host keeps Q4. Nothing about "the host runs integers" is hardcoded.
    #[test]
    fn the_host_level_follows_its_measured_rates() {
        let mut pb = problem(2, 0);
        pb.devices[0].rate = vec![(F16, 1e-10), (Q4, 4e-10)];
        let o = solve(&pb);
        assert!(
            o.assignment
                .layers
                .iter()
                .all(|c| c.levels == vec![F16, F16])
        );
        pb.devices[0].rate = vec![(F16, 1e-10), (Q4, 1e-10)];
        let o = solve(&pb);
        assert!(o.assignment.layers.iter().all(|c| c.levels == vec![Q4, Q4]));
    }

    /// The card takes as many whole layers as fit beside its working set,
    /// as a prefix (one crossing), and the rest stays on the host.
    #[test]
    fn the_card_fills_with_a_prefix_of_layers() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let cap = (64 << 20) + 5 * per + per / 2;
        let o = solve(&problem(12, cap));
        assert!(o.feasible);
        assert_eq!(on_card(&o.assignment), 5);
        assert!(o.assignment.layers[..5].iter().all(|c| c.device == 1));
        assert_eq!(crossings(&o.assignment), 1);
        assert!(o.used[1] <= cap);
    }

    /// When every layer fits, leftover card bytes go to precision — the
    /// attention parts first (they weigh double) — and never past capacity.
    #[test]
    fn leftover_card_capacity_buys_precision_not_speed() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let cap = (64 << 20) + 4 * per + 4 * (bytes_at(2 * M, Q8) - bytes_at(2 * M, Q4));
        let mut pb = problem(4, cap);
        pb.tolerance = 0.25;
        let o = solve(&pb);
        assert_eq!(on_card(&o.assignment), 4);
        assert!(
            o.assignment.layers.iter().all(|c| c.levels[0] == Q8),
            "{:?}",
            o.assignment
        );
        assert!(o.assignment.layers.iter().all(|c| c.levels[1] == Q4));
        assert!(o.used[1] <= cap);
    }

    /// Precision is bought with time only up to the tolerance: at 1% the
    /// same leftover that buys four attention upgrades at 25% buys none,
    /// because each costs more than 1% of this all-card stack's time.
    #[test]
    fn precision_never_costs_more_than_the_tolerance() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let cap = (64 << 20) + 4 * per + 4 * (bytes_at(2 * M, Q8) - bytes_at(2 * M, Q4));
        let pb = problem(4, cap);
        let fastest = solve(&Problem {
            tolerance: 0.0,
            ..pb.clone()
        });
        let o = solve(&pb);
        assert!(o.time_s <= fastest.time_s * (1.0 + pb.tolerance) + 1e-15);
        assert!(o.assignment.layers.iter().all(|c| c.levels == vec![Q4, Q4]));
    }

    /// Memory is charged at what a device actually holds: a card that keeps
    /// Q4 at 3x its pack bytes fits a third as many layers.
    #[test]
    fn capacity_is_charged_at_resident_bytes() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let cap = (64 << 20) + 6 * per;
        assert_eq!(on_card(&solve(&problem(12, cap)).assignment), 6);
        let mut pb = problem(12, cap);
        pb.devices[1].resident = vec![(Q4, 3.0)];
        let o = solve(&pb);
        assert_eq!(on_card(&o.assignment), 2, "{:?}", o.used);
        assert!(o.used[1] <= cap);
    }

    /// A layer whose part the card cannot execute (no rate at any stored
    /// level) stays on the host rather than being placed where it cannot run.
    #[test]
    fn a_part_the_card_cannot_run_keeps_its_layer_home() {
        let mut pb = problem(3, u64::MAX / 4);
        pb.layers[1].parts[1].levels = stored(8 * M, &[Off]);
        let o = solve(&pb);
        assert_eq!(o.assignment.layers[1].device, 0);
        assert_eq!(o.assignment.layers[0].device, 1);
    }

    /// Ambient grew under a live placement: repair frees exactly enough,
    /// first by what costs least time — and never leaves the card over.
    #[test]
    fn repair_brings_an_over_capacity_card_back_inside() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let roomy = solve(&problem(10, (64 << 20) + 8 * per));
        assert_eq!(on_card(&roomy.assignment), 8);
        let tight = problem(10, (64 << 20) + 5 * per);
        let (v, o) = replan(&tight, &roomy.assignment);
        assert_eq!(v, Verdict::Repair);
        assert!(o.feasible);
        assert!(o.used[1] <= tight.devices[1].capacity);
        assert_eq!(on_card(&o.assignment), 5);
    }

    /// Repair demotes precision spent from leftovers before it evicts a
    /// layer: dropping an attention part Q8 -> Q4 loses no time on a
    /// bandwidth-bound card, evicting a layer loses a host layer's time.
    #[test]
    fn repair_gives_back_precision_before_layers() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let up = bytes_at(2 * M, Q8) - bytes_at(2 * M, Q4);
        let mut roomy = problem(4, (64 << 20) + 4 * per + 4 * up);
        roomy.tolerance = 0.25;
        let o = solve(&roomy);
        assert!(o.assignment.layers.iter().all(|c| c.levels[0] == Q8));
        let mut tight = problem(4, (64 << 20) + 4 * per + 2 * up);
        tight.tolerance = 0.25;
        let (v, r) = replan(&tight, &o.assignment);
        assert_eq!(v, Verdict::Repair);
        assert_eq!(on_card(&r.assignment), 4, "{:?}", r.assignment);
        assert!(r.used[1] <= tight.devices[1].capacity);
    }

    /// Freed capacity is taken back only when the saving over the horizon
    /// pays for re-reading the moved bytes, (4).
    #[test]
    fn improvement_must_pay_for_its_disk_reads() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let small = solve(&problem(10, (64 << 20) + 5 * per));
        let mut roomy = problem(10, (64 << 20) + 9 * per);
        roomy.horizon_tokens = 1.0; // one token left: not worth it
        let (v, _) = replan(&roomy, &small.assignment);
        assert_eq!(v, Verdict::Hold);
        roomy.horizon_tokens = 1e6; // a busy server: take it
        let (v, o) = replan(&roomy, &small.assignment);
        assert_eq!(v, Verdict::Improve);
        assert_eq!(on_card(&o.assignment), 9);
    }

    /// Nothing moves when nothing changed: the solver is deterministic, so a
    /// re-plan with the same inputs is a Hold, not churn.
    #[test]
    fn same_inputs_hold() {
        let per = bytes_at(8 * M, Q4) + bytes_at(2 * M, Q4) + 4096 + (1 << 20);
        let pb = problem(10, (64 << 20) + 6 * per);
        let o = solve(&pb);
        let (v, again) = replan(&pb, &o.assignment);
        assert_eq!(v, Verdict::Hold);
        assert_eq!(again.assignment, o.assignment);
    }

    /// The floor holds under any pressure: Q2 is never chosen with a Q4
    /// floor, even though it would fit more on the card.
    #[test]
    fn the_floor_is_never_crossed() {
        let mut pb = problem(6, 64 << 20);
        for l in &mut pb.layers {
            for p in &mut l.parts {
                p.levels.push((Q2, bytes_at(p.params, Q2)));
            }
        }
        pb.devices[1].rate.push((Q2, 7.7e-12));
        let o = solve(&pb);
        assert!(
            o.assignment
                .layers
                .iter()
                .flat_map(|c| &c.levels)
                .all(|&q| q != Q2)
        );
    }
}
