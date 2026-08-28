//! **The residency knapsack: which layers live on the GPU.**
//!
//! Given the VRAM budget the watermark leaves us, choose the set of layers
//! whose weights are resident on the GPU. Per token, layer `l` costs `g[l]`
//! ms decoded on the GPU or `h[l]` ms on the host, and every *boundary*
//! where consecutive layers sit on different devices costs `kappa` ms — the
//! activations cross the bus there, every token. Minimize
//! `sum(chosen costs) + kappa * transitions` subject to the GPU-resident
//! bytes fitting in the budget.
//!
//! [`place`] adds the constraint the serve path actually wants: at most 2
//! transitions along the open chain, which is exactly "the GPU set is one
//! contiguous run" (host prefix -> GPU run -> host suffix; either side may
//! be empty). Contiguity is what keeps prefill's chunked pipeline simple —
//! one upload boundary, one download boundary — and it makes the problem
//! exactly solvable by enumerating the O(L^2) runs with prefix sums, which
//! at L <= 128 is a few thousand additions. No approximation needed.
//!
//! [`place_free`] drops the contiguity constraint so callers can *measure*
//! what it costs: run both, diff the costs. Without a byte budget that
//! would be a chain DP over (layer, device) in O(4L); the budget makes it a
//! knapsack (kappa = 0 IS 0/1 knapsack, which is NP-hard on arbitrary
//! byte weights), so the exact answer here is the chain DP carrying a
//! Pareto frontier of (bytes used, cost) per device state. Dominated pairs
//! are pruned; the frontier size is bounded by the number of distinct
//! reachable byte sums — at most L+1 in the real deployment where every
//! layer weighs the same ~236 MB, exponential only for adversarial weight
//! vectors this repo does not have.

/// The chosen residency, with enough attached to audit it.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    /// `gpu[l]` — layer `l` decodes on the GPU.
    pub gpu: Vec<bool>,
    /// When the GPU set is exactly one contiguous nonempty run, its
    /// inclusive `(first, last)` layer indices; `None` for an empty or
    /// (from [`place_free`]) non-contiguous set.
    pub run: Option<(usize, usize)>,
    /// Total per-token cost: chosen decode costs plus `kappa` per boundary.
    pub cost_ms: f64,
    /// Bytes the GPU-resident weights take; always <= the budget given.
    pub gpu_bytes: u64,
}

/// `kappa * transitions` without the `inf * 0 = NaN` trap: a caller passing
/// `kappa = f64::INFINITY` means "transitions are forbidden", and zero
/// forbidden things still cost nothing.
fn transition_cost(kappa: f64, transitions: u32) -> f64 {
    if transitions == 0 {
        0.0
    } else {
        kappa * f64::from(transitions)
    }
}

/// `Some((first, last))` when the true bits form exactly one contiguous
/// nonempty block.
fn contiguous_run(gpu: &[bool]) -> Option<(usize, usize)> {
    let first = gpu.iter().position(|&b| b)?;
    let last = gpu.iter().rposition(|&b| b)?;
    if gpu[first..=last].iter().all(|&b| b) {
        Some((first, last))
    } else {
        None
    }
}

/// Best placement with the GPU set constrained to one contiguous run
/// (i.e. at most 2 device transitions on the open chain).
///
/// Exact: enumerates every run `(start, end)` — and the empty run — using
/// prefix sums, O(L^2) time, O(L) space. Ties in cost keep the first
/// candidate found (the empty run, then runs in start-then-end order).
///
/// # Panics
/// If `w`, `g`, `h` differ in length. `kappa` may be `f64::INFINITY`
/// (transitions forbidden) but must not be NaN; costs must be finite.
#[must_use]
pub fn place(w: &[u64], g: &[f64], h: &[f64], kappa: f64, budget: u64) -> Placement {
    let l = w.len();
    assert_eq!(g.len(), l, "place: g must have one entry per layer");
    assert_eq!(h.len(), l, "place: h must have one entry per layer");
    debug_assert!(!kappa.is_nan(), "place: kappa must not be NaN");

    // Prefix sums; bytes in u128 so a hostile u64 weight vector cannot
    // overflow the accumulation (the chosen run itself must fit in the
    // u64 budget, so the reported bytes always fit back into u64).
    let mut pw = vec![0u128; l + 1];
    let mut pg = vec![0f64; l + 1];
    let mut ph = vec![0f64; l + 1];
    for i in 0..l {
        pw[i + 1] = pw[i] + u128::from(w[i]);
        pg[i + 1] = pg[i] + g[i];
        ph[i + 1] = ph[i] + h[i];
    }
    let total_h = ph[l];

    // The empty run is always feasible: everything on the host, no
    // transitions, no bytes.
    let mut best_cost = total_h;
    let mut best_run: Option<(usize, usize)> = None;
    let mut best_bytes = 0u64;

    for s in 0..l {
        for e in s..l {
            let bytes = pw[e + 1] - pw[s];
            if bytes > u128::from(budget) {
                break; // bytes only grow with e; the rest of the row is out
            }
            let transitions = u32::from(s > 0) + u32::from(e + 1 < l);
            let cost = (total_h - (ph[e + 1] - ph[s]))
                + (pg[e + 1] - pg[s])
                + transition_cost(kappa, transitions);
            if cost < best_cost {
                best_cost = cost;
                best_run = Some((s, e));
                best_bytes = bytes as u64;
            }
        }
    }

    let mut gpu = vec![false; l];
    if let Some((s, e)) = best_run {
        gpu[s..=e].iter_mut().for_each(|b| *b = true);
    }
    Placement {
        gpu,
        run: best_run,
        cost_ms: best_cost,
        gpu_bytes: best_bytes,
    }
}

/// One partial assignment surviving the Pareto pruning: which layers so
/// far are on the GPU (`mask`), the bytes that costs, and the accumulated
/// decode + transition cost.
#[derive(Clone)]
struct Entry {
    bytes: u128,
    cost: f64,
    mask: u128,
}

/// Drop dominated entries: sort by bytes, keep only strictly improving
/// costs. Two entries with equal bytes keep the cheaper.
fn prune(mut v: Vec<Entry>) -> Vec<Entry> {
    v.sort_by(|a, b| {
        a.bytes.cmp(&b.bytes).then(
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    let mut out: Vec<Entry> = Vec::with_capacity(v.len());
    let mut best = f64::INFINITY;
    for e in v {
        if e.cost < best {
            best = e.cost;
            out.push(e);
        }
    }
    out
}

/// Best placement with transitions unlimited — the relaxation of [`place`],
/// so `place_free(..).cost_ms <= place(..).cost_ms` always, and the gap is
/// the measured price of contiguity.
///
/// Chain DP over (layer, device) states, each carrying a Pareto frontier
/// of (GPU bytes used, cost); see the module doc for why the frontier is
/// necessary (the byte budget makes this a knapsack) and why it stays
/// small in practice.
///
/// # Panics
/// If slice lengths differ, or `w.len() > 128` (assignments are
/// reconstructed through a u128 bitmask; the deployment ceiling in this
/// repo is 128 layers).
#[must_use]
pub fn place_free(w: &[u64], g: &[f64], h: &[f64], kappa: f64, budget: u64) -> Placement {
    let l = w.len();
    assert_eq!(g.len(), l, "place_free: g must have one entry per layer");
    assert_eq!(h.len(), l, "place_free: h must have one entry per layer");
    assert!(l <= 128, "place_free: at most 128 layers (u128 mask)");
    debug_assert!(!kappa.is_nan(), "place_free: kappa must not be NaN");

    if l == 0 {
        return Placement {
            gpu: vec![],
            run: None,
            cost_ms: 0.0,
            gpu_bytes: 0,
        };
    }

    let budget = u128::from(budget);

    // Frontiers after placing layer 0. Starting on either device costs no
    // transition — consistent with `place`, which charges boundaries
    // between layers, not the chain's open ends.
    let mut host = vec![Entry {
        bytes: 0,
        cost: h[0],
        mask: 0,
    }];
    let mut gpu = if u128::from(w[0]) <= budget {
        vec![Entry {
            bytes: u128::from(w[0]),
            cost: g[0],
            mask: 1,
        }]
    } else {
        vec![]
    };

    for i in 1..l {
        let wl = u128::from(w[i]);
        let bit = 1u128 << i;

        let mut next_host = Vec::with_capacity(host.len() + gpu.len());
        for e in &host {
            next_host.push(Entry {
                bytes: e.bytes,
                cost: e.cost + h[i],
                mask: e.mask,
            });
        }
        for e in &gpu {
            next_host.push(Entry {
                bytes: e.bytes,
                cost: e.cost + h[i] + kappa,
                mask: e.mask,
            });
        }

        let mut next_gpu = Vec::with_capacity(host.len() + gpu.len());
        for e in &gpu {
            if e.bytes + wl <= budget {
                next_gpu.push(Entry {
                    bytes: e.bytes + wl,
                    cost: e.cost + g[i],
                    mask: e.mask | bit,
                });
            }
        }
        for e in &host {
            if e.bytes + wl <= budget {
                next_gpu.push(Entry {
                    bytes: e.bytes + wl,
                    cost: e.cost + g[i] + kappa,
                    mask: e.mask | bit,
                });
            }
        }

        host = prune(next_host);
        gpu = prune(next_gpu);
    }

    // The all-host chain is always alive, so a best entry always exists.
    let best = host
        .iter()
        .chain(gpu.iter())
        .min_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("the all-host assignment is always feasible");

    let gpu_mask: Vec<bool> = (0..l).map(|i| best.mask >> i & 1 == 1).collect();
    let gpu_bytes = gpu_mask
        .iter()
        .zip(w)
        .filter(|&(&on, _)| on)
        .map(|(_, &b)| b)
        .sum();
    Placement {
        run: contiguous_run(&gpu_mask),
        gpu: gpu_mask,
        cost_ms: best.cost,
        gpu_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same 10-line PCG32 as the sibling modules — self-contained on
    /// purpose in a dependency-free crate.
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

    /// A random instance shaped like the real one: weights around the
    /// 27B's ~236 MB/layer (jittered so byte sums are distinct and the
    /// Pareto frontier is exercised), GPU decode ~10x faster than host,
    /// budget anywhere from tight to loose.
    fn instance(seed: u64, l: usize) -> (Vec<u64>, Vec<f64>, Vec<f64>, f64, u64) {
        let mut r = Pcg::new(seed);
        let w: Vec<u64> = (0..l)
            .map(|_| 150_000_000 + (u64::from(r.next_u32()) % 200_000_000))
            .collect();
        let g: Vec<f64> = (0..l).map(|_| 0.5 + 1.5 * r.uniform()).collect();
        let h: Vec<f64> = (0..l).map(|_| 4.0 + 12.0 * r.uniform()).collect();
        let kappa = 0.2 + 3.0 * r.uniform();
        let total: u64 = w.iter().sum();
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let budget = (total as f64 * (0.1 + 1.1 * r.uniform())) as u64;
        (w, g, h, kappa, budget)
    }

    /// Direct cost of an arbitrary assignment, computed the slow honest
    /// way — per layer, no prefix sums — so it shares no code with the
    /// implementations it checks.
    fn cost_of(mask: &[bool], g: &[f64], h: &[f64], kappa: f64) -> f64 {
        let decode: f64 = mask
            .iter()
            .enumerate()
            .map(|(i, &on)| if on { g[i] } else { h[i] })
            .sum();
        let transitions = mask.windows(2).filter(|p| p[0] != p[1]).count() as u32;
        decode + transition_cost(kappa, transitions)
    }

    fn bytes_of(mask: &[bool], w: &[u64]) -> u64 {
        mask.iter()
            .zip(w)
            .filter(|&(&on, _)| on)
            .map(|(_, &b)| b)
            .sum()
    }

    /// place() against brute force over every contiguous run (plus the
    /// empty run) on random instances.
    #[test]
    fn place_matches_contiguous_brute_force() {
        for seed in 0..25 {
            let l = 12;
            let (w, g, h, kappa, budget) = instance(seed, l);
            let got = place(&w, &g, &h, kappa, budget);

            let mut best = h.iter().sum::<f64>(); // empty run
            for s in 0..l {
                for e in s..l {
                    let mut mask = vec![false; l];
                    mask[s..=e].iter_mut().for_each(|b| *b = true);
                    if bytes_of(&mask, &w) <= budget {
                        best = best.min(cost_of(&mask, &g, &h, kappa));
                    }
                }
            }
            assert!(
                (got.cost_ms - best).abs() < 1e-9,
                "seed {seed}: place {} vs brute {best}",
                got.cost_ms
            );
            // The returned artifacts must describe the returned cost.
            assert!(got.gpu_bytes <= budget);
            assert!((cost_of(&got.gpu, &g, &h, kappa) - got.cost_ms).abs() < 1e-9);
            match got.run {
                Some((s, e)) => assert!(got.gpu[s..=e].iter().all(|&b| b)),
                None => assert!(got.gpu.iter().all(|&b| !b)),
            }
        }
    }

    /// place_free() against brute force over ALL 2^L assignments under the
    /// budget — the frontier DP must be exact, not just Pareto-plausible.
    #[test]
    fn place_free_matches_exhaustive_brute_force() {
        for seed in 100..115 {
            let l = 12;
            let (w, g, h, kappa, budget) = instance(seed, l);
            let got = place_free(&w, &g, &h, kappa, budget);

            let mut best = f64::INFINITY;
            for bits in 0u32..(1 << l) {
                let mask: Vec<bool> = (0..l).map(|i| bits >> i & 1 == 1).collect();
                if bytes_of(&mask, &w) <= budget {
                    best = best.min(cost_of(&mask, &g, &h, kappa));
                }
            }
            assert!(
                (got.cost_ms - best).abs() < 1e-9,
                "seed {seed}: place_free {} vs brute {best}",
                got.cost_ms
            );
            assert!(got.gpu_bytes <= budget);
            assert!((cost_of(&got.gpu, &g, &h, kappa) - got.cost_ms).abs() < 1e-9);

            // And the relaxation ordering that makes the pair useful:
            let constrained = place(&w, &g, &h, kappa, budget);
            assert!(
                got.cost_ms <= constrained.cost_ms + 1e-9,
                "seed {seed}: dropping a constraint cannot cost more"
            );
        }
    }

    /// kappa = infinity forbids transitions outright: the only candidates
    /// left are all-host and all-GPU, for both solvers, with no NaN leaking
    /// out of an inf * 0.
    #[test]
    fn infinite_kappa_degenerates_to_all_or_nothing() {
        let w = vec![10u64; 6];
        let g = vec![1.0; 6];
        let h = vec![5.0; 6];

        // Budget fits everything: all-GPU wins (6*1 < 6*5).
        let all = place(&w, &g, &h, f64::INFINITY, 60);
        assert_eq!(all.run, Some((0, 5)));
        assert!((all.cost_ms - 6.0).abs() < 1e-12);
        let all_free = place_free(&w, &g, &h, f64::INFINITY, 60);
        assert_eq!(all_free.gpu, vec![true; 6]);
        assert!((all_free.cost_ms - 6.0).abs() < 1e-12);

        // Budget fits only some: a partial run would need a boundary, so
        // everything stays on the host — finite cost, not NaN or inf.
        let none = place(&w, &g, &h, f64::INFINITY, 30);
        assert_eq!(none.run, None);
        assert!((none.cost_ms - 30.0).abs() < 1e-12);
        let none_free = place_free(&w, &g, &h, f64::INFINITY, 30);
        assert_eq!(none_free.gpu, vec![false; 6]);
        assert!((none_free.cost_ms - 30.0).abs() < 1e-12);
    }

    /// Zero budget: nothing fits, everything on the host, zero bytes.
    #[test]
    fn zero_budget_puts_everything_on_the_host() {
        let (w, g, h, kappa, _) = instance(42, 10);
        for placement in [
            place(&w, &g, &h, kappa, 0),
            place_free(&w, &g, &h, kappa, 0),
        ] {
            assert!(placement.gpu.iter().all(|&b| !b));
            assert_eq!(placement.run, None);
            assert_eq!(placement.gpu_bytes, 0);
            assert!((placement.cost_ms - h.iter().sum::<f64>()).abs() < 1e-9);
        }
    }

    /// The run does not have to touch either end: when the middle layers
    /// are the ones worth accelerating, the sandwich (2 transitions) is
    /// chosen and charged for both boundaries.
    #[test]
    fn a_middle_run_pays_both_boundaries() {
        let w = vec![1u64; 5];
        let g = vec![10.0, 0.1, 0.1, 0.1, 10.0];
        let h = vec![1.0, 10.0, 10.0, 10.0, 1.0];
        let kappa = 0.5;
        let got = place(&w, &g, &h, kappa, 3);
        assert_eq!(got.run, Some((1, 3)));
        // h[0] + h[4] + g[1..=3] + 2 boundaries.
        assert!((got.cost_ms - (1.0 + 1.0 + 0.3 + 2.0 * kappa)).abs() < 1e-12);
        assert_eq!(got.gpu_bytes, 3);
    }

    /// Zero layers is a legal (empty) model: nothing to place, zero cost.
    #[test]
    fn empty_chain_is_the_empty_placement() {
        for placement in [
            place(&[], &[], &[], 1.0, 100),
            place_free(&[], &[], &[], 1.0, 100),
        ] {
            assert_eq!(placement.gpu, Vec::<bool>::new());
            assert_eq!(placement.run, None);
            assert_eq!(placement.cost_ms, 0.0);
            assert_eq!(placement.gpu_bytes, 0);
        }
    }
}
