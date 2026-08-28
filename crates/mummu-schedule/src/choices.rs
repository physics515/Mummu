//! **The multiple-choice residency knapsack: device x precision x KV bits,
//! per item, under one VRAM budget (SPEC P2.4/P2.5) — with the lm_head as a
//! first-class item (SPEC P4.1).**
//!
//! [`placement::place`](crate::placement) answers "which contiguous run of
//! layers lives on the GPU" with every layer offered exactly two shapes
//! (host or GPU at fixed precision). This module answers the finer question
//! the VRAM work opens up: each *item* (a layer, the lm_head, a KV pool)
//! offers several **options** — (GPU @ Q4), (GPU @ F16), (host), (GPU with
//! f16 KV), … — each with its own VRAM bytes and per-token cost, and the
//! plan picks exactly one option per item to minimize total cost under the
//! byte budget.
//!
//! Solved exactly by DP over the budget discretized to a granularity `G`
//! (1 MiB in practice): `F[i][b] = min over options o of item i:
//! cost(o) + F[i+1][b - bytes(o)]`, with bytes rounded UP to `G` so the
//! discretized plan never overspends the true budget. Complexity
//! `O(items x options x budget/G)` — a 64-layer model on a 16 GiB card at
//! 1 MiB is ~3M adds, microseconds at plan time.
//!
//! The head-admission calculus is the special case worth naming
//! ([`admit_head`]): the head's density (ms saved per byte) is compared
//! against the *marginal* layers it would displace — the exact arithmetic
//! that today's measured numbers put at "68 ms / 0.7 GiB ≈ 97 ms/GiB vs a
//! host layer's ~22 ms/GiB, so the head outranks ~4 layers the moment the
//! bytes exist" — so the decision is a computation, not a rule of thumb.

/// One way an item can be realized: `bytes` of the shared budget (0 for a
/// host-resident option), `cost_ms` per token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    pub bytes: u64,
    pub cost_ms: f64,
}

/// One placeable item and its options. Every item must offer at least one
/// zero-byte option (typically "host") so the DP is always feasible.
#[derive(Debug, Clone)]
pub struct Item {
    /// A short label carried through for the audit trail ("layer 17",
    /// "lm_head", "kv@4096").
    pub name: String,
    pub options: Vec<Choice>,
}

/// The chosen plan: one option index per item, and the totals to audit it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoicePlan {
    /// `chosen[i]` indexes `items[i].options`.
    pub chosen: Vec<usize>,
    pub cost_ms: f64,
    pub bytes: u64,
}

/// Exact multiple-choice knapsack over budget granules of `granularity`
/// bytes. Bytes of every option are rounded UP to the granule, so the
/// returned plan's true byte total is `<= budget` whenever its granule
/// total is — the discretization can only *waste* budget, never overdraw
/// it. Cost ties keep the lower option index.
///
/// # Panics
/// If any item has no feasible zero-or-affordable option (every item needs
/// a fallback — typically host at 0 bytes), if `granularity == 0`, or if a
/// cost is not finite.
#[must_use]
pub fn place_choices(items: &[Item], budget: u64, granularity: u64) -> ChoicePlan {
    assert!(granularity > 0, "place_choices: zero granularity");
    let granules = usize::try_from(budget / granularity).expect("budget fits usize") + 1;
    let n = items.len();
    for it in items {
        assert!(!it.options.is_empty(), "item {} has no options", it.name);
        for o in &it.options {
            assert!(
                o.cost_ms.is_finite(),
                "item {}: non-finite cost {}",
                it.name,
                o.cost_ms
            );
        }
    }
    // Option bytes in granules, rounded up.
    let gran_of = |bytes: u64| -> usize {
        usize::try_from(bytes.div_ceil(granularity)).expect("option bytes fit usize")
    };

    // F[i][b] = best cost for items i.. with b granules available.
    // Iterate backward so chosen[] reconstructs forward.
    let mut next = vec![0.0f64; granules];
    let mut pick: Vec<Vec<u16>> = vec![vec![0; granules]; n];
    for i in (0..n).rev() {
        let mut cur = vec![f64::INFINITY; granules];
        for b in 0..granules {
            for (oi, o) in items[i].options.iter().enumerate() {
                let g = gran_of(o.bytes);
                if g <= b {
                    let c = o.cost_ms + next[b - g];
                    if c < cur[b] {
                        cur[b] = c;
                        pick[i][b] = u16::try_from(oi).expect("option index fits u16");
                    }
                }
            }
        }
        assert!(
            cur[granules - 1].is_finite(),
            "item {} has no option affordable even with the full budget — \
             every item needs a zero-byte fallback",
            items[i].name
        );
        next = cur;
    }

    // Reconstruct.
    let mut chosen = Vec::with_capacity(n);
    let mut b = granules - 1;
    let mut cost = 0.0f64;
    let mut bytes = 0u64;
    for (i, item) in items.iter().enumerate() {
        let oi = usize::from(pick[i][b]);
        let o = item.options[oi];
        chosen.push(oi);
        cost += o.cost_ms;
        bytes += o.bytes;
        b -= gran_of(o.bytes);
    }
    debug_assert!(bytes <= budget, "granule rounding must not overdraw");
    ChoicePlan {
        chosen,
        cost_ms: cost,
        bytes,
    }
}

/// The head-admission calculus (SPEC P4.1): with `budget_free` bytes newly
/// available, is pinning the head worth more than spending the same bytes
/// on more layers?
///
/// `head_saving_ms` = host head cost minus GPU head cost; `head_bytes` its
/// VRAM price. `layer_savings` are the per-layer `(saving_ms, bytes)` of
/// the layers NOT yet resident, best-first order not required. The answer
/// compares the head against the best same-byte bundle of layers —
/// exactly `sum of displaced layer savings < head saving` from the spec,
/// with the bundle chosen greedily by density (optimal here because
/// layers are near-uniform in size; the exact DP above is the arbiter
/// when they are not).
#[must_use]
pub fn admit_head(
    head_saving_ms: f64,
    head_bytes: u64,
    layer_savings: &[(f64, u64)],
    budget_free: u64,
) -> bool {
    if head_bytes > budget_free {
        return false;
    }
    // Best layer bundle in the SAME bytes the head would take, by density.
    let mut layers: Vec<(f64, u64)> = layer_savings
        .iter()
        .copied()
        .filter(|&(s, b)| s > 0.0 && b > 0)
        .collect();
    layers.sort_by(|a, b| {
        let da = a.0 / a.1 as f64;
        let db = b.0 / b.1 as f64;
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut bundle = 0.0f64;
    let mut used = 0u64;
    for (s, b) in layers {
        if used + b <= head_bytes {
            bundle += s;
            used += b;
        }
    }
    head_saving_ms > bundle
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(name: &str, options: &[(u64, f64)]) -> Item {
        Item {
            name: name.into(),
            options: options
                .iter()
                .map(|&(bytes, cost_ms)| Choice { bytes, cost_ms })
                .collect(),
        }
    }

    /// Exhaustive check over every option product on small instances.
    #[test]
    fn dp_matches_brute_force() {
        // Deterministic pseudo-random instances (self-contained PCG).
        let mut state = 0x853c_49e6_748f_ea9bu64;
        let mut next = move || {
            let old = state;
            state = old
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((old >> 33) as u32) as u64
        };
        for trial in 0..40 {
            let n = 1 + (next() % 5) as usize;
            let items: Vec<Item> = (0..n)
                .map(|i| {
                    let opts = 1 + (next() % 3) as usize;
                    let mut options: Vec<(u64, f64)> = (0..opts)
                        .map(|_| ((next() % 8) * 10, (next() % 100) as f64 / 7.0))
                        .collect();
                    options.push((0, (next() % 200) as f64 / 7.0)); // host fallback
                    item(&format!("i{i}"), &options)
                })
                .collect();
            let budget = next() % 200;
            let granularity = 10;
            let got = place_choices(&items, budget, granularity);

            // Brute force over all products.
            let mut best = f64::INFINITY;
            let mut idx = vec![0usize; n];
            loop {
                let bytes: u64 = idx
                    .iter()
                    .enumerate()
                    .map(|(i, &o)| items[i].options[o].bytes.div_ceil(granularity) * granularity)
                    .sum();
                if bytes <= budget / granularity * granularity + (budget % granularity) {
                    // Same discretization as the DP: granule totals within
                    // the granule budget.
                    let gran: u64 = idx
                        .iter()
                        .enumerate()
                        .map(|(i, &o)| items[i].options[o].bytes.div_ceil(granularity))
                        .sum();
                    if gran <= budget / granularity {
                        let cost: f64 = idx
                            .iter()
                            .enumerate()
                            .map(|(i, &o)| items[i].options[o].cost_ms)
                            .sum();
                        best = best.min(cost);
                    }
                }
                // Odometer.
                let mut place = 0;
                loop {
                    if place == n {
                        break;
                    }
                    idx[place] += 1;
                    if idx[place] < items[place].options.len() {
                        break;
                    }
                    idx[place] = 0;
                    place += 1;
                }
                if place == n {
                    break;
                }
            }
            assert!(
                (got.cost_ms - best).abs() < 1e-9,
                "trial {trial}: dp {} vs brute {best}",
                got.cost_ms
            );
            assert!(got.bytes <= budget);
        }
    }

    /// The audit fields must describe the chosen plan.
    #[test]
    fn plan_totals_are_consistent() {
        let items = vec![
            item("layer0", &[(100, 1.0), (0, 5.0)]),
            item("layer1", &[(100, 1.0), (0, 5.0)]),
            item("head", &[(50, 0.5), (0, 4.0)]),
        ];
        let plan = place_choices(&items, 150, 10);
        let cost: f64 = plan
            .chosen
            .iter()
            .enumerate()
            .map(|(i, &o)| items[i].options[o].cost_ms)
            .sum();
        let bytes: u64 = plan
            .chosen
            .iter()
            .enumerate()
            .map(|(i, &o)| items[i].options[o].bytes)
            .sum();
        assert!((plan.cost_ms - cost).abs() < 1e-12);
        assert_eq!(plan.bytes, bytes);
        assert!(plan.bytes <= 150);
        // 150 bytes cannot hold both layers (200), so the frontier is one
        // layer + the head (1 + 5 + 0.5 = 6.5) vs one layer + host head
        // (1 + 5 + 4 = 10) vs head only (5 + 5 + 0.5 = 10.5): 6.5 wins.
        assert!((plan.cost_ms - 6.5).abs() < 1e-9);
    }

    /// The spec's own numbers: head 68 ms at 0.7 GiB (~97 ms/GiB) against
    /// host layers at ~26 ms per ~1.2 GiB (~22 ms/GiB). In 0.7 GiB the
    /// best displaced bundle is ZERO whole layers (each needs 1.2 GiB), so
    /// the head is admitted the moment the bytes exist.
    #[test]
    fn head_admission_matches_the_spec_arithmetic() {
        let gib = 1u64 << 30;
        let layers: Vec<(f64, u64)> = (0..4).map(|_| (26.0, gib + gib / 5)).collect();
        assert!(admit_head(68.0, gib * 7 / 10, &layers, gib));
        // No free bytes: not admissible.
        assert!(!admit_head(68.0, gib * 7 / 10, &layers, gib / 2));
        // Against smaller layers that DO fit in the head's bytes, the
        // bundle can win: three 0.2-GiB layers at 26 ms each beat 68 ms.
        let small: Vec<(f64, u64)> = (0..4).map(|_| (26.0, gib / 5)).collect();
        assert!(!admit_head(68.0, gib * 7 / 10, &small, gib));
    }

    /// A model-shaped instance: the DP prefers spending marginal bytes on
    /// the head once layers are dense enough on the card that the next
    /// layer's saving is below the head's.
    #[test]
    fn dp_admits_the_head_before_the_marginal_layer() {
        let mib = 1u64 << 20;
        let mut items: Vec<Item> = (0..4)
            .map(|i| {
                item(
                    &format!("layer{i}"),
                    &[(1200 * mib, 2.0), (0, 28.0)], // GPU saves 26 ms
                )
            })
            .collect();
        items.push(item("head", &[(700 * mib, 1.0), (0, 69.0)])); // saves 68 ms
        // Budget fits three layers OR two layers + the head (+ slack).
        let plan = place_choices(&items, 3700 * mib, mib);
        let head_choice = plan.chosen[4];
        assert_eq!(head_choice, 0, "the head must be admitted: {plan:?}");
        // And exactly two layers fit beside it.
        let gpu_layers = plan.chosen[..4].iter().filter(|&&c| c == 0).count();
        assert_eq!(gpu_layers, 2);
    }
}
