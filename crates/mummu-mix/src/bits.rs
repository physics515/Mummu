//! **Reverse water-filling: spend a bit budget where the model is
//! sensitive (SPEC P2.4/P2.5).**
//!
//! The high-rate quantizer model prices distortion at `D_l(b) = c_l * N_l *
//! 2^(-2b)`: each extra bit on tensor `l` quarters its distortion, weighted
//! by a measured sensitivity `c_l` (the GPTQ-style second-order proxy
//! `tr(H_l . d d^T)`, or this repo's ladder-probe relative errors) and its
//! parameter count `N_l`. Minimizing total distortion under a total-bit
//! budget `sum N_l b_l <= B` has the classic reverse water-filling solution
//!
//! ```text
//! b_l* = b_bar + (1/2) log2( c_l / gm(c) ),   gm = N-weighted geo-mean
//! ```
//!
//! — every tensor gets the average bits plus half a log of how much more
//! sensitive it is than the geometric mean, clamped at zero (a tensor can't
//! store negative bits; the clamp re-waters the freed budget over the rest,
//! exactly the water level rising).
//!
//! Two consumers:
//! - [`waterfill_bits`] — the continuous optimum, the planning target.
//! - [`allocate_bits_integer`] — whole bits by greedy marginal-gain, which
//!   is *optimal* for this objective (the gain sequence per tensor is
//!   strictly decreasing in b, so the greedy exchange argument closes) and
//!   pinned against brute force in the tests.
//!
//! And the checklist number: [`amgm_ratio`], the predicted uniform-vs-mixed
//! gain `J_uniform / J* = am(c)/gm(c)`. If it is under ~1.2, mixed
//! precision cannot pay for its complexity on this model — skip it (the
//! RateQuant checklist, adopted as arithmetic).

/// Continuous reverse water-filling under `sum N_l b_l <= total_bits`,
/// `b_l >= 0`. Returns per-tensor bits (fractional — the planning target,
/// not a storable format).
///
/// # Panics
/// On empty input, nonpositive sensitivities/counts, or a nonpositive
/// budget.
#[must_use]
pub fn waterfill_bits(sensitivity: &[f64], counts: &[u64], total_bits: f64) -> Vec<f64> {
    let n = sensitivity.len();
    assert!(n > 0, "waterfill_bits: no tensors");
    assert_eq!(counts.len(), n, "waterfill_bits: counts length");
    assert!(
        total_bits > 0.0 && total_bits.is_finite(),
        "waterfill_bits: budget must be positive"
    );
    for (&c, &m) in sensitivity.iter().zip(counts) {
        assert!(c > 0.0 && c.is_finite(), "sensitivity must be positive");
        assert!(m > 0, "counts must be positive");
    }

    // Active set iteration: solve on the active tensors, clamp negatives to
    // zero, repeat. Terminates in <= n rounds (each round permanently
    // removes at least one tensor).
    let mut active: Vec<usize> = (0..n).collect();
    let mut bits = vec![0.0f64; n];
    loop {
        let total_n: f64 = active.iter().map(|&i| counts[i] as f64).sum();
        let b_bar = total_bits / total_n;
        // N-weighted geometric mean of sensitivities over the active set.
        let log_gm: f64 = active
            .iter()
            .map(|&i| counts[i] as f64 * sensitivity[i].ln())
            .sum::<f64>()
            / total_n;
        let mut any_negative = false;
        for &i in &active {
            let b = b_bar + 0.5 * (sensitivity[i].ln() - log_gm) / std::f64::consts::LN_2;
            bits[i] = b;
            if b < 0.0 {
                any_negative = true;
            }
        }
        if !any_negative {
            break;
        }
        // Clamp the negatives out and re-water the rest.
        for &i in &active {
            if bits[i] < 0.0 {
                bits[i] = 0.0;
            }
        }
        let next: Vec<usize> = active.iter().copied().filter(|&i| bits[i] > 0.0).collect();
        if next.len() == active.len() || next.is_empty() {
            break;
        }
        active = next;
    }
    bits
}

/// Total distortion of an integer allocation under the model.
#[must_use]
pub fn distortion(sensitivity: &[f64], counts: &[u64], bits: &[u32]) -> f64 {
    sensitivity
        .iter()
        .zip(counts)
        .zip(bits)
        .map(|((&c, &m), &b)| c * m as f64 * (-2.0 * f64::from(b)).exp2())
        .sum()
}

/// Whole-bit allocation by greedy marginal gain: start every tensor at
/// `min_bits`, then, while budget remains, give one more bit to the tensor
/// whose distortion drop per budget bit is largest (`c_l N_l (2^-2b -
/// 2^-2(b+1)) / N_l = c_l * 3/4 * 2^-2b` — counts cancel in the density, so
/// the rule is "most sensitive at its current precision wins").
///
/// Optimality is conditional, and the tests hold the claim to exactly its
/// shape: with **equal parameter counts** every step costs the same budget
/// and the strictly-decreasing gain sequences make the greedy exchange
/// argument close — pinned against brute force below. With heterogeneous
/// counts this is a knapsack and greedy is a (good) heuristic: the
/// leftover-budget effect is bounded by one marginal step, and the test
/// pins that bound too. Real allocations run at tensor granularity where
/// budgets are billions of bits — the continuous [`waterfill_bits`] is the
/// planning optimum and this rounds it; an exact DP over such budgets is
/// not a real option.
///
/// `bit_budget` is total bits ON TOP of the `min_bits` floor
/// (`sum N_l * (b_l - min_bits) <= bit_budget`); `max_bits` caps any tensor.
#[must_use]
pub fn allocate_bits_integer(
    sensitivity: &[f64],
    counts: &[u64],
    bit_budget: u64,
    min_bits: u32,
    max_bits: u32,
) -> Vec<u32> {
    let n = sensitivity.len();
    assert_eq!(counts.len(), n);
    assert!(min_bits <= max_bits);
    let mut bits = vec![min_bits; n];
    let mut left = bit_budget;
    loop {
        // Highest marginal gain per budget bit among tensors that can still
        // grow AND fit the remaining budget.
        let mut best: Option<(usize, f64)> = None;
        for i in 0..n {
            if bits[i] >= max_bits || counts[i] > left {
                continue;
            }
            let gain = sensitivity[i] * 0.75 * (-2.0 * f64::from(bits[i])).exp2();
            if best.is_none_or(|(_, g)| gain > g) {
                best = Some((i, gain));
            }
        }
        let Some((i, _)) = best else { break };
        bits[i] += 1;
        left -= counts[i];
    }
    bits
}

/// Predicted gain of mixed precision over uniform at the same budget:
/// `J_uniform / J* = am(c) / gm(c)` (N-weighted), which is `>= 1` by
/// AM/GM with equality iff every sensitivity is equal. Under ~1.2 the
/// spread does not pay for per-tensor bit plumbing — skip mixed precision.
#[must_use]
pub fn amgm_ratio(sensitivity: &[f64], counts: &[u64]) -> f64 {
    assert!(!sensitivity.is_empty());
    assert_eq!(sensitivity.len(), counts.len());
    let total: f64 = counts.iter().map(|&m| m as f64).sum();
    let am: f64 = sensitivity
        .iter()
        .zip(counts)
        .map(|(&c, &m)| c * m as f64)
        .sum::<f64>()
        / total;
    let log_gm: f64 = sensitivity
        .iter()
        .zip(counts)
        .map(|(&c, &m)| m as f64 * c.ln())
        .sum::<f64>()
        / total;
    am / log_gm.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The continuous optimum satisfies its KKT structure: equal marginal
    /// distortion per bit across active tensors, and the budget is spent
    /// exactly.
    #[test]
    fn continuous_waterfill_is_kkt() {
        let c = [8.0, 2.0, 0.5];
        let n = [100u64, 100, 100];
        let budget = 1200.0; // b_bar = 4
        let b = waterfill_bits(&c, &n, budget);
        // Budget exact.
        let spent: f64 = b.iter().zip(&n).map(|(bi, &ni)| bi * ni as f64).sum();
        assert!((spent - budget).abs() < 1e-9);
        // b_i - b_j = 0.5 log2(c_i / c_j).
        assert!((b[0] - b[1] - 0.5 * (8.0f64 / 2.0).log2()).abs() < 1e-9);
        assert!((b[1] - b[2] - 0.5 * (2.0f64 / 0.5).log2()).abs() < 1e-9);
        // Marginal distortion c_i 2^{-2 b_i} equal across active tensors.
        let m0 = c[0] * (-2.0 * b[0]).exp2();
        for i in 1..3 {
            let mi = c[i] * (-2.0 * b[i]).exp2();
            assert!((mi - m0).abs() / m0 < 1e-9);
        }
    }

    /// A tensor whose closed-form bits go negative is clamped to zero and
    /// its budget re-waters the rest.
    #[test]
    fn clamp_rewaters_the_rest() {
        let c = [1000.0, 1000.0, 1e-12];
        let n = [10u64, 10, 10];
        let b = waterfill_bits(&c, &n, 60.0);
        assert_eq!(b[2], 0.0, "insensitive tensor must be dropped to zero");
        let spent: f64 = b.iter().zip(&n).map(|(bi, &ni)| bi * ni as f64).sum();
        assert!((spent - 60.0).abs() < 1e-9, "freed budget must be re-spent");
        assert!((b[0] - b[1]).abs() < 1e-9);
    }

    /// Brute-force minimum over all allocations in `[lo, hi]^n` within
    /// budget — the honest reference both greedy tests compare against.
    fn brute_min(c: &[f64], counts: &[u64], budget: u64, lo: u32, hi: u32) -> f64 {
        let n = c.len();
        let mut best = f64::INFINITY;
        let mut alloc = vec![lo; n];
        loop {
            let extra: u64 = alloc
                .iter()
                .zip(counts)
                .map(|(&b, &m)| u64::from(b - lo) * m)
                .sum();
            if extra <= budget {
                best = best.min(distortion(c, counts, &alloc));
            }
            let mut i = 0;
            loop {
                if i == n {
                    break;
                }
                alloc[i] += 1;
                if alloc[i] <= hi {
                    break;
                }
                alloc[i] = lo;
                i += 1;
            }
            if i == n {
                break;
            }
        }
        best
    }

    /// With equal counts every step costs the same budget, and greedy is
    /// exactly optimal — pinned against brute force on random instances.
    #[test]
    fn greedy_matches_brute_force_at_uniform_counts() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32
        };
        for trial in 0..30 {
            let n = 1 + (next() % 4) as usize;
            let c: Vec<f64> = (0..n).map(|_| 0.1 + f64::from(next() % 100)).collect();
            let counts = vec![3u64; n];
            let budget = u64::from(next() % 40);
            let (lo, hi) = (2u32, 8u32);
            let got = allocate_bits_integer(&c, &counts, budget, lo, hi);
            let got_d = distortion(&c, &counts, &got);
            let best = brute_min(&c, &counts, budget, lo, hi);
            assert!(
                got_d <= best + 1e-9,
                "trial {trial}: greedy {got_d} vs brute {best} (c {c:?}, B {budget})"
            );
        }
    }

    /// With heterogeneous counts greedy is a knapsack heuristic; its gap to
    /// optimal is bounded by one marginal step's gain (the classic greedy
    /// bound), and the budget is never exceeded.
    #[test]
    fn greedy_gap_is_bounded_at_heterogeneous_counts() {
        let mut state = 0x853c_49e6_748f_ea9bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32
        };
        for trial in 0..30 {
            let n = 1 + (next() % 4) as usize;
            let c: Vec<f64> = (0..n).map(|_| 0.1 + f64::from(next() % 100)).collect();
            let counts: Vec<u64> = (0..n).map(|_| 1 + u64::from(next() % 5)).collect();
            let budget = u64::from(next() % 30);
            let (lo, hi) = (2u32, 8u32);
            let got = allocate_bits_integer(&c, &counts, budget, lo, hi);
            let spent: u64 = got
                .iter()
                .zip(&counts)
                .map(|(&b, &m)| u64::from(b - lo) * m)
                .sum();
            assert!(spent <= budget, "trial {trial}: budget exceeded");
            let got_d = distortion(&c, &counts, &got);
            let best = brute_min(&c, &counts, budget, lo, hi);
            // One-step bound: the largest single marginal gain at the floor.
            let max_step: f64 = c
                .iter()
                .zip(&counts)
                .map(|(&ci, &mi)| ci * mi as f64 * 0.75 * (-2.0 * f64::from(lo)).exp2())
                .fold(0.0, f64::max);
            assert!(
                got_d <= best + max_step + 1e-9,
                "trial {trial}: greedy {got_d} vs brute {best} + step {max_step}"
            );
        }
    }

    /// AM/GM: 1 exactly for uniform sensitivities, > 1.2 for a real spread —
    /// the skip-mixed-precision checklist number.
    #[test]
    fn amgm_flags_spread() {
        let n = [5u64, 5, 5];
        assert!((amgm_ratio(&[3.0, 3.0, 3.0], &n) - 1.0).abs() < 1e-12);
        let spread = amgm_ratio(&[10.0, 1.0, 0.1], &n);
        assert!(
            spread > 1.2,
            "an order-of-magnitude spread predicts a win, got {spread}"
        );
        // Weighting matters: the same spread carried by tiny tensors
        // shrinks the predicted win.
        let lopsided = amgm_ratio(&[10.0, 1.0, 0.1], &[1, 1000, 1]);
        assert!(lopsided < spread);
    }
}
