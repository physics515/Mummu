//! Attribution of the orchestration term — exact Shapley values over
//! togglable components (SPEC 2, algorithm A).
//!
//! The 27B decodes at 2.25–2.40 s/token while the sum of its measured pieces
//! explains substantially less; the remainder is *orchestration* — fences,
//! crossings, launches, dispatch — and the obvious way to attribute it
//! ("turn one thing off, measure the delta") is wrong whenever components
//! interact: the delta you measure for component A depends on whether B was
//! on when you toggled A. Toggle order becomes an undocumented parameter of
//! the answer, and this codebase has already burned a day on exactly that
//! kind of mis-attribution (see `prof.rs`'s module header).
//!
//! The Shapley value is the unique attribution that removes the order: each
//! component is credited its marginal cost averaged over **every** possible
//! toggle order, and the axioms that pin it down are the ones a cost
//! breakdown actually needs —
//!
//! * **efficiency**: the shares sum exactly to `v(full) − v(empty)`, the
//!   total cost being explained; nothing is lost or invented;
//! * **dummy**: a component that never changes a measurement gets 0;
//! * **symmetry**: interchangeable components get equal shares;
//! * **linearity**: the Shapley value of an averaged game is the average of
//!   the Shapley values — which is what makes the repeated-measurement
//!   harness below statistically clean.
//!
//! # Mask convention
//!
//! Everything here indexes subsets as bitmasks: bit `j` **set** in `mask`
//! means component `j` is **enabled** — its overhead is *present* in the
//! measurement. `v(mask)` is the measured cost with exactly the components
//! in `mask` active and every other component disabled. `v(0)` is the base
//! cost with all toggles off; `v((1 << n) - 1)` is the full configuration.
//! Under this convention `phi[j] >= 0` for a component that only ever adds
//! cost, and the efficiency axiom reads: `sum(phi) == v(full) - v(empty)`.
//!
//! # Cost
//!
//! Exact Shapley enumerates all `2^n` subsets. For the decode-step
//! decompositions this serves (4–8 components, each subset measured in
//! milliseconds) that is 16–256 measurements per replicate — cheap. The cap
//! is `n <= 16` (65 536 subsets), asserted, because past that the exact
//! method is the wrong tool.

/// A togglable component of the decode step — a lightweight name for one bit
/// of the subset mask. Exists so attribution tables carry human-readable
/// labels next to `phi` without this module dictating what a "component" is
/// (a feature flag, an injected sleep, a skipped fence…).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub name: String,
}

impl Component {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Exact Shapley values for an `n`-component game.
///
/// `v(mask)` is the characteristic function under the module's convention:
/// the measured cost with the components whose bits are set in `mask`
/// **enabled**. `v` is called exactly once per subset (`2^n` calls, masks
/// `0..2^n` in ascending order) and memoized; if a call is a real
/// measurement, its cost dominates and the enumeration order is the
/// measurement order.
///
/// Returns `phi` with `phi[j]` = component `j`'s share of
/// `v(full) − v(empty)`. The efficiency axiom holds by construction up to
/// f64 rounding (the subset weights `|S|!·(n−1−|S|)!/n!` telescope exactly;
/// factorials to 16! are exactly representable in f64).
///
/// # Panics
///
/// If `n > 16`. That is a static contract of the caller's decomposition,
/// not user input: `2^n` subset measurements past 16 components means the
/// exact method is the wrong tool (use sampling), and silently truncating
/// would mis-attribute.
#[must_use]
pub fn shapley(n: usize, v: &mut dyn FnMut(u32) -> f64) -> Vec<f64> {
    assert!(
        n <= 16,
        "shapley: n = {n} components exceeds the exact-enumeration cap of 16 \
         (2^n subsets; use a sampling estimator instead)"
    );
    let size = 1usize << n;
    // Memoize v: each subset is measured exactly once, then reused by every
    // component's marginal sum.
    let mut table = vec![0.0f64; size];
    for (mask, slot) in table.iter_mut().enumerate() {
        *slot = v(mask as u32);
    }
    shapley_from_table(n, &table)
}

/// Shapley from an already-measured subset table (`table[mask] = v(mask)`).
/// Split out so the replicate harness can attribute each replicate's table
/// without re-measuring.
fn shapley_from_table(n: usize, table: &[f64]) -> Vec<f64> {
    // Factorials to 16! are < 2^53, so every weight below is a quotient of
    // exactly-represented integers.
    let mut fact = [1.0f64; 17];
    for i in 1..=16 {
        fact[i] = fact[i - 1] * i as f64;
    }
    let n_fact = fact[n];
    let mut phi = vec![0.0f64; n];
    for (j, phi_j) in phi.iter_mut().enumerate() {
        let bit = 1u32 << j;
        let mut acc = 0.0f64;
        for mask in 0..table.len() as u32 {
            if mask & bit != 0 {
                continue; // enumerate S ⊆ N \ {j}
            }
            let s = mask.count_ones() as usize;
            // Probability that, in a uniformly random toggle order, exactly
            // the components of S precede j.
            let weight = fact[s] * fact[n - 1 - s] / n_fact;
            acc += weight * (table[(mask | bit) as usize] - table[mask as usize]);
        }
        *phi_j = acc;
    }
    phi
}

/// Shapley attribution with uncertainty from repeated measurement.
///
/// `phi[j]` is the mean Shapley value of component `j` across replicates;
/// `ci95[j]` is the half-width of its 95% confidence interval (Student-t on
/// the per-replicate Shapley values; `f64::INFINITY` when `replicates == 1`,
/// because one sample has no variance estimate).
#[derive(Debug, Clone)]
pub struct Attribution {
    /// Mean Shapley share per component, in the measurement's unit (ms).
    pub phi: Vec<f64>,
    /// 95% CI half-width per component, same unit. `phi[j] ± ci95[j]`.
    pub ci95: Vec<f64>,
    /// How many independent replicates produced the estimate.
    pub replicates: usize,
}

/// Run the full subset sweep `replicates` times and attribute each replicate
/// independently.
///
/// `measure(mask)` performs **one** measurement of the configuration `mask`
/// (module convention: set bits = enabled components). It is called
/// `replicates × 2^n` times, replicate-major — every replicate sweeps all
/// subsets before the next begins — so slow environmental drift (thermal,
/// clocks; this box drifts ~10% between sessions) lands *across* replicates,
/// where the CI can see it, instead of confounding one subset against
/// another.
///
/// Why per-replicate attribution is the statistically clean estimator: the
/// Shapley value is **linear in v**, so the mean of the per-replicate
/// `phi_r` equals the Shapley value of the mean game — no bias — and the
/// per-replicate `phi_r` are i.i.d. samples of a scalar per component, so
/// the ordinary t-interval applies directly. (Propagating per-subset
/// variances through the weighted sum instead would need the full
/// covariance across subsets; this route sidesteps it.)
///
/// # Panics
///
/// If `n > 16` (see [`shapley`]) or `replicates == 0` (an attribution from
/// zero measurements is meaningless, and returning NaNs would poison
/// downstream tables silently).
#[must_use]
pub fn attribute(n: usize, replicates: usize, measure: &mut dyn FnMut(u32) -> f64) -> Attribution {
    assert!(
        replicates > 0,
        "attribute: replicates must be >= 1 (got 0) — no measurements, no attribution"
    );
    assert!(
        n <= 16,
        "attribute: n = {n} components exceeds the exact-enumeration cap of 16"
    );
    let size = 1usize << n;
    let mut per_replicate: Vec<Vec<f64>> = Vec::with_capacity(replicates);
    let mut table = vec![0.0f64; size];
    for _ in 0..replicates {
        for (mask, slot) in table.iter_mut().enumerate() {
            *slot = measure(mask as u32);
        }
        per_replicate.push(shapley_from_table(n, &table));
    }

    let r = replicates as f64;
    let mut phi = vec![0.0f64; n];
    let mut ci95 = vec![0.0f64; n];
    for j in 0..n {
        let mean = per_replicate.iter().map(|p| p[j]).sum::<f64>() / r;
        phi[j] = mean;
        ci95[j] = if replicates < 2 {
            // One sample: the variance is unidentifiable. Infinity is the
            // honest answer; 0 would claim false certainty.
            f64::INFINITY
        } else {
            let var = per_replicate
                .iter()
                .map(|p| (p[j] - mean).powi(2))
                .sum::<f64>()
                / (r - 1.0);
            t_975(replicates - 1) * (var / r).sqrt()
        };
    }
    Attribution {
        phi,
        ci95,
        replicates,
    }
}

/// Two-sided 95% Student-t critical value for `df` degrees of freedom.
/// Table for small df where the normal 1.96 badly undercovers (df=1 needs
/// 12.7, not 1.96); beyond 30 the normal approximation is within 2%.
fn t_975(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if df == 0 {
        f64::INFINITY
    } else if df <= 30 {
        TABLE[df - 1]
    } else {
        1.96
    }
}

/// Parse [`crate::prof::folded`] output into `(path, self_time_ms)` pairs,
/// making the profiler's report machine-consumable for attribution tables
/// and phase bisection.
///
/// The format, per `prof.rs`: one line per path, `<path> (<count>x)
/// <self_us>`, where `<path>` is the semicolon-joined scope chain (which may
/// itself contain spaces — scope names are free-form), `<count>` is the
/// entry count and `<self_us>` is **self** time in integer microseconds.
/// Parsing is therefore anchored from the *right*: the last
/// space-separated token is the microsecond count, the one before it the
/// `(Nx)` suffix, and everything left of that is the path verbatim.
///
/// Returns paths in the order they appear (prof emits them sorted, its
/// totals map is a `BTreeMap`) with times converted to milliseconds.
///
/// # Errors
///
/// A line that does not match the format fails the whole parse with a
/// message naming the line — silently skipping would drop time from the
/// attribution, which is exactly the failure mode this module exists to
/// prevent. Empty lines are tolerated (a trailing newline is normal).
pub fn parse_folded(folded: &str) -> Result<Vec<(String, f64)>, String> {
    let mut out = Vec::new();
    for (idx, raw) in folded.lines().enumerate() {
        let line = raw.trim_end_matches('\r'); // tolerate CRLF transcripts
        if line.is_empty() {
            continue;
        }
        let lineno = idx + 1;
        let (rest, us_tok) = line.rsplit_once(' ').ok_or_else(|| {
            format!("folded line {lineno}: expected `<path> (Nx) <self_us>`, got {line:?}")
        })?;
        let self_us: u64 = us_tok.parse().map_err(|_| {
            format!(
                "folded line {lineno}: self-time field {us_tok:?} is not an integer \
                 microsecond count in {line:?}"
            )
        })?;
        let (path, count_tok) = rest.rsplit_once(' ').ok_or_else(|| {
            format!("folded line {lineno}: missing `(Nx)` count before the self-time in {line:?}")
        })?;
        let count_ok = count_tok
            .strip_prefix('(')
            .and_then(|t| t.strip_suffix("x)"))
            .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()));
        if !count_ok {
            return Err(format!(
                "folded line {lineno}: count field {count_tok:?} is not of the form `(Nx)` \
                 in {line:?}"
            ));
        }
        if path.is_empty() {
            return Err(format!("folded line {lineno}: empty path in {line:?}"));
        }
        out.push((path.to_string(), self_us as f64 / 1000.0));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: f64, want: f64, what: &str) {
        let tol = 1e-9 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{what}: got {got}, want {want} (tol {tol})"
        );
    }

    /// Deterministic pseudo-noise so the CI tests can never flake.
    struct Lcg(u64);
    impl Lcg {
        /// Uniform in [-0.5, 0.5).
        fn centered(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        }
    }

    /// On an additive game (no interactions) the fair split IS the addends:
    /// every toggle order sees the same marginals, so Shapley must return
    /// them — with a constant base cost cancelling out entirely.
    #[test]
    fn additive_game_returns_the_addends() {
        let w = [2.5f64, 0.25, 7.0, 1.0];
        let mut v = |mask: u32| -> f64 {
            let mut c = 10.0; // constant base: must not leak into any share
            for (j, wj) in w.iter().enumerate() {
                if mask & (1 << j) != 0 {
                    c += wj;
                }
            }
            c
        };
        let phi = shapley(w.len(), &mut v);
        for (j, (&got, &want)) in phi.iter().zip(w.iter()).enumerate() {
            assert_close(got, want, &format!("phi[{j}]"));
        }
    }

    /// A superadditive 3-player game, checked against the hand enumeration
    /// of all 6 toggle orders: v = {∅:0, 1:1, 2:2, 3:3, 12:4, 13:5, 23:6,
    /// 123:8} gives phi = (5/3, 8/3, 11/3).
    #[test]
    fn superadditive_three_player_game_matches_hand_computation() {
        // Index = mask, bit0 = player 1, bit1 = player 2, bit2 = player 3.
        let table = [0.0, 1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 8.0];
        let mut v = |mask: u32| table[mask as usize];
        let phi = shapley(3, &mut v);
        assert_close(phi[0], 5.0 / 3.0, "phi[0]");
        assert_close(phi[1], 8.0 / 3.0, "phi[1]");
        assert_close(phi[2], 11.0 / 3.0, "phi[2]");
    }

    /// Efficiency: the shares must sum to exactly the cost being explained,
    /// v(full) − v(empty) — on an arbitrary (deterministic, interacting)
    /// game, not just a friendly one.
    #[test]
    fn efficiency_axiom_holds_on_an_arbitrary_game() {
        let n = 5;
        let mut v = |mask: u32| -> f64 {
            // Arbitrary but deterministic: nonlinear in the popcount and
            // mask-dependent, so components interact.
            let s = mask.count_ones() as f64;
            3.0 + 1.7 * s * s + 0.31 * f64::from(mask ^ (mask >> 1))
        };
        let full = (1u32 << n) - 1;
        let total = v(full) - v(0);
        let phi = shapley(n, &mut v);
        assert_close(
            phi.iter().sum::<f64>(),
            total,
            "sum(phi) vs v(full)-v(empty)",
        );
    }

    /// A dummy component (never changes the measurement) must get 0 — the
    /// axiom that keeps a disconnected toggle from stealing attribution.
    #[test]
    fn dummy_component_gets_zero() {
        let mut v = |mask: u32| -> f64 {
            // Bit 1 is wired to nothing.
            if mask & 0b001 != 0 { 4.0 } else { 0.0 }
        };
        let phi = shapley(2, &mut v);
        assert_close(phi[0], 4.0, "active component");
        assert_close(phi[1], 0.0, "dummy component");
    }

    /// More replicates must tighten the interval: same noise process, same
    /// estimator, SE ~ sigma/sqrt(R) (plus the t-factor shrinking toward
    /// 1.96). Deterministic noise, so this is a fixed arithmetic fact, not
    /// a probabilistic one.
    #[test]
    fn ci_shrinks_with_more_replicates() {
        let run = |replicates: usize, seed: u64| -> Attribution {
            let mut rng = Lcg(seed);
            let w = [5.0f64, 3.0, 2.0];
            let mut measure = |mask: u32| -> f64 {
                let mut c = 0.0;
                for (j, wj) in w.iter().enumerate() {
                    if mask & (1 << j) != 0 {
                        c += wj;
                    }
                }
                c + rng.centered() // ±0.5 of measurement noise
            };
            attribute(w.len(), replicates, &mut measure)
        };
        let few = run(8, 42);
        let many = run(128, 42);
        let few_ci: f64 = few.ci95.iter().sum();
        let many_ci: f64 = many.ci95.iter().sum();
        assert!(
            many_ci < few_ci,
            "128 replicates must beat 8: ci sum {many_ci} vs {few_ci}"
        );
        // And the means must still be near the truth: within a generous
        // multiple of the reported interval.
        for (j, want) in [5.0, 3.0, 2.0].iter().enumerate() {
            assert!(
                (many.phi[j] - want).abs() < 5.0 * many.ci95[j].max(0.05),
                "phi[{j}] = {} strayed from {want} (ci {})",
                many.phi[j],
                many.ci95[j]
            );
        }
    }

    /// One replicate has no variance estimate; the CI must say so loudly
    /// (infinite), not claim certainty (zero).
    #[test]
    fn single_replicate_reports_infinite_ci() {
        let a = attribute(2, 1, &mut |mask| f64::from(mask.count_ones()));
        assert!(a.ci95.iter().all(|c| c.is_infinite()), "{:?}", a.ci95);
        assert_close(a.phi[0], 1.0, "phi[0] still exact");
    }

    /// The parser against a hand-written folded string covering the format's
    /// corners: nested paths, a scope name containing a space, multi-digit
    /// counts, and a trailing newline.
    #[test]
    fn folded_parser_reads_the_prof_format() {
        let folded = "forward (1x) 42\n\
                      forward;layer;mlp.down (64x) 12345\n\
                      forward;odd name;sub (2x) 1000\n\
                      token;readback (2x) 1000\n";
        let parsed = parse_folded(folded).expect("well-formed folded input");
        assert_eq!(
            parsed,
            vec![
                ("forward".to_string(), 0.042),
                ("forward;layer;mlp.down".to_string(), 12.345),
                ("forward;odd name;sub".to_string(), 1.0),
                ("token;readback".to_string(), 1.0),
            ]
        );
    }

    /// Malformed lines must fail the parse with the line named — dropped
    /// lines would silently drop time from an attribution.
    #[test]
    fn folded_parser_rejects_malformed_lines() {
        for bad in [
            "no-count-or-time",
            "path (3x) notanumber",
            "path [3x] 100",
            "path (x) 100",
            " (3x) 100",
        ] {
            let err = parse_folded(bad).expect_err(bad);
            assert!(err.contains("line 1"), "error should name the line: {err}");
        }
        // Empty input and blank lines are fine.
        assert_eq!(parse_folded("").expect("empty ok"), vec![]);
        assert_eq!(parse_folded("\n\n").expect("blank ok"), vec![]);
    }

    /// n = 0 is a degenerate but legal game: no components, no shares.
    #[test]
    fn zero_components_is_empty() {
        assert!(shapley(0, &mut |_| 7.0).is_empty());
    }
}
