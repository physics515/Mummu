//! **The DRAM bandwidth governor: who gets the memory bus, decided by
//! measurement and closed-loop control (SPEC P1).**
//!
//! The host GEMV kernel reaches the DRAM roofline in isolation (46-48 GB/s
//! measured against a 42.2 GB/s threaded-read reference) and ~10 GB/s inside
//! the serving process. The difference is not the kernel — it is *sharing*:
//! the weight stream, wgpu's staging/readback traffic and the desktop all
//! queue on one memory controller, and the queueing inflates every
//! consumer's effective latency. This module treats the controller as a
//! processor-sharing server and closes the loop three ways, cheapest first:
//!
//! 1. [`waterfill`] — the proportionally-fair open-loop split: consumer `j`
//!    with `W_j` pending bytes gets `x_j = B * sqrt(W_j) / sum_k sqrt(W_k)`,
//!    the exact maximizer of `sum_j -W_j / x_j` subject to `sum x_j <= B`
//!    (KKT: `W_j / x_j^2 = mu` for every active consumer).
//! 2. [`ShadowPrice`] — the online dual update `mu <- [mu + gamma *
//!    (measured_total - target)]+` that lets each consumer solve its own
//!    `max U_j(x) - mu x` without a central re-solve per token.
//! 3. [`fit_dynamics`] + [`lqr_gain`] — a discrete LQR on the measured
//!    state (achieved bandwidth error, contention signal): fit `s' = A s +
//!    B u` by least squares over the probe grid an in-situ sweep produces,
//!    solve the DARE by fixed-point iteration, and apply `u = -K s`
//!    **between tokens** — the controller never touches a GEMV in flight.
//!
//! Plus the admission side: [`Ledger`], a per-step byte budget that turns
//! "the staging path may not stampede the weight stream" from a priority
//! fight (which Windows arbitrates at millisecond quanta) into arithmetic
//! (which nobody arbitrates). And [`ThreadTuner`], the measurement-driven
//! replacement for every derived thread-count rule this repo has had to
//! retract: golden-section-style search over a discrete unimodal `T(n)`,
//! fed medians, immune to the "8 threads was right once" trap.
//!
//! Everything here is pure arithmetic on caller-supplied measurements —
//! dependency-free like the rest of this crate, wired (or not) by serve.

/// Proportionally-fair bandwidth split: `x_j = B * sqrt(W_j) / sum sqrt(W_k)`.
///
/// `pending` is each consumer's bytes outstanding this step (zero pending
/// gets zero allocation — a consumer with nothing to move must not reserve
/// bus); `bandwidth` is the budget to split, in the same unit the caller
/// wants back. The split maximizes `sum_j -W_j/x_j` (total completion-time
/// utility) subject to `sum x_j <= B`; see the module doc for the KKT
/// derivation.
///
/// # Panics
/// If `bandwidth` is not finite and positive, or any pending is negative.
#[must_use]
pub fn waterfill(pending: &[f64], bandwidth: f64) -> Vec<f64> {
    assert!(
        bandwidth.is_finite() && bandwidth > 0.0,
        "waterfill: bandwidth must be finite and positive, got {bandwidth}"
    );
    let mut roots = Vec::with_capacity(pending.len());
    let mut total = 0.0f64;
    for &w in pending {
        assert!(
            w.is_finite() && w >= 0.0,
            "waterfill: pending bytes must be finite and non-negative, got {w}"
        );
        let r = w.sqrt();
        roots.push(r);
        total += r;
    }
    if total <= 0.0 {
        return vec![0.0; pending.len()];
    }
    roots.iter().map(|r| bandwidth * r / total).collect()
}

/// The online dual variable of the waterfilling problem: a price on bus
/// bytes that rises while measured demand exceeds the target and decays
/// toward zero when the bus has slack. Consumers that solve
/// `max_x U(x) - mu x` locally then converge to the fair split without a
/// central allocator running per token.
#[derive(Debug, Clone)]
pub struct ShadowPrice {
    mu: f64,
    /// Step size of the dual ascent. Larger reacts faster and oscillates
    /// more; the classic stability bound is `gamma < 2 / L` for `L` the
    /// curvature of the demand response, which the caller knows only by
    /// measurement — start small.
    pub gamma: f64,
}

impl ShadowPrice {
    #[must_use]
    pub fn new(gamma: f64) -> Self {
        assert!(
            gamma.is_finite() && gamma > 0.0,
            "ShadowPrice: gamma must be finite and positive"
        );
        Self { mu: 0.0, gamma }
    }

    /// Current price.
    #[must_use]
    pub fn price(&self) -> f64 {
        self.mu
    }

    /// One dual step: `mu <- [mu + gamma * (measured_total - target)]+`.
    /// Returns the updated price.
    pub fn observe(&mut self, measured_total: f64, target: f64) -> f64 {
        assert!(
            measured_total.is_finite() && target.is_finite(),
            "ShadowPrice::observe: non-finite input"
        );
        self.mu = (self.mu + self.gamma * (measured_total - target)).max(0.0);
        self.mu
    }

    /// The utility-maximizing demand of a `-W/x` consumer at the current
    /// price: `x*(mu) = sqrt(W / mu)` (unbounded at price zero — the caller
    /// caps by its physical ceiling).
    #[must_use]
    pub fn demand(&self, pending: f64, ceiling: f64) -> f64 {
        assert!(pending >= 0.0 && ceiling >= 0.0);
        if self.mu <= 0.0 {
            return ceiling;
        }
        (pending / self.mu).sqrt().min(ceiling)
    }
}

// ---------------------------------------------------------------------------
// Small dense linear algebra (row-major Vec<Vec<f64>>) — enough for an LQR
// on a 2-4 dimensional state, no dependency.
// ---------------------------------------------------------------------------

/// Row-major matrix product.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, k) = (a.len(), b.len());
    let m = if k > 0 { b[0].len() } else { 0 };
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        assert_eq!(a[i].len(), k, "matmul: inner dims disagree");
        for (p, brow) in b.iter().enumerate() {
            let aip = a[i][p];
            for j in 0..m {
                out[i][j] += aip * brow[j];
            }
        }
    }
    out
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = if n > 0 { a[0].len() } else { 0 };
    let mut out = vec![vec![0.0; n]; m];
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[j][i] = v;
        }
    }
    out
}

fn add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

/// Solve `A x = rhs` for square `A` by Gaussian elimination with partial
/// pivoting; `rhs` may have many columns. Panics on a numerically singular
/// system — the callers all add ridge regularization precisely so this
/// cannot fire on real data.
fn solve(a: &[Vec<f64>], rhs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = if n > 0 { rhs[0].len() } else { 0 };
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(rhs)
        .map(|(ra, rr)| ra.iter().chain(rr.iter()).copied().collect())
        .collect();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| {
                aug[i][col]
                    .abs()
                    .partial_cmp(&aug[j][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        assert!(
            aug[pivot][col].abs() > 1e-12,
            "solve: singular system (pivot {col})"
        );
        aug.swap(col, pivot);
        let inv = 1.0 / aug[col][col];
        for j in col..n + m {
            aug[col][j] *= inv;
        }
        for row in 0..n {
            if row != col && aug[row][col] != 0.0 {
                let f = aug[row][col];
                for j in col..n + m {
                    aug[row][j] -= f * aug[col][j];
                }
            }
        }
    }
    aug.into_iter().map(|r| r[n..].to_vec()).collect()
}

/// Least-squares fit of `s_{t+1} = A s_t + B u_t` from observed transitions,
/// with a small ridge so a probe grid that under-excites a direction yields
/// a damped estimate instead of a singular solve.
///
/// `states[t]` and `controls[t]` produce `next_states[t]`; all `states` rows
/// share one dimension `ns`, all `controls` rows one dimension `nu`. Returns
/// `(A, B)` with `A: ns x ns`, `B: ns x nu`.
///
/// # Panics
/// On empty or inconsistent observation shapes.
#[must_use]
pub fn fit_dynamics(
    states: &[Vec<f64>],
    controls: &[Vec<f64>],
    next_states: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let t = states.len();
    assert!(t > 0, "fit_dynamics: no observations");
    assert!(
        controls.len() == t && next_states.len() == t,
        "fit_dynamics: observation counts disagree"
    );
    let ns = states[0].len();
    let nu = controls[0].len();
    let d = ns + nu;
    // Regressor rows z_t = [s_t; u_t]; solve (Z'Z + ridge I) Theta = Z' S'.
    let z: Vec<Vec<f64>> = states
        .iter()
        .zip(controls)
        .map(|(s, u)| {
            assert_eq!(s.len(), ns, "fit_dynamics: ragged state");
            assert_eq!(u.len(), nu, "fit_dynamics: ragged control");
            s.iter().chain(u.iter()).copied().collect()
        })
        .collect();
    let zt = transpose(&z);
    let mut ztz = matmul(&zt, &z);
    let ridge = 1e-8 * (0..d).map(|i| ztz[i][i]).fold(1.0f64, f64::max);
    for (i, row) in ztz.iter_mut().enumerate() {
        row[i] += ridge;
    }
    let zts = matmul(&zt, next_states);
    let theta = solve(&ztz, &zts); // d x ns: [A'; B']
    let theta_t = transpose(&theta); // ns x d
    let a = theta_t.iter().map(|r| r[..ns].to_vec()).collect();
    let b = theta_t.iter().map(|r| r[ns..].to_vec()).collect();
    (a, b)
}

/// Solve the discrete algebraic Riccati equation by fixed-point iteration
/// (`P <- Q + A'PA - A'PB (R + B'PB)^-1 B'PA`), then return the
/// infinite-horizon gain `K = (R + B'PB)^-1 B'PA`, so the control law is
/// `u = -K s`.
///
/// Converges for any stabilizable `(A, B)` with `Q >= 0`, `R > 0`; iteration
/// stops when successive `P`s agree to 1e-10 relative or after `max_iter`.
///
/// # Panics
/// On dimension mismatches or a singular `R + B'PB`.
#[must_use]
pub fn lqr_gain(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    max_iter: usize,
) -> Vec<Vec<f64>> {
    let ns = a.len();
    assert!(ns > 0 && a[0].len() == ns, "lqr_gain: A must be square");
    assert!(b.len() == ns, "lqr_gain: B row count must match A");
    let nu = b[0].len();
    assert!(q.len() == ns && r.len() == nu, "lqr_gain: Q/R dims");

    let at = transpose(a);
    let bt = transpose(b);
    let mut p = q.to_vec();
    for _ in 0..max_iter {
        let pa = matmul(&p, a); // P A
        let pb = matmul(&p, b); // P B
        let btpb = add(r, &matmul(&bt, &pb)); // R + B'PB
        let btpa = matmul(&bt, &pa); // B'PA
        let k = solve(&btpb, &btpa); // (R+B'PB)^-1 B'PA
        let atpa = matmul(&at, &pa); // A'PA
        let atpb = matmul(&at, &pb); // A'PB
        let next = add(q, &sub(&atpa, &matmul(&atpb, &k)));
        let delta = diff_norm(&p, &next);
        let scale = norm(&next).max(1.0);
        p = next;
        if delta / scale < 1e-10 {
            break;
        }
    }
    let pb = matmul(&p, b);
    let btpb = add(r, &matmul(&bt, &pb));
    let btpa = matmul(&bt, &matmul(&p, a));
    solve(&btpb, &btpa)
}

fn sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x - y).collect())
        .collect()
}

fn norm(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|r| r.iter())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

fn diff_norm(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb))
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Apply the LQR law `u = -K s`.
#[must_use]
pub fn lqr_control(k: &[Vec<f64>], s: &[f64]) -> Vec<f64> {
    k.iter()
        .map(|row| -row.iter().zip(s).map(|(kij, sj)| kij * sj).sum::<f64>())
        .collect()
}

// ---------------------------------------------------------------------------
// Byte-ledger admission (the TDMA-lite slice of SPEC P1.2)
// ---------------------------------------------------------------------------

/// A per-step DRAM byte budget for one consumer class. The point is
/// *admission over priority*: instead of asking the OS scheduler to keep
/// staging copies from stampeding the weight stream (it arbitrates at
/// millisecond quanta and loses), the step hands each class a byte budget
/// and the class defers work that does not fit to the next step.
///
/// The ledger never blocks — `admit` answers, the caller chooses. Work that
/// MUST happen this step (correctness) is charged with [`Ledger::charge`]
/// unconditionally and simply overdraws; the overdraft shows in
/// [`Ledger::balance`] and the next step's budget can subtract it.
#[derive(Debug, Clone)]
pub struct Ledger {
    budget: u64,
    spent: u64,
}

impl Ledger {
    #[must_use]
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget: budget_bytes,
            spent: 0,
        }
    }

    /// Would `bytes` more fit this step's budget?
    #[must_use]
    pub fn admit(&self, bytes: u64) -> bool {
        self.spent.saturating_add(bytes) <= self.budget
    }

    /// Record `bytes` as moved (admitted or not — mandatory traffic charges
    /// unconditionally and overdraws).
    pub fn charge(&mut self, bytes: u64) {
        self.spent = self.spent.saturating_add(bytes);
    }

    /// Remaining budget; negative (as overdraft) reports how far mandatory
    /// traffic exceeded the plan.
    #[must_use]
    pub fn balance(&self) -> i64 {
        i64::try_from(self.budget).unwrap_or(i64::MAX)
            - i64::try_from(self.spent).unwrap_or(i64::MAX)
    }

    /// Start the next step: budget may carry a correction (e.g. last step's
    /// overdraft subtracted by the caller).
    pub fn reset(&mut self, budget_bytes: u64) {
        self.budget = budget_bytes;
        self.spent = 0;
    }
}

// ---------------------------------------------------------------------------
// Measurement-driven thread-count search
// ---------------------------------------------------------------------------

/// Discrete search for the thread count minimizing a measured, unimodal-ish
/// per-call time, with hysteresis so noise cannot thrash the pool.
///
/// The repo's history is the argument for this type: an 8-thread rule was
/// derived, shipped, later measured as a **null** (60.75 vs 60.98 s at
/// 8-vs-16 after the kernel changed underneath it) — a constant tuned once
/// is a constant wrong later. `ThreadTuner` keeps the decision attached to
/// the measurement: feed it the median per-call time observed at the
/// current width; it walks the candidate list toward the faster neighbor
/// only when the improvement clears `min_gain` (default 5%), and needs
/// `settle` consecutive confirmations before moving again.
#[derive(Debug, Clone)]
pub struct ThreadTuner {
    /// Candidate widths, ascending (e.g. [4, 8, 12, 16, 24, 32]).
    candidates: Vec<usize>,
    /// Index of the width currently in force.
    cur: usize,
    /// Best observed median per width (NaN = not yet measured).
    observed: Vec<f64>,
    /// Fractional improvement a move must clear.
    pub min_gain: f64,
    /// Consecutive confirmations required before a move.
    pub settle: usize,
    streak: usize,
}

impl ThreadTuner {
    /// # Panics
    /// If `candidates` is empty or not strictly ascending, or `start` is not
    /// among them.
    #[must_use]
    pub fn new(candidates: Vec<usize>, start: usize) -> Self {
        assert!(!candidates.is_empty(), "ThreadTuner: no candidates");
        assert!(
            candidates.windows(2).all(|w| w[0] < w[1]),
            "ThreadTuner: candidates must be strictly ascending"
        );
        let cur = candidates
            .iter()
            .position(|&c| c == start)
            .expect("ThreadTuner: start width must be a candidate");
        let n = candidates.len();
        Self {
            candidates,
            cur,
            observed: vec![f64::NAN; n],
            min_gain: 0.05,
            settle: 2,
            streak: 0,
        }
    }

    /// The width the caller should run at now.
    #[must_use]
    pub fn width(&self) -> usize {
        self.candidates[self.cur]
    }

    /// Feed the median per-call milliseconds measured at the CURRENT width;
    /// returns the width to use next (possibly unchanged). The tuner probes
    /// unmeasured neighbors first (each gets its turn as the current width),
    /// then settles on the local minimum; a later regression at the settled
    /// width (>2x its recorded best) clears the neighbors' records so the
    /// search re-opens — machine state changed, and the old picks are stale
    /// (the autotune-cache lesson).
    pub fn observe(&mut self, median_ms: f64) -> usize {
        assert!(
            median_ms.is_finite() && median_ms > 0.0,
            "ThreadTuner::observe: bad measurement {median_ms}"
        );
        let prev = self.observed[self.cur];
        self.observed[self.cur] = if prev.is_nan() {
            median_ms
        } else {
            prev.min(median_ms)
        };
        if !prev.is_nan() && median_ms > prev * 2.0 {
            // The world changed under the tuned pick; forget the neighbors
            // so the walk re-verifies them.
            for (i, o) in self.observed.iter_mut().enumerate() {
                if i != self.cur {
                    *o = f64::NAN;
                }
            }
            self.observed[self.cur] = median_ms;
        }

        // Probe an unmeasured neighbor before judging direction.
        for delta in [1i64, -1] {
            let n = self.cur as i64 + delta;
            if n >= 0 && (n as usize) < self.candidates.len() && self.observed[n as usize].is_nan()
            {
                self.cur = n as usize;
                self.streak = 0;
                return self.width();
            }
        }

        // Both neighbors measured (or absent): move to the best of the
        // neighborhood if it clears the gain bar for `settle` observations.
        let lo = self.cur.saturating_sub(1);
        let hi = (self.cur + 1).min(self.candidates.len() - 1);
        let best = (lo..=hi)
            .filter(|&i| !self.observed[i].is_nan())
            .min_by(|&i, &j| {
                self.observed[i]
                    .partial_cmp(&self.observed[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(self.cur);
        let cur_t = self.observed[self.cur];
        if best != self.cur && self.observed[best] < cur_t * (1.0 - self.min_gain) {
            self.streak += 1;
            if self.streak >= self.settle {
                self.cur = best;
                self.streak = 0;
            }
        } else {
            self.streak = 0;
        }
        self.width()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- waterfill ---------------------------------------------------------

    #[test]
    fn waterfill_is_the_kkt_optimum() {
        let w = [4.0e9, 1.0e9, 0.25e9];
        let b = 42.0;
        let x = waterfill(&w, b);
        // Allocations exhaust the budget…
        assert!((x.iter().sum::<f64>() - b).abs() < 1e-9);
        // …split by sqrt of pending: 2:1:0.5.
        assert!((x[0] / x[1] - 2.0).abs() < 1e-9);
        assert!((x[1] / x[2] - 2.0).abs() < 1e-9);
        // KKT: W_j / x_j^2 equal across active consumers.
        let mu0 = w[0] / (x[0] * x[0]);
        for j in 1..3 {
            assert!((w[j] / (x[j] * x[j]) - mu0).abs() / mu0 < 1e-9);
        }
        // And no feasible perturbation improves total utility sum(-W/x).
        let util = |x: &[f64]| -> f64 { x.iter().zip(&w).map(|(xi, wi)| -wi / xi).sum() };
        let base = util(&x);
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            let mut p = x.clone();
            let eps = 1e-3;
            p[i] += eps;
            p[j] -= eps;
            assert!(util(&p) <= base + 1e-12, "perturbation {i}->{j} improved");
        }
    }

    #[test]
    fn waterfill_zero_pending_gets_zero() {
        let x = waterfill(&[9.0, 0.0, 1.0], 10.0);
        assert_eq!(x[1], 0.0);
        assert!((x.iter().sum::<f64>() - 10.0).abs() < 1e-9);
        assert_eq!(waterfill(&[0.0, 0.0], 10.0), vec![0.0, 0.0]);
    }

    // -- shadow price ------------------------------------------------------

    #[test]
    fn shadow_price_converges_demand_to_target() {
        // Two -W/x consumers with physical ceilings; iterate price + demand.
        let (w1, w2) = (16.0, 4.0);
        let target = 6.0;
        let mut price = ShadowPrice::new(0.05);
        let mut total = 0.0;
        for _ in 0..8000 {
            let x1 = price.demand(w1, 100.0);
            let x2 = price.demand(w2, 100.0);
            total = x1 + x2;
            price.observe(total, target);
        }
        assert!(
            (total - target).abs() < 0.05,
            "demand {total} did not converge to target {target}"
        );
        // At the fixed point the split is the waterfill split.
        let x = waterfill(&[w1, w2], target);
        let x1 = price.demand(w1, 100.0);
        assert!(
            (x1 - x[0]).abs() < 0.05,
            "price split {x1} vs waterfill {}",
            x[0]
        );
    }

    // -- LQR ---------------------------------------------------------------

    #[test]
    fn scalar_dare_matches_closed_form() {
        // a=0.9, b=1, q=1, r=1: P solves P = q + a^2 P - a^2 P^2/(r+P).
        let a = vec![vec![0.9]];
        let b = vec![vec![1.0]];
        let q = vec![vec![1.0]];
        let r = vec![vec![1.0]];
        let k = lqr_gain(&a, &b, &q, &r, 10_000)[0][0];
        // Solve the scalar DARE independently by bisection on P.
        let f = |p: f64| 1.0 + 0.81 * p - 0.81 * p * p / (1.0 + p) - p;
        let (mut lo, mut hi) = (0.0, 100.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let p = 0.5 * (lo + hi);
        let k_ref = 0.9 * p / (1.0 + p);
        assert!((k - k_ref).abs() < 1e-6, "K {k} vs closed form {k_ref}");
    }

    #[test]
    fn fitted_lqr_stabilizes_the_true_system() {
        // Ground-truth 2-state system, mildly unstable open-loop.
        let a_true = [[1.02, 0.10], [0.00, 0.95]];
        let b_true = [[0.0], [1.0]];
        // Generate a probe grid: random-ish states and controls (fixed PCG).
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut states = Vec::new();
        let mut controls = Vec::new();
        let mut nexts = Vec::new();
        for _ in 0..64 {
            let s = [rand() * 4.0, rand() * 4.0];
            let u = rand() * 2.0;
            states.push(s.to_vec());
            controls.push(vec![u]);
            nexts.push(vec![
                a_true[0][0] * s[0] + a_true[0][1] * s[1] + b_true[0][0] * u,
                a_true[1][0] * s[0] + a_true[1][1] * s[1] + b_true[1][0] * u,
            ]);
        }
        let (a, b) = fit_dynamics(&states, &controls, &nexts);
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[i][j] - a_true[i][j]).abs() < 1e-6, "A[{i}][{j}]");
            }
            assert!((b[i][0] - b_true[i][0]).abs() < 1e-6, "B[{i}]");
        }
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![0.1]];
        let k = lqr_gain(&a, &b, &q, &r, 10_000);
        // Closed loop from a large start must contract to ~zero.
        let mut s = vec![10.0, -8.0];
        for _ in 0..200 {
            let u = lqr_control(&k, &s);
            let ns = vec![
                a_true[0][0] * s[0] + a_true[0][1] * s[1] + b_true[0][0] * u[0],
                a_true[1][0] * s[0] + a_true[1][1] * s[1] + b_true[1][0] * u[0],
            ];
            s = ns;
        }
        let norm = (s[0] * s[0] + s[1] * s[1]).sqrt();
        assert!(norm < 1e-3, "closed loop did not contract: |s| = {norm}");
    }

    // -- ledger ------------------------------------------------------------

    #[test]
    fn ledger_admits_within_budget_and_reports_overdraft() {
        let mut l = Ledger::new(100);
        assert!(l.admit(60));
        l.charge(60);
        assert!(l.admit(40));
        assert!(!l.admit(41));
        l.charge(50); // mandatory traffic overdraws
        assert_eq!(l.balance(), -10);
        l.reset(90);
        assert_eq!(l.balance(), 90);
        assert!(l.admit(90));
    }

    // -- thread tuner ------------------------------------------------------

    /// A synthetic unimodal cost curve with the minimum at 12 threads.
    fn cost(width: usize) -> f64 {
        let w = width as f64;
        1.0 + (w - 12.0) * (w - 12.0) * 0.02
    }

    #[test]
    fn tuner_walks_to_the_unimodal_minimum() {
        let mut t = ThreadTuner::new(vec![4, 8, 12, 16, 24], 4);
        let mut width = t.width();
        for _ in 0..64 {
            width = t.observe(cost(width));
        }
        assert_eq!(width, 12, "tuner settled at {width}, not the minimum");
    }

    #[test]
    fn tuner_reopens_after_a_regime_change() {
        let mut t = ThreadTuner::new(vec![4, 8, 12, 16, 24], 4);
        let mut width = t.width();
        for _ in 0..64 {
            width = t.observe(cost(width));
        }
        assert_eq!(width, 12);
        // Regime change: minimum moves to 4 (heavy contention penalizes
        // width), and the settled width's cost triples — the tuner must
        // notice and re-search rather than trusting stale records.
        let cost2 = |w: usize| 3.0 + (w as f64 - 4.0) * (w as f64 - 4.0) * 0.05;
        for _ in 0..64 {
            width = t.observe(cost2(width));
        }
        assert_eq!(width, 4, "tuner stuck at {width} after the regime change");
    }

    #[test]
    fn tuner_hysteresis_ignores_single_noise_spikes() {
        let mut t = ThreadTuner::new(vec![8, 16], 16);
        // Establish both widths: 16 is better.
        let w = t.observe(1.0); // at 16 -> probes 8
        assert_eq!(w, 8);
        let w = t.observe(1.5); // 8 is worse -> back toward 16
        // One noisy observation at the current width must not move it
        // permanently; after clean observations it sits at 16.
        let mut width = w;
        for _ in 0..8 {
            width = t.observe(if width == 16 { 1.0 } else { 1.5 });
        }
        assert_eq!(width, 16);
    }
}
