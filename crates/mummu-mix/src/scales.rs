//! **Low-rank factorization of the block-scale matrix itself (SPEC P2.3,
//! novel).**
//!
//! A block-quantized `[d_out, d_in]` tensor at group `g` carries a scale
//! matrix `S in R^{d_out x d_in/g}` — at f32 scales and g = 32 that is a
//! full 1.0 bit-per-weight of metadata (the difference between this repo's
//! 5.0 effective bits and Q4_K's 4.5). Shrinking the scale *bit-width* is
//! one axis; this module attacks the scale *rank*: block scales across a
//! real weight are highly structured (rows share dynamic range, channels
//! share outlier patterns), so `S ~= U V^T` with `U: d_out x r`,
//! `V: (d_in/g) x r` at r = 2..8 captures most of the matrix while storing
//! `(d_out + d_in/g) * r` floats instead of `d_out * d_in/g` — ~97% scale
//! metadata reduction at r = 4 on a 4096x4096 tensor, orthogonal to and
//! composable with narrower scale dtypes.
//!
//! The factorization is a truncated SVD of the *log*-scales: scales are
//! positive and multiplicative (a scale twice as large is one step of
//! dynamic range, whether at 1e-3 or 1e+1), so the right error metric is
//! relative, and factorizing `log S` makes `||log S - U V^T||_F` exactly
//! the sum of squared *relative* (log) errors. Truncated SVD is the
//! Frobenius-optimal rank-r approximation (Eckart–Young), so the error is
//! guaranteed non-increasing in r — an ALS variant was tried first and
//! rejected on exactly that property failing (a local method's rank-2 fit
//! measured *worse* than its rank-1 fit). The SVD comes from
//! deterministic subspace iteration on `(log S)^T (log S)` — the column
//! count is `d_in/g` (a few hundred), so the Gram matrix is small. And the
//! reconstruction `exp(U V^T)` is positive by construction: a factorized
//! scale can never go zero or negative under the quantizer.
//!
//! Calibration contract (per tensor, offline): sweep r upward until the
//! max multiplicative error [`ScaleFactor::max_ratio_err`] clears the
//! caller's bound (a scale off by ratio rho inflates that group's
//! quantization step by rho — set the bound from the same ladder-probe
//! arithmetic the mix planner uses). The runtime cost is one extra fused
//! multiply per group in the dequant epilogue; the win is bytes.

/// The factorization `S ~= exp(U V^T)` of one tensor's scale matrix.
#[derive(Debug, Clone)]
pub struct ScaleFactor {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    /// Row factors, `rows x rank`, row-major.
    pub u: Vec<f32>,
    /// Column factors, `cols x rank`, row-major.
    pub v: Vec<f32>,
}

impl ScaleFactor {
    /// Reconstructed scale at `(i, j)`: `exp(sum_t U[i,t] V[j,t])`.
    #[must_use]
    pub fn scale_at(&self, i: usize, j: usize) -> f32 {
        let mut dot = 0.0f32;
        for t in 0..self.rank {
            dot += self.u[i * self.rank + t] * self.v[j * self.rank + t];
        }
        dot.exp()
    }

    /// Bytes this factorization stores (f32 factors).
    #[must_use]
    pub fn stored_bytes(&self) -> usize {
        (self.rows + self.cols) * self.rank * 4
    }

    /// Bytes the dense scale matrix stores at `bytes_per_scale`.
    #[must_use]
    pub fn dense_bytes(&self, bytes_per_scale: usize) -> usize {
        self.rows * self.cols * bytes_per_scale
    }

    /// Root-mean-square log error `sqrt(mean((ln s' - ln s)^2))` — the
    /// metric the truncated SVD optimizes, guaranteed non-increasing in
    /// rank (Eckart–Young). [`Self::max_ratio_err`] is the calibration
    /// *gate* (worst case matters for a quantizer step) but is an infinity
    /// norm and can wiggle non-monotonically between ranks; sweep on it,
    /// reason on this.
    #[must_use]
    pub fn rms_log_err(&self, dense: &[f32]) -> f64 {
        assert_eq!(dense.len(), self.rows * self.cols);
        let mut acc = 0.0f64;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let s = f64::from(dense[i * self.cols + j]);
                let r = f64::from(self.scale_at(i, j));
                let d = r.ln() - s.ln();
                acc += d * d;
            }
        }
        (acc / (self.rows * self.cols) as f64).sqrt()
    }

    /// Worst multiplicative error over the matrix:
    /// `max_ij max(s'/s, s/s')` where `s'` is the reconstruction. This is
    /// the number the calibration sweep gates on — a group's quantization
    /// step inflates by exactly this ratio in the worst case.
    #[must_use]
    pub fn max_ratio_err(&self, dense: &[f32]) -> f32 {
        assert_eq!(dense.len(), self.rows * self.cols);
        let mut worst = 1.0f32;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let s = dense[i * self.cols + j];
                let r = self.scale_at(i, j);
                let ratio = if s > 0.0 && r > 0.0 {
                    (r / s).max(s / r)
                } else {
                    f32::INFINITY
                };
                worst = worst.max(ratio);
            }
        }
        worst
    }
}

/// Factorize a positive scale matrix `s` (`rows x cols`, row-major) at
/// `rank`, by truncated SVD of the log matrix computed with `iters` rounds
/// of subspace iteration on the Gram matrix `G = L^T L`.
///
/// Deterministic: the initialization is a fixed quasi-random pattern, so a
/// re-run reproduces the same factors bit-for-bit — calibration artifacts
/// must be reproducible. The result is the Eckart–Young optimum (to
/// iteration convergence), so error is non-increasing in rank — tested.
///
/// # Panics
/// If any scale is not strictly positive and finite, or `rank == 0`, or
/// `rank > min(rows, cols)`.
#[must_use]
pub fn factorize_scales(
    s: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    iters: usize,
) -> ScaleFactor {
    assert_eq!(s.len(), rows * cols, "scale matrix shape");
    assert!(
        rank > 0 && rank <= rows.min(cols),
        "rank {rank} out of range"
    );
    for &v in s {
        assert!(v > 0.0 && v.is_finite(), "scales must be positive, got {v}");
    }
    let logs: Vec<f64> = s.iter().map(|&v| f64::from(v.ln())).collect();

    // Gram matrix G = L^T L, cols x cols (cols = d_in/g is a few hundred).
    let mut g = vec![0.0f64; cols * cols];
    for row in logs.chunks_exact(cols) {
        for (a, &ra) in row.iter().enumerate() {
            for (b, &rb) in row.iter().enumerate() {
                g[a * cols + b] += ra * rb;
            }
        }
    }

    // Subspace iteration: X <- orth(G X), starting from a fixed
    // golden-ratio pattern. Converges to the span of the top-`rank`
    // eigenvectors of G = the top right-singular vectors of L.
    let mut x = vec![0.0f64; cols * rank]; // column-major by factor: x[j*rank + t]
    for j in 0..cols {
        for t in 0..rank {
            let q = (j * rank + t) as f64 * 0.618_033_988_749_895;
            x[j * rank + t] = q.fract() - 0.5;
        }
    }
    orthonormalize(&mut x, cols, rank);
    let mut gx = vec![0.0f64; cols * rank];
    for _ in 0..iters.max(1) {
        // gx = G x.
        for j in 0..cols {
            for t in 0..rank {
                let mut acc = 0.0;
                for l in 0..cols {
                    acc += g[j * cols + l] * x[l * rank + t];
                }
                gx[j * rank + t] = acc;
            }
        }
        std::mem::swap(&mut x, &mut gx);
        orthonormalize(&mut x, cols, rank);
    }

    // V = x (orthonormal columns); U = L V, so U V^T is the projection of
    // L onto span(V) — the truncated SVD once V spans the top subspace.
    let mut u = vec![0.0f64; rows * rank];
    for (i, row) in logs.chunks_exact(cols).enumerate() {
        for t in 0..rank {
            let mut acc = 0.0;
            for (j, &rj) in row.iter().enumerate() {
                acc += rj * x[j * rank + t];
            }
            u[i * rank + t] = acc;
        }
    }

    ScaleFactor {
        rows,
        cols,
        rank,
        #[allow(clippy::cast_possible_truncation)]
        u: u.into_iter().map(|v| v as f32).collect(),
        #[allow(clippy::cast_possible_truncation)]
        v: x.into_iter().map(|v| v as f32).collect(),
    }
}

/// Modified Gram–Schmidt over the `rank` columns of `x` (`n x rank`,
/// row-major as `x[j*rank + t]`). A column that collapses to numerical
/// zero (degenerate subspace) is re-seeded from a fixed pattern and
/// re-orthogonalized, so iteration never divides by zero.
fn orthonormalize(x: &mut [f64], n: usize, rank: usize) {
    for t in 0..rank {
        // Subtract projections onto the earlier columns.
        for p in 0..t {
            let mut dot = 0.0;
            for j in 0..n {
                dot += x[j * rank + t] * x[j * rank + p];
            }
            for j in 0..n {
                x[j * rank + t] -= dot * x[j * rank + p];
            }
        }
        let mut norm = 0.0;
        for j in 0..n {
            norm += x[j * rank + t] * x[j * rank + t];
        }
        let mut norm = norm.sqrt();
        if norm < 1e-12 {
            // Re-seed deterministically and re-orthogonalize this column.
            for j in 0..n {
                let q = (j * 31 + t * 17 + 7) as f64 * 0.754_877_666_246_692_9;
                x[j * rank + t] = q.fract() - 0.5;
            }
            for p in 0..t {
                let mut dot = 0.0;
                for j in 0..n {
                    dot += x[j * rank + t] * x[j * rank + p];
                }
                for j in 0..n {
                    x[j * rank + t] -= dot * x[j * rank + p];
                }
            }
            norm = (0..n)
                .map(|j| x[j * rank + t] * x[j * rank + t])
                .sum::<f64>()
                .sqrt();
            assert!(norm > 1e-12, "orthonormalize: degenerate re-seed");
        }
        for j in 0..n {
            x[j * rank + t] /= norm;
        }
    }
}

/// The calibration sweep: smallest rank whose worst multiplicative error
/// clears `max_ratio`, or `None` if even `max_rank` misses it (keep the
/// dense scales for that tensor).
#[must_use]
pub fn calibrate_rank(
    s: &[f32],
    rows: usize,
    cols: usize,
    max_rank: usize,
    iters: usize,
    max_ratio: f32,
) -> Option<(usize, ScaleFactor)> {
    for r in 1..=max_rank.min(rows.min(cols)) {
        let f = factorize_scales(s, rows, cols, r, iters);
        if f.max_ratio_err(s) <= max_ratio {
            return Some((r, f));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic scale matrix of exact multiplicative rank r: outer
    /// products in log space.
    fn synthetic(rows: usize, cols: usize, rank: usize) -> Vec<f32> {
        let mut s = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut log = 0.0f64;
                for t in 0..rank {
                    let ui = (((i * 31 + t * 7) % 17) as f64 / 17.0 - 0.5) * 2.0;
                    let vj = (((j * 13 + t * 5) % 19) as f64 / 19.0 - 0.5) * 2.0;
                    log += ui * vj;
                }
                #[allow(clippy::cast_possible_truncation)]
                {
                    s[i * cols + j] = (log.exp() * 0.01) as f32; // scales ~1e-2
                }
            }
        }
        s
    }

    /// Exact-rank matrices are recovered to numerical precision at the
    /// true rank (the rank-1 offset from the 0.01 magnitude folds into the
    /// first component, so rank r+1 is exact; use that).
    #[test]
    fn exact_rank_recovers() {
        let (rows, cols, true_r) = (24, 12, 2);
        let s = synthetic(rows, cols, true_r);
        let f = factorize_scales(&s, rows, cols, true_r + 1, 60);
        let err = f.max_ratio_err(&s);
        assert!(err < 1.0001, "exact-rank recovery missed: ratio {err}");
    }

    /// More rank never hurts on the metric the SVD optimizes: the RMS log
    /// error is non-increasing in r (Eckart–Young). The worst-case ratio
    /// is an infinity norm and may wiggle between ranks — that is why
    /// [`calibrate_rank`] sweeps on it instead of assuming monotonicity —
    /// but it too must reach ~exact by full-ish rank.
    #[test]
    fn error_shrinks_with_rank() {
        let (rows, cols) = (20, 10);
        let s = synthetic(rows, cols, 4);
        let facs: Vec<ScaleFactor> = (1..=5)
            .map(|r| factorize_scales(&s, rows, cols, r, 60))
            .collect();
        let rms: Vec<f64> = facs.iter().map(|f| f.rms_log_err(&s)).collect();
        for w in rms.windows(2) {
            assert!(
                w[1] <= w[0] * 1.000_001,
                "RMS log error must not grow with rank: {rms:?}"
            );
        }
        let worst: Vec<f32> = facs.iter().map(|f| f.max_ratio_err(&s)).collect();
        assert!(
            worst[4] < 1.001,
            "full-ish rank must be near exact: {worst:?}"
        );
    }

    /// The storage arithmetic from the spec: r = 4 on a 4096x4096/g=32
    /// tensor stores ~3% of the dense f32 scale bytes.
    #[test]
    fn storage_matches_the_spec_arithmetic() {
        let f = ScaleFactor {
            rows: 4096,
            cols: 4096 / 32,
            rank: 4,
            u: vec![],
            v: vec![],
        };
        let dense = f.dense_bytes(4);
        let stored = f.stored_bytes();
        let ratio = stored as f64 / dense as f64;
        assert!(
            ratio < 0.033,
            "r=4 must cut scale bytes by ~97%, got {:.1}%",
            ratio * 100.0
        );
    }

    /// Reconstructions are positive by construction, even where ALS errs.
    #[test]
    fn reconstruction_is_always_positive() {
        let (rows, cols) = (16, 8);
        let s = synthetic(rows, cols, 5);
        let f = factorize_scales(&s, rows, cols, 1, 8); // deliberately under-ranked
        for i in 0..rows {
            for j in 0..cols {
                assert!(f.scale_at(i, j) > 0.0);
            }
        }
    }

    /// calibrate_rank returns the smallest clearing rank, and None when the
    /// bound is unreachable.
    #[test]
    fn calibration_sweep_finds_the_knee() {
        let (rows, cols) = (24, 12);
        let s = synthetic(rows, cols, 2);
        let (r, f) = calibrate_rank(&s, rows, cols, 8, 60, 1.001).expect("reachable");
        assert!(
            r <= 3,
            "a rank-2(+offset) matrix must clear by rank 3, got {r}"
        );
        assert!(f.max_ratio_err(&s) <= 1.001);
        // An impossible bound: nothing clears ratio 1.0 minus epsilon.
        assert!(calibrate_rank(&s, rows, cols, 1, 4, 0.5).is_none());
    }
}
