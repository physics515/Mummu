//! **Incoherence processing: spread the outliers before quantizing
//! (SPEC P2.2's rotation half).**
//!
//! Wider quantization groups (g = 64–128, the route below 4.5 bpw) pay
//! through group dynamic range: one outlier owns its group's scale and the
//! other 63 values quantize to noise. The fix is algebraic, not
//! statistical: quantize `W~ = H_out W H_in^T` for randomized orthogonal
//! `H = (1/sqrt(n)) F D` (F the Walsh–Hadamard transform, D a random sign
//! diagonal), and fold `H^{-1} = H^T` into the adjacent ops offline — a
//! rotation composed with a linear map is a linear map, so the runtime
//! never sees it. Under the rotation each coefficient becomes a
//! sign-randomized average of a whole row/column, so group maxima
//! concentrate: `max |w~| / sigma -> O(sqrt(log n))` instead of "whatever
//! one channel spiked to", and g = 64–128 becomes near-lossless.
//!
//! This module is the transform + its measurement: [`fwht`] (in-place
//! O(n log n)), [`rotate_rows`]/[`rotate_cols`] with deterministic sign
//! diagonals, exact inverses, and [`incoherence`] — the max/sigma ratio
//! the calibration sweep watches. Folding into neighbors is per-model
//! wiring: for a `y = x W` pair the input rotation folds into the previous
//! layer's output projection (or the embedding), the output rotation into
//! the next consumer; RMSNorm commutes with rotations only through its
//! scalar (the norm), so the fold crosses `diag(gamma)` by absorbing gamma
//! into W first — the same absorption SPEC P3.3 names.

/// In-place Walsh–Hadamard transform (unnormalized butterflies). Length
/// must be a power of two. Applying it twice multiplies by `len`, so the
/// orthonormal transform is `fwht(x); x /= sqrt(len)`.
///
/// # Panics
/// If `x.len()` is not a positive power of two.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two(),
        "fwht: length {n} is not a power of two"
    );
    let mut h = 1;
    while h < n {
        for block in x.chunks_mut(2 * h) {
            let (a, b) = block.split_at_mut(h);
            for i in 0..h {
                let (u, v) = (a[i], b[i]);
                a[i] = u + v;
                b[i] = u - v;
            }
        }
        h *= 2;
    }
}

/// Deterministic sign diagonal from a seed (xorshift; +1/−1 per lane).
#[must_use]
pub fn sign_diagonal(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & (1 << 33) == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Rotate every length-`cols` row of `w` (row-major `rows x cols`) by the
/// orthonormal `H = (1/sqrt(cols)) F D`: signs first, then the transform.
/// `cols` must be a power of two.
pub fn rotate_rows(w: &mut [f32], rows: usize, cols: usize, seed: u64) {
    assert_eq!(w.len(), rows * cols, "rotate_rows: shape");
    let d = sign_diagonal(seed, cols);
    #[allow(clippy::cast_possible_truncation)]
    let inv_sqrt = (1.0 / (cols as f64).sqrt()) as f32;
    for row in w.chunks_mut(cols) {
        for (v, s) in row.iter_mut().zip(&d) {
            *v *= s;
        }
        fwht(row);
        for v in row.iter_mut() {
            *v *= inv_sqrt;
        }
    }
}

/// Inverse of [`rotate_rows`] (`H^{-1} = D F / sqrt(cols)` — signs last).
pub fn rotate_rows_inverse(w: &mut [f32], rows: usize, cols: usize, seed: u64) {
    assert_eq!(w.len(), rows * cols, "rotate_rows_inverse: shape");
    let d = sign_diagonal(seed, cols);
    #[allow(clippy::cast_possible_truncation)]
    let inv_sqrt = (1.0 / (cols as f64).sqrt()) as f32;
    for row in w.chunks_mut(cols) {
        fwht(row);
        for (v, s) in row.iter_mut().zip(&d) {
            *v *= s * inv_sqrt;
        }
    }
}

/// Rotate every column of `w` by the same construction (the `H_out` side).
pub fn rotate_cols(w: &mut [f32], rows: usize, cols: usize, seed: u64) {
    assert_eq!(w.len(), rows * cols, "rotate_cols: shape");
    let d = sign_diagonal(seed, rows);
    #[allow(clippy::cast_possible_truncation)]
    let inv_sqrt = (1.0 / (rows as f64).sqrt()) as f32;
    let mut col = vec![0.0f32; rows];
    for j in 0..cols {
        for i in 0..rows {
            col[i] = w[i * cols + j] * d[i];
        }
        fwht(&mut col);
        for i in 0..rows {
            w[i * cols + j] = col[i] * inv_sqrt;
        }
    }
}

/// The incoherence statistic the sweep watches: `max |w| * sqrt(n) / ||w||_2`
/// — how far the worst coefficient sits above the RMS. 1.0 is perfectly
/// flat; a lone spike in an otherwise-zero tensor scores `sqrt(n)`. The
/// rotation drives this toward `O(sqrt(log n))`, which is what makes wide
/// quantization groups near-lossless.
#[must_use]
pub fn incoherence(w: &[f32]) -> f32 {
    let n = w.len();
    assert!(n > 0);
    let max = w.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let norm = w
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let r = (f64::from(max) * (n as f64).sqrt() / norm) as f32;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H is orthogonal: rotate + inverse is the identity to f32 precision.
    #[test]
    fn rotation_round_trips() {
        let (rows, cols) = (8, 64);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let mut r = w.clone();
        rotate_rows(&mut r, rows, cols, 42);
        rotate_rows_inverse(&mut r, rows, cols, 42);
        for (a, b) in w.iter().zip(&r) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// Norms are preserved (orthogonality, the other face).
    #[test]
    fn rotation_preserves_norms() {
        let (rows, cols) = (4, 128);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.11).cos())
            .collect();
        let before: f64 = w.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let mut r = w;
        rotate_rows(&mut r, rows, cols, 7);
        let after: f64 = r.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!((before - after).abs() / before < 1e-5);
        let mut c = r;
        rotate_cols(&mut c, rows, cols, 9);
        let after2: f64 = c.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!((before - after2).abs() / before < 1e-5);
    }

    /// The point of the exercise: an outlier-dominated tensor's
    /// incoherence collapses under the rotation — a lone spike scoring
    /// ~sqrt(n) lands near the sqrt(log n) regime, which is what lets a
    /// wide quantization group hold it without wrecking its neighbors.
    #[test]
    fn rotation_spreads_outliers() {
        let cols = 256usize;
        let mut w = vec![0.01f32; cols];
        w[17] = 100.0; // the outlier channel
        let before = incoherence(&w);
        assert!(before > 10.0, "a spike must score high, got {before}");
        rotate_rows(&mut w, 1, cols, 1234);
        let after = incoherence(&w);
        let bound = 4.0 * (cols as f32).ln().sqrt(); // generous O(sqrt(log n))
        assert!(
            after < bound && after < before / 3.0,
            "rotation must spread the spike: {before} -> {after} (bound {bound})"
        );
    }

    /// Group dynamic range actually improves: worst group max/mean over
    /// 32-wide groups drops after rotation on a spiky tensor.
    #[test]
    fn group_range_improves() {
        let cols = 512usize;
        let mut w: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.05).sin() * 0.02)
            .collect();
        for spike in [3usize, 100, 301] {
            w[spike] = 5.0;
        }
        let worst_group = |v: &[f32]| -> f32 {
            v.chunks(32)
                .map(|g| {
                    let max = g.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                    let mean = g.iter().map(|&x| x.abs()).sum::<f32>() / 32.0;
                    if mean > 0.0 { max / mean } else { 0.0 }
                })
                .fold(0.0f32, f32::max)
        };
        let before = worst_group(&w);
        let mut r = w;
        rotate_rows(&mut r, 1, cols, 99);
        let after = worst_group(&r);
        assert!(
            after < before / 2.0,
            "worst group max/mean must at least halve: {before} -> {after}"
        );
    }

    /// fwht twice is multiplication by n (the classic identity), and the
    /// length check is loud.
    #[test]
    fn fwht_identity_and_bounds() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = x.clone();
        fwht(&mut x);
        fwht(&mut x);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b * 4.0).abs() < 1e-6);
        }
        assert!(
            std::panic::catch_unwind(|| {
                let mut y = vec![0.0f32; 3];
                fwht(&mut y);
            })
            .is_err()
        );
    }
}
