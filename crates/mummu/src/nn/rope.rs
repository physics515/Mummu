//! Rotary position embeddings (RoPE), HF duplicated-half layout, computed
//! manually so the same tables serve every architecture (Qwen theta 1e6 vs
//! LFM2 theta 1e6 vs others) and every backend.

use burn::tensor::{Device, Tensor, TensorData};

use super::MAX_CONTEXT_TOKENS;

/// RoPE cos/sin tables `[1, 1, t, head_dim]` for absolute positions
/// `past..past+t`, HF duplicated-half layout (each frequency written to both
/// halves of the head dim, so [`apply_rope`]'s rotate-half math lines up).
pub fn rope_tables(
    t: usize,
    past: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> (Tensor<4>, Tensor<4>) {
    assert!(t >= 1, "rope_tables: need at least one position, got t=0");
    assert!(
        head_dim >= 2 && head_dim.is_multiple_of(2),
        "rope_tables: head_dim must be even and >= 2, got {head_dim}"
    );
    assert!(
        past + t <= MAX_CONTEXT_TOKENS,
        "rope_tables: position {past}+{t} exceeds MAX_CONTEXT_TOKENS ({MAX_CONTEXT_TOKENS})"
    );
    debug_assert!(theta > 0.0, "rope_tables: theta must be positive");

    let half = head_dim / 2;
    let mut cos = vec![0f32; t * head_dim];
    let mut sin = vec![0f32; t * head_dim];
    for i in 0..t {
        let pos = (past + i) as f32;
        for k in 0..half {
            let inv = 1.0f32 / theta.powf(2.0 * k as f32 / head_dim as f32);
            let (s, c) = (pos * inv).sin_cos();
            cos[i * head_dim + k] = c;
            cos[i * head_dim + k + half] = c;
            sin[i * head_dim + k] = s;
            sin[i * head_dim + k + half] = s;
        }
    }
    // Dtype pinned to the backend TYPE: unspecified-dtype creation follows
    // the per-DEVICE policy another alias sharing the device may have locked.
    let dtype = crate::backend::float_dtype();
    let cos = Tensor::<2>::from_data(TensorData::new(cos, [t, head_dim]), (device, dtype))
        .reshape([1, 1, t, head_dim]);
    let sin = Tensor::<2>::from_data(TensorData::new(sin, [t, head_dim]), (device, dtype))
        .reshape([1, 1, t, head_dim]);
    (cos, sin)
}

/// `rotate_half`: split the last dim in two, return `cat([-x2, x1])`.
pub fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    let dims = x.dims();
    let half = dims[3] / 2;
    assert!(
        dims[3].is_multiple_of(2) && half >= 1,
        "rotate_half: last dim must be even and >= 2, got {}",
        dims[3]
    );
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

/// Apply RoPE: `x*cos + rotate_half(x)*sin`. `cos`/`sin` come from
/// [`rope_tables`] and must cover the same `t` and `head_dim` as `x`.
pub fn apply_rope(
    x: Tensor<4>,
    cos: &Tensor<4>,
    sin: &Tensor<4>,
) -> Tensor<4> {
    let (xd, cd) = (x.dims(), cos.dims());
    assert!(
        xd[2] == cd[2] && xd[3] == cd[3],
        "apply_rope: x {xd:?} and tables {cd:?} disagree on [t, head_dim]"
    );
    debug_assert!(
        cos.dims() == sin.dims(),
        "apply_rope: cos/sin shape mismatch"
    );
    let rh = rotate_half(x.clone());
    x.mul(cos.clone()).add(rh.mul(sin.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    type Dev = burn::tensor::Device;

    fn to_vec(t: Tensor<4>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().unwrap()
    }

    #[test]
    fn rope_tables_position_zero_is_identity_rotation() {
        let device = crate::backend::cpu_device();
        let (cos, sin) = rope_tables(1, 0, 8, 1e6, &device);
        assert!(to_vec(cos).iter().all(|&c| (c - 1.0).abs() < 1e-7));
        assert!(to_vec(sin).iter().all(|&s| s.abs() < 1e-7));
    }

    #[test]
    fn rope_tables_are_unit_norm_and_offset_consistent() {
        let device = crate::backend::cpu_device();
        // cos^2 + sin^2 == 1 everywhere.
        let (cos, sin) = rope_tables(3, 2, 16, 1e4, &device);
        let (c, s) = (to_vec(cos), to_vec(sin));
        for (ci, si) in c.iter().zip(&s) {
            assert!((ci * ci + si * si - 1.0).abs() < 1e-5);
        }
        // Row for absolute position 4 must match whether reached via past=2+i=2
        // (prefill) or past=4,t=1 (decode) — the KV-cache offset invariant.
        let (cos_dec, sin_dec) = rope_tables(1, 4, 16, 1e4, &device);
        let (cd, sd) = (to_vec(cos_dec), to_vec(sin_dec));
        assert_eq!(&c[2 * 16..3 * 16], &cd[..]);
        assert_eq!(&s[2 * 16..3 * 16], &sd[..]);
    }

    #[test]
    fn rotate_half_swaps_and_negates() {
        let device = crate::backend::cpu_device();
        let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([1, 1, 1, 4]);
        assert_eq!(to_vec(rotate_half(x)), vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn apply_rope_at_position_zero_is_identity() {
        let device = crate::backend::cpu_device();
        let x = Tensor::<1>::from_floats([0.5, -1.5, 2.0, 3.5], &device).reshape([1, 1, 1, 4]);
        let (cos, sin) = rope_tables(1, 0, 4, 1e6, &device);
        let y = apply_rope(x.clone(), &cos, &sin);
        let (xv, yv) = (to_vec(x), to_vec(y));
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "head_dim must be even")]
    fn rope_tables_rejects_odd_head_dim() {
        let device = crate::backend::cpu_device();
        let _ = rope_tables(1, 0, 7, 1e6, &device);
    }

    #[test]
    #[should_panic(expected = "MAX_CONTEXT_TOKENS")]
    fn rope_tables_rejects_runaway_positions() {
        let device = crate::backend::cpu_device();
        let _ = rope_tables(1, MAX_CONTEXT_TOKENS, 8, 1e6, &device);
    }
}
