//! **The fused host GDN decode step (SPEC P3): one function where nine
//! dispatches were.**
//!
//! A host-resident Gated DeltaNet layer at decode (`t == 1`) spends more on
//! *op plumbing* than on arithmetic: the tensor path issues ~9 small Burn
//! ops (conv window cat/mul/sum, SiLU, three narrows, two L2 norms, a
//! repeat, the recurrence's five ops, the gated RMSNorm) per layer per
//! token, each with dispatch overhead and fresh allocations, over tensors
//! of a few kilobytes. The information-theoretic floor is two passes over
//! the recurrent state `S` (~64 KiB/head — L2-resident) plus one sweep
//! over ~50 KB of activations: tens of microseconds, not milliseconds.
//!
//! [`gdn_step`] evaluates the whole middle of the layer — everything
//! between the input projections and the output projection — as one
//! host function on plain `f32` slices:
//!
//! 1. **Conv + SiLU + split, one sweep** (P3.3): the depthwise causal conv
//!    at decode is a `kk`-tap FIR against a rolling ring of the last
//!    `kk-1` mix columns; evaluated per channel with the ring update in
//!    the same pass, and "split" is an offset, not an op.
//! 2. **Gates in scalar registers** (P3.4): `beta = sigma(b)`,
//!    `g = softplus(a_logit + dt_bias) * a`, `gamma = exp(g)` — per head,
//!    exact transcendentals (they are nothing next to the state passes).
//! 3. **Two-pass recurrence via output correction** (P3.1): the identity
//!    `S_t^T q = gamma * (S_{t-1}^T q) + (q . k) * dv` lets ONE read pass
//!    over `S` produce both `S^T q` and `S^T k`, and one write pass apply
//!    the decay and the rank-1 update — two passes over state where the
//!    naive order takes three. Heads are independent and run across the
//!    rayon pool.
//! 4. **Gated RMSNorm fused into the head epilogue**: `RMS(o) * gamma *
//!    silu(z)` per head, written straight into the output slice.
//!
//! The projections stay OUTSIDE this function on purpose: they already run
//! as single packed-GEMV dispatches (the VNNI twin path), and keeping them
//! at the tensor level means the fused middle composes with every weight
//! format the projections support. Exactness: same f32 arithmetic as the
//! tensor path up to summation order — the oracle test in
//! `models/qwen35.rs` holds the two to 1e-5 with the recurrence state
//! carried across steps.
//!
//! `MUMMU_FUSED_GDN=0/off/false` restores the tensor path (the repo's
//! standard downgrade contract); [`force_disable`] is the programmatic
//! kill switch tests use.

use rayon::prelude::*;

/// Everything the fused middle needs besides the per-token activations:
/// dimensions and the small per-layer weights, extracted once per layer at
/// first use and cached on the module.
#[derive(Debug)]
pub struct GdnMiddle {
    /// Key/query heads.
    pub hk: usize,
    /// Value heads (a multiple of `hk`; head `h` reads k-head `h % hk`).
    pub hv: usize,
    /// Per-head key/value width.
    pub ds: usize,
    /// Conv kernel taps.
    pub kk: usize,
    /// `2 * hk * ds + hv * ds` — the mix width.
    pub conv_dim: usize,
    /// `hk * ds`.
    pub key_dim: usize,
    /// `hv * ds`.
    pub d_inner: usize,
    /// Clamp floor for the q/k L2 norms (the model's `rms_norm_eps`).
    pub l2_eps: f32,
    /// Epsilon inside the gated RMSNorm.
    pub norm_eps: f32,
    /// `1 / sqrt(ds)`, folded into the normalized q.
    pub scale: f32,
    /// Depthwise conv taps, `[conv_dim][kk]` row-major, tap 0 = oldest.
    pub conv_w: Vec<f32>,
    /// Decay bias per value head.
    pub dt_bias: Vec<f32>,
    /// `-exp(A_log)` per value head (negative).
    pub a: Vec<f32>,
    /// Gated RMSNorm gain over `ds`.
    pub gamma: Vec<f32>,
}

impl GdnMiddle {
    /// Ring length in floats: `conv_dim * (kk - 1)`, channel-major,
    /// position 0 = oldest.
    #[must_use]
    pub fn ring_len(&self) -> usize {
        self.conv_dim * (self.kk - 1)
    }

    /// State length in floats: `hv * ds * ds`, head-major, each head's
    /// `S[i_key][j_value]` row-major.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.hv * self.ds * self.ds
    }
}

/// Is the fused host path enabled? `MUMMU_FUSED_GDN`, default on;
/// `0`/`off`/`false` restores the tensor path. [`force_disable`] wins over
/// the env — A/B tests of the tensor path use it.
#[must_use]
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env_on = *ON.get_or_init(|| {
        std::env::var("MUMMU_FUSED_GDN").map_or(true, |v| {
            !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        })
    });
    env_on && !FORCE_OFF.load(std::sync::atomic::Ordering::Relaxed)
}

static FORCE_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Programmatic kill switch (stronger than the env). Not per-thread —
/// callers serialize, exactly like `flex::registry::force_disable`.
pub fn force_disable(v: bool) {
    FORCE_OFF.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Numerically-stable `softplus(x) = ln(1 + e^x)`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically-stable logistic sigmoid.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// One fused GDN decode step (batch 1, one token). See the module doc for
/// the pass structure. Slices:
///
/// - `mixed`: the qkv projection's output, `[conv_dim]`, PRE-conv.
/// - `z`: the gate projection's output, `[d_inner]`.
/// - `beta_logits`, `alpha_logits`: `[hv]`.
/// - `ring`: the rolling conv window, `[conv_dim * (kk-1)]` (mutated).
/// - `state`: the recurrent memory, `[hv * ds * ds]` (mutated).
/// - `gated`: the output, `[d_inner]` — the vector the out projection
///   consumes (already RMS-normed and z-gated).
///
/// # Panics
/// On any slice length mismatch — a wiring bug, never a workload.
#[allow(clippy::too_many_arguments)] // the layer's natural arity
pub fn gdn_step(
    p: &GdnMiddle,
    mixed: &[f32],
    z: &[f32],
    beta_logits: &[f32],
    alpha_logits: &[f32],
    ring: &mut [f32],
    state: &mut [f32],
    gated: &mut [f32],
) {
    let (hk, hv, ds, kk) = (p.hk, p.hv, p.ds, p.kk);
    assert_eq!(mixed.len(), p.conv_dim, "mixed width");
    assert_eq!(z.len(), p.d_inner, "z width");
    assert_eq!(beta_logits.len(), hv, "beta width");
    assert_eq!(alpha_logits.len(), hv, "alpha width");
    assert_eq!(ring.len(), p.ring_len(), "ring length");
    assert_eq!(state.len(), p.state_len(), "state length");
    assert_eq!(gated.len(), p.d_inner, "output length");
    assert_eq!(p.conv_w.len(), p.conv_dim * kk, "conv taps");
    assert_eq!(p.gamma.len(), ds, "norm gain");

    // Pass 1: FIR + ring roll + SiLU, one sweep over the mix.
    let taps = kk - 1;
    let mut conv_out = vec![0f32; p.conv_dim];
    for c in 0..p.conv_dim {
        let w = &p.conv_w[c * kk..(c + 1) * kk];
        let r = &mut ring[c * taps..(c + 1) * taps];
        let mut y = mixed[c] * w[taps];
        for t in 0..taps {
            y += r[t] * w[t];
        }
        // Roll: drop the oldest, append this token's mix value.
        for t in 0..taps - 1 {
            r[t] = r[t + 1];
        }
        r[taps - 1] = mixed[c];
        conv_out[c] = silu(y);
    }

    // Split is an offset.
    let (q_raw, rest) = conv_out.split_at(p.key_dim);
    let (k_raw, v) = rest.split_at(p.key_dim);

    // L2-normalize q and k per k-head; the attention scale folds into q.
    let mut qn = vec![0f32; p.key_dim];
    let mut kn = vec![0f32; p.key_dim];
    for h in 0..hk {
        let seg = h * ds..(h + 1) * ds;
        let nq = q_raw[seg.clone()]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(p.l2_eps);
        let nk = k_raw[seg.clone()]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(p.l2_eps);
        let (sq, sk) = (p.scale / nq, 1.0 / nk);
        for i in seg {
            qn[i] = q_raw[i] * sq;
            kn[i] = k_raw[i] * sk;
        }
    }

    // Passes 2+3 per value head, heads across the pool. Each head owns its
    // disjoint state and output slices; q/k/v/gates are shared reads.
    state
        .par_chunks_mut(ds * ds)
        .zip(gated.par_chunks_mut(ds))
        .enumerate()
        .for_each(|(h, (s, out))| {
            let kh = h % hk;
            let q = &qn[kh * ds..(kh + 1) * ds];
            let k = &kn[kh * ds..(kh + 1) * ds];
            let vh = &v[h * ds..(h + 1) * ds];
            let zh = &z[h * ds..(h + 1) * ds];
            let beta = sigmoid(beta_logits[h]);
            let g = softplus(alpha_logits[h] + p.dt_bias[h]) * p.a[h];
            let gamma = g.exp();

            let mut scratch = vec![0f32; 2 * ds];
            let (u, w) = scratch.split_at_mut(ds);
            // Read pass: u = S^T q, w = S^T k — one stream over S.
            for i in 0..ds {
                let row = &s[i * ds..(i + 1) * ds];
                let (qi, ki) = (q[i], k[i]);
                for j in 0..ds {
                    u[j] += row[j] * qi;
                    w[j] += row[j] * ki;
                }
            }
            let qk: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();

            // dv = beta * (v - gamma * S^T k); o = gamma * S^T q + (q.k) dv.
            // (q is pre-scaled, so (q.k) carries the attention scale too —
            // exactly the sequential order, which scales q before both the
            // output read and the correction term.)
            let mut o_and_dv = vec![0f32; 2 * ds];
            let (o, dv) = o_and_dv.split_at_mut(ds);
            for j in 0..ds {
                dv[j] = beta * (vh[j] - gamma * w[j]);
                o[j] = gamma * u[j] + qk * dv[j];
            }

            // Write pass: S = gamma * S + k (x) dv.
            for i in 0..ds {
                let row = &mut s[i * ds..(i + 1) * ds];
                let ki = k[i];
                for j in 0..ds {
                    row[j] = gamma * row[j] + ki * dv[j];
                }
            }

            // Gated RMSNorm epilogue: RMS(o) * gamma_norm * silu(z).
            let ms = o.iter().map(|x| x * x).sum::<f32>() / ds as f32;
            let inv = 1.0 / (ms + p.norm_eps).sqrt();
            for j in 0..ds {
                out[j] = o[j] * inv * p.gamma[j] * silu(zh[j]);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring rolls oldest-out, newest-in, and the FIR reads taps in age
    /// order — pinned on a hand-computed 2-channel, 3-tap example.
    #[test]
    fn fir_and_ring_are_age_ordered() {
        let p = GdnMiddle {
            hk: 1,
            hv: 1,
            ds: 1,
            kk: 3,
            conv_dim: 3, // 2*key_dim + d_inner = 2 + 1
            key_dim: 1,
            d_inner: 1,
            l2_eps: 1e-6,
            norm_eps: 1e-6,
            scale: 1.0,
            // Per channel: taps [oldest, middle, newest].
            conv_w: vec![
                1.0, 10.0, 100.0, // channel 0 (q)
                0.0, 0.0, 1.0, // channel 1 (k): identity on the newest
                0.0, 0.0, 1.0, // channel 2 (v): identity on the newest
            ],
            dt_bias: vec![0.0],
            a: vec![0.0], // gamma = exp(softplus(0)*0) = 1: no decay
            gamma: vec![1.0],
        };
        let mut ring = vec![
            1.0, 2.0, // channel 0: oldest 1, newer 2
            0.0, 0.0, 0.0, 0.0,
        ];
        let mut state = vec![0.0f32];
        let mut out = vec![0.0f32];
        gdn_step(
            &p,
            &[3.0, 1.0, 1.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &mut ring,
            &mut state,
            &mut out,
        );
        // Channel 0 FIR: 1*1 + 2*10 + 3*100 = 321 (then SiLU ~= 321).
        // Ring rolled: [2, 3].
        assert_eq!(&ring[0..2], &[2.0, 3.0]);
        // The state update ran: S was 0, so v_hat = 0, dv = sigmoid(0)*(v),
        // v = silu(1); S[0][0] = k*dv with k normalized to 1.
        let v_act = silu(1.0);
        let expect_s = 0.5 * v_act; // beta = sigmoid(0) = 0.5, k = 1
        assert!((state[0] - expect_s).abs() < 1e-6, "state {state:?}");
    }

    /// The two-pass output correction equals the naive three-pass order on
    /// random data: decay-then-read-then-update vs the fused identity.
    #[test]
    fn output_correction_matches_naive_order() {
        let ds = 16;
        let mut lcg = 12345u64;
        let mut rand = move || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((lcg >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        for _ in 0..20 {
            let s0: Vec<f32> = (0..ds * ds).map(|_| rand()).collect();
            let q: Vec<f32> = (0..ds).map(|_| rand()).collect();
            let k: Vec<f32> = (0..ds).map(|_| rand()).collect();
            let v: Vec<f32> = (0..ds).map(|_| rand()).collect();
            let (gamma, beta) = (0.9f32, 0.7f32);

            // Naive: S' = gamma S; vhat = S'^T k; dv = beta (v - vhat);
            // S1 = S' + k dv^T; o = S1^T q.
            let sp: Vec<f32> = s0.iter().map(|x| gamma * x).collect();
            let mut vhat = vec![0f32; ds];
            for i in 0..ds {
                for j in 0..ds {
                    vhat[j] += sp[i * ds + j] * k[i];
                }
            }
            let dv: Vec<f32> = (0..ds).map(|j| beta * (v[j] - vhat[j])).collect();
            let mut s1 = sp.clone();
            for i in 0..ds {
                for j in 0..ds {
                    s1[i * ds + j] += k[i] * dv[j];
                }
            }
            let mut o_naive = vec![0f32; ds];
            for i in 0..ds {
                for j in 0..ds {
                    o_naive[j] += s1[i * ds + j] * q[i];
                }
            }

            // Fused: u = S0^T q; w = S0^T k; dv = beta (v - gamma w);
            // o = gamma u + (q.k) dv.
            let mut u = vec![0f32; ds];
            let mut w = vec![0f32; ds];
            for i in 0..ds {
                for j in 0..ds {
                    u[j] += s0[i * ds + j] * q[i];
                    w[j] += s0[i * ds + j] * k[i];
                }
            }
            let qk: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();
            for j in 0..ds {
                let dvj = beta * (v[j] - gamma * w[j]);
                let of = gamma * u[j] + qk * dvj;
                assert!(
                    (of - o_naive[j]).abs() < 1e-4,
                    "fused {of} vs naive {} at {j}",
                    o_naive[j]
                );
                assert!((dvj - dv[j]).abs() < 1e-5);
            }
        }
    }

    /// Stable transcendentals at the extremes.
    #[test]
    fn gates_are_stable_at_extremes() {
        assert!(softplus(100.0).is_finite());
        assert!((softplus(100.0) - 100.0).abs() < 1e-3);
        assert!(softplus(-100.0) >= 0.0);
        assert!(sigmoid(100.0) <= 1.0 && sigmoid(-100.0) >= 0.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!(silu(-100.0).abs() < 1e-6);
    }
}
