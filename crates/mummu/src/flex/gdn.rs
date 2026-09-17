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
//!    act(z)` per head, written straight into the output slice. `act` is
//!    the layer's [`GdnGate`]: `silu` for qwen35, `sigmoid` for qwen4exp —
//!    the one numerical difference between the two families' DeltaNets.
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

/// The activation applied to the gate `z` in the DeltaNet's gated output
/// RMSNorm (`RMS(o) * gamma * act(z)`).
///
/// Two families share every other line of the GDN block and differ only
/// here: llama.cpp's `qwen35` graph builds `ggml_silu(z)`, its `qwen4exp`
/// graph `ggml_sigmoid(z)` (`build_norm_gated` in each). The GGUF header
/// carries no key for it — the architecture name decides — so the loader
/// that knows the architecture sets it. Deliberately no `Default`: a config
/// literal that forgets to choose must fail to compile, not silently run
/// the other family's gate (that error is invisible until a parity run).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnGate {
    /// `z * sigmoid(z)` — qwen35 / Qwen3.5-family checkpoints.
    Silu,
    /// `sigmoid(z)` — qwen4exp (Qwen3.8-Flash-Next).
    Sigmoid,
}

/// How the DeltaNet L2-normalizes each q and k head before the recurrence.
///
/// The two forms agree to ~ε/(2‖x‖²) relative, which is invisible at unit
/// norm but not on real checkpoints: Flash-Next's keys reach ‖k‖ ≈ 1e-3
/// after conv + SiLU (layers 16, 28, 34, 38 of the parity prompt), where
/// `max(‖x‖, 1e-6)` and `sqrt(‖x‖² + 1e-6)` differ by up to 40% per head
/// and the block output by 1.5e-2 relative. Measured against a
/// full-precision llama.cpp b10991 dump (teacher-forced, same inputs):
/// [`GdnL2::AddEps`] reproduces its `k_conv_predelta` to 5e-8,
/// [`GdnL2::ClampNorm`] misses by 2.8e-2. Deliberately no `Default`, as
/// for [`GdnGate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnL2 {
    /// `x / max(‖x‖, ε)` — `ggml_l2_norm`, which llama.cpp's DeltaNet graphs
    /// used before PR #28068 (b10760 and older, e.g. ollama 0.34.0's bundled
    /// llama-server) and qwen35 here used until 2026-09-16. Kept to reproduce
    /// those references.
    ClampNorm,
    /// `x / sqrt(‖x‖² + ε)` — transformers' (FLA) `l2norm` and llama.cpp's
    /// `build_gdn_l2_norm` (`rms_norm(x, ε/n) / sqrt(n)`) since PR #28068,
    /// used by its qwen35 and qwen4exp graphs (b10991 onward). Both mummu
    /// families use it.
    AddEps,
}

impl GdnL2 {
    /// The L2 denominator for a head whose squared entries sum to `sum_sq`.
    ///
    /// Returned as the norm rather than its inverse so callers divide by it:
    /// `scale / n` and `scale * (1 / n)` differ by one ulp on about a quarter
    /// of f32 inputs, and the [`GdnL2::ClampNorm`] arm stays bit-identical to
    /// the fused step qwen35 measurements before 2026-09-16 were taken with.
    #[inline]
    #[must_use]
    pub fn norm(self, sum_sq: f32, eps: f32) -> f32 {
        match self {
            Self::ClampNorm => sum_sq.sqrt().max(eps),
            Self::AddEps => (sum_sq + eps).sqrt(),
        }
    }
}

impl GdnGate {
    /// The gate activation at one scalar.
    #[inline]
    #[must_use]
    pub fn apply(self, z: f32) -> f32 {
        match self {
            Self::Silu => silu(z),
            Self::Sigmoid => sigmoid(z),
        }
    }
}

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
    /// Epsilon of the q/k L2 norms (the model's `rms_norm_eps`).
    pub l2_eps: f32,
    /// Where that epsilon enters the q/k L2 norms.
    pub l2: GdnL2,
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
    /// Activation on `z` in the gated RMSNorm epilogue.
    pub gate: GdnGate,
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
        let sum_sq = |x: &[f32]| x.iter().map(|x| x * x).sum::<f32>();
        let nq = p.l2.norm(sum_sq(&q_raw[seg.clone()]), p.l2_eps);
        let nk = p.l2.norm(sum_sq(&k_raw[seg.clone()]), p.l2_eps);
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

            // Gated RMSNorm epilogue: RMS(o) * gamma_norm * act(z). The
            // gate match sits outside the loop so each arm stays a plain
            // inlined scalar loop — the qwen35 (silu) arm is the loop that
            // was here before the gate became a parameter.
            let ms = o.iter().map(|x| x * x).sum::<f32>() / ds as f32;
            let inv = 1.0 / (ms + p.norm_eps).sqrt();
            match p.gate {
                GdnGate::Silu => {
                    for j in 0..ds {
                        out[j] = o[j] * inv * p.gamma[j] * silu(zh[j]);
                    }
                }
                GdnGate::Sigmoid => {
                    for j in 0..ds {
                        out[j] = o[j] * inv * p.gamma[j] * sigmoid(zh[j]);
                    }
                }
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
            l2: GdnL2::ClampNorm,
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
            gate: GdnGate::Silu,
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

    /// The gated RMSNorm epilogue applies the configured gate and nothing
    /// else: with a zero state, no decay, β = ½ and q = k (so q·k = 1), the
    /// head output is `o = ½·silu(v_mix)` exactly, and the step must write
    /// `RMS(o)·gamma·sigmoid(z)` for [`GdnGate::Sigmoid`] and
    /// `RMS(o)·gamma·silu(z)` for [`GdnGate::Silu`] — the expectations are
    /// computed here from the textbook formulas, not the module's helpers.
    /// The recurrent state must not depend on the gate (it only shapes the
    /// output), and the two gates must actually disagree on these inputs,
    /// so a gate field that is ignored cannot pass.
    #[test]
    fn epilogue_applies_the_configured_gate() {
        let middle = |gate: GdnGate| GdnMiddle {
            hk: 1,
            hv: 1,
            ds: 2,
            kk: 2,
            conv_dim: 6, // 2*key_dim + d_inner = 2*2 + 2
            key_dim: 2,
            d_inner: 2,
            l2_eps: 1e-6,
            l2: GdnL2::ClampNorm,
            norm_eps: 1e-6,
            scale: 1.0,
            // Every channel passes its newest mix value straight through.
            conv_w: [0.0, 1.0].repeat(6),
            dt_bias: vec![0.0],
            a: vec![0.0], // decay exp(softplus(0)*0) = 1
            gamma: vec![0.7, -1.3],
            gate,
        };
        // [q (2) | k (2) | v (2)]: q == k, so the normalized q.k is 1.
        let v_mix = [1.2f32, -0.7];
        let mixed = [0.8, -0.3, 0.8, -0.3, v_mix[0], v_mix[1]];
        let z = [-3.0f32, 2.5];
        let run = |gate: GdnGate| {
            let p = middle(gate);
            let mut ring = vec![0.0f32; p.ring_len()];
            let mut state = vec![0.0f32; p.state_len()];
            let mut out = vec![0.0f32; p.d_inner];
            gdn_step(
                &p,
                &mixed,
                &z,
                &[0.0],
                &[0.0],
                &mut ring,
                &mut state,
                &mut out,
            );
            (out, state)
        };

        let ref_sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let ref_silu = |x: f32| x * ref_sigmoid(x);
        // S = 0 and gamma = 1: o = (q.k) * beta * v = 0.5 * silu(v_mix).
        let o: Vec<f32> = v_mix.iter().map(|&m| 0.5 * ref_silu(m)).collect();
        let inv = 1.0 / (o.iter().map(|x| x * x).sum::<f32>() / 2.0 + 1e-6).sqrt();
        let gamma = [0.7f32, -1.3];

        let (out_sig, state_sig) = run(GdnGate::Sigmoid);
        let (out_silu, state_silu) = run(GdnGate::Silu);
        for j in 0..2 {
            let want_sig = o[j] * inv * gamma[j] * ref_sigmoid(z[j]);
            let want_silu = o[j] * inv * gamma[j] * ref_silu(z[j]);
            assert!(
                (out_sig[j] - want_sig).abs() < 1e-5,
                "sigmoid epilogue at {j}: got {} want {want_sig}",
                out_sig[j]
            );
            assert!(
                (out_silu[j] - want_silu).abs() < 1e-5,
                "silu epilogue at {j}: got {} want {want_silu}",
                out_silu[j]
            );
            assert!(
                (out_sig[j] - out_silu[j]).abs() > 1e-2,
                "the gates must disagree at {j}: {} vs {}",
                out_sig[j],
                out_silu[j]
            );
        }
        assert_eq!(state_sig, state_silu, "the gate must not touch the state");
    }

    /// The q/k L2 norms use the configured epsilon form. With the state
    /// empty, no decay and `beta = 1/2`, the step writes
    /// `S[i][j] = k̂_i · ½·silu(v_j)`, so the state exposes the normalized
    /// key directly. Keys of norm ~1e-3 — the regime real Flash-Next keys
    /// reach — make `x/sqrt(‖x‖²+ε)` and `x/max(‖x‖,ε)` disagree by ~25%;
    /// both are checked against textbook formulas written out here, and
    /// against each other so an ignored field cannot pass.
    #[test]
    fn the_state_carries_keys_normalized_in_the_configured_l2_form() {
        let middle = |l2: GdnL2| GdnMiddle {
            hk: 1,
            hv: 1,
            ds: 2,
            kk: 2,
            conv_dim: 6,
            key_dim: 2,
            d_inner: 2,
            l2_eps: 1e-6,
            l2,
            norm_eps: 1e-6,
            scale: 1.0,
            conv_w: [0.0, 1.0].repeat(6), // newest mix value passes through
            dt_bias: vec![0.0],
            a: vec![0.0],
            gamma: vec![1.0, 1.0],
            gate: GdnGate::Sigmoid,
        };
        let key = [2e-3f32, -1e-3];
        let v_mix = [1.2f32, -0.7];
        let mixed = [key[0], key[1], key[0], key[1], v_mix[0], v_mix[1]];
        let run = |l2: GdnL2| {
            let p = middle(l2);
            let mut ring = vec![0.0f32; p.ring_len()];
            let mut state = vec![0.0f32; p.state_len()];
            let mut out = vec![0.0f32; p.d_inner];
            gdn_step(
                &p,
                &mixed,
                &[0.0, 0.0],
                &[0.0],
                &[0.0],
                &mut ring,
                &mut state,
                &mut out,
            );
            state
        };
        let silu64 = |x: f32| {
            let x = f64::from(x);
            x / (1.0 + (-x).exp())
        };
        let k: Vec<f64> = key.iter().map(|&x| silu64(x)).collect();
        let sum_sq: f64 = k.iter().map(|x| x * x).sum();
        let want = |inv: f64| -> Vec<f64> {
            (0..2)
                .flat_map(|i| (0..2).map(move |j| (i, j)))
                .map(|(i, j)| k[i] * inv * 0.5 * silu64(v_mix[j]))
                .collect()
        };
        let add = want(1.0 / (sum_sq + 1e-6).sqrt());
        let clamp = want(1.0 / sum_sq.sqrt().max(1e-6));
        let (got_add, got_clamp) = (run(GdnL2::AddEps), run(GdnL2::ClampNorm));
        for n in 0..4 {
            assert!(
                (f64::from(got_add[n]) - add[n]).abs() < 1e-4,
                "AddEps state[{n}]: got {} want {}",
                got_add[n],
                add[n]
            );
            assert!(
                (f64::from(got_clamp[n]) - clamp[n]).abs() < 1e-4,
                "ClampNorm state[{n}]: got {} want {}",
                got_clamp[n],
                clamp[n]
            );
            assert!(
                (got_add[n] - got_clamp[n]).abs() > 0.1 * got_clamp[n].abs(),
                "the two L2 forms must disagree on a 1e-3-norm key at {n}: {} vs {}",
                got_add[n],
                got_clamp[n]
            );
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
        assert!((GdnGate::Sigmoid.apply(-100.0)).abs() < 1e-6);
        assert!((GdnGate::Sigmoid.apply(100.0) - 1.0).abs() < 1e-6);
        assert!((GdnGate::Silu.apply(100.0) - 100.0).abs() < 1e-3);
    }
}
