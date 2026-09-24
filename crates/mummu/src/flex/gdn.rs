//! **The fused host GDN decode step (SPEC P3): one function where nine
//! dispatches were.**
//!
//! A host-resident Gated `DeltaNet` layer at decode (`t == 1`) spends more on
//! *op plumbing* than on arithmetic: the tensor path issues ~9 small Burn
//! ops (conv window cat/mul/sum, `SiLU`, three narrows, two L2 norms, a
//! repeat, the recurrence's five ops, the gated `RMSNorm`) per layer per
//! token, each with dispatch overhead and fresh allocations, over tensors
//! of a few kilobytes. The information-theoretic floor is two passes over
//! the recurrent state `S` (~64 KiB/head — L2-resident) plus one sweep
//! over ~50 KB of activations: tens of microseconds, not milliseconds.
//!
//! [`gdn_step`] evaluates the whole middle of the layer — everything
//! between the input projections and the output projection — as one
//! host function on plain `f32` slices:
//!
//! 1. **Conv + `SiLU` + split, one sweep** (P3.3): the depthwise causal conv
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
//! 4. **Gated `RMSNorm` fused into the head epilogue**: `RMS(o) * gamma *
//!    act(z)` per head, written straight into the output slice. `act` is
//!    the layer's [`GdnGate`]: `silu` for qwen35, `sigmoid` for qwen4exp —
//!    the one numerical difference between the two families' `DeltaNets`.
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

use mummu_num::f32_from_usize;
use rayon::prelude::*;

/// The activation applied to the gate `z` in the `DeltaNet`'s gated output
/// `RMSNorm` (`RMS(o) * gamma * act(z)`).
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

/// How the `DeltaNet` L2-normalizes each q and k head before the recurrence.
///
/// The two forms agree to ~ε/(2‖x‖²) relative, which is invisible at unit
/// norm but not on real checkpoints: Flash-Next's keys reach ‖k‖ ≈ 1e-3
/// after conv + `SiLU` (layers 16, 28, 34, 38 of the parity prompt), where
/// `max(‖x‖, 1e-6)` and `sqrt(‖x‖² + 1e-6)` differ by up to 40% per head
/// and the block output by 1.5e-2 relative. Measured against a
/// full-precision llama.cpp b10991 dump (teacher-forced, same inputs):
/// [`GdnL2::AddEps`] reproduces its `k_conv_predelta` to 5e-8,
/// [`GdnL2::ClampNorm`] misses by 2.8e-2. Deliberately no `Default`, as
/// for [`GdnGate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnL2 {
    /// `x / max(‖x‖, ε)` — `ggml_l2_norm`, which llama.cpp's `DeltaNet` graphs
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
    /// Epsilon inside the gated `RMSNorm`.
    pub norm_eps: f32,
    /// `1 / sqrt(ds)`, folded into the normalized q.
    pub scale: f32,
    /// Depthwise conv taps, `[conv_dim][kk]` row-major, tap 0 = oldest.
    pub conv_w: Vec<f32>,
    /// Decay bias per value head.
    pub dt_bias: Vec<f32>,
    /// `-exp(A_log)` per value head (negative).
    pub a: Vec<f32>,
    /// Gated `RMSNorm` gain over `ds`.
    pub gamma: Vec<f32>,
    /// Activation on `z` in the gated `RMSNorm` epilogue.
    pub gate: GdnGate,
}

impl GdnMiddle {
    /// Ring length in floats: `conv_dim * (kk - 1)`, channel-major,
    /// position 0 = oldest.
    #[must_use]
    pub const fn ring_len(&self) -> usize {
        self.conv_dim * (self.kk - 1)
    }

    /// State length in floats: `hv * ds * ds`, head-major, each head's
    /// `S[i_key][j_value]` row-major.
    #[must_use]
    pub const fn state_len(&self) -> usize {
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

/// This token's activations, bundled so [`gdn_step`] stays inside the
/// argument budget.
///
/// Every field is a borrowed slice, so the struct is `Copy` and passing it
/// by value moves four fat pointers — the callee destructures it back into
/// the same four bindings it always had.
#[derive(Clone, Copy, Debug)]
pub struct GdnInputs<'a> {
    /// The qkv projection's output, `[conv_dim]`, PRE-conv.
    pub mixed: &'a [f32],
    /// The gate projection's output, `[d_inner]`.
    pub z: &'a [f32],
    /// Per-v-head β logits, `[hv]`.
    pub beta_logits: &'a [f32],
    /// Per-v-head α logits, `[hv]`.
    pub alpha_logits: &'a [f32],
}

/// Every length [`gdn_step`] depends on, in one place so the step itself
/// stays inside the line budget.
///
/// The assertions, their order and their messages are exactly the ones that
/// used to open the step.
#[inline]
fn assert_shapes(
    mid: &GdnMiddle,
    input: GdnInputs<'_>,
    ring: &[f32],
    state: &[f32],
    gated: &[f32],
) {
    let (hv, ds, kk) = (mid.hv, mid.ds, mid.kk);
    assert_eq!(input.mixed.len(), mid.conv_dim, "mixed width");
    assert_eq!(input.z.len(), mid.d_inner, "z width");
    assert_eq!(input.beta_logits.len(), hv, "beta width");
    assert_eq!(input.alpha_logits.len(), hv, "alpha width");
    assert_eq!(ring.len(), mid.ring_len(), "ring length");
    assert_eq!(state.len(), mid.state_len(), "state length");
    assert_eq!(gated.len(), mid.d_inner, "output length");
    assert_eq!(mid.conv_w.len(), mid.conv_dim * kk, "conv taps");
    assert_eq!(mid.gamma.len(), ds, "norm gain");
}

/// One fused GDN decode step (batch 1, one token). See the module doc for
/// the pass structure. Arguments:
///
/// - `input`: this token's activations — see [`GdnInputs`].
/// - `ring`: the rolling conv window, `[conv_dim * (kk-1)]` (mutated).
/// - `state`: the recurrent memory, `[hv * ds * ds]` (mutated).
/// - `gated`: the output, `[d_inner]` — the vector the out projection
///   consumes (already RMS-normed and z-gated).
///
/// Every multiply-then-add below is written as two statements on purpose:
/// two roundings, matching the tensor path and the 1e-5 oracle in
/// `models/qwen35.rs` — a fused multiply-add would round once and drift.
/// Names: `q`, `k`, `z` are the paper's symbols; `stq`/`stk` are `Sᵀq` /
/// `Sᵀk`, `s_mat` the head's state `S`.
///
/// # Panics
/// On any slice length mismatch — a wiring bug, never a workload.
pub fn gdn_step(
    mid: &GdnMiddle,
    input: GdnInputs<'_>,
    ring: &mut [f32],
    state: &mut [f32],
    gated: &mut [f32],
) {
    assert_shapes(mid, input, ring, state, gated);
    let GdnInputs {
        mixed,
        z,
        beta_logits,
        alpha_logits,
    } = input;
    let (hk, ds, kk) = (mid.hk, mid.ds, mid.kk);

    // Pass 1: FIR + ring roll + SiLU, one sweep over the mix.
    let taps = kk - 1;
    let mut conv_out = vec![0f32; mid.conv_dim];
    for ch in 0..mid.conv_dim {
        let taps_w = &mid.conv_w[ch * kk..(ch + 1) * kk];
        let ring_ch = &mut ring[ch * taps..(ch + 1) * taps];
        let mut acc = mixed[ch] * taps_w[taps];
        for tap in 0..taps {
            let tap_term = ring_ch[tap] * taps_w[tap];
            acc += tap_term;
        }
        // Roll: drop the oldest, append this token's mix value.
        for tap in 0..taps - 1 {
            ring_ch[tap] = ring_ch[tap + 1];
        }
        ring_ch[taps - 1] = mixed[ch];
        conv_out[ch] = silu(acc);
    }

    // Split is an offset.
    let (q_raw, rest) = conv_out.split_at(mid.key_dim);
    let (k_raw, v_all) = rest.split_at(mid.key_dim);

    // L2-normalize q and k per k-head; the attention scale folds into q.
    let mut qn = vec![0f32; mid.key_dim];
    let mut kn = vec![0f32; mid.key_dim];
    for head in 0..hk {
        let seg = head * ds..(head + 1) * ds;
        let sum_sq = |xs: &[f32]| xs.iter().map(|e| e * e).sum::<f32>();
        let nq = mid.l2.norm(sum_sq(&q_raw[seg.clone()]), mid.l2_eps);
        let nk = mid.l2.norm(sum_sq(&k_raw[seg.clone()]), mid.l2_eps);
        let (sq, sk) = (mid.scale / nq, 1.0 / nk);
        for idx in seg {
            qn[idx] = q_raw[idx] * sq;
            kn[idx] = k_raw[idx] * sk;
        }
    }

    // Passes 2+3 per value head, heads across the pool. Each head owns its
    // disjoint state and output slices; q/k/v/gates are shared reads.
    state
        .par_chunks_mut(ds * ds)
        .zip(gated.par_chunks_mut(ds))
        .enumerate()
        .for_each(|(head, (s_mat, out))| {
            let kh = head % hk;
            let q = &qn[kh * ds..(kh + 1) * ds];
            let k = &kn[kh * ds..(kh + 1) * ds];
            let vh = &v_all[head * ds..(head + 1) * ds];
            let zh = &z[head * ds..(head + 1) * ds];
            let beta = sigmoid(beta_logits[head]);
            let decay_log = softplus(alpha_logits[head] + mid.dt_bias[head]) * mid.a[head];
            let gamma = decay_log.exp();

            let mut scratch = vec![0f32; 2 * ds];
            let (stq, stk) = scratch.split_at_mut(ds);
            // Read pass: stq = S^T q, stk = S^T k — one stream over S.
            for ri in 0..ds {
                let row = &s_mat[ri * ds..(ri + 1) * ds];
                let (qi, ki) = (q[ri], k[ri]);
                for cj in 0..ds {
                    let q_term = row[cj] * qi;
                    stq[cj] += q_term;
                    let k_term = row[cj] * ki;
                    stk[cj] += k_term;
                }
            }
            let qk: f32 = q.iter().zip(k).map(|(qa, kb)| qa * kb).sum();

            // dv = beta * (v - gamma * S^T k); o = gamma * S^T q + (q.k) dv.
            // (q is pre-scaled, so (q.k) carries the attention scale too —
            // exactly the sequential order, which scales q before both the
            // output read and the correction term.)
            let mut o_and_dv = vec![0f32; 2 * ds];
            let (o_head, dv) = o_and_dv.split_at_mut(ds);
            for cj in 0..ds {
                let decayed_k = gamma * stk[cj];
                dv[cj] = beta * (vh[cj] - decayed_k);
                let decayed_q = gamma * stq[cj];
                let correction = qk * dv[cj];
                o_head[cj] = decayed_q + correction;
            }

            // Write pass: S = gamma * S + k (x) dv.
            for ri in 0..ds {
                let row = &mut s_mat[ri * ds..(ri + 1) * ds];
                let ki = k[ri];
                for cj in 0..ds {
                    let decayed = gamma * row[cj];
                    let update = ki * dv[cj];
                    row[cj] = decayed + update;
                }
            }

            // Gated RMSNorm epilogue: RMS(o) * gamma_norm * act(z). The
            // gate match sits outside the loop so each arm stays a plain
            // inlined scalar loop — the qwen35 (silu) arm is the loop that
            // was here before the gate became a parameter.
            let ms = o_head.iter().map(|e| e * e).sum::<f32>() / f32_from_usize(ds);
            let inv = 1.0 / (ms + mid.norm_eps).sqrt();
            match mid.gate {
                GdnGate::Silu => {
                    for cj in 0..ds {
                        out[cj] = o_head[cj] * inv * mid.gamma[cj] * silu(zh[cj]);
                    }
                }
                GdnGate::Sigmoid => {
                    for cj in 0..ds {
                        out[cj] = o_head[cj] * inv * mid.gamma[cj] * sigmoid(zh[cj]);
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
            GdnInputs {
                mixed: &[3.0, 1.0, 1.0],
                z: &[1.0],
                beta_logits: &[0.0],
                alpha_logits: &[0.0],
            },
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
        use mummu_num::f32_from_u64;
        let ds = 16;
        let mut lcg = 12345u64;
        let mut rand = move || {
            lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (f32_from_u64(lcg >> 33) / f32_from_u64(1u64 << 31)) - 0.5
        };
        for _ in 0..20 {
            let s0: Vec<f32> = (0..ds * ds).map(|_| rand()).collect();
            let q: Vec<f32> = (0..ds).map(|_| rand()).collect();
            let k: Vec<f32> = (0..ds).map(|_| rand()).collect();
            let v: Vec<f32> = (0..ds).map(|_| rand()).collect();
            let (gamma, beta) = (0.9f32, 0.7f32);

            // Naive: S' = gamma S; vhat = S'^T k; dv = beta (v - vhat);
            // S1 = S' + k dv^T; o = S1^T q. Products and sums stay separate
            // statements so this reference rounds the way the kernel does.
            let sp: Vec<f32> = s0.iter().map(|x| gamma * x).collect();
            let mut vhat = vec![0f32; ds];
            for ri in 0..ds {
                for cj in 0..ds {
                    let term = sp[ri * ds + cj] * k[ri];
                    vhat[cj] += term;
                }
            }
            let dv: Vec<f32> = (0..ds).map(|cj| beta * (v[cj] - vhat[cj])).collect();
            let mut s1 = sp;
            for ri in 0..ds {
                for cj in 0..ds {
                    let update = k[ri] * dv[cj];
                    s1[ri * ds + cj] += update;
                }
            }
            let mut o_naive = vec![0f32; ds];
            for ri in 0..ds {
                for cj in 0..ds {
                    let term = s1[ri * ds + cj] * q[ri];
                    o_naive[cj] += term;
                }
            }

            // Fused: stq = S0^T q; stk = S0^T k; dv = beta (v - gamma stk);
            // o = gamma stq + (q.k) dv.
            let mut stq = vec![0f32; ds];
            let mut stk = vec![0f32; ds];
            for ri in 0..ds {
                for cj in 0..ds {
                    let q_term = s0[ri * ds + cj] * q[ri];
                    stq[cj] += q_term;
                    let k_term = s0[ri * ds + cj] * k[ri];
                    stk[cj] += k_term;
                }
            }
            let qk: f32 = q.iter().zip(&k).map(|(qa, kb)| qa * kb).sum();
            for cj in 0..ds {
                let decayed_k = gamma * stk[cj];
                let dvj = beta * (v[cj] - decayed_k);
                let decayed_q = gamma * stq[cj];
                let correction = qk * dvj;
                let of = decayed_q + correction;
                assert!(
                    (of - o_naive[cj]).abs() < 1e-4,
                    "fused {of} vs naive {} at {cj}",
                    o_naive[cj]
                );
                assert!((dvj - dv[cj]).abs() < 1e-5);
            }
        }
    }

    /// The gated `RMSNorm` epilogue applies the configured gate and nothing
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
                GdnInputs {
                    mixed: &mixed,
                    z: &z,
                    beta_logits: &[0.0],
                    alpha_logits: &[0.0],
                },
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
                GdnInputs {
                    mixed: &mixed,
                    z: &[0.0, 0.0],
                    beta_logits: &[0.0],
                    alpha_logits: &[0.0],
                },
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
