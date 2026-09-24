//! Qwen3.5 / Qwen3.8 ("qwen35") hybrid decoder.
//!
//! Gated `DeltaNet` linear attention on three of every four layers, gated
//! full attention (partial `RoPE`) on the fourth, `SwiGLU` MLPs, `RMSNorm`
//! everywhere, tied or untied head.
//!
//! Ported from llama.cpp's reference (`src/models/qwen35.cpp` +
//! `delta-net-base.cpp`, fetched 2026-08-21), the only implementation with
//! local same-weights parity available (`llama-server` runs these GGUFs).
//! Per layer:
//!
//! - **Full attention** (`(i+1) % full_attention_interval == 0`): the q
//!   projection emits query and a per-head **output gate** interleaved
//!   (`[q_h | gate_h]` per head); per-head q/k `RMSNorm`; `RoPE` over only the
//!   first `rope_dim` of the 256-wide heads (the metadata's `MRoPE` sections
//!   degenerate to standard `RoPE` for text-only inputs); softmax attention;
//!   then `out ⊙ sigmoid(gate)` before the output projection.
//! - **Gated `DeltaNet`** (the rest): one projection mixes q/k/v, a second
//!   emits the gate `z`; the mix runs through a depthwise causal conv
//!   (kernel `conv_kernel`, rolling state) + `SiLU`; q/k are L2-normalized
//!   per head (`x / sqrt(‖x‖² + ε)`, [`Qwen35Config::gdn_l2`]) and tiled
//!   from `n_k_heads` to
//!   `n_v_heads`; the recurrence per head with state `S ∈ R^{d_k×d_v}`:
//!   `S ← S·exp(g);  v̂ = Sᵀk;  S += k(β(v − v̂))ᵀ;  o = Sᵀ(q/√d_k)` with
//!   `β = σ(x·Wβ)` and `g = softplus(x·Wα + dt_bias)·a` (`a` holds
//!   `-exp(A_log)`, negative); the output is gated-RMS-normed
//!   (`RMS(o)·silu(z)` per head) and projected back. The gate activation is
//!   [`Qwen35Config::gdn_gate`]: qwen35 is `silu`; qwen4exp reuses these
//!   blocks through a config adapter with `sigmoid`, its one numerical
//!   difference in the `DeltaNet`.
//!
//! The NextN/MTP block some checkpoints append (`nextn_predict_layers = 1`)
//! is a draft head for speculative decoding, unused by the main forward —
//! its tensors are explicitly skipped on import.

use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d, RmsNorm, RmsNormConfig,
};
use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData, activation};
use mummu_num::{f32_from_usize, f64_from_u64, narrow};

use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo, GgufValue};
use crate::import::ImportError;
use crate::models::CausalLm;
use crate::models::qwen2::EosIds;
use crate::nn::hadamard::{DeviceConsts, HadamardRuntime, HadamardSpec};
use crate::nn::{LayerKv, SwiGluMlp, SwiGluMlpConfig, causal_mask, repeat_kv, rope_tables};
use crate::quant::QuantPolicy;

/// Re-exported so config literals outside `flex` (qwen4exp's adapter,
/// mummu-serve's test configs) can name the gate next to the config.
pub use crate::flex::gdn::{GdnGate, GdnL2};

/// Architecture hyperparameters, read from a GGUF header's `qwen35.*`
/// metadata (the family currently ships as GGUF; a safetensors `config.json`
/// path can join later).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Trunk layers only — `block_count - nextn_predict_layers`.
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Attention head width (`key_length` == `value_length`; 256 across the
    /// family — decoupled from `hidden_size / num_heads`).
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// How many leading dims of each head `RoPE` rotates (`rope.dimension_count`).
    pub rope_dim: usize,
    /// Layer `i` is full attention iff `(i+1) % interval == 0`.
    pub full_attention_interval: usize,
    /// Depthwise conv kernel length in the `DeltaNet` mix path.
    pub conv_kernel: usize,
    /// `DeltaNet` value width (`ssm.inner_size` = `n_v_heads · d_state`).
    pub d_inner: usize,
    /// Per-head key/value width (`ssm.state_size`).
    pub d_state: usize,
    /// `DeltaNet` key/query heads (`ssm.group_count`).
    pub n_k_heads: usize,
    /// `DeltaNet` value heads (`ssm.time_step_rank` — llama.cpp's reuse).
    pub n_v_heads: usize,
    /// Activation on `z` in the `DeltaNet`'s gated output `RMSNorm`. Not in
    /// the GGUF header — fixed by the architecture: [`GdnGate::Silu`] for
    /// qwen35, [`GdnGate::Sigmoid`] when qwen4exp drives these blocks.
    pub gdn_gate: GdnGate,
    /// Where `rms_norm_eps` enters the `DeltaNet`'s q/k L2 norms. Not in the
    /// header either: [`GdnL2::AddEps`] for both qwen35 and qwen4exp, the
    /// form the checkpoints were trained with.
    pub gdn_l2: GdnL2,
    pub eos_token_id: EosIds,
    /// Prism's folded Hadamard basis (`prism.hadamard.*` in the header —
    /// Ternary-Bonsai 2): which projections take a transformed input,
    /// whether the token table is stored rotated, and the per-device
    /// constants. `None` for an ordinary checkpoint. See
    /// [`crate::nn::hadamard`].
    pub hadamard: Option<std::sync::Arc<Qwen35Hadamard>>,
}

impl Qwen35Config {
    #[must_use]
    pub const fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// q/k projection width in the `DeltaNet` mix (`n_k_heads · d_state`).
    #[must_use]
    pub const fn key_dim(&self) -> usize {
        self.n_k_heads * self.d_state
    }

    /// Channels through the `DeltaNet` conv: q + k + v concatenated.
    #[must_use]
    pub const fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.d_inner
    }

    /// Hyperparameters from a GGUF header's `qwen35.*` metadata.
    ///
    /// # Errors
    ///
    /// The architecture is not `qwen35`; a required `qwen35.*` key or
    /// `tokenizer.ggml.eos_token_id` is missing; `token_embd.weight` is
    /// absent or not 2-D; `nextn_predict_layers` is not below
    /// `block_count`; `key_length` differs from `value_length`; the EOS id
    /// does not fit `u32`; a layout invariant [`Self::validate`] refuses; or
    /// a `prism.hadamard.*` contract that [`Qwen35Hadamard::from_spec`]
    /// cannot resolve against this file.
    ///
    /// # Panics
    ///
    /// Only on a 32-bit target, when a header dimension or count does not
    /// fit `usize`.
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "qwen35" {
            return Err(format!("GGUF architecture '{arch}' is not qwen35"));
        }
        let meta_usize = |key: &str| -> Result<usize, String> {
            f.get(key)
                .and_then(GgufValue::as_u64)
                .map(|v| usize::try_from(v).expect("metadata fits usize"))
                .ok_or_else(|| format!("GGUF metadata missing {key}"))
        };
        let meta_f32 = |key: &str| -> Result<f32, String> {
            f.get(key)
                .and_then(GgufValue::as_f32)
                .ok_or_else(|| format!("GGUF metadata missing {key}"))
        };
        let embd = f
            .tensor("token_embd.weight")
            .ok_or("GGUF has no token_embd.weight")?;
        // ggml dims are fastest-varying first: [hidden, vocab].
        let vocab_size = usize::try_from(*embd.dims.get(1).ok_or("token_embd is not 2-D")?)
            .expect("vocab fits usize");

        let block_count = meta_usize("qwen35.block_count")?;
        let nextn = f
            .get("qwen35.nextn_predict_layers")
            .and_then(GgufValue::as_u64)
            .map_or(0, |v| usize::try_from(v).expect("small"));
        if nextn >= block_count {
            return Err(format!(
                "nextn_predict_layers ({nextn}) must be below block_count ({block_count})"
            ));
        }
        let key_length = meta_usize("qwen35.attention.key_length")?;
        let value_length = meta_usize("qwen35.attention.value_length")?;
        if key_length != value_length {
            return Err(format!(
                "key_length ({key_length}) != value_length ({value_length}) is not implemented"
            ));
        }

        let eos = f
            .get("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_u64)
            .ok_or("GGUF metadata missing tokenizer.ggml.eos_token_id")?;

        let mut cfg = Self {
            vocab_size,
            hidden_size: meta_usize("qwen35.embedding_length")?,
            num_layers: block_count - nextn,
            num_attention_heads: meta_usize("qwen35.attention.head_count")?,
            num_key_value_heads: meta_usize("qwen35.attention.head_count_kv")?,
            head_dim: key_length,
            intermediate_size: meta_usize("qwen35.feed_forward_length")?,
            rms_norm_eps: f64::from(meta_f32("qwen35.attention.layer_norm_rms_epsilon")?),
            rope_theta: meta_f32("qwen35.rope.freq_base")?,
            rope_dim: meta_usize("qwen35.rope.dimension_count")?,
            full_attention_interval: f
                .get("qwen35.full_attention_interval")
                .and_then(GgufValue::as_u64)
                .map_or(4, |v| usize::try_from(v).expect("small")),
            conv_kernel: meta_usize("qwen35.ssm.conv_kernel")?,
            d_inner: meta_usize("qwen35.ssm.inner_size")?,
            d_state: meta_usize("qwen35.ssm.state_size")?,
            n_k_heads: meta_usize("qwen35.ssm.group_count")?,
            n_v_heads: meta_usize("qwen35.ssm.time_step_rank")?,
            // llama.cpp qwen35.cpp build_norm_gated: ggml_silu(z).
            gdn_gate: GdnGate::Silu,
            // The training-time form (FLA/transformers l2norm) and llama.cpp's
            // since PR #28068 (b10991 has it; ollama 0.34.0's b10760 does
            // not). On this family's real weights the choice is below noise —
            // Qwen3.8-27B's smallest key norm over six prompts is 1.4e-2, and
            // switching moved its logprobs by <= 6.4e-4 with top-5 and greedy
            // ids unchanged; the 2B parity legs pass or fail identically
            // under both forms against both llama.cpp builds — so the tie goes
            // to the form that stays right when keys are tiny (see GdnL2).
            gdn_l2: GdnL2::AddEps,
            eos_token_id: EosIds::One(u32::try_from(eos).map_err(|_| "EOS out of u32")?),
            hadamard: None,
        };
        cfg.validate()?;
        // The folded-basis contract, checked against THIS model's tensors:
        // an unknown fold target is a load error, never an unrotated matmul.
        let untied = f.tensor("output.weight").is_some();
        let width_of = |name: &str| {
            f.tensor(name)
                .and_then(|t| t.dims.first().copied())
                .map(|w| usize::try_from(w).expect("width fits usize"))
        };
        cfg.hadamard = HadamardSpec::from_gguf(f)?
            .map(|spec| Qwen35Hadamard::from_spec(spec, &cfg, untied, &width_of))
            .transpose()?
            .map(std::sync::Arc::new);
        Ok(cfg)
    }

    /// Layout invariants both blocks rely on. `pub(crate)` so qwen4exp's
    /// adapter config is held to the same checks.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.d_inner != self.n_v_heads * self.d_state {
            return Err(format!(
                "ssm.inner_size ({}) != n_v_heads ({}) · d_state ({}) — layout not implemented",
                self.d_inner, self.n_v_heads, self.d_state
            ));
        }
        if !self.n_v_heads.is_multiple_of(self.n_k_heads) {
            return Err(format!(
                "n_v_heads ({}) must be a multiple of n_k_heads ({})",
                self.n_v_heads, self.n_k_heads
            ));
        }
        if self.rope_dim > self.head_dim || !self.rope_dim.is_multiple_of(2) {
            return Err(format!(
                "rope_dim ({}) must be even and <= head_dim ({})",
                self.rope_dim, self.head_dim
            ));
        }
        if self.full_attention_interval == 0 || self.conv_kernel < 2 {
            return Err("degenerate full_attention_interval or conv_kernel".into());
        }
        Ok(())
    }
}

/// One projection of a layer that a Hadamard contract may fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fold {
    /// `attn_q` (attention layers).
    Q,
    /// `attn_k`.
    K,
    /// `attn_v`.
    V,
    /// `attn_output`.
    O,
    /// `attn_qkv` (`DeltaNet` layers).
    Qkv,
    /// `attn_gate`, the `DeltaNet` gate `z`.
    Z,
    /// `ssm_out`.
    Out,
    /// `ffn_gate`.
    Gate,
    /// `ffn_up`.
    Up,
    /// `ffn_down`.
    Down,
}

/// Which of one layer's projections are Hadamard-folded.
///
/// Each folded projection takes the transformed input
/// (`DeviceConsts::forward` of what it would otherwise multiply). Per
/// projection, because the contract lists weights one by one and the fork
/// transforms per weight; a layer's untouched projections keep reading the
/// plain activation. A set of [`Fold`]s, empty by default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerFolds {
    bits: u16,
}

impl LayerFolds {
    const fn bit(fold: Fold) -> u16 {
        match fold {
            Fold::Q => 1 << 0,
            Fold::K => 1 << 1,
            Fold::V => 1 << 2,
            Fold::O => 1 << 3,
            Fold::Qkv => 1 << 4,
            Fold::Z => 1 << 5,
            Fold::Out => 1 << 6,
            Fold::Gate => 1 << 7,
            Fold::Up => 1 << 8,
            Fold::Down => 1 << 9,
        }
    }

    /// Is `fold`'s projection folded?
    #[must_use]
    pub const fn has(self, fold: Fold) -> bool {
        self.bits & Self::bit(fold) != 0
    }

    /// Mark `fold`'s projection as folded.
    pub const fn insert(&mut self, fold: Fold) {
        self.bits |= Self::bit(fold);
    }

    /// Does any projection reading the block INPUT take the transform?
    const fn any_input(self) -> bool {
        self.has(Fold::Q)
            || self.has(Fold::K)
            || self.has(Fold::V)
            || self.has(Fold::Qkv)
            || self.has(Fold::Z)
    }
}

/// The folded-basis contract resolved onto this architecture's forward:
/// per-layer fold flags, the head, the token table, and the runtime that
/// owns the per-device constants.
#[derive(Debug)]
pub struct Qwen35Hadamard {
    pub runtime: HadamardRuntime,
    /// One entry per trunk layer.
    pub layers: Vec<LayerFolds>,
    /// The head's input is transformed: a folded `output.weight`, or a
    /// tied head reading the rotated table (`logits = E'·R h = E·h`).
    pub head: bool,
    /// `token_embd.weight` holds rotated rows: inverse after the gather.
    pub embed_inverse: bool,
}

impl Qwen35Hadamard {
    /// Resolve `spec` against `cfg`. Every folded name must be a projection
    /// this forward transforms the input of (a load error otherwise — the
    /// fork refuses the same way), its input width must cut into whole
    /// blocks and, in explicit sign mode, carry a sign vector; the only
    /// rotated lookup table this forward restores is the token embedding.
    /// `width_of` gives a tensor's input width (ggml's first dim).
    ///
    /// # Errors
    ///
    /// A folded weight the file does not carry, whose input width the
    /// block does not divide, or that lacks its sign vector in explicit
    /// sign mode ([`HadamardSpec::signs_for`]); a folded name that is not a
    /// projection this forward transforms the input of (or a malformed
    /// `blk.N.` name); or an inverse-listed tensor other than
    /// `token_embd.weight`.
    pub fn from_spec(
        spec: HadamardSpec,
        cfg: &Qwen35Config,
        untied: bool,
        width_of: &dyn Fn(&str) -> Option<usize>,
    ) -> Result<Self, String> {
        let check_width = |name: &str, width: usize| -> Result<(), String> {
            if !width.is_multiple_of(spec.block) {
                return Err(format!(
                    "prism.hadamard block {} does not divide the input width {width} of '{name}'",
                    spec.block
                ));
            }
            spec.signs_for(width).map(|_| ())
        };
        let not_verified = |name: &str| {
            format!("prism.hadamard folds '{name}', which is not on a path this forward transforms")
        };
        let mut layers = vec![LayerFolds::default(); cfg.num_layers];
        let mut head = false;
        for name in &spec.weights {
            let width = width_of(name).ok_or_else(|| {
                format!("prism.hadamard folds '{name}', which the file does not carry")
            })?;
            check_width(name, width)?;
            if name == "output.weight" {
                head = true;
                continue;
            }
            let Some((layer, field)) = name.strip_prefix("blk.").and_then(|r| r.split_once('.'))
            else {
                return Err(not_verified(name));
            };
            let layer: usize = layer
                .parse()
                .map_err(|_| format!("bad layer index in '{name}'"))?;
            if layer >= cfg.num_layers {
                continue; // the NextN draft block: never run here
            }
            let attn = cfg.is_attention(layer);
            let fold = match (field, attn) {
                ("attn_q.weight", true) => Fold::Q,
                ("attn_k.weight", true) => Fold::K,
                ("attn_v.weight", true) => Fold::V,
                ("attn_output.weight", true) => Fold::O,
                ("attn_qkv.weight", false) => Fold::Qkv,
                ("attn_gate.weight", false) => Fold::Z,
                ("ssm_out.weight", false) => Fold::Out,
                ("ffn_gate.weight", _) => Fold::Gate,
                ("ffn_up.weight", _) => Fold::Up,
                ("ffn_down.weight", _) => Fold::Down,
                _ => return Err(not_verified(name)),
            };
            layers[layer].insert(fold);
        }
        let mut embed_inverse = false;
        for name in &spec.inverses {
            if name != "token_embd.weight" {
                return Err(format!(
                    "prism.hadamard names '{name}' as a rotated lookup table; token_embd.weight is the only one this forward restores"
                ));
            }
            check_width(name, cfg.hidden_size)?;
            embed_inverse = true;
        }
        // A tied head multiplies by the rotated table itself, so its input
        // takes the forward transform: E'·(R h) = (R E)·(R h) = E·h.
        if !untied && embed_inverse {
            head = true;
        }
        Ok(Self {
            runtime: HadamardRuntime::new(spec),
            layers,
            head,
            embed_inverse,
        })
    }
}

/// One layer's fold flags with the constants on that layer's device — what
/// the block forwards take.
pub struct LayerHadamard {
    pub consts: std::sync::Arc<DeviceConsts>,
    pub folds: LayerFolds,
}

impl LayerHadamard {
    /// The block input `x`, transformed once when any of the projections
    /// reading it is folded; `pick(on)` then hands each projection the
    /// version it was folded for.
    fn input_pair(&self, x: &Tensor<3>) -> Option<Tensor<3>> {
        self.folds
            .any_input()
            .then(|| self.consts.forward(x.clone()))
    }
}

fn pick(on: bool, transformed: Option<&Tensor<3>>, plain: &Tensor<3>) -> Tensor<3> {
    match transformed {
        Some(t) if on => t.clone(),
        _ => plain.clone(),
    }
}

/// `Linear::forward` without touching the weight's shape: burn's Linear
/// unsqueezes the weight to the input rank, and a reshape on **packed**
/// quantized storage is broken in burn 0.21 (physical element count differs
/// from the logical one — measured with Q4 on flex AND wgpu, 2026-08-21).
/// Flatten the FLOAT input instead; the weight goes into the matmul as-is.
/// All qwen35 projections are bias-free.
pub(crate) fn qlinear(lin: &Linear, x: Tensor<3>) -> Tensor<3> {
    debug_assert!(lin.bias.is_none(), "qwen35 projections are bias-free");
    let [b, t, d_in] = x.dims();
    let weight = lin.weight.val(); // [in, out]
    let d_out = weight.dims()[1];
    let x2 = crate::nn::refarith::linear_input2(x.reshape([b * t, d_in]), d_in, d_out);
    // Decode-shape quantized weights take the packed GEMV (reads the
    // stored bytes directly; on flex that is the i8 slab at 1.125 B/elem
    // against the 4 B/elem an f32 slab moves). Anything else — prefill,
    // float weights — keeps the plain matmul.
    let out = crate::nn::try_q4s_gemv(&x2, &weight).unwrap_or_else(|| x2.matmul(weight));
    out.reshape([b, t, d_out])
}

/// The 2-D twin of [`qlinear`] (the lm-head path).
pub(crate) fn qlinear2(lin: &Linear, x: Tensor<2>) -> Tensor<2> {
    debug_assert!(lin.bias.is_none(), "qwen35 projections are bias-free");
    let weight = lin.weight.val();
    let [w_in, w_out] = weight.dims();
    let x = crate::nn::refarith::linear_input2(x, w_in, w_out);
    crate::nn::try_q4s_gemv(&x, &weight).unwrap_or_else(|| x.matmul(weight))
}

/// A bias-free `Linear` — every qwen35 projection is one ([`qlinear`]
/// asserts it).
fn bias_free_linear(inp: usize, out: usize, device: &Device) -> Linear {
    LinearConfig::new(inp, out).with_bias(false).init(device)
}

/// An `RMSNorm` over `dim` at the model's epsilon.
fn rms_norm(cfg: &Qwen35Config, dim: usize, device: &Device) -> RmsNorm {
    RmsNormConfig::new(dim)
        .with_epsilon(cfg.rms_norm_eps)
        .init(device)
}

/// Gated full attention (see the module docs). Field names are this port's
/// own (the family has no HF safetensors convention to mirror yet).
#[derive(Module, Debug)]
pub struct GatedAttention {
    /// Emits `[q_h | gate_h]` interleaved per head — `2·num_heads·head_dim` wide.
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    /// Per-head `RMSNorm` over `head_dim`.
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
}

impl GatedAttention {
    /// A block shaped by `cfg`, with placeholder weights until a loader
    /// assigns the real ones. Shared by `build` and sibling ports that
    /// drive this block with their own loader, so the shapes cannot drift.
    pub(crate) fn init(cfg: &Qwen35Config, device: &Device) -> Self {
        Self {
            q_proj: bias_free_linear(
                cfg.hidden_size,
                2 * cfg.num_attention_heads * cfg.head_dim,
                device,
            ),
            k_proj: bias_free_linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                device,
            ),
            v_proj: bias_free_linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                device,
            ),
            o_proj: bias_free_linear(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                device,
            ),
            q_norm: rms_norm(cfg, cfg.head_dim, device),
            k_norm: rms_norm(cfg, cfg.head_dim, device),
        }
    }

    /// One gated-attention block over `x` `[b, t, hidden]`. `pos` carries
    /// the [`rope_tables`] over `rope_dim` for positions `past..past+t` and
    /// the causal mask when `t > 1`; `kv` grows by `t`.
    pub(crate) fn forward(
        &self,
        x: Tensor<3>,
        cfg: &Qwen35Config,
        pos: &PositionTables<'_>,
        kv: &mut LayerKv,
        had: Option<&LayerHadamard>,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        // Folded projections read the transformed input (once per block).
        let xr = had.and_then(|layer_had| layer_had.input_pair(&x));
        let folds = had.map_or_else(LayerFolds::default, |layer_had| layer_had.folds);
        let (xq, xk, xv) = (
            pick(folds.has(Fold::Q), xr.as_ref(), &x),
            pick(folds.has(Fold::K), xr.as_ref(), &x),
            pick(folds.has(Fold::V), xr.as_ref(), &x),
        );
        drop(xr);
        drop(x);

        // Split the joint projection into q and gate: per head the layout is
        // [q (hd) | gate (hd)], so a [b, t, nh, 2, hd] view separates them.
        let prof_qkv = crate::prof::scope("fa.qkv");
        let qg = qlinear(&self.q_proj, xq).reshape([b, t, nh, 2, hd]);
        let query = qg.clone().narrow(3, 0, 1).reshape([b, t, nh, hd]);
        let gate = qg.narrow(3, 1, 1).reshape([b, t, nh, hd]);

        let query = self.q_norm.forward(query).swap_dims(1, 2); // [b, nh, t, hd]
        let k_new = qlinear(&self.k_proj, xk).reshape([b, t, nkv, hd]);
        let k_new = self.k_norm.forward(k_new).swap_dims(1, 2);
        let v_new = qlinear(&self.v_proj, xv)
            .reshape([b, t, nkv, hd])
            .swap_dims(1, 2);

        drop(prof_qkv);
        let prof_rope = crate::prof::scope("fa.rope");
        // Partial RoPE: rotate the first rope_dim dims, pass the rest through.
        let rope = |heads: Tensor<4>| -> Tensor<4> {
            let rot = heads.clone().narrow(3, 0, cfg.rope_dim);
            let rest = heads.narrow(3, cfg.rope_dim, hd - cfg.rope_dim);
            let rot = crate::nn::apply_rope(rot, pos.cos, pos.sin);
            Tensor::cat(vec![rot, rest], 3)
        };
        let query = rope(query);
        let k_new = rope(k_new);
        drop(prof_rope);
        let prof_cache = crate::prof::scope("fa.kv");

        // Storage dtype (f16 KV when enabled) is the cache helper's call;
        // scores upcast to the f32 island below and the value matmul
        // upcasts to ambient, so precision here is storage-only.
        let ambient = query.dtype();
        let (k_all, v_all) = crate::nn::kv_append(kv, k_new, v_new);

        let group = nh / nkv;
        let scale = 1.0 / f32_from_usize(hd).sqrt();
        // Diagnostic only (`nn::refarith`, off by default): ggml's CPU flash
        // attention arithmetic over an f16 cache, host-side.
        let ctx = if crate::nn::refarith::enabled() {
            drop(prof_cache);
            crate::nn::refarith::flash_attn_f16(query, k_all, v_all, scale)
        } else {
            let keys = repeat_kv(k_all, group);
            let values = repeat_kv(v_all, group).cast(ambient);
            drop(prof_cache);
            let prof_scores = crate::prof::scope("fa.scores");

            // f32 island for the scores — the same overflow guard as GqaAttention.
            let mut scores = query
                .cast(DType::F32)
                .matmul(keys.cast(DType::F32).swap_dims(2, 3))
                .mul_scalar(scale);
            if let Some(mask) = pos.mask {
                scores = scores.add(mask.clone().cast(DType::F32));
            }
            let probs = activation::softmax(scores, 3).cast(ambient);
            let ctx = probs.matmul(values); // [b, nh, t, hd]
            drop(prof_scores);
            ctx
        };
        let _s = crate::prof::scope("fa.out");

        // Per-head output gate: out ⊙ sigmoid(gate).
        let gated = ctx
            .swap_dims(1, 2) // [b, t, nh, hd]
            .mul(activation::sigmoid(gate))
            .reshape([b, t, nh * hd]);
        let gated = match had {
            Some(layer_had) if layer_had.folds.has(Fold::O) => layer_had.consts.forward(gated),
            _ => gated,
        };
        qlinear(&self.o_proj, gated)
    }
}

/// The position-dependent inputs of one attention call: the [`rope_tables`]
/// (`cos`/`sin`, `[1, 1, t, rope_dim]`) for positions `past..past+t`, and
/// the causal mask when `t > 1`.
#[derive(Clone, Copy)]
pub struct PositionTables<'a> {
    pub cos: &'a Tensor<4>,
    pub sin: &'a Tensor<4>,
    pub mask: Option<&'a Tensor<4>>,
}

/// Gated `DeltaNet` linear attention (see the module docs).
#[derive(Module, Debug)]
pub struct GatedDeltaNet {
    /// Mixes q/k/v: `hidden → 2·key_dim + d_inner`.
    pub qkv_proj: Linear,
    /// The gate `z`: `hidden → d_inner`.
    pub z_proj: Linear,
    /// Per-value-head β logits: `hidden → n_v_heads`.
    pub beta_proj: Linear,
    /// Per-value-head decay logits: `hidden → n_v_heads`.
    pub alpha_proj: Linear,
    /// Decay bias added to the α logits before softplus.
    pub dt_bias: Param<Tensor<1>>,
    /// `-exp(A_log)` — negative per-head decay magnitudes.
    pub a: Param<Tensor<1>>,
    /// Depthwise causal conv over the q/k/v mix, kernel `conv_kernel`.
    pub conv1d: Conv1d,
    /// Gated output `RMSNorm` over `d_state` (per value head).
    pub norm: RmsNorm,
    pub out_proj: Linear,
}

/// `DeltaNet` decode state: the rolling conv window and the recurrent memory.
///
/// The state lives in exactly one of two worlds at a time: the tensor
/// fields (prefill and the tensor decode path) or the host fields (the
/// fused decode path, SPEC P3) — each path converts the other's fields on
/// entry and leaves its own. `middle` caches the layer's extracted small
/// weights for the fused kernel; per-request rebuild costs microseconds
/// and avoids any global keyed on tensor storage.
pub struct DeltaState {
    /// Last `conv_kernel - 1` mix columns, `[b, conv_dim, k-1]`.
    pub conv: Option<Tensor<3>>,
    /// Per-head associative memory `[b, n_v_heads, d_state, d_state]`.
    pub state: Option<Tensor<4>>,
    /// Host twin of `conv` for the fused decode step (batch 1),
    /// `[conv_dim * (k-1)]` channel-major, oldest first.
    pub host_conv: Option<Vec<f32>>,
    /// Host twin of `state`, `[n_v_heads * d_state * d_state]` head-major.
    pub host_state: Option<Vec<f32>>,
    /// The fused kernel's per-layer constants, extracted at first use.
    pub middle: Option<std::sync::Arc<crate::flex::gdn::GdnMiddle>>,
}

impl DeltaState {
    /// An empty state (fresh generation).
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            conv: None,
            state: None,
            host_conv: None,
            host_state: None,
            middle: None,
        }
    }
}

impl GatedDeltaNet {
    /// A block shaped by `cfg`, with placeholder weights until a loader
    /// assigns the real ones. Build `DeltaNets` through here rather than a
    /// struct literal: the conv carries `conv_kernel - 1` explicit padding
    /// on both sides, and [`Self::forward`]'s fresh-prefill branch reads
    /// outputs `0..t` of exactly that padded conv as the causal alignment —
    /// any other padding computes a shifted conv with no error.
    pub(crate) fn init(cfg: &Qwen35Config, device: &Device) -> Self {
        Self {
            qkv_proj: bias_free_linear(cfg.hidden_size, cfg.conv_dim(), device),
            z_proj: bias_free_linear(cfg.hidden_size, cfg.d_inner, device),
            beta_proj: bias_free_linear(cfg.hidden_size, cfg.n_v_heads, device),
            alpha_proj: bias_free_linear(cfg.hidden_size, cfg.n_v_heads, device),
            dt_bias: Param::from_tensor(Tensor::zeros([cfg.n_v_heads], device)),
            a: Param::from_tensor(Tensor::zeros([cfg.n_v_heads], device)),
            conv1d: Conv1dConfig::new(cfg.conv_dim(), cfg.conv_dim(), cfg.conv_kernel)
                .with_groups(cfg.conv_dim())
                .with_padding(PaddingConfig1d::Explicit(
                    cfg.conv_kernel - 1,
                    cfg.conv_kernel - 1,
                ))
                .with_bias(false)
                .init(device),
            norm: rms_norm(cfg, cfg.d_state, device),
            out_proj: bias_free_linear(cfg.d_inner, cfg.hidden_size, device),
        }
    }

    /// One Gated `DeltaNet` block over `x` `[b, t, hidden]`, advancing
    /// `cache` (conv window + recurrent state) by `t` tokens.
    pub(crate) fn forward(
        &self,
        x: Tensor<3>,
        cfg: &Qwen35Config,
        cache: &mut DeltaState,
        had: Option<&LayerHadamard>,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (hv, ds) = (cfg.n_v_heads, cfg.d_state);
        let device = x.device();

        // The fused host decode step (SPEC P3): one function replaces the
        // ~9 small-tensor ops between the projections, with the state kept
        // as plain host memory across tokens. Flex only, batch 1, t == 1;
        // MUMMU_FUSED_GDN=0 (or the force switch) restores the path below.
        if t == 1 && b == 1 && crate::flex::gdn::enabled() && crate::backend::is_flex(&device) {
            return self.forward_fused_decode(x, cfg, cache, &device, had);
        }
        materialize_host_state(cache, cfg, &device);

        let prof_proj = crate::prof::scope("delta.proj");
        let DeltaProjections {
            mixed,
            gate: gate_z,
            beta,
            decay,
        } = self.project(x, cfg, had);
        drop(prof_proj);
        // Depthwise causal conv over the sequence, rolling the decode state
        // exactly like nn::ShortConv (algebraic equivalence proven there).
        let prof_conv = crate::prof::scope("delta.conv");
        let conv_out = self.causal_conv(mixed.swap_dims(1, 2), cache, cfg.conv_kernel, &device);
        let conv_out = activation::silu(conv_out.swap_dims(1, 2)); // [b, t, conv_dim]
        trace_tensor("gdn.conv_silu", &conv_out);
        if gdn_l2_probe::enabled() {
            gdn_l2_probe::record(&conv_out, cfg);
        }
        drop(prof_conv);
        let prof_split = crate::prof::scope("delta.split");
        let (query, keys, values) = split_qkv(conv_out, cfg);
        drop(prof_split);
        let prof_recur = crate::prof::scope("delta.recur");

        let scale = 1.0 / f32_from_usize(ds).sqrt();
        let s0 = cache
            .state
            .take()
            .unwrap_or_else(|| Tensor::<4>::zeros([b, hv, ds, ds], &device));
        // Prefill spans beyond a few tokens take the chunkwise-parallel
        // form — algebraically exact, an evaluation-order change and not an
        // approximation (the derivation lives on `gdn_recurrence_chunked`),
        // at roughly C-fold fewer kernel launches. The old threshold was
        // one full chunk (t > 64), which left every 2..=64-token prompt on
        // the sequential loop at ~9 launches per token — the exact cliff
        // SPEC P5.2 names. A partial chunk is handled exactly (`chunk.min`
        // below), so the crossover is only launch arithmetic: one chunk
        // costs ~20 chunk-level ops + 2-3 matmuls per doubling stage
        // against 9·t sequential — even at t = 8 the chunk wins. The
        // sequential ceiling is measured-tunable (`MUMMU_GDN_SEQ_MAX`,
        // default 4); decode (t == 1) stays sequential by construction.
        let inputs = RecurrenceInputs {
            q: &query,
            k: &keys,
            v: &values,
            g: &decay,
            beta: &beta,
        };
        let (out_heads, s_new) = match gdn_chunk() {
            Some(chunk) if t > gdn_seq_max() => gdn_recurrence_chunked(&inputs, s0, scale, chunk),
            _ => gdn_recurrence_sequential(&inputs, s0, scale),
        };
        cache.state = Some(s_new); // [b, hv, ds, ds]; out_heads is [b, hv, t, ds]
        drop(prof_recur);
        let _s = crate::prof::scope("delta.out");
        self.gated_output(out_heads, gate_z, cfg, had)
    }

    /// The four projections of the block input `x` `[b, t, hidden]`.
    fn project(
        &self,
        x: Tensor<3>,
        cfg: &Qwen35Config,
        had: Option<&LayerHadamard>,
    ) -> DeltaProjections {
        let hv = cfg.n_v_heads;
        // The mix and the gate may be folded; β/α stay in the plain basis
        // (the fork keeps the recurrent-state path at full precision,
        // unrotated).
        let xr = had.and_then(|layer_had| layer_had.input_pair(&x));
        let folds = had.map_or_else(LayerFolds::default, |layer_had| layer_had.folds);
        if let Some(rot) = &xr {
            trace_tensor("gdn.x_rot", rot);
        }
        trace_tensor("gdn.x", &x);
        let mixed = qlinear(&self.qkv_proj, pick(folds.has(Fold::Qkv), xr.as_ref(), &x)); // [b, t, conv_dim]
        let gate = qlinear(&self.z_proj, pick(folds.has(Fold::Z), xr.as_ref(), &x)); // [b, t, d_inner]
        drop(xr);
        trace_tensor("gdn.qkv", &mixed);
        trace_tensor("gdn.z", &gate);
        let beta = activation::sigmoid(qlinear(&self.beta_proj, x.clone())); // [b, t, hv]
        // g = softplus(α + dt_bias) · a, with a = -exp(A_log) < 0.
        let alpha = qlinear(&self.alpha_proj, x).add(self.dt_bias.val().reshape([1, 1, hv]));
        let decay = activation::softplus(alpha, 1.0).mul(self.a.val().reshape([1, 1, hv]));
        DeltaProjections {
            mixed,
            gate,
            beta,
            decay,
        }
    }

    /// The depthwise causal conv over `[cached window | mix_cm]`, channel-major
    /// `[b, conv_dim, t]` in and out (before the `SiLU`), rolling `cache.conv`
    /// forward by `t` columns.
    fn causal_conv(
        &self,
        mix_cm: Tensor<3>,
        cache: &mut DeltaState,
        kk: usize,
        device: &Device,
    ) -> Tensor<3> {
        let [b, conv_dim, t] = mix_cm.dims();
        let conv_out = if t > 1 {
            // Continuation (a later prefill chunk, or a prompt after
            // decode steps): the first kk-1 positions' windows reach
            // into the PREVIOUS span, which the rolling cache holds.
            // Running the conv over [cached | new] and taking the
            // outputs aligned to the new span reproduces the
            // uninterrupted conv exactly. The old code fell into the
            // fresh-start branch here and convolved those positions
            // against zero history — wrong at every chunk boundary
            // (found by the forward_advance equivalence test; the
            // chunked-prefill exactness claim held only for prompts
            // within one chunk).
            cache.conv.as_ref().map_or_else(
                || self.conv1d.forward(mix_cm.clone()).narrow(2, 0, t),
                |prev| {
                    let ext = Tensor::cat(vec![prev.clone(), mix_cm.clone()], 2);
                    self.conv1d.forward(ext).narrow(2, kk - 1, t)
                },
            )
        } else {
            let window = cache.conv.as_ref().map_or_else(
                || {
                    let pad = Tensor::<3>::zeros([b, conv_dim, kk - 1], device);
                    Tensor::cat(vec![pad, mix_cm.clone()], 2)
                },
                |prev| Tensor::cat(vec![prev.clone(), mix_cm.clone()], 2),
            );
            let taps = self.conv1d.weight.val().reshape([1, conv_dim, kk]);
            window.mul(taps).sum_dim(2)
        };
        cache.conv = Some({
            let combined = match cache.conv.take() {
                Some(prev) => Tensor::cat(vec![prev, mix_cm], 2),
                None => mix_cm,
            };
            let len = combined.dims()[2];
            if len >= kk - 1 {
                combined.narrow(2, len - (kk - 1), kk - 1)
            } else {
                let pad = Tensor::<3>::zeros([b, conv_dim, (kk - 1) - len], device);
                Tensor::cat(vec![pad, combined], 2)
            }
        });
        conv_out
    }

    /// The gated `RMSNorm` per value head over the recurrence output
    /// `out_heads` `[b, hv, t, ds]`, the family's gate on `z`, and the
    /// out-projection.
    fn gated_output(
        &self,
        out_heads: Tensor<4>,
        gate_z: Tensor<3>,
        cfg: &Qwen35Config,
        had: Option<&LayerHadamard>,
    ) -> Tensor<3> {
        let [b, hv, t, ds] = out_heads.dims();
        let hk = cfg.n_k_heads;
        // Gated RMSNorm per value head, then flatten and project out. The
        // gate activation is the family's (silu for qwen35, sigmoid for
        // qwen4exp); the fused host step applies the same choice.
        let normed = self.norm.forward(out_heads.swap_dims(1, 2)); // [b, t, hv, ds]
        let gate_z = gate_z.reshape([b, t, hv, ds]);
        let gate = match cfg.gdn_gate {
            GdnGate::Silu => activation::silu(gate_z),
            GdnGate::Sigmoid => activation::sigmoid(gate_z),
        };
        let gated = normed.mul(gate).reshape([b, t, cfg.d_inner]);
        trace_tensor("gdn.gated", &gated);
        // A folded out-projection was rotated over HF's grouped value
        // heads; this port's tiled order is permuted to match first.
        let gated = match had {
            Some(layer_had) if layer_had.folds.has(Fold::Out) => {
                let grouped = if layer_had.consts.spec().gdn_v_grouped {
                    crate::nn::hadamard::tiled_to_grouped(gated, hk, hv, ds)
                } else {
                    gated
                };
                trace_tensor("gdn.gated_grouped", &grouped);
                let rot = layer_had.consts.forward(grouped);
                trace_tensor("gdn.gated_rot", &rot);
                rot
            }
            _ => gated,
        };
        let out = qlinear(&self.out_proj, gated);
        trace_tensor("gdn.out", &out);
        out
    }

    /// The fused decode step (SPEC P3): projections stay tensor ops (they
    /// are single packed-GEMV dispatches on the twin path), everything
    /// between them runs as one host function over the state kept in plain
    /// memory. See `flex::gdn` for the pass structure and the algebra.
    fn forward_fused_decode(
        &self,
        x: Tensor<3>,
        cfg: &Qwen35Config,
        cache: &mut DeltaState,
        device: &Device,
        had: Option<&LayerHadamard>,
    ) -> Tensor<3> {
        let prof_proj = crate::prof::scope("delta.proj");
        // The transform on the host (O(n log n) butterflies): this path is
        // flex-only by construction, and the tensor matmul against the
        // block matrix would cost more than the step it decorates.
        let folds = had.map_or_else(LayerFolds::default, |layer_had| layer_had.folds);
        let xr = had.filter(|_| folds.any_input()).map(|layer_had| {
            let mut vals = x
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("flex activations are f32");
            layer_had
                .consts
                .spec()
                .forward_host(&mut vals)
                .expect("folded widths were checked at load");
            Tensor::<3>::from_data(TensorData::new(vals, [1, 1, cfg.hidden_size]), device)
        });
        let mixed_t = qlinear(&self.qkv_proj, pick(folds.has(Fold::Qkv), xr.as_ref(), &x)); // [1, 1, conv_dim]
        let z_t = qlinear(&self.z_proj, pick(folds.has(Fold::Z), xr.as_ref(), &x)); // [1, 1, d_inner]
        drop(xr);
        let beta_t = qlinear(&self.beta_proj, x.clone()); // [1, 1, hv]
        let alpha_t = qlinear(&self.alpha_proj, x); // [1, 1, hv]
        drop(prof_proj);

        let prof_fused = crate::prof::scope("delta.fused");
        let middle = if let Some(mid) = &cache.middle {
            std::sync::Arc::clone(mid)
        } else {
            let mid = std::sync::Arc::new(self.fused_middle(cfg));
            cache.middle = Some(std::sync::Arc::clone(&mid));
            mid
        };
        let host = |tensor: Tensor<3>| -> Vec<f32> {
            tensor
                .into_data()
                .try_to_vec::<f32>()
                .expect("flex activations are f32")
        };
        let (mixed, gate_z, beta, alpha) = (host(mixed_t), host(z_t), host(beta_t), host(alpha_t));

        // The state's host twins, converted from tensors on first use
        // (prefill ran the tensor path) or zero-initialized (no prefix).
        if cache.host_conv.is_none() {
            cache.host_conv = Some(cache.conv.take().map_or_else(
                || vec![0f32; middle.ring_len()],
                |tc| {
                    tc.into_data()
                        .try_to_vec::<f32>()
                        .expect("conv window is f32")
                },
            ));
        }
        if cache.host_state.is_none() {
            cache.host_state = Some(cache.state.take().map_or_else(
                || vec![0f32; middle.state_len()],
                |ts| ts.into_data().try_to_vec::<f32>().expect("state is f32"),
            ));
        }

        let mut gated = vec![0f32; cfg.d_inner];
        crate::flex::gdn::gdn_step(
            &middle,
            crate::flex::gdn::GdnInputs {
                mixed: &mixed,
                z: &gate_z,
                beta_logits: &beta,
                alpha_logits: &alpha,
            },
            cache.host_conv.as_mut().expect("just filled"),
            cache.host_state.as_mut().expect("just filled"),
            &mut gated,
        );
        drop(prof_fused);

        let _s_out = crate::prof::scope("delta.out");
        let gated = match had {
            Some(layer_had) if layer_had.folds.has(Fold::Out) => {
                let mut grouped = if layer_had.consts.spec().gdn_v_grouped {
                    crate::nn::hadamard::tiled_to_grouped_host(
                        &gated,
                        cfg.n_k_heads,
                        cfg.n_v_heads,
                        cfg.d_state,
                    )
                } else {
                    gated
                };
                layer_had
                    .consts
                    .spec()
                    .forward_host(&mut grouped)
                    .expect("folded widths were checked at load");
                grouped
            }
            _ => gated,
        };
        let gated_t = Tensor::<3>::from_data(TensorData::new(gated, [1, 1, cfg.d_inner]), device);
        qlinear(&self.out_proj, gated_t)
    }

    /// Extract the fused kernel's per-layer constants (conv taps, gates'
    /// bias/decay, the norm gain). A few hundred KB, once per request per
    /// layer — cached on the [`DeltaState`].
    fn fused_middle(&self, cfg: &Qwen35Config) -> crate::flex::gdn::GdnMiddle {
        let host = |t: Tensor<1>| -> Vec<f32> {
            t.into_data().try_to_vec::<f32>().expect("params are f32")
        };
        // Conv1d weight is [conv_dim, 1, kk] row-major: channel-major with
        // the taps fastest — exactly the [c][tap] layout the FIR reads.
        let conv_w = self
            .conv1d
            .weight
            .val()
            .reshape([cfg.conv_dim() * cfg.conv_kernel]);
        crate::flex::gdn::GdnMiddle {
            hk: cfg.n_k_heads,
            hv: cfg.n_v_heads,
            ds: cfg.d_state,
            kk: cfg.conv_kernel,
            conv_dim: cfg.conv_dim(),
            key_dim: cfg.key_dim(),
            d_inner: cfg.d_inner,
            l2_eps: narrow(cfg.rms_norm_eps),
            l2: cfg.gdn_l2,
            norm_eps: narrow(cfg.rms_norm_eps),
            scale: 1.0 / f32_from_usize(cfg.d_state).sqrt(),
            conv_w: host(conv_w),
            dt_bias: host(self.dt_bias.val()),
            a: host(self.a.val()),
            gamma: host(self.norm.gamma.val()),
            gate: cfg.gdn_gate,
        }
    }
}

/// The `DeltaNet` block's four projections of one input span.
struct DeltaProjections {
    /// The q/k/v mix, `[b, t, conv_dim]`.
    mixed: Tensor<3>,
    /// The gate `z`, `[b, t, d_inner]`.
    gate: Tensor<3>,
    /// `β = σ(x·Wβ)`, `[b, t, hv]`.
    beta: Tensor<3>,
    /// The decay logits `g = softplus(x·Wα + dt_bias)·a`, `[b, t, hv]`.
    decay: Tensor<3>,
}

/// Entering the tensor path with host-resident state (a prefill after
/// fused decode steps — the multi-turn shape): materialize the tensors the
/// tensor path reads, and drop the host twins.
fn materialize_host_state(cache: &mut DeltaState, cfg: &Qwen35Config, device: &Device) {
    let (hv, ds) = (cfg.n_v_heads, cfg.d_state);
    let conv_dim = cfg.conv_dim();
    let kk = cfg.conv_kernel;
    if let Some(hc) = cache.host_conv.take() {
        debug_assert_eq!(hc.len(), conv_dim * (kk - 1));
        cache.conv =
            Some(
                Tensor::<1>::from_data(TensorData::new(hc, [conv_dim * (kk - 1)]), device)
                    .reshape([1, conv_dim, kk - 1]),
            );
    }
    if let Some(hs) = cache.host_state.take() {
        debug_assert_eq!(hs.len(), hv * ds * ds);
        cache.state = Some(
            Tensor::<1>::from_data(TensorData::new(hs, [hv * ds * ds]), device)
                .reshape([1, hv, ds, ds]),
        );
    }
}

/// Split the conv output `[b, t, conv_dim]` into `(q, k, v)`, each
/// `[b, hv, t, ds]`: q/k L2-normalized per head in the family's form
/// (`x / max(‖x‖, ε)` or `x / sqrt(‖x‖² + ε)`) and tiled from `n_k_heads`
/// to `n_v_heads` (llama.cpp's `ggml_repeat`: value head `h_v` reads
/// key head `h_v % n_k_heads`).
fn split_qkv(conv_out: Tensor<3>, cfg: &Qwen35Config) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let [b, t, _] = conv_out.dims();
    let (hk, hv, ds) = (cfg.n_k_heads, cfg.n_v_heads, cfg.d_state);
    let key_dim = cfg.key_dim();
    let eps = narrow(cfg.rms_norm_eps);
    let l2_form = cfg.gdn_l2;
    let l2 = |heads: Tensor<4>| -> Tensor<4> {
        let sum_sq = heads.clone().powi_scalar(2).sum_dim(3);
        let norm = match l2_form {
            GdnL2::ClampNorm => sum_sq.sqrt().clamp_min(eps),
            GdnL2::AddEps => sum_sq.add_scalar(eps).sqrt(),
        };
        heads.div(norm)
    };
    let query = l2(conv_out
        .clone()
        .narrow(2, 0, key_dim)
        .reshape([b, t, hk, ds]));
    let keys = l2(conv_out
        .clone()
        .narrow(2, key_dim, key_dim)
        .reshape([b, t, hk, ds]));
    let values = conv_out
        .narrow(2, 2 * key_dim, cfg.d_inner)
        .reshape([b, t, hv, ds]);

    // Tile k-heads across the value heads (llama.cpp's ggml_repeat:
    // head h_v reads k-head h_v % n_k_heads).
    let tile = hv / hk;
    let expand = |heads: Tensor<4>| -> Tensor<4> {
        if tile == 1 {
            heads
        } else {
            // [b, t, hk, ds] → [b, t, tile·hk, ds] tiling whole blocks.
            heads.repeat_dim(2, tile)
        }
    };
    let query = expand(query).swap_dims(1, 2); // [b, hv, t, ds]
    let keys = expand(keys).swap_dims(1, 2);
    let values = values.swap_dims(1, 2);
    (query, keys, values)
}

/// Chunk length for the chunkwise-parallel `DeltaNet` prefill, `None` when
/// that path is disabled. One env read per process (mirrors
/// [`lookahead_verify`]): `MUMMU_GDN_CHUNK` unset → the default 64; an
/// integer overrides it; `0` or `off` disables chunking entirely (every
/// prefill then runs the sequential reference); anything unrecognized
/// falls back to the default.
fn gdn_chunk() -> Option<usize> {
    static CHUNK: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("MUMMU_GDN_CHUNK").map_or(Some(64), |v| {
            let v = v.trim();
            if v.eq_ignore_ascii_case("off") {
                None
            } else {
                match v.parse::<usize>() {
                    Ok(0) => None,
                    Ok(c) => Some(c),
                    Err(_) => Some(64),
                }
            }
        })
    })
}

/// Diagnostic tap on the per-head ‖q‖ and ‖k‖ entering the `DeltaNet`'s L2
/// normalization.
///
/// These are the numbers that decide whether the [`GdnL2`] form matters on
/// a checkpoint: the two forms differ by `1 − ‖x‖/sqrt(‖x‖²+ε)` relative,
/// which is 29% at ‖x‖ = 1e-3 and 5e-5 at 0.1 for ε = 1e-6. Off by default;
/// while on, every tensor-path `DeltaNet` forward reads its conv output
/// back to the host. The fused host decode step is not tapped, so probe
/// prefills (or decode with `MUMMU_FUSED_GDN=0`).
pub mod gdn_l2_probe {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::Qwen35Config;
    use burn::tensor::Tensor;
    use mummu_num::narrow;

    static ON: AtomicBool = AtomicBool::new(false);
    static SINK: Mutex<Vec<Record>> = Mutex::new(Vec::new());

    /// One tensor-path `DeltaNet` forward, in call order: a prefill visits
    /// the `DeltaNet` layers in model order, so the `i`-th record of a single
    /// forward is the `i`-th `DeltaNet` layer.
    #[derive(Debug, Clone)]
    pub struct Record {
        /// Positions in the forward (`b · t`).
        pub tokens: usize,
        /// `n_k_heads`.
        pub heads: usize,
        /// Pre-normalization head norms, `[tokens · heads]` token-major.
        pub q: Vec<f32>,
        pub k: Vec<f32>,
    }

    /// Turn the tap on or off (process-wide).
    pub fn set_enabled(on: bool) {
        ON.store(on, Ordering::Relaxed);
    }

    pub(super) fn enabled() -> bool {
        ON.load(Ordering::Relaxed)
    }

    /// Drain everything recorded since the last call.
    pub fn take() -> Vec<Record> {
        std::mem::take(
            &mut *SINK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Read back `conv_out` `[b, t, conv_dim]` (after conv + `SiLU`) and keep
    /// the q and k head norms, summed in f64.
    pub(super) fn record(conv_out: &Tensor<3>, cfg: &Qwen35Config) {
        let [b, t, _] = conv_out.dims();
        let (heads, ds, key_dim) = (cfg.n_k_heads, cfg.d_state, cfg.key_dim());
        let host = conv_out
            .clone()
            .narrow(2, 0, 2 * key_dim)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("conv output reads back as f32");
        let tokens = b * t;
        let (mut q, mut k) = (
            Vec::with_capacity(tokens * heads),
            Vec::with_capacity(tokens * heads),
        );
        for row in host.chunks_exact(2 * key_dim) {
            let (qs, ks) = row.split_at(key_dim);
            let norm = |x: &[f32]| x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
            q.extend(qs.chunks_exact(ds).map(|h| narrow(norm(h))));
            k.extend(ks.chunks_exact(ds).map(|h| narrow(norm(h))));
        }
        SINK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Record {
                tokens,
                heads,
                q,
                k,
            });
    }
}

/// Longest span the sequential recurrence still evaluates when chunking is
/// on (`MUMMU_GDN_SEQ_MAX`, default 4): above it, even a single partial
/// chunk launches fewer kernels than `~9 · t` sequential steps. `0` keeps
/// only decode (`t == 1`) sequential.
fn gdn_seq_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MUMMU_GDN_SEQ_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4)
            .max(1)
    })
}

/// The per-token inputs of the `DeltaNet` recurrence: `q`/`k`/`v` are
/// `[b, hv, t, ds]` (already conv'd, `SiLU`'d, L2-normed and tiled),
/// `g`/`beta` are `[b, t, hv]` (per-head decay logits and update gates).
#[derive(Clone, Copy)]
struct RecurrenceInputs<'a> {
    q: &'a Tensor<4>,
    k: &'a Tensor<4>,
    v: &'a Tensor<4>,
    g: &'a Tensor<3>,
    beta: &'a Tensor<3>,
}

/// The `DeltaNet` recurrence, one token at a time — decode's path (`t == 1`)
/// and the exactness reference the chunked form is tested against. State
/// `S[b, h, i, j]`: `i` indexes the key dim, `j` the value dim.
///
/// `s0` is the carried state `[b, hv, ds, ds]`. Returns the per-token
/// outputs `[b, hv, t, ds]` and the final state.
fn gdn_recurrence_sequential(
    inputs: &RecurrenceInputs<'_>,
    s0: Tensor<4>,
    scale: f32,
) -> (Tensor<4>, Tensor<4>) {
    let [batch, hv, t, _ds] = inputs.q.dims();
    let mut state = s0;
    let mut outs: Vec<Tensor<4>> = Vec::with_capacity(t);
    for tau in 0..t {
        let q_t = inputs.q.clone().narrow(2, tau, 1).mul_scalar(scale); // [b, hv, 1, ds]
        let k_t = inputs.k.clone().narrow(2, tau, 1); // [b, hv, 1, ds]
        let v_t = inputs.v.clone().narrow(2, tau, 1); // [b, hv, 1, ds]
        let g_t = inputs
            .g
            .clone()
            .narrow(1, tau, 1)
            .reshape([batch, hv, 1, 1]); // per-head decay logit
        let b_t = inputs
            .beta
            .clone()
            .narrow(1, tau, 1)
            .reshape([batch, hv, 1, 1]);

        state = state.mul(g_t.exp());
        // v̂[j] = Σ_i S[i, j]·k[i]  — k over the key axis.
        let v_hat = state.clone().mul(k_t.clone().swap_dims(2, 3)).sum_dim(2); // [b, hv, 1, ds]
        let delta = v_t.sub(v_hat).mul(b_t); // [b, hv, 1, ds]
        // S += k ⊗ d  (outer product over [key, value]).
        state = state.add(k_t.swap_dims(2, 3).matmul(delta));
        // o[j] = Σ_i S[i, j]·q[i].
        let out_t = state.clone().mul(q_t.swap_dims(2, 3)).sum_dim(2); // [b, hv, 1, ds]
        outs.push(out_t);
    }
    (Tensor::cat(outs, 2), state) // [b, hv, t, ds]
}

/// `(I + A)⁻¹` for the chunk system `A[t, j] = β_t·exp(L_t − L_j)·(k_t·k_j)`
/// (`j < t`, else 0), from `k [b, h, c, ds]`, `l [b, h, c, 1]` (cumulative
/// log-decay) and `beta [b, h, c, 1]`, by block recursion: with
/// `I + A = [[L₁₁, 0], [A₂₁, L₂₂]]`,
///
/// ```text
/// (I + A)⁻¹ = [[L₁₁⁻¹, 0], [−L₂₂⁻¹·A₂₁·L₁₁⁻¹, L₂₂⁻¹]]
/// ```
///
/// evaluated bottom-up for every diagonal block of size 1, 2, 4, … at once.
/// Why not a power series: every intermediate here is itself the inverse
/// of a contiguous sub-span's system, whose entries are delta-rule
/// sensitivities bounded by `β ≤ 1` (products of `γ(I − βkkᵀ)`
/// contractions), so f32 never cancels large terms.
///
/// Each level builds its `A₂₁` blocks straight from the factors (the rows of
/// the second half of a pair against the columns of the first), so there is
/// no gather and no index upload on a GPU: reshapes, narrows, three matmuls
/// and three concats per level, `⌈log₂c⌉` levels. `c` is padded to a power
/// of two with `β = 0` rows (zero `A` rows, identity inverse rows) and the
/// last `L` repeated (so padded exponents stay `≤ 0`); the leading `c × c`
/// block of the padded inverse is the answer.
fn unit_lower_inverse(
    keys: Tensor<4>,
    logdecay: Tensor<4>,
    beta: Tensor<4>,
    device: &Device,
) -> Tensor<4> {
    // Against the doc's symbols: `len` is c, `padded` its power-of-two pad,
    // `size` the diagonal block size s, `pairs` the (batch·head, pair) rows.
    let [batch, heads, len, ds] = keys.dims();
    let padded = len.next_power_of_two();
    let bh = batch * heads;
    let dtype = keys.dtype();
    let (mut keys, mut logdecay, mut beta) = (keys, logdecay, beta);
    if padded > len {
        let pad = padded - len;
        let zeros =
            |width: usize| Tensor::<4>::zeros([batch, heads, pad, width], device).cast(dtype);
        keys = Tensor::cat(vec![keys, zeros(ds)], 2);
        beta = Tensor::cat(vec![beta, zeros(1)], 2);
        let last = logdecay.clone().narrow(2, len - 1, 1).repeat_dim(2, pad);
        logdecay = Tensor::cat(vec![logdecay, last], 2);
    }
    let (keys, logdecay, beta) = (
        keys.reshape([bh, padded, ds]),
        logdecay.reshape([bh, padded, 1]),
        beta.reshape([bh, padded, 1]),
    );
    // Diagonal-block inverses of size s, one row per (batch·head, block).
    let mut inv = Tensor::<3>::ones([bh * padded, 1, 1], device).cast(dtype);
    let mut size = 1usize;
    while size < padded {
        let pairs = bh * (padded / (2 * size)); // (batch·head, pair) rows
        let halves = |x: &Tensor<3>, width: usize| {
            let x = x.clone().reshape([pairs, 2 * size, width]);
            (x.clone().narrow(1, 0, size), x.narrow(1, size, size))
        };
        let (k1, k2) = halves(&keys, ds);
        let (l1, l2) = halves(&logdecay, 1);
        let (_, b2) = halves(&beta, 1);
        // Every row of the second half comes after every column of the
        // first, so these exponents are ≤ 0 and need no causal mask.
        let a21 = k2
            .matmul(k1.swap_dims(1, 2))
            .mul(l2.sub(l1.swap_dims(1, 2)).exp())
            .mul(b2); // [n, s, s]
        let pair = inv.reshape([pairs, 2, size, size]);
        let inv11 = pair.clone().narrow(1, 0, 1).reshape([pairs, size, size]);
        let inv22 = pair.narrow(1, 1, 1).reshape([pairs, size, size]);
        let m21 = inv22.clone().matmul(a21).matmul(inv11.clone()).neg();
        let zeros = Tensor::<3>::zeros([pairs, size, size], device).cast(dtype);
        let top = Tensor::cat(vec![inv11, zeros], 2);
        let bottom = Tensor::cat(vec![m21, inv22], 2);
        inv = Tensor::cat(vec![top, bottom], 1); // [n, 2s, 2s]
        size *= 2;
    }
    inv.reshape([bh, padded, padded])
        .narrow(1, 0, len)
        .narrow(2, 0, len)
        .reshape([batch, heads, len, len])
}

/// Chunkwise-parallel evaluation of the same recurrence — the Gated
/// `DeltaNet` / `DeltaNet` WY form (Yang et al.), specialised to this
/// parameterization's **scalar** per-head decay. Same signature as
/// [`gdn_recurrence_sequential`] plus the chunk length.
///
/// Within a chunk of length `C` starting from state `S₀`, write
/// `L_t = Σ_{j≤t} g_j` (cumulative log-decay) and `P_t = exp(L_t)`, so
/// `P_t/P_j` is the decay applied between tokens `j` and `t`. Unrolling
/// `S_t = γ_t·S_{t−1} + k_t d_tᵀ` with `d_t = β_t(v_t − γ_t·S_{t−1}ᵀk_t)`
/// gives, by induction on `t`,
///
/// ```text
/// S_t = P_t·S₀ + Σ_{τ≤t} (P_t/P_τ)·k_τ d_τᵀ
/// ```
///
/// and substituting that back into `d_t` makes the `d`s the solution of a
/// unit lower-triangular system:
///
/// ```text
/// (I + A)·D = B,   A[t,τ] = β_t·(P_t/P_τ)·(k_t·k_τ)      (τ < t, else 0)
///                  B[t]   = β_t·v_t − β_t·P_t·(S₀ᵀ k_t)
/// ```
///
/// With `D` solved, the outputs and the chunk-final state are plain
/// batched matmuls:
///
/// ```text
/// o_t = scale·(P_t·S₀ᵀq_t + Σ_{j≤t} (P_t/P_j)·(q_t·k_j)·d_j)
/// S_C = P_C·S₀ + Σ_τ (P_C/P_τ)·k_τ d_τᵀ
/// ```
///
/// The output sum is INCLUSIVE (`j ≤ t`): the sequential order updates `S`
/// with token `t` before reading `o_t`. Hand-checked against two unrolled
/// steps of the sequential recurrence at `C = 2` (both `d₂ = b₂ − A[2,1]d₁`
/// and the `o`/`S` reads land on the same expressions).
///
/// Exactness of the solve: `(I + A)⁻¹` is formed by [`unit_lower_inverse`]
/// (block recursion over halves, nothing truncated), so the chunked path is
/// an evaluation-order change, not an approximation. It used to be the
/// finite Neumann sum `Σ_{k<C} N^k` of `N = −A` in doubling stages, which is
/// algebraically exact too but numerically unusable when keys repeat: with
/// `k_t·k_j ≈ 1`, `β ≈ 1` and little decay, `N^k` holds binomials up to
/// C(62, 31) ≈ 5e17 whose sum cancels to O(1), and f32 turned that into
/// garbage and then NaN (Flash-Next layer 45 on a repetitive 285-token
/// prompt; `chunked_recurrence_survives_repeated_keys`).
///
/// Numerical safety: every decay ratio `P_t/P_j` is `exp(L_t − L_j)` — a
/// difference of cumulative logs, never a ratio of products (γ^64 reaches
/// ~1e-19 at half-life decay per step; the products underflow f32 where
/// the log differences stay O(10)). Non-causal entries are masked to −∞
/// BEFORE the exp — their raw exponents are positive and would overflow
/// under strong decay. The whole chunk runs on an f32 island (the same
/// guard `GatedAttention` uses for its scores) and casts back to the
/// ambient dtype on the way out.
///
/// Launch economics: the sequential loop issues ~9 small ops per token;
/// a chunk issues ~20 chunk-level ops plus ~6 per level of the inverse
/// (6 levels at C = 64) — ~56 launches per 64 tokens, the heavy ones
/// batched over all heads, against ~576 for the same span sequentially.
fn gdn_recurrence_chunked(
    inputs: &RecurrenceInputs<'_>,
    s0: Tensor<4>,
    scale: f32,
    chunk: usize,
) -> (Tensor<4>, Tensor<4>) {
    let [batch, hv, t, _ds] = inputs.q.dims();
    let device = inputs.q.device();
    let ambient = inputs.q.dtype();

    // The f32 island. q is pre-scaled once — equivalent to the sequential
    // per-token mul_scalar, which commutes through everything q touches.
    let qf = inputs.q.clone().cast(DType::F32).mul_scalar(scale);
    let kf = inputs.k.clone().cast(DType::F32);
    let vf = inputs.v.clone().cast(DType::F32);
    // [b, t, hv] → [b, hv, t, 1], ready to broadcast over ds and columns.
    let gf = inputs
        .g
        .clone()
        .cast(DType::F32)
        .swap_dims(1, 2)
        .reshape([batch, hv, t, 1]);
    let bf = inputs
        .beta
        .clone()
        .cast(DType::F32)
        .swap_dims(1, 2)
        .reshape([batch, hv, t, 1]);

    let mut state = s0.cast(DType::F32); // [b, hv, ds(key), ds(value)]
    let mut outs: Vec<Tensor<4>> = Vec::with_capacity(t.div_ceil(chunk));
    let mut start = 0;
    while start < t {
        let len = chunk.min(t - start); // the final chunk may be partial
        let qc = qf.clone().narrow(2, start, len); // [b, hv, c, ds]
        let kc = kf.clone().narrow(2, start, len);
        let vc = vf.clone().narrow(2, start, len);
        let gc = gf.clone().narrow(2, start, len); // [b, hv, c, 1]
        let bc = bf.clone().narrow(2, start, len);

        // Cumulative log-decay L_t and its exponential P_t, both within
        // the chunk (g ≤ 0, so L is non-increasing and P ∈ (0, 1]).
        let l_cum = gc.cumsum(2); // [b, hv, c, 1]
        let p_t = l_cum.clone().exp();
        // decay[t, j] = P_t/P_j = exp(L_t − L_j) for j ≤ t, unit diagonal.
        let noncausal = Tensor::<2, Bool>::tril_mask([len, len], 0, &device).unsqueeze::<4>();
        let decay = l_cum
            .clone()
            .sub(l_cum.clone().swap_dims(2, 3))
            .mask_fill(noncausal, f32::NEG_INFINITY)
            .exp(); // [b, hv, c, c], causal-inclusive

        // D = (I + A)⁻¹·B with A[t, j] = β_t·(P_t/P_j)·(k_t·k_j) strictly
        // lower; the inverse is built by stable block recursion.
        let inv = unit_lower_inverse(kc.clone(), l_cum.clone(), bc.clone(), &device);

        // RHS rows: B[t] = β_t·v_t − β_t·P_t·(S₀ᵀ k_t).
        let u0 = bc
            .clone()
            .mul(vc)
            .sub(bc.mul(p_t.clone()).mul(kc.clone()).matmul(state.clone()));
        let pseudo = inv.matmul(u0); // the solved pseudo-values, [b, hv, c, ds]

        // o_t = P_t·S₀ᵀq_t + Σ_{j≤t} (P_t/P_j)·(q_t·k_j)·u_j — inclusive
        // j ≤ t is exactly the unit diagonal of `decay`.
        let qk = qc.clone().matmul(kc.clone().swap_dims(2, 3)).mul(decay);
        let o_c = p_t
            .mul(qc)
            .matmul(state.clone())
            .add(qk.matmul(pseudo.clone()));
        outs.push(o_c);

        // S_C = P_C·S₀ + Σ_j (P_C/P_j)·k_j u_jᵀ, again via log differences.
        let l_last = l_cum.clone().narrow(2, len - 1, 1); // [b, hv, 1, 1] = L_C
        let to_end = l_last.clone().sub(l_cum).exp(); // (P_C/P_j) ∈ (0, 1]
        state = state
            .mul(l_last.exp())
            .add(kc.mul(to_end).swap_dims(2, 3).matmul(pseudo));

        start += len;
    }
    (Tensor::cat(outs, 2).cast(ambient), state.cast(ambient))
}

/// One trunk layer: exactly one of `self_attn` / `linear_attn`.
#[derive(Module, Debug)]
pub struct Qwen35Layer {
    pub input_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
    pub self_attn: Option<GatedAttention>,
    pub linear_attn: Option<GatedDeltaNet>,
    pub mlp: SwiGluMlp,
}

/// The qwen35 decoder stack.
#[derive(Module, Debug)]
pub struct Qwen35 {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen35Layer>,
    pub norm: RmsNorm,
    /// Untied head when the checkpoint carries `output.weight`; tied to the
    /// embedding otherwise.
    pub lm_head: Option<Linear>,
}

/// Per-layer decode cache.
pub enum Qwen35Kv {
    Attn(LayerKv),
    Delta(DeltaState),
}

/// A weight-loaded qwen35 plus its config.
pub struct LoadedQwen35 {
    pub model: Qwen35,
    pub config: Qwen35Config,
    /// `None` for GGUF loads (self-contained; no sibling file).
    pub tokenizer_config: Option<crate::tok_config::TokenizerConfig>,
    /// P9 stage 3(c): remote FFN clusters of a partitioned pack — each
    /// layer's `mlp` then holds only the *local* clusters and the pool adds
    /// the rest (exact when every cluster runs). `None` = plain dense.
    pub ffn_pool: Option<std::sync::Arc<crate::nn::ExpertPool>>,
    /// Opt-in skipping: clusters whose gate energy is below `tau` × the
    /// row's total energy are not computed (lossy — only from a measured
    /// skip table). `0.0` = exact.
    pub ffn_skip_tau: f32,
    /// P9 stage 4: the working-set schedule this model runs under, one entry
    /// per trunk layer. `None` = every cluster is permanently resident (the
    /// tier design), so there is nothing to stage.
    pub ffn_plan: Option<std::sync::Arc<crate::workingset::Plan>>,
}

fn build(cfg: &Qwen35Config, device: &Device, untied_head: bool) -> Qwen35 {
    let norm = |dim: usize, dev: &Device| rms_norm(cfg, dim, dev);
    let linear = bias_free_linear;
    let mlp_cfg = SwiGluMlpConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
    };
    let layers = (0..cfg.num_layers)
        .map(|i| {
            let attn = cfg.is_attention(i);
            Qwen35Layer {
                input_norm: norm(cfg.hidden_size, device),
                post_attn_norm: norm(cfg.hidden_size, device),
                self_attn: attn.then(|| GatedAttention::init(cfg, device)),
                linear_attn: (!attn).then(|| GatedDeltaNet::init(cfg, device)),
                mlp: mlp_cfg.init(device),
            }
        })
        .collect();
    Qwen35 {
        embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
        layers,
        norm: norm(cfg.hidden_size, device),
        lm_head: untied_head.then(|| linear(cfg.hidden_size, cfg.vocab_size, device)),
    }
}

/// GGUF (llama.cpp `qwen35` arch) names → this port's parameter paths.
/// `trunk_layers` gates the NextN/MTP block: any `blk.i` at or beyond it is
/// the draft head, deliberately skipped (unused by the main forward).
fn gguf_tensor_map(info: &GgufTensorInfo, trunk_layers: usize) -> Option<GgufMap> {
    match info.name.as_str() {
        "token_embd.weight" => {
            return Some(GgufMap::Rename("model.embed_tokens.weight".into()));
        }
        "output_norm.weight" => return Some(GgufMap::Rename("model.norm.weight".into())),
        "output.weight" => return Some(GgufMap::Rename("lm_head.weight".into())),
        _ => {}
    }
    let rest = info.name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= trunk_layers {
        return Some(GgufMap::Skip); // the NextN/MTP draft block
    }
    if field == "ssm_conv1d.weight" {
        // ggml stores the depthwise kernel squeezed [k, channels]; the
        // Conv1d module wants [channels, 1, k] — same bytes.
        let (&k, &channels) = (info.dims.first()?, info.dims.get(1)?);
        return Some(GgufMap::Reshape(
            format!("model.layers.{layer}.linear_attn.conv1d.weight"),
            vec![channels, 1, k],
        ));
    }
    let mapped = qwen35_field(field)?;
    Some(GgufMap::Rename(format!("model.layers.{layer}.{mapped}")))
}

/// GGUF per-layer field → this port's parameter path (minus the layer
/// prefix). Shared by the GGUF map and the pack loader (and qwen4exp, whose
/// GDN and attention tensors carry the same names and shapes).
pub(crate) fn qwen35_field(field: &str) -> Option<&'static str> {
    Some(match field {
        "attn_norm.weight" => "input_norm.weight",
        "post_attention_norm.weight" => "post_attn_norm.weight",
        // Full-attention layers.
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        // Gated DeltaNet layers.
        "attn_qkv.weight" => "linear_attn.qkv_proj.weight",
        "attn_gate.weight" => "linear_attn.z_proj.weight",
        "ssm_beta.weight" => "linear_attn.beta_proj.weight",
        "ssm_alpha.weight" => "linear_attn.alpha_proj.weight",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_a" => "linear_attn.a",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        // FFN.
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    })
}

/// Load a qwen35 model straight from a GGUF file — the classic f32 path,
/// which is [`load_from_gguf_quantized`] with quantization off. One import
/// path serves every precision (P9's "single path" rule).
///
/// # Errors
///
/// As [`load_from_gguf_quantized`].
pub fn load_from_gguf(path: &Path, device: &Device) -> Result<LoadedQwen35, ImportError> {
    load_from_gguf_quantized(path, device, QuantPolicy::Off)
}

/// **Streaming** GGUF import with optional keep-quantized weights.
///
/// One tensor at a time is dequantized to f32 (whatever the source stored —
/// BF16, K-quants, IQ quants), moved to the device, **re-quantized** per
/// `policy` when eligible, and assigned. Peak memory is the finished model
/// plus a single f32 tensor — never the whole model at f32, which is what
/// makes the 27B tier loadable at all (its f32 form is ~109 GB).
///
/// # Errors
///
/// An [`ImportError::Parse`] naming `path` for: a file that does not open
/// or parse as GGUF; a header [`Qwen35Config::from_gguf`] refuses; a tensor
/// name the architecture has no place for; a dimension that does not fit
/// `usize`; a tensor that does not read/dequantize or whose shape its field
/// rejects; or a trunk tensor count other than the architecture's.
pub fn load_from_gguf_quantized(
    path: &Path,
    device: &Device,
    policy: QuantPolicy,
) -> Result<LoadedQwen35, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = Qwen35Config::from_gguf(&f).map_err(parse)?;
    let untied = f.tensor("output.weight").is_some();
    let trunk = config.num_layers;
    let mut model = build(&config, device, untied);

    // The pack-less path: every tensor is read and quantized straight out of
    // the GGUF, which on a spinning array is the slowest load there is. Bytes
    // come from the tensors' own on-disk sizes, so the rate reported is the
    // disk's, not the f32 expansion's.
    let expected = expected_tensor_count(&config, untied);
    crate::progress::begin(
        crate::progress::Phase::Loading,
        expected as u64,
        crate::progress::Unit::Tensors,
    );
    let mut assigned = 0usize;
    let mut bytes_read = 0u64;
    for info in &f.tensors {
        let mapped = gguf_tensor_map(info, trunk)
            .ok_or_else(|| parse(format!("unmapped tensor name '{}'", info.name)))?;
        let (name, shape) = match mapped {
            GgufMap::Skip => continue,
            GgufMap::Rename(name) => (name, dims_usize(info.dims.iter().rev()).map_err(parse)?),
            GgufMap::Reshape(name, shape) => (name, dims_usize(shape.iter()).map_err(parse)?),
        };
        let values = f
            .read_tensor_f32(&info.name)
            .map_err(|e| parse(e.to_string()))?;
        assign_param(
            &mut model,
            &name,
            ParamSrc::F32 { values, shape },
            policy,
            device,
        )
        .map_err(parse)?;
        assigned += 1;
        bytes_read += info.byte_len();
        crate::progress::advance(assigned as u64, bytes_read);
    }

    // Both directions of completeness, loudly: every mapped tensor landed
    // (assign_param errors otherwise) and the count matches what the
    // architecture requires.
    if assigned != expected {
        return Err(parse(format!(
            "GGUF supplied {assigned} trunk tensors, the architecture needs {expected}"
        )));
    }
    Ok(LoadedQwen35 {
        model,
        config,
        tokenizer_config: None,
        ffn_pool: None,
        ffn_skip_tau: 0.0,
        ffn_plan: None,
    })
}

/// GGUF `u64` dims as `usize`, or which one does not fit (a 32-bit target).
fn dims_usize<'a>(dims: impl Iterator<Item = &'a u64>) -> Result<Vec<usize>, String> {
    dims.map(|&d| usize::try_from(d).map_err(|_| format!("dimension {d} does not fit usize")))
        .collect()
}

/// How many trunk tensors a checkpoint must supply (the completeness gate's
/// other half). Per attention layer 11 (2 norms + q/k/v/o + q/k norm +
/// 3 FFN), per `DeltaNet` layer 14 (2 norms + qkv/z + β/α/dt/a + conv +
/// ssm-norm + out + 3 FFN), plus embedding, final norm, and the untied head
/// when present.
fn expected_tensor_count(cfg: &Qwen35Config, untied: bool) -> usize {
    let per_layer: usize = (0..cfg.num_layers)
        .map(|i| if cfg.is_attention(i) { 11 } else { 14 })
        .sum();
    per_layer + 2 + usize::from(untied)
}

/// Build a device tensor from row-major f32 `values` of `shape`, cast to the
/// backend float dtype.
pub(crate) fn device_tensor<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &Device,
) -> Tensor<D> {
    let dtype = crate::backend::float_dtype(device);
    Tensor::from_data(TensorData::new(values, shape), (device, dtype))
}

/// A 2-D **linear weight**: GGUF row-major is `[out, in]`, burn's `Linear`
/// wants `[in, out]` — transpose on device, then quantize when the policy
/// takes it.
pub(crate) fn linear_weight(
    values: Vec<f32>,
    shape: &[usize],
    policy: QuantPolicy,
    device: &Device,
) -> Result<Tensor<2>, String> {
    let &[out, inp] = shape else {
        return Err(format!("linear weight must be 2-D, got {shape:?}"));
    };
    let w = device_tensor::<2>(values, [out, inp], device).swap_dims(0, 1);
    Ok(if policy.eligible(&[inp, out]) {
        crate::quant::quantize_weight(policy, w)
    } else {
        w
    })
}

/// Where a parameter's data comes from: raw f32 (the GGUF streaming path —
/// transposed/quantized here) or a ready 2-D tensor (the pack path — already
/// `[in, out]` at its chosen precision).
pub enum ParamSrc {
    F32 {
        values: Vec<f32>,
        shape: Vec<usize>,
    },
    /// Boxed: a ready tensor is ~256 bytes against `F32`'s ~48. This is a
    /// load-time value, so the extra allocation is free and the enum stops
    /// carrying the larger variant's footprint everywhere.
    Ready2(Box<Tensor<2>>),
}

/// A linear weight from either source.
fn take_linear(
    ready: Option<Tensor<2>>,
    values: Vec<f32>,
    shape: &[usize],
    policy: QuantPolicy,
    device: &Device,
) -> Result<Tensor<2>, String> {
    ready.map_or_else(|| linear_weight(values, shape, policy, device), Ok)
}

/// One parameter's data on its way into a module field: the raw values
/// and shape, the ready tensor when the pack supplied one, and what a
/// linear needs to be transposed/quantized.
struct RawParam<'a> {
    values: Vec<f32>,
    shape: &'a [usize],
    ready: Option<Tensor<2>>,
    policy: QuantPolicy,
    device: &'a Device,
}

impl RawParam<'_> {
    /// A 1-D parameter (norm gammas, `dt_bias`, `a`).
    fn vector(self) -> Result<Tensor<1>, String> {
        let &[n] = self.shape else {
            return Err(format!("expected 1-D, got {:?}", self.shape));
        };
        Ok(device_tensor::<1>(self.values, [n], self.device))
    }

    /// A linear weight from either source.
    fn linear(self) -> Result<Tensor<2>, String> {
        take_linear(
            self.ready,
            self.values,
            self.shape,
            self.policy,
            self.device,
        )
    }
}

/// Route one field of an attention block.
fn assign_attn_field(
    attn: &mut GatedAttention,
    field: &str,
    raw: RawParam<'_>,
) -> Result<(), String> {
    match field {
        "q_proj.weight" => attn.q_proj.weight = Param::from_tensor(raw.linear()?),
        "k_proj.weight" => attn.k_proj.weight = Param::from_tensor(raw.linear()?),
        "v_proj.weight" => attn.v_proj.weight = Param::from_tensor(raw.linear()?),
        "o_proj.weight" => attn.o_proj.weight = Param::from_tensor(raw.linear()?),
        "q_norm.weight" => attn.q_norm.gamma = Param::from_tensor(raw.vector()?),
        "k_norm.weight" => attn.k_norm.gamma = Param::from_tensor(raw.vector()?),
        other => return Err(format!("unknown attention field '{other}'")),
    }
    Ok(())
}

/// Route one field of a `DeltaNet` block.
fn assign_delta_field(
    delta: &mut GatedDeltaNet,
    field: &str,
    raw: RawParam<'_>,
) -> Result<(), String> {
    match field {
        "qkv_proj.weight" => delta.qkv_proj.weight = Param::from_tensor(raw.linear()?),
        "z_proj.weight" => delta.z_proj.weight = Param::from_tensor(raw.linear()?),
        "beta_proj.weight" => delta.beta_proj.weight = Param::from_tensor(raw.linear()?),
        "alpha_proj.weight" => delta.alpha_proj.weight = Param::from_tensor(raw.linear()?),
        "out_proj.weight" => delta.out_proj.weight = Param::from_tensor(raw.linear()?),
        "dt_bias" => delta.dt_bias = Param::from_tensor(raw.vector()?),
        "a" => delta.a = Param::from_tensor(raw.vector()?),
        "norm.weight" => delta.norm.gamma = Param::from_tensor(raw.vector()?),
        "conv1d.weight" => {
            let &[ch, one, k] = raw.shape else {
                return Err(format!("conv kernel must be 3-D, got {:?}", raw.shape));
            };
            if one != 1 {
                return Err(format!("conv kernel middle dim must be 1, got {one}"));
            }
            delta.conv1d.weight =
                Param::from_tensor(device_tensor::<3>(raw.values, [ch, 1, k], raw.device));
        }
        other => return Err(format!("unknown DeltaNet field '{other}'")),
    }
    Ok(())
}

/// Route one layer field (`name` is the full path, for the error text):
/// the norms and MLP here, the block fields on to their block.
fn assign_layer_field(
    layer: &mut Qwen35Layer,
    name: &str,
    field: &str,
    raw: RawParam<'_>,
) -> Result<(), String> {
    match field {
        "input_norm.weight" => layer.input_norm.gamma = Param::from_tensor(raw.vector()?),
        "post_attn_norm.weight" => layer.post_attn_norm.gamma = Param::from_tensor(raw.vector()?),
        "mlp.gate_proj.weight" => layer.mlp.gate_proj.weight = Param::from_tensor(raw.linear()?),
        "mlp.up_proj.weight" => layer.mlp.up_proj.weight = Param::from_tensor(raw.linear()?),
        "mlp.down_proj.weight" => layer.mlp.down_proj.weight = Param::from_tensor(raw.linear()?),
        _ => {
            if let Some(attn_field) = field.strip_prefix("self_attn.") {
                let attn = layer.self_attn.as_mut().ok_or_else(|| {
                    format!("'{name}' targets an attention block on a DeltaNet layer")
                })?;
                return assign_attn_field(attn, attn_field, raw);
            }
            if let Some(delta_field) = field.strip_prefix("linear_attn.") {
                let delta = layer.linear_attn.as_mut().ok_or_else(|| {
                    format!("'{name}' targets a DeltaNet block on an attention layer")
                })?;
                return assign_delta_field(delta, delta_field, raw);
            }
            return Err(format!("unknown layer field '{field}'"));
        }
    }
    Ok(())
}

/// Route one mapped tensor into its module field. An unknown path is a loud
/// error — silence here would mean silently dropped weights.
fn assign_param(
    model: &mut Qwen35,
    name: &str,
    src: ParamSrc,
    policy: QuantPolicy,
    device: &Device,
) -> Result<(), String> {
    let (values, shape_vec, ready): (Vec<f32>, Vec<usize>, Option<Tensor<2>>) = match src {
        ParamSrc::F32 { values, shape } => (values, shape, None),
        ParamSrc::Ready2(t) => (Vec::new(), t.dims().to_vec(), Some(*t)),
    };
    let raw = RawParam {
        values,
        shape: shape_vec.as_slice(),
        ready,
        policy,
        device,
    };

    match name {
        "model.embed_tokens.weight" => {
            // Embeddings stay float — token gather has no quantized kernel.
            let t = if let Some(t) = raw.ready {
                t
            } else {
                let &[v, h] = raw.shape else {
                    return Err(format!("embedding must be 2-D, got {:?}", raw.shape));
                };
                device_tensor::<2>(raw.values, [v, h], device)
            };
            model.embed_tokens.weight = Param::from_tensor(t);
            return Ok(());
        }
        "model.norm.weight" => {
            model.norm.gamma = Param::from_tensor(raw.vector()?);
            return Ok(());
        }
        "lm_head.weight" => {
            let head = model
                .lm_head
                .as_mut()
                .ok_or("checkpoint has output.weight but the model built a tied head")?;
            head.weight = Param::from_tensor(raw.linear()?);
            return Ok(());
        }
        _ => {}
    }

    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| format!("unknown parameter path '{name}'"))?;
    let (layer, field) = rest
        .split_once('.')
        .ok_or_else(|| format!("bad layer path '{name}'"))?;
    let layer: usize = layer
        .parse()
        .map_err(|_| format!("bad layer in '{name}'"))?;
    let layer = model
        .layers
        .get_mut(layer)
        .ok_or_else(|| format!("layer {layer} out of range"))?;
    assign_layer_field(layer, name, field, raw)
}

/// How each GGUF tensor enters a `.mummu` pack (see `crate::pack`).
#[must_use]
pub fn pack_actions(
    info: &GgufTensorInfo,
    trunk_layers: usize,
) -> Option<crate::pack::ImportAction> {
    use crate::pack::ImportAction as A;
    match info.name.as_str() {
        "token_embd.weight" => return Some(A::Embedding),
        "output_norm.weight" => return Some(A::Vector),
        "output.weight" => return Some(A::Linear),
        _ => {}
    }
    let rest = info.name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= trunk_layers {
        return Some(A::Skip);
    }
    Some(match field {
        "ssm_conv1d.weight" => A::Conv,
        "attn_norm.weight"
        | "post_attention_norm.weight"
        | "attn_q_norm.weight"
        | "attn_k_norm.weight"
        | "ssm_dt.bias"
        | "ssm_a"
        | "ssm_norm.weight" => A::Vector,
        _ => {
            qwen35_field(field)?; // known linear fields only
            A::Linear
        }
    })
}

/// The FFN entry names of every trunk layer.
///
/// Re-exported from [`crate::partition::ffn_names`], which is where it
/// belongs: every dense decoder in the zoo stores the same GGUF triple, so
/// this is not a qwen35 fact. Kept as a path so existing callers do not move.
pub use crate::partition::ffn_names;

/// Verify-mode radial lookahead on? (`MUMMU_LOOKAHEAD=verify`). One env
/// read per process; anything but `verify` is off — there is no commit mode
/// until verify-mode acceptance earns it.
fn lookahead_verify() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MUMMU_LOOKAHEAD").is_ok_and(|v| v.eq_ignore_ascii_case("verify"))
    })
}

/// Debug: print a tensor's row-0 head under `MUMMU_LAYER_TRACE`.
pub(crate) fn trace_tensor<const D: usize>(name: &str, t: &Tensor<D>) {
    if !layer_trace() {
        return;
    }
    let dims = t.dims();
    let v = t
        .clone()
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap_or_default();
    let sum: f64 = v.iter().map(|&a| f64::from(a)).sum();
    let last = dims[D - 1];
    eprintln!(
        "[layer-trace] {name}: dims={dims:?} sum={sum:.6} row0={:?} row0_tail={:?}",
        &v[..v.len().min(4)],
        &v[last.saturating_sub(3).min(v.len())..last.min(v.len())]
    );
}

/// Per-layer residual dump (`MUMMU_LAYER_TRACE=1`) — debug only.
fn layer_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MUMMU_LAYER_TRACE").is_ok())
}

/// Residual-geometry probe on? (`MUMMU_RESIDUAL_PROBE=1`).
fn residual_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MUMMU_RESIDUAL_PROBE").is_ok())
}

/// A scratch copy of one layer's decode state for speculation: tensor
/// clones are refcounted handles, and burn ops never mutate buffers in
/// place, so the speculative forward can freely reassign the scratch
/// struct's fields while the real cache entry stays untouched. Cheap by
/// construction — this is the "never pollute S" rule made structural.
fn snapshot_kv(kv: &Qwen35Kv) -> Qwen35Kv {
    match kv {
        Qwen35Kv::Attn(k) => Qwen35Kv::Attn(k.clone()),
        Qwen35Kv::Delta(d) => Qwen35Kv::Delta(DeltaState {
            conv: d.conv.clone(),
            state: d.state.clone(),
            host_conv: d.host_conv.clone(),
            host_state: d.host_state.clone(),
            middle: d.middle.clone(),
        }),
    }
}

/// Pack tensor name → parameter path (the pack keeps GGUF names).
fn pack_param_path(name: &str, trunk_layers: usize) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= trunk_layers {
        return None;
    }
    if field == "ssm_conv1d.weight" {
        return Some(format!("model.layers.{layer}.linear_attn.conv1d.weight"));
    }
    Some(format!("model.layers.{layer}.{}", qwen35_field(field)?))
}

/// Load from a `.mummu` pack, choosing each tensor's precision through
/// `choose` (the planner's tiering hook — per tensor, so a policy can mix
/// levels). Quantized levels arrive pre-packed (no re-quantization).
///
/// # Errors
///
/// An [`ImportError::Parse`] naming `dir` for: a pack that does not open,
/// or whose header [`Qwen35Config::from_gguf`] refuses; an entry with no
/// stored precision, or one that does not read; a tensor its field rejects;
/// or a trunk tensor count other than the architecture's.
pub fn load_from_pack(
    dir: &Path,
    device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
) -> Result<LoadedQwen35, ImportError> {
    load_from_pack_inner(dir, device, choose, None, None)
}

/// Load a **partitioned** pack with only `local(layer)` FFN clusters in each
/// layer's `mlp`.
///
/// Each entry is at the level `choose` picks for it; the caller attaches the
/// remote clusters as an `ExpertPool` via [`LoadedQwen35::with_ffn_pool`].
/// Every layer must keep at least one local cluster.
///
/// # Errors
///
/// As [`load_from_pack`], plus: a Hadamard-folded checkpoint (its
/// down-projection transform spans the whole FFN axis); a pack with no FFN
/// partition; a layer whose `local` set is empty or names a cluster the
/// partition does not have.
pub fn load_from_pack_partitioned(
    dir: &Path,
    device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
    local: &dyn Fn(usize) -> Vec<usize>,
) -> Result<LoadedQwen35, ImportError> {
    load_from_pack_inner(dir, device, choose, Some(local), None)
}

/// [`load_from_pack_partitioned`], with the **embedding table pinned to
/// `embed_device`** instead of following the rest of the model.
///
/// An embedding is a gather, not a matmul, so it is the one large tensor a
/// pack never quantizes — on the 27B it is 5.09 GB, a quarter of the model,
/// and it is read once per token while a layer's weights are read 65 times.
/// Holding it on the host frees that VRAM for weights that actually compute,
/// at the cost of one `[1, hidden]` transfer per token.
///
/// # Errors
///
/// As [`load_from_pack_partitioned`].
pub fn load_from_pack_partitioned_split(
    dir: &Path,
    device: &Device,
    embed_device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
    local: &dyn Fn(usize) -> Vec<usize>,
) -> Result<LoadedQwen35, ImportError> {
    load_from_pack_inner(dir, device, choose, Some(local), Some(embed_device))
}

/// Load a pack with **each layer on its own device** — the dense-model
/// placement (llama.cpp calls the knob `n_gpu_layers`).
///
/// `layer_device(l)` says where layer `l` lives; every tensor of that layer,
/// trunk and FFN alike, is loaded there, so a layer never crosses a device
/// boundary mid-computation. Activations cross once, where the assignment
/// changes.
///
/// This exists because the cluster-granular path is the wrong shape for a
/// dense model: it splits each layer's FFN across devices, and since every
/// cluster runs on every token there is no selectivity to pay for the
/// crossing — measured at 24.7 s/tok against 4.8 for keeping layers whole.
/// Cluster granularity stays right for a routed `MoE`, where only top-k
/// experts are touched.
///
/// # Errors
///
/// An [`ImportError::Parse`] naming `dir` for: a pack that does not open,
/// or whose header [`Qwen35Config::from_gguf`] refuses; an entry with no
/// stored precision, or one that does not read; a tensor its field rejects;
/// or a trunk tensor count other than the architecture's.
pub fn load_from_pack_layered(
    dir: &Path,
    layer_device: &dyn Fn(usize) -> Device,
    embed_device: &Device,
    head_device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
) -> Result<LoadedQwen35, ImportError> {
    use crate::pack::{Pack, Role};
    let parse = |reason: String| ImportError::Parse {
        file: dir.to_path_buf(),
        reason,
    };
    let pack = Pack::open(dir).map_err(parse)?;
    let header = pack.header().map_err(parse)?;
    let config = Qwen35Config::from_gguf(&header).map_err(parse)?;
    let untied = pack.entry("output.weight").is_some();
    let trunk = config.num_layers;

    // Build the skeleton on the HOST, never on an accelerator.
    //
    // `build` materializes every parameter as f32. Building it on layer 0's
    // device asked ONE device for the whole model at f32 — measured from the
    // manifest, 100.20 GiB against a card with ~13 GiB usable, a 7.7x
    // overshoot — so allocation failed roughly a thousand times and the card
    // ended up holding nothing while the planner reported 44 of 64 layers
    // placed. None of it was ever needed: the loop below replaces every
    // parameter through `assign_param`, which takes the destination device
    // per tensor, and the `assigned != expected` check proves none is
    // missed. These are dead allocations — every one is overwritten before
    // any reader sees it — so dropping them cannot change results.
    //
    // The per-layer `to_device` pre-move goes for the same reason: it moved
    // f32 skeleton weights onto the card purely to overwrite them.
    let build_host = crate::backend::cpu_device();
    let mut model = build(&config, &build_host, untied);

    // A cold load is a multi-minute disk read (~17 GB for the 27B off an
    // HDD), and nothing else prints between the fit plan and "resident" —
    // without these lines a slow load is indistinguishable from a hang.
    // Bytes come from the pack's own read counter, so the line reports what
    // actually came off the disk.
    let expected = expected_tensor_count(&config, untied);
    let load_started = std::time::Instant::now();
    let mut last_progress = std::time::Instant::now();
    // Announce before the first read: the embedding is a single multi-GiB
    // range near the front of the manifest, so the first between-tensor
    // progress check can be tens of seconds out.
    eprintln!("[mummu] load: assigning {expected} tensors from the pack");
    // The same numbers, structured, for a progress bar: the print above is
    // throttled to one line per 15 s so `docker logs` stays readable, which is
    // far too coarse for a bar. See `crate::progress`.
    crate::progress::begin(
        crate::progress::Phase::Loading,
        expected as u64,
        crate::progress::Unit::Tensors,
    );

    let mut assigned = 0usize;
    for entry in &pack.manifest.tensors {
        let Some(path) = pack_param_path(&entry.name, trunk) else {
            continue;
        };
        // Which device this tensor belongs on: its layer's, or the special
        // homes for the embedding and the head.
        let device = match entry.role {
            Role::Embedding => embed_device.clone(),
            _ => layer_of_path(&path).map_or_else(|| head_device.clone(), &layer_device),
        };
        let precision = stored_precision(entry, choose)
            .ok_or_else(|| parse(format!("'{}' has no stored precision", entry.name)))?;
        let src = match entry.role {
            Role::Linear | Role::Expert { .. } | Role::Embedding => ParamSrc::Ready2(Box::new(
                pack.tensor::<2>(entry, precision, &device).map_err(parse)?,
            )),
            Role::Vector | Role::Conv => ParamSrc::F32 {
                values: pack.read_f32(entry).map_err(parse)?,
                shape: entry.shape.clone(),
            },
        };
        assign_param(&mut model, &path, src, QuantPolicy::Off, &device).map_err(parse)?;
        assigned += 1;
        // Every tensor, not every 15 s: three relaxed stores next to a
        // multi-megabyte disk read and a dequantize.
        crate::progress::advance(assigned as u64, pack.bytes_read());
        if last_progress.elapsed().as_secs() >= 15 {
            let gib = f64_from_u64(pack.bytes_read()) / f64::from(1u32 << 30);
            let secs = load_started.elapsed().as_secs_f64();
            eprintln!(
                "[mummu] load: {assigned}/{expected} tensors — {gib:.2} GiB off the pack in {secs:.0}s ({:.0} MB/s)",
                f64_from_u64(pack.bytes_read()) / f64::from(1u32 << 20) / secs,
            );
            last_progress = std::time::Instant::now();
        }
    }
    // Every parameter above was assigned onto its own device. Pin the three
    // specials, which the pack may not cover on every path; `to_device` is a
    // no-op for a tensor already home.
    model.embed_tokens = model.embed_tokens.clone().to_device(embed_device);
    model.norm = model.norm.clone().to_device(head_device);
    if let Some(h) = model.lm_head.take() {
        model.lm_head = Some(h.to_device(head_device));
    }
    if assigned != expected {
        return Err(parse(format!(
            "pack supplied {assigned} trunk tensors, the architecture needs {expected}"
        )));
    }
    Ok(LoadedQwen35 {
        model,
        config,
        tokenizer_config: None,
        ffn_pool: None,
        ffn_skip_tau: 0.0,
        ffn_plan: None,
    })
}

/// Move whole layers of a resident [`load_from_pack_layered`] model to
/// `device`, re-reading each of their tensors from the pack at the precision
/// `choose` names. Returns the bytes read off the pack.
///
/// This is how a placement changes without a reload: a layer is ~220 MiB of
/// the 27B, so moving a few costs seconds where reloading the model costs
/// minutes of disk. Tensors are re-read rather than converted from their
/// current home because the two homes keep different precisions (the host
/// runs Q4 for its bandwidth, the card whatever its mix chose), and a
/// requantized copy compounds two roundings where the pack's blob has one —
/// measured 0.0997 vs 0.1090 for Q8-then-requantize on the ladder probe.
///
/// Every tensor of a moved layer lands on `device` before the old copy is
/// released (it is replaced in place), so the peak is the old layer plus the
/// new one, never the whole model twice. The forward pass follows each
/// layer's own device (see `Qwen35::forward`), so a model placed as any mix
/// of layers runs unchanged. Caches are per generation, so the caller must
/// hold the model exclusively (no generation in flight) — `&mut` says so.
///
/// # Errors
/// A pack that cannot be read, or a tensor it does not store.
pub fn relocate_layers(
    loaded: &mut LoadedQwen35,
    dir: &Path,
    layers: &[usize],
    device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
) -> Result<u64, ImportError> {
    use crate::pack::{Pack, Role};
    let parse = |reason: String| ImportError::Parse {
        file: dir.to_path_buf(),
        reason,
    };
    if layers.is_empty() {
        return Ok(0);
    }
    let pack = Pack::open(dir).map_err(parse)?;
    let trunk = loaded.config.num_layers;
    if let Some(&bad) = layers.iter().find(|&&l| l >= trunk) {
        return Err(parse(format!("layer {bad} is past the trunk ({trunk})")));
    }
    let mut moved = 0usize;
    for entry in &pack.manifest.tensors {
        let Some(path) = pack_param_path(&entry.name, trunk) else {
            continue;
        };
        if !layer_of_path(&path).is_some_and(|l| layers.contains(&l)) {
            continue;
        }
        let precision = stored_precision(entry, choose)
            .ok_or_else(|| parse(format!("'{}' has no stored precision", entry.name)))?;
        let src = match entry.role {
            Role::Linear | Role::Expert { .. } | Role::Embedding => ParamSrc::Ready2(Box::new(
                pack.tensor::<2>(entry, precision, device).map_err(parse)?,
            )),
            Role::Vector | Role::Conv => ParamSrc::F32 {
                values: pack.read_f32(entry).map_err(parse)?,
                shape: entry.shape.clone(),
            },
        };
        assign_param(&mut loaded.model, &path, src, QuantPolicy::Off, device).map_err(parse)?;
        moved += 1;
    }
    if moved == 0 {
        return Err(parse(format!(
            "the pack holds no tensors for layers {layers:?}"
        )));
    }
    Ok(pack.bytes_read())
}

/// The device a layer of a resident model lives on (its input norm's, which
/// is where `forward` moves the activations for it).
#[must_use]
pub fn layer_device(loaded: &LoadedQwen35, layer: usize) -> Option<Device> {
    loaded
        .model
        .layers
        .get(layer)
        .map(|l| l.input_norm.gamma.val().device())
}

/// The layer index a parameter path belongs to, if any
/// (`model.layers.7.mlp.gate_proj.weight` -> 7).
fn layer_of_path(path: &str) -> Option<usize> {
    path.strip_prefix("model.layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// One layer's FFN restricted to `clusters` of a partitioned pack, at
/// `precision`, in Linear layout.
///
/// The local slab or a remote executor's weights: columns of gate/up and
/// rows of down are sliced straight from the stored bytes (no
/// re-quantization).
///
/// # Errors
///
/// A pack with no FFN partition; a `layer` past it; a cluster index the
/// layer does not have, or an empty cluster set; a gate/up/down entry the
/// pack does not carry; or a slice that does not read.
///
/// # Panics
///
/// When a partition entry stores no precision level at all — a pack
/// invariant the writer enforces.
pub fn load_ffn_clusters(
    pack: &crate::pack::Pack,
    layer: usize,
    clusters: &[usize],
    precision: crate::pack::Precision,
    device: &Device,
) -> Result<crate::nn::ExpertWeights, String> {
    use burn::module::Param;
    let part = pack
        .manifest
        .ffn_partition
        .as_ref()
        .ok_or("pack has no FFN partition")?;
    let spans = part.layers.get(layer).ok_or("layer out of range")?;
    let names = &part.names[layer];
    let ranges: Vec<(usize, usize)> = clusters
        .iter()
        .map(|&c| {
            spans
                .get(c)
                .map(|s| (s.start, s.len))
                .ok_or_else(|| format!("cluster {c} out of range"))
        })
        .collect::<Result<_, _>>()?;
    if ranges.is_empty() {
        return Err(format!("layer {layer}: empty cluster set"));
    }
    let entry = |name: &str| pack.entry(name).ok_or_else(|| format!("missing {name}"));
    let level_of =
        |e: &crate::pack::TensorEntry| stored_precision(e, &|_| precision).expect("stored level");
    let gate = entry(&names[0])?;
    let up = entry(&names[1])?;
    let down = entry(&names[2])?;
    Ok(crate::nn::ExpertWeights {
        gate: Param::from_tensor(pack.tensor_cols(gate, level_of(gate), &ranges, device)?),
        up: Param::from_tensor(pack.tensor_cols(up, level_of(up), &ranges, device)?),
        down: Param::from_tensor(pack.tensor_rows(down, level_of(down), &ranges, device)?),
    })
}

impl LoadedQwen35 {
    /// Attach the remote FFN clusters (one pool row per layer, ragged).
    ///
    /// # Panics
    ///
    /// When the pool does not have exactly one row per trunk layer, or the
    /// checkpoint is Hadamard-folded (remote clusters would split the
    /// down-projection's transform).
    #[must_use]
    pub fn with_ffn_pool(mut self, pool: std::sync::Arc<crate::nn::ExpertPool>) -> Self {
        assert_eq!(
            pool.num_layers(),
            self.config.num_layers,
            "FFN pool must have one row per layer"
        );
        assert!(
            self.config.hadamard.is_none(),
            "a Hadamard-folded checkpoint cannot run with remote FFN clusters"
        );
        self.ffn_pool = Some(pool);
        self
    }

    /// Opt-in cluster skipping at energy threshold `tau` (see `ffn_skip_tau`).
    #[must_use]
    pub const fn with_ffn_skip(mut self, tau: f32) -> Self {
        self.ffn_skip_tau = tau.max(0.0);
        self
    }

    /// Run the FFN clusters as a **working set** under `plan`: each layer
    /// stages what the next needs while it computes, and evicts behind
    /// itself (P9 stage 4). Without a plan every cluster stays permanently
    /// resident, which is the tier design.
    #[must_use]
    pub fn with_ffn_plan(mut self, plan: std::sync::Arc<crate::workingset::Plan>) -> Self {
        self.ffn_plan = Some(plan);
        self
    }
}

/// The level `choose` picks for `entry` when the pack stores it, else the
/// best level it does store (`None` for an entry with no level at all).
fn stored_precision(
    entry: &crate::pack::TensorEntry,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
) -> Option<crate::pack::Precision> {
    let wanted = choose(entry);
    if entry.precisions.contains_key(&wanted) {
        Some(wanted)
    } else {
        entry.precisions.keys().max().copied()
    }
}

/// Partitioned FFN entries → `(layer, proj index)`, for the local-cluster
/// path; empty when no partitioned load was asked for.
fn ffn_partition_index(
    pack: &crate::pack::Pack,
    partitioned: bool,
) -> Result<std::collections::HashMap<&str, (usize, usize)>, String> {
    match (partitioned, &pack.manifest.ffn_partition) {
        (true, Some(part)) => Ok(part
            .names
            .iter()
            .enumerate()
            .flat_map(|(l, n)| {
                n.iter()
                    .enumerate()
                    .map(move |(i, name)| (name.as_str(), (l, i)))
            })
            .collect()),
        (true, None) => Err("partitioned load requested but the pack has no FFN partition".into()),
        (false, _) => Ok(std::collections::HashMap::new()),
    }
}

/// One partitioned FFN entry (`slot` = its `(layer, proj index)`) restricted
/// to that layer's local `clusters`, at the level `choose` picks: columns of
/// gate/up, rows of down.
fn load_local_ffn(
    pack: &crate::pack::Pack,
    entry: &crate::pack::TensorEntry,
    slot: (usize, usize),
    clusters: &[usize],
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
    device: &Device,
) -> Result<Tensor<2>, String> {
    let (layer, proj) = slot;
    let part = pack.manifest.ffn_partition.as_ref().expect("checked above");
    let spans = &part.layers[layer];
    let ranges: Vec<(usize, usize)> = clusters
        .iter()
        .map(|&c| {
            spans
                .get(c)
                .map(|s| (s.start, s.len))
                .ok_or_else(|| format!("layer {layer}: cluster {c} out of range"))
        })
        .collect::<Result<_, _>>()?;
    if ranges.is_empty() {
        return Err(format!(
            "layer {layer}: no local FFN cluster (every layer needs one)"
        ));
    }
    let precision = stored_precision(entry, choose).expect("stored level");
    if proj == 2 {
        pack.tensor_rows(entry, precision, &ranges, device)
    } else {
        pack.tensor_cols(entry, precision, &ranges, device)
    }
}

fn load_from_pack_inner(
    dir: &Path,
    device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
    local: Option<&dyn Fn(usize) -> Vec<usize>>,
    embed_device: Option<&Device>,
) -> Result<LoadedQwen35, ImportError> {
    use crate::pack::{Pack, Role};
    let parse = |reason: String| ImportError::Parse {
        file: dir.to_path_buf(),
        reason,
    };
    let pack = Pack::open(dir).map_err(parse)?;
    let header = pack.header().map_err(parse)?;
    let config = Qwen35Config::from_gguf(&header).map_err(parse)?;
    let untied = pack.entry("output.weight").is_some();
    let trunk = config.num_layers;
    let mut model = build(&config, device, untied);

    // The FFN down-projection's input transform spans the whole intermediate
    // axis; splitting that axis into clusters (partitioned load, remote
    // pools) would transform each piece alone. Refuse rather than run wrong.
    if local.is_some() && config.hadamard.is_some() {
        return Err(parse(
            "a Hadamard-folded checkpoint cannot be loaded partitioned (the down-projection transform spans the whole FFN axis)".into(),
        ));
    }

    let ffn_index = ffn_partition_index(&pack, local.is_some()).map_err(parse)?;
    // Same structured signal as the layered loader: this is the other pack
    // path a chat request can take, and a bar that only moves for one of them
    // is worse than no bar. No print here — this loader has never had one, and
    // the log page's lines are not what the bar reads (see `crate::progress`).
    crate::progress::begin(
        crate::progress::Phase::Loading,
        expected_tensor_count(&config, untied) as u64,
        crate::progress::Unit::Tensors,
    );
    let mut assigned = 0usize;
    for entry in &pack.manifest.tensors {
        let Some(path) = pack_param_path(&entry.name, trunk) else {
            continue; // NextN block members, if a pack kept any
        };
        let src = if let (Some(local), Some(&slot)) = (local, ffn_index.get(entry.name.as_str())) {
            let clusters = local(slot.0);
            let t = load_local_ffn(&pack, entry, slot, &clusters, choose, device).map_err(parse)?;
            ParamSrc::Ready2(Box::new(t))
        } else {
            let precision = stored_precision(entry, choose)
                .ok_or_else(|| parse(format!("'{}' has no stored precision", entry.name)))?;
            match entry.role {
                // The embedding may live somewhere else entirely — see
                // `load_from_pack_partitioned_split`.
                Role::Embedding => ParamSrc::Ready2(Box::new(
                    pack.tensor::<2>(entry, precision, embed_device.unwrap_or(device))
                        .map_err(parse)?,
                )),
                Role::Linear | Role::Expert { .. } => ParamSrc::Ready2(Box::new(
                    pack.tensor::<2>(entry, precision, device).map_err(parse)?,
                )),
                Role::Vector | Role::Conv => ParamSrc::F32 {
                    values: pack.read_f32(entry).map_err(parse)?,
                    shape: entry.shape.clone(),
                },
            }
        };
        assign_param(&mut model, &path, src, QuantPolicy::Off, device).map_err(parse)?;
        assigned += 1;
        crate::progress::advance(assigned as u64, pack.bytes_read());
    }
    let expected = expected_tensor_count(&config, untied);
    if assigned != expected {
        return Err(parse(format!(
            "pack supplied {assigned} trunk tensors, the architecture needs {expected}"
        )));
    }
    Ok(LoadedQwen35 {
        model,
        config,
        tokenizer_config: None,
        ffn_pool: None,
        ffn_skip_tau: 0.0,
        ffn_plan: None,
    })
}

impl CausalLm for LoadedQwen35 {
    type Cache = Vec<Qwen35Kv>;

    fn is_eos(&self, id: u32) -> bool {
        self.config.eos_token_id.contains(id)
    }

    fn new_cache(&self) -> Self::Cache {
        (0..self.config.num_layers)
            .map(|i| {
                if self.config.is_attention(i) {
                    Qwen35Kv::Attn(None)
                } else {
                    Qwen35Kv::Delta(DeltaState::empty())
                }
            })
            .collect()
    }

    fn forward(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Self::Cache,
        device: &Device,
    ) -> Tensor<2> {
        self.forward_impl(new_ids, past, cache, device, true)
            .expect("forward_impl returns logits when need_logits is true")
    }

    fn forward_advance(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Self::Cache,
        device: &Device,
    ) {
        let _ = self.forward_impl(new_ids, past, cache, device, false);
    }
}

impl LoadedQwen35 {
    /// The shared body of `forward` / `forward_advance`: runs the trunk,
    /// and computes the final norm + `lm_head` only when `need_logits` —
    /// a non-final prefill chunk advances every cache without paying the
    /// head projection (~68 ms/call on the 27B's host head).
    /// Look `new_ids` up in the embedding table: `[1, t, hidden]` on
    /// `device`.
    ///
    /// Split out of [`Self::forward_impl`] so a caller can splice non-text
    /// rows into the sequence — image tokens from the vision tower — and
    /// hand the result to [`Self::forward_embeds`]. Text-only decoding goes
    /// through exactly the same code as before.
    ///
    /// # Panics
    ///
    /// If a token id does not fit an `i32`: the gather indices are built as
    /// `i32`, so an id at or above 2^31 has no representation. Every real
    /// vocabulary is orders of magnitude below that, so this is a
    /// corrupt-id guard rather than a workload condition.
    #[must_use]
    pub fn embed(&self, new_ids: &[u32], device: &Device) -> Tensor<3> {
        let t = new_ids.len();
        // The embedding may live on a different device from the rest of the
        // model (it is a gather, so it is often left on the host to keep VRAM
        // for weights that compute — see `load_from_pack_partitioned_split`).
        // A gather needs its indices on the SAME device as the table, so the
        // indices are built there and only the small `[1, t, hidden]` result
        // crosses over.
        let embed_device = self.model.embed_tokens.weight.val().device();
        let ids32: Vec<i32> = new_ids
            .iter()
            .map(|&i| i32::try_from(i).expect("token id fits i32"))
            .collect();
        let input = Tensor::<1, Int>::from_data(
            TensorData::new(ids32, [t]),
            (&embed_device, crate::backend::int_dtype(&embed_device)),
        )
        .reshape([1, t]);
        let _s = crate::prof::scope("embed");
        let e = self.model.embed_tokens.forward(input);
        // A rotated table stores `R e`: restore the primal basis right after
        // the gather, on the table's own device (the constants live there).
        let e = match &self.config.hadamard {
            Some(h) if h.embed_inverse => h.runtime.on(&embed_device).inverse(e),
            _ => e,
        };
        e.to_device(device)
    }

    fn forward_impl(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut [Qwen35Kv],
        device: &Device,
        need_logits: bool,
    ) -> Option<Tensor<2>> {
        let x = self.embed(new_ids, device);
        self.forward_embeds(x, past, cache, device, need_logits)
    }

    /// The trunk, over embeddings that are already in hidden space.
    ///
    /// This is [`Self::forward_impl`] from the embedding onwards; the split
    /// exists so image embeddings can enter the sequence without pretending
    /// to be token ids (there is no id that means "this patch").
    ///
    /// # Panics
    ///
    /// On an empty span (`x` with no positions), a `cache` whose length is
    /// not the layer count, or — an internal invariant — a cache entry whose
    /// kind does not match its layer's block.
    pub fn forward_embeds(
        &self,
        x: Tensor<3>,
        past: usize,
        cache: &mut [Qwen35Kv],
        device: &Device,
        need_logits: bool,
    ) -> Option<Tensor<2>> {
        let t = x.dims()[1];
        assert!(t >= 1, "qwen35 forward: need at least one token");
        assert!(
            cache.len() == self.config.num_layers,
            "qwen35 forward: cache has {} layers, model has {}",
            cache.len(),
            self.config.num_layers
        );
        let cfg = &self.config;
        let _prof_forward = crate::prof::scope("forward");
        let mut x = x;

        let (cos, sin) = rope_tables(t, past, cfg.rope_dim, cfg.rope_theta, device);
        let mask = (t > 1).then(|| causal_mask(t, past, device));

        // Stage attribution lives in `crate::prof`: the scope guards below feed
        // a flame graph (serve: POST /api/chat with {"profile": true}, then
        // GET /api/profile). This forward is synchronous on the flex trunk, so
        // guards never cross an await and wall time attributes cleanly;
        // device-queued work lands wherever the next readback syncs, so read
        // GPU bars as "where the sync happened", not as kernel time.
        let n_layers = self.model.layers.len();
        // Radial-lookahead carry: layer li+1's speculative post-attn h2,
        // produced during layer li's drain window on SCRATCH state, and
        // verified against the exact h2 once li+1 computes it. Verify mode
        // never uses the speculative value — it only measures how good it
        // would have been, which is the data that decides whether a commit
        // mode is ever legal.
        let mut spec_carry: Option<(usize, Tensor<3>)> = None;
        let mut tally = LookaheadTally::default();
        for li in 0..n_layers {
            let layer = &self.model.layers[li];
            // Layers may live on different devices (the dense placement puts
            // as many whole layers on the GPU as VRAM holds, the rest on the
            // host). Moving `x` here is a no-op while the device does not
            // change, so a same-device model pays nothing, and a split model
            // crosses ONCE — where the assignment changes — instead of twice
            // per layer.
            let layer_device = layer.input_norm.gamma.val().device();
            if x.device() != layer_device {
                x = x.to_device(&layer_device);
            }
            let _prof_layer = crate::prof::scope("layer");
            // The rope tables and mask were built once on the entry device;
            // an attention layer elsewhere needs them there too.
            let (cos_l, sin_l, mask_l) = tables_on(&cos, &sin, mask.as_ref(), &layer_device);
            let pos = PositionTables {
                cos: &cos_l,
                sin: &sin_l,
                mask: mask_l.as_ref(),
            };
            // The folded basis's constants on this layer's device (built
            // once per device, shared by every layer there).
            let lh = self.config.hadamard.as_ref().map(|had| LayerHadamard {
                consts: had.runtime.on(&layer_device),
                folds: had.layers[li],
            });
            let h = self.attn_step(li, &x, &pos, &mut cache[li], lh.as_ref());
            {
                let _s = crate::prof::scope("glue.resid1");
                x = x.add(h);
            }
            let h2 = {
                let _s = crate::prof::scope("norm2");
                layer.post_attn_norm.forward(x.clone())
            };
            // The exact h2 exists: score the speculative one from the
            // previous layer's window. Pure measurement — the exact value
            // is what flows onward, unconditionally.
            if let Some((idx, spec_h2)) = spec_carry.take()
                && idx == li
            {
                tally.score(spec_h2, &h2);
            }
            let pending = self.enqueue_remote_ffn(li, &h2, device);
            let (gate, mut ffn) = mlp_forward(layer, &h2, lh.as_ref());
            // RADIAL LOOKAHEAD (MUMMU_LOOKAHEAD=verify): the dGPU is still
            // draining this layer's remote FFN on its worker; the main
            // thread's wait is the window (see `lookahead_trunk`).
            if lookahead_verify() && pending.is_some() && li + 1 < n_layers && ffn.dims()[1] == 1 {
                let spec_pos = PositionTables {
                    cos: &cos,
                    sin: &sin,
                    mask: None,
                };
                let spec_h2 = self.lookahead_trunk(li, &x, &ffn, &spec_pos, cache);
                spec_carry = Some((li + 1, spec_h2));
            }
            if let Some(resolve) = pending {
                ffn = merge_pending(li, &x, ffn, resolve);
            } else if let Some(pool) = &self.ffn_pool {
                ffn = self.remote_ffn_sequential(pool, li, h2, gate, ffn, device);
            }
            if layer_trace() {
                trace_tensor(&format!("ffn_out-{li}"), &ffn);
            }
            {
                let _s = crate::prof::scope("glue.resid2");
                x = x.add(ffn);
            }
            if layer_trace() {
                trace_residual(li, &x);
            }
        }
        tally.report();
        // Advance-only calls stop here: every cache (KV, conv window,
        // recurrent state) is updated; the final norm and head are the
        // only work skipped, and nothing downstream reads them.
        if !need_logits {
            return None;
        }
        Some(self.head_logits(x, t))
    }

    /// Layer `li`'s token mixer over the residual `x`: the input norm, then
    /// gated attention or Gated `DeltaNet` (advancing `kv`), with the block
    /// output traced under `MUMMU_LAYER_TRACE`.
    fn attn_step(
        &self,
        li: usize,
        x: &Tensor<3>,
        pos: &PositionTables<'_>,
        kv: &mut Qwen35Kv,
        lh: Option<&LayerHadamard>,
    ) -> Tensor<3> {
        let layer = &self.model.layers[li];
        let cfg = &self.config;
        let h = {
            let _s = crate::prof::scope("norm1");
            layer.input_norm.forward(x.clone())
        };
        let h = match (&layer.self_attn, &layer.linear_attn, kv) {
            (Some(attn), None, Qwen35Kv::Attn(kv_state)) => {
                let _s = crate::prof::scope("attn.full");
                attn.forward(h, cfg, pos, kv_state, lh)
            }
            (None, Some(delta), Qwen35Kv::Delta(state)) => {
                let _s = crate::prof::scope("attn.delta");
                delta.forward(h, cfg, state, lh)
            }
            _ => unreachable!("qwen35 forward: layer/cache kind mismatch"),
        };
        if layer_trace() {
            trace_tensor(&format!("block_out-{li}"), &h);
        }
        h
    }

    /// ENQUEUE-FIRST: hand the remote FFN to the accelerators before the
    /// local slab runs, so they work while the host does. The profile that
    /// motivated this: the local mlp (0.67 s/token) ran serially in FRONT of
    /// a 1.59 s/token device wait, card idle. Exact-mode only — the skip
    /// path needs host energies up front and keeps the sequential call
    /// ([`Self::remote_ffn_sequential`]). Returns the join for the in-flight
    /// partial, `None` when nothing was enqueued.
    fn enqueue_remote_ffn(
        &self,
        li: usize,
        h2: &Tensor<3>,
        device: &Device,
    ) -> Option<impl FnOnce() -> Option<Tensor<2>> + use<>> {
        let pool = self
            .ffn_pool
            .as_ref()
            .filter(|_| self.ffn_skip_tau <= 0.0)?;
        if let Some(plan) = &self.ffn_plan
            && let Some(sched) = plan.layers.get(li)
        {
            let _s = crate::prof::scope("glue.sched");
            pool.apply_schedule(li, sched, device);
        }
        let [b, tt, hd] = h2.dims();
        let _s = crate::prof::scope("ffn.enqueue");
        if crate::nn::trace_layer() == Some(li) && tt == 1 {
            eprintln!("[tl] enqueue {}", crate::nn::trace_us());
        }
        let pending = pool.run_dense_pending(li, &h2.clone().reshape([b * tt, hd]))?;
        Some(move || pending.resolve())
    }

    /// RADIAL LOOKAHEAD (`MUMMU_LOOKAHEAD=verify`): while the dGPU drains
    /// layer `li`'s remote FFN, run layer `li+1`'s trunk on the known prefix
    /// `a = x + local_ffn` now, on SCRATCH state (tensor clones are
    /// refcounts; burn ops never mutate in place, and the real cache entry
    /// is untouched). The radial identity makes the parallel component of
    /// the late remote piece exact under a scalar; what this measures is how
    /// much the rest matters. Returns layer `li+1`'s speculative post-attn
    /// normed state.
    fn lookahead_trunk(
        &self,
        li: usize,
        x: &Tensor<3>,
        ffn: &Tensor<3>,
        pos: &PositionTables<'_>,
        cache: &[Qwen35Kv],
    ) -> Tensor<3> {
        let _s = crate::prof::scope("spec.trunk");
        let cfg = &self.config;
        let a3 = x.clone().add(ffn.clone());
        let nxt = &self.model.layers[li + 1];
        let mut scratch = snapshot_kv(&cache[li + 1]);
        let sh = nxt.input_norm.forward(a3.clone());
        let lh_next = self.config.hadamard.as_ref().map(|had| LayerHadamard {
            consts: had.runtime.on(&sh.device()),
            folds: had.layers[li + 1],
        });
        let sh = match (&nxt.self_attn, &nxt.linear_attn, &mut scratch) {
            (Some(attn), None, Qwen35Kv::Attn(kv_state)) => {
                attn.forward(sh, cfg, pos, kv_state, lh_next.as_ref())
            }
            (None, Some(delta), Qwen35Kv::Delta(state)) => {
                delta.forward(sh, cfg, state, lh_next.as_ref())
            }
            _ => unreachable!("qwen35 lookahead: layer/cache kind mismatch"),
        };
        let spec_x = a3.add(sh);
        nxt.post_attn_norm.forward(spec_x)
    }

    /// The working-set / skip path for the remote FFN clusters: issue THIS
    /// layer's staging decisions before its FFN runs (so the transfers for
    /// the next layer overlap this layer's compute instead of stalling in
    /// front of it; nothing here blocks — a cluster that has not landed by
    /// the time its layer runs simply computes on the host), then the
    /// remote clusters of the partitioned FFN (exact sum; skip only when a
    /// measured tau was chosen), added into `ffn`.
    fn remote_ffn_sequential(
        &self,
        pool: &crate::nn::ExpertPool,
        li: usize,
        h2: Tensor<3>,
        gate: Tensor<3>,
        ffn: Tensor<3>,
        device: &Device,
    ) -> Tensor<3> {
        if let Some(plan) = &self.ffn_plan
            && let Some(sched) = plan.layers.get(li)
        {
            let _s = crate::prof::scope("glue.sched");
            pool.apply_schedule(li, sched, device);
        }
        let [b, tt, hd] = ffn.dims();
        let local_energy: Vec<f32> = if self.ffn_skip_tau > 0.0 {
            gate.powf_scalar(2.0)
                .sum_dim(2)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .expect("local gate energy")
        } else {
            Vec::new()
        };
        let skip =
            (self.ffn_skip_tau > 0.0).then_some((self.ffn_skip_tau, local_energy.as_slice()));
        let remote = {
            let _s = crate::prof::scope("ffn.remote");
            pool.run_dense(li, h2.reshape([b * tt, hd]), skip)
        };
        if let Some(remote) = remote {
            let _s = crate::prof::scope("glue.merge");
            ffn.add(remote.reshape([b, tt, hd]))
        } else {
            ffn
        }
    }

    /// The final norm and head over the last of `t` positions of `x` (the
    /// norm and head may live elsewhere than the last layer): logits
    /// `[1, vocab]`.
    fn head_logits(&self, x: Tensor<3>, t: usize) -> Tensor<2> {
        let cfg = &self.config;
        let head_device = self.model.norm.gamma.val().device();
        let x = if x.device() == head_device {
            x
        } else {
            x.to_device(&head_device)
        };
        let x = {
            let _s = crate::prof::scope("final_norm");
            self.model.norm.forward(x)
        };

        // Last position only → logits [1, vocab].
        let last = x.narrow(1, t - 1, 1).reshape([1, cfg.hidden_size]);
        // A folded head (or a tied head over the rotated table) reads the
        // transformed hidden state.
        let last = match &self.config.hadamard {
            Some(had) if had.head => had.runtime.on(&head_device).forward(last),
            _ => last,
        };
        // Suspect number one for unattributed time: the tied head is a
        // [1, 5120] x [5120, 248320] f32 matmul, and it runs on whichever
        // device holds the embedding table — the HOST, for a split model.
        let _s = crate::prof::scope("lm_head");
        if let Some(head) = &self.model.lm_head {
            // The bounded-exact host head (SPEC P4.3/P4.4) engages when
            // serve opted in and the weight is packed on flex; every
            // consulted coordinate equals the dense head's value (see
            // `flex::head`), and the skipped rows never stream.
            crate::nn::try_q4s_head(&last, &head.weight.val())
                .unwrap_or_else(|| qlinear2(head, last))
        } else {
            // Tied head: logits = h · Eᵀ. The embedding may be on another
            // device (host-resident gather table), and unlike the gather
            // this IS a matmul — so run it where the big tensor lives and
            // move only the `[1, vocab]` result, rather than dragging a
            // multi-GB table across the bus every token.
            let table = self.model.embed_tokens.weight.val(); // [vocab, hidden]
            let out_device = last.device();
            let table_device = table.device();
            last.to_device(&table_device)
                .matmul(table.swap_dims(0, 1))
                .to_device(&out_device)
        }
    }
}

/// The lookahead-verify tally (`MUMMU_LOOKAHEAD=verify`): how many layers'
/// speculative post-attention states were scored, the worst relative
/// error, and how many landed under each acceptance threshold.
#[derive(Default)]
struct LookaheadTally {
    scored: u32,
    worst: f32,
    within_1e3: u32,
    within_1e2: u32,
    within_5e2: u32,
}

impl LookaheadTally {
    /// Score one speculative `spec_h2` against the exact `h2`.
    fn score(&mut self, spec_h2: Tensor<3>, h2: &Tensor<3>) {
        let _s = crate::prof::scope("spec.verify");
        let read = |t: Tensor<3>| -> f32 {
            t.abs()
                .max()
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .map_or(f32::NAN, |v| v[0])
        };
        let diff = read(spec_h2.sub(h2.clone()));
        let scale = read(h2.clone()).max(1e-6);
        let rel = diff / scale;
        self.scored += 1;
        self.worst = self.worst.max(rel);
        if rel < 1e-3 {
            self.within_1e3 += 1;
        }
        if rel < 1e-2 {
            self.within_1e2 += 1;
        }
        if rel < 5e-2 {
            self.within_5e2 += 1;
        }
    }

    /// Print the tally, if anything was scored.
    fn report(&self) {
        if self.scored == 0 {
            return;
        }
        let pct = |k: u32| f64::from(k) * 100.0 / f64::from(self.scored);
        eprintln!(
            "[lookahead] verified {} layers: accept@1e-3 {:.0}% | @1e-2 {:.0}% | @5e-2 {:.0}% | worst rel {:.4}",
            self.scored,
            pct(self.within_1e3),
            pct(self.within_1e2),
            pct(self.within_5e2),
            self.worst,
        );
    }
}

/// The rope tables and mask, built once on the entry device, as a layer on
/// `device` needs them (a no-op for the entry device itself).
fn tables_on(
    cos: &Tensor<4>,
    sin: &Tensor<4>,
    mask: Option<&Tensor<4>>,
    device: &Device,
) -> (Tensor<4>, Tensor<4>, Option<Tensor<4>>) {
    let _s = crate::prof::scope("glue.rope");
    let (cos_l, sin_l) = if cos.device() == *device {
        (cos.clone(), sin.clone())
    } else {
        (cos.clone().to_device(device), sin.clone().to_device(device))
    };
    let mask_l = mask.map(|m| {
        if m.device() == *device {
            m.clone()
        } else {
            m.clone().to_device(device)
        }
    });
    (cos_l, sin_l, mask_l)
}

/// The layer's `SwiGLU` spelled through `qlinear` (the mlp's own forward
/// would reshape a packed quantized weight — see `qlinear`). Three separate
/// scopes: gate/up multiply `[1,h]x[h,inter]` while down multiplies
/// `[1,inter]x[inter,h]` — if one shape hits a slow kernel path, the graph
/// should say which. Returns `(silu(gate), down(silu(gate) ⊙ up))`; the
/// gate is kept for the skip path's energies.
fn mlp_forward(
    layer: &Qwen35Layer,
    h2: &Tensor<3>,
    lh: Option<&LayerHadamard>,
) -> (Tensor<3>, Tensor<3>) {
    let folds = lh.map_or_else(LayerFolds::default, |had| had.folds);
    let h2r = lh
        .filter(|_| folds.has(Fold::Gate) || folds.has(Fold::Up))
        .map(|had| had.consts.forward(h2.clone()));
    let gate = {
        let _s = crate::prof::scope("mlp.gate");
        activation::silu(qlinear(
            &layer.mlp.gate_proj,
            pick(folds.has(Fold::Gate), h2r.as_ref(), h2),
        ))
    };
    let up = {
        let _s = crate::prof::scope("mlp.up");
        qlinear(
            &layer.mlp.up_proj,
            pick(folds.has(Fold::Up), h2r.as_ref(), h2),
        )
    };
    drop(h2r);
    let ffn = {
        let _s = crate::prof::scope("mlp.down");
        let act = gate.clone().mul(up);
        let act = match lh {
            Some(had) if folds.has(Fold::Down) => had.consts.forward(act),
            _ => act,
        };
        qlinear(&layer.mlp.down_proj, act)
    };
    (gate, ffn)
}

/// Join the in-flight remote FFN partial (`resolve`, from
/// [`LoadedQwen35::enqueue_remote_ffn`]) and add it into the local `ffn`;
/// a partial that never materialized leaves `ffn` as it is.
fn merge_pending(
    li: usize,
    x: &Tensor<3>,
    ffn: Tensor<3>,
    resolve: impl FnOnce() -> Option<Tensor<2>>,
) -> Tensor<3> {
    let _s = crate::prof::scope("glue.merge");
    let tl = crate::nn::trace_layer() == Some(li) && ffn.dims()[1] == 1;
    if tl {
        eprintln!("[tl] join-start {}", crate::nn::trace_us());
    }
    let resolved = {
        let _s = crate::prof::scope("merge.resolve_call");
        resolve()
    };
    if tl {
        eprintln!("[tl] join-done {}", crate::nn::trace_us());
    }
    match resolved {
        Some(remote) => merge_remote(li, x, ffn, remote),
        None => ffn,
    }
}

/// Add the resolved remote FFN partial `remote` `[b·t, hidden]` into the
/// local `ffn`, with the residual probe and the first-op timing probes
/// around it.
fn merge_remote(li: usize, x: &Tensor<3>, ffn: Tensor<3>, remote: Tensor<2>) -> Tensor<3> {
    let [b, tt, hd] = ffn.dims();
    let remote = {
        let _s = crate::prof::scope("merge.reshape");
        remote.reshape([b, tt, hd])
    };
    if residual_probe() && tt == 1 {
        let a = x.clone().add(ffn.clone());
        residual_probe_report(li, a, &remote);
    }
    // Which operand carries the ~28 ms that lands on the FIRST op after
    // the worker cycle? Three scoped probes: an op touching only the
    // local ffn (ambient/first-op effects), an op touching only the
    // remote partial (its first-use materialization), then the real add.
    // The 28ms lands in exactly one of these and names its class.
    {
        let _s = crate::prof::scope("merge.warm_local");
        let _ = ffn.clone().add_scalar(0.0);
    }
    {
        let _s = crate::prof::scope("merge.warm_remote");
        let _ = remote.clone().add_scalar(0.0);
    }
    let _s = crate::prof::scope("merge.add");
    ffn.add(remote)
}

/// Residual-geometry probe (`MUMMU_RESIDUAL_PROBE=1`): the radial split
/// `N(a+b) = alpha*N(a) + rstd(a+b)*(g.*b_perp)` is exact, so lookahead's
/// commit-mode viability is the size of `b_perp` against `a` — measured,
/// not argued. `a` is the local prefix `x + ffn`, `remote` the late piece.
fn residual_probe_report(li: usize, a: Tensor<3>, remote: &Tensor<3>) {
    let read = |t: Tensor<3>| -> f32 {
        t.sum()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_or(f32::NAN, |v| v[0])
    };
    let dot = read(a.clone().mul(remote.clone()));
    let na2 = read(a.clone().mul(a));
    let nb2 = read(remote.clone().mul(remote.clone()));
    if na2 > 0.0 {
        let sigma = 1.0 + dot / na2;
        let bperp2 = (nb2 - dot * dot / na2).max(0.0);
        // Separate multiply and add (no mul_add): the same arithmetic as
        // the split it measures.
        let twice_dot = 2.0 * dot;
        let alpha = sigma * (na2 / (na2 + nb2 + twice_dot).max(1e-12)).sqrt();
        eprintln!(
            "[residual-probe] layer={li} b_over_a={:.4} bperp_over_a={:.4} alpha_minus_1={:+.5}",
            (nb2 / na2).sqrt(),
            (bperp2 / na2).sqrt(),
            alpha - 1.0,
        );
    }
}

/// `MUMMU_LAYER_TRACE`: the residual after layer `li`.
fn trace_residual(li: usize, x: &Tensor<3>) {
    let v = x
        .clone()
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap_or_default();
    let sum: f64 = v.iter().map(|&a| f64::from(a)).sum();
    eprintln!(
        "[layer-trace] l_out-{li}: sum={sum:.6} first={:?}",
        &v[..v.len().min(4)]
    );
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;

    /// A toy config exercising both layer kinds: layer 1 is full attention
    /// (`(1+1) % 2 == 0`), layers 0 and 2 are `DeltaNet`.
    fn toy_config() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 64,
            hidden_size: 16,
            num_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            intermediate_size: 24,
            rms_norm_eps: 1e-6,
            rope_theta: 1e4,
            rope_dim: 4,
            full_attention_interval: 2,
            conv_kernel: 3,
            d_inner: 12, // 3 v-heads × d_state 4
            d_state: 4,
            n_k_heads: 1,
            n_v_heads: 3,
            gdn_gate: GdnGate::Silu,
            gdn_l2: GdnL2::ClampNorm,
            eos_token_id: EosIds::One(0),
            hadamard: None,
        }
    }

    fn toy_model() -> LoadedQwen35 {
        let cfg = toy_config();
        cfg.validate().expect("toy config validates");
        let device = crate::backend::cpu_device();
        LoadedQwen35 {
            model: build(&cfg, &device, false),
            config: cfg,
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        }
    }

    /// The load-bearing invariant for BOTH caches (attention KV and the
    /// `DeltaNet` conv window + recurrent state): prefill + one-token decode
    /// steps must produce exactly the logits of a single full prefill.
    #[test]
    fn cached_decode_matches_full_prefill() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let ids: Vec<u32> = vec![3, 17, 42, 9, 60, 11];

        let mut full_cache = m.new_cache();
        let full = m
            .forward(&ids, 0, &mut full_cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let mut cache = m.new_cache();
        let _ = m.forward(&ids[..3], 0, &mut cache, &device);
        let mut last = Vec::new();
        for (i, &id) in ids.iter().enumerate().skip(3) {
            last = m
                .forward(&[id], i, &mut cache, &device)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
        }
        assert_eq!(full.len(), last.len());
        for (i, (f, s)) in full.iter().zip(&last).enumerate() {
            assert!((f - s).abs() < 1e-4, "logit {i}: full {f} vs stepped {s}");
        }
    }

    /// `DeltaNet` state actually carries information: the same final token
    /// after different prefixes must produce different logits.
    #[test]
    fn recurrent_state_carries_the_prefix() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut c1 = m.new_cache();
        let mut c2 = m.new_cache();
        let _ = m.forward(&[1, 2, 3], 0, &mut c1, &device);
        let _ = m.forward(&[9, 8, 7], 0, &mut c2, &device);
        let a = m
            .forward(&[5], 3, &mut c1, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let b = m
            .forward(&[5], 3, &mut c2, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-6, "different prefixes must change the logits");
    }

    /// Fold a `[in, out]` linear weight the way Prism's converter folds it:
    /// every output column `w_o` becomes `R w_o` (through the grouped-head
    /// permutation first when `grouped`), so that `⟨w'_o, R x⟩ = ⟨w_o, x⟩`.
    fn fold_linear(lin: &mut Linear, spec: &HadamardSpec, grouped: Option<(usize, usize, usize)>) {
        let w = lin.weight.val();
        let device = w.device();
        let [inp, out] = w.dims();
        let mut v = w.into_data().try_to_vec::<f32>().expect("f32 weights");
        for o in 0..out {
            let mut col: Vec<f32> = (0..inp).map(|i| v[i * out + o]).collect();
            if let Some((nk, nv, hd)) = grouped {
                col = crate::nn::hadamard::tiled_to_grouped_host(&col, nk, nv, hd);
            }
            spec.forward_host(&mut col).expect("width cuts into blocks");
            for (i, c) in col.into_iter().enumerate() {
                v[i * out + o] = c;
            }
        }
        lin.weight = Param::from_tensor(Tensor::from_data(TensorData::new(v, [inp, out]), &device));
    }

    /// The toy fold contract: every folded projection of every layer except
    /// `unfolded`, the untied head, and the token table as a rotated
    /// lookup; block 4 divides every folded width (16, 24, 16).
    fn toy_fold_spec(cfg: &Qwen35Config, unfolded: &[&str]) -> HadamardSpec {
        use std::collections::{BTreeMap, BTreeSet};
        let block = 4;
        let mut signs = BTreeMap::new();
        for w in [cfg.hidden_size, cfg.d_inner, cfg.intermediate_size] {
            signs.insert(w, mummu_mix::hadamard::sign_diagonal(w as u64 * 11 + 3, w));
        }
        let mut weights = BTreeSet::new();
        weights.insert("output.weight".to_string());
        for l in 0..cfg.num_layers {
            let fields: Vec<&str> = if cfg.is_attention(l) {
                vec!["attn_q", "attn_k", "attn_v", "attn_output"]
            } else {
                vec!["attn_qkv", "attn_gate", "ssm_out"]
            };
            for f in fields.into_iter().chain(["ffn_gate", "ffn_up", "ffn_down"]) {
                let name = format!("blk.{l}.{f}.weight");
                if !unfolded.contains(&name.as_str()) {
                    weights.insert(name);
                }
            }
        }
        HadamardSpec {
            block,
            signs,
            weights,
            inverses: std::iter::once("token_embd.weight".to_string()).collect(),
            gdn_v_grouped: true,
        }
    }

    /// Fold a model's weights in place the way Prism's converter folds
    /// them: every weight `spec` names through [`fold_linear`] (`ssm_out`
    /// through the grouped-head permutation), and the token table stored
    /// as `R e_v`.
    fn fold_model(model: &mut Qwen35, cfg: &Qwen35Config, spec: &HadamardSpec, device: &Device) {
        let grouped = Some((cfg.n_k_heads, cfg.n_v_heads, cfg.d_state));
        for (l, layer) in model.layers.iter_mut().enumerate() {
            let on = |f: &str| spec.folds(&format!("blk.{l}.{f}.weight"));
            if let Some(a) = layer.self_attn.as_mut() {
                for (f, lin) in [
                    ("attn_q", &mut a.q_proj),
                    ("attn_k", &mut a.k_proj),
                    ("attn_v", &mut a.v_proj),
                    ("attn_output", &mut a.o_proj),
                ] {
                    if on(f) {
                        fold_linear(lin, spec, None);
                    }
                }
            }
            if let Some(d) = layer.linear_attn.as_mut() {
                if on("attn_qkv") {
                    fold_linear(&mut d.qkv_proj, spec, None);
                }
                if on("attn_gate") {
                    fold_linear(&mut d.z_proj, spec, None);
                }
                if on("ssm_out") {
                    fold_linear(&mut d.out_proj, spec, grouped);
                }
            }
            for (f, lin) in [
                ("ffn_gate", &mut layer.mlp.gate_proj),
                ("ffn_up", &mut layer.mlp.up_proj),
                ("ffn_down", &mut layer.mlp.down_proj),
            ] {
                if on(f) {
                    fold_linear(lin, spec, None);
                }
            }
        }
        fold_linear(model.lm_head.as_mut().expect("untied head"), spec, None);
        let e = model.embed_tokens.weight.val();
        let [vocab, hidden] = e.dims();
        let mut v = e.into_data().try_to_vec::<f32>().expect("f32 table");
        for row in v.chunks_mut(hidden) {
            spec.forward_host(row).expect("hidden cuts into blocks");
        }
        model.embed_tokens.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(v, [vocab, hidden]),
            device,
        ));
    }

    /// A 4-token prefill and then single tokens (which on flex take the
    /// fused host decode step): the logits of every step.
    fn step_logits(m: &LoadedQwen35, ids: &[u32], device: &Device) -> Vec<Vec<f32>> {
        let mut cache = m.new_cache();
        let mut out = vec![
            m.forward(&ids[..4], 0, &mut cache, device)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
        ];
        for (i, &id) in ids.iter().enumerate().skip(4) {
            out.push(
                m.forward(&[id], i, &mut cache, device)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            );
        }
        out
    }

    /// A checkpoint folded the way Prism's converter folds it must give the
    /// unfolded model's logits: every folded weight's input axis rotated by
    /// `R = H·S` per block (signs, then the transform), `ssm_out` through
    /// the grouped-head permutation, the token table stored as `R e_v`, the
    /// head folded — then the contract declared and the two compared. This
    /// pins the sign order, the inverse-after-gather, the head, and the
    /// tiled→grouped permutation, on the tensor path AND the fused host
    /// decode step (single tokens on flex take it). Two projections are
    /// deliberately left unfolded: the contract is per weight.
    #[test]
    fn hadamard_folded_weights_reproduce_the_unfolded_logits() {
        let _serial = FUSED_TOGGLE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = crate::backend::cpu_device();
        // Two key heads tiled over six value heads, so the permutation moves
        // something; block 4 divides every folded width (16, 24, 16).
        let cfg = Qwen35Config {
            d_inner: 24,
            n_k_heads: 3,
            n_v_heads: 6,
            ..toy_config()
        };
        cfg.validate().expect("toy config validates");
        let plain = LoadedQwen35 {
            model: build(&cfg, &device, true),
            config: cfg.clone(),
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        };
        let spec = toy_fold_spec(&cfg, &["blk.1.attn_k.weight", "blk.2.attn_gate.weight"]);

        // Fold a copy of the weights.
        let mut model = plain.model.clone();
        fold_model(&mut model, &cfg, &spec, &device);
        let width_of = |name: &str| -> Option<usize> {
            Some(match name.rsplit('.').nth(1)? {
                "ssm_out" => cfg.d_inner,
                "ffn_down" => cfg.intermediate_size,
                "attn_output" => cfg.num_attention_heads * cfg.head_dim,
                _ => cfg.hidden_size,
            })
        };
        let mut cfg_folded = cfg.clone();
        cfg_folded.hadamard = Some(std::sync::Arc::new(
            Qwen35Hadamard::from_spec(spec, &cfg, true, &width_of).expect("contract resolves"),
        ));
        let had = cfg_folded.hadamard.as_ref().unwrap();
        assert!(had.head && had.embed_inverse);
        assert!(
            had.layers[1].has(Fold::Q) && !had.layers[1].has(Fold::K) && had.layers[1].has(Fold::V)
        );
        assert!(
            had.layers[2].has(Fold::Qkv)
                && !had.layers[2].has(Fold::Z)
                && had.layers[2].has(Fold::Out)
        );
        let folded = LoadedQwen35 {
            model,
            config: cfg_folded,
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        };

        let ids: Vec<u32> = vec![3, 17, 42, 9, 60, 11];
        let (a, b) = (
            step_logits(&plain, &ids, &device),
            step_logits(&folded, &ids, &device),
        );
        for (step, (pa, pb)) in a.iter().zip(&b).enumerate() {
            let worst = pa
                .iter()
                .zip(pb)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let scale = pa.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            assert!(
                worst / scale < 1e-4,
                "step {step}: folded logits differ from plain by {worst} (scale {scale})"
            );
        }
        // And the fold is not a no-op: the folded weights run WITHOUT the
        // contract would be wrong.
        let bare = LoadedQwen35 {
            model: folded.model,
            config: cfg,
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        };
        let c = step_logits(&bare, &ids, &device);
        let worst = a[0]
            .iter()
            .zip(&c[0])
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(worst > 1e-3, "the rotation must matter: {worst}");
    }

    /// Max |a − b| across two same-shape tensors, read back on the host.
    fn max_abs_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
        a.sub(b)
            .abs()
            .max()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap()[0]
    }

    /// Random recurrence inputs shaped like the post-conv/norm/tile tensors:
    /// q/k/v `[b, hv, t, ds]`, decay logits g ≤ 0 (softplus·a with a < 0
    /// guarantees that in the real model), β ∈ (0, 1) (a sigmoid). q and k
    /// are L2-normalized per position like the model's are — unit ‖k‖ is
    /// what keeps the delta-rule map `γ(I − βkkᵀ)` contractive, so long
    /// test sequences stay O(1) instead of drifting past an abs tolerance.
    fn random_recurrence_inputs(
        t: usize,
        g_range: (f64, f64),
        device: &Device,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<3>, Tensor<3>) {
        let (b, hv, ds) = (1, 3, 4);
        let dims = [b, hv, t, ds];
        let uni = Distribution::Uniform(-1.0, 1.0);
        let l2 = |x: Tensor<4>| {
            let norm = x.clone().powi_scalar(2).sum_dim(3).sqrt().clamp_min(1e-6);
            x.div(norm)
        };
        (
            l2(Tensor::<4>::random(dims, uni, device)),
            l2(Tensor::<4>::random(dims, uni, device)),
            Tensor::<4>::random(dims, uni, device),
            Tensor::<3>::random(
                [b, t, hv],
                Distribution::Uniform(g_range.0, g_range.1),
                device,
            ),
            Tensor::<3>::random([b, t, hv], Distribution::Uniform(0.05, 0.95), device),
        )
    }

    /// The chunked evaluation is exact: identical inputs and initial state
    /// must reproduce the sequential recurrence's outputs AND final state.
    /// Lengths cross the chunk boundaries of C = 64 (5 < C; 64 = exactly
    /// one chunk; 100 = one full + one partial; 129 = two full + a final
    /// chunk of ONE token), each from both a zero and a random carried
    /// state (`None` vs `Some(S)` in the cache).
    /// The five recurrence inputs as the borrowed bundle the kernels take.
    fn inputs_of(
        (q, k, v, g, beta): &(Tensor<4>, Tensor<4>, Tensor<4>, Tensor<3>, Tensor<3>),
    ) -> RecurrenceInputs<'_> {
        RecurrenceInputs { q, k, v, g, beta }
    }

    #[test]
    fn chunked_recurrence_matches_sequential() {
        let device = crate::backend::cpu_device();
        let (batch, hv, ds) = (1, 3, 4);
        let scale = 1.0 / f32_from_usize(ds).sqrt();
        for &len in &[5usize, 64, 100, 129] {
            for random_state in [false, true] {
                let inputs = random_recurrence_inputs(len, (-1.0, 0.0), &device);
                let inputs = inputs_of(&inputs);
                let s0 = if random_state {
                    Tensor::<4>::random(
                        [batch, hv, ds, ds],
                        Distribution::Uniform(-1.0, 1.0),
                        &device,
                    )
                } else {
                    Tensor::<4>::zeros([batch, hv, ds, ds], &device)
                };
                let (o_seq, s_seq) = gdn_recurrence_sequential(&inputs, s0.clone(), scale);
                let (o_chk, s_chk) = gdn_recurrence_chunked(&inputs, s0, scale, 64);
                let od = max_abs_diff(o_seq, o_chk);
                let sd = max_abs_diff(s_seq, s_chk);
                assert!(
                    od < 1e-4,
                    "outputs diverge at t={len} (random_state={random_state}): {od}"
                );
                assert!(
                    sd < 1e-4,
                    "final state diverges at t={len} (random_state={random_state}): {sd}"
                );
            }
        }
    }

    /// γ-underflow stress: decay near 0.5/step over t = 128 puts the raw
    /// cumulative product at ~0.5¹²⁸ ≈ 3e-39 — below f32's smallest
    /// normal — so any `P_t/P_j` formed as a ratio of products dies. The
    /// exp-of-cumsum-difference form must keep the chunked path on top of
    /// the sequential reference anyway.
    #[test]
    fn chunked_recurrence_survives_gamma_underflow() {
        let device = crate::backend::cpu_device();
        let (batch, hv, ds) = (1, 3, 4);
        let scale = 1.0 / f32_from_usize(ds).sqrt();
        let inputs = random_recurrence_inputs(128, (-0.8, -0.6), &device);
        let inputs = inputs_of(&inputs);
        let s0 = Tensor::<4>::random(
            [batch, hv, ds, ds],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let (o_seq, s_seq) = gdn_recurrence_sequential(&inputs, s0.clone(), scale);
        let (o_chk, s_chk) = gdn_recurrence_chunked(&inputs, s0, scale, 64);
        let od = max_abs_diff(o_seq, o_chk);
        let sd = max_abs_diff(s_seq, s_chk);
        assert!(od < 1e-4, "outputs diverge under strong decay: {od}");
        assert!(sd < 1e-4, "final state diverges under strong decay: {sd}");
    }

    /// Repeated-key stress, the failure the real Flash-Next weights hit on a
    /// repetitive 285-token prompt (layer 45 went NaN at token 192, the
    /// first token of a later full chunk): with (nearly) parallel keys,
    /// β near 1 and almost no decay, `A[t, j] ≈ 1` below the diagonal. The
    /// true `(I + A)⁻¹` stays bounded (its entries are delta-rule
    /// sensitivities, at most 1), but its power-series terms `N^k` are
    /// binomials up to C(62, 31) ≈ 5e17 at C = 64, so any evaluation that
    /// forms them cancels catastrophically in f32. The chunked path must
    /// still track the sequential reference.
    #[test]
    fn chunked_recurrence_survives_repeated_keys() {
        let device = crate::backend::cpu_device();
        let (batch, hv, ds, len) = (1, 3, 4, 192);
        let scale = 1.0 / f32_from_usize(ds).sqrt();
        let (query, _, values, _, _) = random_recurrence_inputs(len, (-1.0, 0.0), &device);
        // One unit key per head, repeated at every position, with a tiny
        // per-token wobble so the keys are parallel but not bit-identical.
        let base = Tensor::<4>::random(
            [batch, hv, 1, ds],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        )
        .repeat_dim(2, len)
        .add(Tensor::<4>::random(
            [batch, hv, len, ds],
            Distribution::Uniform(-1e-3, 1e-3),
            &device,
        ));
        let keys = base
            .clone()
            .div(base.powi_scalar(2).sum_dim(3).sqrt().clamp_min(1e-6));
        let decay =
            Tensor::<3>::random([batch, len, hv], Distribution::Uniform(-1e-3, 0.0), &device);
        let beta = Tensor::<3>::random([batch, len, hv], Distribution::Uniform(0.9, 0.99), &device);
        let inputs = RecurrenceInputs {
            q: &query,
            k: &keys,
            v: &values,
            g: &decay,
            beta: &beta,
        };
        let s0 = Tensor::<4>::zeros([batch, hv, ds, ds], &device);
        let (o_seq, s_seq) = gdn_recurrence_sequential(&inputs, s0.clone(), scale);
        let (o_chk, s_chk) = gdn_recurrence_chunked(&inputs, s0, scale, 64);
        let od = max_abs_diff(o_seq, o_chk);
        let sd = max_abs_diff(s_seq, s_chk);
        assert!(od < 1e-3, "outputs diverge under repeated keys: {od}");
        assert!(sd < 1e-3, "final state diverges under repeated keys: {sd}");
    }

    /// The cache invariant again, with the prefill long enough (100 > the
    /// default chunk of 64, and `MUMMU_GDN_CHUNK` is unset under test) that
    /// it runs the CHUNKED path: stepped decode after it must still equal
    /// one full forward. Together with
    /// `chunked_recurrence_matches_sequential` this pins the chunked
    /// prefill → sequential decode handoff (conv window + carried state).
    #[test]
    fn cached_decode_matches_chunked_prefill() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let ids: Vec<u32> = (0..104u32).map(|i| (i * 37 + 11) % 64).collect();

        let mut full_cache = m.new_cache();
        let full = m
            .forward(&ids, 0, &mut full_cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let mut cache = m.new_cache();
        let _ = m.forward(&ids[..100], 0, &mut cache, &device);
        let mut last = Vec::new();
        for (i, &id) in ids.iter().enumerate().skip(100) {
            last = m
                .forward(&[id], i, &mut cache, &device)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
        }
        assert_eq!(full.len(), last.len());
        for (i, (f, s)) in full.iter().zip(&last).enumerate() {
            assert!((f - s).abs() < 1e-4, "logit {i}: full {f} vs stepped {s}");
        }
    }

    #[test]
    fn config_layer_kinds_follow_the_interval() {
        let cfg = toy_config();
        assert!(!cfg.is_attention(0));
        assert!(cfg.is_attention(1));
        assert!(!cfg.is_attention(2));
        // The 27B pattern: every 4th layer.
        let mut real = cfg;
        real.full_attention_interval = 4;
        let attn: Vec<usize> = (0..8).filter(|&i| real.is_attention(i)).collect();
        assert_eq!(attn, vec![3, 7]);
    }

    /// The skip-head advance (non-final prefill chunks) updates every
    /// cache exactly like a full forward: advancing over a prefix and then
    /// stepping must equal the one-shot forward.
    #[test]
    fn forward_advance_then_forward_matches_full() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let ids: Vec<u32> = vec![3, 17, 42, 9, 60, 11];

        let mut c1 = m.new_cache();
        let full = m
            .forward(&ids, 0, &mut c1, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let mut c2 = m.new_cache();
        m.forward_advance(&ids[..4], 0, &mut c2, &device);
        let stepped = m
            .forward(&ids[4..], 4, &mut c2, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (i, (f, s)) in full.iter().zip(&stepped).enumerate() {
            assert!((f - s).abs() < 1e-4, "logit {i}: full {f} vs advanced {s}");
        }
    }

    /// Serializes the tests that toggle the fused-GDN force switch — it is
    /// process-global, and the tolerant equality tests above must not have
    /// the path flipped underneath a single run.
    static FUSED_TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Re-enables the fused path on drop so a failing assert cannot leave
    /// the process degraded for other tests.
    struct RestoreFused;
    impl Drop for RestoreFused {
        fn drop(&mut self) {
            crate::flex::gdn::force_disable(false);
        }
    }

    /// The fused host decode step (SPEC P3) against the tensor path it
    /// replaces: same prefix, same decode tokens, logits equal to fold
    /// order. This is the oracle for the whole fused middle — conv ring,
    /// L2 norms, gates, two-pass recurrence, gated `RMSNorm`.
    #[test]
    fn fused_gdn_decode_matches_tensor_decode() {
        let _serial = FUSED_TOGGLE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let m = toy_model();
        let tensor = oracle_decode_steps(&m, false);
        let fused = oracle_decode_steps(&m, true);
        assert_steps_close(&tensor, &fused);
    }

    /// The oracle's token script: a 3-token tensor-path prefill, then four
    /// single-token decode steps carrying the conv ring and the recurrent
    /// state from one step to the next.
    const ORACLE_PREFIX: [u32; 3] = [3, 17, 42];
    const ORACLE_DECODE: [u32; 4] = [9, 60, 11, 5];

    /// Every decode step's logits for the oracle script, with the fused
    /// host step on (`fused`) or forced off. Callers hold
    /// `FUSED_TOGGLE_LOCK`.
    fn oracle_decode_steps(m: &LoadedQwen35, fused: bool) -> Vec<Vec<f32>> {
        let device = crate::backend::cpu_device();
        crate::flex::gdn::force_disable(!fused);
        let _restore = RestoreFused;
        let mut cache = m.new_cache();
        let _ = m.forward(&ORACLE_PREFIX, 0, &mut cache, &device);
        ORACLE_DECODE
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                m.forward(&[id], ORACLE_PREFIX.len() + i, &mut cache, &device)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
            })
            .collect()
    }

    fn assert_steps_close(tensor: &[Vec<f32>], fused: &[Vec<f32>]) {
        assert_eq!(tensor.len(), fused.len());
        for (step, (a, b)) in tensor.iter().zip(fused).enumerate() {
            assert_eq!(a.len(), b.len());
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() < 1e-4,
                    "decode step {step} logit {i}: tensor {x} vs fused {y}"
                );
            }
        }
    }

    /// The same oracle under qwen4exp's gate, `sigmoid(z)` in the gated
    /// `RMSNorm`: the fused host step and the tensor path must agree at every
    /// carried-state decode step. Agreement alone would also hold if both
    /// paths ignored `gdn_gate`, so the gate must visibly reach them — the
    /// sigmoid model's logits differ from the silu logits of the SAME
    /// weights — and a one-shot prefill of the whole script (tensor path;
    /// the chunked recurrence under the default env, t = 7 being above the
    /// sequential ceiling of 4) must land on the fused path's last decode
    /// step.
    #[test]
    fn fused_gdn_decode_matches_tensor_decode_with_sigmoid_gate() {
        let _serial = FUSED_TOGGLE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let silu = toy_model();
        let sigmoid = LoadedQwen35 {
            model: silu.model.clone(),
            config: Qwen35Config {
                gdn_gate: GdnGate::Sigmoid,
                ..silu.config.clone()
            },
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        };

        let tensor = oracle_decode_steps(&sigmoid, false);
        let fused = oracle_decode_steps(&sigmoid, true);
        assert_steps_close(&tensor, &fused);

        let silu_tensor = oracle_decode_steps(&silu, false);
        let gate_effect = tensor
            .iter()
            .flatten()
            .zip(silu_tensor.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            gate_effect > 1e-3,
            "sigmoid and silu gates gave the same logits (max diff {gate_effect}): \
             gdn_gate is not reaching the DeltaNet"
        );

        let device = crate::backend::cpu_device();
        let script: Vec<u32> = ORACLE_PREFIX
            .iter()
            .chain(&ORACLE_DECODE)
            .copied()
            .collect();
        let mut cache = sigmoid.new_cache();
        let one_shot = sigmoid
            .forward(&script, 0, &mut cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let last = fused.last().expect("decode steps ran");
        assert_eq!(one_shot.len(), last.len());
        for (i, (x, y)) in one_shot.iter().zip(last).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "logit {i}: one-shot prefill {x} vs fused decode {y}"
            );
        }
    }

    /// The oracle under qwen4exp's q/k L2 form, `x / sqrt(‖x‖² + ε)`: the
    /// fused host step and the tensor path must agree at every carried-state
    /// decode step, and a one-shot prefill (chunked recurrence) must land on
    /// the last fused step. The toy's ε is raised to 0.25 so the form is
    /// visible at the toy's O(1) key norms — on real weights it shows at
    /// ‖k‖ ≈ 1e-3 with ε = 1e-6 — and the two forms must give different
    /// logits on the same weights, so a `gdn_l2` that is ignored cannot pass.
    #[test]
    fn fused_gdn_decode_matches_tensor_decode_with_add_eps_l2() {
        let _serial = FUSED_TOGGLE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cfg = Qwen35Config {
            rms_norm_eps: 0.25,
            gdn_gate: GdnGate::Sigmoid,
            gdn_l2: GdnL2::AddEps,
            ..toy_config()
        };
        cfg.validate().expect("toy config validates");
        let device = crate::backend::cpu_device();
        let add = LoadedQwen35 {
            model: build(&cfg, &device, false),
            config: cfg.clone(),
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        };
        let clamp = LoadedQwen35 {
            model: add.model.clone(),
            config: Qwen35Config {
                gdn_l2: GdnL2::ClampNorm,
                ..cfg
            },
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        };

        let tensor = oracle_decode_steps(&add, false);
        let fused = oracle_decode_steps(&add, true);
        assert_steps_close(&tensor, &fused);

        let clamp_tensor = oracle_decode_steps(&clamp, false);
        let form_effect = tensor
            .iter()
            .flatten()
            .zip(clamp_tensor.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            form_effect > 1e-3,
            "AddEps and ClampNorm gave the same logits (max diff {form_effect}): \
             gdn_l2 is not reaching the DeltaNet"
        );

        let script: Vec<u32> = ORACLE_PREFIX
            .iter()
            .chain(&ORACLE_DECODE)
            .copied()
            .collect();
        let mut cache = add.new_cache();
        let one_shot = add
            .forward(&script, 0, &mut cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let last = fused.last().expect("decode steps ran");
        for (i, (x, y)) in one_shot.iter().zip(last).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "logit {i}: one-shot prefill {x} vs fused decode {y}"
            );
        }
    }

    /// The host state converts back for a tensor-path prefill (the
    /// multi-turn shape: prefill, fused decode, prefill again) without
    /// losing the carried recurrence.
    #[test]
    fn fused_decode_then_prefill_round_trips_state() {
        let _serial = FUSED_TOGGLE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let m = toy_model();
        let device = crate::backend::cpu_device();

        let run = |fused: bool| -> Vec<f32> {
            crate::flex::gdn::force_disable(!fused);
            let _restore = RestoreFused;
            let mut cache = m.new_cache();
            let _ = m.forward(&[1, 2, 3], 0, &mut cache, &device);
            let _ = m.forward(&[7], 3, &mut cache, &device); // decode (fused when on)
            let _ = m.forward(&[4, 9], 4, &mut cache, &device); // t = 2: tensor path
            m.forward(&[8], 6, &mut cache, &device)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
        };
        let tensor = run(false);
        let fused = run(true);
        for (i, (x, y)) in tensor.iter().zip(&fused).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "logit {i}: tensor {x} vs fused-then-tensor {y}"
            );
        }
    }
}
