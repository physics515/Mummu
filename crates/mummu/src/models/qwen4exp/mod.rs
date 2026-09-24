//! Qwen3.8-Flash-Next (`qwen4exp`) — header parsing for the Qwen4-preview
//! architecture.
//!
//! This is the CONFIG half of the port: the hyperparameters, read from a real
//! `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL header, with nothing about
//! the forward pass yet. It exists first because the shape of that header is
//! what the rest of the port has to satisfy, and because every value here was
//! observed rather than assumed.
//!
//! What the architecture is, from `qwen4exp.*` metadata (48 blocks, hidden
//! 2560, `general.size_label = "512x56B"`):
//!
//! * **Gated `DeltaNet` + full attention**, in qwen35's exact proportions —
//!   `full_attention_interval = 4`, so 12 of 48 layers are attention and the
//!   other 36 are GDN at qwen35's shape (conv 4, state 128, 16 groups,
//!   inner 6144). That is why this port reuses `qwen35` rather than starting
//!   over.
//! * **A 512-expert `MoE` in every layer**, top-10 routed (`expert_count` /
//!   `expert_used_count`), each expert 640 wide, plus a shared expert of the
//!   same width.
//! * **Hyper-connections**: 4 streams, low-rank 320.
//! * **A sparse-attention indexer** on the attention layers: 4 heads, key
//!   width 128, top-k 2048.
//! * **PLE** (per-layer embeddings) at layer 1: an n-gram hashed lookup.
//!   `ngram_size` 3 with 3 hash multipliers, and — measured against the
//!   shipped file, NOT inferred from `heads_per_ngram` (which is 8) — **16
//!   head vocabularies**, each a distinct prime near 20 M, whose offsets form
//!   a contiguous partition of a **320,001,446-row** table 160 wide. That is
//!   the "16 rows per token" the 2026-09-11 assessment measured. The
//!   multipliers/offsets/vocab sizes are DATA in the header, not constants —
//!   a port that hardcodes them silently selects wrong rows, so they are
//!   carried here verbatim.
//!
//! Not present, and worth stating because both were once assumed: there are
//! **no MTP/nextn tensors** and **no vision tensors**.

// Forward-pass building blocks, one file each so they can be built and
// oracle-tested independently; `model` assembles them into the causal LM.
pub mod experts;
pub mod hc;
pub mod model;
pub mod ple;
mod teacher;

pub use model::{LoadedQwen4exp, Qwen4exp, Qwen4expCache, Qwen4expLayer, load_from_gguf};

use crate::gguf::{GgufFile, GgufValue};
use crate::models::qwen2::EosIds;
use crate::models::qwen35::{GdnGate, GdnL2, Qwen35Config};

/// Upper bound on PLE hash tables and per-layer arrays carried in the
/// header. The shipped model has 16 head vocabularies, 3 multipliers and 48
/// compress ratios; this only stops a corrupt header from driving an
/// unbounded allocation.
const MAX_PLE_TABLE: usize = 1024;

/// An integer header value as u64 whatever its stored width or signedness,
/// refusing negatives. The shipped file stores `attention.compress_ratios`
/// as a SIGNED array (a strict unsigned read refused the real header), and
/// converters are free to pick either for ids.
fn non_negative(v: &GgufValue) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_i64().and_then(|x| u64::try_from(x).ok()))
}

/// A `u64` header value as `usize`, or which key is missing.
fn meta_usize(f: &GgufFile, key: &str) -> Result<usize, String> {
    f.get(key)
        .and_then(GgufValue::as_u64)
        .map(|v| usize::try_from(v).expect("metadata fits usize"))
        .ok_or_else(|| format!("GGUF metadata missing {key}"))
}

/// An `f32` header value, or which key is missing.
fn meta_f32(f: &GgufFile, key: &str) -> Result<f32, String> {
    f.get(key)
        .and_then(GgufValue::as_f32)
        .ok_or_else(|| format!("GGUF metadata missing {key}"))
}

/// A header array of non-negative integers, bounded by [`MAX_PLE_TABLE`].
///
/// Header arrays are the PLE hash function's parameters. They are read,
/// never assumed: a wrong multiplier silently selects wrong embedding
/// rows, which reads as a quality regression rather than an error.
fn meta_u64_array(f: &GgufFile, key: &str) -> Result<Vec<u64>, String> {
    let vals = f
        .get(key)
        .and_then(GgufValue::as_array)
        .ok_or_else(|| format!("GGUF metadata missing array {key}"))?;
    if vals.len() > MAX_PLE_TABLE {
        return Err(format!(
            "{key} has {} entries, past the {MAX_PLE_TABLE} bound",
            vals.len()
        ));
    }
    vals.iter()
        .map(|v| {
            non_negative(v).ok_or_else(|| format!("{key} holds a non-integer or negative entry"))
        })
        .collect()
}

/// `qwen4exp.ple.layers` (absent → none), refusing more than one layer or
/// one past the trunk.
fn ple_layers(f: &GgufFile, num_layers: usize) -> Result<Vec<usize>, String> {
    let ple_layers = f
        .get("qwen4exp.ple.layers")
        .and_then(GgufValue::as_array)
        .map_or_else(Vec::new, |vals| {
            vals.iter()
                .filter_map(GgufValue::as_i64)
                .filter_map(|v| usize::try_from(v).ok())
                .collect::<Vec<_>>()
        });
    // One PLE module at most, as llama.cpp: hparams hold one set of hash
    // constants and the cache one conv history.
    if ple_layers.len() > 1 {
        return Err(format!(
            "qwen4exp.ple.layers lists {} layers; only one PLE layer is supported",
            ple_layers.len()
        ));
    }
    if let Some(&l) = ple_layers.iter().find(|&&l| l >= num_layers) {
        return Err(format!(
            "PLE layer {l} is out of range ({num_layers} layers)"
        ));
    }
    Ok(ple_layers)
}

/// `qwen4exp.attention.compress_ratios`, exactly one entry per layer.
fn compress_ratios(f: &GgufFile, num_layers: usize) -> Result<Vec<usize>, String> {
    let ratios = meta_u64_array(f, "qwen4exp.attention.compress_ratios")?
        .into_iter()
        .map(|v| usize::try_from(v).map_err(|_| "compress ratio does not fit usize".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if ratios.len() != num_layers {
        return Err(format!(
            "qwen4exp.attention.compress_ratios has {} entries for {num_layers} layers",
            ratios.len()
        ));
    }
    Ok(ratios)
}

/// The PLE hash arrays as the header carries them.
struct PleHashArrays {
    multipliers: Vec<u64>,
    head_offsets: Vec<u64>,
    head_vocab_sizes: Vec<u64>,
}

/// `qwen4exp.ple.{layer_multipliers, head_offsets, head_vocab_sizes}`.
fn ple_hash_arrays(f: &GgufFile) -> Result<PleHashArrays, String> {
    let multipliers = meta_u64_array(f, "qwen4exp.ple.layer_multipliers")?;
    let head_offsets = meta_u64_array(f, "qwen4exp.ple.head_offsets")?;
    let head_vocab_sizes = meta_u64_array(f, "qwen4exp.ple.head_vocab_sizes")?;
    // Negative space: one offset per head vocabulary, or the table cannot
    // be indexed at all.
    if head_offsets.len() != head_vocab_sizes.len() {
        return Err(format!(
            "PLE has {} head offsets but {} head vocab sizes",
            head_offsets.len(),
            head_vocab_sizes.len()
        ));
    }
    Ok(PleHashArrays {
        multipliers,
        head_offsets,
        head_vocab_sizes,
    })
}

/// `qwen4exp.rope.dimension_count`, accepted only when the header's
/// `rope.dimension_sections` are text-degenerate.
///
/// The attention layers run plain partial `RoPE` over `rope_dim` dims.
/// That is exact for llama.cpp's IMROPE only while every rotated
/// frequency is assigned to the t/h/w sections, which all equal the
/// token position for text; a non-zero 4th (e) section would give
/// those dims angle 0 in llama.cpp and a real rotation here. The
/// shipped header is `[11, 11, 10, 0]`, so refuse anything else loudly
/// rather than rotate the wrong dims silently.
fn text_rope_dim(f: &GgufFile) -> Result<usize, String> {
    let rope_dim = meta_usize(f, "qwen4exp.rope.dimension_count")?;
    let sections = f
        .get("qwen4exp.rope.dimension_sections")
        .and_then(GgufValue::as_array)
        .ok_or("GGUF metadata missing array qwen4exp.rope.dimension_sections")?
        .iter()
        .map(|v| {
            v.as_i64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| "rope.dimension_sections holds a non-integer entry".to_string())
        })
        .collect::<Result<Vec<usize>, String>>()?;
    let rotated: usize = sections.iter().take(3).sum();
    if sections.len() != 4 || sections[3] != 0 || 2 * rotated != rope_dim {
        return Err(format!(
            "rope.dimension_sections {sections:?} is not text-degenerate for rope.dimension_count \
             {rope_dim}: only [t, h, w, 0] with 2*(t+h+w) == rope_dim is implemented"
        ));
    }
    Ok(rope_dim)
}

/// Hyperparameters of the `qwen4exp` architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4expConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Attention head width (`key_length` == `value_length` == 256).
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// Leading dims of each head that `RoPE` rotates.
    pub rope_dim: usize,
    /// Layer `i` is full attention iff `(i+1) % interval == 0`.
    pub full_attention_interval: usize,
    // --- Gated DeltaNet (shared shape with qwen35) ---
    pub conv_kernel: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    // --- Mixture of experts (every layer) ---
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_ffn_size: usize,
    pub expert_shared_ffn_size: usize,
    // --- Hyper-connections ---
    pub hyper_connection_count: usize,
    pub hyper_connection_low_rank: usize,
    // --- Sparse-attention indexer ---
    pub indexer_head_count: usize,
    pub indexer_key_length: usize,
    pub indexer_top_k: usize,
    // --- Per-layer embeddings (PLE) ---
    /// Which layers consume a PLE row (the shipped model: `[1]`).
    pub ple_layers: Vec<usize>,
    pub ple_ngram_size: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_conv_kernel: usize,
    /// Width of one PLE row (`embedding_length_per_layer_input`).
    pub ple_row_width: usize,
    /// Hash multipliers, one per n-gram position.
    pub ple_layer_multipliers: Vec<u64>,
    /// Start of each head's slice of the PLE table.
    pub ple_head_offsets: Vec<u64>,
    /// Vocabulary size of each head's slice.
    pub ple_head_vocab_sizes: Vec<u64>,
    /// The PLE n-gram window's reset token (`qwen4exp.ple.eos_token_id`,
    /// 248044 in the shipped file). NOT [`Self::eos_token_id`] (248046):
    /// hashing with the tokenizer EOS silently selects wrong rows after
    /// every document boundary.
    pub ple_eos_token_id: u32,
    /// The token llama.cpp hashes at image-embedding positions
    /// (`qwen4exp.ple.image_token_id`, optional). Unused by this text-only
    /// port; carried so the header read is complete.
    pub ple_image_token_id: Option<u32>,
    /// `qwen4exp.attention.compress_ratios`, one per layer: the QSA block
    /// size on attention layers (4 in the shipped file), 0 where a layer
    /// has no sparse attention.
    pub attention_compress_ratios: Vec<usize>,
    /// The tokenizer EOS (`tokenizer.ggml.eos_token_id`).
    pub eos_token_id: u32,
}

impl Qwen4expConfig {
    /// Is `layer` a full-attention layer (as opposed to Gated `DeltaNet`)?
    #[must_use]
    pub const fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// q/k projection width in the `DeltaNet` mix.
    #[must_use]
    pub const fn key_dim(&self) -> usize {
        self.n_k_heads * self.d_state
    }

    /// Channels through the `DeltaNet` conv: q + k + v concatenated.
    #[must_use]
    pub const fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.d_inner
    }

    /// How many of `num_layers` are full attention.
    #[must_use]
    pub fn attention_layers(&self) -> usize {
        (0..self.num_layers)
            .filter(|&l| self.is_attention(l))
            .count()
    }

    /// How many tokens the attention cache may hold while DENSE attention
    /// is still exactly the reference's sparse (QSA) attention, or `None`
    /// when no layer has a compress ratio (then attention is dense anyway).
    ///
    /// QSA selects `indexer_top_k` tokens' worth of whole blocks of
    /// `compress_ratio` plus the incomplete tail block, so up to
    /// `top_k + ratio - 1` cached tokens every cell is selected and the
    /// result is bit-identical to dense attention (llama.cpp PR #27742):
    /// 2048 + 4 - 1 = 2051 in the shipped file. Past it this port, which
    /// has no indexer, would silently compute a different model — so the
    /// forward refuses instead. The smallest ratio across layers bounds.
    #[must_use]
    pub fn dense_attention_limit(&self) -> Option<usize> {
        let ratio = self
            .attention_compress_ratios
            .iter()
            .copied()
            .filter(|&r| r > 0)
            .min()?;
        Some(self.indexer_top_k + ratio - 1)
    }

    /// The config the reused qwen35 blocks read ([`crate::models::qwen35::GatedAttention`],
    /// [`crate::models::qwen35::GatedDeltaNet`]): the same shapes, and the
    /// `DeltaNet` output gate set to **sigmoid** — qwen4exp's one numerical
    /// difference from qwen35 in those blocks (llama.cpp
    /// `llama-qwen4exp.cpp` `build_norm_gated`). `intermediate_size` has no
    /// qwen4exp meaning (the FFN is the `MoE`) and neither block reads it; it
    /// carries the shared-expert width only so the field is not a zero.
    #[must_use]
    pub const fn blocks_config(&self) -> Qwen35Config {
        Qwen35Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            intermediate_size: self.expert_shared_ffn_size,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_dim: self.rope_dim,
            full_attention_interval: self.full_attention_interval,
            conv_kernel: self.conv_kernel,
            d_inner: self.d_inner,
            d_state: self.d_state,
            n_k_heads: self.n_k_heads,
            n_v_heads: self.n_v_heads,
            gdn_gate: GdnGate::Sigmoid,
            // llama.cpp qwen4exp.cpp build_gdn_l2_norm and transformers'
            // l2norm: x / sqrt(‖x‖² + ε). The clamp form missed llama.cpp's
            // keys by up to 2.8e-2 on real prompts (see GdnL2).
            gdn_l2: GdnL2::AddEps,
            eos_token_id: EosIds::One(self.eos_token_id),
            hadamard: None,
        }
    }

    /// Width of one token's PLE embedding: `(ngram-1)·heads_per_ngram` rows
    /// of `ple_row_width` (16 · 160 = 2560).
    #[must_use]
    pub const fn ple_embed_width(&self) -> usize {
        self.ple_ngram_size.saturating_sub(1) * self.ple_heads_per_ngram * self.ple_row_width
    }

    /// Total rows in the PLE table — the sum of every head's vocabulary.
    /// The shipped table is 28.8 GB at `IQ4_NL`, so this is the number that
    /// decides whether it can be resident or must stay on disk.
    #[must_use]
    pub fn ple_total_rows(&self) -> u64 {
        self.ple_head_vocab_sizes.iter().sum()
    }

    /// Hyperparameters from a GGUF header's `qwen4exp.*` metadata.
    ///
    /// `f` should be the MERGED header of the split set
    /// ([`GgufFile::open_sharded`]): the shipped model's first shard carries
    /// the full KV but **zero tensors**, so `token_embd.weight` — which is
    /// where the vocabulary size comes from — only exists once the shards
    /// are joined.
    ///
    /// # Errors
    ///
    /// The architecture is not `qwen4exp`; a required `qwen4exp.*` key or
    /// array is missing, negative, or over the [`MAX_PLE_TABLE`] bound;
    /// `token_embd.weight` is absent or not 2-D; `full_attention_interval`
    /// is zero; `expert_used_count` exceeds `expert_count`; more than one
    /// PLE layer, or one past the trunk; a compress-ratio array that is not
    /// one entry per layer; PLE head offsets and vocab sizes of different
    /// lengths; `rope.dimension_sections` that are not text-degenerate; or
    /// any invariant [`Self::validate`] refuses.
    ///
    /// # Panics
    ///
    /// Only on a 32-bit target, when a header dimension or count does not
    /// fit `usize`.
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "qwen4exp" {
            return Err(format!("GGUF architecture '{arch}' is not qwen4exp"));
        }
        let usize_at = |key: &str| meta_usize(f, key);
        let f32_at = |key: &str| meta_f32(f, key);

        let embd = f
            .tensor("token_embd.weight")
            .ok_or("GGUF has no token_embd.weight (open the split set with open_sharded)")?;
        // ggml dims are fastest-varying first: [hidden, vocab].
        let vocab_size = usize::try_from(*embd.dims.get(1).ok_or("token_embd is not 2-D")?)
            .expect("vocab fits usize");

        let num_layers = usize_at("qwen4exp.block_count")?;
        let full_attention_interval = usize_at("qwen4exp.full_attention_interval")?;
        if full_attention_interval == 0 {
            return Err("full_attention_interval must be non-zero".into());
        }
        let expert_count = usize_at("qwen4exp.expert_count")?;
        let expert_used_count = usize_at("qwen4exp.expert_used_count")?;
        if expert_used_count > expert_count {
            return Err(format!(
                "expert_used_count ({expert_used_count}) exceeds expert_count ({expert_count})"
            ));
        }

        let ple_layers = ple_layers(f, num_layers)?;
        let attention_compress_ratios = compress_ratios(f, num_layers)?;
        let PleHashArrays {
            multipliers: ple_layer_multipliers,
            head_offsets: ple_head_offsets,
            head_vocab_sizes: ple_head_vocab_sizes,
        } = ple_hash_arrays(f)?;
        let rope_dim = text_rope_dim(f)?;

        let cfg = Self {
            vocab_size,
            hidden_size: usize_at("qwen4exp.embedding_length")?,
            num_layers,
            num_attention_heads: usize_at("qwen4exp.attention.head_count")?,
            num_key_value_heads: usize_at("qwen4exp.attention.head_count_kv")?,
            head_dim: usize_at("qwen4exp.attention.key_length")?,
            rms_norm_eps: f64::from(f32_at("qwen4exp.attention.layer_norm_rms_epsilon")?),
            rope_theta: f32_at("qwen4exp.rope.freq_base")?,
            rope_dim,
            full_attention_interval,
            conv_kernel: usize_at("qwen4exp.ssm.conv_kernel")?,
            d_inner: usize_at("qwen4exp.ssm.inner_size")?,
            d_state: usize_at("qwen4exp.ssm.state_size")?,
            n_k_heads: usize_at("qwen4exp.ssm.group_count")?,
            n_v_heads: usize_at("qwen4exp.ssm.time_step_rank")?,
            expert_count,
            expert_used_count,
            expert_ffn_size: usize_at("qwen4exp.expert_feed_forward_length")?,
            expert_shared_ffn_size: usize_at("qwen4exp.expert_shared_feed_forward_length")?,
            hyper_connection_count: usize_at("qwen4exp.hyper_connection.count")?,
            hyper_connection_low_rank: usize_at("qwen4exp.hyper_connection.low_rank")?,
            indexer_head_count: usize_at("qwen4exp.attention.indexer.head_count")?,
            indexer_key_length: usize_at("qwen4exp.attention.indexer.key_length")?,
            indexer_top_k: usize_at("qwen4exp.attention.indexer.top_k")?,
            ple_layers,
            ple_ngram_size: usize_at("qwen4exp.ple.ngram_size")?,
            ple_heads_per_ngram: usize_at("qwen4exp.ple.heads_per_ngram")?,
            ple_conv_kernel: usize_at("qwen4exp.ple.conv_kernel")?,
            ple_row_width: usize_at("qwen4exp.embedding_length_per_layer_input")?,
            ple_layer_multipliers,
            ple_head_offsets,
            ple_head_vocab_sizes,
            // Required, never defaulted to the tokenizer EOS (they differ).
            ple_eos_token_id: u32::try_from(
                f.get("qwen4exp.ple.eos_token_id")
                    .and_then(non_negative)
                    .ok_or("GGUF metadata missing qwen4exp.ple.eos_token_id")?,
            )
            .map_err(|_| "PLE eos token id does not fit u32".to_string())?,
            ple_image_token_id: f
                .get("qwen4exp.ple.image_token_id")
                .and_then(non_negative)
                .map(|v| u32::try_from(v).map_err(|_| "PLE image token id does not fit u32"))
                .transpose()?,
            attention_compress_ratios,
            eos_token_id: u32::try_from(
                f.get("tokenizer.ggml.eos_token_id")
                    .and_then(GgufValue::as_u64)
                    .ok_or("GGUF metadata missing tokenizer.ggml.eos_token_id")?,
            )
            .map_err(|_| "eos token id does not fit u32".to_string())?,
        };
        // Positive space: the derived layer split must agree with the header.
        debug_assert!(
            cfg.attention_layers() <= cfg.num_layers,
            "attention layers are a subset"
        );
        cfg.validate()?;
        Ok(cfg)
    }

    /// Shape invariants the forward depends on, checked once at load.
    ///
    /// # Errors
    ///
    /// Fewer than two hyper-connection streams or a zero low rank; a top-k
    /// that is zero or exceeds the expert count; a PLE layer that is an
    /// attention layer; a PLE layer with an empty embedding; a non-zero
    /// compress ratio on a `DeltaNet` layer; or a `DeltaNet`/attention
    /// layout the reused qwen35 blocks refuse
    /// ([`Qwen35Config::validate`]).
    pub fn validate(&self) -> Result<(), String> {
        if self.hyper_connection_count < 2 || self.hyper_connection_low_rank == 0 {
            // llama.cpp and transformers both refuse hc <= 1: nothing to mix.
            return Err(format!(
                "hyper_connection.count {} / low_rank {} is degenerate",
                self.hyper_connection_count, self.hyper_connection_low_rank
            ));
        }
        if self.expert_used_count == 0 || self.expert_used_count > self.expert_count {
            return Err(format!(
                "top-{} of {} experts is not a routing",
                self.expert_used_count, self.expert_count
            ));
        }
        if let Some(&l) = self.ple_layers.first()
            && self.is_attention(l)
        {
            // The PLE conv history rides in the recurrent cache row.
            return Err(format!("PLE layer {l} must be a DeltaNet layer"));
        }
        if !self.ple_layers.is_empty() && self.ple_embed_width() == 0 {
            return Err("PLE layer present but the PLE embedding is empty".into());
        }
        for (l, &r) in self.attention_compress_ratios.iter().enumerate() {
            if r > 0 && !self.is_attention(l) {
                return Err(format!(
                    "layer {l} has compress ratio {r} but is not attention"
                ));
            }
        }
        self.blocks_config().validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real shipped header's values (unsloth UD-Q4_K_XL, all four
    /// shards), so a drift in parsing shows up as a failing test rather than
    /// a bad load. Shared with the model tests (tensor-map completeness).
    pub fn shipped() -> Qwen4expConfig {
        Qwen4expConfig {
            vocab_size: 248_320,
            hidden_size: 2560,
            num_layers: 48,
            num_attention_heads: 24,
            num_key_value_heads: 2,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rope_dim: 64,
            full_attention_interval: 4,
            conv_kernel: 4,
            d_inner: 6144,
            d_state: 128,
            n_k_heads: 16,
            n_v_heads: 48,
            expert_count: 512,
            expert_used_count: 10,
            expert_ffn_size: 640,
            expert_shared_ffn_size: 640,
            hyper_connection_count: 4,
            hyper_connection_low_rank: 320,
            indexer_head_count: 4,
            indexer_key_length: 128,
            indexer_top_k: 2048,
            ple_layers: vec![1],
            ple_ngram_size: 3,
            ple_heads_per_ngram: 8,
            ple_conv_kernel: 4,
            ple_row_width: 160,
            ple_layer_multipliers: vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            ple_head_offsets: vec![
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275,
            ],
            ple_head_vocab_sizes: vec![
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ],
            ple_eos_token_id: 248_044,
            ple_image_token_id: Some(248_056),
            attention_compress_ratios: (0..48)
                .map(|l| if (l + 1) % 4 == 0 { 4 } else { 0 })
                .collect(),
            eos_token_id: 248_046,
        }
    }

    #[test]
    fn twelve_of_forty_eight_layers_are_full_attention() {
        let c = shipped();
        // interval 4 over 48 blocks — the same proportion qwen35 uses, which
        // is what lets that port's DeltaNet be reused for the other 36.
        assert_eq!(c.attention_layers(), 12);
        assert!(c.is_attention(3), "layer 3 is the first attention layer");
        assert!(!c.is_attention(0), "layer 0 is DeltaNet");
        assert!(!c.is_attention(2));
    }

    #[test]
    fn deltanet_widths_match_qwen35s_shape() {
        let c = shipped();
        // 16 groups x 128 state = 2048 per projection; conv carries q+k+v.
        assert_eq!(c.key_dim(), 2048);
        assert_eq!(c.conv_dim(), 2 * 2048 + 6144);
    }

    #[test]
    fn the_ple_table_is_summed_from_the_headers_head_vocabs() {
        let c = shipped();
        // Read from the header, never assumed: a wrong total means wrong rows.
        assert_eq!(c.ple_total_rows(), 320_001_446);
        assert_eq!(c.ple_head_offsets.len(), c.ple_head_vocab_sizes.len());
        // 16 rows of 160 per token: the PLE embedding is exactly E wide.
        assert_eq!(c.ple_embed_width(), 2560);
    }

    #[test]
    fn the_shipped_config_validates_and_bounds_dense_attention_at_2051() {
        let c = shipped();
        c.validate().expect("shipped config validates");
        // top_k 2048 + ratio 4 - 1: past this, dense != QSA.
        assert_eq!(c.dense_attention_limit(), Some(2051));
        let mut dense = c;
        dense.attention_compress_ratios.fill(0);
        assert_eq!(
            dense.dense_attention_limit(),
            None,
            "no ratio, no indexer bound"
        );
    }

    #[test]
    fn the_blocks_adapter_gates_the_deltanet_with_sigmoid() {
        // The numerical differences from mummu's qwen35 inside the reused
        // blocks; a silu gate or the clamp L2 form computes a plausible,
        // wrong model (the clamp form moved Flash-Next's layer-28 DeltaNet
        // output by 1.5e-2 against llama.cpp).
        let b = shipped().blocks_config();
        assert_eq!(b.gdn_gate, GdnGate::Sigmoid);
        assert_eq!(b.gdn_l2, GdnL2::AddEps);
        assert_eq!(b.conv_dim(), 10_240);
        assert_eq!((b.hidden_size, b.head_dim, b.rope_dim), (2560, 256, 64));
    }

    #[test]
    fn header_integers_are_read_whatever_their_signedness() {
        // The shipped file stores attention.compress_ratios as a SIGNED
        // array; a strict unsigned read refused the real header.
        assert_eq!(non_negative(&GgufValue::I32(4)), Some(4));
        assert_eq!(non_negative(&GgufValue::U32(248_044)), Some(248_044));
        assert_eq!(non_negative(&GgufValue::I32(-1)), None, "negatives refused");
        assert_eq!(non_negative(&GgufValue::F32(4.0)), None, "floats refused");
    }

    #[test]
    fn a_ple_layer_on_an_attention_layer_is_refused() {
        let mut c = shipped();
        c.ple_layers = vec![3];
        let err = c.validate().expect_err("attention PLE layer refused");
        assert!(err.contains("PLE layer 3"), "names the layer: {err}");
    }

    #[test]
    fn a_non_qwen4exp_architecture_is_refused() {
        // Guards the dispatch seam: loading a qwen35 file as qwen4exp would
        // otherwise half-work and produce garbage.
        let f = crate::gguf::GgufFile {
            path: std::path::PathBuf::from("x.gguf"),
            version: 3,
            metadata: vec![(
                "general.architecture".into(),
                GgufValue::Str("qwen35".into()),
            )],
            tensors: Vec::new(),
            alignment: 32,
            data_offset: 0,
            shards: Vec::new(),
        };
        let err = Qwen4expConfig::from_gguf(&f).expect_err("wrong arch refused");
        assert!(err.contains("not qwen4exp"), "names the mismatch: {err}");
    }
}
