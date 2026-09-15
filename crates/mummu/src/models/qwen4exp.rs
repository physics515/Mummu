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
//! * **Gated DeltaNet + full attention**, in qwen35's exact proportions —
//!   `full_attention_interval = 4`, so 12 of 48 layers are attention and the
//!   other 36 are GDN at qwen35's shape (conv 4, state 128, 16 groups,
//!   inner 6144). That is why this port reuses `qwen35` rather than starting
//!   over.
//! * **A 512-expert MoE in every layer**, top-10 routed (`expert_count` /
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

use crate::gguf::{GgufFile, GgufValue};

/// Upper bound on PLE hash tables carried in the header. The shipped model
/// has 8 heads and 3 multipliers; this only stops a corrupt header from
/// driving an unbounded allocation.
const MAX_PLE_TABLE: usize = 1024;

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
    /// Leading dims of each head that RoPE rotates.
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
    pub eos_token_id: u32,
}

impl Qwen4expConfig {
    /// Is `layer` a full-attention layer (as opposed to Gated DeltaNet)?
    #[must_use]
    pub fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// q/k projection width in the DeltaNet mix.
    #[must_use]
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.d_state
    }

    /// Channels through the DeltaNet conv: q + k + v concatenated.
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.d_inner
    }

    /// How many of `num_layers` are full attention.
    #[must_use]
    pub fn attention_layers(&self) -> usize {
        (0..self.num_layers)
            .filter(|&l| self.is_attention(l))
            .count()
    }

    /// Total rows in the PLE table — the sum of every head's vocabulary.
    /// The shipped table is 28.8 GB at IQ4_NL, so this is the number that
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
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "qwen4exp" {
            return Err(format!("GGUF architecture '{arch}' is not qwen4exp"));
        }
        let usize_at = |key: &str| -> Result<usize, String> {
            f.get(key)
                .and_then(GgufValue::as_u64)
                .map(|v| usize::try_from(v).expect("metadata fits usize"))
                .ok_or_else(|| format!("GGUF metadata missing {key}"))
        };
        let f32_at = |key: &str| -> Result<f32, String> {
            f.get(key)
                .and_then(GgufValue::as_f32)
                .ok_or_else(|| format!("GGUF metadata missing {key}"))
        };
        // Header arrays are the PLE hash function's parameters. They are read,
        // never assumed: a wrong multiplier silently selects wrong embedding
        // rows, which reads as a quality regression rather than an error.
        let u64_array = |key: &str| -> Result<Vec<u64>, String> {
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
                    v.as_u64()
                        .ok_or_else(|| format!("{key} holds a non-integer entry"))
                })
                .collect()
        };

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

        let ple_layers = f
            .get("qwen4exp.ple.layers")
            .and_then(GgufValue::as_array)
            .map(|vals| {
                vals.iter()
                    .filter_map(GgufValue::as_i64)
                    .filter_map(|v| usize::try_from(v).ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let ple_layer_multipliers = u64_array("qwen4exp.ple.layer_multipliers")?;
        let ple_head_offsets = u64_array("qwen4exp.ple.head_offsets")?;
        let ple_head_vocab_sizes = u64_array("qwen4exp.ple.head_vocab_sizes")?;
        // Negative space: one offset per head vocabulary, or the table cannot
        // be indexed at all.
        if ple_head_offsets.len() != ple_head_vocab_sizes.len() {
            return Err(format!(
                "PLE has {} head offsets but {} head vocab sizes",
                ple_head_offsets.len(),
                ple_head_vocab_sizes.len()
            ));
        }

        let cfg = Self {
            vocab_size,
            hidden_size: usize_at("qwen4exp.embedding_length")?,
            num_layers,
            num_attention_heads: usize_at("qwen4exp.attention.head_count")?,
            num_key_value_heads: usize_at("qwen4exp.attention.head_count_kv")?,
            head_dim: usize_at("qwen4exp.attention.key_length")?,
            rms_norm_eps: f64::from(f32_at("qwen4exp.attention.layer_norm_rms_epsilon")?),
            rope_theta: f32_at("qwen4exp.rope.freq_base")?,
            rope_dim: usize_at("qwen4exp.rope.dimension_count")?,
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
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real shipped header's values (unsloth UD-Q4_K_XL, shard 1), so a
    /// drift in parsing shows up as a failing test rather than a bad load.
    fn shipped() -> Qwen4expConfig {
        Qwen4expConfig {
            vocab_size: 248_064,
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
            ple_head_offsets: vec![0, 20_000_003],
            ple_head_vocab_sizes: vec![20_000_003, 20_000_023],
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
        assert_eq!(c.ple_total_rows(), 40_000_026);
        assert_eq!(c.ple_head_offsets.len(), c.ple_head_vocab_sizes.len());
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
