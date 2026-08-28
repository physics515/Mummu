//! Shared transformer blocks, generic over `B: Backend`.
//!
//! Extracted from laurelane's parity-validated model ports (Qwen2 and the
//! LFM2.5 hybrid — byte-identical single-forward logits and greedy sequences
//! vs Candle / Ollama references). Every model in the zoo composes these; the
//! per-model files own only architecture wiring and weight-key remaps.

mod attention;
mod conv;
mod mlp;
mod moe;
mod rope;

pub use attention::{
    GqaAttention, GqaAttentionConfig, LayerKv, causal_mask, kv_append, kv_append_as,
    kv_f16_enabled, repeat_kv,
};
pub use conv::{ConvState, ShortConv, ShortConvConfig};
pub use mlp::{SwiGluMlp, SwiGluMlpConfig};
pub mod packed_gemv;
pub use packed_gemv::{
    Q4GemmOps, Q4GemvOps, Q4HeadOps, packed_gemv_enabled, try_q4s_gemv, try_q4s_head,
};
pub use moe::{
    DeviceExpert, ExpertExec, ExpertPool, ExpertWeights, MoeExperts, Routing, SparseMoe, SparseMoeConfig,
    SparseMoePerExpert, StagedExpert, trace_layer, trace_us,
};
pub use rope::{apply_rope, rope_tables, rotate_half};

/// Hard ceiling on `past + t` everywhere a sequence position is materialized.
/// Nothing in the zoo has a longer trained context; a position beyond this is
/// a caller bug (e.g. a decode loop that forgot its stop condition), not a
/// workload.
pub const MAX_CONTEXT_TOKENS: usize = 131_072;
