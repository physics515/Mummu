//! LFM2 / LFM2.5 (Liquid Foundation Model) hybrid decoder on the shared `nn`
//! blocks: Embedding → N×{operator_norm, (gated ShortConv "LIV" | GQA attention
//! with per-head q/k RMSNorm), ffn_norm, SwiGLU} → embedding_norm → tied
//! lm-head. `layer_types[i]` in the checkpoint's `config.json` selects the
//! operator per layer (10 conv + 6 attention for the 1.2B).
//!
//! Ported from laurelane's implementation (greedy temperature-0 token match vs
//! a local Ollama `lfm2.5` reference). Mummu's parity gate (P7) re-verifies
//! here before the port is marked trusted.

use std::path::Path;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::attn_config::{RopeScaling, check_sliding_window, sliding_window_from_gguf};
use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo, GgufValue};
use crate::import::{
    FloatCastAdapter, DequantSink, ImportError, gguf_store, load_checked, required_file,
};
use crate::models::CausalLm;
use crate::models::qwen2::EosIds;
use crate::nn::{
    ConvState, GqaAttention, GqaAttentionConfig, LayerKv, ShortConv, ShortConvConfig, SwiGluMlp,
    SwiGluMlpConfig, causal_mask, rope_tables,
};

/// LFM2 architecture hyperparameters, read from the checkpoint's `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    /// Top-level in older checkpoints (LFM2.5-1.2B); newer ones (LFM2.5-230M)
    /// nest it under `rope_parameters` instead — resolved (and required) by
    /// [`Self::from_json_bytes`] / [`Self::validate`].
    #[serde(default)]
    pub rope_theta: f32,
    /// The newer transformers convention: `{"rope_theta": …, "rope_type": …}`.
    #[serde(default, skip_serializing)]
    rope_parameters: Option<RopeParameters>,
    /// The third spelling: a top-level `rope_scaling` object, as Qwen and the
    /// Llama family write it. Refused unless plain ([`crate::attn_config`]).
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// The trained context length (LFM2.5-1.2B: 128 000), used to tell an
    /// inert sliding window from a clipping one.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// LFM2 has no `use_sliding_window` flag: its attention layers are full,
    /// and its short-conv layers are a different mechanism entirely — so a
    /// declared window would be live, and is refused.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Conv kernel length `K` (the rolling decode state keeps `K-1`).
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub block_ff_dim: usize,
    pub block_multiple_of: usize,
    #[serde(default)]
    pub block_auto_adjust_ff_dim: bool,
    /// `"full_attention"` or `"conv"` per layer.
    pub layer_types: Vec<String>,
    /// `<|im_end|>` (id 7) for the instruct checkpoints.
    #[serde(default)]
    pub eos_token_id: EosIds,
}

impl Lfm2Config {
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// SwiGLU hidden dim: `block_auto_adjust_ff_dim` shrinks to 2/3 then
    /// rounds up to `block_multiple_of` (8192 for the 1.2B).
    #[must_use]
    pub fn ff_dim(&self) -> usize {
        if self.block_auto_adjust_ff_dim {
            let d = (2 * self.block_ff_dim) / 3;
            let m = self.block_multiple_of.max(1);
            d.div_ceil(m) * m
        } else {
            self.block_ff_dim
        }
    }

    fn is_attention(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .is_some_and(|s| s == "full_attention")
    }

    /// Parse `config.json` bytes. `rope_theta` may arrive top-level (older
    /// checkpoints) or nested under `rope_parameters` (the newer transformers
    /// convention, LFM2.5-230M) — only plain rotary (`rope_type: "default"`)
    /// is implemented, anything else fails loudly.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cfg: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if let Some(rp) = cfg.rope_parameters.take() {
            // Same rule, same message as every other family's `rope_scaling`:
            // only plain rotary is computed, anything else is named and
            // refused rather than silently approximated.
            RopeScaling {
                rope_type: Some(rp.rope_type.clone()),
                ..RopeScaling::default()
            }
            .check("lfm2 config.json rope_parameters")?;
            if cfg.rope_theta != 0.0 && cfg.rope_theta != rp.rope_theta {
                return Err(format!(
                    "rope_theta given twice and disagreeing: {} (top-level) vs {} (rope_parameters)",
                    cfg.rope_theta, rp.rope_theta
                ));
            }
            cfg.rope_theta = rp.rope_theta;
        }
        cfg.validate("lfm2 config.json")?;
        Ok(cfg)
    }

    /// Hyperparameters from a GGUF header's `lfm2.*` metadata. Layer kinds
    /// come from llama.cpp's per-layer `attention.head_count_kv` array:
    /// `0` marks a shortconv layer, nonzero an attention layer.
    /// `feed_forward_length` in a GGUF is the already-adjusted SwiGLU dim,
    /// so auto-adjust is off.
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "lfm2" {
            return Err(format!("GGUF architecture '{arch}' is not lfm2"));
        }
        let meta_usize = |key: &str| -> Result<usize, String> {
            f.get(key)
                .and_then(GgufValue::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("missing or non-integer GGUF metadata '{key}'"))
        };
        let hidden_size = meta_usize("lfm2.embedding_length")?;
        let num_hidden_layers = meta_usize("lfm2.block_count")?;

        // Per-layer kv-head counts: 0 = conv, nonzero = attention (all
        // nonzero entries must agree — one KV geometry per model).
        let kv_per_layer = f
            .get("lfm2.attention.head_count_kv")
            .and_then(GgufValue::as_array)
            .ok_or("missing per-layer array 'lfm2.attention.head_count_kv'")?;
        if kv_per_layer.len() != num_hidden_layers {
            return Err(format!(
                "head_count_kv has {} entries for {num_hidden_layers} layers",
                kv_per_layer.len()
            ));
        }
        let mut layer_types = Vec::with_capacity(num_hidden_layers);
        let mut kv_heads: Option<u64> = None;
        for (i, v) in kv_per_layer.iter().enumerate() {
            let n = v
                .as_i64()
                .and_then(|n| u64::try_from(n).ok()) // i32 array in real files
                .ok_or_else(|| format!("head_count_kv[{i}] is not a non-negative integer"))?;
            if n == 0 {
                layer_types.push("conv".to_string());
            } else {
                if kv_heads.is_some_and(|k| k != n) {
                    return Err(format!("head_count_kv mixes {kv_heads:?} and {n}"));
                }
                kv_heads = Some(n);
                layer_types.push("full_attention".to_string());
            }
        }
        let Some(kv_heads) = kv_heads else {
            return Err("no attention layers in head_count_kv".into());
        };

        let embd = f
            .tensor("token_embd.weight")
            .ok_or("GGUF has no token_embd.weight tensor")?;
        if embd.dims.len() != 2 || embd.dims[0] != hidden_size as u64 {
            return Err(format!(
                "token_embd.weight dims {:?} do not match embedding_length {hidden_size}",
                embd.dims
            ));
        }
        if f.tensor("output.weight").is_some() {
            return Err("LFM2 GGUF carries an untied output.weight — unsupported".into());
        }
        let eps = f
            .get("lfm2.attention.layer_norm_rms_epsilon")
            .and_then(GgufValue::as_f32)
            .ok_or("missing 'lfm2.attention.layer_norm_rms_epsilon'")?;
        let theta = f
            .get("lfm2.rope.freq_base")
            .and_then(GgufValue::as_f32)
            .ok_or("missing 'lfm2.rope.freq_base'")?;
        let cfg = Self {
            vocab_size: usize::try_from(embd.dims[1]).map_err(|_| "vocab too large")?,
            hidden_size,
            num_hidden_layers,
            num_attention_heads: meta_usize("lfm2.attention.head_count")?,
            num_key_value_heads: usize::try_from(kv_heads).map_err(|_| "kv heads too large")?,
            norm_eps: f64::from(eps),
            rope_theta: theta,
            rope_parameters: None,
            rope_scaling: RopeScaling::from_gguf(f, "lfm2"),
            max_position_embeddings: f
                .get("lfm2.context_length")
                .and_then(GgufValue::as_u64)
                .and_then(|v| usize::try_from(v).ok()),
            sliding_window: sliding_window_from_gguf(f, "lfm2"),
            conv_l_cache: meta_usize("lfm2.shortconv.l_cache")?,
            block_ff_dim: meta_usize("lfm2.feed_forward_length")?,
            block_multiple_of: 1,
            block_auto_adjust_ff_dim: false,
            layer_types,
            eos_token_id: f
                .get("tokenizer.ggml.eos_token_id")
                .and_then(GgufValue::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .map_or(EosIds::None, EosIds::One),
        };
        cfg.validate("lfm2 GGUF header")?;
        Ok(cfg)
    }

    fn validate(&self, whose: &str) -> Result<(), String> {
        if let Some(scaling) = &self.rope_scaling {
            scaling.check(whose)?;
        }
        check_sliding_window(
            self.sliding_window,
            self.sliding_window.is_some(),
            self.max_position_embeddings,
            whose,
        )?;
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "layer_types has {} entries but num_hidden_layers is {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "num_attention_heads ({}) must be a positive multiple of num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.conv_l_cache < 2 {
            return Err(format!(
                "conv_L_cache must be >= 2, got {}",
                self.conv_l_cache
            ));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(format!(
                "rope_theta must be a positive float (top-level or in rope_parameters), got {}",
                self.rope_theta
            ));
        }
        Ok(())
    }
}

/// The nested rope block newer checkpoints carry in `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
struct RopeParameters {
    rope_theta: f32,
    #[serde(default = "RopeParameters::default_type")]
    rope_type: String,
}

impl RopeParameters {
    fn default_type() -> String {
        "default".into()
    }
}

/// One hybrid layer: exactly one of `conv` / `self_attn` is present, per
/// `layer_types[i]`. Field names mirror the HF checkpoint.
#[derive(Module, Debug)]
pub struct HybridLayer {
    pub conv: Option<ShortConv>,
    pub self_attn: Option<GqaAttention>,
    pub feed_forward: SwiGluMlp,
    pub operator_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
}

/// The LFM2 decoder stack (tied lm-head).
#[derive(Module, Debug)]
pub struct Lfm2 {
    pub embed_tokens: Embedding,
    pub layers: Vec<HybridLayer>,
    pub embedding_norm: RmsNorm,
}

/// Per-layer decode cache: conv layers roll the last `K-1` gated inputs,
/// attention layers keep the running k/v.
pub enum HybridKv {
    Conv(ConvState),
    Attn(LayerKv),
}

/// A weight-loaded LFM2 plus its config.
pub struct LoadedLfm2 {
    pub model: Lfm2,
    pub config: Lfm2Config,
    /// The parsed sibling `tokenizer_config.json`, when one was present and
    /// well-formed beside a safetensors checkpoint (the load-time gate has
    /// already cross-checked its EOS against `config.json`). A consumer reads
    /// config-driven EOS/BOS/PAD ids from it (`eos_id()`, `bos_id()`, …). `None`
    /// for a GGUF load (self-contained; no sibling file) or a dir without one.
    pub tokenizer_config: Option<crate::tok_config::TokenizerConfig>,
}

fn build(cfg: &Lfm2Config, device: &Device) -> Lfm2 {
    let attn_cfg = GqaAttentionConfig {
        hidden_size: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        bias: false,                     // LFM2 projections are bias-free
        qk_norm_eps: Some(cfg.norm_eps), // per-head q/k RMSNorm
        qk_norm_projection: false,
    };
    let conv_cfg = ShortConvConfig {
        hidden_size: cfg.hidden_size,
        kernel_len: cfg.conv_l_cache,
    };
    let mlp_cfg = SwiGluMlpConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.ff_dim(),
    };
    let norm = |dev: &Device| {
        RmsNormConfig::new(cfg.hidden_size)
            .with_epsilon(cfg.norm_eps)
            .init(dev)
    };
    let layers = (0..cfg.num_hidden_layers)
        .map(|i| {
            let attn = cfg.is_attention(i);
            HybridLayer {
                conv: (!attn).then(|| conv_cfg.init(device)),
                self_attn: attn.then(|| attn_cfg.init(device)),
                feed_forward: mlp_cfg.init(device),
                operator_norm: norm(device),
                ffn_norm: norm(device),
            }
        })
        .collect();
    Lfm2 {
        embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
        layers,
        embedding_norm: norm(device),
    }
}

/// Build the architecture from `dir/config.json` and load
/// `dir/model.safetensors`, checked. Key remap: strip `model.`, RmsNorm
/// `weight` → `gamma`, and LFM2's `out_proj`/`q_layernorm`/`k_layernorm` onto
/// the shared block's `o_proj`/`q_norm`/`k_norm`.
pub fn load_from_dir(dir: &Path, device: &Device) -> Result<LoadedLfm2, ImportError> {
    let cfg_path = required_file(dir, "config.json")?;
    let weights = required_file(dir, "model.safetensors")?;
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| ImportError::Parse {
        file: cfg_path.clone(),
        reason: e.to_string(),
    })?;
    let config = Lfm2Config::from_json_bytes(&cfg_bytes).map_err(|reason| ImportError::Parse {
        file: cfg_path,
        reason,
    })?;

    // Cross-check the sibling metadata (when present) before touching weights:
    // tokenizer_config.json EOS agreement with config.json, a chat-template that
    // speaks LFM2.5's bracket-notation tool-call convention, and added-token ids
    // that match the real tokenizer.json — a repackaging mismatch fails loudly.
    let tokenizer_config = crate::tokenizer::validate_checkpoint_dir(
        dir,
        &config.eos_token_id.to_vec(),
        Some(crate::tok_config::ToolCallConvention::Lfm),
    )?;
    debug_assert!(
        tokenizer_config
            .as_ref()
            .and_then(crate::tok_config::TokenizerConfig::eos_id)
            .is_none_or(|id| config.eos_token_id.contains(id)),
        "validate_checkpoint_dir returned a config whose EOS disagrees with config.json"
    );

    let mut model = build(&config, device);
    // The float dtype comes from the DEVICE — burn 0.22 keeps the element
    // type there as a runtime setting, not on a backend type. Creation sites
    // still name it explicitly rather than riding the unspecified default.
    let target_float = crate::backend::float_dtype(device);
    let mut store = SafetensorsStore::from_file(weights.clone())
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(target_float)))
        .allow_partial(true)
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(self_attn)\.out_proj\.", "$1.o_proj.")
        .with_key_remapping(r"(self_attn)\.q_layernorm\.weight$", "$1.q_norm.gamma")
        .with_key_remapping(r"(self_attn)\.k_layernorm\.weight$", "$1.k_norm.gamma")
        .with_key_remapping(r"(feed_forward)\.w1\.", "$1.gate_proj.")
        .with_key_remapping(r"(feed_forward)\.w2\.", "$1.down_proj.")
        .with_key_remapping(r"(feed_forward)\.w3\.", "$1.up_proj.")
        .with_key_remapping(
            r"(operator_norm|ffn_norm|embedding_norm)\.weight$",
            "$1.gamma",
        );
    load_checked(&mut model, &mut store, &weights)?;
    Ok(LoadedLfm2 {
        model,
        config,
        tokenizer_config,
    })
}

/// GGUF (llama.cpp `lfm2` arch) tensor names → the HF checkpoint names the
/// safetensors remap chain already handles. The depthwise conv kernel is the
/// one shape special-case: llama.cpp stores it squeezed (`[k, channels]` in
/// ggml dims), the checkpoint as `[channels, 1, k]` — same bytes.
fn gguf_tensor_to_hf(info: &GgufTensorInfo) -> Option<GgufMap> {
    match info.name.as_str() {
        "token_embd.weight" => {
            return Some(GgufMap::Rename("model.embed_tokens.weight".into()));
        }
        "token_embd_norm.weight" => {
            return Some(GgufMap::Rename("model.embedding_norm.weight".into()));
        }
        _ => {}
    }
    let rest = info.name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if field == "shortconv.conv.weight" {
        // ggml dims [k, channels] → row-major [channels, k] → the
        // checkpoint's depthwise-Conv1d shape [channels, 1, k].
        let (&k, &channels) = (info.dims.first()?, info.dims.get(1)?);
        return Some(GgufMap::Reshape(
            format!("model.layers.{layer}.conv.conv.weight"),
            vec![channels, 1, k],
        ));
    }
    let mapped = match field {
        "attn_norm.weight" => "operator_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.out_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_layernorm.weight",
        "attn_k_norm.weight" => "self_attn.k_layernorm.weight",
        "ffn_gate.weight" => "feed_forward.w1.weight",
        "ffn_down.weight" => "feed_forward.w2.weight",
        "ffn_up.weight" => "feed_forward.w3.weight",
        "shortconv.in_proj.weight" => "conv.in_proj.weight",
        "shortconv.out_proj.weight" => "conv.out_proj.weight",
        _ => return None,
    };
    Some(GgufMap::Rename(format!("model.layers.{layer}.{mapped}")))
}

/// Load an LFM2/LFM2.5 model straight from a **GGUF** file: hyperparameters
/// from the `lfm2.*` metadata (layer kinds from the per-layer kv-head
/// array), weights dequantized and driven through the exact store pipeline
/// the safetensors path uses.
pub fn load_from_gguf(path: &Path, device: &Device) -> Result<LoadedLfm2, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = Lfm2Config::from_gguf(&f).map_err(parse)?;
    // The scratch guard (Some only when the payload went to disk) must
    // outlive `load_checked`: the store reads that file lazily.
    let (base, _scratch) = gguf_store(&f, &gguf_tensor_to_hf, DequantSink::Auto, device)?;

    let mut model = build(&config, device);
    let mut store = base
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(self_attn)\.out_proj\.", "$1.o_proj.")
        .with_key_remapping(r"(self_attn)\.q_layernorm\.weight$", "$1.q_norm.gamma")
        .with_key_remapping(r"(self_attn)\.k_layernorm\.weight$", "$1.k_norm.gamma")
        .with_key_remapping(r"(feed_forward)\.w1\.", "$1.gate_proj.")
        .with_key_remapping(r"(feed_forward)\.w2\.", "$1.down_proj.")
        .with_key_remapping(r"(feed_forward)\.w3\.", "$1.up_proj.")
        .with_key_remapping(
            r"(operator_norm|ffn_norm|embedding_norm)\.weight$",
            "$1.gamma",
        );
    load_checked(&mut model, &mut store, path)?;
    // A GGUF is self-contained — no sibling tokenizer_config.json in this path.
    Ok(LoadedLfm2 {
        model,
        config,
        tokenizer_config: None,
    })
}

impl CausalLm for LoadedLfm2 {
    type Cache = Vec<HybridKv>;

    fn is_eos(&self, id: u32) -> bool {
        self.config.eos_token_id.contains(id)
    }

    /// A fresh per-layer cache matching `layer_types`.
    fn new_cache(&self) -> Self::Cache {
        (0..self.config.num_hidden_layers)
            .map(|i| {
                if self.config.is_attention(i) {
                    HybridKv::Attn(None)
                } else {
                    HybridKv::Conv(None)
                }
            })
            .collect()
    }

    /// Forward `new_ids` (the whole prompt when `past == 0`, else one decode
    /// token), updating `cache`; returns logits for the last position `[1, vocab]`.
    fn forward(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Self::Cache,
        device: &Device,
    ) -> Tensor<2> {
        let t = new_ids.len();
        assert!(t >= 1, "LFM2 forward: need at least one token");
        assert!(
            cache.len() == self.config.num_hidden_layers,
            "LFM2 forward: cache has {} layers, model has {}",
            cache.len(),
            self.config.num_hidden_layers
        );
        let cfg = &self.config;

        // Dtype pinned to the backend TYPE, never the per-device policy.
        let ids32: Vec<i32> = new_ids.iter().map(|&i| i as i32).collect();
        let input = Tensor::<1, Int>::from_data(
            TensorData::new(ids32, [t]),
            (device, crate::backend::int_dtype(device)),
        )
        .reshape([1, t]);
        let mut x = self.model.embed_tokens.forward(input);

        let (cos, sin) = rope_tables(t, past, cfg.head_dim(), cfg.rope_theta, device);
        let mask = (t > 1).then(|| causal_mask(t, past, device));
        let kk = cfg.conv_l_cache;

        for (layer, kv) in self.model.layers.iter().zip(cache.iter_mut()) {
            let h = layer.operator_norm.forward(x.clone());
            let h = match (&layer.conv, &layer.self_attn, kv) {
                (Some(conv), None, HybridKv::Conv(state)) => conv.forward(h, kk, state),
                (None, Some(attn), HybridKv::Attn(kv_state)) => attn.forward(
                    h,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim(),
                    &cos,
                    &sin,
                    mask.as_ref(),
                    kv_state,
                ),
                // Layer kind and cache kind disagree — a caller bug.
                _ => unreachable!("LFM2 forward: layer/cache kind mismatch"),
            };
            x = x.add(h);
            let h2 = layer.ffn_norm.forward(x.clone());
            x = x.add(layer.feed_forward.forward(h2));
        }
        let x = self.model.embedding_norm.forward(x);
        let last = x.narrow(1, t - 1, 1).reshape([1, cfg.hidden_size]);
        let w = self.model.embed_tokens.weight.val(); // tied lm-head
        last.matmul(w.swap_dims(0, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-layer toy hybrid: conv, attention, conv.
    fn toy_config() -> Lfm2Config {
        Lfm2Config {
            vocab_size: 48,
            hidden_size: 16,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            norm_eps: 1e-5,
            rope_theta: 1e4,
            rope_parameters: None,
            rope_scaling: None,
            max_position_embeddings: Some(512),
            sliding_window: None,
            conv_l_cache: 3,
            block_ff_dim: 48,
            block_multiple_of: 8,
            block_auto_adjust_ff_dim: true,
            layer_types: vec!["conv".into(), "full_attention".into(), "conv".into()],
            eos_token_id: EosIds::One(7),
        }
    }

    #[test]
    fn ff_dim_auto_adjust_matches_reference_formula() {
        let cfg = toy_config();
        // 2*48/3 = 32, already a multiple of 8.
        assert_eq!(cfg.ff_dim(), 32);
        // The 1.2B's real numbers: 2/3 of 12288 = 8192, multiple_of 8192 → 8192.
        let mut big = toy_config();
        big.block_ff_dim = 12288;
        big.block_multiple_of = 8192;
        assert_eq!(big.ff_dim(), 8192);
    }

    #[test]
    fn config_rejects_layer_type_count_mismatch() {
        let mut cfg = toy_config();
        cfg.layer_types.pop();
        assert!(cfg.validate("test").is_err());
    }

    /// Both `rope_theta` spellings parse; anything but plain rotary is loud.
    #[test]
    fn config_reads_rope_theta_top_level_or_nested() {
        let base = r#"{
            "vocab_size": 48, "hidden_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "norm_eps": 1e-5, "conv_L_cache": 3,
            "block_ff_dim": 48, "block_multiple_of": 8,
            "layer_types": ["full_attention"]"#;
        let top = format!(r#"{base}, "rope_theta": 10000.0}}"#);
        assert_eq!(
            Lfm2Config::from_json_bytes(top.as_bytes())
                .unwrap()
                .rope_theta,
            1e4
        );
        // The 230M spelling: nested, with an explicit default rope_type.
        let nested = format!(
            r#"{base}, "rope_parameters": {{"rope_theta": 1000000.0, "rope_type": "default"}}}}"#
        );
        assert_eq!(
            Lfm2Config::from_json_bytes(nested.as_bytes())
                .unwrap()
                .rope_theta,
            1e6
        );
        // Negative space: a rope scheme we don't implement must not load,
        // and a config with neither spelling must not default silently.
        let yarn = format!(
            r#"{base}, "rope_parameters": {{"rope_theta": 1000000.0, "rope_type": "yarn"}}}}"#
        );
        assert!(Lfm2Config::from_json_bytes(yarn.as_bytes()).is_err());
        let missing = format!("{base}}}");
        assert!(Lfm2Config::from_json_bytes(missing.as_bytes()).is_err());
    }

    #[test]
    fn build_places_operators_by_layer_type() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let model = build(&cfg, &device);
        assert!(model.layers[0].conv.is_some() && model.layers[0].self_attn.is_none());
        assert!(model.layers[1].conv.is_none() && model.layers[1].self_attn.is_some());
        assert!(model.layers[2].conv.is_some() && model.layers[2].self_attn.is_none());
    }

    /// Whole-stack cache equivalence for the hybrid: prefill + cached decode
    /// equals one full forward, across BOTH cache kinds at once.
    #[test]
    fn toy_hybrid_cached_decode_matches_full_forward() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let loaded = LoadedLfm2 {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };

        let prompt: Vec<u32> = vec![5, 11, 2, 30];
        let mut cache = loaded.new_cache();
        let _ = loaded.forward(&prompt, 0, &mut cache, &device);
        let step = loaded
            .forward(&[19], prompt.len(), &mut cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let mut full_cache = loaded.new_cache();
        let all: Vec<u32> = prompt.iter().copied().chain([19]).collect();
        let full = loaded
            .forward(&all, 0, &mut full_cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        for (i, (c, f)) in step.iter().zip(&full).enumerate() {
            assert!((c - f).abs() < 1e-4, "logit {i}: cached {c} vs full {f}");
        }
    }

    #[tokio::test]
    async fn greedy_generate_respects_max_tokens_bound() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let loaded = LoadedLfm2 {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };
        let out = loaded.greedy_generate(&[1, 2], 3, &device).await.unwrap();
        assert!(out.len() <= 3);
    }

    /// A synthetic GGUF header shaped like the LFM2.5-1.2B file.
    fn toy_gguf() -> GgufFile {
        use crate::gguf::GgmlType;
        let meta = |k: &str, v: GgufValue| (k.to_string(), v);
        let kv_array = GgufValue::Array(
            [0u32, 2, 0]
                .iter()
                .map(|&v| GgufValue::U32(v))
                .collect::<Vec<_>>(),
        );
        GgufFile {
            path: std::path::PathBuf::new(),
            version: 3,
            metadata: vec![
                meta("general.architecture", GgufValue::Str("lfm2".into())),
                meta("lfm2.embedding_length", GgufValue::U32(16)),
                meta("lfm2.block_count", GgufValue::U32(3)),
                meta("lfm2.feed_forward_length", GgufValue::U32(32)),
                meta("lfm2.attention.head_count", GgufValue::U32(4)),
                meta("lfm2.attention.head_count_kv", kv_array),
                meta(
                    "lfm2.attention.layer_norm_rms_epsilon",
                    GgufValue::F32(1e-5),
                ),
                meta("lfm2.rope.freq_base", GgufValue::F32(1e6)),
                meta("lfm2.shortconv.l_cache", GgufValue::U32(3)),
                meta("tokenizer.ggml.eos_token_id", GgufValue::U32(7)),
            ],
            tensors: vec![GgufTensorInfo {
                name: "token_embd.weight".into(),
                dims: vec![16, 48],
                dtype: GgmlType::F32,
                offset: 0,
            }],
            alignment: 32,
            data_offset: 0,
        }
    }

    #[test]
    fn config_from_gguf_derives_layer_types_from_kv_array() {
        let cfg = Lfm2Config::from_gguf(&toy_gguf()).expect("parses");
        assert_eq!(
            cfg.layer_types,
            vec!["conv", "full_attention", "conv"],
            "0 = conv, nonzero = attention"
        );
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.vocab_size, 48); // from token_embd dims[1]
        assert_eq!(cfg.ff_dim(), 32); // GGUF value is pre-adjusted
        assert!(cfg.eos_token_id.contains(7));
    }

    #[test]
    fn config_from_gguf_rejects_mixed_kv_and_missing_array() {
        let mut f = toy_gguf();
        f.metadata[5].1 = GgufValue::Array(vec![
            GgufValue::U32(4),
            GgufValue::U32(2),
            GgufValue::U32(0),
        ]);
        assert!(Lfm2Config::from_gguf(&f).unwrap_err().contains("mixes"));

        let mut f = toy_gguf();
        f.metadata
            .retain(|(k, _)| k != "lfm2.attention.head_count_kv");
        assert!(Lfm2Config::from_gguf(&f).is_err());
    }

    #[test]
    fn gguf_names_map_onto_hf_checkpoint_names_incl_conv_reshape() {
        use crate::gguf::GgmlType;
        let info = |name: &str, dims: &[u64]| GgufTensorInfo {
            name: name.into(),
            dims: dims.to_vec(),
            dtype: GgmlType::F32,
            offset: 0,
        };
        let name_of = |i: &GgufTensorInfo| match gguf_tensor_to_hf(i) {
            Some(GgufMap::Rename(n)) => n,
            other => panic!("expected Rename, got {other:?}"),
        };
        assert_eq!(
            name_of(&info("blk.2.attn_q_norm.weight", &[64])),
            "model.layers.2.self_attn.q_layernorm.weight"
        );
        assert_eq!(
            name_of(&info("blk.0.shortconv.in_proj.weight", &[2048, 6144])),
            "model.layers.0.conv.in_proj.weight"
        );
        assert_eq!(
            name_of(&info("token_embd_norm.weight", &[2048])),
            "model.embedding_norm.weight"
        );
        // The conv kernel un-squeezes to the checkpoint's [channels, 1, k].
        match gguf_tensor_to_hf(&info("blk.0.shortconv.conv.weight", &[3, 2048])) {
            Some(GgufMap::Reshape(n, shape)) => {
                assert_eq!(n, "model.layers.0.conv.conv.weight");
                assert_eq!(shape, vec![2048, 1, 3]);
            }
            other => panic!("expected Reshape, got {other:?}"),
        }
        assert!(gguf_tensor_to_hf(&info("rope_freqs.weight", &[64])).is_none());
    }
}
