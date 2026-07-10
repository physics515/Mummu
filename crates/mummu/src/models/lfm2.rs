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
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::import::{CastFloatAdapter, ImportError, load_checked, required_file};
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
    pub rope_theta: f32,
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

    /// Parse `config.json` bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let cfg: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
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
        Ok(())
    }
}

/// One hybrid layer: exactly one of `conv` / `self_attn` is present, per
/// `layer_types[i]`. Field names mirror the HF checkpoint.
#[derive(Module, Debug)]
pub struct HybridLayer<B: Backend> {
    pub conv: Option<ShortConv<B>>,
    pub self_attn: Option<GqaAttention<B>>,
    pub feed_forward: SwiGluMlp<B>,
    pub operator_norm: RmsNorm<B>,
    pub ffn_norm: RmsNorm<B>,
}

/// The LFM2 decoder stack (tied lm-head).
#[derive(Module, Debug)]
pub struct Lfm2<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<HybridLayer<B>>,
    pub embedding_norm: RmsNorm<B>,
}

/// Per-layer decode cache: conv layers roll the last `K-1` gated inputs,
/// attention layers keep the running k/v.
pub enum HybridKv<B: Backend> {
    Conv(ConvState<B>),
    Attn(LayerKv<B>),
}

/// A weight-loaded LFM2 plus its config.
pub struct LoadedLfm2<B: Backend> {
    pub model: Lfm2<B>,
    pub config: Lfm2Config,
}

fn build<B: Backend>(cfg: &Lfm2Config, device: &B::Device) -> Lfm2<B> {
    let attn_cfg = GqaAttentionConfig {
        hidden_size: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        bias: false,                     // LFM2 projections are bias-free
        qk_norm_eps: Some(cfg.norm_eps), // per-head q/k RMSNorm
    };
    let conv_cfg = ShortConvConfig {
        hidden_size: cfg.hidden_size,
        kernel_len: cfg.conv_l_cache,
    };
    let mlp_cfg = SwiGluMlpConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.ff_dim(),
    };
    let norm = |dev: &B::Device| {
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
pub fn load_from_dir<B: Backend>(
    dir: &Path,
    device: &B::Device,
) -> Result<LoadedLfm2<B>, ImportError> {
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

    let mut model = build::<B>(&config, device);
    let target_float = Tensor::<B, 1>::zeros([1], device).dtype();
    let mut store = SafetensorsStore::from_file(weights.clone())
        .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
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
    Ok(LoadedLfm2 { model, config })
}

impl<B: Backend> CausalLm<B> for LoadedLfm2<B> {
    type Cache = Vec<HybridKv<B>>;

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
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let t = new_ids.len();
        assert!(t >= 1, "LFM2 forward: need at least one token");
        assert!(
            cache.len() == self.config.num_hidden_layers,
            "LFM2 forward: cache has {} layers, model has {}",
            cache.len(),
            self.config.num_hidden_layers
        );
        let cfg = &self.config;

        let ids32: Vec<i32> = new_ids.iter().map(|&i| i as i32).collect();
        let input =
            Tensor::<B, 1, Int>::from_data(TensorData::new(ids32, [t]), device).reshape([1, t]);
        let mut x = self.model.embed_tokens.forward(input);

        let (cos, sin) = rope_tables::<B>(t, past, cfg.head_dim(), cfg.rope_theta, device);
        let mask = (t > 1).then(|| causal_mask::<B>(t, past, device));
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
    use crate::backend::Cpu;

    type Dev = burn::tensor::Device<Cpu>;

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
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn build_places_operators_by_layer_type() {
        let device = Dev::default();
        let cfg = toy_config();
        let model = build::<Cpu>(&cfg, &device);
        assert!(model.layers[0].conv.is_some() && model.layers[0].self_attn.is_none());
        assert!(model.layers[1].conv.is_none() && model.layers[1].self_attn.is_some());
        assert!(model.layers[2].conv.is_some() && model.layers[2].self_attn.is_none());
    }

    /// Whole-stack cache equivalence for the hybrid: prefill + cached decode
    /// equals one full forward, across BOTH cache kinds at once.
    #[test]
    fn toy_hybrid_cached_decode_matches_full_forward() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedLfm2::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
        };

        let prompt: Vec<u32> = vec![5, 11, 2, 30];
        let mut cache = loaded.new_cache();
        let _ = loaded.forward(&prompt, 0, &mut cache, &device);
        let step = loaded
            .forward(&[19], prompt.len(), &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let mut full_cache = loaded.new_cache();
        let all: Vec<u32> = prompt.iter().copied().chain([19]).collect();
        let full = loaded
            .forward(&all, 0, &mut full_cache, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        for (i, (c, f)) in step.iter().zip(&full).enumerate() {
            assert!((c - f).abs() < 1e-4, "logit {i}: cached {c} vs full {f}");
        }
    }

    #[test]
    fn greedy_generate_respects_max_tokens_bound() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedLfm2::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
        };
        let out = loaded.greedy_generate(&[1, 2], 3, &device).unwrap();
        assert!(out.len() <= 3);
    }
}
