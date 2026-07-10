//! Qwen2 / Qwen2.5 decoder, from scratch on the shared `nn` blocks:
//! Embedding → N×{RmsNorm, GQA attention (RoPE, KV cache), RmsNorm, SwiGLU}
//! → RmsNorm → tied lm-head. Config-driven — the 0.5B and 1.5B tiers (and any
//! other single-file Qwen2 checkpoint) load through the same code.
//!
//! Ported from laurelane's parity-validated implementation (single-forward
//! next-token logits byte-identical to Candle's `qwen2` on identical f32
//! weights, CPU and wgpu GPU). Mummu's own parity gate (P7) re-verifies here
//! before the port is marked trusted.

use std::path::Path;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::import::{CastFloatAdapter, ImportError, load_checked, required_file};
use crate::models::CausalLm;
use crate::nn::{
    GqaAttention, GqaAttentionConfig, LayerKv, SwiGluMlp, SwiGluMlpConfig, causal_mask, rope_tables,
};

/// Qwen2 architecture hyperparameters, read from the checkpoint's `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// EOS token id(s) — `<|im_end|>` first for the instruct checkpoints.
    #[serde(default)]
    pub eos_token_id: EosIds,
}

/// `eos_token_id` is a bare int in some checkpoints, a list in others.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    #[default]
    None,
    One(u32),
    Many(Vec<u32>),
}

impl EosIds {
    /// Is `id` one of the EOS ids?
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        match self {
            Self::None => false,
            Self::One(e) => *e == id,
            Self::Many(v) => v.contains(&id),
        }
    }
}

impl Qwen2Config {
    /// Parse `config.json` bytes; derives `head_dim` when absent.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cfg: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if cfg.head_dim == 0 {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
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
        if self.num_hidden_layers == 0 || self.vocab_size == 0 {
            return Err("num_hidden_layers and vocab_size must be positive".into());
        }
        Ok(())
    }
}

/// One decoder layer. Field names mirror the HF checkpoint so the key remap
/// stays trivial.
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    pub self_attn: GqaAttention<B>,
    pub mlp: SwiGluMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

/// The Qwen2 decoder stack (HF's `model.*` subtree; the lm-head is tied).
#[derive(Module, Debug)]
pub struct Qwen2<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

/// A weight-loaded Qwen2 plus its config — everything a forward needs.
pub struct LoadedQwen2<B: Backend> {
    pub model: Qwen2<B>,
    pub config: Qwen2Config,
}

fn build<B: Backend>(cfg: &Qwen2Config, device: &B::Device) -> Qwen2<B> {
    let attn_cfg = GqaAttentionConfig {
        hidden_size: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        bias: true,        // Qwen2 has q/k/v projection bias
        qk_norm_eps: None, // and no per-head q/k norm
    };
    let mlp_cfg = SwiGluMlpConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
    };
    let norm = |dev: &B::Device| {
        RmsNormConfig::new(cfg.hidden_size)
            .with_epsilon(cfg.rms_norm_eps)
            .init(dev)
    };
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| DecoderLayer {
            self_attn: attn_cfg.init(device),
            mlp: mlp_cfg.init(device),
            input_layernorm: norm(device),
            post_attention_layernorm: norm(device),
        })
        .collect();
    Qwen2 {
        embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
        layers,
        norm: norm(device),
    }
}

/// Build the architecture from `dir/config.json` and load
/// `dir/model.safetensors` into it, checked (no silent partial loads).
///
/// The key remap maps HF names onto our HF-mirroring field paths: strip the
/// `model.` prefix and rename RmsNorm `weight` → Burn's `gamma`
/// (`PyTorchToBurnAdapter` renames Layer/Batch/Group norm params but NOT
/// RmsNorm; it does transpose the Linear weights).
pub fn load_from_dir<B: Backend>(
    dir: &Path,
    device: &B::Device,
) -> Result<LoadedQwen2<B>, ImportError> {
    let cfg_path = required_file(dir, "config.json")?;
    let weights = required_file(dir, "model.safetensors")?;
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| ImportError::Parse {
        file: cfg_path.clone(),
        reason: e.to_string(),
    })?;
    let config = Qwen2Config::from_json_bytes(&cfg_bytes).map_err(|reason| ImportError::Parse {
        file: cfg_path,
        reason,
    })?;

    let mut model = build::<B>(&config, device);
    // The backend's own float dtype (f32, or f16 on the GpuF16 alias).
    let target_float = Tensor::<B, 1>::zeros([1], device).dtype();
    let mut store = SafetensorsStore::from_file(weights.clone())
        .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
        .allow_partial(true)
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(input_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(post_attention_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"^norm\.weight$", "norm.gamma");
    load_checked(&mut model, &mut store, &weights)?;
    Ok(LoadedQwen2 { model, config })
}

impl<B: Backend> CausalLm<B> for LoadedQwen2<B> {
    type Cache = Vec<LayerKv<B>>;

    fn new_cache(&self) -> Self::Cache {
        (0..self.config.num_hidden_layers).map(|_| None).collect()
    }

    fn is_eos(&self, id: u32) -> bool {
        self.config.eos_token_id.contains(id)
    }

    fn forward(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Self::Cache,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let t = new_ids.len();
        assert!(t >= 1, "Qwen2 forward: need at least one token");
        assert!(
            cache.len() == self.config.num_hidden_layers,
            "Qwen2 forward: cache has {} layers, model has {}",
            cache.len(),
            self.config.num_hidden_layers
        );
        let cfg = &self.config;

        // i32 token ids: native for wgpu and the flex CPU backend alike.
        let ids32: Vec<i32> = new_ids.iter().map(|&i| i as i32).collect();
        let input =
            Tensor::<B, 1, Int>::from_data(TensorData::new(ids32, [t]), device).reshape([1, t]);
        let mut x = self.model.embed_tokens.forward(input); // [1, t, hidden]

        let (cos, sin) = rope_tables::<B>(t, past, cfg.head_dim, cfg.rope_theta, device);
        // A single-token decode step needs no mask (the one query attends to
        // all cached keys); only a multi-token prefill needs the triangle.
        let mask = (t > 1).then(|| causal_mask::<B>(t, past, device));

        for (layer, kv) in self.model.layers.iter().zip(cache.iter_mut()) {
            let h = layer.input_layernorm.forward(x.clone());
            let h = layer.self_attn.forward(
                h,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                &cos,
                &sin,
                mask.as_ref(),
                kv,
            );
            x = x.add(h);
            let h2 = layer.post_attention_layernorm.forward(x.clone());
            x = x.add(layer.mlp.forward(h2));
        }
        let x = self.model.norm.forward(x);

        // Tied lm-head: logits = last_hidden @ embed_weight^T.
        let last = x.narrow(1, t - 1, 1).reshape([1, cfg.hidden_size]);
        let w = self.model.embed_tokens.weight.val(); // [vocab, hidden]
        last.matmul(w.swap_dims(0, 1)) // [1, vocab]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    type Dev = burn::tensor::Device<Cpu>;

    /// A synthetic 2-layer toy config: everything runs without real weights.
    fn toy_config() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 1e4,
            tie_word_embeddings: true,
            eos_token_id: EosIds::One(2),
        }
    }

    #[test]
    fn config_parses_hf_shape_and_derives_head_dim() {
        let json = br#"{
            "vocab_size": 151936, "hidden_size": 1536, "intermediate_size": 8960,
            "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true,
            "eos_token_id": 151645
        }"#;
        let cfg = Qwen2Config::from_json_bytes(json).unwrap();
        assert_eq!(cfg.head_dim, 128); // derived 1536/12
        assert!(cfg.eos_token_id.contains(151_645));
        assert!(!cfg.eos_token_id.contains(151_644));
    }

    #[test]
    fn config_rejects_indivisible_heads() {
        let json = br#"{
            "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 5, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0
        }"#;
        assert!(Qwen2Config::from_json_bytes(json).is_err());
    }

    #[test]
    fn eos_ids_parse_as_int_or_list() {
        let one: EosIds = serde_json::from_str("7").unwrap();
        let many: EosIds = serde_json::from_str("[7, 9]").unwrap();
        assert!(one.contains(7) && !one.contains(9));
        assert!(many.contains(7) && many.contains(9) && !many.contains(8));
    }

    /// The toy model decodes through the cache identically to full re-forwards
    /// — the whole-stack version of the nn-level equivalence tests.
    #[test]
    fn toy_model_cached_decode_matches_full_forward() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedQwen2::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
        };

        let prompt: Vec<u32> = vec![3, 14, 15, 9, 26];

        // Cached: prefill then one decode step for token at position 5.
        let mut cache = loaded.new_cache();
        let _ = loaded.forward(&prompt, 0, &mut cache, &device);
        let step = loaded
            .forward(&[42], prompt.len(), &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Full: all six tokens in one forward, no cache reuse.
        let mut full_cache = loaded.new_cache();
        let all: Vec<u32> = prompt.iter().copied().chain([42]).collect();
        let full = loaded
            .forward(&all, 0, &mut full_cache, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(step.len(), full.len());
        for (i, (c, f)) in step.iter().zip(&full).enumerate() {
            assert!((c - f).abs() < 1e-4, "logit {i}: cached {c} vs full {f}");
        }
    }

    #[test]
    fn greedy_generate_respects_max_tokens_bound() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedQwen2::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
        };
        let out = loaded.greedy_generate(&[1, 2, 3], 4, &device).unwrap();
        assert!(out.len() <= 4);
    }
}
