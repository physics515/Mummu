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
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo, GgufValue};
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

    /// The EOS ids as an owned list (empty when `None`) — the shape the
    /// `tokenizer_config.json` cross-check ([`crate::tok_config`]) consumes.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            Self::None => Vec::new(),
            Self::One(e) => vec![*e],
            Self::Many(v) => v.clone(),
        }
    }
}

/// A required GGUF metadata integer, as usize. Shared with the Qwen3 loader.
pub(crate) fn gguf_usize(f: &GgufFile, key: &str) -> Result<usize, String> {
    f.get(key)
        .and_then(GgufValue::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("missing or non-integer GGUF metadata '{key}'"))
}

/// A required GGUF metadata f32. Shared with the Qwen3 loader.
pub(crate) fn gguf_f32(f: &GgufFile, key: &str) -> Result<f32, String> {
    f.get(key)
        .and_then(GgufValue::as_f32)
        .ok_or_else(|| format!("missing or non-f32 GGUF metadata '{key}'"))
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

    /// Hyperparameters from a GGUF header's `qwen2.*` metadata — a GGUF file
    /// is self-contained, no `config.json` beside it. `vocab_size` comes from
    /// the embedding tensor (llama.cpp may pad it past the tokenizer vocab).
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "qwen2" {
            return Err(format!("GGUF architecture '{arch}' is not qwen2"));
        }
        let hidden_size = gguf_usize(f, "qwen2.embedding_length")?;
        let num_attention_heads = gguf_usize(f, "qwen2.attention.head_count")?;
        let embd = f
            .tensor("token_embd.weight")
            .ok_or("GGUF has no token_embd.weight tensor")?;
        if embd.dims.len() != 2 || embd.dims[0] != hidden_size as u64 {
            return Err(format!(
                "token_embd.weight dims {:?} do not match embedding_length {hidden_size}",
                embd.dims
            ));
        }
        let vocab_size = usize::try_from(embd.dims[1]).map_err(|_| "vocab too large")?;
        if let Some(tokens) = f.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array)
            && tokens.len() > vocab_size
        {
            return Err(format!(
                "tokenizer vocab {} exceeds embedding rows {vocab_size}",
                tokens.len()
            ));
        }
        let eos_token_id = f
            .get("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .map_or(EosIds::None, EosIds::One);
        let head_dim = f
            .get("qwen2.attention.key_length")
            .and_then(GgufValue::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(hidden_size / num_attention_heads);
        let cfg = Self {
            vocab_size,
            hidden_size,
            intermediate_size: gguf_usize(f, "qwen2.feed_forward_length")?,
            num_hidden_layers: gguf_usize(f, "qwen2.block_count")?,
            num_attention_heads,
            num_key_value_heads: gguf_usize(f, "qwen2.attention.head_count_kv")?,
            head_dim,
            rms_norm_eps: f64::from(gguf_f32(f, "qwen2.attention.layer_norm_rms_epsilon")?),
            rope_theta: gguf_f32(f, "qwen2.rope.freq_base")?,
            // No separate output.weight tensor means the lm-head is tied.
            tie_word_embeddings: f.tensor("output.weight").is_none(),
            eos_token_id,
        };
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

/// The Qwen2 decoder stack (HF's `model.*` subtree). The lm-head is tied on
/// the small tiers (0.5B/1.5B safetensors); untied checkpoints — the 7B, and
/// every llama.cpp GGUF (which materializes the head as `output.weight`, at
/// higher precision than the embedding) — carry it explicitly.
#[derive(Module, Debug)]
pub struct Qwen2<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Option<Linear<B>>,
}

/// A weight-loaded Qwen2 plus its config — everything a forward needs.
pub struct LoadedQwen2<B: Backend> {
    pub model: Qwen2<B>,
    pub config: Qwen2Config,
    /// The parsed sibling `tokenizer_config.json`, when one was present and
    /// well-formed beside a safetensors checkpoint (the load-time gate has
    /// already cross-checked its EOS against `config.json`). A consumer reads
    /// config-driven EOS/BOS/PAD ids from it (`eos_id()`, `bos_id()`, …). `None`
    /// for a GGUF load (self-contained; no sibling file) or a dir without one.
    pub tokenizer_config: Option<crate::tok_config::TokenizerConfig>,
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
    let lm_head = (!cfg.tie_word_embeddings).then(|| {
        LinearConfig::new(cfg.hidden_size, cfg.vocab_size)
            .with_bias(false)
            .init(device)
    });
    Qwen2 {
        embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
        layers,
        norm: norm(device),
        lm_head,
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

    // Cross-check the sibling metadata (when present) before touching weights:
    // tokenizer_config.json EOS agreement with config.json, a chat-template that
    // speaks Qwen2's Hermes/ChatML tool-call convention, and added-token ids that
    // match the real tokenizer.json — a repackaging mismatch fails loudly at load.
    let tokenizer_config = crate::tokenizer::validate_checkpoint_dir(
        dir,
        &config.eos_token_id.to_vec(),
        Some(crate::tok_config::ToolCallConvention::Hermes),
    )?;
    debug_assert!(
        tokenizer_config
            .as_ref()
            .and_then(crate::tok_config::TokenizerConfig::eos_id)
            .is_none_or(|id| config.eos_token_id.contains(id)),
        "validate_checkpoint_dir returned a config whose EOS disagrees with config.json"
    );

    let mut model = build::<B>(&config, device);
    // The backend's own float dtype (f32, or f16 on the GpuF16 alias), taken
    // from the TYPE (`B::FloatElem`), never from a probe tensor: unspecified-
    // dtype tensor creation follows the per-DEVICE default policy, which
    // another backend alias sharing the device may have flipped in-process.
    let target_float = <B::FloatElem as burn::tensor::Element>::dtype();
    let mut store = SafetensorsStore::from_file(weights.clone())
        .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
        .allow_partial(true)
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(input_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(post_attention_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"^norm\.weight$", "norm.gamma");
    load_checked(&mut model, &mut store, &weights)?;
    Ok(LoadedQwen2 {
        model,
        config,
        tokenizer_config,
    })
}

/// GGUF (llama.cpp) tensor names → the HF checkpoint names the safetensors
/// remap chain already handles. `None` for anything unrecognized — the blob
/// writer turns that into a loud error rather than dropping weights.
fn gguf_tensor_to_hf(info: &GgufTensorInfo) -> Option<GgufMap> {
    qwen2_gguf_name(&info.name).map(GgufMap::Rename)
}

fn qwen2_gguf_name(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let mapped = match field {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{mapped}"))
}

/// Load a Qwen2 model straight from a **GGUF** file (any dtype the dequant
/// suite covers — Q4_K_M, Q8_0, F16, …): hyperparameters from the GGUF
/// metadata, weights dequantized to f32 and driven through the exact store
/// pipeline (adapters + remaps + checked load) the safetensors path uses.
pub fn load_from_gguf<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<LoadedQwen2<B>, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = Qwen2Config::from_gguf(&f).map_err(parse)?;
    let blob = f
        .dequant_to_safetensors(&gguf_tensor_to_hf)
        .map_err(|e| parse(e.to_string()))?;
    assert!(blob.len() > 8, "a parsed GGUF yields a non-empty blob");

    let mut model = build::<B>(&config, device);
    // Type-level float dtype (`B::FloatElem`) — a probe tensor would follow
    // the per-DEVICE default policy, which another backend alias sharing the
    // device (Gpu vs GpuF16) may have flipped in this process.
    let target_float = <B::FloatElem as burn::tensor::Element>::dtype();
    let mut store = SafetensorsStore::from_bytes(Some(blob))
        .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
        .allow_partial(true)
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(input_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(post_attention_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"^norm\.weight$", "norm.gamma");
    load_checked(&mut model, &mut store, path)?;
    // A GGUF is self-contained — no sibling tokenizer_config.json in this path.
    Ok(LoadedQwen2 {
        model,
        config,
        tokenizer_config: None,
    })
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

        let last = x.narrow(1, t - 1, 1).reshape([1, cfg.hidden_size]);
        debug_assert!(
            self.model.lm_head.is_some() != cfg.tie_word_embeddings,
            "lm_head presence must match the config's tie flag"
        );
        match &self.model.lm_head {
            // Untied: the checkpoint's own head projection.
            Some(head) => head.forward(last), // [1, vocab]
            // Tied lm-head: logits = last_hidden @ embed_weight^T.
            None => {
                let w = self.model.embed_tokens.weight.val(); // [vocab, hidden]
                last.matmul(w.swap_dims(0, 1)) // [1, vocab]
            }
        }
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
            tokenizer_config: None,
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

    /// A synthetic in-memory GGUF header shaped like the Qwen2.5-1.5B file.
    fn toy_gguf() -> GgufFile {
        use crate::gguf::{GgmlType, GgufTensorInfo};
        let meta = |k: &str, v: GgufValue| (k.to_string(), v);
        GgufFile {
            path: std::path::PathBuf::new(),
            version: 3,
            metadata: vec![
                meta("general.architecture", GgufValue::Str("qwen2".into())),
                meta("qwen2.embedding_length", GgufValue::U32(16)),
                meta("qwen2.block_count", GgufValue::U32(2)),
                meta("qwen2.feed_forward_length", GgufValue::U32(32)),
                meta("qwen2.attention.head_count", GgufValue::U32(4)),
                meta("qwen2.attention.head_count_kv", GgufValue::U32(2)),
                meta(
                    "qwen2.attention.layer_norm_rms_epsilon",
                    GgufValue::F32(1e-6),
                ),
                meta("qwen2.rope.freq_base", GgufValue::F32(1e4)),
                meta("tokenizer.ggml.eos_token_id", GgufValue::U32(2)),
            ],
            tensors: vec![GgufTensorInfo {
                name: "token_embd.weight".into(),
                dims: vec![16, 64], // ggml order: [hidden, vocab]
                dtype: GgmlType::F32,
                offset: 0,
            }],
            alignment: 32,
            data_offset: 0,
        }
    }

    #[test]
    fn config_from_gguf_reads_metadata_and_embedding_dims() {
        let cfg = Qwen2Config::from_gguf(&toy_gguf()).expect("parses");
        assert_eq!(cfg.vocab_size, 64); // from token_embd dims[1]
        assert_eq!(cfg.hidden_size, 16);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.head_dim, 4); // derived: hidden / heads
        assert!(cfg.eos_token_id.contains(2));
        assert!(cfg.tie_word_embeddings); // no output.weight tensor
    }

    #[test]
    fn config_from_gguf_fails_loudly_on_missing_keys_and_wrong_arch() {
        let mut f = toy_gguf();
        f.metadata.retain(|(k, _)| k != "qwen2.block_count");
        let err = Qwen2Config::from_gguf(&f).unwrap_err();
        assert!(err.contains("qwen2.block_count"), "{err}");

        let mut f = toy_gguf();
        f.metadata[0].1 = GgufValue::Str("llama".into());
        assert!(Qwen2Config::from_gguf(&f).is_err());

        // Embedding dims that contradict the metadata are rejected.
        let mut f = toy_gguf();
        f.tensors[0].dims = vec![8, 64];
        assert!(Qwen2Config::from_gguf(&f).is_err());
    }

    #[test]
    fn untied_toy_config_builds_and_uses_an_lm_head() {
        let device = Dev::default();
        let mut cfg = toy_config();
        cfg.tie_word_embeddings = false;
        let vocab = cfg.vocab_size;
        let loaded = LoadedQwen2::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };
        assert!(loaded.model.lm_head.is_some());
        let mut cache = loaded.new_cache();
        let logits = loaded.forward(&[1, 2], 0, &mut cache, &device);
        assert_eq!(logits.dims(), [1, vocab]);
    }

    #[test]
    fn config_from_gguf_detects_an_untied_head() {
        use crate::gguf::{GgmlType, GgufTensorInfo};
        let mut f = toy_gguf();
        f.tensors.push(GgufTensorInfo {
            name: "output.weight".into(),
            dims: vec![16, 64],
            dtype: GgmlType::F32,
            offset: 4096,
        });
        let cfg = Qwen2Config::from_gguf(&f).expect("parses");
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn gguf_names_map_onto_hf_checkpoint_names() {
        assert_eq!(
            qwen2_gguf_name("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            qwen2_gguf_name("blk.27.attn_q.bias").as_deref(),
            Some("model.layers.27.self_attn.q_proj.bias")
        );
        assert_eq!(
            qwen2_gguf_name("blk.0.ffn_down.weight").as_deref(),
            Some("model.layers.0.mlp.down_proj.weight")
        );
        assert_eq!(
            qwen2_gguf_name("output_norm.weight").as_deref(),
            Some("model.norm.weight")
        );
        // Unknown names must map to None (the writer errors loudly).
        assert_eq!(qwen2_gguf_name("rope_freqs.weight"), None);
        assert_eq!(qwen2_gguf_name("blk.x.attn_q.weight"), None);
    }

    #[test]
    fn sanity_check_passes_on_a_live_toy_model_and_flags_a_vocab_mismatch() {
        let device = Dev::default();
        let cfg = toy_config();
        let vocab = cfg.vocab_size;
        let loaded = LoadedQwen2::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };
        // A built (random-weight) model computes a live, finite, non-degenerate
        // distribution — the smoke passes and reports a valid top id.
        let smoke = loaded
            .sanity_check(&[1, 2, 3], vocab, &device)
            .expect("live toy model passes the smoke");
        assert!((smoke.top_id as usize) < vocab, "top id in vocab range");
        assert!(smoke.spread > 0.0, "a live forward has positive spread");
        // The wrong expected vocab is caught as a mismatch, not a silent pass.
        assert!(loaded.sanity_check(&[1, 2, 3], vocab + 1, &device).is_err());
    }

    #[test]
    fn greedy_generate_respects_max_tokens_bound() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedQwen2::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };
        let out = loaded.greedy_generate(&[1, 2, 3], 4, &device).unwrap();
        assert!(out.len() <= 4);
    }
}
