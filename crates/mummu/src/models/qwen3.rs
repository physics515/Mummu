//! Qwen3 dense decoder, from scratch on the shared `nn` blocks. Structurally
//! Qwen2 with three deltas the shared blocks already cover:
//!   * **per-head q/k RMSNorm** over `head_dim`, applied post-projection before
//!     RoPE — `GqaAttention`'s `qk_norm_eps` path (the same code the LFM2 port
//!     validated against Ollama; HF Qwen3 orders it identically:
//!     `q_norm(q_proj(x).view(b,t,nh,hd)).transpose(1,2)`);
//!   * **no q/k/v projection bias** (`attention_bias: false`);
//!   * a **decoupled `head_dim`** — `num_heads * head_dim` need not equal
//!     `hidden_size` (Qwen3-4B: 32·128 = 4096 vs hidden 2560), which
//!     `GqaAttentionConfig` already treats as independent.
//!
//! Everything else — RmsNorm, GQA + KV cache, SwiGLU, tied/untied lm-head — is
//! the shared stack, so this file is config + weight-key remaps only. The port
//! stays `[ ]` in the roadmap until Mummu's parity gate (P7) re-verifies it
//! against a same-weights reference; loading and decoding are proven here.

use std::path::Path;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo, GgufValue};
use crate::import::{CastFloatAdapter, ImportError, load_checked, required_file};
use crate::models::CausalLm;
use crate::models::qwen2::{EosIds, gguf_f32, gguf_usize};
use crate::nn::{
    GqaAttention, GqaAttentionConfig, LayerKv, SwiGluMlp, SwiGluMlpConfig, causal_mask, rope_tables,
};

/// Qwen3 architecture hyperparameters, read from the checkpoint's `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Qwen3 always ships `head_dim` explicitly (it is decoupled from
    /// `hidden_size / num_attention_heads`); we still derive it if absent.
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

impl Qwen3Config {
    /// Parse `config.json` bytes; derives `head_dim` when absent.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cfg: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if cfg.head_dim == 0 {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Hyperparameters from a GGUF header's `qwen3.*` metadata. Unlike Qwen2,
    /// `head_dim` is **required** metadata (`qwen3.attention.key_length`),
    /// because Qwen3's head_dim is decoupled — deriving it from
    /// `hidden / heads` is wrong for these checkpoints (4B: 80 ≠ 128).
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "qwen3" {
            return Err(format!("GGUF architecture '{arch}' is not qwen3"));
        }
        let hidden_size = gguf_usize(f, "qwen3.embedding_length")?;
        let num_attention_heads = gguf_usize(f, "qwen3.attention.head_count")?;
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
        // key_length is Qwen3's real head_dim; only fall back for a malformed
        // file, and let validate() catch an impossible result.
        let head_dim = gguf_usize(f, "qwen3.attention.key_length")
            .unwrap_or(hidden_size / num_attention_heads.max(1));
        let cfg = Self {
            vocab_size,
            hidden_size,
            intermediate_size: gguf_usize(f, "qwen3.feed_forward_length")?,
            num_hidden_layers: gguf_usize(f, "qwen3.block_count")?,
            num_attention_heads,
            num_key_value_heads: gguf_usize(f, "qwen3.attention.head_count_kv")?,
            head_dim,
            rms_norm_eps: f64::from(gguf_f32(f, "qwen3.attention.layer_norm_rms_epsilon")?),
            rope_theta: gguf_f32(f, "qwen3.rope.freq_base")?,
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
        if self.head_dim < 2 || !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "head_dim ({}) must be even and >= 2",
                self.head_dim
            ));
        }
        Ok(())
    }
}

/// One Qwen3 decoder layer. Field names mirror the HF checkpoint (the
/// `self_attn` submodule additionally carries `q_norm` / `k_norm`).
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    pub self_attn: GqaAttention<B>,
    pub mlp: SwiGluMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

/// The Qwen3 decoder stack (HF's `model.*` subtree). Tied on the small tiers
/// (0.6B/4B safetensors, and any GGUF without a separate `output.weight`).
#[derive(Module, Debug)]
pub struct Qwen3<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Option<Linear<B>>,
}

/// A weight-loaded Qwen3 plus its config — everything a forward needs.
pub struct LoadedQwen3<B: Backend> {
    pub model: Qwen3<B>,
    pub config: Qwen3Config,
}

fn build<B: Backend>(cfg: &Qwen3Config, device: &B::Device) -> Qwen3<B> {
    let attn_cfg = GqaAttentionConfig {
        hidden_size: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        bias: false,                         // Qwen3 dropped the q/k/v bias
        qk_norm_eps: Some(cfg.rms_norm_eps), // and added per-head q/k RMSNorm
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
    Qwen3 {
        embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
        layers,
        norm: norm(device),
        lm_head,
    }
}

/// The safetensors key remap: strip `model.`, rename every RmsNorm `weight` →
/// Burn's `gamma`. Qwen3 adds the per-head `self_attn.q_norm` / `k_norm` to the
/// set of norms the qwen2 chain already handled.
fn install_remaps(store: SafetensorsStore) -> SafetensorsStore {
    store
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(input_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(post_attention_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(self_attn\.q_norm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(self_attn\.k_norm)\.weight$", "$1.gamma")
        .with_key_remapping(r"^norm\.weight$", "norm.gamma")
}

/// Build from `dir/config.json` and load `dir/model.safetensors`, checked.
pub fn load_from_dir<B: Backend>(
    dir: &Path,
    device: &B::Device,
) -> Result<LoadedQwen3<B>, ImportError> {
    let cfg_path = required_file(dir, "config.json")?;
    let weights = required_file(dir, "model.safetensors")?;
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| ImportError::Parse {
        file: cfg_path.clone(),
        reason: e.to_string(),
    })?;
    let config = Qwen3Config::from_json_bytes(&cfg_bytes).map_err(|reason| ImportError::Parse {
        file: cfg_path,
        reason,
    })?;

    let mut model = build::<B>(&config, device);
    let target_float = Tensor::<B, 1>::zeros([1], device).dtype();
    let mut store = install_remaps(
        SafetensorsStore::from_file(weights.clone())
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
            .allow_partial(true),
    );
    load_checked(&mut model, &mut store, &weights)?;
    Ok(LoadedQwen3 { model, config })
}

/// GGUF (llama.cpp `qwen3` arch) tensor names → the HF checkpoint names the
/// safetensors remap chain handles. `None` for anything unrecognized.
fn gguf_tensor_to_hf(info: &GgufTensorInfo) -> Option<GgufMap> {
    qwen3_gguf_name(&info.name).map(GgufMap::Rename)
}

fn qwen3_gguf_name(name: &str) -> Option<String> {
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
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{mapped}"))
}

/// Load a Qwen3 model straight from a **GGUF** file: hyperparameters from the
/// `qwen3.*` metadata, weights dequantized to f32 and driven through the same
/// checked-load pipeline (adapters + remaps) the safetensors path uses.
pub fn load_from_gguf<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<LoadedQwen3<B>, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = Qwen3Config::from_gguf(&f).map_err(parse)?;
    let blob = f
        .dequant_to_safetensors(&gguf_tensor_to_hf)
        .map_err(|e| parse(e.to_string()))?;
    assert!(blob.len() > 8, "a parsed GGUF yields a non-empty blob");

    let mut model = build::<B>(&config, device);
    let target_float = Tensor::<B, 1>::zeros([1], device).dtype();
    let mut store = install_remaps(
        SafetensorsStore::from_bytes(Some(blob))
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
            .allow_partial(true),
    );
    load_checked(&mut model, &mut store, path)?;
    Ok(LoadedQwen3 { model, config })
}

impl<B: Backend> CausalLm<B> for LoadedQwen3<B> {
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
        assert!(t >= 1, "Qwen3 forward: need at least one token");
        assert!(
            cache.len() == self.config.num_hidden_layers,
            "Qwen3 forward: cache has {} layers, model has {}",
            cache.len(),
            self.config.num_hidden_layers
        );
        let cfg = &self.config;

        let ids32: Vec<i32> = new_ids.iter().map(|&i| i as i32).collect();
        let input =
            Tensor::<B, 1, Int>::from_data(TensorData::new(ids32, [t]), device).reshape([1, t]);
        let mut x = self.model.embed_tokens.forward(input); // [1, t, hidden]

        let (cos, sin) = rope_tables::<B>(t, past, cfg.head_dim, cfg.rope_theta, device);
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
            Some(head) => head.forward(last), // [1, vocab]
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
    use crate::gguf::{GgmlType, GgufTensorInfo};

    type Dev = burn::tensor::Device<Cpu>;

    /// A synthetic toy config with a **decoupled** head_dim (num_heads·head_dim
    /// = 4·6 = 24 ≠ hidden 16), exercising the Qwen3-specific shape path.
    fn toy_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 6,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            tie_word_embeddings: true,
            eos_token_id: EosIds::One(2),
        }
    }

    #[test]
    fn config_parses_qwen3_4b_shape() {
        // The real Qwen3-4B config.json shape: head_dim is explicit and
        // decoupled (32·128 = 4096 ≠ hidden 2560).
        let json = br#"{
            "vocab_size": 151936, "hidden_size": 2560, "intermediate_size": 9728,
            "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
            "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "tie_word_embeddings": true, "eos_token_id": 151645
        }"#;
        let cfg = Qwen3Config::from_json_bytes(json).unwrap();
        assert_eq!(cfg.head_dim, 128); // taken verbatim, NOT derived to 80
        assert_ne!(cfg.head_dim, cfg.hidden_size / cfg.num_attention_heads);
        assert!(cfg.eos_token_id.contains(151_645));
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn config_derives_head_dim_when_absent() {
        let json = br#"{
            "vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0
        }"#;
        let cfg = Qwen3Config::from_json_bytes(json).unwrap();
        assert_eq!(cfg.head_dim, 8); // 32 / 4
    }

    #[test]
    fn config_rejects_indivisible_heads_and_odd_head_dim() {
        let bad_heads = br#"{
            "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 5, "num_key_value_heads": 2,
            "head_dim": 4, "rms_norm_eps": 1e-6, "rope_theta": 1e6
        }"#;
        assert!(Qwen3Config::from_json_bytes(bad_heads).is_err());
        let odd_hd = br#"{
            "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 5, "rms_norm_eps": 1e-6, "rope_theta": 1e6
        }"#;
        assert!(Qwen3Config::from_json_bytes(odd_hd).is_err());
    }

    /// The load-bearing invariant: cached prefill+decode == one full forward.
    #[test]
    fn toy_model_cached_decode_matches_full_forward() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedQwen3::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
        };

        let prompt: Vec<u32> = vec![3, 14, 15, 9, 26];
        let mut cache = loaded.new_cache();
        let _ = loaded.forward(&prompt, 0, &mut cache, &device);
        let step = loaded
            .forward(&[42], prompt.len(), &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

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
    fn untied_toy_config_builds_and_uses_an_lm_head() {
        let device = Dev::default();
        let mut cfg = toy_config();
        cfg.tie_word_embeddings = false;
        let vocab = cfg.vocab_size;
        let loaded = LoadedQwen3::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
        };
        assert!(loaded.model.lm_head.is_some());
        let mut cache = loaded.new_cache();
        let logits = loaded.forward(&[1, 2], 0, &mut cache, &device);
        assert_eq!(logits.dims(), [1, vocab]);
    }

    /// A synthetic in-memory GGUF header shaped like a small `qwen3` file,
    /// with the decoupled `key_length` metadata.
    fn toy_gguf() -> GgufFile {
        let meta = |k: &str, v: GgufValue| (k.to_string(), v);
        GgufFile {
            path: std::path::PathBuf::new(),
            version: 3,
            metadata: vec![
                meta("general.architecture", GgufValue::Str("qwen3".into())),
                meta("qwen3.embedding_length", GgufValue::U32(16)),
                meta("qwen3.block_count", GgufValue::U32(2)),
                meta("qwen3.feed_forward_length", GgufValue::U32(32)),
                meta("qwen3.attention.head_count", GgufValue::U32(4)),
                meta("qwen3.attention.head_count_kv", GgufValue::U32(2)),
                meta("qwen3.attention.key_length", GgufValue::U32(6)),
                meta(
                    "qwen3.attention.layer_norm_rms_epsilon",
                    GgufValue::F32(1e-6),
                ),
                meta("qwen3.rope.freq_base", GgufValue::F32(1e6)),
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
    fn config_from_gguf_reads_decoupled_head_dim_from_key_length() {
        let cfg = Qwen3Config::from_gguf(&toy_gguf()).expect("parses");
        assert_eq!(cfg.vocab_size, 64);
        assert_eq!(cfg.hidden_size, 16);
        assert_eq!(cfg.num_hidden_layers, 2);
        // key_length (6), NOT hidden/heads (4) — the decoupled path.
        assert_eq!(cfg.head_dim, 6);
        assert!(cfg.eos_token_id.contains(2));
        assert!(cfg.tie_word_embeddings); // no output.weight tensor
    }

    #[test]
    fn config_from_gguf_fails_loudly_on_missing_keys_and_wrong_arch() {
        let mut f = toy_gguf();
        f.metadata.retain(|(k, _)| k != "qwen3.block_count");
        assert!(
            Qwen3Config::from_gguf(&f)
                .unwrap_err()
                .contains("block_count")
        );

        let mut f = toy_gguf();
        f.metadata[0].1 = GgufValue::Str("qwen2".into());
        assert!(Qwen3Config::from_gguf(&f).is_err());
    }

    #[test]
    fn config_from_gguf_detects_untied_head() {
        let mut f = toy_gguf();
        f.tensors.push(GgufTensorInfo {
            name: "output.weight".into(),
            dims: vec![16, 64],
            dtype: GgmlType::F32,
            offset: 4096,
        });
        assert!(!Qwen3Config::from_gguf(&f).unwrap().tie_word_embeddings);
    }

    #[test]
    fn gguf_names_map_including_qk_norms() {
        assert_eq!(
            qwen3_gguf_name("blk.0.attn_q_norm.weight").as_deref(),
            Some("model.layers.0.self_attn.q_norm.weight")
        );
        assert_eq!(
            qwen3_gguf_name("blk.35.attn_k_norm.weight").as_deref(),
            Some("model.layers.35.self_attn.k_norm.weight")
        );
        assert_eq!(
            qwen3_gguf_name("blk.7.attn_q.weight").as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
        assert_eq!(
            qwen3_gguf_name("output.weight").as_deref(),
            Some("lm_head.weight")
        );
        // Qwen3 has no q/k/v bias — a bias tensor is unrecognized (loud error).
        assert_eq!(qwen3_gguf_name("blk.0.attn_q.bias"), None);
        assert_eq!(qwen3_gguf_name("rope_freqs.weight"), None);
    }

    #[test]
    fn greedy_generate_respects_max_tokens_bound() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedQwen3::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
        };
        let out = loaded.greedy_generate(&[1, 2, 3], 4, &device).unwrap();
        assert!(out.len() <= 4);
    }
}
