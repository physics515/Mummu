//! Qwen3 dense decoder, from scratch on the shared `nn` blocks. Structurally
//! Qwen2 with three deltas the shared blocks already cover:
//!   * **per-head q/k `RMSNorm`** over `head_dim`, applied post-projection before
//!     `RoPE` — `GqaAttention`'s `qk_norm_eps` path (the same code the LFM2 port
//!     validated against Ollama; HF Qwen3 orders it identically:
//!     `q_norm(q_proj(x).view(b,t,nh,hd)).transpose(1,2)`);
//!   * **no q/k/v projection bias** (`attention_bias: false`);
//!   * a **decoupled `head_dim`** — `num_heads * head_dim` need not equal
//!     `hidden_size` (Qwen3-4B: 32·128 = 4096 vs hidden 2560), which
//!     `GqaAttentionConfig` already treats as independent.
//!
//! Everything else — `RmsNorm`, GQA + KV cache, `SwiGLU`, tied/untied lm-head — is
//! the shared stack, so this file is config + weight-key remaps only. The port
//! stays `[ ]` in the roadmap until Mummu's parity gate (P7) re-verifies it
//! against a same-weights reference; loading and decoding are proven here.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::attn_config::{RopeScaling, check_sliding_window, sliding_window_from_gguf};
use crate::embed::Pooling;
use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo, GgufValue};
use crate::import::{
    DequantSink, FloatCastAdapter, ImportError, gguf_store, load_checked, load_checked_shards,
    required_file,
};
use crate::models::CausalLm;
use crate::models::qwen2::{EosIds, gguf_f32, gguf_usize};
use crate::nn::static_kv::{StaticKv, StaticKvConfig};
use crate::nn::{
    GqaAttention, GqaAttentionConfig, HeadShape, LayerKv, SwiGluMlp, SwiGluMlpConfig, causal_mask,
    rope_tables,
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
    /// Frequency scaling (`YaRN` / linear / …). `null` on the Qwen3 checkpoints
    /// in the zoo; a scaled one is refused at load rather than answered wrong
    /// ([`crate::attn_config`]).
    /// `rope_parameters` is the same object under the name newer transformers
    /// writes; reading only `rope_scaling` would let a freshly-serialized
    /// scaled checkpoint through as unscaled.
    #[serde(default, alias = "rope_parameters")]
    pub rope_scaling: Option<RopeScaling>,
    /// The trained context length (Qwen3-0.6B: 40 960), used to tell an inert
    /// sliding window from a clipping one.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// Declared window span — `null` on Qwen3-0.6B, and gated by
    /// `use_sliding_window` regardless.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// EOS token id(s) — `<|im_end|>` first for the instruct checkpoints.
    #[serde(default)]
    pub eos_token_id: EosIds,
}

impl Qwen3Config {
    /// Parse `config.json` bytes; derives `head_dim` when absent.
    ///
    /// # Errors
    ///
    /// Returns the JSON error as a string, or an error when validation
    /// refuses the config: a non-plain rope scaling, a live (clipping)
    /// sliding window, a head count that is not a positive multiple of the
    /// KV heads, zero layers or vocab, or an odd `head_dim` below 2.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cfg: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if cfg.head_dim == 0 {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        cfg.validate("qwen3 config.json")?;
        Ok(cfg)
    }

    /// Hyperparameters from a GGUF header's `qwen3.*` metadata. Unlike Qwen2,
    /// `head_dim` is **required** metadata (`qwen3.attention.key_length`),
    /// because Qwen3's `head_dim` is decoupled — deriving it from
    /// `hidden / heads` is wrong for these checkpoints (4B: 80 ≠ 128).
    ///
    /// # Errors
    ///
    /// Returns an error when the architecture is not `qwen3`, a required
    /// `qwen3.*` key is missing or not an integer/float, `token_embd.weight`
    /// is absent or does not match `embedding_length`, the tokenizer vocab
    /// exceeds the embedding rows, or the resulting config fails the same
    /// validation as [`Self::from_json_bytes`].
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
            .unwrap_or_else(|_| hidden_size / num_attention_heads.max(1));
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
            rope_scaling: RopeScaling::from_gguf(f, "qwen3"),
            max_position_embeddings: f
                .get("qwen3.context_length")
                .and_then(GgufValue::as_u64)
                .and_then(|v| usize::try_from(v).ok()),
            sliding_window: sliding_window_from_gguf(f, "qwen3"),
            // A GGUF header has no `use_sliding_window` twin: llama.cpp writes
            // the key only for architectures that window, so presence enables.
            use_sliding_window: true,
            // No separate output.weight tensor means the lm-head is tied.
            tie_word_embeddings: f.tensor("output.weight").is_none(),
            eos_token_id,
        };
        cfg.validate("qwen3 GGUF header")?;
        Ok(cfg)
    }

    fn validate(&self, whose: &str) -> Result<(), String> {
        if let Some(scaling) = &self.rope_scaling {
            scaling.check(whose)?;
        }
        check_sliding_window(
            self.sliding_window,
            self.use_sliding_window,
            self.max_position_embeddings,
            whose,
        )?;
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
pub struct DecoderLayer {
    pub self_attn: GqaAttention,
    pub mlp: SwiGluMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

/// The Qwen3 decoder stack (HF's `model.*` subtree). Tied on the small tiers
/// (0.6B/4B safetensors, and any GGUF without a separate `output.weight`).
#[derive(Module, Debug)]
pub struct Qwen3 {
    pub embed_tokens: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Option<Linear>,
}

/// A weight-loaded Qwen3 plus its config — everything a forward needs.
pub struct LoadedQwen3 {
    pub model: Qwen3,
    pub config: Qwen3Config,
    /// The parsed sibling `tokenizer_config.json`, when one was present and
    /// well-formed beside a safetensors checkpoint (the load-time gate has
    /// already cross-checked its EOS against `config.json`). A consumer reads
    /// config-driven EOS/BOS/PAD ids from it (`eos_id()`, `bos_id()`, …). `None`
    /// for a GGUF load (self-contained; no sibling file) or a dir without one.
    pub tokenizer_config: Option<crate::tok_config::TokenizerConfig>,
}

fn build(cfg: &Qwen3Config, device: &Device) -> Qwen3 {
    build_with_head(cfg, device, !cfg.tie_word_embeddings)
}

/// [`build`] with the lm-head's presence chosen by the caller: a separate
/// head when `untied_head`, none otherwise. The trunk-only encoder load
/// passes `false` whatever the tie flag says — it never reads the head, and
/// an embedding checkpoint saved as `Qwen3Model` (Qwen3-Embedding-8B) declares
/// an untied head it does not ship.
fn build_with_head(cfg: &Qwen3Config, device: &Device, untied_head: bool) -> Qwen3 {
    let attn_cfg = GqaAttentionConfig {
        hidden_size: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        bias: false,                         // Qwen3 dropped the q/k/v bias
        qk_norm_eps: Some(cfg.rms_norm_eps), // and added per-head q/k RMSNorm
        qk_norm_projection: false,
    };
    let mlp_cfg = SwiGluMlpConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
    };
    let norm = |dev: &Device| {
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
    let lm_head = untied_head.then(|| {
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

/// The safetensors key remap: strip `model.`, rename every `RmsNorm` `weight` →
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

/// Every safetensors shard of the checkpoint in `dir`: the one
/// `model.safetensors`, or the shards `model.safetensors.index.json` names
/// (Qwen3-4B and up ship split — the 4B reranker in two shards, the 8B
/// embedder in four).
fn weight_shards(dir: &Path) -> Result<Vec<PathBuf>, ImportError> {
    crate::safetensors::checkpoint_shards(dir).map_err(|e| match e {
        crate::safetensors::SafetensorsError::NoCheckpoint(_) => {
            ImportError::MissingFile(dir.join("model.safetensors"))
        }
        other => ImportError::Parse {
            file: dir.to_path_buf(),
            reason: other.to_string(),
        },
    })
}

/// `dir/config.json`, parsed and validated.
fn read_config(dir: &Path) -> Result<Qwen3Config, ImportError> {
    let cfg_path = required_file(dir, "config.json")?;
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| ImportError::Parse {
        file: cfg_path.clone(),
        reason: e.to_string(),
    })?;
    Qwen3Config::from_json_bytes(&cfg_bytes).map_err(|reason| ImportError::Parse {
        file: cfg_path,
        reason,
    })
}

/// Load `shards` into `model` through the remap chain, checked across all
/// of them.
fn load_shards(model: &mut Qwen3, shards: &[PathBuf], device: &Device) -> Result<(), ImportError> {
    // The float dtype comes from the DEVICE — burn 0.22 keeps the element
    // type there as a runtime setting, not on a backend type. Creation sites
    // still name it explicitly rather than riding the unspecified default.
    let target_float = crate::backend::float_dtype(device);
    let stores = shards
        .iter()
        .map(|path| {
            let store = install_remaps(
                SafetensorsStore::from_file(path.clone())
                    .with_from_adapter(
                        PyTorchToBurnAdapter.chain(FloatCastAdapter::to(target_float)),
                    )
                    .allow_partial(true),
            );
            (store, path.clone())
        })
        .collect();
    load_checked_shards(model, stores)
}

/// Build from `dir/config.json` and load the safetensors checkpoint beside
/// it (one `model.safetensors` or an indexed set of shards), checked.
///
/// # Errors
///
/// Returns an [`ImportError`] when `config.json` or the weights are
/// missing, when the config is unreadable or invalid, when the sibling
/// tokenizer metadata contradicts it, or when the checked load finds the
/// checkpoint incomplete or mismatched.
pub fn load_from_dir(dir: &Path, device: &Device) -> Result<LoadedQwen3, ImportError> {
    let _ = required_file(dir, "config.json")?;
    let shards = weight_shards(dir)?;
    let config = read_config(dir)?;

    // Cross-check the sibling metadata (when present) before touching weights:
    // tokenizer_config.json's EOS must agree with config.json's, its chat-template
    // must not speak a different tool-call convention than Qwen3's Hermes/ChatML
    // renderer, and every added-token id it declares must match the real
    // tokenizer.json. A repackaging mismatch fails loudly here rather than
    // mis-stopping / mis-templating / mis-tokenizing later.
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

    let mut model = build(&config, device);
    load_shards(&mut model, &shards, device)?;
    Ok(LoadedQwen3 {
        model,
        config,
        tokenizer_config,
    })
}

/// Load a Qwen3 checkpoint as an **encoder**: the trunk alone, no lm-head.
///
/// What an embedding checkpoint is — `Qwen3-Embedding-*` and
/// `harrier-oss-v1-0.6b` are saved as `Qwen3Model` (no `model.` prefix, no
/// `lm_head.weight`), and the 8B declares `tie_word_embeddings: false` for a
/// head it does not ship, so the generative [`load_from_dir`] would refuse
/// it as incomplete. A causal-LM checkpoint loads here too; its head is
/// simply not read.
///
/// The tokenizer metadata gate is [`crate::tokenizer::validate_tokenizer_ids`]
/// rather than the generative one: an encoder never stops on an EOS or
/// renders a chat (see that function for the real checkpoint this admits).
///
/// # Errors
///
/// As [`load_from_dir`], minus the EOS / chat-template agreement checks.
pub fn load_trunk_from_dir(dir: &Path, device: &Device) -> Result<Qwen3Trunk, ImportError> {
    let _ = required_file(dir, "config.json")?;
    let shards = weight_shards(dir)?;
    let config = read_config(dir)?;
    let tokenizer_config = crate::tokenizer::validate_tokenizer_ids(dir)?;
    let mut model = build_with_head(&config, device, false);
    load_shards(&mut model, &shards, device)?;
    Ok(Qwen3Trunk {
        model,
        config,
        tokenizer_config,
    })
}

/// [`load_trunk_from_dir`] for a single **GGUF** file (the `*-Embedding-GGUF`
/// releases). An `output.weight` in the file is left unread.
///
/// # Errors
///
/// As [`load_from_gguf`].
pub fn load_trunk_from_gguf(path: &Path, device: &Device) -> Result<Qwen3Trunk, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = Qwen3Config::from_gguf(&f).map_err(parse)?;
    let (base, _scratch) = gguf_store(&f, &gguf_tensor_to_hf, DequantSink::Auto, device)?;
    let mut model = build_with_head(&config, device, false);
    let mut store = install_remaps(base);
    load_checked(&mut model, &mut store, path)?;
    Ok(Qwen3Trunk {
        model,
        config,
        tokenizer_config: None,
    })
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

/// Load a Qwen3 model straight from a **GGUF** file.
///
/// Hyperparameters come from the `qwen3.*` metadata, weights are dequantized
/// to f32 and driven through the same checked-load pipeline (adapters +
/// remaps) the safetensors path uses.
///
/// # Errors
///
/// Returns an [`ImportError`] when the file cannot be opened or parsed as
/// a GGUF, when its metadata fails [`Qwen3Config::from_gguf`], when a
/// tensor name is unmapped or the dequant fails, or when the checked load
/// finds the checkpoint incomplete or mismatched.
pub fn load_from_gguf(path: &Path, device: &Device) -> Result<LoadedQwen3, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = Qwen3Config::from_gguf(&f).map_err(parse)?;
    // The scratch guard (Some only when the payload went to disk) must
    // outlive `load_checked`: the store reads that file lazily.
    let (base, _scratch) = gguf_store(&f, &gguf_tensor_to_hf, DequantSink::Auto, device)?;

    let mut model = build(&config, device);
    let mut store = install_remaps(base);
    load_checked(&mut model, &mut store, path)?;
    // A GGUF is self-contained — no sibling tokenizer_config.json in this path.
    Ok(LoadedQwen3 {
        model,
        config,
        tokenizer_config: None,
    })
}

impl CausalLm for LoadedQwen3 {
    type Cache = Vec<LayerKv>;

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
        device: &Device,
    ) -> Tensor<2> {
        let t = new_ids.len();
        let cfg = &self.config;
        let x = trunk(&self.model, cfg, new_ids, past, cache, device);
        let last = x.narrow(1, t - 1, 1).reshape([1, cfg.hidden_size]);
        debug_assert!(
            self.model.lm_head.is_some() != cfg.tie_word_embeddings,
            "lm_head presence must match the config's tie flag"
        );
        if let Some(head) = &self.model.lm_head {
            head.forward(last) // [1, vocab]
        } else {
            let w = self.model.embed_tokens.weight.val(); // [vocab, hidden]
            last.matmul(w.swap_dims(0, 1)) // [1, vocab]
        }
    }
}

/// The decoder trunk — embed, every layer, final norm — over `new_ids` at
/// positions `past..past + t`, updating `cache`. Returns the normed hidden
/// state of every new position, `[1, t, hidden]`: what the lm-head reads,
/// and what an embedder pools.
///
/// # Panics
///
/// When `new_ids` is empty, a token id does not fit `i32`, or `cache` does
/// not hold one entry per layer.
fn trunk(
    model: &Qwen3,
    cfg: &Qwen3Config,
    new_ids: &[u32],
    past: usize,
    cache: &mut [LayerKv],
    device: &Device,
) -> Tensor<3> {
    let t = new_ids.len();
    assert!(t >= 1, "Qwen3 forward: need at least one token");
    assert!(
        cache.len() == cfg.num_hidden_layers,
        "Qwen3 forward: cache has {} layers, model has {}",
        cache.len(),
        cfg.num_hidden_layers
    );

    // Dtype pinned to the backend TYPE, never the per-device policy.
    let ids32: Vec<i32> = new_ids
        .iter()
        .map(|&i| i32::try_from(i).expect("token id fits i32"))
        .collect();
    let input = Tensor::<1, Int>::from_data(
        TensorData::new(ids32, [t]),
        (device, crate::backend::int_dtype(device)),
    )
    .reshape([1, t]);
    let mut x = model.embed_tokens.forward(input); // [1, t, hidden]

    let (cos, sin) = rope_tables(t, past, cfg.head_dim, cfg.rope_theta, device);
    let mask = (t > 1).then(|| causal_mask(t, past, device));

    for (layer, kv) in model.layers.iter().zip(cache.iter_mut()) {
        let h = layer.input_layernorm.forward(x.clone());
        let shape = HeadShape {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        };
        let h = layer
            .self_attn
            .forward(h, shape, &cos, &sin, mask.as_ref(), kv);
        x = x.add(h);
        let h2 = layer.post_attention_layernorm.forward(x.clone());
        x = x.add(layer.mlp.forward(h2));
    }
    model.norm.forward(x)
}

/// Prefill chunk for [`pooled_hidden`]: long inputs run through the KV cache
/// this many tokens at a time, so the attention score matrix stays
/// `chunk x (past + chunk)` instead of growing with the square of the input.
const POOL_CHUNK: usize = 512;

/// One `[1, hidden]` vector for the whole of `ids`, pooled from the trunk's
/// final hidden states as `pooling` says. Prefills in [`POOL_CHUNK`] pieces
/// through a throwaway cache (exact: chunked prefill ≡ one-shot, the same
/// invariant the decode driver rests on).
///
/// # Panics
///
/// When `ids` is empty or a token id does not fit `i32`.
fn pooled_hidden(
    model: &Qwen3,
    cfg: &Qwen3Config,
    ids: &[u32],
    pooling: Pooling,
    device: &Device,
) -> Tensor<2> {
    assert!(!ids.is_empty(), "pooled_hidden: empty input");
    let mut cache: Vec<LayerKv> = (0..cfg.num_hidden_layers).map(|_| None).collect();
    let mut past = 0;
    let mut first: Option<Tensor<2>> = None;
    let mut sum: Option<Tensor<2>> = None;
    let mut last: Option<Tensor<2>> = None;
    for chunk in ids.chunks(POOL_CHUNK) {
        let x = trunk(model, cfg, chunk, past, &mut cache, device); // [1, t, h]
        let t = chunk.len();
        match pooling {
            Pooling::Cls => {
                if first.is_none() {
                    first = Some(x.narrow(1, 0, 1).reshape([1, cfg.hidden_size]));
                }
            }
            Pooling::Mean => {
                let chunk_sum = x.sum_dim(1).reshape([1, cfg.hidden_size]);
                sum = Some(match sum {
                    None => chunk_sum,
                    Some(s) => s.add(chunk_sum),
                });
            }
            Pooling::LastToken => {
                last = Some(x.narrow(1, t - 1, 1).reshape([1, cfg.hidden_size]));
            }
        }
        past += t;
    }
    match pooling {
        Pooling::Cls => first.expect("at least one chunk"),
        Pooling::Mean => sum
            .expect("at least one chunk")
            .div_scalar(mummu_num::f32_from_usize(ids.len())),
        Pooling::LastToken => last.expect("at least one chunk"),
    }
}

impl LoadedQwen3 {
    /// Logits for the vocabulary rows `rows` only, at `hidden` (`[1, hidden]`,
    /// a final-normed state): `[1, rows.len()]`, in `rows` order.
    ///
    /// The head projection restricted to the rows a caller will read — a
    /// yes/no relevance judgement reads two of Qwen3's 151 669, so the full
    /// head is ~75 000x more work than the answer needs. Mathematically the
    /// same numbers as those columns of [`CausalLm::forward`]'s logits.
    ///
    /// # Panics
    ///
    /// When `rows` is empty, or a row is not a vocabulary index.
    #[must_use]
    pub fn head_rows(&self, hidden: Tensor<2>, rows: &[u32], device: &Device) -> Tensor<2> {
        assert!(!rows.is_empty(), "head_rows: no rows");
        assert!(
            rows.iter()
                .all(|&r| usize::try_from(r).is_ok_and(|r| r < self.config.vocab_size)),
            "head_rows: a row is outside the vocabulary ({})",
            self.config.vocab_size
        );
        let idx: Vec<i32> = rows
            .iter()
            .map(|&r| i32::try_from(r).expect("row fits i32"))
            .collect();
        let idx = Tensor::<1, Int>::from_data(
            TensorData::new(idx, [rows.len()]),
            (device, crate::backend::int_dtype(device)),
        );
        if let Some(head) = &self.model.lm_head {
            // burn's Linear stores [d_in, d_out] = [hidden, vocab].
            hidden.matmul(head.weight.val().select(1, idx))
        } else {
            let w = self.model.embed_tokens.weight.val(); // [vocab, hidden]
            hidden.matmul(w.select(0, idx).swap_dims(0, 1))
        }
    }

    /// The final-normed hidden state at the last position of `ids`,
    /// `[1, hidden]` — the input [`Self::head_rows`] projects.
    ///
    /// # Panics
    ///
    /// When `ids` is empty or a token id does not fit `i32`.
    #[must_use]
    pub fn last_hidden(&self, ids: &[u32], device: &Device) -> Tensor<2> {
        pooled_hidden(&self.model, &self.config, ids, Pooling::LastToken, device)
    }
}

impl LoadedQwen3 {
    const fn static_kv_config(&self, slots: usize, max_ctx: usize) -> StaticKvConfig {
        StaticKvConfig {
            slots,
            max_ctx,
            layers: self.config.num_hidden_layers,
            kv_heads: self.config.num_key_value_heads,
            head_dim: self.config.head_dim,
            rope_dim: self.config.head_dim,
            rope_theta: self.config.rope_theta,
        }
    }
}

impl crate::capture::StaticDecode for LoadedQwen3 {
    type State = StaticKv;

    /// One device holds the whole model; the step must run there.
    fn static_supported(&self, device: &Device) -> bool {
        self.model.embed_tokens.weight.val().device() == *device
    }

    fn static_bytes(&self, slots: usize, max_ctx: usize, device: &Device) -> u64 {
        StaticKv::bytes(self.static_kv_config(slots, max_ctx), device)
    }

    fn static_state(&self, slots: usize, max_ctx: usize, device: &Device) -> StaticKv {
        StaticKv::new(self.static_kv_config(slots, max_ctx), device)
    }

    fn trained_context(&self) -> Option<usize> {
        self.config.max_position_embeddings
    }

    fn forward_static(
        &self,
        tokens: &Tensor<2, Int>,
        positions: &Tensor<1, Int>,
        kv: &mut StaticKv,
        len: usize,
    ) -> Tensor<2> {
        let cfg = &self.config;
        let [slots, one] = tokens.dims();
        assert_eq!(one, 1, "forward_static: one token per slot");
        let step = kv.step_inputs(positions, len);
        let mut x = self.model.embed_tokens.forward(tokens.clone()); // [slots, 1, hidden]
        let shape = HeadShape {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        };
        for (l, layer) in self.model.layers.iter().enumerate() {
            let h = layer.input_layernorm.forward(x.clone());
            let h = layer
                .self_attn
                .forward_static(h, shape, &step, kv.layer_mut(l));
            x = x.add(h);
            let h2 = layer.post_attention_layernorm.forward(x.clone());
            x = x.add(layer.mlp.forward(h2));
        }
        let last = self.model.norm.forward(x).reshape([slots, cfg.hidden_size]);
        if let Some(head) = &self.model.lm_head {
            head.forward(last)
        } else {
            let w = self.model.embed_tokens.weight.val();
            last.matmul(w.swap_dims(0, 1))
        }
    }

    fn seed_static(&self, kv: &mut StaticKv, slot: usize, cache: &Self::Cache) {
        for (l, layer) in cache.iter().enumerate() {
            let (k, v) = layer.as_ref().expect("seed_static: a prefilled cache");
            kv.seed(l, slot, k.clone(), v.clone());
        }
    }
}

/// A small random Qwen3 with no EOS (a decode always runs its full length),
/// for tests elsewhere in the crate.
#[cfg(test)]
pub(crate) fn toy_for_tests(device: &Device) -> LoadedQwen3 {
    let config = Qwen3Config {
        vocab_size: 64,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 6,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        rope_scaling: None,
        max_position_embeddings: Some(512),
        sliding_window: None,
        use_sliding_window: false,
        tie_word_embeddings: true,
        eos_token_id: EosIds::None,
    };
    LoadedQwen3 {
        model: build(&config, device),
        config,
        tokenizer_config: None,
    }
}

/// A Qwen3 checkpoint loaded as an encoder (see [`load_trunk_from_dir`]).
///
/// The trunk, with `model.lm_head` always `None`. Deliberately not a
/// [`CausalLm`] — without its head an untied checkpoint has no logits to
/// give, and borrowing the embedding matrix instead would answer wrong.
pub struct Qwen3Trunk {
    pub model: Qwen3,
    pub config: Qwen3Config,
    /// As on [`LoadedQwen3`]: the parsed sibling `tokenizer_config.json`.
    pub tokenizer_config: Option<crate::tok_config::TokenizerConfig>,
}

impl Qwen3Trunk {
    /// `ids` pooled to one `[1, hidden]` vector from the final hidden states
    /// (not normalized — that is the embedder's call).
    ///
    /// # Panics
    ///
    /// When `ids` is empty or a token id does not fit `i32`.
    #[must_use]
    pub fn pooled(&self, ids: &[u32], pooling: Pooling, device: &Device) -> Tensor<2> {
        pooled_hidden(&self.model, &self.config, ids, pooling, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, GgufTensorInfo};

    /// A synthetic toy config with a **decoupled** `head_dim` (`num_heads·head_dim`
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
            rope_scaling: None,
            max_position_embeddings: Some(512),
            sliding_window: None,
            use_sliding_window: false,
            tie_word_embeddings: true,
            eos_token_id: EosIds::One(2),
        }
    }

    /// [`toy_config`] with no EOS id, for tests that need a decode to run its
    /// full `max_tokens`. The toy's weights are an unseeded random draw, and
    /// about one draw in 600 makes EOS (2) one of the first four greedy
    /// tokens, which stops the decode short of the bound.
    fn toy_config_without_eos() -> Qwen3Config {
        Qwen3Config {
            eos_token_id: EosIds::None,
            ..toy_config()
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

    /// Qwen3-0.6B's own config: `rope_scaling: null`, `sliding_window: null`,
    /// `use_sliding_window: false`. The load Mummu parity-verifies must keep
    /// working with the new fields present.
    #[test]
    fn the_real_qwen3_06b_shape_with_null_scaling_still_loads() {
        let json = br#"{
            "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "rope_scaling": null, "sliding_window": null, "use_sliding_window": false,
            "max_window_layers": 28, "max_position_embeddings": 40960,
            "tie_word_embeddings": true, "eos_token_id": 151645
        }"#;
        let cfg = Qwen3Config::from_json_bytes(json).expect("the zoo's own shape must load");
        assert!(cfg.rope_scaling.is_none() && cfg.sliding_window.is_none());
        assert_eq!(cfg.max_position_embeddings, Some(40960));
    }

    /// The pre-4.38 spelling (`"type"` rather than `"rope_type"`) is read too,
    /// so an older linear-scaled checkpoint cannot slip past.
    #[test]
    fn a_linear_scaled_checkpoint_is_refused_in_the_legacy_spelling() {
        let json = br#"{
            "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "rope_scaling": {"type": "linear", "factor": 2.0}
        }"#;
        let err = Qwen3Config::from_json_bytes(json).expect_err("linear must refuse");
        assert!(err.contains("linear"), "{err}");
        assert!(err.contains("qwen3 config.json"), "{err}");
    }

    /// Newer transformers serializes the same object as `rope_parameters`.
    /// Reading only `rope_scaling` would let a scaled checkpoint through as
    /// unscaled — the exact silent failure this whole gate exists to stop.
    #[test]
    fn the_newer_rope_parameters_spelling_is_read_as_well() {
        let json = br#"{
            "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0}
        }"#;
        let err = Qwen3Config::from_json_bytes(json).expect_err("yarn must refuse");
        assert!(err.contains("yarn"), "{err}");
        // And the plain spelling of the same field still loads.
        let plain = br#"{
            "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}
        }"#;
        assert!(Qwen3Config::from_json_bytes(plain).is_ok());
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
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let loaded = LoadedQwen3 {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };

        let prompt: Vec<u32> = vec![3, 14, 15, 9, 26];
        let mut cache = loaded.new_cache();
        let _ = loaded.forward(&prompt, 0, &mut cache, &device);
        let step = loaded
            .forward(&[42], prompt.len(), &mut cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let mut full_cache = loaded.new_cache();
        let all: Vec<u32> = prompt.iter().copied().chain([42]).collect();
        let full = loaded
            .forward(&all, 0, &mut full_cache, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        assert_eq!(step.len(), full.len());
        for (i, (c, f)) in step.iter().zip(&full).enumerate() {
            assert!((c - f).abs() < 1e-4, "logit {i}: cached {c} vs full {f}");
        }
    }

    #[test]
    fn untied_toy_config_builds_and_uses_an_lm_head() {
        let device = crate::backend::cpu_device();
        let mut cfg = toy_config();
        cfg.tie_word_embeddings = false;
        let vocab = cfg.vocab_size;
        let loaded = LoadedQwen3 {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
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
            // Single-file fixture: no split set behind it.
            shards: Vec::new(),
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

    fn to_vec(t: Tensor<2>) -> Vec<f32> {
        t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
    }

    /// The head restricted to some rows is those columns of the full logits,
    /// for a tied head and an untied one alike.
    #[test]
    fn head_rows_are_the_matching_columns_of_the_full_logits() {
        let device = crate::backend::cpu_device();
        for tied in [true, false] {
            let mut cfg = toy_config();
            cfg.tie_word_embeddings = tied;
            let loaded = LoadedQwen3 {
                model: build(&cfg, &device),
                config: cfg,
                tokenizer_config: None,
            };
            let ids = [5u32, 9, 3, 33];
            let mut cache = loaded.new_cache();
            let full = to_vec(loaded.forward(&ids, 0, &mut cache, &device));
            let rows = [7u32, 2, 63];
            let some = to_vec(loaded.head_rows(loaded.last_hidden(&ids, &device), &rows, &device));
            for (k, &r) in rows.iter().enumerate() {
                let want = full[usize::try_from(r).unwrap()];
                assert!(
                    (some[k] - want).abs() < 1e-4,
                    "tied={tied} row {r}: {} vs {want}",
                    some[k]
                );
            }
        }
    }

    /// Pooling prefills long inputs in chunks through the cache; the pooled
    /// vector must be the one-shot one, for every pooling mode.
    #[test]
    fn chunked_pooling_matches_one_shot() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let model = build(&cfg, &device);
        let ids: Vec<u32> = (0..u32::try_from(POOL_CHUNK + 37).unwrap())
            .map(|i| (i * 7 + 3) % 64)
            .collect();
        let mut cache: Vec<LayerKv> = (0..cfg.num_hidden_layers).map(|_| None).collect();
        let all = trunk(&model, &cfg, &ids, 0, &mut cache, &device); // [1, t, h]
        let t = ids.len();
        let h = cfg.hidden_size;
        let want_last = to_vec(all.clone().narrow(1, t - 1, 1).reshape([1, h]));
        let want_first = to_vec(all.clone().narrow(1, 0, 1).reshape([1, h]));
        let want_mean = to_vec(
            all.sum_dim(1)
                .reshape([1, h])
                .div_scalar(mummu_num::f32_from_usize(t)),
        );
        for (pooling, want) in [
            (Pooling::LastToken, want_last),
            (Pooling::Cls, want_first),
            (Pooling::Mean, want_mean),
        ] {
            let got = to_vec(pooled_hidden(&model, &cfg, &ids, pooling, &device));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() < 1e-3,
                    "{pooling:?} elem {i}: chunked {g} vs one-shot {w}"
                );
            }
        }
    }

    /// The encoder build never carries a head, whatever the tie flag says —
    /// Qwen3-Embedding-8B declares an untied head it does not ship.
    #[test]
    fn the_trunk_build_has_no_head_even_when_untied() {
        let device = crate::backend::cpu_device();
        let mut cfg = toy_config();
        cfg.tie_word_embeddings = false;
        let trunk_only = Qwen3Trunk {
            model: build_with_head(&cfg, &device, false),
            config: cfg,
            tokenizer_config: None,
        };
        assert!(trunk_only.model.lm_head.is_none());
        let v = trunk_only.pooled(&[1, 2, 3], Pooling::LastToken, &device);
        assert_eq!(v.dims(), [1, 16]);
    }

    #[tokio::test]
    async fn greedy_generate_respects_max_tokens_bound() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config_without_eos();
        let loaded = LoadedQwen3 {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };
        let out = loaded
            .greedy_generate(&[1, 2, 3], 4, &device)
            .await
            .unwrap();
        // With no EOS to end the decode early, only the bound can stop it.
        assert_eq!(out.len(), 4);
    }
}
