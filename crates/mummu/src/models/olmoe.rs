//! OLMoE sparse mixture-of-experts decoder (allenai OLMoE-1B-7B), from
//! scratch on the shared `nn` blocks — the zoo's first MoE architecture.
//! Structure per layer is pre-norm like Qwen, with two deltas:
//!   * the FFN is a [`SparseMoe`] — a softmax top-k router (`k = 8` of 64)
//!     over narrow SwiGLU experts, `norm_topk_prob = false`;
//!   * q/k RMSNorm applies to the **whole projection** (width
//!     `num_heads * head_dim`) before the head split — `GqaAttention`'s
//!     projection placement (OLMoE is MHA: 16 query heads, 16 KV heads).
//!
//! This is the **resident-everything** first cut: all 64 experts' weights
//! live in memory and every expert computes every token (the router mask
//! zeroes the unrouted ones) — ~7B params in f32 is ~28 GB, which targets the
//! CPU backend on the reference 128 GB machine. Expert streaming / offload is
//! the P6 placement item; keep-quantized is P9.
//!
//! Import covers **both** sources. A GGUF ships the experts already fused
//! (`ffn_*_exps`) in exactly the layout `MoeExperts` holds; the HF safetensors
//! checkpoint stores each expert separately
//! (`mlp.experts.{i}.{gate,up,down}_proj.weight`) and is sharded, so
//! [`load_from_dir`] runs it through
//! [`crate::safetensors::fuse_checkpoint_to_file`], which reads every shard and
//! stacks each 64-member expert group into one `[experts, out, in]` tensor
//! before the ordinary checked-load pipeline. It fuses to a temp file rather
//! than to RAM deliberately: the in-memory twin would need the whole payload
//! resident (13.8 GB for the 1B-7B) on top of the ~28 GB f32 model the load
//! then builds. [`load_from_gguf`] makes the same trade for the same reason:
//! its dequant streams to a temp file (~28 GB of f32) that `burn-store` then
//! mmaps back, so the payload is file-backed rather than charged to commit.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo, GgufValue};
use crate::import::{CastFloatAdapter, ImportError, load_checked, required_file};
use crate::models::CausalLm;
use crate::models::qwen2::{EosIds, gguf_f32, gguf_usize};
use crate::nn::{
    GqaAttention, GqaAttentionConfig, LayerKv, SparseMoe, SparseMoeConfig, causal_mask, rope_tables,
};
use crate::safetensors::{Fuse, fuse_checkpoint_to_file};

/// OLMoE architecture hyperparameters (HF `config.json` field names).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OlmoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Per-expert SwiGLU intermediate width (1B-7B: 1024).
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Renormalize the top-k routing weights to sum 1 (OLMoE ships `false`:
    /// the raw softmax probabilities weight the mixture).
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: EosIds,
}

impl OlmoeConfig {
    /// Parse `config.json` bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let cfg: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Hyperparameters from a GGUF header's `olmoe.*` metadata.
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "olmoe" {
            return Err(format!("GGUF architecture '{arch}' is not olmoe"));
        }
        let hidden_size = gguf_usize(f, "olmoe.embedding_length")?;
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
        // llama.cpp writes the per-expert width as expert_feed_forward_length
        // when it differs from feed_forward_length; accept either spelling.
        let intermediate_size = gguf_usize(f, "olmoe.expert_feed_forward_length")
            .or_else(|_| gguf_usize(f, "olmoe.feed_forward_length"))?;
        // OLMoE does not renormalize top-k weights; honor the metadata key
        // when a file carries one, default to the architecture's `false`.
        let norm_topk_prob = f
            .get("olmoe.expert_weights_norm")
            .and_then(GgufValue::as_bool)
            .unwrap_or(false);
        let cfg = Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers: gguf_usize(f, "olmoe.block_count")?,
            num_attention_heads: gguf_usize(f, "olmoe.attention.head_count")?,
            num_key_value_heads: gguf_usize(f, "olmoe.attention.head_count_kv")?,
            num_experts: gguf_usize(f, "olmoe.expert_count")?,
            num_experts_per_tok: gguf_usize(f, "olmoe.expert_used_count")?,
            norm_topk_prob,
            rms_norm_eps: f64::from(gguf_f32(f, "olmoe.attention.layer_norm_rms_epsilon")?),
            rope_theta: gguf_f32(f, "olmoe.rope.freq_base")?,
            // No separate output.weight tensor means the lm-head is tied.
            tie_word_embeddings: f.tensor("output.weight").is_none(),
            eos_token_id,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// `hidden_size / num_attention_heads` — OLMoE's head_dim is not decoupled.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
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
        if self.num_experts < 2 || !(1..=self.num_experts).contains(&self.num_experts_per_tok) {
            return Err(format!(
                "num_experts ({}) must be >= 2 with num_experts_per_tok ({}) in 1..=num_experts",
                self.num_experts, self.num_experts_per_tok
            ));
        }
        let hd = self.head_dim();
        if hd < 2 || !hd.is_multiple_of(2) || hd * self.num_attention_heads != self.hidden_size {
            return Err(format!(
                "hidden_size ({}) must split evenly into num_attention_heads ({}) even-sized heads",
                self.hidden_size, self.num_attention_heads
            ));
        }
        Ok(())
    }
}

/// One OLMoE decoder layer. Field names mirror the HF checkpoint layout.
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    pub self_attn: GqaAttention<B>,
    pub mlp: SparseMoe<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

/// The OLMoE decoder stack (HF's `model.*` subtree). The 1B-7B ships untied.
#[derive(Module, Debug)]
pub struct Olmoe<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Option<Linear<B>>,
}

/// A weight-loaded OLMoE plus its config — everything a forward needs.
pub struct LoadedOlmoe<B: Backend> {
    pub model: Olmoe<B>,
    pub config: OlmoeConfig,
    /// The sibling `tokenizer_config.json`, when the checkpoint dir ships one
    /// — config-driven EOS/BOS/PAD for a consumer to read. `None` for a GGUF
    /// load (self-contained: EOS rides the GGUF metadata).
    pub tokenizer_config: Option<crate::tok_config::TokenizerConfig>,
}

fn build<B: Backend>(cfg: &OlmoeConfig, device: &B::Device) -> Olmoe<B> {
    let attn_cfg = GqaAttentionConfig {
        hidden_size: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        bias: false,                         // OLMoE projections are bias-free
        qk_norm_eps: Some(cfg.rms_norm_eps), // q/k RMSNorm over the whole
        qk_norm_projection: true,            // projection, pre head-split
    };
    let moe_cfg = SparseMoeConfig {
        hidden_size: cfg.hidden_size,
        expert_intermediate_size: cfg.intermediate_size,
        num_experts: cfg.num_experts,
        num_experts_per_tok: cfg.num_experts_per_tok,
    };
    let norm = |dev: &B::Device| {
        RmsNormConfig::new(cfg.hidden_size)
            .with_epsilon(cfg.rms_norm_eps)
            .init(dev)
    };
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| DecoderLayer {
            self_attn: attn_cfg.init(device),
            mlp: moe_cfg.init(device),
            input_layernorm: norm(device),
            post_attention_layernorm: norm(device),
        })
        .collect();
    let lm_head = (!cfg.tie_word_embeddings).then(|| {
        LinearConfig::new(cfg.hidden_size, cfg.vocab_size)
            .with_bias(false)
            .init(device)
    });
    Olmoe {
        embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
        layers,
        norm: norm(device),
        lm_head,
    }
}

/// The key remap: strip `model.`, rename every RmsNorm `weight` → Burn's
/// `gamma` (incl. the projection-wide `self_attn.{q,k}_norm`). The fused
/// expert params (`mlp.experts.{gate,up,down}`) carry no `.weight` suffix —
/// they are raw `Param<Tensor>` fields, named by the GGUF map directly.
fn install_remaps(store: SafetensorsStore) -> SafetensorsStore {
    store
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(input_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(post_attention_layernorm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(self_attn\.q_norm)\.weight$", "$1.gamma")
        .with_key_remapping(r"(self_attn\.k_norm)\.weight$", "$1.gamma")
        .with_key_remapping(r"^norm\.weight$", "norm.gamma")
}

/// GGUF (llama.cpp `olmoe` arch) tensor names → the HF-shaped names the remap
/// chain handles. `None` for anything unrecognized (a loud load error).
fn gguf_tensor_to_hf(info: &GgufTensorInfo) -> Option<GgufMap> {
    olmoe_gguf_name(&info.name).map(GgufMap::Rename)
}

fn olmoe_gguf_name(name: &str) -> Option<String> {
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
        // The router Linear — a plain 2-D weight, transposed by the adapter.
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        // The fused 3-D expert banks — raw params, no `.weight` suffix in the
        // module path. ggml dims reverse to [experts, out, in], exactly the
        // `MoeExperts` layout.
        "ffn_gate_exps.weight" => "mlp.experts.gate",
        "ffn_up_exps.weight" => "mlp.experts.up",
        "ffn_down_exps.weight" => "mlp.experts.down",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{mapped}"))
}

/// Load an OLMoE model straight from a **GGUF** file: hyperparameters from
/// the `olmoe.*` metadata, weights dequantized to f32 and driven through the
/// same checked-load pipeline every other port uses. Budget note: the 1B-7B's
/// ~7B params dequantize to ~28 GB of f32 — size the target device (the
/// reference machine runs it on the 128 GB CPU backend).
///
/// The dequant goes to a TEMP FILE, not to RAM, for the same reason
/// `load_from_dir`'s fuse does: holding ~28 GB of f32 payload *and* building
/// the ~28 GB model from it is the sum a 128 GB box with other tenants
/// actually fails to satisfy. Streaming keeps the peak at the model plus one
/// tensor. The file is this process's to delete, on success or failure.
pub fn load_from_gguf<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<LoadedOlmoe<B>, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = OlmoeConfig::from_gguf(&f).map_err(parse)?;
    // Beside the GGUF, so the scratch write lands on the volume the weights
    // already live on.
    let scratch = FusedTemp::new(path.parent().unwrap_or(Path::new(".")))?;
    let bytes = f
        .dequant_to_safetensors_file(&gguf_tensor_to_hf, scratch.path())
        .map_err(|e| parse(e.to_string()))?;
    assert!(bytes > 0, "a parsed GGUF yields a non-empty payload");

    let mut model = build::<B>(&config, device);
    // The backend's float dtype, taken from the TYPE (`B::FloatElem`), never
    // from a probe tensor (per-device default policy hazard).
    let target_float = <B::FloatElem as burn::tensor::Element>::dtype();
    let mut store = install_remaps(
        SafetensorsStore::from_file(scratch.path().to_path_buf())
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
            .allow_partial(true),
    );
    load_checked(&mut model, &mut store, path)?;
    Ok(LoadedOlmoe {
        model,
        config,
        tokenizer_config: None,
    })
}

/// How a source tensor of an HF OLMoE checkpoint reaches the module.
///
/// Everything but the expert bank passes through under its own name (the
/// `install_remaps` chain does the HF→module renaming downstream, exactly as
/// on the single-file safetensors path). The per-expert projections are the
/// N:1 case: `model.layers.3.mlp.experts.7.gate_proj.weight` is member 7 of
/// the fused `model.layers.3.mlp.experts.gate`.
///
/// Deliberately *not* a strict allow-list: unrecognized tensors are kept, and
/// a checkpoint that renames the expert projections then fails loudly one
/// stage later in `load_checked` — which names the missing `experts.gate`
/// param in its report — rather than here with a less specific message.
fn olmoe_hf_fuse(name: &str, num_experts: usize) -> Fuse {
    let Some(target) = fused_expert_target(name, num_experts) else {
        return Fuse::Keep(name.to_string());
    };
    target
}

/// `model.layers.{L}.mlp.experts.{I}.{gate,up,down}_proj.weight` → its slot in
/// the fused bank, or `None` when the name is not a per-expert projection.
fn fused_expert_target(name: &str, num_experts: usize) -> Option<Fuse> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    layer.parse::<usize>().ok()?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (index, rest) = rest.split_once('.')?;
    let index: usize = index.parse().ok()?;
    let projection = match rest {
        "gate_proj.weight" => "gate",
        "up_proj.weight" => "up",
        "down_proj.weight" => "down",
        _ => return None,
    };
    Some(Fuse::Stack {
        target: format!("model.layers.{layer}.mlp.experts.{projection}"),
        index,
        count: num_experts,
    })
}

/// Load an OLMoE model from an **HF safetensors checkpoint dir**.
///
/// The checkpoint is sharded (`model-0000N-of-0000M.safetensors` +
/// `model.safetensors.index.json`) and stores every expert separately, so the
/// shards are read and the expert groups fused into `[experts, out, in]`
/// tensors first; the fused blob then rides the SAME adapter chain and
/// `load_checked` as every other import path. Source dtype is preserved by
/// the fuse (HF ships bf16) and cast to the backend float on load.
///
/// Budget note: the fused blob is the checkpoint's own size (~13.8 GB in bf16
/// for the 1B-7B) and the loaded f32 model is ~28 GB — size the target device.
/// Owns the scratch safetensors file for the life of one load — the fused
/// checkpoint on the HF path, the dequantized payload on the GGUF path.
///
/// It is as large as the weights it carries (13.8 GB fusing the 1B-7B, ~28 GB
/// dequantizing its Q4_K_M), so leaving one behind on a failed load would
/// quietly fill the disk over a few retries. `Drop` removes it on every exit
/// path, success or `?`.
struct FusedTemp {
    path: PathBuf,
}

impl FusedTemp {
    /// Placed beside the checkpoint, so the scratch write lands on the same
    /// volume as the weights rather than on a small system temp drive.
    ///
    /// The name carries a process-unique counter as well as the pid: two
    /// concurrent loads in one process must not choose the same scratch file
    /// and interleave their writes into it.
    fn new(dir: &Path) -> Result<Self, ImportError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        assert!(!dir.as_os_str().is_empty(), "checkpoint dir must be named");
        let path = dir.join(format!(
            "mummu-fused-{}-{}.safetensors.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // A leftover from a killed process must never be mistaken for ours.
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| ImportError::Parse {
                file: path.clone(),
                reason: format!("could not clear a stale fused scratch file: {e}"),
            })?;
        }
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        debug_assert!(!self.path.as_os_str().is_empty(), "scratch path is named");
        &self.path
    }
}

impl Drop for FusedTemp {
    fn drop(&mut self) {
        // Best effort by construction: a Drop that can fail has nowhere to
        // report to, and a stranded scratch file is not worth a panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn load_from_dir<B: Backend>(
    dir: &Path,
    device: &B::Device,
) -> Result<LoadedOlmoe<B>, ImportError> {
    let cfg_path = required_file(dir, "config.json")?;
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| ImportError::Parse {
        file: cfg_path.clone(),
        reason: e.to_string(),
    })?;
    let config = OlmoeConfig::from_json_bytes(&cfg_bytes).map_err(|reason| ImportError::Parse {
        file: cfg_path,
        reason,
    })?;

    // Cross-check the sibling metadata before touching weights, same as the
    // dense loaders. `None` for the expected tool-call convention: Mummu ships
    // no hardcoded OLMoE `chat` renderer to contradict, so only the EOS and
    // added-token-id checks apply.
    let tokenizer_config =
        crate::tokenizer::validate_checkpoint_dir(dir, &config.eos_token_id.to_vec(), None)?;

    let num_experts = config.num_experts;
    // Fuse to a TEMP FILE, not to RAM. The in-memory fuse needs the whole
    // payload resident (13.8 GB for the 1B-7B) on top of the ~28 GB f32 model
    // the load then builds; streaming it to disk keeps the peak at the model
    // alone. The file is this process's to delete, and it is deleted whether
    // the load succeeds or fails.
    let fused = FusedTemp::new(dir)?;
    let bytes = fuse_checkpoint_to_file(
        dir,
        &|name| Some(olmoe_hf_fuse(name, num_experts)),
        fused.path(),
    )
    .map_err(|e| ImportError::Parse {
        file: dir.to_path_buf(),
        reason: e.to_string(),
    })?;
    assert!(bytes > 8, "a fused checkpoint yields a non-empty payload");

    let mut model = build::<B>(&config, device);
    // The backend's float dtype from the TYPE (`B::FloatElem`), never a probe
    // tensor (the per-device default-dtype policy hazard).
    let target_float = <B::FloatElem as burn::tensor::Element>::dtype();
    let mut store = install_remaps(
        SafetensorsStore::from_file(fused.path().to_path_buf())
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
            .allow_partial(true),
    );
    load_checked(&mut model, &mut store, dir)?;
    Ok(LoadedOlmoe {
        model,
        config,
        tokenizer_config,
    })
}

impl<B: Backend> CausalLm<B> for LoadedOlmoe<B> {
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
        assert!(t >= 1, "OLMoE forward: need at least one token");
        assert!(
            cache.len() == self.config.num_hidden_layers,
            "OLMoE forward: cache has {} layers, model has {}",
            cache.len(),
            self.config.num_hidden_layers
        );
        let cfg = &self.config;
        let hd = cfg.head_dim();

        // Dtype pinned to the backend TYPE, never the per-device policy.
        let ids32: Vec<i32> = new_ids.iter().map(|&i| i as i32).collect();
        let input = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids32, [t]),
            (device, crate::backend::int_dtype::<B>()),
        )
        .reshape([1, t]);
        let mut x = self.model.embed_tokens.forward(input); // [1, t, hidden]

        let (cos, sin) = rope_tables::<B>(t, past, hd, cfg.rope_theta, device);
        let mask = (t > 1).then(|| causal_mask::<B>(t, past, device));

        for (layer, kv) in self.model.layers.iter().zip(cache.iter_mut()) {
            let h = layer.input_layernorm.forward(x.clone());
            let h = layer.self_attn.forward(
                h,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                hd,
                &cos,
                &sin,
                mask.as_ref(),
                kv,
            );
            x = x.add(h);
            let h2 = layer.post_attention_layernorm.forward(x.clone());
            x = x.add(
                layer
                    .mlp
                    .forward(h2, cfg.num_experts_per_tok, cfg.norm_topk_prob),
            );
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

    /// A synthetic toy MoE config: 4 experts, top-2, MHA, untied head.
    fn toy_config() -> OlmoeConfig {
        OlmoeConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: false,
            rms_norm_eps: 1e-5,
            rope_theta: 1e4,
            tie_word_embeddings: false,
            eos_token_id: EosIds::One(2),
        }
    }

    #[test]
    fn config_parses_the_real_1b_7b_shape() {
        // The real OLMoE-1B-7B-0125-Instruct config.json shape.
        let json = br#"{
            "vocab_size": 50304, "hidden_size": 2048, "intermediate_size": 1024,
            "num_hidden_layers": 16, "num_attention_heads": 16, "num_key_value_heads": 16,
            "num_experts": 64, "num_experts_per_tok": 8, "norm_topk_prob": false,
            "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "eos_token_id": 50279
        }"#;
        let cfg = OlmoeConfig::from_json_bytes(json).unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert!(!cfg.norm_topk_prob);
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.eos_token_id.contains(50_279));
    }

    #[test]
    fn config_rejects_bad_expert_counts() {
        let mut cfg = toy_config();
        cfg.num_experts = 1;
        assert!(cfg.validate().is_err(), "one expert is not a mixture");
        let mut cfg = toy_config();
        cfg.num_experts_per_tok = 5;
        assert!(cfg.validate().is_err(), "top-k above expert count");
        let mut cfg = toy_config();
        cfg.num_experts_per_tok = 0;
        assert!(cfg.validate().is_err(), "zero top-k");
    }

    /// The load-bearing invariant: cached prefill+decode == one full forward,
    /// through the MoE layers and the projection-wide q/k norm.
    #[test]
    fn toy_model_cached_decode_matches_full_forward() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedOlmoe::<Cpu> {
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
    fn projection_qk_norm_spans_the_whole_projection() {
        let device = Dev::default();
        let cfg = toy_config();
        let model = build::<Cpu>(&cfg, &device);
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        assert_eq!(
            model.layers[0]
                .self_attn
                .q_norm
                .as_ref()
                .unwrap()
                .gamma
                .dims(),
            [q_dim],
            "OLMoE q_norm must span num_heads * head_dim, not head_dim"
        );
    }

    /// A synthetic in-memory GGUF header shaped like a small `olmoe` file.
    fn toy_gguf() -> GgufFile {
        let meta = |k: &str, v: GgufValue| (k.to_string(), v);
        GgufFile {
            path: std::path::PathBuf::new(),
            version: 3,
            metadata: vec![
                meta("general.architecture", GgufValue::Str("olmoe".into())),
                meta("olmoe.embedding_length", GgufValue::U32(16)),
                meta("olmoe.block_count", GgufValue::U32(2)),
                meta("olmoe.feed_forward_length", GgufValue::U32(8)),
                meta("olmoe.attention.head_count", GgufValue::U32(4)),
                meta("olmoe.attention.head_count_kv", GgufValue::U32(4)),
                meta("olmoe.expert_count", GgufValue::U32(4)),
                meta("olmoe.expert_used_count", GgufValue::U32(2)),
                meta(
                    "olmoe.attention.layer_norm_rms_epsilon",
                    GgufValue::F32(1e-5),
                ),
                meta("olmoe.rope.freq_base", GgufValue::F32(1e4)),
                meta("tokenizer.ggml.eos_token_id", GgufValue::U32(2)),
            ],
            tensors: vec![
                GgufTensorInfo {
                    name: "token_embd.weight".into(),
                    dims: vec![16, 64], // ggml order: [hidden, vocab]
                    dtype: GgmlType::F32,
                    offset: 0,
                },
                GgufTensorInfo {
                    name: "output.weight".into(),
                    dims: vec![16, 64],
                    dtype: GgmlType::F32,
                    offset: 4096,
                },
            ],
            alignment: 32,
            data_offset: 0,
        }
    }

    #[test]
    fn config_from_gguf_reads_expert_metadata() {
        let cfg = OlmoeConfig::from_gguf(&toy_gguf()).expect("parses");
        assert_eq!(cfg.num_experts, 4);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert_eq!(cfg.intermediate_size, 8);
        assert!(!cfg.norm_topk_prob);
        assert!(!cfg.tie_word_embeddings); // output.weight present → untied
        assert!(cfg.eos_token_id.contains(2));
    }

    #[test]
    fn config_from_gguf_prefers_expert_feed_forward_length() {
        let mut f = toy_gguf();
        f.metadata.push((
            "olmoe.expert_feed_forward_length".into(),
            GgufValue::U32(12),
        ));
        assert_eq!(OlmoeConfig::from_gguf(&f).unwrap().intermediate_size, 12);
    }

    #[test]
    fn config_from_gguf_fails_loudly_on_missing_keys_and_wrong_arch() {
        let mut f = toy_gguf();
        f.metadata.retain(|(k, _)| k != "olmoe.expert_count");
        assert!(
            OlmoeConfig::from_gguf(&f)
                .unwrap_err()
                .contains("expert_count")
        );

        let mut f = toy_gguf();
        f.metadata[0].1 = GgufValue::Str("qwen3".into());
        assert!(OlmoeConfig::from_gguf(&f).is_err());
    }

    #[test]
    fn gguf_names_map_router_and_fused_expert_banks() {
        assert_eq!(
            olmoe_gguf_name("blk.0.ffn_gate_inp.weight").as_deref(),
            Some("model.layers.0.mlp.gate.weight")
        );
        // Fused expert banks are raw params — no `.weight` suffix.
        assert_eq!(
            olmoe_gguf_name("blk.3.ffn_gate_exps.weight").as_deref(),
            Some("model.layers.3.mlp.experts.gate")
        );
        assert_eq!(
            olmoe_gguf_name("blk.15.ffn_down_exps.weight").as_deref(),
            Some("model.layers.15.mlp.experts.down")
        );
        assert_eq!(
            olmoe_gguf_name("blk.1.attn_q_norm.weight").as_deref(),
            Some("model.layers.1.self_attn.q_norm.weight")
        );
        assert_eq!(
            olmoe_gguf_name("output.weight").as_deref(),
            Some("lm_head.weight")
        );
        // A dense FFN tensor is not part of this architecture — loud None.
        assert_eq!(olmoe_gguf_name("blk.0.ffn_gate.weight"), None);
        assert_eq!(olmoe_gguf_name("rope_freqs.weight"), None);
    }

    #[test]
    fn greedy_generate_respects_max_tokens_bound() {
        let device = Dev::default();
        let cfg = toy_config();
        let loaded = LoadedOlmoe::<Cpu> {
            model: build(&cfg, &device),
            config: cfg,
            tokenizer_config: None,
        };
        let out = loaded.greedy_generate(&[1, 2, 3], 4, &device).unwrap();
        assert!(out.len() <= 4);
    }

    /// The HF per-expert projections fuse onto EXACTLY the module names the
    /// GGUF path renames its pre-fused banks to. Both import paths must land
    /// on the same params, or one of them is loading a different model.
    #[test]
    fn hf_expert_fusion_targets_match_the_gguf_names() {
        for (projection, ggml) in [
            ("gate", "ffn_gate_exps"),
            ("up", "ffn_up_exps"),
            ("down", "ffn_down_exps"),
        ] {
            let hf = format!("model.layers.3.mlp.experts.7.{projection}_proj.weight");
            let expected = olmoe_gguf_name(&format!("blk.3.{ggml}.weight")).unwrap();
            assert_eq!(
                olmoe_hf_fuse(&hf, 64),
                Fuse::Stack {
                    target: expected,
                    index: 7,
                    count: 64,
                },
                "{projection}: safetensors and GGUF must fuse to the same param"
            );
        }
    }

    /// Non-expert tensors pass through untouched — the `install_remaps` chain
    /// does the HF→module renaming downstream, same as the dense loaders.
    #[test]
    fn non_expert_tensors_pass_through_by_name() {
        for name in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.input_layernorm.weight",
            // The ROUTER is a plain Linear, not part of the expert bank —
            // fusing it would be a silent disaster.
            "model.layers.0.mlp.gate.weight",
        ] {
            assert_eq!(
                olmoe_hf_fuse(name, 64),
                Fuse::Keep(name.to_string()),
                "{name} must pass through"
            );
        }
    }

    /// The expert index is parsed as a NUMBER, so the fuse plan can order the
    /// bank numerically; `experts.10` must not be read as expert 1.
    #[test]
    fn expert_index_is_parsed_numerically() {
        let at = |i: usize| match olmoe_hf_fuse(
            &format!("model.layers.0.mlp.experts.{i}.gate_proj.weight"),
            64,
        ) {
            Fuse::Stack { index, .. } => index,
            other => panic!("expected a Stack, got {other:?}"),
        };
        assert_eq!(at(0), 0);
        assert_eq!(at(1), 1);
        assert_eq!(at(10), 10);
        assert_eq!(at(63), 63);
        // A non-numeric member is not an expert projection at all.
        assert_eq!(
            olmoe_hf_fuse("model.layers.0.mlp.experts.x.gate_proj.weight", 64),
            Fuse::Keep("model.layers.0.mlp.experts.x.gate_proj.weight".to_string())
        );
    }
}
