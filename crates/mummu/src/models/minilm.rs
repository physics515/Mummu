//! all-MiniLM-class sentence embedder: a 6-layer post-LayerNorm BERT with
//! absolute position embeddings, full bidirectional attention over an additive
//! padding mask, exact (erf) GeLU, masked-mean pooling + L2 normalize (the
//! sentence-transformers recipe — the checkpoint's `pooler.*` weights are
//! intentionally unused).
//!
//! Ported from laurelane's implementation (cosine ~1.0 vs the Candle BERT on
//! identical weights). BERT's bidirectional attention and LayerNorm differ
//! from the causal GQA blocks in `nn`, so this file is self-contained.
//!
//! The library takes token ids + attention mask and returns embeddings; the
//! caller owns tokenization (HF `tokenizers` in the integration tests).

use std::path::Path;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore};
use burn::tensor::{Device, Int, Tensor, TensorData, activation};

use crate::import::{
    CastFloatAdapter, ImportError, WeightsFile, load_checked, required_file, weights_file,
};

/// BERT hyperparameters, read from the checkpoint's `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
}

fn default_eps() -> f64 {
    1e-12
}

impl BertConfig {
    /// Parse `config.json` bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let cfg: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if cfg.num_attention_heads == 0 || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
        {
            return Err(format!(
                "hidden_size ({}) must be a positive multiple of num_attention_heads ({})",
                cfg.hidden_size, cfg.num_attention_heads
            ));
        }
        Ok(cfg)
    }
}

/// Word + position + token-type embeddings, post-LN.
#[derive(Module, Debug)]
pub struct Embeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
}

/// One post-LN BERT encoder layer (flat field names; HF's nested keys remap
/// onto these at load).
#[derive(Module, Debug)]
pub struct EncoderLayer {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub attn_output: Linear,
    pub attn_layer_norm: LayerNorm,
    pub intermediate: Linear,
    pub output: Linear,
    pub output_layer_norm: LayerNorm,
}

/// The BERT encoder stack.
#[derive(Module, Debug)]
pub struct Bert {
    pub embeddings: Embeddings,
    pub layers: Vec<EncoderLayer>,
}

/// A weight-loaded MiniLM/BERT plus its config.
pub struct LoadedMiniLm {
    pub model: Bert,
    pub config: BertConfig,
}

fn build(cfg: &BertConfig, device: &Device) -> Bert {
    let h = cfg.hidden_size;
    let eps = cfg.layer_norm_eps;
    let lin = |i: usize, o: usize| LinearConfig::new(i, o).init(device);
    let ln = || LayerNormConfig::new(h).with_epsilon(eps).init(device);
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| EncoderLayer {
            query: lin(h, h),
            key: lin(h, h),
            value: lin(h, h),
            attn_output: lin(h, h),
            attn_layer_norm: ln(),
            intermediate: lin(h, cfg.intermediate_size),
            output: lin(cfg.intermediate_size, h),
            output_layer_norm: ln(),
        })
        .collect();
    Bert {
        embeddings: Embeddings {
            word_embeddings: EmbeddingConfig::new(cfg.vocab_size, h).init(device),
            position_embeddings: EmbeddingConfig::new(cfg.max_position_embeddings, h).init(device),
            token_type_embeddings: EmbeddingConfig::new(cfg.type_vocab_size, h).init(device),
            layer_norm: ln(),
        },
        layers,
    }
}

/// HF BERT checkpoint names → our field paths, shared by every weight format.
const KEY_REMAPS: &[(&str, &str)] = &[
    (r"^bert\.", ""), // some checkpoints prefix `bert.`
    (r"^embeddings\.LayerNorm\.", "embeddings.layer_norm."),
    (
        r"^encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.",
        "layers.$1.$2.",
    ),
    (
        r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.",
        "layers.$1.attn_output.",
    ),
    (
        r"^encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.",
        "layers.$1.attn_layer_norm.",
    ),
    (
        r"^encoder\.layer\.(\d+)\.intermediate\.dense\.",
        "layers.$1.intermediate.",
    ),
    (
        r"^encoder\.layer\.(\d+)\.output\.dense\.",
        "layers.$1.output.",
    ),
    (
        r"^encoder\.layer\.(\d+)\.output\.LayerNorm\.",
        "layers.$1.output_layer_norm.",
    ),
];

/// Build from `dir/config.json` and load the checkpoint, checked —
/// `model.safetensors` preferred, `pytorch_model.bin` (the state dict this
/// model family originally shipped) as the fallback.
/// NOTE: LayerNorm keys keep their `.weight`/`.bias` suffixes — the
/// `PyTorchToBurnAdapter` renames those to `gamma`/`beta` itself (unlike
/// RmsNorm in the decoder models, where the rename is manual — the asymmetry
/// is intentional; `PytorchStore` applies that adapter internally).
/// `pooler.*` stays unused (masked-mean pooling instead).
pub fn load_from_dir(
    dir: &Path,
    device: &Device,
) -> Result<LoadedMiniLm, ImportError> {
    let cfg_path = required_file(dir, "config.json")?;
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| ImportError::Parse {
        file: cfg_path.clone(),
        reason: e.to_string(),
    })?;
    let config = BertConfig::from_json_bytes(&cfg_bytes).map_err(|reason| ImportError::Parse {
        file: cfg_path,
        reason,
    })?;

    let mut model = build(&config, device);
    // Type-level float dtype (`f32`) — a probe tensor would follow
    // the per-DEVICE default policy, which another backend alias sharing the
    // device (Gpu vs GpuF16) may have flipped in this process.
    let target_float = crate::backend::float_dtype();
    match weights_file(dir)? {
        WeightsFile::Safetensors(weights) => {
            let mut store = SafetensorsStore::from_file(weights.clone())
                .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float)))
                .allow_partial(true);
            for (pattern, replacement) in KEY_REMAPS {
                store = store.with_key_remapping(*pattern, *replacement);
            }
            load_checked(&mut model, &mut store, &weights)?;
        }
        WeightsFile::PytorchBin(weights) => {
            // No cast adapter on this path (PytorchStore has no adapter
            // chaining) — .bin-era checkpoints are f32, which every backend
            // ingests directly.
            let mut store = PytorchStore::from_file(weights.clone()).allow_partial(true);
            for (pattern, replacement) in KEY_REMAPS {
                store = store.with_key_remapping(*pattern, *replacement);
            }
            load_checked(&mut model, &mut store, &weights)?;
        }
    }
    Ok(LoadedMiniLm { model, config })
}

fn embeddings_forward(
    e: &Embeddings,
    ids: &Tensor<2, Int>,
    device: &Device,
) -> Tensor<3> {
    let [b, n] = ids.dims();
    let w = e.word_embeddings.forward(ids.clone()); // [b, n, h]
    let pos = Tensor::<1, Int>::arange(0..n as i64, device).reshape([1, n]);
    let p = e.position_embeddings.forward(pos); // [1, n, h]
    let tt = Tensor::<2, Int>::zeros([b, n], device);
    let t = e.token_type_embeddings.forward(tt); // [b, n, h]
    e.layer_norm.forward(w.add(p).add(t))
}

fn layer_forward(
    l: &EncoderLayer,
    x: Tensor<3>,
    add_mask: &Tensor<4>,
    num_heads: usize,
) -> Tensor<3> {
    let [b, n, h] = x.dims();
    debug_assert!(h.is_multiple_of(num_heads), "hidden not divisible by heads");
    let hd = h / num_heads;

    let q = l
        .query
        .forward(x.clone())
        .reshape([b, n, num_heads, hd])
        .swap_dims(1, 2);
    let k = l
        .key
        .forward(x.clone())
        .reshape([b, n, num_heads, hd])
        .swap_dims(1, 2);
    let v = l
        .value
        .forward(x.clone())
        .reshape([b, n, num_heads, hd])
        .swap_dims(1, 2);

    let scale = 1.0 / (hd as f32).sqrt();
    let scores = q
        .matmul(k.swap_dims(2, 3))
        .mul_scalar(scale)
        .add(add_mask.clone());
    let probs = activation::softmax(scores, 3);
    let ctx = probs.matmul(v).swap_dims(1, 2).reshape([b, n, h]);

    // Self-attention output: dense → LayerNorm(+ residual).
    let x = l.attn_layer_norm.forward(l.attn_output.forward(ctx).add(x));
    // Feed-forward: dense → exact GeLU → dense → LayerNorm(+ residual).
    let inter = activation::gelu(l.intermediate.forward(x.clone()));
    l.output_layer_norm.forward(l.output.forward(inter).add(x))
}

impl LoadedMiniLm {
    /// Encode one tokenized string (`ids` + `mask`, 1.0 = real token) into a
    /// masked-mean-pooled, **L2-normalized** sentence embedding of
    /// `hidden_size` floats.
    pub fn embed_ids(
        &self,
        ids: &[u32],
        mask: &[f32],
        device: &Device,
    ) -> Result<Vec<f32>, String> {
        let n = ids.len();
        assert!(n >= 1, "embed_ids: empty token sequence");
        assert!(
            mask.len() == n,
            "embed_ids: mask length {} != ids length {n}",
            mask.len()
        );
        debug_assert!(
            n <= self.config.max_position_embeddings,
            "embed_ids: sequence longer than max_position_embeddings"
        );

        let ids32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        // Dtypes pinned to the backend TYPE, never the per-device policy.
        let id_t = Tensor::<1, Int>::from_data(
            TensorData::new(ids32, [n]),
            (device, crate::backend::int_dtype()),
        )
        .reshape([1, n]);
        let mask_t = Tensor::<1>::from_data(
            TensorData::new(mask.to_vec(), [n]),
            (device, crate::backend::float_dtype()),
        )
        .reshape([1, n]);

        // Additive padding mask [1, 1, 1, n]: 0 for real tokens, large-negative
        // for padding, broadcast across heads and query positions.
        let add_mask = mask_t
            .clone()
            .reshape([1, 1, 1, n])
            .neg()
            .add_scalar(1.0)
            .mul_scalar(-1e30);

        let mut x = embeddings_forward(&self.model.embeddings, &id_t, device);
        for layer in &self.model.layers {
            x = layer_forward(layer, x, &add_mask, self.config.num_attention_heads);
        }

        // Masked mean pool → L2 normalize.
        let m3 = mask_t.reshape([1, n, 1]);
        let summed = x.mul(m3.clone()).sum_dim(1); // [1, 1, h]
        let counts = m3.sum_dim(1); // [1, 1, 1]
        let pooled = summed.div(counts).reshape([1, self.config.hidden_size]);
        let norm = pooled.clone().powf_scalar(2.0).sum_dim(1).sqrt(); // [1, 1]
        let normalized = pooled.div(norm);

        normalized
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| format!("embedding readback: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Dev = burn::tensor::Device;

    fn toy_config() -> BertConfig {
        BertConfig {
            vocab_size: 50,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 64,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }

    #[test]
    fn config_rejects_indivisible_heads() {
        let json = br#"{
            "vocab_size": 50, "hidden_size": 15, "num_hidden_layers": 1,
            "num_attention_heads": 4, "intermediate_size": 32,
            "max_position_embeddings": 64, "type_vocab_size": 2
        }"#;
        assert!(BertConfig::from_json_bytes(json).is_err());
    }

    #[test]
    fn embeddings_are_unit_norm() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let loaded = LoadedMiniLm {
            model: build(&cfg, &device),
            config: cfg,
        };
        let ids = [3u32, 7, 12, 4];
        let mask = [1.0f32; 4];
        let e = loaded.embed_ids(&ids, &mask, &device).unwrap();
        assert_eq!(e.len(), 16);
        let norm: f32 = e.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "L2 norm should be 1, got {norm}");
    }

    /// Padding must not change the embedding: the padded positions are masked
    /// out of attention AND the mean pool.
    #[test]
    fn padding_is_invisible_to_the_embedding() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let loaded = LoadedMiniLm {
            model: build(&cfg, &device),
            config: cfg,
        };
        let bare = loaded
            .embed_ids(&[3, 7, 12], &[1.0, 1.0, 1.0], &device)
            .unwrap();
        let padded = loaded
            .embed_ids(&[3, 7, 12, 0, 0], &[1.0, 1.0, 1.0, 0.0, 0.0], &device)
            .unwrap();
        for (i, (a, b)) in bare.iter().zip(&padded).enumerate() {
            assert!((a - b).abs() < 1e-4, "elem {i}: bare {a} vs padded {b}");
        }
    }

    #[test]
    #[should_panic(expected = "mask length")]
    fn embed_rejects_mismatched_mask() {
        let device = crate::backend::cpu_device();
        let cfg = toy_config();
        let loaded = LoadedMiniLm {
            model: build(&cfg, &device),
            config: cfg,
        };
        let _ = loaded.embed_ids(&[1, 2, 3], &[1.0, 1.0], &device);
    }
}
