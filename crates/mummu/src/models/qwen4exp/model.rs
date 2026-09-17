//! The qwen4exp causal LM: the trunk assembled from the building blocks
//! (hyper-connections, PLE, routed experts) and the reused qwen35 Gated
//! DeltaNet / gated attention, plus the streaming GGUF loader.
//!
//! Forward (llama.cpp `src/models/qwen4exp.cpp`, `graph::graph`):
//!
//! ```text
//! res = H copies of token_embd[ids]                      [b, t, H·E], stream-major
//! for L in 0..48:
//!     if L is the PLE layer:  res = PLE(res, ple_emb(ids))
//!     m, inj = hc_attn.mix(res);  res = combine(res, GDN(m) or ATTN(m), inj)
//!     m, inj = hc_ffn.mix(res);   res = combine(res, MOE(m), inj)
//! logits = output · output_hc.mix(res)[last token]        (the final mixer IS the output norm)
//! MOE(m) = routed_top10(m) + shared(m) · sigmoid(ffn_gate_inp_shexp · m)
//! ```
//!
//! **Placement (milestone 1).** Everything that is not an expert bank or the
//! PLE table is dequantized to f32 on the flex device (~20 GB). The 144
//! expert banks (~77 GB quantized) stay on disk and are served per call by
//! [`RoutedExperts`]; the 28.8 GB PLE table likewise by [`PleTable`]. The
//! loader streams one trunk tensor at a time, so peak memory is the finished
//! trunk plus one f32 tensor.
//!
//! **Attention is dense.** The shipped file carries a sparse-attention
//! indexer (QSA). While the cache holds at most
//! [`Qwen4expConfig::dense_attention_limit`] tokens (2051) the indexer
//! selects every cell and dense attention is bit-identical; past that this
//! port would compute a different model, so the forward refuses loudly. The
//! indexer tensors are mapped to an explicit skip.

use std::collections::HashMap;
use std::path::Path;

use burn::module::Param;
use burn::nn::{Embedding, Linear};
use burn::tensor::{Device, Int, Tensor, TensorData, activation};

use super::Qwen4expConfig;
use super::experts::{self, RoutedExperts};
use super::hc::HyperConnection;
use super::ple::{PLE_TABLE_TENSOR, PleBlock, PleConvState, PleHash, PleTable};
use super::teacher;
use crate::gguf::GgufFile;
use crate::import::ImportError;
use crate::models::CausalLm;
use crate::models::qwen35::{
    DeltaState, GatedAttention, GatedDeltaNet, Qwen35Config, Qwen35Kv, device_tensor,
    linear_weight, qlinear, qlinear2,
};
use crate::nn::{causal_mask, rope_tables};
use crate::quant::QuantPolicy;

// ---- Trace (parity debugging) ------------------------------------------------

/// `MUMMU_QWEN4EXP_TRACE=1`: print the named intermediates llama.cpp's
/// `llama-eval-callback` prints (whole-tensor sum, last-token first/last 3
/// values per stream), so a parity failure can be walked to the FIRST op
/// that diverges instead of guessed at. One env read per process.
pub(super) fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MUMMU_QWEN4EXP_TRACE")
            .is_ok_and(|v| !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("off")))
    })
}

/// Print one `[b, t, W]` intermediate: the sum over every element, then the
/// last token's `W / groups`-wide slices (stream-major for HC tensors) as
/// the first and last three values at 4 decimals — the reference summary's
/// format. Call only under [`trace_enabled`] (it reads the tensor back).
pub(super) fn trace3(name: &str, x: &Tensor<3>, groups: usize) {
    let [b, t, w] = x.dims();
    let vals = x
        .clone()
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("trace readback");
    let sum: f64 = vals.iter().map(|&v| f64::from(v)).sum();
    eprintln!("[qwen4exp-trace] {name}: ne=[{b}, {t}, {w}] sum={sum:.6}");
    let last = &vals[(b * t - 1) * w..];
    let width = w / groups.max(1);
    for g in 0..groups.max(1) {
        let s = &last[g * width..(g + 1) * width];
        let fmt = |v: &[f32]| {
            v.iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        if s.len() <= 6 {
            eprintln!("[qwen4exp-trace]   [{}]", fmt(s));
        } else {
            eprintln!(
                "[qwen4exp-trace]   [{}, ..., {}]",
                fmt(&s[..3]),
                fmt(&s[s.len() - 3..])
            );
        }
    }
}

// ---- The trunk tensor map ----------------------------------------------------

/// How a trunk tensor's f32 values become a device parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `token_embd`: row-major `[vocab, E]`, kept float for the gather.
    Embedding,
    /// A projection: GGUF row-major `[out, in]` → burn `[in, out]`.
    Linear,
    /// A 1-D vector (norm gammas, `ssm_a`, `ssm_dt`, the shared gate).
    Vector,
    /// `ssm_conv1d`: ggml `[k, channels]` → burn Conv1d `[channels, 1, k]`
    /// (the same bytes, as qwen35's map).
    DeltaConv,
    /// `ple_conv1d`: the flat ggml order kept as is (tap `k` of channel `c`
    /// at `k + kernel·c`), which is what [`PleBlock`] indexes.
    FlatKernel,
}

/// One trunk tensor the architecture requires: its GGUF name, ggml `ne`,
/// and how it is ingested. The single source of truth for the loader's
/// shape check, its completeness count, and the toy model's weights.
#[derive(Debug, Clone)]
struct TrunkSpec {
    name: String,
    ne: Vec<u64>,
    kind: Kind,
}

/// Every trunk tensor `cfg` requires, in a stable order.
fn trunk_specs(cfg: &Qwen4expConfig) -> Vec<TrunkSpec> {
    let u = |v: usize| v as u64;
    let (e, h, r) = (
        u(cfg.hidden_size),
        u(cfg.hyper_connection_count),
        u(cfg.hyper_connection_low_rank),
    );
    let he = h * e;
    let mut out = Vec::new();
    let mut push = |name: String, ne: Vec<u64>, kind: Kind| out.push(TrunkSpec { name, ne, kind });
    push(
        "token_embd.weight".into(),
        vec![e, u(cfg.vocab_size)],
        Kind::Embedding,
    );
    push(
        "output.weight".into(),
        vec![e, u(cfg.vocab_size)],
        Kind::Linear,
    );
    push("output_hc_norm.weight".into(), vec![he], Kind::Vector);
    push("output_hc_down.weight".into(), vec![he, r], Kind::Linear);
    push("output_hc_up.weight".into(), vec![r, he], Kind::Linear);
    for l in 0..cfg.num_layers {
        let p = |f: &str| format!("blk.{l}.{f}");
        for side in ["attn", "ffn"] {
            push(p(&format!("hc_{side}_norm.weight")), vec![he], Kind::Vector);
            push(
                p(&format!("hc_{side}_down.weight")),
                vec![he, r],
                Kind::Linear,
            );
            push(
                p(&format!("hc_{side}_up.weight")),
                vec![r, he],
                Kind::Linear,
            );
            push(
                p(&format!("hc_{side}_inject.weight")),
                vec![he, h],
                Kind::Linear,
            );
        }
        let s = u(cfg.expert_shared_ffn_size);
        push(
            p("ffn_gate_inp.weight"),
            vec![e, u(cfg.expert_count)],
            Kind::Linear,
        );
        push(p("ffn_gate_inp_shexp.weight"), vec![e], Kind::Vector);
        push(p("ffn_gate_shexp.weight"), vec![e, s], Kind::Linear);
        push(p("ffn_up_shexp.weight"), vec![e, s], Kind::Linear);
        push(p("ffn_down_shexp.weight"), vec![s, e], Kind::Linear);
        if cfg.is_attention(l) {
            let (nh, nkv, hd) = (
                u(cfg.num_attention_heads),
                u(cfg.num_key_value_heads),
                u(cfg.head_dim),
            );
            push(p("attn_q.weight"), vec![e, 2 * nh * hd], Kind::Linear);
            push(p("attn_k.weight"), vec![e, nkv * hd], Kind::Linear);
            push(p("attn_v.weight"), vec![e, nkv * hd], Kind::Linear);
            push(p("attn_output.weight"), vec![nh * hd, e], Kind::Linear);
            push(p("attn_q_norm.weight"), vec![hd], Kind::Vector);
            push(p("attn_k_norm.weight"), vec![hd], Kind::Vector);
        } else {
            let (nv, ds, di) = (u(cfg.n_v_heads), u(cfg.d_state), u(cfg.d_inner));
            let conv = u(cfg.conv_dim());
            push(p("attn_qkv.weight"), vec![e, conv], Kind::Linear);
            push(p("attn_gate.weight"), vec![e, di], Kind::Linear);
            push(
                p("ssm_conv1d.weight"),
                vec![u(cfg.conv_kernel), conv],
                Kind::DeltaConv,
            );
            push(p("ssm_dt.bias"), vec![nv], Kind::Vector);
            push(p("ssm_a"), vec![nv], Kind::Vector);
            push(p("ssm_beta.weight"), vec![e, nv], Kind::Linear);
            push(p("ssm_alpha.weight"), vec![e, nv], Kind::Linear);
            push(p("ssm_norm.weight"), vec![ds], Kind::Vector);
            push(p("ssm_out.weight"), vec![di, e], Kind::Linear);
        }
        if cfg.ple_layers.contains(&l) {
            let e_ple = u(cfg.ple_embed_width());
            push(p("ple_key.weight"), vec![e_ple, he], Kind::Linear);
            push(p("ple_value.weight"), vec![e_ple, e], Kind::Linear);
            push(
                p("ple_conv1d.weight"),
                vec![u(cfg.ple_conv_kernel), he],
                Kind::FlatKernel,
            );
            for n in ["key", "query", "conv"] {
                push(p(&format!("ple_norm_{n}.weight")), vec![he], Kind::Vector);
            }
        }
    }
    out
}

/// What the loader does with a GGUF tensor that is not a trunk tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NonTrunk {
    /// Served on demand from the shards: the routed expert banks
    /// ([`RoutedExperts`]) and the PLE table ([`PleTable`]). Never loaded —
    /// together they are ~106 GB of the 111 GB file.
    Served,
    /// The QSA indexer (`indexer.{q,k}_{proj,norm}` on attention layers).
    /// Skipped on purpose: dense attention is exact inside
    /// [`Qwen4expConfig::dense_attention_limit`], and the forward refuses
    /// past it, so these weights are provably unused.
    SkippedIndexer,
}

/// Classify a non-trunk GGUF tensor name, or `None` when the architecture
/// has no place for it (a loud load error).
fn non_trunk(name: &str, cfg: &Qwen4expConfig) -> Option<NonTrunk> {
    if name == PLE_TABLE_TENSOR {
        return (!cfg.ple_layers.is_empty()).then_some(NonTrunk::Served);
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= cfg.num_layers {
        return None;
    }
    match field {
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            Some(NonTrunk::Served)
        }
        "indexer.q_proj.weight"
        | "indexer.k_proj.weight"
        | "indexer.q_norm.weight"
        | "indexer.k_norm.weight"
            if cfg.is_attention(layer) =>
        {
            Some(NonTrunk::SkippedIndexer)
        }
        _ => None,
    }
}

/// A trunk tensor on the device in its final form.
enum Loaded {
    T1(Tensor<1>),
    T2(Tensor<2>),
    T3(Tensor<3>),
}

/// Turn one trunk tensor's f32 values (ggml order, as
/// [`GgufFile::read_tensor_f32`] returns them) into its device parameter.
fn ingest(spec: &TrunkSpec, values: Vec<f32>, device: &Device) -> Result<Loaded, String> {
    let dims: Vec<usize> = spec.ne.iter().map(|&d| d as usize).collect();
    let n: usize = dims.iter().product();
    if values.len() != n {
        return Err(format!(
            "{}: {} values for ne {:?}",
            spec.name,
            values.len(),
            spec.ne
        ));
    }
    Ok(match (spec.kind, dims.as_slice()) {
        (Kind::Embedding, &[e, vocab]) => {
            Loaded::T2(device_tensor::<2>(values, [vocab, e], device))
        }
        (Kind::Linear, &[inp, out]) => Loaded::T2(linear_weight(
            values,
            &[out, inp],
            QuantPolicy::Off,
            device,
        )?),
        (Kind::Vector, &[len]) => Loaded::T1(device_tensor::<1>(values, [len], device)),
        (Kind::DeltaConv, &[k, ch]) => Loaded::T3(device_tensor::<3>(values, [ch, 1, k], device)),
        (Kind::FlatKernel, &[k, ch]) => Loaded::T1(device_tensor::<1>(values, [k * ch], device)),
        (kind, _) => {
            return Err(format!(
                "{}: ne {:?} does not fit {kind:?}",
                spec.name, spec.ne
            ));
        }
    })
}

/// Trunk tensors by GGUF name, consumed as the model is assembled.
struct TrunkWeights(HashMap<String, Loaded>);

impl TrunkWeights {
    fn take(&mut self, name: &str) -> Result<Loaded, String> {
        self.0
            .remove(name)
            .ok_or_else(|| format!("trunk tensor {name} was not loaded"))
    }
    fn t1(&mut self, name: &str) -> Result<Tensor<1>, String> {
        match self.take(name)? {
            Loaded::T1(t) => Ok(t),
            _ => Err(format!("{name} is not 1-D")),
        }
    }
    fn t2(&mut self, name: &str) -> Result<Tensor<2>, String> {
        match self.take(name)? {
            Loaded::T2(t) => Ok(t),
            _ => Err(format!("{name} is not 2-D")),
        }
    }
    fn t3(&mut self, name: &str) -> Result<Tensor<3>, String> {
        match self.take(name)? {
            Loaded::T3(t) => Ok(t),
            _ => Err(format!("{name} is not 3-D")),
        }
    }
    fn linear(&mut self, name: &str) -> Result<Linear, String> {
        Ok(Linear {
            weight: Param::from_tensor(self.t2(name)?),
            bias: None,
        })
    }
}

// ---- The model -----------------------------------------------------------------

/// One block: the two hyper-connection mixers, exactly one token mixer
/// (gated attention or Gated DeltaNet), the PLE block on the PLE layer, and
/// the MoE's trunk half (router + shared expert; the routed experts live in
/// [`RoutedExperts`]).
#[derive(Debug)]
pub struct Qwen4expLayer {
    pub hc_attn: HyperConnection,
    pub hc_ffn: HyperConnection,
    pub self_attn: Option<GatedAttention>,
    pub linear_attn: Option<GatedDeltaNet>,
    pub ple: Option<PleBlock>,
    /// `E → n_experts` router logits (`ffn_gate_inp`, F32 in the file).
    pub router: Linear,
    pub shared_gate_proj: Linear,
    pub shared_up_proj: Linear,
    pub shared_down_proj: Linear,
    /// `ffn_gate_inp_shexp`: a VECTOR, one sigmoid gate per token
    /// (`sigmoid(dot(v, m))`), not a projection.
    pub shared_expert_gate: Param<Tensor<1>>,
}

/// The trunk: embedding, blocks, the output mixer (which is the final norm)
/// and the untied head.
#[derive(Debug)]
pub struct Qwen4exp {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen4expLayer>,
    pub output_hc: HyperConnection,
    pub lm_head: Linear,
}

/// The PLE lookup half: the hash parameters and the on-disk table.
#[derive(Debug)]
pub struct PleInput {
    pub hash: PleHash,
    pub table: PleTable,
}

/// A loaded qwen4exp: trunk, config, and the on-demand stores.
#[derive(Debug)]
pub struct LoadedQwen4exp {
    pub model: Qwen4exp,
    pub config: Qwen4expConfig,
    /// [`Qwen4expConfig::blocks_config`], built once (sigmoid DeltaNet gate).
    pub blocks: Qwen35Config,
    pub experts: RoutedExperts,
    /// `None` only for a header without PLE.
    pub ple: Option<PleInput>,
}

/// Per-generation state: one KV cache or DeltaNet state per layer, the PLE
/// conv history (9 columns of `H·E`), the last `ngram-1` token ids the PLE
/// hash reads, and how many tokens have been consumed.
pub struct Qwen4expCache {
    pub layers: Vec<Qwen35Kv>,
    pub ple_conv: Option<PleConvState>,
    pub ple_history: Vec<u32>,
    pub position: usize,
}

/// Assemble the model from ingested trunk tensors. Consumes every entry;
/// a leftover is an error (it would be a silently ignored weight).
fn assemble(
    cfg: &Qwen4expConfig,
    mut w: TrunkWeights,
    experts: RoutedExperts,
    ple: Option<PleInput>,
    device: &Device,
) -> Result<LoadedQwen4exp, String> {
    cfg.validate()?;
    let blocks = cfg.blocks_config();
    let (e, h, eps) = (
        cfg.hidden_size,
        cfg.hyper_connection_count,
        cfg.rms_norm_eps,
    );
    let hc =
        |w: &mut TrunkWeights, prefix: &str, inject: bool| -> Result<HyperConnection, String> {
            Ok(HyperConnection::from_weights(
                w.t1(&format!("{prefix}_norm.weight"))?,
                w.t2(&format!("{prefix}_down.weight"))?,
                w.t2(&format!("{prefix}_up.weight"))?,
                if inject {
                    Some(w.t2(&format!("{prefix}_inject.weight"))?)
                } else {
                    None
                },
                e,
                h,
                eps,
            ))
        };

    if experts.n_layers() != cfg.num_layers
        || experts.n_experts() != cfg.expert_count
        || experts.hidden_size() != e
        || experts.ffn_size() != cfg.expert_ffn_size
    {
        return Err(format!(
            "expert store is {} layers x {} experts ({} -> {}), config wants {} x {} ({e} -> {})",
            experts.n_layers(),
            experts.n_experts(),
            experts.hidden_size(),
            experts.ffn_size(),
            cfg.num_layers,
            cfg.expert_count,
            cfg.expert_ffn_size
        ));
    }
    match (&ple, cfg.ple_layers.is_empty()) {
        (Some(p), false) => {
            if p.hash.n_heads() * p.table.row_width() != cfg.ple_embed_width()
                || p.hash.eos != cfg.ple_eos_token_id
            {
                return Err(format!(
                    "PLE store ({} heads x {} wide, eos {}) disagrees with the config ({} wide, eos {})",
                    p.hash.n_heads(),
                    p.table.row_width(),
                    p.hash.eos,
                    cfg.ple_embed_width(),
                    cfg.ple_eos_token_id
                ));
            }
        }
        (None, true) => {}
        _ => return Err("PLE store presence disagrees with qwen4exp.ple.layers".into()),
    }

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for l in 0..cfg.num_layers {
        let p = |f: &str| format!("blk.{l}.{f}");
        let hc_attn = hc(&mut w, &p("hc_attn"), true)?;
        let hc_ffn = hc(&mut w, &p("hc_ffn"), true)?;
        let (self_attn, linear_attn) = if cfg.is_attention(l) {
            // `init` is lazy (burn params initialize on first read), so the
            // placeholders cost nothing before being replaced.
            let mut a = GatedAttention::init(&blocks, device);
            a.q_proj = w.linear(&p("attn_q.weight"))?;
            a.k_proj = w.linear(&p("attn_k.weight"))?;
            a.v_proj = w.linear(&p("attn_v.weight"))?;
            a.o_proj = w.linear(&p("attn_output.weight"))?;
            a.q_norm.gamma = Param::from_tensor(w.t1(&p("attn_q_norm.weight"))?);
            a.k_norm.gamma = Param::from_tensor(w.t1(&p("attn_k_norm.weight"))?);
            (Some(a), None)
        } else {
            // Built through `init` for the conv's causal padding (see its
            // docs); only the weights are replaced.
            let mut d = GatedDeltaNet::init(&blocks, device);
            d.qkv_proj = w.linear(&p("attn_qkv.weight"))?;
            d.z_proj = w.linear(&p("attn_gate.weight"))?;
            d.beta_proj = w.linear(&p("ssm_beta.weight"))?;
            d.alpha_proj = w.linear(&p("ssm_alpha.weight"))?;
            d.dt_bias = Param::from_tensor(w.t1(&p("ssm_dt.bias"))?);
            d.a = Param::from_tensor(w.t1(&p("ssm_a"))?);
            d.conv1d.weight = Param::from_tensor(w.t3(&p("ssm_conv1d.weight"))?);
            d.norm.gamma = Param::from_tensor(w.t1(&p("ssm_norm.weight"))?);
            d.out_proj = w.linear(&p("ssm_out.weight"))?;
            (None, Some(d))
        };
        let ple_block = if cfg.ple_layers.contains(&l) {
            Some(PleBlock::from_weights(
                w.t2(&p("ple_key.weight"))?,
                w.t2(&p("ple_value.weight"))?,
                w.t1(&p("ple_norm_key.weight"))?,
                w.t1(&p("ple_norm_query.weight"))?,
                w.t1(&p("ple_norm_conv.weight"))?,
                w.t1(&p("ple_conv1d.weight"))?,
                e,
                h,
                cfg.ple_conv_kernel,
                // Dilation is the n-gram size in both references.
                cfg.ple_ngram_size,
                eps,
            ))
        } else {
            None
        };
        layers.push(Qwen4expLayer {
            hc_attn,
            hc_ffn,
            self_attn,
            linear_attn,
            ple: ple_block,
            router: w.linear(&p("ffn_gate_inp.weight"))?,
            shared_gate_proj: w.linear(&p("ffn_gate_shexp.weight"))?,
            shared_up_proj: w.linear(&p("ffn_up_shexp.weight"))?,
            shared_down_proj: w.linear(&p("ffn_down_shexp.weight"))?,
            shared_expert_gate: Param::from_tensor(w.t1(&p("ffn_gate_inp_shexp.weight"))?),
        });
    }
    let model = Qwen4exp {
        embed_tokens: Embedding {
            weight: Param::from_tensor(w.t2("token_embd.weight")?),
        },
        layers,
        output_hc: hc(&mut w, "output_hc", false)?,
        lm_head: w.linear("output.weight")?,
    };
    if !w.0.is_empty() {
        let mut left: Vec<_> = w.0.keys().cloned().collect();
        left.sort();
        return Err(format!(
            "trunk tensors loaded but never assembled: {left:?}"
        ));
    }
    Ok(LoadedQwen4exp {
        model,
        config: cfg.clone(),
        blocks,
        experts,
        ple,
    })
}

/// Load a qwen4exp split GGUF from its FIRST shard onto `device`.
///
/// Streams the trunk: each of the 1031 trunk tensors is dequantized to f32
/// ([`GgufFile::read_tensor_f32`]), shape-checked against [`trunk_specs`],
/// moved to the device and assembled, so peak memory is the f32 trunk
/// (~20 GB) plus one tensor. The 144 expert banks and the PLE table are
/// opened as on-demand stores, and the 48 indexer tensors are skipped.
/// Completeness is enforced both ways: an unknown file tensor is an error,
/// and every required trunk tensor must arrive exactly once.
///
/// The routed-expert f32 cache size comes from
/// [`experts::CACHE_ENV`] (default [`experts::DEFAULT_CACHE_GB`] GiB).
pub fn load_from_gguf(first_shard: &Path, device: &Device) -> Result<LoadedQwen4exp, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: first_shard.to_path_buf(),
        reason,
    };
    let f = GgufFile::open_sharded(first_shard).map_err(|e| parse(e.to_string()))?;
    let config = Qwen4expConfig::from_gguf(&f).map_err(parse)?;
    let specs: HashMap<String, TrunkSpec> = trunk_specs(&config)
        .into_iter()
        .map(|s| (s.name.clone(), s))
        .collect();

    let mut loaded: HashMap<String, Loaded> = HashMap::with_capacity(specs.len());
    let (mut served, mut skipped) = (0usize, 0usize);
    for info in &f.tensors {
        if let Some(spec) = specs.get(&info.name) {
            if info.dims != spec.ne {
                return Err(parse(format!(
                    "{}: ne {:?}, the architecture needs {:?}",
                    info.name, info.dims, spec.ne
                )));
            }
            if spec.kind == Kind::Linear && info.dtype != crate::gguf::GgmlType::F32 {
                // `nn::refarith` needs each quantized linear's file dtype to
                // pick llama.cpp's activation grid; inert unless enabled.
                crate::nn::refarith::register_linear(
                    spec.ne[0] as usize,
                    spec.ne[1] as usize,
                    info.dtype,
                );
            }
            let values = f
                .read_tensor_f32(&info.name)
                .map_err(|e| parse(e.to_string()))?;
            let t = ingest(spec, values, device).map_err(parse)?;
            if loaded.insert(info.name.clone(), t).is_some() {
                return Err(parse(format!("{} appears twice", info.name)));
            }
            continue;
        }
        match non_trunk(&info.name, &config) {
            Some(NonTrunk::Served) => served += 1,
            Some(NonTrunk::SkippedIndexer) => skipped += 1,
            None => {
                return Err(parse(format!(
                    "unmapped tensor '{}' (ne {:?})",
                    info.name, info.dims
                )));
            }
        }
    }
    let want_served = 3 * config.num_layers + usize::from(!config.ple_layers.is_empty());
    if loaded.len() != specs.len() || served != want_served {
        let mut missing: Vec<_> = specs
            .keys()
            .filter(|k| !loaded.contains_key(*k))
            .cloned()
            .collect();
        missing.sort();
        return Err(parse(format!(
            "GGUF supplied {} of {} trunk tensors (missing {missing:?}) and {served} of {want_served} served tensors",
            loaded.len(),
            specs.len()
        )));
    }
    debug_assert_eq!(
        loaded.len() + served + skipped,
        f.tensors.len(),
        "every file tensor is accounted for"
    );

    let experts = RoutedExperts::open(&f, config.num_layers, experts::cache_bytes_from_env())
        .map_err(parse)?;
    let ple = if config.ple_layers.is_empty() {
        None
    } else {
        let hash = PleHash::from_gguf(&f).map_err(parse)?;
        let table = PleTable::open(&f, hash.total_rows()).map_err(parse)?;
        Some(PleInput { hash, table })
    };
    assemble(&config, TrunkWeights(loaded), experts, ple, device).map_err(parse)
}

impl LoadedQwen4exp {
    /// The MoE block over the HC-mixed `m [b, t, E]`: routed top-k experts
    /// (host, from the banks) plus the sigmoid-gated shared expert.
    fn moe(&self, li: usize, layer: &Qwen4expLayer, m: Tensor<3>) -> Tensor<3> {
        let [b, t, e] = m.dims();
        let device = m.device();
        let host = |x: Tensor<3>| -> Vec<f32> {
            x.into_data()
                .convert::<f32>()
                .try_into_vec::<f32>()
                .expect("activation readback")
        };
        // F32 router: a plain matmul (never activation-quantized, and its
        // 2560->512 shape collides with the Q8_0 attn_k/attn_v registry key).
        let logits = m
            .clone()
            .reshape([b * t, e])
            .matmul(layer.router.weight.val())
            .reshape([b, t, self.config.expert_count]);
        let logits = teacher::teach(&format!("ffn_moe_logits-{li}"), 0, logits);
        let logits = host(logits);
        if trace_enabled() {
            // The last token's routing, as llama.cpp's ffn_moe_topk /
            // ffn_moe_weights_norm print it (slot order, best first).
            let n_exp = self.config.expert_count;
            let last = &logits[(b * t - 1) * n_exp..];
            if let Ok(r) = experts::route(last, 1, n_exp, self.config.expert_used_count) {
                let w: Vec<String> = r.weights.iter().map(|v| format!("{v:.4}")).collect();
                eprintln!(
                    "[qwen4exp-trace] ffn_moe_topk-{li} last token: {:?} weights [{}]",
                    r.ids,
                    w.join(", ")
                );
            }
        }
        let routed = self
            .experts
            .routed_moe(
                li,
                &logits,
                &host(m.clone()),
                b * t,
                self.config.expert_used_count,
            )
            .unwrap_or_else(|err| panic!("qwen4exp layer {li} routed experts: {err}"));
        let routed = Tensor::<3>::from_data(
            TensorData::new(routed, [b, t, e]),
            (&device, crate::backend::float_dtype(&device)),
        );
        let gate = activation::silu(qlinear(&layer.shared_gate_proj, m.clone()));
        let up = qlinear(&layer.shared_up_proj, m.clone());
        let shared = qlinear(&layer.shared_down_proj, gate.mul(up));
        let sgate = activation::sigmoid(
            m.reshape([b * t, e])
                .matmul(layer.shared_expert_gate.val().reshape([e, 1]))
                .reshape([b, t, 1]),
        );
        let sgate = teacher::teach(&format!("shared_expert_gate_sigmoid-{li}"), 0, sgate);
        let shared = shared.mul(sgate.clone());
        let routed = teacher::teach(&format!("ffn_moe_out-{li}"), 0, routed);
        let shared = teacher::teach(&format!("ffn_shexp_gated-{li}"), 0, shared);
        if trace_enabled() {
            trace3(&format!("ffn_moe_out-{li}"), &routed, 1);
            trace3(&format!("shared_expert_gate_sigmoid-{li}"), &sgate, 1);
            trace3(&format!("ffn_shexp_gated-{li}"), &shared, 1);
        }
        routed.add(shared)
    }

    /// The shared body of `forward` / `forward_advance`.
    fn forward_impl(
        &self,
        ids: &[u32],
        past: usize,
        cache: &mut Qwen4expCache,
        device: &Device,
        need_logits: bool,
    ) -> Option<Tensor<2>> {
        let cfg = &self.config;
        let t = ids.len();
        assert!(t >= 1, "qwen4exp forward: need at least one token");
        assert_eq!(
            cache.layers.len(),
            cfg.num_layers,
            "qwen4exp forward: cache has the wrong layer count"
        );
        // The PLE hash and conv history are positional state: a `past` that
        // disagrees with what the cache consumed would hash wrong n-grams.
        assert_eq!(
            past, cache.position,
            "qwen4exp forward: past {past} but the cache holds {} tokens",
            cache.position
        );
        if let Some(limit) = cfg.dense_attention_limit() {
            assert!(
                past + t <= limit,
                "qwen4exp forward: {past}+{t} tokens exceed {limit}, the longest context where \
                 dense attention equals the model's sparse (QSA) attention; the QSA indexer is \
                 not implemented, so this would compute a different model"
            );
        }
        let trace = trace_enabled();
        let (h, e) = (cfg.hyper_connection_count, cfg.hidden_size);

        let embed_device = self.model.embed_tokens.weight.val().device();
        let ids32: Vec<i32> = ids
            .iter()
            .map(|&i| i32::try_from(i).expect("token id fits i32"))
            .collect();
        let input = Tensor::<1, Int>::from_data(
            TensorData::new(ids32, [t]),
            (&embed_device, crate::backend::int_dtype(&embed_device)),
        )
        .reshape([1, t]);
        let x = self.model.embed_tokens.forward(input).to_device(device); // [1, t, E]
        let x = crate::nn::refarith::perturb_embedding(x);
        let x = teacher::teach("model.input_embed", 0, x);
        if trace {
            trace3("model.input_embed", &x, 1);
        }
        // hc identical copies of the embedding, stream-major.
        let mut res = x
            .reshape([1, t, 1, e])
            .repeat_dim(2, h)
            .reshape([1, t, h * e]);

        let (cos, sin) = rope_tables(t, past, cfg.rope_dim, cfg.rope_theta, device);
        let mask = (t > 1).then(|| causal_mask(t, past, device));

        for (li, layer) in self.model.layers.iter().enumerate() {
            if let Some(block) = &layer.ple {
                let ple = self
                    .ple
                    .as_ref()
                    .expect("a PLE block implies a PLE store (checked at assembly)");
                let emb = ple
                    .table
                    .embed_tensor(&ple.hash, &cache.ple_history, ids, device)
                    .unwrap_or_else(|err| panic!("qwen4exp PLE rows: {err}"));
                let emb = teacher::teach("ple_embd", 0, emb);
                if trace {
                    trace3("ple_embd", &emb, 1);
                }
                let state = cache
                    .ple_conv
                    .as_mut()
                    .expect("a PLE model's cache carries the conv history");
                res = teacher::teach_ple_out(li, block.forward(res, emb, state));
                if trace {
                    trace3(&format!("ple_out-{li}"), &res, h);
                }
            }

            let (m, inject) = layer.hc_attn.mix(res.clone());
            let inject = inject.expect("block mixers carry an inject projection");
            let m = teacher::teach(&format!("hc_mixed-{li}"), 0, m);
            let inject = teacher::teach(&format!("hc_inject-{li}"), 0, inject);
            let out = match (&layer.self_attn, &layer.linear_attn, &mut cache.layers[li]) {
                (Some(attn), None, Qwen35Kv::Attn(kv)) => {
                    attn.forward(m.clone(), &self.blocks, &cos, &sin, mask.as_ref(), kv)
                }
                (None, Some(delta), Qwen35Kv::Delta(state)) => {
                    delta.forward(m.clone(), &self.blocks, state)
                }
                _ => unreachable!("qwen4exp forward: layer/cache kind mismatch at {li}"),
            };
            let out = if layer.self_attn.is_some() {
                teacher::teach(&format!("attn_output-{li}"), 0, out)
            } else {
                teacher::teach(&format!("linear_attn_out-{li}"), 0, out)
            };
            if trace {
                trace3(&format!("hc_mixed-{li} #1"), &m, 1);
                trace3(&format!("hc_inject-{li} #1"), &inject, 1);
                let kind = if layer.self_attn.is_some() {
                    "attn_output"
                } else {
                    "linear_attn_out"
                };
                trace3(&format!("{kind}-{li}"), &out, 1);
            }
            res = teacher::teach(
                &format!("hc_combine-{li}"),
                0,
                layer.hc_attn.combine(res, out, inject),
            );
            if trace {
                trace3(&format!("hc_combine-{li}"), &res, h);
            }

            let (m, inject) = layer.hc_ffn.mix(res.clone());
            let inject = inject.expect("block mixers carry an inject projection");
            let m = teacher::teach(&format!("hc_mixed-{li}"), 1, m);
            let inject = teacher::teach(&format!("hc_inject-{li}"), 1, inject);
            if trace {
                trace3(&format!("hc_mixed-{li} #2"), &m, 1);
                trace3(&format!("hc_inject-{li} #2"), &inject, 1);
            }
            let out = teacher::teach(&format!("ffn_out-{li}"), 0, self.moe(li, layer, m));
            if trace {
                trace3(&format!("ffn_out-{li}"), &out, 1);
            }
            res = teacher::teach(
                &format!("l_last-{li}"),
                0,
                layer.hc_ffn.combine(res, out, inject),
            );
            if trace {
                trace3(&format!("l_last-{li}"), &res, h);
            }
        }
        if let Some(ple) = &self.ple {
            ple.hash.advance_history(&mut cache.ple_history, ids);
        }
        cache.position += t;
        if !need_logits {
            return None;
        }

        // Every op after the last block is per token: mix only the last one.
        let last = res.narrow(1, t - 1, 1); // [1, 1, H·E]
        let (hn, _) = self.model.output_hc.mix(last);
        let hn = teacher::teach("result_norm", 0, hn);
        if trace {
            trace3("result_norm", &hn, 1);
        }
        let logits = qlinear2(&self.model.lm_head, hn.reshape([1, e]));
        if teacher::active() {
            let v = cfg.vocab_size;
            let _ = teacher::teach("result_output", 0, logits.clone().reshape([1, 1, v]));
        }
        if trace {
            trace3(
                "result_output",
                &logits.clone().reshape([1, 1, cfg.vocab_size]),
                1,
            );
        }
        Some(logits)
    }
}

impl CausalLm for LoadedQwen4exp {
    type Cache = Qwen4expCache;

    fn new_cache(&self) -> Qwen4expCache {
        Qwen4expCache {
            layers: (0..self.config.num_layers)
                .map(|l| {
                    if self.config.is_attention(l) {
                        Qwen35Kv::Attn(None)
                    } else {
                        Qwen35Kv::Delta(DeltaState::empty())
                    }
                })
                .collect(),
            ple_conv: self
                .model
                .layers
                .iter()
                .find_map(|l| l.ple.as_ref())
                .map(|b| b.new_state(1)),
            ple_history: Vec::new(),
            position: 0,
        }
    }

    fn forward(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Qwen4expCache,
        device: &Device,
    ) -> Tensor<2> {
        self.forward_impl(new_ids, past, cache, device, true)
            .expect("forward_impl returns logits when asked")
    }

    fn forward_advance(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Qwen4expCache,
        device: &Device,
    ) {
        let _ = self.forward_impl(new_ids, past, cache, device, false);
    }

    /// The tokenizer EOS (248046, `<|im_end|>`) and the PLE reset token
    /// (248044, `<|endoftext|>`): either ends a generation.
    fn is_eos(&self, id: u32) -> bool {
        id == self.config.eos_token_id || id == self.config.ple_eos_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::super::experts::{ExpertBank, LayerExperts};
    use super::*;
    use crate::gguf::GgmlType;

    /// Deterministic xorshift in `[-1, 1)`.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * scale).collect()
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A tiny qwen4exp with every mechanism present: 4 layers (0, 1, 3
    /// DeltaNet with the PLE block on 1; 2 attention), 2 HC streams, 6
    /// experts top-2, a 4-head PLE hash whose reset token (1) differs from
    /// the tokenizer EOS (0), and a dense-attention bound of 16 + 4 - 1.
    fn toy_config() -> Qwen4expConfig {
        Qwen4expConfig {
            vocab_size: 40,
            hidden_size: 16,
            num_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1e4,
            rope_dim: 4,
            full_attention_interval: 3,
            conv_kernel: 4,
            d_inner: 12,
            d_state: 4,
            n_k_heads: 1,
            n_v_heads: 3,
            expert_count: 6,
            expert_used_count: 2,
            expert_ffn_size: 8,
            expert_shared_ffn_size: 8,
            hyper_connection_count: 2,
            hyper_connection_low_rank: 4,
            indexer_head_count: 1,
            indexer_key_length: 4,
            indexer_top_k: 16,
            ple_layers: vec![1],
            ple_ngram_size: 3,
            ple_heads_per_ngram: 2,
            ple_conv_kernel: 4,
            ple_row_width: 4,
            ple_layer_multipliers: vec![1_000_003, 999_983, 998_001],
            ple_head_offsets: vec![0, 7, 18, 31],
            ple_head_vocab_sizes: vec![7, 11, 13, 17],
            ple_eos_token_id: 1,
            ple_image_token_id: None,
            attention_compress_ratios: vec![0, 0, 4, 0],
            eos_token_id: 0,
        }
    }

    /// Random weights pushed through the SAME spec → ingest → assemble path
    /// the loader uses, with in-memory F32 expert banks and PLE table.
    fn toy_model() -> LoadedQwen4exp {
        let cfg = toy_config();
        cfg.validate().expect("toy config validates");
        let device = crate::backend::cpu_device();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut loaded = HashMap::new();
        for spec in trunk_specs(&cfg) {
            let n: usize = spec.ne.iter().map(|&d| d as usize).product();
            let name = spec.name.as_str();
            let values = if spec.kind == Kind::Vector {
                if name.ends_with("ssm_a") {
                    // -exp(A_log) < 0: a real decay.
                    (0..n).map(|_| -0.2 - 0.5 * rng.next().abs()).collect()
                } else if name.ends_with("ssm_dt.bias") || name.contains("gate_inp_shexp") {
                    rng.vec(n, 0.5)
                } else {
                    // Folded (1 + w) gammas sit near 1.
                    (0..n).map(|_| 1.0 + 0.2 * rng.next()).collect()
                }
            } else {
                let fan_in = spec.ne[0] as f32;
                rng.vec(n, 1.5 / fan_in.sqrt())
            };
            let t = ingest(&spec, values, &device).expect("ingest");
            loaded.insert(spec.name.clone(), t);
        }

        let (e, f, ne) = (cfg.hidden_size, cfg.expert_ffn_size, cfg.expert_count);
        let bank = |rng: &mut Rng, inp: usize, out: usize| {
            let v = rng.vec(inp * out * ne, 1.5 / (inp as f32).sqrt());
            ExpertBank::from_bytes(GgmlType::F32, [inp, out, ne], f32_bytes(&v)).expect("bank")
        };
        let layers = (0..cfg.num_layers)
            .map(|_| LayerExperts {
                gate: bank(&mut rng, e, f),
                up: bank(&mut rng, e, f),
                down: bank(&mut rng, f, e),
            })
            .collect();
        let experts = RoutedExperts::from_layers(layers, 0).expect("expert store");

        let hash = PleHash::new(
            cfg.ple_ngram_size,
            cfg.ple_heads_per_ngram,
            cfg.ple_layer_multipliers.clone(),
            cfg.ple_head_vocab_sizes.clone(),
            cfg.ple_head_offsets.clone(),
            cfg.ple_eos_token_id,
        )
        .expect("hash");
        let rows = cfg.ple_total_rows();
        let table_vals = rng.vec(rows as usize * cfg.ple_row_width, 1.0);
        let table =
            PleTable::from_bytes(f32_bytes(&table_vals), GgmlType::F32, rows).expect("table");
        assemble(
            &cfg,
            TrunkWeights(loaded),
            experts,
            Some(PleInput { hash, table }),
            &device,
        )
        .expect("toy model assembles")
    }

    fn logits_vec(t: Tensor<2>) -> Vec<f32> {
        t.into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("logits")
    }

    fn assert_close(what: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{what}: lengths");
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = a.iter().map(|x| x.abs()).fold(0f32, f32::max);
        assert!(scale > 1e-3, "{what}: degenerate logits (max |x| {scale})");
        assert!(
            max < 1e-4 * scale.max(1.0),
            "{what}: max |diff| {max} (scale {scale})"
        );
    }

    /// Ids spanning the PLE reset token (1) mid-sequence, so the hash's EOS
    /// cut, the 9-column conv history and every cache are all exercised.
    const IDS: [u32; 12] = [5, 17, 1, 33, 8, 22, 1, 1, 39, 12, 3, 27];

    /// The shipped file's tensor inventory is exactly trunk + served +
    /// skipped: 1031 + 145 + 48 = 1224 — the completeness count the loader
    /// enforces, checked here against the real config without the file.
    #[test]
    fn the_shipped_config_maps_all_1224_tensors() {
        let cfg = super::super::tests::shipped();
        let specs = trunk_specs(&cfg);
        assert_eq!(specs.len(), 1031, "trunk tensors");
        let names: std::collections::HashSet<_> = specs.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names.len(), specs.len(), "no duplicate trunk names");
        let mut served = usize::from(non_trunk(PLE_TABLE_TENSOR, &cfg) == Some(NonTrunk::Served));
        let mut skipped = 0;
        for l in 0..cfg.num_layers {
            for k in ["gate", "up", "down"] {
                assert_eq!(
                    non_trunk(&format!("blk.{l}.ffn_{k}_exps.weight"), &cfg),
                    Some(NonTrunk::Served)
                );
                served += 1;
            }
            for f in ["q_proj", "k_proj", "q_norm", "k_norm"] {
                let got = non_trunk(&format!("blk.{l}.indexer.{f}.weight"), &cfg);
                if cfg.is_attention(l) {
                    assert_eq!(got, Some(NonTrunk::SkippedIndexer));
                    skipped += 1;
                } else {
                    assert_eq!(got, None, "an indexer on a DeltaNet layer is unmapped");
                }
            }
        }
        assert_eq!((served, skipped), (145, 48));
        assert_eq!(specs.len() + served + skipped, 1224);
        // Unknown names are refused, not skipped.
        assert_eq!(non_trunk("blk.0.nextn.eh_proj.weight", &cfg), None);
        assert_eq!(non_trunk("blk.48.ffn_up_exps.weight", &cfg), None);
    }

    /// Cached single-token decode after a prefill == one-shot prefill: the
    /// KV caches, DeltaNet conv/recurrent state (fused host decode step),
    /// the PLE conv history and the PLE token window all carry exactly.
    #[test]
    fn cached_decode_matches_one_shot_prefill() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut c = m.new_cache();
        let full = logits_vec(m.forward(&IDS, 0, &mut c, &device));

        let mut c = m.new_cache();
        let _ = m.forward(&IDS[..3], 0, &mut c, &device);
        let mut last = Vec::new();
        for (i, &id) in IDS.iter().enumerate().skip(3) {
            last = logits_vec(m.forward(&[id], i, &mut c, &device));
        }
        assert_close("decode vs one-shot", &full, &last);
    }

    /// Chunked prefill (t > 1 on both sides of every split) == one-shot,
    /// including an advance-only first chunk — the bug class PR #42 found
    /// in conv caches. Splits land inside the 9-column PLE conv history and
    /// across the PLE reset token.
    #[test]
    fn chunked_prefill_matches_one_shot() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut c = m.new_cache();
        let full = logits_vec(m.forward(&IDS, 0, &mut c, &device));

        for splits in [vec![2, 7], vec![5], vec![3, 6, 9]] {
            let mut c = m.new_cache();
            let mut start = 0;
            for &end in &splits {
                m.forward_advance(&IDS[start..end], start, &mut c, &device);
                start = end;
            }
            let got = logits_vec(m.forward(&IDS[start..], start, &mut c, &device));
            assert_close(&format!("chunks at {splits:?}"), &full, &got);
        }
    }

    /// The prefix actually matters (the caches are not inert): the same
    /// last token after different prefixes gives different logits.
    #[test]
    fn the_prefix_changes_the_next_token_logits() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut a = m.new_cache();
        let mut b = m.new_cache();
        let _ = m.forward(&[5, 6, 7], 0, &mut a, &device);
        let _ = m.forward(&[9, 10, 11], 0, &mut b, &device);
        let la = logits_vec(m.forward(&[4], 3, &mut a, &device));
        let lb = logits_vec(m.forward(&[4], 3, &mut b, &device));
        let diff = la
            .iter()
            .zip(&lb)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(diff > 1e-4, "different prefixes must change the logits");
    }

    /// Past the QSA-exact window the forward refuses instead of computing a
    /// different model (toy bound: 16 + 4 - 1 = 19 tokens).
    #[test]
    #[should_panic(expected = "sparse (QSA) attention")]
    fn the_forward_refuses_past_the_dense_attention_bound() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut c = m.new_cache();
        let ids: Vec<u32> = (0..20).map(|i| (i * 7 + 2) % 40).collect();
        let _ = m.forward(&ids, 0, &mut c, &device);
    }

    /// Exactly at the bound is still allowed (the refusal is not off by one).
    #[test]
    fn the_forward_accepts_exactly_the_dense_attention_bound() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut c = m.new_cache();
        let ids: Vec<u32> = (0..19).map(|i| (i * 7 + 2) % 40).collect();
        let logits = logits_vec(m.forward(&ids, 0, &mut c, &device));
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// A `past` that disagrees with the cache would hash wrong n-grams.
    #[test]
    #[should_panic(expected = "but the cache holds")]
    fn a_past_that_disagrees_with_the_cache_is_refused() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut c = m.new_cache();
        let _ = m.forward(&[3, 4], 0, &mut c, &device);
        let _ = m.forward(&[5], 3, &mut c, &device);
    }

    #[test]
    fn both_the_tokenizer_eos_and_the_ple_reset_token_stop_generation() {
        let m = toy_model();
        assert!(m.is_eos(0) && m.is_eos(1) && !m.is_eos(2));
    }
}
