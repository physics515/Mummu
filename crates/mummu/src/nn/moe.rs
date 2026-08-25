//! Sparse mixture-of-experts SwiGLU feed-forward (OLMoE-style): a softmax
//! top-k router over a bank of SwiGLU experts stored as **fused 3-D tensors**
//! — exactly the GGUF `ffn_{gate,up,down}_exps` layout, so a checkpoint's
//! expert bank loads as one tensor per projection instead of `num_experts`
//! separate matrices.
//!
//! First cut is **dense-mask compute**: every expert processes every token and
//! the router's sparse weight row (zero for the unrouted experts) scales the
//! results away. That wastes `1 - k/E` of the FLOPs but keeps the whole
//! forward on-device (no data-dependent gather, no host readback of routing
//! decisions) and is numerically identical to the sparse formulation. A
//! gather-based path is a perf follow-up, not a correctness one.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{DType, Device, Distribution, Int, Tensor, TensorData, activation};

/// The expert bank: `num_experts` SwiGLU MLPs as three fused params in
/// `[experts, out, in]` layout (the row-major twin of ggml's
/// `ffn_*_exps.weight`). Forward transposes lazily; no per-expert modules.
#[derive(Module, Debug)]
pub struct MoeExperts {
    /// `[num_experts, intermediate, hidden]` — SiLU branch.
    pub gate: Param<Tensor<3>>,
    /// `[num_experts, intermediate, hidden]` — multiplicative branch.
    pub up: Param<Tensor<3>>,
    /// `[num_experts, hidden, intermediate]` — back to the model width.
    pub down: Param<Tensor<3>>,
}

/// Router + expert bank. Field names follow the HF `Olmoe` checkpoint layout
/// (`mlp.gate` is the router Linear, `mlp.experts` the bank).
#[derive(Module, Debug)]
pub struct SparseMoe {
    /// The routing projection: `hidden -> num_experts`, no bias.
    pub gate: Linear,
    pub experts: MoeExperts,
}

/// Shape config for [`SparseMoe`].
#[derive(Debug, Clone)]
pub struct SparseMoeConfig {
    pub hidden_size: usize,
    /// Per-expert SwiGLU intermediate width (OLMoE: 1024 — each expert is
    /// narrow; capacity comes from the count).
    pub expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl SparseMoeConfig {
    /// Initialize the module (random weights; real weights come from import).
    pub fn init(&self, device: &Device) -> SparseMoe {
        assert!(
            self.num_experts >= 2,
            "MoE: num_experts must be >= 2 (got {}); use SwiGluMlp for a dense FFN",
            self.num_experts
        );
        assert!(
            (1..=self.num_experts).contains(&self.num_experts_per_tok),
            "MoE: num_experts_per_tok ({}) must be in 1..=num_experts ({})",
            self.num_experts_per_tok,
            self.num_experts
        );
        assert!(
            self.hidden_size >= 1 && self.expert_intermediate_size >= 1,
            "MoE: hidden_size and expert_intermediate_size must be >= 1"
        );
        let (e, h, inter) = (
            self.num_experts,
            self.hidden_size,
            self.expert_intermediate_size,
        );
        // Linear-style uniform init, bound by each projection's fan-in.
        let init = |out: usize, inp: usize, dev: &Device| {
            let bound = 1.0 / (inp as f64).sqrt();
            Param::from_tensor(Tensor::random(
                [e, out, inp],
                Distribution::Uniform(-bound, bound),
                dev,
            ))
        };
        SparseMoe {
            gate: LinearConfig::new(h, e).with_bias(false).init(device),
            experts: MoeExperts {
                gate: init(inter, h, device),
                up: init(inter, h, device),
                down: init(h, inter, device),
            },
        }
    }
}

impl SparseMoe {
    /// `[b, t, hidden]` → `[b, t, hidden]`.
    ///
    /// Router math mirrors HF `OlmoeSparseMoeBlock`: softmax over **all**
    /// experts in f32, keep the top-`top_k` probabilities as the mixture
    /// weights (renormalized to sum 1 iff `norm_topk_prob` — OLMoE ships
    /// `false`). The f32 island matters on f16 backends (softmax of wide
    /// logits); every cast is a no-op on f32.
    pub fn forward(&self, x: Tensor<3>, top_k: usize, norm_topk_prob: bool) -> Tensor<3> {
        let [b, t, h] = x.dims();
        let [e, _inter, h_in] = self.experts.gate.dims();
        assert!(
            (1..=e).contains(&top_k),
            "MoE forward: top_k ({top_k}) must be in 1..=num_experts ({e})"
        );
        assert!(
            h == h_in,
            "MoE forward: input hidden {h} does not match expert hidden {h_in}"
        );
        debug_assert!(
            self.experts.down.dims()[1] == h,
            "MoE forward: down projection must return to the model width"
        );
        let ambient = x.dtype();
        let bt = b * t;
        let xt = x.reshape([bt, h]);

        // Router → dense per-token weight rows [bt, e]: softmax probabilities
        // where an expert is in the token's top-k, exact zero elsewhere. The
        // scatter is an on-device arange-compare — burn's `one_hot` reads the
        // indices back to the host, which would sync every layer.
        let logits = self.gate.forward(xt.clone()); // [bt, e]
        debug_assert!(logits.dims() == [bt, e], "router width must be num_experts");
        let probs = activation::softmax(logits.cast(DType::F32), 1);
        let (vals, idx) = probs.topk_with_indices(top_k, 1); // both [bt, k]
        let vals = if norm_topk_prob {
            vals.clone().div(vals.sum_dim(1)) // [bt, k] / [bt, 1]
        } else {
            vals
        };
        let classes = Tensor::<1, Int>::arange(0..e as i64, &xt.device()).reshape([1, 1, e as i32]);
        let hit = idx
            .reshape([bt, top_k, 1])
            .equal(classes.expand([bt, top_k, e])); // [bt, k, e]
        let weights = hit
            .float()
            .cast(DType::F32)
            .mul(vals.reshape([bt, top_k, 1]))
            .sum_dim(1) // [bt, 1, e]
            .reshape([bt, e])
            .cast(ambient);

        // Dense expert compute: one batched matmul per projection across the
        // whole bank ([1, bt, h] broadcast against [e, h, *]), then the weight
        // rows zero out the unrouted experts in the reduction.
        let xb = xt.reshape([1, bt, h]);
        let gate = xb.clone().matmul(self.experts.gate.val().swap_dims(1, 2)); // [e, bt, inter]
        let up = xb.matmul(self.experts.up.val().swap_dims(1, 2));
        let acts = activation::silu(gate).mul(up);
        let out = acts.matmul(self.experts.down.val().swap_dims(1, 2)); // [e, bt, h]
        let w_per_expert = weights.swap_dims(0, 1).reshape([e, bt, 1]);
        out.mul(w_per_expert)
            .sum_dim(0) // [1, bt, h]
            .reshape([b, t, h])
    }
}

/// One expert's SwiGLU weights stored **separately** in Linear layout
/// (`[in, out]`), so each expert quantizes independently (its own block
/// scales) and only routed experts are touched at all.
#[derive(Module, Debug)]
pub struct ExpertWeights {
    /// `[hidden, intermediate]` — SiLU branch.
    pub gate: Param<Tensor<2>>,
    /// `[hidden, intermediate]` — multiplicative branch.
    pub up: Param<Tensor<2>>,
    /// `[intermediate, hidden]` — back to the model width.
    pub down: Param<Tensor<2>>,
}

/// The P9 MoE variant of [`SparseMoe`]: the same router, but experts as
/// separate (typically quantized) weight triples and **routed** compute —
/// per token only its top-k experts run, so exactly `n` experts are in
/// service at a time instead of the dense-mask path's all-of-them. Routing
/// indices are read back to the host (small: `[tokens, k]` ints); the
/// dense path's no-readback rationale trades away here for the k/E FLOPs
/// and the per-expert weight independence quantization needs.
#[derive(Module, Debug)]
pub struct SparseMoePerExpert {
    /// The routing projection: `hidden -> num_experts`, no bias, float.
    pub gate: Linear,
    pub experts: Vec<ExpertWeights>,
}

impl SparseMoePerExpert {
    /// `[b, t, hidden]` → `[b, t, hidden]`. Router math identical to
    /// [`SparseMoe::forward`]; expert compute is gather → three 2-D
    /// matmuls (never reshaping the possibly-packed weights) →
    /// scatter-add of the weighted outputs.
    pub fn forward(&self, x: Tensor<3>, top_k: usize, norm_topk_prob: bool) -> Tensor<3> {
        let [b, t, h] = x.dims();
        let xt = x.reshape([b * t, h]);
        let routing = self.route(xt.clone(), top_k, norm_topk_prob);
        self.run_local(xt, &routing).reshape([b, t, h])
    }

    /// The same layer with the experts executed by an [`ExpertPool`] — each
    /// expert wherever and at whatever precision the tier plan put it (P9
    /// stage 3b). Router math stays here on `B`; `layer` indexes the pool.
    pub fn forward_pooled(
        &self,
        x: Tensor<3>,
        top_k: usize,
        norm_topk_prob: bool,
        pool: &ExpertPool,
        layer: usize,
    ) -> Tensor<3> {
        let [b, t, h] = x.dims();
        let xt = x.reshape([b * t, h]);
        let routing = self.route(xt.clone(), top_k, norm_topk_prob);
        pool.run_layer(layer, xt, &routing).reshape([b, t, h])
    }

    /// Router: softmax → top-k → (optionally renormalized) weights, read back
    /// to the host as per-expert (token rows, weights) lists.
    pub fn route(&self, xt: Tensor<2>, top_k: usize, norm_topk_prob: bool) -> Routing {
        let [bt, _h] = xt.dims();
        let e = self.experts.len();
        assert!(
            (1..=e).contains(&top_k),
            "MoE forward: top_k ({top_k}) must be in 1..=num_experts ({e})"
        );

        let logits = self.gate.forward(xt); // [bt, e]
        let probs = activation::softmax(logits.cast(DType::F32), 1);
        let (vals, idx) = probs.topk_with_indices(top_k, 1); // both [bt, k]
        let vals = if norm_topk_prob {
            vals.clone().div(vals.sum_dim(1))
        } else {
            vals
        };

        // Host routing: which tokens each expert serves, with what weight.
        let idx_host: Vec<i64> = idx
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("routing indices read back");
        let vals_host: Vec<f32> = vals
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("routing weights read back");
        let mut routed: Vec<(Vec<i32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); e];
        for token in 0..bt {
            for slot in 0..top_k {
                let expert = usize::try_from(idx_host[token * top_k + slot])
                    .expect("router indices are in 0..e");
                routed[expert].0.push(token as i32);
                routed[expert].1.push(vals_host[token * top_k + slot]);
            }
        }
        Routing {
            tokens: bt,
            per_expert: routed,
        }
    }

    /// Expert compute on this module's own (same-backend) experts: gather →
    /// three 2-D matmuls (never reshaping the possibly-packed weights) →
    /// scatter-add of the weighted outputs. `[bt, h]` → `[bt, h]`.
    pub fn run_local(&self, xt: Tensor<2>, routing: &Routing) -> Tensor<2> {
        let [bt, h] = xt.dims();
        let ambient = xt.dtype();
        let device = xt.device();
        let mut out = Tensor::<2>::zeros([bt, h], &device);
        for (expert, (rows, weights)) in routing.per_expert.iter().enumerate() {
            if rows.is_empty() {
                continue; // not in service this batch
            }
            let n = rows.len();
            let rows = rows.clone();
            let weights = weights.clone();
            let rows_t = Tensor::<1, Int>::from_data(
                burn::tensor::TensorData::new(rows, [n]),
                (&device, crate::backend::int_dtype(&device)),
            );
            let x_e = xt.clone().select(0, rows_t.clone()); // [n, h]
            let w = &self.experts[expert];
            let acts =
                activation::silu(x_e.clone().matmul(w.gate.val())).mul(x_e.matmul(w.up.val()));
            let y = acts.matmul(w.down.val()); // [n, h]
            let scale = Tensor::<1>::from_data(
                burn::tensor::TensorData::new(weights, [n]),
                (&device, crate::backend::float_dtype(&device)),
            )
            .reshape([n, 1])
            .cast(ambient);
            // select_assign accumulates (scatter-add) — a token served by
            // several experts sums their weighted outputs.
            out = out.select_assign(0, rows_t, y.mul(scale), burn::tensor::IndexingUpdateOp::Add);
        }
        out
    }
}

/// One batch's routing decision, host-side: for every expert, the token
/// rows (into the flattened `[b·t, h]` input) it serves and their weights.
#[derive(Debug, Clone)]
pub struct Routing {
    pub tokens: usize,
    pub per_expert: Vec<(Vec<i32>, Vec<f32>)>,
}

// ===========================================================================
// P9 stage 3(b): tiered expert execution. An expert lives on *some* device
// at *some* stored precision behind `ExpertExec`; the pool holds one per
// (layer, expert), swaps them at runtime, and counts routing hits for the
// tier planner (`crate::tier`). Data crosses devices through host f32 —
// per step only the routed rows (decode: one row per active expert).
// ===========================================================================

/// One expert, resident somewhere, runnable from the host.
pub trait ExpertExec: Send + Sync {
    /// Where/how this expert is resident.
    fn tier(&self) -> crate::tier::Tier;
    /// Bytes it holds on its device.
    fn resident_bytes(&self) -> u64;
    /// SwiGLU on `rows × hidden` f32 (row-major) → `rows × hidden`.
    ///
    /// The host-buffer form. Prefer [`Self::run_tensor`], which keeps the
    /// data on-device when the caller is already on this expert's device.
    fn run(&self, x: &[f32], rows: usize, hidden: usize) -> Vec<f32>;

    /// Like [`Self::run_tensor`], but the result stays wherever this expert
    /// computed it — the caller moves it later, batched with every other
    /// partial, so the enqueue returns without waiting on the device.
    ///
    /// Default: `run_tensor`, whose result is already caller-resident — the
    /// later move is then a no-op. Device-pinned executors override to skip
    /// their trailing move.
    fn run_tensor_resident(&self, x: Tensor<2>) -> Tensor<2> {
        self.run_tensor(x)
    }

    /// Stop handing this executor's packed weights to the native quantized
    /// matmul — dequantize before every multiply instead. Called after a
    /// caught kernel-gap panic; default is a no-op for executors that never
    /// take the native path.
    fn disable_native(&self) {}

    /// Bring this expert's weights onto `device` (the working set's
    /// prefetch). Default: nothing — an executor pinned to one device is
    /// already where it will run, so staging it is a no-op rather than an
    /// error.
    fn stage(&self, _device: &Device) {}

    /// Release a staged device copy (the working set's eviction). Default:
    /// nothing, for the same reason.
    fn evict(&self) {}

    /// Is a device copy currently held? Pinned executors answer `true` —
    /// they are always "resident" on their own device.
    fn is_staged(&self) -> bool {
        true
    }

    /// SwiGLU on `[rows, hidden]` **as a tensor**: burn 0.22 has one tensor
    /// type across devices, so this moves data only when the caller's device
    /// differs from the expert's — and not at all when they match, which is
    /// the difference between a per-layer host round trip and none.
    ///
    /// The default keeps the old behavior for executors that only implement
    /// the host form.
    fn run_tensor(&self, x: Tensor<2>) -> Tensor<2> {
        let [rows, hidden] = x.dims();
        let device = x.device();
        let host = x
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("expert input read back");
        let out = self.run(&host, rows, hidden);
        Tensor::<2>::from_data(
            burn::tensor::TensorData::new(out, [rows, hidden]),
            (&device, crate::backend::float_dtype(&device)),
        )
    }
    /// Per-row energy of this expert's gate activations, `Σ silu(x·g)²` —
    /// the training-free router signal for skipping (P9 stage 3c). Costs
    /// the gate matmul only. The default never skips.
    fn gate_energy(&self, _x: &[f32], rows: usize, _hidden: usize) -> Vec<f32> {
        vec![f32::INFINITY; rows]
    }
}

/// [`ExpertWeights`] on a concrete backend's device, exposed as an
/// [`ExpertExec`]. Generic over `B`, so a pool can mix CPU, wgpu and CUDA
/// experts — each at its own precision — behind one trait object.
pub struct DeviceExpert {
    pub weights: ExpertWeights,
    pub device: Device,
    pub tier: crate::tier::Tier,
    pub bytes: u64,
    /// Cleared the first time this group's native quantized matmul panics
    /// (the width-dependent cubecl kernel gap); afterwards every multiply
    /// dequantizes first. Per GROUP, because the gap is per shape.
    pub native_ok: std::sync::atomic::AtomicBool,
}

impl ExpertExec for DeviceExpert
where
    ExpertWeights: Send + Sync,
    Device: Send + Sync,
{
    fn tier(&self) -> crate::tier::Tier {
        self.tier
    }

    fn resident_bytes(&self) -> u64 {
        self.bytes
    }

    fn run(&self, x: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), rows * hidden);
        let xt = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(x.to_vec(), [rows, hidden]),
            (&self.device, crate::backend::float_dtype(&self.device)),
        );
        let w = &self.weights;
        let acts = activation::silu(xt.clone().matmul(compute_weight(&w.gate)))
            .mul(xt.matmul(compute_weight(&w.up)));
        acts.matmul(compute_weight(&w.down))
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("expert output read back")
    }

    fn run_tensor(&self, x: Tensor<2>) -> Tensor<2> {
        // `to_device` is a no-op when the tensor is already here, so a
        // cluster group living on the caller's device costs zero transfers.
        let caller = x.device();
        crate::backend::move_to(self.run_tensor_resident(x), &caller)
    }

    fn run_tensor_resident(&self, x: Tensor<2>) -> Tensor<2> {
        let xt = crate::backend::move_to(x, &self.device);
        let w = &self.weights;
        // After a caught kernel-gap panic, `native_ok` is false and packed
        // weights are dequantized up front — the path that is correct at
        // every width, at measured-identical speed for a single group.
        let native = self.native_ok.load(std::sync::atomic::Ordering::Relaxed);
        // At decode shape with a Q4S weight, the packed GEMV reads the
        // stored nibbles directly — no dequant transient, no f32 weight
        // traffic (nn/packed_gemv.rs). Its launch errors surface at the
        // caller's readback, inside the same catch that guards q_matmul,
        // so the downgrade contract is unchanged.
        let mm = |xin: &Tensor<2>, p: &burn::module::Param<Tensor<2>>| -> Tensor<2> {
            if native {
                let wv = p.val();
                if let Some(y) = crate::nn::packed_gemv::try_q4s_gemv(xin, &wv) {
                    return y;
                }
                xin.clone().matmul(compute_weight(p))
            } else {
                let t = p.val();
                let t = match t.dtype() {
                    DType::QFloat(_) => t.dequantize(),
                    _ => t,
                };
                xin.clone().matmul(t)
            }
        };
        let acts = activation::silu(mm(&xt, &w.gate)).mul(mm(&xt, &w.up));
        mm(&acts, &w.down)
    }

    fn disable_native(&self) {
        self.native_ok
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn gate_energy(&self, x: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
        let xt = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(x.to_vec(), [rows, hidden]),
            (&self.device, crate::backend::float_dtype(&self.device)),
        );
        activation::silu(xt.matmul(compute_weight(&self.weights.gate)))
            .powf_scalar(2.0)
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("gate energy read back")
    }
}

/// A pooled expert's weight, ready to multiply.
///
/// A quantized weight is handed to `matmul` **as-is** when this device's
/// backend multiplies it natively — that reads 4–8x fewer weight bytes and
/// skips materializing an f32 copy. Where the native path is broken it is
/// dequantized first instead (transient; the `Param` stays quantized).
///
/// Which backends work is a property of the burn version and the device, so
/// it is **probed**, not hardcoded — see [`native_qmatmul_ok`].
fn compute_weight(w: &Param<Tensor<2>>) -> Tensor<2> {
    let t = w.val();
    match t.dtype() {
        // wgpu: dequantize the 8-BIT family before the multiply — and only
        // that family. This is not the retracted "wgpu Q4 is broken" claim
        // of 2026-08-23 (a probe artifact); it has a production backtrace:
        // burn 0.22's wgpu q_matmul panics at cubecl-std quant/view.rs:223
        // ("quantized view float vector size 1 must be a positive multiple
        // of num_quants 4") when the kernel chosen for an m=1 decode step
        // vectorizes the float side at 1. `num_quants 4` is four values per
        // u32 — the 8-bit schemes. Q4S packs eight and has never produced
        // this panic in any log; a blanket wgpu-dequantize guard was tried
        // and traded the panic for something worse: dequantizing every Q4
        // group churned ~90-210 MB f32 transients through a pool that
        // allocates ~1 GiB chunks, and OOM-killed generations once other
        // apps held part of the card. Narrow beats broad here:
        //   Q8-family -> dequantize (small groups, tiny transients, covers
        //                the entire observed panic family);
        //   Q4S       -> native, zero transients (speed measured identical
        //                either way: 1.61 ms both, bit-identical results).
        // CUDA keeps the probed native path for everything.
        DType::QFloat(scheme)
            if is_wgpu(&t.device())
                && matches!(
                    scheme.value,
                    burn::tensor::quantization::QuantValue::Q8S
                        | burn::tensor::quantization::QuantValue::Q8F
                        | burn::tensor::quantization::QuantValue::E4M3
                        | burn::tensor::quantization::QuantValue::E5M2
                ) =>
        {
            t.dequantize()
        }
        DType::QFloat(_) if native_qmatmul_ok(&t.device(), t.dtype()) => t,
        DType::QFloat(_) => t.dequantize(),
        _ => t,
    }
}

/// Is this a wgpu device? burn 0.22 selects backends by runtime `Device`
/// value and exposes no kind accessor, so the debug form is the handle
/// available. Covers the discrete and integrated adapters alike.
fn is_wgpu(device: &Device) -> bool {
    format!("{device:?}").contains("Wgpu")
}

/// Does this device multiply a quantized weight natively — without panicking,
/// and with the right answer?
///
/// Probed once per (device, scheme) and cached. Burn 0.21's CUDA `q_matmul`
/// panicked in kernel expansion; 0.22 fixed CUDA but wgpu still panics on Q4,
/// and upstream's own autotune candidates are documented as panicking "on the
/// level rather than declining it". A capability that varies by version,
/// backend and dtype is exactly the kind that should be measured on the
/// machine in front of us rather than asserted from a table.
///
/// Conservative by construction: any panic, or an answer that disagrees with
/// the dequantized path, means "no".
fn native_qmatmul_ok(device: &Device, dtype: DType) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let key = format!("{device:?}/{dtype:?}");
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&hit) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return hit;
    }

    // Small enough to be free, wide enough to cross a quantization block.
    let (k, n) = (64usize, 64usize);
    let ok = std::panic::catch_unwind(|| {
        let x = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![0.5f32; k], [1, k]),
            (device, crate::backend::float_dtype(device)),
        );
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(
                (0..k * n)
                    .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
                    .collect::<Vec<f32>>(),
                [k, n],
            ),
            (device, crate::backend::float_dtype(device)),
        );
        let DType::QFloat(scheme) = dtype else {
            return false;
        };
        let qw = w.clone().quantize_dynamic(&scheme);
        let native = x
            .clone()
            .matmul(qw.clone())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>();
        let deq = x
            .matmul(qw.dequantize())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>();
        match (native, deq) {
            (Ok(a), Ok(b)) => {
                // Agreement, not just absence of a panic: a native path that
                // silently computes something else is worse than one that fails.
                let scale = b.iter().map(|v| v.abs()).fold(1e-3, f32::max);
                a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 0.05 * scale)
            }
            _ => false,
        }
    })
    .unwrap_or(false);

    if !ok {
        // Worth saying once: it silently costs bandwidth on every matmul.
        eprintln!(
            "[mummu] {key}: no usable native quantized matmul — dequantizing before each matmul"
        );
    }
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, ok);
    ok
}

/// An expert whose weights live in **host RAM**, staged onto a device only
/// while it computes (P9 stage 4 — see [`crate::workingset`]).
///
/// [`DeviceExpert`] pins its weights to one device for the process's life,
/// which is what caps how much of a model can ever run on the fast one.
/// This type inverts that: the host copy is authoritative, the device copy
/// is a cache entry the scheduler creates and drops. That is what lets a
/// model larger than VRAM still execute every layer on the GPU.
///
/// `resident` is the staged device copy. `None` means "not staged": `run`
/// then computes on the host, which is the overflow path — never a stall,
/// because the host already holds the bytes.
pub struct StagedExpert {
    /// The authoritative copy, always present, on the host device.
    host: ExpertWeights,
    /// The host device the weights live on (where overflow computes).
    host_device: Device,
    /// The staged device copy, when the scheduler has brought it in.
    resident: std::sync::RwLock<Option<(Device, ExpertWeights)>>,
    tier: crate::tier::Tier,
    bytes: u64,
}

impl StagedExpert {
    /// Hold `weights` in host RAM, unstaged.
    #[must_use]
    pub fn new(
        weights: ExpertWeights,
        host_device: Device,
        tier: crate::tier::Tier,
        bytes: u64,
    ) -> Self {
        Self {
            host: weights,
            host_device,
            resident: std::sync::RwLock::new(None),
            tier,
            bytes,
        }
    }

    /// Stage onto `device` (the scheduler's prefetch). Idempotent: staging
    /// onto the device it already sits on does nothing, so a redundant
    /// prefetch costs a comparison rather than a transfer.
    fn stage_on(&self, device: &Device) {
        {
            let held = self.resident.read().unwrap_or_else(|e| e.into_inner());
            if held.as_ref().is_some_and(|(d, _)| d == device) {
                return;
            }
        }
        let staged = ExpertWeights {
            gate: burn::module::Param::from_tensor(self.host.gate.val().to_device(device)),
            up: burn::module::Param::from_tensor(self.host.up.val().to_device(device)),
            down: burn::module::Param::from_tensor(self.host.down.val().to_device(device)),
        };
        *self.resident.write().unwrap_or_else(|e| e.into_inner()) = Some((device.clone(), staged));
    }

    /// Drop the device copy (the scheduler's eviction), freeing its memory.
    /// The host copy is untouched, so the expert stays runnable.
    fn evict_copy(&self) {
        *self.resident.write().unwrap_or_else(|e| e.into_inner()) = None;
    }

    fn staged(&self) -> bool {
        self.resident
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
}

impl ExpertExec for StagedExpert {
    fn tier(&self) -> crate::tier::Tier {
        self.tier
    }

    fn resident_bytes(&self) -> u64 {
        // Device bytes only: the host copy is the backing store, not part of
        // the working set the planner budgets.
        if self.staged() { self.bytes } else { 0 }
    }

    fn stage(&self, device: &Device) {
        self.stage_on(device);
    }

    fn evict(&self) {
        self.evict_copy();
    }

    fn is_staged(&self) -> bool {
        self.staged()
    }

    fn run(&self, x: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
        let device = self
            .resident
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map_or_else(|| self.host_device.clone(), |(d, _)| d.clone());
        let xt = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(x.to_vec(), [rows, hidden]),
            (&device, crate::backend::float_dtype(&device)),
        );
        self.run_tensor(xt)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("expert output read back")
    }

    fn run_tensor(&self, x: Tensor<2>) -> Tensor<2> {
        let caller = x.device();
        let held = self.resident.read().unwrap_or_else(|e| e.into_inner());
        let (device, w) = match held.as_ref() {
            // Staged: compute on the device the scheduler put it on.
            Some((d, w)) => (d.clone(), w),
            // Not staged — the overflow path. Compute on the host rather
            // than stall waiting for a transfer that was never issued.
            None => (self.host_device.clone(), &self.host),
        };
        let xt = crate::backend::move_to(x, &device);
        let acts = activation::silu(xt.clone().matmul(compute_weight(&w.gate)))
            .mul(xt.matmul(compute_weight(&w.up)));
        crate::backend::move_to(acts.matmul(compute_weight(&w.down)), &caller)
    }
}

/// The tiered expert bank of one model: `[layer][expert]` executors,
/// hot-swappable, with routing hit counters. Shared (`Arc`) between the
/// model that runs it and the planner that re-tiers it.
pub struct ExpertPool {
    slots: Vec<Vec<std::sync::RwLock<std::sync::Arc<dyn ExpertExec>>>>,
    hits: Vec<Vec<std::sync::atomic::AtomicU64>>,
    /// Output energy per executor (milli-units), for calibration hotness.
    energy: Vec<Vec<std::sync::atomic::AtomicU64>>,
    /// Dense-path rows offered / computed (skip accounting).
    dense_rows: [std::sync::atomic::AtomicU64; 2],
}

impl ExpertPool {
    /// Build from `[layer][expert]` executors (every layer the same width).
    #[must_use]
    pub fn new(slots: Vec<Vec<std::sync::Arc<dyn ExpertExec>>>) -> Self {
        let counters = || -> Vec<Vec<std::sync::atomic::AtomicU64>> {
            slots
                .iter()
                .map(|l| {
                    (0..l.len())
                        .map(|_| std::sync::atomic::AtomicU64::new(0))
                        .collect()
                })
                .collect()
        };
        let hits = counters();
        let energy = counters();
        Self {
            slots: slots
                .into_iter()
                .map(|l| l.into_iter().map(std::sync::RwLock::new).collect())
                .collect(),
            hits,
            energy,
            dense_rows: [
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
            ],
        }
    }

    /// Output energy accumulated per executor since the last call (flat,
    /// layer-major, ragged rows concatenated); resets.
    pub fn take_energy(&self) -> Vec<f64> {
        self.energy
            .iter()
            .flat_map(|l| {
                l.iter()
                    .map(|e| e.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e3)
            })
            .collect()
    }

    /// Dense-path skip accounting since the last call: (rows computed, rows
    /// offered) summed over executors; resets.
    pub fn take_dense_rows(&self) -> (u64, u64) {
        (
            self.dense_rows[0].swap(0, std::sync::atomic::Ordering::Relaxed),
            self.dense_rows[1].swap(0, std::sync::atomic::Ordering::Relaxed),
        )
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn experts_per_layer(&self) -> usize {
        self.slots.first().map_or(0, Vec::len)
    }

    /// Flat index of `(layer, expert)` — the tier planner's expert order.
    #[must_use]
    pub fn flat(&self, layer: usize, expert: usize) -> usize {
        layer * self.experts_per_layer() + expert
    }

    pub fn get(&self, layer: usize, expert: usize) -> std::sync::Arc<dyn ExpertExec> {
        self.slots[layer][expert]
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Replace one expert's executor (the hot-swap). The old one is
    /// returned so the caller controls when its device memory is freed.
    /// Apply one layer's working-set decisions: evict what the schedule
    /// gave up, stage what it prefetched. Both are cheap in-place calls on
    /// the executors — no slot swap, because a [`StagedExpert`] owns its
    /// host copy and its device copy at once.
    ///
    /// Called for layer `L` *while layer `L` computes*, so the transfers
    /// overlap compute rather than sitting on the critical path. Nothing
    /// here blocks: a stage that has not landed by the time its layer runs
    /// simply computes on the host (see [`StagedExpert::run_tensor`]).
    pub fn apply_schedule(
        &self,
        layer: usize,
        sched: &crate::workingset::LayerSchedule,
        device: &Device,
    ) {
        for &u in &sched.evict {
            if let Some(e) = self.unit(layer, u) {
                e.evict();
            }
        }
        for &u in &sched.prefetch {
            if let Some(e) = self.unit(layer, u) {
                e.stage(device);
            }
        }
    }

    /// The executor for a flat unit id, if this pool holds it. Unit ids are
    /// layer-major (`layer * experts_per_layer + index`), matching the
    /// scheduler's numbering.
    #[must_use]
    pub fn unit(
        &self,
        layer: usize,
        unit: crate::workingset::UnitId,
    ) -> Option<std::sync::Arc<dyn ExpertExec>> {
        let per = self.experts_per_layer();
        let (l, i) = unit
            .checked_div(per)
            .map_or((layer, unit), |l| (l, unit % per));
        // A unit id addresses its own layer; fall back to the caller's layer
        // for pools whose rows are ragged (dense FFN groups).
        let (l, i) = if self.slots.get(l).is_some_and(|r| i < r.len()) {
            (l, i)
        } else if self.slots.get(layer).is_some_and(|r| unit < r.len()) {
            (layer, unit)
        } else {
            return None;
        };
        Some(
            self.slots[l][i]
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
        )
    }

    /// Device bytes the working set currently holds — what the scheduler
    /// budgets against, counting only staged copies.
    #[must_use]
    pub fn staged_bytes(&self) -> u64 {
        (0..self.num_layers())
            .flat_map(|l| (0..self.row_len(l)).map(move |e| (l, e)))
            .map(|(l, e)| {
                let x = self.get(l, e);
                if x.is_staged() { x.resident_bytes() } else { 0 }
            })
            .sum()
    }

    pub fn swap(
        &self,
        layer: usize,
        expert: usize,
        next: std::sync::Arc<dyn ExpertExec>,
    ) -> std::sync::Arc<dyn ExpertExec> {
        let mut slot = self.slots[layer][expert]
            .write()
            .unwrap_or_else(|e| e.into_inner());
        std::mem::replace(&mut *slot, next)
    }

    /// Experts in `layer` (rows may be ragged — a dense model's remote
    /// clusters differ per layer).
    #[must_use]
    pub fn row_len(&self, layer: usize) -> usize {
        self.slots.get(layer).map_or(0, Vec::len)
    }

    /// Every expert's tier, flat layer-major (ragged rows concatenated).
    #[must_use]
    pub fn tiers(&self) -> Vec<crate::tier::Tier> {
        (0..self.num_layers())
            .flat_map(|l| (0..self.row_len(l)).map(move |e| (l, e)))
            .map(|(l, e)| self.get(l, e).tier())
            .collect()
    }

    /// Resident bytes per device index.
    #[must_use]
    pub fn used_bytes(&self, num_devices: usize) -> Vec<u64> {
        let mut used = vec![0u64; num_devices];
        for l in 0..self.num_layers() {
            for e in 0..self.row_len(l) {
                let x = self.get(l, e);
                if let Some(u) = used.get_mut(x.tier().device) {
                    *u += x.resident_bytes();
                }
            }
        }
        used
    }
}

/// Wall-clock in µs since first use, for the layer timeline trace.
pub fn trace_us() -> u128 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_micros()
}

/// Which layer (if any) the timeline trace follows (`MUMMU_TRACE_LAYER`).
pub fn trace_layer() -> Option<usize> {
    static ON: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MUMMU_TRACE_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// Raise the calling worker thread above the trunk's gemm pool. The pool
/// saturates every core through the local slab, and a default-priority
/// worker measurably could not win a core even to SUBMIT its GPU work
/// until the caller reached the join — zero overlap, the full device time
/// exposed (merge.join 27.1 ms/layer with enqueue-first ordering in
/// place). The worker needs microseconds of CPU to submit, then blocks on
/// the fence; above-normal priority preempts one pool thread for exactly
/// that sliver.
fn boost_worker_priority() {
    #[cfg(windows)]
    {
        #[link(name = "kernel32.dll", kind = "raw-dylib", modifiers = "+verbatim")]
        unsafe extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(handle: isize, priority: i32) -> i32;
        }
        // SAFETY: plain kernel32 calls on the current thread's pseudo
        // handle; 1 = THREAD_PRIORITY_ABOVE_NORMAL.
        unsafe {
            SetThreadPriority(GetCurrentThread(), 1);
        }
    }
}

/// Run one dense executor resident and read its result back as plain
/// bytes, with the same catch-once-downgrade-retry contract as
/// [`run_with_native_fallback`]. The catch MUST span the readback: on the
/// wgpu path the native quantized matmul only enqueues at the op call —
/// kernel expansion happens at the blocking read, where the device server
/// re-raises its panic on the reading thread — so a catch around the
/// compute alone can never see the width-dependent kernel-gap panic
/// (`num_quants 4`/`8`) this fallback exists for. Any second panic — a
/// genuine failure, OOM included — propagates loudly as ever.
fn run_readback_with_fallback(exec: &std::sync::Arc<dyn ExpertExec>, xt: &Tensor<2>) -> TensorData {
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        exec.run_tensor_resident(xt.clone()).into_data()
    }));
    match attempt {
        Ok(data) => data,
        Err(_) => {
            eprintln!(
                "[mummu] native quantized matmul panicked for one expert group \
                 (cubecl kernel gap; width-dependent) — group switched to \
                 dequantize-first and retried"
            );
            exec.disable_native();
            exec.run_tensor_resident(xt.clone()).into_data()
        }
    }
}

/// Run one dense executor with the adaptive native fallback: a panic from
/// the width-dependent cubecl kernel gap (quant/view vector-size assert,
/// `num_quants 4`/`8`) is caught ONCE, the group is switched to
/// dequantize-first, and the same input is retried. Any second panic — a
/// genuine failure, OOM included — propagates loudly as ever.
///
/// The catch is sound here for the same reason `native_qmatmul_ok` could
/// probe this panic: it fires at kernel expand on the calling thread, before
/// submission, and the device server measurably survives it.
fn run_with_native_fallback(exec: &std::sync::Arc<dyn ExpertExec>, xt: &Tensor<2>) -> Tensor<2> {
    let attempt =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exec.run_tensor(xt.clone())));
    match attempt {
        Ok(y) => y,
        Err(_) => {
            eprintln!(
                "[mummu] native quantized matmul panicked for one expert group \
                 (cubecl kernel gap; width-dependent) — group switched to \
                 dequantize-first and retried"
            );
            exec.disable_native();
            exec.run_tensor(xt.clone())
        }
    }
}

/// One layer's remote FFN, in flight: each device's worker thread is
/// computing AND draining its own device — concurrently, exactly as the old
/// synchronous path did — and only the JOIN is deferred, so the caller's
/// local slab runs while the devices chew.
///
/// This is the third design. The first deferred the device sync itself
/// into [`Self::resolve`], and adversarial review proved against the
/// burn-fusion sources that nothing executes at enqueue — the worker
/// streams' queued IR first ran inside resolve's drain, and resolving
/// partials one at a time serialized the devices where the old workers
/// drained them in parallel (wall `local + T_a + T_b` instead of
/// `local + max(T_a, T_b)`). The second deferred only the join, with each
/// worker building the caller-device result tensor from its readback — and
/// every layer still paid ~27 ms SOMEWHERE, migrating between scopes as
/// the code moved (worker build, join, a main-thread touch), because
/// wgpu's `into_data` returns deferred-mapped bytes: it comes back in
/// low ms while the FIRST CPU TOUCH of the bytes blocks on the GPU fence
/// (`examples/mapped-wait-probe.rs` reproduces it standalone: into_data
/// 1-4 ms, first touch 27.0-27.5 ms behind a queued GPU chain). The cost
/// was never queues, scheduling, or allocation — it is the remote FFN's
/// real GPU time surfacing at first byte access. So the third design
/// makes the WORKER touch the bytes (`to_vec` in the accumulate): the
/// fence wait lands here, concurrent with whatever the caller still has
/// to do. The layer timeline (MUMMU_TRACE_LAYER) then showed how little
/// that is: remote-heavy layers keep ~1 local cluster, so the caller
/// reaches the join ~1 ms after the enqueue and ~26 ms of GPU time is
/// exposed with nothing to overlap against. Measured per cluster at m=1:
/// dGPU 0.91 ms vs CPU 0.40 ms — the GPU is 2.3x SLOWER than the host it
/// is meant to relieve, because the quantized matmul dequantizes to f32.
/// Shrinking that is a kernel problem (packed m=1 GEMV), not a
/// scheduling one; every scheduling fix here is already in place.
pub struct PendingRemote {
    handles: Vec<std::thread::JoinHandle<Option<TensorData>>>,
    /// The caller's device, captured at enqueue: partials come home to
    /// wherever the trunk lives (wgpu-trunk placements exist), never to a
    /// hardcoded CPU device.
    home: Device,
}

impl PendingRemote {
    /// Join the workers, then build and sum their partials on the CALLER's
    /// thread, on the caller's own device, with each readback's own dtype.
    /// The workers hand back plain FULLY-MATERIALIZED bytes: wgpu readback
    /// bytes are deferred-mapped, and their first CPU touch blocks on the
    /// GPU fence (~27 ms measured; see `mapped-wait-probe`), so the worker
    /// touches them in its accumulate — concurrent with the local slab —
    /// and everything on this thread is microseconds.
    #[must_use]
    pub fn resolve(self) -> Option<Tensor<2>> {
        let mut out: Option<Tensor<2>> = None;
        for handle in self.handles {
            let partial = {
                let _s = crate::prof::scope("merge.join");
                match handle.join() {
                    Ok(p) => p,
                    // Same rule as everywhere in this pool: a dead worker is
                    // a dead generation, never a silently smaller sum.
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            };
            if let Some(data) = partial {
                let _s = crate::prof::scope("merge.sum");
                let dtype = data.dtype;
                let p = Tensor::from_data(data, (&self.home, dtype));
                out = Some(match out {
                    Some(acc) => acc.add(p),
                    None => p,
                });
            }
        }
        out
    }
}

impl ExpertPool {
    /// The exact dense remote FFN with a deferred join: spawn one worker per
    /// device (the bounded-stream rule), each running its members and moving
    /// its accumulated partial to the caller — the drain — on its own
    /// thread, then return without joining. Skip-mode (tau > 0) stays on
    /// [`Self::run_dense`]: it needs host energies up front.
    ///
    /// Plain `std::thread::spawn`, not a scope: the whole point is that the
    /// threads outlive this call. Everything moved in is `Arc`s and owned
    /// tensors. ~2 spawns per layer is microseconds against multi-ms drains;
    /// a persistent pool is the upgrade if a profile ever says otherwise.
    pub fn run_dense_pending(&self, layer: usize, xt: Tensor<2>) -> Option<PendingRemote> {
        let n = self.row_len(layer);
        if n == 0 {
            return None;
        }
        let [bt, _h] = xt.dims();
        let execs: Vec<std::sync::Arc<dyn ExpertExec>> =
            (0..n).map(|e| self.get(layer, e)).collect();

        let mut by_device: Vec<(String, Vec<std::sync::Arc<dyn ExpertExec>>)> = Vec::new();
        for exec in &execs {
            let key = format!("{:?}", exec.tier().device);
            match by_device.iter_mut().find(|(k, _)| *k == key) {
                Some((_, list)) => list.push(exec.clone()),
                None => by_device.push((key, vec![exec.clone()])),
            }
        }
        for (e, _) in execs.iter().enumerate() {
            self.hits[layer][e].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
        }
        self.dense_rows[0].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
        self.dense_rows[1].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);

        let handles: Vec<std::thread::JoinHandle<Option<TensorData>>> = by_device
            .into_iter()
            .map(|(key, members)| {
                let xt = xt.clone();
                let traced = trace_layer() == Some(layer);
                std::thread::spawn(move || {
                    boost_worker_priority();
                    if traced {
                        eprintln!("[tl] worker-run {}", trace_us());
                    }
                    let _w = crate::prof::scope("ffn_worker");
                    let _d = crate::prof::scope(key);
                    // Bytes only on this thread: per executor, one
                    // compute-and-readback (the catch spans both — the
                    // kernel-gap panic surfaces at the READ, see
                    // run_readback_with_fallback) and a plain-f32
                    // accumulate that needs no backend at all. The
                    // accumulate's `to_vec` is deliberate: wgpu readback
                    // bytes are deferred-mapped and their first CPU touch
                    // blocks on the GPU fence (~27 ms measured), so THIS
                    // thread absorbs that wait concurrent with the
                    // caller's local slab instead of leaking it into the
                    // merge. The caller rebuilds the tensor in resolve().
                    let mut acc: Option<(Vec<f32>, burn::tensor::Shape)> = None;
                    for exec in &members {
                        let data = {
                            let _s = crate::prof::scope("compute");
                            run_readback_with_fallback(exec, &xt)
                        };
                        if traced {
                            eprintln!("[tl] readback-done {}", trace_us());
                        }
                        let shape = data.shape.clone();
                        let vals = data
                            .convert::<f32>()
                            .to_vec::<f32>()
                            .expect("remote FFN partial reads back as f32");
                        match &mut acc {
                            Some((a, s)) => {
                                debug_assert_eq!(*s, shape);
                                for (d, v) in a.iter_mut().zip(&vals) {
                                    *d += v;
                                }
                            }
                            None => acc = Some((vals, shape)),
                        }
                    }
                    if traced {
                        eprintln!("[tl] touch-done {}", trace_us());
                    }
                    acc.map(|(vals, shape)| TensorData::new(vals, shape))
                })
            })
            .collect();
        Some(PendingRemote {
            handles,
            home: xt.device(),
        })
    }
}

impl ExpertPool {
    /// Dense-model FFN path (P9 stage 3c): run **every** executor of
    /// `layer` on every row and sum — the remote clusters' share of a
    /// partitioned SwiGLU (the local slab runs on the model's own device).
    /// `None` when the layer has no remote clusters. With `skip`
    /// (`Some((tau, local_energy))`), a cluster is skipped for a row when its
    /// gate energy is below `tau` × the row's total energy (local + all
    /// remote) — the opt-in lossy mode; hit counters accumulate energy so
    /// the re-tier planner sees hot clusters.
    pub fn run_dense(
        &self,
        layer: usize,
        xt: Tensor<2>,
        skip: Option<(f32, &[f32])>,
    ) -> Option<Tensor<2>> {
        let n = self.row_len(layer);
        if n == 0 {
            return None;
        }
        let [bt, h] = xt.dims();
        let device = xt.device();
        let execs: Vec<std::sync::Arc<dyn ExpertExec>> =
            (0..n).map(|e| self.get(layer, e)).collect();

        // Exact mode (no skipping): every cluster runs on every row, so there
        // is nothing to gather — sum the groups as tensors and never touch
        // the host. This is the path a dense model takes, and it removes the
        // per-layer round trip that dominated the 27B's decode.
        if skip.is_none() {
            // Run the devices CONCURRENTLY, one thread per device, and sum
            // what comes back. Sequentially the layer costs the SUM of every
            // device's share, so adding a second GPU bought nothing: 885
            // clusters moved from the CPU to the integrated GPU and decode
            // measured 4.72 s/tok against 4.32 before, because an iGPU
            // cluster (14.15 ms) is no faster than the CPU cluster it
            // replaced (13.82 ms) and the move added a transfer. Run in
            // parallel the layer costs the MAX instead, which is the entire
            // reason to spread work across devices at all.
            //
            // One thread per DEVICE, never per executor. Thread-per-executor
            // is what made cubecl-cuda open a stream per thread until a
            // 64-layer forward exhausted VRAM (`CUDA_ERROR_OUT_OF_MEMORY`,
            // "Can create a new stream"). Devices are bounded — three on this
            // box — so the stream count is bounded with them.
            let mut by_device: Vec<(String, Vec<usize>)> = Vec::new();
            for (e, exec) in execs.iter().enumerate() {
                let key = format!("{:?}", exec.tier().device);
                match by_device.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, list)) => list.push(e),
                    None => by_device.push((key, vec![e])),
                }
            }
            for (e, _) in execs.iter().enumerate() {
                self.hits[layer][e].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
            }
            self.dense_rows[0].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
            self.dense_rows[1].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);

            // One device: no threads, no join, exactly the old path.
            if by_device.len() < 2 {
                let mut out: Option<Tensor<2>> = None;
                for exec in &execs {
                    let y = run_with_native_fallback(exec, &xt);
                    out = Some(match out {
                        Some(acc) => acc.add(y),
                        None => y,
                    });
                }
                return out;
            }

            let partials: Vec<Tensor<2>> = std::thread::scope(|scope| {
                let handles: Vec<_> = by_device
                    .iter()
                    .map(|(key, members)| {
                        let execs = &execs;
                        let xt = xt.clone();
                        scope.spawn(move || {
                            // A worker thread's stack starts empty, so this
                            // is a new flame-graph ROOT beside the forward's
                            // — read the widths as parallel wall time.
                            let _w = crate::prof::scope("ffn_worker");
                            let _d = crate::prof::scope(key.clone());
                            let mut acc: Option<Tensor<2>> = None;
                            for &e in members {
                                let y = execs[e].run_tensor(xt.clone());
                                acc = Some(match acc {
                                    Some(a) => a.add(y),
                                    None => y,
                                });
                            }
                            acc
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|h| match h.join() {
                        Ok(partial) => partial,
                        // A worker panic must never become a silently missing
                        // partial: dropping one device's clusters from the
                        // FFN sum produces a WRONG answer that still reads
                        // fluently — observed in production when the iGPU's
                        // workers OOM'd and `.ok()` discarded their share.
                        // Re-raise on the caller so the generation fails
                        // loudly instead of lying.
                        Err(payload) => std::panic::resume_unwind(payload),
                    })
                    .collect()
            });

            // Sum the per-device partials on the caller's device.
            let mut out: Option<Tensor<2>> = None;
            for p in partials {
                let p = crate::backend::move_to(p, &device);
                out = Some(match out {
                    Some(acc) => acc.add(p),
                    None => p,
                });
            }
            return out;
        }

        let host: Vec<f32> = xt
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("FFN input read back");
        // Which rows each executor runs: all, or the rows where it matters.
        let rows_per_exec: Vec<Vec<i32>> = match skip {
            None => vec![(0..bt as i32).collect(); n],
            Some((tau, local_energy)) => {
                // Sequential, not one thread per executor: a fresh OS thread
                // makes cubecl-cuda open a new CUDA stream, and a 64-layer
                // forward would exhaust VRAM creating them (CUDA_ERROR_OUT_OF_MEMORY
                // "Can create a new stream"). The calling thread already owns a
                // stream; reuse it.
                let energies: Vec<Vec<f32>> = execs
                    .iter()
                    .map(|exec| exec.gate_energy(&host, bt, h))
                    .collect();
                let mut total: Vec<f32> = local_energy.to_vec();
                total.resize(bt, 0.0);
                for e in &energies {
                    for (t, &v) in total.iter_mut().zip(e) {
                        *t += v;
                    }
                }
                energies
                    .iter()
                    .map(|e| {
                        (0..bt)
                            .filter(|&r| e[r] >= tau * total[r])
                            .map(|r| r as i32)
                            .collect()
                    })
                    .collect()
            }
        };
        // Sequential per executor (see the energy path above): threads here
        // would each open a CUDA stream and OOM the device over 64 layers.
        let outputs: Vec<Vec<f32>> = execs
            .iter()
            .zip(&rows_per_exec)
            .map(|(exec, rows)| {
                if rows.is_empty() {
                    return Vec::new();
                }
                let mut x = Vec::with_capacity(rows.len() * h);
                for &r in rows {
                    let r = r as usize;
                    x.extend_from_slice(&host[r * h..(r + 1) * h]);
                }
                exec.run(&x, rows.len(), h)
            })
            .collect();
        let mut out = vec![0f32; bt * h];
        for ((e, rows), y) in rows_per_exec.iter().enumerate().zip(&outputs) {
            let mut energy = 0f32;
            for (i, &r) in rows.iter().enumerate() {
                let dst = &mut out[r as usize * h..(r as usize + 1) * h];
                for (d, &v) in dst.iter_mut().zip(&y[i * h..(i + 1) * h]) {
                    *d += v;
                    energy += v * v;
                }
            }
            // Energy-weighted "hits": what the planner treats as hotness.
            self.hits[layer][e].fetch_add(
                (energy * 1e3) as u64 + rows.len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.energy[layer][e]
                .fetch_add((energy * 1e3) as u64, std::sync::atomic::Ordering::Relaxed);
            self.dense_rows[0].fetch_add(rows.len() as u64, std::sync::atomic::Ordering::Relaxed);
            self.dense_rows[1].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
        }
        Some(Tensor::<2>::from_data(
            burn::tensor::TensorData::new(out, [bt, h]),
            (&device, crate::backend::float_dtype(&device)),
        ))
    }

    /// Routing hits since the last call (token-rows served per expert),
    /// flat layer-major; resets the counters.
    pub fn take_hits(&self) -> Vec<u64> {
        self.hits
            .iter()
            .flat_map(|l| {
                l.iter()
                    .map(|h| h.swap(0, std::sync::atomic::Ordering::Relaxed))
            })
            .collect()
    }

    /// Run one layer's routed experts: gather the routed rows on the host,
    /// execute every in-service expert concurrently on its own device,
    /// scatter-add the weighted outputs, upload once. `[bt, h]` → `[bt, h]`.
    pub fn run_layer(&self, layer: usize, xt: Tensor<2>, routing: &Routing) -> Tensor<2> {
        let [bt, h] = xt.dims();
        let device = xt.device();
        let host: Vec<f32> = xt
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("MoE input read back");
        type Routed<'a> = (usize, &'a (Vec<i32>, Vec<f32>));
        let active: Vec<Routed<'_>> = routing
            .per_expert
            .iter()
            .enumerate()
            .filter(|(_, (rows, _))| !rows.is_empty())
            .collect();
        let execs: Vec<std::sync::Arc<dyn ExpertExec>> =
            active.iter().map(|(e, _)| self.get(layer, *e)).collect();
        for (e, (rows, _)) in &active {
            self.hits[layer][*e].fetch_add(rows.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let outputs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = active
                .iter()
                .zip(&execs)
                .map(|((_, (rows, _)), exec)| {
                    let host = &host;
                    s.spawn(move || {
                        let mut x = Vec::with_capacity(rows.len() * h);
                        for &r in rows {
                            let r = r as usize;
                            x.extend_from_slice(&host[r * h..(r + 1) * h]);
                        }
                        exec.run(&x, rows.len(), h)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|hd| hd.join().expect("expert thread"))
                .collect()
        });
        let mut out = vec![0f32; bt * h];
        for ((_, (rows, weights)), y) in active.iter().zip(&outputs) {
            for (i, &r) in rows.iter().enumerate() {
                let w = weights[i];
                let dst = &mut out[r as usize * h..(r as usize + 1) * h];
                for (d, &v) in dst.iter_mut().zip(&y[i * h..(i + 1) * h]) {
                    *d += w * v;
                }
            }
        }
        Tensor::<2>::from_data(
            burn::tensor::TensorData::new(out, [bt, h]),
            (&device, crate::backend::float_dtype(&device)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type Dev = burn::tensor::Device;

    const HIDDEN: usize = 4;
    const INTER: usize = 3;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;

    /// Deterministic weights: expert `e`'s matrices are small distinct
    /// sinusoids so every expert computes something different.
    fn moe(device: &Dev) -> SparseMoe {
        let fill = |seed: f32, dims: [usize; 3]| {
            let n = dims[0] * dims[1] * dims[2];
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
                .collect();
            Param::from_tensor(Tensor::<3>::from_data(TensorData::new(data, dims), device))
        };
        let router_data: Vec<f32> = (0..HIDDEN * EXPERTS)
            .map(|i| ((i as f32) * 0.61 + 1.0).cos() * 0.5)
            .collect();
        let mut m = SparseMoeConfig {
            hidden_size: HIDDEN,
            expert_intermediate_size: INTER,
            num_experts: EXPERTS,
            num_experts_per_tok: TOP_K,
        }
        .init(device);
        // Burn Linear stores weight as [in, out].
        m.gate.weight = Param::from_tensor(Tensor::<2>::from_data(
            TensorData::new(router_data, [HIDDEN, EXPERTS]),
            device,
        ));
        m.experts.gate = fill(0.1, [EXPERTS, INTER, HIDDEN]);
        m.experts.up = fill(1.7, [EXPERTS, INTER, HIDDEN]);
        m.experts.down = fill(3.3, [EXPERTS, HIDDEN, INTER]);
        m
    }

    /// The per-expert routed path must equal the dense-mask path exactly
    /// (same math, different data movement) — the P9 MoE gate.
    #[test]
    fn per_expert_routed_matches_dense_mask() {
        let device = crate::backend::cpu_device();
        let dense = moe(&device);
        // Split the fused banks into per-expert Linear-layout triples.
        let experts: Vec<ExpertWeights> = (0..EXPERTS)
            .map(|e| {
                let slice = |bank: &Param<Tensor<3>>| -> Tensor<2> {
                    let [_, out, inp] = bank.val().dims();
                    bank.val()
                        .narrow(0, e, 1)
                        .reshape([out, inp])
                        .swap_dims(0, 1) // [in, out] Linear layout
                };
                ExpertWeights {
                    gate: Param::from_tensor(slice(&dense.experts.gate)),
                    up: Param::from_tensor(slice(&dense.experts.up)),
                    down: Param::from_tensor(slice(&dense.experts.down)),
                }
            })
            .collect();
        let per_expert = SparseMoePerExpert {
            gate: dense.gate.clone(),
            experts,
        };

        for (t, seed, norm) in [(1usize, 2.0f32, false), (5, 7.0, false), (5, 7.0, true)] {
            let x = input(t, seed, &device);
            let a = dense
                .forward(x.clone(), TOP_K, norm)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let b = per_expert
                .forward(x, TOP_K, norm)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for (i, (da, db)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (da - db).abs() < 1e-5,
                    "t={t} norm={norm} elem {i}: dense {da} vs routed {db}"
                );
            }
        }
    }

    /// The working set end to end: a scheduler plan drives real staging and
    /// eviction in the pool, the device budget is respected, and the layer
    /// still computes the right answer whether its experts were staged or
    /// overflowed to the host.
    #[test]
    fn a_schedule_drives_staging_and_the_answer_is_unchanged() {
        use crate::tier::{Precision, Tier};
        use crate::workingset::{Budget, LayerDemand, schedule};
        use std::sync::Arc;
        let device = crate::backend::cpu_device();
        let dense = moe(&device);
        let split = |e: usize| -> ExpertWeights {
            let slice = |bank: &Param<Tensor<3>>| -> Tensor<2> {
                let [_, out, inp] = bank.val().dims();
                bank.val()
                    .narrow(0, e, 1)
                    .reshape([out, inp])
                    .swap_dims(0, 1)
            };
            ExpertWeights {
                gate: Param::from_tensor(slice(&dense.experts.gate)),
                up: Param::from_tensor(slice(&dense.experts.up)),
                down: Param::from_tensor(slice(&dense.experts.down)),
            }
        };
        let tier = Tier {
            device: 0,
            precision: Precision::F32,
        };
        const UNIT_BYTES: u64 = 1_000;

        // One layer holding every expert, each staged-capable.
        let row: Vec<Arc<dyn ExpertExec>> = (0..EXPERTS)
            .map(|e| {
                Arc::new(StagedExpert::new(
                    split(e),
                    device.clone(),
                    tier,
                    UNIT_BYTES,
                )) as Arc<dyn ExpertExec>
            })
            .collect();
        let pool = ExpertPool::new(vec![row]);

        // Nothing staged yet: the working set costs no device memory.
        assert_eq!(
            pool.staged_bytes(),
            0,
            "an unstaged pool holds no device bytes"
        );

        // A schedule over two passes of this layer, with room for half the
        // experts — so it must both stage and evict.
        let demands: Vec<LayerDemand> = (0..2)
            .map(|l| LayerDemand {
                layer: l,
                units: (0..EXPERTS).collect(),
            })
            .collect();
        let budget = Budget {
            device_bytes: UNIT_BYTES * (EXPERTS as u64 / 2),
            unit_bytes: UNIT_BYTES,
            stage_bytes_per_sec: (UNIT_BYTES as f64) * 2.0 / 0.010,
            layer_compute_secs: 0.010,
        };
        let plan = schedule(&demands, &budget);

        // Apply the first layer's decisions, as the runtime would.
        pool.apply_schedule(0, &plan.layers[0], &device);
        let staged = pool.staged_bytes();
        assert!(
            staged <= budget.device_bytes,
            "the working set must respect the device budget: {staged} > {}",
            budget.device_bytes
        );
        assert!(staged > 0, "a prefetching schedule must stage something");

        // The layer's answer must match the unpooled reference regardless of
        // which experts happened to be staged.
        let local = SparseMoePerExpert {
            gate: dense.gate.clone(),
            experts: (0..EXPERTS).map(split).collect(),
        };
        let x = input(3, 5.0, &device);
        let want = local
            .forward(x.clone(), TOP_K, true)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let got = local
            .forward_pooled(x, TOP_K, true, &pool, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "elem {i}: {a} vs {b} (staging changed the answer)"
            );
        }
    }

    /// P9 stage 4: staging must move WHERE an expert computes without
    /// changing WHAT it computes. Same weights, same input, same answer —
    /// staged, evicted, and re-staged.
    #[test]
    fn staging_and_eviction_never_change_the_result() {
        use crate::tier::{Precision, Tier};
        let device = crate::backend::cpu_device();
        let dense = moe(&device);
        let split = |e: usize| -> ExpertWeights {
            let slice = |bank: &Param<Tensor<3>>| -> Tensor<2> {
                let [_, out, inp] = bank.val().dims();
                bank.val()
                    .narrow(0, e, 1)
                    .reshape([out, inp])
                    .swap_dims(0, 1)
            };
            ExpertWeights {
                gate: Param::from_tensor(slice(&dense.experts.gate)),
                up: Param::from_tensor(slice(&dense.experts.up)),
                down: Param::from_tensor(slice(&dense.experts.down)),
            }
        };
        let tier = Tier {
            device: 0,
            precision: Precision::F32,
        };
        let staged = StagedExpert::new(split(0), device.clone(), tier, 0);
        // A pinned reference on the same device, for comparison.
        let pinned = DeviceExpert {
            weights: split(0),
            device: device.clone(),
            tier,
            bytes: 0,
            native_ok: std::sync::atomic::AtomicBool::new(true),
        };

        let x = Tensor::<2>::from_data(
            TensorData::new(
                (0..HIDDEN)
                    .map(|i| (i as f32) * 0.25 - 0.5)
                    .collect::<Vec<f32>>(),
                [1, HIDDEN],
            ),
            (&device, crate::backend::float_dtype(&device)),
        );
        let want = pinned
            .run_tensor(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Unstaged (the overflow path: computes on the host).
        assert!(
            !staged.is_staged(),
            "a fresh StagedExpert holds no device copy"
        );
        assert_eq!(
            staged.resident_bytes(),
            0,
            "unstaged costs no device memory"
        );
        let overflow = staged
            .run_tensor(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Staged, then evicted, then staged again.
        staged.stage(&device);
        assert!(staged.is_staged());
        let hot = staged
            .run_tensor(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        staged.stage(&device); // idempotent: no second transfer, same answer
        let again = staged
            .run_tensor(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        staged.evict();
        assert!(!staged.is_staged(), "eviction drops the device copy");
        let after_evict = staged.run_tensor(x).into_data().to_vec::<f32>().unwrap();

        for (i, w) in want.iter().enumerate() {
            for (label, got) in [
                ("overflow", &overflow),
                ("staged", &hot),
                ("re-staged", &again),
                ("after evict", &after_evict),
            ] {
                assert!(
                    (w - got[i]).abs() < 1e-5,
                    "{label} elem {i}: {} vs pinned {w}",
                    got[i]
                );
            }
        }
    }

    /// P9 stage 3b: the pooled path (experts behind `ExpertExec`, host
    /// round trip, concurrent execution, host scatter-add) equals the
    /// same-backend routed path — here with the pool holding Q8 experts
    /// on the "GPU" tier and f32 ones on the "CPU" tier side by side, and
    /// a hot-swap mid-way. The f32-tier experts must match exactly; the
    /// Q8 ones within quantization noise.
    #[test]
    fn pooled_experts_match_local_and_hot_swap() {
        use crate::tier::{Precision, Tier};
        use std::sync::Arc;
        let device = crate::backend::cpu_device();
        let dense = moe(&device);
        let split = |e: usize| -> ExpertWeights {
            let slice = |bank: &Param<Tensor<3>>| -> Tensor<2> {
                let [_, out, inp] = bank.val().dims();
                bank.val()
                    .narrow(0, e, 1)
                    .reshape([out, inp])
                    .swap_dims(0, 1)
            };
            ExpertWeights {
                gate: Param::from_tensor(slice(&dense.experts.gate)),
                up: Param::from_tensor(slice(&dense.experts.up)),
                down: Param::from_tensor(slice(&dense.experts.down)),
            }
        };
        let local = SparseMoePerExpert {
            gate: dense.gate.clone(),
            experts: (0..EXPERTS).map(split).collect(),
        };
        // `Device` is a clonable runtime value in burn 0.22 (not a `Copy`
        // marker type), so the closure clones per expert instead of copying.
        let exec = |e: usize, tier: Tier| -> Arc<dyn ExpertExec> {
            Arc::new(DeviceExpert {
                native_ok: std::sync::atomic::AtomicBool::new(true),
                weights: split(e),
                device: device.clone(),
                tier,
                bytes: 0,
            })
        };
        let f32_tier = Tier {
            device: 0,
            precision: Precision::F32,
        };
        let pool = ExpertPool::new(vec![(0..EXPERTS).map(|e| exec(e, f32_tier)).collect()]);
        assert_eq!((pool.num_layers(), pool.experts_per_layer()), (1, EXPERTS));

        for (t, seed, norm) in [(1usize, 2.0f32, false), (5, 7.0, true)] {
            let x = input(t, seed, &device);
            let a = local
                .forward(x.clone(), TOP_K, norm)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let b = local
                .forward_pooled(x, TOP_K, norm, &pool, 0)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for (i, (da, db)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (da - db).abs() < 1e-5,
                    "t={t} elem {i}: local {da} vs pooled {db}"
                );
            }
        }
        // Hits were counted: t=1 and t=5 with top-2 → 12 routed rows total.
        let hits = pool.take_hits();
        assert_eq!(hits.iter().sum::<u64>(), 12, "{hits:?}");
        assert_eq!(pool.take_hits().iter().sum::<u64>(), 0);

        // Hot-swap expert 1 onto a different tier (same weights): output
        // unchanged, tier bookkeeping updated, old executor handed back.
        let q_tier = Tier {
            device: 1,
            precision: Precision::Q8,
        };
        let old = pool.swap(0, 1, exec(1, q_tier));
        assert_eq!(old.tier(), f32_tier);
        assert_eq!(pool.tiers()[1], q_tier);
        let x = input(5, 7.0, &device);
        let a = local
            .forward(x.clone(), TOP_K, true)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = local
            .forward_pooled(x, TOP_K, true, &pool, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (i, (da, db)) in a.iter().zip(&b).enumerate() {
            assert!((da - db).abs() < 1e-5, "after swap elem {i}: {da} vs {db}");
        }
    }

    /// `compute_weight` hands `matmul` something that produces the RIGHT
    /// ANSWER — either the quantized weight itself (where the backend
    /// multiplies it natively) or a dequantized copy (where it does not).
    ///
    /// The test asserts the invariant, not the mechanism: which branch runs
    /// depends on the backend, the burn version and the dtype, and is probed
    /// at runtime. Asserting "it always dequantizes" would have to be
    /// rewritten every time a backend gains a working kernel — and would
    /// have failed the moment burn 0.22 fixed CUDA.
    #[test]
    fn compute_weight_yields_a_matmul_ready_weight() {
        use crate::quant::{QuantPolicy, quantize_weight};
        use burn::tensor::TensorData;
        let device = crate::backend::cpu_device();
        let vals: Vec<f32> = (0..32 * 64).map(|i| ((i as f32) * 0.05).sin()).collect();
        let t = Tensor::<2>::from_data(TensorData::new(vals.clone(), [32, 64]), &device);

        // Float param: returned as-is (still float, never re-quantized).
        let out_f32 = compute_weight(&Param::from_tensor(t.clone()));
        assert!(
            !matches!(out_f32.dtype(), DType::QFloat(_)),
            "float weight must stay float"
        );

        // Quantized param: whatever comes back must MULTIPLY correctly.
        let q = quantize_weight(QuantPolicy::Q8, t.clone());
        assert!(
            matches!(q.dtype(), DType::QFloat(_)),
            "weight should be quantized"
        );
        let ready = compute_weight(&Param::from_tensor(q));

        let x = Tensor::<2>::from_data(
            TensorData::new(vec![0.25f32; 32], [1, 32]),
            (&device, crate::backend::float_dtype(&device)),
        );
        let want = x.clone().matmul(t).into_data().to_vec::<f32>().unwrap();
        let got = x.matmul(ready).into_data().to_vec::<f32>().unwrap();
        let scale = want.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert!(
                (a - b).abs() <= 0.05 * scale,
                "elem {i}: quantized path gave {b}, f32 gave {a}"
            );
        }
    }

    /// On wgpu a quantized weight must come back float from
    /// `compute_weight`: the native q_matmul panics on shapes the tier
    /// planner routinely produces (m=1 decode x some group widths), so the
    /// packed tensor must never reach `matmul`. Storage stays quantized;
    /// only the multiply dequantizes. Skips without a wgpu device.
    #[test]
    fn wgpu_weights_are_dequantized_before_matmul() {
        if !crate::backend::inventory().has_gpu() {
            return;
        }
        let device = crate::backend::gpu_device();
        if !is_wgpu(&device) {
            return;
        }
        let vals: Vec<f32> = (0..64 * 64).map(|i| ((i as f32) * 0.03).sin()).collect();
        let t = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vals, [64, 64]),
            (&device, crate::backend::float_dtype(&device)),
        );
        let q = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q8, t);
        assert!(
            matches!(q.dtype(), DType::QFloat(_)),
            "setup: weight quantized"
        );
        let ready = compute_weight(&Param::from_tensor(q));
        assert!(
            !matches!(ready.dtype(), DType::QFloat(_)),
            "a packed weight must never reach matmul on wgpu"
        );
    }

    fn input(t: usize, seed: f32, device: &Dev) -> Tensor<3> {
        let data: Vec<f32> = (0..t * HIDDEN)
            .map(|i| ((i as f32 + seed) * 0.9).sin())
            .collect();
        Tensor::<1>::from_data(TensorData::new(data, [t * HIDDEN]), device).reshape([1, t, HIDDEN])
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Hand-rolled f32 reference of the whole block (per token: router
    /// softmax, top-k, sparse weighted sum of per-expert SwiGLUs).
    fn reference(m: &SparseMoe, x: &[f32], t: usize, norm: bool) -> Vec<f32> {
        let rw = m.gate.weight.val().into_data().to_vec::<f32>().unwrap(); // [h, e]
        let gw = m.experts.gate.val().into_data().to_vec::<f32>().unwrap(); // [e, inter, h]
        let uw = m.experts.up.val().into_data().to_vec::<f32>().unwrap();
        let dw = m.experts.down.val().into_data().to_vec::<f32>().unwrap(); // [e, h, inter]
        let mut out = vec![0f32; t * HIDDEN];
        for tok in 0..t {
            let xrow = &x[tok * HIDDEN..][..HIDDEN];
            // Router logits then softmax over all experts.
            let mut logits = [0f32; EXPERTS];
            for (e, logit) in logits.iter_mut().enumerate() {
                *logit = (0..HIDDEN).map(|i| xrow[i] * rw[i * EXPERTS + e]).sum();
            }
            let max = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|v| v / z).collect();
            // Top-k expert ids by probability.
            let mut order: Vec<usize> = (0..EXPERTS).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let picked = &order[..TOP_K];
            let denom: f32 = if norm {
                picked.iter().map(|&e| probs[e]).sum()
            } else {
                1.0
            };
            for &e in picked {
                let w = probs[e] / denom;
                // SwiGLU of expert e.
                let mut act = [0f32; INTER];
                for (j, a) in act.iter_mut().enumerate() {
                    let g: f32 = (0..HIDDEN)
                        .map(|i| xrow[i] * gw[(e * INTER + j) * HIDDEN + i])
                        .sum();
                    let u: f32 = (0..HIDDEN)
                        .map(|i| xrow[i] * uw[(e * INTER + j) * HIDDEN + i])
                        .sum();
                    *a = silu(g) * u;
                }
                for i in 0..HIDDEN {
                    let d: f32 = (0..INTER)
                        .map(|j| act[j] * dw[(e * HIDDEN + i) * INTER + j])
                        .sum();
                    out[tok * HIDDEN + i] += w * d;
                }
            }
        }
        out
    }

    #[test]
    fn forward_matches_the_hand_rolled_sparse_reference() {
        let device = crate::backend::cpu_device();
        let m = moe(&device);
        // Both shapes the decoder actually runs: a multi-token prefill and
        // the single-token decode step.
        for t in [5, 1] {
            for norm in [false, true] {
                let x = input(t, 2.0, &device);
                let xv = x.clone().into_data().to_vec::<f32>().unwrap();
                let got = m
                    .forward(x, TOP_K, norm)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let want = reference(&m, &xv, t, norm);
                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-5,
                        "t={t} norm={norm} elem {i}: got {g} vs reference {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn top_k_equal_to_num_experts_uses_every_expert() {
        // With k == E and renorm the block degenerates to a full softmax
        // mixture — the reference covers it; this pins the k=E edge.
        let device = crate::backend::cpu_device();
        let m = moe(&device);
        let x = input(3, 0.5, &device);
        let xv = x.clone().into_data().to_vec::<f32>().unwrap();
        let got = m
            .forward(x, EXPERTS, false)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // Reference with TOP_K replaced by all experts: weights are the full
        // softmax row, every expert contributes.
        let mut want = vec![0f32; 3 * HIDDEN];
        {
            let full = reference_all_experts(&m, &xv, 3);
            want.copy_from_slice(&full);
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "elem {i}: got {g} vs full-mixture {w}"
            );
        }
    }

    /// Full-mixture reference (every expert, softmax-weighted) for the k=E edge.
    fn reference_all_experts(m: &SparseMoe, x: &[f32], t: usize) -> Vec<f32> {
        let rw = m.gate.weight.val().into_data().to_vec::<f32>().unwrap();
        let gw = m.experts.gate.val().into_data().to_vec::<f32>().unwrap();
        let uw = m.experts.up.val().into_data().to_vec::<f32>().unwrap();
        let dw = m.experts.down.val().into_data().to_vec::<f32>().unwrap();
        let mut out = vec![0f32; t * HIDDEN];
        for tok in 0..t {
            let xrow = &x[tok * HIDDEN..][..HIDDEN];
            let mut logits = [0f32; EXPERTS];
            for (e, logit) in logits.iter_mut().enumerate() {
                *logit = (0..HIDDEN).map(|i| xrow[i] * rw[i * EXPERTS + e]).sum();
            }
            let max = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            for e in 0..EXPERTS {
                let w = exps[e] / z;
                let mut act = [0f32; INTER];
                for (j, a) in act.iter_mut().enumerate() {
                    let g: f32 = (0..HIDDEN)
                        .map(|i| xrow[i] * gw[(e * INTER + j) * HIDDEN + i])
                        .sum();
                    let u: f32 = (0..HIDDEN)
                        .map(|i| xrow[i] * uw[(e * INTER + j) * HIDDEN + i])
                        .sum();
                    *a = silu(g) * u;
                }
                for i in 0..HIDDEN {
                    let d: f32 = (0..INTER)
                        .map(|j| act[j] * dw[(e * HIDDEN + i) * INTER + j])
                        .sum();
                    out[tok * HIDDEN + i] += w * d;
                }
            }
        }
        out
    }

    #[test]
    fn norm_topk_weights_change_the_mixture() {
        // norm_topk_prob renormalizes the k weights to sum 1 — unless the
        // top-k already captured all the mass, outputs must differ.
        let device = crate::backend::cpu_device();
        let m = moe(&device);
        let x = input(4, 7.0, &device);
        let a = m
            .forward(x.clone(), TOP_K, false)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = m
            .forward(x, TOP_K, true)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let differs = a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-7);
        assert!(differs, "renormalized weights should scale the output");
    }

    #[test]
    fn forward_is_position_independent() {
        // MoE acts per-token: the same row in different positions/batches
        // routes and computes identically.
        let device = crate::backend::cpu_device();
        let m = moe(&device);
        let row = input(1, 11.0, &device);
        let double = Tensor::cat(vec![row.clone(), row.clone()], 1);
        let s = m
            .forward(row, TOP_K, false)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let d = m
            .forward(double, TOP_K, false)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(s.as_slice(), &d[..HIDDEN]);
        assert_eq!(s.as_slice(), &d[HIDDEN..]);
    }

    #[test]
    fn zero_input_gives_zero_output() {
        // Bias-free SwiGLU experts map 0 to 0 regardless of routing.
        let device = crate::backend::cpu_device();
        let m = moe(&device);
        let x = Tensor::<3>::zeros([1, 2, HIDDEN], &device);
        let out = m
            .forward(x, TOP_K, false)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "top_k")]
    fn forward_rejects_top_k_above_num_experts() {
        let device = crate::backend::cpu_device();
        let m = moe(&device);
        let x = input(1, 0.0, &device);
        let _ = m.forward(x, EXPERTS + 1, false);
    }

    #[test]
    #[should_panic(expected = "num_experts_per_tok")]
    fn config_rejects_zero_top_k() {
        let device = crate::backend::cpu_device();
        let _ = SparseMoeConfig {
            hidden_size: 4,
            expert_intermediate_size: 3,
            num_experts: 4,
            num_experts_per_tok: 0,
        }
        .init(&device);
    }
}
