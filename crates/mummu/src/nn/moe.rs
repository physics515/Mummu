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
use burn::tensor::{Device, DType, Distribution, Int, Tensor, activation};

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
        let classes =
            Tensor::<1, Int>::arange(0..e as i64, &xt.device()).reshape([1, 1, e as i32]);
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
                (&device, crate::backend::int_dtype()),
            );
            let x_e = xt.clone().select(0, rows_t.clone()); // [n, h]
            let w = &self.experts[expert];
            let acts = activation::silu(x_e.clone().matmul(w.gate.val()))
                .mul(x_e.matmul(w.up.val()));
            let y = acts.matmul(w.down.val()); // [n, h]
            let scale = Tensor::<1>::from_data(
                burn::tensor::TensorData::new(weights, [n]),
                (&device, crate::backend::float_dtype()),
            )
            .reshape([n, 1])
            .cast(ambient);
            // select_assign accumulates (scatter-add) — a token served by
            // several experts sums their weighted outputs.
            out = out.select_assign(
                0,
                rows_t,
                y.mul(scale),
                burn::tensor::IndexingUpdateOp::Add,
            );
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
            (&device, crate::backend::float_dtype()),
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
            (&self.device, crate::backend::float_dtype()),
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
        let xt = x.to_device(&self.device);
        let w = &self.weights;
        let acts = activation::silu(xt.clone().matmul(compute_weight(&w.gate)))
            .mul(xt.matmul(compute_weight(&w.up)));
        acts.matmul(compute_weight(&w.down)).to_device(&caller)
    }

    fn gate_energy(&self, x: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
        let xt = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(x.to_vec(), [rows, hidden]),
            (&self.device, crate::backend::float_dtype()),
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

/// A pooled expert's weight ready for an `f32`-input matmul: a quantized
/// weight (Q4/Q8, stored compactly) is **dequantized on its own device**
/// first, so the matmul is always float×float. This keeps the storage
/// savings while sidestepping the mixed f32-input × quantized-weight
/// `q_matmul` — which burn 0.21's CUDA backend panics on ("Cast element
/// count must match") and wgpu computes wrong. The dequantized tensor is
/// transient (freed after the matmul); the `Param` stays quantized.
fn compute_weight(w: &Param<Tensor<2>>) -> Tensor<2> {
    let t = w.val();
    if matches!(t.dtype(), DType::QFloat(_)) {
        t.dequantize()
    } else {
        t
    }
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
    pub fn new(weights: ExpertWeights, host_device: Device, tier: crate::tier::Tier, bytes: u64) -> Self {
        Self {
            host: weights,
            host_device,
            resident: std::sync::RwLock::new(None),
            tier,
            bytes,
        }
    }

    /// Is a device copy currently staged?
    #[must_use]
    pub fn is_staged(&self) -> bool {
        self.resident
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Stage onto `device` (the scheduler's prefetch). Idempotent: staging
    /// onto the device it already sits on does nothing, so a redundant
    /// prefetch costs a comparison rather than a transfer.
    pub fn stage(&self, device: &Device) {
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
    pub fn evict(&self) {
        *self.resident.write().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

impl ExpertExec for StagedExpert {
    fn tier(&self) -> crate::tier::Tier {
        self.tier
    }

    fn resident_bytes(&self) -> u64 {
        // Device bytes only: the host copy is the backing store, not part of
        // the working set the planner budgets.
        if self.is_staged() { self.bytes } else { 0 }
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
            (&device, crate::backend::float_dtype()),
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
        let xt = x.to_device(&device);
        let acts = activation::silu(xt.clone().matmul(compute_weight(&w.gate)))
            .mul(xt.matmul(compute_weight(&w.up)));
        acts.matmul(compute_weight(&w.down)).to_device(&caller)
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
                .map(|l| (0..l.len()).map(|_| std::sync::atomic::AtomicU64::new(0)).collect())
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
            dense_rows: [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)],
        }
    }

    /// Output energy accumulated per executor since the last call (flat,
    /// layer-major, ragged rows concatenated); resets.
    pub fn take_energy(&self) -> Vec<f64> {
        self.energy
            .iter()
            .flat_map(|l| l.iter().map(|e| e.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e3))
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
        let execs: Vec<std::sync::Arc<dyn ExpertExec>> = (0..n).map(|e| self.get(layer, e)).collect();

        // Exact mode (no skipping): every cluster runs on every row, so there
        // is nothing to gather — sum the groups as tensors and never touch
        // the host. This is the path a dense model takes, and it removes the
        // per-layer round trip that dominated the 27B's decode.
        if skip.is_none() {
            let mut out: Option<Tensor<2>> = None;
            for (e, exec) in execs.iter().enumerate() {
                let y = exec.run_tensor(xt.clone());
                out = Some(match out {
                    Some(acc) => acc.add(y),
                    None => y,
                });
                self.hits[layer][e].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
                self.dense_rows[0].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
                self.dense_rows[1].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
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
                let energies: Vec<Vec<f32>> =
                    execs.iter().map(|exec| exec.gate_energy(&host, bt, h)).collect();
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
            self.hits[layer][e].fetch_add((energy * 1e3) as u64 + rows.len() as u64, std::sync::atomic::Ordering::Relaxed);
            self.energy[layer][e].fetch_add((energy * 1e3) as u64, std::sync::atomic::Ordering::Relaxed);
            self.dense_rows[0].fetch_add(rows.len() as u64, std::sync::atomic::Ordering::Relaxed);
            self.dense_rows[1].fetch_add(bt as u64, std::sync::atomic::Ordering::Relaxed);
        }
        Some(Tensor::<2>::from_data(
            burn::tensor::TensorData::new(out, [bt, h]),
            (&device, crate::backend::float_dtype()),
        ))
    }

    /// Routing hits since the last call (token-rows served per expert),
    /// flat layer-major; resets the counters.
    pub fn take_hits(&self) -> Vec<u64> {
        self.hits
            .iter()
            .flat_map(|l| l.iter().map(|h| h.swap(0, std::sync::atomic::Ordering::Relaxed)))
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
            (&device, crate::backend::float_dtype()),
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
            Param::from_tensor(Tensor::<3>::from_data(
                TensorData::new(data, dims),
                device,
            ))
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
                bank.val().narrow(0, e, 1).reshape([out, inp]).swap_dims(0, 1)
            };
            ExpertWeights {
                gate: Param::from_tensor(slice(&dense.experts.gate)),
                up: Param::from_tensor(slice(&dense.experts.up)),
                down: Param::from_tensor(slice(&dense.experts.down)),
            }
        };
        let tier = Tier { device: 0, precision: Precision::F32 };
        let staged = StagedExpert::new(split(0), device.clone(), tier, 0);
        // A pinned reference on the same device, for comparison.
        let pinned = DeviceExpert { weights: split(0), device: device.clone(), tier, bytes: 0 };

        let x = Tensor::<2>::from_data(
            TensorData::new((0..HIDDEN).map(|i| (i as f32) * 0.25 - 0.5).collect::<Vec<f32>>(), [1, HIDDEN]),
            (&device, crate::backend::float_dtype()),
        );
        let want = pinned.run_tensor(x.clone()).into_data().to_vec::<f32>().unwrap();

        // Unstaged (the overflow path: computes on the host).
        assert!(!staged.is_staged(), "a fresh StagedExpert holds no device copy");
        assert_eq!(staged.resident_bytes(), 0, "unstaged costs no device memory");
        let overflow = staged.run_tensor(x.clone()).into_data().to_vec::<f32>().unwrap();

        // Staged, then evicted, then staged again.
        staged.stage(&device);
        assert!(staged.is_staged());
        let hot = staged.run_tensor(x.clone()).into_data().to_vec::<f32>().unwrap();
        staged.stage(&device); // idempotent: no second transfer, same answer
        let again = staged.run_tensor(x.clone()).into_data().to_vec::<f32>().unwrap();
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
                bank.val().narrow(0, e, 1).reshape([out, inp]).swap_dims(0, 1)
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
                weights: split(e),
                device: device.clone(),
                tier,
                bytes: 0,
            })
        };
        let f32_tier = Tier { device: 0, precision: Precision::F32 };
        let pool = ExpertPool::new(vec![(0..EXPERTS).map(|e| exec(e, f32_tier)).collect()]);
        assert_eq!((pool.num_layers(), pool.experts_per_layer()), (1, EXPERTS));

        for (t, seed, norm) in [(1usize, 2.0f32, false), (5, 7.0, true)] {
            let x = input(t, seed, &device);
            let a = local.forward(x.clone(), TOP_K, norm).into_data().to_vec::<f32>().unwrap();
            let b = local
                .forward_pooled(x, TOP_K, norm, &pool, 0)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for (i, (da, db)) in a.iter().zip(&b).enumerate() {
                assert!((da - db).abs() < 1e-5, "t={t} elem {i}: local {da} vs pooled {db}");
            }
        }
        // Hits were counted: t=1 and t=5 with top-2 → 12 routed rows total.
        let hits = pool.take_hits();
        assert_eq!(hits.iter().sum::<u64>(), 12, "{hits:?}");
        assert_eq!(pool.take_hits().iter().sum::<u64>(), 0);

        // Hot-swap expert 1 onto a different tier (same weights): output
        // unchanged, tier bookkeeping updated, old executor handed back.
        let q_tier = Tier { device: 1, precision: Precision::Q8 };
        let old = pool.swap(0, 1, exec(1, q_tier));
        assert_eq!(old.tier(), f32_tier);
        assert_eq!(pool.tiers()[1], q_tier);
        let x = input(5, 7.0, &device);
        let a = local.forward(x.clone(), TOP_K, true).into_data().to_vec::<f32>().unwrap();
        let b = local
            .forward_pooled(x, TOP_K, true, &pool, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (i, (da, db)) in a.iter().zip(&b).enumerate() {
            assert!((da - db).abs() < 1e-5, "after swap elem {i}: {da} vs {db}");
        }
    }

    /// The P9-3c fix: a pooled expert with a **quantized** weight must run
    /// through `compute_weight`'s on-device dequantize (never burn's mixed
    /// f32-input × quantized-weight q_matmul, which the CUDA/wgpu backends
    /// mishandle). Here on CPU we check the dequantize itself: an f32 param
    /// passes through untouched; a Q8 param comes back float and ≈ the
    /// original within block-32 quant error.
    #[test]
    fn compute_weight_dequantizes_quantized_params() {
        use crate::quant::{QuantPolicy, quantize_weight};
        use burn::tensor::TensorData;
        let device = crate::backend::cpu_device();
        let vals: Vec<f32> = (0..32 * 64).map(|i| ((i as f32) * 0.05).sin()).collect();
        let t = Tensor::<2>::from_data(TensorData::new(vals.clone(), [32, 64]), &device);

        // Float param: returned as-is (still float).
        let out_f32 = compute_weight(&Param::from_tensor(t.clone()));
        assert!(!matches!(out_f32.dtype(), DType::QFloat(_)), "float weight must stay float");

        // Quantized param: really quantized, and compute_weight floats it back.
        let q = quantize_weight(QuantPolicy::Q8, t.clone());
        assert!(matches!(q.dtype(), DType::QFloat(_)), "weight should be quantized");
        let out_q = compute_weight(&Param::from_tensor(q));
        assert!(!matches!(out_q.dtype(), DType::QFloat(_)), "compute_weight must dequantize");
        let back = out_q.into_data().to_vec::<f32>().unwrap();
        for (i, (x, y)) in vals.iter().zip(&back).enumerate() {
            assert!((x - y).abs() < 0.05, "Q8 dequant elem {i}: {x} vs {y}");
        }
    }

    fn input(t: usize, seed: f32, device: &Dev) -> Tensor<3> {
        let data: Vec<f32> = (0..t * HIDDEN)
            .map(|i| ((i as f32 + seed) * 0.9).sin())
            .collect();
        Tensor::<1>::from_data(TensorData::new(data, [t * HIDDEN]), device)
            .reshape([1, t, HIDDEN])
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
