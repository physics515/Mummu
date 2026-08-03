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
use burn::tensor::{DType, Distribution, Int, Tensor, activation, backend::Backend};

/// The expert bank: `num_experts` SwiGLU MLPs as three fused params in
/// `[experts, out, in]` layout (the row-major twin of ggml's
/// `ffn_*_exps.weight`). Forward transposes lazily; no per-expert modules.
#[derive(Module, Debug)]
pub struct MoeExperts<B: Backend> {
    /// `[num_experts, intermediate, hidden]` — SiLU branch.
    pub gate: Param<Tensor<B, 3>>,
    /// `[num_experts, intermediate, hidden]` — multiplicative branch.
    pub up: Param<Tensor<B, 3>>,
    /// `[num_experts, hidden, intermediate]` — back to the model width.
    pub down: Param<Tensor<B, 3>>,
}

/// Router + expert bank. Field names follow the HF `Olmoe` checkpoint layout
/// (`mlp.gate` is the router Linear, `mlp.experts` the bank).
#[derive(Module, Debug)]
pub struct SparseMoe<B: Backend> {
    /// The routing projection: `hidden -> num_experts`, no bias.
    pub gate: Linear<B>,
    pub experts: MoeExperts<B>,
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> SparseMoe<B> {
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
        let init = |out: usize, inp: usize, dev: &B::Device| {
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

impl<B: Backend> SparseMoe<B> {
    /// `[b, t, hidden]` → `[b, t, hidden]`.
    ///
    /// Router math mirrors HF `OlmoeSparseMoeBlock`: softmax over **all**
    /// experts in f32, keep the top-`top_k` probabilities as the mixture
    /// weights (renormalized to sum 1 iff `norm_topk_prob` — OLMoE ships
    /// `false`). The f32 island matters on f16 backends (softmax of wide
    /// logits); every cast is a no-op on f32.
    pub fn forward(&self, x: Tensor<B, 3>, top_k: usize, norm_topk_prob: bool) -> Tensor<B, 3> {
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
            Tensor::<B, 1, Int>::arange(0..e as i64, &xt.device()).reshape([1, 1, e as i32]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use burn::tensor::TensorData;

    type Dev = burn::tensor::Device<Cpu>;

    const HIDDEN: usize = 4;
    const INTER: usize = 3;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;

    /// Deterministic weights: expert `e`'s matrices are small distinct
    /// sinusoids so every expert computes something different.
    fn moe(device: &Dev) -> SparseMoe<Cpu> {
        let fill = |seed: f32, dims: [usize; 3]| {
            let n = dims[0] * dims[1] * dims[2];
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
                .collect();
            Param::from_tensor(Tensor::<Cpu, 3>::from_data(
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
        .init::<Cpu>(device);
        // Burn Linear stores weight as [in, out].
        m.gate.weight = Param::from_tensor(Tensor::<Cpu, 2>::from_data(
            TensorData::new(router_data, [HIDDEN, EXPERTS]),
            device,
        ));
        m.experts.gate = fill(0.1, [EXPERTS, INTER, HIDDEN]);
        m.experts.up = fill(1.7, [EXPERTS, INTER, HIDDEN]);
        m.experts.down = fill(3.3, [EXPERTS, HIDDEN, INTER]);
        m
    }

    fn input(t: usize, seed: f32, device: &Dev) -> Tensor<Cpu, 3> {
        let data: Vec<f32> = (0..t * HIDDEN)
            .map(|i| ((i as f32 + seed) * 0.9).sin())
            .collect();
        Tensor::<Cpu, 1>::from_data(TensorData::new(data, [t * HIDDEN]), device)
            .reshape([1, t, HIDDEN])
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Hand-rolled f32 reference of the whole block (per token: router
    /// softmax, top-k, sparse weighted sum of per-expert SwiGLUs).
    fn reference(m: &SparseMoe<Cpu>, x: &[f32], t: usize, norm: bool) -> Vec<f32> {
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
        let device = Dev::default();
        let m = moe(&device);
        for norm in [false, true] {
            let x = input(5, 2.0, &device);
            let xv = x.clone().into_data().to_vec::<f32>().unwrap();
            let got = m
                .forward(x, TOP_K, norm)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let want = reference(&m, &xv, 5, norm);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() < 1e-5,
                    "norm={norm} elem {i}: got {g} vs reference {w}"
                );
            }
        }
    }

    #[test]
    fn top_k_equal_to_num_experts_uses_every_expert() {
        // With k == E and renorm the block degenerates to a full softmax
        // mixture — the reference covers it; this pins the k=E edge.
        let device = Dev::default();
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
    fn reference_all_experts(m: &SparseMoe<Cpu>, x: &[f32], t: usize) -> Vec<f32> {
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
        let device = Dev::default();
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
        let device = Dev::default();
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
        let device = Dev::default();
        let m = moe(&device);
        let x = Tensor::<Cpu, 3>::zeros([1, 2, HIDDEN], &device);
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
        let device = Dev::default();
        let m = moe(&device);
        let x = input(1, 0.0, &device);
        let _ = m.forward(x, EXPERTS + 1, false);
    }

    #[test]
    #[should_panic(expected = "num_experts_per_tok")]
    fn config_rejects_zero_top_k() {
        let device = Dev::default();
        let _ = SparseMoeConfig {
            hidden_size: 4,
            expert_intermediate_size: 3,
            num_experts: 4,
            num_experts_per_tok: 0,
        }
        .init::<Cpu>(&device);
    }
}
