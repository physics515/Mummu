//! LFM2's double-gated causal short-convolution ("LIV") operator:
//! `in_proj` → split B/C/x → `B*x` → causal depthwise conv1d → `C*conv` →
//! `out_proj`, with a rolling `K-1` state so decode steps cost O(K) instead
//! of re-forwarding the sequence. The decode path was verified algebraically
//! equivalent to the padded conv in laurelane (greedy parity vs Ollama).

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::{Tensor, backend::Backend};

/// Rolling decode state: the last `K-1` gated inputs `[b, d, K-1]`, `None`
/// until the first forward.
pub type ConvState<B> = Option<Tensor<B, 3>>;

/// Double-gated causal short conv (LFM2 "LIV" block). Field names mirror the
/// HF checkpoint (`in_proj`/`conv`/`out_proj`).
#[derive(Module, Debug)]
pub struct ShortConv<B: Backend> {
    pub in_proj: Linear<B>,
    pub conv: Conv1d<B>,
    pub out_proj: Linear<B>,
}

/// Shape config for [`ShortConv`].
#[derive(Debug, Clone)]
pub struct ShortConvConfig {
    pub hidden_size: usize,
    /// Conv kernel length `K` (LFM2's `conv_L_cache`).
    pub kernel_len: usize,
}

impl ShortConvConfig {
    /// Initialize the module (random weights; real weights come from import).
    pub fn init<B: Backend>(&self, device: &B::Device) -> ShortConv<B> {
        let (d, k) = (self.hidden_size, self.kernel_len);
        assert!(d >= 1, "ShortConv: hidden_size must be >= 1");
        assert!(
            k >= 2,
            "ShortConv: kernel_len must be >= 2 (a 1-tap conv is a no-op gate)"
        );
        ShortConv {
            in_proj: LinearConfig::new(d, 3 * d).with_bias(false).init(device),
            conv: Conv1dConfig::new(d, d, k)
                .with_groups(d)
                .with_padding(PaddingConfig1d::Explicit(k - 1, k - 1))
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(d, d).with_bias(false).init(device),
        }
    }
}

impl<B: Backend> ShortConv<B> {
    /// Cache-aware forward. `x` is `[b, t, d]`: the whole prompt at prefill,
    /// one token per decode step after. Rolls `state` forward either way.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        kernel_len: usize,
        state: &mut ConvState<B>,
    ) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        let kk = kernel_len;
        assert!(kk >= 2, "ShortConv forward: kernel_len must be >= 2");
        assert!(t >= 1, "ShortConv forward: need at least one position");
        debug_assert!(
            state.as_ref().is_none_or(|s| s.dims() == [b, d, kk - 1]),
            "ShortConv forward: stale state shape"
        );

        let bcx = self.in_proj.forward(x); // [b, t, 3d]
        let bb = bcx.clone().narrow(2, 0, d);
        let cc = bcx.clone().narrow(2, d, d);
        let xx = bcx.narrow(2, 2 * d, d);
        let bx = bb.mul(xx).swap_dims(1, 2); // input gate, channel-major [b, d, t]

        let conv_out = if t > 1 {
            // Prefill: the padded depthwise Conv1d gives the full causal output.
            self.conv.forward(bx.clone()).narrow(2, 0, t) // [b, d, t]
        } else {
            // Decode: weighted sum over the last K inputs = [cached (K-1), new (1)].
            // Equivalent to the padded conv at the new position.
            let window = match state {
                Some(prev) => Tensor::cat(vec![prev.clone(), bx.clone()], 2), // [b, d, K]
                None => {
                    let pad = Tensor::<B, 3>::zeros([b, d, kk - 1], &bx.device());
                    Tensor::cat(vec![pad, bx.clone()], 2)
                }
            };
            let w = self.conv.weight.val().reshape([1, d, kk]); // depthwise kernel
            window.mul(w).sum_dim(2) // [b, d, 1]
        };

        // Roll the state forward: keep the last (K-1) gated inputs.
        let new_state = {
            let combined = match state {
                Some(prev) => Tensor::cat(vec![prev.clone(), bx.clone()], 2),
                None => bx.clone(),
            };
            let len = combined.dims()[2];
            if len >= kk - 1 {
                combined.narrow(2, len - (kk - 1), kk - 1)
            } else {
                let pad = Tensor::<B, 3>::zeros([b, d, (kk - 1) - len], &combined.device());
                Tensor::cat(vec![pad, combined], 2)
            }
        };
        *state = Some(new_state);

        let conv_out = conv_out.swap_dims(1, 2); // [b, t, d]
        self.out_proj.forward(cc.mul(conv_out)) // output gate + proj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use burn::tensor::TensorData;

    type Dev = burn::tensor::Device<Cpu>;

    const D: usize = 6;
    const K: usize = 3;

    fn conv(device: &Dev) -> ShortConv<Cpu> {
        ShortConvConfig {
            hidden_size: D,
            kernel_len: K,
        }
        .init(device)
    }

    fn input(t: usize, seed: f32, device: &Dev) -> Tensor<Cpu, 3> {
        let data: Vec<f32> = (0..t * D)
            .map(|i| ((i as f32 + seed) * 0.9).cos())
            .collect();
        Tensor::<Cpu, 2>::from_data(TensorData::new(data, [t, D]), device).reshape([1, t, D])
    }

    /// The load-bearing invariant: prefill + one-token-at-a-time decode via
    /// the rolling state must equal one full prefill over the same tokens.
    #[test]
    fn rolling_state_decode_matches_full_prefill() {
        let device = Dev::default();
        let c = conv(&device);
        let x = input(7, 5.0, &device);

        // Reference: all 7 positions through the padded conv.
        let mut ref_state: ConvState<Cpu> = None;
        let full = c
            .forward(x.clone(), K, &mut ref_state)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Cached: prefill 4, then decode 5th..7th one at a time.
        let mut state: ConvState<Cpu> = None;
        let _ = c.forward(x.clone().narrow(1, 0, 4), K, &mut state);
        for pos in 4..7 {
            let out = c
                .forward(x.clone().narrow(1, pos, 1), K, &mut state)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let expect = &full[pos * D..(pos + 1) * D];
            for (i, (got, want)) in out.iter().zip(expect).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "pos {pos} elem {i}: cached {got} vs full {want}"
                );
            }
        }
    }

    /// The conv is causal: a future token cannot change an earlier output.
    #[test]
    fn conv_is_causal() {
        let device = Dev::default();
        let c = conv(&device);
        let x1 = input(5, 1.0, &device);
        let x2 = Tensor::cat(vec![x1.clone().narrow(1, 0, 4), input(1, 77.0, &device)], 1);
        let mut s1: ConvState<Cpu> = None;
        let mut s2: ConvState<Cpu> = None;
        let o1 = c
            .forward(x1, K, &mut s1)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let o2 = c
            .forward(x2, K, &mut s2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for i in 0..4 * D {
            assert!((o1[i] - o2[i]).abs() < 1e-6, "past row changed at {i}");
        }
    }

    /// Decode from a fresh (None) state equals a length-1 prefill: the
    /// zero-pad seeding path.
    #[test]
    fn first_decode_step_seeds_state_like_prefill() {
        let device = Dev::default();
        let c = conv(&device);
        let x = input(1, 2.0, &device);

        let mut s_decode: ConvState<Cpu> = None;
        let via_decode = c
            .forward(x.clone(), K, &mut s_decode)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Same single token inside a longer prefill whose first position it is.
        let longer = Tensor::cat(vec![x, input(2, 50.0, &device)], 1);
        let mut s_pre: ConvState<Cpu> = None;
        let via_prefill = c
            .forward(longer, K, &mut s_pre)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        for i in 0..D {
            assert!(
                (via_decode[i] - via_prefill[i]).abs() < 1e-5,
                "elem {i}: decode {} vs prefill {}",
                via_decode[i],
                via_prefill[i]
            );
        }
        // State must exist and have the rolling shape after either path.
        assert_eq!(s_decode.unwrap().dims(), [1, D, K - 1]);
        assert_eq!(s_pre.unwrap().dims(), [1, D, K - 1]);
    }

    #[test]
    #[should_panic(expected = "kernel_len must be >= 2")]
    fn init_rejects_degenerate_kernel() {
        let device = Dev::default();
        let _ = ShortConvConfig {
            hidden_size: D,
            kernel_len: 1,
        }
        .init::<Cpu>(&device);
    }
}
