//! SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`. Field names
//! mirror the HF Qwen2 layout (`gate_proj`/`up_proj`/`down_proj`); LFM2's
//! `w1`/`w3`/`w2` remap onto these at load time.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, activation};

/// SwiGLU MLP, no biases (both proven architectures ship it bias-free).
#[derive(Module, Debug)]
pub struct SwiGluMlp {
    /// SiLU branch (LFM2: `w1`).
    pub gate_proj: Linear,
    /// Multiplicative branch (LFM2: `w3`).
    pub up_proj: Linear,
    /// Projection back to the model width (LFM2: `w2`).
    pub down_proj: Linear,
}

/// Shape config for [`SwiGluMlp`].
#[derive(Debug, Clone)]
pub struct SwiGluMlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl SwiGluMlpConfig {
    /// Initialize the module (random weights; real weights come from import).
    pub fn init(&self, device: &Device) -> SwiGluMlp {
        assert!(self.hidden_size >= 1, "SwiGLU: hidden_size must be >= 1");
        assert!(
            self.intermediate_size >= 1,
            "SwiGLU: intermediate_size must be >= 1"
        );
        SwiGluMlp {
            gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

impl SwiGluMlp {
    /// `[b, t, hidden]` → `[b, t, hidden]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type Dev = burn::tensor::Device;

    #[test]
    fn forward_preserves_shape() {
        let device = crate::backend::cpu_device();
        let mlp = SwiGluMlpConfig {
            hidden_size: 8,
            intermediate_size: 20,
        }
        .init(&device);
        let x = Tensor::<3>::zeros([2, 3, 8], &device);
        assert_eq!(mlp.forward(x).dims(), [2, 3, 8]);
    }

    #[test]
    fn zero_input_gives_zero_output_without_biases() {
        let device = crate::backend::cpu_device();
        let mlp = SwiGluMlpConfig {
            hidden_size: 4,
            intermediate_size: 8,
        }
        .init(&device);
        let x = Tensor::<3>::zeros([1, 2, 4], &device);
        let out = mlp.forward(x).into_data().to_vec::<f32>().unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "bias-free SwiGLU must map 0 to 0"
        );
    }

    #[test]
    fn forward_is_position_independent() {
        // An MLP acts per-position: the same row through a [1,1,h] and a
        // [1,2,h] batch must give identical outputs.
        let device = crate::backend::cpu_device();
        let mlp = SwiGluMlpConfig {
            hidden_size: 4,
            intermediate_size: 8,
        }
        .init(&device);
        let row: Vec<f32> = vec![0.3, -1.2, 0.8, 2.0];
        let single = Tensor::<1>::from_data(TensorData::new(row.clone(), [4]), &device)
            .reshape([1, 1, 4]);
        let double =
            Tensor::<1>::from_data(TensorData::new([row.clone(), row].concat(), [8]), &device)
                .reshape([1, 2, 4]);
        let s = mlp.forward(single).into_data().to_vec::<f32>().unwrap();
        let d = mlp.forward(double).into_data().to_vec::<f32>().unwrap();
        assert_eq!(s.as_slice(), &d[..4]);
        assert_eq!(s.as_slice(), &d[4..]);
    }
}
