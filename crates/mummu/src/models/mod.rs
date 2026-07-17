//! The model zoo: from-scratch architectures on the shared `nn` blocks, all
//! generic over `B: Backend`, all config-driven (hyperparameters come from the
//! checkpoint's `config.json`, never hardcoded).

use burn::tensor::{Tensor, backend::Backend};

use crate::decode::{SamplerOptions, generate_loop, top_k_ids};

pub mod lfm2;
pub mod minilm;
pub mod qwen2;
pub mod qwen3;

/// The contract every causal LM in the zoo implements. A new architecture
/// (Hermes-class function-caller, Gemma, Qwen3, …) provides its cache type,
/// its forward pass, and its EOS check — decoding (greedy, sampled, streamed,
/// cancellable) comes for free from the shared driver.
pub trait CausalLm<B: Backend> {
    /// Per-generation decode state (KV cache, conv state, …).
    type Cache;

    /// A fresh (empty) cache for one generation.
    fn new_cache(&self) -> Self::Cache;

    /// Forward `new_ids` (the whole prompt when `past == 0`, else one decode
    /// token), updating `cache`; returns logits for the **last** position,
    /// `[1, vocab]`.
    fn forward(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Self::Cache,
        device: &B::Device,
    ) -> Tensor<B, 2>;

    /// Is `id` an end-of-sequence token for this model?
    fn is_eos(&self, id: u32) -> bool;

    /// Full decode: prefill once, then one token per step, stopping at EOS,
    /// `max_tokens`, or a `Break` from `on_token` (streaming + cooperative
    /// cancellation). Greedy (`temperature == 0`) keeps the argmax on-device.
    fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        opts: &SamplerOptions,
        device: &B::Device,
        on_token: impl FnMut(u32) -> std::ops::ControlFlow<()>,
    ) -> Result<Vec<u32>, String> {
        let mut cache = self.new_cache();
        generate_loop(
            |ids, past| self.forward(ids, past, &mut cache, device),
            prompt_ids,
            max_tokens,
            opts,
            |id| self.is_eos(id),
            on_token,
        )
    }

    /// Greedy decode (the parity-gate path): [`Self::generate`] at
    /// temperature 0 with no streaming.
    fn greedy_generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        device: &B::Device,
    ) -> Result<Vec<u32>, String> {
        self.generate(
            prompt_ids,
            max_tokens,
            &SamplerOptions::greedy(),
            device,
            |_| std::ops::ControlFlow::Continue(()),
        )
    }

    /// Parity probe: top-k next-token ids for a single prefill.
    fn first_token(
        &self,
        prompt_ids: &[u32],
        k: usize,
        device: &B::Device,
    ) -> Result<Vec<u32>, String> {
        assert!(!prompt_ids.is_empty(), "first_token: empty prompt");
        assert!(k >= 1, "first_token: k must be >= 1");
        let mut cache = self.new_cache();
        let logits = self.forward(prompt_ids, 0, &mut cache, device);
        let v = logits
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| format!("logits readback: {e:?}"))?;
        Ok(top_k_ids(&v, k))
    }
}
