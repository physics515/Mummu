//! The model zoo: from-scratch architectures on the shared `nn` blocks, all
//! generic over `B: Backend`, all config-driven (hyperparameters come from the
//! checkpoint's `config.json`, never hardcoded).

use burn::tensor::{Device, Tensor};

use crate::decode::{SamplerOptions, argmax_id, generate_loop, top_k_ids};

pub mod lfm2;
pub mod minilm;
pub mod olmoe;
pub mod qwen2;
pub mod qwen3;
pub mod qwen35;

/// Upper bound on one [`CausalLm::warm_up`] call. A warm-up is a fixed,
/// bounded cost paid off the user's critical path — not a place to spend
/// unbounded GPU time — and the measured curve flattens after ~32 steps
/// (`mummu-bench/tests/warmup_f16.rs`), so this ceiling is 8x the useful
/// depth, not a tuning knob.
pub const MAX_WARM_UP_STEPS: usize = 256;

/// The contract every causal LM in the zoo implements. A new architecture
/// (Hermes-class function-caller, Gemma, Qwen3, …) provides its cache type,
/// its forward pass, and its EOS check — decoding (greedy, sampled, streamed,
/// cancellable) comes for free from the shared driver.
pub trait CausalLm {
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
        device: &Device,
    ) -> Tensor<2>;

    /// Is `id` an end-of-sequence token for this model?
    fn is_eos(&self, id: u32) -> bool;

    /// Advance the cache over `new_ids` WITHOUT producing logits — the
    /// non-final chunks of a chunked prefill, where a computed head
    /// projection is pure waste (the 27B's host head costs ~68 ms per
    /// chunk). The default runs the full forward and discards it, so a
    /// model earns the saving by overriding (qwen35 skips the final norm
    /// and head); semantics are otherwise identical to [`Self::forward`].
    fn forward_advance(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Self::Cache,
        device: &Device,
    ) {
        let _ = self.forward(new_ids, past, cache, device);
    }

    /// Full decode: prefill once, then one token per step, stopping at EOS,
    /// `max_tokens`, or a `Break` from `on_token` (streaming + cooperative
    /// cancellation). Greedy (`temperature == 0`) keeps the argmax on-device.
    fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        opts: &SamplerOptions,
        device: &Device,
        on_token: impl FnMut(u32) -> std::ops::ControlFlow<()>,
    ) -> impl std::future::Future<Output = Result<Vec<u32>, String>> {
        async move {
            let mut cache = self.new_cache();
            generate_loop(
                |ids, past, need_logits| {
                    if need_logits {
                        Some(self.forward(ids, past, &mut cache, device))
                    } else {
                        self.forward_advance(ids, past, &mut cache, device);
                        None
                    }
                },
                prompt_ids,
                max_tokens,
                opts,
                |id| self.is_eos(id),
                on_token,
            )
            .await
        }
    }

    /// Greedy decode (the parity-gate path): [`Self::generate`] at
    /// temperature 0 with no streaming.
    fn greedy_generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        device: &Device,
    ) -> impl std::future::Future<Output = Result<Vec<u32>, String>> {
        // The options must outlive the future, so own them here rather than
        // passing a temporary that dies at the end of this statement.
        async move {
            let opts = SamplerOptions::greedy();
            self.generate(prompt_ids, max_tokens, &opts, device, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .await
        }
    }

    /// Parity probe: top-k next-token ids for a single prefill.
    fn first_token(
        &self,
        prompt_ids: &[u32],
        k: usize,
        device: &Device,
    ) -> impl std::future::Future<Output = Result<Vec<u32>, String>> {
        assert!(!prompt_ids.is_empty(), "first_token: empty prompt");
        assert!(k >= 1, "first_token: k must be >= 1");
        async move {
            let mut cache = self.new_cache();
            let logits = self.forward(prompt_ids, 0, &mut cache, device);
            let v = logits
                .into_data_async()
                .await
                .map_err(|e| format!("logits readback: {e:?}"))?
                .convert::<f32>()
                .try_to_vec::<f32>()
                .map_err(|e| format!("logits readback: {e:?}"))?;
            Ok(top_k_ids(&v, k))
        }
    }

    /// Post-import **sanity smoke**: one forward on `probe_ids` must yield
    /// finite, correctly-sized (`expected_vocab`-wide), non-degenerate logits.
    /// The liveness gate an app calls right after `install` to catch a
    /// silently-broken import — corrupt weights (NaN), a config/tokenizer vocab
    /// mismatch, or a dead/zero-init forward — none of which a checked *load*
    /// can see. This is not parity (an arbitrary import has no reference); it
    /// proves the model actually computes. See [`crate::import::logit_sanity`].
    fn sanity_check(
        &self,
        probe_ids: &[u32],
        expected_vocab: usize,
        device: &Device,
    ) -> impl std::future::Future<Output = Result<crate::import::SanitySmoke, String>> {
        assert!(!probe_ids.is_empty(), "sanity_check: empty probe prompt");
        assert!(
            expected_vocab > 0,
            "sanity_check: expected_vocab must be positive"
        );
        async move {
            let mut cache = self.new_cache();
            let logits = self.forward(probe_ids, 0, &mut cache, device);
            let v = logits
                .into_data_async()
                .await
                .map_err(|e| format!("logits readback: {e:?}"))?
                .convert::<f32>()
                .try_to_vec::<f32>()
                .map_err(|e| format!("logits readback: {e:?}"))?;
            crate::import::logit_sanity(&v, expected_vocab).map_err(|e| e.to_string())
        }
    }

    /// Pay the **cold-start tax off the user's critical path**: one prefill
    /// plus `steps` greedy decode steps on a throwaway cache, discarded.
    ///
    /// A freshly-started process decodes its first tokens far slower than its
    /// steady state — measured on Qwen2.5-1.5B at f16, the first 32 tokens run
    /// at 12.5 tok/s against a steady 37.6, and the curve is *flat* from token
    /// 33 on (`mummu-bench/tests/warmup_f16.rs`). CubeCL already persists its
    /// **autotune** choices to disk across processes, so what is left is
    /// per-process kernel compilation and pipeline creation, which the wgpu
    /// runtime does not cache anywhere (`CompilationCache` is wired for CUDA
    /// and HIP only in cubecl 0.10) — no configuration can carry it, only
    /// spending it earlier can. A consumer that opens short agent turns should
    /// call this once after `install`/load, beside
    /// [`Self::sanity_check`].
    ///
    /// Warms the **decode** step, whose kernels are shape-stable (`t == 1`).
    /// Prefill kernels are keyed by prompt length, so a caller who cares about
    /// TTFT should pass a `probe_ids` of its own typical prompt length rather
    /// than a token or two.
    ///
    /// Returns the number of forwards executed (`steps + 1`). Every step reads
    /// its argmax back, exactly as real decoding does — an unsynchronized
    /// warm-up would queue work and return before the GPU had run any of it.
    fn warm_up(
        &self,
        probe_ids: &[u32],
        steps: usize,
        device: &Device,
    ) -> impl std::future::Future<Output = Result<usize, String>> {
        // Validate EAGERLY, outside the future: an argument bound that only
        // fires when the future is awaited is a contract the caller can hold
        // wrong indefinitely (and a `should_panic` test never sees).
        assert!(!probe_ids.is_empty(), "warm_up: empty probe prompt");
        assert!(steps >= 1, "warm_up: steps must be >= 1");
        assert!(
            steps <= MAX_WARM_UP_STEPS,
            "warm_up: {steps} steps exceeds the {MAX_WARM_UP_STEPS} bound"
        );
        async move {
            let mut cache = self.new_cache();
            let logits = self.forward(probe_ids, 0, &mut cache, device);
            let mut next = argmax_id(logits).await?;
            let mut forwards = 1usize;
            for past in (probe_ids.len()..).take(steps) {
                let logits = self.forward(&[next], past, &mut cache, device);
                next = argmax_id(logits).await?;
                forwards += 1;
            }

            debug_assert_eq!(
                forwards,
                steps + 1,
                "warm_up must run exactly one prefill plus `steps` decode forwards"
            );
            Ok(forwards)
        }
    }
}
