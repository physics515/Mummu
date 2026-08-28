//! Decode-loop primitives shared by every causal model: on-device argmax, a
//! top-k probe, temperature/top-p sampling with a deterministic in-house RNG,
//! and the streaming `generate_loop` driver with cooperative cancellation.

use std::ops::ControlFlow;

use burn::tensor::Tensor;

/// Hard ceiling on the vocab a sampled step will read back to the CPU
/// (~4 MB of f32 at the bound); anything larger is a wiring bug, not a model.
const VOCAB_READBACK_BOUND: usize = 1 << 20;

/// Candidate-set cap when sampling: top-p truncation happens *within* the
/// `top_k` highest-logit tokens, so the post-softmax walk is O(k log k), not
/// O(vocab log vocab). 1024 keeps >99.9% of realistic nucleus mass.
const DEFAULT_TOP_K: usize = 1024;

/// Greedy next-token id from `[1, vocab]` logits. The argmax runs
/// **on-device** and only the single winning index is synced back — vs.
/// copying a whole ~150k-logit vector to the CPU every decode step.
pub async fn argmax_id(logits: Tensor<2>) -> Result<u32, String> {
    debug_assert!(logits.dims()[0] == 1, "argmax_id expects [1, vocab] logits");
    let data = logits
        .argmax(1)
        .into_data_async()
        .await
        .map_err(|e| format!("argmax readback: {e:?}"))?
        .convert::<i64>()
        .try_to_vec::<i64>()
        .map_err(|e| format!("argmax readback: {e:?}"))?;
    debug_assert!(data.len() == 1, "argmax over [1, vocab] must yield one id");
    let id = data.first().copied().ok_or("argmax returned no data")?;
    Ok(id as u32)
}

/// Indices of the `k` largest values, descending (the parity probe's top-k).
#[must_use]
pub fn top_k_ids(v: &[f32], k: usize) -> Vec<u32> {
    assert!(!v.is_empty(), "top_k_ids: empty logits");
    assert!(k >= 1, "top_k_ids: k must be >= 1");
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(k).map(|i| i as u32).collect()
}

/// Sampling knobs for one generation. `temperature == 0` means exact greedy
/// (argmax stays on-device; nothing else is consulted).
#[derive(Debug, Clone)]
pub struct SamplerOptions {
    /// 0 = greedy; higher flattens the distribution. Must be finite and >= 0.
    pub temperature: f32,
    /// Nucleus mass in (0, 1]: sample only from the smallest prefix of
    /// probability-sorted candidates whose mass reaches this.
    pub top_p: f32,
    /// Candidate-set cap applied before top-p (>= 1).
    pub top_k: usize,
    /// RNG seed: the same (options, logits, seed) always picks the same token.
    pub seed: u64,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: DEFAULT_TOP_K,
            seed: 0,
        }
    }
}

impl SamplerOptions {
    /// Greedy decoding (temperature 0) — the parity-gate configuration.
    #[must_use]
    pub fn greedy() -> Self {
        Self::default()
    }

    fn validate(&self) {
        assert!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "sampler: temperature must be finite and >= 0, got {}",
            self.temperature
        );
        assert!(
            self.top_p > 0.0 && self.top_p <= 1.0,
            "sampler: top_p must be in (0, 1], got {}",
            self.top_p
        );
        assert!(self.top_k >= 1, "sampler: top_k must be >= 1");
    }
}

/// PCG-XSH-RR 32 (O'Neill): a tiny deterministic RNG so sampling is
/// reproducible from a seed without pulling in a rand dependency.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULT: u64 = 6_364_136_223_846_793_005;

    #[must_use]
    pub fn new(seed: u64) -> Self {
        // Fixed stream; the standard seeding dance (advance, add, advance).
        let mut rng = Self {
            state: 0,
            inc: (54 << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1) with 24 bits of mantissa.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1 << 24) as f32)
    }
}

/// Sample a token id from raw logits with temperature + top-k + top-p.
/// Pure and deterministic given (logits, opts, rng state). `temperature == 0`
/// callers should use [`argmax_id`] instead (asserted here).
#[must_use]
pub fn sample_id(logits: &[f32], opts: &SamplerOptions, rng: &mut Pcg32) -> u32 {
    opts.validate();
    assert!(!logits.is_empty(), "sample_id: empty logits");
    assert!(
        logits.len() <= VOCAB_READBACK_BOUND,
        "sample_id: vocab {} exceeds the readback bound",
        logits.len()
    );
    assert!(
        opts.temperature > 0.0,
        "sample_id: temperature 0 is the argmax path"
    );

    // Top-k prefilter: O(vocab) partial select, then sort just the candidates.
    let k = opts.top_k.min(logits.len());
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let by_logit_desc = |&a: &u32, &b: &u32| logits[b as usize].total_cmp(&logits[a as usize]);
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, by_logit_desc);
        idx.truncate(k);
    }
    idx.sort_unstable_by(by_logit_desc);

    // Temperature softmax over the candidates (max-subtracted: never overflows).
    let max_logit = logits[idx[0] as usize];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i as usize] - max_logit) / opts.temperature).exp())
        .collect();
    let total: f32 = probs.iter().sum();
    debug_assert!(total > 0.0, "softmax mass must be positive");
    for p in &mut probs {
        *p /= total;
    }

    // Nucleus: keep the smallest probability-sorted prefix with mass >= top_p
    // (probs are already descending because idx is logit-sorted).
    let mut cut = probs.len();
    let mut mass = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        mass += p;
        if mass >= opts.top_p {
            cut = i + 1;
            break;
        }
    }
    debug_assert!(cut >= 1, "nucleus must keep at least the top token");

    // Draw within the (renormalized) nucleus by cumulative walk.
    let nucleus_mass: f32 = probs[..cut].iter().sum();
    let mut u = rng.next_f32() * nucleus_mass;
    let mut chosen = idx[cut - 1]; // fallback: rounding can leave u > 0 at the end
    for (i, &p) in probs[..cut].iter().enumerate() {
        if u < p {
            chosen = idx[i];
            break;
        }
        u -= p;
    }
    assert!(
        (chosen as usize) < logits.len(),
        "sampled id out of the vocab"
    );
    chosen
}

/// Prompt tokens fed per prefill call (`MUMMU_PREFILL_CHUNK`; `0`/`off`
/// disables chunking). The default 1024 keeps the peak SwiGLU activation
/// (`3 · chunk · intermediate · 4 B`) near 200 MiB on the 27B — the number
/// the accelerator reserve in mummu-serve is derived from, so the two must
/// move together. `mummu_schedule::prefill::best_chunk` solves for the
/// optimum from measured constants; this is its memory-feasible default.
#[must_use]
pub fn prefill_chunk_len() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("MUMMU_PREFILL_CHUNK") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => usize::MAX,
        Ok(v) => v.parse::<usize>().ok().filter(|&c| c >= 1).unwrap_or(1024),
        Err(_) => 1024,
    })
}

/// The shared decode driver: prefill via `step` (in chunks — see
/// [`prefill_chunk_len`]), then one token per iteration. Emits each accepted
/// token through `on_token`; a `Break` return cancels cooperatively *before*
/// the next forward. EOS is never emitted.
/// `step(new_ids, past)` returns `[1, vocab]` logits for the last position.
pub async fn generate_loop(
    mut step: impl FnMut(&[u32], usize) -> Tensor<2>,
    prompt_ids: &[u32],
    max_tokens: usize,
    opts: &SamplerOptions,
    is_eos: impl Fn(u32) -> bool,
    mut on_token: impl FnMut(u32) -> ControlFlow<()>,
) -> Result<Vec<u32>, String> {
    opts.validate();
    assert!(!prompt_ids.is_empty(), "generate_loop: empty prompt");
    assert!(max_tokens >= 1, "generate_loop: max_tokens must be >= 1");

    let mut rng = Pcg32::new(opts.seed);
    let greedy = opts.temperature == 0.0;
    let mut logits = {
        let _s = crate::prof::scope("prefill");
        // Chunked prefill: feeding the prompt in slices bounds the widest
        // live activation at [chunk, intermediate] instead of
        // [prompt, intermediate] — on the 27B that turns an ~816 MiB
        // accelerator reserve into ~214 MiB, which is two to three more
        // resident layers, every token, forever. Exact by the same invariant
        // the caches are tested on (prefill+decode ≡ full forward): the KV
        // cat, the conv window and the DeltaNet state all carry across
        // calls. Cost: each non-final chunk computes one throwaway lm_head
        // projection (`CausalLm::forward` has no skip-head mode yet).
        let chunk = prefill_chunk_len();
        let mut done = 0usize;
        let mut last = None;
        while done < prompt_ids.len() {
            let end = done.saturating_add(chunk).min(prompt_ids.len());
            last = Some(step(&prompt_ids[done..end], done));
            done = end;
        }
        last.expect("non-empty prompt")
    };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    for past in (prompt_ids.len()..).take(max_tokens) {
        let vocab = logits.dims()[1] as u32;
        // This span crosses an await, so it must not hold a scope guard
        // (thread-local stack; the future may resume on another worker) —
        // timed by hand and attributed with `record` instead. It is also
        // where every enqueued-but-unfinished GPU op comes due: the readback
        // is the sync point, so GPU-side FFN time surfaces HERE, not in the
        // scopes that enqueued it.
        let readback_started = std::time::Instant::now();
        let next = if greedy {
            argmax_id(logits).await?
        } else {
            let v = logits
                .into_data_async()
                .await
                .map_err(|e| format!("logits readback: {e:?}"))?
                .convert::<f32>()
                .try_to_vec::<f32>()
                .map_err(|e| format!("logits readback: {e:?}"))?;
            sample_id(&v, opts, &mut rng)
        };
        crate::prof::record("logits_readback+sample", readback_started.elapsed());
        // A GPU argmax over NaN logits can return an out-of-range sentinel
        // (observed: exactly `vocab` on f16 numeric collapse) — fail loudly
        // instead of emitting garbage ids the tokenizer silently drops.
        if next >= vocab {
            return Err(format!(
                "decode step {past}: id {next} is outside the {vocab}-token vocab — NaN logits / numeric collapse on this backend?"
            ));
        }
        if is_eos(next) {
            break;
        }
        out.push(next);
        if on_token(next).is_break() {
            break;
        }
        // Cooperative yield: a CPU-backend decode is a long stretch of
        // blocking compute between awaits, and without this a single
        // generation would monopolize its worker for the whole request.
        tokio::task::yield_now().await;
        logits = {
            let _s = crate::prof::scope("step");
            step(&[next], past)
        };
    }
    debug_assert!(out.len() <= max_tokens);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    #[tokio::test]
    async fn argmax_id_finds_the_peak() {
        let device = crate::backend::cpu_device();
        let logits = Tensor::<1>::from_floats([0.1, -2.0, 7.5, 3.0], &device).reshape([1, 4]);
        assert_eq!(argmax_id(logits).await.unwrap(), 2);
    }

    #[test]
    fn pcg32_is_deterministic_per_seed_and_in_unit_range() {
        let (mut a, mut b) = (Pcg32::new(7), Pcg32::new(7));
        let seq_a: Vec<u32> = (0..8).map(|_| a.next_u32()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.next_u32()).collect();
        assert_eq!(seq_a, seq_b, "same seed must replay the same stream");

        let mut c = Pcg32::new(8);
        let seq_c: Vec<u32> = (0..8).map(|_| c.next_u32()).collect();
        assert_ne!(seq_a, seq_c, "different seeds must diverge");

        let mut r = Pcg32::new(99);
        for _ in 0..1000 {
            let f = r.next_f32();
            assert!((0.0..1.0).contains(&f), "next_f32 out of [0,1): {f}");
        }
    }

    #[test]
    fn sample_id_peaked_logits_always_pick_the_peak() {
        let logits = [0.0f32, 30.0, -5.0, 1.0];
        let opts = SamplerOptions {
            temperature: 0.8,
            top_p: 0.95,
            ..SamplerOptions::default()
        };
        for seed in 0..32 {
            let mut rng = Pcg32::new(seed);
            assert_eq!(sample_id(&logits, &opts, &mut rng), 1);
        }
    }

    #[test]
    fn sample_id_top_k_one_is_argmax_at_any_temperature() {
        let logits = [1.0f32, 3.0, 2.0, 2.9];
        let opts = SamplerOptions {
            temperature: 10.0,
            top_p: 1.0,
            top_k: 1,
            seed: 0,
        };
        for seed in 0..16 {
            let mut rng = Pcg32::new(seed);
            assert_eq!(
                sample_id(
                    &logits,
                    &SamplerOptions {
                        seed,
                        ..opts.clone()
                    },
                    &mut rng
                ),
                1
            );
        }
    }

    #[test]
    fn sample_id_tiny_top_p_degenerates_to_argmax() {
        let logits = [1.0f32, 1.1, 0.9, 1.05];
        let opts = SamplerOptions {
            temperature: 5.0,
            top_p: 0.01,
            ..SamplerOptions::default()
        };
        for seed in 0..16 {
            let mut rng = Pcg32::new(seed);
            assert_eq!(sample_id(&logits, &opts, &mut rng), 1);
        }
    }

    #[test]
    fn sample_id_high_temperature_spreads_over_candidates() {
        let logits = [2.0f32, 2.0, 2.0, 2.0];
        let opts = SamplerOptions {
            temperature: 1.0,
            top_p: 1.0,
            ..SamplerOptions::default()
        };
        let picks: std::collections::HashSet<u32> = (0..64)
            .map(|seed| sample_id(&logits, &opts, &mut Pcg32::new(seed)))
            .collect();
        assert!(
            picks.len() >= 3,
            "uniform logits over 64 seeds should hit >= 3 of 4 ids, got {picks:?}"
        );
        for &p in &picks {
            assert!(p < 4);
        }
    }

    #[test]
    #[should_panic(expected = "argmax path")]
    fn sample_id_rejects_temperature_zero() {
        let mut rng = Pcg32::new(0);
        let _ = sample_id(&[1.0, 2.0], &SamplerOptions::greedy(), &mut rng);
    }

    /// A fixed toy vocab where the "model" always prefers id 2, then id 3
    /// after seeing 2 — enough to drive the loop without weights.
    fn toy_step(device: &burn::tensor::Device) -> impl FnMut(&[u32], usize) -> Tensor<2> {
        let device = device.clone();
        move |new_ids: &[u32], _past: usize| {
            let peak = if new_ids.last() == Some(&2) { 3 } else { 2 };
            let mut v = vec![0.0f32; 8];
            v[peak] = 9.0;
            Tensor::<1>::from_floats(v.as_slice(), &device).reshape([1, 8])
        }
    }

    #[tokio::test]
    async fn generate_loop_greedy_follows_argmax_and_stops_at_eos() {
        let device = crate::backend::cpu_device();
        let out = generate_loop(
            toy_step(&device),
            &[1],
            6,
            &SamplerOptions::greedy(),
            |id| id == 3, // treat the follow-up token as EOS
            |_| std::ops::ControlFlow::Continue(()),
        )
        .await.unwrap();
        assert_eq!(out, vec![2], "one token, then EOS never emitted");
    }

    #[tokio::test]
    async fn generate_loop_cancels_cooperatively_between_tokens() {
        let device = crate::backend::cpu_device();
        let mut streamed = Vec::new();
        let out = generate_loop(
            toy_step(&device),
            &[1],
            100,
            &SamplerOptions::greedy(),
            |_| false, // no EOS: only the callback can stop this
            |id| {
                streamed.push(id);
                if streamed.len() == 2 {
                    std::ops::ControlFlow::Break(())
                } else {
                    std::ops::ControlFlow::Continue(())
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(out.len(), 2, "break after the 2nd token stops the loop");
        assert_eq!(streamed, out, "every emitted token was streamed");
    }

    #[test]
    fn top_k_ids_orders_descending() {
        let v = [0.1f32, 5.0, -2.0, 3.0];
        assert_eq!(top_k_ids(&v, 3), vec![1, 3, 0]);
    }
}
