//! Decode-loop primitives shared by every causal model.
//!
//! On-device argmax, a top-k probe, temperature/top-p sampling with a
//! deterministic in-house RNG, and the streaming `generate_loop` driver with
//! cooperative cancellation.

use std::ops::ControlFlow;

use burn::tensor::Tensor;
use mummu_num::f32_from_u32;

use crate::constrain::Constraint;

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
///
/// # Errors
///
/// A message when the device readback fails, returns no element, or returns
/// an index that is negative or does not fit a `u32`.
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
    u32::try_from(id).map_err(|_| format!("argmax returned an out-of-range id {id}"))
}

/// Indices of the `k` largest values, descending (the parity probe's top-k).
///
/// # Panics
///
/// When `v` is empty or `k` is 0. The indices are also converted to `u32`,
/// which cannot fail for any vocabulary this crate loads.
#[must_use]
pub fn top_k_ids(v: &[f32], k: usize) -> Vec<u32> {
    assert!(!v.is_empty(), "top_k_ids: empty logits");
    assert!(k >= 1, "top_k_ids: k must be >= 1");
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter()
        .take(k)
        .map(|i| u32::try_from(i).expect("vocab index fits u32"))
        .collect()
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
    pub const fn new(seed: u64) -> Self {
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

    pub const fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        // The low 32 bits are the output word; the mask makes that explicit.
        let xorshifted = ((((old >> 18) ^ old) >> 27) & 0xFFFF_FFFF) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1) with 24 bits of mantissa.
    pub fn next_f32(&mut self) -> f32 {
        // 24 random bits scaled by 2^-24: both factors are exact in f32.
        f32_from_u32(self.next_u32() >> 8) * (1.0 / 16_777_216.0)
    }
}

/// Sample a token id from raw logits with temperature + top-k + top-p.
/// Pure and deterministic given (logits, opts, rng state). `temperature == 0`
/// callers should use [`argmax_id`] instead (asserted here).
///
/// # Panics
///
/// As [`sample_id_filtered`]: an invalid `opts` (non-finite or negative
/// temperature, `top_p` outside `(0, 1]`, `top_k` of 0), empty logits or
/// more than the readback bound of them, or a temperature of 0. The
/// unfiltered candidate set is never empty, so the unwrap here cannot fail.
#[must_use]
pub fn sample_id(logits: &[f32], opts: &SamplerOptions, rng: &mut Pcg32) -> u32 {
    sample_id_filtered(logits, opts, rng, |_| true)
        .expect("an unfiltered candidate set is never empty")
}

/// [`sample_id`] restricted to the ids `allowed` accepts — the constrained
/// sampling path.
///
/// The filter is applied **inside** the top-k candidate set, so a constraint
/// costs at most `top_k` grammar tests per token rather than one per
/// vocabulary entry, and the nucleus is renormalized over what survives.
/// That keeps the draw a proper sample from the restricted distribution
/// instead of sample-then-reject, which would quietly bias toward whatever
/// the grammar happens to permit. `None` means the grammar rejected every
/// candidate in the top-k; callers widen the search from there.
///
/// # Panics
///
/// When `opts` is invalid (non-finite or negative temperature, `top_p`
/// outside `(0, 1]`, `top_k` of 0), when `logits` is empty or longer than
/// the readback bound, or when `temperature` is 0 — that is the
/// [`argmax_id`] path.
#[must_use]
pub fn sample_id_filtered(
    logits: &[f32],
    opts: &SamplerOptions,
    rng: &mut Pcg32,
    allowed: impl Fn(u32) -> bool,
) -> Option<u32> {
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
    let vocab = u32::try_from(logits.len()).expect("vocab size fits u32");
    let mut idx: Vec<u32> = (0..vocab).collect();
    let by_logit_desc = |&a: &u32, &b: &u32| logits[b as usize].total_cmp(&logits[a as usize]);
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, by_logit_desc);
        idx.truncate(k);
    }
    idx.sort_unstable_by(by_logit_desc);
    idx.retain(|&i| allowed(i));
    if idx.is_empty() {
        return None;
    }

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
    Some(chosen)
}

/// The highest-logit id `allowed` accepts, or `None` when it accepts none.
///
/// Two-stage on purpose: the legal token is essentially always among the
/// strongest few hundred, and `select_nth` is linear where a full sort of a
/// ~150k vocabulary is not. The tail is only sorted when the head had
/// nothing, which in practice means a grammar that has painted the model
/// into a corner.
///
/// # Panics
///
/// Only on an internal invariant that cannot fail: the vocabulary size is
/// converted to `u32` for the ids.
#[must_use]
pub fn best_allowed(logits: &[f32], allowed: impl Fn(u32) -> bool) -> Option<u32> {
    /// Head width for the first stage.
    const PROBE: usize = 512;

    let by_logit_desc = |&a: &u32, &b: &u32| logits[b as usize].total_cmp(&logits[a as usize]);
    let vocab = u32::try_from(logits.len()).expect("vocab size fits u32");
    let mut idx: Vec<u32> = (0..vocab).collect();
    if idx.len() > PROBE {
        idx.select_nth_unstable_by(PROBE - 1, by_logit_desc);
        let (head, tail) = idx.split_at_mut(PROBE);
        head.sort_unstable_by(by_logit_desc);
        if let Some(&hit) = head.iter().find(|&&i| allowed(i)) {
            return Some(hit);
        }
        tail.sort_unstable_by(by_logit_desc);
        return tail.iter().copied().find(|&i| allowed(i));
    }
    idx.sort_unstable_by(by_logit_desc);
    idx.into_iter().find(|&i| allowed(i))
}

/// Prompt tokens fed per prefill call (`MUMMU_PREFILL_CHUNK`; `0`/`off`
/// disables chunking).
///
/// The default 1024 keeps the peak `SwiGLU` activation
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

/// Why a generation stopped — what a client is told as ollama's
/// `done_reason` or `OpenAI`'s `finish_reason`.
///
/// One enum for both decode drivers. [`generate_loop`] never reports
/// [`Self::Context`] (it has no ceiling of its own), and a batch
/// ([`crate::batch`]) never reports [`Self::Cancelled`] (a cancelled
/// sequence just leaves).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// It picked an end-of-sequence id (not emitted).
    Eos,
    /// It emitted its `max_tokens`.
    Length,
    /// Its constraint's value closed.
    Complete,
    /// It reached the model's context.
    Context,
    /// Its caller stopped it: `on_token` returned `Break`.
    Cancelled,
}

/// What [`generate_loop`] decoded: the emitted ids, and why it stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generated {
    pub ids: Vec<u32>,
    pub finish: Finish,
}

/// The shared decode driver: prefill via `step` (in chunks — see
/// [`prefill_chunk_len`]), then one token per iteration.
///
/// Emits each accepted token through `on_token`; a `Break` return cancels
/// cooperatively *before* the next forward, and so does the `max_tokens`-th
/// token (its logits would never be read). EOS is never emitted. The result
/// says which of these stopped it (see [`Finish`]).
///
/// `step(new_ids, past, need_logits)` advances the cache; it must return
/// `Some([1, vocab] logits)` for the last position whenever `need_logits`
/// is true, and MAY return `None` when it is false — the non-final chunks
/// of a chunked prefill, where a computed head projection is a throwaway
/// (~68 ms/token of host head on the 27B, once per chunk). Models plug in
/// via `CausalLm::forward` / `CausalLm::forward_advance`.
///
/// `constraint`, when present, decides which ids may be emitted (see
/// [`crate::constrain`]). It also owns the stop condition: EOS is suppressed
/// until the constrained value is complete, and the loop ends the moment it
/// is.
///
/// # Errors
///
/// A message when a device readback (the on-device argmax or the logit row)
/// fails, when the argmax returns an id outside the vocabulary — the NaN /
/// numeric-collapse signature — or when the constraint rejects every token
/// in the vocabulary.
///
/// # Panics
///
/// When `opts` is invalid (see [`sample_id_filtered`]), `prompt_ids` is
/// empty, `max_tokens` is 0, or `step` returns `None` for a call that asked
/// for logits. The vocabulary width is also converted to `u32`, which
/// cannot fail for any model this crate loads.
pub async fn generate_loop(
    mut step: impl FnMut(&[u32], usize, bool) -> Option<Tensor<2>>,
    prompt_ids: &[u32],
    max_tokens: usize,
    opts: &SamplerOptions,
    is_eos: impl Fn(u32) -> bool,
    mut on_token: impl FnMut(u32) -> ControlFlow<()>,
    mut constraint: Option<&mut dyn Constraint>,
) -> Result<Generated, String> {
    opts.validate();
    assert!(!prompt_ids.is_empty(), "generate_loop: empty prompt");
    assert!(max_tokens >= 1, "generate_loop: max_tokens must be >= 1");

    let mut picker = Picker::new(opts);
    let mut logits = {
        let _s = crate::prof::scope("prefill");
        // Chunked prefill: feeding the prompt in slices bounds the widest
        // live activation at [chunk, intermediate] instead of
        // [prompt, intermediate] — on the 27B that turns an ~816 MiB
        // accelerator reserve into ~214 MiB, which is two to three more
        // resident layers, every token, forever. Exact by the same invariant
        // the caches are tested on (prefill+decode ≡ full forward): the KV
        // cat, the conv window and the DeltaNet state all carry across
        // calls. Non-final chunks advance with `need_logits = false`, so a
        // model with a skip-head mode never computes their throwaway
        // lm_head projections (the 27B's host head is ~68 ms per chunk).
        let chunk = prefill_chunk_len();
        let mut done = 0usize;
        let mut last = None;
        while done < prompt_ids.len() {
            let end = done.saturating_add(chunk).min(prompt_ids.len());
            let is_final = end == prompt_ids.len();
            let out = step(&prompt_ids[done..end], done, is_final);
            if is_final {
                last = Some(out.expect("step must return logits when need_logits is true"));
            }
            done = end;
        }
        last.expect("non-empty prompt")
    };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    // The position the picked token is fed at.
    let mut past = prompt_ids.len();
    let finish = loop {
        let vocab = u32::try_from(logits.dims()[1]).expect("vocab size fits u32");
        // This span crosses an await, so it must not hold a scope guard
        // (thread-local stack; the future may resume on another worker) —
        // timed by hand and attributed with `record` instead. It is also
        // where every enqueued-but-unfinished GPU op comes due: the readback
        // is the sync point, so GPU-side FFN time surfaces HERE, not in the
        // scopes that enqueued it.
        let readback_started = std::time::Instant::now();
        let next = picker
            .pick(logits, None, constraint.as_deref(), past, &is_eos)
            .await?;
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
            break Finish::Eos;
        }
        out.push(next);
        if let Some(c) = constraint.as_deref_mut() {
            c.accept(next);
        }
        if on_token(next).is_break() {
            break Finish::Cancelled;
        }
        // The constrained value closed. Nothing after it can belong to the
        // value, and letting the model free-associate past the last brace
        // costs a full token of decode per word of it.
        if constraint
            .as_deref()
            .is_some_and(super::constrain::Constraint::is_complete)
        {
            break Finish::Complete;
        }
        // The budget is spent. Stopping HERE, before the step, is what saves
        // a whole forward: its logits would never be read (a token on the
        // 27B; a batch retires the sequence at the same point).
        if out.len() == max_tokens {
            break Finish::Length;
        }
        // Cooperative yield: a CPU-backend decode is a long stretch of
        // blocking compute between awaits, and without this a single
        // generation would monopolize its worker for the whole request.
        tokio::task::yield_now().await;
        logits = {
            let _s = crate::prof::scope("step");
            step(&[next], past, true).expect("step must return logits when need_logits is true")
        };
        past += 1;
    };
    debug_assert!(out.len() <= max_tokens);
    Ok(Generated { ids: out, finish })
}

/// One sequence's token choice: its sampler options and RNG, applied to a
/// row of logits — [`generate_loop`]'s pick, and every batched sequence's.
///
/// The constraint is the caller's (it also decides when to stop), and so is
/// EOS handling: a pick may BE the EOS id, which the caller does not emit.
pub struct Picker {
    opts: SamplerOptions,
    rng: Pcg32,
}

impl Picker {
    /// # Panics
    ///
    /// When `opts` is invalid (see [`sample_id_filtered`]).
    #[must_use]
    pub fn new(opts: &SamplerOptions) -> Self {
        opts.validate();
        Self {
            opts: opts.clone(),
            rng: Pcg32::new(opts.seed),
        }
    }

    /// Whether picks are greedy (an argmax) rather than sampled.
    #[must_use]
    pub fn greedy(&self) -> bool {
        self.opts.temperature == 0.0
    }

    /// The next id from `[1, vocab]` logits for position `past`.
    /// `argmax`, when the caller already read it back (a batch reads every
    /// row's at once), saves the on-device argmax and its sync; a sampled
    /// pick never uses it.
    ///
    /// # Errors
    ///
    /// A failed readback, or a constraint that rejects every token.
    ///
    /// # Panics
    ///
    /// Only when the vocabulary does not fit `u32`, which no model this crate
    /// loads comes near.
    pub async fn pick(
        &mut self,
        logits: Tensor<2>,
        argmax: Option<u32>,
        constraint: Option<&dyn Constraint>,
        past: usize,
        is_eos: impl Fn(u32) -> bool,
    ) -> Result<u32, String> {
        let greedy = self.greedy();
        let argmax_of = |l: Tensor<2>| async move {
            match argmax {
                Some(id) => Ok(id),
                None => argmax_id(l).await,
            }
        };
        let vocab = u32::try_from(logits.dims()[1]).expect("vocab size fits u32");
        Ok(match constraint {
            // Unconstrained: greedy keeps its argmax on-device and never
            // reads a vocabulary back.
            None => {
                if greedy {
                    argmax_of(logits).await?
                } else {
                    let v = read_logits(logits).await?;
                    sample_id(&v, &self.opts, &mut self.rng)
                }
            }
            Some(c) => {
                // EOS counts as legal only once the value is complete — the
                // rule that stops a model abandoning an object halfway and
                // handing the client something that cannot parse.
                let legal = |id: u32| {
                    if is_eos(id) {
                        c.is_complete()
                    } else {
                        c.allows(id)
                    }
                };
                if greedy {
                    // Fast path: test the model's own pick, which costs one
                    // automaton replay. A model already emitting valid JSON
                    // never pays for the readback the slow path needs.
                    let probe = argmax_of(logits.clone()).await?;
                    if probe < vocab && legal(probe) {
                        probe
                    } else {
                        let v = read_logits(logits).await?;
                        best_allowed(&v, legal).ok_or_else(|| no_legal_token(past))?
                    }
                } else {
                    // Sampling already reads the vocabulary back, so masking
                    // inside the top-k is the whole added cost.
                    let v = read_logits(logits).await?;
                    sample_id_filtered(&v, &self.opts, &mut self.rng, legal)
                        .or_else(|| best_allowed(&v, legal))
                        .ok_or_else(|| no_legal_token(past))?
                }
            }
        })
    }
}

/// Pull a `[1, vocab]` logit row back to the host as f32.
async fn read_logits(logits: Tensor<2>) -> Result<Vec<f32>, String> {
    logits
        .into_data_async()
        .await
        .map_err(|e| format!("logits readback: {e:?}"))?
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|e| format!("logits readback: {e:?}"))
}

/// The constraint left the decoder with nowhere to go. Unreachable for the
/// JSON grammar over a real vocabulary — every state has a legal
/// continuation and a byte-level BPE can spell all of them — so this names
/// the constraint as the suspect rather than the model.
fn no_legal_token(step: usize) -> String {
    format!(
        "decode step {step}: the output constraint rejected every token in the vocabulary — \
         the grammar has no legal continuation here, which is a constraint bug, not a \
         model failure"
    )
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
        let (mut first, mut second) = (Pcg32::new(7), Pcg32::new(7));
        let seq_a: Vec<u32> = (0..8).map(|_| first.next_u32()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| second.next_u32()).collect();
        assert_eq!(seq_a, seq_b, "same seed must replay the same stream");

        let mut other = Pcg32::new(8);
        let seq_c: Vec<u32> = (0..8).map(|_| other.next_u32()).collect();
        assert_ne!(seq_a, seq_c, "different seeds must diverge");

        let mut rng = Pcg32::new(99);
        for _ in 0..1000 {
            let sample = rng.next_f32();
            assert!(
                (0.0..1.0).contains(&sample),
                "next_f32 out of [0,1): {sample}"
            );
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
    fn toy_step(
        device: &burn::tensor::Device,
    ) -> impl FnMut(&[u32], usize, bool) -> Option<Tensor<2>> {
        let device = device.clone();
        move |new_ids: &[u32], _past: usize, need_logits: bool| {
            if !need_logits {
                return None;
            }
            let peak = if new_ids.last() == Some(&2) { 3 } else { 2 };
            let mut v = vec![0.0f32; 8];
            v[peak] = 9.0;
            Some(Tensor::<1>::from_floats(v.as_slice(), &device).reshape([1, 8]))
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
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.ids, vec![2], "one token, then EOS never emitted");
        assert_eq!(out.finish, Finish::Eos);
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
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.ids.len(), 2, "break after the 2nd token stops the loop");
        assert_eq!(streamed, out.ids, "every emitted token was streamed");
        assert_eq!(out.finish, Finish::Cancelled);
    }

    /// Running out of budget is told apart from the model ending the answer:
    /// it is what a client sees as `done_reason: "length"`.
    #[tokio::test]
    async fn generate_loop_that_spends_its_budget_stops_for_length() {
        let device = crate::backend::cpu_device();
        let out = generate_loop(
            toy_step(&device),
            &[1],
            3,
            &SamplerOptions::greedy(),
            |_| false,
            |_| std::ops::ControlFlow::Continue(()),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.ids, vec![2, 3, 2]);
        assert_eq!(out.finish, Finish::Length);
    }

    /// A generation that spends its budget stops before the next forward:
    /// the prefill, then one step per token after the first — not one more,
    /// whose logits nothing would read. Each step feeds the token it was
    /// given at the position after the last.
    #[tokio::test]
    async fn generate_loop_does_not_step_past_its_budget() {
        let device = crate::backend::cpu_device();
        let mut toy = toy_step(&device);
        let mut steps = Vec::new();
        let out = generate_loop(
            |ids: &[u32], past: usize, need_logits: bool| {
                steps.push((ids.to_vec(), past));
                toy(ids, past, need_logits)
            },
            &[1, 1],
            3,
            &SamplerOptions::greedy(),
            |_| false,
            |_| std::ops::ControlFlow::Continue(()),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.ids, vec![2, 3, 2]);
        assert_eq!(out.finish, Finish::Length);
        assert_eq!(steps, vec![(vec![1, 1], 0), (vec![2], 2), (vec![3], 3)]);
    }

    /// Accepts anything; its value closes after `.0` tokens.
    struct CloseAfter(usize);

    impl Constraint for CloseAfter {
        fn allows(&self, _: u32) -> bool {
            true
        }

        fn accept(&mut self, _: u32) {
            self.0 = self.0.saturating_sub(1);
        }

        fn is_complete(&self) -> bool {
            self.0 == 0
        }
    }

    /// A value that closes stops the answer, not the budget — even when it
    /// closes on the budget's last token.
    #[tokio::test]
    async fn generate_loop_stops_when_the_constrained_value_closes() {
        let device = crate::backend::cpu_device();
        for (closes_after, budget) in [(2, 5), (3, 3)] {
            let mut constraint = CloseAfter(closes_after);
            let out = generate_loop(
                toy_step(&device),
                &[1],
                budget,
                &SamplerOptions::greedy(),
                |_| false,
                |_| std::ops::ControlFlow::Continue(()),
                Some(&mut constraint),
            )
            .await
            .unwrap();
            assert_eq!(out.ids.len(), closes_after);
            assert_eq!(out.finish, Finish::Complete, "{closes_after} in {budget}");
        }
    }

    #[test]
    fn top_k_ids_orders_descending() {
        let v = [0.1f32, 5.0, -2.0, 3.0];
        assert_eq!(top_k_ids(&v, 3), vec![1, 3, 0]);
    }
}
