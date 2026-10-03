//! Captured decode: the decode step recorded once, replayed every token.
//!
//! Batch-1 decode is launch-bound on this stack: a Qwen2.5-1.5B token is 1289
//! kernel dispatches (`examples/decode-dispatch-count.rs`), and each launch
//! spends ~8-17 µs of CPU on pipeline lookup, binding resolution and uniform
//! upload before the GPU sees it (`examples/graph-capture-probe.rs`) — the GPU
//! sits idle in between, which is why a model resident on the card reads
//! well under half busy. [`burn::tensor::capture`] records a closure's
//! launches once and replays them with that work already done (a true CUDA
//! graph on CUDA; a software graph on wgpu).
//!
//! A replay re-runs the recorded kernels against the recorded buffers, so the
//! step must not move: [`StaticDecode::forward_static`] is the model's decode
//! step over preallocated state ([`crate::nn::static_kv`]), reading its
//! token and position from two small device buffers that [`generate`]
//! rewrites in place between replays. The prompt is prefilled on the
//! ordinary (dynamic) path and copied into the static state; from the first
//! generated token on, every step is a replay of the graph for its length
//! bucket.
//!
//! Measured on the RTX 4070 Ti SUPER (Vulkan, `tests/real_capture.rs`),
//! greedy, f16: Qwen3.5-2B 17.1 → 11.7 ms/token, Qwen3-0.6B 14.2 → 10.2. In
//! f32 the step is kernel-bound and capture buys 6-25 %. The same step over
//! several slots ([`generate_batch_greedy`]) is the other half: 16 slots of
//! the 2B decode 461 tokens/s together against 54 alone.
//!
//! # One thread
//!
//! cubecl keeps one stream per thread, and a graph replays on the stream it
//! was recorded on: refreshing an input from another thread is unordered
//! against the replay that reads it. [`generate`] must therefore be polled on
//! one thread for its whole life — [`on_this_thread`] does that from inside
//! tokio. A multi-thread executor that moves the task between workers at an
//! `.await` breaks it.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Mutex;

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::constrain::Constraint;
use crate::decode::{Generated, SamplerOptions, generate_loop};
use crate::models::CausalLm;
use crate::nn::MAX_CONTEXT_TOKENS;
use crate::nn::static_kv::{BUCKET, StaticKv, bucket};

/// A causal LM with a fixed-shape decode step (see the module docs).
pub trait StaticDecode: CausalLm {
    /// What the step runs over: the preallocated KV cache, plus any state a
    /// recurrent layer carries.
    type State: StaticState + Send;

    /// Whether the model, as loaded, can take the static path on `device`
    /// (not with a feature the static step does not implement) — the caller
    /// decodes on the ordinary path otherwise. A model may accept with some
    /// layers on the host; then its step runs op by op (see
    /// [`Self::static_capturable`]). Cheap: no allocation.
    fn static_supported(&self, device: &Device) -> bool;

    /// Whether the step can also be captured on `device`: every buffer it
    /// touches is a tensor there. Not when layers are elsewhere, or when the
    /// step keeps state in host memory. The batcher steps such a model op by
    /// op, and the one-sequence driver ([`generate`]) leaves it to the
    /// ordinary path.
    fn static_capturable(&self, device: &Device) -> bool {
        self.static_supported(device)
    }

    /// Whether a one-sequence step reads its logits through the bounded
    /// host head (`crate::flex::head`), which must be told the k the pick
    /// consults ([`crate::flex::head::RequestTopK`], a process-wide scope).
    /// The batcher sets that scope only then.
    fn static_bounded_head(&self) -> bool {
        false
    }

    /// The bytes [`Self::static_state`] allocates for that shape on
    /// `device` — what a caller planning the card's memory charges a batch.
    fn static_bytes(&self, slots: usize, max_ctx: usize, device: &Device) -> u64;

    /// A zeroed static state for `slots` sequences of up to `max_ctx`
    /// positions on `device`, which [`Self::static_supported`] accepted.
    fn static_state(&self, slots: usize, max_ctx: usize, device: &Device) -> Self::State;

    /// The context it was trained to, when known: a static cache is never
    /// sized past it.
    fn trained_context(&self) -> Option<usize>;

    /// One decode step for the state's first `k` slots: `tokens` `[k, 1]`
    /// at `positions` `[k]`, attending over a bucket of `len` keys. Returns
    /// the logits, `[k, vocab]`. Where [`Self::static_capturable`] holds it
    /// must touch nothing but device tensors (no uploads): it runs inside a
    /// capture window. Every buffer of `state` it advances it must advance
    /// in place.
    fn forward_static(
        &self,
        tokens: &Tensor<2, Int>,
        positions: &Tensor<1, Int>,
        state: &mut Self::State,
        len: usize,
    ) -> Tensor<2>;

    /// Copy a prefilled dynamic cache into `slot` of the static state.
    fn seed_static(&self, state: &mut Self::State, slot: usize, cache: &Self::Cache);
}

/// What the driver needs of a static state besides the step itself.
///
/// Capturing runs the step several times before recording it (to settle its
/// kernels and populate its buffers), and the recorded run is not the step
/// the driver wanted either: the replay after capture is. A KV write is an
/// assignment — the same row, the same values, however often it runs — but a
/// recurrent state advances every time. [`StaticState::mark`] copies it
/// before the capture, [`StaticState::rewind`] writes the copy back in place
/// after it.
pub trait StaticState {
    type Mark;
    fn mark(&self) -> Self::Mark;
    fn rewind(&mut self, mark: Self::Mark);

    /// Slots and positions the state holds.
    fn shape(&self) -> (usize, usize);

    /// Reshape to `slots` slots of `max_ctx` positions (`max_ctx` never
    /// shrinks), keeping every slot both shapes have. New buffers: every
    /// graph captured over the old ones must be dropped first.
    fn resize(&mut self, slots: usize, max_ctx: usize);

    /// Make room for `max_ctx` positions, keeping every slot's contents (see
    /// [`Self::resize`]).
    fn grow(&mut self, max_ctx: usize) {
        self.resize(self.shape().0, max_ctx);
    }

    /// Copy slot `from` over slot `to`, in place — the buffers a captured
    /// graph recorded stay the ones it reads. What keeps a batch's live
    /// sequences in its first slots when one ahead of them finishes.
    fn move_slot(&mut self, from: usize, to: usize);
}

/// Attention alone: its writes are assignments, nothing to undo.
impl StaticState for StaticKv {
    type Mark = ();
    fn mark(&self) {}
    fn rewind(&mut self, (): ()) {}
    fn shape(&self) -> (usize, usize) {
        let c = self.config();
        (c.slots, c.max_ctx)
    }
    fn resize(&mut self, slots: usize, max_ctx: usize) {
        Self::resize(self, slots, max_ctx);
    }
    fn move_slot(&mut self, from: usize, to: usize) {
        Self::move_slot(self, from, to);
    }
}

/// How the decode steps after the prefill run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMode {
    /// The ordinary path: a growing cache, per-step uploads.
    Dynamic,
    /// The static step, executed op by op (the A/B arm that isolates what the
    /// fixed shape costs from what capture saves).
    Static,
    /// The static step, recorded once per bucket and replayed.
    Captured,
}

/// The mode the process runs: `MUMMU_GRAPH=off` (or `0`/`false`) for the
/// ordinary path, `static` for the uncaptured static step, anything else —
/// unset included — captured. Read once.
#[must_use]
pub fn step_mode() -> StepMode {
    static MODE: std::sync::OnceLock<StepMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("MUMMU_GRAPH").ok().as_deref() {
        Some("0" | "off" | "false" | "dynamic") => StepMode::Dynamic,
        Some("static") => StepMode::Static,
        _ => StepMode::Captured,
    })
}

/// Run `fut` to completion without leaving this thread (see the module
/// docs).
///
/// On a multi-thread tokio runtime the worker hands its other tasks
/// to the pool for the duration and polls `fut` itself (inside
/// `block_in_place` tokio's cooperative `yield_now` wakes at once, so
/// `pollster` can drive the decode loop); anywhere else — a current-thread
/// runtime, `pollster` — a future never changes thread, and is awaited.
pub async fn on_this_thread<T>(fut: impl Future<Output = T>) -> T {
    match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => {
            tokio::task::block_in_place(|| pollster::block_on(fut))
        }
        _ => fut.await,
    }
}

/// Rewrite a whole input buffer in place.
///
/// In place is load-bearing: the graph reads the buffer it recorded, so a
/// write that copied would leave the graph reading the stale value — a
/// replay that decodes the previous token forever, silently. A shared buffer
/// cannot be written in place, so that is a loud failure here, not a quiet
/// one later.
pub(crate) fn write_all<const D: usize>(
    slot: &Mutex<Tensor<D, Int>>,
    values: Vec<i32>,
    shape: [usize; D],
    device: &Device,
) {
    let mut t = lock(slot);
    assert!(
        t.can_mut(),
        "a captured decode input is shared, so it cannot be rewritten in place"
    );
    assert_eq!(t.dims(), shape, "input buffer shape");
    let new = Tensor::<D, Int>::from_data(
        TensorData::new(values, shape),
        (device, crate::backend::int_dtype(device)),
    );
    let at: [std::ops::Range<usize>; D] = core::array::from_fn(|d| 0..shape[d]);
    t.inplace(|x| x.slice_assign(at, new));
}

pub(crate) fn index(v: impl TryInto<i32>) -> i32 {
    v.try_into()
        .unwrap_or_else(|_| panic!("token ids and positions fit i32"))
}

/// `Send` so that a generation's future is: the engine awaits it on a
/// current-thread runtime, where it cannot move (see [`on_this_thread`]).
pub(crate) type Step<'a> = Box<dyn FnMut() -> Tensor<2> + Send + 'a>;
type Graphs<'a> = HashMap<usize, burn::tensor::Graph<Tensor<2>, Step<'a>>>;

/// How many times a step runs uncaptured before it is recorded: the first
/// run compiles, autotunes and builds the fused kernels, the second takes the
/// cached fused path.
const PREWARM: usize = 2;

/// Record `step`, after running it uncaptured until its kernels settle.
///
/// Inside a capture window every allocation is an exact-fit slice of the
/// persistent pool, held until an explicit cleanup. The first run of a new
/// shape allocates for things the steady state never needs again (kernel
/// compilation, autotune trials, building a fused kernel); settled first,
/// the window holds only the step's own working set.
fn capture_step<'a>(
    device: &Device,
    mut step: Step<'a>,
) -> burn::tensor::Graph<Tensor<2>, Step<'a>> {
    for _ in 0..PREWARM {
        // Held across the sync: a lazy backend skips work whose result is
        // dropped before it drains.
        let out = step();
        let _ = Device::sync(device);
        drop(out);
    }
    burn::tensor::capture(device, step)
}

/// [`capture_step`] with the state rewound to before it (see
/// [`StaticState`]): the step runs several times while capturing, and the
/// replay that follows must start from the state the driver left.
pub(crate) fn capture_rewound<'a, S: StaticState>(
    state: &Mutex<S>,
    device: &Device,
    step: Step<'a>,
) -> burn::tensor::Graph<Tensor<2>, Step<'a>> {
    let mark = lock(state).mark();
    let graph = capture_step(device, step);
    lock(state).rewind(mark);
    // The rewind is queued work; the replay is not ordered after it.
    let _ = Device::sync(device);
    graph
}

/// Drop every graph and hand their memory back to the device.
///
/// A graph's buffers are exact-fit slices of the persistent pool; dropped,
/// they are free but stay reserved, reusable only by an allocation of the
/// same size — another graph of the same batch shape and bucket. A process
/// that sees many shapes would hold the sum of every shape's working set
/// (measured: Qwen3-0.6B at 1, 2, 4, 8 and then 16 slots ran the 16 GB card
/// out of memory). An explicit cleanup returns the free slices.
fn drop_graphs(graphs: &mut Graphs<'_>, device: &Device) {
    if graphs.is_empty() {
        return;
    }
    graphs.clear();
    crate::backend::return_memory(device);
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Positions a static state holds past the prompt at first. A generation
/// that outgrows them doubles the state (and recaptures): memory follows
/// what a request uses, as the dynamic cache's does, not what it may use —
/// a `max_tokens` of 32k would otherwise reserve its whole cache up front.
const INITIAL_ROOM: usize = 512;

/// The context a static state may grow to for `total` positions, or `None`
/// when that is past the model's trained context or the cache's limit.
fn ceiling<M: StaticDecode>(model: &M, total: usize) -> Option<usize> {
    let fits = model.trained_context().is_none_or(|t| total <= t);
    let rounded = total.div_ceil(BUCKET).saturating_mul(BUCKET);
    let ceiling = model.trained_context().map_or(rounded, |t| rounded.min(t));
    (fits && ceiling <= MAX_CONTEXT_TOKENS).then_some(ceiling)
}

/// The first size of a static state that must hold `start` positions.
pub(crate) fn initial_ctx(start: usize, ceiling: usize) -> usize {
    start
        .saturating_add(INITIAL_ROOM)
        .div_ceil(BUCKET)
        .saturating_mul(BUCKET)
        .min(ceiling)
}

/// Make sure position `pos` fits: double the state until it does,
/// dropping every graph over the old buffers first.
fn ensure_room<S: StaticState>(
    state: &Mutex<S>,
    graphs: &mut Graphs<'_>,
    max_ctx: &mut usize,
    pos: usize,
    ceiling: usize,
    device: &Device,
) {
    if pos < *max_ctx {
        return;
    }
    drop_graphs(graphs, device);
    while pos >= *max_ctx {
        *max_ctx = max_ctx.saturating_mul(2).min(ceiling);
    }
    lock(state).grow(*max_ctx);
}

/// One generation's parameters for [`generate`].
#[derive(Clone, Copy)]
pub struct DecodeRequest<'a> {
    pub prompt_ids: &'a [u32],
    pub max_tokens: usize,
    pub opts: &'a SamplerOptions,
    pub device: &'a Device,
    /// How the decode steps run (see [`step_mode`]).
    pub mode: StepMode,
}

/// Whether [`generate`] takes the static path for `req`, else the ordinary
/// one.
///
/// It needs a static mode, a model that supports it on the device, and a
/// request within the model's trained context. Allocates nothing — a caller that must arrange something for the static path (one
/// thread, see [`on_this_thread`]) asks first.
#[must_use]
pub fn applies<M: StaticDecode>(model: &M, req: &DecodeRequest<'_>) -> bool {
    req.mode != StepMode::Dynamic
        && ceiling(model, req.prompt_ids.len().saturating_add(req.max_tokens)).is_some()
        && model.static_capturable(req.device)
}

/// Generate one sequence with the decode steps on the static path,
/// recorded and replayed when the mode is [`StepMode::Captured`].
///
/// Same contract as [`crate::models::generate_constrained`] (greedy: the
/// same tokens in f32; in f16 a different reduction order can tip a
/// near-tie). Falls back to the ordinary path when the model cannot take the static
/// path as placed or the request would not fit its trained context.
///
/// # Errors
///
/// Whatever the decode driver reports (a failed readback).
///
/// # Panics
///
/// See the module docs: polled on more than one thread, a replay can race
/// its input refresh; that is not detected. Also when a captured input turns
/// out shared (see `write_all`).
pub async fn generate<M: StaticDecode + Sync>(
    model: &M,
    req: DecodeRequest<'_>,
    on_token: impl FnMut(u32) -> ControlFlow<()>,
    constraint: Option<&mut dyn Constraint>,
) -> Result<Generated, String> {
    let DecodeRequest {
        prompt_ids,
        max_tokens,
        opts,
        device,
        mode,
    } = req;
    let prompt_len = prompt_ids.len();
    let ceiling = ceiling(model, prompt_len.saturating_add(max_tokens));
    let (true, Some(ceiling)) = (applies(model, &req), ceiling) else {
        return crate::models::generate_constrained(
            model, prompt_ids, max_tokens, opts, device, on_token, constraint,
        )
        .await;
    };
    let mut max_ctx = initial_ctx(prompt_len, ceiling);
    let state = model.static_state(1, max_ctx, device);
    let int = crate::backend::int_dtype(device);
    let kv = Mutex::new(state);
    let tokens = Mutex::new(Tensor::<2, Int>::zeros([1, 1], (device, int)));
    let positions = Mutex::new(Tensor::<1, Int>::zeros([1], (device, int)));
    let mut dynamic = Some(model.new_cache());
    let mut graphs: Graphs<'_> = HashMap::new();
    let step = |ids: &[u32], past: usize, need_logits: bool| -> Option<Tensor<2>> {
        if past < prompt_len {
            let cache = dynamic
                .as_mut()
                .expect("prefill runs before the first decode step");
            if need_logits {
                return Some(model.forward(ids, past, cache, device));
            }
            model.forward_advance(ids, past, cache, device);
            return None;
        }
        if let Some(cache) = dynamic.take() {
            model.seed_static(&mut lock(&kv), 0, &cache);
        }
        debug_assert_eq!(ids.len(), 1, "decode steps are one token");
        ensure_room(&kv, &mut graphs, &mut max_ctx, past, ceiling, device);
        write_all(&tokens, vec![index(ids[0])], [1, 1], device);
        write_all(&positions, vec![index(past)], [1], device);
        // The refreshes are queued writes (fusion is lazy); the replay below
        // is a direct dispatch that does not wait for the queue.
        let _ = Device::sync(device);
        let len = bucket(past, max_ctx);
        if mode == StepMode::Static {
            return Some(model.forward_static(
                &lock(&tokens),
                &lock(&positions),
                &mut lock(&kv),
                len,
            ));
        }
        // One graph at a time: the bucket only grows within a generation, and
        // every graph held keeps its whole working set on the card.
        if !graphs.contains_key(&len) {
            drop_graphs(&mut graphs, device);
        }
        let graph = graphs.entry(len).or_insert_with(|| {
            let (kv, tokens, positions) = (&kv, &tokens, &positions);
            let record: Step<'_> = Box::new(move || {
                model.forward_static(&lock(tokens), &lock(positions), &mut lock(kv), len)
            });
            let started = std::time::Instant::now();
            let graph = capture_rewound(kv, device, record);
            eprintln!(
                "[mummu] decode step captured for a {len}-key bucket in {:.0} ms ({})",
                started.elapsed().as_secs_f64() * 1e3,
                if graph.is_hardware() {
                    "replayed as a graph"
                } else {
                    "this backend has no graphs: replay re-runs the step"
                }
            );
            graph
        });
        // SAFETY: every buffer the graph touched is alive for its whole life
        // (the state, the two input buffers and the model outlive `graphs`,
        // and `ensure_room` drops every graph before it replaces a buffer);
        // nothing else uses them concurrently (one sequence, one thread); and
        // the refreshes above were issued — and synced — on this thread's
        // stream before the replay. See `burn::tensor::Graph::replay`.
        let out = unsafe { graph.replay() };
        Some(out.clone())
    };
    let out = generate_loop(
        step,
        prompt_ids,
        max_tokens,
        opts,
        |id| model.is_eos(id),
        on_token,
        constraint,
    )
    .await;
    drop_graphs(&mut graphs, device);
    out
}

/// Greedy-decode several prompts together: one slot each, every decode step
/// one dispatch for all of them — a [`crate::batch::Batcher`] that admits
/// them all and steps until each has stopped.
///
/// Each slot decodes what it would alone: the batch shares weight reads and
/// dispatches, not state — the gate in `tests/real_capture.rs` (exact in
/// f32).
///
/// # Errors
///
/// A failed readback, a request past the model's trained context, the
/// dynamic mode, or a model that cannot take the static path as placed (see
/// [`StaticDecode::static_supported`]).
///
/// # Panics
///
/// When `prompts` is empty or a prompt is empty; the single-thread contract
/// of the module docs applies.
pub fn generate_batch_greedy<M: StaticDecode + Sync + 'static>(
    model: &M,
    prompts: &[&[u32]],
    max_tokens: usize,
    device: &Device,
    mode: StepMode,
) -> Result<Vec<Vec<u32>>, String> {
    use crate::batch::{Admission, Batcher, Event};
    assert!(!prompts.is_empty(), "generate_batch_greedy: no prompts");
    let mut batch = Batcher::new(model, device, prompts.len(), mode)
        .ok_or_else(|| "this model cannot take the static decode path as placed".to_owned())?;
    let greedy = SamplerOptions::greedy();
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let mut ids = Vec::with_capacity(prompts.len());
    let take = |out: &mut Vec<Vec<u32>>, ids: &[crate::batch::SeqId], id, event| {
        if let (Some(i), Event::Token(t)) = (ids.iter().position(|&s| s == id), event) {
            out[i].push(t);
        }
    };
    for prompt in prompts {
        if !batch.fits(prompt.len(), max_tokens) {
            return Err("the batch would not fit the model's context".to_owned());
        }
        let (id, events) = batch.admit(
            model,
            Admission {
                prompt_ids: prompt,
                max_tokens,
                opts: &greedy,
                constraint: None,
            },
        )?;
        ids.push(id);
        for e in events {
            take(&mut out, &ids, id, e);
        }
    }
    while batch.active() > 0 {
        for (id, e) in batch.step(model)? {
            take(&mut out, &ids, id, e);
        }
    }
    batch.release();
    Ok(out)
}

/// [`generate`] greedy to completion on this thread, for tests.
#[cfg(test)]
pub(crate) fn greedy_decode<M: StaticDecode + Sync>(
    model: &M,
    prompt_ids: &[u32],
    max_tokens: usize,
    device: &Device,
    mode: StepMode,
) -> Vec<u32> {
    let opts = SamplerOptions::greedy();
    let req = DecodeRequest {
        prompt_ids,
        max_tokens,
        opts: &opts,
        device,
        mode,
    };
    pollster::block_on(generate(model, req, |_| ControlFlow::Continue(()), None))
        .expect("decodes")
        .ids
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three modes decode the same tokens on the toy Qwen3 (CPU, where
    /// capture falls back to re-running the closure — the static step and
    /// the driver are what is under test here; the GPU legs are in
    /// `tests/real_capture.rs`).
    #[test]
    fn every_mode_decodes_the_same_tokens() {
        let device = crate::backend::cpu_device();
        let model = crate::models::qwen3::toy_for_tests(&device);
        let prompt = [3u32, 14, 15, 9, 26];
        let runs: Vec<Vec<u32>> = [StepMode::Dynamic, StepMode::Static, StepMode::Captured]
            .into_iter()
            .map(|mode| greedy_decode(&model, &prompt, 12, &device, mode))
            .collect();
        assert_eq!(runs[0].len(), 12);
        assert_eq!(runs[0], runs[1], "static step");
        assert_eq!(runs[0], runs[2], "captured step");
    }

    /// Each slot of a batch decodes what it decodes alone — different
    /// prompts, different lengths, one dispatch per step.
    #[test]
    fn a_batch_decodes_each_slot_as_it_would_alone() {
        let device = crate::backend::cpu_device();
        let model = crate::models::qwen3::toy_for_tests(&device);
        let prompts: [&[u32]; 3] = [&[3, 14, 15], &[9, 26, 5, 35, 8, 9, 7], &[1]];
        for mode in [StepMode::Static, StepMode::Captured] {
            let batch = generate_batch_greedy(&model, &prompts, 10, &device, mode).unwrap();
            for (p, got) in prompts.iter().zip(&batch) {
                let alone = greedy_decode(&model, p, 10, &device, StepMode::Dynamic);
                assert_eq!(got, &alone, "{mode:?}: slot for prompt {p:?}");
            }
        }
    }

    /// A generation that crosses a bucket boundary (256 keys) keeps decoding
    /// the same tokens: the step re-captures at the larger bucket.
    #[test]
    fn crossing_a_bucket_boundary_changes_nothing() {
        let device = crate::backend::cpu_device();
        let model = crate::models::qwen3::toy_for_tests(&device);
        let prompt: Vec<u32> = (0..250).map(|i| (i * 7 + 3) % 64).collect();
        let dynamic = greedy_decode(&model, &prompt, 12, &device, StepMode::Dynamic);
        let captured = greedy_decode(&model, &prompt, 12, &device, StepMode::Captured);
        assert_eq!(dynamic, captured);
    }
}
