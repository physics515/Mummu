//! Continuous batching over the captured decode step.
//!
//! A [`Batcher`] holds one static state ([`crate::capture::StaticDecode`])
//! with room for several sequences, and steps every live one in a single
//! dispatch. Sequences come and go between steps: [`Batcher::admit`]
//! prefills a prompt on the ordinary path and seeds it into the next free
//! slot, [`Batcher::step`] advances them all and reports each one's token,
//! and a sequence that stops (EOS, its token budget, a closed constraint)
//! leaves its slot — the last live sequence moves into it, so the live ones
//! always occupy the first slots and a step runs only those (rounded up to a
//! power of two: one graph per tier, not per count).
//!
//! The graphs outlive any one call: the batcher keeps every graph it
//! captured, per (slot tier, length bucket), for as long as the buffers they
//! recorded exist — across requests, which is the point (a capture costs
//! several runs of the step). What can invalidate them is the caller's to
//! say: [`Batcher::set_epoch`] drops them all when the model's weights moved,
//! and [`Batcher::release`] drops everything.
//!
//! # The model
//!
//! A graph's closure is run again only while it is being captured, or on a
//! backend without hardware graphs, where a replay re-runs it. Either way
//! that happens inside a batcher call, which borrows the model; the closures
//! reach it through a cell that holds the pointer for exactly that call and
//! is empty otherwise, so a graph kept between requests holds no reference
//! to a model that may since have moved. One thread, as for
//! [`crate::capture`]: every call on a batcher from the thread that made it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

use burn::tensor::{Device, Int, Tensor};

use crate::capture::{
    StaticDecode, StaticState, Step, StepMode, capture_rewound, index, initial_ctx, lock, write_all,
};
use crate::constrain::Constraint;
pub use crate::decode::Finish;
use crate::decode::{Picker, SamplerOptions, prefill_chunk_len};
use crate::nn::MAX_CONTEXT_TOKENS;
use crate::nn::static_kv::bucket;

/// The model a batcher call is running on, for the graphs' closures: set for
/// the duration of the call (see the module docs), empty otherwise.
struct ModelCell<M> {
    ptr: AtomicPtr<M>,
}

impl<M> ModelCell<M> {
    const fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// The model of the call in progress.
    ///
    /// # Panics
    ///
    /// Outside a batcher call — a graph's closure run with no model attached.
    fn get(&self) -> &M {
        let p = self.ptr.load(Ordering::Acquire);
        assert!(!p.is_null(), "a batched step ran with no model attached");
        // SAFETY: the pointer is only non-null between `Attached::new` and
        // its drop, which bracket a batcher call holding `&M` for longer
        // than any use made here (the closures run synchronously inside
        // that call). `M: Sync` makes the shared access sound.
        unsafe { &*p }
    }
}

/// Holds `m` in the cell until dropped (a panic included).
struct Attached<'a, M>(&'a ModelCell<M>);

impl<'a, M> Attached<'a, M> {
    fn new(cell: &'a ModelCell<M>, m: &M) -> Self {
        cell.ptr
            .store(std::ptr::from_ref(m).cast_mut(), Ordering::Release);
        Self(cell)
    }
}

impl<M> Drop for Attached<'_, M> {
    fn drop(&mut self) {
        self.0.ptr.store(std::ptr::null_mut(), Ordering::Release);
    }
}

/// A sequence's handle, stable while it lives (its slot is not).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(u64);

/// What a sequence did on a step (or on admission).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// It emitted this token.
    Token(u32),
    /// It stopped, and has left the batch.
    Done(Finish),
}

/// One sequence to admit.
pub struct Admission<'a> {
    pub prompt_ids: &'a [u32],
    pub max_tokens: usize,
    pub opts: &'a SamplerOptions,
    /// Decides which ids may be emitted, and when the value is complete
    /// (see [`crate::constrain`]).
    pub constraint: Option<Box<dyn Constraint>>,
}

struct Seq {
    id: SeqId,
    picker: Picker,
    constraint: Option<Box<dyn Constraint>>,
    /// The token the next step feeds, at position `pos`.
    next: u32,
    pos: usize,
    /// Positions it can reach: its prompt plus its `max_tokens`.
    limit: usize,
    emitted: usize,
    max_tokens: usize,
}

/// The step's two inputs for one slot tier, rewritten in place.
struct Inputs {
    tokens: Arc<Mutex<Tensor<2, Int>>>,
    positions: Arc<Mutex<Tensor<1, Int>>>,
}

/// The longest prompt + generation `model` takes on the static path.
#[must_use]
pub fn context<M: StaticDecode>(model: &M) -> usize {
    model
        .trained_context()
        .unwrap_or(MAX_CONTEXT_TOKENS)
        .min(MAX_CONTEXT_TOKENS)
}

/// Whether a request of `prompt_len` tokens and `max_tokens` fits
/// [`context`].
#[must_use]
pub fn fits<M: StaticDecode>(model: &M, prompt_len: usize, max_tokens: usize) -> bool {
    prompt_len.saturating_add(max_tokens) <= context(model)
}

/// A step's events, and the slots whose sequences stopped.
type Picked = (Vec<(SeqId, Event)>, Vec<usize>);

/// See the module docs.
pub struct Batcher<M: StaticDecode> {
    device: Device,
    mode: StepMode,
    max_slots: usize,
    /// The longest context the model takes (positions).
    ceiling: usize,
    cell: Arc<ModelCell<M>>,
    epoch: u64,
    state: Option<Arc<Mutex<M::State>>>,
    inputs: HashMap<usize, Inputs>,
    /// Per (slot tier, length bucket) — at most one per tier, for tier 1 and
    /// the tier in use (see [`Batcher::forward`]).
    graphs: HashMap<(usize, usize), burn::tensor::Graph<Tensor<2>, Step<'static>>>,
    /// The most card memory a captured graph has held per slot of its tier:
    /// what a wider tier's graph is charged before it exists (see
    /// [`Batcher::charge`]). 0 until the first capture.
    graph_per_slot: u64,
    /// Live sequences; index = slot.
    seqs: Vec<Seq>,
    next_id: u64,
}

impl<M: StaticDecode + Sync + 'static> Batcher<M> {
    /// A batcher for up to `max_slots` sequences of `model` on `device`, or
    /// `None` when the model cannot take the static path there or `mode` is
    /// [`StepMode::Dynamic`]. A model whose step cannot be captured there (a
    /// split model, see [`StaticDecode::static_capturable`]) steps op by op
    /// whatever `mode` asks.
    #[must_use]
    pub fn new(model: &M, device: &Device, max_slots: usize, mode: StepMode) -> Option<Self> {
        if mode == StepMode::Dynamic || !model.static_supported(device) {
            return None;
        }
        let mode = if model.static_capturable(device) {
            mode
        } else {
            StepMode::Static
        };
        let ceiling = context(model);
        Some(Self {
            device: device.clone(),
            mode,
            max_slots: max_slots.max(1),
            ceiling,
            cell: Arc::new(ModelCell::new()),
            epoch: 0,
            state: None,
            inputs: HashMap::new(),
            graphs: HashMap::new(),
            graph_per_slot: 0,
            seqs: Vec::new(),
            next_id: 0,
        })
    }

    /// Live sequences.
    #[must_use]
    pub const fn active(&self) -> usize {
        self.seqs.len()
    }

    /// Whether another sequence can be admitted.
    #[must_use]
    pub const fn has_room(&self) -> bool {
        self.seqs.len() < self.max_slots
    }

    /// Whether a request of `prompt_len + max_tokens` positions fits the
    /// model's context (else decode it on the ordinary path).
    #[must_use]
    pub const fn fits(&self, prompt_len: usize, max_tokens: usize) -> bool {
        prompt_len.saturating_add(max_tokens) <= self.ceiling
    }

    /// The model's weights are at `epoch` (the caller counts every move):
    /// graphs recorded against other weights are dropped.
    pub fn set_epoch(&mut self, epoch: u64) {
        if epoch != self.epoch {
            self.drop_graphs();
            self.epoch = epoch;
        }
    }

    /// Drop every graph, the state and the sequences in it, and hand the
    /// memory back to the device.
    pub fn release(&mut self) {
        self.seqs.clear();
        self.graphs.clear();
        self.inputs.clear();
        self.state = None;
        crate::backend::return_memory(&self.device);
    }

    fn drop_graphs(&mut self) {
        if !self.graphs.is_empty() {
            self.graphs.clear();
            crate::backend::return_memory(&self.device);
        }
    }

    /// Stop sequence `id` now (its request went away). Unknown ids are
    /// ignored.
    pub fn cancel(&mut self, id: SeqId) {
        if let Some(slot) = self.seqs.iter().position(|s| s.id == id) {
            self.retire(slot);
        }
    }

    /// Admit a sequence: prefill its prompt on the ordinary path, seed it
    /// into the next slot and pick its first token. The events are that
    /// token and/or why it stopped at once (it is then not in the batch).
    ///
    /// # Errors
    ///
    /// A failed readback, a constraint with no legal token, a full batch or
    /// a request past the model's context.
    ///
    /// # Panics
    ///
    /// When the prompt is empty, `max_tokens` is 0, or the options are
    /// invalid (see [`crate::decode::sample_id_filtered`]).
    pub fn admit(&mut self, model: &M, adm: Admission<'_>) -> Result<(SeqId, Vec<Event>), String> {
        let Admission {
            prompt_ids,
            max_tokens,
            opts,
            mut constraint,
        } = adm;
        assert!(!prompt_ids.is_empty(), "admit: empty prompt");
        assert!(max_tokens >= 1, "admit: max_tokens must be >= 1");
        if !self.has_room() {
            return Err("the batch is full".to_owned());
        }
        if !self.fits(prompt_ids.len(), max_tokens) {
            return Err("the request is longer than the model's context".to_owned());
        }
        let cell = Arc::clone(&self.cell);
        let _attached = Attached::new(&cell, model);
        let prompt_len = prompt_ids.len();
        let slot = self.seqs.len();
        self.make_room(model, slot + 1, prompt_len);
        // The prompt, on the ordinary path, in the same chunks as
        // `generate_loop`'s prefill.
        let mut cache = model.new_cache();
        let chunk = prefill_chunk_len();
        let mut done = 0usize;
        let mut logits = None;
        while done < prompt_len {
            let end = done.saturating_add(chunk).min(prompt_len);
            if end == prompt_len {
                logits =
                    Some(model.forward(&prompt_ids[done..end], done, &mut cache, &self.device));
            } else {
                model.forward_advance(&prompt_ids[done..end], done, &mut cache, &self.device);
            }
            done = end;
        }
        let logits = logits.expect("a non-empty prompt");
        // The prompt's working set peaks now; it may be handed back before
        // the request ends.
        crate::backend::note_pool_peak(&self.device);
        let state = Arc::clone(self.state.as_ref().expect("make_room built the state"));
        model.seed_static(&mut lock(&state), slot, &cache);
        drop(cache);
        let mut picker = Picker::new(opts);
        let id = pollster::block_on(picker.pick(
            logits,
            None,
            constraint.as_deref(),
            prompt_len,
            |t| model.is_eos(t),
        ))?;
        let seq_id = SeqId(self.next_id);
        self.next_id += 1;
        let (token, stop) = settle(
            model,
            id,
            &mut constraint,
            1,
            max_tokens,
            prompt_len,
            self.ceiling,
        );
        let events: Vec<Event> = token
            .map(Event::Token)
            .into_iter()
            .chain(stop.map(Event::Done))
            .collect();
        if stop.is_none() {
            self.seqs.push(Seq {
                id: seq_id,
                picker,
                constraint,
                next: id,
                pos: prompt_len,
                limit: prompt_len.saturating_add(max_tokens),
                emitted: 1,
                max_tokens,
            });
        }
        Ok((seq_id, events))
    }

    /// One step for every live sequence: each one's events (a token, a
    /// stop, or a token and then a stop), in slot order. Sequences that stop
    /// leave the batch.
    ///
    /// # Errors
    ///
    /// A failed readback, an id outside the vocabulary (NaN logits), or a
    /// constraint with no legal token — the batch is then in an unknown
    /// state and should be released.
    pub fn step(&mut self, model: &M) -> Result<Vec<(SeqId, Event)>, String> {
        if self.seqs.is_empty() {
            return Ok(Vec::new());
        }
        let cell = Arc::clone(&self.cell);
        let _attached = Attached::new(&cell, model);
        let logits = self.forward(model);
        let (events, finished) = self.pick_all(model, &logits)?;
        // Highest slot first, so each retirement leaves the lower ones put.
        for slot in finished.into_iter().rev() {
            self.retire(slot);
        }
        Ok(events)
    }

    /// The step itself over the live sequences' tier: inputs rewritten in
    /// place, then the tier's graph for this bucket replayed (captured the
    /// first time) — or run op by op in [`StepMode::Static`]. `[tier, vocab]`
    /// logits; rows past the live sequences are idle slots.
    fn forward(&mut self, model: &M) -> Tensor<2> {
        let live = self.seqs.len();
        let furthest = self.seqs.iter().map(|s| s.pos).max().unwrap_or(0);
        self.make_room(model, live, furthest);
        let state = Arc::clone(self.state.as_ref().expect("make_room built the state"));
        let (slots, max_ctx) = lock(&state).shape();
        let tier = self.tier_for(live, slots);
        let int = crate::backend::int_dtype(&self.device);
        let inputs = self.inputs.entry(tier).or_insert_with(|| Inputs {
            tokens: Arc::new(Mutex::new(Tensor::<2, Int>::zeros(
                [tier, 1],
                (&self.device, int),
            ))),
            positions: Arc::new(Mutex::new(Tensor::<1, Int>::zeros(
                [tier],
                (&self.device, int),
            ))),
        });
        let (tokens, positions) = (Arc::clone(&inputs.tokens), Arc::clone(&inputs.positions));
        // Idle slots of the tier run at position 0 on token 0: their own rows
        // only, overwritten by the next sequence seeded there.
        let fill = |f: fn(&Seq) -> usize| -> Vec<i32> {
            (0..tier)
                .map(|i| self.seqs.get(i).map_or(0, |s| index(f(s))))
                .collect()
        };
        write_all(&tokens, fill(|s| s.next as usize), [tier, 1], &self.device);
        write_all(&positions, fill(|s| s.pos), [tier], &self.device);
        let _ = Device::sync(&self.device);
        let len = bucket(furthest, max_ctx);
        // A sequence alone decodes as the ordinary path does, a bounded host
        // head included (`flex::head`): told the k its pick consults. Not
        // under a constraint, whose legal tokens may lie anywhere, and only
        // for a model whose step reads one — the scope is process-wide.
        let _head = match self.seqs.as_slice() {
            [only]
                if model.static_bounded_head()
                    && only.constraint.is_none()
                    && only.picker.consults() >= 1 =>
            {
                Some(crate::flex::head::RequestTopK::set(only.picker.consults()))
            }
            _ => None,
        };
        if self.mode != StepMode::Captured {
            return model.forward_static(&lock(&tokens), &lock(&positions), &mut lock(&state), len);
        }
        // Graphs held: one per tier, for tier 1 — a request alone, the common
        // case — and one wider tier, the last one used. Every graph keeps its
        // whole working set on the card (on the 2B ~210 MiB per slot of its
        // tier), so keeping each tier's and each bucket's was the batch's
        // largest cost: 3.2 GB of graphs for one 8-slot burst, which ran the
        // card out of memory. A new graph replaces its tier's other bucket,
        // and a wider tier replaces the wider tier held.
        if !self.graphs.contains_key(&(tier, len)) {
            let before = self.graphs.len();
            self.graphs
                .retain(|&(t, _), _| t != tier && (tier == 1 || t == 1));
            if self.graphs.len() != before {
                crate::backend::return_memory(&self.device);
            }
        }
        let graph = self.graphs.entry((tier, len)).or_insert_with(|| {
            let (cell, st, tk, ps) = (
                Arc::clone(&self.cell),
                Arc::clone(&state),
                Arc::clone(&tokens),
                Arc::clone(&positions),
            );
            let record: Step<'static> = Box::new(move || {
                cell.get().forward_static(&lock(&tk), &lock(&ps), &mut lock(&st), len)
            });
            let started = std::time::Instant::now();
            let reserved = || self.device.memory_pool_usage().map_or(0, |u| u.bytes_reserved);
            let before = reserved();
            let graph = capture_rewound(&state, &self.device, record);
            let held = reserved().saturating_sub(before);
            self.graph_per_slot = self.graph_per_slot.max(held / tier as u64);
            eprintln!(
                "[mummu] batched decode step captured for {tier} slot(s), a {len}-key bucket, in {:.0} ms ({} MiB held)",
                started.elapsed().as_secs_f64() * 1e3,
                held >> 20
            );
            graph
        });
        // SAFETY: every buffer the graph touched — the state, this tier's
        // inputs, the model's weights — is alive: the state and inputs are
        // only replaced after `drop_graphs`, and weights that moved were
        // announced through `set_epoch`, which dropped the graphs. One
        // thread, and the inputs were synced above.
        unsafe { graph.replay() }.clone()
    }

    /// The slot tier a step over `live` sequences runs: the smallest power of
    /// two holding them — or, with more than one live, a wider tier whose
    /// graph is held. A batch shrinking as its sequences finish keeps its
    /// graph rather than recapturing at every halving: on the 2B an 8-slot
    /// step costs a few milliseconds more than a 2-slot one, and a capture
    /// 0.5-0.9 s of every sequence's time.
    fn tier_for(&self, live: usize, slots: usize) -> usize {
        let needed = live.next_power_of_two().min(slots);
        if needed == 1 {
            return 1;
        }
        self.graphs
            .keys()
            .map(|&(t, _)| t)
            .filter(|&t| t >= needed && t <= slots)
            .min()
            .unwrap_or(needed)
    }

    /// Each live sequence's pick from its row of `logits`, applied: the
    /// events, and the slots that stopped.
    fn pick_all(&mut self, model: &M, logits: &Tensor<2>) -> Result<Picked, String> {
        // One readback for every greedy row's argmax.
        let argmax: Option<Vec<i64>> = if self.seqs.iter().any(|s| s.picker.greedy()) {
            Some(
                logits
                    .clone()
                    .argmax(1)
                    .into_data()
                    .convert::<i64>()
                    .try_to_vec::<i64>()
                    .map_err(|e| format!("argmax readback: {e:?}"))?,
            )
        } else {
            None
        };
        let vocab = logits.dims()[1];
        let mut events = Vec::with_capacity(self.seqs.len());
        let mut finished = Vec::new();
        for (slot, seq) in self.seqs.iter_mut().enumerate() {
            let hint = argmax
                .as_ref()
                .and_then(|a| u32::try_from(a[slot]).ok())
                .filter(|_| seq.picker.greedy());
            let row = logits.clone().narrow(0, slot, 1);
            let past = seq.pos + 1;
            let id = pollster::block_on(seq.picker.pick(
                row,
                hint,
                seq.constraint.as_deref(),
                past,
                |t| model.is_eos(t),
            ))?;
            if id as usize >= vocab {
                return Err(format!(
                    "decode step {past}: id {id} is outside the {vocab}-token vocab — NaN logits / numeric collapse on this backend?"
                ));
            }
            let (token, stop) = settle(
                model,
                id,
                &mut seq.constraint,
                seq.emitted + 1,
                seq.max_tokens,
                past,
                self.ceiling,
            );
            if let Some(t) = token {
                seq.next = t;
                seq.pos = past;
                seq.emitted += 1;
                events.push((seq.id, Event::Token(t)));
            }
            if let Some(why) = stop {
                finished.push(slot);
                events.push((seq.id, Event::Done(why)));
            }
        }
        Ok((events, finished))
    }

    /// Ensure a state with at least `slots` slots and room past position
    /// `pos`, keeping what it holds. New buffers drop the graphs.
    fn make_room(&mut self, model: &M, slots: usize, pos: usize) {
        let want = self.shape_for(slots, pos);
        let Some(state) = &self.state else {
            self.state = Some(Arc::new(Mutex::new(model.static_state(
                want.0,
                want.1,
                &self.device,
            ))));
            return;
        };
        if want != lock(state).shape() {
            let state = Arc::clone(state);
            self.drop_graphs();
            lock(&state).resize(want.0, want.1);
        }
    }

    /// The state shape that holds `slots` sequences and position `pos`,
    /// grown from the current one the way [`Self::make_room`] grows it —
    /// doubling, so what [`Self::bytes_with`] charges is what it allocates.
    fn shape_for(&self, slots: usize, pos: usize) -> (usize, usize) {
        let Some(state) = &self.state else {
            let want = slots.next_power_of_two().min(self.max_slots).max(slots);
            return (want, initial_ctx(pos + 1, self.ceiling));
        };
        let (have, mut max_ctx) = lock(state).shape();
        let mut want = have;
        while want < slots {
            want = (want * 2).min(self.max_slots).max(slots);
        }
        while pos >= max_ctx && max_ctx < self.ceiling {
            max_ctx = max_ctx.saturating_mul(2).min(self.ceiling);
        }
        (want, max_ctx)
    }

    /// Bytes the batch's state holds now on its device.
    #[must_use]
    pub fn bytes(&self, model: &M) -> u64 {
        self.bytes_on(model, &self.device)
    }

    fn bytes_on(&self, model: &M, device: &Device) -> u64 {
        self.state.as_ref().map_or(0, |s| {
            let (slots, ctx) = lock(s).shape();
            model.static_bytes(slots, ctx, device)
        })
    }

    /// What admitting a request of `prompt` tokens and `max_tokens` adds to
    /// the card at most: its state at full length, the live sequences' grown
    /// to theirs, and — when the batch would then need a wider slot tier
    /// than it has a graph for — that tier's graph, priced per slot from the
    /// graphs captured so far. `None` while that price is unknown (nothing
    /// captured yet); a caller then waits a step.
    #[must_use]
    pub fn charge(&self, model: &M, prompt: usize, max_tokens: usize) -> Option<u64> {
        let state = self
            .bytes_with(model, Some((prompt, max_tokens)))
            .saturating_sub(self.bytes(model));
        if self.mode != StepMode::Captured {
            return Some(state);
        }
        let tier = (self.seqs.len() + 1)
            .next_power_of_two()
            .min(self.max_slots);
        if self.graphs.keys().any(|&(t, _)| t >= tier) {
            return Some(state);
        }
        if self.graph_per_slot == 0 {
            return None;
        }
        Some(state.saturating_add(self.graph_per_slot.saturating_mul(tier as u64)))
    }

    /// [`Self::charge`]'s twin for the host: what admitting the request
    /// adds at most to the state of the model's layers on the host — a split
    /// model keeps their KV and recurrent state in host memory. Nothing for
    /// a model wholly on the batch's device (and a model on the host has
    /// all of it in [`Self::charge`]).
    #[must_use]
    pub fn charge_host(&self, model: &M, prompt: usize, max_tokens: usize) -> u64 {
        let host = crate::backend::cpu_device();
        if host == self.device {
            return 0;
        }
        self.bytes_with_on(model, Some((prompt, max_tokens)), &host)
            .saturating_sub(self.bytes_on(model, &host))
    }

    /// Bytes the batch's state grows to at most — every live sequence run to
    /// its last position, and, when `extra` is a request's prompt length and
    /// `max_tokens`, that one admitted too. What a caller that plans the
    /// card's memory charges a batch before it admits one more.
    #[must_use]
    pub fn bytes_with(&self, model: &M, extra: Option<(usize, usize)>) -> u64 {
        self.bytes_with_on(model, extra, &self.device)
    }

    fn bytes_with_on(&self, model: &M, extra: Option<(usize, usize)>, device: &Device) -> u64 {
        let slots = self.seqs.len() + usize::from(extra.is_some());
        if slots == 0 {
            return self.bytes_on(model, device);
        }
        let last = self
            .seqs
            .iter()
            .map(|s| s.limit)
            .chain(extra.map(|(prompt, max)| prompt.saturating_add(max)))
            .max()
            .unwrap_or(1)
            .min(self.ceiling)
            .saturating_sub(1);
        let (s, ctx) = self.shape_for(slots, last);
        model.static_bytes(s, ctx, device)
    }

    /// Remove the sequence in `slot`; the last live one moves into it.
    fn retire(&mut self, slot: usize) {
        let last = self.seqs.len() - 1;
        if slot != last
            && let Some(state) = &self.state
        {
            lock(state).move_slot(last, slot);
        }
        self.seqs.swap_remove(slot);
    }
}

/// What a picked `id` means for its sequence: whether it is emitted, and
/// whether the sequence stops — `generate_loop`'s rules, in its order. EOS
/// stops without emitting; otherwise the token is emitted (the constraint
/// accepts it) and the sequence stops if its value closed, it has emitted
/// its `max_tokens` (`emitted` counts this one), or position `next_pos`,
/// where this token would be fed, is past the model's context.
fn settle<M: StaticDecode>(
    model: &M,
    id: u32,
    constraint: &mut Option<Box<dyn Constraint>>,
    emitted: usize,
    max_tokens: usize,
    next_pos: usize,
    ceiling: usize,
) -> (Option<u32>, Option<Finish>) {
    if model.is_eos(id) {
        return (None, Some(Finish::Eos));
    }
    if let Some(c) = constraint.as_deref_mut() {
        c.accept(id);
    }
    let stop = if constraint.as_deref().is_some_and(Constraint::is_complete) {
        Some(Finish::Complete)
    } else if emitted >= max_tokens {
        Some(Finish::Length)
    } else if next_pos >= ceiling {
        Some(Finish::Context)
    } else {
        None
    };
    (Some(id), stop)
}

/// One sequence for [`run`]: admitted before step `at`, with its options,
/// cancelled before step `cancel_at` when set.
#[cfg(test)]
pub(crate) struct Planned<'a> {
    pub at: usize,
    pub prompt: &'a [u32],
    pub max_tokens: usize,
    pub opts: SamplerOptions,
    pub cancel_at: Option<usize>,
}

/// Drive a batcher through `plan` until every sequence stops: each one's
/// emitted tokens, in plan order.
#[cfg(test)]
pub(crate) fn run<M: StaticDecode + Sync + 'static>(
    model: &M,
    device: &Device,
    max_slots: usize,
    mode: StepMode,
    plan: &[Planned<'_>],
) -> Vec<Vec<u32>> {
    let mut batch = Batcher::new(model, device, max_slots, mode).expect("static path");
    let mut out = vec![Vec::new(); plan.len()];
    let mut ids: Vec<Option<SeqId>> = vec![None; plan.len()];
    let record = |out: &mut Vec<Vec<u32>>, ids: &[Option<SeqId>], id: SeqId, e: Event| {
        if let (Some(i), Event::Token(t)) = (ids.iter().position(|&s| s == Some(id)), e) {
            out[i].push(t);
        }
    };
    let mut step = 0;
    loop {
        for (i, p) in plan.iter().enumerate() {
            if p.at == step {
                let (id, events) = batch
                    .admit(
                        model,
                        Admission {
                            prompt_ids: p.prompt,
                            max_tokens: p.max_tokens,
                            opts: &p.opts,
                            constraint: None,
                        },
                    )
                    .expect("admits");
                ids[i] = Some(id);
                for e in events {
                    record(&mut out, &ids, id, e);
                }
            }
            if p.cancel_at == Some(step)
                && let Some(id) = ids[i]
            {
                batch.cancel(id);
            }
        }
        let pending = plan.iter().any(|p| p.at > step);
        if batch.active() == 0 && !pending {
            break;
        }
        for (id, e) in batch.step(model).expect("steps") {
            record(&mut out, &ids, id, e);
        }
        step += 1;
    }
    batch.release();
    out
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use super::*;
    use crate::capture::greedy_decode;
    use crate::models::CausalLm;

    fn planned(at: usize, prompt: &[u32], max_tokens: usize) -> Planned<'_> {
        Planned {
            at,
            prompt,
            max_tokens,
            opts: SamplerOptions::greedy(),
            cancel_at: None,
        }
    }

    /// Sequences that arrive while others are decoding, and stop at
    /// different lengths (so slots move under the ones still running),
    /// decode exactly what they decode alone — in both static modes.
    #[test]
    fn late_arrivals_decode_as_they_would_alone() {
        let device = crate::backend::cpu_device();
        let model = crate::models::qwen3::toy_for_tests(&device);
        let prompts: [&[u32]; 4] = [
            &[3, 14, 15],
            &[9, 26, 5, 35, 8],
            &[1],
            &[7, 7, 2, 40, 41, 42, 43],
        ];
        let plan = [
            planned(0, prompts[0], 14),
            planned(2, prompts[1], 4),
            planned(3, prompts[2], 9),
            planned(5, prompts[3], 6),
        ];
        for mode in [StepMode::Static, StepMode::Captured] {
            let got = run(&model, &device, 4, mode, &plan);
            for (p, g) in plan.iter().zip(&got) {
                let alone =
                    greedy_decode(&model, p.prompt, p.max_tokens, &device, StepMode::Dynamic);
                assert_eq!(g, &alone, "{mode:?}: prompt {:?}", p.prompt);
            }
        }
    }

    /// A sequence cancelled mid-decode leaves the batch; the ones around it
    /// (one of them moved into its slot) carry on unchanged, and a slot
    /// count of 2 forces a third sequence to wait for the first to leave.
    #[test]
    fn a_cancelled_sequence_leaves_the_others_unchanged() {
        let device = crate::backend::cpu_device();
        let model = crate::models::qwen3::toy_for_tests(&device);
        let prompts: [&[u32]; 3] = [&[3, 14, 15], &[9, 26, 5], &[11, 12]];
        let mut plan = [
            planned(0, prompts[0], 10),
            planned(0, prompts[1], 10),
            planned(6, prompts[2], 8),
        ];
        plan[0].cancel_at = Some(4);
        let got = run(&model, &device, 2, StepMode::Captured, &plan);
        // Admission emits the first token, each of the four steps one more.
        assert_eq!(got[0].len(), 5, "cancelled before its sixth token");
        let alone0 = greedy_decode(&model, prompts[0], 10, &device, StepMode::Dynamic);
        assert_eq!(got[0], alone0[..5], "the cancelled one, up to the cancel");
        for i in [1, 2] {
            let alone = greedy_decode(
                &model,
                plan[i].prompt,
                plan[i].max_tokens,
                &device,
                StepMode::Dynamic,
            );
            assert_eq!(got[i], alone, "sequence {i}");
        }
    }

    /// A sampled sequence draws the same tokens in a batch as alone: its RNG
    /// and options are its own, and the greedy row beside it does not touch
    /// them.
    #[test]
    fn seeded_sampling_in_a_batch_matches_alone() {
        let device = crate::backend::cpu_device();
        let model = crate::models::qwen3::toy_for_tests(&device);
        let sampled = SamplerOptions {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 20,
            seed: 7,
        };
        let prompt: &[u32] = &[3, 14, 15, 9];
        let mut plan = [planned(0, prompt, 16), planned(1, &[5, 6], 12)];
        plan[0].opts = sampled.clone();
        let got = run(&model, &device, 2, StepMode::Captured, &plan);
        let alone = pollster::block_on(
            model.generate(prompt, 16, &sampled, &device, |_| ControlFlow::Continue(())),
        )
        .expect("decodes");
        assert_eq!(got[0], alone);
        assert_eq!(
            got[1],
            greedy_decode(&model, &[5, 6], 12, &device, StepMode::Dynamic)
        );
    }
}
