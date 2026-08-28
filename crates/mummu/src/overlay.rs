//! The overlay floor: ring-streamed layers (SPEC 5).
//!
//! [`crate::workingset`] treats VRAM as a cache over *units* (experts,
//! neuron clusters). This module drops one level of granularity and treats it
//! as a cache over *whole layers*, which buys a stronger claim: with a small
//! fixed ring of slots, a model of ANY depth runs its every layer on the
//! device — VRAM stops being a capacity bound and becomes a bandwidth bound.
//! That is the overlay floor, and [`min_vram_bytes`] states it as a theorem:
//! the VRAM needed is `ring_slots * largest_layer + activations + KV`,
//! independent of layer count.
//!
//! # Why this is legal on this box
//!
//! Streaming only wins if the transfer engine and the compute engines do not
//! fight each other. Measured (2026-08-26, `probe_contention` on the real
//! pack): dGPU paired-vs-alone C = 1.01, host C = 1.08 — effectively private
//! memory systems, so hide windows that pair dGPU compute with host->device
//! DMA are real. The iGPU/flex pair shares DDR5 and is UNMEASURED; nothing
//! here assumes it is free.
//!
//! # The three-state planner
//!
//! Per layer, three placements: **Resident** (weights pinned in VRAM),
//! **Host** (compute where the bytes already live), **Stream** (weights ride
//! the ring through a recycled slot). [`plan`] chooses per layer from
//! *measured inputs only* — per-layer bytes, device/host compute times, and
//! staging bandwidth. Nothing is hardcoded, deliberately: a concurrent lane
//! is building a VNNI host kernel that may drop host FFN time from ~36 ms to
//! ~5-8 ms per layer, and the planner must flip its answers when the measured
//! `host_ms` input moves rather than silently fight that lane.
//!
//! # The executor
//!
//! [`Ring`] is the runtime half: `ring_slots` device slots, one background
//! prefetch thread that walks the repeating per-token layer order and keeps
//! every slot filled as far ahead of the consumer as it can reach, and an
//! in-order [`Ring::acquire`] that hands each layer's staged payload to the
//! compute loop. [`pipelined_layer_ms`] and [`SlotStage::upload_blocks`] add
//! the row-block refinement: splitting one layer's slab into row blocks lets
//! its own compute overlap its own transfer, collapsing the transfer-bound
//! layer cost from `tx + compute` toward `tx` alone.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use burn::tensor::{Device, Tensor};

// ---------------------------------------------------------------------------
// Deliverable 1: the three-state per-layer planner
// ---------------------------------------------------------------------------

/// Where one layer's weights live and compute happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAction {
    /// Pinned in VRAM for the whole run; costs `gpu_ms`, holds `bytes`.
    Resident,
    /// Computed on the host, where the authoritative bytes already are;
    /// costs `host_ms`, holds no VRAM. Contiguous host runs pay crossings.
    Host,
    /// Weights ride the ring through a recycled slot; computed on the device.
    Stream,
}

/// One layer's measured inputs. All measured, never assumed — the same
/// discipline as [`crate::workingset::Budget`].
#[derive(Debug, Clone)]
pub struct LayerCost {
    /// Device bytes the layer's weights occupy (Q4-packed, in practice).
    pub bytes: u64,
    /// Compute time on the device when the weights are there (resident or
    /// streamed — same kernels either way), milliseconds.
    pub gpu_ms: f64,
    /// Compute time on the host, milliseconds. The input the VNNI lane moves.
    pub host_ms: f64,
}

/// The machine model the planner prices against.
#[derive(Debug, Clone)]
pub struct OverlayModel {
    /// Host->device staging bandwidth, bytes per millisecond (measure it —
    /// see `examples/overlay-floor-probe.rs`; never assume the PCIe sticker).
    pub tx_bytes_per_ms: f64,
    /// Fixed latency of one slot handoff (submit + fence visibility), ms.
    /// Amortized over the ring depth in the per-layer stream cost.
    pub slot_latency_ms: f64,
    /// Ring depth: how many layer slots the device holds at once. Two is the
    /// minimum that overlaps; more hides jitter at a linear VRAM cost.
    pub ring_slots: usize,
    /// One host<->device activation hop, ms. A contiguous host run pays two.
    pub crossing_ms: f64,
}

/// The planner's answer.
#[derive(Debug, Clone)]
pub struct OverlayPlan {
    /// Per-layer placement, same order as the input.
    pub actions: Vec<LayerAction>,
    /// Predicted steady-state time for one token, ms — an upper bound under
    /// the documented model (see [`plan`]), used to order candidate plans.
    pub predicted_token_ms: f64,
    /// Bytes pinned by `Resident` layers.
    pub resident_bytes: u64,
    /// Bytes reserved for the ring: `ring_slots * max streamed layer bytes`.
    /// The ring is a FIXED VRAM cost paid once for the plan — slots are
    /// recycled, so it never scales with how many layers stream.
    pub ring_bytes: u64,
}

/// Transfer time for `bytes` under `m`, ms. Infinite when the model says the
/// link cannot move bytes at all — which makes Stream unpickable, not a NaN.
fn tx_ms(bytes: u64, m: &OverlayModel) -> f64 {
    if m.tx_bytes_per_ms <= 0.0 {
        return f64::INFINITY;
    }
    bytes as f64 / m.tx_bytes_per_ms
}

/// A streamed layer's per-layer charge: `max(gpu_ms, tx_ms)` — the steady
/// state of a double-buffered ring, where each layer occupies the timeline
/// for the slower of its compute and its own transfer, never their sum —
/// plus the slot handoff latency amortized over the ring depth. This is the
/// pinned decision-rule cost: a layer streams exactly when this beats
/// `host_ms`, all from the inputs.
fn stream_slot_cost(l: &LayerCost, m: &OverlayModel) -> f64 {
    if m.ring_slots == 0 {
        return f64::INFINITY; // no slots, no stream
    }
    l.gpu_ms.max(tx_ms(l.bytes, m)) + m.slot_latency_ms / m.ring_slots as f64
}

/// What [`evaluate`] reports for a feasible assignment.
struct Eval {
    predicted_ms: f64,
    resident_bytes: u64,
    ring_bytes: u64,
}

/// Price one complete assignment under the documented model, or `None` when
/// it does not fit the budget (resident bytes + the ring reserve exceed it,
/// or the arithmetic overflows u64 — treated as "does not fit").
///
/// The model, term by term:
///
/// - **base**: each layer's charge on its engine — `gpu_ms` resident,
///   `host_ms` host, [`stream_slot_cost`] streamed.
/// - **crossings**: activation hops, counted cyclically over the repeating
///   token loop. Every boundary where the plan changes between Host and
///   non-Host costs one `crossing_ms`, so each contiguous host run pays two
///   — and the degenerate all-host plan pays zero, because nothing ever
///   crosses. Crossings are latency, so they are NOT part of the hide
///   window below (conservative).
/// - **exposed**: the oversubscription penalty. Per-layer `max()` credits a
///   streamed transfer with hiding under its own pipeline slot, which is
///   exact when every layer streams but too kind when a few large transfers
///   depend on OTHER layers' compute for cover. So the plan is additionally
///   charged `max(0, sum_tx - hide_window)` where `hide_window` is the sum
///   of every layer's compute on its assigned engine (streamed layers
///   contribute `gpu_ms`) minus the single largest streamed transfer — the
///   largest transfer is denied hiding credit because the window it needs
///   may not line up with where the compute sits (conservative).
///
/// In the deeply oversubscribed regime the per-layer `max()` and `exposed`
/// BOTH bind, double-charging some transfer time. Accepted: the prediction
/// is an upper bound used to order candidate plans — honest and documented
/// over clever.
fn evaluate(
    layers: &[LayerCost],
    actions: &[LayerAction],
    vram_budget_bytes: u64,
    m: &OverlayModel,
) -> Option<Eval> {
    debug_assert_eq!(layers.len(), actions.len());

    // -- fit ---------------------------------------------------------------
    let mut resident_bytes: u64 = 0;
    let mut max_stream_bytes: u64 = 0;
    for (l, a) in layers.iter().zip(actions) {
        match a {
            LayerAction::Resident => resident_bytes = resident_bytes.checked_add(l.bytes)?,
            LayerAction::Stream => max_stream_bytes = max_stream_bytes.max(l.bytes),
            LayerAction::Host => {}
        }
    }
    let ring_bytes = (m.ring_slots as u64).checked_mul(max_stream_bytes)?;
    if resident_bytes.checked_add(ring_bytes)? > vram_budget_bytes {
        return None;
    }

    // -- price -------------------------------------------------------------
    let mut base = 0.0;
    let mut hide_window = 0.0;
    let mut sum_tx = 0.0;
    let mut max_tx = 0.0f64;
    for (l, a) in layers.iter().zip(actions) {
        match a {
            LayerAction::Resident => {
                base += l.gpu_ms;
                hide_window += l.gpu_ms;
            }
            LayerAction::Host => {
                base += l.host_ms;
                hide_window += l.host_ms;
            }
            LayerAction::Stream => {
                let t = tx_ms(l.bytes, m);
                base += stream_slot_cost(l, m);
                hide_window += l.gpu_ms;
                sum_tx += t;
                max_tx = max_tx.max(t);
            }
        }
    }

    let n = actions.len();
    let mut boundaries = 0usize;
    for i in 0..n {
        let here = matches!(actions[i], LayerAction::Host);
        let next = matches!(actions[(i + 1) % n], LayerAction::Host);
        if here != next {
            boundaries += 1;
        }
    }
    let crossings = m.crossing_ms * boundaries as f64;

    let hide = (hide_window - max_tx).max(0.0);
    let exposed = (sum_tx - hide).max(0.0);

    Some(Eval {
        predicted_ms: base + crossings + exposed,
        resident_bytes,
        ring_bytes,
    })
}

/// Choose per-layer placements minimizing predicted steady-state token time.
///
/// # The decision rule (pinned — "do not fight the host-kernel lane")
///
/// A layer that cannot stay resident goes to the **host** when
/// `host_ms < max(gpu_ms, tx_ms) + slot_latency_ms / ring_slots`, and
/// **streams** when the inequality reverses — computed from the INPUT
/// numbers, never a constant. Today's host FFN (~36 ms) loses to a ~6-12 ms
/// streamed layer; a VNNI host kernel at ~5 ms beats a 12 ms transfer, and
/// this planner flips with it. The unit tests pin both directions.
///
/// # Solver
///
/// Greedy + local exchange, deterministic:
///
/// 1. Every layer takes its cheapest action as if the budget were infinite
///    (ties prefer Resident, then Stream — the cheaper-VRAM tie already lost
///    on cost, so prefer the faster engine).
/// 2. While the plan does not fit (resident bytes + ring reserve over
///    budget), evict the resident layer with the smallest off-device penalty
///    per byte (`min(stream, host) - gpu_ms` per byte) to its best
///    non-resident action. Evicting to Stream can GROW the ring reserve;
///    the loop keeps going — residents strictly decrease, so it terminates.
///    If the ring reserve alone exceeds the budget with no residents left,
///    streaming is unaffordable: every Stream falls back to Host, which is
///    always feasible (the host already holds the bytes).
/// 3. Local exchange: sweep the layers, trying each of the three actions
///    per layer against the full [`evaluate`] model (crossings and
///    oversubscription included), adopting strict improvements, until a
///    sweep changes nothing (capped). This is what repairs greedy scars —
///    e.g. a resident evicted while chasing a ring that then proved
///    unaffordable gets re-pinned here.
///
/// O(sweeps * layers^2): trivial at model scale (tens of layers).
#[must_use]
pub fn plan(layers: &[LayerCost], vram_budget_bytes: u64, m: &OverlayModel) -> OverlayPlan {
    if layers.is_empty() {
        return OverlayPlan {
            actions: Vec::new(),
            predicted_token_ms: 0.0,
            resident_bytes: 0,
            ring_bytes: 0,
        };
    }

    // -- 1. unconstrained best per layer -----------------------------------
    let mut actions: Vec<LayerAction> = layers
        .iter()
        .map(|l| {
            let s = stream_slot_cost(l, m);
            if l.gpu_ms <= s && l.gpu_ms <= l.host_ms {
                LayerAction::Resident
            } else if s <= l.host_ms {
                LayerAction::Stream
            } else {
                LayerAction::Host
            }
        })
        .collect();

    // -- 2. evict until it fits --------------------------------------------
    while evaluate(layers, &actions, vram_budget_bytes, m).is_none() {
        let mut victim: Option<(usize, f64)> = None;
        for (i, a) in actions.iter().enumerate() {
            if *a != LayerAction::Resident {
                continue;
            }
            let l = &layers[i];
            let off = stream_slot_cost(l, m).min(l.host_ms);
            let per_byte = (off - l.gpu_ms) / l.bytes.max(1) as f64;
            if victim.is_none_or(|(_, best)| per_byte < best) {
                victim = Some((i, per_byte));
            }
        }
        match victim {
            Some((i, _)) => {
                let l = &layers[i];
                actions[i] = if stream_slot_cost(l, m) < l.host_ms {
                    LayerAction::Stream
                } else {
                    LayerAction::Host
                };
            }
            None => {
                // No residents left and still over budget: the ring reserve
                // alone does not fit. Streaming is unaffordable here — fall
                // back to the host, which needs no device bytes at all.
                for a in &mut actions {
                    if *a == LayerAction::Stream {
                        *a = LayerAction::Host;
                    }
                }
                break;
            }
        }
    }

    // -- 3. local exchange against the full model --------------------------
    let mut current = evaluate(layers, &actions, vram_budget_bytes, m)
        .expect("the all-host fallback is always feasible");
    for _sweep in 0..8 {
        let mut improved = false;
        for i in 0..layers.len() {
            for cand in [
                LayerAction::Resident,
                LayerAction::Stream,
                LayerAction::Host,
            ] {
                if cand == actions[i] {
                    continue;
                }
                let prev = actions[i];
                actions[i] = cand;
                match evaluate(layers, &actions, vram_budget_bytes, m) {
                    Some(e) if e.predicted_ms + 1e-9 < current.predicted_ms => {
                        current = e;
                        improved = true;
                    }
                    _ => actions[i] = prev,
                }
            }
        }
        if !improved {
            break;
        }
    }

    OverlayPlan {
        actions,
        predicted_token_ms: current.predicted_ms,
        resident_bytes: current.resident_bytes,
        ring_bytes: current.ring_bytes,
    }
}

/// **The capacity theorem.** The VRAM a model needs to run every layer on
/// the device via the ring is
///
/// ```text
/// ring_slots * max_layer_bytes + activation_reserve + kv_bytes
/// ```
///
/// — independent of how many layers the model has. Depth costs bandwidth
/// (every streamed layer's bytes cross the link every token), never
/// capacity: slots are recycled, so a 27B and a 270B with the same layer
/// width need the same VRAM to stream. Whether streaming is *fast* is
/// [`plan`]'s question; whether it *fits* is settled here.
#[must_use]
pub fn min_vram_bytes(
    layers: &[LayerCost],
    m: &OverlayModel,
    activation_reserve_bytes: u64,
    kv_bytes: u64,
) -> u64 {
    let max_layer = layers.iter().map(|l| l.bytes).max().unwrap_or(0);
    (m.ring_slots as u64)
        .saturating_mul(max_layer)
        .saturating_add(activation_reserve_bytes)
        .saturating_add(kv_bytes)
}

// ---------------------------------------------------------------------------
// Deliverable 3: row-block pipelining (the floor-collapse math)
// ---------------------------------------------------------------------------

/// Predicted cost of streaming one layer in `blocks` equal row-blocks with
/// compute overlapping transfer at block granularity.
///
/// # Derivation
///
/// Two serial resources: the link transfers blocks one after another, the
/// device computes them one after another, and block `k`'s compute waits
/// only on block `k`'s transfer. Per block:
///
/// - transfer `a = tx_ms/blocks + block_latency_ms` — each block is its own
///   submit, so the fixed issue latency is paid PER BLOCK (this is what
///   makes [`best_row_blocks`] a real trade-off instead of "always max");
/// - compute `b = compute_ms/blocks`.
///
/// With identical blocks the makespan is the classic two-stage pipeline:
///
/// ```text
/// a                  fill: nothing overlaps the first block's transfer
/// + (blocks-1) * max(a, b)   steady state: both resources run, the slower paces
/// + b                drain: the last block's compute
/// ```
///
/// Consequences (each pinned by a test):
///
/// - `blocks = 1`: `tx + compute + latency` — no overlap, the serial floor.
/// - transfer-bound (`tx >= compute`) with small latency:
///   `≈ tx + compute/blocks + blocks*latency` — the layer costs its
///   TRANSFER plus one compute block, not `tx + compute`. This is the floor
///   collapse: an oversubscribed streamed layer approaches the link
///   bandwidth alone.
/// - `blocks -> inf` at zero latency: `max(tx, compute)` plus one block's
///   worth — perfect overlap.
/// - At zero latency more blocks never hurt (the serial remainder shrinks
///   like `min(tx, compute)/blocks`).
///
/// # Panics
///
/// `blocks` must be at least 1.
#[must_use]
pub fn pipelined_layer_ms(
    tx_ms: f64,
    compute_ms: f64,
    blocks: usize,
    block_latency_ms: f64,
) -> f64 {
    assert!(blocks >= 1, "a layer streams in at least one block");
    let n = blocks as f64;
    let a = tx_ms / n + block_latency_ms;
    let b = compute_ms / n;
    a + (n - 1.0) * a.max(b) + b
}

/// The block count minimizing [`pipelined_layer_ms`], scanned exactly over
/// `1..=max_blocks`.
///
/// The trade is the latency-bandwidth product: finer blocks shrink the
/// serial remainder like `min(tx, compute)/blocks` but pay
/// `blocks * latency` in issue overhead. The turn sits near the block whose
/// transfer time equals the issue latency — block bytes ≈ latency ×
/// bandwidth — giving `n* ≈ sqrt(min(tx, compute) / latency)` in the
/// transfer-bound regime. The scan is exact, O(max_blocks), and ties keep
/// the smaller count (fewer submits for the same time).
///
/// # Panics
///
/// `max_blocks` must be at least 1.
#[must_use]
pub fn best_row_blocks(tx_ms: f64, compute_ms: f64, latency_ms: f64, max_blocks: usize) -> usize {
    assert!(max_blocks >= 1, "need at least one block to choose from");
    let mut best = (1usize, pipelined_layer_ms(tx_ms, compute_ms, 1, latency_ms));
    for n in 2..=max_blocks {
        let t = pipelined_layer_ms(tx_ms, compute_ms, n, latency_ms);
        if t < best.1 {
            best = (n, t);
        }
    }
    best.0
}

// ---------------------------------------------------------------------------
// Deliverable 2: the ring executor
// ---------------------------------------------------------------------------

/// One row-block landing during a blocked upload — progress, not payload.
/// The payload itself is what [`SlotStage::upload_blocks`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// The layer being uploaded.
    pub layer: usize,
    /// This block's index, `0..of`.
    pub index: usize,
    /// Total blocks in this upload.
    pub of: usize,
}

/// Something that can make one layer's weights device-resident.
///
/// `upload` blocks until the layer's weights are on the device (or, for
/// deferred-completion backends like wgpu, until the transfer is queued such
/// that any consumer of the payload will observe it — see [`TensorStage`]),
/// returning the device handle. The ring calls it from its ONE prefetch
/// thread, so implementations need no internal ordering.
pub trait SlotStage: Send + Sync {
    /// The device-resident handle `upload` produces.
    type Payload: Send;

    /// Make `layer`'s weights device-resident; return the handle.
    fn upload(&self, layer: usize) -> Self::Payload;

    /// Block-granular upload: land the layer in row blocks, reporting each
    /// block through `on_block` as it arrives, so a consumer can start
    /// computing on early rows while later rows are still in flight (the
    /// [`pipelined_layer_ms`] mechanism). Default: one call to [`upload`],
    /// reported as a single block — correct for any stage, overlap-free.
    fn upload_blocks(&self, layer: usize, on_block: &mut dyn FnMut(Block)) -> Self::Payload {
        let payload = self.upload(layer);
        on_block(Block {
            layer,
            index: 0,
            of: 1,
        });
        payload
    }
}

/// State behind the ring's mutex.
struct RingState<P> {
    /// Uploaded-but-unconsumed payloads, in schedule-position order. Uploads
    /// complete in order (one prefetch thread) and acquires pop in order, so
    /// the front is always the next position to hand out.
    ready: VecDeque<(u64, P)>,
    /// Next schedule position the prefetcher will claim. Positions index the
    /// INFINITE repetition of the order (`layer = order[pos % order.len()]`),
    /// so wrapping from token to token is nothing special — u64 does not
    /// wrap in any plausible run.
    next_upload: u64,
    /// Next schedule position `acquire` will hand out.
    next_acquire: u64,
    /// Slots in use: the in-flight upload + ready payloads + guards handed
    /// out. The prefetcher only claims a position while `occupied < slots`,
    /// which is the operational capacity theorem: at most `slots` layers'
    /// bytes exist on the device at once, whatever the model's depth.
    occupied: usize,
    shutdown: bool,
    /// The prefetch thread exited (shutdown, or an upload panicked). Lets a
    /// blocked `acquire` fail loudly instead of waiting forever.
    prefetcher_gone: bool,
}

struct RingShared<P> {
    state: Mutex<RingState<P>>,
    cv: Condvar,
    /// The per-token layer order, repeated indefinitely.
    order: Vec<usize>,
    slots: usize,
    /// Blocks reported by uploads so far (observability; the default
    /// [`SlotStage::upload_blocks`] reports exactly one per upload).
    blocks_landed: AtomicU64,
}

/// Sets `prefetcher_gone` however the prefetch loop exits — clean shutdown
/// or a panic inside an upload — so consumers never wait on a dead thread.
struct Retire<'a, P>(&'a RingShared<P>);

impl<P> Drop for Retire<'_, P> {
    fn drop(&mut self) {
        let mut st = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        st.prefetcher_gone = true;
        self.0.cv.notify_all();
    }
}

fn prefetch_loop<S: SlotStage>(shared: &RingShared<S::Payload>, stage: &S) {
    let _retire = Retire(shared);
    loop {
        let pos = {
            let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            while !st.shutdown && st.occupied >= shared.slots {
                st = shared.cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if st.shutdown {
                return;
            }
            let pos = st.next_upload;
            st.next_upload += 1;
            st.occupied += 1; // the slot is spoken for while the upload flies
            pos
        };
        let layer = shared.order[(pos % shared.order.len() as u64) as usize];
        let payload = stage.upload_blocks(layer, &mut |b: Block| {
            debug_assert!(b.of >= 1 && b.index < b.of, "malformed block {b:?}");
            shared.blocks_landed.fetch_add(1, Ordering::Relaxed);
        });
        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.ready.push_back((pos, payload));
        shared.cv.notify_all();
    }
}

/// A fixed set of device slots streaming an endless repetition of a layer
/// order, filled by one background prefetch thread.
///
/// # Why farthest-ahead-first, and why it is optimal
///
/// Uploads are serial on one link and consumption follows the schedule, so
/// the uploaded-but-unconsumed positions always form a contiguous window
/// `[next_acquire, next_upload)`. The prefetcher's only freedom is which
/// scheduled-but-missing layer to fetch into a freed slot, and it always
/// extends the window — the farthest-ahead layer a free slot admits. The
/// exchange argument: in any schedule that fetches some later position
/// before an earlier one, swapping the two finishes the earlier (sooner
/// needed) one no later and the later one no later than the earlier one
/// previously finished — so no acquire ever waits longer under the in-order,
/// maximally-ahead walk. With `S` slots, steady state runs `S - 1` transfers
/// ahead of the guard the consumer holds.
///
/// # Contract
///
/// [`Ring::acquire`] must be called in schedule order (asserted): the ring
/// exists to serve a KNOWN repeating order, exactly like the working-set
/// scheduler — that foreknowledge is what makes prefetching a decision
/// rather than a guess. Dropping the returned [`SlotGuard`] frees the slot
/// and un-blocks the prefetcher. Dropping the ring itself shuts the thread
/// down cleanly (it finishes at most one in-flight upload first).
pub struct Ring<S: SlotStage> {
    shared: Arc<RingShared<S::Payload>>,
    stage: Arc<S>,
    prefetcher: Option<std::thread::JoinHandle<()>>,
}

impl<S> Ring<S>
where
    S: SlotStage + 'static,
    S::Payload: 'static,
{
    /// Spawn the prefetch thread over `order` (one token's streamed layers,
    /// repeated indefinitely) with `slots` device slots.
    ///
    /// # Panics
    ///
    /// `slots` must be at least 1 and `order` non-empty.
    #[must_use]
    pub fn new(stage: S, slots: usize, order: Vec<usize>) -> Self {
        assert!(slots >= 1, "a ring needs at least one slot");
        assert!(!order.is_empty(), "a ring needs a layer order to stream");
        let shared = Arc::new(RingShared {
            state: Mutex::new(RingState {
                ready: VecDeque::new(),
                next_upload: 0,
                next_acquire: 0,
                occupied: 0,
                shutdown: false,
                prefetcher_gone: false,
            }),
            cv: Condvar::new(),
            order,
            slots,
            blocks_landed: AtomicU64::new(0),
        });
        let stage = Arc::new(stage);
        let t_shared = Arc::clone(&shared);
        let t_stage = Arc::clone(&stage);
        let prefetcher = std::thread::Builder::new()
            .name("mummu-overlay-ring".into())
            .spawn(move || prefetch_loop::<S>(&t_shared, &t_stage))
            .expect("spawn the overlay ring prefetch thread");
        Self {
            shared,
            stage,
            prefetcher: Some(prefetcher),
        }
    }

    /// Block until `layer`'s payload has landed, in schedule order, and hand
    /// it over. The slot stays occupied until the guard drops.
    ///
    /// # Panics
    ///
    /// When `layer` is not the next layer in the schedule (the ring serves a
    /// known order — acquiring out of order is a caller bug, not a runtime
    /// condition), or when the prefetch thread died before delivering.
    #[must_use]
    pub fn acquire(&self, layer: usize) -> SlotGuard<S::Payload> {
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let pos = st.next_acquire;
        let expected = self.shared.order[(pos % self.shared.order.len() as u64) as usize];
        assert_eq!(
            layer, expected,
            "ring acquire must follow the scheduled order: position {pos} is layer {expected}"
        );
        st.next_acquire += 1;
        loop {
            if st.ready.front().map(|(p, _)| *p) == Some(pos) {
                let (_, payload) = st.ready.pop_front().expect("front just checked");
                return SlotGuard {
                    layer,
                    payload: Some(payload),
                    shared: Arc::clone(&self.shared),
                };
            }
            assert!(
                !st.prefetcher_gone,
                "ring prefetch thread exited before layer {layer} landed"
            );
            st = self.shared.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The stage this ring uploads through.
    #[must_use]
    pub fn stage(&self) -> &S {
        &self.stage
    }

    /// Blocks reported by uploads so far (one per upload for stages using
    /// the default [`SlotStage::upload_blocks`]).
    #[must_use]
    pub fn blocks_landed(&self) -> u64 {
        self.shared.blocks_landed.load(Ordering::Relaxed)
    }
}

impl<S: SlotStage> Drop for Ring<S> {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            st.shutdown = true;
            self.shared.cv.notify_all();
        }
        if let Some(t) = self.prefetcher.take() {
            // Swallow a panicked upload's poison: tearing down a ring whose
            // stage failed should not double-panic the caller.
            let _ = t.join();
        }
    }
}

/// Holds one acquired payload and, through it, one ring slot. Dropping the
/// guard drops the payload FIRST (freeing its device memory) and only then
/// releases the slot — the other order would let the prefetcher briefly
/// over-commit the device by one layer.
pub struct SlotGuard<P> {
    layer: usize,
    payload: Option<P>,
    shared: Arc<RingShared<P>>,
}

impl<P> SlotGuard<P> {
    /// The layer this payload belongs to.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.layer
    }
}

impl<P> std::ops::Deref for SlotGuard<P> {
    type Target = P;
    fn deref(&self) -> &P {
        self.payload.as_ref().expect("payload present until drop")
    }
}

impl<P> std::ops::DerefMut for SlotGuard<P> {
    fn deref_mut(&mut self) -> &mut P {
        self.payload.as_mut().expect("payload present until drop")
    }
}

impl<P> Drop for SlotGuard<P> {
    fn drop(&mut self) {
        drop(self.payload.take());
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert!(st.occupied >= 1, "a live guard implies an occupied slot");
        st.occupied -= 1;
        self.shared.cv.notify_all();
    }
}

/// A [`SlotStage`] over per-layer host tensors, uploading through
/// [`crate::backend::move_to`] — the one cross-device transport.
///
/// Completion is NOT forced here: wgpu transfers are deferred-mapped, so the
/// wait surfaces at the payload's first use (whoever first touches the bytes
/// pays the fence — see the staging notes in `backend.rs`). `acquire` hands
/// the tensor over without forcing, and the pipelining benefit still holds
/// because the copy was QUEUED while earlier layers computed — the queue is
/// filled early even though the fence is observed late.
///
/// Kept minimal on purpose: the production wiring (which tensors, which
/// device, eviction interplay with the working set) belongs to the engine
/// lane, not here.
pub struct TensorStage {
    host: Vec<Tensor<2>>,
    target: Device,
}

impl TensorStage {
    /// Stage over `host` tensors (index = layer), uploading to `target`.
    #[must_use]
    pub fn new(host: Vec<Tensor<2>>, target: Device) -> Self {
        Self { host, target }
    }
}

impl SlotStage for TensorStage {
    type Payload = Tensor<2>;

    fn upload(&self, layer: usize) -> Tensor<2> {
        let t = self
            .host
            .get(layer)
            .unwrap_or_else(|| {
                panic!(
                    "TensorStage: layer {layer} out of range ({} layers held)",
                    self.host.len()
                )
            })
            .clone();
        crate::backend::move_to(t, &self.target)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicIsize;
    use std::time::{Duration, Instant};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // ---- planner ---------------------------------------------------------

    fn model(tx_bytes_per_ms: f64) -> OverlayModel {
        OverlayModel {
            tx_bytes_per_ms,
            slot_latency_ms: 1.0,
            ring_slots: 2,
            crossing_ms: 0.5,
        }
    }

    fn uniform(n: usize, bytes: u64, gpu_ms: f64, host_ms: f64) -> Vec<LayerCost> {
        (0..n)
            .map(|_| LayerCost {
                bytes,
                gpu_ms,
                host_ms,
            })
            .collect()
    }

    #[test]
    fn empty_model_plans_to_nothing() {
        let p = plan(&[], 1_000, &model(1.0));
        assert!(p.actions.is_empty());
        assert!(close(p.predicted_token_ms, 0.0));
        assert_eq!(p.resident_bytes, 0);
        assert_eq!(p.ring_bytes, 0);
    }

    #[test]
    fn a_model_that_fits_is_entirely_resident() {
        let layers = uniform(3, 10, 1.0, 5.0);
        let p = plan(&layers, 1_000, &model(1.0));
        assert!(p.actions.iter().all(|a| *a == LayerAction::Resident));
        assert!(close(p.predicted_token_ms, 3.0), "{}", p.predicted_token_ms);
        assert_eq!(p.resident_bytes, 30);
        assert_eq!(p.ring_bytes, 0);
    }

    /// The pinned rule, host direction: host_ms 5 vs tx 12 (gpu 2) — a fast
    /// host kernel must WIN over streaming, from the input numbers alone.
    /// This is the "planner must not silently fight the host-kernel lane"
    /// test: when the VNNI lane lands, host_ms drops and this is the branch
    /// the planner takes.
    #[test]
    fn a_fast_host_kernel_beats_streaming() {
        // tx_ms = 50 bytes / (50/12 bytes per ms) = 12 ms.
        let layers = uniform(8, 50, 2.0, 5.0);
        let m = model(50.0 / 12.0);
        let p = plan(&layers, 100, &m);
        assert!(
            p.actions.iter().all(|a| *a != LayerAction::Stream),
            "host 5 ms < max(gpu 2, tx 12) + 0.5 amortized ring: nothing may stream: {:?}",
            p.actions
        );
        let residents = p
            .actions
            .iter()
            .filter(|a| **a == LayerAction::Resident)
            .count();
        assert_eq!(residents, 2, "budget 100 pins exactly two 50-byte layers");
        assert!(p.resident_bytes <= 100);
        assert_eq!(p.ring_bytes, 0, "no stream, no ring reserve");
    }

    /// The pinned rule, stream direction: host_ms 36 vs tx 12, gpu 14 —
    /// streaming wins (max(14, 12) + 0.5 = 14.5 < 36), again from inputs.
    #[test]
    fn streaming_beats_a_slow_host() {
        let layers = uniform(8, 50, 14.0, 36.0);
        let m = model(50.0 / 12.0);
        let p = plan(&layers, 100, &m);
        assert!(
            p.actions.iter().all(|a| *a == LayerAction::Stream),
            "stream 14.5 ms < host 36 ms and the ring (2*50) exactly fits: {:?}",
            p.actions
        );
        assert_eq!(p.resident_bytes, 0);
        assert_eq!(p.ring_bytes, 100);
    }

    /// The capacity theorem, exercised: a model whose total bytes are 4x the
    /// budget still streams every layer inside `min_vram_bytes` — depth
    /// costs bandwidth, not capacity.
    #[test]
    fn an_oversubscribed_model_streams_inside_the_ring_budget() {
        let layers = uniform(8, 100, 1.0, 1_000.0);
        let m = OverlayModel {
            tx_bytes_per_ms: 100.0, // tx = 1 ms/layer
            slot_latency_ms: 0.5,
            ring_slots: 2,
            crossing_ms: 1.0,
        };
        let budget = min_vram_bytes(&layers, &m, 0, 0);
        assert_eq!(budget, 200);
        let total: u64 = layers.iter().map(|l| l.bytes).sum();
        assert!(total >= 2 * budget, "the point: the model does not fit");
        let p = plan(&layers, budget, &m);
        assert!(
            p.actions.iter().all(|a| *a == LayerAction::Stream),
            "{:?}",
            p.actions
        );
        assert_eq!(p.resident_bytes, 0);
        assert_eq!(p.ring_bytes, budget);
        assert!(
            p.predicted_token_ms.is_finite() && p.predicted_token_ms < 100.0,
            "streamed, not host-bound: {}",
            p.predicted_token_ms
        );
    }

    #[test]
    fn min_vram_is_independent_of_model_depth() {
        let m = OverlayModel {
            tx_bytes_per_ms: 1.0,
            slot_latency_ms: 0.0,
            ring_slots: 3,
            crossing_ms: 0.0,
        };
        let shallow = uniform(4, 100, 1.0, 1.0);
        let deep = uniform(400, 100, 1.0, 1.0);
        assert_eq!(
            min_vram_bytes(&shallow, &m, 50, 25),
            min_vram_bytes(&deep, &m, 50, 25)
        );
        assert_eq!(min_vram_bytes(&shallow, &m, 50, 25), 3 * 100 + 50 + 25);
    }

    /// The ring reserve is `slots * max streamed bytes`, charged once — not
    /// summed over the streamed layers.
    #[test]
    fn the_ring_is_reserved_once_not_per_layer() {
        let layers: Vec<LayerCost> = (0..6)
            .map(|i| LayerCost {
                bytes: if i % 2 == 0 { 100 } else { 80 },
                gpu_ms: 1.0,
                host_ms: 1_000.0,
            })
            .collect();
        let m = OverlayModel {
            tx_bytes_per_ms: 100.0,
            slot_latency_ms: 0.5,
            ring_slots: 2,
            crossing_ms: 1.0,
        };
        let p = plan(&layers, 200, &m);
        let streamed = p
            .actions
            .iter()
            .filter(|a| **a == LayerAction::Stream)
            .count();
        assert_eq!(streamed, 6, "{:?}", p.actions);
        assert_eq!(p.ring_bytes, 200, "2 slots x the 100-byte max, once");
    }

    /// When even the ring reserve cannot fit, streaming is off the table and
    /// the host — which needs no device bytes — is the backstop. All-host
    /// also pays zero crossings (nothing ever crosses).
    #[test]
    fn streaming_falls_back_to_host_when_the_ring_cannot_fit() {
        let layers = uniform(4, 50, 1.0, 5.0);
        let m = model(50.0 / 2.0); // tx = 2 ms: streaming would be attractive
        let p = plan(&layers, 30, &m); // < one layer, < the 100-byte ring
        assert!(
            p.actions.iter().all(|a| *a == LayerAction::Host),
            "{:?}",
            p.actions
        );
        assert_eq!(p.resident_bytes, 0);
        assert_eq!(p.ring_bytes, 0);
        assert!(
            close(p.predicted_token_ms, 20.0),
            "{}",
            p.predicted_token_ms
        );
    }

    /// Crossings are counted cyclically: each contiguous host run pays two,
    /// wherever the token boundary falls.
    #[test]
    fn host_runs_pay_two_crossings_each() {
        let layers = uniform(5, 10, 1.0, 2.0);
        let actions = [
            LayerAction::Resident,
            LayerAction::Host,
            LayerAction::Host,
            LayerAction::Resident,
            LayerAction::Host,
        ];
        let m = OverlayModel {
            tx_bytes_per_ms: 1_000.0,
            slot_latency_ms: 0.0,
            ring_slots: 2,
            crossing_ms: 2.0,
        };
        let e = evaluate(&layers, &actions, 1_000, &m).expect("fits");
        // base = 1+2+2+1+2 = 8; two host runs (one wraps 4 -> 0) = 4
        // boundaries = 4 crossings at 2 ms.
        assert!(close(e.predicted_ms, 8.0 + 8.0), "{}", e.predicted_ms);
        let zero = OverlayModel {
            crossing_ms: 0.0,
            ..m
        };
        let e0 = evaluate(&layers, &actions, 1_000, &zero).expect("fits");
        assert!(close(e0.predicted_ms, 8.0), "{}", e0.predicted_ms);
    }

    /// Transfers that outrun the compute window are exposed, not hidden:
    /// with 4 ms of total compute against 150 ms of transfers, the model
    /// must charge the excess.
    #[test]
    fn oversubscribed_transfers_are_exposed_not_hidden() {
        let layers = uniform(4, 50, 1.0, 10.0);
        let actions = [
            LayerAction::Resident,
            LayerAction::Stream,
            LayerAction::Stream,
            LayerAction::Stream,
        ];
        let m = OverlayModel {
            tx_bytes_per_ms: 1.0, // tx = 50 ms per streamed layer
            slot_latency_ms: 0.5,
            ring_slots: 2,
            crossing_ms: 0.0,
        };
        let e = evaluate(&layers, &actions, 1_000, &m).expect("fits");
        // base = 1 + 3 * (max(1, 50) + 0.25) = 151.75
        // hide_window = 4 (all-gpu compute) - 50 (largest tx) -> 0
        // exposed = 150 - 0 = 150
        assert!(close(e.predicted_ms, 151.75 + 150.0), "{}", e.predicted_ms);
    }

    // ---- row-block pipelining --------------------------------------------

    #[test]
    fn one_block_means_no_overlap() {
        assert!(close(pipelined_layer_ms(10.0, 4.0, 1, 0.0), 14.0));
        assert!(close(pipelined_layer_ms(10.0, 4.0, 1, 2.0), 16.0));
    }

    #[test]
    fn many_blocks_approach_the_transfer_compute_bound() {
        // Transfer-bound: collapses onto tx (+ one vanishing block).
        let v = pipelined_layer_ms(12.0, 5.0, 100_000, 0.0);
        assert!(v >= 12.0 && v - 12.0 < 0.01, "{v}");
        // Compute-bound: collapses onto compute.
        let v = pipelined_layer_ms(5.0, 12.0, 100_000, 0.0);
        assert!(v >= 12.0 && v - 12.0 < 0.01, "{v}");
    }

    #[test]
    fn more_blocks_never_hurt_without_latency() {
        let mut prev = f64::INFINITY;
        for n in 1..=64 {
            let v = pipelined_layer_ms(12.0, 7.0, n, 0.0);
            assert!(v <= prev + 1e-12, "blocks {n}: {v} > {prev}");
            prev = v;
        }
    }

    #[test]
    fn best_row_blocks_balances_latency_against_overlap() {
        // Free issues: finer is always better.
        assert_eq!(best_row_blocks(12.0, 12.0, 0.0, 64), 64);
        // Ruinous issues: one block, the serial floor.
        assert_eq!(best_row_blocks(12.0, 12.0, 100.0, 64), 1);
        // Real latency: an interior optimum near sqrt(compute/latency)
        // (sqrt(12/0.05) ~ 15.5 — the latency-bandwidth product).
        let b = best_row_blocks(12.0, 12.0, 0.05, 4096);
        assert!((12..=20).contains(&b), "{b}");
        let at = |n| pipelined_layer_ms(12.0, 12.0, n, 0.05);
        assert!(at(b) <= at(1) && at(b) <= at(4096));
    }

    // ---- the ring --------------------------------------------------------

    /// Upload = sleep(tx) then hand back the layer id. Tracks live payloads
    /// so tests can pin the operational capacity theorem (never more than
    /// `slots` payloads alive).
    struct FakeStage {
        tx: Duration,
        live: Arc<AtomicIsize>,
        peak: Arc<AtomicIsize>,
    }

    struct FakePayload {
        layer: usize,
        live: Arc<AtomicIsize>,
    }

    impl Drop for FakePayload {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl SlotStage for FakeStage {
        type Payload = FakePayload;
        fn upload(&self, layer: usize) -> FakePayload {
            std::thread::sleep(self.tx);
            let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            FakePayload {
                layer,
                live: Arc::clone(&self.live),
            }
        }
    }

    fn fake(tx_ms: u64) -> (FakeStage, Arc<AtomicIsize>) {
        let live = Arc::new(AtomicIsize::new(0));
        let peak = Arc::new(AtomicIsize::new(0));
        (
            FakeStage {
                tx: Duration::from_millis(tx_ms),
                live: Arc::clone(&live),
                peak: Arc::clone(&peak),
            },
            peak,
        )
    }

    /// (a) tx <= compute: transfers hide, wall ~= sum of compute. Margins
    /// are deliberately coarse (sleep-based, CI-safe): the bound sits well
    /// under the no-overlap serial time (~1080 ms) while allowing ~120 ms
    /// of scheduler noise.
    #[test]
    fn transfers_hide_under_compute_with_two_slots() {
        let order = vec![0usize, 1, 2];
        let (stage, peak) = fake(40);
        let ring = Ring::new(stage, 2, order.clone());
        let compute = Duration::from_millis(80);
        let t = Instant::now();
        for _token in 0..3 {
            for &l in &order {
                let g = ring.acquire(l);
                assert_eq!(g.layer(), l);
                std::thread::sleep(compute);
            }
        }
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let sum_compute = 9.0 * 80.0;
        assert!(
            wall >= sum_compute - 20.0,
            "compute alone is {sum_compute} ms: {wall}"
        );
        assert!(
            wall < sum_compute + 2.0 * 40.0 + 120.0,
            "transfers must hide under compute (serial would be ~1080 ms): {wall}"
        );
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "at most `slots` payloads may be alive at once"
        );
    }

    /// (b) tx > compute: the link paces the ring, wall ~= sum of tx.
    #[test]
    fn a_transfer_bound_ring_paces_at_the_link() {
        let order = vec![0usize, 1, 2];
        let (stage, _) = fake(60);
        let ring = Ring::new(stage, 2, order.clone());
        let compute = Duration::from_millis(25);
        let t = Instant::now();
        for _token in 0..2 {
            for &l in &order {
                let g = ring.acquire(l);
                assert_eq!(g.layer(), l);
                std::thread::sleep(compute);
            }
        }
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let sum_tx = 6.0 * 60.0;
        assert!(
            wall >= sum_tx - 20.0,
            "uploads are serial on one link: {wall}"
        );
        assert!(
            wall < sum_tx + 100.0,
            "transfer-bound wall must track sum-of-tx (~{sum_tx} ms), \
             not sum of tx + compute (~510 ms): {wall}"
        );
    }

    /// (c) correctness across token wraps: acquire(l) always returns l's
    /// payload, indefinitely (the order repeats every token).
    #[test]
    fn acquire_returns_the_scheduled_layer_across_token_wraps() {
        let order = vec![0usize, 1, 2, 3, 4];
        let (stage, peak) = fake(1);
        let ring = Ring::new(stage, 2, order.clone());
        for _token in 0..3 {
            for &l in &order {
                let g = ring.acquire(l);
                assert_eq!((*g).layer, l, "payload must belong to the acquired layer");
                assert_eq!(g.layer(), l);
            }
        }
        assert!(peak.load(Ordering::SeqCst) <= 2);
        assert!(
            ring.blocks_landed() >= 15,
            "the default upload_blocks reports one block per upload"
        );
    }

    /// (d) dropping the ring mid-token (guards released or not, prefetcher
    /// mid-upload) must not deadlock. A watchdog turns a hang into a
    /// failure instead of a stuck CI job.
    #[test]
    fn dropping_the_ring_mid_token_never_deadlocks() {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (stage, _) = fake(50);
            let ring = Ring::new(stage, 2, vec![0, 1, 2]);
            let g0 = ring.acquire(0);
            drop(g0);
            let _g1 = ring.acquire(1); // still held when the ring drops
            drop(ring); // prefetcher is mid-upload or blocked on slots
            done_tx.send(()).expect("watchdog listens");
        });
        done_rx
            .recv_timeout(Duration::from_secs(20))
            .expect("dropping the ring mid-token deadlocked");
    }

    /// The prefetch thread goes through `upload_blocks`, so a blocked stage
    /// sees its per-block callback — `upload` itself must never be called
    /// once a stage overrides the blocked path.
    struct BlockedStage {
        blocks: usize,
        per_block: Duration,
        events: Arc<Mutex<Vec<Block>>>,
    }

    impl SlotStage for BlockedStage {
        type Payload = usize;
        fn upload(&self, _layer: usize) -> usize {
            unreachable!("the ring must upload through upload_blocks");
        }
        fn upload_blocks(&self, layer: usize, on_block: &mut dyn FnMut(Block)) -> usize {
            for index in 0..self.blocks {
                std::thread::sleep(self.per_block);
                let b = Block {
                    layer,
                    index,
                    of: self.blocks,
                };
                self.events
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(b);
                on_block(b);
            }
            layer
        }
    }

    #[test]
    fn the_ring_prefetches_through_upload_blocks() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let stage = BlockedStage {
            blocks: 4,
            per_block: Duration::from_millis(2),
            events: Arc::clone(&events),
        };
        let ring = Ring::new(stage, 2, vec![0, 1]);
        for _token in 0..2 {
            for l in 0..2usize {
                let g = ring.acquire(l);
                assert_eq!(*g, l, "payload is the uploaded layer");
            }
        }
        assert!(ring.blocks_landed() >= 16, "4 uploads x 4 blocks");
        let seen = events.lock().unwrap_or_else(|e| e.into_inner());
        for chunk in seen.chunks(4).take(4) {
            assert_eq!(chunk.len(), 4);
            for (i, b) in chunk.iter().enumerate() {
                assert_eq!(b.index, i);
                assert_eq!(b.of, 4);
                assert_eq!(b.layer, chunk[0].layer);
            }
        }
    }

    /// The floor-collapse measurement: consuming a blocked upload block by
    /// block (transfer thread feeding a compute thread — the mechanism a
    /// block-granular executor runs inside one slot) lands near
    /// `pipelined_layer_ms`, far from the serial tx + compute.
    #[test]
    fn block_pipelining_approaches_the_predicted_floor() {
        let blocks = 8usize;
        let stage = BlockedStage {
            blocks,
            per_block: Duration::from_millis(30), // tx total 240 ms
            events: Arc::new(Mutex::new(Vec::new())),
        };
        let block_compute = Duration::from_millis(15); // compute total 120 ms
        let (btx, brx) = std::sync::mpsc::channel::<Block>();
        let t = Instant::now();
        let producer = std::thread::spawn(move || {
            let _ = stage.upload_blocks(0, &mut |b| btx.send(b).expect("consumer listens"));
        });
        for _ in 0..blocks {
            let b = brx.recv().expect("producer sends every block");
            assert_eq!(b.of, blocks);
            std::thread::sleep(block_compute);
        }
        producer.join().expect("producer exits cleanly");
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let predicted = pipelined_layer_ms(240.0, 120.0, blocks, 0.0); // 255 ms
        assert!(wall >= 240.0 - 20.0, "the link is serial: {wall}");
        assert!(
            wall < predicted + 65.0,
            "block pipelining must approach {predicted} ms (serial is 360 ms): {wall}"
        );
    }

    /// TensorStage plumbing on the CPU device: the ring hands back the
    /// right layer's tensor with the right shape (no GPU required; the
    /// device-transfer semantics ride `move_to`, tested in backend.rs).
    #[test]
    fn tensor_stage_hands_over_device_tensors() {
        let cpu = crate::backend::cpu_device();
        let host: Vec<Tensor<2>> = (0..2)
            .map(|i| {
                Tensor::<2>::from_data(
                    burn::tensor::TensorData::new(vec![i as f32; 6], [2, 3]),
                    (&cpu, crate::backend::float_dtype(&cpu)),
                )
            })
            .collect();
        let ring = Ring::new(TensorStage::new(host, cpu.clone()), 2, vec![0, 1]);
        for _token in 0..2 {
            for l in 0..2usize {
                let g = ring.acquire(l);
                assert_eq!(g.dims(), [2, 3]);
                let data = g
                    .clone()
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .expect("payload reads back");
                assert!(
                    data.iter().all(|v| close(f64::from(*v), l as f64)),
                    "layer {l} must get layer {l}'s tensor"
                );
            }
        }
    }
}
