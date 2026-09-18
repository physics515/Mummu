//! A structured, process-global view of "what is the loader doing right now".
//!
//! # Why this is not log scraping
//!
//! The loaders already narrate themselves — `[mummu] load: 673/851 tensors —
//! 14.46 GiB off the pack in 91s (162 MB/s)` — and a UI *could* read that back
//! out of the log ring. It would be a bad bar. That line is printed **once
//! every 15 seconds**, deliberately, because it goes to stderr and to
//! `docker logs`, where one line per tensor would be unreadable. A bar fed
//! from it would sit still for a quarter of a minute and then jump, which is
//! exactly the "is it hung?" feeling the bar exists to remove. It would also
//! be a parser over free text that nobody promised to keep stable.
//!
//! So the numbers live here, in atomics, updated once per tensor. The prints
//! stay exactly as they are — the log page still wants them.
//!
//! # Why atomics and why `Relaxed`
//!
//! [`advance`] runs once per tensor, on the same loop that reads a tensor off
//! a spinning disk and dequantizes it. It must be free, and it is: three
//! `Relaxed` stores, no fence, no lock, no allocation. `Relaxed` is the
//! *correct* ordering here rather than a shortcut, because of what a reader
//! needs:
//!
//! * **No field may be torn.** Each one is a single atomic of its own width,
//!   so every read returns some value that was actually written. That holds
//!   under `Relaxed`, which is all Rust's atomics ever permit.
//! * **Fields may disagree with each other by microseconds.** A snapshot can
//!   catch `done` from one tensor and `bytes` from the next. In a progress bar
//!   that is invisible — the bar is redrawn once a second and the two numbers
//!   move together at roughly a megabyte a millisecond. Buying cross-field
//!   consistency would mean a lock or a sequence-lock on the loader's hot
//!   loop, to fix something no human eye can see.
//!
//! The one thing a reader must NOT do is show a *finished or abandoned* load
//! as live, which is not an ordering problem but a lifecycle one — see
//! [`Load`] and [`Snapshot::generation`].

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

/// What the runner is doing. A phase that carries counts renders as a
/// determinate bar; one that cannot know its own size renders indeterminate,
/// which is honest rather than a fake percentage that stalls at 90%.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    /// Nothing is loading. The only phase a *failed* load may leave behind.
    Idle = 0,
    /// Reading weights off the pack/checkpoint. Counted: tensors and bytes.
    Loading = 1,
    /// Building the VNNI host twins from resident weights. Uncounted — the
    /// work is per-projection and the count is not known before it starts.
    Packing = 2,
    /// First forward pass: kernel compilation, autotune, cache warm-up.
    /// Uncounted; it is one pass whose duration is the unknown.
    Warming = 3,
    /// Resident and answering. Held until the next load begins.
    Ready = 4,
}

impl Phase {
    /// The wire name, as the `/api/logs` status object spells it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Loading => "loading",
            Self::Packing => "packing",
            Self::Warming => "warming",
            Self::Ready => "ready",
        }
    }

    const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Loading,
            2 => Self::Packing,
            3 => Self::Warming,
            4 => Self::Ready,
            // Anything else cannot happen (only this module stores the byte),
            // and Idle is the safe reading if it somehow did: a bar that is
            // not shown beats a bar stuck on a phase nobody is in.
            _ => Self::Idle,
        }
    }

    /// Is a load actually in flight? `Ready` is a resting state, not work.
    #[must_use]
    pub const fn is_working(self) -> bool {
        matches!(self, Self::Loading | Self::Packing | Self::Warming)
    }
}

/// What a counted phase is counting.
///
/// # Why the count carries its unit
///
/// Three different loops feed this bar and they count three different things:
/// the pack/GGUF trunk loaders count TENSORS, the partitioned loader's second
/// pass counts LAYERS (a cluster count would move in jumps of 30), and the
/// tiered loader counts EXPERTS. Both pages printed the word "tensors" under
/// all three, so a 27B whose remote FFN pass says `12/64` claimed to be 12
/// tensors into an 851-tensor model — a number that is not merely imprecise,
/// it is the wrong quantity, and it is the number the ETA beside it is
/// extrapolated from. The unit travels with the count so the label cannot
/// drift from what the loop is actually doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Unit {
    /// Individual parameters assigned off the pack/checkpoint.
    Tensors = 0,
    /// Whole transformer layers, for a loop whose body is one layer.
    Layers = 1,
    /// MoE experts, for the tiered loader's `layers * experts_per_layer`.
    Experts = 2,
}

impl Unit {
    /// The wire name, and the word the pages print after the count.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tensors => "tensors",
            Self::Layers => "layers",
            Self::Experts => "experts",
        }
    }

    const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Layers,
            2 => Self::Experts,
            // Only this module stores the byte, and tensors is what the
            // overwhelming majority of loads count.
            _ => Self::Tensors,
        }
    }
}

static PHASE: AtomicU8 = AtomicU8::new(Phase::Idle as u8);
static DONE: AtomicU64 = AtomicU64::new(0);
static EXPECTED: AtomicU64 = AtomicU64::new(0);
static UNIT: AtomicU8 = AtomicU8::new(Unit::Tensors as u8);
static BYTES: AtomicU64 = AtomicU64::new(0);
static STARTED_MS: AtomicU64 = AtomicU64::new(0);
static UPDATED_MS: AtomicU64 = AtomicU64::new(0);
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// Which counted phase of THIS load the bar is on; see [`Snapshot::step`].
static STEP: AtomicU64 = AtomicU64::new(0);

/// The model a load is for.
///
/// A `String` cannot be an atomic, so it sits behind its own mutex — which is
/// fine precisely because it is NOT on the hot path: it is written once per
/// load (in [`Load::begin`]) and read once per snapshot, at most once a second
/// by the status endpoint's cache. The per-tensor path never touches it.
static MODEL: Mutex<String> = Mutex::new(String::new());

fn model() -> MutexGuard<'static, String> {
    // A poisoned name is still a perfectly readable name, and refusing to show
    // a progress bar because a *label* lock was poisoned would be absurd.
    MODEL.lock().unwrap_or_else(|e| e.into_inner())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

// ---------------------------------------------------------------------------
// The loader-facing API. Everything here is a handful of relaxed stores.
// ---------------------------------------------------------------------------

/// Enter a counted phase: `expected` items of `unit`, zero done, the clock
/// restarted.
///
/// `expected` of 0 means "this phase has no count" and renders indeterminate.
///
/// Restarting the clock is deliberate. `started_ms` marks the beginning of the
/// **current phase**, not of the whole load, so the rate and the ETA describe
/// the work actually being done: a load that spent 40 s planning the fit and
/// then began reading reports the read's 162 MB/s, not an average diluted by
/// the planning. See [`Snapshot::elapsed_s`].
///
/// A counted phase entered while this load already had one is a **restart**:
/// it bumps [`Snapshot::step`], because the tiered and partitioned paths load
/// in two counted passes and the second one legitimately begins at 0 with a
/// different denominator. Without the step the bar simply appears to go
/// backwards, which is what a hang looks like to the person watching.
pub fn begin(phase: Phase, expected: u64, unit: Unit) {
    let now = now_ms();
    DONE.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    EXPECTED.store(expected, Relaxed);
    UNIT.store(unit as u8, Relaxed);
    // Only a phase that can actually draw a bar counts as a step. `Load::begin`
    // opens every load with an uncounted `Loading`, and calling that step 1
    // would make the first real bar step 2 on every load there has ever been.
    if expected > 0 {
        STEP.fetch_add(1, Relaxed);
    }
    STARTED_MS.store(now, Relaxed);
    UPDATED_MS.store(now, Relaxed);
    PHASE.store(phase as u8, Relaxed);
}

/// Report progress inside the current phase: `done` items, `bytes` read so
/// far (both cumulative, not deltas).
///
/// Called once per tensor. Three relaxed stores and a clock read — see the
/// module header for why that is the right cost and the right ordering.
pub fn advance(done: u64, bytes: u64) {
    DONE.store(done, Relaxed);
    BYTES.store(bytes, Relaxed);
    UPDATED_MS.store(now_ms(), Relaxed);
}

/// Drop the counts and restart the clock, leaving the phase itself alone.
///
/// Shared by [`phase`] and [`evicted`], which differ only in how they are
/// allowed to decide the phase — not in what a phase change does to the
/// numbers under it.
fn reset_counts() {
    let now = now_ms();
    DONE.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    EXPECTED.store(0, Relaxed);
    STARTED_MS.store(now, Relaxed);
    UPDATED_MS.store(now, Relaxed);
}

/// Switch to an uncounted phase.
///
/// The previous phase's counts are dropped rather than carried: `673/851
/// tensors` under a label that says "warming" would be a lie the bar tells
/// with a straight face. An uncounted phase renders indeterminate.
pub fn phase(p: Phase) {
    reset_counts();
    PHASE.store(p as u8, Relaxed);
}

/// The load succeeded: the model is resident and answering.
pub fn finish() {
    phase(Phase::Ready);
}

/// Back to nothing-in-flight. What a failed or abandoned load leaves behind.
pub fn idle() {
    phase(Phase::Idle);
    STEP.store(0, Relaxed);
    model().clear();
}

/// The resident model is gone: something dropped it out from under this state.
///
/// [`Phase::Ready`] is not a memory of a load that once succeeded, it is a
/// claim that a model is resident **now** — so an eviction makes it a lie, and
/// a visible one: after `POST /api/unload` the page read `ready — qwen3.5-2b`
/// beside a RAM gauge that had just dropped 7 GiB, which is the same "the
/// display disagrees with the machine" confusion the whole panel exists to
/// end. `finish()` has an owner (the load that succeeded); losing residency
/// does not, so it gets a free function of its own.
///
/// # Why a working phase is left alone
///
/// A load in flight owns this state through its [`Load`] guard, and blanking
/// a bar that is legitimately moving would be the worse lie of the two. The
/// caller's own structure normally rules that out — mummu-serve's eviction
/// path cannot take the model slot while a load holds it — but "normally" is
/// not a thing to leave a global on, and the guard's `Drop` settles a load
/// that really is abandoned anyway.
///
/// # Why the retraction proves what it is retracting
///
/// A generation re-check around the phase read does NOT make this safe, and
/// the interleaving that breaks it is ordinary. [`Load::begin`] used to bump
/// the generation first and publish the name and the phase after, and an
/// eviction deciding across that gap read the NEW generation, the OLD `Ready`
/// phase, re-read the same new generation and concluded nothing had moved. It
/// then blanked the incoming load's model name — for the whole load, and for
/// as long as the model stayed resident after it, because nothing
/// re-publishes a name that was already published.
///
/// Two things close it, and both are needed because the claim being retracted
/// is spread over two variables:
///
/// * The **phase** is read and retracted in ONE `compare_exchange`. There is
///   no window between deciding that a `Ready` claim exists and replacing it,
///   so a `Load::ready()` landing alongside cannot be silently overwritten.
/// * The **name** is retracted under the same lock `Load::begin` publishes
///   its whole state under. A load that begins here either finishes
///   publishing before this reads the phase (which is then `Loading`, and
///   this declines) or starts after the name is cleared (and its own
///   `push_str` is the last word). There is no ordering in between.
pub fn evicted() {
    // Taken FIRST and held across the whole decision — see above. Cheap: the
    // callers are `POST /api/unload` and a 5 s watcher, never the hot path.
    let mut name = model();
    // Only a `Ready` claim needs retracting. A working phase owns this state,
    // and `Idle` already IS the answer — re-asserting it would restart the
    // clock every 5 seconds for the watcher that calls this on a timer.
    if PHASE
        .compare_exchange(Phase::Ready as u8, Phase::Idle as u8, Relaxed, Relaxed)
        .is_err()
    {
        return;
    }
    // Past the exchange the state is ours: the phase already reads `idle`,
    // and an idle phase's counts are not rendered by anything, so clearing
    // them a few nanoseconds later cannot be seen.
    reset_counts();
    STEP.store(0, Relaxed);
    name.clear();
}

// ---------------------------------------------------------------------------
// The lifecycle guard.
// ---------------------------------------------------------------------------

/// Owns one load's claim on the global progress state.
///
/// # Why a guard rather than a `finish()` call at the end
///
/// A load can end four ways and only one of them reaches the bottom of the
/// function: it succeeds, it returns an `Err` (a missing tensor, a pack whose
/// count disagrees with the architecture), it **panics** (OOM in a kernel, an
/// `unreachable!` — both have happened in production), or the task driving it
/// is **cancelled** because the browser tab went away mid-load. Three of those
/// four unwind or drop straight past any bookkeeping written at the end of the
/// happy path, and each one would leave the phase pinned at `Loading` with a
/// bar that creeps and never completes, forever, for every client — a worse
/// lie than no bar at all.
///
/// `Drop` runs on all four. So the rule is: the guard puts the state back to
/// [`Phase::Idle`] unless [`Load::ready`] consumed it first.
///
/// The [`generation`](Snapshot::generation) check inside `Drop` is what makes
/// that safe when loads overlap: if a newer load has already called
/// [`Load::begin`], this guard's cleanup would be stomping on someone else's
/// live progress, so it does nothing at all.
#[derive(Debug)]
pub struct Load {
    generation: u64,
    /// Set by [`Load::ready`], which then forgets the guard's reset duty.
    ///
    /// An `AtomicBool` rather than a `Cell` so the guard stays `Sync`: the
    /// server marks the load ready from inside the decode callback, which is
    /// held across an `.await` in a `tokio::spawn`ed future — and a future is
    /// only `Send` if the `&Load` it holds is, which is only true if `Load` is
    /// `Sync`. A `Cell` here compiles until exactly that call site.
    done: std::sync::atomic::AtomicBool,
    /// Set by [`Load::resident`]: the weights landed, so a `Drop` from here on
    /// settles on [`Phase::Ready`] instead of [`Phase::Idle`].
    landed: AtomicBool,
}

impl Load {
    /// Begin a load for `model`, taking ownership of the progress state.
    ///
    /// Bumps the generation, so any snapshot taken from an older load can be
    /// recognised as stale rather than shown as current, and so an older
    /// guard's `Drop` becomes a no-op.
    ///
    /// # Publishing is one step, not three
    ///
    /// The generation, the name and the phase all go out under the name lock,
    /// and nothing observes a mixture of them. An eviction deciding
    /// concurrently is the reason — see [`evicted`], where reading a new
    /// generation beside the previous load's `Ready` phase is exactly how the
    /// incoming load lost its name.
    #[must_use]
    pub fn begin(model_name: &str) -> Self {
        let mut m = model();
        let generation = GENERATION.fetch_add(1, Relaxed) + 1;
        m.clear();
        m.push_str(model_name);
        // A new load's bar starts at step zero whatever the last one reached.
        STEP.store(0, Relaxed);
        begin(Phase::Loading, 0, Unit::Tensors);
        drop(m);
        Self {
            generation,
            done: std::sync::atomic::AtomicBool::new(false),
            landed: AtomicBool::new(false),
        }
    }

    /// The generation this guard owns. A [`Snapshot`] carrying a different one
    /// describes a different load.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Move to `p`, unless a newer load has taken over.
    pub fn phase(&self, p: Phase) {
        if self.current() {
            phase(p);
        }
    }

    /// Enter a counted phase, unless a newer load has taken over.
    pub fn begin_phase(&self, p: Phase, expected: u64, unit: Unit) {
        if self.current() {
            begin(p, expected, unit);
        }
    }

    /// The weights are resident: the load itself COMPLETED, whatever happens
    /// to the request that paid for it.
    ///
    /// # Why this is not the same as [`ready`](Self::ready)
    ///
    /// `ready` fires from the first decoded token, and between a model
    /// landing in the slot and that token there are at least three exits that
    /// leave the model resident and answering: the prompt fails to encode, it
    /// encodes to zero tokens, or the request is cancelled because the browser
    /// tab went away. Every one of them dropped the guard, which reset the
    /// state to `idle` / `model: null` — a server claiming to hold nothing
    /// while it holds a 27B and answers the next request warm. That is the
    /// `ready`-after-eviction lie with the sign flipped, and it is worse:
    /// eviction at least has a watcher that can correct it.
    ///
    /// So the phase after this is settled by what the LOAD did, not by what
    /// the request did. The bar still moves to `warming` and still finishes on
    /// the first token; it just no longer forgets a model that is in memory.
    pub fn resident(&self) {
        self.landed.store(true, Relaxed);
    }

    /// The load succeeded: settle on [`Phase::Ready`] and disarm the reset.
    ///
    /// Takes `&self` rather than consuming the guard because the caller that
    /// knows the load is over is a `FnMut` callback (the first decoded token),
    /// which cannot move out of its captures. Calling it twice is harmless.
    pub fn ready(&self) {
        self.done.store(true, Relaxed);
        if self.current() {
            finish();
        }
    }

    fn current(&self) -> bool {
        GENERATION.load(Relaxed) == self.generation
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        // Only if we are still the live load: a newer `Load::begin` means this
        // guard is the *old* one unwinding behind a load already in progress,
        // and resetting would blank a bar that is legitimately moving.
        if self.done.load(Relaxed) || !self.current() {
            return;
        }
        if self.landed.load(Relaxed) {
            // The weights are in memory. A request that never reached a token
            // does not un-load them — see `resident`.
            finish();
        } else {
            idle();
        }
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// One reading of the loader's state. Fields are individually consistent; see
/// the module header on why they may be microseconds apart from each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// Which load this describes. Bumped by every [`Load::begin`], so a reader
    /// holding an older value knows its numbers belong to a load that is over.
    pub generation: u64,
    pub phase: Phase,
    /// The model being loaded, empty when nothing is.
    pub model: String,
    pub done: u64,
    /// 0 means "no count" — render indeterminate, not 0%.
    pub expected: u64,
    /// What `done` and `expected` are counting. Meaningless when
    /// `expected == 0`, which is a phase with nothing to label.
    pub unit: Unit,
    /// Which counted phase of this load the bar is on: 0 before the first
    /// one, 1 for the ordinary single-pass load, 2 for the second pass of a
    /// tiered or partitioned load.
    ///
    /// The page shows it only past 1, where it is the difference between "the
    /// bar restarted because a second pass began" and "the bar went
    /// backwards", which is what a hang looks like from a chair.
    pub step: u64,
    pub bytes: u64,
    /// Unix ms when the CURRENT PHASE began.
    pub started_ms: u64,
    /// Unix ms of the last [`advance`] or phase change.
    pub updated_ms: u64,
}

impl Snapshot {
    /// Seconds spent in the current phase, or `None` when no load has ever run
    /// (and so there is no clock to read).
    ///
    /// Measured from wall-clock unix millis rather than an `Instant`, because
    /// the snapshot crosses a process boundary as JSON and a monotonic tick
    /// means nothing to a browser. A clock step backwards is clamped to 0
    /// rather than producing a negative elapsed that would make the rate
    /// negative and the ETA nonsense.
    #[must_use]
    pub fn elapsed_s(&self) -> Option<f64> {
        if self.started_ms == 0 {
            return None;
        }
        let end = self.updated_ms.max(now_ms());
        Some(end.saturating_sub(self.started_ms) as f64 / 1000.0)
    }

    /// Bytes per second over the current phase, or `None` when nothing has
    /// been read yet or no measurable time has passed.
    ///
    /// `None` rather than 0 on purpose: "we have not measured a rate" and
    /// "the disk is delivering nothing" look identical as a number and must
    /// not look identical in the UI.
    #[must_use]
    pub fn rate_bps(&self) -> Option<f64> {
        let elapsed = self.elapsed_s()?;
        if self.bytes == 0 || elapsed <= 0.0 {
            return None;
        }
        Some(self.bytes as f64 / elapsed)
    }

    /// Seconds left, extrapolated from the share of items already done.
    ///
    /// Requires all three of: a total (`expected > 0`), at least one item done
    /// (a rate of zero extrapolates to infinity), and a measurable elapsed
    /// time. Any of those missing is `None`, and the UI must print nothing
    /// rather than invent a number.
    ///
    /// Extrapolated from ITEMS, not bytes: tensors are the only quantity whose
    /// total is known before the load starts. The byte total is not — a pack
    /// stores several precisions and only the chosen ones are read.
    #[must_use]
    pub fn eta_s(&self) -> Option<f64> {
        let elapsed = self.elapsed_s()?;
        if self.expected == 0 || self.done == 0 || elapsed <= 0.0 {
            return None;
        }
        let remaining = self.expected.saturating_sub(self.done) as f64;
        Some(elapsed * remaining / self.done as f64)
    }

    /// The fraction complete in `0.0..=1.0`, or `None` when this phase has no
    /// count and must render as an indeterminate animation.
    #[must_use]
    pub fn fraction(&self) -> Option<f64> {
        if self.expected == 0 {
            return None;
        }
        Some((self.done as f64 / self.expected as f64).clamp(0.0, 1.0))
    }
}

/// Read the current state. Cheap: nine relaxed loads and one short `String`
/// clone, so a status endpoint may call it whenever it likes.
#[must_use]
pub fn snapshot() -> Snapshot {
    Snapshot {
        generation: GENERATION.load(Relaxed),
        phase: Phase::from_u8(PHASE.load(Relaxed)),
        model: model().clone(),
        done: DONE.load(Relaxed),
        expected: EXPECTED.load(Relaxed),
        unit: Unit::from_u8(UNIT.load(Relaxed)),
        step: STEP.load(Relaxed),
        bytes: BYTES.load(Relaxed),
        started_ms: STARTED_MS.load(Relaxed),
        updated_ms: UPDATED_MS.load(Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state is process-wide and `cargo test` runs in parallel threads, so
    /// every test that writes to it takes this first. The arithmetic tests
    /// build `Snapshot` values by hand and need no lock at all.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn snap(done: u64, expected: u64, bytes: u64, elapsed_ms: u64) -> Snapshot {
        let now = now_ms();
        Snapshot {
            generation: 1,
            phase: Phase::Loading,
            model: "gemma3:27b".into(),
            done,
            expected,
            unit: Unit::Tensors,
            step: 1,
            bytes,
            started_ms: now.saturating_sub(elapsed_ms),
            updated_ms: now,
        }
    }

    // -- arithmetic --------------------------------------------------------

    /// The worked example from the owner's own load, to the numbers the status
    /// object is specified to carry.
    #[test]
    fn rate_and_eta_match_the_load_they_describe() {
        let s = snap(673, 851, 14_800_000_000, 91_000);
        let elapsed = s.elapsed_s().expect("a started phase has a clock");
        assert!((elapsed - 91.0).abs() < 1.0, "elapsed {elapsed}");
        let rate = s.rate_bps().expect("bytes were read");
        assert!(
            (rate - 162_600_000.0).abs() < 4_000_000.0,
            "~162 MB/s, got {rate}"
        );
        let eta = s.eta_s().expect("a total and a done count give an eta");
        // 91 s for 673 of 851 leaves 178 to go at the same pace.
        assert!((eta - 24.1).abs() < 1.0, "~24 s left, got {eta}");
        assert!((s.fraction().expect("counted") - 0.7908).abs() < 0.001);
    }

    /// Every way the arithmetic can be asked to divide by zero.
    #[test]
    fn missing_inputs_give_none_never_zero_and_never_infinity() {
        // No total: the phase is uncounted (Packing/Warming).
        let uncounted = snap(0, 0, 1 << 20, 5_000);
        assert_eq!(uncounted.eta_s(), None, "no total, no eta");
        assert_eq!(
            uncounted.fraction(),
            None,
            "uncounted renders indeterminate"
        );
        assert!(uncounted.rate_bps().is_some(), "bytes still give a rate");

        // A total but nothing done yet: extrapolating from zero is infinity.
        let nothing_done = snap(0, 851, 0, 3_000);
        assert_eq!(nothing_done.eta_s(), None);
        assert_eq!(nothing_done.rate_bps(), None, "no bytes, no rate");
        assert_eq!(nothing_done.fraction(), Some(0.0));

        // The very first millisecond: elapsed rounds to zero.
        let instant = snap(4, 851, 1 << 20, 0);
        assert_eq!(instant.eta_s(), None, "no measurable time, no eta");
        assert_eq!(instant.rate_bps(), None);

        // No load has ever run: there is no clock at all.
        let never = Snapshot {
            started_ms: 0,
            updated_ms: 0,
            ..snap(0, 0, 0, 0)
        };
        assert_eq!(never.elapsed_s(), None);
        assert_eq!(never.rate_bps(), None);
        assert_eq!(never.eta_s(), None);
    }

    /// A count past its total (a loader that assigned more than the manifest
    /// predicted) must not produce a bar over 100% or a negative eta.
    #[test]
    fn an_overshooting_count_is_clamped_not_wrapped() {
        let over = snap(900, 851, 1 << 30, 10_000);
        assert_eq!(over.fraction(), Some(1.0));
        assert_eq!(over.eta_s(), Some(0.0), "saturating, never negative");
    }

    // -- lifecycle ---------------------------------------------------------

    #[test]
    fn a_load_reports_its_phases_and_settles_on_ready() {
        let _serial = serial();
        let load = Load::begin("qwen3.5-2b");
        assert_eq!(snapshot().phase, Phase::Loading);
        assert_eq!(snapshot().model, "qwen3.5-2b");

        load.begin_phase(Phase::Loading, 851, Unit::Tensors);
        advance(673, 14 << 30);
        let s = snapshot();
        assert_eq!((s.done, s.expected), (673, 851));
        assert_eq!(s.fraction().map(|f| (f * 100.0) as u32), Some(79));

        load.phase(Phase::Warming);
        let warming = snapshot();
        assert_eq!(warming.phase, Phase::Warming);
        assert_eq!(
            warming.fraction(),
            None,
            "the read's counts must not survive into a phase that has none"
        );

        load.ready();
        assert_eq!(snapshot().phase, Phase::Ready);
        idle();
    }

    /// A loader that gives up part way, shaped like the real one: it holds the
    /// guard, reports progress, then returns `Err` from the middle of its body
    /// — which is the completeness check qwen35's pack loader actually fails.
    fn a_load_that_gives_up() -> Result<(), &'static str> {
        let load = Load::begin("qwen3.8-27b");
        load.begin_phase(Phase::Loading, 851, Unit::Tensors);
        advance(300, 5 << 30);
        assert_eq!(snapshot().phase, Phase::Loading);
        Err("pack supplied 300 trunk tensors, the architecture needs 851")?;
        load.ready(); // never reached
        Ok(())
    }

    /// The whole reason the guard exists: three of the four ways a load can
    /// end never reach the end of the function.
    #[test]
    fn a_load_that_fails_returns_to_idle_instead_of_loading_forever() {
        let _serial = serial();
        let failed = a_load_that_gives_up();
        assert!(failed.is_err());
        assert_eq!(
            snapshot().phase,
            Phase::Idle,
            "an early return must not pin the bar at loading"
        );
        assert_eq!(snapshot().model, "", "and must not leave the name behind");
    }

    /// A panic unwinds past every line of cleanup a function could have
    /// written at its end. `Drop` is the only thing that still runs.
    #[test]
    fn a_load_that_panics_returns_to_idle() {
        let _serial = serial();
        let panicked = std::panic::catch_unwind(|| {
            let load = Load::begin("qwen3.8-27b");
            load.begin_phase(Phase::Loading, 851, Unit::Tensors);
            advance(10, 1 << 30);
            panic!("CUDA_ERROR_OUT_OF_MEMORY");
        });
        assert!(panicked.is_err());
        assert_eq!(snapshot().phase, Phase::Idle);
    }

    /// A new load bumps the generation, and the old guard's cleanup must then
    /// keep its hands off — otherwise an abandoned load finishing unwinding a
    /// second after a new one started would blank a bar that is moving.
    #[test]
    fn a_new_load_takes_over_and_the_old_guards_drop_does_nothing() {
        let _serial = serial();
        let stale = Load::begin("old-model");
        let stale_gen = stale.generation();
        let fresh = Load::begin("new-model");
        assert!(fresh.generation() > stale_gen, "generation is monotonic");
        fresh.begin_phase(Phase::Loading, 100, Unit::Tensors);
        advance(42, 1 << 20);

        // The abandoned load unwinds now, one phase behind.
        stale.phase(Phase::Warming);
        drop(stale);

        let s = snapshot();
        assert_eq!(s.generation, fresh.generation());
        assert_eq!(s.phase, Phase::Loading, "the live load kept its phase");
        assert_eq!(s.done, 42, "and its counts");
        assert_eq!(s.model, "new-model");
        drop(fresh);
        assert_eq!(snapshot().phase, Phase::Idle);
    }

    /// The counts from a previous load must never be visible under the next
    /// one's label — `begin` zeroes them, it does not merely relabel.
    #[test]
    fn a_new_load_resets_a_stale_loads_counts() {
        let _serial = serial();
        {
            let old = Load::begin("old-model");
            old.begin_phase(Phase::Loading, 851, Unit::Tensors);
            advance(800, 14 << 30);
        }
        let fresh = Load::begin("new-model");
        let s = snapshot();
        assert_eq!((s.done, s.expected, s.bytes), (0, 0, 0));
        assert_eq!(s.model, "new-model");
        drop(fresh);
        idle();
    }

    /// The state `/logs` showed live after `POST /api/unload`: phase `ready`,
    /// model `qwen3.5-2b`, and 7 GiB less RAM held than the sentence implies.
    /// `Ready` is a claim about right now, so losing the model has to retract
    /// it — name included, or the next idle page still says whose model it is.
    #[test]
    fn an_eviction_retracts_the_ready_claim() {
        let _serial = serial();
        let load = Load::begin("qwen3.5-2b");
        load.ready();
        assert_eq!(snapshot().phase, Phase::Ready);
        assert_eq!(snapshot().model, "qwen3.5-2b");

        evicted();
        assert_eq!(
            snapshot().phase,
            Phase::Idle,
            "nothing is resident, so nothing is ready"
        );
        assert_eq!(snapshot().model, "", "and no model is named");
        drop(load);
    }

    /// Eviction is also reachable from a 5 s watcher that knows nothing about
    /// loads. It must not blank a bar that is moving — a load in flight owns
    /// this state and its guard is what ends it.
    #[test]
    fn an_eviction_leaves_a_load_in_flight_alone() {
        let _serial = serial();
        let load = Load::begin("gemma3:27b");
        load.begin_phase(Phase::Loading, 851, Unit::Tensors);
        advance(300, 5 << 30);

        evicted();
        let s = snapshot();
        assert_eq!(s.phase, Phase::Loading, "the live load kept its phase");
        assert_eq!((s.done, s.expected), (300, 851), "and its counts");
        assert_eq!(s.model, "gemma3:27b");

        // The same for the uncounted phases — `Warming` is work too.
        load.phase(Phase::Warming);
        evicted();
        assert_eq!(snapshot().phase, Phase::Warming);
        drop(load);
        assert_eq!(snapshot().phase, Phase::Idle);
    }

    /// MAJOR 1: the race the generation re-check could not see.
    ///
    /// `Load::begin` used to publish the generation BEFORE the name and the
    /// phase, so an eviction deciding across that gap read the new generation,
    /// the previous load's `Ready` phase, re-read the same new generation and
    /// retracted — leaving the incoming load nameless for its whole life and
    /// for as long as the model then stayed resident.
    ///
    /// A two-instruction window is not reproducible on demand, so this hammers
    /// it: one thread evicts continuously while this one starts loads, and
    /// every load checks that its own name survives the moment it is
    /// published. On the code this replaces it fails within a few hundred
    /// iterations; the publish being one locked step makes it impossible.
    #[test]
    fn a_load_that_begins_while_an_eviction_decides_keeps_its_name() {
        let _serial = serial();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let evictor = {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Relaxed) {
                    evicted();
                }
            })
        };

        for i in 0..2_000 {
            let name = format!("model-{i}");
            let load = Load::begin(&name);
            // Checked repeatedly: the retraction that steals the name can land
            // a moment after `begin` returns, which is precisely why reading
            // the state once and acting on it is not enough anywhere here.
            for _ in 0..32 {
                let s = snapshot();
                assert_eq!(s.model, name, "the incoming load lost its name");
                assert_eq!(s.phase, Phase::Loading, "and its phase");
                std::hint::spin_loop();
            }
            // Leave a `Ready` claim standing, so the next iteration's `begin`
            // races an eviction that has something to retract.
            load.ready();
            drop(load);
        }

        stop.store(true, Relaxed);
        evictor.join().expect("the evictor thread panicked");
        idle();
    }

    /// MAJOR 2: a load can succeed and never reach a token. The prompt fails
    /// to encode, it encodes to zero tokens, or the tab goes away mid-warm —
    /// and in all three the model IS resident and answers the next request.
    /// The guard used to reset to `idle` / `model: null` on every one of them,
    /// which is the eviction lie with the sign flipped.
    #[test]
    fn a_load_that_lands_but_never_decodes_still_reports_the_model_it_holds() {
        let _serial = serial();
        {
            let load = Load::begin("qwen3.5-2b");
            load.begin_phase(Phase::Loading, 851, Unit::Tensors);
            advance(851, 14 << 30);
            // The slot took the model; from here the load is over whatever
            // happens to the request that paid for it.
            load.resident();
            load.phase(Phase::Warming);
            // ...and the request dies here: `prompt encoded to zero tokens`.
        }
        let s = snapshot();
        assert_eq!(s.phase, Phase::Ready, "the weights are in memory");
        assert_eq!(s.model, "qwen3.5-2b", "and the page must name them");
        idle();
    }

    /// A load that never landed is still a load that failed: the settle is
    /// armed by residency, not by having got as far as trying.
    #[test]
    fn a_load_that_never_landed_still_returns_to_idle() {
        let _serial = serial();
        {
            let load = Load::begin("qwen3.8-27b");
            load.begin_phase(Phase::Loading, 851, Unit::Tensors);
            advance(300, 5 << 30);
        }
        assert_eq!(snapshot().phase, Phase::Idle);
        assert_eq!(snapshot().model, "");
    }

    /// MINOR 4: the tiered and partitioned paths load in two counted passes
    /// with different denominators and different units. The count has to say
    /// what it counts, and the restart has to be visible as a restart.
    #[test]
    fn a_second_counted_pass_carries_its_own_unit_and_says_it_restarted() {
        let _serial = serial();
        let load = Load::begin("qwen3.8-27b");
        assert_eq!(snapshot().step, 0, "an uncounted phase is not a step");

        load.begin_phase(Phase::Loading, 851, Unit::Tensors);
        advance(851, 14 << 30);
        let trunk = snapshot();
        assert_eq!((trunk.done, trunk.expected), (851, 851));
        assert_eq!(trunk.unit, Unit::Tensors);
        assert_eq!(trunk.step, 1, "the first bar of this load");

        // The remote FFN pass: 64 layers, from zero, with its own eta.
        load.begin_phase(Phase::Loading, 64, Unit::Layers);
        advance(12, 1 << 30);
        let second = snapshot();
        assert_eq!((second.done, second.expected), (12, 64));
        assert_eq!(second.unit, Unit::Layers, "12/64 LAYERS, not tensors");
        assert_eq!(
            second.step, 2,
            "a bar that restarts at 0 with no step reads as a hang"
        );

        // And the next load starts over: step is per-load, not per-process.
        let next = Load::begin("qwen3.5-2b");
        assert_eq!(snapshot().step, 0);
        drop(next);
        drop(load);
        idle();
    }

    /// The units are the words the pages print; they are wire values, so a
    /// rename here is a rename in two HTML files.
    #[test]
    fn unit_names_are_the_wire_names() {
        assert_eq!(Unit::Tensors.as_str(), "tensors");
        assert_eq!(Unit::Layers.as_str(), "layers");
        assert_eq!(Unit::Experts.as_str(), "experts");
    }

    /// Evicting when nothing was resident is a no-op, not a state change —
    /// the watcher calls this on a timer and must not churn the generation.
    #[test]
    fn evicting_nothing_changes_nothing() {
        let _serial = serial();
        idle();
        let before = snapshot();
        evicted();
        assert_eq!(snapshot(), before);
    }

    #[test]
    fn phase_names_are_the_wire_names() {
        assert_eq!(Phase::Idle.as_str(), "idle");
        assert_eq!(Phase::Loading.as_str(), "loading");
        assert_eq!(Phase::Packing.as_str(), "packing");
        assert_eq!(Phase::Warming.as_str(), "warming");
        assert_eq!(Phase::Ready.as_str(), "ready");
        assert!(Phase::Loading.is_working());
        assert!(!Phase::Ready.is_working(), "resident is not work");
        assert!(!Phase::Idle.is_working());
    }
}
