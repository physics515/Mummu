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

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
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

static PHASE: AtomicU8 = AtomicU8::new(Phase::Idle as u8);
static DONE: AtomicU64 = AtomicU64::new(0);
static EXPECTED: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static STARTED_MS: AtomicU64 = AtomicU64::new(0);
static UPDATED_MS: AtomicU64 = AtomicU64::new(0);
static GENERATION: AtomicU64 = AtomicU64::new(0);

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

/// Enter a counted phase: `expected` items, zero done, the clock restarted.
///
/// `expected` of 0 means "this phase has no count" and renders indeterminate.
///
/// Restarting the clock is deliberate. `started_ms` marks the beginning of the
/// **current phase**, not of the whole load, so the rate and the ETA describe
/// the work actually being done: a load that spent 40 s planning the fit and
/// then began reading reports the read's 162 MB/s, not an average diluted by
/// the planning. See [`Snapshot::elapsed_s`].
pub fn begin(phase: Phase, expected: u64) {
    let now = now_ms();
    DONE.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    EXPECTED.store(expected, Relaxed);
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

/// Switch to an uncounted phase.
///
/// The previous phase's counts are dropped rather than carried: `673/851
/// tensors` under a label that says "warming" would be a lie the bar tells
/// with a straight face. An uncounted phase renders indeterminate.
pub fn phase(p: Phase) {
    let now = now_ms();
    DONE.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    EXPECTED.store(0, Relaxed);
    STARTED_MS.store(now, Relaxed);
    UPDATED_MS.store(now, Relaxed);
    PHASE.store(p as u8, Relaxed);
}

/// The load succeeded: the model is resident and answering.
pub fn finish() {
    phase(Phase::Ready);
}

/// Back to nothing-in-flight. What a failed or abandoned load leaves behind.
pub fn idle() {
    phase(Phase::Idle);
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
pub fn evicted() {
    let generation = GENERATION.load(Relaxed);
    // Only a `Ready` claim needs retracting. A working phase owns this state,
    // and `Idle` already IS the answer — re-asserting it would restart the
    // clock every 5 seconds for the watcher that calls this on a timer.
    if Phase::from_u8(PHASE.load(Relaxed)) != Phase::Ready {
        return;
    }
    // A load that began while we were deciding owns the state now.
    if GENERATION.load(Relaxed) == generation {
        idle();
    }
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
}

impl Load {
    /// Begin a load for `model`, taking ownership of the progress state.
    ///
    /// Bumps the generation, so any snapshot taken from an older load can be
    /// recognised as stale rather than shown as current, and so an older
    /// guard's `Drop` becomes a no-op.
    #[must_use]
    pub fn begin(model_name: &str) -> Self {
        let generation = GENERATION.fetch_add(1, Relaxed) + 1;
        {
            let mut m = model();
            m.clear();
            m.push_str(model_name);
        }
        begin(Phase::Loading, 0);
        Self {
            generation,
            done: std::sync::atomic::AtomicBool::new(false),
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
    pub fn begin_phase(&self, p: Phase, expected: u64) {
        if self.current() {
            begin(p, expected);
        }
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
        if !self.done.load(Relaxed) && self.current() {
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

/// Read the current state. Cheap: seven relaxed loads and one short `String`
/// clone, so a status endpoint may call it whenever it likes.
#[must_use]
pub fn snapshot() -> Snapshot {
    Snapshot {
        generation: GENERATION.load(Relaxed),
        phase: Phase::from_u8(PHASE.load(Relaxed)),
        model: model().clone(),
        done: DONE.load(Relaxed),
        expected: EXPECTED.load(Relaxed),
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

        load.begin_phase(Phase::Loading, 851);
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
        load.begin_phase(Phase::Loading, 851);
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
            load.begin_phase(Phase::Loading, 851);
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
        fresh.begin_phase(Phase::Loading, 100);
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
            old.begin_phase(Phase::Loading, 851);
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
        load.begin_phase(Phase::Loading, 851);
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
