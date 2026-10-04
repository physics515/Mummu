//! Live placement of a layered model: which device each layer is on and at
//! what precision, decided by [`mummu::mix::joint`] from measurements and
//! re-decided while the model serves. The serve side of that solver: where
//! its inputs come from, and how its decisions reach a resident model.
//!
//! # No configured budget
//!
//! Every quantity below is measured or derived from the model's shapes. The
//! knobs that used to cap placement (`MUMMU_GPU_BUDGET_GB`,
//! `MUMMU_ACTIVATION_RESERVE_GB`, `MUMMU_VRAM_GUARD_GB`, `MUMMU_CTX`,
//! `MUMMU_LAYER_PREFIX`, the `MUMMU_VRAM_LIVE_BUDGET` gate) are gone: a static
//! cap is right on exactly one day on one box, and wrong in both directions
//! every other day — too big when a co-tenant arrives (an OOM), too small
//! when it leaves (layers on the host for nothing). Measured on the 27B
//! before this change: a fixed 9 GiB cap less a 1.36 GiB vision reserve held
//! for text-only traffic put 27 of 64 layers on a card with room for more.
//!
//! # The equations
//!
//! For the accelerator, with NVML's global `total`/`used` and our own
//! allocator's `reserved`/`in_use` (cubecl's pool, per device):
//!
//! ```text
//!   ambient      A  = used − reserved                          everything not ours
//!   guard        G  = Watermark(A)                             (1−α)-quantile envelope
//!                                                              + fragmentation slack,
//!                                                              ×1.5 per allocation failure
//!   capacity     K  = total − G                                what we may occupy
//!   non-layer    N  = in_use − Σ_{l on card} resident(l)       head, vision tower, residue
//!   for layers   C  = K − N                                    joint::Device::capacity
//!   working set  W  = act(min(ctx, chunk)) + T(ctx) + ε̂
//!                     + V·[tower needed, not resident]         joint::Device::fixed
//!   state        s_l = KV_l(ctx)  or  conv_l + S_l              joint::Layer::state_bytes
//! ```
//!
//! `act` is the widest live prefill buffer (three `[chunk, intermediate]`
//! f32 tensors through `SwiGLU`); `T` the card's other prefill transients,
//! which grow with the context ([`transient_bytes`]: a chunk's attention
//! scores over the whole context, and one projection dequantized for the
//! chunk's matmul); `ε̂` is the allocator residual the analytic terms do not
//! explain — measured after every generation as
//! `reserved − in_use_before − Σ s_l − act − T` and tracked as an envelope
//! over the last requests (vLLM's profiling pass does the same for its KV
//! budget: run the real workload and budget from the peak it produced, not
//! from a constant). Before the first measurement ε̂ is a prior, and it is
//! replaced by what the card shows. What it shows is mostly fragmentation:
//! the weights do not fill the pool's pages (measured 2026-10-04 on the
//! Bonsai 27B: 1.8 GiB reserved beyond in-use with nothing running). Each
//! generation hands its emptied pages back to the driver once measured, so
//! one long prompt's pages are not charged again to every request after it.
//!
//! `ctx` is the context actually being served: the request's own prompt plus
//! its token budget when a request is about to run, and the envelope of
//! recent requests' contexts when re-planning at idle.
//!
//! The host is device 0, the fallback: its capacity is 85% of
//! `MemAvailable` plus what our own host layers already hold.
//!
//! Rates are measured, not assumed: one real decode-shape projection is timed
//! on each device at each level it can run, under whatever contention the
//! machine has right now, and divided by that projection's pack bytes
//! (`s[d][p]`, joint's equation (1)). The host's is clamped to its DRAM
//! floor — the probe tensor fits L3, production streams every host layer per
//! token. The crossing cost κ is one timed activation round trip.
//!
//! # When it re-plans
//!
//! * **Before every request** (holding the model): with the request's own
//!   context and whether it needs the vision tower. If the resident placement
//!   would not fit beside that, joint repairs it first — the cheapest relief
//!   per byte — so the request never OOMs the card because it is long or
//!   carries an image.
//! * **Every 2 s at idle** (only if nobody holds the model): with the current
//!   ambient. Ambient grew → repair now. Ambient shrank, or a tower went
//!   idle and was dropped → improve, but only when (4) says the moved bytes
//!   pay for themselves over the horizon `H` (tokens served in the last
//!   hour), and only after the improvement has been wanted on three ticks in
//!   a row.
//!
//! Rates are scaled by how busy each device has been over the last ten
//! minutes, and an improvement must fit the least room each device left in
//! that time — the settled view, [`SETTLE_WINDOW`]. Only a repair answers
//! the card as it is this moment. Planned on the last few seconds, the
//! placement chased load that flips every minute on the reference box.
//!
//! Moves are applied releases-first: layers leaving the card (and in-place
//! demotions) go before arrivals, the allocator returns the freed pages to
//! the driver, and only then do layers arrive. An improvement moves at most
//! [`IMPROVE_STEP`] layers per tick so a request arriving mid-rebalance waits
//! for seconds, not for the whole move — and each step is a placement that
//! is faster on its own ([`joint::step`]), since the next tick may want
//! something else.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mummu::mix::joint;
use mummu::mix::{Kind, QuantPolicy};
use mummu::models::qwen35;
use mummu::pack::{Pack, Precision, Role, TensorEntry};
use mummu_num::{f64_from_u64, f64_from_usize, trunc_u64};

use super::{AnyLm, BackendChoice, Loaded, SLOT, device_of, gib, label_of, layer_index};

/// How often the idle rebalancer looks. Two seconds: a co-tenant that
/// starts wanting the card (a game launching) is answered within a tick of
/// `sysmon` deciding so, and the guard is fed at this cadence (see
/// [`GUARD_PERIOD`]).
const TICK: Duration = Duration::from_secs(2);
/// Tokens an IDLE improvement is priced over, at least.
///
/// `H` is the tokens served in the last hour, floored at 256 — on a server
/// that has been quiet, (4) then prices "take the card back" at 256 tokens
/// of saving against a full pack re-read, and the re-read wins: an idle
/// server held 30 layers where 36 fit, indefinitely (ROADMAP P6). At idle the
/// re-read costs nobody anything but disk time, and the rule is to use all
/// the capacity nobody else wants — so idle improvements are priced over a
/// long horizon and a free card is taken back, two layers a tick.
const IDLE_HORIZON_TOKENS: f64 = 65_536.0;
/// The guard is fed at most this often, so its window is a TIME window: it
/// used to be fed on every capacity call, which made "120 samples" mean
/// anything from ten minutes to ten seconds.
const GUARD_PERIOD: Duration = Duration::from_secs(2);
/// Ticks an improvement must be wanted on before it starts.
const IMPROVE_DWELL: u32 = 3;
/// Layers an improvement may move per tick.
pub(super) const IMPROVE_STEP: usize = 2;
/// Fraction of token time precision upgrades may cost (joint `tolerance`).
///
/// Wider than the probes' noise, so that where two levels run about as fast
/// the better one wins rather than whichever the last probe favoured. On
/// this card the 2B's Q8 and Q4 decode within a few percent of each other
/// (Q8 at the memory's bandwidth, Q4 at its unpacking), and at 1 % the
/// choice flipped from load to load — at Q4 the 2B misbehaved (a think
/// block it was told not to open). A model stored at Q4 has nothing to
/// upgrade to, so the big quantized checkpoints are untouched.
const PRECISION_TOLERANCE: f64 = 0.10;
/// How long a resident vision tower may sit unused before its VRAM goes back
/// to layers.
const TOWER_IDLE: Duration = Duration::from_secs(600);
/// The window the token horizon `H` counts over.
const HORIZON_WINDOW: Duration = Duration::from_secs(3600);
/// Bound on the NVML read (the driver can wedge; see `status`).
const READ_BUDGET: Duration = Duration::from_millis(500);
/// Requests the working-set and context envelopes remember.
const ENVELOPE: usize = 8;

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

/// One reading of the accelerator, our own share separated out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Card {
    pub total: u64,
    pub used: u64,
    /// cubecl pool: bytes it has taken from the driver.
    pub reserved: u64,
    /// cubecl pool: bytes live tensors hold.
    pub in_use: u64,
}

/// Our pool on `backend`'s device. Asked of the device server, so it waits
/// behind queued work — callers hold the model slot, where nothing is queued.
fn pool(backend: BackendChoice) -> (u64, u64) {
    device_of(backend).memory_pool_usage().map_or((0, 0), |u| {
        (u.bytes_reserved, u.bytes_in_use.min(u.bytes_reserved))
    })
}

/// The card, if anything on this machine reports it.
///
/// NVML first (global used/free, every process). Without it — a machine
/// with no NVIDIA driver — the adapter inventory gives the total, and
/// "used" is only what we can see of ourselves; the guard's floor and slack
/// are then the whole margin, which is the most any machine without a
/// reading can honestly do.
pub(super) fn card(backend: BackendChoice) -> Option<Card> {
    if backend == BackendChoice::Cpu {
        return None;
    }
    let (reserved, in_use) = pool(backend);
    if let Some(m) = crate::status::vram_reading(READ_BUDGET) {
        Some(Card {
            total: m.total,
            used: m.used.max(reserved),
            reserved,
            in_use,
        })
    } else {
        let total = mummu::backend::inventory()
            .gpus
            .iter()
            .filter_map(|g| g.vram_bytes)
            .max()?;
        Some(Card {
            total,
            used: reserved,
            reserved,
            in_use,
        })
    }
}

/// The chance-constrained guard on ambient VRAM (SPEC 3), fed once per
/// [`GUARD_PERIOD`] (a call inside the period reads the current guard).
fn guard(ambient: u64) -> u64 {
    use mummu::schedule::watermark::{Watermark, WatermarkConfig};
    static WM: Mutex<Option<(Watermark, Option<Instant>)>> = Mutex::new(None);
    let mut g = WM.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let (wm, fed) = g.get_or_insert_with(|| {
        (
            Watermark::new(WatermarkConfig {
                floor_bytes: 1 << 30,
                frag_slack_bytes: 512 << 20,
                // A minute of 2 s samples, and half a minute of quiet before
                // any shrink: covers a co-tenant's bursts within that minute
                // (a game streaming textures, a browser tab), and gives the
                // card back about a minute after one leaves. Was ten
                // minutes, which left a card a game had quit idle for ten
                // minutes — the opposite of using all the excess.
                window: 30,
                hysteresis_window: 15,
                ..WatermarkConfig::default()
            }),
            None,
        )
    });
    let out = if fed.is_some_and(|t| t.elapsed() < GUARD_PERIOD) {
        wm.guard_bytes().max(ambient)
    } else {
        *fed = Some(Instant::now());
        feed_guard(wm, ambient)
    };
    drop(g);
    out
}

/// One observation into the watermark, and the guard it now asks for.
fn feed_guard(wm: &mut mummu::schedule::watermark::Watermark, ambient: u64) -> u64 {
    wm.observe_ambient(ambient);
    // Taken, not read: a breach is one piece of evidence, and reading the
    // flag on every poll would boost the guard 1.5x per poll until the next
    // generation cleared it.
    if super::ALLOC_FAILED.swap(false, std::sync::atomic::Ordering::SeqCst) {
        wm.breach();
    }
    wm.guard_bytes().max(ambient)
}

/// Bytes our own pool handed back that the driver may not have returned to
/// the card yet, and the ambient reading from just before it handed them
/// back. See [`ambient`].
#[derive(Debug, Clone, Copy)]
struct InFlight {
    bytes: u64,
    ambient_before: u64,
    since: Instant,
}

/// The previous reading's `reserved`, and the ambient it produced.
static LAST_READING: Mutex<Option<(u64, u64)>> = Mutex::new(None);

/// Bytes released but possibly not yet reclaimed by the driver.
static IN_FLIGHT: Mutex<Option<InFlight>> = Mutex::new(None);

/// How long the driver may hold pages our pool released. A reading still
/// high after this is a co-tenant, not us, and is believed in full.
const RELEASE_SETTLE: Duration = Duration::from_secs(30);

/// The smallest drop in `reserved` worth correcting for. Below it the
/// correction is inside the noise of an NVML sample and only costs guard.
const RELEASE_FLOOR: u64 = 256 << 20;

/// `A = used − reserved`, corrected for bytes we just gave back.
///
/// The raw subtraction is right only while our pool and the driver agree
/// about what we hold. They disagree for seconds after a drop: the pool
/// reports `reserved` down immediately, the driver keeps the pages
/// attributed to this process until it reclaims them, and every one of those
/// bytes then reads as somebody else's. Measured 2026-09-23 on both 27Bs —
/// ambient read 12.1 GiB directly after a model drop on a box whose desktop
/// ambient is 3.1-3.3 GiB. That reading is not merely wrong once: the guard
/// is an envelope with a 120-sample window, so ONE post-drop sample holds the
/// guard above 12 GiB for ten minutes, the reload that follows fits nothing
/// on the card ("nothing fits everywhere", 0/64 layers), and the next drop
/// re-feeds it. That is the recovery loop, and this is where it starts.
///
/// So a fall in `reserved` is treated as ours-in-flight rather than as
/// somebody else's arrival: for [`RELEASE_SETTLE`] the reading is credited
/// back by at most what we released, and never below the ambient observed
/// before we released it. Growth beyond `ambient_before + released` is still
/// believed immediately — the guard's "up at once" property survives, it is
/// only the bytes we can account for as our own that stop counting twice.
fn ambient(c: &Card) -> u64 {
    let mut last = LAST_READING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut flight = IN_FLIGHT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    correct_ambient(c, &mut last, &mut flight, Instant::now())
}

/// [`ambient`] with its state passed in: the whole rule, no globals, so the
/// sequences that matter (a drop, a slow reclaim, a co-tenant arriving during
/// one) can be written down as tests.
///
/// `last` is `(reserved, ambient)` from the previous reading; `flight` is the
/// open credit window, if any. Both are updated in place.
fn correct_ambient(
    c: &Card,
    last: &mut Option<(u64, u64)>,
    flight: &mut Option<InFlight>,
    now: Instant,
) -> u64 {
    let raw = c.used.saturating_sub(c.reserved);

    // A pool that shrank by a real amount: open (or refresh) the window.
    if let Some((prev_reserved, prev_ambient)) = *last
        && prev_reserved >= c.reserved.saturating_add(RELEASE_FLOOR)
    {
        let released = prev_reserved - c.reserved;
        let carried = flight
            .filter(|f| now.duration_since(f.since) < RELEASE_SETTLE)
            .map_or(0, |f| f.bytes);
        *flight = Some(InFlight {
            bytes: released.saturating_add(carried),
            ambient_before: prev_ambient,
            since: now,
        });
    }

    let corrected = match *flight {
        Some(f) if now.duration_since(f.since) < RELEASE_SETTLE => {
            // What the driver has NOT given back yet: never more than we
            // released, and never so much that the reading falls below the
            // ambient we trusted before releasing.
            let outstanding = raw.saturating_sub(f.ambient_before).min(f.bytes);
            if outstanding == 0 {
                // Settled early: the card is back where it was.
                *flight = None;
            } else {
                // Shrink the credit as pages come back, so a co-tenant that
                // arrives mid-window is not hidden by a stale entitlement.
                *flight = Some(InFlight {
                    bytes: outstanding,
                    ..f
                });
            }
            raw - outstanding
        }
        _ => {
            *flight = None;
            raw
        }
    };

    assert!(corrected <= raw, "the correction only ever credits back");
    debug_assert!(
        flight.is_none_or(|f| corrected >= f.ambient_before.min(raw)),
        "the correction never reads below the last trusted ambient"
    );
    *last = Some((c.reserved, corrected));
    corrected
}

/// Is another process using the card's compute right now (`sysmon`'s
/// hysteresis decided so)? Then the card is theirs: no layer stays on it,
/// nothing new goes onto it, and our work runs on the host — where it only
/// ever gets idle cycles (`sysmon::install_compute_pool`).
pub(super) fn gpu_yielded() -> bool {
    crate::sysmon::pressure().gpu == crate::sysmon::Yield::Yield
}

/// `K = total − G`: what this process may occupy on the card.
pub(super) fn capacity(c: &Card) -> u64 {
    c.total.saturating_sub(guard(ambient(c)))
}

/// Free for a NEW placement on `backend` — what every planner that has no
/// resident layers to count (the fit planner, the precision mix, the `MoE`
/// tiers) spends: `K − in_use − V_pending`.
pub(super) fn free_for_new(backend: BackendChoice) -> u64 {
    // A co-tenant is using the card's compute: nothing new goes there.
    if gpu_yielded() {
        return 0;
    }
    let Some(c) = card(backend) else {
        return 0;
    };
    capacity(&c)
        .saturating_sub(c.in_use)
        .saturating_sub(super::VISION_RESERVE.load(std::sync::atomic::Ordering::SeqCst))
}

/// Recent samples, envelope semantics: the max of the last [`ENVELOPE`].
#[derive(Debug, Default)]
struct Envelope(VecDeque<u64>);

impl Envelope {
    fn push(&mut self, v: u64) {
        if self.0.len() == ENVELOPE {
            self.0.pop_front();
        }
        self.0.push_back(v);
    }
    fn max(&self) -> Option<u64> {
        self.0.iter().copied().max()
    }
}

/// ε̂ — the allocator residual, measured after every generation.
static RESIDUAL: Mutex<Envelope> = Mutex::new(Envelope(VecDeque::new()));

/// The evidence-file key ε̂ is remembered under.
///
/// Renamed when ε̂ stopped absorbing the context-dependent transients
/// ([`transient_bytes`]): the old `residual_bytes` held them too — 5.65 GiB
/// on the production Bonsai, where a 4786-token prompt needed 3.4 — and read
/// as the new quantity it would hold that back from layers for good.
const RESIDUAL_KEY: &str = "residual_base_bytes";

/// ε̂ before anything was measured: a quarter of the card.
///
/// Not a small number on purpose. The first load has to fit BEFORE the first
/// generation can measure anything, so an optimistic prior is exactly an OOM
/// on a cold card — v0.4.0's 1 GiB prior put 48 of the 27B's 64 layers on a
/// 16 GiB card and the load died reserving its 250 MB pool pages. v0.3's
/// production pool held 4.3 GiB beyond the 27B's 7.05 GiB of weights during
/// an 1100-token prefill; a quarter of this card is 4 GiB. The measurement
/// replaces it after the first generation (and persists, see
/// [`remember_residual`]), so a conservative prior costs one request's worth
/// of layers, while an optimistic one costs the request.
fn residual_prior(card_total: Option<u64>) -> u64 {
    card_total.map_or(1 << 30, |t| (t / 4).max(1 << 30))
}

/// The most ε̂ may ever claim: half the card.
///
/// [`note_device_failure`] doubles ε̂ on every out-of-memory, and doubling is
/// unbounded — from the 4 GiB prior on this 16 GiB card it reaches 8, then 16,
/// and at 16 the working set alone exceeds the card, so nothing fits anywhere
/// and the load reports "nothing fits everywhere" with 0/64 layers. Worse, the
/// value is remembered ([`remember_residual`]), so the next process starts
/// there too: a single bad night is written to disk and every restart after it
/// serves entirely from the host. That is a ratchet, not an estimate.
///
/// Half the card allows exactly the one doubling that carries information —
/// "the working set was bigger than measured, reserve more" — and refuses the
/// one that cannot: if a run still runs out with half the card held back, the
/// answer is not another doubling, it is that this working set does not fit
/// beside this model, and [`note_device_failure`] says so instead of writing a
/// larger number. The clamp is applied on READ as well as on write, so a file
/// left by an older build (or a hand-seeded value) cannot carry a dead card
/// into a fresh process.
fn residual_ceiling(card_total: Option<u64>) -> u64 {
    card_total.map_or(u64::MAX, |t| (t / 2).max(residual_prior(Some(t))))
}

fn residual_for(card_total: Option<u64>) -> u64 {
    RESIDUAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .max()
        .map_or_else(
            || residual_prior(card_total),
            |v| v.min(residual_ceiling(card_total)),
        )
}

fn inventory_vram() -> Option<u64> {
    mummu::backend::inventory()
        .gpus
        .iter()
        .filter_map(|g| g.vram_bytes)
        .max()
}

/// Non-weight bytes a planner without a model config should hold back: the
/// measured residual, but never less than its prior. ε̂ no longer covers the
/// prefill transients ([`transient_bytes`]), and such a planner cannot add
/// them — it has no context and no shapes to add them from.
pub(super) fn nonweight_estimate() -> u64 {
    let card = inventory_vram();
    residual_for(card).max(residual_prior(card))
}

/// Where ε̂ is kept between processes: a recovery restart must not forget
/// what the card taught it and repeat the load that failed.
static RESIDUAL_FILE: Mutex<Option<PathBuf>> = Mutex::new(None);

fn residual_file_for(pack_dir: &Path) -> Option<PathBuf> {
    // <models root>/<model>/pack -> <models root>/.mummu-serve/placement-<model>.json
    let model_dir = pack_dir.parent()?;
    let root = model_dir.parent()?;
    let name = model_dir.file_name()?.to_string_lossy().into_owned();
    Some(
        root.join(crate::recovery::EVIDENCE_DIR)
            .join(format!("placement-{name}.json")),
    )
}

/// Load the remembered ε̂ for the model at `pack_dir`, if this process has
/// not measured one yet.
fn recall_residual(pack_dir: &Path) {
    let Some(path) = residual_file_for(pack_dir) else {
        return;
    };
    *RESIDUAL_FILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(path.clone());
    let mut env = RESIDUAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if env.max().is_some() {
        return;
    }
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    let Some(v) = remembered_residual(&text) else {
        return;
    };
    // Clamped on the way in: a file written before the ceiling existed — or
    // seeded by hand during an incident — must not start this process with a
    // working set the card cannot hold.
    let ceiling = residual_ceiling(inventory_vram());
    let kept = v.min(ceiling);
    env.push(kept);
    drop(env);
    let clamped = if v > ceiling {
        format!(" (clamped from {:.2} GiB)", gib(v))
    } else {
        String::new()
    };
    eprintln!(
        "[mummu-serve] placement: working-set residual {:.2} GiB, remembered from {}{}",
        gib(kept),
        path.display(),
        clamped,
    );
}

/// ε̂ from an evidence file's text, under [`RESIDUAL_KEY`] only: an older
/// build's `residual_bytes` measured a different quantity.
fn remembered_residual(text: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()?
        .get(RESIDUAL_KEY)?
        .as_u64()
}

/// Persist the current ε̂ (best effort; a failed write only costs a prior).
fn remember_residual() {
    let Some(path) = RESIDUAL_FILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
    else {
        return;
    };
    let Some(v) = RESIDUAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .max()
    else {
        return;
    };
    update_evidence(&path, |e| e[RESIDUAL_KEY] = serde_json::json!(v));
}

/// Rewrite one field of the placement evidence file, keeping the others
/// (best effort, like everything written there).
fn update_evidence(path: &Path, change: impl FnOnce(&mut serde_json::Value)) {
    let mut evidence = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    change(&mut evidence);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, evidence.to_string());
}

// ---------------------------------------------------------------------------
// Remembered rates
// ---------------------------------------------------------------------------

/// The levels a rate can be remembered for, by name.
const RATE_LEVELS: [QuantPolicy; 4] = [
    QuantPolicy::Q4,
    QuantPolicy::Q8,
    QuantPolicy::F16,
    QuantPolicy::Off,
];

/// Where a build's rates are kept in the evidence file: a new build may
/// have new kernels, so it starts its own record.
fn rates_key(backend: BackendChoice) -> String {
    format!("{}/{}", env!("CARGO_PKG_VERSION"), label_of(backend))
}

/// The best rates (seconds per byte) this build has measured for the
/// model's probe tensor, host and accelerator, if any were remembered.
fn recall_rates(path: &Path, backend: BackendChoice) -> Option<[Vec<(QuantPolicy, f64)>; 2]> {
    let text = std::fs::read_to_string(path).ok()?;
    let evidence = serde_json::from_str::<serde_json::Value>(&text).ok()?;
    let rec = &evidence["rates"][rates_key(backend)];
    let side = |name: &str| -> Vec<(QuantPolicy, f64)> {
        RATE_LEVELS
            .iter()
            .filter_map(|&q| {
                rec[name][format!("{q:?}")]
                    .as_f64()
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .map(|s| (q, s))
            })
            .collect()
    };
    let (host, accel) = (side("host"), side("accel"));
    (!host.is_empty() || !accel.is_empty()).then_some([host, accel])
}

fn remember_rates(
    path: &Path,
    backend: BackendChoice,
    host: &[(QuantPolicy, f64)],
    accel: &[(QuantPolicy, f64)],
) {
    let side = |rates: &[(QuantPolicy, f64)]| -> serde_json::Value {
        rates
            .iter()
            .map(|(q, s)| (format!("{q:?}"), serde_json::json!(s)))
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    update_evidence(path, |e| {
        if !e["rates"].is_object() {
            e["rates"] = serde_json::json!({});
        }
        e["rates"][rates_key(backend)] =
            serde_json::json!({ "host": side(host), "accel": side(accel) });
    });
}

/// Per level, the faster of a fresh measurement and the remembered best.
///
/// A rate is what the device does free (the live share is applied per
/// plan, see [`contended`]), and anything that slows a probe — a co-tenant
/// starting, a card still clocking up — only ever makes it slower. A probe
/// taken in such a moment once pinned the 2B at Q4 for its whole life:
/// measured card Q8 90 GB/s against its usual 309 made Q4 the faster
/// level, and the Q4 2B opened a think block even when told not to. Only
/// the levels measured now are kept: one that failed to probe stays out.
fn best_rates(
    measured: &[(QuantPolicy, f64)],
    remembered: &[(QuantPolicy, f64)],
) -> Vec<(QuantPolicy, f64)> {
    measured
        .iter()
        .map(|&(q, s)| {
            let best = remembered
                .iter()
                .find(|(r, _)| *r == q)
                .map_or(s, |&(_, r)| r.min(s));
            (q, best)
        })
        .collect()
}

/// Whether any measured level ran slower than `margin` × its remembered
/// best — the probe met something it should not have.
fn slower_than_remembered(
    measured: &[(QuantPolicy, f64)],
    remembered: &[(QuantPolicy, f64)],
    margin: f64,
) -> bool {
    measured.iter().any(|&(q, s)| {
        remembered
            .iter()
            .find(|(r, _)| *r == q)
            .is_some_and(|&(_, r)| s > r * margin)
    })
}

/// How much slower than its best a probe may run before the load asks for a
/// re-probe, and how close a re-probe must come to the best to confirm it.
const REPROBE_MARGIN: f64 = 1.25;

/// The least free share of the host at which its idle re-probe runs (its
/// probe uses every core; the rate is normalized by the share it had).
const HOST_PROBE_SHARE: f64 = 0.5;

/// Host cores that must be idle for the card's re-probe: every kernel it
/// times is launched from here, and with the host oversubscribed (24
/// spinners on 16 cores) each launch waits for a time slice — a free card
/// read Q4 at 30 GB/s and Q8 at 14. A couple of idle cores is all a launch
/// thread and the device thread need; most of the host is not.
const CARD_PROBE_IDLE_CORES: f64 = 2.0;

/// The most of the card other processes may use for its re-probe, percent,
/// from a reading at most [`QUIET_READING_AGE`] old.
const QUIET_OTHERS_PERCENT: u32 = 20;
const QUIET_READING_AGE: u64 = 10;

/// The first re-probe's delay after a load — also what the pressure
/// readings need to stop reading "all free" (they start there) — and the
/// cap the delay doubles up to.
const REPROBE_FIRST: Duration = Duration::from_secs(15);
const REPROBE_MAX_WAIT: Duration = Duration::from_secs(600);

/// Re-probes a device gets at most after a load.
const REPROBE_TRIES: u32 = 6;

/// One device's re-probe schedule after a load whose probe may not have
/// seen it free (see [`Live::reprobe`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reprobe {
    /// When the next try is due; `None` when none is wanted.
    due: Option<Instant>,
    /// What the try after it waits.
    wait: Duration,
    tries: u32,
}

impl Reprobe {
    fn wanted(now: Instant) -> Self {
        Self {
            due: Some(now + REPROBE_FIRST),
            wait: REPROBE_FIRST,
            tries: 0,
        }
    }

    const fn done() -> Self {
        Self {
            due: None,
            wait: REPROBE_FIRST,
            tries: 0,
        }
    }

    fn is_due(&self, now: Instant) -> bool {
        self.due.is_some_and(|d| now >= d)
    }

    /// After a try that `confirmed` the remembered best (came within the
    /// margin of it): finished once a second try confirms — the first may
    /// itself have raised the best, a later one then saw the same free
    /// device — or the tries run out; otherwise it waits twice as long.
    fn after(&mut self, confirmed: bool, now: Instant) {
        self.tries += 1;
        if (confirmed && self.tries >= 2) || self.tries >= REPROBE_TRIES {
            self.due = None;
            return;
        }
        self.wait = (self.wait * 2).min(REPROBE_MAX_WAIT);
        self.due = Some(now + self.wait);
    }
}

/// A device ran out of memory under a placement this module made: the
/// working set was bigger than ε̂ said. Double it (and let the guard count a
/// breach), so the reload that follows plans smaller instead of repeating
/// the failure — and remember it, so a restart does too.
///
/// Bounded by [`residual_ceiling`]: at the ceiling the doubling stops, and
/// says so rather than writing a number that guarantees an empty card.
pub(super) fn note_device_failure(cause: &str) {
    if !cause.contains("out of device memory") {
        return;
    }
    super::ALLOC_FAILED.store(true, std::sync::atomic::Ordering::SeqCst);
    let card = inventory_vram();
    let before = residual_for(card);
    let after = escalate_residual(before, card);
    if after == before {
        eprintln!(
            "[mummu-serve] placement: out of device memory with the working-set estimate already at its ceiling ({:.2} GiB of a {:.2} GiB card) — holding it there; this model's working set does not fit beside its own weights on this device",
            gib(before),
            gib(card.unwrap_or(0)),
        );
        return;
    }
    RESIDUAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(after);
    remember_residual();
    eprintln!(
        "[mummu-serve] placement: out of device memory — working-set estimate {:.2} -> {:.2} GiB; the next load places fewer layers",
        gib(before),
        gib(after),
    );
}

/// One escalation step: double, but never past [`residual_ceiling`].
fn escalate_residual(before: u64, card_total: Option<u64>) -> u64 {
    let ceiling = residual_ceiling(card_total);
    debug_assert!(before <= ceiling, "ε̂ is clamped on every read");
    let after = before.saturating_mul(2).min(ceiling);
    debug_assert!(after >= before, "evidence of an OOM never lowers ε̂");
    debug_assert!(after <= ceiling, "and never raises it past the ceiling");
    after
}

/// Contexts (prompt + budget) of recent requests.
static CONTEXTS: Mutex<Envelope> = Mutex::new(Envelope(VecDeque::new()));

/// The context an idle re-plan provisions for.
fn idle_context() -> usize {
    CONTEXTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .max()
        .map_or(4096, |c| usize::try_from(c).unwrap_or(usize::MAX))
}

/// Tokens served, for the horizon `H`.
static SERVED: Mutex<VecDeque<(Instant, usize)>> = Mutex::new(VecDeque::new());

fn horizon_tokens() -> f64 {
    let mut s = SERVED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    while s
        .front()
        .is_some_and(|(t, _)| now.duration_since(*t) > HORIZON_WINDOW)
    {
        s.pop_front();
    }
    // At least one reply's worth: a server that just started serving should
    // be able to take back a card that freed up.
    f64_from_usize((s.iter().map(|(_, n)| *n).sum::<usize>()).max(256))
}

/// Measured pack read rate, seconds per byte (prior: this array's quiet
/// 150 MB/s until a load or a move measures it).
static DISK_S_PER_BYTE: Mutex<Option<f64>> = Mutex::new(None);
const DISK_PRIOR_S_PER_BYTE: f64 = 1.0 / 150e6;

pub(super) fn note_disk(bytes: u64, secs: f64) {
    if bytes < (64 << 20) || secs <= 0.0 {
        return;
    }
    let seen = secs / f64_from_u64(bytes);
    let mut d = DISK_S_PER_BYTE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The first measurement replaces the prior outright; after that an
    // EWMA, so one slow read under a co-tenant does not define the disk.
    *d = Some(d.map_or(seen, |prev| 0.3f64.mul_add(seen, 0.7 * prev)));
}

fn disk_s_per_byte() -> f64 {
    DISK_S_PER_BYTE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .unwrap_or(DISK_PRIOR_S_PER_BYTE)
}

/// What the current request needs beyond the weights: its context and
/// whether it brings an image. Set per request before planning.
static REQUEST: Mutex<Option<(usize, bool)>> = Mutex::new(None);

/// Publish the request about to be planned for. `ctx` may be an estimate
/// here; [`before_request`] replaces it with the exact token count.
pub(super) fn set_request(ctx: usize, needs_tower: bool) {
    *REQUEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((ctx, needs_tower));
}

fn request() -> Option<(usize, bool)> {
    *REQUEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// When the vision tower was last used, for [`TOWER_IDLE`].
static TOWER_USED: Mutex<Option<Instant>> = Mutex::new(None);

pub(super) fn note_tower_use() {
    *TOWER_USED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
}

// ---------------------------------------------------------------------------
// The model as the solver sees it
// ---------------------------------------------------------------------------

const fn policy_of(p: Precision) -> QuantPolicy {
    match p {
        Precision::Q4 => QuantPolicy::Q4,
        Precision::Q8 => QuantPolicy::Q8,
        Precision::F16 => QuantPolicy::F16,
        Precision::F32 => QuantPolicy::Off,
    }
}

const fn precision_of(q: QuantPolicy) -> Precision {
    match q {
        QuantPolicy::Q4 | QuantPolicy::Q2 => Precision::Q4,
        QuantPolicy::Q8 => Precision::Q8,
        QuantPolicy::F16 => Precision::F16,
        QuantPolicy::Off => Precision::F32,
    }
}

fn blob_bytes(e: &TensorEntry, p: Precision) -> Option<u64> {
    e.precisions.get(&p).map(|b| b.values_len + b.scales_len)
}

/// Is this tensor one whose precision the solver chooses? A linear weight
/// stored at both integer levels; anything else keeps its load precision.
fn choosable(e: &TensorEntry) -> bool {
    matches!(e.role, Role::Linear | Role::Expert { .. })
        && e.precisions.contains_key(&Precision::Q4)
        && e.precisions.contains_key(&Precision::Q8)
}

fn kind_of(e: &TensorEntry) -> Kind {
    if e.name.contains("ffn") {
        Kind::Ffn
    } else {
        Kind::Attention
    }
}

/// The precision a non-choosable tensor always loads at: its most precise
/// stored level (norms, conv kernels and the narrow projections are read as
/// f32 whatever is asked).
fn fixed_precision(e: &TensorEntry) -> Precision {
    e.precisions.keys().copied().max().unwrap_or(Precision::F32)
}

/// The precision a tensor outside every layer loads at: the embedding, the
/// output head and the final norm.
///
/// NOT [`fixed_precision`]. v0.4.0 sent these through it, so the 27B's head
/// (`output.weight`, [5120, 248320]) loaded at f32 — a 5.08 GB buffer — and,
/// following the last layer onto the card, failed to allocate there
/// (`failed to reserve 5085593600 bytes`); on the host it would have streamed
/// 5 GB per token instead of 0.7. The rules are the ones the precision mix
/// applied before: the head is a projection like any other and runs at Q4
/// (the host's fastest level, and the smallest on the card); the embedding is
/// a gather that lands at the backend's float dtype either way, so for a
/// quantization-born source (1-16 bits/param) it is read at f16 — half the
/// bytes off the disk, holding the source's values well under its own quant
/// error — and at f32 otherwise.
fn trunk_precision(e: &TensorEntry, source_bits: f64) -> Precision {
    match e.role {
        Role::Embedding
            if (1.0..16.0).contains(&source_bits) && e.precisions.contains_key(&Precision::F16) =>
        {
            Precision::F16
        }
        Role::Linear | Role::Expert { .. } => [Precision::Q4, Precision::Q8]
            .into_iter()
            .find(|p| e.precisions.contains_key(p))
            .unwrap_or_else(|| fixed_precision(e)),
        _ => fixed_precision(e),
    }
}

/// One layer's tensors grouped the way the solver chooses them.
#[derive(Debug, Clone)]
struct LayerMap {
    /// Tensor names of each part, parallel to `parts`.
    names: Vec<Vec<String>>,
    parts: Vec<joint::Part>,
    fixed_bytes: u64,
}

fn layer_maps(pack: &Pack, layers: usize, ceiling: QuantPolicy) -> Vec<LayerMap> {
    let mut maps = vec![
        LayerMap {
            names: vec![Vec::new(), Vec::new()],
            parts: vec![
                joint::Part {
                    params: 0,
                    kind: Kind::Attention,
                    levels: Vec::new(),
                },
                joint::Part {
                    params: 0,
                    kind: Kind::Ffn,
                    levels: Vec::new(),
                },
            ],
            fixed_bytes: 0,
        };
        layers
    ];
    // Levels every tensor of a part stores, with the part's summed bytes.
    let mut sums: Vec<[Option<std::collections::BTreeMap<Precision, u64>>; 2]> =
        vec![[None, None]; layers];
    for e in &pack.manifest.tensors {
        let Some(l) = layer_index(&e.name).filter(|&l| l < layers) else {
            continue;
        };
        if !choosable(e) {
            maps[l].fixed_bytes += e.shape.iter().product::<usize>() as u64 * 4;
            continue;
        }
        let pi = usize::from(kind_of(e) == Kind::Ffn);
        maps[l].names[pi].push(e.name.clone());
        maps[l].parts[pi].params += e.shape.iter().product::<usize>();
        let here: std::collections::BTreeMap<Precision, u64> = Precision::ALL
            .iter()
            .filter(|&&p| policy_of(p).bits() <= ceiling.bits())
            .filter_map(|&p| blob_bytes(e, p).map(|b| (p, b)))
            .collect();
        let acc = &mut sums[l][pi];
        *acc = Some(match acc.take() {
            None => here,
            Some(prev) => prev
                .into_iter()
                .filter_map(|(p, b)| here.get(&p).map(|h| (p, b + h)))
                .collect(),
        });
    }
    for (l, s) in sums.into_iter().enumerate() {
        for (pi, levels) in s.into_iter().enumerate() {
            maps[l].parts[pi].levels = levels
                .unwrap_or_default()
                .into_iter()
                .map(|(p, b)| (policy_of(p), b))
                .collect();
        }
        // A layer with no choosable tensor of a kind has an empty part; it
        // costs nothing and stores nothing, so drop it.
        let keep: Vec<bool> = maps[l].parts.iter().map(|p| p.params > 0).collect();
        let mut i = 0;
        maps[l].parts.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        let mut i = 0;
        maps[l].names.retain(|_| {
            i += 1;
            keep[i - 1]
        });
    }
    maps
}

/// Per-generation state a layer carries at `ctx` tokens (joint's `s_l`).
pub(super) fn state_bytes(cfg: &qwen35::Qwen35Config, layer: usize, ctx: usize) -> u64 {
    let f32b = 4u64;
    if cfg.is_attention(layer) {
        let kv = if mummu::nn::kv_f16_enabled() { 2 } else { f32b };
        2 * cfg.num_key_value_heads as u64 * ctx as u64 * cfg.head_dim as u64 * kv
    } else {
        cfg.conv_dim() as u64 * cfg.conv_kernel.saturating_sub(1) as u64 * f32b
            + cfg.n_v_heads as u64 * (cfg.d_state as u64).pow(2) * f32b
    }
}

/// The widest live activation: three `[chunk, intermediate]` f32 buffers
/// through `SwiGLU`. Prefill is chunked, so this is bounded by the chunk, not
/// the context.
pub(super) fn act_bytes(cfg: &qwen35::Qwen35Config, ctx: usize) -> u64 {
    3 * ctx.min(mummu::decode::prefill_chunk_len()).max(1) as u64 * cfg.intermediate_size as u64 * 4
}

/// Prefill spans up to this many rows run the packed GEMV row by row on the
/// card; wider ones take the plain matmul, which dequantizes the weight
/// (`nn::try_q4s_gemv`).
const GEMV_ROWS: usize = 64;

/// The card's prefill transients beyond [`act_bytes`] at `ctx` tokens.
///
/// Attention: a chunk's scores against the whole context, `[heads, chunk,
/// ctx]` f32, live as scores, masked scores and probabilities at once. And a
/// span wider than [`GEMV_ROWS`] multiplies by its projection dequantized to
/// f32, the largest one at a time. Measured on the Bonsai 27B (2026-10-04,
/// 1024-token chunks): the pool grew 1.73 GiB over a 4786-token prefill,
/// against 1.84 here with `act` and the state; at 54 tokens 0.08 against
/// 0.03. Without this term the residual absorbed it — a flat 5.65 GiB that
/// over-reserved every short request and still under-reserved a 16k one.
pub(super) fn transient_bytes(cfg: &qwen35::Qwen35Config, ctx: usize) -> u64 {
    let rows = ctx.min(mummu::decode::prefill_chunk_len()).max(1) as u64;
    let scores = 3 * cfg.num_attention_heads as u64 * rows * ctx.max(1) as u64 * 4;
    let hidden = cfg.hidden_size as u64;
    let widest = [
        cfg.intermediate_size as u64,
        2 * (cfg.num_attention_heads * cfg.head_dim) as u64,
        cfg.conv_dim() as u64,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let dequant = if rows > GEMV_ROWS as u64 {
        hidden * widest * 4
    } else {
        0
    };
    scores + dequant
}

/// A device's measured behaviour.
#[derive(Debug, Clone, Default)]
struct DeviceModel {
    rate: Vec<(QuantPolicy, f64)>,
    resident: Vec<(QuantPolicy, f64)>,
}

/// The least share of a device we plan on having. Below it the device is as
/// good as gone, and a smaller number would only make the solver's
/// arithmetic explode.
const MIN_FREE_SHARE: f64 = 0.02;

/// How a device's speed falls with the share of it that is free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scaling {
    /// A co-tenant's kernels time-slice the card: half the time, half the
    /// speed.
    TimeSliced,
    /// The host GEMV is DRAM-bound, so fewer cores lose less than their
    /// share — measured on the reference box (in-situ ANOVA, SPEC P1.1):
    /// 4 / 8 / 16 threads read 26.3 / 30 / 37.4 GB/s, i.e. a quarter of the
    /// cores keeps 70 % of the speed. The square root is the simple curve
    /// through those points (0.50 / 0.71 / 1.0 against 0.70 / 0.80 / 1.0)
    /// that still errs slow — the safe side for a host that is being
    /// claimed by someone else.
    MemoryBound,
}

impl Scaling {
    fn speed(self, free: f64) -> f64 {
        let free = free.clamp(MIN_FREE_SHARE, 1.0);
        match self {
            Self::TimeSliced => free,
            Self::MemoryBound => free.sqrt(),
        }
    }

    /// How many times longer a byte takes with only `free` of the device.
    fn slowdown(self, free: f64) -> f64 {
        1.0 / self.speed(free)
    }
}

/// `rate` (seconds per byte on a FREE device) on a device `slowdown` times
/// slower: our compute herd runs only on idle cycles
/// (`sysmon::install_compute_pool`), and a co-tenant's kernels time-slice
/// the card, so a busy device takes longer per byte. This is what turns one
/// solver into the whole-system schedule: a busy CPU makes the card the
/// cheap place for a layer, a busy GPU the host, and both busy make
/// everything slow. The slowdown is the settled one ([`SETTLE_WINDOW`]),
/// not the last few seconds'.
fn slowed(rate: &[(QuantPolicy, f64)], slowdown: f64) -> Vec<(QuantPolicy, f64)> {
    rate.iter().map(|&(q, s)| (q, s * slowdown)).collect()
}

/// [`slowed`] for a device of which only `free` is ours.
#[cfg(test)]
fn contended(rate: &[(QuantPolicy, f64)], free: f64, how: Scaling) -> Vec<(QuantPolicy, f64)> {
    slowed(rate, how.slowdown(free))
}

/// How far back the readings a placement change plans on reach.
///
/// A move re-reads layers off the disk — tens of seconds a layer on the
/// array — and pays off over the hours the model then serves. The readings
/// it is decided on used to be the last few seconds': `sysmon`'s free shares
/// (a three-second average) and this tick's capacities. On the reference
/// box other routines take 12-15 of the 16 cores and give them back every
/// 30-60 s, and the desktop's GPU share and free memory swing as fast; the
/// best placement flipped with them, and the live 27B made 72 layer moves
/// in its first 70 minutes (2026-10-02), re-reading ~19 GiB to chase load
/// that had moved on before each move finished. So moves plan on ten
/// minutes: the mean slowdown each device suffered (the expected cost of a
/// byte there — not the slowdown of the mean share, which flatters a
/// device that is busy half the time), and for an improvement, the least
/// room each device left. A repair still answers the card as it is this
/// moment: that is the out-of-memory guard, and it cannot wait.
const SETTLE_WINDOW: Duration = Duration::from_secs(600);

/// Timestamped readings over the last [`SETTLE_WINDOW`], oldest first.
#[derive(Debug)]
struct Window<T>(VecDeque<(Instant, T)>);

impl<T: Copy> Window<T> {
    const fn new() -> Self {
        Self(VecDeque::new())
    }

    fn push(&mut self, at: Instant, v: T) {
        while self
            .0
            .front()
            .is_some_and(|&(t, _)| at.saturating_duration_since(t) > SETTLE_WINDOW)
        {
            self.0.pop_front();
        }
        self.0.push_back((at, v));
    }

    fn values(&self) -> impl Iterator<Item = T> + '_ {
        self.0.iter().map(|&(_, v)| v)
    }
}

impl Window<f64> {
    fn mean(&self) -> Option<f64> {
        let n = self.0.len();
        (n > 0).then(|| self.values().sum::<f64>() / f64_from_usize(n))
    }
}

impl Window<u64> {
    /// The least of the window and `now` (which is always counted).
    fn least(&self, now: u64) -> u64 {
        self.values().fold(now, u64::min)
    }
}

/// The machine as placement changes see it (see [`SETTLE_WINDOW`]).
#[derive(Debug)]
struct Settle {
    /// How many times slower a host byte ran, sample by sample.
    host_slowdown: Window<f64>,
    /// The same for the card, while it was ours: a yielded card has no
    /// capacity at all, which is the repair's business, not a rate.
    card_slowdown: Window<f64>,
    /// `K`, what the card let us occupy, while it was ours.
    card_room: Window<u64>,
    /// The host's `MemAvailable`.
    host_room: Window<u64>,
}

impl Settle {
    const fn new() -> Self {
        Self {
            host_slowdown: Window::new(),
            card_slowdown: Window::new(),
            card_room: Window::new(),
            host_room: Window::new(),
        }
    }

    /// One sample of the machine's free shares and memory.
    fn observe(&mut self, at: Instant, p: &crate::sysmon::Pressure, mem_available: Option<u64>) {
        self.host_slowdown
            .push(at, Scaling::MemoryBound.slowdown(p.cpu_free));
        if p.gpu != crate::sysmon::Yield::Yield {
            self.card_slowdown
                .push(at, Scaling::TimeSliced.slowdown(p.gpu_free));
        }
        if let Some(m) = mem_available {
            self.host_room.push(at, m);
        }
    }

    /// The settled slowdowns, `(host, card)`; before the first sample, the
    /// current shares'.
    fn slowdowns(&self, p: &crate::sysmon::Pressure) -> (f64, f64) {
        (
            self.host_slowdown
                .mean()
                .unwrap_or_else(|| Scaling::MemoryBound.slowdown(p.cpu_free)),
            self.card_slowdown
                .mean()
                .unwrap_or_else(|| Scaling::TimeSliced.slowdown(p.gpu_free)),
        )
    }
}

static SETTLE: Mutex<Settle> = Mutex::new(Settle::new());

fn settle() -> std::sync::MutexGuard<'static, Settle> {
    SETTLE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Which readings a [`Live::problem`] is built on. Both plan on the settled
/// rates; they differ in the room they allow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    /// The card and host as they are this moment: what a placement must fit
    /// now — the load, a repair, a move's bytes.
    Now,
    /// The least room each device left over [`SETTLE_WINDOW`]: what an
    /// improvement must fit, so a layer one burst evicted comes back only
    /// once the card has stayed clear, not at the burst's first pause. Never
    /// more room than [`View::Now`], so a step it allows fits now too.
    Settled,
}

/// The inverse: a rate PROBED while only `free` of the device was ours, as
/// the rate of the free device. Without it the probe's contention is
/// counted twice — once in the reading, once by [`contended`] — and kept
/// after the co-tenant leaves: a 2B probed while other builds held the host
/// read 11.6 GB/s where a quiet host reads ~40.
fn uncontended(rate: &[(QuantPolicy, f64)], free: f64, how: Scaling) -> Vec<(QuantPolicy, f64)> {
    let speed = how.speed(free);
    rate.iter().map(|&(q, s)| (q, s * speed)).collect()
}

/// Time a real projection at each level `device` can run, as seconds per
/// pack byte. The host's reading is clamped to its DRAM floor.
fn measure_device(
    pack: &Pack,
    device: &burn::tensor::Device,
    host: bool,
    levels: &[Precision],
) -> DeviceModel {
    let Some(entry) = pack.entry("blk.0.ffn_gate.weight") else {
        return DeviceModel::default();
    };
    let numel = f64_from_usize(entry.shape.iter().product::<usize>());
    let mut m = DeviceModel::default();
    for &p in levels {
        let Some(bytes) = blob_bytes(entry, p) else {
            continue;
        };
        let Some(ms) = super::probe_projection_ms(pack, device, p) else {
            continue;
        };
        let ms = if host {
            ms.max(super::host_probe_floor_ms(pack, p).unwrap_or(0.0))
        } else {
            ms.max(super::card_probe_floor_ms(pack, p).unwrap_or(0.0))
        };
        let q = policy_of(p);
        let bytes = f64_from_u64(bytes);
        m.rate.push((q, ms / 1e3 / bytes));
        // Resident bytes per pack byte on this device at this level.
        let resident = match (host, p) {
            // i8 slab + f32 block scales, plus the packed VNNI twin beside it.
            (true, Precision::Q4) => {
                let twin = if mummu::flex::registry::enabled() {
                    0.5625
                } else {
                    0.0
                };
                numel * (1.0 + 0.125 + twin)
            }
            (true, Precision::Q8) => numel * 1.125,
            // Floats are widened to f32 on load (host, wgpu; assumed on CUDA).
            (_, Precision::F16 | Precision::F32) => numel * 4.0,
            // The card's pool pads what it holds: 11.28 GiB resident for a
            // 10.82 GiB plan on the 27B (v0.4.0's first production load).
            (false, _) => bytes * 1.05,
        };
        m.resident.push((q, resident / bytes));
    }
    m
}

/// Probe the host's rates at its levels, normalized to a free host.
///
/// The probes run under whatever the rest of the machine is doing; the
/// rates are kept as the FREE device's (see `uncontended`), and the live
/// share is applied per plan.
fn probe_host(pack: &Pack) -> DeviceModel {
    let probed_under = crate::sysmon::pressure();
    let mut host = measure_device(
        pack,
        &mummu::backend::cpu_device(),
        true,
        &[Precision::Q4, Precision::Q8, Precision::F16],
    );
    host.rate = uncontended(&host.rate, probed_under.cpu_free, Scaling::MemoryBound);
    host
}

/// [`probe_host`] for the accelerator; empty without one.
fn probe_card(pack: &Pack, backend: BackendChoice) -> DeviceModel {
    if backend == BackendChoice::Cpu {
        return DeviceModel::default();
    }
    let probed_under = crate::sysmon::pressure();
    // f32-widened float weights exceed wgpu's 256 MiB max buffer on the big
    // projections (1015 failed reservations, 2026-08); the integer levels
    // are the card's.
    let levels: &[Precision] = match backend {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => &[Precision::Q4, Precision::Q8, Precision::F16],
        _ => &[Precision::Q4, Precision::Q8],
    };
    let mut accel = measure_device(pack, &device_of(backend), false, levels);
    accel.rate = uncontended(&accel.rate, probed_under.gpu_free, Scaling::TimeSliced);
    // Scaled to a free card, a rate still cannot beat the card's memory: the
    // floor `measure_device` applied is re-applied after the scaling, which
    // otherwise lifted a 672 GB/s card to 748.
    if let Some(peak) = mummu::vram::memory_bandwidth() {
        for (_, s_per_byte) in &mut accel.rate {
            *s_per_byte = s_per_byte.max(1.0 / peak);
        }
    }
    accel
}

fn show_rates(m: &DeviceModel) -> String {
    m.rate
        .iter()
        .map(|(q, s)| format!("{q:?} {:.2} GB/s", 1.0 / s / 1e9))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One activation round trip host -> card -> host, seconds.
fn measure_crossing(hidden: usize, accel: &burn::tensor::Device) -> f64 {
    use burn::tensor::Tensor;
    let host = mummu::backend::cpu_device();
    let x = Tensor::<2>::zeros([1, hidden], &host);
    let trip = || {
        let y = x.clone().to_device(accel).to_device(&host);
        let _ = y.into_data().try_to_vec::<f32>();
    };
    trip();
    let reps = 5;
    let t0 = Instant::now();
    for _ in 0..reps {
        trip();
    }
    // A token crosses once each way at a boundary; the round trip is two.
    t0.elapsed().as_secs_f64() / f64::from(reps) / 2.0
}

/// A resident layered model's placement and everything needed to re-plan it.
pub(super) struct Live {
    pub pack_dir: PathBuf,
    pub backend: BackendChoice,
    pub model: String,
    cfg: qwen35::Qwen35Config,
    maps: Vec<LayerMap>,
    host: DeviceModel,
    accel: DeviceModel,
    crossing_s: f64,
    pub assignment: joint::Assignment,
    improve_wanted: u32,
    source_bits: f64,
    /// The head's bytes on the card at its load precision, and whether the
    /// load put it there (it follows the last layer, and its bytes were
    /// reserved in the solve that decided so).
    head_card_bytes: u64,
    pub head_on_card: bool,
    /// The head is the token table (a tied checkpoint): where the head goes,
    /// the table goes — see [`Self::measure`].
    pub head_is_table: bool,
    /// The host's and the card's re-probe schedules: wanted when the load's
    /// probe may not have seen the device free (no remembered best to
    /// compare with, or slower than it). See [`Live::reprobe`].
    reprobe: [Reprobe; 2],
}

pub(super) static LIVE: Mutex<Option<Live>> = Mutex::new(None);

impl Live {
    /// Measure the devices for `pack` and build the solver's view of it.
    pub fn measure(pack_dir: &Path, backend: BackendChoice) -> Result<Self, String> {
        let pack = Pack::open(pack_dir)?;
        let cfg = qwen35::Qwen35Config::from_gguf(&pack.header()?)?;
        let source_bits = super::source_bits_per_param(&pack);
        let ceiling = QuantPolicy::ceiling_for_source(source_bits);
        // A tied checkpoint has no `output.weight`: its head is a matmul
        // against the token table, every token, wherever the table is. Left
        // on the host as a mere gather, the table made the head a host read
        // of the whole vocabulary per token — the 2B, all 24 layers on the
        // card, decoded at 86 ms/token against a predicted 5. So a tied
        // head reserves the table, and the table follows it.
        let head_is_table = pack.entry("output.weight").is_none();
        let head_entry = pack.entry(if head_is_table {
            "token_embd.weight"
        } else {
            "output.weight"
        });
        let head_card_bytes = head_entry.map_or(0, |e| {
            let bytes = blob_bytes(e, trunk_precision(e, source_bits)).unwrap_or(0);
            // The card's pool padding, as for the layers.
            trunc_u64(f64_from_u64(bytes) * 1.05)
        });
        let maps = layer_maps(&pack, cfg.num_layers, ceiling);
        let mut host = probe_host(&pack);
        let mut accel = probe_card(&pack, backend);
        let crossing_s = if backend == BackendChoice::Cpu {
            0.0
        } else {
            measure_crossing(cfg.hidden_size, &device_of(backend))
        };
        // What the devices have shown at their best before, for this model
        // and build: a probe can only be slowed, never sped up, by what it
        // meets (see `best_rates`).
        let evidence = residual_file_for(pack_dir);
        let remembered = evidence.as_deref().and_then(|p| recall_rates(p, backend));
        let wanted = remembered
            .as_ref()
            .map_or([true, backend != BackendChoice::Cpu], |[h, a]| {
                [
                    slower_than_remembered(&host.rate, h, REPROBE_MARGIN),
                    slower_than_remembered(&accel.rate, a, REPROBE_MARGIN),
                ]
            });
        let reprobe = wanted.map(|w| {
            if w {
                Reprobe::wanted(Instant::now())
            } else {
                Reprobe::done()
            }
        });
        if let Some([h, a]) = &remembered {
            host.rate = best_rates(&host.rate, h);
            accel.rate = best_rates(&accel.rate, a);
        }
        if let Some(path) = &evidence {
            remember_rates(path, backend, &host.rate, &accel.rate);
        }
        let model = pack_dir.parent().and_then(|p| p.file_name()).map_or_else(
            || pack_dir.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let n = cfg.num_layers;
        eprintln!(
            "[mummu-serve] placement rates: host [{}]; {} [{}]; crossing {:.3} ms{}",
            show_rates(&host),
            label_of(backend),
            show_rates(&accel),
            crossing_s * 1e3,
            if remembered.is_some() {
                " (each the better of this probe and the remembered best)"
            } else {
                ""
            }
        );
        Ok(Self {
            pack_dir: pack_dir.to_path_buf(),
            backend,
            model,
            cfg,
            maps,
            host,
            accel,
            crossing_s,
            assignment: joint::Assignment {
                layers: (0..n)
                    .map(|_| joint::Choice {
                        device: 0,
                        levels: Vec::new(),
                    })
                    .collect(),
            },
            improve_wanted: 0,
            source_bits,
            head_card_bytes,
            head_on_card: false,
            head_is_table,
            reprobe,
        })
    }

    /// The idle re-probe: each device whose schedule is due, and whose gate
    /// `open` says it can be measured now, is probed again and keeps the
    /// better rate per level — a re-probe can only raise a rate. The idle
    /// re-plan that follows sees the devices as they are, and moves
    /// precisions with them. Not "wait for a quiet machine": a host that is
    /// never quiet still gets its card re-measured, because what the card's
    /// probe needs is a couple of idle cores, not most of the host.
    fn reprobe(&mut self, open: [bool; 2]) {
        let now = Instant::now();
        let due = [0, 1].map(|d| open[d] && self.reprobe[d].is_due(now));
        if !due.contains(&true) {
            return;
        }
        let Ok(pack) = Pack::open(&self.pack_dir) else {
            return;
        };
        for d in [0, 1].into_iter().filter(|&d| due[d]) {
            let (fresh, model) = if d == 0 {
                (probe_host(&pack), &mut self.host)
            } else {
                (probe_card(&pack, self.backend), &mut self.accel)
            };
            let confirmed = !slower_than_remembered(&fresh.rate, &model.rate, REPROBE_MARGIN);
            model.rate = best_rates(&fresh.rate, &model.rate);
            self.reprobe[d].after(confirmed, now);
            eprintln!(
                "[mummu-serve] placement: re-probed the {} (try {}): measured [{}], planning on [{}]{}",
                if d == 0 {
                    "host"
                } else {
                    label_of(self.backend)
                },
                self.reprobe[d].tries,
                show_rates(&fresh),
                show_rates(model),
                if self.reprobe[d].due.is_none() {
                    " — settled"
                } else {
                    ""
                }
            );
        }
        if let Some(path) = residual_file_for(&self.pack_dir) {
            remember_rates(&path, self.backend, &self.host.rate, &self.accel.rate);
        }
    }

    fn layers_on_card(&self) -> usize {
        self.assignment
            .layers
            .iter()
            .filter(|c| c.device == 1)
            .count()
    }

    /// The solver's problem for serving `ctx` tokens, with `card` as the
    /// current reading (None: no accelerator, or no reading at all).
    fn problem(
        &self,
        ctx: usize,
        needs_tower: bool,
        card: Option<&Card>,
        view: View,
    ) -> joint::Problem {
        let layers: Vec<joint::Layer> = self
            .maps
            .iter()
            .enumerate()
            .map(|(l, m)| joint::Layer {
                parts: m.parts.clone(),
                fixed_bytes: m.fixed_bytes,
                state_bytes: state_bytes(&self.cfg, l, ctx),
            })
            .collect();
        // Host bytes our own host layers hold now: MemAvailable excludes
        // them, and they are ours to re-spend.
        let host_layers: u64 = self
            .assignment
            .layers
            .iter()
            .zip(&layers)
            .filter(|(c, _)| c.device == 0 && !c.levels.is_empty())
            .map(|(c, l)| {
                l.parts
                    .iter()
                    .zip(&c.levels)
                    .map(|(p, &q)| {
                        let b = p.levels.iter().find(|(x, _)| *x == q).map_or(0, |x| x.1);
                        let m = self
                            .host
                            .resident
                            .iter()
                            .find(|(x, _)| *x == q)
                            .map_or(1.0, |x| x.1);
                        trunc_u64(f64_from_u64(b) * m)
                    })
                    .sum::<u64>()
                    + l.fixed_bytes
            })
            .sum();
        let pressure = crate::sysmon::pressure();
        let (host_slowdown, card_slowdown) = settle().slowdowns(&pressure);
        let mem = super::mem_available_bytes().map(|now| match view {
            View::Now => now,
            View::Settled => settle().host_room.least(now),
        });
        let host_cap = mem
            .map_or(u64::MAX / 4, |a| a / 100 * 85)
            .saturating_add(host_layers);
        let mut devices = vec![joint::Device {
            capacity: host_cap,
            fixed: act_bytes(&self.cfg, ctx),
            rate: slowed(&self.host.rate, host_slowdown),
            resident: self.host.resident.clone(),
        }];
        if let Some(c) = card
            && !self.accel.rate.is_empty()
        {
            let tower_pending = if needs_tower && !super::tower_resident() {
                super::VISION_RESERVE.load(std::sync::atomic::Ordering::SeqCst)
            } else {
                0
            };
            let ours_layers = self.planned_card_bytes();
            // N: what we hold on the card that is not a layer.
            let non_layer = c.in_use.saturating_sub(ours_layers);
            // A co-tenant using the card's compute gets all of it: capacity
            // 0 makes the current placement infeasible, and Repair moves
            // every layer to the host at once.
            let k = if gpu_yielded() {
                0
            } else {
                let k = capacity(c);
                let mut settled = settle();
                if view == View::Now {
                    settled.card_room.push(Instant::now(), k);
                }
                match view {
                    View::Now => k,
                    View::Settled => settled.card_room.least(k),
                }
            };
            devices.push(joint::Device {
                capacity: k.saturating_sub(non_layer),
                fixed: act_bytes(&self.cfg, ctx)
                    + transient_bytes(&self.cfg, ctx)
                    + residual_for(Some(c.total))
                    + tower_pending,
                rate: slowed(&self.accel.rate, card_slowdown),
                resident: self.accel.resident.clone(),
            });
        }
        joint::Problem {
            layers,
            devices,
            crossing_s: self.crossing_s,
            disk_s_per_byte: disk_s_per_byte(),
            horizon_tokens: horizon_tokens(),
            tolerance: PRECISION_TOLERANCE,
            floor: QuantPolicy::Q4,
        }
    }

    /// The pack precision `entry` loads at under `choice`.
    fn precision_for(
        &self,
        layer: usize,
        choice: &joint::Choice,
        entry: &TensorEntry,
    ) -> Precision {
        let m = &self.maps[layer];
        for (pi, names) in m.names.iter().enumerate() {
            if names.iter().any(|n| n == &entry.name) {
                return precision_of(choice.levels[pi]);
            }
        }
        fixed_precision(entry)
    }

    /// Device and precision for every tensor, for the initial load.
    pub fn loader_choice(&self) -> impl Fn(usize) -> BackendChoice + '_ {
        move |l| {
            if self.assignment.layers.get(l).is_some_and(|c| c.device == 1) {
                self.backend
            } else {
                BackendChoice::Cpu
            }
        }
    }

    pub fn loader_precision(&self) -> impl Fn(&TensorEntry) -> Precision + '_ {
        move |e| match layer_index(&e.name) {
            Some(l) if l < self.assignment.layers.len() => {
                let c = &self.assignment.layers[l];
                if c.levels.is_empty() {
                    fixed_precision(e)
                } else {
                    self.precision_for(l, c, e)
                }
            }
            _ => trunk_precision(e, self.source_bits),
        }
    }

    /// Weight bytes the placement puts on `device` (0 host, 1 card), as that
    /// device holds them.
    fn planned_bytes(&self, device: usize) -> u64 {
        let model = if device == 0 { &self.host } else { &self.accel };
        self.assignment
            .layers
            .iter()
            .zip(&self.maps)
            .filter(|(c, _)| c.device == device && !c.levels.is_empty())
            .map(|(c, m)| {
                m.parts
                    .iter()
                    .zip(&c.levels)
                    .map(|(p, &q)| {
                        let b = p.levels.iter().find(|(x, _)| *x == q).map_or(0, |x| x.1);
                        let r = model
                            .resident
                            .iter()
                            .find(|(x, _)| *x == q)
                            .map_or(1.0, |x| x.1);
                        trunc_u64(f64_from_u64(b) * r)
                    })
                    .sum::<u64>()
                    + m.fixed_bytes
            })
            .sum()
    }

    /// Pack bytes the placement reads — what a load of it costs the disk.
    pub fn planned_disk_bytes(&self) -> u64 {
        self.assignment
            .layers
            .iter()
            .zip(&self.maps)
            .filter(|(c, _)| !c.levels.is_empty())
            .map(|(c, m)| {
                m.parts
                    .iter()
                    .zip(&c.levels)
                    .map(|(p, &q)| p.levels.iter().find(|(x, _)| *x == q).map_or(0, |x| x.1))
                    .sum::<u64>()
                    + m.fixed_bytes
            })
            .sum()
    }

    /// Card bytes the placement's layers occupy — the residency check's plan.
    pub fn planned_card_bytes(&self) -> u64 {
        self.planned_bytes(1)
    }

    /// The head's card bytes when the load put it there, else 0 — beside
    /// [`Self::planned_card_bytes`] for what a load should leave resident.
    pub const fn head_bytes_on_card(&self) -> u64 {
        if self.head_on_card {
            self.head_card_bytes
        } else {
            0
        }
    }

    /// Host bytes the placement's layers occupy.
    pub fn planned_host_bytes(&self) -> u64 {
        self.planned_bytes(0)
    }

    fn summary(&self, pb: &joint::Problem) -> String {
        let mut hist: std::collections::BTreeMap<(usize, String), usize> =
            std::collections::BTreeMap::new();
        for c in &self.assignment.layers {
            for q in &c.levels {
                *hist.entry((c.device, format!("{q:?}"))).or_default() += 1;
            }
        }
        let parts: Vec<String> = hist
            .iter()
            .map(|((d, q), n)| {
                format!(
                    "{n} {q} on {}",
                    if *d == 0 {
                        "host"
                    } else {
                        label_of(self.backend)
                    }
                )
            })
            .collect();
        format!(
            "{}/{} layers on {}; parts {}; predicted {:.1} ms/token",
            self.layers_on_card(),
            self.assignment.layers.len(),
            label_of(self.backend),
            parts.join(", "),
            joint::time_of(pb, &self.assignment) * 1e3
        )
    }

    /// The card's side of (3) at reading `c`, for a repair's log line: what
    /// shrank, when a repair answers no co-tenant anyone can see.
    fn card_terms(&self, c: &Card, pb: &joint::Problem) -> String {
        let Some(dev) = pb.devices.get(1) else {
            return String::new();
        };
        let k = capacity(c);
        format!(
            " — card: {:.2} GiB for layers (K {:.2} = total {:.2} − guard {:.2}; non-layer {:.2}), working set {:.2} (ε̂ {:.2}), this placement {:.2}",
            gib(dev.capacity),
            gib(k),
            gib(c.total),
            gib(c.total.saturating_sub(k)),
            gib(c.in_use.saturating_sub(self.planned_card_bytes())),
            gib(dev.fixed),
            gib(residual_for(Some(c.total))),
            gib(joint::used_of(pb, &self.assignment)[1]),
        )
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Decide the initial placement for a load: the joint solve against what the
/// card and host hold right now, for the request that caused the load.
pub(super) fn plan_load(pack_dir: &Path, backend: BackendChoice) -> Result<Live, String> {
    if backend != BackendChoice::Cpu {
        // Whatever the previous model left in the pool goes back to the
        // driver before we read what is free.
        mummu::backend::return_memory(&device_of(backend));
    }
    recall_residual(pack_dir);
    let mut live = Live::measure(pack_dir, backend)?;
    let (ctx, tower) = request().unwrap_or_else(|| (idle_context(), false));
    let reading = card(backend);
    // The head follows the last layer. Solve with its bytes reserved on the
    // card first; if the last layer does not land there after all, the head
    // stays on the host and the reservation is given back to layers.
    let mut pb = live.problem(ctx, tower, reading.as_ref(), View::Now);
    let with_head = pb.devices.len() > 1 && live.head_card_bytes > 0;
    if with_head {
        pb.devices[1].fixed += live.head_card_bytes;
    }
    let mut out = joint::solve(&pb);
    let last_on_card = out.assignment.layers.last().is_some_and(|c| c.device == 1);
    if with_head && !last_on_card {
        pb.devices[1].fixed -= live.head_card_bytes;
        out = joint::solve(&pb);
        // Freed room may have pulled the last layer on after all; the head
        // was not reserved for, so it stays home.
    }
    live.head_on_card = with_head && last_on_card;
    if !out.feasible {
        eprintln!(
            "[mummu-serve] placement: nothing fits everywhere (host {:.1} GiB needed of {:.1}) — loading anyway at the fastest host levels",
            gib(out.used[0]),
            gib(pb.devices[0].capacity),
        );
    }
    live.assignment = out.assignment;
    if let Some(c) = reading {
        // The corrected ambient is what the guard was built from; when it
        // differs from the raw subtraction, say by how much, because that
        // gap IS the post-drop window and an incident is read from here.
        // One `ambient` call, not two: it advances the release window, and
        // reporting a line must not age the state the next plan reads.
        let raw = c.used.saturating_sub(c.reserved);
        let a = ambient(&c);
        let credited = if a == raw {
            String::new()
        } else {
            format!(
                " ({:.1} raw, {:.1} of it ours in flight)",
                gib(raw),
                gib(raw - a),
            )
        };
        eprintln!(
            "[mummu-serve] placement: card {:.1} GiB, ambient {:.1}{}, guard {:.1}, capacity {:.1} GiB for a {ctx}-token context{}",
            gib(c.total),
            gib(a),
            credited,
            gib(guard(a)),
            gib(pb.devices.get(1).map_or(0, |d| d.capacity)),
            if tower { " + vision tower" } else { "" },
        );
    }
    eprintln!("[mummu-serve] placement: {}", live.summary(&pb));
    Ok(live)
}

// ---------------------------------------------------------------------------
// Moving layers
// ---------------------------------------------------------------------------

/// Moves of the resident model's weights so far: a captured graph records
/// the buffers it read, so one recorded before a move must not be replayed
/// after it (see `decoder`).
static EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The current weight-move count (see [`EPOCH`]).
pub(super) fn epoch() -> u64 {
    EPOCH.load(std::sync::atomic::Ordering::SeqCst)
}

/// Apply `target` to the resident model, releases first. `limit` bounds the
/// layers moved (an improvement's step); a repair passes `None`.
///
/// # Errors
/// A pack read failure. A device failure panics out of the relocation, and
/// the caller decides it like any other device failure.
fn apply(
    live: &mut Live,
    lm: &mut qwen35::LoadedQwen35,
    target: &joint::Assignment,
    limit: Option<usize>,
) -> Result<usize, String> {
    let backend = live.backend;
    let changed: Vec<usize> = (0..target.layers.len())
        .filter(|&l| live.assignment.layers[l] != target.layers[l])
        .collect();
    if changed.is_empty() {
        return Ok(0);
    }
    // A move streams layers onto and off the card: our GPU use, not a
    // co-tenant's (see `crate::sysmon`).
    let _work = crate::sysmon::DeviceWork::enter();
    // Releases: leaving the card, or shrinking on it. Everything else is an
    // arrival or a host-side change.
    let pb = live.problem(idle_context(), false, None, View::Now);
    let card_bytes = |l: usize, c: &joint::Choice| -> u64 {
        if c.device != 1 {
            return 0;
        }
        let mut a = live.assignment.clone();
        for (i, x) in a.layers.iter_mut().enumerate() {
            if i != l {
                x.device = 0;
            }
        }
        a.layers[l] = c.clone();
        let mut p = pb.clone();
        if p.devices.len() < 2 {
            p.devices.push(joint::Device {
                capacity: u64::MAX,
                fixed: 0,
                rate: live.accel.rate.clone(),
                resident: live.accel.resident.clone(),
            });
        }
        joint::used_of(&p, &a).get(1).copied().unwrap_or(0)
    };
    let (mut releases, mut arrivals): (Vec<usize>, Vec<usize>) = changed.iter().partition(|&&l| {
        card_bytes(l, &target.layers[l]) <= card_bytes(l, &live.assignment.layers[l])
    });
    releases.sort_unstable();
    arrivals.sort_unstable();
    let budget = limit.unwrap_or(usize::MAX);
    let order: Vec<usize> = releases.into_iter().chain(arrivals).take(budget).collect();

    let started = Instant::now();
    let mut bytes = 0u64;
    let mut host_touched = false;
    let mut pages_to_return = false;
    if !order.is_empty() {
        EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // The decode thread's cached graphs read the buffers being replaced;
        // its state goes too — a move answers pressure.
        super::decoder::release();
    }
    for &l in &order {
        let from = live.assignment.layers[l].device;
        let to = target.layers[l].clone();
        // Pages freed by the releases go back to the driver before the first
        // arrival allocates — so the arrival sees them, and so would a
        // co-tenant if the arrival never comes.
        if to.device == 1 && pages_to_return && backend != BackendChoice::Cpu {
            mummu::backend::return_memory(&device_of(backend));
            pages_to_return = false;
        }
        let dev = if to.device == 1 {
            device_of(backend)
        } else {
            mummu::backend::cpu_device()
        };
        let choose = |e: &TensorEntry| live.precision_for(l, &to, e);
        bytes += qwen35::relocate_layers(lm, &live.pack_dir, &[l], &dev, &choose)
            .map_err(|e| e.to_string())?;
        host_touched |= from == 0 || to.device == 0;
        pages_to_return |= from == 1;
        live.assignment.layers[l] = to;
    }
    if backend != BackendChoice::Cpu {
        device_of(backend).memory_cleanup();
        device_of(backend).sync().map_err(|e| e.to_string())?;
    }
    if host_touched {
        // Twins are keyed by slab address; a moved layer's old slabs are
        // gone, and a fresh twin set is cheaper than tracking which.
        mummu::flex::registry::clear();
        super::warm_host_twins(lm, 0);
    }
    note_disk(bytes, started.elapsed().as_secs_f64());
    Ok(order.len())
}

/// Say once why a better placement is not being taken — the move-cost gate
/// (4) or nothing better existing is otherwise indistinguishable from a
/// watch that stopped working.
fn explain_hold(live: &Live, pb: &joint::Problem) {
    static SAID: Mutex<Option<(usize, usize)>> = Mutex::new(None);
    let target = joint::solve(pb);
    let now_t = joint::time_of(pb, &live.assignment);
    let key = (
        live.layers_on_card(),
        target
            .assignment
            .layers
            .iter()
            .filter(|c| c.device == 1)
            .count(),
    );
    if !target.feasible || target.time_s >= now_t * 0.99 {
        return;
    }
    let mut said = SAID
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if *said == Some(key) {
        return;
    }
    *said = Some(key);
    drop(said);
    let reread = joint::changed_bytes(pb, &live.assignment, &target.assignment);
    let saved_s = (now_t - target.time_s) * pb.horizon_tokens;
    let reread_s = f64_from_u64(reread) * pb.disk_s_per_byte;
    // The move-cost gate (4) is one reason; the other is that the room is
    // not room an improvement may take: it must have lasted the settled
    // window, with a layer to spare (`joint::replan`'s dead band).
    let why = if saved_s > reread_s {
        "the card has not kept that room, with a layer to spare, for the settled window".to_owned()
    } else {
        format!(
            "less than re-reading {:.2} GiB ({reread_s:.1}s at {:.0} MB/s)",
            gib(reread),
            1.0 / pb.disk_s_per_byte / 1e6,
        )
    };
    eprintln!(
        "[mummu-serve] placement hold: {} -> {} layers on {} would save {:.1} ms/token, {saved_s:.1}s over the {:.0}-token horizon — {why}",
        key.0,
        key.1,
        label_of(live.backend),
        (now_t - target.time_s) * 1e3,
        pb.horizon_tokens,
    );
}

fn replan_and_apply(
    live: &mut Live,
    lm: &mut qwen35::LoadedQwen35,
    ctx: usize,
    needs_tower: bool,
    idle: bool,
) -> Result<(), String> {
    let reading = card(live.backend);
    if live.backend != BackendChoice::Cpu && reading.is_none() {
        // No reading is no information: hold. Assuming room risks an OOM,
        // assuming pressure demotes a model that was running fine.
        return Ok(());
    }
    let mut pb = live.problem(ctx, needs_tower, reading.as_ref(), View::Now);
    if pb.devices.len() < 2 && live.layers_on_card() > 0 {
        return Ok(());
    }
    if idle {
        pb.horizon_tokens = pb.horizon_tokens.max(IDLE_HORIZON_TOKENS);
    }
    let (verdict, out) = joint::replan(&pb, &live.assignment);
    if verdict == joint::Verdict::Repair {
        live.improve_wanted = 0;
        let room = reading.as_ref().map(|c| live.card_terms(c, &pb));
        let before = joint::time_of(&pb, &live.assignment);
        let started = Instant::now();
        let moved = apply(live, lm, &out.assignment, None)?;
        eprintln!(
            "[mummu-serve] placement repair: moved {moved} layer(s) in {:.1}s, {:.1} -> {:.1} ms/token predicted; {}{}",
            started.elapsed().as_secs_f64(),
            before * 1e3,
            joint::time_of(&pb, &live.assignment) * 1e3,
            live.summary(&pb),
            room.unwrap_or_default(),
        );
        return Ok(());
    }
    // Only at idle, and on the settled view: an improvement must still be
    // one with the last ten minutes' load and room, not just this tick's.
    if !idle {
        return Ok(());
    }
    let mut settled = live.problem(ctx, needs_tower, reading.as_ref(), View::Settled);
    settled.horizon_tokens = pb.horizon_tokens;
    let (verdict, target) = joint::replan(&settled, &live.assignment);
    if verdict != joint::Verdict::Improve {
        live.improve_wanted = 0;
        if verdict == joint::Verdict::Hold {
            explain_hold(live, &settled);
        }
        return Ok(());
    }
    // And only once it has been wanted for a while.
    live.improve_wanted += 1;
    if live.improve_wanted < IMPROVE_DWELL {
        return Ok(());
    }
    // A step at a time, each one a placement that is faster on its own
    // (`joint::step`): the next tick may want something else.
    let Some(step) = joint::step(&settled, &live.assignment, &target.assignment, IMPROVE_STEP)
    else {
        live.improve_wanted = 0;
        return Ok(());
    };
    let before = joint::time_of(&settled, &live.assignment);
    let started = Instant::now();
    let moved = apply(live, lm, &step.assignment, None)?;
    if live.assignment == target.assignment {
        live.improve_wanted = 0;
    }
    eprintln!(
        "[mummu-serve] placement improve: moved {moved} layer(s) in {:.1}s, {:.1} -> {:.1} ms/token predicted; {}",
        started.elapsed().as_secs_f64(),
        before * 1e3,
        joint::time_of(&settled, &live.assignment) * 1e3,
        live.summary(&settled),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/// Before a generation, holding the model: make room for THIS request (its
/// context, its image) if the placement would not fit beside it. Returns the
/// pool's in-use bytes, the baseline [`after_request`] measures from.
pub(super) fn before_request(
    m: &mut Loaded,
    key: &Path,
    ctx: usize,
    needs_tower: bool,
) -> Option<u64> {
    CONTEXTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(ctx as u64);
    set_request(ctx, needs_tower);
    let AnyLm::Qwen35(lm) = &mut m.lm else {
        return None;
    };
    LIVE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_mut()
        .filter(|l| l.pack_dir.starts_with(key))
        .and_then(|live| {
            if let Err(e) = replan_and_apply(live, lm, ctx, needs_tower, false) {
                eprintln!("[mummu-serve] placement: could not re-place before the request: {e}");
            }
            (live.backend != BackendChoice::Cpu).then(|| pool(live.backend).1)
        })
}

/// After a generation, holding the model: measure the working set it
/// actually used (ε̂) and count its tokens toward the horizon.
pub(super) fn after_request(ctx: usize, tokens: usize, in_use_before: Option<u64>) {
    SERVED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push_back((Instant::now(), tokens));
    let Some(before) = in_use_before else { return };
    let residual = LIVE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(|live| (live.residual_after(ctx, before), live.backend));
    let Some((residual, backend)) = residual else {
        return;
    };
    // Measured: the pages this generation emptied go back to the driver, so
    // the next one's measurement — and the card meanwhile — carries only
    // what is still in use (a short request after a long one read 3.5 GiB of
    // the long one's pages as its own).
    mummu::backend::return_memory(&device_of(backend));
    let estimate = {
        let mut env = RESIDUAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        env.push(residual.sample);
        env.max().unwrap_or(residual.sample)
    };
    eprintln!(
        "[mummu-serve] working set: a {ctx}-token context left the pool at {:.2} GiB reserved ({:.2} in use) from {:.2} in use before; layers' state {:.2}, activations {:.2}, transients {:.2} -> residual {:.2} GiB, estimate {:.2}",
        gib(residual.reserved),
        gib(residual.in_use),
        gib(before),
        gib(residual.state),
        gib(residual.act),
        gib(residual.transient),
        gib(residual.sample),
        gib(estimate),
    );
    remember_residual();
}

/// One generation's working-set measurement (see [`Live::residual_after`]).
struct ResidualSample {
    reserved: u64,
    in_use: u64,
    state: u64,
    act: u64,
    transient: u64,
    /// ε̂'s new evidence.
    sample: u64,
}

impl Live {
    /// ε̂ for a generation that just ran at `ctx` tokens, from the pool's
    /// in-use bytes `before` it: what the card holds beyond the layers'
    /// state and the activations.
    fn residual_after(&self, ctx: usize, before: u64) -> ResidualSample {
        let (reserved, in_use) = pool(self.backend);
        let state: u64 = self
            .assignment
            .layers
            .iter()
            .enumerate()
            .filter(|(_, c)| c.device == 1)
            .map(|(l, _)| state_bytes(&self.cfg, l, ctx))
            .sum();
        let act = act_bytes(&self.cfg, ctx);
        let transient = transient_bytes(&self.cfg, ctx);
        ResidualSample {
            reserved,
            in_use,
            state,
            act,
            transient,
            sample: reserved
                .saturating_sub(before)
                .saturating_sub(state)
                .saturating_sub(act)
                .saturating_sub(transient),
        }
    }
}

/// Make `live` the placement of the model now resident.
pub(super) fn adopt(live: Live) {
    *LIVE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(live);
}

/// Forget the placement (the model it described left the slot).
pub(super) fn forget(pack_dir: Option<&Path>) {
    let mut g = LIVE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if pack_dir.is_none_or(|p| g.as_ref().is_some_and(|l| l.pack_dir == p)) {
        *g = None;
    }
}

/// One idle look: feed the guard, drop an idle tower, and re-plan if the
/// model is free.
fn tick() {
    settle().observe(
        Instant::now(),
        &crate::sysmon::pressure(),
        super::mem_available_bytes(),
    );
    // Idle retrieval models: theirs goes back too (see `crate::retrieval`).
    crate::retrieval::drop_idle();
    // A co-tenant using the card's compute gets all of it: retrieval models
    // leave it now, and the chat model's layers below.
    let yielded = gpu_yielded();
    if yielded {
        crate::retrieval::evict_from_accelerator();
        // The decode thread's batch stops admitting (its live sequences
        // finish, and the slot comes free for the moves below) and its
        // cached graphs and state leave the card.
        super::decoder::release();
    }
    // A chat model whose loader cannot move layers (qwen2/qwen3/lfm2) is
    // moved whole: unloaded, so its next request loads it where the plan
    // then says — the host while yielding, the card again after.
    let mut move_whole: Option<&'static str> = None;
    // Idle tower: its VRAM goes back to layers.
    let idle_tower = TOWER_USED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some_and(|t| t.elapsed() > TOWER_IDLE);
    let mut failure: Option<(String, BackendChoice, String)> = None;
    let _ = SLOT.try_with_mut(|key, m: &mut Loaded| {
        if idle_tower && super::drop_tower() {
            *TOWER_USED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            mummu::backend::return_memory(&device_of(m.backend));
            eprintln!(
                "[mummu-serve] placement: vision tower idle for {}s — its VRAM goes back to layers",
                TOWER_IDLE.as_secs()
            );
        }
        if matches!(m.lm, AnyLm::Qwen2(_) | AnyLm::Qwen3(_) | AnyLm::Lfm2(_)) {
            let on_card = m.backend != BackendChoice::Cpu;
            if yielded && on_card {
                move_whole = Some("another process is using the GPU's compute");
            } else if !yielded && !on_card && super::backend_choice() != BackendChoice::Cpu {
                move_whole = Some("the GPU is free again");
            }
            return;
        }
        let AnyLm::Qwen35(lm) = &mut m.lm else { return };
        let ctx = idle_context();
        let open = probe_gates(&crate::sysmon::pressure(), yielded);
        let replanned = LIVE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
            .filter(|l| l.pack_dir.starts_with(key))
            .map(|live| {
                live.reprobe(open);
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    replan_and_apply(live, lm, ctx, false, true)
                }));
                (live.model.clone(), live.backend, outcome)
            });
        let Some((model, backend, outcome)) = replanned else {
            return;
        };
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[mummu-serve] placement: idle re-plan failed: {e}");
            }
            Err(p) => {
                failure = Some((model, backend, crate::recovery::payload_text(&*p)));
            }
        }
    });
    if let Some(why) = move_whole {
        let _ = super::evict_for_placement(why);
    }
    // A panic mid-move leaves a layer half on each device: the model is not
    // servable. Decide it as a device failure and drop the model, so the next
    // request reloads it whole.
    if let Some((model, backend, message)) = failure {
        eprintln!(
            "[mummu-serve] placement: a move failed mid-layer — dropping the model: {message}"
        );
        let _ = crate::recovery::record_failure(
            &model,
            &[super::device_key(backend)],
            &crate::recovery::summarize(&message),
        );
        forget(None);
        if SLOT.clear() {
            mummu::progress::evicted();
        }
    }
}

/// Whether the host and the card can each be re-probed now (see
/// [`Live::reprobe`]): the host with at least half of it free; the card with
/// no co-tenant on it (a fresh reading), and a couple of host cores idle to
/// launch its kernels from.
fn probe_gates(pressure: &crate::sysmon::Pressure, yielded: bool) -> [bool; 2] {
    let idle_cores = pressure
        .cpu
        .as_ref()
        .map(|c| f64_from_usize(c.cores) - c.ours - c.others);
    let card_free = !yielded
        && pressure
            .gpu_others
            .is_some_and(|o| o <= QUIET_OTHERS_PERCENT)
        && pressure
            .gpu_others_age_s
            .is_some_and(|a| a <= QUIET_READING_AGE);
    [
        idle_cores.is_some() && pressure.cpu_free >= HOST_PROBE_SHARE,
        card_free && idle_cores.is_some_and(|i| i >= CARD_PROBE_IDLE_CORES),
    ]
}

/// Start the idle rebalancer. One per process.
pub(super) fn spawn_watch() {
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    eprintln!(
        "[mummu-serve] placement watch: every {}s — layers and their precisions follow the card",
        TICK.as_secs()
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(TICK);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let _ = tokio::task::spawn_blocking(tick).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_27b() -> qwen35::Qwen35Config {
        super::super::cfg_27b_for_tests()
    }

    /// A small model's problem with the rates measured on the reference box
    /// for qwen3.5-2b (host Q4 37 GB/s, card Q8 30 GB/s, Q4 14 GB/s).
    fn small_model(host_free: f64, card_free: f64, card_capacity: u64) -> joint::Problem {
        small_model_slowed(
            Scaling::MemoryBound.slowdown(host_free),
            Scaling::TimeSliced.slowdown(card_free),
            card_capacity,
        )
    }

    /// [`small_model`] with each device's slowdown given directly.
    fn small_model_slowed(host: f64, card: f64, card_capacity: u64) -> joint::Problem {
        let gib = 1u64 << 30;
        let part = |kind| joint::Part {
            params: 30_000_000,
            kind,
            levels: vec![(QuantPolicy::Q4, 19 << 20), (QuantPolicy::Q8, 34 << 20)],
        };
        let layer = joint::Layer {
            parts: vec![part(Kind::Attention), part(Kind::Ffn)],
            fixed_bytes: 1 << 20,
            state_bytes: 4 << 20,
        };
        let per = |gbs: f64| 1.0 / (gbs * 1e9);
        joint::Problem {
            layers: vec![layer; 24],
            devices: vec![
                joint::Device {
                    capacity: 64 * gib,
                    fixed: 256 << 20,
                    rate: slowed(
                        &[(QuantPolicy::Q4, per(37.4)), (QuantPolicy::Q8, per(15.1))],
                        host,
                    ),
                    resident: vec![],
                },
                joint::Device {
                    capacity: card_capacity,
                    fixed: gib,
                    rate: slowed(
                        &[(QuantPolicy::Q4, per(14.0)), (QuantPolicy::Q8, per(30.2))],
                        card,
                    ),
                    resident: vec![],
                },
            ],
            crossing_s: 0.000_628,
            disk_s_per_byte: 1.0 / 150e6,
            horizon_tokens: IDLE_HORIZON_TOKENS,
            tolerance: PRECISION_TOLERANCE,
            floor: QuantPolicy::Q4,
        }
    }

    fn on_card(a: &joint::Assignment) -> usize {
        a.layers.iter().filter(|c| c.device == 1).count()
    }

    /// The whole-system schedule, at the solver: a free host is the faster
    /// place for a small model, a host whose cores another process holds
    /// sends the layers to the card, and a card another process is using
    /// (capacity 0 while yielding) sends them back — onto a host that is
    /// itself busy, which is "almost nothing", but nothing is stolen.
    #[test]
    fn layers_follow_whichever_device_nobody_else_wants() {
        let gib = 1u64 << 30;
        let free = joint::solve(&small_model(1.0, 1.0, 12 * gib));
        assert_eq!(on_card(&free.assignment), 0, "a free host wins for a 2B");
        let busy_host = joint::solve(&small_model(0.02, 1.0, 12 * gib));
        assert_eq!(
            on_card(&busy_host.assignment),
            24,
            "a busy host sends it all to the card"
        );
        let both = joint::solve(&small_model(0.02, 0.0, 0));
        assert_eq!(on_card(&both.assignment), 0, "a yielded card holds nothing");
        assert!(both.feasible);
        assert!(
            both.time_s > free.time_s * 5.0,
            "and it is slow, which is the point"
        );
    }

    /// The reference box on 2026-10-02: other routines take the host for
    /// 30 s and give it back for 30 s, over and over. Planned on the last
    /// few seconds, the 2B's best placement flips between all-host and
    /// all-card with them — what drove the live 27B's 72 moves. Planned on
    /// the settled view it is one placement, held through every flip once
    /// the window has filled.
    #[test]
    fn minute_scale_load_does_not_move_layers() {
        let gib = 1u64 << 30;
        let t0 = Instant::now();
        let mut settle = Settle::new();
        let mut instant = std::collections::BTreeSet::new();
        let mut placed: Option<joint::Assignment> = None;
        let mut held = 0;
        for i in 0..600u64 {
            let at = t0 + Duration::from_secs(2 * i);
            let busy = (i / 15) % 2 == 0;
            let p = crate::sysmon::Pressure {
                cpu_free: if busy { 0.06 } else { 0.8 },
                gpu_free: 0.7,
                ..crate::sysmon::Pressure::default()
            };
            settle.observe(at, &p, None);
            let now = joint::solve(&small_model(p.cpu_free, p.gpu_free, 6 * gib));
            instant.insert(on_card(&now.assignment));
            if 2 * i < SETTLE_WINDOW.as_secs() {
                continue;
            }
            let (host, card) = settle.slowdowns(&p);
            let pb = small_model_slowed(host, card, 6 * gib);
            match &placed {
                None => placed = Some(joint::solve(&pb).assignment),
                Some(a) => {
                    let (v, _) = joint::replan(&pb, a);
                    assert_eq!(v, joint::Verdict::Hold, "tick {i}");
                    held += 1;
                }
            }
        }
        assert!(
            instant.len() > 1,
            "the last few seconds do flip: {instant:?}"
        );
        assert_eq!(held, 299);
    }

    /// The window keeps ten minutes and no more, its mean is over what it
    /// keeps, and its least counts the reading at hand.
    #[test]
    fn the_settle_window_forgets_what_is_older_than_ten_minutes() {
        let t0 = Instant::now();
        let mut w = Window::<u64>::new();
        w.push(t0, 3);
        w.push(t0 + Duration::from_secs(300), 9);
        assert_eq!(w.least(5), 3);
        w.push(t0 + SETTLE_WINDOW + Duration::from_secs(1), 7);
        assert_eq!(w.values().collect::<Vec<_>>(), vec![9, 7]);
        assert_eq!(w.least(8), 7);
        assert_eq!(w.least(2), 2);
        let mut m = Window::<f64>::new();
        assert!(m.mean().is_none());
        m.push(t0, 1.0);
        m.push(t0, 4.0);
        assert!((m.mean().unwrap() - 2.5).abs() < 1e-12);
    }

    /// A yielded card leaves its rate and room samples out: while yielding
    /// it holds nothing at all, which repair handles at once, and averaging
    /// the 50x slowdown of "no share" would keep layers off it for ten
    /// minutes after the co-tenant left.
    #[test]
    fn a_yielded_card_is_not_averaged_in() {
        let t0 = Instant::now();
        let mut settle = Settle::new();
        let free = crate::sysmon::Pressure {
            gpu_free: 0.5,
            ..crate::sysmon::Pressure::default()
        };
        let yielded = crate::sysmon::Pressure {
            gpu: crate::sysmon::Yield::Yield,
            gpu_free: 0.0,
            ..crate::sysmon::Pressure::default()
        };
        settle.observe(t0, &free, Some(4 << 30));
        settle.observe(t0 + Duration::from_secs(2), &yielded, Some(2 << 30));
        let (_, card) = settle.slowdowns(&free);
        assert!((card - 2.0).abs() < 1e-12);
        assert_eq!(settle.host_room.least(8 << 30), 2 << 30);
    }

    #[test]
    fn a_contended_rate_is_divided_by_the_free_speed_and_floored() {
        let r = [(QuantPolicy::Q4, 1e-9)];
        let t = Scaling::TimeSliced;
        assert!((contended(&r, 1.0, t)[0].1 - 1e-9).abs() < 1e-18);
        assert!((contended(&r, 0.5, t)[0].1 - 2e-9).abs() < 1e-18);
        assert!((contended(&r, 0.0, t)[0].1 - 1e-9 / MIN_FREE_SHARE).abs() < 1e-15);
        // The host loses less than its share of cores (DRAM-bound).
        let m = Scaling::MemoryBound;
        assert!((contended(&r, 0.25, m)[0].1 - 2e-9).abs() < 1e-18);
    }

    /// A probe taken on a busy device is stored as the free device's rate,
    /// so applying the same share again gets the probe back — counted once.
    #[test]
    fn a_probe_under_contention_is_counted_once() {
        let probed = [(QuantPolicy::Q4, 3e-9), (QuantPolicy::Q8, 7e-9)];
        for how in [Scaling::TimeSliced, Scaling::MemoryBound] {
            for free in [1.0, 0.6, 0.3, 0.05] {
                let stored = uncontended(&probed, free, how);
                let back = contended(&stored, free, how);
                for (a, b) in back.iter().zip(&probed) {
                    assert!((a.1 - b.1).abs() < 1e-18, "{how:?} {free}");
                }
                // And a free device is faster than the busy one it was probed on.
                assert!(stored[0].1 <= probed[0].1);
            }
        }
    }

    /// KV grows with context on attention layers; recurrent state does not.
    #[test]
    fn state_follows_context_only_where_it_should() {
        let cfg = cfg_27b();
        let attn = (0..cfg.num_layers).find(|&l| cfg.is_attention(l)).unwrap();
        let delta = (0..cfg.num_layers).find(|&l| !cfg.is_attention(l)).unwrap();
        assert!(state_bytes(&cfg, attn, 8192) > state_bytes(&cfg, attn, 1024));
        assert_eq!(
            state_bytes(&cfg, delta, 8192),
            state_bytes(&cfg, delta, 1024)
        );
        assert!(state_bytes(&cfg, delta, 1) > 0);
    }

    /// The card's prefill transients against what the Bonsai 27B's pool did
    /// (2026-10-04, 1024-token chunks): a 4786-token prefill grew it 1.73 GiB
    /// with 0.08 of it the layers' state. The model covers that, so the
    /// residual is left with only what it cannot explain; it grows with the
    /// context, where the flat residual fell short of a long prompt; and a
    /// span the packed GEMV takes row by row dequantizes nothing.
    #[test]
    fn prefill_transients_cover_what_the_card_measured() {
        let cfg = cfg_27b();
        let gib = |b: u64| f64_from_u64(b) / f64::from(1u32 << 30);
        if mummu::decode::prefill_chunk_len() == 1024 {
            let long = gib(act_bytes(&cfg, 4786) + transient_bytes(&cfg, 4786));
            assert!(long >= 1.73 - 0.08, "{long:.2} GiB");
            assert!(long <= 2.2, "and not wildly over: {long:.2} GiB");
        }
        let at = |ctx| transient_bytes(&cfg, ctx);
        assert!(at(16_384) > at(4096) && at(4096) > at(1024));
        let hidden = cfg.hidden_size as u64;
        let dequant = hidden * cfg.intermediate_size as u64 * 4;
        assert!(
            at(GEMV_ROWS) < dequant,
            "no dequantized weight at {GEMV_ROWS} rows"
        );
        assert!(at(GEMV_ROWS + 1) >= dequant);
    }

    /// ε̂ is recalled from its own key only: the old key's 5.65 GiB held the
    /// transients too, and read back as the residual it would keep them off
    /// the card for good.
    #[test]
    fn an_older_residual_is_not_recalled() {
        let old = r#"{"residual_bytes": 6068741336}"#;
        assert_eq!(remembered_residual(old), None);
        let new = format!(r#"{{"residual_bytes": 6068741336, "{RESIDUAL_KEY}": 2000000000}}"#);
        assert_eq!(remembered_residual(&new), Some(2_000_000_000));
        assert_eq!(remembered_residual("not json"), None);
    }

    /// Prefill is chunked, so the activation term stops growing at the chunk.
    #[test]
    fn activations_are_bounded_by_the_prefill_chunk() {
        let cfg = cfg_27b();
        let chunk = mummu::decode::prefill_chunk_len();
        assert_eq!(act_bytes(&cfg, chunk), act_bytes(&cfg, chunk * 8));
        assert!(act_bytes(&cfg, 16) < act_bytes(&cfg, chunk));
    }

    /// Capacity is the card less the guard on AMBIENT — our own reserved
    /// bytes are not ambient, so holding more of the card ourselves must not
    /// shrink what we are allowed to hold.
    #[test]
    fn our_own_bytes_are_not_ambient() {
        let gib = 1u64 << 30;
        let quiet = Card {
            total: 16 * gib,
            used: 3 * gib,
            reserved: 0,
            in_use: 0,
        };
        let busy_with_us = Card {
            total: 16 * gib,
            used: 12 * gib,
            reserved: 9 * gib,
            in_use: 8 * gib,
        };
        let a = capacity(&quiet);
        let b = capacity(&busy_with_us);
        assert_eq!(a, b, "same ambient, same capacity");
        assert!(a < 16 * gib - 3 * gib + 1);
    }

    /// Envelopes keep the max of the recent window and forget the old.
    #[test]
    fn envelopes_forget_what_left_the_window() {
        let mut e = Envelope::default();
        e.push(100);
        for _ in 0..ENVELOPE {
            e.push(1);
        }
        assert_eq!(e.max(), Some(1));
    }

    // -----------------------------------------------------------------
    // The post-release ambient window. Numbers are the 2026-09-23 incident:
    // a 16 GiB card, 3.1 GiB of desktop ambient, a 27B holding a 9.5 GiB
    // pool, dropped — and the driver still reporting all of it as used.
    // -----------------------------------------------------------------

    const MIB: u64 = 1 << 20;

    /// A reading of the incident card: `used` and `reserved` in MiB.
    fn reading(used: u64, reserved: u64) -> Card {
        Card {
            total: 16 * 1024 * MIB,
            used: used * MIB,
            reserved: reserved * MIB,
            in_use: reserved * MIB,
        }
    }

    /// A fresh, empty state for the correction.
    fn fresh() -> (Option<(u64, u64)>, Option<InFlight>) {
        (None, None)
    }

    /// The incident itself: the model is dropped, our pool reports zero, the
    /// driver still attributes 9.5 GiB to us — and that must NOT read as a
    /// co-tenant that just took three quarters of the card.
    #[test]
    fn a_model_drop_is_not_a_co_tenant_arriving() {
        let (mut last, mut flight) = fresh();
        let t = Instant::now();

        let resident = correct_ambient(&reading(12_600, 9_500), &mut last, &mut flight, t);
        assert_eq!(resident, 3_100 * MIB, "3.1 GiB of desktop, ours excluded");

        let dropped = correct_ambient(&reading(12_600, 0), &mut last, &mut flight, t);
        assert_eq!(
            dropped,
            3_100 * MIB,
            "the 9.5 GiB the driver has not reclaimed is still ours"
        );
        assert!(flight.is_some(), "the window is open");
    }

    /// The driver gives the pages back a few at a time; the credit shrinks
    /// with them and the window closes by itself once the card is level.
    #[test]
    fn the_credit_shrinks_as_the_driver_returns_pages() {
        let (mut last, mut flight) = fresh();
        let t = Instant::now();
        correct_ambient(&reading(12_600, 9_500), &mut last, &mut flight, t);
        correct_ambient(&reading(12_600, 0), &mut last, &mut flight, t);

        let half_back = correct_ambient(&reading(7_000, 0), &mut last, &mut flight, t);
        assert_eq!(half_back, 3_100 * MIB, "still ours, just less of it");
        assert_eq!(flight.map(|f| f.bytes), Some(3_900 * MIB));

        let level = correct_ambient(&reading(3_100, 0), &mut last, &mut flight, t);
        assert_eq!(level, 3_100 * MIB);
        assert!(flight.is_none(), "settled: nothing left to credit back");
    }

    /// The window credits back what WE released and not one byte more: a
    /// co-tenant that arrives while it is open still moves ambient, so the
    /// guard's "up at once" property survives the correction.
    #[test]
    fn a_co_tenant_arriving_during_the_window_is_still_believed() {
        let (mut last, mut flight) = fresh();
        let t = Instant::now();
        correct_ambient(&reading(12_600, 9_500), &mut last, &mut flight, t);
        correct_ambient(&reading(12_600, 0), &mut last, &mut flight, t);

        // +2 GiB on top of the bytes we have not been given back.
        let intruder = correct_ambient(&reading(14_600, 0), &mut last, &mut flight, t);
        assert_eq!(intruder, 5_100 * MIB, "3.1 desktop + 2.0 of somebody else");
    }

    /// The credit is a settling window, not an entitlement: a reading still
    /// high after it expires is believed in full, because by then it is not
    /// ours.
    #[test]
    fn the_window_expires_and_the_raw_reading_is_believed() {
        let (mut last, mut flight) = fresh();
        let t = Instant::now();
        correct_ambient(&reading(12_600, 9_500), &mut last, &mut flight, t);
        correct_ambient(&reading(12_600, 0), &mut last, &mut flight, t);

        let later = t + RELEASE_SETTLE + Duration::from_secs(1);
        let raw = correct_ambient(&reading(12_600, 0), &mut last, &mut flight, later);
        assert_eq!(raw, 12_600 * MIB, "no longer explainable as ours");
        assert!(flight.is_none());
    }

    /// The escalation converges. Doubling from the prior on the reference
    /// card reaches the ceiling in one step and stays there however many
    /// failures follow — the ratchet that emptied the card is bounded.
    #[test]
    fn the_working_set_estimate_stops_doubling_at_half_the_card() {
        let total = 16 * 1024 * MIB;
        let card = Some(total);
        let ceiling = residual_ceiling(card);
        assert_eq!(ceiling, 8 * 1024 * MIB);

        let mut e = residual_prior(card);
        assert_eq!(e, 4 * 1024 * MIB);
        e = escalate_residual(e, card);
        assert_eq!(e, ceiling, "the one doubling that carries information");
        for _ in 0..8 {
            e = escalate_residual(e, card);
            assert_eq!(e, ceiling, "and no more");
        }
        assert!(e < total, "a working set the card can still hold");
    }

    /// The ceiling is never below the prior, or the very first load would
    /// start out clamped.
    #[test]
    fn the_ceiling_leaves_room_for_the_prior_on_a_small_card() {
        for gib in [1u64, 2, 4, 8, 16, 24, 48] {
            let card = Some(gib * 1024 * MIB);
            assert!(
                residual_ceiling(card) >= residual_prior(card),
                "{gib} GiB card: ceiling below its own prior"
            );
        }
    }

    /// Pool jitter is not a release. A window opened on every small wobble
    /// would sit open forever and hide a real arrival.
    #[test]
    fn a_small_pool_fluctuation_opens_no_window() {
        let (mut last, mut flight) = fresh();
        let t = Instant::now();
        correct_ambient(&reading(12_600, 9_500), &mut last, &mut flight, t);

        let jitter = correct_ambient(&reading(12_500, 9_400), &mut last, &mut flight, t);
        assert_eq!(jitter, 3_100 * MIB);
        assert!(flight.is_none(), "100 MiB is under the release floor");
    }

    /// A probe slowed by something on the machine never lowers a rate: per
    /// level the faster of now and the remembered best is kept, a level not
    /// measured now stays out, and a slow probe is recognized as one.
    #[test]
    fn rates_keep_the_best_the_device_has_shown() {
        let ns = |gbps: f64| 1.0 / (gbps * 1e9);
        let remembered = [(QuantPolicy::Q4, ns(109.0)), (QuantPolicy::Q8, ns(309.0))];
        // The 2B's contended load: card Q4 64 GB/s, Q8 90.
        let measured = [(QuantPolicy::Q4, ns(64.0)), (QuantPolicy::Q8, ns(90.0))];
        let best = best_rates(&measured, &remembered);
        assert_eq!(best, remembered.to_vec());
        assert!(slower_than_remembered(
            &measured,
            &remembered,
            REPROBE_MARGIN
        ));
        // A faster probe wins; a level the probe did not produce is dropped.
        let faster = [(QuantPolicy::Q8, ns(400.0))];
        assert_eq!(
            best_rates(&faster, &remembered),
            vec![(QuantPolicy::Q8, ns(400.0))]
        );
        assert!(!slower_than_remembered(
            &faster,
            &remembered,
            REPROBE_MARGIN
        ));
        // Within the margin is noise, not contention.
        let noisy = [(QuantPolicy::Q8, ns(260.0))];
        assert!(!slower_than_remembered(&noisy, &remembered, REPROBE_MARGIN));
    }

    /// The evidence file holds the residual and each build's rates side by
    /// side: writing one keeps the other.
    #[test]
    fn evidence_keeps_rates_and_residual_together() {
        let dir = std::env::temp_dir().join(format!("mummu-rates-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("placement-x.json");
        update_evidence(&path, |e| e[RESIDUAL_KEY] = serde_json::json!(123));
        let host = [(QuantPolicy::Q4, 2.5e-11), (QuantPolicy::F16, 9.0e-11)];
        let accel = [(QuantPolicy::Q8, 3.2e-12)];
        remember_rates(&path, BackendChoice::Wgpu, &host, &accel);
        update_evidence(&path, |e| e[RESIDUAL_KEY] = serde_json::json!(456));
        let [h, a] = recall_rates(&path, BackendChoice::Wgpu).expect("rates kept");
        assert_eq!([h, a], [host.to_vec(), accel.to_vec()]);
        let text = std::fs::read_to_string(&path).expect("written");
        let v: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(v[RESIDUAL_KEY].as_u64(), Some(456));
        assert!(recall_rates(&dir.join("absent.json"), BackendChoice::Wgpu).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A re-probe schedule waits 15 s, then doubles; it settles once a
    /// second try confirms the best (the first may have raised it), and
    /// gives up after six.
    #[test]
    fn a_reprobe_settles_on_the_second_confirmation() {
        let t0 = Instant::now();
        let mut r = Reprobe::wanted(t0);
        assert!(!r.is_due(t0));
        assert!(r.is_due(t0 + REPROBE_FIRST));
        r.after(true, t0);
        assert_eq!(
            (r.tries, r.due),
            (1, Some(t0 + 2 * REPROBE_FIRST)),
            "one confirmation is not enough"
        );
        r.after(true, t0);
        assert_eq!(r.due, None, "settled");
        let mut slow = Reprobe::wanted(t0);
        for _ in 0..REPROBE_TRIES {
            assert!(slow.due.is_some());
            slow.after(false, t0);
        }
        assert_eq!(slow.due, None, "gave up after {REPROBE_TRIES}");
        assert!(slow.wait <= REPROBE_MAX_WAIT);
    }

    /// The card is re-probed on a busy host as long as two cores are idle
    /// and nobody else is on the card; the host only when half of it is free.
    #[test]
    fn the_card_is_reprobed_on_a_busy_host() {
        let reading = |others: f64, gpu_others: Option<u32>| crate::sysmon::Pressure {
            cpu: Some(crate::sysmon::CpuShare {
                cores: 16,
                ours: 0.0,
                others,
            }),
            gpu_others,
            gpu_others_age_s: gpu_others.map(|_| 1),
            cpu_free: 1.0 - others / 16.0,
            ..crate::sysmon::Pressure::default()
        };
        // This box on a normal day: other routines on 11 of 16 cores.
        assert_eq!(probe_gates(&reading(11.0, Some(2)), false), [false, true]);
        assert_eq!(probe_gates(&reading(4.0, Some(2)), false), [true, true]);
        assert_eq!(
            probe_gates(&reading(15.0, Some(2)), false),
            [false, false],
            "oversubscribed"
        );
        assert_eq!(
            probe_gates(&reading(4.0, Some(60)), false),
            [true, false],
            "a co-tenant on the card"
        );
        assert_eq!(
            probe_gates(&reading(4.0, Some(2)), true),
            [true, false],
            "yielded"
        );
        assert_eq!(
            probe_gates(&reading(4.0, None), false),
            [true, false],
            "no clean card reading yet"
        );
    }
}
