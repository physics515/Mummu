//! Where the bytes are going: the `status` object on `GET /api/logs`.
//!
//! The log feed answers "is it doing anything?". This answers the two
//! questions the operator asked next, by name — **how far along is the load**
//! and **how full are RAM and VRAM while it lands**. It rides the poll the two
//! pages already make once a second rather than adding a second loop, so a
//! browser tab costs exactly what it cost before.
//!
//! Three sources, none of which may be allowed to cost anything:
//!
//! * [`mummu::progress`] — atomics the loader writes once per tensor. Free.
//! * `/proc/meminfo` and `/proc/self/statm` — two small text reads.
//! * NVML, through [`mummu::vram`] — two FFI calls into the driver.
//!
//! The last two are cheap but not free, and they are behind an endpoint any
//! client may poll as fast as it likes, from as many tabs as it likes. So
//! every reading goes through [`memory`], which samples at most once a second
//! and hands every caller in that second the same numbers. Ten tabs at 1 Hz
//! cost one `/proc` read and one NVML call per second between them, not
//! twenty.
//!
//! # The two sources are cached differently, because they fail differently
//!
//! `/proc` is a kernel interface that answers or does not; a read of
//! `meminfo` cannot park for seconds. So [`host_sample`] is an ordinary
//! time-to-live cache and holds its lock across the read.
//!
//! NVML is a call into the **NVIDIA driver**, and it takes a driver-wide lock
//! to make it. On a wedged card `nvmlDeviceGetMemoryInfo` blocks — for
//! seconds, for as long as the reset takes, sometimes until the module is
//! reloaded. Sampling that under a shared mutex put every `/api/logs` poll
//! from every tab in a queue behind one hung FFI call, which breaks the one
//! page that has to keep answering precisely when the GPU is dead: `/logs` is
//! where you go to find out *why* it is dead. So the VRAM reading goes
//! through [`Stale`], which answers instantly from the last sample and does
//! the refresh on a thread of its own. A driver that never answers costs a
//! stale VRAM gauge and one parked thread; it does not cost the endpoint.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// The crate version, which is the workspace version — bumped to 0.3.0 for
/// the release this ships in. It had sat at 0.1.0 through the v0.1.0, v0.1.1
/// and v0.2.0 tags, which is exactly why nobody could tell what was deployed.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The git short sha this binary was built from, `<sha>-dirty` when it was
/// built from uncommitted changes, or `unknown` where nothing named it.
///
/// Set by `build.rs` from the repository, or — in the Docker build, which has
/// the sources but deliberately not the repository — from the
/// `MUMMU_BUILD_SHA` build argument the compose service passes. See
/// `src/build_sha.rs` for why a passed-in stamp outranks git, and why
/// `unknown` must remain reachable rather than becoming a build failure.
pub const BUILD: &str = env!("MUMMU_BUILD_SHA");

/// How long one memory reading stands in for the next.
///
/// Matched to the pages' own 1 Hz floor: sampling faster than the fastest
/// well-behaved client redraws would be work nobody can see, and sampling
/// slower would make the RAM gauge visibly lag the bar beside it during the
/// minute when both are moving.
const SAMPLE_TTL: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Host memory
// ---------------------------------------------------------------------------

/// What the host has and what this process is holding of it.
///
/// All three are needed to answer the question being asked. `rss` alone says
/// what mummu took but not whether the machine can afford it; `available`
/// alone moves for reasons that have nothing to do with mummu (this box runs
/// ~63 other containers); `total` is what turns either into a percentage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    /// Resident set size of this process, in bytes.
    pub rss: u64,
    /// `MemAvailable`: what a new allocation could get without swapping.
    pub available: u64,
    /// `MemTotal`: physical RAM.
    pub total: u64,
}

/// Host memory, or `None` off linux.
///
/// # Not available elsewhere
///
/// Both readings come from `/proc`, which is a linux interface. Windows has
/// `GlobalMemoryStatusEx` and macOS has `host_statistics64`, and neither is
/// worth an FFI declaration here: the deployment this exists for is a linux
/// container, and the desktop shell renders "RAM n/a" from the same `null`
/// the VRAM field already uses when nothing will say. An honest absence beats
/// a number from a third code path nobody runs.
#[must_use]
pub fn host_memory() -> Option<HostMemory> {
    let (available, total) = meminfo()?;
    Some(HostMemory {
        rss: rss_bytes().unwrap_or(0),
        available,
        total,
    })
}

/// `MemAvailable` and `MemTotal`, in bytes, from ONE read of `/proc/meminfo`.
///
/// One parse for both fields on purpose: this is the function `engine` used to
/// carry as `mem_available_bytes`, and the host-pressure watcher calls it
/// every 5 seconds forever. Reading the file twice to get two numbers out of
/// it would double that for nothing.
fn meminfo() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<u64> {
        text.lines()
            .find(|l| l.starts_with(name))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
            .map(|kb| kb * 1024)
    };
    Some((field("MemAvailable:")?, field("MemTotal:")?))
}

/// Linux `MemAvailable` in bytes (`None` elsewhere).
///
/// The fit planner's and the host-pressure watcher's reading, kept here beside
/// the one the status object uses so there is a single parser rather than two
/// that can disagree about what "available" means.
#[must_use]
pub fn mem_available_bytes() -> Option<u64> {
    meminfo().map(|(available, _)| available)
}

/// This process's resident set size in bytes.
///
/// From `/proc/self/statm`, not `/proc/self/status`: statm is two numbers on
/// one line (total pages, resident pages) with no formatting to parse and no
/// kernel-version-dependent field list, where `status` is ~50 lines that must
/// be scanned for `VmRSS:`. Both report the same figure.
///
/// The page size is read from `getconf`-free arithmetic: 4 KiB is the page
/// size on x86-64 and aarch64 linux, the only targets this server is built
/// for. A hypothetical 16 KiB-page host would read 4x low — visible, wrong in
/// the safe direction (it under-claims), and not worth an FFI call to
/// `sysconf` to rule out.
fn rss_bytes() -> Option<u64> {
    const PAGE: u64 = 4096;
    let text = std::fs::read_to_string("/proc/self/statm").ok()?;
    text.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|p| p * PAGE)
}

// ---------------------------------------------------------------------------
// The sample cache
// ---------------------------------------------------------------------------

/// One reading of both memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Memory {
    pub host: Option<HostMemory>,
    pub vram: Option<mummu::vram::Memory>,
}

/// A reading that is always answered from the last sample, and refreshed on a
/// thread of its own — so no caller ever waits on the source.
///
/// # Why a source gets this instead of a plain TTL cache
///
/// A TTL cache holds its lock across the sample, which is right when the
/// source is a file read and wrong when the source can *block*. NVML takes
/// the driver's lock; on a wedged GPU the call does not return for seconds.
/// Under a shared mutex that one call becomes the whole endpoint's latency,
/// for every tab, and `/logs` stops answering at the exact moment it is the
/// only thing worth looking at.
///
/// So: whoever finds the sample stale starts ONE refresh and is handed the
/// previous reading immediately, as is everyone who arrives while it runs.
/// The cost of a source that never answers is a gauge that stops moving and
/// one parked thread — never a stalled request.
///
/// A thread rather than a `spawn_blocking`: this is called from the tests too,
/// where there is no tokio runtime to spawn into, and one thread per stale
/// poll (so at most one per [`SAMPLE_TTL`], and none at all while nobody is
/// watching) is cheaper than the one request it would otherwise delay.
struct Stale<T: Copy + Send + 'static> {
    state: Mutex<StaleState<T>>,
}

struct StaleState<T> {
    sample: Option<(Instant, T)>,
    /// A refresh is in flight. Single-flight: ten tabs arriving together make
    /// one call into the driver, not ten.
    refreshing: bool,
}

/// Clears [`StaleState::refreshing`] however the refresh ends.
///
/// A `Drop` and not a line at the end of the thread body, because a sampler
/// that panics would otherwise leave the flag set forever and freeze the
/// reading for the life of the process — the same shape as the NVML resolve
/// that cached its own failure.
struct Refreshing<T: Copy + Send + 'static>(&'static Stale<T>);

impl<T: Copy + Send + 'static> Drop for Refreshing<T> {
    fn drop(&mut self) {
        self.0.lock().refreshing = false;
    }
}

impl<T: Copy + Send + 'static> Stale<T> {
    const fn new() -> Self {
        Self {
            state: Mutex::new(StaleState {
                sample: None,
                refreshing: false,
            }),
        }
    }

    /// A poisoned cache still holds a perfectly good reading, and refusing to
    /// draw a gauge because a *sample* lock was poisoned would be absurd.
    fn lock(&self) -> std::sync::MutexGuard<'_, StaleState<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The last reading — `None` only until the first refresh lands — kicking
    /// off a refresh when the one we have is older than `ttl`.
    ///
    /// Returns without waiting, always. `name` is what the refresh thread is
    /// called in a stack dump, which is the whole story when one is wedged.
    fn get(&'static self, ttl: Duration, name: &'static str, sample: fn() -> T) -> Option<T> {
        self.get_at(ttl, name, sample).map(|(_, v)| v)
    }

    /// The same reading, with the instant it was taken.
    ///
    /// A gauge does not care how old its number is — it is redrawn a second
    /// later either way. A *measurement* does: `engine::certify_residency`
    /// subtracts a before from an after, and handing it the same sample twice
    /// would report zero bytes resident and cry residency failure at a
    /// perfectly good load. So the timestamp travels with the value and the
    /// caller decides whether this reading is new enough to mean anything.
    fn get_at(
        &'static self,
        ttl: Duration,
        name: &'static str,
        sample: fn() -> T,
    ) -> Option<(Instant, T)> {
        let mut st = self.lock();
        if let Some((at, v)) = st.sample
            && at.elapsed() < ttl
        {
            return Some((at, v));
        }
        let stale = st.sample;
        if st.refreshing {
            return stale; // someone is already asking; do not ask again
        }
        st.refreshing = true;
        drop(st);
        let spawned = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                // Armed BEFORE the call that can block or panic.
                let _flag = Refreshing(self);
                let fresh = sample();
                self.lock().sample = Some((Instant::now(), fresh));
            });
        if spawned.is_err() {
            // Out of threads. Clear the flag by hand or the reading never
            // refreshes again, which is a worse failure than this one.
            self.lock().refreshing = false;
        }
        stale
    }

    /// Wait — bounded — for a reading taken after `since`.
    ///
    /// The wait is on THIS cache, never on the source: each poll hands back
    /// whatever has landed and asks for a refresh if the sample is stale, so a
    /// driver that never answers costs the caller `budget` and not a minute.
    /// `None` means no fresh reading arrived in time, which is a fact the
    /// caller must report rather than paper over with the stale one.
    ///
    /// It sleeps, so it belongs on a thread of its own — never on an async
    /// runtime and never under a lock somebody else needs.
    fn wait_after(
        &'static self,
        since: Instant,
        budget: Duration,
        ttl: Duration,
        name: &'static str,
        sample: fn() -> T,
    ) -> Option<T> {
        let deadline = Instant::now() + budget;
        loop {
            if let Some((at, v)) = self.get_at(ttl, name, sample)
                && at > since
            {
                return Some(v);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(POLL_INTERVAL.min(budget));
        }
    }
}

/// How often [`Stale::wait_after`] looks again. Short enough that a healthy
/// driver's refresh (milliseconds) is not made to look slow, long enough that
/// waiting out a wedged one costs a handful of wakeups.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// The VRAM reading. Stale-served — NVML is the source that can block.
static VRAM: Stale<Option<mummu::vram::Memory>> = Stale::new();

/// The host reading. An ordinary TTL cache: `/proc` cannot park on a driver.
static HOST: Mutex<Option<(Instant, Option<HostMemory>)>> = Mutex::new(None);

/// Host memory, resampled at most once per [`SAMPLE_TTL`].
///
/// The lock IS held across the two `/proc` reads, deliberately: two clients
/// arriving together should produce one reading rather than two racing ones,
/// and the work under it is two small text files — microseconds, with no
/// driver anywhere near it.
fn host_sample() -> Option<HostMemory> {
    let mut slot = HOST.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, sample)) = *slot
        && at.elapsed() < SAMPLE_TTL
    {
        return sample;
    }
    let fresh = host_memory();
    *slot = Some((Instant::now(), fresh));
    fresh
}

/// The current memory reading. Never waits on the GPU driver.
#[must_use]
pub fn memory() -> Memory {
    Memory {
        host: host_sample(),
        // `flatten`: the outer `None` is "no sample has landed yet", the
        // inner one is "nothing on this machine will say". Both render as
        // "n/a", and neither is a card with no memory in it.
        vram: VRAM
            .get(SAMPLE_TTL, "mummu-vram", mummu::vram::memory)
            .flatten(),
    }
}

/// The VRAM reading for the MODEL-LOAD path: whatever has already been
/// sampled, and a refresh asked for if that is stale. Never a call into the
/// driver on this thread.
///
/// # Why the load path is the worst place of all to block
///
/// `nvmlDeviceGetMemoryInfo` takes the driver's lock, and on a wedged card it
/// does not come back for seconds — sometimes not until the module is
/// reloaded. Every other caller of NVML here is a gauge, and a gauge that
/// stops moving is a cosmetic failure. The load path is different in two ways
/// that compound:
///
/// * It runs with the **model slot lock held**, so a parked FFI call does not
///   stall one load, it stalls every request behind that slot — including the
///   `/logs` poll the operator opened *because* the GPU looks dead.
/// * It is the one moment that must not get slower. A cold 27B off the
///   spinning array is already minutes; the whole of v0.3.0 exists because
///   two of those minutes looked like a hang.
///
/// A gauge may be a second stale. A load may not be a second late, and it may
/// never be indefinitely late.
#[must_use]
pub fn vram_used() -> Option<u64> {
    VRAM.get_at(SAMPLE_TTL, "mummu-vram", mummu::vram::memory)
        .and_then(|(_, v)| v)
        .map(|m| m.used)
}

/// Wait, bounded and off the driver, for a VRAM reading taken after `since`.
///
/// The residency check subtracts a before from an after, so it needs a
/// reading that actually postdates the load — the cached one may have been
/// taken while the weights were still landing. `None` means none arrived
/// inside `budget`, which the caller reports as "unverified" rather than
/// turning into a number it did not measure.
///
/// Sleeps: call it from a thread that exists for this, never from an async
/// task and never while holding a lock.
#[must_use]
pub fn vram_used_after(since: Instant, budget: Duration) -> Option<u64> {
    VRAM.wait_after(since, budget, SAMPLE_TTL, "mummu-vram", mummu::vram::memory)
        .flatten()
        .map(|m| m.used)
}

/// Take the first readings now, before anything needs one.
///
/// [`Stale`] answers from the last sample and refreshes behind it, so the
/// very first call after boot has nothing to hand back. That is fine for a
/// gauge (it fills in a second later) and not fine for the first load's VRAM
/// baseline, which has exactly one chance to be taken. One call at startup,
/// on the refresh thread, and the cache is warm before a model ever lands.
pub fn prime() {
    let _ = memory();
}

// ---------------------------------------------------------------------------
// The wire shape
// ---------------------------------------------------------------------------

/// Round a float to one decimal for the wire.
///
/// Seconds are reported to a tenth because that is the precision a human reads
/// off a bar; shipping `91.23456789012345` would be noise in every log line,
/// every test assertion and every diff of this endpoint's output.
fn tenths(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// A finite number, or JSON `null`.
///
/// Every derived field here divides by something that can be zero, and
/// `serde_json` turns a NaN or an infinity into `null` silently — which would
/// make a bug look like a missing reading. Going through this makes the
/// absence deliberate and the arithmetic's guards the only thing producing it.
fn finite(v: Option<f64>) -> Value {
    match v {
        Some(v) if v.is_finite() => json!(v),
        _ => Value::Null,
    }
}

/// The `status` object carried by `GET /api/logs`.
///
/// # Shape
///
/// ```json
/// "status": {
///   "version": "0.3.0", "build": "abc1234",
///   "phase": "loading", "generation": 7, "model": "gemma3:27b",
///   "done": 673, "total": 851, "unit": "tensors", "step": 1,
///   "bytes": 15527000000,
///   "rate_bps": 169000000.0, "elapsed_s": 91.2, "eta_s": 24.0,
///   "host": {"rss": 22000000000, "available": 90000000000, "total": 133000000000},
///   "vram": {"used": 7900000000, "free": 9300000000, "total": 17170000000}
/// }
/// ```
///
/// Every field is always present. The ones that can be absent are `null`, not
/// missing and not zero — a client must be able to tell "no VRAM reading on
/// this machine" from "the card is empty", and "no ETA yet" from "zero
/// seconds left". `total` is `0` rather than `null` when a phase has no count
/// at all, which is the signal to render an indeterminate bar.
#[must_use]
pub fn to_json() -> Value {
    let p = mummu::progress::snapshot();
    let mem = memory();
    json!({
        "version": VERSION,
        "build": BUILD,
        "phase": p.phase.as_str(),
        // Which load these numbers belong to. A client that sees it change
        // knows the previous load is over, whatever the other fields say.
        "generation": p.generation,
        "model": if p.model.is_empty() { Value::Null } else { json!(p.model) },
        "done": p.done,
        "total": p.expected,
        // What the two counts above are counting. Three loaders feed this bar
        // and they count tensors, layers and experts respectively; the pages
        // used to print "tensors" under all three.
        "unit": p.unit.as_str(),
        // Which counted pass of this load the bar is on. Past 1 it is what
        // separates "the second pass started" from "the bar went backwards".
        "step": p.step,
        "bytes": p.bytes,
        "rate_bps": finite(p.rate_bps()),
        "elapsed_s": finite(p.elapsed_s().map(tenths)),
        "eta_s": finite(p.eta_s().map(tenths)),
        "host": mem.host.map_or(Value::Null, |h| json!({
            "rss": h.rss, "available": h.available, "total": h.total,
        })),
        "vram": mem.vram.map_or(Value::Null, |v| json!({
            "used": v.used, "free": v.free, "total": v.total,
        })),
    })
}

/// The build fields, for `GET /api/health`.
#[must_use]
pub fn build_json() -> (&'static str, &'static str) {
    (VERSION, BUILD)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract both pages are written against. A field disappearing or
    /// changing shape here silently breaks a bar that is only ever looked at
    /// while something is going wrong.
    #[test]
    fn the_status_object_always_carries_every_field() {
        let s = to_json();
        let o = s.as_object().expect("status is an object");
        for key in [
            "version",
            "build",
            "phase",
            "generation",
            "model",
            "done",
            "total",
            "unit",
            "step",
            "bytes",
            "rate_bps",
            "elapsed_s",
            "eta_s",
            "host",
            "vram",
        ] {
            assert!(o.contains_key(key), "status is missing {key}");
        }
        assert_eq!(s["version"], json!(VERSION));
        assert!(
            !s["build"].as_str().expect("build is a string").is_empty(),
            "an unknown build is the string \"unknown\", never empty"
        );
        // Counts are numbers even when there is nothing to count, so a client
        // never has to special-case their type.
        assert!(s["done"].is_u64() && s["total"].is_u64() && s["bytes"].is_u64());
        // The derived fields are a number or null — never NaN, never a string.
        for key in ["rate_bps", "elapsed_s", "eta_s"] {
            assert!(
                s[key].is_null() || s[key].is_f64() || s[key].is_u64(),
                "{key} must be a number or null, got {}",
                s[key]
            );
        }
    }

    /// Idle is the state a client sees for most of the server's life, and it
    /// must render as "ready"/"idle" beside two gauges — not as a bar stuck at
    /// 0% with an ETA of zero.
    #[test]
    fn an_idle_server_reports_no_counts_and_no_eta() {
        let _serial = crate::progress_serial();
        mummu::progress::idle();
        let s = to_json();
        assert_eq!(s["phase"], json!("idle"));
        assert_eq!(s["model"], Value::Null, "no model, not an empty string");
        assert_eq!(s["total"], json!(0), "0 total is the indeterminate signal");
        assert_eq!(s["eta_s"], Value::Null, "no eta invented out of nothing");
    }

    /// The gauges are drawn from these, so an inconsistent reading would draw
    /// a bar past its own track. Values are never asserted — this has to pass
    /// on a machine with no NVIDIA card and inside a container with a
    /// cgroup-limited view of memory.
    #[test]
    fn memory_readings_are_self_consistent_or_absent() {
        let m = memory();
        if let Some(h) = m.host {
            assert!(h.total > 0, "a machine with no RAM is a bad reading");
            assert!(
                h.available <= h.total,
                "available {} > total {}",
                h.available,
                h.total
            );
            assert!(h.rss > 0, "this test process is resident somewhere");
            assert!(h.rss <= h.total, "rss {} > total {}", h.rss, h.total);
        }
        if let Some(v) = m.vram {
            assert!(v.total > 0);
            assert!(
                v.used + v.free <= v.total + (64 << 20),
                "used {} + free {} overshoots total {}",
                v.used,
                v.free,
                v.total
            );
        }
    }

    /// The cache is the only thing standing between this endpoint and ten
    /// browser tabs hammering `/proc`. Two calls inside one TTL must be the
    /// same reading, byte for byte.
    #[test]
    fn host_readings_inside_the_ttl_are_the_same_sample() {
        let a = host_sample();
        let b = host_sample();
        assert_eq!(a, b, "a second poll must not resample");
    }

    /// The failure `/logs` exists for is a dead GPU, and a dead GPU is
    /// exactly when NVML stops returning. A caller must be handed the last
    /// reading and let go — never queued behind the driver.
    ///
    /// The sampler here blocks for 400 ms, which is what a wedged
    /// `nvmlDeviceGetMemoryInfo` does (only for longer). Twenty-one callers
    /// arrive during it; all twenty-one must return at once, and between them
    /// they must make ONE call, not twenty-one.
    #[test]
    fn a_blocked_source_is_never_waited_on_and_never_asked_twice() {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static SLOW: Stale<u64> = Stale::new();
        static CALLS: AtomicU64 = AtomicU64::new(0);
        fn wedged_driver() -> u64 {
            std::thread::sleep(Duration::from_millis(400));
            CALLS.fetch_add(1, Relaxed) + 1
        }
        let ttl = Duration::from_millis(50);

        let t = Instant::now();
        assert_eq!(
            SLOW.get(ttl, "test-wedged", wedged_driver),
            None,
            "nothing sampled yet, and the honest answer is not to wait for one"
        );
        for _ in 0..20 {
            assert_eq!(SLOW.get(ttl, "test-wedged", wedged_driver), None);
        }
        assert!(
            t.elapsed() < Duration::from_millis(200),
            "21 polls waited {:?} on a source that blocks for 400 ms",
            t.elapsed()
        );

        // And the reading does land, once the source finally answers.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut landed = None;
        while Instant::now() < deadline {
            if let Some(v) = SLOW.get(ttl, "test-wedged", wedged_driver) {
                landed = Some(v);
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(landed, Some(1), "one refresh between 21 callers, not 21");
    }

    /// MINOR 4: the count the bar draws has to say what it counts. Three
    /// loaders feed it — tensors, layers, experts — and the pages printed
    /// "tensors" under all three until the unit rode along with the count.
    #[test]
    fn the_counts_carry_the_unit_and_the_pass_they_belong_to() {
        let _serial = crate::progress_serial();
        mummu::progress::begin(
            mummu::progress::Phase::Loading,
            64,
            mummu::progress::Unit::Layers,
        );
        mummu::progress::advance(12, 1 << 30);
        let s = to_json();
        assert_eq!(s["unit"], json!("layers"), "12/64 layers, not tensors");
        assert_eq!(s["done"], json!(12));
        assert_eq!(s["total"], json!(64));
        assert!(s["step"].is_u64(), "the pass number is a number");
        mummu::progress::idle();
        // Idle still answers with a unit rather than a missing field: every
        // field of this object is always present (see above).
        assert!(to_json()["unit"].is_string());
    }

    /// MAJOR 3: the load path reads VRAM through this cache, and the waiting
    /// it does when it needs a reading NEWER than something must be bounded
    /// by its own budget — never by whether the driver ever answers. A wedged
    /// `nvmlDeviceGetMemoryInfo` parks for seconds; a load holding the model
    /// slot lock cannot park with it.
    #[test]
    fn a_wedged_source_never_holds_a_waiter_past_its_budget() {
        static WEDGED: Stale<u64> = Stale::new();
        fn never_answers() -> u64 {
            std::thread::sleep(Duration::from_secs(30));
            0
        }
        let budget = Duration::from_millis(300);
        let t = Instant::now();
        let got = WEDGED.wait_after(
            Instant::now(),
            budget,
            Duration::from_millis(10),
            "test-wedged-wait",
            never_answers,
        );
        assert_eq!(got, None, "no reading arrived, and none may be invented");
        assert!(
            t.elapsed() < budget * 4,
            "waited {:?} on a {budget:?} budget",
            t.elapsed()
        );
    }

    /// And when the source does answer, the wait returns the reading that
    /// actually postdates the caller's mark — not the stale one it started
    /// with, which for `certify_residency` would be "before minus before" and
    /// a residency alarm on a load that worked.
    #[test]
    fn a_wait_returns_only_a_reading_newer_than_its_mark() {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static SOURCE: Stale<u64> = Stale::new();
        static NEXT: AtomicU64 = AtomicU64::new(1);
        fn counted() -> u64 {
            NEXT.fetch_add(1, Relaxed)
        }
        let ttl = Duration::from_millis(10);

        // Land a first reading, and mark the moment after it.
        let deadline = Instant::now() + Duration::from_secs(5);
        while SOURCE.get(ttl, "test-mark", counted).is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let first = SOURCE.get(ttl, "test-mark", counted).expect("a reading");
        let mark = Instant::now();

        let after = SOURCE
            .wait_after(mark, Duration::from_secs(5), ttl, "test-mark", counted)
            .expect("the source answers, so a fresh reading must arrive");
        assert!(
            after > first,
            "handed back a reading from before the mark: {after} after {first}"
        );
    }

    /// A stale reading is served while the refresh runs — the gauge keeps
    /// showing the last thing the card said rather than blanking to "n/a"
    /// every time the TTL expires.
    #[test]
    fn a_refresh_serves_the_previous_reading_rather_than_nothing() {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static SOURCE: Stale<u64> = Stale::new();
        static NEXT: AtomicU64 = AtomicU64::new(1);
        fn slowish() -> u64 {
            std::thread::sleep(Duration::from_millis(150));
            NEXT.fetch_add(1, Relaxed)
        }
        let ttl = Duration::from_millis(30);

        let deadline = Instant::now() + Duration::from_secs(5);
        while SOURCE.get(ttl, "test-slowish", slowish).is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            SOURCE.get(ttl, "test-slowish", slowish),
            Some(1),
            "the first reading landed"
        );
        // Past the TTL: a refresh starts, and the caller still gets a number.
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(
            SOURCE.get(ttl, "test-slowish", slowish),
            Some(1),
            "stale, but a reading — never a hole in the gauge"
        );
    }

    /// `MemAvailable` is what the fit planner and the host-pressure watcher
    /// read; it moved here so there is one parser. It must still answer.
    #[test]
    fn mem_available_agrees_with_the_full_reading() {
        let (Some(a), Some(h)) = (mem_available_bytes(), host_memory()) else {
            return; // not linux
        };
        // Sampled a moment apart, so they move — but not by gigabytes, and
        // never past the machine's own size.
        assert!(a > 0 && a <= h.total);
    }

    #[test]
    fn tenths_rounds_for_humans_not_for_floats() {
        assert!((tenths(91.23456) - 91.2).abs() < 1e-9);
        assert!((tenths(24.05) - 24.1).abs() < 1e-9);
        assert!((tenths(0.0) - 0.0).abs() < 1e-9);
    }

    /// An infinity reaching the wire as `null` would be indistinguishable from
    /// "no reading"; the guard makes the absence deliberate.
    #[test]
    fn non_finite_numbers_never_reach_the_wire_as_numbers() {
        assert_eq!(finite(Some(f64::INFINITY)), Value::Null);
        assert_eq!(finite(Some(f64::NAN)), Value::Null);
        assert_eq!(finite(None), Value::Null);
        assert_eq!(finite(Some(1.5)), json!(1.5));
    }
}
