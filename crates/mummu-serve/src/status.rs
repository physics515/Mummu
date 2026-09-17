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

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// The crate version, which is the workspace version — bumped to 0.3.0 for
/// the release this ships in. It had sat at 0.1.0 through the v0.1.0, v0.1.1
/// and v0.2.0 tags, which is exactly why nobody could tell what was deployed.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The git short sha this binary was built from, or `unknown` where there was
/// no git to ask (see `build.rs` — that case must not fail a build).
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

static SAMPLE: Mutex<Option<(Instant, Memory)>> = Mutex::new(None);

/// The current memory reading, resampled at most once per [`SAMPLE_TTL`].
///
/// The lock is held across the sampling, deliberately: two clients arriving
/// together should produce ONE reading, not two racing ones, and the work
/// under it is two small file reads and two FFI calls — microseconds. The
/// alternative (sample outside, then store) is the shape that lets ten tabs
/// each take their own sample and then argue about which one wins.
#[must_use]
pub fn memory() -> Memory {
    let mut slot = SAMPLE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, sample)) = *slot
        && at.elapsed() < SAMPLE_TTL
    {
        return sample;
    }
    let fresh = Memory {
        host: host_memory(),
        vram: mummu::vram::memory(),
    };
    *slot = Some((Instant::now(), fresh));
    fresh
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
///   "done": 673, "total": 851, "bytes": 15527000000,
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
    /// browser tabs hammering `/proc` and NVML. Two calls inside one TTL must
    /// be the same reading, byte for byte.
    #[test]
    fn readings_inside_the_ttl_are_the_same_sample() {
        let a = memory();
        let b = memory();
        assert_eq!(a, b, "a second poll must not resample");
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
