//! What the rest of the machine wants, sampled once a second.
//!
//! The scheduler's rule is to use only the capacity nobody else wants — and
//! all of it. That needs a reading of what everyone else is doing, per
//! resource, that does not count our own work as a co-tenant's:
//!
//! - **CPU**: `/proc/stat` is the whole host (a container sees the host's
//!   counters), `/proc/self/stat` is this process. Their difference is how
//!   many cores *other* processes kept busy. Exact, every second.
//! - **GPU compute**: NVML's utilization scaled by the SM clock it ran at
//!   ([`mummu::vram::Utilization::effective`] — the raw busy share reads
//!   15 % for an idle compositor at 210 MHz). It is the whole card, ours
//!   included, and there is no portable per-process split (inside a
//!   container NVML's process ids are not ours). So a GPU sample counts
//!   toward the co-tenant estimate only when this process had **no device
//!   work in flight** during it ([`DeviceWork`]); while we run, the last
//!   clean estimate stands. A game is a sustained load, so the gaps between
//!   our requests see it.
//! - **Memory**: `MemAvailable`, and pressure-stall (`/proc/pressure`) where
//!   the kernel has it.
//!
//! VRAM is not sampled here: placement already reads it (with the guard and
//! the ambient credit) under its own rules.
//!
//! The readings feed [`Contention`], a hysteresis state machine per
//! resource: a co-tenant that saturates the GPU or most of the CPU for a few
//! seconds puts us in *yield*, and only a sustained quiet stretch takes us
//! out — a game's loading screen should not bounce a 27B model on and off
//! the card.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use mummu_num::{f64_from_u64, f64_from_usize};
use serde::Serialize;

/// Sampling period.
pub const PERIOD: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Our own device work
// ---------------------------------------------------------------------------

static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static TOUCHED: AtomicBool = AtomicBool::new(false);

/// Held while this process has work on the accelerator.
///
/// A generation, a retrieval forward, a layer move — so the sampler can tell
/// our GPU use from a co-tenant's. Marks the period on entry AND on exit:
/// work that ended just before a sample still ran inside that sample's
/// window.
pub struct DeviceWork(());

impl DeviceWork {
    #[must_use]
    pub fn enter() -> Self {
        IN_FLIGHT.fetch_add(1, SeqCst);
        TOUCHED.store(true, SeqCst);
        Self(())
    }
}

impl Drop for DeviceWork {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, SeqCst);
        TOUCHED.store(true, SeqCst);
    }
}

/// Did any of our device work overlap the period since the last call?
fn ours_touched_the_period() -> bool {
    let touched = TOUCHED.swap(false, SeqCst);
    touched || IN_FLIGHT.load(SeqCst) > 0
}

// ---------------------------------------------------------------------------
// Raw readings
// ---------------------------------------------------------------------------

/// Cumulative CPU counters, in clock ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct CpuTicks {
    /// Every tick on every core.
    total: u64,
    /// Ticks spent idle or waiting on I/O.
    idle: u64,
    /// Ticks this process ran (user + system).
    ours: u64,
    /// Cores the host has (`cpuN` lines).
    cores: usize,
}

/// Parse `/proc/stat`'s aggregate line and core count.
fn parse_proc_stat(text: &str) -> Option<(u64, u64, usize)> {
    let mut total = None;
    let mut cores = 0usize;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("cpu ") {
            let v: Vec<u64> = rest
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            // user nice system idle iowait irq softirq steal [guest guest_nice]
            // — guest time is already inside user/nice, so it is not added.
            if v.len() < 5 {
                return None;
            }
            let sum: u64 = v.iter().take(8).sum();
            let idle = v[3] + v[4];
            total = Some((sum, idle));
        } else if line.starts_with("cpu") {
            cores += 1;
        }
    }
    let (sum, idle) = total?;
    Some((sum, idle, cores.max(1)))
}

/// Parse `/proc/self/stat`'s utime + stime (fields 14 and 15, counted after
/// the parenthesised command name, which may itself contain spaces).
fn parse_self_stat(text: &str) -> Option<u64> {
    let after = &text[text.rfind(')')? + 1..];
    let fields: Vec<&str> = after.split_whitespace().collect();
    // `after` starts at field 3 (state), so utime/stime are at 11 and 12.
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// `some avg10=` from a `/proc/pressure/*` file, in percent.
fn parse_psi_some(text: &str) -> Option<f64> {
    let line = text.lines().find(|l| l.starts_with("some "))?;
    line.split_whitespace()
        .find_map(|kv| kv.strip_prefix("avg10="))?
        .parse()
        .ok()
}

fn read_cpu() -> Option<CpuTicks> {
    let (total, idle, cores) = parse_proc_stat(&std::fs::read_to_string("/proc/stat").ok()?)?;
    let ours = parse_self_stat(&std::fs::read_to_string("/proc/self/stat").ok()?)?;
    Some(CpuTicks {
        total,
        idle,
        ours,
        cores,
    })
}

fn read_psi(which: &str) -> Option<f64> {
    parse_psi_some(&std::fs::read_to_string(format!("/proc/pressure/{which}")).ok()?)
}

// ---------------------------------------------------------------------------
// The derived picture
// ---------------------------------------------------------------------------

/// Cores busy over one sampling period, split into ours and everyone else's.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CpuShare {
    pub cores: usize,
    pub ours: f64,
    pub others: f64,
}

/// One period's CPU split, from two counter snapshots.
fn cpu_share(before: CpuTicks, after: CpuTicks) -> Option<CpuShare> {
    let dt = after.total.checked_sub(before.total)?;
    if dt == 0 {
        return None;
    }
    let idle = after.idle.saturating_sub(before.idle).min(dt);
    let ours = after.ours.saturating_sub(before.ours);
    let cores = f64_from_usize(after.cores);
    let per_core = f64_from_u64(dt) / cores;
    let busy_cores = f64_from_u64(dt - idle) / per_core;
    let our_cores = (f64_from_u64(ours) / per_core).min(busy_cores);
    Some(CpuShare {
        cores: after.cores,
        ours: our_cores,
        others: (busy_cores - our_cores).max(0.0),
    })
}

/// One resource's yield state, with hysteresis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Yield {
    /// Nobody else wants it: take all of it.
    Free,
    /// A co-tenant is saturating it: shrink to almost nothing.
    Yield,
}

/// Thresholds and dwell for one resource's [`Yield`] machine.
#[derive(Debug, Clone, Copy)]
pub struct Hysteresis {
    /// Others' share (0..1) at or above which a sample counts as busy.
    pub enter: f64,
    /// Others' share at or below which a sample counts as quiet.
    pub leave: f64,
    /// Consecutive busy samples to start yielding.
    pub enter_after: u32,
    /// Consecutive quiet samples to stop.
    pub leave_after: u32,
}

/// GPU compute (effective share, see [`mummu::vram::Utilization::effective`]).
///
/// A co-tenant using 60 % of the card's compute for three seconds is a game
/// or a training run, not a desktop compositor (~1 % here); a quarter-minute
/// below 25 % is it gone. Asymmetric on purpose — yielding late costs the user frames,
/// reclaiming late costs us only seconds of a slower decode.
pub const GPU_HYSTERESIS: Hysteresis = Hysteresis {
    enter: 0.60,
    leave: 0.25,
    enter_after: 3,
    leave_after: 15,
};

/// CPU: others holding three quarters of the cores.
///
/// Our threads already run in the idle scheduling class, so the kernel yields every cycle the
/// instant someone wants it; this state is for the placement decisions that
/// sit above that (do not move work TO the host while it is saturated).
pub const CPU_HYSTERESIS: Hysteresis = Hysteresis {
    enter: 0.75,
    leave: 0.40,
    enter_after: 3,
    leave_after: 15,
};

/// The machine for one resource.
#[derive(Debug, Clone, Copy)]
pub struct Contention {
    rule: Hysteresis,
    state: Yield,
    run: u32,
}

impl Contention {
    #[must_use]
    pub const fn new(rule: Hysteresis) -> Self {
        Self {
            rule,
            state: Yield::Free,
            run: 0,
        }
    }

    #[must_use]
    pub const fn state(&self) -> Yield {
        self.state
    }

    /// Feed one sample of others' share (0..1); returns the new state when
    /// it changed.
    pub fn feed(&mut self, others: f64) -> Option<Yield> {
        let (toward, counts, after) = match self.state {
            Yield::Free => (
                Yield::Yield,
                others >= self.rule.enter,
                self.rule.enter_after,
            ),
            Yield::Yield => (
                Yield::Free,
                others <= self.rule.leave,
                self.rule.leave_after,
            ),
        };
        if counts {
            self.run += 1;
        } else {
            self.run = 0;
        }
        if self.run >= after {
            self.state = toward;
            self.run = 0;
            return Some(toward);
        }
        None
    }
}

/// The scheduler's view of the machine, as of the last sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Pressure {
    /// CPU split over the last period.
    pub cpu: Option<CpuShare>,
    /// Whole-card effective GPU compute share, percent, over NVML's last
    /// period (busy share × clock / max clock).
    pub gpu_busy: Option<u32>,
    /// Others' effective GPU share, percent: the last sample taken while we
    /// had no device work in flight (`None` until one has been).
    pub gpu_others: Option<u32>,
    /// Seconds since `gpu_others` was measured.
    pub gpu_others_age_s: Option<u64>,
    pub mem_available: Option<u64>,
    /// Pressure-stall `some avg10`, percent.
    pub psi_cpu: Option<f64>,
    pub psi_memory: Option<f64>,
    pub gpu: Yield,
    pub cpu_state: Yield,
    /// Share of the host's compute free for us, 0..1, smoothed: the cores
    /// nobody else is using (ours count as free — they are what we would
    /// use). What the placement scales the host's measured rate by.
    pub cpu_free: f64,
    /// Share of the card's compute free for us, 0..1, smoothed (from the
    /// clean co-tenant estimate). Zero while yielding.
    pub gpu_free: f64,
}

impl Default for Pressure {
    fn default() -> Self {
        Self {
            cpu: None,
            gpu_busy: None,
            gpu_others: None,
            gpu_others_age_s: None,
            mem_available: None,
            psi_cpu: None,
            psi_memory: None,
            gpu: Yield::Free,
            cpu_state: Yield::Free,
            cpu_free: 1.0,
            gpu_free: 1.0,
        }
    }
}

/// Weight of the newest sample in the free-share averages (time constant
/// about three samples: a game's load moves it within seconds, a single
/// busy second barely does).
const FREE_EWMA: f64 = 0.3;

fn ewma(old: f64, new: f64) -> f64 {
    // Two statements, not `mul_add`: the fused form rounds differently and
    // is a libm call without hardware FMA (the workspace's rule).
    let kept = old * (1.0 - FREE_EWMA);
    let added = new * FREE_EWMA;
    kept + added
}

struct State {
    last_cpu: Option<CpuTicks>,
    gpu_others: Option<(u32, Instant)>,
    gpu: Contention,
    cpu: Contention,
    now: Pressure,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// A 0..1 share as a whole percent.
fn percent(share: f64) -> u32 {
    mummu_num::trunc_u64((share.clamp(0.0, 1.0) * 100.0).round())
        .try_into()
        .unwrap_or(100)
}

/// The latest picture. All-free (and empty) until the sampler has run.
#[must_use]
pub fn pressure() -> Pressure {
    STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map_or_else(Pressure::default, |s| s.now)
}

/// One sampling step: read everything, update the machines, log a change.
fn step(state: &mut State) {
    let cpu_now = read_cpu();
    let share = match (state.last_cpu, cpu_now) {
        (Some(a), Some(b)) => cpu_share(a, b),
        _ => None,
    };
    state.last_cpu = cpu_now;

    let clean = !ours_touched_the_period();
    let util = mummu::vram::utilization().map(|u| percent(u.effective()));
    if clean && let Some(u) = util {
        state.gpu_others = Some((u, Instant::now()));
    }

    if let Some(c) = share
        && let Some(change) = state.cpu.feed(c.others / f64_from_usize(c.cores))
    {
        eprintln!(
            "[mummu-serve] pressure: other processes hold {:.1} of {} cores — host compute {}",
            c.others,
            c.cores,
            match change {
                Yield::Yield => "is theirs (ours runs only on idle cycles)",
                Yield::Free => "is free again",
            }
        );
    }
    if let Some((others, _)) = state.gpu_others
        && clean
        && let Some(change) = state.gpu.feed(f64::from(others) / 100.0)
    {
        eprintln!(
            "[mummu-serve] pressure: other processes keep the GPU {others}% busy — {}",
            match change {
                Yield::Yield => "yielding the card",
                Yield::Free => "the card is ours again",
            }
        );
    }

    let cpu_free = share.map_or(state.now.cpu_free, |c| {
        ewma(
            state.now.cpu_free,
            (1.0 - c.others / f64_from_usize(c.cores)).clamp(0.0, 1.0),
        )
    });
    let gpu_free = match (state.gpu.state(), state.gpu_others) {
        (Yield::Yield, _) => 0.0,
        (Yield::Free, Some((others, _))) => {
            ewma(state.now.gpu_free, 1.0 - f64::from(others) / 100.0)
        }
        (Yield::Free, None) => state.now.gpu_free,
    };
    state.now = Pressure {
        cpu: share,
        gpu_busy: util,
        gpu_others: state.gpu_others.map(|(g, _)| g),
        gpu_others_age_s: state.gpu_others.map(|(_, at)| at.elapsed().as_secs()),
        mem_available: crate::status::mem_available_bytes(),
        psi_cpu: read_psi("cpu"),
        psi_memory: read_psi("memory"),
        gpu: state.gpu.state(),
        cpu_state: state.cpu.state(),
        cpu_free,
        gpu_free,
    };
}

/// Start the sampler (once per process). A plain thread rather than a task:
/// an NVML call can stall for seconds on a busy driver, and that must cost
/// this loop a sample, not a runtime worker.
pub fn spawn() {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }
    *STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(State {
        last_cpu: read_cpu(),
        gpu_others: None,
        gpu: Contention::new(GPU_HYSTERESIS),
        cpu: Contention::new(CPU_HYSTERESIS),
        now: Pressure::default(),
    });
    let spawned = std::thread::Builder::new()
        .name("mummu-sysmon".into())
        .spawn(|| {
            loop {
                std::thread::sleep(PERIOD);
                let mut g = STATE
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(state) = g.as_mut() {
                    step(state);
                }
            }
        });
    if let Err(e) = spawned {
        eprintln!("[mummu-serve] pressure sampler did not start ({e}) — co-tenants are invisible");
    }
}

// ---------------------------------------------------------------------------
// Our compute threads yield to everyone
// ---------------------------------------------------------------------------

/// Put the calling thread in the scheduler's idle class: it runs only on
/// cycles no other thread on the machine wants, and is preempted the moment
/// one does.
///
/// For the compute herd (the rayon pool burn-flex and the gemm kernels run
/// on). The service threads — the HTTP runtime, cubecl's device servers and
/// poll thread — are left at normal priority, which is the inversion the
/// Windows build measured (fence latency 26.5 → 8.7 ms once services
/// preempted the herd), and now also the whole of "use only excess CPU":
/// the kernel enforces it at microsecond granularity, with no reading and no
/// tuning, and an idle machine still gives the herd every core.
///
/// Linux only; lowering a thread's own priority needs no privilege, so it
/// works in an unprivileged container. Elsewhere the Windows binary keeps
/// its own `BELOW_NORMAL` demotion.
#[cfg(target_os = "linux")]
#[must_use]
pub fn demote_current_thread() -> bool {
    let param = libc::sched_param { sched_priority: 0 };
    // SAFETY: pid 0 is the calling thread on Linux; `param` is a valid
    // pointer to a stack local for the duration of the call.
    unsafe { libc::sched_setscheduler(0, libc::SCHED_IDLE, &raw const param) == 0 }
}

/// Windows: `BELOW_NORMAL`, the measured default (fence 26.5 → 8.7 ms).
/// There is no idle class a thread may enter by itself there that keeps
/// a busy desktop responsive and the herd alive both; this is the old
/// behaviour, unchanged.
#[cfg(windows)]
#[must_use]
pub fn demote_current_thread() -> bool {
    #[link(name = "kernel32.dll", kind = "raw-dylib", modifiers = "+verbatim")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(handle: isize, priority: i32) -> i32;
    }
    // SAFETY: plain kernel32 calls on the current thread's pseudo handle;
    // -1 = THREAD_PRIORITY_BELOW_NORMAL.
    unsafe { SetThreadPriority(GetCurrentThread(), -1) != 0 }
}

/// Neither Linux nor Windows: nothing to do.
#[cfg(not(any(target_os = "linux", windows)))]
#[must_use]
pub const fn demote_current_thread() -> bool {
    false
}

/// Build the global rayon pool with every worker demoted
/// ([`demote_current_thread`]) — call once, before anything computes.
///
/// Both front ends call it: the headless binary and the desktop app, which
/// before this ran its herd at normal priority because it never built the
/// pool itself. `MUMMU_GEMM_PRIORITY=normal` keeps the herd un-demoted (the
/// in-situ ANOVA's A/B axis, SPEC P1.1).
pub fn install_compute_pool() {
    let demote =
        !std::env::var("MUMMU_GEMM_PRIORITY").is_ok_and(|v| v.eq_ignore_ascii_case("normal"));
    let mut builder = rayon::ThreadPoolBuilder::new();
    if demote {
        builder = builder.start_handler(|_| {
            let _ = demote_current_thread();
        });
    }
    match builder.build_global() {
        Ok(()) if demote => eprintln!(
            "[mummu-serve] compute herd: {} threads, {}",
            rayon::current_num_threads(),
            if cfg!(target_os = "linux") {
                "idle scheduling class — they run only on cycles nothing else on the machine wants"
            } else {
                "below normal priority"
            }
        ),
        Ok(()) => {
            eprintln!("[mummu-serve] compute herd: normal priority (MUMMU_GEMM_PRIORITY=normal)");
        }
        Err(e) => eprintln!(
            "[mummu-serve] rayon pool was already initialized ({e}); the compute herd keeps its priority"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_stat_parses_the_aggregate_and_counts_cores() {
        let text = "cpu  100 5 50 800 45 0 0 0 0 0\ncpu0 50 2 25 400 20 0 0 0 0 0\ncpu1 50 3 25 400 25 0 0 0 0 0\nintr 1\n";
        assert_eq!(parse_proc_stat(text), Some((1000, 845, 2)));
        assert_eq!(parse_proc_stat("intr 1\n"), None);
    }

    #[test]
    fn self_stat_survives_a_command_name_with_spaces() {
        let text =
            "1234 (mummu serve) S 1 1234 1234 0 -1 4194560 100 0 0 0 70 30 0 0 20 0 8 0 1 2 3";
        assert_eq!(parse_self_stat(text), Some(100));
    }

    #[test]
    fn psi_reads_some_avg10() {
        let text = "some avg10=12.50 avg60=3.00 avg300=1.00 total=5\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n";
        assert_eq!(parse_psi_some(text), Some(12.5));
    }

    /// 4 cores for one period of 100 ticks each = 400 ticks; 100 idle →
    /// 3 busy cores, of which we ran 1.
    #[test]
    fn the_cpu_split_subtracts_our_own_cores() {
        let a = CpuTicks {
            total: 1000,
            idle: 500,
            ours: 10,
            cores: 4,
        };
        let b = CpuTicks {
            total: 1400,
            idle: 600,
            ours: 110,
            cores: 4,
        };
        let s = cpu_share(a, b).unwrap();
        assert!(
            (s.ours - 1.0).abs() < 1e-9 && (s.others - 2.0).abs() < 1e-9,
            "{s:?}"
        );
        assert_eq!(cpu_share(b, b), None, "no time passed");
    }

    #[test]
    fn contention_enters_fast_leaves_slow_and_ignores_blips() {
        let mut c = Contention::new(GPU_HYSTERESIS);
        assert_eq!(c.feed(0.9), None);
        assert_eq!(c.feed(0.1), None, "a blip resets the run");
        assert_eq!(c.feed(0.9), None);
        assert_eq!(c.feed(0.9), None);
        assert_eq!(c.feed(0.9), Some(Yield::Yield));
        // Leaving needs 15 quiet samples in a row; a middling one resets.
        for _ in 0..14 {
            assert_eq!(c.feed(0.1), None);
        }
        assert_eq!(c.feed(0.4), None);
        for _ in 0..14 {
            assert_eq!(c.feed(0.1), None);
        }
        assert_eq!(c.feed(0.1), Some(Yield::Free));
        assert_eq!(c.state(), Yield::Free);
    }

    #[test]
    fn device_work_marks_the_period_on_entry_and_exit() {
        // The engine tests that drive a chat hold the same globals; they
        // run under this lock.
        let _serial = crate::progress_serial_blocking();
        let _ = ours_touched_the_period();
        assert!(!ours_touched_the_period());
        let w = DeviceWork::enter();
        assert!(ours_touched_the_period(), "in flight");
        assert!(ours_touched_the_period(), "still in flight");
        drop(w);
        assert!(ours_touched_the_period(), "ended inside this period");
        assert!(!ours_touched_the_period(), "a quiet period after it");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_thread_can_demote_itself_without_privilege() {
        let ok = std::thread::spawn(|| {
            let ok = demote_current_thread();
            // SAFETY: plain query of the calling thread's policy.
            let policy = unsafe { libc::sched_getscheduler(0) };
            (ok, policy)
        })
        .join()
        .unwrap();
        assert_eq!(ok, (true, libc::SCHED_IDLE));
    }
}
