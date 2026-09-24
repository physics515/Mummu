//! Fault injection: make the generation path fail exactly the way the
//! 2026-09-18 incident did — and the ways around it the review asked about —
//! on a machine with no GPU at all.
//!
//! Each fault is armed with a count and consumed one per use:
//!
//! * **`load_oom`** — at the end of the next load, cubecl's device-thread
//!   panic, raised the way cubecl raises it and swallowed the way cubecl
//!   swallows it: on a thread named `DSD-0-0` (CUDA device 0), inside
//!   `catch_unwind`, never reaching the loader. The loader returns `Ok`, as
//!   it did in production, and only the recovery's load check can tell (see
//!   `recovery::load_fault`). Charged to the device the thread names.
//! * **`read_invalid`** — at the next forward, the panic production's chats
//!   hit on their first read: `bytes: host access failed: Read("The server is
//!   in an invalid state …")`, on the request's own runtime worker, with the
//!   model slot held. Charged to the model's own devices.
//! * **`read_invalid_err`** — the same failure arriving as an `Err`, not a
//!   panic: what an EAGER readback reports (`argmax readback: …`, exactly as
//!   `mummu::decode::argmax_id` words it). The path whose failures used not
//!   to move the fault epoch.
//! * **`kernel_gap`** — the panic mummu handles on purpose: the device
//!   thread panics on a cubecl kernel-expansion assert, the reading thread
//!   re-raises it, and the reader catches it once and retries — the contract
//!   of `mummu::nn::moe`'s `run_readback_with_fallback` (moe.rs:1012). The
//!   generation then completes. It must not poison anything.
//! * **`kernel_gap_oom`** — the same window, but what the device thread
//!   raised is an out-of-memory. Caught and retried just the same — and it
//!   must still poison the device.
//! * **`stall_ms`** — the next generation sleeps this long before its first
//!   read, holding the slot: how a test queues a request behind another.
//! * **`device_oom_now`** — raise cubecl's OOM on `DSD-0-0` right now, while
//!   nothing is running: a device failure no request caught, which only the
//!   resident model's fault stamp can act on.
//! * **`exit_stall`** — the next self-restart's normal exit hangs, the way
//!   `atexit` teardown with cubecl's device threads alive can: only the exit
//!   watchdog ends the process.
//!
//! Armed by `POST /api/fault {…}`, which REPLACES what is armed and answers
//! with the state; `GET /api/fault` answers with it and changes nothing. The
//! state includes the fault epoch, every device's books and how many loads
//! ran — what a test or a live run reads to see what recovery did.
//!
//! # Why this cannot fire in production
//!
//! * It is not compiled. This module, every injection point in the engine
//!   and in `recovery`, and the `/api/fault` route are all
//!   `#[cfg(feature = "fault-injection")]`. The feature is off by default, and
//!   the image builds `cargo build --release -p mummu-serve --features cuda`
//!   — a test pins the Dockerfile's feature list
//!   (`the_docker_image_never_builds_fault_injection`). Without the feature,
//!   `POST /api/fault` is an unknown path and answers 404.
//! * Even a build that has it injects nothing until someone arms it: every
//!   counter starts at zero and nothing but that endpoint raises them.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst};

use axum::body::Bytes;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::engine::BackendChoice;
use crate::{json_response, parse_json};

/// The device-thread panic from production's log, verbatim.
pub const LOAD_OOM_MESSAGE: &str = "failed to reserve 22020096 bytes of device memory: out of \
                                    device memory allocating 261319680 bytes";

/// The read panic from production's log, verbatim up to where it was cut.
pub const INVALID_READ_MESSAGE: &str = "bytes: host access failed: Read(\"The server is in an \
     invalid state\\nCaused by:\\n  An IO error happened\\nCaused by:\\n  couldn't find resource \
     for that handle: Memory location was never initialized\\nNo backtrace available\")";

/// The same failure as an eager readback reports it: `ServerUnhealthy`,
/// `Debug`-formatted behind `argmax readback:` by `mummu::decode::argmax_id`.
pub const INVALID_READ_ERR: &str = "argmax readback: ServerUnhealthy { errors: [\"The server is \
     in an invalid state\\nCaused by:\\n  couldn't find resource for that handle: Memory \
     location was never initialized\"] }";

/// cubecl's kernel-expansion assert — the width-dependent gap mummu catches
/// and retries around (`cubecl-std quant/view.rs:340`). No device-failure
/// signature in it, by design of the rule it tests.
pub const KERNEL_GAP_MESSAGE: &str =
    "quantized view float vector size 4 must be a positive multiple of num_quants 8";

static LOAD_OOM: AtomicU32 = AtomicU32::new(0);
static READ_INVALID: AtomicU32 = AtomicU32::new(0);
static READ_INVALID_ERR: AtomicU32 = AtomicU32::new(0);
static KERNEL_GAP: AtomicU32 = AtomicU32::new(0);
static KERNEL_GAP_OOM: AtomicU32 = AtomicU32::new(0);
static STALL_MS: AtomicU64 = AtomicU64::new(0);
static EXIT_STALL: AtomicBool = AtomicBool::new(false);

/// Loads that ran (the slot's load closure reached the loader).
static LOADS: AtomicU64 = AtomicU64::new(0);

/// What the most recent load was planned as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadInfo {
    pub backend: BackendChoice,
    pub policy: mummu::quant::QuantPolicy,
    /// The plan the load RAN under came from the "already resident"
    /// shortcut — which is only ever right for a hit, never for a load.
    pub from_resident: bool,
    /// The slot held this model and it was found stale.
    pub found_stale: bool,
}

static LAST_LOAD: Mutex<Option<LoadInfo>> = Mutex::new(None);

/// What is armed, as the endpoint takes it.
#[derive(Deserialize, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    #[serde(default)]
    pub load_oom: u32,
    #[serde(default)]
    pub read_invalid: u32,
    #[serde(default)]
    pub read_invalid_err: u32,
    #[serde(default)]
    pub kernel_gap: u32,
    #[serde(default)]
    pub kernel_gap_oom: u32,
    #[serde(default)]
    pub stall_ms: u64,
    #[serde(default)]
    pub exit_stall: bool,
    /// An action, not a count: raise the idle device OOM now.
    #[serde(default)]
    pub device_oom_now: bool,
}

/// Take one from `counter` if it has any.
fn take(counter: &AtomicU32) -> bool {
    let mut n = counter.load(SeqCst);
    while n > 0 {
        match counter.compare_exchange(n, n - 1, SeqCst, SeqCst) {
            Ok(_) => return true,
            Err(now) => n = now,
        }
    }
    false
}

/// Replace what is armed. `device_oom_now` fires here.
pub fn arm(a: Arm) {
    LOAD_OOM.store(a.load_oom, SeqCst);
    READ_INVALID.store(a.read_invalid, SeqCst);
    READ_INVALID_ERR.store(a.read_invalid_err, SeqCst);
    KERNEL_GAP.store(a.kernel_gap, SeqCst);
    KERNEL_GAP_OOM.store(a.kernel_gap_oom, SeqCst);
    STALL_MS.store(a.stall_ms, SeqCst);
    EXIT_STALL.store(a.exit_stall, SeqCst);
    if a.device_oom_now {
        eprintln!(
            "[mummu-serve] fault injection: raising cubecl's OOM on DSD-0-0 now, with no request \
             running"
        );
        raise_on_device_thread(LOAD_OOM_MESSAGE);
    }
}

/// What is armed now (`device_oom_now` is never "armed": it has fired).
#[must_use]
pub fn armed() -> Arm {
    Arm {
        load_oom: LOAD_OOM.load(SeqCst),
        read_invalid: READ_INVALID.load(SeqCst),
        read_invalid_err: READ_INVALID_ERR.load(SeqCst),
        kernel_gap: KERNEL_GAP.load(SeqCst),
        kernel_gap_oom: KERNEL_GAP_OOM.load(SeqCst),
        stall_ms: STALL_MS.load(SeqCst),
        exit_stall: EXIT_STALL.load(SeqCst),
        device_oom_now: false,
    }
}

/// Loads that ran so far.
#[must_use]
pub fn loads() -> u64 {
    LOADS.load(SeqCst)
}

/// The most recent load's plan.
#[must_use]
pub fn last_load() -> Option<LoadInfo> {
    *LAST_LOAD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Called by the slot's load closure as a load starts, with its plan.
pub fn loading(info: LoadInfo) {
    LOADS.fetch_add(1, SeqCst);
    *LAST_LOAD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(info);
}

/// A panic exactly as cubecl raises and swallows it: on a thread named like
/// CUDA device 0's runner, caught right there (cubecl-common
/// device/handle/channel.rs:713-714 does exactly this), never seen by the
/// caller.
fn raise_on_device_thread(message: &'static str) {
    let spawned = std::thread::Builder::new()
        .name("DSD-0-0".to_owned())
        .spawn(move || {
            let _ = std::panic::catch_unwind(|| panic!("{message}"));
        });
    if let Ok(handle) = spawned {
        let _ = handle.join();
    }
}

/// Called inside the load, after the loader returned: raise cubecl's
/// device-thread OOM the way cubecl does, so the loader never hears of it.
pub fn during_load() {
    if take(&LOAD_OOM) {
        eprintln!(
            "[mummu-serve] fault injection: raising cubecl's device-thread OOM panic on DSD-0-0, \
             as during the 2026-09-18 load"
        );
        raise_on_device_thread(LOAD_OOM_MESSAGE);
    }
}

/// A handled kernel-gap window, as `run_readback_with_fallback` sees one:
/// the device thread raises `device_message`, the reading thread re-raises
/// it, and the reader catches that ONCE and retries.
fn catch_and_retry_window(device_message: &'static str) {
    raise_on_device_thread(device_message);
    let reraised = std::panic::catch_unwind(|| panic!("{device_message}"));
    debug_assert!(reraised.is_err());
    eprintln!(
        "[mummu-serve] fault injection: a device panic was caught once and retried, as \
         mummu::nn::moe::run_readback_with_fallback does"
    );
}

/// Called before the first forward: the read that finds the device server in
/// an invalid state, as every chat after the incident's load did — or one of
/// the other ways around it the module header lists.
pub async fn before_first_read() -> Result<(), String> {
    let stall = STALL_MS.swap(0, SeqCst);
    if stall > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(stall)).await;
    }
    if take(&READ_INVALID) {
        eprintln!(
            "[mummu-serve] fault injection: failing the first read the way the 2026-09-18 chats \
             did"
        );
        panic!("{INVALID_READ_MESSAGE}");
    }
    if take(&READ_INVALID_ERR) {
        eprintln!("[mummu-serve] fault injection: failing the first read with an error");
        return Err(INVALID_READ_ERR.to_owned());
    }
    if take(&KERNEL_GAP) {
        catch_and_retry_window(KERNEL_GAP_MESSAGE);
    }
    if take(&KERNEL_GAP_OOM) {
        catch_and_retry_window(LOAD_OOM_MESSAGE);
    }
    Ok(())
}

/// Called by the exit right before it calls `exit`: hang there if armed.
pub fn before_exit() {
    if EXIT_STALL.swap(false, SeqCst) {
        eprintln!(
            "[mummu-serve] fault injection: the exit hangs here, as driver teardown can — only \
             the watchdog ends this process now"
        );
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }
}

/// Everything a test or a live run reads.
#[must_use]
pub fn report() -> Value {
    let a = armed();
    json!({
        "armed": {
            "load_oom": a.load_oom,
            "read_invalid": a.read_invalid,
            "read_invalid_err": a.read_invalid_err,
            "kernel_gap": a.kernel_gap,
            "kernel_gap_oom": a.kernel_gap_oom,
            "stall_ms": a.stall_ms,
            "exit_stall": a.exit_stall,
        },
        "fault_epoch": crate::recovery::fault_epoch(),
        "poisoned": crate::recovery::poisoned(),
        "restarting": crate::recovery::restarting(),
        "in_flight": crate::recovery::in_flight(),
        "loads": loads(),
        "last_load": last_load().map(|l| json!({
            "backend": format!("{:?}", l.backend),
            "policy": format!("{:?}", l.policy),
            "from_resident": l.from_resident,
            "found_stale": l.found_stale,
        })),
        "devices": crate::recovery::snapshot().iter().map(|d| json!({
            "device": d.label,
            "epoch": d.epoch,
            "consecutive": d.consecutive,
            "poisoned": d.poisoned,
        })).collect::<Vec<_>>(),
    })
}

/// `POST /api/fault {…}` — replaces what is armed and answers with the state.
pub async fn endpoint(body: Bytes) -> Response {
    let a: Arm = match parse_json(&body) {
        Ok(a) => a,
        Err(response) => return *response,
    };
    eprintln!("[mummu-serve] fault injection: armed {a:?}");
    arm(a);
    json_response(200, &report())
}

/// `GET /api/fault` — the state, unchanged.
pub async fn state() -> Response {
    json_response(200, &report())
}

#[cfg(test)]
mod engine_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// Each armed fault fires once per count and then never again — a demo
    /// that arms one failure must get exactly one.
    #[tokio::test]
    async fn armed_faults_are_consumed_one_per_use() {
        let _serial = crate::progress_serial().await;
        crate::recovery::reset_for_tests();
        crate::recovery::install_panic_hook();
        arm(Arm {
            load_oom: 1,
            read_invalid: 2,
            ..Arm::default()
        });
        assert_eq!((armed().load_oom, armed().read_invalid), (1, 2));
        let before = crate::recovery::fault_epoch();
        during_load();
        assert_eq!(
            crate::recovery::fault_epoch(),
            before + 1,
            "the injected load OOM is counted exactly as cubecl's would be"
        );
        during_load();
        assert_eq!(crate::recovery::fault_epoch(), before + 1, "and only once");
        for _ in 0..2 {
            let payload =
                futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(before_first_read()))
                    .await
                    .expect_err("armed");
            let message = crate::recovery::payload_text(&*payload);
            assert!(crate::recovery::is_device_failure(&message), "{message}");
        }
        assert!(before_first_read().await.is_ok(), "disarmed after two");
        assert_eq!(armed(), Arm::default());
        crate::recovery::reset_for_tests();
    }

    /// The endpoint refuses a field it does not know rather than arming
    /// nothing and answering 200 — a typo in a live run must not look like
    /// an armed fault.
    #[test]
    fn an_unknown_fault_is_refused() {
        assert!(serde_json::from_str::<Arm>(r#"{"load_om": 1}"#).is_err());
        assert!(serde_json::from_str::<Arm>(r#"{"load_oom": 1}"#).is_ok());
    }
}
