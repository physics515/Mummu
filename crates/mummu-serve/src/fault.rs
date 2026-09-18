//! Fault injection: make the generation path fail exactly the way the
//! 2026-09-18 incident did, on a machine with no GPU at all.
//!
//! Two faults, one per signature of the incident, each armed with a count and
//! consumed one per use:
//!
//! * **`load_oom`** — at the end of the next load, cubecl's device-thread
//!   panic, raised the way cubecl raises it and swallowed the way cubecl
//!   swallows it: on a thread named `DSD-0-0`, inside `catch_unwind`, never
//!   reaching the loader. The loader returns `Ok`, as it did in production,
//!   and only the recovery's load check can tell (see `recovery::load_fault`).
//! * **`read_invalid`** — at the next forward, the panic production's chats
//!   hit on their first read: `bytes: host access failed: Read("The server is
//!   in an invalid state …")`, on the request's own runtime worker, with the
//!   model slot held. This is the path where the load check did not catch the
//!   failure and the poisoned model reached a request. Two of them in a row
//!   is a failure a reload does not cure, which is what escalates to a
//!   restart.
//!
//! Armed by `POST /api/fault {"load_oom": n, "read_invalid": n}`, which
//! answers with what is armed.
//!
//! # Why this cannot fire in production
//!
//! * It is not compiled. This module, both injection points in the engine and
//!   the `/api/fault` route are all `#[cfg(feature = "fault-injection")]`. The
//!   feature is off by default, and the image builds `cargo build --release
//!   -p mummu-serve --features cuda` — a test pins the Dockerfile's feature
//!   list (`the_docker_image_never_builds_fault_injection`). Without the
//!   feature, `POST /api/fault` is an unknown path and answers 404.
//! * Even a build that has it injects nothing until someone arms it: both
//!   counters start at zero and nothing but that endpoint raises them.

use std::sync::atomic::{AtomicU32, Ordering::SeqCst};

use axum::body::Bytes;
use axum::response::Response;
use serde::Deserialize;
use serde_json::json;

use crate::{json_response, parse_json};

/// The device-thread panic from production's log, verbatim.
pub const LOAD_OOM_MESSAGE: &str = "failed to reserve 22020096 bytes of device memory: out of \
                                    device memory allocating 261319680 bytes";

/// The read panic from production's log, verbatim up to where it was cut.
pub const INVALID_READ_MESSAGE: &str = "bytes: host access failed: Read(\"The server is in an \
     invalid state\\nCaused by:\\n  An IO error happened\\nCaused by:\\n  couldn't find resource \
     for that handle: Memory location was never initialized\\nNo backtrace available\")";

static LOAD_OOM: AtomicU32 = AtomicU32::new(0);
static READ_INVALID: AtomicU32 = AtomicU32::new(0);

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

/// Arm the faults: the next `load_oom` loads and `read_invalid` forwards fail.
pub fn arm(load_oom: u32, read_invalid: u32) {
    LOAD_OOM.store(load_oom, SeqCst);
    READ_INVALID.store(read_invalid, SeqCst);
}

/// What is armed: (`load_oom`, `read_invalid`).
#[must_use]
pub fn armed() -> (u32, u32) {
    (LOAD_OOM.load(SeqCst), READ_INVALID.load(SeqCst))
}

/// Called inside the load, after the loader returned: raise cubecl's
/// device-thread OOM exactly as cubecl does — on its runner thread's name,
/// caught right there, so the loader never hears of it.
pub(crate) fn during_load() {
    if !take(&LOAD_OOM) {
        return;
    }
    eprintln!(
        "[mummu-serve] fault injection: raising cubecl's device-thread OOM panic on DSD-0-0, as \
         during the 2026-09-18 load"
    );
    let spawned = std::thread::Builder::new()
        .name("DSD-0-0".to_owned())
        .spawn(|| {
            // cubecl-common device/handle/channel.rs:713-714 does exactly this.
            let _ = std::panic::catch_unwind(|| panic!("{LOAD_OOM_MESSAGE}"));
        });
    if let Ok(handle) = spawned {
        let _ = handle.join();
    }
}

/// Called before the first forward: the read that finds the device server in
/// an invalid state, as every chat after the incident's load did.
pub(crate) fn before_first_read() {
    if take(&READ_INVALID) {
        eprintln!(
            "[mummu-serve] fault injection: failing the first read the way the 2026-09-18 chats \
             did"
        );
        panic!("{INVALID_READ_MESSAGE}");
    }
}

#[derive(Deserialize, Default)]
struct Arm {
    #[serde(default)]
    load_oom: u32,
    #[serde(default)]
    read_invalid: u32,
}

/// `POST /api/fault {"load_oom": n, "read_invalid": n}` — replaces what is
/// armed and answers with it.
pub(crate) async fn endpoint(body: Bytes) -> Response {
    let a: Arm = match parse_json(&body) {
        Ok(a) => a,
        Err(response) => return *response,
    };
    arm(a.load_oom, a.read_invalid);
    eprintln!(
        "[mummu-serve] fault injection: armed load_oom = {}, read_invalid = {}",
        a.load_oom, a.read_invalid
    );
    let (load_oom, read_invalid) = armed();
    json_response(
        200,
        json!({"armed": {"load_oom": load_oom, "read_invalid": read_invalid}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each armed fault fires once per count and then never again — a demo
    /// that arms one failure must get exactly one.
    #[test]
    fn armed_faults_are_consumed_one_per_use() {
        let _serial = crate::progress_serial();
        crate::recovery::install_panic_hook();
        arm(1, 2);
        assert_eq!(armed(), (1, 2));
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
            let payload = std::panic::catch_unwind(before_first_read).expect_err("armed");
            let message = crate::recovery::payload_text(&*payload);
            assert!(
                crate::recovery::is_device_failure(None, &message),
                "{message}"
            );
        }
        std::panic::catch_unwind(before_first_read).expect("disarmed after two");
        assert_eq!(armed(), (0, 0));
        crate::recovery::reset_for_tests();
    }
}
