//! A real panic, through the real file descriptor, into the real ring.
//!
//! Why this earns a test of its own: the failure that motivated the whole
//! module was a CUDA fault that appeared **only** as a panic on stderr while
//! the request hung and `/api/health` still answered `ok`. Every other line the
//! feed shows is an `eprintln!` we control. A panic is formatted and written by
//! the runtime, so the only thing that can carry it into the ring is the
//! fd-level tee in `logs::install` — and the only way to know the tee does is
//! to panic and look.
//!
//! Why it runs as a **subprocess**: libtest redirects Rust-level output through
//! `io::set_output_capture`, and that redirect is inherited by spawned threads,
//! so under a normal `cargo test` *nothing* — not `eprintln!`, not a panic, not
//! from any thread — reaches fd 2, and a tee sitting on fd 2 would see an empty
//! stream. `--nocapture` turns that redirect off. Rather than demand the whole
//! suite be run that way, the test re-runs this binary as a child with
//! `--nocapture` and checks the child's exit status; the child is the
//! `#[ignore]`d half below.

use std::process::Command;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use mummu_serve::logs::{self, Level, Query, Source};

/// Set by the parent on the child it spawns. Also what keeps each half from
/// doing the other's job if someone runs the suite with `--ignored`.
const CHILD: &str = "MUMMU_SERVE_PANIC_CAPTURE_CHILD";

/// Everything the ring holds, oldest first.
fn snapshot() -> Vec<(Level, Source, String)> {
    logs::query(&Query {
        since: 0,
        limit: logs::MAX_LIMIT,
        source: None,
    })
    .lines
    .into_iter()
    .map(|l| (l.level, l.source, l.text))
    .collect()
}

/// Wait for a line containing `want`. The tee is a separate thread reading a
/// pipe, so a line is not in the ring the instant the writer returns.
fn wait_for(want: &str) -> Option<(Level, Source, String)> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let found = snapshot().into_iter().find(|(_, _, t)| t.contains(want));
        if found.is_some() {
            return found;
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_panic_lands_in_the_ring() {
    if std::env::var_os(CHILD).is_some() {
        return; // we are the child; our job is `panic_capture_child`
    }
    let exe = std::env::current_exe().expect("this test binary's own path");
    let out = Command::new(exe)
        .args([
            "--exact",
            "panic_capture_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .expect("re-run this test binary as a child");
    assert!(
        out.status.success(),
        "the capture child failed ({})\n--- child stdout ---\n{}\n--- child stderr ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // The tee must write through as well as capture, or `docker logs` — the
    // record of last resort — loses the panic it used to be the only witness
    // to. The child's stderr is the original fd, downstream of the tee.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("mummu-serve-test-panic-9c21"),
        "the panic must still reach the original stderr:\n{stderr}"
    );
}

#[test]
#[ignore = "spawned by a_panic_lands_in_the_ring; needs --nocapture to see fd 2"]
fn panic_capture_child() {
    assert!(
        std::env::var_os(CHILD).is_some(),
        "run this through a_panic_lands_in_the_ring, not directly"
    );
    logs::install();

    // A marker through the ordinary print path first, so a missing panic below
    // cannot be blamed on the capture never having been installed at all.
    let marker = "mummu-serve-test-marker-7f3a";
    eprintln!("[test] {marker}");
    let (_, source, _) = wait_for(marker).expect("a plain eprintln! must reach the ring");
    assert_eq!(source, Source::Server);

    // The real thing. The barrier keeps the panic strictly after the marker,
    // so the ring's ordering means something.
    let gate = Arc::new(Barrier::new(2));
    let theirs = Arc::clone(&gate);
    let worker = std::thread::Builder::new()
        .name("DSD-0-0".into()) // the thread name the live CUDA fault carried
        .spawn(move || {
            theirs.wait();
            panic!("CUDA_ERROR_UNKNOWN (mummu-serve-test-panic-9c21)");
        })
        .expect("spawn the panicking worker");
    gate.wait();
    assert!(worker.join().is_err(), "the worker must actually panic");

    // The runtime prints a panic as SEVERAL lines — a header naming the thread
    // and the site, then the payload, then the backtrace note — so it arrives
    // as several entries, in that order. Waiting on the payload therefore also
    // guarantees the header is already in.
    let (payload_level, _, payload) =
        wait_for("mummu-serve-test-panic-9c21").expect("the panic payload must reach the ring");
    let lines = snapshot();
    let (level, source, text) = lines
        .iter()
        .find(|(_, _, t)| t.contains("panicked"))
        .expect("the panic's header line must reach the ring");

    assert_eq!(
        *source,
        Source::Server,
        "a panic is captured output, not a request"
    );
    assert_eq!(
        *level,
        Level::Error,
        "a panic that reads as routine info is the bug this module exists to fix: {text}"
    );
    assert!(
        text.contains("DSD-0-0"),
        "the thread name is how an operator tells which worker died: {text}"
    );
    assert_eq!(
        payload_level,
        Level::Error,
        "the message itself must read as an error too: {payload}"
    );
}
