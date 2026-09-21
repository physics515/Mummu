//! Recovering from a crashed GPU backend — by itself, and truthfully.
//!
//! # The incident
//!
//! Production, 2026-09-18, the first cold load after the v0.3.1 deploy.
//! deepseek-ocr, a co-tenant on the shared 16 GiB card, was mid-job and held
//! most of it. The planner of that release read free VRAM only when
//! `MUMMU_VRAM_LIVE_BUDGET` said so, so it placed 7.35 GiB there anyway (the
//! planner now always measures the card — see `placement`). cubecl's device thread
//! `DSD-0-0` panicked thirty times during the load (counted in `docker logs`,
//! 15:55:52 to 15:55:58 UTC) with `failed to reserve 22020096 bytes of device
//! memory: out of device memory allocating 261319680 bytes`; the load
//! "succeeded" (the residency check even said `RESIDENCY SUSPECT: planned
//! 7.35 GiB on GPU (cuda), card grew 0.00 GiB`), and from then on EVERY chat
//! panicked on its first read with `bytes: host access failed: Read("The
//! server is in an invalid state … couldn't find resource for that handle:
//! Memory locat…")` — cut there by the log; for a handle that was never
//! bound, cubecl's message is `Memory location was never initialized` (step
//! 3 below). The client got an HTTP 200 with an empty stream, the status
//! object said `ready`, `/api/health` said `ok`, and only `docker restart
//! mummu` cleared it — after which the same load, on a card the co-tenant had
//! let go of, came up clean with zero device-thread panics.
//!
//! # What "The server is in an invalid state" is, from the source
//!
//! cubecl 0.11.0-pre.3; paths are relative to each crate's `src/`.
//!
//! 1. **An upload is submitted, not awaited, and a failed allocation leaves
//!    its handle unbound.** The client hands the handle back at once
//!    (`cubecl-runtime client.rs:405`: `self.device.submit(move |server| {
//!    server.initialize_memory(memory, size, stream_id); .. })`). On the
//!    device thread, `initialize_memory` panics on a failed reservation
//!    *before* it binds that handle to memory (`cubecl-cuda
//!    compute/server.rs:244-245`: `panic!("failed to reserve {size} bytes of
//!    device memory: {err}")`, then `command.bind(reserved, memory)`).
//! 2. **Nobody is told.** Every device task runs under `catch_unwind` and a
//!    panic is only logged (`cubecl-common device/handle/channel.rs:713-714`:
//!    `if let Err(payload) = catch_unwind(AssertUnwindSafe(f)) { log::warn!(..)
//!    }`). The runner thread survives and the loader returns `Ok`. cubecl's
//!    own comment names what is left behind (`cubecl-cuda
//!    compute/command.rs:158-161`): "a never-initialized handle whose every
//!    downstream use fails".
//! 3. **Every use of that handle fails, and the failure is queued, not
//!    raised.** Resolving it for a kernel returns `Memory location was never
//!    initialized` (`cubecl-runtime memory_management/memory_manage.rs:979-982`),
//!    the kernel is not launched, and the error is pushed onto the stream
//!    (`cubecl-cuda compute/server.rs:282`: `stream.current().errors.push(err)`).
//! 4. **The next read reports the queue, and drains it.** A read is a
//!    flushing command: `let errors = self.flush_errors(stream_id); if
//!    !mode.ignore && !errors.is_empty() { return Err(ServerError::ServerUnhealthy
//!    { errors, .. })` (`server.rs:990-993`), and `flush_errors` is
//!    `core::mem::take(&mut stream.current().errors)` (`server.rs:1009`). That
//!    is the ONLY place cubecl-cuda constructs `ServerUnhealthy`, and a
//!    stream's health is nothing but `stream.errors.is_empty()`
//!    (`compute/stream.rs:204-205`). The read's bytes turn the error into the
//!    panic production saw (`cubecl-environment bytes/base.rs:446`:
//!    `.expect("bytes: host access failed")`).
//!
//! So the state lives in the device's ONE `CudaServer`, per stream, on the
//! runner thread `DSD-<type>-<index>` that every client of that device in the
//! process shares — per device, not per client — and it is **not sticky**:
//! nothing latches, and every read clears what it reports. What failed every
//! later chat was not the server but the RESIDENT MODEL: its tensors carry the
//! never-initialized handles and re-queue the same errors on every forward. A
//! `ManagedMemoryHandle` is two `Arc`s with no `Drop` of its own
//! (`cubecl-runtime memory_management/memory_pool/handle.rs`), so dropping
//! those tensors costs the device nothing; and a driver out-of-memory is not a
//! sticky CUDA error, so the context stays usable.
//!
//! Re-creating the client is neither possible in practice nor needed.
//! `ComputeClient::load` (`cubecl-runtime client.rs:189-190`) goes through
//! `DeviceHandle::new`, which hands back the device's cached entry
//! (`cubecl-common device/handle/channel.rs:367-374`: `return
//! Ok(existing.clone())`). The only road to a fresh `CudaServer` is
//! `DeviceHandle::shutdown` (`channel.rs:507`), which is device-wide, waits
//! until EVERY handle to the device is dropped — every tensor holds one, and
//! so do burn's own caches — and leaks the runner thread when that has not
//! happened within `SHUTDOWN_JOIN_TIMEOUT` (30 s, `channel.rs:309`). Dropping
//! the model is what removes the bad handles, and it is enough.
//!
//! One residue can outlive the eviction: errors are queued per STREAM, and a
//! stream is per thread unless configured otherwise (`cubecl-environment
//! stream/id.rs:82-97`), so an error queued on a thread nobody reads from
//! again waits there for that thread's next flushing command — which may be
//! the reload's — and is reported once. That can cost one reload, which the
//! escalation rule below absorbs.
//!
//! # The decision
//!
//! **Unload and reload, in-process, first** — exactly what was asked for. On
//! a confirmed device failure the poisoned model is dropped (the slot, the
//! tier runtime, the residency notes), the status says `error`, `/api/health`
//! answers 503, and the next request loads the model again, which works as
//! soon as the card has room.
//!
//! **A process restart only when that did not cure it.** Some failures cannot
//! be cured from inside the process. The CUDA context is the device's primary
//! context, retained once and never reset (`cubecl-cuda runtime.rs:94`:
//! `primary_ctx::retain(device_ptr)`), and the client that owns it is cached in
//! a process-wide registry for the life of the process (`cubecl-common
//! device/handle/channel.rs:278-284`, `RUNNERS` / `CHANNELS`). A sticky CUDA
//! error (an illegal address, a launch failure) poisons that context until the
//! process ends, and no reload can fix it; nor can a reload hand back pool
//! pages cubecl still holds on streams other than the one that reclaims. So
//! when a SECOND device failure follows the first ON THE SAME DEVICE with no
//! generation in between that produced a token THERE — the reload was tried
//! and it did not hold — a binary running under a supervisor ([`supervised`])
//! delivers the error it owes every client, writes the tail of its log where
//! the next process will find it, and exits with [`EXIT_RESTART`] so Docker's
//! `restart: unless-stopped` starts a clean process.
//!
//! # Every book is kept per device
//!
//! The poison (what `/api/health` and the status object report), what clears
//! it, and the count that escalates to a restart all belong to the DEVICE that
//! failed ([`DeviceKey`]). A process can run one model on the card and the
//! next on the host: a clean load of a CPU model proves nothing about the
//! card (`Device::sync()` on burn-flex is a no-op), and a token computed on
//! the host does not show that a sticky CUDA fault went away. Before this was
//! per device, a GPU fault interleaved with CPU traffic cleared its own poison
//! and reset its own count on every CPU request, and never reached the
//! restart it needed.
//!
//! Only the fault EPOCH is global — the one number every resident model is
//! stamped with. See "Trade-offs" below.
//!
//! # Every device-failure path is handled the same way, under the slot lock
//!
//! A failure reaches us three ways: a panic on cubecl's device thread (the
//! hook sees it), a panic on the request's thread (the incident's read), or an
//! `Err` carrying cubecl's text (an eager readback: `argmax readback: …
//! ServerUnhealthy`, `mummu::decode::argmax_id`). All three move the global
//! epoch and the failed device's book, and all three are DECIDED — counted,
//! escalated, and a restart latched — by [`record_failure`], which the engine
//! calls while it still holds the model slot, then evicts the model through
//! the slot guard before releasing it. So a request queued on the slot finds
//! either an empty slot or a refusal: never the poisoned model, and never a
//! gap between "the slot is free" and "the process is exiting" in which it
//! could start a load on a device about to disappear.
//!
//! # Why that restart cannot loop
//!
//! * Models load lazily. A new process touches the GPU only when a request
//!   arrives, so a restart cannot trigger another restart by itself.
//! * An exit needs TWO consecutive device failures on one device that each
//!   cost a request — in practice two full load attempts — in one process.
//! * At most one such restart per [`RESTART_COOLDOWN`]. The history travels
//!   from process to process in the evidence file's header, so if the next
//!   load hits the same wall (a co-tenant still holding the card), the new
//!   process does NOT exit again: it stays up, says `error`, answers every
//!   request with the truth, and tries a fresh load per request until one
//!   fits. An operator's own restart (or a deploy) resets the budget.
//! * The history is written FIRST to [`local_dir`] — the container's own
//!   writable layer, fast, and kept across a `restart: unless-stopped` of the
//!   same container — and only then, best effort and with a bounded wait, to
//!   the models root, which is a bind mount of a spinning array that was
//!   failing the day this was written. The header (which carries the history)
//!   is the first line of the file, the file is capped by bytes on write, and
//!   the reader keeps the header of a file too big to replay — so no
//!   truncation of the log lines can lose the budget.
//! * The exit itself has a deadline ([`EXIT_DEADLINE`]): a watchdog armed at
//!   its start terminates the process however the exit got stuck, so the
//!   `restarting` refusal can never become a permanent state.
//!
//! # What is not a GPU failure
//!
//! A panic or an error is a device failure when its text carries one of
//! cubecl's device-failure signatures ([`DEVICE_SIGNATURES`]): an allocation
//! that failed, a server in an invalid state, a never-initialized handle, a
//! CUDA driver error. The thread it happened on only says WHICH device
//! (`DSD-`/`DSU-<type>-<index>`, named at `cubecl-common
//! device/handle/channel.rs:898-907`); it does not make a panic a failure.
//!
//! That rule is deliberate, and mummu's own code is why. mummu catches some
//! device panics on purpose and retries: `mummu::nn::moe::native_qmatmul_ok`
//! (moe.rs:572, "any panic means no"), `run_readback_with_fallback`
//! (moe.rs:1012, whose kernel-gap panic is documented as raised on the device
//! server and re-raised on the reading thread) and `run_with_native_fallback`
//! (moe.rs:1040). What they catch is a kernel-expansion assert — `quantized
//! view float vector size … must be a positive multiple of num_quants …`
//! (`cubecl-std quant/view.rs:340`) — after which "the device server
//! measurably survives it" (moe.rs:1033). Counting any device-thread panic
//! would mark a healthy server poisoned there: a 503, a forced reload of a
//! 27B, and a step toward a restart, for a panic mummu handled. Marking the
//! expected-panic windows instead cannot work cleanly: the panic lands on the
//! DEVICE thread, not the thread that would open the window, so the window
//! would have to be process-wide — and a process-wide window would also hide
//! a genuine OOM that happened inside it. The signature rule counts that OOM
//! and ignores the kernel gap, wherever each happens. (The fourth site the
//! review named, `qwen4exp/experts.rs:1685`, is inside a `#[test]` and never
//! runs in this process.)
//!
//! What the rule gives up: a device-thread panic whose text carries no
//! signature goes uncounted. That fails safe — such a failure still breaks the
//! first read, and the read's own `The server is in an invalid state` is a
//! signature — so it costs one failed request, where the thread rule cost a
//! healthy server its model. Anything else — an index out of bounds, an
//! `unreachable!` in a model — is reported to the client as an internal
//! error and unloads nothing. The evidence that the rule does not fire on a
//! working card is production's own log: after the 15:59 UTC restart, a cold
//! load of the same 27B and four chats printed zero panics of any kind.
//!
//! # Trade-offs, stated
//!
//! * **Any device failure evicts the resident model, even when its weights
//!   are intact.** A transient activation OOM after a verified load leaves
//!   the weights good, and dropping them costs a reload (minutes, for the
//!   27B). The failure's text cannot tell an activation from a weight, and a
//!   wrong guess the other way is the incident: a model whose handles were
//!   never bound, served forever. So the epoch is global and every resident
//!   model loaded before the latest device failure is reloaded — the
//!   conservative side, on purpose.
//! * **A reload can fail on a card that has room.** cubecl-cuda keeps a
//!   memory pool PER STREAM (`cubecl-cuda compute/stream.rs:141-157`:
//!   `create_stream` builds its own `MemoryManagement` for each stream), and a
//!   stream is per thread. Pages the failed load reserved on one thread's
//!   stream are not reusable by a reload running on another thread's stream,
//!   so the reload can OOM while `nvidia-smi` shows the card with room. That
//!   is exactly the case the self-restart is for: a second failure on the
//!   same device with no token between escalates, and a clean process starts
//!   with every pool empty.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use serde_json::{Value, json};

use crate::logs;

/// The exit code of a restart this module asked for: `EX_TEMPFAIL` from
/// `sysexits.h`, "a temporary failure; the user is invited to retry" — which
/// is exactly the promise. Docker's `restart: unless-stopped` restarts on any
/// exit; the code is there so `docker inspect` says which kind this was.
pub const EXIT_RESTART: i32 = 75;

/// No second self-restart inside this window. Thirty minutes is long enough
/// that a card which stays full (a co-tenant that holds its memory until it
/// has been idle for five minutes, and is not idle) costs one restart rather
/// than a stream of them, and short enough that a sticky fault on a card that
/// was fine for an afternoon still gets its clean process.
pub const RESTART_COOLDOWN: Duration = Duration::from_secs(30 * 60);

/// How long an exiting process waits for open chat responses to deliver the
/// error frames they owe. Every request queued behind the failure is refused
/// at once (see [`restarting`]), so this is normally milliseconds; it is a
/// ceiling for a client that is slow to read, not an expected wait.
const DRAIN_BUDGET: Duration = Duration::from_secs(10);

/// After the drain: time for the last frames to leave their sockets and for
/// the stderr tee to carry the exit line into the ring before the tail is
/// taken.
const FLUSH_GRACE: Duration = Duration::from_millis(300);

/// How long the exit waits for the models-root copy of the evidence. The
/// container-local copy is already on disk by then, so this wait buys only
/// the second copy; a disk that stalls past it (the failing array under
/// `/models`) costs nothing but that copy.
const EVIDENCE_COPY_WAIT: Duration = Duration::from_secs(2);

/// From the moment an exit starts to the moment the process is gone, however
/// the exit got stuck: the watchdog armed at its start terminates the process
/// at this deadline. It covers the drain, the grace, the evidence and the
/// exit itself (`atexit` handlers and driver teardown run with cubecl's
/// device threads still alive, and either can block) with room to spare —
/// 10 s + 0.3 s + 2 s of budgets against 20 s.
pub const EXIT_DEADLINE: Duration = Duration::from_secs(20);

/// Lines of log a dying process leaves for the next one. A cold 27B load
/// prints ~200, so this holds the load that failed and the requests that
/// failed with it.
pub const EVIDENCE_LINES: usize = 300;

/// The most bytes the writer puts in an evidence file, header included. Well
/// under [`EVIDENCE_MAX_BYTES`], so a file this server wrote is always one the
/// next process replays — [`EVIDENCE_LINES`] alone does not bound it: a line
/// is up to [`logs::MAX_LINE_BYTES`] and JSON escaping can grow a control
/// character to six bytes, so 300 lines can reach several megabytes. The
/// NEWEST lines that fit are kept.
pub const EVIDENCE_WRITE_BUDGET: usize = 512 << 10;

/// A previous-process file larger than this is not one we wrote in full
/// (see [`EVIDENCE_WRITE_BUDGET`]). Its lines are set aside unread, but its
/// header — the first line, which carries the restart history — is still
/// read, so the cooldown survives whatever happened to the rest.
pub const EVIDENCE_MAX_BYTES: u64 = 1 << 20;

/// The evidence directory under the models root (`/models`, bind-mounted,
/// already home to cubecl's autotune cache at `/models/.cubecl-cache`).
/// Hidden, so nothing that lists models sees it. The SECOND copy — see
/// [`local_dir`] for the first.
pub const EVIDENCE_DIR: &str = ".mummu-serve";

/// The file a dying process writes and the next one replays.
pub const EVIDENCE_FILE: &str = "previous-process.jsonl";

/// cubecl's device-failure signatures: text that only a failing device puts
/// in a panic message or an error. Each is quoted from the source it comes
/// from. A panic is a device failure exactly when its text carries one of
/// these — see the module header for why the thread it ran on is not enough.
pub const DEVICE_SIGNATURES: &[&str] = &[
    // cubecl-runtime server/base.rs:324, `ServerError::ServerUnhealthy`.
    "The server is in an invalid state",
    // cubecl-environment bytes/base.rs:446 — a device readback that failed.
    "bytes: host access failed",
    // cubecl-cuda compute/server.rs:244 and cubecl-wgpu compute/server.rs:318
    // — the device-thread panic itself.
    "bytes of device memory",
    // cubecl-runtime server/base.rs:904, `IoError::OutOfMemory` (Display)…
    "out of device memory",
    // …and the same variant as `Debug` prints it, which is how an
    // `.expect(..)` on the device thread spells it (cubecl-wgpu
    // compute/mem_manager.rs:133, "Must have enough memory for a uniform").
    "OutOfMemory",
    // cubecl-runtime memory_management/memory_manage.rs:982.
    "Memory location was never initialized",
    // cubecl-wgpu compute/stream.rs:334 — a readback whose buffer could not
    // be mapped: the device is gone or lost.
    "Failed to map buffer",
    // cudarc's driver errors (`CUDA_ERROR_ILLEGAL_ADDRESS`, a stream that
    // could not be created: `CUDA_ERROR_OUT_OF_MEMORY`, ...), which is how a
    // sticky context failure is spelled.
    "CUDA_ERROR_",
];

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Which device a failure belongs to — the unit every book is kept in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeviceKey {
    /// A cubecl device server, by the `(type, index)` its runner threads are
    /// named with — `DSD-0-0` is CUDA device 0. See `engine::device_key` for
    /// how each of mummu's backends maps onto one.
    Cubecl { type_id: u16, index: u16 },
    /// The host (burn-flex). It has no device server and no device threads.
    Host,
    /// A failure reported where no device was known — a generation that is
    /// not the engine's (the tests' fakes), or a signature on a thread that
    /// is not a device's. Any clean load clears it and any token resets its
    /// count, because nothing better is known about it.
    Unattributed,
}

impl DeviceKey {
    /// The device a cubecl runner thread serves, from its name
    /// (`DS{U|D}-{type}-{index}`, `cubecl-common device/handle/channel.rs:
    /// 898-907`). Matched exactly rather than by prefix, so a thread of ours
    /// that merely starts with the same letters is never taken for a device.
    #[must_use]
    pub fn of_thread(name: &str) -> Option<Self> {
        let rest = name
            .strip_prefix("DSD-")
            .or_else(|| name.strip_prefix("DSU-"))?;
        let mut parts = rest.split('-');
        let number = |p: Option<&str>| {
            p.filter(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|p| p.parse::<u16>().ok())
        };
        let type_id = number(parts.next())?;
        let index = number(parts.next())?;
        parts
            .next()
            .is_none()
            .then_some(Self::Cubecl { type_id, index })
    }

    /// Is this an accelerator (anything with a cubecl device server)?
    #[must_use]
    pub const fn is_accelerator(self) -> bool {
        matches!(self, Self::Cubecl { .. })
    }
}

/// Is `name` one of cubecl's device runner threads?
#[must_use]
pub fn is_device_server_thread(name: &str) -> bool {
    DeviceKey::of_thread(name).is_some()
}

/// Does this text — a panic message or an error — say a device failed?
/// Exactly when it carries one of [`DEVICE_SIGNATURES`].
#[must_use]
pub fn is_device_failure(message: &str) -> bool {
    DEVICE_SIGNATURES.iter().any(|s| message.contains(s))
}

/// One readable line out of a panic message: the escapes the `Debug`-formatted
/// payloads carry (`Read("…\nCaused by:\n …")`) folded, whitespace collapsed,
/// and the whole clipped — it is headed for a chat bubble and a status line.
#[must_use]
pub fn summarize(message: &str) -> String {
    const MAX: usize = 240;
    let folded = message.replace("\\n", " / ").replace('\n', " / ");
    let mut out = String::with_capacity(folded.len().min(MAX + 4));
    for word in folded.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    clip(&mut out, MAX);
    out
}

/// Cut `s` to at most `max` bytes at a char boundary, marking the cut.
fn clip(s: &mut String, max: usize) {
    if s.len() > max {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        s.truncate(end);
        s.push('…');
    }
}

/// The text of a panic payload (`&str` or `String`; anything else is named
/// as such rather than guessed at).
#[must_use]
pub fn payload_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with a non-string payload".to_owned())
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// What recovery is doing about a failure. Travels to the client in the error
/// frame and to the pages in the status object, which render it as it is —
/// a page that promised "the next request loads the model again" while the
/// process was exiting would be the lie this module exists to end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    /// The model was dropped; the next request loads it again.
    Reload,
    /// This process is exiting so its supervisor starts a clean one.
    Restart,
    /// The PREVIOUS process exited to restart the backend; this one is clean
    /// and has not loaded a model yet.
    Restarted,
    /// The reload did not hold, and the restart budget is spent until the
    /// cooldown ends: the model was dropped, and every request tries a fresh
    /// load in THIS process.
    CoolingDown,
    /// The reload did not hold, and nothing supervises this process (the
    /// desktop shell): every request tries a fresh load; restart it by hand.
    Unsupervised,
}

impl Recovery {
    /// Every variant, for the pages' test: each must render its own words.
    pub const ALL: [Self; 5] = [
        Self::Reload,
        Self::Restart,
        Self::Restarted,
        Self::CoolingDown,
        Self::Unsupervised,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reload => "reload",
            Self::Restart => "restart",
            Self::Restarted => "restarted",
            Self::CoolingDown => "cooldown",
            Self::Unsupervised => "unsupervised",
        }
    }
}

/// An unresolved GPU failure the status object and `/api/health` report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError {
    /// What a person needs to read: what failed, and what happens next.
    pub message: String,
    /// Unix ms of the latest event of this failure.
    pub at_ms: u64,
    pub recovery: Recovery,
    /// This is the previous process's failure, replayed so the page can show
    /// what happened. This process's own backend has not failed.
    pub previous_process: bool,
    /// Device-failure panics recorded since this failure began — the
    /// incident's load raised thirty.
    pub faults: u64,
    /// Which device, as the engine names it ("GPU (cuda)"). Empty when a
    /// previous process did not say.
    pub device: String,
}

impl BackendError {
    /// The wire shape, shared by the status object and `/api/health`.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "message": self.message,
            "at_ms": self.at_ms,
            "at": crate::shim::rfc3339(UNIX_EPOCH + Duration::from_millis(self.at_ms)),
            "recovery": self.recovery.as_str(),
            "previous_process": self.previous_process,
            "faults": self.faults,
            "device": self.device,
        })
    }
}

/// How long each stage of an exit may take.
#[derive(Debug, Clone, Copy)]
struct ExitTiming {
    drain: Duration,
    grace: Duration,
    copy_wait: Duration,
    deadline: Duration,
}

const PRODUCTION_TIMING: ExitTiming = ExitTiming {
    drain: DRAIN_BUDGET,
    grace: FLUSH_GRACE,
    copy_wait: EVIDENCE_COPY_WAIT,
    deadline: EXIT_DEADLINE,
};

/// Where a supervised process writes its evidence, and how it exits. The
/// exits are function pointers only so the tests can watch them being taken.
#[derive(Clone)]
struct Supervisor {
    /// `<models root>/.mummu-serve` — the second copy, best effort.
    models_dir: PathBuf,
    /// [`local_dir`] — the first copy.
    local: PathBuf,
    /// The normal exit (`std::process::exit`).
    exit: fn(i32),
    /// What the watchdog calls at the deadline ([`terminate_now`]).
    hard_exit: fn(i32),
    timing: ExitTiming,
}

/// Where the evidence goes FIRST: a private directory in the process's temp
/// dir. In the container that is its own writable layer (`/tmp`, not a
/// tmpfs mount, and the image runs no `USER`, so it is root's), which a
/// `restart: unless-stopped` restart keeps — Docker restarts the SAME
/// container — and a recreate (a deploy) discards, which is exactly when the
/// restart budget should reset. It is written before the models root because
/// it is fast and local: `/models` is a bind mount of a spinning array that
/// was failing the day this was written, and a write there can stall.
///
/// On a desktop the temp dir is shared, so the directory is created 0700 and
/// refused if it is a symlink, someone else's, or writable by others; and no
/// file in it is opened through a symlink (see [`evidence`]).
#[must_use]
pub fn local_dir() -> PathBuf {
    std::env::temp_dir().join("mummu-serve")
}

/// One device's books.
#[derive(Debug)]
struct Book {
    key: DeviceKey,
    /// Device failures recorded against this device, ever. A load or a
    /// request compares it with a [`mark`] to learn WHICH device failed.
    epoch: u64,
    /// Device failures that each cost a request since this device last
    /// produced a token. Two in a row means the reload did not cure it.
    consecutive: u32,
    /// The unresolved failure, until a load on this device comes up clean.
    error: Option<BackendError>,
    /// The last failure text the hook saw for this device.
    last_cause: Option<String>,
}

struct State {
    books: Vec<Book>,
    /// Display names for devices, as the engine registered them.
    labels: Vec<(DeviceKey, &'static str)>,
    /// The previous process's failure, shown for the record.
    previous: Option<BackendError>,
    /// The last device-failure text seen anywhere, for a load check whose
    /// failure no device thread claimed.
    last_cause: Option<String>,
    /// Unix ms of this process's and its predecessors' self-restarts, oldest
    /// first — the budget [`RESTART_COOLDOWN`] is checked against.
    restarts_ms: Vec<u64>,
    supervisor: Option<Supervisor>,
}

impl State {
    fn book(&mut self, key: DeviceKey) -> &mut Book {
        let at = match self.books.iter().position(|b| b.key == key) {
            Some(at) => at,
            None => {
                self.books.push(Book {
                    key,
                    epoch: 0,
                    consecutive: 0,
                    error: None,
                    last_cause: None,
                });
                self.books.len() - 1
            }
        };
        &mut self.books[at]
    }

    fn label(&self, key: DeviceKey) -> String {
        if let Some((_, label)) = self.labels.iter().find(|(k, _)| *k == key) {
            return (*label).to_owned();
        }
        match key {
            DeviceKey::Cubecl { type_id, index } => format!("cubecl device {type_id}-{index}"),
            DeviceKey::Host => "CPU (flex)".to_owned(),
            DeviceKey::Unattributed => "the GPU backend".to_owned(),
        }
    }

    fn labels_of(&self, keys: &[DeviceKey]) -> String {
        keys.iter()
            .map(|&k| self.label(k))
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

static STATE: Mutex<State> = Mutex::new(State {
    books: Vec::new(),
    labels: Vec::new(),
    previous: None,
    last_cause: None,
    restarts_ms: Vec::new(),
    supervisor: None,
});

/// A poisoned lock still holds a perfectly good record, and losing the record
/// of a GPU failure because some other thread panicked is the wrong response
/// to a module about panics.
fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Every device failure, on every path and every device. A model remembers
/// the value it was loaded under; any movement since means a device failed
/// after its load began, and it is never served again (see `engine::drive`).
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// Latched by the first decision to exit, at the moment it is decided and
/// under the model slot's lock; never cleared. What [`restarting`] reads.
static EXITING: AtomicBool = AtomicBool::new(false);

/// The exit's threads were started — "taken once".
static EXIT_STARTED: AtomicBool = AtomicBool::new(false);

/// Chat responses still open (see [`InFlight`]).
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Exit threads still running. Only the tests' exits return, and a test that
/// fails before its exit is taken must not leave that thread pushing a
/// restart onto the NEXT test's history — `reset_for_tests` waits for it.
#[cfg(test)]
static EXIT_THREADS: AtomicUsize = AtomicUsize::new(0);

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn rfc3339_ms(ms: u64) -> String {
    crate::shim::rfc3339(UNIX_EPOCH + Duration::from_millis(ms))
}

/// Give a device its display name ("GPU (cuda)"). The engine calls this for
/// every device it can hand out; idempotent.
pub fn register_device(key: DeviceKey, label: &'static str) {
    let mut st = state();
    if !st.labels.iter().any(|(k, _)| *k == key) {
        st.labels.push((key, label));
    }
}

/// The global device-failure count a resident model is compared against.
#[must_use]
pub fn fault_epoch() -> u64 {
    EPOCH.load(SeqCst)
}

/// The failures to report: this process's latest unresolved device failure,
/// else the previous process's, replayed for the record.
#[must_use]
pub fn current() -> Option<BackendError> {
    let st = state();
    st.books
        .iter()
        .filter_map(|b| b.error.as_ref())
        .max_by_key(|e| e.at_ms)
        .or(st.previous.as_ref())
        .cloned()
}

/// Has a device of THIS process failed, with no clean load on it since? The
/// one predicate behind both the status object's `error` phase and a non-2xx
/// `/api/health`, so the two cannot disagree.
#[must_use]
pub fn poisoned() -> bool {
    state().books.iter().any(|b| b.error.is_some())
}

/// One device's books, as the fault-injection endpoint and the tests read
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSnapshot {
    pub key: DeviceKey,
    pub label: String,
    pub epoch: u64,
    pub consecutive: u32,
    pub poisoned: bool,
}

/// Every device's books.
#[must_use]
pub fn snapshot() -> Vec<DeviceSnapshot> {
    let st = state();
    st.books
        .iter()
        .map(|b| DeviceSnapshot {
            key: b.key,
            label: st.label(b.key),
            epoch: b.epoch,
            consecutive: b.consecutive,
            poisoned: b.error.is_some(),
        })
        .collect()
}

/// Is this process on its way out to restart the backend? New chats are
/// refused with [`RESTARTING_MESSAGE`] rather than started on a device that is
/// about to disappear.
#[must_use]
pub fn restarting() -> bool {
    EXITING.load(SeqCst)
}

/// What a chat refused during a restart is told.
pub const RESTARTING_MESSAGE: &str = "mummu is restarting its GPU backend after a failure — \
     try again in a minute";

/// [`RESTARTING_MESSAGE`], as a function for call sites that format it.
#[must_use]
pub fn restarting_message() -> &'static str {
    RESTARTING_MESSAGE
}

/// Chat responses currently open — what an exit drains.
#[must_use]
pub fn in_flight() -> usize {
    IN_FLIGHT.load(SeqCst)
}

// ---------------------------------------------------------------------------
// Detection: the panic hook
// ---------------------------------------------------------------------------

/// Install the hook that notices device failures. Idempotent; the previous
/// hook (the runtime's printer, which is what carries the panic into
/// `docker logs` and the ring) still runs first.
///
/// # Why a hook and not only `catch_unwind`
///
/// The load-time failure never reaches us as a panic: cubecl catches it on
/// its device thread and logs it (see the module header, step 2). The hook is
/// the one thing that sees every panic on every thread, at the moment it
/// happens and before anything unwinds, so it is where a device failure is
/// counted. Catching is still how a failing REQUEST is turned into an error
/// frame and a decision (see [`record_failure`]); the hook only records.
pub fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            previous(info);
            let thread = std::thread::current();
            note_panic(thread.name(), &payload_text(info.payload()));
        }));
    });
}

/// Record one panic if it is a device failure. Runs inside the panic hook, so
/// it takes one lock, allocates a little, and cannot panic itself.
fn note_panic(thread: Option<&str>, message: &str) {
    if !is_device_failure(message) {
        // A kernel gap mummu catches and retries, or an ordinary bug: not
        // the device's failure, wherever it happened (module header).
        return;
    }
    let cause = summarize(message);
    let Some((name, device)) = thread.and_then(|t| DeviceKey::of_thread(t).map(|d| (t, d))) else {
        // cubecl's text on a thread that is not a device's — the request's
        // read, say. Whoever catches it knows the device and records it
        // there (the engine, under the slot lock); the epoch moves now, so
        // nothing loaded before it is served in between.
        let mut st = state();
        EPOCH.fetch_add(1, SeqCst);
        st.last_cause = Some(cause);
        return;
    };
    let first = {
        let mut st = state();
        EPOCH.fetch_add(1, SeqCst);
        st.last_cause = Some(cause.clone());
        let label = st.label(device);
        let book = st.book(device);
        book.epoch += 1;
        book.last_cause = Some(cause.clone());
        match book.error.as_mut() {
            Some(e) => {
                e.faults += 1;
                false
            }
            None => {
                book.error = Some(BackendError {
                    message: format!(
                        "{label} failed on its device thread {name}: {cause} — a model loaded \
                         before this is not used again; the next request loads it fresh"
                    ),
                    at_ms: now_ms(),
                    recovery: Recovery::Reload,
                    previous_process: false,
                    faults: 1,
                    device: label,
                });
                true
            }
        }
    };
    // Once per failure, not once per panic: the incident's load raised thirty,
    // and the runtime has already printed every one of them.
    if first {
        eprintln!(
            "[mummu-serve] recovery: the GPU backend failed on {name} ({cause}) — whatever the \
             device was given since is suspect; the model will be dropped and loaded again"
        );
    }
}

// ---------------------------------------------------------------------------
// Detection: the load and the request
// ---------------------------------------------------------------------------

/// The epochs at one moment: the global one and every device's. Taken before
/// a load or a generation, compared after, to learn whether a device failed
/// in between and which.
#[derive(Debug, Clone)]
pub struct EpochMark {
    global: u64,
    devices: Vec<(DeviceKey, u64)>,
}

/// Take an [`EpochMark`].
#[must_use]
pub fn mark() -> EpochMark {
    let st = state();
    EpochMark {
        // Read under the lock the hook bumps both under, so the two agree.
        global: EPOCH.load(SeqCst),
        devices: st.books.iter().map(|b| (b.key, b.epoch)).collect(),
    }
}

impl EpochMark {
    /// The global epoch at the mark — what a model loaded now is stamped with.
    #[must_use]
    pub const fn global(&self) -> u64 {
        self.global
    }

    /// Has any device failed since the mark?
    #[must_use]
    pub fn moved(&self) -> bool {
        fault_epoch() != self.global
    }

    /// The devices whose own books moved since the mark — the ones a device
    /// thread named.
    #[must_use]
    pub fn moved_devices(&self) -> Vec<DeviceKey> {
        let st = state();
        st.books
            .iter()
            .filter(|b| {
                let then = self
                    .devices
                    .iter()
                    .find(|(k, _)| *k == b.key)
                    .map_or(0, |(_, e)| *e);
                b.epoch > then
            })
            .map(|b| b.key)
            .collect()
    }
}

/// Which devices a failure seen since `mark` belongs to: the ones a device
/// thread named, if any did; otherwise the accelerators among `used` (only
/// an accelerator carries cubecl's text); otherwise whatever `used` names;
/// otherwise [`DeviceKey::Unattributed`].
#[must_use]
pub fn attribute(mark: &EpochMark, used: &[DeviceKey]) -> Vec<DeviceKey> {
    let moved = mark.moved_devices();
    if !moved.is_empty() {
        return moved;
    }
    let accelerators: Vec<DeviceKey> = used
        .iter()
        .copied()
        .filter(|d| d.is_accelerator())
        .collect();
    if !accelerators.is_empty() {
        accelerators
    } else if !used.is_empty() {
        used.to_vec()
    } else {
        vec![DeviceKey::Unattributed]
    }
}

/// What a load check found wrong: which devices, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadFault {
    pub devices: Vec<DeviceKey>,
    pub cause: String,
}

/// Did a device fail while a load ran? `mark` is from before the load;
/// `touched` the devices it placed anything on; `sync` waits for them to
/// finish everything the load submitted and names the device that refused.
///
/// The sync comes FIRST and is the reason this works: an upload is submitted,
/// not awaited (module header, step 1), so when the loader returns, a failed
/// allocation may still be sitting in the device queue. Draining the queue
/// makes its panic land — and move the epoch — before the epoch is compared.
/// A sync that itself reports an error is the same verdict by another route:
/// the device refused work this load gave it.
///
/// Any movement of the GLOBAL epoch fails the load, even a failure on a
/// device the load did not touch: nothing can tell a load that raced another
/// device's failure from one that caused it, and a false failure costs one
/// request where a false success is the incident. It is attributed to the
/// device that failed, though, so it poisons and counts against THAT device.
pub fn load_fault(
    mark: &EpochMark,
    touched: &[DeviceKey],
    sync: impl FnOnce() -> Result<(), (DeviceKey, String)>,
) -> Option<LoadFault> {
    let synced = sync();
    if mark.moved() {
        let devices = attribute(mark, touched);
        let cause = {
            let st = state();
            devices
                .iter()
                .find_map(|d| st.books.iter().find(|b| b.key == *d))
                .and_then(|b| b.last_cause.clone())
                .or_else(|| st.last_cause.clone())
                .unwrap_or_else(|| "a device-thread panic during the load".to_owned())
        };
        return Some(LoadFault { devices, cause });
    }
    synced.err().map(|(device, e)| LoadFault {
        devices: vec![device],
        cause: format!(
            "the device reported an error for this load's work: {}",
            summarize(&e)
        ),
    })
}

/// A load came up clean on `devices`: THEIR failures are resolved, as far as
/// the status object and `/api/health` are concerned — and only theirs. A
/// CPU load says nothing about the card. (The restart budget is not earned
/// back here: only a token does that — see [`generation_succeeded`].)
pub fn load_succeeded(model: &str, devices: &[DeviceKey]) {
    let mut lines = Vec::new();
    {
        let mut st = state();
        let proves = |k: DeviceKey| devices.contains(&k) || k == DeviceKey::Unattributed;
        let labels: Vec<String> = devices.iter().map(|&d| st.label(d)).collect();
        for book in st.books.iter_mut().filter(|b| proves(b.key)) {
            if let Some(e) = book.error.take() {
                lines.push(format!(
                    "{model} loaded cleanly on {} — clearing its failure from {} ({} device \
                     fault(s))",
                    e.device,
                    rfc3339_ms(e.at_ms),
                    e.faults
                ));
            }
        }
        let clears_previous = st
            .previous
            .as_ref()
            .is_some_and(|p| p.device.is_empty() || labels.contains(&p.device));
        if clears_previous {
            st.previous = None;
            lines.push(format!(
                "{model} loaded cleanly on {} — the previous process's failure is behind us",
                labels.join(" + ")
            ));
        }
    }
    for line in lines {
        eprintln!("[mummu-serve] recovery: {line}");
    }
}

/// A generation produced its first token on `devices`, so they compute and
/// read back: whatever failed on THEM before is behind us. A token on the
/// host resets nothing on the card.
pub fn generation_succeeded(devices: &[DeviceKey]) {
    let mut st = state();
    for book in st
        .books
        .iter_mut()
        .filter(|b| devices.contains(&b.key) || b.key == DeviceKey::Unattributed)
    {
        book.consecutive = 0;
    }
}

// ---------------------------------------------------------------------------
// The response: errors, decisions, eviction
// ---------------------------------------------------------------------------

/// Why a chat failed, and what recovery is doing about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatError {
    pub message: String,
    kind: Kind,
    recovery: Option<Recovery>,
    /// The failure was already counted, decided and acted on
    /// ([`record_failure`]); whoever sees it next only reports it.
    decided: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The request itself was bad, or a load failed for an ordinary reason.
    Request,
    /// A panic that is not a GPU failure.
    Internal,
    /// The GPU backend failed.
    Device,
    /// Refused: the process is exiting to restart the backend.
    Restarting,
}

impl ChatError {
    /// An ordinary failure.
    pub fn request(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: Kind::Request,
            recovery: None,
            decided: false,
        }
    }

    pub(crate) fn restarting() -> Self {
        Self {
            message: RESTARTING_MESSAGE.to_owned(),
            kind: Kind::Restarting,
            recovery: Some(Recovery::Restart),
            decided: true,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: Kind::Internal,
            recovery: None,
            decided: false,
        }
    }

    /// Was this the GPU backend's failure (or a refusal because of one)?
    #[must_use]
    pub fn is_device(&self) -> bool {
        matches!(self.kind, Kind::Device | Kind::Restarting)
    }

    /// Is this a device failure nobody has decided yet — a device error, or
    /// an ordinary one carrying cubecl's text (the eager readback)?
    #[must_use]
    pub fn needs_decision(&self) -> bool {
        !self.decided
            && (self.kind == Kind::Device
                || (self.kind == Kind::Request && is_device_failure(&self.message)))
    }

    /// What recovery is doing, when there is anything to do.
    #[must_use]
    pub const fn recovery(&self) -> Option<Recovery> {
        self.recovery
    }

    /// The status a non-streaming client gets. 503 for the GPU — the server
    /// is temporarily unable, and a retry is exactly the right response —
    /// 500 for everything else, as before.
    #[must_use]
    pub fn http_status(&self) -> u16 {
        if self.is_device() { 503 } else { 500 }
    }

    /// The native API's error frame (SSE and WebSocket alike): the frame type
    /// that already existed, plus what recovery is doing.
    #[must_use]
    pub fn frame(&self) -> Value {
        let mut frame = json!({"type": "error", "error": self.message});
        if let Some(r) = self.recovery {
            frame["recovery"] = json!(r.as_str());
        }
        frame
    }
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<String> for ChatError {
    fn from(message: String) -> Self {
        Self::request(message)
    }
}

impl From<&str> for ChatError {
    fn from(message: &str) -> Self {
        Self::request(message)
    }
}

/// Run a chat, turning every way it can fail into a [`ChatError`] — a panic
/// included.
///
/// This is the fix for the empty stream. A generation that panicked used to
/// take its channel sender down with it inside `tokio::spawn`, and the client
/// got a 200 with nothing in it. Every chat surface runs its generation
/// through here, so a panic comes back as an error like any other.
///
/// The engine decides the device failures it meets itself, under the slot
/// lock (see the module header), and hands back an error that says so; this
/// only reports those. What it decides here is a device failure that reached
/// it UNdecided — a panic or an error carrying cubecl's text from a
/// generation that is not the engine's — and it is decided the same way
/// ([`record_failure`]), plus a best-effort eviction of whatever is resident.
/// Any other panic is an internal error: it is reported, and nothing is
/// unloaded.
pub async fn contain<T>(
    model: &str,
    run: impl Future<Output = Result<T, ChatError>>,
) -> Result<T, ChatError> {
    let mark = mark();
    match AssertUnwindSafe(run).catch_unwind().await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) if e.needs_decision() => Err(undecided_failure(model, &mark, &e.message)),
        Ok(Err(e)) => Err(e),
        Err(payload) => {
            let message = payload_text(&*payload);
            if mark.moved() || is_device_failure(&message) {
                Err(undecided_failure(model, &mark, &message))
            } else {
                let cause = summarize(&message);
                eprintln!(
                    "[mummu-serve] chat {model}: the generation panicked ({cause}) — not a GPU \
                     failure, so nothing is unloaded"
                );
                Err(ChatError::internal(format!(
                    "the generation crashed: {cause}"
                )))
            }
        }
    }
}

/// A device failure that reached [`contain`] undecided: decide it, then
/// evict whatever the slot holds, if it can.
fn undecided_failure(model: &str, mark: &EpochMark, message: &str) -> ChatError {
    let devices = attribute(mark, &[]);
    let decided = record_failure(model, &devices, &summarize(message));
    crate::engine::evict_after_device_failure();
    decided
}

/// What a device failure that cost a request leads to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// First failure on this device (or the first since it produced a
    /// token): the model was dropped, the next request reloads it.
    Reload,
    /// The reload did not hold and a supervisor will restart us: exit.
    Restart,
    /// The reload did not hold, but this process (or its predecessor)
    /// restarted itself at `last_restart_ms`, inside [`RESTART_COOLDOWN`].
    ReloadCoolingDown { last_restart_ms: u64 },
    /// The reload did not hold, and nothing would restart us (the desktop
    /// shell runs these routers in-process).
    ReloadUnsupervised,
}

impl Decision {
    /// What the client and the pages are told recovery is doing.
    #[must_use]
    pub const fn recovery(self) -> Recovery {
        match self {
            Self::Reload => Recovery::Reload,
            Self::Restart => Recovery::Restart,
            Self::ReloadCoolingDown { .. } => Recovery::CoolingDown,
            Self::ReloadUnsupervised => Recovery::Unsupervised,
        }
    }
}

/// The escalation rule, as a pure function of what it depends on.
#[must_use]
pub fn decide(consecutive: u32, supervised: bool, restarts_ms: &[u64], now_ms: u64) -> Decision {
    if consecutive < 2 {
        return Decision::Reload;
    }
    if !supervised {
        return Decision::ReloadUnsupervised;
    }
    let cooldown = RESTART_COOLDOWN.as_millis() as u64;
    if let Some(&last) = restarts_ms
        .iter()
        .rev()
        .find(|&&t| now_ms.saturating_sub(t) < cooldown)
    {
        return Decision::ReloadCoolingDown {
            last_restart_ms: last,
        };
    }
    Decision::Restart
}

/// Who failed, as the client reads it: "the GPU backend failed on GPU
/// (cuda)"; just "the GPU backend failed" when no device is known; and never
/// "GPU" for a failure charged to the host alone.
fn who_failed(devices: &[DeviceKey], labels: &str) -> String {
    if devices.iter().all(|d| *d == DeviceKey::Unattributed) {
        "the GPU backend failed".to_owned()
    } else if devices.iter().any(|d| d.is_accelerator()) {
        format!("the GPU backend failed on {labels}")
    } else {
        format!("the backend failed on {labels}")
    }
}

/// The client-facing sentence for a decision.
///
/// Worded to be true whether the failure hit a resident model or a load that
/// never finished: "unloaded the model" told a client whose FIRST load failed
/// that something had been unloaded when nothing was ever resident.
fn decision_message(decision: Decision, who: &str, cause: &str) -> String {
    match decision {
        Decision::Reload => format!(
            "{who} ({cause}). mummu dropped everything it had placed on the device and will \
             load the model fresh on the next request — try again in a moment"
        ),
        Decision::Restart => format!(
            "{who} again after a reload ({cause}). mummu is restarting itself for a clean \
             start — try again in a minute"
        ),
        Decision::ReloadCoolingDown { last_restart_ms } => format!(
            "{who} again after a reload ({cause}). mummu already restarted itself at {} and will \
             not restart again before {}; it dropped what it had placed on the device and the \
             next request tries a fresh load",
            rfc3339_ms(last_restart_ms),
            rfc3339_ms(last_restart_ms + RESTART_COOLDOWN.as_millis() as u64),
        ),
        Decision::ReloadUnsupervised => format!(
            "{who} again after a reload ({cause}). Nothing supervises this process, so it will \
             not restart itself; it dropped what it had placed on the device and the next \
             request tries a fresh load — restart mummu if this keeps happening"
        ),
    }
}

/// Record a device failure that cost a request, DECIDE what happens, and
/// hand back the error the client is owed — already marked decided.
///
/// The engine calls this while it still holds the model slot, and evicts the
/// model through the slot guard right after (see the module header), so
/// everything this sets is in place before any queued request can take the
/// slot:
///
/// * the global epoch and each failed device's epoch move — on EVERY path,
///   an `Err` as much as a panic, so "a model loaded before the latest device
///   failure is never served again" holds for all of them;
/// * each failed device is poisoned and its count goes up — its own count,
///   so a failure on the card is never offset by traffic on the host;
/// * a decision to exit is latched ([`restarting`]) here, before this
///   returns — not after the slot is released — so a queued request cannot
///   slip into a load on a device that is about to go.
pub(crate) fn record_failure(model: &str, devices: &[DeviceKey], cause: &str) -> ChatError {
    let devices: Vec<DeviceKey> = if devices.is_empty() {
        vec![DeviceKey::Unattributed]
    } else {
        devices.to_vec()
    };
    let now = now_ms();
    let (decision, label, message) = {
        let mut st = state();
        EPOCH.fetch_add(1, SeqCst);
        let mut worst = 0;
        for &d in &devices {
            let book = st.book(d);
            book.epoch += 1;
            book.consecutive += 1;
            worst = worst.max(book.consecutive);
        }
        let decision = decide(worst, st.supervisor.is_some(), &st.restarts_ms, now);
        let label = st.labels_of(&devices);
        let message = decision_message(decision, &who_failed(&devices, &label), cause);
        for &d in &devices {
            let device = st.label(d);
            let book = st.book(d);
            let faults = book.error.as_ref().map_or(0, |e| e.faults);
            book.error = Some(BackendError {
                message: message.clone(),
                at_ms: now,
                recovery: decision.recovery(),
                previous_process: false,
                faults: faults.max(1),
                device,
            });
        }
        if decision == Decision::Restart {
            EXITING.store(true, SeqCst);
        }
        (decision, label, message)
    };
    eprintln!("[mummu-serve] recovery: chat {model}: {message}");
    if decision == Decision::Restart {
        begin_exit(
            format!(
                "the GPU backend failed twice in a row on {label}, the second time after a reload \
                 ({cause})"
            ),
            label,
        );
    }
    ChatError {
        message,
        kind: Kind::Device,
        recovery: Some(decision.recovery()),
        decided: true,
    }
}

// ---------------------------------------------------------------------------
// Draining and exiting
// ---------------------------------------------------------------------------

/// A chat response that is still open. Held by every chat surface for as long
/// as its response is being written, so an exiting process can wait for the
/// error frames it owes (see [`DRAIN_BUDGET`]).
pub struct InFlight(());

impl InFlight {
    #[must_use]
    pub fn enter() -> Self {
        IN_FLIGHT.fetch_add(1, SeqCst);
        Self(())
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, SeqCst);
    }
}

/// Start the exit, once: arm the watchdog, then run the exit on a thread of
/// its own. Two requests failing together, or a second failure while the
/// first exit is draining, cannot start a second exit or write the evidence
/// twice.
fn begin_exit(reason: String, device: String) {
    let Some(supervisor) = state().supervisor.clone() else {
        return; // `decide` never says Restart without one; belt and braces
    };
    if EXIT_STARTED.swap(true, SeqCst) {
        return;
    }
    #[cfg(test)]
    EXIT_THREADS.fetch_add(1, SeqCst);
    // FIRST, before anything that can block: whatever happens below, the
    // process is gone by the deadline, so `restarting` cannot become a
    // permanent state that needs a person to clear.
    arm_watchdog(supervisor.hard_exit, supervisor.timing.deadline);
    let spawned = std::thread::Builder::new()
        .name("mummu-restart".to_owned())
        .spawn({
            let supervisor = supervisor.clone();
            let reason = reason.clone();
            let device = device.clone();
            move || run_exit(&supervisor, &reason, &device, supervisor.timing)
        });
    if spawned.is_err() {
        // No thread to wait on: skip the drain rather than never exit.
        let rushed = ExitTiming {
            drain: Duration::ZERO,
            grace: Duration::ZERO,
            ..supervisor.timing
        };
        run_exit(&supervisor, &reason, &device, rushed);
    }
}

/// Terminate at `deadline`, whatever the exit is doing by then.
fn arm_watchdog(hard_exit: fn(i32), deadline: Duration) {
    let armed = std::thread::Builder::new()
        .name("mummu-exit-watchdog".to_owned())
        .spawn(move || {
            std::thread::sleep(deadline);
            eprintln!(
                "[mummu-serve] recovery: the exit did not finish within {deadline:?} — terminating \
                 now with code {EXIT_RESTART}"
            );
            hard_exit(EXIT_RESTART);
        });
    if let Err(e) = armed {
        eprintln!(
            "[mummu-serve] recovery: could not arm the exit watchdog ({e}); exiting without one"
        );
    }
}

/// What the watchdog calls: end the process NOW, with the restart's code.
///
/// `_exit`, not `exit`: it runs no `atexit` handlers and no destructors —
/// which is the point, because driver teardown with cubecl's device threads
/// still alive is one of the things that can hang an exit — and it ends every
/// thread at once. It keeps [`EXIT_RESTART`], so `docker inspect` still says
/// this was a restart. `abort` would end it too, but as SIGABRT (exit 134)
/// with a core dump of a process that may hold a 27B model's host memory,
/// written to a disk that may be the failing one; it is used only where
/// there is no `_exit` (non-unix, where this binary is not run supervised).
fn terminate_now(code: i32) {
    #[cfg(unix)]
    {
        // SAFETY: `_exit` has no preconditions; it does not return.
        unsafe { libc::_exit(code) }
    }
    #[cfg(not(unix))]
    {
        let _ = code;
        std::process::abort()
    }
}

/// The exit itself: say why, let open responses finish, leave the evidence —
/// the fast local copy first, the models-root copy with a bounded wait — and
/// go. Blocking, on a thread of its own, under the watchdog.
fn run_exit(supervisor: &Supervisor, reason: &str, device: &str, timing: ExitTiming) {
    eprintln!(
        "[mummu-serve] recovery: {reason} — exiting with code {EXIT_RESTART} so the supervisor \
         starts a clean process (Docker: restart: unless-stopped); the next request loads the \
         model again"
    );
    let deadline = Instant::now() + timing.drain;
    while IN_FLIGHT.load(SeqCst) > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let open = IN_FLIGHT.load(SeqCst);
    if open > 0 {
        eprintln!(
            "[mummu-serve] recovery: {open} chat response(s) still open after {:?}; exiting \
             anyway",
            timing.drain
        );
    }
    std::thread::sleep(timing.grace);
    let restarts_ms = {
        let mut st = state();
        st.restarts_ms.push(now_ms());
        st.restarts_ms.clone()
    };
    let header = evidence::Header {
        reason: reason.to_owned(),
        exit_code: EXIT_RESTART,
        exited_at_ms: now_ms(),
        version: crate::status::VERSION.to_owned(),
        build: crate::status::BUILD.to_owned(),
        restarts_ms,
        device: Some(device.to_owned()),
    };
    let rendered = evidence::render(&header, &logs::tail(EVIDENCE_LINES));
    // The copy the next process can count on, first.
    let local_written = match evidence::write_private(&supervisor.local, &rendered) {
        Ok(path) => {
            eprintln!(
                "[mummu-serve] recovery: the restart history and the last log lines are in {} for \
                 the next process",
                path.display()
            );
            true
        }
        Err(e) => {
            eprintln!(
                "[mummu-serve] recovery: could not write {} ({e}); trying the models root instead",
                supervisor.local.display()
            );
            false
        }
    };
    // The models root is a FALLBACK, never a second copy. The local copy is in
    // the container's writable layer, which Docker's restart of this same
    // container keeps — and a self-restart is always that. The models root is
    // `/mnt/deepmem/AI Models` in production, the btrfs array whose sda was
    // throwing SATA link resets on 2026-09-18, and the watchdog's `_exit`
    // cannot end a thread parked in uninterruptible I/O: a best-effort fsync
    // there could hold the whole exit for as long as the disk takes to time
    // out. So the exit only touches it when there is nothing else.
    if !local_written {
        // A stall there is not an error. Waited for, but not past the budget.
        let (tx, rx) = std::sync::mpsc::channel();
        let target = supervisor.models_dir.clone();
        let spawned = std::thread::Builder::new()
            .name("mummu-evidence-copy".to_owned())
            .spawn(move || {
                let _ = tx.send(evidence::write(&target, &rendered));
            });
        match spawned.map(|_| rx.recv_timeout(timing.copy_wait)) {
            Ok(Ok(Ok(path))) => eprintln!(
                "[mummu-serve] recovery: the fallback copy is in {}",
                path.display()
            ),
            Ok(Ok(Err(e))) => eprintln!(
                "[mummu-serve] recovery: no fallback copy in {} either ({e}) — the next process \
                 starts without the restart history",
                supervisor.models_dir.display()
            ),
            Ok(Err(std::sync::mpsc::RecvTimeoutError::Timeout)) => eprintln!(
                "[mummu-serve] recovery: the copy to {} did not finish within {:?} (a stalled \
                 disk?) — exiting without it",
                supervisor.models_dir.display(),
                timing.copy_wait
            ),
            Ok(Err(std::sync::mpsc::RecvTimeoutError::Disconnected)) => eprintln!(
                "[mummu-serve] recovery: the copy to {} failed without a result",
                supervisor.models_dir.display()
            ),
            Err(e) => eprintln!("[mummu-serve] recovery: no fallback copy (no thread: {e})"),
        }
    }
    #[cfg(feature = "fault-injection")]
    crate::fault::before_exit();
    (supervisor.exit)(EXIT_RESTART);
    // Reached only when the exit returned: a test's.
    #[cfg(test)]
    EXIT_THREADS.fetch_sub(1, SeqCst);
}

/// This process runs under a supervisor that restarts it when it exits — the
/// `mummu-serve` binary in its container, under `restart: unless-stopped`.
///
/// Called by the binary only, never by the library on its own: the desktop
/// shell runs the same routers in-process, and exiting there would close the
/// user's window. Without this call a failure that a reload does not cure is
/// still reported truthfully, and every request still retries a fresh load;
/// the process just never exits.
///
/// Also replays what the previous process left — in [`local_dir`] and in
/// `<root>/.mummu-serve/`, whichever is newer — if anything; see
/// [`evidence`]. Never fails: a missing, unreadable, corrupt or oversized file
/// costs a line in the log, not the startup.
pub fn supervised(models_root: &Path) {
    supervise_with(
        models_root,
        &local_dir(),
        |code| std::process::exit(code),
        terminate_now,
        PRODUCTION_TIMING,
    );
}

fn supervise_with(
    models_root: &Path,
    local: &Path,
    exit: fn(i32),
    hard_exit: fn(i32),
    timing: ExitTiming,
) {
    // As early as the binary can: a device failure before this is unseen.
    install_panic_hook();
    crate::engine::register_devices();
    let models_dir = models_root.join(EVIDENCE_DIR);
    // Both places, every start: whichever holds a file is set aside, so an
    // old one cannot be replayed at some later start as if it were news.
    let taken = evidence::newer(evidence::take_private(local), evidence::take(&models_dir));
    let replayed = taken.lines.len();
    {
        let mut st = state();
        st.supervisor = Some(Supervisor {
            models_dir: models_dir.clone(),
            local: local.to_path_buf(),
            exit,
            hard_exit,
            timing,
        });
        if let Some(header) = &taken.header {
            let now = now_ms();
            // Only what can still matter: the cooldown's window, bounded.
            let day = 24 * 60 * 60 * 1000;
            let mut recent: Vec<u64> = header
                .restarts_ms
                .iter()
                .copied()
                .filter(|&t| now.saturating_sub(t) < day)
                .collect();
            let excess = recent.len().saturating_sub(16);
            recent.drain(..excess);
            st.restarts_ms = recent;
            // Shown, not acted on: this process's backend is fresh. The page
            // says what happened until a load proves the device works again.
            st.previous = Some(BackendError {
                message: format!(
                    "the previous process exited at {} (code {}) to restart the GPU backend: {}. \
                     Nothing is loaded yet; the next request loads the model",
                    rfc3339_ms(header.exited_at_ms),
                    header.exit_code,
                    header.reason
                ),
                at_ms: header.exited_at_ms,
                recovery: Recovery::Restarted,
                previous_process: true,
                faults: 0,
                device: header.device.clone().unwrap_or_default(),
            });
        }
    }
    taken.replay();
    if let Some(note) = &taken.note {
        eprintln!("[mummu-serve] recovery: {note}");
    }
    if replayed > 0 {
        eprintln!(
            "[mummu-serve] recovery: replayed {replayed} line(s) the previous process left in {} \
             — see /logs",
            taken.from.as_deref().unwrap_or(&models_dir).display()
        );
    }
}

// ---------------------------------------------------------------------------
// Evidence across a restart
// ---------------------------------------------------------------------------

/// The previous process's last words.
///
/// The log ring lives in memory, so a restart would wipe the very lines that
/// explain it — from the page built to show them. Before a self-restart the
/// dying process writes the tail of its ring to [`super::local_dir`] (fast,
/// container-local, kept across a restart of the same container) and then,
/// best effort, to `<models root>/.mummu-serve/`; the next process looks in
/// both, replays the newer into its own ring, marked `[previous process]`,
/// and rotates each file it found to `.1` so nothing is replayed twice.
///
/// JSON Lines: a header object — which carries the restart history — then one
/// object per log line, capped at [`EVIDENCE_WRITE_BUDGET`] bytes by keeping
/// the newest lines that fit. Written to a temporary name and renamed into
/// place, so a crash mid-write leaves the previous file or none, never a torn
/// one. No file is opened through a symlink, and nothing but a regular file
/// is read.
///
/// **A normal shutdown writes nothing** (ctrl-c, `docker stop`, a deploy): the
/// file exists only between a self-restart and the next process's startup,
/// and there is nothing to explain after a shutdown someone asked for.
pub mod evidence {
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize};

    use super::{EVIDENCE_FILE, EVIDENCE_LINES, EVIDENCE_MAX_BYTES, EVIDENCE_WRITE_BUDGET};
    use crate::logs::{self, Level, LogLine, Source};

    /// Why the previous process exited, and the restart budget it spent.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Header {
        pub reason: String,
        pub exit_code: i32,
        pub exited_at_ms: u64,
        pub version: String,
        pub build: String,
        /// Self-restarts so far, oldest first, this one included.
        pub restarts_ms: Vec<u64>,
        /// The device that failed, as the engine names it.
        #[serde(default)]
        pub device: Option<String>,
    }

    /// The header on the wire: a marker that says what this file is, so a
    /// stray JSON object is not mistaken for one.
    #[derive(Serialize, Deserialize)]
    struct HeaderLine {
        mummu_previous_process: u32,
        #[serde(flatten)]
        header: Header,
    }

    /// One log line on the wire.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Line {
        pub ts: u64,
        pub source: String,
        pub level: String,
        pub text: String,
    }

    impl Line {
        fn of(l: &LogLine) -> Self {
            Self {
                ts: l.unix_ms,
                source: l.source.as_str().to_owned(),
                level: l.level.as_str().to_owned(),
                text: l.text.clone(),
            }
        }
    }

    /// The file's bytes: the header first, then the newest of `lines` (at most
    /// [`EVIDENCE_LINES`]) that fit in [`EVIDENCE_WRITE_BUDGET`] with it.
    ///
    /// The header goes first and is itself bounded (the reason is clipped,
    /// the history is at most 16 entries), so the restart budget is in the
    /// file whatever the lines cost — and a reader that can parse nothing
    /// else can still parse the first line.
    #[must_use]
    pub fn render(header: &Header, lines: &[LogLine]) -> Vec<u8> {
        let mut head = header.clone();
        super::clip(&mut head.reason, 1024);
        let excess = head.restarts_ms.len().saturating_sub(16);
        head.restarts_ms.drain(..excess);
        let mut out = serde_json::to_vec(&HeaderLine {
            mummu_previous_process: 1,
            header: head,
        })
        .unwrap_or_default();
        out.push(b'\n');
        let mut kept: Vec<Vec<u8>> = Vec::new();
        let mut used = out.len();
        for l in lines.iter().rev().take(EVIDENCE_LINES) {
            let Ok(mut row) = serde_json::to_vec(&Line::of(l)) else {
                continue;
            };
            row.push(b'\n');
            if used + row.len() > EVIDENCE_WRITE_BUDGET {
                break;
            }
            used += row.len();
            kept.push(row);
        }
        for row in kept.iter().rev() {
            out.extend_from_slice(row);
        }
        out
    }

    /// Write `bytes` as the evidence file under `dir` (the models root's
    /// copy): created if missing, never through a symlink.
    ///
    /// # Errors
    /// If the directory cannot be created or the file written.
    pub fn write(dir: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        write_into(dir, bytes)
    }

    /// Write `bytes` as the evidence file under the PRIVATE `dir`
    /// ([`super::local_dir`]): created 0700 if missing, refused if it is a
    /// symlink, someone else's, or writable by others.
    ///
    /// # Errors
    /// If the directory is refused or cannot be made, or the file written.
    pub fn write_private(dir: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
        ensure_private_dir(dir)?;
        write_into(dir, bytes)
    }

    fn write_into(dir: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
        let path = dir.join(EVIDENCE_FILE);
        let tmp = dir.join(format!("{EVIDENCE_FILE}.tmp"));
        {
            let mut out = create_nofollow(&tmp)?;
            out.write_all(bytes)?;
            out.sync_all()?;
        }
        // `rename` replaces whatever is at `path` — a symlink planted there
        // included — and never writes through it.
        std::fs::rename(&tmp, &path)?;
        Ok(path)
    }

    /// Open for writing, never following a symlink at the last component
    /// (`O_NOFOLLOW`: a planted link makes the open fail with `ELOOP`), with
    /// the file private to us.
    fn create_nofollow(path: &Path) -> std::io::Result<std::fs::File> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW).mode(0o600);
        }
        options.open(path)
    }

    /// Open for reading without following a symlink and without blocking on
    /// a FIFO (`O_NONBLOCK`; the caller then refuses anything that is not a
    /// regular file), so a planted file can neither redirect nor stall the
    /// startup that reads it.
    fn open_nofollow(path: &Path) -> std::io::Result<std::fs::File> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        options.open(path)
    }

    /// Make `dir` the private directory the local evidence lives in, or say
    /// why it cannot be.
    ///
    /// # Errors
    /// If `dir` is a symlink, not a directory, not ours, or writable by
    /// others — or cannot be created.
    pub fn ensure_private_dir(dir: &Path) -> std::io::Result<()> {
        match std::fs::symlink_metadata(dir) {
            Ok(meta) => check_private(dir, &meta, euid())?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let mut builder = std::fs::DirBuilder::new();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::DirBuilderExt;
                    builder.mode(0o700);
                }
                // Not `recursive`: the parent is the temp dir, and a missing
                // one is not ours to make.
                match builder.create(dir) {
                    Ok(()) => {}
                    // Someone made it between the look and the make: judge
                    // THAT, below.
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(e),
                }
                check_private(dir, &std::fs::symlink_metadata(dir)?, euid())?;
            }
            Err(e) => return Err(e),
        }
        // Ours, and not writable by anyone else: tighten what others may read
        // (a log tail names models and requests).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = std::fs::symlink_metadata(dir)?;
            if meta.permissions().mode() & 0o077 != 0 {
                std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    fn euid() -> u32 {
        // SAFETY: `geteuid` has no preconditions and cannot fail.
        unsafe { libc::geteuid() }
    }

    #[cfg(not(unix))]
    fn euid() -> u32 {
        0
    }

    /// The private-directory rule, as a function of what `lstat` said and
    /// who we are — so the ownership refusal can be tested without a second
    /// user to own something.
    ///
    /// # Errors
    /// A symlink, a non-directory, another owner, or group/other write.
    pub fn check_private(dir: &Path, meta: &std::fs::Metadata, euid: u32) -> std::io::Result<()> {
        let refuse = |why: String| {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("{} {why} — refused", dir.display()),
            ))
        };
        if meta.file_type().is_symlink() {
            return refuse("is a symlink".to_owned());
        }
        if !meta.is_dir() {
            return refuse("is not a directory".to_owned());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if meta.uid() != euid {
                return refuse(format!("belongs to uid {}, not {euid}", meta.uid()));
            }
            if meta.mode() & 0o022 != 0 {
                return refuse(format!(
                    "is writable by others (mode {:o})",
                    meta.mode() & 0o777
                ));
            }
        }
        #[cfg(not(unix))]
        let _ = euid;
        Ok(())
    }

    /// What startup found.
    #[derive(Debug, Default)]
    pub struct Taken {
        pub header: Option<Header>,
        pub lines: Vec<Line>,
        /// Lines present but unreadable.
        pub skipped: usize,
        /// Anything worth telling the operator about the file itself.
        pub note: Option<String>,
        /// The file this came from, when there was one.
        pub from: Option<PathBuf>,
    }

    fn noted(note: String) -> Taken {
        Taken {
            note: Some(note),
            ..Taken::default()
        }
    }

    /// Read what the previous process left in the PRIVATE `dir`
    /// ([`super::local_dir`]) — refused, with a note, if `dir` fails the
    /// private-directory rule, so a directory someone else planted in a
    /// shared temp dir is never replayed onto `/logs`.
    #[must_use]
    pub fn take_private(dir: &Path) -> Taken {
        match std::fs::symlink_metadata(dir) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Taken::default(),
            Err(e) => noted(format!("could not read {}: {e}", dir.display())),
            Ok(meta) => match check_private(dir, &meta, euid()) {
                Ok(()) => take(dir),
                Err(e) => noted(format!("not replaying from it: {e}")),
            },
        }
    }

    /// Read what the previous process left in `dir` and set the file aside so
    /// it is replayed once. Never fails: every problem becomes a `note`.
    #[must_use]
    pub fn take(dir: &Path) -> Taken {
        let path = dir.join(EVIDENCE_FILE);
        let mut file = match open_nofollow(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Taken::default(),
            Err(e) => return noted(format!("could not read {}: {e}", path.display())),
        };
        let mut taken = match file.metadata() {
            Ok(m) if !m.is_file() => noted(format!(
                "{} is not a regular file — ignored",
                path.display()
            )),
            Err(e) => noted(format!("could not read {}: {e}", path.display())),
            Ok(_) => {
                let mut bytes = Vec::new();
                match (&mut file)
                    .take(EVIDENCE_MAX_BYTES + 1)
                    .read_to_end(&mut bytes)
                {
                    Err(e) => noted(format!("could not read {}: {e}", path.display())),
                    Ok(_) if bytes.len() as u64 > EVIDENCE_MAX_BYTES => {
                        let mut t = header_only(&bytes);
                        t.note = Some(format!(
                            "{} is larger than any this server writes ({EVIDENCE_MAX_BYTES} \
                             bytes max) — its log lines were set aside unread{}",
                            path.display(),
                            if t.header.is_some() {
                                "; its restart history was kept"
                            } else {
                                ", and it had no readable header"
                            }
                        ));
                        t
                    }
                    Ok(_) => parse(&String::from_utf8_lossy(&bytes)),
                }
            }
        };
        drop(file);
        taken.from = Some(path.clone());
        let rotated = dir.join(format!("{EVIDENCE_FILE}.1"));
        if std::fs::rename(&path, &rotated).is_err() && std::fs::remove_file(&path).is_err() {
            let note = format!(
                "could not set {} aside; it may be replayed again at the next start",
                path.display()
            );
            taken.note = Some(match taken.note {
                Some(n) => format!("{n}; {note}"),
                None => note,
            });
        }
        taken
    }

    /// The header of a file too big to replay: its first line, which the
    /// writer always puts first. The lines are not read.
    fn header_only(bytes: &[u8]) -> Taken {
        let first = bytes.split(|&b| b == b'\n').next().unwrap_or_default();
        let header = serde_json::from_slice::<HeaderLine>(first)
            .ok()
            .filter(|h| h.mummu_previous_process == 1)
            .map(|h| h.header);
        Taken {
            header,
            ..Taken::default()
        }
    }

    /// Of what the two places held, the one a restart wrote last: a header
    /// beats none, a later exit beats an earlier one, and with no header on
    /// either, lines beat none. The other's note, if any, is kept — it is
    /// about a file the operator may want to know is unreadable.
    #[must_use]
    pub fn newer(a: Taken, b: Taken) -> Taken {
        let at = |t: &Taken| t.header.as_ref().map(|h| h.exited_at_ms);
        let b_wins =
            at(&b) > at(&a) || (at(&a).is_none() && a.lines.is_empty() && !b.lines.is_empty());
        let (mut keep, other) = if b_wins { (b, a) } else { (a, b) };
        if let Some(n) = other.note {
            keep.note = Some(match keep.note {
                Some(k) => format!("{k}; {n}"),
                None => n,
            });
        }
        keep
    }

    /// Parse a file's text. Tolerant by design: a bad header costs the header,
    /// a bad line costs that line, and neither costs the rest.
    #[must_use]
    pub fn parse(text: &str) -> Taken {
        let mut taken = Taken::default();
        let mut rows = text.lines().filter(|l| !l.trim().is_empty()).peekable();
        if let Some(first) = rows.peek()
            && let Ok(head) = serde_json::from_str::<HeaderLine>(first)
            && head.mummu_previous_process == 1
        {
            taken.header = Some(head.header);
            rows.next();
        }
        for row in rows {
            match serde_json::from_str::<Line>(row) {
                Ok(line) => taken.lines.push(line),
                Err(_) => taken.skipped += 1,
            }
        }
        let excess = taken.lines.len().saturating_sub(EVIDENCE_LINES);
        taken.lines.drain(..excess);
        if taken.header.is_none() && (!taken.lines.is_empty() || taken.skipped > 0) {
            taken.note = Some(
                "the previous process's log had no readable header — its lines are replayed, \
                 but not why it exited"
                    .to_owned(),
            );
        }
        if taken.skipped > 0 {
            let note = format!("{} unreadable line(s) in it were skipped", taken.skipped);
            taken.note = Some(match taken.note.take() {
                Some(n) => format!("{n}; {note}"),
                None => format!("the previous process's log: {note}"),
            });
        }
        taken
    }

    impl Taken {
        /// Put the lines into this process's ring, between two banners, each
        /// marked as the previous process's and keeping its own time.
        pub fn replay(&self) {
            if self.lines.is_empty() {
                return;
            }
            let why = self.header.as_ref().map_or_else(
                || "it did not say why".to_owned(),
                |h| {
                    format!(
                        "exited at {} with code {}: {}",
                        super::rfc3339_ms(h.exited_at_ms),
                        h.exit_code,
                        h.reason
                    )
                },
            );
            logs::push(
                Source::Server,
                format!(
                    "[mummu-serve] ── the previous process {why}. Its last {} log lines follow, \
                     marked [previous process] ──",
                    self.lines.len()
                ),
            );
            for l in &self.lines {
                let source = Source::parse_filter(&l.source)
                    .ok()
                    .flatten()
                    .unwrap_or(Source::Server);
                let level = Level::from_wire(&l.level).unwrap_or(Level::Info);
                logs::push_replayed(
                    source,
                    l.ts,
                    level,
                    format!("[previous process] {}", l.text),
                );
            }
            logs::push(
                Source::Server,
                "[mummu-serve] ── end of the previous process's lines; this process starts here ──",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Put every piece of global state back. Tests that touch it hold
/// `crate::progress_serial()`, which the status tests hold too — so an error
/// set here can never leak into their assertions about the phase. The device
/// names stay registered: they are facts about the build, not state.
#[cfg(test)]
pub(crate) fn reset_for_tests() {
    wait_for_exit_threads();
    let mut st = state();
    st.books.clear();
    st.previous = None;
    st.last_cause = None;
    st.restarts_ms.clear();
    st.supervisor = None;
    drop(st);
    EXITING.store(false, SeqCst);
    EXIT_STARTED.store(false, SeqCst);
}

/// Let any exit a test started finish — it writes evidence into that test's
/// scratch dirs and pushes onto the restart history — before the dirs are
/// removed or the state is reset (bounded: the test drain is 3 s, and the
/// hanging-exit test's exit sleeps 3 s).
#[cfg(test)]
pub(crate) fn wait_for_exit_threads() {
    let deadline = Instant::now() + Duration::from_secs(10);
    while EXIT_THREADS.load(SeqCst) > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// [`supervise_with`] for tests elsewhere in the crate: the exits are the
/// test's, the drain and the copy wait are short, and the watchdog's deadline
/// is whatever the test says.
#[cfg(test)]
pub(crate) fn supervise_for_tests(
    models_root: &Path,
    local: &Path,
    exit: fn(i32),
    hard_exit: fn(i32),
    deadline: Duration,
) {
    supervise_with(
        models_root,
        local,
        exit,
        hard_exit,
        ExitTiming {
            drain: Duration::from_secs(3),
            grace: Duration::from_millis(20),
            copy_wait: Duration::from_millis(300),
            deadline,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logs::LogLine;
    use crate::test_seams::Scratch;
    use std::sync::atomic::{AtomicI32, AtomicU32};

    /// A panic exactly as cubecl raises and swallows it: on a thread named
    /// like its device runner, caught right there, never seen by the caller.
    fn device_thread_panic(name: &str, message: &'static str) {
        std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let _ = std::panic::catch_unwind(|| panic!("{message}"));
            })
            .expect("spawn")
            .join()
            .expect("the device thread survives its own panic, as cubecl's does");
    }

    const LOAD_OOM: &str = "failed to reserve 22020096 bytes of device memory: out of device \
                            memory allocating 261319680 bytes";
    const INVALID_READ: &str = "bytes: host access failed: Read(\"The server is in an invalid \
                                state\\nCaused by:\\n  An IO error happened\\nCaused by:\\n  \
                                couldn't find resource for that handle: Memory location was \
                                never initialized\")";
    /// cubecl's kernel-expansion assert (cubecl-std quant/view.rs:340): what
    /// mummu's catch-and-retry sites catch on purpose.
    const KERNEL_GAP: &str =
        "quantized view float vector size 4 must be a positive multiple of num_quants 8";

    const CUDA0: DeviceKey = DeviceKey::Cubecl {
        type_id: 0,
        index: 0,
    };
    const IGPU: DeviceKey = DeviceKey::Cubecl {
        type_id: 1,
        index: 0,
    };

    fn book(key: DeviceKey) -> Option<DeviceSnapshot> {
        snapshot().into_iter().find(|d| d.key == key)
    }

    fn poisoned_on(key: DeviceKey) -> bool {
        book(key).is_some_and(|d| d.poisoned)
    }

    #[test]
    fn device_threads_are_matched_exactly() {
        for (yes, key) in [
            ("DSD-0-0", CUDA0),
            (
                "DSU-3-12",
                DeviceKey::Cubecl {
                    type_id: 3,
                    index: 12,
                },
            ),
            ("DSD-1-0", IGPU),
        ] {
            assert!(is_device_server_thread(yes), "{yes}");
            assert_eq!(DeviceKey::of_thread(yes), Some(key), "{yes}");
        }
        for no in [
            "DSD-",
            "DSD-0",
            "DSD-0-0-0",
            "DSD-a-0",
            "dsd-0-0",
            "DSDX-0-0",
            "DSD-99999-0",
            "tokio-rt-worker",
            "",
        ] {
            assert!(
                !is_device_server_thread(no),
                "{no:?} is not a cubecl device thread"
            );
        }
    }

    /// The signature rule: cubecl's failure text is a device failure wherever
    /// it appears, and a panic on a device thread without it is NOT — the
    /// kernel gap mummu catches and retries (moe.rs:572/1012/1040).
    #[test]
    fn only_cubecl_failure_text_is_a_device_failure() {
        assert!(is_device_failure(LOAD_OOM));
        assert!(is_device_failure(INVALID_READ));
        assert!(is_device_failure(
            "Can create a new stream.: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        assert!(is_device_failure(
            "Must have enough memory for a uniform: OutOfMemory { size: 256 }"
        ));
        for handled_or_ordinary in [
            KERNEL_GAP,
            "Tensor maps not supported in WGPU",
            "index out of bounds: the len is 3 but the index is 7",
            "called `Option::unwrap()` on a `None` value",
            "internal error: entered unreachable code",
            "prompt encoded to zero tokens",
        ] {
            assert!(
                !is_device_failure(handled_or_ordinary),
                "{handled_or_ordinary:?} would unload a model for a panic that is not the GPU's"
            );
        }
    }

    #[test]
    fn a_summary_is_one_readable_bounded_line() {
        let s = summarize(INVALID_READ);
        assert!(!s.contains('\n') && !s.contains("\\n"), "{s}");
        assert!(s.contains("The server is in an invalid state"), "{s}");
        assert!(summarize(&"x".repeat(10_000)).chars().count() <= 241);
    }

    /// The hook counts a device-thread failure that nobody else ever sees —
    /// the load-time half of the incident — against the device whose thread
    /// raised it; a handled kernel gap on the same thread counts for nothing;
    /// and cubecl's text on a thread that is not a device's moves the epoch
    /// but poisons no device (whoever catches it knows which).
    #[test]
    fn the_hook_charges_a_device_failure_to_the_device_that_raised_it() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();

        let before = fault_epoch();
        device_thread_panic("mummu-test-worker", "an ordinary bug");
        device_thread_panic("DSD-0-0", KERNEL_GAP);
        assert_eq!(
            fault_epoch(),
            before,
            "a handled kernel gap on the device thread is not a GPU failure"
        );
        assert!(!poisoned(), "and must not make a healthy server say error");

        device_thread_panic("DSD-0-0", LOAD_OOM);
        assert_eq!(
            fault_epoch(),
            before + 1,
            "cubecl swallowed it; the hook did not"
        );
        assert!(poisoned_on(CUDA0), "charged to CUDA device 0");
        assert!(!poisoned_on(IGPU), "and to no other device");
        let e = current().expect("recorded");
        assert!(e.message.contains("out of device memory"), "{}", e.message);
        assert_eq!(e.recovery, Recovery::Reload);

        device_thread_panic("tokio-rt-worker", INVALID_READ);
        assert_eq!(fault_epoch(), before + 2, "the epoch moves on any path");
        assert_eq!(
            snapshot().iter().filter(|d| d.poisoned).count(),
            1,
            "a failure on a thread that is not a device's poisons no device by guesswork"
        );
        reset_for_tests();
    }

    /// A load during which the device failed must fail AS A LOAD, whether the
    /// device said so by panicking on its own thread or through the sync —
    /// and the failure is the failed device's, not the load's.
    #[test]
    fn a_load_fails_when_a_device_failed_under_it() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();

        let mark = mark();
        assert_eq!(
            load_fault(&mark, &[DeviceKey::Host], || Ok(())),
            None,
            "a clean load is clean"
        );

        // The panic lands DURING the sync — the upload was still queued when
        // the loader returned — and must still fail the load.
        let mark = super::mark();
        let fault = load_fault(&mark, &[CUDA0], || {
            device_thread_panic("DSD-0-0", LOAD_OOM);
            Ok(())
        })
        .expect("the device failed during this load");
        assert!(fault.cause.contains("out of device memory"), "{fault:?}");
        assert_eq!(fault.devices, vec![CUDA0]);

        // A CPU load during which the CARD failed fails too (nothing can
        // tell a race from a cause), but it is the card that is charged.
        let mark = super::mark();
        let fault = load_fault(&mark, &[DeviceKey::Host], || {
            device_thread_panic("DSD-1-0", LOAD_OOM);
            Ok(())
        })
        .expect("failed");
        assert_eq!(fault.devices, vec![IGPU]);

        // No panic, but the device refused the load's work.
        let mark = super::mark();
        let fault = load_fault(&mark, &[CUDA0], || {
            Err((CUDA0, "The server is in an invalid state".into()))
        })
        .expect("a failed sync is a failed load");
        assert!(fault.cause.contains("invalid state"), "{fault:?}");
        assert_eq!(fault.devices, vec![CUDA0]);
        reset_for_tests();
    }

    /// The fix for the empty stream, at its root: a generation that panics
    /// comes back as an error, and which kind depends on whose fault it was.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn contain_turns_panics_into_errors_and_only_device_ones_into_recovery() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();

        let ordinary = contain::<()>("m", async { panic!("index out of bounds") }).await;
        let e = ordinary.expect_err("a panic is an error, not a lost stream");
        assert!(!e.is_device() && e.recovery().is_none(), "{e:?}");
        assert_eq!(e.http_status(), 500);
        assert!(!poisoned(), "an ordinary panic must not poison the GPU");

        let device = contain::<()>("m", async { panic!("{INVALID_READ}") }).await;
        let e = device.expect_err("the incident's read");
        assert!(e.is_device(), "{e:?}");
        assert_eq!(e.recovery(), Some(Recovery::Reload));
        assert_eq!(e.http_status(), 503);
        assert!(e.message.contains("try again"), "{}", e.message);
        assert!(poisoned());
        assert_eq!(e.frame()["type"], json!("error"));
        assert_eq!(e.frame()["recovery"], json!("reload"));

        // A device panic on cubecl's own thread while the request ran makes
        // the request's failure the GPU's even when its own message is not —
        // and it is charged to that device.
        reset_for_tests();
        let e = contain::<()>("m", async {
            device_thread_panic("DSD-0-0", LOAD_OOM);
            panic!("some consequence far from the device");
        })
        .await
        .expect_err("failed");
        assert!(e.is_device(), "{e:?}");
        assert_eq!(book(CUDA0).map(|d| d.consecutive), Some(1));
        reset_for_tests();
    }

    /// MAJOR 1, at the unit: the same failure reported as an ERROR instead
    /// of a panic — an eager readback's `ServerUnhealthy`, as
    /// `mummu::decode::argmax_id` words it — moves the fault epoch exactly as
    /// a panic does, so a model loaded before it is never served again. An
    /// ordinary request error moves nothing.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn an_error_carrying_the_device_failure_moves_the_epoch_like_a_panic() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();

        let before = fault_epoch();
        let e = contain::<()>("m", async {
            Err(ChatError::request(
                "argmax readback: An error happened during execution\nCaused by:\n  The server \
                 is in an invalid state\nCaused by:\n  An IO error happened",
            ))
        })
        .await
        .expect_err("failed");
        assert!(e.is_device(), "{e:?}");
        assert_eq!(e.recovery(), Some(Recovery::Reload));
        assert!(poisoned(), "the status must say error, not ready");
        assert_eq!(
            fault_epoch(),
            before + 1,
            "an Err-path device failure left the epoch where it was: a model loaded before it \
             would be served again"
        );
        // And directly, as the engine records it under the slot lock.
        let e = record_failure("m", &[DeviceKey::Host], "readback failed");
        assert!(!e.needs_decision(), "decided once, never again");
        assert_eq!(fault_epoch(), before + 2);
        assert_eq!(book(DeviceKey::Host).map(|d| d.epoch), Some(1));

        reset_for_tests();
        let before = fault_epoch();
        let e = contain::<()>("m", async {
            Err(ChatError::request("unknown model \"nope\""))
        })
        .await
        .expect_err("failed");
        assert!(!e.is_device() && e.recovery().is_none(), "{e:?}");
        assert!(!poisoned(), "an ordinary error is not a poisoned GPU");
        assert_eq!(fault_epoch(), before);
        reset_for_tests();
    }

    /// The escalation rule: one failure reloads; a second in a row restarts,
    /// once per cooldown, and only under a supervisor.
    #[test]
    fn a_restart_needs_two_failures_a_supervisor_and_an_unspent_budget() {
        let now = 10 * 60 * 60 * 1000;
        let cooldown = RESTART_COOLDOWN.as_millis() as u64;
        assert_eq!(decide(1, true, &[], now), Decision::Reload);
        assert_eq!(decide(2, false, &[], now), Decision::ReloadUnsupervised);
        assert_eq!(decide(2, true, &[], now), Decision::Restart);
        assert_eq!(
            decide(5, true, &[now - cooldown - 1], now),
            Decision::Restart
        );
        assert_eq!(
            decide(2, true, &[now - cooldown - 1, now - 60_000], now),
            Decision::ReloadCoolingDown {
                last_restart_ms: now - 60_000
            },
            "a restart inside the cooldown is the loop this budget exists to stop"
        );
        // And each is told as what it is.
        assert_eq!(Decision::Reload.recovery(), Recovery::Reload);
        assert_eq!(Decision::Restart.recovery(), Recovery::Restart);
        assert_eq!(
            Decision::ReloadCoolingDown { last_restart_ms: 0 }.recovery(),
            Recovery::CoolingDown
        );
        assert_eq!(
            Decision::ReloadUnsupervised.recovery(),
            Recovery::Unsupervised
        );
    }

    static EXITS: AtomicU32 = AtomicU32::new(0);
    static LAST_CODE: AtomicI32 = AtomicI32::new(0);
    fn fake_exit(code: i32) {
        EXITS.fetch_add(1, SeqCst);
        LAST_CODE.store(code, SeqCst);
    }
    fn never_called(_: i32) {
        panic!("the watchdog fired in a test that did not arm a short one");
    }
    const NO_WATCHDOG: Duration = Duration::from_secs(3600);

    fn wait_for(n: &AtomicU32, at_least: u32, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while n.load(SeqCst) < at_least && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        n.load(SeqCst) >= at_least
    }

    /// The exit is taken exactly once, only for a confirmed poison — a device
    /// failure that survived a reload — and it leaves the evidence behind:
    /// the local copy FIRST, the models root's too when it can.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn the_exit_is_taken_once_and_only_on_a_confirmed_poison() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        let root = Scratch::new("exit");
        let local = Scratch::new("exit-local");
        supervise_for_tests(
            root.path(),
            local.path(),
            fake_exit,
            never_called,
            NO_WATCHDOG,
        );
        let before = EXITS.load(SeqCst);

        // Ordinary panics, however many, never exit.
        for _ in 0..3 {
            let _ = contain::<()>("m", async { panic!("an ordinary bug") }).await;
        }
        // One device failure reloads; it does not exit.
        let first = contain::<()>("m", async { panic!("{LOAD_OOM}") }).await;
        assert_eq!(
            first.expect_err("failed").recovery(),
            Some(Recovery::Reload)
        );
        assert!(!restarting());
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            EXITS.load(SeqCst),
            before,
            "one failure is a reload, not a restart"
        );

        // The second in a row, with no token between: the reload did not hold.
        let second = contain::<()>("m", async { panic!("{INVALID_READ}") }).await;
        assert_eq!(
            second.expect_err("failed").recovery(),
            Some(Recovery::Restart)
        );
        assert!(
            restarting(),
            "latched by the time the decision returns, before any exit thread ran"
        );
        // A third failure while exiting must not start a second exit.
        let _ = contain::<()>("m", async { panic!("{INVALID_READ}") }).await;
        assert!(
            wait_for(&EXITS, before + 1, Duration::from_secs(10)),
            "the exit was never taken"
        );
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(EXITS.load(SeqCst), before + 1, "taken once");
        assert_eq!(LAST_CODE.load(SeqCst), EXIT_RESTART);

        let written = local.path().join(EVIDENCE_FILE);
        let taken = evidence::parse(&std::fs::read_to_string(&written).expect("evidence left"));
        let header = taken.header.expect("a header");
        assert_eq!(header.exit_code, EXIT_RESTART);
        assert_eq!(header.restarts_ms.len(), 1);
        // The models root is only the fallback for a local write that failed
        // (see run_exit), so with the local copy on disk it is left alone.
        assert!(
            !root.path().join(EVIDENCE_DIR).join(EVIDENCE_FILE).exists(),
            "the exit wrote to the models root although the local copy was on disk"
        );
        reset_for_tests();
    }

    /// A token between two failures resets the count — ON THE DEVICE THAT
    /// PRODUCED IT. Two failures on the card with a CPU token between them
    /// are still two in a row for the card: that is the sticky fault a
    /// restart exists for, and CPU traffic must not hide it.
    #[test]
    fn a_token_resets_the_count_only_for_its_own_device() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        let root = Scratch::new("reset");
        let local = Scratch::new("reset-local");
        supervise_for_tests(
            root.path(),
            local.path(),
            fake_exit,
            never_called,
            NO_WATCHDOG,
        );

        let _ = record_failure("m", &[DeviceKey::Host], "readback failed");
        generation_succeeded(&[DeviceKey::Host]);
        let e = record_failure("m", &[DeviceKey::Host], "readback failed");
        assert_eq!(
            e.recovery(),
            Some(Recovery::Reload),
            "a token between two failures on one device resets its count"
        );
        assert!(!restarting());

        let exits = EXITS.load(SeqCst);
        let _ = record_failure("m", &[CUDA0], "out of device memory");
        generation_succeeded(&[DeviceKey::Host]);
        assert_eq!(book(CUDA0).map(|d| d.consecutive), Some(1));
        let e = record_failure("m", &[CUDA0], "out of device memory");
        assert_eq!(
            e.recovery(),
            Some(Recovery::Restart),
            "a CPU token reset the card's count, and a sticky card fault would never restart"
        );
        // Let the exit finish writing its evidence before the scratch dirs go
        // (it would recreate them after the guards removed them).
        assert!(wait_for(&EXITS, exits + 1, Duration::from_secs(10)));
        reset_for_tests();
    }

    /// A clean load clears ITS devices' failures and nothing else: a CPU
    /// load proves nothing about the card. A new process shows the previous
    /// one's failure without calling itself poisoned.
    #[test]
    fn a_clean_load_clears_only_its_own_devices() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        device_thread_panic("DSD-0-0", LOAD_OOM);
        assert!(poisoned());
        load_succeeded("qwen2.5-0.5b-instruct", &[DeviceKey::Host]);
        assert!(
            poisoned_on(CUDA0),
            "a CPU load cleared the card's poison (Device::sync on flex is a no-op)"
        );
        assert!(poisoned());
        load_succeeded("qwen3.8-27b-ud-q4ks", &[CUDA0, DeviceKey::Host]);
        assert!(!poisoned());
        assert_eq!(current(), None);
        reset_for_tests();
    }

    fn line(ts: u64, text: &str) -> LogLine {
        LogLine {
            seq: 0,
            unix_ms: ts,
            level: logs::classify(text),
            source: logs::Source::Server,
            quiet: false,
            text: text.to_owned(),
        }
    }

    fn header(at: u64) -> evidence::Header {
        evidence::Header {
            reason: "the GPU backend failed twice".into(),
            exit_code: EXIT_RESTART,
            exited_at_ms: at,
            version: "0.3.2".into(),
            build: "abc1234".into(),
            restarts_ms: vec![at],
            device: Some("GPU (cuda)".into()),
        }
    }

    /// The round trip the restart depends on, and every way the file can be
    /// wrong without costing the startup.
    #[test]
    fn evidence_round_trips_and_a_bad_file_never_fails_startup() {
        let dir = Scratch::new("evidence");
        let head = header(1_758_210_000_000);
        let lines: Vec<LogLine> = (0..EVIDENCE_LINES + 50)
            .map(|i| line(1_758_200_000_000 + i as u64, &format!("line {i}")))
            .collect();
        evidence::write(dir.path(), &evidence::render(&head, &lines)).expect("write");
        let taken = evidence::take(dir.path());
        assert_eq!(taken.header.as_ref(), Some(&head));
        assert_eq!(taken.lines.len(), EVIDENCE_LINES, "bounded to the tail");
        assert_eq!(
            taken.lines.last().map(|l| l.text.as_str()),
            Some(format!("line {}", EVIDENCE_LINES + 49).as_str()),
            "the NEWEST lines are the ones kept"
        );
        assert!(taken.note.is_none(), "{:?}", taken.note);
        assert!(
            !dir.path().join(EVIDENCE_FILE).exists(),
            "replayed once, then set aside"
        );
        assert!(dir.path().join(format!("{EVIDENCE_FILE}.1")).exists());
        assert!(
            evidence::take(dir.path()).header.is_none(),
            "and not replayed twice"
        );

        // Missing: nothing, silently.
        let empty = Scratch::new("evidence-missing");
        let t = evidence::take(empty.path());
        assert!(t.header.is_none() && t.lines.is_empty() && t.note.is_none());

        // Corrupt: a torn header and a torn line cost themselves only.
        std::fs::write(
            dir.path().join(EVIDENCE_FILE),
            "{\"mummu_previous_process\": 1, \"reason\": \n{\"ts\":1,\"source\":\"server\",\"level\":\"error\",\"text\":\"kept\"}\nnot json\n\u{0}\u{1}garbage",
        )
        .expect("write corrupt");
        let t = evidence::take(dir.path());
        assert!(t.header.is_none());
        assert_eq!(t.lines.len(), 1, "the readable line survives");
        assert_eq!(t.lines[0].text, "kept");
        assert_eq!(t.skipped, 3);
        assert!(
            t.note
                .as_deref()
                .is_some_and(|n| n.contains("no readable header"))
        );
    }

    /// MINOR 4: the writer enforces the reader's cap, whatever the lines
    /// cost — 300 lines at the ring's 2048-byte limit of control characters,
    /// each of which JSON writes as six bytes, is ~3.6 MB raw — and the
    /// restart history survives any truncation of the lines: the header is
    /// written first, and a reader handed a file too big to replay still
    /// keeps the header.
    #[test]
    fn the_evidence_is_capped_by_bytes_and_the_restart_history_always_survives() {
        let dir = Scratch::new("evidence-cap");
        let now = now_ms();
        let head = header(now);
        let worst = "\u{1}".repeat(logs::MAX_LINE_BYTES);
        let lines: Vec<LogLine> = (0..EVIDENCE_LINES)
            .map(|i| line(now + i as u64, &format!("{i:04}{worst}")))
            .collect();
        let rendered = evidence::render(&head, &lines);
        assert!(
            rendered.len() <= EVIDENCE_WRITE_BUDGET && (rendered.len() as u64) < EVIDENCE_MAX_BYTES,
            "{} bytes written; the reader sets aside anything over {EVIDENCE_MAX_BYTES}",
            rendered.len()
        );
        evidence::write(dir.path(), &rendered).expect("write");
        let taken = evidence::take(dir.path());
        assert_eq!(
            taken.header.as_ref().map(|h| &h.restarts_ms),
            Some(&vec![now]),
            "the cooldown was lost with the lines"
        );
        assert!(!taken.lines.is_empty(), "as many lines as fit are kept");
        assert!(
            taken
                .lines
                .last()
                .is_some_and(|l| l.text.starts_with(&format!("{:04}", EVIDENCE_LINES - 1))),
            "and they are the newest"
        );

        // A file far past the cap — written by anything — keeps its header.
        let mut huge = evidence::render(&head, &[]);
        huge.extend(std::iter::repeat_n(b'x', (EVIDENCE_MAX_BYTES * 2) as usize));
        std::fs::write(dir.path().join(EVIDENCE_FILE), &huge).expect("write huge");
        let t = evidence::take(dir.path());
        assert_eq!(
            t.header.map(|h| h.restarts_ms),
            Some(vec![now]),
            "an oversized file's restart history was discarded with its lines"
        );
        assert!(t.lines.is_empty());
        assert!(t.note.as_deref().is_some_and(|n| n.contains("set aside")));
        assert!(
            !dir.path().join(EVIDENCE_FILE).exists(),
            "and not tripped over again"
        );
    }

    /// The new process shows what happened: the lines, marked, in its own
    /// ring; the previous failure in its status, without calling its own
    /// fresh backend poisoned; and the restart budget carried forward.
    #[test]
    fn a_restarted_process_shows_what_happened_and_keeps_the_budget() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        let root = Scratch::new("replay");
        let local = Scratch::new("replay-local");
        let now = now_ms();
        let marker = format!("replay-marker-{now}");
        let head = evidence::Header {
            exited_at_ms: now - 5_000,
            restarts_ms: vec![now - 5_000],
            ..header(now)
        };
        evidence::write_private(
            local.path(),
            &evidence::render(
                &head,
                &[line(
                    now - 6_000,
                    &format!("thread 'DSD-0-0' panicked {marker}"),
                )],
            ),
        )
        .expect("write");
        supervise_for_tests(
            root.path(),
            local.path(),
            fake_exit,
            never_called,
            NO_WATCHDOG,
        );

        let e = current().expect("the previous failure is shown");
        assert!(
            e.previous_process && e.recovery == Recovery::Restarted,
            "{e:?}"
        );
        assert_eq!(e.device, "GPU (cuda)");
        assert!(!poisoned(), "this process's backend has not failed");
        let ring = logs::tail(logs::MAX_LINES);
        let replayed = ring
            .iter()
            .find(|l| l.text.contains(&marker))
            .expect("the previous process's line is in this ring");
        assert!(
            replayed.text.starts_with("[previous process] "),
            "{}",
            replayed.text
        );
        assert_eq!(replayed.unix_ms, now - 6_000, "with its own time");
        assert_eq!(replayed.level, logs::Level::Error);
        assert_eq!(
            decide(2, true, &state().restarts_ms, now),
            Decision::ReloadCoolingDown {
                last_restart_ms: now - 5_000
            },
            "a restart five seconds ago must stop the next one"
        );
        // A CPU load does not clear the record of a card failure...
        load_succeeded("m", &[DeviceKey::Host]);
        assert!(current().is_some_and(|e| e.previous_process));
        // ...a load on the card does.
        register_device(CUDA0, "GPU (cuda)");
        load_succeeded("m", &[CUDA0]);
        assert_eq!(current(), None);
        reset_for_tests();
    }

    /// The cooldown is what stops a restart loop, and it lives in the file the
    /// dying process writes — so a models root that cannot take that file
    /// (read-only, full, or the failing array under `/models`) must not
    /// quietly lift it. The local copy is written first, and the next process
    /// finds it there: the budget holds and the lines are replayed.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn the_restart_budget_survives_a_models_root_that_cannot_be_written() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        let root = Scratch::new("unwritable");
        // A FILE where the evidence dir should be: every write under it fails.
        std::fs::write(root.path().join(EVIDENCE_DIR), b"not a directory").expect("block the dir");
        let local = Scratch::new("unwritable-local");
        supervise_for_tests(
            root.path(),
            local.path(),
            fake_exit,
            never_called,
            NO_WATCHDOG,
        );
        let before = EXITS.load(SeqCst);

        let _ = contain::<()>("m", async { panic!("{LOAD_OOM}") }).await;
        let _ = contain::<()>("m", async { panic!("{INVALID_READ}") }).await;
        assert!(
            wait_for(&EXITS, before + 1, Duration::from_secs(10)),
            "the exit was never taken"
        );
        assert!(
            local.path().join(EVIDENCE_FILE).exists(),
            "the models root refused the evidence and nothing else took it"
        );

        // The next process, in the same place.
        reset_for_tests();
        supervise_for_tests(
            root.path(),
            local.path(),
            fake_exit,
            never_called,
            NO_WATCHDOG,
        );
        let e = current().expect("the previous process's failure is shown");
        assert!(e.previous_process, "{e:?}");
        assert!(
            matches!(
                decide(2, true, &state().restarts_ms, now_ms()),
                Decision::ReloadCoolingDown { .. }
            ),
            "the restart budget was lost with the models root: {:?}",
            state().restarts_ms
        );
        assert!(
            !local.path().join(EVIDENCE_FILE).exists(),
            "replayed once, then set aside"
        );
        reset_for_tests();
    }

    #[cfg(unix)]
    fn mkfifo(path: &Path) {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(path.as_os_str().as_bytes()).expect("path");
        // SAFETY: a valid NUL-terminated path; mkfifo has no other contract.
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0, "mkfifo");
    }

    /// Unblock whoever is stuck opening `fifo` for writing, and drain it
    /// until it closes — bounded, and without blocking if nobody is there, so
    /// a regression cannot hang the suite here.
    #[cfg(unix)]
    fn release_fifo(fifo: &Path) {
        use std::io::Read;
        use std::os::unix::fs::OpenOptionsExt;
        let Ok(mut reader) = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(fifo)
        else {
            return;
        };
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut buf = [0u8; 4096];
        while Instant::now() < deadline {
            match reader.read(&mut buf) {
                // EOF: the writer came and went (or never opened; either way
                // nothing is left blocked on this FIFO).
                Ok(0) => {
                    std::thread::sleep(Duration::from_millis(50));
                    if matches!(reader.read(&mut buf), Ok(0)) {
                        return;
                    }
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => return,
            }
        }
    }

    static STALLED_EXITS: AtomicU32 = AtomicU32::new(0);
    fn stalled_exit(_: i32) {
        STALLED_EXITS.fetch_add(1, SeqCst);
    }

    /// MINOR 3, the evidence half: a models root whose write BLOCKS (a FIFO
    /// at the file the writer opens stalls `open()` exactly as the failing
    /// array can — a stall, not an error) does not hold the exit. The models
    /// root is only a fallback, so this makes the LOCAL write fail first (its
    /// directory sits under a regular file) — otherwise the fallback would
    /// never run and the budget below would be tested by nothing.
    #[cfg(unix)]
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn the_exit_finishes_in_time_when_the_models_root_stalls() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        let root = Scratch::new("stall");
        let local = Scratch::new("stall-local");
        let blocker = local.path().join("not-a-dir");
        std::fs::write(&blocker, b"").expect("blocker file");
        let unwritable_local = blocker.join("evidence");
        let dir = root.path().join(EVIDENCE_DIR);
        std::fs::create_dir_all(&dir).expect("dir");
        let fifo = dir.join(format!("{EVIDENCE_FILE}.tmp"));
        mkfifo(&fifo);
        supervise_for_tests(
            root.path(),
            &unwritable_local,
            stalled_exit,
            never_called,
            NO_WATCHDOG,
        );

        let started = Instant::now();
        let _ = record_failure("m", &[CUDA0], "out of device memory");
        let _ = record_failure("m", &[CUDA0], "out of device memory");
        assert!(restarting());
        // drain (nothing open) + grace (20 ms) + copy wait (300 ms) + slack.
        let exited = wait_for(&STALLED_EXITS, 1, Duration::from_secs(3));
        let took = started.elapsed();
        release_fifo(&fifo);
        assert!(
            exited,
            "the exit waited on a stalled models root: {took:?} and counting"
        );
        assert!(
            !unwritable_local.join(EVIDENCE_FILE).exists(),
            "the local write was meant to fail, so that the fallback ran into the stall"
        );
        reset_for_tests();
    }

    /// The exit never touches the models root when the local copy was
    /// written. In production the models root is the btrfs array whose sda
    /// was throwing SATA link resets, and `_exit` cannot end a thread parked
    /// in uninterruptible I/O — so a best-effort second copy there could hold
    /// the restart for as long as the disk takes to time out. The local copy
    /// is what Docker's restart of the same container keeps, so it is enough.
    #[test]
    fn a_written_local_copy_keeps_the_exit_off_the_models_root() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        let root = Scratch::new("offroot");
        let local = Scratch::new("offroot-local");
        supervise_for_tests(
            root.path(),
            local.path(),
            stalled_exit,
            never_called,
            NO_WATCHDOG,
        );
        let before = STALLED_EXITS.load(SeqCst);

        let _ = record_failure("m", &[CUDA0], "out of device memory");
        let _ = record_failure("m", &[CUDA0], "out of device memory");
        assert!(restarting());
        assert!(
            wait_for(&STALLED_EXITS, before + 1, Duration::from_secs(3)),
            "the exit was not taken"
        );
        wait_for_exit_threads();
        assert!(
            local.path().join(EVIDENCE_FILE).exists(),
            "the local copy is the one the next process can count on — and it was not written"
        );
        assert!(
            !root.path().join(EVIDENCE_DIR).exists(),
            "the exit wrote to the models root although the local copy was already on disk"
        );
        reset_for_tests();
    }

    static HANGING_EXITS: AtomicU32 = AtomicU32::new(0);
    static WATCHDOG_EXITS: AtomicU32 = AtomicU32::new(0);
    static WATCHDOG_CODE: AtomicI32 = AtomicI32::new(0);
    fn hanging_exit(_: i32) {
        HANGING_EXITS.fetch_add(1, SeqCst);
        // What driver teardown with cubecl's threads alive can do.
        std::thread::sleep(Duration::from_secs(3));
    }
    fn watchdog_exit(code: i32) {
        WATCHDOG_CODE.store(code, SeqCst);
        WATCHDOG_EXITS.fetch_add(1, SeqCst);
    }

    /// MINOR 3, the exit half: an exit that hangs — in `exit()` itself, the
    /// way `atexit` handlers and driver teardown can — is ended by the
    /// watchdog armed at the start of the exit, at its deadline, with the
    /// restart's exit code. Without it `restarting` would stay latched and
    /// every chat would be refused until a person intervened.
    #[test]
    fn a_hung_exit_is_ended_by_the_watchdog_at_its_deadline() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        let root = Scratch::new("hang");
        let local = Scratch::new("hang-local");
        supervise_for_tests(
            root.path(),
            local.path(),
            hanging_exit,
            watchdog_exit,
            Duration::from_millis(600),
        );
        let started = Instant::now();
        let _ = record_failure("m", &[CUDA0], "CUDA_ERROR_ILLEGAL_ADDRESS");
        let _ = record_failure("m", &[CUDA0], "CUDA_ERROR_ILLEGAL_ADDRESS");
        assert!(
            wait_for(&HANGING_EXITS, 1, Duration::from_secs(3)),
            "the normal exit was never reached"
        );
        assert!(
            wait_for(&WATCHDOG_EXITS, 1, Duration::from_secs(3)),
            "the exit hung and nothing ended the process"
        );
        let took = started.elapsed();
        assert!(
            took >= Duration::from_millis(600) && took < Duration::from_secs(2),
            "the watchdog fired at {took:?}, not at its deadline"
        );
        assert_eq!(WATCHDOG_CODE.load(SeqCst), EXIT_RESTART);
        reset_for_tests();
    }

    /// MINOR 11: the local evidence directory sits in a temp dir other users
    /// share (on a desktop), so it is made private and trusted only if it
    /// is: a symlink, someone else's directory, or one others can write is
    /// refused — for writing AND for replaying onto the public `/logs` — and
    /// no file in it is opened through a symlink.
    #[cfg(unix)]
    #[test]
    fn the_local_evidence_dir_is_private_and_never_followed() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let base = Scratch::new("private");
        let bytes = evidence::render(&header(now_ms()), &[]);

        // Made 0700, and written.
        let dir = base.path().join("mummu-serve");
        let path = evidence::write_private(&dir, &bytes).expect("a fresh private dir");
        assert_eq!(std::fs::metadata(&dir).expect("dir").mode() & 0o777, 0o700);
        assert_eq!(
            std::fs::metadata(&path).expect("file").mode() & 0o077,
            0,
            "the file is ours alone"
        );

        // A symlink where the dir should be: refused, the target untouched.
        let elsewhere = base.path().join("elsewhere");
        std::fs::create_dir(&elsewhere).expect("target");
        let planted = base.path().join("planted");
        std::os::unix::fs::symlink(&elsewhere, &planted).expect("symlink");
        assert!(evidence::write_private(&planted, &bytes).is_err());
        assert!(
            std::fs::read_dir(&elsewhere).expect("dir").next().is_none(),
            "the evidence was written through a symlink"
        );
        assert!(
            evidence::take_private(&planted).header.is_none(),
            "a symlinked dir was replayed"
        );

        // Someone else's: refused (checked against a uid that is not ours).
        let meta = std::fs::symlink_metadata(&dir).expect("meta");
        assert!(evidence::check_private(&dir, &meta, meta.uid() + 1).is_err());
        assert!(evidence::check_private(&dir, &meta, meta.uid()).is_ok());

        // Writable by others: refused.
        let open = base.path().join("open");
        std::fs::create_dir(&open).expect("dir");
        std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o777)).expect("chmod");
        assert!(evidence::write_private(&open, &bytes).is_err());

        // A symlink planted at the file the writer opens: not followed.
        let victim = base.path().join("victim");
        std::fs::write(&victim, b"untouched").expect("victim");
        let tmp = dir.join(format!("{EVIDENCE_FILE}.tmp"));
        std::os::unix::fs::symlink(&victim, &tmp).expect("symlink");
        assert!(
            evidence::write_private(&dir, &bytes).is_err(),
            "opened through a symlink"
        );
        assert_eq!(std::fs::read(&victim).expect("victim"), b"untouched");

        // A FIFO where the file should be: skipped, not waited on.
        let fifo_dir = base.path().join("fifo");
        std::fs::create_dir(&fifo_dir).expect("dir");
        mkfifo(&fifo_dir.join(EVIDENCE_FILE));
        let t = evidence::take(&fifo_dir);
        assert!(t.header.is_none());
        assert!(t.note.is_some_and(|n| n.contains("not a regular file")));
    }

    /// Of the two places, the file a restart wrote last is the one replayed.
    #[test]
    fn the_newer_of_two_evidence_files_wins() {
        let at = |ms: u64| evidence::Taken {
            header: Some(header(ms)),
            ..evidence::Taken::default()
        };
        let pick = |a, b| evidence::newer(a, b).header.map(|h| h.exited_at_ms);
        assert_eq!(pick(at(2), at(1)), Some(2));
        assert_eq!(pick(at(1), at(2)), Some(2));
        assert_eq!(pick(evidence::Taken::default(), at(1)), Some(1));
        assert_eq!(pick(at(1), evidence::Taken::default()), Some(1));
        let noted = evidence::Taken {
            note: Some("could not read it".into()),
            ..evidence::Taken::default()
        };
        let kept = evidence::newer(noted, at(3));
        assert_eq!(kept.header.map(|h| h.exited_at_ms), Some(3));
        assert_eq!(kept.note.as_deref(), Some("could not read it"));
    }
}
