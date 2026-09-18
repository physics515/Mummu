//! Recovering from a crashed GPU backend — by itself, and truthfully.
//!
//! # The incident
//!
//! Production, 2026-09-18, the first cold load after the v0.3.1 deploy.
//! deepseek-ocr, a co-tenant on the shared 16 GiB card, was mid-job and held
//! most of it. The planner reads free VRAM only when `MUMMU_VRAM_LIVE_BUDGET`
//! says so, so it placed 7.35 GiB there anyway. cubecl's device thread
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
//! when a SECOND device failure follows the first with no generation in
//! between that produced a token — the reload was tried and it did not hold —
//! a binary running under a supervisor ([`supervised`]) delivers the error it
//! owes every client, writes the tail of its log where the next process will
//! find it, and exits with [`EXIT_RESTART`] so Docker's `restart:
//! unless-stopped` starts a clean process.
//!
//! # Why that restart cannot loop
//!
//! * Models load lazily. A new process touches the GPU only when a request
//!   arrives, so a restart cannot trigger another restart by itself.
//! * An exit needs TWO consecutive device failures that each cost a request —
//!   in practice two full load attempts — in the same process.
//! * At most one such restart per [`RESTART_COOLDOWN`]. The history travels
//!   from process to process in the evidence file's header, so if the next
//!   load hits the same wall (a co-tenant still holding the card), the new
//!   process does NOT exit again: it stays up, says `error`, answers every
//!   request with the truth, and tries a fresh load per request until one
//!   fits. An operator's own restart (or a deploy) resets the budget.
//! * The history does not hang on one disk. It goes under the models root,
//!   and if that cannot take it, into [`fallback_dir`] — the container's own
//!   writable layer, which a restart of the same container keeps. Only a
//!   write that fails in both places loses it, and even then each restart
//!   still costs two failed requests: the loop is bounded by traffic, never
//!   by a timer.
//!
//! # What is not a GPU failure
//!
//! A panic is a device failure when it happened on one of cubecl's device
//! threads (`DSD-`/`DSU-<type>-<index>`, named at `cubecl-common
//! device/handle/channel.rs:898-907` — a panic there is a device task the
//! client already believes finished), or when its message carries one of
//! cubecl's device-failure signatures ([`DEVICE_SIGNATURES`]). Anything else —
//! an index out of bounds, an `unreachable!` in a model — is reported to the
//! client as an internal error and unloads nothing. The evidence that the
//! rule does not fire on a working card is production's own log: after the
//! 15:59 UTC restart, a cold load of the same 27B and four chats printed zero
//! panics of any kind.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::SeqCst};
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
const DRAIN_BUDGET: Duration = Duration::from_secs(20);

/// After the drain: time for the last frames to leave their sockets and for
/// the stderr tee to carry the exit line into the ring before the tail is
/// taken.
const FLUSH_GRACE: Duration = Duration::from_millis(300);

/// Lines of log a dying process leaves for the next one. A cold 27B load
/// prints ~200, so this holds the load that failed and the requests that
/// failed with it; at [`logs::MAX_LINE_BYTES`] per line it is bounded at well
/// under a megabyte.
pub const EVIDENCE_LINES: usize = 300;

/// A previous-process file larger than this is not ours (ours is bounded by
/// [`EVIDENCE_LINES`]) and is set aside unread rather than parsed at startup.
pub const EVIDENCE_MAX_BYTES: u64 = 1 << 20;

/// Under the models root, which is the one path the container keeps across a
/// restart (`/models`, bind-mounted, already home to cubecl's autotune cache
/// at `/models/.cubecl-cache`). Hidden, so nothing that lists models sees it.
pub const EVIDENCE_DIR: &str = ".mummu-serve";

/// The file a dying process writes and the next one replays.
pub const EVIDENCE_FILE: &str = "previous-process.jsonl";

/// cubecl's device-failure signatures: text that only a failing device puts
/// in a panic message. Each is quoted from the source it comes from.
pub const DEVICE_SIGNATURES: &[&str] = &[
    // cubecl-runtime server/base.rs:324, `ServerError::ServerUnhealthy`.
    "The server is in an invalid state",
    // cubecl-environment bytes/base.rs:446 — a device readback that failed.
    "bytes: host access failed",
    // cubecl-cuda compute/server.rs:244 — the device-thread panic itself.
    "bytes of device memory",
    // cubecl-runtime server/base.rs:904, `IoError::OutOfMemory`.
    "out of device memory",
    // cubecl-runtime memory_management/memory_manage.rs:982.
    "Memory location was never initialized",
    // cudarc's driver errors (`CUDA_ERROR_ILLEGAL_ADDRESS`, ...), which is
    // how a sticky context failure is spelled.
    "CUDA_ERROR_",
];

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Is `name` one of cubecl's device runner threads?
///
/// cubecl names them `DS{U|D}-{type}-{index}` (`cubecl-common
/// device/handle/channel.rs:898-907`) — `DSD-0-0` in the incident. Matched
/// exactly rather than by prefix, so a thread of ours that merely starts with
/// the same letters can never be mistaken for a device.
#[must_use]
pub fn is_device_server_thread(name: &str) -> bool {
    let Some(rest) = name
        .strip_prefix("DSD-")
        .or_else(|| name.strip_prefix("DSU-"))
    else {
        return false;
    };
    let mut parts = rest.split('-');
    let numeric =
        |p: Option<&str>| p.is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    numeric(parts.next()) && numeric(parts.next()) && parts.next().is_none()
}

/// Does a panic on `thread` with `message` mean the GPU backend failed?
///
/// Yes when it happened on a cubecl device thread (a panic there is a device
/// task that the client already believes has finished), or when the message
/// carries one of [`DEVICE_SIGNATURES`]. Everything else is an ordinary bug
/// and must not unload a model or restart a process.
#[must_use]
pub fn is_device_failure(thread: Option<&str>, message: &str) -> bool {
    thread.is_some_and(is_device_server_thread)
        || DEVICE_SIGNATURES.iter().any(|s| message.contains(s))
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
    if out.len() > MAX {
        let mut end = MAX;
        while !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
        out.push('…');
    }
    out
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
/// frame and to the pages in the status object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    /// The model was dropped; the next request loads it again.
    Reload,
    /// This process is exiting so its supervisor starts a clean one.
    Restart,
    /// The PREVIOUS process exited to restart the backend; this one is clean
    /// and has not loaded a model yet.
    Restarted,
}

impl Recovery {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reload => "reload",
            Self::Restart => "restart",
            Self::Restarted => "restarted",
        }
    }
}

/// The unresolved GPU failure the status object and `/api/health` report.
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
        })
    }
}

/// Where a supervised process writes its evidence, and how it exits. The exit
/// is a function pointer only so the tests can watch it being taken.
#[derive(Clone)]
struct Supervisor {
    /// `<models root>/.mummu-serve` — tried first.
    dir: PathBuf,
    /// Where the evidence goes when `dir` cannot take it (see
    /// [`fallback_dir`]). The restart budget travels in that file, so a
    /// write that fails everywhere would let the next process restart
    /// again without waiting out the cooldown.
    fallback: PathBuf,
    exit: fn(i32),
}

/// The second place the evidence can go: the process's temp dir, which in the
/// container is its own writable layer — kept across a `restart:
/// unless-stopped` restart of the same container, and discarded by a
/// recreate (a deploy), which is exactly when the restart budget should
/// reset. `/models` is a bind mount of a spinning array that was failing
/// the day this was written, so the budget must not hang on it alone.
#[must_use]
pub fn fallback_dir() -> PathBuf {
    std::env::temp_dir().join("mummu-serve")
}

struct State {
    error: Option<BackendError>,
    /// The last device-failure cause the panic hook saw, for the load check.
    last_cause: Option<String>,
    /// Unix ms of this process's and its predecessors' self-restarts, oldest
    /// first — the budget [`RESTART_COOLDOWN`] is checked against.
    restarts_ms: Vec<u64>,
    supervisor: Option<Supervisor>,
}

static STATE: Mutex<State> = Mutex::new(State {
    error: None,
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

/// Bumped by every device-failure panic, on the panicking thread, before it
/// unwinds. A model remembers the value it was loaded under; any movement
/// since means the device failed under it (see `engine::drive`).
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// Device failures that each cost a request, since the last generation that
/// produced a token. Two in a row means the reload did not cure it.
static CONSECUTIVE: AtomicU32 = AtomicU32::new(0);

/// Latched by the first decision to exit; never cleared. "Taken once".
static EXITING: AtomicBool = AtomicBool::new(false);

/// Chat responses still open (see [`InFlight`]).
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn rfc3339_ms(ms: u64) -> String {
    crate::shim::rfc3339(UNIX_EPOCH + Duration::from_millis(ms))
}

/// The device-failure count a resident model is compared against.
#[must_use]
pub fn fault_epoch() -> u64 {
    EPOCH.load(SeqCst)
}

/// The unresolved GPU failure, if any — this process's, or the previous
/// one's replayed for the record.
#[must_use]
pub fn current() -> Option<BackendError> {
    state().error.clone()
}

/// Has THIS process's GPU backend failed, with no clean load since? The one
/// predicate behind both the status object's `error` phase and a non-2xx
/// `/api/health`, so the two cannot disagree.
#[must_use]
pub fn poisoned() -> bool {
    state().error.as_ref().is_some_and(|e| !e.previous_process)
}

/// Is this process on its way out to restart the backend? New chats are
/// refused with [`restarting_message`] rather than started on a device that is
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
/// frame (see [`contain`]); the hook only records.
pub fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            previous(info);
            let thread = std::thread::current();
            let message = payload_text(info.payload());
            if is_device_failure(thread.name(), &message) {
                note_device_panic(thread.name().unwrap_or("an unnamed thread"), &message);
            }
        }));
    });
}

/// Record one device-failure panic. Runs inside the panic hook, so it takes
/// one lock, allocates a little, and cannot panic itself.
fn note_device_panic(thread: &str, message: &str) {
    EPOCH.fetch_add(1, SeqCst);
    let cause = summarize(message);
    let first = {
        let mut st = state();
        st.last_cause = Some(cause.clone());
        match st.error.as_mut() {
            Some(e) if !e.previous_process => {
                e.faults += 1;
                false
            }
            _ => {
                st.error = Some(BackendError {
                    message: format!(
                        "the GPU backend failed on {thread}: {cause} — the model will be \
                         dropped and loaded again"
                    ),
                    at_ms: now_ms(),
                    recovery: Recovery::Reload,
                    previous_process: false,
                    faults: 1,
                });
                true
            }
        }
    };
    // Once per failure, not once per panic: the incident's load raised thirty,
    // and the runtime has already printed every one of them.
    if first {
        eprintln!(
            "[mummu-serve] recovery: the GPU backend failed on {thread} ({cause}) — whatever \
             the device was given since is suspect; the model will be dropped and loaded again"
        );
    }
}

// ---------------------------------------------------------------------------
// Detection: the load
// ---------------------------------------------------------------------------

/// Did the device fail while a load ran? `mark` is [`fault_epoch`] from before
/// the load; `sync` waits for the device to finish everything the load
/// submitted.
///
/// The sync comes FIRST and is the reason this works: an upload is submitted,
/// not awaited (module header, step 1), so when the loader returns, a failed
/// allocation may still be sitting in the device queue. Draining the queue
/// makes its panic land — and bump the epoch — before the epoch is compared.
/// A sync that itself reports an error is the same verdict by another route:
/// the device refused work this load gave it.
pub fn load_fault(mark: u64, sync: impl FnOnce() -> Result<(), String>) -> Option<String> {
    let synced = sync();
    if fault_epoch() != mark {
        let cause = state()
            .last_cause
            .clone()
            .unwrap_or_else(|| "a device-thread panic during the load".to_owned());
        return Some(cause);
    }
    synced.err().map(|e| {
        format!(
            "the device reported an error for this load's work: {}",
            summarize(&e)
        )
    })
}

/// A load came up clean: the failure is resolved, as far as the status object
/// and `/api/health` are concerned. (The restart budget is not: that is only
/// earned back by a generation — see [`generation_succeeded`].)
pub fn load_succeeded(model: &str) {
    let cleared = state().error.take();
    if let Some(e) = cleared {
        eprintln!(
            "[mummu-serve] recovery: {model} loaded cleanly — clearing the GPU failure from {} \
             ({} device fault(s))",
            rfc3339_ms(e.at_ms),
            e.faults
        );
    }
}

/// A generation produced its first token, so the device computes and reads
/// back: whatever failed before is behind us.
pub fn generation_succeeded() {
    CONSECUTIVE.store(0, SeqCst);
}

// ---------------------------------------------------------------------------
// Detection and response: the request
// ---------------------------------------------------------------------------

/// Why a chat failed, and what recovery is doing about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatError {
    pub message: String,
    kind: Kind,
    recovery: Option<Recovery>,
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
        }
    }

    /// A GPU failure found WITHOUT a panic — a load whose device work failed.
    /// [`contain`] is what acts on it, exactly as on a device panic.
    pub(crate) fn device(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: Kind::Device,
            recovery: None,
        }
    }

    pub(crate) fn restarting() -> Self {
        Self {
            message: RESTARTING_MESSAGE.to_owned(),
            kind: Kind::Restarting,
            recovery: Some(Recovery::Restart),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: Kind::Internal,
            recovery: None,
        }
    }

    /// Was this the GPU backend's failure (or a refusal because of one)?
    #[must_use]
    pub fn is_device(&self) -> bool {
        matches!(self.kind, Kind::Device | Kind::Restarting)
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
/// included — and acting on the ones that were the GPU's.
///
/// This is the fix for the empty stream. A generation that panicked used to
/// take its channel sender down with it inside `tokio::spawn`, and the client
/// got a 200 with nothing in it. Every chat surface runs its generation
/// through here, so a panic comes back as an error like any other.
///
/// A panic is the GPU's when the panic hook counted a device failure while
/// this request ran, or when its own message says so. Any other panic is an
/// internal error: it is reported, and nothing is unloaded.
///
/// An ERROR can be the GPU's too. The incident's read failed as a panic
/// because cubecl's lazy readback defers the failure to the first touch of
/// the bytes, but an eager read reports the same `ServerUnhealthy` as an
/// `Err`, and mummu's decode turns that into a string (`argmax readback:
/// {e:?}`, `mummu::decode::argmax_id`). A request error carrying a device
/// signature is therefore the same failure by another route, and gets the
/// same response — otherwise the poisoned model would stay resident and
/// fail every chat with a visible but permanent error, which is only half
/// the fix.
pub async fn contain<T>(
    model: &str,
    run: impl Future<Output = Result<T, ChatError>>,
) -> Result<T, ChatError> {
    let mark = fault_epoch();
    match AssertUnwindSafe(run).catch_unwind().await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) if e.kind == Kind::Device => Err(device_failure(model, &e.message)),
        Ok(Err(e)) if e.kind == Kind::Request && is_device_failure(None, &e.message) => {
            Err(device_failure(model, &summarize(&e.message)))
        }
        Ok(Err(e)) => Err(e),
        Err(payload) => {
            let message = payload_text(&*payload);
            if fault_epoch() != mark || is_device_failure(None, &message) {
                Err(device_failure(model, &summarize(&message)))
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

/// What a device failure that cost a request leads to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// First failure (or the first since a generation worked): the model was
    /// dropped, the next request reloads it.
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

/// The client-facing sentence for a decision.
fn decision_message(decision: Decision, cause: &str) -> String {
    match decision {
        Decision::Reload => format!(
            "the GPU backend failed ({cause}). mummu unloaded the model and will load it again \
             on the next request — try again in a moment"
        ),
        Decision::Restart => format!(
            "the GPU backend failed again after a reload ({cause}). mummu is restarting to get a \
             clean GPU context — try again in a minute"
        ),
        Decision::ReloadCoolingDown { last_restart_ms } => format!(
            "the GPU backend failed again after a reload ({cause}). mummu already restarted \
             itself at {} and will not restart again before {}; it unloaded the model and the \
             next request tries a fresh load",
            rfc3339_ms(last_restart_ms),
            rfc3339_ms(last_restart_ms + RESTART_COOLDOWN.as_millis() as u64),
        ),
        Decision::ReloadUnsupervised => format!(
            "the GPU backend failed again after a reload ({cause}). Nothing supervises this \
             process, so it will not restart itself; it unloaded the model and the next request \
             tries a fresh load — restart mummu if this keeps happening"
        ),
    }
}

/// Act on a device failure that cost a request: drop what the device holds,
/// record the failure, decide whether a reload is still worth trying, and
/// hand back the error the client is owed.
fn device_failure(model: &str, cause: &str) -> ChatError {
    // Whatever the decision, what is on the device is suspect.
    crate::engine::evict_after_device_failure();
    let consecutive = CONSECUTIVE.fetch_add(1, SeqCst) + 1;
    let now = now_ms();
    let decision = {
        let st = state();
        decide(consecutive, st.supervisor.is_some(), &st.restarts_ms, now)
    };
    let recovery = if decision == Decision::Restart {
        Recovery::Restart
    } else {
        Recovery::Reload
    };
    let message = decision_message(decision, cause);
    {
        let mut st = state();
        let faults = st
            .error
            .as_ref()
            .filter(|e| !e.previous_process)
            .map_or(0, |e| e.faults);
        st.error = Some(BackendError {
            message: message.clone(),
            at_ms: now,
            recovery,
            previous_process: false,
            faults: faults.max(1),
        });
    }
    eprintln!("[mummu-serve] recovery: chat {model}: {message}");
    if decision == Decision::Restart {
        begin_exit(format!(
            "the GPU backend failed twice in a row, the second time after a reload ({cause})"
        ));
    }
    ChatError {
        message,
        kind: Kind::Device,
        recovery: Some(recovery),
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

/// Start the exit, once. The latch is what makes it "taken once": two
/// requests failing together, or a second failure while the first exit is
/// draining, cannot start a second exit or write the evidence twice.
fn begin_exit(reason: String) {
    let Some(supervisor) = state().supervisor.clone() else {
        return; // `decide` never says Restart without one; belt and braces
    };
    if EXITING.swap(true, SeqCst) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("mummu-restart".to_owned())
        .spawn({
            let supervisor = supervisor.clone();
            let reason = reason.clone();
            move || run_exit(&supervisor, &reason, DRAIN_BUDGET, FLUSH_GRACE)
        });
    if spawned.is_err() {
        // No thread to wait on: skip the drain rather than never exit.
        run_exit(&supervisor, &reason, Duration::ZERO, Duration::ZERO);
    }
}

/// The exit itself: say why, let open responses finish, leave the evidence,
/// go. Blocking, on a thread of its own.
fn run_exit(supervisor: &Supervisor, reason: &str, drain: Duration, grace: Duration) {
    eprintln!(
        "[mummu-serve] recovery: {reason} — exiting with code {EXIT_RESTART} so the supervisor \
         starts a clean process (Docker: restart: unless-stopped); the next request loads the \
         model again"
    );
    let deadline = Instant::now() + drain;
    while IN_FLIGHT.load(SeqCst) > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let open = IN_FLIGHT.load(SeqCst);
    if open > 0 {
        eprintln!(
            "[mummu-serve] recovery: {open} chat response(s) still open after {drain:?}; exiting \
             anyway"
        );
    }
    std::thread::sleep(grace);
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
    };
    let tail = logs::tail(EVIDENCE_LINES);
    let written = evidence::write(&supervisor.dir, &header, &tail).or_else(|e| {
        eprintln!(
            "[mummu-serve] recovery: could not write {} ({e}); trying {}",
            supervisor.dir.display(),
            supervisor.fallback.display()
        );
        evidence::write(&supervisor.fallback, &header, &tail)
    });
    match written {
        Ok(path) => eprintln!(
            "[mummu-serve] recovery: the last {EVIDENCE_LINES} log lines are in {} for the next \
             process",
            path.display()
        ),
        Err(e) => eprintln!(
            "[mummu-serve] recovery: could not leave the log for the next process anywhere ({e}); \
             `docker logs` still has it, but the next process will not know this one restarted"
        ),
    }
    (supervisor.exit)(EXIT_RESTART);
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
/// Also replays what the previous process left in `<root>/.mummu-serve/` (or,
/// if that could not be written, in [`fallback_dir`]), if anything — see
/// [`evidence`]. Never fails: a missing, unreadable, corrupt or oversized file
/// costs a line in the log, not the startup.
pub fn supervised(models_root: &Path) {
    supervise_with(models_root, &fallback_dir(), |code| {
        std::process::exit(code)
    });
}

fn supervise_with(models_root: &Path, fallback: &Path, exit: fn(i32)) {
    // As early as the binary can: a device failure before this is unseen.
    install_panic_hook();
    let dir = models_root.join(EVIDENCE_DIR);
    // Both places, every start: whichever holds a file is set aside, so an
    // old one cannot be replayed at some later start as if it were news.
    let taken = evidence::newer(evidence::take(&dir), evidence::take(fallback));
    let replayed = taken.lines.len();
    {
        let mut st = state();
        st.supervisor = Some(Supervisor {
            dir: dir.clone(),
            fallback: fallback.to_path_buf(),
            exit,
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
            // says what happened until a load proves the card works again.
            st.error = Some(BackendError {
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
            taken.from.as_deref().unwrap_or(&dir).display()
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
/// dying process writes the tail of its ring to `<models root>/.mummu-serve/`
/// (in the container `/models`, the bind mount that already keeps cubecl's
/// autotune cache across restarts), or to [`super::fallback_dir`] when that
/// cannot be written; the next process looks in both, replays the newer into
/// its own ring, marked `[previous process]`, and rotates each file it found
/// to `.1` so nothing is replayed twice.
///
/// JSON Lines: a header object, then one object per log line. Written to a
/// temporary name and renamed into place, so a crash mid-write leaves the
/// previous file or none, never a torn one.
///
/// **A normal shutdown writes nothing** (ctrl-c, `docker stop`, a deploy): the
/// file exists only between a self-restart and the next process's startup,
/// and there is nothing to explain after a shutdown someone asked for.
pub mod evidence {
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize};

    use super::{EVIDENCE_FILE, EVIDENCE_LINES, EVIDENCE_MAX_BYTES};
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

    /// Write `lines` (the newest [`EVIDENCE_LINES`] of them) under `dir`.
    ///
    /// # Errors
    /// If the directory cannot be created or the file written.
    pub fn write(dir: &Path, header: &Header, lines: &[LogLine]) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(EVIDENCE_FILE);
        let tmp = dir.join(format!("{EVIDENCE_FILE}.tmp"));
        {
            let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            let head = HeaderLine {
                mummu_previous_process: 1,
                header: header.clone(),
            };
            serde_json::to_writer(&mut out, &head)?;
            out.write_all(b"\n")?;
            let skip = lines.len().saturating_sub(EVIDENCE_LINES);
            for l in &lines[skip..] {
                serde_json::to_writer(&mut out, &Line::of(l))?;
                out.write_all(b"\n")?;
            }
            out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(path)
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

    /// Read what the previous process left in `dir` and set the file aside so
    /// it is replayed once. Never fails: every problem becomes a `note`.
    #[must_use]
    pub fn take(dir: &Path) -> Taken {
        let path = dir.join(EVIDENCE_FILE);
        let size = match std::fs::metadata(&path) {
            Ok(m) => m.len(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Taken::default(),
            Err(e) => {
                return Taken {
                    note: Some(format!("could not read {}: {e}", path.display())),
                    ..Taken::default()
                };
            }
        };
        let mut taken = if size > EVIDENCE_MAX_BYTES {
            Taken {
                note: Some(format!(
                    "{} is {size} bytes — larger than any this server writes \
                     ({EVIDENCE_MAX_BYTES} max) — set aside unread",
                    path.display()
                )),
                ..Taken::default()
            }
        } else {
            match std::fs::read(&path) {
                Ok(bytes) => parse(&String::from_utf8_lossy(&bytes)),
                Err(e) => Taken {
                    note: Some(format!("could not read {}: {e}", path.display())),
                    ..Taken::default()
                },
            }
        };
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
/// set here can never leak into their assertions about the phase.
#[cfg(test)]
pub(crate) fn reset_for_tests() {
    let mut st = state();
    st.error = None;
    st.last_cause = None;
    st.restarts_ms.clear();
    st.supervisor = None;
    drop(st);
    CONSECUTIVE.store(0, SeqCst);
    EXITING.store(false, SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn device_threads_are_matched_exactly() {
        for yes in ["DSD-0-0", "DSU-3-12", "DSD-10-1"] {
            assert!(is_device_server_thread(yes), "{yes}");
        }
        for no in [
            "DSD-",
            "DSD-0",
            "DSD-0-0-0",
            "DSD-a-0",
            "dsd-0-0",
            "DSDX-0-0",
            "tokio-rt-worker",
            "",
        ] {
            assert!(
                !is_device_server_thread(no),
                "{no:?} is not a cubecl device thread"
            );
        }
    }

    /// Both signatures of the incident are device failures, on the thread
    /// each actually happened on; an ordinary panic on the same threads a
    /// generation runs on is not.
    #[test]
    fn the_incidents_panics_are_device_failures_and_ordinary_ones_are_not() {
        assert!(is_device_failure(Some("DSD-0-0"), LOAD_OOM));
        assert!(is_device_failure(Some("tokio-rt-worker"), INVALID_READ));
        // Anything at all on a device thread is a device task that failed.
        assert!(is_device_failure(Some("DSD-0-0"), "index out of bounds"));
        for ordinary in [
            "index out of bounds: the len is 3 but the index is 7",
            "called `Option::unwrap()` on a `None` value",
            "internal error: entered unreachable code",
            "prompt encoded to zero tokens",
        ] {
            assert!(
                !is_device_failure(Some("tokio-rt-worker"), ordinary),
                "{ordinary:?} would unload a model for a bug that is not the GPU's"
            );
            assert!(!is_device_failure(None, ordinary));
        }
    }

    #[test]
    fn a_summary_is_one_readable_bounded_line() {
        let s = summarize(INVALID_READ);
        assert!(!s.contains('\n') && !s.contains("\\n"), "{s}");
        assert!(s.contains("The server is in an invalid state"), "{s}");
        assert!(summarize(&"x".repeat(10_000)).chars().count() <= 241);
    }

    /// The hook counts a device-thread panic that nobody else ever sees —
    /// the load-time half of the incident — and ignores an ordinary one.
    #[test]
    fn the_hook_counts_what_cubecl_swallows_and_nothing_else() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();

        let before = fault_epoch();
        device_thread_panic("mummu-test-worker", "an ordinary bug");
        assert_eq!(
            fault_epoch(),
            before,
            "an ordinary panic is not a GPU failure"
        );
        assert!(!poisoned());

        device_thread_panic("DSD-0-0", LOAD_OOM);
        assert_eq!(
            fault_epoch(),
            before + 1,
            "cubecl swallowed it; the hook did not"
        );
        assert!(
            poisoned(),
            "a device failure is reported the moment it happens"
        );
        let e = current().expect("recorded");
        assert!(e.message.contains("out of device memory"), "{}", e.message);
        assert_eq!(e.recovery, Recovery::Reload);
        reset_for_tests();
    }

    /// A load during which the device failed must fail AS A LOAD, whether the
    /// device said so by panicking on its own thread or through the sync.
    #[test]
    fn a_load_fails_when_the_device_failed_under_it() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();

        let mark = fault_epoch();
        assert_eq!(load_fault(mark, || Ok(())), None, "a clean load is clean");

        // The panic lands DURING the sync — the upload was still queued when
        // the loader returned — and must still fail the load.
        let mark = fault_epoch();
        let cause = load_fault(mark, || {
            device_thread_panic("DSD-0-0", LOAD_OOM);
            Ok(())
        })
        .expect("the device failed during this load");
        assert!(cause.contains("out of device memory"), "{cause}");

        // No panic, but the device refused the load's work.
        let mark = fault_epoch();
        let cause = load_fault(mark, || Err("The server is in an invalid state".into()))
            .expect("a failed sync is a failed load");
        assert!(cause.contains("invalid state"), "{cause}");
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
        // the request's failure the GPU's even when its own message is not.
        reset_for_tests();
        let e = contain::<()>("m", async {
            device_thread_panic("DSD-0-0", LOAD_OOM);
            panic!("some consequence far from the device");
        })
        .await
        .expect_err("failed");
        assert!(e.is_device(), "{e:?}");
        reset_for_tests();
    }

    /// The same failure, reported as an error instead of a panic: an eager
    /// readback's `ServerUnhealthy`, as `mummu::decode::argmax_id` words it.
    /// It must unload the model like the panic does; an ordinary request
    /// error must not.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_readback_error_carrying_the_device_failure_is_one() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();

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

        reset_for_tests();
        let e = contain::<()>("m", async {
            Err(ChatError::request("unknown model \"nope\""))
        })
        .await
        .expect_err("failed");
        assert!(!e.is_device() && e.recovery().is_none(), "{e:?}");
        assert!(!poisoned(), "an ordinary error is not a poisoned GPU");
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
    }

    static EXITS: AtomicU32 = AtomicU32::new(0);
    static LAST_CODE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    fn fake_exit(code: i32) {
        EXITS.fetch_add(1, SeqCst);
        LAST_CODE.store(code, SeqCst);
    }

    fn wait_for_exits(n: u32) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while EXITS.load(SeqCst) < n && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        EXITS.load(SeqCst) >= n
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mummu-serve-recovery-{name}-{}-{}",
            std::process::id(),
            now_ms()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The exit is taken exactly once, only for a confirmed poison — a device
    /// failure that survived a reload — and it leaves the evidence behind.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn the_exit_is_taken_once_and_only_on_a_confirmed_poison() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        let root = scratch("exit");
        let fallback = scratch("exit-fallback");
        supervise_with(&root, &fallback, fake_exit);
        EXITS.store(0, SeqCst);

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
            0,
            "one failure is a reload, not a restart"
        );

        // The second in a row, with no token between: the reload did not hold.
        let second = contain::<()>("m", async { panic!("{INVALID_READ}") }).await;
        assert_eq!(
            second.expect_err("failed").recovery(),
            Some(Recovery::Restart)
        );
        assert!(restarting(), "new chats are refused from here on");
        // A third failure while exiting must not start a second exit.
        let _ = contain::<()>("m", async { panic!("{INVALID_READ}") }).await;
        assert!(wait_for_exits(1), "the exit was never taken");
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(EXITS.load(SeqCst), 1, "taken once");
        assert_eq!(LAST_CODE.load(SeqCst), EXIT_RESTART);

        let written = root.join(EVIDENCE_DIR).join(EVIDENCE_FILE);
        let taken = evidence::parse(&std::fs::read_to_string(&written).expect("evidence left"));
        let header = taken.header.expect("a header");
        assert_eq!(header.exit_code, EXIT_RESTART);
        assert_eq!(header.restarts_ms.len(), 1);
        assert!(
            !fallback.join(EVIDENCE_FILE).exists(),
            "the models root took it; the fallback is for when it cannot"
        );
        reset_for_tests();
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&fallback);
    }

    /// A generation that produced a token between two failures resets the
    /// count: two unrelated failures an hour apart are two reloads.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_generation_between_failures_resets_the_count() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        let root = scratch("reset");
        let fallback = scratch("reset-fallback");
        supervise_with(&root, &fallback, fake_exit);
        let before = EXITS.load(SeqCst);

        let _ = contain::<()>("m", async { panic!("{LOAD_OOM}") }).await;
        generation_succeeded();
        let e = contain::<()>("m", async { panic!("{LOAD_OOM}") })
            .await
            .expect_err("failed");
        assert_eq!(e.recovery(), Some(Recovery::Reload));
        assert!(!restarting());
        assert_eq!(EXITS.load(SeqCst), before);
        reset_for_tests();
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&fallback);
    }

    /// A clean load clears the status error; a new process shows the previous
    /// one's failure without calling itself poisoned.
    #[test]
    fn a_clean_load_clears_the_error() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        device_thread_panic("DSD-0-0", LOAD_OOM);
        assert!(poisoned());
        load_succeeded("qwen3.5-2b");
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

    use crate::logs::LogLine;

    /// The round trip the restart depends on, and every way the file can be
    /// wrong without costing the startup.
    #[test]
    fn evidence_round_trips_and_a_bad_file_never_fails_startup() {
        let dir = scratch("evidence");
        let header = evidence::Header {
            reason: "the GPU backend failed twice".into(),
            exit_code: EXIT_RESTART,
            exited_at_ms: 1_758_210_000_000,
            version: "0.3.2".into(),
            build: "abc1234".into(),
            restarts_ms: vec![1_758_210_000_000],
        };
        let lines: Vec<LogLine> = (0..EVIDENCE_LINES + 50)
            .map(|i| line(1_758_200_000_000 + i as u64, &format!("line {i}")))
            .collect();
        evidence::write(&dir, &header, &lines).expect("write");
        let taken = evidence::take(&dir);
        assert_eq!(taken.header.as_ref(), Some(&header));
        assert_eq!(taken.lines.len(), EVIDENCE_LINES, "bounded to the tail");
        assert_eq!(
            taken.lines.last().map(|l| l.text.as_str()),
            Some(format!("line {}", EVIDENCE_LINES + 49).as_str()),
            "the NEWEST lines are the ones kept"
        );
        assert!(taken.note.is_none(), "{:?}", taken.note);
        assert!(
            !dir.join(EVIDENCE_FILE).exists(),
            "replayed once, then set aside"
        );
        assert!(dir.join(format!("{EVIDENCE_FILE}.1")).exists());
        assert!(
            evidence::take(&dir).header.is_none(),
            "and not replayed twice"
        );

        // Missing: nothing, silently.
        let empty = scratch("evidence-missing");
        let t = evidence::take(&empty);
        assert!(t.header.is_none() && t.lines.is_empty() && t.note.is_none());

        // Corrupt: a torn header and a torn line cost themselves only.
        std::fs::write(
            dir.join(EVIDENCE_FILE),
            "{\"mummu_previous_process\": 1, \"reason\": \n{\"ts\":1,\"source\":\"server\",\"level\":\"error\",\"text\":\"kept\"}\nnot json\n\u{0}\u{1}garbage",
        )
        .expect("write corrupt");
        let t = evidence::take(&dir);
        assert!(t.header.is_none());
        assert_eq!(t.lines.len(), 1, "the readable line survives");
        assert_eq!(t.lines[0].text, "kept");
        assert_eq!(t.skipped, 3);
        assert!(
            t.note
                .as_deref()
                .is_some_and(|n| n.contains("no readable header"))
        );

        // Huge: set aside unread.
        std::fs::write(
            dir.join(EVIDENCE_FILE),
            vec![b'x'; (EVIDENCE_MAX_BYTES + 1) as usize],
        )
        .expect("write huge");
        let t = evidence::take(&dir);
        assert!(t.lines.is_empty() && t.header.is_none());
        assert!(
            t.note
                .as_deref()
                .is_some_and(|n| n.contains("set aside unread"))
        );
        assert!(
            !dir.join(EVIDENCE_FILE).exists(),
            "and not tripped over again"
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// The new process shows what happened: the lines, marked, in its own
    /// ring; the previous failure in its status, without calling its own
    /// fresh backend poisoned; and the restart budget carried forward.
    #[test]
    fn a_restarted_process_shows_what_happened_and_keeps_the_budget() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        let root = scratch("replay");
        let now = now_ms();
        let header = evidence::Header {
            reason: "the GPU backend failed twice in a row".into(),
            exit_code: EXIT_RESTART,
            exited_at_ms: now - 5_000,
            version: "0.3.2".into(),
            build: "abc1234".into(),
            restarts_ms: vec![now - 5_000],
        };
        let marker = format!("replay-marker-{now}");
        evidence::write(
            &root.join(EVIDENCE_DIR),
            &header,
            &[line(
                now - 6_000,
                &format!("thread 'DSD-0-0' panicked {marker}"),
            )],
        )
        .expect("write");

        let fallback = scratch("replay-fallback");
        supervise_with(&root, &fallback, fake_exit);

        let e = current().expect("the previous failure is shown");
        assert!(
            e.previous_process && e.recovery == Recovery::Restarted,
            "{e:?}"
        );
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
        reset_for_tests();
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&fallback);
    }

    /// The cooldown is what stops a restart loop, and it lives in the file the
    /// dying process writes — so a models root that cannot take that file
    /// (read-only, full, or the failing array under `/models`) must not
    /// quietly lift it. The evidence goes to the fallback, and the next
    /// process finds it there: the budget holds and the lines are replayed.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn the_restart_budget_survives_a_models_root_that_cannot_be_written() {
        let _serial = crate::progress_serial();
        reset_for_tests();
        install_panic_hook();
        let root = scratch("unwritable");
        // A FILE where the evidence dir should be: every write under it fails.
        std::fs::write(root.join(EVIDENCE_DIR), b"not a directory").expect("block the dir");
        let fallback = scratch("unwritable-fallback");
        supervise_with(&root, &fallback, fake_exit);
        let before = EXITS.load(SeqCst);

        let _ = contain::<()>("m", async { panic!("{LOAD_OOM}") }).await;
        let _ = contain::<()>("m", async { panic!("{INVALID_READ}") }).await;
        assert!(wait_for_exits(before + 1), "the exit was never taken");
        assert!(
            fallback.join(EVIDENCE_FILE).exists(),
            "the models root refused the evidence and nothing else took it"
        );

        // The next process, in the same place.
        reset_for_tests();
        supervise_with(&root, &fallback, fake_exit);
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
            !fallback.join(EVIDENCE_FILE).exists(),
            "replayed once, then set aside"
        );
        reset_for_tests();
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&fallback);
    }

    /// Of the two places, the file a restart wrote last is the one replayed.
    #[test]
    fn the_newer_of_two_evidence_files_wins() {
        let at = |ms: u64| evidence::Taken {
            header: Some(evidence::Header {
                reason: format!("exit at {ms}"),
                exit_code: EXIT_RESTART,
                exited_at_ms: ms,
                version: "0.3.2".into(),
                build: "abc1234".into(),
                restarts_ms: vec![ms],
            }),
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
