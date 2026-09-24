//! mummu-serve — a minimal HTTP server + single-page chat UI over mummu.
//!
//! This is the library half. The `mummu-serve` binary is a thin wrapper that
//! reads the environment and calls [`serve`]; the Tauri desktop shell
//! (`mummu-app`) binds its own listeners with [`bind`] and drives them with
//! [`serve_on`], so both front ends run byte-identical routers over the same
//! process-wide model slot.
//!
//! axum on a multi-threaded tokio runtime. A request that starts long work
//! answers immediately with a stream and lets the work run as its own task,
//! feeding a `tokio::sync::mpsc` channel the response body drains: the async
//! engine (`engine::run_chat`) as a spawned task, the still-synchronous
//! blocking pieces — the hub downloader, the device probe, dropping a
//! resident model — on `spawn_blocking`, so a runtime worker is never
//! parked on minutes of CPU/GPU work. Endpoints:
//!
//! - `GET  /`            the embedded chat UI
//! - `GET  /logs`        the embedded merged-log page (see [`logs`])
//! - `GET  /favicon.ico` 204 — there is no icon, and a 404 would read as a scan
//! - `GET  /api/health`  device policy + adapter inventory
//! - `GET  /api/logs`    the merged server/api/shim log, since a cursor
//! - `GET  /api/models`  the catalog with installed flags
//! - `POST /api/pull`    download a catalog model (SSE progress)
//! - `POST /api/chat`    stream a chat completion (SSE deltas)
//! - `GET  /api/chat/ws` the same frames over a WebSocket — what the UI uses,
//!   because a proxy that cuts a 100-second HTTP response cannot serve a model
//!   that takes minutes to load (see `chat_ws`)
//! - `POST /api/unload`  drop the resident model (frees VRAM/RAM)
//!
//! Configuration is the *caller's* job — addresses are arguments here, not
//! environment reads, so the binary and the desktop shell can default
//! differently (`0.0.0.0` in a container, loopback on a desktop) without one
//! silently overriding the other. `MUMMU_MODELS_DIR` and the engine's own
//! `MUMMU_BACKEND` / `MUMMU_FORCE_CPU` / fit-planner variables stay where
//! they were, read at the point of use.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

/// How a build names itself. Compiled here only for the tests: `build.rs`
/// includes the same file with `#[path]` and is the thing that calls it, and
/// a build script is not a test target — so without this the stamp rules
/// would be the one part of the release nothing checks.
#[cfg(test)]
mod build_sha;
mod engine;
/// Makes the generation path fail the way the 2026-09-18 incident did, on a
/// machine with no GPU. Compiled only with the `fault-injection` feature —
/// see the module for why it cannot reach production.
#[cfg(feature = "fault-injection")]
mod fault;
pub mod logs;
mod openai;
pub mod recovery;
mod shim;
pub mod status;
mod think;
pub mod trace;

/// `mummu::progress` is process-wide state and `cargo test` runs this crate's
/// tests in parallel threads of one process, so every test that WRITES it —
/// in any module of this crate — takes this first. Reading it under the lock
/// is what makes an assertion about the phase mean anything.
///
/// A tokio mutex rather than a std one: an async test holds the guard across
/// its awaits, which is the whole point, and a std guard held across an
/// `.await` is the deadlock shape `clippy::await_holding_lock` exists for.
#[cfg(test)]
pub(crate) static PROGRESS_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Take [`PROGRESS_SERIAL`] from an async test.
#[cfg(test)]
pub(crate) async fn progress_serial() -> tokio::sync::MutexGuard<'static, ()> {
    PROGRESS_SERIAL.lock().await
}

/// Take [`PROGRESS_SERIAL`] from a synchronous test (no runtime on the thread).
#[cfg(test)]
pub(crate) fn progress_serial_blocking() -> tokio::sync::MutexGuard<'static, ()> {
    PROGRESS_SERIAL.blocking_lock()
}

use std::convert::Infallible;
use std::future::Future;
use std::ops::ControlFlow;
use std::path::PathBuf;

use axum::Router;
use axum::body::Bytes;
use axum::extract::DefaultBodyLimit;
use axum::http::header;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use mummu::chat::{Role, Turn};
use mummu::decode::SamplerOptions;
use mummu::manage::ModelManager;
use mummu_num::{f64_from_u64, f64_from_usize, trunc_i64};
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};

pub use engine::device_label;

/// The single-page chat UI, exactly as `GET /` serves it. Exposed so a shell
/// that wants to embed the same bytes (the Tauri app's offline fallback) has
/// one source of truth instead of a copy that drifts.
pub const UI_HTML: &str = include_str!("ui.html");

/// The logs page, exactly as `GET /logs` serves it — same reason as
/// [`UI_HTML`].
pub const LOGS_HTML: &str = logs::LOGS_HTML;

/// Default listen address of the native API + UI.
pub const DEFAULT_ADDR: &str = "0.0.0.0:8095";
/// Default listen address of the ollama-compatibility shim. 11435 rather
/// than ollama's 11434 so a real ollama sharing the network namespace can
/// never collide with it.
pub const DEFAULT_OLLAMA_ADDR: &str = "0.0.0.0:11435";

/// Hard ceilings so one request can't wedge the process.
pub(crate) const MAX_BODY_BYTES: usize = 4 << 20;
pub(crate) const MAX_TURNS: usize = 256;
pub(crate) const MAX_MAX_TOKENS: usize = 4096;
pub(crate) const DEFAULT_MAX_TOKENS: usize = 512;

pub(crate) fn models_root() -> PathBuf {
    #[cfg(test)]
    if let Some(root) = test_seams::models_root() {
        return root;
    }
    std::env::var_os("MUMMU_MODELS_DIR").map_or_else(|| PathBuf::from("models"), PathBuf::from)
}

/// What the environment would say, said by a test instead: the models root
/// and the backend. `std::env::set_var` is `unsafe` in a process whose other
/// threads read the environment — which is every test binary — so the tests
/// that drive the real engine set these, under `progress_serial`, and put
/// them back when they are done. Nothing else about the path changes.
#[cfg(test)]
pub(crate) mod test_seams {
    use std::path::PathBuf;
    use std::sync::Mutex;

    static MODELS_ROOT: Mutex<Option<PathBuf>> = Mutex::new(None);
    static BACKEND: Mutex<Option<crate::engine::BackendChoice>> = Mutex::new(None);

    pub fn models_root() -> Option<PathBuf> {
        MODELS_ROOT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    #[cfg(feature = "fault-injection")]
    pub fn set_models_root(root: Option<PathBuf>) {
        *MODELS_ROOT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = root;
    }

    pub fn backend() -> Option<crate::engine::BackendChoice> {
        *BACKEND
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(feature = "fault-injection")]
    pub fn set_backend(backend: Option<crate::engine::BackendChoice>) {
        *BACKEND
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = backend;
    }

    /// A scratch directory that removes itself when dropped — on a failed
    /// assertion too, which unwinds through it. An earlier run of these tests
    /// cleaned up only on success and left seventeen directories in `/tmp`.
    pub struct Scratch(PathBuf);

    impl Scratch {
        pub(crate) fn new(name: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "mummu-serve-test-{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Self(dir)
        }

        pub(crate) fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // An exit a test started writes its evidence here; let it finish,
            // or it recreates the directory after this removes it (which is
            // how a failing test used to leak dirs into /tmp).
            crate::recovery::wait_for_exit_threads();
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points. The binary reads the environment and calls `serve`;
// the desktop shell binds first (so it can point a window at the port it
// actually got) and calls `serve_on` with its own shutdown trigger.
// ---------------------------------------------------------------------------

/// The models root, from `MUMMU_MODELS_DIR` or `./models`, created if
/// missing. Both front ends call this before serving — a missing root is a
/// startup error, not a per-request one.
///
/// # Errors
/// If the directory can't be created.
pub fn prepare_models_root() -> std::io::Result<PathBuf> {
    let root = models_root();
    std::fs::create_dir_all(&root)?;
    Ok(root)
}

/// Report the device policy once at startup — the honest record of whether
/// this environment actually has a usable GPU adapter.
///
/// Probing adapters is blocking work; callers on an async thread should hand
/// this to `spawn_blocking`.
pub fn log_device_policy(root: &std::path::Path) {
    let inv = mummu::backend::inventory();
    for gpu in &inv.gpus {
        eprintln!(
            "[mummu-serve] adapter: {} ({:?} / {:?}), SHADER_F16 = {}",
            gpu.name, gpu.backend, gpu.device_type, gpu.shader_f16
        );
    }
    eprintln!(
        "[mummu-serve] device policy: {} | models root: {}",
        engine::device_label(),
        root.display()
    );
}

/// Is this `MUMMU_OLLAMA_ADDR` value one of the spellings that turn the shim
/// off? Kept here so the binary and the app agree on what "off" means.
#[must_use]
pub fn shim_disabled(addr: &str) -> bool {
    matches!(addr, "" | "off" | "disabled" | "0")
}

/// Bind a listener, naming the address in the error so a failure to take
/// port 8095 says which port it was.
///
/// # Errors
/// If the address can't be resolved or bound.
pub async fn bind(addr: &str) -> std::io::Result<TcpListener> {
    TcpListener::bind(addr)
        .await
        .map_err(|e| std::io::Error::new(e.kind(), format!("bind {addr}: {e}")))
}

/// Bind both listeners and serve until ctrl-c. This is exactly what the
/// `mummu-serve` binary does; `shim_addr` of `None` (or one of the "off"
/// spellings) leaves the ollama surface unbound.
///
/// # Errors
/// If the native API listener can't be bound, or if axum's accept loop
/// fails. A shim that can't bind is logged and skipped — the native API
/// keeps serving, which is the behavior the binary has always had.
pub async fn serve(addr: &str, shim_addr: Option<&str>) -> std::io::Result<()> {
    let api = bind(addr).await?;
    let shim = match shim_addr.filter(|a| !shim_disabled(a)) {
        Some(a) => match bind(a).await {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("[mummu-serve] ollama shim: {e} — shim disabled");
                None
            }
        },
        None => None,
    };
    serve_on(api, shim, shutdown_signal()).await
}

/// Serve pre-bound listeners until `shutdown` resolves, then drain both.
///
/// Taking listeners rather than addresses is what lets a caller learn the
/// port before anything is served on it — the desktop shell needs the bound
/// `local_addr` to point its window at, and a caller asking for port 0 would
/// otherwise never find out what it got.
///
/// # Errors
/// If either accept loop fails.
pub async fn serve_on<F>(
    api: TcpListener,
    shim: Option<TcpListener>,
    shutdown: F,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    // Before anything else prints: everything written before the tee is
    // installed reaches only `docker logs`, and the lines worth seeing start
    // at the first model load. Idempotent, so the binary having already
    // installed it (earlier, in `main`) costs nothing.
    logs::install();
    // Next, before anything can touch a GPU: a device failure is noticed by
    // this hook (cubecl swallows it on its own thread — see `recovery`), so a
    // load that runs before it is installed could fail unseen. Idempotent.
    recovery::install_panic_hook();
    // Start watching host memory as soon as we are serving: the pressure it
    // guards against arrives from OTHER processes, so it must not depend on
    // this one receiving traffic. See `engine::spawn_host_pressure_watch`.
    engine::spawn_host_pressure_watch();
    // And the card: which layers live there, at what precision, follows what
    // the card has free — also without waiting for traffic.
    engine::spawn_placement_watch();
    // And take the first memory readings now. The VRAM cache answers from its
    // last sample and refreshes behind it, so the first load's baseline would
    // otherwise be "nothing sampled yet" on a server nobody has polled —
    // see `status::prime`.
    status::prime();
    // One trigger, two listeners: `with_graceful_shutdown` consumes a
    // future, and futures aren't cloneable, so the trigger fans out through
    // a watch channel.
    let (tx, rx) = watch::channel(false);
    tokio::spawn(async move {
        shutdown.await;
        let _ = tx.send(true);
    });

    let shim_task = shim.map(|listener| {
        let rx = rx.clone();
        match listener.local_addr() {
            Ok(a) => {
                eprintln!("[mummu-serve] ollama-compatible shim listening on http://{a}");
            }
            Err(e) => eprintln!("[mummu-serve] ollama shim: local_addr: {e}"),
        }
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, shim::router())
                .with_graceful_shutdown(triggered(rx, "ollama shim"))
                .await
            {
                eprintln!("[mummu-serve] ollama shim: serve failed: {e}");
            }
        })
    });

    match api.local_addr() {
        Ok(a) => eprintln!("[mummu-serve] listening on http://{a}"),
        Err(e) => eprintln!("[mummu-serve] api: local_addr: {e}"),
    }

    // One router, many concurrent requests: generations serialize on the
    // model-slot mutex inside `spawn_blocking`, but health/models/UI
    // requests keep answering on the async workers while one runs.
    let result = axum::serve(api, router())
        .with_graceful_shutdown(triggered(rx, "api"))
        .await;
    if let Err(e) = &result {
        eprintln!("[mummu-serve] serve failed: {e}");
    }
    if let Some(task) = shim_task {
        let _ = task.await;
    }
    result
}

/// Resolve once the watch channel has been flipped (or dropped, which can
/// only happen if the trigger task was cancelled — treat that as "stop").
async fn triggered(mut rx: watch::Receiver<bool>, which: &'static str) {
    while !*rx.borrow_and_update() {
        if rx.changed().await.is_err() {
            break;
        }
    }
    eprintln!("[mummu-serve] draining the {which} listener");
}

/// The native API + UI router.
pub fn router() -> Router {
    let routes = Router::new()
        .route("/", get(ui))
        .route("/index.html", get(ui))
        // The merged log feed and the page that reads it. Beside /api/health
        // because they answer the same question — is this thing alive? — and
        // the log is the half that says what it is *doing*.
        .route("/logs", get(logs::page))
        .route("/favicon.ico", get(favicon))
        .route("/api/health", get(health))
        .route("/api/logs", get(logs::endpoint))
        .route("/api/models", get(models))
        .route("/api/pull", post(pull))
        .route("/api/chat", post(chat))
        // Same frames as /api/chat, over a transport Cloudflare will not cut
        // at 100 s — see `chat_ws`.
        .route("/api/chat/ws", get(chat_ws))
        .route("/api/unload", post(unload))
        // Flame graphs from a profiled generation — see `profile_svg`.
        .route("/api/profile", get(profile_svg))
        .route("/api/profile/folded", get(profile_folded));
    // Arms the incident's failures on a build that asked for them, and does
    // not exist on any other: without the feature this route is not compiled,
    // and the request falls through to the 404 below like any unknown path.
    #[cfg(feature = "fault-injection")]
    let routes = routes.route("/api/fault", post(fault::endpoint).get(fault::state));
    routes
        // The sync server matched on (method, path) and answered anything
        // else with the same 404 JSON — keep that, rather than axum's bare
        // 405, so a client sees one error shape.
        .fallback(not_found)
        .method_not_allowed_fallback(not_found)
        // `+ 1` so a body just over the ceiling still reaches the handler
        // and gets the JSON "body too large" the sync reader produced.
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES + 1))
        // Outermost, so it also records the requests the layers below reject
        // (a body over the ceiling, a route that does not exist) — those are
        // exactly the ones an operator is hunting when nothing works.
        .layer(axum::middleware::from_fn(logs::record_api))
}

/// The ollama-compatibility router, for a caller that wants to mount or
/// serve that surface itself.
pub fn ollama_router() -> Router {
    shim::router()
}

/// Resolve when the process is asked to stop. `serve_on` fans this out to
/// every listener through a watch channel.
pub async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => eprintln!("[mummu-serve] ctrl-c"),
        Err(e) => eprintln!("[mummu-serve] ctrl-c handler unavailable: {e}"),
    }
}

/// Run blocking work (device probes, disk scans, the engine) on a blocking
/// pool thread. A panic inside is re-raised here, exactly as it would have
/// surfaced on the sync server's worker thread.
pub(crate) async fn blocking<T, F>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(value) => value,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

pub(crate) fn json_response(status: u16, body: &serde_json::Value) -> Response {
    let status = axum::http::StatusCode::from_u16(status)
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

/// Whole milliseconds of `d`, saturating at `u64::MAX`: `Duration::as_millis`
/// is a `u128`, which nothing on the wire or in a trace wants.
pub(crate) fn millis(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Whole nanoseconds of `d`, saturating at `u64::MAX` (see [`millis`]).
pub(crate) fn nanos(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// How long a buffered answer may take before the connection needs
/// reassuring. Everything that fails fast — validation, an unknown model,
/// the 503 while a model loads — settles far inside this, so those keep
/// their real status code.
pub(crate) const KEEPALIVE_GRACE: std::time::Duration = std::time::Duration::from_secs(20);

/// Gap between keep-alive bytes once padding has started. Comfortably
/// under every proxy timeout worth caring about.
const KEEPALIVE_TICK: std::time::Duration = std::time::Duration::from_secs(10);

/// Run `work` and answer with its JSON, keeping the connection alive if it
/// takes a while.
///
/// A non-streaming completion sends nothing at all until the whole
/// generation is done. Behind a proxy with an origin-response timeout that
/// is indistinguishable from a dead origin: measured live 2026-09-21, a
/// phone request through Cloudflare was cut at exactly 125.0 s having
/// received 0 bytes, and the app showed HTTP 524. The generation was fine
/// and still running.
///
/// So a slow answer starts its body immediately and drips whitespace until
/// the real object is ready. JSON ignores whitespace before a value, so the
/// response is still exactly one object and every parser accepts it
/// unchanged.
///
/// **The trade-off, stated because it is real:** the status code goes out
/// with the headers, so an answer that takes longer than
/// [`KEEPALIVE_GRACE`] is committed to 200 before its outcome is known. A
/// generation that then fails answers 200 with an error *body* rather than
/// a 5xx. Racing the grace period first is what keeps that narrow — every
/// fast failure still gets its proper status, and only a request already
/// past 20 seconds of real work can land in it.
pub(crate) async fn keepalive_json<F>(work: F) -> Response
where
    // `'static` because the padded path hands the result channel to a
    // response body, which outlives this call.
    F: Future<Output = (u16, serde_json::Value)> + Send + 'static,
{
    // The work runs on its OWN task, and that is the whole trick. Selecting
    // directly on the future would put the timer and the generation on one
    // task, where anything that blocks rather than awaits — `plan_fit` runs
    // under `block_in_place`, and a cold load is minutes of it — starves the
    // timer, no padding goes out, and the proxy cuts the connection anyway.
    //
    // Measured 2026-09-21: with the work inline, a warm request padded
    // correctly (first byte at 20.7 s) while a cold one sent nothing and
    // died at Cloudflare's 125 s. Same code, opposite outcomes, decided
    // entirely by whether the generation happened to yield.
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let _ = tx.send(work.await);
    });

    tokio::select! {
        res = &mut rx => {
            let (status, body) = res.unwrap_or_else(|_| (500, crate::worker_vanished()));
            json_response(status, &body)
        }
        () = tokio::time::sleep(KEEPALIVE_GRACE) => {
            let stream = async_stream::stream! {
                loop {
                    tokio::select! {
                        res = &mut rx => {
                            let (_status, body) =
                                res.unwrap_or_else(|_| (500, crate::worker_vanished()));
                            yield Ok::<String, Infallible>(body.to_string());
                            break;
                        }
                        () = tokio::time::sleep(KEEPALIVE_TICK) => {
                            // A space: legal JSON leading whitespace, and one
                            // byte is enough to prove the origin is alive.
                            yield Ok(" ".to_string());
                        }
                    }
                }
            };
            (
                [(header::CONTENT_TYPE, "application/json")],
                axum::body::Body::from_stream(stream),
            )
                .into_response()
        }
    }
}

/// The body for a worker that ended without sending its result — the task
/// panicked outside what `recovery::contain` guards, or the runtime dropped
/// it. Should be unreachable; said plainly rather than answering with an
/// empty object.
fn worker_vanished() -> serde_json::Value {
    json!({"error": "the server's worker for this request ended without a result — this is a \
                     mummu-serve bug, not your request; try again"})
}

/// Parse a JSON body, or hand back the 400 response to return as-is. Keeps
/// the sync server's error wire format (`{"error": "bad json: …"}`).
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, Box<Response>> {
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(e) => {
            return Err(Box::new(json_response(
                400,
                &json!({"error": format!("body read: {e}")}),
            )));
        }
    };
    if text.len() > MAX_BODY_BYTES {
        return Err(Box::new(json_response(
            400,
            &json!({"error": "body too large"}),
        )));
    }
    serde_json::from_str::<T>(text).map_err(|e| {
        Box::new(json_response(
            400,
            &json!({"error": format!("bad json: {e}")}),
        ))
    })
}

async fn not_found() -> Response {
    json_response(404, &json!({"error": "not found"}))
}

async fn ui() -> Response {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("ui.html"),
    )
        .into_response()
}

/// `GET /favicon.ico` — a deliberate `204 No Content`: "there is no icon".
///
/// Every browser that opens `/` or `/logs` asks for this, and so does a
/// dashboard that shows a monitored site's icon. With no route each of those
/// was a `404` — and a 404 is LOUD on the log page, by design, because on a
/// public listener it is usually a scanner (see `logs::quiet_request`). A
/// tab opening is not a scanner, so it gets a success the log can fold away,
/// cached for a day so the browser stops asking on every load.
async fn favicon() -> Response {
    (
        axum::http::StatusCode::NO_CONTENT,
        [(header::CACHE_CONTROL, "public, max-age=86400")],
    )
        .into_response()
}

/// `GET /api/health` — 200 while the backend is healthy, **503 while it is
/// poisoned**: from a GPU failure in this process until the next load comes
/// up clean (see [`recovery::poisoned`], which also puts the status object in
/// its `error` phase, so the two cannot disagree). The body says why.
///
/// # What a 503 does in production
///
/// The container's healthcheck is `curl -sf …/api/health` every 30 s with 5
/// retries, so about two and a half minutes of 503s mark it `unhealthy` in
/// `docker ps` and on every dashboard that reads Docker's health — which is
/// the point: v0.3.1 answered `ok` through the whole incident. It restarts
/// NOTHING. Docker's `restart: unless-stopped` acts only on an exit, never on
/// health, and the autoheal container acts only on containers labelled
/// `autoheal`, which mummu is not. The recovery is mummu's own: the next
/// request reloads the model in-process, and if that fails too the process
/// exits and Docker restarts it (`recovery::supervised`). With no traffic a
/// poisoned server stays `unhealthy` until a request proves the card works —
/// the honest answer, since nothing has.
async fn health() -> Response {
    blocking(|| {
        let poisoned = recovery::poisoned();
        let body = health_json(recovery::current().as_ref(), poisoned);
        json_response(if poisoned { 503 } else { 200 }, &body)
    })
    .await
}

/// The health body, as a value.
///
/// Split out from the handler so a test can pin the shape without standing up
/// a runtime — and the shape is worth pinning, because clients read these
/// fields and a probe silently losing `status` would look like a healthy
/// server right up until something depended on it. The failure is passed in
/// rather than read, for the same reason, and so a test of one cannot see
/// another test's GPU failure.
fn health_json(error: Option<&recovery::BackendError>, poisoned: bool) -> serde_json::Value {
    let inv = mummu::backend::inventory();
    let gpus: Vec<_> = inv
        .gpus
        .iter()
        .map(|g| {
            json!({
                "name": g.name,
                "api": format!("{:?}", g.backend),
                "kind": format!("{:?}", g.device_type),
                "shader_f16": g.shader_f16,
            })
        })
        .collect();
    let (version, build) = status::build_json();
    json!({
        // "error" exactly when the handler answers 503: the backend failed in
        // this process and no load has come up clean since.
        "status": if poisoned { "error" } else { "ok" },
        // Why, when there is anything to say — including the previous
        // process's failure after a self-restart, which is shown here without
        // making THIS process unhealthy.
        "error": error.map(recovery::BackendError::to_json),
        "device": engine::device_label(),
        "gpus": gpus,
        "cpu_cores": inv.cpu.logical_cores,
        // "Is the new release deployed?" — asked, and unanswerable from here
        // until now. The version alone does not settle it (two builds of
        // 0.3.0 from either side of a fix carry the same string), so the
        // commit comes with it.
        "version": version,
        "build": build,
    })
}

async fn models() -> Response {
    blocking(|| {
        let root = models_root();
        let manager = ModelManager::new(root.clone());
        let list: Vec<_> = manager
            .catalog()
            .iter()
            .filter(|s| !matches!(s.architecture, mummu::registry::Architecture::MiniLm))
            .map(|s| {
                json!({
                    "name": s.name,
                    "repo": s.repo,
                    "architecture": format!("{:?}", s.architecture),
                    "format": match &s.format {
                        mummu::registry::WeightFormat::Safetensors => "safetensors",
                        mummu::registry::WeightFormat::Gguf { .. } => "gguf",
                    },
                    "disk_bytes_estimate": s.disk_bytes_estimate,
                    "installed": engine::is_installed(s, &root),
                })
            })
            .collect();
        json_response(
            200,
            &json!({"models": list, "device": engine::device_label()}),
        )
    })
    .await
}

async fn unload() -> Response {
    // Dropping a resident model frees VRAM/RAM — device work, not async work.
    // Report what actually happened: a generation holding the slot means the
    // model is still resident, and answering "unloaded" would be a lie the
    // caller acts on (it frees nothing, and the next request still hits the
    // old model).
    if blocking(engine::unload_all).await {
        json_response(200, &json!({"status": "unloaded"}))
    } else {
        json_response(
            409,
            &json!({"error": "a generation is in flight — the model stays resident until it finishes"}),
        )
    }
}

// ---------------------------------------------------------------------------
// SSE plumbing: the handler hands the worker an mpsc sender and streams what
// comes back as `data: {json}\n\n` frames. When the client goes away axum
// drops the stream, dropping the receiver; the worker's next send fails and
// it breaks off cooperatively — which is what stops a 27B generation from
// holding the model slot for minutes after the browser tab that asked for it
// is gone. The channel is unbounded because the producers are *synchronous*
// callbacks (the engine's delta hook, the downloader's progress hook) that
// must never park the thread they run on; the backlog is bounded in practice
// by `MAX_MAX_TOKENS` short strings.
// ---------------------------------------------------------------------------

fn sse_response(
    mut rx: mpsc::UnboundedReceiver<serde_json::Value>,
    inflight: Option<recovery::InFlight>,
) -> Response {
    let stream = async_stream::stream! {
        // Held until the last frame has been handed to the connection, so a
        // process exiting to restart the GPU backend waits for it.
        let _inflight = inflight;
        let mut ended = false;
        while let Some(frame) = rx.recv().await {
            ended |= is_final_frame(&frame);
            yield Ok::<Event, Infallible>(Event::default().data(frame.to_string()));
        }
        // The worker is gone without saying how it ended. Say so: a stream
        // that simply stops is the incident's empty bubble.
        if !ended {
            yield Ok(Event::default().data(ended_without_result().to_string()));
        }
    };
    // axum's `Sse` sets `text/event-stream` + `no-cache`; the third header
    // is the one that keeps nginx from buffering the stream into silence.
    ([("x-accel-buffering", "no")], Sse::new(stream)).into_response()
}

/// Is this the frame a stream ends on? Every native stream — chat and pull —
/// ends on exactly one `done` or one `error`.
fn is_final_frame(frame: &serde_json::Value) -> bool {
    matches!(
        frame.get("type").and_then(serde_json::Value::as_str),
        Some("done" | "error")
    )
}

/// The frame a stream ends on when its worker vanished without choosing one.
/// Should be unreachable — every worker ends in `done` or `error` — and it is
/// the reason "unreachable" can never again mean "an empty 200".
fn ended_without_result() -> serde_json::Value {
    json!({
        "type": "error",
        "error": "the server's worker for this request ended without a result — this is a \
                  mummu-serve bug, not your request; try again",
    })
}

/// Sends a stream's final frame exactly once: the one the worker chose, or,
/// if the worker never got to choose — it panicked outside the part
/// [`recovery::contain`] guards, or the runtime dropped it — `fallback`.
pub(crate) struct FinalFrame {
    tx: Option<mpsc::UnboundedSender<serde_json::Value>>,
    fallback: serde_json::Value,
}

impl FinalFrame {
    pub(crate) const fn new(
        tx: mpsc::UnboundedSender<serde_json::Value>,
        fallback: serde_json::Value,
    ) -> Self {
        Self {
            tx: Some(tx),
            fallback,
        }
    }

    pub(crate) fn send(mut self, frame: serde_json::Value) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(frame);
        }
    }
}

impl Drop for FinalFrame {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(std::mem::take(&mut self.fallback));
        }
    }
}

// ---------------------------------------------------------------------------
// POST /api/pull — download a catalog model, streaming progress frames.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PullRequest {
    model: String,
}

async fn pull(body: Bytes) -> Response {
    let parsed: PullRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    let root = models_root();
    let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();
    // The hub downloader is still synchronous, so it gets a blocking thread.
    tokio::task::spawn_blocking(move || {
        let manager = ModelManager::new(root);
        let mut last_pct: i64 = -1;
        let mut cancelled = false;
        let result = manager.install(&parsed.model, |p| {
            if cancelled {
                return; // client gone; drain the remaining callbacks quietly
            }
            let pct = p.total_bytes.map(|t| {
                trunc_i64((f64_from_u64(p.received_bytes) / f64_from_u64(t.max(1))) * 100.0)
            });
            // Throttle: one frame per whole percent (or per 64 MiB when the
            // server didn't announce a total).
            let tick = pct.unwrap_or_else(|| (p.received_bytes >> 26).cast_signed());
            if tick == last_pct {
                return;
            }
            last_pct = tick;
            let frame = json!({
                "type": "progress",
                "file": p.file,
                "received_bytes": p.received_bytes,
                "total_bytes": p.total_bytes,
                "percent": pct,
            });
            if tx.send(frame).is_err() {
                // The hub downloader has no cancel hook; remember the drop so
                // we at least stop building frames. The download itself runs
                // to completion, which also leaves the cache warm.
                cancelled = true;
            }
        });
        let done = match result {
            Ok(dir) => json!({"type": "done", "dir": dir.display().to_string()}),
            Err(e) => json!({"type": "error", "error": e}),
        };
        let _ = tx.send(done);
    });
    sse_response(rx, None)
}

// ---------------------------------------------------------------------------
// GET /api/chat/ws — the same stream over a WebSocket.

/// Chat over a WebSocket, carrying exactly the frames [`chat`] sends over SSE.
///
/// This exists because of a proxy limit, not a protocol preference. Behind
/// Cloudflare, an HTTP response that produces no bytes for 100 seconds is cut
/// with a 524 — and a cold `/api/chat` produces none for **seven minutes**
/// while a 27B is read from disk and placed across devices. SSE does not help:
/// the clock runs from the request, and there is nothing to stream yet.
/// Cloudflare does not apply that timeout to `WebSockets`.
///
/// The upgrade alone is not the fix, though. An idle WebSocket is still
/// reaped, so this **heartbeats while the model loads** — without the ping
/// below the connection dies at the same place, just with a different error.
async fn chat_ws(upgrade: axum::extract::ws::WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(|socket| async move {
        if let Err(e) = drive_chat_ws(socket, start_chat).await {
            eprintln!("[mummu-serve] chat ws: {e}");
        }
    })
}

/// Drive one chat over `socket`. `start` is [`start_chat`] in production; a
/// test hands in one whose generation fails the way production's did, and
/// reads the frames off a real socket.
async fn drive_chat_ws(
    mut socket: axum::extract::ws::WebSocket,
    start: fn(&ChatRequest) -> Result<ChatStream, Rejection>,
) -> Result<(), String> {
    use axum::extract::ws::Message;
    // `close` is `SinkExt::close`, not an inherent method on WebSocket.
    use futures::SinkExt;

    /// Well inside the ~100 s a proxy will tolerate, and cheap enough that
    /// sending it through a seven-minute load costs nothing.
    const HEARTBEAT: std::time::Duration = std::time::Duration::from_secs(15);

    // The request arrives as the first text frame — same JSON body the POST
    // endpoint takes, so a client can switch transports without changing it.
    let request = loop {
        match socket.recv().await {
            Some(Ok(Message::Text(text))) => break text,
            // A client may ping before sending; keep waiting for the body.
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
            Some(Ok(Message::Close(_))) | None => return Ok(()),
            Some(Ok(_)) => return Err("expected a text frame carrying the request".into()),
            Some(Err(e)) => return Err(e.to_string()),
        }
    };
    let started = serde_json::from_str::<ChatRequest>(&request)
        .map_err(|e| Rejection {
            status: 400,
            error: format!("bad json: {e}"),
        })
        .and_then(|parsed| start(&parsed));
    let mut chat = match started {
        Ok(chat) => chat,
        Err(rejection) => {
            // Report the rejection in-band and close cleanly: a WebSocket
            // client cannot read the HTTP status of a request it never made,
            // so the frame carries the status AND the reason.
            let frame = rejection.frame();
            let _ = socket.send(Message::Text(frame.to_string().into())).await;
            let _ = socket.close().await;
            return Ok(());
        }
    };
    let rx = &mut chat.rx;

    let mut beat = tokio::time::interval(HEARTBEAT);
    // Missed ticks are worthless: if we were not polled for a minute, sending
    // four pings at once proves nothing to a proxy that already timed us out.
    beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    beat.tick().await; // the first tick is immediate; skip it
    loop {
        tokio::select! {
            event = rx.recv() => if let Some(frame) = event {
                let done = is_final_frame(&frame);
                if socket.send(Message::Text(frame.to_string().into())).await.is_err() {
                    return Ok(()); // client gone; the generation task sees the closed channel
                }
                if done {
                    let _ = socket.close().await;
                    return Ok(());
                }
            } else {
                // The worker is gone without a final frame (see
                // `ended_without_result`): never close on silence.
                let frame = ended_without_result();
                let _ = socket.send(Message::Text(frame.to_string().into())).await;
                let _ = socket.close().await;
                return Ok(());
            },
            _ = beat.tick() => {
                // Keeps the proxy from reaping a connection that is waiting on
                // a model load rather than idling.
                if socket.send(Message::Ping(Vec::new().into())).await.is_err() {
                    return Ok(());
                }
            }
            // A client that closes mid-generation lands here through `recv`
            // returning an error on the next send, which is handled above.
        }
    }
}

// ---------------------------------------------------------------------------
// POST /api/chat — stream a completion as SSE `delta` frames.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    /// Absent or `null` on an assistant turn that only made calls, which is
    /// how ollama's own clients replay one.
    #[serde(default, deserialize_with = "null_as_empty")]
    pub(crate) content: String,
    /// Base64 image payloads, ollama's per-message spelling. `OpenAI` puts
    /// them in `content` parts instead; both land here before planning.
    #[serde(default)]
    pub(crate) images: Vec<String>,
    /// The calls an assistant turn made, when the client is replaying a
    /// tool loop back to us. Dropping these leaves an EMPTY assistant turn
    /// in the history followed by a tool result the model never asked for.
    #[serde(default, deserialize_with = "replayed_calls")]
    pub(crate) tool_calls: Vec<mummu::chat::ToolCall>,
}

fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

/// A replayed call in either spelling: ollama's `{"function": {"name",
/// "arguments"}}`, or the flat `{"name", "arguments"}` the native API takes.
#[derive(Deserialize)]
#[serde(untagged)]
enum ReplayedCall {
    Wrapped { function: mummu::chat::ToolCall },
    Flat(mummu::chat::ToolCall),
}

fn replayed_calls<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Vec<mummu::chat::ToolCall>, D::Error> {
    let calls = Option::<Vec<ReplayedCall>>::deserialize(d)?.unwrap_or_default();
    Ok(calls
        .into_iter()
        .map(|c| match c {
            ReplayedCall::Wrapped { function } | ReplayedCall::Flat(function) => function,
        })
        .collect())
}

/// An output grammar the decoder must obey, asked for by a request.
///
/// Both compatibility surfaces spell the same thing differently — ollama's
/// `"format": "json"` and `OpenAI`'s `"response_format": {"type":
/// "json_object"}` — so they translate into this one type and share the
/// machinery in `mummu::constrain`.
///
/// Before this existed, `format` was parsed by nobody: serde dropped the
/// unknown field and the request generated ordinary prose. A client asking
/// for JSON got prose, its parse failed, and nothing anywhere said why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    /// A single JSON object or array, enforced token by token.
    Json,
}

#[derive(Deserialize, Default)]
struct ChatOptions {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    max_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    options: ChatOptions,
    /// Also accepted at the top level, OpenAI-style, because clients keep
    /// putting it there and serde ignores unknown fields — a top-level cap
    /// used to be silently dropped and the request generated up to
    /// [`DEFAULT_MAX_TOKENS`]. `options.max_tokens` wins when both are set.
    max_tokens: Option<usize>,
    /// Profile this generation: scope wall-times are collected during the
    /// run and the flame graph is published at `GET /api/profile` when it
    /// completes. The profiler is process-global, so profile one request at
    /// a time; `MUMMU_PROFILE` in the environment forces this on for every
    /// request.
    #[serde(default)]
    profile: bool,
}

impl ChatRequest {
    /// The effective decode cap: `options.max_tokens`, then the top-level
    /// `max_tokens`, then [`DEFAULT_MAX_TOKENS`]; clamped to the hard ceiling.
    fn max_tokens(&self) -> usize {
        self.options
            .max_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS)
            .clamp(1, MAX_MAX_TOKENS)
    }
}

/// A request's messages as `arch`'s renderer takes them.
///
/// The family matters for one kind of turn: an assistant turn replaying the
/// calls the model made is written back in the family's own call syntax
/// (see `engine::CallSyntax::replay`) — an LFM model shown Hermes JSON for
/// its own request is shown a turn it never wrote.
pub(crate) fn to_turns(
    messages: &[ChatMessage],
    arch: mummu::registry::Architecture,
) -> Result<Vec<Turn>, String> {
    if messages.is_empty() {
        return Err("messages must be non-empty".into());
    }
    if messages.len() > MAX_TURNS {
        return Err(format!("more than {MAX_TURNS} messages"));
    }
    // A family with no call syntax of its own is never offered tools; a
    // history that carries calls anyway gets them in Hermes, which at
    // least reads as what it is.
    let replay = engine::tool_calls(arch).map_or(
        Turn::assistant_tool_calls as fn(&[mummu::chat::ToolCall]) -> Turn,
        |c| c.replay,
    );
    let turns: Vec<Turn> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| match m.role.as_str() {
            "system" => Ok(Turn::system(m.content.clone())),
            "user" => Ok(Turn::user(m.content.clone())),
            // An assistant turn that made calls keeps them, so the model
            // sees its own request and not a blank turn before the answer
            // comes back.
            "assistant" if !m.tool_calls.is_empty() => replayable(&m.tool_calls)
                .map(|calls| replay(&calls))
                .map_err(|e| format!("message {i}: {e}")),
            "assistant" => Ok(Turn::assistant(m.content.clone())),
            // The second half of a tool loop: the client ran the function
            // and is handing back its result. The family renderer decides
            // where that goes — Hermes puts it in a `<tool_response>` block
            // of a user turn, LFM gives it a turn of its own.
            "tool" | "function" => Ok(Turn::tool_response(m.content.clone())),
            other => Err(format!("unsupported role {other:?}")),
        })
        .collect::<Result<_, _>>()?;
    if turns.last().map(|t| t.role) == Some(Role::Assistant) {
        return Err("the last message must not be an assistant turn".into());
    }
    Ok(turns)
}

fn sampler_options(o: &ChatOptions) -> Result<SamplerOptions, String> {
    let defaults = SamplerOptions::default();
    let temperature = o.temperature.unwrap_or(0.7);
    let top_p = o.top_p.unwrap_or(0.9);
    let top_k = o.top_k.unwrap_or(defaults.top_k);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(format!(
            "temperature must be finite and >= 0, got {temperature}"
        ));
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(format!("top_p must be in (0, 1], got {top_p}"));
    }
    if top_k == 0 {
        return Err("top_k must be >= 1".into());
    }
    // A fresh seed per request unless pinned — reproducibility on demand.
    let seed = o.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, nanos)
    });
    Ok(SamplerOptions {
        temperature,
        top_p,
        top_k,
        seed,
    })
}

/// The most recent profiled generation, as (svg, folded stacks). One slot —
/// each profiled request replaces the last — served by `GET /api/profile`.
static LAST_PROFILE: std::sync::Mutex<Option<(String, String)>> = std::sync::Mutex::new(None);

/// Disables the process-global profiler on drop — the panic backstop for a
/// profiled generation (see the comment at its use).
struct DisableProfilerOnDrop;

impl Drop for DisableProfilerOnDrop {
    fn drop(&mut self) {
        mummu::prof::set_enabled(false);
    }
}

/// Fold what the profiler collected and render the flame graph.
fn publish_profile() {
    let folded = mummu::prof::folded();
    if folded.is_empty() {
        eprintln!("[mummu-serve] profile: nothing collected (no instrumented code ran?)");
        return;
    }
    match mummu::prof::flamegraph_svg(&folded) {
        Ok(svg) => {
            eprintln!(
                "[mummu-serve] profile: {} stacks — GET /api/profile for the flame graph",
                folded.lines().count()
            );
            *LAST_PROFILE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((svg, folded));
        }
        Err(e) => eprintln!("[mummu-serve] profile: flamegraph failed: {e}"),
    }
}

/// GET /api/profile — the last profiled generation's flame graph, as SVG a
/// browser renders directly (frames are zoomable; widths are self time).
async fn profile_svg() -> Response {
    let last = LAST_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    match last {
        Some((svg, _)) => Response::builder()
            .header("content-type", "image/svg+xml")
            .body(svg.into())
            .expect("static response"),
        None => json_response(
            404,
            &json!({"error": "no profiled generation yet — POST /api/chat with \"profile\": true, then retry"}),
        ),
    }
}

/// GET /api/profile/folded — the same data as folded stacks (`path (Nx) µs`
/// per line), for tooling or a quick sort in a terminal.
async fn profile_folded() -> Response {
    let last = LAST_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    match last {
        Some((_, folded)) => Response::builder()
            .header("content-type", "text/plain; charset=utf-8")
            .body(folded.into())
            .expect("static response"),
        None => json_response(
            404,
            &json!({"error": "no profiled generation yet — POST /api/chat with \"profile\": true, then retry"}),
        ),
    }
}

async fn chat(body: Bytes) -> Response {
    let parsed: ChatRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    match start_chat(&parsed) {
        Ok(chat) => sse_response(chat.rx, Some(chat.inflight)),
        Err(rejection) => rejection.response(),
    }
}

/// A chat that has started: its frames, and the claim that its response is
/// still open (see [`recovery::InFlight`]) — carried together so that
/// whichever transport drains the frames also holds the claim until it is
/// done writing them.
struct ChatStream {
    rx: mpsc::UnboundedReceiver<serde_json::Value>,
    inflight: recovery::InFlight,
}

/// A chat refused before it started. A POST client gets the status; a
/// WebSocket client, which never sees one, gets both in a frame.
struct Rejection {
    status: u16,
    error: String,
}

impl Rejection {
    fn response(&self) -> Response {
        json_response(self.status, &json!({"error": self.error}))
    }

    fn frame(&self) -> serde_json::Value {
        json!({"type": "error", "status": self.status, "error": self.error})
    }
}

/// A client's replayed calls, checked against what the renderers assume —
/// they assert it, and a replayed call is whatever the client sent.
///
/// Arguments must be an object, as ollama's own API types them; one that
/// arrives JSON-encoded in a string, `OpenAI`'s spelling, is decoded. `null`
/// is a call with no arguments.
fn replayable(calls: &[mummu::chat::ToolCall]) -> Result<Vec<mummu::chat::ToolCall>, String> {
    use mummu::chat::{MAX_TOOL_CALLS, MAX_VALUE_DEPTH};
    use serde_json::Value;
    /// Every value within the Pythonic renderer's nesting bound, counted
    /// the way it counts: an argument's value is depth 0.
    fn shallow(v: &Value, depth: usize) -> bool {
        depth <= MAX_VALUE_DEPTH
            && match v {
                Value::Array(items) => items.iter().all(|v| shallow(v, depth + 1)),
                Value::Object(map) => map.values().all(|v| shallow(v, depth + 1)),
                _ => true,
            }
    }
    if calls.len() > MAX_TOOL_CALLS {
        return Err(format!("more than {MAX_TOOL_CALLS} tool calls in one turn"));
    }
    calls
        .iter()
        .map(|c| {
            if c.name.trim().is_empty() {
                return Err("a replayed tool call has no name".to_string());
            }
            let arguments = match &c.arguments {
                Value::String(s) => serde_json::from_str(s).map_err(|e| {
                    format!(
                        "tool call {:?}: arguments are a string that is not JSON: {e}",
                        c.name
                    )
                })?,
                other => other.clone(),
            };
            match &arguments {
                Value::Object(map) if map.values().all(|v| shallow(v, 0)) => {}
                Value::Object(_) => {
                    return Err(format!(
                        "tool call {:?}: arguments nest deeper than {MAX_VALUE_DEPTH} levels",
                        c.name
                    ));
                }
                Value::Null => {}
                _ => {
                    return Err(format!(
                        "tool call {:?}: arguments must be an object",
                        c.name
                    ));
                }
            }
            Ok(mummu::chat::ToolCall {
                name: c.name.clone(),
                arguments,
            })
        })
        .collect()
}

/// Validate a chat request and start generating, returning the stream of
/// events. Shared by the SSE and WebSocket endpoints so the two cannot drift.
///
/// The generation outlives the caller: it runs as its own task and whoever
/// holds the receiver drains it.
fn start_chat(parsed: &ChatRequest) -> Result<ChatStream, Rejection> {
    let reject = |status: u16, error: String| Rejection { status, error };
    // The process is exiting to restart the GPU backend (see `recovery`).
    if recovery::restarting() {
        return Err(reject(503, recovery::restarting_message().to_owned()));
    }
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let Some(spec) = manager
        .catalog()
        .iter()
        .find(|s| s.name == parsed.model)
        .cloned()
    else {
        return Err(reject(404, format!("unknown model {:?}", parsed.model)));
    };
    let turns = to_turns(&parsed.messages, spec.architecture).map_err(|e| reject(400, e))?;
    let opts = sampler_options(&parsed.options).map_err(|e| reject(400, e))?;
    let max_tokens = parsed.max_tokens();
    if !engine::is_installed(&spec, &root) {
        return Err(reject(
            409,
            format!("{} is not installed — pull it first", spec.name),
        ));
    }

    let profile = parsed.profile || std::env::var("MUMMU_PROFILE").is_ok();
    let name = spec.name.clone();
    Ok(spawn_chat(name, profile, move |sink| async move {
        let req = engine::GenerationRequest {
            spec: &spec,
            models_root: &root,
            turns: &turns,
            opts: &opts,
            max_tokens,
            format: None,
            think: false,
            images: Vec::new(),
            tools: Vec::new(),
        };
        engine::run_chat(&req, |delta| sink.delta(delta)).await
    }))
}

/// Where a generation's text goes: one `delta` frame per piece.
struct DeltaSink(mpsc::UnboundedSender<serde_json::Value>);

impl DeltaSink {
    fn delta(&self, text: &str) -> ControlFlow<()> {
        if self.0.send(json!({"type": "delta", "text": text})).is_err() {
            return ControlFlow::Break(()); // client gone: stop decoding
        }
        ControlFlow::Continue(())
    }
}

/// Run one generation as its own task and hand back its frames — the ONE
/// place a native chat's outcome becomes a frame, for SSE and WebSocket
/// alike.
///
/// The stream it returns ends in exactly one `done` or `error` frame, by
/// construction rather than by care:
///
/// * the generation runs under [`recovery::contain`], so a panic in it — the
///   incident's `bytes: host access failed` — comes back as an error, and a
///   GPU failure is acted on (the model is dropped; see `recovery`);
/// * a [`FinalFrame`] guard sends an error if the task ends any other way.
///
/// So the 200-with-an-empty-stream this line of work began with cannot be
/// produced here. `run` is the generation; production passes
/// `engine::run_chat`, and a test passes one that fails the way production
/// did.
fn spawn_chat<F, Fut>(model: String, profile: bool, run: F) -> ChatStream
where
    F: FnOnce(DeltaSink) -> Fut + Send + 'static,
    Fut: Future<Output = Result<engine::ChatResult, recovery::ChatError>> + Send,
{
    let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();
    let inflight = recovery::InFlight::enter();
    // A normal async task: the generation is mostly awaits. The one part
    // that genuinely blocks — the model load — declares itself as blocking
    // where it happens, in `mummu::cache`, rather than this pushing the whole
    // future onto a blocking thread.
    tokio::spawn(async move {
        let last = FinalFrame::new(tx.clone(), ended_without_result());
        // Panic-safe by construction: the guard disables the profiler on
        // Drop even when the generation unwinds. Without it, a panic
        // anywhere in a profiled run — and generation panics have happened
        // (OOM, unreachable!) — would be swallowed by tokio::spawn with the
        // process-global flag left on, so every later request from every
        // client would silently pay a String join and a global mutex lock
        // per scope while the stale graph kept serving. Found in review,
        // before production found it. Holding this across the await is fine:
        // its Drop flips an atomic and never touches a thread-local stack.
        let profile_session = profile.then(|| {
            mummu::prof::reset();
            mummu::prof::set_enabled(true);
            DisableProfilerOnDrop
        });
        let started = std::time::Instant::now();
        let result = recovery::contain(&model, run(DeltaSink(tx))).await;
        if let Some(session) = profile_session {
            drop(session); // stop collecting before folding the report
            publish_profile();
        }
        let frame = match result {
            Ok(r) => {
                let secs = (f64_from_u64(r.elapsed_ms) / 1000.0).max(1e-3);
                json!({
                    "type": "done",
                    "text": r.text,
                    "tokens": r.tokens,
                    "device": r.device,
                    "elapsed_ms": r.elapsed_ms,
                    "tokens_per_second": (f64_from_usize(r.tokens) / secs * 10.0).round() / 10.0,
                })
            }
            Err(e) => {
                eprintln!(
                    "[mummu-serve] chat {model}: {e} (after {} ms)",
                    started.elapsed().as_millis()
                );
                e.frame()
            }
        };
        last.send(frame);
    });
    ChatStream { rx, inflight }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_request(body: &str) -> ChatRequest {
        serde_json::from_str(body).expect("valid chat request")
    }

    /// The bug this guards against: a top-level `max_tokens` was an unknown
    /// field serde silently dropped, so a request asking for 12 tokens
    /// generated up to [`DEFAULT_MAX_TOKENS`]. Both endpoints share
    /// `ChatRequest`, so this covers the SSE and WebSocket paths alike.
    #[test]
    fn top_level_max_tokens_is_honored() {
        let parsed = chat_request(
            r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 12}"#,
        );
        assert_eq!(parsed.max_tokens(), 12);
    }

    #[test]
    fn options_max_tokens_still_works_and_wins() {
        let parsed =
            chat_request(r#"{"model": "m", "messages": [], "options": {"max_tokens": 7}}"#);
        assert_eq!(parsed.max_tokens(), 7);

        let parsed = chat_request(
            r#"{"model": "m", "messages": [], "options": {"max_tokens": 7}, "max_tokens": 12}"#,
        );
        assert_eq!(parsed.max_tokens(), 7);
    }

    #[test]
    fn absent_max_tokens_falls_back_to_default() {
        let parsed = chat_request(r#"{"model": "m", "messages": []}"#);
        assert_eq!(parsed.max_tokens(), DEFAULT_MAX_TOKENS);
    }

    /// `GET /api/health` is the one endpoint other things are wired to. Its
    /// existing fields are a contract — a client reads `status`, a dashboard
    /// reads `device`, the UI badge reads `gpus` — and this release ADDS to
    /// it rather than reshaping it.
    #[test]
    fn health_keeps_its_fields_and_now_names_the_build() {
        let h = health_json(None, false);
        let o = h.as_object().expect("health is an object");
        for key in ["status", "device", "gpus", "cpu_cores"] {
            assert!(o.contains_key(key), "health lost its {key} field");
        }
        assert_eq!(h["status"], json!("ok"));
        assert_eq!(h["error"], json!(null), "no failure, nothing to say");
        assert!(h["gpus"].is_array());
        assert!(h["cpu_cores"].is_u64());
        // The new half: which release, and which commit of it.
        assert_eq!(h["version"], json!(status::VERSION));
        assert_eq!(
            h["version"],
            json!("0.5.0"),
            "this branch ships as v0.5.0; the workspace version is what says so"
        );
        let build = h["build"].as_str().expect("build is a string");
        assert!(
            !build.is_empty(),
            "a build with no git to ask reads \"unknown\", never empty"
        );
    }

    #[test]
    fn max_tokens_is_clamped_to_the_hard_ceiling() {
        let parsed = chat_request(r#"{"model": "m", "messages": [], "max_tokens": 0}"#);
        assert_eq!(parsed.max_tokens(), 1);

        let parsed = chat_request(r#"{"model": "m", "messages": [], "max_tokens": 999999}"#);
        assert_eq!(parsed.max_tokens(), MAX_MAX_TOKENS);
    }

    // -- v0.3.2: a GPU failure is told to every client, never swallowed ------

    /// The read that failed every chat after the 2026-09-18 load, verbatim
    /// up to where the log cut it.
    const INVALID_READ: &str = "bytes: host access failed: Read(\"The server is in an invalid \
                                state\\nCaused by:\\n  An IO error happened\\nCaused by:\\n  \
                                couldn't find resource for that handle: Memory location was \
                                never initialized\")";

    /// The device-thread panic of the incident's load, verbatim.
    const LOAD_OOM: &str = "failed to reserve 22020096 bytes of device memory: out of device \
                            memory allocating 261319680 bytes";

    /// A generation that fails the way production's did: a panic on the
    /// request's own worker, from the first read. Called from inside the
    /// generation's own future, so the panic lands where production's did.
    fn fails_like_production() -> Result<engine::ChatResult, recovery::ChatError> {
        panic!("{INVALID_READ}")
    }

    /// A generation that panics for a reason that is NOT the GPU's.
    fn fails_with_an_ordinary_bug() -> Result<engine::ChatResult, recovery::ChatError> {
        panic!("index out of bounds: the len is 3 but the index is 7")
    }

    /// The incident's load-time failure: cubecl's device thread panics and
    /// catches its own panic, so nothing but the hook ever sees it.
    fn device_thread_oom() {
        std::thread::Builder::new()
            .name("DSD-0-0".to_owned())
            .spawn(|| {
                let _ = std::panic::catch_unwind(|| panic!("{LOAD_OOM}"));
            })
            .expect("spawn")
            .join()
            .expect("the device thread survives its panic");
    }

    /// An SSE body as the frames a client parses out of it.
    async fn sse_frames(response: Response) -> Vec<serde_json::Value> {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        String::from_utf8_lossy(&body)
            .split("\n\n")
            .filter_map(|f| f.strip_prefix("data: "))
            .map(|d| serde_json::from_str(d).expect("each frame is JSON"))
            .collect()
    }

    /// POST /api/chat, the incident, on the transport it happened on: the
    /// client got a 200 carrying NOTHING. It must get exactly one final
    /// frame, an error, saying what failed and what happens next.
    #[tokio::test]
    async fn an_sse_chat_that_hits_the_gpu_failure_ends_in_an_error_frame() {
        let _serial = progress_serial().await;
        recovery::reset_for_tests();
        recovery::install_panic_hook();

        let chat = spawn_chat("qwen3.8-27b-ud-q4ks".into(), false, |_| async {
            fails_like_production()
        });
        let frames = sse_frames(sse_response(chat.rx, Some(chat.inflight))).await;
        assert!(!frames.is_empty(), "the incident's empty 200");
        assert_eq!(
            frames.iter().filter(|f| is_final_frame(f)).count(),
            1,
            "exactly one final frame: {frames:?}"
        );
        let last = frames.last().expect("a frame");
        assert_eq!(last["type"], json!("error"), "{last}");
        assert_eq!(last["recovery"], json!("reload"), "{last}");
        let message = last["error"].as_str().expect("a message");
        assert!(
            message.contains("GPU backend failed") && message.contains("try again"),
            "{message}"
        );
        recovery::reset_for_tests();
    }

    /// A panic that is not the GPU's still reaches the client — as an error
    /// that promises no recovery, because none is happening.
    #[tokio::test]
    async fn an_ordinary_panic_is_an_error_frame_that_promises_no_recovery() {
        let _serial = progress_serial().await;
        recovery::reset_for_tests();
        recovery::install_panic_hook();

        let chat = spawn_chat("m".into(), false, |_| async {
            fails_with_an_ordinary_bug()
        });
        let frames = sse_frames(sse_response(chat.rx, Some(chat.inflight))).await;
        let last = frames.last().expect("a frame");
        assert_eq!(last["type"], json!("error"), "{last}");
        assert!(last.get("recovery").is_none(), "{last}");
        assert!(!recovery::poisoned(), "a bug is not a poisoned GPU");
        recovery::reset_for_tests();
    }

    /// The belt under the braces: whatever becomes of the worker, the stream
    /// a client reads ends on a final frame, never on silence.
    #[tokio::test]
    async fn a_stream_whose_worker_vanished_still_ends_in_an_error() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(json!({"type": "delta", "text": "half an ans"}))
            .expect("open");
        drop(tx);
        let frames = sse_frames(sse_response(rx, None)).await;
        assert_eq!(frames.len(), 2, "{frames:?}");
        assert_eq!(frames[1]["type"], json!("error"));
    }

    /// The worker's own guard: the final frame it chose, or the fallback if
    /// it never got to choose — and never both.
    #[test]
    fn a_worker_sends_exactly_one_final_frame_however_it_ends() {
        let fallback = json!({"type": "error", "error": "fallback"});
        let (tx, mut rx) = mpsc::unbounded_channel();
        drop(FinalFrame::new(tx, fallback.clone()));
        assert_eq!(rx.try_recv().expect("sent on drop"), fallback);

        let (tx, mut rx) = mpsc::unbounded_channel();
        FinalFrame::new(tx, fallback).send(json!({"type": "done"}));
        assert_eq!(rx.try_recv().expect("sent")["type"], json!("done"));
        assert!(rx.try_recv().is_err(), "exactly one final frame");
    }

    fn restarting_start(_: &ChatRequest) -> Result<ChatStream, Rejection> {
        Err(Rejection {
            status: 503,
            error: recovery::restarting_message().to_owned(),
        })
    }

    /// Every text frame a real WebSocket client receives for one chat.
    async fn ws_frames(
        start: fn(&ChatRequest) -> Result<ChatStream, Rejection>,
    ) -> Vec<serde_json::Value> {
        use futures::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;
        let app = Router::new().route(
            "/ws",
            get(move |up: axum::extract::ws::WebSocketUpgrade| async move {
                up.on_upgrade(move |socket| async move {
                    let _ = drive_chat_ws(socket, start).await;
                })
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .expect("connect");
        ws.send(Message::Text(
            r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}]}"#.into(),
        ))
        .await
        .expect("send");
        let mut frames = Vec::new();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                frames.push(serde_json::from_str(text.as_str()).expect("JSON frame"));
            }
        }
        frames
    }

    /// The UI's transport. A GPU failure arrives as an error frame on the
    /// socket before it closes — never as a socket that simply closes, which
    /// is what the chat page drew as an empty bubble.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_websocket_chat_that_hits_the_gpu_failure_gets_an_error_frame() {
        let _serial = progress_serial().await;
        recovery::reset_for_tests();
        recovery::install_panic_hook();

        // A start whose generation fails the way production's did.
        let frames = ws_frames(|_| {
            Ok(spawn_chat("qwen3.8-27b-ud-q4ks".into(), false, |_| async {
                fails_like_production()
            }))
        })
        .await;
        let last = frames.last().expect("at least one frame before the close");
        assert_eq!(last["type"], json!("error"), "{frames:?}");
        assert_eq!(last["recovery"], json!("reload"), "{last}");
        assert!(
            last["error"]
                .as_str()
                .is_some_and(|m| m.contains("GPU backend failed")),
            "{last}"
        );

        // Refused while restarting: the status AND the reason, in-band.
        let frames = ws_frames(restarting_start).await;
        assert_eq!(frames.len(), 1, "{frames:?}");
        assert_eq!(frames[0]["status"], json!(503));
        assert_eq!(frames[0]["error"], json!(recovery::restarting_message()));
        recovery::reset_for_tests();
    }

    /// `/api/health` said `ok` through the whole incident. While the backend
    /// is poisoned it must answer non-2xx — Docker's `curl -sf` fails on it —
    /// and say why; and a clean load must bring it back.
    #[tokio::test]
    async fn health_is_503_and_says_why_while_the_gpu_is_poisoned() {
        let _serial = progress_serial().await;
        recovery::reset_for_tests();
        recovery::install_panic_hook();
        assert_eq!(health().await.status(), 200);

        device_thread_oom();
        let response = health().await;
        assert_eq!(response.status(), 503, "a poisoned GPU answered healthy");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let h: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
        assert_eq!(h["status"], json!("error"));
        assert!(
            h["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("out of device memory")),
            "{h}"
        );
        for key in ["device", "gpus", "cpu_cores", "version", "build"] {
            assert!(h.get(key).is_some(), "a 503 still carries {key}");
        }

        // A clean load on the device that failed (the thread was DSD-0-0).
        recovery::load_succeeded(
            "qwen3.8-27b-ud-q4ks",
            &[recovery::DeviceKey::Cubecl {
                type_id: 0,
                index: 0,
            }],
        );
        assert_eq!(health().await.status(), 200, "a clean load clears it");
        recovery::reset_for_tests();
    }

    /// Fault injection must never reach the image: the Dockerfile's build
    /// names its features, and this one is not among them nor a default.
    #[test]
    fn the_docker_image_never_builds_fault_injection() {
        let dockerfile = include_str!("../Dockerfile");
        let build = dockerfile
            .lines()
            .find(|l| l.trim_start().starts_with("RUN cargo build"))
            .expect("the image builds with cargo");
        assert!(
            build.contains("--features cuda")
                && !build.contains("fault-injection")
                && !build.contains("--all-features"),
            "{build}"
        );
        assert!(!dockerfile.contains("fault-injection"));
        let manifest = include_str!("../Cargo.toml");
        let default = manifest
            .lines()
            .find(|l| l.starts_with("default ="))
            .expect("a default feature list");
        assert!(!default.contains("fault-injection"), "{default}");
    }

    /// POST to a router served on a real socket; the status line's code.
    async fn post_status(app: Router, path: &str, body: &str) -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).await.expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).await.expect("read");
        response
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .expect("a status line")
    }

    /// Without the feature there is no route to arm anything: the request is
    /// an unknown path like any other.
    #[cfg(not(feature = "fault-injection"))]
    #[tokio::test]
    async fn a_build_without_fault_injection_has_no_fault_route() {
        assert_eq!(
            post_status(router(), "/api/fault", r#"{"load_oom": 1}"#).await,
            404
        );
    }

    /// With it, the route arms what it is told to and nothing more.
    #[cfg(feature = "fault-injection")]
    #[tokio::test]
    async fn a_fault_injection_build_arms_faults_through_its_route() {
        let _serial = progress_serial().await;
        assert_eq!(
            post_status(
                router(),
                "/api/fault",
                r#"{"load_oom": 1, "read_invalid": 2}"#
            )
            .await,
            200
        );
        assert_eq!(
            (fault::armed().load_oom, fault::armed().read_invalid),
            (1, 2)
        );
        fault::arm(fault::Arm::default());
    }

    /// Both pages render the status object's error: the `error` phase, and
    /// the message in the error colour. Asserted against the shipped bytes,
    /// which have no test runner of their own.
    #[test]
    fn both_pages_render_a_gpu_failure_in_red() {
        for (name, html) in [("ui.html", UI_HTML), ("logs.html", LOGS_HTML)] {
            assert!(
                html.contains(".perr { color: var(--err);"),
                "{name} has no red failure line"
            );
            assert!(
                html.contains("note.textContent = st.error ? failureText(st.error) : \"\";"),
                "{name} ignores the status object's error"
            );
            assert!(
                html.contains("st.phase === \"error\""),
                "{name} has no error phase"
            );
            assert!(
                html.contains("function failureText(e) {"),
                "{name} does not share the failure line"
            );
        }
        assert!(
            LOGS_HTML.contains("label.classList.toggle(\"bad\", st.phase === \"error\");"),
            "the logs page's label must go red on the error phase"
        );
    }

    /// The source of one JavaScript function, from its `function` line to
    /// the first line that closes it at column zero.
    fn js_function<'a>(html: &'a str, signature: &str) -> &'a str {
        let start = html.find(signature).expect("the function is on the page");
        let end = html[start..].find("\n}\n").expect("the function ends") + start + 2;
        &html[start..end]
    }

    /// MINOR 8: the red label says what recovery is ACTUALLY doing — the
    /// status object's `error.recovery` — never a fixed promise of a reload
    /// that, while the process is exiting or its restart budget is spent, is
    /// not what happens. Every value the server can send has its own words,
    /// and both pages say them identically.
    #[test]
    fn both_pages_render_the_recovery_the_status_reports() {
        let words = js_function(UI_HTML, "function recoveryText(e) {");
        assert_eq!(
            words,
            js_function(LOGS_HTML, "function recoveryText(e) {"),
            "the two pages describe recovery differently"
        );
        for r in recovery::Recovery::ALL {
            assert!(
                words.contains(&format!("case \"{}\":", r.as_str())),
                "the pages have no words for recovery {:?}",
                r.as_str()
            );
        }
        for (name, html) in [("ui.html", UI_HTML), ("logs.html", LOGS_HTML)] {
            assert!(
                html.contains("`error — the GPU backend failed; ${recoveryText(st.error)}`"),
                "{name}'s error label ignores error.recovery"
            );
            for fixed in [
                "failed; the next request loads the model again\"",
                "failed; this request loads the model again\"",
            ] {
                assert!(!html.contains(fixed), "{name} still promises a reload");
            }
        }
    }

    /// The chat page never leaves an empty bubble: an error frame is shown
    /// (as text, never as markup off the wire), and a stream that ends with
    /// neither `done` nor `error` is reported as a dropped connection.
    #[test]
    fn the_chat_page_never_leaves_an_empty_bubble() {
        assert!(
            UI_HTML.contains("if (!ended) {"),
            "a stream that ended without done or error is drawn as nothing"
        );
        assert!(UI_HTML.contains("the connection closed before the reply finished"));
        assert!(
            UI_HTML.contains("err.textContent = `⚠ ${e.message}`;"),
            "the error is not rendered, or is rendered as markup"
        );
        assert!(
            !UI_HTML.contains("body.innerHTML += `<div class=\"err\">"),
            "a server message reached innerHTML"
        );
    }
}
