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
//! - `GET  /api/health`  device policy + adapter inventory
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

mod engine;
mod shim;

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
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};

pub use engine::device_label;

/// The single-page chat UI, exactly as `GET /` serves it. Exposed so a shell
/// that wants to embed the same bytes (the Tauri app's offline fallback) has
/// one source of truth instead of a copy that drifts.
pub const UI_HTML: &str = include_str!("ui.html");

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
    std::env::var_os("MUMMU_MODELS_DIR").map_or_else(|| PathBuf::from("models"), PathBuf::from)
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
/// this environment actually has a usable GPU adapter. Probing adapters is
/// blocking work; callers on an async thread should hand this to
/// `spawn_blocking`.
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
    Router::new()
        .route("/", get(ui))
        .route("/index.html", get(ui))
        .route("/api/health", get(health))
        .route("/api/models", get(models))
        .route("/api/pull", post(pull))
        .route("/api/chat", post(chat))
        // Same frames as /api/chat, over a transport Cloudflare will not cut
        // at 100 s — see `chat_ws`.
        .route("/api/chat/ws", get(chat_ws))
        .route("/api/unload", post(unload))
        // Flame graphs from a profiled generation — see `profile_svg`.
        .route("/api/profile", get(profile_svg))
        .route("/api/profile/folded", get(profile_folded))
        // The sync server matched on (method, path) and answered anything
        // else with the same 404 JSON — keep that, rather than axum's bare
        // 405, so a client sees one error shape.
        .fallback(not_found)
        .method_not_allowed_fallback(not_found)
        // `+ 1` so a body just over the ceiling still reaches the handler
        // and gets the JSON "body too large" the sync reader produced.
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES + 1))
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

pub(crate) fn json_response(status: u16, body: serde_json::Value) -> Response {
    let status = axum::http::StatusCode::from_u16(status)
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

/// Parse a JSON body, or hand back the 400 response to return as-is. Keeps
/// the sync server's error wire format (`{"error": "bad json: …"}`).
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, Box<Response>> {
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(e) => {
            return Err(Box::new(json_response(
                400,
                json!({"error": format!("body read: {e}")}),
            )));
        }
    };
    if text.len() > MAX_BODY_BYTES {
        return Err(Box::new(json_response(
            400,
            json!({"error": "body too large"}),
        )));
    }
    serde_json::from_str::<T>(text).map_err(|e| {
        Box::new(json_response(
            400,
            json!({"error": format!("bad json: {e}")}),
        ))
    })
}

async fn not_found() -> Response {
    json_response(404, json!({"error": "not found"}))
}

async fn ui() -> Response {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("ui.html"),
    )
        .into_response()
}

async fn health() -> Response {
    blocking(|| {
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
        json_response(
            200,
            json!({
                "status": "ok",
                "device": engine::device_label(),
                "gpus": gpus,
                "cpu_cores": inv.cpu.logical_cores,
            }),
        )
    })
    .await
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
            json!({"models": list, "device": engine::device_label()}),
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
        json_response(200, json!({"status": "unloaded"}))
    } else {
        json_response(
            409,
            json!({"error": "a generation is in flight — the model stays resident until it finishes"}),
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

fn sse_response(mut rx: mpsc::UnboundedReceiver<serde_json::Value>) -> Response {
    let stream = async_stream::stream! {
        while let Some(frame) = rx.recv().await {
            yield Ok::<Event, Infallible>(Event::default().data(frame.to_string()));
        }
    };
    // axum's `Sse` sets `text/event-stream` + `no-cache`; the third header
    // is the one that keeps nginx from buffering the stream into silence.
    ([("x-accel-buffering", "no")], Sse::new(stream)).into_response()
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
            let pct = p
                .total_bytes
                .map(|t| ((p.received_bytes as f64 / t.max(1) as f64) * 100.0) as i64);
            // Throttle: one frame per whole percent (or per 64 MiB when the
            // server didn't announce a total).
            let tick = pct.unwrap_or((p.received_bytes >> 26) as i64);
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
    sse_response(rx)
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
/// Cloudflare does not apply that timeout to WebSockets.
///
/// The upgrade alone is not the fix, though. An idle WebSocket is still
/// reaped, so this **heartbeats while the model loads** — without the ping
/// below the connection dies at the same place, just with a different error.
async fn chat_ws(upgrade: axum::extract::ws::WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(|socket| async move {
        if let Err(e) = drive_chat_ws(socket).await {
            eprintln!("[mummu-serve] chat ws: {e}");
        }
    })
}

async fn drive_chat_ws(mut socket: axum::extract::ws::WebSocket) -> Result<(), String> {
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
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Close(_))) | None => return Ok(()),
            Some(Ok(_)) => return Err("expected a text frame carrying the request".into()),
            Some(Err(e)) => return Err(e.to_string()),
        }
    };
    let parsed: ChatRequest = serde_json::from_str(&request).map_err(|e| e.to_string())?;

    let mut rx = match start_chat(parsed) {
        Ok(rx) => rx,
        Err(response) => {
            // Report the rejection in-band and close cleanly: a WebSocket
            // client cannot read the HTTP status of a request it never made.
            let status = response.status().as_u16();
            let frame = json!({"type": "error", "status": status});
            let _ = socket.send(Message::Text(frame.to_string().into())).await;
            return Ok(());
        }
    };

    let mut beat = tokio::time::interval(HEARTBEAT);
    // Missed ticks are worthless: if we were not polled for a minute, sending
    // four pings at once proves nothing to a proxy that already timed us out.
    beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    beat.tick().await; // the first tick is immediate; skip it
    loop {
        tokio::select! {
            event = rx.recv() => match event {
                Some(frame) => {
                    let done = frame.get("type").and_then(|t| t.as_str()) == Some("done")
                        || frame.get("type").and_then(|t| t.as_str()) == Some("error");
                    if socket.send(Message::Text(frame.to_string().into())).await.is_err() {
                        return Ok(()); // client gone; the generation task sees the closed channel
                    }
                    if done {
                        let _ = socket.close().await;
                        return Ok(());
                    }
                }
                None => {
                    let _ = socket.close().await;
                    return Ok(());
                }
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
    pub(crate) content: String,
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

pub(crate) fn to_turns(messages: &[ChatMessage]) -> Result<Vec<Turn>, String> {
    if messages.is_empty() {
        return Err("messages must be non-empty".into());
    }
    if messages.len() > MAX_TURNS {
        return Err(format!("more than {MAX_TURNS} messages"));
    }
    let turns: Vec<Turn> = messages
        .iter()
        .map(|m| match m.role.as_str() {
            "system" => Ok(Turn::system(m.content.clone())),
            "user" => Ok(Turn::user(m.content.clone())),
            "assistant" => Ok(Turn::assistant(m.content.clone())),
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
            .map_or(0, |d| d.as_nanos() as u64)
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
            *LAST_PROFILE.lock().unwrap_or_else(|e| e.into_inner()) = Some((svg, folded));
        }
        Err(e) => eprintln!("[mummu-serve] profile: flamegraph failed: {e}"),
    }
}

/// GET /api/profile — the last profiled generation's flame graph, as SVG a
/// browser renders directly (frames are zoomable; widths are self time).
async fn profile_svg() -> Response {
    let last = LAST_PROFILE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    match last {
        Some((svg, _)) => Response::builder()
            .header("content-type", "image/svg+xml")
            .body(svg.into())
            .expect("static response"),
        None => json_response(
            404,
            json!({"error": "no profiled generation yet — POST /api/chat with \"profile\": true, then retry"}),
        ),
    }
}

/// GET /api/profile/folded — the same data as folded stacks (`path (Nx) µs`
/// per line), for tooling or a quick sort in a terminal.
async fn profile_folded() -> Response {
    let last = LAST_PROFILE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    match last {
        Some((_, folded)) => Response::builder()
            .header("content-type", "text/plain; charset=utf-8")
            .body(folded.into())
            .expect("static response"),
        None => json_response(
            404,
            json!({"error": "no profiled generation yet — POST /api/chat with \"profile\": true, then retry"}),
        ),
    }
}

async fn chat(body: Bytes) -> Response {
    let parsed: ChatRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    match start_chat(parsed) {
        Ok(rx) => sse_response(rx),
        Err(response) => *response,
    }
}

/// Validate a chat request and start generating, returning the stream of
/// events. Shared by the SSE and WebSocket endpoints so the two cannot drift.
///
/// The generation outlives the caller: it runs as its own task and whoever
/// holds the receiver drains it.
fn start_chat(
    parsed: ChatRequest,
) -> Result<mpsc::UnboundedReceiver<serde_json::Value>, Box<Response>> {
    let turns = match to_turns(&parsed.messages) {
        Ok(t) => t,
        Err(e) => return Err(Box::new(json_response(400, json!({"error": e})))),
    };
    let opts = match sampler_options(&parsed.options) {
        Ok(o) => o,
        Err(e) => return Err(Box::new(json_response(400, json!({"error": e})))),
    };
    let max_tokens = parsed.max_tokens();

    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let Some(spec) = manager
        .catalog()
        .iter()
        .find(|s| s.name == parsed.model)
        .cloned()
    else {
        return Err(Box::new(json_response(
            404,
            json!({"error": format!("unknown model {:?}", parsed.model)}),
        )));
    };
    if !engine::is_installed(&spec, &root) {
        return Err(Box::new(json_response(
            409,
            json!({"error": format!("{} is not installed — pull it first", spec.name)}),
        )));
    }

    let profile = parsed.profile || std::env::var("MUMMU_PROFILE").is_ok();
    let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();
    // A normal async task: the generation is mostly awaits. The one part
    // that genuinely blocks — the model load — declares itself as blocking
    // where it happens, in `mummu::cache`, rather than this pushing the whole
    // future onto a blocking thread.
    tokio::spawn(async move {
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
        let result = engine::run_chat(&spec, &root, &turns, &opts, max_tokens, |delta| {
            if tx.send(json!({"type": "delta", "text": delta})).is_err() {
                return ControlFlow::Break(()); // client gone: stop decoding
            }
            ControlFlow::Continue(())
        })
        .await;
        if let Some(session) = profile_session {
            drop(session); // stop collecting before folding the report
            publish_profile();
        }
        let done = match result {
            Ok(r) => {
                let secs = (r.elapsed_ms as f64 / 1000.0).max(1e-3);
                json!({
                    "type": "done",
                    "text": r.text,
                    "tokens": r.tokens,
                    "device": r.device,
                    "elapsed_ms": r.elapsed_ms,
                    "tokens_per_second": (r.tokens as f64 / secs * 10.0).round() / 10.0,
                })
            }
            Err(e) => {
                eprintln!(
                    "[mummu-serve] chat {}: {e} (after {} ms)",
                    spec.name,
                    started.elapsed().as_millis()
                );
                json!({"type": "error", "error": e})
            }
        };
        let _ = tx.send(done);
    });
    Ok(rx)
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

    #[test]
    fn max_tokens_is_clamped_to_the_hard_ceiling() {
        let parsed = chat_request(r#"{"model": "m", "messages": [], "max_tokens": 0}"#);
        assert_eq!(parsed.max_tokens(), 1);

        let parsed = chat_request(r#"{"model": "m", "messages": [], "max_tokens": 999999}"#);
        assert_eq!(parsed.max_tokens(), MAX_MAX_TOKENS);
    }
}
