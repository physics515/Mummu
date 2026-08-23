//! mummu-serve — a minimal HTTP server + single-page chat UI over mummu.
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
//! - `POST /api/unload`  drop the resident model (frees VRAM/RAM)
//!
//! Configuration (env): `MUMMU_ADDR` (default `0.0.0.0:8095`),
//! `MUMMU_MODELS_DIR` (default `./models`), `MUMMU_FORCE_CPU`.

mod engine;
mod shim;

use std::convert::Infallible;
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
use tokio::sync::mpsc;

/// Hard ceilings so one request can't wedge the process.
pub(crate) const MAX_BODY_BYTES: usize = 4 << 20;
pub(crate) const MAX_TURNS: usize = 256;
pub(crate) const MAX_MAX_TOKENS: usize = 4096;
pub(crate) const DEFAULT_MAX_TOKENS: usize = 512;

pub(crate) fn models_root() -> PathBuf {
    std::env::var_os("MUMMU_MODELS_DIR").map_or_else(|| PathBuf::from("models"), PathBuf::from)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let addr = std::env::var("MUMMU_ADDR").unwrap_or_else(|_| "0.0.0.0:8095".into());
    let root = models_root();
    std::fs::create_dir_all(&root).expect("models dir must be creatable");

    // Report the device policy once at startup — the honest record of
    // whether this environment actually has a usable GPU adapter.
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

    // Ollama-compatibility shim on its own port (`MUMMU_OLLAMA_ADDR`;
    // "off" disables). 11435 rather than ollama's 11434 so a real ollama
    // sharing the network namespace can never collide with the shim.
    let shim_addr =
        std::env::var("MUMMU_OLLAMA_ADDR").unwrap_or_else(|_| "0.0.0.0:11435".into());
    if !matches!(shim_addr.as_str(), "" | "off" | "disabled" | "0") {
        shim::spawn(&shim_addr).await;
    }

    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    eprintln!("[mummu-serve] listening on http://{addr}");

    // One router, many concurrent requests: generations serialize on the
    // model-slot mutex inside `spawn_blocking`, but health/models/UI
    // requests keep answering on the async workers while one runs.
    if let Err(e) = axum::serve(listener, router())
        .with_graceful_shutdown(shutdown_signal("api"))
        .await
    {
        eprintln!("[mummu-serve] serve failed: {e}");
    }
}

fn router() -> Router {
    Router::new()
        .route("/", get(ui))
        .route("/index.html", get(ui))
        .route("/api/health", get(health))
        .route("/api/models", get(models))
        .route("/api/pull", post(pull))
        .route("/api/chat", post(chat))
        .route("/api/unload", post(unload))
        // The sync server matched on (method, path) and answered anything
        // else with the same 404 JSON — keep that, rather than axum's bare
        // 405, so a client sees one error shape.
        .fallback(not_found)
        .method_not_allowed_fallback(not_found)
        // `+ 1` so a body just over the ceiling still reaches the handler
        // and gets the JSON "body too large" the sync reader produced.
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES + 1))
}

/// Resolve when the process is asked to stop. Both listeners await this;
/// tokio's ctrl-c stream fans out to every waiter.
pub(crate) async fn shutdown_signal(which: &'static str) {
    match tokio::signal::ctrl_c().await {
        Ok(()) => eprintln!("[mummu-serve] ctrl-c — draining the {which} listener"),
        Err(e) => eprintln!("[mummu-serve] {which}: ctrl-c handler unavailable: {e}"),
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
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(
    body: &Bytes,
) -> Result<T, Box<Response>> {
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(e) => return Err(Box::new(json_response(400, json!({"error": format!("body read: {e}")})))),
    };
    if text.len() > MAX_BODY_BYTES {
        return Err(Box::new(json_response(400, json!({"error": "body too large"}))));
    }
    serde_json::from_str::<T>(text)
        .map_err(|e| Box::new(json_response(400, json!({"error": format!("bad json: {e}")}))))
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
        return Err(format!("temperature must be finite and >= 0, got {temperature}"));
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

async fn chat(body: Bytes) -> Response {
    let parsed: ChatRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    let turns = match to_turns(&parsed.messages) {
        Ok(t) => t,
        Err(e) => return json_response(400, json!({"error": e})),
    };
    let opts = match sampler_options(&parsed.options) {
        Ok(o) => o,
        Err(e) => return json_response(400, json!({"error": e})),
    };
    let max_tokens = parsed
        .options
        .max_tokens
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .clamp(1, MAX_MAX_TOKENS);

    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let Some(spec) = manager
        .catalog()
        .iter()
        .find(|s| s.name == parsed.model)
        .cloned()
    else {
        return json_response(404, json!({"error": format!("unknown model {:?}", parsed.model)}));
    };
    if !engine::is_installed(&spec, &root) {
        return json_response(
            409,
            json!({"error": format!("{} is not installed — pull it first", spec.name)}),
        );
    }

    let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();
    // The generation outlives this handler: it runs as its own task and the
    // response body below drains what it sends.
    tokio::spawn(async move {
        let started = std::time::Instant::now();
        let result = engine::run_chat(&spec, &root, &turns, &opts, max_tokens, |delta| {
            if tx.send(json!({"type": "delta", "text": delta})).is_err() {
                return ControlFlow::Break(()); // client gone: stop decoding
            }
            ControlFlow::Continue(())
        })
        .await;
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
    sse_response(rx)
}
