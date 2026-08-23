//! mummu-serve — a minimal HTTP server + single-page chat UI over mummu.
//!
//! Sync all the way down (tiny_http worker threads, no async runtime),
//! matching the library it serves. Endpoints:
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

use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use mummu::chat::{Role, Turn};
use mummu::decode::SamplerOptions;
use mummu::manage::ModelManager;
use serde::Deserialize;
use serde_json::json;
use tiny_http::{Header, Method, Request, Response, Server};

/// Hard ceilings so one request can't wedge the process.
pub(crate) const MAX_BODY_BYTES: usize = 4 << 20;
pub(crate) const MAX_TURNS: usize = 256;
pub(crate) const MAX_MAX_TOKENS: usize = 4096;
pub(crate) const DEFAULT_MAX_TOKENS: usize = 512;

pub(crate) fn models_root() -> PathBuf {
    std::env::var_os("MUMMU_MODELS_DIR").map_or_else(|| PathBuf::from("models"), PathBuf::from)
}

fn main() {
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
        shim::spawn(&shim_addr);
    }

    let server = Arc::new(Server::http(&addr).expect("bind"));
    eprintln!("[mummu-serve] listening on http://{addr}");

    // A small worker pool: generations serialize on the model-slot mutex
    // anyway, but health/models/UI requests keep answering while one runs.
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let server = Arc::clone(&server);
            std::thread::spawn(move || {
                while let Ok(req) = server.recv() {
                    handle(req);
                }
            })
        })
        .collect();
    for w in workers {
        let _ = w.join();
    }
}

fn handle(req: Request) {
    let method = req.method().clone();
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let result = match (&method, path.as_str()) {
        (Method::Get, "/") | (Method::Get, "/index.html") => ui(req),
        (Method::Get, "/api/health") => health(req),
        (Method::Get, "/api/models") => models(req),
        (Method::Post, "/api/pull") => pull(req),
        (Method::Post, "/api/chat") => chat(req),
        (Method::Post, "/api/unload") => unload(req),
        _ => respond_json(req, 404, json!({"error": "not found"})),
    };
    if let Err(e) = result {
        eprintln!("[mummu-serve] {method} {path}: respond failed: {e}");
    }
}

pub(crate) fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("static header")
}

pub(crate) fn respond_json(req: Request, status: u16, body: serde_json::Value) -> std::io::Result<()> {
    req.respond(
        Response::from_string(body.to_string())
            .with_status_code(status)
            .with_header(json_header()),
    )
}

/// Read + parse a JSON body, or answer the 400 and return `None`.
pub(crate) fn read_json<T: serde::de::DeserializeOwned>(mut req: Request) -> std::io::Result<Option<(Request, T)>> {
    let mut body = String::new();
    let read = req
        .as_reader()
        .take(MAX_BODY_BYTES as u64 + 1)
        .read_to_string(&mut body);
    if let Err(e) = read {
        respond_json(req, 400, json!({"error": format!("body read: {e}")}))?;
        return Ok(None);
    }
    if body.len() > MAX_BODY_BYTES {
        respond_json(req, 400, json!({"error": "body too large"}))?;
        return Ok(None);
    }
    match serde_json::from_str::<T>(&body) {
        Ok(parsed) => Ok(Some((req, parsed))),
        Err(e) => {
            respond_json(req, 400, json!({"error": format!("bad json: {e}")}))?;
            Ok(None)
        }
    }
}

fn ui(req: Request) -> std::io::Result<()> {
    req.respond(
        Response::from_string(include_str!("ui.html")).with_header(
            Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
                .expect("static header"),
        ),
    )
}

fn health(req: Request) -> std::io::Result<()> {
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
    respond_json(
        req,
        200,
        json!({
            "status": "ok",
            "device": engine::device_label(),
            "gpus": gpus,
            "cpu_cores": inv.cpu.logical_cores,
        }),
    )
}

fn models(req: Request) -> std::io::Result<()> {
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
    respond_json(
        req,
        200,
        json!({"models": list, "device": engine::device_label()}),
    )
}

fn unload(req: Request) -> std::io::Result<()> {
    engine::unload_all();
    respond_json(req, 200, json!({"status": "unloaded"}))
}

// ---------------------------------------------------------------------------
// SSE plumbing: the handler spawns the work on its own thread, which feeds
// `data: {json}` frames through a bounded channel; the request thread pumps
// the channel out as a chunked response. A dropped client closes the pump,
// the next send fails, and the worker breaks off cooperatively.
// ---------------------------------------------------------------------------

pub(crate) struct ChannelReader {
    pub(crate) rx: Receiver<Vec<u8>>,
    pub(crate) buf: Vec<u8>,
    pub(crate) pos: usize,
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(chunk) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Err(_) => return Ok(0), // sender done -> EOF
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn sse_frame(value: &serde_json::Value) -> Vec<u8> {
    format!("data: {value}\n\n").into_bytes()
}

fn respond_sse(
    req: Request,
    work: impl FnOnce(&SyncSender<Vec<u8>>) + Send + 'static,
) -> std::io::Result<()> {
    let (tx, rx) = sync_channel::<Vec<u8>>(64);
    std::thread::spawn(move || work(&tx));
    let headers = vec![
        Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).expect("static"),
        Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).expect("static"),
        Header::from_bytes(&b"X-Accel-Buffering"[..], &b"no"[..]).expect("static"),
    ];
    req.respond(Response::new(
        200.into(),
        headers,
        ChannelReader {
            rx,
            buf: Vec::new(),
            pos: 0,
        },
        None,
        None,
    ))
}

// ---------------------------------------------------------------------------
// POST /api/pull — download a catalog model, streaming progress frames.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PullRequest {
    model: String,
}

fn pull(req: Request) -> std::io::Result<()> {
    let Some((req, parsed)) = read_json::<PullRequest>(req)? else {
        return Ok(());
    };
    let root = models_root();
    respond_sse(req, move |tx| {
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
            let frame = sse_frame(&json!({
                "type": "progress",
                "file": p.file,
                "received_bytes": p.received_bytes,
                "total_bytes": p.total_bytes,
                "percent": pct,
            }));
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
        let _ = tx.send(sse_frame(&done));
    })
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

fn chat(req: Request) -> std::io::Result<()> {
    let Some((req, parsed)) = read_json::<ChatRequest>(req)? else {
        return Ok(());
    };
    let turns = match to_turns(&parsed.messages) {
        Ok(t) => t,
        Err(e) => return respond_json(req, 400, json!({"error": e})),
    };
    let opts = match sampler_options(&parsed.options) {
        Ok(o) => o,
        Err(e) => return respond_json(req, 400, json!({"error": e})),
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
        return respond_json(req, 404, json!({"error": format!("unknown model {:?}", parsed.model)}));
    };
    if !engine::is_installed(&spec, &root) {
        return respond_json(
            req,
            409,
            json!({"error": format!("{} is not installed — pull it first", spec.name)}),
        );
    }

    respond_sse(req, move |tx| {
        let started = std::time::Instant::now();
        let result = engine::run_chat(&spec, &root, &turns, &opts, max_tokens, |delta| {
            let frame = sse_frame(&json!({"type": "delta", "text": delta}));
            if tx.send(frame).is_err() {
                return ControlFlow::Break(()); // client gone: stop decoding
            }
            ControlFlow::Continue(())
        });
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
        let _ = tx.send(sse_frame(&done));
    })
}
