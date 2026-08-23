//! Ollama-compatibility shim: a second listener that speaks the Ollama HTTP
//! protocol (NDJSON streaming) and drives the same engine as the native API,
//! so Ollama clients — open-webui, LangChain's Ollama integration, plain
//! `curl` scripts — can use mummu without knowing it isn't ollama. The two
//! surfaces share the backend slots, so a model loaded here is the same
//! resident model the native UI talks to.
//!
//! Implemented: `GET /`, `GET /api/version`, `GET /api/tags`,
//! `POST /api/show`, `GET /api/ps`, `POST /api/chat`, `POST /api/generate`
//! (both stream and non-stream), `POST /api/pull`, `DELETE /api/delete`.
//! Embeddings/create/copy/push answer with an explicit error rather than
//! pretending. Model names are mummu's catalog names; a trailing `:latest`
//! (which ollama CLIs append) is accepted and stripped.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::Instant;

use mummu::manage::ModelManager;
use mummu::registry::{ModelSpec, WeightFormat};
use serde::Deserialize;
use serde_json::json;
use tiny_http::{Header, Method, Request, Response, Server};

use crate::{
    ChannelReader, ChatMessage, DEFAULT_MAX_TOKENS, MAX_MAX_TOKENS, engine, models_root,
    read_json, respond_json, to_turns,
};

/// Spawn the shim's listener + workers; returns quietly (with a log line)
/// when the address can't be bound so the native API keeps serving.
pub(crate) fn spawn(addr: &str) {
    let server = match Server::http(addr) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("[mummu-serve] ollama shim: bind {addr} failed ({e}) — shim disabled");
            return;
        }
    };
    eprintln!("[mummu-serve] ollama-compatible shim listening on http://{addr}");
    for _ in 0..2 {
        let server = Arc::clone(&server);
        std::thread::spawn(move || {
            while let Ok(req) = server.recv() {
                handle(req);
            }
        });
    }
}

fn handle(req: Request) {
    let method = req.method().clone();
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let result = match (&method, path.as_str()) {
        (Method::Get, "/") => req.respond(Response::from_string("Ollama is running")),
        (Method::Head, "/") => req.respond(Response::empty(200)),
        (Method::Get, "/api/version") => respond_json(req, 200, json!({"version": "0.1.0"})),
        (Method::Get, "/api/tags") => tags(req),
        (Method::Get, "/api/ps") => ps(req),
        (Method::Post, "/api/show") => show(req),
        (Method::Post, "/api/chat") => chat(req),
        (Method::Post, "/api/generate") => generate(req),
        (Method::Post, "/api/pull") => pull(req),
        (Method::Delete, "/api/delete") => delete(req),
        (Method::Post, "/api/embed") | (Method::Post, "/api/embeddings") => respond_json(
            req,
            501,
            json!({"error": "embeddings are not supported by the mummu-serve shim"}),
        ),
        (Method::Post, "/api/create") | (Method::Post, "/api/copy") | (Method::Post, "/api/push") => {
            respond_json(req, 501, json!({"error": "not supported by the mummu-serve shim"}))
        }
        _ => respond_json(req, 404, json!({"error": "not found"})),
    };
    if let Err(e) = result {
        eprintln!("[mummu-serve] shim {method} {path}: respond failed: {e}");
    }
}

// ---------------------------------------------------------------------------
// Naming, timestamps, digests
// ---------------------------------------------------------------------------

/// Resolve an ollama-style model reference to a catalog spec: exact name
/// first, then with a `:latest` tag stripped (ollama CLIs append it).
fn resolve(manager: &ModelManager, name: &str) -> Option<ModelSpec> {
    let bare = name.strip_suffix(":latest").unwrap_or(name);
    manager
        .catalog()
        .iter()
        .find(|s| s.name == name || s.name == bare)
        .cloned()
}

/// RFC 3339 UTC from a `SystemTime` (no chrono dependency — civil-from-days,
/// Howard Hinnant's algorithm).
fn rfc3339(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        sod / 3600,
        (sod / 60) % 60,
        sod % 60
    )
}

fn now_rfc3339() -> String {
    rfc3339(std::time::SystemTime::now())
}

/// Stable fake digest: ollama clients treat it as an opaque identity.
fn digest(name: &str) -> String {
    use sha2::Digest;
    let hash = sha2::Sha256::digest(name.as_bytes());
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

fn details(spec: &ModelSpec) -> serde_json::Value {
    let family = format!("{:?}", spec.architecture).to_lowercase();
    json!({
        "parent_model": "",
        "format": match &spec.format {
            WeightFormat::Safetensors => "safetensors",
            WeightFormat::Gguf { .. } => "gguf",
        },
        "family": family,
        "families": [family],
        "parameter_size": "",
        "quantization_level": "",
    })
}

fn model_entry(spec: &ModelSpec, root: &std::path::Path) -> serde_json::Value {
    let dir = spec.dir(root);
    let size = mummu::manage::dir_size(&dir);
    let modified = std::fs::metadata(&dir)
        .and_then(|m| m.modified())
        .map_or_else(|_| now_rfc3339(), rfc3339);
    json!({
        "name": spec.name,
        "model": spec.name,
        "modified_at": modified,
        "size": size,
        "digest": digest(&spec.name),
        "details": details(spec),
    })
}

// ---------------------------------------------------------------------------
// Catalog endpoints
// ---------------------------------------------------------------------------

fn tags(req: Request) -> std::io::Result<()> {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let models: Vec<_> = manager
        .catalog()
        .iter()
        .filter(|s| !matches!(s.architecture, mummu::registry::Architecture::MiniLm))
        .filter(|s| engine::is_installed(s, &root))
        .map(|s| model_entry(s, &root))
        .collect();
    respond_json(req, 200, json!({"models": models}))
}

fn ps(req: Request) -> std::io::Result<()> {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let resident = engine::resident_dirs();
    let models: Vec<_> = manager
        .catalog()
        .iter()
        .filter(|s| resident.iter().any(|d| *d == s.dir(&root)))
        .map(|s| {
            let mut entry = model_entry(s, &root);
            entry["expires_at"] = json!(now_rfc3339());
            entry["size_vram"] = entry["size"].clone();
            entry
        })
        .collect();
    respond_json(req, 200, json!({"models": models}))
}

#[derive(Deserialize)]
struct NameRequest {
    #[serde(alias = "name")]
    model: String,
}

fn show(req: Request) -> std::io::Result<()> {
    let Some((req, parsed)) = read_json::<NameRequest>(req)? else {
        return Ok(());
    };
    let root = models_root();
    let manager = ModelManager::new(root);
    let Some(spec) = resolve(&manager, &parsed.model) else {
        return not_found(req, &parsed.model);
    };
    let family = format!("{:?}", spec.architecture).to_lowercase();
    respond_json(
        req,
        200,
        json!({
            "modelfile": format!("# mummu catalog model {} ({})", spec.name, spec.repo),
            "parameters": "",
            "template": "{{ .Prompt }}",
            "details": details(&spec),
            "model_info": { "general.architecture": family },
            "capabilities": ["completion"],
        }),
    )
}

fn delete(req: Request) -> std::io::Result<()> {
    let Some((req, parsed)) = read_json::<NameRequest>(req)? else {
        return Ok(());
    };
    let manager = ModelManager::new(models_root());
    let Some(spec) = resolve(&manager, &parsed.model) else {
        return not_found(req, &parsed.model);
    };
    engine::unload_all(); // the dir may be the resident model's backing store
    match manager.remove(&spec.name) {
        Ok(()) => respond_json(req, 200, json!({})),
        Err(e) => respond_json(req, 500, json!({"error": e})),
    }
}

fn not_found(req: Request, model: &str) -> std::io::Result<()> {
    respond_json(
        req,
        404,
        json!({"error": format!("model {model:?} not found, try pulling it first")}),
    )
}

// ---------------------------------------------------------------------------
// NDJSON plumbing (ollama streams one JSON object per line)
// ---------------------------------------------------------------------------

fn ndjson_frame(value: &serde_json::Value) -> Vec<u8> {
    format!("{value}\n").into_bytes()
}

fn respond_ndjson(
    req: Request,
    work: impl FnOnce(&SyncSender<Vec<u8>>) + Send + 'static,
) -> std::io::Result<()> {
    let (tx, rx) = sync_channel::<Vec<u8>>(64);
    std::thread::spawn(move || work(&tx));
    let headers = vec![
        Header::from_bytes(&b"Content-Type"[..], &b"application/x-ndjson"[..]).expect("static"),
        Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).expect("static"),
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
// Chat + generate
// ---------------------------------------------------------------------------

/// The sampling knobs ollama clients put in `options` (names are ollama's).
#[derive(Deserialize, Default)]
struct OllamaOptions {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    num_predict: Option<i64>,
}

impl OllamaOptions {
    fn sampler(&self) -> Result<mummu::decode::SamplerOptions, String> {
        let defaults = mummu::decode::SamplerOptions::default();
        let temperature = self.temperature.unwrap_or(0.7);
        let top_p = self.top_p.unwrap_or(0.9);
        let top_k = self.top_k.unwrap_or(defaults.top_k);
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(format!("temperature must be finite and >= 0, got {temperature}"));
        }
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {top_p}"));
        }
        if top_k == 0 {
            return Err("top_k must be >= 1".into());
        }
        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        });
        Ok(mummu::decode::SamplerOptions {
            temperature,
            top_p,
            top_k,
            seed,
        })
    }

    /// Ollama's `num_predict`: -1 (and 0/absent) mean "model default".
    fn max_tokens(&self) -> usize {
        match self.num_predict {
            Some(n) if n > 0 => (n as usize).min(MAX_MAX_TOKENS),
            _ => DEFAULT_MAX_TOKENS,
        }
    }
}

#[derive(Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    options: OllamaOptions,
    /// Ollama defaults to streaming.
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct OllamaGenerateRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    options: OllamaOptions,
    stream: Option<bool>,
}

/// Everything a chat/generate run needs after validation.
struct RunPlan {
    spec: ModelSpec,
    root: std::path::PathBuf,
    turns: Vec<mummu::chat::Turn>,
    opts: mummu::decode::SamplerOptions,
    max_tokens: usize,
}

fn plan(
    req: Request,
    model: &str,
    messages: &[ChatMessage],
    options: &OllamaOptions,
) -> std::io::Result<Option<(Request, RunPlan)>> {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let Some(spec) = resolve(&manager, model) else {
        not_found(req, model)?;
        return Ok(None);
    };
    if !engine::is_installed(&spec, &root) {
        not_found(req, model)?;
        return Ok(None);
    }
    let turns = match to_turns(messages) {
        Ok(t) => t,
        Err(e) => {
            respond_json(req, 400, json!({"error": e}))?;
            return Ok(None);
        }
    };
    let opts = match options.sampler() {
        Ok(o) => o,
        Err(e) => {
            respond_json(req, 400, json!({"error": e}))?;
            return Ok(None);
        }
    };
    let max_tokens = options.max_tokens();
    Ok(Some((
        req,
        RunPlan {
            spec,
            root,
            turns,
            opts,
            max_tokens,
        },
    )))
}

/// Ollama's final frame: timing in nanoseconds.
fn done_value(model: &str, r: &engine::ChatResult, started: Instant) -> serde_json::Value {
    let total_ns = started.elapsed().as_nanos() as u64;
    let eval_ns = (r.elapsed_ms as u64).saturating_mul(1_000_000);
    json!({
        "model": model,
        "created_at": now_rfc3339(),
        "done": true,
        "done_reason": "stop",
        "total_duration": total_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": r.tokens,
        "eval_duration": eval_ns,
    })
}

/// Run one completion for the shim: streamed (one NDJSON frame per delta)
/// or buffered, with `wrap` turning a text delta into the endpoint's frame
/// shape (`message.content` for /api/chat, `response` for /api/generate).
fn run(
    req: Request,
    p: RunPlan,
    stream: bool,
    wrap: fn(&str, &str) -> serde_json::Value,
    finish: fn(&str, &str, &engine::ChatResult, Instant) -> serde_json::Value,
) -> std::io::Result<()> {
    let started = Instant::now();
    if stream {
        return respond_ndjson(req, move |tx| {
            let result = engine::run_chat(
                &p.spec,
                &p.root,
                &p.turns,
                &p.opts,
                p.max_tokens,
                |delta| {
                    if tx.send(ndjson_frame(&wrap(&p.spec.name, delta))).is_err() {
                        return ControlFlow::Break(());
                    }
                    ControlFlow::Continue(())
                },
            );
            let last = match result {
                Ok(r) => finish(&p.spec.name, "", &r, started),
                Err(e) => {
                    eprintln!("[mummu-serve] shim chat {}: {e}", p.spec.name);
                    json!({"error": e})
                }
            };
            let _ = tx.send(ndjson_frame(&last));
        });
    }
    // Non-stream: run to completion, answer with one object.
    let result = engine::run_chat(&p.spec, &p.root, &p.turns, &p.opts, p.max_tokens, |_| {
        ControlFlow::Continue(())
    });
    match result {
        Ok(r) => {
            let text = r.text.clone();
            respond_json(req, 200, finish(&p.spec.name, &text, &r, started))
        }
        Err(e) => {
            eprintln!("[mummu-serve] shim chat {}: {e}", p.spec.name);
            respond_json(req, 500, json!({"error": e}))
        }
    }
}

fn chat(req: Request) -> std::io::Result<()> {
    let Some((req, parsed)) = read_json::<OllamaChatRequest>(req)? else {
        return Ok(());
    };
    let Some((req, p)) = plan(req, &parsed.model, &parsed.messages, &parsed.options)? else {
        return Ok(());
    };
    run(
        req,
        p,
        parsed.stream.unwrap_or(true),
        |model, delta| {
            json!({
                "model": model,
                "created_at": now_rfc3339(),
                "message": {"role": "assistant", "content": delta},
                "done": false,
            })
        },
        |model, text, r, started| {
            let mut v = done_value(model, r, started);
            v["message"] = json!({"role": "assistant", "content": text});
            v
        },
    )
}

fn generate(req: Request) -> std::io::Result<()> {
    let Some((req, parsed)) = read_json::<OllamaGenerateRequest>(req)? else {
        return Ok(());
    };
    // Ollama applies the model's chat template to `prompt` (unless raw);
    // mirror that by wrapping it as (system +) user turns.
    let mut messages = Vec::new();
    if let Some(system) = &parsed.system {
        messages.push(ChatMessage {
            role: "system".into(),
            content: system.clone(),
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: parsed.prompt.clone(),
    });
    let Some((req, p)) = plan(req, &parsed.model, &messages, &parsed.options)? else {
        return Ok(());
    };
    run(
        req,
        p,
        parsed.stream.unwrap_or(true),
        |model, delta| {
            json!({
                "model": model,
                "created_at": now_rfc3339(),
                "response": delta,
                "done": false,
            })
        },
        |model, text, r, started| {
            let mut v = done_value(model, r, started);
            v["response"] = json!(text);
            v
        },
    )
}

// ---------------------------------------------------------------------------
// Pull
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PullRequest {
    #[serde(alias = "name")]
    model: String,
    stream: Option<bool>,
}

fn pull(req: Request) -> std::io::Result<()> {
    let Some((req, parsed)) = read_json::<PullRequest>(req)? else {
        return Ok(());
    };
    let manager = ModelManager::new(models_root());
    let Some(spec) = resolve(&manager, &parsed.model) else {
        return respond_json(
            req,
            404,
            json!({"error": format!(
                "model {:?} is not in the mummu catalog (the shim can only pull catalog models)",
                parsed.model
            )}),
        );
    };
    let stream = parsed.stream.unwrap_or(true);
    if !stream {
        let result = manager.install(&spec.name, |_| {});
        return match result {
            Ok(_) => respond_json(req, 200, json!({"status": "success"})),
            Err(e) => respond_json(req, 500, json!({"error": e})),
        };
    }
    respond_ndjson(req, move |tx| {
        let manager = ModelManager::new(models_root());
        let mut last_pct: i64 = -1;
        let mut cancelled = false;
        let result = manager.install(&spec.name, |p| {
            if cancelled {
                return;
            }
            let total = p.total_bytes.unwrap_or(0);
            let pct = if total > 0 {
                ((p.received_bytes as f64 / total as f64) * 100.0) as i64
            } else {
                (p.received_bytes >> 26) as i64
            };
            if pct == last_pct {
                return;
            }
            last_pct = pct;
            let frame = ndjson_frame(&json!({
                "status": format!("pulling {}", p.file),
                "digest": "",
                "total": total,
                "completed": p.received_bytes,
            }));
            if tx.send(frame).is_err() {
                cancelled = true;
            }
        });
        let last = match result {
            Ok(_) => json!({"status": "success"}),
            Err(e) => json!({"error": e}),
        };
        let _ = tx.send(ndjson_frame(&last));
    })
}
