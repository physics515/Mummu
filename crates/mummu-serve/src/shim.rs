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

use std::convert::Infallible;
use std::ops::ControlFlow;
use std::time::Instant;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::DefaultBodyLimit;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete as delete_route, get, post};
use mummu::manage::ModelManager;
use mummu::registry::{ModelSpec, WeightFormat};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use crate::{
    ChatMessage, DEFAULT_MAX_TOKENS, MAX_BODY_BYTES, MAX_MAX_TOKENS, blocking, engine,
    json_response, models_root, parse_json, to_turns,
};

/// The shim's routes. Binding and serving them (and draining them on
/// shutdown) belongs to `crate::serve_on`, which owns both listeners.
pub(crate) fn router() -> Router {
    Router::new()
        // `get` also answers HEAD (axum strips the body), which is what the
        // sync shim spelled out as a separate `HEAD /` arm.
        .route("/", get(root))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/ps", get(ps))
        .route("/api/show", post(show))
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .route("/api/pull", post(pull))
        .route("/api/delete", delete_route(delete))
        .route("/api/embed", post(no_embeddings))
        .route("/api/embeddings", post(no_embeddings))
        .route("/api/create", post(unsupported))
        .route("/api/copy", post(unsupported))
        .route("/api/push", post(unsupported))
        .fallback(not_found_path)
        .method_not_allowed_fallback(not_found_path)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES + 1))
}

async fn root() -> &'static str {
    "Ollama is running"
}

async fn version() -> Response {
    json_response(200, json!({"version": "0.1.0"}))
}

async fn no_embeddings() -> Response {
    json_response(
        501,
        json!({"error": "embeddings are not supported by the mummu-serve shim"}),
    )
}

async fn unsupported() -> Response {
    json_response(501, json!({"error": "not supported by the mummu-serve shim"}))
}

async fn not_found_path() -> Response {
    json_response(404, json!({"error": "not found"}))
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

async fn tags() -> Response {
    // `dir_size` walks every installed model's directory — disk work.
    blocking(|| {
        let root = models_root();
        let manager = ModelManager::new(root.clone());
        let models: Vec<_> = manager
            .catalog()
            .iter()
            .filter(|s| !matches!(s.architecture, mummu::registry::Architecture::MiniLm))
            .filter(|s| engine::is_installed(s, &root))
            .map(|s| model_entry(s, &root))
            .collect();
        json_response(200, json!({"models": models}))
    })
    .await
}

async fn ps() -> Response {
    blocking(|| {
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
        json_response(200, json!({"models": models}))
    })
    .await
}

#[derive(Deserialize)]
struct NameRequest {
    #[serde(alias = "name")]
    model: String,
}

async fn show(body: Bytes) -> Response {
    let parsed: NameRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    let root = models_root();
    let manager = ModelManager::new(root);
    let Some(spec) = resolve(&manager, &parsed.model) else {
        return not_found(&parsed.model);
    };
    let family = format!("{:?}", spec.architecture).to_lowercase();
    json_response(
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

async fn delete(body: Bytes) -> Response {
    let parsed: NameRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    blocking(move || {
        let manager = ModelManager::new(models_root());
        let Some(spec) = resolve(&manager, &parsed.model) else {
            return not_found(&parsed.model);
        };
        // The dir may be the resident model's backing store — refuse rather
        // than delete files out from under a running generation.
        if !engine::unload_all() {
            return json_response(
                409,
                json!({"error": "a generation is in flight — cannot delete a model that is loaded"}),
            );
        }
        match manager.remove(&spec.name) {
            Ok(()) => json_response(200, json!({})),
            Err(e) => json_response(500, json!({"error": e})),
        }
    })
    .await
}

fn not_found(model: &str) -> Response {
    json_response(
        404,
        json!({"error": format!("model {model:?} not found, try pulling it first")}),
    )
}

// ---------------------------------------------------------------------------
// NDJSON plumbing (ollama streams one JSON object per line). Same shape as
// the native API's SSE: a worker task feeds an mpsc channel, and the
// response body drains it — a dropped client closes the receiver, the next
// send fails, and the worker breaks off cooperatively.
// ---------------------------------------------------------------------------

fn ndjson_frame(value: &serde_json::Value) -> String {
    format!("{value}\n")
}

fn ndjson_response(mut rx: mpsc::UnboundedReceiver<serde_json::Value>) -> Response {
    let stream = async_stream::stream! {
        while let Some(frame) = rx.recv().await {
            yield Ok::<String, Infallible>(ndjson_frame(&frame));
        }
    };
    (
        [
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(stream),
    )
        .into_response()
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

/// Validate a request into a `RunPlan`, or hand back the error response.
fn plan(
    model: &str,
    messages: &[ChatMessage],
    options: &OllamaOptions,
) -> Result<RunPlan, Box<Response>> {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let Some(spec) = resolve(&manager, model) else {
        return Err(Box::new(not_found(model)));
    };
    if !engine::is_installed(&spec, &root) {
        return Err(Box::new(not_found(model)));
    }
    let turns = to_turns(messages).map_err(|e| json_response(400, json!({"error": e})))?;
    let opts = options
        .sampler()
        .map_err(|e| json_response(400, json!({"error": e})))?;
    let max_tokens = options.max_tokens();
    Ok(RunPlan {
        spec,
        root,
        turns,
        opts,
        max_tokens,
    })
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
/// Either way the engine runs on a blocking thread.
async fn run(
    p: RunPlan,
    stream: bool,
    wrap: fn(&str, &str) -> serde_json::Value,
    finish: fn(&str, &str, &engine::ChatResult, Instant) -> serde_json::Value,
) -> Response {
    let started = Instant::now();
    if stream {
        let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();
        tokio::spawn(async move {
            let result = engine::run_chat(
                &p.spec,
                &p.root,
                &p.turns,
                &p.opts,
                p.max_tokens,
                |delta| {
                    if tx.send(wrap(&p.spec.name, delta)).is_err() {
                        return ControlFlow::Break(());
                    }
                    ControlFlow::Continue(())
                },
            )
            .await;
            let last = match result {
                Ok(r) => finish(&p.spec.name, "", &r, started),
                Err(e) => {
                    eprintln!("[mummu-serve] shim chat {}: {e}", p.spec.name);
                    json!({"error": e})
                }
            };
            let _ = tx.send(last);
        });
        return ndjson_response(rx);
    }
    // Non-stream: run to completion, answer with one object.
    let result = engine::run_chat(&p.spec, &p.root, &p.turns, &p.opts, p.max_tokens, |_| {
        ControlFlow::Continue(())
    })
    .await;
    match result {
        Ok(r) => {
            let text = r.text.clone();
            json_response(200, finish(&p.spec.name, &text, &r, started))
        }
        Err(e) => {
            eprintln!("[mummu-serve] shim chat {}: {e}", p.spec.name);
            json_response(500, json!({"error": e}))
        }
    }
}

async fn chat(body: Bytes) -> Response {
    let parsed: OllamaChatRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    let p = match plan(&parsed.model, &parsed.messages, &parsed.options) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    run(
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
    .await
}

async fn generate(body: Bytes) -> Response {
    let parsed: OllamaGenerateRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
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
    let p = match plan(&parsed.model, &messages, &parsed.options) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    run(
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
    .await
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

async fn pull(body: Bytes) -> Response {
    let parsed: PullRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    let manager = ModelManager::new(models_root());
    let Some(spec) = resolve(&manager, &parsed.model) else {
        return json_response(
            404,
            json!({"error": format!(
                "model {:?} is not in the mummu catalog (the shim can only pull catalog models)",
                parsed.model
            )}),
        );
    };
    let stream = parsed.stream.unwrap_or(true);
    if !stream {
        let name = spec.name.clone();
        return blocking(move || {
            let manager = ModelManager::new(models_root());
            match manager.install(&name, |_| {}) {
                Ok(_) => json_response(200, json!({"status": "success"})),
                Err(e) => json_response(500, json!({"error": e})),
            }
        })
        .await;
    }
    let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();
    // The hub downloader is still synchronous, so it gets a blocking thread.
    tokio::task::spawn_blocking(move || {
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
            let frame = json!({
                "status": format!("pulling {}", p.file),
                "digest": "",
                "total": total,
                "completed": p.received_bytes,
            });
            if tx.send(frame).is_err() {
                cancelled = true;
            }
        });
        let last = match result {
            Ok(_) => json!({"status": "success"}),
            Err(e) => json!({"error": e}),
        };
        let _ = tx.send(last);
    });
    ndjson_response(rx)
}
