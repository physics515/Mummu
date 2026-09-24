//! Ollama-compatibility shim: a second listener that speaks the Ollama HTTP
//! protocol (NDJSON streaming) and drives the same engine as the native API,
//! so Ollama clients — open-webui, `LangChain`'s Ollama integration, plain
//! `curl` scripts — can use mummu without knowing it isn't ollama. The two
//! surfaces share the backend slots, so a model loaded here is the same
//! resident model the native UI talks to.
//!
//! Implemented: `GET /`, `GET /api/version`, `GET /api/tags`,
//! `POST /api/show`, `GET /api/ps`, `POST /api/chat`, `POST /api/generate`
//! (both stream and non-stream), `POST /api/pull`, `DELETE /api/delete`.
//! `/api/chat` takes `tools` for the families whose calls mummu reads back,
//! and `/api/show` reports each model's capabilities from the same source.
//! A request that sets `think: true` gets a reasoning model's thinking in
//! ollama's own field, `message.thinking` (`thinking` on /api/generate),
//! with `content` holding only the answer; `think` absent is off, which is
//! where mummu parts from ollama (see [`OllamaChatRequest::think`]).
//! Embeddings/create/copy/push answer with an explicit error rather than
//! pretending. Model names are mummu's catalog names; a trailing `:latest`
//! (which ollama CLIs append) is accepted and stripped.

use std::convert::Infallible;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::DefaultBodyLimit;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete as delete_route, get, post};
use mummu::chat::{ToolCall, ToolSpec};
use mummu::manage::ModelManager;
use mummu::registry::{Architecture, ModelSpec, WeightFormat};
use mummu_num::{f64_from_u64, trunc_i64};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use crate::engine::CallSyntax;
use crate::recovery::{self, ChatError, InFlight};
use crate::think::{Filter, Split};
use crate::{
    ChatMessage, DEFAULT_MAX_TOKENS, FinalFrame, MAX_BODY_BYTES, MAX_MAX_TOKENS, OutputFormat,
    blocking, engine, json_response, models_root, parse_json, to_turns,
};

/// The shim's routes. Binding and serving them (and draining them on
/// shutdown) belongs to `crate::serve_on`, which owns both listeners.
pub fn router() -> Router {
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
        .merge(crate::openai::router())
        .fallback(not_found_path)
        .method_not_allowed_fallback(not_found_path)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES + 1))
        // Same ring as the native API, tagged `shim`: an ollama client and a
        // browser talking to the same resident model belong in one feed, in
        // the order they arrived. Outermost, so rejections are recorded too.
        .layer(axum::middleware::from_fn(crate::logs::record_shim))
}

async fn root() -> &'static str {
    "Ollama is running"
}

async fn version() -> Response {
    json_response(200, &json!({"version": "0.1.0"}))
}

async fn no_embeddings() -> Response {
    json_response(
        501,
        &json!({"error": "embeddings are not supported by the mummu-serve shim"}),
    )
}

async fn unsupported() -> Response {
    json_response(
        501,
        &json!({"error": "not supported by the mummu-serve shim"}),
    )
}

async fn not_found_path() -> Response {
    json_response(404, &json!({"error": "not found"}))
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
/// Howard Hinnant's algorithm). Also how `recovery` stamps the times it
/// reports, so the two surfaces spell a time the same way.
pub fn rfc3339(time: std::time::SystemTime) -> String {
    let secs = time
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
        .cast_signed();
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
    use std::fmt::Write as _;
    let hash = sha2::Sha256::digest(name.as_bytes());
    let mut out = String::with_capacity(hash.len() * 2);
    for b in hash {
        // Writing into a `String` cannot fail.
        let _ = write!(out, "{b:02x}");
    }
    out
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
    // Size = the weight artifact, not the directory. A GGUF model's dir also
    // holds its `.mummu` pack — 221 GB on the 27B — which every ollama
    // client would display as the "model size", and which is pure extra
    // disk traffic to walk while a load is saturating the same disk.
    let size = spec
        .gguf_path(root)
        .and_then(|p| std::fs::metadata(&p).ok().map(|m| m.len()))
        .unwrap_or_else(|| mummu::manage::dir_size(&dir));
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

/// The last `/api/tags` answer and when it was built. Ollama clients poll
/// tags every few seconds, and while a model load saturates the disk even
/// the handful of metadata reads behind it can queue for minutes (observed
/// live 2026-08-28: tags timed out during the 27B cold load). Fresh entries
/// are served as-is; a stale one is served immediately while a single
/// background rebuild refreshes it.
static TAGS_CACHE: std::sync::Mutex<Option<(Instant, serde_json::Value)>> =
    std::sync::Mutex::new(None);
static TAGS_REFRESHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const TAGS_TTL: std::time::Duration = std::time::Duration::from_secs(5);

/// The `/api/tags` body, read from disk (catalog stats + weight sizes).
fn build_tags() -> serde_json::Value {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let models: Vec<_> = manager
        .catalog()
        .iter()
        .filter(|s| !matches!(s.architecture, mummu::registry::Architecture::MiniLm))
        .filter(|s| engine::is_installed(s, &root))
        .map(|s| model_entry(s, &root))
        .collect();
    json!({"models": models})
}

fn store_tags(body: serde_json::Value) {
    *TAGS_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((Instant::now(), body));
}

async fn tags() -> Response {
    json_response(200, &tags_body().await)
}

/// The `/api/tags` body, cache and all — the catalog source `/v1/models`
/// maps into `OpenAI`'s shape, so the two surfaces never disagree about what
/// is installed and neither one walks the disk twice.
pub async fn tags_body() -> serde_json::Value {
    let cached = TAGS_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    match cached {
        Some((at, body)) if at.elapsed() < TAGS_TTL => body,
        Some((_, body)) => {
            // Stale: answer with it now and rebuild behind the response —
            // one rebuilder at a time, so a slow disk queues one walk, not
            // one per poll.
            if !TAGS_REFRESHING.swap(true, std::sync::atomic::Ordering::SeqCst) {
                tokio::task::spawn_blocking(|| {
                    if let Ok(fresh) = std::panic::catch_unwind(build_tags) {
                        store_tags(fresh);
                    }
                    TAGS_REFRESHING.store(false, std::sync::atomic::Ordering::SeqCst);
                });
            }
            body
        }
        None => {
            let body = blocking(build_tags).await;
            store_tags(body.clone());
            body
        }
    }
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
        json_response(200, &json!({"models": models}))
    })
    .await
}

#[derive(Deserialize)]
struct NameRequest {
    #[serde(alias = "name")]
    model: String,
}

/// What a model can do, in ollama's words and ollama's order — the
/// `capabilities` of `/api/show`, which clients read to decide whether to
/// offer image upload, send tools, or show a thinking toggle.
///
/// Each is what a request to THIS server gets, not what the checkpoint could
/// do in principle: [`engine::supports_vision`], [`engine::supports_tools`]
/// and [`engine::thinks`] read the same places a request is served from.
/// all-MiniLM is an embedder, which is what ollama reports for `all-minilm`;
/// it cannot chat here, so "completion" would only invite a chat that fails
/// (`/api/embed` answers 501 and says why).
fn capabilities(spec: &ModelSpec, root: &Path) -> Vec<&'static str> {
    let arch = spec.architecture;
    if arch == Architecture::MiniLm {
        return vec!["embedding"];
    }
    let mut caps = vec!["completion"];
    if engine::supports_vision(spec, root) {
        caps.push("vision");
    }
    if engine::supports_tools(arch) {
        caps.push("tools");
    }
    if engine::thinks(arch) {
        caps.push("thinking");
    }
    caps
}

async fn show(body: Bytes) -> Response {
    let parsed: NameRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let Some(spec) = resolve(&manager, &parsed.model) else {
        return not_found(&parsed.model);
    };
    // Vision is a file beside the weights, and looking for one is a
    // directory read — seconds, on a disk a cold load is saturating — so it
    // stays off the async workers.
    let (spec, capabilities) = blocking(move || {
        let caps = capabilities(&spec, &root);
        (spec, caps)
    })
    .await;
    let family = format!("{:?}", spec.architecture).to_lowercase();
    json_response(
        200,
        &json!({
            "modelfile": format!("# mummu catalog model {} ({})", spec.name, spec.repo),
            "parameters": "",
            "template": "{{ .Prompt }}",
            "details": details(&spec),
            "model_info": { "general.architecture": family },
            "capabilities": capabilities,
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
                &json!({"error": "a generation is in flight — cannot delete a model that is loaded"}),
            );
        }
        match manager.remove(&spec.name) {
            Ok(()) => json_response(200, &json!({})),
            Err(e) => json_response(500, &json!({"error": e})),
        }
    })
    .await
}

fn not_found(model: &str) -> Response {
    json_response(
        404,
        &json!({"error": format!("model {model:?} not found, try pulling it first")}),
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

/// `chat` is `Some` for a chat stream, which must end on a final line — the
/// `done: true` object or an `{"error": …}` — and holds the response open for
/// a restarting process (see [`InFlight`]). A pull ends on its own `status`
/// line and passes `None`.
fn ndjson_response(
    mut rx: mpsc::UnboundedReceiver<serde_json::Value>,
    chat: Option<InFlight>,
) -> Response {
    let stream = async_stream::stream! {
        let must_end = chat.is_some();
        let _chat = chat;
        let mut ended = false;
        while let Some(frame) = rx.recv().await {
            ended |= is_final_line(&frame);
            yield Ok::<String, Infallible>(ndjson_frame(&frame));
        }
        // Never a stream that simply stops: ollama clients read a missing
        // final line as a model that answered nothing.
        if must_end && !ended {
            yield Ok(ndjson_frame(&ended_without_result()));
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

/// Does a chat stream end on this line? Ollama's own convention: the last
/// object carries `"done": true`, or it is an error object.
fn is_final_line(frame: &serde_json::Value) -> bool {
    frame.get("done") == Some(&json!(true)) || frame.get("error").is_some()
}

/// Ollama's error shape — `{"error": "…"}`, the whole object — for a stream's
/// last line and a non-stream's body alike.
fn error_line(e: &ChatError) -> serde_json::Value {
    json!({"error": e.message})
}

/// The last line when the worker vanished without writing one. Should be
/// unreachable; see `crate::ended_without_result`.
fn ended_without_result() -> serde_json::Value {
    json!({"error": "the server's worker for this request ended without a result — this is a \
                     mummu-serve bug, not your request; try again"})
}

// ---------------------------------------------------------------------------
// Chat + generate
// ---------------------------------------------------------------------------

/// The sampling knobs ollama clients put in `options` (names are ollama's).
#[derive(Deserialize, Default)]
pub struct OllamaOptions {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) seed: Option<u64>,
    pub(crate) num_predict: Option<i64>,
}

impl OllamaOptions {
    fn sampler(&self) -> Result<mummu::decode::SamplerOptions, String> {
        let defaults = mummu::decode::SamplerOptions::default();
        let temperature = self.temperature.unwrap_or(0.7);
        let top_p = self.top_p.unwrap_or(0.9);
        let top_k = self.top_k.unwrap_or(defaults.top_k);
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
        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, crate::nanos)
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
            // A count too big for a `usize` is certainly past the cap.
            Some(n) if n > 0 => {
                usize::try_from(n).map_or(MAX_MAX_TOKENS, |n| n.min(MAX_MAX_TOKENS))
            }
            _ => DEFAULT_MAX_TOKENS,
        }
    }
}

/// Ollama's `format`: the string `"json"`, or a JSON Schema object.
///
/// The schema form is accepted as far as *recognising* it and then refused
/// by name — mummu constrains to "some JSON value", not to a given shape.
/// Silently downgrading a schema request to plain JSON mode would hand the
/// client a document that parses and then fails its own validation, which is
/// the harder bug to find.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum OllamaFormat {
    Named(String),
    Schema(serde_json::Value),
}

impl OllamaFormat {
    /// The grammar this asks for, or the reason it cannot be served.
    pub(crate) fn resolve(&self) -> Result<Option<OutputFormat>, String> {
        match self {
            Self::Named(s) if s.eq_ignore_ascii_case("json") => Ok(Some(OutputFormat::Json)),
            Self::Named(s) if s.is_empty() => Ok(None),
            Self::Named(s) => Err(format!(
                "unsupported format {s:?} — this server understands \"json\""
            )),
            Self::Schema(schema) => Err(format!(
                "a JSON Schema in `format` ({} top-level keys) is not supported — this server \
                 can constrain output to JSON, but not to a given schema; use \"format\": \
                 \"json\" and validate the shape client-side",
                schema.as_object().map_or(0, serde_json::Map::len)
            )),
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
    #[serde(default)]
    format: Option<OllamaFormat>,
    /// Show a reasoning model's thinking, in `message.thinking`. `true` for
    /// a model that does not think is ollama's 400 (see [`allow_thinking`]).
    ///
    /// Absent means off. Real ollama turns it on for a thinking model, and
    /// mummu does not, on purpose: the model reasons either way — mummu does
    /// not render Qwen's no-think prompt — so the flag only decides what the
    /// client is shown, and a client that never asked is shown the answer
    /// alone. When a reply is all reasoning because the token cap cut it
    /// off, that client gets an error that says so (see `crate::think`)
    /// instead of an empty `content` it has no reason to look behind. The
    /// ollama CLI sends `true` to any model `/api/show` says thinks, so it
    /// gets the split either way.
    #[serde(default)]
    think: Option<bool>,
    /// Functions the model may call. Before these were read, serde dropped
    /// them: the model never saw them, and a client asking for a call got
    /// prose with no way to tell why.
    #[serde(default)]
    tools: Option<Vec<ToolDef>>,
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
    #[serde(default)]
    format: Option<OllamaFormat>,
    /// As on /api/chat, with the thinking in `thinking`.
    #[serde(default)]
    think: Option<bool>,
}

/// One entry of a request's `tools` array. Ollama and `OpenAI` spell it the
/// same way: `{"type": "function", "function": {name, description,
/// parameters}}`.
#[derive(Deserialize)]
pub struct ToolDef {
    #[serde(default)]
    function: Option<FunctionDef>,
}

#[derive(Deserialize)]
struct FunctionDef {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: serde_json::Value,
}

/// A request's tool definitions as the renderer takes them. An entry that is
/// not a function is skipped.
pub fn tool_specs(defs: &[ToolDef]) -> Vec<ToolSpec> {
    defs.iter()
        .filter_map(|t| t.function.as_ref())
        .map(|f| ToolSpec {
            name: f.name.clone(),
            description: f.description.clone(),
            parameters: if f.parameters.is_null() {
                json!({"type": "object", "properties": {}})
            } else {
                f.parameters.clone()
            },
        })
        .collect()
}

/// Offer `tools` to the model a plan runs, or say why it cannot take them —
/// in the words ollama's own server uses, so a client that recognises that
/// refusal and retries without tools still can.
pub fn offer_tools(p: &mut RunPlan, model: &str, tools: Vec<ToolSpec>) -> Result<(), String> {
    if !tools.is_empty() && !engine::supports_tools(p.spec.architecture) {
        return Err(format!("{model:?} does not support tools"));
    }
    p.tools = tools;
    Ok(())
}

/// Refuse a request to see the thinking of a model that does not think —
/// ollama's 400, in its words. `/api/show` reports "thinking" from the same
/// [`engine::thinks`], so a client that read it first is never refused.
/// Asking NOT to see it is never refused, as in ollama.
fn allow_thinking(p: &RunPlan, model: &str) -> Result<(), String> {
    if p.think && !engine::thinks(p.spec.architecture) {
        return Err(format!("{model:?} does not support thinking"));
    }
    Ok(())
}

/// Everything a chat/generate run needs after validation.
pub struct RunPlan {
    pub(crate) spec: ModelSpec,
    pub(crate) root: std::path::PathBuf,
    pub(crate) turns: Vec<mummu::chat::Turn>,
    pub(crate) opts: mummu::decode::SamplerOptions,
    pub(crate) max_tokens: usize,
    pub(crate) format: Option<OutputFormat>,
    pub(crate) images: Vec<mummu::vision::Patches>,
    /// Let the client see a reasoning model's thinking: the engine passes the
    /// `<think>` block through, and the surface decides where it goes —
    /// inline on `OpenAI`'s, ollama's `thinking` field on the shim's.
    pub(crate) think: bool,
    /// Tool definitions to advertise to the model (see [`offer_tools`]).
    pub(crate) tools: Vec<ToolSpec>,
}

/// Validate a request into a `RunPlan`, or hand back the error response.
pub fn plan(
    model: &str,
    messages: &[ChatMessage],
    options: &OllamaOptions,
    format: Option<&OllamaFormat>,
    think: bool,
) -> Result<RunPlan, Box<Response>> {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let Some(spec) = resolve(&manager, model) else {
        return Err(Box::new(not_found(model)));
    };
    if !engine::is_installed(&spec, &root) {
        return Err(Box::new(not_found(model)));
    }
    // Images are decoded and sized BEFORE the turns are rendered: the number
    // of placeholder tokens a prompt must reserve is a function of each
    // image's patch grid, so the prompt cannot be built until they are laid
    // out. `prepare_images` only reads the tower's header, not its weights.
    let raw = match decode_images(messages) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(json_response(400, &json!({"error": e})))),
    };
    let images = match engine::prepare_images(&spec, &root, &raw) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(json_response(400, &json!({"error": e})))),
    };
    let marks = match engine::placeholders(&spec, &root, &images) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(json_response(500, &json!({"error": e})))),
    };
    let messages = with_placeholders(messages, &marks);
    let turns = to_turns(&messages, spec.architecture)
        .map_err(|e| json_response(400, &json!({"error": e})))?;
    let opts = options
        .sampler()
        .map_err(|e| json_response(400, &json!({"error": e})))?;
    let max_tokens = options.max_tokens();
    let format = match format.map(OllamaFormat::resolve).transpose() {
        Ok(f) => f.flatten(),
        Err(e) => return Err(Box::new(json_response(400, &json!({"error": e})))),
    };
    Ok(RunPlan {
        spec,
        root,
        turns,
        opts,
        max_tokens,
        format,
        images,
        think,
        tools: Vec::new(),
    })
}

/// Every message's base64 image payloads, decoded in message order.
fn decode_images(messages: &[ChatMessage]) -> Result<Vec<Vec<u8>>, String> {
    use base64::Engine as _;
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        for (j, b64) in m.images.iter().enumerate() {
            // Some clients send a whole data: URI where the field wants the
            // payload alone; accept both rather than failing on a comma.
            let payload = b64.rsplit_once(',').map_or(b64.as_str(), |(_, p)| p);
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(payload.trim())
                .map_err(|e| format!("message {i} image {j} is not valid base64: {e}"))?;
            if bytes.is_empty() {
                return Err(format!("message {i} image {j} is empty"));
            }
            out.push(bytes);
        }
    }
    Ok(out)
}

/// Prefix each image-bearing message with its placeholder runs, in the same
/// order [`decode_images`] collected them.
fn with_placeholders(messages: &[ChatMessage], marks: &[String]) -> Vec<ChatMessage> {
    let mut next = 0;
    messages
        .iter()
        .map(|m| {
            let mut content = String::new();
            for _ in 0..m.images.len() {
                if let Some(mark) = marks.get(next) {
                    content.push_str(mark);
                }
                next += 1;
            }
            content.push_str(&m.content);
            ChatMessage {
                role: m.role.clone(),
                content,
                images: Vec::new(),
                tool_calls: Vec::new(),
            }
        })
        .collect()
}

/// Ollama's final frame: timing in nanoseconds.
fn done_value(model: &str, r: &engine::ChatResult, started: Instant) -> serde_json::Value {
    let total_ns = crate::nanos(started.elapsed());
    let eval_ns = r.elapsed_ms.saturating_mul(1_000_000);
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

/// Text on its way to an ollama client, a delta of it or the whole: the
/// answer, and the thinking — empty unless the request asked to see it.
#[derive(Clone, Copy, Default)]
struct Said<'a> {
    content: &'a str,
    thinking: &'a str,
}

/// Turns a piece of the answer into the endpoint's streamed frame:
/// `message` for /api/chat, `response` for /api/generate.
type Wrap = fn(&str, Said<'_>) -> serde_json::Value;

/// The endpoint's last line — the whole answer, when buffered.
type Finish = fn(&str, Said<'_>, &engine::ChatResult, Instant) -> serde_json::Value;

/// Ollama's `thinking`, beside `content` or `response`. Ollama omits it when
/// empty, and so does this: a client that never asked sees no new field.
fn with_thinking(mut v: serde_json::Value, thinking: &str) -> serde_json::Value {
    if !thinking.is_empty() {
        v["thinking"] = json!(thinking);
    }
    v
}

/// An assistant message in ollama's shape, with its thinking and the calls
/// it made.
fn assistant_message(said: Said<'_>, calls: &[ToolCall]) -> serde_json::Value {
    let mut message = with_thinking(
        json!({"role": "assistant", "content": said.content}),
        said.thinking,
    );
    if !calls.is_empty() {
        message["tool_calls"] = calls
            .iter()
            .enumerate()
            .map(|(i, c)| json!({"function": {"index": i, "name": c.name, "arguments": c.arguments}}))
            .collect();
    }
    message
}

/// Run one completion for the shim: streamed (one NDJSON frame per delta)
/// or buffered, with `wrap` turning a piece of the answer into the
/// endpoint's frame shape (`message` for /api/chat, `response` and
/// `thinking` for /api/generate).
async fn run(p: RunPlan, stream: bool, wrap: Wrap, finish: Finish) -> Response {
    // One line per accepted request, before any engine work: a request that
    // queues behind a cold model load produces nothing for its whole wait,
    // and a surface that logs nothing while that happens reads as wedged
    // (it did, live, 2026-08-28).
    eprintln!(
        "[mummu-serve] shim chat {}: request accepted (stream = {stream})",
        p.spec.name
    );
    let begin = crate::trace::Begin {
        surface: "ollama",
        model: p.spec.name.clone(),
        asked: crate::trace::Asked {
            stream,
            think: p.think,
            json_mode: p.format.is_some(),
        },
        tools: p.tools.len(),
        images: p.images.len(),
        started: Instant::now(),
    };
    // The process is exiting to restart the GPU backend (see `recovery`).
    if recovery::restarting() {
        return json_response(503, &json!({"error": recovery::restarting_message()}));
    }
    if !stream && let Some(dir) = engine::load_in_flight() {
        // A non-stream response sends no bytes until the whole generation is
        // done. Parked behind a model load — an hour on this disk — that is
        // indistinguishable from a dead server, so refuse it honestly now.
        let loading = dir.file_name().map_or_else(
            || dir.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        eprintln!(
            "[mummu-serve] shim chat {}: 503 — the {loading} load is in flight",
            p.spec.name
        );
        begin.finish(
            "",
            crate::trace::Timings::default(),
            false,
            Err(&format!("503: the {loading} load is in flight")),
        );
        return json_response(
            503,
            &json!({"error": format!(
                "a model is loading ({loading}); a non-streaming request would wait \
                 silently for the whole load — retry once it completes, or set \"stream\": true"
            )}),
        );
    }
    let model = p.spec.name.clone();
    // `offer_tools` let tools through only to a family with a convention.
    let calls = if p.tools.is_empty() {
        None
    } else {
        engine::tool_calls(p.spec.architecture)
    };
    let think = p.think;
    Box::pin(respond(
        model,
        stream,
        think,
        calls,
        wrap,
        finish,
        move |sink| async move {
            let req = engine::GenerationRequest {
                spec: &p.spec,
                models_root: &p.root,
                turns: &p.turns,
                opts: &p.opts,
                max_tokens: p.max_tokens,
                format: p.format,
                think: p.think,
                images: p.images,
                tools: p.tools,
            };
            let r = engine::run_chat(&req, |delta| sink.delta(delta)).await;
            // Padded exactly when a buffered answer outlived the keep-alive
            // grace — the condition `keepalive_json` pads on.
            let padded = !stream && begin.started.elapsed() >= crate::KEEPALIVE_GRACE;
            match &r {
                Ok(res) => begin.finish(res.device, res.timings.clone(), padded, Ok(())),
                Err(e) => begin.finish(
                    "",
                    crate::trace::Timings::default(),
                    padded,
                    Err(&e.message),
                ),
            }
            r
        },
    ))
    .await
}

/// Splits an answer that was allowed to think into ollama's `thinking` and
/// `content`, the way ollama's own parser does (`thinking/parser.go`): the
/// whitespace between `<think>` and the reasoning, and between `</think>`
/// and the answer, belongs to neither — so `content` does not open on the
/// blank line Qwen writes after `</think>`. The trailing whitespace of the
/// reasoning is kept, as there. A streamed answer and a buffered one go
/// through the same splitter, so the two cannot disagree.
///
/// The answer's side of that is the think filter's own rule, the one a
/// request that did not ask to see the thinking gets too (see
/// `crate::think`); only the thinking's side is trimmed here.
#[derive(Default)]
struct Reasoning {
    filter: Filter,
    /// Some thinking has been shown: its whitespace is its own from here on.
    thinking: bool,
}

impl Reasoning {
    /// One whole answer, split.
    fn split(text: &str) -> Split {
        let mut r = Self::default();
        let mut whole = r.push(text);
        let tail = r.finish();
        whole.visible.push_str(&tail.visible);
        whole.thought.push_str(&tail.thought);
        whole
    }

    fn push(&mut self, delta: &str) -> Split {
        let split = self.filter.push_split(delta);
        self.shape(split)
    }

    /// Whatever is still held at the end. A block the token cap cut off is
    /// thinking all the same, and goes out as `thinking`.
    fn finish(&mut self) -> Split {
        let split = self.filter.finish_split();
        self.shape(split)
    }

    fn shape(&mut self, split: Split) -> Split {
        let Split {
            visible,
            mut thought,
        } = split;
        if !self.thinking {
            thought = thought.trim_start().to_owned();
            self.thinking = !thought.is_empty();
        }
        Split { visible, thought }
    }
}

/// What a streamed answer holds back between deltas, and why.
struct Held {
    /// The request asked to see the thinking: it is split out of the answer
    /// into ollama's `thinking` field.
    reasoning: Option<Reasoning>,
    /// The request offered tools: the family's call markup is held back from
    /// the answer. The calls go out structured, and whole, once the answer
    /// is — see [`tool_tail`]. Only the answer is looked in: markup inside
    /// the thinking is thinking, which is also all the engine reads calls
    /// from (`engine::lift_tool_calls`).
    calls: Option<Filter>,
}

impl Held {
    /// One delta, as the client may see it now.
    fn push(&mut self, delta: &str) -> Split {
        let Split { visible, thought } = self.reasoning.as_mut().map_or_else(
            || Split {
                visible: delta.to_owned(),
                thought: String::new(),
            },
            |r| r.push(delta),
        );
        let visible = match &mut self.calls {
            Some(f) => f.push(&visible),
            None => visible,
        };
        Split { visible, thought }
    }

    /// What the client is still owed once the model is done, before the
    /// final line: the thinking and answer text still held, then the calls.
    fn tail(
        &mut self,
        model: &str,
        wrap: Wrap,
        r: &mut engine::ChatResult,
    ) -> Vec<serde_json::Value> {
        let mut frames = Vec::new();
        if let Some(reasoning) = &mut self.reasoning {
            let Split { visible, thought } = reasoning.finish();
            let content = match &mut self.calls {
                Some(f) => f.push(&visible),
                None => visible,
            };
            if !content.is_empty() || !thought.is_empty() {
                frames.push(wrap(
                    model,
                    Said {
                        content: &content,
                        thinking: &thought,
                    },
                ));
            }
        }
        if let Some(calls) = &mut self.calls {
            frames.extend(tool_tail(model, wrap, calls, r));
        }
        frames
    }
}

/// Where a shim generation's text goes: one NDJSON line per piece when
/// streaming, nowhere when the answer is buffered.
struct ShimSink {
    tx: Option<mpsc::UnboundedSender<serde_json::Value>>,
    model: String,
    wrap: Wrap,
    /// What the answer holds back between deltas; `None` when the request
    /// neither asked to see the thinking nor offered tools.
    held: Option<Arc<Mutex<Held>>>,
}

impl ShimSink {
    fn delta(&self, text: &str) -> ControlFlow<()> {
        let Some(tx) = &self.tx else {
            return ControlFlow::Continue(());
        };
        let split = self.held.as_ref().map_or_else(
            || Split {
                visible: text.to_owned(),
                thought: String::new(),
            },
            |h| {
                h.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(text)
            },
        );
        if split.visible.is_empty() && split.thought.is_empty() {
            return ControlFlow::Continue(());
        }
        let said = Said {
            content: &split.visible,
            thinking: &split.thought,
        };
        match tx.send((self.wrap)(&self.model, said)) {
            Ok(()) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        }
    }
}

/// What a streamed answer to a request that offered tools still owes its
/// client once the model is done. Ollama's shape: the calls in a frame of
/// their own before the final line — taken out of `r`, so the final line
/// does not repeat them. When the model made none, or wrote markup that did
/// not parse, whatever was held back goes out as ordinary text instead:
/// nothing the model wrote goes missing.
fn tool_tail(
    model: &str,
    wrap: Wrap,
    held: &mut Filter,
    r: &mut engine::ChatResult,
) -> Vec<serde_json::Value> {
    let text = held.settle(!r.tool_calls.is_empty());
    let mut frames = Vec::new();
    if !text.is_empty() {
        frames.push(wrap(
            model,
            Said {
                content: &text,
                thinking: "",
            },
        ));
    }
    if !r.tool_calls.is_empty() {
        frames.push(json!({
            "model": model,
            "created_at": now_rfc3339(),
            "message": assistant_message(Said::default(), &std::mem::take(&mut r.tool_calls)),
            "done": false,
        }));
    }
    frames
}

/// Turn one generation into the shim's response — the ONE place its outcome
/// becomes ollama's wire shape, streamed or not.
///
/// The generation runs under [`recovery::contain`]: a panic in it comes back
/// as an error instead of taking the response down with it, and a GPU
/// failure is acted on. A streamed answer then always ends on a final line —
/// `done: true`, or ollama's `{"error": "…"}` — and a buffered one on a
/// non-2xx with `{"error": "…"}`: 503 for the GPU, 500 for anything else.
/// `run` is the generation; production passes `engine::run_chat`, a test one
/// that fails the way production did. `think` is the request's: the engine
/// then passes the `<think>` block through, and it is split out of the
/// answer here, into ollama's `thinking` (see [`Reasoning`]). `calls` is the
/// family's tool-call convention when the request offered tools: a streamed
/// answer holds that markup back and sends the calls structured instead
/// (see [`tool_tail`]).
async fn respond<F, Fut>(
    model: String,
    stream: bool,
    think: bool,
    calls: Option<CallSyntax>,
    wrap: Wrap,
    finish: Finish,
    run: F,
) -> Response
where
    F: FnOnce(ShimSink) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<engine::ChatResult, ChatError>> + Send,
{
    let started = Instant::now();
    if stream {
        let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();
        let inflight = InFlight::enter();
        tokio::spawn(async move {
            let last = FinalFrame::new(tx.clone(), ended_without_result());
            let held = (think || calls.is_some()).then(|| {
                Arc::new(Mutex::new(Held {
                    reasoning: think.then(Reasoning::default),
                    calls: calls.map(|c| Filter::spans(c.open, c.close)),
                }))
            });
            let sink = ShimSink {
                tx: Some(tx.clone()),
                model: model.clone(),
                wrap,
                held: held.clone(),
            };
            let line = match recovery::contain(&model, run(sink)).await {
                Ok(mut r) => {
                    if let Some(held) = held {
                        let mut held = held
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        for frame in held.tail(&model, wrap, &mut r) {
                            let _ = tx.send(frame);
                        }
                    }
                    finish(&model, Said::default(), &r, started)
                }
                Err(e) => {
                    eprintln!("[mummu-serve] shim chat {model}: {e}");
                    error_line(&e)
                }
            };
            last.send(line);
        });
        return ndjson_response(rx, Some(inflight));
    }
    // Non-stream: run to completion, answer with one object — under the
    // keep-alive, because a buffered answer that sends nothing for minutes
    // is what a proxy reports as a dead origin (see `crate::keepalive_json`).
    crate::keepalive_json(async move {
        let _inflight = InFlight::enter();
        let sink = ShimSink {
            tx: None,
            model: model.clone(),
            wrap,
            held: None,
        };
        match recovery::contain(&model, run(sink)).await {
            Ok(r) => {
                let whole = if think {
                    Reasoning::split(&r.text)
                } else {
                    Split {
                        visible: r.text.clone(),
                        thought: String::new(),
                    }
                };
                let said = Said {
                    content: &whole.visible,
                    thinking: &whole.thought,
                };
                (200, finish(&model, said, &r, started))
            }
            Err(e) => {
                eprintln!("[mummu-serve] shim chat {model}: {e}");
                (e.http_status(), error_line(&e))
            }
        }
    })
    .await
}

pub async fn chat(body: Bytes) -> Response {
    let parsed: OllamaChatRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    let mut p = match plan(
        &parsed.model,
        &parsed.messages,
        &parsed.options,
        parsed.format.as_ref(),
        parsed.think.unwrap_or(false),
    ) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    if let Err(e) = allow_thinking(&p, &parsed.model) {
        return json_response(400, &json!({"error": e}));
    }
    let tools = tool_specs(parsed.tools.as_deref().unwrap_or_default());
    if let Err(e) = offer_tools(&mut p, &parsed.model, tools) {
        return json_response(400, &json!({"error": e}));
    }
    Box::pin(run(p, parsed.stream.unwrap_or(true), chat_delta, chat_done)).await
}

/// One streamed piece of an `/api/chat` answer.
fn chat_delta(model: &str, said: Said<'_>) -> serde_json::Value {
    json!({
        "model": model,
        "created_at": now_rfc3339(),
        "message": assistant_message(said, &[]),
        "done": false,
    })
}

/// The last line of an `/api/chat` answer — the whole answer, when buffered.
fn chat_done(
    model: &str,
    said: Said<'_>,
    r: &engine::ChatResult,
    started: Instant,
) -> serde_json::Value {
    let mut v = done_value(model, r, started);
    v["message"] = assistant_message(said, &r.tool_calls);
    v
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
            images: Vec::new(),
            tool_calls: Vec::new(),
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: parsed.prompt.clone(),
        images: Vec::new(),
        tool_calls: Vec::new(),
    });
    let p = match plan(
        &parsed.model,
        &messages,
        &parsed.options,
        parsed.format.as_ref(),
        parsed.think.unwrap_or(false),
    ) {
        Ok(p) => p,
        Err(response) => return *response,
    };
    if let Err(e) = allow_thinking(&p, &parsed.model) {
        return json_response(400, &json!({"error": e}));
    }
    Box::pin(run(
        p,
        parsed.stream.unwrap_or(true),
        generate_delta,
        generate_done,
    ))
    .await
}

/// One streamed piece of an `/api/generate` answer.
fn generate_delta(model: &str, said: Said<'_>) -> serde_json::Value {
    with_thinking(
        json!({
            "model": model,
            "created_at": now_rfc3339(),
            "response": said.content,
            "done": false,
        }),
        said.thinking,
    )
}

/// The last line of an `/api/generate` answer — the whole answer, when
/// buffered.
fn generate_done(
    model: &str,
    said: Said<'_>,
    r: &engine::ChatResult,
    started: Instant,
) -> serde_json::Value {
    let mut v = done_value(model, r, started);
    v["response"] = json!(said.content);
    with_thinking(v, said.thinking)
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
            &json!({"error": format!(
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
                Ok(_) => json_response(200, &json!({"status": "success"})),
                Err(e) => json_response(500, &json!({"error": e})),
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
                trunc_i64((f64_from_u64(p.received_bytes) / f64_from_u64(total)) * 100.0)
            } else {
                (p.received_bytes >> 26).cast_signed()
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
    ndjson_response(rx, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The read that failed every chat after the 2026-09-18 load.
    const INVALID_READ: &str = "bytes: host access failed: Read(\"The server is in an invalid \
                                state\\nCaused by:\\n  An IO error happened\\nCaused by:\\n  \
                                couldn't find resource for that handle: Memory location was \
                                never initialized\")";

    /// Called from inside the generation's own future, so the panic lands
    /// where production's did.
    fn fails_like_production() -> Result<engine::ChatResult, ChatError> {
        panic!("{INVALID_READ}")
    }

    fn fails_with_an_ordinary_bug() -> Result<engine::ChatResult, ChatError> {
        panic!("called `Option::unwrap()` on a `None` value")
    }

    fn wrap(model: &str, said: Said<'_>) -> serde_json::Value {
        json!({"model": model, "response": said.content, "done": false})
    }

    fn finish(
        model: &str,
        said: Said<'_>,
        r: &engine::ChatResult,
        started: Instant,
    ) -> serde_json::Value {
        let mut v = done_value(model, r, started);
        v["response"] = json!(said.content);
        v
    }

    async fn body_text(response: Response) -> String {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        String::from_utf8_lossy(&body).into_owned()
    }

    /// Streaming NDJSON: ollama's own convention for a failure mid-stream is
    /// a last line `{"error": "…"}`. The incident's clients got a stream that
    /// ended on nothing.
    #[tokio::test]
    async fn a_streamed_shim_chat_that_hits_the_gpu_failure_ends_on_an_error_line() {
        let _serial = crate::progress_serial().await;
        recovery::reset_for_tests();
        recovery::install_panic_hook();

        let response = respond("m".into(), true, false, None, wrap, finish, |_| async {
            fails_like_production()
        })
        .await;
        let text = body_text(response).await;
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("each line is JSON"))
            .collect();
        let last = lines.last().expect("the stream is not empty");
        let error = last["error"].as_str().expect("ollama's error line");
        assert!(error.contains("GPU backend failed"), "{error}");
        assert_eq!(
            lines.iter().filter(|l| is_final_line(l)).count(),
            1,
            "{text}"
        );
        recovery::reset_for_tests();
    }

    /// Buffered: a non-2xx carrying `{"error": "…"}` — 503 for the GPU,
    /// because retrying is the right thing to do, and 500 for a plain bug.
    #[tokio::test]
    async fn a_buffered_shim_chat_answers_non_2xx_with_the_error() {
        let _serial = crate::progress_serial().await;
        recovery::reset_for_tests();
        recovery::install_panic_hook();

        let response = respond("m".into(), false, false, None, wrap, finish, |_| async {
            fails_like_production()
        })
        .await;
        assert_eq!(response.status(), 503);
        let body: serde_json::Value =
            serde_json::from_str(&body_text(response).await).expect("JSON");
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|e| e.contains("GPU backend failed")),
            "{body}"
        );

        recovery::reset_for_tests();
        let response = respond("m".into(), false, false, None, wrap, finish, |_| async {
            fails_with_an_ordinary_bug()
        })
        .await;
        assert_eq!(response.status(), 500);
        assert!(!recovery::poisoned(), "a bug is not a poisoned GPU");
        recovery::reset_for_tests();
    }

    // -- capabilities and tools ---------------------------------------------

    /// A models root of the test's own, empty unless the test fills it — so
    /// nothing here can touch real weights.
    fn scratch_root(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "mummu-shim-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch root");
        root
    }

    fn catalog_spec(name: &str) -> ModelSpec {
        mummu::registry::catalog()
            .into_iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("{name} is in the catalog"))
    }

    fn a_plan_for(spec: ModelSpec, root: &Path) -> RunPlan {
        RunPlan {
            spec,
            root: root.to_path_buf(),
            turns: vec![mummu::chat::Turn::user("hi")],
            opts: mummu::decode::SamplerOptions::default(),
            max_tokens: 16,
            format: None,
            images: Vec::new(),
            think: false,
            tools: Vec::new(),
        }
    }

    fn weather_tool() -> Vec<ToolSpec> {
        tool_specs(
            &serde_json::from_value::<Vec<ToolDef>>(json!([{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }]))
            .expect("ollama's tool shape parses"),
        )
    }

    fn weather_call() -> ToolCall {
        ToolCall {
            name: "get_weather".into(),
            arguments: json!({"city": "Paris"}),
        }
    }

    fn answered(text: &str, tool_calls: Vec<ToolCall>) -> engine::ChatResult {
        engine::ChatResult {
            text: text.into(),
            tokens: 1,
            device: "test",
            elapsed_ms: 0,
            timings: crate::trace::Timings::default(),
            tool_calls,
        }
    }

    fn ndjson(text: &str) -> Vec<serde_json::Value> {
        text.lines()
            .map(|l| serde_json::from_str(l).expect("each line is JSON"))
            .collect()
    }

    /// The hard-coded `["completion"]` this replaced told every client the
    /// same thing about every model. Each family's answer is pinned here,
    /// for every catalog entry, in the strings and order real ollama uses
    /// (checked 2026-09-21 against ollama's `server/images.go` and its
    /// registry: qwen2.5 → completion, tools; qwen3 → completion, tools,
    /// thinking; qwen3.5 → completion, vision, tools, thinking; all-minilm →
    /// embedding).
    #[test]
    fn show_reports_what_a_request_to_each_catalog_model_gets() {
        let root = scratch_root("caps");
        for spec in mummu::registry::catalog() {
            let expected: &[&str] = match spec.architecture {
                // LFM2's Pythonic calls are read back like Hermes ones, in
                // both builds (see `engine::preserved_tokens`).
                Architecture::Qwen2 | Architecture::Lfm2 => &["completion", "tools"],
                Architecture::Qwen3 | Architecture::Qwen35 => &["completion", "tools", "thinking"],
                // OLMoE's Tulu template has no tool convention at all.
                Architecture::Olmoe => &["completion"],
                Architecture::MiniLm => &["embedding"],
            };
            assert_eq!(capabilities(&spec, &root), expected, "{}", spec.name);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Vision is the `mmproj-*.gguf` beside the weights, and only for the
    /// family whose tower mummu runs — a Qwen2.5 directory holding one does
    /// not make it see.
    #[test]
    fn an_mmproj_beside_a_qwen35_is_vision_and_beside_anything_else_is_not() {
        let root = scratch_root("vision");
        for name in ["qwen3.5-2b", "qwen2.5-1.5b-instruct"] {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).expect("model dir");
            std::fs::write(dir.join("mmproj-F16.gguf"), b"").expect("mmproj");
        }
        assert_eq!(
            capabilities(&catalog_spec("qwen3.5-2b"), &root),
            ["completion", "vision", "tools", "thinking"],
            "ollama's own list for qwen3.5, in its order"
        );
        assert_eq!(
            capabilities(&catalog_spec("qwen2.5-1.5b-instruct"), &root),
            ["completion", "tools"]
        );
        // The same Qwen3.5 without its tower is text-only.
        assert!(!capabilities(&catalog_spec("qwen3.5-2b-q8"), &root).contains(&"vision"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `/api/show` advertises tools exactly where a request carrying them is
    /// accepted — both answers come from `engine::supports_tools` — and the
    /// refusal is ollama's own sentence, which clients match on to retry
    /// without tools.
    #[test]
    fn tools_are_advertised_exactly_where_a_request_may_carry_them() {
        let root = scratch_root("offer");
        for spec in mummu::registry::catalog() {
            if spec.architecture == Architecture::MiniLm {
                continue; // not chat-servable: `plan` never gets this far
            }
            let name = spec.name.clone();
            let advertised = capabilities(&spec, &root).contains(&"tools");
            let mut p = a_plan_for(spec, &root);
            match offer_tools(&mut p, &name, weather_tool()) {
                Ok(()) => {
                    assert!(advertised, "{name} took tools /api/show does not advertise");
                    assert_eq!(p.tools.len(), 1, "{name}");
                }
                Err(e) => {
                    assert!(!advertised, "{name} refused tools /api/show advertises");
                    assert_eq!(e, format!("{name:?} does not support tools"));
                    assert!(p.tools.is_empty(), "{name}");
                }
            }
            // No tools is never a refusal.
            let mut p = a_plan_for(catalog_spec(&name), &root);
            assert!(offer_tools(&mut p, &name, Vec::new()).is_ok(), "{name}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `/api/chat` reads `tools` now. Before, serde dropped the field: the
    /// model never saw the functions and the client got prose.
    #[test]
    fn an_ollama_chat_request_carries_its_tools() {
        let parsed: OllamaChatRequest = serde_json::from_value(json!({
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": null}},
                {"type": "not-a-function"},
            ],
        }))
        .expect("request parses");
        let tools = tool_specs(parsed.tools.as_deref().unwrap_or_default());
        assert_eq!(tools.len(), 1, "an entry with no function is skipped");
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(
            tools[0].parameters,
            json!({"type": "object", "properties": {}}),
            "a null schema becomes an empty object schema"
        );
    }

    /// Ollama's clients replay a tool loop with their own call shape and,
    /// often, no `content` on the assistant turn. Both were a 400 before.
    #[test]
    fn a_replayed_tool_loop_parses_in_ollamas_shape_and_the_flat_one() {
        let messages: Vec<ChatMessage> = serde_json::from_value(json!([
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                {"function": {"index": 0, "name": "get_weather", "arguments": {"city": "Paris"}}},
            ]},
            {"role": "tool", "content": "18C, clear", "tool_name": "get_weather"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "Paris"}},
            ]},
        ]))
        .expect("messages parse");
        assert_eq!(messages[1].content, "");
        assert_eq!(messages[1].tool_calls, [weather_call()]);
        assert_eq!(messages[3].content, "", "null content is empty content");
        assert_eq!(messages[3].tool_calls, [weather_call()]);
        let turns = to_turns(&messages[..3], Architecture::Qwen3).expect("the loop renders");
        assert_eq!(turns.len(), 3);
    }

    /// The model is shown the call it made the way it made it: Hermes JSON
    /// for Qwen, LFM's Pythonic list for LFM2 — which is shown a turn it
    /// never wrote if it gets Hermes, with or without tools on the request.
    #[test]
    fn a_replayed_call_is_written_back_in_the_familys_own_syntax() {
        let messages: Vec<ChatMessage> = serde_json::from_value(json!([
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
            ]},
            {"role": "tool", "content": "18C, clear"},
        ]))
        .expect("messages parse");
        let lfm = to_turns(&messages, Architecture::Lfm2).expect("LFM2 replays");
        assert_eq!(
            lfm[1].content,
            "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>"
        );
        assert_eq!(lfm[1].tool_calls, [weather_call()]);
        let qwen = to_turns(&messages, Architecture::Qwen3).expect("Qwen3 replays");
        assert_eq!(
            qwen[1].content,
            mummu::chat::Turn::assistant_tool_calls(&[weather_call()]).content
        );
        // No convention of its own: Hermes, which at least reads as a call.
        let olmoe = to_turns(&messages, Architecture::Olmoe).expect("renders");
        assert_eq!(olmoe[1].content, qwen[1].content);
    }

    /// A replayed call is whatever the client sent, and the renderers assert
    /// their input — an LFM2 history with a string for arguments panicked
    /// mid-request. Each is refused as the client's mistake instead, for
    /// every family, and a JSON-encoded object (`OpenAI`'s spelling) is read.
    #[test]
    fn a_replayed_call_the_renderers_cannot_take_is_refused_not_panicked_on() {
        let with_calls = |calls: serde_json::Value| -> Vec<ChatMessage> {
            serde_json::from_value(json!([
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "tool_calls": calls},
                {"role": "tool", "content": "18C, clear"},
            ]))
            .expect("messages parse")
        };
        // As deep as the Pythonic renderer goes, and one level past it.
        let mut at_bound = json!("bottom");
        for _ in 0..mummu::chat::MAX_VALUE_DEPTH {
            at_bound = json!([at_bound]);
        }
        let deep = json!([at_bound]);
        let too_many: Vec<_> = (0..=mummu::chat::MAX_TOOL_CALLS)
            .map(|_| json!({"name": "get_weather", "arguments": {}}))
            .collect();
        let refused = [
            (
                json!([{"name": "get_weather", "arguments": "{not json"}]),
                "not JSON",
            ),
            (
                json!([{"name": "get_weather", "arguments": "\"Paris\""}]),
                "must be an object",
            ),
            (
                json!([{"name": "get_weather", "arguments": ["Paris"]}]),
                "must be an object",
            ),
            (json!([{"name": "", "arguments": {}}]), "no name"),
            (
                json!([{"name": "f", "arguments": {"x": deep}}]),
                "deeper than",
            ),
            (json!(too_many), "more than"),
        ];
        for arch in [
            Architecture::Qwen2,
            Architecture::Qwen3,
            Architecture::Lfm2,
            Architecture::Olmoe,
        ] {
            for (calls, why) in &refused {
                let e = to_turns(&with_calls(calls.clone()), arch).expect_err(why);
                assert!(e.contains(why), "{arch:?}: {e}");
                assert!(e.starts_with("message 1: "), "{arch:?}: {e}");
            }
            let encoded = with_calls(json!([
                {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                {"name": "get_weather", "arguments": null},
                {"name": "f", "arguments": {"x": at_bound.clone()}},
            ]));
            let turns = to_turns(&encoded, arch).expect("a JSON-encoded object is read");
            assert_eq!(turns[1].tool_calls[0], weather_call(), "{arch:?}");
            assert_eq!(turns[1].tool_calls.len(), 3, "{arch:?}");
        }
    }

    /// Streamed, a call goes out the way ollama sends one: the markup never
    /// reaches `content`, even split across deltas, and the call arrives
    /// structured in a frame of its own before the final line — which does
    /// not repeat it.
    #[tokio::test]
    async fn a_streamed_tool_call_arrives_structured_and_its_markup_never_does() {
        let _serial = crate::progress_serial().await;
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let response = respond(
            "m".into(),
            true,
            false,
            hermes,
            chat_delta,
            chat_done,
            |sink| async move {
                let _ = sink.delta("Checking. <tool");
                let _ = sink.delta("_call>{\"name\": \"get_weather\", ");
                let _ = sink.delta("\"arguments\": {\"city\": \"Paris\"}}</tool_call>");
                // What the engine hands back once it has lifted the call.
                Ok(answered("Checking.", vec![weather_call()]))
            },
        )
        .await;
        let lines = ndjson(&body_text(response).await);
        let content: String = lines
            .iter()
            .filter_map(|l| l["message"]["content"].as_str())
            .collect();
        assert_eq!(content, "Checking. ", "no markup in content: {lines:?}");
        let calls: Vec<_> = lines
            .iter()
            .filter(|l| l["message"].get("tool_calls").is_some())
            .collect();
        assert_eq!(calls.len(), 1, "one frame carries the calls: {lines:?}");
        assert_eq!(calls[0]["done"], json!(false));
        assert_eq!(
            calls[0]["message"]["tool_calls"],
            json!([{"function": {"index": 0, "name": "get_weather", "arguments": {"city": "Paris"}}}])
        );
        let last = lines.last().expect("lines");
        assert_eq!(last["done"], json!(true));
        assert!(last["message"].get("tool_calls").is_none(), "{last}");
    }

    /// Markup that did not parse is not swallowed: what was held back goes
    /// out as text, so the client sees everything the model wrote.
    #[tokio::test]
    async fn held_back_markup_that_was_not_a_call_is_released_as_text() {
        let _serial = crate::progress_serial().await;
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let raw = "Hmm <tool_call>{not json</tool_call> then <tool_call>{\"name\": \"cut";
        let response = respond(
            "m".into(),
            true,
            false,
            hermes,
            chat_delta,
            chat_done,
            move |sink| async move {
                let _ = sink.delta(raw);
                // The engine could not lift anything, so the text is verbatim.
                Ok(answered(raw, Vec::new()))
            },
        )
        .await;
        let lines = ndjson(&body_text(response).await);
        let content: String = lines
            .iter()
            .filter_map(|l| l["message"]["content"].as_str())
            .collect();
        assert_eq!(
            content,
            "Hmm  then <tool_call>{not json</tool_call><tool_call>{\"name\": \"cut"
        );
        assert!(
            lines
                .iter()
                .all(|l| l["message"].get("tool_calls").is_none())
        );
    }

    /// Buffered, the one object carries the prose and the calls together.
    #[tokio::test]
    async fn a_buffered_tool_call_comes_back_in_the_message() {
        let _serial = crate::progress_serial().await;
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let response = respond(
            "m".into(),
            false,
            false,
            hermes,
            chat_delta,
            chat_done,
            |_| async { Ok(answered("", vec![weather_call()])) },
        )
        .await;
        assert_eq!(response.status(), 200);
        let body: serde_json::Value =
            serde_json::from_str(&body_text(response).await).expect("JSON");
        assert_eq!(body["done"], json!(true));
        assert_eq!(body["message"]["content"], json!(""));
        assert_eq!(
            body["message"]["tool_calls"][0]["function"]["name"],
            json!("get_weather")
        );
    }

    // -- thinking -------------------------------------------------------------

    /// A Qwen3 answer as the engine streams it to a request that asked to see
    /// the thinking: the block passes through, a token-shaped piece at a
    /// time, both tags split across pieces.
    const THOUGHT_DELTAS: [&str; 8] = [
        "<th",
        "ink>",
        "\nThe user",
        " said hi.",
        "\n</th",
        "ink>",
        "\n\n",
        "Hello!",
    ];
    /// The same answer whole, as the engine hands it back.
    const THOUGHT: &str = "<think>\nThe user said hi.\n</think>\n\nHello!";

    /// One field of every frame, joined in order.
    fn joined(
        lines: &[serde_json::Value],
        pick: impl Fn(&serde_json::Value) -> Option<&str>,
    ) -> String {
        lines.iter().filter_map(pick).collect()
    }

    /// Streamed, in ollama's shape: the reasoning in `message.thinking` as it
    /// arrives, the answer in `message.content`, no tag in either, and not
    /// the blank line Qwen writes after `</think>` — what ollama's own
    /// parser makes of the same tokens.
    #[tokio::test]
    async fn a_streamed_answer_that_thinks_splits_into_thinking_and_content() {
        let _serial = crate::progress_serial().await;
        let response = respond(
            "m".into(),
            true,
            true,
            None,
            chat_delta,
            chat_done,
            |sink| async move {
                for d in THOUGHT_DELTAS {
                    let _ = sink.delta(d);
                }
                Ok(answered(THOUGHT, Vec::new()))
            },
        )
        .await;
        let lines = ndjson(&body_text(response).await);
        let (last, pieces) = lines.split_last().expect("lines");
        assert_eq!(
            joined(pieces, |l| l["message"]["thinking"].as_str()),
            "The user said hi.\n"
        );
        assert_eq!(
            joined(pieces, |l| l["message"]["content"].as_str()),
            "Hello!"
        );
        assert!(
            pieces
                .iter()
                .filter(|l| l["message"].get("thinking").is_some())
                .count()
                > 1,
            "the thinking streams as it comes, not in one piece at the end: {lines:?}"
        );
        for l in pieces {
            assert_eq!(l["done"], json!(false));
            assert!(!l.to_string().contains("think>"), "a tag leaked: {l}");
            assert!(
                l["message"]["thinking"]
                    .as_str()
                    .is_none_or(|t| !t.is_empty()),
                "an empty `thinking` is omitted, as ollama omits it: {l}"
            );
        }
        assert_eq!(last["done"], json!(true));
        assert_eq!(last["message"]["content"], json!(""));
        assert!(last["message"].get("thinking").is_none(), "{last}");
    }

    /// Buffered, the one object carries both fields.
    #[tokio::test]
    async fn a_buffered_answer_that_thinks_carries_both_fields() {
        let _serial = crate::progress_serial().await;
        let response = respond(
            "m".into(),
            false,
            true,
            None,
            chat_delta,
            chat_done,
            |_| async { Ok(answered(THOUGHT, Vec::new())) },
        )
        .await;
        assert_eq!(response.status(), 200);
        let body: serde_json::Value =
            serde_json::from_str(&body_text(response).await).expect("JSON");
        assert_eq!(body["message"]["thinking"], json!("The user said hi.\n"));
        assert_eq!(body["message"]["content"], json!("Hello!"));
    }

    /// /api/generate is the same split in its own fields: `thinking` beside
    /// `response`, streamed and buffered.
    #[tokio::test]
    async fn generate_puts_the_thinking_beside_the_response() {
        let _serial = crate::progress_serial().await;
        let response = respond(
            "m".into(),
            true,
            true,
            None,
            generate_delta,
            generate_done,
            |sink| async move {
                for d in THOUGHT_DELTAS {
                    let _ = sink.delta(d);
                }
                Ok(answered(THOUGHT, Vec::new()))
            },
        )
        .await;
        let lines = ndjson(&body_text(response).await);
        assert_eq!(
            joined(&lines, |l| l["thinking"].as_str()),
            "The user said hi.\n"
        );
        assert_eq!(joined(&lines, |l| l["response"].as_str()), "Hello!");
        assert!(
            lines.iter().all(|l| l.get("message").is_none()),
            "{lines:?}"
        );

        let response = respond(
            "m".into(),
            false,
            true,
            None,
            generate_delta,
            generate_done,
            |_| async { Ok(answered(THOUGHT, Vec::new())) },
        )
        .await;
        let body: serde_json::Value =
            serde_json::from_str(&body_text(response).await).expect("JSON");
        assert_eq!(body["thinking"], json!("The user said hi.\n"));
        assert_eq!(body["response"], json!("Hello!"));
    }

    /// A request that did not ask gets what the engine hands it, untouched —
    /// no `thinking` field, and none of the whitespace the split eats.
    #[tokio::test]
    async fn an_answer_that_did_not_ask_is_passed_through_as_it_was() {
        let _serial = crate::progress_serial().await;
        let response = respond(
            "m".into(),
            true,
            false,
            None,
            chat_delta,
            chat_done,
            |sink| async move {
                let _ = sink.delta("\n\n");
                let _ = sink.delta("Hello!");
                Ok(answered("\n\nHello!", Vec::new()))
            },
        )
        .await;
        let lines = ndjson(&body_text(response).await);
        assert_eq!(
            joined(&lines, |l| l["message"]["content"].as_str()),
            "\n\nHello!"
        );
        assert!(
            lines.iter().all(|l| l["message"].get("thinking").is_none()),
            "{lines:?}"
        );
    }

    /// A block the token cap cut off is reasoning all the same: it goes out
    /// as `thinking`, to the model's last byte, with an empty `content` —
    /// what ollama answers. (A request that did not ask gets the engine's
    /// error instead, which says to raise the cap; see `crate::think`.)
    #[tokio::test]
    async fn thinking_cut_off_by_the_token_cap_is_still_thinking() {
        let _serial = crate::progress_serial().await;
        let cut = "<think>\nStep one, step two</thi";
        let response = respond(
            "m".into(),
            true,
            true,
            None,
            chat_delta,
            chat_done,
            move |sink| async move {
                for d in ["<think>\nStep one", ", step tw", "o</thi"] {
                    let _ = sink.delta(d);
                }
                Ok(answered(cut, Vec::new()))
            },
        )
        .await;
        let lines = ndjson(&body_text(response).await);
        assert_eq!(
            joined(&lines, |l| l["message"]["thinking"].as_str()),
            "Step one, step two</thi"
        );
        assert_eq!(joined(&lines, |l| l["message"]["content"].as_str()), "");

        let response = respond(
            "m".into(),
            false,
            true,
            None,
            chat_delta,
            chat_done,
            move |_| async move { Ok(answered(cut, Vec::new())) },
        )
        .await;
        let body: serde_json::Value =
            serde_json::from_str(&body_text(response).await).expect("JSON");
        assert_eq!(
            body["message"]["thinking"],
            json!("Step one, step two</thi")
        );
        assert_eq!(body["message"]["content"], json!(""));
    }

    /// Thinking and tools together. The thinking is split out first and only
    /// the answer is looked in for calls, so a call the model drafted while
    /// thinking stays in `thinking`, verbatim, and is not one it made — the
    /// engine reads calls from the same place (`engine::lift_tool_calls`).
    #[tokio::test]
    async fn a_call_drafted_while_thinking_stays_in_the_thinking() {
        let _serial = crate::progress_serial().await;
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let block = "<think>\nMaybe <tool_call>{\"name\": \"x\"}</tool_call>?\n</think>";
        let response = respond(
            "m".into(),
            true,
            true,
            hermes,
            chat_delta,
            chat_done,
            move |sink| async move {
                let _ = sink.delta(&format!("{block}\n\n"));
                let _ = sink.delta("Checking. <tool");
                let _ = sink.delta(
                    "_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
                );
                Ok(answered(
                    &format!("{block}\n\nChecking."),
                    vec![weather_call()],
                ))
            },
        )
        .await;
        let lines = ndjson(&body_text(response).await);
        assert_eq!(
            joined(&lines, |l| l["message"]["thinking"].as_str()),
            "Maybe <tool_call>{\"name\": \"x\"}</tool_call>?\n"
        );
        assert_eq!(
            joined(&lines, |l| l["message"]["content"].as_str()),
            "Checking. "
        );
        let calls: Vec<_> = lines
            .iter()
            .filter_map(|l| l["message"].get("tool_calls"))
            .collect();
        assert_eq!(
            calls,
            [
                &json!([{"function": {"index": 0, "name": "get_weather", "arguments": {"city": "Paris"}}}])
            ],
            "{lines:?}"
        );
        assert_eq!(lines.last().expect("lines")["done"], json!(true));
    }

    /// The splitter eats what ollama's parser eats — the whitespace around
    /// the block — and nothing else; and a stream split anywhere, down to a
    /// character at a time, comes out the same as the whole.
    #[test]
    fn the_split_is_ollamas_and_does_not_depend_on_where_the_deltas_fall() {
        // (model output, thinking, content)
        let cases = [
            (THOUGHT, "The user said hi.\n", "Hello!"),
            (
                "\n<think>\n\nwhy\n</think>\n\nAnswer.\n",
                "why\n",
                "Answer.\n",
            ),
            ("<think></think>\n\nAnswer.", "", "Answer."),
            ("  no block, indented", "", "  no block, indented"),
            ("\n", "", "\n"),
            ("5 < 7 and 8 > 2", "", "5 < 7 and 8 > 2"),
            ("café <think>☕</think> ok", "☕", "café  ok"),
            // The answer began before the block, so its leading whitespace
            // is its own — however the text was cut up.
            ("  hi <think>x</think> there", "x", "  hi  there"),
        ];
        for (text, thinking, content) in cases {
            let whole = Reasoning::split(text);
            assert_eq!(
                (whole.thought.as_str(), whole.visible.as_str()),
                (thinking, content),
                "{text:?}"
            );
            let mut r = Reasoning::default();
            let mut streamed = Split::default();
            let mut take = |piece: Split| {
                streamed.visible.push_str(&piece.visible);
                streamed.thought.push_str(&piece.thought);
            };
            for (i, c) in text.char_indices() {
                take(r.push(&text[i..i + c.len_utf8()]));
            }
            take(r.finish());
            assert_eq!(streamed, whole, "{text:?}, a character at a time");
        }
    }

    /// `think: true` is refused exactly where `/api/show` does not report
    /// "thinking" — both read `engine::thinks` — and in ollama's words.
    /// Asking not to see it is never refused.
    #[test]
    fn thinking_is_advertised_exactly_where_a_request_may_ask_to_see_it() {
        let root = scratch_root("think");
        let (mut accepted, mut refused) = (0, 0);
        for spec in mummu::registry::catalog() {
            if spec.architecture == Architecture::MiniLm {
                continue; // not chat-servable: `plan` never gets this far
            }
            let name = spec.name.clone();
            let advertised = capabilities(&spec, &root).contains(&"thinking");
            let mut p = a_plan_for(spec, &root);
            assert!(allow_thinking(&p, &name).is_ok(), "{name}");
            p.think = true;
            match allow_thinking(&p, &name) {
                Ok(()) => {
                    assert!(advertised, "{name} took think /api/show does not advertise");
                    accepted += 1;
                }
                Err(e) => {
                    assert!(!advertised, "{name} refused think /api/show advertises");
                    assert_eq!(e, format!("{name:?} does not support thinking"));
                    refused += 1;
                }
            }
        }
        assert!(accepted > 0 && refused > 0, "{accepted} / {refused}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A pull ends on its own `status` line: only a chat stream is held to
    /// ending on a final line.
    #[tokio::test]
    async fn only_a_chat_stream_is_given_a_final_line() {
        // Holds an `InFlight`, which an exit test elsewhere would wait on.
        let _serial = crate::progress_serial().await;
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(json!({"status": "success"})).expect("open");
        drop(tx);
        assert_eq!(
            body_text(ndjson_response(rx, None)).await,
            "{\"status\":\"success\"}\n"
        );

        let (tx, rx) = mpsc::unbounded_channel();
        let half = Said {
            content: "half",
            thinking: "",
        };
        tx.send(wrap("m", half)).expect("open");
        drop(tx);
        let text = body_text(ndjson_response(rx, Some(InFlight::enter()))).await;
        let last: serde_json::Value =
            serde_json::from_str(text.lines().last().expect("lines")).expect("JSON");
        assert!(last["error"].is_string(), "{text}");
    }
}
