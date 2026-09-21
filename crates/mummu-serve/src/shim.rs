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
//! `/api/chat` takes `tools` for the families whose calls mummu reads back,
//! and `/api/show` reports each model's capabilities from the same source.
//! Embeddings/create/copy/push answer with an explicit error rather than
//! pretending. Model names are mummu's catalog names; a trailing `:latest`
//! (which ollama CLIs append) is accepted and stripped.

use std::borrow::Cow;
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
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use crate::engine::CallSyntax;
use crate::recovery::{self, ChatError, InFlight};
use crate::think::Filter;
use crate::{
    ChatMessage, DEFAULT_MAX_TOKENS, FinalFrame, MAX_BODY_BYTES, MAX_MAX_TOKENS, OutputFormat,
    blocking, engine, json_response, models_root, parse_json, to_turns,
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
    json_response(200, json!({"version": "0.1.0"}))
}

async fn no_embeddings() -> Response {
    json_response(
        501,
        json!({"error": "embeddings are not supported by the mummu-serve shim"}),
    )
}

async fn unsupported() -> Response {
    json_response(
        501,
        json!({"error": "not supported by the mummu-serve shim"}),
    )
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
/// Howard Hinnant's algorithm). Also how `recovery` stamps the times it
/// reports, so the two surfaces spell a time the same way.
pub(crate) fn rfc3339(t: std::time::SystemTime) -> String {
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
    *TAGS_CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), body));
}

async fn tags() -> Response {
    json_response(200, tags_body().await)
}

/// The `/api/tags` body, cache and all — the catalog source `/v1/models`
/// maps into OpenAI's shape, so the two surfaces never disagree about what
/// is installed and neither one walks the disk twice.
pub(crate) async fn tags_body() -> serde_json::Value {
    let cached = TAGS_CACHE.lock().unwrap_or_else(|e| e.into_inner()).clone();
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
        json_response(200, json!({"models": models}))
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
        json!({
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
pub(crate) struct OllamaOptions {
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

/// Ollama's `format`: the string `"json"`, or a JSON Schema object.
///
/// The schema form is accepted as far as *recognising* it and then refused
/// by name — mummu constrains to "some JSON value", not to a given shape.
/// Silently downgrading a schema request to plain JSON mode would hand the
/// client a document that parses and then fails its own validation, which is
/// the harder bug to find.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum OllamaFormat {
    Named(String),
    Schema(
        #[expect(
            dead_code,
            reason = "refused today; the schema to constrain to once the ROADMAP's \
                      grammar-constrained decoding (JSON Schema via llguidance) lands"
        )]
        serde_json::Value,
    ),
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
            Self::Schema(_) => Err(
                "a JSON Schema in `format` is not supported — this server can constrain output \
                 to JSON, but not to a given schema; use \"format\": \"json\" and validate the \
                 shape client-side"
                    .into(),
            ),
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
    /// Ollama's opt-in for reasoning output. Absent means off: a client
    /// that did not ask for thinking should not have its token budget
    /// spent on it (see `crate::think`).
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
    #[serde(default)]
    think: Option<bool>,
}

/// One entry of a request's `tools` array. Ollama and OpenAI spell it the
/// same way: `{"type": "function", "function": {name, description,
/// parameters}}`.
#[derive(Deserialize)]
pub(crate) struct ToolDef {
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
pub(crate) fn tool_specs(defs: &[ToolDef]) -> Vec<ToolSpec> {
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
pub(crate) fn offer_tools(
    p: &mut RunPlan,
    model: &str,
    tools: Vec<ToolSpec>,
) -> Result<(), String> {
    if !tools.is_empty() && !engine::supports_tools(p.spec.architecture) {
        return Err(format!("{model:?} does not support tools"));
    }
    p.tools = tools;
    Ok(())
}

/// Everything a chat/generate run needs after validation.
pub(crate) struct RunPlan {
    pub(crate) spec: ModelSpec,
    pub(crate) root: std::path::PathBuf,
    pub(crate) turns: Vec<mummu::chat::Turn>,
    pub(crate) opts: mummu::decode::SamplerOptions,
    pub(crate) max_tokens: usize,
    pub(crate) format: Option<OutputFormat>,
    pub(crate) images: Vec<mummu::vision::Patches>,
    /// Pass a reasoning model's `<think>` block through to the client.
    pub(crate) think: bool,
    /// Tool definitions to advertise to the model (see [`offer_tools`]).
    pub(crate) tools: Vec<ToolSpec>,
}

/// Validate a request into a `RunPlan`, or hand back the error response.
pub(crate) fn plan(
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
        Err(e) => return Err(Box::new(json_response(400, json!({"error": e})))),
    };
    let images = match engine::prepare_images(&spec, &root, &raw) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(json_response(400, json!({"error": e})))),
    };
    let marks = match engine::placeholders(&spec, &root, &images) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(json_response(500, json!({"error": e})))),
    };
    let messages = with_placeholders(messages, &marks);
    let turns = to_turns(&messages, spec.architecture)
        .map_err(|e| json_response(400, json!({"error": e})))?;
    let opts = options
        .sampler()
        .map_err(|e| json_response(400, json!({"error": e})))?;
    let max_tokens = options.max_tokens();
    let format = match format.map(OllamaFormat::resolve).transpose() {
        Ok(f) => f.flatten(),
        Err(e) => return Err(Box::new(json_response(400, json!({"error": e})))),
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

/// An assistant message in ollama's shape, with the calls it made.
fn assistant_message(content: &str, calls: &[ToolCall]) -> serde_json::Value {
    let mut message = json!({"role": "assistant", "content": content});
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
/// or buffered, with `wrap` turning a text delta into the endpoint's frame
/// shape (`message.content` for /api/chat, `response` for /api/generate).
async fn run(
    p: RunPlan,
    stream: bool,
    wrap: fn(&str, &str) -> serde_json::Value,
    finish: fn(&str, &str, &engine::ChatResult, Instant) -> serde_json::Value,
) -> Response {
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
        stream,
        think: p.think,
        json_mode: p.format.is_some(),
        tools: p.tools.len(),
        images: p.images.len(),
        started: Instant::now(),
    };
    // The process is exiting to restart the GPU backend (see `recovery`).
    if recovery::restarting() {
        return json_response(503, json!({"error": recovery::restarting_message()}));
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
            json!({"error": format!(
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
    respond(model, stream, calls, wrap, finish, move |sink| async move {
        let r = engine::run_chat(
            &p.spec,
            &p.root,
            &p.turns,
            &p.opts,
            p.max_tokens,
            p.format,
            p.think,
            p.images,
            p.tools,
            |delta| sink.delta(delta),
        )
        .await;
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
    })
    .await
}

/// Where a shim generation's text goes: one NDJSON line per piece when
/// streaming, nowhere when the answer is buffered.
struct ShimSink {
    tx: Option<mpsc::UnboundedSender<serde_json::Value>>,
    model: String,
    wrap: fn(&str, &str) -> serde_json::Value,
    /// For a request that offered tools: holds the family's call markup
    /// back from the stream. The calls go out structured, and whole, once
    /// the answer is — see [`tool_tail`].
    held: Option<Arc<Mutex<Filter>>>,
}

impl ShimSink {
    fn delta(&self, text: &str) -> ControlFlow<()> {
        let Some(tx) = &self.tx else {
            return ControlFlow::Continue(());
        };
        let visible = match &self.held {
            Some(f) => Cow::Owned(f.lock().unwrap_or_else(|e| e.into_inner()).push(text)),
            None => Cow::Borrowed(text),
        };
        if visible.is_empty() {
            return ControlFlow::Continue(());
        }
        match tx.send((self.wrap)(&self.model, &visible)) {
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
    wrap: fn(&str, &str) -> serde_json::Value,
    held: &mut Filter,
    r: &mut engine::ChatResult,
) -> Vec<serde_json::Value> {
    let text = held.settle(!r.tool_calls.is_empty());
    let mut frames = Vec::new();
    if !text.is_empty() {
        frames.push(wrap(model, &text));
    }
    if !r.tool_calls.is_empty() {
        frames.push(json!({
            "model": model,
            "created_at": now_rfc3339(),
            "message": assistant_message("", &std::mem::take(&mut r.tool_calls)),
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
/// that fails the way production did. `calls` is the family's tool-call
/// convention when the request offered tools: a streamed answer holds that
/// markup back and sends the calls structured instead (see [`tool_tail`]).
async fn respond<F, Fut>(
    model: String,
    stream: bool,
    calls: Option<CallSyntax>,
    wrap: fn(&str, &str) -> serde_json::Value,
    finish: fn(&str, &str, &engine::ChatResult, Instant) -> serde_json::Value,
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
            let held = calls.map(|c| Arc::new(Mutex::new(Filter::spans(c.open, c.close))));
            let sink = ShimSink {
                tx: Some(tx.clone()),
                model: model.clone(),
                wrap,
                held: held.clone(),
            };
            let line = match recovery::contain(&model, run(sink)).await {
                Ok(mut r) => {
                    if let Some(held) = held {
                        let mut held = held.lock().unwrap_or_else(|e| e.into_inner());
                        for frame in tool_tail(&model, wrap, &mut held, &mut r) {
                            let _ = tx.send(frame);
                        }
                    }
                    finish(&model, "", &r, started)
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
                let text = r.text.clone();
                (200, finish(&model, &text, &r, started))
            }
            Err(e) => {
                eprintln!("[mummu-serve] shim chat {model}: {e}");
                (e.http_status(), error_line(&e))
            }
        }
    })
    .await
}

pub(crate) async fn chat(body: Bytes) -> Response {
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
    let tools = tool_specs(parsed.tools.as_deref().unwrap_or_default());
    if let Err(e) = offer_tools(&mut p, &parsed.model, tools) {
        return json_response(400, json!({"error": e}));
    }
    run(p, parsed.stream.unwrap_or(true), chat_delta, chat_done).await
}

/// One streamed piece of an `/api/chat` answer.
fn chat_delta(model: &str, delta: &str) -> serde_json::Value {
    json!({
        "model": model,
        "created_at": now_rfc3339(),
        "message": {"role": "assistant", "content": delta},
        "done": false,
    })
}

/// The last line of an `/api/chat` answer — the whole answer, when buffered.
fn chat_done(
    model: &str,
    text: &str,
    r: &engine::ChatResult,
    started: Instant,
) -> serde_json::Value {
    let mut v = done_value(model, r, started);
    v["message"] = assistant_message(text, &r.tool_calls);
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

    async fn fails_like_production() -> Result<engine::ChatResult, ChatError> {
        panic!("{INVALID_READ}")
    }

    async fn fails_with_an_ordinary_bug() -> Result<engine::ChatResult, ChatError> {
        panic!("called `Option::unwrap()` on a `None` value")
    }

    fn wrap(model: &str, delta: &str) -> serde_json::Value {
        json!({"model": model, "response": delta, "done": false})
    }

    fn finish(
        model: &str,
        text: &str,
        r: &engine::ChatResult,
        started: Instant,
    ) -> serde_json::Value {
        let mut v = done_value(model, r, started);
        v["response"] = json!(text);
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
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_streamed_shim_chat_that_hits_the_gpu_failure_ends_on_an_error_line() {
        let _serial = crate::progress_serial();
        recovery::reset_for_tests();
        recovery::install_panic_hook();

        let response = respond("m".into(), true, None, wrap, finish, |_| {
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
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_buffered_shim_chat_answers_non_2xx_with_the_error() {
        let _serial = crate::progress_serial();
        recovery::reset_for_tests();
        recovery::install_panic_hook();

        let response = respond("m".into(), false, None, wrap, finish, |_| {
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
        let response = respond("m".into(), false, None, wrap, finish, |_| {
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
    /// every family, and a JSON-encoded object (OpenAI's spelling) is read.
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
        let deep = json!([at_bound.clone()]);
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
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_streamed_tool_call_arrives_structured_and_its_markup_never_does() {
        let _serial = crate::progress_serial();
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let response = respond(
            "m".into(),
            true,
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
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn held_back_markup_that_was_not_a_call_is_released_as_text() {
        let _serial = crate::progress_serial();
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let raw = "Hmm <tool_call>{not json</tool_call> then <tool_call>{\"name\": \"cut";
        let response = respond(
            "m".into(),
            true,
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
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_buffered_tool_call_comes_back_in_the_message() {
        let _serial = crate::progress_serial();
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let response = respond(
            "m".into(),
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

    /// A pull ends on its own `status` line: only a chat stream is held to
    /// ending on a final line.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn only_a_chat_stream_is_given_a_final_line() {
        // Holds an `InFlight`, which an exit test elsewhere would wait on.
        let _serial = crate::progress_serial();
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(json!({"status": "success"})).expect("open");
        drop(tx);
        assert_eq!(
            body_text(ndjson_response(rx, None)).await,
            "{\"status\":\"success\"}\n"
        );

        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(wrap("m", "half")).expect("open");
        drop(tx);
        let text = body_text(ndjson_response(rx, Some(InFlight::enter()))).await;
        let last: serde_json::Value =
            serde_json::from_str(text.lines().last().expect("lines")).expect("JSON");
        assert!(last["error"].is_string(), "{text}");
    }
}
