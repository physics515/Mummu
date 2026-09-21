//! OpenAI-compatible surface: `/v1/chat/completions` and `/v1/models`,
//! served on the same listener as the Ollama shim and driving the same
//! engine.
//!
//! **Why this exists.** A client that calls itself an "Ollama" integration
//! is not necessarily speaking the Ollama protocol. Traced live on
//! 2026-09-20: an Android app configured for Ollama posted
//! `/chat/completions`, mummu's router had nothing there, and the 404 came
//! back as "your provider couldn't find this model" — a message about the
//! *model* for a failure about the *path*. Nothing in mummu's log named the
//! cause, because the router rejects an unknown path before any handler
//! runs. The protocol is small and the engine underneath is the same one,
//! so serving it is cheaper than every such client being unable to say what
//! went wrong.
//!
//! **Path aliases.** Clients append `chat/completions` to a configured base
//! URL, and whether that base ends in `/v1` is the user's guess. Both
//! spellings are served rather than making a person get it right by trial
//! and error — the same reason a trailing `:latest` is stripped from model
//! names.
//!
//! **Tool calls.** The model answers a request that offered `tools` in its
//! family's call markup (`engine::tool_calls`), and the engine lifts the
//! calls out of the finished text. A buffered answer puts them in
//! `message.tool_calls`. A streamed one holds the markup back from
//! `delta.content` as it arrives, even when a tag is split across deltas, and
//! sends each call in a `delta.tool_calls` chunk of its own once the answer
//! is whole. Either way `finish_reason` is `"tool_calls"`. Markup that does
//! not parse, such as a call the token cap cut off, is released as ordinary
//! text instead, so nothing the model wrote goes missing.
//!
//! Not implemented, and refused by name rather than ignored: `n` > 1, an
//! image by remote URL, and `tools` for a model whose calls nothing reads
//! back (see `engine::tool_calls`).

use std::borrow::Cow;
use std::future::Future;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use mummu::chat::ToolCall;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use crate::engine::{CallSyntax, ChatResult};
use crate::recovery::{self, ChatError, InFlight};
use crate::shim::{OllamaOptions, RunPlan, ToolDef, offer_tools, plan, tags_body, tool_specs};
use crate::think::Filter;
use crate::{ChatMessage, OutputFormat, engine, json_response, parse_json};

/// Both spellings of every route. See the module comment.
pub(crate) fn router() -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        // Per-request visibility (see `crate::trace`). Served here as well as
        // on the native API because this is the listener the public
        // hostname reaches.
        .route("/api/requests", get(requests))
        .route("/api/stats", get(stats))
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// A message's `content`: a bare string, or the "parts" array multimodal
/// clients send. `null` is legal on an assistant turn that carried only tool
/// calls.
#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<Part>),
    Null,
}

#[derive(Deserialize)]
struct Part {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<ImageUrl>,
}

#[derive(Deserialize)]
struct ImageUrl {
    url: String,
}

impl Content {
    /// Split into the text mummu renders and the base64 image payloads.
    ///
    /// OpenAI carries images inline in `content` parts; ollama puts them in
    /// a sibling `images` field. Both are normalised to the same pair here,
    /// so only one path downstream ever has to know about pictures.
    ///
    /// A remote `http(s)` URL is refused rather than fetched: a server that
    /// dereferences a URL out of a request body will happily be pointed at
    /// its own metadata endpoint or anything else inside the network it sits
    /// in. Clients that want an image sent must send the image.
    fn split(&self) -> Result<(String, Vec<String>), String> {
        match self {
            Self::Text(s) => Ok((s.clone(), Vec::new())),
            Self::Null => Ok((String::new(), Vec::new())),
            Self::Parts(parts) => {
                let (mut text, mut images) = (String::new(), Vec::new());
                for p in parts {
                    match p.kind.as_str() {
                        "text" => text.push_str(p.text.as_deref().unwrap_or_default()),
                        "image_url" => {
                            let url = p
                                .image_url
                                .as_ref()
                                .map(|u| u.url.as_str())
                                .unwrap_or_default();
                            if !url.starts_with("data:") {
                                return Err(
                                    "image_url must be a data: URI with the image inline — this \
                                     server does not fetch remote URLs"
                                        .into(),
                                );
                            }
                            images.push(url.to_string());
                        }
                        other => {
                            return Err(format!(
                                "message content part of type {other:?} is not supported"
                            ));
                        }
                    }
                }
                Ok((text, images))
            }
        }
    }
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Option<Content>,
    /// Present when the client replays an assistant turn that made calls.
    #[serde(default)]
    tool_calls: Vec<ToolCallIn>,
}

/// OpenAI's wire shape for a call the assistant already made.
#[derive(Deserialize)]
struct ToolCallIn {
    #[serde(default)]
    function: Option<FunctionCallIn>,
}

#[derive(Deserialize)]
struct FunctionCallIn {
    name: String,
    /// A JSON *string*, per the protocol — not an object.
    #[serde(default)]
    arguments: String,
}

/// OpenAI's `response_format`. `json_schema` is recognised and refused by
/// name for the same reason ollama's schema form is: mummu constrains to
/// "some JSON value", and quietly downgrading a schema request would return
/// a document that parses and then fails the client's own validation.
#[derive(Deserialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

impl ResponseFormat {
    fn resolve(&self) -> Result<Option<OutputFormat>, String> {
        match self.kind.as_str() {
            "json_object" => Ok(Some(OutputFormat::Json)),
            "text" => Ok(None),
            "json_schema" => Err(
                "response_format \"json_schema\" is not supported — this server can constrain \
                 output to JSON, but not to a given schema; use {\"type\": \"json_object\"} and \
                 validate the shape client-side"
                    .into(),
            ),
            other => Err(format!("unsupported response_format {other:?}")),
        }
    }
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    /// Deprecated by OpenAI in favour of `max_completion_tokens`; both are
    /// still sent in the wild, and the newer one wins when both appear.
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    /// OpenAI function definitions. Rendered into the prompt through the
    /// family's own tool convention (Hermes `<tools>` for Qwen), and the
    /// model's `<tool_call>` blocks are parsed back out of the answer.
    #[serde(default)]
    tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    n: Option<u32>,
    /// OpenAI's reasoning control. Anything but `"none"` opts in; absent
    /// means off, so a client that never asked for thinking does not get
    /// its token budget spent on it (see `crate::think`).
    #[serde(default)]
    reasoning_effort: Option<String>,
}

impl ChatCompletionRequest {
    /// Translate into the shim's validated plan, reusing its sampler checks
    /// and its catalog lookup so the two surfaces cannot drift apart on what
    /// counts as a valid request.
    fn to_plan(&self) -> Result<RunPlan, Box<Response>> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(Box::new(bad_request(
                "n > 1 is not supported — this server returns a single choice",
                "unsupported_parameter",
            )));
        }
        let messages = self
            .messages
            .iter()
            .map(|m| {
                let (content, images) = match &m.content {
                    Some(c) => c.split()?,
                    None => (String::new(), Vec::new()),
                };
                let tool_calls = m
                    .tool_calls
                    .iter()
                    .filter_map(|c| c.function.as_ref())
                    .map(|f| mummu::chat::ToolCall {
                        name: f.name.clone(),
                        // Arguments arrive JSON-encoded in a string; a call
                        // whose arguments do not parse is kept with an empty
                        // object rather than dropped, so the turn still shows
                        // the model what it asked for.
                        arguments: serde_json::from_str(&f.arguments)
                            .unwrap_or_else(|_| serde_json::json!({})),
                    })
                    .collect();
                Ok(ChatMessage {
                    role: m.role.clone(),
                    content,
                    images,
                    tool_calls,
                })
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(|e| Box::new(bad_request(&e, "unsupported_parameter")))?;

        let format = match self.response_format.as_ref().map(ResponseFormat::resolve) {
            Some(Ok(f)) => f,
            Some(Err(e)) => return Err(Box::new(bad_request(&e, "unsupported_parameter"))),
            None => None,
        };

        // OpenAI's documented defaults, not ollama's: a client that sends
        // neither knob should get what the protocol it is speaking promises.
        let options = OllamaOptions {
            temperature: Some(self.temperature.unwrap_or(1.0)),
            top_p: Some(self.top_p.unwrap_or(1.0)),
            top_k: None,
            seed: self.seed,
            num_predict: self
                .max_completion_tokens
                .or(self.max_tokens)
                .and_then(|n| i64::try_from(n).ok()),
        };

        let think = self
            .reasoning_effort
            .as_deref()
            .is_some_and(|r| !r.eq_ignore_ascii_case("none"));
        // `plan` answers in the ollama error shape, which would be wrong
        // here; re-dress whatever it says as an OpenAI error.
        let mut p = plan(&self.model, &messages, &options, None, think)
            .map_err(|r| Box::new(reshape(*r)))?;
        p.format = format;
        offer_tools(
            &mut p,
            &self.model,
            tool_specs(self.tools.as_deref().unwrap_or_default()),
        )
        .map_err(|e| Box::new(bad_request(&e, "unsupported_parameter")))?;
        Ok(p)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// OpenAI's error envelope. Clients key their messages off `code`, so the
/// codes here are the real ones: a wrong model name must read as
/// `model_not_found` and nothing else.
fn error_body(message: &str, kind: &str, code: &str) -> serde_json::Value {
    json!({"error": {"message": message, "type": kind, "code": code}})
}

fn bad_request(message: &str, code: &str) -> Response {
    json_response(400, error_body(message, "invalid_request_error", code))
}

/// Re-dress one of `plan`'s ollama-shaped refusals as an OpenAI error,
/// keeping its status and its text.
fn reshape(r: Response) -> Response {
    let status = r.status();
    let code = if status == axum::http::StatusCode::NOT_FOUND {
        "model_not_found"
    } else {
        "invalid_request_error"
    };
    // The body is already consumed into a Response; rebuild from the status
    // alone rather than buffering it back, and say the one thing a client
    // can act on.
    let message = if status == axum::http::StatusCode::NOT_FOUND {
        "the requested model is not installed on this server — GET /v1/models lists what is"
    } else {
        "the request was rejected by this server"
    };
    json_response(
        status.as_u16(),
        error_body(message, "invalid_request_error", code),
    )
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

async fn models() -> Response {
    let tags = tags_body().await;
    let data: Vec<serde_json::Value> = tags
        .get("models")
        .and_then(serde_json::Value::as_array)
        .map(|ms| {
            ms.iter()
                .filter_map(|m| m.get("name").and_then(serde_json::Value::as_str))
                .map(|name| {
                    json!({
                        "id": name,
                        "object": "model",
                        // No real creation date is known; clients only ever
                        // sort or display this.
                        "created": 0,
                        "owned_by": "mummu",
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json_response(200, json!({"object": "list", "data": data}))
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

fn completion_id() -> String {
    // Unique per response, which is all a client needs it for.
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("chatcmpl-{n:032x}")
}

fn created_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// One streamed chunk: `choices[0].delta`.
fn chunk(id: &str, model: &str, created: u64, delta: serde_json::Value) -> serde_json::Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
    })
}

async fn chat_completions(body: Bytes) -> Response {
    let started = std::time::Instant::now();
    let parsed: ChatCompletionRequest = match parse_json(&body) {
        Ok(p) => p,
        Err(response) => return reshape(*response),
    };
    let stream = parsed.stream.unwrap_or(false);
    let model = parsed.model.clone();
    // One line per accepted request, before any engine work — the same
    // contract the ollama shim keeps, and the thing whose absence made the
    // original 404 invisible.
    eprintln!("[mummu-serve] openai chat {model}: request accepted (stream = {stream})");
    let p = match parsed.to_plan() {
        Ok(p) => p,
        Err(response) => {
            eprintln!("[mummu-serve] openai chat {model}: rejected before the engine");
            let status = response.status();
            crate::trace::Begin {
                surface: "openai",
                model: model.clone(),
                stream,
                think: false,
                json_mode: false,
                tools: parsed.tools.as_ref().map_or(0, Vec::len),
                images: 0,
                started,
            }
            .finish(
                "",
                crate::trace::Timings::default(),
                false,
                Err(&format!(
                    "rejected before the engine: HTTP {}",
                    status.as_u16()
                )),
            );
            return *response;
        }
    };

    if recovery::restarting() {
        return json_response(
            503,
            error_body(
                recovery::restarting_message(),
                "server_error",
                "server_restarting",
            ),
        );
    }
    // Same honesty as the ollama shim: a buffered response parked behind a
    // cold load sends nothing for the whole wait, which is indistinguishable
    // from a hung server.
    if !stream && let Some(dir) = engine::load_in_flight() {
        let loading = dir.file_name().map_or_else(
            || dir.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        eprintln!("[mummu-serve] openai chat {model}: 503 — the {loading} load is in flight");
        return json_response(
            503,
            error_body(
                &format!(
                    "a model is loading ({loading}); a non-streaming request would wait silently \
                     for the whole load — retry once it completes, or set \"stream\": true"
                ),
                "server_error",
                "model_loading",
            ),
        );
    }
    let begin = crate::trace::Begin {
        surface: "openai",
        model: model.clone(),
        stream,
        think: p.think,
        json_mode: p.format.is_some(),
        tools: p.tools.len(),
        images: p.images.len(),
        started,
    };
    // `offer_tools` let tools through only to a family with a convention.
    let calls = if p.tools.is_empty() {
        None
    } else {
        engine::tool_calls(p.spec.architecture)
    };
    // The plan moves INTO the generation: a streamed answer runs it on a
    // task of its own, and a padded buffered one hands it to a response
    // body; both outlive this call, so it cannot borrow anything from here.
    respond(model, stream, calls, begin, move |sink| async move {
        engine::run_chat(
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
        .await
    })
    .await
}

/// Turn one generation into an OpenAI response, streamed as SSE or buffered.
///
/// `run` is the generation, fed the [`Deltas`] its text goes to: production
/// passes `engine::run_chat`, a test a scripted answer. `calls` is the
/// family's tool-call convention when the request offered tools: a streamed
/// answer holds that markup back and sends the calls structured at the end
/// instead (see [`stream_tail`]).
async fn respond<F, Fut>(
    model: String,
    stream: bool,
    calls: Option<CallSyntax>,
    begin: crate::trace::Begin,
    run: F,
) -> Response
where
    F: FnOnce(Deltas) -> Fut + Send + 'static,
    Fut: Future<Output = Result<ChatResult, ChatError>> + Send,
{
    let id = completion_id();
    let created = created_now();

    if !stream {
        return crate::keepalive_json(async move {
            let _inflight = InFlight::enter();
            let outcome = recovery::contain(&model, run(Deltas::nowhere())).await;
            // Padded exactly when the answer outlived the keep-alive grace:
            // that is the condition `keepalive_json` pads on.
            let padded = begin.started.elapsed() >= crate::KEEPALIVE_GRACE;
            match outcome {
                Ok(r) => {
                    begin.finish(r.device, r.timings.clone(), padded, Ok(()));
                    // The model answers a tool request with its family's call
                    // markup, which the engine lifts out of the text (see
                    // `engine::lift_tool_calls`); OpenAI clients expect the calls
                    // in a structured field, with `finish_reason` saying so — a
                    // client that gets the raw markers in `content` cannot act.
                    let calls = &r.tool_calls;
                    let message = if calls.is_empty() {
                        json!({"role": "assistant", "content": r.text})
                    } else {
                        json!({
                            "role": "assistant",
                            "content": (!r.text.trim().is_empty()).then_some(&r.text),
                            "tool_calls": calls.iter().enumerate().map(|(i, c)| call_json(i, c))
                                .collect::<Vec<_>>(),
                        })
                    };
                    (
                        200,
                        json!({
                            "id": id,
                            "object": "chat.completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": message,
                                "finish_reason": finish_reason(calls),
                            }],
                            "usage": {
                                "prompt_tokens": r.timings.prompt_tokens,
                                "completion_tokens": r.tokens,
                                "total_tokens": r.timings.prompt_tokens + r.tokens,
                            },
                        }),
                    )
                }
                Err(e) => {
                    eprintln!("[mummu-serve] openai chat {model}: {}", e.message);
                    begin.finish(
                        "",
                        crate::trace::Timings::default(),
                        padded,
                        Err(&e.message),
                    );
                    (
                        e.http_status(),
                        error_body(&e.message, "server_error", "generation_failed"),
                    )
                }
            }
        })
        .await;
    }

    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let inflight = InFlight::enter();
    tokio::spawn(async move {
        let last = FinalText::new(
            tx.clone(),
            format!(
                "{}{}",
                sse(&error_body(
                    "the server's worker for this request ended without a result — this is a \
                 mummu-serve bug, not your request; try again",
                    "server_error",
                    "worker_vanished",
                )),
                "data: [DONE]\n\n"
            ),
        );
        // The role arrives in its own first chunk, as OpenAI does it.
        let _ = tx.send(sse(&chunk(
            &id,
            &model,
            created,
            json!({"role": "assistant"}),
        )));

        let held = calls.map(|c| Arc::new(Mutex::new(Filter::spans(c.open, c.close))));
        let sink = Deltas {
            tx: Some(tx.clone()),
            id: id.clone(),
            model: model.clone(),
            created,
            held: held.clone(),
        };
        let tail = match recovery::contain(&model, run(sink)).await {
            Ok(r) => {
                // Streaming is never padded: its first chunk is the keep-alive.
                begin.finish(r.device, r.timings.clone(), false, Ok(()));
                let mut held = held
                    .as_ref()
                    .map(|h| h.lock().unwrap_or_else(|e| e.into_inner()));
                stream_tail(&id, &model, created, held.as_deref_mut(), &r)
            }
            Err(e) => {
                eprintln!("[mummu-serve] openai chat {model}: {}", e.message);
                begin.finish("", crate::trace::Timings::default(), false, Err(&e.message));
                format!(
                    "{}{}",
                    sse(&error_body(&e.message, "server_error", "generation_failed")),
                    "data: [DONE]\n\n"
                )
            }
        };
        last.send(tail);
    });
    sse_response(rx, inflight)
}

/// Where a generation's text goes: one SSE chunk per piece of it the client
/// may see now. A buffered answer's goes nowhere; it is read off the result.
struct Deltas {
    tx: Option<mpsc::UnboundedSender<String>>,
    id: String,
    model: String,
    created: u64,
    /// For a request that offered tools: holds the family's call markup
    /// back from `delta.content`, as the ollama shim does. The calls go out
    /// structured, and whole, once the answer is — see [`stream_tail`].
    held: Option<Arc<Mutex<Filter>>>,
}

impl Deltas {
    fn nowhere() -> Self {
        Self {
            tx: None,
            id: String::new(),
            model: String::new(),
            created: 0,
            held: None,
        }
    }

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
        let frame = chunk(
            &self.id,
            &self.model,
            self.created,
            json!({"content": visible}),
        );
        if tx.send(sse(&frame)).is_err() {
            // The client hung up; stop generating for it.
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// One lifted call in OpenAI's shape, with its arguments JSON-encoded in a
/// string, as the protocol has them.
fn call_json(i: usize, c: &ToolCall) -> serde_json::Value {
    json!({
        "id": format!("call_{i}_{}", c.name),
        "type": "function",
        "function": {
            "name": c.name,
            "arguments": c.arguments.to_string(),
        },
    })
}

/// `"tool_calls"` when the answer made calls, which is what tells a client to
/// run them and reply, rather than show the answer and stop.
fn finish_reason(calls: &[ToolCall]) -> &'static str {
    if calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    }
}

/// Everything a finished stream still owes its client, rendered: what
/// `held` kept back that turned out not to be a call (see
/// [`Filter::settle`]), each lifted call in a chunk of its own, the chunk
/// carrying `finish_reason`, and the terminator.
///
/// Each call goes out whole, with its `index` and its arguments complete in
/// one string. OpenAI itself splits the arguments across chunks, but a
/// client concatenates them per `index` either way, and mummu only knows a
/// call once it has parsed all of it.
fn stream_tail(
    id: &str,
    model: &str,
    created: u64,
    held: Option<&mut Filter>,
    r: &ChatResult,
) -> String {
    let mut frames = Vec::new();
    let owed = held.map(|h| h.settle(!r.tool_calls.is_empty()));
    if let Some(text) = owed.filter(|t| !t.is_empty()) {
        frames.push(chunk(id, model, created, json!({"content": text})));
    }
    for (i, c) in r.tool_calls.iter().enumerate() {
        let mut call = call_json(i, c);
        call["index"] = json!(i);
        frames.push(chunk(id, model, created, json!({"tool_calls": [call]})));
    }
    let mut stop = chunk(id, model, created, json!({}));
    stop["choices"][0]["finish_reason"] = json!(finish_reason(&r.tool_calls));
    frames.push(stop);
    let mut out: String = frames.iter().map(sse).collect();
    out.push_str("data: [DONE]\n\n");
    out
}

/// Guarantees an SSE stream ends, exactly once: with the terminator the
/// worker chose, or — if the worker panicked outside what
/// [`recovery::contain`] guards, or the runtime dropped it — with
/// `fallback`. A client that never sees `data: [DONE]` waits for its whole
/// read timeout, which is 600 s on the app this was written for.
///
/// The crate's `FinalFrame` does the same job for the ollama shim, but
/// carries a `serde_json::Value`; SSE frames are pre-rendered strings.
struct FinalText {
    tx: Option<mpsc::UnboundedSender<String>>,
    fallback: String,
}

impl FinalText {
    fn new(tx: mpsc::UnboundedSender<String>, fallback: String) -> Self {
        Self {
            tx: Some(tx),
            fallback,
        }
    }

    fn send(mut self, frame: String) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(frame);
        }
    }
}

impl Drop for FinalText {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(std::mem::take(&mut self.fallback));
        }
    }
}

/// One server-sent event carrying `value`.
fn sse(value: &serde_json::Value) -> String {
    format!("data: {value}\n\n")
}

/// The SSE body: frames as the worker produces them, and nothing else. The
/// worker always writes a terminator (`data: [DONE]`), including on the
/// paths where it dies — see [`FinalFrame`].
fn sse_response(mut rx: mpsc::UnboundedReceiver<String>, inflight: InFlight) -> Response {
    let stream = async_stream::stream! {
        let _inflight = inflight;
        while let Some(frame) = rx.recv().await {
            yield Ok::<String, std::convert::Infallible>(frame);
        }
    };
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
            (header::CONNECTION, "keep-alive"),
        ],
        Body::from_stream(stream),
    )
        .into_response()
}

/// `GET /api/requests?limit=N` — the most recent traces, newest first.
async fn requests(
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let limit = q
        .get("limit")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50)
        .clamp(1, 256);
    json_response(200, crate::trace::recent_json(limit))
}

/// `GET /api/stats` — latency and throughput percentiles, warm vs cold.
async fn stats() -> Response {
    json_response(200, crate::trace::stats_json())
}

#[cfg(test)]
mod tests {
    use mummu::registry::Architecture;

    use super::*;

    fn begin(stream: bool) -> crate::trace::Begin {
        crate::trace::Begin {
            surface: "openai",
            model: "m".into(),
            stream,
            think: false,
            json_mode: false,
            tools: 1,
            images: 0,
            started: std::time::Instant::now(),
        }
    }

    fn weather_call(city: &str) -> ToolCall {
        ToolCall {
            name: "get_weather".into(),
            arguments: json!({"city": city}),
        }
    }

    /// What the engine hands back once it has lifted the calls.
    fn answered(text: &str, tool_calls: Vec<ToolCall>) -> ChatResult {
        ChatResult {
            text: text.into(),
            tokens: 1,
            device: "test",
            elapsed_ms: 0,
            timings: crate::trace::Timings::default(),
            tool_calls,
        }
    }

    async fn body_text(response: Response) -> String {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        String::from_utf8_lossy(&body).into_owned()
    }

    /// A stream's JSON events, once it is known to end on the terminator,
    /// and only once.
    fn sse_events(body: &str) -> Vec<serde_json::Value> {
        assert_eq!(body.matches("data: [DONE]").count(), 1, "{body}");
        body.strip_suffix("data: [DONE]\n\n")
            .expect("the stream ends on [DONE]")
            .split_terminator("\n\n")
            .map(|e| {
                let data = e.strip_prefix("data: ").expect("an SSE data line");
                serde_json::from_str(data).expect("each event is JSON")
            })
            .collect()
    }

    fn content(events: &[serde_json::Value]) -> String {
        events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect()
    }

    /// The `finish_reason` the stream ends with — after checking that only
    /// its last chunk carries one.
    fn finish_reason_of(events: &[serde_json::Value]) -> &str {
        let (last, rest) = events.split_last().expect("events");
        assert!(
            rest.iter()
                .all(|e| e["choices"][0]["finish_reason"].is_null()),
            "{events:?}"
        );
        last["choices"][0]["finish_reason"]
            .as_str()
            .expect("the last chunk says why the answer ended")
    }

    /// Streamed, a call reaches the client the way OpenAI sends one: its
    /// markup never reaches `delta.content`, even split across deltas; the
    /// call arrives in a `delta.tool_calls` chunk of its own, arguments as a
    /// JSON string; and the stream ends on `finish_reason: "tool_calls"`.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_streamed_tool_call_arrives_as_a_tool_call_delta_and_its_markup_never_does() {
        let _serial = crate::progress_serial();
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let response = respond("m".into(), true, hermes, begin(true), |sink| async move {
            let _ = sink.delta("Checking. <tool");
            let _ = sink.delta("_call>{\"name\": \"get_weather\", ");
            let _ = sink.delta("\"arguments\": {\"city\": \"Paris\"}}</tool_call>");
            Ok(answered("Checking.", vec![weather_call("Paris")]))
        })
        .await;
        let body = body_text(response).await;
        assert!(!body.contains("tool_call>"), "no markup anywhere: {body}");
        let events = sse_events(&body);
        assert_eq!(
            events[0]["choices"][0]["delta"],
            json!({"role": "assistant"})
        );
        assert_eq!(content(&events), "Checking. ");
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"].get("tool_calls"))
            .collect();
        assert_eq!(calls.len(), 1, "one chunk carries the call: {events:?}");
        assert_eq!(
            *calls[0],
            json!([{
                "index": 0,
                "id": "call_0_get_weather",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            }])
        );
        // After every piece of content, and right before the finish.
        let at = events
            .iter()
            .position(|e| e["choices"][0]["delta"].get("tool_calls").is_some())
            .expect("the call's chunk");
        assert_eq!(at, events.len() - 2, "{events:?}");
        assert_eq!(finish_reason_of(&events), "tool_calls");
    }

    /// Parallel calls go out one chunk each, told apart by `index`, which is
    /// what a client accumulates on.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn parallel_calls_each_get_a_chunk_and_an_index() {
        let _serial = crate::progress_serial();
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let response = respond("m".into(), true, hermes, begin(true), |sink| async move {
            let _ = sink.delta(
                "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\
                 </tool_call>\n<tool_call>{\"name\": \"get_weather\", \"arguments\": \
                 {\"city\": \"Oslo\"}}</tool_call>",
            );
            Ok(answered(
                "",
                vec![weather_call("Paris"), weather_call("Oslo")],
            ))
        })
        .await;
        let events = sse_events(&body_text(response).await);
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
            .map(|calls| {
                assert_eq!(calls.len(), 1, "one call per chunk: {calls:?}");
                &calls[0]
            })
            .collect();
        assert_eq!(calls.len(), 2, "{events:?}");
        for (i, (call, city)) in calls.iter().zip(["Paris", "Oslo"]).enumerate() {
            assert_eq!(call["index"], json!(i));
            assert_eq!(call["id"], json!(format!("call_{i}_get_weather")));
            let arguments: serde_json::Value =
                serde_json::from_str(call["function"]["arguments"].as_str().expect("a string"))
                    .expect("the arguments string is JSON");
            assert_eq!(arguments, json!({"city": city}));
        }
        assert_eq!(content(&events), "\n", "only the text between the calls");
        assert_eq!(finish_reason_of(&events), "tool_calls");
    }

    /// Markup that did not parse is not swallowed: what was held back goes
    /// out as content at the end, and the answer stops like any other.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn held_back_markup_that_was_not_a_call_is_released_as_content() {
        let _serial = crate::progress_serial();
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let raw = "Hmm <tool_call>{not json</tool_call> then <tool_call>{\"name\": \"cut";
        let response = respond(
            "m".into(),
            true,
            hermes,
            begin(true),
            move |sink| async move {
                let _ = sink.delta(raw);
                // The engine could not lift anything, so the text is verbatim.
                Ok(answered(raw, Vec::new()))
            },
        )
        .await;
        let events = sse_events(&body_text(response).await);
        assert_eq!(
            content(&events),
            "Hmm  then <tool_call>{not json</tool_call><tool_call>{\"name\": \"cut"
        );
        assert!(
            events
                .iter()
                .all(|e| e["choices"][0]["delta"].get("tool_calls").is_none())
        );
        assert_eq!(finish_reason_of(&events), "stop");
    }

    /// A request that offered no tools holds nothing back: its text streams
    /// exactly as the model wrote it, tags and all.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn without_tools_the_stream_is_the_models_text_verbatim() {
        let _serial = crate::progress_serial();
        let text = "Use <tool_call> tags, like <tool_call>{}</tool_call>.";
        let response = respond(
            "m".into(),
            true,
            None,
            begin(true),
            move |sink| async move {
                for piece in text.split_inclusive(' ') {
                    let _ = sink.delta(piece);
                }
                Ok(answered(text, Vec::new()))
            },
        )
        .await;
        let events = sse_events(&body_text(response).await);
        assert_eq!(content(&events), text);
        assert_eq!(finish_reason_of(&events), "stop");
    }

    /// Buffered, the one object carries the calls, with `content` null when
    /// there was no prose around them.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // serializes tests; nothing else waits on it
    async fn a_buffered_tool_call_comes_back_in_the_message() {
        let _serial = crate::progress_serial();
        let hermes = engine::tool_calls(Architecture::Qwen3);
        let response = respond("m".into(), false, hermes, begin(false), |_| async {
            Ok(answered("", vec![weather_call("Paris")]))
        })
        .await;
        assert_eq!(response.status(), 200);
        let body: serde_json::Value =
            serde_json::from_str(&body_text(response).await).expect("JSON");
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], json!("tool_calls"));
        assert_eq!(
            choice["message"],
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0_get_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                }],
            })
        );
    }
}
