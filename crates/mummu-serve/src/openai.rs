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
//! Not implemented, and refused by name rather than ignored: `tools` /
//! function calling, `n` > 1, and image parts in a message (mummu has no
//! vision path — the text model and the vision tower are separate files and
//! only the former is loaded).

use std::ops::ControlFlow;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use crate::recovery::{self, InFlight};
use crate::shim::{OllamaOptions, RunPlan, plan, tags_body};
use crate::{ChatMessage, OutputFormat, engine, json_response, parse_json};

/// Both spellings of every route. See the module comment.
pub(crate) fn router() -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/models", get(models))
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
    /// Recognised only to refuse it — see the module comment.
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    n: Option<u32>,
}

impl ChatCompletionRequest {
    /// Translate into the shim's validated plan, reusing its sampler checks
    /// and its catalog lookup so the two surfaces cannot drift apart on what
    /// counts as a valid request.
    fn to_plan(&self) -> Result<RunPlan, Response> {
        if self.tools.is_some() {
            return Err(bad_request(
                "tools / function calling are not supported by this server",
                "unsupported_parameter",
            ));
        }
        if self.n.is_some_and(|n| n != 1) {
            return Err(bad_request(
                "n > 1 is not supported — this server returns a single choice",
                "unsupported_parameter",
            ));
        }
        let messages = self
            .messages
            .iter()
            .map(|m| {
                let (content, images) = match &m.content {
                    Some(c) => c.split()?,
                    None => (String::new(), Vec::new()),
                };
                Ok(ChatMessage {
                    role: m.role.clone(),
                    content,
                    images,
                })
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(|e| bad_request(&e, "unsupported_parameter"))?;

        let format = match self.response_format.as_ref().map(ResponseFormat::resolve) {
            Some(Ok(f)) => f,
            Some(Err(e)) => return Err(bad_request(&e, "unsupported_parameter")),
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

        // `plan` answers in the ollama error shape, which would be wrong
        // here; re-dress whatever it says as an OpenAI error.
        let mut p = plan(&self.model, &messages, &options, None).map_err(|r| reshape(*r))?;
        p.format = format;
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
            return response;
        }
    };

    if recovery::restarting() {
        return json_response(
            503,
            error_body(
                &recovery::restarting_message(),
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
    respond(model, p, stream).await
}

/// Turn one generation into an OpenAI response, streamed as SSE or buffered.
async fn respond(model: String, p: RunPlan, stream: bool) -> Response {
    let id = completion_id();
    let created = created_now();

    if !stream {
        let _inflight = InFlight::enter();
        let run = engine::run_chat(
            &p.spec,
            &p.root,
            &p.turns,
            &p.opts,
            p.max_tokens,
            p.format,
            p.images,
            |_| ControlFlow::Continue(()),
        );
        return match recovery::contain(&model, run).await {
            Ok(r) => json_response(
                200,
                json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": r.text},
                        "finish_reason": "stop",
                    }],
                    // `prompt_tokens` is not counted here; reporting 0 rather
                    // than a guess keeps a client's arithmetic honest.
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": r.tokens,
                        "total_tokens": r.tokens,
                    },
                }),
            ),
            Err(e) => {
                eprintln!("[mummu-serve] openai chat {model}: {}", e.message);
                json_response(
                    e.http_status(),
                    error_body(&e.message, "server_error", "generation_failed"),
                )
            }
        };
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

        let deltas = tx.clone();
        let (cid, cmodel) = (id.clone(), model.clone());
        let run = engine::run_chat(
            &p.spec,
            &p.root,
            &p.turns,
            &p.opts,
            p.max_tokens,
            p.format,
            p.images,
            move |delta| {
                let frame = chunk(&cid, &cmodel, created, json!({"content": delta}));
                if deltas.send(sse(&frame)).is_err() {
                    // The client hung up; stop generating for it.
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(())
            },
        );
        let tail = match recovery::contain(&model, run).await {
            Ok(_) => {
                let mut stop = chunk(&id, &model, created, json!({}));
                stop["choices"][0]["finish_reason"] = json!("stop");
                format!("{}{}", sse(&stop), "data: [DONE]\n\n")
            }
            Err(e) => {
                eprintln!("[mummu-serve] openai chat {model}: {}", e.message);
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
