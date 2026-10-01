//! Retrieval: embeddings, reranking and RAG, served beside the chat model.
//!
//! # Endpoints
//!
//! | surface | routes | shape |
//! |---|---|---|
//! | ollama (shim) | `POST /api/embed`, `POST /api/embeddings` | ollama's |
//! | `OpenAI` (shim) | `POST /v1/embeddings` | `OpenAI`'s |
//! | rerank (shim + native) | `POST /v1/rerank`, `POST /rerank`, `POST /api/rerank` | Jina / Cohere / llama.cpp's — one body all three accept |
//! | native | `POST /api/embed`, `POST /api/rerank`, `POST /api/rag` | ollama's embed; the rerank body above; RAG below |
//!
//! `/api/rag` is the whole retrieval half of RAG in one stateless call:
//! chunk the documents the request carries, embed them and the query, keep
//! the nearest chunks, rerank those, and answer with the best passages plus
//! the system message that grounds a chat model in them (numbered sources,
//! cite-or-say-so). The caller sends that message with its chat request to
//! any chat model here. Nothing is stored server-side: every listener this
//! server binds is reachable without authentication in its deployment, and a
//! document store anyone can read is not a feature.
//!
//! # Where retrieval models run
//!
//! An embedder and a reranker each have a slot of their own, so neither
//! evicts the chat model (or each other). **The card belongs to the chat
//! model**: its placement spends the card down to a measured margin, and it
//! measures our pool by the bytes *in use* — a second model's freed-but-held
//! pool pages (~1.5 GiB per 0.6B after a 1000-token forward, even after a
//! cleanup) read to it as free, and the next chat load ran out of device
//! memory on a card it had planned to fit. Measured on this change, twice.
//! So each retrieval model is placed when it loads:
//!
//! - on the **accelerator** only while no chat model holds or is loading
//!   onto it ([`engine::chat_on_accelerator`]) and the measured card budget
//!   ([`engine::accelerator_room`]) holds its f32 weights plus working room;
//!   its loads and forwards then run *holding the chat slot*
//!   ([`engine::hold_slot`]), so no chat load can start under them, and give
//!   their pool pages back before letting go. A chat model that is about to
//!   load evicts it from the card first ([`evict_from_accelerator`]);
//! - on the **CPU** otherwise, concurrently with chat, when free RAM holds
//!   it; refused by name when it does not.
//!
//! A retrieval model idle for [`IDLE`] is dropped (the placement watch
//! calls [`drop_idle`]). One retrieval forward runs at a time.
//!
//! # Prompts
//!
//! The ollama and `OpenAI` surfaces embed text **as sent** — those callers
//! already add any prefix their model needs (Open `WebUI`'s query/content
//! prefixes, a framework's instruction), and adding another would be wrong
//! twice. A caller that wants the checkpoint's own convention says which side
//! of the pair a text is with `input_type` (`query` / `document`, Cohere's
//! `search_query` / `search_document` and Voyage's spellings accepted), and
//! optionally `instruction` for a custom retrieval task. `/api/rag` always
//! applies the convention — it knows which text is the query.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::response::Response;
use base64::Engine as _;
use burn::tensor::Device;
use mummu::embed::{Embedder, TextKind};
use mummu::manage::ModelManager;
use mummu::rag::{self, ChunkOptions, Passage};
use mummu::registry::{ModelSpec, Task};
use mummu::rerank::Reranker;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::engine::{self, BackendChoice};
use crate::recovery::{self, InFlight};
use crate::{json_response, millis, models_root, nanos, parse_json};

// ---------------------------------------------------------------------------
// Residency
// ---------------------------------------------------------------------------

/// A retrieval model unused this long is dropped.
pub const IDLE: Duration = Duration::from_secs(600);

/// What a retrieval forward needs beyond its weights: activations, the KV
/// state of a chunked prefill, and allocator slack. A 4096-token input to a
/// 0.6B is ~0.9 GiB of f32 KV alone.
const WORKING_BYTES: u64 = 3 << 29;

struct Resident<T> {
    dir: PathBuf,
    name: String,
    model: Arc<T>,
    backend: BackendChoice,
    bytes: u64,
    used: Instant,
    /// [`recovery::fault_epoch`] when the load began. A device failure
    /// since then may have left these weights pointing at memory that was
    /// never initialized — the chat model's rule (`engine::Loaded`), and the
    /// same answer: never served again, reloaded instead.
    epoch: u64,
}

/// One retrieval model kind: how it loads and which slot it lives in.
trait Retrieval: Send + Sync + Sized + 'static {
    const TASK: Task;
    fn slot() -> &'static Mutex<Option<Resident<Self>>>;
    fn load(dir: &Path, device: &Device) -> Result<Self, String>;
}

static EMBEDDER: Mutex<Option<Resident<Embedder>>> = Mutex::new(None);
static RERANKER: Mutex<Option<Resident<Reranker>>> = Mutex::new(None);

/// One retrieval forward (or load) at a time, across both kinds: two would
/// double the activations the placement decision budgeted for, and the CPU
/// path is already as wide as the host.
static RUN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

impl Retrieval for Embedder {
    const TASK: Task = Task::Embed;
    fn slot() -> &'static Mutex<Option<Resident<Self>>> {
        &EMBEDDER
    }
    fn load(dir: &Path, device: &Device) -> Result<Self, String> {
        Self::load_from_dir(dir, device).map_err(|e| e.to_string())
    }
}

impl Retrieval for Reranker {
    const TASK: Task = Task::Rerank;
    fn slot() -> &'static Mutex<Option<Resident<Self>>> {
        &RERANKER
    }
    fn load(dir: &Path, device: &Device) -> Result<Self, String> {
        Self::load_from_dir(dir, device).map_err(|e| e.to_string())
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Drop a resident model and, when it was on the accelerator, hand its
/// pages back to the driver so the chat model's next placement sees them.
fn release<T>(r: Resident<T>) {
    let backend = r.backend;
    let name = r.name.clone();
    drop(r);
    if backend != BackendChoice::Cpu {
        engine::device_of(backend).memory_cleanup();
    }
    eprintln!("[mummu-serve] retrieval: {name} unloaded");
}

/// Drop every retrieval model unused for [`IDLE`]. Called from the
/// placement watch's idle tick.
pub fn drop_idle() {
    fn one<T>(slot: &Mutex<Option<Resident<T>>>) {
        let idle = lock(slot).as_ref().is_some_and(|r| r.used.elapsed() > IDLE);
        if !idle {
            return;
        }
        // Not while a request uses it: that request holds RUN.
        let Ok(_run) = RUN.try_lock() else {
            return;
        };
        let taken = lock(slot).take();
        if let Some(r) = taken {
            release(r);
        }
    }
    one(&EMBEDDER);
    one(&RERANKER);
}

/// Drop the retrieval models that are on an accelerator: a chat model is
/// about to load there, and the card is its (see the module docs). A
/// request mid-forward keeps its own `Arc` and finishes first.
pub fn evict_from_accelerator() {
    if let Some(r) = take_if_on_card(&EMBEDDER) {
        eprintln!(
            "[mummu-serve] retrieval: {} leaves the card to the chat model",
            r.name
        );
        release(r);
    }
    if let Some(r) = take_if_on_card(&RERANKER) {
        eprintln!(
            "[mummu-serve] retrieval: {} leaves the card to the chat model",
            r.name
        );
        release(r);
    }
}

fn take_if_on_card<T>(slot: &Mutex<Option<Resident<T>>>) -> Option<Resident<T>> {
    let mut g = lock(slot);
    if g.as_ref().is_some_and(|r| r.backend != BackendChoice::Cpu) {
        g.take()
    } else {
        None
    }
}

/// Drop every retrieval model now (`/api/unload`). A request mid-forward
/// keeps its own `Arc` and finishes; the model goes when it does.
pub fn unload_all() {
    let embedder = lock(&EMBEDDER).take();
    if let Some(r) = embedder {
        release(r);
    }
    let reranker = lock(&RERANKER).take();
    if let Some(r) = reranker {
        release(r);
    }
}

/// The resident retrieval models: (name, task, device label, bytes) — what
/// `/api/ps` adds to the chat model.
#[must_use]
pub fn resident() -> Vec<(String, Task, &'static str, u64)> {
    let mut out = Vec::new();
    if let Some(r) = lock(&EMBEDDER).as_ref() {
        out.push((
            r.name.clone(),
            Task::Embed,
            engine::label_of(r.backend),
            r.bytes,
        ));
    }
    if let Some(r) = lock(&RERANKER).as_ref() {
        out.push((
            r.name.clone(),
            Task::Rerank,
            engine::label_of(r.backend),
            r.bytes,
        ));
    }
    out
}

/// Bytes a model's weights take resident: the checkpoint's float weights
/// widen to f32 (bf16 doubles), a GGUF dequantizes to f32 (~4 bytes per
/// stored byte-ish, bounded here by its f32 size estimate of 4x).
fn resident_bytes(spec: &ModelSpec, root: &Path) -> u64 {
    let dir = spec.dir(root);
    let on_disk: u64 = std::fs::read_dir(&dir).map_or(0, |rd| {
        rd.flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|x| x == "safetensors" || x == "gguf")
            })
            // `fs::metadata`, not `DirEntry::metadata`: a staged or
            // shared models dir links its weights, and the link's own
            // size is the length of a path.
            .filter_map(|e| std::fs::metadata(e.path()).ok())
            .map(|m| m.len())
            .sum()
    });
    let factor = match spec.format {
        mummu::registry::WeightFormat::Safetensors => 2,
        mummu::registry::WeightFormat::Gguf { .. } => 4,
    };
    on_disk.saturating_mul(factor)
}

/// Where a retrieval model needing `need` bytes runs: the accelerator when
/// the live card budget holds it, else the CPU when free RAM does.
fn place(need: u64) -> Result<BackendChoice, String> {
    let chat = engine::backend_choice();
    let room = if engine::chat_on_accelerator() {
        0
    } else {
        engine::accelerator_room(chat)
    };
    let ram = crate::status::mem_available_bytes().map(|b| b / 100 * 85);
    place_with(chat, room, ram, need)
}

/// [`place`] on given readings: `room` is the card budget left beside the
/// chat model, `ram` the host bytes free to spend (`None`: unknown, which
/// is not a reason to refuse).
fn place_with(
    chat: BackendChoice,
    room: u64,
    ram: Option<u64>,
    need: u64,
) -> Result<BackendChoice, String> {
    if chat != BackendChoice::Cpu && room >= need {
        return Ok(chat);
    }
    match ram {
        Some(free) if free < need => Err(format!(
            "not enough free memory to load it: it needs ~{:.1} GiB, the accelerator has \
             {:.1} GiB to spare beside the chat model and the host {:.1} GiB",
            gib(need),
            gib(room),
            gib(free)
        )),
        _ => Ok(BackendChoice::Cpu),
    }
}

fn gib(bytes: u64) -> f64 {
    mummu_num::f64_from_u64(bytes) / f64::from(1u32 << 30)
}

/// Where `spec` would run if it were loaded now, without loading it — what
/// the default-model choice ranks on.
fn would_place(spec: &ModelSpec, root: &Path) -> Option<BackendChoice> {
    let slot_backend = |dir: &Path| -> Option<BackendChoice> {
        match spec.task() {
            Task::Embed => lock(&EMBEDDER)
                .as_ref()
                .filter(|r| r.dir == dir)
                .map(|r| r.backend),
            Task::Rerank => lock(&RERANKER)
                .as_ref()
                .filter(|r| r.dir == dir)
                .map(|r| r.backend),
            Task::Generate => None,
        }
    };
    let dir = spec.dir(root);
    slot_backend(&dir).or_else(|| place(resident_bytes(spec, root) + WORKING_BYTES).ok())
}

/// The model, loaded (or found resident) for `spec`, and where it runs.
/// The caller holds [`RUN`].
async fn acquire<T: Retrieval>(
    spec: &ModelSpec,
    root: &Path,
) -> Result<(Arc<T>, BackendChoice, Duration), String> {
    let dir = spec.dir(root);
    {
        let mut slot = lock(T::slot());
        if let Some(r) = slot.as_mut()
            && r.dir == dir
            && r.epoch == recovery::fault_epoch()
        {
            r.used = Instant::now();
            return Ok((Arc::clone(&r.model), r.backend, Duration::ZERO));
        }
        // A different model of the same kind: it goes first, so the two
        // never hold memory at once.
        if let Some(old) = slot.take() {
            drop(slot);
            release(old);
        }
    }
    let bytes = resident_bytes(spec, root);
    let backend = place(bytes + WORKING_BYTES).map_err(|e| format!("{}: {e}", spec.name))?;
    let epoch = recovery::fault_epoch();
    let started = Instant::now();
    // An accelerator load holds the chat slot for the same reason the
    // forwards do: its allocations must not interleave with a generation's.
    let hold = if backend == BackendChoice::Cpu {
        None
    } else {
        Some(engine::hold_slot().await)
    };
    let load_dir = dir.clone();
    let model = crate::blocking(move || {
        let device = engine::device_of(backend);
        let model = T::load(&load_dir, &device);
        // The load's staging buffers (the bf16 → f32 casts) are free pages
        // now; hand them back for the same reason a forward does (see `run`).
        if backend != BackendChoice::Cpu {
            device.memory_cleanup();
        }
        model
    })
    .await
    .map_err(|e| format!("{}: {e}", spec.name))?;
    drop(hold);
    let load = started.elapsed();
    eprintln!(
        "[mummu-serve] retrieval: {} loaded on {} in {load:?} (~{:.1} GiB)",
        spec.name,
        engine::label_of(backend),
        gib(bytes)
    );
    let model = Arc::new(model);
    *lock(T::slot()) = Some(Resident {
        dir,
        name: spec.name.clone(),
        model: Arc::clone(&model),
        backend,
        bytes,
        used: Instant::now(),
        epoch,
    });
    Ok((model, backend, load))
}

/// A failure, with the status it is answered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub status: u16,
    pub message: String,
}

impl Failure {
    fn bad(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }
    fn server(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
        }
    }
}

/// How long the parts of one retrieval call took.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub queue: Duration,
    pub load: Duration,
    pub compute: Duration,
    pub device: &'static str,
}

/// Run `work` on the model for `spec` — loaded or found resident, on the
/// device it was placed on, holding the chat slot when that is the
/// accelerator — on a blocking thread, with every way it can fail turned into
/// a [`Failure`] (a device failure is decided by [`recovery::contain`], which
/// also drops the chat model when the backend is gone; the retrieval model
/// goes with it).
async fn run<T, R, F>(spec: &ModelSpec, work: F) -> Result<(R, Timing), Failure>
where
    T: Retrieval,
    R: Send + 'static,
    F: FnOnce(&T, &Device) -> Result<R, Failure> + Send + 'static,
{
    if recovery::restarting() {
        return Err(Failure {
            status: 503,
            message: recovery::restarting_message().into(),
        });
    }
    debug_assert_eq!(spec.task(), T::TASK, "{} run as the wrong kind", spec.name);
    let root = models_root();
    let spec = spec.clone();
    let name = spec.name.clone();
    let queued = Instant::now();
    let outcome = recovery::contain(&name, async move {
        let _run = RUN.lock().await;
        let queue = queued.elapsed();
        let (model, backend, load) = acquire::<T>(&spec, &root)
            .await
            .map_err(recovery::ChatError::request)?;
        let hold = if backend == BackendChoice::Cpu {
            None
        } else {
            Some(engine::hold_slot().await)
        };
        let started = Instant::now();
        let result = crate::blocking(move || {
            let device = engine::device_of(backend);
            let out = work(&model, &device);
            // Give the forward's activation pages back before the chat slot
            // is released. The pool keeps freed pages reserved, and the chat
            // planner's budget counts only bytes IN USE: pages a 1000-token
            // prefill left behind read as free to it while the driver has
            // none, and the next chat load ran out of device memory on a
            // card it had planned to fit (measured on this change, before
            // this line: 11.6 GiB held after two 0.6B retrieval models).
            if backend != BackendChoice::Cpu {
                device.memory_cleanup();
            }
            out
        })
        .await;
        drop(hold);
        let timing = Timing {
            queue,
            load,
            compute: started.elapsed(),
            device: engine::label_of(backend),
        };
        Ok((result, timing))
    })
    .await;
    match outcome {
        Ok((Ok(r), timing)) => Ok((r, timing)),
        Ok((Err(f), _)) => Err(f),
        Err(e) => {
            if e.is_device()
                && let Some(r) = lock(T::slot()).take()
            {
                release(r);
            }
            Err(Failure {
                status: if e.is_device() { 503 } else { 500 },
                message: e.message,
            })
        }
    }
}

/// Block on an async library call from the blocking thread [`run`] puts a
/// retrieval forward on.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Handle::current().block_on(f)
}

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

/// The catalog spec a request names, if it is installed and serves `task`.
fn named(name: &str, task: Task) -> Result<ModelSpec, Failure> {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    let bare = name.strip_suffix(":latest").unwrap_or(name);
    let Some(spec) = manager
        .catalog()
        .iter()
        .find(|s| s.name == name || s.name == bare)
        .cloned()
    else {
        return Err(Failure {
            status: 404,
            message: format!("model {name:?} not found"),
        });
    };
    if !engine::is_installed(&spec, &root) {
        return Err(Failure {
            status: 404,
            message: format!("model {name:?} is not installed — pull it first"),
        });
    }
    if spec.task() != task {
        let what = match task {
            Task::Embed => "an embedding model",
            Task::Rerank => "a reranker",
            Task::Generate => "a chat model",
        };
        return Err(Failure::bad(format!("{name:?} is not {what}")));
    }
    Ok(spec)
}

/// The model a request that names none gets for `task`: the best installed
/// one that would run on the accelerator right now, else — everything on
/// the CPU — the smallest, because there a reranker's cost is per document
/// and the 4B is ~5x the 0.6B's (measured: 2.2 s vs 0.41 s per pair).
fn default_model(task: Task) -> Result<ModelSpec, Failure> {
    let root = models_root();
    let manager = ModelManager::new(root.clone());
    // Quality order: the retrieval tier's own entries (best first, as the
    // registry lists them) ahead of the legacy MiniLM.
    let mut installed: Vec<ModelSpec> = manager
        .catalog()
        .iter()
        .filter(|s| s.task() == task && engine::is_installed(s, &root))
        .cloned()
        .collect();
    installed.sort_by_key(|s| u8::from(s.architecture == mummu::registry::Architecture::MiniLm));
    if installed.is_empty() {
        let what = match task {
            Task::Embed => "embedding model (pull harrier-oss-v1-0.6b)",
            Task::Rerank => "reranker (pull qwen3-reranker-0.6b or qwen3-reranker-4b)",
            Task::Generate => "chat model",
        };
        return Err(Failure {
            status: 409,
            message: format!("no {what} is installed"),
        });
    }
    if let Some(on_card) = installed
        .iter()
        .find(|s| would_place(s, &root).is_some_and(|b| b != BackendChoice::Cpu))
    {
        return Ok(on_card.clone());
    }
    Ok(installed
        .iter()
        .min_by_key(|s| s.disk_bytes_estimate)
        .cloned()
        .expect("non-empty"))
}

fn model_for(name: Option<&str>, task: Task) -> Result<ModelSpec, Failure> {
    name.map(str::trim)
        .filter(|n| !n.is_empty())
        .map_or_else(|| default_model(task), |n| named(n, task))
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Most inputs one embedding request may carry.
const MAX_INPUTS: usize = 2048;

/// One input: text, or token ids (`OpenAI` accepts both).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    Text(String),
    Ids(Vec<u32>),
}

/// `input` in every shape the two APIs allow: a string, a list of strings,
/// a list of token ids, or a list of lists of token ids.
fn inputs_of(v: &Value) -> Result<Vec<Input>, Failure> {
    let ids = |a: &[Value]| -> Option<Vec<u32>> {
        a.iter()
            .map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok()))
            .collect()
    };
    let out = match v {
        Value::String(s) => vec![Input::Text(s.clone())],
        Value::Array(a) if a.is_empty() => Vec::new(),
        Value::Array(a) if a.iter().all(Value::is_string) => a
            .iter()
            .map(|s| Input::Text(s.as_str().unwrap_or_default().to_owned()))
            .collect(),
        Value::Array(a) if a.iter().all(Value::is_number) => {
            vec![Input::Ids(
                ids(a).ok_or_else(|| Failure::bad("token ids must be u32"))?,
            )]
        }
        Value::Array(a) if a.iter().all(Value::is_array) => a
            .iter()
            .map(|x| {
                ids(x.as_array().map_or(&[][..], Vec::as_slice))
                    .map(Input::Ids)
                    .ok_or_else(|| Failure::bad("token ids must be u32"))
            })
            .collect::<Result<_, _>>()?,
        Value::Null => return Err(Failure::bad("input is required")),
        _ => {
            return Err(Failure::bad(
                "input must be a string, a list of strings, or token ids",
            ));
        }
    };
    if out.len() > MAX_INPUTS {
        return Err(Failure::bad(format!(
            "{} inputs is over the {MAX_INPUTS} per request",
            out.len()
        )));
    }
    if out.iter().any(|i| match i {
        Input::Text(t) => t.is_empty(),
        Input::Ids(v) => v.is_empty(),
    }) {
        return Err(Failure::bad("an input is empty"));
    }
    Ok(out)
}

/// `input_type`, in the spellings the embedding APIs use for it.
fn kind_of(input_type: Option<&str>) -> Result<Option<TextKind>, Failure> {
    match input_type.map(str::to_ascii_lowercase).as_deref() {
        None | Some("" | "none" | "raw") => Ok(None),
        Some("query" | "search_query" | "retrieval.query") => Ok(Some(TextKind::Query)),
        Some("document" | "search_document" | "passage" | "retrieval.passage") => {
            Ok(Some(TextKind::Document))
        }
        Some(other) => Err(Failure::bad(format!(
            "input_type {other:?} is not one of query, document"
        ))),
    }
}

/// What every embedding surface asks, once parsed.
#[derive(Debug, Clone)]
pub struct EmbedAsk {
    pub model: Option<String>,
    pub inputs: Vec<Input>,
    pub kind: Option<TextKind>,
    pub instruction: Option<String>,
    pub dimensions: Option<usize>,
    /// Truncate an over-long input (ollama's default) rather than refusing it.
    pub truncate: bool,
}

/// What an embedding call produced.
pub struct Embedded {
    pub model: String,
    pub vectors: Vec<Vec<f32>>,
    pub tokens: usize,
    pub timing: Timing,
}

/// Matryoshka truncation: the leading `d` components, renormalized.
fn shorten(v: &mut Vec<f32>, d: usize) {
    v.truncate(d);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Embed every input of `ask`.
///
/// # Errors
///
/// A [`Failure`] for an unknown or non-embedding model, a bad dimension, an
/// over-long input when truncation is off, or a model/device failure.
pub async fn embed(ask: EmbedAsk, cancel: Arc<AtomicBool>) -> Result<Embedded, Failure> {
    let spec = model_for(ask.model.as_deref(), Task::Embed)?;
    let name = spec.name.clone();
    let ((vectors, tokens), timing) = run::<Embedder, _, _>(&spec, move |e, device| {
        if let Some(d) = ask.dimensions
            && (d == 0 || d > e.dims())
        {
            return Err(Failure::bad(format!(
                "dimensions must be between 1 and {}, got {d}",
                e.dims()
            )));
        }
        let mut vectors = Vec::with_capacity(ask.inputs.len());
        let mut tokens = 0;
        for input in &ask.inputs {
            if cancel.load(SeqCst) {
                return Err(Failure::server("cancelled"));
            }
            let (mut vector, n, truncated) = match input {
                Input::Text(t) => {
                    let text = match ask.kind {
                        Some(k) => e.prompted(t, k, ask.instruction.as_deref()),
                        None => t.clone(),
                    };
                    let out = block_on(e.embed_prompted(&text, device)).map_err(Failure::server)?;
                    (out.vector, out.tokens, out.truncated)
                }
                Input::Ids(ids) => {
                    let n = ids.len().min(e.max_tokens());
                    let v = block_on(e.embed_ids(ids, device)).map_err(Failure::server)?;
                    (v, n, ids.len() > e.max_tokens())
                }
            };
            if truncated && !ask.truncate {
                return Err(Failure::bad(format!(
                    "input length exceeds the context length ({} tokens)",
                    e.max_tokens()
                )));
            }
            if let Some(d) = ask.dimensions {
                shorten(&mut vector, d);
            }
            tokens += n;
            vectors.push(vector);
        }
        Ok((vectors, tokens))
    })
    .await?;
    Ok(Embedded {
        model: name,
        vectors,
        tokens,
        timing,
    })
}

// ---------------------------------------------------------------------------
// Reranking
// ---------------------------------------------------------------------------

/// Most documents one rerank request may carry.
const MAX_DOCUMENTS: usize = 1000;

/// What a rerank produced, best first.
pub struct Reranked {
    pub model: String,
    pub scores: Vec<mummu::rerank::Score>,
    pub tokens: usize,
    pub timing: Timing,
}

/// Score `documents` against `query`, best first.
///
/// # Errors
///
/// A [`Failure`] for an unknown or non-reranker model, an empty query or
/// list, or a model/device failure.
pub async fn rerank(
    model: Option<&str>,
    query: String,
    documents: Vec<String>,
    instruction: Option<String>,
    cancel: Arc<AtomicBool>,
) -> Result<Reranked, Failure> {
    if query.trim().is_empty() {
        return Err(Failure::bad("query is required"));
    }
    if documents.is_empty() {
        return Err(Failure::bad("documents is empty"));
    }
    if documents.len() > MAX_DOCUMENTS {
        return Err(Failure::bad(format!(
            "{} documents is over the {MAX_DOCUMENTS} per request",
            documents.len()
        )));
    }
    let spec = model_for(model, Task::Rerank)?;
    let name = spec.name.clone();
    let (scores, timing) = run::<Reranker, _, _>(&spec, move |r, device| {
        let scored = block_on(
            r.rank(&query, &documents, instruction.as_deref(), device, |_| {
                if cancel.load(SeqCst) {
                    std::ops::ControlFlow::Break(())
                } else {
                    std::ops::ControlFlow::Continue(())
                }
            }),
        )
        .map_err(Failure::server)?;
        if cancel.load(SeqCst) {
            return Err(Failure::server("cancelled"));
        }
        Ok(scored)
    })
    .await?;
    let tokens = scores.iter().map(|s| s.tokens).sum();
    Ok(Reranked {
        model: name,
        scores,
        tokens,
        timing,
    })
}

// ---------------------------------------------------------------------------
// HTTP: shared plumbing
// ---------------------------------------------------------------------------

/// Sets its flag when dropped — the request future is dropped when the
/// client goes away, which is how a long batch learns to stop.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, SeqCst);
    }
}

fn cancel_flag() -> (Arc<AtomicBool>, CancelOnDrop) {
    let flag = Arc::new(AtomicBool::new(false));
    (Arc::clone(&flag), CancelOnDrop(flag))
}

/// Record one retrieval request in the trace ring (kept out of the chat
/// aggregates — see `trace::stats_json`).
fn trace(
    surface: &'static str,
    model: &str,
    started: Instant,
    timing: Timing,
    tokens: usize,
    outcome: Result<(), &str>,
) {
    crate::trace::Begin {
        surface,
        model: model.to_owned(),
        asked: crate::trace::Asked::default(),
        tools: 0,
        images: 0,
        started,
    }
    .finish(
        timing.device,
        crate::trace::Timings {
            queue_ms: millis(timing.queue),
            load_ms: millis(timing.load),
            prefill_ms: millis(timing.compute),
            prompt_tokens: tokens,
            ..crate::trace::Timings::default()
        },
        false,
        outcome,
    );
}

/// The body every embedding surface reads (a superset; each surface uses
/// its own fields).
#[derive(Deserialize)]
struct EmbedBody {
    #[serde(default)]
    model: Option<String>,
    /// ollama `/api/embed` and `OpenAI`.
    #[serde(default)]
    input: Value,
    /// ollama's legacy `/api/embeddings`.
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    truncate: Option<bool>,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    input_type: Option<String>,
    #[serde(default)]
    instruction: Option<String>,
}

impl EmbedBody {
    fn ask(self, inputs: Vec<Input>) -> Result<EmbedAsk, Failure> {
        Ok(EmbedAsk {
            model: self.model,
            inputs,
            kind: kind_of(self.input_type.as_deref())?,
            instruction: self.instruction.filter(|i| !i.trim().is_empty()),
            dimensions: self.dimensions,
            truncate: self.truncate.unwrap_or(true),
        })
    }
}

fn ollama_error(f: &Failure) -> Response {
    json_response(f.status, &json!({"error": f.message}))
}

fn openai_error(f: &Failure) -> Response {
    let (kind, code) = match f.status {
        404 => ("invalid_request_error", "model_not_found"),
        400 | 409 => ("invalid_request_error", "invalid_request_error"),
        503 => ("server_error", "service_unavailable"),
        _ => ("server_error", "internal_error"),
    };
    json_response(
        f.status,
        &json!({"error": {"message": f.message, "type": kind, "code": code}}),
    )
}

// ---------------------------------------------------------------------------
// HTTP: ollama
// ---------------------------------------------------------------------------

/// `POST /api/embed` (ollama; also the native API's embed route).
pub async fn ollama_embed(body: Bytes) -> Response {
    let started = Instant::now();
    let parsed: EmbedBody = match parse_json(&body) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let ask = match inputs_of(&parsed.input).and_then(|i| parsed.ask(i)) {
        Ok(a) => a,
        Err(f) => return ollama_error(&f),
    };
    let label = ask.model.clone().unwrap_or_default();
    let count = ask.inputs.len();
    eprintln!("[mummu-serve] embed {label}: request accepted ({count} inputs)");
    let _inflight = InFlight::enter();
    let (cancel, _guard) = cancel_flag();
    match embed(ask, cancel).await {
        Ok(e) => {
            trace("embed", &e.model, started, e.timing, e.tokens, Ok(()));
            json_response(
                200,
                &json!({
                    "model": e.model,
                    "embeddings": e.vectors,
                    "total_duration": nanos(started.elapsed()),
                    "load_duration": nanos(e.timing.load),
                    "prompt_eval_count": e.tokens,
                }),
            )
        }
        Err(f) => {
            trace(
                "embed",
                &label,
                started,
                Timing::default(),
                0,
                Err(&f.message),
            );
            ollama_error(&f)
        }
    }
}

/// `POST /api/embeddings` — ollama's legacy single-prompt shape.
pub async fn ollama_embeddings(body: Bytes) -> Response {
    let started = Instant::now();
    let mut parsed: EmbedBody = match parse_json(&body) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Some(prompt) = parsed.prompt.take().filter(|p| !p.is_empty()) else {
        return ollama_error(&Failure::bad("prompt is required"));
    };
    let ask = match parsed.ask(vec![Input::Text(prompt)]) {
        Ok(a) => a,
        Err(f) => return ollama_error(&f),
    };
    let label = ask.model.clone().unwrap_or_default();
    let _inflight = InFlight::enter();
    let (cancel, _guard) = cancel_flag();
    match embed(ask, cancel).await {
        Ok(mut e) => {
            trace("embed", &e.model, started, e.timing, e.tokens, Ok(()));
            json_response(
                200,
                &json!({"embedding": e.vectors.pop().unwrap_or_default()}),
            )
        }
        Err(f) => {
            trace(
                "embed",
                &label,
                started,
                Timing::default(),
                0,
                Err(&f.message),
            );
            ollama_error(&f)
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP: OpenAI
// ---------------------------------------------------------------------------

/// `POST /v1/embeddings`.
pub async fn openai_embeddings(body: Bytes) -> Response {
    let started = Instant::now();
    let parsed: EmbedBody = match parse_json(&body) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let base64 = match parsed.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return openai_error(&Failure::bad(format!(
                "encoding_format {other:?} is not one of float, base64"
            )));
        }
    };
    if parsed.model.as_deref().is_none_or(|m| m.trim().is_empty()) {
        return openai_error(&Failure::bad("model is required"));
    }
    let ask = match inputs_of(&parsed.input).and_then(|i| parsed.ask(i)) {
        Ok(a) => a,
        Err(f) => return openai_error(&f),
    };
    let label = ask.model.clone().unwrap_or_default();
    eprintln!(
        "[mummu-serve] openai embeddings {label}: request accepted ({} inputs)",
        ask.inputs.len()
    );
    let _inflight = InFlight::enter();
    let (cancel, _guard) = cancel_flag();
    match embed(ask, cancel).await {
        Ok(e) => {
            trace("embed", &e.model, started, e.timing, e.tokens, Ok(()));
            let data: Vec<Value> = e
                .vectors
                .iter()
                .enumerate()
                .map(|(index, v)| {
                    let embedding = if base64 {
                        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                        json!(base64::engine::general_purpose::STANDARD.encode(bytes))
                    } else {
                        json!(v)
                    };
                    json!({"object": "embedding", "index": index, "embedding": embedding})
                })
                .collect();
            json_response(
                200,
                &json!({
                    "object": "list",
                    "data": data,
                    "model": e.model,
                    "usage": {"prompt_tokens": e.tokens, "total_tokens": e.tokens},
                }),
            )
        }
        Err(f) => {
            trace(
                "embed",
                &label,
                started,
                Timing::default(),
                0,
                Err(&f.message),
            );
            openai_error(&f)
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP: rerank
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RerankBody {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    query: String,
    #[serde(default)]
    documents: Vec<Value>,
    #[serde(default, alias = "top_k")]
    top_n: Option<usize>,
    #[serde(default)]
    return_documents: Option<bool>,
    #[serde(default)]
    instruction: Option<String>,
}

/// A document in every shape the rerank APIs send: a string, or an object
/// with `text` (Jina, Cohere v2) or `content`.
fn document_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o
            .get("text")
            .or_else(|| o.get("content"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

static RERANK_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `POST /v1/rerank`, `/rerank`, `/api/rerank` — one body that Jina's,
/// Cohere's and llama.cpp's clients all send, one answer they all read:
/// `results[] = {index, relevance_score, document?}`, best first.
pub async fn rerank_endpoint(body: Bytes) -> Response {
    let started = Instant::now();
    let parsed: RerankBody = match parse_json(&body) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Some(documents) = parsed
        .documents
        .iter()
        .map(document_text)
        .collect::<Option<Vec<_>>>()
    else {
        return openai_error(&Failure::bad(
            "documents must be strings or objects with a text field",
        ));
    };
    let label = parsed.model.clone().unwrap_or_default();
    eprintln!(
        "[mummu-serve] rerank {label}: request accepted ({} documents)",
        documents.len()
    );
    let _inflight = InFlight::enter();
    let (cancel, _guard) = cancel_flag();
    let texts = documents.clone();
    let result = rerank(
        parsed.model.as_deref(),
        parsed.query,
        documents,
        parsed.instruction.filter(|i| !i.trim().is_empty()),
        cancel,
    )
    .await;
    match result {
        Ok(r) => {
            trace("rerank", &r.model, started, r.timing, r.tokens, Ok(()));
            let keep = parsed.top_n.unwrap_or(r.scores.len()).min(r.scores.len());
            let with_docs = parsed.return_documents.unwrap_or(false);
            let results: Vec<Value> = r
                .scores
                .iter()
                .take(keep)
                .map(|s| {
                    let mut item = json!({"index": s.index, "relevance_score": s.relevance});
                    if with_docs {
                        item["document"] = json!({"text": texts[s.index]});
                    }
                    item
                })
                .collect();
            json_response(
                200,
                &json!({
                    "id": format!("rerank-{}", RERANK_SEQ.fetch_add(1, SeqCst)),
                    "object": "list",
                    "model": r.model,
                    "results": results,
                    "usage": {"prompt_tokens": r.tokens, "total_tokens": r.tokens},
                }),
            )
        }
        Err(f) => {
            trace(
                "rerank",
                &label,
                started,
                Timing::default(),
                0,
                Err(&f.message),
            );
            openai_error(&f)
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP: RAG
// ---------------------------------------------------------------------------

/// Most passages a RAG answer returns, and candidates it reranks.
const MAX_TOP_K: usize = 50;
const MAX_CANDIDATES: usize = 200;

#[derive(Deserialize)]
struct RagBody {
    #[serde(default)]
    query: String,
    #[serde(default)]
    documents: Vec<Value>,
    #[serde(default)]
    embed_model: Option<String>,
    /// A model name, `false` to skip reranking, or absent for the default.
    #[serde(default)]
    rerank_model: Option<Value>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    candidates: Option<usize>,
    #[serde(default)]
    max_chars: Option<usize>,
    #[serde(default)]
    overlap_chars: Option<usize>,
    #[serde(default)]
    instruction: Option<String>,
}

/// One document the RAG request carries.
struct Doc {
    id: String,
    source: String,
    text: String,
}

fn docs_of(values: &[Value]) -> Result<Vec<Doc>, Failure> {
    values
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            Value::String(s) => Ok(Doc {
                id: format!("{i}"),
                source: format!("document {}", i + 1),
                text: s.clone(),
            }),
            Value::Object(o) => {
                let text = o
                    .get("text")
                    .or_else(|| o.get("content"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| Failure::bad(format!("document {i} has no text")))?;
                let id = o
                    .get("id")
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("{i}"), str::to_owned);
                let source = o
                    .get("source")
                    .or_else(|| o.get("title"))
                    .and_then(Value::as_str)
                    .map_or_else(|| id.clone(), str::to_owned);
                Ok(Doc {
                    id,
                    source,
                    text: text.to_owned(),
                })
            }
            _ => Err(Failure::bad(format!(
                "document {i} must be a string or an object with text"
            ))),
        })
        .collect()
}

/// One retrieved passage.
#[derive(Debug, Clone)]
struct Found {
    doc: usize,
    chunk: usize,
    text: String,
    start: usize,
    end: usize,
    similarity: Option<f32>,
    relevance: Option<f32>,
}

/// `POST /api/rag` — retrieve the passages of `documents` that answer
/// `query`, and the system message that grounds a chat model in them.
pub async fn rag_endpoint(body: Bytes) -> Response {
    let started = Instant::now();
    let parsed: RagBody = match parse_json(&body) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let _inflight = InFlight::enter();
    let (cancel, _guard) = cancel_flag();
    match rag_answer(parsed, cancel).await {
        Ok((body, model, timing, tokens)) => {
            trace("rag", &model, started, timing, tokens, Ok(()));
            json_response(200, &body)
        }
        Err(f) => {
            trace("rag", "", started, Timing::default(), 0, Err(&f.message));
            ollama_error(&f)
        }
    }
}

/// A validated RAG request: its documents chunked, its models chosen.
struct RagPlan {
    query: String,
    docs: Vec<Doc>,
    chunks: Vec<Found>,
    top_k: usize,
    candidates: usize,
    embed: Option<ModelSpec>,
    rerank: Option<ModelSpec>,
    instruction: Option<String>,
}

/// What the stages spent, and which models they ran.
#[derive(Default)]
struct Spent {
    timing: Timing,
    tokens: usize,
    used: serde_json::Map<String, Value>,
}

impl Spent {
    fn add(&mut self, role: &str, model: &str, timing: Timing, tokens: usize) {
        self.used.insert(role.into(), json!(model));
        self.timing = merge(self.timing, timing);
        self.tokens += tokens;
    }
}

/// Validate the request, chunk its documents and choose its models.
fn plan_rag(req: RagBody) -> Result<RagPlan, Failure> {
    if req.query.trim().is_empty() {
        return Err(Failure::bad("query is required"));
    }
    let docs = docs_of(&req.documents)?;
    if docs.is_empty() {
        return Err(Failure::bad("documents is empty"));
    }
    let defaults = ChunkOptions::default();
    let opts = ChunkOptions {
        max_chars: req
            .max_chars
            .unwrap_or(defaults.max_chars)
            .clamp(64, 16_000),
        overlap_chars: req.overlap_chars.unwrap_or(defaults.overlap_chars),
    };
    if opts.overlap_chars >= opts.max_chars {
        return Err(Failure::bad("overlap_chars must be smaller than max_chars"));
    }
    let top_k = req.top_k.unwrap_or(4).clamp(1, MAX_TOP_K);
    let candidates = req.candidates.unwrap_or(20).clamp(top_k, MAX_CANDIDATES);

    let mut chunks: Vec<Found> = Vec::new();
    for (d, doc) in docs.iter().enumerate() {
        for (c, ch) in rag::chunk_text(&doc.text, opts).into_iter().enumerate() {
            chunks.push(Found {
                doc: d,
                chunk: c,
                text: ch.text,
                start: ch.start,
                end: ch.end,
                similarity: None,
                relevance: None,
            });
        }
    }
    if chunks.is_empty() {
        return Err(Failure::bad("the documents hold no text"));
    }
    if chunks.len() > MAX_INPUTS {
        return Err(Failure::bad(format!(
            "the documents chunk into {} pieces, over the {MAX_INPUTS} one request may embed — \
             send fewer or larger chunks",
            chunks.len()
        )));
    }
    // Which models: a name the request gives must exist and serve the task;
    // with none given, the default is used when one is installed and the
    // stage is skipped when none is (a corpus small enough to rerank whole
    // needs no embedder; an embedder alone still retrieves).
    let rerank = match &req.rerank_model {
        Some(Value::Bool(false)) => None,
        Some(Value::String(s)) => Some(named(s, Task::Rerank)?),
        None | Some(Value::Null | Value::Bool(true)) => default_model(Task::Rerank).ok(),
        Some(_) => return Err(Failure::bad("rerank_model must be a name or false")),
    };
    let embed = match req.embed_model.as_deref().map(str::trim) {
        Some(name) if !name.is_empty() => Some(named(name, Task::Embed)?),
        _ => default_model(Task::Embed).ok(),
    };
    match (&embed, &rerank) {
        (None, None) => {
            return Err(Failure {
                status: 409,
                message: "RAG needs an embedding model or a reranker installed \
                          (pull harrier-oss-v1-0.6b and qwen3-reranker-0.6b)"
                    .into(),
            });
        }
        (None, Some(_)) if chunks.len() > MAX_DOCUMENTS => {
            return Err(Failure::bad(format!(
                "{} chunks is too many to rerank without an embedding model to narrow them \
                 first (install harrier-oss-v1-0.6b)",
                chunks.len()
            )));
        }
        _ => {}
    }
    Ok(RagPlan {
        query: req.query,
        docs,
        chunks,
        top_k,
        candidates,
        embed,
        rerank,
        instruction: req.instruction.filter(|i| !i.trim().is_empty()),
    })
}

/// Stage 1: embed every chunk and the query, keep the `keep` nearest.
async fn nearest(
    plan: &RagPlan,
    spec: &ModelSpec,
    keep: usize,
    cancel: &Arc<AtomicBool>,
    spent: &mut Spent,
) -> Result<Vec<Found>, Failure> {
    let docs_ask = EmbedAsk {
        model: Some(spec.name.clone()),
        inputs: plan
            .chunks
            .iter()
            .map(|c| Input::Text(c.text.clone()))
            .collect(),
        kind: Some(TextKind::Document),
        instruction: None,
        dimensions: None,
        truncate: true,
    };
    let query_ask = EmbedAsk {
        inputs: vec![Input::Text(plan.query.clone())],
        kind: Some(TextKind::Query),
        instruction: plan.instruction.clone(),
        ..docs_ask.clone()
    };
    let d = embed(docs_ask, Arc::clone(cancel)).await?;
    let q = embed(query_ask, Arc::clone(cancel)).await?;
    spent.add("embed", &d.model, d.timing, d.tokens);
    spent.add("embed", &q.model, q.timing, q.tokens);
    let dims = q.vectors.first().map_or(0, Vec::len);
    let mut index = rag::Index::new(d.model.clone(), dims.max(1));
    for (i, v) in d.vectors.iter().enumerate() {
        let entry = rag::Entry {
            doc: format!("{i}"),
            chunk: i,
            text: String::new(),
            meta: Value::Null,
        };
        index
            .add(v, entry)
            .map_err(|e| Failure::server(e.to_string()))?;
    }
    Ok(index
        .search(&q.vectors[0], keep)
        .map_err(|e| Failure::server(e.to_string()))?
        .into_iter()
        .map(|h| {
            let mut f = plan.chunks[h.slot].clone();
            f.similarity = Some(h.score);
            f
        })
        .collect())
}

/// Stage 2: rerank `pool`, best first.
async fn reranked(
    plan: &RagPlan,
    spec: &ModelSpec,
    pool: &[Found],
    cancel: &Arc<AtomicBool>,
    spent: &mut Spent,
) -> Result<Vec<Found>, Failure> {
    let texts: Vec<String> = pool.iter().map(|f| f.text.clone()).collect();
    let r = rerank(
        Some(&spec.name),
        plan.query.clone(),
        texts,
        plan.instruction.clone(),
        Arc::clone(cancel),
    )
    .await?;
    spent.add("rerank", &r.model, r.timing, r.tokens);
    Ok(r.scores
        .iter()
        .map(|s| {
            let mut f = pool[s.index].clone();
            f.relevance = Some(s.relevance);
            f
        })
        .collect())
}

/// The answer body: the passages, and the grounding message built from them.
fn rag_body(docs: &[Doc], pool: &[Found], used: &serde_json::Map<String, Value>) -> Value {
    let passages: Vec<Passage<'_>> = pool
        .iter()
        .map(|f| Passage {
            text: &f.text,
            source: &docs[f.doc].source,
        })
        .collect();
    let system = rag::grounding_system_prompt(&passages);
    let out: Vec<Value> = pool
        .iter()
        .enumerate()
        .map(|(n, f)| {
            json!({
                "citation": n + 1,
                "id": docs[f.doc].id,
                "source": docs[f.doc].source,
                "chunk": f.chunk,
                "start": f.start,
                "end": f.end,
                "text": f.text,
                "similarity": f.similarity,
                "relevance": f.relevance,
            })
        })
        .collect();
    json!({
        "passages": out,
        "system_prompt": system,
        "messages": [{"role": "system", "content": system}],
        "models": used,
    })
}

async fn rag_answer(
    req: RagBody,
    cancel: Arc<AtomicBool>,
) -> Result<(Value, String, Timing, usize), Failure> {
    let plan = plan_rag(req)?;
    let mut spent = Spent::default();
    // Embedding narrows the set to `candidates` for the reranker (or to the
    // answer itself when nothing reranks); a set already that small skips it.
    let needs_embedding = plan.chunks.len() > plan.candidates || plan.rerank.is_none();
    let mut pool = match &plan.embed {
        Some(spec) if needs_embedding => {
            let keep = if plan.rerank.is_some() {
                plan.candidates
            } else {
                plan.top_k
            };
            nearest(&plan, spec, keep, &cancel, &mut spent).await?
        }
        _ => plan.chunks.clone(),
    };
    if let Some(spec) = &plan.rerank {
        pool = reranked(&plan, spec, &pool, &cancel, &mut spent).await?;
    }
    pool.truncate(plan.top_k);
    let label = spent
        .used
        .values()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join("+");
    let body = rag_body(&plan.docs, &pool, &spent.used);
    Ok((body, label, spent.timing, spent.tokens))
}

/// Two calls' timings as one: durations add, the device is the last one's.
fn merge(a: Timing, b: Timing) -> Timing {
    Timing {
        queue: a.queue + b.queue,
        load: a.load + b.load,
        compute: a.compute + b.compute,
        device: if b.device.is_empty() {
            a.device
        } else {
            b.device
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_input_shape_both_apis_allow_parses() {
        assert_eq!(
            inputs_of(&json!("hi")).unwrap(),
            vec![Input::Text("hi".into())]
        );
        assert_eq!(
            inputs_of(&json!(["a", "b"])).unwrap(),
            vec![Input::Text("a".into()), Input::Text("b".into())]
        );
        assert_eq!(
            inputs_of(&json!([1, 2, 3])).unwrap(),
            vec![Input::Ids(vec![1, 2, 3])]
        );
        assert_eq!(
            inputs_of(&json!([[1], [2, 3]])).unwrap(),
            vec![Input::Ids(vec![1]), Input::Ids(vec![2, 3])]
        );
        assert_eq!(
            inputs_of(&json!([])).unwrap(),
            [] as [crate::retrieval::Input; 0]
        );
        assert!(inputs_of(&json!(null)).is_err());
        assert!(inputs_of(&json!(["a", ""])).is_err(), "empty input");
        assert!(inputs_of(&json!([-1])).is_err(), "negative id");
        assert!(inputs_of(&json!({"text": "x"})).is_err());
        let too_many: Vec<&str> = vec!["x"; MAX_INPUTS + 1];
        assert!(inputs_of(&json!(too_many)).is_err());
    }

    #[test]
    fn input_type_reads_every_vendors_spelling() {
        for q in ["query", "search_query", "retrieval.query", "QUERY"] {
            assert_eq!(kind_of(Some(q)).unwrap(), Some(TextKind::Query), "{q}");
        }
        for d in [
            "document",
            "search_document",
            "passage",
            "retrieval.passage",
        ] {
            assert_eq!(kind_of(Some(d)).unwrap(), Some(TextKind::Document), "{d}");
        }
        assert_eq!(kind_of(None).unwrap(), None);
        assert!(kind_of(Some("classification")).is_err());
    }

    #[test]
    fn shortening_keeps_the_head_and_renormalizes() {
        let mut v = vec![0.6, 0.8, 0.0, 0.0];
        shorten(&mut v, 1);
        assert_eq!(v, vec![1.0]);
        let mut z = vec![0.0, 0.0, 1.0];
        shorten(&mut z, 2);
        assert_eq!(z, vec![0.0, 0.0], "a zero head stays zero, not NaN");
    }

    #[test]
    fn rerank_documents_come_in_every_vendors_shape() {
        assert_eq!(document_text(&json!("a")).as_deref(), Some("a"));
        assert_eq!(document_text(&json!({"text": "b"})).as_deref(), Some("b"));
        assert_eq!(
            document_text(&json!({"content": "c"})).as_deref(),
            Some("c")
        );
        assert_eq!(document_text(&json!(3)), None);
    }

    #[test]
    fn rag_documents_get_ids_and_sources() {
        let docs = docs_of(&[
            json!("plain"),
            json!({"id": "a.md", "text": "x"}),
            json!({"text": "y", "title": "Y"}),
        ])
        .unwrap();
        assert_eq!(
            (docs[0].id.as_str(), docs[0].source.as_str()),
            ("0", "document 1")
        );
        assert_eq!(
            (docs[1].id.as_str(), docs[1].source.as_str()),
            ("a.md", "a.md")
        );
        assert_eq!(docs[2].source, "Y");
        assert!(docs_of(&[json!({"id": "no text"})]).is_err());
    }

    /// The accelerator only with room beside the chat model; the CPU when
    /// RAM holds it (or nobody can say); refused by name otherwise.
    #[test]
    fn placement_takes_the_card_only_with_room_and_refuses_by_name() {
        let gpu = BackendChoice::Wgpu;
        let gb = 1u64 << 30;
        assert_eq!(place_with(gpu, 4 * gb, Some(64 * gb), 3 * gb), Ok(gpu));
        assert_eq!(
            place_with(gpu, 2 * gb, Some(64 * gb), 3 * gb),
            Ok(BackendChoice::Cpu),
            "a card the chat model filled sends it to the host"
        );
        assert_eq!(
            place_with(BackendChoice::Cpu, 64 * gb, Some(64 * gb), 3 * gb),
            Ok(BackendChoice::Cpu),
            "a CPU server never reaches for a card"
        );
        assert_eq!(place_with(gpu, 0, None, 3 * gb), Ok(BackendChoice::Cpu));
        let err = place_with(gpu, gb, Some(2 * gb), 3 * gb).unwrap_err();
        assert!(err.contains("not enough free memory"), "{err}");
    }
}
