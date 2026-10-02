//! The decode thread: generations on a model wholly on the card, batched,
//! with their captured steps kept between requests.
//!
//! A captured graph replays on the cubecl stream of the thread that recorded
//! it, and requests land on whichever tokio worker is free — so a graph that
//! is to outlive its request needs a thread of its own. This is that thread.
//! A request that holds the model slot (`drive`) hands it over with its
//! generation ([`start`]); from then on the thread holds the slot, and every
//! request for the same model that arrives while it does joins the running
//! batch between steps ([`join`]) instead of queueing for the slot. The
//! thread steps every live sequence in one dispatch
//! ([`mummu::batch::Batcher`]), streams each one's tokens back to its
//! request, and releases the slot when the batch is empty.
//!
//! The batcher — its state and graphs — stays cached after the slot is
//! released, for the next session on the same load of the same model, as
//! long as no layer moved in between ([`super::placement::epoch`]), and at
//! most [`KEEP_IDLE`]. [`release`] drops it at once (the card is wanted
//! elsewhere); [`drain`] stops a session admitting, so whatever waits for
//! the slot gets it once the live sequences finish. A session also stops
//! admitting after [`SESSION_SPAN`] on its own, so placement's idle
//! maintenance is never starved by a steady stream of requests.
//!
//! A failure while the thread holds the slot is decided there, exactly as
//! `serve_held` decides one: recorded, the model evicted through the guard,
//! and every sequence in the batch told.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use mummu::batch::{Admission, Batcher, Event, SeqId};
use mummu::cache::SlotGuard;
use mummu::capture::{StaticDecode, StepMode};
use mummu::constrain::Constraint;
use mummu::decode::SamplerOptions;
use mummu::models::{qwen3, qwen35};
use tokenizers::Tokenizer;

use super::{AnyLm, BackendChoice, Loaded, device_of, evict_held, recovery};
use crate::recovery::{ChatError, DeviceKey};

/// How long a session keeps admitting new requests before it lets the slot
/// go (once its live sequences finish).
pub(super) const SESSION_SPAN: Duration = Duration::from_secs(30);

/// How long a cached batcher outlives its last session.
pub(super) const KEEP_IDLE: Duration = Duration::from_secs(120);

/// Sequences one batch decodes at once (`MUMMU_DECODE_SLOTS`, default 8).
fn max_slots() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MUMMU_DECODE_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(8)
    })
}

/// One generation for the thread.
pub(super) struct Job {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub opts: SamplerOptions,
    pub constraint: Option<Box<dyn Constraint>>,
    pub updates: tokio::sync::mpsc::UnboundedSender<Update>,
}

/// What the thread tells a request about its generation.
pub(super) enum Update {
    Token(u32),
    /// The generation is over: it stopped, or it failed. A device failure
    /// arrives already decided (the model was evicted on the thread).
    Done(Result<(), ChatError>),
}

/// What a request needs of the session its generation runs in.
#[derive(Clone)]
pub(super) struct SessionInfo {
    pub key: PathBuf,
    pub tokenizer: Arc<Tokenizer>,
    pub devices: Vec<DeviceKey>,
    pub backend: BackendChoice,
    /// The longest prompt + generation the model takes.
    pub context: usize,
}

enum Msg {
    Start {
        guard: SlotGuard<'static, Loaded>,
        key: PathBuf,
        model: String,
        job: Job,
    },
    Join(Job),
    /// Drop the cached batcher; acknowledged on the sender once done.
    Release(Option<mpsc::Sender<()>>),
}

/// The running session, for [`join`]: published while it admits.
struct Registry {
    info: Option<SessionInfo>,
    accepting: bool,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    info: None,
    accepting: false,
});

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn sender() -> &'static mpsc::Sender<Msg> {
    static TX: OnceLock<mpsc::Sender<Msg>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("mummu-decode".to_owned())
            .spawn(move || Thread::default().run(&rx))
            .expect("spawn the decode thread");
        tx
    })
}

/// Whether `m` decodes through the thread for a request of `prompt_len`
/// tokens and `max_tokens`: a Qwen3 / qwen35 model wholly on the card, a
/// static step mode, no images.
pub(super) fn applies(m: &Loaded, prompt_len: usize, max_tokens: usize, images: bool) -> bool {
    if images || m.backend == BackendChoice::Cpu || mummu::capture::step_mode() == StepMode::Dynamic
    {
        return false;
    }
    let device = device_of(m.backend);
    match &m.lm {
        AnyLm::Qwen3(q) => {
            q.static_supported(&device) && mummu::batch::fits(q, prompt_len, max_tokens)
        }
        AnyLm::Qwen35(q) => {
            q.static_supported(&device) && mummu::batch::fits(q, prompt_len, max_tokens)
        }
        _ => false,
    }
}

fn context_of(m: &Loaded) -> usize {
    match &m.lm {
        AnyLm::Qwen3(q) => mummu::batch::context(q),
        AnyLm::Qwen35(q) => mummu::batch::context(q),
        _ => 0,
    }
}

/// Hand the slot `guard` holds (model `key`, named `model`) to the thread,
/// with this request's generation. Returns what the request needs of the
/// session; its tokens come through `job.updates`.
pub(super) fn start(
    guard: SlotGuard<'static, Loaded>,
    key: &Path,
    model: &str,
    job: Job,
) -> SessionInfo {
    let info = SessionInfo {
        key: key.to_path_buf(),
        tokenizer: Arc::clone(&guard.tokenizer),
        devices: guard.devices.clone(),
        backend: guard.backend,
        context: context_of(&guard),
    };
    // Published before the thread sees the job: a request that arrives in
    // between joins rather than queueing behind a slot about to be busy.
    {
        let mut r = registry();
        r.info = Some(info.clone());
        r.accepting = true;
    }
    let _ = sender().send(Msg::Start {
        guard,
        key: key.to_path_buf(),
        model: model.to_owned(),
        job,
    });
    starts().send_modify(|n| *n += 1);
    info
}

/// Sessions started so far, for requests queued on the slot (see
/// [`subscribe`]).
fn starts() -> &'static tokio::sync::watch::Sender<u64> {
    static STARTS: OnceLock<tokio::sync::watch::Sender<u64>> = OnceLock::new();
    STARTS.get_or_init(|| tokio::sync::watch::channel(0).0)
}

/// A request queued on the slot watches this: a session that starts on its
/// model while it waits is one it can join instead of waiting for that
/// session to end.
pub(super) fn subscribe() -> tokio::sync::watch::Receiver<u64> {
    starts().subscribe()
}

/// The running session on model `key`, if it is admitting.
pub(super) fn session(key: &Path) -> Option<SessionInfo> {
    let r = registry();
    r.info
        .as_ref()
        .filter(|i| r.accepting && i.key == key)
        .cloned()
}

/// Join the running session on model `key` with `job`. `Err(job)` when it
/// stopped admitting in the meantime — the request then queues for the slot.
pub(super) fn join(key: &Path, job: Job) -> Result<(), Job> {
    let r = registry();
    if !(r.accepting && r.info.as_ref().is_some_and(|i| i.key == key)) {
        return Err(job);
    }
    // Sent under the registry lock: the thread decides a session is over
    // under the same lock, after draining its channel, so a join it has
    // accepted is never stranded.
    let sent = sender().send(Msg::Join(job));
    drop(r);
    sent.map_err(|mpsc::SendError(m)| match m {
        Msg::Join(job) => job,
        _ => unreachable!("sent a Join"),
    })
}

/// Stop the running session admitting, so the slot comes free when its live
/// sequences finish.
pub(super) fn drain() {
    registry().accepting = false;
}

/// Drop the cached batcher (graphs, state) now — or, while a session runs,
/// as soon as it ends — and stop that session admitting.
pub(super) fn release() {
    drain();
    let _ = sender().send(Msg::Release(None));
}

/// [`release`], waiting (up to `budget`) for the thread to have dropped it:
/// before a load, whose bytes must not land beside the last model's batch.
/// Never call it from the decode thread itself.
pub(super) fn release_wait(budget: Duration) {
    drain();
    let (tx, rx) = mpsc::channel();
    if sender().send(Msg::Release(Some(tx))).is_ok() {
        let _ = rx.recv_timeout(budget);
    }
}

/// The batcher for one model family.
enum AnyBatcher {
    Qwen3(Batcher<qwen3::LoadedQwen3>),
    Qwen35(Batcher<qwen35::LoadedQwen35>),
}

/// The batcher kept between sessions, and what it was made for.
struct Cached {
    key: PathBuf,
    load_id: u64,
    batcher: AnyBatcher,
    idle_since: Option<Instant>,
}

struct Held {
    guard: SlotGuard<'static, Loaded>,
    key: PathBuf,
    model: String,
    started: Instant,
    jobs: HashMap<SeqId, tokio::sync::mpsc::UnboundedSender<Update>>,
    /// Accepted while the batch was full: admitted as slots free up.
    waiting: VecDeque<Job>,
}

impl Held {
    fn busy(&self) -> bool {
        !self.jobs.is_empty() || !self.waiting.is_empty()
    }
}

#[derive(Default)]
struct Thread {
    cached: Option<Cached>,
    held: Option<Held>,
    release_after: bool,
}

impl Thread {
    fn run(mut self, rx: &mpsc::Receiver<Msg>) {
        loop {
            if self.held.is_none() {
                // Idle: wait for work, dropping a stale cache on the way.
                let wait = self
                    .cached
                    .as_ref()
                    .and_then(|c| c.idle_since)
                    .map_or(KEEP_IDLE, |t| KEEP_IDLE.saturating_sub(t.elapsed()));
                match rx.recv_timeout(wait.max(Duration::from_millis(10))) {
                    Ok(msg) => self.handle(msg),
                    Err(RecvTimeoutError::Timeout) => {
                        if self
                            .cached
                            .as_ref()
                            .and_then(|c| c.idle_since)
                            .is_some_and(|t| t.elapsed() >= KEEP_IDLE)
                        {
                            self.drop_cache("idle");
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                }
                continue;
            }
            while let Ok(msg) = rx.try_recv() {
                self.handle(msg);
            }
            if self
                .held
                .as_ref()
                .is_some_and(|h| h.started.elapsed() >= SESSION_SPAN)
            {
                drain();
            }
            self.step();
            self.maybe_end(rx);
        }
    }

    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Start {
                guard,
                key,
                model,
                job,
            } => {
                self.begin(guard, key, model);
                self.admit(job);
            }
            Msg::Join(job) => {
                if self.held.is_some() {
                    self.admit(job);
                } else {
                    // Unreachable by the registry protocol; said rather
                    // than dropped.
                    let _ = job.updates.send(Update::Done(Err(ChatError::request(
                        "the decode session ended before this request joined it",
                    ))));
                }
            }
            Msg::Release(ack) => {
                if self.held.is_some() {
                    self.release_after = true;
                } else {
                    self.drop_cache("released");
                }
                if let Some(ack) = ack {
                    let _ = ack.send(());
                }
            }
        }
    }

    /// Take the slot, and the batcher for its model.
    fn begin(&mut self, guard: SlotGuard<'static, Loaded>, key: PathBuf, model: String) {
        let load_id = guard.load_id;
        let fits = self
            .cached
            .as_ref()
            .is_some_and(|c| c.key == key && c.load_id == load_id);
        if !fits {
            self.drop_cache("another model");
            let device = device_of(guard.backend);
            let mode = mummu::capture::step_mode();
            let batcher = match &guard.lm {
                AnyLm::Qwen3(q) => {
                    Batcher::new(q, &device, max_slots(), mode).map(AnyBatcher::Qwen3)
                }
                AnyLm::Qwen35(q) => {
                    Batcher::new(q, &device, max_slots(), mode).map(AnyBatcher::Qwen35)
                }
                _ => None,
            };
            self.cached = batcher.map(|batcher| Cached {
                key: key.clone(),
                load_id,
                batcher,
                idle_since: None,
            });
        }
        if let Some(c) = &mut self.cached {
            c.idle_since = None;
            let epoch = super::placement::epoch();
            match &mut c.batcher {
                AnyBatcher::Qwen3(b) => b.set_epoch(epoch),
                AnyBatcher::Qwen35(b) => b.set_epoch(epoch),
            }
        }
        self.held = Some(Held {
            guard,
            key,
            model,
            started: Instant::now(),
            jobs: HashMap::new(),
            waiting: VecDeque::new(),
        });
    }

    fn admit(&mut self, job: Job) {
        let (Some(held), Some(cached)) = (&mut self.held, &mut self.cached) else {
            let _ = job.updates.send(Update::Done(Err(ChatError::request(
                "this model cannot take the batched decode path",
            ))));
            return;
        };
        let room = match &cached.batcher {
            AnyBatcher::Qwen3(b) => b.has_room(),
            AnyBatcher::Qwen35(b) => b.has_room(),
        };
        if !room {
            held.waiting.push_back(job);
            return;
        }
        let Job {
            prompt_ids,
            max_tokens,
            opts,
            constraint,
            updates,
        } = job;
        let adm = Admission {
            prompt_ids: &prompt_ids,
            max_tokens,
            opts: &opts,
            constraint,
        };
        let model = &held.guard.lm;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match (&mut cached.batcher, model) {
                (AnyBatcher::Qwen3(b), AnyLm::Qwen3(m)) => b.admit(m, adm),
                (AnyBatcher::Qwen35(b), AnyLm::Qwen35(m)) => b.admit(m, adm),
                _ => Err("the cached batcher is for another model".to_owned()),
            }
        }));
        match outcome {
            Ok(Ok((id, events))) => {
                let finished = events.iter().any(|e| matches!(e, Event::Done(_)));
                // A request gone after one event needs no more of them.
                let open = events.into_iter().all(|e| deliver(&updates, e));
                if !finished {
                    if open {
                        held.jobs.insert(id, updates);
                    } else {
                        self.cancel(id);
                    }
                }
            }
            Ok(Err(message)) => self.fail(&[updates], message, None),
            Err(payload) => self.fail(&[updates], recovery::payload_text(&*payload), Some(payload)),
        }
    }

    /// Admit what waited for a slot, while there is room.
    fn admit_waiting(&mut self) {
        loop {
            let room = self.cached.as_ref().is_some_and(|c| match &c.batcher {
                AnyBatcher::Qwen3(b) => b.has_room(),
                AnyBatcher::Qwen35(b) => b.has_room(),
            });
            let Some(job) = self
                .held
                .as_mut()
                .filter(|_| room)
                .and_then(|h| h.waiting.pop_front())
            else {
                return;
            };
            self.admit(job);
        }
    }

    fn cancel(&mut self, id: SeqId) {
        if let Some(c) = &mut self.cached {
            match &mut c.batcher {
                AnyBatcher::Qwen3(b) => b.cancel(id),
                AnyBatcher::Qwen35(b) => b.cancel(id),
            }
        }
    }

    /// One step for the whole batch, every token delivered.
    fn step(&mut self) {
        let (Some(held), Some(cached)) = (&mut self.held, &mut self.cached) else {
            return;
        };
        if held.jobs.is_empty() {
            self.admit_waiting();
            return;
        }
        let model = &held.guard.lm;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match (&mut cached.batcher, model) {
                (AnyBatcher::Qwen3(b), AnyLm::Qwen3(m)) => b.step(m),
                (AnyBatcher::Qwen35(b), AnyLm::Qwen35(m)) => b.step(m),
                _ => Err("the cached batcher is for another model".to_owned()),
            }
        }));
        let events = match outcome {
            Ok(Ok(events)) => events,
            Ok(Err(message)) => {
                let all: Vec<_> = held.jobs.drain().map(|(_, tx)| tx).collect();
                self.fail(&all, message, None);
                return;
            }
            Err(payload) => {
                let all: Vec<_> = held.jobs.drain().map(|(_, tx)| tx).collect();
                self.fail(&all, recovery::payload_text(&*payload), Some(payload));
                return;
            }
        };
        let mut gone = Vec::new();
        for (id, e) in events {
            let Some(tx) = held.jobs.get(&id) else {
                continue;
            };
            let open = deliver(tx, e);
            if matches!(e, Event::Done(_)) {
                held.jobs.remove(&id);
            } else if !open {
                // The request went away (a closed tab, a cancel).
                held.jobs.remove(&id);
                gone.push(id);
            }
        }
        for id in gone {
            self.cancel(id);
        }
        self.admit_waiting();
    }

    /// End the session when its batch is empty and nothing is waiting to
    /// join — decided under the registry lock, after draining the channel.
    fn maybe_end(&mut self, rx: &mpsc::Receiver<Msg>) {
        if self.held.as_ref().is_none_or(Held::busy) {
            return;
        }
        let mut r = registry();
        while let Ok(msg) = rx.try_recv() {
            self.handle(msg);
        }
        if self.held.as_ref().is_some_and(Held::busy) {
            return;
        }
        r.info = None;
        r.accepting = false;
        drop(r);
        // The slot goes back with the batcher kept for the next session.
        self.held = None;
        if let Some(c) = &mut self.cached {
            c.idle_since = Some(Instant::now());
        }
        if std::mem::take(&mut self.release_after) {
            self.drop_cache("released");
        }
    }

    /// A failure on the thread: told to every sequence in `to`; a device
    /// failure decided like `serve_held` decides one — recorded, and the
    /// model evicted through the guard — and the batch dropped.
    fn fail(
        &mut self,
        to: &[tokio::sync::mpsc::UnboundedSender<Update>],
        message: String,
        payload: Option<Box<dyn std::any::Any + Send>>,
    ) {
        let device = recovery::is_device_failure(&message);
        if !device {
            if let Some(payload) = payload {
                // An ordinary bug: the sequences are told, the slot is
                // released with the model still in it, and the panic is
                // logged rather than taking the thread down.
                eprintln!(
                    "[mummu-serve] decode thread: a generation panicked: {}",
                    recovery::payload_text(&*payload)
                );
            }
            let error = ChatError::request(message);
            for tx in to {
                let _ = tx.send(Update::Done(Err(error.clone())));
            }
            // The batch's state is unknown after a panic mid-step.
            if let Some(held) = &mut self.held {
                for (_, tx) in held.jobs.drain() {
                    let _ = tx.send(Update::Done(Err(error.clone())));
                }
            }
            self.drop_cache("a failed step");
            // What waited is admitted into a fresh batch, if one can be made.
            if let Some(held) = &self.held {
                let device = device_of(held.guard.backend);
                let mode = mummu::capture::step_mode();
                let batcher = match &held.guard.lm {
                    AnyLm::Qwen3(q) => {
                        Batcher::new(q, &device, max_slots(), mode).map(AnyBatcher::Qwen3)
                    }
                    AnyLm::Qwen35(q) => {
                        Batcher::new(q, &device, max_slots(), mode).map(AnyBatcher::Qwen35)
                    }
                    _ => None,
                };
                self.cached = batcher.map(|batcher| Cached {
                    key: held.key.clone(),
                    load_id: held.guard.load_id,
                    batcher,
                    idle_since: None,
                });
            }
            self.admit_waiting();
            return;
        }
        let Some(held) = self.held.take() else { return };
        {
            let mut r = registry();
            r.info = None;
            r.accepting = false;
        }
        let decided = recovery::record_failure(
            &held.model,
            &held.guard.devices,
            &recovery::summarize(&message),
        );
        super::placement::note_device_failure(&message);
        for tx in to
            .iter()
            .chain(held.jobs.values())
            .chain(held.waiting.iter().map(|j| &j.updates))
        {
            let _ = tx.send(Update::Done(Err(decided.clone())));
        }
        self.cached = None;
        evict_held(held.guard, &held.key, &held.model);
    }

    fn drop_cache(&mut self, why: &str) {
        if let Some(mut c) = self.cached.take() {
            match &mut c.batcher {
                AnyBatcher::Qwen3(b) => b.release(),
                AnyBatcher::Qwen35(b) => b.release(),
            }
            eprintln!("[mummu-serve] decode thread: dropped the cached graphs and state ({why})");
        }
    }
}

/// Send one batch event to its request; `false` when the request is gone.
fn deliver(tx: &tokio::sync::mpsc::UnboundedSender<Update>, e: Event) -> bool {
    let update = match e {
        Event::Token(t) => Update::Token(t),
        Event::Done(_) => Update::Done(Ok(())),
    };
    tx.send(update).is_ok()
}
