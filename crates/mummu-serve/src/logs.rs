//! One merged, in-memory view of everything this process knows it did.
//!
//! The problem this solves is not "mummu-serve has no logging" — it prints
//! plenty. The problem is that all of it goes to stderr, where only
//! `docker logs` can see it. A cold 27B load spends **two minutes** reading a
//! pack off a spinning array while printing exactly the lines the operator
//! needs (`[mummu] load: 673/851 tensors — 14.46 GiB off the pack in 91s
//! (162 MB/s)`), and the browser shows an empty assistant bubble the whole
//! time. Worse, a CUDA failure has shown up *only* as a panic on stderr while
//! the request hung and `/api/health` still answered `ok`.
//!
//! So this module keeps a bounded log of lines that the server can serve back
//! over HTTP, fed from two places:
//!
//! - **Captured output** ([`Source::Server`]): on unix, fds 1 and 2 are
//!   re-pointed at a pipe whose reader tees every byte back to the original
//!   fd *and* splits it into ring entries. Existing `eprintln!`s need no
//!   changes, panics land here for free (the runtime prints them to fd 2), and
//!   `docker logs` still shows byte-identical output.
//! - **Request entries** ([`Source::Api`] / [`Source::Shim`]): an axum
//!   middleware on both routers records one line when a request arrives (if it
//!   can start real work) and one when its response head goes out.
//!
//! Both feed **one sequence space**, so `GET /api/logs` hands back a single
//! chronological stream that a page can filter — a request entry and the
//! load-progress line it triggered sit next to each other in the order they
//! actually happened, which is the whole point.
//!
//! That one sequence is stored in **two bounded rings**: [`MAX_LINES`] for
//! captured output and every request that did something, and a much smaller
//! [`QUIET_LINES`] for the [quiet](LogLine::quiet) ones — the health probes
//! and monitor polls that arrive on a timer whether or not anything is
//! happening. v0.3.0 kept them in one ring and merely flagged the polls, and
//! the flag hid them without stopping them from taking slots: six hours after
//! that deploy the live ring held four hours, 97% of it polling, and the cold
//! load the page exists to show had been evicted. A read merges the two rings
//! back into seq order, so a client sees one stream exactly as before.
//!
//! What this is *not*: a logging framework. Captured output carries no level,
//! because the call sites are free-text `eprintln!`s that predate this module
//! by a year and are not worth rewriting, so [`classify`] guesses one from the
//! text — a heuristic, documented as one, and wrong sometimes (a line that
//! merely mentions the word "error" reads as an error). It is there to colour
//! a feed, not to drive behaviour. Request entries are the exception: their
//! status code *states* the level, so they skip the guess entirely.

use std::collections::VecDeque;
use std::sync::{Mutex, Once};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::Request;
use axum::http::{Method, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::json_response;

/// Lines the MAIN ring holds: captured server output, and every request that
/// is not [quiet](LogLine::quiet).
///
/// Sized against the thing this exists to show: a cold 27B load prints a few
/// hundred lines (per-tier fit plans, the per-tensor load ticker, VNNI packing,
/// residency), so 2000 holds a whole cold start *plus* the conversation that
/// followed it with room to spare, and a client that misses lines is told
/// exactly how many (see [`Page::dropped`]) rather than being left to wonder.
///
/// What it does NOT hold is polling, and that is what makes the size mean
/// anything. v0.3.0's single ring was measured on the live server six hours
/// after its deploy: it held exactly four hours, and 97% of it was automated
/// polls (720 health probes, ~240 each of three monitors) — the cold load
/// had already been evicted by traffic the page was hiding. Quiet requests
/// now live in their own [`QUIET_LINES`] ring, so nothing that arrives on a
/// timer can take a slot from this one, however often it arrives.
pub const MAX_LINES: usize = 2000;

/// Lines the QUIET ring holds: the successful page and listing reads that
/// `quiet_request` names — Docker's health probe, dashboard monitors,
/// scheduled flows and ollama clients, all polling on a timer.
///
/// The question this ring answers is "are the probes still arriving, and when
/// did each last run?", which needs a recent window rather than history. In
/// production the four pollers measured in v0.3.0's ring (Docker's probe, an
/// ollama client on the shim, a Glance monitor and a Kestra flow) put 1,450
/// lines there in four hours, about six a minute, so 500 lines is over an
/// hour: each of them shows up dozens of times, and anything on a timer of up
/// to an hour shows up at least once. It is small on purpose, too. Every
/// quiet line is a request line for a path on a fixed list — method, path,
/// status, duration, size, well under 100 bytes — so the whole ring is a few
/// tens of KiB, and a flood of probes cycles it in seconds without touching
/// the main ring at all.
pub const QUIET_LINES: usize = 500;

/// How many main-ring evictions the ring remembers the seq of, so that
/// [`Page::dropped`] counts main-ring losses and nothing else.
///
/// With two rings evicting independently the seq column is no longer
/// contiguous, so "the gap since your cursor" stopped being `oldest - 1 -
/// since`: that span now also holds quiet lines that aged out of their own
/// smaller ring, and counting them would have the page draw a gap marker in
/// front of lines nobody could have seen (quiet lines are hidden by default).
/// Which seqs in the span were main-ring lines is information that left with
/// the lines themselves, so the ring keeps just their seqs — 8 bytes each.
///
/// One ring's worth covers every reader that exists: a polling page is a
/// second behind, a fresh one asks from 0 (exact whatever this holds), and a
/// paused one stays exact until it has fallen two whole rings of server
/// output behind. Past that, `dropped` is a floor, and [`Page::dropped_exact`]
/// says so rather than letting the page print it as a count.
const EVICTION_MEMORY: usize = MAX_LINES;

/// Longest single line kept, in bytes. A runaway print (a panic payload
/// carrying a whole tensor manifest, a client-supplied name, a binary blob
/// that reached stderr) must not be able to grow the ring without bound:
/// 2000 x 2 KiB caps the worst case at ~4 MiB, while the real lines here run
/// well under 200 bytes. Nothing is actually lost — the tee wrote the full
/// bytes through to the original fd before truncating for the ring.
pub const MAX_LINE_BYTES: usize = 2048;

/// Marker appended to a line cut at [`MAX_LINE_BYTES`].
const TRUNCATED: &str = " …[truncated]";

/// Largest page `GET /api/logs` will return, whatever `limit` asks for.
///
/// Equal to the MAIN ring, so one request can always hand a fresh client
/// every line that ring holds. A full read of both rings can be up to
/// [`QUIET_LINES`] longer than that and then takes one more request, which
/// [`Page::more`] announces and `/logs` follows immediately — the same stop
/// the byte budget already produces, and unchanged for a client that ignores
/// `more` and simply polls again on its timer.
pub const MAX_LIMIT: usize = MAX_LINES;
/// `limit` when the query does not say.
pub const DEFAULT_LIMIT: usize = 500;

/// Largest payload ONE response will build, whatever `limit` asks for.
///
/// `/logs` and `/api/logs` are public on `mummu.basicautomation.io`, by
/// decision — no token, no IP gate — so the cost of a single anonymous
/// request is a property the endpoint has to own rather than delegate to
/// whoever is asking. `limit` alone bounds it at
/// [`MAX_LIMIT`] x [`MAX_LINE_BYTES`], about 4 MiB, which is a lot of work
/// and a lot of bandwidth to hand a caller who asks in a loop.
///
/// The value is set at the largest thing a REAL reader can ask for, so it
/// costs them nothing at all: the lines this ring holds run well under 200
/// bytes, so `/logs`'s own `limit=2000` bootstrap of a completely full ring
/// is ~330 KiB with the envelope charged, and it arrives in one response
/// exactly as it always did. What it takes away is the pathological case —
/// 2000 lines each at the [`MAX_LINE_BYTES`] truncation limit — which is the
/// only way to reach 4 MiB and is not a thing the server ever produces.
///
/// And meeting the bound is not an error and loses nothing: it is the same
/// stop `limit` already produces. The page ends, `cursor` points at the last
/// line handed over, [`Page::more`] says there is more, and the next poll
/// continues from there. Semantics unchanged, worst case eight times smaller.
pub const MAX_PAGE_BYTES: usize = 512 * 1024;

/// Charged per line on top of its text, for the JSON that wraps it (`seq`,
/// `ts`, `level`, `source`, `quiet`, the key names, the quoting).
///
/// Measured against the real shape — `{"seq":1234,"ts":1758150000000,
/// "level":"info","source":"server","quiet":false,"text":""}` is ~85 bytes —
/// and rounded up, because [`MAX_PAGE_BYTES`] is a bound on the work one
/// request may cause, not a content-length promise. Charging it at all is
/// what keeps a page of 2000 empty lines from being free.
const LINE_ENVELOPE_BYTES: usize = 96;

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

/// Where a line came from. One ring, three tags — the merge is the feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The process's own stdout/stderr, captured verbatim.
    Server,
    /// A request on the native API listener (8095 in production).
    Api,
    /// A request on the ollama-compatible shim listener (11435).
    Shim,
}

impl Source {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Server => "server",
            Self::Api => "api",
            Self::Shim => "shim",
        }
    }

    /// Parse the `source` query value. `all` (and an absent value) mean "no
    /// filter", so it maps to `None` *inside* the `Ok`; an unrecognised value
    /// is an `Err` rather than a silent "all", because a client that typoed
    /// `sever` should find out instead of quietly reading everything.
    ///
    /// # Errors
    /// If the value is not `all`, `server`, `api` or `shim`.
    pub fn parse_filter(value: &str) -> Result<Option<Self>, String> {
        match value {
            "all" | "" => Ok(None),
            "server" => Ok(Some(Self::Server)),
            "api" => Ok(Some(Self::Api)),
            "shim" => Ok(Some(Self::Shim)),
            other => Err(format!(
                "unknown source {other:?} — expected all, server, api or shim"
            )),
        }
    }
}

/// Severity, guessed from the text. See [`classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

impl Level {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }

    /// The inverse of [`Self::as_str`], for lines read back from a previous
    /// process's evidence file (see `recovery`). `None` for anything else.
    #[must_use]
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "info" => Some(Self::Info),
            "warn" => Some(Self::Warn),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

/// Guess a level from free text.
///
/// This is a **heuristic over prints that were never written for it**, not a
/// logging framework's level. mummu's output is `eprintln!` strings from a
/// year of debugging; rewriting them all to carry a level would be a far
/// bigger change than the one the operator asked for, and would still not
/// cover panics, which the runtime formats itself.
///
/// The rules, in order: a line mentioning `panicked`, `error` or `failed` is
/// an error; one mentioning `warn`, `unverified`, or the literal uppercase
/// `OVER` (the fit planner's `(OVER — will spill)` budget flag) is a warning;
/// everything else is info. It is deliberately case-insensitive except for
/// `OVER`, which is only meaningful as that exact token.
#[must_use]
pub fn classify(text: &str) -> Level {
    let lower = text.to_ascii_lowercase();
    if lower.contains("panicked") || lower.contains("error") || lower.contains("failed") {
        Level::Error
    } else if lower.contains("warn") || lower.contains("unverified") || text.contains("OVER") {
        Level::Warn
    } else {
        Level::Info
    }
}

/// One line in the ring.
#[derive(Debug, Clone)]
pub struct LogLine {
    /// Position in the single, process-wide sequence. Starts at 1, so a
    /// `since` of 0 means "everything you still have".
    pub seq: u64,
    /// Unix milliseconds when the line was recorded.
    pub unix_ms: u64,
    pub level: Level,
    pub source: Source,
    /// Routine chatter (health checks, monitor and client polling) that a
    /// reader normally wants hidden. Kept rather than dropped, because "is
    /// Docker's health check still passing?" is a real question — but kept in
    /// the smaller quiet ring ([`QUIET_LINES`]), never in the main one.
    pub quiet: bool,
    pub text: String,
}

impl LogLine {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "seq": self.seq,
            "ts": self.unix_ms,
            "level": self.level.as_str(),
            "source": self.source.as_str(),
            "quiet": self.quiet,
            "text": self.text,
        })
    }
}

// ---------------------------------------------------------------------------
// The ring
// ---------------------------------------------------------------------------

/// Both rings and the one sequence they share, behind one lock.
///
/// Each ring is ordered by seq on its own (a line takes the next seq and is
/// pushed in the same critical section), and every seq ever assigned lives
/// in exactly one of them until it is evicted — which is what lets a read
/// merge them back into the single stream a client has always been handed.
struct Ring {
    /// The MAIN ring: server output and loud requests, [`MAX_LINES`].
    lines: VecDeque<LogLine>,
    /// The QUIET ring: routine polling, [`QUIET_LINES`].
    quiet: VecDeque<LogLine>,
    /// The seq the next recorded line will take. Never reused, never reset.
    next_seq: u64,
    /// Main-ring evictions over the life of the process.
    dropped_total: u64,
    /// Quiet-ring evictions over the life of the process.
    dropped_quiet_total: u64,
    /// The seqs of the most recent main-ring evictions, oldest first, at most
    /// [`EVICTION_MEMORY`] of them — see there for why they are worth 8 bytes
    /// each.
    evicted: VecDeque<u64>,
}

impl Ring {
    const fn new() -> Self {
        Self {
            lines: VecDeque::new(),
            quiet: VecDeque::new(),
            next_seq: 1,
            dropped_total: 0,
            dropped_quiet_total: 0,
            evicted: VecDeque::new(),
        }
    }

    fn record(&mut self, mut line: LogLine) {
        line.seq = self.next_seq;
        self.next_seq += 1;
        if line.quiet {
            self.quiet.push_back(line);
            while self.quiet.len() > QUIET_LINES {
                self.quiet.pop_front();
                self.dropped_quiet_total += 1;
            }
            return;
        }
        self.lines.push_back(line);
        while self.lines.len() > MAX_LINES {
            let Some(gone) = self.lines.pop_front() else {
                break;
            };
            self.dropped_total += 1;
            self.evicted.push_back(gone.seq);
            if self.evicted.len() > EVICTION_MEMORY {
                self.evicted.pop_front();
            }
        }
    }

    /// The oldest seq either ring still holds, or `next_seq` when both are
    /// empty — so `oldest - 1` is still "just before everything I have".
    fn oldest(&self) -> u64 {
        let main = self.lines.front().map(|l| l.seq);
        let quiet = self.quiet.front().map(|l| l.seq);
        main.into_iter().chain(quiet).min().unwrap_or(self.next_seq)
    }

    /// Every held line with `seq > since`, from both rings, in seq order.
    ///
    /// A merge of two already-sorted runs, each entered by binary search: no
    /// sort, no allocation, and nothing cloned — the caller clones the lines
    /// it actually keeps.
    fn after(&self, since: u64) -> impl Iterator<Item = &LogLine> {
        let main_start = self.lines.partition_point(|l| l.seq <= since);
        let quiet_start = self.quiet.partition_point(|l| l.seq <= since);
        let mut main = self.lines.range(main_start..).peekable();
        let mut quiet = self.quiet.range(quiet_start..).peekable();
        std::iter::from_fn(move || match (main.peek(), quiet.peek()) {
            (Some(m), Some(q)) if q.seq < m.seq => quiet.next(),
            (Some(_), _) => main.next(),
            (None, _) => quiet.next(),
        })
    }

    /// How many MAIN-ring lines with `seq > since` have been evicted, and
    /// whether that number is exact.
    ///
    /// Quiet lines are not counted, whatever became of them: they age out of
    /// their own ring by design, they are hidden unless asked for, and a gap
    /// marker for them would be a marker for nothing the reader was shown.
    ///
    /// Exact whenever [`EVICTION_MEMORY`] reaches back to `since`, or the
    /// ring has never forgotten an eviction, or `since` is so early that the
    /// forgotten ones must all lie after it (a fresh client's `since = 0`).
    /// Otherwise the forgotten evictions are bounded rather than counted, and
    /// the floor is what comes back — never more than was really lost, and
    /// never zero when anything was.
    fn dropped_after(&self, since: u64) -> (u64, bool) {
        let (Some(&first), Some(&last)) = (self.evicted.front(), self.evicted.back()) else {
            return (0, true); // nothing has ever left the main ring
        };
        if since >= last {
            return (0, true); // everything that left, left before the cursor
        }
        let remembered = self.evicted.len() as u64;
        let after = remembered - self.evicted.partition_point(|&s| s <= since) as u64;
        let forgotten = self.dropped_total - remembered;
        if after < remembered || forgotten == 0 {
            // The memory reaches back past `since` (so every eviction after it
            // is in there), or it holds every eviction there has ever been.
            return (after, true);
        }
        // All `remembered` are after `since`; of the `forgotten` ones — all
        // older than `first` — at most `since` can sit at or before it (one
        // line per seq), and at most `first - 1 - since` after it.
        let floor = forgotten.saturating_sub(since);
        let ceiling = forgotten.min(first - 1 - since);
        (remembered + floor, floor == ceiling)
    }
}

static RING: Mutex<Ring> = Mutex::new(Ring::new());

/// A poisoned ring is still a perfectly good ring: the only thing a panicking
/// holder could have left behind is a half-pushed `VecDeque`, and losing the
/// log is exactly the wrong response to the panic we are here to show.
fn ring() -> std::sync::MutexGuard<'static, Ring> {
    RING.lock().unwrap_or_else(|e| e.into_inner())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Record a line under `source`, classifying its level from the text.
pub fn push(source: Source, text: impl Into<String>) {
    record(source, None, false, text.into());
}

/// Record a line marked [quiet](LogLine::quiet) — routine traffic a reader
/// normally wants folded away.
pub fn push_quiet(source: Source, text: impl Into<String>) {
    record(source, None, true, text.into());
}

/// Build the entry, outside the lock. `seq` is filled in by [`Ring::record`],
/// under the lock, so sequence order is insertion order.
///
/// `level` of `None` means "guess it from the text" — right for captured
/// output, where there is nothing else to go on, and wrong for a request
/// entry, whose status code already *states* what happened.
fn line(source: Source, level: Option<Level>, quiet: bool, text: String) -> LogLine {
    let text = truncate(text);
    let level = level.unwrap_or_else(|| classify(&text));
    LogLine {
        seq: 0,
        unix_ms: now_ms(),
        level,
        source,
        quiet,
        text,
    }
}

fn record(source: Source, level: Option<Level>, quiet: bool, text: String) {
    let entry = line(source, level, quiet, text);
    ring().record(entry);
}

/// Record a line that an EARLIER process printed, keeping its own time and
/// level. It takes a seq in this process's sequence like any other line —
/// which is what puts it in front of everything this process prints, where a
/// reader expects the story of the restart to start.
///
/// Only `recovery` calls this, replaying the tail the previous process wrote
/// before it exited to restart the GPU backend.
pub fn push_replayed(source: Source, unix_ms: u64, level: Level, text: impl Into<String>) {
    let mut entry = line(source, Some(level), false, text.into());
    entry.unix_ms = unix_ms;
    ring().record(entry);
}

/// The newest `n` lines of the MAIN ring, oldest first.
///
/// What a process about to exit leaves for the next one (see `recovery`): the
/// lines that explain the exit are by definition the last ones printed, and
/// the quiet ring holds nothing but health probes, so it is not consulted.
#[must_use]
pub fn tail(n: usize) -> Vec<LogLine> {
    let ring = ring();
    let skip = ring.lines.len().saturating_sub(n);
    ring.lines.iter().skip(skip).cloned().collect()
}

/// Cut an over-long line at a char boundary and mark it. Kept separate from
/// the reassembler's own limit because explicit pushes never go through it.
fn truncate(mut text: String) -> String {
    if text.len() <= MAX_LINE_BYTES {
        return text;
    }
    let mut end = MAX_LINE_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str(TRUNCATED);
    text
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// A resolved `GET /api/logs` query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Query {
    /// Return lines with `seq` **strictly greater** than this.
    pub since: u64,
    /// At most this many lines, already clamped to `1..=MAX_LIMIT`.
    pub limit: usize,
    /// `None` = every source.
    pub source: Option<Source>,
}

/// One answer to a [`Query`].
///
/// The contract against v0.3.0, which had one ring, field by field:
///
/// - `lines`, `cursor`, `newest`, `more` — unchanged. The lines come from
///   both rings merged into seq order, the cursor is still a seq in the one
///   seq space, and a client polling `since=<cursor>` sees every line once.
/// - `dropped`, `dropped_total`, `capacity` — describe the MAIN ring, which
///   is what v0.3.0 called "the ring" minus the polling that no longer lives
///   in it. A v0.3.0 client draws a marker for the same losses it always
///   did, and no longer draws one for a poll ageing out of its own ring.
/// - `oldest` — still the oldest seq held anywhere, so `oldest - 1` still
///   means "everything"; it is no longer `dropped + 1` for a fresh client.
/// - ADDED: `dropped_exact`, `dropped_quiet_total`, `quiet_capacity`.
///
/// What a v0.3.0 client must not do is infer a gap from a seq jump: a merged
/// read is no longer contiguous. Nothing shipped ever did — both embedded
/// pages trust `dropped`, and `ui.html` reads only the `status` object.
pub struct Page {
    /// Held lines with `seq` after the query's `since`, from BOTH rings,
    /// merged into seq order — quiet ones carry `quiet: true`, as before.
    /// Consecutive lines need not have consecutive seqs: a quiet line that
    /// aged out of its ring leaves a hole that is not a gap.
    pub lines: Vec<LogLine>,
    /// What the client should send as `since` next time.
    ///
    /// It advances past lines that the `source` filter excluded, so a filtered
    /// client does not rescan the same span forever. It can also come back
    /// **lower** than the `since` that was sent: that means this process has
    /// never produced a seq that high (the server restarted under a client
    /// that kept its cursor), and the client has been re-synced — see [`read`]
    /// for where to.
    pub cursor: u64,
    /// MAIN-ring lines after the client's `since` that had already been
    /// evicted — the size of the gap this page starts with. A client shows a
    /// marker instead of silently missing them.
    ///
    /// Quiet lines that aged out of their own ring are never counted here
    /// (see [`Page::dropped_quiet_total`]): they are hidden by default, so a
    /// marker for them would sit in front of nothing the reader was missing.
    pub dropped: u64,
    /// Whether `dropped` is a count or a floor. It is a count for every
    /// realistic reader; it becomes a floor only for a cursor that has fallen
    /// more than two whole main rings behind (see [`EVICTION_MEMORY`]), and
    /// then the true number is at least `dropped`, never less.
    pub dropped_exact: bool,
    /// Main-ring lines evicted over the whole life of the process.
    pub dropped_total: u64,
    /// Quiet-ring lines aged out over the whole life of the process — the
    /// polling that was recorded and has since been let go, by design.
    pub dropped_quiet_total: u64,
    /// The oldest seq held in EITHER ring (the next seq to be assigned when
    /// both are empty), so `since = oldest - 1` still means "everything you
    /// have". Unlike v0.3.0 it is no longer `dropped + 1` for a fresh client:
    /// the seqs below it include quiet lines, which `dropped` does not count.
    pub oldest: u64,
    /// The highest seq ever assigned, in either ring (0 before the first
    /// line). This is the one-seq-space high-water mark a cursor is compared
    /// against, so it has to cover both: a cursor that rests on a quiet line
    /// must not look like a cursor from the future.
    pub newest: u64,
    /// The rings hold lines past this page's `cursor` — the page stopped on
    /// `limit` or on [`MAX_PAGE_BYTES`], not because it ran out.
    ///
    /// Always derivable from `cursor < newest`, and sent anyway: a reader
    /// that has to know whether to poll again should be told, not made to
    /// re-derive it, and a client written against an earlier build that
    /// ignores the field still behaves exactly as it did (it polls again on
    /// its own timer and the cursor picks up where it left off).
    pub more: bool,
}

impl Page {
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "lines": self.lines.iter().map(LogLine::to_json).collect::<Vec<_>>(),
            "cursor": self.cursor,
            "dropped": self.dropped,
            "dropped_exact": self.dropped_exact,
            "dropped_total": self.dropped_total,
            "dropped_quiet_total": self.dropped_quiet_total,
            "oldest": self.oldest,
            "newest": self.newest,
            // The MAIN ring's, as in v0.3.0 — the ring `dropped` is about.
            "capacity": MAX_LINES,
            "quiet_capacity": QUIET_LINES,
            "more": self.more,
        })
    }
}

/// Read a page out of the rings.
///
/// Deliberately cheap enough to poll once a second forever: one mutex
/// acquisition, a binary search into each of the two seq-ordered rings, and
/// one merge walk over them that clones at most `limit` lines — and never
/// more than [`MAX_PAGE_BYTES`] of them, which is what bounds the cost of one
/// anonymous request on a public endpoint. Nothing here touches the disk,
/// the device, or the model slot, so a client hammering `/api/logs` cannot
/// stall a generation.
#[must_use]
pub fn query(q: &Query) -> Page {
    read(&ring(), q)
}

/// The cursor arithmetic, over a ring passed in rather than the global one —
/// so its edges (an empty ring, an overrun one, a cursor from a dead process)
/// are testable without a process-wide singleton the tests would race on.
fn read(ring: &Ring, q: &Query) -> Page {
    let newest = ring.next_seq.saturating_sub(1);
    let oldest = ring.oldest();
    // A cursor from a previous process (or a client that invented one) would
    // otherwise park forever waiting for seqs this process will never reach.
    //
    // Re-synced to just before the OLDEST line still held, NOT to `newest`.
    // The whole situation this handles is "the server restarted under a page
    // that kept its cursor" — and everything the new process has printed so
    // far is precisely what that page has not seen: the startup banner, the
    // adapter inventory, the device policy, the first fit plan. Landing on
    // `newest` skipped all of it and left the page blank until the next thing
    // happened to be printed, which for an idle server is half a minute
    // (Docker's health check) and for a hung one is never. Restarting the
    // server and watching is the operator's own workflow; it has to work.
    let resynced = q.since > newest;
    let since = if resynced {
        oldest.saturating_sub(1)
    } else {
        q.since
    };
    // In the re-synced case the client's own cursor came from a different
    // sequence space, so "the gap since your cursor" is not a question with an
    // answer. The gap that IS real is measured from the start of this process:
    // the lines it printed and evicted before this client ever asked, which it
    // will never see. Reporting them lets the page draw a break instead of
    // splicing a restart onto an overrun. Main-ring lines only, as everywhere.
    let (dropped, dropped_exact) = if resynced {
        (ring.dropped_total, true)
    } else {
        ring.dropped_after(since)
    };

    let mut lines = Vec::new();
    let mut cursor = since;
    let mut stopped_early = false;
    let mut bytes = 0usize;
    for line in ring.after(since) {
        if q.source.is_some_and(|want| want != line.source) {
            cursor = line.seq; // skipped, but the client has still seen past it
            continue;
        }
        if lines.len() >= q.limit {
            stopped_early = true;
            break;
        }
        // The byte budget stops the page exactly as `limit` does. The first
        // line is always taken, whatever it weighs: a line longer than the
        // whole budget would otherwise be skipped by every request forever
        // and park the cursor in front of it.
        let cost = line.text.len() + LINE_ENVELOPE_BYTES;
        if !lines.is_empty() && bytes + cost > MAX_PAGE_BYTES {
            stopped_early = true;
            break;
        }
        bytes += cost;
        cursor = line.seq;
        lines.push(line.clone());
    }
    if !stopped_early {
        cursor = cursor.max(newest);
    }
    Page {
        lines,
        cursor,
        dropped,
        dropped_exact,
        dropped_total: ring.dropped_total,
        dropped_quiet_total: ring.dropped_quiet_total,
        oldest,
        newest,
        more: cursor < newest,
    }
}

// ---------------------------------------------------------------------------
// GET /api/logs
// ---------------------------------------------------------------------------

/// The raw query string, taken as text so a malformed number is *clamped*
/// rather than answered with serde's 400 — a log reader that mistypes a
/// cursor should still see logs.
#[derive(serde::Deserialize, Default)]
pub struct LogsParams {
    since: Option<String>,
    limit: Option<String>,
    source: Option<String>,
}

impl LogsParams {
    /// Resolve to a [`Query`]: unparseable `since`/`limit` fall back to the
    /// defaults, `limit` is clamped to `1..=MAX_LIMIT`, and only a bad
    /// `source` is an error (see [`Source::parse_filter`]).
    ///
    /// # Errors
    /// If `source` names something that is not a known source.
    pub fn resolve(&self) -> Result<Query, String> {
        let since = self
            .since
            .as_deref()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0);
        let limit = self
            .limit
            .as_deref()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_LIMIT)
            .clamp(1, MAX_LIMIT);
        let source = match self.source.as_deref() {
            Some(s) => Source::parse_filter(s.trim())?,
            None => None,
        };
        Ok(Query {
            since,
            limit,
            source,
        })
    }
}

/// `GET /api/logs?since=<seq>&limit=<n>&source=<all|server|api|shim>`.
///
/// Carries a `status` object alongside the lines — the load's progress and
/// the two memory readings (see [`crate::status`]). It rides THIS poll rather
/// than getting an endpoint of its own because both pages already poll here
/// once a second; a second loop would double a browser tab's cost to show
/// numbers that are read in the same glance as the lines beside them.
pub async fn endpoint(params: axum::extract::Query<LogsParams>) -> Response {
    let q = match params.resolve() {
        Ok(q) => q,
        Err(e) => return json_response(400, json!({ "error": e })),
    };
    // On a blocking thread for the same reason `/api/health` is: the status
    // object reads `/proc` and calls into NVML, which are file and FFI work,
    // not async work, and an async worker parked on them is a worker not
    // driving the generation this page is watching.
    crate::blocking(move || {
        let mut body = query(&q).to_json();
        if let Some(object) = body.as_object_mut() {
            object.insert("status".to_owned(), crate::status::to_json());
        }
        json_response(200, body)
    })
    .await
}

/// The logs page, served at `GET /logs`.
pub const LOGS_HTML: &str = include_str!("logs.html");

/// `GET /logs` — the standalone merged-feed page.
pub async fn page() -> Response {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        LOGS_HTML,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Request logging
// ---------------------------------------------------------------------------

/// Largest request body we will look inside to name the model. A prompt can be
/// megabytes; peeking at one costs a copy we would otherwise not make, and the
/// model name is the only field worth having.
const MAX_MODEL_PEEK: usize = 256 * 1024;

/// Longest client-controlled fragment (a path, a model name) we will paste
/// into a log line. Anything can arrive on a public listener; the feed renders
/// as text, but an unbounded or control-character-laden field would still
/// wreck it.
const MAX_FIELD: usize = 96;

/// Record requests on the native API listener.
pub async fn record_api(req: Request, next: Next) -> Response {
    observe(Source::Api, req, next).await
}

/// Record requests on the ollama-compatibility listener.
pub async fn record_shim(req: Request, next: Next) -> Response {
    observe(Source::Shim, req, next).await
}

/// The feed's own poll. Recording it would make the ring mostly a record of
/// the page reading the ring — one line a second, evicting in half an hour the
/// cold-load progress the page exists to show.
///
/// Not recorded even as [quiet](LogLine::quiet), now that quiet lines have a
/// ring of their own: one open tab would cycle that ring in about eight
/// minutes, and push out the probe history it is there to keep.
fn ignored(source: Source, path: &str) -> bool {
    source == Source::Api && path == "/api/logs"
}

/// Is this request *someone else's* heartbeat — [quiet](LogLine::quiet)?
///
/// **The rule: a SUCCESSFUL, idempotent read of a page or a listing is
/// quiet. Everything else is loud.** Concretely, all three must hold:
///
/// - the method is `GET` or `HEAD` — any `POST` does work, and is loud;
/// - the path is one of the pages and listings things poll on a timer
///   ([`polled_path`]) — which leaves out the WebSocket upgrade, the one GET
///   here that starts a generation;
/// - the status is 2xx or 3xx.
///
/// The status clause is what keeps the public listeners legible. Scanners
/// probe the shim for `/.env`, `/v1/.env`, `/.git/config` and friends and
/// get a 404, and the owner should see that on the page without ticking a
/// box; so should a monitor whose listing starts answering 500. A poll is
/// only routine while it keeps succeeding.
///
/// Why the path list is what it is: measured on the live server, the traffic
/// that filled v0.3.0's ring was Docker's 30 s health probe and an ollama
/// client polling the shim's `/` (both already quiet), plus a Glance
/// dashboard monitor on `GET /` and a Kestra flow on `GET /api/models`, once
/// a minute each — reads of the same page and listing a person opens, which
/// is why the whole set is quiet rather than just those two.
fn quiet_request(source: Source, method: &Method, path: &str, status: u16) -> bool {
    let read = method == Method::GET || method == Method::HEAD;
    let succeeded = (200..400).contains(&status);
    read && succeeded && polled_path(source, path)
}

/// The pages and listings — the only paths [`quiet_request`] can call quiet.
///
/// On the API listener: the chat page (`/` and its `/index.html` alias),
/// the logs page, the model listing, the health probe and the favicon a
/// browser or a dashboard asks for beside a page. On the shim: the four
/// reads an ollama client polls to fill a model picker.
fn polled_path(source: Source, path: &str) -> bool {
    match source {
        Source::Api => matches!(
            path,
            "/" | "/index.html" | "/logs" | "/api/models" | "/api/health" | "/favicon.ico"
        ),
        Source::Shim => matches!(path, "/" | "/api/version" | "/api/tags" | "/api/ps"),
        Source::Server => false,
    }
}

/// Should this request announce itself *before* it runs?
///
/// The whole reason this module exists is the two minutes in which nothing
/// happens, so anything that can start model work says so on arrival: every
/// POST, and the WebSocket upgrade the chat UI uses. A GET answered from
/// memory in three milliseconds gets one line when it finishes, because two
/// would be noise.
fn announces_start(method: &Method, path: &str) -> bool {
    method == Method::POST || path.ends_with("/ws")
}

/// Fold a client-controlled string into something safe to paste into a line:
/// printable, bounded, single-line.
fn sanitize(value: &str) -> String {
    let mut out: String = value
        .chars()
        .take(MAX_FIELD)
        .map(|c| if c.is_control() { '·' } else { c })
        .collect();
    if value.chars().nth(MAX_FIELD).is_some() {
        out.push('…');
    }
    out
}

/// The one field worth pulling out of a request body.
///
/// **We never log bodies.** A chat body carries the user's whole prompt and
/// conversation — private, and routinely far bigger than a log line. The model
/// name answers the question an operator actually has ("which model is this
/// request waiting on?"); the sizes are read from headers. Everything else
/// stays where it was.
#[derive(serde::Deserialize)]
struct ModelHint {
    #[serde(default)]
    model: Option<String>,
    /// ollama's `/api/show` and `/api/delete` spell it `name`.
    #[serde(default)]
    name: Option<String>,
}

/// Buffer a small JSON body, read the model name out of it, and hand the
/// request back with its body intact.
///
/// Only for the handful of POST routes that name a model, and only when the
/// announced length is small: everything here already buffers its body (the
/// handlers take `Bytes`), so for those routes this costs one extra copy and
/// nothing else, and a large or chunked body is passed through untouched.
async fn take_model_hint(req: Request) -> (Request, Option<String>) {
    let interesting = req.method() == Method::POST
        && matches!(
            req.uri().path(),
            "/api/chat" | "/api/generate" | "/api/pull" | "/api/show" | "/api/delete"
        );
    let small = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|n| n <= MAX_MODEL_PEEK);
    if !(interesting && small) {
        return (req, None);
    }
    let (parts, body) = req.into_parts();
    // hyper enforces the announced content-length, and we checked it above, so
    // this can only fail on a transport error — one the handler would have hit
    // anyway. Rebuild with an empty body rather than leave the request broken.
    let Ok(bytes) = axum::body::to_bytes(body, MAX_MODEL_PEEK).await else {
        return (Request::from_parts(parts, axum::body::Body::empty()), None);
    };
    // serde skips the fields it does not know without allocating them, so a
    // 200 KiB prompt is walked, not copied.
    let hint = serde_json::from_slice::<ModelHint>(&bytes)
        .ok()
        .and_then(|h| h.model.or(h.name))
        .filter(|m| !m.is_empty())
        .map(|m| sanitize(&m));
    (
        Request::from_parts(parts, axum::body::Body::from(bytes)),
        hint,
    )
}

/// Is this response a stream whose length nobody knows yet?
///
/// Worth saying explicitly, because for a stream every *other* number on the
/// line means something different: the status is the head's, and the duration
/// is time-to-first-byte, not the generation.
fn is_streaming(response: &Response) -> &'static str {
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if content_type.starts_with("text/event-stream") || content_type.contains("x-ndjson") {
        " streaming"
    } else {
        ""
    }
}

/// How many bytes the response carries, when that is already known.
///
/// Read from the body's own size hint rather than a `content-length` header:
/// axum builds buffered responses without one (hyper adds it at the wire), so
/// the header is absent for every JSON answer this server gives. A stream's
/// hint has no exact value, which is exactly the case [`is_streaming`] names.
fn response_bytes(response: &Response) -> Option<u64> {
    use http_body::Body as _;
    response.body().size_hint().exact()
}

async fn observe(source: Source, req: Request, next: Next) -> Response {
    let raw_path = req.uri().path().to_owned();
    if ignored(source, &raw_path) {
        return next.run(req).await;
    }
    let method = req.method().clone();
    // Route decisions read the real path; only the *printed* copy is folded.
    let starts = announces_start(&method, &raw_path);
    let path = sanitize(&raw_path);

    if starts {
        // On ARRIVAL, before anything else — and in particular before the
        // model peek below, which buffers the body.
        //
        // That ordering is the whole point. A request that queues behind a
        // cold load produces nothing for its entire wait, and a feed that only
        // logs completions shows nothing during the minutes that matter. But
        // waiting for the body first reintroduced exactly that silence at a
        // smaller scale: a POST /api/chat arriving in two halves four seconds
        // apart logged NOTHING for those four seconds and then all its lines
        // at once, and a body that never completed logged nothing at all for
        // the life of the connection. A stalled upload is a thing an operator
        // needs to see, so the line goes out when the head arrives. The model
        // name is not known yet; it is attached to the completion line, which
        // is the one that can carry it.
        //
        // Never quiet: a request that announces itself is one that does work
        // (a POST, the WebSocket upgrade), and no such request is a poll —
        // `quiet_request` could not say otherwise even with the status in hand.
        record(
            source,
            Some(Level::Info),
            false,
            format!("{method} {path} → started"),
        );
    }

    let (req, model) = take_model_hint(req).await;
    let detail = model.map_or_else(String::new, |m| format!(" model={m}"));

    let started = Instant::now();
    let response = next.run(req).await;
    let ms = started.elapsed().as_millis();
    let status = response.status().as_u16();
    // Decided here, with the status in hand, not on arrival: the same poll is
    // routine while it succeeds and exactly what the owner needs to see the
    // moment it does not.
    let quiet = quiet_request(source, &method, &raw_path, status);
    let streaming = is_streaming(&response);
    let bytes = response_bytes(&response).map_or_else(String::new, |n| format!(" {n} B"));
    // For a streamed response this is time-to-first-byte, not the whole
    // generation: axum hands back the response head as soon as the stream
    // exists. The generation's own progress arrives on the `server` source.
    record(
        source,
        Some(level_for_status(status)),
        quiet,
        format!("{method} {path} {status} {ms} ms{bytes}{streaming}{detail}"),
    );
    response
}

/// A request entry's level is not a guess — the status code states it. This is
/// why request lines bypass [`classify`]: "POST /api/chat 409" contains none
/// of the heuristic's words, and a 409 that reads as routine info is precisely
/// the failure an operator is looking for when they open the page.
const fn level_for_status(status: u16) -> Level {
    if status >= 500 {
        Level::Error
    } else if status >= 400 {
        Level::Warn
    } else {
        Level::Info
    }
}

// ---------------------------------------------------------------------------
// Capturing the process's own output
// ---------------------------------------------------------------------------

/// Splits a byte stream into lines for the ring.
///
/// Separate from the pump so the fiddly parts — a line split across two reads,
/// a line longer than the ring's per-line budget, `\r\n`, a bare `\r` — are
/// testable without touching a file descriptor.
///
/// Off unix nothing feeds it outside the tests, and it is kept rather than
/// `cfg`'d away so those tests still run there: the reassembly rules are pure
/// byte handling, and a platform where they are never exercised is a platform
/// where a regression in them would land unnoticed.
#[cfg_attr(not(unix), allow(dead_code))]
struct Reassembler {
    pending: Vec<u8>,
    /// We already emitted this line's truncated head; swallow the rest of it.
    overlong: bool,
}

impl Reassembler {
    const fn new() -> Self {
        Self {
            pending: Vec::new(),
            overlong: false,
        }
    }

    /// Feed a chunk, calling `emit` once per completed line.
    ///
    /// Both `\n` and a bare `\r` end a line: a carriage-return progress
    /// updater would otherwise accumulate into one unbounded "line". Empty
    /// lines are dropped, which is also what makes `\r\n` one line and not two.
    fn feed(&mut self, chunk: &[u8], emit: &mut impl FnMut(String)) {
        for &byte in chunk {
            if byte == b'\n' || byte == b'\r' {
                self.end_line(emit);
                continue;
            }
            if self.overlong {
                continue;
            }
            self.pending.push(byte);
            if self.pending.len() >= MAX_LINE_BYTES {
                let mut text = String::from_utf8_lossy(&self.pending).into_owned();
                text.push_str(TRUNCATED);
                self.pending.clear();
                self.overlong = true;
                emit(text);
            }
        }
    }

    fn end_line(&mut self, emit: &mut impl FnMut(String)) {
        if self.overlong {
            self.overlong = false;
            self.pending.clear();
            return;
        }
        if self.pending.is_empty() {
            return;
        }
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        emit(text);
    }

    /// Flush a trailing line with no terminator (the writer closed mid-line).
    fn finish(&mut self, emit: &mut impl FnMut(String)) {
        self.end_line(emit);
    }
}

/// Start capturing this process's stdout and stderr into the ring. Idempotent:
/// both the binary and [`crate::serve_on`] call it, because whichever runs
/// first must win and the second must be a no-op — installing twice would
/// point fd 1 at a pipe whose reader writes to another pipe.
///
/// Call it as early as possible: everything printed before it lands only in
/// `docker logs`, and the lines worth seeing start at the first model load.
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // The banner has to say which of the two things actually happened.
        // Claiming capture unconditionally meant the Windows desktop shell's
        // log page asserted a capture it does not have, and then sat with an
        // empty `server` source forever — which reads as "the server printed
        // nothing", the exact wrong conclusion on the page you open when
        // nothing works.
        let text = if capture::install() {
            format!(
                "[mummu-serve] logs: capturing stdout/stderr into a {MAX_LINES}-line ring (+{QUIET_LINES} for health/poll traffic) — GET /logs"
            )
        } else {
            format!(
                "[mummu-serve] logs: request entries only — this platform has no stdout/stderr capture, so the `server` source stays empty and the process's own prints go to the console; {MAX_LINES}-line ring (+{QUIET_LINES} for health/poll traffic) — GET /logs"
            )
        };
        push(Source::Server, text);
    });
}

#[cfg(unix)]
mod capture {
    //! Tee fds 1 and 2 through a pipe.
    //!
    //! `dup` the original fd so we keep somewhere to write through to, `dup2`
    //! a pipe's write end over the fd itself, and read the pipe on a thread
    //! that (a) writes every byte straight back to the original fd, so
    //! `docker logs` is byte-identical, and (b) splits lines into the ring.
    //!
    //! Doing it at the *fd* level rather than by swapping `std::io::stderr` is
    //! what makes it free for existing code: `eprintln!`, `println!`, the
    //! panic runtime's own writes, and anything a C dependency prints all go
    //! through fd 1/2 and are all captured, with no call sites changed.

    use std::io::ErrorKind;
    use std::os::fd::RawFd;

    use super::{Reassembler, Source, push};

    /// Returns whether capture is in effect — always true here, because even
    /// if one `tee` fails the other may not, and either way the fds are left
    /// exactly as they were.
    pub(super) fn install() -> bool {
        for fd in [libc::STDOUT_FILENO, libc::STDERR_FILENO] {
            tee(fd);
        }
        true
    }

    fn tee(fd: RawFd) {
        // SAFETY: plain POSIX fd calls. `dup`/`pipe`/`dup2` are checked for
        // failure and every fd opened here is either closed on the failure
        // path or handed to the pump thread, which owns it for the life of the
        // process. Nothing here frees memory or aliases a Rust reference.
        let (read_fd, original) = unsafe {
            let original = libc::dup(fd);
            if original < 0 {
                return; // no way to write through; leave the fd alone entirely
            }
            let mut ends = [0 as libc::c_int; 2];
            if libc::pipe(ends.as_mut_ptr()) != 0 {
                libc::close(original);
                return;
            }
            let (read_fd, write_fd) = (ends[0], ends[1]);
            if libc::dup2(write_fd, fd) < 0 {
                libc::close(read_fd);
                libc::close(write_fd);
                libc::close(original);
                return;
            }
            // `fd` is now its own reference to the pipe's write end.
            libc::close(write_fd);
            (read_fd, original)
        };
        let name = if fd == libc::STDOUT_FILENO {
            "mummu-log-tee-out"
        } else {
            "mummu-log-tee-err"
        };
        if std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || pump(fd, read_fd, original))
            .is_err()
        {
            // Without a reader the pipe fills at 64 KiB and every print in the
            // process blocks forever, so put the fd back the way it was.
            // SAFETY: same plain fd calls; `original` is still ours here.
            unsafe {
                libc::dup2(original, fd);
                libc::close(original);
                libc::close(read_fd);
            }
        }
    }

    /// Undo the tee when the pump stops, however it stops.
    ///
    /// Without this, a pump thread that leaves its loop is a **silent,
    /// permanent wedge of the whole process**: fd 1/2 is still the pipe's
    /// write end, nobody is reading it, and the next 64 KiB of output fills
    /// the pipe buffer and blocks every printing thread in `write` forever.
    /// No log line, no panic message, no way to tell from outside — the
    /// server simply goes quiet and then stops answering as threads pile up.
    ///
    /// Two ways out of the loop reach here: a read error that is not `EINTR`,
    /// and a panic inside `push` (an allocation failure, a poisoned lock
    /// handler). The install path already reverts the `dup2` when the thread
    /// cannot be spawned, for exactly this reason; this is the same revert on
    /// the other way out, and `Drop` is what makes it cover the panic too.
    struct RestoreOnExit {
        fd: RawFd,
        original: RawFd,
        read_fd: RawFd,
    }

    impl Drop for RestoreOnExit {
        fn drop(&mut self) {
            // SAFETY: plain POSIX fd calls on fds this thread owns.
            // `dup2` closes `fd`'s current description (the pipe) as it
            // replaces it, which is what releases the writers.
            unsafe {
                libc::dup2(self.original, self.fd);
                libc::close(self.original);
                libc::close(self.read_fd);
            }
        }
    }

    fn pump(fd: RawFd, read_fd: RawFd, original: RawFd) {
        let restore = RestoreOnExit {
            fd,
            original,
            read_fd,
        };
        let mut buf = [0u8; 8192];
        let mut lines = Reassembler::new();
        let mut emit = |text: String| push(Source::Server, text);
        loop {
            // SAFETY: `buf` is a live, correctly sized local; `read_fd` is
            // owned by this thread (through `restore`) until it returns.
            let read = unsafe { libc::read(read_fd, buf.as_mut_ptr().cast(), buf.len()) };
            if read < 0 {
                if std::io::Error::last_os_error().kind() == ErrorKind::Interrupted {
                    continue;
                }
                break;
            }
            if read == 0 {
                break; // every writer is gone; the process is on its way out
            }
            let chunk = &buf[..read as usize];
            // Write through FIRST. The ring is a convenience; `docker logs` is
            // the record, and it must not lag or lose a byte because we were
            // busy classifying.
            write_all(original, chunk);
            lines.feed(chunk, &mut emit);
        }
        lines.finish(&mut emit);
        drop(restore);
    }

    fn write_all(fd: RawFd, mut buf: &[u8]) {
        while !buf.is_empty() {
            // SAFETY: `buf` is a live slice; `fd` is the dup of the original
            // stream, owned by this thread.
            let wrote = unsafe { libc::write(fd, buf.as_ptr().cast(), buf.len()) };
            if wrote > 0 {
                buf = &buf[wrote as usize..];
                continue;
            }
            if wrote < 0 && std::io::Error::last_os_error().kind() == ErrorKind::Interrupted {
                continue;
            }
            return; // the original stream is gone; there is nowhere left to complain
        }
    }
}

#[cfg(not(unix))]
mod capture {
    //! No capture off unix.
    //!
    //! The tee needs `dup`/`dup2` over fds 1 and 2. Windows has
    //! `SetStdHandle`, but std caches its `Stdout`/`Stderr` handles on first
    //! use, so a swap installed after startup would be honoured by some
    //! writers and not others — a half-captured log is worse than an honest
    //! none. The ring, the endpoint and the page all still work: request
    //! entries are explicit [`super::push`] calls and appear exactly as they
    //! do on unix; only the `server` source stays empty, and the process's
    //! prints go to the console as they always did.
    //!
    //! `install` returns `false` so [`super::install`]'s banner says that
    //! rather than announcing a capture this target does not have.
    pub(super) fn install() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The ring the endpoint serves is process-wide, and `cargo test` runs
    // these in parallel threads. So the cursor/bounds tests drive a ring they
    // own; only `the_global_ring_records_what_is_pushed_to_it` touches the
    // singleton, and it is the only test that pushes to it.
    fn ring_of(entries: &[(Source, &str)]) -> Ring {
        let mut ring = Ring::new();
        for (source, text) in entries {
            ring.record(line(*source, None, false, (*text).to_owned()));
        }
        ring
    }

    fn page(ring: &Ring, since: u64, limit: usize, source: Option<Source>) -> Page {
        read(
            ring,
            &Query {
                since,
                limit,
                source,
            },
        )
    }

    fn texts(page: &Page) -> Vec<&str> {
        page.lines.iter().map(|l| l.text.as_str()).collect()
    }

    fn seqs(page: &Page) -> Vec<u64> {
        page.lines.iter().map(|l| l.seq).collect()
    }

    /// A main-ring line: server output, or a request that did something.
    fn loud(ring: &mut Ring, source: Source, text: &str) {
        ring.record(line(source, None, false, text.to_owned()));
    }

    /// A quiet-ring line: a poll, exactly as the middleware records one.
    fn poll(ring: &mut Ring, text: &str) {
        ring.record(line(Source::Api, Some(Level::Info), true, text.to_owned()));
    }

    /// A deterministic, irregular traffic mix (xorshift64), so the merge and
    /// the gap count are exercised on interleavings no hand-written pattern
    /// would think of — and the same ones on every run.
    struct Mix(u64);

    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    // -- level classification ---------------------------------------------

    #[test]
    fn classify_reads_the_levels_it_promises() {
        // The line that matters most: a Rust panic, exactly as the runtime
        // prints it to stderr. If this ever stops classifying as an error, the
        // CUDA failure that started all this goes back to being invisible.
        assert_eq!(
            classify("thread 'DSD-0-0' panicked at src/nn/moe.rs:118: CUDA_ERROR_UNKNOWN"),
            Level::Error
        );
        assert_eq!(
            classify("[mummu-serve] serve failed: address in use"),
            Level::Error
        );
        assert_eq!(classify("pull ERROR: 404"), Level::Error);
        // The fit planner's budget flag is only meaningful as that exact token.
        assert_eq!(classify("cuda:0 9.32 GiB (OVER — will spill)"), Level::Warn);
        assert_eq!(
            classify("[mummu-serve] residency: no fresh VRAM baseline — placement unverified"),
            Level::Warn
        );
        assert_eq!(
            classify("[mummu] load: 673/851 tensors — 14.46 GiB off the pack in 91s (162 MB/s)"),
            Level::Info
        );
        // Error beats warn when one line says both.
        assert_eq!(classify("warning: the load failed"), Level::Error);
    }

    // -- cursor semantics --------------------------------------------------

    #[test]
    fn a_cursor_returns_only_what_is_new() {
        let mut ring = ring_of(&[(Source::Server, "one"), (Source::Server, "two")]);
        let first = page(&ring, 0, 10, None);
        assert_eq!(texts(&first), ["one", "two"]);
        assert_eq!(first.dropped, 0);
        // Polling again with the handed-back cursor yields nothing new.
        assert!(page(&ring, first.cursor, 10, None).lines.is_empty());

        ring.record(line(Source::Server, None, false, "three".into()));
        let next = page(&ring, first.cursor, 10, None);
        assert_eq!(texts(&next), ["three"]);
        assert!(next.lines[0].seq > first.lines[1].seq, "seq is monotonic");
    }

    #[test]
    fn an_empty_ring_answers_an_empty_page() {
        let ring = Ring::new();
        let first = page(&ring, 0, 10, None);
        assert!(first.lines.is_empty());
        assert_eq!(first.cursor, 0);
        assert_eq!(first.newest, 0);
        assert_eq!(first.dropped, 0);
    }

    /// A `since` beyond anything this process produced means the client kept a
    /// cursor across a server restart. Re-sync it to everything the NEW
    /// process still holds — which is the startup banner it just printed, the
    /// thing the reader restarted the server to watch. Re-syncing to `newest`
    /// instead left the page blank until something else happened to print.
    #[test]
    fn a_since_in_the_future_replays_what_the_new_process_printed() {
        let ring = ring_of(&[
            (
                Source::Server,
                "[mummu-serve] logs: capturing stdout/stderr",
            ),
            (Source::Server, "[mummu-serve] adapter: RTX 4070 Ti SUPER"),
            (
                Source::Server,
                "[mummu-serve] listening on http://0.0.0.0:8095",
            ),
        ]);
        let stale = page(&ring, u64::MAX, 10, None);
        assert_eq!(
            texts(&stale),
            [
                "[mummu-serve] logs: capturing stdout/stderr",
                "[mummu-serve] adapter: RTX 4070 Ti SUPER",
                "[mummu-serve] listening on http://0.0.0.0:8095",
            ],
            "the new process's whole startup lands, not nothing"
        );
        assert_eq!(
            stale.cursor, stale.newest,
            "a cursor that comes back lower is the re-sync signal"
        );
        assert_eq!(stale.dropped, 0, "nothing was missed — nothing existed");
        // And the re-synced cursor is then a normal cursor: no replay.
        assert!(page(&ring, stale.cursor, 10, None).lines.is_empty());
    }

    /// The same re-sync against a ring that has already overrun: the client
    /// gets what is left, and is told how much it missed rather than being
    /// handed a silent splice.
    #[test]
    fn a_future_since_against_an_overrun_ring_reports_the_gap() {
        let mut ring = Ring::new();
        for i in 0..MAX_LINES + 10 {
            ring.record(line(Source::Server, None, false, format!("restarted {i}")));
        }
        let stale = page(&ring, u64::MAX, MAX_LIMIT, None);
        assert_eq!(stale.lines.len(), MAX_LINES);
        assert_eq!(stale.lines[0].text, "restarted 10");
        assert_eq!(stale.dropped, 10, "the gap is confessed, not hidden");
    }

    /// A `since` older than anything the ring still holds must report the size
    /// of the gap, so the page draws a break instead of silently splicing.
    #[test]
    fn a_since_far_in_the_past_reports_the_gap() {
        let mut ring = Ring::new();
        for i in 0..MAX_LINES + 50 {
            ring.record(line(Source::Server, None, false, format!("flood {i}")));
        }
        let all = page(&ring, 0, MAX_LIMIT, None);
        assert_eq!(all.lines.len(), MAX_LINES);
        assert_eq!(
            all.dropped, 50,
            "the 50 evicted lines are reported, not hidden"
        );
        assert_eq!(all.dropped, all.oldest - 1);
        assert_eq!(all.dropped_total, 50);
        assert_eq!(all.lines[0].text, "flood 50");
        // A cursor that is still inside the ring has no gap.
        assert_eq!(page(&ring, all.cursor, 10, None).dropped, 0);
        // Nor does one exactly at the oldest surviving line's predecessor.
        assert_eq!(page(&ring, all.oldest - 1, 10, None).dropped, 0);
    }

    #[test]
    fn the_ring_never_grows_past_its_cap() {
        let mut ring = Ring::new();
        for i in 0..MAX_LINES * 2 {
            ring.record(line(Source::Server, None, false, format!("bound {i}")));
        }
        assert_eq!(ring.lines.len(), MAX_LINES);
        let all = page(&ring, 0, MAX_LIMIT, None);
        assert_eq!(all.newest - all.oldest + 1, MAX_LINES as u64);
        assert_eq!(all.newest, (MAX_LINES * 2) as u64);
    }

    #[test]
    fn a_limit_stops_the_page_and_leaves_the_cursor_mid_span() {
        let ring = ring_of(&[
            (Source::Server, "a"),
            (Source::Server, "b"),
            (Source::Server, "c"),
            (Source::Server, "d"),
        ]);
        let first = page(&ring, 0, 2, None);
        assert_eq!(texts(&first), ["a", "b"]);
        assert_eq!(first.cursor, first.lines[1].seq);
        assert!(
            first.cursor < first.newest,
            "the cursor stops where the page did"
        );
        assert_eq!(texts(&page(&ring, first.cursor, 10, None)), ["c", "d"]);
        assert!(first.more, "the ring is holding c and d");
        assert!(
            !page(&ring, first.cursor, 10, None).more,
            "and nothing once they have been handed over"
        );
    }

    /// `/logs` and `/api/logs` are public and unauthenticated by decision, so
    /// the cost of ONE request cannot be the caller's to choose. `limit`
    /// alone allowed ~4 MiB; the byte budget makes the worst case small
    /// without changing what a reader sees — the page stops, the cursor
    /// points at the last line handed over, and the next poll continues.
    #[test]
    fn one_response_is_bounded_in_bytes_however_much_the_caller_asks_for() {
        // A ring full of the largest lines the ring will hold: ~4 MiB of
        // text, which is exactly what an anonymous caller could ask for in a
        // loop before this bound existed.
        let mut ring = Ring::new();
        for i in 0..MAX_LINES {
            ring.record(line(
                Source::Server,
                None,
                false,
                format!("{i:06} {}", "x".repeat(MAX_LINE_BYTES - 7)),
            ));
        }
        let first = page(&ring, 0, MAX_LIMIT, None);
        let bytes: usize = first.lines.iter().map(|l| l.text.len()).sum();
        assert!(
            bytes <= MAX_PAGE_BYTES,
            "one response carried {bytes} bytes of text against a {MAX_PAGE_BYTES} budget"
        );
        assert!(
            first.lines.len() < MAX_LINES,
            "this ring is bigger than the budget, so the page must stop early"
        );
        assert!(first.more, "and must say that it stopped early");

        // Nothing is lost: the whole ring still arrives, in order, with no
        // gap and no repeat — the same semantics `limit` already had.
        let mut seen = first.lines.len();
        let mut cursor = first.cursor;
        let mut guard = 0;
        while seen < MAX_LINES {
            guard += 1;
            assert!(guard < 100, "paging the ring should not take 100 requests");
            let next = page(&ring, cursor, MAX_LIMIT, None);
            assert!(!next.lines.is_empty(), "a page must always make progress");
            assert_eq!(
                next.lines[0].seq,
                cursor + 1,
                "the next page resumes at the line after the cursor"
            );
            seen += next.lines.len();
            cursor = next.cursor;
        }
        assert_eq!(seen, MAX_LINES, "every line arrived exactly once");
        assert!(!page(&ring, cursor, MAX_LIMIT, None).more);
    }

    /// The ordinary reader must not be able to tell the budget exists: a
    /// whole ring of REAL log lines (the ones this server actually prints)
    /// fits in one page, with room to spare.
    #[test]
    fn a_ring_of_real_lines_still_arrives_in_one_page() {
        let mut ring = Ring::new();
        for i in 0..MAX_LINES {
            ring.record(line(
                Source::Server,
                None,
                false,
                format!("[mummu] load: {i}/851 tensors — 14.46 GiB off the pack in 91s (162 MB/s)"),
            ));
        }
        let all = page(&ring, 0, MAX_LIMIT, None);
        assert_eq!(
            all.lines.len(),
            MAX_LINES,
            "a full ring of real lines must still come back in one response"
        );
        assert!(!all.more);
    }

    /// A single line larger than the whole budget must still be delivered.
    /// Skipping it would park the cursor in front of it forever — the client
    /// would ask again, get nothing again, and the feed would stop dead at
    /// the one line most likely to be the interesting one (a panic payload).
    #[test]
    fn a_line_bigger_than_the_budget_is_still_handed_over() {
        // Two lines, each already truncated to MAX_LINE_BYTES; the budget is
        // set below one of them only in the pathological case, so this checks
        // the rule directly with the first-line exemption.
        let ring = ring_of(&[
            (Source::Server, &"x".repeat(MAX_LINE_BYTES * 3)),
            (Source::Server, "the line after it"),
        ]);
        let first = page(&ring, 0, MAX_LIMIT, None);
        assert!(!first.lines.is_empty(), "the big line came back");
        assert_eq!(
            texts(&page(&ring, first.lines[0].seq, MAX_LIMIT, None)),
            ["the line after it"],
            "and the feed continued past it"
        );
    }

    // -- the merge ---------------------------------------------------------

    #[test]
    fn one_ring_keeps_every_source_in_the_order_things_happened() {
        let ring = ring_of(&[
            (Source::Api, "POST /api/chat → started model=qwen3-0.6b"),
            (Source::Server, "[mummu-serve] chat request for qwen3-0.6b"),
            (Source::Shim, "POST /api/generate → started"),
            (Source::Server, "[mummu] load: 1/311 tensors"),
        ]);
        let merged = page(&ring, 0, 10, None);
        let sources: Vec<_> = merged.lines.iter().map(|l| l.source).collect();
        assert_eq!(
            sources,
            [Source::Api, Source::Server, Source::Shim, Source::Server]
        );
    }

    /// A filtered read must still advance the cursor over the lines it skipped,
    /// or a source-filtered client rescans the same span on every poll.
    #[test]
    fn a_source_filter_advances_past_what_it_skipped() {
        let ring = ring_of(&[
            (Source::Server, "server line"),
            (Source::Api, "GET /api/models 200 2 ms"),
            (Source::Shim, "POST /api/chat → started"),
            (Source::Server, "another server line"),
        ]);
        let api_only = page(&ring, 0, 10, Some(Source::Api));
        assert_eq!(texts(&api_only), ["GET /api/models 200 2 ms"]);
        assert_eq!(
            api_only.cursor, api_only.newest,
            "an exhausted scan lands on the newest seq, filtered out or not"
        );
        assert!(
            page(&ring, api_only.cursor, 10, Some(Source::Api))
                .lines
                .is_empty()
        );

        // With a limit, the cursor must stop at the last line actually handed
        // over — not past an unread match.
        let ring = ring_of(&[
            (Source::Api, "first"),
            (Source::Server, "skipped"),
            (Source::Api, "second"),
        ]);
        let one = page(&ring, 0, 1, Some(Source::Api));
        assert_eq!(texts(&one), ["first"]);
        assert_eq!(
            texts(&page(&ring, one.cursor, 10, Some(Source::Api))),
            ["second"]
        );
    }

    // -- the two rings -------------------------------------------------------
    //
    // The first two tests are the regression guards for the defect v0.3.1
    // exists for, and they read only what v0.3.0's `Page` already had
    // (`lines`, `dropped`, `cursor`, `dropped_total`) — no `dropped_exact`,
    // no quiet totals. That is deliberate: pasted into v0.3.0's module (with
    // `QUIET_LINES` defined as the number it is) they compile, and they FAIL.
    // A guard that cannot fail against the code it guards against is not
    // guarding anything.

    /// The defect, as a test. v0.3.0 flagged polls quiet but kept them in the
    /// one ring, and on the live server they evicted a cold load within six
    /// hours. Forty quiet rings' worth of probes later — ten of v0.3.0's whole
    /// ring — the load line and the request that caused it must still be
    /// readable, and nothing may be reported missing in front of them.
    #[test]
    fn server_lines_survive_any_amount_of_quiet_traffic() {
        let mut ring = Ring::new();
        let load = "[mummu] load: 851/851 tensors — 14.46 GiB off the pack in 121s (122 MB/s)";
        let chat = "POST /api/chat 200 121034 ms streaming model=qwen3.6-27b";
        loud(&mut ring, Source::Server, load);
        loud(&mut ring, Source::Api, chat);
        for i in 0..QUIET_LINES * 40 {
            poll(&mut ring, &format!("GET /api/health 200 {i} ms"));
        }

        let all = page(&ring, 0, MAX_LIMIT, None);
        let kept: Vec<&str> = all
            .lines
            .iter()
            .filter(|l| !l.quiet)
            .map(|l| l.text.as_str())
            .collect();
        assert_eq!(kept, [load, chat], "both main-ring lines are still held");
        assert_eq!(all.dropped, 0, "nothing the reader could see was lost");
        // And the reader who asks for the server's own output gets the load.
        let server = page(&ring, 0, MAX_LIMIT, Some(Source::Server));
        assert_eq!(texts(&server), [load]);
    }

    /// `dropped` counts MAIN-ring losses and nothing else. The page draws a
    /// gap marker for it, and quiet lines are hidden by default — so a quiet
    /// line ageing out of its own ring must never register as a gap.
    #[test]
    fn dropped_does_not_count_quiet_evictions() {
        // Only quiet lines have ever been evicted: no gap, from any cursor.
        let mut ring = Ring::new();
        loud(&mut ring, Source::Server, "startup");
        for i in 0..QUIET_LINES * 4 {
            poll(&mut ring, &format!("GET /api/health 200 {i} ms"));
        }
        for since in [0, 1, 2, 700, ring.next_seq - 2] {
            assert_eq!(page(&ring, since, 10, None).dropped, 0, "since={since}");
        }

        // The case a seq-span count gets wrong. The client last saw a server
        // line; then only polls arrived for a while (and aged out); then
        // server output overran the main ring — evicting only lines the
        // client HAD seen. Between its cursor and the main ring's front lie a
        // thousand seqs, every one a quiet line: nothing it could have missed.
        let mut ring = Ring::new();
        for i in 0..10 {
            loud(&mut ring, Source::Server, &format!("seen {i}"));
        }
        let cursor = page(&ring, 0, MAX_LIMIT, None).cursor;
        for i in 0..QUIET_LINES * 2 {
            poll(&mut ring, &format!("GET / 200 {i} ms"));
        }
        for i in 0..MAX_LINES {
            loud(&mut ring, Source::Server, &format!("later {i}"));
        }
        let front = ring.lines.front().expect("held").seq;
        assert_eq!(
            front - 1 - cursor,
            (QUIET_LINES * 2) as u64,
            "the naive span"
        );
        let p = page(&ring, cursor, MAX_LIMIT, None);
        assert_eq!(p.dropped, 0, "no marker for a span of aged-out polls");
        assert_eq!(p.dropped_total, 10, "the ten it evicted were all seen");

        // And when main-ring lines ARE lost, exactly those are counted, not
        // the polls interleaved with them.
        let mut ring = Ring::new();
        loud(&mut ring, Source::Server, "seen");
        for i in 0..MAX_LINES + 25 {
            loud(&mut ring, Source::Server, &format!("missed {i}"));
            poll(&mut ring, "GET /api/health 200 1 ms");
            poll(&mut ring, "GET /api/models 200 3 ms");
        }
        assert_eq!(page(&ring, 1, MAX_LIMIT, None).dropped, 25);
        assert_eq!(page(&ring, 0, MAX_LIMIT, None).dropped, 26);
    }

    /// The quiet ring is bounded, keeps the most recent probes, and counts
    /// what it let go on a total of its own — the part of the first guard's
    /// story that only v0.3.1's fields can tell.
    #[test]
    fn the_quiet_ring_keeps_the_newest_probes_and_counts_the_rest() {
        let mut ring = Ring::new();
        loud(&mut ring, Source::Server, "[mummu-serve] listening");
        let flood = QUIET_LINES * 20 + 7;
        for i in 0..flood {
            poll(&mut ring, &format!("GET /api/health 200 {i} ms"));
        }
        assert_eq!(ring.quiet.len(), QUIET_LINES, "bounded");

        let all = page(&ring, 0, MAX_LIMIT, None);
        let quiet: Vec<_> = all.lines.iter().filter(|l| l.quiet).collect();
        assert_eq!(quiet.len(), QUIET_LINES);
        assert_eq!(
            quiet.last().expect("probes held").text,
            format!("GET /api/health 200 {} ms", flood - 1),
            "the most recent probe is the one kept — \"when did it last run?\""
        );
        assert_eq!(all.dropped_quiet_total, (flood - QUIET_LINES) as u64);
        assert_eq!(all.dropped_total, 0, "the main ring lost nothing");
        assert!(all.dropped_exact);
        assert_eq!(all.newest, (flood + 1) as u64, "one seq space for both");
    }

    /// Paged through in small pieces, the merged read is every held line of
    /// both rings, each exactly once, in strictly increasing seq — on an
    /// irregular mix where both rings have evicted at their own rates. And it
    /// is NOT contiguous, which is exactly why nothing may treat a seq jump
    /// as a gap.
    #[test]
    fn the_merged_read_is_in_seq_order_across_both_rings() {
        let mut ring = Ring::new();
        let mut mix = Mix(0x5eed_1234_abcd_0001);
        for i in 0..(MAX_LINES + QUIET_LINES) * 4 {
            if mix.next().is_multiple_of(3) {
                loud(&mut ring, Source::Server, &format!("server {i}"));
            } else {
                poll(&mut ring, &format!("GET /api/health 200 {i} ms"));
            }
        }
        assert!(ring.dropped_total > 0 && ring.dropped_quiet_total > 0);

        let mut held: Vec<u64> = ring
            .lines
            .iter()
            .chain(&ring.quiet)
            .map(|l| l.seq)
            .collect();
        held.sort_unstable();
        let (mut cursor, mut seen, mut guard) = (0, Vec::new(), 0);
        loop {
            guard += 1;
            assert!(guard < 1000, "paging must terminate");
            let next = page(&ring, cursor, 97, None);
            seen.extend(seqs(&next));
            cursor = next.cursor;
            if !next.more {
                break;
            }
        }
        assert!(
            seen.windows(2).all(|w| w[0] < w[1]),
            "strictly increasing seq"
        );
        assert_eq!(seen, held, "every held line of both rings, once");
        assert!(
            seen.windows(2).any(|w| w[1] > w[0] + 1),
            "a merged read has holes where quiet lines aged out — holes, not gaps"
        );

        // A source filter over the merge keeps the order, and keeps the quiet
        // lines of its own source.
        let api = page(&ring, 0, MAX_LIMIT, Some(Source::Api));
        assert!(api.lines.iter().all(|l| l.source == Source::Api && l.quiet));
        assert!(seqs(&api).windows(2).all(|w| w[0] < w[1]));
    }

    /// A client that polls with `since=<cursor>` — both embedded pages, and
    /// anything written against v0.3.0 — sees what it always saw: every new
    /// line exactly once, loud or quiet, in order, with the cursor landing on
    /// `newest` and no gap reported. Run long enough that BOTH rings evict
    /// many times over underneath it, because that is production: a poller
    /// that keeps up must never be told it missed anything.
    #[test]
    fn a_polling_cursor_sees_every_line_once_across_both_rings() {
        let mut ring = Ring::new();
        let (mut cursor, mut seen, mut round) = (0, Vec::new(), 0u64);
        while ring.next_seq < ((MAX_LINES + QUIET_LINES) * 3) as u64 {
            round += 1;
            for i in 0..=round % 7 {
                if (round + i) % 3 == 0 {
                    loud(&mut ring, Source::Server, &format!("server {round}.{i}"));
                } else {
                    poll(&mut ring, &format!("GET / 200 {round}.{i} ms"));
                }
            }
            let next = page(&ring, cursor, MAX_LIMIT, None);
            assert!(next.lines.iter().all(|l| l.seq > cursor));
            assert_eq!(next.cursor, next.newest);
            assert!(!next.more);
            assert_eq!(next.dropped, 0, "round {round}");
            seen.extend(seqs(&next));
            cursor = next.cursor;
        }
        assert!(
            ring.dropped_total > 0 && ring.dropped_quiet_total > 0,
            "both rings evicted under the poller"
        );
        let every: Vec<u64> = (1..ring.next_seq).collect();
        assert_eq!(seen, every, "every line, once, in order");

        // A cursor resting on a quiet line that has since aged out of its
        // ring is still a cursor in THIS process — not a restart, not a gap.
        poll(&mut ring, "GET /api/health 200 1 ms");
        let rest = page(&ring, cursor, MAX_LIMIT, None).cursor;
        for i in 0..QUIET_LINES * 2 {
            poll(&mut ring, &format!("GET /api/models 200 {i} ms"));
        }
        let back = page(&ring, rest, MAX_LIMIT, None);
        assert!(back.cursor > rest, "carried forward, not re-synced");
        assert_eq!(back.dropped, 0, "only quiet lines went, so no gap");
        assert_eq!(back.lines.len(), QUIET_LINES);
    }

    /// A restart under a page that kept its cursor: the new process's whole
    /// startup — server lines AND the probes that arrived during it — is
    /// replayed from both rings, in order, and the gap reported is main-ring
    /// losses only.
    #[test]
    fn a_since_in_the_future_replays_both_rings() {
        let mut ring = Ring::new();
        loud(&mut ring, Source::Server, "[mummu-serve] logs: capturing");
        poll(&mut ring, "GET /api/health 200 2 ms");
        loud(&mut ring, Source::Server, "[mummu-serve] listening");
        poll(&mut ring, "GET /api/health 200 1 ms");
        let stale = page(&ring, 12_345, MAX_LIMIT, None);
        assert_eq!(
            texts(&stale),
            [
                "[mummu-serve] logs: capturing",
                "GET /api/health 200 2 ms",
                "[mummu-serve] listening",
                "GET /api/health 200 1 ms",
            ]
        );
        assert_eq!(
            stale.cursor, stale.newest,
            "lower than sent: the re-sync signal"
        );
        assert_eq!(stale.dropped, 0);

        // The same restart against a process that has overrun both rings.
        let mut ring = Ring::new();
        for i in 0..MAX_LINES + 10 {
            loud(&mut ring, Source::Server, &format!("restarted {i}"));
        }
        for i in 0..QUIET_LINES + 99 {
            poll(&mut ring, &format!("GET /api/health 200 {i} ms"));
        }
        let stale = page(&ring, u64::MAX, MAX_LIMIT, None);
        assert_eq!(stale.dropped, 10, "main-ring losses only");
        assert!(stale.dropped_exact);
        assert_eq!(stale.dropped_quiet_total, 99);
        assert_eq!(
            stale.lines[0].seq, stale.oldest,
            "replayed from the oldest held"
        );
        assert_eq!(stale.lines[0].text, "restarted 10");
        assert!(
            stale.more,
            "2000 + 500 held lines is two pages at limit 2000"
        );
    }

    /// `dropped` against a brute-force count, from every cursor position, on
    /// an irregular mix long enough to overrun the eviction memory — so both
    /// the exact path and the floor path run. Exact means equal; a floor
    /// must never exceed the truth and never hide a real gap.
    #[test]
    fn dropped_matches_a_brute_force_count_or_says_it_is_a_floor() {
        let mut ring = Ring::new();
        let mut main_seqs = Vec::new();
        let mut mix = Mix(0x0bad_cafe_f00d_0042);
        for i in 0..(MAX_LINES + EVICTION_MEMORY) * 4 {
            if mix.next().is_multiple_of(3) {
                loud(&mut ring, Source::Server, &format!("server {i}"));
                main_seqs.push(ring.next_seq - 1);
            } else {
                poll(&mut ring, "GET /api/health 200 1 ms");
            }
        }
        assert!(
            ring.dropped_total > EVICTION_MEMORY as u64,
            "the mix must overrun the eviction memory for the floor path to run"
        );
        let front = ring.lines.front().expect("held").seq;
        let (mut exact, mut floors) = (0, 0);
        for since in (0..ring.next_seq + 5)
            .step_by(7)
            .chain([0, 1, front - 1, front])
        {
            if since >= ring.next_seq {
                continue; // a re-sync, covered by its own test
            }
            let truth = main_seqs
                .iter()
                .filter(|&&s| s > since && s < front)
                .count() as u64;
            let p = page(&ring, since, 1, None);
            if p.dropped_exact {
                assert_eq!(p.dropped, truth, "since={since}");
                exact += 1;
            } else {
                assert!(
                    p.dropped <= truth,
                    "a floor above the truth at since={since}"
                );
                assert!(
                    p.dropped > 0,
                    "a real gap reported as none at since={since}"
                );
                floors += 1;
            }
        }
        assert!(exact > 0 && floors > 0, "exact {exact}, floors {floors}");
    }

    /// Everything a client written against v0.3.0 reads is still there, with
    /// the same type and — for `capacity` — the same meaning: the ring that
    /// `dropped` is about. What v0.3.1 has to add, it adds beside them.
    #[test]
    fn the_response_keeps_every_field_a_v030_client_reads() {
        let mut ring = Ring::new();
        loud(&mut ring, Source::Server, "startup");
        poll(&mut ring, "GET /api/health 200 1 ms");
        let body = page(&ring, 0, MAX_LIMIT, None).to_json();
        for key in [
            "cursor",
            "dropped",
            "dropped_total",
            "oldest",
            "newest",
            "capacity",
        ] {
            assert!(body[key].is_u64(), "{key} is still a number: {body}");
        }
        assert!(body["more"].is_boolean());
        assert_eq!(body["capacity"], json!(MAX_LINES), "the main ring's");
        let lines = body["lines"].as_array().expect("lines is still an array");
        assert_eq!(lines.len(), 2, "quiet lines are still in the same list");
        for key in ["seq", "ts"] {
            assert!(lines[1][key].is_u64(), "line.{key}");
        }
        for key in ["level", "source", "text"] {
            assert!(lines[1][key].is_string(), "line.{key}");
        }
        assert_eq!(lines[1]["quiet"], json!(true));
        // Added, never substituted.
        assert_eq!(body["quiet_capacity"], json!(QUIET_LINES));
        assert_eq!(body["dropped_quiet_total"], json!(0));
        assert_eq!(body["dropped_exact"], json!(true));
    }

    #[test]
    fn an_over_long_push_is_truncated_not_dropped() {
        let ring = ring_of(&[(Source::Server, &"x".repeat(MAX_LINE_BYTES * 3))]);
        let only = page(&ring, 0, 10, None);
        assert_eq!(only.lines.len(), 1);
        assert!(only.lines[0].text.ends_with(TRUNCATED));
        assert_eq!(only.lines[0].text.len(), MAX_LINE_BYTES + TRUNCATED.len());
    }

    /// The one test that touches the process-wide ring the endpoint serves:
    /// pushes really do land in it, with a level, a source and a quiet flag.
    #[test]
    fn the_global_ring_records_what_is_pushed_to_it() {
        let before = query(&Query {
            since: 0,
            limit: 1,
            source: None,
        })
        .newest;
        push(Source::Api, "GET /api/health 200 1 ms");
        push_quiet(Source::Shim, "GET /api/tags 200 3 ms");
        let after = query(&Query {
            since: before,
            limit: MAX_LIMIT,
            source: None,
        });
        let mine: Vec<_> = after
            .lines
            .iter()
            .filter(|l| l.text.starts_with("GET /api/"))
            .collect();
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].source, Source::Api);
        assert!(!mine[0].quiet);
        assert_eq!(mine[1].source, Source::Shim);
        assert!(mine[1].quiet, "routine traffic stays foldable");
        assert!(mine[0].unix_ms > 0);
    }

    // -- query parsing -----------------------------------------------------

    fn params(since: Option<&str>, limit: Option<&str>, source: Option<&str>) -> LogsParams {
        LogsParams {
            since: since.map(str::to_owned),
            limit: limit.map(str::to_owned),
            source: source.map(str::to_owned),
        }
    }

    #[test]
    fn query_defaults_apply_when_nothing_is_asked_for() {
        let q = params(None, None, None)
            .resolve()
            .expect("defaults resolve");
        assert_eq!(
            q,
            Query {
                since: 0,
                limit: DEFAULT_LIMIT,
                source: None
            }
        );
    }

    #[test]
    fn limit_is_clamped_at_both_ends() {
        assert_eq!(
            params(None, Some("0"), None)
                .resolve()
                .expect("clamped")
                .limit,
            1,
            "a limit of zero would poll forever and return nothing"
        );
        assert_eq!(
            params(None, Some("99999"), None)
                .resolve()
                .expect("clamped")
                .limit,
            MAX_LIMIT
        );
        assert_eq!(
            params(None, Some(" 7 "), None)
                .resolve()
                .expect("trimmed")
                .limit,
            7
        );
    }

    /// A mistyped cursor should still show logs — that is the whole job.
    #[test]
    fn unparseable_numbers_fall_back_instead_of_failing() {
        let q = params(Some("not-a-number"), Some("neither"), None)
            .resolve()
            .expect("a bad number is not a rejection");
        assert_eq!(q.since, 0);
        assert_eq!(q.limit, DEFAULT_LIMIT);
    }

    #[test]
    fn source_is_parsed_strictly() {
        assert_eq!(
            params(None, None, Some("all"))
                .resolve()
                .expect("all")
                .source,
            None
        );
        assert_eq!(
            params(None, None, Some("shim"))
                .resolve()
                .expect("shim")
                .source,
            Some(Source::Shim)
        );
        let err = params(None, None, Some("sever"))
            .resolve()
            .expect_err("a typo is worth reporting");
        assert!(
            err.contains("sever"),
            "the error names what was sent: {err}"
        );
    }

    // -- line reassembly ---------------------------------------------------

    fn collect(chunks: &[&[u8]]) -> Vec<String> {
        let mut out = Vec::new();
        let mut asm = Reassembler::new();
        let mut emit = |l: String| out.push(l);
        for chunk in chunks {
            asm.feed(chunk, &mut emit);
        }
        asm.finish(&mut emit);
        out
    }

    /// A read boundary lands wherever the pipe decides — mid-line, mid-word,
    /// mid-UTF-8 — and the line must come out whole regardless.
    #[test]
    fn a_line_split_across_reads_is_rejoined() {
        assert_eq!(
            collect(&[
                b"[mummu] load: 673/851 ten",
                b"sors \xe2\x80",
                b"\x94 14.46 GiB\n"
            ]),
            ["[mummu] load: 673/851 tensors — 14.46 GiB"]
        );
    }

    #[test]
    fn terminators_and_empty_lines() {
        assert_eq!(
            collect(&[b"a\r\nb\n"]),
            ["a", "b"],
            "CRLF is one break, not two"
        );
        assert_eq!(collect(&[b"\n\n\nsolo\n"]), ["solo"]);
        assert_eq!(
            collect(&[b"progress 1\rprogress 2\r"]),
            ["progress 1", "progress 2"],
            "a bare CR ends a line, so a progress updater cannot grow without bound"
        );
    }

    /// The writer closed mid-line (the process is exiting). Keep what we have.
    #[test]
    fn a_trailing_line_with_no_newline_is_still_emitted() {
        assert_eq!(collect(&[b"half a line"]), ["half a line"]);
    }

    /// A single line longer than the per-line budget must produce ONE truncated
    /// entry and then get out of the way — not a burst of fragments, and not an
    /// unbounded `pending`.
    #[test]
    fn an_over_long_line_yields_one_truncated_entry() {
        let mut blob = vec![b'z'; MAX_LINE_BYTES * 4];
        blob.push(b'\n');
        blob.extend_from_slice(b"next line\n");
        let lines = collect(&[&blob]);
        assert_eq!(lines.len(), 2, "one truncated entry, then the next line");
        assert!(lines[0].ends_with(TRUNCATED));
        assert_eq!(lines[0].len(), MAX_LINE_BYTES + TRUNCATED.len());
        assert_eq!(lines[1], "next line");
    }

    /// A panic is several lines on stderr, arriving in whatever chunks the pipe
    /// hands over; the one naming the thread must survive reassembly intact and
    /// classify as an error. This is the CUDA failure that was invisible.
    #[test]
    fn a_panic_reassembles_into_an_error_line() {
        let lines = collect(&[
            b"\nthread 'DSD-0-0' panicked at crates/mummu/src/nn/mo",
            b"e.rs:118:\nCUDA_ERROR_UNKNOWN\nnote: run with `RUST_BACKTRACE=1`\n",
        ]);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("panicked at crates/mummu/src/nn/moe.rs:118:"));
        assert_eq!(classify(&lines[0]), Level::Error);
    }

    // -- request-line helpers ---------------------------------------------

    #[test]
    fn the_pages_own_polling_is_not_recorded_at_all() {
        assert!(
            ignored(Source::Api, "/api/logs"),
            "recording the feed's own poll would evict the feed"
        );
        assert!(
            !ignored(Source::Api, "/api/health"),
            "health is recorded — quiet, not ignored"
        );
        assert!(
            !ignored(Source::Shim, "/api/logs"),
            "the shim has no such route; nothing to suppress"
        );
    }

    /// The rule, as a table: a SUCCESSFUL, idempotent read of a page or a
    /// listing is quiet; any POST, the WebSocket upgrade and every non-2xx/3xx
    /// answer is loud. The first rows are the traffic measured on the live
    /// server six hours after v0.3.0 shipped; the 404 rows are the scanners
    /// that were probing the public shim in the same window.
    #[test]
    fn a_successful_read_of_a_page_or_listing_is_quiet_and_nothing_else_is() {
        use Source::{Api, Server, Shim};
        let (get, head, post) = (Method::GET, Method::HEAD, Method::POST);
        let delete = Method::DELETE;
        #[rustfmt::skip]
        let table: &[(Source, &Method, &str, u16, bool, &str)] = &[
            // What filled v0.3.0's ring, one row per poller.
            (Api,  &get,  "/api/health",   200, true,  "Docker's 30 s health probe"),
            (Shim, &get,  "/",             200, true,  "an ollama client polling the shim"),
            (Api,  &get,  "/",             200, true,  "a Glance monitor, once a minute — was loud"),
            (Api,  &get,  "/api/models",   200, true,  "a Kestra flow, once a minute — was loud"),
            // The rest of the pages and listings, and HEAD for any of them.
            (Api,  &head, "/",             200, true,  "HEAD is the same read"),
            (Api,  &get,  "/index.html",   200, true,  "the same page as /"),
            (Api,  &get,  "/logs",         200, true,  "the logs page"),
            (Api,  &head, "/logs",         200, true,  "HEAD of the logs page"),
            (Api,  &head, "/api/models",   200, true,  "HEAD of the listing"),
            (Api,  &get,  "/favicon.ico",  204, true,  "the icon a tab or a dashboard asks for"),
            (Api,  &get,  "/",             304, true,  "a 3xx is still a success"),
            (Shim, &get,  "/api/version",  200, true,  "ollama client"),
            (Shim, &get,  "/api/tags",     200, true,  "ollama model picker"),
            (Shim, &get,  "/api/ps",       200, true,  "ollama model picker"),
            (Shim, &head, "/",             200, true,  "HEAD of the shim's root"),
            // Anything that does work is loud, however routine.
            (Api,  &post, "/api/chat",     200, false, "a generation"),
            (Api,  &get,  "/api/chat/ws",  101, false, "the upgrade that starts a generation"),
            (Api,  &post, "/api/pull",     200, false, "a download"),
            (Api,  &post, "/api/unload",   200, false, "frees the model"),
            (Shim, &post, "/",             200, false, "the POST / seen on the public shim"),
            (Shim, &post, "/api/chat",     200, false, "a generation"),
            (Shim, &post, "/api/generate", 200, false, "a generation"),
            (Shim, &post, "/api/show",     200, false, "a POST, even one that only reads"),
            (Shim, &delete, "/api/delete", 200, false, "deletes a model"),
            (Api,  &post, "/api/models",   404, false, "the path alone does not make a poll"),
            // Every failure is loud, whatever the path.
            (Shim, &get,  "/.env",         404, false, "a scanner"),
            (Shim, &get,  "/v1/.env",      404, false, "a scanner"),
            (Shim, &get,  "/.git/config",  404, false, "a scanner"),
            (Shim, &get,  "/.env.backup",  404, false, "a scanner"),
            (Api,  &get,  "/favicon.ico",  404, false, "what v0.3.0 answered — a 404 is a 404"),
            (Api,  &get,  "/api/models",   500, false, "a listing that starts failing"),
            (Api,  &get,  "/api/health",   503, false, "the rule binds the probe too"),
            (Shim, &get,  "/api/tags",     400, false, "a 4xx on a polled path"),
            // A read that is not a polled page or listing stays loud.
            (Api,  &get,  "/api/profile",  200, false, "a flame graph someone asked for"),
            (Api,  &get,  "/api/tags",     404, false, "a shim path on the API listener"),
            (Shim, &get,  "/logs",         404, false, "the shim has no logs page"),
            (Server, &get, "/",            200, false, "captured output is never a request"),
        ];
        for (source, method, path, status, want, why) in table {
            assert_eq!(
                quiet_request(*source, method, path, *status),
                *want,
                "{} {method} {path} {status} ({why})",
                source.as_str()
            );
            // The arrival line is recorded loud unconditionally; that is only
            // right while no request that announces itself can ever be quiet.
            if announces_start(method, path) {
                assert!(
                    !quiet_request(*source, method, path, 200),
                    "{method} {path} announces itself, so it must be loud"
                );
            }
        }
    }

    /// The status code is not a guess, and `classify` would miss every one of
    /// these: "POST /api/chat 409 2 ms" contains none of its words.
    #[test]
    fn a_requests_level_comes_from_its_status_not_the_text() {
        assert_eq!(level_for_status(200), Level::Info);
        assert_eq!(level_for_status(101), Level::Info);
        assert_eq!(level_for_status(404), Level::Warn);
        assert_eq!(level_for_status(409), Level::Warn);
        assert_eq!(level_for_status(500), Level::Error);
        assert_eq!(
            classify("POST /api/chat 409 2 ms model=qwen3-4b"),
            Level::Info,
            "which is exactly why request entries do not use the heuristic"
        );
    }

    #[test]
    fn work_starting_requests_announce_themselves() {
        assert!(announces_start(&Method::POST, "/api/chat"));
        assert!(
            announces_start(&Method::GET, "/api/chat/ws"),
            "the upgrade the chat UI uses is where the two minutes happen"
        );
        assert!(!announces_start(&Method::GET, "/api/models"));
    }

    #[test]
    fn client_controlled_text_cannot_wreck_a_line() {
        assert_eq!(sanitize("qwen3-0.6b"), "qwen3-0.6b");
        assert_eq!(
            sanitize("a\nb\tc"),
            "a·b·c",
            "no entry can contain a newline"
        );
        let long = sanitize(&"m".repeat(MAX_FIELD * 2));
        assert_eq!(
            long.chars().count(),
            MAX_FIELD + 1,
            "bounded, with an ellipsis to say so"
        );
    }
}
