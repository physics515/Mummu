//! Per-request visibility: where one request's time actually went.
//!
//! The server already had a flame-graph profiler (`/api/profile`), bandwidth
//! ledgers (`MUMMU_INSITU_REPORT`) and a log page. None of them answered the
//! question every production problem on 2026-09-20/21 started with — *what
//! happened to this request* — because `ChatResult` carried four fields
//! (text, token count, device, total ms) and threw the rest away before it
//! reached a surface.
//!
//! That cost real time. A proxy timeout, a cold-load stall and a slow decode
//! all look identical from a phone: an error after a while. Telling them
//! apart meant reading Caddy's access log, the container log and a
//! hand-run `curl -w` side by side, and one of those comparisons was wrong
//! because the test request had silently taken a different network path.
//!
//! So every request now leaves a [`RequestTrace`]: one structured log line,
//! and an entry in a ring buffer served at `GET /api/requests`, with
//! aggregates at `GET /api/stats`.

use std::collections::VecDeque;
use std::sync::Mutex;

use mummu_num::{f64_from_u64, f64_from_usize, trunc_usize};
use serde::Serialize;
use serde_json::{Value, json};

/// Phase durations the engine measured, in milliseconds.
///
/// Every field is filled by the code that owns that phase, and a phase that
/// did not happen is zero rather than absent — a warm request has
/// `load_ms: 0`, which is itself the fact worth seeing.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Timings {
    /// Reading GGUF/pack metadata to decide where the model goes.
    pub plan_ms: u64,
    /// Waiting for the model slot while another generation held it.
    pub queue_ms: u64,
    /// Loading weights. Zero when the model was already resident.
    pub load_ms: u64,
    /// Preprocessing and encoding images through the vision tower.
    pub vision_ms: u64,
    /// From the start of generation to the first token: the prompt's
    /// forward pass, image tokens included.
    pub prefill_ms: u64,
    /// From the first token to the last.
    pub decode_ms: u64,
    /// Tokens in the rendered prompt, image placeholders included.
    pub prompt_tokens: usize,
    /// How many of `prompt_tokens` were image tokens.
    pub image_tokens: usize,
    /// Tokens generated.
    pub completion_tokens: usize,
}

impl Timings {
    /// Decode throughput, the number that decides whether a reply finishes
    /// inside a client's timeout. `None` below two tokens, where the rate is
    /// not a rate.
    #[must_use]
    pub fn tokens_per_second(&self) -> Option<f64> {
        (self.completion_tokens > 1 && self.decode_ms > 0).then(|| {
            let n = f64_from_usize(self.completion_tokens - 1);
            n / (f64_from_u64(self.decode_ms) / 1000.0)
        })
    }

    /// Time from the engine seeing the request to the client seeing a token.
    #[must_use]
    pub const fn ttft_ms(&self) -> u64 {
        self.plan_ms + self.queue_ms + self.load_ms + self.vision_ms + self.prefill_ms
    }
}

/// What a request asked for, as its surface read it. Flattened into the
/// trace's JSON, so the wire shape is the three top-level fields it always
/// was.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Asked {
    /// The answer goes out as it is generated.
    pub stream: bool,
    /// The model's `<think>` block is shown rather than withheld.
    pub think: bool,
    /// The output is constrained to a JSON value.
    pub json_mode: bool,
}

/// Everything recorded about one request.
#[derive(Debug, Clone, Serialize)]
pub struct RequestTrace {
    /// Monotonic, so traces sort and a log line can be matched to its entry.
    pub seq: u64,
    /// RFC 3339 UTC.
    pub at: String,
    /// `openai`, `ollama` or `native`.
    pub surface: &'static str,
    pub model: String,
    #[serde(flatten)]
    pub asked: Asked,
    pub tools: usize,
    pub images: usize,
    pub device: String,
    pub timings: Timings,
    pub ttft_ms: u64,
    pub tokens_per_second: Option<f64>,
    pub total_ms: u64,
    /// A buffered answer that outlived the keep-alive grace period and was
    /// padded (see `crate::keepalive_json`). Its status went out as 200
    /// before the outcome was known.
    pub padded: bool,
    /// `ok`, or the error text the client received.
    pub outcome: String,
}

/// How many recent requests are kept for `/api/requests`.
const RING: usize = 256;

static TRACES: Mutex<VecDeque<RequestTrace>> = Mutex::new(VecDeque::new());
static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a surface knows about a request before the engine runs.
pub struct Begin {
    pub surface: &'static str,
    pub model: String,
    pub asked: Asked,
    pub tools: usize,
    pub images: usize,
    pub started: std::time::Instant,
}

impl Begin {
    /// Record the finished request: log one line and keep it in the ring.
    pub fn finish(self, device: &str, timings: Timings, padded: bool, outcome: Result<(), &str>) {
        let total_ms = crate::millis(self.started.elapsed());
        let trace = RequestTrace {
            seq: SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            at: now_rfc3339(),
            surface: self.surface,
            model: self.model,
            asked: self.asked,
            tools: self.tools,
            images: self.images,
            device: device.to_string(),
            ttft_ms: timings.ttft_ms(),
            tokens_per_second: timings.tokens_per_second(),
            timings,
            total_ms,
            padded,
            outcome: match outcome {
                Ok(()) => "ok".into(),
                Err(e) => e.to_string(),
            },
        };
        // One greppable line per request, whole: the log is the record when
        // the ring has rolled over.
        if let Ok(line) = serde_json::to_string(&trace) {
            eprintln!("[mummu-serve] trace {line}");
        }
        let mut ring = TRACES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ring.len() == RING {
            ring.pop_front();
        }
        ring.push_back(trace);
    }
}

fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // Civil-from-days (Howard Hinnant), to avoid a date dependency for one
    // timestamp.
    let days = (secs / 86_400).cast_signed();
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// `GET /api/requests` — the recent traces, newest first.
pub fn recent_json(limit: usize) -> Value {
    let ring = TRACES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    json!({"requests": ring.iter().rev().take(limit).collect::<Vec<&RequestTrace>>()})
}

/// The `p`-th percentile of `v` (nearest rank), or `None` when empty.
fn percentile(v: &mut [f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let i = trunc_usize(((p / 100.0) * f64_from_usize(v.len() - 1)).round());
    v.get(i).copied()
}

/// The aggregates over the traces in `ring` that `pred` picks.
fn summarize(ring: &VecDeque<RequestTrace>, pred: &dyn Fn(&RequestTrace) -> bool) -> Value {
    let picked: Vec<&RequestTrace> = ring.iter().filter(|t| pred(t)).collect();
    let mut ttft: Vec<f64> = picked.iter().map(|t| f64_from_u64(t.ttft_ms)).collect();
    let mut total: Vec<f64> = picked.iter().map(|t| f64_from_u64(t.total_ms)).collect();
    let mut tps: Vec<f64> = picked.iter().filter_map(|t| t.tokens_per_second).collect();
    json!({
        "count": picked.len(),
        "errors": picked.iter().filter(|t| t.outcome != "ok").count(),
        "padded": picked.iter().filter(|t| t.padded).count(),
        "ttft_ms": {"p50": percentile(&mut ttft.clone(), 50.0), "p95": percentile(&mut ttft, 95.0)},
        "total_ms": {"p50": percentile(&mut total.clone(), 50.0), "p95": percentile(&mut total, 95.0)},
        "tokens_per_second": {"p50": percentile(&mut tps.clone(), 50.0), "p95": percentile(&mut tps, 95.0)},
    })
}

/// `GET /api/stats` — aggregates over the ring, split warm/cold.
///
/// Warm and cold are reported separately because pooling them hides both:
/// one cold load is minutes, and averaged into a handful of warm requests
/// it turns a healthy p50 into a scary one and a real regression into
/// noise.
pub fn stats_json() -> Value {
    // Every aggregate is computed under the lock and the guard dropped at
    // the end of this block, before the response is assembled: serializing
    // JSON is not worth holding every tracer's writes behind.
    let (window, all, warm, cold, with_images, openai, ollama, native) = {
        let ring = TRACES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            ring.len(),
            summarize(&ring, &|_| true),
            summarize(&ring, &|t| t.timings.load_ms == 0),
            summarize(&ring, &|t| t.timings.load_ms > 0),
            summarize(&ring, &|t| t.images > 0),
            summarize(&ring, &|t| t.surface == "openai"),
            summarize(&ring, &|t| t.surface == "ollama"),
            summarize(&ring, &|t| t.surface == "native"),
        )
    };
    json!({
        "window": window,
        "all": all,
        "warm": warm,
        "cold": cold,
        "with_images": with_images,
        "by_surface": {
            "openai": openai,
            "ollama": ollama,
            "native": native,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttft_sums_every_phase_before_the_first_token() {
        let t = Timings {
            plan_ms: 10,
            queue_ms: 20,
            load_ms: 300,
            vision_ms: 40,
            prefill_ms: 50,
            decode_ms: 999,
            ..Timings::default()
        };
        assert_eq!(t.ttft_ms(), 420, "decode is after the first token");
    }

    #[test]
    fn throughput_needs_at_least_two_tokens() {
        let one = Timings {
            completion_tokens: 1,
            decode_ms: 100,
            ..Timings::default()
        };
        assert_eq!(one.tokens_per_second(), None);
        // 11 tokens: the first arrives at the end of prefill, so 10 intervals.
        let many = Timings {
            completion_tokens: 11,
            decode_ms: 5000,
            ..Timings::default()
        };
        assert_eq!(many.tokens_per_second(), Some(2.0));
    }

    #[test]
    fn percentile_is_nearest_rank() {
        let mut v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&mut v, 50.0), Some(3.0));
        assert_eq!(percentile(&mut v, 100.0), Some(5.0));
        assert_eq!(percentile(&mut [], 50.0), None);
    }

    #[test]
    fn timestamp_is_rfc3339() {
        let s = now_rfc3339();
        assert_eq!(s.len(), 20, "{s}");
        assert!(
            s.ends_with('Z') && &s[4..5] == "-" && &s[10..11] == "T",
            "{s}"
        );
    }
}
