//! A scope-based wall-time profiler that renders flame graphs.
//!
//! Exists because this codebase spent a day mis-attributing a 27B's decode
//! time: three successive "the slow part is X" claims (device round trips,
//! small-op overhead, cluster dispatch) were each falsified by measurement,
//! while ~2.2 s of every 3.9 s token stayed unexplained. Sums of stage
//! timers answer "how much"; only a call-chain profile answers "where".
//!
//! Design:
//! * [`scope`] returns an RAII guard. Guards nest via a thread-local stack,
//!   so a scope's identity is its full path from the thread's root —
//!   `forward;layer;mlp.down` — which is exactly the folded-stack format
//!   flame-graph tooling consumes.
//! * Aggregation is by path: 64 layers all fold into one `forward;layer;…`
//!   bar, which is what you want — the question is "which *stage* costs",
//!   not "which layer".
//! * [`folded`] emits **self time** per path (inclusive minus direct
//!   children), so parent bars are never wider than their children's sum
//!   and the graph does not double-count.
//! * Off by default. When disabled, [`scope`] is one atomic load and no
//!   allocation; when enabled it costs one `String` join per scope —
//!   hundreds of nanoseconds against stages measured in milliseconds.
//!
//! Two hard rules for instrumenting with this:
//! * **Never hold a guard across an `.await`.** The stack is thread-local;
//!   a work-stealing runtime may resume the future — and drop the guard —
//!   on another thread, corrupting both threads' stacks. For a timed span
//!   that must cross an await, measure with `Instant` and call [`record`],
//!   which writes the aggregate directly without touching any stack.
//! * Worker threads start their own root. Name the root scope after the
//!   worker (`ffn_worker;<device>`), and read those bars as parallel wall
//!   time beside the main thread's — the roots of a flame graph are
//!   per-thread, so their widths can legitimately sum past 100%.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Path → (inclusive nanoseconds, times entered). `BTreeMap` for a
/// deterministic report; `const`-constructible, unlike `HashMap`.
static TOTALS: Mutex<BTreeMap<String, (u64, u64)>> = Mutex::new(BTreeMap::new());

thread_local! {
    static STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Turn collection on or off. Serving flips this per profiled request;
/// `MUMMU_PROFILE` in the environment forces it on for every request.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

#[must_use]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Drop everything collected so far — the start of a profiled request, so
/// one flame graph describes one generation rather than a process lifetime.
pub fn reset() {
    TOTALS.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// Enter a scope. The returned guard records on drop; hold it for exactly
/// the region being measured, and never across an `.await` (see module docs).
#[must_use]
pub fn scope(name: impl Into<String>) -> ScopeGuard {
    if !enabled() {
        return ScopeGuard { start: None };
    }
    STACK.with(|s| s.borrow_mut().push(name.into()));
    ScopeGuard {
        start: Some(Instant::now()),
    }
}

/// Record a span directly under `path` (semicolon-separated), bypassing the
/// thread-local stack. The async-safe escape hatch: measure with `Instant`
/// across the `.await`, then attribute it here.
pub fn record(path: &str, elapsed: Duration) {
    if !enabled() {
        return;
    }
    let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let entry = totals.entry(path.to_string()).or_insert((0, 0));
    entry.0 += elapsed.as_nanos() as u64;
    entry.1 += 1;
}

pub struct ScopeGuard {
    /// `None` when profiling was off at creation — then the guard also never
    /// pushed, so it must not pop. Enabled-state changes mid-scope are
    /// handled by trusting the guard's own record, not the global flag.
    start: Option<Instant>,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let elapsed = start.elapsed().as_nanos() as u64;
        STACK.with(|s| {
            let mut stack = s.borrow_mut();
            let path = stack.join(";");
            stack.pop();
            let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
            let entry = totals.entry(path).or_insert((0, 0));
            entry.0 += elapsed;
            entry.1 += 1;
        });
    }
}

/// The collected profile in folded-stack form: one `path count` line per
/// path, where the count is **self time in microseconds** — inclusive time
/// minus the inclusive time of direct children. Standard input for any
/// flame-graph renderer, and readable enough to eyeball sorted.
#[must_use]
pub fn folded() -> String {
    let totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    // Sum each path's direct children so self time can be derived. A direct
    // child of `a;b` is `a;b;c` — one more segment, no deeper.
    let mut child_sum: BTreeMap<&str, u64> = BTreeMap::new();
    for (path, &(ns, _)) in totals.iter() {
        if let Some(cut) = path.rfind(';') {
            *child_sum.entry(&path[..cut]).or_insert(0) += ns;
        }
    }
    let mut out = String::new();
    for (path, &(ns, count)) in totals.iter() {
        let children = child_sum.get(path.as_str()).copied().unwrap_or(0);
        // A parent can measure marginally less than its children (clock
        // granularity); clamp rather than emit negative bars.
        let self_us = ns.saturating_sub(children) / 1_000;
        if self_us == 0 {
            continue;
        }
        // The entry count rides along as a suffix on the leaf name so the
        // rendered frame reads `mlp.down (64x)` — dispatch-count questions
        // ("is this called once or 2048 times?") answer themselves.
        out.push_str(path);
        out.push_str(&format!(" ({count}x) {self_us}\n"));
    }
    out
}

/// Render folded lines to a self-contained flame-graph SVG.
#[cfg(feature = "flamegraph")]
pub fn flamegraph_svg(folded: &str) -> Result<String, String> {
    let mut opts = inferno::flamegraph::Options::default();
    opts.title = "mummu decode".to_string();
    opts.count_name = "µs".to_string();
    let mut svg = Vec::new();
    inferno::flamegraph::from_lines(&mut opts, folded.lines(), &mut svg)
        .map_err(|e| format!("flamegraph render: {e}"))?;
    String::from_utf8(svg).map_err(|e| format!("flamegraph svg not utf-8: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize: every test toggles the one global profiler.
    static GATE: Mutex<()> = Mutex::new(());

    fn run_isolated(f: impl FnOnce()) {
        let _g = GATE.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        set_enabled(true);
        f();
        set_enabled(false);
        reset();
    }

    /// Nested guards must produce nested paths, and the folded output must
    /// carry SELF time: the parent's bar shrinks by exactly its child's
    /// share, or the flame graph double-counts every nanosecond.
    #[test]
    fn nested_scopes_fold_into_self_time() {
        run_isolated(|| {
            {
                let _outer = scope("outer");
                std::thread::sleep(Duration::from_millis(20));
                {
                    let _inner = scope("inner");
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
            let folded = folded();
            let get = |needle: &str| -> u64 {
                folded
                    .lines()
                    .find(|l| l.starts_with(needle))
                    .and_then(|l| l.rsplit(' ').next())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| panic!("missing {needle:?} in {folded:?}"))
            };
            let outer = get("outer (");
            let inner = get("outer;inner (");
            assert!(inner >= 15_000, "inner self ~20ms, got {inner}µs");
            assert!(
                (15_000..=45_000).contains(&outer),
                "outer self must exclude the child: got {outer}µs"
            );
        });
    }

    /// Disabled must mean disabled: no entries, and — the part that broke a
    /// naive design — a guard created while off must not pop a stack it
    /// never pushed, even if profiling turns on before it drops.
    #[test]
    fn a_guard_created_while_off_never_touches_the_stack() {
        run_isolated(|| {
            set_enabled(false);
            let dead = scope("phantom");
            set_enabled(true);
            let _live = scope("real");
            drop(dead); // must NOT pop "real"
            drop(_live);
            let folded = folded();
            assert!(!folded.contains("phantom"), "{folded:?}");
            assert!(folded.lines().count() <= 1, "only 'real': {folded:?}");
        });
    }

    /// `record` attributes across threads and awaits without a stack; its
    /// counts accumulate.
    #[test]
    fn record_is_stackless_and_accumulates() {
        run_isolated(|| {
            record("token;readback", Duration::from_micros(500));
            record("token;readback", Duration::from_micros(500));
            let folded = folded();
            assert!(
                folded.contains("token;readback (2x) 1000"),
                "{folded:?}"
            );
        });
    }

    /// Repeated siblings aggregate into one line — 64 layers, one bar.
    #[test]
    fn repeated_scopes_aggregate() {
        run_isolated(|| {
            for _ in 0..64 {
                let _l = scope("layer");
                std::thread::sleep(Duration::from_micros(300));
            }
            let folded = folded();
            let line = folded
                .lines()
                .find(|l| l.starts_with("layer ("))
                .expect("layer line");
            assert!(line.contains("(64x)"), "{line}");
        });
    }
}
