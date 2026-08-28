//! Synchronous-dataflow model of the decode step (SPEC 2, algorithm B) —
//! a priced dependency graph and the period floor T* it implies.
//!
//! One decode step is one synchronous pass (`CausalLm::forward` per token,
//! driven by `generate_loop`): a fixed chain of layer computations, machine
//! crossings where placement changes, a readback, and then the same chain
//! again for the next token. That is a synchronous dataflow graph: nodes do
//! fixed work, edges carry fixed crossing costs, and each edge holds a fixed
//! number of *tokens in flight* (its `delay`) — 0 for an intra-token
//! dependency, 1 for the token-loop back edge (token t+1's input depends on
//! token t's output).
//!
//! For such a graph the minimum achievable steady-state period is a
//! classical result: no schedule — however deeply pipelined or overlapped —
//! can beat
//!
//! ```text
//! T* = max over cycles C of ( Σ node work + Σ edge cost on C ) / ( Σ delay on C )
//! ```
//!
//! the **maximum cycle ratio**. Every cycle is a feedback loop that must be
//! traversed once per `Σ delay` tokens, so its cost divided by its delay
//! lower-bounds the per-token period; the binding cycle is the max. This is
//! what makes T* useful as a *floor*: measured s/token at or near T* means
//! the orchestration is done and only the costs themselves (kernels,
//! crossings, fences) can be attacked; measured far above T* means the
//! *schedule* is leaving time on the table, and [`regression_gate`] turns
//! that into a pass/fail check.
//!
//! # Why parametric binary search, not Karp
//!
//! Karp's algorithm computes the maximum cycle **mean** — cost per *edge* —
//! and needs an O(V·E) dynamic-programming table indexed by path length.
//! Here cycles are weighted by their **delay sum**, which is heterogeneous
//! (0 on intra-token edges, ≥1 on back edges), so the mean is the wrong
//! quantity, and adapting Karp means expanding delays into unit-delay chains
//! plus keeping the table. Instead this module uses the parametric
//! (Lawler-style) method: T* is the unique threshold of the monotone
//! predicate
//!
//! ```text
//! P(λ) = "some cycle has positive total (cost − λ·delay)"
//! ```
//!
//! — true for every λ < T*, false for every λ > T* — so a binary search over
//! λ with a Bellman-Ford positive-cycle test at each probe converges
//! geometrically, needs no table, handles delay-0 edges natively, and its
//! only numeric operation is a comparison (robust: a monotone predicate
//! cannot oscillate the way an accumulated DP can).
//!
//! # Ill-formed graphs
//!
//! A cycle whose total delay is 0 but whose cost is positive is a
//! contradiction in the model: it says "this much work must complete within
//! one token with no token in flight to amortize it" — its ratio is
//! infinite, P(λ) is true for every λ, and the search cannot terminate.
//! [`max_cycle_ratio`] detects this up front (positive-cycle test restricted
//! to delay-0 edges) and returns `None`; a real decode graph always has
//! delay ≥ 1 on its feedback edges, so `None` means the *graph construction*
//! is wrong, not the schedule.

/// Where a layer's compute runs. The decode graph only needs to know when
/// two consecutive layers *differ* (that boundary costs a crossing) and
/// whether a layer pays a launch (GPU dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Machine {
    Host,
    Gpu,
}

/// One computation in the decode step. `work` is milliseconds on the
/// critical path — kernel time plus any per-node overhead the caller folds
/// in (launch, encode).
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub work: f64,
}

/// A dependency: `to` cannot start (for a given token) until `from` finished
/// it `delay` tokens earlier. `cost` is milliseconds paid on the edge itself
/// — a crossing, a fence — and `delay` is tokens in flight on the
/// dependency: 0 for same-token edges, 1 for the token-loop back edge.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub cost: f64,
    pub delay: u32,
}

/// A priced directed graph. Build with [`Graph::add_node`] /
/// [`Graph::add_edge`], or via [`decode_graph`] for the standard per-token
/// chain.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    /// Add a node; returns its index for wiring edges.
    pub fn add_node(&mut self, name: impl Into<String>, work: f64) -> usize {
        self.nodes.push(Node {
            name: name.into(),
            work,
        });
        self.nodes.len() - 1
    }

    /// Add a directed edge `from → to` with crossing cost (ms) and delay
    /// (tokens in flight). Indices are validated by [`max_cycle_ratio`], not
    /// here, so graph construction stays infallible.
    pub fn add_edge(&mut self, from: usize, to: usize, cost: f64, delay: u32) {
        self.edges.push(Edge {
            from,
            to,
            cost,
            delay,
        });
    }
}

/// Longest-walk Bellman-Ford positive-cycle test: does the arc set contain a
/// cycle of strictly positive total weight?
///
/// All distances start at 0 — equivalent to a virtual source with a 0-weight
/// arc to every node, which makes the test independent of reachability and
/// of disconnected components. If relaxation still improves anything after
/// `n` full passes, some improving walk has ≥ `n` edges, hence repeats a
/// node, hence contains a strictly positive cycle; conversely with no
/// positive cycle the longest improving walk is simple (≤ `n−1` edges) and
/// the passes quiesce. Strict `>` comparison: a zero-weight cycle (the
/// critical cycle probed exactly at λ = T*) must *not* register as positive.
fn has_positive_cycle(n_nodes: usize, arcs: &[(usize, usize, f64)]) -> bool {
    if n_nodes == 0 || arcs.is_empty() {
        return false;
    }
    let mut dist = vec![0.0f64; n_nodes];
    for _ in 0..n_nodes {
        let mut changed = false;
        for &(u, v, w) in arcs {
            let cand = dist[u] + w;
            if cand > dist[v] {
                dist[v] = cand;
                changed = true;
            }
        }
        if !changed {
            return false; // quiesced: every walk stopped improving
        }
    }
    true // still improving after n passes: a positive cycle feeds it
}

/// The maximum cycle ratio of `g`: the minimum achievable steady-state
/// period T* in milliseconds per token (see the module header).
///
/// Returns:
/// * `Some(t_star)` — converged to ~1e-12 relative (200 bisection steps
///   with an early exit; the predicate is monotone so the bracket only
///   shrinks). A graph with **no** positively-priced cycle — a DAG, or only
///   zero-cost cycles — returns `Some(0.0)`: nothing feeds back, so
///   pipelining is unbounded and no cycle constrains the period. Ratios
///   would-be negative (negative-cost cycles) are also floored at 0.0 — a
///   period cannot be negative.
/// * `None` — the graph is ill-formed: an edge references a missing node, a
///   cost/work is non-finite (NaN or ±inf poisons every comparison), or a
///   **zero-delay cycle has positive cost** (infinite ratio; see the module
///   header). `None` always means "fix the graph", never "the schedule is
///   slow".
#[must_use]
pub fn max_cycle_ratio(g: &Graph) -> Option<f64> {
    let n = g.nodes.len();
    if g.nodes.iter().any(|node| !node.work.is_finite()) {
        return None;
    }
    if g.edges
        .iter()
        .any(|e| e.from >= n || e.to >= n || !e.cost.is_finite())
    {
        return None;
    }

    // Fold node work onto outgoing arcs: on any cycle each node contributes
    // exactly one outgoing edge, so charging `work(from)` to every arc
    // prices each cycle at (Σ work + Σ cost) with no double count.
    let arcs: Vec<(usize, usize, f64, u32)> = g
        .edges
        .iter()
        .map(|e| (e.from, e.to, e.cost + g.nodes[e.from].work, e.delay))
        .collect();

    // Ill-formed check: a positively-priced cycle among delay-0 edges alone.
    let zero_delay: Vec<(usize, usize, f64)> = arcs
        .iter()
        .filter(|a| a.3 == 0)
        .map(|a| (a.0, a.1, a.2))
        .collect();
    if has_positive_cycle(n, &zero_delay) {
        return None;
    }

    let probe = |lambda: f64| -> bool {
        let adjusted: Vec<(usize, usize, f64)> = arcs
            .iter()
            .map(|a| (a.0, a.1, a.2 - lambda * f64::from(a.3)))
            .collect();
        has_positive_cycle(n, &adjusted)
    };

    // P(0) false: no cycle prices above zero — the floor is 0 (see doc).
    if !probe(0.0) {
        return Some(0.0);
    }

    // Upper bracket: any cycle's price ≤ Σ positive arc weights, and after
    // the screen above every priced cycle has delay ≥ 1, so its ratio is
    // below this sum; +1.0 makes P(hi) strictly false.
    let mut lo = 0.0f64;
    let mut hi = arcs.iter().map(|a| a.2.max(0.0)).sum::<f64>() + 1.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if probe(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-12 * hi.max(1.0) {
            break;
        }
    }
    Some(0.5 * (lo + hi))
}

/// The measured cost table a decode graph is priced from. All milliseconds,
/// all measured (never assumed — see `examples/orchestration-attribution.rs`
/// for the 27B's recorded numbers).
#[derive(Debug, Clone)]
pub struct LayerCosts {
    /// Kernel time per layer, in execution order. Length defines the layer
    /// count and must match the placement vector.
    pub per_layer_ms: Vec<f64>,
    /// One host↔device activation crossing (paid where consecutive layers'
    /// placement differs).
    pub crossing_ms: f64,
    /// Per-layer GPU dispatch overhead (queue submit + driver). Charged to
    /// GPU-placed layers only: a host layer is a function call, there is
    /// nothing to launch.
    pub launch_ms: f64,
    /// The end-of-token readback/fence (the per-token GPU sync — `argmax_id`
    /// in the decode loop, ~8.5 ms/group measured on the 27B). Set 0.0 for
    /// an all-host graph with nothing to read back.
    pub readback_ms: f64,
}

/// Build the standard decode-step graph: the per-token layer chain with
/// crossing costs at placement boundaries, a final readback edge into a
/// zero-work `sample` node (the host-side argmax), and the token-loop back
/// edge (`sample → layer 0`, delay 1) closing the cycle.
///
/// The next token's input token id is a few bytes; its upload is treated as
/// part of layer 0's work, so the back edge itself is free — the delay is
/// what matters.
///
/// For this shape there is exactly one cycle, so T* of the result is the
/// analytic sum: Σ work + Σ boundary crossings + readback (pinned by test).
/// The graph type is more general on purpose — fork/join (fission) shapes
/// built by hand get max-of-branches from the same [`max_cycle_ratio`].
///
/// # Errors
///
/// When `placement.len() != per_layer_ms.len()` (a mispriced graph would
/// produce a confidently wrong floor), when the model is empty, or when any
/// cost is negative or non-finite.
pub fn decode_graph(costs: &LayerCosts, placement: &[Machine]) -> Result<Graph, String> {
    let n = costs.per_layer_ms.len();
    if n == 0 {
        return Err("decode_graph: no layers (per_layer_ms is empty)".into());
    }
    if placement.len() != n {
        return Err(format!(
            "decode_graph: placement has {} entries but per_layer_ms has {n} — every layer \
             needs a machine",
            placement.len()
        ));
    }
    let bad = |name: &str, v: f64| -> Result<(), String> {
        if !v.is_finite() || v < 0.0 {
            Err(format!("decode_graph: {name} = {v} must be finite and >= 0"))
        } else {
            Ok(())
        }
    };
    for (i, &ms) in costs.per_layer_ms.iter().enumerate() {
        bad(&format!("per_layer_ms[{i}]"), ms)?;
    }
    bad("crossing_ms", costs.crossing_ms)?;
    bad("launch_ms", costs.launch_ms)?;
    bad("readback_ms", costs.readback_ms)?;

    let mut g = Graph::default();
    for (i, (&ms, &m)) in costs.per_layer_ms.iter().zip(placement).enumerate() {
        let launch = if m == Machine::Gpu {
            costs.launch_ms
        } else {
            0.0
        };
        g.add_node(format!("layer{i}"), ms + launch);
    }
    for i in 0..n - 1 {
        let cost = if placement[i] == placement[i + 1] {
            0.0
        } else {
            costs.crossing_ms
        };
        g.add_edge(i, i + 1, cost, 0);
    }
    let sample = g.add_node("sample", 0.0);
    g.add_edge(n - 1, sample, costs.readback_ms, 0);
    g.add_edge(sample, 0, 0.0, 1); // the token loop: next token needs this one
    Ok(g)
}

/// The floor as a regression gate: measured steady-state token time must be
/// within 5% of T*. 5% is headroom for this box's session-to-session drift
/// (~10% cold, far less warm+quiet+same-session, which is the only regime
/// decode numbers are quoted from) without letting a real scheduling
/// regression — which shows up as tens of percent — hide inside it.
///
/// # Errors
///
/// A message with the measured time, the floor, the limit, and the overshoot
/// — everything needed to file the regression without re-deriving it. Also
/// errs on non-finite or negative inputs and on a non-positive floor (a gate
/// against T* = 0 would fail every real measurement; that is a graph
/// problem, not a regression).
pub fn regression_gate(measured_ms: f64, t_star_ms: f64) -> Result<(), String> {
    if !measured_ms.is_finite() || measured_ms < 0.0 {
        return Err(format!(
            "regression gate: measured token time {measured_ms} ms is not a valid measurement"
        ));
    }
    if !t_star_ms.is_finite() || t_star_ms <= 0.0 {
        return Err(format!(
            "regression gate: T* = {t_star_ms} ms is not a usable floor — the decode graph \
             (or its cost table) is wrong, fix that before gating"
        ));
    }
    let limit = 1.05 * t_star_ms;
    if measured_ms <= limit {
        Ok(())
    } else {
        Err(format!(
            "decode regression: measured {measured_ms:.3} ms/token exceeds the SDF floor \
             T* = {t_star_ms:.3} ms by {:.1}% (limit 1.05x = {limit:.3} ms) — the schedule \
             is leaving {:.3} ms/token on the table",
            100.0 * (measured_ms / t_star_ms - 1.0),
            measured_ms - t_star_ms,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rel(got: f64, want: f64, rel: f64, what: &str) {
        let tol = rel * want.abs().max(1e-12);
        assert!(
            (got - want).abs() <= tol,
            "{what}: got {got}, want {want} (tol {tol})"
        );
    }

    /// One node, one self-loop: ratio = (work + cost) / delay, computable in
    /// your head — (3 + 1) / 2 = 2.
    #[test]
    fn single_cycle_ratio_is_cost_over_delay() {
        let mut g = Graph::default();
        let a = g.add_node("a", 3.0);
        g.add_edge(a, a, 1.0, 2);
        let t = max_cycle_ratio(&g).expect("well-formed");
        assert_rel(t, 2.0, 1e-9, "self-loop ratio");
    }

    /// Two cycles through a shared node; the *second* one is binding, so the
    /// max (not the first found, not the sum) must come back:
    /// A→B→A prices (1+2+1)/1 = 4, A→C→A prices (1+10)/2 = 5.5.
    #[test]
    fn two_cycles_take_the_maximum() {
        let mut g = Graph::default();
        let a = g.add_node("a", 1.0);
        let b = g.add_node("b", 2.0);
        let c = g.add_node("c", 10.0);
        g.add_edge(a, b, 0.0, 0);
        g.add_edge(b, a, 1.0, 1);
        g.add_edge(a, c, 0.0, 0);
        g.add_edge(c, a, 0.0, 2);
        let t = max_cycle_ratio(&g).expect("well-formed");
        assert_rel(t, 5.5, 1e-9, "binding cycle");
    }

    /// The fission shape: fork → {A ∥ B} → join, token back edge. Each
    /// branch closes its own cycle, so T* is common + the LONGER branch —
    /// max, not sum. This is exactly why the SDF floor rewards overlap.
    #[test]
    fn parallel_branches_bind_at_the_longer_branch_not_the_sum() {
        let mut g = Graph::default();
        let fork = g.add_node("fork", 2.0);
        let a = g.add_node("branch_a", 10.0);
        let b = g.add_node("branch_b", 3.0);
        let join = g.add_node("join", 1.0);
        g.add_edge(fork, a, 0.0, 0);
        g.add_edge(fork, b, 0.0, 0);
        g.add_edge(a, join, 0.0, 0);
        g.add_edge(b, join, 0.0, 0);
        g.add_edge(join, fork, 0.5, 1);
        let t = max_cycle_ratio(&g).expect("well-formed");
        // 2 (fork) + 10 (longer branch) + 1 (join) + 0.5 (back edge).
        assert_rel(t, 13.5, 1e-9, "fork/join period");
        // Sanity: distinctly below the serial sum (which would add 3 more).
        assert!(t < 16.0, "T* {t} must not sum the branches");
    }

    /// The standard decode chain has one cycle, so T* must equal the
    /// analytic sum: Σ (layer + gpu launch) + boundary crossings + readback.
    #[test]
    fn decode_chain_period_equals_the_analytic_sum() {
        use Machine::{Gpu, Host};
        let costs = LayerCosts {
            per_layer_ms: vec![14.0, 14.0, 14.0, 36.0, 36.0],
            crossing_ms: 0.5,
            launch_ms: 0.05,
            readback_ms: 8.5,
        };
        let placement = [Gpu, Gpu, Gpu, Host, Host];
        let g = decode_graph(&costs, &placement).expect("valid table");
        let t = max_cycle_ratio(&g).expect("well-formed");
        // 3 GPU layers pay launch; one Gpu→Host boundary pays a crossing.
        let analytic = (14.0 + 0.05) * 3.0 + 36.0 * 2.0 + 0.5 + 8.5;
        assert_rel(t, analytic, 1e-9, "chain+backedge");
    }

    /// Bisection must converge to at least 1e-9 relative on a ratio that is
    /// not representable exactly (10/63), not just on friendly integers.
    #[test]
    fn binary_search_converges_to_1e9_relative() {
        let mut g = Graph::default();
        let a = g.add_node("a", 1.0 / 3.0);
        g.add_edge(a, a, 1.0 / 7.0, 3);
        let t = max_cycle_ratio(&g).expect("well-formed");
        let want = (1.0 / 3.0 + 1.0 / 7.0) / 3.0; // 10/63
        assert_rel(t, want, 1e-9, "irrational-ish ratio");
    }

    /// A zero-delay cycle with positive cost has an infinite ratio — the
    /// model is self-contradictory and must be refused, not searched.
    #[test]
    fn zero_delay_positive_cycle_is_rejected() {
        let mut g = Graph::default();
        let a = g.add_node("a", 1.0);
        let b = g.add_node("b", 1.0);
        g.add_edge(a, b, 0.0, 0);
        g.add_edge(b, a, 0.0, 0); // cycle: 2 ms of work, 0 tokens in flight
        assert!(max_cycle_ratio(&g).is_none());
    }

    /// No feedback, no floor: a DAG (and the empty graph) reports 0.0 —
    /// nothing cyclic constrains the period.
    #[test]
    fn acyclic_graph_has_zero_floor() {
        let mut g = Graph::default();
        let a = g.add_node("a", 5.0);
        let b = g.add_node("b", 7.0);
        g.add_edge(a, b, 1.0, 0);
        assert_eq!(max_cycle_ratio(&g), Some(0.0));
        assert_eq!(max_cycle_ratio(&Graph::default()), Some(0.0));
    }

    /// Garbage in must be `None`, never a number: dangling edges and NaN
    /// prices both poison the search invisibly if let through.
    #[test]
    fn invalid_graphs_are_refused() {
        let mut dangling = Graph::default();
        dangling.add_node("a", 1.0);
        dangling.add_edge(0, 5, 0.0, 1);
        assert!(max_cycle_ratio(&dangling).is_none());

        let mut nan = Graph::default();
        let a = nan.add_node("a", f64::NAN);
        nan.add_edge(a, a, 1.0, 1);
        assert!(max_cycle_ratio(&nan).is_none());
    }

    /// The graph builder refuses a mispriced model instead of producing a
    /// confidently wrong floor.
    #[test]
    fn decode_graph_validates_its_inputs() {
        let costs = LayerCosts {
            per_layer_ms: vec![1.0, 2.0],
            crossing_ms: 0.5,
            launch_ms: 0.0,
            readback_ms: 0.0,
        };
        let err = decode_graph(&costs, &[Machine::Host]).expect_err("length mismatch");
        assert!(err.contains("placement"), "{err}");

        let neg = LayerCosts {
            per_layer_ms: vec![1.0, -2.0],
            ..costs.clone()
        };
        assert!(decode_graph(&neg, &[Machine::Host, Machine::Host]).is_err());

        let empty = LayerCosts {
            per_layer_ms: vec![],
            ..costs
        };
        assert!(decode_graph(&empty, &[]).is_err());
    }

    /// The gate: within 5% passes, beyond fails with the numbers in the
    /// message, and a broken floor is its own loud error.
    #[test]
    fn regression_gate_passes_at_the_floor_and_fails_beyond_five_percent() {
        assert!(regression_gate(100.0, 100.0).is_ok());
        assert!(regression_gate(104.9, 100.0).is_ok());
        let err = regression_gate(120.0, 100.0).expect_err("20% over");
        assert!(err.contains("120.000") && err.contains("105.000"), "{err}");
        assert!(regression_gate(50.0, 0.0).is_err());
        assert!(regression_gate(f64::NAN, 100.0).is_err());
    }
}
