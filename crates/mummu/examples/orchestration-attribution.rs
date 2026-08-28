//! Where does the token go? — the SDF floor and a Shapley attribution table.
//!
//! Two questions this prints answers to, from a cost table alone (no GPU,
//! no model — the modeling machinery is framework-independent):
//!
//! 1. **What is the best this placement can ever do?** Build the decode
//!    step's synchronous-dataflow graph from per-piece measured costs and
//!    compute T*, the maximum-cycle-ratio period floor (`mummu::sdf`). If a
//!    measured token time is supplied, print the gap and run the 1.05x
//!    regression gate: a small gap means "attack the costs", a large gap
//!    means "attack the schedule".
//! 2. **Who owns the orchestration overhead?** Demonstrate the Shapley
//!    harness (`mummu::attrib`) on the four-component synthetic
//!    decomposition — dispatch / fences / crossings / launches — priced FROM
//!    the same cost table, with an interaction term and measurement noise so
//!    the confidence-interval machinery has something to do. This
//!    establishes the attribution table format for the day the toggles are
//!    real measurements instead of a synthetic model.
//!
//! Usage:
//! ```text
//! orchestration-attribution [costs.json] [measured_ms_per_token]
//! ```
//! `costs.json` (optional):
//! ```json
//! { "per_layer_ms": [14.0, ...], "crossing_ms": 0.5, "launch_ms": 0.05,
//!   "readback_ms": 8.5, "placement": ["gpu", "gpu", ..., "host"] }
//! ```
//! With no file, a built-in table of the 27B's recorded session numbers is
//! used (see [`builtin_27b`]).

use mummu::attrib;
use mummu::sdf::{self, LayerCosts, Machine};

/// The 27B's recorded session numbers (quiet + warm + same-session; this box
/// drifts ~10% across sessions, so treat these as a coherent snapshot, not
/// gospel absolutes):
///
/// * 64 layers total (48 GDN + 16 attention);
/// * a 42-layer GPU-resident prefix at the measured ~14 ms/layer GPU FFN
///   cost;
/// * the remaining 22 layers on the host at the measured ~36 ms/layer
///   host-Q4 FFN cost;
/// * activation crossing 0.5 ms where placement changes;
/// * end-of-token readback/residual fence ~8.5 ms;
/// * launch 0.05 ms/dispatch — a *nominal* wgpu submit cost, not a recorded
///   number; replace via the JSON table once measured.
fn builtin_27b() -> (LayerCosts, Vec<Machine>) {
    const LAYERS: usize = 64;
    const GPU_PREFIX: usize = 42;
    let per_layer_ms: Vec<f64> = (0..LAYERS)
        .map(|i| if i < GPU_PREFIX { 14.0 } else { 36.0 })
        .collect();
    let placement: Vec<Machine> = (0..LAYERS)
        .map(|i| {
            if i < GPU_PREFIX {
                Machine::Gpu
            } else {
                Machine::Host
            }
        })
        .collect();
    (
        LayerCosts {
            per_layer_ms,
            crossing_ms: 0.5,
            launch_ms: 0.05,
            readback_ms: 8.5,
        },
        placement,
    )
}

/// Parse the JSON cost table. Hand-walked `serde_json::Value` rather than a
/// derive so a malformed table names its missing field instead of dumping a
/// serde type error.
fn load_costs(path: &str) -> Result<(LayerCosts, Vec<Machine>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))?;
    let num = |key: &str| -> Result<f64, String> {
        v.get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("{path}: missing or non-numeric field {key:?}"))
    };
    let per_layer_ms: Vec<f64> = v
        .get("per_layer_ms")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{path}: missing array field \"per_layer_ms\""))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| format!("{path}: non-numeric entry in \"per_layer_ms\""))
        })
        .collect::<Result<_, _>>()?;
    let placement: Vec<Machine> = v
        .get("placement")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{path}: missing array field \"placement\""))?
        .iter()
        .map(|x| match x.as_str() {
            Some(s) if s.eq_ignore_ascii_case("host") => Ok(Machine::Host),
            Some(s) if s.eq_ignore_ascii_case("gpu") => Ok(Machine::Gpu),
            other => Err(format!(
                "{path}: placement entries must be \"host\" or \"gpu\", got {other:?}"
            )),
        })
        .collect::<Result<_, _>>()?;
    Ok((
        LayerCosts {
            per_layer_ms,
            crossing_ms: num("crossing_ms")?,
            launch_ms: num("launch_ms")?,
            readback_ms: num("readback_ms")?,
        },
        placement,
    ))
}

/// Deterministic measurement noise for the synthetic demo (±2%), so the CI
/// columns are non-trivial and reproducible run to run.
struct Lcg(u64);
impl Lcg {
    fn centered(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (costs, placement, source) = match args.get(1) {
        Some(path) => {
            let (c, p) = load_costs(path)?;
            (c, p, path.clone())
        }
        None => {
            let (c, p) = builtin_27b();
            (c, p, "built-in 27B recorded session numbers".to_string())
        }
    };
    let measured_ms: Option<f64> = match args.get(2) {
        Some(s) => Some(
            s.parse::<f64>()
                .map_err(|_| format!("measured token time {s:?} is not a number (ms)"))?,
        ),
        None => None,
    };

    let gpu_layers = placement.iter().filter(|m| **m == Machine::Gpu).count();
    let host_layers = placement.len() - gpu_layers;
    let boundaries = placement.windows(2).filter(|w| w[0] != w[1]).count();

    println!("== SDF floor for the decode step ==");
    println!("cost table: {source}");
    println!(
        "layers: {} ({gpu_layers} gpu / {host_layers} host), placement boundaries: {boundaries}",
        placement.len()
    );

    let graph = sdf::decode_graph(&costs, &placement)?;
    let t_star = sdf::max_cycle_ratio(&graph)
        .ok_or_else(|| "decode graph is ill-formed (see sdf::max_cycle_ratio docs)".to_string())?;
    println!("T* = {t_star:.2} ms/token  (maximum cycle ratio of the priced decode graph)");

    if let Some(measured) = measured_ms {
        let gap = measured - t_star;
        println!(
            "measured = {measured:.2} ms/token   gap = {gap:+.2} ms ({:+.1}% vs floor)",
            100.0 * (measured / t_star - 1.0)
        );
        match sdf::regression_gate(measured, t_star) {
            Ok(()) => println!("gate(1.05x): PASS — the schedule is at its floor; attack costs"),
            Err(msg) => println!("gate(1.05x): FAIL — {msg}"),
        }
    } else {
        println!("(pass a measured ms/token as argv[2] to see the gap and the 1.05x gate)");
    }

    // ---- Shapley demo: the four-component orchestration decomposition ----
    //
    // Components priced FROM the cost table. `dispatch` has no measured
    // entry in the table, so it carries a documented nominal 0.02 ms/layer
    // of CPU-side encode; everything else is table-derived. The
    // fences×launches interaction (+15% of the launch total when both are
    // enabled — fences flush the queue under the launches) is what makes
    // this a demo of *Shapley* rather than of subtraction: the interaction
    // belongs to no single toggle, and the Shapley split shares it fairly.
    let components = [
        attrib::Component::new("dispatch"),
        attrib::Component::new("fences"),
        attrib::Component::new("crossings"),
        attrib::Component::new("launches"),
    ];
    let cost_of = [
        0.02 * placement.len() as f64,         // dispatch: nominal encode/layer
        costs.readback_ms,                     // fences: the end-of-token sync
        boundaries as f64 * costs.crossing_ms, // crossings: paid per boundary
        gpu_layers as f64 * costs.launch_ms,   // launches: paid per GPU dispatch
    ];
    const FENCES: u32 = 1 << 1;
    const LAUNCHES: u32 = 1 << 3;
    let mut rng = Lcg(0x6d75_6d6d_7532_3742); // fixed seed: reproducible table
    let mut measure = |mask: u32| -> f64 {
        let mut ms: f64 = cost_of
            .iter()
            .enumerate()
            .filter(|(j, _)| mask & (1 << j) != 0)
            .map(|(_, c)| c)
            .sum();
        if mask & FENCES != 0 && mask & LAUNCHES != 0 {
            ms += 0.15 * cost_of[3]; // the interaction term
        }
        ms * (1.0 + 0.04 * rng.centered()) // ±2% measurement noise
    };
    let replicates = 32;
    let a = attrib::attribute(components.len(), replicates, &mut measure);
    let total: f64 = a.phi.iter().sum();

    println!();
    println!("== Shapley attribution of the orchestration term ==");
    println!("(synthetic decomposition from the cost table; {replicates} replicates/subset)");
    println!();
    println!("| component | phi (ms/token) | 95% CI (± ms) | share |");
    println!("|-----------|---------------:|--------------:|------:|");
    for (c, (phi, ci)) in components.iter().zip(a.phi.iter().zip(&a.ci95)) {
        let share = if total.abs() > f64::EPSILON {
            100.0 * phi / total
        } else {
            0.0
        };
        println!("| {} | {phi:.3} | {ci:.3} | {share:.1}% |", c.name);
    }
    println!("| **total** | **{total:.3}** | | 100.0% |");
    println!();
    println!(
        "efficiency check: sum(phi) = {total:.3} ms attributes the whole enabled-minus-empty \
         overhead; interactions (fences x launches here) are shared, not dropped."
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("orchestration-attribution: {e}");
        std::process::exit(1);
    }
}
