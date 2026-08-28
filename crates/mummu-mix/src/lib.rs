//! **Scheduler B — how many bits the weights on a device are stored in.**
//!
//! Distinct from the `mummu-schedule` crate, which is scheduler A: that one
//! decides *how much* work each device gets, this one decides *how precisely*
//! the work it got is stored. Separate objectives — makespan there, accuracy
//! here — and they were conflated for a long time by a planner that picked
//! one precision for a whole model and filled devices in speed order.
//!
//! Dependency-free on purpose: choosing precisions is arithmetic over
//! parameter counts, byte costs and a measured error model. It needs no
//! tensor library, so it compiles and tests in a second and can be reasoned
//! about without a GPU. The one thing that genuinely needs burn — turning a
//! rung into a `QuantScheme` — stays in `mummu::quant`.

/// Which quantization the keep-quantized path applies on import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPolicy {
    /// No quantization — the classic f32 path.
    Off,
    /// Half precision. Not a quantization at all: no codes, no block scales,
    /// just a narrower float — so it has no [`Self::scheme`].
    ///
    /// It earns a rung because a **quantized source makes f32 redundant**.
    /// This 27B ships as Q4_K_S: 15.36 GB for ~27 G parameters is 4.55
    /// bits/param, so the f32 in a pack is an upcast copy of 4-bit data and
    /// carries no more information than f16 does. Storing it at 4 B/param
    /// buys nothing over 2 B/param — and f16 measured the same speed on the
    /// host (13.26 ms against 13.82 for f32, `device-throughput.rs`). Above
    /// f16 there is nothing to gain for such a checkpoint, which is also why
    /// there is no f64 rung: it would double the bytes above a ceiling the
    /// source already sits far below.
    F16,
    /// 8-bit symmetric, block-32 scales: the proven default (≈4x smaller
    /// than f32, ~0.04% matmul error on every backend).
    Q8,
    /// 4-bit symmetric, block-32 scales (≈8x smaller, ~0.8% matmul error).
    ///
    /// *(2026-08-23)* The long-standing "wgpu's Q4 kernel returns garbage"
    /// note is **withdrawn**. It came from a probe that quantized synthetic
    /// tensors *on the device*; re-measured along the path production
    /// actually takes — packed bytes read from a pack onto the device, then
    /// multiplied — Q4 on wgpu is correct and bit-identical across repeated
    /// rounds at this model's dimensions (relative error 0.062 against the
    /// pack's own f32 bytes, 2.9 ms warm). See
    /// `examples/pack-precision-probe.rs`.
    Q4,
    /// 2-bit symmetric, block-32 scales (≈16x smaller than f32). The bottom
    /// of the ladder that burn can hold *packed in VRAM*: cubecl has a
    /// `Q2S` value, and nothing below it (a 1-bit tier would need a custom
    /// kernel). Quality falls off sharply, so this is a pressure-relief
    /// tier — the point where staying resident on a fast device still beats
    /// spilling the tensor to the host, not a default.
    Q2,
}

impl QuantPolicy {
    /// Parse the `MUMMU_QUANT` convention: `q8` / `int8` → [`Self::Q8`],
    /// `off`/empty/unset → [`Self::Off`]. Unknown values are a loud error.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("MUMMU_QUANT") {
            Err(_) => Ok(Self::Off),
            Ok(v) if v.is_empty() || v.eq_ignore_ascii_case("off") => Ok(Self::Off),
            Ok(v) if v.eq_ignore_ascii_case("q8") || v.eq_ignore_ascii_case("int8") => Ok(Self::Q8),
            Ok(v) if v.eq_ignore_ascii_case("q4") || v.eq_ignore_ascii_case("int4") => Ok(Self::Q4),
            Ok(v) if v.eq_ignore_ascii_case("q2") || v.eq_ignore_ascii_case("int2") => Ok(Self::Q2),
            Ok(v) if v.eq_ignore_ascii_case("f16") || v.eq_ignore_ascii_case("fp16") => {
                Ok(Self::F16)
            }
            Ok(other) => Err(format!(
                "unknown MUMMU_QUANT {other:?} (expected f16, q8, q4, q2, or off)"
            )),
        }
    }

    /// Bits per stored weight — what the placement planner budgets with.
    /// `Off` is f32.
    #[must_use]
    pub fn bits(self) -> usize {
        match self {
            Self::Off => 32,
            Self::F16 => 16,
            Self::Q8 => 8,
            Self::Q4 => 4,
            Self::Q2 => 2,
        }
    }

    /// The ladder, most precise first. Placement walks *down* it to free
    /// bytes under pressure and back *up* as pressure eases.
    pub const LADDER: [Self; 5] = [Self::Off, Self::F16, Self::Q8, Self::Q4, Self::Q2];

    /// The next rung down, or `None` at the bottom.
    #[must_use]
    pub fn demote(self) -> Option<Self> {
        match self {
            Self::Off => Some(Self::F16),
            Self::F16 => Some(Self::Q8),
            Self::Q8 => Some(Self::Q4),
            Self::Q4 => Some(Self::Q2),
            Self::Q2 => None,
        }
    }

    /// The next rung up, or `None` at the top.
    #[must_use]
    pub fn promote(self) -> Option<Self> {
        match self {
            Self::Off => None,
            Self::F16 => Some(Self::Off),
            Self::Q8 => Some(Self::F16),
            Self::Q4 => Some(Self::Q8),
            Self::Q2 => Some(Self::Q4),
        }
    }

    /// The most precise rung worth storing for a source of
    /// `source_bits_per_param`.
    ///
    /// Precision above the source is bytes without information. A checkpoint
    /// that ships at 4.55 bits/param (this 27B's Q4_K_S) gains nothing from
    /// f32 over f16 — measured 0.0000 relative error for f16 against the
    /// pack's own f32 bytes, because those bytes are themselves an upcast of
    /// 4-bit data. This is also the answer to "why not f64": the ceiling is
    /// the source, and no LLM checkpoint ships anywhere near it.
    ///
    /// Deliberately not tighter than f16, even though the source is ~4.5
    /// bits: the quantized rungs below are **lossy relative to the source**,
    /// not equivalent to it. Our block-32 Q8 measured 0.58% relative error,
    /// so "the source is 4.5 bits, therefore Q4 suffices" does not follow — a
    /// K-quant's super-block scaling is better than ours at the same width.
    #[must_use]
    pub fn ceiling_for_source(source_bits_per_param: f64) -> Self {
        if source_bits_per_param > 16.0 {
            Self::Off
        } else {
            Self::F16
        }
    }

    /// Is a 2-D weight of `dims` worth quantizing? Small projections lose
    /// more accuracy than the bytes they return; 256×256 (65k elements) is
    /// the floor — everything matmul-heavy in a real LLM clears it. The
    /// last dim must also divide by the block width: burn's block scales
    /// run along rows and a non-divisible row (qwen35-27B's 48-wide β/α
    /// projections) crashes the block-range reshape.
    #[must_use]
    pub fn eligible(self, dims: &[usize]) -> bool {
        const BLOCK: usize = 32; // keep in sync with `scheme()`
        self != Self::Off
            && dims.len() == 2
            && dims.iter().product::<usize>() >= (1 << 16)
            && dims[1].is_multiple_of(BLOCK)
    }
}

/// What a tensor is, for the purpose of deciding how much precision it
/// deserves. Coarse on purpose — a finer split would need per-tensor
/// calibration data we do not have, and would imply a confidence this
/// heuristic has not earned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Q/K/V/O projections. More sensitive than FFN at equal size: they
    /// steer where information moves, and llama.cpp's K-quant mixes keep
    /// them a rung above the feed-forward weights for the same reason.
    Attention,
    /// Feed-forward / expert weights. Where nearly all the bytes are, and
    /// the most tolerant of quantization.
    Ffn,
    /// Token embedding, norm gammas, biases, conv kernels. Never quantized
    /// here — the embedding is a gather (no quantized kernel on the GPU
    /// backends) and the vectors are both tiny and load-bearing for the
    /// numerics.
    Fixed,
}

/// One tensor as the placement planner sees it.
#[derive(Debug, Clone)]
pub struct TensorFacts {
    /// Element count. Bytes follow from this and the precision.
    pub params: usize,
    pub kind: Kind,
    /// Layer index, if the tensor belongs to one. Drives the edge weighting
    /// below; `None` (a trunk tensor) is treated as an edge.
    pub layer: Option<usize>,
}

/// The relative matmul error each rung costs, measured on a real pack tensor
/// (`examples/ladder-probe.rs`, `blk.63.ffn_up.weight`, 89.1 M params,
/// against the pack's own f32 bytes).
///
/// The absolute values matter less than their *ratios*: Q4 costs ~17x Q8,
/// and Q2 ~6x Q4 again. That steepness is why the planner spends its budget
/// keeping tensors off Q2 rather than spreading the pain evenly.
#[must_use]
pub fn rel_error(p: QuantPolicy) -> f64 {
    match p {
        QuantPolicy::Off => 0.0,
        // Measured at 0.0000 — below the probe's resolution against the
        // pack's own f32 bytes, because those bytes are themselves an upcast
        // of a 4.55 bits/param source. Zero here is deliberate and load
        // bearing: it makes f32 -> f16 the first demotion the planner
        // reaches for, which is correct, since for such a checkpoint it
        // halves the bytes for no accuracy at all.
        QuantPolicy::F16 => 0.0,
        QuantPolicy::Q8 => 0.0058,
        QuantPolicy::Q4 => 0.0997,
        QuantPolicy::Q2 => 0.6226,
    }
}

/// Resident bytes for `params` at `precision`, block scales included.
///
/// The scales are not a rounding detail: at Q2 an f32 scale per 32 values is
/// 37% of the tensor, so a "16x smaller than f32" claim that ignores them is
/// off by more than a third.
#[must_use]
pub fn bytes_at(params: usize, precision: QuantPolicy) -> u64 {
    const BLOCK: usize = 32;
    let values = params * precision.bits() / 8;
    let scales = if precision == QuantPolicy::Off {
        0
    } else {
        params.div_ceil(BLOCK) * size_of::<f32>()
    };
    (values + scales) as u64
}

/// Position on [`QuantPolicy::LADDER`] — larger means less precise.
fn rung(p: QuantPolicy) -> usize {
    QuantPolicy::LADDER
        .iter()
        .position(|&q| q == p)
        .unwrap_or(0)
}

impl TensorFacts {
    /// How much this tensor's error matters, relative to a mid-stack FFN
    /// weight at 1.0. Multiplies the error delta when ranking demotions.
    ///
    /// Two effects, both standard practice in mixed-precision quantization
    /// and neither calibrated here: attention weighs more than FFN, and the
    /// first and last layers weigh more than the middle (they sit closest to
    /// the embedding and the logits, so their error is least attenuated).
    fn sensitivity(&self, layers: usize) -> f64 {
        let base = match self.kind {
            Kind::Fixed => return f64::INFINITY,
            Kind::Attention => 2.0,
            Kind::Ffn => 1.0,
        };
        let edge = match self.layer {
            // A trunk tensor sits outside the stack entirely; treat it as an
            // edge rather than silently giving it the mildest weighting.
            None => true,
            Some(l) => l == 0 || l + 1 >= layers.max(1),
        };
        if edge { base * 2.0 } else { base }
    }
}

/// A precision assignment for every tensor, and what it costs.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Parallel to the `tensors` slice handed to [`plan`].
    pub precision: Vec<QuantPolicy>,
    /// Total resident bytes.
    pub bytes: u64,
    /// True when even the floor precision everywhere does not fit — the
    /// caller must spill something to the host, and this plan is only the
    /// smallest the device side can be made.
    pub over_budget: bool,
}

impl Plan {
    /// How many tensors sit at each rung — what an operator wants in a log
    /// line, and what a test can assert on.
    #[must_use]
    pub fn histogram(&self) -> Vec<(QuantPolicy, usize)> {
        QuantPolicy::LADDER
            .iter()
            .map(|&p| (p, self.precision.iter().filter(|&&q| q == p).count()))
            .filter(|&(_, n)| n > 0)
            .collect()
    }
}

/// Fill `budget` bytes with the most precision it will hold.
///
/// Starts every eligible tensor at `ceiling` and walks the cheapest ones
/// down: at each step the tensor demoted is the one whose
/// `sensitivity x error-increase` per byte freed is smallest, so the budget
/// is spent where it buys the most accuracy. `floor` bounds how far any
/// tensor may fall — a caller that would rather spill to the host than run
/// 2-bit weights passes `Q4`.
///
/// Greedy, not optimal: this is a knapsack, and the exact answer is not
/// worth the time inside a rebalance that runs while a model is serving.
/// Greedy-by-ratio is the standard approximation, and its failure mode
/// (spending slightly too much on one large tensor) is bounded by the size
/// of a single demotion step.
#[must_use]
pub fn plan(
    tensors: &[TensorFacts],
    layers: usize,
    budget: u64,
    ceiling: QuantPolicy,
    floor: QuantPolicy,
) -> Plan {
    let mut precision: Vec<QuantPolicy> = tensors
        .iter()
        .map(|t| {
            if t.kind == Kind::Fixed {
                QuantPolicy::Off
            } else {
                ceiling
            }
        })
        .collect();
    let mut bytes: u64 = tensors
        .iter()
        .zip(&precision)
        .map(|(t, &p)| bytes_at(t.params, p))
        .sum();

    let floor_rung = rung(floor);

    // What one rung down would trade for this tensor: (cost per byte freed,
    // bytes freed, the rung). `None` when it cannot or should not move.
    let step = |i: usize, from: QuantPolicy| -> Option<(f64, u64, QuantPolicy)> {
        let to = from.demote()?;
        if rung(to) > floor_rung {
            return None;
        }
        let before = bytes_at(tensors[i].params, from);
        let after = bytes_at(tensors[i].params, to);
        if after >= before {
            return None; // no bytes freed, no reason to lose accuracy
        }
        let sensitivity = tensors[i].sensitivity(layers);
        if !sensitivity.is_finite() {
            return None; // Fixed: never quantized, at any pressure
        }
        let freed = before - after;
        let cost = sensitivity * (rel_error(to) - rel_error(from));
        Some((cost / freed as f64, freed, to))
    };

    while bytes > budget {
        let mut best: Option<(f64, usize, u64, QuantPolicy)> = None;
        for i in 0..tensors.len() {
            if let Some((ratio, freed, to)) = step(i, precision[i])
                && best.is_none_or(|(b, ..)| ratio < b)
            {
                best = Some((ratio, i, freed, to));
            }
        }
        match best {
            Some((_, i, freed, to)) => {
                precision[i] = to;
                bytes -= freed;
            }
            // Everything eligible is at the floor: this is as small as the
            // device side gets, and the caller has to spill.
            None => {
                return Plan {
                    precision,
                    bytes,
                    over_budget: true,
                };
            }
        }
    }

    Plan {
        precision,
        bytes,
        over_budget: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffn(params: usize, layer: usize) -> TensorFacts {
        TensorFacts {
            params,
            kind: Kind::Ffn,
            layer: Some(layer),
        }
    }
    fn attn(params: usize, layer: usize) -> TensorFacts {
        TensorFacts {
            params,
            kind: Kind::Attention,
            layer: Some(layer),
        }
    }

    /// A model that already fits keeps every tensor at the ceiling — the
    /// planner must not quantize for its own sake.
    #[test]
    fn a_model_that_fits_is_left_at_the_ceiling() {
        let ts: Vec<_> = (1..9).map(|l| ffn(1 << 20, l)).collect();
        let need: u64 = ts.iter().map(|t| bytes_at(t.params, QuantPolicy::Q8)).sum();
        let p = plan(&ts, 10, need, QuantPolicy::Q8, QuantPolicy::Q2);
        assert!(p.precision.iter().all(|&q| q == QuantPolicy::Q8));
        assert_eq!(p.bytes, need);
        assert!(!p.over_budget);
    }

    /// Under pressure, FFN gives way before attention of the same size —
    /// that is the whole reason for placing precision per tensor.
    #[test]
    fn ffn_is_demoted_before_attention_of_equal_size() {
        let ts = vec![ffn(1 << 22, 5), attn(1 << 22, 5)];
        let full: u64 = ts.iter().map(|t| bytes_at(t.params, QuantPolicy::Q8)).sum();
        // Shave enough that exactly one of the two must drop a rung.
        let budget = full - bytes_at(1 << 22, QuantPolicy::Q8) / 3;
        let p = plan(&ts, 10, budget, QuantPolicy::Q8, QuantPolicy::Q2);
        assert_eq!(
            p.precision[0],
            QuantPolicy::Q4,
            "FFN should absorb the pressure"
        );
        assert_eq!(
            p.precision[1],
            QuantPolicy::Q8,
            "attention should be spared"
        );
    }

    /// The edge layers are held at higher precision than the middle when
    /// only some tensors can keep it.
    #[test]
    fn middle_layers_are_demoted_before_the_edges() {
        let ts = vec![ffn(1 << 22, 0), ffn(1 << 22, 8), ffn(1 << 22, 15)];
        let full: u64 = ts.iter().map(|t| bytes_at(t.params, QuantPolicy::Q8)).sum();
        let budget = full - bytes_at(1 << 22, QuantPolicy::Q8) / 3;
        let p = plan(&ts, 16, budget, QuantPolicy::Q8, QuantPolicy::Q2);
        assert_eq!(
            p.precision[1],
            QuantPolicy::Q4,
            "the middle layer gives way first"
        );
        assert_eq!(p.precision[0], QuantPolicy::Q8);
        assert_eq!(p.precision[2], QuantPolicy::Q8);
    }

    /// Fixed tensors (embedding, norms) are never quantized, however tight
    /// the budget gets — and a budget only they could satisfy reports
    /// `over_budget` rather than quietly quantizing them.
    #[test]
    fn fixed_tensors_are_never_quantized_and_a_hopeless_budget_says_so() {
        let ts = vec![
            TensorFacts {
                params: 1 << 20,
                kind: Kind::Fixed,
                layer: None,
            },
            ffn(1 << 22, 4),
        ];
        let p = plan(&ts, 10, 1, QuantPolicy::Q8, QuantPolicy::Q2);
        assert_eq!(
            p.precision[0],
            QuantPolicy::Off,
            "embedding must stay float"
        );
        assert_eq!(
            p.precision[1],
            QuantPolicy::Q2,
            "everything else goes to the floor"
        );
        assert!(
            p.over_budget,
            "an unmeetable budget must be reported, not faked"
        );
    }

    /// The floor is respected: a caller that would rather spill than run
    /// 2-bit weights never gets Q2 back.
    #[test]
    fn the_floor_bounds_how_far_a_tensor_falls() {
        let ts: Vec<_> = (1..5).map(|l| ffn(1 << 22, l)).collect();
        let p = plan(&ts, 10, 1, QuantPolicy::Q8, QuantPolicy::Q4);
        assert!(p.precision.iter().all(|&q| q == QuantPolicy::Q4));
        assert!(p.over_budget);
    }

    /// Tightening the budget never *raises* anyone's precision: the plan has
    /// to move monotonically or the rebalancer will oscillate.
    #[test]
    fn plans_move_monotonically_with_the_budget() {
        let ts: Vec<_> = (0..24)
            .map(|l| {
                if l % 3 == 0 {
                    attn(1 << 21, l)
                } else {
                    ffn(1 << 22, l)
                }
            })
            .collect();
        let full: u64 = ts.iter().map(|t| bytes_at(t.params, QuantPolicy::Q8)).sum();
        let mut previous: Option<Plan> = None;
        for step in 1..12 {
            let p = plan(
                &ts,
                24,
                full * (12 - step) / 12,
                QuantPolicy::Q8,
                QuantPolicy::Q2,
            );
            if let Some(prev) = &previous {
                for (i, (&now, &before)) in p.precision.iter().zip(&prev.precision).enumerate() {
                    assert!(
                        rung(now) >= rung(before),
                        "tensor {i} rose from {before:?} to {now:?} as the budget shrank"
                    );
                }
            }
            previous = Some(p);
        }
    }

    /// Block scales are counted. At Q2 they are 37% of the tensor, so a
    /// planner that ignored them would promise far more headroom than it
    /// delivers.
    #[test]
    fn scales_are_counted_in_the_byte_estimate() {
        let params = 1 << 20;
        let q2 = bytes_at(params, QuantPolicy::Q2);
        let values_only = (params * 2 / 8) as u64;
        assert!(q2 > values_only, "scales must be in the total");
        assert_eq!(q2, values_only + (params / 32 * 4) as u64);
        // ...and f32 carries none.
        assert_eq!(bytes_at(params, QuantPolicy::Off), (params * 4) as u64);
    }
}
