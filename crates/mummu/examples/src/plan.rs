//! **Precision selection** — the first piece of the P6 hardware planner:
//! given a model's shape and one adapter's real capabilities, which float
//! precision is the *highest* one that still fits?
//!
//! Deliberately narrow. This is arithmetic over numbers the rest of the crate
//! already produces ([`crate::backend::inventory`] for VRAM and `SHADER_F16`,
//! a checkpoint's `config.json` for parameter count and cache geometry), and
//! it decides exactly one thing: `Gpu` (f32) or `GpuF16`. Layer placement,
//! multi-GPU sharding and CPU spill are separate ROADMAP items; quantized
//! tiers arrive with P9 and will extend [`Precision`] rather than reshape this.
//!
//! The model is calibrated against measurements, not first principles.
//! Qwen2.5-1.5B on the reference card measures ~8.0 GiB of runner VRAM in f32
//! and ~3.6 GiB in f16 (`bench/BASELINE.md`) — roughly weights + KV cache plus
//! a fixed overhead for activations, workspaces and CubeCL's memory pools.
//! [`OVERHEAD_BYTES`] is that fixed term, and [`Fit::projected_bytes`] is the
//! whole model; both are honest approximations whose job is to keep a plan on
//! the right side of a cliff, not to predict allocator behaviour to the byte.

use crate::backend::GpuAdapter;

/// Fixed VRAM a live runner needs beyond weights and KV cache: activations,
/// matmul workspaces, and CubeCL's memory pools. Derived from the reference
/// measurements — Qwen2.5-1.5B (1.54 G params) reads ~8.0 GiB runner VRAM in
/// f32 against 6.2 GiB of weights, and ~3.6 GiB in f16 against 3.1 GiB of
/// weights, so the residual is ~0.5-1.8 GiB depending on dtype. 1 GiB is the
/// middle of that band and errs toward *not* promising a fit.
pub const OVERHEAD_BYTES: u64 = 1 << 30;

/// Fraction of an adapter's VRAM a plan may claim. The rest is the display
/// server's: the reference box runs 3.5-6.5 GiB of desktop ambient on the same
/// card, and a plan that ignores it produces an allocation failure at load
/// rather than a slow model.
pub const USABLE_VRAM_FRACTION: f64 = 0.75;

/// Float precisions the planner can pick today. Ordered highest-quality
/// first; P9's int8/int4 tiers extend this enum downward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precision {
    /// `backend::Gpu` — f32 weights and KV cache.
    F32,
    /// `backend::GpuF16` — f16 weights and KV cache, f32 attention-score
    /// island. Needs an adapter advertising `SHADER_F16`.
    F16,
}

impl Precision {
    /// Bytes per stored float.
    #[must_use]
    pub fn bytes_per_float(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    /// Highest-first, the order the planner tries them in.
    #[must_use]
    pub fn descending() -> [Self; 2] {
        [Self::F32, Self::F16]
    }
}

/// What the planner needs to know about a model to size it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// Total parameters (weights + embeddings), as the checkpoint reports.
    pub params: u64,
    /// KV-cache floats stored **per token**, summed over layers:
    /// `2 * layers * num_kv_heads * head_dim`.
    pub kv_floats_per_token: u64,
    /// Context length the plan must hold.
    pub context_tokens: usize,
}

impl ModelShape {
    /// Sizing from a decoder's hyperparameters, so a caller passes the
    /// `config.json` numbers rather than pre-computing cache geometry.
    #[must_use]
    pub fn from_decoder(
        params: u64,
        layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        context_tokens: usize,
    ) -> Self {
        assert!(params > 0, "ModelShape: a model has parameters");
        assert!(
            layers > 0 && num_kv_heads > 0 && head_dim > 0,
            "ModelShape: degenerate decoder geometry \
             (layers {layers}, kv heads {num_kv_heads}, head_dim {head_dim})"
        );
        let kv_floats_per_token = 2 * layers as u64 * num_kv_heads as u64 * head_dim as u64;
        Self {
            params,
            kv_floats_per_token,
            context_tokens,
        }
    }

    /// Projected resident bytes at `precision`: weights + KV cache + the fixed
    /// runner overhead.
    #[must_use]
    pub fn projected_bytes(&self, precision: Precision) -> u64 {
        let per_float = precision.bytes_per_float();
        let weights = self.params.saturating_mul(per_float);
        let kv = self
            .kv_floats_per_token
            .saturating_mul(self.context_tokens as u64)
            .saturating_mul(per_float);
        weights.saturating_add(kv).saturating_add(OVERHEAD_BYTES)
    }
}

/// One adapter's budget, as the planner sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBudget {
    /// Total VRAM the adapter reports.
    pub vram_bytes: u64,
    /// Does it advertise `SHADER_F16`? Without it, f16 is not a candidate at
    /// all — the dev box's own DX12 rows are exactly this case while its
    /// Vulkan rows are not.
    pub shader_f16: bool,
}

impl DeviceBudget {
    /// Read a budget off an enumerated adapter. `None` when VRAM is unknown
    /// (wgpu exposes no portable query; only the Windows DXGI walk fills it
    /// in today) — the planner refuses to guess rather than promise a fit it
    /// cannot size.
    #[must_use]
    pub fn from_adapter(adapter: &GpuAdapter) -> Option<Self> {
        Some(Self {
            vram_bytes: adapter.vram_bytes?,
            shader_f16: adapter.shader_f16,
        })
    }

    /// Bytes a plan may claim, after leaving the display its share.
    #[must_use]
    pub fn usable_bytes(&self) -> u64 {
        let usable = (self.vram_bytes as f64 * USABLE_VRAM_FRACTION) as u64;
        debug_assert!(usable <= self.vram_bytes, "usable VRAM cannot exceed total");
        usable
    }
}

/// A precision decision, with the numbers behind it — the shape the
/// `plan`/`doctor` introspection item will render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fit {
    /// The chosen precision.
    pub precision: Precision,
    /// Projected resident bytes at that precision.
    pub projected_bytes: u64,
    /// Bytes the adapter allows a plan to claim.
    pub usable_bytes: u64,
}

impl Fit {
    /// Headroom left over — the slack a longer context or a second model
    /// would eat into.
    #[must_use]
    pub fn headroom_bytes(&self) -> u64 {
        self.usable_bytes.saturating_sub(self.projected_bytes)
    }
}

/// Pick the **highest** precision that fits `budget`, or `None` when even f16
/// does not — the signal that this model needs quantization (P9) or a
/// multi-device plan, not a smaller float.
///
/// Never silently ships a worse tier than the hardware can hold, and never
/// picks f16 on an adapter that does not advertise `SHADER_F16`.
#[must_use]
pub fn pick_precision(shape: &ModelShape, budget: &DeviceBudget) -> Option<Fit> {
    assert!(shape.params > 0, "pick_precision: a model has parameters");
    let usable = budget.usable_bytes();
    for precision in Precision::descending() {
        if precision == Precision::F16 && !budget.shader_f16 {
            continue;
        }
        let projected = shape.projected_bytes(precision);
        if projected <= usable {
            return Some(Fit {
                precision,
                projected_bytes: projected,
                usable_bytes: usable,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// Qwen2.5-1.5B: 1.54 G params, 28 layers, 2 kv heads, head_dim 128.
    fn qwen2_1_5b(context: usize) -> ModelShape {
        ModelShape::from_decoder(1_543_714_304, 28, 2, 128, context)
    }

    #[test]
    fn projected_size_tracks_the_measured_reference_numbers() {
        let shape = qwen2_1_5b(4096);
        let f32_bytes = shape.projected_bytes(Precision::F32);
        let f16_bytes = shape.projected_bytes(Precision::F16);
        // Measured on the reference card: ~8.0 GiB runner f32, ~3.6 GiB f16
        // (bench/BASELINE.md). The projection must land in the same
        // neighbourhood, not merely be monotone — it reads 7.0 / 3.9 GiB
        // here, i.e. slightly under f32's measurement and slightly over
        // f16's, which is the accuracy a single fixed overhead term buys.
        assert!(
            (6 * GIB..=9 * GIB).contains(&f32_bytes),
            "f32 projection {f32_bytes} outside the measured ~8 GiB band"
        );
        assert!(
            (3 * GIB..=5 * GIB).contains(&f16_bytes),
            "f16 projection {f16_bytes} outside the measured ~3.6 GiB band"
        );
        assert!(f16_bytes < f32_bytes, "f16 must project smaller than f32");
    }

    #[test]
    fn the_reference_card_gets_f32_and_a_small_card_gets_f16() {
        let shape = qwen2_1_5b(4096);
        // RTX 4070 Ti SUPER as the inventory reports it: 15.7 GiB, f16 on Vulkan.
        let big = DeviceBudget {
            vram_bytes: 16_852_000_000,
            shader_f16: true,
        };
        let fit = pick_precision(&shape, &big).expect("1.5B fits a 16 GB card");
        assert_eq!(fit.precision, Precision::F32, "highest that fits wins");
        assert!(fit.headroom_bytes() > 0, "a fit leaves headroom");

        // An 8 GB card cannot hold the f32 build but holds f16 comfortably.
        let small = DeviceBudget {
            vram_bytes: 8 * GIB,
            shader_f16: true,
        };
        let fit = pick_precision(&shape, &small).expect("f16 fits 8 GB");
        assert_eq!(fit.precision, Precision::F16);
        assert!(fit.projected_bytes <= fit.usable_bytes, "a fit must fit");
    }

    #[test]
    fn an_adapter_without_shader_f16_never_gets_an_f16_plan() {
        let shape = qwen2_1_5b(4096);
        // The dev box's own DX12 rows: same card, no SHADER_F16.
        let dx12 = DeviceBudget {
            vram_bytes: 8 * GIB,
            shader_f16: false,
        };
        assert!(
            pick_precision(&shape, &dx12).is_none(),
            "without SHADER_F16 the only candidate is f32, which does not fit"
        );
        // With room for f32, the same adapter plans fine.
        let roomy = DeviceBudget {
            vram_bytes: 24 * GIB,
            shader_f16: false,
        };
        assert_eq!(
            pick_precision(&shape, &roomy).map(|f| f.precision),
            Some(Precision::F32)
        );
    }

    #[test]
    fn a_model_too_big_for_f16_reports_no_fit_rather_than_a_bad_plan() {
        // OLMoE-1B-7B's ~7 G params: ~14 GiB in f16, past a 16 GiB card's
        // usable share — exactly the case bench/BASELINE.md records as
        // "GPU is out of reach until keep-quantized VRAM (P9)".
        let moe = ModelShape::from_decoder(6_919_000_000, 16, 16, 64, 4096);
        let card = DeviceBudget {
            vram_bytes: 16_852_000_000,
            shader_f16: true,
        };
        assert!(
            pick_precision(&moe, &card).is_none(),
            "no float precision fits; the answer is quantization, not a guess"
        );
    }

    #[test]
    fn context_length_moves_the_decision() {
        // The KV cache is the term that grows with context, so a long enough
        // context must be able to push a model off f32 onto f16.
        let card = DeviceBudget {
            vram_bytes: 12 * GIB,
            shader_f16: true,
        };
        let short = pick_precision(&qwen2_1_5b(1024), &card).expect("short context fits");
        // 64k of KV adds ~3.8 GiB in f32 — past a 12 GiB card's usable share,
        // but comfortable at half the width.
        let long = pick_precision(&qwen2_1_5b(65_536), &card).expect("long context still fits");
        assert_eq!(short.precision, Precision::F32);
        assert_eq!(long.precision, Precision::F16, "KV growth forces the drop");
        assert!(
            long.projected_bytes > qwen2_1_5b(1024).projected_bytes(Precision::F16),
            "a longer context must project larger at the same precision"
        );
    }

    #[test]
    fn a_budget_needs_known_vram_before_it_will_plan() {
        // `vram_bytes: None` (every non-Windows adapter today) must yield no
        // budget at all rather than a guessed one.
        let unknown = GpuAdapter {
            name: "test adapter".into(),
            backend: wgpu::Backend::Vulkan,
            device_type: wgpu::DeviceType::DiscreteGpu,
            shader_f16: true,
            max_buffer_bytes: 4 * GIB,
            vram_bytes: None,
        };
        assert!(DeviceBudget::from_adapter(&unknown).is_none());
    }

    #[test]
    fn usable_vram_leaves_the_display_its_share() {
        let budget = DeviceBudget {
            vram_bytes: 16 * GIB,
            shader_f16: true,
        };
        let usable = budget.usable_bytes();
        assert!(usable < budget.vram_bytes, "never plan the whole card");
        assert_eq!(usable, 12 * GIB, "75% of 16 GiB");
    }
}
