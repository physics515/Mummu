//! The Qwen3-VL vision tower: a ViT plus the `qwen3vl_merger` projector,
//! loaded from the companion `mmproj-*.gguf` and producing embeddings in the
//! language model's hidden space.
//!
//! **Where the weights come from.** llama.cpp-style VLM packaging splits the
//! model in two: the language model in the main GGUF, the vision tower in a
//! separate `mmproj` file under `general.architecture = "clip"`. Loading one
//! without the other is the normal state — mummu served the 27B as a
//! text-only model for weeks — so the tower is optional and loaded lazily,
//! the first time a request actually carries an image.
//!
//! **What the checkpoint says.** The geometry is read from the file rather
//! than hardcoded, and anything this module cannot honour is refused loudly
//! at load time instead of being approximated. That matters more here than
//! in most places: a vision tower with a subtly wrong patch order or
//! position interpolation does not error, it produces embeddings that are
//! merely *wrong*, and the language model then describes a meal that was
//! never in the photo. There is no downstream check that catches it.
//!
//! Qwen3.8-27B's tower, for orientation: 27 blocks, hidden 1152, 16 heads
//! (head_dim 72), FFN 4304, patch 16, reference image 768 (a 48x48 patch
//! grid), 2x2 spatial merge, GELU, LN eps 1e-6, projecting to 5120.

use burn::tensor::{Device, Tensor, TensorData};

use crate::gguf::GgufFile;

/// Geometry read from an mmproj checkpoint.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Reference square image the position table was trained at.
    pub image_size: usize,
    pub patch: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub blocks: usize,
    pub heads: usize,
    /// Patches folded per side by the projector (2 => 2x2 => 4 per token).
    pub merge: usize,
    pub eps: f64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Width of the language model's embedding space.
    pub out_dim: usize,
}

impl VisionConfig {
    /// Patches per side in the position table.
    fn pos_grid(&self) -> usize {
        self.image_size / self.patch
    }

    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Read the geometry, refusing anything this module does not implement.
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or_default();
        if arch != "clip" {
            return Err(format!(
                "not an mmproj file: general.architecture is {arch:?}, expected \"clip\""
            ));
        }
        let projector = f
            .get("clip.projector_type")
            .and_then(crate::gguf::GgufValue::as_str)
            .unwrap_or_default();
        if projector != "qwen3vl_merger" {
            return Err(format!(
                "unsupported projector {projector:?} — this build implements \"qwen3vl_merger\""
            ));
        }
        if f.get("clip.has_vision_encoder")
            .and_then(crate::gguf::GgufValue::as_bool)
            != Some(true)
        {
            return Err("mmproj has no vision encoder".into());
        }
        // DeepStack feeds selected ViT layers into the language model at
        // several depths. Qwen3.8-27B's mmproj has it off for every layer,
        // and a checkpoint that turns it on needs code this module does not
        // have — so refuse rather than silently drop the extra features.
        if let Some(flags) = f.get("clip.vision.is_deepstack_layers")
            && let Some(items) = flags.as_array()
            && items.iter().any(|v| v.as_bool() == Some(true))
        {
            return Err(
                "this mmproj enables DeepStack layers, which this build does not implement — \
                 the extra vision features would be silently dropped and the answer would be \
                 about the wrong image"
                    .into(),
            );
        }

        let u = |k: &str| -> Result<usize, String> {
            f.get(k)
                .and_then(crate::gguf::GgufValue::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("mmproj is missing {k}"))
        };
        let rgb = |k: &str| -> [f32; 3] {
            f.get(k)
                .and_then(crate::gguf::GgufValue::as_array)
                .map(|a| {
                    let mut out = [0.5f32; 3];
                    for (slot, v) in out.iter_mut().zip(a) {
                        if let Some(x) = v.as_f32() {
                            *slot = x;
                        }
                    }
                    out
                })
                .unwrap_or([0.5; 3])
        };

        let cfg = Self {
            image_size: u("clip.vision.image_size")?,
            patch: u("clip.vision.patch_size")?,
            hidden: u("clip.vision.embedding_length")?,
            ffn: u("clip.vision.feed_forward_length")?,
            blocks: u("clip.vision.block_count")?,
            heads: u("clip.vision.attention.head_count")?,
            merge: u("clip.vision.spatial_merge_size").unwrap_or(2),
            eps: f
                .get("clip.vision.attention.layer_norm_epsilon")
                .and_then(crate::gguf::GgufValue::as_f32)
                .unwrap_or(1e-6) as f64,
            mean: rgb("clip.vision.image_mean"),
            std: rgb("clip.vision.image_std"),
            out_dim: u("clip.vision.projection_dim")?,
        };
        if cfg.patch == 0 || cfg.heads == 0 || !cfg.hidden.is_multiple_of(cfg.heads) {
            return Err(format!(
                "inconsistent vision geometry: hidden {} over {} heads, patch {}",
                cfg.hidden, cfg.heads, cfg.patch
            ));
        }
        if !cfg.image_size.is_multiple_of(cfg.patch) {
            return Err(format!(
                "image_size {} is not a multiple of patch {}",
                cfg.image_size, cfg.patch
            ));
        }
        Ok(cfg)
    }
}

/// A 2-D weight as GGUF stores it. GGML's `ne` is fastest-varying first, so
/// a `[in, out]` header describes a row-major `[out, in]` matrix — the usual
/// orientation for `y = x · Wᵀ`.
fn weight_2d(
    f: &GgufFile,
    name: &str,
    out: usize,
    inp: usize,
    device: &Device,
) -> Result<Tensor<2>, String> {
    let data = f
        .read_tensor_f32(name)
        .map_err(|e| format!("{name}: {e:?}"))?;
    if data.len() != out * inp {
        return Err(format!(
            "{name}: expected {out}x{inp} = {} values, got {}",
            out * inp,
            data.len()
        ));
    }
    Ok(half_precision(Tensor::<2>::from_data(
        TensorData::new(data, [out, inp]),
        device,
    )))
}

/// Store a loaded weight at the precision the checkpoint used.
///
/// `read_tensor_f32` materializes everything as f32, so an F16 mmproj would
/// otherwise land on the card at double its file size — 0.93 GB becoming
/// ~1.85 GiB, all of which the fit planner has to reserve and therefore take
/// away from the language model's layers (measured: 10 of 64 on the 27B, and
/// decode from ~0.83 to ~2.3 s/token, for text requests too).
///
/// Casting back to f16 is lossless here: the values came from f16 in the
/// file. `MUMMU_VISION_F32=1` keeps the wide copy for a numerics
/// investigation.
fn half_precision<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    if std::env::var("MUMMU_VISION_F32").is_ok_and(|v| v != "0") {
        return t;
    }
    t.cast(burn::tensor::FloatDType::F16)
}

fn vector(f: &GgufFile, name: &str, n: usize, device: &Device) -> Result<Tensor<1>, String> {
    let data = f
        .read_tensor_f32(name)
        .map_err(|e| format!("{name}: {e:?}"))?;
    if data.len() != n {
        return Err(format!("{name}: expected {n} values, got {}", data.len()));
    }
    // Biases and norms are F32 in the file; casting them too keeps every
    // operand of a matmul at one width.
    Ok(half_precision(Tensor::<1>::from_data(
        TensorData::new(data, [n]),
        device,
    )))
}

/// Widen a stored weight to the compute width, at the point of use.
///
/// Weights are *stored* at the checkpoint's own precision to keep the
/// tower's resident footprint small (see [`half_precision`]); activations
/// stay f32. burn does not multiply mixed dtypes — the CPU backend asserts
/// `matmul: dtype mismatch`, and that assertion is exactly what the first
/// f16 build hit, because it cast the weights and never the operations.
/// Widening per use keeps compute numerically identical to an all-f32 tower
/// while only ever holding one weight wide at a time.
fn wide<const D: usize>(t: &Tensor<D>) -> Tensor<D> {
    t.clone().cast(burn::tensor::FloatDType::F32)
}

/// `y = x · Wᵀ + b` for `[tokens, in] · [out, in]ᵀ`.
fn linear(x: Tensor<2>, w: &Tensor<2>, b: &Tensor<1>) -> Tensor<2> {
    let out = w.dims()[0];
    x.matmul(wide(w).swap_dims(0, 1)) + wide(b).reshape([1, out])
}

/// Layer norm with learned scale and shift, over the last dimension.
fn layer_norm(x: Tensor<2>, w: &Tensor<1>, b: &Tensor<1>, eps: f64) -> Tensor<2> {
    let d = x.dims()[1];
    let mean = x.clone().mean_dim(1);
    let centered = x - mean.clone();
    let var = centered.clone().powf_scalar(2.0).mean_dim(1);
    let normed = centered / var.add_scalar(eps).sqrt();
    normed * wide(w).reshape([1, d]) + wide(b).reshape([1, d])
}

/// The exact GELU (erf form), which is what `clip.use_gelu` selects.
fn gelu(x: Tensor<2>) -> Tensor<2> {
    burn::tensor::activation::gelu(x)
}

/// Base for the vision tower's rotary frequencies. Not stored in the
/// checkpoint (RoPE has no weights), so it is the architecture's constant.
const VISION_ROPE_THETA: f32 = 10_000.0;

/// 2-D rotary tables for a `gh x gw` patch grid: `[n, head_dim]` cos and
/// sin, in the same row-major patch order the tower is fed in.
///
/// **Why this exists.** The learned table added before the blocks is an
/// *absolute* signal; Qwen3-VL's encoder also rotates queries and keys by
/// each patch's (y, x) inside attention, which is what carries *relative*
/// geometry. Without it the tower aggregates colour and texture correctly
/// and loses spatial structure: measured 2026-09-20, two hot dogs on a
/// plate came back as "a single long strip of nachos" — condiment right,
/// shape and count wrong.
///
/// The head splits in half: the low half rotates by the row, the high half
/// by the column, each laid out as the pairs [`rotate_half`] expects.
fn rope_2d(gh: usize, gw: usize, head_dim: usize, device: &Device) -> (Tensor<2>, Tensor<2>) {
    let half = head_dim / 2;
    let pairs = half / 2;
    let n = gh * gw;
    let mut cos = vec![0f32; n * head_dim];
    let mut sin = vec![0f32; n * head_dim];
    for y in 0..gh {
        for x in 0..gw {
            let row = (y * gw + x) * head_dim;
            for i in 0..pairs {
                #[allow(clippy::cast_precision_loss)]
                let inv = 1.0 / VISION_ROPE_THETA.powf(2.0 * i as f32 / half as f32);
                #[allow(clippy::cast_precision_loss)]
                let angles = [(0usize, y as f32 * inv), (pairs, x as f32 * inv)];
                // Layout is [h, w, h, w]: the reference builds
                // `freqs = cat(h_freqs, w_freqs)` — `pairs` of each — and
                // then `emb = cat(freqs, freqs)`, so the second copy sits a
                // whole `half` away, not a `pairs` away.
                //
                // Writing each axis into its own contiguous half instead
                // ([h, h, w, w]) is self-consistent and therefore compiles
                // and answers fluently — while rotating dims 0..half by the
                // ROW that the weights expect to carry the column. Measured
                // 2026-09-20: that made the tower amplify a 0.1% input
                // change into a 12% change in the projected tokens.
                for (base, angle) in angles {
                    let (c, sn) = (angle.cos(), angle.sin());
                    cos[row + base + i] = c;
                    cos[row + half + base + i] = c;
                    sin[row + base + i] = sn;
                    sin[row + half + base + i] = sn;
                }
            }
        }
    }
    (
        Tensor::<2>::from_data(TensorData::new(cos, [n, head_dim]), device),
        Tensor::<2>::from_data(TensorData::new(sin, [n, head_dim]), device),
    )
}

/// The `(a, b) -> (-b, a)` companion of [`rope_2d`]'s layout, applied within
/// each half independently so the row and column rotations never mix.
fn rotate_half(x: Tensor<3>) -> Tensor<3> {
    let [h, n, d] = x.dims();
    let half = d / 2;
    // The plain GPT-NeoX rotation: the second half negated in front of the
    // first. Pairing `i` with `i + half` is what the [h, w, h, w] table
    // layout requires — each dimension meets its own duplicate, so the row
    // and column rotations stay on the dimensions the weights expect.
    let a = x.clone().slice([0..h, 0..n, 0..half]);
    let b = x.slice([0..h, 0..n, half..d]);
    Tensor::cat(vec![-b, a], 2)
}

/// Apply the rotation to `[heads, n, head_dim]`.
fn apply_rope(x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<3> {
    let [heads, n, d] = x.dims();
    let c = cos.clone().reshape([1, n, d]).repeat_dim(0, heads);
    let s = sin.clone().reshape([1, n, d]).repeat_dim(0, heads);
    x.clone() * c + rotate_half(x) * s
}

/// One stage's activation, read back to the host by
/// [`VisionTower::forward_staged`].
pub struct Stage {
    pub name: String,
    pub values: Vec<f32>,
}

impl Stage {
    fn capture(name: String, t: &Tensor<2>) -> Self {
        // A readback failure is reported, not turned into an empty stage: an
        // empty stage makes every cosine `None`, which prints as "n/a" and
        // reads like "not comparable" rather than "broken".
        let values = t
            .clone()
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .unwrap_or_else(|e| panic!("vision stage {name}: readback failed: {e:?}"));
        Self { name, values }
    }

    /// Mean, standard deviation, L2 norm and largest magnitude.
    #[must_use]
    pub fn summary(&self) -> (f64, f64, f64, f32) {
        let n = self.values.len().max(1) as f64;
        let mean = self.values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = self
            .values
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let l2 = self
            .values
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        let maxabs = self.values.iter().fold(0f32, |m, &v| m.max(v.abs()));
        (mean, var.sqrt(), l2, maxabs)
    }

    /// Cosine similarity with the same stage of another run, or `None` when
    /// the shapes differ (a different grid is a different sequence length,
    /// not a different answer).
    #[must_use]
    pub fn cosine(&self, other: &Self) -> Option<f64> {
        if self.values.len() != other.values.len() || self.values.is_empty() {
            return None;
        }
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&a, &b) in self.values.iter().zip(&other.values) {
            let (a, b) = (f64::from(a), f64::from(b));
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        Some(dot / (na.sqrt() * nb.sqrt()).max(1e-12))
    }
}

struct Block {
    ln1_w: Tensor<1>,
    ln1_b: Tensor<1>,
    qkv_w: Tensor<2>,
    qkv_b: Tensor<1>,
    out_w: Tensor<2>,
    out_b: Tensor<1>,
    ln2_w: Tensor<1>,
    ln2_b: Tensor<1>,
    up_w: Tensor<2>,
    up_b: Tensor<1>,
    down_w: Tensor<2>,
    down_b: Tensor<1>,
}

/// The loaded tower.
pub struct VisionTower {
    pub cfg: VisionConfig,
    /// `[hidden, 3 * patch * patch]` — the two temporal kernels summed; see
    /// [`VisionTower::load`].
    patch_w: Tensor<2>,
    patch_b: Tensor<1>,
    /// `[pos_grid², hidden]`, interpolated per image.
    pos: Tensor<2>,
    blocks: Vec<Block>,
    post_ln_w: Tensor<1>,
    post_ln_b: Tensor<1>,
    mm0_w: Tensor<2>,
    mm0_b: Tensor<1>,
    mm2_w: Tensor<2>,
    mm2_b: Tensor<1>,
}

impl VisionTower {
    /// Load from an mmproj GGUF onto `device`.
    ///
    /// **The two patch kernels.** Qwen3-VL's patch embedding is a 3-D
    /// convolution with temporal extent 2, stored as two 2-D kernels
    /// (`v.patch_embd.weight` and `.weight.1`) for the two frames. A still
    /// image is fed as both frames — that is what the reference
    /// implementations do — so the convolution reduces to `(W₀ + W₁) · patch`
    /// and the two kernels are summed once here rather than per image.
    pub fn load(path: &std::path::Path, device: &Device) -> Result<Self, String> {
        let f = GgufFile::open(path).map_err(|e| format!("open mmproj {path:?}: {e:?}"))?;
        let cfg = VisionConfig::from_gguf(&f)?;
        let (h, p) = (cfg.hidden, cfg.patch);
        let patch_in = 3 * p * p;

        let w0 = weight_2d(&f, "v.patch_embd.weight", h, patch_in, device)?;
        let patch_w = match weight_2d(&f, "v.patch_embd.weight.1", h, patch_in, device) {
            Ok(w1) => w0 + w1,
            // A tower with temporal extent 1 has only the one kernel.
            Err(_) => w0,
        };
        let patch_b = vector(&f, "v.patch_embd.bias", h, device)?;

        let positions = cfg.pos_grid() * cfg.pos_grid();
        let pos = weight_2d(&f, "v.position_embd.weight", positions, h, device)?;

        let mut blocks = Vec::with_capacity(cfg.blocks);
        for i in 0..cfg.blocks {
            let k = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(Block {
                ln1_w: vector(&f, &k("ln1.weight"), h, device)?,
                ln1_b: vector(&f, &k("ln1.bias"), h, device)?,
                qkv_w: weight_2d(&f, &k("attn_qkv.weight"), 3 * h, h, device)?,
                qkv_b: vector(&f, &k("attn_qkv.bias"), 3 * h, device)?,
                out_w: weight_2d(&f, &k("attn_out.weight"), h, h, device)?,
                out_b: vector(&f, &k("attn_out.bias"), h, device)?,
                ln2_w: vector(&f, &k("ln2.weight"), h, device)?,
                ln2_b: vector(&f, &k("ln2.bias"), h, device)?,
                up_w: weight_2d(&f, &k("ffn_up.weight"), cfg.ffn, h, device)?,
                up_b: vector(&f, &k("ffn_up.bias"), cfg.ffn, device)?,
                down_w: weight_2d(&f, &k("ffn_down.weight"), h, cfg.ffn, device)?,
                down_b: vector(&f, &k("ffn_down.bias"), h, device)?,
            });
        }

        let merged = h * cfg.merge * cfg.merge;
        let out_dim = cfg.out_dim;
        Ok(Self {
            cfg,
            patch_w,
            patch_b,
            pos,
            blocks,
            post_ln_w: vector(&f, "v.post_ln.weight", h, device)?,
            post_ln_b: vector(&f, "v.post_ln.bias", h, device)?,
            mm0_w: weight_2d(&f, "mm.0.weight", merged, merged, device)?,
            mm0_b: vector(&f, "mm.0.bias", merged, device)?,
            mm2_w: weight_2d(&f, "mm.2.weight", out_dim, merged, device)?,
            mm2_b: vector(&f, "mm.2.bias", out_dim, device)?,
        })
    }

    /// Bilinearly resample the learned `[g, g]` position grid to `[gh, gw]`.
    ///
    /// The table is trained at one resolution (48x48 here) and images arrive
    /// at whatever the client sent, so every grid but the reference one needs
    /// this. Done on the host in f32: it is a few hundred KiB and once per
    /// image, against a ViT that is about to run 27 blocks.
    fn positions_for(&self, gh: usize, gw: usize, device: &Device) -> Tensor<2> {
        let g = self.cfg.pos_grid();
        let h = self.cfg.hidden;
        if gh == g && gw == g {
            // The stored table is f16; the activations it is added to are not.
            return wide(&self.pos);
        }
        let table = self
            .pos
            .clone()
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("position table readback");
        let mut out = vec![0f32; gh * gw * h];
        // `align_corners = False`, which is what `F.interpolate` does by
        // default and therefore what the reference resampler does: an output
        // cell maps to the CENTRE of its source footprint, `(i + 0.5) *
        // g / n - 0.5`, not to a corner-anchored fraction.
        //
        // The distinction is not academic. Corner-anchored interpolation
        // agrees closely when the target grid is near the table's native
        // 48x48 and drifts as the scale factor moves away — measured
        // 2026-09-20, the same photo read correctly at 40x50 patches and
        // came back as "three crepes" at 24x32 and "12 tacos" at 50x40.
        #[allow(clippy::cast_precision_loss)]
        let scale = |i: usize, n: usize| -> f32 {
            let src = (i as f32 + 0.5) * (g as f32 / n as f32) - 0.5;
            src.clamp(0.0, (g - 1) as f32)
        };
        for y in 0..gh {
            let fy = scale(y, gh);
            let (y0, wy) = (fy.floor() as usize, fy - fy.floor());
            let y1 = (y0 + 1).min(g - 1);
            for x in 0..gw {
                let fx = scale(x, gw);
                let (x0, wx) = (fx.floor() as usize, fx - fx.floor());
                let x1 = (x0 + 1).min(g - 1);
                let (a, b, c, d) = (
                    (y0 * g + x0) * h,
                    (y0 * g + x1) * h,
                    (y1 * g + x0) * h,
                    (y1 * g + x1) * h,
                );
                let dst = (y * gw + x) * h;
                for k in 0..h {
                    let top = table[a + k] * (1.0 - wx) + table[b + k] * wx;
                    let bot = table[c + k] * (1.0 - wx) + table[d + k] * wx;
                    out[dst + k] = top * (1.0 - wy) + bot * wy;
                }
            }
        }
        Tensor::<2>::from_data(TensorData::new(out, [gh * gw, h]), device)
    }

    /// Run the tower over one preprocessed image.
    ///
    /// `patches` is `[gh * gw, 3 * patch * patch]` in row-major patch order
    /// (top-left to bottom-right), each patch laid out channel-major —
    /// `(c, y, x)` — to match how the convolution kernel is stored.
    /// Returns `[gh/merge * gw/merge, out_dim]`: the tokens to splice into
    /// the language model's embedding sequence.
    pub fn forward(
        &self,
        patches: Tensor<2>,
        gh: usize,
        gw: usize,
        device: &Device,
    ) -> Result<Tensor<2>, String> {
        self.forward_inner(patches, gh, gw, device, None)
    }

    /// [`Self::forward`], also returning the activation after every stage.
    ///
    /// The encoder's output drifts between near-identical inputs, and the
    /// output alone cannot say *where* the drift enters: 27 blocks, a merge
    /// and a projector are all suspects. Comparing two inputs stage by stage
    /// finds the first stage where they separate, which turns "somewhere in
    /// the tower" into one operation to inspect.
    ///
    /// Every stage is read back to the host, a full device sync each — so
    /// this is a diagnostic path, never the serving one.
    pub fn forward_staged(
        &self,
        patches: Tensor<2>,
        gh: usize,
        gw: usize,
        device: &Device,
    ) -> Result<(Tensor<2>, Vec<Stage>), String> {
        let mut stages = Vec::new();
        let out = self.forward_inner(patches, gh, gw, device, Some(&mut stages))?;
        Ok((out, stages))
    }

    fn forward_inner(
        &self,
        patches: Tensor<2>,
        gh: usize,
        gw: usize,
        device: &Device,
        mut stages: Option<&mut Vec<Stage>>,
    ) -> Result<Tensor<2>, String> {
        let mut record = |name: String, t: &Tensor<2>| {
            if let Some(s) = stages.as_deref_mut() {
                s.push(Stage::capture(name, t));
            }
        };
        let m = self.cfg.merge;
        if !gh.is_multiple_of(m) || !gw.is_multiple_of(m) {
            return Err(format!(
                "patch grid {gh}x{gw} is not a multiple of the {m}x{m} spatial merge"
            ));
        }
        let (n, heads, hd) = (gh * gw, self.cfg.heads, self.cfg.head_dim());
        let h = self.cfg.hidden;

        let embedded = linear(patches, &self.patch_w, &self.patch_b);
        record("patch_embed".into(), &embedded);
        let mut x = embedded + self.positions_for(gh, gw, device);
        record("pos_add".into(), &x);
        let (cos, sin) = rope_2d(gh, gw, hd, device);

        for (bi, b) in self.blocks.iter().enumerate() {
            // Attention: bidirectional, no mask. Position reaches it twice —
            // the absolute table added above, and the 2-D rotation applied to
            // queries and keys below. (Until 2026-09-20 this comment said "no
            // RoPE", which was true only of the first version, and was wrong
            // for as long as it survived the RoPE going in.)
            let normed = layer_norm(x.clone(), &b.ln1_w, &b.ln1_b, self.cfg.eps);
            let qkv = linear(normed, &b.qkv_w, &b.qkv_b);
            // `[n, 3h]` is laid out q|k|v along the feature axis, so the
            // three projections are contiguous slices — not an interleave.
            let head = |lo: usize| {
                qkv.clone()
                    .slice([0..n, lo * h..(lo + 1) * h])
                    .reshape([n, heads, hd])
                    .swap_dims(0, 1)
            };
            let (q, k, v) = (head(0), head(1), head(2));
            // Queries and keys carry the 2-D rotation; values do not.
            let q = apply_rope(q, &cos, &sin);
            let k = apply_rope(k, &cos, &sin);
            let scores = q.matmul(k.swap_dims(1, 2)) / (hd as f32).sqrt();
            let probs = burn::tensor::activation::softmax(scores, 2);
            let attn = probs.matmul(v).swap_dims(0, 1).reshape([n, h]);
            x = x + linear(attn, &b.out_w, &b.out_b);

            let normed = layer_norm(x.clone(), &b.ln2_w, &b.ln2_b, self.cfg.eps);
            let up = gelu(linear(normed, &b.up_w, &b.up_b));
            x = x + linear(up, &b.down_w, &b.down_b);
            record(format!("block_{bi:02}"), &x);
        }

        let x = layer_norm(x, &self.post_ln_w, &self.post_ln_b, self.cfg.eps);
        record("post_ln".into(), &x);

        // Spatial merge: fold each m x m block of neighbouring patches into
        // one token by concatenating their features. Row-major patch order
        // makes this a reshape-and-permute rather than a gather.
        let (mh, mw) = (gh / m, gw / m);
        let merged = x
            .reshape([mh, m, mw, m, h])
            .swap_dims(1, 2)
            .reshape([mh * mw, m * m * h]);
        record("merge".into(), &merged);

        let hidden = gelu(linear(merged, &self.mm0_w, &self.mm0_b));
        record("mm0_gelu".into(), &hidden);
        let projected = linear(hidden, &self.mm2_w, &self.mm2_b);
        record("mm2_out".into(), &projected);
        Ok(projected)
    }

    /// How many language-model tokens an image of this patch grid becomes.
    #[must_use]
    pub fn token_count(&self, gh: usize, gw: usize) -> usize {
        (gh / self.cfg.merge) * (gw / self.cfg.merge)
    }
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Upper bound on merged image tokens per image.
///
/// Each one costs a position in the language model's context and a slice of
/// prefill. Qwen3-VL will happily take thousands from a big photo; on a 27B
/// decoding at sub-token-per-second that turns a meal snapshot into minutes
/// of prefill before the first word. 512 merged tokens is enough to keep a
/// 4:3 photo at roughly 26x19 cells, which preserves the shape of what is
/// in it; the whole tower measured ~22 s against 0.83 s per decoded token,
/// so the budget belongs on detail, not on prefill thrift.
pub const MAX_IMAGE_TOKENS: usize = 512;

/// Floor on merged tokens per image, for the same reason there is a cap.
///
/// The position table is fitted at one grid; too far below it and the
/// encoder's spatial signal degrades just as it does too far above. 480 is
/// just under the cap, so every image lands in a narrow band around the
/// grid that was measured working, whatever size it arrived at.
pub const MIN_IMAGE_TOKENS: usize = 480;

/// An image, preprocessed into the tower's input.
pub struct Patches {
    /// `[gh * gw, 3 * patch * patch]`, each row one patch in `(c, y, x)`.
    pub data: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl VisionConfig {
    /// The patch grid to resample an image of `(w, h)` onto.
    ///
    /// Both sides land on a multiple of `patch * merge`, because a partial
    /// merge block has no meaning, and the total is capped at
    /// [`MAX_IMAGE_TOKENS`] merged tokens.
    ///
    /// **Both axes scale by the same factor.** An earlier version shrank the
    /// longer side one cell at a time until the budget was met, which
    /// converges on a square: a 1500x1125 photo became a 16x16 grid, and the
    /// model read two hot dogs as "slices stacked in a row" and reported the
    /// image as rotated. Aspect ratio is not cosmetic here — it is most of
    /// what distinguishes one food from another.
    fn grid_for(&self, w: u32, h: u32) -> (usize, usize) {
        let step = self.patch * self.merge;
        // Work in merged cells; the patch grid is this times `merge`.
        let cells = |px: u32| ((px as usize).div_ceil(step)).max(1);
        let (mut mh, mut mw) = (cells(h), cells(w));
        // Scale UP as well as down. The learned position table is trained at
        // one grid (48x48 here) and the encoder reads a grid far from it
        // poorly: measured 2026-09-20 on one photo of two hot dogs, a 40x50
        // patch grid described it correctly and a 24x32 grid called it
        // "three crepes". A small image is therefore enlarged to land near
        // the same working range rather than passed through at its own tiny
        // grid — upsampling invents no detail, but it keeps the position
        // signal in the regime the table was fitted for.
        if mh * mw < MIN_IMAGE_TOKENS {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let scale = (MIN_IMAGE_TOKENS as f64 / (mh * mw) as f64).sqrt();
            mh = ((mh as f64 * scale).round() as usize).max(1);
            mw = ((mw as f64 * scale).round() as usize).max(1);
        }
        if mh * mw > MAX_IMAGE_TOKENS {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let scale = (MAX_IMAGE_TOKENS as f64 / (mh * mw) as f64).sqrt();
            mh = ((mh as f64 * scale).floor() as usize).max(1);
            mw = ((mw as f64 * scale).floor() as usize).max(1);
            // Flooring both can leave budget on the table; spend it on the
            // longer side, which is the one carrying the shape.
            while (mh + 1) * mw <= MAX_IMAGE_TOKENS && mh <= mw {
                mh += 1;
            }
            while mh * (mw + 1) <= MAX_IMAGE_TOKENS && mw <= mh {
                mw += 1;
            }
        }
        (mh * self.merge, mw * self.merge)
    }

    /// Decode `bytes` and lay it out as patches for [`VisionTower::forward`].
    ///
    /// The format is sniffed from the content, never from a filename or a
    /// client-supplied MIME type — those are attacker-controlled on this
    /// path, and `image` is perfectly able to tell a PNG from a JPEG itself.
    pub fn preprocess(&self, bytes: &[u8]) -> Result<Patches, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| format!("could not decode the image: {e}"))?;
        let (w, h) = (img.width(), img.height());
        if w == 0 || h == 0 {
            return Err("image has a zero dimension".into());
        }
        let (grid_h, grid_w) = self.grid_for(w, h);
        let (px_h, px_w) = (grid_h * self.patch, grid_w * self.patch);

        // Triangle filter: a plain nearest-neighbour resize aliases badly on
        // photographs, and the tower was trained on smoothly resampled input.
        let scaled = image::imageops::resize(
            &img.to_rgb8(),
            px_w as u32,
            px_h as u32,
            image::imageops::FilterType::Triangle,
        );

        let p = self.patch;
        let mut data = vec![0f32; grid_h * grid_w * 3 * p * p];
        let stride = 3 * p * p;
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let row = (gy * grid_w + gx) * stride;
                for c in 0..3 {
                    for y in 0..p {
                        for x in 0..p {
                            let px = scaled.get_pixel((gx * p + x) as u32, (gy * p + y) as u32);
                            // Normalize with the checkpoint's own statistics.
                            let v = f32::from(px.0[c]) / 255.0;
                            data[row + c * p * p + y * p + x] = (v - self.mean[c]) / self.std[c];
                        }
                    }
                }
            }
        }
        let p = Patches {
            data,
            grid_h,
            grid_w,
        };
        // One line per image, because the arithmetic and the behaviour
        // disagree: 900x675 and 1500x1125 compute to the same grid and the
        // same post-resize size, and the model describes them differently.
        // Printing what actually happened — rather than what `grid_for` is
        // believed to do — is the only way to find where they diverge.
        if std::env::var("MUMMU_VISION_TRACE").is_ok_and(|v| v != "0") {
            eprintln!(
                "[mummu] vision trace: in {w}x{h} -> grid {grid_h}x{grid_w} patches \
                 ({} tokens) -> resized {px_w}x{px_h} -> {} floats, {}",
                (grid_h / self.merge) * (grid_w / self.merge),
                p.data.len(),
                p.fingerprint(),
            );
        }
        Ok(p)
    }
}

impl Patches {
    /// A summary of the patch tensor: enough to tell two preprocessings
    /// apart without printing a million floats.
    ///
    /// Deliberately more than a hash. A hash answers "are these the same",
    /// which is the question already known to be interesting; the mean and
    /// the extremes also answer "and if not, how" — a normalization or
    /// channel-order mistake moves the mean, a resampling one moves the
    /// range, and a reordering moves neither.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let n = self.data.len().max(1) as f64;
        let sum: f64 = self.data.iter().map(|&v| f64::from(v)).sum();
        let mean = sum / n;
        let var = self
            .data
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let (lo, hi) = self
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        // Order-sensitive checksum: two tensors with identical contents in a
        // different patch order must NOT look alike here, since a wrong
        // ordering is one of the live suspects.
        let mut ck: u64 = 0xcbf2_9ce4_8422_2325;
        for (i, &v) in self.data.iter().enumerate() {
            ck ^= u64::from(v.to_bits()).wrapping_mul(i as u64 | 1);
            ck = ck.wrapping_mul(0x0100_0000_01b3);
        }
        format!(
            "mean {mean:+.5} sd {:.5} range [{lo:+.3}, {hi:+.3}] ck {ck:016x}",
            var.sqrt()
        )
    }

    /// Upload as `[grid_h * grid_w, 3 * patch * patch]`.
    #[must_use]
    pub fn to_tensor(&self, device: &Device) -> Tensor<2> {
        let rows = self.grid_h * self.grid_w;
        let cols = self.data.len() / rows;
        Tensor::<2>::from_data(TensorData::new(self.data.clone(), [rows, cols]), device)
    }
}

// ---------------------------------------------------------------------------
// Splicing images into a prompt
// ---------------------------------------------------------------------------

/// Qwen3-VL's placeholder tokens, as the 27B's tokenizer spells them.
///
/// The ids are looked up by name at load time rather than hardcoded: they
/// are tokenizer facts, not architecture facts, and a checkpoint that
/// renumbers them would otherwise splice image rows over whatever text
/// happened to land on the old id.
pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";

/// The placeholder ids this tokenizer uses.
#[derive(Debug, Clone, Copy)]
pub struct VisionTokens {
    pub start: u32,
    pub end: u32,
    pub pad: u32,
}

impl VisionTokens {
    /// Resolve the ids, or say which one is missing.
    pub fn resolve(tok: &tokenizers::Tokenizer) -> Result<Self, String> {
        let id = |s: &str| {
            tok.token_to_id(s)
                .ok_or_else(|| format!("this tokenizer has no {s} token, so it cannot take images"))
        };
        Ok(Self {
            start: id(VISION_START)?,
            end: id(VISION_END)?,
            pad: id(IMAGE_PAD)?,
        })
    }
}

/// The text that reserves `count` positions for one image.
///
/// Free rather than a method on [`VisionTokens`] because it is needed
/// *before* the model (and so the tokenizer) is loaded: the number of
/// placeholders depends on the image's patch grid, and the prompt has to
/// carry them before it can be rendered and tokenized.
#[must_use]
pub fn placeholder_text(count: usize) -> String {
    let mut s = String::with_capacity((count + 2) * IMAGE_PAD.len());
    s.push_str(VISION_START);
    for _ in 0..count {
        s.push_str(IMAGE_PAD);
    }
    s.push_str(VISION_END);
    s
}

/// Merged token count for a preprocessed image.
#[must_use]
pub fn token_count(p: &Patches, merge: usize) -> usize {
    (p.grid_h / merge) * (p.grid_w / merge)
}

/// One image's projected tokens, and where they belong in the prompt.
pub struct Placed {
    /// `[count, hidden]` in the language model's embedding space.
    pub rows: Tensor<2>,
    /// Absolute index of the first row in the prompt's token sequence.
    pub at: usize,
}

/// Match each run of `pad` ids in `prompt_ids` to one image's rows, in order.
///
/// Refuses on a mismatch rather than splicing what it can: a prompt whose
/// placeholder count disagrees with the tower's output means the two were
/// built from different assumptions, and the resulting sequence would be
/// silently misaligned — every image row landing one position off reads as
/// a different picture, with no error anywhere.
pub fn place(prompt_ids: &[u32], pad: u32, images: Vec<Tensor<2>>) -> Result<Vec<Placed>, String> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < prompt_ids.len() {
        if prompt_ids[i] == pad {
            let start = i;
            while i < prompt_ids.len() && prompt_ids[i] == pad {
                i += 1;
            }
            runs.push((start, i - start));
        } else {
            i += 1;
        }
    }
    if runs.len() != images.len() {
        return Err(format!(
            "prompt has {} image placeholder run(s) but {} image(s) were encoded",
            runs.len(),
            images.len()
        ));
    }
    runs.iter()
        .zip(images)
        .map(|(&(at, len), rows)| {
            let n = rows.dims()[0];
            if n != len {
                return Err(format!(
                    "image placeholder run at {at} reserves {len} positions but the vision \
                     tower produced {n} tokens"
                ));
            }
            Ok(Placed { rows, at })
        })
        .collect()
}

/// Overwrite the rows of `x` (`[1, t, hidden]`, covering absolute positions
/// `past..past + t`) that belong to images.
///
/// Called per prefill chunk, so an image that straddles a chunk boundary is
/// spliced in the pieces that land in each — which is why the overlap is
/// computed rather than assumed to be whole.
pub fn splice(x: Tensor<3>, past: usize, placed: &[Placed]) -> Tensor<3> {
    let [_, t, hidden] = x.dims();
    let mut x = x;
    for p in placed {
        let n = p.rows.dims()[0];
        let (lo, hi) = (p.at.max(past), (p.at + n).min(past + t));
        if lo >= hi {
            continue;
        }
        let src = p
            .rows
            .clone()
            .slice([lo - p.at..hi - p.at, 0..hidden])
            .reshape([1, hi - lo, hidden]);
        x = x.slice_assign([0..1, lo - past..hi - past, 0..hidden], src);
    }
    x
}
