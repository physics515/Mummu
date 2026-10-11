//! Packed m=1 GEMV for block-quantized weights — the kernel the whole
//! scheduler hunt pointed at.
//!
//! Burn's own quantized matmul at m=1 lands on a documented "extremely
//! hacky fix" (burn-cubecl `kernel/matmul/base.rs`): dequantize the ENTIRE
//! weight to f32, then float-matmul — 4x the traffic of the packed bytes,
//! a transient f32 weight allocation per call (the VRAM pool churn), and
//! the measured 0.91 ms/cluster against the host slab's 0.40. This module
//! reads the packed representation directly on every backend that holds
//! clusters:
//!
//! - **wgpu/CUDA** (`CubeBackend<R>`): a `#[cube]` kernel where each unit
//!   owns one u32 word (8 Q4S nibbles) per k-step — one packed word, one
//!   shared f32 scale, one x[k] broadcast, eight fused mul-adds, no
//!   cross-unit reduction. Weight traffic is the packed bytes, nothing
//!   else.
//! - **flex** (host): a threaded i8 GEMV over the backend's i8-unpacked
//!   storage (flex stores Q4 as one i8 per element) — 1.125 B/elem of
//!   traffic instead of the 4 B/elem of the f32 slab it can replace.
//!
//! The op is exact with respect to the stored quantization: it computes
//! `y[n] = Σ_k x[k] · scale(k, n/32) · q(k, n)` in f32, the same math the
//! dequantize-then-matmul reference performs, so parity against that
//! reference is a summation-order question only (tested below).
//!
//! Layout contract (asserted): `QuantValue::Q4S`, `QuantLevel::block([32])`,
//! `QuantParam::F32`, `QuantStore::PackedU32(0)` — the only format mummu
//! puts on a device (pack.rs `quantized_tensor_data`). Weights are
//! `[K, N]` row-major with blocks along N; blocks never straddle rows.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "vulkan-spirv")]
use burn::backend::Vulkan;
use burn::backend::backend_extension;
use burn::backend::tensor::{FloatTensor, QuantizedTensor};
use burn::backend::{Backend, Flex, Wgpu};
use burn::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{DType, Tensor};

/// Is a weight tensor in one of the packed formats this module reads?
fn scheme_supported(scheme: &QuantScheme) -> bool {
    let n_blocks = matches!(scheme.value, QuantValue::Q4S | QuantValue::Q8S)
        && scheme.block_size().is_some_and(|b| b.to_dim_vec(1) == [32]);
    (n_blocks || is_kblock(scheme))
        && scheme.scale_dtype() == ScaleDtype::F32
        // PackedU32(0) is what accelerators hold; flex re-tags Native after
        // unpacking to i8 — both are exactly what the per-backend impls read.
        && matches!(scheme.store, QuantStore::PackedU32(0) | QuantStore::Native)
}

/// Input rows per scale in the K-axis layout ([`is_kblock`]) — the source
/// group of a ternary checkpoint (`PQ2_0`/`PTQ1_0` scale every 128 inputs of
/// one output row).
pub const KBLOCK: usize = 128;

/// The native ternary layout: `Q2S` values with `[KBLOCK, 1]` blocks, i.e.
/// one f32 scale per [`KBLOCK`] input rows of ONE output column.
///
/// Why a second layout: the pack's `[32]` blocks run along the output axis,
/// so one scale spans 32 output rows whose source scales differ — exact
/// for nothing ternary (5% error at Q4, measured on Bonsai's `attn_qkv`).
/// Blocks along the input axis are the source's own groups, so `−1/0/+1`
/// codes times the source scale reproduce the checkpoint bit for bit at
/// 2 bits per weight plus one scale per 128.
#[must_use]
pub fn is_kblock(scheme: &QuantScheme) -> bool {
    matches!(scheme.value, QuantValue::Q2S)
        && scheme
            .block_size()
            .is_some_and(|b| b.as_slice() == [128, 1])
}

/// The burn scheme of the native ternary layout ([`is_kblock`]).
#[must_use]
pub fn kblock_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q2S)
        .per_block([128, 1], ScaleDtype::F32)
}

/// Whether the packed path is enabled (`MUMMU_PACKED_GEMV`, default on —
/// `0`/`off`/`false` falls back to burn's dequantize-first matmul
/// everywhere, matching the repo's other default-on switches).
pub fn packed_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MUMMU_PACKED_GEMV").map_or(true, |v| {
            !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        })
    })
}

/// Route one decode-shape matmul through the packed GEMV when everything
/// lines up (m=1, Q4S block-32, path enabled); `None` means the caller
/// should use its existing matmul.
#[must_use]
pub fn try_q4s_gemv(x: &Tensor<2>, w: &Tensor<2>) -> Option<Tensor<2>> {
    let DType::QFloat(scheme) = w.dtype() else {
        return None;
    };
    if !scheme_supported(&scheme) {
        return None;
    }
    let m = x.dims()[0];
    // Ahead of the `MUMMU_PACKED_GEMV` switch on purpose: for this layout
    // the "fallback" it selects computes garbage (see `kblock_matmul`).
    if is_kblock(&scheme) {
        return Some(kblock_matmul(x, w, m));
    }
    if !packed_gemv_enabled() {
        return None;
    }
    if m == 1 {
        let y = <burn::backend::Dispatch as Q4GemvOps>::q4s_gemv(
            x.clone().into_dispatch(),
            w.clone().into_dispatch(),
        );
        return Some(Tensor::from_dispatch(y));
    }
    // Prefill on the HOST: a real packed GEMM (SPEC P5.1) — the weight
    // bytes stream once for the whole batch instead of once per row. The
    // extension impl exists only for Flex, hence the device gate; the cap
    // matches the prefill chunk so a pathological batch shape still finds
    // the generic matmul.
    if (2..=FLEX_GEMM_MAX).contains(&m) && crate::backend::is_flex(&x.device()) {
        let y = <burn::backend::Dispatch as Q4GemmOps>::q4s_gemm(
            x.clone().into_dispatch(),
            w.clone().into_dispatch(),
        );
        return Some(Tensor::from_dispatch(y));
    }
    // Prefill elsewhere: row-by-row through the same exact op. Slower per
    // element than a real GEMM, but it never materializes the f32 weight —
    // the dequantize-first fallback's ~260 MB transient per matmul was the
    // VRAM-pool churn during prefill. Capped so pathological batch shapes
    // keep the old path.
    if m <= 64 {
        let rows: Vec<Tensor<2>> = (0..m)
            .map(|r| {
                let xr = x.clone().slice([r..r + 1, 0..x.dims()[1]]);
                let y = <burn::backend::Dispatch as Q4GemvOps>::q4s_gemv(
                    xr.into_dispatch(),
                    w.clone().into_dispatch(),
                );
                Tensor::from_dispatch(y)
            })
            .collect();
        return Some(Tensor::cat(rows, 0));
    }
    None
}

/// `x · w` for a weight in the K-axis ternary layout, at any batch.
///
/// Never `None`, unlike the block-32 routes: burn's own dequantize reads
/// `[128, 1]` blocks WRONG on the GPU (83% error on CUDA, measured
/// 2026-10-10 with `examples/ternary-gemv-probe`), so the generic
/// `x.matmul(w)` fallback must never see one of these weights. Decode and
/// short spans take the packed GEMV; the host batches through its GEMM;
/// wide spans on a device dequantize with [`KblockDequantOps`] — our own
/// kernel — and multiply in f32, the same transient the block-32 path's
/// dequantize-first matmul pays.
fn kblock_matmul(x: &Tensor<2>, w: &Tensor<2>, m: usize) -> Tensor<2> {
    let gemv = |xr: Tensor<2>| {
        Tensor::from_dispatch(<burn::backend::Dispatch as Q4GemvOps>::q4s_gemv(
            xr.into_dispatch(),
            w.clone().into_dispatch(),
        ))
    };
    if m == 1 {
        return gemv(x.clone());
    }
    if (2..=FLEX_GEMM_MAX).contains(&m) && crate::backend::is_flex(&x.device()) {
        return Tensor::from_dispatch(<burn::backend::Dispatch as Q4GemmOps>::q4s_gemm(
            x.clone().into_dispatch(),
            w.clone().into_dispatch(),
        ));
    }
    if m <= KBLOCK_ROWS_MAX {
        return gemv(x.clone());
    }
    // Column slabs of at most `KBLOCK_SLAB_BYTES` of f32 each. One whole
    // projection at a time ran the 27B out of device memory with every layer
    // on the card (2026-10-10, a 1505-token prompt): the FFN's three f32
    // copies (356 MB each) were live together and the pool reserved 9.0 GiB
    // for 5.7 in use. Slabs bound the transient to a few pool slices.
    let [k_len, width] = w.dims();
    let slab = kblock_slab(k_len, width);
    let parts: Vec<Tensor<2>> = (0..width)
        .step_by(slab)
        .map(|start| {
            let count = slab.min(width - start);
            let wf = Tensor::<2>::from_dispatch(
                <burn::backend::Dispatch as KblockDequantOps>::kblock_dequant(
                    w.clone().into_dispatch(),
                    start,
                    count,
                ),
            );
            x.clone().matmul(wf)
        })
        .collect();
    if parts.len() == 1 {
        parts.into_iter().next().expect("one slab")
    } else {
        Tensor::cat(parts, 1)
    }
}

/// The widest batch the multi-row packed kernel takes; wider spans
/// dequantize and use the tensor cores.
pub const KBLOCK_ROWS_MAX: usize = 1024;

/// The f32 bytes one dequantized prefill slab may hold.
pub const KBLOCK_SLAB_BYTES: usize = 64 << 20;

/// Columns per prefill slab for a `[k_len, width]` ternary weight: as many
/// as fit [`KBLOCK_SLAB_BYTES`] of f32, in whole 256-column steps (never
/// fewer than 256, never more than the weight).
#[must_use]
pub fn kblock_slab(k_len: usize, width: usize) -> usize {
    let fit = KBLOCK_SLAB_BYTES / (k_len.max(1) * 4);
    (fit / 256 * 256).max(256).min(width)
}

#[backend_extension(
    Flex,
    Wgpu,
    Vulkan: cfg(feature = "vulkan-spirv"),
    Cuda: cfg(feature = "cuda"),
)]
/// Dequantize columns of a weight in the K-axis ternary layout
/// ([`is_kblock`]) to f32 — what burn's own `dequantize` gets wrong for
/// these blocks on the GPU backends.
pub trait KblockDequantOps: Backend {
    /// `out[k, j] = q(k, start + j) · scale(k / KBLOCK, start + j)` in f32,
    /// `[K, count]`. `start` and `count` are multiples of the codes per word
    /// (16).
    fn kblock_dequant(w: QuantizedTensor<Self>, start: usize, count: usize) -> FloatTensor<Self>;
}

/// The widest host batch the packed GEMM accepts — the prefill chunk's
/// default. Wider means the caller is doing something the cost model has
/// not seen; the generic matmul takes it.
pub const FLEX_GEMM_MAX: usize = 1024;

#[backend_extension(
    Flex,
    Wgpu,
    Vulkan: cfg(feature = "vulkan-spirv"),
    Cuda: cfg(feature = "cuda"),
)]
/// The packed-GEMV extension op. `x` is `[1, K]` f32, `w` is `[K, N]`
/// `QFloat` (Q4S/Q8S block-32 along N, or the [`is_kblock`] ternary layout;
/// f32 scales, PackedU32(0)); returns `[1, N]` f32.
pub trait Q4GemvOps: Backend {
    /// y = x · w, reading w's packed values and scales directly.
    fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self>;
}

#[backend_extension(Flex)]
/// The packed-GEMM extension op (SPEC P5.1), host only.
///
/// `x` is `[m, K]` f32, `w` as in [`Q4GemvOps`]; returns `[m, N]` f32 with
/// the weight bytes streamed once for the whole batch. Only the Flex impl
/// exists — callers gate on `backend::is_flex` (accelerators keep their own
/// dispatch economics and the split-K GEMV).
pub trait Q4GemmOps: Backend {
    /// Y = X · w over the packed representation.
    fn q4s_gemm(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self>;
}

#[backend_extension(Flex)]
/// The bounded-exact host `lm_head` (SPEC P4.3/P4.4).
///
/// `[1, vocab]` logits whose top-`flex::head::head_k()` coordinates are the
/// exact dense values and every other coordinate is `flex::head::SENTINEL`.
/// Tile bounds (Cauchy–Schwarz + the activation-error aggregate) prove the
/// skipped rows out of the top-k, so argmax and top-k sampling see exactly
/// what the dense head would give them, at a fraction of the streamed bytes.
pub trait Q4HeadOps: Backend {
    /// Sentinel-dense bounded head evaluation.
    fn q4s_head_topk(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self>;
}

/// Route the `lm_head` through the bounded-exact top-k path when everything
/// lines up.
///
/// That is: the path is enabled (serve opts in; the parity harness's
/// full-softmax logprob legs must keep the dense head), the tensor lives
/// on flex, it is a decode-shape call, and the weight is packed Q4.
/// `None` means: use the dense head.
#[must_use]
pub fn try_q4s_head(x: &Tensor<2>, w: &Tensor<2>) -> Option<Tensor<2>> {
    if !crate::flex::head::enabled() || !packed_gemv_enabled() {
        return None;
    }
    if x.dims()[0] != 1 || !crate::backend::is_flex(&x.device()) {
        return None;
    }
    let DType::QFloat(scheme) = w.dtype() else {
        return None;
    };
    if !scheme_supported(&scheme)
        || !matches!(scheme.value, burn::tensor::quantization::QuantValue::Q4S)
    {
        return None;
    }
    let y = <burn::backend::Dispatch as Q4HeadOps>::q4s_head_topk(
        x.clone().into_dispatch(),
        w.clone().into_dispatch(),
    );
    Some(Tensor::from_dispatch(y))
}

// ---------------------------------------------------------------------------
// CubeCL backends (wgpu / vulkan / cuda), non-fusion primitive level.
// ---------------------------------------------------------------------------

mod cube_impl {
    use super::Q4GemvOps;
    use burn::backend::tensor::{FloatTensor, QuantizedTensor};
    use burn::tensor::Shape;
    use burn_cubecl::{
        CubeBackend, CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device,
        tensor::CubeTensor,
    };
    use cubecl::prelude::*;

    /// Bits per packed value: `per_word` is 8 (Q4S) or 4 (Q8S), so this is
    /// 4 or 8 — the shift/mask width the kernel decodes with.
    fn value_bits(per_word: usize) -> u32 {
        u32::try_from(32 / per_word).expect("per_word divides 32")
    }

    /// Every field's sign bit set: `0xAAAA_AAAA` at 2 bits, `0x8888_8888`
    /// at 4, `0x8080_8080` at 8.
    fn sign_pattern(per_word: usize) -> u32 {
        let bits = value_bits(per_word);
        let n = u32::try_from(per_word).expect("at most 32 values per word");
        (0..n).fold(0, |p, j| p | (1u32 << (j * bits + bits - 1)))
    }

    /// The offset [`sign_pattern`] adds to every value, as f32.
    fn f32_offset(per_word: usize) -> f32 {
        f32::from(1u16 << (value_bits(per_word) - 1))
    }

    /// Split-K packed GEMV.
    ///
    /// A workgroup is 32 word-columns wide (`UNIT_POS_X`; coalesced — at any
    /// k the 32 lanes read 32 consecutive u32 words) by `split` k-slices
    /// deep (`UNIT_POS_Y`). Each thread accumulates its word's `per_word`
    /// outputs over `k_len/split` steps in registers; partials meet in shared
    /// memory (32 * split * `per_word` f32 — 16 KiB at split 16, Q4) and the
    /// slice-0 threads reduce and store.
    ///
    /// Why: the split-1 shape gave gate/up 2176 threads on a card with 8448
    /// cores, each walking 5120 dependent FMAs — measured 0.6-0.7x against
    /// burn's dequantize path. `split` multiplies resident threads and
    /// divides the dependent chain by the same factor; what it buys costs
    /// one barrier plus a strided sum, orders of magnitude below the chain
    /// it removes.
    ///
    /// The barrier sits OUTSIDE the validity guard: in a partial workgroup
    /// (`n_words` not a multiple of 32 — test shapes, never the 27B's) every
    /// thread must still reach `sync_cube`, so invalid lanes contribute zero
    /// partials and skip only the final store.
    #[cube(launch)]
    fn packed_gemv_kernel(
        w_packed: &Tensor<u32>,
        scales: &Tensor<f32>,
        x: &Tensor<f32>,
        out: &mut Tensor<f32>,
        #[comptime] per_word: usize,
        #[comptime] split: usize,
    ) {
        let lane = usize::cast_from(UNIT_POS_X);
        let slice = usize::cast_from(UNIT_POS_Y);
        let wc = usize::cast_from(CUBE_POS_X) * 32 + lane;
        let n_words = w_packed.shape(1);
        let valid = wc < n_words;

        let k_len = x.shape(1);
        let bits = comptime!(value_bits(per_word));
        let mask = comptime!((1u32 << value_bits(per_word)) - 1);
        let sign = comptime!(1u32 << (value_bits(per_word) - 1));
        let span = comptime!(1i32 << value_bits(per_word));

        let mut acc = Array::<f32>::new(per_word);
        #[unroll]
        for j in 0..per_word {
            acc[j] = 0.0f32;
        }
        if valid {
            let stride_w = w_packed.stride(0);
            let stride_s = scales.stride(0);
            // 32 values per scale block, `per_word` per u32 word, and words
            // never straddle a block — every value this thread decodes at a
            // given k shares one scale.
            let sc_col = wc / comptime!(32 / per_word);
            let k_per = k_len / split;
            let k0 = slice * k_per;
            for kk in 0..k_per {
                let k = k0 + kk;
                let word = w_packed[k * stride_w + wc];
                let xs = x[k] * scales[k * stride_s + sc_col];
                #[unroll]
                for j in 0..per_word {
                    let raw = (word >> (u32::cast_from(j) * bits)) & mask;
                    let mut q = i32::cast_from(raw);
                    if raw >= sign {
                        q -= span;
                    }
                    // Two roundings on purpose: the same multiply-then-add
                    // the dequantize-then-matmul reference performs.
                    let prod = f32::cast_from(q) * xs;
                    acc[j] += prod;
                }
            }
        }
        if comptime!(split == 1) {
            if valid {
                let base = wc * per_word;
                #[unroll]
                for j in 0..per_word {
                    out[base + j] = acc[j];
                }
            }
        } else {
            let mut partials = Shared::<[f32]>::new_slice(comptime!(32 * split * per_word));
            let slot = (lane * split + slice) * per_word;
            #[unroll]
            for j in 0..per_word {
                partials[slot + j] = acc[j];
            }
            sync_cube();
            if valid && slice == 0 {
                let base = wc * per_word;
                #[unroll]
                for j in 0..per_word {
                    let mut total = 0.0f32;
                    for ss in 0..split {
                        total += partials[(lane * split + ss) * per_word + j];
                    }
                    out[base + j] = total;
                }
            }
        }
    }

    /// Writes a tile's accumulators to `out` rows at `out_row0` (`live` rows
    /// of `rows`, this thread's columns from `col`), folding the in-workgroup
    /// k-slices through shared partials one row at a time.
    #[cube]
    fn kblock_store(
        acc: Array<f32>,
        out: &mut Tensor<f32>,
        out_row0: usize,
        col: usize,
        live: usize,
        #[comptime] per_word: usize,
        #[comptime] split: usize,
        #[comptime] rows: usize,
    ) -> usize {
        let width = out.shape(1);
        let lane = usize::cast_from(UNIT_POS_X);
        let slice = usize::cast_from(UNIT_POS_Y);
        if comptime!(split == 1) {
            #[unroll]
            for r in 0..rows {
                if r < live {
                    #[unroll]
                    for j in 0..per_word {
                        out[out_row0 + r * width + col + j] = acc[r * per_word + j];
                    }
                }
            }
        } else {
            let mut partials = Shared::<[f32]>::new_slice(comptime!(32 * split * per_word));
            let slot = (lane * split + slice) * per_word;
            #[unroll]
            for r in 0..rows {
                #[unroll]
                for j in 0..per_word {
                    partials[slot + j] = acc[r * per_word + j];
                }
                sync_cube();
                if slice == 0 && r < live {
                    #[unroll]
                    for j in 0..per_word {
                        let mut total = 0.0f32;
                        for ss in 0..split {
                            total += partials[(lane * split + ss) * per_word + j];
                        }
                        out[out_row0 + r * width + col + j] = total;
                    }
                }
                sync_cube();
            }
        }
        live
    }

    /// Split-K packed GEMV over the K-axis ternary layout
    /// ([`super::is_kblock`]): the same workgroup shape as
    /// [`packed_gemv_kernel`] (32 coalesced word-columns by `split`
    /// k-slices), but each word holds `per_word` codes of consecutive
    /// OUTPUT columns at one k, and the scale changes with k, not with the
    /// column. So the thread sums `code · x[k]` per column inside a
    /// `kblock` group and folds the group's scale in once when the group
    /// (or its slice) ends — one multiply per column per 128 inputs
    /// instead of per input.
    ///
    /// Any `split` dividing `k_len` is valid: a slice that ends mid-group
    /// flushes its partial group with that group's scale, and the slice
    /// that starts there flushes the rest — the group's sum is split, never
    /// its scale.
    ///
    /// K is ALSO split across the grid (`CUBE_POS_Y`, one `out` row per
    /// grid slice, summed by [`sum_rows_kernel`]). The reason is occupancy:
    /// at 16 codes per word a workgroup covers 512 output columns, so the
    /// 27B's widest projection is 34 workgroups on a 66-SM card — the first
    /// version left half the card idle and ran at ~125 GB/s, slower than Q4
    /// over twice the bytes.
    #[cube(launch)]
    fn kblock_gemv_kernel(
        w_packed: &Tensor<u32>,
        scales: &Tensor<f32>,
        x: &Tensor<f32>,
        out: &mut Tensor<f32>,
        #[comptime] per_word: usize,
        #[comptime] kblock: usize,
        #[comptime] split: usize,
        #[comptime] rows: usize,
    ) {
        let lane = usize::cast_from(UNIT_POS_X);
        let slice = usize::cast_from(UNIT_POS_Y);
        let wc = usize::cast_from(CUBE_POS_X) * 32 + lane;
        let n_words = w_packed.shape(1);
        let valid = wc < n_words;

        let m = x.shape(0);
        let width = scales.shape(1);
        let k_chunk = x.shape(1) / usize::cast_from(CUBE_COUNT_Y);
        let k_base = usize::cast_from(CUBE_POS_Y) * k_chunk;
        let tile = usize::cast_from(CUBE_POS_Z) * rows;
        let part_base = usize::cast_from(CUBE_POS_Y) * m * width;
        let bits = comptime!(value_bits(per_word));
        let mask = comptime!((1u32 << value_bits(per_word)) - 1);
        // Two's complement as offset binary: flipping each field's sign bit
        // maps q to q + 2^(bits-1), so ONE xor per word replaces a
        // compare-and-subtract per value, and the offset — the same for
        // every column at a given k — is taken out once per group as
        // `offset · Σx` at the flush. The decode is what bounded this
        // kernel (~190 GB/s on a ~670 GB/s card), not the bytes.
        let flip = comptime!(sign_pattern(per_word));
        let offset = comptime!(f32_offset(per_word));

        // `rows` rows of x share every decoded word: the weight is read
        // once per row tile, not once per row. Rows past `m` repeat the
        // last row and are not stored.
        let mut xbase = Array::<usize>::new(rows);
        let mut xsum = Array::<f32>::new(rows);
        let mut xs = Array::<f32>::new(rows);
        #[unroll]
        for r in 0..rows {
            let mut row = tile + r;
            if row >= m {
                row = m - 1;
            }
            xbase[r] = row * x.stride(0);
            xsum[r] = 0.0f32;
        }
        let mut acc = Array::<f32>::new(rows * per_word);
        let mut group = Array::<f32>::new(rows * per_word);
        #[unroll]
        for j in 0..rows * per_word {
            acc[j] = 0.0f32;
            group[j] = 0.0f32;
        }
        if valid {
            let stride_w = w_packed.stride(0);
            let stride_s = scales.stride(0);
            let col0 = wc * per_word;
            let k_per = k_chunk / split;
            let k0 = k_base + slice * k_per;
            for kk in 0..k_per {
                let k = k0 + kk;
                let word = w_packed[k * stride_w + wc] ^ flip;
                #[unroll]
                for r in 0..rows {
                    let xk = x[xbase[r] + k];
                    xs[r] = xk;
                    xsum[r] += xk;
                }
                #[unroll]
                for j in 0..per_word {
                    let field = (word >> (u32::cast_from(j) * bits)) & mask;
                    let code = f32::cast_from(field);
                    #[unroll]
                    for r in 0..rows {
                        let prod = code * xs[r];
                        group[r * per_word + j] += prod;
                    }
                }
                if k % kblock == kblock - 1 || kk + 1 == k_per {
                    let srow = (k / kblock) * stride_s + col0;
                    #[unroll]
                    for j in 0..per_word {
                        let scale = scales[srow + j];
                        #[unroll]
                        for r in 0..rows {
                            let shift = offset * xsum[r];
                            let term = (group[r * per_word + j] - shift) * scale;
                            acc[r * per_word + j] += term;
                            group[r * per_word + j] = 0.0f32;
                        }
                    }
                    #[unroll]
                    for r in 0..rows {
                        xsum[r] = 0.0f32;
                    }
                }
            }
        }
        let mut live = m - tile;
        if live > rows {
            live = rows;
        }
        if !valid {
            live = 0;
        }
        let _stored = kblock_store(
            acc,
            out,
            part_base + tile * width,
            wc * per_word,
            live,
            per_word,
            split,
            rows,
        );
    }

    /// `out[0, c] = Σ_y parts[y, c]` — folds [`kblock_gemv_kernel`]'s grid
    /// slices, one thread per column.
    #[cube(launch)]
    fn sum_rows_kernel(parts: &Tensor<f32>, out: &mut Tensor<f32>) {
        let c = ABSOLUTE_POS;
        let n = out.shape(0) * out.shape(1);
        if c < n {
            let mut total = 0.0f32;
            for y in 0..parts.shape(0) {
                total += parts[y * n + c];
            }
            out[c] = total;
        }
    }

    /// A [`kblock_gemv_kernel`] launch shape, packed into the autotune's
    /// one `usize`: the in-workgroup split in the low byte, the grid split
    /// above it.
    const fn kblock_config(split: usize, grid: usize) -> usize {
        split | (grid << 8)
    }

    /// The (split, grid) pairs worth timing for a K-axis weight: every
    /// combination that divides `k_len` and keeps the shared partials
    /// within 32 KiB.
    fn kblock_candidates(k_len: usize, per_word: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for split in [4usize, 8, 16] {
            for grid in [1usize, 2, 4, 8] {
                if k_len.is_multiple_of(split * grid) && 32 * split * per_word * 4 <= 32 << 10 {
                    out.push(kblock_config(split, grid));
                }
            }
        }
        out
    }

    /// Split-K factor (`MUMMU_GEMV_SPLIT`, default 16, max 32: the shared
    /// partial buffer is 32 * split * `per_word` f32 — 16 KiB at (16, Q4)).
    fn gemv_split_override() -> Option<usize> {
        static S: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *S.get_or_init(|| {
            std::env::var("MUMMU_GEMV_SPLIT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| (1..=32).contains(&v))
        })
    }

    /// Candidate split factors, ordered by a CAPACITY prior.
    ///
    /// The prior is residency, not occupancy: a weight that fits the card's
    /// L2 is not starved for memory parallelism, so splitting it only adds
    /// partial-sum traffic and shared-memory pressure. A weight that misses
    /// L2 streams from DRAM, where extra outstanding transactions do buy
    /// bandwidth. `l2_bytes` is the usable fraction (~0.75) of the device's
    /// L2; when it is unknown the prior degrades to "try everything", which
    /// is exactly what the measurement then resolves.
    ///
    /// This ORDERS the search. It never decides. The previous revision
    /// decided — it shipped `split = 16` derived from a Little's-law
    /// occupancy argument, and the card answered that `gate/up` is flat
    /// across every split while `down` gains 2.3x. Two independent peer
    /// reviews made the same derivation. A rule three parties derive and
    /// the hardware refuses is not a rule.
    fn split_candidates(weight_bytes: u64, l2_bytes: Option<u64>) -> Vec<usize> {
        let resident = l2_bytes.is_some_and(|l2| weight_bytes <= l2);
        if resident {
            // L2-resident: 1 first, and only modest splits are worth a try.
            vec![1, 2, 4, 8]
        } else {
            vec![1, 4, 8, 16, 32]
        }
    }

    /// The autotune answers, keyed by `(device, n, k_len, per_word, layout)`.
    type SplitCache = std::collections::HashMap<(String, usize, usize, usize, Layout), usize>;

    /// Which packed layout a split was tuned for — the two kernels share
    /// shapes (a Q8S and a ternary weight can both be `[5120, 17408]`) but
    /// not timings.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Layout {
        /// Block-32 scales along N ([`packed_gemv_kernel`]).
        NBlocks,
        /// `[KBLOCK, 1]` scales along K ([`kblock_gemv_kernel`]).
        KBlocks,
    }

    /// The packed-GEMV problem the split is chosen for: output width `n`,
    /// `n_words` packed weight words, contraction length `k_len`, and the
    /// `per_word` element count implied by the quantization scheme.
    #[derive(Clone, Copy)]
    struct GemvShape {
        n: usize,
        n_words: usize,
        k_len: usize,
        per_word: usize,
    }

    /// The measured best split for one (device, shape, packing), cached for
    /// the process.
    ///
    /// Autotune rather than arithmetic, because the arithmetic was wrong and
    /// because the published numbers for these exact shapes disagree across
    /// GPU generations. Each entry costs a handful of warm launches, once,
    /// and is keyed so a different card or a different projection shape gets
    /// its own answer.
    fn gemv_split_for<R: CubeRuntime>(
        client: &ComputeClient<R>,
        device: &R::Device,
        w_scales: &CubeTensor<R>,
        shape: GemvShape,
        layout: Layout,
        candidates: Option<Vec<usize>>,
        launch: &dyn Fn(usize) -> CubeTensor<R>,
    ) -> usize {
        static CACHE: std::sync::OnceLock<std::sync::Mutex<SplitCache>> =
            std::sync::OnceLock::new();
        let GemvShape {
            n,
            n_words,
            k_len,
            per_word,
        } = shape;
        if layout == Layout::NBlocks
            && let Some(forced) = gemv_split_override()
        {
            return forced;
        }
        let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        let key = (format!("{device:?}"), n, k_len, per_word, layout);
        if let Some(&hit) = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
        {
            return hit;
        }

        let props = client.properties();
        // Usable L2 for a streaming weight; 3/4 is the common working
        // fraction once the activation and output tiles are accounted for.
        let l2 = (props.hardware.max_shared_memory_size as u64)
            .checked_mul(0)
            .and(None::<u64>)
            .or_else(|| {
                std::env::var("MUMMU_L2_MIB")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|m| m << 20)
            })
            .map(|b| b / 4 * 3);
        let weight_bytes =
            (n_words as u64 * k_len as u64 * 4) + (w_scales.meta.num_elements() as u64 * 4);
        let candidates = candidates.unwrap_or_else(|| {
            split_candidates(weight_bytes, l2)
                .into_iter()
                // The shared partials are 32 * split * per_word f32: cap
                // them at 32 KiB.
                .filter(|&c| k_len.is_multiple_of(c) && 32 * c * per_word * 4 <= 32 << 10)
                .collect()
        });
        let mut best = (candidates.first().copied().unwrap_or(1), f64::INFINITY);
        for cand in candidates {
            let run = || launch(cand);
            // Warm (kernel compile, autotune, pool), then time. `sync`
            // blocks until the device drains, which is what makes these
            // numbers comparable — a launch alone returns immediately.
            run();
            let _ = cubecl::future::block_on(client.sync());
            let t0 = std::time::Instant::now();
            for _ in 0..8 {
                run();
            }
            let _ = cubecl::future::block_on(client.sync());
            let ms = t0.elapsed().as_secs_f64() * 1e3 / 8.0;
            if ms < best.1 {
                best = (cand, ms);
            }
        }
        eprintln!(
            "[mummu] gemv split for [{k_len} x {n}] (per_word {per_word}, {layout:?}): {:#x} at {:.3} ms",
            best.0, best.1
        );
        cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, best.0);
        best.0
    }

    /// A packed weight as the GEMV launches read it.
    struct PackedView<'a, R: CubeRuntime> {
        values: &'a CubeTensor<R>,
        scales: &'a CubeTensor<R>,
        /// Logical output width.
        width: usize,
        per_word: usize,
        /// Workgroups along the packed words (32 words each).
        cubes: u32,
    }

    /// One block-32 launch at split `split` ([`packed_gemv_kernel`]).
    fn launch_nblocks<R: CubeRuntime>(
        client: &ComputeClient<R>,
        x: &CubeTensor<R>,
        view: &PackedView<'_, R>,
        split: usize,
    ) -> CubeTensor<R> {
        let out = empty_device::<R, f32>(
            client.clone(),
            x.device.clone(),
            Shape::new([1, view.width]),
        );
        packed_gemv_kernel::launch::<R>(
            client,
            CubeCount::Static(view.cubes, 1, 1),
            CubeDim::new_2d(32, u32::try_from(split).expect("split is at most 32")),
            view.values.clone().into_tensor_arg(),
            view.scales.clone().into_tensor_arg(),
            x.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            view.per_word,
            split,
        );
        out
    }

    /// One K-axis ternary launch at `config` ([`kblock_config`]): the
    /// split-K kernel over the grid's slices, then their sum when the grid
    /// split them.
    fn launch_kblocks<R: CubeRuntime>(
        client: &ComputeClient<R>,
        x: &CubeTensor<R>,
        view: &PackedView<'_, R>,
        config: usize,
    ) -> CubeTensor<R> {
        let (split, grid) = (config & 0xff, (config >> 8).max(1));
        let m = x.meta.shape()[0];
        let rows = kblock_rows(m);
        let out = empty_device::<R, f32>(
            client.clone(),
            x.device.clone(),
            Shape::new([m, view.width]),
        );
        let parts = if grid == 1 {
            out.clone()
        } else {
            empty_device::<R, f32>(
                client.clone(),
                x.device.clone(),
                Shape::new([grid, m, view.width]),
            )
        };
        kblock_gemv_kernel::launch::<R>(
            client,
            CubeCount::Static(
                view.cubes,
                u32::try_from(grid).expect("grid is small"),
                u32::try_from(m.div_ceil(rows)).expect("row tiles fit u32"),
            ),
            CubeDim::new_2d(32, u32::try_from(split).expect("split is at most 16")),
            view.values.clone().into_tensor_arg(),
            view.scales.clone().into_tensor_arg(),
            x.clone().into_tensor_arg(),
            parts.clone().into_tensor_arg(),
            view.per_word,
            super::KBLOCK,
            split,
            rows,
        );
        if grid > 1 {
            sum_rows_kernel::launch::<R>(
                client,
                CubeCount::Static(
                    u32::try_from((m * view.width).div_ceil(256)).expect("width fits u32"),
                    1,
                    1,
                ),
                CubeDim::new_1d(256),
                parts.into_tensor_arg(),
                out.clone().into_tensor_arg(),
            );
        }
        out
    }

    /// Rows of x one workgroup carries through the K-axis kernel: 1 for the
    /// decode GEMV, `MUMMU_KROWS` (default 4) for a batch.
    fn kblock_rows(m: usize) -> usize {
        static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        if m == 1 {
            return 1;
        }
        *ROWS.get_or_init(|| {
            std::env::var("MUMMU_KROWS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| matches!(v, 1 | 2 | 4 | 8))
                .unwrap_or(4)
        })
    }

    pub(super) fn q4s_gemv_cube<R: CubeRuntime>(
        x: CubeTensor<R>,
        w: &CubeTensor<R>,
    ) -> CubeTensor<R> {
        // Values per u32 word, straight off the scheme: 16 codes for Q2S,
        // 8 nibbles for Q4S, 4 bytes for Q8S. Derived, not assumed — the
        // values view's own shape below must agree with it.
        let (per_word, layout) = match w.dtype {
            burn::tensor::DType::QFloat(scheme) => (
                32 / scheme.value.size_bits(),
                if super::is_kblock(&scheme) {
                    Layout::KBlocks
                } else {
                    Layout::NBlocks
                },
            ),
            other => unreachable!("packed gemv on a non-quantized weight: {other:?}"),
        };
        let x = into_contiguous(x);
        let (w_vals, w_scales) = w
            .quantized_handles()
            .expect("q4s_gemv: weight must be a quantized CubeTensor");
        let width = w.meta.shape()[1];
        let n_words = w_vals.meta.shape()[1];
        let client = x.client.clone();
        let k_len = x.meta.shape()[1];
        debug_assert_eq!(
            n_words * per_word,
            width,
            "packed values view must cover exactly the logical width"
        );
        let cubes = u32::try_from(n_words)
            .expect("packed width fits u32")
            .div_ceil(32);
        let launch = |config: usize| {
            let view = PackedView {
                values: &w_vals,
                scales: &w_scales,
                width,
                per_word,
                cubes,
            };
            match layout {
                Layout::NBlocks => launch_nblocks(&client, &x, &view, config),
                Layout::KBlocks => launch_kblocks(&client, &x, &view, config),
            }
        };
        // A batch of rows has the parallelism the single-row GEMV had to
        // split K for: one fixed shape, no tuning pass.
        if layout == Layout::KBlocks && x.meta.shape()[0] > 1 {
            let split = std::env::var("MUMMU_KSPLIT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| matches!(v, 1 | 2 | 4 | 8))
                .unwrap_or(4);
            return launch(kblock_config(
                if k_len.is_multiple_of(split) {
                    split
                } else {
                    1
                },
                1,
            ));
        }
        // Split-K factor: autotuned per shape (or `MUMMU_GEMV_SPLIT`).
        // Forced to 1 when it does not divide k, so odd shapes keep the
        // exact split-1 path.
        let mut split = gemv_split_for(
            &client,
            &x.device,
            &w_scales,
            GemvShape {
                n: width,
                n_words,
                k_len,
                per_word,
            },
            layout,
            (layout == Layout::KBlocks).then(|| kblock_candidates(k_len, per_word)),
            &launch,
        );
        let shared_ok = |s: usize| 32 * s * per_word * 4 <= 32 << 10;
        match layout {
            Layout::NBlocks => {
                if split == 0 || !k_len.is_multiple_of(split) || !shared_ok(split) {
                    split = 1;
                }
            }
            Layout::KBlocks => {
                let (s, g) = (split & 0xff, (split >> 8).max(1));
                if s == 0 || !k_len.is_multiple_of(s * g) || !shared_ok(s) {
                    split = kblock_config(1, 1);
                }
            }
        }
        launch(split)
    }

    /// One thread per packed word: its `per_word` consecutive columns at
    /// one k, scaled by that k's group.
    #[cube(launch)]
    fn kblock_dequant_kernel(
        w_packed: &Tensor<u32>,
        scales: &Tensor<f32>,
        out: &mut Tensor<f32>,
        word0: u32,
        #[comptime] per_word: usize,
        #[comptime] kblock: usize,
    ) {
        let out_words = out.shape(1) / per_word;
        let pos = ABSOLUTE_POS;
        if pos < w_packed.shape(0) * out_words {
            let k = pos / out_words;
            let local = pos % out_words;
            let wc = usize::cast_from(word0) + local;
            let bits = comptime!(value_bits(per_word));
            let mask = comptime!((1u32 << value_bits(per_word)) - 1);
            let flip = comptime!(sign_pattern(per_word));
            let offset = comptime!(f32_offset(per_word));
            let word = w_packed[k * w_packed.stride(0) + wc] ^ flip;
            let srow = (k / kblock) * scales.stride(0) + wc * per_word;
            let orow = k * out.shape(1) + local * per_word;
            #[unroll]
            for j in 0..per_word {
                let field = (word >> (u32::cast_from(j) * bits)) & mask;
                out[orow + j] = (f32::cast_from(field) - offset) * scales[srow + j];
            }
        }
    }

    impl<R: CubeRuntime> super::KblockDequantOps for CubeBackend<R> {
        fn kblock_dequant(
            w: QuantizedTensor<Self>,
            start: usize,
            count: usize,
        ) -> FloatTensor<Self> {
            let per_word = match w.dtype {
                burn::tensor::DType::QFloat(scheme) => 32 / scheme.value.size_bits(),
                other => unreachable!("kblock dequant on a non-quantized weight: {other:?}"),
            };
            let (w_vals, w_scales) = w
                .quantized_handles()
                .expect("kblock_dequant: weight must be a quantized CubeTensor");
            let k = w.meta.shape()[0];
            debug_assert!(
                start.is_multiple_of(per_word) && count.is_multiple_of(per_word),
                "slab must cover whole words"
            );
            debug_assert!(start + count <= w.meta.shape()[1], "slab inside the weight");
            let client = w.client.clone();
            let out = empty_device::<R, f32>(client.clone(), w.device, Shape::new([k, count]));
            let words = k * (count / per_word);
            kblock_dequant_kernel::launch::<R>(
                &client,
                CubeCount::Static(
                    u32::try_from(words.div_ceil(256)).expect("word count fits u32"),
                    1,
                    1,
                ),
                CubeDim::new_1d(256),
                w_vals.into_tensor_arg(),
                w_scales.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                u32::try_from(start / per_word).expect("word index fits u32"),
                per_word,
                super::KBLOCK,
            );
            out
        }
    }

    impl<R: CubeRuntime> Q4GemvOps for CubeBackend<R> {
        fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            q4s_gemv_cube::<R>(x, &w)
        }
    }
}

// ---------------------------------------------------------------------------
// Flex (host): threaded i8 GEMV over the backend's unpacked storage.
// ---------------------------------------------------------------------------

mod flex_impl {
    use super::{Q4GemmOps, Q4GemvOps};
    use burn::backend::tensor::{FloatTensor, QuantizedTensor};
    use burn::backend::{Flex, TensorMetadata};
    use burn::tensor::TensorData;
    use burn::tensor::quantization::QuantValue;
    use burn_flex::FlexTensor;

    /// The i8-slab GEMV over one activation row — the pre-VNNI baseline,
    /// kept as the fallback for non-Q4S weights and as the GEMM's per-row
    /// fallback. Rayon par-chunks: persistent pool threads (a scope-spawn
    /// per call measured its cost — mlp stayed at f32-slab speed), each
    /// owning a disjoint 32-aligned output range and walking every k. The
    /// k-inner loop is written so LLVM vectorizes the i8-widen + FMA.
    fn i8_gemv(xs: &[f32], wq: &[i8], scales: &[f32], k_len: usize, n: usize, out: &mut [f32]) {
        use rayon::prelude::*;
        let blocks = n / 32;
        let chunk_blocks = blocks.div_ceil(rayon::current_num_threads().max(1)).max(4);
        out.par_chunks_mut(chunk_blocks * 32)
            .enumerate()
            .for_each(|(t, chunk)| {
                let b0 = t * chunk_blocks;
                let nb = chunk.len() / 32;
                for k in 0..k_len {
                    let xk = xs[k];
                    let row = &wq[k * n + b0 * 32..k * n + b0 * 32 + nb * 32];
                    let srow = &scales[k * blocks + b0..k * blocks + b0 + nb];
                    for b in 0..nb {
                        let xs_s = xk * srow[b];
                        let src = &row[b * 32..b * 32 + 32];
                        let dst = &mut chunk[b * 32..b * 32 + 32];
                        for j in 0..32 {
                            // Two roundings on purpose: matches the
                            // dequantize-then-matmul reference bit for bit.
                            let prod = xs_s * f32::from(src[j]);
                            dst[j] += prod;
                        }
                    }
                }
            });
    }

    /// The i8-slab GEMV for the K-axis ternary layout (`super::is_kblock`):
    /// scales are `[K / KBLOCK, N]`, one per input group of one output
    /// column. Same partitioning as [`i8_gemv`]; the scale row changes with
    /// k instead of with the column block.
    fn i8_gemv_kblock(
        xs: &[f32],
        wq: &[i8],
        scales: &[f32],
        k_len: usize,
        n: usize,
        out: &mut [f32],
    ) {
        use rayon::prelude::*;
        let cols = n.div_ceil(rayon::current_num_threads().max(1)).max(128);
        out.par_chunks_mut(cols).enumerate().for_each(|(t, chunk)| {
            let c0 = t * cols;
            let w = chunk.len();
            for k in 0..k_len {
                let xk = xs[k];
                let row = &wq[k * n + c0..k * n + c0 + w];
                let srow = &scales[(k / super::KBLOCK) * n + c0..(k / super::KBLOCK) * n + c0 + w];
                for j in 0..w {
                    // Two roundings, as the dequantize-then-matmul reference.
                    let prod = xk * (srow[j] * f32::from(row[j]));
                    chunk[j] += prod;
                }
            }
        });
    }

    /// The activation row(s) as a host slice, borrowing when contiguous.
    fn host_rows(x: &FloatTensor<Flex>) -> Vec<f32> {
        x.as_slice::<f32>().map_or_else(
            || {
                x.clone()
                    .into_data()
                    .try_to_vec::<f32>()
                    .expect("f32 activations")
            },
            <[f32]>::to_vec,
        )
    }

    /// The activation row as a host slice: borrowed when contiguous, else
    /// read back once (the rare non-contiguous case).
    fn host_row(x: &FloatTensor<Flex>) -> std::borrow::Cow<'_, [f32]> {
        x.as_slice::<f32>().map_or_else(
            || {
                std::borrow::Cow::Owned(
                    x.clone()
                        .into_data()
                        .try_to_vec::<f32>()
                        .expect("f32 activations"),
                )
            },
            std::borrow::Cow::Borrowed,
        )
    }

    /// Does this weight qualify for the packed twin (SPEC 1's grid)?
    fn twin_eligible(w: &QuantizedTensor<Flex>, k_len: usize) -> bool {
        matches!(
            w.dtype(),
            burn::tensor::DType::QFloat(s) if matches!(s.value, QuantValue::Q4S)
        ) && k_len.is_multiple_of(32)
            && crate::flex::registry::enabled()
    }

    impl Q4GemvOps for Flex {
        fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            let shape = w.shape();
            let [k_len, n] = shape.dims::<2>();
            let started = std::time::Instant::now();
            let xs = host_row(&x);
            let xs: &[f32] = &xs;
            let wq: &[i8] = w
                .tensor()
                .as_slice::<i8>()
                .expect("flex quantized weight is contiguous i8");
            let scales: &[f32] = w.scales();
            if let burn::tensor::DType::QFloat(s) = w.dtype()
                && super::is_kblock(&s)
            {
                let mut out = vec![0f32; n];
                i8_gemv_kblock(xs, wq, scales, k_len, n, &mut out);
                crate::flex::insitu::record(
                    k_len,
                    n,
                    1,
                    k_len * n + scales.len() * 4,
                    started.elapsed(),
                );
                return FlexTensor::from_data(TensorData::new(out, [1, n]));
            }
            // The packed-nibble VNNI path (SPEC 1): its own host
            // quantization grid (groups along K), so it engages only for
            // Q4S — repacking a Q8S slab through a 4-bit grid would throw
            // away real precision — and only while K divides the group
            // width. MUMMU_VNNI_GEMV=0 restores the i8 loop below.
            if twin_eligible(&w, k_len)
                && let Some(packed) = crate::flex::registry::resolve(wq, scales, k_len, n)
            {
                let mut out = vec![0f32; n];
                crate::flex::kernels::gemv_q4n_auto(&packed, xs, &mut out);
                crate::flex::insitu::record(
                    k_len,
                    n,
                    1,
                    packed.streamed_bytes(),
                    started.elapsed(),
                );
                return FlexTensor::from_data(TensorData::new(out, [1, n]));
            }
            let mut out = vec![0f32; n];
            i8_gemv(xs, wq, scales, k_len, n, &mut out);
            crate::flex::insitu::record(
                k_len,
                n,
                1,
                k_len * n + scales.len() * 4,
                started.elapsed(),
            );
            FlexTensor::from_data(TensorData::new(out, [1, n]))
        }
    }

    impl super::KblockDequantOps for Flex {
        fn kblock_dequant(
            w: QuantizedTensor<Self>,
            start: usize,
            count: usize,
        ) -> FloatTensor<Self> {
            let shape = w.shape();
            let [k_len, width] = shape.dims::<2>();
            let wq: &[i8] = w
                .tensor()
                .as_slice::<i8>()
                .expect("flex quantized weight is contiguous i8");
            let scales: &[f32] = w.scales();
            let mut out = Vec::with_capacity(k_len * count);
            for k in 0..k_len {
                let srow = (k / super::KBLOCK) * width;
                for c in start..start + count {
                    out.push(f32::from(wq[k * width + c]) * scales[srow + c]);
                }
            }
            FlexTensor::from_data(TensorData::new(out, [k_len, count]))
        }
    }

    impl super::Q4HeadOps for Flex {
        fn q4s_head_topk(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            use std::sync::Mutex;
            // A process-global hot set purely improves the visiting order;
            // interleaved requests only degrade the seeds, never the answer.
            static HOT: Mutex<Option<crate::flex::head::HotSet>> = Mutex::new(None);
            let shape = w.shape();
            let [k_len, n] = shape.dims::<2>();
            let started = std::time::Instant::now();
            let xs = host_row(&x);
            let xs: &[f32] = &xs;
            let wq: &[i8] = w
                .tensor()
                .as_slice::<i8>()
                .expect("flex quantized weight is contiguous i8");
            let scales: &[f32] = w.scales();
            let Some(packed) = (twin_eligible(&w, k_len))
                .then(|| crate::flex::registry::resolve(wq, scales, k_len, n))
                .flatten()
            else {
                // No twin: the dense path answers.
                let mut out = vec![0f32; n];
                i8_gemv(xs, wq, scales, k_len, n, &mut out);
                return FlexTensor::from_data(TensorData::new(out, [1, n]));
            };
            let meta = crate::flex::head::meta_for(&packed);
            let k = crate::flex::head::effective_k();
            let seeds = {
                let mut hot = HOT
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                hot.get_or_insert_with(crate::flex::head::HotSet::new)
                    .seeds(n.div_ceil(crate::flex::head::TILE), 4)
            };
            let top = crate::flex::head::head_topk(&packed, &meta, xs, k, &seeds);
            {
                let mut hot = HOT
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(h) = hot.as_mut() {
                    h.observe(xs, &top);
                }
            }
            // The pruning ratio is the whole point — the ledger records the
            // bytes actually streamed, which is what the report divides by.
            let evaluated_bytes = (packed.streamed_bytes() as u128 * top.evaluated_rows as u128
                / n.max(1) as u128) as usize;
            crate::flex::insitu::record(k_len, n, 1, evaluated_bytes, started.elapsed());
            let mut out = vec![crate::flex::head::SENTINEL; n];
            for (&id, &v) in top.ids.iter().zip(&top.vals) {
                out[id as usize] = v;
            }
            FlexTensor::from_data(TensorData::new(out, [1, n]))
        }
    }

    impl Q4GemmOps for Flex {
        fn q4s_gemm(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            let shape = w.shape();
            let [k_len, n] = shape.dims::<2>();
            let m = x.shape().dims::<2>()[0];
            let started = std::time::Instant::now();
            let xs = host_rows(&x);
            assert_eq!(xs.len(), m * k_len, "activation batch shape");
            let wq: &[i8] = w
                .tensor()
                .as_slice::<i8>()
                .expect("flex quantized weight is contiguous i8");
            let scales: &[f32] = w.scales();
            if twin_eligible(&w, k_len)
                && let Some(packed) = crate::flex::registry::resolve(wq, scales, k_len, n)
            {
                let mut out = vec![0f32; m * n];
                crate::flex::kernels::gemm_q4n_auto(&packed, &xs, m, &mut out);
                // The whole point of the GEMM: the weight streams ONCE for
                // the batch — record it that way.
                crate::flex::insitu::record(
                    k_len,
                    n,
                    m,
                    packed.streamed_bytes(),
                    started.elapsed(),
                );
                return FlexTensor::from_data(TensorData::new(out, [m, n]));
            }
            // Non-Q4S (or twin disabled): the i8 loop per row. Correct, and
            // still one op instead of m dispatches.
            let kblock = matches!(
                w.dtype(),
                burn::tensor::DType::QFloat(scheme) if super::is_kblock(&scheme)
            );
            let mut out = vec![0f32; m * n];
            for r in 0..m {
                let (xr, orow) = (
                    &xs[r * k_len..(r + 1) * k_len],
                    &mut out[r * n..(r + 1) * n],
                );
                if kblock {
                    i8_gemv_kblock(xr, wq, scales, k_len, n, orow);
                } else {
                    i8_gemv(xr, wq, scales, k_len, n, orow);
                }
            }
            crate::flex::insitu::record(
                k_len,
                n,
                m,
                (k_len * n + scales.len() * 4) * m,
                started.elapsed(),
            );
            FlexTensor::from_data(TensorData::new(out, [m, n]))
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion wrapper: re-enter the stream with a custom op so `Fusion<B>`
// backends (the default `Wgpu`) hand the inner CubeBackend real tensors.
// ---------------------------------------------------------------------------

#[cfg(feature = "fusion")]
mod fusion_impl {
    use super::{KblockDequantOps, Q4GemvOps};
    use burn::backend::tensor::{FloatTensor, QuantizedTensor};
    use burn::tensor::{DType, Shape};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, StreamId},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

    /// The custom stream op that re-enters the inner backend's packed GEMV.
    #[derive(Clone, Debug)]
    struct Gemv<B> {
        desc: CustomOpIr,
        _b: core::marker::PhantomData<B>,
    }
    impl<B> Gemv<B> {
        const fn new(desc: CustomOpIr) -> Self {
            Self {
                desc,
                _b: core::marker::PhantomData,
            }
        }
    }
    impl<B1: FusionBackend + Q4GemvOps> Operation<B1::FusionRuntime> for Gemv<B1> {
        fn execute(
            &self,
            handles: &mut HandleContainer<<B1::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let ([x_ir, w_ir], [out_ir]) = self.desc.as_fixed();
            let xt = handles.get_float_tensor::<B1>(x_ir);
            let wt = handles.get_quantized_tensor::<B1>(w_ir);
            let y = B1::q4s_gemv(xt, wt);
            handles.register_float_tensor::<B1>(&out_ir.id, y);
        }
    }

    /// The custom stream op that re-enters the inner backend's ternary
    /// dequantize.
    #[derive(Clone, Debug)]
    struct Dequant<B> {
        desc: CustomOpIr,
        start: usize,
        count: usize,
        _b: core::marker::PhantomData<B>,
    }
    impl<B1: FusionBackend + KblockDequantOps> Operation<B1::FusionRuntime> for Dequant<B1> {
        fn execute(
            &self,
            handles: &mut HandleContainer<<B1::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let ([w_ir], [out_ir]) = self.desc.as_fixed();
            let wt = handles.get_quantized_tensor::<B1>(w_ir);
            let y = B1::kblock_dequant(wt, self.start, self.count);
            handles.register_float_tensor::<B1>(&out_ir.id, y);
        }
    }

    impl<B: FusionBackend + KblockDequantOps> KblockDequantOps for Fusion<B> {
        fn kblock_dequant(
            w: QuantizedTensor<Self>,
            start: usize,
            count: usize,
        ) -> FloatTensor<Self> {
            let client = w.client.clone();
            let shape_out = Shape::new([w.shape[0], count]);
            let stream = StreamId::current();
            let out = TensorIr::uninit(client.create_empty_handle(), shape_out, DType::F32);
            let desc = CustomOpIr::new("mummu_kblock_dequant", &[w.into_ir()], &[out]);
            client
                .register(
                    stream,
                    OperationIr::Custom(desc.clone()),
                    Dequant::<B> {
                        desc,
                        start,
                        count,
                        _b: core::marker::PhantomData,
                    },
                )
                .output()
        }
    }

    impl<B: FusionBackend + Q4GemvOps> Q4GemvOps for Fusion<B> {
        fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            let client = x.client.clone();
            let shape_out = Shape::new([x.shape[0], w.shape[1]]);

            let stream = StreamId::current();
            let out = TensorIr::uninit(client.create_empty_handle(), shape_out, DType::F32);
            let desc = CustomOpIr::new("mummu_q4s_gemv", &[x.into_ir(), w.into_ir()], &[out]);
            client
                .register(
                    stream,
                    OperationIr::Custom(desc.clone()),
                    Gemv::<B>::new(desc),
                )
                .output()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor, TensorData};
    use mummu_num::f32_from_u64;

    /// `SplitMix64`: a tiny deterministic generator for test inputs whose
    /// assertion depends on the draw.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in `[-1, 1)`.
        fn uniform_vec(&mut self, len: usize) -> Vec<f32> {
            (0..len)
                .map(|_| f32_from_u64(self.next_u64() >> 40) / f32_from_u64(1u64 << 23) - 1.0)
                .collect()
        }
    }

    /// Serializes the flex-path tests: `force_disable` is process-global and
    /// they set it in opposite directions.
    static FLEX_PATH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Restores the packed host path on drop, so a failing assert cannot
    /// leave the process with the fast path disabled for other tests.
    struct RestoreFastPath;
    impl Drop for RestoreFastPath {
        fn drop(&mut self) {
            crate::flex::registry::force_disable(false);
        }
    }

    /// The baseline i8 GEMV against the dequantize-then-matmul reference on
    /// the host backend — same math, different summation order, so the bound
    /// is tight. The VNNI twin is forced off: it computes on its own host
    /// quantization grid with quantized activations, which is gated by its
    /// own error-bound tests in `flex::kernels`, not by this one.
    #[test]
    fn q4s_gemv_matches_dequant_matmul_on_flex() {
        let _serial = FLEX_PATH_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::flex::registry::force_disable(true);
        let _restore = RestoreFastPath;
        let device = crate::backend::cpu_device();
        let (k, n) = (192, 160);
        let x = Tensor::<2>::random([1, k], Distribution::Uniform(-1.0, 1.0), &device);
        let w = Tensor::<2>::random([k, n], Distribution::Uniform(-1.0, 1.0), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, w);
        assert!(matches!(wq.dtype(), burn::tensor::DType::QFloat(_)));

        let got = try_q4s_gemv(&x, &wq).expect("packed path must engage");
        let want = x.matmul(wq.dequantize());

        let diff = got
            .sub(want.clone())
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        let scale = want.abs().max().into_data().try_to_vec::<f32>().unwrap()[0].max(1e-6);
        assert!(
            diff / scale < 1e-4,
            "packed vs reference rel err {} (abs {diff})",
            diff / scale
        );
    }

    /// The VNNI twin path end to end through `try_q4s_gemv`: deterministic,
    /// and a faithful 4-bit evaluation (its result sits within the combined
    /// requantization + activation budget of the device-grid reference —
    /// a few percent — while the tight per-row bounds are asserted in
    /// `flex::kernels`). K % 64 == 0 here, the production shape class.
    ///
    /// Seeded, because the ratio is a property of the draw, not of the
    /// kernel: the lazy twin re-rounds the device grid along K, and that
    /// second rounding alone spreads the max-abs ratio over uniform draws
    /// at p50 0.035 / p99 0.057 / max 0.086 (20k draws; the activation term
    /// never exceeded 0.01). About 4.5% of unseeded draws cleared 0.05, and
    /// that was this test's flake. This seed measures 0.028; a broken twin
    /// (wrong scales, a ghost weight, a dropped offset) lands at O(1).
    #[test]
    fn q4s_gemv_vnni_twin_is_deterministic_and_faithful() {
        let _serial = FLEX_PATH_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::flex::registry::force_disable(false);
        let device = crate::backend::cpu_device();
        let (k, n) = (256, 96);
        let mut rng = Rng(0);
        let x = Tensor::<2>::from_data(TensorData::new(rng.uniform_vec(k), [1, k]), &device);
        let wf = Tensor::<2>::from_data(TensorData::new(rng.uniform_vec(k * n), [k, n]), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, wf);

        let first = try_q4s_gemv(&x, &wq).expect("packed path must engage");
        let second = try_q4s_gemv(&x, &wq).expect("second call");
        let rep = first
            .clone()
            .sub(second)
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(rep, 0.0, "twin path must be deterministic call to call");

        let want = x.matmul(wq.dequantize());
        let diff = first
            .sub(want.clone())
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        let scale = want.abs().max().into_data().try_to_vec::<f32>().unwrap()[0].max(1e-6);
        assert!(
            diff / scale < 0.05,
            "twin vs device-grid reference rel err {} (abs {diff}) — outside the \
             requant+activation budget",
            diff / scale
        );
    }

    /// A random ternary weight in the K-axis layout: `−1/0/+1` codes
    /// `[k, n]` row-major, scales `[k / KBLOCK, n]` varying per column AND
    /// per group (a test with one shared scale cannot tell the axes apart),
    /// the device tensor, and its exact f32 values.
    fn ternary_weight(
        rng: &mut Rng,
        k: usize,
        n: usize,
        device: &burn::tensor::Device,
    ) -> (Tensor<2>, Vec<f32>) {
        let codes: Vec<i8> = (0..k * n)
            .map(|_| match rng.next_u64() % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let scales: Vec<f32> = rng
            .uniform_vec(k / KBLOCK * n)
            .into_iter()
            .map(|u| 0.01 + u.abs())
            .collect();
        let exact: Vec<f32> = (0..k * n)
            .map(|i| f32::from(codes[i]) * scales[(i / n) / KBLOCK * n + i % n])
            .collect();
        let data = crate::pack::quantized_tensor_data(&codes, &scales, [k, n], kblock_scheme());
        (Tensor::<2>::from_data(data, device), exact)
    }

    /// `y = x · W` in f64 from the exact values — the reference both the
    /// packed path and burn's own dequantize must meet.
    fn reference_gemv(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|c| {
                let sum: f64 = (0..k)
                    .map(|r| f64::from(x[r]) * f64::from(w[r * n + c]))
                    .sum();
                mummu_num::narrow(sum)
            })
            .collect()
    }

    /// The K-axis ternary layout on the host: burn's own dequantize of a
    /// `[128, 1]`-block `Q2S` tensor reproduces the exact values (so the
    /// scale grid is `[k/128, n]` row-major, as the kernels read it), and
    /// the packed GEMV — single row and the batched host GEMM — meets the
    /// f64 reference to summation order.
    #[test]
    fn kblock_ternary_gemv_is_exact_on_flex() {
        let device = crate::backend::cpu_device();
        let (k_len, width) = (384, 96);
        let mut rng = Rng(7);
        let (weight, exact) = ternary_weight(&mut rng, k_len, width, &device);
        let deq = weight
            .clone()
            .dequantize()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(
            deq, exact,
            "burn's dequantize must read the K-axis scale grid"
        );
        let wf = Tensor::<2>::from_dispatch(
            <burn::backend::Dispatch as KblockDequantOps>::kblock_dequant(
                weight.clone().into_dispatch(),
                0,
                width,
            ),
        )
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
        assert_eq!(wf, exact, "the ternary dequant op is exact on the host");
        let slab = Tensor::<2>::from_dispatch(
            <burn::backend::Dispatch as KblockDequantOps>::kblock_dequant(
                weight.clone().into_dispatch(),
                32,
                48,
            ),
        )
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
        let want_slab: Vec<f32> = (0..k_len)
            .flat_map(|k| exact[k * width + 32..k * width + 80].to_vec())
            .collect();
        assert_eq!(slab, want_slab, "a column slab is the same columns");

        let max_err = |got: &[f32], want: &[f32]| {
            got.iter()
                .zip(want)
                .fold(0f32, |acc, (g, r)| acc.max((g - r).abs()))
        };
        let row = rng.uniform_vec(k_len);
        let input = Tensor::<2>::from_data(TensorData::new(row.clone(), [1, k_len]), &device);
        let want = reference_gemv(&row, &exact, k_len, width);
        let got = try_q4s_gemv(&input, &weight)
            .expect("packed path must engage for the ternary layout")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let peak = want.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-6);
        let err = max_err(&got, &want);
        assert!(err / peak < 1e-5, "ternary gemv rel err {}", err / peak);

        let rows = 3;
        let batch = rng.uniform_vec(rows * k_len);
        let input = Tensor::<2>::from_data(TensorData::new(batch.clone(), [rows, k_len]), &device);
        let got = try_q4s_gemv(&input, &weight)
            .expect("host gemm must engage for the ternary layout")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for r in 0..rows {
            let want = reference_gemv(&batch[r * k_len..(r + 1) * k_len], &exact, k_len, width);
            let err = max_err(&got[r * width..(r + 1) * width], &want);
            assert!(err / peak < 1e-4, "row {r} rel err {}", err / peak);
        }
    }

    /// m != 1 and non-Q4S weights must decline, not compute.
    #[test]
    fn q4s_gemv_declines_out_of_contract() {
        let device = crate::backend::cpu_device();
        let x2 = Tensor::<2>::random([2, 64], Distribution::Uniform(-1.0, 1.0), &device);
        let wf = Tensor::<2>::random([64, 64], Distribution::Uniform(-1.0, 1.0), &device);
        assert!(try_q4s_gemv(&x2, &wf).is_none(), "m=2 float must decline");
        let x1 = Tensor::<2>::random([1, 64], Distribution::Uniform(-1.0, 1.0), &device);
        assert!(
            try_q4s_gemv(&x1, &wf).is_none(),
            "float weight must decline"
        );
    }

    /// The host GEMM path (m > 1 on flex) equals m stacked single-row calls
    /// through the same routing — the weight-stream-once evaluation must be
    /// an execution change, not a numerics change.
    #[test]
    fn q4s_gemm_matches_stacked_gemvs_on_flex() {
        let _serial = FLEX_PATH_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::flex::registry::force_disable(false);
        let device = crate::backend::cpu_device();
        let (k, n, m) = (192, 96, 5);
        let x = Tensor::<2>::random([m, k], Distribution::Uniform(-1.0, 1.0), &device);
        let wf = Tensor::<2>::random([k, n], Distribution::Uniform(-1.0, 1.0), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, wf);

        let gemm = try_q4s_gemv(&x, &wq).expect("gemm path must engage");
        assert_eq!(gemm.dims(), [m, n]);
        let rows: Vec<Tensor<2>> = (0..m)
            .map(|r| {
                let xr = x.clone().slice([r..r + 1, 0..k]);
                try_q4s_gemv(&xr, &wq).expect("row path must engage")
            })
            .collect();
        let stacked = Tensor::cat(rows, 0);
        let diff = gemm
            .sub(stacked.clone())
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        let scale = stacked.abs().max().into_data().try_to_vec::<f32>().unwrap()[0].max(1e-6);
        assert!(
            diff / scale < 1e-5,
            "gemm vs stacked gemvs rel err {} (abs {diff})",
            diff / scale
        );
    }

    /// The bounded head end to end through `try_q4s_head`: with k covering
    /// the whole vocab every row is evaluated, so the sentinel-dense output
    /// must equal the dense GEMV bitwise — the plumbing proof (pruning
    /// exactness is pinned at the kernel level in `flex::head`). Disabled
    /// by default: the gate must decline until serve opts in.
    #[test]
    fn bounded_head_scatter_matches_dense() {
        let _serial = FLEX_PATH_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::flex::registry::force_disable(false);
        let device = crate::backend::cpu_device();
        let (k, vocab) = (128, 512);
        let x = Tensor::<2>::random([1, k], Distribution::Uniform(-1.0, 1.0), &device);
        let w = Tensor::<2>::random([k, vocab], Distribution::Uniform(-1.0, 1.0), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, w);

        assert!(
            try_q4s_head(&x, &wq).is_none(),
            "the bounded head must be opt-in"
        );
        crate::flex::head::set_enabled(true);
        let got = try_q4s_head(&x, &wq).expect("enabled head must engage");
        crate::flex::head::set_enabled(false);

        let dense = try_q4s_gemv(&x, &wq).expect("dense twin path");
        // head_k (default 1024) >= vocab: everything is evaluated, so the
        // scatter must reproduce the dense vector exactly.
        let diff = got
            .sub(dense)
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(diff, 0.0, "full-coverage bounded head must equal dense");
    }

    /// The GEMM also serves the i8 fallback (twin off): same equality, so
    /// disabling the VNNI path never disables prefill batching.
    #[test]
    fn q4s_gemm_i8_fallback_matches_stacked_gemvs() {
        let _serial = FLEX_PATH_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::flex::registry::force_disable(true);
        let _restore = RestoreFastPath;
        let device = crate::backend::cpu_device();
        let (k, n, m) = (128, 64, 3);
        let x = Tensor::<2>::random([m, k], Distribution::Uniform(-1.0, 1.0), &device);
        let wf = Tensor::<2>::random([k, n], Distribution::Uniform(-1.0, 1.0), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, wf);

        let gemm = try_q4s_gemv(&x, &wq).expect("gemm path must engage");
        let rows: Vec<Tensor<2>> = (0..m)
            .map(|r| {
                let xr = x.clone().slice([r..r + 1, 0..k]);
                try_q4s_gemv(&xr, &wq).expect("row path must engage")
            })
            .collect();
        let stacked = Tensor::cat(rows, 0);
        let diff = gemm
            .sub(stacked)
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert!(diff < 1e-5, "i8 gemm vs stacked gemvs abs err {diff}");
    }
}
