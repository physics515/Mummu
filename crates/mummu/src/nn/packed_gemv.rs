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

use burn::backend::backend_extension;
use burn::backend::tensor::{FloatTensor, QuantizedTensor};
use burn::backend::{Backend, Flex, Wgpu};
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "vulkan-spirv")]
use burn::backend::Vulkan;
use burn::tensor::{DType, Tensor};
use burn::tensor::quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};

/// Is a weight tensor in the one packed format this module reads?
fn scheme_supported(scheme: &QuantScheme) -> bool {
    matches!(scheme.value, QuantValue::Q4S | QuantValue::Q8S)
        && matches!(scheme.level, QuantLevel::Block(b) if b.to_dim_vec(1) == [32])
        && matches!(scheme.param, QuantParam::F32)
        // PackedU32(0) is what accelerators hold; flex re-tags Native after
        // unpacking to i8 — both are exactly what the per-backend impls read.
        && matches!(scheme.store, QuantStore::PackedU32(0) | QuantStore::Native)
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
pub fn try_q4s_gemv(x: &Tensor<2>, w: &Tensor<2>) -> Option<Tensor<2>> {
    if !packed_gemv_enabled() {
        return None;
    }
    let DType::QFloat(scheme) = w.dtype() else {
        return None;
    };
    if !scheme_supported(&scheme) {
        return None;
    }
    let m = x.dims()[0];
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
/// QFloat (Q4S, block-32, f32 scales, PackedU32(0)); returns `[1, N]` f32.
pub trait Q4GemvOps: Backend {
    /// y = x · w, reading w's packed values and scales directly.
    fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self>;
}

#[backend_extension(Flex)]
/// The packed-GEMM extension op (SPEC P5.1), host only: `x` is `[m, K]`
/// f32, `w` as in [`Q4GemvOps`]; returns `[m, N]` f32 with the weight
/// bytes streamed once for the whole batch. Only the Flex impl exists —
/// callers gate on `backend::is_flex` (accelerators keep their own
/// dispatch economics and the split-K GEMV).
pub trait Q4GemmOps: Backend {
    /// Y = X · w over the packed representation.
    fn q4s_gemm(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self>;
}

#[backend_extension(Flex)]
/// The bounded-exact host lm_head (SPEC P4.3/P4.4): `[1, vocab]` logits
/// whose top-`flex::head::head_k()` coordinates are the exact dense values
/// and every other coordinate is `flex::head::SENTINEL`. Tile bounds
/// (Cauchy–Schwarz + the activation-error aggregate) prove the skipped
/// rows out of the top-k, so argmax and top-k sampling see exactly what
/// the dense head would give them, at a fraction of the streamed bytes.
pub trait Q4HeadOps: Backend {
    /// Sentinel-dense bounded head evaluation.
    fn q4s_head_topk(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self>;
}

/// Route the lm_head through the bounded-exact top-k path when everything
/// lines up: the path is enabled (serve opts in; the parity harness's
/// full-softmax logprob legs must keep the dense head), the tensor lives
/// on flex, it is a decode-shape call, and the weight is packed Q4.
/// `None` means: use the dense head.
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

    /// Split-K packed GEMV.
    ///
    /// A workgroup is 32 word-columns wide (UNIT_POS_X; coalesced — at any
    /// k the 32 lanes read 32 consecutive u32 words) by `split` k-slices
    /// deep (UNIT_POS_Y). Each thread accumulates its word's `per_word`
    /// outputs over k_len/split steps in registers; partials meet in shared
    /// memory (32 * split * per_word f32 — 16 KiB at split 16, Q4) and the
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
    /// (n_words not a multiple of 32 — test shapes, never the 27B's) every
    /// thread must still reach sync_cube, so invalid lanes contribute zero
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
        let bits = comptime!(32u32 / per_word as u32);
        let mask = comptime!((1u32 << (32u32 / per_word as u32)) - 1);
        let sign = comptime!(1u32 << (32u32 / per_word as u32 - 1));
        let span = comptime!(1i32 << (32u32 / per_word as u32));

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
                    acc[j] += f32::cast_from(q) * xs;
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

    /// Split-K factor (`MUMMU_GEMV_SPLIT`, default 16, max 32: the shared
    /// partial buffer is 32 * split * per_word f32 — 16 KiB at (16, Q4)).
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
        w_vals: &CubeTensor<R>,
        w_scales: &CubeTensor<R>,
        x: &CubeTensor<R>,
        n: usize,
        n_words: usize,
        k_len: usize,
        per_word: usize,
    ) -> usize {
        if let Some(forced) = gemv_split_override() {
            return forced;
        }
        static CACHE: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<(String, usize, usize, usize), usize>>,
        > = std::sync::OnceLock::new();
        let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        let key = (format!("{device:?}"), n, k_len, per_word);
        if let Some(&hit) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return hit;
        }

        let props = client.properties();
        // Usable L2 for a streaming weight; 3/4 is the common working
        // fraction once the activation and output tiles are accounted for.
        let l2 = u64::from(props.hardware.max_shared_memory_size as u32)
            .checked_mul(0)
            .and(None::<u64>)
            .or_else(|| std::env::var("MUMMU_L2_MIB").ok().and_then(|v| v.parse::<u64>().ok()).map(|m| m << 20))
            .map(|b| b / 4 * 3);
        let weight_bytes = (n_words as u64 * k_len as u64 * 4) + (w_scales.meta.num_elements() as u64 * 4);

        let mut best = (1usize, f64::INFINITY);
        for cand in split_candidates(weight_bytes, l2) {
            if k_len % cand != 0 {
                continue;
            }
            let run = || {
                let out = empty_device::<R, f32>(
                    client.clone(),
                    x.device.clone(),
                    Shape::new([1, n]),
                );
                packed_gemv_kernel::launch::<R>(
                    client,
                    CubeCount::Static((n_words as u32).div_ceil(32), 1, 1),
                    CubeDim { x: 32, y: cand as u32, z: 1 },
                    w_vals.clone().into_tensor_arg(),
                    w_scales.clone().into_tensor_arg(),
                    x.clone().into_tensor_arg(),
                    out.clone().into_tensor_arg(),
                    per_word,
                    cand,
                );
                out
            };
            // Warm (kernel compile, autotune, pool), then time. `sync`
            // blocks until the device drains, which is what makes these
            // numbers comparable — a launch alone returns immediately.
            run();
            let _ = cubecl::future::block_on(client.sync());
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let _ = cubecl::future::block_on(client.sync());
            let ms = t0.elapsed().as_secs_f64() * 1e3 / 3.0;
            if ms < best.1 {
                best = (cand, ms);
            }
        }
        eprintln!(
            "[mummu] gemv split for [{k_len} x {n}] (per_word {per_word}): {} at {:.3} ms",
            best.0, best.1
        );
        cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, best.0);
        best.0
    }

    pub(super) fn q4s_gemv_cube<R: CubeRuntime>(
        x: CubeTensor<R>,
        w: CubeTensor<R>,
    ) -> CubeTensor<R> {
        // Values per u32 word, straight off the scheme: 8 nibbles for Q4S,
        // 4 bytes for Q8S. Derived, not assumed — the values view's own
        // shape below must agree with it.
        let per_word: usize = match w.dtype {
            burn::tensor::DType::QFloat(scheme) => 32 / scheme.value.size_bits() as usize,
            other => unreachable!("packed gemv on a non-quantized weight: {other:?}"),
        };
        let x = into_contiguous(x);
        let (w_vals, w_scales) = w
            .quantized_handles()
            .expect("q4s_gemv: weight must be a quantized CubeTensor");
        let n = w.meta.shape()[1];
        let n_words = w_vals.meta.shape()[1];
        let client = x.client.clone();
        let device = x.device.clone();
        let out = empty_device::<R, f32>(client.clone(), device, Shape::new([1, n]));
        // Split-K factor: `MUMMU_GEMV_SPLIT`, default 16 — near the
        // occupancy knee for both production shapes. Forced to 1 when it
        // does not divide k, so odd shapes keep the exact split-1 path.
        let k_len = x.meta.shape()[1];
        let mut split = gemv_split_for(
            &client,
            &x.device,
            &w_vals,
            &w_scales,
            &x,
            n,
            n_words,
            k_len,
            per_word,
        );
        if split == 0 || k_len % split != 0 {
            split = 1;
        }
        let cube_dim = CubeDim {
            x: 32,
            y: split as u32,
            z: 1,
        };
        let cubes = (n_words as u32).div_ceil(32);
        debug_assert_eq!(
            n_words * per_word,
            n,
            "packed values view must cover exactly the logical width"
        );
        packed_gemv_kernel::launch::<R>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            cube_dim,
            w_vals.into_tensor_arg(),
            w_scales.into_tensor_arg(),
            x.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            per_word,
            split,
        );
        out
    }

    impl<R: CubeRuntime> Q4GemvOps for CubeBackend<R> {
        fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            q4s_gemv_cube::<R>(x, w)
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
                            dst[j] += xs_s * f32::from(src[j]);
                        }
                    }
                }
            });
    }

    /// The activation row(s) as a host slice, borrowing when contiguous.
    fn host_rows(x: &FloatTensor<Flex>) -> Vec<f32> {
        match x.as_slice::<f32>() {
            Some(s) => s.to_vec(),
            None => x.clone().into_data().to_vec::<f32>().expect("f32 activations"),
        }
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
            let xs_owned;
            let xs: &[f32] = match x.as_slice::<f32>() {
                Some(s) => s,
                None => {
                    xs_owned = x.into_data().to_vec::<f32>().expect("f32 activations");
                    &xs_owned
                }
            };
            let wq: &[i8] = w
                .tensor()
                .as_slice::<i8>()
                .expect("flex quantized weight is contiguous i8");
            let scales: &[f32] = w.scales();
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
                crate::flex::insitu::record(k_len, n, 1, packed.streamed_bytes(), started.elapsed());
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

    impl super::Q4HeadOps for Flex {
        fn q4s_head_topk(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            use std::sync::Mutex;
            let shape = w.shape();
            let [k_len, n] = shape.dims::<2>();
            let started = std::time::Instant::now();
            let xs_owned;
            let xs: &[f32] = match x.as_slice::<f32>() {
                Some(s) => s,
                None => {
                    xs_owned = x.into_data().to_vec::<f32>().expect("f32 activations");
                    &xs_owned
                }
            };
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
            // A process-global hot set purely improves the visiting order;
            // interleaved requests only degrade the seeds, never the answer.
            static HOT: Mutex<Option<crate::flex::head::HotSet>> = Mutex::new(None);
            let k = crate::flex::head::effective_k();
            let seeds = {
                let mut hot = HOT.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                hot.get_or_insert_with(crate::flex::head::HotSet::new)
                    .seeds(n.div_ceil(crate::flex::head::TILE), 4)
            };
            let top = crate::flex::head::head_topk(&packed, &meta, xs, k, &seeds);
            {
                let mut hot = HOT.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(h) = hot.as_mut() {
                    h.observe(xs, &top);
                }
            }
            // The pruning ratio is the whole point — the ledger records the
            // bytes actually streamed, which is what the report divides by.
            let evaluated_bytes = (packed.streamed_bytes() as u128
                * top.evaluated_rows as u128
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
                crate::flex::insitu::record(k_len, n, m, packed.streamed_bytes(), started.elapsed());
                return FlexTensor::from_data(TensorData::new(out, [m, n]));
            }
            // Non-Q4S (or twin disabled): the i8 loop per row. Correct, and
            // still one op instead of m dispatches.
            let mut out = vec![0f32; m * n];
            for r in 0..m {
                let (xr, orow) = (
                    &xs[r * k_len..(r + 1) * k_len],
                    &mut out[r * n..(r + 1) * n],
                );
                i8_gemv(xr, wq, scales, k_len, n, orow);
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
    use super::Q4GemvOps;
    use burn::backend::tensor::{FloatTensor, QuantizedTensor};
    use burn::tensor::{DType, Shape};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, StreamId},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

    impl<B: FusionBackend + Q4GemvOps> Q4GemvOps for Fusion<B> {
        fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            let client = x.client.clone();
            let shape_out = Shape::new([x.shape[0], w.shape[1]]);

            #[derive(derive_new::new, Clone, Debug)]
            struct Gemv<B> {
                desc: CustomOpIr,
                _b: core::marker::PhantomData<B>,
            }
            impl<B1: FusionBackend + Q4GemvOps> Operation<B1::FusionRuntime> for Gemv<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([x_ir, w_ir], [out_ir]) = self.desc.as_fixed();
                    let xt = handles.get_float_tensor::<B1>(x_ir);
                    let wt = handles.get_quantized_tensor::<B1>(w_ir);
                    let y = B1::q4s_gemv(xt, wt);
                    handles.register_float_tensor::<B1>(&out_ir.id, y);
                }
            }

            let stream = StreamId::current();
            let out = TensorIr::uninit(client.create_empty_handle(), shape_out, DType::F32);
            let desc = CustomOpIr::new("mummu_q4s_gemv", &[x.into_ir(), w.into_ir()], &[out]);
            client
                .register(stream, OperationIr::Custom(desc.clone()), Gemv::<B>::new(desc))
                .output()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};

    /// Serializes the two flex-path tests: `force_disable` is process-global
    /// and they set it in opposite directions.
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
        let _serial = FLEX_PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
            .to_vec::<f32>()
            .unwrap()[0];
        let scale = want.abs().max().into_data().to_vec::<f32>().unwrap()[0].max(1e-6);
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
    #[test]
    fn q4s_gemv_vnni_twin_is_deterministic_and_faithful() {
        let _serial = FLEX_PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::flex::registry::force_disable(false);
        let device = crate::backend::cpu_device();
        let (k, n) = (256, 96);
        let x = Tensor::<2>::random([1, k], Distribution::Uniform(-1.0, 1.0), &device);
        let w = Tensor::<2>::random([k, n], Distribution::Uniform(-1.0, 1.0), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, w);

        let a = try_q4s_gemv(&x, &wq).expect("packed path must engage");
        let b = try_q4s_gemv(&x, &wq).expect("second call");
        let rep = a
            .clone()
            .sub(b)
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(rep, 0.0, "twin path must be deterministic call to call");

        let want = x.matmul(wq.dequantize());
        let diff = a.sub(want.clone()).abs().max().into_data().to_vec::<f32>().unwrap()[0];
        let scale = want.abs().max().into_data().to_vec::<f32>().unwrap()[0].max(1e-6);
        assert!(
            diff / scale < 0.05,
            "twin vs device-grid reference rel err {} (abs {diff}) — outside the \
             requant+activation budget",
            diff / scale
        );
    }

    /// m != 1 and non-Q4S weights must decline, not compute.
    #[test]
    fn q4s_gemv_declines_out_of_contract() {
        let device = crate::backend::cpu_device();
        let x2 = Tensor::<2>::random([2, 64], Distribution::Uniform(-1.0, 1.0), &device);
        let wf = Tensor::<2>::random([64, 64], Distribution::Uniform(-1.0, 1.0), &device);
        assert!(try_q4s_gemv(&x2, &wf).is_none(), "m=2 float must decline");
        let x1 = Tensor::<2>::random([1, 64], Distribution::Uniform(-1.0, 1.0), &device);
        assert!(try_q4s_gemv(&x1, &wf).is_none(), "float weight must decline");
    }

    /// The host GEMM path (m > 1 on flex) equals m stacked single-row calls
    /// through the same routing — the weight-stream-once evaluation must be
    /// an execution change, not a numerics change.
    #[test]
    fn q4s_gemm_matches_stacked_gemvs_on_flex() {
        let _serial = FLEX_PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::flex::registry::force_disable(false);
        let device = crate::backend::cpu_device();
        let (k, n, m) = (192, 96, 5);
        let x = Tensor::<2>::random([m, k], Distribution::Uniform(-1.0, 1.0), &device);
        let w = Tensor::<2>::random([k, n], Distribution::Uniform(-1.0, 1.0), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, w);

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
            .to_vec::<f32>()
            .unwrap()[0];
        let scale = stacked.abs().max().into_data().to_vec::<f32>().unwrap()[0].max(1e-6);
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
        let _serial = FLEX_PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
            .to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(diff, 0.0, "full-coverage bounded head must equal dense");
    }

    /// The GEMM also serves the i8 fallback (twin off): same equality, so
    /// disabling the VNNI path never disables prefill batching.
    #[test]
    fn q4s_gemm_i8_fallback_matches_stacked_gemvs() {
        let _serial = FLEX_PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::flex::registry::force_disable(true);
        let _restore = RestoreFastPath;
        let device = crate::backend::cpu_device();
        let (k, n, m) = (128, 64, 3);
        let x = Tensor::<2>::random([m, k], Distribution::Uniform(-1.0, 1.0), &device);
        let w = Tensor::<2>::random([k, n], Distribution::Uniform(-1.0, 1.0), &device);
        let wq = crate::quant::quantize_weight(crate::quant::QuantPolicy::Q4, w);

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
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(diff < 1e-5, "i8 gemm vs stacked gemvs abs err {diff}");
    }
}
