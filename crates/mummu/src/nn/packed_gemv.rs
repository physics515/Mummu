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
    // Prefill: row-by-row through the same exact op. Slower per element
    // than a real GEMM, but it never materializes the f32 weight — the
    // dequantize-first fallback's ~260 MB transient per matmul was the
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

    /// One unit per u32 word of the packed weight: per k-step it loads the
    /// word (8 nibbles), the one f32 scale those 8 columns share (8 ≤ 32
    /// and words never straddle a block), and x[k], then does 8 fused
    /// mul-adds into private accumulators. No cross-unit reduction; the
    /// unit writes its 8 output columns at the end.
    #[cube(launch)]
    fn packed_gemv_kernel(
        w_packed: &Tensor<u32>,
        scales: &Tensor<f32>,
        x: &Tensor<f32>,
        out: &mut Tensor<f32>,
        #[comptime] per_word: usize,
    ) {
        let wc = ABSOLUTE_POS;
        let n_words = w_packed.shape(1);
        if wc < n_words {
            let k_len = x.shape(1);
            let stride_w = w_packed.stride(0);
            let stride_s = scales.stride(0);
            // 32 values per scale block, `per_word` of them per u32 word, and
            // words never straddle a block — so every value this unit decodes
            // shares one scale.
            let sc_col = wc / comptime!(32 / per_word);
            let bits = comptime!(32u32 / per_word as u32);
            let mask = comptime!((1u32 << (32u32 / per_word as u32)) - 1);
            let sign = comptime!(1u32 << (32u32 / per_word as u32 - 1));
            let span = comptime!(1i32 << (32u32 / per_word as u32));
            let mut acc = Array::<f32>::new(per_word);
            #[unroll]
            for j in 0..per_word {
                acc[j] = 0.0f32;
            }
            for k in 0..k_len {
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
            let base = wc * per_word;
            #[unroll]
            for j in 0..per_word {
                out[base + j] = acc[j];
            }
        }
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
        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
        let cubes = (n_words as u32).div_ceil(256);
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
    use super::Q4GemvOps;
    use burn::backend::tensor::{FloatTensor, QuantizedTensor};
    use burn::backend::{Flex, TensorMetadata};
    use burn::tensor::TensorData;
    use burn_flex::FlexTensor;

    impl Q4GemvOps for Flex {
        fn q4s_gemv(x: FloatTensor<Self>, w: QuantizedTensor<Self>) -> FloatTensor<Self> {
            let shape = w.shape();
            let [k_len, n] = shape.dims::<2>();
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
            let blocks = n / 32;
            let mut out = vec![0f32; n];
            // Rayon par-chunks: persistent pool threads (a scope-spawn per
            // call measured its cost — mlp stayed at f32-slab speed), each
            // owning a disjoint 32-aligned output range and walking every
            // k. The k-inner loop is written so LLVM vectorizes the
            // i8-widen + FMA.
            use rayon::prelude::*;
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
            FlexTensor::from_data(TensorData::new(out, [1, n]))
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

    /// The packed GEMV against the dequantize-then-matmul reference on the
    /// host backend — same math, different summation order, so the bound
    /// is tight.
    #[test]
    fn q4s_gemv_matches_dequant_matmul_on_flex() {
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

    /// m != 1 and non-Q4S weights must decline, not compute.
    #[test]
    fn q4s_gemv_declines_out_of_contract() {
        let device = crate::backend::cpu_device();
        let x2 = Tensor::<2>::random([2, 64], Distribution::Uniform(-1.0, 1.0), &device);
        let wf = Tensor::<2>::random([64, 64], Distribution::Uniform(-1.0, 1.0), &device);
        assert!(try_q4s_gemv(&x2, &wf).is_none(), "m=2 must decline");
        let x1 = Tensor::<2>::random([1, 64], Distribution::Uniform(-1.0, 1.0), &device);
        assert!(try_q4s_gemv(&x1, &wf).is_none(), "float weight must decline");
    }
}
