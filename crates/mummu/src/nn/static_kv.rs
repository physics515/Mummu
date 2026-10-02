//! Fixed-shape decode state: what a captured decode step replays against.
//!
//! A captured graph ([`burn::tensor::capture`]) replays the exact kernels it
//! recorded, against the exact buffers they touched, with every shape and
//! every scalar baked in. The ordinary decode path cannot be captured: its KV
//! cache grows by a `cat` every token (a new buffer, a new shape), its `RoPE`
//! tables and causal mask are uploaded from the host per step, and its
//! attention spans `past + 1` keys. This module is the same arithmetic laid
//! out so that **nothing a step touches changes shape or address**:
//!
//! - the KV cache is preallocated, `[slots, kv_heads, max_ctx, head_dim]`
//!   per layer, and each step's k/v row is written **in place** with an
//!   index-driven `scatter_nd` assignment — the row index is computed on the device
//!   from a positions buffer, so it moves without the graph changing;
//! - attention spans a fixed **bucket** of `len` keys (`narrow`, a view),
//!   with the keys past each slot's position masked out by a mask built on
//!   the device from the same positions buffer (`-1e4`, as
//!   [`super::causal_mask`]: `exp(-1e4)` is exactly 0 after softmax, f16
//!   included). A graph is captured per bucket; a longer context moves to the
//!   next bucket and captures once more;
//! - `RoPE` rows are gathered from full tables uploaded once.
//!
//! `slots` is the batch: every slot is an independent sequence with its own
//! position, which is what lets one dispatch decode several requests (the
//! batching half of the same change).
//!
//! The only inputs a step reads are two small buffers — the tokens and the
//! positions — which the driver refreshes in place between replays. Every
//! constant the step needs (the tables, the row bases, the key-index ramp)
//! is created here, outside any capture window: an upload inside one is
//! refused by the backend.

use burn::tensor::{DType, Device, IndexingUpdateOp, Int, Tensor, TensorData};

use super::rope::rope_tables;

/// Geometry of a static decode state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticKvConfig {
    /// Independent sequences decoded per step.
    pub slots: usize,
    /// Positions each slot can hold (prompt + generation).
    pub max_ctx: usize,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Leading dims of each head `RoPE` rotates (`head_dim` for a full
    /// rotation; qwen35 rotates a prefix and passes the rest through).
    pub rope_dim: usize,
    pub rope_theta: f32,
}

/// Keys a bucket grows by.
///
/// Small enough that a short chat attends to few masked keys (the cost of a
/// bucket is its masked tail), large enough that a long generation captures
/// a handful of graphs, not hundreds.
pub const BUCKET: usize = 256;

/// Per-step tensors every layer shares, derived on the device from the
/// positions buffer (see [`StaticKv::step_inputs`]).
pub struct StepInputs {
    /// `RoPE` rows at each slot's position, `[slots, 1, 1, rope_dim]`.
    pub cos: Tensor<4>,
    pub sin: Tensor<4>,
    /// Flattened cache row each slot's new k/v lands in, `[slots * kv_heads]`.
    pub write_rows: Tensor<1, Int>,
    /// Additive mask over the bucket, `[slots, 1, 1, len]`: 0 for keys at or
    /// before the slot's position, `-1e4` after.
    pub mask: Tensor<4>,
    /// Keys attended this step (the bucket).
    pub len: usize,
}

/// The preallocated cache and constants of a static decode.
pub struct StaticKv {
    cfg: StaticKvConfig,
    /// Per layer, `Some((k, v))` at `[slots, kv_heads, max_ctx, head_dim]`.
    /// An `Option` so a step can move the pair out, write it in place (a
    /// uniquely-held buffer), and put it back.
    layers: Vec<super::LayerKv>,
    /// Full `RoPE` tables, `[max_ctx, rope_dim]`.
    cos: Tensor<2>,
    sin: Tensor<2>,
    /// `(slot * kv_heads + h) * max_ctx` for every (slot, head), the flat row
    /// a slot's position offsets.
    row_base: Tensor<1, Int>,
    /// `0..max_ctx` as one row, `[1, max_ctx]` — narrowed to a bucket per step.
    key_index: Tensor<2, Int>,
    dtype: DType,
}

impl StaticKv {
    /// Allocate the cache (zeroed) and upload the constants.
    ///
    /// # Panics
    ///
    /// When a dimension is zero, `max_ctx` exceeds
    /// [`super::MAX_CONTEXT_TOKENS`], or `rope_dim` is odd or wider than
    /// `head_dim`.
    #[must_use]
    pub fn new(cfg: StaticKvConfig, device: &Device) -> Self {
        assert!(
            cfg.slots >= 1 && cfg.max_ctx >= 1 && cfg.layers >= 1 && cfg.kv_heads >= 1,
            "StaticKv: degenerate geometry {cfg:?}"
        );
        assert!(
            cfg.max_ctx <= super::MAX_CONTEXT_TOKENS,
            "StaticKv: max_ctx {} exceeds MAX_CONTEXT_TOKENS",
            cfg.max_ctx
        );
        assert!(
            cfg.rope_dim >= 2 && cfg.rope_dim.is_multiple_of(2) && cfg.rope_dim <= cfg.head_dim,
            "StaticKv: rope_dim {} must be even and within head_dim {}",
            cfg.rope_dim,
            cfg.head_dim
        );
        // The dynamic cache's storage rule (`kv_append`): f16 when the
        // half-precision KV switch is on over f32 compute, else the compute
        // dtype — so both paths round the same keys the same way.
        let dtype = match crate::backend::float_dtype(device) {
            DType::F32 if super::kv_f16_enabled() => DType::F16,
            compute => compute,
        };
        let shape = [cfg.slots, cfg.kv_heads, cfg.max_ctx, cfg.head_dim];
        let layers = (0..cfg.layers)
            .map(|_| {
                Some((
                    Tensor::<4>::zeros(shape, (device, dtype)),
                    Tensor::<4>::zeros(shape, (device, dtype)),
                ))
            })
            .collect();
        // The very tables the dynamic path builds, for every position at
        // once — same formula per position, so the same values.
        let (cos, sin) = rope_tables(cfg.max_ctx, 0, cfg.rope_dim, cfg.rope_theta, device);
        let cos = cos.reshape([cfg.max_ctx, cfg.rope_dim]);
        let sin = sin.reshape([cfg.max_ctx, cfg.rope_dim]);
        let rows = cfg.slots * cfg.kv_heads;
        let base: Vec<i32> = (0..rows)
            .map(|r| i32::try_from(r * cfg.max_ctx).expect("cache rows fit i32"))
            .collect();
        let int = crate::backend::int_dtype(device);
        let row_base = Tensor::<1, Int>::from_data(TensorData::new(base, [rows]), (device, int));
        let ramp: Vec<i32> = (0..cfg.max_ctx)
            .map(|j| i32::try_from(j).expect("positions fit i32"))
            .collect();
        let key_index =
            Tensor::<1, Int>::from_data(TensorData::new(ramp, [cfg.max_ctx]), (device, int))
                .reshape([1, cfg.max_ctx]);
        Self {
            cfg,
            layers,
            cos,
            sin,
            row_base,
            key_index,
            dtype,
        }
    }

    #[must_use]
    pub const fn config(&self) -> StaticKvConfig {
        self.cfg
    }

    /// The bucket a step needs when the highest position written this step
    /// is `max_pos`: the next multiple of [`BUCKET`] holding keys
    /// `0..=max_pos`, capped at `max_ctx`.
    ///
    /// # Panics
    ///
    /// When `max_pos` is outside the cache.
    #[must_use]
    pub fn bucket_for(&self, max_pos: usize) -> usize {
        bucket(max_pos, self.cfg.max_ctx)
    }

    /// The per-step tensors, from the `[slots]` positions buffer, for a
    /// bucket of `len` keys. Device-side only: safe inside a capture window.
    ///
    /// # Panics
    ///
    /// When `len` is 0 or past `max_ctx`, or `positions` is not `[slots]`.
    #[must_use]
    pub fn step_inputs(&self, positions: &Tensor<1, Int>, len: usize) -> StepInputs {
        let StaticKvConfig {
            slots,
            kv_heads,
            rope_dim,
            max_ctx,
            ..
        } = self.cfg;
        assert!(
            len >= 1 && len <= max_ctx,
            "bucket {len} outside 1..={max_ctx}"
        );
        assert_eq!(positions.dims(), [slots], "positions must be [slots]");
        let cos = self
            .cos
            .clone()
            .select(0, positions.clone())
            .reshape([slots, 1, 1, rope_dim]);
        let sin = self
            .sin
            .clone()
            .select(0, positions.clone())
            .reshape([slots, 1, 1, rope_dim]);
        let write_rows = positions
            .clone()
            .reshape([slots, 1])
            .repeat_dim(1, kv_heads)
            .reshape([slots * kv_heads])
            .add(self.row_base.clone());
        let after = self
            .key_index
            .clone()
            .narrow(1, 0, len)
            .repeat_dim(0, slots)
            .greater(positions.clone().reshape([slots, 1]).repeat_dim(1, len));
        let mask = after
            .float()
            .cast(self.dtype)
            .mul_scalar(-1e4)
            .reshape([slots, 1, 1, len]);
        StepInputs {
            cos,
            sin,
            write_rows,
            mask,
            len,
        }
    }

    /// Room for `max_ctx` positions, every slot's keys and values kept: new
    /// buffers (a captured graph over the old ones must be dropped first)
    /// and the tables for the longer range.
    ///
    /// # Panics
    ///
    /// When `max_ctx` is smaller than the current size, or a layer's pair
    /// is checked out.
    pub fn grow(&mut self, max_ctx: usize) {
        let old = self.cfg.max_ctx;
        assert!(
            max_ctx >= old,
            "grow: {max_ctx} positions is smaller than {old}"
        );
        if max_ctx == old {
            return;
        }
        let device = self.cos.device();
        let mut bigger = Self::new(
            StaticKvConfig {
                max_ctx,
                ..self.cfg
            },
            &device,
        );
        let StaticKvConfig {
            slots,
            kv_heads,
            head_dim,
            ..
        } = self.cfg;
        for (pair, room) in self.layers.iter_mut().zip(&mut bigger.layers) {
            let (k, v) = pair.take().expect("grow: layer checked out");
            let (kb, vb) = room.take().expect("a fresh cache");
            let at = [0..slots, 0..kv_heads, 0..old, 0..head_dim];
            *room = Some((kb.slice_assign(at.clone(), k), vb.slice_assign(at, v)));
        }
        *self = bigger;
    }

    /// Layer `l`'s cache, for a step to take, write and put back.
    ///
    /// # Panics
    ///
    /// When `l` is not a layer.
    pub fn layer_mut(&mut self, l: usize) -> &mut super::LayerKv {
        &mut self.layers[l]
    }

    /// Copy a prefill's k/v (`[1, kv_heads, t, head_dim]`, from the dynamic
    /// path's cache) into `slot` at positions `0..t`. Outside any capture.
    ///
    /// # Panics
    ///
    /// When the shapes disagree with the cache, the prefill is longer than
    /// the cache, or the layer's pair is checked out.
    pub fn seed(&mut self, layer: usize, slot: usize, keys: Tensor<4>, values: Tensor<4>) {
        let StaticKvConfig {
            kv_heads,
            head_dim,
            max_ctx,
            slots,
            ..
        } = self.cfg;
        let [one, heads, len, width] = keys.dims();
        assert!(
            one == 1 && heads == kv_heads && width == head_dim && len <= max_ctx && slot < slots,
            "seed: keys {:?} do not fit slot {slot} of {:?}",
            keys.dims(),
            self.cfg
        );
        assert_eq!(keys.dims(), values.dims(), "seed: k/v shapes differ");
        let (kc, vc) = self.layers[layer].take().expect("seed: layer checked out");
        let at = [slot..slot + 1, 0..kv_heads, 0..len, 0..head_dim];
        let kc = kc.slice_assign(at.clone(), keys.cast(self.dtype));
        let vc = vc.slice_assign(at, values.cast(self.dtype));
        self.layers[layer] = Some((kc, vc));
    }
}

/// The bucket a step needs when the highest position written this step is
/// `max_pos`, in a cache of `max_ctx` positions: the next multiple of
/// [`BUCKET`] holding keys `0..=max_pos`, capped at `max_ctx`.
///
/// # Panics
///
/// When `max_pos` is outside the cache.
#[must_use]
pub fn bucket(max_pos: usize, max_ctx: usize) -> usize {
    assert!(
        max_pos < max_ctx,
        "position {max_pos} is outside a {max_ctx}-position static cache"
    );
    (max_pos + 1)
        .div_ceil(BUCKET)
        .saturating_mul(BUCKET)
        .min(max_ctx)
}

/// A copy of `t` in a buffer of its own (a clone shares the buffer), for a
/// state that must survive the step running again. Synced, so the copy
/// reads `t` before anything later writes it in place.
#[must_use]
pub fn copy_of<const D: usize>(t: &Tensor<D>) -> Tensor<D> {
    let copy = t.clone().add_scalar(0.0);
    let _ = Device::sync(&copy.device());
    copy
}

/// Overwrite all of `dst` with `src`, in place: `dst` keeps its buffer, which
/// is what a captured graph recorded and reads on every replay.
///
/// # Panics
///
/// When `dst` is shared (it could not be written in place — a replay would
/// keep reading the stale buffer), or the shapes differ.
pub fn overwrite<const D: usize>(dst: &mut Tensor<D>, src: Tensor<D>) {
    assert!(
        dst.can_mut(),
        "a static state buffer is shared, so it cannot be written in place"
    );
    let shape = dst.dims();
    assert_eq!(src.dims(), shape, "overwrite: shapes differ");
    let at: [std::ops::Range<usize>; D] = core::array::from_fn(|d| 0..shape[d]);
    let store = dst.dtype();
    dst.inplace(|t| t.slice_assign(at, src.cast(store)));
}

/// Write one step's k/v into the cache and return the bucket's keys and
/// values.
///
/// The new k/v are `[slots, kv_heads, 1, head_dim]`, written at `rows`; the
/// bucket's are `[slots, kv_heads, len, head_dim]`, views of the cache. The
/// write is in place when the cache pair
/// is uniquely held, which the step guarantees by moving it out of the
/// [`StaticKv`] for the duration.
///
/// # Panics
///
/// When the layer's pair is checked out (a step that did not put it back).
pub fn write_and_view(
    kv: &mut super::LayerKv,
    k_new: Tensor<4>,
    v_new: Tensor<4>,
    rows: &Tensor<1, Int>,
    len: usize,
) -> (Tensor<4>, Tensor<4>) {
    let (kc, vc) = kv.take().expect("static cache layer checked out twice");
    let [slots, kv_heads, cap, head_dim] = kc.dims();
    let flat = slots * kv_heads;
    let store = kc.dtype();
    // `scatter_nd` with `Assign`: the one indexed write both backends
    // implement as an assignment (`select_assign` only adds on flex), and
    // the cubecl kernel writes in place when the cache is uniquely held.
    // An assignment is also idempotent — the capture warm-up runs the step
    // three times on the same inputs, and three adds would not be.
    let index = rows.clone().reshape([flat, 1]);
    let put = |cache: Tensor<4>, new: Tensor<4>| {
        cache
            .reshape([flat * cap, head_dim])
            .scatter_nd::<2, 2>(
                index.clone(),
                new.reshape([flat, head_dim]).cast(store),
                IndexingUpdateOp::Assign,
            )
            .reshape([slots, kv_heads, cap, head_dim])
    };
    let kc = put(kc, k_new);
    let vc = put(vc, v_new);
    let keys = kc.clone().narrow(2, 0, len);
    let values = vc.clone().narrow(2, 0, len);
    *kv = Some((kc, vc));
    (keys, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(slots: usize) -> StaticKvConfig {
        StaticKvConfig {
            slots,
            max_ctx: 600,
            layers: 1,
            kv_heads: 2,
            head_dim: 4,
            rope_dim: 4,
            rope_theta: 1e4,
        }
    }

    fn ints(v: &[i32], device: &Device) -> Tensor<1, Int> {
        Tensor::<1, Int>::from_data(
            TensorData::new(v.to_vec(), [v.len()]),
            (device, crate::backend::int_dtype(device)),
        )
    }

    fn vec4(t: Tensor<4>) -> Vec<f32> {
        t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
    }

    #[test]
    fn buckets_round_up_and_cap() {
        let device = crate::backend::cpu_device();
        let kv = StaticKv::new(cfg(1), &device);
        assert_eq!(kv.bucket_for(0), 256);
        assert_eq!(kv.bucket_for(255), 256);
        assert_eq!(kv.bucket_for(256), 512);
        assert_eq!(kv.bucket_for(599), 600, "capped at max_ctx");
    }

    /// The mask lets each slot see exactly keys `0..=pos`, and the `RoPE` rows
    /// are the dynamic path's rows at that position.
    #[test]
    fn step_inputs_mask_and_rope_follow_each_slots_position() {
        let device = crate::backend::cpu_device();
        let kv = StaticKv::new(cfg(2), &device);
        let step = kv.step_inputs(&ints(&[3, 0], &device), 8);
        let mask = vec4(step.mask);
        let open = |slot: usize| (0..8).filter(|&j| mask[slot * 8 + j] == 0.0).count();
        assert_eq!((open(0), open(1)), (4, 1));
        assert!(mask[4] <= -1e4 + 1.0, "key 4 is after slot 0's position");
        let (want_cos, want_sin) = rope_tables(1, 3, 4, 1e4, &device);
        let cos = vec4(step.cos);
        let sin = vec4(step.sin);
        assert_eq!(&cos[..4], &vec4(want_cos)[..]);
        assert_eq!(&sin[..4], &vec4(want_sin)[..]);
        let rows = step
            .write_rows
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap();
        // (slot*kv_heads + h) * max_ctx + pos
        assert_eq!(rows, vec![3, 603, 1200, 1800]);
    }

    /// A write lands at the slot's position, and only there.
    #[test]
    fn write_and_view_writes_one_row_per_slot_and_head() {
        let device = crate::backend::cpu_device();
        let mut kv = StaticKv::new(cfg(2), &device);
        let positions = ints(&[2, 5], &device);
        let step = kv.step_inputs(&positions, 8);
        let dtype = crate::backend::float_dtype(&device);
        let new = Tensor::<4>::ones([2, 2, 1, 4], (&device, dtype));
        let (keys, _values) = write_and_view(
            kv.layer_mut(0),
            new.clone(),
            new.mul_scalar(2.0),
            &step.write_rows,
            8,
        );
        let k = vec4(keys);
        // keys: [slot, head, len=8, hd=4]
        let at = |s: usize, h: usize, j: usize| k[((s * 2 + h) * 8 + j) * 4];
        for h in 0..2 {
            assert_eq!(at(0, h, 2), 1.0);
            assert_eq!(at(1, h, 5), 1.0);
            assert_eq!(at(0, h, 5), 0.0);
            assert_eq!(at(1, h, 2), 0.0);
        }
        assert_eq!(k.iter().filter(|&&x| x > 0.5).count(), 2 * 2 * 4);
    }

    /// Growing keeps every written row where it was and leaves the new
    /// positions zero.
    #[test]
    fn grow_keeps_every_row() {
        let device = crate::backend::cpu_device();
        let mut kv = StaticKv::new(cfg(2), &device);
        let step = kv.step_inputs(&ints(&[2, 599], &device), 600);
        let dtype = crate::backend::float_dtype(&device);
        let new = Tensor::<4>::ones([2, 2, 1, 4], (&device, dtype));
        let _ = write_and_view(kv.layer_mut(0), new.clone(), new, &step.write_rows, 600);
        kv.grow(1000);
        assert_eq!(kv.config().max_ctx, 1000);
        let (k, _) = kv.layer_mut(0).clone().expect("in place");
        assert_eq!(k.dims(), [2, 2, 1000, 4]);
        let k = vec4(k);
        let at = |s: usize, h: usize, j: usize| k[((s * 2 + h) * 1000 + j) * 4];
        for h in 0..2 {
            assert_eq!((at(0, h, 2), at(1, h, 599)), (1.0, 1.0));
            assert_eq!((at(0, h, 600), at(1, h, 999)), (0.0, 0.0));
        }
        assert_eq!(k.iter().filter(|&&x| x > 0.5).count(), 2 * 2 * 4);
        let step = kv.step_inputs(&ints(&[999, 0], &device), 1000);
        assert_eq!(step.cos.dims(), [2, 1, 1, 4], "tables cover the new range");
    }
}
