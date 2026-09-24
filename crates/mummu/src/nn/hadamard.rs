//! **Prism's Hadamard weight basis** — the activation-side half of a
//! rotation folded into stored weights (Ternary-Bonsai 2).
//!
//! A checkpoint can store each linear weight in a rotated basis: with `R`
//! orthogonal, `y = W x = (W Rᵀ)(R x)`, so the file holds `W' = W Rᵀ` and
//! the runtime feeds it `R x` instead of `x`. Prism's `R` is blockwise —
//! the input axis is cut into `block`-wide pieces and each piece is
//! multiplied by a fixed ±1 sign per position and then by the normalized
//! Sylvester Walsh–Hadamard matrix `H = (1/√n)·[(−1)^{popcount(i & j)}]`
//! — which spreads outliers so a 2-bit ternary quantization stays
//! near-lossless (the same idea `mummu_mix::hadamard` measures for imports).
//! The rotation costs no weight bits; what it costs the runtime is one
//! sign-multiply and one `[·, n]·[n, n]` matmul per folded projection input.
//!
//! The GGUF declares the contract under `prism.hadamard.*` (version 1):
//! the block size, the transform and axis names, the sign mode with the
//! explicit sign vectors per input width, the folded weight names, the
//! lookup tables stored in the rotated basis (`inverse_weight_names` — the
//! token embedding, whose rows come out rotated and are restored with the
//! inverse right after the gather), and `gdn_v_grouped`, which says the
//! `DeltaNet` output projection was folded over the value heads in HF's
//! grouped order while the runtime's activation arrives in llama.cpp's
//! tiled order (see [`tiled_to_grouped`]). Everything here mirrors the
//! Prism llama.cpp fork (`llama-model.cpp` parses the keys; `build_lora_mm`
//! applies signs then `llama_mul_mat_hadamard`; `build_embd_rows` applies
//! the inverse as H then signs), and an unknown or unverifiable contract is
//! a load error — never a silently unrotated forward.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use burn::tensor::{Device, Tensor, TensorData};
use mummu_num::f32_from_usize;

use crate::gguf::{GgufFile, GgufValue};

/// The metadata prefix every key of the contract carries.
const PREFIX: &str = "prism.hadamard.";
/// The only contract version this runtime implements.
const VERSION: u64 = 1;

/// The full metadata key for `k`.
fn key(k: &str) -> String {
    format!("{PREFIX}{k}")
}

/// `prism.hadamard.<k>` as a string.
fn str_key<'a>(f: &'a GgufFile, k: &str) -> Result<&'a str, String> {
    f.get(&key(k))
        .and_then(GgufValue::as_str)
        .ok_or_else(|| format!("prism.hadamard.{k} missing or not a string"))
}

/// `prism.hadamard.<k>` as an unsigned integer (a non-negative signed
/// value counts).
fn uint_key(f: &GgufFile, k: &str) -> Result<u64, String> {
    let v = f
        .get(&key(k))
        .ok_or_else(|| format!("prism.hadamard.{k} missing"))?;
    v.as_u64()
        .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
        .ok_or_else(|| format!("prism.hadamard.{k} is not an integer"))
}

/// `prism.hadamard.<k>` as a string array; an absent key is empty unless
/// `required`.
fn str_array(f: &GgufFile, k: &str, required: bool) -> Result<Vec<String>, String> {
    match f.get(&key(k)) {
        None if !required => Ok(Vec::new()),
        None => Err(format!("prism.hadamard.{k} missing")),
        Some(v) => v
            .as_array()
            .ok_or_else(|| format!("prism.hadamard.{k} is not an array"))?
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("prism.hadamard.{k} holds a non-string"))
            })
            .collect(),
    }
}

/// `prism.hadamard.<k>` as an integer array.
fn int_array(f: &GgufFile, k: &str) -> Result<Vec<i64>, String> {
    f.get(&key(k))
        .ok_or_else(|| format!("prism.hadamard.{k} missing"))?
        .as_array()
        .ok_or_else(|| format!("prism.hadamard.{k} is not an array"))?
        .iter()
        .map(|v| {
            v.as_i64()
                .or_else(|| v.as_u64().and_then(|u| i64::try_from(u).ok()))
                .ok_or_else(|| format!("prism.hadamard.{k} holds a non-integer"))
        })
        .collect()
}

/// The explicit sign table: `sign_values` cut into one ±1 vector per
/// `sign_widths` entry, each width a positive multiple of `block`.
fn explicit_signs(
    widths: &[i64],
    values: &[i64],
    block: usize,
) -> Result<BTreeMap<usize, Vec<f32>>, String> {
    if widths.is_empty() {
        return Err("prism.hadamard.sign_mode is explicit but sign_widths is empty".into());
    }
    let mut signs = BTreeMap::new();
    let mut off = 0usize;
    for &w in widths {
        let width = usize::try_from(w)
            .ok()
            .filter(|&w| w > 0 && w.is_multiple_of(block))
            .ok_or_else(|| {
                format!(
                    "prism.hadamard sign width {w} is not a positive multiple of the block {block}"
                )
            })?;
        let end = off
            .checked_add(width)
            .filter(|&e| e <= values.len())
            .ok_or("prism.hadamard.sign_values is shorter than sign_widths declares")?;
        let vec: Vec<f32> = values[off..end]
            .iter()
            .map(|&v| match v {
                1 => Ok(1.0f32),
                -1 => Ok(-1.0f32),
                other => Err(format!("prism.hadamard sign value {other} is not ±1")),
            })
            .collect::<Result<_, _>>()?;
        if signs.insert(width, vec).is_some() {
            return Err(format!("prism.hadamard declares width {width} twice"));
        }
        off = end;
    }
    if off != values.len() {
        return Err("prism.hadamard.sign_values length does not match sign_widths".into());
    }
    Ok(signs)
}

/// The `prism.hadamard.*` contract as a GGUF declares it, validated.
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardSpec {
    /// Width of one rotated block along the input axis (a power of two).
    pub block: usize,
    /// Per input width, the ±1 sign applied before the transform. Empty in
    /// `identity` sign mode; in `explicit` mode every folded width must
    /// have an entry (checked when the transform is built).
    pub signs: BTreeMap<usize, Vec<f32>>,
    /// GGUF names of the folded weights: their matmul INPUT is transformed.
    pub weights: BTreeSet<String>,
    /// GGUF names of lookup tables stored rotated: the lookup RESULT is
    /// inverse-transformed.
    pub inverses: BTreeSet<String>,
    /// The `DeltaNet` `ssm_out` fold was computed over grouped value heads.
    pub gdn_v_grouped: bool,
}

impl HadamardSpec {
    /// Parse the contract from a header. `Ok(None)` when the file declares
    /// none (an ordinary checkpoint).
    ///
    /// # Errors
    ///
    /// An error for a contract this runtime cannot honour — a different
    /// version, transform, axis or sign mode, a block that is not a power
    /// of two, sign vectors that do not cut into whole blocks or hold
    /// anything but ±1, a missing or malformed key, or a weight name listed
    /// twice.
    pub fn from_gguf(f: &GgufFile) -> Result<Option<Self>, String> {
        if f.get(&key("version")).is_none() {
            return Ok(None);
        }
        let version = uint_key(f, "version")?;
        if version != VERSION {
            return Err(format!(
                "prism.hadamard.version {version} is not supported (this runtime implements {VERSION})"
            ));
        }
        let block =
            usize::try_from(uint_key(f, "block_size")?).map_err(|_| "block_size too large")?;
        if block == 0 || !block.is_power_of_two() {
            return Err(format!(
                "prism.hadamard.block_size {block} is not a power of two"
            ));
        }
        let transform = str_key(f, "transform")?;
        if transform != "normalized-sylvester-walsh-hadamard" {
            return Err(format!(
                "prism.hadamard.transform {transform:?} is not supported (only normalized-sylvester-walsh-hadamard)"
            ));
        }
        let axis = str_key(f, "axis")?;
        if axis != "input-last-dimension" {
            return Err(format!(
                "prism.hadamard.axis {axis:?} is not supported (only input-last-dimension)"
            ));
        }
        let sign_mode = str_key(f, "sign_mode")?;
        let signs = match sign_mode {
            "identity" => BTreeMap::new(),
            "explicit" => {
                let widths = int_array(f, "sign_widths")?;
                let values = int_array(f, "sign_values")?;
                explicit_signs(&widths, &values, block)?
            }
            other => {
                return Err(format!(
                    "prism.hadamard.sign_mode {other:?} is not supported (identity or explicit)"
                ));
            }
        };
        let weights: Vec<String> = str_array(f, "weight_names", true)?;
        if weights.is_empty() {
            return Err("prism.hadamard.weight_names is empty".into());
        }
        let mut weight_set = BTreeSet::new();
        for w in weights {
            if !weight_set.insert(w.clone()) {
                return Err(format!("prism.hadamard.weight_names lists {w:?} twice"));
            }
        }
        let mut inverses = BTreeSet::new();
        for name in str_array(f, "inverse_weight_names", false)? {
            if weight_set.contains(&name) || !inverses.insert(name.clone()) {
                return Err(format!("prism.hadamard lists {name:?} twice"));
            }
        }
        let gdn_v_grouped = f
            .get(&key("gdn_v_grouped"))
            .map(|v| {
                v.as_bool()
                    .or_else(|| v.as_u64().map(|u| u != 0))
                    .or_else(|| v.as_i64().map(|i| i != 0))
                    .ok_or("prism.hadamard.gdn_v_grouped is not a bool")
            })
            .transpose()?
            .unwrap_or(false);
        Ok(Some(Self {
            block,
            signs,
            weights: weight_set,
            inverses,
            gdn_v_grouped,
        }))
    }

    /// Is `name`'s matmul input transformed?
    #[must_use]
    pub fn folds(&self, name: &str) -> bool {
        self.weights.contains(name)
    }

    /// Is `name` a rotated lookup table (inverse after the gather)?
    #[must_use]
    pub fn inverts(&self, name: &str) -> bool {
        self.inverses.contains(name)
    }

    /// The sign vector for an input `width`: `None` in identity mode, an
    /// error in explicit mode when the width has none (the fold would run
    /// with the wrong signs — refuse).
    ///
    /// # Errors
    ///
    /// In explicit sign mode, an error when no sign vector was declared
    /// for `width`.
    pub fn signs_for(&self, width: usize) -> Result<Option<&[f32]>, String> {
        if self.signs.is_empty() {
            return Ok(None);
        }
        self.signs
            .get(&width)
            .map(|v| Some(v.as_slice()))
            .ok_or_else(|| format!("prism.hadamard has no sign vector for width {width}"))
    }

    /// The transform on host memory: `x` is one row of `width` values.
    /// Signs first, then the normalized transform per block — exactly what
    /// [`DeviceConsts::forward`] computes on a device, for the paths that
    /// keep activations on the host (the fused `DeltaNet` decode step).
    ///
    /// # Errors
    ///
    /// An error when the block does not divide `x.len()`, or (explicit
    /// sign mode) when `x.len()` has no declared sign vector.
    pub fn forward_host(&self, x: &mut [f32]) -> Result<(), String> {
        let width = x.len();
        if !width.is_multiple_of(self.block) {
            return Err(format!(
                "prism.hadamard block {} does not divide width {width}",
                self.block
            ));
        }
        if let Some(s) = self.signs_for(width)? {
            for (v, s) in x.iter_mut().zip(s) {
                *v *= s;
            }
        }
        let scale = 1.0 / f32_from_usize(self.block).sqrt();
        for chunk in x.chunks_mut(self.block) {
            mummu_mix::hadamard::fwht(chunk);
            for v in chunk.iter_mut() {
                *v *= scale;
            }
        }
        Ok(())
    }
}

/// The normalized Sylvester Walsh–Hadamard matrix of order `n`, row-major:
/// `H[i][j] = (−1)^{popcount(i & j)} / √n`. Symmetric and its own inverse.
///
/// # Panics
///
/// Panics if `n` is not a power of two.
#[must_use]
pub fn sylvester_matrix(n: usize) -> Vec<f32> {
    assert!(
        n.is_power_of_two(),
        "Hadamard order {n} is not a power of two"
    );
    let scale = 1.0 / f32_from_usize(n).sqrt();
    let mut h = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            h.push(if (i & j).count_ones() & 1 == 1 {
                -scale
            } else {
                scale
            });
        }
    }
    h
}

/// The transform's constants resident on one device: the block matrix and
/// one sign vector per width seen so far.
pub struct DeviceConsts {
    spec: Arc<HadamardSpec>,
    device: Device,
    h: Tensor<2>,
    signs: Mutex<BTreeMap<usize, Tensor<1>>>,
}

impl std::fmt::Debug for DeviceConsts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceConsts")
            .field("block", &self.spec.block)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl DeviceConsts {
    fn new(spec: Arc<HadamardSpec>, device: &Device) -> Self {
        let n = spec.block;
        let dtype = crate::backend::float_dtype(device);
        let h = Tensor::<2>::from_data(
            TensorData::new(sylvester_matrix(n), [n, n]),
            (device, dtype),
        );
        Self {
            spec,
            device: device.clone(),
            h,
            signs: Mutex::new(BTreeMap::new()),
        }
    }

    /// The contract these constants implement.
    #[must_use]
    pub fn spec(&self) -> &HadamardSpec {
        &self.spec
    }

    /// The sign vector for `width` on this device (built on first use).
    fn signs(&self, width: usize) -> Option<Tensor<1>> {
        let s = self
            .spec
            .signs_for(width)
            .expect("folded widths were checked at load")?;
        let mut cache = self.signs.lock().expect("sign cache");
        Some(
            cache
                .entry(width)
                .or_insert_with(|| {
                    let dtype = crate::backend::float_dtype(&self.device);
                    Tensor::<1>::from_data(
                        TensorData::new(s.to_vec(), [width]),
                        (&self.device, dtype),
                    )
                })
                .clone(),
        )
    }

    fn apply_signs<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dims = x.dims();
        let width = dims[D - 1];
        match self.signs(width) {
            Some(s) => {
                let mut shape = [1usize; D];
                shape[D - 1] = width;
                x.mul(s.reshape(shape))
            }
            None => x,
        }
    }

    fn blockwise<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dims = x.dims();
        let n = self.spec.block;
        let width = dims[D - 1];
        assert!(
            width.is_multiple_of(n),
            "Hadamard block {n} does not divide activation width {width}"
        );
        let rows = dims.iter().product::<usize>() / n;
        x.reshape([rows, n]).matmul(self.h.clone()).reshape(dims)
    }

    /// The fold's input-side transform: signs, then the blockwise
    /// transform along the last axis. Feed the result to a folded weight.
    #[must_use]
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.blockwise(self.apply_signs(x))
    }

    /// The inverse: the blockwise transform, then signs — restores a rotated
    /// lookup row to the primal basis.
    #[must_use]
    pub fn inverse<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.apply_signs(self.blockwise(x))
    }
}

/// The contract plus its per-device constants, built lazily.
///
/// Layers of one model live on several devices and each needs its own copy
/// of the block matrix (4 MiB at block 1024, once per device, not per
/// layer).
pub struct HadamardRuntime {
    spec: Arc<HadamardSpec>,
    per_device: Mutex<Vec<Arc<DeviceConsts>>>,
}

impl std::fmt::Debug for HadamardRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HadamardRuntime")
            .field("block", &self.spec.block)
            .field("folded_weights", &self.spec.weights.len())
            .field("inverse_tables", &self.spec.inverses.len())
            .field("gdn_v_grouped", &self.spec.gdn_v_grouped)
            .finish_non_exhaustive()
    }
}

impl HadamardRuntime {
    #[must_use]
    pub fn new(spec: HadamardSpec) -> Self {
        Self {
            spec: Arc::new(spec),
            per_device: Mutex::new(Vec::new()),
        }
    }

    #[must_use]
    pub fn spec(&self) -> &HadamardSpec {
        &self.spec
    }

    /// The constants on `device`, built on first use.
    ///
    /// # Panics
    ///
    /// Panics if the per-device cache mutex is poisoned (a builder panicked
    /// on another thread).
    pub fn on(&self, device: &Device) -> Arc<DeviceConsts> {
        let mut per = self.per_device.lock().expect("hadamard device cache");
        if let Some(c) = per.iter().find(|c| &c.device == device) {
            return Arc::clone(c);
        }
        let c = Arc::new(DeviceConsts::new(Arc::clone(&self.spec), device));
        per.push(Arc::clone(&c));
        c
    }
}

/// Reorder a `DeltaNet` value activation `[b, t, n_v·hd]` from tiled to
/// grouped head order.
///
/// The tiled order is what this port computes in (head `h` reads key-head
/// `h % n_k`, i.e. `[rep, n_k, hd]` row-major); HF's grouped order
/// (`[n_k, rep, hd]`, head `h` reads key-head `h / rep`) is the order a
/// `gdn_v_grouped` fold of `ssm_out` expects its input in.
///
/// # Panics
///
/// Panics if `n_k == 0`, `n_v` is not a multiple of `n_k`, or the last dim
/// of `x` is not `n_v * hd`.
#[must_use]
pub fn tiled_to_grouped(x: Tensor<3>, n_k: usize, n_v: usize, hd: usize) -> Tensor<3> {
    let [b, t, n] = x.dims();
    assert!(
        n_k > 0 && n_v.is_multiple_of(n_k) && n == n_v * hd,
        "tiled_to_grouped: {n} is not {n_v} heads of {hd} tiled over {n_k}"
    );
    let rep = n_v / n_k;
    if rep == 1 {
        return x;
    }
    x.reshape([b, t, rep, n_k, hd])
        .swap_dims(2, 3)
        .reshape([b, t, n])
}

/// The same reorder on a host row (see [`tiled_to_grouped`]).
///
/// # Panics
///
/// Panics if `n_k == 0`, `n_v` is not a multiple of `n_k`, or `x.len()` is
/// not `n_v * hd`.
#[must_use]
pub fn tiled_to_grouped_host(x: &[f32], n_k: usize, n_v: usize, hd: usize) -> Vec<f32> {
    assert!(
        n_k > 0 && n_v.is_multiple_of(n_k) && x.len() == n_v * hd,
        "tiled_to_grouped_host: {} is not {n_v} heads of {hd} tiled over {n_k}",
        x.len()
    );
    let rep = n_v / n_k;
    let mut out = Vec::with_capacity(x.len());
    for k in 0..n_k {
        for r in 0..rep {
            let src = (r * n_k + k) * hd;
            out.extend_from_slice(&x[src..src + hd]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::TestGguf;

    fn spec(block: usize, widths: &[usize]) -> HadamardSpec {
        let mut signs = BTreeMap::new();
        for &w in widths {
            signs.insert(w, mummu_mix::hadamard::sign_diagonal(7 + w as u64, w));
        }
        HadamardSpec {
            block,
            signs,
            weights: std::iter::once("blk.0.ffn_up.weight".to_string()).collect(),
            inverses: std::iter::once("token_embd.weight".to_string()).collect(),
            gdn_v_grouped: true,
        }
    }

    /// `H` is symmetric, orthonormal and its own inverse; the natural-order
    /// Sylvester construction pins entry `(i, j)` to the parity of `i & j`.
    #[test]
    fn sylvester_matrix_is_orthonormal_and_involutory() {
        let n = 16;
        let h = sylvester_matrix(n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(h[i * n + j], h[j * n + i]);
                let dot: f32 = (0..n).map(|k| h[i * n + k] * h[j * n + k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-6, "({i},{j}) {dot}");
            }
        }
        assert_eq!(h[0], 0.25);
        assert_eq!(h[n + 1], -0.25);
        assert_eq!(h[3 * n + 5], -0.25); // popcount(3 & 5) = popcount(1) = 1 → −
        assert_eq!(h[3 * n + 6], -0.25); // popcount(3 & 6) = popcount(2) = 1 → −
        assert_eq!(h[3 * n + 7], 0.25); // popcount(3 & 7) = popcount(3) = 2 → +
    }

    /// The device transform equals the host transform, and the inverse
    /// undoes it — on a 3-D activation with several blocks per row.
    #[test]
    fn device_forward_matches_host_and_inverse_round_trips() {
        let s = Arc::new(spec(8, &[24]));
        let rt = HadamardRuntime::new((*s).clone());
        let device = crate::backend::cpu_device();
        let consts = rt.on(&device);
        let vals: Vec<f32> = (0..2 * 3 * 24usize)
            .map(|i| f32_from_usize((i * 37) % 11).mul_add(0.25, -1.0))
            .collect();
        let x = Tensor::<3>::from_data(TensorData::new(vals.clone(), [2, 3, 24]), &device);
        let got = consts
            .forward(x.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let mut want = vals.clone();
        for row in want.chunks_mut(24) {
            s.forward_host(row).unwrap();
        }
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
        let back = consts
            .inverse(consts.forward(x))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (b, v) in back.iter().zip(&vals) {
            assert!((b - v).abs() < 1e-5, "{b} vs {v}");
        }
        // The same device hands back the same constants; another device would not.
        assert!(Arc::ptr_eq(&consts, &rt.on(&device)));
    }

    /// Tiled → grouped: value head `h = r·n_k + k` moves to `k·rep + r`.
    #[test]
    fn tiled_to_grouped_reorders_value_heads() {
        let (n_k, n_v, hd) = (2usize, 6usize, 2usize);
        let vals: Vec<f32> = (0..n_v * hd).map(f32_from_usize).collect();
        let host = tiled_to_grouped_host(&vals, n_k, n_v, hd);
        // grouped head (k=0, r=0) = tiled head 0, (k=0, r=1) = tiled head 2, (k=0,r=2) = 4
        assert_eq!(&host[..6], &[0.0, 1.0, 4.0, 5.0, 8.0, 9.0]);
        assert_eq!(&host[6..], &[2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);
        let device = crate::backend::cpu_device();
        let t = tiled_to_grouped(
            Tensor::<3>::from_data(TensorData::new(vals, [1, 1, n_v * hd]), &device),
            n_k,
            n_v,
            hd,
        )
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
        assert_eq!(t, host);
    }

    fn header(overrides: &[(&str, &str)], block: u32) -> Vec<u8> {
        let mut strs: Vec<(&str, &str)> = vec![
            (
                "prism.hadamard.transform",
                "normalized-sylvester-walsh-hadamard",
            ),
            ("prism.hadamard.axis", "input-last-dimension"),
            ("prism.hadamard.sign_mode", "explicit"),
        ];
        for (k, v) in overrides {
            match strs.iter_mut().find(|(sk, _)| sk == k) {
                Some(slot) => slot.1 = v,
                None => strs.push((k, v)),
            }
        }
        let mut g = TestGguf::new()
            .kv_str("general.architecture", "qwen35")
            .kv_u32("prism.hadamard.version", 1)
            .kv_u32("prism.hadamard.block_size", block)
            .kv_str_array(
                "prism.hadamard.weight_names",
                &["output.weight", "blk.0.ffn_up.weight"],
            )
            .kv_str_array(
                "prism.hadamard.inverse_weight_names",
                &["token_embd.weight"],
            );
        for (k, v) in strs {
            g = g.kv_str(k, v);
        }
        g.kv_i32_array("prism.hadamard.sign_widths", &[8, 16])
            .kv_i32_array(
                "prism.hadamard.sign_values",
                &[
                    1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1,
                    -1,
                ],
            )
            .build()
    }

    /// The contract parses from a header, with its sign table cut by width.
    #[test]
    fn explicit_contract_parses_from_a_header() {
        let bytes = header(&[], 8);
        crate::gguf::tests::with_gguf_bytes(&bytes, |f| {
            let s = HadamardSpec::from_gguf(&f.expect("parses"))
                .expect("contract ok")
                .expect("declared");
            assert_eq!(s.block, 8);
            assert_eq!(s.signs.len(), 2);
            assert_eq!(
                s.signs[&8],
                vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
            );
            assert_eq!(s.signs[&16].len(), 16);
            assert!(s.folds("output.weight") && s.folds("blk.0.ffn_up.weight"));
            assert!(!s.folds("blk.0.ffn_down.weight"));
            assert!(s.inverts("token_embd.weight"));
            assert!(!s.gdn_v_grouped);
            assert!(s.signs_for(24).is_err(), "an undeclared width is refused");
        });
        // No contract at all is simply an ordinary checkpoint.
        let plain = TestGguf::new()
            .kv_str("general.architecture", "qwen35")
            .build();
        crate::gguf::tests::with_gguf_bytes(&plain, |f| {
            assert_eq!(HadamardSpec::from_gguf(&f.expect("parses")).unwrap(), None);
        });
    }

    /// Anything this runtime does not implement is a load error, named.
    #[test]
    fn unsupported_contracts_are_refused_by_name() {
        for (kvs, block, needle) in [
            (vec![("prism.hadamard.sign_mode", "random")], 8, "sign_mode"),
            (vec![("prism.hadamard.transform", "walsh")], 8, "transform"),
            (vec![("prism.hadamard.axis", "output")], 8, "axis"),
            (vec![], 12, "power of two"),
            // Width 8 is not a multiple of block 16.
            (vec![], 16, "multiple of the block"),
        ] {
            let bytes = header(&kvs, block);
            crate::gguf::tests::with_gguf_bytes(&bytes, |f| {
                let err = HadamardSpec::from_gguf(&f.expect("parses")).expect_err("refused");
                assert!(err.contains(needle), "{err} should mention {needle}");
            });
        }
    }
}
