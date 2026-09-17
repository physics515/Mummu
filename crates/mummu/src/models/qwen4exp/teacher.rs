//! Teacher-forced per-op comparison against a FULL-precision llama.cpp dump:
//! a parity DIAGNOSTIC, off unless `MUMMU_QWEN4EXP_TEACHER` names a dump.
//!
//! Why: the first-forward gate compares tail logprobs at -15..-20 nats after
//! 48 layers of llama.cpp's activation-quantization noise, which cannot tell a
//! small port bug from the reference's own rounding. Here every op of our
//! forward is fed llama.cpp's exact input (its dumped tensor replaces ours at
//! each named point), so the printed error of one op is that op's error
//! alone. Under `MUMMU_REF_ARITH=1` (llama.cpp's activation grids) a
//! structurally identical op agrees to ~1e-4 relative or better; a
//! structural difference is an outlier by orders of magnitude.
//!
//! The dump comes from `tools/qwen4exp_dump_tensors.cpp` (a `cb_eval`
//! dumper linked against the reference image's libllama): `index.tsv` rows
//! `seq name occurrence type ne0 ne1 ne2 ne3` and `NNNNN.bin` raw
//! little-endian values in ggml flat order. That order is ours: a ggml
//! `{E, H, T}` tensor at `e + E·(s + H·t)` is our stream-major `[1, T, H·E]`.
//!
//! `MUMMU_QWEN4EXP_TEACHER_FORCE=0` compares without replacing, which shows
//! how the free-running difference grows instead.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use burn::tensor::{Tensor, TensorData};

struct Entry {
    seq: usize,
    elems: usize,
    dtype: String,
}

struct Teacher {
    dir: PathBuf,
    entries: HashMap<(String, usize), Entry>,
    force: bool,
}

fn teacher() -> Option<&'static Teacher> {
    static T: OnceLock<Option<Teacher>> = OnceLock::new();
    T.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN4EXP_TEACHER")?);
        let index = std::fs::read_to_string(dir.join("index.tsv"))
            .unwrap_or_else(|e| panic!("teacher index {}: {e}", dir.display()));
        let mut entries = HashMap::new();
        for line in index.lines().filter(|l| !l.is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 8, "teacher index row {line:?}");
            let num = |s: &str| s.parse::<usize>().expect("teacher index number");
            let elems = f[4..8].iter().map(|s| num(s)).product();
            entries.insert(
                (f[1].to_string(), num(f[2])),
                Entry {
                    seq: num(f[0]),
                    elems,
                    dtype: f[3].to_string(),
                },
            );
        }
        let force = std::env::var("MUMMU_QWEN4EXP_TEACHER_FORCE").map_or(true, |v| v != "0");
        eprintln!(
            "[qwen4exp-teacher] {} tensors from {} (force={force})",
            entries.len(),
            dir.display()
        );
        Some(Teacher {
            dir,
            entries,
            force,
        })
    })
    .as_ref()
}

/// Whether a teacher dump is configured.
pub(super) fn active() -> bool {
    teacher().is_some()
}

/// The dumped f32 values of `name` (its `occ`-th evaluation), if present.
fn values(t: &Teacher, name: &str, occ: usize) -> Option<Vec<f32>> {
    let e = t.entries.get(&(name.to_string(), occ))?;
    if e.dtype != "f32" {
        return None;
    }
    let bytes = std::fs::read(t.dir.join(format!("{:05}.bin", e.seq)))
        .unwrap_or_else(|err| panic!("teacher {name}: {err}"));
    assert_eq!(bytes.len(), e.elems * 4, "teacher {name}: byte count");
    let (values, _) = bytes.as_chunks::<4>();
    Some(values.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Compare `ours [b, t, w]` with the dump's `name#occ` (all tokens, or the
/// last token when llama.cpp already dropped the others), print the error,
/// and in force mode return the reference values in our shape.
pub(super) fn teach(name: &str, occ: usize, ours: Tensor<3>) -> Tensor<3> {
    let Some(t) = teacher() else {
        return ours;
    };
    let Some(reference) = values(t, name, occ) else {
        eprintln!("[qwen4exp-teacher] {name}#{occ}: not in the dump");
        return ours;
    };
    teach_values(t, &format!("{name}#{occ}"), ours, &reference)
}

/// [`teach`] against a reference the caller assembled (e.g. an op llama.cpp
/// does not name).
fn teach_values(t: &Teacher, label: &str, ours: Tensor<3>, reference: &[f32]) -> Tensor<3> {
    let dims = ours.dims();
    let [b, tokens, w] = dims;
    let device = ours.device();
    let dtype = ours.dtype();
    let mut vals = ours
        .clone()
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("teacher readback");
    let start = if reference.len() == vals.len() {
        0
    } else if reference.len() == w && b * tokens > 1 {
        vals.len() - w
    } else {
        eprintln!(
            "[qwen4exp-teacher] {label}: {} reference values for ours {dims:?}; skipped",
            reference.len()
        );
        return ours;
    };
    let (mut d2, mut r2, mut dmax, mut rmax) = (0f64, 0f64, 0f64, 0f64);
    for (o, r) in vals[start..].iter().zip(reference) {
        let d = f64::from(*o) - f64::from(*r);
        d2 += d * d;
        r2 += f64::from(*r) * f64::from(*r);
        dmax = dmax.max(d.abs());
        rmax = rmax.max(f64::from(*r).abs());
    }
    eprintln!(
        "[qwen4exp-teacher] {label}: n={} rel={:.3e} max|d|={dmax:.3e} max|ref|={rmax:.3e}",
        reference.len(),
        (d2 / r2.max(f64::MIN_POSITIVE)).sqrt(),
    );
    if !t.force {
        return ours;
    }
    vals[start..].copy_from_slice(reference);
    Tensor::from_data(TensorData::new(vals, dims), (&device, dtype))
}

/// The PLE block's output, which llama.cpp does not name: its
/// `add(hidden, add(ple_gated_value, ple_conv_out))` rebuilt from the dump
/// in the same order, with `hidden` the previous layer's `l_last`.
pub(super) fn teach_ple_out(layer: usize, ours: Tensor<3>) -> Tensor<3> {
    let Some(t) = teacher() else {
        return ours;
    };
    let hidden = if layer == 0 {
        values(t, "hc_init", 0)
    } else {
        values(t, &format!("l_last-{}", layer - 1), 0)
    };
    let gated = values(t, &format!("ple_gated_value-{layer}"), 0);
    let conv = values(t, &format!("ple_conv_out-{layer}"), 0);
    let (Some(h), Some(g), Some(c)) = (hidden, gated, conv) else {
        eprintln!("[qwen4exp-teacher] ple_out-{layer}: inputs not in the dump");
        return ours;
    };
    let reference: Vec<f32> = h
        .iter()
        .zip(g.iter().zip(&c))
        .map(|(h, (g, c))| h + (g + c))
        .collect();
    teach_values(t, &format!("ple_out-{layer}"), ours, &reference)
}
