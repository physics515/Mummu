//! Stream-dequantize every tensor of a GGUF file and report per-dtype
//! coverage + value sanity (finite, plausible range) — the cheap whole-file
//! gate for new dequant formats:
//! `cargo run --release -p mummu --example gguf-verify -- <path.gguf>`.

use std::collections::BTreeMap;

use mummu::gguf::GgufFile;

fn main() {
    let path = std::env::args().nth(1).expect("usage: gguf-verify <path.gguf>");
    let f = GgufFile::open(std::path::Path::new(&path)).expect("gguf opens");
    println!("architecture: {:?} | tensors: {}", f.architecture(), f.tensors.len());

    let mut per_dtype: BTreeMap<String, (usize, f32)> = BTreeMap::new();
    let mut bad = 0usize;
    for (i, t) in f.tensors.iter().enumerate() {
        let v = match f.read_tensor_f32(&t.name) {
            Ok(v) => v,
            Err(e) => {
                println!("FAIL {} ({:?}): {e}", t.name, t.dtype);
                bad += 1;
                continue;
            }
        };
        let mut max_abs = 0.0f32;
        let mut finite = true;
        for &x in &v {
            if !x.is_finite() {
                finite = false;
                break;
            }
            max_abs = max_abs.max(x.abs());
        }
        if !finite {
            println!("NON-FINITE {} ({:?})", t.name, t.dtype);
            bad += 1;
            continue;
        }
        let e = per_dtype.entry(format!("{:?}", t.dtype)).or_insert((0, 0.0));
        e.0 += 1;
        e.1 = e.1.max(max_abs);
        if i % 100 == 0 {
            eprintln!("... {i}/{} {}", f.tensors.len(), t.name);
        }
    }
    for (dtype, (n, max_abs)) in &per_dtype {
        println!("{dtype}: {n} tensors, max |x| = {max_abs:.3}");
    }
    println!("bad: {bad}");
    assert_eq!(bad, 0, "every tensor must dequantize to finite values");
}
