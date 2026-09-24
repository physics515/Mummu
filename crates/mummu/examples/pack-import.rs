//! Import a GGUF into a `.mummu` pack — the same one-time conversion
//! `mummu-serve` performs on first load, runnable ahead of time (or on a
//! different machine than the one that serves it).
//!
//! ```text
//! cargo run --release --example pack-import -- <model.gguf> <out dir> [q4,q8,f16,f32]
//! ```
//!
//! The architecture is read from the GGUF header (`qwen35`, `olmoe`). The
//! pack is built in `<out dir>.importing` and renamed on success, so an
//! interrupted run never leaves a directory the loaders would accept.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use mummu::gguf::GgufFile;
use mummu::models::{olmoe, qwen35};
use mummu::pack::{ImportAction, Pack, Precision};
use mummu_num::f64_from_u64;

/// The per-tensor import decision for one architecture.
type ActionMap = Box<dyn Fn(&mummu::gguf::GgufTensorInfo) -> Option<ImportAction>>;

/// The importer for the header's architecture; exits with a message for an
/// architecture without one.
fn action_map(arch: &str, header: &GgufFile) -> ActionMap {
    match arch {
        "qwen35" => {
            let cfg = qwen35::Qwen35Config::from_gguf(header).unwrap_or_else(|e| {
                eprintln!("qwen35 config: {e}");
                std::process::exit(1);
            });
            let trunk = cfg.num_layers;
            Box::new(move |info| qwen35::pack_actions(info, trunk))
        }
        "olmoe" => Box::new(olmoe::pack_actions),
        other => {
            eprintln!("no pack importer for architecture {other:?}");
            std::process::exit(1);
        }
    }
}

/// P9 stage 3(c): partition the dense FFNs in place (exact; enables tiering).
fn partition_in_place(out: &Path) {
    let mut pack = Pack::open(out).unwrap_or_else(|e| {
        eprintln!("reopen pack: {e}");
        std::process::exit(1);
    });
    let header = pack.header().expect("pack header");
    let trunk = qwen35::Qwen35Config::from_gguf(&header)
        .expect("config")
        .num_layers;
    drop(header);
    let t = Instant::now();
    mummu::partition::partition_pack(
        &mut pack,
        &qwen35::ffn_names(trunk),
        mummu::partition::DEFAULT_CLUSTERS,
        |i, n| {
            if i % 8 == 0 {
                eprintln!(
                    "  partition layer {i}/{n}  ({:.0}s)",
                    t.elapsed().as_secs_f32()
                );
            }
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("partition failed: {e}");
        std::process::exit(1);
    });
    eprintln!("  FFNs partitioned in {:.0}s", t.elapsed().as_secs_f32());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(gguf), Some(out)) = (args.first(), args.get(1)) else {
        eprintln!("usage: pack-import <model.gguf> <out dir> [q4,q8,f16,f32]");
        std::process::exit(2);
    };
    let gguf = PathBuf::from(gguf);
    let out = PathBuf::from(out);
    let precisions = args.get(2).map_or_else(
        || Precision::ALL.to_vec(),
        |list| {
            Precision::parse_list(list).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            })
        },
    );

    let header = GgufFile::open(&gguf).unwrap_or_else(|e| {
        eprintln!("{}: {e}", gguf.display());
        std::process::exit(1);
    });
    let arch = header.architecture().unwrap_or("").to_string();
    let map = action_map(&arch, &header);
    drop(header);

    let tmp = out.with_extension("importing");
    let _ = std::fs::remove_dir_all(&tmp);
    eprintln!(
        "importing {} → {} at {precisions:?}",
        gguf.display(),
        out.display()
    );
    let started = Instant::now();
    let mut last_pct = usize::MAX;
    let manifest = mummu::pack::import_gguf(&gguf, &tmp, &precisions, &*map, |i, n, name| {
        let pct = i * 100 / n.max(1);
        if pct != last_pct && pct % 5 == 0 {
            last_pct = pct;
            eprintln!(
                "  {pct:>3}%  {name}  ({:.0}s)",
                started.elapsed().as_secs_f32()
            );
        }
    })
    .unwrap_or_else(|e| {
        eprintln!("import failed: {e}");
        std::process::exit(1);
    });
    if out.exists() {
        eprintln!(
            "{} already exists — leaving the new pack at {}",
            out.display(),
            tmp.display()
        );
    } else {
        std::fs::rename(&tmp, &out).unwrap_or_else(|e| {
            eprintln!("finalize: {e}");
            std::process::exit(1);
        });
    }
    let folded = Pack::open(&out)
        .and_then(|p| mummu::partition::pack_is_hadamard_folded(&p))
        .unwrap_or_else(|e| {
            eprintln!("reopen pack: {e}");
            std::process::exit(1);
        });
    if folded {
        eprintln!(
            "  Hadamard-folded checkpoint: FFNs stay in their original neuron order (not partitioned)"
        );
    }
    if arch == "qwen35" && !folded {
        partition_in_place(&out);
    }
    let total: u64 = precisions
        .iter()
        .map(|p| std::fs::metadata(out.join(p.blob_name())).map_or(0, |m| m.len()))
        .sum();
    eprintln!(
        "pack ready: {} tensors, {:.1} GiB across {precisions:?}, {:.0}s",
        manifest.tensors.len(),
        f64_from_u64(total) / f64::from(1u32 << 30),
        started.elapsed().as_secs_f32()
    );
}
