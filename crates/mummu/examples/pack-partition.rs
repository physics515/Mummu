//! Partition the FFNs of an existing `.mummu` pack into neuron clusters
//! **in place** (P9 stage 3c) — what `pack-import` and `mummu-serve` do
//! automatically for qwen35 packs; this runs it by hand on a pack imported
//! before partitioning existed.
//!
//! ```text
//! cargo run --release --example pack-partition -- <pack dir> [clusters]
//! ```
//!
//! Exact: the model is unchanged (neurons reordered), every stored level
//! rewritten from the permuted f32. Crash-safe per layer via a journal.

use std::path::PathBuf;
use std::time::Instant;

use mummu::models::qwen35;
use mummu::pack::Pack;
use mummu::partition::{DEFAULT_CLUSTERS, partition_pack};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        eprintln!("usage: pack-partition <pack dir> [clusters]");
        std::process::exit(2);
    };
    let dir = PathBuf::from(dir);
    let clusters = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CLUSTERS);
    let mut pack = Pack::open(&dir).unwrap_or_else(|e| {
        eprintln!("{}: {e}", dir.display());
        std::process::exit(1);
    });
    if pack.manifest.ffn_partition.is_some() {
        eprintln!("{} is already partitioned", dir.display());
        return;
    }
    let names = match pack.manifest.architecture.as_str() {
        "qwen35" => {
            let header = pack.header().unwrap_or_else(|e| {
                eprintln!("header: {e}");
                std::process::exit(1);
            });
            let cfg = qwen35::Qwen35Config::from_gguf(&header).unwrap_or_else(|e| {
                eprintln!("qwen35 config: {e}");
                std::process::exit(1);
            });
            qwen35::ffn_names(cfg.num_layers)
        }
        other => {
            eprintln!("no FFN partitioner for architecture {other:?}");
            std::process::exit(1);
        }
    };
    let started = Instant::now();
    partition_pack(&mut pack, &names, clusters, |i, n| {
        eprintln!("  layer {i}/{n}  ({:.0}s)", started.elapsed().as_secs_f32());
    })
    .unwrap_or_else(|e| {
        eprintln!("partition failed: {e}");
        std::process::exit(1);
    });
    let part = pack.manifest.ffn_partition.as_ref().expect("written");
    eprintln!(
        "partitioned {} layers into {} clusters each in {:.0}s",
        part.layers.len(),
        part.layers.first().map_or(0, Vec::len),
        started.elapsed().as_secs_f32()
    );
}
