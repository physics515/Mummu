//! **FFN partitioning** — P9 stage 3(c): turn a dense model's `SwiGLU` FFNs
//! into neuron clusters so the tier machinery built for `MoE` experts applies
//! to dense models too, **without changing the model**.
//!
//! `SwiGLU` is a sum over intermediate neurons: `down(silu(gate(x)) * up(x))`
//! = `Σ_j` `silu(x·g_j)(x·u_j)` `d_j`. Any partition of the neurons into clusters
//! therefore computes the *same* function when every cluster runs — so the
//! importer may reorder the intermediate dimension so clusters are
//! contiguous, record the cluster spans, and the runtime can hold different
//! clusters on different devices at different precisions (exact), or skip
//! low-energy clusters per token (opt-in, measured — see the skip table).
//!
//! The partition is stored **in place**: the FFN entries keep their names,
//! shapes and byte sizes, only the neuron order changes (columns of
//! `gate`/`up` in their `[hidden, inter]` Linear layout, rows of `down`),
//! every stored precision rewritten from the permuted f32. Loaders that
//! know nothing about partitions keep working unchanged.
//!
//! Clustering is MoEfication-style *parameter* clustering — balanced
//! k-means on each neuron's `gate ‖ up` weight vector — which needs no
//! calibration data (so it runs on import, for any size of model). A
//! Johnson–Lindenstrauss projection to [`PROJ_DIMS`] keeps it cheap. For
//! the exact path the cluster quality is irrelevant; it matters for
//! skipping, and the skip table measures that honestly.

use std::collections::BTreeMap;

use mummu_num::f32_from_usize;

use crate::pack::{ClusterSpan, FfnPartition, Pack, Precision, quantize_blocks};

/// Default clusters per layer (reduced to the largest divisor that keeps
/// every cluster a whole number of quantization blocks).
pub const DEFAULT_CLUSTERS: usize = 32;
/// JL projection width for the clustering features.
pub const PROJ_DIMS: usize = 192;
/// k-means iterations.
const KMEANS_ITERS: usize = 10;

/// The three FFN entries of one layer, by pack name.
#[derive(Debug, Clone)]
pub struct FfnNames {
    pub gate: String,
    pub up: String,
    pub down: String,
}

/// Pick the cluster count: the largest `c <= want` with `inter % (c * block) == 0`.
#[must_use]
pub fn cluster_count(inter: usize, want: usize, block: usize) -> usize {
    (1..=want)
        .rev()
        .find(|c| inter.is_multiple_of(c * block))
        .unwrap_or(1)
}

/// Deterministic ±1 projection of `dims`-long vectors to [`PROJ_DIMS`]
/// (splitmix64 stream, so every import of a model yields the same clusters).
fn projection(dims: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..dims * PROJ_DIMS)
        .map(|_| if next() & 1 == 0 { 1.0 } else { -1.0 })
        .collect()
}

/// Balanced k-means: `points` rows of `dims` features into `clusters` equal
/// groups. Returns each point's cluster. Assignment is greedy by distance
/// with a capacity per cluster (the classic balanced heuristic) —
/// deterministic.
fn balanced_kmeans(features: &[f32], points: usize, dims: usize, clusters: usize) -> Vec<usize> {
    assert!(
        points.is_multiple_of(clusters),
        "balanced k-means: n must divide by k"
    );
    let cap = points / clusters;
    // Init: evenly spaced points.
    let mut centroids: Vec<f32> = (0..clusters)
        .flat_map(|c| {
            let i = c * points / clusters;
            features[i * dims..(i + 1) * dims].iter().copied()
        })
        .collect();
    let mut assign = vec![0usize; points];
    let mut dist = vec![0f32; points * clusters];
    for _ in 0..KMEANS_ITERS {
        // Distances, parallel over points.
        let threads = std::thread::available_parallelism()
            .map_or(4, std::num::NonZero::get)
            .min(32);
        let chunk = points.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (ti, slab) in dist.chunks_mut(chunk * clusters).enumerate() {
                let centroids = &centroids;
                scope.spawn(move || {
                    let start = ti * chunk;
                    for (li, row) in slab.chunks_mut(clusters).enumerate() {
                        let point = &features[(start + li) * dims..(start + li + 1) * dims];
                        for (c, out) in row.iter_mut().enumerate() {
                            let cen = &centroids[c * dims..(c + 1) * dims];
                            *out = point.iter().zip(cen).map(|(a, b)| (a - b) * (a - b)).sum();
                        }
                    }
                });
            }
        });
        // Greedy balanced assignment: every (point, cluster) pair by distance.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::with_capacity(points * clusters);
        for point in 0..points {
            for cluster in 0..clusters {
                pairs.push((dist[point * clusters + cluster], point, cluster));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut fill = vec![0usize; clusters];
        let mut done = vec![false; points];
        let mut left = points;
        for (_, point, cluster) in pairs {
            if done[point] || fill[cluster] == cap {
                continue;
            }
            assign[point] = cluster;
            done[point] = true;
            fill[cluster] += 1;
            left -= 1;
            if left == 0 {
                break;
            }
        }
        // Update centroids.
        centroids.fill(0.0);
        for (point, &cluster) in assign.iter().enumerate() {
            for (acc, v) in centroids[cluster * dims..(cluster + 1) * dims]
                .iter_mut()
                .zip(&features[point * dims..(point + 1) * dims])
            {
                *acc += v;
            }
        }
        let inv = 1.0 / f32_from_usize(cap);
        for v in &mut centroids {
            *v *= inv;
        }
    }
    assign
}

/// Cluster the neurons of one layer from its `gate` and `up` weights (both
/// `[hidden, inter]` row-major).
///
/// Returns the permutation (new position → old neuron index) with clusters
/// contiguous, and the cluster spans.
///
/// # Panics
///
/// If `gate` or `up` is not `hidden × inter` long, or if `inter` is not a
/// multiple of `clusters` (a balanced partition needs equal clusters).
#[must_use]
pub fn cluster_neurons(
    gate: &[f32],
    up: &[f32],
    hidden: usize,
    inter: usize,
    clusters: usize,
    seed: u64,
) -> (Vec<usize>, Vec<ClusterSpan>) {
    assert_eq!(gate.len(), hidden * inter);
    assert_eq!(up.len(), hidden * inter);
    assert!(inter.is_multiple_of(clusters));
    // Features: JL projection of the 2·hidden-long neuron vector (gate col ‖ up col).
    let proj = projection(2 * hidden, seed);
    let mut features = vec![0f32; inter * PROJ_DIMS];
    let threads = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .min(32);
    let chunk = inter.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ti, slab) in features.chunks_mut(chunk * PROJ_DIMS).enumerate() {
            let proj = &proj;
            s.spawn(move || {
                let start = ti * chunk;
                for (li, out) in slab.chunks_mut(PROJ_DIMS).enumerate() {
                    let j = start + li;
                    // Column j of gate/up: stride `inter`.
                    for (f, o) in out.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for h in 0..hidden {
                            // Two products, then the sum — no fused multiply-add,
                            // so every import of a model yields the same clusters.
                            let from_gate = gate[h * inter + j] * proj[h * PROJ_DIMS + f];
                            let from_up = up[h * inter + j] * proj[(hidden + h) * PROJ_DIMS + f];
                            acc += from_gate + from_up;
                        }
                        *o = acc;
                    }
                }
            });
        }
    });
    let assign = balanced_kmeans(&features, inter, PROJ_DIMS, clusters);
    let cap = inter / clusters;
    let mut perm = Vec::with_capacity(inter);
    let mut spans = Vec::with_capacity(clusters);
    for c in 0..clusters {
        let start = perm.len();
        perm.extend((0..inter).filter(|&j| assign[j] == c));
        debug_assert_eq!(perm.len() - start, cap);
        spans.push(ClusterSpan { start, len: cap });
    }
    (perm, spans)
}

/// Permute the columns of a row-major `[rows, cols]` matrix: new column `p`
/// takes old column `perm[p]`.
fn permute_cols(values: &[f32], rows: usize, cols: usize, perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; values.len()];
    for r in 0..rows {
        let src = &values[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (p, &j) in perm.iter().enumerate() {
            dst[p] = src[j];
        }
    }
    out
}

/// The FFN entry names of every trunk layer — the partitioner's input.
///
/// Not architecture-specific: Qwen2, Qwen3, LFM2 and Qwen3.5 all store the
/// standard GGUF triple, so this lives here rather than in any one model.
/// A model whose FFN is named differently, or which has none, supplies its
/// own list (or an empty one).
#[must_use]
pub fn ffn_names(trunk_layers: usize) -> Vec<FfnNames> {
    (0..trunk_layers)
        .map(|l| FfnNames {
            gate: format!("blk.{l}.ffn_gate.weight"),
            up: format!("blk.{l}.ffn_up.weight"),
            down: format!("blk.{l}.ffn_down.weight"),
        })
        .collect()
}

/// Permute the rows of a row-major `[rows, cols]` matrix.
fn permute_rows(values: &[f32], rows: usize, cols: usize, perm: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for &j in perm {
        out.extend_from_slice(&values[j * cols..(j + 1) * cols]);
    }
    debug_assert_eq!(out.len(), rows * cols);
    out
}

/// Partition every layer's FFN of a pack in place.
///
/// `layers` names each layer's three entries; `want` is the requested
/// cluster count. Writes the partition into the manifest (replacing any
/// previous one — which must not exist, since the entries would already be
/// permuted).
///
/// # Errors
///
/// When the pack is already partitioned or is Hadamard-folded (or its
/// `header.gguf` cannot be read); when the journal directory cannot be
/// created or a layer journal cannot be written or parsed; when a layer's
/// entry is missing from the manifest, its gate is not 2-D, or the three
/// shapes disagree; when a level cannot be read or rewritten in place; or
/// when the manifest cannot be saved.
pub fn partition_pack(
    pack: &mut Pack,
    layers: &[FfnNames],
    want: usize,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<(), String> {
    if pack.manifest.ffn_partition.is_some() {
        return Err("pack is already partitioned".into());
    }
    if pack_is_hadamard_folded(pack)? {
        return Err(
            "a Hadamard-folded checkpoint cannot be partitioned: the down-projection's input \
             transform is blockwise over the ORIGINAL intermediate order, and reordering the \
             neurons would silently break it (measured: layer-0 FFN wrong, garbage output)"
                .into(),
        );
    }
    // Crash safety: a layer's three entries are rewritten one after the
    // other, so a crash in between leaves gate permuted and down not — a
    // corrupt layer. The journal records, per layer, the permutation and a
    // fingerprint of each entry's f32 prefix BEFORE rewriting; on a rerun a
    // journaled layer is repaired by permuting only the entries whose
    // fingerprint still matches the pre-state. Journals of finished layers
    // are kept until the manifest is written, then removed.
    let journal_dir = pack.dir.join("partition.journal");
    std::fs::create_dir_all(&journal_dir).map_err(|e| e.to_string())?;
    let mut spans_per_layer = Vec::with_capacity(layers.len());
    let mut names = Vec::with_capacity(layers.len());
    for (li, n) in layers.iter().enumerate() {
        on_progress(li, layers.len());
        let journal = journal_dir.join(format!("layer-{li}.json"));
        if let Some(spans) = repair_layer(pack, n, &journal)? {
            spans_per_layer.push(spans);
            names.push([n.gate.clone(), n.up.clone(), n.down.clone()]);
            continue;
        }
        let gate_e = pack
            .entry(&n.gate)
            .ok_or_else(|| format!("missing {}", n.gate))?
            .clone();
        let up_e = pack
            .entry(&n.up)
            .ok_or_else(|| format!("missing {}", n.up))?
            .clone();
        let down_e = pack
            .entry(&n.down)
            .ok_or_else(|| format!("missing {}", n.down))?
            .clone();
        let &[hidden, inter] = gate_e.shape.as_slice() else {
            return Err(format!("{} is not 2-D", n.gate));
        };
        if up_e.shape != vec![hidden, inter] || down_e.shape != vec![inter, hidden] {
            return Err(format!(
                "layer {li}: FFN shapes disagree ({:?} / {:?} / {:?})",
                gate_e.shape, up_e.shape, down_e.shape
            ));
        }
        let clusters = cluster_count(inter, want, crate::pack::BLOCK);
        let gate = pack.read_f32(&gate_e)?;
        let up = pack.read_f32(&up_e)?;
        let down = pack.read_f32(&down_e)?;
        let (perm, spans) = cluster_neurons(&gate, &up, hidden, inter, clusters, li as u64);
        let entry = LayerJournal {
            perm: perm.clone(),
            spans: spans.clone(),
            before: [fingerprint(&gate), fingerprint(&up), fingerprint(&down)],
            done: false,
        };
        std::fs::write(
            &journal,
            serde_json::to_string(&entry).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        pack.rewrite_entry(&gate_e, &permute_cols(&gate, hidden, inter, &perm))?;
        pack.rewrite_entry(&up_e, &permute_cols(&up, hidden, inter, &perm))?;
        pack.rewrite_entry(&down_e, &permute_rows(&down, inter, hidden, &perm))?;
        let entry = LayerJournal {
            done: true,
            ..entry
        };
        std::fs::write(
            &journal,
            serde_json::to_string(&entry).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        spans_per_layer.push(spans);
        names.push([n.gate.clone(), n.up.clone(), n.down.clone()]);
    }
    on_progress(layers.len(), layers.len());
    pack.manifest.ffn_partition = Some(FfnPartition {
        layers: spans_per_layer,
        names,
        hotness: Vec::new(),
        skip_table: Vec::new(),
    });
    pack.save_manifest()?;
    let _ = std::fs::remove_dir_all(&journal_dir);
    Ok(())
}

/// Per-layer crash journal (see [`partition_pack`]).
#[derive(serde::Serialize, serde::Deserialize)]
struct LayerJournal {
    perm: Vec<usize>,
    spans: Vec<ClusterSpan>,
    /// Fingerprints of gate / up / down f32 before the rewrite.
    before: [u64; 3],
    done: bool,
}

/// Does the pack's source declare Prism's folded Hadamard basis
/// (`prism.hadamard.*`)? Such a pack must keep its FFN neuron order: the
/// fold is blockwise over that order (see `crate::nn::hadamard`).
///
/// # Errors
///
/// When `header.gguf` cannot be opened or parsed, or when the
/// `prism.hadamard.*` keys are present but malformed.
pub fn pack_is_hadamard_folded(pack: &Pack) -> Result<bool, String> {
    let header = pack.header()?;
    Ok(crate::nn::hadamard::HadamardSpec::from_gguf(&header)?.is_some())
}

/// FNV-1a over the first 4096 values (enough to tell permuted from not).
fn fingerprint(values: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values.iter().take(4096) {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// If `journal` exists, finish that layer: entries still matching their
/// pre-rewrite fingerprint get permuted, the rest are already done.
/// Returns the layer's spans, or `None` when there was no journal.
fn repair_layer(
    pack: &Pack,
    n: &FfnNames,
    journal: &std::path::Path,
) -> Result<Option<Vec<ClusterSpan>>, String> {
    let Ok(text) = std::fs::read_to_string(journal) else {
        return Ok(None);
    };
    let j: LayerJournal =
        serde_json::from_str(&text).map_err(|e| format!("journal {}: {e}", journal.display()))?;
    if j.done {
        return Ok(Some(j.spans));
    }
    let gate_e = pack
        .entry(&n.gate)
        .ok_or_else(|| format!("missing {}", n.gate))?
        .clone();
    let up_e = pack
        .entry(&n.up)
        .ok_or_else(|| format!("missing {}", n.up))?
        .clone();
    let down_e = pack
        .entry(&n.down)
        .ok_or_else(|| format!("missing {}", n.down))?
        .clone();
    let (hidden, inter) = (gate_e.shape[0], gate_e.shape[1]);
    for (i, e) in [&gate_e, &up_e, &down_e].into_iter().enumerate() {
        let vals = pack.read_f32(e)?;
        if fingerprint(&vals) == j.before[i] {
            let permuted = if i == 2 {
                permute_rows(&vals, inter, hidden, &j.perm)
            } else {
                permute_cols(&vals, hidden, inter, &j.perm)
            };
            pack.rewrite_entry(e, &permuted)?;
        }
    }
    let done = LayerJournal { done: true, ..j };
    std::fs::write(
        journal,
        serde_json::to_string(&done).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(Some(done.spans))
}

/// Bytes one cluster of a layer's FFN costs at each stored level (gate +
/// up + down slices), for the tier planner.
///
/// # Errors
///
/// When the pack has no FFN partition, `layer` is past the partitioned
/// layers, or one of the layer's three entries is missing from the
/// manifest.
pub fn cluster_costs(pack: &Pack, layer: usize) -> Result<Vec<crate::tier::ExpertCost>, String> {
    let part = pack
        .manifest
        .ffn_partition
        .as_ref()
        .ok_or("pack has no FFN partition")?;
    let names = part.names.get(layer).ok_or("layer out of range")?;
    let spans = &part.layers[layer];
    let mut per_neuron: BTreeMap<Precision, u64> = BTreeMap::new();
    for name in names {
        let e = pack.entry(name).ok_or_else(|| format!("missing {name}"))?;
        let inter = if e.shape[0] > e.shape[1] {
            e.shape[0]
        } else {
            e.shape[1]
        };
        let numel = e.shape.iter().product::<usize>() as u64;
        for (&p, blob) in &e.precisions {
            let bytes = match p {
                Precision::Q4 | Precision::Q8 => blob.values_len + blob.scales_len,
                Precision::F32 => numel * 4,
                // f16 is TWO bytes. Costing it as four made it look exactly
                // as expensive as f32 to the tier planner, so it could never
                // win a placement — which is why an f16 rung never appeared
                // in a plan despite being the cheapest lossless option for a
                // quantized source.
                Precision::F16 => numel * 2,
            };
            // Bytes per neuron (column/row) — the entry is [hidden, inter] or [inter, hidden].
            *per_neuron.entry(p).or_insert(0) += bytes / inter as u64;
        }
    }
    Ok(spans
        .iter()
        .map(|s| crate::tier::ExpertCost {
            bytes: per_neuron
                .iter()
                .map(|(&p, &b)| (p, b * s.len as u64))
                .collect(),
        })
        .collect())
}

/// Quantize-and-pack helper re-exported for the rewrite path's tests.
#[doc(hidden)]
#[must_use]
pub fn requantize(values: &[f32], last_dim: usize, p: Precision) -> (Vec<i8>, Vec<f32>) {
    quantize_blocks(values, last_dim, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mummu_num::f32_from_u32;

    #[test]
    fn cluster_count_keeps_whole_blocks() {
        assert_eq!(cluster_count(6144, 32, 32), 32);
        assert_eq!(cluster_count(17408, 32, 32), 32); // 544 blocks = 32 × 17
        assert_eq!(cluster_count(17408, 64, 32), 34); // 544 = 34 × 16
        assert_eq!(cluster_count(96, 32, 32), 3);
        assert_eq!(cluster_count(32, 32, 32), 1);
    }

    #[test]
    fn balanced_kmeans_is_balanced_and_groups_obvious_clusters() {
        // 4 clear groups of 8 points in 2-D.
        let mut f = Vec::new();
        for g in 0..4u8 {
            for i in 0..8u8 {
                let group = f32::from(g) * 10.0;
                let jitter_x = f32::from(i) * 0.01;
                let jitter_y = f32::from(i) * 0.02;
                f.push(group + jitter_x);
                f.push(-group + jitter_y);
            }
        }
        let a = balanced_kmeans(&f, 32, 2, 4);
        for g in 0..4 {
            let first = a[g * 8];
            assert!(
                (0..8).all(|i| a[g * 8 + i] == first),
                "group {g} split: {a:?}"
            );
        }
        let mut counts = [0; 4];
        for &c in &a {
            counts[c] += 1;
        }
        assert_eq!(counts, [8; 4]);
    }

    /// A folded checkpoint's pack keeps its neuron order: partitioning is
    /// refused by name, and the loaders' skip predicate agrees.
    #[test]
    fn a_hadamard_folded_pack_is_never_partitioned() {
        use crate::gguf::tests::TestGguf;
        let _serial = crate::progress::test_serial();
        let payload: Vec<u8> = (0..256u32 * 256)
            .flat_map(|i| (f32_from_u32(i % 7) * 0.1).to_le_bytes())
            .collect();
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen35")
            .kv_u32("prism.hadamard.version", 1)
            .kv_u32("prism.hadamard.block_size", 8)
            .kv_str(
                "prism.hadamard.transform",
                "normalized-sylvester-walsh-hadamard",
            )
            .kv_str("prism.hadamard.axis", "input-last-dimension")
            .kv_str("prism.hadamard.sign_mode", "identity")
            .kv_str_array("prism.hadamard.weight_names", &["blk.0.ffn_up.weight"])
            .tensor("blk.0.ffn_up.weight", &[256, 256], 0, 0)
            .build_with_payload(&payload);
        let root = std::env::temp_dir().join(format!("mummu-part-folded-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("root");
        let gguf = root.join("t.gguf");
        std::fs::write(&gguf, &bytes).expect("gguf");
        let out = root.join("pack");
        crate::pack::import_gguf(
            &gguf,
            &out,
            &[Precision::Q4],
            &|_| Some(crate::pack::ImportAction::Linear),
            |_, _, _| {},
        )
        .expect("imports");
        let mut pack = Pack::open(&out).expect("opens");
        assert!(pack_is_hadamard_folded(&pack).expect("header parses"));
        let err = partition_pack(&mut pack, &ffn_names(1), DEFAULT_CLUSTERS, |_, _| {})
            .expect_err("refused");
        assert!(err.contains("Hadamard"), "{err}");
        assert!(pack.manifest.ffn_partition.is_none());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn permutation_covers_every_neuron_once_in_contiguous_spans() {
        let (hidden, inter) = (6, 64);
        let gate: Vec<f32> = (0..hidden * inter)
            .map(|i| (f32_from_usize(i) * 0.3).sin())
            .collect();
        let up: Vec<f32> = (0..hidden * inter)
            .map(|i| (f32_from_usize(i) * 0.7).cos())
            .collect();
        let (perm, spans) = cluster_neurons(&gate, &up, hidden, inter, 4, 1);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..inter).collect::<Vec<_>>());
        assert_eq!(spans.len(), 4);
        assert!(spans.iter().all(|s| s.len == 16));
        assert_eq!(
            spans.iter().map(|s| s.start).collect::<Vec<_>>(),
            vec![0, 16, 32, 48]
        );
        // Permuting columns then rows keeps the SwiGLU sum: check one input.
        let x: Vec<f32> = (0..hidden)
            .map(|h| (f32_from_usize(h) + 1.0) * 0.1)
            .collect();
        let down: Vec<f32> = (0..inter * hidden)
            .map(|i| (f32_from_usize(i) * 0.11).sin())
            .collect();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let dense = |g: &[f32], u: &[f32], d: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; hidden];
            for j in 0..inter {
                let gj: f32 = (0..hidden).map(|h| x[h] * g[h * inter + j]).sum();
                let uj: f32 = (0..hidden).map(|h| x[h] * u[h * inter + j]).sum();
                let a = silu(gj) * uj;
                for h in 0..hidden {
                    let term = a * d[j * hidden + h];
                    out[h] += term;
                }
            }
            out
        };
        let ref_out = dense(&gate, &up, &down);
        let p_out = dense(
            &permute_cols(&gate, hidden, inter, &perm),
            &permute_cols(&up, hidden, inter, &perm),
            &permute_rows(&down, inter, hidden, &perm),
        );
        for (a, b) in ref_out.iter().zip(&p_out) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
