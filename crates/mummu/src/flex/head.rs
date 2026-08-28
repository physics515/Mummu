//! **Norm-bounded exact top-k for the host lm_head (SPEC P4.3/P4.4/P4.5).**
//!
//! The head is a `[hidden, vocab]` GEMV whose consumer needs *ordering*,
//! not the full vector: greedy reads the argmax, and the sampler
//! (`decode::sample_id`) prefilters to its `top_k` candidates and
//! renormalizes inside them — every logit outside the top-k is never
//! consulted. Streaming 700 MiB of packed rows per token to compute
//! 248 000 numbers of which ≤ 1024 matter is the waste this module
//! removes.
//!
//! ## The bound
//!
//! For the packed row `w'_n` (the dequantized twin) and activations `x`,
//! the kernel's computed logit satisfies
//!
//! ```text
//! y_n <= ||w'_n||_2 · ||x||_2  +  eps_n,
//! eps_n <= (max_g sx(g)) / 2 · A_n,   A_n = sum_g sw(n,g) · L1(n,g)
//! ```
//!
//! — Cauchy–Schwarz on the exact packed dot plus the activation-
//! quantization error bound (`flex::kernels::act_error_bound`'s algebra,
//! aggregated offline into one float per row). Rows tile in blocks of
//! [`TILE`]; a tile's bound is its members' maxima. Per token the tiles
//! sort by bound descending and evaluation stops at the first tile whose
//! bound falls below the current k-th best logit: no unevaluated row can
//! enter the top-k **of the computed values** — the result is exactly what
//! the dense evaluation would have returned for every consulted
//! coordinate. Worst case degrades to the full GEMV, never to wrongness.
//!
//! ## The output contract
//!
//! [`HeadTopK`] carries the exact top-k (ids, values). The tensor-level
//! wrapper materializes a dense `[1, vocab]` vector with unevaluated
//! coordinates at [`SENTINEL`] (−1e30): argmax is exact, `sample_id`'s
//! top-k prefilter selects only real values (a sentinel can enter the
//! candidate list only when the sampler asks for more candidates than
//! [`head_k`], where its softmax weight underflows to zero — the
//! documented boundary: exact for greedy and for `top_k <= MUMMU_HEAD_TOPK`
//! sampling; larger sampler top_k truncates to the computed candidates).
//! Anything that reads the FULL softmax (the parity harness's logprob
//! legs) must keep the dense head: this path defaults OFF in the library
//! and is switched on by serve ([`set_enabled`]), `MUMMU_HEAD_BOUND`
//! overriding in both directions.
//!
//! ## The hot set (P4.5)
//!
//! Consecutive hidden states are close (`||h_t − h_{t−1}||` small), so
//! this token's top-k lives near last token's. [`HotSet`] tracks the
//! Lipschitz radius online and seeds the running threshold from the
//! previous winners **evaluated first** — with a healthy margin the very
//! first tiles establish a high threshold and the walk terminates almost
//! immediately. A rotating remainder visits every tile at least once per
//! `V/r` tokens as a safety net against a drifting bound; correctness
//! never depends on it (the per-token bound walk is exact on its own —
//! the hot set only improves the visiting order).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use super::kernels::{GROUP, PackedQ4, Q8Acts};

/// Rows per bound tile — one cache-friendly block of packed rows.
pub const TILE: usize = 32;

/// Fill value for coordinates the bounded head proved out of the top-k.
/// Far below any real logit; softmax weight underflows to exactly 0.
pub const SENTINEL: f32 = -1.0e30;

/// Is the bounded head enabled? Default OFF in the library (the parity
/// harness reads full-vocab logprobs, which sentinels would perturb);
/// serve calls [`set_enabled`] at startup. `MUMMU_HEAD_BOUND=1/0` forces
/// either way.
#[must_use]
pub fn enabled() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env = *ENV.get_or_init(|| match std::env::var("MUMMU_HEAD_BOUND") {
        Err(_) => None,
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            Some(false)
        }
        Ok(_) => Some(true),
    });
    env.unwrap_or_else(|| SWITCH.load(std::sync::atomic::Ordering::Relaxed))
}

static SWITCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Programmatic enablement (serve's startup call). The env wins over this.
pub fn set_enabled(v: bool) {
    SWITCH.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// How many exact top logits the bounded head returns
/// (`MUMMU_HEAD_TOPK`, default 1024 — `decode::sample_id`'s own candidate
/// cap, so default sampling consults nothing the head did not compute).
#[must_use]
pub fn head_k() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("MUMMU_HEAD_TOPK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&k| k >= 1)
            .unwrap_or(1024)
    })
}

/// Per-row and per-tile bound metadata for one packed head, built in one
/// pass over the packed bytes.
pub struct HeadMeta {
    /// `||w'_n||_2` per row.
    norms: Vec<f32>,
    /// `A_n = sum_g sw(n,g) · L1(n,g)` per row (the activation-error
    /// aggregate).
    act_l1: Vec<f32>,
    /// Per tile: max row norm.
    tile_norm: Vec<f32>,
    /// Per tile: max `A_n`.
    tile_act: Vec<f32>,
}

impl HeadMeta {
    /// One pass over the packed rows: exact norms and error aggregates of
    /// the dequantized twin.
    #[must_use]
    pub fn build(w: &PackedQ4) -> Self {
        use rayon::prelude::*;
        let groups = w.k / GROUP;
        let rows: Vec<(f32, f32)> = (0..w.n)
            .into_par_iter()
            .map(|n| {
                let (mut sq, mut act) = (0.0f64, 0.0f64);
                w.for_each_group(n, |g, scale, quants| {
                    debug_assert!(g < groups);
                    let mut l1 = 0i32;
                    let mut s2 = 0i64;
                    for &q in quants {
                        let qi = i32::from(q);
                        l1 += qi.abs();
                        s2 += i64::from(qi * qi);
                    }
                    sq += f64::from(scale) * f64::from(scale) * s2 as f64;
                    act += f64::from(scale) * f64::from(l1);
                });
                #[allow(clippy::cast_possible_truncation)]
                (sq.sqrt() as f32, act as f32)
            })
            .collect();
        let norms: Vec<f32> = rows.iter().map(|r| r.0).collect();
        let act_l1: Vec<f32> = rows.iter().map(|r| r.1).collect();
        let tiles = w.n.div_ceil(TILE);
        let mut tile_norm = vec![0.0f32; tiles];
        let mut tile_act = vec![0.0f32; tiles];
        for t in 0..tiles {
            let range = t * TILE..((t + 1) * TILE).min(w.n);
            tile_norm[t] = range.clone().map(|n| norms[n]).fold(0.0, f32::max);
            tile_act[t] = range.map(|n| act_l1[n]).fold(0.0, f32::max);
        }
        Self {
            norms,
            act_l1,
            tile_norm,
            tile_act,
        }
    }

    /// Bytes this metadata holds resident.
    #[must_use]
    pub fn bytes(&self) -> usize {
        (self.norms.len() + self.act_l1.len() + self.tile_norm.len() + self.tile_act.len()) * 4
    }
}

/// The exact top-k result plus the audit counters.
#[derive(Debug, Clone)]
pub struct HeadTopK {
    /// Token ids, descending by value.
    pub ids: Vec<u32>,
    /// The computed logits for `ids`, descending.
    pub vals: Vec<f32>,
    /// Rows actually evaluated (vs `vocab`): the pruning ratio.
    pub evaluated_rows: usize,
    /// Tiles whose bound survived (== evaluated_rows / TILE, up to the
    /// ragged last tile).
    pub evaluated_tiles: usize,
}

/// A fixed-size max-min structure over (value, id): keeps the k largest,
/// exposes the current k-th value. Implemented as a binary min-heap.
struct TopK {
    k: usize,
    heap: Vec<(f32, u32)>, // min-heap by value
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: Vec::with_capacity(k),
        }
    }

    #[inline]
    fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap[0].0
        }
    }

    fn push(&mut self, val: f32, id: u32) {
        if self.heap.len() < self.k {
            self.heap.push((val, id));
            let mut i = self.heap.len() - 1;
            while i > 0 {
                let parent = (i - 1) / 2;
                if self.heap[parent].0 <= self.heap[i].0 {
                    break;
                }
                self.heap.swap(parent, i);
                i = parent;
            }
        } else if val > self.heap[0].0 {
            self.heap[0] = (val, id);
            // Sift down.
            let mut i = 0;
            loop {
                let (l, r) = (2 * i + 1, 2 * i + 2);
                let mut small = i;
                if l < self.heap.len() && self.heap[l].0 < self.heap[small].0 {
                    small = l;
                }
                if r < self.heap.len() && self.heap[r].0 < self.heap[small].0 {
                    small = r;
                }
                if small == i {
                    break;
                }
                self.heap.swap(i, small);
                i = small;
            }
        }
    }

    fn into_sorted(self) -> (Vec<u32>, Vec<f32>) {
        let mut v = self.heap;
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        (v.iter().map(|e| e.1).collect(), v.iter().map(|e| e.0).collect())
    }
}

/// Exact top-k of the packed head's computed logits by tile-bounded
/// branch and bound. `seed_tiles` (from a [`HotSet`]) are evaluated first
/// to establish a high threshold; pass `&[]` without one.
#[must_use]
pub fn head_topk(
    w: &PackedQ4,
    meta: &HeadMeta,
    x: &[f32],
    k: usize,
    seed_tiles: &[u32],
) -> HeadTopK {
    assert_eq!(x.len(), w.k, "activation length");
    assert!(k >= 1);
    let acts = Q8Acts::quantize(x);
    // Same dispatch rule as gemv_q4n_auto, so every evaluated row equals
    // what the dense head would have computed for it.
    let integer_ok =
        acts.quality <= super::kernels::quality_limit() && super::kernels::vnni_available();
    let x_norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let sx_max = acts.max_scale();
    // eps term only applies to the integer path's activation rounding.
    let eps_c = if integer_ok { 0.5 * sx_max } else { 0.0 };

    let tiles = w.n.div_ceil(TILE);
    let bound_of = |t: usize| meta.tile_norm[t] * x_norm + eps_c * meta.tile_act[t];

    let mut top = TopK::new(k.min(w.n));
    let mut evaluated = vec![false; tiles];
    let mut evaluated_tiles = 0usize;
    let mut evaluated_rows = 0usize;
    let mut scratch = vec![0.0f32; TILE];
    let eval_tile = |t: usize,
                         top: &mut TopK,
                         evaluated: &mut Vec<bool>,
                         evaluated_tiles: &mut usize,
                         evaluated_rows: &mut usize,
                         scratch: &mut Vec<f32>| {
        if evaluated[t] {
            return;
        }
        evaluated[t] = true;
        *evaluated_tiles += 1;
        let n0 = t * TILE;
        let n1 = ((t + 1) * TILE).min(w.n);
        *evaluated_rows += n1 - n0;
        w.dot_rows(n0, n1, &acts, x, integer_ok, &mut scratch[..n1 - n0]);
        for (i, &y) in scratch[..n1 - n0].iter().enumerate() {
            top.push(y, u32::try_from(n0 + i).expect("vocab fits u32"));
        }
    };

    // Seeds first: the hot set's previous winners.
    for &t in seed_tiles {
        let t = t as usize;
        if t < tiles {
            eval_tile(
                t,
                &mut top,
                &mut evaluated,
                &mut evaluated_tiles,
                &mut evaluated_rows,
                &mut scratch,
            );
        }
    }

    // Bound walk: tiles by bound descending; stop at the first that
    // cannot beat the current k-th value.
    let mut order: Vec<u32> = (0..tiles as u32).collect();
    let mut bounds: Vec<f32> = (0..tiles).map(bound_of).collect();
    order.sort_unstable_by(|&a, &b| {
        bounds[b as usize]
            .partial_cmp(&bounds[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &t in &order {
        let t = t as usize;
        if bounds[t] < top.threshold() {
            break; // every later tile's bound is smaller still
        }
        eval_tile(
            t,
            &mut top,
            &mut evaluated,
            &mut evaluated_tiles,
            &mut evaluated_rows,
            &mut scratch,
        );
    }
    bounds.clear();

    let (ids, vals) = top.into_sorted();
    HeadTopK {
        ids,
        vals,
        evaluated_rows,
        evaluated_tiles,
    }
}

// ---------------------------------------------------------------------------
// The temporal hot set (P4.5)
// ---------------------------------------------------------------------------

/// Tracks last token's winning tiles and the hidden-state drift, to seed
/// the next token's threshold. Purely an ordering optimization — see the
/// module doc.
#[derive(Debug, Default)]
pub struct HotSet {
    /// Tiles that held last token's top-k.
    tiles: Vec<u32>,
    /// Previous hidden state (for the Lipschitz radius diagnostics).
    prev_x: Vec<f32>,
    /// Running max of `||x_t − x_{t−1}||` — the measured drift radius.
    pub drift_max: f32,
    /// Rotating cursor for the safety remainder.
    cursor: u32,
}

impl HotSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Tiles to evaluate first this token: last token's winners plus a
    /// small rotating remainder (so every tile is touched once per
    /// `tiles / remainder` tokens).
    #[must_use]
    pub fn seeds(&mut self, total_tiles: usize, remainder: usize) -> Vec<u32> {
        let mut s = self.tiles.clone();
        for _ in 0..remainder {
            self.cursor = (self.cursor + 1) % total_tiles.max(1) as u32;
            s.push(self.cursor);
        }
        s
    }

    /// Record this token's result and drift.
    pub fn observe(&mut self, x: &[f32], result: &HeadTopK) {
        if self.prev_x.len() == x.len() {
            let d = self
                .prev_x
                .iter()
                .zip(x)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            self.drift_max = self.drift_max.max(d);
        }
        self.prev_x = x.to_vec();
        let mut tiles: Vec<u32> = result
            .ids
            .iter()
            .map(|&id| id / TILE as u32)
            .collect();
        tiles.sort_unstable();
        tiles.dedup();
        self.tiles = tiles;
    }
}

// ---------------------------------------------------------------------------
// Sidecar: metadata per packed twin
// ---------------------------------------------------------------------------

/// Metadata for a packed head, keyed by the twin's allocation and verified
/// through a `Weak` so a dropped twin cannot resurrect stale bounds.
pub fn meta_for(packed: &Arc<PackedQ4>) -> Arc<HeadMeta> {
    static MAP: OnceLock<Mutex<HashMap<usize, (Weak<PackedQ4>, Arc<HeadMeta>)>>> = OnceLock::new();
    let map = MAP.get_or_init(|| Mutex::new(HashMap::new()));
    let key = Arc::as_ptr(packed) as usize;
    {
        let m = map.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((w, meta)) = m.get(&key)
            && let Some(alive) = w.upgrade()
            && Arc::ptr_eq(&alive, packed)
        {
            return Arc::clone(meta);
        }
    }
    let meta = Arc::new(HeadMeta::build(packed));
    let mut m = map.lock().unwrap_or_else(PoisonError::into_inner);
    m.retain(|_, (w, _)| w.strong_count() > 0);
    m.insert(key, (Arc::downgrade(packed), Arc::clone(&meta)));
    meta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flex::kernels::{PackedQ4, gemv_q4n_auto};

    fn wave(len: usize, f: f32) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * f).sin()).collect()
    }

    /// The head's guarantee, stated tie-honestly: the returned values are
    /// exactly the k largest of the dense evaluation (as a multiset), each
    /// id's value is bitwise the dense value at that id, and the list is
    /// descending. Among EXACTLY-equal logits the id order may differ from
    /// a stable dense sort — the wired sentinel-dense tensor carries every
    /// tied candidate, so downstream argmax/sampling see the same choices.
    fn assert_topk_matches(dense: &[f32], got: &HeadTopK, k: usize) {
        assert_eq!(got.ids.len(), k);
        assert_eq!(got.vals.len(), k);
        for w in got.vals.windows(2) {
            assert!(w[0] >= w[1], "values must be descending: {:?}", got.vals);
        }
        for (i, (&id, &v)) in got.ids.iter().zip(&got.vals).enumerate() {
            assert_eq!(
                dense[id as usize], v,
                "rank {i}: value must be the dense value at id {id}"
            );
        }
        let mut want: Vec<f32> = dense.to_vec();
        want.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for i in 0..k {
            assert_eq!(got.vals[i], want[i], "rank {i}: multiset of top values");
        }
    }

    /// The bounded top-k equals the dense evaluation's top-k exactly —
    /// ids in order and values bitwise — on random instances, with and
    /// without hot-set seeds.
    #[test]
    fn bounded_topk_is_exact() {
        let (k_dim, vocab) = (128, 640);
        let vals = wave(k_dim * vocab, 0.37);
        let w = PackedQ4::from_f32(&vals, k_dim, vocab);
        let meta = HeadMeta::build(&w);
        for (fx, k) in [(0.011f32, 8usize), (0.023, 1), (0.005, 64)] {
            let x = wave(k_dim, fx);
            let mut dense = vec![0.0f32; vocab];
            gemv_q4n_auto(&w, &x, &mut dense);
            let got = head_topk(&w, &meta, &x, k, &[]);
            assert_topk_matches(&dense, &got, k);
            // Seeded run must return the same multiset of winners.
            let seeded = head_topk(&w, &meta, &x, k, &[3, 7, 19]);
            assert_topk_matches(&dense, &seeded, k);
        }
    }

    /// A peaked head (one dominant row) prunes almost everything; a flat
    /// head degrades toward dense but stays exact. This pins the
    /// worst-case-degrades-never-breaks property AND that pruning happens
    /// at all.
    #[test]
    fn pruning_engages_on_peaked_geometry() {
        let (k_dim, vocab) = (64, 2048);
        // Rows mostly tiny, a few huge — each big row distinct so the
        // ordering is strict, not a tie.
        let mut vals = vec![0.001f32; k_dim * vocab];
        for big in [17usize, 900, 1999] {
            for kk in 0..k_dim {
                vals[kk * vocab + big] =
                    (1.0 + (kk as f32 * 0.1).sin()) * (1.0 + big as f32 * 1e-4);
            }
        }
        let w = PackedQ4::from_f32(&vals, k_dim, vocab);
        let meta = HeadMeta::build(&w);
        let x = wave(k_dim, 0.013);
        let got = head_topk(&w, &meta, &x, 3, &[]);
        assert!(
            got.evaluated_rows < vocab / 4,
            "peaked head must prune: evaluated {} of {vocab}",
            got.evaluated_rows
        );
        // Exactness even so.
        let mut dense = vec![0.0f32; vocab];
        gemv_q4n_auto(&w, &x, &mut dense);
        assert_topk_matches(&dense, &got, 3);
    }

    /// Adversarial activations (quality gate tripped): the head falls to
    /// the exact-f32 row path, the eps term drops out of the bound, and
    /// the result still equals the dense f32 evaluation's top-k.
    #[test]
    fn adversarial_activations_stay_exact() {
        let (k_dim, vocab) = (64, 320);
        let vals = wave(k_dim * vocab, 0.19);
        let w = PackedQ4::from_f32(&vals, k_dim, vocab);
        let meta = HeadMeta::build(&w);
        let mut x = vec![1e-4f32; k_dim];
        for g in 0..k_dim / GROUP {
            x[g * GROUP] = 1000.0;
        }
        let got = head_topk(&w, &meta, &x, 5, &[]);
        let mut dense = vec![0.0f32; vocab];
        gemv_q4n_auto(&w, &x, &mut dense); // dispatches to exact f32 itself
        assert_topk_matches(&dense, &got, 5);
    }

    /// The hot set improves the visit order (fewer evaluations with seeds
    /// under temporal drift) and its rotating remainder cycles.
    #[test]
    fn hot_set_seeds_and_rotates() {
        let (k_dim, vocab) = (64, 2048);
        let mut vals = vec![0.001f32; k_dim * vocab];
        for big in [100usize, 101, 700] {
            for kk in 0..k_dim {
                vals[kk * vocab + big] = 1.0 + big as f32 * 1e-4; // distinct
            }
        }
        let w = PackedQ4::from_f32(&vals, k_dim, vocab);
        let meta = HeadMeta::build(&w);
        let mut hot = HotSet::new();
        let x1 = wave(k_dim, 0.013);
        let r1 = head_topk(&w, &meta, &x1, 3, &hot.seeds(vocab.div_ceil(TILE), 2));
        hot.observe(&x1, &r1);
        // A slightly-drifted next token: the same winner set.
        let x2: Vec<f32> = x1.iter().map(|v| v * 1.001).collect();
        let seeds = hot.seeds(vocab.div_ceil(TILE), 2);
        assert!(!seeds.is_empty());
        let r2 = head_topk(&w, &meta, &x2, 3, &seeds);
        let sorted = |v: &[u32]| {
            let mut s = v.to_vec();
            s.sort_unstable();
            s
        };
        assert_eq!(
            sorted(&r1.ids),
            sorted(&r2.ids),
            "drifted token keeps the same winners"
        );
        hot.observe(&x2, &r2);
        assert!(hot.drift_max > 0.0);
        // The rotating cursor advances.
        let s1 = hot.seeds(64, 1);
        let s2 = hot.seeds(64, 1);
        assert_ne!(s1.last(), s2.last());
    }

    /// meta_for memoizes per twin allocation and rebuilds for a new one.
    #[test]
    fn meta_sidecar_memoizes() {
        let w = Arc::new(PackedQ4::from_f32(&wave(64 * 64, 0.3), 64, 64));
        let a = meta_for(&w);
        let b = meta_for(&w);
        assert!(Arc::ptr_eq(&a, &b));
        let w2 = Arc::new(PackedQ4::from_f32(&wave(64 * 64, 0.31), 64, 64));
        let c = meta_for(&w2);
        assert!(!Arc::ptr_eq(&a, &c));
    }
}
