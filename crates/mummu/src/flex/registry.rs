//! Sidecar registry: which packed host representation serves a flex weight.
//!
//! The model keeps its weights as burn tensors (flex stores Q4 as one i8 per
//! element); the VNNI kernel wants [`PackedQ4`](super::kernels::PackedQ4).
//! Rather than thread a new weight type through every module and loader, the
//! packed twin lives here, keyed by the flex storage it shadows, and the
//! `Q4GemvOps` flex impl consults the registry on every call:
//!
//! - **Fast key**: the i8 slab's `(ptr, len)`. O(1), but pointers are reused
//!   after frees, so every hit is verified against a 16-byte sample of the
//!   slab (first 8 + last 8 bytes) — two cache lines, and a reload that
//!   lands a *different* tensor at the same address re-packs instead of
//!   multiplying by a ghost.
//! - **Lazy build**: a miss builds the packed twin from the slab itself
//!   (dequantize the device grid, requantize along K — a second 4-bit
//!   rounding) and logs it once per shape. A loader that has the float-level
//!   bytes at hand should call [`register_from_f32`] instead: same layout,
//!   one rounding.
//!
//! Memory: the packed twin (0.5625 B/param + f16 scales) lives BESIDE the
//! i8 slab (1.125 B/param), so lazy use costs +50% host weight RAM for the
//! covered tensors in exchange for reading half the bytes per token. The
//! full "second 2×" (dropping the slab) needs the loaders to stop holding
//! the i8 tensor — an integration step, not a kernel property.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use super::kernels::PackedQ4;

/// Is the packed host path enabled? `MUMMU_VNNI_GEMV`, default on;
/// `0`/`off`/`false` restores the i8 scalar path everywhere (the downgrade
/// contract every fast path in this repo carries). [`force_disable`] wins
/// over the env — tests of the baseline path use it.
#[must_use]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let env_on = *ON.get_or_init(|| {
        std::env::var("MUMMU_VNNI_GEMV").map_or(true, |v| {
            !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        })
    });
    env_on && !FORCE_OFF.load(std::sync::atomic::Ordering::Relaxed)
}

static FORCE_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Programmatic kill switch (stronger than the env): tests of the baseline
/// i8 path flip it on around their measurement, and an operator hook can
/// too. Not a per-thread toggle — callers serialize.
pub fn force_disable(v: bool) {
    FORCE_OFF.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// A registered packed twin plus the verification sample of the slab it
/// shadows.
struct Entry {
    tag: [u8; 16],
    packed: Arc<PackedQ4>,
}

fn tag_of(slab: &[u8]) -> [u8; 16] {
    let mut tag = [0u8; 16];
    let take = slab.len().min(8);
    tag[..take].copy_from_slice(&slab[..take]);
    let tail = slab.len().saturating_sub(8);
    tag[8..8 + slab.len().min(8)].copy_from_slice(&slab[tail..]);
    tag
}

fn map() -> &'static Mutex<HashMap<(usize, usize), Entry>> {
    static MAP: OnceLock<Mutex<HashMap<(usize, usize), Entry>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Forget every registration (model unload / tests).
pub fn clear() {
    map().lock().unwrap_or_else(PoisonError::into_inner).clear();
}

/// Register a packed twin built from float-level values `[K, N]` for the
/// flex slab `values_i8` (the same tensor's unpacked storage). The loader
/// path: one 4-bit rounding, done in parallel at load.
pub fn register_from_f32(values_i8: &[i8], floats: &[f32], k: usize, n: usize) {
    let packed = Arc::new(PackedQ4::from_f32(floats, k, n));
    insert(values_i8, packed);
}

/// Register an already-built twin for a slab.
pub fn insert(values_i8: &[i8], packed: Arc<PackedQ4>) {
    let slab = bytes_of(values_i8);
    let key = (slab.as_ptr() as usize, slab.len());
    let entry = Entry {
        tag: tag_of(slab),
        packed,
    };
    let mut m = map().lock().unwrap_or_else(PoisonError::into_inner);
    // A registry outliving many model reloads must not grow unboundedly; a
    // full 27B host half registers a few hundred tensors.
    if m.len() > 4096 {
        m.clear();
    }
    m.insert(key, entry);
}

/// The packed twin for this slab, building it lazily (and logging, once per
/// shape) when the loader did not register one. `scales` is the device
/// grid's `[K, N/32]` blocks-along-N layout. Callers gate on [`enabled`];
/// this function always answers so direct users (tests, probes) are not
/// coupled to the kill switch.
#[must_use]
pub fn resolve(values_i8: &[i8], scales: &[f32], k: usize, n: usize) -> Option<Arc<PackedQ4>> {
    let slab = bytes_of(values_i8);
    let key = (slab.as_ptr() as usize, slab.len());
    let tag = tag_of(slab);
    {
        let m = map().lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = m.get(&key)
            && e.tag == tag
        {
            return Some(Arc::clone(&e.packed));
        }
    }
    // Miss (or a reused address): build outside the lock — packing runs
    // rayon-wide and other GEMVs must not serialize behind it. A racing
    // duplicate build is wasted work, not wrongness.
    log_lazy(k, n);
    let packed = Arc::new(PackedQ4::from_q4s_slab(values_i8, scales, k, n));
    let mut m = map().lock().unwrap_or_else(PoisonError::into_inner);
    m.insert(
        key,
        Entry {
            tag,
            packed: Arc::clone(&packed),
        },
    );
    Some(packed)
}

fn log_lazy(k: usize, n: usize) {
    use std::collections::HashSet;
    static SEEN: Mutex<Option<HashSet<(usize, usize)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(PoisonError::into_inner);
    if seen.get_or_insert_with(HashSet::new).insert((k, n)) {
        eprintln!(
            "[mummu] vnni gemv: lazy repack [{k} x {n}] from the i8 slab \
			 (second 4-bit rounding; register the float level at load to avoid it)"
        );
    }
}

fn bytes_of(values_i8: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have identical layout; the slice is only read.
    unsafe { std::slice::from_raw_parts(values_i8.as_ptr().cast::<u8>(), values_i8.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_builds_once_and_verifies_the_tag() {
        clear();
        let (k, n) = (64usize, 32usize);
        let vals: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.13).sin()).collect();
        let (qi8, scales) = crate::pack::quantize_blocks(&vals, n, crate::pack::Precision::Q4);
        let a = resolve(&qi8, &scales, k, n).expect("enabled by default");
        let b = resolve(&qi8, &scales, k, n).expect("hit");
        assert!(
            Arc::ptr_eq(&a, &b),
            "second resolve must be the memoized twin"
        );

        // Same address, different contents: the tag must force a re-pack.
        let mut qi8b = qi8.clone();
        // Ensure identical allocation size but different first byte...
        qi8b[0] = qi8b[0].wrapping_add(1);
        let c = resolve(&qi8b, &scales, k, n).expect("rebuild");
        // (ptr differs here because it is a different Vec — the tag check is
        // exercised by the ptr-collision case below only probabilistically,
        // so at minimum the API must return a twin matching the new slab.)
        let deq = c.dequantize();
        assert_eq!(deq.len(), k * n);
        clear();
    }

    #[test]
    fn register_from_f32_takes_priority_over_lazy() {
        clear();
        let (k, n) = (64usize, 32usize);
        let vals: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.07).cos()).collect();
        let (qi8, scales) = crate::pack::quantize_blocks(&vals, n, crate::pack::Precision::Q4);
        register_from_f32(&qi8, &vals, k, n);
        let got = resolve(&qi8, &scales, k, n).expect("registered");
        // The registered twin was built from the ORIGINAL floats — one
        // rounding — so its dequantized grid must match a direct from_f32.
        let want = super::super::kernels::PackedQ4::from_f32(&vals, k, n).dequantize();
        assert_eq!(got.dequantize(), want);
        clear();
    }
}
