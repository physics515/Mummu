//! **NVMe as a cache tier in front of the model's home disk.**
//!
//! A pack can live on bulk storage — on this host `/mnt/deepmem` is a
//! 4-device btrfs over four SPINNING disks, shared with a household server
//! stack — while the machine also has a fast NVMe with far less capacity
//! than the model set. Neither "keep everything on NVMe" (a 207 GB pack and
//! a 111 GB model do not both fit, and copying is manual) nor "read
//! everything from the array" (measured: a qwen4exp reference paging its
//! working set off that array ran at 63 s/token) is right.
//!
//! So: leave the pack where it lives, and cache the ranges actually read on
//! NVMe.
//!
//! # Why keying on the exact range works here
//!
//! A general cache must guess its block size. This one does not have to: a
//! pack is read through [`crate::pack::Pack::read_range`] at
//! `(precision, offset, len)` triples that are **identical every token** —
//! the same tensor, the same slice. Caching whole requested ranges therefore
//! gives an exact hit or an exact miss, with no partial-block reassembly and
//! no read amplification from a block size that fits nothing.
//!
//! # Why not plain LRU
//!
//! MoE routing is skewed, and plain LRU throws away a hot expert after one
//! cold burst (llama.cpp PR #25294 reports the same and evicts on decaying
//! hotness instead). Entries here carry a hit count that **decays** as other
//! entries are touched, and eviction takes the coldest, using
//! least-recently-used only to break ties.
//!
//! # What this is not, yet
//!
//! Read-through only: it populates on miss and never prefetches. The
//! access order IS known in advance (manifest order, and
//! [`crate::workingset`] already exploits exactly that for the RAM->VRAM
//! tier), so preloading is the natural next increment rather than a
//! different design. Reads also go through the OS page cache; the research
//! on this path says `O_DIRECT` starts to matter once the model exceeds RAM,
//! which is a change to make with a measurement in hand, not on principle.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Largest single range this will cache. A pack's biggest tensors are a few
/// hundred MB; anything past this is streamed straight through rather than
/// evicting a large share of the cache for one entry.
const MAX_ENTRY_BYTES: u64 = 512 << 20;

/// Default ceiling when `MUMMU_DISK_CACHE_GB` is unset.
const DEFAULT_CAPACITY_BYTES: u64 = 64 << 30;

/// Hotness added on a hit, and the decay applied to every other entry's
/// hotness when one is touched. Chosen so a single cold burst cannot evict
/// an expert that has been hot for many tokens: at 0.99 an entry needs ~70
/// untouched accesses to halve.
const HIT_BONUS: f64 = 1.0;
const DECAY: f64 = 0.99;

/// One cached range, as bookkeeping. The bytes live in a file.
#[derive(Debug, Clone)]
struct Entry {
    bytes: u64,
    hotness: f64,
    /// Monotonic touch counter — the LRU tiebreak, not a wall clock.
    last_touch: u64,
}

/// Bounded NVMe-backed cache of pack blob ranges.
#[derive(Debug)]
pub struct DiskCache {
    dir: PathBuf,
    capacity_bytes: u64,
    state: Mutex<State>,
    hits: AtomicU64,
    misses: AtomicU64,
    bytes_served: AtomicU64,
    bytes_written: AtomicU64,
}

#[derive(Debug, Default)]
struct State {
    entries: HashMap<String, Entry>,
    used_bytes: u64,
    clock: u64,
}

/// Configuration from the environment, when the operator asked for it.
///
/// Opt-in on purpose: this writes tens of GB to whatever filesystem it is
/// pointed at, which is not a thing to start doing to someone's root volume
/// because a model happened to load.
#[must_use]
pub fn from_env() -> Option<DiskCache> {
    let dir = std::env::var_os("MUMMU_DISK_CACHE_DIR")?;
    let capacity = std::env::var("MUMMU_DISK_CACHE_GB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(DEFAULT_CAPACITY_BYTES, |gb| gb << 30);
    DiskCache::open(Path::new(&dir), capacity).ok()
}

impl DiskCache {
    /// Open (creating if needed) a cache rooted at `dir` holding at most
    /// `capacity_bytes`.
    ///
    /// # Errors
    /// If `dir` cannot be created.
    pub fn open(dir: &Path, capacity_bytes: u64) -> Result<Self, String> {
        assert!(
            capacity_bytes > 0,
            "a zero-capacity cache is a bug, not a config"
        );
        std::fs::create_dir_all(dir).map_err(|e| format!("disk cache {}: {e}", dir.display()))?;
        // Start from empty bookkeeping over a possibly non-empty directory:
        // entries left by a previous process have no hotness history, and
        // adopting them without it would let stale ranges outrank live ones.
        // They are reclaimed by `reset_dir` below.
        let cache = Self {
            dir: dir.to_path_buf(),
            capacity_bytes,
            state: Mutex::new(State::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bytes_served: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
        };
        cache.reset_dir();
        Ok(cache)
    }

    /// Drop anything a previous run left behind, so `used_bytes` and the
    /// directory agree. Failures are ignored on purpose: a cache that cannot
    /// clean up is still a correct cache, just a smaller one.
    fn reset_dir(&self) {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for e in rd.flatten() {
            if e.path().extension().is_some_and(|x| x == "blk") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    /// Cache-file name for a range. Includes every component of the identity
    /// so two different ranges can never collide on one file.
    fn key_of(blob: &str, offset: u64, len: u64) -> String {
        format!("{blob}-{offset:016x}-{len:016x}")
    }

    fn path_of(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.blk"))
    }

    /// Bytes for a range if cached. `None` is a miss — never an error: a
    /// cache that cannot read its own file must fall through to the real
    /// source, not fail the load.
    #[must_use]
    pub fn get(&self, blob: &str, offset: u64, len: u64) -> Option<Vec<u8>> {
        let key = Self::key_of(blob, offset, len);
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !guard.entries.contains_key(&key) {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        // Read before committing to the hit: a truncated or missing file
        // means the entry is a lie, so drop it and report a miss.
        let mut buf = Vec::new();
        let ok = std::fs::File::open(self.path_of(&key))
            .and_then(|mut f| f.read_to_end(&mut buf))
            .is_ok_and(|n| n as u64 == len);
        if !ok {
            if let Some(e) = guard.entries.remove(&key) {
                guard.used_bytes = guard.used_bytes.saturating_sub(e.bytes);
            }
            let _ = std::fs::remove_file(self.path_of(&key));
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Self::touch(&mut guard, &key);
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.bytes_served.fetch_add(len, Ordering::Relaxed);
        debug_assert_eq!(buf.len() as u64, len, "hit returns exactly the range");
        Some(buf)
    }

    /// Record a hit: raise this entry's hotness, decay everyone else's, and
    /// advance the tiebreak clock.
    fn touch(state: &mut State, key: &str) {
        state.clock += 1;
        let clock = state.clock;
        for (k, e) in state.entries.iter_mut() {
            if k == key {
                e.hotness += HIT_BONUS;
                e.last_touch = clock;
            } else {
                e.hotness *= DECAY;
            }
        }
    }

    /// Offer a freshly-read range to the cache. Best-effort: any failure
    /// leaves the cache smaller and the caller unaffected.
    pub fn put(&self, blob: &str, offset: u64, len: u64, bytes: &[u8]) {
        if len > MAX_ENTRY_BYTES || len == 0 || bytes.len() as u64 != len {
            return;
        }
        if len > self.capacity_bytes {
            return; // cannot ever fit; do not evict everything trying
        }
        let key = Self::key_of(blob, offset, len);
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if guard.entries.contains_key(&key) {
            return;
        }
        self.evict_until_fits(&mut guard, len);
        // Write to a temp name then rename, so a crash mid-write cannot
        // leave a short file that a later `get` would trust by size alone.
        let tmp = self.dir.join(format!("{key}.tmp"));
        let written = std::fs::File::create(&tmp)
            .and_then(|mut f| f.write_all(bytes).map(|()| f))
            .and_then(|f| f.sync_all())
            .and_then(|()| std::fs::rename(&tmp, self.path_of(&key)));
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        guard.clock += 1;
        let clock = guard.clock;
        guard.entries.insert(
            key,
            Entry {
                bytes: len,
                hotness: HIT_BONUS,
                last_touch: clock,
            },
        );
        guard.used_bytes += len;
        self.bytes_written.fetch_add(len, Ordering::Relaxed);
        debug_assert!(
            guard.used_bytes <= self.capacity_bytes,
            "eviction ran before insert"
        );
    }

    /// Evict coldest-first until `need` more bytes fit.
    ///
    /// Coldest = lowest hotness, least-recently-touched breaking ties. Plain
    /// LRU is deliberately NOT used: MoE routing is skewed, and LRU discards
    /// a persistently hot expert after one cold burst.
    fn evict_until_fits(&self, state: &mut State, need: u64) {
        assert!(
            need <= self.capacity_bytes,
            "caller checked the entry can fit"
        );
        while state.used_bytes + need > self.capacity_bytes {
            let victim = state
                .entries
                .iter()
                .min_by(|(_, a), (_, b)| {
                    a.hotness
                        .partial_cmp(&b.hotness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.last_touch.cmp(&b.last_touch))
                })
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else {
                // Nothing left to evict but still short: the bookkeeping and
                // the capacity disagree, so stop rather than spin.
                debug_assert_eq!(state.used_bytes, 0, "empty cache reports no usage");
                return;
            };
            if let Some(e) = state.entries.remove(&victim) {
                state.used_bytes = state.used_bytes.saturating_sub(e.bytes);
            }
            let _ = std::fs::remove_file(self.path_of(&victim));
        }
    }

    /// `(hits, misses, bytes served from cache, bytes written to cache)`.
    #[must_use]
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.bytes_served.load(Ordering::Relaxed),
            self.bytes_written.load(Ordering::Relaxed),
        )
    }

    /// Fraction of lookups served from cache, or `None` before any lookup.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        let (h, m, _, _) = self.stats();
        let total = h + m;
        (total > 0).then(|| h as f64 / total as f64)
    }

    /// Bytes currently held.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .used_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_cache(capacity: u64) -> (DiskCache, PathBuf) {
        use std::sync::atomic::AtomicU64;
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mummu-diskcache-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let c = DiskCache::open(&dir, capacity).expect("cache opens");
        (c, dir)
    }

    #[test]
    fn a_stored_range_comes_back_byte_identical() {
        let (c, dir) = tmp_cache(1 << 20);
        let bytes: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        assert!(c.get("q4", 0, 1024).is_none(), "cold lookup misses");
        c.put("q4", 0, 1024, &bytes);
        assert_eq!(c.get("q4", 0, 1024).as_deref(), Some(&bytes[..]));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ranges_with_the_same_offset_but_different_length_do_not_collide() {
        // The identity is (blob, offset, len) — a shorter read at the same
        // offset is a DIFFERENT range, and serving one for the other would
        // be silent corruption rather than a miss.
        let (c, dir) = tmp_cache(1 << 20);
        c.put("q4", 64, 8, &[1u8; 8]);
        c.put("q4", 64, 16, &[2u8; 16]);
        assert_eq!(c.get("q4", 64, 8).as_deref(), Some(&[1u8; 8][..]));
        assert_eq!(c.get("q4", 64, 16).as_deref(), Some(&[2u8; 16][..]));
        // And a different blob at the same coordinates is different again.
        assert!(c.get("q8", 64, 8).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn eviction_keeps_the_hot_entry_and_drops_the_cold_one() {
        // The MoE case the design exists for: one expert routed repeatedly,
        // others touched once. Plain LRU would evict the hot entry as soon
        // as it was not the most recent; hotness must not.
        // Room for exactly TWO entries, so the third insert must evict one.
        let (c, dir) = tmp_cache(2 * 1024);
        c.put("q4", 0, 1024, &[1u8; 1024]); // the hot one
        c.put("q4", 1024, 1024, &[2u8; 1024]);
        for _ in 0..10 {
            assert!(c.get("q4", 0, 1024).is_some(), "hot entry stays resident");
        }
        // A third insert must evict something; it must not be the hot entry,
        // even though it is now the least RECENTLY inserted.
        c.put("q4", 2048, 1024, &[3u8; 1024]);
        assert!(c.get("q4", 0, 1024).is_some(), "hot survived");
        assert!(c.get("q4", 1024, 1024).is_none(), "cold was evicted");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_cache_never_exceeds_its_capacity() {
        let (c, dir) = tmp_cache(4 * 1024);
        for i in 0..20u64 {
            c.put("q4", i * 1024, 1024, &[i as u8; 1024]);
            assert!(
                c.used_bytes() <= 4 * 1024,
                "used {} past the 4096 ceiling",
                c.used_bytes()
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_entry_larger_than_the_whole_cache_is_declined_not_ruinous() {
        // Accepting it would evict everything and still not fit.
        let (c, dir) = tmp_cache(1024);
        c.put("q4", 0, 512, &[7u8; 512]);
        c.put("q4", 4096, 4096, &[9u8; 4096]);
        assert!(c.get("q4", 4096, 4096).is_none(), "oversized declined");
        assert!(c.get("q4", 0, 512).is_some(), "existing entry survived");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_deleted_cache_file_reads_as_a_miss_not_a_corrupt_hit() {
        // Something else cleaning /var/tmp must degrade to a miss, never to
        // short or stale bytes handed back as a tensor.
        let (c, dir) = tmp_cache(1 << 20);
        c.put("q4", 0, 64, &[5u8; 64]);
        for e in std::fs::read_dir(&dir).expect("read dir").flatten() {
            if e.path().extension().is_some_and(|x| x == "blk") {
                std::fs::remove_file(e.path()).expect("remove");
            }
        }
        assert!(c.get("q4", 0, 64).is_none(), "vanished file is a miss");
        assert_eq!(c.used_bytes(), 0, "bookkeeping dropped the dead entry");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn hit_rate_is_none_before_any_lookup_then_tracks_lookups() {
        let (c, dir) = tmp_cache(1 << 20);
        assert!(c.hit_rate().is_none(), "no lookups yet");
        c.put("q4", 0, 8, &[1u8; 8]);
        assert!(c.get("q4", 0, 8).is_some());
        assert!(c.get("q4", 99, 8).is_none());
        let r = c.hit_rate().expect("lookups happened");
        assert!((r - 0.5).abs() < 1e-9, "one hit, one miss => 0.5, got {r}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
