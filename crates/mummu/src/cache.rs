//! Process-lifetime model caching. Loading a checkpoint costs seconds and
//! gigabytes, so consumers keep one [`ModelSlot`] static per (model, backend)
//! and pay the load once. Burn's `Param` is not `Sync`, so the loaded value
//! lives behind a `Mutex` and is only reachable inside [`ModelSlot::with`] —
//! which also serializes inference, the right default for a single GPU.
//!
//! Switching to a different checkpoint dir through the same slot drops the
//! old model (freeing its VRAM/RAM) and loads the new one — this is the
//! active-model-switch primitive the P8 management API builds on.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

struct Entry<T> {
    key: PathBuf,
    value: T,
}

/// A one-model cache slot, keyed by checkpoint directory.
pub struct ModelSlot<T> {
    inner: Mutex<Option<Entry<T>>>,
}

impl<T> Default for ModelSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ModelSlot<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// Run `f` with the model for `key`, loading it first if the slot is
    /// empty or holds a different checkpoint (the old model is dropped
    /// before `load` runs, so peak memory stays one model per slot).
    pub fn with<R, E>(
        &self,
        key: &Path,
        load: impl FnOnce(&Path) -> Result<T, E>,
        f: impl FnOnce(&T) -> R,
    ) -> Result<R, E> {
        assert!(!key.as_os_str().is_empty(), "model cache: empty key");
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let hit = guard.as_ref().is_some_and(|e| e.key == key);
        if !hit {
            *guard = None; // free the old model before loading the new one
            let value = load(key)?;
            *guard = Some(Entry {
                key: key.to_path_buf(),
                value,
            });
        }
        let entry = guard.as_ref().expect("slot was just filled");
        debug_assert!(entry.key == key, "slot must hold the requested model");
        Ok(f(&entry.value))
    }

    /// Drop the cached model (freeing its VRAM/RAM). No-op when empty.
    pub fn clear(&self) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// The checkpoint dir currently loaded, if any — for settings UIs.
    #[must_use]
    pub fn loaded_key(&self) -> Option<PathBuf> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|e| e.key.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[test]
    fn loads_once_and_reuses_for_the_same_key() {
        let slot: ModelSlot<String> = ModelSlot::new();
        let mut loads = 0;
        for _ in 0..3 {
            let got = slot
                .with::<_, Infallible>(
                    Path::new("model-a"),
                    |k| {
                        loads += 1;
                        Ok(k.display().to_string())
                    },
                    |m| m.clone(),
                )
                .unwrap();
            assert_eq!(got, "model-a");
        }
        assert_eq!(loads, 1, "same key must load exactly once");
        assert_eq!(slot.loaded_key().as_deref(), Some(Path::new("model-a")));
    }

    #[test]
    fn switching_key_reloads_and_replaces() {
        let slot: ModelSlot<String> = ModelSlot::new();
        let mut loads = 0;
        let mut run = |key: &str| {
            slot.with::<_, Infallible>(
                Path::new(key),
                |k| {
                    loads += 1;
                    Ok(k.display().to_string())
                },
                |m| m.clone(),
            )
            .unwrap()
        };
        assert_eq!(run("model-a"), "model-a");
        assert_eq!(run("model-b"), "model-b"); // switch: drop a, load b
        assert_eq!(run("model-b"), "model-b"); // hit
        assert_eq!(loads, 2);
        assert_eq!(slot.loaded_key().as_deref(), Some(Path::new("model-b")));
    }

    #[test]
    fn failed_load_leaves_the_slot_empty() {
        let slot: ModelSlot<String> = ModelSlot::new();
        let err = slot.with(Path::new("bad"), |_| Err("boom"), |m: &String| m.clone());
        assert_eq!(err, Err("boom"));
        assert_eq!(slot.loaded_key(), None, "a failed load must not cache");
    }

    #[test]
    fn clear_unloads() {
        let slot: ModelSlot<u32> = ModelSlot::new();
        slot.with::<_, Infallible>(Path::new("m"), |_| Ok(7), |_| ())
            .unwrap();
        assert!(slot.loaded_key().is_some());
        slot.clear();
        assert_eq!(slot.loaded_key(), None);
    }

    #[test]
    #[should_panic(expected = "empty key")]
    fn empty_key_is_rejected() {
        let slot: ModelSlot<u32> = ModelSlot::new();
        let _ = slot.with::<_, Infallible>(Path::new(""), |_| Ok(1), |_| ());
    }

    /// The slot is usable as a `static` (the whole point).
    static GLOBAL: ModelSlot<u32> = ModelSlot::new();

    #[test]
    fn works_as_a_static_across_threads() {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    GLOBAL
                        .with::<_, Infallible>(Path::new("shared"), |_| Ok(41), |v| v + 1)
                        .unwrap()
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 42);
        }
    }
}
