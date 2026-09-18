//! Process-lifetime model caching. Loading a checkpoint costs seconds and
//! gigabytes, so consumers keep one [`ModelSlot`] static per (model, backend)
//! and pay the load once. Burn's `Param` is not `Sync`, so the loaded value
//! lives behind a `Mutex` and is only reachable inside [`ModelSlot::with`] /
//! [`ModelSlot::with_async`] — which also serializes inference, the right
//! default for a single GPU.
//!
//! The mutex is tokio's, because the async accessor holds its guard across
//! an await (a `std` guard is not `Send`, so it would not survive one). The
//! sync accessor takes the same lock with `blocking_lock`, which is exactly
//! what it says: callers outside a runtime — tests, examples, the parity
//! harness — block as they always did. One lock, not two: a second one would
//! be a second slot, and the model would load twice.
//!
//! Switching to a different checkpoint dir through the same slot drops the
//! old model (freeing its VRAM/RAM) and loads the new one — this is the
//! active-model-switch primitive the P8 management API builds on.

use std::path::{Path, PathBuf};

use tokio::sync::Mutex;

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
            // `const_new`, not `new`: the slot is used as a `static`, so its
            // constructor must be const (tokio's plain `new` is not).
            inner: Mutex::const_new(None),
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
        let mut guard = self.inner.blocking_lock();
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

    /// Async access to the slot: loads if needed and returns a **guard**
    /// holding the model, so the caller can `.await` across it.
    ///
    /// A closure-taking twin of [`Self::with`] cannot express this: the
    /// future would borrow from the `&T` the closure receives, and
    /// `FnOnce(&T) -> Fut` has no way to tie `Fut`'s lifetime to that
    /// borrow. A guard says the same thing without the higher-ranked
    /// gymnastics — hold it, await through it, drop it to release the slot
    /// (which is what serializes generations and protects VRAM).
    pub async fn acquire<E>(
        &self,
        key: &Path,
        load: impl FnOnce(&Path) -> Result<T, E>,
    ) -> Result<SlotGuard<'_, T>, E> {
        self.acquire_valid(key, |_| true, load).await
    }

    /// [`Self::acquire`], where a resident value only counts as a hit if
    /// `still_valid` says so. One that does not is dropped — before `load`
    /// runs, exactly like a different key — and loaded again.
    ///
    /// # Why the check has to happen here, under the lock
    ///
    /// A model can be resident and broken. mummu-serve's case: a GPU
    /// allocation that failed on cubecl's device thread was swallowed there,
    /// so the loader returned a model whose weights point at device memory
    /// that was never initialized, and every later read of it fails. Asking
    /// "is the resident model still good?" and then acquiring is two lock
    /// acquisitions, and a request queued behind the one that discovered the
    /// failure takes the slot in between and runs on the broken model again.
    /// Deciding inside the same critical section as the key comparison is the
    /// only way the answer and the model it describes cannot come apart.
    pub async fn acquire_valid<E>(
        &self,
        key: &Path,
        still_valid: impl FnOnce(&T) -> bool,
        load: impl FnOnce(&Path) -> Result<T, E>,
    ) -> Result<SlotGuard<'_, T>, E> {
        assert!(!key.as_os_str().is_empty(), "model cache: empty key");
        let mut guard = self.inner.lock().await;
        let hit = guard
            .as_ref()
            .is_some_and(|e| e.key == key && still_valid(&e.value));
        if !hit {
            *guard = None; // free the old model before loading the new one
            // Loading is the one genuinely blocking thing on this path:
            // minutes of CPU-bound work with no await in it, reading weights
            // off disk and placing them across devices. Left on an async
            // worker it starves the runtime — measured, a WebSocket heartbeat
            // elsewhere in the process was not polled ONCE in 557 seconds and
            // then fired 38 missed pings at the end, by which point the
            // connection it existed to keep alive would already be reaped.
            //
            // `block_in_place` hands this worker's other tasks to a sibling
            // thread for the duration, so only this call blocks. Deliberately
            // narrow: the decode loop around it stays async and awaits.
            let value = match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
                Ok(tokio::runtime::RuntimeFlavor::MultiThread) => {
                    tokio::task::block_in_place(|| load(key))?
                }
                // A current-thread runtime (tests) has no sibling worker to
                // hand the other tasks to, and `block_in_place` panics there.
                _ => load(key)?,
            };
            *guard = Some(Entry {
                key: key.to_path_buf(),
                value,
            });
        }
        debug_assert!(
            guard.as_ref().is_some_and(|e| e.key == key),
            "slot must hold the requested model"
        );
        Ok(SlotGuard { guard })
    }

    /// Drop the cached model (freeing its VRAM/RAM). No-op when empty.
    ///
    /// Returns `false` when a generation currently holds the slot — the model
    /// cannot be freed under it, and *saying so* beats the alternatives:
    /// blocking here panics inside a tokio runtime ("cannot block the current
    /// thread from within a runtime"), and waiting would park a worker behind
    /// a decode that can run for minutes. Use [`Self::clear_async`] to wait.
    pub fn clear(&self) -> bool {
        match self.inner.try_lock() {
            Ok(mut guard) => {
                *guard = None;
                true
            }
            Err(_) => false,
        }
    }

    /// [`Self::clear`], waiting for any in-flight generation to release the
    /// slot first.
    pub async fn clear_async(&self) {
        *self.inner.lock().await = None;
    }

    /// The checkpoint dir currently loaded, if any — for settings UIs.
    ///
    /// A **peek**: `None` when the slot is empty *or* busy serving a
    /// generation. Callers use this to answer "is this already loaded?" and
    /// "what is resident?", where waiting behind a multi-minute decode would
    /// be worse than a conservative answer (and blocking inside a runtime
    /// would panic outright).
    #[must_use]
    pub fn loaded_key(&self) -> Option<PathBuf> {
        self.inner
            .try_lock()
            .ok()
            .and_then(|g| g.as_ref().map(|e| e.key.clone()))
    }

    /// [`Self::loaded_key`], waiting for the slot instead of reporting busy.
    pub async fn loaded_key_async(&self) -> Option<PathBuf> {
        self.inner.lock().await.as_ref().map(|e| e.key.clone())
    }
}

/// A held model slot (see [`ModelSlot::acquire`]). Deref to the model;
/// dropping it releases the slot for the next generation.
pub struct SlotGuard<'a, T> {
    guard: tokio::sync::MutexGuard<'a, Option<Entry<T>>>,
}

impl<T> std::ops::Deref for SlotGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self
            .guard
            .as_ref()
            .expect("a slot guard always holds a loaded model")
            .value
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

    /// `loaded_key_async` is a reading, not a reservation: it takes the slot
    /// lock, answers, and **releases it** — so anything that clears the slot
    /// before the caller's own `acquire` re-takes it turns a "warm" answer
    /// into a real load.
    ///
    /// This is not hypothetical bookkeeping. mummu-serve decided from exactly
    /// this answer whether to arm its progress guard, and a `POST /api/unload`
    /// (or the 5 s host-pressure eviction) landing in the window produced a
    /// load with no guard at all: the loaders kept writing their counts to the
    /// global progress state, the bar reached N/N `loading`, and nothing owned
    /// it to finish it. Whatever a caller wants to happen once per load
    /// belongs inside the `load` closure, which runs under the lock and only
    /// on a miss.
    #[tokio::test]
    async fn a_clear_between_the_peek_and_the_acquire_makes_a_warm_answer_cold() {
        use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
        let slot: ModelSlot<String> = ModelSlot::new();
        let key = Path::new("model-a");
        let loads = AtomicU32::new(0);
        let load = |_: &Path| {
            loads.fetch_add(1, SeqCst);
            Ok::<_, Infallible>("model-a".to_owned())
        };

        drop(slot.acquire(key, load).await.unwrap());
        assert_eq!(loads.load(SeqCst), 1);
        assert_eq!(
            slot.loaded_key_async().await.as_deref(),
            Some(key),
            "the peek says warm — and is telling the truth, for now"
        );

        // The window. Nothing is held here; the peek's lock is long gone.
        assert!(slot.clear(), "an unload lands between the two");

        drop(slot.acquire(key, load).await.unwrap());
        assert_eq!(
            loads.load(SeqCst),
            2,
            "the request the peek called warm paid for a load after all"
        );
    }

    /// The invariant `acquire_valid` exists for: a resident model that is no
    /// longer valid is never handed out, not even to a request that finds its
    /// key already in the slot. It is dropped and loaded again, under the
    /// same lock that compared the key.
    ///
    /// mummu-serve's poisoned-GPU case in miniature: the resident value
    /// carries the fault count it was loaded under, and a fault since then
    /// makes it stale.
    #[tokio::test]
    async fn a_resident_value_that_is_no_longer_valid_is_reloaded_not_served() {
        use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
        let slot: ModelSlot<(String, u32)> = ModelSlot::new();
        let key = Path::new("model-a");
        let faults = AtomicU32::new(0);
        let loads = AtomicU32::new(0);
        let load = |_: &Path| {
            loads.fetch_add(1, SeqCst);
            Ok::<_, Infallible>(("model-a".to_owned(), faults.load(SeqCst)))
        };
        let valid = |m: &(String, u32)| m.1 == faults.load(SeqCst);

        drop(slot.acquire_valid(key, valid, load).await.unwrap());
        drop(slot.acquire_valid(key, valid, load).await.unwrap());
        assert_eq!(loads.load(SeqCst), 1, "a valid resident model is a hit");

        faults.fetch_add(1, SeqCst); // the device failed under the resident model
        let m = slot.acquire_valid(key, valid, load).await.unwrap();
        assert_eq!(
            loads.load(SeqCst),
            2,
            "a model loaded before the failure was served instead of reloaded"
        );
        assert_eq!(m.1, 1, "and what is handed out is the fresh load");
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
