//! How much video memory is *actually* free right now, across every process.
//!
//! Placement needs a number that moves when someone else takes VRAM. Two
//! sources, and the difference between them matters:
//!
//! * **DXGI's budget** ([`crate::backend::video_memory`]) is what the OS says
//!   *this process* may use. Windows permits oversubscription and pages VRAM
//!   behind your back, so it happily reports ~15 GiB of a 16 GiB card while
//!   another process holds 9 GiB of it. Measured on this box, 2026-08-23.
//!   Useful as a ceiling, useless as "what is free".
//! * **NVML** reports the card's global `total`/`used`/`free` — the same
//!   numbers `nvidia-smi` prints, because that is what nvidia-smi calls. This
//!   is the honest answer, and it is what a rebalance needs.
//!
//! NVML is loaded at runtime rather than linked, because it ships with the
//! NVIDIA driver and a machine without one must still run: a missing
//! `nvml.dll` / `libnvidia-ml.so.1` degrades to `None`, never to a failed
//! process start. That is not hypothetical — the same binary has to run on a
//! Mac, on an AMD box, and in a container the NVIDIA runtime did not inject a
//! driver into, and in every one of those cases the honest answer is "nothing
//! here will say".

/// A snapshot of one adapter's global memory use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memory {
    pub total: u64,
    /// Held by every process on the machine, this one included.
    pub used: u64,
    pub free: u64,
}

impl Memory {
    /// What a model may take without pushing the card into paging, leaving
    /// `reserve` for the desktop and for allocations that are not weights
    /// (activations, KV state, kernel workspaces).
    ///
    /// Saturating on purpose: when the card is already fuller than the
    /// reserve, the answer is zero, not a wrapped enormous number.
    #[must_use]
    pub fn headroom(self, reserve: u64) -> u64 {
        self.free.saturating_sub(reserve)
    }
}

/// Global VRAM use for the primary GPU, or `None` when nothing on this
/// machine will say.
///
/// Callers must treat `None` as "no information" and hold their current
/// placement — assuming plenty risks an OOM mid-generation, and assuming
/// pressure needlessly demotes a model that was running fine.
#[must_use]
pub fn memory() -> Option<Memory> {
    nvml::memory()
}

/// NVML, loaded by hand so its absence is a `None` and not a link error.
#[cfg(windows)]
mod nvml {
    use super::Memory;
    use core::ffi::{c_char, c_void};
    use std::sync::OnceLock;

    /// `nvmlMemory_t`, verbatim layout.
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        used: u64,
    }

    type Init = unsafe extern "C" fn() -> i32;
    type HandleByIndex = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
    type GetMemoryInfo = unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> i32;

    #[link(name = "kernel32", kind = "raw-dylib")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    /// The three entry points we need, resolved once.
    struct Api {
        handle_by_index: HandleByIndex,
        get_memory_info: GetMemoryInfo,
    }

    // SAFETY: the fields are function pointers into a DLL that is never
    // unloaded (no FreeLibrary anywhere), so they stay valid for the process
    // lifetime and are safe to call from any thread — NVML is thread-safe.
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    /// Resolve NVML once, but only CACHE A SUCCESS.
    ///
    /// Caching a failure was a real bug: one transient miss made this return
    /// `None` for the life of the process, `backend_budget` then fell back to
    /// its configured ceiling unreduced, and the planner put 14.53 GiB of
    /// weights on a 16 GiB card — every generation dying with `wgpu error:
    /// Out of Memory` while a standalone probe read the card fine. A reading
    /// this load-bearing must be allowed to recover.
    fn api() -> Option<&'static Api> {
        static API: OnceLock<Api> = OnceLock::new();
        if let Some(api) = API.get() {
            return Some(api);
        }
        let resolved = resolve()?;
        // A race just means two threads resolved it; both are equivalent.
        let _ = API.set(resolved);
        API.get()
    }

    /// One attempt at loading NVML and finding its entry points.
    fn resolve() -> Option<Api> {
        (|| {
            // SAFETY: literal, NUL-terminated names; every returned pointer
            // is null-checked before it is transmuted to a function pointer.
            unsafe {
                let module = LoadLibraryA(c"nvml.dll".as_ptr());
                if module.is_null() {
                    return None;
                }
                let symbol = |name: &core::ffi::CStr| {
                    let p = GetProcAddress(module, name.as_ptr());
                    (!p.is_null()).then_some(p)
                };
                // `_v2` where NVML versioned the ABI; the unsuffixed names
                // are the older, incompatible signatures.
                let init: Init = core::mem::transmute(symbol(c"nvmlInit_v2")?);
                let handle_by_index: HandleByIndex =
                    core::mem::transmute(symbol(c"nvmlDeviceGetHandleByIndex_v2")?);
                let get_memory_info: GetMemoryInfo =
                    core::mem::transmute(symbol(c"nvmlDeviceGetMemoryInfo")?);
                // NVML_SUCCESS is 0. Init is idempotent and refcounted; we
                // never shut down, matching the never-unloaded module above.
                if init() != 0 {
                    return None;
                }
                Some(Api {
                    handle_by_index,
                    get_memory_info,
                })
            }
        })()
    }

    pub fn memory() -> Option<Memory> {
        let api = api()?;
        // SAFETY: `api` resolved successfully, so NVML is initialised. Both
        // calls write through out-pointers to stack locals and are checked
        // against NVML_SUCCESS before the values are read.
        unsafe {
            let mut device: *mut c_void = core::ptr::null_mut();
            // Device 0: the primary GPU. Multi-GPU placement picks its own
            // devices and is a separate concern from this global reading.
            if (api.handle_by_index)(0, &mut device) != 0 || device.is_null() {
                return None;
            }
            let mut mem = NvmlMemory::default();
            if (api.get_memory_info)(device, &mut mem) != 0 {
                return None;
            }
            Some(Memory {
                total: mem.total,
                used: mem.used,
                free: mem.free,
            })
        }
    }
}

/// The same thing on unix, through `dlopen`/`dlsym` instead of
/// `LoadLibraryA`/`GetProcAddress`.
///
/// Deliberately `cfg(unix)` rather than `cfg(linux)`: NVML only ships on
/// Linux, but `dlopen` exists everywhere unix does, and on a Mac or a BSD the
/// library is simply not found — which is the quiet `None` this must produce
/// anyway. One code path, fewer cfgs, and the same answer.
#[cfg(all(unix, not(windows)))]
mod nvml {
    use super::Memory;
    use core::ffi::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    /// `nvmlMemory_t`, verbatim layout — the same struct the Windows module
    /// declares, because it is the same ABI.
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        used: u64,
    }

    type Init = unsafe extern "C" fn() -> i32;
    type HandleByIndex = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
    type GetMemoryInfo = unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> i32;

    // `dlopen`/`dlsym` declared by hand.
    //
    // mummu has no `libc` dependency and this is not worth adding one for: two
    // functions with a stable, decades-old ABI. They live in libc itself on
    // every glibc since 2.34 and in libdl before that, both of which are
    // already linked into any Rust std binary — so this resolves at link time
    // without an explicit `-ldl`.
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    /// `RTLD_LAZY | RTLD_LOCAL`. Lazy because we call three symbols out of a
    /// large library; local because nothing else in the process should start
    /// resolving against NVML's symbols by accident.
    const RTLD_LAZY: c_int = 1;

    /// The two entry points we call per reading, resolved once.
    struct Api {
        handle_by_index: HandleByIndex,
        get_memory_info: GetMemoryInfo,
    }

    // SAFETY: the fields are function pointers into a library that is never
    // unloaded (no dlclose anywhere), so they stay valid for the process
    // lifetime and are safe to call from any thread — NVML is thread-safe.
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    /// Resolve NVML once, but only CACHE A SUCCESS.
    ///
    /// The rule, and the reason for it, are the Windows module's verbatim:
    /// caching a failure made this return `None` for the life of the process
    /// after one transient miss, `backend_budget` fell back to its configured
    /// ceiling unreduced, and the planner put 14.53 GiB of weights on a 16 GiB
    /// card. A reading this load-bearing must be allowed to recover, so a
    /// failed resolve is simply retried on the next call.
    fn api() -> Option<&'static Api> {
        static API: OnceLock<Api> = OnceLock::new();
        if let Some(api) = API.get() {
            return Some(api);
        }
        let resolved = resolve()?;
        // A race just means two threads resolved it; both are equivalent.
        let _ = API.set(resolved);
        API.get()
    }

    /// One attempt at loading NVML and finding its entry points.
    ///
    /// The versioned SONAME `libnvidia-ml.so.1` rather than the bare
    /// `libnvidia-ml.so`: the latter is a symlink from the *driver
    /// development* package and is absent on a plain runtime install, while
    /// `.so.1` is what the driver itself ships and what the NVIDIA container
    /// runtime injects into a container.
    fn resolve() -> Option<Api> {
        // SAFETY: literal, NUL-terminated names; every returned pointer is
        // null-checked before it is transmuted to a function pointer. A
        // missing library returns null from `dlopen` and we stop there — the
        // whole point of loading by hand rather than linking.
        unsafe {
            let module = dlopen(c"libnvidia-ml.so.1".as_ptr(), RTLD_LAZY);
            if module.is_null() {
                return None;
            }
            let symbol = |name: &core::ffi::CStr| {
                let p = dlsym(module, name.as_ptr());
                (!p.is_null()).then_some(p)
            };
            // `_v2` where NVML versioned the ABI; the unsuffixed names are the
            // older, incompatible signatures.
            let init: Init = core::mem::transmute(symbol(c"nvmlInit_v2")?);
            let handle_by_index: HandleByIndex =
                core::mem::transmute(symbol(c"nvmlDeviceGetHandleByIndex_v2")?);
            let get_memory_info: GetMemoryInfo =
                core::mem::transmute(symbol(c"nvmlDeviceGetMemoryInfo")?);
            // NVML_SUCCESS is 0. Init is idempotent and refcounted; we never
            // shut down, matching the never-unloaded module above. A container
            // that has the library but no device reaches here and fails, which
            // is a failure worth retrying — hence the no-caching rule.
            if init() != 0 {
                return None;
            }
            Some(Api {
                handle_by_index,
                get_memory_info,
            })
        }
    }

    pub fn memory() -> Option<Memory> {
        let api = api()?;
        // SAFETY: `api` resolved successfully, so NVML is initialised. Both
        // calls write through out-pointers to stack locals and are checked
        // against NVML_SUCCESS before the values are read.
        unsafe {
            let mut device: *mut c_void = core::ptr::null_mut();
            // Device 0: the primary GPU. Multi-GPU placement picks its own
            // devices and is a separate concern from this global reading.
            if (api.handle_by_index)(0, &mut device) != 0 || device.is_null() {
                return None;
            }
            let mut mem = NvmlMemory::default();
            if (api.get_memory_info)(device, &mut mem) != 0 {
                return None;
            }
            Some(Memory {
                total: mem.total,
                used: mem.used,
                free: mem.free,
            })
        }
    }
}

/// Neither Windows nor unix: no NVML, and no `dlopen` to look for one with.
#[cfg(not(any(windows, unix)))]
mod nvml {
    use super::Memory;

    pub fn memory() -> Option<Memory> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever NVML reports has to be internally consistent and match the
    /// card. This is the guard against a wrong struct layout or a mis-resolved
    /// symbol, both of which would return plausible-looking nonsense.
    #[test]
    fn reported_memory_is_self_consistent() {
        let Some(m) = memory() else {
            return; // no NVIDIA driver here; nothing to check
        };
        assert!(m.total > 0, "a card with no memory is a bad reading");
        assert!(
            m.used + m.free <= m.total + (64 << 20),
            "used {} + free {} overshoots total {}",
            m.used,
            m.free,
            m.total
        );
        assert!(m.free <= m.total);
        // Anything under 256 MiB or over 256 GiB is not a GPU we can believe.
        assert!(
            (256 << 20..=256u64 << 30).contains(&m.total),
            "implausible total {}",
            m.total
        );
    }

    /// Two readings a moment apart must agree on the CARD, even though `used`
    /// legitimately moves between them. A wandering `total` is the signature
    /// of a wrong struct layout or a symbol resolved to the wrong function —
    /// the failure modes that return plausible-looking nonsense rather than
    /// an error.
    #[test]
    fn the_cards_size_does_not_change_between_readings() {
        let (Some(a), Some(b)) = (memory(), memory()) else {
            return; // no NVIDIA driver here; nothing to check
        };
        assert_eq!(a.total, b.total, "the card did not change size");
    }

    /// The contract every caller relies on: absence is `None`, never a panic,
    /// never a link error, never a zeroed `Memory` that reads as an empty
    /// card. On a machine with no NVIDIA driver this is the ONLY thing that
    /// runs here, so it is the test that has to hold everywhere.
    #[test]
    fn a_machine_with_no_driver_answers_none_rather_than_failing() {
        match memory() {
            None => {}
            Some(m) => assert!(
                m.total > 0,
                "a Some must be a real reading; absence is spelled None"
            ),
        }
    }

    /// Headroom never wraps, however full the card is.
    #[test]
    fn headroom_saturates_when_the_card_is_full() {
        let m = Memory {
            total: 16 << 30,
            used: 16 << 30,
            free: 0,
        };
        assert_eq!(m.headroom(2 << 30), 0);
    }
}
