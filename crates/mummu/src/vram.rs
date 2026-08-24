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
//! NVIDIA driver and a machine without one must still run: a missing DLL
//! degrades to `None`, never to a failed process start.

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

    /// Resolve NVML once. `None` on any machine without the NVIDIA driver.
    fn api() -> Option<&'static Api> {
        static API: OnceLock<Option<Api>> = OnceLock::new();
        API.get_or_init(|| {
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
        })
        .as_ref()
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

#[cfg(not(windows))]
mod nvml {
    use super::Memory;

    /// NVML exists on Linux as `libnvidia-ml.so.1`; wiring it up is the same
    /// shape as the Windows path and a follow-up.
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
