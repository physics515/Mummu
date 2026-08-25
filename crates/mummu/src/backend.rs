//! Backend selection: one binary compiles BOTH backends and picks at runtime.
//!
//! The model code is generic over `B: Backend`; nothing here forces a device
//! choice on a consumer. What this module owns is the *default* policy proven
//! in laurelane: enumerate GPU adapters once with a cheap `wgpu` probe (no
//! device creation), run on **wgpu (GPU via Vulkan/DX12/Metal — no CUDA
//! toolchain)** when a hardware adapter is present, else **burn-flex (CPU)**.
//! No feature-split builds.
//!
//! The probe also records which adapters advertise `SHADER_F16` — the input
//! the hardware planner (P6) uses to decide whether the f16 element type is
//! viable on this machine.

use once_cell::sync::OnceCell;

/// The default GPU device (wgpu: Vulkan / DX12 / Metal). burn 0.22 selects
/// backends at runtime through [`burn::tensor::Device`]; with the workspace
/// `fusion` feature, fusion applies to supporting devices automatically.
#[must_use]
/// Raise every cubecl device-server thread ("DSD-*") above the compute
/// pools. Those threads encode command buffers, submit to the driver, and
/// signal readback-map completions — microseconds of CPU each — but at
/// normal priority they starve behind the trunk's spinning gemm workers:
/// measured, a remote FFN group whose kernels total well under 1 ms still
/// held its caller ~26 ms at the fence, and the wait tracked scheduler
/// quanta, not GPU time. Call after model load (the servers spawn on first
/// device use); repeat calls are cheap and idempotent.
pub fn boost_device_server_threads() {
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32.dll", kind = "raw-dylib", modifiers = "+verbatim")]
        unsafe extern "system" {
            fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> isize;
            fn Thread32First(snap: isize, entry: *mut ThreadEntry32) -> i32;
            fn Thread32Next(snap: isize, entry: *mut ThreadEntry32) -> i32;
            fn OpenThread(access: u32, inherit: i32, tid: u32) -> isize;
            fn GetThreadDescription(handle: isize, desc: *mut *mut u16) -> i32;
            fn SetThreadPriority(handle: isize, priority: i32) -> i32;
            fn CloseHandle(handle: isize) -> i32;
            fn GetCurrentProcessId() -> u32;
            fn LocalFree(mem: isize) -> isize;
        }
        #[repr(C)]
        struct ThreadEntry32 {
            size: u32,
            usage: u32,
            thread_id: u32,
            owner_pid: u32,
            base_pri: i32,
            delta_pri: i32,
            flags: u32,
        }
        const TH32CS_SNAPTHREAD: u32 = 0x4;
        const THREAD_SET_INFORMATION: u32 = 0x20;
        const THREAD_QUERY_LIMITED_INFORMATION: u32 = 0x800;
        const ABOVE_NORMAL: i32 = 1;

        let pid = GetCurrentProcessId();
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == -1 || snap == 0 {
            return;
        }
        let mut entry = ThreadEntry32 {
            size: size_of::<ThreadEntry32>() as u32,
            usage: 0,
            thread_id: 0,
            owner_pid: 0,
            base_pri: 0,
            delta_pri: 0,
            flags: 0,
        };
        let mut boosted = 0u32;
        let mut ok = Thread32First(snap, &raw mut entry);
        while ok != 0 {
            if entry.owner_pid == pid {
                let h = OpenThread(
                    THREAD_SET_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION,
                    0,
                    entry.thread_id,
                );
                if h != 0 {
                    let mut desc: *mut u16 = std::ptr::null_mut();
                    if GetThreadDescription(h, &raw mut desc) >= 0 && !desc.is_null() {
                        let mut len = 0usize;
                        while *desc.add(len) != 0 {
                            len += 1;
                        }
                        let name = String::from_utf16_lossy(std::slice::from_raw_parts(desc, len));
                        if name.starts_with("DSD") {
                            SetThreadPriority(h, ABOVE_NORMAL);
                            boosted += 1;
                        }
                        LocalFree(desc as isize);
                    }
                    CloseHandle(h);
                }
            }
            ok = Thread32Next(snap, &raw mut entry);
        }
        CloseHandle(snap);
        if boosted > 0 {
            eprintln!("[mummu] raised {boosted} device-server thread(s) above the compute pools");
        }
    }
}

pub fn gpu_device() -> burn::tensor::Device {
    burn::tensor::Device::wgpu(Default::default())
}

/// Move a tensor to `device`, staging through host memory when both ends are
/// GPUs.
///
/// cubecl does not implement peer-to-peer transfer for wgpu: `comm_init` and
/// `send` on its server trait are `unimplemented!()`, so a direct
/// discrete-GPU -> integrated-GPU move panics with a bare "not implemented"
/// (cubecl-runtime `server/base.rs`). Staging through the host is the only
/// portable path, and it is what makes a placement spanning two GPUs work at
/// all.
///
/// A same-device move is a no-op, and a move with the host at either end is a
/// single transfer, so this costs nothing on the common paths.
#[must_use]
pub fn move_to<const D: usize>(
    tensor: burn::tensor::Tensor<D>,
    device: &burn::tensor::Device,
) -> burn::tensor::Tensor<D> {
    let from = tensor.device();
    if from == *device {
        return tensor;
    }
    if is_accelerator(&from) && is_accelerator(device) {
        return tensor.to_device(&cpu_device()).to_device(device);
    }
    tensor.to_device(device)
}

/// Is this an accelerator (as opposed to the host)? burn 0.22 selects
/// backends by runtime `Device` value and exposes no kind accessor, so the
/// debug form is the handle available.
fn is_accelerator(device: &burn::tensor::Device) -> bool {
    let name = format!("{device:?}");
    name.contains("Wgpu") || name.contains("Cuda")
}

/// The integrated GPU, when one exists.
///
/// Addressed explicitly because [`gpu_device`] resolves to wgpu's default,
/// which is the *discrete* card on any machine that has one — so the
/// integrated adapter is invisible to a placement that only ever asks for
/// "the GPU", however much idle capacity it has.
#[must_use]
pub fn integrated_gpu_device() -> burn::tensor::Device {
    burn::tensor::Device::wgpu(burn::tensor::DeviceKind::IntegratedGpu(0))
}

/// Does this machine expose an integrated GPU distinct from the discrete one?
#[must_use]
pub fn has_integrated_gpu() -> bool {
    inventory()
        .gpus
        .iter()
        .any(|g| g.device_type == wgpu::DeviceType::IntegratedGpu)
}

/// The CPU device (burn-flex: pure-Rust SIMD + gemm).
#[must_use]
pub fn cpu_device() -> burn::tensor::Device {
    burn::tensor::Device::flex()
}

/// The CUDA device (feature `cuda`). NVRTC compiles kernels at runtime — the
/// WSL2-container GPU path where no correct Vulkan reaches the process.
#[cfg(feature = "cuda")]
#[must_use]
pub fn cuda_device() -> burn::tensor::Device {
    burn::tensor::Device::cuda(0)
}

/// The float dtype mummu creates tensors with **on `device`**.
///
/// burn 0.22 moved the element type off the backend type and onto the device
/// as a runtime setting ([`burn::tensor::Device::configure`]), so the precision
/// a model runs in is a property of the device it was handed - not of a type
/// alias, and not of a process-wide constant. Reading it back here is what lets
/// one process hold an f16 GPU device beside an f32 host device, which the 0.21
/// `Gpu`/`GpuF16` alias split could not express at all.
///
/// Tensor-creation sites still name a dtype **explicitly** - the 0.21 rationale
/// is unchanged. What changed is where the answer comes from.
#[must_use]
pub fn float_dtype(device: &burn::tensor::Device) -> burn::tensor::DType {
    let dtype: burn::tensor::DType = device.settings().float_dtype.into();
    debug_assert!(
        dtype.is_float(),
        "a device's float setting must be a float dtype, got {dtype:?}"
    );
    dtype
}

/// Int dtype counterpart of [`float_dtype`], read from the same device.
#[must_use]
pub fn int_dtype(device: &burn::tensor::Device) -> burn::tensor::DType {
    let dtype: burn::tensor::DType = device.settings().int_dtype.into();
    debug_assert!(
        dtype.is_int(),
        "a device's int setting must be an int dtype, got {dtype:?}"
    );
    dtype
}

/// A GPU device configured to compute in **f16**.
///
/// The 0.22 replacement for the `GpuF16` type alias. Device settings lock on
/// first use and cannot be changed afterwards, so this must run before any
/// tensor exists on the discrete GPU - which is why every f16 gate lives in its
/// own test binary, exactly as it did under the one-alias-per-process rule.
///
/// # Errors
///
/// `AlreadyInitialized` when the device has already computed something, and an
/// unsupported-dtype error when the adapter cannot do f16 at all (check
/// [`DeviceInventory::any_shader_f16`] first).
pub fn gpu_device_f16() -> Result<burn::tensor::Device, burn::tensor::DeviceError> {
    let mut device = gpu_device();
    match device.configure((burn::tensor::FloatDType::F16, burn::tensor::IntDType::I32)) {
        Ok(()) => {}
        // Idempotent on purpose: several f16 gates share one process, and the
        // second caller must get the same device rather than an error - but
        // ONLY when the lock actually landed on f16. An f32-locked device is
        // reported, never handed back wearing an f16 label (the 2026-07-11
        // mislabelling bug is exactly this branch going the other way).
        Err(e) => {
            if float_dtype(&device) != burn::tensor::DType::F16 {
                return Err(e);
            }
        }
    }
    assert_eq!(
        float_dtype(&device),
        burn::tensor::DType::F16,
        "gpu_device_f16 must return an f16 device or an error, never an f32 device"
    );
    Ok(device)
}

/// One enumerated GPU adapter, as reported by wgpu.
#[derive(Debug, Clone)]
pub struct GpuAdapter {
    /// Driver-reported adapter name, e.g. `"NVIDIA GeForce RTX 4070 Ti SUPER"`.
    pub name: String,
    /// Graphics API carrying this adapter (`Vulkan`, `Dx12`, `Metal`, ...).
    pub backend: wgpu::Backend,
    /// Discrete / integrated / virtual — never [`wgpu::DeviceType::Cpu`]
    /// (software rasterizers are filtered out of the inventory).
    pub device_type: wgpu::DeviceType,
    /// Does this adapter advertise `SHADER_F16` (native f16 shader arithmetic)?
    pub shader_f16: bool,
    /// Largest single buffer this adapter permits — a hard bound the placement
    /// planner respects per tensor/shard. (True VRAM capacity is NOT exposed
    /// portably by wgpu; querying it per-API via wgpu-hal is a P6 follow-up.)
    pub max_buffer_bytes: u64,
    /// Dedicated video memory (true VRAM capacity) — the P6 planner's fit
    /// budget. wgpu exposes no portable query (gfx-rs/wgpu#2447), so this is
    /// filled per-OS: DXGI `DedicatedVideoMemory` on Windows (matched to the
    /// wgpu adapter by name, covering every API's view of the same card);
    /// `None` where no query is implemented yet (Linux/macOS follow-ups).
    /// Integrated GPUs legitimately report small values here — their real
    /// budget is shared system RAM ([`CpuInfo::total_ram_bytes`]).
    pub vram_bytes: Option<u64>,
}

/// The host CPU as a compute device (the `burn-flex` target and the P6
/// offload pool).
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Logical cores (SMT threads) available to this process; at least 1.
    pub logical_cores: usize,
    /// Total physical RAM; `None` where no query is implemented yet (macOS).
    pub total_ram_bytes: Option<u64>,
}

impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            logical_cores: 1,
            total_ram_bytes: None,
        }
    }
}

/// Every hardware GPU visible to wgpu plus the host CPU, enumerated once per
/// process — the device set the P6 hardware planner (and app settings UIs)
/// read.
#[derive(Debug, Clone, Default)]
pub struct DeviceInventory {
    /// Hardware adapters across the primary graphics APIs. The same physical
    /// card appears once per API that exposes it (e.g. Vulkan AND DX12) —
    /// deliberate, because features like `SHADER_F16` differ per API.
    pub gpus: Vec<GpuAdapter>,
    /// The host CPU (cores + RAM).
    pub cpu: CpuInfo,
}

impl DeviceInventory {
    /// Is at least one hardware GPU present (on any API)?
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// Does any adapter advertise `SHADER_F16`? Gates [`gpu_device_f16`].
    #[must_use]
    pub fn any_shader_f16(&self) -> bool {
        self.gpus.iter().any(|g| g.shader_f16)
    }
}

/// Enumerate hardware adapters on `backends`. Cheap: adapter listing only, no
/// device creation. wgpu 29's enumeration is async, so block on it here.
fn enumerate(instance: &wgpu::Instance, backends: wgpu::Backends) -> Vec<GpuAdapter> {
    let vram = vram_by_adapter_name();
    pollster::block_on(instance.enumerate_adapters(backends))
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            if matches!(info.device_type, wgpu::DeviceType::Cpu) {
                return None; // software rasterizer, not a hardware GPU
            }
            let shader_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
            let vram_bytes = lookup_vram(&vram, &info.name);
            Some(GpuAdapter {
                name: info.name,
                backend: info.backend,
                device_type: info.device_type,
                shader_f16,
                max_buffer_bytes: adapter.limits().max_buffer_size,
                vram_bytes,
            })
        })
        .collect()
}

/// Find `name`'s dedicated VRAM in the per-OS `(adapter name, bytes)` table.
/// Driver stacks decorate the same card's name slightly differently per API
/// (e.g. a `(TM)` suffix), so fall back to a case-insensitive prefix match in
/// either direction when the exact name misses.
fn lookup_vram(table: &[(String, u64)], name: &str) -> Option<u64> {
    debug_assert!(!name.is_empty(), "wgpu adapters always carry a name");
    if let Some(&(_, bytes)) = table.iter().find(|(n, _)| n == name) {
        return Some(bytes);
    }
    let lower = name.to_lowercase();
    table
        .iter()
        .find(|(n, _)| {
            let n = n.to_lowercase();
            n.starts_with(&lower) || lower.starts_with(&n)
        })
        .map(|&(_, bytes)| bytes)
}

/// Minimal hand-bound slice of the DXGI 1.1 COM ABI — the one Windows API
/// that reports true VRAM capacity for every GPU regardless of which graphics
/// API wgpu reached it through. Bound by hand because `windows-sys` 0.60+
/// dropped COM interface bindings and the full `windows` crate is a heavy
/// dependency for two vtable calls. The ABI is frozen (shipped with Windows 7,
/// 2009): vtables are declared slot-exact below and the dev-box unit test
/// (`windows_discrete_adapters_report_plausible_vram`) cross-checks the
/// numbers against reality.
#[cfg(windows)]
mod dxgi {
    use core::ffi::c_void;

    #[repr(C)]
    pub struct Guid {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }

    /// `IID_IDXGIFactory1` = `{770aae78-f26f-4dba-a829-253c83d1b387}`.
    pub const IID_IDXGI_FACTORY1: Guid = Guid {
        data1: 0x770a_ae78,
        data2: 0xf26f,
        data3: 0x4dba,
        data4: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
    };

    pub const DXGI_ERROR_NOT_FOUND: i32 = 0x887A_0002_u32 as i32;

    /// `IID_IDXGIAdapter3` = `{645967a4-1392-4310-a798-8053ce3e93fd}`. The
    /// interface that reports the OS's *current* video-memory budget for
    /// this process, which is what shrinks when another process takes VRAM.
    pub const IID_IDXGI_ADAPTER3: Guid = Guid {
        data1: 0x6459_67a4,
        data2: 0x1392,
        data3: 0x4310,
        data4: [0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd],
    };

    /// `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` — the adapter's own VRAM, as
    /// opposed to system memory it may spill into.
    pub const MEMORY_SEGMENT_LOCAL: u32 = 0;

    /// `DXGI_QUERY_VIDEO_MEMORY_INFO`, verbatim layout.
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct QueryVideoMemoryInfo {
        /// Bytes the OS is currently willing to let this process use.
        pub budget: u64,
        /// Bytes this process currently has resident.
        pub current_usage: u64,
        pub available_for_reservation: u64,
        pub current_reservation: u64,
    }

    /// `IDXGIAdapter3`'s vtable. Slot arithmetic, in order: IUnknown (3),
    /// IDXGIObject (4), IDXGIAdapter (3), IDXGIAdapter1 (1: GetDesc1),
    /// IDXGIAdapter2 (1: GetDesc2), then IDXGIAdapter3's own six — of which
    /// `QueryVideoMemoryInfo` is the third. Getting this count wrong calls
    /// the wrong function pointer, so it is spelled out rather than padded.
    #[repr(C)]
    pub struct Adapter3Vtbl {
        _query_interface: usize,
        _add_ref: usize,
        pub release: unsafe extern "system" fn(*mut c_void) -> u32,
        _idxgi_object: [usize; 4],
        _idxgi_adapter: [usize; 3],
        _get_desc1: usize,
        _get_desc2: usize,
        _register_teardown: usize,
        _unregister_teardown: usize,
        pub query_video_memory_info:
            unsafe extern "system" fn(*mut c_void, u32, u32, *mut QueryVideoMemoryInfo) -> i32,
        _set_video_memory_reservation: usize,
        _register_budget_change: usize,
        _unregister_budget_change: usize,
    }

    /// `DXGI_ADAPTER_DESC1`, verbatim layout.
    #[repr(C)]
    pub struct AdapterDesc1 {
        pub description: [u16; 128],
        pub vendor_id: u32,
        pub device_id: u32,
        pub sub_sys_id: u32,
        pub revision: u32,
        pub dedicated_video_memory: usize,
        pub dedicated_system_memory: usize,
        pub shared_system_memory: usize,
        pub adapter_luid: [u32; 2],
        pub flags: u32,
    }

    /// `IDXGIFactory1`'s vtable: IUnknown (3 slots) + IDXGIObject (4) +
    /// IDXGIFactory (5) + IDXGIFactory1 (2). Uncalled slots are opaque.
    #[repr(C)]
    pub struct Factory1Vtbl {
        _query_interface: usize,
        _add_ref: usize,
        pub release: unsafe extern "system" fn(*mut c_void) -> u32,
        _idxgi_object: [usize; 4],
        _idxgi_factory: [usize; 5],
        pub enum_adapters1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
        _is_current: usize,
    }

    /// `IDXGIAdapter1`'s vtable: IUnknown (3) + IDXGIObject (4) +
    /// IDXGIAdapter (3) + IDXGIAdapter1 (1: GetDesc1).
    #[repr(C)]
    pub struct Adapter1Vtbl {
        pub query_interface:
            unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
        _add_ref: usize,
        pub release: unsafe extern "system" fn(*mut c_void) -> u32,
        _idxgi_object: [usize; 4],
        _idxgi_adapter: [usize; 3],
        pub get_desc1: unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> i32,
    }

    // raw-dylib: rustc generates the import stubs itself, so linking needs
    // no Windows SDK .lib at all — the same mechanism windows-sys uses.
    #[link(name = "dxgi.dll", kind = "raw-dylib", modifiers = "+verbatim")]
    unsafe extern "system" {
        pub fn CreateDXGIFactory1(riid: *const Guid, factory: *mut *mut c_void) -> i32;
    }

    /// The vtable pointer every COM object starts with.
    #[inline]
    pub unsafe fn vtbl<T>(object: *mut c_void) -> *const T {
        debug_assert!(!object.is_null(), "COM object must be live");
        // SAFETY: caller guarantees `object` is a live COM interface pointer;
        // its first field is the vtable pointer.
        unsafe { *object.cast::<*const T>() }
    }
}

/// Every adapter's `(name, dedicated video memory)` as DXGI reports it: one
/// factory, a bounded adapter walk, everything released before returning.
/// Failures degrade to an empty table (VRAM stays `None`), never to a panic —
/// this runs once at inventory time.
#[cfg(windows)]
fn vram_by_adapter_name() -> Vec<(String, u64)> {
    use dxgi::{Adapter1Vtbl, Factory1Vtbl};

    /// More adapters than any real machine exposes; bounds the walk.
    const MAX_ADAPTERS: u32 = 64;

    let mut factory: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: CreateDXGIFactory1 writes a factory pointer on success; checked
    // via the HRESULT before use.
    let hr = unsafe { dxgi::CreateDXGIFactory1(&dxgi::IID_IDXGI_FACTORY1, &mut factory) };
    if hr < 0 || factory.is_null() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for index in 0..MAX_ADAPTERS {
        let mut adapter: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: factory is the live IDXGIFactory1 created above;
        // EnumAdapters1 writes an adapter or returns DXGI_ERROR_NOT_FOUND.
        let hr = unsafe {
            ((*dxgi::vtbl::<Factory1Vtbl>(factory)).enum_adapters1)(factory, index, &mut adapter)
        };
        if hr == dxgi::DXGI_ERROR_NOT_FOUND {
            break;
        }
        if hr < 0 || adapter.is_null() {
            continue;
        }
        // SAFETY: adapter is the live IDXGIAdapter1 just handed out; GetDesc1
        // fills the struct; Release balances EnumAdapters1's reference.
        let (hr, desc) = unsafe {
            let mut desc = core::mem::zeroed::<dxgi::AdapterDesc1>();
            let hr = ((*dxgi::vtbl::<Adapter1Vtbl>(adapter)).get_desc1)(adapter, &mut desc);
            ((*dxgi::vtbl::<Adapter1Vtbl>(adapter)).release)(adapter);
            (hr, desc)
        };
        if hr < 0 {
            continue;
        }
        let len = desc
            .description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.description.len());
        let name = String::from_utf16_lossy(&desc.description[..len]);
        let dedicated = desc.dedicated_video_memory as u64;
        // Negative space: skip software adapters (they report ~0 dedicated
        // VRAM; the "Microsoft Basic Render Driver") rather than risk mapping
        // a real card's lookup onto them.
        if !name.is_empty() && dedicated > 0 {
            out.push((name, dedicated));
        }
    }
    // SAFETY: Release on the factory created above.
    unsafe { ((*dxgi::vtbl::<Factory1Vtbl>(factory)).release)(factory) };
    debug_assert!(
        out.iter().all(|&(_, b)| b < 1u64 << 42),
        "implausible VRAM in the DXGI table: {out:?}"
    );
    out
}

#[cfg(not(windows))]
fn vram_by_adapter_name() -> Vec<(String, u64)> {
    Vec::new() // Linux (Vulkan memory heaps) and macOS are P6 follow-ups.
}

/// How much VRAM the OS is *currently* willing to give this process, and how
/// much of it we already hold.
///
/// This is the external-pressure signal: `budget` is not the card's size, it
/// is the driver's running allocation to this process, and it falls when
/// another process (a game, a second model, a browser compositing video)
/// takes memory. Placement uses it to decide how much of a model can stay
/// resident, and at what precision — see [`crate::mix`].
///
/// Distinct from the total in [`GpuAdapter::vram_bytes`], which never moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoMemory {
    /// Bytes the OS currently allows this process on the local segment.
    pub budget: u64,
    /// Bytes this process currently holds there.
    pub current_usage: u64,
}

impl VideoMemory {
    /// Headroom before the driver starts demoting our allocations. Saturating:
    /// usage legitimately exceeds budget when the OS has just cut it, and that
    /// means *zero* headroom, not a huge negative one.
    #[must_use]
    pub fn headroom(self) -> u64 {
        self.budget.saturating_sub(self.current_usage)
    }
}

/// Query the discrete GPU's current video-memory budget.
///
/// Two things to know about the numbers. **`current_usage` is this process
/// only** — it reads 0 from a program that has allocated nothing, even while
/// the card is full. Other processes do not appear there; they appear as a
/// *smaller `budget`*, which is exactly the pressure signal we want.
/// **Adapter choice is by dedicated VRAM, not by budget**: an integrated
/// GPU's "local" segment is system RAM, so on this box the iGPU reports a
/// ~101 GiB budget and would win any largest-budget contest.
///
/// `None` when the platform or driver will not say (non-Windows today, or a
/// pre-DXGI-1.4 adapter). Callers must treat `None` as "no information" and
/// hold their current placement rather than assuming either plenty or
/// pressure — guessing in either direction is worse than not adapting.
#[cfg(windows)]
#[must_use]
pub fn video_memory() -> Option<VideoMemory> {
    use dxgi::{Adapter1Vtbl, Adapter3Vtbl, Factory1Vtbl};

    const MAX_ADAPTERS: u32 = 64;
    let mut factory: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: writes a factory pointer on success, checked via the HRESULT.
    let hr = unsafe { dxgi::CreateDXGIFactory1(&dxgi::IID_IDXGI_FACTORY1, &mut factory) };
    if hr < 0 || factory.is_null() {
        return None;
    }

    // (dedicated VRAM, budget) — the largest dedicated wins, see the note above.
    let mut best: Option<(u64, VideoMemory)> = None;
    for index in 0..MAX_ADAPTERS {
        let mut adapter: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: factory is the live IDXGIFactory1 created above.
        let hr = unsafe {
            ((*dxgi::vtbl::<Factory1Vtbl>(factory)).enum_adapters1)(factory, index, &mut adapter)
        };
        if hr == dxgi::DXGI_ERROR_NOT_FOUND {
            break;
        }
        if hr < 0 || adapter.is_null() {
            continue;
        }
        // SAFETY: `adapter` is a live IDXGIAdapter1. QueryInterface either
        // hands back a live IDXGIAdapter3 or leaves the pointer null; both
        // references are released before the loop continues.
        let info = unsafe {
            // The adapter's fixed VRAM, used only to tell discrete from
            // integrated; the budget itself comes from IDXGIAdapter3.
            let mut desc = core::mem::zeroed::<dxgi::AdapterDesc1>();
            let dedicated =
                if ((*dxgi::vtbl::<Adapter1Vtbl>(adapter)).get_desc1)(adapter, &mut desc) >= 0 {
                    desc.dedicated_video_memory as u64
                } else {
                    0
                };
            let mut adapter3: *mut core::ffi::c_void = core::ptr::null_mut();
            let hr = ((*dxgi::vtbl::<Adapter1Vtbl>(adapter)).query_interface)(
                adapter,
                &dxgi::IID_IDXGI_ADAPTER3,
                &mut adapter3,
            );
            let info = if hr >= 0 && !adapter3.is_null() {
                let mut info = dxgi::QueryVideoMemoryInfo::default();
                let vt = dxgi::vtbl::<Adapter3Vtbl>(adapter3);
                let hr = ((*vt).query_video_memory_info)(
                    adapter3,
                    0, // node 0: single-GPU adapters have exactly one
                    dxgi::MEMORY_SEGMENT_LOCAL,
                    &mut info,
                );
                ((*vt).release)(adapter3);
                (hr >= 0).then_some(info)
            } else {
                None
            };
            ((*dxgi::vtbl::<Adapter1Vtbl>(adapter)).release)(adapter);
            info.map(|i| (dedicated, i))
        };

        if let Some((dedicated, info)) = info {
            let seen = VideoMemory {
                budget: info.budget,
                current_usage: info.current_usage,
            };
            if best.is_none_or(|(d, _)| dedicated > d) {
                best = Some((dedicated, seen));
            }
        }
    }
    // SAFETY: `factory` is live and owned here; this balances its creation.
    unsafe { ((*dxgi::vtbl::<Factory1Vtbl>(factory)).release)(factory) };
    best.map(|(_, v)| v)
}

/// No portable budget query off Windows yet — Vulkan's
/// `VK_EXT_memory_budget` is the equivalent and a P6 follow-up.
#[cfg(not(windows))]
#[must_use]
pub fn video_memory() -> Option<VideoMemory> {
    None
}

/// Total physical RAM, per platform. Kept syscall-thin — this runs once, at
/// inventory time.
#[cfg(windows)]
fn total_ram_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: core::mem::size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: `status` is a live, writable MEMORYSTATUSEX with dwLength set,
    // exactly what the API contract requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0).then_some(status.ullTotalPhys)
}

#[cfg(target_os = "linux")]
fn total_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: u64 = meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kib * 1024)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn total_ram_bytes() -> Option<u64> {
    None // macOS et al.: a sysctl query is a P6 follow-up
}

/// The host CPU: logical cores + total RAM.
fn cpu_info() -> CpuInfo {
    let logical_cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let total_ram_bytes = total_ram_bytes();
    assert!(
        logical_cores >= 1,
        "a running process has at least one core"
    );
    // Negative space: an answer below 64 MiB is a parse/API bug, not a machine.
    debug_assert!(
        total_ram_bytes.is_none_or(|b| b >= 64 << 20),
        "implausible total RAM: {total_ram_bytes:?}"
    );
    CpuInfo {
        logical_cores,
        total_ram_bytes,
    }
}

/// The process-lifetime device inventory. Enumerated once (first call pays
/// ~tens of milliseconds); every later call is a cache read.
pub fn inventory() -> &'static DeviceInventory {
    static INVENTORY: OnceCell<DeviceInventory> = OnceCell::new();
    INVENTORY.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let gpus = enumerate(&instance, wgpu::Backends::PRIMARY);
        let inv = DeviceInventory {
            gpus,
            cpu: cpu_info(),
        };
        // Positive space: every inventoried adapter is real hardware.
        debug_assert!(
            inv.gpus
                .iter()
                .all(|g| !matches!(g.device_type, wgpu::DeviceType::Cpu)),
            "CPU adapters must be filtered out of the GPU inventory"
        );
        inv
    })
}

/// Default device policy: run on the GPU when a hardware adapter is present.
/// Stable for the process lifetime (backed by [`inventory`]).
#[must_use]
pub fn use_gpu() -> bool {
    let gpu = inventory().has_gpu();
    // Negative space: the decision must agree with the inventory it came from.
    debug_assert!(gpu == !inventory().gpus.is_empty());
    gpu
}

/// Human label for where the default policy will run.
#[must_use]
pub fn device_label() -> &'static str {
    if use_gpu() {
        "GPU (wgpu)"
    } else {
        "CPU (flex)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_is_cached_and_consistent() {
        let first = inventory();
        let second = inventory();
        // Same allocation: the OnceCell caches, never re-enumerates.
        assert!(std::ptr::eq(first, second));
        assert_eq!(first.has_gpu(), !first.gpus.is_empty());
    }

    #[test]
    fn no_cpu_adapters_in_inventory() {
        assert!(
            inventory()
                .gpus
                .iter()
                .all(|g| !matches!(g.device_type, wgpu::DeviceType::Cpu))
        );
    }

    #[test]
    fn device_label_matches_policy() {
        let label = device_label();
        if use_gpu() {
            assert_eq!(label, "GPU (wgpu)");
        } else {
            assert_eq!(label, "CPU (flex)");
        }
    }

    #[test]
    fn f16_gate_requires_a_gpu() {
        let inv = inventory();
        // Negative space: SHADER_F16 can't be advertised with no GPUs at all.
        if inv.any_shader_f16() {
            assert!(inv.has_gpu());
        }
    }

    #[test]
    fn cpu_inventory_reports_cores_and_plausible_ram() {
        let cpu = &inventory().cpu;
        assert!(cpu.logical_cores >= 1);
        // Windows and Linux have a RAM query; its answer must be a real
        // machine's (1 GiB ..= 64 TiB), not a unit slip.
        if cfg!(any(windows, target_os = "linux")) {
            let ram = cpu.total_ram_bytes.expect("RAM query exists here");
            assert!((1 << 30..=1u64 << 46).contains(&ram), "implausible: {ram}");
        }
    }

    #[test]
    fn windows_discrete_adapters_report_plausible_vram() {
        // The DXGI walk exists on Windows: every *discrete* adapter must get
        // a dedicated-VRAM figure inside a real card's range (256 MiB..4 TiB).
        // Integrated/virtual adapters may report None (name mismatch) or a
        // small carve-out — both fine, the planner budgets those via RAM.
        for gpu in &inventory().gpus {
            if cfg!(windows) && matches!(gpu.device_type, wgpu::DeviceType::DiscreteGpu) {
                let vram = gpu
                    .vram_bytes
                    .unwrap_or_else(|| panic!("{}: no DXGI VRAM match", gpu.name));
                assert!(
                    (256 << 20..1u64 << 42).contains(&vram),
                    "{}: implausible VRAM {vram}",
                    gpu.name
                );
            }
        }
    }

    #[test]
    fn vram_lookup_matches_exact_then_prefix() {
        let table = vec![
            ("NVIDIA GeForce RTX 4070 Ti SUPER".to_string(), 16 << 30),
            ("AMD Radeon(TM) Graphics".to_string(), 512 << 20),
        ];
        // Exact hit.
        assert_eq!(
            lookup_vram(&table, "NVIDIA GeForce RTX 4070 Ti SUPER"),
            Some(16 << 30)
        );
        // Per-API name decoration: prefix in either direction, any case.
        assert_eq!(
            lookup_vram(&table, "AMD Radeon(TM) Graphics (RADV)"),
            Some(512 << 20)
        );
        assert_eq!(lookup_vram(&table, "amd radeon(tm)"), Some(512 << 20));
        // Negative space: a different card must never borrow a table entry.
        assert_eq!(lookup_vram(&table, "Intel(R) Arc(TM) A770"), None);
    }

    #[test]
    fn adapters_report_a_usable_buffer_bound() {
        // Every real adapter permits at least the WebGPU floor (256 MiB);
        // a smaller answer means the limits plumbing broke.
        for gpu in &inventory().gpus {
            assert!(
                gpu.max_buffer_bytes >= 256 << 20,
                "{}: max_buffer_bytes {} below the WebGPU floor",
                gpu.name,
                gpu.max_buffer_bytes
            );
        }
    }

    /// Print the inventory so the nightly log records what this machine has.
    #[test]
    fn report_inventory() {
        for gpu in &inventory().gpus {
            eprintln!(
                "[mummu] {:?} / {} ({:?}): SHADER_F16 = {}, max buffer {:.1} GiB, VRAM {}",
                gpu.backend,
                gpu.name,
                gpu.device_type,
                gpu.shader_f16,
                gpu.max_buffer_bytes as f64 / f64::from(1u32 << 30),
                gpu.vram_bytes.map_or("unknown".into(), |b| format!(
                    "{:.1} GiB",
                    b as f64 / f64::from(1u32 << 30)
                )),
            );
        }
        let cpu = &inventory().cpu;
        eprintln!(
            "[mummu] CPU: {} logical cores, RAM {:?} GiB",
            cpu.logical_cores,
            cpu.total_ram_bytes.map(|b| b >> 30),
        );
        eprintln!("[mummu] policy: {}", device_label());
    }
}
