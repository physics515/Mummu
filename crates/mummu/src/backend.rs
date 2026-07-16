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

/// GPU backend (wgpu: Vulkan / DX12 / Metal). With the workspace `fusion`
/// feature this is transparently `Fusion<Wgpu>` — streams of tensor ops are
/// compiled into far fewer GPU kernels.
pub type Gpu = burn::backend::Wgpu;

/// GPU backend with the f16 element type — ~2x throughput and half the VRAM
/// where `SHADER_F16` is available (check [`DeviceInventory::any_shader_f16`]).
pub type GpuF16 = burn::backend::Wgpu<half::f16, i32>;

/// CPU backend (burn-flex: pure-Rust SIMD + gemm; burn-ndarray's successor).
pub type Cpu = burn_flex::Flex<f32, i32>;

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

    /// Does any adapter advertise `SHADER_F16`? Gates the [`GpuF16`] backend.
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
        _query_interface: usize,
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
