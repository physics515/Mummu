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
    pollster::block_on(instance.enumerate_adapters(backends))
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            if matches!(info.device_type, wgpu::DeviceType::Cpu) {
                return None; // software rasterizer, not a hardware GPU
            }
            let shader_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
            Some(GpuAdapter {
                name: info.name,
                backend: info.backend,
                device_type: info.device_type,
                shader_f16,
                max_buffer_bytes: adapter.limits().max_buffer_size,
            })
        })
        .collect()
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
                "[mummu] {:?} / {} ({:?}): SHADER_F16 = {}, max buffer {:.1} GiB",
                gpu.backend,
                gpu.name,
                gpu.device_type,
                gpu.shader_f16,
                gpu.max_buffer_bytes as f64 / f64::from(1u32 << 30),
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
