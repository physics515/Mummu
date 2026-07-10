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
}

/// Every hardware GPU visible to wgpu, enumerated once per process.
#[derive(Debug, Clone, Default)]
pub struct DeviceInventory {
    /// Hardware adapters across the primary graphics APIs. The same physical
    /// card appears once per API that exposes it (e.g. Vulkan AND DX12) —
    /// deliberate, because features like `SHADER_F16` differ per API.
    pub gpus: Vec<GpuAdapter>,
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
            })
        })
        .collect()
}

/// The process-lifetime device inventory. Enumerated once (first call pays
/// ~tens of milliseconds); every later call is a cache read.
pub fn inventory() -> &'static DeviceInventory {
    static INVENTORY: OnceCell<DeviceInventory> = OnceCell::new();
    INVENTORY.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let gpus = enumerate(&instance, wgpu::Backends::PRIMARY);
        let inv = DeviceInventory { gpus };
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

    /// Print the inventory so the nightly log records what this machine has.
    #[test]
    fn report_inventory() {
        for gpu in &inventory().gpus {
            eprintln!(
                "[mummu] {:?} / {} ({:?}): SHADER_F16 = {}",
                gpu.backend, gpu.name, gpu.device_type, gpu.shader_f16
            );
        }
        eprintln!("[mummu] policy: {}", device_label());
    }
}
