//! The serving engine: one loaded model at a time (per backend), streamed
//! chat completions over it. All the heavy lifting is mummu's — this module
//! only picks a backend at runtime, renders the family's chat template, and
//! turns `CausalLm::generate`'s token callback into incremental text deltas.

use std::ops::ControlFlow;
use std::path::Path;
use std::time::Instant;

use burn::tensor::Device;
use mummu::cache::ModelSlot;
use mummu::chat::{ChatMl, Role, Turn};
use mummu::decode::SamplerOptions;
use mummu::gguf::GgufFile;
use mummu::models::CausalLm;
use mummu::models::{lfm2, olmoe, qwen2, qwen3, qwen35};
use mummu::registry::{Architecture, ModelSpec, WeightFormat};
use tokenizers::Tokenizer;

/// One chat-servable model: the architecture-erased LM plus its tokenizer.
pub struct Loaded {
    pub lm: AnyLm,
    pub tokenizer: Tokenizer,
}

/// Architecture-erased causal LM. `CausalLm` itself can't be a trait object
/// (associated `Cache` type, generic `on_token`), so erase by enum instead —
/// the zoo is closed and small.
pub enum AnyLm {
    Qwen2(qwen2::LoadedQwen2),
    Qwen3(qwen3::LoadedQwen3),
    Lfm2(lfm2::LoadedLfm2),
    Olmoe(olmoe::LoadedOlmoe),
    OlmoeQ(olmoe::LoadedOlmoeQ),
    Qwen35(qwen35::LoadedQwen35),
}

impl AnyLm {
    async fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        opts: &SamplerOptions,
        device: &Device,
        on_token: impl FnMut(u32) -> ControlFlow<()>,
    ) -> Result<Vec<u32>, String> {
        match self {
            Self::Qwen2(m) => m.generate(prompt_ids, max_tokens, opts, device, on_token).await,
            Self::Qwen3(m) => m.generate(prompt_ids, max_tokens, opts, device, on_token).await,
            Self::Lfm2(m) => m.generate(prompt_ids, max_tokens, opts, device, on_token).await,
            Self::Olmoe(m) => m.generate(prompt_ids, max_tokens, opts, device, on_token).await,
            Self::OlmoeQ(m) => m.generate(prompt_ids, max_tokens, opts, device, on_token).await,
            Self::Qwen35(m) => m.generate(prompt_ids, max_tokens, opts, device, on_token).await,
        }
    }
}

/// Which backend serves generations, decided once per process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// `burn::backend::Cuda` — only when built with the `cuda` feature AND
    /// explicitly requested (`MUMMU_BACKEND=cuda`).
    #[cfg(feature = "cuda")]
    Cuda,
    /// wgpu (Vulkan/DX12/Metal) — the parity-validated GPU stack.
    Wgpu,
    /// The integrated GPU, through wgpu. A separate choice from [`Self::Wgpu`]
    /// because it is a different device with different economics: it has no
    /// VRAM of its own (it allocates from system RAM) and it is far slower
    /// than the discrete card — but it is a genuinely *additional* worker,
    /// and it does not contend with the CPU for the memory controller.
    /// Measured on this box for the decode shape, `device-throughput.rs`:
    ///
    ///   dGPU 1.59 ms | iGPU 14.15 ms | CPU (f32) 13.82 ms
    ///
    /// so the iGPU is ~8.9x slower than the discrete card and about level
    /// with the CPU — its value is capacity beside the CPU, not speed over
    /// it. Run concurrently, the iGPU held 14.3 ms while the CPU ran flat
    /// out, i.e. no measurable contention.
    IntegratedGpu,
    /// burn-flex.
    Cpu,
}

/// Backend policy: `MUMMU_BACKEND` = `cuda` | `wgpu`/`gpu` | `cpu` picks
/// explicitly (a `cuda` request without the compiled feature falls back to
/// auto with a warning); unset = auto (mummu's wgpu adapter probe, else
/// CPU — unchanged host behavior). `MUMMU_FORCE_CPU=1` still wins.
pub fn backend_choice() -> BackendChoice {
    let forced_cpu = std::env::var("MUMMU_FORCE_CPU")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if forced_cpu {
        return BackendChoice::Cpu;
    }
    match std::env::var("MUMMU_BACKEND").as_deref() {
        #[cfg(feature = "cuda")]
        Ok(s) if s.eq_ignore_ascii_case("cuda") => return BackendChoice::Cuda,
        #[cfg(not(feature = "cuda"))]
        Ok(s) if s.eq_ignore_ascii_case("cuda") => {
            eprintln!(
                "[mummu-serve] MUMMU_BACKEND=cuda but this binary was built \
                 without the `cuda` feature — falling back to auto"
            );
        }
        Ok(s) if s.eq_ignore_ascii_case("wgpu") || s.eq_ignore_ascii_case("gpu") => {
            return BackendChoice::Wgpu;
        }
        Ok(s) if s.eq_ignore_ascii_case("cpu") => return BackendChoice::Cpu,
        Ok(other) if !other.is_empty() => {
            eprintln!("[mummu-serve] unknown MUMMU_BACKEND {other:?}, using auto");
        }
        _ => {}
    }
    if mummu::backend::use_gpu() {
        BackendChoice::Wgpu
    } else {
        BackendChoice::Cpu
    }
}

/// Human label for where generation runs, honoring the env overrides.
pub fn device_label() -> &'static str {
    match backend_choice() {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => "GPU (cuda)",
        BackendChoice::Wgpu => "GPU (wgpu)",
        BackendChoice::IntegratedGpu => "iGPU (wgpu)",
        BackendChoice::Cpu => "CPU (flex)",
    }
}

/// One process-wide model slot. Under burn 0.22 every backend is the same
/// Rust type (the device is a runtime value), so the per-backend slots of
/// the 0.21 design collapse into this one; its mutex still serializes
/// generations, which is what protects VRAM.
static SLOT: ModelSlot<Loaded> = ModelSlot::new();

/// What each backend slot currently holds, as the planner sees it: the
/// model dir and the resident bytes its plan estimated. Lets `plan_fit`
/// (a) route a request for an already-resident model straight to its
/// slot — no fit check, no reload — and (b) count the memory that
/// evicting a slot's current occupant would free.
static RESIDENT: std::sync::Mutex<Vec<(BackendChoice, std::path::PathBuf, u64)>> =
    std::sync::Mutex::new(Vec::new());

fn note_resident(backend: BackendChoice, dir: std::path::PathBuf, bytes: u64) {
    let mut r = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
    r.retain(|(b, _, _)| *b != backend);
    r.push((backend, dir, bytes));
}

fn resident_in(backend: BackendChoice) -> Option<(std::path::PathBuf, u64)> {
    let r = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
    r.iter()
        .find(|(b, _, _)| *b == backend)
        .map(|(_, d, n)| (d.clone(), *n))
}

/// Drop any resident model (frees VRAM/RAM).
pub fn unload_all() -> bool {
    clear_tiers();
    let freed = SLOT.clear();
    if freed {
        RESIDENT
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
    freed
}

/// The model dirs currently resident in any backend slot (the ollama shim's
/// `/api/ps` answer). At most one per backend.
pub fn resident_dirs() -> Vec<std::path::PathBuf> {
    SLOT.loaded_key().into_iter().collect()
}

fn load_any(
    spec: &ModelSpec,
    models_root: &Path,
    device: &Device,
    policy: mummu::quant::QuantPolicy,
    backend: BackendChoice,
) -> Result<Loaded, String> {
    let dir = spec.dir(models_root);
    match &spec.format {
        WeightFormat::Safetensors => {
            let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| format!("tokenizer.json: {e}"))?;
            let lm = match spec.architecture {
                Architecture::Qwen2 => AnyLm::Qwen2(
                    qwen2::load_from_dir(&dir, device).map_err(|e| e.to_string())?,
                ),
                Architecture::Qwen3 => AnyLm::Qwen3(
                    qwen3::load_from_dir(&dir, device).map_err(|e| e.to_string())?,
                ),
                Architecture::Lfm2 => AnyLm::Lfm2(
                    lfm2::load_from_dir(&dir, device).map_err(|e| e.to_string())?,
                ),
                Architecture::Olmoe => AnyLm::Olmoe(
                    olmoe::load_from_dir(&dir, device).map_err(|e| e.to_string())?,
                ),
                Architecture::MiniLm => {
                    return Err("all-MiniLM is an embedding model — not chat-servable".into());
                }
                Architecture::Qwen35 => {
                    return Err(
                        "qwen35 loads from GGUF only (no safetensors import yet)".into(),
                    );
                }
            };
            Ok(Loaded { lm, tokenizer })
        }
        WeightFormat::Gguf { file } => {
            let path = dir.join(file);
            let f = GgufFile::open(&path).map_err(|e| e.to_string())?;
            let tokenizer = mummu::tokenizer::tokenizer_from_gguf(&f)?;
            drop(f);
            let lm = match spec.architecture {
                Architecture::Qwen2 => AnyLm::Qwen2(
                    qwen2::load_from_gguf(&path, device).map_err(|e| e.to_string())?,
                ),
                Architecture::Qwen3 => AnyLm::Qwen3(
                    qwen3::load_from_gguf(&path, device).map_err(|e| e.to_string())?,
                ),
                Architecture::Lfm2 => AnyLm::Lfm2(
                    lfm2::load_from_gguf(&path, device).map_err(|e| e.to_string())?,
                ),
                Architecture::Olmoe => {
                    if let Some(pack_dir) = ensure_pack(&dir, &path, spec)? {
                        if let Some(cpu_only) = tiers_mode() {
                            // P9 stage 3b: trunk on this backend, experts
                            // tiered across every device present.
                            clear_tiers();
                            let pool = build_tiered_experts(&pack_dir, backend, cpu_only)?;
                            AnyLm::OlmoeQ(
                                olmoe::load_trunk_from_pack(&pack_dir, device)
                                    .map_err(|e| e.to_string())?
                                    .with_pool(pool),
                            )
                        } else {
                            // Pack path: experts pre-quantized per member, all
                            // on this backend at the planned level.
                            let level = precision_for(policy);
                            AnyLm::OlmoeQ(
                                olmoe::load_from_pack(&pack_dir, device, &|_| level)
                                    .map_err(|e| e.to_string())?,
                            )
                        }
                    } else if policy == mummu::quant::QuantPolicy::Off {
                        AnyLm::Olmoe(
                            olmoe::load_from_gguf(&path, device).map_err(|e| e.to_string())?,
                        )
                    } else {
                        // P9 MoE: per-expert quantized experts, routed compute.
                        AnyLm::OlmoeQ(
                            olmoe::load_from_gguf_quantized(&path, device, policy)
                                .map_err(|e| e.to_string())?,
                        )
                    }
                }
                Architecture::Qwen35 => {
                    if let Some(pack_dir) = ensure_pack(&dir, &path, spec)? {
                        let level = precision_for(policy);
                        let partitioned = mummu::pack::Pack::open(&pack_dir)
                            .ok()
                            .is_some_and(|p| p.manifest.ffn_partition.is_some());
                        match tiers_mode() {
                            // qwen35 is DENSE: every FFN cluster runs on every
                            // token, so splitting a layer across devices buys
                            // nothing and costs a crossing. Place whole layers
                            // instead (`MUMMU_TIERS=clusters` restores the
                            // cluster-granular path for experimenting).
                            Some(_) if partitioned && !cluster_granular() => {
                                clear_tiers();
                                AnyLm::Qwen35(build_layered_qwen35(&pack_dir, backend, policy)?)
                            }
                            Some(cpu_only) if partitioned => {
                                // P9 stage 3c: trunk + local FFN clusters here,
                                // the other clusters tiered across devices.
                                clear_tiers();
                                AnyLm::Qwen35(build_partitioned_qwen35(
                                    &pack_dir, device, backend, cpu_only, policy,
                                )?)
                            }
                            _ => {
                                // Per-tensor precision rather than one level
                                // for the whole model — see `mixed_precision`.
                                // The fit planner's policy is the *lowest*
                                // precision that fits wholesale; the mix
                                // starts from the best a pack stores and
                                // demotes only as far as the budget forces,
                                // so spare VRAM is spent rather than left.
                                let ceiling = match policy {
                                    mummu::quant::QuantPolicy::Off => {
                                        mummu::quant::QuantPolicy::Off
                                    }
                                    _ => mummu::quant::QuantPolicy::Q8,
                                };
                                let mix = mummu::pack::Pack::open(&pack_dir)
                                    .ok()
                                    .map(|p| mixed_precision(&p, backend, ceiling, &|_| true));
                                let choose = |e: &mummu::pack::TensorEntry| {
                                    mix.as_ref()
                                        .and_then(|m| m.get(&e.name).copied())
                                        .unwrap_or(level)
                                };
                                AnyLm::Qwen35(
                                    qwen35::load_from_pack(&pack_dir, device, &choose)
                                        .map_err(|e| e.to_string())?,
                                )
                            }
                        }
                    } else {
                        // Streaming importer with the fit-planned policy.
                        AnyLm::Qwen35(
                            qwen35::load_from_gguf_quantized(&path, device, policy)
                                .map_err(|e| e.to_string())?,
                        )
                    }
                }
                Architecture::MiniLm => {
                    return Err("all-MiniLM is an embedding model — not chat-servable".into());
                }
            };
            Ok(Loaded { lm, tokenizer })
        }
    }
}

/// Render `turns` into the family's prompt string. The ChatML families are
/// mummu's byte-verified renderers; OLMoE-Instruct speaks the Tulu template
/// (`<|user|>` / `<|assistant|>` behind an `<|endoftext|>` BOS), which has no
/// hardcoded renderer in the library yet, so it is spelled out here in the
/// shape `tests/real_olmoe.rs` decodes with.
fn render_prompt(arch: Architecture, turns: &[Turn]) -> Result<String, String> {
    match arch {
        Architecture::Qwen2 => Ok(ChatMl::qwen2().render(turns)),
        Architecture::Qwen3 => Ok(ChatMl::qwen3().render(turns)),
        // qwen35's imported chat template is ChatML with Qwen3's think
        // conventions (its vision macros never fire on text-only turns).
        Architecture::Qwen35 => Ok(ChatMl::qwen3().render(turns)),
        Architecture::Lfm2 => Ok(ChatMl::lfm2().render(turns)),
        Architecture::Olmoe => {
            let mut out = String::from("<|endoftext|>");
            for t in turns {
                let tag = match t.role {
                    Role::System => "<|system|>",
                    Role::User => "<|user|>",
                    Role::Assistant => "<|assistant|>",
                    Role::Tool => return Err("OLMoE template has no tool role".into()),
                };
                out.push_str(tag);
                out.push('\n');
                out.push_str(&t.content);
                out.push('\n');
            }
            out.push_str("<|assistant|>\n");
            Ok(out)
        }
        Architecture::MiniLm => Err("all-MiniLM is an embedding model — not chat-servable".into()),
    }
}

/// The finished-generation summary the API returns after the token stream.
pub struct ChatResult {
    pub text: String,
    pub tokens: usize,
    pub device: &'static str,
    pub elapsed_ms: u128,
}

/// Run one chat completion, streaming decoded-text deltas through `on_delta`
/// (return `Break` to cancel cooperatively). Loads the model into the
/// backend's slot on first use; generations are serialized by the slot mutex.
pub async fn run_chat(
    spec: &ModelSpec,
    models_root: &Path,
    turns: &[Turn],
    opts: &SamplerOptions,
    max_tokens: usize,
    on_delta: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<ChatResult, String> {
    let prompt = render_prompt(spec.architecture, turns)?;
    let plan = plan_fit(spec, models_root)?;
    eprintln!(
        "[mummu-serve] fit plan for {}: {:?} @ {:?}",
        spec.name, plan.backend, plan.policy
    );
    // One slot, one device value: the plan picks *where*, not *which type*.
    drive(
        &SLOT, spec, models_root, &prompt, opts, max_tokens, plan.policy,
        plan.backend, label_of(plan.backend), on_delta,
    )
    .await
}

/// The device a backend choice denotes (burn 0.22 selects at runtime).
pub(crate) fn device_of(backend: BackendChoice) -> Device {
    match backend {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => mummu::backend::cuda_device(),
        BackendChoice::Wgpu => mummu::backend::gpu_device(),
        BackendChoice::IntegratedGpu => mummu::backend::integrated_gpu_device(),
        BackendChoice::Cpu => mummu::backend::cpu_device(),
    }
}

/// Human label for a backend choice.
pub(crate) fn label_of(backend: BackendChoice) -> &'static str {
    match backend {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => "GPU (cuda)",
        BackendChoice::Wgpu => "GPU (wgpu)",
        BackendChoice::IntegratedGpu => "iGPU (wgpu)",
        BackendChoice::Cpu => "CPU (flex)",
    }
}

// ===========================================================================
// P9 stage 3(b): tiered MoE experts. On the pack path an MoE model's experts
// are not all loaded onto the main backend at one precision: they are placed
// across every device present (CPU and the wgpu/CUDA GPU) at the precision
// each runs best — by `mummu::tier::plan_tiers`, within each device's budget
// after the trunk — and re-tiered after every request from routing hits
// (hot-swap, bounded moves per pass). `MUMMU_TIERS=off` restores
// single-backend experts; `MUMMU_TIERS=cpu` keeps experts on the CPU only.
// ===========================================================================

/// The live tiered model: its pool (shared with the model in its slot), the
/// planner inputs, the smoothed hotness and the plan the pool currently holds.
struct TierRuntime {
    pack_dir: std::path::PathBuf,
    main: BackendChoice,
    pool: std::sync::Arc<mummu::nn::ExpertPool>,
    devices: Vec<(BackendChoice, mummu::tier::TierDevice)>,
    costs: Vec<mummu::tier::ExpertCost>,
    hotness: Vec<f64>,
    plan: mummu::tier::TierPlan,
    experts_per_layer: usize,
}

static TIERS: std::sync::Mutex<Option<TierRuntime>> = std::sync::Mutex::new(None);

/// The adaptive placement controller (see [`mummu::adapt`]). Present once a
/// tiered model is loaded; fed by every completed generation.
static PLACEMENT: std::sync::Mutex<Option<mummu::adapt::Controller>> =
    std::sync::Mutex::new(None);

/// Set when a device allocation failed during a generation — the controller's
/// one hard signal, and the reason it can act without waiting for the dwell.
static ALLOC_FAILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Report a completed generation to the placement controller, and re-tier if
/// it asks for a different device budget.
///
/// Called after every request, which is the natural observation window: it is
/// exactly one placement's worth of real work, measured the way the user
/// experiences it (tokens per second), rather than a synthetic probe.
fn observe_placement(tokens: usize, elapsed_ms: u128) {
    use mummu::adapt::{Adjust, Sample};
    if tokens == 0 || elapsed_ms == 0 {
        return; // nothing to learn from
    }
    let mut guard = PLACEMENT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(controller) = guard.as_mut() else {
        return;
    };
    let sample = Sample {
        tokens_per_sec: tokens as f64 / (elapsed_ms as f64 / 1000.0),
        device_alloc_failed: ALLOC_FAILED.swap(false, std::sync::atomic::Ordering::SeqCst),
        host_available_bytes: mem_available_bytes(),
        device_bytes_in_use: {
            let g = TIERS.lock().unwrap_or_else(|e| e.into_inner());
            g.as_ref().map_or(0, |rt| {
                let used = rt.pool.used_bytes(rt.devices.len());
                rt.devices
                    .iter()
                    .zip(&used)
                    .filter(|((b, _), _)| *b != BackendChoice::Cpu)
                    .map(|(_, &u)| u)
                    .sum()
            })
        },
    };
    let decision = controller.observe(&sample, Instant::now());
    let budget = controller.budget();
    drop(guard);

    match decision {
        Adjust::Hold => {}
        Adjust::Grow(_) | Adjust::Shrink(_) => {
            eprintln!(
                "[mummu-serve] placement: {:.2} tok/s -> device budget {} GiB ({decision:?})",
                sample.tokens_per_sec,
                budget >> 30
            );
            // Push the new budget into the tier runtime and let the existing
            // bounded hot-swap machinery move what it can. Never blocks the
            // request that produced the observation.
            {
                let mut g = TIERS.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(rt) = g.as_mut() {
                    for (backend, dev) in &mut rt.devices {
                        if *backend != BackendChoice::Cpu {
                            dev.budget_bytes = budget;
                        }
                    }
                }
            }
            rebalance_tiers();
        }
    }
}
static REBALANCING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `None` = tiers off; `Some(cpu_only)` otherwise (default on).
fn tiers_mode() -> Option<bool> {
    match std::env::var("MUMMU_TIERS").ok().as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("off" | "0" | "false") => None,
        Some("cpu") => Some(true),
        _ => Some(false),
    }
}

fn clear_tiers() {
    *TIERS.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Drop the tier runtime if its trunk lives in `backend`'s slot.
fn clear_tiers_if_slot(backend: BackendChoice) {
    let mut g = TIERS.lock().unwrap_or_else(|e| e.into_inner());
    if g.as_ref().is_some_and(|r| r.main == backend) {
        *g = None;
    }
}

/// Is `dir` the model whose experts are currently tiered?
fn tiers_pack_in(dir: &Path) -> Option<()> {
    TIERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .filter(|r| r.pack_dir.parent() == Some(dir))
        .map(|_| ())
}

/// Expert bytes the tiered model holds on `backend` (the fit planner
/// subtracts them from that backend's budget).
fn tier_used_bytes(backend: BackendChoice) -> u64 {
    let g = TIERS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(rt) = g.as_ref() else { return 0 };
    let used = rt.pool.used_bytes(rt.devices.len());
    rt.devices
        .iter()
        .zip(&used)
        .filter(|((b, _), _)| *b == backend)
        .map(|(_, &u)| u)
        .sum()
}

/// The devices experts may live on, with their precision ladders and the
/// budget left after the trunk (which lives on `main`). CPU runs int8/int4;
/// the GPU runs f32 first, then int8 (CUDA also int4; wgpu's int4 kernel is
/// broken in burn 0.21 and is never planned).
fn tier_devices(
    main: BackendChoice,
    trunk_bytes: u64,
    cpu_only: bool,
) -> Vec<(BackendChoice, mummu::tier::TierDevice)> {
    use mummu::tier::{DeviceClass, Precision, TierDevice};
    let budget = |b: BackendChoice| {
        let raw = backend_budget(b).saturating_sub(if b == main { trunk_bytes } else { 0 });
        // Accelerator budgets must leave room for what is not a weight —
        // activations, KV state, dequantize temporaries, and cubecl's own
        // pool chunking. Without this the tier planner happily filled a card
        // to its last byte and generation died with `out of device memory`
        // mid-token. The host is not reserved against: it has 96 GiB and its
        // allocator does not fail this way.
        match b {
            BackendChoice::Cpu => raw,
            _ => raw.saturating_sub(ACTIVATION_RESERVE),
        }
    };
    // `speed` ranks devices for this workload. Measured on the reference box
    // for the decode shape `[1,5120]x[5120,6144]`, WARM and averaged over 20
    // iterations (`examples/{qmatmul-probe,cpu-matmul-probe}.rs`):
    //
    //   CUDA native q_matmul   0.08 ms        CPU (flex)   4.72 ms
    //
    // The GPU is ~59x faster, so it ranks first and `plan_tiers` fills it
    // before spilling to the CPU.
    //
    // History worth keeping, because it cost a wrong decision: an earlier
    // reading of 96 ms put the GPU ~25x SLOWER and this function ranked the
    // CPU first on that basis — which sent 1984 of 2048 FFN clusters to the
    // CPU while 5.2 GiB of VRAM sat idle. That 96 ms was a COLD autotune
    // cache, i.e. autotune running inside the timing loop, not throughput.
    // Warm every probe before believing a number.
    //
    // `MUMMU_TIER_SPEED=cpu-first` restores the old ordering for a host whose
    // GPU genuinely is slower (no working native quantized matmul, so every
    // matmul pays a dequantize — see `nn::compute_weight`).
    // Three tiers now, ranked from the same decode-shape measurement
    // (`examples/device-throughput.rs`, 2026-08-23, warm, real pack weight):
    //
    //   dGPU 1.59 ms | iGPU 14.15 ms | CPU 13.82 ms (f32)
    //
    // The iGPU is level with the CPU, not above it — its value is that it is
    // an ADDITIONAL worker: run concurrently, it held 14.3 ms while the CPU
    // ran flat out, so there is no measurable contention for the memory
    // controller between them. It ranks just above the CPU so overflow
    // spreads onto it first and the CPU takes what is left, rather than one
    // device carrying everything while the other idles.
    // MEASURED throughput, not an ordinal rank. Scheduler A divides work in
    // proportion to these numbers, so an invented ordering becomes an
    // invented split: ranking the iGPU 2 against the CPU's 1 — when the two
    // are level — handed it 457 clusters against the CPU's 64 and made it the
    // makespan, at which point extra VRAM on the discrete card bought nothing
    // because the quota, not memory, was binding.
    //
    // From `examples/device-throughput.rs` (warm, real pack weight, decode
    // shape), inverted into work per second and scaled to keep integer
    // granularity:
    //
    //   dGPU  1.59 ms -> 629    iGPU 14.15 ms -> 71    CPU 13.82 ms -> 72
    //
    // Re-measure after any change to kernels, dtypes or hardware; these are
    // facts about this box, not constants of the universe.
    let decode_speed = |backend: BackendChoice| -> u32 {
        let cpu_first = std::env::var("MUMMU_TIER_SPEED")
            .ok()
            .as_deref()
            .is_some_and(|v| v.eq_ignore_ascii_case("cpu-first"));
        let measured = match backend {
            BackendChoice::Cpu => 72,
            BackendChoice::IntegratedGpu => 71,
            _ => 629,
        };
        // The override exists for a host whose GPU genuinely is the slow one;
        // invert the ordering without pretending to know its ratios.
        if cpu_first { 700 - measured } else { measured }
    };
    let mut out = vec![(
        BackendChoice::Cpu,
        TierDevice {
            name: "cpu".into(),
            class: DeviceClass::Cpu,
            // F32 FIRST, and it is not a memory-for-speed nicety: on
            // burn-flex a quantized matmul is 18x slower than a float one.
            // The same 89 M-param weight measured 268 ms at Q4 and 244 ms at
            // Q8 against 13.3 ms at F16 / 13.8 ms dequantized to f32 — 0.2
            // GB/s against 4.2 (`examples/device-throughput.rs`, 2026-08-23).
            // Quantization exists to fit a device short of memory; the host
            // has 96 GiB and is short of speed. The ladder previously started
            // at Q8, so a host that took a large share of the clusters was
            // pushed into its slowest mode precisely when it had the most
            // work to do.
            ladder: vec![Precision::F32, Precision::Q8, Precision::Q4],
            budget_bytes: budget(BackendChoice::Cpu),
            speed: decode_speed(BackendChoice::Cpu),
            preload_units: 0, // set by `charge_trunk_to_its_device`
        },
    )];
    if cpu_only {
        return out;
    }
    #[cfg(feature = "cuda")]
    if main == BackendChoice::Cuda || backend_choice() == BackendChoice::Cuda {
        out.push((
            BackendChoice::Cuda,
            TierDevice {
                name: "cuda".into(),
                class: DeviceClass::DiscreteGpu,
                // Q4/Q8 on the GPU are safe because pooled experts dequantize
                // their weight on-device before the matmul (see
                // `nn::compute_weight`) — burn 0.21's CUDA q_matmul panics on
                // the mixed f32-input × quantized-weight path, so we never
                // take it; storage stays compact, so the GPU holds ~4× more
                // clusters at Q4 than at F32, which is what makes a 27B fit.
                ladder: vec![Precision::F32, Precision::Q8, Precision::Q4],
                budget_bytes: budget(BackendChoice::Cuda),
                speed: decode_speed(BackendChoice::Cuda),
                preload_units: 0,
            },
        ));
        return out;
    }
    if main == BackendChoice::Wgpu || mummu::backend::use_gpu() {
        out.push((
            BackendChoice::Wgpu,
            TierDevice {
                name: "wgpu".into(),
                class: DeviceClass::DiscreteGpu,
                // Q4 is on this ladder as of 2026-08-23. It was withheld from
                // wgpu because the kernel was believed to return garbage; that
                // was a probe artifact (see `mixed_precision` / the pack
                // probe), and along the production path Q4 measures 0.062
                // relative error at 2.9 ms — the same speed as Q8 for half the
                // bytes. On a device whose scarce resource is memory, half the
                // bytes is twice the clusters, and every cluster runs on every
                // token.
                ladder: vec![Precision::F32, Precision::Q8, Precision::Q4],
                budget_bytes: budget(BackendChoice::Wgpu),
                speed: decode_speed(BackendChoice::Wgpu),
                preload_units: 0,
            },
        ));
        // The integrated GPU, if this machine has one that is not already the
        // main device. Left out until now, so on a box like this one it sat
        // idle while the CPU carried every spilled cluster.
        if mummu::backend::has_integrated_gpu() && main != BackendChoice::IntegratedGpu {
            out.push((
                BackendChoice::IntegratedGpu,
                TierDevice {
                    name: "igpu".into(),
                    class: DeviceClass::IntegratedGpu,
                    // Same ladder as the discrete card: this is a wgpu device,
                    // so it runs the same cubecl kernels and a quantized
                    // matmul costs it nothing (unlike burn-flex, where one is
                    // 18x slower — which is why the CPU tier differs).
                    ladder: vec![Precision::F32, Precision::Q8, Precision::Q4],
                    budget_bytes: budget(BackendChoice::IntegratedGpu),
                    speed: decode_speed(BackendChoice::IntegratedGpu),
                    preload_units: 0,
                },
            ));
        }
    }
    out
}

/// One OLMoE expert from the pack onto `backend`'s device at `tier`.
fn load_expert_on(
    backend: BackendChoice,
    pack: &mummu::pack::Pack,
    layer: usize,
    index: usize,
    tier: mummu::tier::Tier,
) -> Result<std::sync::Arc<dyn mummu::nn::ExpertExec>, String> {
    // burn 0.22: the backend IS the device, so the per-backend dispatch of
    // the 0.21 design is one call against the device this tier names.
    Ok(std::sync::Arc::new(olmoe::load_expert_from_pack(
        pack,
        layer,
        index,
        tier,
        &device_of(backend),
    )?))
}

/// All of a layer's FFN clusters for one device as **one** executor
/// (`load_ffn_clusters` concatenates their ranges), so the exact dense path
/// runs one matmul per (layer, device) instead of one per cluster.
fn load_ffn_group_on(
    backend: BackendChoice,
    pack: &mummu::pack::Pack,
    layer: usize,
    clusters: &[usize],
    tier: mummu::tier::Tier,
    bytes: u64,
) -> Result<std::sync::Arc<dyn mummu::nn::ExpertExec>, String> {
    // burn 0.22: one call against the device this tier names (the 0.21
    // per-backend dispatch collapsed with the backend type parameter).
    let device = device_of(backend);
    let weights = qwen35::load_ffn_clusters(pack, layer, clusters, tier.precision, &device)?;
    // P9 stage 4: under `MUMMU_WORKING_SET`, clusters are loaded into HOST
    // RAM and staged onto the device by the schedule, so the device holds
    // only the working set rather than a permanent assignment. Off by
    // default — the tier design stays the tested path until the working set
    // is measured on this hardware.
    if working_set_enabled() && backend != BackendChoice::Cpu {
        let host = mummu::backend::cpu_device();
        let host_weights = qwen35::load_ffn_clusters(pack, layer, clusters, tier.precision, &host)?;
        return Ok(std::sync::Arc::new(mummu::nn::StagedExpert::new(
            host_weights,
            host,
            tier,
            bytes,
        )));
    }
    Ok(std::sync::Arc::new(mummu::nn::DeviceExpert {
        weights,
        device,
        tier,
        bytes,
    }))
}

/// Use the cluster-granular tier path for a dense model?
///
/// Off by default: measured 24.7 s/tok against 4.8 for whole-layer placement
/// on the 27B, because a dense model touches every cluster every token, so
/// splitting a layer across devices adds a crossing and saves nothing.
/// `MUMMU_TIERS=clusters` opts back in for experiments.
fn cluster_granular() -> bool {
    std::env::var("MUMMU_TIERS").is_ok_and(|v| v.eq_ignore_ascii_case("clusters"))
}

/// P9 stage 4: is the working set on? `MUMMU_WORKING_SET=on` loads FFN
/// clusters into host RAM and lets the schedule stage them onto the device,
/// instead of assigning each one a permanent home.
///
/// Off by default, and deliberately so: for a DENSE model every cluster is
/// needed every layer, so streaming them all costs the whole model across
/// the bus per token — which the module header of `mummu::workingset`
/// shows cannot beat the CPU reading the same bytes from DDR. It earns its
/// keep on selective (routed MoE) workloads and where staging overlaps
/// compute. Until that is measured on a given host, the tier design stays
/// the default.
fn working_set_enabled() -> bool {
    std::env::var("MUMMU_WORKING_SET")
        .is_ok_and(|v| v.eq_ignore_ascii_case("on") || v == "1")
}

/// The skip threshold for a partitioned dense model: `MUMMU_FFN_SKIP_TAU`
/// (explicit, logged as unmeasured when the table lacks it) or the largest
/// measured tau whose max |Δ log-prob| stays within
/// `MUMMU_FFN_SKIP_TOLERANCE`. Unset → exact (0).
fn ffn_skip_tau(part: &mummu::pack::FfnPartition, name: &str) -> f32 {
    if let Some(t) = std::env::var("MUMMU_FFN_SKIP_TAU").ok().and_then(|v| v.parse::<f32>().ok()) {
        let measured = part.skip_table.iter().find(|p| (p.tau - t).abs() < 1e-9);
        match measured {
            Some(m) => eprintln!(
                "[mummu-serve] {name}: FFN skip tau={t} (measured: max |Δlogprob| {:.3}, argmax agreement {:.1}%, {:.0}% clusters kept)",
                m.max_delta_logprob, m.argmax_agreement * 100.0, m.kept_fraction * 100.0
            ),
            None => eprintln!("[mummu-serve] {name}: FFN skip tau={t} — UNMEASURED for this pack (run pack-calibrate)"),
        }
        return t.max(0.0);
    }
    let Some(tol) = std::env::var("MUMMU_FFN_SKIP_TOLERANCE").ok().and_then(|v| v.parse::<f32>().ok()) else {
        return 0.0;
    };
    if part.skip_table.is_empty() {
        eprintln!("[mummu-serve] {name}: MUMMU_FFN_SKIP_TOLERANCE set but the pack has no skip table (run pack-calibrate) — staying exact");
        return 0.0;
    }
    let pick = part
        .skip_table
        .iter()
        .filter(|p| p.max_delta_logprob <= tol)
        .max_by(|a, b| a.tau.partial_cmp(&b.tau).unwrap_or(std::cmp::Ordering::Equal));
    match pick {
        Some(m) => {
            eprintln!(
                "[mummu-serve] {name}: FFN skip tau={} within tolerance {tol} (measured max |Δlogprob| {:.3}, argmax agreement {:.1}%, {:.0}% clusters kept)",
                m.tau, m.max_delta_logprob, m.argmax_agreement * 100.0, m.kept_fraction * 100.0
            );
            m.tau
        }
        None => {
            eprintln!("[mummu-serve] {name}: no measured skip point within tolerance {tol} — staying exact");
            0.0
        }
    }
}

/// Trunk bytes of a pack on `level`: every non-expert, non-FFN-cluster
/// tensor (2-D linears at the level, the rest f32) × the planner's slack +
/// 1 GiB workspace. What must fit the main backend when the units tier out.
fn pack_trunk_bytes(pack: &mummu::pack::Pack, level: mummu::pack::Precision) -> u64 {
    use mummu::pack::{Precision, Role};
    let ffn: std::collections::HashSet<&str> = pack
        .manifest
        .ffn_partition
        .as_ref()
        .map(|p| p.names.iter().flat_map(|n| n.iter().map(String::as_str)).collect())
        .unwrap_or_default();
    let mut bytes = 0u64;
    for t in &pack.manifest.tensors {
        if matches!(t.role, Role::Expert { .. }) || ffn.contains(t.name.as_str()) {
            continue;
        }
        // The embedding does not ride with the trunk: it is a gather, never
        // quantized, and on the 27B it is 5.09 GB — a quarter of the model —
        // read once per token while a layer's weights are read 65 times.
        // `build_partitioned_qwen35` pins it to the host, so it must not be
        // charged against the device budget here either.
        if matches!(t.role, Role::Embedding) {
            continue;
        }
        let numel = t.shape.iter().product::<usize>() as u64;
        bytes += match (&t.role, t.precisions.get(&level)) {
            (Role::Linear, Some(b)) if matches!(level, Precision::Q4 | Precision::Q8) => b.values_len + b.scales_len,
            _ => numel * 4,
        };
    }
    bytes * 135 / 100 + (1 << 30)
}

/// Bytes one layer of a pack costs on a device at `level` — every tensor
/// whose parameter path names that layer, trunk and FFN alike.
fn pack_layer_bytes(
    pack: &mummu::pack::Pack,
    level: &dyn Fn(&mummu::pack::TensorEntry) -> mummu::pack::Precision,
) -> Vec<u64> {
    use mummu::pack::{Precision, Role};
    let mut per_layer: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();
    for t in &pack.manifest.tensors {
        // `blk.<n>.` is the GGUF naming the pack preserves.
        let Some(rest) = t.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((idx, _)) = rest.split_once('.') else {
            continue;
        };
        let Ok(layer) = idx.parse::<usize>() else {
            continue;
        };
        let numel = t.shape.iter().product::<usize>() as u64;
        let chosen = level(t);
        let bytes = match (&t.role, t.precisions.get(&chosen)) {
            (Role::Linear | Role::Expert { .. }, Some(b))
                if matches!(chosen, Precision::Q4 | Precision::Q8) =>
            {
                b.values_len + b.scales_len
            }
            _ => numel * 4,
        };
        *per_layer.entry(layer).or_insert(0) += bytes;
    }
    per_layer.into_values().collect()
}

/// Device memory held back from weights for everything that is not a weight:
/// activations, KV and recurrent state, dequantize temporaries, and the
/// allocator's own chunking. Sized from the failure it prevents — see
/// [`layers_that_fit`].
const ACTIVATION_RESERVE: u64 = 3 << 30;

/// How many whole layers fit `budget_bytes`, leaving room for activations.
fn layers_that_fit(layer_bytes: &[u64], budget_bytes: u64) -> usize {
    // Activations, KV/recurrent state and kernel workspaces are not weights.
    // 1 GiB was too little and produced `out of device memory` mid-generation
    // with 12.03 GiB of weights placed on a card with 15.2 GiB free: a
    // dequantize of one [5120, 17408] slab alone is 356 MB, several are live
    // at once, and the pooled allocator reserves in ~1 GiB chunks on top.
    let usable = budget_bytes.saturating_sub(ACTIVATION_RESERVE);
    let mut used = 0u64;
    let mut n = 0usize;
    for &b in layer_bytes {
        if used + b > usable {
            break;
        }
        used += b;
        n += 1;
    }
    n
}

/// Load a dense qwen35 pack with **whole layers** on the device — as many as
/// VRAM holds — and the rest on the host.
///
/// This is llama.cpp's `n_gpu_layers` shape, and it exists because the
/// cluster-granular path is the wrong granularity for a dense model. There,
/// every layer's FFN is split across devices, and since every cluster runs on
/// every token there is no selectivity to pay for the crossing: measured
/// 24.7 s/tok, against 4.8 for a placement that kept layers whole. Whole
/// layers cross ONCE, where the assignment changes.
///
/// Cluster granularity remains correct for a routed MoE, where only top-k of
/// E experts are touched and the k/E saving does pay for the crossing.
fn build_layered_qwen35(
    pack_dir: &Path,
    main: BackendChoice,
    policy: mummu::quant::QuantPolicy,
) -> Result<qwen35::LoadedQwen35, String> {
    use mummu::pack::Pack;
    let pack = Pack::open(pack_dir)?;
    let level = precision_for(policy);
    // Per-tensor precision, starting from the best a pack stores and demoting
    // only as far as this device's live budget forces — so the layers that do
    // land here carry as much precision as the card can hold, and more of them
    // fit than a single-precision plan would allow. See `mixed_precision`.
    let ceiling = match policy {
        mummu::quant::QuantPolicy::Off => mummu::quant::QuantPolicy::Off,
        _ => mummu::quant::QuantPolicy::Q8,
    };
    // The embedding goes to the host below (`embed_device`), so it must not
    // be charged against this device's budget.
    let mix = mixed_precision(&pack, main, ceiling, &|t| {
        !matches!(t.role, mummu::pack::Role::Embedding)
    });
    // Precision the DEVICE gets. Sizing the split needs only this, since the
    // split is decided by what fits on the device.
    let device_choose = |e: &mummu::pack::TensorEntry| mix.get(&e.name).copied().unwrap_or(level);
    let layer_bytes = pack_layer_bytes(&pack, &device_choose);
    if layer_bytes.is_empty() {
        return Err("pack has no per-layer tensors".into());
    }
    let device = device_of(main);
    let host = mummu::backend::cpu_device();

    let on_device = if main == BackendChoice::Cpu {
        layer_bytes.len() // everything is already on the host
    } else {
        layers_that_fit(&layer_bytes, backend_budget(main))
    };
    let total: u64 = layer_bytes.iter().sum();
    let placed: u64 = layer_bytes.iter().take(on_device).sum();
    eprintln!(
        "[mummu-serve] layers: {on_device}/{} on {} ({:.2} of {:.2} GiB); the rest on the host",
        layer_bytes.len(),
        label_of(main),
        placed as f64 / f64::from(1u32 << 30),
        total as f64 / f64::from(1u32 << 30),
    );

    // Host room for the layers that stay behind, plus slack.
    let host_bytes: u64 = layer_bytes.iter().skip(on_device).sum();
    ensure_host_room(host_bytes + (6u64 << 30), main);

    let dev_for = |l: usize| if l < on_device { device.clone() } else { host.clone() };
    // Embedding on the host (a gather; see `pack_trunk_bytes`), and the head
    // with the last layer so the final projection does not cross.
    let head = dev_for(layer_bytes.len().saturating_sub(1));
    let started = Instant::now();
    // A weight that lands on the HOST should not be quantized, because on
    // burn-flex a quantized matmul is pathologically slow: the same 89 M-param
    // weight measured 268 ms at Q4 and 244 ms at Q8 against **13.3 ms** read
    // from the pack at F16 — an 18x difference, and 0.2 GB/s against 4.2
    // (`examples/device-throughput.rs`, 2026-08-23). Quantization exists to
    // fit a device that is short of memory; the host has 96 GiB and is short
    // of speed, so paying 2 B/param there buys back an order of magnitude.
    //
    // Only Linear/Expert weights switch: the embedding is a gather and the
    // norm vectors anchor the numerics, so both stay as `mixed_precision`
    // left them.
    let host_layers = on_device..layer_bytes.len();
    let choose = |e: &mummu::pack::TensorEntry| {
        let on_host = match layer_index(&e.name) {
            Some(l) => host_layers.contains(&l),
            // Trunk tensors (the head, the final norm) follow the head device.
            None => on_device < layer_bytes.len(),
        };
        let quantizable = matches!(
            e.role,
            mummu::pack::Role::Linear | mummu::pack::Role::Expert { .. }
        );
        if on_host && quantizable && e.precisions.contains_key(&mummu::pack::Precision::F16) {
            mummu::pack::Precision::F16
        } else {
            device_choose(e)
        }
    };
    let model = qwen35::load_from_pack_layered(pack_dir, &dev_for, &host, &head, &choose)
        .map_err(|e| e.to_string())?;
    eprintln!(
        "[mummu-serve] layered model resident in {:.0}s",
        started.elapsed().as_secs_f32()
    );
    Ok(model)
}

/// Plan and load a partitioned qwen35 pack: trunk + local FFN clusters on
/// this backend at `policy`'s level, the remaining clusters tiered across
/// the other devices (CPU int8/int4, GPU f32/int8). Exact unless a skip
/// tau is configured. Records the runtime for re-tiering (remote clusters
/// only — the local slab is pinned).
fn build_partitioned_qwen35(
    pack_dir: &Path,
    device: &Device,
    main: BackendChoice,
    cpu_only: bool,
    policy: mummu::quant::QuantPolicy,
) -> Result<qwen35::LoadedQwen35, String> {
    use mummu::pack::Pack;
    let pack = Pack::open(pack_dir)?;
    let part = pack.manifest.ffn_partition.clone().ok_or("pack is not partitioned")?;
    let layers = part.layers.len();
    let epl = part.layers.first().map_or(0, Vec::len);
    if epl == 0 || part.layers.iter().any(|l| l.len() != epl) {
        return Err("partition rows must be uniform and non-empty".into());
    }
    let level = precision_for(policy);
    let mut costs = Vec::with_capacity(layers * epl);
    for l in 0..layers {
        costs.extend(mummu::partition::cluster_costs(&pack, l)?);
    }
    let trunk = pack_trunk_bytes(&pack, level);
    // Host room for the worst case on the CPU (every cluster at int8).
    let cpu_worst: u64 = costs
        .iter()
        .map(|c| c.bytes.get(&mummu::pack::Precision::Q8).copied().unwrap_or(0))
        .sum();
    ensure_host_room(cpu_worst + if main == BackendChoice::Cpu { trunk } else { 0 } + (6u64 << 30), main);
    let mut devices = tier_devices(main, trunk, cpu_only);
    cap_ladders_at_source(&mut devices, &pack);
    charge_trunk_to_its_device(&mut devices, main, trunk, &costs);
    let main_idx = devices
        .iter()
        .position(|(b, _)| *b == main)
        .ok_or("main backend missing from the tier devices")?;
    // The local slab is one tensor per projection: a single level there.
    devices[main_idx].1.ladder = vec![level];
    let planner_devices: Vec<mummu::tier::TierDevice> = devices.iter().map(|(_, d)| d.clone()).collect();
    let hotness: Vec<f64> = if part.hotness.len() == layers && part.hotness.iter().all(|h| h.len() == epl) {
        part.hotness.iter().flatten().map(|&h| f64::from(h)).collect()
    } else {
        let g = TIERS.lock().unwrap_or_else(|e| e.into_inner());
        g.as_ref()
            .filter(|r| r.pack_dir == pack_dir && r.hotness.len() == costs.len())
            .map(|r| r.hotness.clone())
            .unwrap_or_default()
    };
    let mut plan = mummu::tier::plan_tiers(&planner_devices, &costs, &hotness)?;
    // Every layer keeps at least one local cluster (its mlp must exist).
    let mut forced = 0usize;
    for l in 0..layers {
        if !(0..epl).any(|c| plan.tiers[l * epl + c].device == main_idx) {
            let c = (0..epl)
                .max_by(|&a, &b| {
                    let ha = hotness.get(l * epl + a).copied().unwrap_or(0.0);
                    let hb = hotness.get(l * epl + b).copied().unwrap_or(0.0);
                    ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            plan.tiers[l * epl + c] = mummu::tier::Tier { device: main_idx, precision: level };
            forced += 1;
        }
    }
    if forced > 0 {
        eprintln!("[mummu-serve] tiers: forced one local FFN cluster on {} layers (budget was full)", forced);
    }
    for ((d, p), n) in plan.histogram() {
        eprintln!(
            "[mummu-serve] tiers: {n} FFN clusters on {} @ {p:?} ({:.1} GiB budget there{})",
            devices[d].1.name,
            devices[d].1.budget_bytes as f64 / f64::from(1u32 << 30),
            if d == main_idx { ", local slab" } else { "" }
        );
    }
    let local: Vec<Vec<usize>> = (0..layers)
        .map(|l| (0..epl).filter(|&c| plan.tiers[l * epl + c].device == main_idx).collect())
        .collect();
    let started = Instant::now();
    // Trunk on the planned device, embedding on the host (see
    // `pack_trunk_bytes` — it is a gather, and 5 GB of VRAM spent on it is
    // 5 GB not spent on weights that run 65 times a token).
    let host = mummu::backend::cpu_device();
    let model = qwen35::load_from_pack_partitioned_split(
        pack_dir,
        device,
        &host,
        &|_| level,
        &|l| local[l].clone(),
    )
    .map_err(|e| e.to_string())?;
    // Remote clusters grouped by (device, PRECISION): one executor per group,
    // covering every cluster the plan put there at that rung.
    // `load_ffn_clusters` concatenates their ranges, so a group is one matmul
    // — not one per cluster. Measured on the 27B: 62 of 64 layers come out as
    // a single group, 2 as two, so 2048 clusters become 66 matmuls.
    //
    // Keying on the device ALONE was a latent bug. It was safe only while the
    // FFN plan gave each device one ladder level, and that stopped being true
    // when promotion gained a second phase that upgrades individual clusters
    // with leftover bytes: a real plan now reads `wgpu @ Q4: 1677` and
    // `wgpu @ Q8: 107`. The old grouping took the FIRST cluster's tier and
    // loaded the whole group at it, so 107 clusters were silently materialized
    // at a precision the planner never chose, while the byte total added up
    // costs for rungs that were never loaded.
    let mut rows: Vec<Vec<std::sync::Arc<dyn mummu::nn::ExpertExec>>> = Vec::with_capacity(layers);
    for l in 0..layers {
        let mut by_slot: std::collections::BTreeMap<
            (usize, mummu::pack::Precision),
            (mummu::tier::Tier, Vec<usize>, u64),
        > = std::collections::BTreeMap::new();
        for c in 0..epl {
            let tier = plan.tiers[l * epl + c];
            if tier.device == main_idx {
                continue; // the local slab is in the model's own mlp
            }
            let bytes = costs[l * epl + c].bytes.get(&tier.precision).copied().unwrap_or(0);
            let g = by_slot
                .entry((tier.device, tier.precision))
                .or_insert((tier, Vec::new(), 0));
            g.1.push(c);
            g.2 += bytes;
        }
        let mut row = Vec::new();
        for ((dev, _), (tier, clusters, bytes)) in by_slot {
            row.push(load_ffn_group_on(devices[dev].0, &pack, l, &clusters, tier, bytes)?);
        }
        rows.push(row);
    }
    let pool = std::sync::Arc::new(mummu::nn::ExpertPool::new(rows));
    let used = pool.used_bytes(devices.len());
    eprintln!(
        "[mummu-serve] partitioned FFN resident in {:.0}s — remote {}",
        started.elapsed().as_secs_f32(),
        devices
            .iter()
            .zip(&used)
            .filter(|((b, _), _)| *b != main)
            .map(|((_, d), u)| format!("{} {:.2} GiB", d.name, *u as f64 / f64::from(1u32 << 30)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Arm the placement controller for this model: start where the planner
    // put us, and let it adapt from measured throughput from here on.
    {
        let total = mummu::backend::inventory()
            .gpus
            .iter()
            .filter_map(|g| g.vram_bytes)
            .max()
            .unwrap_or(devices[main_idx].1.budget_bytes);
        // Leave the desktop (and anything else sharing the card) its share.
        let policy = mummu::adapt::Policy::for_device(total, 2 << 30);
        *PLACEMENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(
            mummu::adapt::Controller::new(policy, devices[main_idx].1.budget_bytes),
        );
    }
    let tau = ffn_skip_tau(&part, &pack.manifest.source_file);
    // A dense model's placement is static — every cluster runs each token in
    // exact mode, so there is nothing to re-tier at runtime (the calibrated
    // hotness already steered the plan). Leave no TierRuntime; the pool is
    // owned by the model.
    clear_tiers();
    Ok(model.with_ffn_pool(pool).with_ffn_skip(tau))
}

/// Plan and load every expert of the OLMoE pack at `pack_dir` across the
/// tier devices; records the runtime for re-tiering. The trunk is the
/// caller's (it goes on `main`).
fn build_tiered_experts(
    pack_dir: &Path,
    main: BackendChoice,
    cpu_only: bool,
) -> Result<std::sync::Arc<mummu::nn::ExpertPool>, String> {
    use mummu::pack::{Pack, Role};
    let pack = Pack::open(pack_dir)?;
    let costs = olmoe::pack_expert_costs(&pack)?;
    let header = pack.header()?;
    let cfg = olmoe::OlmoeConfig::from_gguf(&header)?;
    drop(header);
    let (layers, epl) = (cfg.num_hidden_layers, cfg.num_experts);
    if costs.len() != layers * epl {
        return Err(format!("pack has {} experts, config says {}×{epl}", costs.len(), layers));
    }
    // Trunk estimate on the main backend: every non-expert tensor at f32,
    // the fit planner's 1.35 slack, plus 1 GiB of workspace.
    let trunk: u64 = pack
        .manifest
        .tensors
        .iter()
        .filter(|t| !matches!(t.role, Role::Expert { .. }))
        .map(|t| t.shape.iter().product::<usize>() as u64 * 4)
        .sum::<u64>()
        * 135
        / 100
        + (1 << 30);
    // Host room for the worst-case CPU tier (every expert at int8) plus the
    // trunk if it lands on the CPU, plus staging slack — evicting another
    // model's CPU residency if that is what it takes.
    let cpu_worst: u64 = costs
        .iter()
        .map(|c| c.bytes.get(&mummu::pack::Precision::Q8).copied().unwrap_or(0))
        .sum();
    let host_need = cpu_worst
        + if main == BackendChoice::Cpu { trunk } else { 0 }
        + (6u64 << 30);
    ensure_host_room(host_need, main);
    let mut devices = tier_devices(main, trunk, cpu_only);
    cap_ladders_at_source(&mut devices, &pack);
    charge_trunk_to_its_device(&mut devices, main, trunk, &costs);
    let devices = devices;
    let hotness = {
        let g = TIERS.lock().unwrap_or_else(|e| e.into_inner());
        g.as_ref()
            .filter(|r| r.pack_dir == pack_dir && r.hotness.len() == costs.len())
            .map(|r| r.hotness.clone())
            .unwrap_or_default()
    };
    let planner_devices: Vec<mummu::tier::TierDevice> = devices.iter().map(|(_, d)| d.clone()).collect();
    let plan = mummu::tier::plan_tiers(&planner_devices, &costs, &hotness)?;
    for ((d, p), n) in plan.histogram() {
        eprintln!(
            "[mummu-serve] tiers: {n} experts on {} @ {p:?} ({:.1} GiB budget there)",
            devices[d].1.name,
            devices[d].1.budget_bytes as f64 / f64::from(1u32 << 30)
        );
    }
    let started = Instant::now();
    let mut slots = Vec::with_capacity(layers);
    for layer in 0..layers {
        let mut row = Vec::with_capacity(epl);
        for index in 0..epl {
            let tier = plan.tiers[layer * epl + index];
            row.push(load_expert_on(devices[tier.device].0, &pack, layer, index, tier)?);
        }
        slots.push(row);
    }
    let pool = std::sync::Arc::new(mummu::nn::ExpertPool::new(slots));
    let used = pool.used_bytes(devices.len());
    eprintln!(
        "[mummu-serve] {} experts resident in {:.0}s — {}",
        layers * epl,
        started.elapsed().as_secs_f32(),
        devices
            .iter()
            .zip(&used)
            .map(|((_, d), u)| format!("{} {:.2} GiB", d.name, *u as f64 / f64::from(1u32 << 30)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    *TIERS.lock().unwrap_or_else(|e| e.into_inner()) = Some(TierRuntime {
        pack_dir: pack_dir.to_path_buf(),
        main,
        pool: pool.clone(),
        devices,
        costs,
        hotness,
        plan,
        experts_per_layer: epl,
    });
    Ok(pool)
}

/// After a request: fold the routing hits into the hotness, re-plan, and
/// apply a bounded batch of expert moves on a background thread (the pool
/// swaps are atomic per expert; generation keeps running). One pass at a
/// time; the rest of a large shift lands over the following requests.
fn rebalance_tiers() {
    use std::sync::atomic::Ordering;
    if REBALANCING.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(|| {
        if let Err(e) = rebalance_inner() {
            eprintln!("[mummu-serve] re-tier skipped: {e}");
        }
        REBALANCING.store(false, Ordering::SeqCst);
    });
}

fn rebalance_inner() -> Result<(), String> {
    const MAX_MOVES: usize = 16;
    const ALPHA: f64 = 0.3;
    let (pack_dir, pool, devices, epl, moves, pending) = {
        let mut g = TIERS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(rt) = g.as_mut() else { return Ok(()) };
        let hits = rt.pool.take_hits();
        mummu::tier::smooth_hotness(&mut rt.hotness, &hits, ALPHA);
        let planner_devices: Vec<mummu::tier::TierDevice> =
            rt.devices.iter().map(|(_, d)| d.clone()).collect();
        let next = mummu::tier::plan_tiers(&planner_devices, &rt.costs, &rt.hotness)?;
        let mut moves = rt.plan.diff(&next);
        if moves.is_empty() {
            return Ok(());
        }
        let pending = moves.len().saturating_sub(MAX_MOVES);
        moves.truncate(MAX_MOVES);
        for &(i, t) in &moves {
            rt.plan.tiers[i] = t;
        }
        (
            rt.pack_dir.clone(),
            rt.pool.clone(),
            rt.devices.clone(),
            rt.experts_per_layer,
            moves,
            pending,
        )
    };
    let pack = mummu::pack::Pack::open(&pack_dir)?;
    let started = Instant::now();
    let n = moves.len();
    for (flat, tier) in moves {
        let (layer, index) = (flat / epl, flat % epl);
        let next = load_expert_on(devices[tier.device].0, &pack, layer, index, tier)?;
        let old = pool.swap(layer, index, next);
        drop(old);
    }
    eprintln!(
        "[mummu-serve] re-tiered {n} experts in {:.1}s ({pending} more pending)",
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Per-tensor precision for a pack about to be loaded onto `backend`.
///
/// The fit planner picks ONE precision for a whole model, which is the
/// coarsest possible answer: at Q8 this 27B needs ~13 GiB and does not fit a
/// 16 GiB card with activation headroom, while at Q4 it needs ~7 GiB and
/// leaves ~6 GiB of the budget unspent. Neither is right. This spends that
/// headroom where it buys the most accuracy — attention projections, the LM
/// head, and the first and last layers ride at Q8 while the bulk FFN slabs
/// sit at Q4 — so the card holds the whole model AND the tensors that need
/// precision keep it. See [`mummu::mix`] for the ranking and its error model.
///
/// Returns a name -> precision map; names absent from it fall back to the
/// planner's single policy, so a pack this cannot classify still loads.
fn mixed_precision(
    pack: &mummu::pack::Pack,
    backend: BackendChoice,
    ceiling: mummu::quant::QuantPolicy,
    on_device: &dyn Fn(&mummu::pack::TensorEntry) -> bool,
) -> std::collections::HashMap<String, mummu::pack::Precision> {
    use mummu::mix::{Kind, TensorFacts};
    use mummu::pack::Role;
    use mummu::quant::QuantPolicy;

    let layers = pack
        .manifest
        .tensors
        .iter()
        .filter_map(|t| layer_index(&t.name))
        .max()
        .map_or(0, |n| n + 1);

    // Only tensors that will actually occupy this device may spend its
    // budget. The embedding is the reason this matters rather than being a
    // technicality: `token_embd.weight` is 1.27 G parameters and never
    // quantized, so charging its 5.09 GiB of f32 against a 12 GiB card
    // declared the model unfittable while the loader was putting it on the
    // host all along.
    let counted: Vec<&mummu::pack::TensorEntry> = pack
        .manifest
        .tensors
        .iter()
        .filter(|t| on_device(t))
        .collect();

    let facts: Vec<TensorFacts> = counted
        .iter()
        .map(|t| TensorFacts {
            params: t.shape.iter().product(),
            kind: match &t.role {
                // The embedding is a gather and the vectors anchor the
                // numerics; neither is ever quantized here.
                Role::Embedding | Role::Vector | Role::Conv => Kind::Fixed,
                // `output.weight` is the LM head — it writes the logits
                // directly, so it is graded with attention rather than with
                // the feed-forward bulk it superficially resembles.
                Role::Linear if t.name.contains("attn") || t.name == "output.weight" => {
                    Kind::Attention
                }
                Role::Linear | Role::Expert { .. } => Kind::Ffn,
            },
            layer: layer_index(&t.name),
        })
        .collect();

    // Never store more precision than the SOURCE carries. `source_bytes` over
    // the parameter count gives the checkpoint's bits/param — 4.55 for this
    // 27B (Q4_K_S), so its f32 is an upcast of 4-bit data and f16 holds it
    // exactly (measured 0.0000 relative error, `ladder-probe.rs`). Spending
    // 4 B/param there would be twice the bytes for none of the information,
    // which is also the answer to "why not f64": the ceiling is the source,
    // and the source is nowhere near it.
    let source_bits = source_bits_per_param(pack);
    let source_cap = mummu::quant::QuantPolicy::ceiling_for_source(source_bits);
    let ceiling = if ceiling.bits() > source_cap.bits() {
        eprintln!(
            "[mummu-serve] source is {source_bits:.2} bits/param — capping the ladder at {source_cap:?} (asked {ceiling:?})"
        );
        source_cap
    } else {
        ceiling
    };

    let budget = backend_budget(backend).saturating_sub(ACTIVATION_RESERVE);
    // Floor at Q4: it is the least precise thing any pack stores, so it is as
    // far as a *load* can go. Q2 exists below it but is reachable only by
    // requantizing a resident tensor, which is the rebalancer's job.
    let plan = mummu::mix::plan(&facts, layers, budget, ceiling, QuantPolicy::Q4);

    let summary: Vec<String> = plan
        .histogram()
        .iter()
        .map(|(p, n)| format!("{n} @ {p:?}"))
        .collect();
    eprintln!(
        "[mummu-serve] precision mix on {}: {} — {:.2} GiB of {:.2} GiB budget{}",
        label_of(backend),
        summary.join(", "),
        plan.bytes as f64 / f64::from(1u32 << 30),
        budget as f64 / f64::from(1u32 << 30),
        if plan.over_budget { " (OVER — will spill)" } else { "" },
    );

    let mut chosen: std::collections::HashMap<String, mummu::pack::Precision> = counted
        .iter()
        .zip(&plan.precision)
        .map(|(t, &p)| (t.name.clone(), precision_for(p)))
        .collect();
    // Everything else lives on the host, where RAM is not the scarce
    // resource: give it the ceiling rather than a pressure-driven rung.
    for t in &pack.manifest.tensors {
        chosen.entry(t.name.clone()).or_insert_with(|| {
            if matches!(t.role, Role::Embedding | Role::Vector | Role::Conv) {
                mummu::pack::Precision::F32
            } else {
                precision_for(ceiling)
            }
        });
    }
    chosen
}

/// Tell the scheduler what the trunk already costs the device that holds it.
///
/// The trunk runs on every token, exactly like every FFN cluster, but it is
/// placed by the fit planner rather than by scheduler A — so without this the
/// device carrying it looks completely idle and is handed a full share of
/// clusters on top. Measured: the host, already running a trunk that
/// saturates it, was given 191 further clusters as its "fair share".
///
/// Expressed in cluster equivalents so it is the same unit the scheduler
/// divides in: how many clusters' worth of work the trunk represents.
fn charge_trunk_to_its_device(
    devices: &mut [(BackendChoice, mummu::tier::TierDevice)],
    main: BackendChoice,
    trunk_bytes: u64,
    costs: &[mummu::tier::ExpertCost],
) {
    let cluster_bytes = {
        let mut total = 0u64;
        let mut n = 0u64;
        for c in costs {
            if let Some(&b) = c.bytes.values().max() {
                total += b;
                n += 1;
            }
        }
        if n == 0 { 0 } else { total / n }
    };
    if cluster_bytes == 0 {
        return;
    }
    let units = (trunk_bytes / cluster_bytes) as usize;
    for (backend, dev) in devices.iter_mut() {
        if *backend == main {
            dev.preload_units = units;
            eprintln!(
                "[mummu-serve] trunk charged to {}: {units} cluster-equivalents of work per token",
                dev.name
            );
        }
    }
}

/// The checkpoint's own bits per parameter, from the pack manifest.
fn source_bits_per_param(pack: &mummu::pack::Pack) -> f64 {
    let params: u64 = pack
        .manifest
        .tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>() as u64)
        .sum();
    if params == 0 {
        f64::INFINITY
    } else {
        pack.manifest.source_bytes as f64 * 8.0 / params as f64
    }
}

/// Drop rungs finer than the source from every device's ladder.
///
/// Precision above the source is bytes without information, and on a device
/// whose scarce resource is memory those bytes are *capacity* — every one
/// spent is an expert that could have been resident. Concretely: with F32 on
/// the ladder, the discrete GPU spent most of its 9 GiB on 219 F32 clusters
/// and could hold only 610 in total, while the 8.9x slower integrated GPU
/// carried 1374. This 27B ships at 4.55 bits/param, so its f32 is an upcast
/// of 4-bit data and f16 reproduces it exactly (measured 0.0000 relative
/// error) — those 219 clusters were paying quadruple for nothing.
fn cap_ladders_at_source(devices: &mut [(BackendChoice, mummu::tier::TierDevice)], pack: &mummu::pack::Pack) {
    use mummu::pack::Precision;
    let bits = source_bits_per_param(pack);
    let cap = mummu::quant::QuantPolicy::ceiling_for_source(bits);
    let cap_bits = cap.bits();
    let rung_bits = |p: Precision| match p {
        Precision::F32 => 32,
        Precision::F16 => 16,
        Precision::Q8 => 8,
        Precision::Q4 => 4,
    };
    let mut trimmed = Vec::new();
    for (_, dev) in devices.iter_mut() {
        let before = dev.ladder.len();
        dev.ladder.retain(|&p| rung_bits(p) <= cap_bits);
        // Never leave a device with no rung at all: if the cap removed
        // everything, keep the cheapest it had.
        if dev.ladder.is_empty() {
            dev.ladder.push(Precision::Q4);
        }
        if dev.ladder.len() != before {
            trimmed.push(dev.name.clone());
        }
    }
    if !trimmed.is_empty() {
        eprintln!(
            "[mummu-serve] source is {bits:.2} bits/param — ladders capped at {cap:?} on {}",
            trimmed.join(", ")
        );
    }
}

/// The layer a GGUF tensor name belongs to (`blk.7.ffn_up.weight` -> 7).
fn layer_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// The pack precision a fit policy denotes.
fn precision_for(policy: mummu::quant::QuantPolicy) -> mummu::pack::Precision {
    use mummu::pack::Precision;
    use mummu::quant::QuantPolicy;
    match policy {
        // No pack stores 2-bit: Q2 is reached by requantizing a tensor that
        // is already resident, never by reading one. Loading "at Q2" means
        // loading the smallest thing on disk and demoting from there.
        QuantPolicy::Q2 | QuantPolicy::Q4 => Precision::Q4,
        QuantPolicy::Q8 => Precision::Q8,
        QuantPolicy::F16 => Precision::F16,
        QuantPolicy::Off => Precision::F32,
    }
}

/// The `.mummu` pack beside a GGUF (`<model dir>/pack`), importing it on
/// first use when packing is on (`MUMMU_PACK` unset or `on`; `off` keeps
/// the GGUF streaming path). Stored precisions come from
/// `MUMMU_PACK_PRECISIONS` (default every level: q4,q8,f16,f32). Returns
/// `None` when packing is off; errors if the import fails.
fn ensure_pack(
    model_dir: &Path,
    gguf_path: &Path,
    spec: &ModelSpec,
) -> Result<Option<std::path::PathBuf>, String> {
    use mummu::pack::{Pack, Precision};
    if std::env::var("MUMMU_PACK").is_ok_and(|v| v.eq_ignore_ascii_case("off") || v == "0") {
        return Ok(None);
    }
    let pack_dir = model_dir.join("pack");
    if Pack::is_pack(&pack_dir) {
        ensure_partition(&pack_dir, spec)?;
        return Ok(Some(pack_dir));
    }
    let precisions = match std::env::var("MUMMU_PACK_PRECISIONS") {
        Ok(v) => Precision::parse_list(&v)?,
        Err(_) => Precision::ALL.to_vec(),
    };
    eprintln!(
        "[mummu-serve] importing {} to a .mummu pack ({precisions:?}) at {} — one-time",
        spec.name,
        pack_dir.display()
    );
    let started = Instant::now();
    let header = GgufFile::open(gguf_path).map_err(|e| e.to_string())?;
    type ActionMap = Box<dyn Fn(&mummu::gguf::GgufTensorInfo) -> Option<mummu::pack::ImportAction>>;
    let map: ActionMap = match spec.architecture {
            Architecture::Qwen35 => {
                let cfg = qwen35::Qwen35Config::from_gguf(&header)?;
                let trunk = cfg.num_layers;
                Box::new(move |info| qwen35::pack_actions(info, trunk))
            }
            Architecture::Olmoe => Box::new(olmoe::pack_actions),
            _ => return Ok(None), // other families stay on their classic paths
        };
    drop(header);
    // Import into a temp dir, then rename — a crash mid-import never leaves
    // a half pack that `is_pack` would accept.
    let tmp = model_dir.join("pack.importing");
    let _ = std::fs::remove_dir_all(&tmp);
    let mut last_pct = usize::MAX;
    mummu::pack::import_gguf(gguf_path, &tmp, &precisions, &*map, |i, n, name| {
        let pct = i * 100 / n.max(1);
        if pct != last_pct && pct % 5 == 0 {
            last_pct = pct;
            eprintln!("[mummu-serve] pack import {pct}% ({name})");
        }
    })?;
    std::fs::rename(&tmp, &pack_dir).map_err(|e| format!("finalize pack: {e}"))?;
    eprintln!(
        "[mummu-serve] pack ready in {:.0}s: {}",
        started.elapsed().as_secs_f32(),
        pack_dir.display()
    );
    ensure_partition(&pack_dir, spec)?;
    Ok(Some(pack_dir))
}

/// P9 stage 3(c): a dense qwen35 pack gets its FFNs partitioned into neuron
/// clusters **in place**, once (exact — the neurons are only reordered), so
/// the tier runtime can spread them across devices. Crash-safe per layer
/// (journal); a pack partitioned before is left alone.
fn ensure_partition(pack_dir: &Path, spec: &ModelSpec) -> Result<(), String> {
    // Every dense decoder, not just qwen35. Partitioning is EXACT — the
    // neurons of a layer's FFN are only reordered, consistently across
    // gate/up columns and down rows — so a pack is safe to partition even
    // for a loader that ignores the partition entirely. That is what makes
    // one path for every model possible: the pack carries the clusters, and
    // whether a given model's loader spreads them across devices is then its
    // own business rather than a fork in the format.
    //
    // OLMoE is excluded because its FFN is already a bank of experts with its
    // own tiering; MiniLm is an embedder with nothing worth spreading.
    let dense = matches!(
        spec.architecture,
        Architecture::Qwen2 | Architecture::Qwen3 | Architecture::Lfm2 | Architecture::Qwen35
    );
    if !dense {
        return Ok(());
    }
    let mut pack = mummu::pack::Pack::open(pack_dir)?;
    if pack.manifest.ffn_partition.is_some() {
        return Ok(());
    }
    // Trunk depth from the GGUF header, which every architecture records the
    // same way; the FFN names are the standard triple for all of them.
    let header = pack.header()?;
    let trunk = header
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".block_count"))
        .and_then(|(_, v)| v.as_u64())
        .map(|n| n as usize)
        .ok_or("pack header has no <arch>.block_count")?;
    drop(header);
    eprintln!(
        "[mummu-serve] partitioning {}'s FFNs into {} clusters per layer (one-time, in place)",
        spec.name,
        mummu::partition::DEFAULT_CLUSTERS
    );
    let started = Instant::now();
    mummu::partition::partition_pack(
        &mut pack,
        &mummu::partition::ffn_names(trunk),
        mummu::partition::DEFAULT_CLUSTERS,
        |i, n| {
            if i % 8 == 0 {
                eprintln!("[mummu-serve] partition layer {i}/{n} ({:.0}s)", started.elapsed().as_secs_f32());
            }
        },
    )?;
    eprintln!("[mummu-serve] FFNs partitioned in {:.0}s", started.elapsed().as_secs_f32());
    Ok(())
}

/// Where and how one model will run: the P6-lite fit decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitPlan {
    pub backend: BackendChoice,
    pub policy: mummu::quant::QuantPolicy,
}

/// Resident-bytes estimate for a GGUF under `policy`, from the header's
/// exact tensor inventory: quantize-eligible 2-D weights (everything but
/// the embedding) at the policy's storage cost, the rest at f32. The 1.35
/// factor covers allocator/runtime overhead (measured on the 27B: 33.7 GB
/// actual vs 25 GB raw at Q4).
fn estimate_resident_bytes(f: &GgufFile, policy: mummu::quant::QuantPolicy) -> u64 {
    use mummu::quant::QuantPolicy;
    let mut bytes = 0u64;
    for t in &f.tensors {
        let n = t.element_count();
        let dims_rev: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        // 3-D expert banks quantize per member; other 2-D weights (minus the
        // embedding) quantize whole. The loaders quantize TRANSPOSED views,
        // so eligibility checks [in, out].
        let quantizable = if t.name.ends_with("_exps.weight") && dims_rev.len() == 3 {
            policy.eligible(&[dims_rev[2], dims_rev[1]])
        } else {
            t.name != "token_embd.weight"
                && dims_rev.len() == 2
                && policy.eligible(&[dims_rev[1], dims_rev[0]])
        };
        bytes += if quantizable {
            match policy {
                QuantPolicy::Off => n * 4,
                QuantPolicy::F16 => n * 2,
                // i8 + f32 scale per 32 on flex; packed layouts are smaller.
                QuantPolicy::Q8 => n + n / 8,
                // measured flex packing: ~0.75 B/element incl. scales
                QuantPolicy::Q4 => (n * 3) / 4,
                // 2 bits + an f32 scale per 32 values, which is a third of it.
                QuantPolicy::Q2 => n / 4 + n / 8,
            }
        } else {
            n * 4
        };
    }
    // The streaming loaders hold ONE tensor at f32 while converting it —
    // the import's true peak is the resident model plus that largest tensor
    // (the 27B's 5 GiB embedding), and a plan that fits only the resident
    // bytes dies during load.
    let largest_f32 = f
        .tensors
        .iter()
        .map(|t| t.element_count() * 4)
        .max()
        .unwrap_or(0);
    (bytes as f64 * 1.35) as u64 + largest_f32
}

/// The memory budget of one backend. GPU budgets come from the wgpu
/// inventory where it exists (native Windows/Linux); the CUDA container has
/// no wgpu adapters, so a conservative default applies, overridable with
/// `MUMMU_GPU_BUDGET_GB`. CPU gets 3/4 of physical RAM.
/// Linux `MemAvailable` in bytes (None elsewhere).
fn mem_available_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
}

/// Make sure `need` bytes of host RAM are free before a load that lands on
/// the host: when `MemAvailable` is short and the CPU slot holds some other
/// model, evict it (the alternative is the VM's OOM killer taking the whole
/// server down — which is exactly what a resident 27B plus a tiered OLMoE
/// did on 2026-08-22). `keep` is the backend the new model's trunk goes to;
/// its own slot is evicted by `ModelSlot::with` anyway.
fn ensure_host_room(need: u64, keep: BackendChoice) {
    let Some(avail) = mem_available_bytes() else { return };
    if avail >= need || keep == BackendChoice::Cpu {
        return;
    }
    if let Some((dir, bytes)) = resident_in(BackendChoice::Cpu) {
        eprintln!(
            "[mummu-serve] host RAM: {} GiB available, {} GiB needed — evicting {} (~{} GiB) from the CPU slot",
            avail >> 30,
            need >> 30,
            dir.display(),
            bytes >> 30
        );
        if SLOT.clear() {
            RESIDENT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|(b, _, _)| *b != BackendChoice::Cpu);
        } else {
            // Busy: a generation holds the model. Say so — pretending the
            // eviction happened is how the caller walks into an OOM.
            eprintln!(
                "[mummu-serve] host RAM: could not evict — a generation is holding the model slot"
            );
        }
    }
}

/// System RAM the integrated GPU may claim, when nothing says otherwise.
/// Bounded because it shares that RAM with the CPU tier and with everything
/// else on the machine; `MUMMU_IGPU_BUDGET_GB` overrides.
const INTEGRATED_GPU_BUDGET: u64 = 8 << 30;

fn backend_budget(backend: BackendChoice) -> u64 {
    let inv = mummu::backend::inventory();
    match backend {
        // RAM that is actually free right now (a shared VM's total lies):
        // 85% of MemAvailable on Linux, 3/4 of total elsewhere — whichever
        // is smaller when both are known.
        // The integrated GPU has no memory of its own: it allocates from
        // system RAM, the same pool the CPU tier budgets against. Give it a
        // bounded slice rather than the ~101 GiB DXGI cheerfully reports for
        // it, and subtract that slice from the CPU below so the two tiers do
        // not both spend the same bytes.
        BackendChoice::IntegratedGpu => std::env::var("MUMMU_IGPU_BUDGET_GB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(INTEGRATED_GPU_BUDGET, |gb| gb << 30),
        BackendChoice::Cpu => {
            let total = inv.cpu.total_ram_bytes.map(|b| b / 4 * 3);
            let available = mem_available_bytes().map(|b| b / 100 * 85);
            let ram = match (total, available) {
                (Some(t), Some(a)) => t.min(a),
                (Some(t), None) => t,
                (None, Some(a)) => a,
                (None, None) => 16 << 30,
            };
            // Whatever the integrated GPU may take is already spoken for.
            if mummu::backend::has_integrated_gpu() {
                ram.saturating_sub(backend_budget(BackendChoice::IntegratedGpu))
            } else {
                ram
            }
        }
        // A GPU budget has to be what is free *now*, not what the card
        // holds. This desktop runs Firefox, Discord, Steam, Docker and two
        // driver overlays, which together took enough of a 16 GiB card that
        // a 9.8 GiB placement — comfortably inside its configured budget —
        // died with `out of device memory` mid-generation (2026-08-23).
        //
        // So the configured value is a CEILING, and the live reading from
        // NVML caps it. `MUMMU_GPU_BUDGET_GB` still means "never use more
        // than this", which is what an operator setting it wants; it just no
        // longer means "this much is definitely available".
        _ => {
            let configured = std::env::var("MUMMU_GPU_BUDGET_GB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(|gb| gb << 30)
                .or_else(|| {
                    inv.gpus
                        .iter()
                        .filter_map(|g| g.vram_bytes)
                        .max()
                        .map(|b| b / 8 * 7)
                })
                .unwrap_or(15 << 30);
            match mummu::vram::memory() {
                // Leave the desktop its working set. 2 GiB is what this box
                // idles at with a browser and the usual tray software up;
                // below that, compositing starts stuttering before we OOM.
                Some(m) => {
                    let live = m.headroom(2 << 30);
                    if live < configured {
                        eprintln!(
                            "[mummu-serve] VRAM: {:.1} GiB free of {:.1} GiB                              ({:.1} GiB held elsewhere) — budget {:.1} -> {:.1} GiB",
                            m.free as f64 / f64::from(1u32 << 30),
                            m.total as f64 / f64::from(1u32 << 30),
                            m.used as f64 / f64::from(1u32 << 30),
                            configured as f64 / f64::from(1u32 << 30),
                            live as f64 / f64::from(1u32 << 30),
                        );
                    }
                    configured.min(live)
                }
                // No reading: hold the configured value rather than guess in
                // either direction.
                None => configured,
            }
        }
    }
}

/// Pick where and how `spec` runs: start from the global backend choice and
/// the `MUMMU_QUANT` policy, escalate quantization until the model fits,
/// then fall back to the CPU with the same escalation. Q4-on-wgpu is never
/// planned (broken upstream kernel). Non-GGUF specs (and models whose
/// loader ignores quantization) plan as the global backend at `Off` — the
/// pre-planner behavior, unchanged.
fn plan_fit(spec: &ModelSpec, models_root: &Path) -> Result<FitPlan, String> {
    use mummu::quant::QuantPolicy;
    let preferred = backend_choice();
    let base_policy = QuantPolicy::from_env()?;

    // Only the qwen35 loader consumes a policy today; everything else keeps
    // its classic path on the preferred backend.
    let WeightFormat::Gguf { file } = &spec.format else {
        return Ok(FitPlan {
            backend: preferred,
            policy: QuantPolicy::Off,
        });
    };
    if !matches!(
        spec.architecture,
        Architecture::Qwen35 | Architecture::Olmoe
    ) {
        return Ok(FitPlan {
            backend: preferred,
            policy: QuantPolicy::Off,
        });
    }

    // Already resident somewhere? Serve it from there — no fit check, and
    // no reload (the slot is keyed by the model dir).
    let dir = spec.dir(models_root);
    let all_backends: &[BackendChoice] = &[
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda,
        BackendChoice::Wgpu,
        BackendChoice::Cpu,
    ];
    for &b in all_backends {
        if let Some((d, _)) = resident_in(b)
            && d == dir
        {
            return Ok(FitPlan {
                backend: b,
                policy: QuantPolicy::Off, // ignored: the slot skips the load
            });
        }
    }

    let path = dir.join(file);
    let f = GgufFile::open(&path).map_err(|e| e.to_string())?;
    // Units that tier out across devices (MoE experts, partitioned FFN
    // clusters) leave only the trunk to fit the preferred backend.
    let tiered_pack = if tiers_mode().is_some() {
        mummu::pack::Pack::open(&dir.join("pack")).ok().filter(|p| {
            p.manifest.ffn_partition.is_some()
                || p.manifest.tensors.iter().any(|t| matches!(t.role, mummu::pack::Role::Expert { .. }))
        })
    } else {
        None
    };

    let ladder = |from: QuantPolicy| -> Vec<QuantPolicy> {
        match from {
            // Whole-model fallbacks only. Q2 is not on this ladder: it is a
            // per-tensor pressure response (`mummu::mix`), not a precision
            // any model should be planned into wholesale.
            QuantPolicy::Off => vec![QuantPolicy::Off, QuantPolicy::Q8, QuantPolicy::Q4],
            QuantPolicy::F16 => vec![QuantPolicy::F16, QuantPolicy::Q8, QuantPolicy::Q4],
            QuantPolicy::Q8 => vec![QuantPolicy::Q8, QuantPolicy::Q4],
            QuantPolicy::Q4 | QuantPolicy::Q2 => vec![QuantPolicy::Q4],
        }
    };
    let mut candidates: Vec<(BackendChoice, QuantPolicy)> = Vec::new();
    // When the FFN is going to be TIERED, accelerator memory is worth more as
    // cluster capacity than as trunk capacity, so the trunk is tried on the
    // host first. Measured twice on the 27B, both times the same way round:
    // trunk on the GPU 24.7 s/tok against 4.8 for trunk on the host, and
    // again on 2026-08-23, 6.2 s/tok against 4.32. The mechanism is visible
    // in the placement — a trunk on the card took 7.8 of its 9 GiB and left
    // room for 302 of 2048 clusters, where a trunk on the host left the whole
    // budget for 996 of them. Every cluster runs on every token, so the card
    // holding more of them beats the card holding the trunk.
    //
    // Without tiering this does not apply: the whole model goes to one device
    // and the fastest one that fits is simply the right answer.
    let tiering = tiered_pack.is_some();
    if tiering && preferred != BackendChoice::Cpu {
        for policy in ladder(base_policy) {
            candidates.push((BackendChoice::Cpu, policy));
        }
    }
    for policy in ladder(base_policy) {
        candidates.push((preferred, policy));
    }
    if !tiering && preferred != BackendChoice::Cpu {
        for policy in ladder(base_policy) {
            candidates.push((BackendChoice::Cpu, policy));
        }
    }
    for (backend, policy) in candidates {
        // (The "wgpu Q4 kernel is broken" exclusion that used to live here
        // was withdrawn 2026-08-23. It rested on a probe that quantized
        // synthetic tensors on the device; along the path production takes —
        // packed bytes from the pack onto the device, then multiplied — Q4 on
        // wgpu is correct and bit-identical across repeated rounds at this
        // model's dimensions. See `examples/pack-precision-probe.rs`.)
        let need = match &tiered_pack {
            Some(pack) => pack_trunk_bytes(pack, precision_for(policy)),
            None => estimate_resident_bytes(&f, policy),
        };
        // Loading evicts the slot's current occupant first — its bytes come
        // back to the budget.
        let evictable = resident_in(backend).map_or(0, |(_, n)| n);
        let budget = (backend_budget(backend) + evictable).saturating_sub(tier_used_bytes(backend));
        eprintln!(
            "[mummu-serve] fit check {}: {backend:?} @ {policy:?} needs ~{} GiB, budget {} GiB (incl. {} GiB evictable)",
            spec.name,
            need >> 30,
            budget >> 30,
            evictable >> 30
        );
        if need <= budget {
            note_resident(backend, dir.clone(), need);
            return Ok(FitPlan { backend, policy });
        }
    }
    Err(format!(
        "{} does not fit any backend even at Q4 (needs ~{} GiB quantized)",
        spec.name,
        estimate_resident_bytes(&f, QuantPolicy::Q4) >> 30
    ))
}

#[allow(clippy::too_many_arguments)] // one call site, mirrors the plan
async fn drive(
    slot: &ModelSlot<Loaded>,
    spec: &ModelSpec,
    models_root: &Path,
    prompt: &str,
    opts: &SamplerOptions,
    max_tokens: usize,
    policy: mummu::quant::QuantPolicy,
    backend: BackendChoice,
    label: &'static str,
    mut on_delta: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<ChatResult, String> {
    // The device the fit plan chose — NOT `Device::default()`, which would
    // silently run wherever the process first touched a backend and make the
    // planner's decision cosmetic.
    let device = device_of(backend);
    let key = spec.dir(models_root);
    if slot.loaded_key_async().await.as_deref() != Some(key.as_path())
        && tiers_pack_in(&key).is_none()
    {
        // This slot is about to evict its occupant; if that was the tiered
        // model, its experts on the *other* devices go with it.
        clear_tiers_if_slot(backend);
    }
    // The load closure and the async body both want `device`; give the loader
    // its own handle so the future can capture the original by reference.
    let load_device = device.clone();
    let m = slot
        .acquire(&key, move |_| {
            // One bar for the whole load: a profiled COLD request would
            // otherwise smear minutes of import across the decode graph.
            let _s = mummu::prof::scope("model_load");
            load_any(spec, models_root, &load_device, policy, backend)
        })
        .await?;
    {
            // ChatML renderers leave specials to the tokenizer; the Tulu
            // render already embeds its own BOS (the real_olmoe.rs pattern).
            let add_special = spec.architecture != Architecture::Olmoe;
            let prompt_ids = m
                .tokenizer
                .encode(prompt, add_special)
                .map_err(|e| format!("prompt encode: {e}"))?
                .get_ids()
                .to_vec();
            if prompt_ids.is_empty() {
                return Err("prompt encoded to zero tokens".into());
            }

            let start = Instant::now();
            let mut ids: Vec<u32> = Vec::new();
            let mut emitted = String::new();
            let out = m.lm.generate(&prompt_ids, max_tokens, opts, &device, |id| {
                ids.push(id);
                // Incremental decode: re-decode the whole tail and emit the
                // suffix beyond what was already streamed. A trailing U+FFFD
                // means we're mid-way through a multi-byte char — hold the
                // delta until the next token completes it.
                let Ok(text) = m.tokenizer.decode(&ids, true) else {
                    return ControlFlow::Continue(());
                };
                if text.ends_with('\u{FFFD}') || text.len() <= emitted.len() {
                    return ControlFlow::Continue(());
                }
                let delta = text[emitted.len()..].to_string();
                emitted = text;
                on_delta(&delta)
            })
            .await?;

            let text = m
                .tokenizer
                .decode(&out, true)
                .map_err(|e| format!("decode: {e}"))?;
            let elapsed_ms = start.elapsed().as_millis();
            // Every completed generation is one placement's worth of evidence.
            observe_placement(out.len(), elapsed_ms);
            if matches!(&m.lm, AnyLm::OlmoeQ(q) if q.pool.is_some()) {
                rebalance_tiers();
            }
        Ok(ChatResult {
            text,
            tokens: out.len(),
            device: label,
            elapsed_ms,
        })
    }
}

/// Is every artifact of `spec` on disk? (`ModelManager::is_installed` only
/// knows the safetensors layout; GGUF installs are the single model file.)
pub fn is_installed(spec: &ModelSpec, models_root: &Path) -> bool {
    match &spec.format {
        WeightFormat::Safetensors => {
            let dir = spec.dir(models_root);
            let weights = dir.join("model.safetensors").is_file()
                || dir.join("model.safetensors.index.json").is_file();
            weights && dir.join("config.json").is_file() && dir.join("tokenizer.json").is_file()
        }
        WeightFormat::Gguf { .. } => spec
            .gguf_path(models_root)
            .is_some_and(|p| p.is_file()),
    }
}
