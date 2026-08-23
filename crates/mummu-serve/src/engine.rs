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
                            Some(cpu_only) if partitioned => {
                                // P9 stage 3c: trunk + local FFN clusters here,
                                // the other clusters tiered across devices.
                                clear_tiers();
                                AnyLm::Qwen35(build_partitioned_qwen35(
                                    &pack_dir, device, backend, cpu_only, policy,
                                )?)
                            }
                            _ => AnyLm::Qwen35(
                                qwen35::load_from_pack(&pack_dir, device, &|_| level)
                                    .map_err(|e| e.to_string())?,
                            ),
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
        BackendChoice::Cpu => mummu::backend::cpu_device(),
    }
}

/// Human label for a backend choice.
pub(crate) fn label_of(backend: BackendChoice) -> &'static str {
    match backend {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => "GPU (cuda)",
        BackendChoice::Wgpu => "GPU (wgpu)",
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
        backend_budget(b).saturating_sub(if b == main { trunk_bytes } else { 0 })
    };
    // `speed` ranks devices for THIS workload, and the workload here is
    // decode: one token at a time, so every projection is a matrix-VECTOR
    // product. Measured on the reference box for `[1,5120]x[5120,2176]`,
    // with a WARM autotune cache:
    //
    //   GPU (+sync)  2.5-3.3 ms      CPU (flex)  ~1.7 ms equivalent
    //
    // Close, with the CPU modestly ahead — nothing like the 25x gap an
    // earlier COLD-cache measurement suggested (96 ms, which was autotune
    // being paid inside the timing loop, not steady-state throughput; see
    // `examples/sync-probe.rs` and warm it before believing a number).
    //
    // The GPU's 2.5 ms is launch-latency bound, not bandwidth bound — it
    // moves 44 MB in that time, far under VRAM bandwidth — so it amortizes
    // much better as work per launch grows: at m=8 the same FFN matmul costs
    // the SAME wall-clock as m=1 (8x the work for free). Batched prefill and
    // speculative decode are what make GPU tiers pay properly.
    //
    // The ranking is therefore a close call that depends on batch size, and
    // the default keeps the CPU marginally ahead for single-token decode
    // while `MUMMU_TIER_SPEED=gpu-first` flips it for batched serving. Do
    // not read either ordering as a large effect.
    let decode_speed = |gpu: bool| -> u32 {
        if std::env::var("MUMMU_TIER_SPEED")
            .ok()
            .as_deref()
            .is_some_and(|v| v.eq_ignore_ascii_case("gpu-first"))
        {
            // Escape hatch for batched/prefill-heavy serving, where the
            // measured ordering flips.
            if gpu { 10 } else { 1 }
        } else if gpu {
            1
        } else {
            10
        }
    };
    let mut out = vec![(
        BackendChoice::Cpu,
        TierDevice {
            name: "cpu".into(),
            class: DeviceClass::Cpu,
            ladder: vec![Precision::Q8, Precision::Q4],
            budget_bytes: budget(BackendChoice::Cpu),
            speed: decode_speed(false),
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
                speed: decode_speed(true),
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
                ladder: vec![Precision::F32, Precision::Q8],
                budget_bytes: budget(BackendChoice::Wgpu),
                speed: decode_speed(true),
            },
        ));
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
    // Remote clusters grouped by device: one executor per (layer, device),
    // covering every cluster the plan put there at that device's tier. In
    // exact mode all clusters run, so this is one matmul per device per
    // layer — not one per cluster. (Different clusters on one device could
    // in principle carry different precisions, but the FFN plan uses each
    // device's single ladder level, so a group shares its precision.)
    let mut rows: Vec<Vec<std::sync::Arc<dyn mummu::nn::ExpertExec>>> = Vec::with_capacity(layers);
    for l in 0..layers {
        let mut by_device: std::collections::BTreeMap<usize, (mummu::tier::Tier, Vec<usize>, u64)> =
            std::collections::BTreeMap::new();
        for c in 0..epl {
            let tier = plan.tiers[l * epl + c];
            if tier.device == main_idx {
                continue; // the local slab is in the model's own mlp
            }
            let bytes = costs[l * epl + c].bytes.get(&tier.precision).copied().unwrap_or(0);
            let g = by_device.entry(tier.device).or_insert((tier, Vec::new(), 0));
            g.1.push(c);
            g.2 += bytes;
        }
        let mut row = Vec::new();
        for (dev, (tier, clusters, bytes)) in by_device {
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
    let devices = tier_devices(main, trunk, cpu_only);
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

/// The pack precision a fit policy denotes.
fn precision_for(policy: mummu::quant::QuantPolicy) -> mummu::pack::Precision {
    use mummu::pack::Precision;
    use mummu::quant::QuantPolicy;
    match policy {
        QuantPolicy::Q4 => Precision::Q4,
        QuantPolicy::Q8 => Precision::Q8,
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
    if spec.architecture != Architecture::Qwen35 {
        return Ok(());
    }
    let mut pack = mummu::pack::Pack::open(pack_dir)?;
    if pack.manifest.ffn_partition.is_some() {
        return Ok(());
    }
    let header = pack.header()?;
    let trunk = qwen35::Qwen35Config::from_gguf(&header)?.num_layers;
    drop(header);
    eprintln!(
        "[mummu-serve] partitioning {}'s FFNs into {} clusters per layer (one-time, in place)",
        spec.name,
        mummu::partition::DEFAULT_CLUSTERS
    );
    let started = Instant::now();
    mummu::partition::partition_pack(
        &mut pack,
        &qwen35::ffn_names(trunk),
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
                // i8 + f32 scale per 32 on flex; packed layouts are smaller.
                QuantPolicy::Q8 => n + n / 8,
                // measured flex packing: ~0.75 B/element incl. scales
                QuantPolicy::Q4 => (n * 3) / 4,
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

fn backend_budget(backend: BackendChoice) -> u64 {
    let inv = mummu::backend::inventory();
    match backend {
        // RAM that is actually free right now (a shared VM's total lies):
        // 85% of MemAvailable on Linux, 3/4 of total elsewhere — whichever
        // is smaller when both are known.
        BackendChoice::Cpu => {
            let total = inv.cpu.total_ram_bytes.map(|b| b / 4 * 3);
            let available = mem_available_bytes().map(|b| b / 100 * 85);
            match (total, available) {
                (Some(t), Some(a)) => t.min(a),
                (Some(t), None) => t,
                (None, Some(a)) => a,
                (None, None) => 16 << 30,
            }
        }
        _ => std::env::var("MUMMU_GPU_BUDGET_GB")
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
            .unwrap_or(15 << 30),
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
            QuantPolicy::Off => vec![QuantPolicy::Off, QuantPolicy::Q8, QuantPolicy::Q4],
            QuantPolicy::Q8 => vec![QuantPolicy::Q8, QuantPolicy::Q4],
            QuantPolicy::Q4 => vec![QuantPolicy::Q4],
        }
    };
    let mut candidates: Vec<(BackendChoice, QuantPolicy)> = Vec::new();
    for policy in ladder(base_policy) {
        candidates.push((preferred, policy));
    }
    if preferred != BackendChoice::Cpu {
        for policy in ladder(base_policy) {
            candidates.push((BackendChoice::Cpu, policy));
        }
    }
    for (backend, policy) in candidates {
        if policy == QuantPolicy::Q4 && backend == BackendChoice::Wgpu {
            continue; // wgpu Q4 kernel is broken — never plan it
        }
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
            if matches!(&m.lm, AnyLm::OlmoeQ(q) if q.pool.is_some()) {
                rebalance_tiers();
            }
        Ok(ChatResult {
            text,
            tokens: out.len(),
            device: label,
            elapsed_ms: start.elapsed().as_millis(),
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
