//! Live placement of a layered model: which device each layer is on and at
//! what precision, decided by [`mummu::mix::joint`] from measurements and
//! re-decided while the model serves. The serve side of that solver: where
//! its inputs come from, and how its decisions reach a resident model.
//!
//! # No configured budget
//!
//! Every quantity below is measured or derived from the model's shapes. The
//! knobs that used to cap placement (`MUMMU_GPU_BUDGET_GB`,
//! `MUMMU_ACTIVATION_RESERVE_GB`, `MUMMU_VRAM_GUARD_GB`, `MUMMU_CTX`,
//! `MUMMU_LAYER_PREFIX`, the `MUMMU_VRAM_LIVE_BUDGET` gate) are gone: a static
//! cap is right on exactly one day on one box, and wrong in both directions
//! every other day — too big when a co-tenant arrives (an OOM), too small
//! when it leaves (layers on the host for nothing). Measured on the 27B
//! before this change: a fixed 9 GiB cap less a 1.36 GiB vision reserve held
//! for text-only traffic put 27 of 64 layers on a card with room for more.
//!
//! # The equations
//!
//! For the accelerator, with NVML's global `total`/`used` and our own
//! allocator's `reserved`/`in_use` (cubecl's pool, per device):
//!
//! ```text
//!   ambient      A  = used − reserved                          everything not ours
//!   guard        G  = Watermark(A)                             (1−α)-quantile envelope
//!                                                              + fragmentation slack,
//!                                                              ×1.5 per allocation failure
//!   capacity     K  = total − G                                what we may occupy
//!   non-layer    N  = in_use − Σ_{l on card} resident(l)       head, vision tower, residue
//!   for layers   C  = K − N                                    joint::Device::capacity
//!   working set  W  = act(min(ctx, chunk)) + ε̂ + V·[tower needed, not resident]
//!                                                              joint::Device::fixed
//!   state        s_l = KV_l(ctx)  or  conv_l + S_l              joint::Layer::state_bytes
//! ```
//!
//! `act` is the widest live prefill buffer (three `[chunk, intermediate]`
//! f32 tensors through SwiGLU); `ε̂` is the allocator residual the analytic
//! terms do not explain — measured after every generation as
//! `reserved − in_use_before − Σ s_l − act` and tracked as an envelope over
//! the last requests (vLLM's profiling pass does the same for its KV budget:
//! run the real workload and budget from the peak it produced, not from a
//! constant). Before the first measurement ε̂ is a prior of one allocator
//! page, and it is replaced by what the card shows.
//!
//! `ctx` is the context actually being served: the request's own prompt plus
//! its token budget when a request is about to run, and the envelope of
//! recent requests' contexts when re-planning at idle.
//!
//! The host is device 0, the fallback: its capacity is 85% of
//! `MemAvailable` plus what our own host layers already hold.
//!
//! Rates are measured, not assumed: one real decode-shape projection is timed
//! on each device at each level it can run, under whatever contention the
//! machine has right now, and divided by that projection's pack bytes
//! (`s[d][p]`, joint's equation (1)). The host's is clamped to its DRAM
//! floor — the probe tensor fits L3, production streams every host layer per
//! token. The crossing cost κ is one timed activation round trip.
//!
//! # When it re-plans
//!
//! * **Before every request** (holding the model): with the request's own
//!   context and whether it needs the vision tower. If the resident placement
//!   would not fit beside that, joint repairs it first — the cheapest relief
//!   per byte — so the request never OOMs the card because it is long or
//!   carries an image.
//! * **Every 5 s at idle** (only if nobody holds the model): with the current
//!   ambient. Ambient grew → repair now. Ambient shrank, or a tower went
//!   idle and was dropped → improve, but only when (4) says the moved bytes
//!   pay for themselves over the horizon `H` (tokens served in the last
//!   hour), and only after the improvement has been wanted on three ticks in
//!   a row. The watermark only shrinks after a quiet window, so a co-tenant
//!   that allocates in bursts does not cause a reload storm.
//!
//! Moves are applied releases-first: layers leaving the card (and in-place
//! demotions) go before arrivals, the allocator returns the freed pages to
//! the driver, and only then do layers arrive. An improvement moves at most
//! [`IMPROVE_STEP`] layers per tick so a request arriving mid-rebalance waits
//! for seconds, not for the whole move.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mummu::mix::joint;
use mummu::mix::{Kind, QuantPolicy};
use mummu::models::qwen35;
use mummu::pack::{Pack, Precision, Role, TensorEntry};

use super::{AnyLm, BackendChoice, Loaded, SLOT, device_of, label_of, layer_index};

/// How often the idle rebalancer looks.
const TICK: Duration = Duration::from_secs(5);
/// Ticks an improvement must be wanted on before it starts.
const IMPROVE_DWELL: u32 = 3;
/// Layers an improvement may move per tick.
pub(super) const IMPROVE_STEP: usize = 2;
/// Fraction of token time precision upgrades may cost (joint `tolerance`).
const PRECISION_TOLERANCE: f64 = 0.01;
/// How long a resident vision tower may sit unused before its VRAM goes back
/// to layers.
const TOWER_IDLE: Duration = Duration::from_secs(600);
/// The window the token horizon `H` counts over.
const HORIZON_WINDOW: Duration = Duration::from_secs(3600);
/// Bound on the NVML read (the driver can wedge; see `status`).
const READ_BUDGET: Duration = Duration::from_millis(500);
/// Requests the working-set and context envelopes remember.
const ENVELOPE: usize = 8;

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

/// One reading of the accelerator, our own share separated out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Card {
    pub total: u64,
    pub used: u64,
    /// cubecl pool: bytes it has taken from the driver.
    pub reserved: u64,
    /// cubecl pool: bytes live tensors hold.
    pub in_use: u64,
}

/// Our pool on `backend`'s device. Asked of the device server, so it waits
/// behind queued work — callers hold the model slot, where nothing is queued.
fn pool(backend: BackendChoice) -> (u64, u64) {
    device_of(backend).memory_pool_usage().map_or((0, 0), |u| {
        (u.bytes_reserved, u.bytes_in_use.min(u.bytes_reserved))
    })
}

/// The card, if anything on this machine reports it.
///
/// NVML first (global used/free, every process). Without it — a machine
/// with no NVIDIA driver — the adapter inventory gives the total, and
/// "used" is only what we can see of ourselves; the guard's floor and slack
/// are then the whole margin, which is the most any machine without a
/// reading can honestly do.
pub(super) fn card(backend: BackendChoice) -> Option<Card> {
    if backend == BackendChoice::Cpu {
        return None;
    }
    let (reserved, in_use) = pool(backend);
    match crate::status::vram_reading(READ_BUDGET) {
        Some(m) => Some(Card {
            total: m.total,
            used: m.used.max(reserved),
            reserved,
            in_use,
        }),
        None => {
            let total = mummu::backend::inventory()
                .gpus
                .iter()
                .filter_map(|g| g.vram_bytes)
                .max()?;
            Some(Card {
                total,
                used: reserved,
                reserved,
                in_use,
            })
        }
    }
}

/// The chance-constrained guard on ambient VRAM (SPEC 3), fed on every call.
fn guard(ambient: u64) -> u64 {
    use mummu::schedule::watermark::{Watermark, WatermarkConfig};
    static WM: Mutex<Option<Watermark>> = Mutex::new(None);
    let mut g = WM.lock().unwrap_or_else(|e| e.into_inner());
    let wm = g.get_or_insert_with(|| {
        Watermark::new(WatermarkConfig {
            floor_bytes: 1 << 30,
            frag_slack_bytes: 512 << 20,
            ..WatermarkConfig::default()
        })
    });
    wm.observe_ambient(ambient);
    if super::ALLOC_FAILED.load(std::sync::atomic::Ordering::SeqCst) {
        wm.breach();
    }
    wm.guard_bytes().max(ambient)
}

/// `K = total − G`: what this process may occupy on the card.
pub(super) fn capacity(c: &Card) -> u64 {
    let ambient = c.used.saturating_sub(c.reserved);
    c.total.saturating_sub(guard(ambient))
}

/// Free for a NEW placement on `backend` — what every planner that has no
/// resident layers to count (the fit planner, the precision mix, the MoE
/// tiers) spends: `K − in_use − V_pending`.
pub(super) fn free_for_new(backend: BackendChoice) -> u64 {
    let Some(c) = card(backend) else {
        return 0;
    };
    capacity(&c)
        .saturating_sub(c.in_use)
        .saturating_sub(super::VISION_RESERVE.load(std::sync::atomic::Ordering::SeqCst))
}

/// Recent samples, envelope semantics: the max of the last [`ENVELOPE`].
#[derive(Debug, Default)]
struct Envelope(VecDeque<u64>);

impl Envelope {
    fn push(&mut self, v: u64) {
        if self.0.len() == ENVELOPE {
            self.0.pop_front();
        }
        self.0.push_back(v);
    }
    fn max(&self) -> Option<u64> {
        self.0.iter().copied().max()
    }
}

/// ε̂ — the allocator residual; prior one pool page until measured.
static RESIDUAL: Mutex<Envelope> = Mutex::new(Envelope(VecDeque::new()));
const RESIDUAL_PRIOR: u64 = 1 << 30;

fn residual() -> u64 {
    RESIDUAL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .max()
        .unwrap_or(RESIDUAL_PRIOR)
}

/// Non-weight bytes a planner without a model config should hold back:
/// the measured residual plus, before any measurement, its prior.
pub(super) fn nonweight_estimate() -> u64 {
    residual()
}

/// Contexts (prompt + budget) of recent requests.
static CONTEXTS: Mutex<Envelope> = Mutex::new(Envelope(VecDeque::new()));

/// The context an idle re-plan provisions for.
fn idle_context() -> usize {
    CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .max()
        .map_or(4096, |c| c as usize)
}

/// Tokens served, for the horizon `H`.
static SERVED: Mutex<VecDeque<(Instant, usize)>> = Mutex::new(VecDeque::new());

fn horizon_tokens() -> f64 {
    let mut s = SERVED.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    while s
        .front()
        .is_some_and(|(t, _)| now.duration_since(*t) > HORIZON_WINDOW)
    {
        s.pop_front();
    }
    // At least one reply's worth: a server that just started serving should
    // be able to take back a card that freed up.
    (s.iter().map(|(_, n)| *n).sum::<usize>()).max(256) as f64
}

/// Measured pack read rate, seconds per byte (prior: this array's quiet
/// 150 MB/s until a load or a move measures it).
static DISK_S_PER_BYTE: Mutex<f64> = Mutex::new(1.0 / 150e6);

fn note_disk(bytes: u64, secs: f64) {
    if bytes < (64 << 20) || secs <= 0.0 {
        return;
    }
    let mut d = DISK_S_PER_BYTE.lock().unwrap_or_else(|e| e.into_inner());
    // EWMA: one slow read under a co-tenant should not define the disk.
    *d = 0.7 * *d + 0.3 * (secs / bytes as f64);
}

fn disk_s_per_byte() -> f64 {
    *DISK_S_PER_BYTE.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the current request needs beyond the weights: its context and
/// whether it brings an image. Set per request before planning.
static REQUEST: Mutex<Option<(usize, bool)>> = Mutex::new(None);

/// Publish the request about to be planned for. `ctx` may be an estimate
/// here; [`before_request`] replaces it with the exact token count.
pub(super) fn set_request(ctx: usize, needs_tower: bool) {
    *REQUEST.lock().unwrap_or_else(|e| e.into_inner()) = Some((ctx, needs_tower));
}

fn request() -> Option<(usize, bool)> {
    *REQUEST.lock().unwrap_or_else(|e| e.into_inner())
}

/// When the vision tower was last used, for [`TOWER_IDLE`].
static TOWER_USED: Mutex<Option<Instant>> = Mutex::new(None);

pub(super) fn note_tower_use() {
    *TOWER_USED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
}

// ---------------------------------------------------------------------------
// The model as the solver sees it
// ---------------------------------------------------------------------------

fn policy_of(p: Precision) -> QuantPolicy {
    match p {
        Precision::Q4 => QuantPolicy::Q4,
        Precision::Q8 => QuantPolicy::Q8,
        Precision::F16 => QuantPolicy::F16,
        Precision::F32 => QuantPolicy::Off,
    }
}

fn precision_of(q: QuantPolicy) -> Precision {
    match q {
        QuantPolicy::Q4 | QuantPolicy::Q2 => Precision::Q4,
        QuantPolicy::Q8 => Precision::Q8,
        QuantPolicy::F16 => Precision::F16,
        QuantPolicy::Off => Precision::F32,
    }
}

fn blob_bytes(e: &TensorEntry, p: Precision) -> Option<u64> {
    e.precisions.get(&p).map(|b| b.values_len + b.scales_len)
}

/// Is this tensor one whose precision the solver chooses? A linear weight
/// stored at both integer levels; anything else keeps its load precision.
fn choosable(e: &TensorEntry) -> bool {
    matches!(e.role, Role::Linear | Role::Expert { .. })
        && e.precisions.contains_key(&Precision::Q4)
        && e.precisions.contains_key(&Precision::Q8)
}

fn kind_of(e: &TensorEntry) -> Kind {
    if e.name.contains("ffn") {
        Kind::Ffn
    } else {
        Kind::Attention
    }
}

/// The precision a non-choosable tensor always loads at: its most precise
/// stored level (norms, conv kernels and the narrow projections are read as
/// f32 whatever is asked).
fn fixed_precision(e: &TensorEntry) -> Precision {
    e.precisions.keys().copied().max().unwrap_or(Precision::F32)
}

/// One layer's tensors grouped the way the solver chooses them.
#[derive(Debug, Clone)]
struct LayerMap {
    /// Tensor names of each part, parallel to `parts`.
    names: Vec<Vec<String>>,
    parts: Vec<joint::Part>,
    fixed_bytes: u64,
}

fn layer_maps(pack: &Pack, layers: usize, ceiling: QuantPolicy) -> Vec<LayerMap> {
    let mut maps = vec![
        LayerMap {
            names: vec![Vec::new(), Vec::new()],
            parts: vec![
                joint::Part {
                    params: 0,
                    kind: Kind::Attention,
                    levels: Vec::new(),
                },
                joint::Part {
                    params: 0,
                    kind: Kind::Ffn,
                    levels: Vec::new(),
                },
            ],
            fixed_bytes: 0,
        };
        layers
    ];
    // Levels every tensor of a part stores, with the part's summed bytes.
    let mut sums: Vec<[Option<std::collections::BTreeMap<Precision, u64>>; 2]> =
        vec![[None, None]; layers];
    for e in &pack.manifest.tensors {
        let Some(l) = layer_index(&e.name).filter(|&l| l < layers) else {
            continue;
        };
        if !choosable(e) {
            maps[l].fixed_bytes += e.shape.iter().product::<usize>() as u64 * 4;
            continue;
        }
        let pi = usize::from(kind_of(e) == Kind::Ffn);
        maps[l].names[pi].push(e.name.clone());
        maps[l].parts[pi].params += e.shape.iter().product::<usize>();
        let here: std::collections::BTreeMap<Precision, u64> = Precision::ALL
            .iter()
            .filter(|&&p| policy_of(p).bits() <= ceiling.bits())
            .filter_map(|&p| blob_bytes(e, p).map(|b| (p, b)))
            .collect();
        let acc = &mut sums[l][pi];
        *acc = Some(match acc.take() {
            None => here,
            Some(prev) => prev
                .into_iter()
                .filter_map(|(p, b)| here.get(&p).map(|h| (p, b + h)))
                .collect(),
        });
    }
    for (l, s) in sums.into_iter().enumerate() {
        for (pi, levels) in s.into_iter().enumerate() {
            maps[l].parts[pi].levels = levels
                .unwrap_or_default()
                .into_iter()
                .map(|(p, b)| (policy_of(p), b))
                .collect();
        }
        // A layer with no choosable tensor of a kind has an empty part; it
        // costs nothing and stores nothing, so drop it.
        let keep: Vec<bool> = maps[l].parts.iter().map(|p| p.params > 0).collect();
        let mut i = 0;
        maps[l].parts.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        let mut i = 0;
        maps[l].names.retain(|_| {
            i += 1;
            keep[i - 1]
        });
    }
    maps
}

/// Per-generation state a layer carries at `ctx` tokens (joint's `s_l`).
pub(super) fn state_bytes(cfg: &qwen35::Qwen35Config, layer: usize, ctx: usize) -> u64 {
    let f32b = 4u64;
    if cfg.is_attention(layer) {
        let kv = if mummu::nn::kv_f16_enabled() { 2 } else { f32b };
        2 * cfg.num_key_value_heads as u64 * ctx as u64 * cfg.head_dim as u64 * kv
    } else {
        cfg.conv_dim() as u64 * cfg.conv_kernel.saturating_sub(1) as u64 * f32b
            + cfg.n_v_heads as u64 * (cfg.d_state as u64).pow(2) * f32b
    }
}

/// The widest live activation: three `[chunk, intermediate]` f32 buffers
/// through SwiGLU. Prefill is chunked, so this is bounded by the chunk, not
/// the context.
pub(super) fn act_bytes(cfg: &qwen35::Qwen35Config, ctx: usize) -> u64 {
    3 * ctx.min(mummu::decode::prefill_chunk_len()).max(1) as u64 * cfg.intermediate_size as u64 * 4
}

/// A device's measured behaviour.
#[derive(Debug, Clone, Default)]
struct DeviceModel {
    rate: Vec<(QuantPolicy, f64)>,
    resident: Vec<(QuantPolicy, f64)>,
}

/// Time a real projection at each level `device` can run, as seconds per
/// pack byte. The host's reading is clamped to its DRAM floor.
fn measure_device(
    pack: &Pack,
    device: &burn::tensor::Device,
    host: bool,
    levels: &[Precision],
) -> DeviceModel {
    let Some(entry) = pack.entry("blk.0.ffn_gate.weight") else {
        return DeviceModel::default();
    };
    let numel = entry.shape.iter().product::<usize>() as f64;
    let mut m = DeviceModel::default();
    for &p in levels {
        let Some(bytes) = blob_bytes(entry, p) else {
            continue;
        };
        let Some(ms) = super::probe_projection_ms(pack, device, p) else {
            continue;
        };
        let ms = if host {
            ms.max(super::host_probe_floor_ms(pack, p).unwrap_or(0.0))
        } else {
            ms
        };
        let q = policy_of(p);
        m.rate.push((q, ms / 1e3 / bytes as f64));
        // Resident bytes per pack byte on this device at this level.
        let resident = match (host, p) {
            // i8 slab + f32 block scales, plus the packed VNNI twin beside it.
            (true, Precision::Q4) => {
                let twin = if mummu::flex::registry::enabled() {
                    0.5625
                } else {
                    0.0
                };
                numel * (1.0 + 0.125 + twin)
            }
            (true, Precision::Q8) => numel * 1.125,
            // Floats are widened to f32 on load (host, wgpu; assumed on CUDA).
            (_, Precision::F16) => numel * 4.0,
            (_, Precision::F32) => numel * 4.0,
            (false, _) => bytes as f64,
        };
        m.resident.push((q, resident / bytes as f64));
    }
    m
}

/// One activation round trip host -> card -> host, seconds.
fn measure_crossing(hidden: usize, accel: &burn::tensor::Device) -> f64 {
    use burn::tensor::Tensor;
    let host = mummu::backend::cpu_device();
    let x = Tensor::<2>::zeros([1, hidden], &host);
    let trip = || {
        let y = x.clone().to_device(accel).to_device(&host);
        let _ = y.into_data().try_to_vec::<f32>();
    };
    trip();
    let reps = 5;
    let t0 = Instant::now();
    for _ in 0..reps {
        trip();
    }
    // A token crosses once each way at a boundary; the round trip is two.
    t0.elapsed().as_secs_f64() / f64::from(reps) / 2.0
}

/// A resident layered model's placement and everything needed to re-plan it.
pub(super) struct Live {
    pub pack_dir: PathBuf,
    pub backend: BackendChoice,
    pub model: String,
    cfg: qwen35::Qwen35Config,
    maps: Vec<LayerMap>,
    host: DeviceModel,
    accel: DeviceModel,
    crossing_s: f64,
    pub assignment: joint::Assignment,
    improve_wanted: u32,
}

pub(super) static LIVE: Mutex<Option<Live>> = Mutex::new(None);

impl Live {
    /// Measure the devices for `pack` and build the solver's view of it.
    pub fn measure(pack_dir: &Path, backend: BackendChoice) -> Result<Self, String> {
        let pack = Pack::open(pack_dir)?;
        let cfg = qwen35::Qwen35Config::from_gguf(&pack.header()?)?;
        let ceiling = QuantPolicy::ceiling_for_source(super::source_bits_per_param(&pack));
        let maps = layer_maps(&pack, cfg.num_layers, ceiling);
        let host_dev = mummu::backend::cpu_device();
        let host = measure_device(
            &pack,
            &host_dev,
            true,
            &[Precision::Q4, Precision::Q8, Precision::F16],
        );
        let (accel, crossing_s) = if backend == BackendChoice::Cpu {
            (DeviceModel::default(), 0.0)
        } else {
            let dev = device_of(backend);
            // f32-widened float weights exceed wgpu's 256 MiB max buffer on
            // the big projections (1015 failed reservations, 2026-08); the
            // integer levels are the card's.
            let levels: &[Precision] = match backend {
                #[cfg(feature = "cuda")]
                BackendChoice::Cuda => &[Precision::Q4, Precision::Q8, Precision::F16],
                _ => &[Precision::Q4, Precision::Q8],
            };
            (
                measure_device(&pack, &dev, false, levels),
                measure_crossing(cfg.hidden_size, &dev),
            )
        };
        let model = pack_dir.parent().and_then(|p| p.file_name()).map_or_else(
            || pack_dir.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let n = cfg.num_layers;
        let show = |m: &DeviceModel| {
            m.rate
                .iter()
                .map(|(q, s)| format!("{q:?} {:.2} GB/s", 1.0 / s / 1e9))
                .collect::<Vec<_>>()
                .join(", ")
        };
        eprintln!(
            "[mummu-serve] placement rates: host [{}]; {} [{}]; crossing {:.3} ms",
            show(&host),
            label_of(backend),
            show(&accel),
            crossing_s * 1e3
        );
        Ok(Self {
            pack_dir: pack_dir.to_path_buf(),
            backend,
            model,
            cfg,
            maps,
            host,
            accel,
            crossing_s,
            assignment: joint::Assignment {
                layers: (0..n)
                    .map(|_| joint::Choice {
                        device: 0,
                        levels: Vec::new(),
                    })
                    .collect(),
            },
            improve_wanted: 0,
        })
    }

    fn layers_on_card(&self) -> usize {
        self.assignment
            .layers
            .iter()
            .filter(|c| c.device == 1)
            .count()
    }

    /// The solver's problem for serving `ctx` tokens, with `card` as the
    /// current reading (None: no accelerator, or no reading at all).
    fn problem(&self, ctx: usize, needs_tower: bool, card: Option<&Card>) -> joint::Problem {
        let layers: Vec<joint::Layer> = self
            .maps
            .iter()
            .enumerate()
            .map(|(l, m)| joint::Layer {
                parts: m.parts.clone(),
                fixed_bytes: m.fixed_bytes,
                state_bytes: state_bytes(&self.cfg, l, ctx),
            })
            .collect();
        // Host bytes our own host layers hold now: MemAvailable excludes
        // them, and they are ours to re-spend.
        let host_layers: u64 = self
            .assignment
            .layers
            .iter()
            .zip(&layers)
            .filter(|(c, _)| c.device == 0 && !c.levels.is_empty())
            .map(|(c, l)| {
                l.parts
                    .iter()
                    .zip(&c.levels)
                    .map(|(p, &q)| {
                        let b = p.levels.iter().find(|(x, _)| *x == q).map_or(0, |x| x.1);
                        let m = self
                            .host
                            .resident
                            .iter()
                            .find(|(x, _)| *x == q)
                            .map_or(1.0, |x| x.1);
                        (b as f64 * m) as u64
                    })
                    .sum::<u64>()
                    + l.fixed_bytes
            })
            .sum();
        let host_cap = super::mem_available_bytes()
            .map_or(u64::MAX / 4, |a| a / 100 * 85)
            .saturating_add(host_layers);
        let mut devices = vec![joint::Device {
            capacity: host_cap,
            fixed: act_bytes(&self.cfg, ctx),
            rate: self.host.rate.clone(),
            resident: self.host.resident.clone(),
        }];
        if let Some(c) = card
            && !self.accel.rate.is_empty()
        {
            let tower_pending = if needs_tower && !super::tower_resident() {
                super::VISION_RESERVE.load(std::sync::atomic::Ordering::SeqCst)
            } else {
                0
            };
            let ours_layers = self.planned_card_bytes();
            // N: what we hold on the card that is not a layer.
            let non_layer = c.in_use.saturating_sub(ours_layers);
            let k = capacity(c);
            devices.push(joint::Device {
                capacity: k.saturating_sub(non_layer),
                fixed: act_bytes(&self.cfg, ctx) + residual() + tower_pending,
                rate: self.accel.rate.clone(),
                resident: self.accel.resident.clone(),
            });
        }
        joint::Problem {
            layers,
            devices,
            crossing_s: self.crossing_s,
            disk_s_per_byte: disk_s_per_byte(),
            horizon_tokens: horizon_tokens(),
            tolerance: PRECISION_TOLERANCE,
            floor: QuantPolicy::Q4,
        }
    }

    /// The pack precision `entry` loads at under `choice`.
    fn precision_for(
        &self,
        layer: usize,
        choice: &joint::Choice,
        entry: &TensorEntry,
    ) -> Precision {
        let m = &self.maps[layer];
        for (pi, names) in m.names.iter().enumerate() {
            if names.iter().any(|n| n == &entry.name) {
                return precision_of(choice.levels[pi]);
            }
        }
        fixed_precision(entry)
    }

    /// Device and precision for every tensor, for the initial load.
    pub fn loader_choice(&self) -> impl Fn(usize) -> BackendChoice + '_ {
        move |l| {
            if self.assignment.layers.get(l).is_some_and(|c| c.device == 1) {
                self.backend
            } else {
                BackendChoice::Cpu
            }
        }
    }

    pub fn loader_precision(&self) -> impl Fn(&TensorEntry) -> Precision + '_ {
        move |e| match layer_index(&e.name) {
            Some(l) if l < self.assignment.layers.len() => {
                let c = &self.assignment.layers[l];
                if c.levels.is_empty() {
                    fixed_precision(e)
                } else {
                    self.precision_for(l, c, e)
                }
            }
            _ => fixed_precision(e),
        }
    }

    /// Weight bytes the placement puts on `device` (0 host, 1 card), as that
    /// device holds them.
    fn planned_bytes(&self, device: usize) -> u64 {
        let model = if device == 0 { &self.host } else { &self.accel };
        self.assignment
            .layers
            .iter()
            .zip(&self.maps)
            .filter(|(c, _)| c.device == device && !c.levels.is_empty())
            .map(|(c, m)| {
                m.parts
                    .iter()
                    .zip(&c.levels)
                    .map(|(p, &q)| {
                        let b = p.levels.iter().find(|(x, _)| *x == q).map_or(0, |x| x.1);
                        let r = model
                            .resident
                            .iter()
                            .find(|(x, _)| *x == q)
                            .map_or(1.0, |x| x.1);
                        (b as f64 * r) as u64
                    })
                    .sum::<u64>()
                    + m.fixed_bytes
            })
            .sum()
    }

    /// Card bytes the placement's layers occupy — the residency check's plan.
    pub fn planned_card_bytes(&self) -> u64 {
        self.planned_bytes(1)
    }

    /// Host bytes the placement's layers occupy.
    pub fn planned_host_bytes(&self) -> u64 {
        self.planned_bytes(0)
    }

    fn summary(&self, pb: &joint::Problem) -> String {
        let mut hist: std::collections::BTreeMap<(usize, String), usize> = Default::default();
        for c in &self.assignment.layers {
            for q in &c.levels {
                *hist.entry((c.device, format!("{q:?}"))).or_default() += 1;
            }
        }
        let parts: Vec<String> = hist
            .iter()
            .map(|((d, q), n)| {
                format!(
                    "{n} {q} on {}",
                    if *d == 0 {
                        "host"
                    } else {
                        label_of(self.backend)
                    }
                )
            })
            .collect();
        format!(
            "{}/{} layers on {}; parts {}; predicted {:.1} ms/token",
            self.layers_on_card(),
            self.assignment.layers.len(),
            label_of(self.backend),
            parts.join(", "),
            joint::time_of(pb, &self.assignment) * 1e3
        )
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Decide the initial placement for a load: the joint solve against what the
/// card and host hold right now, for the request that caused the load.
pub(super) fn plan_load(pack_dir: &Path, backend: BackendChoice) -> Result<Live, String> {
    if backend != BackendChoice::Cpu {
        // Whatever the previous model left in the pool goes back to the
        // driver before we read what is free.
        device_of(backend).memory_cleanup();
    }
    let mut live = Live::measure(pack_dir, backend)?;
    let (ctx, tower) = request().unwrap_or((idle_context(), false));
    let reading = card(backend);
    let pb = live.problem(ctx, tower, reading.as_ref());
    let out = joint::solve(&pb);
    if !out.feasible {
        eprintln!(
            "[mummu-serve] placement: nothing fits everywhere (host {:.1} GiB needed of {:.1}) — loading anyway at the fastest host levels",
            out.used[0] as f64 / f64::from(1u32 << 30),
            pb.devices[0].capacity as f64 / f64::from(1u32 << 30),
        );
    }
    live.assignment = out.assignment;
    if let Some(c) = reading {
        eprintln!(
            "[mummu-serve] placement: card {:.1} GiB, ambient {:.1}, guard {:.1}, capacity {:.1} GiB for a {ctx}-token context{}",
            c.total as f64 / f64::from(1u32 << 30),
            c.used.saturating_sub(c.reserved) as f64 / f64::from(1u32 << 30),
            c.total.saturating_sub(capacity(&c)) as f64 / f64::from(1u32 << 30),
            pb.devices.get(1).map_or(0, |d| d.capacity) as f64 / f64::from(1u32 << 30),
            if tower { " + vision tower" } else { "" },
        );
    }
    eprintln!("[mummu-serve] placement: {}", live.summary(&pb));
    Ok(live)
}

// ---------------------------------------------------------------------------
// Moving layers
// ---------------------------------------------------------------------------

/// Apply `target` to the resident model, releases first. `limit` bounds the
/// layers moved (an improvement's step); a repair passes `None`.
///
/// # Errors
/// A pack read failure. A device failure panics out of the relocation, and
/// the caller decides it like any other device failure.
fn apply(
    live: &mut Live,
    lm: &mut qwen35::LoadedQwen35,
    target: &joint::Assignment,
    limit: Option<usize>,
) -> Result<usize, String> {
    let backend = live.backend;
    let changed: Vec<usize> = (0..target.layers.len())
        .filter(|&l| live.assignment.layers[l] != target.layers[l])
        .collect();
    if changed.is_empty() {
        return Ok(0);
    }
    // Releases: leaving the card, or shrinking on it. Everything else is an
    // arrival or a host-side change.
    let pb = live.problem(idle_context(), false, None);
    let card_bytes = |l: usize, c: &joint::Choice| -> u64 {
        if c.device != 1 {
            return 0;
        }
        let mut a = live.assignment.clone();
        for (i, x) in a.layers.iter_mut().enumerate() {
            if i != l {
                x.device = 0;
            }
        }
        a.layers[l] = c.clone();
        let mut p = pb.clone();
        if p.devices.len() < 2 {
            p.devices.push(joint::Device {
                capacity: u64::MAX,
                fixed: 0,
                rate: live.accel.rate.clone(),
                resident: live.accel.resident.clone(),
            });
        }
        joint::used_of(&p, &a).get(1).copied().unwrap_or(0)
    };
    let (mut releases, mut arrivals): (Vec<usize>, Vec<usize>) = changed.iter().partition(|&&l| {
        card_bytes(l, &target.layers[l]) <= card_bytes(l, &live.assignment.layers[l])
    });
    releases.sort_unstable();
    arrivals.sort_unstable();
    let budget = limit.unwrap_or(usize::MAX);
    let order: Vec<usize> = releases.into_iter().chain(arrivals).take(budget).collect();

    let started = Instant::now();
    let mut bytes = 0u64;
    let mut host_touched = false;
    let mut pages_to_return = false;
    for &l in &order {
        let from = live.assignment.layers[l].device;
        let to = target.layers[l].clone();
        // Pages freed by the releases go back to the driver before the first
        // arrival allocates — so the arrival sees them, and so would a
        // co-tenant if the arrival never comes.
        if to.device == 1 && pages_to_return && backend != BackendChoice::Cpu {
            device_of(backend).memory_cleanup();
            pages_to_return = false;
        }
        let dev = if to.device == 1 {
            device_of(backend)
        } else {
            mummu::backend::cpu_device()
        };
        let choose = |e: &TensorEntry| live.precision_for(l, &to, e);
        bytes += qwen35::relocate_layers(lm, &live.pack_dir, &[l], &dev, &choose)
            .map_err(|e| e.to_string())?;
        host_touched |= from == 0 || to.device == 0;
        pages_to_return |= from == 1;
        live.assignment.layers[l] = to;
    }
    if backend != BackendChoice::Cpu {
        device_of(backend).memory_cleanup();
        device_of(backend).sync().map_err(|e| e.to_string())?;
    }
    if host_touched {
        // Twins are keyed by slab address; a moved layer's old slabs are
        // gone, and a fresh twin set is cheaper than tracking which.
        mummu::flex::registry::clear();
        super::warm_host_twins(lm, 0);
    }
    note_disk(bytes, started.elapsed().as_secs_f64());
    Ok(order.len())
}

/// What a re-plan decided, for the log.
fn verdict_word(v: joint::Verdict) -> &'static str {
    match v {
        joint::Verdict::Hold => "hold",
        joint::Verdict::Repair => "repair",
        joint::Verdict::Improve => "improve",
    }
}

fn replan_and_apply(
    live: &mut Live,
    lm: &mut qwen35::LoadedQwen35,
    ctx: usize,
    needs_tower: bool,
    idle: bool,
) -> Result<(), String> {
    let reading = card(live.backend);
    if live.backend != BackendChoice::Cpu && reading.is_none() {
        // No reading is no information: hold. Assuming room risks an OOM,
        // assuming pressure demotes a model that was running fine.
        return Ok(());
    }
    let pb = live.problem(ctx, needs_tower, reading.as_ref());
    if pb.devices.len() < 2 && live.layers_on_card() > 0 {
        return Ok(());
    }
    let (verdict, out) = joint::replan(&pb, &live.assignment);
    let limit = match verdict {
        joint::Verdict::Hold => {
            live.improve_wanted = 0;
            return Ok(());
        }
        joint::Verdict::Repair => None,
        joint::Verdict::Improve => {
            // Only at idle, and only once it has been wanted for a while.
            if !idle {
                return Ok(());
            }
            live.improve_wanted += 1;
            if live.improve_wanted < IMPROVE_DWELL {
                return Ok(());
            }
            Some(IMPROVE_STEP)
        }
    };
    let before = joint::time_of(&pb, &live.assignment);
    let started = Instant::now();
    let moved = apply(live, lm, &out.assignment, limit)?;
    if live.assignment == out.assignment {
        live.improve_wanted = 0;
    }
    eprintln!(
        "[mummu-serve] placement {}: moved {moved} layer(s) in {:.1}s, {:.1} -> {:.1} ms/token predicted; {}",
        verdict_word(verdict),
        started.elapsed().as_secs_f64(),
        before * 1e3,
        joint::time_of(&pb, &live.assignment) * 1e3,
        live.summary(&pb),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/// Before a generation, holding the model: make room for THIS request (its
/// context, its image) if the placement would not fit beside it. Returns the
/// pool's in-use bytes, the baseline [`after_request`] measures from.
pub(super) fn before_request(
    m: &mut Loaded,
    key: &Path,
    ctx: usize,
    needs_tower: bool,
) -> Option<u64> {
    CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(ctx as u64);
    set_request(ctx, needs_tower);
    let AnyLm::Qwen35(lm) = &mut m.lm else {
        return None;
    };
    let mut guard = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    let live = guard.as_mut().filter(|l| l.pack_dir.starts_with(key))?;
    if let Err(e) = replan_and_apply(live, lm, ctx, needs_tower, false) {
        eprintln!("[mummu-serve] placement: could not re-place before the request: {e}");
    }
    (live.backend != BackendChoice::Cpu).then(|| pool(live.backend).1)
}

/// After a generation, holding the model: measure the working set it
/// actually used (ε̂) and count its tokens toward the horizon.
pub(super) fn after_request(ctx: usize, tokens: usize, in_use_before: Option<u64>) {
    SERVED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((Instant::now(), tokens));
    let Some(before) = in_use_before else { return };
    let guard = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(live) = guard.as_ref() else { return };
    let (reserved, _) = pool(live.backend);
    let state: u64 = live
        .assignment
        .layers
        .iter()
        .enumerate()
        .filter(|(_, c)| c.device == 1)
        .map(|(l, _)| state_bytes(&live.cfg, l, ctx))
        .sum();
    let residual = reserved
        .saturating_sub(before)
        .saturating_sub(state)
        .saturating_sub(act_bytes(&live.cfg, ctx));
    RESIDUAL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(residual);
}

/// Make `live` the placement of the model now resident.
pub(super) fn adopt(live: Live) {
    *LIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(live);
}

/// Forget the placement (the model it described left the slot).
pub(super) fn forget(pack_dir: Option<&Path>) {
    let mut g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if pack_dir.is_none_or(|p| g.as_ref().is_some_and(|l| l.pack_dir == p)) {
        *g = None;
    }
}

/// One idle look: feed the guard, drop an idle tower, and re-plan if the
/// model is free.
fn tick() {
    // Idle tower: its VRAM goes back to layers.
    let idle_tower = TOWER_USED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some_and(|t| t.elapsed() > TOWER_IDLE);
    let mut failure: Option<(String, BackendChoice, String)> = None;
    let _ = SLOT.try_with_mut(|key, m: &mut Loaded| {
        if idle_tower && super::drop_tower() {
            *TOWER_USED.lock().unwrap_or_else(|e| e.into_inner()) = None;
            device_of(m.backend).memory_cleanup();
            eprintln!(
                "[mummu-serve] placement: vision tower idle for {}s — its VRAM goes back to layers",
                TOWER_IDLE.as_secs()
            );
        }
        let AnyLm::Qwen35(lm) = &mut m.lm else { return };
        let mut guard = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(live) = guard.as_mut().filter(|l| l.pack_dir.starts_with(key)) else {
            return;
        };
        let ctx = idle_context();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            replan_and_apply(live, lm, ctx, false, true)
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[mummu-serve] placement: idle re-plan failed: {e}");
            }
            Err(p) => {
                failure = Some((
                    live.model.clone(),
                    live.backend,
                    crate::recovery::payload_text(&*p),
                ));
            }
        }
    });
    // A panic mid-move leaves a layer half on each device: the model is not
    // servable. Decide it as a device failure and drop the model, so the next
    // request reloads it whole.
    if let Some((model, backend, message)) = failure {
        eprintln!(
            "[mummu-serve] placement: a move failed mid-layer — dropping the model: {message}"
        );
        let _ = crate::recovery::record_failure(
            &model,
            &[super::device_key(backend)],
            &crate::recovery::summarize(&message),
        );
        forget(None);
        if SLOT.clear() {
            mummu::progress::evicted();
        }
    }
}

/// Start the idle rebalancer. One per process.
pub(super) fn spawn_watch() {
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    eprintln!(
        "[mummu-serve] placement watch: every {}s — layers and their precisions follow the card",
        TICK.as_secs()
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(TICK);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let _ = tokio::task::spawn_blocking(tick).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_27b() -> qwen35::Qwen35Config {
        super::super::cfg_27b_for_tests()
    }

    /// KV grows with context on attention layers; recurrent state does not.
    #[test]
    fn state_follows_context_only_where_it_should() {
        let cfg = cfg_27b();
        let attn = (0..cfg.num_layers).find(|&l| cfg.is_attention(l)).unwrap();
        let delta = (0..cfg.num_layers).find(|&l| !cfg.is_attention(l)).unwrap();
        assert!(state_bytes(&cfg, attn, 8192) > state_bytes(&cfg, attn, 1024));
        assert_eq!(
            state_bytes(&cfg, delta, 8192),
            state_bytes(&cfg, delta, 1024)
        );
        assert!(state_bytes(&cfg, delta, 1) > 0);
    }

    /// Prefill is chunked, so the activation term stops growing at the chunk.
    #[test]
    fn activations_are_bounded_by_the_prefill_chunk() {
        let cfg = cfg_27b();
        let chunk = mummu::decode::prefill_chunk_len();
        assert_eq!(act_bytes(&cfg, chunk), act_bytes(&cfg, chunk * 8));
        assert!(act_bytes(&cfg, 16) < act_bytes(&cfg, chunk));
    }

    /// Capacity is the card less the guard on AMBIENT — our own reserved
    /// bytes are not ambient, so holding more of the card ourselves must not
    /// shrink what we are allowed to hold.
    #[test]
    fn our_own_bytes_are_not_ambient() {
        let gib = 1u64 << 30;
        let quiet = Card {
            total: 16 * gib,
            used: 3 * gib,
            reserved: 0,
            in_use: 0,
        };
        let busy_with_us = Card {
            total: 16 * gib,
            used: 12 * gib,
            reserved: 9 * gib,
            in_use: 8 * gib,
        };
        let a = capacity(&quiet);
        let b = capacity(&busy_with_us);
        assert_eq!(a, b, "same ambient, same capacity");
        assert!(a < 16 * gib - 3 * gib + 1);
    }

    /// Envelopes keep the max of the recent window and forget the old.
    #[test]
    fn envelopes_forget_what_left_the_window() {
        let mut e = Envelope::default();
        e.push(100);
        for _ in 0..ENVELOPE {
            e.push(1);
        }
        assert_eq!(e.max(), Some(1));
    }
}
