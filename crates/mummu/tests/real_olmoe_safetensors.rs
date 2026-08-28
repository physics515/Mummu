//! REAL-WEIGHTS proof for the OLMoE **HF safetensors** import path (P2).
//!
//! The GGUF path ships the 64 experts already fused (`ffn_*_exps`); the HF
//! checkpoint stores each expert separately across three shards. This suite
//! proves the fusing loader lands on the same model:
//!
//!   1. the sharded checkpoint fetches through the registry spec;
//!   2. `load_from_dir` fuses 16 layers x 64 experts x 3 projections and
//!      checked-loads with zero missing params;
//!   3. it decodes coherently on the CPU backend;
//!   4. **the fused expert bank is bit-exact against the same expert read
//!      independently out of the raw shard bytes** — the check that a silent
//!      mis-ordering (expert 10 in slot 2) could not survive.
//!
//! Ignored by default (13.8 GB of weights, ~28 GB of RAM once loaded). Run:
//!
//! ```text
//! MUMMU_HUB_DEST=C:\Users\me\.cache\mummu-models \
//!   cargo test -p mummu --test real_olmoe_safetensors -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::models::{CausalLm, olmoe};
use mummu::registry;
use mummu::safetensors::{SafetensorsHeader, checkpoint_shards};

type Dev = burn::tensor::Device;

const SPEC: &str = "olmoe-1b-7b-0125-instruct";

/// Where the checkpoint lives, fetching it through the registry if needed.
fn checkpoint_dir() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("MUMMU_HUB_DEST")?);
    let spec = registry::catalog()
        .into_iter()
        .find(|s| s.name == SPEC)
        .expect("the catalog carries the OLMoE safetensors spec");
    let dir = spec.dir(&root);
    if dir.join("config.json").is_file() && checkpoint_shards(&dir).is_ok() {
        return Some(dir);
    }
    eprintln!("fetching {SPEC} (~13.8 GB) into {}", dir.display());
    spec.fetch(&root, &mut |p: mummu::hub::Progress| {
        if let Some(total) = p.total_bytes.filter(|t| *t > 0) {
            eprint!("\r  {} {:>3} %", p.file, 100 * p.received_bytes / total);
        }
    })
    .expect("registry fetch of the sharded checkpoint succeeds");
    eprintln!();
    Some(dir)
}

/// The whole path: sharded fetch -> fuse -> checked load -> real decode.
#[tokio::test]
#[ignore = "needs 13.8 GB of weights + ~28 GB RAM (MUMMU_HUB_DEST)"]
async fn olmoe_safetensors_fuses_and_decodes_on_cpu() {
    let Some(dir) = checkpoint_dir() else {
        eprintln!("set MUMMU_HUB_DEST to run this test");
        return;
    };
    let shards = checkpoint_shards(&dir).expect("shards discovered");
    println!("checkpoint: {} shard(s)", shards.len());
    assert!(
        shards.len() > 1,
        "the 1B-7B ships sharded — this exercises the multi-shard path"
    );

    let device = Dev::default();
    let started = std::time::Instant::now();
    let loaded = olmoe::load_from_dir(&dir, &device).expect("fused safetensors load");
    println!(
        "loaded {} layers x {} experts in {:.1} s",
        loaded.config.num_hidden_layers,
        loaded.config.num_experts,
        started.elapsed().as_secs_f32()
    );
    assert_eq!(loaded.config.num_experts, 64);
    assert_eq!(loaded.config.num_experts_per_tok, 8);
    // The sibling tokenizer_config.json is surfaced when the dir has one, and
    // is legitimately absent otherwise (a registry fetch pulls config.json +
    // tokenizer.json + weights only) — the loader must reflect the dir, not
    // invent a config.
    assert_eq!(
        loaded.tokenizer_config.is_some(),
        dir.join("tokenizer_config.json").is_file(),
        "surfaced tokenizer_config must match what the dir actually ships"
    );

    // Liveness: a real forward, checked for finiteness / width / spread.
    let probe: Vec<u32> = vec![100, 200, 300, 400];
    let smoke = loaded
        .sanity_check(&probe, loaded.config.vocab_size, &device)
        .await
        .expect("a fused load computes a live distribution");
    println!("sanity: top {} spread {:.1}", smoke.top_id, smoke.spread);

    let out = loaded
        .greedy_generate(&probe, 8, &device)
        .await
        .expect("greedy decode runs");
    println!("decoded {} tokens: {out:?}", out.len());
    assert!(!out.is_empty(), "the model emits tokens");
}

/// The ordering proof. Read expert 37's `gate_proj` for layer 5 straight out
/// of the raw shard bytes, then read slot 37 of the fused bank the loader
/// built, and require them **bit-identical**. A mis-ordered fuse (the
/// lexicographic trap) puts a different expert in that slot and fails here.
#[test]
#[ignore = "needs 13.8 GB of weights (MUMMU_HUB_DEST)"]
fn fused_expert_slot_is_bit_exact_against_the_raw_shard_bytes() {
    let Some(dir) = checkpoint_dir() else {
        eprintln!("set MUMMU_HUB_DEST to run this test");
        return;
    };
    const LAYER: usize = 5;
    const EXPERT: usize = 37;
    let source_name = format!("model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight");

    // Independent read: find the tensor in whichever shard holds it and pull
    // its raw bytes, using nothing the loader used.
    let mut truth: Option<(Vec<u64>, Vec<u8>)> = None;
    for shard in checkpoint_shards(&dir).expect("shards") {
        let header = SafetensorsHeader::open(&shard).expect("shard header parses");
        let Some((_, entry)) = header.tensors.iter().find(|(n, _)| *n == source_name) else {
            continue;
        };
        let mut file = std::fs::File::open(&shard).expect("shard opens");
        use std::io::{Read, Seek, SeekFrom};
        file.seek(SeekFrom::Start(header.data_offset + entry.offsets.0))
            .expect("seek");
        let mut bytes = vec![0u8; entry.byte_len() as usize];
        file.read_exact(&mut bytes).expect("read");
        truth = Some((entry.shape.clone(), bytes));
        break;
    }
    let (shape, truth_bytes) = truth.expect("the checkpoint carries that expert");
    println!(
        "raw {source_name}: shape {shape:?}, {} B",
        truth_bytes.len()
    );

    // What the loader produced, read back out of the loaded module.
    let device = Dev::default();
    let loaded = olmoe::load_from_dir(&dir, &device).expect("fused safetensors load");
    let bank = loaded.model.layers[LAYER].mlp.experts.gate.val();
    let dims = bank.dims();
    println!("fused bank dims {dims:?}");
    assert_eq!(dims[0], loaded.config.num_experts, "leading expert axis");
    assert_eq!(
        [dims[1] as u64, dims[2] as u64],
        [shape[0], shape[1]],
        "each slot keeps the source [out, in] shape"
    );

    let slot: Vec<f32> = bank
        .slice(EXPERT..EXPERT + 1)
        .into_data()
        .try_into_vec()
        .expect("readback");
    // The checkpoint is bf16; the loader cast it to the backend float (f32).
    // Compare against the same cast applied to the truth bytes: bf16 -> f32
    // is exact (it is a 16-bit truncation of f32), so this is bit-equality.
    let (halves, rest) = truth_bytes.as_chunks::<2>();
    assert!(rest.is_empty(), "bf16 payload is a whole number of halves");
    let expected: Vec<f32> = halves
        .iter()
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
        .collect();
    assert_eq!(slot.len(), expected.len(), "same element count");
    let mismatches = slot
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "expert {EXPERT} slot: {} values, {mismatches} bit-mismatches",
        slot.len()
    );
    assert_eq!(
        mismatches, 0,
        "the fused slot must be bit-exact against the raw shard bytes"
    );
}
