//! Parse the REAL Qwen3.8-Flash-Next header — the split-GGUF and config
//! paths against the shipped file rather than a synthetic one.
//!
//! ```text
//! MUMMU_QWEN4EXP_DIR=/mnt/deepmem/AI\ Models/qwen3.8-flash-next \
//!   cargo test -p mummu --test real_qwen4exp -- --ignored --nocapture
//! ```
//!
//! The directory must hold the four `...-0000N-of-00004.gguf` shards. Unlike
//! the older real-model gates, a missing env var here SKIPS with a message
//! instead of panicking — an absent 111 GB fixture is a "not run", not a
//! failure (ROADMAP has this as a standing complaint about the other gates).

use mummu::gguf::GgufFile;
use mummu::models::qwen4exp::Qwen4expConfig;

/// First shard of the split set, when the fixture is present.
fn first_shard() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from(std::env::var_os("MUMMU_QWEN4EXP_DIR")?);
    let p = dir.join("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf");
    p.is_file().then_some(p)
}

#[test]
#[ignore = "needs the 111 GB Flash-Next split set (MUMMU_QWEN4EXP_DIR)"]
fn the_shipped_split_set_opens_as_one_namespace() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let f = GgufFile::open_sharded(&first).expect("split set opens");
    assert_eq!(f.shard_count(), 4, "four shards joined");
    assert_eq!(f.architecture(), Some("qwen4exp"));
    // The header declares 1224 tensors across the set; shard 1 holds none of
    // them, which is precisely why the merge has to exist.
    assert_eq!(f.tensors.len(), 1224, "every shard's tensors are visible");
    assert!(
        f.tensor("token_embd.weight").is_some(),
        "an embedding from a later shard resolves"
    );
}

#[test]
#[ignore = "needs the 111 GB Flash-Next split set (MUMMU_QWEN4EXP_DIR)"]
fn the_shipped_header_parses_to_the_expected_config() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let f = GgufFile::open_sharded(&first).expect("split set opens");
    let c = Qwen4expConfig::from_gguf(&f).expect("qwen4exp config parses");

    // Values observed in the shipped UD-Q4_K_XL header. These are the shape
    // the rest of the port must satisfy, so drift should fail loudly here.
    assert_eq!(c.num_layers, 48);
    assert_eq!(c.hidden_size, 2560);
    assert_eq!(c.num_attention_heads, 24);
    assert_eq!(c.num_key_value_heads, 2);
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.rope_dim, 64);
    assert_eq!(c.full_attention_interval, 4);
    assert_eq!(c.attention_layers(), 12, "12 attention, 36 DeltaNet");
    // DeltaNet at qwen35's shape — the reason that port is reusable here.
    assert_eq!(c.conv_kernel, 4);
    assert_eq!(c.d_state, 128);
    assert_eq!(c.n_k_heads, 16);
    assert_eq!(c.d_inner, 6144);
    // MoE in every layer.
    assert_eq!(c.expert_count, 512);
    assert_eq!(c.expert_used_count, 10);
    assert_eq!(c.expert_ffn_size, 640);
    assert_eq!(c.expert_shared_ffn_size, 640);
    // The three mechanisms with no equivalent anywhere in the zoo yet.
    assert_eq!(c.hyper_connection_count, 4);
    assert_eq!(c.hyper_connection_low_rank, 320);
    assert_eq!(c.indexer_top_k, 2048);
    assert_eq!(c.ple_layers, vec![1]);
    assert_eq!(c.ple_ngram_size, 3);
    assert_eq!(c.ple_heads_per_ngram, 8);
    assert_eq!(c.ple_row_width, 160);
    assert_eq!(
        c.ple_layer_multipliers.len(),
        c.ple_ngram_size,
        "one hash multiplier per n-gram position"
    );
    assert_eq!(
        c.ple_head_offsets.len(),
        8,
        "eight head vocabularies, per the header"
    );
    println!(
        "qwen4exp: {} layers ({} attn), {} experts top-{}, PLE {} rows x {} wide",
        c.num_layers,
        c.attention_layers(),
        c.expert_count,
        c.expert_used_count,
        c.ple_total_rows(),
        c.ple_row_width
    );
}
