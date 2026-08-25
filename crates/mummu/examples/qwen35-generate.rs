//! Generate from any qwen35 GGUF through the streaming quantized importer —
//! the P9 demonstration vehicle (the 27B tier only exists through it):
//!
//! ```text
//! cargo run --release -p mummu --example qwen35-generate -- \
//!     <path.gguf> <off|q8|q4> [max_tokens] [prompt…]
//! ```
//!
//! Runs on the default device policy (GPU when an adapter is present;
//! `MUMMU_FORCE_CPU=1` forces flex). Prints load time, resident policy, the
//! streamed text, and tokens/s.

use std::io::Write as _;
use std::time::Instant;

use mummu::models::CausalLm;
use mummu::models::qwen35;
use mummu::quant::QuantPolicy;

/// The default GPU policy, overridable by `MUMMU_FORCE_CPU=1` — a 27B's
/// float side alone (5 GB embedding) plus quantized linears outgrows a
/// 16 GB card, so the big tiers need the CPU explicitly.
fn use_gpu() -> bool {
    let forced_cpu =
        std::env::var("MUMMU_FORCE_CPU").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    !forced_cpu && mummu::backend::use_gpu()
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(
        args.next()
            .expect("usage: qwen35-generate <gguf> <off|q8|q4> [max_tokens] [prompt…]"),
    );
    let policy = match args.next().as_deref() {
        Some("off") | None => QuantPolicy::Off,
        Some("q8") => QuantPolicy::Q8,
        Some("q4") => QuantPolicy::Q4,
        Some(other) => panic!("unknown quant policy {other:?} (off|q8|q4)"),
    };
    let max_tokens: usize = args
        .next()
        .map_or(64, |s| s.parse().expect("max_tokens must be a number"));
    let prompt_text = {
        let rest: Vec<String> = args.collect();
        if rest.is_empty() {
            "What is 2+2? Answer in one short sentence.".to_string()
        } else {
            rest.join(" ")
        }
    };

    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);
    let rendered =
        mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(prompt_text.clone())]);
    let ids = tok
        .encode(rendered.as_str(), false)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();

    eprintln!(
        "[qwen35-generate] {policy:?} | {} | prompt {} tokens",
        if use_gpu() {
            "GPU (wgpu)"
        } else {
            "CPU (flex)"
        },
        ids.len()
    );

    // The device is a runtime value under burn 0.22, so the two arms differ by
    // which device they pass rather than by a backend type parameter.
    let device = if use_gpu() {
        mummu::backend::gpu_device()
    } else {
        mummu::backend::cpu_device()
    };
    run(&path, policy, &ids, max_tokens, &tok, &device).await;
}

async fn run(
    path: &std::path::Path,
    policy: QuantPolicy,
    ids: &[u32],
    max_tokens: usize,
    tok: &tokenizers::Tokenizer,
    device: &burn::tensor::Device,
) {
    let t0 = Instant::now();
    let loaded = qwen35::load_from_gguf_quantized(path, device, policy).expect("model loads");
    eprintln!(
        "[qwen35-generate] loaded in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let t1 = Instant::now();
    let mut emitted = String::new();
    let mut out_ids: Vec<u32> = Vec::new();
    // The options must outlive the future, so own them here rather than
    // passing a temporary that dies at the end of this statement.
    let opts = mummu::decode::SamplerOptions::greedy();
    let result = loaded.generate(ids, max_tokens, &opts, device, |id| {
        out_ids.push(id);
        if let Ok(text) = tok.decode(&out_ids, true)
            && !text.ends_with('\u{FFFD}')
            && text.len() > emitted.len()
        {
            print!("{}", &text[emitted.len()..]);
            let _ = std::io::stdout().flush();
            emitted = text;
        }
        std::ops::ControlFlow::Continue(())
    });
    println!();
    let n = result.await.expect("decode").len();
    let secs = t1.elapsed().as_secs_f32();
    eprintln!(
        "[qwen35-generate] {n} tokens in {secs:.1}s ({:.2} tok/s)",
        n as f32 / secs.max(1e-3)
    );
}
