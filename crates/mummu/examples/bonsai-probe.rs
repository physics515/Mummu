//! Debug probe: load a qwen35 pack on the CPU and print the activations a
//! llama.cpp `eval-callback` run prints (embedding after the Hadamard
//! inverse, the residual after each layer, the first-token top-5), so a
//! folded checkpoint can be compared point by point against the reference.
//!
//! ```text
//! MUMMU_LAYER_TRACE=1 cargo run --release -p mummu --example bonsai-probe -- <pack dir> [prompt]
//! ```
//!
//! `PROBE_LEVEL` = `q2` (the native ternary level), `q8`, or `q4` (default).
//! `PROBE_ADD_Q2=<source gguf>` first adds the native level to the pack in
//! place (`pack::add_ternary_level`; `PROBE_ONLY_ADD=1` stops there). With
//! the `cuda` feature,
//! `PROBE_CUDA=1` puts every layer and the head on the card (the embedding
//! stays on the host) — the production path for a fully resident model.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use mummu::models::CausalLm;
use mummu::models::qwen35;
use mummu::pack::Precision;

/// The pack at the `PROBE_LEVEL` level, on the host — or, with
/// `PROBE_CUDA` (cuda builds), every layer and the head on the card with the
/// embedding on the host.
fn load(dir: &std::path::Path, host: &burn::tensor::Device) -> qwen35::LoadedQwen35 {
    let level = match std::env::var("PROBE_LEVEL").as_deref() {
        Ok("q8") => Precision::Q8,
        Ok("q2") => Precision::Q2,
        _ => Precision::Q4,
    };
    eprintln!("level: {level:?}");
    let choose = move |_: &mummu::pack::TensorEntry| level;
    #[cfg(feature = "cuda")]
    if std::env::var("PROBE_CUDA").is_ok() {
        let card = mummu::backend::cuda_device();
        eprintln!("layers + head on {card:?}, embedding on the host");
        return qwen35::load_from_pack_layered(dir, &|_| card.clone(), host, &card, &choose)
            .expect("loads");
    }
    qwen35::load_from_pack(dir, host, &choose).expect("loads")
}

/// `PROBE_REPEAT=n`: the prompt n times over — a long prefill for the
/// chunked memory probe.
fn repeated(prompt: String) -> String {
    match std::env::var("PROBE_REPEAT")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
    {
        Some(n) => vec![prompt.as_str(); n].join(" "),
        None => prompt,
    }
}

/// `PROBE_ADD_Q2=<gguf>`: add the native ternary level first. True when
/// `PROBE_ONLY_ADD` asks to stop there.
fn add_level(dir: &std::path::Path) -> bool {
    let Ok(gguf) = std::env::var("PROBE_ADD_Q2") else {
        return false;
    };
    let added = mummu::pack::add_ternary_level(dir, std::path::Path::new(&gguf), |i, n, name| {
        eprintln!("q2 level {i}/{n} {name}");
    })
    .expect("adds the native ternary level");
    eprintln!("native ternary level added to {added} tensors");
    std::env::var("PROBE_ONLY_ADD").is_ok()
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(
        args.next()
            .expect("usage: bonsai-probe <pack dir> [prompt]"),
    );
    let prompt = repeated(
        args.next()
            .unwrap_or_else(|| "List the first five prime numbers.".into()),
    );
    if add_level(&dir) {
        return;
    }
    let pack = mummu::pack::Pack::open(&dir).expect("pack opens");
    let header = pack.header().expect("header");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&header).expect("tokenizer");
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(prompt)]);
    let ids: Vec<u32> = std::env::var("PROBE_IDS").map_or_else(
        |_| {
            tok.encode(rendered.as_str(), false)
                .expect("encodes")
                .get_ids()
                .to_vec()
        },
        |list| {
            list.split(',')
                .map(|s| s.trim().parse().expect("id"))
                .collect()
        },
    );
    eprintln!("prompt ids ({}): {ids:?}", ids.len());
    let device = mummu::backend::cpu_device();
    let loaded = load(&dir, &device);
    eprintln!(
        "hadamard: {:?}",
        loaded.config.hadamard.as_ref().map(|h| (
            h.head,
            h.embed_inverse,
            h.layers[0],
            h.layers[3]
        ))
    );

    let stats = |name: &str, v: &[f32]| {
        let sum: f64 = v.iter().map(|&x| f64::from(x)).sum();
        let sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        eprintln!(
            "{name}: n={} sum={sum:.6} l2={:.6} first={:?}",
            v.len(),
            sq.sqrt(),
            &v[..v.len().min(6)]
        );
    };
    // The stored table row and the row after the inverse, for the first id.
    let e = loaded.model.embed_tokens.weight.val();
    let hidden = loaded.config.hidden_size;
    let row = e
        .narrow(0, ids[0] as usize, 1)
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    stats(&format!("stored row {}", ids[0]), &row);
    let emb = loaded
        .embed(&ids, &device)
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    stats("inp_embd (all tokens)", &emb);
    stats(&format!("inp_embd row0 (token {})", ids[0]), &emb[..hidden]);
    if let Ok(chunk) = std::env::var("PROBE_CHUNK") {
        prefill_in_chunks(&loaded, &ids, chunk.parse().expect("chunk length"), &device);
        return;
    }
    let mut cache = loaded.new_cache();
    let logits = loaded
        .forward(&ids, 0, &mut cache, &device)
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    stats("logits", &logits);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let max = logits[idx[0]];
    let lse: f32 = logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln() + max;
    for &i in &idx[..5] {
        eprintln!(
            "top: id={i} logit={:.4} logprob={:.4} tok={:?}",
            logits[i],
            logits[i] - lse,
            tok.decode(&[u32::try_from(i).expect("vocab index fits u32")], false)
                .unwrap_or_default()
        );
    }
}

/// `PROBE_CHUNK=n`: prefill `ids` in spans of `n`, as the server does, and
/// report the card's pool after each (with `MUMMU_POOL_TRACE=1` the model
/// also reports per-layer peaks inside each span).
fn prefill_in_chunks(
    loaded: &qwen35::LoadedQwen35,
    ids: &[u32],
    chunk: usize,
    device: &burn::tensor::Device,
) {
    #[cfg(feature = "cuda")]
    let card = Some(mummu::backend::cuda_device());
    #[cfg(not(feature = "cuda"))]
    let card: Option<burn::tensor::Device> = None;
    let pool = || {
        card.as_ref()
            .and_then(burn::tensor::Device::memory_pool_usage)
            .map_or((0, 0), |u| (u.bytes_reserved >> 20, u.bytes_in_use >> 20))
    };
    eprintln!(
        "prefill {} ids in spans of {chunk}; pool before (MiB reserved, in use): {:?}",
        ids.len(),
        pool()
    );
    let mut cache = loaded.new_cache();
    for (i, span) in ids.chunks(chunk).enumerate() {
        let started = std::time::Instant::now();
        let logits = loaded.forward(span, i * chunk, &mut cache, device);
        let _ = logits.into_data();
        eprintln!(
            "span {i} ({} ids) in {:.1}s; pool after: {:?}",
            span.len(),
            started.elapsed().as_secs_f64(),
            pool()
        );
    }
}
