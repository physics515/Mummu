//! Debug probe: load a qwen35 pack on the CPU and print the activations a
//! llama.cpp `eval-callback` run prints (embedding after the Hadamard
//! inverse, the residual after each layer, the first-token top-5), so a
//! folded checkpoint can be compared point by point against the reference.
//!
//! ```text
//! MUMMU_LAYER_TRACE=1 cargo run --release -p mummu --example bonsai-probe -- <pack dir> [prompt]
//! ```

use mummu::models::CausalLm;
use mummu::models::qwen35;
use mummu::pack::Precision;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(
        args.next()
            .expect("usage: bonsai-probe <pack dir> [prompt]"),
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "List the first five prime numbers.".into());
    let pack = mummu::pack::Pack::open(&dir).expect("pack opens");
    let header = pack.header().expect("header");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&header).expect("tokenizer");
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(prompt)]);
    let ids: Vec<u32> = match std::env::var("PROBE_IDS") {
        Ok(list) => list
            .split(',')
            .map(|s| s.trim().parse().expect("id"))
            .collect(),
        Err(_) => tok
            .encode(rendered.as_str(), false)
            .expect("encodes")
            .get_ids()
            .to_vec(),
    };
    eprintln!("prompt ids ({}): {ids:?}", ids.len());
    let device = mummu::backend::cpu_device();
    let loaded = qwen35::load_from_pack(&dir, &device, &|_| match std::env::var("PROBE_LEVEL")
        .as_deref()
    {
        Ok("q8") => Precision::Q8,
        _ => Precision::Q4,
    })
    .expect("loads");
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
        .clone()
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
            tok.decode(&[i as u32], false).unwrap_or_default()
        );
    }
}
