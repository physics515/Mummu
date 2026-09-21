//! How small do a qwen35 checkpoint's DeltaNet q/k head norms get, and how
//! far does the q/k L2 form ([`GdnL2`]) move its logits?
//!
//! ```text
//! cargo run --release -p mummu --example gdn-l2-probe -- \
//!     <path.gguf> <off|q8|q4> [greedy_tokens]
//! ```
//!
//! For each built-in prompt: one prefill under `ClampNorm` with the
//! `gdn_l2_probe` tap on, one under `AddEps`, then the logits compared
//! (top-5 ids, max |Δlogprob| over both top-5s, KL). The per-DeltaNet-layer
//! table pools every prompt's head norms and counts heads where the forms
//! differ by ≥ 1e-3 / ≥ 1e-2 relative (`1 − max(‖x‖,ε)/sqrt(‖x‖²+ε)`).
//! `greedy_tokens > 0` also greedy-decodes each prompt under both forms.
//!
//! Runs on flex unless `PROBE_DEVICE=gpu`. Set `MUMMU_GDN_CHUNK=off` for
//! prompts past ~256 tokens on a tree without the chunked-prefill NaN fix.

use std::time::Instant;

use mummu::chat::Turn;
use mummu::models::CausalLm;
use mummu::models::qwen35::{self, GdnL2, gdn_l2_probe};
use mummu::quant::QuantPolicy;

fn prompts() -> Vec<(&'static str, Vec<Turn>)> {
    vec![
        (
            "parity",
            vec![Turn::user("List the first five prime numbers.")],
        ),
        (
            "arith",
            vec![Turn::user("What is 2+2? Answer in one short sentence.")],
        ),
        (
            "code",
            vec![Turn::user(
                "Write a Rust function that returns the n-th Fibonacci number \
                 iteratively, then explain its time and space complexity in two \
                 sentences.",
            )],
        ),
        (
            "multiturn-json",
            vec![
                Turn::system("You are a helpful assistant for a kitchen cabinet showroom."),
                Turn::user(
                    "I need a quote for 12 linear feet of base cabinets in walnut. \
                     What information do you need?",
                ),
                Turn::assistant(
                    "To prepare a quote I need the door style, the finish, the \
                     countertop material, whether you want soft-close hardware, and \
                     your delivery zip code.",
                ),
                Turn::user(
                    "Shaker doors, natural oil finish, quartz top, soft-close yes, \
                     zip 30303. Reply as JSON with fields style, finish, top, \
                     soft_close, zip.",
                ),
            ],
        ),
        (
            "translate",
            vec![Turn::user(
                "Traduis en français puis en allemand : The meeting moved to \
                 Thursday at 3 pm because the server migration is not finished.",
            )],
        ),
        (
            "summarize",
            vec![Turn::user(
                "Summarize in three bullet points. The town's water plant was \
                 built in 1962 and expanded twice, in 1985 and 2004. The second \
                 expansion added a membrane filtration stage that cut turbidity \
                 by a factor of ten, but the intake pipes were never replaced and \
                 now leak roughly 8% of the raw water before treatment. A 2025 \
                 study estimated that relining the pipes would cost 4.2 million \
                 dollars and pay for itself in eleven years through reduced \
                 pumping energy. The council deferred the vote twice, citing the \
                 cost of a new fire station, and residents have since organized a \
                 petition asking for the relining to be funded from the capital \
                 reserve instead of a rate increase.",
            )],
        ),
    ]
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + x
        .iter()
        .map(|&v| (f64::from(v) - m).exp())
        .sum::<f64>()
        .ln();
    x.iter().map(|&v| f64::from(v) - lse).collect()
}

fn top(lp: &[f64], k: usize) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..lp.len()).collect();
    idx.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
    idx[..k].iter().map(|&i| i as u32).collect()
}

/// Relative difference between the two forms' normalized head at norm `n`.
fn rel(n: f32, eps: f64) -> f64 {
    let n = f64::from(n);
    1.0 - n.max(eps) / (n * n + eps).sqrt()
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(
        args.next()
            .expect("usage: gdn-l2-probe <gguf> <off|q8|q4> [greedy_tokens]"),
    );
    let policy = match args.next().as_deref() {
        Some("off") | None => QuantPolicy::Off,
        Some("q8") => QuantPolicy::Q8,
        Some("q4") => QuantPolicy::Q4,
        Some(other) => panic!("unknown quant policy {other:?} (off|q8|q4)"),
    };
    let greedy: usize = args.next().map_or(0, |s| s.parse().expect("greedy_tokens"));
    let device = match std::env::var("PROBE_DEVICE").as_deref() {
        Ok("gpu") => mummu::backend::gpu_device(),
        _ => mummu::backend::cpu_device(),
    };

    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);

    let t0 = Instant::now();
    let mut loaded = qwen35::load_from_gguf_quantized(&path, &device, policy).expect("model loads");
    let cfg = loaded.config.clone();
    let eps = cfg.rms_norm_eps;
    let delta_layers: Vec<usize> = (0..cfg.num_layers)
        .filter(|&i| !cfg.is_attention(i))
        .collect();
    eprintln!(
        "[gdn-l2] {} | {policy:?} on {device:?} | loaded in {:.1}s | layers {} ({} DeltaNet) \
         hk {} hv {} ds {} eps {eps:e} | default form {:?}",
        path.display(),
        t0.elapsed().as_secs_f32(),
        cfg.num_layers,
        delta_layers.len(),
        cfg.n_k_heads,
        cfg.n_v_heads,
        cfg.d_state,
        cfg.gdn_l2,
    );

    let mut pooled_q: Vec<Vec<f32>> = vec![Vec::new(); delta_layers.len()];
    let mut pooled_k: Vec<Vec<f32>> = vec![Vec::new(); delta_layers.len()];

    for (name, turns) in prompts() {
        let rendered = mummu::chat::ChatMl::qwen3().render(&turns);
        let ids = tok
            .encode(rendered.as_str(), false)
            .expect("prompt encodes")
            .get_ids()
            .to_vec();

        let mut logits = Vec::new();
        for form in [GdnL2::ClampNorm, GdnL2::AddEps] {
            loaded.config.gdn_l2 = form;
            gdn_l2_probe::set_enabled(form == GdnL2::ClampNorm);
            let t = Instant::now();
            let mut cache = loaded.new_cache();
            let l = loaded
                .forward(&ids, 0, &mut cache, &device)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .expect("logits read back");
            gdn_l2_probe::set_enabled(false);
            eprintln!(
                "[gdn-l2] {name}: {} tokens, {form:?} prefill {:.1}s",
                ids.len(),
                t.elapsed().as_secs_f32()
            );
            assert!(
                l.iter().all(|v| v.is_finite()),
                "{name}/{form:?}: non-finite logits"
            );
            logits.push(l);
        }
        let records = gdn_l2_probe::take();
        assert_eq!(
            records.len(),
            delta_layers.len(),
            "one record per DeltaNet layer per prefill"
        );
        for (i, r) in records.into_iter().enumerate() {
            pooled_q[i].extend(r.q);
            pooled_k[i].extend(r.k);
        }

        let (lc, la) = (log_softmax(&logits[0]), log_softmax(&logits[1]));
        let (tc, ta) = (top(&lc, 5), top(&la, 5));
        let max_dlp = tc
            .iter()
            .chain(&ta)
            .map(|&i| (lc[i as usize] - la[i as usize]).abs())
            .fold(0.0f64, f64::max);
        let kl: f64 = lc.iter().zip(&la).map(|(&c, &a)| c.exp() * (c - a)).sum();
        let max_dlogit = logits[0]
            .iter()
            .zip(&logits[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "prompt {name:<15} n={:<4} top5 clamp {tc:?} add {ta:?} same_order={} \
             max|dlogprob|(top5 union)={max_dlp:.3e} KL(clamp||add)={kl:.3e} max|dlogit|={max_dlogit:.3e}",
            ids.len(),
            tc == ta,
        );
        let top10 = |lp: &[f64]| -> String {
            top(lp, 10)
                .iter()
                .map(|&i| format!("{i}:{:.6}", lp[i as usize]))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!("  ids   {ids:?}");
        println!("  clamp {}", top10(&lc));
        println!("  add   {}", top10(&la));

        if greedy > 0 {
            let mut outs = Vec::new();
            for form in [GdnL2::ClampNorm, GdnL2::AddEps] {
                loaded.config.gdn_l2 = form;
                let out = loaded
                    .greedy_generate(&ids, greedy, &device)
                    .await
                    .expect("greedy decode");
                outs.push(out);
            }
            let agree = outs[0]
                .iter()
                .zip(&outs[1])
                .take_while(|(a, b)| a == b)
                .count();
            println!(
                "greedy {name:<15} agree {agree}/{} | clamp {:?} | add {:?}",
                outs[0].len().min(outs[1].len()),
                tok.decode(&outs[0], false).unwrap_or_default(),
                tok.decode(&outs[1], false).unwrap_or_default(),
            );
        }
    }

    println!(
        "\n{:>5} | {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>9} | {:>9} {:>9} {:>8} {:>8} {:>9}",
        "layer",
        "k min",
        "k p0.1",
        "k p1",
        "k p50",
        "k>=1e-3",
        "k>=1e-2",
        "k maxrel",
        "q min",
        "q p1",
        "q>=1e-3",
        "q>=1e-2",
        "q maxrel",
    );
    let frac = |v: &[f32], thr: f64| v.iter().filter(|&&n| rel(n, eps) >= thr).count();
    let (mut worst_k, mut worst_q) = ((f32::INFINITY, 0), (f32::INFINITY, 0));
    let (mut tot, mut k3, mut k2, mut q3, mut q2) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (i, &layer) in delta_layers.iter().enumerate() {
        let mut k = pooled_k[i].clone();
        let mut q = pooled_q[i].clone();
        k.sort_by(f32::total_cmp);
        q.sort_by(f32::total_cmp);
        let (fk3, fk2, fq3, fq2) = (
            frac(&k, 1e-3),
            frac(&k, 1e-2),
            frac(&q, 1e-3),
            frac(&q, 1e-2),
        );
        tot += k.len();
        k3 += fk3;
        k2 += fk2;
        q3 += fq3;
        q2 += fq2;
        if k[0] < worst_k.0 {
            worst_k = (k[0], layer);
        }
        if q[0] < worst_q.0 {
            worst_q = (q[0], layer);
        }
        println!(
            "{layer:>5} | {:>9.3e} {:>9.3e} {:>9.3e} {:>9.3e} {:>8} {:>8} {:>9.2e} | {:>9.3e} {:>9.3e} {:>8} {:>8} {:>9.2e}",
            k[0],
            quantile(&k, 0.001),
            quantile(&k, 0.01),
            quantile(&k, 0.5),
            fk3,
            fk2,
            rel(k[0], eps),
            q[0],
            quantile(&q, 0.01),
            fq3,
            fq2,
            rel(q[0], eps),
        );
    }
    println!(
        "\nsummary: {tot} (token, k-head) samples pooled over {} DeltaNet layers x {} prompts; \
         min |k| {:.3e} (layer {}, rel {:.2e}), min |q| {:.3e} (layer {}, rel {:.2e}); \
         k heads >=1e-3 rel: {k3} ({:.4}%), >=1e-2: {k2} ({:.4}%); q heads >=1e-3: {q3}, >=1e-2: {q2}",
        delta_layers.len(),
        prompts().len(),
        worst_k.0,
        worst_k.1,
        rel(worst_k.0, eps),
        worst_q.0,
        worst_q.1,
        rel(worst_q.0, eps),
        100.0 * k3 as f64 / tot as f64,
        100.0 * k2 as f64 / tot as f64,
    );
}
