//! Preprocess images through the vision pipeline and print what actually
//! happened to each — without loading a language model.
//!
//! The whole vision investigation so far has run at the wrong granularity:
//! every question cost a container rebuild, a cold 27B load, and a decode,
//! and the answer was a sentence of English that could not distinguish
//! "the embeddings are wrong" from "the model miscounted". This runs in
//! seconds and reports numbers.
//!
//! ```text
//! cargo run --example vision-trace -- <mmproj.gguf> <image> [image ...]
//! ```
//!
//! Two images that the arithmetic says should preprocess identically — the
//! 900x675 and 1500x1125 case — must print the same grid AND the same
//! fingerprint. If they do not, the divergence is here, in preprocessing.
//! If they do, it is downstream in the tower, which halves the search.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use mummu::vision::VisionConfig;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(mmproj) = args.next() else {
        eprintln!("usage: vision-trace <mmproj.gguf> <image> [image ...]");
        std::process::exit(2);
    };
    let images: Vec<String> = args.collect();
    if images.is_empty() {
        eprintln!("no images given");
        std::process::exit(2);
    }

    let f = mummu::gguf::GgufFile::open(std::path::Path::new(&mmproj))
        .unwrap_or_else(|e| panic!("open {mmproj}: {e:?}"));
    let cfg = VisionConfig::from_gguf(&f).unwrap_or_else(|e| panic!("mmproj config: {e}"));
    println!(
        "tower: {}x{} reference, patch {}, merge {}, hidden {} -> {}",
        cfg.image_size, cfg.image_size, cfg.patch, cfg.merge, cfg.hidden, cfg.out_dim
    );

    let mut seen: Vec<(String, String, String)> = Vec::new();
    for path in &images {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        match cfg.preprocess(&bytes) {
            Ok(p) => {
                let grid = format!("{}x{}", p.grid_h, p.grid_w);
                let tokens = (p.grid_h / cfg.merge) * (p.grid_w / cfg.merge);
                let fp = p.fingerprint();
                println!(
                    "{:<28} grid {:>7} ({:>4} tokens, {:>4}x{:<4} px)  {fp}",
                    short(path),
                    grid,
                    tokens,
                    p.grid_w * cfg.patch,
                    p.grid_h * cfg.patch,
                );
                seen.push((short(path), grid, fp));
            }
            Err(e) => println!("{:<28} REFUSED: {e}", short(path)),
        }
    }

    // The point of the exercise: say out loud which inputs the pipeline
    // treats as the same, so a disagreement downstream is attributable.
    println!("\ngroups by (grid, fingerprint):");
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for (name, grid, fp) in seen {
        let key = format!("{grid} {fp}");
        if let Some(g) = groups.iter_mut().find(|(k, _)| *k == key) {
            g.1.push(name);
        } else {
            groups.push((key, vec![name]));
        }
    }
    for (key, names) in &groups {
        println!("  [{}] {key}", names.join(", "));
    }
    if std::env::var("MUMMU_VISION_EMBED").is_ok_and(|v| v != "0") {
        compare_embeddings(&cfg, &mmproj, &images);
    }
    if std::env::var("MUMMU_VISION_STAGES").is_ok_and(|v| v != "0") && images.len() >= 2 {
        compare_stages(&cfg, &mmproj, &images[0], &images[1]);
    }
    if groups.len() == 1 {
        println!(
            "\nAll inputs preprocess IDENTICALLY. Any difference in what the model says \
             about them comes from the tower or below, not from here."
        );
    }
}

/// Run the tower over each image and report how similar the projected
/// tokens are, pairwise.
///
/// This is the check that needs no reference implementation. Two images
/// that preprocess to near-identical tensors must produce near-identical
/// image tokens — a correct encoder is not sensitive to resampling noise.
/// If similarity between them is low, the tower is *unstable*, and an
/// unstable encoder is a wrong one however plausible its output reads.
fn compare_embeddings(cfg: &VisionConfig, mmproj: &str, images: &[String]) {
    let device = mummu::backend::cpu_device();
    let tower = match mummu::vision::VisionTower::load(std::path::Path::new(mmproj), &device) {
        Ok(t) => t,
        Err(e) => {
            println!("\ntower load failed: {e}");
            return;
        }
    };
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    for path in images {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok(p) = cfg.preprocess(&bytes) else {
            continue;
        };
        let Ok(out) = tower.forward(p.to_tensor(&device), p.grid_h, p.grid_w, &device) else {
            continue;
        };
        let Ok(v) = out.into_data().convert::<f32>().try_into_vec::<f32>() else {
            continue;
        };
        rows.push((short(path), v));
    }

    println!("\npairwise cosine similarity of the projected image tokens:");
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let (a, b) = (&rows[i].1, &rows[j].1);
            // Only comparable when the token counts match; a different grid
            // is a different sequence length, not a different answer.
            if a.len() != b.len() {
                println!(
                    "  {:<22} vs {:<22}  (different token counts)",
                    rows[i].0, rows[j].0
                );
                continue;
            }
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            let na: f64 = a
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = b
                .iter()
                .map(|&y| f64::from(y) * f64::from(y))
                .sum::<f64>()
                .sqrt();
            let cos = dot / (na * nb).max(1e-12);
            let verdict = if cos > 0.99 {
                "stable"
            } else if cos > 0.9 {
                "drifting"
            } else {
                "UNSTABLE"
            };
            println!(
                "  {:<22} vs {:<22}  cos {cos:.4}  {verdict}",
                rows[i].0, rows[j].0
            );
        }
    }
}

/// Run two images through the tower and report, stage by stage, how far
/// apart they are.
///
/// The final-output cosine says the tower drifts; this says where. The first
/// stage whose similarity falls is where the divergence enters. Also
/// prints each stage's own statistics for the first image, since a stage
/// whose activations explode (a large `maxabs`, a growing `l2`) is a suspect
/// on its own.
fn compare_stages(cfg: &VisionConfig, mmproj: &str, a: &str, b: &str) {
    let device = mummu::backend::cpu_device();
    let tower = match mummu::vision::VisionTower::load(std::path::Path::new(mmproj), &device) {
        Ok(t) => t,
        Err(e) => {
            println!("\ntower load failed: {e}");
            return;
        }
    };
    let run = |path: &str| -> Option<Vec<mummu::vision::Stage>> {
        let bytes = std::fs::read(path).ok()?;
        let p = cfg.preprocess(&bytes).ok()?;
        let (_, stages) = tower
            .forward_staged(p.to_tensor(&device), p.grid_h, p.grid_w, &device)
            .ok()?;
        Some(stages)
    };
    let (Some(sa), Some(sb)) = (run(a), run(b)) else {
        println!("\nstage run failed");
        return;
    };

    println!("\nstage-by-stage: {} vs {}", short(a), short(b));
    println!(
        "  {:<11} {:>9} {:>9} {:>11} {:>9}   {:>8}  ",
        "stage", "mean", "sd", "l2", "maxabs", "cos(a,b)"
    );
    let mut prev: Option<f64> = None;
    for (x, y) in sa.iter().zip(&sb) {
        let (mean, sd, l2, maxabs) = x.summary();
        let cos = x.cosine(y);
        // Mark the stage where agreement falls most sharply: that is where to
        // look first.
        let flag = match (prev, cos) {
            (Some(p), Some(c)) if p - c > 0.01 => format!("  <- dropped {:.4}", p - c),
            _ => String::new(),
        };
        println!(
            "  {:<11} {mean:>+9.4} {sd:>9.4} {l2:>11.2} {maxabs:>9.3}   {:>8}{flag}",
            x.name,
            cos.map_or_else(|| "n/a".into(), |c| format!("{c:.4}")),
        );
        if cos.is_some() {
            prev = cos;
        }
    }
}

fn short(p: &str) -> String {
    std::path::Path::new(p)
        .file_name()
        .map_or_else(|| p.to_string(), |n| n.to_string_lossy().into_owned())
}
