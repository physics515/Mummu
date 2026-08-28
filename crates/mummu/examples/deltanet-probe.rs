//! Times ONE DeltaNet recurrence step at the 27B's real state shape, op by
//! op. The bandwidth and per-op probes both came in far under the observed
//! 29.7 s/token, so the cost has to be inside these ops at this shape.
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};

fn main() {
    // Which device to probe: the recurrence is 48 of the 27B's 65 layers and
    // is a SEQUENTIAL loop of small ops, so it is the least GPU-friendly part
    // of the model — exactly what a matmul microbenchmark cannot see.
    let device = match std::env::var("PROBE_DEVICE").as_deref() {
        Ok("cuda") => {
            #[cfg(feature = "cuda")]
            {
                mummu::backend::cuda_device()
            }
            #[cfg(not(feature = "cuda"))]
            {
                mummu::backend::cpu_device()
            }
        }
        Ok("gpu") => mummu::backend::gpu_device(),
        _ => mummu::backend::cpu_device(),
    };
    println!("device: {:?}", device);
    let (b, hv, ds) = (1usize, 24usize, 256usize);
    let n = 20;
    let rnd4 = |dims: [usize; 4]| Tensor::<4>::random(dims, Distribution::Default, &device);

    let s0 = rnd4([b, hv, ds, ds]);
    let k_t = rnd4([b, hv, 1, ds]);
    let q_t = rnd4([b, hv, 1, ds]);
    let v_t = rnd4([b, hv, 1, ds]);
    let g_t = rnd4([b, hv, 1, 1]);

    macro_rules! time {
        ($label:expr, $body:expr) => {{
            let _ = $body; // warm
            let t = Instant::now();
            for _ in 0..n {
                let _ = $body;
            }
            println!(
                "{:<42} {:>8.3} ms",
                $label,
                t.elapsed().as_secs_f64() * 1e3 / f64::from(n)
            );
        }};
    }

    time!(
        "s.mul(g.exp())            [decay]",
        s0.clone().mul(g_t.clone().exp())
    );
    time!(
        "s.mul(k^T).sum_dim(2)     [v_hat]",
        s0.clone().mul(k_t.clone().swap_dims(2, 3)).sum_dim(2)
    );
    time!(
        "k^T.matmul(d)             [outer]",
        k_t.clone().swap_dims(2, 3).matmul(v_t.clone())
    );
    time!(
        "s.add(outer)              [update]",
        s0.clone().add(s0.clone())
    );
    time!(
        "s.mul(q^T).sum_dim(2)     [output]",
        s0.clone().mul(q_t.clone().swap_dims(2, 3)).sum_dim(2)
    );

    // The whole step, as the model runs it.
    let t = Instant::now();
    for _ in 0..n {
        let mut s = s0.clone();
        s = s.mul(g_t.clone().exp());
        let v_hat = s.clone().mul(k_t.clone().swap_dims(2, 3)).sum_dim(2);
        let d = v_t.clone().sub(v_hat).mul(g_t.clone());
        s = s.add(k_t.clone().swap_dims(2, 3).matmul(d));
        let o = s.clone().mul(q_t.clone().swap_dims(2, 3)).sum_dim(2);
        let _ = o.into_data();
    }
    let per_step = t.elapsed().as_secs_f64() / f64::from(n);
    println!(
        "\nFULL DeltaNet step (1 layer, 1 token): {:.3} ms",
        per_step * 1e3
    );
    println!(
        "x48 DeltaNet layers                  : {:.2} s/token",
        per_step * 48.0
    );
}
