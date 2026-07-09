//! Runner benchmark stub. Real benches (TTFT, decode tok/s, VRAM per model ×
//! device-set) land with the decode engine (P5); until then this measures a
//! trivial baseline so the criterion harness itself is wired and green.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_harness_smoke(c: &mut Criterion) {
    c.bench_function("harness_smoke", |b| {
        b.iter(|| black_box(1u64).wrapping_add(black_box(1u64)));
    });
}

criterion_group!(benches, bench_harness_smoke);
criterion_main!(benches);
