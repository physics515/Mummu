//! **Chunked prefill: pick the chunk size, delete the 855 MB reserve.**
//!
//! Unchunked prefill of a 4096-token prompt through the 27B's SwiGLU FFN
//! holds three ctx-by-intermediate f32 buffers live at once — gate, up,
//! and their product — which at intermediate = 17408 is
//! 3 * 4096 * 17408 * 4 ~= 855 MB of activation peak. Today that peak is
//! charged to the VRAM reserve for the whole session, evicting ~3.5 layers
//! of Q4 weights (~236 MB each) to protect a burst that lasts one prefill.
//! Chunking the prompt into c-token slices caps the peak at the same
//! formula with c in place of ctx, so the reserve term becomes a knob:
//! this module turns "how big should c be" from a guess into an argmin.
//!
//! The time model: a chunk costs a fixed `t_sync` (submit + fence + host
//! bookkeeping, independent of c) plus a kernel term `t_k(c) = k0 + k1*c`.
//! Affine-in-c is the right first model because the layers this serve path
//! runs are bandwidth-bound: every chunk re-streams the same weight bytes
//! (k0's main component at small c is per-dispatch overhead) and does
//! per-token work proportional to c. It stops being right once c is small
//! enough that occupancy collapses — which is exactly the regime the
//! optimizer below steers away from, because tiny c multiplies the chunk
//! count. Total: `T(c) = ceil(S/c) * (t_sync + k0 + k1*c)`.
//!
//! Memory: `M(c) = a0 + a1*c + a2*c^2` — affine covers the SwiGLU peak
//! (a1 = [`activation_peak_per_token_bytes`]), the quadratic term is there
//! for attention scores if a caller materializes the c-by-ctx logits.

/// Peak activation memory model `M(c) = a0 + a1*c + a2*c^2`, coefficients
/// in bytes (per token, per token squared). All coefficients must be
/// nonnegative and finite — the peak of a real kernel does not shrink as
/// its input grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemModel {
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
}

impl MemModel {
    /// `M(c)` in bytes.
    #[must_use]
    pub fn peak_bytes(&self, c: usize) -> f64 {
        let c = c as f64;
        self.a0 + self.a1 * c + self.a2 * c * c
    }
}

/// The solver's answer, with the model's own accounting attached so the
/// caller can log WHY this chunk size, not just which.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkChoice {
    /// The chosen chunk size, in tokens.
    pub chunk: usize,
    /// `ceil(S / chunk)` — how many chunks the prompt splits into.
    pub chunks: usize,
    /// `T(chunk)`, in the same time unit as `t_sync`/`k0`/`k1` (ms in this
    /// repo's calibrations).
    pub time: f64,
    /// `M(chunk)` in bytes — what the reserve actually has to carry.
    pub peak_bytes: f64,
    /// The closed-form unconstrained optimum (clamped to the scan range),
    /// kept for diagnostics: how far the memory budget pushed the choice
    /// off the overhead-vs-waste balance point. See the derivation in
    /// [`best_chunk`].
    pub hint: usize,
}

/// The SwiGLU three-live-buffer peak per prefill token: gate, up, and
/// their elementwise product, each `intermediate` f32 lanes wide. This is
/// the `a1` coefficient the serve crate should build its [`MemModel`] from
/// once it knows the config's intermediate size — 3 * 17408 * 4 bytes for
/// the 27B, the source of the ~855 MB unchunked term this module removes.
#[must_use]
pub fn activation_peak_per_token_bytes(intermediate: usize) -> u64 {
    3 * intermediate as u64 * 4
}

/// Minimize `T(c) = ceil(S/c) * (t_sync + k0 + k1*c)` over `c` in
/// `1..=min(c_max, s)` subject to `M(c) <= mem_budget`.
///
/// Returns `None` when there is nothing to solve (`s == 0`, `c_max == 0`)
/// or when no chunk size fits the budget — i.e. even one token's
/// activations exceed it, in which case chunking cannot save the caller
/// and it must shrink something else.
///
/// The exact argmin is the scan below (S is a few thousand; this is
/// microseconds once per prefill). The closed form is documentation of the
/// shape, and worth having exactly right:
///
/// Relaxing `ceil(S/c)` to `S/c` gives `T ~ S*(t_sync + k0)/c + S*k1`,
/// which is monotone DECREASING in c — under the smooth model the kernel
/// term is c-independent (the same S tokens get processed either way) and
/// bigger chunks only amortize per-chunk overhead, so the smooth optimum
/// is "one chunk" and the memory budget is what actually caps c. The
/// interesting stationary point comes from the ceiling: `ceil(S/c) <=
/// S/c + 1`, and minimizing the upper bound
/// `S*(t_sync+k0)/c + S*k1 + (t_sync + k0) + k1*c` balances the amortized
/// overhead term against `k1*c` — the c-dependent cost of the one extra
/// ragged chunk the ceiling can force (this model charges every chunk the
/// full `k1*c`, partial or not, matching kernels padded to the chunk
/// shape). Setting the derivative to zero:
///
///   d/dc [ S*(t_sync+k0)/c + k1*c ] = -S*(t_sync+k0)/c^2 + k1 = 0
///   =>  c* = sqrt(S * (t_sync + k0) / k1)
///
/// Past c*, growing c risks more ragged-chunk waste than it amortizes;
/// below it, sync overhead dominates. The scan finds the true argmin of
/// the stepped objective (which always sits at some `c = ceil(S/m)`);
/// c* is reported as [`ChunkChoice::hint`]. Ties in `T` break toward the
/// SMALLER c — equal time, smaller activation peak, more room under the
/// watermark.
#[must_use]
pub fn best_chunk(
    s: usize,
    t_sync: f64,
    k0: f64,
    k1: f64,
    mem: MemModel,
    mem_budget: u64,
    c_max: usize,
) -> Option<ChunkChoice> {
    debug_assert!(
        t_sync.is_finite() && t_sync >= 0.0,
        "t_sync must be finite, nonnegative"
    );
    debug_assert!(
        k0.is_finite() && k0 >= 0.0,
        "k0 must be finite, nonnegative"
    );
    debug_assert!(
        k1.is_finite() && k1 >= 0.0,
        "k1 must be finite, nonnegative"
    );
    debug_assert!(
        mem.a0 >= 0.0 && mem.a1 >= 0.0 && mem.a2 >= 0.0,
        "memory coefficients must be nonnegative"
    );

    if s == 0 || c_max == 0 {
        return None;
    }
    // c > S buys nothing over c = S (still one chunk) and the memory model
    // would charge for tokens that do not exist, so cap the range.
    let upper = c_max.min(s);

    // Closed-form hint (see the derivation above), clamped into the scan
    // range. With k1 = 0 the extra ragged chunk costs nothing c-dependent
    // and the amortization argument runs unopposed: as big as allowed.
    let hint = if k1 > 0.0 {
        let c_star = (s as f64 * (t_sync + k0) / k1).sqrt().round();
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let c_star = if c_star < 1.0 { 1 } else { c_star as usize };
        c_star.min(upper)
    } else {
        upper
    };

    let time_at = |c: usize| -> f64 { s.div_ceil(c) as f64 * (t_sync + k0 + k1 * c as f64) };

    let budget = mem_budget as f64;
    let mut best: Option<(usize, f64)> = None;
    for c in 1..=upper {
        if mem.peak_bytes(c) > budget {
            // M is nondecreasing in c (nonnegative coefficients): nothing
            // larger fits either.
            break;
        }
        let t = time_at(c);
        // Strict improvement only: ties keep the smaller, cheaper-in-memory c.
        if best.map_or(true, |(_, bt)| t < bt) {
            best = Some((c, t));
        }
    }

    best.map(|(chunk, time)| ChunkChoice {
        chunk,
        chunks: s.div_ceil(chunk),
        time,
        peak_bytes: mem.peak_bytes(chunk),
        hint,
    })
}

/// The chunkwise GDN prefill's cost model and argmin (SPEC P5.5's
/// adaptive-chunk half): evaluating a `t_tokens` prefill in chunks of `c`
/// costs
///
/// ```text
/// T(c) = ceil(t/c) * (a0 + a1*c + a2*c^2 + a3*c^3*log2(c))
/// ```
///
/// per layer, where `a0` is per-chunk launch/sync overhead (the term that
/// murders small chunks — the sequential loop is the degenerate c = 1 of
/// this model, ~9 launches per token), `a1*c` the per-token projections,
/// `a2*c^2` the C-by-C interaction matrices (decay, `K K^T`, `Q K^T`), and
/// `a3*c^3*log2(c)` the finite Neumann doubling (log2(c) stages of C-by-C
/// matmul). All coefficients are measured, not derived — fit them from
/// three timed chunk sizes and re-fit when the kernel changes.
///
/// Distinct from [`best_chunk`] deliberately: the memory-vs-time trade
/// there is about activation peaks; this one is pure time — the GDN chunk
/// never materializes anything bigger than `c x c` per head.
#[derive(Debug, Clone, PartialEq)]
pub struct GdnChunkChoice {
    pub chunk: usize,
    pub chunks: usize,
    /// `T(chunk)` in the coefficients' time unit.
    pub time: f64,
    /// The sequential loop's cost under the same model (`c = 1`), so the
    /// caller can log the ratio this choice buys.
    pub sequential_time: f64,
}

/// Minimize the GDN chunk model over `c in 1..=min(c_max, t_tokens)`.
/// Returns `None` for an empty prefill. Ties break toward the smaller `c`
/// (smaller transient state, same time).
#[must_use]
pub fn best_gdn_chunk(
    t_tokens: usize,
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    c_max: usize,
) -> Option<GdnChunkChoice> {
    debug_assert!(
        a0 >= 0.0 && a1 >= 0.0 && a2 >= 0.0 && a3 >= 0.0,
        "cost coefficients must be nonnegative"
    );
    if t_tokens == 0 || c_max == 0 {
        return None;
    }
    let per_chunk = |c: usize| -> f64 {
        let cf = c as f64;
        a0 + a1 * cf + a2 * cf * cf + a3 * cf * cf * cf * (cf.log2().max(0.0))
    };
    let time_at = |c: usize| -> f64 { t_tokens.div_ceil(c) as f64 * per_chunk(c) };
    let upper = c_max.min(t_tokens);
    let mut best: Option<(usize, f64)> = None;
    for c in 1..=upper {
        let t = time_at(c);
        if best.is_none_or(|(_, bt)| t < bt) {
            best = Some((c, t));
        }
    }
    best.map(|(chunk, time)| GdnChunkChoice {
        chunk,
        chunks: t_tokens.div_ceil(chunk),
        time,
        sequential_time: time_at(1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: f64 = 1024.0 * 1024.0;

    /// The 27B's numbers: 4096-token prompt, SwiGLU peak per token from
    /// intermediate = 17408, and plausible per-chunk timings (ms).
    fn model_27b() -> (usize, f64, f64, f64, MemModel) {
        let s = 4096;
        let (t_sync, k0, k1) = (0.35, 0.2, 0.004);
        let mem = MemModel {
            a0: 0.0,
            a1: activation_peak_per_token_bytes(17408) as f64,
            a2: 0.0,
        };
        (s, t_sync, k0, k1, mem)
    }

    /// The exact scan can never lose to the closed-form guess: the hint is
    /// one candidate the scan already considered.
    #[test]
    fn scan_beats_or_ties_the_closed_form_hint() {
        let (s, t_sync, k0, k1, mem) = model_27b();
        let budget = (300.0 * MIB) as u64;
        let got = best_chunk(s, t_sync, k0, k1, mem, budget, s).unwrap();
        assert!(got.hint >= 1 && got.hint <= s);
        // The hint is feasible in this instance, so compare directly.
        assert!(mem.peak_bytes(got.hint) <= budget as f64);
        let t_hint = s.div_ceil(got.hint) as f64 * (t_sync + k0 + k1 * got.hint as f64);
        assert!(
            got.time <= t_hint + 1e-12,
            "scan ({} @ {}) lost to its own hint ({t_hint} @ {})",
            got.time,
            got.chunk,
            got.hint
        );
    }

    /// The memory constraint binds correctly: with room to spare the
    /// solver takes one big chunk (the smooth model's answer), and a tight
    /// budget pushes it down to a c whose peak fits.
    #[test]
    fn memory_constraint_binds() {
        let (s, t_sync, k0, k1, mem) = model_27b();

        let roomy = best_chunk(s, t_sync, k0, k1, mem, u64::MAX, s).unwrap();
        assert_eq!(roomy.chunk, s, "with no memory pressure, one chunk wins");
        assert_eq!(roomy.chunks, 1);

        let budget = (300.0 * MIB) as u64;
        let tight = best_chunk(s, t_sync, k0, k1, mem, budget, s).unwrap();
        assert!(
            mem.peak_bytes(s) > budget as f64,
            "the test only means something if c = S is infeasible"
        );
        assert!(tight.chunk < s);
        assert!(tight.peak_bytes <= budget as f64);

        // And nothing at all feasible -> None, not a lie.
        assert_eq!(best_chunk(s, t_sync, k0, k1, mem, 1000, s), None);
    }

    /// Degenerate t_sync = k0 = 0: T(c) = ceil(S/c) * k1 * c >= S*k1 with
    /// equality exactly when c divides S, so c = 1 ties the optimum and the
    /// smaller-c tie-break must return it.
    #[test]
    fn zero_overhead_makes_the_smallest_chunk_optimal() {
        let mem = MemModel {
            a0: 0.0,
            a1: 1.0,
            a2: 0.0,
        };
        let got = best_chunk(1000, 0.0, 0.0, 0.01, mem, u64::MAX, 1000).unwrap();
        assert_eq!(got.chunk, 1);
        assert!((got.time - 1000.0 * 0.01).abs() < 1e-9);
    }

    /// The reported time is the model evaluated at the reported chunk —
    /// no drift between the argmin loop and the returned artifact.
    #[test]
    fn reported_time_matches_direct_evaluation() {
        let (s, t_sync, k0, k1, mem) = model_27b();
        let got = best_chunk(s, t_sync, k0, k1, mem, (300.0 * MIB) as u64, s).unwrap();
        let direct = s.div_ceil(got.chunk) as f64 * (t_sync + k0 + k1 * got.chunk as f64);
        assert!((got.time - direct).abs() < 1e-12);
        assert_eq!(got.chunks, s.div_ceil(got.chunk));
        assert!((got.peak_bytes - mem.peak_bytes(got.chunk)).abs() < 1e-9);
    }

    /// The case this module ships for: the 27B's 855 MB unchunked peak,
    /// capped to a 300 MB budget. The solver must fit the budget, and the
    /// eprintln reports what that safety costs in prefill time versus the
    /// infeasible one-chunk ideal.
    #[test]
    fn the_27b_case_fits_300_mb() {
        let (s, t_sync, k0, k1, mem) = model_27b();
        let budget = (300.0 * MIB) as u64;

        // Sanity on the constants: unchunked peak ~855 MB (3*4096*17408*4),
        // feasible c cap a bit over 1500 tokens.
        let unchunked_mb = mem.peak_bytes(s) / 1e6;
        assert!(
            (850.0..860.0).contains(&unchunked_mb),
            "unchunked peak {unchunked_mb} MB"
        );
        let c_cap = (budget as f64 / mem.a1) as usize;
        assert!(
            (1400..1600).contains(&c_cap),
            "feasible cap ~1505, got {c_cap}"
        );

        let got = best_chunk(s, t_sync, k0, k1, mem, budget, s).unwrap();
        assert!(got.peak_bytes <= budget as f64);
        assert!(got.chunk <= c_cap);
        assert!(
            got.chunks >= 3,
            "4096 tokens under a ~1505 cap is at least 3 chunks"
        );

        let t_one_chunk = t_sync + k0 + k1 * s as f64;
        eprintln!(
            "27B chunked prefill: c={} ({} chunks), T={:.2} ms vs one-chunk T={:.2} ms \
             (+{:.1}% time), peak {:.0} MiB vs {:.0} MiB (unchunked, infeasible)",
            got.chunk,
            got.chunks,
            got.time,
            t_one_chunk,
            (got.time / t_one_chunk - 1.0) * 100.0,
            got.peak_bytes / MIB,
            mem.peak_bytes(s) / MIB,
        );
    }

    /// Nothing to prefill or no chunk allowed: no answer, by contract.
    #[test]
    fn degenerate_inputs_return_none() {
        let mem = MemModel {
            a0: 0.0,
            a1: 1.0,
            a2: 0.0,
        };
        assert_eq!(best_chunk(0, 1.0, 1.0, 1.0, mem, 1000, 64), None);
        assert_eq!(best_chunk(64, 1.0, 1.0, 1.0, mem, 1000, 0), None);
    }

    /// The quadratic term participates: a2 > 0 must cap c sooner than the
    /// affine part alone would.
    #[test]
    fn quadratic_memory_term_caps_the_chunk() {
        let affine = MemModel {
            a0: 0.0,
            a1: 100.0,
            a2: 0.0,
        };
        let quad = MemModel {
            a0: 0.0,
            a1: 100.0,
            a2: 10.0,
        };
        let budget = 1_000_000u64; // affine cap: 10_000; quad cap: ~311
        let a = best_chunk(4096, 0.5, 0.1, 0.001, affine, budget, 4096).unwrap();
        let q = best_chunk(4096, 0.5, 0.1, 0.001, quad, budget, 4096).unwrap();
        assert!(q.chunk < a.chunk, "quad {} vs affine {}", q.chunk, a.chunk);
        assert!(q.peak_bytes <= budget as f64);
    }

    /// The SwiGLU constant, spelled out once: gate + up + product, f32.
    #[test]
    fn activation_peak_is_three_f32_buffers() {
        assert_eq!(activation_peak_per_token_bytes(17408), 3 * 17408 * 4);
        assert_eq!(activation_peak_per_token_bytes(0), 0);
    }

    /// GDN chunk model: launch-dominated costs pick a big chunk, cubic
    /// (solve)-dominated costs pick a small one, and the reported times are
    /// the model at the reported points — checked against a brute scan.
    #[test]
    fn gdn_chunk_tracks_the_dominant_term() {
        // Launch-heavy (a0 huge): the sequential loop pays a0 per token, so
        // the argmin runs to the cap.
        let launchy = best_gdn_chunk(2048, 5.0, 0.01, 0.0, 0.0, 64).unwrap();
        assert_eq!(launchy.chunk, 64);
        assert!(launchy.sequential_time > launchy.time * 10.0);

        // Solve-heavy (a3 huge): big chunks pay c^3 log c, so small wins.
        let solvey = best_gdn_chunk(2048, 0.01, 0.0, 0.0, 1.0, 64).unwrap();
        assert!(
            solvey.chunk <= 2,
            "cubic-dominated argmin at {}",
            solvey.chunk
        );

        // A balanced instance has an interior minimum; verify vs brute force.
        let (a0, a1, a2, a3) = (2.0, 0.05, 0.001, 1e-6);
        let got = best_gdn_chunk(4096, a0, a1, a2, a3, 256).unwrap();
        let model = |c: usize| {
            let cf = c as f64;
            4096f64.div_euclid(cf).max(0.0); // silence: use div_ceil below
            (4096usize.div_ceil(c)) as f64
                * (a0 + a1 * cf + a2 * cf * cf + a3 * cf * cf * cf * cf.log2().max(0.0))
        };
        let brute = (1..=256).map(model).fold(f64::INFINITY, f64::min);
        assert!((got.time - brute).abs() < 1e-9);
        assert!(
            got.chunk > 1 && got.chunk < 256,
            "interior argmin, got {}",
            got.chunk
        );
        assert!((got.sequential_time - model(1)).abs() < 1e-9);
    }

    /// Degenerate GDN inputs return None.
    #[test]
    fn gdn_chunk_degenerate_inputs() {
        assert_eq!(best_gdn_chunk(0, 1.0, 1.0, 0.0, 0.0, 64), None);
        assert_eq!(best_gdn_chunk(64, 1.0, 1.0, 0.0, 0.0, 0), None);
    }
}
