//! Benchmark support crate. The criterion benches live in `benches/`, the
//! budget gates in `tests/`; what this lib carries is the two things a
//! recorded number needs to be trustworthy — the label it wears, and the
//! precondition that decides whether it should have been taken at all.

/// The GPU feature set this binary was compiled with, as it appears beside
/// every recorded number.
///
/// Numbers from the two sets are **not** interchangeable — `fusion` compiles
/// streams of ops into fewer kernels and `vulkan-spirv` changes which
/// compiler emits them — and this crate has already been bitten once by a
/// measurement wearing the wrong label (the 2026-07-11 "f16" rows that were
/// f32; see `bench/BASELINE.md`). A gate that prints this cannot record a
/// number under the wrong heading without the transcript saying so.
#[must_use]
pub fn gpu_feature_set() -> &'static str {
    let set = match (cfg!(feature = "fusion"), cfg!(feature = "vulkan-spirv")) {
        (true, true) => "fusion+vulkan-spirv",
        (true, false) => "fusion",
        (false, true) => "vulkan-spirv",
        (false, false) => "plain-wgsl",
    };
    // Positive space: the label always names a set.
    debug_assert!(!set.is_empty(), "feature-set label is never empty");
    // Negative space: it never claims a feature that is off.
    debug_assert!(
        cfg!(feature = "fusion") || !set.contains("fusion"),
        "label claims fusion while the feature is off"
    );
    set
}

/// Bytes in one mebibyte — the unit `bench/BASELINE.md` records its VRAM
/// figures in and the one `nvidia-smi` prints.
const MIB: u64 = 1024 * 1024;

/// Is `free_mib` enough for a gate that needs `need_mib`?
///
/// Split out from [`gpu_has_room_for`] so the decision is a pure leaf with a
/// test, rather than a branch that only a particular card on a particular
/// afternoon can exercise.
#[must_use]
fn fits(free_mib: u64, need_mib: u64) -> bool {
    debug_assert!(need_mib > 0, "a gate that needs no VRAM should not ask");
    free_mib >= need_mib
}

/// Does the card have room for a gate needing `need_mib` MiB of VRAM — and
/// if not, say so in a line that cannot be mistaken for a pass or a failure?
///
/// A GPU perf gate on a shared card has three possible outcomes and only two
/// of them are legitimate: it ran, or it could not run. Blocking until the
/// card clears is the illegitimate third — a routine that waits spends its
/// budget on somebody else's model, and a gate that waits silently is
/// indistinguishable from one that passed. So this reports rather than waits.
/// `false` means the caller should return WITHOUT asserting; the reading has
/// already been printed for it.
///
/// Reads NVML through [`mummu::vram`], i.e. the card's *global* free memory —
/// the number that moves when another process takes VRAM, not the per-process
/// DXGI budget, which happily reports plenty while someone else holds the
/// card. When nothing on the machine will say, this returns `true`: no
/// information is not the same as no room, and a box without an NVIDIA driver
/// must keep behaving exactly as it did before this check existed.
#[must_use]
pub fn gpu_has_room_for(need_mib: u64, label: &str) -> bool {
    assert!(need_mib > 0, "a gate that needs no VRAM should not ask");
    assert!(
        !label.is_empty(),
        "the skip line has to name which gate skipped"
    );

    let Some(mem) = mummu::vram::memory() else {
        return true;
    };
    let (free_mib, used_mib, total_mib) = (mem.free / MIB, mem.used / MIB, mem.total / MIB);
    if fits(free_mib, need_mib) {
        return true;
    }
    eprintln!(
        "[{label}] SKIPPED, not a regression: needs {need_mib} MiB of VRAM, \
         card has {free_mib} MiB free ({used_mib} of {total_mib} MiB held by every \
         process on this machine). Re-run when the card is quiet."
    );
    false
}

#[cfg(test)]
mod tests {
    use super::{fits, gpu_feature_set};

    /// The label must agree with the cfg the test binary itself was built
    /// under — the whole point is that it cannot drift from the build.
    #[test]
    fn the_label_names_exactly_the_features_that_are_on() {
        let set = gpu_feature_set();
        assert_eq!(set.contains("fusion"), cfg!(feature = "fusion"));
        assert_eq!(set.contains("vulkan-spirv"), cfg!(feature = "vulkan-spirv"));
        assert!(!set.is_empty());
    }

    /// The boundary is the whole point: a card with exactly the need free
    /// runs, one MiB short skips. Getting this backwards either flakes every
    /// gate at the margin or lets one OOM inside its own budget.
    #[test]
    fn the_room_check_is_inclusive_at_the_need_and_exclusive_below_it() {
        assert!(fits(9500, 9500), "exactly enough is enough");
        assert!(fits(16376, 9500));
        assert!(!fits(9499, 9500), "one MiB short does not run");
        assert!(!fits(0, 1), "an entirely held card never runs");
    }
}
