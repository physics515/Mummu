//! Exact, `as`-free numeric conversions for the mummu workspace.
//!
//! An `as` cast between an integer and a float is silent about what it
//! loses: `usize as f32` rounds past 2^24, `f64 as f32` narrows, `f64 as u64`
//! truncates and saturates. The workspace lints (`clippy::pedantic` with no
//! `allow`s) reject every lossy cast, so this crate is the one place those
//! conversions are spelled out at the bit level. Every function here is
//! **bit-identical to the `as` cast it replaces** on the domain its doc
//! states — the tests prove that against an independent reference (decimal
//! round-trips, which the standard library parses correctly rounded) — so a
//! call site that swaps `x as f32` for [`narrow(x)`](narrow) changes nothing
//! a parity gate could see.
//!
//! Naming: `f64_from_*` / `f32_from_*` widen an integer, [`narrow`] shortens
//! a double, and `trunc_*` go float → integer with `as` semantics (toward
//! zero, saturating, `NaN` → 0).
//!
//! Every function is `#[inline]`. They stand where a single machine
//! instruction used to, some of them inside the host GEMV's inner loop, and
//! a cross-crate call there would cost more than the whole kernel saves —
//! `lto = "thin"` would probably inline them anyway, but this does not leave
//! the hot path to a profile setting.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

/// Round `m >> s` to nearest, ties to even, for any shift (a shift past the
/// width is a shift to zero: `m` is always below 2^53 here).
#[inline]
const fn round_half_even(m: u64, s: u32) -> u64 {
    if s == 0 {
        return m;
    }
    if s > 63 {
        return 0;
    }
    let q = m >> s;
    let rem = m & ((1u64 << s) - 1);
    let half = 1u64 << (s - 1);
    if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// `n as f64`: exact below 2^53, correctly rounded (ties to even) above.
///
/// The high word times 2^32 is exact (32 significant bits), so the one
/// addition is the single rounding `as` performs.
#[inline]
#[must_use]
pub fn f64_from_u64(n: u64) -> f64 {
    let hi = (n >> 32) as u32;
    let lo = (n & 0xFFFF_FFFF) as u32;
    let high = f64::from(hi) * 4_294_967_296.0;
    high + f64::from(lo)
}

/// `n as f64` for a `usize`; see [`f64_from_u64`].
#[inline]
#[must_use]
pub fn f64_from_usize(n: usize) -> f64 {
    f64_from_u64(n as u64)
}

/// `n as f64` for a signed 64-bit integer; see [`f64_from_u64`].
#[inline]
#[must_use]
pub fn f64_from_i64(n: i64) -> f64 {
    let magnitude = f64_from_u64(n.unsigned_abs());
    if n < 0 { -magnitude } else { magnitude }
}

/// `n as f64` for an `isize`; see [`f64_from_i64`].
#[inline]
#[must_use]
pub fn f64_from_isize(n: isize) -> f64 {
    f64_from_i64(n as i64)
}

/// `n as f32`: exact below 2^24, correctly rounded (ties to even) above.
///
/// The high half times 2^16 is exact (16 significant bits), so the one
/// addition is the single rounding `as` performs.
#[inline]
#[must_use]
pub fn f32_from_u32(n: u32) -> f32 {
    let hi = (n >> 16) as u16;
    let lo = (n & 0xFFFF) as u16;
    let high = f32::from(hi) * 65_536.0;
    high + f32::from(lo)
}

/// `n as f32` for a 64-bit integer.
///
/// The detour through `f64` is safe for the whole `u64` range, not just
/// below 2^53: double rounding can only bite when the wide format carries
/// fewer than `2p + 2` bits of the narrow one, and f64's 53 clears f32's
/// 50. The sweep in this module's tests checks that against a correctly
/// rounded reference up to `u64::MAX`.
#[inline]
#[must_use]
pub fn f32_from_u64(n: u64) -> f32 {
    u32::try_from(n).map_or_else(|_| narrow(f64_from_u64(n)), f32_from_u32)
}

/// `n as f32` for a `usize`; see [`f32_from_u64`].
#[inline]
#[must_use]
pub fn f32_from_usize(n: usize) -> f32 {
    f32_from_u64(n as u64)
}

/// `n as f32` for a signed 32-bit integer; see [`f32_from_u32`].
#[inline]
#[must_use]
pub fn f32_from_i32(n: i32) -> f32 {
    let magnitude = f32_from_u32(n.unsigned_abs());
    if n < 0 { -magnitude } else { magnitude }
}

/// `n as f32` for a signed 64-bit integer; see [`f32_from_u64`].
#[inline]
#[must_use]
pub fn f32_from_i64(n: i64) -> f32 {
    let magnitude = f32_from_u64(n.unsigned_abs());
    if n < 0 { -magnitude } else { magnitude }
}

const F64_FRAC_MASK: u64 = (1u64 << 52) - 1;
const F64_HIDDEN_BIT: u64 = 1u64 << 52;
const F32_INF_BITS: u32 = 0x7F80_0000;
/// Biased f64 exponents that map onto a *normal* f32: 2^-126 ..= 2^127.
const F32_NORMAL_MIN_BIASED64: u32 = 1023 - 126;
const F32_NORMAL_MAX_BIASED64: u32 = 1023 + 127;
/// f64 bias minus f32 bias.
const BIAS_SHIFT: u32 = 1023 - 127;

/// `x as f32`: round to nearest, ties to even.
///
/// Overflow goes to the signed infinity, subnormal results are rounded in
/// their own grid, and `NaN` stays `NaN` (quiet, payload truncated, exactly
/// as the x86 `cvtsd2ss` narrowing does).
#[inline]
#[must_use]
pub fn narrow(x: f64) -> f32 {
    let bits = x.to_bits();
    let sign = ((bits >> 63) as u32) << 31;
    let biased = u32::from(((bits >> 52) & 0x7FF) as u16);
    let frac = bits & F64_FRAC_MASK;
    if biased == 0x7FF {
        let payload = if frac == 0 {
            0
        } else {
            ((frac >> 29) as u32) | 0x0040_0000
        };
        return f32::from_bits(sign | F32_INF_BITS | payload);
    }
    if biased == 0 {
        // An f64 subnormal is below 2^-1022, far under f32's 2^-149 grid.
        return f32::from_bits(sign);
    }
    let mant = frac | F64_HIDDEN_BIT;
    if biased > F32_NORMAL_MAX_BIASED64 {
        return f32::from_bits(sign | F32_INF_BITS);
    }
    if biased < F32_NORMAL_MIN_BIASED64 {
        // Subnormal target: express the value in units of 2^-149, i.e. shift
        // the 53-bit mantissa (worth 2^(exp-52)) right by 29 + (-126 - exp).
        // Rounding up to exactly 2^23 lands on the smallest normal, whose
        // bit pattern is the same number, so no carry fix-up is needed.
        let shift = 29 + (F32_NORMAL_MIN_BIASED64 - biased);
        let m = round_half_even(mant, shift);
        return f32::from_bits(sign | (m & 0x00FF_FFFF) as u32);
    }
    let mut m = round_half_even(mant, 29);
    let mut exponent = biased - BIAS_SHIFT;
    if m == 1 << 24 {
        m = 1 << 23;
        exponent += 1;
        if exponent > F32_NORMAL_MAX_BIASED64 - BIAS_SHIFT {
            return f32::from_bits(sign | F32_INF_BITS);
        }
    }
    f32::from_bits(sign | (exponent << 23) | (m & 0x007F_FFFF) as u32)
}

/// `x as u64` for a float: truncate toward zero, saturate at both ends,
/// `NaN` → 0. An `f32` argument widens exactly first.
#[inline]
#[must_use]
pub fn trunc_u64(x: impl Into<f64>) -> u64 {
    let x: f64 = x.into();
    if x.is_nan() || x < 1.0 {
        return 0;
    }
    if x >= 18_446_744_073_709_551_616.0 {
        return u64::MAX;
    }
    let bits = x.to_bits();
    // 1 <= x < 2^64 pins the biased exponent to 1023..=1086.
    let exponent = u32::from(((bits >> 52) & 0x7FF) as u16) - 1023;
    let mant = (bits & F64_FRAC_MASK) | F64_HIDDEN_BIT;
    if exponent >= 52 {
        mant << (exponent - 52)
    } else {
        mant >> (52 - exponent)
    }
}

/// `x as i64` for a float: truncate toward zero, saturate, `NaN` → 0.
#[inline]
#[must_use]
pub fn trunc_i64(x: impl Into<f64>) -> i64 {
    let x: f64 = x.into();
    if x.is_nan() {
        return 0;
    }
    if x <= -9_223_372_036_854_775_808.0 {
        return i64::MIN;
    }
    if x >= 9_223_372_036_854_775_808.0 {
        return i64::MAX;
    }
    // |x| < 2^63 here, so the magnitude always fits.
    let magnitude = i64::try_from(trunc_u64(x.abs())).unwrap_or(i64::MAX);
    if x < 0.0 { -magnitude } else { magnitude }
}

/// `x as usize`; see [`trunc_u64`].
#[inline]
#[must_use]
pub fn trunc_usize(x: impl Into<f64>) -> usize {
    usize::try_from(trunc_u64(x)).unwrap_or(usize::MAX)
}

/// `x as u32`; see [`trunc_u64`].
#[inline]
#[must_use]
pub fn trunc_u32(x: impl Into<f64>) -> u32 {
    u32::try_from(trunc_u64(x)).unwrap_or(u32::MAX)
}

/// `x as u16`; see [`trunc_u64`].
#[inline]
#[must_use]
pub fn trunc_u16(x: impl Into<f64>) -> u16 {
    u16::try_from(trunc_u64(x)).unwrap_or(u16::MAX)
}

/// `x as u8`; see [`trunc_u64`].
#[inline]
#[must_use]
pub fn trunc_u8(x: impl Into<f64>) -> u8 {
    u8::try_from(trunc_u64(x)).unwrap_or(u8::MAX)
}

/// `x as isize`; see [`trunc_i64`].
#[inline]
#[must_use]
pub fn trunc_isize(x: impl Into<f64>) -> isize {
    let v = trunc_i64(x);
    isize::try_from(v).unwrap_or(if v < 0 { isize::MIN } else { isize::MAX })
}

/// `x as i32`; see [`trunc_i64`].
#[inline]
#[must_use]
pub fn trunc_i32(x: impl Into<f64>) -> i32 {
    let v = trunc_i64(x);
    i32::try_from(v).unwrap_or(if v < 0 { i32::MIN } else { i32::MAX })
}

/// `x as i16`; see [`trunc_i64`].
#[inline]
#[must_use]
pub fn trunc_i16(x: impl Into<f64>) -> i16 {
    let v = trunc_i64(x);
    i16::try_from(v).unwrap_or(if v < 0 { i16::MIN } else { i16::MAX })
}

/// `x as i8`; see [`trunc_i64`].
#[inline]
#[must_use]
pub fn trunc_i8(x: impl Into<f64>) -> i8 {
    let v = trunc_i64(x);
    i8::try_from(v).unwrap_or(if v < 0 { i8::MIN } else { i8::MAX })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PCG stream, so the sweeps are reproducible and dependency-free.
    struct Pcg(u64);

    impl Pcg {
        fn next_u64(&mut self) -> u64 {
            let old = self.0;
            self.0 = old
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let xorshifted = ((((old >> 18) ^ old) >> 27) & 0xFFFF_FFFF) as u32;
            let rot = (old >> 59) as u32;
            let word = xorshifted.rotate_right(rot);
            (u64::from(word) << 32) | u64::from(self.next_word())
        }

        fn next_word(&mut self) -> u32 {
            let old = self.0;
            self.0 = old
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let xorshifted = ((((old >> 18) ^ old) >> 27) & 0xFFFF_FFFF) as u32;
            let rot = (old >> 59) as u32;
            xorshifted.rotate_right(rot)
        }
    }

    /// Correctly rounded reference: the standard library's float parser.
    fn parse_f32<T: std::fmt::Display>(v: T) -> f32 {
        v.to_string().parse().expect("decimal round-trip")
    }

    fn parse_f64<T: std::fmt::Display>(v: T) -> f64 {
        v.to_string().parse().expect("decimal round-trip")
    }

    /// The exact decimal expansion of a finite double (1100 fractional
    /// digits cover 2^-1074), parsed back as a single, correctly rounded
    /// narrowing.
    fn narrow_reference(x: f64) -> f32 {
        format!("{x:.1100}").parse().expect("exact decimal parses")
    }

    fn assert_bits_f32(got: f32, want: f32, what: &str) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{what}: got {got:e} ({:#010x}), want {want:e} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }

    #[test]
    fn widening_u64_matches_the_decimal_round_trip() {
        let mut r = Pcg(7);
        let edges = [
            0u64,
            1,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 53) - 1,
            1 << 53,
            (1 << 53) + 1,
            (1 << 53) + 3,
            u64::MAX - 1,
            u64::MAX,
        ];
        for n in edges.into_iter().chain((0..20_000).map(|_| {
            let raw = r.next_u64();
            raw >> (r.next_word() % 64)
        })) {
            assert_eq!(f64_from_u64(n).to_bits(), parse_f64(n).to_bits(), "u64 {n}");
            assert_bits_f32(f32_from_u64(n), parse_f32(n), &format!("u64 {n}"));
            let i = i64::from_ne_bytes(n.to_ne_bytes());
            assert_eq!(f64_from_i64(i).to_bits(), parse_f64(i).to_bits(), "i64 {i}");
            assert_bits_f32(f32_from_i64(i), parse_f32(i), &format!("i64 {i}"));
        }
    }

    #[test]
    fn widening_u32_matches_the_decimal_round_trip() {
        let mut r = Pcg(11);
        let edges = [
            0u32,
            1,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            u32::MAX - 1,
            u32::MAX,
        ];
        for n in edges
            .into_iter()
            .chain((0..20_000).map(|_| r.next_word() >> (r.next_word() % 32)))
        {
            assert_bits_f32(f32_from_u32(n), parse_f32(n), &format!("u32 {n}"));
            let i = i32::from_ne_bytes(n.to_ne_bytes());
            assert_bits_f32(f32_from_i32(i), parse_f32(i), &format!("i32 {i}"));
        }
        assert_eq!(
            f64_from_usize(usize::MAX).to_bits(),
            parse_f64(usize::MAX).to_bits()
        );
        assert_bits_f32(f32_from_usize(12_345), 12_345.0, "usize");
        assert_eq!(f64_from_isize(-3).to_bits(), (-3.0f64).to_bits());
    }

    #[test]
    fn narrowing_matches_a_correctly_rounded_reference() {
        let mut r = Pcg(23);
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MIN_POSITIVE,
            f64::from(f32::MIN_POSITIVE),
            f64::from(f32::MAX),
            f64::from(f32::MAX) * (1.0 + 2f64.powi(-25)),
            f64::from(f32::MAX) * (1.0 + 2f64.powi(-24)),
            2f64.powi(128),
            2f64.powi(-149),
            2f64.powi(-150),
            2f64.powi(-150) * (1.0 + 2f64.powi(-40)),
            3.0 * 2f64.powi(-150),
            2f64.powi(-126),
            2f64.powi(-127) * 1.999_999_9,
            std::f64::consts::PI,
            1.0 / 3.0,
            0.1,
            1e-40,
            1e-45,
            1e38,
            3.402_823_5e38,
            f64::from(1.000_000_1f32) + 2f64.powi(-25),
            f64::from(1.000_000_1f32) + 2f64.powi(-25) + 2f64.powi(-60),
        ];
        for x in specials {
            for v in [x, -x] {
                assert_bits_f32(narrow(v), narrow_reference(v), &format!("narrow {v:e}"));
            }
        }
        for _ in 0..50_000 {
            let raw = r.next_u64();
            let x = f64::from_bits(raw);
            if x.is_nan() {
                continue;
            }
            assert_bits_f32(narrow(x), narrow_reference(x), &format!("narrow {x:e}"));
        }
        // Values shaped like an f32 with a fractional tail land in f32's own
        // range, where ties and carries live.
        for _ in 0..50_000 {
            let base = f64::from(f32::from_bits(r.next_word()));
            if !base.is_finite() {
                continue;
            }
            let tail = f64::from_bits(r.next_u64() >> 12) * 2f64.powi(-1074);
            let x = (base.abs() * 2f64.powi(-24)).mul_add(tail.fract() - 0.5, base);
            assert_bits_f32(narrow(x), narrow_reference(x), &format!("narrow {x:e}"));
        }
    }

    #[test]
    fn narrowing_keeps_infinities_and_quiet_nans() {
        assert_bits_f32(narrow(f64::INFINITY), f32::INFINITY, "+inf");
        assert_bits_f32(narrow(f64::NEG_INFINITY), f32::NEG_INFINITY, "-inf");
        assert!(narrow(f64::NAN).is_nan());
        assert!(narrow(-f64::NAN).is_nan());
        assert!(narrow(-f64::NAN).is_sign_negative());
        assert_bits_f32(narrow(2f64.powi(200)), f32::INFINITY, "overflow");
        assert_bits_f32(narrow(-2f64.powi(200)), f32::NEG_INFINITY, "overflow");
    }

    /// `x as u64` reference: the truncated value printed exactly, then read
    /// back as a big integer and clamped.
    fn trunc_reference(x: f64) -> i128 {
        if x.is_nan() {
            return 0;
        }
        let t = x.trunc();
        // Anything this far out saturates every target type; keep the decimal
        // within what an i128 parses.
        if t >= 1e30 {
            return i128::from(u64::MAX) + 1;
        }
        if t <= -1e30 {
            return -i128::from(u64::MAX) - 1;
        }
        format!("{t:.0}").parse().expect("integral decimal")
    }

    fn clamp<T: Copy + TryFrom<i128>>(v: i128, min: T, max: T) -> T
    where
        i128: From<T>,
    {
        if v < i128::from(min) {
            min
        } else if v > i128::from(max) {
            max
        } else {
            T::try_from(v).ok().expect("in range")
        }
    }

    #[test]
    fn truncation_matches_as_semantics() {
        let mut r = Pcg(29);
        let specials = [
            0.0,
            -0.0,
            0.5,
            -0.5,
            0.999_999_999,
            1.0,
            -1.0,
            1.5,
            -1.5,
            255.0,
            255.9,
            256.0,
            -128.0,
            -128.5,
            -129.0,
            65_535.5,
            4_294_967_295.5,
            4_294_967_296.0,
            2f64.powi(52) + 0.5,
            2f64.powi(53),
            2f64.powi(53) + 2.0,
            2f64.powi(63),
            2f64.powi(63) - 1024.0,
            -2f64.powi(63),
            -2f64.powi(63) - 2048.0,
            2f64.powi(64),
            2f64.powi(64) - 2048.0,
            2f64.powi(64) + 4096.0,
            1e300,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
        ];
        let randoms = (0..50_000).map(|_| {
            let raw = r.next_u64();
            // Bias toward integer-sized magnitudes.
            let x = f64::from_bits(raw);
            if r.next_word().is_multiple_of(2) {
                x
            } else {
                x % 1e20
            }
        });
        for x in specials.into_iter().chain(randoms) {
            let want = trunc_reference(x);
            assert_eq!(trunc_u64(x), clamp(want, 0u64, u64::MAX), "u64 {x:e}");
            assert_eq!(trunc_i64(x), clamp(want, i64::MIN, i64::MAX), "i64 {x:e}");
            assert_eq!(trunc_u32(x), clamp(want, 0u32, u32::MAX), "u32 {x:e}");
            assert_eq!(trunc_i32(x), clamp(want, i32::MIN, i32::MAX), "i32 {x:e}");
            assert_eq!(trunc_u16(x), clamp(want, 0u16, u16::MAX), "u16 {x:e}");
            assert_eq!(trunc_i16(x), clamp(want, i16::MIN, i16::MAX), "i16 {x:e}");
            assert_eq!(trunc_u8(x), clamp(want, 0u8, u8::MAX), "u8 {x:e}");
            assert_eq!(trunc_i8(x), clamp(want, i8::MIN, i8::MAX), "i8 {x:e}");
            assert_eq!(
                u64::try_from(trunc_usize(x)).expect("usize fits"),
                clamp(want, 0u64, usize::MAX as u64),
                "usize {x:e}"
            );
            assert_eq!(
                i64::try_from(trunc_isize(x)).expect("isize fits"),
                clamp(want, isize::MIN as i64, isize::MAX as i64),
                "isize {x:e}"
            );
        }
        // An f32 argument widens exactly first.
        assert_eq!(trunc_i32(-2.75f32), -2);
        assert_eq!(trunc_u8(300.0f32), 255);
    }
}
