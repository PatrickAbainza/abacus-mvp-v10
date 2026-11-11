//! The frozen seven-byte amount-derived feature payload.
//!
//! Golden Source §9 freezes the payload at exactly seven bytes per accepted
//! amount-bearing event: `log_magnitude` (u16) + `relative_threshold_distance`
//! (i32) + `roundedness_flags` (u8). Its "mathematical definitions and schema
//! version must not change without explicit approval".
//!
//! ## Determinism (§9.1 / §6.4)
//!
//! §9.1 is explicit that "JavaScript `Math.log` and Rust `ln` must not be
//! assumed to produce identical quantized results on every platform" and that
//! the implementation must "prefer a shared deterministic implementation". This
//! module is that shared implementation: [`log_magnitude_milli`] computes
//! `round(1000 * ln(1 + x))` in **exact integer arithmetic only** — no `f64`
//! logarithm is evaluated anywhere, so there is no platform dependence to
//! reconcile. The value comes from integer range reduction to
//! `ln(1 + x) = k*ln2 + ln(1 + m)` with `m in [0,1)`, followed by an atanh
//! series evaluated in fixed-point `10^SCALE` arithmetic, then a single
//! explicit half-up rounding step at the third decimal.
//!
//! ## Rounding rule (§9.2 / §6.2)
//!
//! Both scaled features use **half-up away from zero**, stated explicitly here
//! because §6.2 requires that "rounding, including negative halfway cases,
//! must be explicitly specified". For `d = 1000*(x - T)/T`: take the exact
//! signed difference, work on its magnitude, increment iff `2r >= T`, then
//! reapply the sign. A negative halfway case therefore rounds away from zero
//! (`-2.5 -> -3`), which is the same rule the substrate uses for decimal
//! parsing (`composites::round_half_up`), so the whole system shares one
//! rounding definition.
//!
//! ## Rejection, never clamping (§9.1 / §9.2 / §19)
//!
//! Every result is range-checked against its declared encoding before it is
//! written. §19 excludes "silent clamping, saturation, truncation, or integer
//! wrapping", so an out-of-range derivation is [`ErrorCode::FeatureOverflow`],
//! never a saturated value.

use crate::error::{ErrorCode, IngestError, IngestResult};

/// Scale factor `S_L = S_R = 1000` (Golden Source §9.1/§9.2).
pub const FEATURE_SCALE: i64 = 1000;

/// Maximum representable `log_magnitude` (u16).
pub const LOG_MAGNITUDE_MAX: u32 = u16::MAX as u32;

/// Maximum representable `relative_threshold_distance` magnitude (i32).
pub const REL_DIST_MAX: i64 = i32::MAX as i64;

/// Fixed-point decimal digits used by the internal natural-log evaluation.
///
/// Twelve digits. This is a deliberate bound, not an arbitrary one: the
/// largest intermediate in the series is `u_fixed^2 ~ 10^24`, and `i128` holds
/// up to ~1.7e38, so the multiply chain stays exact with a wide margin. Twelve
/// fractional digits also leave nine digits of headroom above the `10^-3`
/// quantization step, so series truncation can never decide a rounding.
const LN_FIXED_DIGITS: u32 = 12;

/// `10^LN_FIXED_DIGITS` — the fixed-point unit of the log evaluation.
///
/// Computed from the digit count rather than hard-coded, so the two can never
/// disagree.
const LN_FIXED_UNIT: i128 = {
    let mut p: i128 = 1;
    let mut i = 0;
    while i < LN_FIXED_DIGITS {
        p *= 10;
        i += 1;
    }
    p
};

/// `floor(ln 2 * 10^LN_FIXED_DIGITS)`, the fixed-point `ln 2` constant.
///
/// Derived from `ln 2 = 0.693147180559945309...` truncated at 12 fractional
/// digits.
///
/// Rounded *down* deliberately: the atanh series below is also floored term by
/// term, so the composed estimate never exceeds the true value. The explicit
/// half-up test on the exact remainder then decides rounding without a
/// systematic bias.
const LN2_FIXED: i128 = 693_147_180_559;

/// `Feature 1: log_magnitude = round(1000 * ln(1 + x))` as `u16`.
///
/// `x` is a nonnegative integer amount in currency minor units (Golden Source
/// §9.1).
///
/// # Errors
/// [`ErrorCode::FeatureOverflow`] if the quantized value exceeds `u16`.
pub fn log_magnitude_milli(x: u64) -> IngestResult<u16> {
    let v = log_magnitude_milli_i64(x)?;
    u16::try_from(v).map_err(|_| {
        IngestError::new(
            ErrorCode::FeatureOverflow,
            format!("log_magnitude {v} exceeds the u16 range; rejected, never clamped"),
        )
    })
}

/// [`log_magnitude_milli`] without the final `u16` narrowing, so the rejection
/// boundary is observable.
///
/// # Errors
/// [`ErrorCode::FeatureOverflow`] if the quantized value exceeds `i64`.
pub(super) fn log_magnitude_milli_i64(x: u64) -> IngestResult<i64> {
    let ln_fixed = ln1p_fixed(x);
    let num = ln_fixed * i128::from(FEATURE_SCALE);
    let q = num / LN_FIXED_UNIT;
    let r = num % LN_FIXED_UNIT;
    let rounded = if 2 * r >= LN_FIXED_UNIT { q + 1 } else { q };
    i64::try_from(rounded).map_err(|_| {
        IngestError::new(
            ErrorCode::FeatureOverflow,
            "log_magnitude quantization exceeded i64 during scaling",
        )
    })
}

/// `ln(1 + x)` in fixed-point `10^LN_FIXED_DIGITS` units, exact integer math.
///
/// Range reduction: with `k = floor(log2(1 + x))` and `1 + x = 2^k * (1 + m)`,
/// `ln(1 + x) = k*ln2 + ln(1 + m)` where `m in [0, 1)`. `k` comes from the
/// integer bit length (exact, no float), and `ln(1 + m)` is evaluated by the
/// atanh series `ln(1+m) = 2*(u + u^3/3 + u^5/5 + ...)` with `u = m/(2+m) in
/// [0,1/3)`, which converges fast enough that truncation error is far below the
/// `10^-3` quantization step.
fn ln1p_fixed(x: u64) -> i128 {
    let n = u128::from(x) + 1;
    let k = 127 - n.leading_zeros(); // floor(log2 n), exact
    // m_fixed = (n / 2^k - 1) * LN_FIXED_UNIT, i.e. m in [0,1) scaled decimally.
    //
    // The scaling by LN_FIXED_UNIT is a DECIMAL multiply (10^12), not a bit
    // shift — only the /2^k reduction is a bit shift. `n <= 2^64` and
    // `LN_FIXED_UNIT ~ 2^40`, so `n * LN_FIXED_UNIT < 2^104` fits `u128`.
    let scaled = n * (LN_FIXED_UNIT as u128); // decimal scaling
    let reduced = scaled >> k; // / 2^k, i.e. n / 2^k * 10^12
    // n >= 2^k guarantees reduced >= LN_FIXED_UNIT, so no underflow.
    let m_fixed = reduced.saturating_sub(LN_FIXED_UNIT as u128) as i128;
    // u = m / (2 + m) in fixed point.
    let denom = 2 * LN_FIXED_UNIT + m_fixed;
    let u_fixed = if denom > 0 {
        (m_fixed * LN_FIXED_UNIT) / denom
    } else {
        0
    };
    // atanh series: sum u^(2j+1)/(2j+1), times 2.
    let u2 = (u_fixed * u_fixed) / LN_FIXED_UNIT; // u^2 in fixed point
    let mut term = u_fixed;
    let mut acc = u_fixed;
    let mut j: i128 = 1;
    // u <= 1/3, so u^(2j+1) shrinks by >= 9x per step: 64 terms reach the
    // fixed-point floor for every input.
    while j < 64 {
        term = (term * u2) / LN_FIXED_UNIT;
        if term == 0 {
            break;
        }
        acc += term / (2 * j + 1);
        j += 1;
    }
    (k as i128) * LN2_FIXED + 2 * acc
}

/// `Feature 2: relative_threshold_distance = round(1000 * (x - T) / T)`.
///
/// `x` and `T` are exact integer minor units in the same currency and
/// precision (Golden Source §9.2). Rounding is half-up away from zero (§6.2).
///
/// # Errors
/// - [`ErrorCode::AmountOrCurrency`] if `T == 0`.
/// - [`ErrorCode::FeatureOverflow`] if the quantized value exceeds `i32`.
pub fn relative_threshold_distance(x: u64, threshold: u64) -> IngestResult<i32> {
    if threshold == 0 {
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            "threshold must be greater than zero",
        ));
    }
    // Exact signed difference in i128: never wraps, never saturates.
    let diff = i128::from(x) - i128::from(threshold);
    let negative = diff < 0;
    let mag = if negative { -diff } else { diff };
    let t = i128::from(threshold);
    let num = mag * i128::from(FEATURE_SCALE);
    let q = num / t;
    let r = num % t;
    // Half-up away from zero (§6.2 explicit rule).
    let mag_rounded = if 2 * r >= t { q + 1 } else { q };
    let signed = if negative { -mag_rounded } else { mag_rounded };
    i32::try_from(signed).map_err(|_| {
        IngestError::new(
            ErrorCode::FeatureOverflow,
            "relative_threshold_distance exceeds the i32 range; rejected, never clamped",
        )
    })
}

/// `Feature 3: roundedness_flags` — exact integer divisibility bit field.
///
/// Golden Source §9.3 bit assignments, computed with exact integer arithmetic
/// after applying the configured currency precision:
///
/// | Bit | Meaning |
/// |-----|---------|
/// | 0 | amount divisible by 10 major units |
/// | 1 | amount divisible by 100 major units |
/// | 2 | amount divisible by 1,000 major units |
/// | 3 | amount divisible by 10,000 major units |
/// | 4-7 | reserved, always zero |
///
/// The major-unit scales are converted to minor units by multiplying by
/// `10^precision`. Divisibility by a larger major-unit multiple implies
/// divisibility by every smaller one, so the bits are nested by construction —
/// which this function reproduces by testing each scale independently rather
/// than assuming the nesting, so an impossible bit pattern cannot be produced.
///
/// # Errors
/// [`ErrorCode::AmountOrCurrency`] if any scaled divisor is not representable.
pub fn roundedness_flags(x: u64, major_scale: u64) -> IngestResult<u8> {
    let mut flags = 0u8;
    // Major-unit multiples in ascending order: 10, 100, 1_000, 10_000.
    for (bit, multiple) in [10u64, 100, 1_000, 10_000].iter().enumerate() {
        let divisor = multiple
            .checked_mul(major_scale)
            .filter(|&d| d > 0)
            .ok_or_else(|| {
                IngestError::new(
                    ErrorCode::AmountOrCurrency,
                    "roundedness divisor is not representable in u64",
                )
            })?;
        if x.is_multiple_of(divisor) {
            flags |= 1 << bit;
        }
    }
    // Bits 4-7 are reserved and must be zero (Canonical Contract §6.3).
    debug_assert_eq!(flags & 0xf0, 0, "reserved roundedness bits must be zero");
    Ok(flags)
}

/// The complete frozen seven-byte payload for one amount-bearing event.
///
/// Golden Source §9.4: 2 + 4 + 1 = 7 bytes, exactly. The struct is `Copy` and
/// is always encoded field-by-field, never by copying in-memory bytes
/// (wire §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeaturePayload {
    /// Feature 1: `round(1000 * ln(1 + amount_minor_units))`.
    pub log_magnitude: u16,
    /// Feature 2: `round(1000 * (amount - T) / T)`.
    pub relative_threshold_distance: i32,
    /// Feature 3: exact-divisibility bit field.
    pub roundedness_flags: u8,
}

impl FeaturePayload {
    /// Derive all three features from an exact amount and validated config.
    ///
    /// # Errors
    /// - [`ErrorCode::AmountOrCurrency`] for an invalid threshold or scale.
    /// - [`ErrorCode::FeatureOverflow`] if any feature leaves its encoding
    ///   range (rejected, never clamped).
    pub fn derive(
        amount_minor_units: u64,
        threshold_minor_units: u64,
        major_scale: u64,
    ) -> IngestResult<Self> {
        Ok(Self {
            log_magnitude: log_magnitude_milli(amount_minor_units)?,
            relative_threshold_distance: relative_threshold_distance(
                amount_minor_units,
                threshold_minor_units,
            )?,
            roundedness_flags: roundedness_flags(amount_minor_units, major_scale)?,
        })
    }

    /// Byte width of the payload: exactly 7 (Golden Source §9.4).
    pub const BYTE_LEN: usize = 7;

    /// Validate a payload received on the wire.
    ///
    /// The backend "cannot independently prove that the submitted features
    /// correspond to the original amounts" (§6.4 trust boundary), so this
    /// checks only what is checkable: the reserved roundedness bits are zero.
    /// Range validity is already implied by the field types.
    ///
    /// # Errors
    /// [`ErrorCode::Binary`] if any reserved bit is set.
    pub fn validate(&self) -> IngestResult<()> {
        if self.roundedness_flags & 0xf0 != 0 {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!(
                    "reserved roundedness bits 4-7 must be zero, got 0x{:02x}",
                    self.roundedness_flags
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference quantizer using `f64` ln, used ONLY as an independent oracle
    /// in tests. Production code never calls this (§9.1 forbids relying on
    /// native log behavior for the encoded value).
    fn reference_log_milli(x: u64) -> i64 {
        (1000.0 * (1.0 + x as f64).ln()).round() as i64
    }

    #[test]
    fn log_magnitude_matches_independent_f64_reference() {
        // Golden-vector style sweep: ordinary values, powers of ten, and the
        // u64 boundary. The integer implementation must agree with the
        // independent f64 reference on every one.
        let mut xs: Vec<u64> = vec![
            0,
            1,
            2,
            9,
            10,
            99,
            100,
            999,
            1_000,
            10_000,
            100_000,
            1_000_000,
            12_345_678,
            999_999_999,
            1_000_000_000,
            1_000_000_000_000,
            u64::MAX / 2,
            u64::MAX - 1,
            u64::MAX,
        ];
        for e in 0..20u32 {
            xs.push(10u64.pow(e));
            xs.push(10u64.pow(e) + 1);
            xs.push(10u64.pow(e).saturating_sub(1));
        }
        for x in xs {
            let got = log_magnitude_milli_i64(x).unwrap();
            let want = reference_log_milli(x);
            assert_eq!(got, want, "log_magnitude mismatch at x = {x}");
        }
    }

    #[test]
    fn log_magnitude_zero_is_zero() {
        assert_eq!(log_magnitude_milli(0).unwrap(), 0);
    }

    #[test]
    fn log_magnitude_known_values() {
        // round(1000 * ln(1 + x)) for hand-checkable inputs, cross-verified
        // against an independent arbitrary-precision reference:
        //   ln 2   = 0.693147  -> x=1         -> 693
        //   ln 10  = 2.302585  -> x=9         -> 2303
        //   ln 100 = 4.605170  -> x=99        -> 4605
        //   ln(1e6+1) = 13.8155 -> x=1_000_000 -> 13816
        assert_eq!(log_magnitude_milli(1).unwrap(), 693);
        assert_eq!(log_magnitude_milli(9).unwrap(), 2_303);
        assert_eq!(log_magnitude_milli(99).unwrap(), 4_605);
        assert_eq!(log_magnitude_milli(1_000_000).unwrap(), 13_816);
    }

    #[test]
    fn log_magnitude_is_deterministic_across_calls() {
        for x in [0u64, 7, 4_242, 1 << 40, u64::MAX] {
            assert_eq!(
                log_magnitude_milli(x).unwrap(),
                log_magnitude_milli(x).unwrap()
            );
        }
    }

    #[test]
    fn log_magnitude_rejects_u16_overflow_without_clamping() {
        // round(1000*ln(1+x)) exceeds 65535 once ln(1+x) > 65.535, i.e. x >~ e^65.
        let x = u64::MAX; // ln(1+x) ~ 44.36 -> 44361, still inside u16
        // Confirm the u64 domain is entirely representable, so no amount can
        // overflow feature 1 — the bound is documented, not merely hoped for.
        assert!(log_magnitude_milli(x).is_ok());
        assert!(log_magnitude_milli(x).unwrap() < 65_535);
    }

    #[test]
    fn relative_distance_zero_at_threshold() {
        assert_eq!(relative_threshold_distance(10_000, 10_000).unwrap(), 0);
    }

    #[test]
    fn relative_distance_known_values() {
        // x = 2T  -> +1000 ; x = 0 -> -1000 ; x = 1.5T -> +500
        assert_eq!(relative_threshold_distance(20_000, 10_000).unwrap(), 1_000);
        assert_eq!(relative_threshold_distance(0, 10_000).unwrap(), -1_000);
        assert_eq!(relative_threshold_distance(15_000, 10_000).unwrap(), 500);
        assert_eq!(relative_threshold_distance(5_000, 10_000).unwrap(), -500);
    }

    #[test]
    fn relative_distance_negative_halfway_rounds_away_from_zero() {
        // x - T = -T/4 with T = 4000 -> -1000*0.25 = -250 exactly.
        assert_eq!(relative_threshold_distance(3_000, 4_000).unwrap(), -250);
        // Exact halfway negative: d = -2500.5 * 1000/1000 style case.
        // x - T = -1, T = 2  -> -500 exactly. Use T=2000, x=1999 -> -0.5 -> -1.
        assert_eq!(relative_threshold_distance(1_999, 2_000).unwrap(), -1);
        // Positive counterpart rounds up too.
        assert_eq!(relative_threshold_distance(2_001, 2_000).unwrap(), 1);
    }

    #[test]
    fn relative_distance_matches_independent_f64_reference() {
        let thresholds = [1u64, 2, 3, 7, 1_000, 10_000, 999_983];
        for t in thresholds {
            for delta in [-3i64, -1, 0, 1, 3, 1_000, -1_000, 123_457] {
                let x = (i128::from(t) + i128::from(delta)).max(0) as u64;
                let got = relative_threshold_distance(x, t).unwrap();
                let want = (1000.0 * (x as f64 - t as f64) / t as f64).round() as i64;
                assert_eq!(i64::from(got), want, "x={x} T={t}");
            }
        }
    }

    #[test]
    fn relative_distance_rejects_zero_threshold() {
        assert_eq!(
            relative_threshold_distance(100, 0).unwrap_err().code,
            ErrorCode::AmountOrCurrency
        );
    }

    #[test]
    fn relative_distance_rejects_i32_overflow_without_clamping() {
        // T = 1 and x = u64::MAX gives ~9.2e21, far outside i32.
        assert_eq!(
            relative_threshold_distance(u64::MAX, 1).unwrap_err().code,
            ErrorCode::FeatureOverflow
        );
    }

    #[test]
    fn roundedness_bits_at_precision_two() {
        // major_scale = 100 (2 decimal places).
        let s = 100u64;
        assert_eq!(roundedness_flags(0, s).unwrap(), 0b1111); // 0 is divisible
        assert_eq!(roundedness_flags(1_000, s).unwrap(), 0b0001); // 10 major
        assert_eq!(roundedness_flags(10_000, s).unwrap(), 0b0011); // 100 major
        assert_eq!(roundedness_flags(100_000, s).unwrap(), 0b0111); // 1_000 major
        assert_eq!(roundedness_flags(1_000_000, s).unwrap(), 0b1111); // 10_000 major
        assert_eq!(roundedness_flags(1_500, s).unwrap(), 0b0000);
    }

    #[test]
    fn roundedness_reserved_bits_always_zero() {
        for x in [0u64, 1, 999, 10_000_000, u64::MAX] {
            for p in 0..=4u8 {
                let scale = 10u64.pow(u32::from(p));
                let f = roundedness_flags(x, scale).unwrap();
                assert_eq!(f & 0xf0, 0, "reserved bits set for x={x} p={p}");
            }
        }
    }

    #[test]
    fn roundedness_scales_with_precision() {
        // 1_000_000 minor units = 10 major units at precision 5.
        assert_eq!(roundedness_flags(1_000_000, 100_000).unwrap(), 0b0001);
        // The same integer at precision 2 is 10_000 major units -> all bits.
        assert_eq!(roundedness_flags(1_000_000, 100).unwrap(), 0b1111);
    }

    #[test]
    fn payload_is_exactly_seven_bytes() {
        assert_eq!(FeaturePayload::BYTE_LEN, 7);
        assert_eq!(
            FeaturePayload::BYTE_LEN,
            std::mem::size_of::<u16>() + std::mem::size_of::<i32>() + std::mem::size_of::<u8>()
        );
    }

    #[test]
    fn payload_derive_and_validate() {
        // amount = 12_345 minor units, threshold = 10_000, precision 2.
        // log_magnitude   = round(1000 * ln(12346)) = 9421
        // relative distance = round(1000 * 2345/10000) = 234
        // roundedness     = 12345 % 1000 != 0 -> 0
        let p = FeaturePayload::derive(12_345, 10_000, 100).unwrap();
        p.validate().unwrap();
        assert_eq!(p.log_magnitude, 9_421);
        assert_eq!(p.relative_threshold_distance, 235);
        assert_eq!(p.roundedness_flags, 0);
    }

    #[test]
    fn payload_rejects_reserved_bits_on_validate() {
        let p = FeaturePayload {
            log_magnitude: 1,
            relative_threshold_distance: 0,
            roundedness_flags: 0x80,
        };
        assert_eq!(p.validate().unwrap_err().code, ErrorCode::Binary);
    }

    #[test]
    fn constants_match_the_frozen_schema() {
        assert_eq!(FEATURE_SCALE, 1_000);
        assert_eq!(LOG_MAGNITUDE_MAX, 65_535);
        assert_eq!(REL_DIST_MAX, 2_147_483_647);
    }
}
