//! P11 diagnostic math (addendum §S5): sign-preserving log1p,
//! population z-score inputs, whole-unit roundedness.
//! NEVER on the value path; f64 presentation only (K10 class).

/// Sign-preserving log1p of a major-unit amount (total function).
/// `ln(1+A)` for `A >= 0`, else `-ln(1+|A|)`.
#[must_use]
pub fn log1p_signed(major: f64) -> f64 {
    if major >= 0.0 {
        (1.0 + major).ln()
    } else {
        -((1.0 + (-major)).ln())
    }
}

/// Whole-major-unit test on canonical micro-units (exact, no float).
#[must_use]
pub fn roundedness_of(amount_micro: i64) -> u8 {
    u8::from(amount_micro % 1_000_000 == 0)
}

/// Two-pass population mean/std over major-unit amounts.
/// Deterministic fold order = slice order. Returns `(mean, std)`.
#[must_use]
pub fn mean_std_major(amounts_major: &[f64]) -> (f64, f64) {
    if amounts_major.is_empty() {
        return (0.0, 0.0);
    }
    let n = amounts_major.len() as f64;
    let mut sum = 0.0f64;
    for &a in amounts_major {
        sum += a;
    }
    let mean = sum / n;
    let mut sq = 0.0f64;
    for &a in amounts_major {
        let d = a - mean;
        sq += d * d;
    }
    (mean, (sq / n).sqrt())
}

/// z-score with zero-std guard (total function).
#[must_use]
pub fn z_of(v: f64, mean: f64, std: f64) -> f64 {
    if std > 0.0 {
        (v - mean) / std
    } else {
        0.0
    }
}
