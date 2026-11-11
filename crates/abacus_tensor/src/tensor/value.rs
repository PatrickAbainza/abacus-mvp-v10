//! Row 2 and row 3 (§B.2) + P2: the value set `V` as exact `i64` micro-units.
//!
//! `V = {-2^63, …, 2^63 - 1}` at scale `10^6` (P2). Value arithmetic is
//! **partial** (only `a_{02}`–`a_{05}` may touch the value path) and is
//! realized in [`crate::prims`]; this module owns the value discipline itself:
//! decimal-text canonicalization and the inverse text rendering.
//!
//! Row 2 decomposition: `a_{18}.1` trim / sign strip / split at the decimal
//! point → digit checks (`a_{01}` range tests joined by `a_{17}`) → integer part
//! by `d_horner` per digit → `a_{04}(x, 10^6)` → first six fractional digits
//! (`a_{09}` place weights, `R_3`) → `d_round` on the 7th digit → `a_{02}` →
//! `a_{05}` sign — all under `R_5`.
//!
//! Row 3 decomposition: `a_{08}.1`, `a_{08}.2` widened quotient / remainder and
//! `a_{08}.3` widened negation (so the minimum `i64` formats safely) →
//! trailing-zero strip (`a_{01}` + `R_3`) → `a_{18}.2` text assembly.

use crate::composites::{horner_digit, round_half_up};
use crate::prims::{add, mul, neg};
use crate::TensorError;

/// Minor-unit integer value (A2; `V` as exact `i64`).
pub type MinorUnits = i64;

/// Fixed decimal exponent: `value = decimal × 10^6` (P2 / §A2.1).
pub const MINOR_SCALE_EXP: u32 = 6;
/// Fixed decimal scale factor (P2 / §A2.1).
pub const MINOR_SCALE: i64 = 1_000_000;

/// Row 2 — decimal text → minor units (exact; `d_round` at the 7th digit).
///
/// Inputs with at most six fractional digits are exact; more are rounded
/// half-up away from zero by the single-digit test of `d_round`. Never
/// truncated, never float-converted, never wrapping.
///
/// # Errors
/// [`TensorError::Parse`] on empty/bad syntax; [`TensorError::Range`] when the
/// canonical value is not representable in `V` (P2).
pub fn parse_decimal_to_minor(s: &str) -> Result<MinorUnits, TensorError> {
    let t = s.trim();
    if t.is_empty() {
        return Err(TensorError::Parse("empty amount".to_string()));
    }
    let (negative, digits) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() {
        return Err(TensorError::Parse(format!("bad amount {s:?}")));
    }
    let (int_part, frac_part) = match digits.split_once('.') {
        Some((i, f)) => (i, f),
        None => (digits, ""),
    };
    let what = "amount parse (row 2)";
    if int_part.is_empty() || !int_part.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TensorError::Parse(format!("bad amount {s:?}")));
    }
    if !frac_part.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TensorError::Parse(format!("bad amount {s:?}")));
    }
    // Integer part: d_horner per digit, then the scale embedding a_04(x, 10^6).
    let mut acc: i64 = 0;
    for b in int_part.bytes() {
        acc = horner_digit(acc, b, what)?;
    }
    acc = mul(acc, MINOR_SCALE, what)?;
    // First six fractional digits: a_09 place weights under R_3.
    let mut frac: i64 = 0;
    let mut weight: i64 = 100_000;
    for b in frac_part.bytes().take(6) {
        frac += i64::from(b - b'0') * weight;
        weight /= 10;
    }
    let seventh = if frac_part.len() > 6 {
        frac_part.as_bytes().get(6).copied()
    } else {
        None
    };
    let micro = round_half_up(add(acc, frac, what)?, seventh, what)?;
    if negative {
        neg(micro, what)
    } else {
        Ok(micro)
    }
}

/// Row 3 — minor units → canonical decimal text (exact inverse form).
///
/// `a_{08}` widening guarantees the minimum `i64` formats without overflow;
/// trailing zeros are stripped by `a_{01}` + `R_3`; `a_{18}.2` assembles text.
#[must_use]
pub fn format_minor(v: MinorUnits) -> String {
    let negative = v < 0;
    let a: i128 = if negative {
        -i128::from(v)
    } else {
        i128::from(v)
    };
    let ip = a / i128::from(MINOR_SCALE);
    let fp = (a % i128::from(MINOR_SCALE)) as i64;
    if fp == 0 {
        return format!("{}{ip}.0", if negative { "-" } else { "" });
    }
    let mut s = format!("{fp:06}");
    while s.ends_with('0') {
        s.pop();
    }
    format!("{}{ip}.{s}", if negative { "-" } else { "" })
}
