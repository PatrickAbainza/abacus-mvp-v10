//! Exact amount parsing: a decimal string to integer minor units, with no
//! floating point anywhere.
//!
//! Golden Source §8: "The browser parses the source amount as an exact decimal
//! string or exact integer minor-unit value. **Binary floating-point must not be
//! the authoritative representation of money.**" and "If an amount cannot be
//! parsed exactly or falls outside the supported range, reject or quarantine the
//! row under the approved policy. Do not invent a feature value."
//!
//! So this parser is pure integer arithmetic and produces
//! `minor = int_part * 10^precision + frac_part * 10^(precision - frac_len)`.
//! Every multiply and add is `checked`, so an amount too large for `u64` is a
//! rejection rather than a wrap (§19). A fraction longer than the currency's
//! precision is **rejected**, never rounded: rounding an amount is inventing a
//! value (§8: "Do not invent a feature value"), and it would change which
//! roundedness bit the frozen feature derivation sets (§9.3), so the features
//! would no longer reproduce from the attested amount at commit time.
//!
//! ## The locale is a policy, not a guess
//!
//! `1.234,56` and `1,234.56` are the same number in different locales and
//! different numbers if the separators are swapped. [`AmountFormat`] names the
//! characters, and the parser refuses a field it cannot read under that policy.

use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};

/// Parse one amount field into exact minor units under `fmt`.
///
/// Also takes the plan's [`AmountFormat`] (re-exported here for the public
/// signature); the concrete type lives in [`crate::policy`].
///
/// # Parameters
/// - `text` — the raw field, exactly as the CSV carried it.
/// - `fmt` — the declared locale policy.
/// - `precision` — the currency's minor-unit digits.
/// - `ceiling` — the configured supported amount ceiling in minor units (§8).
///
/// # Errors
/// [`ErrorCode::AmountOrCurrency`] for a malformed number, a misplaced separator,
/// a fraction longer than `precision`, a negative value, or an amount above
/// `ceiling`. Every one of these rejects the row; nothing is coerced.
pub fn parse_amount(
    text: &str,
    fmt: crate::policy::AmountFormat,
    precision: u8,
    ceiling: u64,
) -> IngestResult<u64> {
    let s = text.trim();
    if s.is_empty() {
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            "the amount field is empty",
        ));
    }
    let mut negative = false;
    if s.starts_with('-') {
        if !fmt.accept_negative {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "a negative amount is not permitted under the declared amount policy; \
                 the canonical amount is a nonnegative minor-unit count",
            ));
        }
        negative = true;
    }
    // A second sign, or a trailing sign, is malformed rather than ignorable.
    let body = s
        .strip_prefix('-')
        .or_else(|| s.strip_prefix('+'))
        .unwrap_or(s);
    if body.contains('-') || body.contains('+') {
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            "the amount field has a misplaced sign",
        ));
    }
    if negative {
        // §9.1: x is nonnegative, and the feature derivation is defined only on
        // nonnegative minor units. `accept_negative` therefore controls whether
        // a signed input is *recognised* (and reported as a policy rejection) or
        // *rejected as unparseable*; it never makes a negative canonical.
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            "a negative amount has no canonical representation; the feature derivation \
             is defined on nonnegative minor units only",
        ));
    }

    // Split on the declared decimal separator. At most one is legal.
    let dec = fmt.decimal_separator as char;
    if s.matches(dec).count() > 1 {
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            "the amount field has more than one decimal separator",
        ));
    }
    let mut parts = s.splitn(2, dec);
    let int_part = parts.next().unwrap_or("");
    let frac_part = parts.next().unwrap_or("");

    let int_units = parse_grouped(int_part, fmt, "amount")?;
    if frac_part.len() > usize::from(precision) {
        // Reject, never round: rounding would make the frozen features
        // unreproducible from the attested amount (§8, §9.3).
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            format!(
                "the amount has {n} fractional digits but the currency declares {precision}; \
                 excess precision is rejected, never rounded",
                n = frac_part.len()
            ),
        ));
    }
    let frac_units = parse_plain(frac_part, "amount fraction")?;
    let shift = u32::from(precision).saturating_sub(u32::try_from(frac_part.len()).unwrap_or(0));
    let major_scale = 10_u64.checked_pow(u32::from(precision)).ok_or_else(|| {
        IngestError::new(
            ErrorCode::AmountOrCurrency,
            "the currency precision scale is not representable",
        )
    })?;
    let frac_scale = 10_u64.checked_pow(shift).ok_or_else(|| {
        IngestError::new(
            ErrorCode::AmountOrCurrency,
            "the currency precision scale is not representable",
        )
    })?;
    let value = int_units
        .checked_mul(major_scale)
        .and_then(|a| {
            frac_units
                .checked_mul(frac_scale)
                .and_then(|b| a.checked_add(b))
        })
        .ok_or_else(|| {
            IngestError::new(
                ErrorCode::AmountOrCurrency,
                "the amount exceeds the representable u64 minor-unit range",
            )
        })?;
    if value > ceiling {
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            format!("the amount exceeds the configured {ceiling}-minor-unit supported ceiling"),
        ));
    }
    Ok(value)
}

/// Parse a digit run that may contain the declared grouping separator.
///
/// Grouping is validated, not merely stripped: `1,23,456` is malformed under a
/// `,`-every-three-digits locale and is rejected, because silently accepting it
/// would let a typo change an amount's magnitude (§19: "Never silently clamp,
/// wrap, truncate, or reinterpret values").
fn parse_grouped(s: &str, fmt: crate::policy::AmountFormat, what: &str) -> IngestResult<u64> {
    let Some(group) = fmt.group_separator else {
        return parse_plain(s, what);
    };
    let g = group as char;
    if !s.contains(g) {
        return parse_plain(s, what);
    }
    if s.starts_with(g) || s.ends_with(g) {
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            format!("the {what} has a misplaced grouping separator"),
        ));
    }
    let groups: Vec<&str> = s.split(g).collect();
    if groups[0].is_empty() || groups[0].len() > 3 {
        return Err(IngestError::new(
            ErrorCode::AmountOrCurrency,
            format!("the {what} has a malformed leading digit group"),
        ));
    }
    for grp in &groups[1..] {
        if grp.len() != 3 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                format!("the {what} has a digit group that is not exactly three digits"),
            ));
        }
    }
    parse_plain(&groups.concat(), what)
}

/// Parse a pure ASCII digit run into `u64` with checked arithmetic.
fn parse_plain(s: &str, what: &str) -> IngestResult<u64> {
    if s.is_empty() {
        return Ok(0);
    }
    let mut acc: u64 = 0;
    for c in s.chars() {
        let d = c.to_digit(10).ok_or_else(|| {
            IngestError::new(
                ErrorCode::AmountOrCurrency,
                format!("the {what} contains a non-digit character"),
            )
        })?;
        acc = acc
            .checked_mul(10)
            .and_then(|a| a.checked_add(u64::from(d)))
            .ok_or_else(|| {
                IngestError::new(
                    ErrorCode::AmountOrCurrency,
                    format!("the {what} exceeds the representable u64 range"),
                )
            })?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::AmountFormat;

    /// en-US: `1,234.56`, precision 2.
    fn us() -> AmountFormat {
        AmountFormat {
            decimal_separator: b'.',
            group_separator: Some(b','),
            accept_negative: false,
        }
    }

    const CEIL: u64 = 1_000_000_000_000;

    #[test]
    fn parses_plain_and_grouped_amounts_exactly() {
        assert_eq!(parse_amount("12345", us(), 2, CEIL).unwrap(), 1_234_500);
        assert_eq!(parse_amount("123.45", us(), 2, CEIL).unwrap(), 12_345);
        assert_eq!(parse_amount("1,234.56", us(), 2, CEIL).unwrap(), 123_456);
        assert_eq!(
            parse_amount("1,234,567.89", us(), 2, CEIL).unwrap(),
            123_456_789
        );
        assert_eq!(parse_amount("0.29", us(), 2, CEIL).unwrap(), 29);
    }

    #[test]
    fn excess_fractional_precision_is_rejected_not_rounded() {
        let e = parse_amount("10.005", us(), 2, CEIL).unwrap_err();
        assert_eq!(e.code, ErrorCode::AmountOrCurrency);
        assert!(e.detail.contains("never rounded"));
    }

    #[test]
    fn negatives_are_rejected_by_default_and_when_opted_in() {
        let e = parse_amount("-5.00", us(), 2, CEIL).unwrap_err();
        assert_eq!(e.code, ErrorCode::AmountOrCurrency);
        let opt_in = AmountFormat {
            accept_negative: true,
            ..us()
        };
        assert_eq!(
            parse_amount("-5.00", opt_in, 2, CEIL).unwrap_err().code,
            ErrorCode::AmountOrCurrency
        );
    }

    #[test]
    fn malformed_numbers_are_rejected() {
        for bad in [
            "", "  ", "abc", "1.2.3", "1,,2", ",123", "123,", "1-2", "+", "-",
        ] {
            assert!(
                parse_amount(bad, us(), 2, CEIL).is_err(),
                "'{bad}' must be rejected"
            );
        }
    }

    #[test]
    fn malformed_grouping_is_rejected() {
        for bad in ["1,23,456.00", "12,34.00", "1234,56.00"] {
            assert!(
                parse_amount(bad, us(), 2, CEIL).is_err(),
                "'{bad}' must be rejected"
            );
        }
    }

    #[test]
    fn the_ceiling_is_a_rejection_not_a_clamp() {
        // `max_amount_minor_units` is the largest ACCEPTED amount: the server
        // pre-flight rejects only `a_u > max` (session.rs), so the client
        // parser must match — an amount exactly at the ceiling parses.
        assert_eq!(parse_amount("10.00", us(), 2, 1_000).unwrap(), 1_000);
        let e = parse_amount("10.01", us(), 2, 1_000).unwrap_err();
        assert_eq!(e.code, ErrorCode::AmountOrCurrency);
        assert!(e.detail.contains("ceiling"));
        assert_eq!(parse_amount("999.99", us(), 2, 100_000).unwrap(), 99_999);
    }

    #[test]
    fn a_huge_amount_is_a_rejection_not_a_wrap() {
        let e = parse_amount("99999999999999999999.99", us(), 2, u64::MAX).unwrap_err();
        assert_eq!(e.code, ErrorCode::AmountOrCurrency);
    }

    #[test]
    fn no_grouping_policy_rejects_grouped_input() {
        let none = AmountFormat {
            decimal_separator: b'.',
            group_separator: None,
            accept_negative: false,
        };
        assert!(parse_amount("1,234.56", none, 2, CEIL).is_err());
        assert_eq!(parse_amount("1234.56", none, 2, CEIL).unwrap(), 123_456);
    }

    #[test]
    fn zero_is_a_valid_amount() {
        assert_eq!(parse_amount("0", us(), 2, CEIL).unwrap(), 0);
        assert_eq!(parse_amount("0.00", us(), 2, CEIL).unwrap(), 0);
    }
}
