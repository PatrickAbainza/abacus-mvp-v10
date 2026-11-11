//! Timestamp normalization to the canonical signed-i64 UTC milliseconds.
//!
//! Canonical Contract §5 / Golden Source §7:
//!
//! - "Accept ISO-8601 timestamps with explicit UTC offsets."
//! - "Require an explicit source timezone when timestamps have no timezone."
//! - "Require explicit date-format selection for ambiguous date strings."
//! - "Require an explicit unit for numeric timestamps, such as seconds or
//!   milliseconds."
//! - "Reject impossible dates and out-of-range values."
//! - "Do not silently discard sub-millisecond precision."
//! - "Keep event time distinct from ingestion time."
//!
//! ## No timezone database, no locale guessing
//!
//! The only offset information used is either the offset written in the value
//! itself or the single offset the user declared for the file. Named zones
//! (`CET`, `EST`) are **not** accepted: they are ambiguous, they change meaning
//! across DST boundaries, and honouring them would need a zone database this
//! crate does not have. A named zone is therefore a rejection that names the
//! problem, not a silent reinterpretation.
//!
//! ## Calendar arithmetic
//!
//! Days-since-epoch uses the standard `days_from_civil` algorithm (Howard
//! Hinnant's formula), which is exact integer arithmetic including the Gregorian
//! leap rules. That is why `2024-02-29` is accepted and `2023-02-29` is rejected
//! with no lookup table.

use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};

use crate::policy::{EpochUnit, NumberPolicy, TimestampPolicy, TimestampStyle};

/// Milliseconds in one second.
const MS_PER_SECOND: i64 = 1_000;

/// Normalize one timestamp field to UTC milliseconds since the Unix epoch.
///
/// # Errors
/// [`ErrorCode::Timestamp`] for an unparseable value, an impossible date, a
/// missing offset under a style that requires one, a named timezone, a
/// fractional part under `RejectSubMilli`, or a value outside the `i64`
/// millisecond range.
pub fn parse_timestamp(
    text: &str,
    policy: TimestampPolicy,
    numbers: NumberPolicy,
) -> IngestResult<i64> {
    let s = text.trim();
    if s.is_empty() {
        return Err(IngestError::new(
            ErrorCode::Timestamp,
            "the timestamp field is empty",
        ));
    }
    match policy.style {
        TimestampStyle::NumericEpoch => parse_numeric_epoch(s, numbers),
        TimestampStyle::Iso8601WithOffset => {
            if !has_offset(s) {
                // §7: an offset-less value needs an explicit source timezone.
                return Err(IngestError::new(
                    ErrorCode::Timestamp,
                    "an ISO-8601 timestamp without a UTC offset requires an explicit \
                     source timezone; configure the offset-less style with a source offset",
                ));
            }
            parse_iso8601(s, None, numbers)
        }
        TimestampStyle::Iso8601WithSourceOffset => {
            let off = policy.source_offset_minutes.ok_or_else(|| {
                IngestError::new(
                    ErrorCode::Timestamp,
                    "the plan declares offset-less timestamps but states no source timezone",
                )
            })?;
            parse_iso8601(s, Some(off), numbers)
        }
    }
}

/// True iff `s` carries an explicit UTC offset or a `Z` designator.
fn has_offset(s: &str) -> bool {
    s.ends_with('Z') || s.ends_with('z') || trailing_offset_minutes(s).is_some()
}

/// Parse an all-digits epoch value under the declared unit.
fn parse_numeric_epoch(s: &str, numbers: NumberPolicy) -> IngestResult<i64> {
    let digits: String = s.chars().filter(|c| *c != '_' && *c != ' ').collect();
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(IngestError::new(
            ErrorCode::Timestamp,
            "a numeric epoch timestamp must be an unsigned integer under the declared unit",
        ));
    }
    // Parse as u128 first so a value beyond i64 milliseconds is a rejection
    // rather than a wrap, then convert under the declared unit.
    let v: u128 = digits.parse().map_err(|_| {
        IngestError::new(
            ErrorCode::Timestamp,
            "the numeric epoch timestamp is out of range",
        )
    })?;
    i64::try_from(v)
        .ok()
        .and_then(|n| match numbers.epoch_unit {
            EpochUnit::Seconds => n.checked_mul(MS_PER_SECOND),
            EpochUnit::Millis => Some(n),
        })
        .ok_or_else(|| {
            IngestError::new(
                ErrorCode::Timestamp,
                "the numeric epoch timestamp does not fit the i64 millisecond range",
            )
        })
}

/// Parse an ISO-8601 date/time, applying `source_offset` when the value has none.
fn parse_iso8601(s: &str, source_offset: Option<i32>, numbers: NumberPolicy) -> IngestResult<i64> {
    // Reject named zones outright rather than guessing their offset.
    const ZONES: [&str; 8] = [
        " CET", " CEST", " EST", " EDT", " PST", " PDT", " GMT", "UTC",
    ];
    for zone in ZONES {
        if s.contains(zone) {
            return Err(IngestError::new(
                ErrorCode::Timestamp,
                "a named timezone is ambiguous across DST boundaries; use an explicit \
                 numeric UTC offset or declare a source timezone for the file",
            ));
        }
    }
    let (datetime, offset) = split_offset(s, source_offset)?;
    let (date_part, time_part) = match datetime.split_once(['T', 't', ' ']) {
        Some((d, t)) => (d, Some(t)),
        None => (datetime.as_str(), None),
    };
    let (y, m, d) = parse_date(date_part)?;
    let (hh, mm, ss, frac_ms) = match time_part {
        Some(t) => parse_time(t, numbers.subsecond)?,
        None => (0, 0, 0, 0),
    };
    let secs = days_from_civil(y, m, d)
        .checked_mul(86_400)
        .and_then(|x| x.checked_add(i64::from(hh) * 3_600 + i64::from(mm) * 60 + i64::from(ss)))
        .ok_or_else(|| {
            IngestError::new(
                ErrorCode::Timestamp,
                "the timestamp is outside the representable range",
            )
        })?;
    secs.checked_mul(MS_PER_SECOND)
        .and_then(|x| x.checked_add(i64::from(frac_ms)))
        .and_then(|x| x.checked_sub(i64::from(offset) * 60 * MS_PER_SECOND))
        .ok_or_else(|| {
            IngestError::new(
                ErrorCode::Timestamp,
                "the timestamp does not fit the i64 millisecond range",
            )
        })
}

/// The trailing `+HH:MM` / `-HHMM` / `+HH` offset in minutes, if present.
///
/// The search starts **after** the date/time separator, because a date's own
/// dashes (`2026-10-12`) must never be mistaken for an offset — ISO 8601 places
/// the offset after the time-of-day, and only there.
fn trailing_offset_minutes(s: &str) -> Option<i32> {
    let start = s.find(['T', 't', ' ']).map(|i| i + 1).unwrap_or(s.len());
    let bytes = s.as_bytes();
    for i in (start.max(1)..bytes.len()).rev() {
        let c = bytes[i];
        if c != b'+' && c != b'-' {
            continue;
        }
        let tail = &s[i + 1..];
        if tail.is_empty() || tail.len() > 5 {
            return None;
        }
        let colons = tail.matches(':').count();
        let digits: String = tail.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() + colons != tail.len() || !(2..=4).contains(&digits.len()) {
            return None;
        }
        let (hh, mm) = match digits.len() {
            2 => (digits.parse::<i32>().ok()?, 0),
            _ => (
                digits[..2].parse::<i32>().ok()?,
                digits[2..].parse::<i32>().ok()?,
            ),
        };
        if !(0..=23).contains(&hh) || !(0..=59).contains(&mm) {
            return None;
        }
        let sign = if c == b'-' { -1 } else { 1 };
        return Some(sign * (hh * 60 + mm));
    }
    None
}

/// Split a trailing UTC offset off a timestamp, or apply the source offset.
///
/// Returns `(datetime_without_offset, offset_minutes)`.
fn split_offset(s: &str, source_offset: Option<i32>) -> IngestResult<(String, i32)> {
    if s.ends_with('Z') || s.ends_with('z') {
        return Ok((s[..s.len() - 1].to_string(), 0));
    }
    if let Some(off) = trailing_offset_minutes(s) {
        let cut = s
            .char_indices()
            .rev()
            .find(|(_, c)| *c == '+' || *c == '-')
            .map(|(i, _)| i)
            .ok_or_else(|| {
                IngestError::new(
                    ErrorCode::Timestamp,
                    "the timestamp's UTC offset could not be located",
                )
            })?;
        return Ok((s[..cut].to_string(), off));
    }
    match source_offset {
        Some(off) => Ok((s.to_string(), off)),
        None => Err(IngestError::new(
            ErrorCode::Timestamp,
            "an ISO-8601 timestamp without a UTC offset requires an explicit source timezone",
        )),
    }
}

/// Parse an ISO calendar date `YYYY-MM-DD` (or `YYYY/MM/DD`).
///
/// # Errors
/// [`ErrorCode::Timestamp`] for a malformed date, an out-of-range month or day,
/// or an impossible day for that month and year (including leap-year rules).
fn parse_date(s: &str) -> IngestResult<(i64, u32, u32)> {
    let norm = s.replace('/', "-");
    let mut it = norm.splitn(3, '-');
    let (y, m, d) = match (it.next(), it.next(), it.next()) {
        (Some(y), Some(m), Some(d)) => (y, m, d),
        _ => {
            return Err(IngestError::new(
                ErrorCode::Timestamp,
                "a date must be in YYYY-MM-DD form",
            ))
        }
    };
    let year: i64 = y
        .parse()
        .map_err(|_| IngestError::new(ErrorCode::Timestamp, "the date's year is not an integer"))?;
    let month: u32 = m.parse().map_err(|_| {
        IngestError::new(ErrorCode::Timestamp, "the date's month is not an integer")
    })?;
    let day: u32 = d
        .parse()
        .map_err(|_| IngestError::new(ErrorCode::Timestamp, "the date's day is not an integer"))?;
    if !(1..=12).contains(&month) {
        return Err(IngestError::new(
            ErrorCode::Timestamp,
            format!("month {month} is not a calendar month"),
        ));
    }
    let max_day = days_in_month(year, month);
    if day == 0 || day > max_day {
        return Err(IngestError::new(
            ErrorCode::Timestamp,
            format!("day {day} is impossible for month {month} of {year}"),
        ));
    }
    Ok((year, month, day))
}

/// Days in a Gregorian month, honouring the leap-year rule exactly.
fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// The Gregorian leap-year rule.
fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Parse a time part `HH:MM[:SS[.frac]]` or the compact `HHMMSS[.frac]` form.
///
/// Returns `(hours, minutes, seconds, milliseconds)`.
fn parse_time(
    t: &str,
    subsecond: crate::policy::SubSecondPolicy,
) -> IngestResult<(u32, u32, u32, u32)> {
    let (main, frac) = match t.split_once(['.', ',']) {
        Some((m, f)) => (m, Some(f)),
        None => (t, None),
    };
    // Two accepted layouts, and only these two: the extended `HH:MM[:SS]` form
    // and the ISO basic `HHMMSS` form. Anything else is rejected rather than
    // reinterpreted, because guessing a layout would silently change the instant.
    let (hh, mm, ss) = if main.contains(':') {
        let mut it = main.splitn(3, ':');
        let hh = parse_part(it.next().unwrap_or(""), "hour")?;
        let mm = parse_part(it.next().unwrap_or("0"), "minute")?;
        let ss = match it.next() {
            Some(s) if !s.is_empty() => parse_part(s, "second")?,
            _ => 0,
        };
        (hh, mm, ss)
    } else {
        // Basic form: exactly 6 digits is HHMMSS, exactly 4 is HHMM.
        if main.len() != 6 && main.len() != 4 {
            return Err(IngestError::new(
                ErrorCode::Timestamp,
                "a time without ':' separators must be the basic HHMMSS or HHMM form",
            ));
        }
        let hh = parse_part(&main[0..2], "hour")?;
        let mm = parse_part(&main[2..4], "minute")?;
        let ss = if main.len() == 6 {
            parse_part(&main[4..6], "second")?
        } else {
            0
        };
        (hh, mm, ss)
    };
    if hh > 23 || mm > 59 || ss > 59 {
        return Err(IngestError::new(
            ErrorCode::Timestamp,
            "the time of day is out of range (a leap second is not representable here)",
        ));
    }
    let mut ms = 0u32;
    if let Some(f) = frac {
        if f.is_empty() || !f.chars().all(|c| c.is_ascii_digit()) {
            return Err(IngestError::new(
                ErrorCode::Timestamp,
                "the fractional second is not a digit string",
            ));
        }
        // Canonical Contract §5: never silently discard sub-millisecond
        // precision — either reject it or declare the conversion rule.
        if f.len() > 3 {
            match subsecond {
                crate::policy::SubSecondPolicy::RejectSubMilli => {
                    return Err(IngestError::new(
                        ErrorCode::Timestamp,
                        format!(
                            "the timestamp carries {n} fractional digits, which is finer than \
                             the canonical millisecond; the declared policy rejects it rather \
                             than discarding precision",
                            n = f.len()
                        ),
                    ))
                }
                crate::policy::SubSecondPolicy::TruncateToMilli => {
                    ms = f[..3].parse().unwrap_or(0);
                }
            }
        } else {
            // Pad to exactly three digits: ".5" is 500 ms, ".05" is 50 ms.
            // Written out rather than with a format spec, because the fill
            // character and alignment are the whole point here and a `{:0<3}`
            // is easy to misread as a left-pad.
            let mut padded = String::with_capacity(3);
            padded.push_str(&f[..f.len().min(3)]);
            while padded.len() < 3 {
                padded.push('0');
            }
            ms = padded.parse().unwrap_or(0);
        }
    }
    Ok((hh, mm, ss, ms))
}

/// Parse one numeric time component, rejecting a non-integer or empty value.
fn parse_part(s: &str, what: &str) -> IngestResult<u32> {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_digit()) {
        return Err(IngestError::new(
            ErrorCode::Timestamp,
            format!("the {what} component of the timestamp is not an unsigned integer"),
        ));
    }
    s.parse().map_err(|_| {
        IngestError::new(
            ErrorCode::Timestamp,
            format!("the {what} component is out of range"),
        )
    })
}

/// Days from 1970-01-01 to the given civil date (Howard Hinnant's algorithm).
///
/// Exact integer arithmetic, valid for the whole proleptic Gregorian calendar,
/// and negative for dates before the epoch.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (i64::from(m) + 9) % 12; // March = 0
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{EpochUnit, SubSecondPolicy};

    fn nums(unit: EpochUnit, sub: SubSecondPolicy) -> NumberPolicy {
        NumberPolicy {
            epoch_unit: unit,
            subsecond: sub,
        }
    }

    fn offset_style() -> TimestampPolicy {
        TimestampPolicy {
            style: TimestampStyle::Iso8601WithOffset,
            source_offset_minutes: None,
        }
    }

    fn source_style(minutes: i32) -> TimestampPolicy {
        TimestampPolicy {
            style: TimestampStyle::Iso8601WithSourceOffset,
            source_offset_minutes: Some(minutes),
        }
    }

    fn numeric_style() -> TimestampPolicy {
        TimestampPolicy {
            style: TimestampStyle::NumericEpoch,
            source_offset_minutes: None,
        }
    }

    const REJECT: SubSecondPolicy = SubSecondPolicy::RejectSubMilli;
    const TRUNC: SubSecondPolicy = SubSecondPolicy::TruncateToMilli;

    #[test]
    fn z_suffix_is_utc() {
        let ms = parse_timestamp(
            "2026-10-12T09:30:00Z",
            offset_style(),
            nums(EpochUnit::Millis, REJECT),
        )
        .unwrap();
        assert_eq!(ms, (20_738 * 86_400 + 34_200) * 1_000);
    }

    #[test]
    fn explicit_offsets_normalize_to_the_same_instant() {
        let n = nums(EpochUnit::Millis, REJECT);
        let z = parse_timestamp("2026-10-12T09:30:00Z", offset_style(), n).unwrap();
        let plus = parse_timestamp("2026-10-12T11:30:00+02:00", offset_style(), n).unwrap();
        let minus = parse_timestamp("2026-10-12T04:30:00-05:00", offset_style(), n).unwrap();
        let compact = parse_timestamp("2026-10-12T113000+0200", offset_style(), n).unwrap();
        assert_eq!(z, plus);
        assert_eq!(z, minus);
        assert_eq!(z, compact);
    }

    #[test]
    fn an_offset_less_value_is_rejected_under_the_offset_style() {
        let e = parse_timestamp(
            "2026-10-12T09:30:00",
            offset_style(),
            nums(EpochUnit::Millis, REJECT),
        )
        .unwrap_err();
        assert_eq!(e.code, ErrorCode::Timestamp);
        assert!(e.detail.contains("explicit source timezone"));
    }

    #[test]
    fn an_explicit_source_offset_is_applied() {
        let n = nums(EpochUnit::Millis, REJECT);
        let ms = parse_timestamp("2026-10-12T09:30:00", source_style(120), n).unwrap();
        assert_eq!(ms, (20_738 * 86_400 + 7 * 3_600 + 30 * 60) * 1_000);
        let ms2 = parse_timestamp("2026-10-12T09:30:00", source_style(-300), n).unwrap();
        assert_eq!(ms2, (20_738 * 86_400 + 14 * 3_600 + 30 * 60) * 1_000);
    }

    #[test]
    fn numeric_epoch_requires_the_declared_unit() {
        let secs = parse_timestamp(
            "1700000000",
            numeric_style(),
            nums(EpochUnit::Seconds, REJECT),
        )
        .unwrap();
        let millis = parse_timestamp(
            "1700000000",
            numeric_style(),
            nums(EpochUnit::Millis, REJECT),
        )
        .unwrap();
        assert_eq!(secs, 1_700_000_000_000);
        assert_eq!(millis, 1_700_000_000);
    }

    #[test]
    fn a_non_integer_epoch_is_rejected() {
        for bad in ["1700000000.5", "1.7e9", "abc", "-1"] {
            let r = parse_timestamp(bad, numeric_style(), nums(EpochUnit::Seconds, REJECT));
            assert!(r.is_err(), "'{bad}' must be rejected");
        }
    }

    #[test]
    fn impossible_dates_are_rejected() {
        let n = nums(EpochUnit::Millis, REJECT);
        for bad in [
            "2023-02-29T00:00:00Z",
            "2026-13-01T00:00:00Z",
            "2026-00-10T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-10-32T00:00:00Z",
            "2026-10-12T24:00:00Z",
            "2026-10-12T09:60:00Z",
        ] {
            assert!(
                parse_timestamp(bad, offset_style(), n).is_err(),
                "'{bad}' must be rejected"
            );
        }
        assert!(parse_timestamp("2024-02-29T00:00:00Z", offset_style(), n).is_ok());
    }

    #[test]
    fn sub_millisecond_precision_is_rejected_unless_declared() {
        let rej = parse_timestamp(
            "2026-10-12T09:30:00.1234Z",
            offset_style(),
            nums(EpochUnit::Millis, REJECT),
        );
        assert_eq!(rej.unwrap_err().code, ErrorCode::Timestamp);
        let trunc = parse_timestamp(
            "2026-10-12T09:30:00.1234Z",
            offset_style(),
            nums(EpochUnit::Millis, TRUNC),
        )
        .unwrap();
        let plain = parse_timestamp(
            "2026-10-12T09:30:00.123Z",
            offset_style(),
            nums(EpochUnit::Millis, REJECT),
        )
        .unwrap();
        assert_eq!(trunc, plain, "truncation must drop only the 4th digit");
    }

    #[test]
    fn millisecond_fractions_pad_correctly() {
        let n = nums(EpochUnit::Millis, REJECT);
        let a = parse_timestamp("2026-10-12T09:30:00.5Z", offset_style(), n).unwrap();
        let b = parse_timestamp("2026-10-12T09:30:00.500Z", offset_style(), n).unwrap();
        assert_eq!(a, b);
        assert!(a % 1_000 == 500);
    }

    #[test]
    fn date_only_values_are_midnight() {
        let n = nums(EpochUnit::Millis, REJECT);
        let ms = parse_timestamp("2026-10-12", source_style(0), n).unwrap();
        assert_eq!(ms, 20_738 * 86_400 * 1_000);
    }

    #[test]
    fn days_from_civil_matches_known_epoch_values() {
        // Values cross-checked against an independent proleptic-Gregorian
        // day-number calculation.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(days_from_civil(2026, 10, 12), 20_738);
    }

    #[test]
    fn leap_century_rules_are_exact() {
        assert!(!is_leap(1900));
        assert!(is_leap(2000));
        assert!(is_leap(2024));
        assert!(!is_leap(2023));
    }
}
