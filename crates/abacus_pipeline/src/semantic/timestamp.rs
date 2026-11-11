//! Timestamp translation utilities for the semantic layer.
//!
//! Provides conversions between Unix epoch seconds (the canonical internal
//! representation) and ISO 8601 UTC timestamps (the external, human-readable
//! representation).
//!
//! All conversions are timezone-aware: explicit offsets are honored, `Z`
//! denotes UTC, and ambiguous local times are never silently interpreted using
//! the machine's local timezone.

use chrono::{FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc, Datelike};

/// Supported date range: years 1970–2100 (covers realistic banking data).
const MIN_YEAR: i32 = 1970;
const MAX_YEAR: i32 = 2100;

/// Human-readable ISO 8601 UTC timestamp string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtcTimestamp(pub String);

impl UtcTimestamp {
    /// Format as an ISO 8601 UTC string (`YYYY-MM-DDTHH:MM:SSZ`).
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for UtcTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Error returned by timestamp translation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimestampError {
    /// The input string could not be parsed as a supported ISO 8601 format.
    InvalidFormat(String),
    /// The timestamp is outside the supported range (years 1970–2100).
    OutOfRange(String),
    /// A timezone offset was malformed (e.g. `+25:00` or `+08:60`).
    InvalidOffset(String),
    /// The input was ambiguous (a naive local time without timezone).
    AmbiguousLocalTime(String),
}

impl std::fmt::Display for TimestampError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(s) => write!(f, "invalid timestamp format: {s}"),
            Self::OutOfRange(s) => write!(f, "timestamp out of range: {s}"),
            Self::InvalidOffset(s) => write!(f, "invalid timezone offset: {s}"),
            Self::AmbiguousLocalTime(s) => {
                write!(f, "ambiguous local time (no timezone specified): {s}")
            }
        }
    }
}

impl std::error::Error for TimestampError {}

/// Convert Unix epoch seconds to an ISO 8601 UTC timestamp.
///
/// # Errors
/// Returns [`TimestampError::OutOfRange`] if the seconds value falls outside
/// the supported date range.
pub fn to_utc_iso8601(secs: u64) -> Result<UtcTimestamp, TimestampError> {
    let dt = Utc
        .timestamp_opt(secs as i64, 0)
        .single()
        .ok_or_else(|| TimestampError::OutOfRange(secs.to_string()))?;

    if dt.year() < MIN_YEAR || dt.year() > MAX_YEAR {
        return Err(TimestampError::OutOfRange(secs.to_string()));
    }

    let formatted = dt.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    Ok(UtcTimestamp(formatted))
}

/// Parse an ISO 8601 timestamp string into Unix epoch seconds.
///
/// Accepts:
/// - `YYYY-MM-DDTHH:MM:SSZ` (UTC)
/// - `YYYY-MM-DDTHH:MM:SS+HH:MM` (explicit positive offset)
/// - `YYYY-MM-DDTHH:MM:SS-HH:MM` (explicit negative offset)
///
/// Rejects naive timestamps without a timezone designator.
///
/// # Errors
/// Returns [`TimestampError`] for invalid formats, malformed offsets,
/// out-of-range values, or ambiguous local times.
pub fn from_iso8601(s: &str) -> Result<u64, TimestampError> {
    let s = s.trim();

    // Must contain a 'T' separator.
    let Some(t_pos) = s.find('T') else {
        return Err(TimestampError::InvalidFormat(s.to_string()));
    };

    let date_part = &s[..t_pos];
    let time_part = &s[t_pos + 1..];

    // Parse the date: YYYY-MM-DD
    let date_parts: Vec<&str> = date_part.split('-').collect();
    if date_parts.len() != 3 {
        return Err(TimestampError::InvalidFormat(s.to_string()));
    }
    let year: i32 = date_parts[0]
        .parse()
        .map_err(|_| TimestampError::InvalidFormat(s.to_string()))?;
    let month: u32 = date_parts[1]
        .parse()
        .map_err(|_| TimestampError::InvalidFormat(s.to_string()))?;
    let day: u32 = date_parts[2]
        .parse()
        .map_err(|_| TimestampError::InvalidFormat(s.to_string()))?;

    // Determine timezone designator and split time from offset.
    // Look for Z at the end, or +/− offset separator after HH:MM:SS.
    let (time_without_tz, offset_seconds) = if time_part.ends_with('Z') {
        // UTC designator.
        let time_str = time_part.strip_suffix('Z').unwrap_or(time_part);
        (time_str, 0)
    } else {
        // Look for the last '+' or '-' that starts the offset.
        // The offset separator comes after HH:MM:SS, so find the last
        // '+' or '-' in the time part.
        let offset_start = time_part
            .rfind(|c: char| c == '+' || c == '-')
            .ok_or_else(|| TimestampError::AmbiguousLocalTime(s.to_string()))?;
        let time_str = &time_part[..offset_start];
        let offset_str = &time_part[offset_start + 1..];
        let offset_secs = parse_offset(offset_str)?;
        let sign = if time_part.as_bytes()[offset_start] == b'+' {
            1
        } else {
            -1
        };
        (time_str, sign * offset_secs)
    };

    // Parse the time: HH:MM:SS
    let time_parts: Vec<&str> = time_without_tz.split(':').collect();
    if time_parts.len() != 3 {
        return Err(TimestampError::InvalidFormat(s.to_string()));
    }
    let hour: u32 = time_parts[0]
        .parse()
        .map_err(|_| TimestampError::InvalidFormat(s.to_string()))?;
    let minute: u32 = time_parts[1]
        .parse()
        .map_err(|_| TimestampError::InvalidFormat(s.to_string()))?;
    let second: u32 = time_parts[2]
        .parse()
        .map_err(|_| TimestampError::InvalidFormat(s.to_string()))?;

    if year < MIN_YEAR || year > MAX_YEAR {
        return Err(TimestampError::OutOfRange(s.to_string()));
    }
    if month == 0 || month > 12 {
        return Err(TimestampError::InvalidFormat(s.to_string()));
    }
    if day == 0 || day > 31 {
        return Err(TimestampError::InvalidFormat(s.to_string()));
    }
    if hour > 23 || minute > 59 || second > 59 {
        return Err(TimestampError::InvalidFormat(s.to_string()));
    }

    let naive_date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| TimestampError::InvalidFormat(s.to_string()))?;
    let naive_time = NaiveTime::from_hms_opt(hour, minute, second)
        .ok_or_else(|| TimestampError::InvalidFormat(s.to_string()))?;
    let naive_dt = NaiveDateTime::new(naive_date, naive_time);

    let utc_dt = match FixedOffset::east_opt(offset_seconds) {
        Some(offset) => {
            // Interpret naive_dt as local time in the given offset, then
            // convert to UTC.
            let local_dt = offset
                .from_local_datetime(&naive_dt)
                .single()
                .ok_or_else(|| TimestampError::InvalidFormat(s.to_string()))?;
            local_dt.with_timezone(&Utc)
        }
        None => return Err(TimestampError::InvalidOffset(offset_seconds.to_string())),
    };

    let secs = utc_dt.timestamp();
    if secs < 0 {
        return Err(TimestampError::OutOfRange(s.to_string()));
    }
    Ok(secs as u64)
}

/// Parse a timezone offset string like `+08:00` or `-04:30` into seconds east of UTC.
fn parse_offset(offset: &str) -> Result<i32, TimestampError> {
    let parts: Vec<&str> = offset.split(':').collect();
    if parts.len() != 2 {
        return Err(TimestampError::InvalidOffset(offset.to_string()));
    }
    let hours: i32 = parts[0]
        .parse()
        .map_err(|_| TimestampError::InvalidOffset(offset.to_string()))?;
    let minutes: i32 = parts[1]
        .parse()
        .map_err(|_| TimestampError::InvalidOffset(offset.to_string()))?;
    if hours > 23 || minutes > 59 {
        return Err(TimestampError::InvalidOffset(offset.to_string()));
    }
    Ok(hours * 3600 + minutes * 60)
}

/// Format Unix epoch seconds as a human-readable UTC timestamp string.
pub fn format_utc(secs: u64) -> Result<String, TimestampError> {
    Ok(to_utc_iso8601(secs)?.0)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_seconds_to_utc_iso8601() {
        let ts = to_utc_iso8601(1_792_152_060).unwrap();
        assert_eq!(ts.as_str(), "2026-10-16T12:01:00Z");
    }

    #[test]
    fn utc_iso8601_to_epoch_seconds() {
        let secs = from_iso8601("2026-10-16T02:01:00Z").unwrap();
        assert_eq!(secs, 1_792_116_060);
    }

    #[test]
    fn timezone_offset_to_epoch_seconds() {
        let secs = from_iso8601("2026-10-16T10:01:00+08:00").unwrap();
        assert_eq!(secs, 1_792_116_060);
    }

    #[test]
    fn negative_timezone_offset_to_epoch_seconds() {
        let secs = from_iso8601("2026-10-16T02:01:00-04:00").unwrap();
        assert_eq!(secs, 1_792_130_460);
    }

    #[test]
    fn equivalent_instants_different_timezones() {
        let utc_secs = from_iso8601("2026-10-16T02:01:00Z").unwrap();
        let plus8_secs = from_iso8601("2026-10-16T10:01:00+08:00").unwrap();
        let minus4_correct = from_iso8601("2026-10-15T22:01:00-04:00").unwrap();
        assert_eq!(utc_secs, plus8_secs);
        assert_eq!(utc_secs, minus4_correct);
    }

    #[test]
    fn roundtrip_epoch_to_iso_to_epoch() {
        let original: u64 = 1_700_000_000;
        let iso = to_utc_iso8601(original).unwrap();
        let back = from_iso8601(iso.as_str()).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn rejects_naive_local_time_without_timezone() {
        let err = from_iso8601("2026-10-16T10:01:00").unwrap_err();
        assert!(matches!(err, TimestampError::AmbiguousLocalTime { .. }));
    }

    #[test]
    fn rejects_invalid_offset_hours() {
        let err = from_iso8601("2026-10-16T10:01:00+25:00").unwrap_err();
        assert!(matches!(err, TimestampError::InvalidOffset { .. }));
    }

    #[test]
    fn rejects_invalid_offset_minutes() {
        let err = from_iso8601("2026-10-16T10:01:00+08:60").unwrap_err();
        assert!(matches!(err, TimestampError::InvalidOffset { .. }));
    }

    #[test]
    fn rejects_out_of_range_year() {
        let err = from_iso8601("1899-10-16T10:01:00Z").unwrap_err();
        assert!(matches!(err, TimestampError::OutOfRange { .. }));
    }

    #[test]
    fn rejects_malformed_format() {
        let err = from_iso8601("not-a-timestamp").unwrap_err();
        assert!(matches!(err, TimestampError::InvalidFormat { .. }));
    }

    #[test]
    fn rejects_out_of_range_epoch_seconds() {
        let err = to_utc_iso8601(32_503_680_000).unwrap_err();
        assert!(matches!(err, TimestampError::OutOfRange { .. }));
    }
}