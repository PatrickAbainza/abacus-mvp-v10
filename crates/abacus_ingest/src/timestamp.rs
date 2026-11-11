//! Timestamp handling: wire milliseconds to substrate seconds.
//!
//! ## The two representations
//!
//! - The wire contract (`timestamp_encoding = 1`) is a **signed i64 of UTC Unix
//!   milliseconds** (wire §2), and Canonical Contract §5 makes `event_time_ms`
//!   "a signed 64-bit integer representing milliseconds since
//!   1970-01-01T00:00:00Z".
//! - The existing substrate coordinate is a **u64 of seconds** since the Unix
//!   epoch (`abacus_tensor::tensor::coordinate::TimeSec`). Its own date
//!   derivation rejects pre-epoch input outright
//!   (`composites::epoch_seconds` -> "pre-epoch date").
//!
//! These are different units AND different signedness, so a conversion is
//! unavoidable. Canonical Contract §5 forbids discarding precision silently:
//! "Do not silently discard sub-millisecond precision. Either reject excess
//! precision or apply a separately versioned conversion rule."
//!
//! ## The rule applied here
//!
//! This module implements the **versioned conversion rule** branch, and it is
//! the smallest possible lossy step:
//!
//! [`TIMESTAMP_CONVERSION_VERSION`] = 1, and the conversion is
//! `t_seconds = floor(event_time_ms / 1000)` — Euclidean floor division, so
//! `-1 ms` maps to `-1 s` before the pre-epoch check rejects it, never to `0`.
//! Two consequences are explicit and tested:
//!
//! 1. **Negative (pre-epoch) times are rejected**, because the substrate's `T`
//!    set is `u64` seconds since epoch and its own parser already rejects
//!    pre-epoch dates. Accepting them would require inventing a representation
//!    the substrate does not have.
//! 2. **Sub-second precision IS discarded**, by this one declared step. That is
//!    a real limitation and it is recorded as such in the completion report
//!    rather than hidden: the substrate has no sub-second coordinate, so no
//!    implementation of this ingestion path can preserve it without changing
//!    the substrate's declared coordinate type — which Golden Source §19
//!    forbids ("Unapproved changes to SAMCORE02 v3", "No silent changes to
//!    existing binary formats").
//!
//! ## Event time vs ingestion time
//!
//! Golden Source §7 requires that "event time" and "ingestion time" stay
//! distinct. Only event time passes through this module; ingestion time is
//! never written into a coordinate.

use crate::error::{ErrorCode, IngestError, IngestResult};

/// Version of the millisecond-to-second conversion rule (Canonical Contract §5).
///
/// Bump this whenever the conversion changes, so a report can state which rule
/// produced a given artifact.
pub const TIMESTAMP_CONVERSION_VERSION: u32 = 1;

/// Milliseconds per second.
const MS_PER_SECOND: i64 = 1_000;

/// Convert a wire `event_time_ms` to the substrate's seconds coordinate.
///
/// Rule: `t_seconds = floor(event_time_ms / 1000)`, Euclidean floor so the
/// pre-epoch check sees the true sign.
///
/// # Errors
/// - [`ErrorCode::Timestamp`] for a pre-epoch (negative) time, which the
///   substrate's `u64`-seconds coordinate cannot represent.
/// - [`ErrorCode::Timestamp`] if the floored value exceeds `u64`.
pub fn event_time_ms_to_seconds(event_time_ms: i64) -> IngestResult<u64> {
    let secs = event_time_ms.div_euclid(MS_PER_SECOND);
    if secs < 0 {
        return Err(IngestError::new(
            ErrorCode::Timestamp,
            "pre-epoch event time is not representable in the substrate's u64-seconds coordinate",
        ));
    }
    u64::try_from(secs).map_err(|_| {
        IngestError::new(
            ErrorCode::Timestamp,
            "event time exceeds the representable u64-seconds range",
        )
    })
}

/// True iff the conversion discards sub-second precision for this value.
///
/// Reported per session so the loss is visible in the ingestion report rather
/// than implied.
#[must_use]
pub fn discards_subsecond(event_time_ms: i64) -> bool {
    event_time_ms.rem_euclid(MS_PER_SECOND) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_seconds_pass_through() {
        assert_eq!(event_time_ms_to_seconds(0).unwrap(), 0);
        assert_eq!(event_time_ms_to_seconds(1_000).unwrap(), 1);
        assert_eq!(event_time_ms_to_seconds(1_700_000_000_000).unwrap(), 1_700_000_000);
        // floor(i64::MAX / 1000) = 9_223_372_036_854_775_807 / 1000
        assert_eq!(event_time_ms_to_seconds(i64::MAX).unwrap(), 9_223_372_036_854_775);
    }

    #[test]
    fn subsecond_is_floored_not_rounded() {
        // Floor, never round: 1_999 ms -> 1 s (not 2 s).
        assert_eq!(event_time_ms_to_seconds(1_999).unwrap(), 1);
        assert_eq!(event_time_ms_to_seconds(1).unwrap(), 0);
        assert_eq!(event_time_ms_to_seconds(999).unwrap(), 0);
    }

    #[test]
    fn pre_epoch_is_rejected_not_wrapped_to_zero() {
        // The critical case: naive integer division would map -1 to 0.
        assert_eq!(
            event_time_ms_to_seconds(-1).unwrap_err().code,
            ErrorCode::Timestamp
        );
        assert_eq!(
            event_time_ms_to_seconds(-1_000).unwrap_err().code,
            ErrorCode::Timestamp
        );
        assert_eq!(
            event_time_ms_to_seconds(i64::MIN).unwrap_err().code,
            ErrorCode::Timestamp
        );
    }

    #[test]
    fn subsecond_discard_is_detectable_and_reported() {
        assert!(!discards_subsecond(1_000));
        assert!(!discards_subsecond(0));
        assert!(discards_subsecond(1));
        assert!(discards_subsecond(1_700_000_000_123));
    }

    #[test]
    fn conversion_version_is_declared() {
        assert_eq!(TIMESTAMP_CONVERSION_VERSION, 1);
    }
}
