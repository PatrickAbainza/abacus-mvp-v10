//! Contract A — Internal Rust semantic translation layer.
//!
//! Provides typed, in-memory interfaces for semantic tensor operations that
//! internal callers can invoke directly without HTTP, JSON, or transport
//! encodings. All operations delegate to the canonical [`IngestPipeline`] read
//! boundary, backed by `abacus_tensor` (sealed CSF + frontier overlay).
//! No algebra is added, no storage structures are exposed, and no JSON values
//! are constructed for internal calls.

use crate::ingest_model::FeaturePayload;
use crate::ingest_pipeline::IngestPipeline;
use crate::semantic::timestamp::{from_iso8601, to_utc_iso8601, TimestampError};

pub mod timestamp;

/// A typed semantic operation request. Internal callers construct these
/// directly and pass them to [`SemanticLayer`]. No HTTP types appear here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticOp {
    /// Exact coordinate lookup: `lookup(t, s, r)`.
    Lookup { t: u64, s: u32, r: u32 },
    /// Time-window retrieval over half-open `[t_start, t_end)`.
    TimeSlice { t_start: u64, t_end: u64 },
    /// Outgoing relationships: `sender(t, s)` → `[(receiver, value)]`.
    Sender { t: u64, s: u32 },
    /// Incoming relationships: `receiver(t, r)` → `[(sender, value)]`.
    Receiver { t: u64, r: u32 },
    /// Derived features: `features(t, s, r)` → one payload per contributing
    /// event at the coordinate.
    Features { t: u64, s: u32, r: u32 },
    /// Tensor statistics: `stats()` → live nonzero count.
    Stats,
    /// Full tensor decode: `decode()` → all `(t, s, r, value)` entries.
    /// Explicitly authorized; callers must request it by name.
    Decode,
}

/// Validation error for a semantic request before it reaches the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticError {
    /// `t_start >= t_end` for an inverted or empty window.
    InvalidTimeWindow { t_start: u64, t_end: u64 },
    /// Bad request: invalid or conflicting client-supplied parameters.
    BadRequest(String),
    /// The requested operation is not supported.
    UnsupportedOperation(String),
    /// Backend (pipeline/tensor) failure propagated unchanged.
    Backend(String),
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTimeWindow { t_start, t_end } => write!(
                f,
                "invalid time window: [{t_start}, {t_end}) is empty or inverted"
            ),
            Self::BadRequest(msg) => write!(f, "bad request: {msg}"),
            Self::UnsupportedOperation(op) => {
                write!(f, "unsupported semantic operation: {op}")
            }
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for SemanticError {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Typed response structures (Contract A)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Result of `lookup(t, s, r)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupResult {
    pub t: u64,
    /// Human-readable UTC representation of `t`.
    pub timestamp_utc: String,
    pub s: u32,
    pub r: u32,
    pub found: bool,
    pub value: Option<i64>,
}

/// One entry of a time-slice: `(t, s, r, value)` in canonical `prec_3` order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSliceEntry {
    pub t: u64,
    /// Human-readable UTC representation of `t`.
    pub timestamp_utc: String,
    pub s: u32,
    pub r: u32,
    pub value: i64,
}

/// Result of `time_slice(t_start, t_end)` — half-open window `[start, end)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSliceResult {
    pub t_start: u64,
    /// Human-readable UTC representation of `t_start`.
    pub t_start_utc: String,
    pub t_end: u64,
    /// Human-readable UTC representation of `t_end`.
    pub t_end_utc: String,
    /// Entries in canonical `t → s → r` order.
    pub entries: Vec<TimeSliceEntry>,
}

/// One entry of a sender/receiver-at-time view: `(peer, value)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionEntry {
    pub peer: u32,
    /// Human-readable UTC representation of the entry's time coordinate.
    pub timestamp_utc: String,
    pub value: i64,
}

/// Result of `sender(t, s)` — outgoing relationships from `s` at time `t`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SenderResult {
    pub t: u64,
    /// Human-readable UTC representation of `t`.
    pub timestamp_utc: String,
    pub s: u32,
    pub entries: Vec<DimensionEntry>,
}

/// Result of `receiver(t, r)` — incoming relationships to `r` at time `t`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiverResult {
    pub t: u64,
    /// Human-readable UTC representation of `t`.
    pub timestamp_utc: String,
    pub r: u32,
    pub entries: Vec<DimensionEntry>,
}

/// Result of `features(t, s, r)` — derived amount-based feature metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaturesResult {
    pub t: u64,
    /// Human-readable UTC representation of `t`.
    pub timestamp_utc: String,
    pub s: u32,
    pub r: u32,
    pub entries: Vec<FeatureEntry>,
}

/// One derived feature entry (privacy-safe: metrics only, never the amount).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureEntry {
    pub log_magnitude: u16,
    pub relative_threshold_distance: i32,
    pub roundedness_flags: u8,
}

/// Result of `stats()` — live nonzero count of `T_live`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsResult {
    pub nnz: u64,
}

/// One entry of a full decode: `(t, s, r, value)` in canonical order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeEntry {
    pub t: u64,
    /// Human-readable UTC representation of `t`.
    pub timestamp_utc: String,
    pub s: u32,
    pub r: u32,
    pub value: i64,
}

/// Result of `decode()` — full live decode, explicitly authorized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeResult {
    pub entries: Vec<DecodeEntry>,
}

/// Unified typed result of any semantic operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticResult {
    Lookup(LookupResult),
    TimeSlice(TimeSliceResult),
    Sender(SenderResult),
    Receiver(ReceiverResult),
    Features(FeaturesResult),
    Stats(StatsResult),
    Decode(DecodeResult),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Semantic layer — direct in-memory Rust contract
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Direct in-memory semantic interface over an [`IngestPipeline`].
///
/// Internal callers hold a reference to a pipeline and invoke typed methods
/// directly. No HTTP, no JSON, no serialization overhead.
pub struct SemanticLayer<'a> {
    pipeline: &'a IngestPipeline,
}

impl<'a> SemanticLayer<'a> {
    /// Construct a semantic layer borrowing the given pipeline.
    #[must_use]
    pub fn new(pipeline: &'a IngestPipeline) -> Self {
        Self { pipeline }
    }

    /// Borrow the underlying pipeline (for callers that need direct access).
    #[must_use]
    pub fn pipeline(&self) -> &'a IngestPipeline {
        self.pipeline
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate a request before execution. Returns `Ok(())` when the request
    /// is well-formed, or a [`SemanticError`] describing the problem.
    pub fn validate(op: &SemanticOp) -> Result<(), SemanticError> {
        match op {
            SemanticOp::TimeSlice { t_start, t_end } => {
                if t_start >= t_end {
                    return Err(SemanticError::InvalidTimeWindow {
                        t_start: *t_start,
                        t_end: *t_end,
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Execution — one method per semantic operation
    // -----------------------------------------------------------------------

    /// Exact coordinate lookup: `lookup(t, s, r)`.
    #[must_use]
    pub fn lookup(&self, t: u64, s: u32, r: u32) -> Result<LookupResult, SemanticError> {
        let found_value = self.pipeline.csf_live_at(t, s, r);
        let timestamp_utc = to_utc_iso8601(t)
            .map(|ts| ts.0)
            .unwrap_or_else(|_| format!("{t}"));
        Ok(LookupResult {
            t,
            timestamp_utc,
            s,
            r,
            found: found_value.is_some(),
            value: found_value,
        })
    }

    /// Time-window retrieval over half-open `[t_start, t_end)`.
    #[must_use]
    pub fn time_slice(&self, t_start: u64, t_end: u64) -> Result<TimeSliceResult, SemanticError> {
        Self::validate(&SemanticOp::TimeSlice { t_start, t_end })?;
        let t_start_utc = to_utc_iso8601(t_start)
            .map(|ts| ts.0)
            .unwrap_or_else(|_| format!("{t_start}"));
        let t_end_utc = to_utc_iso8601(t_end)
            .map(|ts| ts.0)
            .unwrap_or_else(|_| format!("{t_end}"));
        let mut entries = Vec::new();
        for loc in self.pipeline.temporal().global_range(t_start, t_end) {
            if let Some((t, s, r, _feats)) = self.pipeline.resolve(*loc) {
                if let Some(value) = self.pipeline.csf_live_at(t, s, r) {
                    let entry_utc = to_utc_iso8601(t)
                        .map(|ts| ts.0)
                        .unwrap_or_else(|_| format!("{t}"));
                    entries.push(TimeSliceEntry {
                        t,
                        timestamp_utc: entry_utc,
                        s,
                        r,
                        value,
                    });
                }
            }
        }
        Ok(TimeSliceResult {
            t_start,
            t_start_utc,
            t_end,
            t_end_utc,
            entries,
        })
    }

    /// Outgoing relationships: `sender(t, s)` → `[(receiver, value)]`.
    #[must_use]
    pub fn sender(&self, t: u64, s: u32) -> Result<SenderResult, SemanticError> {
        let timestamp_utc = to_utc_iso8601(t)
            .map(|ts| ts.0)
            .unwrap_or_else(|_| format!("{t}"));
        let entries: Vec<DimensionEntry> = self
            .pipeline
            .csf_sender_at_time(t, s)
            .into_iter()
            .map(|(peer, value)| DimensionEntry {
                peer,
                timestamp_utc: timestamp_utc.clone(),
                value,
            })
            .collect();
        Ok(SenderResult {
            t,
            timestamp_utc,
            s,
            entries,
        })
    }

    /// Incoming relationships: `receiver(t, r)` → `[(sender, value)]`.
    #[must_use]
    pub fn receiver(&self, t: u64, r: u32) -> Result<ReceiverResult, SemanticError> {
        let timestamp_utc = to_utc_iso8601(t)
            .map(|ts| ts.0)
            .unwrap_or_else(|_| format!("{t}"));
        let entries: Vec<DimensionEntry> = self
            .pipeline
            .csf_receiver_at_time(t, r)
            .into_iter()
            .map(|(peer, value)| DimensionEntry {
                peer,
                timestamp_utc: timestamp_utc.clone(),
                value,
            })
            .collect();
        Ok(ReceiverResult {
            t,
            timestamp_utc,
            r,
            entries,
        })
    }

    /// Derived features: `features(t, s, r)` → one entry per contributing event.
    #[must_use]
    pub fn features(&self, t: u64, s: u32, r: u32) -> Result<FeaturesResult, SemanticError> {
        let timestamp_utc = to_utc_iso8601(t)
            .map(|ts| ts.0)
            .unwrap_or_else(|_| format!("{t}"));
        let entries: Vec<FeatureEntry> = self
            .pipeline
            .csf_features_at(t, s, r)
            .into_iter()
            .map(|f| FeatureEntry {
                log_magnitude: f.log_magnitude,
                relative_threshold_distance: f.relative_threshold_distance,
                roundedness_flags: f.roundedness_flags,
            })
            .collect();
        Ok(FeaturesResult {
            t,
            timestamp_utc,
            s,
            r,
            entries,
        })
    }

    /// Tensor statistics: `stats()` → live nonzero count.
    #[must_use]
    pub fn stats(&self) -> Result<StatsResult, SemanticError> {
        let st = self.pipeline.csf_stats();
        Ok(StatsResult { nnz: st.nnz })
    }

    /// Full tensor decode: `decode()` → all entries.
    #[must_use]
    pub fn decode(&self) -> Result<DecodeResult, SemanticError> {
        let entries: Vec<DecodeEntry> = self
            .pipeline
            .csf_decode_live()
            .into_iter()
            .map(|(t, s, r, value)| {
                let timestamp_utc = to_utc_iso8601(t)
                    .map(|ts| ts.0)
                    .unwrap_or_else(|_| format!("{t}"));
                DecodeEntry {
                    t,
                    timestamp_utc,
                    s,
                    r,
                    value,
                }
            })
            .collect();
        Ok(DecodeResult { entries })
    }

    /// Convert Unix epoch seconds to an ISO 8601 UTC timestamp string.
    ///
    /// Internal callers can use this to get a human-readable representation
    /// of a canonical tensor coordinate without HTTP or JSON.
    #[must_use]
    pub fn timestamp_to_utc(&self, t: u64) -> Result<String, TimestampError> {
        Ok(to_utc_iso8601(t)?.0)
    }

    /// Parse an ISO 8601 timestamp string into Unix epoch seconds.
    ///
    /// Accepts `YYYY-MM-DDTHH:MM:SSZ` or `YYYY-MM-DDTHH:MM:SS±HH:MM`.
    /// Rejects naive timestamps without a timezone designator.
    #[must_use]
    pub fn utc_to_timestamp(&self, iso: &str) -> Result<u64, TimestampError> {
        from_iso8601(iso)
    }

    /// Execute any [`SemanticOp`] and return the unified [`SemanticResult`].
    #[must_use]
    pub fn execute(&self, op: SemanticOp) -> Result<SemanticResult, SemanticError> {
        match op {
            SemanticOp::Lookup { t, s, r } => self.lookup(t, s, r).map(SemanticResult::Lookup),
            SemanticOp::TimeSlice { t_start, t_end } => {
                self.time_slice(t_start, t_end).map(SemanticResult::TimeSlice)
            }
            SemanticOp::Sender { t, s } => self.sender(t, s).map(SemanticResult::Sender),
            SemanticOp::Receiver { t, r } => self.receiver(t, r).map(SemanticResult::Receiver),
            SemanticOp::Features { t, s, r } => {
                self.features(t, s, r).map(SemanticResult::Features)
            }
            SemanticOp::Stats => self.stats().map(SemanticResult::Stats),
            SemanticOp::Decode => self.decode().map(SemanticResult::Decode),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Conversion helpers — typed result → typed response fields (no JSON)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl From<FeaturePayload> for FeatureEntry {
    fn from(f: FeaturePayload) -> Self {
        Self {
            log_magnitude: f.log_magnitude,
            relative_threshold_distance: f.relative_threshold_distance,
            roundedness_flags: f.roundedness_flags,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest_pipeline::IngestPipeline;
    use abacus_ingest::features::FeaturePayload;
    use abacus_ingest::namespace::{EventIdentity, EventNamespace};
    use abacus_ingest::record::CanonicalRecord;

    fn ns(byte: u8) -> EventNamespace {
        EventNamespace::from_bytes([byte; 16])
    }

    fn rec(event_id: u64, s: u64, r: u64) -> CanonicalRecord {
        CanonicalRecord {
            identity: EventIdentity { namespace: ns(1), event_id },
            event_time_ms: 1_700_000_000_000,
            sender_id: s,
            receiver_id: r,
            features: FeaturePayload {
                log_magnitude: 1,
                relative_threshold_distance: 0,
                roundedness_flags: 0,
            },
        }
    }

    fn seeded() -> IngestPipeline {
        let mut p = IngestPipeline::new();
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100).unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100).unwrap();
        p
    }

    #[test]
    fn validate_rejects_inverted_time_window() {
        let err = SemanticLayer::validate(&SemanticOp::TimeSlice {
            t_start: 1_700_000_001,
            t_end: 1_700_000_000,
        })
        .unwrap_err();
        assert!(matches!(err, SemanticError::InvalidTimeWindow { .. }));
    }

    #[test]
    fn validate_accepts_valid_time_window() {
        assert!(SemanticLayer::validate(&SemanticOp::TimeSlice {
            t_start: 1_700_000_000,
            t_end: 1_700_000_001,
        })
        .is_ok());
    }

    #[test]
    fn validate_accepts_non_window_ops() {
        assert!(SemanticLayer::validate(&SemanticOp::Lookup {
            t: 1,
            s: 2,
            r: 3,
        })
        .is_ok());
        assert!(SemanticLayer::validate(&SemanticOp::Stats).is_ok());
    }

    #[test]
    fn lookup_returns_exact_value_and_absent() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let (t, s, r, _) = p.resolve(0).unwrap();
        let res = layer.lookup(t, s, r).unwrap();
        assert!(res.found);
        assert_eq!(res.value, Some(12_345_000_000));
        assert_eq!((res.t, res.s, res.r), (t, s, r));

        let res = layer.lookup(t, 9, 9).unwrap();
        assert!(!res.found);
        assert_eq!(res.value, None);
    }

    #[test]
    fn time_slice_returns_half_open_window() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let t = 1_700_000_000u64;
        let res = layer.time_slice(t, t + 1).unwrap();
        assert_eq!(res.t_start, t);
        assert_eq!(res.t_end, t + 1);
        assert_eq!(res.entries.len(), 2);
        assert_eq!(res.entries[0].value, 12_345_000_000);
        assert_eq!(res.entries[1].value, 12_345_000_000);

        let res = layer.time_slice(t + 1000, t + 2000).unwrap();
        assert!(res.entries.is_empty());
    }

    #[test]
    fn time_slice_rejects_inverted_window() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);
        let err = layer.time_slice(1_700_000_001, 1_700_000_000).unwrap_err();
        assert!(matches!(err, SemanticError::InvalidTimeWindow { .. }));
    }

    #[test]
    fn sender_returns_outgoing_relationships() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let t = 1_700_000_000u64;
        let (_, s0, _, _) = p.resolve(0).unwrap();

        let res = layer.sender(t, s0).unwrap();
        assert_eq!(res.t, t);
        assert_eq!(res.s, s0);
        assert_eq!(res.entries.len(), 1);
        assert_eq!(res.entries[0].value, 12_345_000_000);

        let res = layer.sender(t, 9).unwrap();
        assert!(res.entries.is_empty());
    }

    #[test]
    fn receiver_returns_incoming_relationships() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let t = 1_700_000_000u64;
        let (_, _, r1, _) = p.resolve(1).unwrap();

        let res = layer.receiver(t, r1).unwrap();
        assert_eq!(res.t, t);
        assert_eq!(res.r, r1);
        assert_eq!(res.entries.len(), 1);
        assert_eq!(res.entries[0].value, 12_345_000_000);

        let res = layer.receiver(t, 9).unwrap();
        assert!(res.entries.is_empty());
    }

    #[test]
    fn features_return_derived_metrics_not_amount() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let t = 1_700_000_000u64;
        let (_, s, r, _) = p.resolve(0).unwrap();

        let res = layer.features(t, s, r).unwrap();
        assert_eq!(res.t, t);
        assert_eq!(res.s, s);
        assert_eq!(res.r, r);
        assert_eq!(res.entries.len(), 1);
        assert_eq!(res.entries[0].log_magnitude, 9_421);
        assert_eq!(res.entries[0].relative_threshold_distance, 235);
        assert_eq!(res.entries[0].roundedness_flags, 0);

        let res = layer.features(t, 9, 9).unwrap();
        assert!(res.entries.is_empty());
    }

    #[test]
    fn stats_returns_live_nonzero_count() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let res = layer.stats().unwrap();
        assert_eq!(res.nnz, 2);
    }

    #[test]
    fn decode_returns_all_entries() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let res = layer.decode().unwrap();
        assert_eq!(res.entries.len(), 2);
        for e in &res.entries {
            assert_eq!(e.value, 12_345_000_000);
        }
    }

    #[test]
    fn execute_dispatches_all_ops() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let t = 1_700_000_000u64;
        let (_, s, r, _) = p.resolve(0).unwrap();

        match layer.execute(SemanticOp::Lookup { t, s, r }).unwrap() {
            SemanticResult::Lookup(res) => assert!(res.found),
            _ => panic!("expected Lookup result"),
        }

        match layer
            .execute(SemanticOp::TimeSlice { t_start: t, t_end: t + 1 })
            .unwrap()
        {
            SemanticResult::TimeSlice(res) => assert_eq!(res.entries.len(), 2),
            _ => panic!("expected TimeSlice result"),
        }

        match layer.execute(SemanticOp::Sender { t, s }).unwrap() {
            SemanticResult::Sender(res) => assert_eq!(res.entries.len(), 1),
            _ => panic!("expected Sender result"),
        }

        match layer.execute(SemanticOp::Receiver { t, r }).unwrap() {
            SemanticResult::Receiver(res) => assert_eq!(res.entries.len(), 1),
            _ => panic!("expected Receiver result"),
        }

        match layer.execute(SemanticOp::Features { t, s, r }).unwrap() {
            SemanticResult::Features(res) => assert_eq!(res.entries.len(), 1),
            _ => panic!("expected Features result"),
        }

        match layer.execute(SemanticOp::Stats).unwrap() {
            SemanticResult::Stats(res) => assert_eq!(res.nnz, 2),
            _ => panic!("expected Stats result"),
        }

        match layer.execute(SemanticOp::Decode).unwrap() {
            SemanticResult::Decode(res) => assert_eq!(res.entries.len(), 2),
            _ => panic!("expected Decode result"),
        }
    }

    #[test]
    fn direct_rust_call_without_http() {
        let p = seeded();
        let layer = SemanticLayer::new(&p);

        let t = 1_700_000_000u64;
        let (_, s, r, _) = p.resolve(0).unwrap();
        let res = layer.lookup(t, s, r).unwrap();
        assert_eq!(res.value, Some(12_345_000_000));
    }
}