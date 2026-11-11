//! Structured ingestion errors.
//!
//! Golden Source v2.0 §16 ("Use distinct error categories") enumerates the
//! required categories; each one is a variant here so that a rejection can be
//! attributed to a stable category without string matching, and so the report
//! can distinguish *rejected*, *quarantined*, *duplicate* and *conflicting*
//! outcomes (§11: "The report must distinguish rejected records, quarantined
//! records, duplicate deliveries, and conflicting identities").
//!
//! Privacy rule (§16): errors "must not disclose raw account identifiers or raw
//! amounts". Every variant therefore carries only structural facts — field
//! names, offsets, widths, versions, ordinals. No variant can hold an account
//! string or an amount.

use abacus_tensor::TensorError;

/// Stable machine-readable error category codes (Golden Source §16).
///
/// These codes are part of the public API contract: a GUI switches on them, so
/// they must not be renamed without a version bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// File or CSV parsing error.
    FileOrCsvParse,
    /// Missing or ambiguous column mapping.
    ColumnMapping,
    /// Invalid timestamp or timezone.
    Timestamp,
    /// Invalid amount or currency configuration.
    AmountOrCurrency,
    /// Invalid or colliding account identifier.
    AccountId,
    /// Account-ID capacity exceeded for the declared width.
    AccountIdCapacity,
    /// Feature calculation overflow.
    FeatureOverflow,
    /// Unsupported schema or feature version.
    SchemaVersion,
    /// Binary header, length, or checksum failure.
    Binary,
    /// Duplicate or conflicting event identity.
    Identity,
    /// Chunk protocol or session error.
    Session,
    /// StreamingIngestionDemo or late-event policy violation.
    StreamingIngestionDemoPolicy,
    /// CSF/index invariant failure.
    SubstrateInvariant,
    /// A material decision the specifications leave unresolved.
    SpecificationGap,
    /// Durable-journal or recovery failure (Amendment A2).
    Persistence,
}

impl ErrorCode {
    /// Stable wire spelling of the category (used verbatim in JSON reports).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FileOrCsvParse => "file_or_csv_parse",
            Self::ColumnMapping => "column_mapping",
            Self::Timestamp => "timestamp",
            Self::AmountOrCurrency => "amount_or_currency",
            Self::AccountId => "account_id",
            Self::AccountIdCapacity => "account_id_capacity",
            Self::FeatureOverflow => "feature_overflow",
            Self::SchemaVersion => "schema_version",
            Self::Binary => "binary",
            Self::Identity => "identity",
            Self::Session => "session",
            Self::StreamingIngestionDemoPolicy => "streaming_ingestion_demo_policy",
            Self::SubstrateInvariant => "substrate_invariant",
            Self::SpecificationGap => "specification_gap",
            Self::Persistence => "persistence",
        }
    }
}

/// One ingestion failure: a stable category plus a structural detail message.
///
/// `detail` is intentionally restricted to non-sensitive structure. Callers
/// pass field names, numeric bounds and record ordinals — never account
/// strings, never amounts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestError {
    /// Stable category (Golden Source §16).
    pub code: ErrorCode,
    /// Structural detail. MUST NOT contain raw account IDs or amounts (§16).
    pub detail: String,
}

impl IngestError {
    /// Construct a categorized error.
    #[must_use]
    pub fn new(code: ErrorCode, detail: impl Into<String>) -> Self {
        Self { code, detail: detail.into() }
    }
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.detail)
    }
}

impl std::error::Error for IngestError {}

/// Substrate failures lift unchanged, re-categorized as invariant failures
/// unless they are value-range violations (which map to feature overflow).
impl From<TensorError> for IngestError {
    fn from(e: TensorError) -> Self {
        let code = match e {
            TensorError::Range(_) => ErrorCode::FeatureOverflow,
            TensorError::Parse(_) => ErrorCode::FileOrCsvParse,
            TensorError::Decode(_) | TensorError::Io(_) => ErrorCode::SubstrateInvariant,
        };
        Self::new(code, e.to_string())
    }
}

/// Convenience alias.
pub type IngestResult<T> = Result<T, IngestError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_strings() {
        assert_eq!(ErrorCode::Binary.as_str(), "binary");
        assert_eq!(ErrorCode::SpecificationGap.as_str(), "specification_gap");
    }

    #[test]
    fn tensor_range_maps_to_feature_overflow() {
        let e = IngestError::from(TensorError::Range("x".into()));
        assert_eq!(e.code, ErrorCode::FeatureOverflow);
    }

    #[test]
    fn tensor_decode_maps_to_substrate_invariant() {
        let e = IngestError::from(TensorError::Decode("x".into()));
        assert_eq!(e.code, ErrorCode::SubstrateInvariant);
    }
}
