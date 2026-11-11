//! The explicit normalization policy: everything §7/§8 require the user to
//! state, in one place.
//!
//! Golden Source §7 (timestamps) and §8 (amounts) both forbid guessing:
//!
//! - "Require an explicit source timezone when timestamps have no timezone."
//! - "Require explicit date-format selection for ambiguous date strings."
//! - "Require an explicit unit for numeric timestamps, such as seconds or
//!   milliseconds."
//! - "Locale and decimal/thousands separator policy" must be configured.
//! - "Record the source timezone and parsing-policy version in session
//!   configuration metadata."
//!
//! So every one of those is a field here with no `Default` that silently picks
//! a value. [`IngestPlan`] is constructed field-by-field, and a plan that omits
//! a required decision is rejected by [`IngestPlan::validate`] rather than
//! defaulted — which is exactly §9 Stage 1's "Do not guess missing
//! configuration when the guess changes event meaning."
//!
//! The policy also feeds the session's configuration manifest, so a different
//! locale or timestamp style produces a different `config_fingerprint` and is
//! therefore rejected as a conflicting configuration (§13). That is the
//! mechanism by which "record the source timezone and parsing-policy version in
//! session configuration metadata" is enforced rather than merely stored: a
//! retry that changes the locale cannot silently join the same session.

use abacus_ingest::config::{CurrencyCode, FEATURE_ALGORITHM_VERSION, FEATURE_SCHEMA_VERSION};
use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};

/// How a timestamp column is to be interpreted. All variants are explicit
/// choices; none is inferred from the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampStyle {
    /// ISO-8601 with an explicit UTC offset or `Z` (`2026-10-12T09:30:00+02:00`,
    /// `2026-10-12T09:30:00Z`). An ISO-8601 value **without** an offset is
    /// rejected under this style, because §7 requires an explicit source
    /// timezone for offset-less input.
    Iso8601WithOffset,
    /// ISO-8601 calendar date/time **without** an offset, interpreted in the
    /// plan's explicit [`IngestPlan::source_offset_minutes`].
    ///
    /// This is the §7 "explicit source timezone" branch: the user states the
    /// offset once for the whole file instead of per row.
    Iso8601WithSourceOffset,
    /// A bare numeric epoch value, scaled by the explicit
    /// [`NumberPolicy::epoch_unit`].
    NumericEpoch,
}

/// How the amount column's number format is to be read.
///
/// §8: "Locale and decimal/thousands separator policy" is required
/// configuration, and "Binary floating-point must not be the authoritative
/// representation of money". So the separators are named characters and the
/// parse is exact integer arithmetic (see [`crate::amount`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmountFormat {
    /// Decimal separator character, e.g. `b'.'` or `b','`.
    pub decimal_separator: u8,
    /// Optional thousands/grouping separator character, e.g. `b','` or `b'.'`
    /// or a non-breaking space. `None` means grouping is not permitted.
    pub group_separator: Option<u8>,
    /// Whether a leading `-` is accepted at all.
    ///
    /// The canonical amount is an unsigned minor-unit count (§9.1: "x is a
    /// nonnegative integer amount in currency minor units"), so a signed input
    /// must be an explicit, declared decision — never a silent absolute value.
    pub accept_negative: bool,
}

/// The unit of a numeric epoch timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochUnit {
    /// Whole seconds since the Unix epoch.
    Seconds,
    /// Whole milliseconds since the Unix epoch (the canonical wire unit).
    Millis,
}

/// Numeric parsing policy: the explicit unit and the sub-second rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumberPolicy {
    /// Unit of a [`TimestampStyle::NumericEpoch`] column (§7: "Require an
    /// explicit unit for numeric timestamps").
    pub epoch_unit: EpochUnit,
    /// What to do with a fractional part on a numeric epoch or a sub-millisecond
    /// ISO fraction.
    ///
    /// Canonical Contract §5: "Do not silently discard sub-millisecond
    /// precision. Either reject excess precision or apply a separately
    /// versioned conversion rule." The client therefore has exactly two legal
    /// behaviours and this enum names them.
    pub subsecond: SubSecondPolicy,
}

/// The declared rule for precision finer than one millisecond.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubSecondPolicy {
    /// Reject any value carrying sub-millisecond precision.
    ///
    /// This is the default and the fail-closed choice: it never invents a value
    /// and never hides a loss.
    RejectSubMilli,
    /// Truncate toward zero to whole milliseconds.
    ///
    /// Allowed only because the substrate coordinate is whole seconds, so the
    /// millisecond itself is already lossy downstream; the truncation is
    /// declared here (and therefore in the configuration fingerprint) rather
    /// than happening invisibly.
    TruncateToMilli,
}

/// The complete timestamp policy: the style plus the optional source offset.
///
/// Both fields are required to be *decided*, which is what §7 demands. A
/// `TimestampPolicy` with `style = Iso8601WithSourceOffset` and no offset is
/// constructible but is rejected by [`IngestPlan::validate`], so the failure is
/// reported before any row is read rather than as a per-row surprise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampPolicy {
    /// How the timestamp column is interpreted.
    pub style: TimestampStyle,
    /// Source UTC offset in **minutes** east of UTC, when the style needs one.
    ///
    /// Only meaningful for [`TimestampStyle::Iso8601WithSourceOffset`]; setting
    /// it alongside `Iso8601WithOffset` is a validation error because it would
    /// be silently ignored.
    pub source_offset_minutes: Option<i32>,
}

/// One complete, explicit plan for turning a CSV into an ingestion session.
///
/// Everything a row's meaning depends on is here, and every field is required.
/// [`IngestPlan::validate`] is the gate; [`IngestPlan::to_source_config`] maps
/// the plan onto the pipeline's own [`abacus_ingest::config::SourceConfig`] so
/// the fingerprint the server binds is derived from exactly these decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestPlan {
    /// Column mapping from source column to canonical field.
    pub mapping: crate::mapping::ColumnMapping,
    /// CSV reader options (delimiter, quoting, header presence).
    pub csv: crate::csv::CsvOptions,
    /// Timestamp interpretation.
    pub timestamp: TimestampPolicy,
    /// Amount/number interpretation.
    pub amount: AmountFormat,
    /// Numeric epoch unit and the sub-second rule.
    pub numbers: NumberPolicy,
    /// Identifier handling (trimming, width, capacity).
    pub identifiers: crate::identifiers::IdentifierPolicy,
    /// ISO 4217 currency code, three uppercase letters.
    pub currency_code: String,
    /// Minor-unit decimal digits for that currency (2 for USD cents).
    pub currency_precision: u8,
    /// Positive threshold in the same minor units (§8, §9.2).
    pub threshold_minor_units: u64,
    /// Threshold effective day: signed days since the Unix epoch.
    pub threshold_effective_day: i32,
    /// Threshold configuration version.
    pub threshold_config_version: u32,
    /// Supported amount ceiling in minor units (§8: "Supported amount ceiling
    /// and validation policy").
    pub max_amount_minor_units: u64,
    /// Parsing-policy version recorded in session metadata (§7).
    ///
    /// Bumped by the operator whenever any parsing decision above changes, so a
    /// report can state which policy produced a given artifact.
    pub parsing_policy_version: u32,
    /// Records per uploaded chunk. Must fit the server's enforced bounds (§12:
    /// "Chunk size must be configurable within server-enforced bounds").
    pub records_per_chunk: usize,
    /// Namespace to reuse for a retry/resume, or `None` for a fresh import
    /// (§3.2: "Never regenerate it during a retry").
    pub namespace: Option<abacus_ingest::namespace::EventNamespace>,
}

impl IngestPlan {
    /// Validate every decision that changes event meaning, failing closed.
    ///
    /// This runs **before** any row is parsed, so a misconfigured import is
    /// refused up front rather than discovered halfway through an upload.
    ///
    /// # Errors
    /// - [`ErrorCode::AmountOrCurrency`] for a malformed currency code, a
    ///   non-positive threshold or ceiling, or an unusable precision.
    /// - [`ErrorCode::Timestamp`] for an offset-less ISO style with no source
    ///   offset, or a source offset outside the real UTC range.
    /// - [`ErrorCode::FileOrCsvParse`] for a zero chunk size or a separator that
    ///   collides with the CSV delimiter or quote character.
    pub fn validate(&self) -> IngestResult<()> {
        CurrencyCode::parse(&self.currency_code)?;
        if self.currency_precision > 18 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                format!("unsupported currency_precision {}", self.currency_precision),
            ));
        }
        10_u64
            .checked_pow(u32::from(self.currency_precision))
            .filter(|&s| s > 0)
            .ok_or_else(|| {
                IngestError::new(
                    ErrorCode::AmountOrCurrency,
                    "10^precision is not representable in u64",
                )
            })?;
        // §9.2: "T must be greater than zero. Invalid or missing threshold
        // configuration must not be replaced with zero."
        if self.threshold_minor_units == 0 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "threshold_minor_units must be positive (0 is never a valid threshold)",
            ));
        }
        if self.max_amount_minor_units == 0 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "max_amount_minor_units must be positive",
            ));
        }
        match self.timestamp.style {
            TimestampStyle::Iso8601WithOffset => {
                if self.timestamp.source_offset_minutes.is_some() {
                    return Err(IngestError::new(
                        ErrorCode::Timestamp,
                        "Iso8601WithOffset requires an offset in every value; a source \
                         offset would be ignored, so the plan is ambiguous",
                    ));
                }
            }
            TimestampStyle::Iso8601WithSourceOffset => {
                if self.timestamp.source_offset_minutes.is_none() {
                    return Err(IngestError::new(
                        ErrorCode::Timestamp,
                        "timestamps without an offset require an explicit source timezone",
                    ));
                }
            }
            TimestampStyle::NumericEpoch => {}
        }
        if let Some(off) = self.timestamp.source_offset_minutes {
            if !(-12 * 60..=14 * 60).contains(&off) {
                return Err(IngestError::new(
                    ErrorCode::Timestamp,
                    format!("source offset {off} minutes is outside the real UTC range"),
                ));
            }
        }
        self.validate_separators()
    }

    /// Separator, delimiter and chunk-size well-formedness.
    fn validate_separators(&self) -> IngestResult<()> {
        if let Some(g) = self.amount.group_separator {
            if g == self.amount.decimal_separator {
                return Err(IngestError::new(
                    ErrorCode::AmountOrCurrency,
                    "decimal and group separators must differ",
                ));
            }
        }
        if self.amount.decimal_separator == self.csv.delimiter
            || self.amount.decimal_separator == self.csv.quote
            || self.amount.group_separator == Some(self.csv.delimiter)
            || self.amount.group_separator == Some(self.csv.quote)
        {
            return Err(IngestError::new(
                ErrorCode::FileOrCsvParse,
                "a number separator collides with the CSV delimiter or quote character",
            ));
        }
        if self.csv.delimiter == self.csv.quote {
            return Err(IngestError::new(
                ErrorCode::FileOrCsvParse,
                "the CSV delimiter and quote character must differ",
            ));
        }
        if self.records_per_chunk == 0 {
            return Err(IngestError::new(
                ErrorCode::FileOrCsvParse,
                "records_per_chunk must be at least 1",
            ));
        }
        self.identifiers.validate()
    }

    /// Map this plan onto the pipeline's validated [`abacus_ingest::config::SourceConfig`].
    ///
    /// Browser CSV is declared as `SourceType::BrowserCsv` with
    /// `IdentityScheme::CsvRecordOrdinal` (wire §2 values 1 and 1): the event id
    /// is the stable parsed CSV record ordinal, which is what makes a retry of
    /// the same file re-derive identical identities.
    ///
    /// # Errors
    /// Any [`Self::validate`] rejection, plus any `SourceConfig::validate`
    /// rejection.
    pub fn to_source_config(&self) -> IngestResult<abacus_ingest::config::SourceConfig> {
        self.validate()?;
        let cfg = abacus_ingest::config::SourceConfig {
            source_type: abacus_ingest::config::SourceType::BrowserCsv,
            identity_scheme: abacus_ingest::config::IdentityScheme::CsvRecordOrdinal,
            account_id_width: self.identifiers.account_id_width,
            currency_code: CurrencyCode::parse(&self.currency_code)?,
            currency_precision: self.currency_precision,
            threshold_minor_units: self.threshold_minor_units,
            threshold_effective_day: self.threshold_effective_day,
            threshold_config_version: self.threshold_config_version,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            feature_algorithm_version: FEATURE_ALGORITHM_VERSION,
            parsing_policy_version: self.parsing_policy_version,
            max_amount_minor_units: self.max_amount_minor_units,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
