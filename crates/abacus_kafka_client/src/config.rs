//! Kafka source configuration.
//!
//! Builds the validated [`abacus_ingest::config::SourceConfig`] the canonical
//! pipeline requires, pinned to `SourceType::Kafka` (=3) and
//! `IdentityScheme::KafkaPartitionOffset` (=3) per Wire §2. Every other field is
//! an explicit operator decision (currency, precision, threshold, width,
//! ceilings, parsing-policy version) — Golden Source §7/§8 forbid silent
//! defaults, so the builder has no `Default` and every field is required.

use abacus_ingest::config::{
    CurrencyCode, IdentityScheme, SourceConfig, SourceType, FEATURE_ALGORITHM_VERSION,
    FEATURE_SCHEMA_VERSION,
};
use abacus_ingest::IngestResult;

/// The operator-declared Kafka source configuration.
///
/// The Kafka-specific pair (`source_type`/`identity_scheme`) is fixed by this
/// adapter; the remaining fields mirror [`SourceConfig`] one-for-one and are
/// carried into it unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KafkaConfig {
    /// Account-ID width in bytes: 1, 2, 4 or 8 (selected per dataset/segment).
    pub account_id_width: u8,
    /// ISO 4217 currency code, three uppercase letters.
    pub currency_code: String,
    /// Minor-unit decimal digits (e.g. 2 for USD cents).
    pub currency_precision: u8,
    /// Positive threshold in integer minor units (the `relative_threshold_distance`
    /// feature is measured against this).
    pub threshold_minor_units: u64,
    /// Threshold effective day: signed days since the Unix epoch.
    pub threshold_effective_day: i32,
    /// Threshold configuration version.
    pub threshold_config_version: u32,
    /// Parsing-policy version (bumped whenever a parsing decision changes).
    pub parsing_policy_version: u32,
    /// Supported amount ceiling in minor units.
    pub max_amount_minor_units: u64,
}

impl KafkaConfig {
    /// Build and validate the canonical [`SourceConfig`] for this Kafka source.
    ///
    /// The feature schema/algorithm versions are the frozen constants the oracle
    /// implements — an adapter cannot choose a different feature algorithm
    /// (Golden Source §9/§19). Validation is delegated to
    /// [`SourceConfig::validate`], so a malformed currency code, a non-positive
    /// threshold, an unsupported width or an out-of-range feature version is
    /// rejected here, fail-closed, before any message is read.
    ///
    /// # Errors
    /// - [`abacus_ingest::ErrorCode::AmountOrCurrency`] for a malformed currency
    ///   code or an out-of-range threshold/precision/ceiling.
    /// - Any other [`SourceConfig::validate`] rejection.
    pub fn to_source_config(&self) -> IngestResult<SourceConfig> {
        let currency_code = CurrencyCode::parse(&self.currency_code)?;
        let cfg = SourceConfig {
            source_type: SourceType::Kafka,
            identity_scheme: IdentityScheme::KafkaPartitionOffset,
            account_id_width: self.account_id_width,
            currency_code,
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
