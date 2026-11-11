//! Source configuration and its canonical manifest bytes.
//!
//! The `config_fingerprint` is defined by the wire contract §4 as "the SHA-256
//! digest of the exact configuration-manifest bytes used to create the
//! segment", with the explicit rule that "different manifest bytes produce
//! different fingerprints, even if someone believes the configurations are
//! semantically equivalent". So the manifest must be a *total, injective*
//! encoding of the configuration: every field is emitted, in a fixed order,
//! with fixed widths and explicit tags. No field may be omitted or normalized
//! away, otherwise two genuinely different configurations could collide.
//!
//! Golden Source §8/§9 Stage 1 requires the configuration to declare currency,
//! precision, threshold, threshold effective day and version, feature schema
//! and algorithm versions, source type, identity scheme and account-ID width.
//! §9 Stage 1 also requires: "Do not guess missing configuration when the
//! guess changes event meaning." — hence [`SourceConfig::validate`] rejects
//! incomplete configuration instead of defaulting it.

use crate::error::{ErrorCode, IngestError, IngestResult};

/// Magic bytes of an ingestion segment (`ABACSEG1`), wire §1.
pub const MAGIC: &[u8; 8] = b"ABACSEG1";

/// Wire-format version implemented here (wire §1: `format_version` = 1).
pub const FORMAT_VERSION: u16 = 1;

/// Exact header length in bytes (wire §1: 176, no padding).
pub const HEADER_LEN: usize = 176;

/// SHA-256 digest width.
pub const SHA256_LEN: usize = 32;

/// Supported account-ID widths in bytes (Golden Source §6: 8/16/32/64-bit only).
pub const ACCOUNT_ID_WIDTHS: [u8; 4] = [1, 2, 4, 8];

/// Feature-schema version implemented here.
pub const FEATURE_SCHEMA_VERSION: u16 = 1;

/// Feature-algorithm version implemented here.
///
/// Bumped independently of the schema version whenever the deterministic
/// derivation changes, per Golden Source §9.4/§10.
pub const FEATURE_ALGORITHM_VERSION: u16 = 1;

/// Maximum accepted `record_count` (bounds a hostile count before allocation).
pub const MAX_RECORDS: u64 = 100_000_000;

/// Maximum accepted `payload_length` in bytes (2 GiB bound).
pub const MAX_PAYLOAD_BYTES: u64 = 2_u64 << 30;

/// Default maximum chunk payload accepted by a session (8 MiB).
///
/// Golden Source §13: "The server enforces maximum chunk size, record count,
/// total upload size, and resource limits." The value is a server-enforced
/// bound, configurable per session.
pub const DEFAULT_MAX_CHUNK_BYTES: u64 = 8 * 1024 * 1024;

/// Source type enum (wire §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    /// Browser CSV import (1).
    BrowserCsv,
    /// SAML-D fixture (2).
    SamldFixture,
    /// Kafka ingestion (3).
    Kafka,
}

impl SourceType {
    /// Decode the wire enum, rejecting unsupported values (wire §2).
    ///
    /// # Errors
    /// [`ErrorCode::Binary`] on an unsupported enum value.
    pub fn from_u8(v: u8) -> IngestResult<Self> {
        match v {
            1 => Ok(Self::BrowserCsv),
            2 => Ok(Self::SamldFixture),
            3 => Ok(Self::Kafka),
            other => Err(IngestError::new(
                ErrorCode::Binary,
                format!("unsupported source_type {other}"),
            )),
        }
    }

    /// Encode to the wire enum value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::BrowserCsv => 1,
            Self::SamldFixture => 2,
            Self::Kafka => 3,
        }
    }

    /// Stable spelling for reports and API responses.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BrowserCsv => "browser_csv",
            Self::SamldFixture => "samld_fixture",
            Self::Kafka => "kafka",
        }
    }
}

/// Identity scheme enum (wire §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityScheme {
    /// Stable parsed CSV record ordinal (1).
    CsvRecordOrdinal,
    /// Stable SAML-D fixture record ordinal (2).
    SamldRecordOrdinal,
    /// Kafka partition offset (3).
    KafkaPartitionOffset,
}

impl IdentityScheme {
    /// Decode the wire enum, rejecting unsupported values (wire §2).
    ///
    /// # Errors
    /// [`ErrorCode::Binary`] on an unsupported enum value.
    pub fn from_u8(v: u8) -> IngestResult<Self> {
        match v {
            1 => Ok(Self::CsvRecordOrdinal),
            2 => Ok(Self::SamldRecordOrdinal),
            3 => Ok(Self::KafkaPartitionOffset),
            other => Err(IngestError::new(
                ErrorCode::Binary,
                format!("unsupported identity_scheme {other}"),
            )),
        }
    }

    /// Encode to the wire enum value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::CsvRecordOrdinal => 1,
            Self::SamldRecordOrdinal => 2,
            Self::KafkaPartitionOffset => 3,
        }
    }

    /// Stable spelling for reports and API responses.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CsvRecordOrdinal => "csv_record_ordinal",
            Self::SamldRecordOrdinal => "samld_record_ordinal",
            Self::KafkaPartitionOffset => "kafka_partition_offset",
        }
    }
}

/// A validated ISO 4217 currency code (three uppercase ASCII letters).
///
/// Golden Source §8 requires an ISO 4217 code; §11 requires rejection of
/// malformed values rather than coercion. The type cannot be constructed with
/// an invalid code, so validation happens once at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrencyCode([u8; 3]);

impl CurrencyCode {
    /// Parse three uppercase ASCII letters (wire §1).
    ///
    /// # Errors
    /// [`ErrorCode::AmountOrCurrency`] if the code is not exactly three
    /// uppercase ASCII letters.
    pub fn parse(s: &str) -> IngestResult<Self> {
        let b = s.as_bytes();
        if b.len() != 3 || !b.iter().all(u8::is_ascii_uppercase) {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "currency_code must be exactly three uppercase ASCII letters",
            ));
        }
        Ok(Self([b[0], b[1], b[2]]))
    }

    /// The three ASCII bytes (wire §1 offsets 56–58).
    #[must_use]
    pub const fn bytes(&self) -> [u8; 3] {
        self.0
    }

    /// Render back to the three-character code.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // Safe: constructed only from ASCII uppercase.
        std::str::from_utf8(&self.0).unwrap_or("???")
    }
}

/// Lowercase hex encoding of a digest.
#[must_use]
pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// The full validated source configuration for one segment or session.
///
/// Every field is part of the identity of the configuration: the manifest
/// encoding is injective, so changing any field changes the fingerprint and
/// therefore must be rejected as a conflicting configuration (Golden Source
/// §13: "Conflicting configuration fingerprints must not be accepted into the
/// same logical session without an explicit migration policy.").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceConfig {
    /// Source type (wire §2).
    pub source_type: SourceType,
    /// Identity scheme (wire §2).
    pub identity_scheme: IdentityScheme,
    /// Account-ID width in bytes: 1, 2, 4 or 8.
    pub account_id_width: u8,
    /// ISO 4217 currency code.
    pub currency_code: CurrencyCode,
    /// Minor-unit decimal digits (e.g. 2 for cents).
    pub currency_precision: u8,
    /// Positive threshold in integer minor units.
    pub threshold_minor_units: u64,
    /// Threshold effective day: signed days since Unix epoch.
    pub threshold_effective_day: i32,
    /// Threshold configuration version.
    pub threshold_config_version: u32,
    /// Feature-schema version.
    pub feature_schema_version: u16,
    /// Feature-algorithm version.
    pub feature_algorithm_version: u16,
    /// Parsing-policy version (Golden Source §7: recorded in session metadata).
    pub parsing_policy_version: u32,
    /// Supported amount ceiling in minor units (Golden Source §8).
    pub max_amount_minor_units: u64,
}

impl SourceConfig {
    /// Validate the configuration (fail closed; never guess, Golden Source
    /// §9 Stage 1).
    ///
    /// # Errors
    /// - [`ErrorCode::AmountOrCurrency`] for a non-positive threshold,
    ///   unsupported currency precision, or a non-positive amount ceiling.
    /// - [`ErrorCode::SchemaVersion`] for unsupported feature versions.
    /// - [`ErrorCode::AccountId`] for an unsupported account-ID width.
    pub fn validate(&self) -> IngestResult<()> {
        if !ACCOUNT_ID_WIDTHS.contains(&self.account_id_width) {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!(
                    "unsupported account_id_width {}; permitted: 1, 2, 4, 8",
                    self.account_id_width
                ),
            ));
        }
        if self.threshold_minor_units == 0 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "threshold_minor_units must be positive (0 is never a valid threshold)",
            ));
        }
        if self.currency_precision > 18 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                format!("unsupported currency_precision {}", self.currency_precision),
            ));
        }
        if self.max_amount_minor_units == 0 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "max_amount_minor_units must be positive",
            ));
        }
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION {
            return Err(IngestError::new(
                ErrorCode::SchemaVersion,
                format!(
                    "unsupported feature_schema_version {}; implemented: {FEATURE_SCHEMA_VERSION}",
                    self.feature_schema_version
                ),
            ));
        }
        if self.feature_algorithm_version != FEATURE_ALGORITHM_VERSION {
            return Err(IngestError::new(
                ErrorCode::SchemaVersion,
                format!(
                    "unsupported feature_algorithm_version {}; implemented: {FEATURE_ALGORITHM_VERSION}",
                    self.feature_algorithm_version
                ),
            ));
        }
        Ok(())
    }

    /// The major-unit scale factor `10^precision` as `u64` (exact integer
    /// arithmetic only; Golden Source §9.3).
    ///
    /// # Errors
    /// [`ErrorCode::AmountOrCurrency`] if `10^precision` is not representable.
    pub fn major_scale(&self) -> IngestResult<u64> {
        10_u64
            .checked_pow(u32::from(self.currency_precision))
            .filter(|&s| s > 0)
            .ok_or_else(|| {
                IngestError::new(
                    ErrorCode::AmountOrCurrency,
                    "10^precision is not representable in u64",
                )
            })
    }

    /// Encode the configuration as its canonical manifest bytes.
    ///
    /// INJECTIVITY REQUIREMENT (wire §4): every field is emitted in a fixed
    /// order with fixed widths and a format tag, so distinct configurations
    /// produce distinct byte strings. There is no omission and no
    /// normalization step.
    #[must_use]
    pub fn manifest_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128);
        // Manifest format tag: versioned so a future manifest layout change
        // cannot silently reinterpret old bytes.
        out.extend_from_slice(b"ABACCFG1");
        out.push(self.source_type.to_u8());
        out.push(self.identity_scheme.to_u8());
        out.push(self.account_id_width);
        out.extend_from_slice(&self.currency_code.bytes());
        out.push(self.currency_precision);
        out.extend_from_slice(&self.threshold_minor_units.to_le_bytes());
        out.extend_from_slice(&self.threshold_effective_day.to_le_bytes());
        out.extend_from_slice(&self.threshold_config_version.to_le_bytes());
        out.extend_from_slice(&self.feature_schema_version.to_le_bytes());
        out.extend_from_slice(&self.feature_algorithm_version.to_le_bytes());
        out.extend_from_slice(&self.parsing_policy_version.to_le_bytes());
        out.extend_from_slice(&self.max_amount_minor_units.to_le_bytes());
        out
    }

    /// `config_fingerprint` = SHA-256 of [`Self::manifest_bytes`] (wire §4).
    #[must_use]
    pub fn fingerprint(&self) -> [u8; SHA256_LEN] {
        crate::hash::sha256(&self.manifest_bytes())
    }

    /// The manifest fingerprint rendered as lowercase hex (report/API form).
    #[must_use]
    pub fn fingerprint_hex(&self) -> String {
        hex_encode(&self.fingerprint())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SourceConfig {
        SourceConfig {
            source_type: SourceType::BrowserCsv,
            identity_scheme: IdentityScheme::CsvRecordOrdinal,
            account_id_width: 4,
            currency_code: CurrencyCode::parse("USD").unwrap(),
            currency_precision: 2,
            threshold_minor_units: 10_000,
            threshold_effective_day: 19_000,
            threshold_config_version: 3,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            feature_algorithm_version: FEATURE_ALGORITHM_VERSION,
            parsing_policy_version: 1,
            max_amount_minor_units: 1_000_000_000,
        }
    }

    #[test]
    fn valid_config_passes() {
        cfg().validate().unwrap();
    }

    #[test]
    fn zero_threshold_rejected() {
        let mut c = cfg();
        c.threshold_minor_units = 0;
        assert_eq!(c.validate().unwrap_err().code, ErrorCode::AmountOrCurrency);
    }

    #[test]
    fn bad_width_rejected() {
        for w in [0u8, 3, 5, 16] {
            let mut c = cfg();
            c.account_id_width = w;
            assert_eq!(c.validate().unwrap_err().code, ErrorCode::AccountId);
        }
    }

    #[test]
    fn bad_currency_rejected() {
        assert!(CurrencyCode::parse("usd").is_err());
        assert!(CurrencyCode::parse("US").is_err());
        assert!(CurrencyCode::parse("USDD").is_err());
        assert!(CurrencyCode::parse("US1").is_err());
    }

    #[test]
    fn unsupported_feature_versions_rejected() {
        let mut c = cfg();
        c.feature_schema_version = 99;
        assert_eq!(c.validate().unwrap_err().code, ErrorCode::SchemaVersion);
        let mut d = cfg();
        d.feature_algorithm_version = 99;
        assert_eq!(d.validate().unwrap_err().code, ErrorCode::SchemaVersion);
    }

    #[test]
    fn manifest_is_injective_per_field() {
        let base = cfg();
        let base_bytes = base.manifest_bytes();
        let base_fp = base.fingerprint();

        let mut a = base.clone();
        a.account_id_width = 8;
        let mut b = base.clone();
        b.currency_precision = 0;
        let mut c = base.clone();
        c.threshold_minor_units = 10_001;
        let mut d = base.clone();
        d.parsing_policy_version = 2;
        let mut e = base.clone();
        e.source_type = SourceType::Kafka;
        let mut f = base.clone();
        f.identity_scheme = IdentityScheme::SamldRecordOrdinal;
        let mut g = base.clone();
        g.threshold_effective_day = 19_001;
        let mut h = base.clone();
        h.threshold_config_version = 4;
        let mut i = base.clone();
        i.max_amount_minor_units = 1;
        let mut j = base.clone();
        j.currency_code = CurrencyCode::parse("EUR").unwrap();

        for variant in [a, b, c, d, e, f, g, h, i, j] {
            assert_ne!(variant.manifest_bytes(), base_bytes, "collision");
            assert_ne!(variant.fingerprint(), base_fp, "fingerprint collision");
        }
    }

    #[test]
    fn manifest_is_deterministic() {
        assert_eq!(cfg().manifest_bytes(), cfg().manifest_bytes());
        assert_eq!(cfg().fingerprint(), cfg().fingerprint());
    }

    #[test]
    fn major_scale_matches_precision() {
        let mut c = cfg();
        c.currency_precision = 2;
        assert_eq!(c.major_scale().unwrap(), 100);
        c.currency_precision = 0;
        assert_eq!(c.major_scale().unwrap(), 1);
    }

    #[test]
    fn hex_encodes_known_bytes() {
        assert_eq!(hex_encode(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }

    #[test]
    fn enum_roundtrips() {
        for v in 1..=3u8 {
            assert_eq!(SourceType::from_u8(v).unwrap().to_u8(), v);
            assert_eq!(IdentityScheme::from_u8(v).unwrap().to_u8(), v);
        }
        assert!(SourceType::from_u8(0).is_err());
        assert!(SourceType::from_u8(4).is_err());
        assert!(IdentityScheme::from_u8(255).is_err());
    }
}
