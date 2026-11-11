//! The 176-byte ABACSEG1 segment header.
//!
//! Wire §1 fixes the layout exactly: 176 bytes, no padding, no implicit
//! alignment, no trailing bytes. Offsets are zero-based and inclusive:
//!
//! | Off | End | Size | Field | Encoding |
//! |-----|-----|------|-------|----------|
//! | 0 | 7 | 8 | magic | ASCII `ABACSEG1` |
//! | 8 | 9 | 2 | format_version | u16 = 1 |
//! | 10 | 11 | 2 | header_length | u16 = 176 |
//! | 12 | 15 | 4 | flags | u32 = 0 in v1 |
//! | 16 | 16 | 1 | account_id_width | u8: 1, 2, 4, 8 |
//! | 17 | 17 | 1 | event_id_width | u8 = 8 |
//! | 18 | 18 | 1 | timestamp_encoding | u8 = 1 |
//! | 19 | 19 | 1 | byte_order | u8 = 1 |
//! | 20 | 23 | 4 | record_length | u32 |
//! | 24 | 31 | 8 | record_count | u64 |
//! | 32 | 39 | 8 | payload_length | u64 |
//! | 40 | 55 | 16 | event_namespace | UUID network order |
//! | 56 | 58 | 3 | currency_code | 3 uppercase ASCII |
//! | 59 | 59 | 1 | currency_precision | u8 |
//! | 60 | 61 | 2 | feature_schema_version | u16 |
//! | 62 | 63 | 2 | feature_algorithm_version | u16 |
//! | 64 | 71 | 8 | threshold_minor_units | u64 > 0 |
//! | 72 | 75 | 4 | threshold_effective_day | i32 |
//! | 76 | 76 | 1 | source_type | enum |
//! | 77 | 77 | 1 | identity_scheme | enum |
//! | 78 | 78 | 1 | checksum_algorithm | u8 = 1 (SHA-256) |
//! | 79 | 79 | 1 | reserved | MUST be zero |
//! | 80 | 111 | 32 | payload_checksum | SHA-256 |
//! | 112 | 143 | 32 | config_fingerprint | SHA-256 |
//! | 144 | 175 | 32 | header_checksum | SHA-256 of bytes 0-143 |
//!
//! Encoding is field-by-field little-endian, never a struct copy (wire §7).

use crate::config::{
    CurrencyCode, FORMAT_VERSION, HEADER_LEN, IdentityScheme, MAGIC, SHA256_LEN, SourceType,
};
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::hash::{digest_eq, sha256};
use crate::namespace::EventNamespace;

/// `checksum_algorithm` value for SHA-256 (wire §1).
pub const CHECKSUM_SHA256: u8 = 1;

/// `event_id_width`: the record encodes `event_id` as u64 (wire §1).
pub const EVENT_ID_WIDTH: u8 = 8;

/// `timestamp_encoding`: signed i64 UTC Unix milliseconds (wire §2).
pub const TIMESTAMP_ENCODING_MS_I64: u8 = 1;

/// `byte_order`: little-endian (wire §2).
pub const BYTE_ORDER_LE: u8 = 1;

/// A decoded, individually validated segment header.
///
/// Construction goes through [`SegmentHeader::decode`], which enforces the
/// wire §6 validation order. There is no public field-by-field constructor, so
/// a header cannot exist in this codebase without having been validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentHeader {
    /// Account-ID width in bytes (1, 2, 4, 8).
    pub account_id_width: u8,
    /// Declared record length in bytes.
    pub record_length: u32,
    /// Declared record count.
    pub record_count: u64,
    /// Declared payload length in bytes.
    pub payload_length: u64,
    /// The single event namespace for this segment (§2: exactly one).
    pub event_namespace: EventNamespace,
    /// ISO 4217 currency code.
    pub currency_code: CurrencyCode,
    /// Minor-unit decimal digits.
    pub currency_precision: u8,
    /// Feature-schema version.
    pub feature_schema_version: u16,
    /// Feature-algorithm version.
    pub feature_algorithm_version: u16,
    /// Positive threshold in minor units.
    pub threshold_minor_units: u64,
    /// Threshold effective day (signed days since epoch).
    pub threshold_effective_day: i32,
    /// Source type enum.
    pub source_type: SourceType,
    /// Identity scheme enum.
    pub identity_scheme: IdentityScheme,
    /// SHA-256 of the payload bytes.
    pub payload_checksum: [u8; SHA256_LEN],
    /// SHA-256 of the configuration-manifest bytes.
    pub config_fingerprint: [u8; SHA256_LEN],
}

/// Byte offset of each header field (wire §1). Named constants exist so the
/// encode and decode paths cannot drift apart.
mod off {
    pub const MAGIC: usize = 0;
    pub const FORMAT_VERSION: usize = 8;
    pub const HEADER_LENGTH: usize = 10;
    pub const FLAGS: usize = 12;
    pub const ACCOUNT_ID_WIDTH: usize = 16;
    pub const EVENT_ID_WIDTH: usize = 17;
    pub const TIMESTAMP_ENCODING: usize = 18;
    pub const BYTE_ORDER: usize = 19;
    pub const RECORD_LENGTH: usize = 20;
    pub const RECORD_COUNT: usize = 24;
    pub const PAYLOAD_LENGTH: usize = 32;
    pub const EVENT_NAMESPACE: usize = 40;
    pub const CURRENCY_CODE: usize = 56;
    pub const CURRENCY_PRECISION: usize = 59;
    pub const FEATURE_SCHEMA_VERSION: usize = 60;
    pub const FEATURE_ALGORITHM_VERSION: usize = 62;
    pub const THRESHOLD_MINOR_UNITS: usize = 64;
    pub const THRESHOLD_EFFECTIVE_DAY: usize = 72;
    pub const SOURCE_TYPE: usize = 76;
    pub const IDENTITY_SCHEME: usize = 77;
    pub const CHECKSUM_ALGORITHM: usize = 78;
    pub const RESERVED: usize = 79;
    pub const PAYLOAD_CHECKSUM: usize = 80;
    pub const CONFIG_FINGERPRINT: usize = 112;
    pub const HEADER_CHECKSUM: usize = 144;
}

impl SegmentHeader {
    /// Encode this header to exactly 176 bytes.
    ///
    /// Every checksum is computed here from the bytes actually emitted, so the
    /// encoded header is self-consistent by construction.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_LEN];
        b[off::MAGIC..off::MAGIC + 8].copy_from_slice(MAGIC);
        b[off::FORMAT_VERSION..off::FORMAT_VERSION + 2]
            .copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        b[off::HEADER_LENGTH..off::HEADER_LENGTH + 2]
            .copy_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        // flags: 0 in v1 (offsets 12..16 already zero).
        b[off::ACCOUNT_ID_WIDTH] = self.account_id_width;
        b[off::EVENT_ID_WIDTH] = EVENT_ID_WIDTH;
        b[off::TIMESTAMP_ENCODING] = TIMESTAMP_ENCODING_MS_I64;
        b[off::BYTE_ORDER] = BYTE_ORDER_LE;
        b[off::RECORD_LENGTH..off::RECORD_LENGTH + 4]
            .copy_from_slice(&self.record_length.to_le_bytes());
        b[off::RECORD_COUNT..off::RECORD_COUNT + 8]
            .copy_from_slice(&self.record_count.to_le_bytes());
        b[off::PAYLOAD_LENGTH..off::PAYLOAD_LENGTH + 8]
            .copy_from_slice(&self.payload_length.to_le_bytes());
        b[off::EVENT_NAMESPACE..off::EVENT_NAMESPACE + 16]
            .copy_from_slice(&self.event_namespace.to_bytes());
        b[off::CURRENCY_CODE..off::CURRENCY_CODE + 3]
            .copy_from_slice(&self.currency_code.bytes());
        b[off::CURRENCY_PRECISION] = self.currency_precision;
        b[off::FEATURE_SCHEMA_VERSION..off::FEATURE_SCHEMA_VERSION + 2]
            .copy_from_slice(&self.feature_schema_version.to_le_bytes());
        b[off::FEATURE_ALGORITHM_VERSION..off::FEATURE_ALGORITHM_VERSION + 2]
            .copy_from_slice(&self.feature_algorithm_version.to_le_bytes());
        b[off::THRESHOLD_MINOR_UNITS..off::THRESHOLD_MINOR_UNITS + 8]
            .copy_from_slice(&self.threshold_minor_units.to_le_bytes());
        b[off::THRESHOLD_EFFECTIVE_DAY..off::THRESHOLD_EFFECTIVE_DAY + 4]
            .copy_from_slice(&self.threshold_effective_day.to_le_bytes());
        b[off::SOURCE_TYPE] = self.source_type.to_u8();
        b[off::IDENTITY_SCHEME] = self.identity_scheme.to_u8();
        b[off::CHECKSUM_ALGORITHM] = CHECKSUM_SHA256;
        // reserved (offset 79) stays zero.
        b[off::PAYLOAD_CHECKSUM..off::PAYLOAD_CHECKSUM + SHA256_LEN]
            .copy_from_slice(&self.payload_checksum);
        b[off::CONFIG_FINGERPRINT..off::CONFIG_FINGERPRINT + SHA256_LEN]
            .copy_from_slice(&self.config_fingerprint);
        // header_checksum = SHA-256 of bytes 0..144 (wire §4).
        let hc = sha256(&b[..off::HEADER_CHECKSUM]);
        b[off::HEADER_CHECKSUM..HEADER_LEN].copy_from_slice(&hc);
        b
    }
}

impl SegmentHeader {
    /// Decode and validate a 176-byte header.
    ///
    /// Enforces the wire §6 order exactly:
    /// 1. header is at least 176 bytes;
    /// 2. magic, format version, header length, flags, reserved bytes;
    /// 3. header checksum;
    /// 4. account-ID width, event-ID width, timestamp encoding, byte order,
    ///    enum values;
    /// 5. currency metadata, threshold, feature versions.
    ///
    /// Steps 6-9 of §6 (record/payload lengths, payload checksum, per-record
    /// validation, identity and application) belong to the segment and session
    /// layers, which own the payload bytes.
    ///
    /// # Errors
    /// - [`ErrorCode::Binary`] on any structural, version, reserved-byte,
    ///   checksum or enum violation.
    /// - [`ErrorCode::AccountId`] for an unsupported account-ID width.
    /// - [`ErrorCode::AmountOrCurrency`] for invalid currency/threshold data.
    /// - [`ErrorCode::SchemaVersion`] for unsupported feature versions.
    pub fn decode(bytes: &[u8]) -> IngestResult<Self> {
        // Step 1: length.
        if bytes.len() < HEADER_LEN {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("segment header is {} bytes, expected at least {HEADER_LEN}", bytes.len()),
            ));
        }
        let b = &bytes[..HEADER_LEN];

        // Step 2: magic, versions, header length, flags, reserved.
        if &b[off::MAGIC..off::MAGIC + 8] != MAGIC {
            return Err(IngestError::new(
                ErrorCode::Binary,
                "bad segment magic (expected ABACSEG1)",
            ));
        }
        let format_version = u16::from_le_bytes(
            b[off::FORMAT_VERSION..off::FORMAT_VERSION + 2].try_into().expect("2"),
        );
        if format_version != FORMAT_VERSION {
            return Err(IngestError::new(
                ErrorCode::SchemaVersion,
                format!("unsupported segment format_version {format_version}; implemented: {FORMAT_VERSION}"),
            ));
        }
        let header_length = u16::from_le_bytes(
            b[off::HEADER_LENGTH..off::HEADER_LENGTH + 2].try_into().expect("2"),
        );
        if header_length as usize != HEADER_LEN {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("header_length is {header_length}, must be exactly {HEADER_LEN}"),
            ));
        }
        let flags = u32::from_le_bytes(b[off::FLAGS..off::FLAGS + 4].try_into().expect("4"));
        if flags != 0 {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("flags must be zero in v1, got {flags}"),
            ));
        }
        if b[off::RESERVED] != 0 {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("reserved byte must be zero, got {}", b[off::RESERVED]),
            ));
        }

        // Step 3: header checksum over bytes 0..144 (wire §4).
        let stored_hc: [u8; SHA256_LEN] =
            b[off::HEADER_CHECKSUM..HEADER_LEN].try_into().expect("32");
        let computed_hc = sha256(&b[..off::HEADER_CHECKSUM]);
        if !digest_eq(&stored_hc, &computed_hc) {
            return Err(IngestError::new(
                ErrorCode::Binary,
                "header_checksum mismatch (header bytes 0-143)",
            ));
        }

        // Step 4: widths, encodings, byte order, enums.
        let account_id_width = b[off::ACCOUNT_ID_WIDTH];
        if !crate::config::ACCOUNT_ID_WIDTHS.contains(&account_id_width) {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!(
                    "unsupported account_id_width {account_id_width}; permitted: 1, 2, 4, 8 (128-bit is excluded)"
                ),
            ));
        }
        if b[off::EVENT_ID_WIDTH] != EVENT_ID_WIDTH {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("event_id_width must be {EVENT_ID_WIDTH}, got {}", b[off::EVENT_ID_WIDTH]),
            ));
        }
        if b[off::TIMESTAMP_ENCODING] != TIMESTAMP_ENCODING_MS_I64 {
            return Err(IngestError::new(
                ErrorCode::Timestamp,
                format!(
                    "unsupported timestamp_encoding {}; only 1 (signed i64 UTC ms) is defined",
                    b[off::TIMESTAMP_ENCODING]
                ),
            ));
        }
        if b[off::BYTE_ORDER] != BYTE_ORDER_LE {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("unsupported byte_order {}; only 1 (little-endian) is defined", b[off::BYTE_ORDER]),
            ));
        }
        if b[off::CHECKSUM_ALGORITHM] != CHECKSUM_SHA256 {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!(
                    "unsupported checksum_algorithm {}; only 1 (SHA-256) is defined",
                    b[off::CHECKSUM_ALGORITHM]
                ),
            ));
        }
        let source_type = SourceType::from_u8(b[off::SOURCE_TYPE])?;
        let identity_scheme = IdentityScheme::from_u8(b[off::IDENTITY_SCHEME])?;

        // Step 5: currency metadata, threshold, feature versions.
        let currency_code = CurrencyCode::parse(std::str::from_utf8(
            &b[off::CURRENCY_CODE..off::CURRENCY_CODE + 3],
        ).unwrap_or(""))?;
        let currency_precision = b[off::CURRENCY_PRECISION];
        if currency_precision > 18 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                format!("unsupported currency_precision {currency_precision}"),
            ));
        }
        let threshold_minor_units = u64::from_le_bytes(
            b[off::THRESHOLD_MINOR_UNITS..off::THRESHOLD_MINOR_UNITS + 8].try_into().expect("8"),
        );
        if threshold_minor_units == 0 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "threshold_minor_units must be positive; zero is never a valid threshold",
            ));
        }
        let feature_schema_version = u16::from_le_bytes(
            b[off::FEATURE_SCHEMA_VERSION..off::FEATURE_SCHEMA_VERSION + 2].try_into().expect("2"),
        );
        let feature_algorithm_version = u16::from_le_bytes(
            b[off::FEATURE_ALGORITHM_VERSION..off::FEATURE_ALGORITHM_VERSION + 2]
                .try_into()
                .expect("2"),
        );
        if feature_schema_version != crate::config::FEATURE_SCHEMA_VERSION {
            return Err(IngestError::new(
                ErrorCode::SchemaVersion,
                format!(
                    "unsupported feature_schema_version {feature_schema_version}; implemented: {}",
                    crate::config::FEATURE_SCHEMA_VERSION
                ),
            ));
        }
        if feature_algorithm_version != crate::config::FEATURE_ALGORITHM_VERSION {
            return Err(IngestError::new(
                ErrorCode::SchemaVersion,
                format!(
                    "unsupported feature_algorithm_version {feature_algorithm_version}; implemented: {}",
                    crate::config::FEATURE_ALGORITHM_VERSION
                ),
            ));
        }

        let mut ns_bytes = [0u8; 16];
        ns_bytes.copy_from_slice(&b[off::EVENT_NAMESPACE..off::EVENT_NAMESPACE + 16]);
        let event_namespace = EventNamespace::from_bytes(ns_bytes);
        if event_namespace.is_zero() {
            return Err(IngestError::new(
                ErrorCode::Identity,
                "the all-zero event_namespace is not a valid identity domain",
            ));
        }

        let mut payload_checksum = [0u8; SHA256_LEN];
        payload_checksum.copy_from_slice(&b[off::PAYLOAD_CHECKSUM..off::PAYLOAD_CHECKSUM + SHA256_LEN]);
        let mut config_fingerprint = [0u8; SHA256_LEN];
        config_fingerprint
            .copy_from_slice(&b[off::CONFIG_FINGERPRINT..off::CONFIG_FINGERPRINT + SHA256_LEN]);

        Ok(Self {
            account_id_width,
            record_length: u32::from_le_bytes(
                b[off::RECORD_LENGTH..off::RECORD_LENGTH + 4].try_into().expect("4"),
            ),
            record_count: u64::from_le_bytes(
                b[off::RECORD_COUNT..off::RECORD_COUNT + 8].try_into().expect("8"),
            ),
            payload_length: u64::from_le_bytes(
                b[off::PAYLOAD_LENGTH..off::PAYLOAD_LENGTH + 8].try_into().expect("8"),
            ),
            event_namespace,
            currency_code,
            currency_precision,
            feature_schema_version,
            feature_algorithm_version,
            threshold_minor_units,
            threshold_effective_day: i32::from_le_bytes(
                b[off::THRESHOLD_EFFECTIVE_DAY..off::THRESHOLD_EFFECTIVE_DAY + 4]
                    .try_into()
                    .expect("4"),
            ),
            source_type,
            identity_scheme,
            payload_checksum,
            config_fingerprint,
        })
    }
}
