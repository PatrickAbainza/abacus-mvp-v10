//! Segment construction from validated configuration and records.
//!
//! The builder exists so that tests, the SAML-D fixture adapter and the Kafka
//! adapter can all produce byte-exact ABACSEG1 segments through one path. It
//! never bypasses validation: it derives every header field from a validated
//! [`SourceConfig`] so the emitted header cannot disagree with the records.

use crate::config::{SourceConfig, HEADER_LEN};
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::hash::sha256;
use crate::record::{record_length_for_width, CanonicalRecord};
use crate::wire::header::SegmentHeader;
use crate::wire::segment::Segment;

/// Assemble a segment from a validated configuration, a namespace and records.
///
/// # Errors
/// - [`ErrorCode::AccountId`] if the configuration's account-ID width is
///   unsupported.
/// - [`ErrorCode::Identity`] if any record's namespace differs from
///   `namespace` (§2: exactly one namespace per segment).
/// - Any [`SourceConfig::validate`] rejection.
pub fn build_segment(
    config: &SourceConfig,
    namespace: crate::namespace::EventNamespace,
    records: &[CanonicalRecord],
) -> IngestResult<Segment> {
    config.validate()?;
    let record_length = record_length_for_width(config.account_id_width)? as u32;
    for (i, r) in records.iter().enumerate() {
        if r.identity.namespace != namespace {
            return Err(IngestError::new(
                ErrorCode::Identity,
                format!(
                    "record {i} belongs to a different namespace; a segment must contain exactly one namespace"
                ),
            ));
        }
        // Confirm each record actually fits the declared width.
        let mask = if config.account_id_width >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * u32::from(config.account_id_width))) - 1
        };
        if r.sender_id > mask || r.receiver_id > mask {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!("record {i} has an identifier that does not fit the declared width"),
            ));
        }
        r.features.validate()?;
    }
    let header = SegmentHeader {
        account_id_width: config.account_id_width,
        record_length,
        record_count: records.len() as u64,
        payload_length: (records.len() * record_length as usize) as u64,
        event_namespace: namespace,
        currency_code: config.currency_code,
        currency_precision: config.currency_precision,
        feature_schema_version: config.feature_schema_version,
        feature_algorithm_version: config.feature_algorithm_version,
        threshold_minor_units: config.threshold_minor_units,
        threshold_effective_day: config.threshold_effective_day,
        source_type: config.source_type,
        identity_scheme: config.identity_scheme,
        // Placeholder; `Segment::encode` recomputes from the real payload.
        payload_checksum: [0u8; 32],
        config_fingerprint: config.fingerprint(),
    };
    Ok(Segment {
        header,
        records: records.to_vec(),
    })
}

/// Convenience: build a segment and return its exact bytes.
///
/// # Errors
/// As [`build_segment`], plus any encoding rejection.
pub fn encode_segment(
    config: &SourceConfig,
    namespace: crate::namespace::EventNamespace,
    records: &[CanonicalRecord],
) -> IngestResult<Vec<u8>> {
    let seg = build_segment(config, namespace, records)?;
    seg.encode()
}

/// The declared header length, re-exported for callers that need to split a
/// segment buffer without depending on `config`.
pub const HEADER_BYTES: usize = HEADER_LEN;

/// SHA-256 helper re-export so adapters can fingerprint payloads consistently.
#[must_use]
pub fn payload_checksum(payload: &[u8]) -> [u8; 32] {
    sha256(payload)
}
