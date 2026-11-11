//! Shared fixtures for the integration tests.
//!
//! Deterministic by construction: fixed namespace, fixed configuration, fixed
//! amounts. No clock, no randomness, no filesystem.

#![allow(dead_code)]

use abacus_ingest::config::{
    CurrencyCode, FEATURE_ALGORITHM_VERSION, FEATURE_SCHEMA_VERSION, IdentityScheme, SourceConfig,
    SourceType,
};
use abacus_ingest::features::FeaturePayload;
use abacus_ingest::namespace::EventNamespace;
use abacus_ingest::record::CanonicalRecord;
use abacus_ingest::session::SessionLimits;

/// The threshold used by every fixture, in minor units (100.00 at precision 2).
pub const THRESHOLD: u64 = 10_000;
/// The major-unit scale used by every fixture (precision 2).
pub const MAJOR_SCALE: u64 = 100;

/// A fixed, non-zero namespace for reproducible fixtures.
pub fn ns() -> EventNamespace {
    EventNamespace::from_bytes([
        0x9a, 0x1f, 0x4c, 0x07, 0x6b, 0x2d, 0x4e, 0x81, 0x93, 0x05, 0xc7, 0x22, 0x48, 0xbd, 0x6e,
        0x10,
    ])
}

/// A second, distinct fixed namespace (for cross-namespace tests).
pub fn ns_other() -> EventNamespace {
    EventNamespace::from_bytes([
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x00,
    ])
}

/// Build a validated configuration at the given account-ID width.
pub fn config(width: u8) -> SourceConfig {
    SourceConfig {
        source_type: SourceType::SamldFixture,
        identity_scheme: IdentityScheme::SamldRecordOrdinal,
        account_id_width: width,
        currency_code: CurrencyCode::parse("USD").unwrap(),
        currency_precision: 2,
        threshold_minor_units: THRESHOLD,
        threshold_effective_day: 19_000,
        threshold_config_version: 3,
        feature_schema_version: FEATURE_SCHEMA_VERSION,
        feature_algorithm_version: FEATURE_ALGORITHM_VERSION,
        parsing_policy_version: 1,
        max_amount_minor_units: 100_000_000_000,
    }
}

/// Build one canonical record with derived features.
pub fn record(event_id: u64, t_ms: i64, sender: u64, receiver: u64, amount: u64) -> CanonicalRecord {
    CanonicalRecord {
        identity: abacus_ingest::EventIdentity::new(ns(), event_id),
        event_time_ms: t_ms,
        sender_id: sender,
        receiver_id: receiver,
        features: FeaturePayload::derive(amount, THRESHOLD, MAJOR_SCALE).unwrap(),
    }
}

/// Generous limits for tests that are not exercising the limit paths.
pub fn wide_limits() -> SessionLimits {
    SessionLimits {
        max_chunk_bytes: 64 * 1024 * 1024,
        max_chunk_records: 1_000_000,
        max_session_bytes: 256 * 1024 * 1024,
        max_session_records: 1_000_000,
        max_chunks: 10_000,
    }
}

/// Tight limits for the limit-enforcement tests.
pub fn tight_limits() -> SessionLimits {
    SessionLimits {
        max_chunk_bytes: 400,
        max_chunk_records: 2,
        max_session_bytes: 1_000,
        max_session_records: 4,
        max_chunks: 2,
    }
}

/// Attestation closure that supplies a deterministic amount per record.
///
/// Stands in for a trusted adapter's normalization boundary: the SAML-D and
/// Kafka adapters have the exact amount available and supply it here.
pub fn attest_fixed(amount: i64) -> impl FnMut(usize, &CanonicalRecord) -> Option<i64> {
    move |_i, _r| Some(amount)
}

/// Recompute the header checksum after a header field was mutated.
///
/// This models a producer that legitimately *declared* the mutated value, so
/// validation proceeds past wire §6 step 3 (the header checksum) and reaches
/// the specific field check under test. Without it, every header mutation is
/// caught by the checksum first — which is itself correct behavior, but tests
/// the wrong thing.
pub fn resign_header(bytes: &mut [u8]) {
    let hc = abacus_ingest::hash::sha256(&bytes[..144]);
    bytes[144..176].copy_from_slice(&hc);
}

/// Recompute the payload checksum, then the header checksum.
///
/// Models a producer that legitimately emitted the mutated payload.
pub fn resign_payload_then_header(bytes: &mut [u8]) {
    let pc = abacus_ingest::hash::sha256(&bytes[176..]);
    bytes[80..112].copy_from_slice(&pc);
    resign_header(bytes);
}

/// Read a `u16` little-endian field at `offset`.
pub fn rd_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Read a `u64` little-endian field at `offset`.
pub fn rd_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(b)
}
