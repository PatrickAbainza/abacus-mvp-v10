//! The message → canonical-record seam.
//!
//! ## Gap A (blocked): the Kafka message schema
//!
//! No governing specification (`ABACUS INGESTION PIPELINE.md`, the Canonical
//! Contract, or the Wire Format) defines the Kafka topic payload — its
//! serialization, or which fields carry sender, receiver, amount and time. Per
//! the Golden Source Compliance Rule this adapter therefore does **not** invent
//! field names, formats, timestamp rules or amount semantics. The
//! [`MessageMapper`] trait below is the exact seam a concrete decoder will fill
//! once the schema is decided; it is intentionally left unimplemented here.
//!
//! ## What *is* implemented and tested
//!
//! Everything **above** the schema boundary is complete and independently
//! testable: given the canonical fields (however they are extracted),
//! [`build_record`] derives the frozen seven-byte feature payload via the shared
//! oracle and assembles a [`CanonicalRecord`] under the compound
//! `(namespace, offset)` identity, and [`encode_batch`] emits a byte-exact
//! `ABACSEG1` segment. Those paths are exercised by the crate's tests without
//! any knowledge of the payload format.

use abacus_ingest::features::FeaturePayload;
use abacus_ingest::record::CanonicalRecord;
use abacus_ingest::wire::encode_segment;
use abacus_ingest::{ErrorCode, IngestError};
use abacus_ingest::{EventIdentity, EventNamespace, IngestResult, SourceConfig};

use crate::partition::event_id_for_offset;
use crate::transport::KafkaMessage;

/// The canonical fields one mapped Kafka message must supply.
///
/// A [`MessageMapper`] produces exactly these; nothing about the wire format
/// leaks past it into canonical ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedEvent {
    /// UTC milliseconds since the Unix epoch (canonical wire unit, Contract §2).
    pub event_time_ms: i64,
    /// Opaque sender account id.
    pub sender_id: u64,
    /// Opaque receiver account id.
    pub receiver_id: u64,
    /// Exact nonnegative amount in currency minor units (A1 attested value).
    pub amount_minor_units: u64,
}

/// Decode a Kafka message into its canonical fields.
///
/// **Unresolved (Gap A):** no implementation is provided because the topic
/// message schema is not specified. A future decoder implements this trait once
/// the schema, serialization, timestamp rule and amount semantics are decided;
/// every other part of this crate is already wired to consume its output.
pub trait MessageMapper {
    /// Map one message to its canonical fields.
    ///
    /// # Errors
    /// A structural parse/validation rejection for a malformed message (the
    /// concrete error taxonomy is the decoder's, expressed through
    /// [`IngestError`]).
    fn map(&self, msg: &KafkaMessage) -> IngestResult<MappedEvent>;
}

/// The substrate's minor-unit scale is `10^6`; the *feature* derivation's major
/// scale is `10^currency_precision`. This returns the latter.
fn major_scale_for(config: &SourceConfig) -> IngestResult<u64> {
    10u64
        .checked_pow(u32::from(config.currency_precision))
        .ok_or_else(|| {
            IngestError::new(
                ErrorCode::AmountOrCurrency,
                "currency_precision is too large to form the major scale",
            )
        })
}

/// Build one [`CanonicalRecord`] from mapped fields under the compound identity.
///
/// The event id is the Kafka offset (Wire §2); the namespace is the partition's
/// persistent namespace. The frozen seven-byte feature payload is derived by the
/// shared oracle implementation — never reimplemented (Golden Source §9.1) — and
/// an out-of-range amount is rejected, never clamped (§19).
///
/// # Errors
/// - [`ErrorCode::Binary`] for a negative offset (not a valid event id).
/// - [`ErrorCode::FeatureOverflow`] for an amount whose features are not
///   representable.
/// - [`ErrorCode::AmountOrCurrency`] for an unformable major scale.
pub fn build_record(
    config: &SourceConfig,
    namespace: EventNamespace,
    offset: i64,
    mapped: &MappedEvent,
) -> IngestResult<CanonicalRecord> {
    let event_id = event_id_for_offset(offset)?;
    let features = FeaturePayload::derive(
        mapped.amount_minor_units,
        config.threshold_minor_units,
        major_scale_for(config)?,
    )?;
    Ok(CanonicalRecord {
        identity: EventIdentity::new(namespace, event_id),
        event_time_ms: mapped.event_time_ms,
        sender_id: mapped.sender_id,
        receiver_id: mapped.receiver_id,
        features,
    })
}

/// Encode a batch of records into one byte-exact `ABACSEG1` segment.
///
/// Every record must share `namespace` (Contract §2: exactly one namespace per
/// segment); the encoder enforces that and recomputes the header/payload
/// checksums.
///
/// # Errors
/// As [`encode_segment`]: a namespace mismatch, an identifier that does not fit
/// the declared width, or any header/encoding rejection.
pub fn encode_batch(
    config: &SourceConfig,
    namespace: EventNamespace,
    records: &[CanonicalRecord],
) -> IngestResult<Vec<u8>> {
    encode_segment(config, namespace, records)
}
