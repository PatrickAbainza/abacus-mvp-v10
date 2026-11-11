//! Kafka partition offsets as canonical event identity, and the idempotency
//! relationship they establish.
//!
//! Wire §2 fixes `IdentityScheme::KafkaPartitionOffset` (=3): the Kafka
//! partition offset **is** the canonical `event_id` (Canonical Contract §3.1).
//! This module makes that mapping explicit and defines its interaction with
//! resume and idempotency:
//!
//! * `event_id = offset`, so a message streaming_ingestion_demoed after a crash carries the same
//!   compound identity `(namespace, offset)`.
//! * Event-level idempotency is therefore the substrate's own CAS: applying the
//!   same `(namespace, event_id)` twice resolves to the same substrate id and
//!   the second application is a `Retry` (already applied) — exactly-once at the
//!   model level (P7), the same guarantee the CSV and fixture paths rely on.
//! * [`PartitionCursor`] tracks the consumer resume position: the next offset to
//!   read for a partition, derived from the last processed offset.
//!
//! Kafka's special negative offsets (`-2` earliest, `-1` latest) are **not**
//! valid event ids and are rejected rather than coerced.

use abacus_ingest::{ErrorCode, IngestError, IngestResult};

/// Map a Kafka offset to the canonical `event_id`.
///
/// # Errors
/// [`ErrorCode::Binary`] for a negative offset (a special sentinel such as
/// `earliest`/`latest`), which cannot serve as a stable event id.
pub fn event_id_for_offset(offset: i64) -> IngestResult<u64> {
    if offset < 0 {
        return Err(IngestError::new(
            ErrorCode::Binary,
            "a negative Kafka offset (e.g. earliest/latest sentinel) is not a valid event_id",
        ));
    }
    Ok(offset as u64)
}

/// The consumer resume position for one partition.
///
/// `next_offset` is the next offset to poll; it advances past every offset that
/// was successfully processed. Persisting this (alongside the namespace store)
/// is what lets a restarted consumer resume without re-reading or skipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionCursor {
    /// The partition this cursor tracks.
    pub partition: i32,
    /// The next offset to read from the partition.
    pub next_offset: i64,
}

impl PartitionCursor {
    /// A cursor positioned at the start of `partition`.
    #[must_use]
    pub const fn start(partition: i32) -> Self {
        Self {
            partition,
            next_offset: 0,
        }
    }

    /// A cursor positioned at an arbitrary resume offset.
    #[must_use]
    pub const fn at(partition: i32, next_offset: i64) -> Self {
        Self {
            partition,
            next_offset,
        }
    }

    /// Advance the cursor past `processed_offset`.
    ///
    /// Monotonic: an out-of-order or already-processed offset never moves the
    /// cursor backwards, so a redelivered message cannot cause a rewind.
    pub fn advance_past(&mut self, processed_offset: i64) {
        if processed_offset + 1 > self.next_offset {
            self.next_offset = processed_offset + 1;
        }
    }
}
