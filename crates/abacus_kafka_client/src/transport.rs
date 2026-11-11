//! The bounded Kafka transport seam.
//!
//! Gap B of the discovery report is a *dependency* decision: adding a real
//! broker client is outside any governing specification, and this workspace is
//! deliberately std-only apart from `sha2`. Rather than silently pulling in a
//! heavy crate, the broker is isolated behind this trait — the same approach
//! [`abacus_csv_client`] takes with its `Transport` seam. The in-memory
//! implementation below lets every schema-independent path be tested offline;
//! a real broker adapter can implement [`MessageSource`] later without touching
//! canonical logic.

use abacus_ingest::IngestResult;

/// One message delivered by the broker.
///
/// The `payload` is **opaque**: its structure is the unresolved Kafka message
/// schema (Gap A), decoded only by a [`crate::map::MessageMapper`]. This type
/// deliberately carries no assumption about that schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KafkaMessage {
    /// Kafka partition number.
    pub partition: i32,
    /// Kafka offset within the partition. Non-negative offsets become the
    /// canonical `event_id` (see [`crate::partition::event_id_for_offset`]).
    pub offset: i64,
    /// Optional message key (opaque bytes).
    pub key: Vec<u8>,
    /// The message value (opaque bytes; schema defined by Gap A).
    pub payload: Vec<u8>,
}

/// A bounded source of Kafka messages.
///
/// `poll` returns **at most** `max` messages for `partition` starting at
/// `from_offset`, in offset order. Bounded by construction: a caller can never
/// request an unbounded read, which keeps memory flat regardless of backlog.
pub trait MessageSource {
    /// Fetch up to `max` messages for `partition` from `from_offset`.
    ///
    /// # Errors
    /// [`ErrorCode::Persistence`] on a transport failure (the trait's only
    /// sanctioned error; parse-level problems are the mapper's concern).
    fn poll(
        &mut self,
        partition: i32,
        from_offset: i64,
        max: usize,
    ) -> IngestResult<Vec<KafkaMessage>>;
}

/// An in-memory [`MessageSource`] for offline tests and local streaming_ingestion_demo.
///
/// Not a broker: it serves messages previously pushed in, so the transport
/// seam and everything above it can be exercised without any Kafka
/// infrastructure.
#[derive(Debug, Default)]
pub struct InMemorySource {
    messages: Vec<KafkaMessage>,
}

impl InMemorySource {
    /// An empty source.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push one message.
    pub fn push(&mut self, msg: KafkaMessage) {
        self.messages.push(msg);
    }
}

impl MessageSource for InMemorySource {
    fn poll(
        &mut self,
        partition: i32,
        from_offset: i64,
        max: usize,
    ) -> IngestResult<Vec<KafkaMessage>> {
        let mut matches: Vec<KafkaMessage> = self
            .messages
            .iter()
            .filter(|m| m.partition == partition && m.offset >= from_offset)
            .cloned()
            .collect();
        // Order first, then bound: a bounded read returns the earliest `max`
        // offsets at or after `from_offset` (a real consumer reads in offset
        // order), not an arbitrary `max` in arrival order.
        matches.sort_by_key(|m| m.offset);
        matches.truncate(max);
        Ok(matches)
    }
}
