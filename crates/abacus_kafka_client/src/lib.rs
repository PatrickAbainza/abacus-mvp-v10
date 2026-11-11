//! `abacus_kafka_client` — the Kafka source adapter for the Abacus ingestion
//! pipeline, following the structure and conventions of `abacus_csv_client`.
//!
//! It reuses the `/abacus_ingestion` oracle for every canonical concern — the
//! frozen seven-byte feature derivation, the 23+2W canonical record, the
//! 176-byte `ABACSEG1` envelope, the compound `(namespace, event_id)` identity
//! model, the validated `SourceConfig`, and the `SubstrateApplier` commit path —
//! and adds only the Kafka-specific surface:
//!
//! | Module | Responsibility | Status |
//! |--------|----------------|--------|
//! | [`config`] | `SourceType::Kafka` + `KafkaPartitionOffset` configuration | done |
//! | [`namespace_store`] | persistent per-partition `event_namespace` | done |
//! | [`partition`] | offset-as-`event_id` + resume/idempotency semantics | done |
//! | [`transport`] | bounded broker seam (no dependency added) | done |
//! | [`map`] | message → canonical record + `ABACSEG1` encoding | **seam only** |
//!
//! ## Blocked: the Kafka message schema (Gap A)
//!
//! The one unresolved piece is decoding a broker message into canonical fields
//! ([`map::MessageMapper`]). No governing specification defines the topic
//! payload, so this adapter does not invent field names, serialization, timestamp
//! rules or amount semantics. Everything that does not depend on the schema is
//! implemented and tested; see `KAFKA_SCHEMA_GAP.md` for the exact decision
//! required to finish the mapper and the transport.
//!
//! Kafka integration is therefore **not** claimed complete: a broker consumer
//! and message mapping remain unimplemented by design.

#![warn(missing_docs)]

pub mod config;
pub mod map;
pub mod namespace_store;
pub mod partition;
pub mod transport;

pub use config::KafkaConfig;
pub use map::{build_record, encode_batch, MappedEvent, MessageMapper};
pub use namespace_store::PartitionNamespaceStore;
pub use partition::{event_id_for_offset, PartitionCursor};
pub use transport::{InMemorySource, KafkaMessage, MessageSource};
