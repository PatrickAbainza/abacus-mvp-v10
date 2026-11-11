# Kafka adapter: unresolved specification gaps

This document records the two specification gaps identified in the discovery
report (`docs/ingestion_oracle_discovery.md`) that gate completion of the Kafka
adapter, states exactly what is and is not implemented, and names the decisions
required to finish it. Nothing here invents behavior to paper over a gap
(Golden Source Compliance Rule).

## Gap A — Kafka message schema (BLOCKS message→record mapping)

**Status: unresolved. No governing section defines it.**

The three authoritative specifications define the *canonical* event schema the
Kafka data must converge on (compound `(namespace, event_id)`, `event_time_ms`,
opaque `sender_id`/`receiver_id`, the frozen 7-byte feature payload, the
`ABACSEG1` envelope, `SourceType::Kafka`, `IdentityScheme::KafkaPartitionOffset`).
None of them defines the **topic payload** a broker actually delivers:

- serialization format (JSON? Avro? Protobuf? raw bytes?);
- which field carries the timestamp, and in what unit/timezone;
- which fields carry sender and receiver account identifiers;
- which field carries the amount, in what minor-unit/precision.

Because of this, `src/map.rs` provides the `MessageMapper` trait as the exact
seam a decoder fills, plus schema-independent `build_record` / `encode_batch`
helpers, but **no concrete decoder is implemented**. Inventing field names,
formats, timestamp rules or amount semantics would violate the Golden Source
Compliance Rule and could silently change event meaning.

**Decision required to unblock:** the authoritative Kafka topic message schema —
serialization format and the field mapping (timestamp + unit, sender, receiver,
amount + precision). Once provided, a `MessageMapper` implementation is a small,
mechanical addition; every other part of the adapter already consumes its
output.

## Gap B — Transport / dependency + namespace store

**Status: partially resolved.**

Two sub-decisions:

1. **Broker client dependency.** The ingestion workspace is deliberately
   std-only apart from `sha2`, and adding a real broker client is a dependency
   decision outside any specification. This adapter therefore isolates the
   broker behind the bounded `transport::MessageSource` seam (the same approach
   `abacus_csv_client` takes with its `Transport` seam) and ships an in-memory
   `InMemorySource` for offline tests. **No broker consumer is implemented.**
   *Decision required:* whether to add a Kafka client crate (e.g. `rdkafka`, or
   a pure-Rust protocol client) and, if so, implement `MessageSource` over it.
   Until then, Kafka integration is **not** claimed complete.

2. **Namespace persistence.** This part **is** implemented:
   `namespace_store::PartitionNamespaceStore` is a std-only, restart-durable
   `ABACNS1` file that binds each logical partition key to a UUIDv4
   `event_namespace`, generated once and reused across restarts/retries
   (Canonical Contract §3.2; Wire §5). It is tested for persistence across a
   simulated restart. If the project later standardizes a different registry
   store, only this module changes.

## What is complete and tested today

- `SourceType::Kafka` + `IdentityScheme::KafkaPartitionOffset` configuration and
  fail-closed validation (`config.rs`).
- Persistent per-partition namespace registration with restart reuse
  (`namespace_store.rs`).
- Offset-as-`event_id` mapping, negative-offset rejection, and monotonic resume
  cursor (`partition.rs`).
- Bounded, ordered, partition-filtered transport seam (`transport.rs`).
- Schema-independent canonical record construction (shared feature derivation)
  and `ABACSEG1` encoding, plus offset-identity idempotency proven through the
  real `abacus_tensor` substrate (`map.rs` + tests).
