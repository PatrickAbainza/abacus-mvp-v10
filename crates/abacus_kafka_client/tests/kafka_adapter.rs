//! Tests for the Kafka source adapter's independently implementable surface.
//!
//! These exercise everything that does **not** depend on the unresolved Kafka
//! message schema (Gap A): configuration, persistent per-partition namespaces,
//! offset-as-event-id idempotency through the real substrate, `ABACSEG1`
//! encoding, and the bounded transport seam. The message→record mapper is a
//! documented blocker and is deliberately not tested here (there is no schema
//! to test against).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use abacus_ingest::session::{IngestionSession, SessionLimits};
use abacus_ingest::{EventNamespace, SubstrateApplier, SubstrateValuePolicy};
use abacus_kafka_client::{
    build_record, encode_batch, event_id_for_offset, KafkaConfig, KafkaMessage, MappedEvent,
    MessageSource, PartitionCursor, PartitionNamespaceStore,
};
use abacus_tensor::tensor::value::MINOR_SCALE;

/// A fixed, non-zero namespace standing in for a partition's registered one.
fn ns() -> EventNamespace {
    EventNamespace::from_bytes([
        0x9a, 0x1f, 0x4c, 0x07, 0x6b, 0x2d, 0x4e, 0x81, 0x93, 0x05, 0xc7, 0x22, 0x48, 0xbd, 0x6e,
        0x10,
    ])
}

fn config() -> KafkaConfig {
    KafkaConfig {
        account_id_width: 4,
        currency_code: "USD".to_string(),
        currency_precision: 2,
        threshold_minor_units: 10_000,
        threshold_effective_day: 20_738,
        threshold_config_version: 3,
        parsing_policy_version: 1,
        max_amount_minor_units: 100_000_000_000,
    }
}

fn fresh_dir(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "abacus_kafka_client_{}_{}_{}",
        std::process::id(),
        tag,
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn kafka_config_builds_the_canonical_source_config() {
    let cfg = config().to_source_config().expect("valid config");
    assert_eq!(cfg.source_type.as_str(), "kafka");
    assert_eq!(cfg.identity_scheme.as_str(), "kafka_partition_offset");
    assert_eq!(cfg.source_type.to_u8(), 3);
    assert_eq!(cfg.identity_scheme.to_u8(), 3);
}

#[test]
fn a_malformed_kafka_config_is_rejected_fail_closed() {
    for bad in [
        KafkaConfig {
            currency_code: "usd".to_string(),
            ..config()
        }, // not uppercase
        KafkaConfig {
            currency_code: "US".to_string(),
            ..config()
        }, // wrong length
        KafkaConfig {
            threshold_minor_units: 0,
            ..config()
        }, // non-positive threshold
        KafkaConfig {
            account_id_width: 3,
            ..config()
        }, // unsupported width
        KafkaConfig {
            max_amount_minor_units: 0,
            ..config()
        }, // non-positive ceiling
    ] {
        assert!(
            bad.to_source_config().is_err(),
            "an invalid Kafka config must be rejected before any message is read"
        );
    }
}

#[test]
fn a_partition_namespace_is_persistent_across_restart() {
    let dir = fresh_dir("ns_persist");

    // First run: register two partitions.
    let ns0;
    let ns1;
    {
        let mut store = PartitionNamespaceStore::open(&dir).expect("open");
        ns0 = store.namespace_for("orders-0").expect("register p0");
        ns1 = store.namespace_for("orders-1").expect("register p1");
        assert_ne!(ns0, ns1, "distinct partitions get distinct namespaces");
        assert!(!ns0.is_zero() && !ns1.is_zero());
    }

    // Restart: reopen from disk. The same keys must map to the same namespaces,
    // and asking again must not regenerate (Contract §3.2).
    {
        let mut store = PartitionNamespaceStore::open(&dir).expect("reopen");
        assert_eq!(store.get("orders-0"), Some(ns0));
        assert_eq!(store.get("orders-1"), Some(ns1));
        assert_eq!(store.namespace_for("orders-0").expect("reuse p0"), ns0);
        assert_eq!(store.namespace_for("orders-1").expect("reuse p1"), ns1);
        assert_eq!(store.len(), 2, "no duplicate entry was appended");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_negative_offset_is_not_a_valid_event_id() {
    assert!(event_id_for_offset(-1).is_err(), "latest sentinel rejected");
    assert!(
        event_id_for_offset(-2).is_err(),
        "earliest sentinel rejected"
    );
    assert_eq!(event_id_for_offset(0).unwrap(), 0);
    assert_eq!(event_id_for_offset(42).unwrap(), 42);
}

#[test]
fn partition_cursor_resumes_monotonically() {
    let mut c = PartitionCursor::start(0);
    assert_eq!(c.next_offset, 0);
    c.advance_past(7);
    assert_eq!(c.next_offset, 8);
    // A redelivered / out-of-order offset never rewinds the cursor.
    c.advance_past(3);
    assert_eq!(c.next_offset, 8);
    c.advance_past(20);
    assert_eq!(c.next_offset, 21);
}

#[test]
fn offset_identity_is_idempotent_through_the_substrate() {
    // The event id IS the partition offset, so re-applying the same offset under
    // the same namespace resolves to the same substrate id and the second apply
    // is absorbed by the substrate CAS as already-applied (P7 exactly-once).
    let cfg = config().to_source_config().unwrap();
    let mapped = MappedEvent {
        event_time_ms: 1_760_299_200_000, // 2025-10-12T09:30:00Z
        sender_id: 11,
        receiver_id: 22,
        amount_minor_units: 12_345,
    };
    let record = build_record(&cfg, ns(), 5, &mapped).expect("build record");
    let event_id = event_id_for_offset(5).unwrap();

    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    let first = applier
        .apply(&record, event_id, Some(mapped.amount_minor_units as i64))
        .expect("first apply");
    assert!(matches!(first, abacus_ingest::ApplyOutcome::Applied { .. }));

    // The substrate now holds the event at the exact attested micro value.
    assert_eq!(applier.applied(), 1);
    assert_eq!(applier.engine().frontier().len(), 1);
    assert_eq!(
        applier.engine().frontier().entries()[0].value,
        i64::from(12_345) * MINOR_SCALE
    );

    // StreamingIngestionDemo the same (namespace, offset): idempotent, no second delta.
    let again = applier
        .apply(&record, event_id, Some(mapped.amount_minor_units as i64))
        .expect("retry apply");
    assert!(matches!(again, abacus_ingest::ApplyOutcome::AlreadyApplied));
    assert_eq!(applier.applied(), 1, "no double apply");
    assert_eq!(applier.duplicates(), 1, "the retry was recognized");
    assert_eq!(applier.engine().frontier().len(), 1, "no second delta");
}

#[test]
fn mapped_records_encode_to_a_valid_abacseg1_segment() {
    let cfg = config().to_source_config().unwrap();
    let records = vec![
        build_record(
            &cfg,
            ns(),
            0,
            &MappedEvent {
                event_time_ms: 1_760_299_200_000,
                sender_id: 1,
                receiver_id: 2,
                amount_minor_units: 100,
            },
        )
        .unwrap(),
        build_record(
            &cfg,
            ns(),
            1,
            &MappedEvent {
                event_time_ms: 1_760_299_260_000,
                sender_id: 2,
                receiver_id: 3,
                amount_minor_units: 250,
            },
        )
        .unwrap(),
    ];

    let segment = encode_batch(&cfg, ns(), &records).expect("encode");
    assert_eq!(
        &segment[..8],
        b"ABACSEG1",
        "the Kafka adapter emits ABACSEG1"
    );
    assert!(segment.len() >= 176, "a complete segment header");

    // The encoded segment is accepted by the real segment validator/session.
    let mut session =
        IngestionSession::create("kafka-seg", cfg.clone(), ns(), SessionLimits::default()).unwrap();
    session
        .submit_chunk(1, &segment)
        .expect("the segment must validate");
}

#[test]
fn an_out_of_range_amount_is_rejected_not_clamped() {
    let cfg = config().to_source_config().unwrap();
    // u64::MAX cannot be turned into a representable log-magnitude feature.
    let err = build_record(
        &cfg,
        ns(),
        0,
        &MappedEvent {
            event_time_ms: 1_760_299_200_000,
            sender_id: 1,
            receiver_id: 2,
            amount_minor_units: u64::MAX,
        },
    );
    assert!(err.is_err(), "an unrepresentable amount must be rejected");
}

#[test]
fn the_transport_seam_is_bounded_and_ordered() {
    use abacus_kafka_client::InMemorySource;
    let mut src = InMemorySource::new();
    for off in [3i64, 1, 2, 10] {
        src.push(KafkaMessage {
            partition: 0,
            offset: off,
            key: Vec::new(),
            payload: Vec::new(),
        });
    }
    src.push(KafkaMessage {
        partition: 1,
        offset: 0,
        key: Vec::new(),
        payload: Vec::new(),
    });

    // Bounded read: at most `max`, in offset order, filtered to the partition.
    let got = src.poll(0, 0, 2).unwrap();
    assert_eq!(got.len(), 2, "the read is bounded by max");
    assert_eq!(got[0].offset, 1);
    assert_eq!(got[1].offset, 2);

    // Resume from a cursor position skips already-processed offsets.
    let got = src.poll(0, 3, 10).unwrap();
    assert_eq!(
        got.iter().map(|m| m.offset).collect::<Vec<_>>(),
        vec![3, 10]
    );

    // A different partition is isolated.
    assert_eq!(src.poll(1, 0, 10).unwrap().len(), 1);
}
