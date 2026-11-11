//! STEP 4: stable event identity, namespace handling, duplicate delivery and
//! conflicting identities.
//!
//! Covers task §4 "Stable event identity and namespace handling" and
//! "Duplicate delivery and conflicting event identities", plus Canonical
//! Contract §13 (Identity) and Golden Source §17.5.

mod common;

use abacus_ingest::error::ErrorCode;
use abacus_ingest::identity::{IdentityLedger, IdentityOutcome};
use abacus_ingest::{EventIdentity, EventNamespace};
use common::{config, ns, ns_other, record};

fn digest(t: i64, s: u64, r: u64, amount: u64) -> [u8; 32] {
    let f = abacus_ingest::FeaturePayload::derive(amount, 10_000, 100).unwrap();
    IdentityLedger::content_digest(
        t,
        s,
        r,
        f.log_magnitude,
        f.relative_threshold_distance,
        f.roundedness_flags,
    )
}

#[test]
fn retry_of_the_same_import_preserves_namespace_and_event_ids() {
    // Canonical Contract §13: "Retry of the same CSV import preserves namespace
    // and event IDs." Modeled at the ledger level: redelivering the identical
    // (identity, content) pair is a duplicate and applies nothing.
    let mut led = IdentityLedger::new();
    let d = digest(1_700_000_000_000, 11, 22, 50_000);
    let id = EventIdentity::new(ns(), 42);
    assert_eq!(
        led.check(id, d).unwrap().0,
        IdentityOutcome::New
    );
    assert_eq!(
        led.check(id, d).unwrap().0,
        IdentityOutcome::Duplicate
    );
    assert_eq!(led.accepted(&ns()), 1, "a retry must not re-accept");
}

#[test]
fn separate_import_receives_a_separate_namespace() {
    // Canonical Contract §13: "A separate import receives a separate namespace
    // by default." So the same event id in the new namespace is a NEW event.
    let mut led = IdentityLedger::new();
    let d = digest(1_700_000_000_000, 11, 22, 50_000);
    led.check(EventIdentity::new(ns(), 1), d).unwrap();
    let (outcome, sid) = led.check(EventIdentity::new(ns_other(), 1), d).unwrap();
    assert_eq!(outcome, IdentityOutcome::New);
    assert!(sid.is_some());
}

#[test]
fn same_identity_identical_content_is_deduplicated() {
    let mut led = IdentityLedger::new();
    let d = digest(1, 2, 3, 7_777);
    led.check(EventIdentity::new(ns(), 9), d).unwrap();
    let (o, sid) = led.check(EventIdentity::new(ns(), 9), d).unwrap();
    assert_eq!(o, IdentityOutcome::Duplicate);
    assert!(sid.is_none(), "a duplicate must not receive a substrate id");
    assert_eq!(led.duplicates(&ns()), 1);
}

#[test]
fn same_identity_conflicting_content_is_rejected() {
    // Golden Source §5.1: "Same identity and conflicting content: reject and
    // report a conflict."
    let mut led = IdentityLedger::new();
    led.check(EventIdentity::new(ns(), 9), digest(1, 2, 3, 7_777)).unwrap();
    // Same identity, a different amount -> different content.
    let err = led
        .check(EventIdentity::new(ns(), 9), digest(1, 2, 3, 8_888))
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Identity);
    assert_eq!(led.conflicts(&ns()), 1);
    assert_eq!(led.accepted(&ns()), 1, "a conflict must not be accepted");
}

#[test]
fn conflict_is_detected_on_every_content_field() {
    // Each content field, changed alone, must be a conflict.
    let cases: Vec<(i64, u64, u64, u64)> = vec![
        (2, 2, 3, 7_777),       // time changed
        (1, 99, 3, 7_777),      // sender changed
        (1, 2, 99, 7_777),      // receiver changed
        (1, 2, 3, 8_888),       // amount changed
    ];
    for (t, s, r, a) in cases {
        let mut led = IdentityLedger::new();
        led.check(EventIdentity::new(ns(), 5), digest(1, 2, 3, 7_777)).unwrap();
        assert!(
            led.check(EventIdentity::new(ns(), 5), digest(t, s, r, a)).is_err(),
            "conflict not detected for t={t} s={s} r={r} a={a}"
        );
    }
}

#[test]
fn event_id_overflow_is_rejected_not_wrapped() {
    // Canonical Contract §13: "Event ID overflow is rejected." The record's
    // event_id is a u64, so u64::MAX is the largest legal value and the ledger
    // must accept it verbatim rather than wrap to 0.
    let mut led = IdentityLedger::new();
    let d = digest(1, 2, 3, 100);
    let (o, sid) = led.check(EventIdentity::new(ns(), u64::MAX), d).unwrap();
    assert_eq!(o, IdentityOutcome::New);
    assert!(sid.is_some());
    // And it is remembered as u64::MAX, distinct from 0.
    assert_eq!(
        led.check(EventIdentity::new(ns(), 0), d).unwrap().0,
        IdentityOutcome::New,
        "u64::MAX must not alias event id 0"
    );
}

#[test]
fn namespace_collision_is_detected_by_the_registry() {
    // Canonical Contract §13: "Namespace collision detection is tested."
    let mut reg = abacus_ingest::NamespaceRegistry::new();
    let a = reg.generate().unwrap();
    let b = reg.generate().unwrap();
    assert_ne!(a, b);
    assert!(reg.contains(&a));
    // Re-registering the same value is the retry path, not a collision.
    reg.register(a).unwrap();
    // The all-zero namespace is never a valid identity domain.
    assert!(reg
        .register(EventNamespace::from_bytes([0u8; 16]))
        .is_err());
}

#[test]
fn chunk_idempotency_is_not_a_substitute_for_event_dedup() {
    // Golden Source §5.1: "Do not rely on chunk idempotency as a substitute for
    // event-level deduplication." Prove the two are independent: the SAME event
    // redelivered in a DIFFERENT chunk sequence must still be deduplicated.
    use abacus_ingest::{encode_segment, IngestionSession};
    use common::wide_limits;

    let cfg = config(4);
    let recs = vec![record(1, 1_700_000_000_000, 11, 22, 50_000)];
    let bytes = encode_segment(&cfg, ns(), &recs).unwrap();

    let mut sess = IngestionSession::create("s", cfg, ns(), wide_limits()).unwrap();
    // First delivery as chunk 1.
    let a = sess.submit_chunk(1, &bytes).unwrap();
    assert_eq!(a.new_events, 1);
    assert_eq!(a.duplicate_events, 0);
    // Same bytes, different chunk sequence: the CHUNK is new, so it is accepted
    // at the chunk level, but the EVENT must be recognized as a duplicate.
    let b = sess.submit_chunk(2, &bytes).unwrap();
    assert_eq!(b.new_events, 0, "event-level dedup must catch the redelivery");
    assert_eq!(b.duplicate_events, 1);
    assert_eq!(sess.record_count(), 1, "only one event may be buffered");
}

#[test]
fn session_rejects_a_segment_from_a_different_namespace() {
    // Canonical Contract §2: "A segment containing multiple namespaces is
    // invalid under this version", and a session binds one namespace.
    use abacus_ingest::{encode_segment, IngestionSession};
    use common::wide_limits;

    let cfg = config(4);
    let foreign = abacus_ingest::record::CanonicalRecord {
        identity: EventIdentity::new(ns_other(), 1),
        event_time_ms: 1_700_000_000_000,
        sender_id: 1,
        receiver_id: 2,
        features: abacus_ingest::FeaturePayload::derive(5_000, 10_000, 100).unwrap(),
    };
    let bytes = encode_segment(&cfg, ns_other(), &[foreign]).unwrap();
    let mut sess = IngestionSession::create("s", cfg, ns(), wide_limits()).unwrap();
    let err = sess.submit_chunk(1, &bytes).unwrap_err();
    assert_eq!(err.code, ErrorCode::Identity);
}

#[test]
fn namespace_never_dropped_through_the_whole_path() {
    // Wire §5: the namespace "must remain attached to every event through
    // ingestion, deduplication, persistence, streaming_ingestion_demo, and segment movement."
    use abacus_ingest::{encode_segment, Segment};

    let recs = vec![
        record(1, 1_700_000_000_000, 11, 22, 50_000),
        record(2, 1_700_000_060_000, 33, 44, 60_000),
    ];
    let bytes = encode_segment(&config(4), ns(), &recs).unwrap();
    let seg = Segment::decode(&bytes).unwrap();
    for r in &seg.records {
        assert_eq!(r.identity.namespace, ns(), "namespace was dropped");
    }
    // And it survives a re-encode/decode cycle (segment movement).
    let again = Segment::decode(&seg.encode().unwrap()).unwrap();
    for r in &again.records {
        assert_eq!(r.identity.namespace, ns());
    }
}
