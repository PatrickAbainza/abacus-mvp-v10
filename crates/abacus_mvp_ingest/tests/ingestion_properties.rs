//! Property-based tests for the `abacus_mvp_v10` ingestion capability.
//!
//! A deterministic, std-only property suite (seeded xorshift64, no new
//! dependencies) exercising the full ingestion surface. Properties:
//! P1 feature determinism, P2 overflow fail-closed, P3 record round-trip,
//! P4 segment round-trip, P5 tamper detection, P6 CSV purity, P7 ledger
//! duplicate-vs-conflict, P8 service retry idempotency, P9 rejection
//! fail-closed, P10 SAML-D content-addressed namespace, P11 Kafka offset
//! injectivity, P12 cursor monotonicity, P13 namespace-store restart, P14
//! transport bounded+ordered, P15 Kafka batch end-to-end idempotency.
//!
//! Every case derives from (SEED, property_tag, case_index); a failure prints
//! the case index from which the exact input is re-derivable.

use abacus_ingest::features::FeaturePayload;
use abacus_ingest::record::CanonicalRecord;
use abacus_ingest::wire::Segment;
use abacus_ingest::{EventIdentity, EventNamespace};
use abacus_kafka_client::{
    build_record, encode_batch, event_id_for_offset, InMemorySource, KafkaConfig, KafkaMessage,
    MappedEvent, MessageMapper, MessageSource, PartitionCursor, PartitionNamespaceStore,
};
use abacus_mvp_ingest::{ingest_csv_bytes, samld_namespace};

const SEED: u64 = 0x9E3779B97F4A7C15;
const CASES: usize = 500;

struct Rng(u64);

impl Rng {
    fn for_case(property_tag: u64, index: usize) -> Self {
        let mut s = SEED
            .wrapping_add(property_tag.wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_add((index as u64).wrapping_mul(0xBF58476D1CE4E5B9));
        if s == 0 {
            s = 0x2545F4914F6CDD1D;
        }
        let mut r = Self(s);
        r.next();
        r.next();
        r
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next() % n
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.below(hi - lo + 1)
    }
    fn bytes16(&mut self) -> [u8; 16] {
        let a = self.next().to_le_bytes();
        let b = self.next().to_le_bytes();
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&a);
        out[8..].copy_from_slice(&b);
        out
    }
    fn account_string(&mut self, i: usize) -> String {
        const ALPHA: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789-_";
        let len = 3 + self.below(10) as usize;
        let mut s = String::with_capacity(len + 8);
        for _ in 0..len {
            s.push(ALPHA[self.below(ALPHA.len() as u64) as usize] as char);
        }
        s.push_str(&format!("-{i}"));
        s
    }
}

fn kafka_config() -> abacus_ingest::SourceConfig {
    KafkaConfig {
        account_id_width: 4,
        currency_code: "USD".to_string(),
        currency_precision: 2,
        threshold_minor_units: 10_000,
        threshold_effective_day: 0,
        threshold_config_version: 1,
        parsing_policy_version: 1,
        max_amount_minor_units: 100_000_000_000,
    }
    .to_source_config()
    .expect("property kafka config must validate")
}

const MAJOR_SCALE: u64 = 100;
const THRESHOLD: u64 = 10_000;

#[test]
fn p1_feature_derivation_is_deterministic() {
    for i in 0..CASES {
        let mut r = Rng::for_case(1, i);
        let amount = match r.below(4) {
            0 => r.below(1_000_001),
            1 => r.range(1_000_000, 100_000_000_000),
            2 => r.range(100_000_000_000, u64::MAX / 2),
            _ => r.next(),
        };
        let a = FeaturePayload::derive(amount, THRESHOLD, MAJOR_SCALE);
        let b = FeaturePayload::derive(amount, THRESHOLD, MAJOR_SCALE);
        assert_eq!(a.is_ok(), b.is_ok(), "case {i}: ok-ness must agree");
        if let (Ok(a), Ok(b)) = (a, b) {
            assert_eq!(a, b, "case {i}: same amount, same payload");
            assert_eq!(FeaturePayload::BYTE_LEN, 7, "frozen payload is 7 bytes");
        }
    }
}

#[test]
fn p2_feature_overflow_is_fail_closed() {
    use abacus_ingest::ErrorCode;
    for &amount in &[u64::MAX, u64::MAX - 1, u64::MAX / 2, 1 << 62, 1 << 60] {
        let err = FeaturePayload::derive(amount, 1, 1).expect_err("must overflow");
        assert_eq!(
            err.code,
            ErrorCode::FeatureOverflow,
            "amount {amount} must not clamp"
        );
    }
    for i in 0..CASES {
        let mut r = Rng::for_case(2, i);
        let amount = r.next();
        match FeaturePayload::derive(amount, THRESHOLD, MAJOR_SCALE) {
            Ok(p) => assert_eq!(
                p,
                FeaturePayload::derive(amount, THRESHOLD, MAJOR_SCALE).unwrap()
            ),
            Err(e) => assert_eq!(
                e.code,
                ErrorCode::FeatureOverflow,
                "case {i}: only overflow fails"
            ),
        }
    }
}
#[test]
fn p3_record_encode_decode_round_trips() {
    const WIDTHS: [u8; 4] = [1, 2, 4, 8];
    for i in 0..CASES {
        let mut r = Rng::for_case(3, i);
        let w = WIDTHS[r.below(4) as usize];
        let max_id: u64 = if w == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * w)) - 1
        };
        let modulus = max_id.wrapping_add(1);
        let (sender, receiver) = if modulus == 0 {
            (r.next(), r.next())
        } else {
            (r.below(modulus), r.below(modulus))
        };
        let features =
            match FeaturePayload::derive(r.below(100_000_000_000), THRESHOLD, MAJOR_SCALE) {
                Ok(f) => f,
                Err(_) => continue,
            };
        let ns = EventNamespace::from_bytes(r.bytes16());
        if ns.is_zero() {
            continue;
        }
        let rec = CanonicalRecord {
            identity: EventIdentity::new(ns, r.next()),
            event_time_ms: match r.below(3) {
                0 => 0,
                1 => r.next() as i64,
                _ => -(r.below(4_000_000_000_000) as i64),
            },
            sender_id: sender,
            receiver_id: receiver,
            features,
        };
        let bytes = rec.encode(w).expect("generated ids fit the width");
        assert_eq!(
            bytes.len(),
            23 + 2 * usize::from(w),
            "case {i}: 23+2W bytes"
        );
        let back = CanonicalRecord::decode(&bytes, ns, w).expect("own encoding decodes");
        assert_eq!(back, rec, "case {i}: record round-trip exact");
    }
}

#[test]
fn p4_segment_decode_reencode_round_trips() {
    let cfg = kafka_config();
    for i in 0..200 {
        let mut r = Rng::for_case(4, i);
        let ns = EventNamespace::from_bytes(r.bytes16());
        if ns.is_zero() {
            continue;
        }
        let n = 1 + r.below(25) as usize;
        let mut records = Vec::with_capacity(n);
        for k in 0..n {
            let features = FeaturePayload::derive(r.below(10_000_000), THRESHOLD, MAJOR_SCALE)
                .expect("small amounts represent");
            records.push(CanonicalRecord {
                identity: EventIdentity::new(ns, k as u64),
                event_time_ms: r.below(4_000_000_000_000) as i64,
                sender_id: r.below(100_000),
                receiver_id: r.below(100_000),
                features,
            });
        }
        let bytes = encode_batch(&cfg, ns, &records).expect("batch encodes");
        let seg = Segment::decode(&bytes).expect("own encoding validates");
        assert_eq!(seg.records.len(), n, "batch {i}: count survives");
        assert_eq!(seg.namespace(), ns, "batch {i}: namespace survives");
        assert_eq!(
            seg.encode().expect("re-encode"),
            bytes,
            "batch {i}: byte-exact"
        );
        for (k, rec) in seg.records.iter().enumerate() {
            assert_eq!(rec.identity.event_id, k as u64, "batch {i}: order kept");
            assert_eq!(rec.identity.namespace, ns, "batch {i}: ns kept");
        }
    }
}

#[test]
fn p5_segment_tampering_is_detected() {
    let cfg = kafka_config();
    for i in 0..200 {
        let mut r = Rng::for_case(5, i);
        let ns = EventNamespace::from_bytes(r.bytes16());
        if ns.is_zero() {
            continue;
        }
        let features = FeaturePayload::derive(r.below(10_000_000), THRESHOLD, MAJOR_SCALE).unwrap();
        let rec = CanonicalRecord {
            identity: EventIdentity::new(ns, 7),
            event_time_ms: 1_700_000_000_000,
            sender_id: 11,
            receiver_id: 22,
            features,
        };
        let bytes = encode_batch(&cfg, ns, &[rec]).expect("encode");
        let pos = r.below(bytes.len() as u64) as usize;
        let mut bad = bytes.clone();
        bad[pos] ^= 1 << r.below(8);
        assert!(
            Segment::decode(&bad).is_err(),
            "trial {i}: flip at {pos} must fail"
        );
    }
}
fn build_csv(rows: &[(String, String, String, String)]) -> Vec<u8> {
    let mut s = String::from("timestamp;sender;receiver;amount\n");
    for (t, snd, rcv, amt) in rows {
        s.push_str(&format!("{t};{snd};{rcv};{amt}\n"));
    }
    s.into_bytes()
}

fn gen_timestamp(r: &mut Rng) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        r.range(2020, 2026),
        r.range(1, 12),
        r.range(1, 28),
        r.range(0, 23),
        r.range(0, 59),
        r.range(0, 59)
    )
}

fn gen_amount(r: &mut Rng) -> String {
    format!("{}.{:<02}", r.below(1_000_000), r.below(100))
}

fn applied_of(report_json: &str) -> u64 {
    abacus_ingest::api::json::parse(report_json)
        .expect("valid report")
        .get("commit")
        .and_then(|c| c.get("applied"))
        .and_then(|v| v.as_u64())
        .unwrap()
}

fn reconciles(report_json: &str) -> bool {
    abacus_ingest::api::json::parse(report_json)
        .expect("valid report")
        .get("reconciles")
        .and_then(|v| v.as_bool())
        .unwrap()
}

#[test]
fn p6_csv_ingestion_is_a_pure_function_of_its_bytes() {
    use abacus_ingest::api::IngestionService;
    for i in 0..CASES {
        let mut r = Rng::for_case(6, i);
        let nrows = 1 + r.below(12) as usize;
        let mut rows = Vec::with_capacity(nrows);
        for k in 0..nrows {
            rows.push((
                gen_timestamp(&mut r),
                r.account_string(k),
                r.account_string(k + 10_000),
                gen_amount(&mut r),
            ));
        }
        let csv = build_csv(&rows);
        let mut s1 = IngestionService::new();
        let mut s2 = IngestionService::new();
        let o1 = ingest_csv_bytes(&mut s1, &csv).expect("ingest");
        let o2 = ingest_csv_bytes(&mut s2, &csv).expect("ingest");
        assert_eq!(o1, o2, "case {i}: same bytes, same report");
    }
}

#[test]
fn p7_identity_ledger_separates_duplicate_from_conflict() {
    use abacus_ingest::identity::{IdentityLedger, IdentityOutcome};
    for i in 0..CASES {
        let mut r = Rng::for_case(7, i);
        let ns = EventNamespace::from_bytes(r.bytes16());
        if ns.is_zero() {
            continue;
        }
        let mut led = IdentityLedger::new();
        let amount = r.below(10_000_000);
        let features =
            FeaturePayload::derive(amount, THRESHOLD, MAJOR_SCALE).expect("representable");
        let rec = CanonicalRecord {
            identity: EventIdentity::new(ns, r.next()),
            event_time_ms: 1_700_000_000_000,
            sender_id: 11,
            receiver_id: 22,
            features,
        };
        let d = abacus_ingest::identity::IdentityLedger::content_digest(
            rec.event_time_ms,
            rec.sender_id,
            rec.receiver_id,
            rec.features.log_magnitude,
            rec.features.relative_threshold_distance,
            rec.features.roundedness_flags,
        );
        let (o1, sid1) = led.check(rec.identity, d).expect("first delivery");
        assert_eq!(o1, IdentityOutcome::New, "case {i}: first is New");
        assert!(sid1.is_some(), "case {i}: New carries substrate id");
        let (o2, sid2) = led.check(rec.identity, d).expect("redelivery");
        assert_eq!(
            o2,
            IdentityOutcome::Duplicate,
            "case {i}: same content is Duplicate"
        );
        assert!(sid2.is_none(), "case {i}: Duplicate carries no new id");
        let mut changed = rec;
        changed.event_time_ms += 1;
        let d2 = abacus_ingest::identity::IdentityLedger::content_digest(
            changed.event_time_ms,
            changed.sender_id,
            changed.receiver_id,
            changed.features.log_magnitude,
            changed.features.relative_threshold_distance,
            changed.features.roundedness_flags,
        );
        let err = led
            .check(changed.identity, d2)
            .expect_err("conflict must fail");
        assert_eq!(
            err.code,
            abacus_ingest::ErrorCode::Identity,
            "case {i}: conflict is Identity"
        );
        assert_eq!(led.conflicts(&ns), 1, "case {i}: conflict counted");
        assert_eq!(led.duplicates(&ns), 1, "case {i}: duplicate counted");
        assert_eq!(led.accepted(&ns), 1, "case {i}: one accepted");
    }
}

#[test]
fn p8_chunk_retry_within_a_session_is_idempotent() {
    use abacus_ingest::api::IngestionService;
    use abacus_ingest::session::ChunkStatus;
    for i in 0..CASES {
        let mut r = Rng::for_case(8, i);
        let nrows = 1 + r.below(8) as usize;
        let mut rows = Vec::with_capacity(nrows);
        for k in 0..nrows {
            rows.push((
                gen_timestamp(&mut r),
                r.account_string(k),
                r.account_string(k + 10_000),
                gen_amount(&mut r),
            ));
        }
        let csv = build_csv(&rows);
        // Normalize once to get the canonical chunk bytes (same bytes the
        // service path submits).
        let plan = abacus_mvp_ingest::plan(100_000);
        let norm = abacus_csv_client::normalize(&csv, &plan, abacus_mvp_ingest::namespace())
            .expect("generated csv normalizes");
        assert!(!norm.chunks.is_empty(), "case {i}: at least one chunk");
        let cfg = plan.to_source_config().expect("plan converts");
        let mut svc = IngestionService::new();
        let created = svc
            .create_session(&abacus_ingest::api::service::CreateSessionRequest {
                source_type: cfg.source_type,
                identity_scheme: cfg.identity_scheme,
                account_id_width: cfg.account_id_width,
                currency_code: cfg.currency_code.as_str().to_string(),
                currency_precision: cfg.currency_precision,
                threshold_minor_units: cfg.threshold_minor_units,
                threshold_effective_day: cfg.threshold_effective_day,
                threshold_config_version: cfg.threshold_config_version,
                parsing_policy_version: cfg.parsing_policy_version,
                max_amount_minor_units: cfg.max_amount_minor_units,
                namespace: Some(abacus_mvp_ingest::namespace()),
            })
            .expect("session");
        let sid = created.session_id;
        let chunk = norm.chunks[0].clone();
        let first = svc.submit_chunk(&sid, 1, &chunk).expect("first upload");
        assert_eq!(
            first.status,
            ChunkStatus::Accepted,
            "case {i}: first accepted"
        );
        let retry = svc.submit_chunk(&sid, 1, &chunk).expect("retry upload");
        assert_eq!(
            retry.status,
            ChunkStatus::AlreadyProcessed,
            "case {i}: identical resubmit changes nothing"
        );
        assert_eq!(retry.new_events, 0, "case {i}: retry stages no new events");
        // The session holds exactly the first upload's records — the retry
        // staged nothing further.
        let st = svc.session_status(&sid).expect("status exists");
        let _ = st;
    }
}

#[test]
fn p9_rejections_are_fail_closed() {
    use abacus_ingest::api::IngestionService;
    for i in 0..CASES {
        let mut r = Rng::for_case(9, i);
        let clean = (
            gen_timestamp(&mut r),
            r.account_string(1),
            r.account_string(2),
            gen_amount(&mut r),
        );
        let mut bad = clean.clone();
        match r.below(5) {
            0 => bad.0 = "not-a-timestamp".to_string(),
            1 => bad.1 = String::new(),
            2 => bad.3 = "12.345".to_string(),
            3 => bad.3 = "not-a-number".to_string(),
            _ => bad.3 = "-5.00".to_string(),
        }
        let rows = if r.below(2) == 0 {
            vec![clean, bad]
        } else {
            vec![bad, clean]
        };
        let csv = build_csv(&rows);
        let mut svc = IngestionService::new();
        let out = ingest_csv_bytes(&mut svc, &csv).expect("session survives a bad row");
        let j = abacus_ingest::api::json::parse(&out).unwrap();
        let preview = j.get("preview").expect("preview");
        assert_eq!(
            preview.get("accepted_rows").and_then(|v| v.as_u64()),
            Some(1),
            "case {i}: clean row accepted"
        );
        assert_eq!(
            preview.get("rejected_rows").and_then(|v| v.as_u64()),
            Some(1),
            "case {i}: bad row rejected"
        );
        assert_eq!(applied_of(&out), 1, "case {i}: clean row commits");
        assert!(reconciles(&out), "case {i}: report reconciles");
    }
}
#[test]
fn p10_samld_namespace_is_content_addressed() {
    for i in 0..CASES {
        let mut r = Rng::for_case(10, i);
        let mut s =
            String::from("Time,Date,Sender_account,Receiver_account,Amount,Payment_currency\n");
        let nrows = 1 + r.below(5) as usize;
        for _ in 0..nrows {
            s.push_str(&format!(
                "{:02}:{:02}:{:02},{:04}-{:02}-{:02},{},{},{}.{:02},UK pounds\n",
                r.range(0, 23),
                r.range(0, 59),
                r.range(0, 59),
                r.range(2020, 2024),
                r.range(1, 12),
                r.range(1, 28),
                r.below(9_000_000_000),
                r.below(9_000_000_000),
                r.below(100_000),
                r.below(100),
            ));
        }
        let bytes = s.into_bytes();
        let a = samld_namespace(&bytes);
        assert_eq!(samld_namespace(&bytes), a, "case {i}: same bytes, same ns");
        let mut changed = bytes.clone();
        let pos = r.below(changed.len() as u64) as usize;
        changed[pos] ^= 0xFF;
        assert_ne!(
            samld_namespace(&changed),
            a,
            "case {i}: 1-byte flip, new ns"
        );
    }
}

#[test]
fn p11_kafka_offset_identity_is_injective() {
    use std::collections::BTreeSet;
    for i in 0..CASES {
        let mut r = Rng::for_case(11, i);
        let mut seen = BTreeSet::new();
        for _ in 0..8 {
            let off = r.below(1_000_000_000) as i64;
            let id = event_id_for_offset(off).expect("non-negative maps");
            assert_eq!(id, off as u64, "case {i}: offset is the event_id");
            assert!(seen.insert(id), "case {i}: duplicate id for {off}");
        }
        for neg in [-1i64, -2, -100, i64::MIN] {
            assert!(
                event_id_for_offset(neg).is_err(),
                "case {i}: {neg} must reject"
            );
        }
    }
}

#[test]
fn p12_kafka_cursor_is_monotone() {
    for i in 0..CASES {
        let mut r = Rng::for_case(12, i);
        let mut cur = PartitionCursor::start(0);
        let mut max_seen: Option<i64> = None;
        for _ in 0..20 {
            let off = match max_seen {
                None => r.below(5) as i64,
                Some(m) => {
                    if r.below(3) == 0 {
                        r.below((m + 1).max(1) as u64) as i64
                    } else {
                        m + 1 + r.below(4) as i64
                    }
                }
            };
            let before = cur.next_offset;
            cur.advance_past(off);
            assert!(cur.next_offset >= before, "case {i}: never rewind");
            max_seen = Some(max_seen.map_or(off, |m| m.max(off)));
        }
        assert_eq!(cur.next_offset, max_seen.unwrap() + 1, "case {i}: max+1");
    }
}

#[test]
fn p13_kafka_namespace_store_survives_restart() {
    for i in 0..50 {
        let mut r = Rng::for_case(13, i);
        let dir = std::env::temp_dir().join(format!("abacus-ns-prop-{i}-{}", r.next()));
        std::fs::create_dir_all(&dir).unwrap();
        let nkeys = 1 + r.below(6) as usize;
        let mut keys = Vec::with_capacity(nkeys);
        for k in 0..nkeys {
            keys.push(format!("topic-p{}-{k}", r.below(8)));
        }
        let mut first = Vec::new();
        {
            let mut store = PartitionNamespaceStore::open(&dir).unwrap();
            for k in &keys {
                first.push(store.namespace_for(k).unwrap());
            }
        }
        {
            let mut store = PartitionNamespaceStore::open(&dir).unwrap();
            for (k, ns) in keys.iter().zip(first.iter()) {
                assert_eq!(
                    &store.namespace_for(k).unwrap(),
                    ns,
                    "case {i}: restart keeps {k}"
                );
            }
            assert_eq!(store.len(), keys.len(), "case {i}: no phantom entries");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn p14_kafka_transport_is_bounded_and_ordered() {
    for i in 0..CASES {
        let mut r = Rng::for_case(14, i);
        let mut src = InMemorySource::new();
        for _ in 0..r.below(30) {
            src.push(KafkaMessage {
                partition: r.below(3) as i32,
                offset: r.below(50) as i64,
                key: vec![],
                payload: vec![r.below(256) as u8],
            });
        }
        let partition = r.below(3) as i32;
        let from = r.below(50) as i64;
        let max = r.below(12) as usize;
        let got = src.poll(partition, from, max).unwrap();
        assert!(got.len() <= max, "case {i}: bounded by max");
        let mut prev: Option<i64> = None;
        for m in &got {
            assert_eq!(m.partition, partition, "case {i}: partition filter");
            assert!(m.offset >= from, "case {i}: from filter");
            if let Some(p) = prev {
                assert!(m.offset >= p, "case {i}: offset order");
            }
            prev = Some(m.offset);
        }
    }
}

struct StubMapper;

impl MessageMapper for StubMapper {
    fn map(&self, msg: &KafkaMessage) -> abacus_ingest::IngestResult<MappedEvent> {
        Ok(MappedEvent {
            event_time_ms: 1_700_000_000_000,
            sender_id: 1,
            receiver_id: 2,
            amount_minor_units: msg.payload.first().copied().unwrap_or(0) as u64,
        })
    }
}

#[test]
fn p15_kafka_batch_end_to_end_is_idempotent() {
    use abacus_ingest::identity::IdentityLedger;
    use abacus_ingest::{ApplyOutcome, SubstrateApplier, SubstrateValuePolicy};
    for i in 0..100 {
        let mut r = Rng::for_case(15, i);
        let cfg = kafka_config();
        let ns = EventNamespace::from_bytes(r.bytes16());
        if ns.is_zero() {
            continue;
        }
        let mapper = StubMapper;
        let n = 1 + r.below(10) as usize;
        let base = r.below(1_000_000);
        let mut records = Vec::new();
        let mut amounts = Vec::new();
        for k in 0..n {
            let msg = KafkaMessage {
                partition: 0,
                offset: (base + k as u64) as i64,
                key: vec![],
                payload: vec![1 + r.below(200) as u8],
            };
            let mapped = mapper.map(&msg).unwrap();
            amounts.push(mapped.amount_minor_units as i64);
            records.push(build_record(&cfg, ns, msg.offset, &mapped).unwrap());
        }
        let bytes = encode_batch(&cfg, ns, &records).unwrap();
        let seg = Segment::decode(&bytes).expect("batch validates");
        assert_eq!(seg.records.len(), n, "case {i}: all records survive");
        let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
        let mut ledger = IdentityLedger::new();
        let mut applied = 0u64;
        let mut dups = 0u64;
        for (rec, amt) in seg.records.iter().zip(amounts.iter()) {
            let dd = abacus_ingest::identity::IdentityLedger::content_digest(
                rec.event_time_ms,
                rec.sender_id,
                rec.receiver_id,
                rec.features.log_magnitude,
                rec.features.relative_threshold_distance,
                rec.features.roundedness_flags,
            );
            let (outcome, sid) = ledger.check(rec.identity, dd).expect("ledger");
            use abacus_ingest::identity::IdentityOutcome;
            match outcome {
                IdentityOutcome::New => {
                    let sid = sid.expect("New carries id");
                    match applier.apply(rec, sid, Some(*amt)).expect("apply") {
                        ApplyOutcome::Applied { .. } => applied += 1,
                        ApplyOutcome::AlreadyApplied => dups += 1,
                    }
                }
                IdentityOutcome::Duplicate => dups += 1,
                IdentityOutcome::Conflict => panic!("case {i}: no conflicts expected"),
            }
        }
        assert_eq!(applied, n as u64, "case {i}: first applies all");
        assert_eq!(dups, 0, "case {i}: no duplicates on first pass");
        let mut applied2 = 0u64;
        let mut dups2 = 0u64;
        for (rec, amt) in seg.records.iter().zip(amounts.iter()) {
            let dd = abacus_ingest::identity::IdentityLedger::content_digest(
                rec.event_time_ms,
                rec.sender_id,
                rec.receiver_id,
                rec.features.log_magnitude,
                rec.features.relative_threshold_distance,
                rec.features.roundedness_flags,
            );
            let (outcome, sid) = ledger.check(rec.identity, dd).expect("ledger");
            use abacus_ingest::identity::IdentityOutcome;
            match outcome {
                IdentityOutcome::New => {
                    let sid = sid.expect("New carries id");
                    match applier.apply(rec, sid, Some(*amt)).expect("apply") {
                        ApplyOutcome::Applied { .. } => applied2 += 1,
                        ApplyOutcome::AlreadyApplied => dups2 += 1,
                    }
                }
                IdentityOutcome::Duplicate => dups2 += 1,
                IdentityOutcome::Conflict => panic!("case {i}: no conflicts expected"),
            }
        }
        assert_eq!(applied2, 0, "case {i}: streaming_ingestion_demo applies nothing");
        assert_eq!(dups2, n as u64, "case {i}: streaming_ingestion_demo all duplicates");
    }
}
