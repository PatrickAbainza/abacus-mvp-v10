//! STEP 4: chunk retries, ordering, idempotency, session state transitions and
//! error reporting.
//!
//! Covers task §4 "Chunk retries, ordering, and idempotency", "Session state
//! transitions and error reporting", "Validation without application",
//! "Application, streaming_ingestion_demo, and recovery behavior", and Golden Source §13/§17.5.

mod common;

use abacus_ingest::error::ErrorCode;
use abacus_ingest::session::{ChunkStatus, SessionState};
use abacus_ingest::{encode_segment, IngestionSession, SubstrateApplier, SubstrateValuePolicy};
use common::{attest_fixed, config, ns, record, tight_limits, wide_limits};

fn chunk_bytes(ids: &[u64]) -> Vec<u8> {
    let recs: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| record(id, 1_700_000_000_000 + i as i64 * 1_000, i as u64 + 1, i as u64 + 2, 50_000))
        .collect();
    encode_segment(&config(4), ns(), &recs).unwrap()
}

#[test]
fn session_starts_open_and_transitions_to_uploading() {
    let mut s = IngestionSession::create("s1", config(4), ns(), wide_limits()).unwrap();
    assert_eq!(s.state(), SessionState::Open);
    s.submit_chunk(1, &chunk_bytes(&[1])).unwrap();
    assert_eq!(s.state(), SessionState::Uploading);
}

#[test]
fn identical_chunk_retry_is_idempotent() {
    // Golden Source §13: "Repeating a chunk with the same identity and
    // identical contents is idempotent."
    let bytes = chunk_bytes(&[1, 2]);
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    let first = s.submit_chunk(1, &bytes).unwrap();
    assert_eq!(first.status, ChunkStatus::Accepted);
    assert_eq!(first.record_count, 2);
    let retry = s.submit_chunk(1, &bytes).unwrap();
    assert_eq!(retry.status, ChunkStatus::AlreadyProcessed);
    // Nothing was double-counted.
    assert_eq!(s.record_count(), 2, "a retry must not buffer records twice");
    assert_eq!(s.chunk_count(), 1);
}

#[test]
fn same_chunk_sequence_with_different_contents_is_rejected() {
    // §13: "Repeating a chunk identity with different contents is rejected."
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1])).unwrap();
    let err = s.submit_chunk(1, &chunk_bytes(&[2])).unwrap_err();
    assert_eq!(err.code, ErrorCode::Session);
}

#[test]
fn chunks_may_arrive_out_of_order_sequence() {
    // §13 requires an explicit ordering policy. The declared policy here is
    // that sequence numbers are *identifiers*, not a required arrival order:
    // each chunk is self-contained and validated independently, so ordering
    // cannot change event-time semantics (Golden Source §15: "Chunk boundaries
    // must not change event-time semantics").
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(7, &chunk_bytes(&[70])).unwrap();
    s.submit_chunk(3, &chunk_bytes(&[30])).unwrap();
    s.submit_chunk(5, &chunk_bytes(&[50])).unwrap();
    assert_eq!(s.chunk_count(), 3);
    assert_eq!(s.record_count(), 3);
    // Every sequence is independently queryable.
    for seq in [3u64, 5, 7] {
        assert_eq!(s.chunk_status(seq).unwrap(), ChunkStatus::Accepted);
    }
}

#[test]
fn unknown_chunk_status_is_an_error() {
    let s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    assert_eq!(s.chunk_status(99).unwrap_err().code, ErrorCode::Session);
}

#[test]
fn seal_blocks_further_chunks() {
    // §13: "Completed sessions cannot silently accept unrelated additional
    // records."
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1])).unwrap();
    s.seal().unwrap();
    assert_eq!(s.state(), SessionState::Sealed);
    let err = s.submit_chunk(2, &chunk_bytes(&[2])).unwrap_err();
    assert_eq!(err.code, ErrorCode::Session);
    assert_eq!(s.record_count(), 1);
    // Sealing twice is rejected: transitions are explicit.
    assert!(s.seal().is_err());
}

#[test]
fn cancel_drops_buffered_records_and_blocks_work() {
    // Task step 3: "Cancel or expire incomplete sessions safely."
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1, 2])).unwrap();
    s.cancel();
    assert_eq!(s.state(), SessionState::Cancelled);
    assert_eq!(s.record_count(), 0, "a cancelled session holds no records");
    assert_eq!(s.total_bytes(), 0);
    assert!(s.submit_chunk(2, &chunk_bytes(&[3])).is_err());
    // An empty session is trivially valid, and stays so after cancel.
    assert!(s.validate_only().is_ok(), "an empty session is trivially valid");
}

#[test]
fn limits_are_enforced() {
    // §13: "The server enforces maximum chunk size, record count, total upload
    // size, and resource limits."
    let mut s = IngestionSession::create("s", config(4), ns(), tight_limits()).unwrap();
    // Chunk of 4 records exceeds the 2-record per-chunk limit.
    let err = s.submit_chunk(1, &chunk_bytes(&[1, 2, 3, 4])).unwrap_err();
    assert_eq!(err.code, ErrorCode::Session);
    // The chunk limit of 2 is also a session-record limit of 4.
    s.submit_chunk(1, &chunk_bytes(&[1, 2])).unwrap();
    s.submit_chunk(2, &chunk_bytes(&[3, 4])).unwrap();
    // A third chunk exceeds max_chunks = 2.
    let err = s.submit_chunk(3, &chunk_bytes(&[5, 6])).unwrap_err();
    assert_eq!(err.code, ErrorCode::Session);
}

#[test]
fn validate_without_apply_changes_nothing() {
    // Task step 3: "Validate a complete segment or session without applying
    // events." Validation must be side-effect free.
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1, 2, 3])).unwrap();
    let before = (
        s.record_count(),
        s.chunk_count(),
        s.state(),
        s.total_bytes(),
        s.duplicate_events(),
    );
    for _ in 0..3 {
        s.validate_only().unwrap();
    }
    let after = (
        s.record_count(),
        s.chunk_count(),
        s.state(),
        s.total_bytes(),
        s.duplicate_events(),
    );
    assert_eq!(before, after, "validate_only must be side-effect free");
    // Crucially, no substrate was touched.
    let applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    assert_eq!(applier.frontier_len(), 0);
    assert_eq!(applier.applied(), 0);
}

#[test]
fn commit_applies_every_validated_event_exactly_once() {
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1, 2, 3])).unwrap();
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    let res = s.commit(&mut applier, attest_fixed(50_000)).unwrap();
    assert_eq!(res.applied, 3);
    assert_eq!(res.already_applied, 0);
    assert_eq!(s.state(), SessionState::Committed);
    // The substrate's own incremental path recorded three frontier deltas.
    assert_eq!(applier.frontier_len(), 3);
    assert_eq!(applier.applied(), 3);
    // Three records with (sender, receiver) = (1,2), (2,3), (3,4): the shared
    // namespace (§6) means senders and receivers draw from ONE code space, so
    // the distinct identifiers are {1,2,3,4} = 4 accounts.
    assert_eq!(applier.accounts().account_count(), 4);
}

#[test]
fn committed_session_cannot_commit_again() {
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1])).unwrap();
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    s.commit(&mut applier, attest_fixed(50_000)).unwrap();
    let err = s.commit(&mut applier, attest_fixed(50_000)).unwrap_err();
    assert_eq!(err.code, ErrorCode::Session);
    assert_eq!(applier.applied(), 1, "no double application");
}

#[test]
fn browser_csv_policy_reports_the_specification_gap() {
    // The material gap: client-derived features carry no amount, so the
    // substrate's exact i64 micro-unit value cannot be populated. This must be
    // surfaced, never silently substituted.
    use abacus_ingest::config::SourceType;
    let mut cfg = config(4);
    cfg.source_type = SourceType::BrowserCsv;
    // The chunk must be encoded with the SAME configuration so its
    // config_fingerprint matches the session binding.
    let recs = vec![record(1, 1_700_000_000_000, 1, 2, 50_000)];
    let bytes = encode_segment(&cfg, ns(), &recs).unwrap();
    let mut s = IngestionSession::create("s", cfg, ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &bytes).unwrap();
    s.validate_only().unwrap();
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::ClientDerivedOnly);
    let err = s.commit(&mut applier, |_i, _r| None).unwrap_err();
    assert_eq!(err.code, ErrorCode::SpecificationGap);
    assert_eq!(applier.frontier_len(), 0, "nothing may be applied");
    assert_ne!(s.state(), SessionState::Committed);
}

#[test]
fn default_policy_follows_the_source_type() {
    use abacus_ingest::config::SourceType;
    // Amendment A1: the exact amount travels in the commit-time attestation
    // channel for EVERY source type (browser CSV included), so every source
    // defaults to AttestedMinorUnits. ClientDerivedOnly remains only as an
    // explicit opt-out that still refuses.
    assert_eq!(
        SubstrateValuePolicy::default_for(SourceType::BrowserCsv),
        SubstrateValuePolicy::AttestedMinorUnits
    );
    assert_eq!(
        SubstrateValuePolicy::default_for(SourceType::SamldFixture),
        SubstrateValuePolicy::AttestedMinorUnits
    );
    assert_eq!(
        SubstrateValuePolicy::default_for(SourceType::Kafka),
        SubstrateValuePolicy::AttestedMinorUnits
    );
}

#[test]
fn streaming_ingestion_demo_of_the_same_stream_reproduces_the_same_substrate_state() {
    // Golden Source §15: "StreamingIngestionDemo of the same accepted event stream under the
    // same configuration and policy must reproduce the same logical CSF/index
    // state and analytical outputs."
    let ids: Vec<u64> = (1..=25).collect();
    let bytes = chunk_bytes(&ids);

    // Run 1.
    let mut s1 = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s1.submit_chunk(1, &bytes).unwrap();
    let mut a1 = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    let r1 = s1.commit(&mut a1, attest_fixed(50_000)).unwrap();

    // Run 2: a fresh session and a fresh substrate, same input.
    let mut s2 = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s2.submit_chunk(1, &bytes).unwrap();
    let mut a2 = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    let r2 = s2.commit(&mut a2, attest_fixed(50_000)).unwrap();

    assert_eq!(r1, r2);
    assert_eq!(a1.frontier_len(), a2.frontier_len());
    // The account mapping — and therefore every coordinate — is identical.
    assert_eq!(a1.export_accounts(), a2.export_accounts());
    // And the decoded substrate tensors are entry-for-entry equal.
    let t1 = a1.engine().decode_live().unwrap();
    let t2 = a2.engine().decode_live().unwrap();
    assert_eq!(t1.entries(), t2.entries(), "streaming_ingestion_demo changed the logical state");
    assert_eq!(t1.nnz(), t2.nnz());
}

#[test]
fn recovery_restores_the_account_mapping_so_identity_is_preserved() {
    // §13: "Session resumption must preserve account mappings and source-row
    // identity."
    let mut a1 = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    let mut s1 = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s1.submit_chunk(1, &chunk_bytes(&[1, 2])).unwrap();
    s1.commit(&mut a1, attest_fixed(50_000)).unwrap();
    let exported = a1.export_accounts();
    let code_before = a1.accounts().code_of(1).unwrap();

    // "Crash": rebuild the applier from the exported mapping.
    let restored = abacus_ingest::AccountMap::restore(&exported).unwrap();
    let mut a2 = SubstrateApplier::with_engine(
        abacus_tensor::ingest::IngestionEngine::new(),
        restored,
        SubstrateValuePolicy::AttestedMinorUnits,
    );
    assert_eq!(
        a2.accounts().code_of(1),
        Some(code_before),
        "recovery changed the logical identity of an account"
    );
    // New accounts continue append-only, never reusing a code.
    let next = a2.accounts_mut().intern(999).unwrap();
    assert!(next > code_before, "a recovered map must not reuse a code");
}

#[test]
fn report_reconciles_every_input_record() {
    // Canonical Contract §10: "Counts must reconcile. Every input record must
    // be accounted for under the declared parsing policy."
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1, 2, 3])).unwrap();
    // Redeliver the same chunk under a new sequence: 3 duplicates.
    s.submit_chunk(2, &chunk_bytes(&[1, 2, 3])).unwrap();
    let report = abacus_ingest::IngestionReport::from_session(&s);
    // 3 records delivered twice = 6 inputs; the redelivery is a duplicate,
    // not a second copy of the data.
    assert_eq!(report.input_records, 6);
    assert_eq!(report.accepted_records, 3);
    assert_eq!(report.duplicate_records, 3);
    assert_eq!(report.conflicting_records, 0);
    assert_eq!(report.rejected_records, 0);
    assert!(report.reconciles(), "counts must partition the input set");
    // Versions and identity are all present (Contract §10 field list).
    assert_eq!(report.account_id_width, 4);
    assert!(!report.config_fingerprint.is_empty());
    assert!(!report.event_namespace.is_empty());
    assert!(report.min_event_time_ms.is_some());
    assert!(report.max_event_time_ms.is_some());
}

#[test]
fn conflicting_identity_blocks_commit_and_is_reported() {
    // §11: "A failed segment must not leave the substrate in an undocumented
    // partially applied state."
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1])).unwrap();
    // Same event id 1, but a different amount -> conflicting content.
    let conflicting = abacus_ingest::record::CanonicalRecord {
        identity: abacus_ingest::EventIdentity::new(ns(), 1),
        event_time_ms: 1_700_000_000_000,
        sender_id: 1,
        receiver_id: 2,
        features: abacus_ingest::FeaturePayload::derive(999_999, 10_000, 100).unwrap(),
    };
    let bytes = abacus_ingest::encode_segment(&config(4), ns(), &[conflicting]).unwrap();
    s.submit_chunk(2, &bytes).unwrap(); // accepted at chunk level, conflict recorded
    assert_eq!(s.conflicting_events(), 1);
    assert_eq!(s.record_count(), 1, "the conflicting record is not buffered");

    // The commit still succeeds for the one valid event, and the report
    // accounts for the conflict explicitly.
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    let res = s.commit(&mut applier, attest_fixed(50_000)).unwrap();
    assert_eq!(res.applied, 1);
    let report = abacus_ingest::IngestionReport::from_session(&s);
    assert_eq!(report.conflicting_records, 1);
    assert!(!report.errors.is_empty(), "the conflict must be reported");
    assert!(report.reconciles());
}

#[test]
fn missing_amount_mid_stream_applies_nothing_at_all() {
    // Canonical Contract §11: "a failed segment must not leave the substrate in
    // an undocumented partially applied state." An amount that is available
    // for record 0 but missing for record 1 must NOT produce a prefix
    // application: the whole commit is pre-flighted before the first mutation.
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1, 2, 3])).unwrap();
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    let err = s
        .commit(&mut applier, |i, _r| {
            if i == 0 {
                Some(50_000)
            } else {
                None // the trusted adapter lost the amount for record 1
            }
        })
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::SpecificationGap);
    assert_eq!(
        applier.frontier_len(),
        0,
        "a failed commit must apply NOTHING, not even the valid prefix"
    );
    assert_eq!(applier.applied(), 0);
    assert_ne!(
        s.state(),
        SessionState::Committed,
        "the session must not be marked committed"
    );
}

// ---------------------------------------------------------------------------
// Phase 4: commit crash-window and failure-path determinism
// ---------------------------------------------------------------------------

#[test]
fn journal_sink_failure_mid_commit_is_fail_closed_then_idempotent_on_retry() {
    // Models crash window W2: the durable journal sink fails after the
    // substrate CAS has committed a prefix of events. The commit must abort
    // fail-closed — the session is NOT marked committed — and the already
    // applied events stay idempotently protected by the substrate CAS, so a
    // retry reports them as already applied and applies only the remainder.
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1, 2, 3])).unwrap();
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);

    // Sink fails on the 2nd applied event (after the 1st was CAS-committed and
    // would have been journaled).
    let mut seen = 0u32;
    let err = s
        .commit_with_sink(&mut applier, attest_fixed(50_000), |_ev| {
            seen += 1;
            if seen == 2 {
                Err(abacus_ingest::IngestError::new(
                    ErrorCode::Persistence,
                    "simulated journal fsync failure",
                ))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Persistence);
    assert_ne!(
        s.state(),
        SessionState::Committed,
        "a sink failure must not mark the session committed (fail-closed)"
    );
    // Two events reached the substrate before the sink failed on the second
    // on_applied call (the failing sink invocation is for the 2nd event, which
    // was already CAS-committed). The journal durably recorded only the first.
    assert_eq!(applier.applied(), 2, "two events were applied before the sink failed");
    assert_eq!(applier.frontier_len(), 2);

    // Retry with a healthy sink: the two already-applied events are idempotent
    // (CAS Retry), the remaining one applies. Total applied is 3, nothing double
    // counted, and the session is now committed.
    let res = s.commit(&mut applier, attest_fixed(50_000)).unwrap();
    assert_eq!(res.already_applied, 2, "the two applied events are reported already applied");
    assert_eq!(res.applied, 1, "the retry applies only the one not-yet-applied event");
    assert_eq!(applier.applied(), 3, "three distinct events are in the substrate");
    assert_eq!(s.state(), SessionState::Committed);
}

#[test]
fn micro_unit_value_overflow_is_rejected_in_preflight_and_applies_nothing() {
    // The substrate value is the attested amount scaled by MINOR_SCALE (10^6)
    // into exact i64 micro-units. An amount within the configured minor-unit
    // ceiling can still overflow that product. The pre-flight must catch it so
    // a valid prefix is never stranded (§11): the whole commit aborts with zero
    // substrate mutations, and the amount buffer is cleared on the way out.
    let mut c = config(4);
    // Raise the ceiling so the amount passes the minor-unit ceiling check but
    // overflows only when scaled by 10^6: i64::MAX / 10^6 + slack.
    c.max_amount_minor_units = i64::MAX as u64;
    // Encode the segment under the SAME config so the fingerprint binds.
    let recs = vec![
        record(1, 1_700_000_000_000, 1, 2, 50_000),
        record(2, 1_700_000_060_000, 3, 4, 50_000),
    ];
    let bytes = encode_segment(&c, ns(), &recs).unwrap();
    let mut s = IngestionSession::create("s", c, ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &bytes).unwrap();
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    // 10^13 * 10^6 = 10^19 > i64::MAX (~9.22 * 10^18).
    let overflowing = 10_000_000_000_000i64;
    let err = s.commit(&mut applier, attest_fixed(overflowing)).unwrap_err();
    assert_eq!(err.code, ErrorCode::FeatureOverflow);
    assert_eq!(applier.applied(), 0, "nothing may be applied");
    assert_eq!(applier.frontier_len(), 0);
    assert_ne!(s.state(), SessionState::Committed);
}

#[test]
fn preflight_rejection_clears_the_attested_amount_buffer() {
    // The RAII amount guard must clear on a pre-flight rejection exit, not only
    // on success. A feature mismatch on record 1 aborts the commit; the guard's
    // Drop zeroes the staged amounts. Observable outcome: zero mutations and a
    // non-committed session (the internal buffer is not directly observable, so
    // the guard's own unit tests cover the byte-level clearing).
    let mut s = IngestionSession::create("s", config(4), ns(), wide_limits()).unwrap();
    s.submit_chunk(1, &chunk_bytes(&[1, 2])).unwrap();
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    // Record 0 attests a matching amount; record 1 attests an amount that does
    // not reproduce its frozen features -> AmountOrCurrency, before any mutation.
    let err = s
        .commit(&mut applier, |i, _r| {
            if i == 0 {
                Some(50_000)
            } else {
                Some(49_999) // wrong amount for record 1's features
            }
        })
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::AmountOrCurrency);
    assert_eq!(applier.applied(), 0);
    assert_eq!(applier.frontier_len(), 0);
    assert_ne!(s.state(), SessionState::Committed);
}
