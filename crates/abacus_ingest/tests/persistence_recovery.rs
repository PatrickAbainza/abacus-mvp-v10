//! Crash-recovery integration tests for the durable journal (Phase 2, Issue 2
//! and deliverable #9).
//!
//! Issue 2's "Critical requirement" is that the identity ledger, the account
//! mapping and the substrate state cannot diverge after a crash, and that
//! recovery is *proven*, not merely implemented. These tests drive the real
//! on-disk path — `IngestionService::with_persistence` opens (or creates) the
//! journal, streaming_ingestion_demos it, and rebuilds every session, namespace, identity ledger,
//! account map and substrate engine — and assert the observable guarantees:
//!
//! 1. A committed session is restored byte-for-byte in its report, with the
//!    applied/already-applied counters rebuilt from the journaled applied
//!    events (the substrate CAS is re-run per event, so account codes are
//!    re-verified against the journal and any divergence fails closed).
//! 2. A torn final write (a length prefix with no frame behind it) is
//!    truncated, and the acknowledged prefix still recovers.
//! 3. A mid-file corruption (a flipped payload byte far from EOF) is refused
//!    with `ErrorCode::Persistence` — never truncated, never silently repaired.
//!
//! No `tempfile` dependency (offline build): a unique directory under the OS
//! temp dir is derived from the process id, a per-test tag and an atomic
//! counter, and removed best-effort at the end of each test.

mod common;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use abacus_ingest::api::{CreateSessionRequest, IngestionService};
use abacus_ingest::config::{IdentityScheme, SourceType};
use abacus_ingest::error::ErrorCode;
use abacus_ingest::session::{ChunkStatus, SessionState};
use abacus_ingest::{encode_segment, CommitResult};
use common::{attest_fixed, config, ns, record, THRESHOLD};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A unique, empty directory under the OS temp dir for one test's journal.
fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "abacus_ingest_recovery_{}_{}_{}",
        std::process::id(),
        tag,
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A create-session request bound to the shared fixture configuration and the
/// fixed namespace, so chunks encoded with `config(4)`/`ns()` bind to it.
fn create_req() -> CreateSessionRequest {
    CreateSessionRequest {
        source_type: SourceType::SamldFixture,
        identity_scheme: IdentityScheme::SamldRecordOrdinal,
        account_id_width: 4,
        currency_code: "USD".to_string(),
        currency_precision: 2,
        threshold_minor_units: THRESHOLD,
        threshold_effective_day: 19_000,
        threshold_config_version: 3,
        parsing_policy_version: 1,
        max_amount_minor_units: 100_000_000_000,
        namespace: Some(ns()),
    }
}

/// Drive a full durable lifecycle to a committed session and return the live
/// service and the session id. Dropping the returned service models a crash:
/// the journal on disk is the only surviving state.
fn committed_service(dir: &Path) -> (IngestionService, String) {
    let mut svc = IngestionService::with_persistence(dir).expect("open durable service");
    let resp = svc.create_session(&create_req()).expect("create session");
    let id = resp.session_id.clone();

    let recs = vec![
        record(1, 1_700_000_000_000, 1, 2, 50_000),
        record(2, 1_700_000_060_000, 3, 4, 50_000),
        record(3, 1_700_000_120_000, 5, 6, 50_000),
    ];
    let bytes = encode_segment(&config(4), ns(), &recs).expect("encode segment");
    let outcome = svc.submit_chunk(&id, 1, &bytes).expect("submit chunk");
    assert_eq!(outcome.status, ChunkStatus::Accepted);
    assert_eq!(outcome.record_count, 3);
    svc.seal(&id).expect("seal");
    let cr = svc.commit(&id, attest_fixed(50_000)).expect("commit");
    assert_eq!(cr, CommitResult { applied: 3, already_applied: 0 });
    (svc, id)
}

#[test]
fn durable_commit_survives_restart_and_restores_the_committed_report() {
    let dir = fresh_dir("survive");
    let (svc, id) = committed_service(&dir);

    let before = svc.report(&id).expect("report before crash");
    assert!(before.reconciles(), "pre-crash report reconciles");
    assert_eq!(before.final_state, SessionState::Committed);
    assert_eq!(before.commit, Some(CommitResult { applied: 3, already_applied: 0 }));

    // "Crash": the process ends here. Only the journal survives.
    drop(svc);

    // Restart over the same durable directory.
    let svc2 = IngestionService::with_persistence(&dir).expect("reopen durable service");
    assert_eq!(svc2.session_count(), 1, "the committed session is restored");

    let after = svc2.report(&id).expect("report after restart");
    assert_eq!(after.final_state, SessionState::Committed, "lifecycle state restored");
    assert_eq!(
        after.commit,
        Some(CommitResult { applied: 3, already_applied: 0 }),
        "applied/already-applied counters are rebuilt from the journaled events"
    );
    assert!(after.reconciles(), "reconciliation invariant survives the restart");
    assert_eq!(after.accepted_records, 3);
    assert_eq!(after.input_records, 3);
    // A recovered committed session refuses further chunks.
    assert!(!after.final_state.accepts_chunks());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn torn_journal_tail_is_truncated_and_the_committed_session_still_recovers() {
    let dir = fresh_dir("torn");
    let (svc, id) = committed_service(&dir);
    drop(svc);

    // Simulate a torn final write: a length prefix claiming 255 bytes with no
    // frame body or checksum behind it.
    let jpath = dir.join(abacus_ingest::persist::JOURNAL_FILE);
    let mut f = std::fs::OpenOptions::new().append(true).open(&jpath).expect("open journal");
    f.write_all(&0x00FFu32.to_le_bytes()).expect("append torn prefix");
    f.sync_all().expect("sync");
    drop(f);

    // Recovery drops the torn tail and restores the acknowledged prefix.
    let svc2 = IngestionService::with_persistence(&dir).expect("reopen after torn tail");
    let r = svc2.report(&id).expect("report after torn-tail recovery");
    assert_eq!(r.final_state, SessionState::Committed);
    assert_eq!(r.commit, Some(CommitResult { applied: 3, already_applied: 0 }));
    assert!(r.reconciles());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mid_file_journal_corruption_is_refused_not_silently_repaired() {
    let dir = fresh_dir("corrupt");
    let (svc, _id) = committed_service(&dir);
    drop(svc);

    let jpath = dir.join(abacus_ingest::persist::JOURNAL_FILE);
    let mut bytes = std::fs::read(&jpath).expect("read journal");
    assert!(bytes.len() > 64, "the journal must hold several frames");
    // Flip a byte inside the FIRST frame's payload (offset 20 is payload[4] of
    // the SESSION_OPEN frame, far from EOF), so the checksum fails mid-file
    // rather than at a torn tail.
    bytes[20] ^= 0xFF;
    std::fs::write(&jpath, &bytes).expect("write corrupted journal");

    // Recovery must fail closed with a Persistence error — never truncate a
    // mid-file frame, never silently repair it.
    let err = match IngestionService::with_persistence(&dir) {
        Ok(_) => panic!("mid-file corruption must be refused, not recovered"),
        Err(e) => e,
    };
    assert_eq!(err.code, ErrorCode::Persistence, "detail: {}", err.detail);

    let _ = std::fs::remove_dir_all(&dir);
}


// ---------------------------------------------------------------------------
// Phase 4: conflict durability, crash-before-commit-marker, duplicate retry
// ---------------------------------------------------------------------------

#[test]
fn conflicting_event_survives_restart_and_the_report_still_reconciles() {
    // A2 §16.2 lists "identity conflicts" as a journaled state change. A
    // conflicting delivery must survive a restart with its count intact, and
    // the §10 reconciliation invariant `input = accepted + duplicate +
    // conflicting + rejected` must still hold afterwards.
    let dir = fresh_dir("conflict");
    let mut svc = IngestionService::with_persistence(&dir).expect("open durable service");
    let resp = svc.create_session(&create_req()).expect("create session");
    let id = resp.session_id.clone();

    // Chunk 1: one valid new event (id 1).
    let good = encode_segment(&config(4), ns(), &[record(1, 1_700_000_000_000, 1, 2, 50_000)])
        .expect("encode good");
    svc.submit_chunk(&id, 1, &good).expect("submit good");

    // Chunk 2: the SAME identity (id 1) with different content -> a conflict.
    let conflicting = abacus_ingest::record::CanonicalRecord {
        identity: abacus_ingest::EventIdentity::new(ns(), 1),
        event_time_ms: 1_700_000_000_000,
        sender_id: 1,
        receiver_id: 2,
        features: abacus_ingest::FeaturePayload::derive(999_999, THRESHOLD, 100).unwrap(),
    };
    let bad = encode_segment(&config(4), ns(), &[conflicting]).expect("encode conflict");
    svc.submit_chunk(&id, 2, &bad).expect("submit conflict");

    svc.seal(&id).expect("seal");
    svc.commit(&id, attest_fixed(50_000)).expect("commit");

    let before = svc.report(&id).expect("report before crash");
    assert_eq!(before.conflicting_records, 1, "one conflict observed pre-crash");
    assert_eq!(before.accepted_records, 1, "only the non-conflicting event is buffered");
    assert_eq!(before.input_records, 2, "both deliveries count as input");
    assert!(before.reconciles(), "pre-crash report reconciles");

    // "Crash": only the journal survives.
    drop(svc);
    let svc2 = IngestionService::with_persistence(&dir).expect("reopen after conflict");
    let after = svc2.report(&id).expect("report after restart");
    assert_eq!(
        after.conflicting_records, 1,
        "the conflict count must survive the restart (A2 §16.2)"
    );
    assert_eq!(after.input_records, 2);
    assert_eq!(after.accepted_records, 1);
    assert!(
        after.reconciles(),
        "the §10 partition must hold after recovery: {:?}",
        after
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn crash_before_commit_marker_recovers_the_prefix_and_retry_is_idempotent() {
    // Crash window W3: every EventApplied frame is durable, but the trailing
    // SessionCommitted frame is not (simulated by truncating exactly that last
    // frame). On restart the session is NOT committed, yet the substrate has
    // all three events (streaming_ingestion_demoed). A retry re-runs the CAS per event: each is
    // reported already-applied, nothing is double counted, and the session
    // becomes committed.
    let dir = fresh_dir("precommit");
    let (svc, id) = committed_service(&dir);
    drop(svc);

    // Truncate the final SessionCommitted frame: re-read the journal, find the
    // last frame boundary, and cut there. We locate it by scanning frame
    // lengths from the 12-byte header and keeping every complete frame except
    // the last.
    let jpath = dir.join(abacus_ingest::persist::JOURNAL_FILE);
    let mut bytes = std::fs::read(&jpath).expect("read journal");
    assert!(bytes.len() > 12, "journal has frames");
    // Walk frames: header(12) then repeated [u32 len | payload | 32-byte sha].
    let mut offsets = vec![12u64];
    let mut pos = 12u64;
    while pos + 4 <= bytes.len() as u64 {
        let plen = u32::from_le_bytes([
            bytes[pos as usize],
            bytes[pos as usize + 1],
            bytes[pos as usize + 2],
            bytes[pos as usize + 3],
        ]) as u64;
        pos += 4 + plen + 32;
        offsets.push(pos);
    }
    // The last complete frame starts at offsets[len-2]; truncate there.
    let cut = offsets[offsets.len().saturating_sub(2)];
    bytes.truncate(cut as usize);
    std::fs::write(&jpath, &bytes).expect("write truncated journal");

    // Restart: the session is restored but NOT committed.
    let mut svc2 = IngestionService::with_persistence(&dir).expect("reopen");
    let r = svc2.report(&id).expect("report after truncated commit marker");
    assert_ne!(
        r.final_state,
        SessionState::Committed,
        "without the SessionCommitted frame the session must not be committed"
    );

    // Retry the commit: every event is already in the substrate -> idempotent.
    let cr = svc2.commit(&id, attest_fixed(50_000)).expect("retry commit");
    assert_eq!(cr.already_applied, 3, "all three events already applied");
    assert_eq!(cr.applied, 0, "nothing is applied a second time");
    let after = svc2.report(&id).expect("report after retry");
    assert_eq!(after.final_state, SessionState::Committed);
    assert!(after.reconciles());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn committed_session_retry_after_restart_is_a_clean_refusal() {
    // A duplicate retry of an already-durably-committed session must be
    // refused (state conflict), never re-applied, and must leave the report's
    // committed outcome untouched.
    let dir = fresh_dir("dupretry");
    let (svc, id) = committed_service(&dir);
    drop(svc);

    let mut svc2 = IngestionService::with_persistence(&dir).expect("reopen");
    let err = svc2
        .commit(&id, attest_fixed(50_000))
        .expect_err("a committed session must refuse a second commit");
    assert_eq!(err.code, ErrorCode::Session);
    let after = svc2.report(&id).expect("report");
    assert_eq!(after.final_state, SessionState::Committed);
    assert_eq!(after.commit, Some(CommitResult { applied: 3, already_applied: 0 }));
    assert!(after.reconciles());

    let _ = std::fs::remove_dir_all(&dir);
}
