//! End-to-end tests for the browser CSV client: the full §12 procedure against
//! the real `/api/v1` server.
//!
//! Each test drives the actual pipeline:
//!
//! 1. `run_import` normalizes a deterministic CSV fixture into bounded
//!    `ABACSEG1` chunks;
//! 2. the client uploads them over a real TCP connection to the real
//!    [`abacus_ingest::api::server::route`] handler;
//! 3. the server seals, commits through the attested-amount channel, and renders
//!    a report;
//! 4. the test asserts the report reconciles and the substrate agrees.
//!
//! The required scenarios from the task are all here: invalid input, attestation
//! mismatch, successful commit, report reconciliation, and restart recovery.
//!
//! Deterministic by construction: the fixture's amounts, timestamps and account
//! identifiers are literals, and the namespace is pinned so a retry re-derives
//! identical identities. No clock, no randomness, no network beyond loopback.

use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use abacus_csv_client::driver::{render_preview, run_import};
use abacus_csv_client::mapping::{CanonicalField, ColumnMapping};
use abacus_csv_client::policy::{
    AmountFormat, EpochUnit, IngestPlan, NumberPolicy, SubSecondPolicy, TimestampPolicy,
    TimestampStyle,
};
use abacus_csv_client::{CsvIngestClient, CsvOptions, EventNamespace, IdentifierPolicy};
/// Shorthand for the pipeline's JSON model, used throughout the assertions.
use abacus_ingest::api::json::Json as J;

use abacus_ingest::api::server::handle_one;
use abacus_ingest::api::IngestionService;

/// The deterministic CSV fixture, a semicolon-delimited export.
///
/// The delimiter is `;` rather than `,` because the declared locale uses `,` as
/// the **grouping** separator and `.` as the decimal separator (a de-DE style
/// export). A comma delimiter with a comma group separator is genuinely
/// ambiguous — `a,1,234.56,b` cannot be split reliably — and the plan's
/// validation correctly refuses that combination. So the fixture states the
/// delimiter explicitly instead, which is also what §12 step 4 asks for.
///
/// The rows cover every row-level rejection category the client can produce:
/// two clean rows, an unparseable amount, a non-UTC-offset timestamp under the
/// offset style, an empty sender, and an amount with excess fractional
/// precision.
const CSV: &str = "\
timestamp;sender;receiver;amount
2026-10-12T09:30:00Z;acct-001;acct-002;123.45
2026-10-12T10:00:00Z;acct-002;acct-003;1,234.56
2026-10-12T10:30:00Z;acct-003;acct-004;not-a-number
2026-10-12T11:00:00;acct-004;acct-005;50.00
2026-10-12T11:30:00Z;;acct-006;75.00
2026-10-12T12:00:00Z;acct-006;acct-007;10.005
2026-10-12T12:30:00Z;acct-007;acct-001;999.99
";

/// The threshold the fixture is configured against (100.00 at precision 2).
const THRESHOLD: u64 = 10_000;

/// The pinned namespace, so a retry re-derives identical event identities.
fn namespace() -> EventNamespace {
    EventNamespace::from_bytes([
        0x5c, 0x11, 0xa7, 0x40, 0x2e, 0x9d, 0x4b, 0x81, 0x93, 0x05, 0xc7, 0x22, 0x48, 0xbd, 0x6e,
        0x10,
    ])
}

/// The complete explicit plan for the fixture.
fn plan(records_per_chunk: usize) -> IngestPlan {
    IngestPlan {
        mapping: ColumnMapping::new([
            (CanonicalField::Timestamp, 0),
            (CanonicalField::Sender, 1),
            (CanonicalField::Receiver, 2),
            (CanonicalField::Amount, 3),
        ])
        .expect("the fixture mapping is complete and non-overlapping"),
        csv: CsvOptions {
            delimiter: b';',
            ..CsvOptions::default()
        },
        timestamp: TimestampPolicy {
            style: TimestampStyle::Iso8601WithOffset,
            source_offset_minutes: None,
        },
        amount: AmountFormat {
            decimal_separator: b'.',
            group_separator: Some(b','),
            accept_negative: false,
        },
        numbers: NumberPolicy {
            epoch_unit: EpochUnit::Millis,
            subsecond: SubSecondPolicy::RejectSubMilli,
        },
        identifiers: IdentifierPolicy {
            trim: true,
            case_sensitive: true,
            account_id_width: 4,
        },
        currency_code: "USD".to_string(),
        currency_precision: 2,
        threshold_minor_units: THRESHOLD,
        threshold_effective_day: 20_738,
        threshold_config_version: 3,
        max_amount_minor_units: 100_000_000_000,
        parsing_policy_version: 1,
        records_per_chunk,
        namespace: Some(namespace()),
    }
}

/// Bind a loopback listener on an ephemeral port and serve the ingestion API.
///
/// Returns the client and the server thread's join handle. The server is the
/// real [`abacus_ingest::api::server::handle_one`], so the client exercises the
/// actual HTTP surface rather than a stub.
fn start_server() -> (CsvIngestClient, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let service = Arc::new(Mutex::new(IngestionService::new()));
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let svc = Arc::clone(&service);
            thread::spawn(move || handle_one(&svc, stream));
        }
    });
    (CsvIngestClient::new(addr.to_string()), handle)
}

/// A loopback listener that accepts each connection and immediately drops it, so
/// the client observes a genuine transport failure and its retry policy runs.
fn start_silent_server() -> (CsvIngestClient, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            drop(stream);
        }
    });
    (CsvIngestClient::new(addr.to_string()), handle)
}

// ---------------------------------------------------------------------------
// Normalization: the preview (§12 steps 1-5)
// ---------------------------------------------------------------------------

#[test]
fn the_preview_reports_every_row_level_problem_by_source_line() {
    // §12 step 5: "Preview normalized records and report invalid rows, ambiguous
    // values, identifier collisions, and unsupported amounts." Every one of the
    // four fixture rejection categories must appear, keyed by source line, and
    // the counts must account for every input row.
    let p = plan(10);
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    let pv = &norm.preview;

    assert_eq!(pv.input_rows, 7, "the fixture has 7 data rows");
    assert_eq!(pv.accepted_rows, 3, "rows 2, 3 and 8 are clean");
    assert_eq!(pv.rejected_rows, 4);
    // §10: every input row is accounted for.
    assert_eq!(pv.input_rows, pv.accepted_rows + pv.rejected_rows);

    let codes: Vec<(usize, &str)> = pv
        .issues
        .iter()
        .map(|i| (i.line, i.code.as_str()))
        .collect();
    assert_eq!(
        codes,
        vec![
            (4, "amount_or_currency"), // "not-a-number"
            (5, "timestamp"),          // no UTC offset under the offset style
            (6, "account_id"),         // empty sender
            (7, "amount_or_currency"), // 10.005 exceeds precision 2
        ]
    );

    // §12: the UI must not display raw accounts or amounts.
    for issue in &pv.issues {
        assert!(
            !issue.detail.contains("acct-"),
            "an error detail leaked an account string: {}",
            issue.detail
        );
        assert!(
            !issue.detail.contains("not-a-number"),
            "an error detail leaked the raw amount: {}",
            issue.detail
        );
    }
}

#[test]
fn the_preview_never_contains_a_raw_account_or_amount() {
    // §17.7: "Browser-uploaded payloads contain no raw account strings" and "no
    // raw amounts". The rendered preview is the one string a UI shows, so it is
    // swept for both.
    let p = plan(10);
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    let rendered = render_preview(&norm);
    assert!(
        !rendered.contains("acct-"),
        "the preview leaked an account string"
    );
    assert!(!rendered.contains("123.45"), "the preview leaked an amount");
    assert!(
        !rendered.contains("not-a-number"),
        "the preview leaked a raw value"
    );
}

#[test]
fn identifiers_fold_to_opaque_codes_and_the_codes_are_reproducible() {
    // §6: "Parse source identifiers as strings" and "IDs must remain stable
    // throughout the applicable dataset."
    //
    // Six distinct accounts, not seven: the row with an empty sender is rejected
    // before its receiver is interned, so `acct-006` enters on the row whose
    // *amount* is rejected instead. Identifiers are interned ahead of the amount
    // check deliberately — it keeps the code assignment a pure function of
    // source order, so a later row's code never shifts when an earlier row's
    // failure category changes. That is what §6's "must remain stable" needs.
    let p = plan(10);
    let a = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    let b = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    assert_eq!(a.preview.distinct_accounts, 6);
    assert_eq!(a.preview.account_id_width, 4);
    // Deterministic: the same file always yields the same codes and the same
    // bytes, which is what makes a retry idempotent at the substrate.
    assert_eq!(
        a.chunks, b.chunks,
        "the encoded bytes must be deterministic"
    );
}

#[test]
fn chunking_is_bounded_by_the_plan() {
    // §13: "Chunk size must be configurable within server-enforced bounds." Two
    // accepted rows with records_per_chunk = 1 must produce two segments; with
    // records_per_chunk = 10, one.
    let one = abacus_csv_client::normalize(CSV.as_bytes(), &plan(1), namespace()).unwrap();
    assert_eq!(one.chunks.len(), 3, "one segment per accepted record");
    for c in &one.chunks {
        assert!(c.len() >= 176, "each chunk is a complete ABACSEG1 segment");
    }
    let many = abacus_csv_client::normalize(CSV.as_bytes(), &plan(10), namespace()).unwrap();
    assert_eq!(many.chunks.len(), 1);
}

// ---------------------------------------------------------------------------
// Upload, commit, report: the live path against the real server
// ---------------------------------------------------------------------------

#[test]
fn a_confirmed_csv_uploads_commits_and_reconciles() {
    // The stop condition: "A CSV can be selected in the browser, normalized,
    // uploaded, committed, and verified in the existing tensor substrate."
    let (client, _server) = start_server();
    let out =
        run_import(CSV.as_bytes(), &plan(10), &client, true).expect("the import must succeed");

    // The three clean rows were accepted, chunked into one segment and applied.
    assert_eq!(out.preview.accepted_rows, 3);
    assert_eq!(out.acks.len(), 1);
    assert!(!out.acks[0].already_processed);
    assert_eq!(out.acks[0].record_count, 3);
    assert_eq!(
        out.commit.applied, 3,
        "all three events applied exactly once"
    );
    assert_eq!(out.commit.already_applied, 0);

    // §10: the report must reconcile, and it must name the source actually used.
    let r = &out.report;
    assert_eq!(r.get("reconciles").and_then(J::as_bool), Some(true));
    assert_eq!(
        r.get("source_type").and_then(J::as_str),
        Some("browser_csv")
    );
    assert_eq!(
        r.get("identity_scheme").and_then(J::as_str),
        Some("csv_record_ordinal")
    );
    let counts = r.get("counts").expect("counts");
    assert_eq!(counts.get("input").and_then(J::as_u64), Some(3));
    assert_eq!(counts.get("accepted").and_then(J::as_u64), Some(3));
    assert_eq!(
        r.get("commit")
            .and_then(|c| c.get("applied"))
            .and_then(J::as_u64),
        Some(3)
    );
}
#[test]
fn an_unconfirmed_import_creates_no_session_and_uploads_nothing() {
    // §12 step 6: "Require explicit user confirmation before uploading." Without
    // it the run stops after the preview: no session, no chunk, no commit.
    let (client, server) = start_server();
    let err = run_import(CSV.as_bytes(), &plan(10), &client, false)
        .expect_err("an unconfirmed import must be refused");
    assert_eq!(err.code, abacus_ingest::ErrorCode::Session);
    assert!(err.detail.contains("not confirmed"));

    // The server saw nothing: there is no session to report.
    let resp = client
        .report("sess-1")
        .expect("the request itself succeeds");
    assert_eq!(resp.status, 404, "no session was created");
    drop(server);
}

#[test]
fn a_file_with_no_accepted_rows_is_refused_before_any_session_is_created() {
    // A file with zero accepted rows is not an import; opening a session for it
    // would be a meaningless state change.
    let (client, server) = start_server();
    for csv in [
        "timestamp;sender;receiver;amount\n",
        "timestamp;sender;receiver;amount\n2026-10-12T09:30:00Z;acct-001;acct-002;not-a-number\n",
    ] {
        let err = run_import(csv.as_bytes(), &plan(10), &client, true)
            .expect_err("nothing accepted means nothing to upload");
        assert_eq!(err.code, abacus_ingest::ErrorCode::FileOrCsvParse);
    }
    let resp = client.report("sess-1").expect("request succeeds");
    assert_eq!(resp.status, 404, "no session was ever created");
    drop(server);
}

#[test]
fn a_wrong_attestation_is_rejected_and_applies_nothing() {
    // Amendment A1: the attested amount must reproduce the record's frozen
    // feature payload. A tampered amount is a rejection with zero substrate
    // mutations — never a fabricated value.
    let (client, _server) = start_server();
    let p = plan(10);
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    let caps = client.capabilities().expect("capabilities");
    assert!(caps.account_id_widths.contains(&4));

    let cfg = p.to_source_config().unwrap();
    let created = client
        .create_session(&session_body(&cfg, &namespace()))
        .expect("create session");
    assert_eq!(created.status, 200);
    let sid = created
        .json
        .as_ref()
        .and_then(|j| j.get("session_id"))
        .and_then(J::as_str)
        .expect("session id")
        .to_string();

    for (i, chunk) in norm.chunks.iter().enumerate() {
        let r = client
            .upload_chunk(&sid, i as u64 + 1, chunk)
            .expect("upload");
        assert_eq!(r.status, 200);
    }
    assert_eq!(client.seal(&sid).expect("seal").status, 200);

    // Attest 99999 everywhere: only one of the three records is actually
    // 999.99, so the pre-flight feature check must reject the other two.
    let amounts = vec!["99999"; norm.preview.accepted_rows as usize].join(",");
    let resp = client
        .commit(&sid, &format!("{{\"amounts\":[{amounts}]}}"))
        .expect("the request completes");
    assert_eq!(
        resp.status, 400,
        "a mismatched attestation must be rejected"
    );
    assert_eq!(
        resp.error_code().as_deref(),
        Some("amount_or_currency"),
        "the server must name the category"
    );

    // And nothing was applied: the report shows no commit outcome.
    let report = client.report(&sid).expect("report");
    assert_eq!(report.status, 200);
    let j = report.json.expect("report json");
    assert!(
        j.get("commit").is_some_and(|c| matches!(c, J::Null)),
        "a failed commit must leave no commit outcome"
    );
    assert_ne!(
        j.get("final_state").and_then(J::as_str),
        Some("committed"),
        "the session must not be committed"
    );
}

#[test]
fn a_repeated_chunk_upload_is_idempotent() {
    // §13: "Repeating a chunk with the same identity and identical contents is
    // idempotent." The client must recognise `already_processed` and not count
    // it as a second delivery.
    let (client, _server) = start_server();
    let p = plan(10);
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    let cfg = p.to_source_config().unwrap();
    let created = client
        .create_session(&session_body(&cfg, &namespace()))
        .expect("create session");
    let sid = created
        .json
        .as_ref()
        .and_then(|j| j.get("session_id"))
        .and_then(J::as_str)
        .expect("session id")
        .to_string();

    let first = client
        .upload_chunk(&sid, 1, &norm.chunks[0])
        .expect("first upload");
    assert_eq!(first.status, 200);
    assert_ne!(
        first
            .json
            .as_ref()
            .and_then(|j| j.get("status"))
            .and_then(J::as_str),
        Some("already_processed")
    );

    // Byte-identical retry: acknowledged, changes nothing.
    let again = client
        .upload_chunk(&sid, 1, &norm.chunks[0])
        .expect("retry");
    assert_eq!(again.status, 200);
    assert_eq!(
        again
            .json
            .as_ref()
            .and_then(|j| j.get("status"))
            .and_then(J::as_str),
        Some("already_processed"),
        "an identical retry must be recognised"
    );
}

#[test]
fn the_same_chunk_sequence_with_different_bytes_is_rejected() {
    // §13: "Repeating a chunk identity with different contents is rejected."
    let (client, _server) = start_server();
    let p = plan(10);
    let cfg = p.to_source_config().unwrap();
    let created = client
        .create_session(&session_body(&cfg, &namespace()))
        .expect("create session");
    let sid = created
        .json
        .as_ref()
        .and_then(|j| j.get("session_id"))
        .and_then(J::as_str)
        .expect("session id")
        .to_string();

    let a = abacus_csv_client::normalize(CSV.as_bytes(), &plan(1), namespace()).unwrap();
    assert_eq!(
        client
            .upload_chunk(&sid, 1, &a.chunks[0])
            .expect("upload")
            .status,
        200
    );

    // A different chunk at the same sequence: a different record set.
    let b = abacus_csv_client::normalize(CSV.as_bytes(), &plan(10), namespace()).unwrap();
    let conflict = client
        .upload_chunk(&sid, 1, &b.chunks[0])
        .expect("request completes");
    assert_eq!(
        conflict.status, 409,
        "different bytes at one sequence is a conflict"
    );
    assert_eq!(conflict.error_code().as_deref(), Some("session"));
}

#[test]
fn a_transport_failure_fails_closed_after_the_bounded_retries() {
    // §18: "explicit resource limits." A dead server must produce an error after
    // the configured attempts, never a silent success and never an unbounded loop.
    use abacus_csv_client::client::RetryPolicy;
    use std::time::Duration;

    let (client, _server) = start_silent_server();
    let client = client.with_retry(RetryPolicy {
        max_attempts: 2,
        backoff: Duration::from_millis(1),
    });
    let err = client
        .capabilities()
        .expect_err("a dropped connection must be an error");
    assert!(
        err.detail.contains("failed after 2 attempts"),
        "the retry bound must be visible in the error: {}",
        err.detail
    );
}

#[test]
fn a_sealed_session_refuses_further_chunks() {
    // §13: "Completed sessions cannot silently accept unrelated additional
    // records."
    let (client, _server) = start_server();
    let p = plan(10);
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    let cfg = p.to_source_config().unwrap();
    let created = client
        .create_session(&session_body(&cfg, &namespace()))
        .expect("create");
    let sid = created
        .json
        .as_ref()
        .and_then(|j| j.get("session_id"))
        .and_then(J::as_str)
        .expect("session id")
        .to_string();
    client
        .upload_chunk(&sid, 1, &norm.chunks[0])
        .expect("upload");
    assert_eq!(client.seal(&sid).expect("seal").status, 200);

    let after = client
        .upload_chunk(&sid, 2, &norm.chunks[0])
        .expect("request completes");
    assert_eq!(
        after.status, 409,
        "a sealed session must refuse more chunks"
    );
}

// ---------------------------------------------------------------------------
// Restart recovery (§13: "Interrupted sessions have explicit resume or abort
// semantics")
// ---------------------------------------------------------------------------

#[test]
fn a_crash_between_upload_and_commit_recovers_and_commits_once() {
    // The client's own durable state is only the resume token, so recovery means:
    // re-derive everything from the same file, then re-attach to the same
    // session and namespace. The substrate CAS must then treat the re-delivered
    // events as already applied rather than applying them twice.
    use abacus_csv_client::ResumeToken;

    let dir = fresh_dir("browser_resume");
    // Phase 1: durable service, upload every chunk, then "crash" — the
    // in-memory service is dropped and only the journal survives.
    let sid = {
        let service = Arc::new(Mutex::new(
            abacus_ingest::api::IngestionService::with_persistence(&dir).expect("durable service"),
        ));
        upload_all(&service, &plan(10))
    };

    // Phase 2: restart over the same directory and serve again.
    let service = Arc::new(Mutex::new(
        abacus_ingest::api::IngestionService::with_persistence(&dir)
            .expect("reopen durable service"),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let svc = Arc::clone(&service);
    thread::spawn(move || {
        for s in listener.incoming() {
            let Ok(s) = s else { break };
            let svc = Arc::clone(&svc);
            thread::spawn(move || handle_one(&svc, s));
        }
    });
    let client = CsvIngestClient::new(addr.to_string());

    // The resume token round-trips, and the fingerprint still matches, which is
    // what §13 requires before the session may be reused.
    let p = plan(10);
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &p, namespace()).unwrap();
    let cfg = p.to_source_config().unwrap();
    let token = ResumeToken {
        session_id: sid.clone(),
        namespace: namespace(),
        config_fingerprint: cfg.fingerprint_hex(),
        ack_chunks: 1,
        accepted_records: norm.preview.accepted_rows,
    };
    assert_eq!(
        ResumeToken::decode(&token.encode()).expect("round-trips"),
        token
    );
    assert!(token.matches_fingerprint(&cfg.fingerprint_hex()));

    // Re-upload: idempotent, so the recovery changes nothing already durable.
    for (i, chunk) in norm.chunks.iter().enumerate() {
        let r = client
            .upload_chunk(&sid, i as u64 + 1, chunk)
            .expect("re-upload");
        assert_eq!(r.status, 200);
        assert_eq!(
            r.json
                .as_ref()
                .and_then(|j| j.get("status"))
                .and_then(J::as_str),
            Some("already_processed"),
            "the re-delivered chunk must be recognised as a duplicate"
        );
    }

    // Seal and commit. Nothing was applied before the crash, so this is the
    // first application.
    assert_eq!(client.seal(&sid).expect("seal").status, 200);
    let body = norm.attested_amounts.commit_body();
    let first = client.commit(&sid, &body).expect("commit");
    assert_eq!(first.status, 200);
    assert_eq!(
        first
            .json
            .as_ref()
            .and_then(|j| j.get("applied"))
            .and_then(J::as_u64),
        Some(3),
        "the three events apply exactly once"
    );

    // A repeated commit is a 409 by design: `Committed` is terminal, and the
    // session state machine refuses to re-enter it. Exactly-once is verified
    // through the report, not through a second commit — which is why the
    // client that loses the first commit response reads the report instead of
    // retrying blindly. The evidence is below.
    let again = client.commit(&sid, &body).expect("commit retry completes");
    assert_eq!(
        again.status, 409,
        "a second commit is refused, never re-applied"
    );
    assert_eq!(
        again.error_code().as_deref(),
        Some("session"),
        "the refusal must carry the session category"
    );

    // And the recovered report reconciles, with the commit outcome intact.
    let report = client.report(&sid).expect("report");
    let j = report.json.expect("report json");
    assert_eq!(j.get("reconciles").and_then(J::as_bool), Some(true));
    assert_eq!(j.get("final_state").and_then(J::as_str), Some("committed"));

    drop(service);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A unique, empty directory under the OS temp dir for one test's journal.
fn fresh_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "abacus_csv_client_{}_{}_{}",
        std::process::id(),
        tag,
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Serve a durable service, upload every chunk of the fixture, and return the
/// session id — stopping short of the commit, which is the crash window.
fn upload_all(
    service: &Arc<Mutex<abacus_ingest::api::IngestionService>>,
    plan: &IngestPlan,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let svc = Arc::clone(service);
    thread::spawn(move || {
        for s in listener.incoming() {
            let Ok(s) = s else { break };
            let svc = Arc::clone(&svc);
            thread::spawn(move || handle_one(&svc, s));
        }
    });
    let client = CsvIngestClient::new(addr.to_string());
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), plan, namespace()).unwrap();
    let cfg = plan.to_source_config().unwrap();
    let created = client
        .create_session(&session_body(&cfg, &namespace()))
        .expect("create");
    let sid = created
        .json
        .as_ref()
        .and_then(|j| j.get("session_id"))
        .and_then(J::as_str)
        .expect("session id")
        .to_string();
    for (i, chunk) in norm.chunks.iter().enumerate() {
        assert_eq!(
            client
                .upload_chunk(&sid, i as u64 + 1, chunk)
                .expect("upload")
                .status,
            200
        );
    }
    sid
}

/// The create-session body for a validated configuration, mirroring the driver's
/// own encoding so the tests exercise the same JSON the client sends.
fn session_body(
    config: &abacus_ingest::config::SourceConfig,
    namespace: &EventNamespace,
) -> String {
    format!(
        "{{\"source_type\":\"browser_csv\",\"identity_scheme\":\"csv_record_ordinal\",\
         \"account_id_width\":{},\"currency_code\":\"{}\",\"currency_precision\":{},\
         \"threshold_minor_units\":{},\"threshold_effective_day\":{},\
         \"threshold_config_version\":{},\"parsing_policy_version\":{},\
         \"max_amount_minor_units\":{},\"event_namespace\":\"{}\"}}",
        config.account_id_width,
        config.currency_code.as_str(),
        config.currency_precision,
        config.threshold_minor_units,
        config.threshold_effective_day,
        config.threshold_config_version,
        config.parsing_policy_version,
        config.max_amount_minor_units,
        namespace.to_hyphenated()
    )
}
