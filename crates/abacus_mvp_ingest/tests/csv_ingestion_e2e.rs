//! Acceptance test: CSV events reach the `abacus_mvp_v10` tensor substrate
//! through the reused `abacus_ingestion` oracle pipeline.
//!
//! Two independent proofs, both required by the task ("prove that CSV events
//! reach the intended `abacus_mvp_v10` tensor substrate"):
//!
//! 1. **In-process, direct substrate read.** [`apply_csv_to_substrate`] runs the
//!    canonical CSV → `ABACSEG1` → session → commit path and hands back the live
//!    [`abacus_ingest::SubstrateApplier`], whose `engine()` *is* the
//!    `abacus_tensor::IngestionEngine`. The test reads that engine's frontier,
//!    ledger and stats directly and asserts the three clean transfers landed
//!    with their exact micro-unit values.
//! 2. **Over the real `/api/v1` HTTP surface.** The completed headless
//!    [`run_import`] drives the genuine loopback server; the test asserts the
//!    attested-amount commit succeeded and the report reconciles.
//!
//! Deterministic by construction: the fixture's amounts, timestamps and
//! accounts are literals and the namespace is pinned. No clock, no randomness,
//! no network beyond loopback.

use abacus_mvp_ingest::{apply_csv_to_substrate, plan, start_ingestion_server, CSV};
use abacus_tensor::tensor::value::MINOR_SCALE;

/// The substrate micro-unit value of each accepted row: exact minor amount
/// scaled by the substrate's `10^6` embedding (the same embedding the fixture
/// path uses, so browser- and CSV-derived values are commensurable).
const EXPECTED_VALUES: [i64; 3] = [12_345, 123_456, 99_999];

#[test]
fn csv_events_reach_the_mvp_tensor_substrate() {
    let applier = apply_csv_to_substrate(10).expect("the CSV import must commit");
    let engine = applier.engine();

    // Three clean rows were accepted and applied exactly once.
    assert_eq!(applier.applied(), 3, "all three clean events applied once");
    assert_eq!(applier.duplicates(), 0, "no retries on a fresh import");

    // The substrate's own accounting agrees: three creates, three deltas.
    let stats = engine.stats();
    assert_eq!(stats.creates, 3, "three Create-mode ops");
    assert_eq!(stats.deltas_emitted, 3, "one frontier delta per event");

    // The committed frontier holds exactly the three events.
    assert_eq!(engine.frontier().len(), 3, "three frontier deltas");

    // And their values are the attested amounts at the substrate's micro scale —
    // a set comparison, independent of account-interning order.
    let mut got: Vec<i64> = engine
        .frontier()
        .entries()
        .iter()
        .map(|e| e.value)
        .collect();
    got.sort_unstable();
    let mut want: Vec<i64> = EXPECTED_VALUES.iter().map(|v| v * MINOR_SCALE).collect();
    want.sort_unstable();
    assert_eq!(got, want, "the frontier holds the exact attested values");

    // The one shared append-only account namespace interned the four distinct
    // accounts the clean rows touch (acct-001/002/003/007); id 0 stays reserved
    // (`account_count` excludes the reserved sentinel that `len` includes).
    assert_eq!(
        applier.accounts().account_count(),
        4,
        "one shared account namespace"
    );
}

#[test]
fn csv_import_over_http_reconciles_and_reaches_the_substrate() {
    let (client, _addr, _server) = start_ingestion_server();

    let out = abacus_csv_client::driver::run_import(CSV.as_bytes(), &plan(10), &client, true)
        .expect("the confirmed import must succeed");

    // The three clean rows were accepted, chunked and committed through the
    // attested-amount path.
    assert_eq!(out.preview.accepted_rows, 3);
    assert_eq!(
        out.commit.applied, 3,
        "all three events applied exactly once"
    );
    assert_eq!(out.commit.already_applied, 0);

    // §10: the report must reconcile and name the source actually used.
    let r = &out.report;
    assert_eq!(r.get("reconciles").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        r.get("source_type").and_then(|v| v.as_str()),
        Some("browser_csv")
    );
    assert_eq!(
        r.get("identity_scheme").and_then(|v| v.as_str()),
        Some("csv_record_ordinal")
    );
    assert_eq!(
        r.get("commit")
            .and_then(|c| c.get("applied"))
            .and_then(|v| v.as_u64()),
        Some(3)
    );
}

#[test]
fn csv_normalizes_to_complete_abacseg1_segments() {
    // The reused adapter emits bounded, independently validatable ABACSEG1
    // segments (each >= the 176-byte header); three accepted rows at one chunk
    // size produce a single segment.
    let p = plan(10);
    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &p, abacus_mvp_ingest::namespace())
        .expect("normalize");
    assert_eq!(norm.preview.accepted_rows, 3);
    assert_eq!(norm.preview.rejected_rows, 4);
    assert_eq!(norm.chunks.len(), 1, "three rows fit one bounded chunk");
    assert!(norm.chunks[0].len() >= 176, "a complete ABACSEG1 segment");
}

#[test]
fn the_csv_test_harness_endpoint_reconciles() {
    // The browser test-harness convenience path (POST /api/ingest/csv) drives
    // the same canonical service in-process and returns preview + reconciled
    // report JSON.
    use abacus_ingest::api::IngestionService;
    let mut svc = IngestionService::new();
    let json = abacus_mvp_ingest::ingest_csv_bytes(&mut svc, CSV.as_bytes())
        .expect("ingest_csv_bytes must succeed");
    let parsed = abacus_ingest::api::json::parse(&json).expect("valid JSON reply");

    assert_eq!(
        parsed.get("reconciles").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        parsed
            .get("commit")
            .and_then(|c| c.get("applied"))
            .and_then(|v| v.as_u64()),
        Some(3)
    );
    let preview = parsed.get("preview").expect("preview");
    assert_eq!(
        preview.get("accepted_rows").and_then(|v| v.as_u64()),
        Some(3)
    );
    assert_eq!(
        preview.get("rejected_rows").and_then(|v| v.as_u64()),
        Some(4)
    );
    assert_eq!(preview.get("input_rows").and_then(|v| v.as_u64()), Some(7));

    // Privacy: the reply must not carry raw account strings or amounts.
    assert!(!json.contains("acct-00"));
    assert!(!json.contains("\"amounts\""));
}
