//! SAML-D dataset ingestion acceptance test.
//!
//! Proves the two real SAML-D export files' schema is supported by the
//! `abacus_mvp_v10` ingestion procedure, following the `abacus_ingestion`
//! oracle's `samld_validate` + `samld_plan` policy exactly. The SAML-D
//! two-column `Time`/`Date` timestamp is merged value-preservingly into one
//! ISO-8601 UTC column, mapped to Sender/Receiver/Amount, and driven through
//! the canonical `ABACSEG1` → session → commit path into this repository's
//! `abacus_tensor` substrate.

use abacus_ingest::api::IngestionService;
use abacus_mvp_ingest::{ingest_samld_bytes, is_samld, samld_namespace};

/// A small SAML-D slice with the real header and representative rows: clean
/// 2-decimal amounts plus one excess-precision amount the pipeline must reject
/// (never round — Golden Source §8/§19). Ground-truth label columns are present
/// but never mapped, so they stay out of feature derivation.
const SAMLD_SLICE: &str = "\
Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location,Payment_type,Is_laundering,Laundering_type
10:35:19,2022-10-07,8724731955,2769355426,1459.15,UK pounds,UK pounds,UK,UK,Cash Deposit,0,Normal_Cash_Deposits
10:35:20,2022-10-07,1491989064,8401255335,6019.64,UK pounds,Dirham,UK,UAE,Cross-border,0,Normal_Fan_Out
00:00:07,2023-04-05,2102474519,5041444792,4371.33,UK pounds,UK pounds,UK,UK,Cheque,0,Normal_Small_Fan_Out
23:59:58,2023-07-03,1,2,0.05,UK pounds,UK pounds,UK,UK,Cash Deposit,0,Normal_Fan_In
12:00:00,2023-01-15,10753.26726981541,5041444792,10753.26726981541,UK pounds,UK pounds,UK,UK,Wire,1,Laundering_Fan_Out
";

#[test]
fn samld_header_is_detected() {
    assert!(is_samld(SAMLD_SLICE.as_bytes()));
    // A non-SAML-D CSV must not be mistaken for one.
    assert!(!is_samld(
        b"timestamp;sender;receiver;amount\n2026-01-01T00:00:00Z;a;b;1.00\n"
    ));
}

#[test]
fn samld_namespace_is_content_addressed_and_stable() {
    let a = samld_namespace(SAMLD_SLICE.as_bytes());
    let b = samld_namespace(SAMLD_SLICE.as_bytes());
    assert_eq!(a, b, "same bytes → same namespace (retry idempotency)");
    assert_ne!(
        a,
        samld_namespace(b"different,bytes,here\n"),
        "different bytes → different namespace"
    );
}

#[test]
fn samld_rows_reach_the_mvp_tensor_substrate() {
    let mut svc = IngestionService::new();
    let json =
        ingest_samld_bytes(&mut svc, SAMLD_SLICE.as_bytes()).expect("SAML-D ingestion must commit");
    let parsed = abacus_ingest::api::json::parse(&json).expect("valid JSON reply");

    // 5 data rows: 4 accepted, 1 rejected for excess precision (never rounded).
    assert_eq!(
        parsed.get("reconciles").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        parsed
            .get("commit")
            .and_then(|c| c.get("applied"))
            .and_then(|v| v.as_u64()),
        Some(4)
    );
    let preview = parsed.get("preview").expect("preview");
    assert_eq!(preview.get("input_rows").and_then(|v| v.as_u64()), Some(5));
    assert_eq!(
        preview.get("accepted_rows").and_then(|v| v.as_u64()),
        Some(4)
    );
    assert_eq!(
        preview.get("rejected_rows").and_then(|v| v.as_u64()),
        Some(1)
    );

    // The lone rejection is the excess-precision amount, by source line (line 6).
    let issues = preview.get("issues").and_then(|v| match v {
        abacus_ingest::api::json::Json::Array(a) => Some(a.clone()),
        _ => None,
    });
    let issues = issues.expect("one issue");
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].get("line").and_then(|v| v.as_u64()), Some(6));
    assert_eq!(
        issues[0].get("code").and_then(|v| v.as_str()),
        Some("amount_or_currency")
    );

    // Privacy: no raw account strings, no raw amounts, no ground-truth labels.
    assert!(!json.contains("8724731955"));
    assert!(!json.contains("1459.15"));
    assert!(!json.contains("Laundering"));
    assert!(!json.contains("\"amounts\""));
}
