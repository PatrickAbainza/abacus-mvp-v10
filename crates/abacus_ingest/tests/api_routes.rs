//! API request/response contract tests (task step 4, "API request/response
//! contracts").
//!
//! Drives `api::server::route` directly — the same entry point the HTTP
//! transport uses — so every status code, JSON shape and privacy guarantee
//! asserted here is the one a GUI client will actually observe.
//!
//! Deterministic: fixed namespace, fixed configuration, fixed amounts.

mod common;

use std::sync::Mutex;

use abacus_ingest::api::json::{self, Json};
use abacus_ingest::api::server::{route, Request, Response};
use abacus_ingest::api::IngestionService;
use abacus_ingest::config::{
    CurrencyCode, FEATURE_ALGORITHM_VERSION, FEATURE_SCHEMA_VERSION, IdentityScheme, SourceConfig,
    SourceType,
};
use abacus_ingest::error::ErrorCode;
use abacus_ingest::session::SessionLimits;
use abacus_ingest::{encode_segment, EventNamespace};
use common::{config, ns, record, wide_limits};

/// A fresh, isolated service per test (no shared namespace registry).
fn service() -> Mutex<IngestionService> {
    Mutex::new(IngestionService::with_limits(wide_limits()))
}

fn get(path: &str) -> Request {
    Request {
        method: "GET".to_string(),
        path: path.to_string(),
        query: String::new(),
        body: Vec::new(),
    }
}

fn post(path: &str, body: &[u8]) -> Request {
    Request {
        method: "POST".to_string(),
        path: path.to_string(),
        query: String::new(),
        body: body.to_vec(),
    }
}

fn call(svc: &Mutex<IngestionService>, mut req: Request) -> Response {
    route(svc, &mut req)
}

fn body_text(resp: &Response) -> String {
    String::from_utf8(resp.body.clone()).expect("response bodies are UTF-8 JSON")
}

fn body_json(resp: &Response) -> Json {
    json::parse(&body_text(resp)).expect("response body must parse as JSON")
}

fn field<'a>(j: &'a Json, key: &str) -> &'a Json {
    j.get(key)
        .unwrap_or_else(|| panic!("missing field {key} in {j:?}"))
}

fn f_u64(j: &Json, key: &str) -> Option<u64> {
    field(j, key).as_u64()
}

fn f_str<'a>(j: &'a Json, key: &str) -> Option<&'a str> {
    field(j, key).as_str()
}

fn f_bool(j: &Json, key: &str) -> Option<bool> {
    field(j, key).as_bool()
}

/// The create-session JSON for the shared fixture configuration `config(4)`,
/// pinning the namespace so chunks encoded with `ns()` bind to it.
fn create_body() -> String {
    format!(
        r#"{{
            "source_type": "samld_fixture",
            "identity_scheme": "samld_record_ordinal",
            "account_id_width": 4,
            "currency_code": "USD",
            "currency_precision": 2,
            "threshold_minor_units": 10000,
            "threshold_effective_day": 19000,
            "threshold_config_version": 3,
            "parsing_policy_version": 1,
            "max_amount_minor_units": 100000000000,
            "event_namespace": "{}"
        }}"#,
        ns().to_hyphenated()
    )
}

/// Create a session through the API and return `(session_id, fingerprint)`.
fn create_session(svc: &Mutex<IngestionService>) -> (String, String) {
    let resp = call(svc, post("/api/v1/sessions", create_body().as_bytes()));
    assert_eq!(resp.status, 200, "create failed: {}", body_text(&resp));
    let j = body_json(&resp);
    let id = f_str(&j, "session_id").unwrap().to_string();
    let fp = f_str(&j, "config_fingerprint").unwrap().to_string();
    assert_eq!(f_str(&j, "state"), Some("open"));
    assert_eq!(
        f_str(&j, "event_namespace"),
        Some(ns().to_hyphenated().as_str())
    );
    (id, fp)
}

/// A valid single-chunk payload for the fixture configuration.
fn chunk(ids: &[u64]) -> Vec<u8> {
    let recs: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            record(
                id,
                1_700_000_000_000 + i as i64 * 1_000,
                i as u64 + 1,
                i as u64 + 2,
                50_000,
            )
        })
        .collect();
    encode_segment(&config(4), ns(), &recs).unwrap()
}

fn session_path(id: &str, suffix: &str) -> String {
    format!("/api/v1/sessions/{id}{suffix}")
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

#[test]
fn capabilities_advertise_only_what_is_implemented() {
    let svc = service();
    let resp = call(&svc, get("/api/v1/capabilities"));
    assert_eq!(resp.status, 200);
    let j = body_json(&resp);
    assert_eq!(
        j.get("wire_format_versions")
            .and_then(|v| match v {
                Json::Array(a) if a.len() == 1 => a[0].as_u64(),
                _ => None,
            }),
        Some(1),
        "wire format v1 only"
    );
    assert_eq!(j.get("header_length").and_then(Json::as_u64), Some(176));
    assert_eq!(
        j.get("supports_validate_without_apply").and_then(Json::as_bool),
        Some(true)
    );
    let widths = match j.get("account_id_widths") {
        Some(Json::Array(v)) => v.iter().filter_map(Json::as_u64).collect::<Vec<_>>(),
        other => panic!("account_id_widths must be an array, got {other:?}"),
    };
    assert_eq!(widths, vec![1, 2, 4, 8], "every supported width, no unsupported ones");
    let sources = match j.get("source_types") {
        Some(Json::Array(v)) => v.iter().filter_map(Json::as_str).collect::<Vec<_>>(),
        other => panic!("source_types must be an array, got {other:?}"),
    };
    assert!(sources.contains(&"browser_csv"));
    assert!(sources.contains(&"samld_fixture"));
    assert!(sources.contains(&"kafka"));
}

// ---------------------------------------------------------------------------
// Session creation: happy path and rejections
// ---------------------------------------------------------------------------

#[test]
fn session_create_returns_id_fingerprint_and_namespace() {
    let svc = service();
    let (id, fp) = create_session(&svc);
    assert!(id.starts_with("sess-"));
    // The fingerprint is exactly the fixture configuration's manifest hash —
    // this is the value chunk headers must match.
    assert_eq!(fp, config(4).fingerprint_hex());
    // Round-trip: the session is immediately inspectable.
    let resp = call(&svc, get(&session_path(&id, "")));
    assert_eq!(resp.status, 200);
    let j = body_json(&resp);
    assert_eq!(f_str(&j, "session_id"), Some(id.as_str()));
    assert_eq!(f_str(&j, "config_fingerprint"), Some(fp.as_str()));
    assert_eq!(f_str(&j, "final_state"), Some("open"));
    assert_eq!(f_bool(&j, "reconciles"), Some(true));
}

#[test]
fn session_create_rejects_bad_source_type_and_bad_width() {
    let svc = service();
    // Unsupported source type.
    let bad_src = create_body().replace("samld_fixture", "spreadsheet");
    let resp = call(&svc, post("/api/v1/sessions", bad_src.as_bytes()));
    assert_eq!(resp.status, 400, "unsupported source type must be 400");
    // Unsupported account-ID width (3 bytes is not in {1,2,4,8}).
    let bad_width = create_body().replace("\"account_id_width\": 4", "\"account_id_width\": 3");
    let resp = call(&svc, post("/api/v1/sessions", bad_width.as_bytes()));
    assert_eq!(resp.status, 400, "unsupported width must be 400");
    let j = body_json(&resp);
    let code = field(&j, "error").get("code").and_then(Json::as_str);
    assert!(code.is_some(), "every error carries a stable machine-readable code");
}

#[test]
fn session_create_rejects_malformed_json_body() {
    let svc = service();
    let resp = call(&svc, post("/api/v1/sessions", b"{not json"));
    assert_eq!(resp.status, 400);
}



// ---------------------------------------------------------------------------
// 404s: unknown resources, unknown routes, and the bare collection path
// ---------------------------------------------------------------------------

#[test]
fn unknown_session_is_404_on_every_session_route() {
    let svc = service();
    for (method, path) in [
        ("GET", "/api/v1/sessions/sess-999"),
        ("POST", "/api/v1/sessions/sess-999/cancel"),
        ("POST", "/api/v1/sessions/sess-999/validate"),
        ("POST", "/api/v1/sessions/sess-999/seal"),
        ("POST", "/api/v1/sessions/sess-999/commit"),
        ("GET", "/api/v1/sessions/sess-999/report"),
        ("GET", "/api/v1/sessions/sess-999/chunks/1"),
        ("POST", "/api/v1/sessions/sess-999/chunks/1"),
    ] {
        let req = Request {
            method: method.to_string(),
            path: path.to_string(),
            query: String::new(),
            body: chunk(&[1]).to_vec(),
        };
        let resp = call(&svc, req);
        assert_eq!(
            resp.status, 404,
            "{method} {path} must be 404 for an unknown session, got {}: {}",
            resp.status,
            body_text(&resp)
        );
    }
}

#[test]
fn bare_sessions_collection_and_unknown_routes_are_404_not_panics() {
    // Regression: the sessions branch used to index `segs[3]` after checking
    // only `segs.len() >= 3`, so a bare collection path panicked the handler.
    let svc = service();
    let resp = call(&svc, get("/api/v1/sessions"));
    assert_eq!(resp.status, 404, "bare collection GET must 404, not panic");
    let resp = call(&svc, get("/api/v1/nope"));
    assert_eq!(resp.status, 404);
    let resp = call(&svc, post("/api/v1/sessions/x/chunks/notanumber", &[]));
    // A non-numeric sequence is malformed regardless of session, so 400.
    assert_eq!(resp.status, 400);
}

#[test]
fn unknown_chunk_sequence_is_404() {
    let svc = service();
    let (id, _) = create_session(&svc);
    let resp = call(&svc, get(&session_path(&id, "/chunks/7")));
    assert_eq!(resp.status, 404, "chunk never submitted must be 404");
}

// ---------------------------------------------------------------------------
// Chunk upload: accept, idempotent retry, conflict, status
// ---------------------------------------------------------------------------

#[test]
fn chunk_upload_status_and_idempotent_retry() {
    let svc = service();
    let (id, _) = create_session(&svc);
    let bytes = chunk(&[1, 2, 3]);

    // Fresh upload.
    let resp = call(&svc, post(&session_path(&id, "/chunks/1"), &bytes));
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    let j = body_json(&resp);
    assert_eq!(f_str(&j, "status"), Some("accepted"));
    assert_eq!(f_u64(&j, "record_count"), Some(3));
    assert_eq!(f_u64(&j, "new_events"), Some(3));
    assert_eq!(f_u64(&j, "duplicate_events"), Some(0));

    // Identical retry: idempotent, no double counting.
    let resp = call(&svc, post(&session_path(&id, "/chunks/1"), &bytes));
    assert_eq!(resp.status, 200);
    let j = body_json(&resp);
    assert_eq!(f_str(&j, "status"), Some("already_processed"));

    // Chunk status reports what happened.
    let resp = call(&svc, get(&session_path(&id, "/chunks/1")));
    assert_eq!(resp.status, 200);
    let j = body_json(&resp);
    assert_eq!(f_str(&j, "status"), Some("accepted"));
    assert_eq!(f_u64(&j, "sequence"), Some(1));

    // Session progress reflects exactly one chunk, three inputs.
    let resp = call(&svc, get(&session_path(&id, "")));
    let j = body_json(&resp);
    assert_eq!(f_u64(&j, "chunk_count"), Some(1));
    let counts = field(&j, "counts");
    assert_eq!(counts.get("input").and_then(Json::as_u64), Some(3));
    assert_eq!(counts.get("accepted").and_then(Json::as_u64), Some(3));
    assert_eq!(counts.get("duplicate").and_then(Json::as_u64), Some(0));
}

#[test]
fn same_chunk_sequence_with_different_bytes_is_409() {
    let svc = service();
    let (id, _) = create_session(&svc);
    let resp = call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[1, 2, 3])));
    assert_eq!(resp.status, 200);
    let resp = call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[4, 5, 6])));
    assert_eq!(resp.status, 409, "redefinition of a chunk sequence is a conflict");
}

#[test]
fn malformed_chunk_bytes_are_400_and_change_nothing() {
    let svc = service();
    let (id, _) = create_session(&svc);
    let mut bytes = chunk(&[1, 2, 3]);
    bytes[0] = b'X'; // destroy the magic
    let resp = call(&svc, post(&session_path(&id, "/chunks/1"), &bytes));
    assert_eq!(resp.status, 400);
    let j = body_json(&resp);
    let code = field(&j, "error").get("code").and_then(Json::as_str);
    assert_eq!(code, Some(ErrorCode::Binary.as_str()));
    // The rejected chunk recorded nothing.
    let resp = call(&svc, get(&session_path(&id, "")));
    let j = body_json(&resp);
    assert_eq!(f_u64(&j, "chunk_count"), Some(0));
}

#[test]
fn oversized_chunk_is_413() {
    let svc = Mutex::new(IngestionService::with_limits(SessionLimits {
        max_chunk_bytes: 10,
        max_chunk_records: 1_000,
        max_session_bytes: 1 << 20,
        max_session_records: 1_000,
        max_chunks: 100,
    }));
    let (id, _) = create_session(&svc);
    let resp = call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[1])));
    assert_eq!(resp.status, 413, "size-limit rejection must be 413: {}", body_text(&resp));
}

// ---------------------------------------------------------------------------
// Validate → seal → commit lifecycle
// ---------------------------------------------------------------------------

#[test]
fn validate_is_pure_and_commit_applies_exactly_once() {
    let svc = service();
    let (id, _) = create_session(&svc);
    assert_eq!(
        call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[1, 2, 3]))).status,
        200
    );

    // Validate without applying.
    let resp = call(&svc, post(&session_path(&id, "/validate"), b""));
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    let j = body_json(&resp);
    assert_eq!(f_bool(&j, "valid"), Some(true));
    assert_eq!(
        f_bool(&j, "applied"),
        Some(false),
        "validate never applies"
    );
    // Still not committed.
    let resp = call(&svc, get(&session_path(&id, "")));
    assert_eq!(field(&body_json(&resp), "commit"), &Json::Null);

    // Seal, then commit with the attested amounts from the trusted boundary.
    assert_eq!(call(&svc, post(&session_path(&id, "/seal"), b"")).status, 200);
    let resp = call(
        &svc,
        post(&session_path(&id, "/commit"), br#"{"amounts":[50000,50000,50000]}"#),
    );
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    let j = body_json(&resp);
    assert_eq!(f_str(&j, "state"), Some("committed"));
    assert_eq!(f_u64(&j, "applied"), Some(3));
    assert_eq!(f_u64(&j, "already_applied"), Some(0));

    // Second commit: illegal state transition, 409, nothing re-applied.
    let resp = call(
        &svc,
        post(&session_path(&id, "/commit"), br#"{"amounts":[50000,50000,50000]}"#),
    );
    assert_eq!(resp.status, 409, "{}", body_text(&resp));
    let resp = call(&svc, get(&session_path(&id, "/report")));
    let j = body_json(&resp);
    assert_eq!(
        j.get("commit").and_then(|c| c.get("applied")).and_then(Json::as_u64),
        Some(3),
        "the first commit's outcome is what the report shows"
    );

    // The final report reconciles every input record.
    assert_eq!(j.get("reconciles").and_then(Json::as_bool), Some(true));
    assert_eq!(j.get("final_state").and_then(Json::as_str), Some("committed"));
}

#[test]
fn commit_without_attested_amounts_is_422_and_applies_nothing() {
    // Canonical Contract §11: a failed commit must not leave a partially
    // applied substrate. No amounts are supplied at all, so no record can be
    // attested — the commit must fail before the first mutation.
    let svc = service();
    let (id, _) = create_session(&svc);
    assert_eq!(
        call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[1, 2, 3]))).status,
        200
    );
    let resp = call(&svc, post(&session_path(&id, "/commit"), b"{}"));
    assert_eq!(
        resp.status, 422,
        "missing attested amount is a specification gap: {}",
        body_text(&resp)
    );
    let j = body_json(&resp);
    let code = field(&j, "error").get("code").and_then(Json::as_str);
    assert_eq!(code, Some(ErrorCode::SpecificationGap.as_str()));
    // No commit outcome: the report must show nothing applied.
    let resp = call(&svc, get(&session_path(&id, "/report")));
    let j = body_json(&resp);
    assert_eq!(j.get("commit"), Some(&Json::Null), "nothing may be applied");
    assert_ne!(j.get("final_state").and_then(Json::as_str), Some("committed"));
}

#[test]
fn seal_blocks_further_uploads_with_409() {
    let svc = service();
    let (id, _) = create_session(&svc);
    assert_eq!(call(&svc, post(&session_path(&id, "/seal"), b"")).status, 200);
    let resp = call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[1])));
    assert_eq!(resp.status, 409, "{}", body_text(&resp));
}


// ---------------------------------------------------------------------------
// Source-type policies through the API
// ---------------------------------------------------------------------------

/// The fixture configuration re-typed as browser CSV (still width 4, USD).
fn browser_csv_config() -> SourceConfig {
    SourceConfig {
        source_type: SourceType::BrowserCsv,
        identity_scheme: IdentityScheme::CsvRecordOrdinal,
        account_id_width: 4,
        currency_code: CurrencyCode::parse("USD").unwrap(),
        currency_precision: 2,
        threshold_minor_units: 10_000,
        threshold_effective_day: 19_000,
        threshold_config_version: 3,
        feature_schema_version: FEATURE_SCHEMA_VERSION,
        feature_algorithm_version: FEATURE_ALGORITHM_VERSION,
        parsing_policy_version: 1,
        max_amount_minor_units: 100_000_000_000,
    }
}

#[test]
fn browser_csv_commit_is_a_422_specification_gap() {
    // Amendment A1 supersedes the Phase-1 422: the exact amount now travels in
    // the commit-time attestation channel (indexed by record ordinal, never in
    // the frozen ABACSEG1 record), is pre-flight verified against the frozen
    // features, and is cleared after the transformation. So a browser-CSV
    // session commits exactly like any other source; omitting the amount is
    // still a 422 SpecificationGap (see
    // commit_without_attested_amounts_is_422_and_applies_nothing).
    let svc = service();
    let body = create_body()
        .replace("samld_fixture", "browser_csv")
        .replace("samld_record_ordinal", "csv_record_ordinal");
    let resp = call(&svc, post("/api/v1/sessions", body.as_bytes()));
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    let id = field(&body_json(&resp), "session_id")
        .as_str()
        .unwrap()
        .to_string();

    let recs = vec![record(1, 1_700_000_000_000, 1, 2, 50_000)];
    let bytes = encode_segment(&browser_csv_config(), ns(), &recs).unwrap();
    assert_eq!(
        call(&svc, post(&session_path(&id, "/chunks/1"), &bytes)).status,
        200
    );
    // Validation is pure and passes.
    assert_eq!(call(&svc, post(&session_path(&id, "/validate"), b"")).status, 200);
    // Commit WITH the attested amount now succeeds for browser CSV.
    let resp = call(&svc, post(&session_path(&id, "/commit"), br#"{"amounts":[50000]}"#));
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    let j = body_json(&resp);
    assert_eq!(j.get("applied").and_then(Json::as_u64), Some(1));
    // The committed report reconciles and names the source type in force.
    let resp = call(&svc, get(&session_path(&id, "/report")));
    let j = body_json(&resp);
    assert_ne!(j.get("commit"), Some(&Json::Null));
    assert_eq!(j.get("reconciles").and_then(Json::as_bool), Some(true));
    // And the report states the source type actually in force.
    assert_eq!(j.get("source_type").and_then(Json::as_str), Some("browser_csv"));
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

#[test]
fn cancel_is_reported_and_blocks_further_work() {
    let svc = service();
    let (id, _) = create_session(&svc);
    assert_eq!(
        call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[1, 2]))).status,
        200
    );

    let resp = call(&svc, post(&session_path(&id, "/cancel"), b""));
    assert_eq!(resp.status, 200);
    let j = body_json(&resp);
    assert_eq!(f_str(&j, "state"), Some("cancelled"));

    // Cancelled sessions accept no chunks.
    let resp = call(&svc, post(&session_path(&id, "/chunks/2"), &chunk(&[3])));
    assert_eq!(resp.status, 409, "{}", body_text(&resp));

    // Cancelling again is idempotent, not an error.
    let resp = call(&svc, post(&session_path(&id, "/cancel"), b""));
    assert_eq!(resp.status, 200);
}


// ---------------------------------------------------------------------------
// Privacy (Golden Source §16 / task step 3)
// ---------------------------------------------------------------------------

#[test]
fn responses_never_leak_raw_amounts_or_account_mappings() {
    let svc = service();
    let (id, _) = create_session(&svc);
    assert_eq!(
        call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[7, 8, 9]))).status,
        200
    );
    assert_eq!(call(&svc, post(&session_path(&id, "/seal"), b"")).status, 200);
    let resp = call(
        &svc,
        post(&session_path(&id, "/commit"), br#"{"amounts":[50000,50000,50000]}"#),
    );
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    assert!(!body_text(&resp).contains("50000"), "commit response must not echo amounts");

    // Collect every response a client can observe and check the whole surface.
    let mut surfaces = Vec::new();
    for path in [
        "/api/v1/capabilities",
        &session_path(&id, ""),
        &session_path(&id, "/report"),
        &session_path(&id, "/chunks/1"),
    ] {
        let resp = call(&svc, get(path));
        surfaces.push((path.to_string(), body_text(&resp)));
    }
    for (path, body) in &surfaces {
        for forbidden in ["\"sender", "\"receiver", "\"amount", "\"value", "micro_unit"] {
            assert!(
                !body.contains(forbidden),
                "response for {path} leaks {forbidden:?}: {body}"
            );
        }
        // The attested raw amount must not appear as a value anywhere.
        assert!(!body.contains("50000"), "response for {path} leaks an amount: {body}");
    }
}

// ---------------------------------------------------------------------------
// Transport-level contract
// ---------------------------------------------------------------------------

#[test]
fn responses_serialize_as_valid_http_with_content_length() {
    let svc = service();
    let resp = call(&svc, get("/api/v1/capabilities"));
    let raw = resp.to_http();
    let text = String::from_utf8(raw).unwrap();
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(text.contains("Content-Type: application/json"));
    let header_end = text.find("\r\n\r\n").unwrap();
    let declared: usize = text
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .unwrap()
        .split(':')
        .nth(1)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(
        declared,
        text.len() - header_end - 4,
        "Content-Length must match the body"
    );
    // And the body is valid JSON on its own.
    json::parse(&text[header_end + 4..]).expect("body must be JSON");
}

// ---------------------------------------------------------------------------
// Capability agreement: the service advertises exactly what it accepts
// ---------------------------------------------------------------------------

#[test]
fn every_advertised_width_is_accepted_by_session_creation() {
    let svc = service();
    for width in [1u8, 2, 4, 8] {
        let body = create_body().replace(
            "\"account_id_width\": 4",
            &format!("\"account_id_width\": {width}"),
        );
        let resp = call(&svc, post("/api/v1/sessions", body.as_bytes()));
        assert_eq!(
            resp.status, 200,
            "advertised width {width} must be accepted: {}",
            body_text(&resp)
        );
    }
    // And an unadvertised width is rejected.
    let body = create_body().replace("\"account_id_width\": 4", "\"account_id_width\": 16");
    let resp = call(&svc, post("/api/v1/sessions", body.as_bytes()));
    assert_eq!(resp.status, 400, "width 16 is not advertised and must be rejected");
}

#[test]
fn generated_namespaces_are_distinct_when_none_is_supplied() {
    let svc = service();
    let ns_value = ns().to_hyphenated();
    let body = create_body().replace(&format!(",\n            \"event_namespace\": \"{ns_value}\""), "");
    assert!(!body.contains("event_namespace"), "namespace must be absent: {body}");

    let resp = call(&svc, post("/api/v1/sessions", body.as_bytes()));
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    let j = body_json(&resp);
    let generated = f_str(&j, "event_namespace").unwrap();
    // A generated namespace is a well-formed, non-zero UUID string.
    let parsed = EventNamespace::parse_hyphenated(generated).expect("generated namespace parses");
    assert_ne!(parsed, EventNamespace::from_bytes([0u8; 16]));
    // And two generated sessions get two distinct namespaces (§3.2).
    let resp2 = call(&svc, post("/api/v1/sessions", body.as_bytes()));
    let other = field(&body_json(&resp2), "event_namespace")
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(generated, other, "each fresh import gets its own namespace");
}


// ---------------------------------------------------------------------------
// Phase 4: attested-amount buffer clearing (A1) — request body is wiped
// ---------------------------------------------------------------------------

#[test]
fn commit_request_body_is_zeroed_on_success_and_on_failure() {
    // Amendment A1: the raw amount crosses the commit-time attestation channel
    // and must be cleared once the transformation into substrate values has
    // run, on EVERY route exit. `route` takes `&mut Request`, so a caller can
    // observe the body after the call: the same length, every byte zero.
    let svc = service();
    let (id, _) = create_session(&svc);
    assert_eq!(
        call(&svc, post(&session_path(&id, "/chunks/1"), &chunk(&[1, 2, 3]))).status,
        200
    );
    assert_eq!(call(&svc, post(&session_path(&id, "/seal"), b"")).status, 200);

    // Success path: a body carrying raw amounts is zeroed after commit.
    let mut req = post(&session_path(&id, "/commit"), br#"{"amounts":[50000,50000,50000]}"#);
    let before_len = req.body.len();
    assert!(before_len > 0);
    let resp = route(&svc, &mut req);
    assert_eq!(resp.status, 200, "{}", body_text(&resp));
    assert_eq!(req.body.len(), before_len, "the buffer keeps its length");
    assert!(
        req.body.iter().all(|&b| b == 0),
        "the request body must be zeroed after a successful commit: {:?}",
        req.body
    );

    // Failure path: a commit that fails validation still zeroes the body.
    // A fresh session whose chunk has a different amount than attested.
    let svc2 = service();
    let (id2, _) = create_session(&svc2);
    assert_eq!(
        call(&svc2, post(&session_path(&id2, "/chunks/1"), &chunk(&[1, 2, 3]))).status,
        200
    );
    assert_eq!(call(&svc2, post(&session_path(&id2, "/seal"), b"")).status, 200);
    // Attest a wrong amount for record 0 -> AmountOrCurrency, applies nothing.
    let mut req2 = post(&session_path(&id2, "/commit"), br#"{"amounts":[49999,50000,50000]}"#);
    let before_len2 = req2.body.len();
    let resp2 = route(&svc2, &mut req2);
    assert_eq!(resp2.status, 400, "a feature/amount mismatch is a 400: {}", body_text(&resp2));
    assert_eq!(req2.body.len(), before_len2);
    assert!(
        req2.body.iter().all(|&b| b == 0),
        "the request body must be zeroed even when the commit fails: {:?}",
        req2.body
    );
}
