//! The versioned ingestion HTTP API.
//!
//! Task step 3: "Provide a versioned API using the repository's existing HTTP
//! framework and conventions." The repository's existing convention (see
//! `abacus_pipeline::web::server`) is a **std-only** thread-per-connection
//! HTTP/1.1 server with a hand-rolled request reader and a route table. This
//! module follows it exactly: no web framework, no async runtime, no new
//! transport dependency.
//!
//! Every route is a thin shell: it extracts parameters, calls
//! [`crate::api::service::IngestionService`], and shapes the result as JSON. All
//! validation, identity and ingestion rules live in the service (task step 3).
//!
//! ## Routes (all under `/api/v1`)
//!
//! | Method | Path | Purpose |
//! |--------|------|---------|
//! | GET | `/api/v1/capabilities` | supported versions, widths, source types |
//! | POST | `/api/v1/sessions` | create a session |
//! | GET | `/api/v1/sessions/{id}` | status, fingerprint, progress |
//! | POST | `/api/v1/sessions/{id}/chunks/{seq}` | upload one bounded chunk |
//! | GET | `/api/v1/sessions/{id}/chunks/{seq}` | chunk status |
//! | POST | `/api/v1/sessions/{id}/validate` | validate without applying |
//! | POST | `/api/v1/sessions/{id}/seal` | stop accepting chunks |
//! | POST | `/api/v1/sessions/{id}/commit` | apply to the substrate |
//! | GET | `/api/v1/sessions/{id}/report` | counts, checksums, errors |
//! | POST | `/api/v1/sessions/{id}/cancel` | cancel / clean up |
//!
//! ## Status codes
//!
//! - `200` success.
//! - `400` validation, binary, identity or limits rejection (body carries the
//!   structured error code).
//! - `404` unknown session or chunk.
//! - `409` an illegal session-state transition.
//! - `422` a specification gap (a material decision the specs leave open) —
//!   distinct from a client mistake, because no client change can fix it.
//! - `413` a chunk or session size limit was exceeded.
//!
//! ## Idempotency
//!
//! `PUT`-style semantics on chunk upload: repeating `{seq}` with identical
//! bytes returns `200` with `"status":"already_processed"` and changes nothing;
//! repeating it with different bytes returns `409`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::api::json::Json;
use crate::api::service::{CreateSessionRequest, IngestionService};
use crate::error::ErrorCode;
use crate::namespace::EventNamespace;

/// A parsed HTTP request.
pub struct Request {
    /// HTTP method.
    pub method: String,
    /// Path without the query string.
    pub path: String,
    /// Raw query string after `?`, if any.
    pub query: String,
    /// Request body bytes.
    pub body: Vec<u8>,
}

impl Request {
    /// Query parameter lookup (first match).
    #[must_use]
    pub fn query_param(&self, key: &str) -> Option<String> {
        for pair in self.query.split('&') {
            let mut it = pair.splitn(2, '=');
            if it.next() == Some(key) {
                return it.next().map(url_decode);
            }
        }
        None
    }
}

/// An HTTP response.
pub struct Response {
    /// Status code.
    pub status: u16,
    /// Content type.
    pub content_type: &'static str,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Response {
    /// JSON response with a status.
    fn json(status: u16, body: String) -> Self {
        Self { status, content_type: "application/json", body: body.into_bytes() }
    }

    /// 200 JSON.
    fn ok(body: String) -> Self {
        Self::json(200, body)
    }

    /// A structured error response. Carries the stable error code so a GUI can
    /// branch on it, and a structural detail that never contains raw account
    /// identifiers or amounts (Golden Source §16).
    fn error(status: u16, code: ErrorCode, detail: &str) -> Self {
        let mut o = Json::object();
        o.set("error", Json::object().set("code", Json::str(code.as_str())).set("detail", Json::str(detail)).clone());
        Self::json(status, o.to_json_string())
    }

    /// Map a service error that names a missing resource to `404`.
    ///
    /// The documented status codes are: `404` for an unknown session or chunk,
    /// `409` for an illegal state transition. Both surface from the service as
    /// `ErrorCode::Session`, so the resource-name form of the detail is what
    /// distinguishes them — and it is the service, not the transport, that
    /// produced the distinction.
    pub fn from_missing_resource(e: &crate::error::IngestError) -> Self {
        let missing = e.detail.starts_with("unknown session")
            || e.detail.contains("never submitted");
        if missing {
            return Self::error(404, e.code, &e.detail);
        }
        Self::from_error(e)
    }

    /// Map an [`crate::error::IngestError`] to its status code.
    pub fn from_error(e: &crate::error::IngestError) -> Self {
        let status = match e.code {
            ErrorCode::Session => 409,
            ErrorCode::SpecificationGap => 422,
            ErrorCode::Identity | ErrorCode::Binary | ErrorCode::SchemaVersion
            | ErrorCode::Timestamp | ErrorCode::AmountOrCurrency | ErrorCode::AccountId
            | ErrorCode::AccountIdCapacity | ErrorCode::FeatureOverflow
            | ErrorCode::StreamingIngestionDemoPolicy | ErrorCode::SubstrateInvariant
            | ErrorCode::FileOrCsvParse | ErrorCode::ColumnMapping => 400,
            ErrorCode::Persistence => 500,
        };
        Self::error(status, e.code, &e.detail)
    }

    /// Serialize to raw HTTP/1.1 bytes.
    #[must_use]
    pub fn to_http(&self) -> Vec<u8> {
        let reason = match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            409 => "Conflict",
            413 => "Payload Too Large",
            422 => "Unprocessable Content",
            500 => "Internal Server Error",
            _ => "Error",
        };
        let mut head = String::new();
        head.push_str(&format!("HTTP/1.1 {} {}\r\n", self.status, reason));
        head.push_str(&format!("Content-Type: {}\r\n", self.content_type));
        head.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
        head.push_str("Access-Control-Allow-Origin: *\r\n");
        head.push_str("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
        head.push_str("Access-Control-Allow-Headers: Content-Type\r\n");
        head.push_str("Connection: close\r\n\r\n");
        let mut out = head.into_bytes();
        out.extend_from_slice(&self.body);
        out
    }
}

/// Minimal percent-decoding for query values.
fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(h), Some(l)) = (hi, lo) {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Read one HTTP request from a buffered stream. `None` on a clean EOF.
///
/// Public so the browser shell's single-origin `shell_server` can serve its
/// static assets and delegate every `/api/v1` request to the same [`route`]
/// handler over the same [`IngestionService`] — one reader, one route table,
/// one ingestion path.
pub fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some(Request { method, path, query, body }))
}

/// Route one request against the shared service and build a response.
///
/// Pure with respect to I/O: no socket handling, so it is directly unit
/// testable. Thin by construction — every rule is in the service.
pub fn route(service: &Mutex<IngestionService>, req: &mut Request) -> Response {
    if req.method == "OPTIONS" {
        return Response::ok(String::new());
    }
    let path = req.path.trim_end_matches('/');
    let path = if path.is_empty() { "/" } else { path };
    let segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    // ---- Capabilities ----
    if path == "/api/v1/capabilities" && req.method == "GET" {
        let svc = service.lock().unwrap();
        let c = svc.capabilities();
        let mut o = Json::object();
        o.set(
            "wire_format_versions",
            Json::Array(c.wire_format_versions.iter().map(|&v| Json::uint(u64::from(v))).collect()),
        );
        o.set(
            "source_types",
            Json::Array(c.source_types.iter().map(|s| Json::str(s.as_str())).collect()),
        );
        o.set(
            "account_id_widths",
            Json::Array(c.account_id_widths.iter().map(|&w| Json::uint(u64::from(w))).collect()),
        );
        o.set(
            "feature_schema_versions",
            Json::Array(c.feature_schema_versions.iter().map(|&v| Json::uint(u64::from(v))).collect()),
        );
        o.set(
            "feature_algorithm_versions",
            Json::Array(
                c.feature_algorithm_versions.iter().map(|&v| Json::uint(u64::from(v))).collect(),
            ),
        );
        o.set(
            "identity_schemes",
            Json::Array(c.identity_schemes.iter().map(|s| Json::str(s.as_str())).collect()),
        );
        o.set("header_length", Json::uint(c.header_length as u64));
        o.set(
            "supports_validate_without_apply",
            Json::bool(c.supports_validate_without_apply),
        );
        return Response::ok(o.to_json_string());
    }

    // ---- Session creation ----
    if path == "/api/v1/sessions" && req.method == "POST" {
        let body = match crate::api::json::parse(&String::from_utf8_lossy(&req.body)) {
            Ok(v) => v,
            Err(e) => {
                return Response::error(400, ErrorCode::FileOrCsvParse, &format!("bad JSON body: {e}"))
            }
        };
        let source_type = match body.get("source_type").and_then(Json::as_str) {
            Some("browser_csv") => crate::config::SourceType::BrowserCsv,
            Some("samld_fixture") => crate::config::SourceType::SamldFixture,
            Some("kafka") => crate::config::SourceType::Kafka,
            other => {
                return Response::error(
                    400,
                    ErrorCode::ColumnMapping,
                    &format!("missing or unsupported source_type {other:?}"),
                )
            }
        };
        let identity_scheme = match body.get("identity_scheme").and_then(Json::as_str) {
            Some("csv_record_ordinal") => crate::config::IdentityScheme::CsvRecordOrdinal,
            Some("samld_record_ordinal") => crate::config::IdentityScheme::SamldRecordOrdinal,
            Some("kafka_partition_offset") => crate::config::IdentityScheme::KafkaPartitionOffset,
            other => {
                return Response::error(
                    400,
                    ErrorCode::ColumnMapping,
                    &format!("missing or unsupported identity_scheme {other:?}"),
                )
            }
        };
        let u = |k: &str| body.get(k).and_then(Json::as_u64);
        let namespace = match body.get("event_namespace").and_then(Json::as_str) {
            Some(s) => match EventNamespace::parse_hyphenated(s) {
                Ok(ns) => Some(ns),
                Err(e) => return Response::from_error(&e),
            },
            None => None,
        };
        let req_obj = CreateSessionRequest {
            source_type,
            identity_scheme,
            account_id_width: u("account_id_width").unwrap_or(0) as u8,
            currency_code: body
                .get("currency_code")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string(),
            currency_precision: u("currency_precision").unwrap_or(0) as u8,
            threshold_minor_units: u("threshold_minor_units").unwrap_or(0),
            threshold_effective_day: body
                .get("threshold_effective_day")
                .and_then(Json::as_i64)
                .unwrap_or(0) as i32,
            threshold_config_version: u("threshold_config_version").unwrap_or(0) as u32,
            parsing_policy_version: u("parsing_policy_version").unwrap_or(1) as u32,
            max_amount_minor_units: u("max_amount_minor_units").unwrap_or(0),
            namespace,
        };
        let mut svc = service.lock().unwrap();
        return match svc.create_session(&req_obj) {
            Ok(r) => {
                let mut o = Json::object();
                o.set("session_id", Json::str(&r.session_id));
                o.set("event_namespace", Json::str(&r.event_namespace));
                o.set("config_fingerprint", Json::str(&r.config_fingerprint));
                o.set("substrate_value_policy", Json::str(&r.substrate_value_policy));
                o.set("state", Json::str(r.state.as_str()));
                Response::ok(o.to_json_string())
            }
            Err(e) => Response::from_error(&e),
        };
    }

    // Everything below needs /sessions/{id}[/...].
    // NOTE: the bound must be `>= 4` — `segs[3]` on a bare `/api/v1/sessions`
    // (3 segments, e.g. an unsupported GET on the collection) would panic.
    if segs.len() >= 4 && segs[0] == "api" && segs[1] == "v1" && segs[2] == "sessions" {
        let id = segs[3];

        // GET /sessions/{id}
        if segs.len() == 4 && req.method == "GET" {
            let svc = service.lock().unwrap();
            return match svc.session_status(id) {
                Some(r) => Response::ok(r.to_json().to_json_string()),
                None => Response::error(404, ErrorCode::Session, "unknown session"),
            };
        }

        // POST /sessions/{id}/cancel
        if segs.len() == 5 && segs[4] == "cancel" && req.method == "POST" {
            let mut svc = service.lock().unwrap();
            return match svc.cancel(id) {
                Ok(()) => {
                    let mut o = Json::object();
                    o.set("session_id", Json::str(id));
                    o.set("state", Json::str("cancelled"));
                    Response::ok(o.to_json_string())
                }
                Err(e) => Response::from_missing_resource(&e),
            };
        }

        // POST /sessions/{id}/validate
        if segs.len() == 5 && segs[4] == "validate" && req.method == "POST" {
            let svc = service.lock().unwrap();
            return match svc.validate(id) {
                Ok(()) => {
                    let mut o = Json::object();
                    o.set("session_id", Json::str(id));
                    o.set("valid", Json::bool(true));
                    o.set("applied", Json::bool(false));
                    Response::ok(o.to_json_string())
                }
                Err(e) => Response::from_missing_resource(&e),
            };
        }

        // POST /sessions/{id}/seal
        if segs.len() == 5 && segs[4] == "seal" && req.method == "POST" {
            let mut svc = service.lock().unwrap();
            return match svc.seal(id) {
                Ok(()) => {
                    let mut o = Json::object();
                    o.set("session_id", Json::str(id));
                    o.set("state", Json::str("sealed"));
                    Response::ok(o.to_json_string())
                }
                Err(e) => Response::from_missing_resource(&e),
            };
        }

        // POST /sessions/{id}/commit
        if segs.len() == 5 && segs[4] == "commit" && req.method == "POST" {
            let mut svc = service.lock().unwrap();
            // Amendment A1: the exact amount now travels in the commit-time
            // attestation channel for EVERY source type, indexed by record
            // ordinal; it is never part of the frozen ABACSEG1 record. The
            // session pre-flight verifies each amount against the record's
            // frozen features before any substrate mutation.
            //
            // Amendment A1 clearing obligation: the parsed amount vector and
            // the request body (which carried the raw amounts) are held in
            // RAII guards so both are wiped on EVERY route exit — success,
            // validation failure, or a panic unwind — as soon as the
            // transformation into substrate values has run. Clearing is
            // logical, not cryptographic (§16.3); no log, response or cached
            // buffer retains an amount.
            let amounts = crate::secret::SecretOptVec::new(
                crate::api::json::parse(&String::from_utf8_lossy(&req.body))
                    .ok()
                    .and_then(|b| b.get("amounts").cloned())
                    .map(|a| match a {
                        Json::Array(v) => v.iter().map(|x| x.as_i64()).collect(),
                        _ => Vec::new(),
                    })
                    .unwrap_or_default(),
            );
            let body_guard = crate::secret::ZeroizingBytes::new(&mut req.body);
            let res = svc.commit(id, |i, _rec| amounts.get(i));
            // Both guards drop here, at the end of the route, on every path.
            drop(body_guard);
            drop(amounts);
            return match res {
                Ok(c) => {
                    let mut o = Json::object();
                    o.set("session_id", Json::str(id));
                    o.set("state", Json::str("committed"));
                    o.set("applied", Json::uint(c.applied));
                    o.set("already_applied", Json::uint(c.already_applied));
                    Response::ok(o.to_json_string())
                }
                Err(e) => Response::from_missing_resource(&e),
            };
        }

        // GET /sessions/{id}/report
        if segs.len() == 5 && segs[4] == "report" && req.method == "GET" {
            let svc = service.lock().unwrap();
            return match svc.report(id) {
                Ok(r) => Response::ok(r.to_json().to_json_string()),
                Err(e) => Response::from_missing_resource(&e),
            };
        }

        // /sessions/{id}/chunks/{seq}
        if segs.len() == 6 && segs[4] == "chunks" {
            let seq = match segs[5].parse::<u64>() {
                Ok(s) => s,
                Err(_) => {
                    return Response::error(
                        400,
                        ErrorCode::Session,
                        "chunk sequence must be a non-negative integer",
                    )
                }
            };
            if req.method == "GET" {
                let svc = service.lock().unwrap();
                return match svc.chunk_status(id, seq) {
                    Ok(st) => {
                        let mut o = Json::object();
                        o.set("session_id", Json::str(id));
                        o.set("sequence", Json::uint(seq));
                        o.set(
                            "status",
                            Json::str(match st {
                                crate::session::ChunkStatus::Accepted => "accepted",
                                crate::session::ChunkStatus::AlreadyProcessed => "already_processed",
                            }),
                        );
                        Response::ok(o.to_json_string())
                    }
                    Err(e) => Response::from_missing_resource(&e),
                };
            }
            if req.method == "POST" {
                let mut svc = service.lock().unwrap();
                return match svc.submit_chunk(id, seq, &req.body) {
                    Ok(o) => {
                        let mut j = Json::object();
                        j.set("session_id", Json::str(id));
                        j.set("sequence", Json::uint(o.sequence));
                        j.set(
                            "status",
                            Json::str(match o.status {
                                crate::session::ChunkStatus::Accepted => "accepted",
                                crate::session::ChunkStatus::AlreadyProcessed => "already_processed",
                            }),
                        );
                        j.set("record_count", Json::uint(o.record_count));
                        j.set("new_events", Json::uint(o.new_events));
                        j.set("duplicate_events", Json::uint(o.duplicate_events));
                        Response::ok(j.to_json_string())
                    }
                    Err(e) => {
                        // A size-limit rejection is 413, not 400 (§13 limits).
                        if e.detail.contains("limit") || e.detail.contains("exceeding") {
                            return Response::error(413, e.code, &e.detail);
                        }
                        Response::from_missing_resource(&e)
                    }
                };
            }
        }
    }

    Response::error(404, ErrorCode::Session, "no such route")
}

/// Serve one accepted socket to completion (test seam).
pub fn handle_one(service: &Arc<Mutex<IngestionService>>, stream: TcpStream) {
    let mut writer = stream.try_clone().expect("clone stream");
    let mut reader = BufReader::new(stream);
    // `Connection: close`: exactly one request per accepted socket, so this is
    // a single match, not a loop over a persistent connection.
    if let Ok(Some(mut req)) = read_request(&mut reader) {
        let resp = route(service, &mut req);
        if writer.write_all(&resp.to_http()).is_ok() {
            let _ = writer.flush();
        }
    }
}

/// Run the ingestion API server. Blocks forever.
///
/// # Errors
/// `Err` if the listener cannot bind.
pub fn serve(service: Arc<Mutex<IngestionService>>, addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let svc = Arc::clone(&service);
                thread::spawn(move || handle_one(&svc, s));
            }
            Err(_) => continue,
        }
    }
    Ok(())
}
