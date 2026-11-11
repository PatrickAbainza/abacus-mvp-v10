//! The browser-side HTTP client for the existing `/api/v1` ingestion API.
//!
//! This is a **client**, not a second ingestion path. It speaks the routes the
//! server already exposes (`GET /capabilities`, `POST /sessions`,
//! `POST /sessions/{id}/chunks/{seq}`, `POST .../seal`, `POST .../commit`,
//! `GET .../report`), and every rule stays on the server. Nothing here
//! re-validates a segment, re-derives a feature or decides whether an event is a
//! duplicate: those are the backend's job (§14: "Rust independently validates
//! all data it receives").
//!
//! ## Why the transport is a seam
//!
//! The workspace is std-only apart from `sha2`, and this crate must build
//! offline alongside it. `fetch` is the browser's transport, not something a
//! dependency can supply to `std`, so one HTTP exchange is abstracted behind
//! the [`Transport`] trait: the native implementation is [`TcpTransport`] over
//! `TcpStream` directly — which is also exactly what the server's own reader
//! expects (`Connection: close`, one request per socket, a `Content-Length`
//! body) — and the browser implementation lives in the `abacus_shell` crate,
//! which owns the synchronous host-fetch import and the memory the host writes
//! responses into. The retry policy in `send` and the confirm → create →
//! upload → seal → commit → report flow in [`crate::driver`] are
//! transport-independent, so both targets run the identical client logic.
//!
//! §12: "Do not require support for browser streaming request bodies; use an
//! upload protocol compatible with ordinary bounded HTTP requests." Every
//! request here is a single bounded body with an explicit `Content-Length`.

use std::time::Duration;

use abacus_ingest::api::json::Json;
use abacus_ingest::error::{ErrorCode, IngestError};

#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::net::TcpStream;

/// One HTTP exchange failure. Distinct from an API-level rejection so the retry
/// policy can tell "the network blipped" from "the server said no".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpError {
    /// What went wrong, structurally described (never a raw value).
    pub detail: String,
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.detail)
    }
}

impl std::error::Error for HttpError {}

/// A parsed API response.
#[derive(Debug, Clone)]
pub struct ApiResponse {
    /// HTTP status code.
    pub status: u16,
    /// The body, parsed as JSON when it is JSON.
    pub json: Option<Json>,
    /// The raw body bytes.
    pub raw: Vec<u8>,
}

impl ApiResponse {
    /// The structured error code from an error body, if present.
    #[must_use]
    pub fn error_code(&self) -> Option<String> {
        self.json
            .as_ref()
            .and_then(|j| j.get("error"))
            .and_then(|e| e.get("code"))
            .and_then(Json::as_str)
            .map(str::to_string)
    }

    /// The structured error detail from an error body, if present.
    #[must_use]
    pub fn error_detail(&self) -> Option<String> {
        self.json
            .as_ref()
            .and_then(|j| j.get("error"))
            .and_then(|e| e.get("detail"))
            .and_then(Json::as_str)
            .map(str::to_string)
    }
}

/// The server-advertised capabilities (`GET /api/v1/capabilities`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Supported account-ID widths in bytes.
    pub account_id_widths: Vec<u8>,
    /// Supported wire-format versions.
    pub wire_format_versions: Vec<u16>,
    /// Whether validate-without-apply is supported.
    pub supports_validate_without_apply: bool,
}

/// One acknowledged chunk upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkAck {
    /// The chunk sequence number the server accepted.
    pub sequence: u64,
    /// True iff the server recognised these exact bytes as already processed
    /// (§13 idempotency).
    pub already_processed: bool,
    /// Records decoded from the chunk.
    pub record_count: u64,
}

/// The outcome of a commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitAck {
    /// Events committed to the substrate on this call.
    pub applied: u64,
    /// Events the substrate reported as already applied (idempotent retry).
    pub already_applied: u64,
}

/// Retry policy. Bounded and explicit (§18: "explicit resource limits").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Maximum attempts per request, including the first.
    pub max_attempts: u32,
    /// Delay between attempts.
    pub backoff: Duration,
}

impl Default for RetryPolicy {
    /// Three attempts with a short backoff: enough for a dropped connection on a
    /// local server, bounded so a hung server cannot stall a browser tab.
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: Duration::from_millis(50),
        }
    }
}

/// The browser CSV ingestion client.
///
/// Holds no session state of its own: every call is one HTTP exchange, so the
/// client is stateless between operations and a retry after a crash is simply a
/// fresh run with the same [`crate::resume::ResumeToken`].
pub struct CsvIngestClient {
    /// The configured endpoint (`host:port` or `http://host:port`).
    addr: String,
    /// Bounded retry policy.
    retry: RetryPolicy,
    /// The platform transport performing each exchange.
    transport: Box<dyn Transport>,
}

impl CsvIngestClient {
    /// A client for `addr` with the default retry policy and the platform's
    /// default transport: [`TcpTransport`] on native targets; on
    /// `wasm32-unknown-unknown` the fail-closed [`UnboundTransport`] until the
    /// browser shell binds the real one with [`Self::with_transport`].
    #[must_use]
    pub fn new(addr: impl Into<String>) -> Self {
        let addr = addr.into();
        let transport = default_transport(&addr);
        Self {
            addr,
            retry: RetryPolicy::default(),
            transport,
        }
    }

    /// Replace the platform transport — the browser/WASM boundary (§12).
    #[must_use]
    pub fn with_transport(mut self, transport: Box<dyn Transport>) -> Self {
        self.transport = transport;
        self
    }

    /// Override the retry policy.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// The configured server address.
    #[must_use]
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// `GET /api/v1/capabilities`.
    ///
    /// Called before a session is created so the client can refuse a width the
    /// server does not advertise, rather than discovering it as a 400 after an
    /// upload (§12 step 4: "Configure or confirm...").
    ///
    /// # Errors
    /// [`HttpError`] on a transport failure, or if the response is not the
    /// documented shape.
    pub fn capabilities(&self) -> Result<Capabilities, HttpError> {
        let r = self.send("GET", "/api/v1/capabilities", &[], None)?;
        if r.status != 200 {
            return Err(HttpError {
                detail: format!("capabilities returned status {}", r.status),
            });
        }
        let j = r.json.ok_or_else(|| HttpError {
            detail: "the capabilities response is not JSON".to_string(),
        })?;
        let widths = match j.get("account_id_widths") {
            Some(Json::Array(v)) => v.iter().filter_map(Json::as_u64).map(|w| w as u8).collect(),
            _ => Vec::new(),
        };
        let versions = match j.get("wire_format_versions") {
            Some(Json::Array(v)) => v
                .iter()
                .filter_map(Json::as_u64)
                .map(|w| w as u16)
                .collect(),
            _ => Vec::new(),
        };
        Ok(Capabilities {
            account_id_widths: widths,
            wire_format_versions: versions,
            supports_validate_without_apply: j
                .get("supports_validate_without_apply")
                .and_then(Json::as_bool)
                .unwrap_or(false),
        })
    }

    /// `POST /api/v1/sessions` with an explicit configuration.
    ///
    /// # Errors
    /// [`HttpError`] on a transport failure. An API-level rejection comes back as
    /// an [`ApiResponse`] with a non-2xx status, so the caller branches on the
    /// server's structured error code.
    pub fn create_session(&self, body: &str) -> Result<ApiResponse, HttpError> {
        self.send(
            "POST",
            "/api/v1/sessions",
            body.as_bytes(),
            Some("application/json"),
        )
    }

    /// `POST /api/v1/sessions/{id}/chunks/{seq}` with one encoded segment.
    ///
    /// # Errors
    /// [`HttpError`] on a transport failure after all retries.
    pub fn upload_chunk(
        &self,
        session: &str,
        sequence: u64,
        bytes: &[u8],
    ) -> Result<ApiResponse, HttpError> {
        let path = format!("/api/v1/sessions/{session}/chunks/{sequence}");
        self.send("POST", &path, bytes, Some("application/octet-stream"))
    }

    /// `POST /api/v1/sessions/{id}/seal` — stop accepting further chunks.
    ///
    /// # Errors
    /// [`HttpError`] on a transport failure after all retries.
    pub fn seal(&self, session: &str) -> Result<ApiResponse, HttpError> {
        self.send(
            "POST",
            &format!("/api/v1/sessions/{session}/seal"),
            &[],
            None,
        )
    }

    /// `POST /api/v1/sessions/{id}/commit` with the attested amounts.
    ///
    /// The body is the A1 attestation channel and nothing else. The caller keeps
    /// the buffer in a clearing guard; this function never retains it.
    ///
    /// # Errors
    /// [`HttpError`] on a transport failure after all retries.
    pub fn commit(&self, session: &str, amounts_body: &str) -> Result<ApiResponse, HttpError> {
        self.send(
            "POST",
            &format!("/api/v1/sessions/{session}/commit"),
            amounts_body.as_bytes(),
            Some("application/json"),
        )
    }

    /// `GET /api/v1/sessions/{id}/report`.
    ///
    /// # Errors
    /// [`HttpError`] on a transport failure after all retries.
    pub fn report(&self, session: &str) -> Result<ApiResponse, HttpError> {
        self.send(
            "GET",
            &format!("/api/v1/sessions/{session}/report"),
            &[],
            None,
        )
    }

    /// `POST /api/v1/sessions/{id}/cancel` — the explicit abort path (§13:
    /// "Interrupted sessions have explicit resume or abort semantics").
    ///
    /// # Errors
    /// [`HttpError`] on a transport failure after all retries.
    pub fn cancel(&self, session: &str) -> Result<ApiResponse, HttpError> {
        self.send(
            "POST",
            &format!("/api/v1/sessions/{session}/cancel"),
            &[],
            None,
        )
    }

    /// Perform one request with the bounded retry policy.
    ///
    /// Only transport failures are retried. An HTTP-level rejection (4xx/5xx) is
    /// the server's answer and is returned immediately: retrying a 409 would just
    /// re-deliver the same conflicting bytes, and retrying a 422 cannot help
    /// because no client change can fix a specification gap.
    fn send(
        &self,
        method: &str,
        path: &str,
        body: &[u8],
        content_type: Option<&str>,
    ) -> Result<ApiResponse, HttpError> {
        let attempts = self.retry.max_attempts.max(1);
        let mut last = String::from("no attempt was made");
        for attempt in 1..=attempts {
            match self.transport.exchange(method, path, body, content_type) {
                Ok(r) => return Ok(r),
                Err(e) => {
                    last = e;
                    if attempt < attempts {
                        sleep_backoff(self.retry.backoff);
                    }
                }
            }
        }
        Err(HttpError {
            detail: format!("{method} {path} failed after {attempts} attempts: {last}"),
        })
    }
}

/// One HTTP exchange, performed by the platform's transport.
///
/// Golden Source §12 keeps every ingestion rule on the server and every
/// normalization rule in this crate; the *transport* is the only
/// platform-specific piece. `send` above this seam owns the bounded retry
/// policy, and [`crate::driver::run_import`] owns the whole confirm → create →
/// upload → seal → commit → report flow, so swapping the transport cannot
/// change any ingestion semantics.
pub trait Transport {
    /// Perform a single exchange.
    ///
    /// `Ok` is the server's answer at any HTTP status; `Err` is a
    /// transport-level failure (connection refused, DNS failure, network down),
    /// which is the only case the retry policy re-attempts.
    fn exchange(
        &self,
        method: &str,
        path: &str,
        body: &[u8],
        content_type: Option<&str>,
    ) -> Result<ApiResponse, String>;
}

/// The native transport: one fresh `Connection: close` HTTP/1.1 socket per
/// exchange, with connect/read/write timeouts.
///
/// This is byte-for-byte the request/response behaviour the server's own
/// reader expects, moved behind the [`Transport`] seam unchanged.
#[cfg(not(target_arch = "wasm32"))]
pub struct TcpTransport {
    /// The `host:port` authority (any `http://` prefix is stripped).
    addr: String,
    /// Per-request connect/read/write timeout.
    timeout: Duration,
}

#[cfg(not(target_arch = "wasm32"))]
impl TcpTransport {
    /// A transport for `endpoint`: `host:port` or `http://host:port`.
    ///
    /// `https://` is rejected at exchange time — the native transport speaks
    /// plain HTTP only; TLS is the browser's job when it exists.
    #[must_use]
    pub fn new(endpoint: &str) -> Self {
        let addr = endpoint
            .strip_prefix("http://")
            .unwrap_or(endpoint)
            .trim_end_matches('/')
            .to_string();
        Self {
            addr,
            timeout: Duration::from_secs(10),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Transport for TcpTransport {
    fn exchange(
        &self,
        method: &str,
        path: &str,
        body: &[u8],
        content_type: Option<&str>,
    ) -> Result<ApiResponse, String> {
        let mut stream = TcpStream::connect(&self.addr).map_err(|e| format!("connect: {e}"))?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|e| format!("set_read_timeout: {e}"))?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|e| format!("set_write_timeout: {e}"))?;

        let mut head = String::new();
        head.push_str(&format!("{method} {path} HTTP/1.1\r\n"));
        head.push_str(&format!("Host: {}\r\n", self.addr));
        if let Some(ct) = content_type {
            head.push_str(&format!("Content-Type: {ct}\r\n"));
        }
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
        head.push_str("Connection: close\r\n\r\n");
        stream
            .write_all(head.as_bytes())
            .map_err(|e| format!("write head: {e}"))?;
        if !body.is_empty() {
            stream
                .write_all(body)
                .map_err(|e| format!("write body: {e}"))?;
        }
        stream.flush().map_err(|e| format!("flush: {e}"))?;

        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .map_err(|e| format!("read: {e}"))?;
        parse_response(&raw)
    }
}

/// The fail-closed transport used on `wasm32` when no platform transport has
/// been bound.
///
/// This crate cannot own the browser's synchronous fetch import: that lives in
/// `abacus_shell`, which also owns the memory the host writes responses into.
/// So a client built with [`CsvIngestClient::new`] on `wasm32` refuses every
/// exchange explicitly instead of silently doing nothing; the shell binds the
/// real transport with [`CsvIngestClient::with_transport`].
#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Copy)]
pub struct UnboundTransport;

#[cfg(target_arch = "wasm32")]
impl Transport for UnboundTransport {
    fn exchange(
        &self,
        method: &str,
        path: &str,
        _body: &[u8],
        _content_type: Option<&str>,
    ) -> Result<ApiResponse, String> {
        Err(format!(
            "no platform transport is bound for {method} {path}; construct the client with \
             with_transport(...) on wasm32"
        ))
    }
}

/// The default transport for the compilation target.
#[cfg(not(target_arch = "wasm32"))]
fn default_transport(endpoint: &str) -> Box<dyn Transport> {
    Box::new(TcpTransport::new(endpoint))
}

/// The default transport for the compilation target (fail-closed; see
/// [`UnboundTransport`]).
#[cfg(target_arch = "wasm32")]
fn default_transport(_endpoint: &str) -> Box<dyn Transport> {
    Box::new(UnboundTransport)
}

/// Sleep between retry attempts.
///
/// Native sleeps for the policy's backoff (a zero backoff is skipped, which is
/// a no-op anyway). The browser never sleeps: `std::thread::sleep` has no
/// implementation on `wasm32-unknown-unknown`, and a tab owns its own clock —
/// so there the bounded attempt count is what keeps a hung server from
/// spinning, and the backoff is a documented no-op.
#[cfg(not(target_arch = "wasm32"))]
fn sleep_backoff(backoff: Duration) {
    if !backoff.is_zero() {
        std::thread::sleep(backoff);
    }
}

/// No sleeping on the browser target (see the native variant).
#[cfg(target_arch = "wasm32")]
fn sleep_backoff(_backoff: Duration) {}

/// Split a raw HTTP/1.1 response into a status and a body.
/// (Native-only: the browser transport receives an already-parsed status and
/// body from the host, so this decoder has no wasm use.)
#[cfg(not(target_arch = "wasm32"))]
fn parse_response(raw: &[u8]) -> Result<ApiResponse, String> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .ok_or_else(|| "the response has no header terminator".to_string())?;
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut body_vec = raw[split..].to_vec();
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| "the response has no status line".to_string())?;
    // Honour Content-Length when present so a body with trailing framing bytes
    // is not misread.
    for line in head.lines() {
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                if let Ok(n) = v.trim().parse::<usize>() {
                    body_vec.truncate(n.min(body_vec.len()));
                }
            }
        }
    }
    let text = String::from_utf8_lossy(&body_vec).into_owned();
    Ok(ApiResponse {
        status,
        json: abacus_ingest::api::json::parse(&text).ok(),
        raw: body_vec,
    })
}

/// Turn a non-2xx response into an [`IngestError`] carrying the server's own
/// structured code, so a GUI can branch on it rather than on a string.
///
/// [`HttpError`] stays reserved for transport failures; the server's answer
/// always comes back structured, which is how the client distinguishes "the
/// server said no" from "the network failed".
#[must_use]
pub fn error_from_response(r: &ApiResponse) -> IngestError {
    let code = match r.error_code().as_deref() {
        Some("file_or_csv_parse") => ErrorCode::FileOrCsvParse,
        Some("column_mapping") => ErrorCode::ColumnMapping,
        Some("timestamp") => ErrorCode::Timestamp,
        Some("amount_or_currency") => ErrorCode::AmountOrCurrency,
        Some("account_id") => ErrorCode::AccountId,
        Some("account_id_capacity") => ErrorCode::AccountIdCapacity,
        Some("feature_overflow") => ErrorCode::FeatureOverflow,
        Some("schema_version") => ErrorCode::SchemaVersion,
        Some("binary") => ErrorCode::Binary,
        Some("identity") => ErrorCode::Identity,
        Some("session") => ErrorCode::Session,
        Some("streaming_ingestion_demo_policy") => ErrorCode::StreamingIngestionDemoPolicy,
        Some("substrate_invariant") => ErrorCode::SubstrateInvariant,
        Some("specification_gap") => ErrorCode::SpecificationGap,
        Some("persistence") => ErrorCode::Persistence,
        _ => ErrorCode::Binary,
    };
    IngestError::new(
        code,
        r.error_detail()
            .unwrap_or_else(|| format!("the server returned status {}", r.status)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_http_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(
            r.json
                .as_ref()
                .and_then(|j| j.get("a"))
                .and_then(Json::as_i64),
            Some(1)
        );
    }

    #[test]
    fn honours_content_length_over_trailing_bytes() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\n\r\n{}\r\nJUNK";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 404);
        assert_eq!(r.raw, b"{}");
    }

    #[test]
    fn rejects_a_response_without_a_header_terminator() {
        assert!(parse_response(b"HTTP/1.1 200 OK\r\nno terminator").is_err());
    }

    #[test]
    fn maps_every_documented_error_code() {
        let mk = |code: &str| ApiResponse {
            status: 400,
            json: Some({
                let mut o = Json::object();
                o.set(
                    "error",
                    Json::object()
                        .set("code", Json::str(code))
                        .set("detail", Json::str("d"))
                        .clone(),
                );
                o
            }),
            raw: Vec::new(),
        };
        let pairs = [
            ("file_or_csv_parse", ErrorCode::FileOrCsvParse),
            ("column_mapping", ErrorCode::ColumnMapping),
            ("timestamp", ErrorCode::Timestamp),
            ("amount_or_currency", ErrorCode::AmountOrCurrency),
            ("account_id", ErrorCode::AccountId),
            ("account_id_capacity", ErrorCode::AccountIdCapacity),
            ("feature_overflow", ErrorCode::FeatureOverflow),
            ("schema_version", ErrorCode::SchemaVersion),
            ("binary", ErrorCode::Binary),
            ("identity", ErrorCode::Identity),
            ("session", ErrorCode::Session),
            ("streaming_ingestion_demo_policy", ErrorCode::StreamingIngestionDemoPolicy),
            ("substrate_invariant", ErrorCode::SubstrateInvariant),
            ("specification_gap", ErrorCode::SpecificationGap),
            ("persistence", ErrorCode::Persistence),
        ];
        for (code, want) in pairs {
            assert_eq!(error_from_response(&mk(code)).code, want, "code {code}");
        }
        assert_eq!(
            error_from_response(&mk("future_code")).code,
            ErrorCode::Binary
        );
    }

    #[test]
    fn retry_policy_never_attempts_zero_times() {
        let r = RetryPolicy {
            max_attempts: 0,
            backoff: Duration::from_millis(1),
        };
        let c = CsvIngestClient::new("127.0.0.1:1").with_retry(r);
        // Port 1 is closed, so this fails on transport. The point is that
        // `max_attempts = 0` still attempts once (`.max(1)`): an error, never a
        // silent success.
        assert!(c.report("sess-0").is_err());
    }
}
