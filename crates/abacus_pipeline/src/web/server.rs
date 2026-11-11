//! `web::server` — std-only HTTP request/response primitives shared by the
//! ingestion-backed web surface (`ingest_server`). No third-party crates (§0 H1).
//!
//! The former streaming_ingestion_demo-UI router/scheduler (over `StreamingIngestionDemoEngine` + a `SAMCORE02`
//! artifact) was removed in the CSV/Kafka ingestion cutover. What remains is the
//! transport layer the ingestion surface reuses: a parsed [`Request`], a
//! serializable [`Response`], and a minimal HTTP/1.1 request reader.

use crate::web::json::Json;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpStream;

/// Default transactions per page.
pub const DEFAULT_PAGE_SIZE: usize = 1_000;

/// A parsed HTTP request (method, path, query, body).
pub struct Request {
    /// HTTP method (e.g. `GET`, `POST`).
    pub method: String,
    /// Path without the query string (e.g. `/api/streaming_ingestion_demo/status`).
    pub path: String,
    /// Raw query string (after `?`), if any.
    pub query: String,
    /// Request body bytes.
    pub body: Vec<u8>,
    /// True when the client sent `Connection: close` (close after responding).
    pub connection_close: bool,
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

    /// Parse a positive integer query parameter, falling back to `default`.
    #[must_use]
    pub fn query_usize(&self, key: &str, default: usize) -> usize {
        self.query_param(key)
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(default)
    }
}

/// An HTTP response: status code, content type, body.
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Content-Type header value.
    pub content_type: &'static str,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Response {
    /// JSON response with the given status.
    pub(crate) fn json(status: u16, body: String) -> Self {
        Self { status, content_type: "application/json", body: body.into_bytes() }
    }

    /// 200 JSON.
    pub(crate) fn ok(body: String) -> Self {
        Self::json(200, body)
    }

    /// 400 JSON error.
    pub(crate) fn bad_request(msg: &str) -> Self {
        let mut o = Json::object();
        o.set("error", Json::str(msg));
        Self::json(400, o.to_string())
    }

    /// 404 JSON error.
    pub(crate) fn not_found(msg: &str) -> Self {
        let mut o = Json::object();
        o.set("error", Json::str(msg));
        Self::json(404, o.to_string())
    }

    /// Serialize to raw HTTP/1.1 bytes (with CORS + content-length).
    #[must_use]
    pub fn to_http(&self) -> Vec<u8> {
        let reason = match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
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

/// Minimal percent-decoding for query values (`%20`, `%2F`, `+` -> space).
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

/// Read one HTTP request from a buffered stream. Returns `None` on a clean EOF
/// (client closed before sending anything).
pub(crate) fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    // Consume headers.
    let mut content_length = 0usize;
    let mut connection_close = false;
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
            } else if name.eq_ignore_ascii_case("connection") {
                connection_close = value.trim().eq_ignore_ascii_case("close");
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
    Ok(Some(Request { method, path, query, body, connection_close }))
}
