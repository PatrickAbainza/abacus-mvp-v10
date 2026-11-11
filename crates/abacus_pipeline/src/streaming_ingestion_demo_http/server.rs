//! `web::server` — std-only HTTP/1.1 server + request routing for the streaming_ingestion_demo
//! UI (your_task_phase_4.md §8). No third-party crates (§0 H1): a small
//! thread-per-connection `TcpListener` loop with a hand-rolled HTTP request
//! reader and a route table over the shared [`StreamingIngestionDemoEngine`].
//!
//! Endpoints (task §8, adapted to the current backend — no redundant routes):
//! - `POST /api/streaming_ingestion_demo/start`   body `{}` (no body needed)
//! - `POST /api/streaming_ingestion_demo/pause`
//! - `POST /api/streaming_ingestion_demo/resume`
//! - `POST /api/streaming_ingestion_demo/reset`
//! - `GET  /api/streaming_ingestion_demo/status`
//! - `GET  /api/transactions?page=1&page_size=1000`
//! - `GET  /api/transactions/{id}`
//! - `GET  /api/transactions/{id}/network/transactions`
//! - `GET  /api/transactions/{id}/network/accounts`
//!
//! Concurrency model: one `Arc<Mutex<StreamingIngestionDemoEngine>>` shared by a scheduler
//! thread (drives `apply_one` one record at a time, as fast as possible — no
//! rate limiting) and every connection thread. The engine's `Pipeline` is
//! single-owner under that mutex, matching the repository's serialized-writer
//! model (`state.rs`); a handler holds the lock only for the duration of one
//! query, so pagination stays responsive while ingestion continues (task §3C).
//! The scheduler applies exactly one record per tick — no read-batch-then-
//! apply-batch step, no per-chunk grouping, no target throughput — pure
//! incremental streaming at maximum speed (task §2B).

use crate::streaming_ingestion_demo::{StreamingIngestionDemoEngine, StreamingIngestionDemoPhase};
use crate::streaming_ingestion_demo_http::graph::{account_network, tensor_network, transaction_network, Network};
use crate::streaming_ingestion_demo_http::json::Json;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

/// Default transactions per page (task §3C: 1,000 by default).
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


/// Serialize a [`Network`] to JSON (nodes + edges + truncated). NO amounts.
fn network_json(n: &Network) -> Json {
    let nodes = Json::Array(
        n.nodes
            .iter()
            .map(|node| {
                let mut o = Json::object();
                o.set("id", Json::str(&node.id));
                o.set("kind", Json::str(node.kind));
                o.set("label", Json::str(&node.label));
                if let Some(t) = node.t {
                    o.set("t", Json::uint(t));
                }
                o.set("selected", Json::bool(node.selected));
                if let Some(h) = node.hop {
                    o.set("hop", Json::int(h as i64));
                }
                o
            })
            .collect(),
    );
    let edges = Json::Array(
        n.edges
            .iter()
            .map(|e| {
                let mut o = Json::object();
                o.set("from", Json::str(&e.from));
                o.set("to", Json::str(&e.to));
                o.set(
                    "txs",
                    Json::Array(e.txs.iter().map(|&t| Json::uint(t)).collect()),
                );
                o
            })
            .collect(),
    );
    let mut root = Json::object();
    root.set("nodes", nodes);
    root.set("edges", edges);
    root.set("truncated", Json::bool(n.truncated));
    root.set("committed", Json::uint(n.committed));
    root
}

/// Route one request against the shared engine and build a response.
///
/// Pure w.r.t. the engine's public API — no socket concerns — so it is unit
/// testable. All state transitions go through `StreamingIngestionDemoEngine` (never the
/// pipeline directly).
fn route(engine: &Mutex<StreamingIngestionDemoEngine>, req: &Request) -> Response {
    // CORS preflight.
    if req.method == "OPTIONS" {
        return Response::ok(String::new());
    }
    let path = req.path.trim_end_matches('/').to_string();
    let path = if path.is_empty() { "/".to_string() } else { path };

    // ---- StreamingIngestionDemo control + status ----
    if path == "/api/streaming_ingestion_demo/status" {
        let eng = engine.lock().unwrap();
        return Response::ok(status_json(&eng).to_string());
    }
    if path == "/api/streaming_ingestion_demo/start" && req.method == "POST" {
        let mut eng = engine.lock().unwrap();
        return match eng.start() {
            Ok(()) => Response::ok(status_json(&eng).to_string()),
            Err(e) => Response::bad_request(&e),
        };
    }
    if path == "/api/streaming_ingestion_demo/pause" && req.method == "POST" {
        let mut eng = engine.lock().unwrap();
        eng.pause();
        return Response::ok(status_json(&eng).to_string());
    }
    if path == "/api/streaming_ingestion_demo/resume" && req.method == "POST" {
        let mut eng = engine.lock().unwrap();
        eng.resume();
        return Response::ok(status_json(&eng).to_string());
    }
    if path == "/api/streaming_ingestion_demo/reset" && req.method == "POST" {
        let mut eng = engine.lock().unwrap();
        eng.reset();
        return Response::ok(status_json(&eng).to_string());
    }

    // ---- Transactions ----
    if path == "/api/transactions" && req.method == "GET" {
        let page = req.query_usize("page", 1);
        let page_size = req.query_usize("page_size", DEFAULT_PAGE_SIZE);
        let eng = engine.lock().unwrap();
        return Response::ok(transactions_page(&eng, page, page_size).to_string());
    }
    if let Some(rest) = path.strip_prefix("/api/transactions/") {
        // rest is `{id}` or `{id}/network/transactions` or `{id}/network/accounts`.
        let mut seg = rest.split('/');
        let id_str = seg.next().unwrap_or("");
        let Ok(id) = id_str.parse::<u64>() else {
            return Response::bad_request("invalid transaction id");
        };
        let eng = engine.lock().unwrap();
        match (seg.next(), seg.next()) {
            (None, _) => transaction_detail(&eng, id),
            (Some("network"), Some("transactions")) => {
                match transaction_network(eng.pipeline(), id) {
                    Some(n) => Response::ok(network_json(&n).to_string()),
                    None => Response::not_found("transaction not found"),
                }
            }
            (Some("network"), Some("accounts")) => match account_network(eng.pipeline(), id) {
                Some(n) => Response::ok(network_json(&n).to_string()),
                None => Response::not_found("transaction not found"),
            }
            (Some("network"), Some("tensor")) => {
                let k = req.query_usize("k", 3).clamp(1, 8);
                match tensor_network(eng.pipeline(), id, k as u8) {
                    Some(n) => Response::ok(network_json(&n).to_string()),
                    None => Response::not_found("transaction not found"),
                }
            }
            _ => Response::not_found("unknown transactions sub-resource"),
        }
    } else {
        Response::not_found("not found")
    }
}


/// Build the `/api/streaming_ingestion_demo/status` JSON from the engine (task §8: actual
/// progress, errors, processed counts, measured throughput).
fn status_json(eng: &StreamingIngestionDemoEngine) -> Json {
    let s = eng.status();
    let phase = match s.phase {
        StreamingIngestionDemoPhase::Idle => "idle",
        StreamingIngestionDemoPhase::Running => "running",
        StreamingIngestionDemoPhase::Paused => "paused",
        StreamingIngestionDemoPhase::Completed => "completed",
        StreamingIngestionDemoPhase::Error => "error",
    };
    let mut o = Json::object();
    o.set("status", Json::str(phase));
    o.set("position", Json::uint(s.position as u64));
    o.set("total", Json::uint(s.total as u64));
    o.set("processed", Json::uint(s.processed as u64));
    o.set("duplicates", Json::uint(s.duplicates as u64));
    o.set("errors", Json::uint(s.errors as u64));
    o.set("measured_tps", Json::float(s.measured_tps));
    o.set("current_rss_bytes", Json::uint(s.current_rss_bytes));
    o.set("peak_rss_bytes", Json::uint(s.peak_rss_bytes));
    match &s.last_error {
        Some(e) => {
            o.set("last_error", Json::str(e));
        }
        None => {
            o.set("last_error", Json::Null);
        }
    }
    o
}

/// Build the paginated `/api/transactions` JSON (task §3C). Each row is one
/// committed record keyed by its stable locator (event id) with sender,
/// receiver, timestamp and status. Amounts are NOT included (task §6).
///
/// Stable ordering = locator ascending = the dataset's canonical `(t,s,r)`
/// ingestion order (task §3C: stable ordering based on the dataset's defined
/// transaction order). Navigation cannot duplicate or omit records because the
/// locator space is dense `[0, committed)` and pages are contiguous slices.
fn transactions_page(eng: &StreamingIngestionDemoEngine, page: usize, page_size: usize) -> Json {
    let p = eng.pipeline();
    let committed = p.snapshot().committed_records;
    // Clamp defensively: pages are 1-based; page_size >= 1. A page past the end
    // simply yields zero rows (never a panic on `page - 1` underflow).
    let page = page.max(1);
    let page_size = page_size.max(1);
    let start = (page - 1).saturating_mul(page_size).min(committed);
    let end = start.saturating_add(page_size).min(committed);
    let mut rows = Vec::new();
    for loc in start..end {
        if let Some(r) = p.snapshot().temporal.resolve(crate::retrieval::EventLocator(loc as u32)) {
            let sender = p
                .snapshot()
                .senders
                .get(r.sender as usize)
                .cloned()
                .unwrap_or_default();
            let receiver = p
                .snapshot()
                .receivers
                .get(r.receiver as usize)
                .cloned()
                .unwrap_or_default();
            let mut row = Json::object();
            row.set("id", Json::uint(u64::from(r.locator.0)));
            row.set("sender", Json::str(sender));
            row.set("receiver", Json::str(receiver));
            row.set("t", Json::uint(r.t));
            // Ingestion + indexing are atomic in `apply`, so any committed
            // record is fully indexed → status "completed" (task §3C/§6).
            row.set("status", Json::str("completed"));
            rows.push(row);
        }
    }
    let total_pages = committed.div_ceil(page_size).max(1);
    let mut o = Json::object();
    o.set("page", Json::uint(page as u64));
    o.set("page_size", Json::uint(page_size as u64));
    o.set("total", Json::uint(committed as u64));
    o.set("total_pages", Json::uint(total_pages as u64));
    o.set("rows", Json::Array(rows));
    o
}

/// Build `/api/transactions/{id}` response (task §6). Structural fields only.
fn transaction_detail(eng: &StreamingIngestionDemoEngine, id: u64) -> Response {
    let p = eng.pipeline();
    let Ok(loc) = u32::try_from(id) else {
        return Response::not_found("transaction not found");
    };
    match p.snapshot().temporal.resolve(crate::retrieval::EventLocator(loc)) {
        Some(r) => {
            let sender = p
                .snapshot()
                .senders
                .get(r.sender as usize)
                .cloned()
                .unwrap_or_default();
            let receiver = p
                .snapshot()
                .receivers
                .get(r.receiver as usize)
                .cloned()
                .unwrap_or_default();
            let mut o = Json::object();
            o.set("id", Json::uint(id));
            o.set("sender", Json::str(sender));
            o.set("receiver", Json::str(receiver));
            o.set("t", Json::uint(r.t));
            o.set("status", Json::str("completed"));
            Response::ok(o.to_string())
        }
        None => Response::not_found("transaction not found"),
    }
}


/// Handle one connection: read requests until EOF/error, routing each.
fn handle_connection(engine: &Arc<Mutex<StreamingIngestionDemoEngine>>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let mut writer = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    loop {
        match read_request(&mut reader) {
            Ok(Some(req)) => {
                let close = req.connection_close;
                let resp = route(engine, &req);
                if writer.write_all(&resp.to_http()).is_err() {
                    break;
                }
                let _ = writer.flush();
                if close {
                    break; // honor `Connection: close`
                }
            }
            Ok(None) => break, // clean EOF
            Err(_) => break,   // malformed / dropped connection
        }
    }
}

/// Public single-connection entry point (test seam): serves one accepted
/// socket to completion. The server loop and tests both funnel through here.
pub fn handle_one(engine: &Arc<Mutex<StreamingIngestionDemoEngine>>, stream: TcpStream) {
    handle_connection(engine, stream);
}

/// The scheduler thread (task §2B/§7): while the engine is `Running`, apply
/// exactly one record per tick via `apply_one` and spin as fast as possible
/// — no rate limiting, no target throughput, natural streaming at the maximum
/// speed the infrastructure allows. Each record is applied independently
/// through `Pipeline::apply` (CAS → frontier append → index → commit) — there
/// is no read-batch-then-apply-batch step, no per-chunk grouping. The lock is
/// held only for a single record, so pagination stays responsive; pause is
/// honored at the next record boundary (a safe cancellation point — task §7
/// Pause).
fn scheduler(engine: Arc<Mutex<StreamingIngestionDemoEngine>>) {
    loop {
        let phase = {
            let eng = engine.lock().unwrap();
            eng.status().phase
        };
        match phase {
            StreamingIngestionDemoPhase::Running => {
                let finished_now = {
                    let mut eng = engine.lock().unwrap();
                    match eng.apply_one() {
                        Ok(false) => true, // source exhausted -> completed
                        Ok(true) => {
                            let st = eng.status();
                            st.position >= st.total
                        }
                        Err(_) => false,
                    }
                };
                if finished_now {
                    // Brief pause before re-checking (phase becomes Completed).
                    thread::sleep(std::time::Duration::from_millis(20));
                }
                // No sleep between records — spin as fast as possible.
            }
            StreamingIngestionDemoPhase::Completed => {
                thread::sleep(std::time::Duration::from_millis(100));
            }
            // Idle / Paused / Error: idle-wait for a state change.
            _ => {
                thread::sleep(std::time::Duration::from_millis(25));
            }
        }
    }
}

/// Run the streaming_ingestion_demo HTTP server: bind, spawn the scheduler, and accept
/// connections (one thread each). Blocks forever.
///
/// # Errors
/// `Err` if the listener cannot bind (address in use / permissions).
pub fn serve(engine: Arc<Mutex<StreamingIngestionDemoEngine>>, addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    // Scheduler thread drives natural streaming as fast as possible.
    let sched_engine = Arc::clone(&engine);
    thread::spawn(move || scheduler(sched_engine));
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let eng = Arc::clone(&engine);
                thread::spawn(move || handle_connection(&eng, s));
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

