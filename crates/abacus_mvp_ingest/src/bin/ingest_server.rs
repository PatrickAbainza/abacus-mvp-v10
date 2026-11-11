//! `ingest_server` — a browser/curl-reachable ingestion server for
//! `abacus_mvp_v10`, backed by this repository's `abacus_tensor` substrate.
//!
//! It serves two things over one std-only HTTP/1.1 socket (thread-per-
//! connection, `Connection: close`, mirroring the repository's `streaming_ingestion_demo_server`
//! and the oracle's `ingest_server`/`shell_server` conventions):
//!
//! * **`/api/v1/*`** — delegated verbatim to the oracle's `route` handler and
//!   `IngestionService` (capabilities, sessions, chunks, seal, commit, report).
//!   This is the canonical, privacy-correct API; drive it with curl or the
//!   oracle's WASM browser shell.
//! * **`/api/ingest/csv`** (POST, raw CSV body) — a **local test-harness**
//!   convenience endpoint that normalizes the CSV server-side and runs the full
//!   create→upload→seal→commit→report path through the same service, returning
//!   privacy-safe preview + reconciled report JSON. This lets a plain browser
//!   (or curl) test CSV ingestion into the substrate without a WASM build.
//!   NOTE: it uploads the raw CSV to the server; the privacy-preserving path is
//!   client-side (WASM) normalization via `/api/v1`. Fine for localhost testing.
//!
//! Usage: `ingest_server [bind_addr] [data_dir]`, default `0.0.0.0:8090`.
//! `ABACUS_ADDR` / `ABACUS_DATA_DIR` override the args. When `data_dir` is set
//! the service is durable (Amendment A2 journal, recovered on start).

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use abacus_ingest::api::json::Json;
use abacus_ingest::api::server::{read_request, route, Response};
use abacus_ingest::api::IngestionService;

fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args
        .next()
        .or_else(|| std::env::var("ABACUS_ADDR").ok())
        .unwrap_or_else(|| "0.0.0.0:8090".to_string());
    let data_dir = args
        .next()
        .or_else(|| std::env::var("ABACUS_DATA_DIR").ok());

    let service = match data_dir {
        Some(dir) => match IngestionService::with_persistence(Path::new(&dir)) {
            Ok(s) => {
                eprintln!("ingest_server: durable journal: {dir}/ingestion_journal.bin");
                s
            }
            Err(e) => {
                eprintln!("ingest_server: recovery failed: {e}");
                std::process::exit(1);
            }
        },
        None => IngestionService::new(),
    };
    let service = Arc::new(Mutex::new(service));

    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ingest_server: cannot bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "ingest_server: listening on http://{addr}  (API: /api/v1, CSV test: POST /api/ingest/csv)"
    );
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let svc = Arc::clone(&service);
                thread::spawn(move || serve_one(&svc, s));
            }
            Err(_) => continue,
        }
    }
}

/// Serve one accepted socket: `/api/v1/*` to the oracle router, `/api/ingest/*`
/// to the local test harness. One request per connection.
fn serve_one(service: &Arc<Mutex<IngestionService>>, stream: TcpStream) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(stream);
    let Ok(Some(mut req)) = read_request(&mut reader) else {
        return;
    };
    let resp = if req.path.starts_with("/api/v1") {
        route(service, &mut req)
    } else if req.path == "/api/ingest/csv" && req.method == "POST" {
        let mut svc = service.lock().unwrap();
        // Route by schema: a SAML-D export (separate Date/Time columns) uses
        // the SAML-D plan + value-preserving timestamp merge; any other CSV
        // uses the generic explicit plan. The selection is the one complete,
        // explicit plan for that exact header — never a partial inference.
        let outcome = if abacus_mvp_ingest::is_samld(&req.body) {
            abacus_mvp_ingest::ingest_samld_bytes(&mut svc, &req.body)
        } else {
            abacus_mvp_ingest::ingest_csv_bytes(&mut svc, &req.body)
        };
        match outcome {
            Ok(json) => json_response(200, json),
            Err(e) => {
                let mut o = Json::object();
                o.set("error", Json::str(&e.detail));
                o.set("code", Json::str(e.code.as_str()));
                json_response(400, o.to_json_string())
            }
        }
    } else if req.path == "/api/ingest/health" && req.method == "GET" {
        json_response(200, "{\"ok\":true}".to_string())
    } else {
        json_response(404, "{\"error\":\"no such route\"}".to_string())
    };
    if writer.write_all(&resp.to_http()).is_err() {
        return;
    }
    let _ = writer.flush();
}

/// Build a JSON [`Response`] directly (the oracle's `Response::ok`/`error`
/// constructors are crate-private; the struct fields are public).
fn json_response(status: u16, body: String) -> Response {
    Response {
        status,
        content_type: "application/json",
        body: body.into_bytes(),
    }
}
