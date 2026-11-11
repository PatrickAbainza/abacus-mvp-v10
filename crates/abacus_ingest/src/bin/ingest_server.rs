//! The ingestion API server binary.
//!
//! Usage: `ingest_server [bind_address] [data_dir]`, default
//! `127.0.0.1:8099`. When `data_dir` is given, the service is durable:
//! every acknowledged operation is journaled there (Amendment A2) and is
//! recovered on the next start.
//!
//! Std-only HTTP/1.1, thread-per-connection, following the repository's
//! existing `streaming_ingestion_demo_server` convention. No business logic lives here.

use abacus_ingest::api::{serve, IngestionService};
use std::sync::{Arc, Mutex};

fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| "127.0.0.1:8099".to_string());
    let data_dir = args.next();
    let service = match data_dir {
        Some(dir) => match IngestionService::with_persistence(std::path::Path::new(&dir)) {
            Ok(s) => {
                eprintln!("durable journal: {dir}/ingestion_journal.bin");
                s
            }
            Err(e) => {
                eprintln!("recovery failed: {e}");
                std::process::exit(1);
            }
        },
        None => IngestionService::new(),
    };
    let service = Arc::new(Mutex::new(service));
    eprintln!("abacus_ingest API listening on http://{addr}/api/v1");
    if let Err(e) = serve(service, &addr) {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
