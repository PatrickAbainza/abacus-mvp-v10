//! StreamingIngestionDemo HTTP server entrypoint (your_task_phase_4.md §8).
//!
//! Boots the std-only web server over a `SAMCORE02` artifact. The artifact
//! directory is configurable (never a hardcoded machine path — §0 convention,
//! `loader::ArtifactPaths`); it defaults to the repository's v10 SAMLD profile.
//!
//! ```text
//! streaming_ingestion_demo_server [artifact_dir] [bind_addr]
//!   artifact_dir  default: <repo>/artifacts/profiles/samld_full
//!   bind_addr     default: 0.0.0.0:8080  (env ABACUS_ADDR overrides)
//! ```
//!
//! Std-only (§0 H1). The frontend (Next.js) proxies `/api/*` here.

use abacus_pipeline::streaming_ingestion_demo_http::serve;
use abacus_pipeline::StreamingIngestionDemoEngine;
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Repo root is two levels above this crate's manifest dir. Built at compile
    // time so no machine-specific absolute path is embedded (§0 convention).
    let repo_root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let artifact_dir = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("ABACUS_ARTIFACT_DIR").ok())
        .unwrap_or_else(|| format!("{repo_root}/artifacts/profiles/samld_full"));
    let addr = args
        .get(2)
        .cloned()
        .or_else(|| std::env::var("ABACUS_ADDR").ok())
        .unwrap_or_else(|| "0.0.0.0:8080".to_string());

    let core_path = format!("{artifact_dir}/samld_core.bin");
    let engine = StreamingIngestionDemoEngine::open(&core_path).unwrap_or_else(|e| {
        eprintln!("streaming_ingestion_demo_server: failed to open artifact {core_path}: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "streaming_ingestion_demo_server: artifact={core_path} records={} bind={addr}",
        engine.status().total
    );
    let shared = Arc::new(Mutex::new(engine));
    serve(shared, &addr).expect("streaming_ingestion_demo_server: bind/serve failed");
}
