//! Ingestion-backed HTTP server entrypoint (reconciliation Phase 4).
//!
//! Serves the ingestion-native web surface over an [`IngestPipeline`] — the
//! CSV/Kafka carrier with the shared account namespace, compound identity and
//! derived feature metrics. Unlike `streaming_ingestion_demo_server`, it needs **no `SAMCORE02`
//! artifact**; it starts from a clean substrate and serves whatever records are
//! applied through `IngestPipeline` (the retired role-separated / SAMCORE02
//! streaming_ingestion_demo model is not used here).
//!
//! ```text
//! ingest_server [bind_addr]
//!   bind_addr  default: 0.0.0.0:8091  (env ABACUS_ADDR overrides)
//! ```
//!
//! Std-only (§0 H1). The frontend routes `/api/ingest/*` (status, transactions,
//! networks, **and the `POST /api/ingest/csv` upload**) here (see
//! `web/app/api/[...path]/route.ts` + `INGEST_ORIGIN`); `/api/v1/*` still
//! routes to the upload server on :8090 (`UPLOAD_ORIGIN`). No amount is ever
//! serialized — only the frozen derived feature metrics (GS §9 / A1).
//!
//! NOTE — no persistence (deferred to a future refactor): the pipeline starts
//! empty (`IngestPipeline::new()`, in-memory) and a restart wipes every upload
//! the dashboard showed. Do not rely on this server for durable state.

use abacus_pipeline::ingest_pipeline::IngestPipeline;
use abacus_pipeline::web::ingest_server::serve_ingest;
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let addr = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("ABACUS_ADDR").ok())
        .unwrap_or_else(|| "0.0.0.0:8091".to_string());

    // Clean ingestion substrate: no SAMCORE02 artifact is read. Records are
    // applied through IngestPipeline by the ingestion driver/session; this
    // server exposes the resulting shared-namespace + derived-feature view.
    // NOTE — no persistence (deferred to a future refactor): this state is
    // in-memory only; a restart wipes every upload (`IngestPipeline::new()`).
    let engine = IngestPipeline::new();
    eprintln!(
        "ingest_server: source=csv_kafka_ingestion accounts={} bind={addr}",
        engine.accounts().account_count()
    );
    let shared = Arc::new(Mutex::new(engine));
    serve_ingest(shared, &addr).expect("ingest_server: bind/serve failed");
}
