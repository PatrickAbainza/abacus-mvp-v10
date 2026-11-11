//! `web` — std-only HTTP surface for the CSV/Kafka ingestion view.
//!
//! Golden-source grounding: the web layer is a read/control boundary over the
//! ingestion-native pipeline (`IngestPipeline`). It introduces NO new tensor
//! behavior, NO graph database, and NO amount exposure — every structural view
//! is derived from the shared Account Map + the reused phase-3 temporal indexes
//! (see [`ingest_graph`]). Std-only (§0 H1: no third-party libraries).

pub mod graph;
pub mod graphblas_server;
pub mod ingest_graph;
pub mod ingest_server;
pub mod json;
pub mod server;

pub use ingest_server::serve_ingest;
pub use server::{Request, Response, DEFAULT_PAGE_SIZE};
