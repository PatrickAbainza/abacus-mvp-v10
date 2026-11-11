//! `streaming_ingestion_demo_http` — std-only HTTP surface for the SAMLD streaming_ingestion_demo UI.
//!
//! Golden-source grounding: the web layer is a read/control boundary over the
//! restored streaming_ingestion_demo pipeline (`StreamingIngestionDemoEngine` + `Pipeline::apply`). It
//! introduces NO new tensor behavior, NO graph database, and NO amount
//! exposure — every structural view is derived from the canonical tensor +
//! phase-3 indexes through the existing retrieval APIs. Std-only (§0 H1).
//!
//! This is a SEPARATE surface from the CSV/Kafka ingestion web (`crate::web`);
//! the two share no state and can run independently on different ports.

pub mod graph;
pub mod json;
pub mod server;

pub use server::{serve, Request, Response, DEFAULT_PAGE_SIZE};
