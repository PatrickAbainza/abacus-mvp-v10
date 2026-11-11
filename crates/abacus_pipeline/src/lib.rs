//! `abacus_pipeline` — ingestion pipeline over the canonical Abacus sparse
//! tensor substrate (v10), carrying the CSV/Kafka model.
//!
//! Migration note (reconciliation directive): the former `SAMCORE02` /
//! role-separated `Pipeline` + `StreamingIngestionDemoEngine` + `LoadedArtifacts` streaming_ingestion_demo model
//! was retired. The pipeline now carries the **CSV/Kafka ingestion** model —
//! one shared Account Map namespace (`AccountMap`, §6), compound
//! `(namespace, event_id)` identity (Contract §3), and the frozen amount-derived
//! `FeaturePayload` metrics (GS §9, privacy-preserving). Substrate mutation is
//! delegated to the ported oracle's `SubstrateApplier` over the existing
//! `IngestionEngine` (ledger CAS + frontier), reused — never reimplemented.
//!
//! Golden-source grounding (no invented tensor behavior):
//! - Canonical tensor + CSF hierarchy `t -> s -> r`: hierarchical-CSF invariant
//!   `golden_source_hiearchical_csf_tensor.md` §§1-4,10; substrate `abacus_tensor`.
//! - Ingestion: ledger CAS + frontier = derivation rows 18-23 / P7-P10, applied
//!   through `abacus_tensor::ingest::{ledger, frontier, engine}`.
//! - Temporal retrieval indexes (`retrieval.rs`) are the shared, reused
//!   phase-3 forward/backward/global/directory structures over committed
//!   `(t, s, r)` — keyed by the shared account namespace.
//!
//! Phase-1 rule: per-event work touches only the affected records/indexes. The
//! sealed CSF is never mutated in place; incremental visibility is sealed CSF +
//! frontier overlay + derived lookups over exactly those two sources.
//
//! Two HTTP surfaces are registered, additively:
//! - `crate::web` — CSV/Kafka ingestion view (`serve_ingest`, `/api/ingest/*`).
//! - `crate::streaming_ingestion_demo_http` — SAMCORE02 streaming_ingestion_demo view (`serve_streaming_ingestion_demo`,
//!   `/api/streaming_ingestion_demo/*` + `/api/transactions/*`), driven by `StreamingIngestionDemoEngine` over a
//!   `SAMCORE02` artifact (e.g. `artifacts/profiles/samld_full`). Streams records through `Pipeline::apply` as fast
//!   as possible — no target throughput, natural streaming only.
//!   The two surfaces share no state and run independently on different ports.

pub mod atomic;
pub mod error;
pub mod bounded_context;
pub mod focal_contract;
pub mod fanin_agents;
pub mod ingest_model;
pub mod ingest_pipeline;
pub mod retrieval;
pub mod semantic;
pub mod web;

// StreamingIngestionDemo surface over SAMCORE02 artifacts (restored phase-4 pipeline), additive
// to the CSV/Kafka ingestion surface above. Reads `artifacts/profiles/samld_full`
// by default and streams records through `Pipeline::apply` as fast as possible —
// no target throughput.
pub mod loader;
pub mod persist;
pub mod pipeline;
pub mod streaming_ingestion_demo;
pub mod streaming_ingestion_demo_http;
pub mod state;

pub use error::PipelineError;
pub use ingest_model::{account_node_id, AccountMap, EventIdentity, EventNamespace, FeaturePayload, IngestedEvent};
pub use ingest_pipeline::{IngestApplyOutcome, IngestPipeline, PreparedApplyOutcome};
pub use retrieval::{
    DirectionalIndex, DirectoryEntry, EventLocator, GlobalTimeIndex, IndexMemoryReport,
    LocatorBuffer, TemporalIndexes, TemporalRef, DEFAULT_DELTA_SECS, INLINE_LOCATOR_CAP,
    MAX_EDGES_PER_LOOKUP, MultiHopQuery, ResolvedEvent, RetrievalResult,
};
pub use web::serve_ingest;
pub use web::server::{Request, Response, DEFAULT_PAGE_SIZE};

pub use loader::{ArtifactPaths, LoadedArtifacts, load_artifacts, LoadReport};
pub use pipeline::{ApplyOutcome, CheckpointImage, NormalizedEvent, Pipeline, PipelineConfig};
pub use persist::{CheckpointState, RecoverReport, checkpoint_read, checkpoint_write, journal_append, journal_read, journal_rewrite_clean, recover};
pub use streaming_ingestion_demo::{StreamingIngestionDemoEngine, StreamingIngestionDemoPhase, StreamingIngestionDemoStatus, SamldStreamReader};
pub use streaming_ingestion_demo_http::serve as serve_streaming_ingestion_demo;
pub use state::{AccountIndex, PipelineSnapshot, StateVersion};
