//! `streaming_ingestion_demo` — bounded, streaming SAMLD streaming_ingestion_demo over the canonical Abacus
//! sparse-tensor substrate (your_task_phase_4.md).
//!
//! Golden-source grounding (no invented tensor behavior; the existing Abacus
//! codebase is the source of truth for ingestion and tensor behavior):
//! - Authoritative CSF + incremental pipeline: `Pipeline::apply` is the sole
//!   ingestion entry point (validate -> CAS -> frontier append -> derived
//!   index insert -> atomic commit; `pipeline.rs`, derivation rows 18-23 /
//!   P7-P10). The streaming_ingestion_demo engine drives ONLY this path; it never mutates the
//!   tensor or indexes directly and never seals on the per-event path.
//! - Source artifact: `SAMCORE02` ordered core (`samld/core_pack.rs`), the
//!   same lossless, non-aggregating per-transaction ledger the loader decodes
//!   (addendum §R5f). The streaming reader below reuses the exact canonical
//!   codec primitives (`varint`, `prims`) and the exact column layout so a
//!   streamed record is byte-for-byte the record `deserialize_core_v2` would
//!   produce — it is NOT a second format.
//! - Amounts: required for ingestion (P3 rejects a zero amount) but never
//!   displayed or used in any amount-dependent calculation (task §1/§6). The
//!   canonical `amount_micro` is ingested verbatim; the web layer exposes
//!   only structural fields (locator, t, sender, receiver).
//! - Determinism (P12/R7): streaming_ingestion_demo from a clean state applies records in the
//!   artifact's canonical `(t,s,r)` order, so the resulting tensor, indexes
//!   and every derived network are deterministic (task §9 test 14).
//! - Memory (task §2B): the compressed core artifact is read once (as the
//!   existing loader does, and as checksum verification requires), but decoded
//!   transaction records are produced incrementally in bounded batches — the
//!   full multi-million-record decode is never materialized to serve a chunk.

pub mod engine;
pub mod reader;

pub use engine::{StreamingIngestionDemoEngine, StreamingIngestionDemoPhase, StreamingIngestionDemoStatus};
pub use reader::{SamldStreamReader, StreamRecord};
