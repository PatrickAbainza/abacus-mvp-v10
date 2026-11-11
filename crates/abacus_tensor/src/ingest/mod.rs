//! Ingestion subsystem: rows 18–23 (§B.2), P7–P10.
//!
//! ```text
//! row 18  ledger compare-and-swap (create/update/close)  -> ledger::Ledger (d_cas)
//! row 19  frontier push / decode / eval / clear          -> frontier::Frontier
//! rows 20–23 ingest, seal, support diff, live view       -> engine::IngestionEngine
//! ```
//!
//! The ledger is point-addressed state on event identities (P7); it is **not**
//! a coordinate index for locating tensor data and no ordered structure is
//! introduced for it. The frontier is a raw append-only delta sequence
//! (Def 9.1): one physical write per event, no aggregation on the append path.
//! No applied-event set exists — the CAS equality check *is* the model-level
//! exactly-once mechanism (P7, Cor-type consequence recorded in the prior
//! derivation), and `K14` bounds that claim to the model level.

pub mod engine;
pub mod frontier;
pub mod ledger;

pub use engine::{
    csf_byte_size, diff_coordinates, sealed_decode, EngineStats, IngestionEngine, SealReport,
    StageTimings,
};
pub use frontier::Frontier;
pub use ledger::{CasOutcome, Event, EventId, Expected, Ledger, LedgerEntry, LedgerOp};
