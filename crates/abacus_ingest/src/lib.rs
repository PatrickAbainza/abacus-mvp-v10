//! `abacus_ingest` — the Abacus ingestion pipeline.
//!
//! Converts supported transaction sources into one canonical event contract and
//! applies those events to the **existing** Abacus sparse-tensor substrate.
//!
//! Governing documents (authoritative; not restated here):
//! 1. `docs/ABACUS INGESTION PIPELINE.md` — Golden Source v2.0.
//! 2. `docs/Abacus Ingestion Pipeline Canonical Contract v1.0.md`.
//! 3. `docs/Abacus Ingestion Wire Format v1.0.md`.
//!
//! # Module map and its governing sections
//!
//! ```text
//! config     wire §1 layout constants, manifest + config fingerprint (§4)
//! hash       SHA-256 for payload / config / header digests (wire §4)
//! namespace  128-bit event namespace + registry (Canonical Contract §3)
//! features   the frozen 7-byte payload (Golden Source §9, Contract §6)
//! record     the 23+2W canonical record (Contract §7, wire §3)
//! wire       the 176-byte ABACSEG1 header + segment (wire §1/§6)
//! identity   dedup + conflicting-identity rejection (Golden Source §5.1)
//! account    wire-width -> substrate u32 code map (Golden Source §4/§6)
//! timestamp  event_time_ms -> substrate seconds (Contract §5)
//! apply      the substrate application boundary (Golden Source §11/§14)
//! session    bounded chunks, idempotency, state machine (Golden Source §13)
//! report     the machine-readable report (Contract §10)
//! secret     best-effort logical clearing of transient amount buffers (A1)
//! error      the structured error categories (Golden Source §16)
//! api        the versioned HTTP surface (task step 3)
//! ```
//!
//! # Architectural invariants held by this crate
//!
//! - **One substrate.** All mutation goes through
//!   `abacus_tensor::ingest::IngestionEngine`, the existing incremental path
//!   (ledger CAS `d_cas`, elementary delta P8, frontier P9). No second CSF,
//!   index or frontier is implemented here (Golden Source §14/§19).
//! - **SAMCORE02 v3 untouched.** `abacus_tensor::samld` is never modified or
//!   reinterpreted; ABACSEG1 is a separate, independently versioned envelope
//!   (wire §7, Contract §12).
//! - **Fail closed.** Every partial operation returns a categorized
//!   [`IngestError`]; nothing clamps, saturates, truncates or wraps
//!   (Golden Source §19).
//! - **No invented behavior.** Where the specifications leave a material
//!   decision open — notably what value a client-derived event contributes to
//!   the substrate's exact `i64` micro-unit value set — the code returns
//!   [`ErrorCode::SpecificationGap`] instead of choosing silently.

pub mod account;
pub mod api;
pub mod apply;
pub mod config;
pub mod error;
pub mod features;
pub mod hash;
pub mod identity;
pub mod namespace;
pub mod persist;
pub mod record;
pub mod report;
pub mod secret;
pub mod session;
pub mod timestamp;
pub mod wire;

pub use account::AccountMap;
pub use apply::{ApplyOutcome, SubstrateApplier, SubstrateValuePolicy};
pub use config::SourceConfig;
pub use error::{ErrorCode, IngestError, IngestResult};
pub use features::FeaturePayload;
pub use identity::{IdentityLedger, IdentityOutcome};
pub use namespace::{EventIdentity, EventNamespace, NamespaceRegistry};
pub use record::{record_length_for_width, CanonicalRecord};
pub use report::{CommitResult, IngestionReport};
pub use session::{
    AppliedEvent, ChunkOutcome, ChunkStatus, IngestionSession, SessionLimits, SessionState,
};
pub use timestamp::event_time_ms_to_seconds;
pub use wire::{build_segment, encode_segment, Segment, SegmentHeader};
