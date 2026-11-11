//! `abacus_tensor` — smallest realization of the sparse transaction tensor
//! system, derived from
//! `docs/2026-09-19_2053_abacus_tensor_atomic_primitive_derivation.md`.
//!
//! # Derivation chain (𝒮 → modules)
//!
//! ```text
//! 𝒫_boundary = {a01..a21}  (§A)          -> prims
//! 𝒟 = {d_lex,d_horner,d_round,d_settle,   -> composites (+ d_cas in ingest::ledger)
//!      d_merge2,d_fnv,d_hinnant,d_cas,
//!      d_delta,d_inv}  (§B.1)
//! R_1..R_7  (§D)                          -> R5 in prims; R1/R2/R3/R4/R6/R7 at call sites
//!
//! rows 1–9    (tensor core)      -> tensor::{coordinate,value,tensor}
//! rows 10,12–14 (COO layer)      -> coo::{entry,encode,decode,builder}
//! rows 11,15,16 (CSF layer)      -> csf::{compress,decode,query}
//! row 17      (serialization)    -> csf::serialize
//! behavioral  (facts/conditions) -> behavioral  (addendum B1–B11)
//! rule disc. (one rule, 3 params) -> rule_discovery (addendum R1–R14)
//! rule disc.2 (fan-in/fan-out)   -> rule_discovery_fanio (addendum F1–F14)
//! row 18      (ledger)           -> ingest::ledger
//! rows 19–23  (ingestion)        -> ingest::{frontier,engine}
//! rows 24,25  (sidecar)          -> sidecar::{types,builder,serialize,deserialize}
//! row 26      (statistics)       -> stats
//! row 27      (API boundary)     -> api
//! rows 28,29  (CLI, benchmark)   -> src/bin/*  (no new mathematics)
//! ```
//!
//! Each module is a distinct semantic component of the integrated system
//! (`ιᵢ : mᵢ ↪ 𝒰`): tensor semantics, COO, CSF, serialization, ingestion,
//! sidecar, statistics and interfaces are injected without collapsing into one
//! another. Sharing of primitives and composites is allowed and is the only
//! shared code. See `DERIVATION.md` for the full map and `completion_report/`
//! for the derivation/integration report.
//!
//! No third-party libraries: the external surface is the language standard
//! library plus the machine (§0, §I H1).

pub mod api;
pub mod atomic;
pub mod behavioral;
pub mod composites;
pub mod coo;
pub mod csf;
pub mod error;
pub mod fanin;
pub mod fanin_kernel;
pub mod graphblas;
pub mod ingest;
pub mod overfit;
pub mod prims;
pub mod rule_discovery;
pub mod rule_discovery_fanio;
pub mod samld;
pub mod sidecar;
pub mod stats;
pub mod tensor;
pub mod varint;

pub use error::TensorError;
