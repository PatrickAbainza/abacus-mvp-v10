//! Row 27 (§B.2): the public API boundary.
//!
//! A single sanctioned **read boundary** for computation consumers. Every
//! operation here is an evaluation, restriction, decode, or load of the
//! canonical tensor and re-exposes rows 16, 15 and 17 without adding algebra:
//!
//! ```text
//! time_slice  row 16 mode-0 restriction, half-open window, prec_3 order
//! lookup      row 16 point read; None iff x ∉ supp(T) (absent, never fabricated)
//! decode      row 15 CSF(T)⟧ (regroup only)
//! deserialize row 17 artifact load
//! checksum    d_fnv over serialized bytes (provenance infrastructure)
//! ```
//!
//! The `graphblas` submodule exposes the approved 23-primitive GraphBLAS
//! algebraic layer (Addendum O, rows 30–51) through this same boundary:
//! graphblas operations consume the canonical tensor via `read` and produce
//! sparse algebraic structures (`PairMatrix`, `Vector`, `SupportTensors`,
//! `FocalContext`, …).
//!
//! Invariants: read-only (nothing here can mutate the canonical object),
//! deterministic (total iteration order only).

pub mod read;
pub mod graphblas;

// Re-export the read-boundary types and functions at the `api` level for
// backward compatibility with existing call sites
// (`crate::api::time_slice`, `crate::api::lookup`, `crate::api::decode`, …).
pub use read::{
    CanonicalTensor, TensorEntry, time_slice, lookup, decode, deserialize, checksum,
};