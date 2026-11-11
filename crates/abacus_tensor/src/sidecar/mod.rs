//! Sidecar subsystem: rows 24–25 (§B.2) + P11.
//!
//! ```text
//! row 24  sidecar build; sort key; type distribution  CSV read + binary label read (Is_laundering ∈ {0,1}) + prec_4 sort + ordered counting
//! row 25  sidecar serialize / deserialize             fixed-order emission; FNV over records; strict order check
//! ```
//!
//! **Classification (P11).** The sidecar is provenance only. It is *not* in
//! the admissible operation set: it consumes CSV metadata, constructs no
//! element of `X`, and never reads `tensor.csf`. It therefore cannot redefine
//! tensor value semantics, and tensor logic never reads it.

pub mod builder;
pub mod deserialize;
pub mod serialize;
pub mod types;

pub use builder::{build_sidecar, build_sidecar_from_csv};
pub use deserialize::deserialize_sidecar;
pub use serialize::{
    serialize_sidecar, FORMAT_VERSION as SIDECAR_FORMAT_VERSION, MAGIC as SIDECAR_MAGIC,
};
pub use types::{LaunderingRecord, LaunderingSidecar};
