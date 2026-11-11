//! `graphblas` — derived pair-plane projection layer (golden Addendum O,
//! rows 30–51; v5 primitives #1–#23 + minimal substrate).
//!
//! Ported from `abacus_mvp_v7/.../graphblas/` into `abacus_tensor` per
//! `your_task_phase_7.md` (no new crate; tensor stays authoritative):
//!
//! * Tensor access only through `crate::api` row 27 (O.1) —
//!   `projection::{prune_time_slice, extract_3d}` wrap
//!   `api::{time_slice, lookup}`; nothing else touches CSF storage.
//! * No new boundary primitive: reductions reuse the existing boundary set
//!   (`a_{01},a_{02},a_{10},a_{11},a_{14},a_{15},R_1,R_2,R_3,R_5,R_6` via
//!   `crate::prims` + std slice ops; P2 checked, P3 zero-drop).
//! * COO `(s,r)`-ordered unique pairs, never CSC (O.1).
//! * `#23 compare_with_oracle` lives in test/validation infrastructure
//!   only (`validation` module = production-callable comparison helper, but
//!   it takes a caller-supplied scalar, never the tensor — diagnostic
//!   infrastructure, not algebra; never wired into the execution path).
//! * Still excluded (no governing section): `fast.rs` FastIndex (untested
//!   duplicate adjacency index — see `context.rs` header), `features`
//!   vector, predicate/AML layers, holdout experiments, classifiers,
//!   normalization, visualization, `dimension`/`results` metadata.

pub mod context;
pub mod fan_in;
pub mod matrix;
pub mod primitives;
pub mod projection;
pub mod pruning;
pub mod support;
pub mod types;
pub mod validation;
pub mod vector;

pub use matrix::{from_occurrences, Overflow, PairMatrix};
pub use vector::Vector;
