//! Canonical pruning operators (#14–#18, rows 43–47; golden O.3).
//!
//! Ported from v7 `graphblas/pruning` (itself from v5
//! `abacus_canonical_execution::pruning`): `prune_time` is the row-16
//! restriction via row 27; `prune_vertices`/`prune_edges` are masked
//! extracts against sparse patterns; `hop_ball` is bounded traversal as
//! composed masked `mxv`/`vxm` under `LOR_LAND` with the complement mask
//! (`difference`) as the unvisited mask — no queue, no visited set, no
//! handwritten BFS.

pub mod edge;
pub mod hops;
pub mod temporal;
pub mod vertex;

pub use edge::prune_edges;
pub use hops::{hop_ball, prune_hops};
pub use temporal::prune_time;
pub use vertex::prune_vertices;
