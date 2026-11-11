//! GraphBLAS operations exposed through the API layer (Addendum O, rows 30–51;
//! row-27 boundary via `crate::api::read`).
//!
//! These functions consume the canonical tensor through `crate::api::read` and
//! produce GraphBLAS algebraic structures. Read-only with respect to the
//! canonical tensor; deterministic; fail-closed on overflow (P2); absent
//! reported absent (P3).
//!
//! Lower-level primitives operating on `PairMatrix` / `Vector` remain
//! directly available at `crate::graphblas::primitives::{…}`; this module
//! provides the API-boundary entry points that start from a
//! `CanonicalTensor`.

use crate::api::read::{CanonicalTensor, TensorEntry};
use crate::graphblas::context::{
    build_focal_context as gb_build_focal_context, FocalContext,
};
use crate::graphblas::fan_in::{compute_fan_in as gb_compute_fan_in};
use crate::graphblas::matrix::PairMatrix;
use crate::graphblas::primitives::support as prims_support;
use crate::graphblas::pruning::{
    hop_ball as gb_hop_ball, prune_edges as gb_prune_edges, prune_hops as gb_prune_hops,
    prune_time as gb_prune_time, prune_vertices as gb_prune_vertices,
};
use crate::graphblas::projection::{
    contract_mode0 as gb_contract_mode0, extract_3d as gb_extract_3d,
    temporal_multiplicity as gb_temporal_multiplicity,
};
use crate::graphblas::support::{build_support as gb_build_support};
use crate::graphblas::types::{
    EligiblePairs, FocalTransaction, HopBound, TimeWindow, TransactionRecord, VertexId,
};
use crate::graphblas::vector::Vector;
use crate::TensorError;

// ---------------------------------------------------------------------------
// Public re-exports so API callers do not need to reach into
// `crate::graphblas::*` for the algebraic types.
// ---------------------------------------------------------------------------

pub use crate::graphblas::context::{ContextParameters, ExecutionError};
pub use crate::graphblas::fan_in::FanInResult;
pub use crate::graphblas::matrix::Overflow;
pub use crate::graphblas::support::SupportTensors;
pub use crate::graphblas::types::{EdgeRef, SelfLoopRule};

// ---------------------------------------------------------------------------
// API-boundary functions: start from `&CanonicalTensor`, return GraphBLAS
// structures.
// ---------------------------------------------------------------------------

/// `T(t, s, r)` — exact point evaluation through the row-27 boundary
/// (delegates to `crate::api::read::lookup`).
#[must_use]
pub fn lookup(csf: &CanonicalTensor, t: u64, s: u32, r: u32) -> Option<i64> {
    crate::api::read::lookup(csf, t, s, r)
}

/// `T|_w` — temporal restriction to `t ∈ [start, end)` through the row-27
/// boundary (delegates to `crate::api::read::time_slice`).
#[must_use]
pub fn time_slice(csf: &CanonicalTensor, start: u64, end: u64) -> Vec<TensorEntry> {
    crate::api::read::time_slice(csf, start, end)
}

/// Mode-0 contraction over a temporal window:
/// `A_w(s, r) = Σ_{t ∈ [start,end)} T(t, s, r)` (row 30).
///
/// # Errors
/// `TensorError::Range` (`E_rng`) when an exact sum is unrepresentable (P2).
pub fn contract_mode0_window(
    csf: &CanonicalTensor,
    start: u64,
    end: u64,
) -> Result<PairMatrix, TensorError> {
    let windowed: Vec<TensorEntry> = time_slice(csf, start, end);
    gb_contract_mode0(&windowed)
}

/// Temporal-support multiplicity over a window: `A_cnt(s, r)` (row 31).
#[must_use]
pub fn temporal_multiplicity_window(
    csf: &CanonicalTensor,
    start: u64,
    end: u64,
) -> PairMatrix {
    let windowed: Vec<TensorEntry> = time_slice(csf, start, end);
    gb_temporal_multiplicity(&windowed)
}

/// Build all three support tensors over a temporal window (row 48):
/// `A_cnt`, `A_one = support(A_cnt)`, `A_vol` (checked contraction).
///
/// # Errors
/// `TensorError::Range` when an exact volume sum is unrepresentable (P2).
pub fn build_support_window(
    csf: &CanonicalTensor,
    start: u64,
    end: u64,
) -> Result<SupportTensors, TensorError> {
    let windowed: Vec<TransactionRecord> = time_slice(csf, start, end);
    gb_build_support(&windowed)
}

/// 3D extract: `T(t, s, r)` at an exact canonical coordinate; `None` when the
/// coordinate is outside `supp(T)` (P3).
#[must_use]
pub fn extract_3d(csf: &CanonicalTensor, t: u64, s: u32, r: u32) -> Option<i64> {
    gb_extract_3d(csf, t, s, r)
}

/// Build the canonical focal context `N_f` from the canonical CSF tensor
/// (row 49, `#20`).
///
/// # Errors
/// [`ExecutionError`] on focal window/absence violations or volume overflow.
pub fn build_focal_context_from_tensor(
    csf: &CanonicalTensor,
    focal: FocalTransaction,
    params: &ContextParameters,
) -> Result<FocalContext, ExecutionError> {
    gb_build_focal_context(csf, focal, params)
}

/// Compute the fan-in observable `d⁻(v_f)` over `A_one` (rows 50–51, `#21`/`#22`),
/// starting from a canonical tensor and building the focal context internally.
///
/// # Errors
/// [`ExecutionError`] (contract parity; fan-in itself is a pure count and
/// cannot fail).
pub fn compute_fan_in_from_tensor(
    csf: &CanonicalTensor,
    focal: FocalTransaction,
    params: &ContextParameters,
) -> Result<FanInResult, ExecutionError> {
    let context = gb_build_focal_context(csf, focal, params)?;
    gb_compute_fan_in(&context, &focal)
}

/// Compute the `<= k`-hop vertex ball around `{u_f, v_f}` over a temporal
/// window, using `support(temporal_multiplicity(window))` as `A_one`
/// (rows 46–47, `#17`/`#18`).
#[must_use]
pub fn hop_ball_from_tensor(
    csf: &CanonicalTensor,
    start: u64,
    end: u64,
    focal_u: VertexId,
    focal_v: VertexId,
    k: HopBound,
) -> Vector {
    let windowed: Vec<TensorEntry> = crate::api::read::time_slice(csf, start, end);
    let a_cnt = gb_temporal_multiplicity(&windowed);
    let a_one = prims_support(&a_cnt);
    gb_hop_ball(&a_one, focal_u, focal_v, k)
}

/// Temporal pruning: `T|_w` as windowed canonical records (row 43, `#14`).
#[must_use]
pub fn prune_time_from_tensor(
    csf: &CanonicalTensor,
    start: u64,
    end: u64,
) -> Vec<TransactionRecord> {
    gb_prune_time(csf, TimeWindow { start, end })
}

/// Vertex pruning on windowed records (row 44, `#15`): retain entries whose
/// sender and receiver both lie in `allowed`.
#[must_use]
pub fn prune_vertices_window(
    records: &[TransactionRecord],
    allowed: &Vector,
) -> Vec<TransactionRecord> {
    gb_prune_vertices(records, allowed)
}

/// Edge pruning on windowed records (row 45, `#16`): retain entries whose
/// `(sender, receiver)` pair lies in the admissible pair mask.
#[must_use]
pub fn prune_edges_window(
    records: &[TransactionRecord],
    allowed: &EligiblePairs,
) -> Vec<TransactionRecord> {
    gb_prune_edges(records, allowed)
}

/// Hop pruning from a tensor over a temporal window (rows 46–47, `#17`/`#18`):
/// retain windowed entries whose endpoints both lie in the `<= k`-hop ball.
#[must_use]
pub fn prune_hops_from_tensor(
    csf: &CanonicalTensor,
    start: u64,
    end: u64,
    focal_u: VertexId,
    focal_v: VertexId,
    k: HopBound,
) -> Vec<TransactionRecord> {
    let windowed = prune_time_from_tensor(csf, start, end);
    let a_cnt = gb_temporal_multiplicity(&windowed);
    let a_one = prims_support(&a_cnt);
    gb_prune_hops(&windowed, &a_one, focal_u, focal_v, k)
}

// ---------------------------------------------------------------------------
// Lower-level primitive re-exports (operate on `PairMatrix` / `Vector`).
// These are API-accessible for callers that already have algebraic
// structures and want to compose primitives directly.
// ---------------------------------------------------------------------------

/// `extract` (#1, row 32): value of `A(s, r)`, or `None` when absent.
#[must_use]
pub fn extract(a: &PairMatrix, s: u32, r: u32) -> Option<i64> {
    crate::graphblas::primitives::extract(a, s, r)
}

/// `reduce_col_plus` (#2, row 33): checked sum of column `v` (all senders
/// into the receiver). Absent column goes to `Ok(0)`.
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (never wraps).
pub fn reduce_col_plus(a: &PairMatrix, v: u32) -> Result<i64, TensorError> {
    crate::graphblas::primitives::reduce_col_plus(a, v)
}

/// `reduce_row_plus` (#3, row 34): checked sum of row `u` (sender into all
/// receivers).
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (never wraps).
pub fn reduce_row_plus(a: &PairMatrix, u: u32) -> Result<i64, TensorError> {
    crate::graphblas::primitives::reduce_row_plus(a, u)
}

/// `mxv` under `LOR_LAND` (#4, row 35): `v' = A × v` — senders of frontier
/// receivers (backward step).
#[must_use]
pub fn mxv(a: &PairMatrix, frontier: &Vector) -> Vector {
    crate::graphblas::primitives::mxv(a, frontier)
}

/// `vxm` under `LOR_LAND` (#5, row 36): `v' = v × A` — receivers of frontier
/// senders (forward step).
#[must_use]
pub fn vxm(a: &PairMatrix, frontier: &Vector) -> Vector {
    crate::graphblas::primitives::vxm(a, frontier)
}

/// `ewise_add_lor` (#6, row 37): Boolean element-wise OR — pattern union.
#[must_use]
pub fn ewise_add_lor(a: &Vector, b: &Vector) -> Vector {
    crate::graphblas::primitives::ewise_add_lor(a, b)
}

/// `support` (#7, row 38): `A_one = [A_cnt != 0]` (existence map).
#[must_use]
pub fn support_matrix(a_cnt: &PairMatrix) -> PairMatrix {
    prims_support(a_cnt)
}

/// `multiplicity` (#8, row 39): build `A_cnt` from raw pair occurrences.
#[must_use]
pub fn multiplicity(pairs: &[(u32, u32)]) -> PairMatrix {
    crate::graphblas::primitives::multiplicity(pairs)
}

/// `volume` (#9, row 40): build `A_vol` from `(pair, amount)` occurrences
/// with checked micro-unit sums.
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (P2).
pub fn volume(
    occurrences: &[((u32, u32), i64)],
) -> Result<PairMatrix, TensorError> {
    crate::graphblas::primitives::volume(occurrences)
}

/// `reduce_count` (#10, row 41): cardinality of a support pattern.
#[must_use]
pub fn reduce_count(members: &[u32]) -> u64 {
    crate::graphblas::primitives::reduce_count(members)
}

/// `reduce_count_set` (#11, row 41): cardinality of a vertex pattern vector.
#[must_use]
pub fn reduce_count_set(s: &Vector) -> u64 {
    crate::graphblas::primitives::reduce_count_set(s)
}

/// `reduce_sum` (#12, row 42): canonical summation over micro-unit values
/// (checked; empty goes to `0`).
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (P2).
pub fn reduce_sum(values: &[i64]) -> Result<i64, TensorError> {
    crate::graphblas::primitives::reduce_sum(values)
}

/// `reduce_max` (#13, row 42): canonical maximum; empty goes to `0` (v5 identity).
#[must_use]
pub fn reduce_max(values: &[i64]) -> i64 {
    crate::graphblas::primitives::reduce_max(values)
}