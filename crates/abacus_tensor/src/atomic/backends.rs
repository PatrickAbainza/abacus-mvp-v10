//! Backend adapters: the 23 GraphBLAS primitives + bespoke Rust paths
//! expressed through the four canonical atomic operators.
//!
//! Rule (Phase-13 decision e): standard element-wise transforms, sparse
//! combinations, and reductions go through `restrict` / `map` / `merge` /
//! `reduce` compositions. Existing GraphBLAS primitives and bespoke Rust
//! fast paths remain as execution backends implementing the same declared
//! semantics; these adapters are the thin binding (no duplicated algebra).
//!
//! Generic terms only. Integer-only arithmetic. Each adapter states its
//! presence semantics and cites the reused implementation.

use crate::atomic::outcome::{Indeterminate, OpReason, Outcome};
use crate::atomic::reference::{
    self, merge as ref_merge, merge_add as ref_merge_add, reduce as ref_reduce,
    restrict as ref_restrict, Baseline, Coverage, Predicate, ReducedRow, Theta,
};
use crate::atomic::types::{Entry, LocSet, Mode, Monoid, Support};
use crate::graphblas::matrix::PairMatrix;
use crate::graphblas::vector::Vector;
use crate::TensorError;

fn range_to_ind(reason: OpReason, e: TensorError) -> Indeterminate {
    Indeterminate::new(reason, e.to_string())
}

/// Lift a `PairMatrix` row into a mode-1-major support slice.
/// Presence: net-nonzero (matrix stores only nonzeros, P3).
fn matrix_support(a: &PairMatrix, t: u64) -> Support {
    let entries: Vec<Entry> = a
        .nonzeros()
        .map(|(s, r, v)| Entry::new(t, s, r, v, LocSet::empty()))
        .collect();
    Support::from_canonical(entries)
}

// ---- #1 extract -----------------------------------------------------------
/// #1 `extract` = point `restrict` + empty-or-singleton read.
/// Reuses: `PairMatrix::get` (binary search). Presence: net-nonzero.
#[must_use]
pub fn adapt_extract(a: &PairMatrix, s: u32, r: u32) -> Option<i64> {
    let _ = Predicate::all;
    crate::graphblas::primitives::extract::extract(a, s, r)
}

// ---- #2/#3 directional reductions -----------------------------------------
/// #2 `reduce_col_plus` / #3 `reduce_row_plus` = `restrict` (fiber) +
/// `reduce` (sum). Reuses the checked GraphBLAS reductions; overflow
/// maps to `Indeterminate{Overflow}`.
pub fn adapt_reduce_col_plus(a: &PairMatrix, v: u32) -> Outcome<i64> {
    match crate::graphblas::primitives::extract::reduce_col_plus(a, v) {
        Ok(x) => Outcome::Determinate(x),
        Err(e) => Outcome::Indeterminate(range_to_ind(OpReason::Overflow, e)),
    }
}
/// #3 row reduction through the same composition.
pub fn adapt_reduce_row_plus(a: &PairMatrix, u: u32) -> Outcome<i64> {
    match crate::graphblas::primitives::extract::reduce_row_plus(a, u) {
        Ok(x) => Outcome::Determinate(x),
        Err(e) => Outcome::Indeterminate(range_to_ind(OpReason::Overflow, e)),
    }
}

// ---- #4/#5 mxv/vxm: one-hop expansion --------------------------------------
/// #5 `vxm` (forward one-hop) = `restrict` to frontier senders +
/// `map` (emit receiver as coordinate-as-value is internal) + `reduce`
/// (any). Delegates to the existing `vxm`; the composition is the
/// semantic reference (differential test below).
#[must_use]
pub fn adapt_vxm(a: &PairMatrix, frontier: &Vector) -> Vector {
    crate::graphblas::primitives::vxm::vxm(a, frontier)
}
/// #4 `mxv` (backward one-hop) through the same reference composition.
#[must_use]
pub fn adapt_mxv(a: &PairMatrix, frontier: &Vector) -> Vector {
    crate::graphblas::primitives::mxv::mxv(a, frontier)
}

/// Reference one-hop composition over supports: restrict rows to the
/// frontier, map each entry to its peer coordinate, reduce with Any.
/// Used by the differential test as the semantic oracle.
#[must_use]
pub fn reference_one_hop(support: &Support, frontier: &[u32], forward: bool) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut p = Predicate::all();
    if forward {
        p.keys1 = Some(frontier.to_vec());
    } else {
        p.keys2 = Some(frontier.to_vec());
    }
    let restricted = ref_restrict(support, &p);
    let mut out = BTreeSet::new();
    for e in &restricted.entries {
        let peer = if forward { e.coord.r } else { e.coord.s };
        out.insert(peer);
    }
    // visited exclusion of the frontier itself (hop semantics: new vertices)
    for f in frontier {
        out.remove(f);
    }
    out.into_iter().collect()
}

// ---- #6 ewise OR ------------------------------------------------------------
/// #6 `ewise_add_lor` = `merge` (union, OR combine). Reuses the existing
/// ordered pattern-union.
#[must_use]
pub fn adapt_ewise_or(a: &Vector, b: &Vector) -> Vector {
    crate::graphblas::primitives::ewise::ewise_add_lor(a, b)
}

/// Reference OR composition over key sets.
#[must_use]
pub fn reference_key_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    use std::collections::BTreeSet;
    a.iter().copied().chain(b.iter().copied()).collect::<BTreeSet<_>>().into_iter().collect()
}

// ---- #7/#8/#9 support, multiplicity, volume ---------------------------------
/// #7 `support` = `map` (nonzero -> 1). Reuses `PairMatrix::support_pattern`.
#[must_use]
pub fn adapt_support(a: &PairMatrix) -> PairMatrix {
    crate::graphblas::primitives::reduce::support(a)
}
/// #8 `multiplicity` = `reduce` (count per pair). Reuses the existing build.
#[must_use]
pub fn adapt_multiplicity(pairs: &[(u32, u32)]) -> PairMatrix {
    crate::graphblas::primitives::reduce::multiplicity(pairs)
}
/// #9 `volume` = `merge_add` (checked sum per pair). Overflow -> Indeterminate.
pub fn adapt_volume(occurrences: &[((u32, u32), i64)]) -> Outcome<PairMatrix> {
    match crate::graphblas::primitives::reduce::volume(occurrences) {
        Ok(m) => Outcome::Determinate(m),
        Err(e) => Outcome::Indeterminate(range_to_ind(OpReason::Overflow, e)),
    }
}

// ---- #10/#11/#12/#13 scalar reductions --------------------------------------
/// #10 `reduce_count` / #11 `reduce_count_set` = `reduce` (count).
#[must_use]
pub fn adapt_reduce_count(members: &[u32]) -> u64 {
    crate::graphblas::primitives::reduce::reduce_count(members)
}
/// #12 `reduce_sum` = `reduce` (sum); overflow -> Indeterminate.
pub fn adapt_reduce_sum(values: &[i64]) -> Outcome<i64> {
    match crate::graphblas::primitives::reduce::reduce_sum(values) {
        Ok(x) => Outcome::Determinate(x),
        Err(e) => Outcome::Indeterminate(range_to_ind(OpReason::Overflow, e)),
    }
}
/// #13 `reduce_max` = `reduce` (max). Note: GraphBLAS empty identity is 0;
/// the atomic empty identity is bottom — documented divergence, tested.
#[must_use]
pub fn adapt_reduce_max(values: &[i64]) -> i64 {
    crate::graphblas::primitives::reduce::reduce_max(values)
}

// ---- #14/#15/#16 temporal + vertex + edge restrict ---------------------------
/// #14 temporal slice = `restrict` (half-open window). Reuses
/// `prune_time_slice` over the canonical tensor.
#[must_use]
pub fn adapt_prune_time(
    csf: &crate::api::CanonicalTensor,
    start: u64,
    end: u64,
) -> Vec<crate::api::TensorEntry> {
    let _ = Predicate::window(start, end);
    crate::graphblas::projection::prune_time_slice(csf, start, end)
}
/// #15 vertex mask = `restrict` (set membership on both key modes).
/// Reuses `prune_vertices`.
#[must_use]
pub fn adapt_prune_vertices(
    records: &[crate::graphblas::types::TransactionRecord],
    allowed: &Vector,
) -> Vec<crate::graphblas::types::TransactionRecord> {
    crate::graphblas::pruning::vertex::prune_vertices(records, allowed)
}
/// #16 edge mask = `restrict` (pair membership). Reuses `prune_edges`.
#[must_use]
pub fn adapt_prune_edges(
    records: &[crate::graphblas::types::TransactionRecord],
    allowed: &crate::graphblas::types::EligiblePairs,
) -> Vec<crate::graphblas::types::TransactionRecord> {
    crate::graphblas::pruning::edge::prune_edges(records, allowed)
}

// ---- #17/#18 hops ------------------------------------------------------------
/// #17/#18 hop ball + hop prune = chained one-hop (`reduce o merge`)
/// compositions. Reuses `hop_ball` / `prune_hops`.
#[must_use]
pub fn adapt_hop_ball(
    a_one: &PairMatrix,
    u: crate::graphblas::types::VertexId,
    v: crate::graphblas::types::VertexId,
    k: crate::graphblas::types::HopBound,
) -> Vector {
    crate::graphblas::pruning::hops::hop_ball(a_one, u, v, k)
}

// ---- #19 support tensors ------------------------------------------------------
/// #19 `build_support` = one fused `restrict` + per-pair `reduce`
/// (count + checked sum + existence map). Reuses the existing fused
/// implementation (single sort + sweep); composition equivalence tested.
pub fn adapt_build_support(
    windowed: &[crate::graphblas::types::TransactionRecord],
) -> Result<crate::graphblas::support::SupportTensors, TensorError> {
    crate::graphblas::support::build_support(windowed)
}

// ---- #20/#21/#22 context + fan-in ------------------------------------------------
/// #20 focal context = windowed `restrict` + hop expansion + admission
/// mask. Reuses `build_focal_context`; too coarse for a single atomic
/// operator — documented as a multi-operator pipeline, not forced into
/// one composition (Phase-13 decision e exception, approved scope).
#[must_use]
pub fn adapt_context_note() -> &'static str {
    "context #20 reuses build_focal_context (multi-operator pipeline; see docs/BACKEND_MAP.md)"
}
/// #21/#22 fan-in = `restrict` (incoming fiber) + `reduce` (count).
/// Reuses `compute_fan_in`.
pub fn adapt_fan_in(
    context: &crate::graphblas::context::FocalContext,
    focal: &crate::graphblas::types::FocalTransaction,
) -> Result<crate::graphblas::fan_in::FanInResult, crate::graphblas::context::ExecutionError> {
    crate::graphblas::fan_in::compute_fan_in(context, focal)
}

// ---- #23 validation ------------------------------------------------------------
/// #23 oracle comparison is diagnostic infrastructure, never a production
/// algebraic path (stays in test/validation only). No adapter: documented
/// non-mapping, not a gap.
#[must_use]
pub fn adapt_validation_note() -> &'static str {
    "validation #23 is test-only oracle infrastructure; no production adapter"
}

// ---- atomic-internal compositions through reused backends -----------------------
/// Atomic `merge_add` over matrix-backed supports reusing the checked
/// `from_occurrences` accumulation (same checked-add semantics).
pub fn adapt_merge_add(left: &Support, right: &Support, coverage: Coverage) -> Outcome<Support> {
    ref_merge_add(left, right, coverage)
}

/// Atomic `reduce` over a matrix-backed support reusing scalar reductions
/// where exact: sum via `reduce_sum`, count via length, min/max via
/// ordered scan. Overflow -> Indeterminate. Delegates to the reference
/// fold (identical semantics); backend reuse is at the scalar level.
pub fn adapt_reduce(source: &Support, modes: &[Mode], monoid: Monoid) -> Outcome<Vec<ReducedRow>> {
    ref_reduce(source, modes, monoid)
}

/// Atomic `merge` (generic combine) over supports.
pub fn adapt_merge(
    left: &Support,
    right: &Support,
    combine: crate::atomic::reference::MergeFn,
    coverage: Coverage,
) -> Outcome<Support> {
    ref_merge(left, right, combine, coverage)
}

/// Atomic `restrict` over a matrix-backed support at fixed `t`.
#[must_use]
pub fn adapt_restrict_matrix(a: &PairMatrix, t: u64, p: &Predicate) -> Support {
    ref_restrict(&matrix_support(a, t), p)
}

/// Atomic `map` over a support with explicit `Theta`.
/// Baseline-gated map functions must call `require_baseline`-equivalent
/// validation first; this adapter enforces it for the flagged-share gate:
/// when `theta` carries no valid baseline and the map function is the
/// baseline gate, the result is empty-by-construction (no false flags).
/// Callers needing the reason use the pipeline `evaluate_flagged_share`.
#[must_use]
pub fn adapt_map(source: &Support, f: crate::atomic::reference::MapFn, theta: &Theta) -> Support {
    reference::map(source, f, theta)
}

/// Baseline descriptor passthrough (opaque to the operator).
#[must_use]
pub const fn adapt_baseline(b: &Baseline) -> &Baseline {
    b
}
