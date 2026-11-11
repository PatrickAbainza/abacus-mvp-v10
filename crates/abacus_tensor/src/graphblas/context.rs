//! Focal-context constructor (#20, row 49): `Gamma(T, t_f, w, k)` (golden O.3).
//!
//! Ported from v7 `graphblas/context.rs` — the `context.rs` (reference)
//! variant, selected per `your_task_phase_7.md` section 1 because it is the
//! variant the reference tests pin: `tests/graphblas_pipeline.rs`
//! (`context_ball_equals_hop_ball_over_window_candidate_support`,
//! `primitives_19_to_22_context_fan_in_vs_oracle`) executes this exact code
//! path (5/5 green in the donor, verified before porting). The `fast.rs`
//! `FastIndex` variant is NOT ported: it has no reference-test coverage
//! (zero hits in the donor test tree), builds a separate `HashMap`-based
//! sender/receiver adjacency index over a full CSF decode (= a duplicate
//! index the task forbids without demonstrated requirement), and would add
//! a new abstraction instead of reusing existing `Vector` set operations.
//!
//! Pipeline: window check goes to focal-existence check
//! (`T(t_f,s_f,r_f) != 0`, P3 — absent is an error, never re-created) goes to
//! `prune_time` goes to bounded hop ball goes to admission (ball +
//! eligible-pair mask + self-loop rule) goes to `build_support` goes to
//! vertex set (reused ball).
//!
//! Execution-level optimization carried over verbatim (semantics unchanged,
//! O.3 row 49): the ball is computed by bounded direct sparse scans over
//! the windowed records using the existing `Vector` set operations
//! (`ewise_add_lor` union + `difference` complement mask) — identical
//! `LOR_LAND` hop semantics to `hop_ball` over the window candidate
//! `A_one`, without materializing the full-window candidate `PairMatrix`
//! (whose `insert` build was quadratic). Admission is one fused pass with
//! the identical predicate conjunction. The vertex set equals the ball
//! exactly: admission requires `s, r in ball`, focal endpoints seed the
//! ball, and masking only shrinks the admitted set, never the ball.

use std::fmt;

use super::primitives::ewise::ewise_add_lor;
use super::pruning::prune_time;
use super::support::{build_support, SupportTensors};
use super::types::{
    EdgeRef, EligiblePairs, FocalTransaction, HopBound, SelfLoopRule, TimeWindow, TransactionRecord,
};
use super::vector::Vector;
use crate::api::CanonicalTensor;
use crate::TensorError;

/// Parameters of one focal-context construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextParameters {
    pub window: TimeWindow,
    pub hop_bound: HopBound,
    pub eligible: Option<EligiblePairs>,
    pub self_loop: SelfLoopRule,
}

impl ContextParameters {
    /// Scope goes to parameters (window from `omega_secs`).
    #[must_use]
    pub fn from_scope(focal_epoch: u64, scope: &super::types::ValidationScope) -> Self {
        Self {
            window: TimeWindow::backward_inclusive(focal_epoch, scope.omega_secs),
            hop_bound: scope.hop_bound,
            eligible: scope.eligible.clone(),
            self_loop: scope.self_loop,
        }
    }
}

/// The canonical focal context `N_f`.
///
/// `admitted_features` is parallel to `admitted`: index `i` holds the
/// pre-computed derived feature vector `(log_magnitude, relative_threshold_distance,
/// roundedness_flags)` of `admitted[i]`, carried verbatim from the commit-time
/// derivation (GS §9 / Amendment A1). Populated only by the pipeline-backed
/// constructors; the tensor-only `build_focal_context` leaves it empty (the
/// canonical tensor stores exact micro-unit values, not the derived metrics).
#[derive(Clone, Debug)]
pub struct FocalContext {
    pub focal: FocalTransaction,
    pub params: ContextParameters,
    pub admitted: Vec<TransactionRecord>,
    pub admitted_features: Vec<crate::graphblas::types::FeatureVector>,
    pub vertices: Vector,
    pub tensors: SupportTensors,
}
/// Canonical error (no panics on missing data; v5 contract preserved).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    VolumeOverflow(TensorError),
    FocalOutsideWindow,
    /// `T(t_f, s_f, r_f) = 0`: the focal coordinate is outside `supp(T)`
    /// (section A3) — no focal transaction exists in the canonical tensor
    /// at this coordinate.
    FocalAbsent,
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VolumeOverflow(o) => write!(f, "volume sum overflowed i64: {o}"),
            Self::FocalOutsideWindow => write!(f, "focal epoch outside window"),
            Self::FocalAbsent => write!(
                f,
                "focal coordinate (t_f, s_f, r_f) is outside supp(T) (section A3)"
            ),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<TensorError> for ExecutionError {
    fn from(o: TensorError) -> Self {
        Self::VolumeOverflow(o)
    }
}

/// Build the canonical focal context from the canonical CSF tensor.
///
/// # Errors
/// [`ExecutionError`] on focal window/absence violations or volume overflow.
pub fn build_focal_context(
    csf: &CanonicalTensor,
    focal: FocalTransaction,
    params: &ContextParameters,
) -> Result<FocalContext, ExecutionError> {
    if !params.window.contains(focal.epoch) {
        return Err(ExecutionError::FocalOutsideWindow);
    }
    super::projection::extract_3d(csf, focal.epoch, focal.sender.0, focal.receiver.0)
        .ok_or(ExecutionError::FocalAbsent)?;
    let time_kept: Vec<TransactionRecord> = prune_time(csf, params.window);
    let mut ball = Vector::zero();
    ball.insert(focal.sender.0);
    ball.insert(focal.receiver.0);
    let mut frontier = ball.clone();
    for _ in 0..params.hop_bound.0 {
        if frontier.is_empty() {
            break;
        }
        let mut stepped = Vector::zero();
        let mut fwd: Vec<u32> = Vec::new();
        let mut bwd: Vec<u32> = Vec::new();
        for r in &time_kept {
            if frontier.contains(r.s) {
                fwd.push(r.r);
            }
            if frontier.contains(r.r) {
                bwd.push(r.s);
            }
        }
        let mut fwd_v = Vector::zero();
        fwd_v.extend_unsorted(fwd);
        let mut bwd_v = Vector::zero();
        bwd_v.extend_unsorted(bwd);
        stepped = ewise_add_lor(&stepped, &fwd_v);
        stepped = ewise_add_lor(&stepped, &bwd_v);
        let next = stepped.difference(&ball);
        if next.is_empty() {
            break;
        }
        ball = ewise_add_lor(&ball, &next);
        frontier = next;
    }
    let exclude_self = params.self_loop == SelfLoopRule::Exclude;
    let mut admitted: Vec<TransactionRecord> = Vec::with_capacity(time_kept.len().min(1024));
    for r in time_kept {
        if !ball.contains(r.s) || !ball.contains(r.r) {
            continue;
        }
        if exclude_self && r.s == r.r {
            continue;
        }
        if let Some(e) = params.eligible.as_ref() {
            if !e.admits(EdgeRef::new(r.s, r.r)) {
                continue;
            }
        }
        admitted.push(r);
    }
    let tensors = build_support(&admitted)?;
    let vertices: Vector = ball.clone();
    Ok(FocalContext {
        focal,
        params: params.clone(),
        admitted,
        admitted_features: Vec::new(),
        vertices,
        tensors,
    })
}
