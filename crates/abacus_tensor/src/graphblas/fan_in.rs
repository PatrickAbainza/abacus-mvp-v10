//! Fan-in observable (#21–#22, row 50): `d-(v_f)` over `A_one` (golden O.3).
//!
//! Ported from v7 `graphblas/fan_in.rs` (itself from v5 `fan_in`; label
//! isolation preserved: takes `&FocalTransaction` only — no label
//! parameter exists at this layer).

use super::types::{FocalTransaction, MetricScalar, SupportSet, VertexId};

/// Incoming support at `v` over `A_one`: pattern of the sparse column
/// `A_one[:, v]`, honoring the context self-loop rule.
#[must_use]
pub fn incoming_support(context: &super::context::FocalContext, v: VertexId) -> SupportSet {
    let mut members = context.tensors.a_one.senders_to(v.0);
    if context.params.self_loop == super::types::SelfLoopRule::Exclude {
        members.retain(|&u| u != v.0);
    }
    SupportSet::from_sorted(members)
}

/// `compute_fan_in`: extract incoming support at `v_f`, reduce cardinality
/// (#10).
///
/// # Errors
/// [`super::context::ExecutionError`] (contract parity; fan-in itself is a
/// pure count and cannot fail).
pub fn compute_fan_in(
    context: &super::context::FocalContext,
    focal: &FocalTransaction,
) -> Result<FanInResult, super::context::ExecutionError> {
    let support = incoming_support(context, focal.receiver);
    let count = super::primitives::reduce::reduce_count(&support.members);
    Ok(FanInResult {
        focal: *focal,
        support_counterparties: support,
        fan_in: MetricScalar::from_count(count),
    })
}

/// The fan-in scalar + witness support.
#[derive(Clone, Debug, PartialEq)]
pub struct FanInResult {
    pub focal: FocalTransaction,
    pub support_counterparties: SupportSet,
    pub fan_in: MetricScalar,
}
