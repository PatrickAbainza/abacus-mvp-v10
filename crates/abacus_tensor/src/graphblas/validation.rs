//! Oracle validation (#23, row 51): `compare_with_oracle` (golden O.3).
//!
//! Ported from v7 `graphblas/validation.rs` (itself from v5 `validation`):
//! the oracle is caller-supplied — this module takes the scalar, never the
//! tensor. Diagnostic infrastructure, not new algebra. Per
//! `your_task_phase_7.md` section 4 it stays in test/validation
//! infrastructure only and is never wired into the production algebraic
//! execution path.

use super::context::FocalContext;
use super::fan_in::FanInResult;

/// Oracle comparison outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleComparison {
    pub canonical: u64,
    pub oracle: u64,
    pub matches: bool,
    pub diagnostic: Option<ValidationDiagnostic>,
}

/// Structured mismatch diagnostic (carries the focal canonical coordinate —
/// the static tensor has no `tx_id`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationDiagnostic {
    pub focal_sender: u32,
    pub focal_receiver: u32,
    pub focal_epoch: u64,
    pub window_start: u64,
    pub window_end: u64,
    pub hop_bound: u64,
    pub admitted_coordinates: Vec<(u64, u32, u32)>,
    pub support: Vec<u32>,
    pub oracle_result: u64,
    pub canonical_result: u64,
}

/// Compare the canonical fan-in against the caller-supplied oracle scalar.
#[must_use]
pub fn compare_with_oracle(
    context: &FocalContext,
    result: &FanInResult,
    oracle_fan_in: u64,
) -> OracleComparison {
    let matches = result.fan_in.count == oracle_fan_in;
    let diagnostic = if matches {
        None
    } else {
        let mut admitted_coordinates: Vec<(u64, u32, u32)> =
            context.admitted.iter().map(|r| (r.t, r.s, r.r)).collect();
        admitted_coordinates.sort_unstable();
        let support = result.support_counterparties.members.clone();
        Some(ValidationDiagnostic {
            focal_sender: result.focal.sender.0,
            focal_receiver: result.focal.receiver.0,
            focal_epoch: result.focal.epoch,
            window_start: context.params.window.start,
            window_end: context.params.window.end,
            hop_bound: context.params.hop_bound.0,
            admitted_coordinates,
            support,
            oracle_result: oracle_fan_in,
            canonical_result: result.fan_in.count,
        })
    };
    OracleComparison {
        canonical: result.fan_in.count,
        oracle: oracle_fan_in,
        matches,
        diagnostic,
    }
}
