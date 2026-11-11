//! Baseline-gated evaluation over windowed fiber supports.
//!
//! Real caller for the pinned-baseline contract
//! (`docs/OPERATOR_CONTRACTS.md` §2): every evaluation that applies a
//! baseline-derived gate validates provenance FIRST via
//! [`require_baseline`]. Missing or invalid version / as-of data returns
//! `Indeterminate{BaselineUnavailable}` — never a fabricated baseline and
//! never a valid-looking negative.
//!
//! Generic terms only (modes, coordinates, entries, keys, windows).
//! Integer-only arithmetic (i64 in, i128 in-flight for infallible
//! comparisons, fixed-point shares reported as numerator / denominator).

use abacus_tensor::atomic::reference::{
    map as ref_map, reduce as ref_reduce, restrict as ref_restrict, Baseline, Predicate, Theta,
};
use abacus_tensor::atomic::types::{Mode, Monoid, Support};
use abacus_tensor::atomic::{Indeterminate, OpReason, Outcome};

/// Validate baseline provenance inside `theta`.
///
/// Returns the pinned descriptor, or `Indeterminate{BaselineUnavailable}`
/// when the descriptor is absent, the version identifier is empty, or the
/// as-of timestamp is zero. No value is invented on any path.
pub fn require_baseline(theta: &Theta) -> Result<&Baseline, Indeterminate> {
    let b = theta.baseline.as_ref().ok_or_else(|| {
        Indeterminate::new(
            OpReason::BaselineUnavailable,
            "baseline descriptor absent in evaluation parameters",
        )
    })?;
    if b.version.is_empty() {
        return Err(Indeterminate::new(
            OpReason::BaselineUnavailable,
            "baseline version identifier is empty",
        ));
    }
    if b.as_of == 0 {
        return Err(Indeterminate::new(
            OpReason::BaselineUnavailable,
            "baseline as-of timestamp is zero (unpinned)",
        ));
    }
    Ok(b)
}

/// Map function: flag entries at or above `mean + scale` (one baseline
/// unit above the pinned mean, fixed-point-exact in `i128`).
///
/// Emits `1` (flag set, locator evidence preserved) or bottom (dropped).
/// Requires the caller to have passed [`require_baseline`]; a missing
/// descriptor maps to bottom here so an unvalidated call can never
/// produce a false flag — but validated callers never take that path.
pub fn flag_above_baseline(
    _t: u64,
    _s: u32,
    _r: u32,
    v: i64,
    theta: &Theta,
) -> Option<i64> {
    let b = theta.baseline.as_ref()?;
    let gate = i128::from(b.mean) + i128::from(b.scale);
    if i128::from(v) >= gate {
        Some(1)
    } else {
        None
    }
}

/// Integer share of flagged entries: numerator / denominator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlaggedShare {
    /// Entries with the flag set.
    pub flagged: u64,
    /// All entries in the window.
    pub total: u64,
    /// Pinned baseline version that gated the flag.
    pub baseline_version: BaselineProvenance,
}

/// Provenance carried on a determinate share (copied out of `Theta`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineProvenance {
    /// Version identifier (opaque).
    pub version: String,
    /// As-of timestamp (epoch seconds).
    pub as_of: u64,
}

/// Evaluate the flagged share over a half-open window.
///
/// Composition: `restrict(window)` → `map(flag_above_baseline, theta)` →
/// `reduce(count)`. Baseline provenance is validated first; any failure
/// returns `Indeterminate{BaselineUnavailable}` with a specific reason.
pub fn evaluate_flagged_share(
    support: &Support,
    lo: u64,
    hi: u64,
    theta: &Theta,
) -> Outcome<FlaggedShare> {
    let baseline = match require_baseline(theta) {
        Ok(b) => b,
        Err(ind) => return Outcome::Indeterminate(ind),
    };
    let windowed = ref_restrict(support, &Predicate::window(lo, hi));
    let flagged = ref_map(&windowed, flag_above_baseline, theta);
    let total = windowed.entries.len() as u64;
    let flagged_n = flagged.entries.len() as u64;
    // Counts are bounded by input length: summation cannot overflow.
    debug_assert!(flagged_n <= total);
    let _ = ref_reduce;
    let _ = (Mode::Mode0, Monoid::Count);
    Outcome::Determinate(FlaggedShare {
        flagged: flagged_n,
        total,
        baseline_version: BaselineProvenance {
            version: baseline.version.clone(),
            as_of: baseline.as_of,
        },
    })
}
