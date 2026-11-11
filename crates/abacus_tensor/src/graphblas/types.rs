//! Minimal domain types for the #14–#23 pipeline (golden Addendum O, O.3).
//!
//! Ported from the v7 subset actually required by the pipeline (itself from
//! v5 `abacus_canonical_execution::types`): vertex/edge ids, half-open
//! `TimeWindow`, `HopBound`, focal coordinate `(t,s,r)` (P1/P3 — the static
//! tensor carries no event ids), explicit `SelfLoopRule`, caller-supplied
//! `EligiblePairs` mask, support pattern, fan-in scalar, and the
//! `ValidationScope` convenience bundle. `LabelledFocal`,
//! `DimensionMapping`-style metadata, and predicate-layer types are not
//! ported (still excluded, no section).

/// A vertex of the transaction graph (sender/receiver dictionary code).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexId(pub u32);

/// A structural directed pair `(sender -> receiver)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeRef {
    pub sender: VertexId,
    pub receiver: VertexId,
}

impl EdgeRef {
    #[must_use]
    pub const fn new(sender: u32, receiver: u32) -> Self {
        Self {
            sender: VertexId(sender),
            receiver: VertexId(receiver),
        }
    }

    #[must_use]
    pub const fn is_self_loop(&self) -> bool {
        self.sender.0 == self.receiver.0
    }
}

/// Canonical half-open temporal window `[start, end)` in epoch seconds
/// (P1 order; identical contract to row-27 `time_slice`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeWindow {
    pub start: u64,
    pub end: u64,
}

impl TimeWindow {
    /// `(t_f - w, t_f]` encoded as the half-open `[t_f - w, t_f + 1)`.
    #[must_use]
    pub fn backward_inclusive(focal_epoch: u64, omega_secs: u64) -> Self {
        Self {
            start: focal_epoch.saturating_sub(omega_secs),
            end: focal_epoch.saturating_add(1),
        }
    }

    #[must_use]
    pub const fn contains(&self, epoch: u64) -> bool {
        epoch >= self.start && epoch < self.end
    }
}

/// Structural hop bound `k` of focal-context extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HopBound(pub u64);

/// The focal transaction under validation, identified by its canonical
/// coordinate `(t, s, r)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FocalTransaction {
    pub epoch: u64,
    pub sender: VertexId,
    pub receiver: VertexId,
}

/// A set of distinct counterparties (support level): sorted, distinct
/// vertex codes — the pattern of a sparse Boolean vector.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SupportSet {
    pub members: Vec<u32>,
}

impl SupportSet {
    /// Wrap an already-sorted distinct member list.
    #[must_use]
    pub fn from_sorted(members: Vec<u32>) -> Self {
        debug_assert!(members.windows(2).all(|w| w[0] < w[1]));
        Self { members }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Binary search for `v` in the pattern.
    #[must_use]
    pub fn contains(&self, v: VertexId) -> bool {
        self.members.binary_search(&v.0).is_ok()
    }
}

/// The scalar fan-in observable (exact count lifted to `f64`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricScalar {
    pub count: u64,
    pub value: f64,
}

impl MetricScalar {
    #[must_use]
    pub fn from_count(count: u64) -> Self {
        Self {
            count,
            value: count as f64,
        }
    }
}

/// Explicit self-loop domain rule. Never silently assumed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SelfLoopRule {
    Admit,
    #[default]
    Exclude,
}

/// Caller-supplied eligible-pair predicate, stored as data: sorted,
/// distinct `(sender, receiver)` codes; membership is binary search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EligiblePairs {
    pub pairs: Vec<(u32, u32)>,
}

impl EligiblePairs {
    /// Wrap an already-sorted distinct pair list.
    #[must_use]
    pub fn from_sorted(pairs: Vec<(u32, u32)>) -> Self {
        debug_assert!(pairs.windows(2).all(|w| w[0] < w[1]));
        Self { pairs }
    }

    #[must_use]
    pub fn admits(&self, e: EdgeRef) -> bool {
        self.pairs
            .binary_search(&(e.sender.0, e.receiver.0))
            .is_ok()
    }
}

/// Scope bundle restricting candidacy (structural knobs only; no numeric
/// thresholds live here).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationScope {
    pub omega_secs: u64,
    pub hop_bound: HopBound,
    pub eligible: Option<EligiblePairs>,
    pub self_loop: SelfLoopRule,
}

/// One canonical windowed entry: `(t, s, r, value)` (row-27 record).
pub type TransactionRecord = crate::api::TensorEntry;

/// Pre-computed derived feature vector for one transaction
/// `(log_magnitude, relative_threshold_distance, roundedness_flags)`.
///
/// GS §9 / Amendment A1: carried verbatim from the commit-time derivation.
/// The canonical tensor stores the exact micro-unit `value`, not these
/// metrics, so this is a pipeline-supplied parallel to the admitted record.
/// A tensor-only `build_focal_context` has no source for these and returns
/// an empty feature list; the pipeline-backed evaluators populate them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FeatureVector {
    pub log_magnitude: u16,
    pub relative_threshold_distance: i32,
    pub roundedness_flags: u8,
}
