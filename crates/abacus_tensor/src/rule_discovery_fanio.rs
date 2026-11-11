//! Fan-in / fan-out rule discovery (Rule Discovery Experiment #2).
//!
//! Governing addendum: `docs/golden_source_rule_discovery_fanio_addendum.md`
//! (F1–F14). This module realizes **only** what that addendum derives:
//!
//! ```text
//! events → CSF tensor T[t,s,r]                     rows 5/11 (§S7)
//!   ↓ one row-16 restriction per day-window         F6, P1 half-open
//!   ↓ existing behavioral state (hop-1 admitted)    B4  (UNCHANGED, reused)
//!   ↓ existing fan-in / fan-out projection          F4  (FV §3/§4)
//!   ↓ ONE rule per quantity: one atom, `>=`         F3, semantic §7.6
//! TRUE / FALSE
//! ```
//!
//! Constraints honoured here:
//!
//! * The canonical tensor is `T[t,s,r]` and is **never** re-materialised. No
//!   second tensor, no dense temporal tensor, no snapshot graph (Time rule).
//! * **No graph**: the only structure is the transient pair projection
//!   `A_one|_W` (an exact set of `(sender, receiver)` pairs) and the exact
//!   `BTreeSet` of predicted coordinates, both discarded per window
//!   (Set-operation rule, HCSF §8/§10). No adjacency, no edge list, no node
//!   table is retained.
//! * **No label is read anywhere in this module** (P11, F9). The sidecar is not
//!   referenced, imported, or named. Scoring lives in the binary's caller.
//! * **No ML**: no model, no weights, no optimiser, no gradient, no feature
//!   vector. Threshold selection is an exhaustive scan of a finite candidate
//!   set under a fixed total order (F7, F8).
//! * **No new primitive**: the fan-in / fan-out quantities are the existing
//!   ones (FV §4 `rin` / `sout`), provably equal to the existing exact set
//!   primitives `samld::sets::fan_in_senders` / `samld::sets::co_receivers`.
//! * Metric comparisons are **exact rationals** on `i128`; `f64` appears only
//!   in rendering, never in a condition or a selection (P11, F8).

use std::collections::{BTreeMap, BTreeSet};

use crate::rule_discovery::{cmp_rational, Confusion};

pub use crate::rule_discovery::{DAY_SECONDS, SPLIT_DECLARATION, TOTAL_DAYS, TRAIN_DAYS};

/// The declared vacuous-sentinel offset (F7): for a `>=` atom the vacuous
/// "no record can satisfy it" value is `(max observed value) + 1`.
pub const VACUOUS_OFFSET: u64 = 1;

/// The two quantities of this experiment (F3). They are tested **separately**
/// and are never combined in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Quantity {
    /// `|in(r)|` over the windowed admitted support (FV §3/§4, `rin`).
    FanIn,
    /// `|out(s)|` over the windowed admitted support (FV §3/§4, `sout`).
    FanOut,
}

impl Quantity {
    /// The declared name, used in the frozen artifact and the report.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FanIn => "fan_in",
            Self::FanOut => "fan_out",
        }
    }
}

/// The key of a transaction coordinate that the quantity is read at.
///
/// Fan-in is a function of the receiver; fan-out is a function of the sender.
/// This is exactly the FV §4 reading (`in(r_f)`, `out(s_f)`).
#[must_use]
pub const fn key(quantity: Quantity, s: u32, r: u32) -> u32 {
    match quantity {
        Quantity::FanIn => r,
        Quantity::FanOut => s,
    }
}

/// The `>=` threshold rule, one per quantity — never conjoined (F3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FanRule {
    /// Which quantity the atom reads.
    pub quantity: Quantity,
    /// `T` — the learned threshold.
    pub threshold: u64,
}

impl FanRule {
    /// Build the one-atom rule.
    #[must_use]
    pub const fn new(quantity: Quantity, threshold: u64) -> Self {
        Self { quantity, threshold }
    }

    /// **F3 — the one rule.** Total: every value gets exactly one of
    /// `TRUE`/`FALSE` (semantic §7.6; L4). Takes a quantity value and nothing
    /// else — it cannot see a label, a day index, a window, or a tensor value.
    #[must_use]
    pub const fn evaluate(&self, value: u64) -> bool {
        value >= self.threshold
    }

    /// Evaluate at a vertex key, reading the precomputed per-vertex count map.
    /// An absent key has fan-in / fan-out `0` (FV §1: absent entries read `0`).
    #[must_use]
    pub fn evaluate_at(&self, counts: &BTreeMap<u32, u64>, k: u32) -> bool {
        self.evaluate(counts.get(&k).copied().unwrap_or(0))
    }
}

impl std::fmt::Display for FanRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} >= {}", self.quantity.as_str(), self.threshold)
    }
}

/// **F4 — the existing fan-in quantity, bulk.** `A_one|_W` is the existence
/// projection of the windowed admitted support `E_W` (FV §1): the exact set of
/// distinct `(receiver, sender)` pairs. Its per-receiver column cardinality is
/// `|in(r)|` (FV §3 `mxv`, §4 `rin`; GraphBLAS #21 + #10).
///
/// `entries` must be the windowed support `E_W` (each canonical coordinate
/// once). Order-independent; the result is an exact set projection.
#[must_use]
pub fn fan_in_counts(entries: &[(u64, u32, u32)]) -> BTreeMap<u32, u64> {
    // A_one|_W as an exact pair set — keyed (receiver, sender) so that the
    // per-receiver count is a contiguous run over the ordered set.
    let pairs: BTreeSet<(u32, u32)> = entries.iter().map(|&(_, s, r)| (r, s)).collect();
    let mut out: BTreeMap<u32, u64> = BTreeMap::new();
    for (r, _s) in pairs {
        *out.entry(r).or_insert(0) += 1;
    }
    out
}

/// **F4 — the existing fan-out quantity, bulk.** The same `A_one|_W` pair set
/// read by row: per-sender cardinality is `|out(s)|` (FV §3 `vxm`, §4 `sout`;
/// GraphBLAS #5 + #11).
#[must_use]
pub fn fan_out_counts(entries: &[(u64, u32, u32)]) -> BTreeMap<u32, u64> {
    let pairs: BTreeSet<(u32, u32)> = entries.iter().map(|&(_, s, r)| (s, r)).collect();
    let mut out: BTreeMap<u32, u64> = BTreeMap::new();
    for (s, _r) in pairs {
        *out.entry(s).or_insert(0) += 1;
    }
    out
}

/// **F4 — reference equality to the existing set primitive for fan-in.**
///
/// `|samld::sets::fan_in_senders(E, r, w0, w1)|` — the cardinality of the
/// existing exact set primitive's shell. `fan_in_counts(E)[r]` MUST equal this
/// for every `r` when `[w0, w1)` covers `E`; a hermetic test asserts it.
#[must_use]
pub fn fan_in_at(entries: &[(u64, u32, u32)], r: u32, w0: u64, w1: u64) -> u64 {
    u64::try_from(crate::samld::sets::fan_in_senders(entries, r, w0, w1).len()).unwrap_or(u64::MAX)
}

/// **F4 — reference equality to the existing set primitive for fan-out.**
///
/// `|samld::sets::co_receivers(E, s, w0, w1)|`.
#[must_use]
pub fn fan_out_at(entries: &[(u64, u32, u32)], s: u32, w0: u64, w1: u64) -> u64 {
    u64::try_from(crate::samld::sets::co_receivers(entries, s, w0, w1).len()).unwrap_or(u64::MAX)
}

/// **F7 — build the candidate threshold ladder from observed values only.**
///
/// Ascending-distinct observed values, deduped, plus the declared vacuous
/// sentinel `(max observed) + 1` (the all-FALSE threshold). Values below `1`
/// are dropped: `fan_in` and `fan_out` are always `>= 1` (F3.1), so `0` is not
/// a rule. An empty input yields the single sentinel `1`.
#[must_use]
pub fn candidates(observed: &[u64]) -> Vec<u64> {
    let mut v: Vec<u64> = observed.iter().copied().filter(|&x| x >= 1).collect();
    v.sort_unstable();
    v.dedup();
    // The sentinel is the one declared non-observed value (all-FALSE).
    let sentinel = observed.iter().copied().max().unwrap_or(0).saturating_add(VACUOUS_OFFSET);
    match v.binary_search(&sentinel) {
        Ok(_) => v,
        Err(at) => {
            v.insert(at, sentinel);
            v
        }
    }
}

/// **F8 — the selection key, compared component-wise in order.**
///
/// 1. maximise F1, 2. maximise precision, 3. maximise recall,
/// 4. maximise coverage, 5. minimise the enumeration index.
///
/// Rule 5 makes the order **total** (indices are unique), so exactly one
/// candidate wins (F8).
#[must_use]
pub fn preferred_to(a: &ScoredFan, b: &ScoredFan) -> bool {
    use std::cmp::Ordering;
    let keys = |s: &ScoredFan| [s.cm.f1(), s.cm.precision(), s.cm.recall(), s.cm.coverage()];
    let (ka, kb) = (keys(a), keys(b));
    for (x, y) in ka.into_iter().zip(kb) {
        match cmp_rational(x, y) {
            Ordering::Greater => return true,
            Ordering::Less => return false,
            Ordering::Equal => {}
        }
    }
    a.index < b.index
}

/// A scored candidate: the one-atom rule, its position in the ascending
/// enumeration order, how many records it flagged, and the confusion matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoredFan {
    /// The candidate's rule.
    pub rule: FanRule,
    /// Its 0-based index in the ascending candidate order.
    pub index: usize,
    /// Records the rule flagged over the scored period (`|Predicted|`).
    pub fired: u64,
    /// Records the rule was evaluated on over the scored period.
    pub total: u64,
    /// The transaction-level confusion matrix (F11).
    pub cm: Confusion,
}

impl ScoredFan {
    /// **F3.1 — non-vacuity.** `true` when the rule flagged at least one record
    /// and fewer than all of them. All-TRUE or all-FALSE verifies nothing.
    #[must_use]
    pub fn is_non_vacuous(&self) -> bool {
        self.fired > 0 && self.fired < self.total
    }
}

/// **F8 — select exactly one candidate.** Total by construction (tie-break 5):
/// the returned element is the unique lexicographic maximum of the whole
/// enumerated set.
#[must_use]
pub fn select1(scored: &[ScoredFan]) -> Option<ScoredFan> {
    let mut best: Option<ScoredFan> = None;
    for s in scored {
        best = Some(match best {
            None => *s,
            Some(b) if preferred_to(s, &b) => *s,
            Some(b) => b,
        });
    }
    best
}

/// A compact, exact value histogram `value -> (records, positives)` over one
/// period, with ascending suffix-sum reads — the accelerator that makes the
/// exhaustive threshold sweep affordable without sampling.
///
/// This is an accelerator, not the definition: the binary re-derives the
/// selected rule's matrix by the direct route and requires exact agreement.
#[derive(Debug, Clone, Default)]
pub struct ValueHist {
    /// Ascending distinct observed values.
    pub values: Vec<u64>,
    /// `records_at[i]` = records whose quantity equals `values[i]`.
    pub records_at: Vec<u64>,
    /// `positives_at[i]` = positives whose quantity equals `values[i]`.
    pub positives_at: Vec<u64>,
    /// Total records accumulated.
    pub records: u64,
    /// Total positives accumulated.
    pub positives: u64,
}

impl ValueHist {
    /// Accumulate one `(value, records, positives)` observation (`a_16.2`
    /// ordered-map insert; ascending-key order maintained).
    pub fn add(&mut self, value: u64, records: u64, positives: u64) {
        match self.values.binary_search(&value) {
            Ok(at) => {
                self.records_at[at] += records;
                self.positives_at[at] += positives;
            }
            Err(at) => {
                self.values.insert(at, value);
                self.records_at.insert(at, records);
                self.positives_at.insert(at, positives);
            }
        }
        self.records += records;
        self.positives += positives;
    }

    /// **Exact `>=` evaluation.** Records and positives with quantity `>= t` —
    /// a suffix sum over the ascending ladder (`partition_point`, `a_12`).
    /// `t` above every observed value yields `(0, 0)` (all-FALSE).
    #[must_use]
    pub fn at_least(&self, t: u64) -> (u64, u64) {
        let first = self.values.partition_point(|&v| v < t);
        let records: u64 = self.records_at[first..].iter().sum();
        let positives: u64 = self.positives_at[first..].iter().sum();
        (records, positives)
    }

    /// Number of distinct observed values.
    #[must_use]
    pub fn distinct(&self) -> usize {
        self.values.len()
    }

    /// **F7 — the admissible candidate surface.**
    ///
    /// A threshold is admissible iff its rule is **non-vacuous on this period**
    /// (F3.1): it flags at least one and strictly fewer than all records.
    /// Used as a **diagnostic** (task §"Final sanity check"), not a gate: the
    /// frozen candidate surface is the full ladder (F7), the criterion is
    /// applied exactly to it, and non-vacuity is reported.
    #[must_use]
    pub fn admissible(&self, ladder: &[u64]) -> Vec<u64> {
        ladder
            .iter()
            .copied()
            .filter(|&t| {
                let (predicted, _) = self.at_least(t);
                predicted > 0 && predicted < self.records
            })
            .collect()
    }

    /// **The exact cumulative ladder** — for every observed value `v` in
    /// ascending order, `(v, records_ge_v, positives_ge_v)`.
    ///
    /// This is the complete information needed to evaluate `>= v` for **every
    /// possible threshold** (task §"cumulative counts/rates needed to evaluate
    /// every possible threshold"). Suffix sums only; no float, no sampling.
    #[must_use]
    pub fn cumulative_ladder(&self) -> Vec<(u64, u64, u64)> {
        let n = self.values.len();
        let mut out = Vec::with_capacity(n);
        let (mut rec, mut pos) = (0u64, 0u64);
        for i in (0..n).rev() {
            rec += self.records_at[i];
            pos += self.positives_at[i];
            out.push((self.values[i], rec, pos));
        }
        out.reverse();
        out
    }
}

/// The declared selection criterion, copied verbatim into the frozen artifact
/// (F10). Any change to this string is a change to the experiment.
pub const SELECTION_CRITERION: &str = concat!(
    "lexicographic max over (F1, precision, recall, coverage) on Days 1-80, ",
    "each compared as an exact rational; ties broken by minimal enumeration ",
    "index in ascending threshold order; fan-in and fan-out selected ",
    "independently"
);


