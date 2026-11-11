//! Learned-threshold deterministic rule over Abacus behavioral state.
//!
//! Governing addendum: `docs/golden_source_rule_discovery_addendum.md`
//! (R1–R14). This module realizes **only** what that addendum derives:
//!
//! ```text
//! events → CSF tensor T[t,s,r]                  rows 5/11 (§S7)
//!   ↓ one row-16 restriction per day-window      R4, P1 half-open
//!   ↓ BehavioralRelation                          B4  (UNCHANGED, reused)
//!   ↓ ONE rule: 3 atoms, one ∧                   R3, semantic §7.6/§18
//! TRUE / FALSE
//! ```
//!
//! Constraints honoured here:
//!
//! * The canonical tensor is `T[t,s,r]` and is **never** re-materialised. No
//!   second tensor, no dense temporal tensor, no snapshot graph (Time rule).
//! * **No graph**: the only grouping is the exact `BTreeSet` union of admitted
//!   sets, discarded by the caller (Set-operation rule, HCSF §8/§10).
//! * **No label is read anywhere in this module** (P11, R9). The sidecar is not
//!   referenced, imported, or named. Scoring lives in the probe's caller.
//! * **No ML**: no model, no weights, no optimiser, no gradient, no feature
//!   vector. Threshold selection is an exhaustive scan of a finite candidate set
//!   under a fixed total order (R5, R6).
//! * **No second rule, no rule engine**: four comparisons and one `∧` (R3).
//! * All metric comparisons are **exact rationals** on `i128`; `f64` appears
//!   only in rendering, never in a condition or a selection (P11, R6, R8.3).

use std::collections::BTreeSet;

use crate::behavioral::BehavioralRelation;
use crate::tensor::coordinate::Coord;

/// Seconds in one declared day-window (R4). The split is by exact event
/// timestamp; `t` is never bucketed into anything coarser than this by the
/// algorithm, and this bucketing lives in the **algorithm**, not the storage.
pub const DAY_SECONDS: u64 = 86_400;

/// Declared day-window count of the SAML-D 90-day profile (R4).
pub const TOTAL_DAYS: u32 = 90;

/// Declared training horizon: Days 1–80 inclusive (R4).
pub const TRAIN_DAYS: u32 = 80;

/// The vacuous "no constraint" value for `temporal_span <= Δ` (R5).
///
/// Declared, not observed: `u64::MAX` admits every representable span. Chosen
/// over an infinity sentinel so the comparison stays a total `u64` test (P11:
/// no float, no `Option` in the hot path).
pub const DELTA_UNBOUNDED: u64 = u64::MAX;

/// Declared rank-fraction denominator for the quantile ladders (R5.2/R5.3).
/// Seventeen points: `{0, 1/16, 2/16, …, 15/16, 1}`.
pub const LADDER_STEPS: usize = 16;

/// The minimal shape the rule needs from a relation (R9: no label, no tensor).
///
/// Implemented for `BehavioralRelation`; kept as a trait so the predicted-set
/// builder cannot accidentally acquire access to anything richer — in
/// particular, it exposes no `hop`, no `window`, and no label.
pub trait BehavioralFact {
    /// `|A_hop|` — admitted-entry cardinality.
    fn event_count(&self) -> u64;
    /// Exact micro-unit sum over the admitted set.
    fn aggregate_amount(&self) -> i64;
    /// `max t − min t` over the admitted set, epoch seconds.
    fn temporal_span(&self) -> u64;
}

impl BehavioralFact for BehavioralRelation {
    fn event_count(&self) -> u64 {
        self.event_count
    }
    fn aggregate_amount(&self) -> i64 {
        self.aggregate_amount
    }
    fn temporal_span(&self) -> u64 {
        self.temporal_span
    }
}

/// The three learned thresholds `C`, `A`, `Δ` (R3).
///
/// These are **declared parameters** in the sense of semantic §7.5: the rule
/// `event_count >= C ∧ aggregate_amount >= A ∧ temporal_span <= Δ` is a
/// predicate whose *truth value* changes with them. Nothing else about the rule
/// changes. `aggregate_amount` is in exact micro-units (P2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Thresholds {
    /// `C` — minimum admitted-entry cardinality `|A_hop|`.
    pub c: u64,
    /// `A` — minimum exact micro-unit sum over the admitted set.
    pub a: i64,
    /// `Δ` — maximum `max t − min t` over the admitted set, epoch seconds.
    pub delta: u64,
}

impl Thresholds {
    /// Build the triple.
    #[must_use]
    pub const fn new(c: u64, a: i64, delta: u64) -> Self {
        Self { c, a, delta }
    }

    /// The three atoms in declaration order (R3): `P1`, `P2`, `P3`.
    #[must_use]
    pub fn atoms(&self, r: &BehavioralRelation) -> [bool; 3] {
        [
            r.event_count >= self.c,
            r.aggregate_amount >= self.a,
            r.temporal_span <= self.delta,
        ]
    }

    /// **R3 — the one rule.** Total: every relation gets exactly one of
    /// `TRUE`/`FALSE` (semantic §7.6; L4). Short-circuit `∧` is `a_17`.
    ///
    /// This function takes a relation and three thresholds and **nothing else**.
    /// It cannot see a label, a day index, a window, or any tensor value
    /// (R9). `hop` is deliberately not read — the task fixes the rule at three
    /// atoms (B6's `P4` is dropped).
    #[must_use]
    pub fn evaluate(&self, r: &BehavioralRelation) -> bool {
        let [a, b, c] = self.atoms(r);
        a && b && c
    }

    /// Compound result vector, aligned with the relation vector.
    #[must_use]
    pub fn evaluate_all(&self, rels: &[BehavioralRelation]) -> Vec<bool> {
        rels.iter().map(|r| self.evaluate(r)).collect()
    }
}

impl std::fmt::Display for Thresholds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "event_count >= {} AND aggregate_amount >= {} AND temporal_span <= {}",
            self.c, self.a, self.delta
        )
    }
}

/// The exact `Predicted` set for one window (R8.1, Set-operation rule):
/// `⋃ { A_hop(r) : rule(r) = TRUE }`.
///
/// `admitted[i]` is the admitted-coordinate set of `rels[i]`. The union is over
/// *coordinates*, so a coordinate admitted by several firing relations is
/// counted **once** (B4 `A_2(A,C)` union-over-witnessing-legs semantics).
/// Discarded by the caller when the window is scored (HCSF §8/§10).
#[must_use]
pub fn predicted_set(
    rels: &[BehavioralRelation],
    admitted: &[&BTreeSet<(Coord, i64)>],
    th: &Thresholds,
) -> BTreeSet<Coord> {
    let mut out = BTreeSet::new();
    for (r, a) in rels.iter().zip(admitted) {
        if th.evaluate(r) {
            out.extend(a.iter().map(|(c, _)| *c));
        }
    }
    out
}

/// A confusion matrix over a period's records (R8.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Confusion {
    /// `|Predicted ∩ Truth|`
    pub tp: u64,
    /// `|Predicted \ Truth|`
    pub fp: u64,
    /// `|Records \ (Predicted ∪ Truth)|`
    pub tn: u64,
    /// `|Truth \ Predicted|`
    pub fn_: u64,
}

impl Confusion {
    /// Total records scored in the period.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.tp + self.fp + self.tn + self.fn_
    }

    /// Number of positives in the period.
    #[must_use]
    pub fn positives(&self) -> u64 {
        self.tp + self.fn_
    }

    /// Number of predicted positives in the period (`|Predicted|`).
    #[must_use]
    pub fn predicted(&self) -> u64 {
        self.tp + self.fp
    }

    /// `TP/(TP+FP)`; `0/1` when no prediction was made.
    #[must_use]
    pub fn precision(&self) -> (i128, i128) {
        Self::ratio(self.tp, self.tp + self.fp)
    }

    /// `TP/(TP+FN)`; `0/1` when there is no positive to catch.
    #[must_use]
    pub fn recall(&self) -> (i128, i128) {
        Self::ratio(self.tp, self.tp + self.fn_)
    }

    /// `2·TP/(2·TP+FP+FN)`; `0/1` when the denominator is zero.
    #[must_use]
    pub fn f1(&self) -> (i128, i128) {
        Self::ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn_)
    }

    /// `FP/(FP+TN)`; `0/1` when there is no negative to misfire on.
    #[must_use]
    pub fn fpr(&self) -> (i128, i128) {
        Self::ratio(self.fp, self.fp + self.tn)
    }

    /// `|Predicted| / |Records|` (R8.4) — the fraction of the period the rule
    /// flags. Distinct from the count of events *processed* (5A.7).
    #[must_use]
    pub fn coverage(&self) -> (i128, i128) {
        Self::ratio(self.predicted(), self.total())
    }

    /// Declared zero-denominator convention: an undefined ratio is `0/1`, and
    /// `0/1` compares as strictly less than any positive rational (R6).
    fn ratio(n: u64, d: u64) -> (i128, i128) {
        (i128::from(n), i128::from(d.max(1)))
    }
}

/// Compare two exact rationals `a/b` and `c/d` by cross-multiplication on
/// `i128` (P11: no float anywhere in the selection path).
#[must_use]
pub fn cmp_rational(a: (i128, i128), b: (i128, i128)) -> std::cmp::Ordering {
    let (a_n, a_d) = a;
    let (b_n, b_d) = b;
    (a_n * b_d).cmp(&(b_n * a_d))
}

/// Render an exact rational as an `f64` for the printed report **only**
/// (P11: floats are presentation-only; this is never called from `evaluate`,
/// from candidate generation, or from selection).
#[must_use]
pub fn render(r: (i128, i128)) -> f64 {
    if r.1 == 0 { 0.0 } else { r.0 as f64 / r.1 as f64 }
}

impl Confusion {
    /// Render every required metric, in the order the task's report lists them.
    #[must_use]
    pub fn report(&self) -> String {
        format!(
            "TP={} FP={} TN={} FN={} | precision={:.4} recall={:.4} F1={:.4} \
             FPR={:.6} coverage={:.6}",
            self.tp,
            self.fp,
            self.tn,
            self.fn_,
            render(self.precision()),
            render(self.recall()),
            render(self.f1()),
            render(self.fpr()),
            render(self.coverage()),
        )
    }
}

/// The declared finite candidate set for the three axes (R5).
///
/// Each axis is ascending and deduped. Every element is either an **observed**
/// Days 1–80 value or a **declared vacuous** constant (`1` for `C`,
/// `DELTA_UNBOUNDED` for `Δ`). The `A` axis is a nearest-rank ladder over
/// observed values, so its elements are observed too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidates {
    /// `C` ladder: distinct observed `event_count` values ∪ `{1}`.
    pub c: Vec<u64>,
    /// `A` ladder: 17-point nearest-rank ladder over observed `aggregate_amount`.
    pub a: Vec<i64>,
    /// `Δ` ladder: 17-point nearest-rank ladder over observed `temporal_span`
    /// ∪ `{DELTA_UNBOUNDED}`.
    pub delta: Vec<u64>,
}

impl Candidates {
    /// Total number of candidate rules (the product, R5).
    #[must_use]
    pub fn len(&self) -> usize {
        self.c.len() * self.a.len() * self.delta.len()
    }

    /// Always false — the candidate set is never empty by construction (a day
    /// of this profile always yields at least one relation).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ascending-dedup helper (`a_16.2` ordered-set semantics).
    fn dedup_asc<T: Ord + Copy>(mut v: Vec<T>) -> Vec<T> {
        v.sort_unstable();
        v.dedup();
        v
    }

    /// **R5 — build the candidate set from Days 1–80 observed values only.**
    ///
    /// `counts`, `amounts` and `spans` are the multisets of the three B4 fields
    /// over **every** relation derived from **Days 1–80**. Holdout values are
    /// structurally absent: the caller has no way to pass them (R4.1).
    ///
    /// The `A` and `Δ` ladders use **nearest-rank** on the ascending sorted
    /// multiset at fixed rank fractions `k/16`, `k = 0..16`, computed with
    /// integer arithmetic only (P11). Nearest-rank (not interpolation) means
    /// every ladder element is an **observed** value, never a synthetic mean.
    #[must_use]
    pub fn build(counts: &[u64], amounts: &[i64], spans: &[u64]) -> Self {
        // Axis 1 — every distinct observed count, plus the vacuous `1`.
        let mut c = counts.to_vec();
        c.push(1);
        let c = Self::dedup_asc(c);

        // Axis 2 — 17-point nearest-rank ladder over observed amounts.
        let a = Self::ladder_i64(amounts, LADDER_STEPS);
        // Axis 3 — 17-point nearest-rank ladder over observed spans, plus the
        // declared unconstrained sentinel.
        let mut delta = Self::ladder_u64(spans, LADDER_STEPS);
        delta.push(DELTA_UNBOUNDED);
        let delta = Self::dedup_asc(delta);

        Self { c, a, delta }
    }

    /// Nearest-rank ladder over an ascending-sorted `i64` multiset.
    ///
    /// Rank of fraction `k/steps` is `floor(k·(n-1)/steps)` — integer
    /// arithmetic, no float (P11). An empty input yields an empty ladder.
    #[must_use]
    pub fn ladder_i64(values: &[i64], steps: usize) -> Vec<i64> {
        let mut v = values.to_vec();
        v.sort_unstable();
        Self::ladder_sorted_i64(&v, steps)
    }

    /// [`Self::ladder_i64`] over `u64` values.
    #[must_use]
    pub fn ladder_u64(values: &[u64], steps: usize) -> Vec<u64> {
        let mut v = values.to_vec();
        v.sort_unstable();
        Self::ladder_sorted_u64(&v, steps)
    }

    fn ladder_sorted_i64(v: &[i64], steps: usize) -> Vec<i64> {
        if v.is_empty() {
            return Vec::new();
        }
        let n = (v.len() - 1) as u128;
        let mut out = Vec::with_capacity(steps + 1);
        for k in 0..=steps {
            let idx = (k as u128 * n / steps as u128) as usize;
            out.push(v[idx.min(v.len() - 1)]);
        }
        Self::dedup_asc(out)
    }

    fn ladder_sorted_u64(v: &[u64], steps: usize) -> Vec<u64> {
        if v.is_empty() {
            return Vec::new();
        }
        let n = (v.len() - 1) as u128;
        let mut out = Vec::with_capacity(steps + 1);
        for k in 0..=steps {
            let idx = (k as u128 * n / steps as u128) as usize;
            out.push(v[idx.min(v.len() - 1)]);
        }
        Self::dedup_asc(out)
    }

    /// Enumerate the candidate set in the **fixed order** `(Δ, C, A)` — the
    /// enumeration order referenced by selection tie-break 5 (R6).
    #[must_use]
    pub fn enumerate(&self) -> Vec<Thresholds> {
        let mut out = Vec::with_capacity(self.len());
        for &delta in &self.delta {
            for &c in &self.c {
                for &a in &self.a {
                    out.push(Thresholds::new(c, a, delta));
                }
            }
        }
        out
    }
}

/// A scored candidate: the thresholds, the position in the enumeration order,
/// how many relations fired, and the resulting confusion matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scored {
    /// The candidate's three thresholds.
    pub th: Thresholds,
    /// Its 0-based index in [`Candidates::enumerate`] order.
    pub index: usize,
    /// How many relations the rule fired on, over the scored period.
    pub fired: u64,
    /// How many relations the rule was evaluated on, over the scored period.
    pub total: u64,
    /// The transaction-level confusion matrix (R8.3).
    pub cm: Confusion,
}

impl Scored {
    /// **R6 — the selection key, compared component-wise in order.**
    ///
    /// 1. maximise F1, 2. maximise precision, 3. maximise recall,
    /// 4. maximise coverage, 5. minimise the enumeration index.
    ///
    /// Returns `true` when `self` is strictly preferred to `other`. Rule 5
    /// makes the order **total** (indices are unique), so exactly one candidate
    /// wins (R6).
    #[must_use]
    pub fn preferred_to(&self, other: &Scored) -> bool {
        use std::cmp::Ordering;
        let keys = |s: &Scored| {
            [s.cm.f1(), s.cm.precision(), s.cm.recall(), s.cm.coverage()]
        };
        let (a, b) = (keys(self), keys(other));
        for (x, y) in a.into_iter().zip(b) {
            match cmp_rational(x, y) {
                Ordering::Greater => return true,
                Ordering::Less => return false,
                Ordering::Equal => {}
            }
        }
        // Tie-break 5: the lower enumeration index wins. Total order.
        self.index < other.index
    }

    /// **R3.1 — non-vacuity.** `true` when the rule fired on at least one
    /// relation and fewer than all of them. A rule that is all-TRUE or
    /// all-FALSE verifies nothing and is a failed experiment.
    #[must_use]
    pub fn is_non_vacuous(&self) -> bool {
        self.fired > 0 && self.fired < self.total
    }
}

/// Select exactly one candidate under the R6 criterion.
///
/// Total by construction (R6 tie-break 5): the returned element is the unique
/// lexicographic maximum over the whole enumerated set.
#[must_use]
pub fn select(scored: &[Scored]) -> Option<Scored> {
    let mut best: Option<Scored> = None;
    for s in scored {
        best = Some(match best {
            None => *s,
            Some(b) if s.preferred_to(&b) => *s,
            Some(b) => b,
        });
    }
    best
}

/// The declared selection criterion, copied verbatim into the frozen artifact
/// (R7). Any change to this string is a change to the experiment.
pub const SELECTION_CRITERION: &str = concat!(
    "lexicographic max over (F1, precision, recall, coverage) on Days 1-80, ",
    "each compared as an exact rational; ties broken by minimal enumeration ",
    "index in fixed (delta, C, A) ascending order"
);

/// The declared chronological split, copied verbatim into the frozen artifact.
pub const SPLIT_DECLARATION: &str = concat!(
    "W_k = [t_min + (k-1)*86400, t_min + k*86400) for k in 1..=90; ",
    "train k in 1..=80, holdout k in 81..=90, full k in 1..=90; ",
    "chronological, not shuffled"
);
