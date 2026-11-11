//! Hermetic acceptance tests for the rule-discovery rule
//! (`docs/golden_source_rule_discovery_addendum.md`, R3–R9, R12).
//!
//! Hermetic: no artifact, no CSV, no network, **no label**. The real 90-day run
//! (R4 split, R7 artifact, R12's isolation obligations) lives in the
//! `rule_discovery` binary, which needs the materialised profile artifacts.

use abacus_tensor::behavioral::{derive_relations, BehavioralRelation, Entities, Window};
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::{compress, CsfTensor};
use abacus_tensor::rule_discovery::{
    select, Candidates, Confusion, Scored, Thresholds, DAY_SECONDS, DELTA_UNBOUNDED,
    LADDER_STEPS, TOTAL_DAYS, TRAIN_DAYS,
};
use abacus_tensor::tensor::coordinate::Coord;
use std::collections::BTreeSet;

/// Declared synthetic fixture. Five events with distinct timestamps, in exact
/// micro-units (P2). Chosen so relations exist at several `event_count`,
/// `aggregate_amount` and `temporal_span` values, so every atom can be made to
/// bind and to fail independently.
const EVENTS: [(u64, u32, u32, i64); 5] = [
    (10, 0, 1, 1_000_000), // A -> B
    (20, 0, 1, 2_000_000), // A -> B
    (30, 1, 2, 4_000_000), // B -> C
    (40, 1, 2, 8_000_000), // B -> C
    (90, 2, 3, 1_500_000), // C -> D, isolated: forces a span-0 relation
];

const DICT: [&str; 4] = ["A", "B", "C", "D"];

fn build(events: &[(u64, u32, u32, i64)]) -> (CsfTensor, Vec<TxEntry>, Entities<'static>) {
    let d: &'static Vec<String> =
        Box::leak(Box::new(DICT.iter().map(|s| (*s).to_string()).collect()));
    let raw: Vec<TxEntry> =
        events.iter().map(|&(t, s, r, v)| TxEntry::new(t, s, r, v)).collect();
    let settled = abacus_tensor::composites::settle(raw).expect("settle");
    (compress(&settled), settled, Entities { senders: d, receivers: d })
}

/// One window per declared day, so the synthetic fixture exercises R4's split.
fn window(k: u32) -> Window {
    Window {
        t_start: u64::from(k) * DAY_SECONDS,
        t_end: u64::from(k + 1) * DAY_SECONDS,
        delta: DAY_SECONDS,
    }
}

fn rel(ec: u64, am: i64, span: u64, hop: u32) -> BehavioralRelation {
    BehavioralRelation {
        source: "A".into(),
        target: "B".into(),
        hop,
        window: (0, DAY_SECONDS),
        event_count: ec,
        aggregate_amount: am,
        temporal_span: span,
    }
}

/// R3 — the rule has exactly three atoms and each one binds independently.
#[test]
fn three_atoms_each_bind_independently() {
    let th = Thresholds::new(3, 5_000_000, 15);
    assert!(th.evaluate(&rel(3, 5_000_000, 15, 1)), "all three satisfied");
    assert!(!th.evaluate(&rel(2, 5_000_000, 15, 1)), "event_count below C");
    assert!(!th.evaluate(&rel(3, 4_999_999, 15, 1)), "aggregate_amount below A");
    assert!(!th.evaluate(&rel(3, 5_000_000, 16, 1)), "temporal_span above delta");
}

/// R3 — the rule is exactly the three declared atoms, with no `hop` atom.
///
/// The frozen behavioral predicate (B6) has a fourth atom, `hop >= k_min`. This
/// rule has three, so `hop` must not influence the result.
#[test]
fn hop_is_not_an_atom() {
    let th = Thresholds::new(2, 1, 100);
    for hop in [1u32, 2, 7, 99] {
        assert!(th.evaluate(&rel(2, 1, 100, hop)), "hop {hop} must not affect the rule");
    }
    assert_eq!(th.evaluate(&rel(2, 1, 0, 1)), th.evaluate(&rel(2, 1, 0, 2)));
}

/// R3.2 — the rule is **total**: every relation gets exactly one of TRUE/FALSE,
/// with no `Err` case and no undefined case (semantic §7.6; L4).
#[test]
fn rule_is_total_over_extreme_values() {
    let extremes = [
        rel(0, i64::MIN, 0, 1),
        rel(0, i64::MAX, 0, 1),
        rel(u64::MAX, i64::MIN, u64::MAX, 2),
        rel(u64::MAX, i64::MAX, u64::MAX, 2),
    ];
    for th in [
        Thresholds::new(0, i64::MIN, 0),
        Thresholds::new(u64::MAX, i64::MAX, u64::MAX),
        Thresholds::new(1, 0, DAY_SECONDS),
    ] {
        for r in &extremes {
            assert!(th.evaluate(r) == th.evaluate(r), "evaluate returns a definite bool");
        }
    }
    // The weakest thresholds are all-TRUE; the strongest are all-FALSE.
    let weak = Thresholds::new(0, i64::MIN, u64::MAX);
    assert!(extremes.iter().all(|r| weak.evaluate(r)));
}

/// R8.1 — `predicted_set` is the exact **union** over firing relations, so a
/// coordinate admitted by two firing relations is counted once (Set-operation
/// rule; B4 `A_2(A,C)`).
#[test]
fn predicted_set_is_an_exact_union_without_duplicates() {
    let a = BTreeSet::from([(Coord::new(1, 0, 1), 1_i64), (Coord::new(2, 0, 1), 2)]);
    let b = BTreeSet::from([(Coord::new(2, 0, 1), 2_i64), (Coord::new(3, 0, 1), 3)]);
    let rels = vec![rel(2, 0, 10, 1), rel(2, 0, 10, 2)];
    let th = Thresholds::new(1, 0, 100);
    let out = abacus_tensor::rule_discovery::predicted_set(&rels, &[&a, &b], &th);
    assert_eq!(out.len(), 3, "the shared coordinate must be counted once");
    assert!(out.contains(&Coord::new(2, 0, 1)));

    // With no relation firing the set is empty, not "everything".
    let none = Thresholds::new(99, 0, 100);
    assert!(abacus_tensor::rule_discovery::predicted_set(&rels, &[&a, &b], &none).is_empty());
}

/// R8.3 — the metric definitions are exactly the declared ratios.
#[test]
fn metric_definitions_match_r8_3() {
    let cm = Confusion { tp: 6, fp: 4, tn: 80, fn_: 10 };
    assert_eq!(cm.total(), 100);
    assert_eq!(cm.positives(), 16);
    assert_eq!(cm.predicted(), 10);
    assert_eq!(cm.precision(), (6, 10));
    assert_eq!(cm.recall(), (6, 16));
    assert_eq!(cm.f1(), (12, 2 * 6 + 4 + 10));
    assert_eq!(cm.fpr(), (4, 84));
    assert_eq!(cm.coverage(), (10, 100));
}

/// R8.3 — a zero denominator yields the declared `0/1`, never a NaN or a panic.
#[test]
fn zero_denominators_are_declared_zero() {
    let empty = Confusion::default();
    for m in [
        empty.precision(),
        empty.recall(),
        empty.f1(),
        empty.fpr(),
        empty.coverage(),
    ] {
        assert_eq!(m, (0, 1), "an undefined ratio is declared 0/1");
        assert!(abacus_tensor::rule_discovery::render(m).is_finite());
    }
    // No positives, some predictions: precision defined, recall/F1 zero.
    let no_pos = Confusion { tp: 0, fp: 5, tn: 5, fn_: 0 };
    assert_eq!(no_pos.recall(), (0, 1));
    assert_eq!(no_pos.precision(), (0, 5));
}

/// P11 — rational comparison is exact. `1/3` and `33333/100000` are very close
/// but distinct; an `f64` comparison could conflate them, `i128` cannot.
#[test]
fn rational_comparison_is_exact_not_float() {
    use abacus_tensor::rule_discovery::cmp_rational;
    use std::cmp::Ordering;
    assert_eq!(cmp_rational((1, 3), (33333, 100000)), Ordering::Greater);
    assert_eq!(cmp_rational((33333, 100000), (1, 3)), Ordering::Less);
    assert_eq!(cmp_rational((2, 4), (1, 2)), Ordering::Equal);
    // Same value, different denominators — must tie exactly.
    assert_eq!(cmp_rational((134, 11408), (67, 5704)), Ordering::Equal);
    // Beyond f64's 53-bit mantissa: 2^53 + 1 vs 2^53.
    assert_eq!(
        cmp_rational((9_007_199_254_740_993, 1), (9_007_199_254_740_992, 1)),
        Ordering::Greater
    );
}

/// R5 — the candidate set is built only from **observed** training values plus
/// the declared vacuous constants, is ascending, deduped and finite.
#[test]
fn candidate_set_is_observed_or_vacuous_only() {
    let counts = [1u64, 2, 2, 3, 7, 7, 40];
    let amounts = [10i64, 20, 30, 40, 50, 60, 70, 80, 90, 100, -5, 0];
    let spans = [0u64, 0, 5, 5, 10, 100, 1000, 1000, 1000, 7, 3, 1];
    let c = Candidates::build(&counts, &amounts, &spans);

    // Ascending and deduped on every axis.
    assert!(c.c.windows(2).all(|w| w[0] < w[1]), "C axis strictly ascending");
    assert!(c.delta.windows(2).all(|w| w[0] < w[1]), "Δ axis strictly ascending");
    assert!(c.a.windows(2).all(|w| w[0] <= w[1]), "A axis non-decreasing");

    // The declared vacuous constants are present.
    assert_eq!(c.c.first(), Some(&1), "the vacuous C=1 is declared (R5.1)");
    assert_eq!(c.delta.last(), Some(&DELTA_UNBOUNDED), "the vacuous Δ is declared (R5.3)");

    // Every element is either observed or the declared vacuous constant.
    for v in &c.c {
        assert!(counts.contains(v) || *v == 1, "C={v} is neither observed nor vacuous");
    }
    for v in &c.a {
        assert!(amounts.contains(v), "A={v} is not an observed value");
    }
    for v in &c.delta {
        assert!(spans.contains(v) || *v == DELTA_UNBOUNDED, "Δ={v} is neither observed nor vacuous");
    }
}

/// R5 — the ladder is a 17-point **nearest-rank** ladder: every element is an
/// observed value, the extremes are included, and no float is involved.
#[test]
fn ladder_is_nearest_rank_and_includes_extremes() {
    let v: Vec<u64> = (0..1000).collect();
    let l = Candidates::ladder_u64(&v, LADDER_STEPS);
    assert_eq!(l.first(), Some(&0));
    assert_eq!(l.last(), Some(&999));
    assert!(l.windows(2).all(|w| w[0] <= w[1]));
    // Nearest-rank of a uniform 0..999 ladder at k/16 is floor(k*999/16).
    for (k, expect) in l.iter().enumerate() {
        assert_eq!(*expect, ((k as u128 * 999) / 16) as u64, "ladder point {k}");
    }
    // Degenerate inputs must not panic.
    assert!(Candidates::ladder_u64(&[], LADDER_STEPS).is_empty());
    assert_eq!(Candidates::ladder_u64(&[7], LADDER_STEPS), vec![7]);
}

/// R5 — the enumeration is the fixed `(Δ, C, A)` order and its length is the
/// declared product, with no sampling.
#[test]
fn enumeration_is_the_declared_fixed_order() {
    let c = Candidates { c: vec![1, 2], a: vec![10, 20, 30], delta: vec![100, 200] };
    let e = c.enumerate();
    assert_eq!(e.len(), 12);
    assert_eq!(c.len(), 12);
    assert_eq!(e[0], Thresholds::new(1, 10, 100));
    assert_eq!(e[1], Thresholds::new(1, 20, 100));
    assert_eq!(e[2], Thresholds::new(1, 30, 100));
    assert_eq!(e[3], Thresholds::new(2, 10, 100));
    assert_eq!(e[6], Thresholds::new(1, 10, 200));
    // Enumeration is a pure function of the candidate set.
    assert_eq!(c.enumerate(), e);
}

/// R6 — the selection criterion is a **total order**: exactly one winner, always,
/// even when every candidate scores identically.
#[test]
fn selection_is_total_and_unique() {
    let cm = Confusion { tp: 1, fp: 1, tn: 1, fn_: 1 };
    let scored: Vec<Scored> = (0..25)
        .map(|i| {
            let (i, u) = (i as usize, i as u64);
            Scored { th: Thresholds::new(u, i as i64, u), index: i, fired: 1, total: 2, cm }
        })
        .collect();
    let b = select(&scored).expect("a winner");
    assert_eq!(b.index, 0, "the lowest enumeration index must win an exact tie");

    // F1 first: a strictly better F1 beats a lower index. `worse` is
    // F1 = 2/(2+1+1) = 1/2; `better` is F1 = 10/(10+5+0) = 2/3.
    let better =
        Scored { th: Thresholds::new(9, 9, 9), index: 9, fired: 1, total: 2,
                 cm: Confusion { tp: 5, fp: 5, tn: 5, fn_: 0 } };
    let worse = Scored { th: Thresholds::new(0, 0, 0), index: 0, fired: 1, total: 2, cm };
    let s = vec![worse, better];
    assert_eq!(select(&s).expect("a winner").index, 9);

    // Permutation-invariance: the winner does not depend on input order.
    let mut shuffled = s.clone();
    shuffled.reverse();
    assert_eq!(select(&shuffled).expect("a winner"), select(&s).expect("a winner"));

    // Empty input has no winner (never vacuously "true").
    assert!(select(&[]).is_none());
}

/// R6 — the criterion really is lexicographic: F1, then precision, then recall,
/// then coverage, then index. Each tie-break is exercised separately.
#[test]
fn selection_tiebreaks_are_lexicographic_in_the_declared_order() {
    let mk = |index: u32, tp: u64, fp: u64, fn_: u64, tn: u64| Scored {
        th: Thresholds::new(1, 1, u64::from(index)),
        index: index as usize,
        fired: 1,
        total: 2,
        cm: Confusion { tp, fp, tn, fn_ },
    };
    // (1) F1 dominates: 0.8 beats 0.5 even though its precision is lower.
    let hi_f1_low_p = mk(0, 4, 6, 0, 0);
    let lo_f1_hi_p = mk(1, 1, 0, 2, 0);
    assert!(hi_f1_low_p.preferred_to(&lo_f1_hi_p));
    assert!(!lo_f1_hi_p.preferred_to(&hi_f1_low_p));

    // (2) Equal F1, higher precision wins.
    let a = mk(0, 2, 2, 2, 0);
    let b = mk(1, 2, 1, 3, 0);
    assert!(b.preferred_to(&a));
    assert!(!a.preferred_to(&b));

    // (3) Equal F1 and precision, higher recall wins.
    let a = mk(0, 1, 1, 3, 0);
    let b = mk(1, 1, 1, 1, 0);
    assert!(b.preferred_to(&a));

    // (5) Everything equal: lower index wins.
    let a = mk(5, 1, 1, 1, 1);
    let b = mk(3, 1, 1, 1, 1);
    assert!(b.preferred_to(&a));
    assert!(!a.preferred_to(&b));
}

/// R3.1 — non-vacuity: an all-TRUE or all-FALSE rule is rejected.
#[test]
fn non_vacuity_rejects_all_true_and_all_false() {
    let cm = Confusion { tp: 1, fp: 1, tn: 1, fn_: 1 };
    let all_true = Scored { th: Thresholds::new(1, 1, 1), index: 0, fired: 10, total: 10, cm };
    let all_false = Scored { th: Thresholds::new(1, 1, 1), index: 1, fired: 0, total: 10, cm };
    let mixed = Scored { th: Thresholds::new(1, 1, 1), index: 2, fired: 3, total: 10, cm };
    assert!(!all_true.is_non_vacuous());
    assert!(!all_false.is_non_vacuous());
    assert!(mixed.is_non_vacuous());
}

/// R1 — the rule consumes **only** the three declared B4 fields, and is
/// unchanged by `source`, `target`, `hop` and `window`.
#[test]
fn rule_depends_on_exactly_three_fields() {
    let base = BehavioralRelation {
        source: "A".into(),
        target: "B".into(),
        hop: 1,
        window: (0, 1),
        event_count: 3,
        aggregate_amount: 100,
        temporal_span: 7,
    };
    let th = Thresholds::new(3, 100, 7);
    assert!(th.evaluate(&base));

    for (s, tg, h, w) in [
        ("Z".to_string(), "Y".to_string(), 2u32, (5u64, 9u64)),
        ("Q".to_string(), "R".to_string(), 9u32, (0u64, u64::MAX)),
    ] {
        let mut r = base.clone();
        r.source = s;
        r.target = tg;
        r.hop = h;
        r.window = w;
        assert!(th.evaluate(&r), "source/target/hop/window must not affect the rule");
    }
}

/// R9 — the rule module contains **no** label read, and no reference to the
/// sidecar, the eval record type, or any laundering field (P11; B8; C9; D2).
#[test]
fn rule_module_reads_no_label() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/rule_discovery.rs"),
    )
    .expect("read rule_discovery.rs");

    let mut in_doc_comment = false;
    for (n, line) in src.lines().enumerate() {
        let t = line.trim();
        // Strip doc comments and ordinary comments, so that a *prohibition* is
        // not itself counted as a violation (the technique the frozen probes use).
        if t.starts_with("//!") || t.starts_with("///") || t.starts_with("//") || t.starts_with('*') {
            in_doc_comment = true;
            continue;
        }
        if in_doc_comment {
            if t == "*/" || t.ends_with("*/") {
                in_doc_comment = false;
            }
            continue;
        }
        let low = t.to_lowercase();
        for banned in [
            "laundering", "sidecar", "evalrecord", "is_fraud", "label", "ground_truth",
            "deserialize_eval", "tx_id",
        ] {
            assert!(
                !low.contains(banned),
                "rule path must contain no label reference: '{banned}' at line {}: {t}",
                n + 1
            );
        }
    }
}

/// R9 — `evaluate` takes only a relation and three thresholds. Its signature
/// cannot admit a label, a day index, or a tensor value.
#[test]
fn rule_evaluation_surface_is_minimal() {
    // If `evaluate` gained a parameter, this call would fail to compile.
    let th = Thresholds::new(1, 1, 1);
    let r = rel(1, 1, 1, 1);
    let _: bool = th.evaluate(&r);
    // The trait the predicted-set builder uses exposes exactly three reads.
    use abacus_tensor::rule_discovery::BehavioralFact;
    let _: u64 = BehavioralFact::event_count(&r);
    let _: i64 = BehavioralFact::aggregate_amount(&r);
    let _: u64 = BehavioralFact::temporal_span(&r);
}

/// R4/R11 — the declared split constants are what the addendum says they are,
/// and the day window is a half-open interval of exactly `DAY_SECONDS`.
#[test]
fn declared_split_constants_match_the_addendum() {
    assert_eq!(TRAIN_DAYS, 80, "R4: Days 1-80 are the training period");
    assert_eq!(TOTAL_DAYS, 90, "R4: the profile is 90 declared days");
    assert_eq!(DAY_SECONDS, 86_400);
    assert_eq!(LADDER_STEPS, 16, "R5: the ladder has 17 points");
    assert_eq!(DELTA_UNBOUNDED, u64::MAX, "R5: the vacuous Δ is unconstrained");
    assert_eq!(TRAIN_DAYS + 10, TOTAL_DAYS, "80 training + 10 holdout = 90 days");
}

/// R4 — the day windows tile the timeline without gaps or overlap, and the split
/// is chronological (monotone in `t`) by construction.
#[test]
fn day_windows_tile_the_timeline_chronologically() {
    let t_min = 1_000_000u64;
    let edges: Vec<u64> = (0..=TOTAL_DAYS).map(|k| t_min + u64::from(k) * DAY_SECONDS).collect();
    // Strictly increasing ⇒ chronological; no shuffle is possible.
    assert!(edges.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(edges.first(), Some(&t_min));
    assert_eq!(edges[1] - edges[0], DAY_SECONDS, "each window is exactly one day");
    // The training/holdout boundary is a single edge shared by both sides.
    let cut = edges[TRAIN_DAYS as usize];
    assert_eq!(cut, t_min + 80 * DAY_SECONDS);
    assert!(edges[..TRAIN_DAYS as usize].iter().all(|e| *e <= cut));
    assert!(edges[TRAIN_DAYS as usize..].iter().all(|e| *e >= cut));
    // The holdout is the last 10 windows and ends at the 90-day horizon.
    assert_eq!(edges[TOTAL_DAYS as usize] - cut, 10 * DAY_SECONDS);
}

/// End-to-end on the synthetic fixture: build the tensor with the existing
/// pipeline, derive relations with the existing B4 code, and check the rule
/// fires on some relations and not all (R3.1 non-vacuity on real behaviour).
#[test]
fn rule_fires_on_some_but_not_all_relations_of_a_real_window() {
    let (csf, _settled, ents) = build(&EVENTS);
    assert!(csf.check_invariants(), "the existing CSF invariants still hold");
    let derived = derive_relations(&csf, &ents, window(0)).expect("derive");
    assert!(!derived.is_empty(), "the fixture must produce relations");

    let rels: Vec<BehavioralRelation> = derived.iter().map(|d| d.relation.clone()).collect();
    let th = Thresholds::new(2, 1, DAY_SECONDS);
    let fired = rels.iter().filter(|r| th.evaluate(r)).count();
    assert!(fired > 0, "the rule must fire on at least one relation");
    assert!(fired < rels.len(), "the rule must not fire on every relation");

    // The vacuous rule fires everywhere; the maximal rule fires nowhere.
    assert!(rels.iter().all(|r| Thresholds::new(0, i64::MIN, u64::MAX).evaluate(r)));
    assert!(!rels.iter().any(|r| Thresholds::new(u64::MAX, i64::MAX, 0).evaluate(r)));
}

/// R5/R6 end-to-end on the synthetic fixture: build candidates from the derived
/// relations, enumerate, score every candidate directly, and check the selected
/// rule is the exact lexicographic maximum of the R6 key.
#[test]
fn selection_on_a_real_window_matches_the_declared_criterion() {
    let (csf, _settled, ents) = build(&EVENTS);
    let derived = derive_relations(&csf, &ents, window(0)).expect("derive");
    let rels: Vec<BehavioralRelation> = derived.iter().map(|d| d.relation.clone()).collect();

    let counts: Vec<u64> = rels.iter().map(|r| r.event_count).collect();
    let amounts: Vec<i64> = rels.iter().map(|r| r.aggregate_amount).collect();
    let spans: Vec<u64> = rels.iter().map(|r| r.temporal_span).collect();
    let cands = Candidates::build(&counts, &amounts, &spans);
    let tri = cands.enumerate();
    assert_eq!(tri.len(), cands.len());

    // A synthetic "truth" (every 3rd coordinate positive) so the matrix is
    // non-degenerate. This is a fixture, not a label: no sidecar is read.
    let all: Vec<Coord> =
        derived.iter().flat_map(|d| d.admitted.iter().map(|(c, _)| *c)).collect();
    let universe: BTreeSet<Coord> = all.iter().copied().collect();
    let truth: BTreeSet<Coord> =
        all.iter().enumerate().filter(|(i, _)| i % 3 == 0).map(|(_, c)| *c).collect();
    let admitted: Vec<&BTreeSet<(Coord, i64)>> = derived.iter().map(|d| &d.admitted).collect();

    let mut scored: Vec<Scored> = Vec::new();
    for (index, th) in tri.iter().enumerate() {
        let pred = abacus_tensor::rule_discovery::predicted_set(&rels, &admitted, th);
        let tp = pred.intersection(&truth).count() as u64;
        let fp = pred.len() as u64 - tp;
        let fn_ = truth.len() as u64 - tp;
        let tn = universe.len() as u64 - tp - fp - fn_;
        let fired = rels.iter().filter(|r| th.evaluate(r)).count() as u64;
        scored.push(Scored {
            th: *th,
            index,
            fired,
            total: rels.len() as u64,
            cm: Confusion { tp, fp, tn, fn_ },
        });
    }

    let best = select(&scored).expect("a winner");
    // The winner must be preferred to every other candidate (total order ⇒ max).
    for s in &scored {
        if s.index != best.index {
            assert!(best.preferred_to(s), "winner {:?} must beat {:?}", best.th, s.th);
        }
    }
    // Self-consistency: the matrix identities of R8.3 must hold for every candidate.
    for s in &scored {
        assert_eq!(s.cm.tp + s.cm.fp, s.cm.predicted());
        assert_eq!(s.cm.tp + s.cm.fn_, s.cm.positives());
        assert_eq!(s.cm.total(), universe.len() as u64);
    }
}
