//! Synthetic acceptance tests for the behavioral-fact / declarative-condition
//! boundary (addendum `docs/golden_source_behavioral_rule_boundary_addendum.md`,
//! B11.1–B11.4 and B11.6). The real-data run (B11.8) lives in the
//! `rules_probe` binary, which needs the materialised profile artifacts.
//!
//! These tests are hermetic: no artifact, no CSV, no network, no label.

use abacus_tensor::behavioral::{
    derive_relations, reference_relations, Entities, RawEvent, Thresholds, Window,
};
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::{compress, no_self_aggregation, CsfTensor};
use abacus_tensor::csf::query::time_slice;
use std::collections::BTreeSet;

/// The declared synthetic set (task §"Synthetic test"), in exact micro-units.
const EVENTS: [RawEvent; 5] = [
    (1, 0, 1, 10_000_000), // t=1 A -> B
    (2, 0, 1, 20_000_000), // t=2 A -> B
    (3, 1, 2, 30_000_000), // t=3 B -> C
    (4, 2, 3, 40_000_000), // t=4 C -> D
    (5, 0, 1, 50_000_000), // t=5 A -> B
];

const W: Window = Window { t_start: 1, t_end: 6, delta: 2 };

/// Build the probe tensor exactly as the pipeline does: events → row 5
/// `d_settle` (P4) → row 11 `compress` (P6).
fn build(events: &[RawEvent]) -> (CsfTensor, Vec<TxEntry>, Entities<'static>) {
    static DICT: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let d = DICT.get_or_init(|| ["A", "B", "C", "D"].iter().map(|s| (*s).to_string()).collect());
    let raw: Vec<TxEntry> = events
        .iter()
        .map(|(t, s, r, v)| TxEntry::new(*t, *s, *r, *v))
        .collect();
    let settled = abacus_tensor::composites::settle(raw).expect("settle");
    (
        compress(&settled),
        settled,
        Entities { senders: d, receivers: d },
    )
}

fn rel_of<'a>(
    rels: &'a [abacus_tensor::behavioral::DerivedRelation],
    s: &str,
    t: &str,
    hop: u32,
) -> Option<&'a abacus_tensor::behavioral::BehavioralRelation> {
    rels.iter()
        .find(|d| d.relation.source == s && d.relation.target == t && d.relation.hop == hop)
        .map(|d| &d.relation)
}

/// B11.1 — CSF invariants and no-self-aggregation.
#[test]
fn csf_construction_invariants_hold() {
    let (csf, settled, _) = build(&EVENTS);
    assert!(csf.check_invariants(), "P6 structural invariants");
    assert!(no_self_aggregation(&settled, &csf), "nnz == |COO| and decode == COO");
    assert_eq!(csf.nnz(), settled.len());
}

/// B11.2 — the row-16 windowed restriction equals the reference filter.
#[test]
fn temporal_selection_is_exact() {
    let (csf, settled, _) = build(&EVENTS);
    let sel = time_slice(&csf, W.t_start, W.t_end).entries;
    let filtered: Vec<TxEntry> = settled
        .iter()
        .filter(|e| e.t >= W.t_start && e.t < W.t_end)
        .copied()
        .collect();
    assert_eq!(sel, filtered);
}

/// B11.3 — `A -> B` has three events and the exact temporal span.
#[test]
fn first_hop_derives_three_events_and_exact_span() {
    let (csf, _, ents) = build(&EVENTS);
    let rels = derive_relations(&csf, &ents, W).expect("derive");
    let ab = rel_of(&rels, "A", "B", 1).expect("A -> B hop 1");
    assert_eq!(ab.event_count, 3);
    assert_eq!(ab.temporal_span, 4); // t = 5 - 1, exact epoch seconds
    assert_eq!(ab.aggregate_amount, 80_000_000);
    assert_eq!(ab.window, (1, 6));
    // Time stayed an explicit coordinate: three distinct t values, no
    // bucketing, no second temporal representation (Time rule).
    let ts: BTreeSet<u64> = rels
        .iter()
        .find(|d| d.relation.source == "A" && d.relation.target == "B" && d.relation.hop == 1)
        .expect("A -> B")
        .admitted
        .iter()
        .map(|(c, _)| c.t)
        .collect();
    assert_eq!(ts, BTreeSet::from([1, 2, 5]));
}


/// B11.4 — `A -> B -> C` is derived as `{source = A, target = C, hop = 2}`
/// exactly when `t1 <= t2` and `t2 - t1 <= Δ`.
#[test]
fn second_hop_respects_the_temporal_bound() {
    let (csf, _, ents) = build(&EVENTS);
    let ac = |delta: u64| {
        rel_of(
            &derive_relations(&csf, &ents, Window { delta, ..W }).expect("derive"),
            "A",
            "C",
            2,
        )
        .cloned()
    };
    // Δ = 2 admits (t1=1,t2=3) and (t1=2,t2=3); the (t1=5) leg fails t1 <= t2.
    let wide = ac(2).expect("A -> C hop 2 at Δ=2");
    assert_eq!(wide.event_count, 3);
    assert_eq!(wide.temporal_span, 2);
    assert_eq!(wide.aggregate_amount, 60_000_000);
    // Δ = 1 drops the (t1=1,t2=3) leg.
    let tight = ac(1).expect("A -> C hop 2 at Δ=1");
    assert_eq!(tight.event_count, 2);
    assert_eq!(tight.aggregate_amount, 50_000_000);
    // Δ = 0 requires t1 == t2 == 3, and A -> B never occurs at t = 3.
    assert!(ac(0).is_none(), "no path satisfies Δ = 0");
    // A -> C must never appear at hop 1: no direct A -> C event exists.
    assert!(rel_of(&derive_relations(&csf, &ents, W).expect("derive"), "A", "C", 1).is_none());
}

/// B11.5 / B11.6 — the Abacus path equals the independent reference, field
/// for field, and the predicate results agree relation by relation.
#[test]
fn abacus_equals_independent_reference_and_predicates() {
    let (csf, _, ents) = build(&EVENTS);
    let abacus = derive_relations(&csf, &ents, W).expect("derive");
    let reference = reference_relations(&EVENTS, &ents, W).expect("reference");
    assert_eq!(abacus, reference, "AbacusBehavior == ReferenceBehavior");

    let th = Thresholds { min_events: 3, min_amount_micro: 60_000_000, max_span_secs: 2, min_hop: 2 };
    assert_eq!(
        th.evaluate_all(&abacus),
        th.evaluate_all(&reference),
        "AbacusPredicate == ReferencePredicate"
    );
    // The compound condition must fire on exactly the A -> C hop-2 relation.
    let fired: Vec<(String, String, u32)> = abacus
        .iter()
        .zip(th.evaluate_all(&abacus))
        .filter(|(_, p)| *p)
        .map(|(d, _)| (d.relation.source.clone(), d.relation.target.clone(), d.relation.hop))
        .collect();
    assert_eq!(fired, vec![("A".to_string(), "C".to_string(), 2)]);
}

/// The four atoms and their conjunction are independent (semantic §18).
#[test]
fn predicate_atoms_are_independent() {
    let (csf, _, ents) = build(&EVENTS);
    let rels = derive_relations(&csf, &ents, W).expect("derive");
    let ac = rel_of(&rels, "A", "C", 2).expect("A -> C hop 2");
    let th = Thresholds { min_events: 3, min_amount_micro: 60_000_000, max_span_secs: 2, min_hop: 2 };
    assert_eq!(th.atoms(ac), [true, true, true, true]);
    assert!(th.evaluate(ac));
    // Raising N above the admitted cardinality falsifies the conjunction.
    let strict = Thresholds { min_events: 4, ..th };
    assert_eq!(strict.atoms(ac), [false, true, true, true]);
    assert!(!strict.evaluate(ac));
    // A hop-1 relation can never satisfy a `hop >= 2` condition.
    let ab = rel_of(&rels, "A", "B", 1).expect("A -> B hop 1");
    assert_eq!(th.atoms(ab), [true, true, false, false]);
    assert!(!th.evaluate(ab));
}
