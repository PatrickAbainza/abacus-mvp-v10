//! Hermetic acceptance tests for the fan-in / fan-out rule-discovery rule
//! (`docs/golden_source_rule_discovery_fanio_addendum.md`, F3–F11, F12).
//!
//! Hermetic: no artifact, no CSV, no network, **no label**. The real 90-day run
//! (F6 split, F10 artifact, F12's isolation obligations) lives in the
//! `rule_discovery_fanio` binary, which needs the materialised profile.

use abacus_tensor::behavioral::{derive_relations, Entities, Window};
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::compress;
use abacus_tensor::rule_discovery::Confusion;
use abacus_tensor::rule_discovery_fanio::{
    candidates, fan_in_at, fan_in_counts, fan_out_at, fan_out_counts, key, preferred_to, select1,
    FanRule, Quantity, ScoredFan, ValueHist, DAY_SECONDS, VACUOUS_OFFSET,
};
use std::collections::BTreeSet;

/// Declared synthetic windowed support `E_W` (each coordinate once), sorted
/// `(t,s,r)`. Chosen so fan-in and fan-out take several distinct values, and so
/// that the counter-party **set** size differs from the entry **count**.
const ENTRIES: [(u64, u32, u32); 7] = [
    (10, 0, 1), // sender 0 -> receiver 1
    (20, 0, 1), // sender 0 -> receiver 1 again (same pair, second entry)
    (30, 1, 1), // sender 1 -> receiver 1
    (40, 2, 1), // sender 2 -> receiver 1
    (50, 1, 2), // sender 1 -> receiver 2
    (60, 1, 3), // sender 1 -> receiver 3
    (70, 3, 3), // sender 3 -> receiver 3
];

fn entries() -> Vec<(u64, u32, u32)> {
    ENTRIES.to_vec()
}

/// F4 — the fan-in projection is exactly the existing set primitive's size.
#[test]
fn fan_in_counts_equals_existing_set_primitive() {
    let e = entries();
    let counts = fan_in_counts(&e);
    for &(_, _, r) in &ENTRIES {
        let via_primitive = fan_in_at(&e, r, 0, u64::MAX);
        assert_eq!(counts.get(&r).copied().unwrap_or(0), via_primitive, "receiver {r}");
    }
    // Receiver 1 has four entries but only three distinct senders.
    assert_eq!(counts[&1], 3);
    assert_eq!(counts[&2], 1);
    assert_eq!(counts[&3], 2);
}

/// F4 — the fan-out projection is exactly the existing set primitive's size.
#[test]
fn fan_out_counts_equals_existing_set_primitive() {
    let e = entries();
    let counts = fan_out_counts(&e);
    for &(_, s, _) in &ENTRIES {
        let via_primitive = fan_out_at(&e, s, 0, u64::MAX);
        assert_eq!(counts.get(&s).copied().unwrap_or(0), via_primitive, "sender {s}");
    }
    assert_eq!(counts[&0], 1);
    assert_eq!(counts[&1], 3);
    assert_eq!(counts[&2], 1);
    assert_eq!(counts[&3], 1);
}

/// F4 — fan-in / fan-out are **set sizes** over counterparties, not entry
/// counts (FV §4 `rin`/`sout` are set sizes; §4A counts are multisets).
#[test]
fn fan_quantities_are_counterparty_set_sizes_not_entry_counts() {
    let counts = fan_in_counts(&entries());
    // Receiver 1 carries 4 entries but 3 distinct senders.
    let entries_at_r1 = ENTRIES.iter().filter(|(_, _, r)| *r == 1).count();
    assert_eq!(entries_at_r1, 4);
    assert_eq!(counts[&1], 3);
    assert_ne!(u64::try_from(entries_at_r1).unwrap(), counts[&1]);
}

/// F4 — the projection is an exact set operation: order-independent and
/// idempotent under duplication of the rows list.
#[test]
fn projection_is_order_independent_and_idempotent() {
    let mut shuffled = entries();
    shuffled.reverse();
    let doubled: Vec<(u64, u32, u32)> = entries().into_iter().chain(entries()).collect();
    let base_in = fan_in_counts(&entries());
    let base_out = fan_out_counts(&entries());
    assert_eq!(fan_in_counts(&shuffled), base_in);
    assert_eq!(fan_out_counts(&shuffled), base_out);
    // Duplicating the same coordinates leaves `A_one|_W` unchanged (it is a set).
    assert_eq!(fan_in_counts(&doubled), base_in);
    assert_eq!(fan_out_counts(&doubled), base_out);
}

/// F3.1 — every present vertex has at least one counterparty, so both
/// quantities are `>= 1`, so `T = 1` is the vacuous all-TRUE threshold.
#[test]
fn quantities_are_always_at_least_one() {
    assert!(fan_in_counts(&entries()).values().all(|&c| c >= 1));
    assert!(fan_out_counts(&entries()).values().all(|&c| c >= 1));
}

/// F4 — the read key is the receiver for fan-in and the sender for fan-out.
#[test]
fn key_selects_the_declared_vertex_axis() {
    assert_eq!(key(Quantity::FanIn, 5, 9), 9);
    assert_eq!(key(Quantity::FanOut, 5, 9), 5);
}

/// F3 — the rule is a single `>=` atom and is total.
#[test]
fn rule_is_one_atom_and_total() {
    let r = FanRule::new(Quantity::FanIn, 4);
    assert!(!r.evaluate(3));
    assert!(r.evaluate(4));
    assert!(r.evaluate(5));
    assert_eq!(r.to_string(), "fan_in >= 4");
    let counts = fan_in_counts(&entries());
    assert!(!r.evaluate_at(&counts, 1)); // |in(1)| = 3 < 4
    let r3 = FanRule::new(Quantity::FanIn, 3);
    assert!(r3.evaluate_at(&counts, 1));
    assert!(!r3.evaluate_at(&counts, 2));
    assert!(!r3.evaluate_at(&counts, 99)); // absent key reads 0 (FV §1)
}

/// F7 — candidates are ascending, deduped, `>= 1`, and every element is
/// observed except the single declared vacuous sentinel `max + 1`.
#[test]
fn candidates_are_observed_or_vacuous() {
    let observed = vec![1u64, 3, 3, 2, 7];
    let cand = candidates(&observed);
    assert_eq!(cand, vec![1, 2, 3, 7, 8]); // 8 == max(7) + VACUOUS_OFFSET
    assert_eq!(*cand.last().unwrap(), 7 + VACUOUS_OFFSET);
    let mut obs = observed.clone();
    obs.sort_unstable();
    obs.dedup();
    for v in &cand {
        assert!(*v == 8 || obs.binary_search(v).is_ok(), "{v} must be observed or vacuous");
    }
    // Values below 1 are never rules (F3.1).
    assert!(!candidates(&[0, 0, 4]).contains(&0));
    // Empty input degenerates to the sentinel alone.
    assert_eq!(candidates(&[]), vec![1]);
}

/// F7/F11 — the value histogram is exactly a direct count, and its `>=` read is
/// an exact suffix sum.
#[test]
fn value_hist_matches_direct_counting() {
    let mut h = ValueHist::default();
    h.add(2, 3, 1);
    h.add(1, 5, 2);
    h.add(2, 2, 1);
    assert_eq!(h.values, vec![1, 2]);
    assert_eq!(h.records, 10);
    assert_eq!(h.positives, 4);
    assert_eq!(h.distinct(), 2);
    assert_eq!(h.at_least(1), (10, 4));
    assert_eq!(h.at_least(2), (5, 2));
    assert_eq!(h.at_least(3), (0, 0)); // above every observed value -> all-FALSE
}

/// F7.1 — admissibility removes the all-TRUE and all-FALSE thresholds.
#[test]
fn admissible_surface_removes_vacuous_thresholds() {
    // Two records with quantity 1, three with quantity 3; one positive.
    let mut h = ValueHist::default();
    h.add(1, 2, 1);
    h.add(3, 3, 0);
    let ladder = vec![1u64, 3, 4]; // 1 = all-TRUE, 3 = partial, 4 = all-FALSE
    assert_eq!(h.admissible(&ladder), vec![3]);
    // A ladder whose every threshold is vacuous yields an empty surface.
    let mut flat = ValueHist::default();
    flat.add(2, 4, 1);
    assert!(flat.admissible(&[2, 3]).is_empty());
}

/// F15 — the exact cumulative ladder carries complete per-value information:
/// every row's suffix sums, with no float and no sampling.
#[test]
fn cumulative_ladder_carries_complete_per_value_information() {
    let mut h = ValueHist::default();
    h.add(2, 3, 1);
    h.add(1, 5, 2);
    h.add(2, 2, 1);
    // (value, records_ge, positives_ge) ascending by value.
    assert_eq!(h.cumulative_ladder(), vec![(1, 10, 4), (2, 5, 2)]);
    // Every row agrees with the direct `at_least` reads.
    for (v, rec, pos) in h.cumulative_ladder() {
        assert_eq!((rec, pos), h.at_least(v));
    }
    // The ladder of a threshold strictly above every observed value is empty.
    assert!(h.at_least(3) == (0, 0));
}

/// F15 — one fan-in value and one fan-out value per canonical transaction:
/// the projection maps contribute exactly one value for every record of the
/// derivation window, and nothing is dropped.
#[test]
fn every_transaction_gets_exactly_one_fan_value() {
    use abacus_tensor::composites::settle;
    use abacus_tensor::behavioral::{derive_relations, Entities, Window};
    use std::collections::BTreeSet;
    let dict: &'static Vec<String> =
        Box::leak(Box::new(["A", "B", "C", "D"].iter().map(|s| s.to_string()).collect()));
    let raw: Vec<TxEntry> = EVENTS.iter().map(|&(t, s, r, v)| TxEntry::new(t, s, r, v)).collect();
    let settled = settle(raw).expect("settle");
    let csf = compress(&settled);
    let ents = Entities { senders: dict, receivers: dict };
    let w = Window { t_start: 0, t_end: DAY_SECONDS, delta: DAY_SECONDS };
    let derived = derive_relations(&csf, &ents, w).expect("derive_relations");
    let e_w: BTreeSet<(u64, u32, u32)> = derived
        .iter()
        .filter(|x| x.relation.hop == 1)
        .flat_map(|x| x.admitted.iter().map(|(c, _)| (c.t, c.s, c.r)))
        .collect();
    let entries: Vec<(u64, u32, u32)> = e_w.iter().copied().collect();
    let fan_in = fan_in_counts(&entries);
    let fan_out = fan_out_counts(&entries);
    // Exactly one value per coordinate — the per-record read below cannot drop.
    assert_eq!(entries.len(), 6);
    for &(_, s, r) in &entries {
        assert!(fan_in.contains_key(&key(Quantity::FanIn, s, r)));
        assert!(fan_out.contains_key(&key(Quantity::FanOut, s, r)));
    }
    // Distinct-value surface identity: frequency rows sum to the population.
    let mut hist = ValueHist::default();
    for &(_, s, r) in &entries {
        hist.add(fan_in[&key(Quantity::FanIn, s, r)], 1, 0);
    }
    assert_eq!(hist.records as usize, entries.len());
    assert_eq!(hist.cumulative_ladder().iter().map(|&(_, rec, _)| rec).max(), Some(entries.len() as u64));
}


/// F8 — the selection key is the lexicographic maximum of
/// `(F1, precision, recall, coverage)`, ties broken by the lower index.
#[test]
fn selection_maximises_f1_first_and_is_total() {
    // A: TP=5 FP=5 TN=90 FN=5  -> F1 = 10/20 = 0.5
    let a = ScoredFan {
        rule: FanRule::new(Quantity::FanIn, 1),
        index: 0,
        fired: 10,
        total: 105,
        cm: Confusion { tp: 5, fp: 5, tn: 90, fn_: 5 },
    };
    // B: TP=4 FP=0 TN=95 FN=6 -> F1 = 8/14 ~ 0.571 (higher), recall lower
    let b = ScoredFan {
        rule: FanRule::new(Quantity::FanIn, 2),
        index: 1,
        fired: 4,
        total: 105,
        cm: Confusion { tp: 4, fp: 0, tn: 95, fn_: 6 },
    };
    assert!(preferred_to(&b, &a), "higher F1 must win");
    assert!(!preferred_to(&a, &b));
    // Identical metrics: the lower index wins (total order).
    let c = ScoredFan { index: 2, ..b };
    assert!(preferred_to(&b, &c));
    assert!(!preferred_to(&c, &b));
    let winner = select1(&[a, b, c]).expect("a winner");
    assert_eq!(winner.index, 1);
}

/// F8 — `select1` returns the element preferred to every other candidate.
#[test]
fn select1_is_the_unique_maximum() {
    let mk = |index: usize, tp: u64, fp: u64, fn_: u64| ScoredFan {
        rule: FanRule::new(Quantity::FanOut, index as u64 + 1),
        index,
        fired: tp + fp,
        total: 100,
        cm: Confusion { tp, fp, tn: 100 - tp - fp - fn_, fn_ },
    };
    let scored = vec![mk(0, 1, 9, 1), mk(1, 8, 2, 2), mk(2, 8, 2, 2), mk(3, 0, 0, 10)];
    let best = select1(&scored).expect("a winner");
    for s in &scored {
        if s.index != best.index {
            assert!(preferred_to(&best, s), "winner must beat index {}", s.index);
        }
    }
    assert_eq!(best.index, 1); // equal metrics to index 2, lower index wins
}

/// F11 — the confusion-matrix identities used by the scorer hold.
#[test]
fn confusion_metric_identities_hold() {
    let cm = Confusion { tp: 67, fp: 8744, tn: 2_370_665, fn_: 2530 };
    assert_eq!(cm.total(), cm.tp + cm.fp + cm.tn + cm.fn_);
    assert_eq!(cm.predicted(), cm.tp + cm.fp);
    assert_eq!(cm.positives(), cm.tp + cm.fn_);
    assert_eq!(cm.f1().0, 2 * i128::from(cm.tp));
    assert_eq!(cm.f1().1, i128::from(2 * cm.tp + cm.fp + cm.fn_));
}

/// F3.1 — non-vacuity distinguishes all-TRUE and all-FALSE from a real rule.
#[test]
fn non_vacuity_distinguishes_the_degenerate_cases() {
    let mk = |fired: u64, total: u64| ScoredFan {
        rule: FanRule::new(Quantity::FanIn, 2),
        index: 0,
        fired,
        total,
        cm: Confusion::default(),
    };
    assert!(!mk(0, 100).is_non_vacuous()); // all-FALSE
    assert!(!mk(100, 100).is_non_vacuous()); // all-TRUE
    assert!(mk(1, 100).is_non_vacuous());
    assert!(mk(99, 100).is_non_vacuous());
}

/// F3 — the declared quantity names are stable (they appear in the artifact).
#[test]
fn quantity_names_are_declared() {
    assert_eq!(Quantity::FanIn.as_str(), "fan_in");
    assert_eq!(Quantity::FanOut.as_str(), "fan_out");
}

/// Declared synthetic events (P2 exact micro-units). Several senders share a
/// receiver so fan-in takes distinct values `{3, 2, 1}` and fan-out takes
/// `{1, 2, 3}` — a non-degenerate selection surface.
const EVENTS: [(u64, u32, u32, i64); 6] = [
    (10, 0, 1, 1_000_000),
    (20, 1, 1, 2_000_000),
    (30, 2, 1, 4_000_000),
    (40, 1, 2, 8_000_000),
    (50, 2, 2, 1_500_000),
    (60, 1, 3, 3_000_000),
];

const DICT: [&str; 4] = ["A", "B", "C", "D"];

/// Build the existing behavioral state (B4) for the fixture, unchanged.
fn derivations() -> Vec<(u64, u32, u32)> {
    let d: &'static Vec<String> =
        Box::leak(Box::new(DICT.iter().map(|s| (*s).to_string()).collect()));
    let raw: Vec<TxEntry> =
        EVENTS.iter().map(|&(t, s, r, v)| TxEntry::new(t, s, r, v)).collect();
    let settled = abacus_tensor::composites::settle(raw).expect("settle");
    let csf = compress(&settled);
    let ents = Entities { senders: d, receivers: d };
    let w = Window { t_start: 0, t_end: DAY_SECONDS, delta: DAY_SECONDS };
    let derived = derive_relations(&csf, &ents, w).expect("derive_relations");
    // E_W = exact union of the hop-1 admitted coordinates (Set-operation rule).
    let support: BTreeSet<(u64, u32, u32)> = derived
        .iter()
        .filter(|x| x.relation.hop == 1)
        .flat_map(|x| x.admitted.iter().map(|(c, _)| (c.t, c.s, c.r)))
        .collect();
    support.into_iter().collect()
}

/// F4/F6/F7/F8 — end-to-end discovery over the **real** B4 derivation, with a
/// synthetic truth (fixture only; no sidecar is read — F9).
#[test]
fn end_to_end_discovery_over_real_behavioral_state() {
    let entries = derivations();
    let universe: BTreeSet<(u64, u32, u32)> = entries.iter().copied().collect();
    assert_eq!(universe.len(), entries.len(), "E_W holds each coordinate once");
    assert_eq!(universe.len(), 6, "six distinct fixture coordinates");

    // Synthetic truth: 0-based indices 1 and 3 are "positive".
    let truth: BTreeSet<(u64, u32, u32)> = universe
        .iter()
        .enumerate()
        .filter(|(i, _)| *i == 1 || *i == 3)
        .map(|(_, c)| *c)
        .collect();
    assert_eq!(truth.len(), 2);

    let mut selected: Vec<(Quantity, u64)> = Vec::new();
    for q in [Quantity::FanIn, Quantity::FanOut] {
        let counts = match q {
            Quantity::FanIn => fan_in_counts(&entries),
            Quantity::FanOut => fan_out_counts(&entries),
        };
        let mut hist = ValueHist::default();
        for c in &universe {
            let v = counts.get(&key(q, c.1, c.2)).copied().unwrap_or(0);
            hist.add(v, 1, u64::from(truth.contains(c)));
        }
        let ladder = candidates(&hist.values);
        assert!(ladder.windows(2).all(|w| w[0] < w[1]), "ascending ladder");
        // F7 — the frozen surface is the FULL ladder: every distinct observed
        // value plus the declared vacuous all-FALSE sentinel.
        assert_eq!(ladder.len(), hist.distinct() + 1, "distinct + 1 sentinel");
        let cand = ladder;
        assert!(!cand.is_empty(), "{q:?}: the fixture surface is non-degenerate");
        let (total, positives) = (hist.records, hist.positives);
        let mut scored: Vec<ScoredFan> = Vec::new();
        for (index, t) in cand.iter().enumerate() {
            let (predicted, tp) = hist.at_least(*t);
            let fp = predicted - tp;
            let fn_ = positives - tp;
            let tn = total - predicted - fn_;
            scored.push(ScoredFan {
                rule: FanRule::new(q, *t),
                index,
                fired: predicted,
                total,
                cm: Confusion { tp, fp, tn, fn_ },
            });
        }
        let best = select1(&scored).expect("a winner");
        for s in &scored {
            if s.index != best.index {
                assert!(preferred_to(&best, s), "{q:?}: winner must beat {}", s.index);
            }
        }
        // The histogram accelerator agrees with the direct set count.
        let direct = universe
            .iter()
            .filter(|c| counts.get(&key(q, c.1, c.2)).copied().unwrap_or(0) >= best.rule.threshold)
            .count();
        assert_eq!(u64::try_from(direct).unwrap(), best.fired, "{q:?}: hist == direct");
        assert!(best.is_non_vacuous(), "{q:?}: the fixture surface is non-degenerate");
        selected.push((q, best.rule.threshold));
    }
    // Fan-in and fan-out are selected independently and their values differ.
    assert_eq!(selected.len(), 2);
    assert_ne!(selected[0].1, selected[1].1, "the two quantities are not the same rule");
}


