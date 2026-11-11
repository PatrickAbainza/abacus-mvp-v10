//! Atomic reference operator tests: oracle semantics, edge cases, reversals.
//! Generic terms only. No new behavior on existing structures.

use abacus_tensor::atomic::outcome::{OpReason, Outcome};
use abacus_tensor::atomic::reference::{
    empty_identity, map, merge, merge_add, pair_earliest, pair_latest, pair_next_after, reduce,
    restrict, support_pair_seek, Baseline, Coverage, Predicate, RowRef, Theta,
};
use abacus_tensor::atomic::types::{Coord3, Entry, LocSet, Mode, Monoid, Support};

fn e(t: u64, s: u32, r: u32, v: i64) -> Entry {
    Entry::new(t, s, r, v, LocSet::empty())
}
fn el(t: u64, s: u32, r: u32, v: i64, loc: u32) -> Entry {
    Entry::new(t, s, r, v, LocSet::from_sorted(vec![loc]))
}
fn sup(v: Vec<Entry>) -> Support {
    let mut v = v;
    v.sort();
    // canonicalize: sum dup coords, drop zeros
    let mut out: Vec<Entry> = Vec::new();
    for en in v {
        if let Some(last) = out.last_mut() {
            if last.coord == en.coord {
                last.value += en.value;
                let u = last.locators.union(&en.locators);
                last.locators = u;
                continue;
            }
        }
        out.push(en);
    }
    out.retain(|en| en.value != 0);
    Support::from_canonical(out)
}

#[test]
fn empty_input_identities() {
    let empty = Support::empty();
    assert!(restrict(&empty, &Predicate::all()).entries.is_empty());
    assert!(map(&empty, |_, _, _, v, _| Some(v), &Theta::default()).entries.is_empty());
    let (Outcome::Determinate(m),) = (merge(&empty, &empty, |a, b| a.or(b), Coverage::Union),) else {
        panic!("union of empties determinate");
    };
    assert!(m.entries.is_empty());
    assert_eq!(empty_identity(Monoid::Sum), Some(0));
    assert_eq!(empty_identity(Monoid::Count), Some(0));
    assert_eq!(empty_identity(Monoid::Min), None);
    assert_eq!(empty_identity(Monoid::Max), None);
    assert_eq!(empty_identity(Monoid::Any), None);
    let Outcome::Determinate(rows) = reduce(&empty, &[Mode::Mode0], Monoid::Sum) else {
        panic!("empty reduce determinate");
    };
    assert!(rows.is_empty());
}

#[test]
fn one_entry_and_window_boundaries() {
    let s = sup(vec![e(10, 1, 2, 5)]);
    // lo inclusive, hi exclusive
    assert_eq!(restrict(&s, &Predicate::window(10, 11)).entries.len(), 1);
    assert!(restrict(&s, &Predicate::window(11, 12)).entries.is_empty());
    assert!(restrict(&s, &Predicate::window(0, 10)).entries.is_empty());
    assert_eq!(restrict(&s, &Predicate::window(0, 11)).entries.len(), 1);
    // inverted window empty
    assert!(restrict(&s, &Predicate::window(11, 10)).entries.is_empty());
}

#[test]
fn several_events_one_coord_and_reversal_to_zero() {
    // three raw events at one coord: +5, +3, -8 => net zero => absent
    let s = sup(vec![e(7, 1, 1, 5), e(7, 1, 1, 3), e(7, 1, 1, -8)]);
    assert!(s.entries.is_empty(), "net-zero coord removed (tensor presence)");
    // reversal partial: +5 then -3 => +2 survives
    let s2 = sup(vec![e(7, 1, 2, 5), e(7, 1, 2, -3)]);
    assert_eq!(s2.entries.len(), 1);
    assert_eq!(s2.entries[0].value, 2);
}

#[test]
fn self_pairs_and_equal_timestamps() {
    let s = sup(vec![el(5, 3, 3, 4, 9), el(5, 3, 3, 2, 4)]);
    // canonical: one coord, value 6, locators {4, 9}
    assert_eq!(s.entries.len(), 1);
    assert_eq!(s.entries[0].value, 6);
    assert_eq!(s.entries[0].locators.locators, vec![4, 9]);
    // pair seek tie: lowest locator witness via support seek earliest
    let got = support_pair_seek(&s, 3, 3, None, false).expect("present");
    assert_eq!(got.locators.lowest(), Some(4));
    // strict next-after excludes equal timestamps
    assert!(support_pair_seek(&s, 3, 3, Some(5), false).is_none());
    assert!(support_pair_seek(&s, 3, 3, Some(4), false).is_some());
}

#[test]
fn union_bottom_handling_and_difference() {
    let a = sup(vec![e(1, 1, 1, 5), e(2, 1, 2, 7)]);
    let b = sup(vec![e(2, 1, 2, 3), e(3, 2, 2, 9)]);
    // set difference A\\B via union combine: keep left-only
    let diff = |l: Option<i64>, r: Option<i64>| match (l, r) {
        (Some(v), None) => Some(v),
        _ => None,
    };
    let Outcome::Determinate(d) = merge(&a, &b, diff, Coverage::Union) else {
        panic!("diff determinate");
    };
    assert_eq!(d.entries.len(), 1);
    assert_eq!(d.entries[0].coord, Coord3::new(1, 1, 1));
    // intersection keeps only shared
    let Outcome::Determinate(i) = merge(&a, &b, |l, r| match (l, r) {
        (Some(x), Some(y)) => Some(x + y),
        _ => None,
    }, Coverage::Intersection) else {
        panic!("intersect determinate");
    };
    assert_eq!(i.entries.len(), 1);
    assert_eq!(i.entries[0].value, 10);
    // missing side receives bottom: union over disjoint supports keeps both sides
    let c = sup(vec![e(9, 9, 9, 1)]);
    let Outcome::Determinate(u) = merge(&a, &c, |l, r| l.or(r), Coverage::Union) else {
        panic!("probe determinate");
    };
    assert_eq!(u.entries.len(), 3, "union exposes bottom-kept entries from both sides");
    let Outcome::Determinate(i2) = merge(&a, &c, |l, r| l.or(r), Coverage::Intersection) else {
        panic!("disjoint intersect determinate");
    };
    assert!(i2.entries.is_empty(), "disjoint intersection empty");
}

#[test]
fn overflow_yields_indeterminate() {
    let a = sup(vec![e(1, 1, 1, i64::MAX)]);
    let b = sup(vec![e(1, 1, 1, 1)]);
    match merge_add(&a, &b, Coverage::Intersection) {
        Outcome::Indeterminate(ind) => assert_eq!(ind.reason, OpReason::Overflow),
        Outcome::Determinate(_) => panic!("expected overflow indeterminate"),
    }
    match reduce(&a, &[], Monoid::Sum) {
        Outcome::Determinate(rows) => {
            assert_eq!(rows.len(), 1);
        }
        Outcome::Indeterminate(_) => panic!("single max sums fine"),
    }
    let c = sup(vec![e(1, 1, 1, i64::MAX), e(1, 1, 2, i64::MAX)]);
    match reduce(&c, &[Mode::Mode0, Mode::Mode1, Mode::Mode2], Monoid::Sum) {
        Outcome::Indeterminate(ind) => assert_eq!(ind.reason, OpReason::Overflow),
        Outcome::Determinate(_) => panic!("expected reduce overflow"),
    }
}

#[test]
fn map_with_theta_and_baseline() {
    // threshold map: keep values >= theta, emit value itself; else bottom
    let theta = Theta { threshold: Some(5), baseline: None };
    let f = |_: u64, _: u32, _: u32, v: i64, th: &Theta| {
        if v >= th.threshold.unwrap_or(i64::MIN) { Some(v) } else { None }
    };
    let s = sup(vec![e(1, 1, 1, 4), e(2, 1, 1, 5), e(3, 1, 1, 9)]);
    let out = map(&s, f, &theta);
    assert_eq!(out.entries.len(), 2);
    // coordinate-as-value: emit mode-1 key as value
    let g = |_: u64, s: u32, _: u32, _: i64, _: &Theta| Some(s as i64);
    let out2 = map(&s, g, &Theta::default());
    assert!(out2.entries.iter().all(|en| en.value == 1));
    // baseline descriptor carried opaquely; missing baseline -> caller maps to Indeterminate
    let th2 = Theta {
        threshold: None,
        baseline: Some(Baseline { mean: 1000, scale: 100, version: "v3".into(), as_of: 1700 }),
    };
    assert_eq!(th2.baseline.as_ref().unwrap().version, "v3");
    assert_eq!(th2.baseline.as_ref().unwrap().as_of, 1700);
}

#[test]
fn row_pair_seek_strict_and_ties() {
    let row = vec![
        RowRef { t: 5, peer: 7, locator: 3 },
        RowRef { t: 5, peer: 7, locator: 1 },
        RowRef { t: 9, peer: 7, locator: 2 },
        RowRef { t: 9, peer: 8, locator: 0 },
    ];
    assert_eq!(pair_earliest(&row, 7), Some(RowRef { t: 5, peer: 7, locator: 1 }));
    assert_eq!(pair_latest(&row, 7), Some(RowRef { t: 9, peer: 7, locator: 2 }));
    assert_eq!(pair_next_after(&row, 7, 5), Some(RowRef { t: 9, peer: 7, locator: 2 }));
    assert_eq!(pair_next_after(&row, 7, 9), None);
    assert_eq!(pair_earliest(&row, 99), None);
    // empty pair
    let empty: Vec<RowRef> = vec![];
    assert_eq!(pair_earliest(&empty, 7), None);
    assert_eq!(pair_latest(&empty, 7), None);
    assert_eq!(pair_next_after(&empty, 7, 0), None);
}

#[test]
fn min_max_any_witness_deterministic() {
    let s = sup(vec![el(1, 1, 1, 5, 8), el(2, 1, 1, 5, 3), el(3, 1, 1, 9, 6)]);
    let Outcome::Determinate(rows) = reduce(&s, &[Mode::Mode0], Monoid::Min) else {
        panic!("min determinate");
    };
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].value, Some(5));
    assert_eq!(rows[0].witness, Some(3), "min tie -> lowest locator");
    let Outcome::Determinate(rows) = reduce(&s, &[Mode::Mode0], Monoid::Max) else {
        panic!("max determinate");
    };
    assert_eq!(rows[0].value, Some(9));
    assert_eq!(rows[0].witness, Some(6));
    let Outcome::Determinate(rows) = reduce(&s, &[Mode::Mode0], Monoid::Any) else {
        panic!("any determinate");
    };
    assert_eq!(rows[0].value, Some(1));
    assert_eq!(rows[0].witness, Some(3), "any witness = lowest locator");
}
