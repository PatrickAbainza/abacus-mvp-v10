//! Backend-map equivalence: adapters vs reference compositions (exact match).
//! Generic terms only.

use abacus_tensor::atomic::backends::*;
use abacus_tensor::atomic::outcome::Outcome;
use abacus_tensor::atomic::reference::{Baseline, Coverage, Predicate, Theta};
use abacus_tensor::atomic::types::{Entry, LocSet, Mode, Monoid, Support};
use abacus_tensor::graphblas::matrix::PairMatrix;
use abacus_tensor::graphblas::types::{EligiblePairs, HopBound, VertexId};
use abacus_tensor::graphblas::vector::Vector;

fn mat(pairs: &[(u32, u32, i64)]) -> PairMatrix {
    let mut m = PairMatrix::zero();
    for &(s, r, v) in pairs {
        m.insert(s, r, v);
    }
    m
}
fn sup(pairs: &[(u64, u32, u32, i64)]) -> Support {
    let mut v: Vec<Entry> =
        pairs.iter().map(|&(t, s, r, v)| Entry::new(t, s, r, v, LocSet::empty())).collect();
    v.sort();
    Support::from_canonical(v)
}

#[test]
fn extract_matches_point_restrict() {
    let a = mat(&[(1, 2, 5), (3, 4, 7)]);
    assert_eq!(adapt_extract(&a, 1, 2), Some(5));
    assert_eq!(adapt_extract(&a, 9, 9), None);
    let s = adapt_restrict_matrix(&a, 0, &Predicate::all());
    assert_eq!(s.entries.len(), 2);
}

#[test]
fn directional_sums_match_fiber_restrict_reduce() {
    let a = mat(&[(1, 2, 5), (3, 2, 7), (1, 5, 4)]);
    let Outcome::Determinate(c) = adapt_reduce_col_plus(&a, 2) else { panic!("col") };
    assert_eq!(c, 12);
    let Outcome::Determinate(r) = adapt_reduce_row_plus(&a, 1) else { panic!("row") };
    assert_eq!(r, 9);
    // reference composition agrees
    let s = sup(&[(0, 1, 2, 5), (0, 3, 2, 7), (0, 1, 5, 4)]);
    let Outcome::Determinate(rows) =
        abacus_tensor::atomic::reference::reduce(&s, &[Mode::Mode0, Mode::Mode1], Monoid::Sum)
    else { panic!("ref") };
    let col: i64 = rows.iter().filter(|x| x.r == Some(2)).filter_map(|x| x.value).sum();
    assert_eq!(col, 12);
}

#[test]
fn one_hop_adapters_match_reference_composition() {
    // forward: senders {1,2} -> receivers; backward: receivers -> senders
    let a = mat(&[(1, 3, 1), (1, 4, 1), (2, 4, 1), (5, 6, 1)]);
    let f = Vector::from_sorted(vec![1, 2]);
    let got_fwd = adapt_vxm(&a, &f);
    assert_eq!(got_fwd.indices(), &[3, 4]);
    let got_bwd = adapt_mxv(&a, &Vector::from_sorted(vec![4]));
    assert_eq!(got_bwd.indices(), &[1, 2]);
    // reference composition over supports agrees
    let s = sup(&[(0, 1, 3, 1), (0, 1, 4, 1), (0, 2, 4, 1), (0, 5, 6, 1)]);
    assert_eq!(reference_one_hop(&s, &[1, 2], true), vec![3, 4]);
    assert_eq!(reference_one_hop(&s, &[4], false), vec![1, 2]);
    // visited exclusion: frontier members never in output
    assert!(!reference_one_hop(&s, &[1], true).contains(&1));
}

#[test]
fn ewise_or_matches_key_union() {
    let a = Vector::from_sorted(vec![1, 3]);
    let b = Vector::from_sorted(vec![2, 3]);
    assert_eq!(adapt_ewise_or(&a, &b).indices(), &[1, 2, 3]);
    assert_eq!(reference_key_union(&[1, 3], &[2, 3]), vec![1, 2, 3]);
}

#[test]
fn support_multiplicity_volume_match_compositions() {
    let a = mat(&[(1, 2, 5), (3, 4, 7)]);
    let one = adapt_support(&a);
    assert!(one.nonzeros().all(|(_, _, v)| v == 1));
    let cnt = adapt_multiplicity(&[(1, 2), (1, 2), (3, 4)]);
    assert_eq!(cnt.get(1, 2), Some(2));
    assert_eq!(cnt.get(3, 4), Some(1));
    let Outcome::Determinate(vol) = adapt_volume(&[((1, 2), 5), ((1, 2), 7)]) else { panic!("vol") };
    assert_eq!(vol.get(1, 2), Some(12));
    // overflow -> Indeterminate
    let big = adapt_volume(&[((1, 2), i64::MAX), ((1, 2), 1)]);
    assert!(big.is_indeterminate());
}

#[test]
fn scalar_reductions_match() {
    assert_eq!(adapt_reduce_count(&[1, 2, 3]), 3);
    let Outcome::Determinate(s) = adapt_reduce_sum(&[5, 7]) else { panic!("sum") };
    assert_eq!(s, 12);
    assert!(adapt_reduce_sum(&[i64::MAX, 1]).is_indeterminate());
    assert_eq!(adapt_reduce_max(&[5, 9, 3]), 9);
    // documented divergence: graphblas empty max = 0, atomic = bottom
    assert_eq!(adapt_reduce_max(&[]), 0);
    assert_eq!(
        abacus_tensor::atomic::reference::empty_identity(Monoid::Max),
        None,
        "atomic empty max is bottom, not 0",
    );
}

#[test]
fn prune_masks_match_restrict_predicates() {
    use abacus_tensor::graphblas::types::TransactionRecord;
    fn rec(t: u64) -> TransactionRecord { TransactionRecord { t, s: 1, r: 2, value: 5 } }
    let records = vec![rec(0), rec(5), rec(9)];
    let v = Vector::from_sorted(vec![1, 2]);
    assert_eq!(adapt_prune_vertices(&records, &v).len(), 3);
    assert!(adapt_prune_vertices(&records, &Vector::from_sorted(vec![9])).is_empty());
    let ep = EligiblePairs::from_sorted(vec![(1, 2)]);
    assert_eq!(adapt_prune_edges(&records, &ep).len(), 3);
    assert!(adapt_prune_edges(&records, &EligiblePairs::from_sorted(vec![(7, 7)])).is_empty());
    // restrict predicate agrees on the support side
    let s = sup(&[(0, 1, 2, 5), (5, 1, 2, 5), (9, 1, 2, 5)]);
    let mut p = Predicate::window(0, 6);
    p.keys1 = Some(vec![1]);
    p.keys2 = Some(vec![2]);
    assert_eq!(abacus_tensor::atomic::reference::restrict(&s, &p).entries.len(), 2);
}

#[test]
fn hop_and_support_adapters_agree() {
    let a = mat(&[(1, 2, 1), (2, 3, 1), (3, 4, 1)]);
    let ball = adapt_hop_ball(&a, VertexId(1), VertexId(2), HopBound(1));
    assert!(ball.contains(1) && ball.contains(2) && ball.contains(3));
    // reference one-hop from {1,2} forward gives {3} (minus visited)
    let s = sup(&[(0, 1, 2, 1), (0, 2, 3, 1), (0, 3, 4, 1)]);
    assert_eq!(reference_one_hop(&s, &[1, 2], true), vec![3]);
    // build_support fused == composition: counts + volumes + existence
    use abacus_tensor::graphblas::types::TransactionRecord;
    let w = vec![
        TransactionRecord { t: 0, s: 1, r: 2, value: 5 },
        TransactionRecord { t: 1, s: 1, r: 2, value: 7 },
    ];
    let st = adapt_build_support(&w).expect("support");
    assert_eq!(st.a_cnt.get(1, 2), Some(2));
    assert_eq!(st.a_vol.get(1, 2), Some(12));
    assert_eq!(st.a_one.get(1, 2), Some(1));
}

#[test]
fn merge_reduce_adapters_match_reference() {
    let a = sup(&[(1, 1, 1, 5)]);
    let b = sup(&[(1, 1, 1, 7), (2, 2, 2, 3)]);
    let Outcome::Determinate(m) = adapt_merge_add(&a, &b, Coverage::Union) else { panic!("merge") };
    assert_eq!(m.entries.len(), 2);
    assert_eq!(m.entries[0].value, 12);
    let Outcome::Determinate(rows) = adapt_reduce(&m, &[Mode::Mode0], Monoid::Sum) else { panic!("reduce") };
    assert_eq!(rows.len(), 2);
    let th = Theta {
        threshold: None,
        baseline: Some(Baseline { mean: 1, scale: 1, version: "v".into(), as_of: 9 }),
    };
    let mapped = adapt_map(&m, |_, _, _, v, _| if v >= 10 { Some(v) } else { None }, &th);
    assert_eq!(mapped.entries.len(), 1);
    assert_eq!(adapt_baseline(th.baseline.as_ref().unwrap()).version, "v");
}

#[test]
fn notes_document_non_mappings() {
    assert!(adapt_context_note().contains("multi-operator pipeline"));
    assert!(adapt_validation_note().contains("test-only"));
}
