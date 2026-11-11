//! Rows 11/15/16 (§B.2): compression, decode, CSF queries, invariants.

use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::composites::settle;
use abacus_tensor::csf::{compress, decode_csf, lookup, no_self_aggregation, time_slice};

fn sample() -> Vec<TxEntry> {
    vec![
        TxEntry::new(1, 2, 3, 7),
        TxEntry::new(1, 2, 4, -2),
        TxEntry::new(1, 5, 0, 1),
        TxEntry::new(4, 0, 0, 100),
    ]
}

#[test]
fn compress_decode_is_lossless_and_invariant_holding() {
    let coo = sample();
    let csf = compress(&coo);
    assert!(csf.check_invariants());
    assert_eq!(csf.nnz(), 4);
    assert_eq!(decode_csf(&csf), coo); // P6: ⟦CSF(T)⟧ = T
}

#[test]
fn empty_csf_is_the_zero_tensor() {
    let csf = compress(&[]);
    assert!(csf.is_empty() && csf.check_invariants());
    assert_eq!(decode_csf(&csf), Vec::<TxEntry>::new());
}

#[test]
fn time_slice_is_a_half_open_window_in_prec3_order() {
    let csf = compress(&sample());
    let s = time_slice(&csf, 1, 4);
    assert_eq!(s.entries.len(), 3);
    assert!(s.entries.windows(2).all(|w| w[0] <= w[1]));
    assert!(time_slice(&csf, 4, 1).entries.is_empty()); // inverted window is empty
    assert_eq!(time_slice(&csf, 1, 5).entries, sample());
}

#[test]
fn point_lookup_reports_absent_as_zero() {
    let csf = compress(&sample());
    assert_eq!(lookup(&csf, 1, 2, 3), 7);
    assert_eq!(lookup(&csf, 1, 2, 9), 0);
    assert_eq!(lookup(&csf, 9, 9, 9), 0);
}

#[test]
fn fiber_views_are_sorted_and_aligned() {
    let csf = compress(&sample());
    assert_eq!(
        abacus_tensor::csf::sender_at_time(&csf, 1, 2),
        vec![(3, 7), (4, -2)]
    );
    assert_eq!(
        abacus_tensor::csf::receiver_at_time(&csf, 1, 3),
        vec![(2, 7)]
    );
    assert!(abacus_tensor::csf::sender_at_time(&csf, 9, 9).is_empty());
}

#[test]
fn invariant_check_rejects_malformed_csf() {
    let mut csf = compress(&sample());
    csf.mode0_fptr.pop(); // break the sentinel termination
    assert!(!csf.check_invariants());
}


#[test]
fn compress_never_self_aggregates() {
    // P6 + P3: one value slot per canonical entry, entry-for-entry equality.
    let coo = sample();
    let csf = compress(&coo);
    assert!(no_self_aggregation(&coo, &csf));
    assert_eq!(csf.nnz(), coo.len());
    assert_eq!(decode_csf(&csf), coo);
}

#[test]
fn aggregation_happens_only_in_settle_not_in_compress() {
    // Raw duplicates at identical (t,s,r): 7 + (-2) = 5 is summed exactly
    // once, by d_settle (P4, R_6), *before* compression. The CSF then carries
    // the canonical entries unchanged — nnz counts canonical entries, never
    // raw duplicates.
    let raw = vec![
        TxEntry::new(1, 2, 3, 7),
        TxEntry::new(1, 2, 3, -2),
        TxEntry::new(1, 2, 4, 1),
    ];
    let coo = settle(raw).expect("settle cannot overflow here");
    assert_eq!(
        coo,
        vec![TxEntry::new(1, 2, 3, 5), TxEntry::new(1, 2, 4, 1)]
    );
    let csf = compress(&coo);
    assert!(no_self_aggregation(&coo, &csf));
    assert_eq!(csf.nnz(), 2); // not 3: the sum came from settle, not from CSF
}

#[test]
fn no_self_aggregation_rejects_a_collapsed_representation() {
    // A CSF holding fewer slots than the canonical COO entries it claims to
    // represent has aggregated (or dropped) something: invariant must fail.
    let coo = sample();
    let csf = compress(&coo[..coo.len() - 1]);
    assert!(!no_self_aggregation(&coo, &csf));
}
