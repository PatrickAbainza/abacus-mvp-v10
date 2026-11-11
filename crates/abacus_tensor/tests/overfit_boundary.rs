//! Hermetic boundary tests for the overfit measurement surface.
//!
//! Validates `abacus_tensor::overfit` on a hand-computed synthetic tensor only
//! (no profile, no labels, no CSV). Governing: derivation rows 5/11/16 (P1/P3),
//! behavioral addendum B4, feature-vector spec §§4/5 — the same sections the
//! module cites.

use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::compress;
use abacus_tensor::overfit::{
    day_bounds, day_maps, fnv_vecs, mask_lits, vector_for, Mvec, NM, DAY_SECONDS,
};
use abacus_tensor::samld::CoreRecord;

/// The synthetic fixture: `t=1 A->B`, `t=2 A->B`, `t=3 B->C` (µ-units).
fn fixture() -> (abacus_tensor::csf::compress::CsfTensor, Vec<TxEntry>) {
    let entries = vec![
        TxEntry::new(1, 0, 1, 10_000_000),
        TxEntry::new(2, 0, 1, 20_000_000),
        TxEntry::new(3, 1, 2, 30_000_000),
    ];
    let settled = abacus_tensor::composites::settle(entries).expect("settle");
    let csf = compress(&settled);
    (csf, settled)
}

#[test]
fn day_maps_projections_are_exact() {
    let (csf, _) = fixture();
    let m = day_maps(&csf, 0, 10).expect("day_maps");
    assert_eq!(m.entry_count, 3);
    // pair (0,1): two entries, sum 30e6, span 1s, max 20e6
    assert_eq!(m.pair_cnt.get(&(0, 1)).copied(), Some(2));
    assert_eq!(m.pair_vol.get(&(0, 1)).copied(), Some(30_000_000));
    assert_eq!(m.pair_span.get(&(0, 1)).copied(), Some(1));
    assert_eq!(m.pair_max.get(&(0, 1)).copied(), Some(20_000_000));
    // min entry and first instant of the pair (basis of pair_min / off_first)
    assert_eq!(m.pair_min.get(&(0, 1)).copied(), Some(10_000_000));
    assert_eq!(m.pair_lo.get(&(0, 1)).copied(), Some(1));
    // in(receiver 1) = {sender 0} => 1 ; out(sender 0) = {receiver 1} => 1
    assert_eq!(m.fan_in.get(&1).copied(), Some(1));
    assert_eq!(m.fan_out.get(&0).copied(), Some(1));
    // absent keys read 0
    assert_eq!(m.fan_in.get(&0).copied().unwrap_or(0), 0);
}

#[test]
fn vector_positions_match_the_declared_order() {
    let (csf, _) = fixture();
    let m = day_maps(&csf, 0, 10).expect("day_maps");
    let v = vector_for(&m, 1, 0, 1, 7);
    let expect: Mvec = [
        10_000_000, // tx_amt
        2,          // pair_cnt
        30_000_000, // pair_vol
        1,          // pair_span
        1,          // fan_in(r=1)
        1,          // fan_out(s=0)
        0,          // deg_sin = |in(0)|
        1,          // deg_rout = |out(1)| = {2}
        20_000_000, // pair_max
        10_000_000, // pair_min
        0,          // off_first = t(1) - pair_lo(1)
        15_000_000, // tap_mean = 30e6 / 2
        7,          // day_idx (passed through verbatim, R4)
        0,          // off_day = t(1) - day_min_t(1)
        0,          // pair_first = pair_lo(1) - day_min_t(1)
    ];
    assert_eq!(v, expect);
    assert_eq!(NM, 15);
}

#[test]
fn day_bounds_partition_is_contiguous() {
    let rec = |t: u64| CoreRecord {
        t,
        s: 0,
        r: 1,
        amount_micro: 1,
        log1p_amt: 0.0,
        z_amt: 0.0,
        roundedness: 0,
    };
    let records = vec![rec(0), rec(DAY_SECONDS), rec(2 * DAY_SECONDS)];
    let b = day_bounds(0, &records);
    assert_eq!(b[1], (0, 1));
    assert_eq!(b[2], (1, 2));
    assert_eq!(b[3], (2, 3));
    // contiguous: each day starts where the previous ended
    for k in 1..b.len() {
        assert_eq!(b[k].0, b[k - 1].1);
    }
}

#[test]
fn digest_and_mask_helpers_are_deterministic() {
    let a: Vec<Mvec> = vec![
        [0; NM],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ];
    let b = a.clone();
    assert_eq!(fnv_vecs(&a), fnv_vecs(&b));
    let mut c = a.clone();
    c[1][8] = 10;
    assert_ne!(fnv_vecs(&a), fnv_vecs(&c));
    assert_eq!(mask_lits(0b1_0101), vec![0, 2, 4]);
    assert!(mask_lits(0).is_empty());
}
