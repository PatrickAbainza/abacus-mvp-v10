//! Rows 1–9 (§B.2): coordinate order, exact values, tensor algebra.

use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::tensor::coordinate::Coord;
use abacus_tensor::tensor::value::{format_minor, parse_decimal_to_minor, MINOR_SCALE};
use abacus_tensor::tensor::Tensor;

#[test]
fn coord_order_is_t_then_s_then_r() {
    let a = Coord::new(1, 5, 9);
    let b = Coord::new(1, 5, 10);
    let c = Coord::new(1, 6, 0);
    let d = Coord::new(2, 0, 0);
    assert!(a < b && b < c && c < d);
    // Entry ordering is the same d_lex (row 1).
    assert!(TxEntry::new(1, 5, 9, 0) < TxEntry::new(1, 5, 10, -1));
}

#[test]
fn decimal_parse_is_exact_scale_1e6() {
    assert_eq!(parse_decimal_to_minor("100.00").unwrap(), 100 * MINOR_SCALE);
    assert_eq!(parse_decimal_to_minor("-0.000001").unwrap(), -1);
    assert_eq!(parse_decimal_to_minor("1.0000004").unwrap(), 1_000_000);
    assert_eq!(parse_decimal_to_minor("1.0000005").unwrap(), 1_000_001); // d_round half-up
    assert!(parse_decimal_to_minor(".5").is_err());
    assert!(parse_decimal_to_minor("").is_err());
    assert!(parse_decimal_to_minor("abc").is_err());
    assert!(parse_decimal_to_minor("9223372036854775808.0").is_err()); // P2 range
}

#[test]
fn minor_format_round_trips_parse() {
    for v in [0i64, 1, -1, 1_000_000, -2_500_000, 1_000_001, i64::MIN + 1] {
        let s = format_minor(v);
        assert_eq!(parse_decimal_to_minor(&s).unwrap(), v, "text {s}");
    }
    // The minimum i64 formats without overflow (widened arithmetic, row 3).
    assert_eq!(format_minor(i64::MIN), "-9223372036854.775808");
}

#[test]
fn construction_settles_duplicates_and_drops_zeros() {
    let t = Tensor::try_from_entries(vec![
        TxEntry::new(3, 0, 0, 5),
        TxEntry::new(1, 0, 0, 2),
        TxEntry::new(1, 0, 0, 3),
        TxEntry::new(2, 0, 0, 1),
        TxEntry::new(2, 0, 0, -1), // zero sum dropped (R_6)
    ])
    .unwrap();
    assert_eq!(t.nnz(), 2);
    let e = t.entries();
    assert_eq!((e[0].t, e[0].value), (1, 5));
    assert_eq!((e[1].t, e[1].value), (3, 5));
}

#[test]
fn point_read_reports_absent_as_zero() {
    let t = Tensor::try_from_entries(vec![TxEntry::new(1, 2, 3, 7)]).unwrap();
    assert_eq!(t.get(Coord::new(1, 2, 3)), 7);
    assert_eq!(t.get(Coord::new(1, 2, 4)), 0);
    assert!(!t.is_zero() && Tensor::zero().is_zero());
}

#[test]
fn add_and_neg_are_checked_and_total() {
    let a = Tensor::try_from_entries(vec![TxEntry::new(1, 0, 0, 1)]).unwrap();
    let b = Tensor::try_from_entries(vec![TxEntry::new(1, 0, 0, -1)]).unwrap();
    let s = a.try_add(&b).unwrap();
    assert!(s.is_zero()); // zero sums dropped (P5)
    let big = Tensor::try_from_entries(vec![TxEntry::new(1, 0, 0, i64::MAX)]).unwrap();
    assert!(big.try_add(&big).is_err()); // P2: no wrap
    assert!(big.try_neg().is_ok());
    let min = Tensor::try_from_entries(vec![TxEntry::new(1, 0, 0, i64::MIN)]).unwrap();
    assert!(min.try_neg().is_err()); // V is not a group (P2)
}
