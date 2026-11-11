//! Rows 10–14 (§B.2): COO evaluation, canonicalization, dataset build.

use abacus_tensor::composites::epoch_seconds;
use abacus_tensor::coo::builder::build_from_reader;
use abacus_tensor::coo::decode::{decode_to_tensor, try_eval_at};
use abacus_tensor::coo::encode::{canonicalize, is_canonical, is_sorted};
use abacus_tensor::coo::entry::TxEntry;

const CSV: &str = include_str!("fixtures/small.csv");

#[test]
fn evaluation_sums_equal_coordinates() {
    let raw = vec![
        TxEntry::new(1, 0, 0, 2),
        TxEntry::new(1, 0, 0, 3),
        TxEntry::new(1, 0, 1, 9),
    ];
    assert_eq!(try_eval_at(&raw, 1, 0, 0).unwrap(), 5); // row 10 direct sum
    assert_eq!(try_eval_at(&raw, 1, 0, 1).unwrap(), 9);
    assert_eq!(try_eval_at(&raw, 9, 9, 9).unwrap(), 0); // absent
    let t = decode_to_tensor(&raw).unwrap(); // row 10 settling decode
    assert_eq!(t.nnz(), 2);
}

#[test]
fn canonicalize_is_settle_and_checks_are_d_inv() {
    let raw = vec![
        TxEntry::new(2, 0, 0, 1),
        TxEntry::new(1, 9, 0, 0), // zero in the raw input
        TxEntry::new(1, 9, 0, 4),
    ];
    assert!(!is_sorted(&raw) || is_sorted(&raw)); // smoke: predicate is total
    assert!(!is_canonical(&raw));
    let c = canonicalize(raw).unwrap();
    assert!(is_canonical(&c));
    assert_eq!(c.len(), 2); // zero dropped, no duplicates
}

#[test]
fn datetime_contract_yields_seconds_since_epoch() {
    assert_eq!(epoch_seconds("00:00:00", "1970-01-01").unwrap(), 0);
    assert_eq!(
        epoch_seconds("10:00:00", "2024-01-01").unwrap(),
        1_704_103_200
    );
    assert!(epoch_seconds("24:00:00", "2024-01-01").is_err());
    assert!(epoch_seconds("00:00:00", "1969-12-31").is_err()); // pre-epoch rejected
}

#[test]
fn dataset_build_settles_duplicates_and_assigns_dict_order() {
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    assert_eq!(build.raw_rows, 6);
    // The two Alice→Bob rows at 10:00 share a coordinate and aggregate;
    // the Bob→Alice ±7.25 rows cancel to zero and are dropped.
    assert_eq!(build.coo.len(), 3);
    // prec_dict: codes follow ascending identifier byte order.
    assert!(build.senders.code_of("Alice") < build.senders.code_of("Carol"));
    assert_eq!(build.t_min, 1_704_103_200);
    assert_eq!(
        build.t_max,
        epoch_seconds("23:59:59", "2024-01-03").unwrap()
    );
    // Dictionary lookup failure mode is total (no panic).
    assert_eq!(build.senders.code_of("Nobody"), None);
}

#[test]
fn dataset_build_rejects_malformed_input() {
    assert!(build_from_reader("".as_bytes()).is_err());
    assert!(build_from_reader("Time,Date\n10:00:00".as_bytes()).is_err());
}
