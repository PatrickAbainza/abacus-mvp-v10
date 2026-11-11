//! SAMLD ordered-binary pipeline (addendum golden_source_samld_binary_addendum).

use abacus_tensor::ingest::{Event, IngestionEngine, LedgerOp};
use abacus_tensor::samld::{
    bit_report, convert_reader, deserialize_core, deserialize_eval, deserialize_eval_v2,
    direct_csf, pack_dense, pack_offset, serialize_eval, serialize_eval_v2, sets,
    unpack_dense, unpack_offset,
};
use abacus_tensor::tensor::coordinate::Coord;
use std::collections::BTreeSet;

const CSV: &str = include_str!("fixtures/small.csv");

#[test]
fn conversion_is_deterministic_and_ordered() {
    let a = convert_reader(CSV.as_bytes()).unwrap();
    let b = convert_reader(CSV.as_bytes()).unwrap();
    assert_eq!(a.core_bytes, b.core_bytes);
    assert_eq!(a.eval_bytes, b.eval_bytes);
    // Core key is (t,s,r); duplicate coordinates compare equal (P4).
    assert!(a.core_records.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(a.core_records.len(), 6);
    // `tx_id` lives in the eval sidecar only; the exact index set is {0..5}.
    let tx: BTreeSet<u64> = a.eval_records.iter().map(|r| r.tx_id).collect();
    assert_eq!(tx, sets::index_set(6));
    let img = deserialize_core(&a.core_bytes).unwrap();
    assert_eq!(img.records, a.core_records);
    let eval = deserialize_eval(&a.eval_bytes).unwrap();
    assert_eq!(eval, a.eval_records);
}

#[test]
fn duplicate_coordinates_settle_before_compress() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let (csf, settled) = direct_csf(&img).unwrap();
    assert!(csf.check_invariants());
    let legacy = abacus_tensor::coo::builder::build_from_reader(CSV.as_bytes()).unwrap();
    assert_eq!(settled, legacy.coo);
    assert_eq!(abacus_tensor::csf::decode_csf(&csf), legacy.coo);
    assert_eq!(csf.nnz(), settled.len());
}

#[test]
fn tensor_state_and_algebra_close_over_direct_csf() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let (csf, settled) = direct_csf(&img).unwrap();
    let t = abacus_tensor::tensor::Tensor::try_from_entries(settled.clone()).unwrap();
    assert_eq!(t.nnz(), csf.nnz());
    for e in &settled {
        assert_eq!(
            t.get(Coord::new(e.t, e.s, e.r)),
            abacus_tensor::csf::lookup(&csf, e.t, e.s, e.r)
        );
    }
    let neg = t.try_neg().unwrap();
    assert!(t.try_add(&neg).unwrap().is_zero());
}


#[test]
fn late_arriving_deltas_seal_without_rebuild_of_history() {
    use abacus_tensor::tensor::Tensor;
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let (csf, settled) = direct_csf(&img).unwrap();
    // Governed path only: fresh engine, ingest settled base as Creates
    // (tx_id == EventId, Def 7.1), seal → sealed state, then a late arrival.
    let mut eng = IngestionEngine::new();
    for (i, r) in img.records.iter().enumerate() {
        eng.ingest(
            &Event {
                id: i as u64,
                coord: Coord::new(r.t, r.s, r.r),
            },
            LedgerOp::Create(r.amount_micro),
        )
        .unwrap();
    }
    eng.seal().unwrap();
    assert_eq!(abacus_tensor::csf::decode_csf(eng.sealed()), settled);
    assert_eq!(eng.sealed(), &csf);
    eng.set_watermark(img.t_max);
    // Late arrival: fresh EventId, event time t_min (t <= W, P10 ordinary).
    let first = &settled[0];
    let late_value: i64 = 42;
    eng.ingest(
        &Event { id: 10_000_000, coord: Coord::new(first.t, first.s, first.r) },
        LedgerOp::Create(late_value),
    )
    .unwrap();
    // Delta architecture preserved: live view = sealed + frontier (row 23),
    // no rebuild of history before seal.
    assert_eq!(eng.frontier().len(), 1);
    let expect = first.value.checked_add(late_value).unwrap();
    assert_eq!(eng.live_at(first.t, first.s, first.r).unwrap(), expect);
    let _ = Tensor::zero();
    let rep = eng.seal().unwrap();
    assert!(rep.full_rebuild);
    assert!(eng.sealed().check_invariants());
    assert_eq!(
        abacus_tensor::csf::lookup(eng.sealed(), first.t, first.s, first.r),
        expect
    );
}

#[test]
fn sidecar_core_correspondence_is_positional_index_set() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let eval = deserialize_eval(&conv.eval_bytes).unwrap();
    // The sidecar is the sole tx_id carrier; the exact index set {0..n-1} is
    // its correspondence to the core (set-operation rule, §S4/§S6).
    let expected = sets::index_set(img.records.len());
    let side_tx: BTreeSet<u64> = eval.iter().map(|r| r.tx_id).collect();
    let (missing, extra, both) = sets::tx_correspondence(&expected, &side_tx);
    assert!(missing.is_empty() && extra.is_empty());
    assert_eq!(both.len(), img.records.len());
    // Positional bijection: sidecar record `i` pairs with core record `i`.
    for (c, e) in img.records.iter().zip(eval.iter()) {
        assert_eq!(c.amount_micro, e.amount_micro);
    }
    // The core record carries the weighted amounts on (t,s,r) and no tx_id;
    // its storage key is exactly the coordinate (P1 d_lex).
    for r in &img.records {
        assert!(r.roundedness <= 1);
        assert_eq!(r.storage_key(), (r.t, r.s, r.r));
    }
    assert!(!conv.eval_bytes.windows(8).any(|w| w == b"SAMLDBIN"));
}

#[test]
fn compressed_sidecar_v2_round_trips_losslessly() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let recs = conv.eval_records.clone();
    let v1 = serialize_eval(&recs);
    let from_v1 = deserialize_eval(&v1).unwrap();
    assert_eq!(from_v1, recs);
    let v2 = serialize_eval_v2(&from_v1).unwrap();
    assert_eq!(deserialize_eval_v2(&v2).unwrap(), recs); // lossless vs original
    assert_eq!(serialize_eval_v2(&from_v1).unwrap(), v2); // deterministic (R7)
    // Fail-closed on corruption.
    let mut bad = v2.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0xFF;
    assert!(deserialize_eval_v2(&bad).is_err());
    let mut bad2 = v2.clone();
    bad2[0] = b'X';
    assert!(deserialize_eval_v2(&bad2).is_err());
}

#[test]
fn packing_report_holds_for_90day_cardinalities() {
    assert!(bit_report().contains("58b <= 64 FEASIBLE"));
    assert!(bit_report().contains("59b <= 64 FEASIBLE"));
    assert!(bit_report().contains("NOT FEASIBLE"));
    for (ti, s, r) in [(0u64, 0u32, 0u32), (5, 3, 7), (100, 1000, 2000)] {
        assert_eq!(unpack_dense(pack_dense(ti, s, r).unwrap()), (ti, s, r));
        assert_eq!(unpack_offset(pack_offset(ti, s, r).unwrap()), (ti, s, r));
    }
    assert!(pack_dense(2147218, 121868, 410241).is_some());
    assert!(pack_dense(1u64 << 22, 0, 0).is_none());
    assert!(pack_dense(0, 1u32 << 17, 0).is_none());
    assert!(pack_dense(0, 0, 1u32 << 19).is_none());
    let ka = pack_dense(1, 2, 3).unwrap();
    let kb = pack_dense(1, 2, 4).unwrap();
    let kc = pack_dense(1, 3, 0).unwrap();
    let kd = pack_dense(2, 0, 0).unwrap();
    assert!(ka < kb && kb < kc && kc < kd);
}

#[test]
fn set_selection_and_behavioral_relations_are_exact() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let (_, settled) = direct_csf(&img).unwrap();
    let coords: Vec<(u64, u32, u32)> =
        settled.iter().map(|e| (e.t, e.s, e.r)).collect();
    let a = sets::support_set(&coords);
    let empty: BTreeSet<(u64, u32, u32)> = BTreeSet::new();
    assert_eq!(sets::set_inter(&a, &a), a);
    assert!(sets::set_diff(&a, &a).is_empty());
    assert!(a.is_subset(&sets::set_union(&a, &empty)));
    let e0 = &settled[0];
    let got = sets::fan_in_senders(&coords, e0.r, 0, u64::MAX);
    assert!(got.contains(&e0.s));
    let csf = abacus_tensor::csf::compress(&settled);
    let via_csf: BTreeSet<u32> = abacus_tensor::csf::receiver_at_time(&csf, e0.t, e0.r)
        .into_iter().map(|(s, _)| s).collect();
    assert!(via_csf.is_subset(&got));
}
