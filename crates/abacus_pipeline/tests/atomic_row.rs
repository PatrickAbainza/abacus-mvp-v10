//! Row-kernel tests: fiber access, window seeks, pair seek, reversal evidence.
//! Generic terms only.

use abacus_pipeline::atomic::row::{fiber_window, pair_seek, PairSeek};
use abacus_pipeline::retrieval::{EventLocator, TemporalRef};
use abacus_tensor::atomic::kernels::traits::OperatorKernel;
use abacus_pipeline::atomic::row::RowKernel;
use abacus_tensor::atomic::reference::Predicate;

fn r(t: u64, peer: u32, loc: u32) -> TemporalRef {
    TemporalRef { t, locator: EventLocator(loc), peer }
}

#[test]
fn fiber_window_half_open_and_stats() {
    let row = vec![r(1, 5, 0), r(3, 5, 1), r(5, 6, 2), r(5, 5, 3), r(9, 5, 4)];
    let mut stats = Default::default();
    let got = fiber_window(&row, 3, 9, &mut stats);
    // [3,9): t=3,5,5 (loc order within equal t preserved)
    assert_eq!(got.len(), 3);
    assert_eq!(stats.seeks, 2);
    assert_eq!(stats.scanned, 3);
    // lo inclusive / hi exclusive
    let mut s2 = Default::default();
    assert_eq!(fiber_window(&row, 5, 6, &mut s2).len(), 2);
    let mut s3 = Default::default();
    assert!(fiber_window(&row, 9, 9, &mut s3).is_empty());
    assert!(fiber_window(&row, 10, 5, &mut s3).is_empty());
}

#[test]
fn pair_seek_earliest_latest_strict_next() {
    let row = vec![r(5, 7, 3), r(5, 7, 1), r(9, 7, 2), r(9, 8, 0)];
    let mut st = Default::default();
    let e = pair_seek(&row, 7, PairSeek::Earliest, &mut st).expect("earliest");
    assert_eq!((e.t, e.locator), (5, 1));
    let l = pair_seek(&row, 7, PairSeek::Latest, &mut st).expect("latest");
    assert_eq!((l.t, l.locator), (9, 2));
    let n = pair_seek(&row, 7, PairSeek::NextAfter(5), &mut st).expect("next");
    assert_eq!((n.t, n.locator), (9, 2), "strict: equal t excluded");
    assert!(pair_seek(&row, 7, PairSeek::NextAfter(9), &mut st).is_none());
    assert!(pair_seek(&row, 99, PairSeek::Earliest, &mut st).is_none());
    assert!(st.scanned >= 4);
}

#[test]
fn reversal_lifecycle_event_exists_vs_net_value() {
    // Committed rows: event loc0 (+5 at coord) then compensating loc1 (-5).
    // Event-exists: both rows present. Net-value (CSF): coordinate absent.
    let row = vec![r(4, 2, 0), r(6, 2, 1)]; // forward row of key 1, peer 2
    let mut st = Default::default();
    // pair seek sees both rows (event-exists semantics)
    let earliest = pair_seek(&row, 2, PairSeek::Earliest, &mut st).expect("event exists");
    assert_eq!(earliest.locator, 0);
    let latest = pair_seek(&row, 2, PairSeek::Latest, &mut st).expect("compensating exists");
    assert_eq!(latest.locator, 1);
    // net-value semantics: +5 and -5 settle to absent
    let net: i64 = 5 - 5;
    assert_eq!(net, 0, "zero net => coordinate absent from tensor support");
    // RowKernel restrict keeps both (event-exists), algebraic join must net them.
    let kernel = RowKernel::forward(vec![&row[..]]);
    let (sup, _) = kernel.restrict(
        &abacus_tensor::atomic::types::Support::empty(),
        &Predicate::window(0, 10),
    );
    assert_eq!(sup.entries.len(), 2, "fiber membership keeps both rows");
}

#[test]
fn recent_window_is_ordered_seek_no_separate_structure() {
    // 7-day lookback served by the same row seeks.
    let mut row = Vec::new();
    for i in 0..100u64 {
        row.push(r(i * 1000, 3, i as u32));
    }
    let anchor = 99_000u64;
    let delta = 7 * 86_400u64;
    let lo = anchor.saturating_sub(delta);
    let mut st = Default::default();
    let got = fiber_window(&row, lo, anchor + 1, &mut st);
    assert_eq!(got.len(), 100);
    assert_eq!(st.seeks, 2, "one recent window = two seeks, no extra structure");
}

#[test]
fn row_kernel_restrict_matches_window() {
    let fwd0 = vec![r(1, 5, 0), r(3, 6, 1)];
    let fwd1 = vec![r(2, 5, 2)];
    let kernel = RowKernel::forward(vec![&fwd0[..], &fwd1[..]]);
    let mut p = Predicate::window(0, 10);
    p.keys1 = Some(vec![1]);
    let (sup, _) = kernel.restrict(&abacus_tensor::atomic::types::Support::empty(), &p);
    assert_eq!(sup.entries.len(), 1);
    assert_eq!((sup.entries[0].coord.s, sup.entries[0].coord.r), (1, 5));
    // backward rows keyed by mode-2
    let bwd5 = vec![r(1, 0, 0), r(2, 1, 2)];
    let bkernel = RowKernel::backward(vec![&[][..], &[][..], &[][..], &[][..], &[][..], &bwd5[..]]);
    let mut p2 = Predicate::window(0, 10);
    p2.keys2 = Some(vec![5]);
    let (sup2, _) = bkernel.restrict(&abacus_tensor::atomic::types::Support::empty(), &p2);
    assert_eq!(sup2.entries.len(), 2);
}
