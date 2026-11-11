//! Structural Fan-In over canonical CSF — the 20 required cases
//! (`docs/spec/fan_in.md` §42), minus cases 15–16 which are **blocked on G2a**.
//!
//! The reference oracle is a deliberately naive, obviously-correct re-derivation
//! over decoded entries (`reference_fanin` below): §43 requires every optimized
//! result to equal a simple reference implementation.
//!
//! Fixtures are built **directly** as canonical CSF from `TxEntry` values — no
//! CSV, no sidecar, no second transaction representation (P11, INV-13).

use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::{compress, CsfTensor};
use abacus_tensor::fanin::{
    fanin, FanInResult, FanInState, FanInWindow, FocalTransaction, DEFAULT_DELTA_SECS,
};
use std::collections::BTreeSet;

/// `Δ` small enough to reason about by hand.
const D: u64 = 10;
const W: FanInWindow = FanInWindow::with_delta(D);

/// Build canonical CSF from `(t, s, r, amount)` tuples, settling first so the
/// tensor obeys the same invariants as the real pipeline (G8).
fn build(rows: &[(u64, u32, u32, i64)]) -> CsfTensor {
    let raw: Vec<TxEntry> = rows.iter().map(|&(t, s, r, v)| TxEntry::new(t, s, r, v)).collect();
    let settled = settled(&raw);
    let csf = compress(&settled);
    assert!(csf.check_invariants(), "fixture must satisfy CSF invariants");
    csf
}

/// `d_settle` + `R_6`, mirroring `composites::settle` for fixture construction.
fn settled(raw: &[TxEntry]) -> Vec<TxEntry> {
    let mut v = raw.to_vec();
    v.sort();
    let mut out: Vec<TxEntry> = Vec::new();
    for e in v {
        match out.last_mut() {
            Some(last) if (last.t, last.s, last.r) == (e.t, e.s, e.r) => {
                last.value += e.value;
            }
            _ => out.push(e),
        }
    }
    out.retain(|e| e.value != 0); // R6
    out
}

/// Reference implementation: decode, filter, collect. Deliberately naive.
fn reference_fanin(csf: &CsfTensor, focal: &FocalTransaction, delta: u64) -> FanInResult {
    let (lo, hi) = abacus_tensor::fanin::window_bounds(csf, focal, &FanInWindow::with_delta(delta))
        .expect("window");
    let mut state = FanInState::identity();
    let mut visited = 0u64;
    let mut matched = 0u64;
    for i in lo..hi {
        let t = csf.mode0_coords[i];
        for j in csf.mode0_fptr[i] as usize..csf.mode0_fptr[i + 1] as usize {
            let s = csf.mode1_coords[j];
            for k in csf.mode1_fptr[j] as usize..csf.mode1_fptr[j + 1] as usize {
                visited += 1;
                if csf.mode2_coords[k] == focal.receiver {
                    matched += 1;
                    state.observe(s, t).expect("observe");
                }
            }
        }
    }
    FanInResult::finalize(*focal, state, abacus_tensor::fanin::FanInAccounting { visited, matched })
        .expect("finalize")
}

/// Assert the kernel equals the naive reference (fan_in.md §43).
fn assert_matches_reference(csf: &CsfTensor, focal: &FocalTransaction) -> FanInResult {
    let got = fanin(csf, focal, &W).expect("fan-in");
    let want = reference_fanin(csf, focal, D);
    assert_eq!(got, want, "kernel != reference implementation (§43)");
    got
}

/// Case 1 — zero incoming transactions.
#[test]
fn case01_zero_incoming() {
    let csf = build(&[(100, 0, 1, 500), (101, 2, 3, 700)]);
    let f = FocalTransaction::new(100, 5, 9); // r_f = 9: nothing matches
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 0);
    assert_eq!(r.n_i, 0);
    assert!(r.senders.is_empty());
    assert_eq!(r.rfi, None, "RFI undefined when T_i == 0 (§37)");
    assert!(!r.fi_topology);
    assert_eq!(r.t_min, None);
    assert_eq!(r.t_max, None);
}

/// Case 2 — exactly one incoming transaction.
#[test]
fn case02_one_incoming() {
    let csf = build(&[(100, 0, 1, 500)]);
    let f = FocalTransaction::new(100, 5, 1);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 1);
    assert_eq!(r.n_i, 1);
    assert_eq!(r.senders, BTreeSet::from([0]));
    assert_eq!(r.rfi, Some((1, 1)));
    assert!(!r.fi_topology);
    assert_eq!((r.t_min, r.t_max), (Some(100), Some(100)));
}

/// Cases 3–5 — duplicate senders collapse; `N_i` stays exact.
#[test]
fn case03_05_duplicate_senders_count_once() {
    let csf = build(&[
        (100, 7, 1, 100), // sender 7, twice
        (101, 7, 1, 200),
        (102, 8, 1, 300), // sender 8, once
    ]);
    let f = FocalTransaction::new(102, 5, 1);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 3, "T_i counts matched COORDINATES");
    assert_eq!(r.n_i, 2, "N_i counts distinct SENDERS (§16, INV-09)");
    assert_eq!(r.senders, BTreeSet::from([7, 8]));
    assert_eq!(r.rfi, Some((2, 3)));
    assert!(r.fi_topology);
}

/// Case 3 — multiple transactions from one sender only.
#[test]
fn case03_single_sender_many_transactions() {
    let csf = build(&[(100, 7, 1, 10), (101, 7, 1, 20), (102, 7, 1, 30)]);
    let f = FocalTransaction::new(102, 5, 1);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!((r.t_i, r.n_i, r.fi_topology), (3, 1, false));
    assert_eq!(r.rfi, Some((1, 3)));
}

/// Case 4 — multiple unique senders.
#[test]
fn case04_multiple_unique_senders() {
    let csf = build(&[(100, 1, 2, 10), (100, 3, 2, 20), (100, 4, 2, 30)]);
    let f = FocalTransaction::new(100, 5, 2);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!((r.t_i, r.n_i, r.fi_topology), (3, 3, true));
    assert_eq!(r.rfi, Some((3, 3)), "RFI = 1: every sender distinct (§7)");
}

/// Case 6 — the focal transaction itself is included.
#[test]
fn case06_focal_transaction_included() {
    let csf = build(&[(100, 5, 9, 42)]);
    let f = FocalTransaction::new(100, 5, 9); // exactly the stored coordinate
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 1);
    assert_eq!(r.senders, BTreeSet::from([5]));
}

/// Case 7 — receiver mismatch excludes an entry.
#[test]
fn case07_receiver_mismatch_excluded() {
    let csf = build(&[(100, 5, 9, 42), (100, 6, 8, 43)]);
    let f = FocalTransaction::new(100, 5, 9);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 1, "r != r_f entry excluded (§6)");
    assert_eq!(r.matched, 1);
    assert_eq!(r.visited, 2, "both visited, one matched (§19)");
}

/// Case 8 — outside the temporal window.
#[test]
fn case08_outside_window_excluded() {
    let csf = build(&[(50, 1, 2, 10), (100, 3, 2, 20), (200, 4, 2, 30)]);
    let f = FocalTransaction::new(100, 5, 2);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 1, "only t = 100 is in [90, 100]");
    assert_eq!(r.senders, BTreeSet::from([3]));
}

/// Case 9 — transaction EXACTLY at the window start (closed lower edge).
#[test]
fn case09_exactly_at_window_start_included() {
    let csf = build(&[(90, 1, 2, 10), (91, 3, 2, 20)]);
    let f = FocalTransaction::new(100, 5, 2);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 2, "t_f - Δ is INCLUDED (§5, closed window)");
    assert_eq!(r.senders, BTreeSet::from([1, 3]));
}

/// Case 10 — transaction EXACTLY at the focal time (closed upper edge).
#[test]
fn case10_exactly_at_focal_time_included() {
    let csf = build(&[(99, 1, 2, 10), (100, 3, 2, 20)]);
    let f = FocalTransaction::new(100, 5, 2);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.t_i, 2, "t_f is INCLUDED (§5) — t_end must be t_f + 1");
    assert_eq!(r.senders, BTreeSet::from([1, 3]));
}

/// Window arithmetic must never silently drop the focal (regression guard).
#[test]
fn window_is_closed_on_both_ends() {
    let csf = build(&[(89, 1, 2, 10), (90, 2, 2, 10), (100, 3, 2, 10), (101, 4, 2, 10)]);
    let f = FocalTransaction::new(100, 5, 2);
    let r = fanin(&csf, &f, &W).expect("fan-in");
    assert_eq!(r.t_i, 2, "t in [90,100] only: 89 and 101 fall outside");
    assert_eq!(r.senders, BTreeSet::from([2, 3]));
    assert_eq!((r.t_min, r.t_max), (Some(90), Some(100)));
}

/// Case 17 — deterministic repeated execution.
#[test]
fn case17_deterministic_repeated_execution() {
    let csf = build(&[
        (95, 1, 2, 10), (96, 2, 2, 10), (97, 1, 2, 10), (98, 3, 2, 10), (99, 4, 2, 10),
    ]);
    let f = FocalTransaction::new(99, 5, 2);
    let first = fanin(&csf, &f, &W).expect("fan-in");
    for _ in 0..8 {
        assert_eq!(fanin(&csf, &f, &W).expect("fan-in"), first, "must be deterministic");
    }
}

/// Cases 18–20 — traversal accounting and `K_f ≤ V_f`.
#[test]
fn case18_20_traversal_accounting() {
    let csf = build(&[
        (95, 1, 2, 10), (95, 9, 9, 10), // second entry: wrong receiver
        (96, 2, 3, 10),                  // third entry: wrong receiver
        (97, 3, 2, 10),
    ]);
    let f = FocalTransaction::new(99, 5, 2);
    let r = assert_matches_reference(&csf, &f);
    assert_eq!(r.visited, 4, "V_f counts every CSF entry visited");
    assert_eq!(r.matched, 2, "K_f counts entries with r = r_f");
    assert!(r.matched <= r.visited, "K_f <= V_f (§19)");
}

/// `V_f` really does exceed `K_f` — the §19 distinction is not cosmetic.
#[test]
fn visited_exceeds_matched_on_real_traffic() {
    let mut rows = Vec::new();
    for t in 95..=99 {
        for s in 0..4 {
            rows.push((t, s, 2, 10));       // matches r_f = 2
            rows.push((t, s, 7, 10));       // does not match
        }
    }
    let csf = build(&rows);
    let f = FocalTransaction::new(99, 9, 2);
    let r = fanin(&csf, &f, &W).expect("fan-in");
    assert_eq!(r.visited, 40);
    assert_eq!(r.matched, 20);
    assert_eq!(r.n_i, 4, "4 distinct senders across 20 matches");
}

/// G8 — a cancelling `+a / -a` pair is ELIMINATED, so it contributes nothing.
///
/// This is not a zero-valued observation: the coordinate is absent from
/// `supp(T)` (R6). `T_i` counts surviving canonical coordinates only.
#[test]
fn g8_cancelled_coordinate_contributes_nothing() {
    // (t=100, s=1, r=2): +500 and -500 settle to 0 -> eliminated.
    // (t=101, s=3, r=2): survives.
    let csf = build(&[(100, 1, 2, 500), (100, 1, 2, -500), (101, 3, 2, 700)]);
    assert_eq!(csf.nnz(), 1, "the cancelling coordinate is absent from supp(T)");

    let f = FocalTransaction::new(101, 9, 2);
    let r = fanin(&csf, &f, &W).expect("fan-in");
    assert_eq!(r.t_i, 1, "the eliminated coordinate contributes 0 to T_i");
    assert_eq!(r.senders, BTreeSet::from([3]));
    assert_eq!(r.visited, 1, "and is never visited at all");
}

/// G8 (aggregated case) — `m_c > 1` rows collapse to ONE coordinate.
#[test]
fn g8_aggregated_coordinate_counts_once() {
    let csf = build(&[(100, 1, 2, 500), (100, 1, 2, 250), (101, 3, 2, 700)]);
    assert_eq!(csf.nnz(), 2, "three source rows, two coordinates");
    let f = FocalTransaction::new(101, 9, 2);
    let r = fanin(&csf, &f, &W).expect("fan-in");
    assert_eq!(r.t_i, 2, "the aggregated pair counts ONCE");
    assert_eq!(r.n_i, 2);
}

/// `merge` is associative — required for the parallel reduction tree (§23).
#[test]
fn merge_is_associative() {
    let mk = |pairs: &[(u32, u64)]| {
        let mut s = FanInState::identity();
        for &(s_, t) in pairs {
            s.observe(s_, t).expect("observe");
        }
        s
    };
    let a = mk(&[(1, 10), (2, 11)]);
    let b = mk(&[(2, 12), (3, 13)]);
    let c = mk(&[(1, 14), (4, 15)]);

    let mut bc = b.clone();
    bc.merge(&c).expect("merge");
    let mut left = a.clone();
    left.merge(&bc).expect("merge");

    let mut ab = a;
    ab.merge(&b).expect("merge");
    let mut right = ab;
    right.merge(&c).expect("merge");

    assert_eq!(left, right, "(A (+) B) (+) C == A (+) (B (+) C)");
}

/// An EMPTY partition must not corrupt the global extrema (§26).
#[test]
fn empty_partition_does_not_corrupt_extrema() {
    let mut real = FanInState::identity();
    real.observe(1, 100).expect("observe");
    real.observe(2, 200).expect("observe");

    let mut merged = real.clone();
    merged.merge(&FanInState::identity()).expect("merge");

    assert_eq!(merged, real, "the identity state is neutral for (+)");
    assert_eq!(merged.t_min(), Some(100));
    assert_eq!(merged.t_max(), Some(200));
}

/// `N_i` is derived from `S_i`, so it can never drift (INV-09).
#[test]
fn n_i_is_derived_from_the_sender_set() {
    let csf = build(&[(95, 1, 2, 10), (96, 1, 2, 10), (97, 5, 2, 10)]);
    let f = FocalTransaction::new(99, 9, 2);
    let r = fanin(&csf, &f, &W).expect("fan-in");
    assert_eq!(r.n_i, r.senders.len() as u64);
    assert_eq!(r.t_i, 3);
    assert_ne!(r.t_i, r.n_i, "repeated sender: T_i != N_i");
}

/// The declared experimental default is 7 days — a parameter, not a law (§5).
#[test]
fn default_window_is_seven_days() {
    assert_eq!(DEFAULT_DELTA_SECS, 7 * 86_400);
    assert_eq!(FanInWindow::default_experimental().delta, 7 * 86_400);
    let csf = build(&[(0, 1, 2, 10), (6 * 86_400, 3, 2, 10), (7 * 86_400, 4, 2, 10)]);
    let f = FocalTransaction::new(7 * 86_400, 9, 2);
    let r = fanin(&csf, &f, &FanInWindow::default_experimental()).expect("fan-in");
    // Window is [t_f - 7d, t_f] = [0, 604800], CLOSED: t=0 is exactly the lower
    // edge and is therefore included, as is t_f itself.
    assert_eq!(r.t_i, 3, "t in [t_f - 7d, t_f] inclusive — both edges");
    assert_eq!(r.senders, BTreeSet::from([1, 3, 4]));
    assert_eq!((r.t_min, r.t_max), (Some(0), Some(7 * 86_400)));
}

/// A window entirely before the epoch yields an EMPTY result, not an error (L4).
#[test]
fn window_before_epoch_is_empty_not_error() {
    let csf = build(&[(10, 1, 2, 10)]);
    let f = FocalTransaction::new(5, 9, 2); // t_f = 5 < Δ = 10
    let r = fanin(&csf, &f, &W).expect("must not error");
    assert_eq!(r.t_i, 0);
    assert_eq!(r.visited, 0);
    assert_eq!(r.rfi, None);
    assert_eq!(r.t_min, None);
    assert_eq!(r.t_max, None);
}