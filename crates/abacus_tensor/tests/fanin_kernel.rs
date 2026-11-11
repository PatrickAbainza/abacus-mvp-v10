//! The 14 required tests for the Abacus-native TACO-style fused sparse kernel
//! (`your_2nd_task.md` §22).
//!
//! **Oracle** (§21): the existing canonical `abacus_tensor::fanin::fanin` for the
//! structural fields, plus an independent naive re-derivation over the decoded
//! entries for `M_log` and `C_R`. TACO is **not** an oracle — it cannot express
//! these operations.
//!
//! Fixtures are built **directly** as canonical CSF from `TxEntry` values: no
//! CSV, no sidecar, no second transaction representation (INV-13).

use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::{compress, CsfTensor};
use abacus_tensor::fanin::{fanin, FanInWindow, FocalTransaction};
use abacus_tensor::fanin_kernel::{
    fused_fanin, FusedFanInState, ReductionSet, SparseExecutionPlan, M_LOG_SCALE,
};
use abacus_tensor::samld::diag;
use abacus_tensor::TensorError;

/// `Δ = 604800 s` (7 days) as mandated by `your_2nd_task.md` §6.
const D: u64 = 604_800;
const W: FanInWindow = FanInWindow::with_delta(D);

/// Build canonical CSF from `(t, s, r, amount)` tuples, settling first so the
/// fixture obeys the same invariants as the real pipeline (G8).
fn build(rows: &[(u64, u32, u32, i64)]) -> CsfTensor {
    let raw: Vec<TxEntry> = rows.iter().map(|&(t, s, r, v)| TxEntry::new(t, s, r, v)).collect();
    let settled = settle(&raw);
    let csf = compress(&settled);
    assert!(csf.check_invariants(), "fixture must satisfy CSF invariants");
    csf
}

fn settle(raw: &[TxEntry]) -> Vec<TxEntry> {
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
    out
}

/// Independent naive oracle for `M_log` and `C_R`: decode the CSF, apply the
/// §S5 definitions, and filter by `t ∈ W_f ∧ r == r_f` (no fused kernel).
fn reference_behavioral(
    csf: &CsfTensor,
    focal: FocalTransaction,
    window: FanInWindow,
) -> (i128, u64) {
    let lo = focal.epoch.saturating_sub(window.delta);
    let mut m_log: i128 = 0;
    let mut c_r: u64 = 0;
    for i in 0..csf.mode0_coords.len() {
        let t = csf.mode0_coords[i];
        if t < lo || t > focal.epoch {
            continue;
        }
        for j in csf.mode0_fptr[i] as usize..csf.mode0_fptr[i + 1] as usize {
            for k in csf.mode1_fptr[j] as usize..csf.mode1_fptr[j + 1] as usize {
                if csf.mode2_coords[k] != focal.receiver {
                    continue;
                }
                let a = csf.values[k];
                // Same quantization as the kernel, written independently.
                let major = a as f64 / 1e6;
                let scaled = diag::log1p_signed(major) * M_LOG_SCALE;
                m_log += scaled.round() as i128;
                c_r += u64::from(diag::roundedness_of(a));
            }
        }
    }
    (m_log, c_r)
}

/// Assert the fused kernel agrees with BOTH oracles on every field.
fn assert_matches_reference(
    csf: &CsfTensor,
    focal: FocalTransaction,
    window: FanInWindow,
    label: &str,
) {
    let got = fused_fanin(csf, focal, window).unwrap_or_else(|e| panic!("{label}: {e}"));

    // Oracle 1: the existing canonical structural implementation.
    let r = fanin(csf, &focal, &window).unwrap_or_else(|e| panic!("{label}: ref {e}"));
    assert_eq!(got.t_i, r.t_i, "{label}: T_i");
    assert_eq!(got.senders, r.senders, "{label}: S_i");
    assert_eq!(got.n_i, r.n_i, "{label}: N_i");
    assert_eq!(got.rfi, r.rfi, "{label}: RFI");
    assert_eq!(got.fi_topology, r.fi_topology, "{label}: FI_topology");
    assert_eq!(got.t_min, r.t_min, "{label}: t_min");
    assert_eq!(got.t_max, r.t_max, "{label}: t_max");
    assert_eq!(got.visited, r.visited, "{label}: V_f");
    assert_eq!(got.matched, r.matched, "{label}: K_f");

    // Oracle 2: naive behavioral re-derivation.
    let (m_log, c_r) = reference_behavioral(csf, focal, window);
    assert_eq!(got.m_log, m_log, "{label}: M_log");
    assert_eq!(got.c_r, c_r, "{label}: C_R");
}

/// Helper: the expected quantized `M_log` contribution for `amount_micro`.
fn fused_q(amount_micro: i64) -> i128 {
    (diag::log1p_signed(amount_micro as f64 / 1e6) * M_LOG_SCALE).round() as i128
}

/// **Test 1** — a single qualifying transaction produces the expected state.
#[test]
fn t1_single_transaction() {
    let csf = build(&[(T0, 5, 9, 2_000_000)]);
    let f = FocalTransaction::new(10_000_000, 5, 9);
    let r = fused_fanin(&csf, f, W).expect("t1");
    assert_eq!(r.t_i, 1);
    assert_eq!(r.n_i, 1);
    assert_eq!(r.c_r, 1, "2_000_000 is a whole number of major units");
    assert_eq!(r.m_log, fused_q(2_000_000));
    assert_eq!(r.t_min, Some(T0));
    assert_eq!(r.t_max, Some(T0));
    assert_matches_reference(&csf, f, W, "t1");
}

/// **Test 2** — the distinct sender set is exact.
#[test]
fn t2_multiple_senders() {
    let csf = build(&[
        (10_000_000, 1, 9, 1_000_000),
        (T0, 2, 9, 1_000_000),
        (T0, 3, 9, 1_000_000),
    ]);
    let f = FocalTransaction::new(T0, 7, 9);
    let r = fused_fanin(&csf, f, W).expect("t2");
    assert_eq!(r.t_i, 3);
    assert_eq!(r.n_i, 3);
    assert!(r.fi_topology);
    assert_matches_reference(&csf, f, W, "t2");
}

/// **Test 3** — a repeated sender does not increase the distinct count.
#[test]
fn t3_repeated_sender_not_counted_twice() {
    let csf = build(&[
        (T0, 1, 9, 1_000_000),
        (T0 + 1, 1, 9, 1_000_000),
        (T0 + 2, 1, 9, 1_000_000),
    ]);
    let f = FocalTransaction::new(T0 + 2, 5, 9);
    let r = fused_fanin(&csf, f, W).expect("t3");
    assert_eq!(r.t_i, 3, "three transactions");
    assert_eq!(r.n_i, 1, "but only ONE distinct sender");
    assert!(!r.fi_topology);
    assert_matches_reference(&csf, f, W, "t3");
}

/// **Test 4** — transactions with `r != r_f` are excluded.
#[test]
fn t4_receiver_mismatch_excluded() {
    let csf = build(&[
        (10_000_000, 1, 9, 1_000_000), // r == r_f  -> counted
        (10_000_000, 2, 8, 1_000_000), // r != r_f  -> excluded
        (10_000_000, 3, 7, 1_000_000), // r != r_f  -> excluded
    ]);
    let f = FocalTransaction::new(10_000_000, 5, 9);
    let r = fused_fanin(&csf, f, W).expect("t4");
    assert_eq!(r.t_i, 1);
    assert_eq!(r.senders.len(), 1);
    assert_eq!(r.visited, 3, "all three visited");
    assert_eq!(r.matched, 1, "only one matched");
    assert_matches_reference(&csf, f, W, "t4");
}

/// **Test 5** — a transaction exactly at `t_f − 604800` is INCLUDED.
#[test]
fn t5_temporal_lower_boundary_included() {
    let csf = build(&[(10_000_000 - D, 1, 9, 1_000_000)]);
    let f = FocalTransaction::new(10_000_000, 5, 9);
    assert_eq!(fused_fanin(&csf, f, W).expect("t5").t_i, 1);
    assert_matches_reference(&csf, f, W, "t5");
}

/// **Test 6** — a transaction exactly at `t_f` is INCLUDED.
#[test]
fn t6_temporal_upper_boundary_included() {
    let csf = build(&[(10_000_000, 1, 9, 1_000_000)]);
    let f = FocalTransaction::new(10_000_000, 5, 9);
    assert_eq!(fused_fanin(&csf, f, W).expect("t6").t_i, 1);
    assert_matches_reference(&csf, f, W, "t6");
}

/// **Test 7** — transactions outside the closed interval are excluded.
#[test]
fn t7_outside_window_excluded() {
    let csf = build(&[
        (10_000_000 - D - 1, 1, 9, 1_000_000), // one second too early
        (10_000_000, 1, 9, 1_000_000),         // inside
        (10_000_000 + 1, 1, 9, 1_000_000),     // one second too late
    ]);
    let f = FocalTransaction::new(10_000_000, 5, 9);
    let r = fused_fanin(&csf, f, W).expect("t7");
    assert_eq!(r.t_i, 1);
    assert_eq!(r.t_min, Some(T0));
    assert_eq!(r.t_max, Some(T0));
    assert_matches_reference(&csf, f, W, "t7");
}
/// An epoch safely beyond `D`, so `t_f - D` cannot underflow.
const T0: u64 = 10_000_000;

/// **Test 8** — amounts divisible by `1e6` increment `C_R`.
#[test]
fn t8_roundedness_counts_whole_major_units() {
    let csf = build(&[
        (T0, 1, 9, 1_000_000),
        (T0, 2, 9, 2_000_000),
        (T0, 3, 9, 0),
        (T0, 4, 9, 12_000_000),
    ]);
    let f = FocalTransaction::new(T0, 5, 9);
    assert_eq!(fused_fanin(&csf, f, W).expect("t8").c_r, 4);
    assert_matches_reference(&csf, f, W, "t8");
}

/// **Test 9** — amounts not divisible by `1e6` do not increment `C_R`.
#[test]
fn t9_non_roundedness_excluded() {
    let csf = build(&[
        (T0, 1, 9, 1_000_001),
        (T0, 2, 9, 1_500_000),
        (T0, 3, 9, -1_000_001), // negative, also not whole
    ]);
    let f = FocalTransaction::new(T0, 5, 9);
    let r = fused_fanin(&csf, f, W).expect("t9");
    assert_eq!(r.c_r, 0);
    assert_eq!(r.t_i, 3, "still matched and counted");
    assert_matches_reference(&csf, f, W, "t9");
}

/// **Test 10** — `M_log` quantization matches the specified definition exactly,
/// including zero and negative amounts.
#[test]
fn t10_m_log_quantization_exact() {
    // Zero amount -> q == 0 exactly.
    assert_eq!(fused_q(0), 0, "log1p_signed(0) == 0");

    // Positive: ln(1 + 1 major unit) = ln 2.
    let expected = (std::f64::consts::LN_2 * M_LOG_SCALE).round() as i128;
    let one_major = fused_q(1_000_000);
    assert!(one_major > 0, "ln(2) > 0 so q > 0, got {one_major}");
    assert_eq!(one_major, expected, "q = round(2^64 * ln(2))");

    // Negative: sign-preserving, exactly symmetric.
    assert_eq!(fused_q(-1_000_000), -expected, "sign-symmetric for +/- 1 major");

    // Positive / negative / zero through the kernel: exact integers must cancel
    // to exactly zero (no float residue).
    let csf = build(&[
        (T0, 1, 9, 1_000_000),
        (T0, 2, 9, -1_000_000),
        (T0, 3, 9, 0),
    ]);
    let f = FocalTransaction::new(T0, 5, 9);
    let r = fused_fanin(&csf, f, W).expect("t10");
    assert_eq!(r.m_log, 0, "q + (-q) + 0 must cancel EXACTLY in i128");
    assert_matches_reference(&csf, f, W, "t10");
}

/// **Test 11** — `M_log` overflow produces `TensorError::Range`: NOT wrapping
/// and NOT saturation. Driven through the real `merge` API, since the authorized
/// domain has ~3e17 headroom and a traversal could never reach the bound.
#[test]
fn t11_m_log_overflow_errors_not_wraps_or_saturates() {
    let mut a = FusedFanInState::with_m_log_for_test(i128::MAX);
    let b = FusedFanInState::with_m_log_for_test(1);
    match a.merge(&b) {
        Err(TensorError::Range(_)) => {}              // expected: fail-closed
        Err(other) => panic!("wrong error class: {other:?}"),
        Ok(()) => panic!("M_log merge overflowed silently — forbidden"),
    }

    // Symmetrically at the negative bound.
    let mut lo = FusedFanInState::with_m_log_for_test(i128::MIN);
    let lo2 = FusedFanInState::with_m_log_for_test(-1);
    assert!(
        matches!(lo.merge(&lo2), Err(TensorError::Range(_))),
        "negative-bound overflow must also fail closed"
    );
}

/// **Test 14 (structural/behavioral half)** — the complete authorized fused
/// state matches the reference on a mixed workload: duplicates, both window
/// boundaries, negative/zero amounts, rounded and non-rounded, and
/// non-matching receivers. Parallel halves live in `abacus_execution`.
#[test]
fn t14_full_authorized_fusion() {
    let rows: Vec<(u64, u32, u32, i64)> = vec![
        (T0 - D, 1, 9, 1_000_000),     // lower boundary, whole
        (T0 - 100, 1, 9, 2_000_000),   // duplicate sender 1, whole
        (T0 - 50, 2, 9, -1_500_000),   // negative, NOT whole
        (T0 - 10, 3, 9, 0),            // zero (whole per S5)
        (T0, 4, 9, 5_000_000),         // upper boundary, whole
        (T0 - 20, 5, 8, 9_000_000),     // r != r_f -> excluded
        (T0 - 5, 6, 7, 1_000_000),      // r != r_f -> excluded from C_R
        (T0 + 1, 7, 9, 1_000_000),      // too late
        (T0 - D - 1, 8, 9, 1_000_000),  // too early
    ];
    let csf = build(&rows);
    let f = FocalTransaction::new(T0, 99, 9);
    assert_matches_reference(&csf, f, W, "t14");

    let serial = fused_fanin(&csf, f, W).expect("t14 serial");
    assert_eq!(serial.t_i, 5, "only the five r==9 in-window transactions");
    assert_eq!(serial.n_i, 4, "senders 1,2,3,4 (sender 1 twice)");
    assert_eq!(serial.c_r, 4, "1e6, 2e6, 5e6 whole; 0 is whole per S5 (0 mod 1e6 = 0); -1.5e6 is not");
    assert_eq!(serial.t_min, Some(T0 - D));
    assert_eq!(serial.t_max, Some(T0));
}

/// A non-authorized reduction set must be **rejected**, not silently applied
/// (§7: all authorized reductions are required; a partial set under-reports).
#[test]
fn partial_reduction_set_is_rejected() {
    let csf = build(&[(T0, 1, 9, 1_000_000)]);
    let f = FocalTransaction::new(T0, 5, 9);
    let mut plan = SparseExecutionPlan::authorized(f, W);
    plan.reductions = ReductionSet::STRUCTURAL; // M_log / C_R disabled
    let err = abacus_tensor::fanin_kernel::execute(&csf, &plan)
        .expect_err("partial reduction set must be rejected");
    assert!(matches!(err, TensorError::Range(_)), "got {err:?}");
    assert!(ReductionSet::AUTHORIZED_FUSED.is_full(), "authorized set is full");
    assert!(!ReductionSet::STRUCTURAL.is_full(), "structural-only is not full");
}

/// An empty window is a legitimate empty result, never an error (`L4`).
#[test]
fn empty_window_is_empty_not_error() {
    let csf = build(&[(10, 1, 9, 1_000_000)]);
    let f = FocalTransaction::new(10, 5, 9); // t_f < D, window precedes epoch
    let r = fused_fanin(&csf, f, W).expect("empty window is total");
    assert_eq!(r.t_i, 0);
    assert_eq!(r.m_log, 0);
    assert_eq!(r.c_r, 0);
    assert_eq!(r.n_i, 0);
    assert_eq!(r.rfi, None, "RFI undefined at T_i == 0");
    assert_eq!(r.t_min, None);
    assert_eq!(r.t_max, None);
}

