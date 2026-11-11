//! Support layer (#19, row 48): `(a_one, a_cnt, a_vol)` over admitted
//! entries (golden O.3).
//!
//! Ported from v7 `graphblas/support.rs` (itself from v5 `support`):
//! `A_cnt` via row 31, `A_one = support(A_cnt)` (row 38), `A_vol` via
//! checked contraction (row 30). Errors on volume overflow (`E_rng`).

use super::matrix::PairMatrix;
use super::types::TransactionRecord;
use crate::TensorError;

/// The three canonical support tensors over one admitted record set.
#[derive(Clone, Debug)]
pub struct SupportTensors {
    pub a_one: PairMatrix,
    pub a_cnt: PairMatrix,
    pub a_vol: PairMatrix,
}

/// Build `A_cnt`, `A_one`, `A_vol` from admitted canonical entries.
///
/// Optimization (semantics unchanged, rows 30/31/38/48): the input is in
/// `prec_3` `(t, s, r)` order, so one sort by `(s, r)` (carrying `t` for
/// stability) yields the canonical `(s, r)` order for all three outputs.
/// A single linear sweep then aggregates `A_cnt` counts (row 31) and
/// `A_vol` checked sums (row 30, P2 fail-closed), `A_one` is the linear
/// existence map of `A_cnt` (row 38), and all three wrap the sorted-unique
/// pairs with no re-sort and no per-pair `Vec::insert` memmoves. Replaces:
/// `temporal_multiplicity` (sort #1) + `support` (`insert` loop, quadratic
/// memmoves) + `from_occurrences` (sort #2 over `(s,r,value)` plus its own
/// ordered push — a third sort-equivalent pass over the same keys).
///
/// # Errors
/// `TensorError::Range` when an exact volume sum is unrepresentable (P2).
pub fn build_support(windowed: &[TransactionRecord]) -> Result<SupportTensors, TensorError> {
    use super::matrix::checked_add;
    // One (s, r)-major ordering pass; `t` carried only to keep the sort
    // total (input prec_3 order is a valid tiebreak).
    let mut ord: Vec<(u32, u32, u64, i64)> = Vec::with_capacity(windowed.len());
    for e in windowed {
        debug_assert!(e.value != 0, "canonical decode never yields zero values");
        ord.push((e.s, e.r, e.t, e.value));
    }
    ord.sort_unstable();
    let mut cnt_indices: Vec<(u32, u32)> = Vec::new();
    let mut cnts: Vec<i64> = Vec::new();
    let mut vol_indices: Vec<(u32, u32)> = Vec::new();
    let mut vols: Vec<i64> = Vec::new();
    if !ord.is_empty() {
        cnt_indices.reserve(ord.len());
        cnts.reserve(ord.len());
        vol_indices.reserve(ord.len());
        vols.reserve(ord.len());
    }
    let mut i = 0usize;
    while i < ord.len() {
        let (s, r, _, v0) = ord[i];
        let mut cnt: i64 = 1;
        let mut vol: i64 = v0;
        i += 1;
        while i < ord.len() && ord[i].0 == s && ord[i].1 == r {
            cnt += 1; // bounded by input length: count overflow impossible (P3 finite support)
            vol = checked_add(vol, ord[i].3)?;
            i += 1;
        }
        // Row 31: A_cnt keeps every observed pair (count >= 1); row 38: A_one
        // is its existence map. Row 30: A_vol drops zero sums (P3 support) —
        // exactly the `from_occurrences` zero-drop.
        cnt_indices.push((s, r));
        cnts.push(cnt);
        if vol != 0 {
            vol_indices.push((s, r));
            vols.push(vol);
        }
    }
    let a_cnt = PairMatrix::from_sorted_unique(cnt_indices, cnts);
    let a_one = a_cnt.support_pattern();
    let a_vol = PairMatrix::from_sorted_unique(vol_indices, vols);
    Ok(SupportTensors {
        a_one,
        a_cnt,
        a_vol,
    })
}
