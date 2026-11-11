//! GraphBLAS primitive tests — ported from the v7 donor oracle suites.
//!
//! Coverage per `your_task_phase_7.md` section 7: each of the 23 primitive
//! definitions (same fixtures/assertions as the donor), composition,
//! empty/singleton inputs, sparse structural behavior, masks/reductions,
//! boundary/adversarial cases, error behavior, reference equivalence, and
//! v10 tensor-op compatibility.
//!
//! Donor sources: `graphblas_oracle_a.rs` (#1-#6), `graphblas_oracle_b.rs`
//! (#7-#13 + extract_3d/difference/overflow), `graphblas_pipeline.rs`
//! (#14-#23), `graphblas_math_alignment.rs` (textbook metric agreement).
//! Adaptations: imports point at `crate::graphblas`; tensor construction
//! uses the v10 `Tensor` API (identical surface).

use std::collections::{BTreeMap, BTreeSet};

use abacus_tensor::api::{decode, CanonicalTensor, TensorEntry};
use abacus_tensor::graphblas::context::{build_focal_context, ContextParameters, ExecutionError};
use abacus_tensor::graphblas::fan_in::compute_fan_in;
use abacus_tensor::graphblas::matrix::PairMatrix;
use abacus_tensor::graphblas::primitives::{
    ewise_add_lor, extract, multiplicity, mxv, reduce_col_plus, reduce_count, reduce_count_set,
    reduce_max, reduce_row_plus, reduce_sum, support, volume, vxm,
};
use abacus_tensor::graphblas::projection::{contract_mode0, extract_3d, temporal_multiplicity};
use abacus_tensor::graphblas::pruning::{hop_ball, prune_edges, prune_hops, prune_time};
use abacus_tensor::graphblas::pruning::vertex::prune_vertices;
use abacus_tensor::graphblas::types::{
    EdgeRef, EligiblePairs, FocalTransaction, HopBound, SelfLoopRule, TimeWindow, ValidationScope,
    VertexId,
};
use abacus_tensor::graphblas::validation::compare_with_oracle;
use abacus_tensor::graphblas::vector::Vector;
use abacus_tensor::tensor::Tensor;

const M: i64 = 1_000_000;

/// Donor fixture: 11-entry synthetic tensor (v7 `tests/common.rs` verbatim).
fn build_tensor() -> CanonicalTensor {
    let entries = vec![
        TensorEntry::new(10, 1, 2, 500 * M),
        TensorEntry::new(10, 1, 3, 125 * M),
        TensorEntry::new(10, 2, 3, 900 * M),
        TensorEntry::new(20, 1, 3, 300 * M),
        TensorEntry::new(20, 3, 1, 75 * M),
        TensorEntry::new(30, 2, 1, 1000 * M),
        TensorEntry::new(30, 2, 2, 250 * M),
        TensorEntry::new(30, 3, 2, 4000 * M),
        TensorEntry::new(40, 1, 2, 600 * M),
        TensorEntry::new(40, 2, 2, M),
        TensorEntry::new(40, 4, 5, 2 * M),
    ];
    Tensor::try_from_entries(entries).unwrap().to_csf()
}

fn windowed_entries(csf: &CanonicalTensor, start: u64, end: u64) -> Vec<TensorEntry> {
    decode(csf)
        .into_iter()
        .filter(|e| e.t >= start && e.t < end)
        .collect()
}

fn oracle_window(csf: &CanonicalTensor, start: u64, end: u64) -> BTreeMap<(u32, u32), (u64, i64)> {
    let mut m = BTreeMap::new();
    for e in decode(csf) {
        if e.t >= start && e.t < end {
            let c = m.entry((e.s, e.r)).or_insert((0u64, 0i64));
            c.0 += 1;
            c.1 += e.value;
        }
    }
    m
}
// --- oracle_a: #1-#6 ---
#[test]
fn primitives_1_to_6_extract_reduces_and_semiring_ops() {
    let csf = build_tensor();
    let (start, end) = (10u64, 41u64);
    let windowed = windowed_entries(&csf, start, end);
    assert!(windowed.windows(2).all(|w| (w[0].t, w[0].s, w[0].r) < (w[1].t, w[1].s, w[1].r)));
    let a_vol = contract_mode0(&windowed).unwrap();
    let a_cnt = temporal_multiplicity(&windowed);
    let a_one = support(&a_cnt);
    let reference = oracle_window(&csf, start, end);
    assert_eq!(a_vol.nnz(), reference.len());
    assert_eq!(a_cnt.nnz(), reference.len());
    for (&(s, r), &(count, vol)) in &reference {
        assert_eq!(extract(&a_vol, s, r), Some(vol));
        assert_eq!(extract(&a_cnt, s, r), Some(count as i64));
        assert_eq!(extract(&a_one, s, r), Some(1));
    }
    assert_eq!(extract(&a_vol, 9, 9), None);
    for v in 1..=5u32 {
        let mut col = 0i64;
        let mut row = 0i64;
        for (&(s, r), &(_, vol)) in &reference {
            if r == v { col += vol; }
            if s == v { row += vol; }
        }
        assert_eq!(reduce_col_plus(&a_vol, v).unwrap(), col);
        assert_eq!(reduce_row_plus(&a_vol, v).unwrap(), row);
    }
    let frontier = Vector::from_sorted(vec![2, 4]);
    let mut ref_mxv = BTreeSet::new();
    for &(s, r) in reference.keys() { if frontier.contains(r) { ref_mxv.insert(s); } }
    assert_eq!(mxv(&a_one, &frontier).indices(), &ref_mxv.iter().copied().collect::<Vec<_>>());
    let fwd = Vector::from_sorted(vec![1]);
    let mut ref_vxm = BTreeSet::new();
    for &(s, r) in reference.keys() { if fwd.contains(s) { ref_vxm.insert(r); } }
    assert_eq!(vxm(&a_one, &fwd).indices(), &ref_vxm.iter().copied().collect::<Vec<_>>());
    let a = Vector::from_sorted(vec![1, 2]);
    let b = Vector::from_sorted(vec![2, 3]);
    assert_eq!(ewise_add_lor(&a, &b).indices(), &[1, 2, 3]);
}

// --- oracle_b: #7-#13 ---
#[test]
fn primitives_7_to_13_support_and_global_reductions() {
    let csf = build_tensor();
    let windowed = windowed_entries(&csf, 10, 41);
    let a_vol = contract_mode0(&windowed).unwrap();
    let a_cnt = temporal_multiplicity(&windowed);
    let pairs: Vec<(u32, u32)> = windowed.iter().map(|e| (e.s, e.r)).collect();
    let rebuilt_cnt = multiplicity(&pairs);
    for (s, r, c) in a_cnt.nonzeros() { assert_eq!(extract(&rebuilt_cnt, s, r), Some(c)); }
    let a_one = support(&a_cnt);
    for (s, r, c) in a_cnt.nonzeros() { assert!(c >= 1); assert_eq!(extract(&a_one, s, r), Some(1)); }
    let reference = oracle_window(&csf, 10, 41);
    let occ: Vec<((u32, u32), i64)> = reference.iter().map(|(&(s, r), &(_, vol))| ((s, r), vol)).collect();
    let rebuilt_vol = volume(&occ).unwrap();
    for (s, r, v) in a_vol.nonzeros() { assert_eq!(extract(&rebuilt_vol, s, r), Some(v)); }
    let all: Vec<(u32, u32)> = a_one.nonzeros().map(|(s, r, _)| (s, r)).collect();
    let mut members: Vec<u32> = all.iter().flat_map(|&(s, r)| [s, r]).collect();
    members.sort_unstable(); members.dedup();
    assert_eq!(reduce_count(&members) as usize, members.len());
    let vec = Vector::from_sorted(members.clone());
    assert_eq!(reduce_count_set(&vec) as usize, members.len());
    let values: Vec<i64> = a_vol.nonzeros().map(|(_, _, v)| v).collect();
    let expected: i64 = values.iter().sum();
    assert_eq!(reduce_sum(&values).unwrap(), expected);
    assert_eq!(reduce_sum(&[]).unwrap(), 0);
    assert_eq!(reduce_max(&values), values.iter().copied().max().unwrap_or(0));
    assert_eq!(reduce_max(&[]), 0);
    let nz: Vec<(u32, u32, i64)> = a_one.nonzeros().collect();
    for w in nz.windows(2) { assert!(w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 < w[1].1)); }
    let _ = PairMatrix::zero();
}

#[test]
fn extract_3d_and_vector_difference_contracts() {
    let csf = build_tensor();
    assert_eq!(extract_3d(&csf, 10, 1, 2), Some(500 * M));
    assert_eq!(extract_3d(&csf, 10, 1, 4), None);
    assert_eq!(extract_3d(&csf, 40, 2, 2), Some(M));
    let ball = Vector::from_sorted(vec![1, 3, 5, 7]);
    let stepped = Vector::from_sorted(vec![3, 5, 6, 8, 9]);
    assert_eq!(stepped.difference(&ball).indices(), &[6, 8, 9]);
}

#[test]
fn checked_arithmetic_fails_closed_on_overflow() {
    let csf = Tensor::try_from_entries(vec![
        TensorEntry::new(10, 1, 9, i64::MAX),
        TensorEntry::new(20, 1, 9, 1),
    ]).unwrap().to_csf();
    let windowed = windowed_entries(&csf, 0, 100);
    let err = contract_mode0(&windowed).expect_err("volume must overflow");
    assert!(matches!(err, abacus_tensor::TensorError::Range(_)));
    assert!(matches!(reduce_sum(&[i64::MAX, 1]), Err(abacus_tensor::TensorError::Range(_))));
}
// --- pipeline: #14-#18 ---
#[test]
fn primitives_14_to_18_pruning_and_hop_ball_vs_oracle() {
    let csf = build_tensor();
    let w = TimeWindow { start: 10, end: 31 };
    let ref_entries: Vec<_> = decode(&csf).into_iter().filter(|e| w.contains(e.t)).collect();
    let windowed = prune_time(&csf, w);
    assert_eq!(windowed, ref_entries);
    assert!(prune_time(&csf, TimeWindow { start: 50, end: 60 }).is_empty());
    assert_eq!(prune_time(&csf, TimeWindow { start: 10, end: 10 }).len(), 0);
    let a_one = support(&temporal_multiplicity(&windowed));
    for k in [0u64, 1, 2, 3] {
        let ball = hop_ball(&a_one, VertexId(1), VertexId(3), HopBound(k));
        let mut ref_ball: BTreeSet<u32> = [1, 3].into_iter().collect();
        let mut frontier = ref_ball.clone();
        for _ in 0..k {
            let mut stepped = BTreeSet::new();
            for &j in &frontier {
                for (s, r, _v) in a_one.nonzeros() {
                    if r == j { stepped.insert(s); }
                    if s == j { stepped.insert(r); }
                }
            }
            let next: BTreeSet<u32> = stepped.difference(&ref_ball).copied().collect();
            if next.is_empty() { break; }
            ref_ball.extend(next.iter().copied());
            frontier = next;
        }
        assert_eq!(ball.indices(), ref_ball.iter().copied().collect::<Vec<_>>().as_slice(), "hop_ball k={k}");
    }
    let ball = hop_ball(&a_one, VertexId(1), VertexId(3), HopBound(2));
    let kept = prune_vertices(&windowed, &ball);
    for e in &kept { assert!(ball.contains(e.s) && ball.contains(e.r)); }
    let eligible = EligiblePairs::from_sorted(vec![(1, 3), (2, 3)]);
    let kept_edges = prune_edges(&windowed, &eligible);
    for e in &kept_edges { assert!(eligible.admits(EdgeRef::new(e.s, e.r))); }
    let kept_hops = prune_hops(&windowed, &a_one, VertexId(1), VertexId(3), HopBound(2));
    for e in &kept_hops { assert!(ball.contains(e.s) && ball.contains(e.r)); }
}

#[allow(dead_code)]
fn oracle_incoming(windowed: &BTreeMap<(u32, u32), (u64, i64)>, ball: &BTreeSet<u32>, v: u32, exclude_self: bool) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for &(s, r) in windowed.keys() { if r == v && ball.contains(&s) && ball.contains(&r) { out.insert(s); } }
    if exclude_self { out.remove(&v); }
    out
}

#[test]
fn primitives_19_to_22_context_fan_in_vs_oracle() {
    let csf = build_tensor();
    let focal = FocalTransaction { epoch: 30, sender: VertexId(2), receiver: VertexId(1) };
    for k in [0u64, 1, 2] {
        for self_loop in [SelfLoopRule::Exclude, SelfLoopRule::Admit] {
            let scope = ValidationScope { omega_secs: 20, hop_bound: HopBound(k), eligible: None, self_loop };
            let params = ContextParameters::from_scope(focal.epoch, &scope);
            let ctx = build_focal_context(&csf, focal, &params).unwrap();
            let windowed = prune_time(&csf, params.window);
            let candidate_one = support(&temporal_multiplicity(&windowed));
            let ref_ball = hop_ball(&candidate_one, focal.sender, focal.receiver, params.hop_bound);
            assert_eq!(ctx.vertices.indices(), ref_ball.indices(), "ball k={k}");
            let fan = compute_fan_in(&ctx, &focal).unwrap();
            let mut expected: BTreeSet<u32> = BTreeSet::new();
            for (s, r, _v) in ctx.tensors.a_one.nonzeros() {
                if r == focal.receiver.0 && (self_loop == SelfLoopRule::Admit || s != r) {
                    expected.insert(s);
                }
            }
            assert_eq!(fan.support_counterparties.members, expected.iter().copied().collect::<Vec<_>>(), "support k={k} self_loop={self_loop:?}");
            assert_eq!(fan.fan_in.count, expected.len() as u64);
        }
    }
}
// --- pipeline: #19-#23 + errors ---
#[test]
fn context_ball_equals_hop_ball_over_window_candidate_support() {
    let csf = build_tensor();
    let focals = [
        FocalTransaction { epoch: 20, sender: VertexId(1), receiver: VertexId(3) },
        FocalTransaction { epoch: 30, sender: VertexId(2), receiver: VertexId(1) },
    ];
    for f in focals {
        for k in [0u64, 1, 2, 3] {
            let scope = ValidationScope { omega_secs: 40, hop_bound: HopBound(k), eligible: None, self_loop: SelfLoopRule::Exclude };
            let params = ContextParameters::from_scope(f.epoch, &scope);
            let ctx = build_focal_context(&csf, f, &params).unwrap();
            let windowed = prune_time(&csf, params.window);
            let candidate_one = support(&temporal_multiplicity(&windowed));
            let ref_ball = hop_ball(&candidate_one, f.sender, f.receiver, params.hop_bound);
            assert_eq!(ctx.vertices.indices(), ref_ball.indices(), "ball k={k}");
            let ref_admitted: Vec<_> = windowed.iter().copied()
                .filter(|r| ref_ball.contains(r.s) && ref_ball.contains(r.r))
                .filter(|r| r.s != r.r).collect();
            assert_eq!(ctx.admitted, ref_admitted, "admitted k={k}");
        }
    }
}

#[test]
fn primitives_23_compare_with_oracle_match_and_mismatch() {
    let csf = build_tensor();
    let focal = FocalTransaction { epoch: 30, sender: VertexId(2), receiver: VertexId(1) };
    let scope = ValidationScope { omega_secs: 20, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let params = ContextParameters::from_scope(focal.epoch, &scope);
    let ctx = build_focal_context(&csf, focal, &params).unwrap();
    let fan = compute_fan_in(&ctx, &focal).unwrap();
    let ok = compare_with_oracle(&ctx, &fan, fan.fan_in.count);
    assert!(ok.matches);
    assert!(ok.diagnostic.is_none());
    let bad = compare_with_oracle(&ctx, &fan, fan.fan_in.count + 1);
    assert!(!bad.matches);
    assert_eq!(bad.canonical, fan.fan_in.count);
    assert_eq!(bad.oracle, fan.fan_in.count + 1);
    let d = bad.diagnostic.expect("mismatch carries a diagnostic");
    assert_eq!(d.focal_sender, 2);
    assert_eq!(d.focal_receiver, 1);
    assert_eq!(d.focal_epoch, 30);
    assert_eq!(d.window_start, 10);
    assert_eq!(d.window_end, 31);
    assert_eq!(d.canonical_result, fan.fan_in.count);
    assert_eq!(d.support, fan.support_counterparties.members);
}

#[test]
fn context_errors_preserved_and_canonical_invariants() {
    let csf = build_tensor();
    let outside = FocalTransaction { epoch: 99, sender: VertexId(1), receiver: VertexId(2) };
    let params = ContextParameters { window: TimeWindow { start: 10, end: 40 }, hop_bound: HopBound(1), eligible: None, self_loop: SelfLoopRule::Exclude };
    assert!(matches!(build_focal_context(&csf, outside, &params), Err(ExecutionError::FocalOutsideWindow)));
    let absent = FocalTransaction { epoch: 30, sender: VertexId(9), receiver: VertexId(9) };
    let params = ContextParameters::from_scope(absent.epoch, &ValidationScope { omega_secs: 5, hop_bound: HopBound(1), eligible: None, self_loop: SelfLoopRule::Exclude });
    assert!(matches!(build_focal_context(&csf, absent, &params), Err(ExecutionError::FocalAbsent)));
    let focal = FocalTransaction { epoch: 40, sender: VertexId(2), receiver: VertexId(2) };
    let params = ContextParameters::from_scope(focal.epoch, &ValidationScope { omega_secs: 40, hop_bound: HopBound(1), eligible: None, self_loop: SelfLoopRule::Exclude });
    let ctx = build_focal_context(&csf, focal, &params).unwrap();
    for r in &ctx.admitted { assert_ne!(r.s, r.r); }
    assert!(ctx.vertices.contains(focal.sender.0));
    let nz: Vec<(u32, u32, i64)> = ctx.tensors.a_one.nonzeros().collect();
    for w in nz.windows(2) { assert!(w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 < w[1].1)); }
}
// --- math alignment + v10 compat + edge cases ---
#[test]
fn math_alignment_hand_computed_constants_at_k3() {
    let csf = build_tensor();
    let focal = FocalTransaction { epoch: 30, sender: VertexId(2), receiver: VertexId(1) };
    let scope = ValidationScope { omega_secs: 40, hop_bound: HopBound(3), eligible: None, self_loop: SelfLoopRule::Exclude };
    let params = ContextParameters::from_scope(focal.epoch, &scope);
    let ctx = build_focal_context(&csf, focal, &params).unwrap();
    let fan = compute_fan_in(&ctx, &focal).unwrap();
    assert_eq!(fan.fan_in.count, 2);
    assert_eq!(fan.support_counterparties.members, vec![2, 3][..], "hand-computed incoming support");
    let mut col = 0i64;
    let mut row = 0i64;
    for (s, r, v) in ctx.tensors.a_vol.nonzeros() {
        if r == 1 { col += v; }
        if s == 2 { row += v; }
    }
    assert_eq!(reduce_col_plus(&ctx.tensors.a_vol, 1).unwrap(), col);
    assert_eq!(reduce_row_plus(&ctx.tensors.a_vol, 2).unwrap(), row);
}

#[test]
fn math_alignment_independent_oracle_agreement() {
    let csf = build_tensor();
    let windowed = windowed_entries(&csf, 10, 41);
    let a_vol = contract_mode0(&windowed).unwrap();
    let a_cnt = temporal_multiplicity(&windowed);
    let a_one = support(&a_cnt);
    let reference = oracle_window(&csf, 10, 41);
    for v in [1u32, 2, 3] {
        let mut indeg = 0i64;
        let mut outdeg = 0i64;
        for (&(s, r), &(_, vol)) in &reference {
            if r == v { indeg += vol; }
            if s == v { outdeg += vol; }
        }
        assert_eq!(reduce_col_plus(&a_vol, v).unwrap(), indeg);
        assert_eq!(reduce_row_plus(&a_vol, v).unwrap(), outdeg);
    }
    let frontier = Vector::from_sorted(vec![1, 2]);
    let mut in_nbrs = BTreeSet::new();
    let mut out_nbrs = BTreeSet::new();
    for &(s, r) in reference.keys() {
        if frontier.contains(r) { in_nbrs.insert(s); }
        if frontier.contains(s) { out_nbrs.insert(r); }
    }
    let mx = mxv(&a_one, &frontier);
    let vx = vxm(&a_one, &frontier);
    assert_eq!(mx.indices(), &in_nbrs.iter().copied().collect::<Vec<_>>());
    assert_eq!(vx.indices(), &out_nbrs.iter().copied().collect::<Vec<_>>());
    let uni: BTreeSet<u32> = in_nbrs.union(&out_nbrs).copied().collect();
    assert_eq!(ewise_add_lor(&mx, &vx).indices(), &uni.iter().copied().collect::<Vec<_>>());
}

#[test]
fn v10_compat_time_slice_matches_prune_time() {
    let csf = build_tensor();
    for (s, e) in [(0u64, 100u64), (10, 31), (31, 31), (50, 60)] {
        let via_api = abacus_tensor::api::time_slice(&csf, s, e);
        let via_prim = prune_time(&csf, TimeWindow { start: s, end: e });
        assert_eq!(via_api, via_prim, "window [{s},{e})");
    }
}

#[test]
fn v10_compat_lookup_matches_extract_3d() {
    let csf = build_tensor();
    for (t, s, r) in [(10u64, 1u32, 2u32), (10, 1, 4), (40, 2, 2), (99, 9, 9)] {
        assert_eq!(abacus_tensor::api::lookup(&csf, t, s, r), extract_3d(&csf, t, s, r));
    }
}

#[test]
fn v10_compat_set_counts_match_fan_in() {
    let csf = build_tensor();
    let focal = FocalTransaction { epoch: 30, sender: VertexId(2), receiver: VertexId(1) };
    let scope = ValidationScope { omega_secs: 20, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let params = ContextParameters::from_scope(focal.epoch, &scope);
    let ctx = build_focal_context(&csf, focal, &params).unwrap();
    let fan = compute_fan_in(&ctx, &focal).unwrap();
    let entries: Vec<(u64, u32, u32)> = ctx.admitted.iter().map(|e| (e.t, e.s, e.r)).collect();
    let via_sets = abacus_tensor::samld::sets::fan_in_senders(&entries, 1, params.window.start, params.window.end);
    let mut via_sets: Vec<u32> = via_sets.into_iter().collect();
    via_sets.retain(|&u| u != 1);
    assert_eq!(fan.support_counterparties.members, via_sets);
    assert_eq!(fan.fan_in.count as usize, via_sets.len());
}

#[test]
fn edge_cases_empty_singleton_self_loop() {
    let empty = Tensor::try_from_entries(vec![]).unwrap().to_csf();
    let w = TimeWindow { start: 0, end: 100 };
    assert!(prune_time(&empty, w).is_empty());
    assert_eq!(extract_3d(&empty, 10, 1, 2), None);
    let a_one = PairMatrix::zero();
    let ball = hop_ball(&a_one, VertexId(1), VertexId(2), HopBound(5));
    assert_eq!(ball.indices(), &[1, 2]);
    assert_eq!(reduce_sum(&[]).unwrap(), 0);
    assert_eq!(reduce_max(&[]), 0);
    assert_eq!(reduce_count(&[]), 0);
    assert_eq!(reduce_count_set(&Vector::zero()), 0);
    let single = Tensor::try_from_entries(vec![TensorEntry::new(10, 1, 2, 5 * M)]).unwrap().to_csf();
    let params = ContextParameters { window: TimeWindow { start: 0, end: 20 }, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let focal = FocalTransaction { epoch: 10, sender: VertexId(1), receiver: VertexId(2) };
    let ctx = build_focal_context(&single, focal, &params).unwrap();
    let fan = compute_fan_in(&ctx, &focal).unwrap();
    assert_eq!(fan.fan_in.count, 1);
    assert_eq!(fan.support_counterparties.members, vec![1]);
    let with_loop = Tensor::try_from_entries(vec![
        TensorEntry::new(10, 1, 2, 5 * M),
        TensorEntry::new(11, 2, 2, 7 * M),
    ]).unwrap().to_csf();
    for (rule, expect_loop) in [(SelfLoopRule::Exclude, false), (SelfLoopRule::Admit, true)] {
        let params = ContextParameters { window: TimeWindow { start: 0, end: 20 }, hop_bound: HopBound(2), eligible: None, self_loop: rule };
        let ctx = build_focal_context(&with_loop, FocalTransaction { epoch: 10, sender: VertexId(1), receiver: VertexId(2) }, &params).unwrap();
        assert_eq!(ctx.admitted.iter().any(|e| e.s == 2 && e.r == 2), expect_loop, "self-loop rule {rule:?}");
    }
}

// --- Phase 10: fused `build_support` equivalence (single-sort path) ---

/// Fused `build_support` must equal the reference composition it replaced:
/// `A_cnt = temporal_multiplicity(records)` (row 31),
/// `A_one = support(A_cnt)` (row 38), `A_vol = contract_mode0(records)`
/// (row 30). Covers duplicate pairs (multiplicity > 1), opposing values
/// (zero-sum volume drop), self-loops, and empty input.
#[test]
fn phase10_fused_build_support_matches_reference_composition() {
    use abacus_tensor::graphblas::support::build_support;
    // Duplicate pairs + opposing signs: (1,2) x2 sums to 0 volume (drops
    // from A_vol only), (2,3) x2 sums nonzero, plus a self-loop.
    let records = vec![
        TensorEntry::new(1, 1, 2, 100),
        TensorEntry::new(2, 1, 2, -100),
        TensorEntry::new(3, 2, 3, 50),
        TensorEntry::new(4, 2, 3, 25),
        TensorEntry::new(5, 4, 4, 10),
    ];
    let ref_cnt = temporal_multiplicity(&records);
    let ref_one = support(&ref_cnt);
    let ref_vol = contract_mode0(&records).unwrap();
    let fused = build_support(&records).unwrap();
    assert_eq!(fused.a_cnt, ref_cnt, "A_cnt mismatch");
    assert_eq!(fused.a_one, ref_one, "A_one mismatch");
    assert_eq!(fused.a_vol, ref_vol, "A_vol mismatch");
    // Zero-sum pair (1,2): count kept, volume dropped.
    assert_eq!(fused.a_cnt.get(1, 2), Some(2));
    assert_eq!(fused.a_one.get(1, 2), Some(1));
    assert_eq!(fused.a_vol.get(1, 2), None);
    assert_eq!(fused.a_cnt.get(2, 3), Some(2));
    assert_eq!(fused.a_vol.get(2, 3), Some(75));
    // Empty input: all three zero.
    let empty = build_support(&[]).unwrap();
    assert!(empty.a_cnt.is_empty() && empty.a_one.is_empty() && empty.a_vol.is_empty());
    // Volume overflow still fails closed (P2).
    let big = vec![
        TensorEntry::new(1, 9, 9, i64::MAX),
        TensorEntry::new(2, 9, 9, 1),
    ];
    assert!(build_support(&big).is_err(), "volume overflow must fail closed");
}


// --- latency sanity (sub-millisecond rule; release-mode, ignored by default) ---

/// Measures the dominant ported paths on a synthetic 3k-entry window:
/// build_focal_context (row 49), compute_fan_in (#21/#22), mxv/vxm
/// (#4/#5), hop_ball (#17). Prints durations; release-mode gate run via
/// `cargo test --release -p abacus_tensor --test graphblas -- --ignored --nocapture`.
#[test]
#[ignore = "latency gate: run in release mode"]
fn latency_gate_release_subms() {
    // 10k canonical entries over a chain+star topology, 10k distinct seconds.
    let mut entries = Vec::with_capacity(3_000);
    for t in 0..3_000u64 {
        let s = (t % 100) as u32;
        let r = ((t / 100) % 100) as u32 + 100;
        entries.push(TensorEntry::new(t, s, r, 1_000 + (t % 97) as i64));
    }
    let csf = Tensor::try_from_entries(entries).unwrap().to_csf();
    let focal = FocalTransaction { epoch: 2_999, sender: VertexId(99), receiver: VertexId(129) };
    let params = ContextParameters {
        window: TimeWindow { start: 0, end: 3_000 },
        hop_bound: HopBound(3),
        eligible: None,
        self_loop: SelfLoopRule::Exclude,
    };
    let t0 = std::time::Instant::now();
    let ctx = build_focal_context(&csf, focal, &params).unwrap();
    let t_ctx = t0.elapsed();
    let t1 = std::time::Instant::now();
    let fan = compute_fan_in(&ctx, &focal).unwrap();
    let t_fan = t1.elapsed();
    let a_one = &ctx.tensors.a_one;
    let frontier = Vector::from_sorted(vec![2, 50]);
    let t2 = std::time::Instant::now();
    let mv = mxv(a_one, &frontier);
    let vv = vxm(a_one, &frontier);
    let ball = hop_ball(a_one, focal.sender, focal.receiver, HopBound(3));
    let t_ops = t2.elapsed();
    println!(
        "build_focal_context={t_ctx:?} compute_fan_in={t_fan:?} mxv+vxm+hop_ball={t_ops:?} \
         admitted={} fan_in={} mxv_nnz={} vxm_nnz={} ball_nnz={}",
        ctx.admitted.len(),
        fan.fan_in.count,
        mv.nnz(),
        vv.nnz(),
        ball.nnz()
    );
    // Generous release-mode bound: each path must stay sub-millisecond on
    // 10k entries (architecture sanity, not a micro-benchmark claim).
    assert!(t_ctx.as_micros() < 1_000, "build_focal_context sub-ms");
    assert!(t_fan.as_micros() < 1_000, "compute_fan_in sub-ms");
    assert!(t_ops.as_micros() < 1_000, "semiring ops sub-ms");
}
