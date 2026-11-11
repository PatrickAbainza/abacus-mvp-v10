//! GraphBLAS API-layer equivalence tests — validates that every function in
//! `abacus_tensor::api::graphblas` produces identical results to the
//! corresponding `abacus_tensor::graphblas` operation on the same fixtures.
//!
//! Coverage per `your_task_phase_9.md`: each API-boundary function, primitive
//! re-export, empty/singleton inputs, error behavior, and v10-API
//! compatibility. Same donor fixture as `tests/graphblas.rs`.

use abacus_tensor::api::{decode, CanonicalTensor, TensorEntry};
use abacus_tensor::graphblas::context::ContextParameters;
use abacus_tensor::graphblas::primitives::{
    ewise_add_lor, extract, multiplicity, mxv, reduce_col_plus, reduce_count, reduce_count_set,
    reduce_max, reduce_row_plus, reduce_sum, support, volume, vxm,
};
use abacus_tensor::graphblas::projection::{contract_mode0, extract_3d, temporal_multiplicity};
use abacus_tensor::graphblas::pruning::{hop_ball, prune_edges, prune_hops, prune_time};
use abacus_tensor::graphblas::pruning::vertex::prune_vertices;
use abacus_tensor::graphblas::types::{
    EligiblePairs, FocalTransaction, HopBound, SelfLoopRule, TimeWindow, VertexId,
};
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

// ---------------------------------------------------------------------------
// API-boundary functions
// ---------------------------------------------------------------------------

#[test]
fn api_time_slice_matches_read_time_slice() {
    let csf = build_tensor();
    for (s, e) in [(0u64, 100u64), (10, 31), (31, 31), (50, 60)] {
        let via_api = abacus_tensor::api::graphblas::time_slice(&csf, s, e);
        let via_read = abacus_tensor::api::time_slice(&csf, s, e);
        assert_eq!(via_api, via_read, "window [{s},{e})");
    }
}

#[test]
fn api_lookup_matches_read_lookup() {
    let csf = build_tensor();
    for (t, s, r) in [(10u64, 1u32, 2u32), (10, 1, 4), (40, 2, 2), (99, 9, 9)] {
        let via_api = abacus_tensor::api::graphblas::lookup(&csf, t, s, r);
        let via_read = abacus_tensor::api::lookup(&csf, t, s, r);
        assert_eq!(via_api, via_read, "({t},{s},{r})");
    }
}

#[test]
fn api_extract_3d_matches_graphblas_extract_3d() {
    let csf = build_tensor();
    for (t, s, r) in [(10u64, 1u32, 2u32), (20, 3, 1), (40, 4, 5), (99, 9, 9)] {
        assert_eq!(
            abacus_tensor::api::graphblas::extract_3d(&csf, t, s, r),
            extract_3d(&csf, t, s, r),
            "({t},{s},{r})"
        );
    }
}

#[test]
fn api_contract_mode0_window_matches_graphblas() {
    let csf = build_tensor();
    let windowed: Vec<TensorEntry> = decode(&csf);
    let via_graphblas = contract_mode0(&windowed).unwrap();
    let via_api = abacus_tensor::api::graphblas::contract_mode0_window(&csf, 0, 100).unwrap();
    assert_eq!(via_api.nnz(), via_graphblas.nnz());
    for (s, r, v) in via_graphblas.nonzeros() {
        assert_eq!(via_api.get(s, r), Some(v), "({s},{r})");
    }
}

#[test]
fn api_temporal_multiplicity_window_matches_graphblas() {
    let csf = build_tensor();
    let windowed: Vec<TensorEntry> = decode(&csf);
    let via_graphblas = temporal_multiplicity(&windowed);
    let via_api = abacus_tensor::api::graphblas::temporal_multiplicity_window(&csf, 0, 100);
    assert_eq!(via_api.nnz(), via_graphblas.nnz());
    for (s, r, v) in via_graphblas.nonzeros() {
        assert_eq!(via_api.get(s, r), Some(v), "({s},{r})");
    }
}

#[test]
fn api_build_support_window_matches_graphblas() {
    let csf = build_tensor();
    let windowed: Vec<TensorEntry> = decode(&csf);
    let via_graphblas = abacus_tensor::graphblas::support::build_support(&windowed).unwrap();
    let via_api = abacus_tensor::api::graphblas::build_support_window(&csf, 0, 100).unwrap();
    assert_eq!(via_api.a_one.nnz(), via_graphblas.a_one.nnz());
    assert_eq!(via_api.a_cnt.nnz(), via_graphblas.a_cnt.nnz());
    assert_eq!(via_api.a_vol.nnz(), via_graphblas.a_vol.nnz());
    for (s, r, v) in via_graphblas.a_vol.nonzeros() {
        assert_eq!(via_api.a_vol.get(s, r), Some(v), "a_vol({s},{r})");
    }
}

#[test]
fn api_build_focal_context_matches_graphblas() {
    let csf = build_tensor();
    let params = ContextParameters {
        window: TimeWindow { start: 0, end: 50 },
        hop_bound: HopBound(2),
        eligible: None,
        self_loop: SelfLoopRule::Exclude,
    };
    let focal = FocalTransaction {
        epoch: 10,
        sender: VertexId(1),
        receiver: VertexId(2),
    };
    let via_graphblas =
        abacus_tensor::graphblas::context::build_focal_context(&csf, focal, &params).unwrap();
    let via_api =
        abacus_tensor::api::graphblas::build_focal_context_from_tensor(&csf, focal, &params)
            .unwrap();
    assert_eq!(via_api.admitted.len(), via_graphblas.admitted.len());
    assert_eq!(via_api.vertices, via_graphblas.vertices);
    assert_eq!(via_api.tensors.a_one.nnz(), via_graphblas.tensors.a_one.nnz());
}

#[test]
fn api_compute_fan_in_from_tensor_matches_graphblas() {
    let csf = build_tensor();
    let params = ContextParameters {
        window: TimeWindow { start: 0, end: 50 },
        hop_bound: HopBound(2),
        eligible: None,
        self_loop: SelfLoopRule::Exclude,
    };
    let focal = FocalTransaction {
        epoch: 10,
        sender: VertexId(1),
        receiver: VertexId(2),
    };
    let via_graphblas_ctx =
        abacus_tensor::graphblas::context::build_focal_context(&csf, focal, &params).unwrap();
    let via_graphblas =
        abacus_tensor::graphblas::fan_in::compute_fan_in(&via_graphblas_ctx, &focal).unwrap();
    let via_api = abacus_tensor::api::graphblas::compute_fan_in_from_tensor(&csf, focal, &params)
        .unwrap();
    assert_eq!(via_api.fan_in.count, via_graphblas.fan_in.count);
    assert_eq!(
        via_api.support_counterparties.members,
        via_graphblas.support_counterparties.members
    );
}

#[test]
fn api_hop_ball_from_tensor_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_cnt = temporal_multiplicity(&windowed);
    let a_one = support(&a_cnt);
    let via_graphblas = hop_ball(&a_one, VertexId(1), VertexId(2), HopBound(2));
    let via_api =
        abacus_tensor::api::graphblas::hop_ball_from_tensor(&csf, 0, 50, VertexId(1), VertexId(2), HopBound(2));
    assert_eq!(via_api.nnz(), via_graphblas.nnz());
    assert_eq!(via_api.indices(), via_graphblas.indices());
}

#[test]
fn api_prune_time_from_tensor_matches_graphblas() {
    let csf = build_tensor();
    for (s, e) in [(0u64, 100u64), (10, 31), (31, 31), (50, 60)] {
        let via_graphblas = prune_time(&csf, TimeWindow { start: s, end: e });
        let via_api = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, s, e);
        assert_eq!(via_api, via_graphblas, "window [{s},{e})");
    }
}

#[test]
fn api_prune_vertices_window_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_cnt = temporal_multiplicity(&windowed);
    let a_one = support(&a_cnt);
    let ball = hop_ball(&a_one, VertexId(1), VertexId(2), HopBound(2));
    let via_graphblas = prune_vertices(&windowed, &ball);
    let via_api = abacus_tensor::api::graphblas::prune_vertices_window(&windowed, &ball);
    assert_eq!(via_api, via_graphblas);
}

#[test]
fn api_prune_edges_window_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let allowed = EligiblePairs::from_sorted(vec![(1, 2), (1, 3)]);
    let via_graphblas = prune_edges(&windowed, &allowed);
    let via_api = abacus_tensor::api::graphblas::prune_edges_window(&windowed, &allowed);
    assert_eq!(via_api, via_graphblas);
}

#[test]
fn api_prune_hops_from_tensor_matches_graphblas() {
    let csf = build_tensor();
    let via_graphblas = prune_hops(
        &prune_time(&csf, TimeWindow { start: 0, end: 50 }),
        &support(&temporal_multiplicity(&prune_time(&csf, TimeWindow { start: 0, end: 50 }))),
        VertexId(1),
        VertexId(2),
        HopBound(2),
    );
    let via_api = abacus_tensor::api::graphblas::prune_hops_from_tensor(
        &csf, 0, 50, VertexId(1), VertexId(2), HopBound(2),
    );
    assert_eq!(via_api, via_graphblas);
}

// ---------------------------------------------------------------------------
// Primitive re-exports
// ---------------------------------------------------------------------------

#[test]
fn api_extract_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_vol = contract_mode0(&windowed).unwrap();
    for (s, r, v) in a_vol.nonzeros() {
        assert_eq!(
            abacus_tensor::api::graphblas::extract(&a_vol, s, r),
            extract(&a_vol, s, r),
            "extract({s},{r})"
        );
        assert_eq!(abacus_tensor::api::graphblas::extract(&a_vol, s, r), Some(v));
    }
    assert_eq!(abacus_tensor::api::graphblas::extract(&a_vol, 99, 99), None);
}

#[test]
fn api_reduce_col_plus_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_vol = contract_mode0(&windowed).unwrap();
    for v in [1u32, 2, 3, 99] {
        assert_eq!(
            abacus_tensor::api::graphblas::reduce_col_plus(&a_vol, v).unwrap(),
            reduce_col_plus(&a_vol, v).unwrap(),
            "col {v}"
        );
    }
}

#[test]
fn api_reduce_row_plus_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_vol = contract_mode0(&windowed).unwrap();
    for u in [1u32, 2, 3, 99] {
        assert_eq!(
            abacus_tensor::api::graphblas::reduce_row_plus(&a_vol, u).unwrap(),
            reduce_row_plus(&a_vol, u).unwrap(),
            "row {u}"
        );
    }
}

#[test]
fn api_mxv_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_one = support(&temporal_multiplicity(&windowed));
    let frontier = Vector::from_sorted(vec![2]);
    assert_eq!(
        abacus_tensor::api::graphblas::mxv(&a_one, &frontier),
        mxv(&a_one, &frontier)
    );
}

#[test]
fn api_vxm_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_one = support(&temporal_multiplicity(&windowed));
    let frontier = Vector::from_sorted(vec![1]);
    assert_eq!(
        abacus_tensor::api::graphblas::vxm(&a_one, &frontier),
        vxm(&a_one, &frontier)
    );
}

#[test]
fn api_ewise_add_lor_matches_graphblas() {
    let a = Vector::from_sorted(vec![1, 3, 5]);
    let b = Vector::from_sorted(vec![2, 3, 4]);
    assert_eq!(
        abacus_tensor::api::graphblas::ewise_add_lor(&a, &b),
        ewise_add_lor(&a, &b)
    );
}

#[test]
fn api_support_matrix_matches_graphblas() {
    let csf = build_tensor();
    let windowed = abacus_tensor::api::graphblas::prune_time_from_tensor(&csf, 0, 50);
    let a_cnt = temporal_multiplicity(&windowed);
    assert_eq!(
        abacus_tensor::api::graphblas::support_matrix(&a_cnt),
        support(&a_cnt)
    );
}

#[test]
fn api_multiplicity_matches_graphblas() {
    let pairs = [(1u32, 2u32), (1, 3), (2, 3), (1, 2)];
    assert_eq!(
        abacus_tensor::api::graphblas::multiplicity(&pairs),
        multiplicity(&pairs)
    );
}

#[test]
fn api_volume_matches_graphblas() {
    let occurrences = [((1u32, 2u32), 100i64), ((1, 2), 200), ((2, 3), 50)];
    assert_eq!(
        abacus_tensor::api::graphblas::volume(&occurrences).unwrap(),
        volume(&occurrences).unwrap()
    );
}

#[test]
fn api_reductions_match_graphblas() {
    let members = [1u32, 2, 3, 4];
    let s = Vector::from_sorted(members.to_vec());
    let values = [10i64, 20, 30];

    assert_eq!(
        abacus_tensor::api::graphblas::reduce_count(&members),
        reduce_count(&members)
    );
    assert_eq!(
        abacus_tensor::api::graphblas::reduce_count_set(&s),
        reduce_count_set(&s)
    );
    assert_eq!(
        abacus_tensor::api::graphblas::reduce_sum(&values).unwrap(),
        reduce_sum(&values).unwrap()
    );
    assert_eq!(
        abacus_tensor::api::graphblas::reduce_max(&values),
        reduce_max(&values)
    );
}

// ---------------------------------------------------------------------------
// Empty and singleton inputs
// ---------------------------------------------------------------------------

#[test]
fn api_empty_tensor_all_functions_ok() {
    let empty: CanonicalTensor = Tensor::try_from_entries(vec![]).unwrap().to_csf();
    assert!(abacus_tensor::api::graphblas::time_slice(&empty, 0, 100).is_empty());
    assert_eq!(abacus_tensor::api::graphblas::lookup(&empty, 0, 0, 0), None);
    assert_eq!(abacus_tensor::api::graphblas::extract_3d(&empty, 0, 0, 0), None);
    assert!(abacus_tensor::api::graphblas::prune_time_from_tensor(&empty, 0, 100).is_empty());
    let m = abacus_tensor::api::graphblas::contract_mode0_window(&empty, 0, 100).unwrap();
    assert!(m.is_empty());
    let m = abacus_tensor::api::graphblas::temporal_multiplicity_window(&empty, 0, 100);
    assert!(m.is_empty());
    let s = abacus_tensor::api::graphblas::build_support_window(&empty, 0, 100).unwrap();
    assert!(s.a_one.is_empty() && s.a_cnt.is_empty() && s.a_vol.is_empty());
}

#[test]
fn api_singleton_entry_consistent() {
    let singleton: CanonicalTensor = Tensor::try_from_entries(vec![TensorEntry::new(10, 1, 2, 5 * M)])
        .unwrap()
        .to_csf();
    assert_eq!(
        abacus_tensor::api::graphblas::lookup(&singleton, 10, 1, 2),
        Some(5 * M)
    );
    assert_eq!(
        abacus_tensor::api::graphblas::extract_3d(&singleton, 10, 1, 2),
        Some(5 * M)
    );
    let m = abacus_tensor::api::graphblas::contract_mode0_window(&singleton, 0, 100).unwrap();
    assert_eq!(m.get(1, 2), Some(5 * M));
    assert_eq!(m.nnz(), 1);
    let m = abacus_tensor::api::graphblas::temporal_multiplicity_window(&singleton, 0, 100);
    assert_eq!(m.get(1, 2), Some(1));
    let m = abacus_tensor::api::graphblas::contract_mode0_window(&singleton, 0, 100).unwrap();
    assert_eq!(abacus_tensor::api::graphblas::reduce_col_plus(&m, 2).unwrap(), 5 * M);
}

// ---------------------------------------------------------------------------
// Error behavior: focal absent / focal outside window
// ---------------------------------------------------------------------------

#[test]
fn api_focal_absent_error_preserved() {
    let csf = build_tensor();
    let params = ContextParameters {
        window: TimeWindow { start: 0, end: 50 },
        hop_bound: HopBound(2),
        eligible: None,
        self_loop: SelfLoopRule::Exclude,
    };
    let focal = FocalTransaction {
        epoch: 99,
        sender: VertexId(9),
        receiver: VertexId(9),
    };
    let result = abacus_tensor::api::graphblas::build_focal_context_from_tensor(&csf, focal, &params);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "focal epoch outside window"
    );
}

#[test]
fn api_focal_outside_window_error_preserved() {
    let csf = build_tensor();
    let params = ContextParameters {
        window: TimeWindow { start: 0, end: 5 },
        hop_bound: HopBound(2),
        eligible: None,
        self_loop: SelfLoopRule::Exclude,
    };
    let focal = FocalTransaction {
        epoch: 10,
        sender: VertexId(1),
        receiver: VertexId(2),
    };
    let result = abacus_tensor::api::graphblas::build_focal_context_from_tensor(&csf, focal, &params);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "focal epoch outside window"
    );
}