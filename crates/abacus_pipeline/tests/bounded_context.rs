//! Phase-11 K=2 focal-context contract tests.
use abacus_pipeline::bounded_context::{build_bounded_k2, BoundedError, MAX_ADMITTED};
use abacus_pipeline::retrieval::EventLocator;
use abacus_pipeline::IngestPipeline;
use abacus_tensor::graphblas::context::{build_focal_context, ContextParameters};
use abacus_tensor::graphblas::types::{FeatureVector, FocalTransaction, HopBound, SelfLoopRule, TimeWindow, VertexId};
fn apply_edges(p: &mut IngestPipeline, edges: &[(u64, u64, u64)]) {
    use abacus_ingest::namespace::EventNamespace;
    use abacus_pipeline::{EventIdentity, FeaturePayload};
    use abacus_ingest::record::CanonicalRecord;
    for (i, (dt, s, r)) in edges.iter().enumerate() {
        let rec = CanonicalRecord {
            identity: EventIdentity { namespace: EventNamespace::from_bytes([1; 16]), event_id: i as u64 },
            event_time_ms: (*dt as i64) * 1000,
            sender_id: *s, receiver_id: *r,
            features: FeaturePayload { log_magnitude: 1, relative_threshold_distance: 0, roundedness_flags: 0 },
        };
        p.apply(&rec, Some(1_000), 10_000, 100).unwrap();
    }
}
fn live(p: &IngestPipeline, loc: EventLocator) -> Option<(u64, u32, u32, i64, FeatureVector)> {
    let (t, s, r, fp) = p.resolve(loc.0)?;
    let v = p.csf_live_at(t, s, r)?;
    let fv = FeatureVector {
        log_magnitude: fp.log_magnitude,
        relative_threshold_distance: fp.relative_threshold_distance,
        roundedness_flags: fp.roundedness_flags,
    };
    Some((t, s, r, v, fv))
}
#[test]
fn k2_matches_reference_on_chain() {
    let mut p = IngestPipeline::new();
    apply_edges(&mut p, &[(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 5, 6)]);
    let dir = p.temporal().resolve(EventLocator(2)).unwrap();
    let focal = FocalTransaction { epoch: dir.t, sender: VertexId(dir.sender), receiver: VertexId(dir.receiver) };
    let params = ContextParameters { window: TimeWindow { start: 0, end: 100 }, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let b = build_bounded_k2(p.temporal(), focal, &params, &|l| live(&p, l)).unwrap();
    p.seal_substrate().unwrap();
    let r = build_focal_context(p.csf_sealed(), focal, &params).unwrap();
    let mut ba: Vec<(u64, u32, u32)> = b.admitted.iter().map(|e| (e.t, e.s, e.r)).collect();
    let mut ra: Vec<(u64, u32, u32)> = r.admitted.iter().map(|e| (e.t, e.s, e.r)).collect();
    ba.sort(); ra.sort();
    assert_eq!(ba, ra);
    assert_eq!(b.vertices.indices(), r.vertices.indices());
}

#[test]
fn temporal_excludes_outside_and_future() {
    let mut p = IngestPipeline::new();
    apply_edges(&mut p, &[(100, 1, 2), (200, 2, 3), (300, 3, 4)]);
    let dir = p.temporal().resolve(EventLocator(1)).unwrap();
    let focal = FocalTransaction { epoch: dir.t, sender: VertexId(dir.sender), receiver: VertexId(dir.receiver) };
    let params = ContextParameters { window: TimeWindow { start: 200, end: 201 }, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let b = build_bounded_k2(p.temporal(), focal, &params, &|l| live(&p, l)).unwrap();
    assert!(b.admitted.iter().all(|e| e.t >= 200 && e.t < 201));
    assert!(b.admitted.iter().any(|e| e.t == 200));
}
#[test]
fn cycles_loops_dedup_empty() {
    let mut p = IngestPipeline::new();
    apply_edges(&mut p, &[(1, 1, 2), (2, 2, 1), (3, 2, 2), (4, 2, 3)]);
    let dir = p.temporal().resolve(EventLocator(0)).unwrap();
    let focal = FocalTransaction { epoch: dir.t, sender: VertexId(dir.sender), receiver: VertexId(dir.receiver) };
    let params = ContextParameters { window: TimeWindow { start: 0, end: 100 }, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let b = build_bounded_k2(p.temporal(), focal, &params, &|l| live(&p, l)).unwrap();
    assert!(!b.admitted.iter().any(|e| e.s == e.r));
    let mut k: Vec<(u64, u32, u32)> = b.admitted.iter().map(|e| (e.t, e.s, e.r)).collect();
    let n = k.len(); k.sort(); k.dedup(); assert_eq!(n, k.len());
    let mut q = IngestPipeline::new();
    apply_edges(&mut q, &[(50, 7, 8)]);
    let d2 = q.temporal().resolve(EventLocator(0)).unwrap();
    let f2 = FocalTransaction { epoch: d2.t, sender: VertexId(d2.sender), receiver: VertexId(d2.receiver) };
    let b2 = build_bounded_k2(q.temporal(), f2, &params, &|l| live(&q, l)).unwrap();
    assert_eq!(b2.admitted.len(), 1);
}
#[test]
fn overflow_explicit() {
    let mut p = IngestPipeline::new();
    let mut edges: Vec<(u64, u64, u64)> = Vec::new();
    for i in 0..(MAX_ADMITTED as u64 + 10) { edges.push((i + 1, 1, 1000 + i)); }
    apply_edges(&mut p, &edges);
    let dir = p.temporal().resolve(EventLocator(0)).unwrap();
    let focal = FocalTransaction { epoch: dir.t, sender: VertexId(dir.sender), receiver: VertexId(dir.receiver) };
    let params = ContextParameters { window: TimeWindow { start: 0, end: (MAX_ADMITTED as u64 + 100) }, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    match build_bounded_k2(p.temporal(), focal, &params, &|l| live(&p, l)) {
        Err(BoundedError::AdmittedOverflow { .. }) | Err(BoundedError::VertexOverflow { .. }) => {}
        Ok(_) => panic!("expected explicit overflow"),
        Err(e) => panic!("wrong class: {e}"),
    }
}
#[test]
fn both_roles_traversed() {
    let mut p = IngestPipeline::new();
    apply_edges(&mut p, &[(10, 1, 2), (11, 3, 1), (12, 2, 4)]);
    let dir = p.temporal().resolve(EventLocator(0)).unwrap();
    let focal = FocalTransaction { epoch: dir.t, sender: VertexId(dir.sender), receiver: VertexId(dir.receiver) };
    let params = ContextParameters { window: TimeWindow { start: 0, end: 100 }, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let b = build_bounded_k2(p.temporal(), focal, &params, &|l| live(&p, l)).unwrap();
    assert_eq!(b.vertices.nnz(), 4, "got {:?}", b.vertices.indices());
    assert_eq!(b.admitted.len(), 3);
}

#[test]
fn admitted_features_carry_verbatim_from_commit_time_derivation() {
    // Fixture: uniform FeaturePayload{log_magnitude:1, rel_threshold_distance:0, roundedness_flags:0}
    let mut p = IngestPipeline::new();
    apply_edges(&mut p, &[(1, 1, 2), (2, 2, 3), (3, 3, 4)]);
    let dir = p.temporal().resolve(EventLocator(1)).unwrap();
    let focal = FocalTransaction { epoch: dir.t, sender: VertexId(dir.sender), receiver: VertexId(dir.receiver) };
    let params = ContextParameters { window: TimeWindow { start: 0, end: 100 }, hop_bound: HopBound(2), eligible: None, self_loop: SelfLoopRule::Exclude };
    let b = build_bounded_k2(p.temporal(), focal, &params, &|l| live(&p, l)).unwrap();
    // Index-aligned: one feature vector per admitted record.
    assert_eq!(b.admitted.len(), b.admitted_features.len(), "features index-aligned with admitted");
    assert!(!b.admitted.is_empty(), "fixture must admit at least one record");
    // Verbatim carry-through: every context feature equals the commit-time
    // derivation that `apply` stored (features are derived from the attested
    // amount, not recomputed in the context). Ground truth = pipeline's own
    // `resolve()` over the full committed locator set.
    for (rec, fv) in b.admitted.iter().zip(b.admitted_features.iter()) {
        let mut expected = None;
        for loc in 0..p.committed() as u32 {
            let (t, s_, r_, fp) = p.resolve(loc).expect("committed locator resolves");
            if (t, s_, r_) == (rec.t, rec.s, rec.r) { expected = Some(fp); break; }
        }
        let fp = expected.expect("admitted coordinate has a commit-time feature");
        assert_eq!(fv.log_magnitude, fp.log_magnitude, "log_magnitude carried verbatim");
        assert_eq!(fv.relative_threshold_distance, fp.relative_threshold_distance, "rel_threshold carried verbatim");
        assert_eq!(fv.roundedness_flags, fp.roundedness_flags, "roundedness carried verbatim");
    }
}
