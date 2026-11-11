//! Phase-11 bounded focal context (K=2, endpoint-driven, L1-bounded).
//!
//! Golden Source grounding (no invented tensor behavior):
//! - Addendum O rows 46/49 parameterize hop bound `k`; K=2 is the
//!   owner-ordered caller parameter (Amendment A1), not a derivation change.
//! - HCSF 5-7: ball via 2 endpoint-driven expansions over EXISTING
//!   `TemporalIndexes` (CSR forward / CSC backward + partition_point).
//! - HCSF 2: NO persistent state here; scratch is per-call transient.
//! - HCSF 9: result equals reference admission at k<=2, else Overflow.
use abacus_tensor::api::TensorEntry;
use abacus_tensor::graphblas::context::{ContextParameters, ExecutionError, FocalContext};
use abacus_tensor::graphblas::support::build_support;
use abacus_tensor::graphblas::types::{EdgeRef, FeatureVector, FocalTransaction, SelfLoopRule};
use abacus_tensor::graphblas::vector::Vector;
use super::retrieval::{EventLocator, TemporalIndexes};
pub const L1_HOT_BUDGET: usize = 32 * 1024;
pub const BYTES_PER_VERTEX: usize = 8;
pub const BYTES_PER_LOCATOR: usize = 8;
pub const MAX_VERTICES: usize = 1024;
pub const MAX_ADMITTED: usize = 1024;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundedError {
    VertexOverflow { hop: u8, visited: usize },
    AdmittedOverflow { admitted: usize },
    Reference(ExecutionError),
}
impl std::fmt::Display for BoundedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VertexOverflow { hop, visited } => write!(f, "vertex overflow hop {hop}: {visited} > {MAX_VERTICES}"),
            Self::AdmittedOverflow { admitted } => write!(f, "admitted overflow: {admitted} > {MAX_ADMITTED}"),
            Self::Reference(e) => write!(f, "reference failure: {e}"),
        }
    }
}
impl std::error::Error for BoundedError {}
#[derive(Debug, Default, Clone)]
struct SortedSet { members: Vec<u32> }
impl SortedSet {
    fn contains(&self, v: u32) -> bool { self.members.binary_search(&v).is_ok() }
    fn insert(&mut self, v: u32) -> bool {
        match self.members.binary_search(&v) { Ok(_) => false, Err(p) => { self.members.insert(p, v); true } }
    }
    fn len(&self) -> usize { self.members.len() }
}
/// Hop expansion over the existing CSR/CSC rows with **zero resolver calls**.
///
/// Each `TemporalRef` already carries `(t, locator, peer)`. Combined with the
/// row side (forward = `c` sent, backward = `c` received) we recover the full
/// `(t, s, r)` of every candidate event without touching the directory, the
/// CSF, or the feature store. Only `locator` is emitted downstream; `(t, s, r)`
/// is used in-flight for the window and frontier checks.
///
/// Output entries are `(locator, t, s, r)`:
/// - forward row: `c` is the sender, `peer` is the receiver → `(t, c, peer)`
/// - backward row: `c` is the receiver, `peer` is the sender → `(t, peer, c)`
fn step_into(
    t: &TemporalIndexes,
    fr: &[u32],
    ws: u64,
    we: u64,
    o: &mut Vec<(u32, u64, u32, u32)>,
) {
    for &c in fr {
        let row = t.forward.row(c);
        let lo = row.partition_point(|e| e.t < ws);
        let hi = row.partition_point(|e| e.t < we);
        for e in &row[lo..hi] {
            // c sent to peer: s = c, r = peer
            o.push((e.locator.0, e.t, c, e.peer));
        }
        let row = t.backward.row(c);
        let lo = row.partition_point(|e| e.t < ws);
        let hi = row.partition_point(|e| e.t < we);
        for e in &row[lo..hi] {
            // peer sent to c: s = peer, r = c
            o.push((e.locator.0, e.t, e.peer, c));
        }
    }
}



pub type ResolveFn<'a> = dyn Fn(EventLocator) -> Option<(u64, u32, u32, i64, FeatureVector)> + 'a;
pub fn build_bounded_k2(
    t: &TemporalIndexes, f: FocalTransaction, p: &ContextParameters, r: &ResolveFn<'_>,
) -> Result<FocalContext, BoundedError> {
    if p.hop_bound.0 != 2 { return Err(BoundedError::Reference(ExecutionError::FocalOutsideWindow)); }
    if !p.window.contains(f.epoch) { return Err(BoundedError::Reference(ExecutionError::FocalOutsideWindow)); }
    let (ws, we) = (p.window.start, p.window.end);
    // Focal existence via row entries only — no resolver call. The forward row
    // holds (t, locator, peer) where c = sender; the backward row holds
    // (t, locator, peer) where c = receiver.
    let mut found = false;
    let frow = t.forward.row(f.sender.0);
    let fa = frow.partition_point(|e| e.t < f.epoch);
    let fb = frow.partition_point(|e| e.t < f.epoch.saturating_add(1));
    for e in &frow[fa..fb] {
        if e.t == f.epoch && e.peer == f.receiver.0 { found = true; break; }
    }
    if !found {
        let brow = t.backward.row(f.receiver.0);
        let ba = brow.partition_point(|e| e.t < f.epoch);
        let bb = brow.partition_point(|e| e.t < f.epoch.saturating_add(1));
        for e in &brow[ba..bb] {
            if e.t == f.epoch && e.peer == f.sender.0 { found = true; break; }
        }
    }
    if !found { return Err(BoundedError::Reference(ExecutionError::FocalAbsent)); }
    let mut vis = SortedSet::default();
    vis.insert(f.sender.0); vis.insert(f.receiver.0);
    let mut fr: Vec<u32> = if f.sender.0 == f.receiver.0 { vec![f.sender.0] } else { let mut v = vec![f.sender.0, f.receiver.0]; v.sort_unstable(); v };
    let mut harv: Vec<u32> = Vec::new();
    for hop in 1u8..=2u8 {
        let mut step: Vec<(u32, u64, u32, u32)> = Vec::new();
        step_into(t, &fr, ws, we, &mut step);
        step.sort_unstable();
        step.dedup_by(|a, b| a.0 == b.0);
        let frset = SortedSet { members: fr.clone() };
        let mut nf = SortedSet::default();
        for (_loc, tt, s, rr) in &step {
            if *tt < ws || *tt >= we { continue }
            if frset.contains(*s) && !vis.contains(*rr) { nf.insert(*rr); }
            if frset.contains(*rr) && !vis.contains(*s) { nf.insert(*s); }
        }
        if nf.len() == 0 { harv.extend(step.iter().map(|(loc, _, _, _)| *loc)); break; }
        if vis.len() + nf.len() > MAX_VERTICES { return Err(BoundedError::VertexOverflow { hop, visited: vis.len() + nf.len() }); }
        if (vis.len() + nf.len()) * BYTES_PER_VERTEX + harv.len() * BYTES_PER_LOCATOR > L1_HOT_BUDGET {
            return Err(BoundedError::VertexOverflow { hop, visited: vis.len() });
        }
        for v in nf.members.clone() { vis.insert(v); }
        let mut nn = nf.members.clone(); nn.sort_unstable(); fr = nn;
        harv.extend(step.iter().map(|(loc, _, _, _)| *loc));
    }
    harv.sort_unstable(); harv.dedup();
    let excl = p.self_loop == SelfLoopRule::Exclude;
    let mut adm: Vec<TensorEntry> = Vec::new();
    let mut adm_features: Vec<FeatureVector> = Vec::new();
    for loc in harv {
        if adm.len() >= MAX_ADMITTED { return Err(BoundedError::AdmittedOverflow { admitted: adm.len() }); }
        // Single resolve per harvested locator: (v, fv). `v` comes from the CSF
        // via `csf_live_at`; `fv` comes from the deterministic feature store
        // `features[locator]` (Vec index, O(1), no hash). Both are carried
        // verbatim — never recomputed (F13).
        let Some((tt, s, rr, v, fv)) = r(EventLocator(loc)) else { continue };
        if tt < ws || tt >= we { continue }
        if !vis.contains(s) || !vis.contains(rr) { continue }
        if excl && s == rr { continue }
        if let Some(e) = p.eligible.as_ref() { if !e.admits(EdgeRef::new(s, rr)) { continue } }
        if v == 0 { continue }
        adm.push(TensorEntry::new(tt, s, rr, v));
        adm_features.push(fv);
    }
    let mut paired: Vec<(TensorEntry, FeatureVector)> =
        adm.into_iter().zip(adm_features.into_iter()).collect();
    paired.sort_by(|a, b| a.0.cmp(&b.0));
    paired.dedup_by(|a, b| a.0 == b.0);
    let (adm_sorted, adm_features_sorted): (Vec<TensorEntry>, Vec<FeatureVector>) =
        paired.into_iter().unzip();
    let adm = adm_sorted;
    let adm_features = adm_features_sorted;
    if adm.len() > MAX_ADMITTED { return Err(BoundedError::AdmittedOverflow { admitted: adm.len() }); }
    let ts = build_support(&adm).map_err(|e| BoundedError::Reference(ExecutionError::VolumeOverflow(e)))?;
    let vx = Vector::from_sorted(vis.members.clone());
    Ok(FocalContext { focal: f, params: p.clone(), admitted: adm, admitted_features: adm_features, vertices: vx, tensors: ts })
}
#[must_use]
pub fn hot_bytes(c: &FocalContext) -> usize {
    c.vertices.nnz() * BYTES_PER_VERTEX
        + c.admitted.len() * std::mem::size_of::<TensorEntry>()
        + c.admitted_features.len() * std::mem::size_of::<FeatureVector>()
}
