//! Compatibility probe: can the Abacus CSF architecture supply behavioral
//! facts to a declarative-condition layer (the condition side of a Drools
//! rule) without installing Drools and without building a rule engine?
//!
//! Governing addendum: `docs/golden_source_behavioral_rule_boundary_addendum.md`
//! (B1–B11). Usage:
//!
//! ```text
//! rules_probe [profile_dir]     (default artifacts/profiles/samld_90d_3rd)
//! ```
//!
//! Pipeline (B1):
//! events → tokenization → CSF `T[t,s,r]` → temporal/relationship selection
//! → tensor algebra → `BehavioralRelation` → declarative predicates → TRUE/FALSE.
//!
//! The identical code path runs on the synthetic fixture and on the existing
//! SAML-D profile. No second tensor representation, no graph, no labels
//! before prediction, no rule engine (B10).

use abacus_tensor::behavioral::{
    derive_relations, reference_relations, BehavioralRelation, DerivedRelation, Entities,
    RawEvent, Thresholds, Window,
};
use abacus_tensor::csf::compress::{no_self_aggregation, CsfTensor};
use abacus_tensor::csf::query::time_slice;
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::TensorError;
use std::collections::BTreeSet;
use std::time::Instant;

/// Synthetic dictionary: one table used for both coordinate sets, so the
/// entity labels read `A`, `B`, `C`, `D` (row 12 `prec_dict` order).
const SYNTH: [&str; 4] = ["A", "B", "C", "D"];

/// The declared synthetic event set (task §"Synthetic test").
/// `t=1 A→B`, `t=2 A→B`, `t=3 B→C`, `t=4 C→D`, `t=5 A→B`.
/// Amounts are in exact micro-units (P2).
const SYNTH_EVENTS: [(u64, u32, u32, i64); 5] = [
    (1, 0, 1, 10_000_000),
    (2, 0, 1, 20_000_000),
    (3, 1, 2, 30_000_000),
    (4, 2, 3, 40_000_000),
    (5, 0, 1, 50_000_000),
];

/// Result of one probe run over one window.
struct Run {
    /// Row-16 windowed support size.
    events: usize,
    /// The row-16 restriction equalled the reference filter (B11.2).
    selection_ok: bool,
    /// Derived relations (Abacus path).
    abacus: Vec<DerivedRelation>,
    /// Derived relations (independent reference path).
    reference: Vec<DerivedRelation>,
    /// Compound predicate results, Abacus path.
    abacus_pred: Vec<bool>,
    /// Compound predicate results, reference path.
    reference_pred: Vec<bool>,
}

fn build_csf(events: &[RawEvent]) -> Result<(CsfTensor, Vec<TxEntry>), TensorError> {
    // Row 5 (`d_settle`, P4) then row 11 (`compress`, P6).
    let raw: Vec<TxEntry> = events
        .iter()
        .map(|(t, s, r, v)| TxEntry::new(*t, *s, *r, *v))
        .collect();
    let settled = abacus_tensor::composites::settle(raw)?;
    let csf = abacus_tensor::csf::compress::compress(&settled);
    Ok((csf, settled))
}

fn probe(
    csf: &CsfTensor,
    settled: &[TxEntry],
    events: &[RawEvent],
    ents: &Entities<'_>,
    w: Window,
    th: &Thresholds,
) -> Result<Run, TensorError> {
    // --- temporal / relationship selection: one row-16 restriction ---
    let selected = time_slice(csf, w.t_start, w.t_end).entries;
    // The reference filter must agree coordinate-for-coordinate (B11.2).
    let mut sel_set: BTreeSet<(u64, u32, u32, i64)> = BTreeSet::new();
    for e in &selected {
        sel_set.insert((e.t, e.s, e.r, e.value));
    }
    let mut ref_set: BTreeSet<(u64, u32, u32, i64)> = BTreeSet::new();
    for e in settled {
        if e.t >= w.t_start && e.t < w.t_end {
            ref_set.insert((e.t, e.s, e.r, e.value));
        }
    }
    let selection_ok = sel_set == ref_set;
    if !selection_ok {
        return Err(TensorError::Decode(format!(
            "temporal selection mismatch: csf={} reference={} (B11.2)",
            sel_set.len(),
            ref_set.len()
        )));
    }

    // --- Abacus path: tensor state -> BehavioralRelation ---
    let abacus = derive_relations(csf, ents, w)?;
    // --- independent reference path (B7) ---
    let reference = reference_relations(events, ents, w)?;

    let abacus_pred = th.evaluate_all(&abacus);
    let reference_pred = th.evaluate_all(&reference);

    Ok(Run {
        events: selected.len(),
        selection_ok,
        abacus,
        reference,
        abacus_pred,
        reference_pred,
    })
}

fn find<'a>(rels: &'a [DerivedRelation], source: &str, target: &str, hop: u32) -> Option<&'a BehavioralRelation> {
    rels.iter()
        .find(|d| d.relation.source == source && d.relation.target == target && d.relation.hop == hop)
        .map(|d| &d.relation)
}

/// Confusion matrix over transactions (B8). Scored **only** after the
/// relations and predicate results already exist.
#[derive(Default, Clone, Copy)]
struct Score {
    tp: u64,
    fp: u64,
    fn_: u64,
    tn: u64,
}

impl Score {
    fn precision(&self) -> f64 {
        let d = self.tp + self.fp;
        if d == 0 { 0.0 } else { self.tp as f64 / d as f64 }
    }
    fn recall(&self) -> f64 {
        let d = self.tp + self.fn_;
        if d == 0 { 0.0 } else { self.tp as f64 / d as f64 }
    }
    fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rules_probe failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let t_start_all = Instant::now();
    let profile_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/profiles/samld_90d_3rd".to_string());

    let mut fails: Vec<&'static str> = Vec::new();
    // ================= SYNTHETIC =================
    let synth_dict: Vec<String> = SYNTH.iter().map(|s| (*s).to_string()).collect();
    let ents = Entities { senders: &synth_dict, receivers: &synth_dict };
    // Δ = 2 admits (t1=1,t2=3) and (t1=2,t2=3); Δ = 1 admits only the latter.
    let w_syn = Window { t_start: 1, t_end: 6, delta: 2 };
    // Declared predicate parameters (B6): N, X (micro-units), Δ_span, k_min.
    let th_syn = Thresholds { min_events: 3, min_amount_micro: 60_000_000, max_span_secs: 2, min_hop: 2 };

    let t0 = Instant::now();
    let (csf_syn, settled_syn) = build_csf(&SYNTH_EVENTS).map_err(|e| e.to_string())?;
    let syn_ms = t0.elapsed().as_millis();

    // Acceptance 1: CSF invariants + no-self-aggregation.
    let csf_ok = csf_syn.check_invariants() && no_self_aggregation(&settled_syn, &csf_syn);

    let run_syn = probe(&csf_syn, &settled_syn, &SYNTH_EVENTS, &ents, w_syn, &th_syn)
        .map_err(|e| e.to_string())?;

    // Acceptance 3: first hop A→B has three events and the correct span.
    let ab = find(&run_syn.abacus, "A", "B", 1);
    let first_hop_ok = ab.is_some_and(|r| r.event_count == 3 && r.temporal_span == 4
        && r.aggregate_amount == 10_000_000 + 20_000_000 + 50_000_000);
    // Acceptance 4: second hop A→B→C derived as (A, C, hop=2). With Δ=2 the
    // witnessing legs are (t1=1,A,B), (t1=2,A,B), (t2=3,B,C) — the (t1=5)
    // leg fails t1 <= t2.
    let ac2 = find(&run_syn.abacus, "A", "C", 2);
    let second_hop_ok = ac2.is_some_and(|r| {
        r.event_count == 3
            && r.temporal_span == 2
            && r.aggregate_amount == 10_000_000 + 20_000_000 + 30_000_000
    });
    // Tightening Δ to 1 must drop the (t1=1,t2=3) leg; Δ=0 must drop the
    // relation entirely (t1 == t2 is required, and A→B never occurs at t=3).
    let tight1 = derive_relations(&csf_syn, &ents, Window { delta: 1, ..w_syn }).map_err(|e| e.to_string())?;
    let tight0 = derive_relations(&csf_syn, &ents, Window { delta: 0, ..w_syn }).map_err(|e| e.to_string())?;
    let bound_respected = find(&tight1, "A", "C", 2)
        .is_some_and(|r| r.event_count == 2 && r.aggregate_amount == 20_000_000 + 30_000_000)
        && find(&tight0, "A", "C", 2).is_none();

    // Acceptance 5 + 6 (synthetic).
    let ref_eq_syn = run_syn.abacus == run_syn.reference;
    let pred_eq_syn = run_syn.abacus_pred == run_syn.reference_pred;
    let syn_matches = run_syn.abacus_pred.iter().filter(|p| **p).count();
    let compound_ok = syn_matches >= 1
        && run_syn
            .abacus
            .iter()
            .zip(&run_syn.abacus_pred)
            .all(|(d, p)| th_syn.evaluate(&d.relation) == *p);

    println!("== synthetic ==");
    println!("  events={} nnz={}", run_syn.events, csf_syn.nnz());
    for (i, d) in run_syn.abacus.iter().enumerate() {
        println!(
            "  {} -> {} hop={} count={} amount={} span={} P={}",
            d.relation.source, d.relation.target, d.relation.hop,
            d.relation.event_count, d.relation.aggregate_amount, d.relation.temporal_span,
            run_syn.abacus_pred[i]
        );
    }
    println!("  a->b hop1 {:?}", ab.map(|r| (r.event_count, r.aggregate_amount, r.temporal_span)));
    println!("  a->c hop2 {:?}", ac2.map(|r| (r.event_count, r.aggregate_amount, r.temporal_span)));
    println!("  build_ms={syn_ms}");

    // ================= REAL DATA (SAML-D profile) =================
    // 5A.7/B9: the profile is 90 days wide, but the probe consumes only a
    // bounded historical context. Deterministic rule (B9): `t_f` is the
    // mode-0 coordinate at index n0/2 of the existing CSF I0 fiber (the
    // median event time), and W = [t_f - omega, t_f + 1) with a declared
    // omega. The temporal window is reported separately from the number of
    // events actually processed.
    //
    // omega is a declared parameter (semantic §7.5), not a hidden cap on the
    // tensor: the full 90-day profile is still read and its CSF is still
    // constructed; only the *relational context* is bounded. Its default is
    // sized so the window contains genuine multi-hop paths while the
    // independent reference (B7) — a deliberate O(n^2) nested loop over the
    // window's events — remains tractable. `ABACUS_PROBE_OMEGA` overrides.
    let omega: u64 = std::env::var("ABACUS_PROBE_OMEGA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21_600); // 6 hours
    let th_real = Thresholds {
        min_events: 3,
        min_amount_micro: 1_000_000_000, // 1000.00 in exact micro-units
        max_span_secs: omega,
        min_hop: 2,
    };

    let t_load = Instant::now();
    let core_bytes = std::fs::read(format!("{profile_dir}/samld_core.bin")).map_err(|e| e.to_string())?;
    let core = abacus_tensor::samld::deserialize_core_v2(&core_bytes).map_err(|e| e.to_string())?;
    drop(core_bytes);
    let (csf_real, settled_real) =
        abacus_tensor::samld::direct_csf(&core).map_err(|e| e.to_string())?;
    let load_s = t_load.elapsed().as_secs_f64();

    // B9.1: the focal time is read from the existing I0 fiber.
    let n0 = csf_real.mode0_coords.len();
    let t_f = csf_real.mode0_coords[n0 / 2];
    let w_real = Window {
        t_start: t_f.saturating_sub(omega),
        t_end: t_f + 1,
        delta: omega, // Δ <= ω, per B9.4
    };
    // B8: the raw event list for the reference path and the label join. This
    // is the existing ordered binary (no CSV, no re-derivation).
    let raw: Vec<RawEvent> = core
        .records
        .iter()
        .map(|r| (r.t, r.s, r.r, r.amount_micro))
        .collect();

    let ents_real = Entities { senders: &core.senders, receivers: &core.receivers };

    let t_probe = Instant::now();
    let run_real = probe(&csf_real, &settled_real, &raw, &ents_real, w_real, &th_real)
        .map_err(|e| e.to_string())?;
    let probe_s = t_probe.elapsed().as_secs_f64();

    let ref_eq_real = run_real.abacus == run_real.reference;
    let pred_eq_real = run_real.abacus_pred == run_real.reference_pred;
    let real_matches = run_real.abacus_pred.iter().filter(|p| **p).count();

    // ---- label isolation (B8) -------------------------------------------
    // Snapshot relations and predicate results BEFORE any label is read; the
    // probe is re-run afterwards and must reproduce them byte-for-byte.
    let before_rels = format!("{:?}", run_real.abacus);
    let before_pred = format!("{:?}", run_real.abacus_pred);

    // ---- scoring (B8) — the ONLY place a label is touched ---------------
    let side_bytes =
        std::fs::read(format!("{profile_dir}/samld_eval.sidecar")).map_err(|e| e.to_string())?;
    let side = abacus_tensor::samld::deserialize_eval_v3(&side_bytes).map_err(|e| e.to_string())?;
    drop(side_bytes);
    if side.len() != core.records.len() {
        return Err("sidecar/core length mismatch (addendum S4/S6)".into());
    }

    // Union of A_hop over the TRUE relations = the predicted coordinate set.
    let mut predicted: BTreeSet<(u64, u32, u32)> = BTreeSet::new();
    for (d, p) in run_real.abacus.iter().zip(&run_real.abacus_pred) {
        if *p {
            predicted.extend(d.admitted.iter().map(|(c, _)| (c.t, c.s, c.r)));
        }
    }

    // Core records are in `(t,s,r)` order, so the window is a contiguous
    // slice; the sidecar joins positionally (`tx_id` is the sole carrier).
    let lo = core.records.partition_point(|r| r.t < w_real.t_start);
    let hi = core.records.partition_point(|r| r.t < w_real.t_end);
    let mut score = Score::default();
    for (i, r) in core.records[lo..hi].iter().enumerate() {
        let truth = side[lo + i].laundering != 0;
        let pred = predicted.contains(&(r.t, r.s, r.r));
        match (pred, truth) {
            (true, true) => score.tp += 1,
            (true, false) => score.fp += 1,
            (false, true) => score.fn_ += 1,
            (false, false) => score.tn += 1,
        }
    }
    drop(side);

    // Re-derive with the labels resident: nothing may change (B8).
    let after = probe(&csf_real, &settled_real, &raw, &ents_real, w_real, &th_real)
        .map_err(|e| e.to_string())?;
    let label_isolation =
        format!("{:?}", after.abacus) == before_rels && format!("{:?}", after.abacus_pred) == before_pred;

    let total_s = t_start_all.elapsed().as_secs_f64();
    let mut rep = |name: &'static str, pass: bool| {
        if !pass {
            fails.push(name);
        }
    };
    rep("CSF construction", csf_ok && csf_real.check_invariants());
    rep("Temporal selection", run_syn.selection_ok && run_real.selection_ok);
    rep("First-hop derivation", first_hop_ok);
    rep("Second-hop derivation", second_hop_ok && bound_respected);
    rep("Reference equivalence", ref_eq_syn && ref_eq_real);
    rep("Predicate evaluation", pred_eq_syn && pred_eq_real && compound_ok);
    rep("Label isolation", label_isolation);
    rep("SAML-D execution", real_matches > 0);

    let pf = |n: &str| if fails.contains(&n) { "FAIL" } else { "PASS" };
    println!();
    println!("CSF construction: {}", pf("CSF construction"));
    println!("Temporal selection: {}", pf("Temporal selection"));
    println!("First-hop derivation: {}", pf("First-hop derivation"));
    println!("Second-hop derivation: {}", pf("Second-hop derivation"));
    println!("Reference equivalence: {}", pf("Reference equivalence"));
    println!("Predicate evaluation: {}", pf("Predicate evaluation"));
    println!("Label isolation: {}", pf("Label isolation"));
    println!("SAML-D execution: {}", pf("SAML-D execution"));
    println!();
    println!("events: {} (window {} s of the {}-day profile)", run_real.events, w_real.t_end - w_real.t_start, 90);
    println!("NNZ: {}", csf_real.nnz());
    println!("behavioral relations: {} (hop1 {}, hop2 {})",
        run_real.abacus.len(),
        run_real.abacus.iter().filter(|d| d.relation.hop == 1).count(),
        run_real.abacus.iter().filter(|d| d.relation.hop == 2).count());
    println!("predicate matches: {} (synthetic {} / SAML-D {})", real_matches, syn_matches, real_matches);
    println!();
    println!("TP={} FP={} FN={} TN={}", score.tp, score.fp, score.fn_, score.tn);
    println!("precision={:.4} recall={:.4} F1={:.4}", score.precision(), score.recall(), score.f1());
    println!("runtime: load+csf={load_s:.2}s probe={probe_s:.2}s total={total_s:.2}s");
    println!("memory: 90-day profile ({} records, nnz {}) read; {} events processed in the {} s window",
        core.records.len(), csf_real.nnz(), run_real.events, w_real.t_end - w_real.t_start);
    if !fails.is_empty() {
        return Err(format!("failed checks: {fails:?}"));
    }
    Ok(())
}

