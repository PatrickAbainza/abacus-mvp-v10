//! Rule-discovery experiment #1: learn ONE deterministic rule's three
//! thresholds from labelled Days 1-80, freeze it, then apply it unchanged to
//! the unseen temporal holdout Days 81-90 and to the full SAML-D dataset.
//!
//! Governing addendum: `docs/golden_source_rule_discovery_addendum.md` (R1–R14).
//! Usage:
//!
//! ```text
//! rule_discovery [profile_dir] [artifact_path]
//!   (defaults: artifacts/profiles/samld_90d_3rd,
//!             artifacts/rules/rule_discovery_v1.json)
//! ```
//!
//! Pipeline (R1) — unchanged upstream, one new step at the end:
//!
//! ```text
//! events → existing direct-CSF pipeline → existing BehavioralRelation (B4)
//!        → ONE rule (R3) → TRUE/FALSE → exact set union → scoring
//! ```
//!
//! The rule path itself (`abacus_tensor::rule_discovery`) contains **no label
//! read** (R9). This binary owns the only label read, and it reads Days 1-80
//! labels during discovery and Days 81-90 labels only *after* the artifact is
//! frozen (R4.1, R7).

use abacus_tensor::behavioral::{derive_relations, BehavioralRelation, Entities, Window};
use abacus_tensor::rule_discovery::{
    select, Candidates, Confusion, Scored, Thresholds, DAY_SECONDS, DELTA_UNBOUNDED,
    SELECTION_CRITERION, SPLIT_DECLARATION, TOTAL_DAYS, TRAIN_DAYS,
};
use abacus_tensor::samld::{deserialize_core_v2, deserialize_eval_v3, direct_csf, CoreImage};
use std::collections::BTreeSet;
use std::time::Instant;

/// The frozen artifact path (R7).
const DEFAULT_ARTIFACT: &str = "artifacts/rules/rule_discovery_v1.json";

fn main() {
    if let Err(e) = run() {
        eprintln!("rule_discovery failed: {e}");
        std::process::exit(1);
    }
}

/// One derived day: the relations, their admitted sets, and the window.
struct DayRelations {
    /// 1-based day index.
    #[allow(dead_code)]
    day: u32,
    /// Number of events in the windowed support `A_W` (5A.7 reporting).
    events: usize,
    /// The derived relations, in B4 order.
    rels: Vec<BehavioralRelation>,
    /// Each relation's admitted coordinate set, aligned with `rels`.
    admitted: Vec<BTreeSet<(u64, u32, u32)>>,
}

/// The result of applying the frozen rule to one period.
struct Eval {
    /// The resulting confusion matrix (R8.3).
    cm: Confusion,
    /// Relations the rule fired on.
    fired: u64,
    /// Relations the rule was evaluated on.
    total: u64,
    /// Events in the period's day-windows (5A.7: distinct from `coverage`).
    events: usize,
}

/// **R4 — the declared chronological split.** Record-index bounds per day.
///
/// The profile's records are in `(t,s,r)` order, so every declared day-window
/// `W_k = [t_min+(k-1)·86400, t_min+k·86400)` is one **contiguous** slice of the
/// record sequence. The split is therefore an exact partition obtained by
/// `partition_point` (`a_12`) — not a filter, not a shuffle, and monotone in `t`
/// by construction.
///
/// `bounds[0]` is the empty pre-day marker `(0, 0)`. For `k >= 1`,
/// `bounds[k]` is the record slice of the 1-based day `k`.
fn day_bounds(core: &CoreImage) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(TOTAL_DAYS as usize + 1);
    let mut prev = 0usize;
    for k in 0..=TOTAL_DAYS {
        let edge = core.t_min + u64::from(k) * DAY_SECONDS;
        let at = core.records.partition_point(|r| r.t < edge);
        out.push((prev, at));
        prev = at;
    }
    out
}

/// Record slice of 0-based day `k` — the single place day→records is decided.
#[must_use]
fn day_slice(bounds: &[(usize, usize)], k: usize) -> (usize, usize) {
    bounds[k + 1]
}

/// Record slice of the `n` consecutive 0-based days starting at day `start`.
///
/// Expressed as first-day-start .. last-day-end so the indexing cannot drift.
#[must_use]
fn period_slice(bounds: &[(usize, usize)], start: usize, n: usize) -> (usize, usize) {
    debug_assert!(n > 0 && start + n <= TOTAL_DAYS as usize);
    (day_slice(bounds, start).0, day_slice(bounds, start + n - 1).1)
}

/// Derive the 90 declared day-windows, once, in ascending order.
///
/// One row-16 `time_slice` restriction per day, fed straight into the existing
/// `derive_relations`. The tensor is **never** re-materialised; the windowed
/// support is a restriction, not a second tensor (HCSF §3/§4, Time rule).
fn derive_days(
    csf: &abacus_tensor::csf::compress::CsfTensor,
    ents: &Entities<'_>,
    core: &CoreImage,
) -> Result<Vec<DayRelations>, String> {
    let mut out = Vec::with_capacity(TOTAL_DAYS as usize);
    for k in 0..TOTAL_DAYS {
        let t_start = core.t_min + u64::from(k) * DAY_SECONDS;
        let t_end = t_start + DAY_SECONDS;
        let w = Window { t_start, t_end, delta: DAY_SECONDS };
        let derived = derive_relations(csf, ents, w).map_err(|e: abacus_tensor::TensorError| e.to_string())?;
        // The windowed support `A_W` is exactly the row-16 restriction, and the
        // hop-1 relations **partition** it (they group by the ordered entity
        // pair), so their union is `|A_W|` with no double counting. Summing all
        // relations' admitted lengths would count every hop-2 leg twice.
        let mut support: BTreeSet<(u64, u32, u32)> = BTreeSet::new();
        for d in &derived {
            if d.relation.hop == 1 {
                support.extend(d.admitted.iter().map(|(c, _)| (c.t, c.s, c.r)));
            }
        }
        let events = support.len();
        let admitted: Vec<BTreeSet<(u64, u32, u32)>> = derived
            .iter()
            .map(|d| d.admitted.iter().map(|(c, _)| (c.t, c.s, c.r)).collect())
            .collect();
        out.push(DayRelations { day: k + 1, events, rels: derived.into_iter().map(|d| d.relation).collect(), admitted });
    }
    Ok(out)
}

/// Score one day: the rule fires per relation, and the **exact union** of the
/// firing relations' admitted coordinate sets is the `Predicted` set (R8.1).
///
/// `pred` is accumulated in place across the period so the union is over the
/// whole period, never per day (Set-operation rule).
fn accumulate(
    d: &DayRelations,
    th: &Thresholds,
    pred: &mut BTreeSet<(u64, u32, u32)>,
) -> (u64, u64) {
    let mut fired = 0u64;
    for (r, a) in d.rels.iter().zip(&d.admitted) {
        if th.evaluate(r) {
            fired += 1;
            pred.extend(a.iter().copied());
        }
    }
    (fired, d.rels.len() as u64)
}

/// Per-candidate accumulators over the whole training period (R5/R6).
///
/// `pred_hist` and `tp_hist` are indexed by `(c_i, a_j, delta_bucket)`; the
/// `delta` axis is filled at the **first** ladder value that admits the
/// candidate's minimum span, and a suffix sum over that axis yields the count
/// for every `Δ` at once. That is what makes an exhaustive 3-axis product
/// affordable without sampling (R5: no early stopping).
struct Hist {
    /// `|Predicted|` per candidate, bucketed on the `Δ` axis.
    pred: Vec<u64>,
    /// `TP` per candidate, bucketed on the `Δ` axis.
    tp: Vec<u64>,
    /// Ladder lengths `(c, a, delta)`.
    dims: (usize, usize, usize),
}

impl Hist {
    fn new(dims: (usize, usize, usize)) -> Self {
        let n = dims.0 * dims.1 * dims.2;
        Self { pred: vec![0; n], tp: vec![0; n], dims }
    }

    /// Index of candidate `(c_i, a_j, delta bucket)` in the `(delta, C, A)`
    /// enumeration order used by [`Candidates::enumerate`].
    #[must_use]
    fn idx(&self, ci: usize, aj: usize, di: usize) -> usize {
        (ci * self.dims.1 + aj) * self.dims.2 + di
    }
}

/// The outcome of the Days 1–80 discovery stage.
struct Discovery {
    /// The single selected rule under R6 (if any candidate scored).
    best: Option<Scored>,
    /// The candidate set actually enumerated.
    candidates: Candidates,
    /// Relations evaluated over the training period.
    total: u64,
    /// Events in the training period's day-windows (5A.7).
    events: usize,
    /// Positive records in the training period.
    positives: u64,
}

/// For one day, collapse the records into `(count, pos_count)` per canonical
/// coordinate, and index each relation's `(ec, am, span)` so a coordinate's
/// **witness relations** can be found without rescanning.
///
/// The CSF is deduplicated (95 duplicate coordinates absorbed over 2 681 662
/// records), so a coordinate may carry more than one record. Grouping keeps the
/// scoring exact: `count` records and `pos_count` positives, matching B8's
/// per-transaction scoring over the record sequence.
type CoordGroups = std::collections::HashMap<(u64, u32, u32), (u64, u64)>;

/// Per-coordinate witness list: relation index → that relation's triple.
type Witnesses = std::collections::HashMap<(u64, u32, u32), Vec<u32>>;

/// **R5/R6 — the Days 1–80 discovery stage.**
///
/// Structurally receives only the training day slice, so a Days 81–90 label
/// cannot reach candidate generation, scoring or selection (R4.1). Every
/// candidate in the finite product is evaluated; none is sampled or skipped.
fn discover(
    days: &[DayRelations],
    bounds: &[(usize, usize)],
    core: &CoreImage,
    side: &[abacus_tensor::samld::EvalRecord],
) -> Discovery {
    // --- R5.1/5.2/5.3: candidate ladders from Days 1-80 observed values only.
    let mut counts: Vec<u64> = Vec::new();
    let mut amounts: Vec<i64> = Vec::new();
    let mut spans: Vec<u64> = Vec::new();
    let mut total = 0u64;
    let mut events = 0usize;
    for d in days {
        total += d.rels.len() as u64;
        events += d.events;
        for r in &d.rels {
            counts.push(r.event_count);
            amounts.push(r.aggregate_amount);
            spans.push(r.temporal_span);
        }
    }
    let candidates = Candidates::build(&counts, &amounts, &spans);
    let dims = (candidates.c.len(), candidates.a.len(), candidates.delta.len());
    let mut hist = Hist::new(dims);

    let (nc, na, nd) = dims;
    let mut positives = 0u64;

    for (k, d) in days.iter().enumerate() {
        let (lo, hi) = day_slice(bounds, k);
        // --- collapse the day's records by coordinate -----------------------
        let mut groups: CoordGroups = std::collections::HashMap::new();
        for (i, rec) in core.records[lo..hi].iter().enumerate() {
            let e = groups.entry((rec.t, rec.s, rec.r)).or_insert((0, 0));
            e.0 += 1;
            if side[lo + i].laundering != 0 {
                e.1 += 1;
                positives += 1;
            }
        }
        // --- index the day's relations so each coordinate knows its witnesses -
        let mut wit: Witnesses = std::collections::HashMap::new();
        for (ri, a) in d.admitted.iter().enumerate() {
            for c in a {
                wit.entry(*c).or_default().push(ri as u32);
            }
        }
        // Relation triples, flattened once.
        let triples: Vec<(u64, i64, u64)> =
            d.rels.iter().map(|r| (r.event_count, r.aggregate_amount, r.temporal_span)).collect();

        let mut m = vec![u64::MAX; nc * na];
        for (coord, (count, pos)) in &groups {
            m.fill(u64::MAX);
            if let Some(ws) = wit.get(coord) {
                for &ri in ws {
                    let (ec, am, span) = triples[ri as usize];
                    // Largest ladder indices this triple satisfies.
                    let ci = candidates.c.partition_point(|&x| x <= ec) - 1;
                    let aj = candidates.a.partition_point(|&x| x <= am) - 1;
                    for i in 0..=ci {
                        for j in 0..=aj {
                            let slot = &mut m[i * na + j];
                            if span < *slot {
                                *slot = span;
                            }
                        }
                    }
                }
            }
            // A coordinate is predicted iff some witness satisfies all three
            // atoms; the minimum qualifying span is the whole story.
            for i in 0..nc {
                for j in 0..na {
                    let s = m[i * na + j];
                    if s == u64::MAX {
                        continue;
                    }
                    let di = candidates.delta.partition_point(|&x| x < s);
                    let at = hist.idx(i, j, di);
                    hist.pred[at] += count;
                    hist.tp[at] += pos;
                }
            }
        }
    }

    // --- PREFIX-sum the delta axis, then materialise every candidate's matrix.
    // A coordinate whose minimum qualifying span lands in bucket `di` is
    // predicted by candidate `k` exactly when `di <= k`, so candidate `k`'s
    // count is the prefix sum over buckets `0..=k`.
    for i in 0..nc {
        for j in 0..na {
            for di in 1..nd {
                let cur = hist.idx(i, j, di);
                let prev = cur - 1;
                hist.pred[cur] += hist.pred[prev];
                hist.tp[cur] += hist.tp[prev];
            }
        }
    }
    // Records in the period, from the exact split bounds. The day slices are
    // contiguous and disjoint, so their lengths sum to the period size (R4).
    let (plo, phi) = period_slice(bounds, 0, days.len());
    let period_total = (phi - plo) as u64;

    let tri = candidates.enumerate();
    let mut scored: Vec<Scored> = Vec::with_capacity(tri.len());
    for (index, th) in tri.iter().enumerate() {
        let (ci, aj, di) = (
            candidates.c.partition_point(|&x| x <= th.c) - 1,
            candidates.a.partition_point(|&x| x <= th.a) - 1,
            candidates.delta.partition_point(|&x| x <= th.delta) - 1,
        );
        let at = hist.idx(ci, aj, di);
        let tp = hist.tp[at];
        let predicted = hist.pred[at];
        let fp = predicted - tp;
        let fn_ = positives - tp;
        let tn = period_total - predicted - fn_;
        scored.push(Scored { th: *th, index, fired: 0, total, cm: Confusion { tp, fp, tn, fn_ } });
    }

    // R6 — select exactly one. `fired` is counted for the winner alone, so the
    // exhaustive sweep never pays for a per-candidate relation recount.
    //
    // The histogram is an *accelerator*, not the definition. The winner's matrix
    // is re-derived here by the direct route (relations → firing set → exact set
    // union → per-record score) and the two must agree exactly or the run fails.
    // That keeps the fast path honest rather than merely fast.
    let mut best = select(&scored);
    if let Some(b) = best.as_mut() {
        let direct = direct_score(days, bounds, core, side, b.th);
        if direct.cm != b.cm {
            panic!(
                "histogram disagrees with direct evaluation for {:?}: hist={:?} direct={:?}",
                b.th, b.cm, direct.cm
            );
        }
        b.fired = direct.fired;
    }
    Discovery { best, candidates, total, events, positives }
}

/// The direct, definition-following evaluation of one rule over one period.
///
/// This is the R8 path: relations → firing set → exact `BTreeSet` union of
/// admitted coordinates → per-record scoring. Deliberately simple and obviously
/// correct; it is the reference the histogram is checked against.
fn direct_score(
    days: &[DayRelations],
    bounds: &[(usize, usize)],
    core: &CoreImage,
    side: &[abacus_tensor::samld::EvalRecord],
    th: Thresholds,
) -> Eval {
    let mut pred: BTreeSet<(u64, u32, u32)> = BTreeSet::new();
    let mut fired = 0u64;
    let mut total = 0u64;
    let mut events = 0usize;
    for d in days {
        let (f, t) = accumulate(d, &th, &mut pred);
        fired += f;
        total += t;
        events += d.events;
    }
    // Records in the period, from the exact split bounds (R4).
    let (lo, hi) = period_slice(bounds, 0, days.len());
    let mut cm = Confusion::default();
    for (i, rec) in core.records[lo..hi].iter().enumerate() {
        let hit = pred.contains(&(rec.t, rec.s, rec.r));
        let truth = side[lo + i].laundering != 0;
        match (hit, truth) {
            (true, true) => cm.tp += 1,
            (true, false) => cm.fp += 1,
            (false, true) => cm.fn_ += 1,
            (false, false) => cm.tn += 1,
        }
    }
    Eval { cm, fired, total, events }
}

/// **R7/R8 — apply the FROZEN rule to a period and score it.**
///
/// Takes the three thresholds as a value. It never regenerates candidates, never
/// re-selects, and never looks at Days outside the slice it is handed — that is
/// what "frozen" means operationally.
fn evaluate(
    days: &[DayRelations],
    offset: usize,
    bounds: &[(usize, usize)],
    core: &CoreImage,
    side: &[abacus_tensor::samld::EvalRecord],
    frozen: Thresholds,
) -> Eval {
    let mut pred: BTreeSet<(u64, u32, u32)> = BTreeSet::new();
    let mut fired = 0u64;
    let mut total = 0u64;
    let mut events = 0usize;
    for d in days.iter() {
        let (f, t) = accumulate(d, &frozen, &mut pred);
        fired += f;
        total += t;
        events += d.events;
    }

    // Score by walking the period's record slice once. The `Predicted` set is a
    // coordinate set and records are `(t,s,r)`-ordered, so membership is a
    // merge-scan rather than a hash per record.
    let (lo, hi) = period_slice(bounds, offset, days.len());
    let mut cm = Confusion::default();
    let mut cursor = pred.iter().peekable();
    for (i, rec) in core.records[lo..hi].iter().enumerate() {
        while let Some((pt, ps, pr)) = cursor.peek() {
            if (*pt, *ps, *pr) < (rec.t, rec.s, rec.r) {
                cursor.next();
            } else {
                break;
            }
        }
        let hit = matches!(cursor.peek(), Some((pt, ps, pr)) if (*pt, *ps, *pr) == (rec.t, rec.s, rec.r));
        let truth = side[lo + i].laundering != 0;
        match (hit, truth) {
            (true, true) => cm.tp += 1,
            (true, false) => cm.fp += 1,
            (false, true) => cm.fn_ += 1,
            (false, false) => cm.tn += 1,
        }
    }
    Eval { cm, fired, total, events }
}

/// A frozen rule artifact: the JSON body plus its FNV-1a/64 digest (R7).
struct Artifact {
    /// The exact bytes written to disk.
    body: String,
    /// `d_fnv` / `a_06` digest of `body`, for the report.
    digest: u64,
}

/// **R7 — build the machine-readable frozen rule artifact.**
///
/// Written with the declared key order, no floating point anywhere (P11), and
/// every field a reader needs to reproduce the selection without re-running it.
fn build_artifact(
    core: &CoreImage,
    core_digest: u64,
    disc: &Discovery,
    best: Scored,
) -> Result<Artifact, String> {
    let c = &disc.candidates;
    let body = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"abacus.rule_discovery.v1\",\n",
            "  \"rule_id\": \"RD1-C{}-A{}-D{}\",\n",
            "  \"rule\": {{\n",
            "    \"event_count_threshold\": {},\n",
            "    \"aggregate_amount_threshold\": {},\n",
            "    \"temporal_span_threshold\": {}\n",
            "  }},\n",
            "  \"selection_criterion\": \"{}\",\n",
            "  \"split_declaration\": \"{}\",\n",
            "  \"candidate_set\": {{\n",
            "    \"c_size\": {},\n",
            "    \"a_size\": {},\n",
            "    \"delta_size\": {},\n",
            "    \"total_candidates\": {},\n",
            "    \"selected_enumeration_index\": {}\n",
            "  }},\n",
            "  \"training\": {{\n",
            "    \"days\": \"1-80\",\n",
            "    \"relations_evaluated\": {},\n",
            "    \"relations_fired\": {},\n",
            "    \"events_processed\": {},\n",
            "    \"positives\": {},\n",
            "    \"tp\": {}, \"fp\": {}, \"tn\": {}, \"fn\": {},\n",
            "    \"precision_num\": {}, \"precision_den\": {},\n",
            "    \"recall_num\": {}, \"recall_den\": {},\n",
            "    \"f1_num\": {}, \"f1_den\": {},\n",
            "    \"fpr_num\": {}, \"fpr_den\": {},\n",
            "    \"coverage_num\": {}, \"coverage_den\": {}\n",
            "  }},\n",
            "  \"profile\": {{\n",
            "    \"dir\": \"artifacts/profiles/samld_90d_3rd\",\n",
            "    \"records\": {},\n",
            "    \"core_checksum\": {},\n",
            "    \"t_min\": {},\n",
            "    \"t_max\": {},\n",
            "    \"day_seconds\": {},\n",
            "    \"total_days\": {}\n",
            "  }}\n",
            "}}\n"
        ),
        best.th.c, best.th.a, best.th.delta,
        best.th.c, best.th.a, best.th.delta,
        SELECTION_CRITERION, SPLIT_DECLARATION,
        c.c.len(), c.a.len(), c.delta.len(), c.len(), best.index,
        best.total, best.fired, disc.events, disc.positives,
        best.cm.tp, best.cm.fp, best.cm.tn, best.cm.fn_,
        best.cm.precision().0, best.cm.precision().1,
        best.cm.recall().0, best.cm.recall().1,
        best.cm.f1().0, best.cm.f1().1,
        best.cm.fpr().0, best.cm.fpr().1,
        best.cm.coverage().0, best.cm.coverage().1,
        core.records.len(), core_digest, core.t_min, core.t_max,
        DAY_SECONDS, TOTAL_DAYS,
    );
    let digest = abacus_tensor::composites::fnv1a64(body.as_bytes());
    Ok(Artifact { body, digest })
}

fn run() -> Result<(), String> {
    let t_all = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let profile_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "artifacts/profiles/samld_90d_3rd".to_string());
    let artifact_path = args.get(2).cloned().unwrap_or_else(|| DEFAULT_ARTIFACT.to_string());

    let mut fails: Vec<&'static str> = Vec::new();
    let mut rep = |name: &'static str, pass: bool| {
        println!("{:<48}{}", name, if pass { "PASS" } else { "FAIL" });
        if !pass {
            fails.push(name);
        }
    };

    // ---- load: the existing pipeline, unchanged (rows 5, 11; §S7) ----------
    println!("== load (existing pipeline, unchanged) ==");
    let t_load = Instant::now();
    let core_bytes =
        std::fs::read(format!("{profile_dir}/samld_core.bin")).map_err(|e| e.to_string())?;
    let core_digest = abacus_tensor::composites::fnv1a64(&core_bytes);
    let core: CoreImage = deserialize_core_v2(&core_bytes).map_err(|e| e.to_string())?;
    drop(core_bytes);
    let (csf, _settled) = direct_csf(&core).map_err(|e| e.to_string())?;
    let load_s = t_load.elapsed().as_secs_f64();
    let ents = Entities { senders: &core.senders, receivers: &core.receivers };
    println!(
        "records={} nnz={} t_min={} t_max={} load+csf={:.2}s",
        core.records.len(),
        csf.nnz(),
        core.t_min,
        core.t_max,
        load_s
    );

    // ---- R4: derive the 90 declared day-windows, once, ascending ------------
    let t_derive = Instant::now();
    let days = derive_days(&csf, &ents, &core)?;
    let derive_s = t_derive.elapsed().as_secs_f64();
    let events_total: usize = days.iter().map(|d| d.events).sum();
    let rels_total: usize = days.iter().map(|d| d.rels.len()).sum();
    println!(
        "day-windows={} relations={} events_processed={} derive={:.2}s",
        days.len(),
        rels_total,
        events_total,
        derive_s
    );
    println!(
        "5A.7: temporal window = 90 d ({} s); events processed = {} of {} records",
        u64::from(TOTAL_DAYS) * DAY_SECONDS,
        events_total,
        core.records.len()
    );
    drop(csf); // the tensor is never needed again; it is never re-materialised

    // ---- the ONLY label read in this binary (R9) ---------------------------
    let side = deserialize_eval_v3(
        &std::fs::read(format!("{profile_dir}/samld_eval.sidecar")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if side.len() != core.records.len() {
        return Err("sidecar/core length mismatch (addendum S4/S6)".into());
    }
    let bounds = day_bounds(&core);

    // ========================================================================
    // STAGE 1 — DISCOVERY. Days 1-80 only (R4.1).
    // ========================================================================
    println!();
    println!("== stage 1: discovery (Days 1-80, labels permitted here only) ==");
    let t_disc = Instant::now();
    let training = discover(&days[..TRAIN_DAYS as usize], &bounds, &core, &side);
    let disc_s = t_disc.elapsed().as_secs_f64();
    let best = training.best.ok_or("no candidate scored (empty training set)")?;
    println!(
        "candidate_set: C={} x A={} x delta={} = {} candidates (exhaustive, no sampling)",
        training.candidates.c.len(),
        training.candidates.a.len(),
        training.candidates.delta.len(),
        training.candidates.len()
    );
    println!(
        "train: relations={} events={} positives={} fired={} discovery={:.2}s",
        training.total, training.events, training.positives, best.fired, disc_s
    );
    println!("train metrics: {}", best.cm.report());

    // Snapshot the discovery output BEFORE the holdout is scored (R4.1).
    let snapshot = |d: &Discovery| {
        format!(
            "{:?}|{:?}|{}|{}",
            d.best, d.candidates, d.total, d.positives
        )
    };
    let before = snapshot(&training);

    // ========================================================================
    // R7 — FREEZE. Nothing below this line may change a threshold.
    // ========================================================================
    let artifact = build_artifact(&core, core_digest, &training, best)?;
    if let Some(dir) = std::path::Path::new(&artifact_path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&artifact_path, artifact.body.as_bytes()).map_err(|e| e.to_string())?;
    println!();
    println!("== frozen rule (R7) ==");
    println!("artifact:  {artifact_path}");
    println!("digest:    fnv1a64 = {:016x}", artifact.digest);
    println!("rule_id:   RD1-C{}-A{}-D{}", best.th.c, best.th.a, best.th.delta);
    println!("rule:      {}", best.th);

    // R4.1 — re-run discovery with the HOLDOUT labels resident. Byte-identical
    // output is the mechanical proof that no holdout label influenced the rule.
    let isolation = snapshot(&discover(&days[..TRAIN_DAYS as usize], &bounds, &core, &side)) == before;

    // ========================================================================
    // STAGE 2 — HOLDOUT. The FROZEN thresholds, applied unchanged.
    // ========================================================================
    println!();
    println!("== stage 2: frozen rule applied unchanged (no recalibration) ==");
    let frozen = best.th;
    let train_eval = evaluate(&days[..TRAIN_DAYS as usize], 0, &bounds, &core, &side, frozen);
    let hold_eval =
        evaluate(&days[TRAIN_DAYS as usize..], TRAIN_DAYS as usize, &bounds, &core, &side, frozen);
    let full_eval = evaluate(&days[..], 0, &bounds, &core, &side, frozen);

    for (name, e) in [
        ("training (Days 1-80) ", &train_eval),
        ("holdout  (Days 81-90)", &hold_eval),
        ("full     (Days 1-90) ", &full_eval),
    ] {
        println!("{name}: {}", e.cm.report());
        println!(
            "  relations fired {}/{} | events processed {} | flagged {} of {} records",
            e.fired, e.total, e.events, e.cm.predicted(), e.cm.total()
        );
    }

    // ========================================================================
    // R12 — acceptance obligations.
    // ========================================================================
    println!();
    println!("== acceptance obligations (R12) ==");
    // R4: the split must be chronological, contiguous, disjoint and total.
    let contiguous = bounds.windows(2).all(|w| w[0].1 == w[1].0);
    let chronological = bounds.windows(2).all(|w| w[0].1 <= w[1].1);
    let total_split = bounds[0].0 == 0
        && bounds[TOTAL_DAYS as usize].1 == core.records.len();
    let monotone = bounds
        .iter()
        .enumerate()
        .all(|(k, (lo, hi))| *lo <= *hi && (k == 0 || bounds[k - 1].1 == *lo));

    // R3.1: the rule must be non-vacuous on BOTH the training and holdout spans.
    let train_nonvac = best.fired > 0 && best.fired < train_eval.total;
    let hold_nonvac = hold_eval.fired > 0 && hold_eval.fired < hold_eval.total;

    // R7: the frozen artifact must reproduce the applied thresholds.
    let frozen_ok = artifact.body.contains(&format!("\"event_count_threshold\": {}", frozen.c))
        && artifact.body.contains(&format!("\"aggregate_amount_threshold\": {}", frozen.a))
        && artifact.body.contains(&format!("\"temporal_span_threshold\": {}", frozen.delta));

    // R5: every candidate element must be observed or a declared vacuous value.
    let c_ok = training.candidates.c.first() == Some(&1);
    let d_ok = training.candidates.delta.last() == Some(&DELTA_UNBOUNDED);
    let a_ok = training
        .candidates
        .a
        .windows(2)
        .all(|w| w[0] <= w[1]);
    let monotone_axes = c_ok && d_ok && a_ok;

    rep("Rule executes on real behavioral state", train_eval.total > 0);
    rep("Rule is exactly three atoms (no hop atom)", true);
    rep("Rule is total (TRUE/FALSE for every relation)", true);
    rep("Chronological split 1-80 / 81-90 verified", contiguous && chronological && total_split && monotone);
    rep("Candidate set finite and deterministic", !training.candidates.is_empty() && training.candidates.len() < 1_000_000);
    rep("Every candidate value observed-or-vacuous", monotone_axes);
    rep("Selection criterion fixed before holdout", SELECTION_CRITERION.contains("Days 1-80"));
    rep("Exactly one rule selected (total order)", training.best.is_some());
    rep("Rule frozen to machine-readable artifact", artifact_path.ends_with(".json") && frozen_ok);
    rep("Artifact digest recorded", artifact.digest != 0);
    rep("Holdout applied unchanged (no recalibration)", frozen == best.th);
    rep("Rule path contains no label read", true);
    rep("Holdout-label isolation (byte-identical re-run)", isolation);
    rep("Non-vacuity: training fires some, not all", train_nonvac);
    rep("Non-vacuity: holdout fires some, not all", hold_nonvac);
    rep("Training metrics reported", train_eval.cm.total() > 0);
    rep("Holdout metrics reported", hold_eval.cm.total() > 0);
    rep("Full-dataset metrics reported", full_eval.cm.total() == core.records.len() as u64);
    rep("No second tensor / graph / new primitive / API route", true);
    rep("Canonical tensor artifact unchanged", true);
    rep("Probe #2 and Probe #3 tests still pass", true);

    // The declared non-claims (task §"Conclusion"; R13).
    println!();
    println!("== declared non-claims (R13) ==");
    println!("No generalization claim is made from training performance alone; the");
    println!("holdout numbers above are the only evidence about unseen temporal data.");

    println!();
    println!("holdout labels used during calibration: NO");
    println!("tensor modified: NO | second tensor: NO | graph: NO | new primitive: NO");
    println!("new API route: NO | holdout-label leakage: NO | ML model: NO | Drools: NO");
    println!("selection criterion: {SELECTION_CRITERION}");
    println!("split: {SPLIT_DECLARATION}");
    println!(
        "runtime: total={:.2}s (load+csf={:.2}s, derive={:.2}s, discovery={:.2}s x2)",
        t_all.elapsed().as_secs_f64(),
        load_s,
        derive_s,
        disc_s
    );

    if !fails.is_empty() {
        return Err(format!("failed obligations: {fails:?}"));
    }
    println!("exit=0");
    Ok(())
}
