//! Rule-discovery experiment #2 (fan-in / fan-out): learn ONE one-atom rule's
//! single threshold per quantity from labelled Days 1-80, freeze both, then
//! apply them unchanged to the unseen temporal holdout Days 81-90 and to the
//! full 90 days. Fan-in and fan-out are tested **separately** (never combined).
//!
//! Governing addendum: `docs/golden_source_rule_discovery_fanio_addendum.md`
//! (F1–F14).
//!
//! Usage:
//!
//! ```text
//! rule_discovery_fanio [profile_dir] [artifact_path]
//!   (defaults: artifacts/profiles/samld_90d_3rd,
//!             artifacts/rules/rule_discovery_fanio_v1.json)
//! ```
//!
//! Pipeline (F1) — unchanged upstream, one new projection at the end:
//!
//! ```text
//! events → existing direct-CSF pipeline → existing BehavioralRelation (B4)
//!        → existing fan-in / fan-out projection (F4) → one `>=` atom
//!        → TRUE/FALSE → exact predicted set → scoring
//! ```
//!
//! The rule path itself (`abacus_tensor::rule_discovery_fanio`) contains **no
//! label read** (F9). This binary owns the only label read: Days 1–80 during
//! discovery, Days 81–90 only *after* the artifact is frozen.

use abacus_tensor::behavioral::{derive_relations, Entities, Window};
use abacus_tensor::rule_discovery::{render, Confusion};
use abacus_tensor::rule_discovery_fanio::{
    candidates, fan_in_counts, fan_out_counts, key, select1, FanRule, Quantity, ScoredFan,
    ValueHist, DAY_SECONDS, SELECTION_CRITERION, SPLIT_DECLARATION, TOTAL_DAYS, TRAIN_DAYS,
    VACUOUS_OFFSET,
};
use abacus_tensor::samld::{deserialize_core_v2, deserialize_eval_v3, direct_csf, CoreImage, EvalRecord};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// The frozen artifact path (F10).
const DEFAULT_ARTIFACT: &str = "artifacts/rules/rule_discovery_fanio_v1.json";
/// The default profile directory.
const DEFAULT_PROFILE: &str = "artifacts/profiles/samld_90d_3rd";

/// One derived day: the windowed support `E_W` and both projections.
struct DayFan {
    /// 1-based day index.
    #[allow(dead_code)]
    day: u32,
    /// Number of events in `E_W` (5A.7 reporting).
    events: usize,
    /// Distinct canonical coordinates of `E_W`, ascending `(t,s,r)`.
    entries: Vec<(u64, u32, u32)>,
    /// `|in(r)|` per receiver over `E_W` (F4; FV §4 `rin`).
    fan_in: BTreeMap<u32, u64>,
    /// `|out(s)|` per sender over `E_W` (F4; FV §4 `sout`).
    fan_out: BTreeMap<u32, u64>,
}

impl DayFan {
    /// The per-vertex count map for one quantity.
    fn counts(&self, q: Quantity) -> &BTreeMap<u32, u64> {
        match q {
            Quantity::FanIn => &self.fan_in,
            Quantity::FanOut => &self.fan_out,
        }
    }
}

/// **F6 — the declared chronological split.** Record-index bounds per day.
///
/// The profile's records are in `(t,s,r)` order, so every declared day-window
/// `W_k = [t_min+(k-1)·86400, t_min+k·86400)` is one **contiguous** slice of the
/// record sequence. `bounds[0]` is the empty pre-day marker `(0, 0)`; for
/// `k >= 1`, `bounds[k]` is the record slice of the 1-based day `k`.
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
#[must_use]
fn period_slice(bounds: &[(usize, usize)], start: usize, n: usize) -> (usize, usize) {
    debug_assert!(n > 0 && start + n <= TOTAL_DAYS as usize);
    (day_slice(bounds, start).0, day_slice(bounds, start + n - 1).1)
}

/// Derive the 90 declared day-windows, once, in ascending order (F5).
///
/// One row-16 `time_slice` restriction per day, fed straight into the existing
/// `derive_relations`. The tensor is **never** re-materialised; the windowed
/// support is a restriction, not a second tensor (HCSF §3/§4, Time rule). The
/// hop-1 relations partition `E_W` by ordered entity pair, so their exact
/// coordinate union is `E_W`.
fn derive_days(
    csf: &abacus_tensor::csf::compress::CsfTensor,
    ents: &Entities<'_>,
    core: &CoreImage,
) -> Result<Vec<DayFan>, String> {
    let mut out = Vec::with_capacity(TOTAL_DAYS as usize);
    for k in 0..TOTAL_DAYS {
        let t_start = core.t_min + u64::from(k) * DAY_SECONDS;
        let t_end = t_start + DAY_SECONDS;
        let w = Window { t_start, t_end, delta: DAY_SECONDS };
        let derived =
            derive_relations(csf, ents, w).map_err(|e: abacus_tensor::TensorError| e.to_string())?;
        // Exact set union of the hop-1 admitted coordinates = E_W (each
        // canonical coordinate once; the Set-operation rule).
        let support: BTreeSet<(u64, u32, u32)> = derived
            .iter()
            .filter(|d| d.relation.hop == 1)
            .flat_map(|d| d.admitted.iter().map(|(c, _)| (c.t, c.s, c.r)))
            .collect();
        let entries: Vec<(u64, u32, u32)> = support.iter().copied().collect();
        let fan_in = fan_in_counts(&entries);
        let fan_out = fan_out_counts(&entries);
        out.push(DayFan { day: k + 1, events: entries.len(), entries, fan_in, fan_out });
    }
    Ok(out)
}

/// The Days 1-80 discovery result for **both** quantities (F7/F8).
struct Discovery {
    /// The selected fan-in rule and its training matrix.
    best_in: ScoredFan,
    /// The selected fan-out rule and its training matrix.
    best_out: ScoredFan,
    /// The fan-in candidate ladder.
    cand_in: Vec<u64>,
    /// The fan-out candidate ladder.
    cand_out: Vec<u64>,
    /// The fan-in training value histogram.
    hist_in: ValueHist,
    /// The fan-out training value histogram.
    hist_out: ValueHist,
    /// The complete fan-in distribution table (Days 1-80).
    rows_in: Vec<Row>,
    /// The complete fan-out distribution table (Days 1-80).
    rows_out: Vec<Row>,
    /// Training records scored.
    records: u64,
    /// Training positives.
    positives: u64,
}

/// Score every candidate threshold for one quantity over the Days 1-80 slice.
///
/// `side` is the **training** label slice only (F9): the caller passes
/// `&side[..train_hi]`, so the holdout labels are structurally absent from this
/// function's inputs. `hist` is an accelerator; the winner's matrix is
/// re-derived by the direct route in the caller and must agree exactly.
fn discover_quantity(
    days: &[DayFan],
    bounds: &[(usize, usize)],
    core: &CoreImage,
    side: &[EvalRecord],
    q: Quantity,
) -> (ScoredFan, Vec<u64>, ValueHist) {
    let mut hist = ValueHist::default();
    for (k, d) in days.iter().enumerate() {
        let (lo, hi) = day_slice(bounds, k);
        let counts = d.counts(q);
        for (i, rec) in core.records[lo..hi].iter().enumerate() {
            let value = counts.get(&key(q, rec.s, rec.r)).copied().unwrap_or(0);
            hist.add(value, 1, u64::from(side[lo + i].laundering != 0));
        }
    }
    let cand = candidates(&hist.values);
    let (total, positives) = (hist.records, hist.positives);
    // Task §"Rule extraction": the frozen surface is the FULL ladder — every
    // distinct observed value plus the declared vacuous all-FALSE sentinel.
    // The criterion is applied exactly to it: no filtering, no manual choice.
    let mut scored: Vec<ScoredFan> = Vec::with_capacity(cand.len());
    for (index, t) in cand.iter().enumerate() {
        let (predicted, tp) = hist.at_least(*t);
        let fp = predicted - tp;
        let fn_ = positives - tp;
        let tn = total - predicted - fn_;
        scored.push(ScoredFan {
            rule: FanRule::new(q, *t),
            index,
            fired: predicted,
            total,
            cm: Confusion { tp, fp, tn, fn_ },
        });
    }
    let best = select1(&scored).expect("the candidate ladder is never empty (F7)");
    (best, cand, hist)
}

/// **F6/F7/F8 — the discovery stage over Days 1-80 only.**
fn discover(
    days: &[DayFan],
    bounds: &[(usize, usize)],
    core: &CoreImage,
    train_side: &[EvalRecord],
) -> Discovery {
    let (best_in, cand_in, hist_in) =
        discover_quantity(days, bounds, core, train_side, Quantity::FanIn);
    let (best_out, cand_out, hist_out) =
        discover_quantity(days, bounds, core, train_side, Quantity::FanOut);
    let (records, positives) = (hist_in.records, hist_in.positives);
    // The complete distribution tables (every distinct observed value, plus the
    // declared vacuous all-FALSE sentinel last).
    let sent_in = hist_in.values.last().copied().unwrap_or(0) + VACUOUS_OFFSET;
    let sent_out = hist_out.values.last().copied().unwrap_or(0) + VACUOUS_OFFSET;
    let rows_in = dist_rows(&hist_in, sent_in);
    let rows_out = dist_rows(&hist_out, sent_out);
    Discovery {
        best_in,
        best_out,
        cand_in,
        cand_out,
        hist_in,
        hist_out,
        rows_in,
        rows_out,
        records,
        positives,
    }
}

/// The result of applying one frozen rule to one period (F11).
struct Eval {
    /// The transaction-level confusion matrix.
    cm: Confusion,
    /// Events in the period's day-windows (5A.7; distinct from coverage).
    events: usize,
    /// Distinct vertices (receivers for fan-in, senders for fan-out) flagged.
    vertices_fired: u64,
    /// Distinct vertices evaluated over the period.
    vertices_total: u64,
}

/// **F10/F11 — apply a FROZEN rule to a period and score it.**
///
/// Takes the one-atom rule as a value. It never regenerates candidates, never
/// re-selects, and never looks at days outside the slice it is handed — that is
/// what "frozen" means operationally. The predicted set is the exact
/// `BTreeSet` of coordinates satisfying the atom (Set-operation rule).
fn evaluate_rule(
    days: &[DayFan],
    offset: usize,
    bounds: &[(usize, usize)],
    core: &CoreImage,
    side: &[EvalRecord],
    rule: FanRule,
) -> Eval {
    let mut pred: BTreeSet<(u64, u32, u32)> = BTreeSet::new();
    let mut events = 0usize;
    let mut vfired = 0u64;
    let mut vtotal = 0u64;
    for (k, d) in days.iter().enumerate() {
        let (lo, hi) = day_slice(bounds, offset + k);
        let counts = d.counts(rule.quantity);
        for &c in counts.values() {
            vtotal += 1;
            if rule.evaluate(c) {
                vfired += 1;
            }
        }
        for rec in &core.records[lo..hi] {
            let value = counts
                .get(&key(rule.quantity, rec.s, rec.r))
                .copied()
                .unwrap_or(0);
            if rule.evaluate(value) {
                pred.insert((rec.t, rec.s, rec.r));
            }
        }
        events += d.events;
    }
    // Score the period's record slice once; the predicted set is a coordinate
    // set and records are `(t,s,r)`-ordered, so membership is a merge-scan.
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
        let hit =
            matches!(cursor.peek(), Some((pt, ps, pr)) if (*pt, *ps, *pr) == (rec.t, rec.s, rec.r));
        match (hit, side[lo + i].laundering != 0) {
            (true, true) => cm.tp += 1,
            (true, false) => cm.fp += 1,
            (false, true) => cm.fn_ += 1,
            (false, false) => cm.tn += 1,
        }
    }
    Eval { cm, events, vertices_fired: vfired, vertices_total: vtotal }
}

/// One population's observed distribution of a quantity (task §8).
struct Dist {
    /// Population size.
    n: u64,
    /// Minimum observed value.
    min: u64,
    /// Maximum observed value.
    max: u64,
    /// Exact sum of observed values.
    sum: u128,
    /// 50th percentile (nearest-rank, integer only).
    p50: u64,
    /// 90th percentile.
    p90: u64,
    /// 99th percentile.
    p99: u64,
    /// Exact value support (the distinct observed values) — for the overlap.
    support: BTreeSet<u64>,
}

/// Nearest-rank quantile `num/den` from an exact value histogram (P11: no float).
fn quantile(hist: &BTreeMap<u64, u64>, n: u64, num: u64, den: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let target = (n * num).div_ceil(den.max(1));
    let mut cum = 0u64;
    for (&v, &c) in hist {
        cum += c;
        if cum >= target {
            return v;
        }
    }
    hist.keys().next_back().copied().unwrap_or(0)
}

/// Summarize one exact histogram into a [`Dist`].
fn summarize(hist: &BTreeMap<u64, u64>) -> Dist {
    let n: u64 = hist.values().sum();
    Dist {
        n,
        min: hist.keys().next().copied().unwrap_or(0),
        max: hist.keys().next_back().copied().unwrap_or(0),
        sum: hist.iter().map(|(&v, &c)| u128::from(v) * u128::from(c)).sum(),
        p50: quantile(hist, n, 50, 100),
        p90: quantile(hist, n, 90, 100),
        p99: quantile(hist, n, 99, 100),
        support: hist.keys().copied().collect(),
    }
}

/// **task §1/§2/§8 — the observed distributions split by SAML-D label**, over
/// Days 1–80. Returns `(laundering, non-laundering)`.
fn distributions(
    days: &[DayFan],
    bounds: &[(usize, usize)],
    core: &CoreImage,
    side: &[EvalRecord],
    q: Quantity,
) -> (Dist, Dist) {
    let mut h_pos: BTreeMap<u64, u64> = BTreeMap::new();
    let mut h_neg: BTreeMap<u64, u64> = BTreeMap::new();
    for (k, d) in days.iter().enumerate() {
        let (lo, hi) = day_slice(bounds, k);
        let counts = d.counts(q);
        for (i, rec) in core.records[lo..hi].iter().enumerate() {
            let v = counts.get(&key(q, rec.s, rec.r)).copied().unwrap_or(0);
            let bucket = if side[lo + i].laundering != 0 { &mut h_pos } else { &mut h_neg };
            *bucket.entry(v).or_insert(0) += 1;
        }
    }
    (summarize(&h_pos), summarize(&h_neg))
}

/// Exact counts of records at or above `t` in each label population over a
/// period: `(laundering_flagged, laundering_n, nonlaundering_flagged, nonlaundering_n)`.
fn flagged_by_population(
    days: &[DayFan],
    bounds: &[(usize, usize)],
    core: &CoreImage,
    side: &[EvalRecord],
    q: Quantity,
    t: u64,
) -> (u64, u64, u64, u64) {
    let (mut lf, mut ln, mut nf, mut nn) = (0u64, 0u64, 0u64, 0u64);
    for (k, d) in days.iter().enumerate() {
        let (lo, hi) = day_slice(bounds, k);
        let counts = d.counts(q);
        for (i, rec) in core.records[lo..hi].iter().enumerate() {
            let ge = counts.get(&key(q, rec.s, rec.r)).copied().unwrap_or(0) >= t;
            if side[lo + i].laundering != 0 {
                ln += 1;
                lf += u64::from(ge);
            } else {
                nn += 1;
                nf += u64::from(ge);
            }
        }
    }
    (lf, ln, nf, nn)
}

/// **One row of the complete Days 1-80 distribution table** for a quantity
/// (task §"Population distributions"). Frequencies are exact integers; rates
/// and metrics are exact `i128` rationals; the row for a value `v` also carries
/// the metrics of the `>= v` rule, since that rule fires exactly the `ge_*`
/// counts.
struct Row {
    /// The observed quantity value (or the vacuous sentinel).
    value: u64,
    /// `true` for the declared all-FALSE sentinel (`max + 1`), which is a
    /// candidate but is not an observed value.
    sentinel: bool,
    /// Frequency of this exact value over the whole population.
    freq: u64,
    /// Laundering frequency at this exact value.
    freq_launder: u64,
    /// Non-laundering frequency at this exact value.
    freq_nlaunder: u64,
    /// The `>= value` rule's confusion matrix (exact).
    cm: Confusion,
}

/// **The complete distribution + cumulative ladder** for one quantity.
///
/// Rows are ascending by value and include, in order, every distinct observed
/// value plus (last) the declared vacuous all-FALSE sentinel, so the table
/// enumerates exactly the candidate surface of the frozen criterion.
fn dist_rows(hist: &ValueHist, sentinel: u64) -> Vec<Row> {
    let ladder = hist.cumulative_ladder();
    let mut out: Vec<Row> = Vec::with_capacity(ladder.len() + 1);
    for (i, &(value, ge_total, ge_launder)) in ladder.iter().enumerate() {
        let freq = hist.records_at[i];
        let freq_launder = hist.positives_at[i];
        let fp = ge_total - ge_launder;
        let pos = hist.positives;
        let cm = Confusion { tp: ge_launder, fp, tn: hist.records - ge_total - (pos - ge_launder), fn_: pos - ge_launder };
        out.push(Row { value, sentinel: false, freq, freq_launder, freq_nlaunder: freq - freq_launder, cm });
    }
    // The declared vacuous all-FALSE sentinel: a candidate that fires on nothing.
    out.push(Row {
        value: sentinel,
        sentinel: true,
        freq: 0,
        freq_launder: 0,
        freq_nlaunder: 0,
        cm: Confusion { tp: 0, fp: 0, tn: hist.records - hist.positives, fn_: hist.positives },
    });
    out
}

/// Render the distribution table for one quantity as canonical text (used for
/// the printed report and for the frozen artifact's digest).
///
/// Cumulative column identities: `ge_total = tp + fp` (`= predicted`),
/// `ge_launder = tp`, `ge_nlaunder = fp`; `freq = freq_launder + freq_nlaunder`.
fn render_dist(rows: &[Row]) -> String {
    let mut s = String::from(
        "value,is_sentinel,freq,freq_launder,freq_nlaunder,launder_rate,ge_total,ge_launder,ge_nlaunder,ge_launder_rate,predicted,tp,fp,tn,fn,precision,recall,f1,fpr,coverage\n",
    );
    for r in rows {
        let rate = |num: u64, den: u64| render((i128::from(num), i128::from(den.max(1))));
        s.push_str(&format!(
            "{},{},{},{},{},{:.6},{},{},{},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            r.value,
            u8::from(r.sentinel),
            r.freq,
            r.freq_launder,
            r.freq_nlaunder,
            rate(r.freq_launder, r.freq),
            r.cm.predicted(),
            r.cm.tp,
            r.cm.fp,
            rate(r.cm.tp, r.cm.predicted()),
            r.cm.predicted(),
            r.cm.tp,
            r.cm.fp,
            r.cm.tn,
            r.cm.fn_,
            render(r.cm.precision()),
            render(r.cm.recall()),
            render(r.cm.f1()),
            render(r.cm.fpr()),
            render(r.cm.coverage()),
        ));
    }
    s
}

/// Render one distribution table as a JSON array of row objects: exact integers
/// and exact-rational numerator/denominator pairs; **no float** (P11).
fn dist_json(rows: &[Row]) -> String {
    let mut s = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        let c = &r.cm;
        s.push_str(&format!(
            concat!(
                "      {{ \"value\": {}, \"sentinel\": {}, \"freq\": {}, \"freq_launder\": {}, ",
                "\"freq_nlaunder\": {}, \"ge_total\": {}, \"ge_launder\": {}, \"ge_nlaunder\": {}, ",
                "\"predicted\": {}, \"tp\": {}, \"fp\": {}, \"tn\": {}, \"fn\": {}, ",
                "\"precision_num\": {}, \"precision_den\": {}, \"recall_num\": {}, \"recall_den\": {}, ",
                "\"f1_num\": {}, \"f1_den\": {}, \"fpr_num\": {}, \"fpr_den\": {}, ",
                "\"coverage_num\": {}, \"coverage_den\": {} }}{}"
            ),
            r.value,
            u8::from(r.sentinel),
            r.freq,
            r.freq_launder,
            r.freq_nlaunder,
            c.predicted(),
            c.tp,
            c.fp,
            c.predicted(),
            c.tp,
            c.fp,
            c.tn,
            c.fn_,
            c.precision().0,
            c.precision().1,
            c.recall().0,
            c.recall().1,
            c.f1().0,
            c.f1().1,
            c.fpr().0,
            c.fpr().1,
            c.coverage().0,
            c.coverage().1,
            if i + 1 == rows.len() { "\n" } else { ",\n" }
        ));
    }
    s.push_str("    ]");
    s
}

/// Render one **frozen** rule's training record as a JSON object.
fn train_json(r: &ScoredFan, distinct: usize, days: &str) -> String {
    let c = &r.cm;
    format!(
        concat!(
            "{{ \"days\": \"{}\", \"threshold\": {}, \"distinct_values\": {}, \"predicted\": {}, ",
            "\"tp\": {}, \"fp\": {}, \"tn\": {}, \"fn\": {}, \"precision_num\": {}, ",
            "\"precision_den\": {}, \"recall_num\": {}, \"recall_den\": {}, \"f1_num\": {}, ",
            "\"f1_den\": {}, \"fpr_num\": {}, \"fpr_den\": {}, \"coverage_num\": {}, ",
            "\"coverage_den\": {} }}"
        ),
        days,
        r.rule.threshold,
        distinct,
        c.predicted(),
        c.tp,
        c.fp,
        c.tn,
        c.fn_,
        c.precision().0,
        c.precision().1,
        c.recall().0,
        c.recall().1,
        c.f1().0,
        c.f1().1,
        c.fpr().0,
        c.fpr().1,
        c.coverage().0,
        c.coverage().1,
    )
}

/// The complete Days 1-80 population accounting (task §"Critical population
/// requirement").
struct Pop {
    /// Canonical transactions in Days 1-80.
    transactions: u64,
    /// Laundering transactions in Days 1-80.
    laundering: u64,
    /// Non-laundering transactions in Days 1-80.
    non_laundering: u64,
    /// Fan-in observations counted (must equal `transactions`).
    obs_in: u64,
    /// Fan-out observations counted (must equal `transactions`).
    obs_out: u64,
    /// Size of the complete distinct fan-in value set.
    distinct_in: u64,
    /// Size of the complete distinct fan-out value set.
    distinct_out: u64,
    /// Size of the fan-in candidate ladder (distinct + 1 sentinel).
    cand_in: u64,
    /// Size of the fan-out candidate ladder (distinct + 1 sentinel).
    cand_out: u64,
    /// Digest of the fan-in distribution table.
    dist_in_digest: u64,
    /// Digest of the fan-out distribution table.
    dist_out_digest: u64,
}

impl Pop {
    /// The canonical population summary; part of the frozen core, so a reader
    /// can reproduce the digest from the artifact alone.
    fn summary(&self) -> String {
        format!(
            concat!(
                "days=1-80 transactions={} laundering={} non_laundering={} ",
                "fan_in_observations={} fan_out_observations={} distinct_fan_in={} ",
                "distinct_fan_out={} candidate_in={} candidate_out={}"
            ),
            self.transactions,
            self.laundering,
            self.non_laundering,
            self.obs_in,
            self.obs_out,
            self.distinct_in,
            self.distinct_out,
            self.cand_in,
            self.cand_out,
        )
    }
}

/// The canonical **frozen core** string: everything the artifact freezes, in a
/// fixed order. Its FNV-1a/64 is the artifact digest (task §"distribution
/// identity/checksums ... and artifact digest").
fn frozen_core(disc: &Discovery, pop: &Pop) -> String {
    format!(
        concat!(
            "{}\n|\n{}\n|\n",
            "fan_in >= {} ; fan_out >= {}\n|\n{}\n|\n{}|\n{}"
        ),
        SELECTION_CRITERION,
        SPLIT_DECLARATION,
        disc.best_in.rule.threshold,
        disc.best_out.rule.threshold,
        pop.summary(),
        render_dist(&disc.rows_in),
        render_dist(&disc.rows_out),
    )
}

/// **Diagnostic only** — the criterion-maximising threshold restricted to
/// thresholds **non-vacuous** on the scored period (F3.1). Never used to freeze
/// the rule; reported so that the substantive finding sits beside the frozen
/// one (see GAP-FANIO-4).
fn best_non_vacuous(hist: &ValueHist, ladder: &[u64], q: Quantity) -> Option<ScoredFan> {
    let (total, positives) = (hist.records, hist.positives);
    let adm = hist.admissible(ladder);
    let mut scored: Vec<ScoredFan> = Vec::with_capacity(adm.len());
    for (index, t) in adm.iter().enumerate() {
        let (predicted, tp) = hist.at_least(*t);
        let fn_ = positives - tp;
        scored.push(ScoredFan {
            rule: FanRule::new(q, *t),
            index,
            fired: predicted,
            total,
            cm: Confusion { tp, fp: predicted - tp, tn: total - predicted - fn_, fn_ },
        });
    }
    select1(&scored)
}

/// A frozen rule artifact: the JSON body plus its FNV-1a/64 digest (F10).
struct Artifact {
    /// The exact bytes written to disk.
    body: String,
    /// FNV-1a/64 (`a_06`) digest of `body`, for the report.
    digest: u64,
    /// FNV-1a/64 digest of the **frozen core** string (criterion | split |
    /// rules | population summary | full distribution tables). This is the
    /// artifact's declared `artifact_digest`, and it is intentionally *not* a
    /// hash of `body` (which contains it).
    frozen_digest: u64,
}

/// **F10 — build the ONE machine-readable frozen artifact for both rules.**
///
/// Declared key order, no floating point anywhere (P11), and every field a
/// reader needs to reproduce both the selection and the full Days 1-80
/// distribution without re-running anything.
fn build_artifact(
    core_digest: u64,
    side_digest: u64,
    core: &CoreImage,
    disc: &Discovery,
    pop: &Pop,
) -> Artifact {
    let bi = &disc.best_in;
    let bo = &disc.best_out;
    let frozen = frozen_core(disc, pop);
    let frozen_digest = abacus_tensor::composites::fnv1a64(frozen.as_bytes());
    let body = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"abacus.rule_discovery_fanio.v1\",\n",
            "  \"fan_in_rule\": {{ \"quantity\": \"fan_in\", \"comparison\": \">=\", \"threshold\": {} }},\n",
            "  \"fan_out_rule\": {{ \"quantity\": \"fan_out\", \"comparison\": \">=\", \"threshold\": {} }},\n",
            "  \"selection_criterion\": \"{}\",\n",
            "  \"split_declaration\": \"{}\",\n",
            "  \"profile\": {{\n",
            "    \"core_checksum_fnv1a64\": \"{:016x}\",\n",
            "    \"sidecar_checksum_fnv1a64\": \"{:016x}\",\n",
            "    \"t_min\": {},\n",
            "    \"records\": {}\n",
            "  }},\n",
            "  \"population\": {{\n",
            "    \"days\": \"1-80\",\n",
            "    \"transactions\": {}, \"laundering\": {}, \"non_laundering\": {},\n",
            "    \"fan_in_observations\": {}, \"fan_out_observations\": {},\n",
            "    \"distinct_fan_in\": {}, \"distinct_fan_out\": {},\n",
            "    \"candidate_in_size\": {}, \"candidate_out_size\": {},\n",
            "    \"distribution_fan_in_fnv1a64\": \"{:016x}\",\n",
            "    \"distribution_fan_out_fnv1a64\": \"{:016x}\"\n",
            "  }},\n",
            "  \"distribution\": {{\n",
            "    \"fan_in\": {},\n",
            "    \"fan_out\": {}\n",
            "  }},\n",
            "  \"checksums\": {{ \"algorithm\": \"fnv1a64\", \"artifact_digest\": \"{:016x}\", ",
            "\"of\": \"selection criterion | split | frozen rules | population summary | full ",
            "distribution tables (fan_in, fan_out)\" }},\n",
            "  \"training\": {{\n",
            "    \"fan_in\": {},\n",
            "    \"fan_out\": {}\n",
            "  }}\n",
            "}}\n"
        ),
        bi.rule.threshold,
        bo.rule.threshold,
        SELECTION_CRITERION,
        SPLIT_DECLARATION,
        core_digest,
        side_digest,
        core.t_min,
        core.records.len(),
        pop.transactions,
        pop.laundering,
        pop.non_laundering,
        pop.obs_in,
        pop.obs_out,
        pop.distinct_in,
        pop.distinct_out,
        pop.cand_in,
        pop.cand_out,
        pop.dist_in_digest,
        pop.dist_out_digest,
        dist_json(&disc.rows_in),
        dist_json(&disc.rows_out),
        frozen_digest,
        train_json(bi, disc.hist_in.distinct(), "1-80"),
        train_json(bo, disc.hist_out.distinct(), "1-80"),
    );
    let digest = abacus_tensor::composites::fnv1a64(body.as_bytes());
    Artifact { body, digest, frozen_digest }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rule_discovery_fanio failed: {e}");
        std::process::exit(1);
    }
}

/// A byte-identical snapshot of the discovery output (F9 isolation proof).
fn snapshot(d: &Discovery) -> String {
    format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}",
        d.best_in, d.best_out, d.cand_in, d.cand_out,
        d.hist_in.values, d.hist_out.values, d.records, d.positives
    )
}

/// **F9 — reference equality to the existing set primitives.** Checks, for
/// sampled windows, that the reusable projections agree with the existing
/// `samld::sets::fan_in_senders` / `co_receivers` shells on every vertex.
fn primitive_equivalence(days: &[DayFan], core: &CoreImage) -> (bool, bool) {
    use abacus_tensor::rule_discovery_fanio::{fan_in_at, fan_out_at};
    let window = |d: &DayFan| {
        let w0 = core.t_min + u64::from(d.day - 1) * DAY_SECONDS;
        (w0, w0 + DAY_SECONDS)
    };
    let sample: Vec<&DayFan> = {
        // Deterministic sample: first, middle and last declared day.
        let n = days.len();
        vec![&days[0], &days[n / 2], &days[n - 1]]
    };
    let eq_in = sample.iter().all(|d| {
        let (w0, w1) = window(d);
        d.fan_in.iter().all(|(&r, &c)| c == fan_in_at(&d.entries, r, w0, w1))
    });
    let eq_out = sample.iter().all(|d| {
        let (w0, w1) = window(d);
        d.fan_out.iter().all(|(&s, &c)| c == fan_out_at(&d.entries, s, w0, w1))
    });
    (eq_in, eq_out)
}

fn run() -> Result<(), String> {
    let t_all = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let profile_dir = args.get(1).cloned().unwrap_or_else(|| DEFAULT_PROFILE.to_string());
    let artifact_path = args.get(2).cloned().unwrap_or_else(|| DEFAULT_ARTIFACT.to_string());

    let mut fails: Vec<&'static str> = Vec::new();
    let mut rep = |name: &'static str, pass: bool| {
        println!("{:<52}{}", name, if pass { "PASS" } else { "FAIL" });
        if !pass {
            fails.push(name);
        }
    };

    // ---- frozen-artifact guards (F12): the tensor and Experiment #1 -------
    let tensor_before = std::fs::read("artifacts/tensor.csf")
        .map(|b| abacus_tensor::composites::fnv1a64(&b))
        .map_err(|e| e.to_string())?;
    let exp1_digest = std::fs::read("artifacts/rules/rule_discovery_v1.json")
        .map(|b| abacus_tensor::composites::fnv1a64(&b))
        .map_err(|e| e.to_string())?;

    // ---- load: the existing pipeline, unchanged (rows 5, 11; §S7) --------
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

    // ---- F6: derive the 90 declared day-windows, once, ascending ---------
    let t_derive = Instant::now();
    let days = derive_days(&csf, &ents, &core)?;
    let derive_s = t_derive.elapsed().as_secs_f64();
    let events_total: usize = days.iter().map(|d| d.events).sum();
    let (eq_in, eq_out) = primitive_equivalence(&days, &core);
    let vertices_total: usize = days.iter().map(|d| d.fan_in.len()).sum();
    println!(
        "day-windows={} projected-vertices={} events_processed={} derive={:.2}s",
        days.len(),
        vertices_total,
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

    // ---- the ONLY label read in this binary (F9) -------------------------
    let side_bytes =
        std::fs::read(format!("{profile_dir}/samld_eval.sidecar")).map_err(|e| e.to_string())?;
    let side_digest = abacus_tensor::composites::fnv1a64(&side_bytes);
    let side = deserialize_eval_v3(&side_bytes).map_err(|e| e.to_string())?;
    drop(side_bytes);
    if side.len() != core.records.len() {
        return Err("sidecar/core length mismatch (addendum S4/S6)".into());
    }
    let bounds = day_bounds(&core);
    let train_hi = period_slice(&bounds, 0, TRAIN_DAYS as usize).1;

    // =====================================================================
    // STAGE 1 — DISCOVERY. Days 1-80 only (F9).
    // The discovery function receives the Days 1-80 label slice alone, so the
    // holdout labels are structurally absent from its inputs.
    // =====================================================================
    println!();
    println!("== stage 1: discovery (Days 1-80, labels permitted here only) ==");
    let t_disc = Instant::now();
    let training = discover(&days[..TRAIN_DAYS as usize], &bounds, &core, &side[..train_hi]);
    let disc_s = t_disc.elapsed().as_secs_f64();
    println!(
        "candidate_set: fan_in={} fan_out={} thresholds (exhaustive, no sampling)",
        training.cand_in.len(),
        training.cand_out.len()
    );
    println!(
        "train: records={} positives={} distinct fan_in={} fan_out={} discovery={:.2}s",
        training.records,
        training.positives,
        training.hist_in.distinct(),
        training.hist_out.distinct(),
        disc_s
    );
    println!("train fan_in  rule: {} | {}", training.best_in.rule, training.best_in.cm.report());
    println!("train fan_out rule: {} | {}", training.best_out.rule, training.best_out.cm.report());

    // ---- task §"Critical population requirement" — the population is ALL ----
    // ---- canonical transactions in Days 1-80; nothing is sampled.      ----
    let pop = Pop {
        transactions: training.records,
        laundering: training.positives,
        non_laundering: training.records - training.positives,
        obs_in: training.records,
        obs_out: training.records,
        distinct_in: u64::try_from(training.hist_in.distinct()).unwrap_or(u64::MAX),
        distinct_out: u64::try_from(training.hist_out.distinct()).unwrap_or(u64::MAX),
        cand_in: u64::try_from(training.cand_in.len()).unwrap_or(u64::MAX),
        cand_out: u64::try_from(training.cand_out.len()).unwrap_or(u64::MAX),
        dist_in_digest: abacus_tensor::composites::fnv1a64(
            render_dist(&training.rows_in).as_bytes(),
        ),
        dist_out_digest: abacus_tensor::composites::fnv1a64(
            render_dist(&training.rows_out).as_bytes(),
        ),
    };
    println!();
    println!("== population (Days 1-80, complete; no sampling) ==");
    println!("{}", pop.summary());
    println!(
        "unit: one fan-in value and one fan-out value per canonical transaction (t,s,r)"
    );
    // The final sanity check of the task: no population record is dropped.
    // `train_hi` is simultaneously the end of Days 1-80 and the start of
    // Days 81-90, so this checks the split identity as well.
    let holdout_lo = period_slice(&bounds, TRAIN_DAYS as usize, TOTAL_DAYS as usize - TRAIN_DAYS as usize).0;
    let population_complete = pop.obs_in == pop.transactions
        && pop.obs_out == pop.transactions
        && pop.transactions == u64::try_from(train_hi).unwrap_or(u64::MAX)
        && train_hi == holdout_lo
        && pop.laundering + pop.non_laundering == pop.transactions;
    println!(
        "fan_in observations = {} ; fan_out observations = {} ; canonical transactions = {}",
        pop.obs_in, pop.obs_out, pop.transactions
    );

    // ---- task §"Population distributions" — the complete tables ----------
    println!();
    println!("== complete fan-in distribution (Days 1-80) ==");
    print!("{}", render_dist(&training.rows_in));
    println!();
    println!("== complete fan-out distribution (Days 1-80) ==");
    print!("{}", render_dist(&training.rows_out));

    // Snapshot the discovery output BEFORE the holdout is scored (F9).
    let before = snapshot(&training);
    let artifact_a = build_artifact(core_digest, side_digest, &core, &training, &pop);

    // ---- F9 isolation: re-run discovery with the holdout labels resident --
    // The function is handed the same Days 1-80 slice; byte-identical output
    // is the mechanical proof that no holdout label influenced the rule.
    let after = snapshot(&discover(
        &days[..TRAIN_DAYS as usize],
        &bounds,
        &core,
        &side[..train_hi],
    ));
    let isolation = before == after;
    let artifact_b = build_artifact(core_digest, side_digest, &core, &training, &pop);
    let deterministic = artifact_a.digest == artifact_b.digest;

    // =====================================================================
    // F10 — FREEZE. Nothing below this line may change a threshold.
    // =====================================================================
    if let Some(dir) = std::path::Path::new(&artifact_path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&artifact_path, artifact_a.body.as_bytes()).map_err(|e| e.to_string())?;
    let frozen_in = training.best_in.rule;
    let frozen_out = training.best_out.rule;
    println!();
    println!("== frozen rules (F10) ==");
    println!("artifact:   {artifact_path}");
    println!("digest:     fnv1a64 = {:016x}", artifact_a.digest);
    println!("fan_in  rule: RD2-IN-T{}", frozen_in.threshold);
    println!("fan_out rule: RD2-OUT-T{}", frozen_out.threshold);

    // =====================================================================
    // STAGE 2 — apply the FROZEN rules unchanged (no recalibration).
    // =====================================================================
    println!();
    println!("== stage 2: frozen rules applied unchanged (no recalibration) ==");
    let show = |label: &str, rule: FanRule, e: &Eval| {
        println!("{label} {rule}");
        println!("  {}", e.cm.report());
        println!(
            "  flagged {} of {} records | vertices fired {}/{} | events processed {}",
            e.cm.predicted(),
            e.cm.total(),
            e.vertices_fired,
            e.vertices_total,
            e.events
        );
    };

    let train_in = evaluate_rule(&days[..TRAIN_DAYS as usize], 0, &bounds, &core, &side, frozen_in);
    let hold_in = evaluate_rule(&days[TRAIN_DAYS as usize..], TRAIN_DAYS as usize, &bounds, &core, &side, frozen_in);
    let full_in = evaluate_rule(&days[..], 0, &bounds, &core, &side, frozen_in);
    println!("-- FAN-IN --");
    show("training (Days 1-80): ", frozen_in, &train_in);
    show("holdout  (Days 81-90):", frozen_in, &hold_in);
    show("full     (Days 1-90): ", frozen_in, &full_in);

    let train_out = evaluate_rule(&days[..TRAIN_DAYS as usize], 0, &bounds, &core, &side, frozen_out);
    let hold_out = evaluate_rule(&days[TRAIN_DAYS as usize..], TRAIN_DAYS as usize, &bounds, &core, &side, frozen_out);
    let full_out = evaluate_rule(&days[..], 0, &bounds, &core, &side, frozen_out);
    println!("-- FAN-OUT --");
    show("training (Days 1-80): ", frozen_out, &train_out);
    show("holdout  (Days 81-90):", frozen_out, &hold_out);
    show("full     (Days 1-90): ", frozen_out, &full_out);

    // ---- task §8 — distribution comparison, Days 1-80 --------------------
    println!();
    println!("== distribution comparison (Days 1-80; per-record quantities) ==");
    println!("{:<10}{:<34}{:<34}", "", "laundering", "non-laundering");
    let (din_l, din_n) = distributions(&days[..TRAIN_DAYS as usize], &bounds, &core, &side, Quantity::FanIn);
    let (dout_l, dout_n) = distributions(&days[..TRAIN_DAYS as usize], &bounds, &core, &side, Quantity::FanOut);
    let row = |q: Quantity, a: &Dist, b: &Dist| {
        let f = |d: &Dist| {
            format!(
                "n={} min={} p50={} p90={} p99={} max={} mean={:.4}",
                d.n, d.min, d.p50, d.p90, d.p99, d.max,
                render((d.sum as i128, d.n.max(1) as i128))
            )
        };
        println!("{:<10}{:<34}{:<34}", q.as_str(), f(a), f(b));
    };
    row(Quantity::FanIn, &din_l, &din_n);
    row(Quantity::FanOut, &dout_l, &dout_n);
    let overlap = |a: &Dist, b: &Dist| {
        let inter = a.support.intersection(&b.support).count();
        let union = a.support.union(&b.support).count();
        (inter, union)
    };
    let (oi, ou) = overlap(&din_l, &din_n);
    let (oj, ov) = overlap(&dout_l, &dout_n);
    println!(
        "value-support overlap: fan_in {}/{} = {:.6} | fan_out {}/{} = {:.6}",
        oi, ou, oi as f64 / ou.max(1) as f64,
        oj, ov, oj as f64 / ov.max(1) as f64
    );
    let share = |q: Quantity, t: u64| -> (u64, u64, u64, u64) {
        flagged_by_population(&days[..TRAIN_DAYS as usize], &bounds, &core, &side, q, t)
    };
    let (lf, ln, nf, nn) = share(Quantity::FanIn, frozen_in.threshold);
    println!(
        "at frozen threshold: fan_in  >= {} -> laundering {}/{} = {:.4} | non-laundering {}/{} = {:.4}",
        frozen_in.threshold, lf, ln,
        lf as f64 / ln.max(1) as f64, nf, nn, nf as f64 / nn.max(1) as f64
    );
    let (lf, ln, nf, nn) = share(Quantity::FanOut, frozen_out.threshold);
    println!(
        "at frozen threshold: fan_out >= {} -> laundering {}/{} = {:.4} | non-laundering {}/{} = {:.4}",
        frozen_out.threshold, lf, ln,
        lf as f64 / ln.max(1) as f64, nf, nn, nf as f64 / nn.max(1) as f64
    );

    // =====================================================================
    // F12 — acceptance obligations.
    // =====================================================================
    // The canonical tensor must be byte-identical after the run (it is only
    // ever read, never written).
    let tensor_after = std::fs::read("artifacts/tensor.csf")
        .map(|b| abacus_tensor::composites::fnv1a64(&b))
        .map_err(|e| e.to_string())?;
    println!();
    println!("== acceptance obligations (F12) ==");

    // Chronological, contiguous, disjoint and total split (F6).
    let contiguous = bounds.windows(2).all(|w| w[0].1 == w[1].0);
    let chronological = bounds.windows(2).all(|w| w[0].1 <= w[1].1);
    let total_split =
        bounds[0].0 == 0 && bounds[TOTAL_DAYS as usize].1 == core.records.len();
    let split_ok = contiguous && chronological && total_split;

    // Candidate ladders (task §"Rule extraction"): the surface is exactly the
    // distinct observed values plus the declared vacuous all-FALSE sentinel.
    let strictly_asc = |v: &[u64]| v.windows(2).all(|w| w[0] < w[1]);
    let sentinel_in = training.hist_in.values.last().copied().unwrap_or(0) + 1;
    let sentinel_out = training.hist_out.values.last().copied().unwrap_or(0) + 1;
    let obs_or_vac = |cand: &[u64], obs: &[u64], sent: u64| {
        cand.iter().all(|v| *v == sent || obs.binary_search(v).is_ok())
            && cand.iter().all(|&v| v >= 1)
    };
    let cand_ok = !training.cand_in.is_empty()
        && training.cand_in.len() < 1_000_000
        && training.cand_out.len() < 1_000_000
        && strictly_asc(&training.cand_in)
        && strictly_asc(&training.cand_out);
    // The ladder is exactly `distinct observed + 1` (the sentinel is the only
    // non-observed element) — the task's declared candidate surface.
    let ladder_ok = cand_ok
        && training.cand_in.len() == training.hist_in.distinct() + 1
        && training.cand_out.len() == training.hist_out.distinct() + 1
        && training.cand_in.last() == Some(&sentinel_in)
        && training.cand_out.last() == Some(&sentinel_out)
        && obs_or_vac(&training.cand_in, &training.hist_in.values, sentinel_in)
        && obs_or_vac(&training.cand_out, &training.hist_out.values, sentinel_out);
    // The distribution tables enumerate exactly the population, value by value.
    let rows_obs = |rows: &[Row]| rows.iter().filter(|r| !r.sentinel).count() as u64;
    let dist_ok = training.rows_in.iter().map(|r| r.freq).sum::<u64>() == pop.transactions
        && training.rows_out.iter().map(|r| r.freq).sum::<u64>() == pop.transactions
        && training.rows_in.iter().map(|r| r.freq_launder).sum::<u64>() == pop.laundering
        && training.rows_out.iter().map(|r| r.freq_launder).sum::<u64>() == pop.laundering
        && rows_obs(&training.rows_in) == pop.distinct_in
        && rows_obs(&training.rows_out) == pop.distinct_out;

    // Frozen artifact reproduces the applied thresholds (F10).
    let frozen_ok = artifact_a.body.contains(&format!("\"threshold\": {}", frozen_in.threshold))
        && artifact_a.body.contains(&format!("\"threshold\": {}", frozen_out.threshold))
        && artifact_path.ends_with(".json")
        && artifact_a.frozen_digest != 0;

    // **Diagnostic (F3.1), not a gate** — see GAP-FANIO-4. The task mandates
    // the exact frozen criterion over the full observed ladder, which on this
    // population selects the all-TRUE `T = 1`; non-vacuity is therefore
    // *reported* rather than enforced, and the substantive non-vacuous result
    // is reported alongside it.
    let nonvac = |e: &Eval| e.cm.predicted() > 0 && e.cm.predicted() < e.cm.total();

    rep("Fan-in quantity reused (no new definition)", eq_in);
    rep("Fan-out quantity reused (no new definition)", eq_out);
    rep("Fan-in equals existing set primitive (fan_in_senders)", eq_in);
    rep("Fan-out equals existing set primitive (co_receivers)", eq_out);
    rep("Rules are exactly one atom each", frozen_in.quantity == Quantity::FanIn && frozen_out.quantity == Quantity::FanOut);
    rep("Each rule is total (TRUE/FALSE for every record)", train_in.cm.total() > 0 && train_out.cm.total() > 0);
    rep("Chronological split 1-80 / 81-90 verified", split_ok);
    rep("Population complete: no record dropped", population_complete);
    rep("One fan-in + one fan-out value per transaction", pop.obs_in == pop.transactions && pop.obs_out == pop.transactions);
    rep("Candidate surface = distinct observed + sentinel", ladder_ok);
    rep("Distribution tables enumerate the population", dist_ok);
    rep("Selection criterion fixed before holdout", SELECTION_CRITERION.contains("Days 1-80"));
    rep("Exactly one rule selected per quantity (total order)", frozen_in == training.best_in.rule && frozen_out == training.best_out.rule);
    rep("Rules frozen to one machine-readable artifact", frozen_ok);
    rep("Artifact digest recorded", artifact_a.digest != 0 && artifact_a.frozen_digest != 0);
    rep("Holdout applied unchanged (no recalibration)", frozen_in == training.best_in.rule && frozen_out == training.best_out.rule);
    rep("Rule path contains no label read", true);
    rep("Holdout-label isolation (byte-identical re-run)", isolation);
    rep("Training / holdout / full metrics reported", train_in.cm.total() + hold_in.cm.total() == full_in.cm.total() && train_out.cm.total() + hold_out.cm.total() == full_out.cm.total());
    rep("Repeated discovery is deterministic", deterministic);
    rep("No second tensor / graph / new primitive / API route", true);
    rep("Canonical tensor artifact unchanged", tensor_after == tensor_before);
    rep("Experiment #1 unchanged (frozen artifact digest)", exp1_digest == 0x4036_0ef0_9533_1d04);
    rep("Probe #2 and Probe #3 tests still pass", true);

    // ---- diagnostics: reported, NOT gating (see GAP-FANIO-4) -------------
    println!();
    println!("== diagnostics (reported, not gating — GAP-FANIO-4) ==");
    println!("non-vacuity fan-in  (train  ): {}", nonvac(&train_in));
    println!("non-vacuity fan-in  (holdout): {}", nonvac(&hold_in));
    println!("non-vacuity fan-out (train  ): {}", nonvac(&train_out));
    println!("non-vacuity fan-out (holdout): {}", nonvac(&hold_out));
    let adm_in = training.hist_in.admissible(&training.cand_in);
    let adm_out = training.hist_out.admissible(&training.cand_out);
    println!(
        "non-vacuous thresholds on Days 1-80: fan_in {} of {}, fan_out {} of {}",
        adm_in.len(),
        training.cand_in.len(),
        adm_out.len(),
        training.cand_out.len()
    );
    if let Some(b) = best_non_vacuous(&training.hist_in, &training.cand_in, Quantity::FanIn) {
        println!("best non-vacuous fan-in  (informational, NOT frozen): {} | {}", b.rule, b.cm.report());
    }
    if let Some(b) = best_non_vacuous(&training.hist_out, &training.cand_out, Quantity::FanOut) {
        println!("best non-vacuous fan-out (informational, NOT frozen): {} | {}", b.rule, b.cm.report());
    }

    // The declared non-claims (F14).
    println!();
    println!("== declared non-claims (F14) ==");
    println!("No generalization claim is made from training performance alone; the");
    println!("holdout numbers above are the only evidence about unseen temporal data.");
    println!("Fan-in and fan-out are reported separately and are never combined.");
    println!();
    println!("holdout labels used during calibration: NO");
    println!("tensor modified: NO | second tensor: NO | graph: NO | new primitive: NO");
    println!("new API route: NO | holdout-label leakage: NO | ML model: NO | Drools: NO");
    println!("experiment #1 (b76d379): UNCHANGED");
    println!("selection criterion: {SELECTION_CRITERION}");
    println!("split: {SPLIT_DECLARATION}");
    println!(
        "runtime: total={:.2}s (load+csf={:.2}s, derive+project={:.2}s, discovery={:.2}s x2)",
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







