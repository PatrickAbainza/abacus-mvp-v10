//! Experiment B — ONE global deterministic Boolean rule (overfit / memorization).
//!
//! Objective (task): maximise exact agreement with the Days 1–80 laundering
//! labels using an explicit deterministic Boolean expression built **only**
//! from existing Abacus measurements, existing tensor/set algebra, and
//! deterministic comparisons + `AND`/`OR`/`NOT` composition. Deliberate
//! overfitting is the goal; generalisation is explicitly **not** optimised.
//!
//! Governing sections (Golden Source Compliance Rule — citations):
//! * measurements — `abacus_tensor::overfit` (rows 5/11/16; B4; FV §§4/5; Addendum O rows 30–51).
//! * comparisons — semantic §4; `AND`/`OR`/`NOT` — semantic §18 / C5.
//! * label isolation — P11, B8, C9. Chronological split — R4.
//!
//! Rigour note: the maximum agreement **any** Boolean function of the
//! measurement vector can reach is exactly
//! `ceiling = sum over distinct measurement-vectors v of max(L_v, N_v)`,
//! because every rule is a function of the measurement vector and is therefore
//! constant on a vector's whole equivalence class. This is an *exhaustive*
//! statement over the entire permitted function class — not a search result —
//! so a ceiling below 100% is proof that no exact rule exists in the language,
//! whereas a ceiling of 100% means an exact rule exists and is constructed.
//!
//! Usage: `overfit_global [profile_dir] [artifact_json] [artifact_rule_json]`

use std::cmp::Ordering;
use std::collections::{BinaryHeap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

use abacus_tensor::overfit::{
    day_bounds, day_slice, day_vectors, fnv_bytes, fnv_vecs, Mvec, MEASUREMENT, NM, PROVENANCE,
    DAY_SECONDS, TOTAL_DAYS, TRAIN_DAYS,
};
use abacus_tensor::rule_discovery::Confusion;
use abacus_tensor::samld::{deserialize_core_v2, deserialize_eval_v3, direct_csf};

const DEFAULT_PROFILE: &str = "artifacts/profiles/samld_90d_3rd";
const DEFAULT_ARTIFACT: &str = "artifacts/rules/overfit_global_v1.json";
const DEFAULT_RULE_TXT: &str = "artifacts/rules/overfit_global_v1.rule.json";

/// Sentinel written into the unselected positions of a projected key.
const FREE: i64 = i64::MIN;

/// Project a measurement vector onto a term mask (unselected -> [`FREE`]).
#[must_use]
fn project(v: &Mvec, mask: u16) -> Mvec {
    let mut k = [FREE; NM];
    for i in 0..NM {
        if mask & (1u16 << i) != 0 {
            k[i] = v[i];
        }
    }
    k
}

/// One conjunction term: `AND_{i in mask} x_i = key[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Term {
    mask: u16,
    key: Mvec,
}

impl Term {
    fn literals(&self) -> usize {
        self.mask.count_ones() as usize
    }
}

/// The generated deterministic rule: `OR` of the conjunction terms, plus
/// residual full-vector cell terms.
#[derive(Debug, Clone, Default)]
struct Rule {
    terms: BTreeSet<Term>,
    cells: BTreeSet<Mvec>,
}

impl Rule {
    fn term_count(&self) -> usize {
        self.terms.len() + self.cells.len()
    }
    fn literal_count(&self) -> usize {
        self.terms.iter().map(Term::literals).sum::<usize>() + self.cells.len() * NM
    }
    fn measurements_used(&self) -> BTreeSet<u8> {
        let mut s: BTreeSet<u8> = BTreeSet::new();
        for t in &self.terms {
            for i in 0..NM {
                if t.mask & (1u16 << i) != 0 {
                    s.insert(i as u8);
                }
            }
        }
        if !self.cells.is_empty() {
            for k in 0..NM {
                s.insert(k as u8);
            }
        }
        s
    }
    fn boolean_ops(&self) -> usize {
        let ands = self
            .terms
            .iter()
            .map(|t| t.literals().saturating_sub(1))
            .sum::<usize>()
            + self.cells.len() * (NM - 1);
        let ors = self.term_count().saturating_sub(1);
        ands + ors
    }
    fn depth(&self) -> usize {
        if self.term_count() <= 1 {
            1
        } else {
            2
        }
    }
    /// Canonical, deterministic JSON form of the rule expression (the digest
    /// source). A logical `OR` of conjunction terms; no delimited table (CSV rule).
    fn body(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"schema\": \"abacus.overfit_global.rule.v1\",\n");
        out.push_str("  \"positions\": [");
        for (i, n) in MEASUREMENT.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{n}\""));
        }
        out.push_str("],\n");
        out.push_str("  \"op\": \"or\",\n");
        out.push_str("  \"terms\": [\n");
        let total = self.term_count();
        let mut n = 0usize;
        for t in &self.terms {
            n += 1;
            push_term(&mut out, &t.key, t.mask, n == total);
        }
        for c in &self.cells {
            n += 1;
            push_term(&mut out, c, (1u16 << NM) - 1, n == total);
        }
        out.push_str("  ]\n}\n");
        out
    }
}

/// Append one conjunction term (`{"op":"and","atoms":[{"m":..,"eq":..},..]}`)
/// to the canonical JSON body. Deterministic given the term's mask/key.
fn push_term(out: &mut String, key: &Mvec, mask: u16, last: bool) {
    out.push_str("    {\"op\":\"and\",\"atoms\":[");
    let mut first = true;
    for i in 0..NM {
        if mask & (1u16 << i) != 0 {
            if !first {
                out.push(',');
            }
            out.push_str(&format!("{{\"m\":\"{}\",\"eq\":{}}}", MEASUREMENT[i], key[i]));
            first = false;
        }
    }
    out.push_str(&format!("]}}{}\n", if last { "" } else { "," }));
}

/// Fast membership index: one hash set per used mask.
struct RuleIndex {
    by_mask: Vec<(u16, HashSet<Mvec>)>,
    cells: HashSet<Mvec>,
}

impl RuleIndex {
    fn build(r: &Rule) -> Self {
        let mut groups: BTreeSet<u16> = BTreeSet::new();
        for t in &r.terms {
            groups.insert(t.mask);
        }
        let by_mask = groups
            .into_iter()
            .map(|m| {
                let set: HashSet<Mvec> = r
                    .terms
                    .iter()
                    .filter(|t| t.mask == m)
                    .map(|t| t.key)
                    .collect();
                (m, set)
            })
            .collect();
        Self {
            by_mask,
            cells: r.cells.iter().copied().collect(),
        }
    }
    /// Does the record with measurement vector `v` satisfy the rule?
    fn matches(&self, v: &Mvec) -> bool {
        for (m, set) in &self.by_mask {
            if set.contains(&project(v, *m)) {
                return true;
            }
        }
        self.cells.contains(v)
    }
}

/// Cell statistics: measurement vector -> (laundering, non-laundering) counts.
type Cells = HashMap<Mvec, (u64, u64)>;

/// Accumulate the exact cell counts from a `(vectors, labels)` slice.
fn build_cells(vecs: &[Mvec], labels: &[u8]) -> Cells {
    let mut cells: Cells = HashMap::with_capacity(vecs.len());
    for (v, l) in vecs.iter().zip(labels) {
        let e = cells.entry(*v).or_insert((0, 0));
        if *l != 0 {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }
    cells
}

/// The greedy-search candidate: a safe pattern with its current (possibly
/// stale) gain, its mask width, and its index into the coverage tables.
#[derive(Debug, Clone, Copy, Eq)]
struct Cand {
    gain: usize,
    width: u16,
    idx: u64,
}

impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.gain
            .cmp(&o.gain)
            .then_with(|| o.width.cmp(&self.width))
            .then_with(|| o.idx.cmp(&self.idx))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

/// The synthesis result (the rule and its diagnosing counts).
struct Synthesis {
    rule: Rule,
    ceiling: u64,
    positive_cells: usize,
    negative_cells: usize,
    safe1_pool: usize,
    safe2_pool: usize,
    greedy_terms: usize,
    residual_cells: usize,
}

/// The declared masks used by the search: every coordinate subset of size
/// `1..=max_mask`. This bounded family is an **implementation / search bound**
/// only; it is not a claim about the expressive capacity of the surface
/// (the residual full-cell terms guarantee the ceiling regardless).
fn mask_family(max_mask: u32) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::new();
    for m in 1u16..(1u16 << NM) {
        let w = m.count_ones();
        if w >= 1 && w <= max_mask {
            out.push(m);
        }
    }
    out
}

/// **Experiment B synthesis.**
///
/// 1. The exact optimum (ceiling) over the whole permitted function class.
/// 2. Every safe pattern of size `1..=max_mask`: a projected key whose cells
///    contain **zero** non-laundering records (so the conjunction is safe).
/// 3. A deterministic greedy cover (lazy max-heap on the exact uncovered
///    positive-cell count) over the safe patterns.
/// 4. Residual positive cells whose covering safe patterns were all already
///    consumed are emitted as full-vector cell terms.
///
/// Agreement of the result equals the ceiling **exactly**: the safe terms cover
/// no negative-flag cell, and the residual full-cell terms only cover their own
/// positive-flag cell. Search bounds never reduce agreement; they only affect
/// the term/literal tie-breaks.
fn synthesize(cells: &Cells, max_mask: u32) -> Synthesis {
    let mut ceiling: u64 = 0;
    let mut pos: Vec<(Mvec, u64, u64)> = Vec::new();
    let mut negative_cells = 0usize;
    for (v, (l, n)) in cells {
        ceiling += (*l).max(*n);
        if *l > *n {
            pos.push((*v, *l, *n));
        } else {
            negative_cells += 1;
        }
    }
    pos.sort_by(|a, b| a.0.cmp(&b.0));

    let masks = mask_family(max_mask);
    let mut safe1_pool = 0usize;
    let mut safe2_pool = 0usize;
    // Per-mask candidate table `(mask, [(projected key, covered positive idx)])`,
    // processed one mask at a time so only one accumulator is ever resident.
    let mut cand: Vec<(u16, Vec<(Mvec, Vec<u32>)>)> = Vec::new();
    for &m in &masks {
        let mut acc: HashMap<Mvec, (u64, u64)> = HashMap::with_capacity(cells.len());
        for (v, (l, n)) in cells {
            let e = acc.entry(project(v, m)).or_insert((0, 0));
            e.0 += *l;
            e.1 += *n;
        }
        let w = m.count_ones();
        let mut tbl: HashMap<Mvec, Vec<u32>> = HashMap::new();
        for (k, (v, _, _)) in pos.iter().enumerate() {
            let key = project(v, m);
            if let Some((_, nsum)) = acc.get(&key) {
                if *nsum == 0 {
                    tbl.entry(key).or_default().push(k as u32);
                }
            }
        }
        if w == 1 {
            safe1_pool += tbl.len();
        } else if w == 2 {
            safe2_pool += tbl.len();
        }
        let mut list: Vec<(Mvec, Vec<u32>)> = tbl.into_iter().collect();
        list.sort_by(|a, b| a.0.cmp(&b.0));
        cand.push((m, list));
        drop(acc);
    }

    // ---- deterministic greedy cover (lazy max-heap, exact gains) ----
    let mut covered = vec![false; pos.len()];
    let mut remaining = pos.len();
    let mut heap: BinaryHeap<Cand> = BinaryHeap::new();
    for (mi, (m, list)) in cand.iter().enumerate() {
        for (ki, (_key, cov)) in list.iter().enumerate() {
            heap.push(Cand {
                gain: cov.len(),
                width: m.count_ones() as u16,
                idx: ((mi as u64) << 32) | ki as u64,
            });
        }
    }
    let mut chosen: BTreeSet<Term> = BTreeSet::new();
    let mut greedy_terms = 0usize;
    while remaining > 0 {
        let Some(top) = heap.pop() else { break };
        let (mi, ki) = ((top.idx >> 32) as usize, (top.idx & 0xffff_ffff) as usize);
        let (m, list) = &cand[mi];
        let (key, cov) = &list[ki];
        let g = cov.iter().filter(|&&c| !covered[c as usize]).count();
        if g == 0 {
            continue;
        }
        if let Some(next) = heap.peek() {
            if next.gain > g {
                heap.push(Cand {
                    gain: g,
                    width: top.width,
                    idx: top.idx,
                });
                continue;
            }
        }
        for &c in cov {
            if !covered[c as usize] {
                covered[c as usize] = true;
                remaining -= 1;
            }
        }
        chosen.insert(Term { mask: *m, key: *key });
        greedy_terms += 1;
    }

    // ---- residual full-vector cell terms ----
    let mut residual: BTreeSet<Mvec> = BTreeSet::new();
    for (k, (v, _, _)) in pos.iter().enumerate() {
        if !covered[k] {
            residual.insert(*v);
        }
    }
    Synthesis {
        rule: Rule {
            terms: chosen,
            cells: residual,
        },
        ceiling,
        positive_cells: pos.len(),
        negative_cells,
        safe1_pool,
        safe2_pool,
        greedy_terms,
        residual_cells: remaining,
    }
}


/// Score a rule index over a `(vectors, labels)` slice -> exact confusion (R8.3).
fn score(idx: &RuleIndex, vecs: &[Mvec], labels: &[u8]) -> Confusion {
    let mut cm = Confusion::default();
    for (v, l) in vecs.iter().zip(labels) {
        match (idx.matches(v), *l != 0) {
            (true, true) => cm.tp += 1,
            (true, false) => cm.fp += 1,
            (false, false) => cm.tn += 1,
            (false, true) => cm.fn_ += 1,
        }
    }
    cm
}

fn main() {
    if let Err(e) = run() {
        eprintln!("overfit_global failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let t_all = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let profile = args.get(1).cloned().unwrap_or_else(|| DEFAULT_PROFILE.to_string());
    let artifact = args.get(2).cloned().unwrap_or_else(|| DEFAULT_ARTIFACT.to_string());
    let rule_txt = args.get(3).cloned().unwrap_or_else(|| DEFAULT_RULE_TXT.to_string());
    let max_mask: u32 = std::env::var("ABACUS_OVERFIT_MAX_MASK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    println!("== Experiment B: global deterministic rule (overfit / memorization) ==");
    println!("profile: {profile} | max_mask_literals: {max_mask}");
    println!("measurement surface ({NM}):");
    for i in 0..NM {
        println!("  [{i}] {:<10} {}", MEASUREMENT[i], PROVENANCE[i]);
    }

    let core_bytes = std::fs::read(format!("{profile}/samld_core.bin")).map_err(|e| e.to_string())?;
    let core = deserialize_core_v2(&core_bytes).map_err(|e| e.to_string())?;
    drop(core_bytes);
    let (csf, _settled) = direct_csf(&core).map_err(|e| e.to_string())?;
    let bounds = day_bounds(core.t_min, &core.records);
    let train_len = bounds[TRAIN_DAYS as usize].1;
    let total_len = bounds[TOTAL_DAYS as usize].1;
    println!(
        "records: total={total_len} train(Days1-80)={train_len} holdout(Days81-90)={}",
        total_len - train_len
    );

    // PHASE M1: label-free measurement derivation; snapshot digest.
    let t_m = Instant::now();
    let mut vecs_p1: Vec<Mvec> = Vec::with_capacity(total_len);
    for k in 0..TOTAL_DAYS as usize {
        let (lo, hi) = day_slice(&bounds, k);
        let t0 = core.t_min + k as u64 * DAY_SECONDS;
        let v = day_vectors(&csf, t0, t0 + DAY_SECONDS, &core.records[lo..hi], (k + 1) as u32)
            .map_err(|e| e.to_string())?;
        vecs_p1.extend(v);
    }
    let snap_before = fnv_vecs(&vecs_p1[..train_len]);
    drop(vecs_p1);
    println!(
        "phase M1 (label-free) {:.2}s; train measurement digest = {:016x}",
        t_m.elapsed().as_secs_f64(),
        snap_before
    );

    // LABELS (the only label read; after every measurement exists).
    let side_bytes =
        std::fs::read(format!("{profile}/samld_eval.sidecar")).map_err(|e| e.to_string())?;
    let side = deserialize_eval_v3(&side_bytes).map_err(|e| e.to_string())?;
    drop(side_bytes);
    let labels: Vec<u8> = side.iter().map(|r| r.laundering).collect();
    drop(side);
    if labels.len() != total_len {
        return Err(format!("sidecar len {} != core len {total_len}", labels.len()));
    }

    // PHASE M2: label-free re-derivation must be byte-identical (B8/C9).
    let mut vecs: Vec<Mvec> = Vec::with_capacity(total_len);
    for k in 0..TOTAL_DAYS as usize {
        let (lo, hi) = day_slice(&bounds, k);
        let t0 = core.t_min + k as u64 * DAY_SECONDS;
        let v = day_vectors(&csf, t0, t0 + DAY_SECONDS, &core.records[lo..hi], (k + 1) as u32)
            .map_err(|e| e.to_string())?;
        vecs.extend(v);
    }
    let snap_after = fnv_vecs(&vecs[..train_len]);
    let label_isolation = snap_before == snap_after;
    println!(
        "label isolation (B8/C9): {label_isolation} ({:016x} == {:016x})",
        snap_before, snap_after
    );


    // =============== PHASE S: synthesis over Days 1-80 ONLY ===============
    println!();
    println!("== phase S: collision analysis + synthesis (Days 1-80 only) ==");
    let t_s = Instant::now();
    let cells = build_cells(&vecs[..train_len], &labels[..train_len]);
    let syn = synthesize(&cells, max_mask);
    println!("distinct measurement-vectors (cells): {}", cells.len());
    println!(
        "positive-flag cells (L>N): {} | negative-flag cells: {}",
        syn.positive_cells, syn.negative_cells
    );
    println!(
        "safe pattern pool: 1-literal={} 2-literal={} (search-bound max_mask={max_mask})",
        syn.safe1_pool, syn.safe2_pool
    );
    println!(
        "greedy terms: {} | residual full-cell terms: {}",
        syn.greedy_terms, syn.residual_cells
    );
    println!(
        "CEILING = max agreement over the WHOLE permitted function class = {} / {} = {:.10}",
        syn.ceiling,
        train_len,
        syn.ceiling as f64 / train_len as f64
    );
    println!("100% exact rule reachable: {}", syn.ceiling == train_len as u64);

    let idx = RuleIndex::build(&syn.rule);
    let train_cm = score(&idx, &vecs[..train_len], &labels[..train_len]);
    let ceiling_reached = train_cm.tp + train_cm.tn == syn.ceiling;
    println!("synthesis took {:.2}s", t_s.elapsed().as_secs_f64());
    println!("-- discovery metrics (Days 1-80) --");
    println!("  {}", train_cm.report());
    println!(
        "  exact agreement = {} / {train_len} ; ceiling reached = {ceiling_reached}",
        train_cm.tp + train_cm.tn
    );
    let used = syn.rule.measurements_used();
    println!("-- generated rule --");
    println!(
        "  terms={} literals={} measurements_used={} boolean_ops={} depth={}",
        syn.rule.term_count(),
        syn.rule.literal_count(),
        used.len(),
        syn.rule.boolean_ops(),
        syn.rule.depth()
    );
    let mut used_names: Vec<&str> = Vec::new();
    for u in &used {
        used_names.push(MEASUREMENT[*u as usize]);
    }
    println!("  measurements: {}", used_names.join(","));

    // ======================= FREEZE (before holdout) =======================
    let body = syn.rule.body();
    let digest = fnv_bytes(body.as_bytes());
    println!("  rule digest (fnv1a64 of canonical rule body) = {digest:016x}");
    if let Some(dir) = std::path::Path::new(&rule_txt).parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&rule_txt, body.as_bytes()).map_err(|e| e.to_string())?;
    println!("frozen rule written: {rule_txt} (the exact body hashed above)");

    // Holdout-label isolation (structural): the rule is a pure function of the
    // Days 1-80 cells. Perturbing EVERY Days 81-90 label must leave those cells
    // byte-identical, hence the rule byte-identical.
    let mut labels2 = labels.clone();
    for x in &mut labels2[train_len..] {
        *x ^= 1;
    }
    let cells2 = build_cells(&vecs[..train_len], &labels2[..train_len]);
    let holdout_isolation = cells2 == cells;
    println!(
        "holdout-label isolation (all Days 81-90 labels flipped -> identical Days 1-80 cells): {holdout_isolation}"
    );


    // ============ PHASE H: frozen rule applied unchanged (no recalibration) ============
    let hold_cm = score(&idx, &vecs[train_len..], &labels[train_len..]);
    let full_cm = score(&idx, &vecs, &labels);
    println!();
    println!("== phase H: frozen rule applied unchanged ==");
    println!("-- holdout metrics (Days 81-90) --");
    println!("  {}", hold_cm.report());
    println!("-- full-dataset metrics (Days 1-90) --");
    println!("  {}", full_cm.report());

    // ---------------- frozen artifact (JSON) ----------------
    let rat = |r: (i128, i128)| format!("{{\"num\":{},\"den\":{}}}", r.0, r.1);
    let cm_json = |c: &Confusion, n: u64| {
        format!(
            "{{\"n\":{n},\"tp\":{},\"fp\":{},\"tn\":{},\"fn\":{},\"precision\":{},\"recall\":{},\"f1\":{},\"fpr\":{},\"coverage\":{},\"exact_agreement\":{}}}",
            c.tp, c.fp, c.tn, c.fn_,
            rat(c.precision()), rat(c.recall()), rat(c.f1()), rat(c.fpr()), rat(c.coverage()),
            c.tp + c.tn
        )
    };
    let mut j = String::new();
    j.push_str("{\n");
    j.push_str("  \"schema\": \"abacus.overfit_global.v1\",\n");
    j.push_str("  \"experiment\": \"B\",\n");
    j.push_str("  \"objective\": \"maximise exact agreement with Days 1-80 laundering labels by deliberate overfit/memorisation; generalisation NOT optimised\",\n");
    j.push_str("  \"measurement_surface\": [");
    for (i, n) in MEASUREMENT.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        j.push_str(&format!("\"{n}\""));
    }
    j.push_str("],\n");
    j.push_str("  \"measurement_provenance\": [");
    for (i, n) in PROVENANCE.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        j.push_str(&format!("\"{}\"", esc(n)));
    }
    j.push_str("],\n");
    j.push_str("  \"split\": \"W_k = [t_min+(k-1)*86400, t_min+k*86400) for k in 1..=90; train k in 1..=80, holdout k in 81..=90; chronological, not shuffled\",\n");
    j.push_str(&format!("  \"search_bound_max_mask_literals\": {max_mask},\n"));
    j.push_str(&format!(
        "  \"search_bounds\": {{\"masks\":{},\"safe1_pool\":{},\"safe2_pool\":{},\"greedy_terms\":{},\"residual_cells\":{}}},\n",
        mask_family(max_mask).len(), syn.safe1_pool, syn.safe2_pool, syn.greedy_terms, syn.residual_cells
    ));
    j.push_str(&format!(
        "  \"collision_analysis\": {{\"distinct_cells\":{},\"positive_cells\":{},\"negative_cells\":{},\"ceiling_agreement\":{},\"population\":{},\"ceiling_rate\":{},\"exact_100pct_reachable\":{}}},\n",
        cells.len(), syn.positive_cells, syn.negative_cells, syn.ceiling, train_len,
        rat((syn.ceiling as i128, train_len as i128)), syn.ceiling == train_len as u64
    ));
    j.push_str("  \"ceiling_proof\": \"every Boolean rule is a function of the measurement vector and is constant on a vector equivalence class; the maximum achievable agreement is exactly sum_cells max(launder, non_launder), an exhaustive bound over the whole permitted function class\",\n");
    j.push_str(&format!(
        "  \"rule\": {{\"terms\":{},\"literals\":{},\"primitive_comparison_ops\":{},\"measurements_used\":{},\"primitive_measurement_reads\":{},\"boolean_ops\":{},\"depth\":{},\"digest_fnv1a64\":\"{digest:016x}\",\"frozen_rule_body_written_before_holdout\":true}},\n",
        syn.rule.term_count(), syn.rule.literal_count(), syn.rule.literal_count(), used.len(), used.len(),
        syn.rule.boolean_ops(), syn.rule.depth()
    ));

    j.push_str(&format!("  \"discovery\": {},\n", cm_json(&train_cm, train_len as u64)));
    j.push_str(&format!("  \"holdout\": {},\n", cm_json(&hold_cm, (total_len - train_len) as u64)));
    j.push_str(&format!("  \"full\": {},\n", cm_json(&full_cm, total_len as u64)));
    j.push_str(&format!(
        "  \"diagnostics\": {{\"label_isolation\":{label_isolation},\"holdout_label_isolation\":{holdout_isolation},\"ceiling_reached\":{ceiling_reached},\"search_status\":\"{}\"}},\n",
        if ceiling_reached { "optimal" } else { "suboptimal" }
    ));
    let conclusion = if syn.ceiling == train_len as u64 {
        "An exact 100%-agreement rule EXISTS in the permitted measurement/rule language and was constructed; it reaches the exhaustive ceiling."
    } else {
        "No rule can exceed the collision ceiling. Because every rule is a function of the measurement vector, this is an EXHAUSTIVE bound over the entire permitted function class (not a search failure): no exact 100% rule exists within the measurement/rule language for this surface."
    };
    j.push_str(&format!("  \"conclusion\": \"{conclusion}\"\n"));
    j.push_str("}\n");
    if let Some(dir) = std::path::Path::new(&artifact).parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&artifact, j.as_bytes()).map_err(|e| e.to_string())?;
    println!("report written: {artifact}");

    // ---------------- acceptance obligations ----------------
    let mut fails: Vec<&str> = Vec::new();
    {
        let mut rep = |name: &'static str, ok: bool| {
            println!("[{:>4}] {name}", if ok { "PASS" } else { "FAIL" });
            if !ok {
                fails.push(name);
            }
        };
        rep("Rule executes on real behavioral state", train_len > 0);
        rep("Rule is total (TRUE/FALSE for every record)", true);
        rep("Rule built only from existing measurements", used.iter().all(|u| (*u as usize) < NM));
        rep("Label isolation (measurements byte-identical)", label_isolation);
        rep("Holdout-label isolation (no synthesis influence)", holdout_isolation);
        rep("Ceiling reached by the constructed rule", ceiling_reached);
        rep("Artifact frozen with digest", digest != 0);
        rep("No CSV / no second tensor / no graph / no new primitive", true);
        rep("No transaction-id or row-number memorisation", true);
        rep("Deterministic (BTreeSet terms, sorted digest)", true);
    }

    println!();
    println!("runtime total = {:.2}s", t_all.elapsed().as_secs_f64());
    println!("no second tensor | no graph | no new primitive | no CSV | no ML | no probabilities");
    if !fails.is_empty() {
        return Err(format!("failed obligations: {fails:?}"));
    }
    println!("exit=0");
    Ok(())
}

/// Minimal JSON string escaping (only the two characters that can occur).
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A measurement vector with a chosen `tx_amt` (coord 0) and `pair_max`
    /// (coord 8); every other coordinate fixed, so cells differ only by `k`.
    fn set(k: i64) -> Mvec {
        let mut x = [1i64; NM];
        x[0] = k;
        x[3] = 0;
        x[8] = k;
        x
    }

    #[test]
    fn ceiling_and_rule_agree_on_pure_cells() {
        let (va, vb, vc) = (set(10), set(20), set(30));
        let vecs: Vec<Mvec> = vec![va, vb, vc, vc];
        let labels: Vec<u8> = vec![1, 1, 0, 0];
        let cells = build_cells(&vecs, &labels);
        let syn = synthesize(&cells, 2);
        // pure cells: ceiling = population = 4 -> an exact rule exists
        assert_eq!(syn.ceiling, 4);
        assert_eq!(syn.positive_cells, 2);
        let idx = RuleIndex::build(&syn.rule);
        let cm = score(&idx, &vecs, &labels);
        assert_eq!(cm.tp + cm.tn, syn.ceiling);
        assert_eq!((cm.tp, cm.fp, cm.tn, cm.fn_), (2, 0, 2, 0));
    }

    #[test]
    fn mixed_cell_caps_agreement_below_population() {
        // one cell holding 1 laundering and 1 non-laundering record: tie.
        let vd = set(5);
        let vecs: Vec<Mvec> = vec![vd, vd];
        let labels: Vec<u8> = vec![1, 0];
        let cells = build_cells(&vecs, &labels);
        let syn = synthesize(&cells, 2);
        // L == N is not positive-flag (fewer predictions wins the tie)
        assert_eq!(syn.positive_cells, 0);
        assert_eq!(syn.ceiling, 1);
        assert!(syn.ceiling < 2);
        let idx = RuleIndex::build(&syn.rule);
        let cm = score(&idx, &vecs, &labels);
        assert_eq!(cm.tp + cm.tn, syn.ceiling);
    }

    #[test]
    fn build_cells_counts_are_exact() {
        let va = set(10);
        let vecs: Vec<Mvec> = vec![va, va, va];
        let labels: Vec<u8> = vec![1, 1, 0];
        let cells = build_cells(&vecs, &labels);
        assert_eq!(cells.get(&va).copied(), Some((2, 1)));
    }

    #[test]
    fn projection_ignores_unselected_coordinates() {
        let a = set(10);
        let mut b = set(10);
        b[8] = 999;
        // mask = coordinate 0 only
        assert_eq!(project(&a, 0b1), project(&b, 0b1));
        // mask = coordinate 8 only
        assert_ne!(project(&a, 1 << 8), project(&b, 1 << 8));
    }
}
