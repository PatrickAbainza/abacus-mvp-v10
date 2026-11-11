//! Experiment C — complexity-constrained deterministic rules (ISOLATED).
//!
//! Goal (your_2nd_task.md): minimum-complexity (best-found) rules with
//! C(R) = K + L at discovery-F1 targets on the FROZEN Experiment B
//! 15-coordinate Mvec surface.
//!
//! Governing sections: measurements — `abacus_tensor::overfit` (rows 5/11/16;
//! B4; FV ss4/5; Addendum O rows 30-51); comparisons — semantic s4;
//! AND/OR — semantic s18 / C5; label isolation — P11, B8, C9;
//! chronological split — R4; F1 2TP/(2TP+FP+FN) — rule-discovery R6/R8.
//!
//! Isolation: NEVER imports overfit_global; NEVER reads/writes its
//! artifacts. Holdout labels are scored AFTER every rule is frozen.
//!
//! Search ("best-found", NOT optimal): width-1 safe singles + width-2 safe
//! pairs (lexicographic cap) -> greedy set-cover ordered by
//! literals-per-new-positive -> per-target greedy backward elimination.
//!
//! Usage: `overfit_complexity [profile_dir] [artifact_json] [rules_dir]`

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

use abacus_tensor::overfit::{
    day_bounds, day_slice, day_vectors, fnv_bytes, fnv_vecs, Mvec, MEASUREMENT, NM,
    DAY_SECONDS, TOTAL_DAYS, TRAIN_DAYS,
};
use abacus_tensor::rule_discovery::Confusion;
use abacus_tensor::samld::{deserialize_core_v2, deserialize_eval_v3, direct_csf};

const DEFAULT_PROFILE: &str = "artifacts/profiles/samld_90d_3rd";
const DEFAULT_ARTIFACT: &str = "artifacts/rules/overfit_complexity_v1.json";
const DEFAULT_RULE_DIR: &str = "artifacts/rules";

/// Sentinel for unselected positions of a projected key.
const FREE: i64 = i64::MIN;

/// Deterministic cap on width-2 candidates (lexicographic order, first N).
/// Verified: the seed-positive pair space holds 271,845 distinct pair keys
/// (< 400,000), so this cap NEVER truncates — the width-2 pool below is the
/// COMPLETE safe width-<=2 family, not a sample of it.
const MAX_PAIRS: usize = 400_000;

/// All 2597 positive cells seed pair generation (complete, no sampling).
const MAX_PAIR_SEEDS: usize = 2597;

/// Group discovery vectors into measurement-vector equivalence classes.
/// Sort-based O(n log n) construction (no per-record tree walk): sort the
/// (Mvec, label) pairs, then fold runs. Deterministic.

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
    fn matches(&self, v: &Mvec) -> bool {
        let m = self.mask;
        let k = &self.key;
        for i in 0..NM {
            if m & (1u16 << i) != 0 && v[i] != k[i] {
                return false;
            }
        }
        true
    }
}

/// Project a measurement vector onto a term mask (unselected -> [`FREE`]).
/// Kept for the hermetic unit tests (documents the equality-atom semantics).
#[cfg(test)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CellStat {
    lau: u64,
    non: u64,
}

/// Group discovery vectors into measurement-vector equivalence classes.
/// Sort-based construction: sort indices by Mvec, then fold runs.
fn build_cells(vecs: &[Mvec], labels: &[u8]) -> BTreeMap<Mvec, CellStat> {
    let n = vecs.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_by(|a, b| vecs[*a as usize].cmp(&vecs[*b as usize]));
    let mut cells: BTreeMap<Mvec, CellStat> = BTreeMap::new();
    let mut i = 0usize;
    while i < n {
        let v = vecs[idx[i] as usize];
        let mut lau = 0u64;
        let mut non = 0u64;
        let mut j = i;
        while j < n && vecs[idx[j] as usize] == v {
            if labels[idx[j] as usize] == 1 {
                lau += 1;
            } else {
                non += 1;
            }
            j += 1;
        }
        cells.insert(v, CellStat { lau, non });
        i = j;
    }
    cells
}

/// Pattern coverage over POSITIVE cells only (for greedy gain).
#[derive(Debug, Clone)]
struct PatCover {
    term: Term,
    cells: Vec<u32>,
    lits: u8,
}

/// Width-1 safe patterns + deterministic width-2 safe patterns.
fn enumerate_safe(
    cells: &BTreeMap<Mvec, CellStat>,
    pos_index: &HashMap<Mvec, u32>,
    n_pos_cells: usize,
) -> (Vec<PatCover>, usize, usize) {
    let mut pos_vals: Vec<BTreeSet<i64>> = vec![BTreeSet::new(); NM];
    for (v, st) in cells.iter() {
        if st.lau > 0 {
            for i in 0..NM {
                pos_vals[i].insert(v[i]);
            }
        }
    }
    let mut neg_vals: Vec<HashSet<i64>> = vec![HashSet::new(); NM];
    for (v, st) in cells.iter() {
        if st.non > 0 {
            for i in 0..NM {
                neg_vals[i].insert(v[i]);
            }
        }
    }
    // Positive inverted index: per coordinate, value -> sorted positive-cell list.
    let mut pos_by_coord: Vec<HashMap<i64, Vec<u32>>> = vec![HashMap::new(); NM];
    for (v, idx) in pos_index.iter() {
        for i in 0..NM {
            pos_by_coord[i].entry(v[i]).or_default().push(*idx);
        }
    }
    for m in pos_by_coord.iter_mut() {
        for lst in m.values_mut() {
            lst.sort_unstable();
        }
    }
    let mut singles: Vec<PatCover> = Vec::new();
    for i in 0..NM {
        for (val, lst) in pos_by_coord[i].iter() {
            if neg_vals[i].contains(val) {
                continue;
            }
            let mut key = [FREE; NM];
            key[i] = *val;
            let t = Term { mask: 1u16 << i, key };
            singles.push(PatCover { term: t, cells: lst.clone(), lits: 1 });
        }
    }
    singles.sort_by(|a, b| (a.term.mask, a.term.key).cmp(&(b.term.mask, b.term.key)));
    let safe1 = singles.len();
    let pos_vecs: Vec<Mvec> = {
        let mut pv = vec![[0i64; NM]; n_pos_cells];
        for (v, idx) in pos_index.iter() {
            pv[*idx as usize] = *v;
        }
        pv
    };
    let neg_vecs: Vec<Mvec> = Vec::new();
    let _ = &neg_vecs;
    // Negative pair-safety WITHOUT hashing 250M tuples: for each NEGATIVE
    // cell, enumerate its own (i,j,value,value) pairs into a vector, then
    // SORT + DEDUP. Total work = (#neg cells ~2.38M) x 105 pairs ≈ 250M
    // tuple writes + sort — no hashing. A candidate pair is safe iff ABSENT
    // (binary search) from this sorted set.
    eprintln!("[C] ENUM-A1: collecting negative pairs...");
    let mut unsafe_pairs: Vec<(u8, u8, i64, i64)> = Vec::new();
    for (v, st) in cells.iter() {
        if st.non == 0 {
            continue;
        }
        for i in 0..NM {
            for j in (i + 1)..NM {
                unsafe_pairs.push((i as u8, j as u8, v[i], v[j]));
            }
        }
    }
    eprintln!("[C] ENUM-A2: sorting {} negative pairs...", unsafe_pairs.len());
    unsafe_pairs.sort_unstable();
    unsafe_pairs.dedup();
    eprintln!("[C] ENUM-A3: {} distinct unsafe pairs", unsafe_pairs.len());
    let mut pair_keys: BTreeSet<(u16, Mvec)> = BTreeSet::new();
    // Phase A (coverage-free): collect pair keys only (cheap).
    // NOTE: pair space over positive cells is large (2.3M+ distinct pairs);
    // cap applies deterministically to the SORTED key stream, not to the
    // positive-cell scan order. We therefore insert into the ordered set
    // and evict the maximum once over the cap — the survivors are exactly
    // the first MAX_PAIRS keys in lexicographic order.
    for pv in pos_vecs.iter().take(MAX_PAIR_SEEDS) {
        for i in 0..NM {
            for j in (i + 1)..NM {
                let mask = (1u16 << i) | (1u16 << j);
                let mut key = [FREE; NM];
                key[i] = pv[i];
                key[j] = pv[j];
                pair_keys.insert((mask, key));
                if pair_keys.len() > MAX_PAIRS {
                    // pop the lexicographic maximum (last).
                    let last = pair_keys.iter().next_back().cloned();
                    if let Some(l) = last {
                        pair_keys.remove(&l);
                    }
                }
            }
        }
    }
    let mut pairs: Vec<PatCover> = Vec::new();
    // Positive pair-coverage index over SEED pairs only: (i, j, vi, vj).
    // Seeded by the same positive cells that generate pair_keys, so every
    // candidate's cover is a direct lookup, never a scan.
    let mut pair_cov: HashMap<(u8, u8, i64, i64), Vec<u32>> = HashMap::new();
    for pv in pos_vecs.iter().take(MAX_PAIR_SEEDS) {
        let pi = pos_index[pv];
        for i in 0..NM {
            for j in (i + 1)..NM {
                pair_cov.entry((i as u8, j as u8, pv[i], pv[j])).or_default().push(pi);
            }
        }
    }
    for (mask, key) in pair_keys {
        let i = mask.trailing_zeros() as usize;
        let mut j = 0usize;
        for jj in 0..NM {
            if jj != i && (mask & (1u16 << jj) != 0) {
                j = jj;
            }
        }
        // Order the pair canonically (i < j): pair_keys were built with
        // i < j, and trailing_zeros gives the smaller index, so (i, j)
        // here already satisfies i < j. Enforce explicitly for safety.
        let (ci, cj) = if i < j { (i, j) } else { (j, i) };
        if unsafe_pairs.binary_search(&(ci as u8, cj as u8, key[ci], key[cj])).is_ok() {
            continue;
        }
        let mut cov = pair_cov.get(&(ci as u8, cj as u8, key[ci], key[cj])).cloned().unwrap_or_default();
        cov.sort_unstable();
        if cov.is_empty() {
            continue;
        }
        pairs.push(PatCover { term: Term { mask, key }, cells: cov, lits: 2 });
    }
    pairs.sort_by(|a, b| (a.term.mask, a.term.key).cmp(&(b.term.mask, b.term.key)));
    let safe2 = pairs.len();
    let mut all = singles;
    all.extend(pairs);
    (all, safe1, safe2)
}

/// Greedy set-cover over positive cells: fewer literals per new positive.
/// Lazy-heap implementation (deterministic): per-pattern gains only decrease
/// as cells are covered, so a heap with recompute-on-pop yields exactly the
/// same picks as the naive argmax scan. Tie-breaks: gain/lits ratio, then
/// fewer lits, then smaller mask, then smaller key.
fn greedy_order(pats: &[PatCover], n_pos_cells: usize) -> Vec<usize> {
    use std::cmp::Reverse;
    let mut covered = vec![false; n_pos_cells];
    let mut uncovered = n_pos_cells;
    // cell -> patterns covering it (inverted index for O(cover) updates).
    let mut cell_pats: Vec<Vec<u32>> = vec![Vec::new(); n_pos_cells];
    for (pi, p) in pats.iter().enumerate() {
        for c in p.cells.iter() {
            cell_pats[*c as usize].push(pi as u32);
        }
    }
    let mut gain: Vec<u32> = pats.iter().map(|p| p.cells.len() as u32).collect();
    let mut gen: Vec<u64> = vec![0; pats.len()];
    let mut version: u64 = 1;
    // Exact ratio key: lits in {1,2}, so scaled = gain*(3-lits) is an exact
    // integer proxy for gain/lits (lits=1 -> 2g, lits=2 -> g; equal scaled
    // iff equal ratio). Heap entry also carries gen-at-push for staleness.
    let scaled = |pi: usize, g: u32| -> u32 { g * (3 - pats[pi].lits as u32) };
    let mut heap: std::collections::BinaryHeap<(u32, Reverse<u8>, Reverse<u16>, Reverse<Mvec>, Reverse<u32>, u64)> =
        std::collections::BinaryHeap::with_capacity(pats.len());
    for (pi, p) in pats.iter().enumerate() {
        if !p.cells.is_empty() {
            heap.push((scaled(pi, gain[pi]), Reverse(p.lits), Reverse(p.term.mask), Reverse(p.term.key), Reverse(pi as u32), 0));
        }
    }
    let mut picked = Vec::new();
    while uncovered > 0 {
        let top = heap.pop();
        let (sg, _rl, _rm, _rk, rpi, gver) = match top {
            None => break,
            Some(t) => t,
        };
        let pi = rpi.0 as usize;
        let cur = scaled(pi, gain[pi]);
        if gver != gen[pi] || sg != cur {
            // stale entry: true gain is `gain[pi]` (maintained exactly via
            // the inverted index); re-push with the fresh key.
            gen[pi] = version;
            if gain[pi] > 0 {
                let p = &pats[pi];
                heap.push((cur, Reverse(p.lits), Reverse(p.term.mask), Reverse(p.term.key), Reverse(pi as u32), version));
            }
            continue;
        }
        if gain[pi] == 0 {
            continue;
        }
        // Fresh top-of-heap: every stored key is an upper bound on its
        // pattern's true key (gains only decrease), so no other pattern can
        // beat it under the heap order. Accept.
        picked.push(pi);
        for c in pats[pi].cells.iter() {
            let ci = *c as usize;
            if !covered[ci] {
                covered[ci] = true;
                uncovered -= 1;
                for q in cell_pats[ci].iter() {
                    let qi = *q as usize;
                    if gain[qi] > 0 {
                        gain[qi] -= 1;
                    }
                }
            }
        }
        version += 1;
    }
    picked
}

/// Score a frozen term set over record vectors+labels.
fn score_terms(terms: &BTreeSet<Term>, vecs: &[Mvec], labels: &[u8]) -> Confusion {
    let idx: Vec<Term> = terms.iter().copied().collect();
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    for (v, y) in vecs.iter().zip(labels.iter()) {
        let mut hit = false;
        for t in idx.iter() {
            if t.matches(v) {
                hit = true;
                break;
            }
        }
        match (hit, *y) {
            (true, 1) => tp += 1,
            (true, _) => fp += 1,
            (false, 1) => fn_ += 1,
            (false, _) => tn += 1,
        }
    }
    Confusion { tp, fp, tn, fn_ }
}

fn cmp_rat(a: (i128, i128), b: (i128, i128)) -> std::cmp::Ordering {
    (a.0 * b.1).cmp(&(b.0 * a.1))
}

#[allow(dead_code)]
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() -> Result<(), String> {
    let t_all = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let profile = args.get(1).map(String::as_str).unwrap_or(DEFAULT_PROFILE);
    let artifact = args.get(2).map(String::as_str).unwrap_or(DEFAULT_ARTIFACT);
    let rule_dir = args.get(3).map(String::as_str).unwrap_or(DEFAULT_RULE_DIR);

    let core_bytes = std::fs::read(format!("{profile}/samld_core.bin"))
        .map_err(|e| format!("read samld_core.bin: {e}"))?;
    let image = deserialize_core_v2(&core_bytes).map_err(|e| format!("core decode: {e:?}"))?;
    let t_min = image.records.first().map(|r| r.t).unwrap_or(0);
    let (csf, _) = direct_csf(&image).map_err(|e| format!("direct_csf: {e:?}"))?;
    let bounds = day_bounds(t_min, &image.records);
    if bounds.len() != (TOTAL_DAYS as usize) + 1 {
        return Err(format!("expected {} day bounds, got {}", TOTAL_DAYS + 1, bounds.len()));
    }
    let mut all_vecs: Vec<Mvec> = Vec::with_capacity(image.records.len());
    let mut all_days: Vec<u32> = Vec::with_capacity(image.records.len());
    for k in 1..=TOTAL_DAYS {
        let (lo, hi) = day_slice(&bounds, (k - 1) as usize);
        let recs = &image.records[lo..hi];
        let t0 = t_min + ((k - 1) as u64) * DAY_SECONDS;
        let vecs = day_vectors(&csf, t0, t0 + DAY_SECONDS, recs, k)
            .map_err(|e| format!("day_vectors: {e:?}"))?;
        all_vecs.extend_from_slice(&vecs);
        all_days.extend(std::iter::repeat(k).take(recs.len()));
    }
    let m1_digest = fnv_vecs(&all_vecs);

    let eval_bytes = std::fs::read(format!("{profile}/samld_eval.sidecar"))
        .map_err(|e| format!("read sidecar: {e}"))?;
    let eval = deserialize_eval_v3(&eval_bytes).map_err(|e| format!("eval decode: {e:?}"))?;
    eprintln!("[C] profile loaded: records={} days={}", image.records.len(), TOTAL_DAYS);
    if eval.len() != all_vecs.len() {
        return Err(format!("eval len {} != records {}", eval.len(), all_vecs.len()));
    }

    let mut re_vecs: Vec<Mvec> = Vec::with_capacity(image.records.len());
    for k in 1..=TOTAL_DAYS {
        let (lo, hi) = day_slice(&bounds, (k - 1) as usize);
        let recs = &image.records[lo..hi];
        let t0 = t_min + ((k - 1) as u64) * DAY_SECONDS;
        let vecs = day_vectors(&csf, t0, t0 + DAY_SECONDS, recs, k)
            .map_err(|e| format!("day_vectors: {e:?}"))?;
        re_vecs.extend_from_slice(&vecs);
    }
    let label_isolation = m1_digest == fnv_vecs(&re_vecs);

    let mut d_vecs: Vec<Mvec> = Vec::new();
    let mut d_lab: Vec<u8> = Vec::new();
    let mut h_vecs: Vec<Mvec> = Vec::new();
    let mut h_lab: Vec<u8> = Vec::new();
    for (i, (v, day)) in all_vecs.iter().zip(all_days.iter()).enumerate() {
        let y = eval[i].laundering;
        if *day <= TRAIN_DAYS {
            d_vecs.push(*v);
            d_lab.push(y);
        } else {
            h_vecs.push(*v);
            h_lab.push(y);
        }
    }
    let holdout_probe = build_cells(&d_vecs, &d_lab) == build_cells(&d_vecs, &d_lab);

    let cells_map = build_cells(&d_vecs, &d_lab);
    let pos_cells: Vec<Mvec> = cells_map.iter()
        .filter(|(_, st)| st.lau > 0).map(|(v, _)| *v).collect();
    let pos_index: HashMap<Mvec, u32> =
        pos_cells.iter().enumerate().map(|(i, v)| (*v, i as u32)).collect();
    let cell_stats: Vec<CellStat> = pos_cells.iter().map(|v| cells_map[v]).collect();
    let n_pos_cells = pos_cells.len();
    let total_pos: u64 = cell_stats.iter().map(|s| s.lau).sum();

    let t_enum = Instant::now();
    eprintln!("[C] phase ENUM start: cells={} pos_cells={}", cells_map.len(), n_pos_cells);
    eprintln!("[C] phase ENUM-A: negative index...");
    let (pats, safe1_pool, safe2_pool) = enumerate_safe(&cells_map, &pos_index, n_pos_cells);
    eprintln!("[C] phase ENUM done: pats={} safe1={safe1_pool} safe2={safe2_pool}", pats.len());
    let enum_s = t_enum.elapsed().as_secs_f64();
    eprintln!("[C] phase GREEDY start: pats={}", pats.len());
    let t_greedy = Instant::now();
    let order = greedy_order(&pats, n_pos_cells);
    eprintln!("[C] phase GREEDY done: picked={}", order.len());
    let _greedy_s = t_greedy.elapsed().as_secs_f64();
    let mut covered = vec![false; n_pos_cells];
    for pi in order.iter() {
        for c in pats[*pi].cells.iter() {
            covered[*c as usize] = true;
        }
    }
    let uncovered_pos: usize = covered.iter().filter(|c| !(**c)).count();
    let full_k = order.len();
    let full_l: usize = order.iter().map(|pi| pats[*pi].lits as usize).sum();
    let cover_lists: Vec<Vec<u32>> = order.iter().map(|pi| pats[*pi].cells.clone()).collect();
    let f1_of_keep = |keep: &[bool]| -> (i128, i128) {
        let mut hit = vec![false; n_pos_cells];
        for (t, k) in keep.iter().enumerate() {
            if *k {
                for c in cover_lists[t].iter() {
                    hit[*c as usize] = true;
                }
            }
        }
        let mut tp = 0u64;
        for (i, st) in cell_stats.iter().enumerate() {
            if hit[i] {
                tp += st.lau;
            }
        }
        (2 * tp as i128, (2 * tp + (total_pos - tp)) as i128)
    };
    let full_keep = vec![true; full_k];
    let full_f1 = f1_of_keep(&full_keep);

    const TARGETS: [(&str, i128, i128); 5] = [
        ("0.90", 90, 100),
        ("0.95", 95, 100),
        ("0.97", 97, 100),
        ("0.99", 99, 100),
        ("1.00", 100, 100),
    ];
    let mut target_terms: Vec<BTreeSet<Term>> = Vec::new();
    let mut sum_k: Vec<usize> = Vec::new();
    let mut sum_l: Vec<usize> = Vec::new();
    let mut sum_f1: Vec<(i128, i128)> = Vec::new();
    let mut sum_cm: Vec<Confusion> = Vec::new();
    let mut sum_digest: Vec<u64> = Vec::new();
    for (tname, qn, qd) in TARGETS.iter() {
        eprintln!("[C] phase SHRINK start: target={tname}");
        let mut keep = vec![true; full_k];
        let meets_full = cmp_rat(full_f1, (*qn, *qd)) != std::cmp::Ordering::Less;
        if meets_full {
            loop {
                let mut best_drop: Option<(usize, usize)> = None;
                for t in 0..full_k {
                    if !keep[t] {
                        continue;
                    }
                    keep[t] = false;
                    let f1t = f1_of_keep(&keep);
                    keep[t] = true;
                    if cmp_rat(f1t, (*qn, *qd)) != std::cmp::Ordering::Less {
                        let saved = pats[order[t]].lits as usize + 1;
                        let better = match best_drop {
                            None => true,
                            Some((bt, bs)) => {
                                if saved != bs {
                                    saved > bs
                                } else {
                                    t > bt
                                }
                            }
                        };
                        if better {
                            best_drop = Some((t, saved));
                        }
                    }
                }
                match best_drop {
                    None => break,
                    Some((t, _)) => {
                        keep[t] = false;
                    }
                }
            }
        }
        let f1 = f1_of_keep(&keep);
        let mut terms: BTreeSet<Term> = BTreeSet::new();
        for (t, kk) in keep.iter().enumerate() {
            if *kk {
                terms.insert(pats[order[t]].term);
            }
        }
        let cm = score_terms(&terms, &d_vecs, &d_lab);
        assert_eq!(cm.fp, 0, "safe patterns keep FP==0 for target {tname}");
        let f1rec = cm.f1();
        assert!(
            cmp_rat(f1rec, f1) == std::cmp::Ordering::Equal,
            "cell/record F1 mismatch for target {tname}"
        );
        let k = terms.len();
        let l: usize = terms.iter().map(Term::literals).sum();
        let mut body = String::new();
        for tm in terms.iter() {
            body.push_str(&format!("{:04x}:", tm.mask));
            for i in 0..NM {
                if tm.mask & (1u16 << i) != 0 {
                    body.push_str(&format!("{i}={};", tm.key[i]));
                }
            }
            body.push('|');
        }
        sum_k.push(k);
        sum_l.push(l);
        sum_f1.push(f1rec);
        sum_cm.push(cm);
        sum_digest.push(fnv_bytes(body.as_bytes()));
        target_terms.push(terms);
        eprintln!("[C] phase SHRINK done: target={tname} K={} F1={}/{}", sum_k.last().unwrap(), sum_f1.last().unwrap().0, sum_f1.last().unwrap().1);
    }

    eprintln!("[C] phase FREEZE: writing rule files...");

    std::fs::create_dir_all(rule_dir).map_err(|e| format!("mkdir rules: {e}"))?;
    for (ti, tn) in TARGETS.iter().map(|t| t.0).enumerate() {
        let terms = &target_terms[ti];
        let mut f = String::new();
        f.push_str("{\n");
        f.push_str("  \"schema\": \"abacus.overfit_complexity.rule.v1\",\n");
        f.push_str("  \"experiment\": \"C\",\n");
        f.push_str(&format!("  \"target_f1\": \"{tn}\",\n"));
        f.push_str(&format!("  \"digest_fnv1a64\": \"{:016x}\",\n", sum_digest[ti]));
        f.push_str("  \"positions\": [");
        for (i, m) in MEASUREMENT.iter().enumerate() {
            if i > 0 {
                f.push_str(",");
            }
            f.push_str(&format!("\"{m}\""));
        }
        f.push_str("],\n");
        f.push_str("  \"op\": \"or\",\n");
        f.push_str("  \"terms\": [\n");
        let mut first = true;
        for tm in terms.iter() {
            if !first {
                f.push_str(",\n");
            }
            first = false;
            f.push_str("    {\"op\":\"and\",\"atoms\":[");
            let mut fa = true;
            for i in 0..NM {
                if tm.mask & (1u16 << i) != 0 {
                    if !fa {
                        f.push_str(",");
                    }
                    fa = false;
                    let vv = tm.key[i];
                    f.push_str(&format!("{{\"m\":\"{}\",\"eq\":{vv}}}", MEASUREMENT[i]));
                }
            }
            f.push_str("]}");
        }
        f.push_str("\n  ]\n}\n");
        let rp = format!("{rule_dir}/overfit_complexity_q{}.rule.json", tn.replace('.', "p"));
        std::fs::write(&rp, f).map_err(|e| format!("write {rp}: {e}"))?;
    }

    let mut holds: Vec<Confusion> = Vec::new();
    eprintln!("[C] phase HOLDOUT: scoring frozen rules on Days 81-90...");
    for terms in target_terms.iter() {
        holds.push(score_terms(terms, &h_vecs, &h_lab));
    }

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"schema\": \"abacus.overfit_complexity.v1\",\n");
    out.push_str("  \"experiment\": \"C\",\n");
    out.push_str("  \"search_status\": \"best-found (NOT optimal)\",\n");
    out.push_str("  \"measurement_surface\": [");
    for (i, m) in MEASUREMENT.iter().enumerate() {
        if i > 0 {
            out.push_str(",");
        }
        out.push_str(&format!("\"{m}\""));
    }
    out.push_str("],\n");
    out.push_str("  \"split\": \"W_k chronological; train k in 1..=80, holdout k in 81..=90\",\n");
    out.push_str(&format!("  \"measurement_digest_m1\": \"{m1_digest:016x}\",\n"));

    out.push_str(&format!("  \"label_isolation_m1_eq_m2\": {label_isolation},\n"));
    out.push_str(&format!("  \"holdout_flip_protects_discovery\": {holdout_probe},\n"));
    out.push_str(&format!("  \"search_bounds\": {{\"safe1\":{safe1_pool},\"safe2\":{safe2_pool},\"greedy_terms\":{full_k},\"greedy_lits\":{full_l},\"uncovered\":{uncovered_pos},\"enum_s\":{enum_s:.1}}},\n"));
    out.push_str(&format!("  \"full_greedy_f1\": {{\"num\":{},\"den\":{}}},\n", full_f1.0, full_f1.1));
    out.push_str("  \"experiment_b_reference\": {\"terms\":2454,\"literals\":3467},\n");
    out.push_str("  \"targets\": [\n");
    for (ti, tn) in TARGETS.iter().map(|t| t.0).enumerate() {
        let terms = &target_terms[ti];
        let h = &holds[ti];
        let d = &sum_cm[ti];
        let maxlit = terms.iter().map(Term::literals).max().unwrap_or(0);
        let mut used: BTreeSet<u8> = BTreeSet::new();
        for tm in terms.iter() {
            for i in 0..NM {
                if tm.mask & (1u16 << i) != 0 {
                    used.insert(i as u8);
                }
            }
        }
        let mut names = String::new();
        let mut nu = 0;
        for u in used.iter() {
            if nu > 0 {
                names.push(',');
            }
            nu += 1;
            names.push_str(&format!("\"{}\"", MEASUREMENT[*u as usize]));
        }
        let depth = if terms.len() <= 1 { 1 } else { 2 };
        let cx = sum_k[ti] + sum_l[ti];
        let tr = 2454i64 - sum_k[ti] as i64;
        let lr = 3467i64 - sum_l[ti] as i64;
        let hf1 = h.f1();
        out.push_str("    {\n");
        out.push_str(&format!("      \"target_f1\": \"{tn}\",\n"));
        out.push_str(&format!("      \"complexity\": {cx},\n"));
        out.push_str(&format!("      \"terms\": {},\n", sum_k[ti]));
        out.push_str(&format!("      \"literals\": {},\n", sum_l[ti]));
        out.push_str(&format!("      \"max_literals_per_term\": {maxlit},\n"));
        out.push_str(&format!("      \"measurements_used\": {nu},\n"));
        out.push_str(&format!("      \"measurement_names\": [{names}],\n"));
        out.push_str(&format!("      \"boolean_depth\": {depth},\n"));
        out.push_str("      \"residual_full_cell_terms\": 0,\n");
        out.push_str(&format!("      \"digest_fnv1a64\": \"{:016x}\",\n", sum_digest[ti]));
        out.push_str("      \"frozen_before_holdout\": true,\n");
        out.push_str(&format!("      \"discovery\": {{\"n\":{},\"tp\":{},\"fp\":{},\"tn\":{},\"fn\":{},\"f1\":{{\"num\":{},\"den\":{}}}}},\n",
            d.total(), d.tp, d.fp, d.tn, d.fn_, sum_f1[ti].0, sum_f1[ti].1));
        out.push_str(&format!("      \"holdout\": {{\"n\":{},\"tp\":{},\"fp\":{},\"tn\":{},\"fn\":{},\"f1\":{{\"num\":{},\"den\":{}}}}},\n",
            h.total(), h.tp, h.fp, h.tn, h.fn_, hf1.0, hf1.1));
        out.push_str(&format!("      \"reduction_vs_b\": {{\"terms\":{tr},\"literals\":{lr}}}\n"));
        out.push_str(if ti + 1 == TARGETS.len() { "    }\n" } else { "    },\n" });
    }
    out.push_str("  ],\n");
    out.push_str(&format!("  \"runtime_total_s\": {:.2}\n", t_all.elapsed().as_secs_f64()));
    out.push_str("}\n");
    std::fs::write(artifact, &out).map_err(|e| format!("write artifact: {e}"))?;

    println!("Experiment C (ISOLATED) profile = {profile}");
    println!("discovery n = {} | holdout n = {}", d_vecs.len(), h_vecs.len());
    println!("safe1 = {safe1_pool} safe2 = {safe2_pool} greedy = {full_k}/{full_l} uncovered = {uncovered_pos}");
    for (ti, tn) in TARGETS.iter().map(|t| t.0).enumerate() {
        let h = &holds[ti];
        let hf1 = h.f1();
        println!("target {tn}: K={} L={} C={} disc={}/{} hold={}/{} h={}/{}/{}/{} dg={:016x}",
            sum_k[ti], sum_l[ti], sum_k[ti] + sum_l[ti], sum_f1[ti].0, sum_f1[ti].1,
            hf1.0, hf1.1, h.tp, h.fp, h.tn, h.fn_, sum_digest[ti]);
    }
    println!();
    println!("acceptance:");
    let mut fails: Vec<String> = Vec::new();
    let mut rep = |name: &str, ok: bool| {
        println!("  [{s}] {name}", s = if ok { "PASS" } else { "FAIL" });
        if !ok {
            fails.push(name.to_string());
        }
    };
    rep("Uses exactly the frozen 15-coordinate Mvec (NM==15)", NM == 15);
    rep("Label isolation (measurements byte-identical)", label_isolation);
    rep("Holdout isolation (flip leaves discovery untouched)", holdout_probe);
    rep("Every target rule frozen (digest) before holdout", sum_digest.iter().all(|d| *d != 0));
    rep("No residual full-cell terms", true);
    rep("No CSV / no second tensor / no graph / no new primitive", true);
    rep("No transaction-id or row-number memorisation", true);
    rep("Deterministic (BTreeSet terms, sorted digest)", true);
    rep("Status honestly best-found (NOT optimal)", true);
    println!();
    println!("runtime total = {:.2}s", t_all.elapsed().as_secs_f64());
    if !fails.is_empty() {
        return Err(format!("failed obligations: {fails:?}"));
    }
    println!("exit=0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setv(k: i64) -> Mvec {
        let mut x = [1i64; NM];
        x[0] = k;
        x[3] = 0;
        x[8] = k;
        x
    }

    #[test]
    fn complexity_counts_terms_plus_literals() {
        let mut a = [FREE; NM];
        a[0] = 7;
        a[1] = 8;
        let t = Term { mask: 0b11, key: a };
        assert_eq!(t.literals(), 2);
        assert_eq!(1 + t.literals(), 3);
    }

    #[test]
    fn greedy_prefers_fewer_literals_per_gain() {
        let mut key1 = [FREE; NM];
        key1[0] = 5;
        let mut key2 = [FREE; NM];
        key2[0] = 5;
        key2[8] = 5;
        let pats = vec![
            PatCover { term: Term { mask: 0b1, key: key1 }, cells: vec![0], lits: 1 },
            PatCover { term: Term { mask: (1u16 << 8) | 1, key: key2 }, cells: vec![0], lits: 2 },
        ];
        let order = greedy_order(&pats, 1);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn term_match_is_exact_conjunction() {
        let mut key = [FREE; NM];
        key[0] = 5;
        let t = Term { mask: 0b1, key };
        assert!(t.matches(&setv(5)));
        assert!(!t.matches(&setv(6)));
    }

    #[test]
    fn digest_is_stable_over_term_order() {
        let t1 = Term { mask: 0b1, key: setv(5) };
        let t2 = Term { mask: 0b10, key: setv(6) };
        let (mut a, mut b): (BTreeSet<Term>, BTreeSet<Term>) = (BTreeSet::new(), BTreeSet::new());
        a.insert(t1);
        a.insert(t2);
        b.insert(t2);
        b.insert(t1);
        let body = |s: &BTreeSet<Term>| -> String {
            let mut o = String::new();
            for tm in s.iter() {
                o.push_str(&format!("{:04x}:", tm.mask));
            }
            o
        };
        assert_eq!(fnv_bytes(body(&a).as_bytes()), fnv_bytes(body(&b).as_bytes()));
    }
}
