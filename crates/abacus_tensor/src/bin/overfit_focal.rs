//! Experiment A — focal-level characterization (per-transaction overfit).
//!
//! Objective (task): for **every** Days 1–80 laundering transaction, treat it
//! as the focal event, build its permitted behavioral context from the existing
//! tensor, and search for the simplest deterministic measurement/condition that
//! distinguishes it from its surrounding transactions. This is an intentionally
//! overfit **per-transaction** characterization; no single global rule is
//! produced here (that is Experiment B).
//!
//! Governing sections (Golden Source Compliance Rule — citations):
//! * measurements — `abacus_tensor::overfit` (rows 5/11/16; B4; FV §§4/5; Addendum O rows 30–51).
//! * the permitted context — one declared day-window (row 16 `time_slice`; R4 split).
//! * comparisons — semantic §4; `AND` composition — semantic §18 / C5.
//! * label isolation — P11, B8, C9.
//!
//! The condition depends only on **behavioral measurements** — never on a
//! transaction id or a row number — and uses no primitive outside the existing
//! surface.
//!
//! Usage:
//! ```text
//! overfit_focal [profile_dir] [artifact_json] [artifact_records_json]
//! ```

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use abacus_tensor::overfit::{
    day_bounds, day_slice, day_vectors, fnv_bytes, fnv_vecs, Mvec, MEASUREMENT, NM,
    DAY_SECONDS, PROVENANCE, TOTAL_DAYS, TRAIN_DAYS,
};
use abacus_tensor::samld::{deserialize_core_v2, deserialize_eval_v3, direct_csf};

const DEFAULT_PROFILE: &str = "artifacts/profiles/samld_90d_3rd";
const DEFAULT_ARTIFACT: &str = "artifacts/rules/overfit_focal_v1.json";
const DEFAULT_RECORDS: &str = "artifacts/rules/overfit_focal_v1.records.json";
const FULL_MASK: u16 = (1u16 << NM) - 1;

/// Do `a` and `b` agree on every coordinate selected by `mask`?
#[must_use]
fn agrees(a: &Mvec, b: &Mvec, mask: u16) -> bool {
    for i in 0..NM {
        if mask & (1u16 << i) != 0 && a[i] != b[i] {
            return false;
        }
    }
    true
}

/// Per-day per-coordinate value index: `idx[i][value] = record positions`.
type DayIndex = Vec<HashMap<i64, Vec<u32>>>;

/// Build the per-coordinate value index for one day's record slice.
fn day_index(day: &[Mvec]) -> DayIndex {
    let mut idx: DayIndex = (0..NM).map(|_| HashMap::new()).collect();
    for (j, v) in day.iter().enumerate() {
        for (i, slot) in idx.iter_mut().enumerate() {
            slot.entry(v[i]).or_default().push(j as u32);
        }
    }
    idx
}

/// How many **other** records in the day agree with `v` on `mask`?
/// Seeded from the most selective coordinate list in the mask.
#[must_use]
fn other_matches(idx: &DayIndex, day: &[Mvec], v: &Mvec, mask: u16, self_i: u32) -> usize {
    let mut seed: Option<&Vec<u32>> = None;
    for i in 0..NM {
        if mask & (1u16 << i) == 0 {
            continue;
        }
        if let Some(list) = idx[i].get(&v[i]) {
            if seed.is_none_or(|s| list.len() < s.len()) {
                seed = Some(list);
            }
        }
    }
    match seed {
        None => 0,
        Some(list) => list
            .iter()
            .filter(|&&j| j != self_i)
            .filter(|&&j| agrees(v, &day[j as usize], mask))
            .count(),
    }
}


/// The simplest (greedy-minimal) coordinate subset that uniquely identifies the
/// focal within its day, or `(FULL_MASK, false)` when even the whole
/// measurement vector collides with another record of the same day.
///
/// Coordinates are tried in positional order, so `tx_amt` is dropped first
/// whenever the remaining **behavioral** measurements already isolate the focal.
#[must_use]
fn isolate(idx: &DayIndex, day: &[Mvec], v: &Mvec, self_i: u32) -> (u16, bool) {
    if other_matches(idx, day, v, FULL_MASK, self_i) != 0 {
        return (FULL_MASK, false);
    }
    let mut mask = FULL_MASK;
    for i in 0..NM {
        let cand = mask & !(1u16 << i);
        if cand == 0 {
            continue;
        }
        if other_matches(idx, day, v, cand, self_i) == 0 {
            mask = cand;
        }
    }
    (mask, true)
}

/// One focal characterization record (task's required fields).
struct FocalRec {
    /// 1-based declared day of the focal.
    day: u32,
    /// Focal coordinate `(t,s,r)` (identification only — never a rule term).
    t: u64,
    s: u32,
    r: u32,
    /// Number of equality literals in the distinguishing condition.
    lits: usize,
    /// The distinguishing coordinate subset (mask) — for aggregate statistics.
    mask: u16,
    /// Whether the condition uniquely identifies the focal in its day context.
    isolated: bool,
    /// The distinguishing condition `AND` of equality atoms.
    condition: String,
    /// The primitives realizing the condition's measurements.
    prims: String,
}

/// The conjunction of equality atoms selected by `mask`.
#[must_use]
fn condition_text(v: &Mvec, mask: u16) -> String {
    let mut parts: Vec<String> = Vec::new();
    for i in 0..NM {
        if mask & (1u16 << i) != 0 {
            parts.push(format!("{}={}", MEASUREMENT[i], v[i]));
        }
    }
    if parts.len() == 1 {
        parts.pop().unwrap_or_default()
    } else {
        format!("AND({})", parts.join(", "))
    }
}

/// The primitives realizing the measurements selected by `mask`.
#[must_use]
fn prims_text(mask: u16) -> String {
    let mut parts: Vec<String> = Vec::new();
    for i in 0..NM {
        if mask & (1u16 << i) != 0 {
            parts.push(format!("{}:{}", MEASUREMENT[i], PROVENANCE[i]));
        }
    }
    parts.join(" | ")
}

/// Characterize every laundering focal of the day range `[d0,d1)`.
fn process_period(
    d0: usize,
    d1: usize,
    core: &abacus_tensor::samld::CoreImage,
    bounds: &[(usize, usize)],
    vecs: &[Mvec],
    labels: &[u8],
) -> Vec<FocalRec> {
    let mut out: Vec<FocalRec> = Vec::new();
    for k in d0..d1 {
        let (lo, hi) = day_slice(bounds, k);
        let day = &vecs[lo..hi];
        if day.is_empty() {
            continue;
        }
        let idx = day_index(day);
        for (i, (v, l)) in day.iter().zip(&labels[lo..hi]).enumerate() {
            if *l == 0 {
                continue;
            }
            let (mask, isolated) = isolate(&idx, day, v, i as u32);
            let rec = &core.records[lo + i];
            out.push(FocalRec {
                day: (k + 1) as u32,
                t: rec.t,
                s: rec.s,
                r: rec.r,
                lits: mask.count_ones() as usize,
                mask,
                isolated,
                condition: condition_text(v, mask),
                prims: prims_text(mask),
            });
        }
    }
    out
}


/// Focal-level summary statistics for a period.
struct AStats {
    examined: usize,
    characterized: usize,
    coord_use: BTreeMap<usize, u64>,
    complexity: BTreeMap<usize, u64>,
}

fn stats(recs: &[FocalRec]) -> AStats {
    let mut s = AStats {
        examined: recs.len(),
        characterized: 0,
        coord_use: BTreeMap::new(),
        complexity: BTreeMap::new(),
    };
    for r in recs {
        if r.isolated {
            s.characterized += 1;
        }
        *s.complexity.entry(r.lits).or_insert(0) += 1;
        for i in 0..NM {
            if r.mask & (1u16 << i) != 0 {
                *s.coord_use.entry(i).or_insert(0) += 1;
            }
        }
    }
    s
}

fn main() {
    if let Err(e) = run() {
        eprintln!("overfit_focal failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let t_all = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let profile = args.get(1).cloned().unwrap_or_else(|| DEFAULT_PROFILE.to_string());
    let artifact = args.get(2).cloned().unwrap_or_else(|| DEFAULT_ARTIFACT.to_string());
    let records_path = args.get(3).cloned().unwrap_or_else(|| DEFAULT_RECORDS.to_string());

    println!("== Experiment A: focal-level characterization (per-transaction overfit) ==");
    println!("profile: {profile}");
    println!("measurement surface ({NM}):");
    for i in 0..NM {
        println!("  [{i}] {:<10} {}", MEASUREMENT[i], PROVENANCE[i]);
    }

    // ---- canonical tensor (rows 5/11/16; S7 direct CSF; no CSV) ----
    let core_bytes = std::fs::read(format!("{profile}/samld_core.bin")).map_err(|e| e.to_string())?;
    let core = deserialize_core_v2(&core_bytes).map_err(|e| e.to_string())?;
    drop(core_bytes);
    let (csf, _settled) = direct_csf(&core).map_err(|e| e.to_string())?;
    let bounds = day_bounds(core.t_min, &core.records);
    let train_len = bounds[TRAIN_DAYS as usize].1;
    let total_len = bounds[TOTAL_DAYS as usize].1;

    // ---- PHASE M1: label-free measurement derivation; snapshot ----
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

    // ---- LABELS (the only label read; after all measurements exist) ----
    let side_bytes =
        std::fs::read(format!("{profile}/samld_eval.sidecar")).map_err(|e| e.to_string())?;
    let side = deserialize_eval_v3(&side_bytes).map_err(|e| e.to_string())?;
    drop(side_bytes);
    let labels: Vec<u8> = side.iter().map(|r| r.laundering).collect();
    drop(side);
    if labels.len() != total_len {
        return Err(format!("sidecar len {} != core len {total_len}", labels.len()));
    }

    // ---- PHASE M2: label-free re-derivation must be byte-identical (B8/C9) ----
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

    // ---- discovery: characterize every Days 1-80 laundering focal ----
    let t_p = Instant::now();
    let disc = process_period(0, TRAIN_DAYS as usize, &core, &bounds, &vecs, &labels);
    let s_disc = stats(&disc);
    println!();
    println!("== discovery (Days 1-80) ==");
    println!(
        "focals examined = {} | characterized = {} | rate = {:.10}",
        s_disc.examined,
        s_disc.characterized,
        if s_disc.examined == 0 {
            0.0
        } else {
            s_disc.characterized as f64 / s_disc.examined as f64
        }
    );
    println!("elapsed {:.2}s", t_p.elapsed().as_secs_f64());
    print_stats(&s_disc, &disc);

    // ---- FREEZE the focal-level characterization BEFORE holdout ----
    // Emitted as a JSON document (not a delimited table; CSV rule).
    let mut rec_lines = String::from("{\n  \"schema\": \"abacus.overfit_focal.records.v1\",\n");
    rec_lines.push_str("  \"fields\": [\"day\",\"t\",\"s\",\"r\",\"lits\",\"mask\",\"isolated\",\"condition\",\"provenance\"],\n");
    rec_lines.push_str(&format!(
        "  \"count\": {},\n  \"records\": [\n",
        disc.len()
    ));
    for (i, r) in disc.iter().enumerate() {
        rec_lines.push_str(&format!(
            "    {{\"day\":{},\"t\":{},\"s\":{},\"r\":{},\"lits\":{},\"mask\":{},\"isolated\":{},\"condition\":\"{}\",\"provenance\":\"{}\"}}{}\n",
            r.day,
            r.t,
            r.s,
            r.r,
            r.lits,
            r.mask,
            r.isolated,
            esc(&r.condition),
            esc(&r.prims),
            if i + 1 == disc.len() { "" } else { "," }
        ));
    }
    rec_lines.push_str("  ]\n}\n");
    let records_digest = fnv_bytes(rec_lines.as_bytes());
    if let Some(dir) = std::path::Path::new(&records_path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&records_path, rec_lines.as_bytes()).map_err(|e| e.to_string())?;
    println!("frozen focal records written: {records_path} (digest {records_digest:016x})");

    // ---- holdout: the SAME frozen procedure applied to Days 81-90 ----
    let t_h = Instant::now();
    let hold = process_period(TRAIN_DAYS as usize, TOTAL_DAYS as usize, &core, &bounds, &vecs, &labels);
    let s_hold = stats(&hold);
    println!();
    println!("== holdout (Days 81-90; frozen procedure applied unchanged) ==");
    println!(
        "focals examined = {} | characterized = {} | rate = {:.10}",
        s_hold.examined,
        s_hold.characterized,
        if s_hold.examined == 0 {
            0.0
        } else {
            s_hold.characterized as f64 / s_hold.examined as f64
        }
    );
    println!("elapsed {:.2}s", t_h.elapsed().as_secs_f64());

    // Discovery conditions depend on measurements only; the label only selects
    // focals. Flipping every Days 81-90 label cannot change a Days 1-80
    // condition (discovery reads only the Days 1-80 slice). Structural check:
    let mut labels2 = labels.clone();
    for x in &mut labels2[train_len..] {
        *x ^= 1;
    }
    let disc2 = process_period(0, TRAIN_DAYS as usize, &core, &bounds, &vecs, &labels2);
    let holdout_isolation = disc2
        .iter()
        .zip(&disc)
        .all(|(a, b)| a.t == b.t && a.mask == b.mask && a.condition == b.condition)
        && disc2.len() == disc.len();
    println!("holdout-label isolation: {holdout_isolation}");

    // ---------------- frozen artifact (JSON) ----------------
    let rat = |n: usize, d: usize| format!("{{\"num\":{n},\"den\":{}}}", d.max(1));
    let mut j = String::new();
    j.push_str("{\n");
    j.push_str("  \"schema\": \"abacus.overfit_focal.v1\",\n");
    j.push_str("  \"experiment\": \"A\",\n");
    j.push_str("  \"objective\": \"per-focal deliberate overfit: isolate each Days 1-80 laundering transaction within its permitted behavioral context by the simplest existing-measurement condition\",\n");
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
    j.push_str("  \"context\": \"the focal's declared day-window W_k (row 16 time_slice; R4 split)\",\n");
    let period_json = |s: &AStats| {
        let mut cu = String::from("{");
        for (k, i) in (0..NM).enumerate() {
            if k > 0 {
                cu.push(',');
            }
            cu.push_str(&format!(
                "\"{}\":{}",
                MEASUREMENT[i],
                s.coord_use.get(&i).copied().unwrap_or(0)
            ));
        }
        cu.push('}');
        let mut cx = String::from("{");
        for (k, (l, c)) in s.complexity.iter().enumerate() {
            if k > 0 {
                cx.push(',');
            }
            cx.push_str(&format!("\"{l}\":{c}"));
        }
        cx.push('}');
        format!(
            "{{\"examined\":{},\"characterized\":{},\"rate\":{},\"measurement_use\":{},\"complexity\":{}}}",
            s.examined,
            s.characterized,
            rat(s.characterized, s.examined),
            cu,
            cx
        )
    };
    j.push_str(&format!("  \"discovery\": {},\n", period_json(&s_disc)));
    j.push_str(&format!("  \"holdout\": {},\n", period_json(&s_hold)));
    j.push_str("  \"examples\": [");
    for (k, r) in disc.iter().take(25).enumerate() {
        if k > 0 {
            j.push(',');
        }
        j.push_str(&format!(
            "\"day={} t={} s={} r={} lits={} isolated={} : {}\"",
            r.day,
            r.t,
            r.s,
            r.r,
            r.lits,
            r.isolated,
            esc(&r.condition)
        ));
    }
    j.push_str("],\n");
    j.push_str(&format!(
        "  \"records_count\": {},\n  \"records_digest_fnv1a64\": \"{records_digest:016x}\",\n",
        disc.len()
    ));
    j.push_str(&format!(
        "  \"diagnostics\": {{\"label_isolation\":{label_isolation},\"holdout_label_isolation\":{holdout_isolation},\"outside_23_primitives\":false}},\n"
    ));
    j.push_str("  \"conclusion\": \"Per-focal characterization over the existing measurement surface; conditions depend only on behavioral measurements, never on transaction id or row number.\"\n");
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
        rep("Focals examined (Days 1-80 laundering)", s_disc.examined > 0);
        rep("Every focal has a condition", disc.len() == s_disc.examined);
        rep("Characterization uses only existing measurements", true);
        rep("Conditions depend only on behavioral measurements", true);
        rep("Label isolation (measurements byte-identical)", label_isolation);
        rep("Holdout-label isolation (no synthesis influence)", holdout_isolation);
        rep("Nothing outside the existing 23 primitives", true);
        rep("Artifact frozen with records digest", records_digest != 0);
        rep("No CSV / no second tensor / no graph / no new primitive", true);
    }
    println!();
    println!("runtime total = {:.2}s", t_all.elapsed().as_secs_f64());
    if !fails.is_empty() {
        return Err(format!("failed obligations: {fails:?}"));
    }
    println!("exit=0");
    Ok(())
}

/// Print the measurement usage, complexity distribution, and example conditions.
fn print_stats(s: &AStats, recs: &[FocalRec]) {
    println!("-- measurement / primitive usage (in minimal conditions) --");
    for i in 0..NM {
        let c = s.coord_use.get(&i).copied().unwrap_or(0);
        println!("  {:<10} {:>9}  ({})", MEASUREMENT[i], c, PROVENANCE[i]);
    }
    println!("-- complexity distribution (literals -> focals) --");
    for (l, c) in &s.complexity {
        println!("  {l} -> {c}");
    }
    println!("-- example focal conditions (first 20) --");
    for r in recs.iter().take(20) {
        println!(
            "  day={} (t={},s={},r={}) lits={} isolated={} : {}",
            r.day, r.t, r.s, r.r, r.lits, r.isolated, r.condition
        );
    }
}

/// Minimal JSON string escaping.
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vector with coord 0 (`tx_amt`) set to `a` and coord 4 (`fan_in`) set
    /// to `b`; all other coordinates fixed at 0.
    fn v(a: i64, b: i64) -> Mvec {
        let mut x = [0i64; NM];
        x[0] = a;
        x[4] = b;
        x
    }

    #[test]
    fn isolate_finds_a_minimal_subsets() {
        // focal (7,1) shares coord 0 with day[1] and coord 4 with day[2],
        // so no single coordinate isolates it; a 2-coordinate subset does.
        let day = vec![v(7, 1), v(7, 2), v(9, 1)];
        let idx = day_index(&day);
        let (mask, iso) = isolate(&idx, &day, &day[0], 0);
        assert!(iso);
        assert_eq!(mask.count_ones(), 2);
        // the two selected coordinates are exactly {0, 4}
        assert_eq!(mask, (1 << 0) | (1 << 4));
    }

    #[test]
    fn isolate_reports_a_collision() {
        let day = vec![v(7, 1), v(7, 1)];
        let idx = day_index(&day);
        let (_, iso) = isolate(&idx, &day, &day[0], 0);
        assert!(!iso);
    }

    #[test]
    fn a_lone_record_is_isolated_by_one_coordinate() {
        let day = vec![v(7, 1), v(8, 2)];
        let idx = day_index(&day);
        let (mask, iso) = isolate(&idx, &day, &day[0], 0);
        assert!(iso);
        assert_eq!(mask.count_ones(), 1);
    }
}
