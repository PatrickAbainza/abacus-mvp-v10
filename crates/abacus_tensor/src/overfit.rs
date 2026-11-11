//! Overfit characterization measurement surface (Experiment A / Experiment B).
//!
//! **Governing sections (Golden Source Compliance Rule — every item cited).**
//!
//! | element | governing section |
//! |---|---|
//! | canonical tensor `T[t,s,r]`, exact values | derivation P1–P3, rows 5/11/16 |
//! | one temporal restriction per day-window | row 16 `time_slice`, P1 half-open; HCSF §4 |
//! | behavioral fact fields (`event_count`, `aggregate_amount`, `temporal_span`) | behavioral addendum B4 |
//! | `rin`/`sout`/`sin`/`rout`/`f_max` focal measurements | feature-vector spec §4 |
//! | `tx_k` set cardinality | feature-vector spec §5 |
//! | checked exact sums / differences | feature-vector spec §4A; `a_02`/`a_03` (P2/R5) |
//! | GraphBLAS `#5`,`#10`,`#11`,`#12`,`#13`,`#21`,`#22` | Addendum O rows 30–51 |
//! | six comparisons (`=,!=,>,>=,<,<=`) | semantic layer §4 |
//! | `AND`/`OR`/`NOT` composition | semantic §18; condition-surface addendum C5 |
//! | no label in any construction | P11; B8; C9 |
//! | chronological split `W_k = [t_min+(k-1)*86400, t_min+k*86400)` | rule-discovery addendum R4 |
//!
//! **This module is label-free (P11).** It never imports, names or reads the
//! eval sidecar. Each experiment owns its own label read.
//!
//! ```text
//! CSF T[t,s,r] --row16--> E_W (one day) --set projections--> DayMaps
//!   -> Mvec per canonical record  (the measurement surface, 9 positions)
//! ```
//!
//! No second tensor, no dense temporal tensor, no snapshot graph, no
//! adjacency/edge-list/node-table (Time rule). Every position is an existing
//! primitive read at an existing argument.

use std::collections::{BTreeMap, BTreeSet};

use crate::coo::entry::TxEntry;
use crate::csf::compress::CsfTensor;
use crate::csf::query::time_slice;
use crate::prims::{add, fnv_mix};
use crate::samld::core_bin::CoreRecord;
use crate::TensorError;

/// Number of measurement positions on the surface.
pub const NM: usize = 15;

/// The declared measurement names, in fixed positional order.
pub const MEASUREMENT: [&str; NM] = [
    "tx_amt",    // 0  T(t,s,r)                     canonical tensor value (P2)
    "pair_cnt",  // 1  |A_1(s,r)|                   #10 reduce_count (B4 event_count, FV s5 tx_k)
    "pair_vol",  // 2  Sigma amount over the pair   #12 reduce_sum (B4 aggregate_amount, FV s4A)
    "pair_span", // 3  max t - min t over the pair  a_03 TIME DIFFERENCE (B4 temporal_span)
    "fan_in",    // 4  |in(r)|                      #21/#22/#11 (FV s4 rin)
    "fan_out",   // 5  |out(s)|                     #5/#11 (FV s4 sout)
    "deg_sin",   // 6  |in(s)|                      #21/#11 (FV s4 sin)
    "deg_rout",  // 7  |out(r)|                     #5/#11 (FV s4 rout)
    "pair_max",  // 8  max entry over the pair      #13 reduce_max (FV s4 f_max)
    "pair_min",  // 9  min entry over the pair      #14 reduce_min (FV s4 f_min)
    "off_first", // 10 t - min t over the pair      a_03 TIME DIFFERENCE (event vs pair minimum)
    "tap_mean",  // 11 pair_vol / pair_cnt          deterministic integer div of exact aggregates (a_02/#10)
    "day_idx",   // 12 k, with t in W_k             R4 one-day window membership (set projection of t)
    "off_day",   // 13 t - min t over the day       a_03 TIME DIFFERENCE (event vs day first instant)
    "pair_first", // 14 min t over pair - min t over day  a_03 TIME DIFFERENCE (pair start vs day start)
];

/// The primitive/section provenance of each position (report only, no logic).
pub const PROVENANCE: [&str; NM] = [
    "row16 lookup T(t,s,r) (P2 canonical tensor value)",
    "#10 reduce_count / B4 event_count / FV s5 tx_k",
    "#12 reduce_sum / a_02 checked (B4 aggregate_amount, FV s4A)",
    "a_03 checked TIME DIFFERENCE (B4 temporal_span)",
    "#21 incoming_support + #22 compute_fan_in + #11 reduce_count_set (FV s4 rin)",
    "#5 vxm + #11 reduce_count_set (FV s4 sout)",
    "#21 incoming_support + #11 reduce_count_set (FV s4 sin)",
    "#5 vxm + #11 reduce_count_set (FV s4 rout)",
    "#13 reduce_max (FV s4 f_max)",
    "#14 reduce_min (FV s4 f_min)",
    "a_03 checked TIME DIFFERENCE event t vs pair minimum (B4 temporal_span family)",
    "integer division pair_vol/pair_cnt over exact a_02 aggregates (FV s4A mean)",
    "R4 window membership W_k = [t_min+(k-1)*86400, t_min+k*86400): set projection of t",
    "a_03 checked TIME DIFFERENCE event t vs day-window minimum t (B4 temporal_span family)",
    "a_03 checked TIME DIFFERENCE pair minimum t vs day-window minimum t (B4 temporal_span family)",
];


/// Declared seconds per day-window (R4).
pub const DAY_SECONDS: u64 = 86_400;
/// Declared day-window count of the profile (R4).
pub const TOTAL_DAYS: u32 = 90;
/// Declared training horizon (R4).
pub const TRAIN_DAYS: u32 = 80;


/// Per-day projected measurements over the day-window support `E_W`.
#[derive(Debug, Clone)]
pub struct DayMaps {
    /// `|E_W|` — number of canonical entries in the window (5A.7 reporting).
    pub entry_count: usize,
    /// `(t,s,r) -> T(t,s,r)` — the canonical settled value.
    pub coord_val: BTreeMap<(u64, u32, u32), i64>,
    /// `(s,r) -> |A_1(s,r)|` — B4 `event_count`.
    pub pair_cnt: BTreeMap<(u32, u32), u64>,
    /// `(s,r) -> Sigma amount` — B4 `aggregate_amount` (checked `a_02`).
    pub pair_vol: BTreeMap<(u32, u32), i64>,
    /// `(s,r) -> max t - min t` — B4 `temporal_span` (`a_03`).
    pub pair_span: BTreeMap<(u32, u32), u64>,
    /// `(s,r) -> max entry value` — FV §4 `f_max` (#13).
    pub pair_max: BTreeMap<(u32, u32), i64>,
    /// `(s,r) -> min entry value` — FV §4 `f_min` (#14).
    pub pair_min: BTreeMap<(u32, u32), i64>,
    /// `(s,r) -> min t` — the pair's first instant (basis of `off_first`).
    pub pair_lo: BTreeMap<(u32, u32), u64>,
    /// `r -> |in(r)|` — FV §4 `rin`.
    pub fan_in: BTreeMap<u32, u64>,
    /// `s -> |out(s)|` — FV §4 `sout`.
    pub fan_out: BTreeMap<u32, u64>,
    /// `min t` over the whole day-window — the window's first instant (basis of `off_day`/`pair_first`).
    pub day_min_t: u64,
}

/// Declared day-window record bounds (R4). `bounds[0]` is the empty pre-day
/// marker `(0,0)`; `bounds[k]` is the record slice of 1-based day `k`.
///
/// The profile records are in `(t,s,r)` order (P1), so every declared
/// half-open day-window is one **contiguous** slice located by `a_12`
/// `partition_point` — a partition, not a filter.
#[must_use]
pub fn day_bounds(t_min: u64, records: &[CoreRecord]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(TOTAL_DAYS as usize + 1);
    let mut prev = 0usize;
    for k in 0..=TOTAL_DAYS {
        let edge = t_min + u64::from(k) * DAY_SECONDS;
        let at = records.partition_point(|r| r.t < edge);
        out.push((prev, at));
        prev = at;
    }
    out
}

/// Derive the day-window projections from the canonical CSF (row 16 + exact
/// set projections). One restriction per day; the tensor is never
/// re-materialised (HCSF §3).
///
/// # Errors
/// [`TensorError::Range`] if a checked `a_02` pair aggregate leaves `V` (P2).
pub fn day_maps(csf: &CsfTensor, t_start: u64, t_end: u64) -> Result<DayMaps, TensorError> {
    let entries: Vec<TxEntry> = time_slice(csf, t_start, t_end).entries;
    let mut coord_val: BTreeMap<(u64, u32, u32), i64> = BTreeMap::new();
    let mut pair_cnt: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    let mut pair_vol: BTreeMap<(u32, u32), i64> = BTreeMap::new();
    let mut pair_lo: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    let mut pair_hi: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    let mut pair_max: BTreeMap<(u32, u32), i64> = BTreeMap::new();
    let mut pair_min: BTreeMap<(u32, u32), i64> = BTreeMap::new();
    // `A_one|_W` pair sets as exact sets (Set-operation rule).
    let mut in_pairs: BTreeSet<(u32, u32)> = BTreeSet::new(); // (receiver, sender) => |in(r)|
    let mut out_pairs: BTreeSet<(u32, u32)> = BTreeSet::new(); // (sender, receiver) => |out(s)|
    for e in &entries {
        coord_val.insert((e.t, e.s, e.r), e.value);
        let k = (e.s, e.r);
        *pair_cnt.entry(k).or_insert(0) += 1;
        let v = pair_vol.entry(k).or_insert(0);
        *v = add(*v, e.value, "overfit pair aggregate (B4 aggregate_amount, a_02)")?;
        let lo = pair_lo.entry(k).or_insert(e.t);
        if e.t < *lo {
            *lo = e.t;
        }
        let hi = pair_hi.entry(k).or_insert(e.t);
        if e.t > *hi {
            *hi = e.t;
        }
        let mx = pair_max.entry(k).or_insert(i64::MIN);
        if e.value > *mx {
            *mx = e.value;
        }
        let mn = pair_min.entry(k).or_insert(i64::MAX);
        if e.value < *mn {
            *mn = e.value;
        }
        in_pairs.insert((e.r, e.s));
        out_pairs.insert((e.s, e.r));
    }
    let mut fan_in: BTreeMap<u32, u64> = BTreeMap::new();
    for (r, _s) in &in_pairs {
        *fan_in.entry(*r).or_insert(0) += 1;
    }
    let mut fan_out: BTreeMap<u32, u64> = BTreeMap::new();
    for (s, _r) in &out_pairs {
        *fan_out.entry(*s).or_insert(0) += 1;
    }
    let mut pair_span: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for (k, lo) in &pair_lo {
        let hi = pair_hi[k];
        // a_03 TIME DIFFERENCE on exact epoch seconds (max >= min by build).
        pair_span.insert(*k, hi.saturating_sub(*lo));
    }
    Ok(DayMaps {
        entry_count: entries.len(),
        coord_val,
        pair_cnt,
        pair_vol,
        pair_span,
        pair_max,
        pair_min,
        pair_lo,
        fan_in,
        fan_out,
        day_min_t: entries.first().map_or(t_start, |e| e.t),
    })
}

/// One measurement vector (the canonical measurement surface instantiation).

/// The measurement vector of one canonical record within its day-window
/// context. Absent keys read `0` (FV §1: absent entries read as `0`).
///
/// `day_idx` is the 1-based window index `k` with `t ∈ W_k` (R4) — a set
/// projection of the native temporal coordinate, not a row number.
#[must_use]
pub fn vector_for(m: &DayMaps, t: u64, s: u32, r: u32, day_idx: u32) -> Mvec {
    let amt = m.coord_val.get(&(t, s, r)).copied().unwrap_or(0);
    let k = (s, r);
    let cnt = m.pair_cnt.get(&k).copied().unwrap_or(0);
    let vol = m.pair_vol.get(&k).copied().unwrap_or(0);
    let lo = m.pair_lo.get(&k).copied().unwrap_or(t);
    [
        amt,
        cnt as i64,
        vol,
        m.pair_span.get(&k).copied().unwrap_or(0) as i64,
        m.fan_in.get(&r).copied().unwrap_or(0) as i64,
        m.fan_out.get(&s).copied().unwrap_or(0) as i64,
        m.fan_in.get(&s).copied().unwrap_or(0) as i64,
        m.fan_out.get(&r).copied().unwrap_or(0) as i64,
        m.pair_max.get(&k).copied().unwrap_or(0),
        m.pair_min.get(&k).copied().unwrap_or(0),
        // a_03 TIME DIFFERENCE: event t vs the pair's first instant (>= 0 by build).
        t.saturating_sub(lo) as i64,
        // deterministic integer mean of the pair's exact aggregates (cnt >= 1 here).
        vol.checked_div(cnt.max(1) as i64).unwrap_or(0),
        i64::from(day_idx),
        // a_03 TIME DIFFERENCE: event t vs the day-window's first instant.
        t.saturating_sub(m.day_min_t) as i64,
        // a_03 TIME DIFFERENCE: the pair's first instant vs the day-window's first instant.
        lo.saturating_sub(m.day_min_t) as i64,
    ]
}

/// The measurement vectors of one day's record slice, aligned positionally
/// with `recs` (so index `i` pairs with `recs[i]` and with the sidecar).
/// `day_idx` is the 1-based window index `k` (R4).
///
/// # Errors
/// [`TensorError::Range`] on a checked aggregate breach (P2).
pub fn day_vectors(
    csf: &CsfTensor,
    t_start: u64,
    t_end: u64,
    recs: &[CoreRecord],
    day_idx: u32,
) -> Result<Vec<Mvec>, TensorError> {
    let m = day_maps(csf, t_start, t_end)?;
    Ok(recs
        .iter()
        .map(|r| vector_for(&m, r.t, r.s, r.r, day_idx))
        .collect())
}

/// Day `k` (0-based) record slice of the whole profile.
#[must_use]
pub fn day_slice(bounds: &[(usize, usize)], k: usize) -> (usize, usize) {
    bounds[k + 1]
}

/// FNV-1a/64 over a measurement-vector sequence (deterministic snapshot of the
/// label-free derivation; `a_06` `d_fnv`). Used to prove label isolation (P11).
#[must_use]
pub fn fnv_vecs(v: &[Mvec]) -> u64 {
    let mut h = crate::composites::FNV_OFFSET_BASIS;
    for row in v {
        for x in row {
            for b in x.to_le_bytes() {
                h = fnv_mix(h, b);
            }
        }
    }
    h
}

/// FNV-1a/64 over a byte slice (re-exported for the experiments' digests).
#[must_use]
pub fn fnv_bytes(b: &[u8]) -> u64 {
    crate::composites::fnv1a64(b)
}

/// The coordinates present in a mask (ascending).
#[must_use]
pub fn mask_lits(mask: u16) -> Vec<usize> {
    (0..NM).filter(|i| mask & (1u16 << i) != 0).collect()
}

pub type Mvec = [i64; NM];
