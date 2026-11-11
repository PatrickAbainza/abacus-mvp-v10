//! Behavioral relations and declarative predicates over the canonical CSF.
//!
//! Governing addendum: `docs/golden_source_behavioral_rule_boundary_addendum.md`
//! (B1–B11). This module realizes **only** what that addendum derives:
//!
//! ```text
//! CSF tensor T[t,s,r]                       P3/P4/P6, rows 5/11
//!   ↓ row 16 `time_slice`  (one restriction) P1 half-open window, HCSF §4
//!   ↓ exact set operations over the windowed canonical support
//! BehavioralRelation                        B4
//!   ↓ four boolean atoms + one conjunction  B6, semantic §7.6/§18, `a_17`
//! TRUE / FALSE
//! ```
//!
//! Constraints honoured here:
//!
//! * The canonical tensor is `T[t,s,r]` and is **never** re-materialised
//!   (HCSF §3). No second tensor representation, no dense temporal tensor,
//!   no snapshot graph (Time rule). `t` stays an exact coordinate until
//!   `temporal_span` is computed.
//! * **No graph is constructed.** The only groupings are exact `a_16.2}`
//!   ordered maps keyed by tensor coordinates, held for the duration of one
//!   window (HCSF §2/§8/§10, Set-operation rule).
//! * All value arithmetic is exact checked micro-unit arithmetic via the
//!   existing primitives `a_02`/`a_03` (P2/R5) — never float, never wrapping.
//! * **No label is read anywhere in this module** (P11, B8). The scorer
//!   lives in the caller and runs after the predicate results exist.
//! * No rule engine: no DRL, Rete, Phreak, agenda, salience, parsing or
//!   chaining. The only "engine" behaviour is four comparisons and one `∧`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::csf::compress::CsfTensor;
use crate::csf::query::time_slice;
use crate::prims::{add, sub};
use crate::tensor::coordinate::Coord;
use crate::TensorError;

/// Dictionary-backed entity resolver: `(mode, code) -> account identifier`
/// (row 12 dictionary; row 17 image dictionaries). `sender` and `receiver`
/// are kept as distinct slices because the two coordinate sets `S` and `R`
/// of A1 are distinct mathematical objects even when a caller supplies the
/// same table for both (which is what the synthetic probe does).
#[derive(Debug, Clone, Copy)]
pub struct Entities<'a> {
    /// Sender dictionary, ascending `prec_dict` (code order).
    pub senders: &'a [String],
    /// Receiver dictionary, ascending `prec_dict` (code order).
    pub receivers: &'a [String],
}

impl<'a> Entities<'a> {
    /// Sender account for a sender code (P1 secondary mode).
    #[must_use]
    pub fn sender(&self, code: u32) -> &'a str {
        self.senders.get(code as usize).map_or("<oob-s>", String::as_str)
    }

    /// Receiver account for a receiver code (P1 tertiary mode).
    #[must_use]
    pub fn receiver(&self, code: u32) -> &'a str {
        self.receivers.get(code as usize).map_or("<oob-r>", String::as_str)
    }
}

/// Half-open temporal window plus the hop gap bound `Δ` (B4, B9; P1;
/// semantic §7.5 parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Inclusive window start.
    pub t_start: u64,
    /// Exclusive window end.
    pub t_end: u64,
    /// Maximum `t2 − t1` admitted between the two legs of a hop-2 path.
    pub delta: u64,
}

/// The behavioral relation fact (B4). Every field is a set operation or a
/// checked primitive over tensor state; see the addendum table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BehavioralRelation {
    /// Sender account of the first leg.
    pub source: String,
    /// Receiver account of the last leg.
    pub target: String,
    /// Number of legs (1 or 2; hop bound `k = 2`).
    pub hop: u32,
    /// The restriction actually used, `(t_start, t_end)`.
    pub window: (u64, u64),
    /// `|A_hop(source,target)|` — admitted-entry cardinality (FV §5 `tx_k`).
    pub event_count: u64,
    /// Exact checked micro-unit sum over the admitted entries (FV §4A/§6A).
    pub aggregate_amount: i64,
    /// `max t − min t` over the admitted entries, exact epoch seconds.
    pub temporal_span: u64,
}

/// A relation together with the exact set of canonical coordinates admitted
/// for it (`A_hop`). Exposed because the scorer needs the admitted set; it is
/// a set, never a list with duplicates (Set-operation rule).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedRelation {
    /// The relation fact.
    pub relation: BehavioralRelation,
    /// Exact admitted-coordinate set `A_hop(source,target)`.
    pub admitted: BTreeSet<(Coord, i64)>,
}

/// Declared predicate parameters `(N, X, Δ_span, k_min)` (B6; semantic
/// §7.5/§19). `min_amount_micro` is in exact micro-units (P2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thresholds {
    /// `N` for `event_count >= N`.
    pub min_events: u64,
    /// `X` for `aggregate_amount >= X` (exact micro-units).
    pub min_amount_micro: i64,
    /// `Δ_span` for `temporal_span <= Δ_span` (exact epoch seconds).
    pub max_span_secs: u64,
    /// `k_min` for `hop >= k_min`.
    pub min_hop: u32,
}

impl Thresholds {
    /// The four boolean atoms `P1..P4` in declaration order (B6).
    #[must_use]
    pub fn atoms(&self, r: &BehavioralRelation) -> [bool; 4] {
        [
            r.event_count >= self.min_events,
            r.aggregate_amount >= self.min_amount_micro,
            r.temporal_span <= self.max_span_secs,
            r.hop >= self.min_hop,
        ]
    }

    /// The one compound condition `P1 ∧ P2 ∧ P3 ∧ P4` (B6, `a_17`
    /// short-circuit `∧`; semantic §18).
    #[must_use]
    pub fn evaluate(&self, r: &BehavioralRelation) -> bool {
        let [a, b, c, d] = self.atoms(r);
        a && b && c && d
    }

    /// Compound result vector, aligned with the relation vector.
    #[must_use]
    pub fn evaluate_all(&self, rels: &[DerivedRelation]) -> Vec<bool> {
        rels.iter().map(|d| self.evaluate(&d.relation)).collect()
    }
}

/// Aggregate an admitted set into the seven relation fields (B4 table).
fn finish(
    source: &str,
    target: &str,
    hop: u32,
    w: Window,
    admitted: BTreeSet<(Coord, i64)>,
) -> Result<DerivedRelation, TensorError> {
    let mut sum: i64 = 0;
    let (mut t_min, mut t_max) = (u64::MAX, 0u64);
    for (c, v) in &admitted {
        // a_02 checked add — P2/R5; never wraps, never saturates.
        sum = add(sum, *v, "behavioral aggregate_amount (B4)")?;
        t_min = t_min.min(c.t);
        t_max = t_max.max(c.t);
    }
    // a_03 checked sub — P2; TIME DIFFERENCE on exact epoch seconds.
    let span = sub(t_max as i64, t_min as i64, "behavioral temporal_span (B4)")?;
    Ok(DerivedRelation {
        relation: BehavioralRelation {
            source: source.to_string(),
            target: target.to_string(),
            hop,
            window: (w.t_start, w.t_end),
            event_count: admitted.len() as u64,
            aggregate_amount: sum,
            temporal_span: span as u64,
        },
        admitted,
    })
}

/// B4 — derive every behavioral relation of hop 1 and hop 2 over one window,
/// **directly from CSF tensor state**.
///
/// * The windowed canonical support `E_W` is obtained by exactly one row-16
///   restriction of the existing CSF time hierarchy (`a_12` partition
///   points) and is then **reused** by both hops (HCSF §4/§5/§7/§8/§11).
///   The tensor is not re-materialised (HCSF §3).
/// * Hop 1 groups `E_W` by the ordered pair `(sender, receiver)`; hop 2
///   composes the hop-1 grouping through the intermediate `B` under the
///   temporal predicate `t1 ≤ t2 ∧ t2 − t1 ≤ Δ` (B4).
/// * Groupings are exact `a_16.2` ordered maps keyed by tensor coordinates
///   and are discarded when this function returns (HCSF §8). **No graph,
///   edge list, adjacency structure or node table is built or retained.**
/// * Output is sorted by `(hop, source, target)` for deterministic
///   comparison (P1/P12).
///
/// # Errors
/// [`TensorError::Range`] if any exact sum or difference of the aggregates
/// leaves `V` (P2), propagated fail-closed (`R_5`).
pub fn derive_relations(
    csf: &CsfTensor,
    ents: &Entities<'_>,
    w: Window,
) -> Result<Vec<DerivedRelation>, TensorError> {
    // Row 16: `T|_W`. One restriction, reused below.
    let entries = time_slice(csf, w.t_start, w.t_end).entries;

    // --- hop 1: group the windowed support by ordered pair (s, r) ---
    // `a_16.2` ordered map; key is the pair of entity ids, value the exact
    // admitted set `A_1(A,C)`.
    let mut one: BTreeMap<(String, String), BTreeSet<(Coord, i64)>> = BTreeMap::new();
    // Second projection: incoming legs keyed by receiver entity, used only
    // to enumerate the intermediate `B` of a hop-2 path.
    let mut by_receiver: BTreeMap<String, Vec<(u64, String, Coord, i64)>> = BTreeMap::new();
    for e in &entries {
        let s = ents.sender(e.s);
        let r = ents.receiver(e.r);
        one.entry((s.to_string(), r.to_string()))
            .or_default()
            .insert((e.coord(), e.value));
        by_receiver
            .entry(r.to_string())
            .or_default()
            .push((e.t, s.to_string(), e.coord(), e.value));
    }

    let mut out: Vec<DerivedRelation> = Vec::with_capacity(one.len());
    for ((s, r), admitted) in one {
        out.push(finish(&s, &r, 1, w, admitted)?);
    }

    // --- hop 2: A_2(A,C) per B4, the exact union over witnessing legs ---
    // Iterate every windowed entry as the *second* leg (t2, B, C) and match
    // it against the incoming legs of the same `B`. `t1 <= t2` and
    // `t2 - t1 <= Δ` are exact `u64` comparisons on the raw coordinates;
    // no timestamp is bucketed (Time rule).
    let mut two: BTreeMap<(String, String), BTreeSet<(Coord, i64)>> = BTreeMap::new();
    for e2 in &entries {
        let b = ents.sender(e2.s); // the intermediate is the *sender* of leg 2
        let Some(incoming) = by_receiver.get(b) else {
            continue;
        };
        for (t1, a, c1, v1) in incoming {
            if *t1 > e2.t {
                continue; // t1 <= t2 required
            }
            if e2.t - *t1 > w.delta {
                continue; // t2 - t1 <= Δ required
            }
            let c = ents.receiver(e2.r);
            let set = two
                .entry((a.clone(), c.to_string()))
                .or_default();
            set.insert((*c1, *v1));
            set.insert((e2.coord(), e2.value));
        }
    }
    for ((a, c), admitted) in two {
        out.push(finish(&a, &c, 2, w, admitted)?);
    }

    out.sort_by(|x, y| {
        (x.relation.hop, &x.relation.source, &x.relation.target).cmp(&(
            y.relation.hop,
            &y.relation.source,
            &y.relation.target,
        ))
    });
    Ok(out)
}

/// One raw event `(t, sender code, receiver code, amount_micro)` — the
/// P4 ingestion shape, exactly as it arrives from the event stream.
pub type RawEvent = (u64, u32, u32, i64);

/// Reference-side key: the ordered entity pair `(source, target)`.
type RefPair = (String, String);

/// Reference-side admitted-set element `(t, s, r, value)`.
type RefEntry = (u64, u32, u32, i64);

/// B7 — the **independent reference calculation**.
///
/// Deliberately naive and deliberately blind to the tensor: plain nested
/// loops over the raw event list, ordinary `HashMap`/`HashSet`
/// accumulators, no CSF, no fiber arrays, no `time_slice`, no shared
/// helper with [`derive_relations`]. It exists so that agreement is
/// evidence rather than tautology (B7).
///
/// `events` must already be restricted to the window; the reference filter
/// is applied by the caller and asserted equal to the row-16 restriction.
pub fn reference_relations(
    events: &[RawEvent],
    ents: &Entities<'_>,
    w: Window,
) -> Result<Vec<DerivedRelation>, TensorError> {
    // Window restriction by ordinary loop (the reference's own filter, B11.2).
    let win: Vec<&RawEvent> = events
        .iter()
        .filter(|(t, _, _, _)| *t >= w.t_start && *t < w.t_end)
        .collect();

    // hop 1 — count/sum/span per ordered pair, with ordinary maps.
    let mut cnt1: HashMap<RefPair, u64> = HashMap::new();
    let mut sum1: HashMap<RefPair, i64> = HashMap::new();
    let mut span1: HashMap<RefPair, (u64, u64)> = HashMap::new();
    let mut adm1: HashMap<RefPair, HashSet<RefEntry>> = HashMap::new();
    for (t, s, r, v) in &win {
        let k = (ents.sender(*s).to_string(), ents.receiver(*r).to_string());
        *cnt1.entry(k.clone()).or_insert(0) += 1;
        *sum1.entry(k.clone()).or_insert(0) = add(
            *sum1.get(&k).unwrap_or(&0),
            *v,
            "reference hop-1 amount (B7)",
        )?;
        let e = span1.entry(k.clone()).or_insert((*t, *t));
        e.0 = e.0.min(*t);
        e.1 = e.1.max(*t);
        adm1.entry(k).or_default().insert((*t, *s, *r, *v));
    }

    // hop 2 — a plain double loop over the window's events, no index.
    let mut cnt2: HashMap<RefPair, u64> = HashMap::new();
    let mut sum2: HashMap<RefPair, i64> = HashMap::new();
    let mut span2: HashMap<RefPair, (u64, u64)> = HashMap::new();
    let mut adm2: HashMap<RefPair, HashSet<RefEntry>> = HashMap::new();
    for (t1, s1, r1, v1) in &win {
        let a = ents.sender(*s1);
        let b = ents.receiver(*r1);
        for (t2, s2, r2, v2) in &win {
            if ents.sender(*s2) != b {
                continue; // legs must meet at the same entity B
            }
            if *t1 > *t2 || t2 - t1 > w.delta {
                continue;
            }
            let k = (a.to_string(), ents.receiver(*r2).to_string());
            let e = adm2.entry(k.clone()).or_default();
            e.insert((*t1, *s1, *r1, *v1));
            e.insert((*t2, *s2, *r2, *v2));
        }
    }
    for (k, set) in &adm2 {
        let mut sum: i64 = 0;
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for (t, _, _, v) in set {
            sum = add(sum, *v, "reference hop-2 amount (B7)")?;
            lo = lo.min(*t);
            hi = hi.max(*t);
        }
        *cnt2.entry(k.clone()).or_insert(0) += set.len() as u64;
        sum2.insert(k.clone(), sum);
        span2.insert(k.clone(), (lo, hi));
    }

    // Assemble in the same canonical order as the Abacus path.
    let mut out: Vec<DerivedRelation> = Vec::new();
    for (k, set) in &adm1 {
        out.push(DerivedRelation {
            relation: BehavioralRelation {
                source: k.0.clone(),
                target: k.1.clone(),
                hop: 1,
                window: (w.t_start, w.t_end),
                event_count: cnt1[k],
                aggregate_amount: sum1[k],
                temporal_span: sub(
                    span1[k].1 as i64,
                    span1[k].0 as i64,
                    "reference hop-1 span (B7)",
                )? as u64,
            },
            admitted: set.iter().map(|(t, s, r, v)| (Coord::new(*t, *s, *r), *v)).collect(),
        });
    }
    for (k, set) in &adm2 {
        out.push(DerivedRelation {
            relation: BehavioralRelation {
                source: k.0.clone(),
                target: k.1.clone(),
                hop: 2,
                window: (w.t_start, w.t_end),
                event_count: cnt2[k],
                aggregate_amount: sum2[k],
                temporal_span: sub(
                    span2[k].1 as i64,
                    span2[k].0 as i64,
                    "reference hop-2 span (B7)",
                )? as u64,
            },
            admitted: set.iter().map(|(t, s, r, v)| (Coord::new(*t, *s, *r), *v)).collect(),
        });
    }
    out.sort_by(|x, y| {
        (x.relation.hop, &x.relation.source, &x.relation.target).cmp(&(
            y.relation.hop,
            &y.relation.source,
            &y.relation.target,
        ))
    });
    Ok(out)
}
