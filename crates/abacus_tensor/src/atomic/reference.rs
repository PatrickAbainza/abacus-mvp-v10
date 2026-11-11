//! Reference (oracle) implementation of the four atomic operators over
//! plain entry supports. Slow, obviously correct, deterministic.
//!
//! Presence: tensor entries use net-value semantics. Ordering:
//! (coordinate, lowest locator).

use super::outcome::{Indeterminate, OpReason, Outcome};
use super::types::{Entry, Mode, Monoid, Support};

/// A coordinate predicate: range, set membership, or exclusion.
/// Half-open windows `[lo, hi)`.
#[derive(Debug, Clone, Default)]
pub struct Predicate {
    /// Mode-0 half-open window `[lo, hi)`. `None` = no constraint.
    pub window: Option<(u64, u64)>,
    /// Allowed mode-1 keys. `None` = all.
    pub keys1: Option<Vec<u32>>,
    /// Allowed mode-2 keys. `None` = all.
    pub keys2: Option<Vec<u32>>,
    /// Excluded coordinates (exact matches).
    pub exclude: Vec<super::types::Coord3>,
}

impl Predicate {
    /// Unconstrained predicate (keeps everything).
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }
    /// Half-open window predicate on the leading mode.
    #[must_use]
    pub fn window(lo: u64, hi: u64) -> Self {
        Self { window: Some((lo, hi)), ..Self::default() }
    }
    /// True iff the entry satisfies the predicate.
    #[must_use]
    pub fn admits(&self, e: &Entry) -> bool {
        if let Some((lo, hi)) = self.window {
            if e.coord.t < lo || e.coord.t >= hi {
                return false;
            }
        }
        if let Some(keys) = &self.keys1 {
            if !keys.contains(&e.coord.s) {
                return false;
            }
        }
        if let Some(keys) = &self.keys2 {
            if !keys.contains(&e.coord.r) {
                return false;
            }
        }
        if self.exclude.contains(&e.coord) {
            return false;
        }
        true
    }
}

/// Explicit map parameters: thresholds and baselines, never hidden globals.
#[derive(Debug, Clone, Default)]
pub struct Theta {
    /// Integer threshold applied by the map function, if any.
    pub threshold: Option<i64>,
    /// Pinned baseline descriptor, if any.
    pub baseline: Option<Baseline>,
}

/// Pinned baseline descriptor: values plus provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Baseline {
    /// Baseline mean in caller-documented fixed point.
    pub mean: i64,
    /// Baseline scale in caller-documented fixed point.
    pub scale: i64,
    /// Version identifier (opaque to the operator).
    pub version: String,
    /// As-of timestamp (epoch seconds).
    pub as_of: u64,
}

/// Which support a merge covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coverage {
    /// Aligned coordinates present in both supports.
    Intersection,
    /// Aligned coordinates present in either (missing side = bottom).
    Union,
}

/// `restrict(X, p)`: keep entries whose coordinates satisfy `p`.
#[must_use]
pub fn restrict(support: &Support, p: &Predicate) -> Support {
    let entries: Vec<Entry> = support.entries.iter().filter(|e| p.admits(e)).cloned().collect();
    Support::from_canonical(entries)
}

/// Map function: coordinates + value + params -> new value or bottom.
/// Integer-only. `None` drops the entry.
pub type MapFn = fn(u64, u32, u32, i64, &Theta) -> Option<i64>;

/// `map(X, f, theta)`: pointwise `f`. Bottom outputs dropped.
#[must_use]
pub fn map(support: &Support, f: MapFn, theta: &Theta) -> Support {
    let mut out = Vec::new();
    for e in &support.entries {
        if let Some(v) = f(e.coord.t, e.coord.s, e.coord.r, e.value, theta) {
            if v != 0 {
                out.push(Entry::new(e.coord.t, e.coord.s, e.coord.r, v, e.locators.clone()));
            }
        }
    }
    Support::from_canonical(out)
}

/// Merge function: (left bottom-or-value, right bottom-or-value) -> combined.
pub type MergeFn = fn(Option<i64>, Option<i64>) -> Option<i64>;
/// Checked integer addition of two values.
fn checked_add(a: i64, b: i64) -> Result<i64, Indeterminate> {
    a.checked_add(b)
        .ok_or_else(|| Indeterminate::new(OpReason::Overflow, "merge: i64 addition overflow"))
}

/// `merge(X, Y, combine, coverage)`: combine on aligned coordinates.
/// Union passes bottom (`None`) for a missing side. Locator sets union.
pub fn merge(
    left: &Support,
    right: &Support,
    combine: MergeFn,
    coverage: Coverage,
) -> Outcome<Support> {
    let (mut i, mut j) = (0usize, 0usize);
    let (a, b) = (&left.entries, &right.entries);
    let mut out = Vec::new();
    while i < a.len() || j < b.len() {
        let order = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) => Some(x.coord.cmp(&y.coord)),
            (Some(_), None) => Some(std::cmp::Ordering::Less),
            (None, Some(_)) => Some(std::cmp::Ordering::Greater),
            (None, None) => None,
        };
        let Some(ord) = order else { break };
        match ord {
            std::cmp::Ordering::Less => {
                if coverage == Coverage::Union {
                    if let Some(v) = combine(Some(a[i].value), None) {
                        if v != 0 {
                            out.push(Entry::new(a[i].coord.t, a[i].coord.s, a[i].coord.r, v, a[i].locators.clone()));
                        }
                    }
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if coverage == Coverage::Union {
                    if let Some(v) = combine(None, Some(b[j].value)) {
                        if v != 0 {
                            out.push(Entry::new(b[j].coord.t, b[j].coord.s, b[j].coord.r, v, b[j].locators.clone()));
                        }
                    }
                }
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if let Some(v) = combine(Some(a[i].value), Some(b[j].value)) {
                    if v != 0 {
                        out.push(Entry::new(
                            a[i].coord.t, a[i].coord.s, a[i].coord.r, v,
                            a[i].locators.union(&b[j].locators),
                        ));
                    }
                }
                i += 1;
                j += 1;
            }
        }
    }
    Outcome::Determinate(Support::from_canonical(out))
}

/// Merge with checked-addition conflict semantics; overflow -> Indeterminate.
pub fn merge_add(left: &Support, right: &Support, coverage: Coverage) -> Outcome<Support> {
    let (mut i, mut j) = (0usize, 0usize);
    let (a, b) = (&left.entries, &right.entries);
    let mut out = Vec::new();
    while i < a.len() || j < b.len() {
        let order = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) => Some(x.coord.cmp(&y.coord)),
            (Some(_), None) => Some(std::cmp::Ordering::Less),
            (None, Some(_)) => Some(std::cmp::Ordering::Greater),
            (None, None) => None,
        };
        let Some(ord) = order else { break };
        match ord {
            std::cmp::Ordering::Less => {
                if coverage == Coverage::Union {
                    out.push(a[i].clone());
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if coverage == Coverage::Union {
                    out.push(b[j].clone());
                }
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                match checked_add(a[i].value, b[j].value) {
                    Ok(v) => {
                        if v != 0 {
                            out.push(Entry::new(
                                a[i].coord.t, a[i].coord.s, a[i].coord.r, v,
                                a[i].locators.union(&b[j].locators),
                            ));
                        }
                    }
                    Err(ind) => return Outcome::Indeterminate(ind),
                }
                i += 1;
                j += 1;
            }
        }
    }
    Outcome::Determinate(Support::from_canonical(out))
}
/// Folded group row after dropping `modes`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducedRow {
    /// Remaining mode-0 coordinate (`None` = folded).
    pub t: Option<u64>,
    /// Remaining mode-1 coordinate.
    pub s: Option<u32>,
    /// Remaining mode-2 coordinate.
    pub r: Option<u32>,
    /// Folded value; `None` = bottom (empty min/max/any).
    pub value: Option<i64>,
    /// Witness locator (lowest) for min/max/any ties.
    pub witness: Option<u32>,
}

/// `reduce(X, modes, monoid)`: fold `modes` with a monoid.
/// Identities: sum = 0, count = 0, min/max/any = bottom.
/// Min/max ties broken by lowest locator. Overflow -> Indeterminate.
pub fn reduce(support: &Support, modes: &[Mode], monoid: Monoid) -> Outcome<Vec<ReducedRow>> {
    use std::collections::BTreeMap;
    let fold_t = modes.contains(&Mode::Mode0);
    let fold_s = modes.contains(&Mode::Mode1);
    let fold_r = modes.contains(&Mode::Mode2);
    let mut groups: BTreeMap<(Option<u64>, Option<u32>, Option<u32>), Vec<&Entry>> = BTreeMap::new();
    for e in &support.entries {
        let key = (
            if fold_t { None } else { Some(e.coord.t) },
            if fold_s { None } else { Some(e.coord.s) },
            if fold_r { None } else { Some(e.coord.r) },
        );
        groups.entry(key).or_default().push(e);
    }
    let mut out = Vec::new();
    for ((t, s, r), members) in groups {
        let row = match monoid {
            Monoid::Sum => {
                let mut acc = 0i64;
                let mut witness: Option<u32> = None;
                for m in &members {
                    acc = match acc.checked_add(m.value) {
                        Some(v) => v,
                        None => {
                            return Outcome::Indeterminate(Indeterminate::new(
                                OpReason::Overflow,
                                "reduce sum: i64 addition overflow",
                            ));
                        }
                    };
                    witness = match (witness, m.locators.lowest()) {
                        (None, l) => l,
                        (w, None) => w,
                        (Some(w), Some(l)) => Some(w.min(l)),
                    };
                }
                ReducedRow { t, s, r, value: Some(acc), witness }
            }
            Monoid::Count => {
                let witness = members.iter().filter_map(|m| m.locators.lowest()).min();
                ReducedRow { t, s, r, value: Some(members.len() as i64), witness }
            }
            Monoid::Min => {
                let mut best: Option<&Entry> = None;
                for m in &members {
                    best = match best {
                        None => Some(m),
                        Some(b) => {
                            if (m.value, m.locators.lowest()) < (b.value, b.locators.lowest()) {
                                Some(m)
                            } else {
                                Some(b)
                            }
                        }
                    };
                }
                let b = best.expect("nonempty group");
                ReducedRow { t, s, r, value: Some(b.value), witness: b.locators.lowest() }
            }
            Monoid::Max => {
                let mut best: Option<&Entry> = None;
                for m in &members {
                    best = match best {
                        None => Some(m),
                        Some(b) => {
                            let km = (m.value, std::cmp::Reverse(m.locators.lowest()));
                            let kb = (b.value, std::cmp::Reverse(b.locators.lowest()));
                            if km > kb { Some(m) } else { Some(b) }
                        }
                    };
                }
                let b = best.expect("nonempty group");
                ReducedRow { t, s, r, value: Some(b.value), witness: b.locators.lowest() }
            }
            Monoid::Any => {
                let witness = members.iter().filter_map(|m| m.locators.lowest()).min();
                ReducedRow { t, s, r, value: Some(1), witness }
            }
        };
        if monoid == Monoid::Sum && row.value == Some(0) {
            continue;
        }
        out.push(row);
    }
    Outcome::Determinate(out)
}

/// Scalar identities on empty input when all modes are folded.
#[must_use]
pub const fn empty_identity(monoid: Monoid) -> Option<i64> {
    match monoid {
        Monoid::Sum | Monoid::Count => Some(0),
        Monoid::Min | Monoid::Max | Monoid::Any => None,
    }
}
/// Pair seek over a single-key-major ordered row (decision a):
/// scan/filter composition. `next_after` uses STRICT `t > tau`
/// (equal timestamps excluded; `docs/OPERATOR_CONTRACTS.md` s4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRef {
    /// Entry time.
    pub t: u64,
    /// Opposite-end key.
    pub peer: u32,
    /// Locator.
    pub locator: u32,
}

/// Earliest entry of the fixed pair, if any.
#[must_use]
pub fn pair_earliest(row: &[RowRef], want_peer: u32) -> Option<RowRef> {
    row.iter().filter(|e| e.peer == want_peer).min_by_key(|e| (e.t, e.locator)).copied()
}

/// Latest entry of the fixed pair, if any.
#[must_use]
pub fn pair_latest(row: &[RowRef], want_peer: u32) -> Option<RowRef> {
    row.iter().filter(|e| e.peer == want_peer).max_by_key(|e| (e.t, e.locator)).copied()
}

/// Earliest entry of the fixed pair with `t > tau` (strict), if any.
#[must_use]
pub fn pair_next_after(row: &[RowRef], want_peer: u32, tau: u64) -> Option<RowRef> {
    row.iter()
        .filter(|e| e.peer == want_peer && e.t > tau)
        .min_by_key(|e| (e.t, e.locator))
        .copied()
}

/// Support-level pair seek: earliest / latest / next-after-tau for a fixed
/// (s, r) pair. Scan/filter composition; no O(log n) claim.
/// Tie-break: lowest locator.
#[must_use]
pub fn support_pair_seek(
    support: &Support,
    s: u32,
    r: u32,
    tau: Option<u64>,
    latest: bool,
) -> Option<Entry> {
    let mut best: Option<&Entry> = None;
    for e in &support.entries {
        if e.coord.s != s || e.coord.r != r {
            continue;
        }
        if let Some(tau) = tau {
            if e.coord.t <= tau {
                continue;
            }
        }
        best = match best {
            None => Some(e),
            Some(b) => {
                let better = if latest {
                    (e.coord.t, e.locators.lowest()) > (b.coord.t, b.locators.lowest())
                } else {
                    (e.coord.t, e.locators.lowest()) < (b.coord.t, b.locators.lowest())
                };
                if better { Some(e) } else { Some(b) }
            }
        };
    }
    best.cloned()
}
