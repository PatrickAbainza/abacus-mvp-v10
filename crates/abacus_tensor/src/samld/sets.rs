//! Set-based selection + correspondence (set-operation rule, §S9).
//! Exact set ops (union/inter/difference/subset/member) over BTreeSet;
//! no ad-hoc reconstruction. Reuses row-16 restrictions (hierarchical
//! invariant: restrict CSF time hierarchy first). Core↔sidecar
//! correspondence is an index-set equality (§S4/§S6).

use std::collections::{BTreeMap, BTreeSet};

/// Coordinate key type.
pub type CoordKey = (u64, u32, u32);

/// Exact set union.
#[must_use]
pub fn set_union(a: &BTreeSet<CoordKey>, b: &BTreeSet<CoordKey>) -> BTreeSet<CoordKey> {
    a.union(b).copied().collect()
}

/// Exact set intersection.
#[must_use]
pub fn set_inter(a: &BTreeSet<CoordKey>, b: &BTreeSet<CoordKey>) -> BTreeSet<CoordKey> {
    a.intersection(b).copied().collect()
}

/// Exact set difference `a ∖ b`.
#[must_use]
pub fn set_diff(a: &BTreeSet<CoordKey>, b: &BTreeSet<CoordKey>) -> BTreeSet<CoordKey> {
    a.difference(b).copied().collect()
}

/// Support set of a canonical entry slice (each coordinate once).
#[must_use]
pub fn support_set(entries: &[(u64, u32, u32)]) -> BTreeSet<CoordKey> {
    entries.iter().copied().collect()
}

/// The contiguous transaction-index set `{0, 1, …, n-1}` (exact set).
/// The core holds one record per source row and the sidecar carries the sole
/// `tx_id`, so their correspondence is exactly this index set (§S4/§S6).
#[must_use]
pub fn index_set(n: usize) -> BTreeSet<u64> {
    (0..n as u64).collect()
}

/// tx_id set correspondence: expected index set vs sidecar tx set.
/// Returns `(expected_only, sidecar_only, in_both)`. Exact set ops only.
#[must_use]
pub fn tx_correspondence(
    expected: &BTreeSet<u64>,
    side_tx: &BTreeSet<u64>,
) -> (BTreeSet<u64>, BTreeSet<u64>, BTreeSet<u64>) {
    (
        expected.difference(side_tx).copied().collect(),
        side_tx.difference(expected).copied().collect(),
        expected.intersection(side_tx).copied().collect(),
    )
}

/// Behavioral relation: fan-in shell at receiver `r` restricted to a
/// time window — senders `s` with an entry `(t,s,r)`, `t ∈ [w0,w1)`.
/// Pure row-16 restriction + set construction; no second index.
#[must_use]
pub fn fan_in_senders(
    entries: &[(u64, u32, u32)],
    r: u32,
    w0: u64,
    w1: u64,
) -> BTreeSet<u32> {
    entries.iter()
        .filter(|(t, _, rr)| *rr == r && *t >= w0 && *t < w1)
        .map(|(_, s, _)| *s)
        .collect()
}

/// Behavioral relation: co-occurring receivers sharing a sender in window.
#[must_use]
pub fn co_receivers(
    entries: &[(u64, u32, u32)],
    s: u32,
    w0: u64,
    w1: u64,
) -> BTreeSet<u32> {
    entries.iter()
        .filter(|(t, ss, _)| *ss == s && *t >= w0 && *t < w1)
        .map(|(_, _, r)| *r)
        .collect()
}

/// Amount per coordinate summed exactly (i64 checked; P2/P5 algebra class).
/// Used only for verification of settled values, not as tensor algebra.
pub fn coord_sums(
    entries: &[(u64, u32, u32, i64)],
) -> BTreeMap<CoordKey, i64> {
    let mut m: BTreeMap<CoordKey, i64> = BTreeMap::new();
    for (t, s, r, v) in entries {
        let e = m.entry((*t, *s, *r)).or_insert(0);
        *e = e.checked_add(*v).expect("test fixture sums are representable");
    }
    m
}
