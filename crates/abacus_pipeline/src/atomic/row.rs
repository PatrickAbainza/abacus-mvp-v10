//! CSR/CSC row kernels: fiber access + fixed-pair ordered seek.
//!
//! Generic terms only. Presence here is event-exists (committed rows,
//! including compensating rows). Pair seek is scan/filter composition
//! (Phase-13 decision a): filter peer, order by (t, locator); earliest =
//! min, latest = max, next-after-tau = min over STRICT t > tau. No O(log n)
//! pair-seek claim. Recent-window queries are ordered seeks on the same
//! rows (no separate structure).

use abacus_tensor::atomic::kernels::traits::{KernelStats, OperatorKernel};
use abacus_tensor::atomic::reference::{
    self, Coverage, MapFn, MergeFn, Predicate,
    ReducedRow, RowRef, Theta,
};
use abacus_tensor::atomic::types::{Entry, LocSet, Mode, Monoid, Support};
use abacus_tensor::atomic::Outcome;

use crate::retrieval::TemporalRef;

/// Convert one index row slice to kernel row refs (order preserved).
fn to_refs(row: &[TemporalRef]) -> Vec<RowRef> {
    row.iter()
        .map(|e| RowRef { t: e.t, peer: e.peer, locator: e.locator.0 })
        .collect()
}

/// Fiber entries for one key over a half-open window: (t, peer, locator)
/// in row order. Event-exists presence: every committed row qualifies,
/// including compensating rows (reversal lifecycle traced by the caller).
#[must_use]
pub fn fiber_window(row: &[TemporalRef], lo_t: u64, hi_t: u64, stats: &mut KernelStats) -> Vec<RowRef> {
    stats.seeks = stats.seeks.saturating_add(2);
    if lo_t >= hi_t {
        return Vec::new();
    }
    let lo = row.partition_point(|e| e.t < lo_t);
    let hi = row.partition_point(|e| e.t < hi_t);
    stats.scanned = stats.scanned.saturating_add((hi.saturating_sub(lo)) as u64);
    to_refs(&row[lo..hi])
}

/// Fixed-pair seek over a single-key-major row (scan/filter composition).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairSeek {
    /// Minimum over (t, locator) among peer matches.
    Earliest,
    /// Maximum over (t, locator) among peer matches.
    Latest,
    /// Minimum over (t, locator) among peer matches with STRICT t > tau.
    NextAfter(u64),
}

/// Seek one fixed pair in a row. Returns the witness ref, if any.
///
/// Single-pass scan/filter over the borrowed row: no allocation, no
/// intermediate copy. Semantics identical to the reference composition
/// (`pair_earliest` / `pair_latest` / `pair_next_after` over converted
/// refs): filter peer, order by `(t, locator)`; STRICT `t > tau` for
/// `NextAfter`. The reference remains the semantic oracle; differential
/// tests prove equality.
#[must_use]
pub fn pair_seek(row: &[TemporalRef], want_peer: u32, seek: PairSeek, stats: &mut KernelStats) -> Option<RowRef> {
    stats.scanned = stats.scanned.saturating_add(row.len() as u64);
    let mut best: Option<(u64, u32)> = None; // (t, locator)
    for e in row {
        if e.peer != want_peer {
            continue;
        }
        if let PairSeek::NextAfter(tau) = seek {
            if e.t <= tau {
                continue; // strict
            }
        }
        let cand = (e.t, e.locator.0);
        best = match (best, seek) {
            (None, _) => Some(cand),
            (Some(b), PairSeek::Latest) => Some(if cand > b { cand } else { b }),
            (Some(b), _) => Some(if cand < b { cand } else { b }),
        };
    }
    best.map(|(t, locator)| RowRef { t, peer: want_peer, locator })
}

/// Row-backed kernel: fiber access + pair seek over borrowed rows.
/// `key_side` selects which coordinate the row key fixes (mode-1 for
/// forward rows, mode-2 for backward rows); the peer fills the other.
pub struct RowKernel<'a> {
    /// Borrowed ordered rows (each `(t, locator)` sorted).
    pub rows: Vec<&'a [TemporalRef]>,
    /// True = rows keyed by mode-1 (forward); false = mode-2 (backward).
    pub forward: bool,
}

impl<'a> RowKernel<'a> {
    /// Borrow forward (mode-1-major) rows.
    #[must_use]
    pub const fn forward(rows: Vec<&'a [TemporalRef]>) -> Self {
        Self { rows, forward: true }
    }
    /// Borrow backward (mode-2-major) rows.
    #[must_use]
    pub const fn backward(rows: Vec<&'a [TemporalRef]>) -> Self {
        Self { rows, forward: false }
    }

    /// Materialize windowed fiber entries for the selected rows.
    /// Values are bottom-marked here (0 placeholder dropped by support
    /// canonicalization); callers join live values by locator.
    fn fiber_support(&self, keys: &[u32], lo_t: u64, hi_t: u64, stats: &mut KernelStats) -> Support {
        let mut entries = Vec::new();
        for (idx, row) in self.rows.iter().enumerate() {
            let key = idx as u32;
            if !keys.contains(&key) {
                continue;
            }
            for r in fiber_window(row, lo_t, hi_t, stats) {
                let (s, rr) = if self.forward { (key, r.peer) } else { (r.peer, key) };
                entries.push(Entry::new(r.t, s, rr, 1, LocSet::from_sorted(vec![r.locator])));
            }
        }
        entries.sort();
        entries.dedup_by(|a, b| a.coord == b.coord && a.locators == b.locators);
        // NOTE: value 1 marks event-exists presence for fiber membership;
        // algebraic joins must merge live values by locator afterwards.
        Support { entries }
    }
}

impl OperatorKernel for RowKernel<'_> {
    fn name(&self) -> &'static str {
        if self.forward { "csr-row" } else { "csc-row" }
    }

    fn restrict(&self, _source: &Support, p: &Predicate) -> (Support, KernelStats) {
        let mut stats = KernelStats::default();
        let (lo_t, hi_t) = p.window.unwrap_or((0, u64::MAX));
        if lo_t >= hi_t {
            return (Support::empty(), stats);
        }
        // Row set: keys1 for forward rows, keys2 for backward rows.
        let all_keys: Vec<u32> = (0..self.rows.len() as u32).collect();
        let keys: &[u32] = if self.forward { p.keys1.as_deref().unwrap_or(&all_keys) } else { p.keys2.as_deref().unwrap_or(&all_keys) };
        let mut sup = self.fiber_support(keys, lo_t, hi_t, &mut stats);
        sup.entries.retain(|e| p.admits(e));
        (sup, stats)
    }

    fn map(&self, source: &Support, f: MapFn, theta: &Theta) -> (Support, KernelStats) {
        let stats = KernelStats { scanned: source.entries.len() as u64, seeks: 0 };
        (reference::map(source, f, theta), stats)
    }

    fn merge(
        &self,
        left: &Support,
        right: &Support,
        combine: MergeFn,
        coverage: Coverage,
    ) -> (Outcome<Support>, KernelStats) {
        let stats = KernelStats {
            scanned: (left.entries.len() + right.entries.len()) as u64,
            seeks: 0,
        };
        (reference::merge(left, right, combine, coverage), stats)
    }

    fn reduce(
        &self,
        source: &Support,
        modes: &[Mode],
        monoid: Monoid,
    ) -> (Outcome<Vec<ReducedRow>>, KernelStats) {
        let stats = KernelStats { scanned: source.entries.len() as u64, seeks: 0 };
        (reference::reduce(source, modes, monoid), stats)
    }
}
