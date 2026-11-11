//! Committed-state snapshot: sealed CSF + frontier overlay + derived
//! account lookups. Read consistency: readers borrow one immutable
//! snapshot; the writer publishes the next snapshot only after all
//! components are consistent (serialized writer, atomic publication).

use crate::retrieval::TemporalIndexes;
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::CsfTensor;
use std::collections::{BTreeMap, BTreeSet};

/// Monotone committed-state version (0 = initial load, +1 per commit).
pub type StateVersion = u64;

/// Derived account-oriented lookup over sealed + frontier (exact set ops).
///
/// This is genuinely derived state (hierarchical invariant section 8):
/// per-account coordinate sets computed from the two canonical sources via
/// exact `BTreeSet` union — never a second canonical store, never written
/// back into the CSF.
#[derive(Debug, Clone, Default)]
pub struct AccountIndex {
    /// sender code -> set of (t, s, r) coordinates.
    pub by_sender: BTreeMap<u32, BTreeSet<(u64, u32, u32)>>,
    /// receiver code -> set of (t, s, r) coordinates.
    pub by_receiver: BTreeMap<u32, BTreeSet<(u64, u32, u32)>>,
}

impl AccountIndex {
    /// Rebuild from sealed entries + frontier entries (exact set union).
    #[must_use]
    pub fn build(sealed: &[TxEntry], frontier: &[TxEntry]) -> Self {
        let mut idx = Self::default();
        for e in sealed.iter().chain(frontier.iter()) {
            let k = (e.t, e.s, e.r);
            idx.by_sender.entry(e.s).or_default().insert(k);
            idx.by_receiver.entry(e.r).or_default().insert(k);
        }
        idx
    }

    /// Insert one committed coordinate (incremental, O(log n)).
    pub fn insert(&mut self, e: &TxEntry) {
        let k = (e.t, e.s, e.r);
        self.by_sender.entry(e.s).or_default().insert(k);
        self.by_receiver.entry(e.r).or_default().insert(k);
    }
}

/// One committed, queryable snapshot.
#[derive(Debug, Clone)]
pub struct PipelineSnapshot {
    /// Committed version.
    pub version: StateVersion,
    /// Immutable sealed CSF (never mutated in place).
    pub sealed: CsfTensor,
    /// Bounded frontier overlay (raw delta sequence, arrival order).
    pub frontier: Vec<TxEntry>,
    /// Derived account lookups over sealed + frontier.
    pub index: AccountIndex,
    /// Committed record count (initial records + accepted events).
    pub committed_records: usize,
    /// Temporal bounds over committed state.
    pub t_min: u64,
    /// Temporal bounds over committed state.
    pub t_max: u64,
    /// Sender/receiver dictionaries (append-only extension).
    pub senders: Vec<String>,
    /// Sender/receiver dictionaries (append-only extension).
    pub receivers: Vec<String>,
    /// Derived temporal retrieval state (forward/backward indexes, global
    /// time path, locator directory) — published in the same immutable
    /// snapshot as the canonical tensor, so readers can never observe the
    /// tensor and the indexes disagreeing on the committed event set
    /// (phase-3 task spec §5).
    pub temporal: TemporalIndexes,
}
