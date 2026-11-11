//! Row 19 (§B.2) + P9: the frontier `C_frontier`.
//!
//! The frontier is a **mutable raw delta sequence** (P4 input form): duplicate
//! coordinates are permitted, entries arrive in arrival order, and nothing is
//! aggregated on append. `Δ_frontier := ⟦C_frontier⟧` (settling) and
//! `T_live = T_sealed + ⟦C_frontier⟧`. Sorting and aggregation happen only at
//! materialization (row 21) — never on the append path, which is the single
//! physical write per event that P9 describes.
//!
//! No ordered index is introduced: the frontier is append-only (`a_{15}`), and
//! the diagnostic point read is an `R_3` scan.

use crate::coo::entry::TxEntry;
use crate::tensor::Tensor;
use crate::TensorError;

/// `C_frontier` (Def 9.1): the live raw delta sequence.
#[derive(Debug, Default, Clone)]
pub struct Frontier {
    entries: Vec<TxEntry>,
}

impl Frontier {
    /// Empty frontier (`Δ_frontier = 0`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one raw delta entry (P4 input; duplicates permitted). This is
    /// the only per-event physical write the system performs (P9).
    pub fn push(&mut self, e: TxEntry) {
        self.entries.push(e);
    }

    /// Raw sequence view (arrival order, duplicates permitted).
    #[must_use]
    pub fn entries(&self) -> &[TxEntry] {
        &self.entries
    }

    /// Number of raw entries currently buffered (not `|supp|`: no aggregation
    /// has happened yet; `P11` keeps this distinct from the logical delta).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the frontier is empty (`Δ_frontier = 0`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `Δ_frontier = ⟦C_frontier⟧` (settling, P4; checked sums).
    ///
    /// # Errors
    /// [`TensorError::Range`] if any coordinate sum is not representable.
    pub fn decode(&self) -> Result<Tensor, TensorError> {
        crate::coo::decode::decode_to_tensor(&self.entries)
    }

    /// `⟦C_frontier⟧(x)` at one coordinate (row 10 evaluation; checked).
    ///
    /// # Errors
    /// [`TensorError::Range`] if the coordinate sum is not representable.
    pub fn eval_at(&self, t: u64, s: u32, r: u32) -> Result<i64, TensorError> {
        crate::coo::decode::try_eval_at(&self.entries, t, s, r)
    }

    /// Drop all entries (materialization resets the live delta to 0).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
