//! Rows 20–23 (§B.2) + P8–P10: event ingestion, sealing, live view, watermark.
//!
//! ```text
//! row 20  Event ingest        d_cas → on Committed only: d_delta ∘ a14 → row 19 push; one atomic step
//! row 21  Seal                row 19 decode (settle) → row 15 decode to tensor → row 7
//!                             (d_merge2, the sealing sum) → d_merge2 support diff → row 11
//!                             compress (full rebuild) → a15 clear (frontier reset)
//! row 22  Support diff; CSF byte-size  d_merge2 comparison-only variant; a09 + a15 length math
//! row 23  Live decode/eval; stats; watermark   P9 invariant; watermark stored, never consulted (P10)
//! ```
//!
//! Watermark semantics (P10): the watermark is a stored sealing boundary. It is
//! never a coordinate, and `ingest` never consults it — a late arrival with
//! `t ≤ W` is an ordinary Create through the same CAS path.
//!
//! Sealing is a **full rebuild** by definition: the derivation defines no
//! in-place CSF mutation, so `Compact(S, C) = CSF(⟦S⟧ + ⟦C)` is realized by
//! merging the decoded canonical tensors and recompressing. No incremental
//! CSF update is invented, and the physical cost is reported honestly.

use crate::composites::{delta, support_diff};
use crate::coo::entry::TxEntry;
use crate::csf::compress::{compress, CsfTensor};
use crate::csf::decode::{decode_csf, decode_to_tensor};
use crate::csf::query::lookup;
use crate::ingest::frontier::Frontier;
use crate::ingest::ledger::{CasOutcome, Event, Ledger, LedgerOp};
use crate::prims::add;
use crate::tensor::Tensor;
use crate::TensorError;
use std::time::{Duration, Instant};

/// Accumulated per-stage wall time from [`IngestionEngine::ingest_timed`].
///
/// Benchmark instrumentation (`a_{20}`: measurement that never influences
/// results). The CAS commit, the delta construction and the frontier emission
/// are ONE atomic step (Axiom 7.4); the stages are separated here purely so
/// the benchmark can report where time goes. Semantics are identical to
/// [`IngestionEngine::ingest`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimings {
    /// Ledger CAS (`d_cas`) accumulated duration.
    pub cas: Duration,
    /// Elementary-delta construction (P8) accumulated duration.
    pub delta: Duration,
    /// Frontier append (row 19) accumulated duration.
    pub frontier: Duration,
}

impl StageTimings {
    /// Combined non-parse, non-materialization ingestion stage time.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.cas + self.delta + self.frontier
    }
}

/// Physical/locality report for one materialization (rows 21/22).
///
/// Only quantities the implementation actually observes are reported.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SealReport {
    /// Raw frontier entries consumed (logical delta size, P4 input form).
    pub frontier_entries: usize,
    /// Nonzero coordinates after settling the segment (P3/P4).
    pub segment_nnz: usize,
    /// Sealed nonzero count before compaction.
    pub sealed_nnz_before: usize,
    /// Sealed nonzero count after compaction.
    pub sealed_nnz_after: usize,
    /// Coordinates whose value changed.
    pub changed_coordinates: usize,
    /// Coordinates newly in the support.
    pub new_coordinates: usize,
    /// Coordinates that left the support (zeroed by the segment).
    pub removed_coordinates: usize,
    /// Analytic sealed-CSF byte size after the rebuild
    /// (`8·|I0| + 8·|P0| + 4·|I1| + 8·|P1| + 4·|I2| + 8·|V|`).
    pub csf_bytes_after: usize,
    /// Whether the entire sealed CSF was rebuilt (always true; by definition).
    pub full_rebuild: bool,
}

/// Live-value counters (diagnostics only; never used for algebra).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineStats {
    /// Successful Create-mode ops.
    pub creates: u64,
    /// Successful Update-mode ops (corrections).
    pub updates: u64,
    /// Successful Close-mode ops (reversals).
    pub closes: u64,
    /// Retried (failed CAS) ops — emitted no delta.
    pub retries: u64,
    /// Deltas emitted into the frontier (equals creates + updates + closes).
    pub deltas_emitted: u64,
    /// Number of materializations performed.
    pub seals: u64,
}

/// The streaming ingestion engine: ledger + frontier + sealed CSF + watermark.
#[derive(Debug, Default)]
pub struct IngestionEngine {
    ledger: Ledger,
    frontier: Frontier,
    sealed: CsfTensor,
    /// P10 watermark: a sealing boundary, never a coordinate.
    watermark: Option<u64>,
    stats: EngineStats,
}

impl IngestionEngine {
    /// Engine over the zero tensor (`T_sealed = 0`, frontier empty).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Row 20 — ingest one event: CAS (`d_cas`) then, on Committed only, one
    /// elementary delta (P8) pushed to the frontier — one atomic step.
    ///
    /// The watermark is not consulted (P10).
    ///
    /// # Errors
    /// [`TensorError::Range`] if the delta `a_new − a_old` is not representable.
    /// The delta is computed before any frontier mutation, so a range error
    /// leaves the frontier unchanged (the CAS bookkeeping is not advanced) and
    /// the caller can abort the stream; `K14` bounds what may be claimed about
    /// the surrounding external guarantees.
    pub fn ingest(&mut self, event: &Event, op: LedgerOp) -> Result<CasOutcome, TensorError> {
        self.ingest_inner(event, op, None)
    }

    /// [`IngestionEngine::ingest`] with per-stage wall-time accumulation
    /// (benchmark instrumentation; identical semantics).
    ///
    /// # Errors
    /// As [`IngestionEngine::ingest`].
    pub fn ingest_timed(
        &mut self,
        event: &Event,
        op: LedgerOp,
        stages: &mut StageTimings,
    ) -> Result<CasOutcome, TensorError> {
        self.ingest_inner(event, op, Some(stages))
    }

    fn ingest_inner(
        &mut self,
        event: &Event,
        op: LedgerOp,
        stages: Option<&mut StageTimings>,
    ) -> Result<CasOutcome, TensorError> {
        let mut stages = stages;
        let t0 = Instant::now();
        let outcome = self.ledger.ledger_op(event.id, op);
        let t_cas = t0.elapsed();
        let committed = match outcome {
            CasOutcome::Committed { a_old, a_new, .. } => Some((a_old, a_new)),
            CasOutcome::Retry { .. } => {
                self.stats.retries += 1;
                None
            }
        };
        if let Some(st) = stages.as_mut() {
            st.cas += t_cas;
        }
        let Some((a_old, a_new)) = committed else {
            return Ok(outcome);
        };
        let t1 = Instant::now();
        let d_res = delta(a_new, a_old);
        if let Some(st) = stages.as_mut() {
            st.delta += t1.elapsed();
        }
        let d = d_res?;
        let t2 = Instant::now();
        // One Δ per successful op, emitted in the same step (P8 + Axiom 7.4).
        self.frontier.push(TxEntry {
            t: event.coord.t,
            s: event.coord.s,
            r: event.coord.r,
            value: d,
        });
        if let Some(st) = stages.as_mut() {
            st.frontier += t2.elapsed();
        }
        self.stats.deltas_emitted += 1;
        match op {
            LedgerOp::Create(_) => self.stats.creates += 1,
            LedgerOp::Update { .. } => self.stats.updates += 1,
            LedgerOp::Close { .. } => self.stats.closes += 1,
        }
        Ok(outcome)
    }

    /// Row 23 + P10 — advance the watermark (a sealing boundary, never a
    /// coordinate). Events remain ordinary deltas regardless of `t` vs `W`.
    pub fn set_watermark(&mut self, w: u64) {
        self.watermark = Some(w);
    }

    /// Current watermark, if one has been set (P10).
    #[must_use]
    pub fn watermark(&self) -> Option<u64> {
        self.watermark
    }

    /// Row 23 — `T_live(x) = T_sealed(x) + ⟦C_frontier⟧(x)` at one coordinate.
    ///
    /// # Errors
    /// [`TensorError::Range`] if the pointwise sum is not representable.
    pub fn live_at(&self, t: u64, s: u32, r: u32) -> Result<i64, TensorError> {
        let sealed_v = lookup(&self.sealed, t, s, r);
        let live_v = self.frontier.eval_at(t, s, r)?;
        add(sealed_v, live_v, "T_live pointwise sum (row 23)")
    }

    /// Row 23 — `T_live = ⟦CSF(T_sealed)⟧ + ⟦C_frontier⟧` as a canonical
    /// `Tensor`: the live-view object (P9 invariant).
    ///
    /// # Errors
    /// [`TensorError::Range`] on unrepresentable sums (checked throughout).
    pub fn decode_live(&self) -> Result<Tensor, TensorError> {
        let sealed_t = decode_to_tensor(&self.sealed);
        let frontier_t = self.frontier.decode()?;
        sealed_t.try_add(&frontier_t)
    }

    /// The sealed canonical object `CSF(T_sealed)`.
    #[must_use]
    pub fn sealed(&self) -> &CsfTensor {
        &self.sealed
    }

    /// Ledger access (diagnostics/tests; Def 7.2).
    #[must_use]
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Frontier access (row 19).
    #[must_use]
    pub fn frontier(&self) -> &Frontier {
        &self.frontier
    }

    /// Live counters (diagnostics only).
    #[must_use]
    pub fn stats(&self) -> EngineStats {
        self.stats
    }

    /// Rows 21/22 — materialization + compaction: seal the whole frontier into
    /// `T_sealed ← Compact(CSF(T_sealed), C_frontier)` and reset the live delta
    /// to 0.
    ///
    /// The sealing sum is the row-7 pointwise addition (one `d_merge2`); the
    /// support difference is the row-22 comparison-only variant of the same
    /// composite; the rebuild is row-11 compression of the merged canonical
    /// COO (no in-place CSF mutation exists to use instead).
    ///
    /// # Errors
    /// [`TensorError::Range`] on any unrepresentable sum (checked paths:
    /// settling of the segment and the pointwise sealing sum).
    pub fn seal(&mut self) -> Result<SealReport, TensorError> {
        let frontier_entries = self.frontier.len();
        let segment = self.frontier.decode()?; // settling (P4)
        let segment_nnz = segment.nnz();
        let sealed_nnz_before = self.sealed.nnz();
        let sealed_decoded = decode_to_tensor(&self.sealed); // row 15
        let merged = sealed_decoded.try_add(&segment)?; // row 7 (sealing sum)
        let merged_nnz = merged.nnz();
        let (changed, added, removed) = support_diff(sealed_decoded.entries(), merged.entries()); // row 22
        let new_csf = compress(merged.entries()); // row 11 (canonical input)
        let csf_bytes = csf_byte_size(&new_csf); // row 22
        self.sealed = new_csf;
        self.frontier.clear(); // live delta reset to 0 (P9)
        self.stats.seals += 1;
        Ok(SealReport {
            frontier_entries,
            segment_nnz,
            sealed_nnz_before,
            sealed_nnz_after: merged_nnz,
            changed_coordinates: changed,
            new_coordinates: added,
            removed_coordinates: removed,
            csf_bytes_after: csf_bytes,
            full_rebuild: true,
        })
    }
}

/// Row 22 — count coordinates whose value/presence differs between two
/// canonical entry sequences (`d_merge2` comparison-only variant, one
/// realization in [`crate::composites::support_diff`]).
#[must_use]
pub fn diff_coordinates(old: &[TxEntry], new: &[TxEntry]) -> (usize, usize, usize) {
    support_diff(old, new)
}

/// Row 22 — analytic serialized CSF byte size from the payload layout
/// (`8·|I0| + 8·|P0| + 4·|I1| + 8·|P1| + 4·|I2| + 8·|V|`, `a_{09}` length math).
#[must_use]
pub fn csf_byte_size(csf: &CsfTensor) -> usize {
    8 * csf.mode0_coords.len()
        + 8 * csf.mode0_fptr.len()
        + 4 * csf.mode1_coords.len()
        + 8 * csf.mode1_fptr.len()
        + 4 * csf.mode2_coords.len()
        + 8 * csf.values.len()
}

/// Row 15 — decode a CSF to canonical COO (re-export for oracles).
#[must_use]
pub fn sealed_decode(csf: &CsfTensor) -> Vec<TxEntry> {
    decode_csf(csf)
}
