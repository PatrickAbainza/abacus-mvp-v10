//! Ingestion-backed pipeline: the `abacus_pipeline` carrier for CSV/Kafka data.
//!
//! Per the reconciliation directive, this is the pipeline's new native form.
//! It replaces the former role-separated string-dictionary / content-key /
//! `SAMCORE02` model with the **CSV/Kafka ingestion** model:
//!
//! - **Substrate + shared account map.** A [`SubstrateApplier`] owns the
//!   existing `IngestionEngine` (ledger CAS + frontier) and the single
//!   append-only [`AccountMap`] (§6: one shared sender/receiver namespace).
//! - **Compound identity + identity-ledger dedup.** [`IdentityLedger::check`]
//!   classifies each delivery as New / Duplicate / Conflict on
//!   `(namespace, event_id)` (Canonical Contract §3) — the G2 ruling "identity
//!   -ledger is authoritative".
//! - **Derived metrics, not raw amounts (privacy).** Each committed event
//!   carries the frozen seven-byte [`FeaturePayload`] (GS §9). The exact amount
//!   is written into the substrate value but never stored in this view; the UI
//!   and graph read only the derived features.
//! - **Temporal indexes over the shared namespace.** The existing
//!   [`TemporalIndexes`] are built from committed `(t, s, r)` where `s`/`r` are
//!   shared-namespace codes, so a transaction's network view uses one account
//!   identity regardless of role.

use crate::error::PipelineError;
use crate::ingest_model::{AccountMap, EventIdentity, FeaturePayload, IngestedEvent};
use abacus_tensor::graphblas::types::FeatureVector;
use crate::retrieval::{
    EventLocator, LocatorBuffer, MultiHopQuery, ResolvedEvent, RetrievalResult, TemporalIndexes,
};
use std::collections::BTreeMap;

use abacus_ingest::apply::{ApplyOutcome, SubstrateApplier, SubstrateValuePolicy};
use abacus_ingest::identity::{IdentityLedger, IdentityOutcome};
use abacus_ingest::record::CanonicalRecord;

use abacus_tensor::api::CanonicalTensor;
use abacus_tensor::csf::query::{
    receiver_at_time as sealed_receiver_at_time, sender_at_time as sealed_sender_at_time,
};

/// Outcome of applying one ingestion record through the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestApplyOutcome {
    /// Newly committed; carries the substrate locator (event id).
    Committed(u32),
    /// Redelivery of an already-applied compound identity (no state change).
    Duplicate,
}

/// Read-only statistics of the pipeline's live CSF tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsfStats {
    /// Number of nonzero entries in the live tensor (`T_live` support size).
    pub nnz: u64,
}

/// The ingestion-backed pipeline state (single owner; see `state.rs`).
pub struct IngestPipeline {
    /// Substrate applier: `IngestionEngine` + shared account namespace.
    applier: SubstrateApplier,
    /// Compound-identity dedup (authoritative for redelivery/conflict).
    identity: IdentityLedger,
    /// Temporal retrieval indexes over the shared namespace.
    temporal: TemporalIndexes,
    /// Per-committed-event derived feature payloads, keyed by locator.
    features: Vec<FeaturePayload>,
    /// Committed compound identities, parallel to `features` (locator order).
    identities: Vec<EventIdentity>,
}

impl IngestPipeline {
    /// Build over a fresh substrate with the attested-amount value policy
    /// (Amendment A1: every source transports its exact amount at commit).
    #[must_use]
    pub fn new() -> Self {
        Self::with_policy(SubstrateValuePolicy::AttestedMinorUnits)
    }

    /// Build with an explicit substrate value policy.
    #[must_use]
    pub fn with_policy(policy: SubstrateValuePolicy) -> Self {
        Self {
            applier: SubstrateApplier::new(policy),
            identity: IdentityLedger::new(),
            temporal: TemporalIndexes::try_build([], 0, 0).expect("empty build cannot exhaust"),
            features: Vec::new(),
            identities: Vec::new(),
        }
    }

    /// The shared account map (§6: one namespace for sender and receiver).
    #[must_use]
    pub fn accounts(&self) -> &AccountMap {
        self.applier.accounts()
    }

    /// Temporal retrieval indexes over the shared namespace.
    #[must_use]
    pub fn temporal(&self) -> &TemporalIndexes {
        &self.temporal
    }

    /// Number of committed events.
    #[must_use]
    pub fn committed(&self) -> usize {
        self.features.len()
    }

    /// The committed event at a locator (features + identity), if present.
    #[must_use]
    pub fn event(&self, locator: u32) -> Option<IngestedEvent> {
        let features = *self.features.get(locator as usize)?;
        let identity = *self.identities.get(locator as usize)?;
        let dir = self
            .temporal
            .resolve(crate::retrieval::EventLocator(locator))?;
        Some(IngestedEvent::new(
            identity,
            dir.t,
            dir.sender,
            dir.receiver,
            features,
        ))
    }

    /// Account label for a shared-namespace code (the wire identifier rendered
    /// as a decimal string). Role-independent: the SAME code yields the SAME
    /// label whether the account sent or received, unlike the retired
    /// role-separated dictionaries.
    #[must_use]
    pub fn account_label(&self, code: u32) -> Option<String> {
        self.applier.accounts().wire_of(code).map(|w| w.to_string())
    }

    /// Outgoing (sender-side) locators for a shared-namespace account code in
    /// `[t0, t1)`, ascending `(t, locator)`, capped at `max_edges`. Queried by
    /// shared code directly — no role-separated sender dictionary involved.
    #[must_use]
    pub fn outgoing_in_range(&self, code: u32, t0: u64, t1: u64, max_edges: usize) -> Vec<u32> {
        if t0 >= t1 {
            return Vec::new();
        }
        let row = self.temporal.forward.row(code);
        let lo = row.partition_point(|e| e.t < t0);
        let hi = row.partition_point(|e| e.t < t1);
        row[lo..hi]
            .iter()
            .take(max_edges)
            .map(|e| e.locator.0)
            .collect()
    }

    /// Incoming (receiver-side) locators for a shared-namespace account code in
    /// `[t0, t1)`, ascending `(t, locator)`, capped at `max_edges`.
    #[must_use]
    pub fn incoming_in_range(&self, code: u32, t0: u64, t1: u64, max_edges: usize) -> Vec<u32> {
        if t0 >= t1 {
            return Vec::new();
        }
        let row = self.temporal.backward.row(code);
        let lo = row.partition_point(|e| e.t < t0);
        let hi = row.partition_point(|e| e.t < t1);
        row[lo..hi]
            .iter()
            .take(max_edges)
            .map(|e| e.locator.0)
            .collect()
    }

    /// Resolve a locator to its shared-namespace `(t, sender_code,
    /// receiver_code)` plus derived features. `None` if unknown (never
    /// fabricated, P3).
    #[must_use]
    pub fn resolve(&self, locator: u32) -> Option<(u64, u32, u32, FeaturePayload)> {
        let ev = self.event(locator)?;
        Some((ev.t, ev.s, ev.r, ev.features))
    }

    /// Batched CSR row lookup: outgoing events for multiple shared-namespace
    /// account codes in `[t0, t1)`, each capped at `max_edges`.
    ///
    /// Delegates to [`TemporalIndexes::batch_outgoing_in_range`].
    #[must_use]
    pub fn batch_outgoing_in_range(
        &self,
        codes: &[u32],
        t0: u64,
        t1: u64,
        max_edges: usize,
    ) -> Vec<(u32, Vec<u32>)> {
        self.temporal
            .batch_outgoing_in_range(codes, t0, t1, max_edges)
    }

    /// Batched CSC column lookup: incoming events for multiple shared-namespace
    /// account codes in `[t0, t1)`, each capped at `max_edges`.
    ///
    /// Delegates to [`TemporalIndexes::batch_incoming_in_range`].
    #[must_use]
    pub fn batch_incoming_in_range(
        &self,
        codes: &[u32],
        t0: u64,
        t1: u64,
        max_edges: usize,
    ) -> Vec<(u32, Vec<u32>)> {
        self.temporal
            .batch_incoming_in_range(codes, t0, t1, max_edges)
    }

    /// Bounded K=2 focal context over the live pipeline (endpoint-driven).
    ///
    /// Golden Source: Addendum O rows 46/49 (k parameterized; K=2 per
    /// Amendment A1), HCSF 5-7 (bounded endpoint-driven traversal over the
    /// EXISTING CSR/CSC indexes), HCSF 9 (equivalence or Overflow).
    /// Value for each admitted edge comes from `csf_live_at` (T_live =
    /// T_sealed + frontier, row 23) — never fabricated (P3).
    ///
    /// # Errors
    /// `BoundedError::VertexOverflow` / `AdmittedOverflow` when the context
    /// exceeds the L1-derived bound; `Reference` for focal/volume failures.
    #[must_use]
    pub fn bounded_context_k2(
        &self,
        focal_locator: u32,
        omega_secs: u64,
    ) -> Result<abacus_tensor::graphblas::context::FocalContext, crate::bounded_context::BoundedError> {
        use crate::bounded_context::{build_bounded_k2, BoundedError};
        use crate::retrieval::EventLocator;
        let dir = self.temporal.resolve(EventLocator(focal_locator)).ok_or(
            BoundedError::Reference(abacus_tensor::graphblas::context::ExecutionError::FocalAbsent),
        )?;
        let focal = abacus_tensor::graphblas::types::FocalTransaction {
            epoch: dir.t,
            sender: abacus_tensor::graphblas::types::VertexId(dir.sender),
            receiver: abacus_tensor::graphblas::types::VertexId(dir.receiver),
        };
        let params = abacus_tensor::graphblas::context::ContextParameters {
            window: abacus_tensor::graphblas::types::TimeWindow::backward_inclusive(dir.t, omega_secs),
            hop_bound: abacus_tensor::graphblas::types::HopBound(2),
            eligible: None,
            self_loop: abacus_tensor::graphblas::types::SelfLoopRule::Exclude,
        };
        let this = self;
        let ctx = build_bounded_k2(self.temporal(), focal, &params, &|loc| {
            let (t, s, r, fp) = this.resolve(loc.0)?;
            let v = this.csf_live_at(t, s, r)?;
            // GS §9 / Amendment A1: carry the pre-computed derived features
            // (log_magnitude, relative_threshold_distance, roundedness_flags)
            // verbatim from `features[locator]`, never recomputed. `FeaturePayload`
            // and `FeatureVector` share field widths (u16/i32/u8): direct field map.
            let fv = FeatureVector {
                log_magnitude: fp.log_magnitude,
                relative_threshold_distance: fp.relative_threshold_distance,
                roundedness_flags: fp.roundedness_flags,
            };
            Some((t, s, r, v, fv))
        })?;
        Ok(ctx)
    }

    /// Resolve an account identifier string to its shared-namespace code.
    /// The identifier is the wire ID rendered as a decimal string (the same
    /// representation returned by [`IngestPipeline::account_label`]).
    /// `None` when the account is unknown (repository lookup convention).
    #[must_use]
    pub fn account_code(&self, entity_id: &str) -> Option<u32> {
        let wire_id: u64 = entity_id.parse().ok()?;
        self.accounts().code_of(wire_id)
    }

    // -----------------------------------------------------------------------
    // Phase-3 temporal retrieval over the shared-namespace indexes
    // (CSR forward = sender-oriented, CSC backward = receiver-oriented).
    // These restore the query block removed in the cutover commit
    // `88fac5c`, adapted to `IngestPipeline` / `TemporalIndexes`.
    // -----------------------------------------------------------------------

    /// Backward / upstream lookback: incoming events to `entity_id` in the
    /// half-open interval `[anchor_time - delta, anchor_time)`.
    ///
    /// - Only events strictly earlier than the anchor are eligible;
    ///   the lower bound is inclusive.
    /// - Results are ordered by `(t, locator)` descending — the most recent
    ///   qualifying events first (deterministic tie-break: newer locator first).
    /// - `is_truncated` is true iff qualifying events exist beyond the returned
    ///   locators; `total_matching_events` is the exact count (both bounds come
    ///   from `partition_point` on the ordered row — no scan, no estimate).
    /// - Unknown account ⇒ empty result with `is_truncated = false`.
    /// - The effective limit is `min(query.max_edges, MAX_EDGES_PER_LOOKUP)`;
    ///   `delta` defaults to seven days when the caller uses
    ///   [`MultiHopQuery::with_default_delta`].
    #[must_use]
    pub fn incoming_relative(&self, query: &MultiHopQuery) -> RetrievalResult {
        let limit = query.effective_limit();
        let lo_t = query.anchor_time.saturating_sub(query.delta);
        let hi_t = query.anchor_time;
        let Some(code) = self.account_code(&query.entity_id) else {
            return RetrievalResult {
                locators: LocatorBuffer::with_limit(limit),
                is_truncated: false,
                total_matching_events: Some(0),
            };
        };
        let row = self.temporal.backward.row(code);
        let lo = row.partition_point(|e| e.t < lo_t);
        let hi = row.partition_point(|e| e.t < hi_t);
        let count = hi - lo;
        let take = count.min(limit);
        let mut locators = LocatorBuffer::with_limit(limit);
        for e in row[hi - take..hi].iter().rev() {
            locators.push(e.locator);
        }
        RetrievalResult {
            locators,
            is_truncated: count > limit,
            total_matching_events: Some(count),
        }
    }

    /// Forward / downstream lookahead: outgoing events from `entity_id` in the
    /// half-open interval `[anchor_time, anchor_time + delta)`.
    ///
    /// - Events at the anchor timestamp are eligible; events at the interval
    ///   end are excluded (half-open).
    /// - Results are ordered by `(t, locator)` ascending — the earliest
    ///   qualifying events first (deterministic tie-break).
    /// - `is_truncated` is true iff qualifying events exist beyond the returned
    ///   locators; `total_matching_events` is the exact count.
    /// - Unknown account ⇒ empty result with `is_truncated = false`.
    /// - If `anchor_time + delta` overflows `u64`, the upper end is unbounded
    ///   (every representable `t >= anchor_time` qualifies).
    #[must_use]
    pub fn outgoing_relative(&self, query: &MultiHopQuery) -> RetrievalResult {
        let limit = query.effective_limit();
        let lo_t = query.anchor_time;
        let hi_opt = query.anchor_time.checked_add(query.delta);
        let Some(code) = self.account_code(&query.entity_id) else {
            return RetrievalResult {
                locators: LocatorBuffer::with_limit(limit),
                is_truncated: false,
                total_matching_events: Some(0),
            };
        };
        let row = self.temporal.forward.row(code);
        let lo = row.partition_point(|e| e.t < lo_t);
        let hi = hi_opt.map_or(row.len(), |h| row.partition_point(|e| e.t < h));
        let count = hi - lo;
        let take = count.min(limit);
        let mut locators = LocatorBuffer::with_limit(limit);
        for e in &row[lo..lo + take] {
            locators.push(e.locator);
        }
        RetrievalResult {
            locators,
            is_truncated: count > limit,
            total_matching_events: Some(count),
        }
    }

    /// Combined incoming and outgoing events for `entity_id` in the interval
    /// `[anchor_time - delta, anchor_time + delta)`.
    ///
    /// Each original event is returned once even when it appears in both roles
    /// (self-transfer). Results are ordered by `(t, locator)` ascending.
    /// `total_matching_events` is `None` when `limit == 0` (early-exit on first
    /// match); otherwise it is the exact count.
    #[must_use]
    pub fn combined_relative(&self, query: &MultiHopQuery) -> RetrievalResult {
        let limit = query.effective_limit();
        let t0 = query.anchor_time.saturating_sub(query.delta);
        let t1 = query.anchor_time.saturating_add(query.delta);
        if t0 >= t1 {
            return RetrievalResult {
                locators: LocatorBuffer::with_limit(limit),
                is_truncated: false,
                total_matching_events: Some(0),
            };
        }
        let empty: &[crate::retrieval::TemporalRef] = &[];
        let fwd_row = match self.account_code(&query.entity_id) {
            Some(c) => self.temporal.forward.row(c),
            None => empty,
        };
        let bwd_row = match self.account_code(&query.entity_id) {
            Some(c) => self.temporal.backward.row(c),
            None => empty,
        };
        let f_lo = fwd_row.partition_point(|e| e.t < t0);
        let f_hi = fwd_row.partition_point(|e| e.t < t1);
        let b_lo = bwd_row.partition_point(|e| e.t < t0);
        let b_hi = bwd_row.partition_point(|e| e.t < t1);
        let f = &fwd_row[f_lo..f_hi];
        let b = &bwd_row[b_lo..b_hi];
        let mut locators = LocatorBuffer::with_limit(limit);
        if limit == 0 {
            let exists = !f.is_empty() || !b.is_empty();
            return RetrievalResult {
                locators,
                is_truncated: exists,
                total_matching_events: None,
            };
        }
        let (mut i, mut j, mut count) = (0usize, 0usize, 0usize);
        while i < f.len() || j < b.len() {
            let same_event =
                i < f.len() && j < b.len() && (f[i].t, f[i].locator) == (b[j].t, b[j].locator);
            if same_event {
                count += 1;
                if locators.len() < limit {
                    locators.push(f[i].locator);
                }
                i += 1;
                j += 1;
                continue;
            }
            let take_fwd = match (i < f.len(), j < b.len()) {
                (true, false) => true,
                (false, true) => false,
                (true, true) => (f[i].t, f[i].locator) < (b[j].t, b[j].locator),
                (false, false) => unreachable!("loop condition"),
            };
            count += 1;
            if take_fwd {
                if locators.len() < limit {
                    locators.push(f[i].locator);
                }
                i += 1;
            } else {
                if locators.len() < limit {
                    locators.push(b[j].locator);
                }
                j += 1;
            }
        }
        RetrievalResult {
            locators,
            is_truncated: count > limit,
            total_matching_events: Some(count),
        }
    }

    /// Global temporal retrieval: every event locator with `t0 <= t < t1`,
    /// ordered `(t, locator)` ascending, **uncapped** (the 2,500 cap applies
    /// only to individual entity-centered directional lookups). Served directly
    /// from the global time path — no per-entity enumeration. An inverted or
    /// empty window (`t0 >= t1`) returns no locators.
    #[must_use]
    pub fn events_in_range(&self, t0: u64, t1: u64) -> Vec<EventLocator> {
        self.temporal
            .global_range(t0, t1)
            .iter()
            .copied()
            .map(EventLocator)
            .collect()
    }

    /// Resolve a locator to the authoritative event's timestamp, sender code
    /// and receiver code (`ResolvedEvent`). `None` for unknown locators.
    #[must_use]
    pub fn resolve_event(&self, locator: EventLocator) -> Option<ResolvedEvent> {
        self.temporal.resolve(locator)
    }

    /// Structural cross-check of tensor/index agreement over the committed set:
    /// directory, global path, both directional indexes and the committed record
    /// count must all agree.
    ///
    /// # Errors
    /// `PipelineError::Mismatch` naming the first divergence found.
    pub fn check_index_consistency(&self) -> Result<(), PipelineError> {
        // `rows_len` counts by code index (0..=max_code); the dictionary
        // reserves code 0, so its row-table length is `len()` (= real
        // accounts + 1 sentinel), not `account_count()`.
        self.temporal.check(
            self.committed(),
            self.accounts().len(),
            self.accounts().len(),
        )
    }

    // -----------------------------------------------------------------------
    // CSF tensor read boundary (rows 15/16/17, re-exposing the live substrate
    // tensor without adding algebra). `T_live = T_sealed + frontier` (row 23),
    // so every committed event is visible here with its exact micro-unit value.
    // Read-only: nothing here mutates the canonical substrate object.
    // -----------------------------------------------------------------------

    /// `T_live(t, s, r)` — exact live value at coordinate `(t, s, r)`; `None`
    /// when the coordinate is absent from the support (never fabricated as zero,
    /// P3).
    #[must_use]
    pub fn csf_live_at(&self, t: u64, s: u32, r: u32) -> Option<i64> {
        let v = self.applier.engine().live_at(t, s, r).ok()?;
        (v != 0).then_some(v)
    }

    /// `CSF(T_sealed)` — the canonical tensor object the GraphBLAS API layer
    /// (`abacus_tensor::api::graphblas`, Addendum O rows 30–51) consumes.
    /// Read-only shared borrow: the canonical object is neither rebuilt nor
    /// re-materialized here (the GraphBLAS operations read only through the
    /// row-27 boundary, exactly as the API layer dictates).
    ///
    /// Scope: the sealed tensor `T_sealed` only — the committed-but-unsealed
    /// frontier overlay is excluded (call [`Self::seal_substrate`] first to
    /// compact pending commits into the sealed tensor; row 22).
    #[must_use]
    pub fn csf_sealed(&self) -> &CanonicalTensor {
        self.applier.engine().sealed()
    }

    /// Compact the committed frontier into the sealed canonical tensor
    /// (row 22 sealing). Additive operator control: GraphBLAS endpoints read
    /// the sealed tensor, so a client that needs freshly committed records in
    /// GraphBLAS results calls `POST /api/ingest/graphblas/seal` first.
    ///
    /// # Errors
    /// [`PipelineError`] wrapping the structured seal failure reported by the
    /// substrate path (fail-closed; nothing is committed to the sealed tensor
    /// on error). The carried detail is structural only, never raw account
    /// identifiers or amounts (Golden Source §16).
    pub fn seal_substrate(&mut self) -> Result<u64, PipelineError> {
        let report = self
            .applier
            .seal()
            .map_err(|e| PipelineError::Invalid(format!("seal failed: {e}")))?;
        Ok(report.sealed_nnz_after as u64)
    }

    /// `⟦T_live⟧` — full live decode as `(t, s, r, value)` entries in canonical
    /// `t → s → r` order. `value` is the exact micro-unit tensor value.
    #[must_use]
    pub fn csf_decode_live(&self) -> Vec<(u64, u32, u32, i64)> {
        let locators: Vec<u32> = self.temporal().global_range(0, u64::MAX).to_vec();
        let mut out = Vec::with_capacity(locators.len());
        for loc in locators {
            if let Some((t, s, r, _feats)) = self.resolve(loc) {
                if let Some(value) = self.csf_live_at(t, s, r) {
                    out.push((t, s, r, value));
                }
            }
        }
        out
    }

    /// Read-only CSF tensor statistics (`nnz` = live nonzero count).
    #[must_use]
    pub fn csf_stats(&self) -> CsfStats {
        CsfStats {
            nnz: self.csf_decode_live().len() as u64,
        }
    }

    /// Derived amount-based feature metrics for every committed event whose
    /// live coordinate is exactly `(t, s, r)`. Because the CSF aggregates by
    /// coordinate (multiple events may share one `(t, s, r)` and sum into a
    /// single tensor `value`), this returns one payload per contributing event
    /// rather than a single fabricated aggregate (privacy-safe: derived metrics
    /// only, never the raw amount — GS §9).
    #[must_use]
    pub fn csf_features_at(&self, t: u64, s: u32, r: u32) -> Vec<FeaturePayload> {
        let mut out = Vec::new();
        // Narrow by time first (t-major), then match the full coordinate.
        for loc in self.temporal().global_range(t, t.saturating_add(1)) {
            if let Some((et, es, er, feats)) = self.resolve(*loc) {
                if (et, es, er) == (t, s, r) {
                    out.push(feats);
                }
            }
        }
        out
    }

    /// Live sender-at-time view `(t, s) → [(r, value)]` over `T_live =
    /// T_sealed + ⟦C_frontier⟧` (row 23), in `prec_3` (`r`) order. Absent
    /// coordinates are never reported (P3); zero-summed entries are dropped.
    #[must_use]
    pub fn csf_sender_at_time(&self, t: u64, s: u32) -> Vec<(u32, i64)> {
        let engine = self.applier.engine();
        let mut out: BTreeMap<u32, i64> = BTreeMap::new();
        // Sealed contribution (binary-search O(log n + degree)).
        for (r, v) in sealed_sender_at_time(engine.sealed(), t, s) {
            out.insert(r, v);
        }
        // Frontier contribution: adjust existing `r` fibers and add new ones.
        for e in engine.frontier().entries() {
            if e.t == t && e.s == s {
                let cur = *out.get(&e.r).unwrap_or(&0);
                if let Some(sum) = cur.checked_add(e.value) {
                    out.insert(e.r, sum);
                }
            }
        }
        out.into_iter().filter(|(_, v)| v != &0).collect()
    }

    /// Live receiver-at-time view `(t, r) → [(s, value)]` over `T_live =
    /// T_sealed + ⟦C_frontier⟧` (row 23), in `prec_3` (`s`) order. Absent
    /// coordinates are never reported (P3); zero-summed entries are dropped.
    #[must_use]
    pub fn csf_receiver_at_time(&self, t: u64, r: u32) -> Vec<(u32, i64)> {
        let engine = self.applier.engine();
        let mut out: BTreeMap<u32, i64> = BTreeMap::new();
        // Sealed contribution.
        for (s, v) in sealed_receiver_at_time(engine.sealed(), t, r) {
            out.insert(s, v);
        }
        // Frontier contribution: adjust existing `s` fibers and add new ones.
        for e in engine.frontier().entries() {
            if e.t == t && e.r == r {
                let cur = *out.get(&e.s).unwrap_or(&0);
                if let Some(sum) = cur.checked_add(e.value) {
                    out.insert(e.s, sum);
                }
            }
        }
        out.into_iter().filter(|(_, v)| v != &0).collect()
    }

    /// Apply one canonical ingestion record.
    ///
    /// Order (fail-closed, nothing mutates on rejection):
    /// 1. classify the compound identity (New / Duplicate / Conflict);
    /// 2. on New, derive the frozen feature payload from the attested amount;
    /// 3. apply through the substrate applier (interns accounts into the shared
    ///    namespace, converts ms->s, writes the attested value);
    /// 4. index `(t, s, r)` and record the derived features.
    ///
    /// # Errors
    /// - `PipelineError::Invalid` on an identity conflict or a substrate/feature
    ///   rejection (missing/negative amount, pre-epoch time, feature overflow,
    ///   capacity).
    pub fn apply(
        &mut self,
        record: &CanonicalRecord,
        attested_minor_units: Option<i64>,
        threshold_minor_units: u64,
        major_scale: u64,
    ) -> Result<IngestApplyOutcome, PipelineError> {
        // 1. Identity-ledger classification (authoritative dedup, G2 ruling A).
        let content = IdentityLedger::content_digest(
            record.event_time_ms,
            record.sender_id,
            record.receiver_id,
            record.features.log_magnitude,
            record.features.relative_threshold_distance,
            record.features.roundedness_flags,
        );
        let (outcome, substrate_id) = self
            .identity
            .check(record.identity, content)
            .map_err(|e| PipelineError::Invalid(e.to_string()))?;
        if !matches!(outcome, IdentityOutcome::New) || substrate_id.is_none() {
            return Ok(IngestApplyOutcome::Duplicate);
        }
        let substrate_id = substrate_id.unwrap_or_default();

        // 2. Derived metrics from the exact amount (privacy: features, not the
        //    raw amount, are what the pipeline carries for the UI/graph).
        let amount_minor = attested_minor_units.ok_or_else(|| {
            PipelineError::Invalid("attested minor-unit amount required (A1)".into())
        })?;
        if amount_minor < 0 {
            return Err(PipelineError::Invalid(
                "attested minor-unit amount must be nonnegative".into(),
            ));
        }
        let features =
            FeaturePayload::derive(amount_minor as u64, threshold_minor_units, major_scale)
                .map_err(|e| PipelineError::Invalid(e.to_string()))?;

        // 3. Substrate application (shared namespace interning + ms->s + value).
        let applied = self
            .applier
            .apply(record, substrate_id, attested_minor_units)
            .map_err(|e| PipelineError::Invalid(e.to_string()))?;
        let ApplyOutcome::Applied { t, s, r, .. } = applied else {
            // Substrate CAS retry (model-level already-applied): duplicate.
            return Ok(IngestApplyOutcome::Duplicate);
        };

        // 4. Index over the shared namespace + record derived features.
        let locator = self
            .temporal
            .insert_event(t, s, r)
            .map_err(|_| PipelineError::Invalid("locator capacity exhausted (u32)".into()))?;
        self.features.push(features);
        self.identities.push(record.identity);
        Ok(IngestApplyOutcome::Committed(locator.0))
    }

    /// Apply one session-validated [`PreparedCsv`] upload to this pipeline.
    ///
    /// Each `(record, attested amount)` pair goes through [`Self::apply`] with
    /// the chunk header's `threshold_minor_units` and its config's
    /// `major_scale` (`10^currency_precision`, Golden Source §9.3), so feature
    /// re-derivation matches the canonical path exactly. A `Duplicate`
    /// outcome counts as already-applied; any `Err` (identity conflict,
    /// substrate/feature rejection) counts as rejected and stops nothing else
    /// — every record is independent, and `apply` itself never partially
    /// commits.
    ///
    /// Callers must pass uploads produced by the canonical prepare path
    /// (`abacus_mvp_ingest::{prepare_csv_bytes, prepare_samld_bytes}`), which
    /// runs the same `normalize → session submit → seal` validation the
    /// `IngestionService` path runs. No canonical rule is reimplemented here.
    pub fn apply_prepared(
        &mut self,
        prepared: &abacus_mvp_ingest::PreparedCsv,
    ) -> Result<PreparedApplyOutcome, PipelineError> {
        let mut out = PreparedApplyOutcome {
            applied: 0,
            already_applied: 0,
            rejected: 0,
        };
        for chunk in &prepared.chunks {
            let threshold = chunk.header.threshold_minor_units;
            let major_scale = abacus_ingest_major_scale(chunk.header.currency_precision)
                .ok_or_else(|| {
                    PipelineError::Invalid("10^precision is not representable in u64".into())
                })?;
            for (record, amount) in &chunk.records {
                match self.apply(record, Some(*amount), threshold, major_scale) {
                    Ok(IngestApplyOutcome::Committed(_)) => out.applied += 1,
                    Ok(IngestApplyOutcome::Duplicate) => out.already_applied += 1,
                    Err(_) => out.rejected += 1,
                }
            }
        }
        Ok(out)
    }
}

/// Outcome of applying one validated [`PreparedCsv`](abacus_mvp_ingest::PreparedCsv)
/// upload through [`IngestPipeline::apply_prepared`].
///
/// Counts mirror the oracle session stages that method replaces at apply
/// time: records the pipeline newly committed, records it classified as
/// duplicate redeliveries, and records it rejected (conflict or apply
/// failure — fail-closed, nothing partially commits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedApplyOutcome {
    /// Records newly committed to this pipeline.
    pub applied: u64,
    /// Records classified as duplicate redeliveries (no state change).
    pub already_applied: u64,
    /// Records rejected (conflict or apply failure).
    pub rejected: u64,
}

/// `10^precision` in exact integer arithmetic (Golden Source §9.3) — the same
/// computation as `SourceConfig::major_scale`, over the validated header's
/// precision byte. `None` when unrepresentable (fail-closed at the call site).
fn abacus_ingest_major_scale(currency_precision: u8) -> Option<u64> {
    10_u64
        .checked_pow(u32::from(currency_precision))
        .filter(|&s| s > 0)
}

impl Default for IngestPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abacus_ingest::features::FeaturePayload;
    use abacus_ingest::namespace::EventNamespace;

    fn ns(byte: u8) -> EventNamespace {
        EventNamespace::from_bytes([byte; 16])
    }

    fn rec(event_id: u64, s: u64, r: u64) -> CanonicalRecord {
        CanonicalRecord {
            identity: EventIdentity {
                namespace: ns(1),
                event_id,
            },
            event_time_ms: 1_700_000_000_000,
            sender_id: s,
            receiver_id: r,
            features: FeaturePayload {
                log_magnitude: 1,
                relative_threshold_distance: 0,
                roundedness_flags: 0,
            },
        }
    }

    #[test]
    fn commits_new_event_and_indexes_shared_namespace() {
        let mut p = IngestPipeline::new();
        let out = p
            .apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        assert_eq!(out, IngestApplyOutcome::Committed(0));
        assert_eq!(p.committed(), 1);
        // Derived metrics carried, not the raw amount.
        let ev = p.event(0).unwrap();
        assert_eq!(ev.features.log_magnitude, 9_421);
        assert_eq!(ev.features.relative_threshold_distance, 235);
        // Shared namespace: sender and receiver intern into one map.
        assert_eq!(p.accounts().account_count(), 2);
    }

    #[test]
    fn identity_ledger_dedups_redelivery() {
        let mut p = IngestPipeline::new();
        let r = rec(0, 100, 200);
        assert_eq!(
            p.apply(&r, Some(12_345), 10_000, 100).unwrap(),
            IngestApplyOutcome::Committed(0)
        );
        // Same (namespace, event_id) redelivered -> Duplicate, no new state.
        assert_eq!(
            p.apply(&r, Some(12_345), 10_000, 100).unwrap(),
            IngestApplyOutcome::Duplicate
        );
        assert_eq!(p.committed(), 1);
    }

    #[test]
    fn conflicting_content_for_same_identity_is_rejected() {
        let mut p = IngestPipeline::new();
        assert!(p
            .apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .is_ok());
        let mut conflict = rec(0, 100, 200);
        conflict.features.log_magnitude = 2; // different content digest
        let err = p.apply(&conflict, Some(9_999), 10_000, 100);
        assert!(matches!(err, Err(PipelineError::Invalid(_))));
        assert_eq!(p.committed(), 1);
    }

    #[test]
    fn missing_attested_amount_is_rejected() {
        let mut p = IngestPipeline::new();
        let err = p.apply(&rec(0, 100, 200), None, 10_000, 100);
        assert!(matches!(err, Err(PipelineError::Invalid(_))));
        assert_eq!(p.committed(), 0);
    }

    #[test]
    fn shared_namespace_label_is_role_independent() {
        // The retired role-separated model gave the same account DIFFERENT codes
        // as sender vs receiver. The shared namespace gives ONE code + ONE label.
        let mut p = IngestPipeline::new();
        // Account 100 sends in event 0, receives in event 1.
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100)
            .unwrap();
        let s0 = p.resolve(0).unwrap().1;
        let r1 = p.resolve(1).unwrap().2;
        assert_eq!(
            s0, r1,
            "same wire account -> same shared code in both roles"
        );
        assert_eq!(p.account_label(s0), Some("100".into()));
    }

    #[test]
    fn outgoing_and_incoming_queries_use_shared_namespace() {
        let mut p = IngestPipeline::new();
        // Both events share the same ms timestamp -> same substrate t (seconds).
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100)
            .unwrap();
        let (t, acct100, _, _) = p.resolve(0).unwrap();
        // Outgoing from 100 in a window covering its t: the event where it sent.
        assert_eq!(p.outgoing_in_range(acct100, t, t + 1, 100), vec![0]);
        // Incoming to 100 in the same window: the event where it received.
        assert_eq!(p.incoming_in_range(acct100, t, t + 1, 100), vec![1]);
        // A window that excludes t yields nothing.
        assert!(p
            .outgoing_in_range(acct100, t + 1000, t + 2000, 100)
            .is_empty());
    }

    #[test]
    fn csf_sender_at_time_exposes_sender_dimension() {
        let mut p = IngestPipeline::new();
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100)
            .unwrap();
        let t = 1_700_000_000u64;
        let (_, s0, r0, _) = p.resolve(0).unwrap(); // event 0: s0 -> r0
        let (_, s1, r1, _) = p.resolve(1).unwrap(); // event 1: s1 -> r1
        assert_eq!(p.csf_sender_at_time(t, s0), vec![(r0, 12_345_000_000)]);
        assert_eq!(p.csf_sender_at_time(t, s1), vec![(r1, 12_345_000_000)]);
        // Unknown sender -> empty (never fabricated).
        assert!(p.csf_sender_at_time(t, 9).is_empty());
    }

    #[test]
    fn csf_receiver_at_time_exposes_receiver_dimension() {
        let mut p = IngestPipeline::new();
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100)
            .unwrap();
        let t = 1_700_000_000u64;
        let (_, s0, r0, _) = p.resolve(0).unwrap(); // event 0: s0 -> r0
        let (_, s1, r1, _) = p.resolve(1).unwrap(); // event 1: s1 -> r1
        assert_eq!(p.csf_receiver_at_time(t, r0), vec![(s0, 12_345_000_000)]);
        assert_eq!(p.csf_receiver_at_time(t, r1), vec![(s1, 12_345_000_000)]);
        // Unknown receiver -> empty (never fabricated).
        assert!(p.csf_receiver_at_time(t, 9).is_empty());
    }

    #[test]
    fn csf_features_at_exposes_amount_derived_metrics() {
        let mut p = IngestPipeline::new();
        // amount=12_345 minor, threshold=10_000, precision 2:
        // log_magnitude=9421, relative_threshold_distance=235, roundedness=0.
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        let t = 1_700_000_000u64;
        let (_, s, r, _) = p.resolve(0).unwrap();
        let feats = p.csf_features_at(t, s, r);
        assert_eq!(feats.len(), 1);
        assert_eq!(feats[0].log_magnitude, 9_421);
        assert_eq!(feats[0].relative_threshold_distance, 235);
        assert_eq!(feats[0].roundedness_flags, 0);
        // Unknown coordinate -> empty (never fabricated).
        assert!(p.csf_features_at(t, 9, 9).is_empty());
    }
}
