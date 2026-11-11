//! Incremental update path: T_{n+1} = Apply(T_n, e_{n+1}).
//!
//! Canonical mechanism reused: ledger CAS (row 18, P7) + elementary delta
//! (P8) + frontier append (row 19, P9) + live view (row 23). The sealed CSF
//! is immutable between seals; per-event visibility comes from the frontier
//! overlay. Seal (row 21, full rebuild) is available as explicit compaction
//! only — never on the per-event path.
//!
//! Identity contract: event id = next tx_id (== committed record count).
//! Redelivery of an already-committed id is rejected as Duplicate without
//! state change (model-level exactly-once via the CAS equality check, P7).
//! Unknown accounts extend dictionaries append-only (prec_dict preserved
//! for existing codes: existing codes never shift).
//!
//! Timestamp policy (explicit contract — SUPPORT/out-of-order accepted):
//! any event time `t` (earlier, equal, later, or spanning the current
//! `[t_min, t_max]` bounds) is accepted as an ordinary Create through the
//! same CAS path, per canonical P10 (watermark stored, never consulted;
//! `ingest/engine.rs`, `watermark_is_stored_not_consulted` test). Committed
//! bounds extend monotonically (`t_min = min`, `t_max = max`). Event-time
//! ordering in queries is exact without per-event rebuilds: `live_at` is a
//! coordinate-exact point read; `live_slice` restricts the sealed CSF to the
//! half-open window and filters the frontier by the same window, then
//! settles (P1 `prec_3` sort). No event is ever silently placed in the wrong
//! temporal position: coordinates are stored verbatim and reads filter +
//! sort by the stored `t`.
//!
//! Durability (see `persist.rs`): `apply` is the in-memory commit;
//! `apply_durable` journals (append + fsync) BEFORE the in-memory commit so
//! a crash can only lose the in-flight event, never expose a partial commit.
//! Validation always precedes any state mutation (fail before modify).

use crate::error::PipelineError;
use crate::loader::LoadedArtifacts;
use crate::retrieval::{EventLocator, TemporalIndexes};
use crate::state::{AccountIndex, PipelineSnapshot, StateVersion};
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::ingest::{CasOutcome, Expected, Ledger, LedgerOp};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

/// One normalized transaction event (pre-canonicalized amounts).
#[derive(Debug, Clone)]
pub struct NormalizedEvent {
    /// Event time t (epoch seconds).
    pub t: u64,
    /// Sender account identifier (verbatim).
    pub sender: String,
    /// Receiver account identifier (verbatim).
    pub receiver: String,
    /// Canonical amount in micro-units (P2; nonzero).
    pub amount_micro: i64,
}

/// Outcome of one `apply` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Committed; carries the new version.
    Committed(StateVersion),
    /// Rejected streaming_ingestion_demo; state unchanged.
    Duplicate(StateVersion),
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Max frontier length before `apply` advises compaction (never forces).
    /// Read by operators to schedule explicit `seal()` compaction.
    pub frontier_advise: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self { frontier_advise: 10_000 }
    }
}

/// Per-event instrumentation: nanosecond latency (monotonic clock).
/// `latency_us` is kept for interface compatibility (truncated, may be 0);
/// use `latency_ns` for real measurements.
#[derive(Debug, Clone, Copy, Default)]
pub struct EventMetrics {
    /// Total apply latency (nanoseconds, monotonic).
    pub latency_ns: u128,
    /// Total apply latency (microseconds, truncated; 0 is possible).
    pub latency_us: u128,
    /// Records touched (always 1 on commit).
    pub records_touched: usize,
    /// Index entries touched (sender + receiver sets).
    pub index_entries_touched: usize,
}

/// Bounded result of a directional range lookup over the temporal indexes.
#[derive(Debug, Clone, Default)]
pub struct DirectedLookup {
    /// True when the time window held more events than `max_edges`; the
    /// returned `locators` are therefore a possibly-incomplete prefix.
    pub is_truncated: bool,
    /// Event locators in ascending `(t, locator)` order, capped at `max_edges`.
    pub locators: Vec<EventLocator>,
}

/// The incremental pipeline: serialized writer + atomically published
/// snapshot + ledger + dictionaries.
pub struct Pipeline {
    snapshot: PipelineSnapshot,
    ledger: Ledger,
    next_tx: u64,
    sender_codes: BTreeMap<String, u32>,
    receiver_codes: BTreeMap<String, u32>,
    event_ids: HashMap<(u64, u32, u32, i64), u64>,
    /// Operator tuning (frontier compaction advisory threshold).
    pub config: PipelineConfig,
    /// Committed per-event latencies (microseconds, truncated; kept for
    /// interface compatibility — may contain 0 for sub-microsecond work).
    pub latencies_us: Vec<u128>,
    /// Committed per-event latencies (nanoseconds, monotonic clock).
    /// This is the authoritative per-event series for benchmarking.
    pub latencies_ns: Vec<u128>,
    /// Rejected-event count.
    pub rejected: u64,
    /// Seal (full rebuild) count; must stay 0 during incremental runs.
    pub seals: u64,
    /// Initial load duration (microseconds).
    pub load_us: u128,
}
impl Pipeline {
    /// Build initial state T_0 from validated artifacts.
    #[must_use]
    pub fn from_loaded(art: LoadedArtifacts, config: PipelineConfig, load_us: u128) -> Self {
        let sealed_entries = abacus_tensor::csf::decode_csf(&art.csf.csf);
        let index = AccountIndex::build(&sealed_entries, &[]);
        let mut sender_codes = BTreeMap::new();
        for (i, s) in art.core.senders.iter().enumerate() {
            sender_codes.insert(s.clone(), i as u32);
        }
        let mut receiver_codes = BTreeMap::new();
        for (i, s) in art.core.receivers.iter().enumerate() {
            receiver_codes.insert(s.clone(), i as u32);
        }
        let n = art.core.records.len() as u64;
        let mut ledger = Ledger::new();
        for (i, r) in art.core.records.iter().enumerate() {
            let id = i as u64;
            let _ = ledger.ledger_op(id, LedgerOp::Create(r.amount_micro));
        }
        // Phase-3 derived temporal state: locator = core row ordinal = the
        // same event id the ledger just created (identity contract). Explicit
        // loud failure if artifacts ever exceed u32 locator capacity — never
        // a silent wrap (task spec §3 overflow policy).
        let temporal = TemporalIndexes::try_build(
            art.core.records.iter().map(|r| (r.t, r.s, r.r)),
            art.core.senders.len(),
            art.core.receivers.len(),
        )
        .expect("core record count exceeds u32 locator capacity");
        let snapshot = PipelineSnapshot {
            version: 0,
            sealed: art.csf.csf.clone(),
            frontier: Vec::new(),
            index,
            committed_records: n as usize,
            t_min: art.core.t_min,
            t_max: art.core.t_max,
            senders: art.core.senders.clone(),
            receivers: art.core.receivers.clone(),
            temporal,
        };
        Self {
            snapshot, ledger, next_tx: n,
            sender_codes, receiver_codes,
            event_ids: HashMap::new(),
            config, latencies_us: Vec::new(), latencies_ns: Vec::new(),
            rejected: 0, seals: 0, load_us,
        }
    }

    /// Current committed snapshot (whole-snapshot borrow).
    #[must_use]
    pub fn snapshot(&self) -> &PipelineSnapshot {
        &self.snapshot
    }

    /// Sender-dictionary code for an account name (retrieval needs the
    /// role-separated code spaces; `None` = account never sent).
    #[must_use]
    pub fn sender_code(&self, name: &str) -> Option<u32> {
        self.sender_codes.get(name).copied()
    }

    /// Receiver-dictionary code for an account name (`None` = account never
    /// received).
    #[must_use]
    pub fn receiver_code(&self, name: &str) -> Option<u32> {
        self.receiver_codes.get(name).copied()
    }

    /// Fail-closed cross-check: the committed record count must equal the
    /// number of events registered in the temporal indexes (locator
    /// directory). Returns `Err` if the two canonical sources disagree.
    pub fn check_index_consistency(&self) -> Result<(), PipelineError> {
        let committed = self.snapshot.committed_records;
        let indexed = self.snapshot.temporal.event_count();
        if committed != indexed {
            return Err(PipelineError::Mismatch(format!(
                "index consistency: committed {committed} != indexed events {indexed}"
            )));
        }
        Ok(())
    }

    /// Bounded directional lookup over the committed temporal indexes.
    ///
    /// Returns the locators (event ids) for events where `name` is the sender
    /// within `[t0, t1)`, capped at `max_edges`. `is_truncated` is true when
    /// the window held more events than `max_edges`, so callers can report an
    /// incomplete neighborhood honestly rather than silently dropping edges.
    /// Amounts are never touched — structural identifiers only.
    #[must_use]
    pub fn get_outgoing_in_range(
        &self,
        name: &str,
        t0: u64,
        t1: u64,
        max_edges: usize,
    ) -> DirectedLookup {
        let Some(code) = self.sender_code(name) else {
            return DirectedLookup {
                is_truncated: false,
                locators: Vec::new(),
            };
        };
        let row = self.snapshot.temporal.forward.row(code);
        let lo = row.partition_point(|e| e.t < t0);
        let hi = row.partition_point(|e| e.t < t1);
        let total = hi.saturating_sub(lo);
        let take = total.min(max_edges);
        let locators: Vec<EventLocator> = row[lo..lo + take]
            .iter()
            .map(|e| e.locator)
            .collect();
        DirectedLookup {
            is_truncated: total > max_edges,
            locators,
        }
    }

    /// Mirror of [`get_outgoing_in_range`] for the receiver's backward index:
    /// locators of events where `name` is the receiver within `[t0, t1)`.
    #[must_use]
    pub fn get_incoming_in_range(
        &self,
        name: &str,
        t0: u64,
        t1: u64,
        max_edges: usize,
    ) -> DirectedLookup {
        let Some(code) = self.receiver_code(name) else {
            return DirectedLookup {
                is_truncated: false,
                locators: Vec::new(),
            };
        };
        let row = self.snapshot.temporal.backward.row(code);
        let lo = row.partition_point(|e| e.t < t0);
        let hi = row.partition_point(|e| e.t < t1);
        let total = hi.saturating_sub(lo);
        let take = total.min(max_edges);
        let locators: Vec<EventLocator> = row[lo..lo + take]
            .iter()
            .map(|e| e.locator)
            .collect();
        DirectedLookup {
            is_truncated: total > max_edges,
            locators,
        }
    }

    /// Live point read: sealed + frontier overlay (row 23 shape).
    #[must_use]
    pub fn live_at(&self, t: u64, s: u32, r: u32) -> i64 {
        let mut v = abacus_tensor::csf::query::lookup(&self.snapshot.sealed, t, s, r);
        for e in &self.snapshot.frontier {
            if e.t == t && e.s == s && e.r == r {
                v = v.saturating_add(e.value);
            }
        }
        v
    }
}
/// Resolved commit plan: codes + identity (no state touched yet).
struct ReadyCommit {
    s: u32,
    r: u32,
    new_s: bool,
    new_r: bool,
    key: (u64, u32, u32, i64),
    id: u64,
}

/// Outcome of the fail-before-mutate pre-check (no state touched).
enum ApplyCommit {
    /// Reject with message (validation / ledger guard).
    Reject(String),
    /// Redelivery of committed content (no state change).
    Duplicate,
    /// Ready to commit (codes + identity resolved).
    Ready(ReadyCommit),
}

/// Explicit capacity policy (task spec §3: "Use an explicit overflow policy;
/// never wrap silently"; §9 test 23: "Locator or entity-ID exhaustion fails
/// explicitly without corrupting state"):
///
/// - next event id (== locator) must fit in `u32`;
/// - a newly allocated sender/receiver code must fit in `u32`.
///
/// Pure and side-effect free so it is unit-testable without allocating
/// 4-billion-entry dictionaries. `Err` becomes `ApplyCommit::Reject` — the
/// event is rejected before any state mutation.
fn capacity_guard(
    next_tx: u64,
    new_sender_rows: Option<usize>,
    new_receiver_rows: Option<usize>,
) -> Result<(), String> {
    if next_tx > u32::MAX as u64 {
        return Err(format!(
            "locator capacity exhausted: next event id {next_tx} exceeds u32::MAX"
        ));
    }
    if new_sender_rows.is_some_and(|len| len > u32::MAX as usize) {
        return Err("sender dictionary exhausted (u32 entity codes)".into());
    }
    if new_receiver_rows.is_some_and(|len| len > u32::MAX as usize) {
        return Err("receiver dictionary exhausted (u32 entity codes)".into());
    }
    Ok(())
}

impl Pipeline {
    /// Apply one normalized event: validate, CAS, append, index, commit.
    /// Failure leaves the last committed snapshot unchanged (atomicity).
    ///
    /// Validation (zero amount, empty account) runs BEFORE any state
    /// mutation: a rejected event never modifies the snapshot, ledger,
    /// dictionaries, or counters other than `rejected`.
    ///
    /// Timestamp policy: any `t` is accepted (SUPPORT/out-of-order per P10);
    /// bounds extend via min/max. Callers needing strict ordering must gate
    /// before calling — this function never rejects on `t`.
    ///
    /// # Errors
    /// `Invalid` on malformed events (zero amount, empty account).
    pub fn apply(&mut self, ev: &NormalizedEvent) -> Result<ApplyOutcome, PipelineError> {
        let t0 = Instant::now();
        let commit = self.check_apply(ev)?;
        match commit {
            ApplyCommit::Reject(msg) => {
                self.rejected += 1;
                return Err(PipelineError::Invalid(msg));
            }
            ApplyCommit::Duplicate => {
                self.rejected += 1;
                return Ok(ApplyOutcome::Duplicate(self.snapshot.version));
            }
            ApplyCommit::Ready(plan) => {
                self.commit_ready(ev, plan);
            }
        }
        self.snapshot.version += 1;
        debug_assert!(self.snapshot.sealed.check_invariants());
        debug_assert_eq!(
            self.snapshot.temporal.event_count(),
            self.snapshot.committed_records,
            "tensor and temporal indexes agree on the committed set"
        );
        let ns = t0.elapsed().as_nanos();
        self.latencies_ns.push(ns);
        self.latencies_us.push(ns / 1_000);
        Ok(ApplyOutcome::Committed(self.snapshot.version))
    }

    /// Fail-before-mutate pre-check shared by `apply` and `apply_durable`.
    /// Returns the commit plan without touching any state.
    fn check_apply(&self, ev: &NormalizedEvent) -> Result<ApplyCommit, PipelineError> {
        if ev.amount_micro == 0 {
            return Ok(ApplyCommit::Reject("zero amount (P3)".into()));
        }
        if ev.sender.is_empty() || ev.receiver.is_empty() {
            return Ok(ApplyCommit::Reject("empty account".into()));
        }
        let s_res = self.sender_codes.get(&ev.sender).copied();
        let r_res = self.receiver_codes.get(&ev.receiver).copied();
        // Capacity guard runs before any code cast or state access: locator
        // and entity-code exhaustion reject here, fail-before-mutate
        // (task spec §3/§9 test 23 — explicit, never a silent wrap).
        if let Err(msg) = capacity_guard(
            self.next_tx,
            s_res.is_none().then_some(self.snapshot.senders.len()),
            r_res.is_none().then_some(self.snapshot.receivers.len()),
        ) {
            return Ok(ApplyCommit::Reject(msg));
        }
        let s = s_res.unwrap_or(self.snapshot.senders.len() as u32);
        let r = r_res.unwrap_or(self.snapshot.receivers.len() as u32);
        let key = (ev.t, s, r, ev.amount_micro);
        if self.event_ids.contains_key(&key) {
            return Ok(ApplyCommit::Duplicate);
        }
        let id = self.next_tx;
        if !matches!(self.ledger.get(id), Expected::Absent) {
            return Ok(ApplyCommit::Reject(format!("ledger id {id} not absent")));
        }
        Ok(ApplyCommit::Ready(ReadyCommit {
            s,
            r,
            new_s: s_res.is_none(),
            new_r: r_res.is_none(),
            key,
            id,
        }))
    }

    /// Second half of `apply`: CAS + append + index + commit. The pre-check
    /// guarantees the CAS succeeds; a CAS failure here is still fail-closed.
    fn commit_ready(&mut self, ev: &NormalizedEvent, plan: ReadyCommit) {
        let ReadyCommit { s, r, new_s, new_r, key, id } = plan;
        let outcome = self.ledger.ledger_op(id, LedgerOp::Create(ev.amount_micro));
        debug_assert!(matches!(outcome, CasOutcome::Committed { .. }));
        if !matches!(outcome, CasOutcome::Committed { .. }) {
            self.rejected += 1;
            return;
        }
        if new_s {
            let c = self.snapshot.senders.len() as u32;
            self.snapshot.senders.push(ev.sender.clone());
            self.sender_codes.insert(ev.sender.clone(), c);
        }
        if new_r {
            let c = self.snapshot.receivers.len() as u32;
            self.snapshot.receivers.push(ev.receiver.clone());
            self.receiver_codes.insert(ev.receiver.clone(), c);
        }
        let entry = TxEntry::new(ev.t, s, r, ev.amount_micro);
        self.snapshot.frontier.push(entry);
        self.snapshot.index.insert(&entry);
        // Phase-3 temporal indexes (forward, backward, global, directory):
        // locator == event id by the identity contract (directory length ==
        // committed count). Capacity was pre-checked fail-before-mutate in
        // `check_apply`; the Err branch would be a silent wrap elsewhere, so
        // it is surfaced as an explicit panic here — unreachable in practice.
        let locator = self
            .snapshot
            .temporal
            .insert_event(ev.t, s, r)
            .unwrap_or_else(|e| panic!("locator capacity pre-checked in check_apply: {e}"));
        debug_assert_eq!(locator.0 as u64, id, "locator must equal event id");
        self.event_ids.insert(key, id);
        self.next_tx += 1;
        self.snapshot.committed_records += 1;
        if ev.t < self.snapshot.t_min {
            self.snapshot.t_min = ev.t;
        }
        if ev.t > self.snapshot.t_max {
            self.snapshot.t_max = ev.t;
        }
    }

    /// Durable apply: journal (append + fsync) BEFORE the in-memory commit.
    /// Pre-check (validation AND duplicate detection) runs BEFORE journaling:
    /// invalid events and redelivered duplicates are never journaled, so the
    /// journal holds only committed intents and recovery streaming_ingestion_demos each once.
    /// A crash between journal fsync and commit streaming_ingestion_demos the event on
    /// recovery; a crash before the fsync loses only the in-flight event.
    ///
    /// # Errors
    /// `Invalid` on malformed events; `Io` on journal failure (state
    /// unchanged in both cases).
    pub fn apply_durable(
        &mut self,
        ev: &NormalizedEvent,
        journal_path: &str,
    ) -> Result<ApplyOutcome, PipelineError> {
        match self.check_apply(ev)? {
            ApplyCommit::Reject(msg) => {
                self.rejected += 1;
                return Err(PipelineError::Invalid(msg));
            }
            ApplyCommit::Duplicate => {
                self.rejected += 1;
                return Ok(ApplyOutcome::Duplicate(self.snapshot.version));
            }
            ApplyCommit::Ready(_) => {}
        }
        crate::persist::journal_append(journal_path, ev)?;
        self.apply(ev)
    }
}

/// Owned checkpoint payload passed to [`Pipeline::install_checkpoint`].
/// Frontier events are NOT included — they must be journaled and streaming_ingestion_demoed
/// so the content-keyed duplicate map rebuilds exactly.
pub struct CheckpointImage {
    /// Sealed canonical CSF.
    pub sealed: abacus_tensor::csf::compress::CsfTensor,
    /// Full sender dictionary.
    pub senders: Vec<String>,
    /// Full receiver dictionary.
    pub receivers: Vec<String>,
    /// Committed version.
    pub version: StateVersion,
    /// Committed record count.
    pub committed_records: usize,
    /// Temporal bounds.
    pub t_min: u64,
    /// Temporal bounds.
    pub t_max: u64,
}

impl Pipeline {
    /// Install a checkpoint image during recovery: sealed CSF + full dicts +
    /// scalars. Rebuilds derived state (index, code maps, ledger) so the
    /// recovered pipeline preserves locator stability and the tx_id identity
    /// contract. Frontier restarts empty.
    ///
    /// DUPLICATE MAP: `event_ids` restarts empty here; [`Self::restore_duplicate_keys`]
    /// then re-inserts the content keys of journal records already compacted
    /// into this checkpoint, so redelivery of pre-checkpoint content is still
    /// detected as a Duplicate after recovery (identical to baseline
    /// `from_loaded` semantics, which only cover in-process history).
    ///
    /// # Errors
    /// `Mismatch` if the sealed image fails invariants.
    pub fn install_checkpoint(&mut self, img: CheckpointImage) -> Result<(), PipelineError> {
        use std::collections::BTreeMap;
        let CheckpointImage { sealed, senders, receivers, version, committed_records, t_min, t_max } =
            img;
        if !sealed.check_invariants() {
            return Err(PipelineError::Mismatch("checkpoint CSF invariant failure".into()));
        }
        // Fail-closed: the checkpoint dictionaries must extend the artifact
        // dictionaries with codes preserved (`prec_dict` — existing codes
        // never shift). Otherwise every stored locator/entity code would
        // silently diverge from the recovered dictionaries.
        if !senders.starts_with(&self.snapshot.senders)
            || !receivers.starts_with(&self.snapshot.receivers)
        {
            return Err(PipelineError::Mismatch(
                "checkpoint dictionary does not extend artifact dictionary".into(),
            ));
        }
        let sealed_entries = abacus_tensor::csf::decode_csf(&sealed);
        let index = AccountIndex::build(&sealed_entries, &[]);
        let mut sender_codes = BTreeMap::new();
        for (i, s) in senders.iter().enumerate() {
            sender_codes.insert(s.clone(), i as u32);
        }
        let mut receiver_codes = BTreeMap::new();
        for (i, r) in receivers.iter().enumerate() {
            receiver_codes.insert(r.clone(), i as u32);
        }
        let mut ledger = Ledger::new();
        for i in 0..committed_records as u64 {
            // Ledger amounts are not checkpointed (only presence matters for
            // the Create-CAS identity contract); re-create with a nonzero
            // placeholder so `Expected::Absent` holds exactly for fresh ids.
            let _ = ledger.ledger_op(i, LedgerOp::Create(1));
        }
        // Preserve the derived temporal state built from the core artifacts
        // by `from_loaded`; the compacted journal prefix is re-appended by
        // `restore_index_state` afterwards (mirrors `restore_duplicate_keys`),
        // so the directory rebuilds exactly and locator ids stay stable.
        let temporal = std::mem::take(&mut self.snapshot.temporal);
        self.snapshot = PipelineSnapshot {
            version,
            sealed,
            frontier: Vec::new(),
            index,
            committed_records,
            t_min,
            t_max,
            senders,
            receivers,
            temporal,
        };
        self.ledger = ledger;
        self.next_tx = committed_records as u64;
        self.sender_codes = sender_codes;
        self.receiver_codes = receiver_codes;
        self.event_ids = HashMap::new();
        Ok(())
    }

    /// Restore duplicate-detection keys for journal records already compacted
    /// into a checkpoint (recovery only; called after [`Self::install_checkpoint`]).
    ///
    /// `events` MUST be exactly the journal prefix `[0, journal_len)` the
    /// checkpoint consumed. Codes resolve against the checkpoint dicts, which
    /// preserve original code-assignment order, so keys match the pre-crash
    /// map. Only membership in `event_ids` drives duplicate detection (the
    /// stored id is never read back), so the journal index is used as the id.
    ///
    /// # Errors
    /// `Mismatch` if a prefix event references an account missing from the
    /// checkpoint dicts (journal/dict divergence → fail closed).
    pub fn restore_duplicate_keys(
        &mut self,
        events: &[NormalizedEvent],
    ) -> Result<(), PipelineError> {
        for (i, ev) in events.iter().enumerate() {
            let s = self
                .sender_codes
                .get(&ev.sender)
                .copied()
                .ok_or_else(|| {
                    PipelineError::Mismatch("checkpoint prefix sender missing from dict".into())
                })?;
            let r = self
                .receiver_codes
                .get(&ev.receiver)
                .copied()
                .ok_or_else(|| {
                    PipelineError::Mismatch("checkpoint prefix receiver missing from dict".into())
                })?;
            self.event_ids.insert((ev.t, s, r, ev.amount_micro), i as u64);
        }
        Ok(())
    }

    /// Restore the derived temporal index entries for journal records already
    /// compacted into a checkpoint (recovery only; call AFTER
    /// [`Self::install_checkpoint`], alongside [`Self::restore_duplicate_keys`]).
    ///
    /// `events` MUST be exactly the journal prefix `[0, journal_len)` the
    /// checkpoint consumed. Locators continue after the core-artifact
    /// directory (`directory.len()` == next event id by the identity
    /// contract), so the ids rebuilt here match the pre-crash assignment
    /// exactly. Codes resolve against the checkpoint dictionaries, which
    /// preserve original code-assignment order (`prec_dict`).
    ///
    /// # Errors
    /// `Mismatch` if a prefix event references an account missing from the
    /// checkpoint dicts (journal/dict divergence → fail closed), or
    /// `Invalid` on locator-capacity exhaustion (never a silent wrap).
    pub fn restore_index_state(&mut self, events: &[NormalizedEvent]) -> Result<(), PipelineError> {
        for ev in events {
            let s = self
                .sender_codes
                .get(&ev.sender)
                .copied()
                .ok_or_else(|| {
                    PipelineError::Mismatch("checkpoint prefix sender missing from dict".into())
                })?;
            let r = self
                .receiver_codes
                .get(&ev.receiver)
                .copied()
                .ok_or_else(|| {
                    PipelineError::Mismatch("checkpoint prefix receiver missing from dict".into())
                })?;
            self.snapshot.temporal.insert_event(ev.t, s, r)?;
        }
        Ok(())
    }

    /// Explicit compaction (row-21 full rebuild). NOT on the per-event path.
    ///
    /// The derived temporal indexes (forward/backward/global/directory) are
    /// deliberately untouched: seal only rewrites the canonical tensor's
    /// storage layout, never the committed event set, and locators reference
    /// event ids — not CSF positions — so they stay valid across the rebuild.
    ///
    /// # Errors
    /// `TensorError::Range` on unrepresentable merge sums.
    pub fn seal(&mut self) -> Result<bool, PipelineError> {
        if self.snapshot.frontier.is_empty() {
            return Ok(false);
        }
        let sealed_t = abacus_tensor::csf::decode::decode_to_tensor(&self.snapshot.sealed);
        let frontier_t = abacus_tensor::tensor::Tensor::try_from_entries(self.snapshot.frontier.clone())?;
        let merged = sealed_t.try_add(&frontier_t)?;
        let new_csf = abacus_tensor::csf::compress::compress(merged.entries());
        if !new_csf.check_invariants() {
            return Err(PipelineError::Mismatch("seal produced invalid CSF".into()));
        }
        self.snapshot.sealed = new_csf;
        self.snapshot.frontier.clear();
        let sealed_entries = abacus_tensor::csf::decode_csf(&self.snapshot.sealed);
        self.snapshot.index = AccountIndex::build(&sealed_entries, &[]);
        self.seals += 1;
        Ok(true)
    }

    /// Live time-slice: sealed restriction + frontier filter, settled.
    pub fn live_slice(&self, t0: u64, t1: u64) -> Vec<TxEntry> {
        let mut entries = abacus_tensor::csf::query::time_slice(&self.snapshot.sealed, t0, t1).entries;
        for e in &self.snapshot.frontier {
            if e.t >= t0 && e.t < t1 {
                entries.push(*e);
            }
        }
        abacus_tensor::composites::settle(entries).unwrap_or_default()
    }
}
