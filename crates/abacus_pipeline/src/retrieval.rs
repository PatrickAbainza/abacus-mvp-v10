//! Phase-3 retrieval: incremental bidirectional indexing (CSR/CSC-style) and
//! bounded event-relative retrieval over the canonical Abacus tensor.
//!
//! Golden-source grounding (no invented tensor behavior):
//! - Derived-state rule: `golden_source_hiearchical_csf_tensor.md` §§1-4, 8,
//!   10 — the forward index, backward index, global time path and locator
//!   directory are DERIVED access paths over the canonical sources (sealed
//!   CSF + frontier + core records). They are never a second canonical store,
//!   never written back into the CSF, and encode no typology assumptions
//!   (hierarchical invariant: no second canonical index).
//! - Ingestion: derivation rows 18-23 / P7-P10 — index maintenance rides the
//!   existing fail-before-mutate commit path; seal (row 21) is event-set
//!   preserving and never touches the temporal indexes; out-of-order `t` is
//!   SUPPORT (P10: watermark stored, never consulted).
//! - Intervals: row 16 (§B.2) + P1 `prec_3` + P6 — half-open `[t0, t1)` and
//!   deterministic `(t, locator)` ordering, reused not reinvented.
//! - Locator: event id = next `tx_id` (`golden_source_samld_binary_addendum.md`
//!   §§S4/S6 positional bijection), narrowed to `u32` with explicit
//!   reject-on-exhaustion (never wraps). Resolution goes through the
//!   append-only locator directory because CSF nnz positions shift at seal and
//!   duplicate coordinates merge (task spec §3 indirection requirement).
//! - Query contract: `your_task_phase_3.md` §§4-8 (2,500 cap, half-open
//!   asymmetric relative intervals, truncation semantics, checked temporal
//!   arithmetic, global retrieval uncapped).
//!
//! Representation choice (task spec §4): static CSR/CSC offset arrays cannot
//! absorb a per-event insert without shifting the shared column arena, so each
//! entity's list is a growable row slice ordered by `(t, locator)` with a
//! binary-search insert and an O(1) append fast path. This preserves the CSR
//! (sender-rows) / CSC (receiver-columns) access patterns while matching the
//! append-heavy incremental model of `Pipeline::apply`.

use crate::error::PipelineError;

/// Hard cap on locators returned by any single directional lookup
/// (task spec §7: applied inside the retrieval implementation, not only at
/// API call sites; does not apply to global temporal retrieval).
pub const MAX_EDGES_PER_LOOKUP: usize = 2_500;

/// Default relative window: seven days in seconds (task spec §6 — a default,
/// not a hard-coded semantic rule of the tensor or index).
pub const DEFAULT_DELTA_SECS: u64 = 7 * 24 * 60 * 60;

/// Inline buffer capacity of [`LocatorBuffer`] before bounded spillover
/// (task spec §6: small results stay allocation-free; no result object embeds
/// storage for 2,500 locators).
pub const INLINE_LOCATOR_CAP: usize = 32;

/// Compact deterministic event locator: the existing stable event ordinal
/// (event id = next `tx_id` = committed record count) narrowed to 32 bits.
///
/// Stability: assigned once at commit, never renumbered, valid across
/// incremental updates, seal, journal streaming_ingestion_demo and checkpoint recovery.
/// Capacity: `0..=u32::MAX` (4,294,967,296 events); exhaustion is rejected
/// fail-before-mutate by `Pipeline::check_apply` — never wrapped silently.
/// Resolution: `TemporalIndexes::resolve` (append-only directory; no process
/// addresses, no heap-allocated pointer objects, no payload duplication).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EventLocator(pub u32);

/// One index entry: timestamp + locator + opposite-end entity code.
/// 16 bytes packed (`repr(C)`: `u64` + two `u32` fields). Carries no amount
/// and no payload — the authoritative event is reached through the locator
/// directory (task spec §3: never store complete event payloads in an index).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TemporalRef {
    /// Event time `t` (epoch seconds, canonical precision — no conversion).
    pub t: u64,
    /// Locator of the authoritative event.
    pub locator: EventLocator,
    /// Opposite endpoint: receiver code in the forward index, sender code in
    /// the backward index (role identity is always preserved).
    pub peer: u32,
}

/// Locator result buffer: 32 inline entries, bounded spillover to a single
/// `Vec` allocated with exactly the effective limit as capacity. Adapted from
/// the task-spec `SmallVec<[EventLocator; 32]>` template because the workspace
/// is std-only by golden source (§0 H1: no third-party libraries).
#[derive(Debug, Clone)]
pub struct LocatorBuffer {
    /// Effective limit this buffer may hold (spill capacity hint).
    limit: usize,
    storage: BufferStorage,
}

#[derive(Debug, Clone)]
enum BufferStorage {
    Inline {
        data: [EventLocator; INLINE_LOCATOR_CAP],
        len: usize,
    },
    Spilled(Vec<EventLocator>),
}

impl LocatorBuffer {
    /// Empty buffer that will never hold more than `limit` locators.
    #[must_use]
    pub fn with_limit(limit: usize) -> Self {
        Self {
            limit,
            storage: BufferStorage::Inline {
                data: [EventLocator(0); INLINE_LOCATOR_CAP],
                len: 0,
            },
        }
    }

    /// Append one locator. Callers never push beyond the effective limit; a
    /// violation is a programming error and trips `debug_assert` (the spill
    /// allocation is sized exactly to the limit, so the retained buffer can
    /// never exceed it — memory contract of task spec §7).
    pub fn push(&mut self, loc: EventLocator) {
        debug_assert!(self.len() < self.limit, "lookup pushed beyond effective limit");
        match &mut self.storage {
            BufferStorage::Inline { data, len } => {
                if *len < INLINE_LOCATOR_CAP {
                    data[*len] = loc;
                    *len += 1;
                } else {
                    let mut v = Vec::with_capacity(self.limit);
                    v.extend_from_slice(&data[..INLINE_LOCATOR_CAP]);
                    v.push(loc);
                    self.storage = BufferStorage::Spilled(v);
                }
            }
            BufferStorage::Spilled(v) => v.push(loc),
        }
    }

    /// Number of stored locators.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            BufferStorage::Inline { len, .. } => *len,
            BufferStorage::Spilled(v) => v.len(),
        }
    }

    /// True when no locators are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the stored locators in insertion order.
    #[must_use]
    pub fn as_slice(&self) -> &[EventLocator] {
        match &self.storage {
            BufferStorage::Inline { data, len } => &data[..*len],
            BufferStorage::Spilled(v) => v.as_slice(),
        }
    }

    /// Iterate stored locators in insertion order.
    pub fn iter(&self) -> std::slice::Iter<'_, EventLocator> {
        self.as_slice().iter()
    }

    /// Effective limit this buffer was created with.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }
}

impl Default for LocatorBuffer {
    fn default() -> Self {
        Self::with_limit(MAX_EDGES_PER_LOOKUP)
    }
}

impl<'a> IntoIterator for &'a LocatorBuffer {
    type Item = &'a EventLocator;
    type IntoIter = std::slice::Iter<'a, EventLocator>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// Event-relative query (task spec §6 template, adapted to repository
/// conventions): entity = verbatim account identifier (the dictionaries are
/// role-separated, so the account *name* — not a numeric code — is the only
/// representation that addresses both roles), `anchor_time`/`delta` are
/// canonical `u64` epoch seconds (the repository's timestamp convention; no
/// sub-second precision exists, so sub-second durations are unsupported by
/// type rather than converted lossily), `max_edges` is clamped to
/// [`MAX_EDGES_PER_LOOKUP`] inside every retrieval implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiHopQuery {
    /// Account identifier (verbatim dictionary string).
    pub entity_id: String,
    /// Anchor event time (epoch seconds).
    pub anchor_time: u64,
    /// Relative window length in seconds (`>= 0`; `0` yields an empty
    /// interval). All `u64` values are valid — no invalid duration exists at
    /// this type level.
    pub delta: u64,
    /// Requested maximum locators (clamped to the hard cap internally).
    pub max_edges: usize,
}

impl MultiHopQuery {
    /// Construct an explicit query.
    #[must_use]
    pub fn new(entity_id: impl Into<String>, anchor_time: u64, delta: u64, max_edges: usize) -> Self {
        Self {
            entity_id: entity_id.into(),
            anchor_time,
            delta,
            max_edges,
        }
    }

    /// Construct a query with the seven-day default delta.
    #[must_use]
    pub fn with_default_delta(entity_id: impl Into<String>, anchor_time: u64, max_edges: usize) -> Self {
        Self::new(entity_id, anchor_time, DEFAULT_DELTA_SECS, max_edges)
    }

    /// Effective limit after the hard cap is applied (task spec §6:
    /// `min(query.max_edges, MAX_EDGES_PER_LOOKUP)`).
    #[must_use]
    pub fn effective_limit(&self) -> usize {
        self.max_edges.min(MAX_EDGES_PER_LOOKUP)
    }
}

/// Result of one bounded directional lookup (task spec §6 template with the
/// std-only [`LocatorBuffer`] standing in for `SmallVec`). An individual
/// lookup retains at most [`MAX_EDGES_PER_LOOKUP`] locator references; the
/// result buffer never grows beyond the effective limit (task spec §7).
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Matching locators, ≤ effective limit, deterministic order.
    pub locators: LocatorBuffer,
    /// True iff at least one qualifying event exists beyond the returned
    /// locators.
    pub is_truncated: bool,
    /// Exact qualifying count when computed correctly without estimation;
    /// `None` only when the exact total is intentionally not computed
    /// (zero-limit combined lookup, which early-exits on first match).
    pub total_matching_events: Option<usize>,
}

/// Locator resolved to its authoritative event fields (timestamp + both role
/// codes, in the same code spaces the CSF coordinate uses). Amounts and any
/// other payload stay in the canonical representation (`live_at` /
/// `snapshot().sealed` + frontier) and are never duplicated here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedEvent {
    /// The locator this event was resolved from.
    pub locator: EventLocator,
    /// Event time `t` (epoch seconds).
    pub t: u64,
    /// Sender dictionary code.
    pub sender: u32,
    /// Receiver dictionary code.
    pub receiver: u32,
}

/// One directory slot: the minimal fields required to resolve a locator
/// (task spec §3). 16 bytes (`repr(C)`); no amount, no payload.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DirectoryEntry {
    /// Event time `t` (epoch seconds).
    pub t: u64,
    /// Sender dictionary code.
    pub sender: u32,
    /// Receiver dictionary code.
    pub receiver: u32,
}

/// One directional index (CSR-style for the sender-oriented forward index,
/// CSC-style for the receiver-oriented backward index): a dense row table
/// indexed by the role's entity code, each row a contiguous slice ordered by
/// `(t, locator)`.
///
/// Incremental behavior: `insert` binary-searches the row and appends in
/// place when the new event sorts last (the common, time-ordered case: O(1));
/// an out-of-order event shifts only its own row (O(row) `memmove`), never a
/// shared arena — unlike static CSR/CSC offset arrays, which would require
/// rebuilding offsets or shifting every subsequent column entry.
#[derive(Debug, Clone, Default)]
pub struct DirectionalIndex {
    rows: Vec<Vec<TemporalRef>>,
}

impl DirectionalIndex {
    /// Index with `n` pre-allocated (possibly empty) rows.
    #[must_use]
    pub fn with_rows(n: usize) -> Self {
        Self {
            rows: vec![Vec::new(); n],
        }
    }

    /// Grow the row table so `code` addresses a row (append-only, mirrors the
    /// append-only dictionaries).
    fn ensure_row(&mut self, code: u32) {
        while self.rows.len() <= code as usize {
            self.rows.push(Vec::new());
        }
    }

    /// Insert one committed reference in `(t, locator)` order.
    pub fn insert(&mut self, code: u32, r: TemporalRef) {
        self.ensure_row(code);
        let row = &mut self.rows[code as usize];
        let pos = row.partition_point(|e| (e.t, e.locator.0) < (r.t, r.locator.0));
        if pos == row.len() {
            row.push(r);
        } else {
            row.insert(pos, r);
        }
    }

    /// Borrow the entity's ordered row (empty when the entity has no events
    /// in this role — unknown accounts read as empty, the repository's
    /// existing lookup convention).
    #[must_use]
    pub fn row(&self, code: u32) -> &[TemporalRef] {
        self.rows.get(code as usize).map_or(&[], Vec::as_slice)
    }

    /// Number of rows (≤ dictionary size; may trail the dictionary right
    /// after checkpoint install until the prefix is restored).
    #[must_use]
    pub fn rows_len(&self) -> usize {
        self.rows.len()
    }

    /// Total references stored (== number of committed events).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    /// Largest per-entity row (high-degree behavior metric).
    #[must_use]
    pub fn max_degree(&self) -> usize {
        self.rows.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Actual heap footprint: row-table capacity + every row's capacity ×
    /// 16-byte entries (capacity, alignment and segment overhead included —
    /// measured, not theoretical field sizes; task spec §4).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.rows.capacity() * std::mem::size_of::<Vec<TemporalRef>>()
            + self
                .rows
                .iter()
                .map(|r| r.capacity() * std::mem::size_of::<TemporalRef>())
                .sum::<usize>()
    }

    /// Iterate every stored reference in row order (consistency checks only).
    pub fn all_refs(&self) -> impl Iterator<Item = &TemporalRef> {
        self.rows.iter().flatten()
    }
}

/// Global temporal access path: two parallel contiguous arrays sorted by
/// `(t, locator)` — `times[i]` pairs with `locators[i]`. Structure-of-arrays
/// keeps the footprint at 12 bytes per event (a `Vec<(u64, u32)>` tuple would
/// pad to 16) while still supporting `partition_point` seeks at both interval
/// ends (task spec §2C: global retrieval must not enumerate per-entity rows).
#[derive(Debug, Clone, Default)]
pub struct GlobalTimeIndex {
    times: Vec<u64>,
    locators: Vec<u32>,
}

impl GlobalTimeIndex {
    /// Insert one committed event in `(t, locator)` order.
    pub fn insert(&mut self, t: u64, loc: EventLocator) {
        let lo = self.times.partition_point(|&x| x < t);
        let hi = self.times.partition_point(|&x| x <= t);
        let pos = lo + self.locators[lo..hi].partition_point(|&l| l < loc.0);
        self.times.insert(pos, t);
        self.locators.insert(pos, loc.0);
    }

    /// Locators of all events with `t0 <= t < t1`, ordered by `(t, locator)`
    /// ascending. An inverted or empty window (`t0 >= t1`) is empty (row-16
    /// convention: `time_slice` treats `t_start >= t_end` as empty).
    #[must_use]
    pub fn range_locators(&self, t0: u64, t1: u64) -> &[u32] {
        if t0 >= t1 {
            return &[];
        }
        let lo = self.times.partition_point(|&x| x < t0);
        let hi = self.times.partition_point(|&x| x < t1);
        &self.locators[lo..hi]
    }

    /// Number of indexed events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.locators.len()
    }

    /// True when no events are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.locators.is_empty()
    }

    /// Actual heap footprint (both arrays at their capacities).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.times.capacity() * std::mem::size_of::<u64>()
            + self.locators.capacity() * std::mem::size_of::<u32>()
    }
}

/// Measured footprint of the derived temporal structures (task spec §4/§10:
/// capacities and per-row segment metadata included — actual consumption, not
/// theoretical field sizes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexMemoryReport {
    /// Indexed events (== committed records; directory length).
    pub events: usize,
    /// Forward-index references.
    pub forward_entries: usize,
    /// Forward-index heap bytes (row table + all row capacities).
    pub forward_bytes: usize,
    /// Backward-index references.
    pub backward_entries: usize,
    /// Backward-index heap bytes (row table + all row capacities).
    pub backward_bytes: usize,
    /// Global time-path entries.
    pub global_entries: usize,
    /// Global time-path heap bytes (both parallel arrays).
    pub global_bytes: usize,
    /// Locator-directory heap bytes (capacity × 16).
    pub directory_bytes: usize,
    /// Sum of the four structures.
    pub total_bytes: usize,
    /// Highest forward degree (sender hub size).
    pub forward_max_degree: usize,
    /// Highest backward degree (receiver hub size).
    pub backward_max_degree: usize,
}

impl IndexMemoryReport {
    /// Average total bytes of additional indexed memory per event.
    #[must_use]
    pub fn bytes_per_event(&self) -> f64 {
        if self.events == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.events as f64
        }
    }
}

/// The complete derived temporal state published with every committed
/// snapshot: forward (sender) index, backward (receiver) index, global time
/// path and locator directory. Maintained incrementally in lockstep with the
/// canonical ingestion commits, so readers always see the temporal indexes and
/// the committed event set agreeing (indexes are built over the shared account
/// namespace `(t, s, r)`).
#[derive(Debug, Clone, Default)]
pub struct TemporalIndexes {
    /// Sender code → outgoing refs (CSR-style rows).
    pub forward: DirectionalIndex,
    /// Receiver code → incoming refs (CSC-style rows).
    pub backward: DirectionalIndex,
    /// Global `(t, locator)`-ordered time path.
    global: GlobalTimeIndex,
    /// Locator → `(t, s, r)` directory; index == locator (append-only).
    directory: Vec<DirectoryEntry>,
}

impl TemporalIndexes {
    /// Build from an ordered stream of `(t, sender, receiver)` records (the
    /// core artifact rows at load, or the journal prefix at recovery).
    /// Locators are assigned positionally, which is exactly the identity
    /// contract event id = committed record ordinal.
    ///
    /// # Errors
    /// `PipelineError::Invalid` when the record count exceeds `u32` locator
    /// capacity — explicit rejection, never a silent wrap (task spec §3).
    pub fn try_build<I: IntoIterator<Item = (u64, u32, u32)>>(
        records: I,
        sender_rows: usize,
        receiver_rows: usize,
    ) -> Result<Self, PipelineError> {
        let mut idx = Self {
            forward: DirectionalIndex::with_rows(sender_rows),
            backward: DirectionalIndex::with_rows(receiver_rows),
            global: GlobalTimeIndex::default(),
            directory: Vec::new(),
        };
        for (t, s, r) in records {
            idx.insert_event(t, s, r)?;
        }
        Ok(idx)
    }

    /// Insert one committed event into all four structures. The locator is
    /// derived from the directory length, which by the identity contract
    /// equals the event id; callers `debug_assert` that equality at commit.
    ///
    /// # Errors
    /// `PipelineError::Invalid` when the directory would exceed `u32::MAX + 1`
    /// entries (locator exhaustion — explicit, never a silent wrap).
    pub fn insert_event(&mut self, t: u64, s: u32, r: u32) -> Result<EventLocator, PipelineError> {
        let n = self.directory.len() as u64;
        let loc = u32::try_from(n)
            .map_err(|_| PipelineError::Invalid("locator capacity exhausted (u32)".into()))?;
        let locator = EventLocator(loc);
        self.forward.insert(
            s,
            TemporalRef {
                t,
                locator,
                peer: r,
            },
        );
        self.backward.insert(
            r,
            TemporalRef {
                t,
                locator,
                peer: s,
            },
        );
        self.global.insert(t, locator);
        self.directory.push(DirectoryEntry {
            t,
            sender: s,
            receiver: r,
        });
        Ok(locator)
    }

    /// Number of indexed events (== locator capacity used).
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.directory.len()
    }

    /// Resolve a locator to its authoritative event fields.
    /// Returns `None` for out-of-range locators (never fabricated, P3).
    #[must_use]
    pub fn resolve(&self, locator: EventLocator) -> Option<ResolvedEvent> {
        let e = *self.directory.get(locator.0 as usize)?;
        Some(ResolvedEvent {
            locator,
            t: e.t,
            sender: e.sender,
            receiver: e.receiver,
        })
    }

    /// Global time-range locator slice (uncapped; `(t, locator)` order).
    #[must_use]
    pub fn global_range(&self, t0: u64, t1: u64) -> &[u32] {
        self.global.range_locators(t0, t1)
    }

    /// Batched CSR row lookup: outgoing events for multiple shared-namespace
    /// account codes in `[t0, t1)`, each capped at `max_edges`.
    ///
    /// Returns per-code results in the same order as `codes`. Each inner
    /// `Vec<u32>` holds locators (event ids) in ascending `(t, locator)` order.
    ///
    /// # Efficiency
    /// Single pass over the account codes; each CSR row is binary-searched
    /// once for the time-window bounds. No per-account function-call
    /// overhead, and rows are accessed in code order for cache locality.
    #[must_use]
    pub fn batch_outgoing_in_range(
        &self,
        codes: &[u32],
        t0: u64,
        t1: u64,
        max_edges: usize,
    ) -> Vec<(u32, Vec<u32>)> {
        if t0 >= t1 {
            return codes.iter().map(|&c| (c, Vec::new())).collect();
        }
        codes
            .iter()
            .map(|&code| {
                let row = self.forward.row(code);
                let lo = row.partition_point(|e| e.t < t0);
                let hi = row.partition_point(|e| e.t < t1);
                let locs: Vec<u32> = row[lo..hi]
                    .iter()
                    .take(max_edges)
                    .map(|e| e.locator.0)
                    .collect();
                (code, locs)
            })
            .collect()
    }

    /// Batched CSC column lookup: incoming events for multiple shared-namespace
    /// account codes in `[t0, t1)`, each capped at `max_edges`.
    ///
    /// Returns per-code results in the same order as `codes`. Each inner
    /// `Vec<u32>` holds locators (event ids) in ascending `(t, locator)` order.
    #[must_use]
    pub fn batch_incoming_in_range(
        &self,
        codes: &[u32],
        t0: u64,
        t1: u64,
        max_edges: usize,
    ) -> Vec<(u32, Vec<u32>)> {
        if t0 >= t1 {
            return codes.iter().map(|&c| (c, Vec::new())).collect();
        }
        codes
            .iter()
            .map(|&code| {
                let row = self.backward.row(code);
                let lo = row.partition_point(|e| e.t < t0);
                let hi = row.partition_point(|e| e.t < t1);
                let locs: Vec<u32> = row[lo..hi]
                    .iter()
                    .take(max_edges)
                    .map(|e| e.locator.0)
                    .collect();
                (code, locs)
            })
            .collect()
    }

    /// Actual footprint of every derived structure (task spec §4/§10).
    #[must_use]
    pub fn memory_report(&self) -> IndexMemoryReport {
        let forward_bytes = self.forward.memory_bytes();
        let backward_bytes = self.backward.memory_bytes();
        let global_bytes = self.global.memory_bytes();
        let directory_bytes = self.directory.capacity() * std::mem::size_of::<DirectoryEntry>();
        IndexMemoryReport {
            events: self.directory.len(),
            forward_entries: self.forward.entry_count(),
            forward_bytes,
            backward_entries: self.backward.entry_count(),
            backward_bytes,
            global_entries: self.global.len(),
            global_bytes,
            directory_bytes,
            total_bytes: forward_bytes + backward_bytes + global_bytes + directory_bytes,
            forward_max_degree: self.forward.max_degree(),
            backward_max_degree: self.backward.max_degree(),
        }
    }

    /// Full structural cross-check: all four structures and the committed
    /// count agree, every stored locator is in range, and no row table
    /// outruns its dictionary. O(events); used by recovery and tests, never
    /// on the per-event path (an O(1) `debug_assert` covers commits).
    ///
    /// # Errors
    /// `PipelineError::Mismatch` naming the first divergence found.
    pub fn check(
        &self,
        committed_records: usize,
        sender_rows: usize,
        receiver_rows: usize,
    ) -> Result<(), PipelineError> {
        let n = self.directory.len();
        if n != committed_records {
            return Err(PipelineError::Mismatch(format!(
                "temporal directory {n} != committed records {committed_records}"
            )));
        }
        if self.global.len() != n {
            return Err(PipelineError::Mismatch(format!(
                "global time index {} != directory {n}",
                self.global.len()
            )));
        }
        let fwd = self.forward.entry_count();
        if fwd != n {
            return Err(PipelineError::Mismatch(format!(
                "forward index {fwd} != directory {n}"
            )));
        }
        let bwd = self.backward.entry_count();
        if bwd != n {
            return Err(PipelineError::Mismatch(format!(
                "backward index {bwd} != directory {n}"
            )));
        }
        if self.forward.rows_len() > sender_rows || self.backward.rows_len() > receiver_rows {
            return Err(PipelineError::Mismatch(
                "temporal row table outruns dictionary".into(),
            ));
        }
        if self.forward.all_refs().any(|e| e.locator.0 as usize >= n) {
            return Err(PipelineError::Mismatch(
                "forward reference has out-of-range locator".into(),
            ));
        }
        if self.backward.all_refs().any(|e| e.locator.0 as usize >= n) {
            return Err(PipelineError::Mismatch(
                "backward reference has out-of-range locator".into(),
            ));
        }
        Ok(())
    }
}

