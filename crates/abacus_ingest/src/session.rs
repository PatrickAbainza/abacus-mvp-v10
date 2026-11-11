//! Ingestion sessions: bounded chunk upload, idempotency and state transitions.
//!
//! Golden Source §13 fixes the protocol rules this module implements:
//!
//! - "Each upload session must bind to an explicit source namespace, schema
//!   version, feature-schema version, and configuration fingerprint."
//! - "Repeating a chunk with the same identity and identical contents is
//!   idempotent. Repeating a chunk identity with different contents is
//!   rejected."
//! - "The server acknowledges a chunk only after the documented validation and
//!   acceptance boundary has been reached."
//! - "The server enforces maximum chunk size, record count, total upload size,
//!   and resource limits."
//! - "Completed sessions cannot silently accept unrelated additional records."
//! - "Interrupted sessions have explicit resume or abort semantics."
//! - "Session resumption must preserve account mappings and source-row
//!   identity."
//! - "Conflicting configuration fingerprints must not be accepted into the same
//!   logical session without an explicit migration policy."
//!
//! ## Chunk idempotency is NOT event deduplication
//!
//! §5.1: "Do not rely on chunk idempotency as a substitute for event-level
//! deduplication." Both exist here, separately: a chunk digest keyed by
//! sequence number makes a *retransmission* free, while
//! [`crate::identity::IdentityLedger`] makes *event application* idempotent
//! even when the same event arrives inside a different chunk.
//!
//! ## Atomicity
//!
//! Canonical Contract §11: "A failed segment must not leave the substrate in an
//! undocumented partially applied state." So [`IngestionSession::commit`]
//! validates **every** buffered record before touching the substrate, and only
//! then applies. Validation is a pure pass over buffered records, which is what
//! makes [`IngestionSession::validate_only`] meaningful: it runs the identical
//! checks and changes nothing.

use std::collections::HashMap;

use crate::config::SourceConfig;
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::hash::sha256;
use crate::identity::{IdentityLedger, IdentityOutcome};
use crate::namespace::EventNamespace;
use crate::record::CanonicalRecord;
use crate::wire::segment::Segment;

/// Session lifecycle states.
///
/// Transitions are one-directional except for the documented resume path, and
/// every transition is explicit — there is no implicit state change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Created; configuration bound; no chunks accepted yet.
    Open,
    /// At least one chunk accepted; still accepting chunks.
    Uploading,
    /// Finalized by the client; no further chunks accepted.
    Sealed,
    /// Validated events applied to the substrate.
    Committed,
    /// Explicitly cancelled or expired; rejects further work.
    Cancelled,
}

impl SessionState {
    /// Stable spelling for the API and reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Uploading => "uploading",
            Self::Sealed => "sealed",
            Self::Committed => "committed",
            Self::Cancelled => "cancelled",
        }
    }

    /// True iff the session will accept another chunk.
    #[must_use]
    pub const fn accepts_chunks(self) -> bool {
        matches!(self, Self::Open | Self::Uploading)
    }
}

/// What happened to one uploaded chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStatus {
    /// Accepted: structurally valid and recorded.
    Accepted,
    /// Already processed with identical bytes (idempotent retry, §13).
    AlreadyProcessed,
}

/// One accepted chunk's bookkeeping.
#[derive(Debug, Clone)]
struct ChunkRecord {
    /// SHA-256 of the exact chunk bytes as delivered.
    digest: [u8; 32],
    /// Records decoded from this chunk.
    record_count: u64,
}

/// Per-chunk outcome, used by both the upload and validate paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkOutcome {
    /// Chunk sequence number.
    pub sequence: u64,
    /// Chunk status.
    pub status: ChunkStatus,
    /// Records decoded and structurally validated from this chunk.
    pub record_count: u64,
    /// Events classified as new by the identity ledger.
    pub new_events: u64,
    /// Events classified as duplicate deliveries.
    pub duplicate_events: u64,
    /// Events whose identity conflicted with an already-seen identity (same
    /// identity, different canonical content — §5.1). Reported by the durable
    /// journal so the conflict count survives a restart (§10 reconciliation).
    pub conflicting_events: u64,
}

/// One applied event, handed to the durable journal sink during commit
/// (persistence, Amendment A2).
///
/// Carries everything recovery needs to streaming_ingestion_demo the event through the
/// substrate's own CAS path and to rebuild the account map deterministically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppliedEvent {
    /// Substrate event id used for the CAS.
    pub substrate_event_id: u64,
    /// Substrate `t` coordinate (epoch seconds).
    pub t: u64,
    /// Substrate `s` dictionary code.
    pub s: u32,
    /// Substrate `r` dictionary code.
    pub r: u32,
    /// Exact micro-unit value written to the substrate.
    pub value_micro: i64,
    /// Wire sender identifier (rebuilds the account map on recovery).
    pub sender_wire_id: u64,
    /// Wire receiver identifier (rebuilds the account map on recovery).
    pub receiver_wire_id: u64,
}

/// Server-enforced resource limits (Golden Source §13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionLimits {
    /// Maximum accepted chunk payload in bytes.
    pub max_chunk_bytes: u64,
    /// Maximum accepted records per chunk.
    pub max_chunk_records: u64,
    /// Maximum accepted total payload across the session.
    pub max_session_bytes: u64,
    /// Maximum accepted total records across the session.
    pub max_session_records: u64,
    /// Maximum accepted chunks in one session.
    pub max_chunks: u64,
}

impl Default for SessionLimits {
    /// Bounded defaults: 8 MiB chunks, 100k records per chunk, 1 GiB and 10M
    /// records per session, 100k chunks. §13 requires explicit limits; these
    /// are the defaults an operator can tighten.
    fn default() -> Self {
        Self {
            max_chunk_bytes: crate::config::DEFAULT_MAX_CHUNK_BYTES,
            max_chunk_records: 100_000,
            max_session_bytes: 1 << 30,
            max_session_records: 10_000_000,
            max_chunks: 100_000,
        }
    }
}

/// One ingestion session.
///
/// Binds an event namespace and a configuration fingerprint, buffers validated
/// chunks with chunk-level idempotency, and runs the namespace-aware event
/// identity ledger. Substrate application is a separate explicit step
/// ([`Self::commit`]) that goes through [`crate::apply::SubstrateApplier`].
#[derive(Debug)]
pub struct IngestionSession {
    /// Opaque session identifier.
    id: String,
    /// The bound configuration (fingerprint is derived from it).
    config: SourceConfig,
    /// The single namespace for this session (§2: exactly one per segment, and
    /// a session binds one logical import/stream).
    namespace: EventNamespace,
    /// Accepted chunks keyed by sequence number.
    chunks: HashMap<u64, ChunkRecord>,
    /// All validated records buffered across chunks, in arrival order.
    records: Vec<CanonicalRecord>,
    /// Namespace-aware event identity ledger.
    identity: IdentityLedger,
    /// Current lifecycle state.
    state: SessionState,
    /// Server-enforced limits.
    limits: SessionLimits,
    /// Total raw payload bytes accepted.
    total_bytes: u64,
    /// Total records DELIVERED across all accepted chunks, including those
    /// later classified as duplicate or conflicting. This is the denominator of
    /// the report's reconciliation invariant (Canonical Contract §10: "Every
    /// input record must be accounted for under the declared parsing policy").
    total_delivered: u64,
    /// Records rejected *within* an otherwise-valid chunk.
    ///
    /// Zero under the current policy, deliberately: a chunk that fails
    /// structural validation is rejected atomically in `submit_chunk`, so its
    /// records never enter the accounting (a corrupt header cannot be trusted
    /// to state how many records it held). The category exists so the four
    /// report outcomes stay disjoint: a conflicting identity is counted under
    /// `conflicting_events`, never also under `rejected_records`
    /// (Canonical Contract §11: the report "must distinguish rejected records,
    /// quarantined records, duplicate deliveries, and conflicting identities").
    rejected_records: u64,
    /// Structured errors accumulated during validation (no raw data).
    errors: Vec<IngestError>,
    /// Count of duplicate event deliveries seen.
    duplicate_events: u64,
    /// Count of conflicting event identities seen.
    conflicting_events: u64,
    /// Minimum accepted event time in milliseconds.
    min_event_time_ms: Option<i64>,
    /// Maximum accepted event time in milliseconds.
    max_event_time_ms: Option<i64>,
}

impl IngestionSession {
    /// Create a session bound to a namespace and configuration.
    ///
    /// # Errors
    /// Any [`SourceConfig::validate`] rejection, or
    /// [`ErrorCode::Identity`] for an all-zero namespace.
    pub fn create(
        id: impl Into<String>,
        config: SourceConfig,
        namespace: EventNamespace,
        limits: SessionLimits,
    ) -> IngestResult<Self> {
        config.validate()?;
        if namespace.is_zero() {
            return Err(IngestError::new(
                ErrorCode::Identity,
                "the all-zero namespace is not a valid identity domain",
            ));
        }
        Ok(Self {
            id: id.into(),
            config,
            namespace,
            chunks: HashMap::new(),
            records: Vec::new(),
            identity: IdentityLedger::new(),
            state: SessionState::Open,
            limits,
            total_bytes: 0,
            total_delivered: 0,
            rejected_records: 0,
            errors: Vec::new(),
            duplicate_events: 0,
            conflicting_events: 0,
            min_event_time_ms: None,
            max_event_time_ms: None,
        })
    }

    /// The session identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The bound configuration.
    #[must_use]
    pub fn config(&self) -> &SourceConfig {
        &self.config
    }

    /// The bound event namespace.
    #[must_use]
    pub fn namespace(&self) -> EventNamespace {
        self.namespace
    }

    /// The configuration fingerprint this session is bound to.
    #[must_use]
    pub fn fingerprint(&self) -> [u8; 32] {
        self.config.fingerprint()
    }

    /// The configuration fingerprint as lowercase hex.
    #[must_use]
    pub fn fingerprint_hex(&self) -> String {
        self.config.fingerprint_hex()
    }

    /// The current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> SessionState {
        self.state
    }

    /// Number of accepted chunks.
    #[must_use]
    pub fn chunk_count(&self) -> u64 {
        self.chunks.len() as u64
    }

    /// Number of validated records buffered.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.records.len() as u64
    }

    /// Total accepted payload bytes.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Total records delivered across all accepted chunks, including records
    /// later classified as duplicate or conflicting.
    ///
    /// This is the reconciliation denominator: Canonical Contract §10 requires
    /// `input = accepted + duplicate + conflicting + rejected`.
    pub const fn total_delivered(&self) -> u64 {
        self.total_delivered
    }

    /// Records rejected within an otherwise-valid chunk (see the field docs).
    pub const fn rejected_records(&self) -> u64 {
        self.rejected_records
    }

    /// Duplicate event deliveries observed across all chunks.
    #[must_use]
    pub const fn duplicate_events(&self) -> u64 {
        self.duplicate_events
    }

    /// Conflicting event identities observed across all chunks.
    #[must_use]
    pub const fn conflicting_events(&self) -> u64 {
        self.conflicting_events
    }

    /// Structured validation errors (never contain raw accounts or amounts).
    #[must_use]
    pub fn errors(&self) -> &[IngestError] {
        &self.errors
    }

    /// Minimum accepted event time in milliseconds, if any record was accepted.
    #[must_use]
    pub const fn min_event_time_ms(&self) -> Option<i64> {
        self.min_event_time_ms
    }

    /// Maximum accepted event time in milliseconds, if any record was accepted.
    #[must_use]
    pub const fn max_event_time_ms(&self) -> Option<i64> {
        self.max_event_time_ms
    }

    /// The buffered validated records (read-only).
    #[must_use]
    pub fn records(&self) -> &[CanonicalRecord] {
        &self.records
    }
}

impl IngestionSession {
    /// Submit one bounded binary chunk.
    ///
    /// A chunk is a complete ABACSEG1 segment: header + payload. Accepting it
    /// means every wire §6 structural check passed, the session/namespace and
    /// configuration binding matched, the limits held, and every record was
    /// classified by the event identity ledger. Substrate application is NOT
    /// performed here.
    ///
    /// Idempotency (§13): the same `sequence` with byte-identical content
    /// returns [`ChunkStatus::AlreadyProcessed`] and changes nothing. The same
    /// `sequence` with different bytes is rejected.
    ///
    /// # Errors
    /// - [`ErrorCode::Session`] if the session no longer accepts chunks, the
    ///   sequence was already used with different bytes, or a limit is
    ///   exceeded.
    /// - Any [`crate::wire::Segment::decode`] rejection.
    /// - [`ErrorCode::Identity`] if the segment's namespace or configuration
    ///   fingerprint differs from the session's binding.
    /// - [`ErrorCode::Identity`] on a conflicting event identity (§5.1).
    pub fn submit_chunk(&mut self, sequence: u64, bytes: &[u8]) -> IngestResult<ChunkOutcome> {
        if !self.state.accepts_chunks() {
            return Err(IngestError::new(
                ErrorCode::Session,
                format!(
                    "session is {} and cannot accept chunk {sequence}",
                    self.state.as_str()
                ),
            ));
        }
        if self.chunks.len() as u64 >= self.limits.max_chunks {
            return Err(IngestError::new(
                ErrorCode::Session,
                format!("chunk limit {} reached", self.limits.max_chunks),
            ));
        }
        // §13: maximum chunk size.
        if bytes.len() as u64 > self.limits.max_chunk_bytes {
            return Err(IngestError::new(
                ErrorCode::Session,
                format!(
                    "chunk is {} bytes, exceeding the {}-byte limit",
                    bytes.len(),
                    self.limits.max_chunk_bytes
                ),
            ));
        }
        // §13: total upload size.
        let next_total = self
            .total_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| IngestError::new(ErrorCode::Session, "total upload size overflow"))?;
        if next_total > self.limits.max_session_bytes {
            return Err(IngestError::new(
                ErrorCode::Session,
                format!(
                    "total upload would reach {next_total} bytes, exceeding the {}-byte limit",
                    self.limits.max_session_bytes
                ),
            ));
        }

        // Chunk-level idempotency BEFORE any decoding work (§13).
        let digest = sha256(bytes);
        if let Some(prev) = self.chunks.get(&sequence) {
            if prev.digest == digest {
                return Ok(ChunkOutcome {
                    sequence,
                    status: ChunkStatus::AlreadyProcessed,
                    record_count: prev.record_count,
                    new_events: 0,
                    duplicate_events: 0,
                    conflicting_events: 0,
                });
            }
            return Err(IngestError::new(
                ErrorCode::Session,
                format!("chunk {sequence} was already accepted with different contents"),
            ));
        }

        // Full structural validation (wire §6).
        let segment = Segment::decode(bytes)?;
        if segment.namespace() != self.namespace {
            return Err(IngestError::new(
                ErrorCode::Identity,
                "segment namespace does not match the session's bound namespace",
            ));
        }
        if segment.header.config_fingerprint != self.fingerprint() {
            return Err(IngestError::new(
                ErrorCode::Identity,
                "segment configuration fingerprint does not match the session's binding",
            ));
        }
        // §13: maximum record count per chunk and per session.
        let n = segment.records.len() as u64;
        if n > self.limits.max_chunk_records {
            return Err(IngestError::new(
                ErrorCode::Session,
                format!(
                    "chunk has {n} records, exceeding the {}-record limit",
                    self.limits.max_chunk_records
                ),
            ));
        }
        let next_records = (self.records.len() as u64)
            .checked_add(n)
            .ok_or_else(|| IngestError::new(ErrorCode::Session, "record count overflow"))?;
        if next_records > self.limits.max_session_records {
            return Err(IngestError::new(
                ErrorCode::Session,
                format!(
                    "session would reach {next_records} records, exceeding the {}-record limit",
                    self.limits.max_session_records
                ),
            ));
        }

        // Event-level identity classification (§5.1).
        let (mut new_events, mut duplicate_events, mut conflicting_events) = (0u64, 0u64, 0u64);
        let mut staged: Vec<CanonicalRecord> = Vec::with_capacity(segment.records.len());
        for rec in &segment.records {
            let content = IdentityLedger::content_digest(
                rec.event_time_ms,
                rec.sender_id,
                rec.receiver_id,
                rec.features.log_magnitude,
                rec.features.relative_threshold_distance,
                rec.features.roundedness_flags,
            );
            match self.identity.check(rec.identity, content) {
                Ok((IdentityOutcome::New, _)) => {
                    new_events += 1;
                    staged.push(*rec);
                }
                Ok((IdentityOutcome::Duplicate, _)) => {
                    duplicate_events += 1;
                }
                Ok((IdentityOutcome::Conflict, _)) | Err(_) => {
                    // §5.1: conflicting content for an existing identity is
                    // rejected AND reported. Counted, surfaced, never applied.
                    self.conflicting_events += 1;
                    conflicting_events += 1;
                    self.errors.push(IngestError::new(
                        ErrorCode::Identity,
                        "conflicting content for an existing event identity",
                    ));
                }
            }
        }

        // Commit the chunk's bookkeeping only after every check passed.
        self.chunks.insert(
            sequence,
            ChunkRecord { digest, record_count: n },
        );
        self.total_bytes = next_total;
        self.duplicate_events += duplicate_events;
        self.total_delivered += n;
        for rec in &staged {
            self.min_event_time_ms = Some(
                self.min_event_time_ms
                    .map_or(rec.event_time_ms, |m| m.min(rec.event_time_ms)),
            );
            self.max_event_time_ms = Some(
                self.max_event_time_ms
                    .map_or(rec.event_time_ms, |m| m.max(rec.event_time_ms)),
            );
        }
        self.records.extend(staged);
        if self.state == SessionState::Open {
            self.state = SessionState::Uploading;
        }
        Ok(ChunkOutcome {
            sequence,
            status: ChunkStatus::Accepted,
            record_count: n,
            new_events,
            duplicate_events,
            conflicting_events,
        })
    }

    /// Chunk status lookup: accepted, or not present.
    ///
    /// # Errors
    /// [`ErrorCode::Session`] if the sequence was never submitted.
    pub fn chunk_status(&self, sequence: u64) -> IngestResult<ChunkStatus> {
        self.chunks
            .get(&sequence)
            .map(|_| ChunkStatus::Accepted)
            .ok_or_else(|| {
                IngestError::new(
                    ErrorCode::Session,
                    format!("chunk {sequence} was never submitted to this session"),
                )
            })
    }

    /// Seal the session: no further chunks accepted.
    ///
    /// # Errors
    /// [`ErrorCode::Session`] if the session is already sealed, committed or
    /// cancelled.
    pub fn seal(&mut self) -> IngestResult<()> {
        match self.state {
            SessionState::Open | SessionState::Uploading => {
                self.state = SessionState::Sealed;
                Ok(())
            }
            other => Err(IngestError::new(
                ErrorCode::Session,
                format!("cannot seal a session in state {}", other.as_str()),
            )),
        }
    }

    /// Cancel the session (task step 3: safe cancellation/cleanup).
    ///
    /// Buffered records are dropped, so nothing can later be applied from a
    /// cancelled session. Idempotent: cancelling twice is not an error.
    pub fn cancel(&mut self) {
        self.state = SessionState::Cancelled;
        self.records.clear();
        self.chunks.clear();
        self.total_bytes = 0;
        self.total_delivered = 0;
        self.rejected_records = 0;
        self.conflicting_events = 0;
        self.duplicate_events = 0;
        self.errors.clear();
        self.identity = IdentityLedger::new();
    }
}

impl IngestionSession {
    /// Validate everything buffered so far **without applying anything**.
    ///
    /// Task step 3 requires "Validate a complete segment or session without
    /// applying events, where supported by the architecture" — and here it is
    /// supported, because submission already performed the full structural
    /// validation and buffered only what passed. This pass therefore re-checks
    /// the properties that depend on the session as a whole rather than on one
    /// chunk: the timestamp domain, the declared account-ID width, the feature
    /// reserved bits, and identity reconciliation.
    ///
    /// It is side-effect free: no counter, state or buffered record changes.
    ///
    /// # Errors
    /// The first violation found, categorized per Golden Source §16.
    pub fn validate_only(&self) -> IngestResult<()> {
        let width = self.config.account_id_width;
        let mask = if width >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * u32::from(width))) - 1
        };
        for (i, rec) in self.records.iter().enumerate() {
            // Timestamp domain: the substrate coordinate is u64 seconds, so a
            // pre-epoch event can never be applied.
            if rec.event_time_ms < 0 {
                return Err(IngestError::new(
                    ErrorCode::Timestamp,
                    format!("buffered record {i} has a pre-epoch event time"),
                ));
            }
            // Account-ID width and reserved ID 0 (§6).
            if rec.sender_id == 0 || rec.receiver_id == 0 {
                return Err(IngestError::new(
                    ErrorCode::AccountId,
                    format!("buffered record {i} uses the reserved account id 0"),
                ));
            }
            if rec.sender_id > mask || rec.receiver_id > mask {
                return Err(IngestError::new(
                    ErrorCode::AccountId,
                    format!(
                        "buffered record {i} has an identifier outside the declared {width}-byte width"
                    ),
                ));
            }
            // Feature reserved bits (Canonical Contract §6.3).
            rec.features.validate()?;
        }
        // Reconciliation: every staged record is a NEW event, and the sum of
        // new events across chunks must equal the buffer length. Canonical
        // Contract §10: "Counts must reconcile."
        let expected = self.identity.accepted(&self.namespace);
        if expected != self.records.len() as u64 {
            return Err(IngestError::new(
                ErrorCode::SubstrateInvariant,
                format!(
                    "count reconciliation failed: identity ledger accepted {expected} but {} records are buffered",
                    self.records.len()
                ),
            ));
        }
        Ok(())
    }

    /// Commit: apply every buffered record through the existing substrate.
    ///
    /// Atomicity (Canonical Contract §11): all buffered records are validated
    /// first, so a failure aborts before any substrate mutation. Application
    /// then runs through [`crate::apply::SubstrateApplier`], which delegates to
    /// the substrate's own CAS path — no parallel update path.
    ///
    /// # Parameters
    /// - `applier` — the substrate application boundary to use.
    /// - `attest` — supplies the exact minor-unit amount per record for the
    ///   [`crate::apply::SubstrateValuePolicy::AttestedMinorUnits`] policy.
    ///   Receives the record index; must return `None` for records whose
    ///   amount is unavailable, which is then a rejection.
    ///
    /// # Errors
    /// - [`ErrorCode::Session`] if the session cannot be committed.
    /// - Any [`Self::validate_only`] rejection (before any mutation).
    /// - Any application rejection from the applier.
    pub fn commit<F>(
        &mut self,
        applier: &mut crate::apply::SubstrateApplier,
        attest: F,
    ) -> IngestResult<crate::report::CommitResult>
    where
        F: FnMut(usize, &CanonicalRecord) -> Option<i64>,
    {
        self.commit_with_sink(applier, attest, |_| Ok(()))
    }

    /// Commit with a durable-journal sink (Amendment A2).
    ///
    /// `on_applied` is invoked once per event **immediately after** that event
    /// was committed through the substrate's CAS path and **before** the next
    /// event is attempted, so a persistence layer can fsync each journal
    /// record before the commit is acknowledged. A sink failure aborts the
    /// commit fail-closed: the session is not marked committed, and any events
    /// already applied stay idempotently protected by the substrate CAS (a
    /// retry reports them as already applied).
    ///
    /// # Errors
    /// As [`Self::commit`], plus any error returned by `on_applied`.
    pub fn commit_with_sink<F, J>(
        &mut self,
        applier: &mut crate::apply::SubstrateApplier,
        mut attest: F,
        mut on_applied: J,
    ) -> IngestResult<crate::report::CommitResult>
    where
        F: FnMut(usize, &CanonicalRecord) -> Option<i64>,
        J: FnMut(&AppliedEvent) -> IngestResult<()>,
    {
        if !matches!(self.state, SessionState::Sealed | SessionState::Uploading | SessionState::Open)
        {
            return Err(IngestError::new(
                ErrorCode::Session,
                format!("cannot commit a session in state {}", self.state.as_str()),
            ));
        }
        // Validate EVERYTHING before touching the substrate (§11 atomicity).
        self.validate_only()?;

        // §11 pre-flight: resolve the value policy and EVERY attested amount
        // before the first substrate mutation. Without this, an amount that is
        // unavailable for a later record would strand an already-applied prefix
        // — a partially applied session, which §11 forbids.
        //
        // Amendment A1 clearing obligation: the attested-amount buffer is an
        // RAII guard, so it is zeroed on EVERY reachable exit — a pre-flight
        // rejection (missing/negative/over-ceiling/feature-mismatch, below), a
        // mid-stream substrate rejection, a durable-journal sink failure, or a
        // panic unwind — not only the success path. It is declared before the
        // value policy is resolved so even a `major_scale()?` or feature-derive
        // `?` clears whatever has already been staged. Clearing is logical, not
        // cryptographic (§16.3); the substrate's own micro-unit entry is the
        // only surviving copy by design.
        let mut amounts = crate::secret::SecretVec::with_capacity(self.records.len());
        match applier.policy() {
            crate::apply::SubstrateValuePolicy::ClientDerivedOnly => {
                return Err(IngestError::new(
                    ErrorCode::SpecificationGap,
                    "a feature-only substrate mode is not defined by the governing \
                     specification, so the exact i64 micro-unit value cannot be populated",
                ));
            }
            crate::apply::SubstrateValuePolicy::AttestedMinorUnits => {
                // Amendment A1 pre-flight checks, possible precisely because
                // the exact amount now crosses the boundary:
                //   1. Range: nonnegative and within the configured supported
                //      ceiling (Golden Source §8/§19 — reject, never clamp).
                //   2. Consistency: the frozen features must be exactly
                //      reproducible from the attested amount under this
                //      session's threshold and currency precision (Golden
                //      Source §14: the backend now CAN validate that
                //      client-derived features match the amount).
                let major_scale = self.config.major_scale()?;
                for (i, rec) in self.records.iter().enumerate() {
                    let a = match attest(i, rec) {
                        Some(a) => a,
                        None => {
                            return Err(IngestError::new(
                                ErrorCode::SpecificationGap,
                                format!(
                                    "no attested amount for buffered record {i}; the policy \
                                     requires one and no substitute is defined"
                                ),
                            ));
                        }
                    };
                    if a < 0 {
                        return Err(IngestError::new(
                            ErrorCode::AmountOrCurrency,
                            format!("attested amount for buffered record {i} is negative"),
                        ));
                    }
                    let a_u = a as u64;
                    if a_u > self.config.max_amount_minor_units {
                        return Err(IngestError::new(
                            ErrorCode::AmountOrCurrency,
                            format!(
                                "attested amount for buffered record {i} exceeds the \
                                 configured supported amount ceiling"
                            ),
                        ));
                    }
                    // Value representability: the substrate value is the attested
                    // amount scaled by MINOR_SCALE (10^6) into exact i64
                    // micro-units (P2). An amount within the configured ceiling
                    // can still overflow that product, and `apply` rejects it
                    // with FeatureOverflow. Checking it here — before the first
                    // substrate mutation — keeps the "validate everything, then
                    // apply" contract genuinely total for the value path, so a
                    // representability failure aborts with zero mutations rather
                    // than stranding an applied prefix (§11). This mirrors the
                    // exact check `SubstrateApplier::apply` performs; it adds no
                    // new rejection category, it only moves this one earlier.
                    if a.checked_mul(abacus_tensor::tensor::value::MINOR_SCALE).is_none() {
                        return Err(IngestError::new(
                            ErrorCode::FeatureOverflow,
                            format!(
                                "attested amount for buffered record {i} overflows the \
                                 substrate's micro-unit value set (V is exact i64 at scale 10^6)"
                            ),
                        ));
                    }
                    let expected = crate::features::FeaturePayload::derive(
                        a_u,
                        self.config.threshold_minor_units,
                        major_scale,
                    )?;
                    if expected != rec.features {
                        return Err(IngestError::new(
                            ErrorCode::AmountOrCurrency,
                            format!(
                                "attested amount for buffered record {i} does not reproduce \
                                 the record's frozen feature payload"
                            ),
                        ));
                    }
                    amounts.push(a);
                }
            }
        };

        let mut applied = 0u64;
        let mut already = 0u64;
        for (i, rec) in self.records.iter().enumerate() {
            // The namespace-local substrate id, reproduced from the identity
            // ledger's assignment order so a streaming_ingestion_demo assigns the same ids.
            let substrate_id = i as u64 + 1;
            match applier.apply(rec, substrate_id, Some(amounts[i]))? {
                crate::apply::ApplyOutcome::Applied {
                    substrate_event_id,
                    t,
                    s,
                    r,
                    value_micro,
                } => {
                    on_applied(&AppliedEvent {
                        substrate_event_id,
                        t,
                        s,
                        r,
                        value_micro,
                        sender_wire_id: rec.sender_id,
                        receiver_wire_id: rec.receiver_id,
                    })?;
                    applied += 1;
                }
                crate::apply::ApplyOutcome::AlreadyApplied => already += 1,
            }
        }
        // Amendment A1 clearing obligation: the transient attested-amount
        // vector is zeroed as soon as the transformation into substrate
        // values is complete. The canonical substrate entry (and the journal
        // record that durably mirrors it) is the only surviving copy by
        // design; no session, report or API surface retains an amount. The
        // explicit clear here covers the success path; the `SecretVec` guard's
        // `Drop` covers every other exit (pre-flight rejection, substrate
        // rejection, journal-sink failure, panic) so no path can leak it.
        amounts.zeroize();
        self.state = SessionState::Committed;
        Ok(crate::report::CommitResult { applied, already_applied: already })
    }
}

impl IngestionSession {
    // ---------------------------------------------------------------------------
    // Journal recovery (Amendment A2).
    //
    // These methods rebuild session state from a validated durable journal.
    // They never accept external client input; a failure here means the
    // journal and the in-memory model have diverged, which must fail closed
    // and never be silently repaired.
    // ---------------------------------------------------------------------------

    /// Restore one accepted chunk's bookkeeping. The journal is the
    /// authoritative record of what was accepted, so no re-decoding happens.
    ///
    /// # Errors
    /// [`ErrorCode::SubstrateInvariant`] if the sequence was already restored
    /// (a well-formed journal cannot contain this), or on counter overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_chunk(
        &mut self,
        sequence: u64,
        digest: [u8; 32],
        record_count: u64,
        payload_bytes: u64,
        new_events: u64,
        duplicate_events: u64,
    ) -> IngestResult<()> {
        if self.chunks.contains_key(&sequence) {
            return Err(IngestError::new(
                ErrorCode::SubstrateInvariant,
                format!("journal restores chunk sequence {sequence} twice"),
            ));
        }
        self.chunks.insert(sequence, ChunkRecord { digest, record_count });
        self.total_bytes = self.total_bytes.checked_add(payload_bytes).ok_or_else(|| {
            IngestError::new(ErrorCode::SubstrateInvariant, "journal byte counter overflow")
        })?;
        self.total_delivered = self.total_delivered.checked_add(record_count).ok_or_else(|| {
            IngestError::new(ErrorCode::SubstrateInvariant, "journal record counter overflow")
        })?;
        self.duplicate_events = self.duplicate_events.checked_add(duplicate_events).ok_or_else(|| {
            IngestError::new(ErrorCode::SubstrateInvariant, "journal duplicate counter overflow")
        })?;
        let _ = new_events; // recomputed by restore_record via the identity ledger
        if self.state == SessionState::Open {
            self.state = SessionState::Uploading;
        }
        Ok(())
    }

    /// Restore one accepted record together with its identity-ledger entry,
    /// streaming_ingestion_demoing the same classification the live path performed.
    ///
    /// # Errors
    /// [`ErrorCode::SubstrateInvariant`] if the journal streaming_ingestion_demo does not
    /// classify the record as NEW (identity divergence).
    pub fn restore_record(&mut self, rec: CanonicalRecord) -> IngestResult<()> {
        let content = IdentityLedger::content_digest(
            rec.event_time_ms,
            rec.sender_id,
            rec.receiver_id,
            rec.features.log_magnitude,
            rec.features.relative_threshold_distance,
            rec.features.roundedness_flags,
        );
        let (outcome, _substrate_id) = self.identity.check(rec.identity, content)?;
        if outcome != IdentityOutcome::New {
            return Err(IngestError::new(
                ErrorCode::SubstrateInvariant,
                "journal streaming_ingestion_demo did not classify a restored record as new; the identity \
                 ledger has diverged from the journal",
            ));
        }
        self.min_event_time_ms = Some(
            self.min_event_time_ms
                .map_or(rec.event_time_ms, |m| m.min(rec.event_time_ms)),
        );
        self.max_event_time_ms = Some(
            self.max_event_time_ms
                .map_or(rec.event_time_ms, |m| m.max(rec.event_time_ms)),
        );
        self.records.push(rec);
        Ok(())
    }

    /// Restore one conflicting-identity observation, reproducing the live
    /// path's counter and error entry so the report is identical across a
    /// restart.
    pub fn restore_conflict(&mut self) {
        self.conflicting_events += 1;
        self.errors.push(IngestError::new(
            ErrorCode::Identity,
            "conflicting content for an existing event identity",
        ));
    }

    /// Restore a lifecycle state reached before the crash. Cancellation
    /// reproduces the live cleanup semantics (buffers dropped).
    pub fn restore_state(&mut self, state: SessionState) {
        if state == SessionState::Cancelled {
            self.cancel();
        } else {
            self.state = state;
        }
    }
}
