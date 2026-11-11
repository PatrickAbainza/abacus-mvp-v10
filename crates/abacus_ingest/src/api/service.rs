//! The reusable ingestion service.
//!
//! Task step 3: "Keep transport handlers thin. They must delegate validation
//! and ingestion operations to reusable Rust services rather than implementing
//! business logic inside HTTP routes." So every rule lives here, and
//! `server.rs` only parses bytes and shapes JSON.
//!
//! The service owns the namespace registry, the sessions, and one
//! [`crate::apply::SubstrateApplier`] per session. It is safe to share behind a
//! mutex (matching the repository's serialized-writer model in `state.rs`): a
//! handler holds the lock only for the duration of one operation.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::apply::{SubstrateApplier, SubstrateValuePolicy};
use crate::config::{
    ACCOUNT_ID_WIDTHS, CurrencyCode, FEATURE_ALGORITHM_VERSION, FEATURE_SCHEMA_VERSION,
    FORMAT_VERSION, IdentityScheme, SourceConfig, SourceType,
};
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::hash::sha256;
use crate::namespace::{EventNamespace, NamespaceRegistry};
use crate::persist::{Journal, JournalRecord};
use crate::report::IngestionReport;
use crate::session::{
    ChunkOutcome, ChunkStatus, IngestionSession, SessionLimits, SessionState,
};

/// Capability advertisement (task step 3: "Retrieve supported schema versions,
/// source types, account-ID widths, and feature versions.").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Supported wire-format versions.
    pub wire_format_versions: Vec<u16>,
    /// Supported source types.
    pub source_types: Vec<SourceType>,
    /// Supported account-ID widths in bytes.
    pub account_id_widths: Vec<u8>,
    /// Supported feature-schema versions.
    pub feature_schema_versions: Vec<u16>,
    /// Supported feature-algorithm versions.
    pub feature_algorithm_versions: Vec<u16>,
    /// Supported identity schemes.
    pub identity_schemes: Vec<IdentityScheme>,
    /// Exact segment header length in bytes.
    pub header_length: usize,
    /// Whether validate-without-apply is supported.
    pub supports_validate_without_apply: bool,
}

impl Default for Capabilities {
    /// The capabilities this build actually implements — never advertised
    /// aspirationally.
    fn default() -> Self {
        Self {
            wire_format_versions: vec![FORMAT_VERSION],
            source_types: vec![SourceType::BrowserCsv, SourceType::SamldFixture, SourceType::Kafka],
            account_id_widths: ACCOUNT_ID_WIDTHS.to_vec(),
            feature_schema_versions: vec![FEATURE_SCHEMA_VERSION],
            feature_algorithm_versions: vec![FEATURE_ALGORITHM_VERSION],
            identity_schemes: vec![
                IdentityScheme::CsvRecordOrdinal,
                IdentityScheme::SamldRecordOrdinal,
                IdentityScheme::KafkaPartitionOffset,
            ],
            header_length: crate::config::HEADER_LEN,
            supports_validate_without_apply: true,
        }
    }
}

/// Request to create a session.
///
/// Deliberately explicit and small: the caller states the configuration and the
/// namespace. An absent namespace generates a fresh one (Golden Source §3.2: a
/// separate import gets a new namespace by default); supplying the previous
/// namespace is how a retry or resume reuses it (§3.2: "Never regenerate it
/// during a retry").
#[derive(Debug, Clone)]
pub struct CreateSessionRequest {
    /// Source type.
    pub source_type: SourceType,
    /// Identity scheme.
    pub identity_scheme: IdentityScheme,
    /// Account-ID width in bytes.
    pub account_id_width: u8,
    /// ISO 4217 currency code (three uppercase letters).
    pub currency_code: String,
    /// Minor-unit decimal digits.
    pub currency_precision: u8,
    /// Positive threshold in minor units.
    pub threshold_minor_units: u64,
    /// Threshold effective day.
    pub threshold_effective_day: i32,
    /// Threshold configuration version.
    pub threshold_config_version: u32,
    /// Parsing-policy version.
    pub parsing_policy_version: u32,
    /// Supported amount ceiling in minor units.
    pub max_amount_minor_units: u64,
    /// Existing namespace to reuse (resume/retry), or `None` to generate one.
    pub namespace: Option<EventNamespace>,
}

/// Response to session creation.
#[derive(Debug, Clone)]
pub struct CreateSessionResponse {
    /// The assigned session id.
    pub session_id: String,
    /// The bound (or reused) namespace, hyphenated.
    pub event_namespace: String,
    /// The accepted configuration fingerprint, hex.
    pub config_fingerprint: String,
    /// The value policy in force for this session's source type.
    pub substrate_value_policy: String,
    /// Initial state.
    pub state: SessionState,
}

/// One live session plus its substrate applier.
struct SessionEntry {
    /// The session.
    session: IngestionSession,
    /// The substrate application boundary for this session.
    applier: SubstrateApplier,
}

/// The reusable ingestion service.
///
/// Holds no transport concerns. `Mutex`-wrapped by the server so that one
/// operation at a time mutates substrate state, matching the repository's
/// serialized-writer convention.
pub struct IngestionService {
    /// Namespace registry with collision detection (Canonical Contract §3.2).
    namespaces: NamespaceRegistry,
    /// Live sessions by id.
    sessions: HashMap<String, SessionEntry>,
    /// Server-enforced limits applied to new sessions.
    limits: SessionLimits,
    /// Monotonic session-id allocator.
    next_id: AtomicU64,
    /// The durable write-ahead journal, when persistence is enabled
    /// (Amendment A2). Without it the service is ephemeral, exactly as in
    /// Phase 1.
    journal: Option<Arc<Journal>>,
    /// Once a journal append fails, every mutating operation is refused until
    /// restart: in-memory state may be ahead of the journal, and serving it
    /// would break the journal-equals-acked-state contract. Recovery restores
    /// the journal-authoritative state.
    journal_failed: bool,
}

impl IngestionService {
    /// Create a service with default limits.
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(SessionLimits::default())
    }

    /// Create a service with explicit server-enforced limits (§13).
    #[must_use]
    pub fn with_limits(limits: SessionLimits) -> Self {
        Self {
            namespaces: NamespaceRegistry::new(),
            sessions: HashMap::new(),
            limits,
            next_id: AtomicU64::new(1),
            journal: None,
            journal_failed: false,
        }
    }

    fn persistence(detail: impl Into<String>) -> IngestError {
        IngestError::new(ErrorCode::Persistence, detail)
    }

    /// Refuse work once the journal is known to be behind memory.
    fn check_journal_healthy(&self) -> IngestResult<()> {
        if self.journal_failed {
            return Err(Self::persistence(
                "the durable journal failed earlier in this process; in-memory state may \
                 exceed the journal, so all operations are refused until restart (recovery \
                 restores the journal-authoritative state)",
            ));
        }
        Ok(())
    }

    /// Append to the journal, poisoning the service on failure.
    fn journal_append(&mut self, rec: JournalRecord) -> IngestResult<()> {
        match &self.journal {
            None => Ok(()),
            Some(j) => j.append(&rec).inspect_err(|_e| {
                self.journal_failed = true;
            }),
        }
    }

    /// Append a batch of journal records with one trailing fsync, poisoning
    /// the service on failure. See `Journal::append_batch` for the
    /// byte-identity and durability argument; the fail-closed contract is
    /// identical to [`Self::journal_append`].
    fn journal_batch(&mut self, recs: Vec<JournalRecord>) -> IngestResult<()> {
        match &self.journal {
            None => Ok(()),
            Some(j) => j.append_batch(&recs).inspect_err(|_e| {
                self.journal_failed = true;
            }),
        }
    }

    /// Create a durable service over `dir` (Amendment A2): open (or create)
    /// the journal, streaming_ingestion_demo it, and rebuild every session, the namespace
    /// registry, the identity ledgers, the account maps and the substrate
    /// engines through the same interfaces the live path uses.
    ///
    /// # Errors
    /// [`ErrorCode::Persistence`] on journal failure or streaming_ingestion_demo divergence.
    pub fn with_persistence(dir: &Path) -> IngestResult<Self> {
        let journal = Arc::new(Journal::open(dir)?);
        let streaming_ingestion_demo = Journal::streaming_ingestion_demo(dir)?;
        let mut svc = Self {
            namespaces: NamespaceRegistry::new(),
            sessions: HashMap::new(),
            limits: SessionLimits::default(),
            next_id: AtomicU64::new(1),
            journal: Some(journal),
            journal_failed: false,
        };
        for rec in streaming_ingestion_demo.records {
            svc.streaming_ingestion_demo_record(rec)?;
        }
        let mut max_id: u64 = 0;
        for key in svc.sessions.keys() {
            if let Some(n) = key.strip_prefix("sess-").and_then(|s| s.parse::<u64>().ok()) {
                max_id = max_id.max(n);
            }
        }
        svc.next_id = AtomicU64::new(max_id + 1);
        Ok(svc)
    }

    fn streaming_ingestion_demo_record(&mut self, rec: JournalRecord) -> IngestResult<()> {
        match rec {
            JournalRecord::SessionOpen { id, namespace, config, limits } => {
                self.namespaces.register(namespace)?;
                let policy = SubstrateValuePolicy::default_for(config.source_type);
                let session = IngestionSession::create(id.clone(), config, namespace, limits)?;
                if self.sessions.contains_key(&id) {
                    return Err(Self::persistence(format!("journal opens session {id} twice")));
                }
                self.sessions
                    .insert(id, SessionEntry { session, applier: SubstrateApplier::new(policy) });
            }
            JournalRecord::ChunkAccepted {
                id,
                sequence,
                digest,
                record_count,
                payload_bytes,
                new_events,
                duplicate_events,
            } => self.session_mut(&id)?.session.restore_chunk(
                sequence,
                digest,
                record_count,
                payload_bytes,
                new_events,
                duplicate_events,
            )?,
            JournalRecord::RecordAccepted { id, record } => {
                self.session_mut(&id)?.session.restore_record(record)?
            }
            JournalRecord::RecordConflict { id } => self.session_mut(&id)?.session.restore_conflict(),
            JournalRecord::SessionSealed { id } => {
                self.session_mut(&id)?.session.restore_state(SessionState::Sealed)
            }
            JournalRecord::SessionCancelled { id } => {
                self.session_mut(&id)?.session.restore_state(SessionState::Cancelled)
            }
            JournalRecord::EventApplied { id, event } => {
                self.session_mut(&id)?.applier.streaming_ingestion_demo_applied(
                    event.substrate_event_id,
                    event.t,
                    event.s,
                    event.r,
                    event.value_micro,
                    event.sender_wire_id,
                    event.receiver_wire_id,
                )?;
            }
            JournalRecord::SessionCommitted { id, .. } => {
                self.session_mut(&id)?.session.restore_state(SessionState::Committed)
            }
        }
        Ok(())
    }

    fn session_mut(&mut self, id: &str) -> IngestResult<&mut SessionEntry> {
        self.sessions
            .get_mut(id)
            .ok_or_else(|| Self::persistence(format!("journal references unknown session {id}")))
    }

    /// The capability advertisement.
    #[must_use]
    pub fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn next_session_id(&self) -> String {
        format!("sess-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Build and validate a [`SourceConfig`] from a request.
    ///
    /// # Errors
    /// - [`ErrorCode::AmountOrCurrency`] for a malformed currency code.
    /// - Any [`SourceConfig::validate`] rejection.
    pub fn config_from_request(req: &CreateSessionRequest) -> IngestResult<SourceConfig> {
        let currency_code = CurrencyCode::parse(&req.currency_code)?;
        let cfg = SourceConfig {
            source_type: req.source_type,
            identity_scheme: req.identity_scheme,
            account_id_width: req.account_id_width,
            currency_code,
            currency_precision: req.currency_precision,
            threshold_minor_units: req.threshold_minor_units,
            threshold_effective_day: req.threshold_effective_day,
            threshold_config_version: req.threshold_config_version,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            feature_algorithm_version: FEATURE_ALGORITHM_VERSION,
            parsing_policy_version: req.parsing_policy_version,
            max_amount_minor_units: req.max_amount_minor_units,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Create a session.
    ///
    /// # Errors
    /// - Any [`Self::config_from_request`] rejection.
    /// - [`ErrorCode::Identity`] for an all-zero or unregistered-colliding
    ///   namespace.
    pub fn create_session(
        &mut self,
        req: &CreateSessionRequest,
    ) -> IngestResult<CreateSessionResponse> {
        self.check_journal_healthy()?;
        let config = Self::config_from_request(req)?;
        // §3.2: reuse a supplied namespace (retry/resume) or generate a new one
        // (a separate import). Never regenerate a supplied namespace.
        let namespace = match req.namespace {
            Some(ns) => {
                self.namespaces.register(ns)?;
                ns
            }
            None => self.namespaces.generate()?,
        };
        let policy = SubstrateValuePolicy::default_for(config.source_type);
        let id = self.next_session_id();
        let session = IngestionSession::create(
            id.clone(),
            config.clone(),
            namespace,
            self.limits,
        )?;
        let applier = SubstrateApplier::new(policy);
        self.sessions.insert(
            id.clone(),
            SessionEntry { session, applier },
        );
        // Amendment A2: the session is acknowledged only after its creation is
        // durable. A journal failure rolls the in-memory entry back out, so
        // memory never permanently exceeds the journal.
        if let Err(e) = self.journal_append(JournalRecord::SessionOpen {
            id: id.clone(),
            namespace,
            config: config.clone(),
            limits: self.limits,
        }) {
            self.sessions.remove(&id);
            return Err(e);
        }
        let fingerprint = self
            .sessions
            .get(&id)
            .expect("session just inserted")
            .session
            .config()
            .fingerprint_hex();
        Ok(CreateSessionResponse {
            session_id: id,
            event_namespace: namespace.to_hyphenated(),
            config_fingerprint: fingerprint,
            substrate_value_policy: policy.as_str().to_string(),
            state: SessionState::Open,
        })
    }

    /// Look up a session, or fail with a session error.
    fn entry(&self, id: &str) -> IngestResult<&SessionEntry> {
        self.sessions.get(id).ok_or_else(|| {
            IngestError::new(
                ErrorCode::Session,
                format!("unknown session {id}"),
            )
        })
    }

    /// Mutable lookup.
    fn entry_mut(&mut self, id: &str) -> IngestResult<&mut SessionEntry> {
        self.sessions.get_mut(id).ok_or_else(|| {
            IngestError::new(
                ErrorCode::Session,
                format!("unknown session {id}"),
            )
        })
    }

    /// Session status, fingerprint and progress.
    #[must_use]
    pub fn session_status(&self, id: &str) -> Option<IngestionReport> {
        self.sessions.get(id).map(|e| IngestionReport::from_session(&e.session))
    }

    /// Submit a chunk.
    ///
    /// # Errors
    /// As [`IngestionSession::submit_chunk`], plus [`ErrorCode::Session`] for an
    /// unknown session id.
    pub fn submit_chunk(&mut self, id: &str, sequence: u64, bytes: &[u8]) -> IngestResult<ChunkOutcome> {
        self.check_journal_healthy()?;
        let prev_records = self.entry(id)?.session.record_count() as usize;
        let outcome = self.entry_mut(id)?.session.submit_chunk(sequence, bytes)?;
        if outcome.status == ChunkStatus::Accepted {
            // Amendment A2: journal the chunk bookkeeping and every newly
            // staged record before acknowledging the upload. The chunk digest
            // is the session's own `sha256(bytes)`, so an idempotent retry
            // after a restart is recognised without re-decoding.
            //
            // One batch, one fsync: the frames are byte-identical to the
            // per-record appends, so recovery is unchanged.
            let digest = sha256(bytes);
            let id_owned = id.to_string();
            let mut batch = Vec::with_capacity(
                outcome.new_events as usize + outcome.conflicting_events as usize + 1,
            );
            batch.push(JournalRecord::ChunkAccepted {
                id: id_owned.clone(),
                sequence,
                digest,
                record_count: outcome.record_count,
                payload_bytes: bytes.len() as u64,
                new_events: outcome.new_events,
                duplicate_events: outcome.duplicate_events,
            });
            let staged: Vec<crate::record::CanonicalRecord> = self
                .entry(id)?
                .session
                .records()
                .get(prev_records..)
                .unwrap_or(&[])
                .to_vec();
            for record in staged {
                batch.push(JournalRecord::RecordAccepted {
                    id: id_owned.clone(),
                    record,
                });
            }
            // Amendment A2 §16.2: "identity conflicts" are an acknowledged
            // state change and MUST be journaled. Without a RecordConflict
            // frame the conflict count is lost on restart while the input
            // record is still counted, breaking the §10 reconciliation
            // invariant `input = accepted + duplicate + conflicting +
            // rejected`. One frame per conflict reproduces the live counter
            // and error entry exactly on streaming_ingestion_demo (`restore_conflict`).
            for _ in 0..outcome.conflicting_events {
                batch.push(JournalRecord::RecordConflict {
                    id: id_owned.clone(),
                });
            }
            self.journal_batch(batch)?;
        }
        Ok(outcome)
    }

    /// Chunk status.
    ///
    /// # Errors
    /// [`ErrorCode::Session`] for an unknown session or unknown sequence.
    pub fn chunk_status(&self, id: &str, sequence: u64) -> IngestResult<ChunkStatus> {
        self.entry(id)?.session.chunk_status(sequence)
    }

    /// Validate the session without applying anything.
    ///
    /// # Errors
    /// As [`IngestionSession::validate_only`].
    pub fn validate(&self, id: &str) -> IngestResult<()> {
        self.entry(id)?.session.validate_only()
    }

    /// Seal a session.
    ///
    /// # Errors
    /// As [`IngestionSession::seal`].
    pub fn seal(&mut self, id: &str) -> IngestResult<()> {
        self.check_journal_healthy()?;
        self.entry_mut(id)?.session.seal()?;
        self.journal_append(JournalRecord::SessionSealed { id: id.to_string() })
    }

    /// Commit a session: apply every validated event to the substrate.
    ///
    /// `attest` supplies the exact minor-unit amount per record where the value
    /// policy requires one. Returning `None` for a record is a rejection, never
    /// a fabricated value.
    ///
    /// # Errors
    /// - [`ErrorCode::Session`] for an unknown session.
    /// - [`ErrorCode::SpecificationGap`] when the policy cannot obtain a value.
    /// - Any validation or application rejection.
    pub fn commit<F>(&mut self, id: &str, attest: F) -> IngestResult<crate::report::CommitResult>
    where
        F: FnMut(usize, &crate::record::CanonicalRecord) -> Option<i64>,
    {
        self.check_journal_healthy()?;
        let journal = self.journal.clone();
        let id_owned = id.to_string();
        let e = self.entry_mut(id)?;
        // The per-event sink buffers EventApplied frames and flushes them in
        // one batch at commit end, so the whole commit pays one fsync instead
        // of one per event. Byte-identity with the per-event appends is
        // preserved, so recovery granularity is unchanged.
        let mut applied_batch: Vec<JournalRecord> = Vec::new();
        let buffered_sink = |ev: &crate::session::AppliedEvent| -> IngestResult<()> {
            match &journal {
                None => Ok(()),
                Some(_) => {
                    applied_batch.push(JournalRecord::EventApplied {
                        id: id_owned.clone(),
                        event: *ev,
                    });
                    Ok(())
                }
            }
        };
        let mut buffered_sink = buffered_sink;
        let result = e.session.commit_with_sink(&mut e.applier, attest, |ev| {
            buffered_sink(ev)
        });
        let result = match result {
            Ok(r) => r,
            Err(err) => {
                if err.code == ErrorCode::Persistence {
                    // The sink failed mid-commit: memory may hold events the
                    // journal never saw. Poison until restart; recovery
                    // streaming_ingestion_demos the durable prefix and the substrate CAS makes
                    // the client's retry idempotent.
                    //
                    // The sink itself never returns Persistence errors anymore
                    // (it only buffers), but the fail-closed branch is kept so
                    // the contract survives any future fallible sink.
                    self.journal_failed = true;
                }
                return Err(err);
            }
        };
        if self.journal.is_some() {
            applied_batch.push(JournalRecord::SessionCommitted {
                id: id.to_string(),
                applied: result.applied,
                already_applied: result.already_applied,
            });
            self.journal_batch(applied_batch)?;
        }
        Ok(result)
    }

    /// Cancel a session (safe cleanup).
    ///
    /// # Errors
    /// [`ErrorCode::Session`] for an unknown session.
    pub fn cancel(&mut self, id: &str) -> IngestResult<()> {
        self.check_journal_healthy()?;
        self.entry_mut(id)?.session.cancel();
        self.journal_append(JournalRecord::SessionCancelled { id: id.to_string() })
    }

    /// The final report for a session, including commit outcome if committed.
    ///
    /// # Errors
    /// [`ErrorCode::Session`] for an unknown session.
    pub fn report(&self, id: &str) -> IngestResult<IngestionReport> {
        let e = self.entry(id)?;
        let mut r = IngestionReport::from_session(&e.session);
        if e.session.state() == SessionState::Committed {
            r = r.with_commit(crate::report::CommitResult {
                applied: e.applier.applied(),
                already_applied: e.applier.duplicates(),
            });
        }
        Ok(r)
    }

    /// Number of live sessions (diagnostics).
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// The server-enforced limits.
    #[must_use]
    pub const fn limits(&self) -> &SessionLimits {
        &self.limits
    }
}

impl Default for IngestionService {
    fn default() -> Self {
        Self::new()
    }
}
