//! The substrate application boundary.
//!
//! Golden Source §14: "Reuse the existing incremental update path. Do not build
//! a second CSF or index implementation inside an adapter." and §11: "All
//! sources converge on one Rust event-application interface."
//!
//! So this module contains **no tensor algebra**. It translates a validated
//! canonical record into the substrate's existing types and calls the existing
//! `IngestionEngine::ingest` — the ledger CAS (`d_cas`), the elementary delta
//! (P8) and the frontier append (P9) all happen inside the substrate, exactly
//! once, unchanged. Everything here is translation plus policy.
//!
//! ## The two translations that are unavoidable
//!
//! 1. **Account width.** The wire permits up to 64-bit identifiers; the
//!    substrate coordinate is `u32`. [`crate::account::AccountMap`] performs
//!    the append-only mapping.
//! 2. **Time unit.** Wire milliseconds to substrate seconds — see
//!    [`crate::timestamp`].
//!
//! ## The value policy (resolved by Amendment A1)
//!
//! The substrate's value set `V` is exact `i64` micro-units (P2): `TxEntry`
//! carries `value`, and `LedgerOp::Create` takes a `MinorUnits`. The canonical
//! upload record carries only the seven derived feature bytes — the amount
//! travels in the commit-time attestation channel, never inside the frozen
//! ABACSEG1 record.
//!
//! Phase 1 refused browser-CSV application because the raw amount never
//! crossed the privacy boundary. **Amendment A1** (owner ruling, recorded in
//! the governing documents) now permits the exact amount to be transported
//! during ingestion provided every transient copy is cleared once the value
//! has been transformed into the substrate entry. So all source types —
//! browser CSV, SAML-D fixture, Kafka — commit through
//! [`SubstrateValuePolicy::AttestedMinorUnits`]: the caller supplies the exact
//! minor-unit amount per event, the session pre-flight verifies it against the
//! record's frozen features, and this module scales it into the substrate's
//! canonical micro-unit representation (`minor × 10^6`, the same embedding
//! `tensor::value::parse_decimal_to_minor` performs for the SAML-D fixture
//! path). [`SubstrateValuePolicy::ClientDerivedOnly`] remains only as an
//! explicit opt-out that still refuses: a feature-only substrate mode is not
//! defined anywhere in the specification, and inventing a substitute value is
//! excluded by Golden Source §19.

use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::ingest::{CasOutcome, Event, IngestionEngine, LedgerOp};

use crate::account::AccountMap;
use crate::config::SourceType;
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::record::CanonicalRecord;
use crate::timestamp::event_time_ms_to_seconds;

/// Where the substrate's exact `i64` micro-unit value comes from.
///
/// See the module documentation for why this must be explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstrateValuePolicy {
    /// The exact minor-unit amount is supplied per event by the caller. This
    /// is the default for **every** source type under Amendment A1: the
    /// browser CSV adapter transports the amount at commit time (Golden Source
    /// Amendment A1, §3/§8/§12), and the SAML-D/Kafka adapters have it inside
    /// their trusted normalization boundary (§4.2/§4.3).
    AttestedMinorUnits,
    /// Explicit opt-out: refuse application entirely. A feature-only substrate
    /// mode is not defined by the governing specification, so the only honest
    /// behaviour is [`ErrorCode::SpecificationGap`]. No caller in this crate
    /// selects this policy by default any more.
    ClientDerivedOnly,
}

impl SubstrateValuePolicy {
    /// The default policy for a source type.
    ///
    /// Amendment A1: every source transports its exact amount during ingestion
    /// (browser CSV at commit time), so every source defaults to
    /// [`SubstrateValuePolicy::AttestedMinorUnits`].
    #[must_use]
    pub const fn default_for(_source: SourceType) -> Self {
        Self::AttestedMinorUnits
    }

    /// Stable spelling for reports and the API.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AttestedMinorUnits => "attested_minor_units",
            Self::ClientDerivedOnly => "client_derived_only",
        }
    }
}

/// Result of applying one validated record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Committed to the substrate's frontier through the CAS path.
    Applied {
        /// The substrate event id used for the CAS.
        substrate_event_id: u64,
        /// Substrate `t` coordinate (seconds), for the durable journal.
        t: u64,
        /// Substrate `s` dictionary code, for the durable journal.
        s: u32,
        /// Substrate `r` dictionary code, for the durable journal.
        r: u32,
        /// The exact micro-unit value written (`minor × 10^6`), for the
        /// durable journal.
        value_micro: i64,
    },
    /// The substrate CAS reported a retry: already applied at the model level,
    /// so no delta was emitted (P7 exactly-once).
    AlreadyApplied,
}

/// The ingestion-side application boundary over the existing substrate.
///
/// Owns the account map (shared sender/receiver namespace, §6) and delegates
/// every state mutation to [`IngestionEngine`]. No CSF, index or frontier
/// structure is reimplemented here.
#[derive(Debug)]
pub struct SubstrateApplier {
    /// The existing substrate ingestion engine (ledger + frontier + sealed CSF).
    engine: IngestionEngine,
    /// Append-only wire-identifier -> substrate-code mapping.
    accounts: AccountMap,
    /// Where the exact tensor value comes from.
    policy: SubstrateValuePolicy,
    /// Total events committed through this applier.
    applied: u64,
    /// Total CAS-retry outcomes observed.
    duplicates: u64,
}

impl SubstrateApplier {
    /// Create an applier over a fresh (zero) substrate engine.
    #[must_use]
    pub fn new(policy: SubstrateValuePolicy) -> Self {
        Self {
            engine: IngestionEngine::new(),
            accounts: AccountMap::new(),
            policy,
            applied: 0,
            duplicates: 0,
        }
    }

    /// Create an applier over an existing engine (recovery / streaming_ingestion_demo).
    #[must_use]
    pub fn with_engine(
        engine: IngestionEngine,
        accounts: AccountMap,
        policy: SubstrateValuePolicy,
    ) -> Self {
        Self { engine, accounts, policy, applied: 0, duplicates: 0 }
    }

    /// The account map, for persistence and session resume.
    #[must_use]
    pub fn accounts(&self) -> &AccountMap {
        &self.accounts
    }

    /// Mutable access to the account map (resume path).
    pub fn accounts_mut(&mut self) -> &mut AccountMap {
        &mut self.accounts
    }

    /// The substrate engine (read-only; used by tests and reports).
    #[must_use]
    pub fn engine(&self) -> &IngestionEngine {
        &self.engine
    }

    /// Mutable access to the substrate engine (explicit seal/compaction only).
    pub fn engine_mut(&mut self) -> &mut IngestionEngine {
        &mut self.engine
    }

    /// Total events committed through this applier.
    #[must_use]
    pub fn applied(&self) -> u64 {
        self.applied
    }

    /// Total CAS-retry (already-applied) outcomes observed.
    #[must_use]
    pub fn duplicates(&self) -> u64 {
        self.duplicates
    }

    /// The declared value policy.
    #[must_use]
    pub const fn policy(&self) -> SubstrateValuePolicy {
        self.policy
    }

    /// Apply one validated record.
    ///
    /// Order matters: the timestamp and both account identifiers are resolved
    /// *before* the CAS, so a rejection leaves the substrate untouched. The
    /// substrate then performs the CAS and, only on `Committed`, emits exactly
    /// one elementary delta to its frontier — the incremental path, reused.
    ///
    /// # Parameters
    /// - `substrate_event_id` — the namespace-local id from
    ///   [`crate::identity::IdentityLedger`], so the compound identity maps
    ///   deterministically onto the substrate's flat `u64` id.
    /// - `attested_minor_units` — required iff the policy is
    ///   [`SubstrateValuePolicy::AttestedMinorUnits`].
    ///
    /// # Errors
    /// - [`ErrorCode::Timestamp`] for a pre-epoch event time.
    /// - [`ErrorCode::AccountId`] or [`ErrorCode::AccountIdCapacity`] for an
    ///   invalid or over-capacity identifier.
    ///   policy requires an amount and none was supplied, or if the policy
    ///   forbids amounts.
    /// - [`ErrorCode::AmountOrCurrency`] for a negative attested amount.
    /// - [`ErrorCode::FeatureOverflow`] on an unrepresentable delta or on an
    ///   amount that overflows the substrate's micro-unit value set.
    pub fn apply(
        &mut self,
        record: &CanonicalRecord,
        substrate_event_id: u64,
        attested_minor_units: Option<i64>,
    ) -> IngestResult<ApplyOutcome> {
        // 1. Value: must be declared, never invented (Amendment A1), and
        //    scaled into the substrate's canonical micro-unit representation.
        //    P2 / tensor::value: `V` is exact `i64` at scale 10^6, the same
        //    embedding `parse_decimal_to_minor` performs for fixture data, so
        //    browser- and fixture-derived values are commensurable in one
        //    tensor. Resolved BEFORE any interning so a rejected value cannot
        //    mutate even the account map (§11 atomicity, fail-closed).
        let value_minor = match self.policy {
            SubstrateValuePolicy::AttestedMinorUnits => {
                attested_minor_units.ok_or_else(|| {
                    IngestError::new(
                        ErrorCode::SpecificationGap,
                        "policy requires an attested minor-unit amount but none was supplied",
                    )
                })?
            }
            SubstrateValuePolicy::ClientDerivedOnly => {
                return Err(IngestError::new(
                    ErrorCode::SpecificationGap,
                    "a feature-only substrate mode is not defined by the governing \
                     specification, so the exact i64 micro-unit value cannot be populated",
                ))
            }
        };
        if value_minor < 0 {
            return Err(IngestError::new(
                ErrorCode::AmountOrCurrency,
                "attested minor-unit amount must be nonnegative",
            ));
        }
        let value =
            value_minor
                .checked_mul(abacus_tensor::tensor::value::MINOR_SCALE)
                .ok_or_else(|| {
                    IngestError::new(
                        ErrorCode::FeatureOverflow,
                        "attested minor-unit amount overflows the substrate's micro-unit \
                         value set (V is exact i64 at scale 10^6)",
                    )
                })?;
        // 2. Time: wire ms -> substrate seconds (rejects pre-epoch).
        let t = event_time_ms_to_seconds(record.event_time_ms)?;
        // 3. Accounts: append-only mapping into the shared u32 code namespace.
        let s = self.accounts.intern(record.sender_id)?;
        let r = self.accounts.intern(record.receiver_id)?;
        // 5. Delegate to the existing substrate update interface.
        let event = Event {
            id: substrate_event_id,
            coord: abacus_tensor::tensor::coordinate::Coord::new(t, s, r),
        };
        let outcome = self.engine.ingest(&event, LedgerOp::Create(value))?;
        match outcome {
            CasOutcome::Committed { .. } => {
                self.applied += 1;
                Ok(ApplyOutcome::Applied {
                    substrate_event_id,
                    t,
                    s,
                    r,
                    value_micro: value,
                })
            }
            CasOutcome::Retry { .. } => {
                self.duplicates += 1;
                Ok(ApplyOutcome::AlreadyApplied)
            }
        }
    }

    /// StreamingIngestionDemo one previously journaled applied event through the same CAS
    /// path (persistence recovery, Amendment A2).
    ///
    /// The durable journal records, per applied event, the substrate id, the
    /// `(t, s, r)` coordinates, the exact micro-unit value and the wire sender/
    /// receiver identifiers. Recovery interns the wire identifiers first —
    /// which deterministically rebuilds the append-only [`AccountMap`] in
    /// first-seen order — and then verifies that interning reproduces exactly
    /// the journaled codes before letting the substrate's own CAS absorb the
    /// event. Any divergence is a journal/account inconsistency and fails
    /// closed; it is never silently repaired.
    ///
    /// # Errors
    /// - [`ErrorCode::SubstrateInvariant`] if the rebuilt account codes differ
    ///   from the journaled ones.
    /// - Any substrate rejection (propagated unchanged).
    // Positional streaming_ingestion_demo of one journaled applied event; a small streaming_ingestion_demo tuple
    // would only add an allocation on the recovery path. The journal already
    // fixes the arity, so the widening is contained to recovery.
    #[allow(clippy::too_many_arguments)]
    pub fn streaming_ingestion_demo_applied(
        &mut self,
        substrate_event_id: u64,
        t: u64,
        s: u32,
        r: u32,
        value_micro: i64,
        sender_wire_id: u64,
        receiver_wire_id: u64,
    ) -> IngestResult<ApplyOutcome> {
        let s_built = self.accounts.intern(sender_wire_id)?;
        let r_built = self.accounts.intern(receiver_wire_id)?;
        if s_built != s || r_built != r {
            return Err(IngestError::new(
                ErrorCode::SubstrateInvariant,
                "journal streaming_ingestion_demo rebuilt different account codes than the journal recorded; \
                 the account map and the journal have diverged",
            ));
        }
        let event = Event {
            id: substrate_event_id,
            coord: abacus_tensor::tensor::coordinate::Coord::new(t, s, r),
        };
        match self.engine.ingest(&event, LedgerOp::Create(value_micro))? {
            CasOutcome::Committed { .. } => {
                self.applied += 1;
                Ok(ApplyOutcome::Applied {
                    substrate_event_id,
                    t,
                    s,
                    r,
                    value_micro,
                })
            }
            CasOutcome::Retry { .. } => {
                self.duplicates += 1;
                Ok(ApplyOutcome::AlreadyApplied)
            }
        }
    }

    /// Number of raw frontier entries currently buffered (diagnostics).
    #[must_use]
    pub fn frontier_len(&self) -> usize {
        self.engine.frontier().len()
    }

    /// The sealed nonzero count (diagnostics).
    #[must_use]
    pub fn sealed_nnz(&self) -> usize {
        self.engine.sealed().nnz()
    }

    /// Explicit compaction (row-21 full rebuild), available but never automatic.
    ///
    /// # Errors
    /// [`ErrorCode::SubstrateInvariant`] on an unrepresentable merge sum.
    pub fn seal(&mut self) -> IngestResult<abacus_tensor::ingest::SealReport> {
        Ok(self.engine.seal()?)
    }

    /// Export the account map for persistence.
    #[must_use]
    pub fn export_accounts(&self) -> Vec<u64> {
        self.accounts.export()
    }

    /// The substrate frontier as raw entries, for a recovery journal.
    #[must_use]
    pub fn frontier_entries(&self) -> &[TxEntry] {
        self.engine.frontier().entries()
    }
}
