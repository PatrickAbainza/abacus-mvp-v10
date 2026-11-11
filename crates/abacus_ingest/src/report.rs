//! Machine-readable ingestion reports.
//!
//! Canonical Contract §10 enumerates what every session must report, and adds
//! the hard requirement: "Counts must reconcile. Every input record must be
//! accounted for under the declared parsing policy." §11 then requires that the
//! report "distinguish rejected records, quarantined records, duplicate
//! deliveries, and conflicting identities."
//!
//! This module renders exactly those fields, and [`IngestionReport::reconciles`]
//! is the machine check of the reconciliation invariant so a caller can assert
//! it rather than trust it.
//!
//! ## Privacy (§16, §17.7)
//!
//! Reports carry no raw account strings and no raw amounts. The only
//! identifiers present are the opaque event namespace, the session id and
//! per-record *ordinals*. Errors are the structured [`crate::error::IngestError`]
//! values, whose `detail` strings are restricted to structural facts.

use crate::config::{hex_encode, SHA256_LEN, SourceConfig, SourceType, IdentityScheme};
use crate::error::IngestError;
use crate::session::{IngestionSession, SessionState};

/// Result of a commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CommitResult {
    /// Events committed to the substrate through the CAS path.
    pub applied: u64,
    /// Events the substrate reported as already applied (model-level
    /// exactly-once, P7).
    pub already_applied: u64,
}

/// A complete, machine-readable ingestion report (Canonical Contract §10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestionReport {
    /// Session identifier.
    pub session_id: String,
    /// Source type.
    pub source_type: SourceType,
    /// Identity scheme.
    pub identity_scheme: IdentityScheme,
    /// Event namespace, hyphenated.
    pub event_namespace: String,
    /// Wire-format version.
    pub wire_format_version: u16,
    /// Feature-schema version.
    pub feature_schema_version: u16,
    /// Feature-algorithm version.
    pub feature_algorithm_version: u16,
    /// Parsing-policy version.
    pub parsing_policy_version: u32,
    /// Timestamp conversion-rule version.
    pub timestamp_conversion_version: u32,
    /// Configuration fingerprint (lowercase hex).
    pub config_fingerprint: String,
    /// Selected account-ID width in bytes.
    pub account_id_width: u8,
    /// Chunks accepted.
    pub chunk_count: u64,
    /// Total input records decoded across all accepted chunks.
    pub input_records: u64,
    /// Events accepted as new.
    pub accepted_records: u64,
    /// Duplicate deliveries (§11 category).
    pub duplicate_records: u64,
    /// Conflicting identities (§11 category).
    pub conflicting_records: u64,
    /// Records rejected by structural validation (§11 category).
    pub rejected_records: u64,
    /// Total accepted payload bytes.
    pub payload_bytes: u64,
    /// Minimum accepted event time in milliseconds.
    pub min_event_time_ms: Option<i64>,
    /// Maximum accepted event time in milliseconds.
    pub max_event_time_ms: Option<i64>,
    /// Substrate-application outcome, if committed.
    pub commit: Option<CommitResult>,
    /// Final session state.
    pub final_state: SessionState,
    /// Structured validation errors (no raw accounts or amounts).
    pub errors: Vec<IngestError>,
    /// SHA-256 of the accepted payload, if computed.
    pub artifact_hash: Option<[u8; SHA256_LEN]>,
}

impl IngestionReport {
    /// Build a report from a session's current state.
    #[must_use]
    pub fn from_session(session: &IngestionSession) -> Self {
        let cfg: &SourceConfig = session.config();
        Self {
            session_id: session.id().to_string(),
            source_type: cfg.source_type,
            identity_scheme: cfg.identity_scheme,
            event_namespace: session.namespace().to_hyphenated(),
            wire_format_version: crate::config::FORMAT_VERSION,
            feature_schema_version: cfg.feature_schema_version,
            feature_algorithm_version: cfg.feature_algorithm_version,
            parsing_policy_version: cfg.parsing_policy_version,
            timestamp_conversion_version: crate::timestamp::TIMESTAMP_CONVERSION_VERSION,
            config_fingerprint: cfg.fingerprint_hex(),
            account_id_width: cfg.account_id_width,
            chunk_count: session.chunk_count(),
            input_records: session.total_delivered(),
            accepted_records: session.record_count(),
            duplicate_records: session.duplicate_events(),
            conflicting_records: session.conflicting_events(),
            // Disjoint from `conflicting_records`: a conflict is counted once,
            // under `conflicting_records`, never also here
            // (Canonical Contract §11).
            rejected_records: session.rejected_records(),
            payload_bytes: session.total_bytes(),
            min_event_time_ms: session.min_event_time_ms(),
            max_event_time_ms: session.max_event_time_ms(),
            commit: None,
            final_state: session.state(),
            errors: session.errors().to_vec(),
            artifact_hash: None,
        }
    }

    /// Attach a commit outcome (builder-style).
    #[must_use]
    pub fn with_commit(mut self, commit: CommitResult) -> Self {
        self.commit = Some(commit);
        self
    }

    /// Attach an artifact hash (builder-style).
    #[must_use]
    pub fn with_artifact_hash(mut self, hash: [u8; SHA256_LEN]) -> Self {
        self.artifact_hash = Some(hash);
        self
    }

    /// The reconciliation invariant (Canonical Contract §10).
    ///
    /// `input = accepted + duplicate + conflicting + rejected`. Every input
    /// record must be accounted for under exactly one category, so the
    /// categories must partition the input set.
    #[must_use]
    pub fn reconciles(&self) -> bool {
        let sum = self
            .accepted_records
            .checked_add(self.duplicate_records)
            .and_then(|s| s.checked_add(self.conflicting_records))
            .and_then(|s| s.checked_add(self.rejected_records));
        sum == Some(self.input_records)
    }

    /// The report as a JSON object (API form).
    ///
    /// Uses the API layer's minimal JSON model so no serialization dependency is
    /// introduced. Contains no raw accounts and no amounts.
    #[must_use]
    pub fn to_json(&self) -> crate::api::json::Json {
        use crate::api::json::Json;
        let mut o = Json::object();
        o.set("session_id", Json::str(&self.session_id));
        o.set("source_type", Json::str(self.source_type.as_str()));
        o.set("identity_scheme", Json::str(self.identity_scheme.as_str()));
        o.set("event_namespace", Json::str(&self.event_namespace));
        o.set("wire_format_version", Json::uint(u64::from(self.wire_format_version)));
        o.set(
            "feature_schema_version",
            Json::uint(u64::from(self.feature_schema_version)),
        );
        o.set(
            "feature_algorithm_version",
            Json::uint(u64::from(self.feature_algorithm_version)),
        );
        o.set(
            "parsing_policy_version",
            Json::uint(u64::from(self.parsing_policy_version)),
        );
        o.set(
            "timestamp_conversion_version",
            Json::uint(u64::from(self.timestamp_conversion_version)),
        );
        o.set("config_fingerprint", Json::str(&self.config_fingerprint));
        o.set("account_id_width", Json::uint(u64::from(self.account_id_width)));
        o.set("chunk_count", Json::uint(self.chunk_count));
        o.set("counts", counts_json(self));
        o.set("payload_bytes", Json::uint(self.payload_bytes));
        match (self.min_event_time_ms, self.max_event_time_ms) {
            (Some(lo), Some(hi)) => {
                o.set("min_event_time_ms", Json::int(lo));
                o.set("max_event_time_ms", Json::int(hi));
            }
            _ => {
                o.set("min_event_time_ms", Json::Null);
                o.set("max_event_time_ms", Json::Null);
            }
        }
        if let Some(c) = &self.commit {
            let mut cj = Json::object();
            cj.set("applied", Json::uint(c.applied));
            cj.set("already_applied", Json::uint(c.already_applied));
            o.set("commit", cj);
        } else {
            o.set("commit", Json::Null);
        }
        o.set("final_state", Json::str(self.final_state.as_str()));
        o.set("reconciles", Json::bool(self.reconciles()));
        if let Some(h) = &self.artifact_hash {
            o.set("artifact_hash", Json::str(hex_encode(h)));
        } else {
            o.set("artifact_hash", Json::Null);
        }
        o.set("errors", errors_json(&self.errors));
        o
    }
}

fn counts_json(r: &IngestionReport) -> crate::api::json::Json {
    use crate::api::json::Json;
    let mut c = Json::object();
    c.set("input", Json::uint(r.input_records));
    c.set("accepted", Json::uint(r.accepted_records));
    c.set("duplicate", Json::uint(r.duplicate_records));
    c.set("conflicting", Json::uint(r.conflicting_records));
    c.set("rejected", Json::uint(r.rejected_records));
    c
}

fn errors_json(errs: &[IngestError]) -> crate::api::json::Json {
    use crate::api::json::Json;
    Json::Array(
        errs.iter()
            .map(|e| {
                let mut o = Json::object();
                o.set("code", Json::str(e.code.as_str()));
                o.set("detail", Json::str(&e.detail));
                o
            })
            .collect(),
    )
}
