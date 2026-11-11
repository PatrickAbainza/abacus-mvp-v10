//! Ingestion-native account/identity/feature model for the pipeline.
//!
//! Migration (per the reconciliation directive): `abacus_pipeline` now carries
//! the **CSV/Kafka ingestion** model instead of its former role-separated
//! string dictionaries and `SAMCORE02` artifact model. This module is the
//! shared vocabulary that the snapshot, apply path, streaming_ingestion_demo source, web and
//! graph layers use.
//!
//! Grounding (the ported `abacus_ingest` oracle is the source of truth):
//! - **One shared account namespace.** A single append-only wire-u64 -> u32 map
//!   (`abacus_ingest::account::AccountMap`, GS §6: "sender and receiver IDs
//!   share one namespace"; code 0 reserved). The pipeline's former
//!   role-separated `sender_codes`/`receiver_codes` (same account -> different
//!   code by role) are retired in favor of this.
//! - **Compound event identity.** `(namespace u128, event_id u64)`
//!   (`abacus_ingest::namespace::EventIdentity`, Canonical Contract §3), used
//!   for the identity-ledger dedup, replacing the flat `tx_id` + content key
//!   `(t, s, r, amount)`.
//! - **Privacy-preserving derived metrics.** Each committed event carries the
//!   frozen seven-byte amount-derived [`FeaturePayload`] (GS §9) rather than a
//!   raw amount or the old global z-score/`log1p` features. The exact amount
//!   still lives in the substrate value (`IngestionEngine`), but the pipeline's
//!   queryable view exposes only these derived metrics.

pub use abacus_ingest::account::AccountMap;
pub use abacus_ingest::features::FeaturePayload;
pub use abacus_ingest::namespace::{EventIdentity, EventNamespace};

use abacus_tensor::coo::entry::TxEntry;

/// One committed event as carried by the pipeline's ingestion-native view.
///
/// The substrate event id (`locator`) is the flat `u64` the ingestion ledger
/// maps the compound identity onto; `s`/`r` are codes in the single shared
/// account namespace. `features` are the privacy-preserving derived metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IngestedEvent {
    /// Compound identity `(namespace, event_id)` (Canonical Contract §3).
    pub identity: EventIdentity,
    /// Event time `t` (epoch seconds).
    pub t: u64,
    /// Sender account code in the shared namespace.
    pub s: u32,
    /// Receiver account code in the shared namespace.
    pub r: u32,
    /// Frozen seven-byte amount-derived feature payload (GS §9).
    pub features: FeaturePayload,
}

impl IngestedEvent {
    /// Build from an applied substrate entry plus its derived features.
    #[must_use]
    pub fn new(identity: EventIdentity, t: u64, s: u32, r: u32, features: FeaturePayload) -> Self {
        Self { identity, t, s, r, features }
    }

    /// The substrate coordinate triple `(t, s, r)`.
    #[must_use]
    pub const fn coord(&self) -> (u64, u32, u32) {
        (self.t, self.s, self.r)
    }

    /// As a substrate `TxEntry` shape (used by derived indexes; the value is
    /// intentionally NOT carried here — it lives in the substrate and is never
    /// exposed through this view).
    #[must_use]
    pub const fn as_entry(&self) -> TxEntry {
        TxEntry::new(self.t, self.s, self.r, 0)
    }
}

/// Render a shared-namespace account code as its stable UI node key.
///
/// Account labels come from the single shared namespace, so the same account
/// has the same key whether it appears as sender or receiver (the retired
/// role-separated model could not guarantee this). The wire identifier is
/// looked up by the caller through [`AccountMap::wire_of`].
#[must_use]
pub fn account_node_id(code: u32) -> String {
    format!("acct:{code}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(byte: u8) -> EventNamespace {
        EventNamespace::from_bytes([byte; 16])
    }

    fn feats() -> FeaturePayload {
        // amount=12_345 minor, threshold=10_000, precision 2 (from the oracle
        // test: log_magnitude 9421, rel_dist 235, roundedness 0).
        FeaturePayload::derive(12_345, 10_000, 100).unwrap()
    }

    #[test]
    fn one_shared_namespace_for_sender_and_receiver() {
        // GS §6: one namespace, so the same wire id interns to one code in
        // both roles. The retired role-separated model would give two codes.
        let mut m = AccountMap::new();
        let as_sender = m.intern(555).unwrap();
        let as_receiver = m.intern(555).unwrap();
        assert_eq!(as_sender, as_receiver);
        assert_eq!(m.account_count(), 1);
    }

    #[test]
    fn compound_identity_distinguishes_namespaces() {
        // Canonical Contract §3: (namespace, event_id) is the identity, not the
        // bare event_id. Same event_id under different namespaces is distinct.
        let a = EventIdentity { namespace: ns(1), event_id: 7 };
        let b = EventIdentity { namespace: ns(2), event_id: 7 };
        assert_ne!(a.namespace, b.namespace);
        assert_eq!(a.event_id, b.event_id);
    }

    #[test]
    fn ingested_event_carries_derived_features_not_amount() {
        let id = EventIdentity { namespace: ns(9), event_id: 3 };
        let ev = IngestedEvent::new(id, 100, 1, 2, feats());
        assert_eq!(ev.coord(), (100, 1, 2));
        // The derived metrics are carried verbatim.
        assert_eq!(ev.features.log_magnitude, 9_421);
        assert_eq!(ev.features.relative_threshold_distance, 235);
        assert_eq!(ev.features.roundedness_flags, 0);
        // The substrate entry shape carries NO amount (value 0) — amount is
        // never surfaced through this view (privacy).
        assert_eq!(ev.as_entry().value, 0);
    }

    #[test]
    fn account_node_id_is_role_independent() {
        // Same code -> same node key regardless of role.
        assert_eq!(account_node_id(42), "acct:42");
    }
}
