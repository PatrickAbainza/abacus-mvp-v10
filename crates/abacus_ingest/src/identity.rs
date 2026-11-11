//! Event-identity ledger: deduplication and conflicting-identity rejection.
//!
//! Golden Source §5.1 fixes the two outcomes that must be distinguished:
//!
//! - "Same identity and same canonical event content: treat as a duplicate
//!   under the declared deduplication policy."
//! - "Same identity and conflicting content: reject and report a conflict."
//!
//! and §5.1 also requires: "Do not rely on chunk idempotency as a substitute
//! for event-level deduplication." So this ledger is keyed on the compound
//! identity `(namespace, event_id)` — never on chunk sequence — and it is the
//! mechanism that makes application idempotent.
//!
//! ## Content fingerprint
//!
//! "Same canonical content" is decided by a SHA-256 digest over the record's
//! canonical fields **excluding** the identity itself (the namespace and event
//! id are already the key). Including them would make every conflict
//! undetectable, and including anything non-canonical would make a redelivery
//! look like a conflict. So the digest covers exactly: `event_time_ms`,
//! `sender_id`, `receiver_id`, and the seven feature bytes — the complete
//! logical content of the event.
//!
//! ## Relationship to the substrate ledger
//!
//! `abacus_tensor::ingest::ledger` implements `d_cas` over a flat `u64` event
//! id. That is the model-level exactly-once mechanism for the substrate, but it
//! cannot see the namespace and cannot distinguish "same content" from
//! "conflicting content" — it only reports `Retry`. This ledger supplies the
//! namespace-aware, content-aware decision and hands the substrate a
//! namespace-local sequential id, so both layers keep their own contract.

use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::hash::sha256;
use crate::namespace::{EventIdentity, EventNamespace};
use std::collections::HashMap;

/// What the identity ledger decided about one delivered event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityOutcome {
    /// First delivery of this identity: apply it.
    New,
    /// Same identity, byte-identical content: duplicate delivery, do not apply
    /// again (Golden Source §5.1).
    Duplicate,
    /// Same identity, different content: reject and report a conflict (§5.1).
    Conflict,
}

/// Per-namespace record of what has been accepted.
#[derive(Debug, Clone)]
struct NamespaceLedger {
    /// Event id -> content digest of the accepted event.
    seen: HashMap<u64, [u8; 32]>,
    /// Next namespace-local sequential id handed to the substrate.
    next_substrate_id: u64,
    /// Count of accepted (first-delivery) events in this namespace.
    accepted: u64,
    /// Count of duplicate deliveries observed.
    duplicates: u64,
    /// Count of conflicting identities observed.
    conflicts: u64,
}

impl NamespaceLedger {
    fn new() -> Self {
        Self {
            seen: HashMap::new(),
            next_substrate_id: 1,
            accepted: 0,
            duplicates: 0,
            conflicts: 0,
        }
    }
}

/// The namespace-aware event-identity ledger.
///
/// One instance covers all namespaces, because the identity is compound: the
/// same `event_id` in two different namespaces is two different events.
#[derive(Debug, Default)]
pub struct IdentityLedger {
    by_namespace: HashMap<EventNamespace, NamespaceLedger>,
}

impl IdentityLedger {
    /// Empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The content digest of a canonical event.
    ///
    /// Covers the logical content only — the identity fields are the key and
    /// are deliberately excluded so that conflicts remain detectable.
    #[must_use]
    pub fn content_digest(
        event_time_ms: i64,
        sender_id: u64,
        receiver_id: u64,
        log_magnitude: u16,
        relative_threshold_distance: i32,
        roundedness_flags: u8,
    ) -> [u8; 32] {
        let mut buf = Vec::with_capacity(23);
        buf.extend_from_slice(&event_time_ms.to_le_bytes());
        buf.extend_from_slice(&sender_id.to_le_bytes());
        buf.extend_from_slice(&receiver_id.to_le_bytes());
        buf.extend_from_slice(&log_magnitude.to_le_bytes());
        buf.extend_from_slice(&relative_threshold_distance.to_le_bytes());
        buf.push(roundedness_flags);
        sha256(&buf)
    }

    /// Record one delivered event and classify it.
    ///
    /// # Returns
    /// - [`IdentityOutcome::New`] plus the namespace-local substrate id to use
    ///   for application.
    /// - [`IdentityOutcome::Duplicate`] with `None`: already applied, skip.
    ///
    /// # Errors
    /// [`ErrorCode::Identity`] on [`IdentityOutcome::Conflict`], so a conflict
    /// is a hard failure the caller must surface, not a silent skip.
    pub fn check(
        &mut self,
        identity: EventIdentity,
        content: [u8; 32],
    ) -> IngestResult<(IdentityOutcome, Option<u64>)> {
        if identity.namespace.is_zero() {
            return Err(IngestError::new(
                ErrorCode::Identity,
                "the all-zero namespace is not a valid identity domain",
            ));
        }
        let nl = self
            .by_namespace
            .entry(identity.namespace)
            .or_insert_with(NamespaceLedger::new);
        match nl.seen.get(&identity.event_id) {
            None => {
                nl.seen.insert(identity.event_id, content);
                let id = nl.next_substrate_id;
                nl.next_substrate_id = nl.next_substrate_id.checked_add(1).ok_or_else(|| {
                    IngestError::new(
                        ErrorCode::Identity,
                        "substrate event-id space exhausted for this namespace",
                    )
                })?;
                nl.accepted += 1;
                Ok((IdentityOutcome::New, Some(id)))
            }
            Some(prev) if *prev == content => {
                nl.duplicates += 1;
                Ok((IdentityOutcome::Duplicate, None))
            }
            Some(_) => {
                nl.conflicts += 1;
                Err(IngestError::new(
                    ErrorCode::Identity,
                    "conflicting content for an existing event identity",
                ))
            }
        }
    }

    /// Accepted (first-delivery) event count for one namespace.
    #[must_use]
    pub fn accepted(&self, ns: &EventNamespace) -> u64 {
        self.by_namespace.get(ns).map_or(0, |n| n.accepted)
    }

    /// Duplicate-delivery count for one namespace.
    #[must_use]
    pub fn duplicates(&self, ns: &EventNamespace) -> u64 {
        self.by_namespace.get(ns).map_or(0, |n| n.duplicates)
    }

    /// Conflicting-identity count for one namespace.
    #[must_use]
    pub fn conflicts(&self, ns: &EventNamespace) -> u64 {
        self.by_namespace.get(ns).map_or(0, |n| n.conflicts)
    }

    /// Total distinct identities recorded across all namespaces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_namespace.values().map(|n| n.seen.len()).sum()
    }

    /// True iff nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeaturePayload;
    use crate::namespace::EventNamespace;

    fn ns_a() -> EventNamespace {
        EventNamespace::from_bytes([1u8; 16])
    }
    fn ns_b() -> EventNamespace {
        EventNamespace::from_bytes([2u8; 16])
    }

    fn digest_of(t: i64, s: u64, r: u64, f: FeaturePayload) -> [u8; 32] {
        IdentityLedger::content_digest(
            t,
            s,
            r,
            f.log_magnitude,
            f.relative_threshold_distance,
            f.roundedness_flags,
        )
    }

    #[test]
    fn first_delivery_is_new_and_gets_a_substrate_id() {
        let mut led = IdentityLedger::new();
        let d = digest_of(1, 2, 3, FeaturePayload { log_magnitude: 1, relative_threshold_distance: 0, roundedness_flags: 0 });
        let (o, id) = led.check(EventIdentity::new(ns_a(), 7), d).unwrap();
        assert_eq!(o, IdentityOutcome::New);
        assert_eq!(id, Some(1));
        assert_eq!(led.accepted(&ns_a()), 1);
    }

    #[test]
    fn identical_content_is_a_duplicate_and_not_reapplied() {
        let mut led = IdentityLedger::new();
        let d = digest_of(1, 2, 3, FeaturePayload { log_magnitude: 1, relative_threshold_distance: 0, roundedness_flags: 0 });
        let (o1, id1) = led.check(EventIdentity::new(ns_a(), 7), d).unwrap();
        let (o2, id2) = led.check(EventIdentity::new(ns_a(), 7), d).unwrap();
        assert_eq!(o1, IdentityOutcome::New);
        assert_eq!(o2, IdentityOutcome::Duplicate);
        assert!(id1.is_some());
        assert!(id2.is_none(), "a duplicate must not receive a substrate id");
        assert_eq!(led.accepted(&ns_a()), 1);
        assert_eq!(led.duplicates(&ns_a()), 1);
    }

    #[test]
    fn conflicting_content_is_rejected_and_counted() {
        let mut led = IdentityLedger::new();
        let d1 = digest_of(1, 2, 3, FeaturePayload { log_magnitude: 1, relative_threshold_distance: 0, roundedness_flags: 0 });
        let d2 = digest_of(1, 2, 4, FeaturePayload { log_magnitude: 1, relative_threshold_distance: 0, roundedness_flags: 0 });
        led.check(EventIdentity::new(ns_a(), 7), d1).unwrap();
        let err = led.check(EventIdentity::new(ns_a(), 7), d2).unwrap_err();
        assert_eq!(err.code, ErrorCode::Identity);
        assert_eq!(led.conflicts(&ns_a()), 1);
        assert_eq!(led.accepted(&ns_a()), 1, "conflict must not be accepted");
    }

    #[test]
    fn same_event_id_in_different_namespace_is_a_different_event() {
        // §3.1: the u64 is unique within its namespace, not globally. So the
        // SAME u64 in two namespaces is two distinct events and the second is
        // NEW, not a duplicate.
        let mut led = IdentityLedger::new();
        let d = digest_of(5, 6, 7, FeaturePayload { log_magnitude: 3, relative_threshold_distance: 0, roundedness_flags: 0 });
        let (o1, _) = led.check(EventIdentity::new(ns_a(), 1), d).unwrap();
        let (o2, _) = led.check(EventIdentity::new(ns_b(), 1), d).unwrap();
        assert_eq!(o1, IdentityOutcome::New);
        assert_eq!(o2, IdentityOutcome::New, "same u64 in another namespace is NEW");
        assert_eq!(led.len(), 2);
        assert_eq!(led.accepted(&ns_a()), 1);
        assert_eq!(led.accepted(&ns_b()), 1);
        // Counters are namespace-local, which is safe because a session binds
        // exactly ONE namespace — so a session's substrate ids never collide.
        assert_eq!(led.duplicates(&ns_a()), 0);
    }

    #[test]
    fn substrate_ids_are_namespace_local_and_monotone() {
        let mut led = IdentityLedger::new();
        for i in 0..5u64 {
            let d = digest_of(i as i64, i, i, FeaturePayload { log_magnitude: 0, relative_threshold_distance: 0, roundedness_flags: 0 });
            let (o, id) = led.check(EventIdentity::new(ns_a(), i), d).unwrap();
            assert_eq!(o, IdentityOutcome::New);
            assert_eq!(id, Some(i + 1));
        }
    }

    #[test]
    fn zero_namespace_rejected() {
        let mut led = IdentityLedger::new();
        let d = digest_of(1, 1, 1, FeaturePayload { log_magnitude: 0, relative_threshold_distance: 0, roundedness_flags: 0 });
        let err = led
            .check(EventIdentity::new(EventNamespace::from_bytes([0u8; 16]), 1), d)
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Identity);
    }

    #[test]
    fn content_digest_ignores_identity_fields() {
        // Two identities with identical content must produce the SAME digest,
        // which is what allows "same content" to be detected.
        let a = digest_of(9, 8, 7, FeaturePayload { log_magnitude: 2, relative_threshold_distance: -3, roundedness_flags: 1 });
        let b = digest_of(9, 8, 7, FeaturePayload { log_magnitude: 2, relative_threshold_distance: -3, roundedness_flags: 1 });
        assert_eq!(a, b);
    }

    #[test]
    fn content_digest_changes_with_every_content_field() {
        let base = FeaturePayload { log_magnitude: 2, relative_threshold_distance: -3, roundedness_flags: 1 };
        let d0 = digest_of(9, 8, 7, base);
        assert_ne!(digest_of(10, 8, 7, base), d0);
        assert_ne!(digest_of(9, 9, 7, base), d0);
        assert_ne!(digest_of(9, 8, 9, base), d0);
        assert_ne!(
            digest_of(9, 8, 7, FeaturePayload { log_magnitude: 3, relative_threshold_distance: -3, roundedness_flags: 1 }),
            d0
        );
        assert_ne!(
            digest_of(9, 8, 7, FeaturePayload { log_magnitude: 2, relative_threshold_distance: -4, roundedness_flags: 1 }),
            d0
        );
        assert_ne!(
            digest_of(9, 8, 7, FeaturePayload { log_magnitude: 2, relative_threshold_distance: -3, roundedness_flags: 0 }),
            d0
        );
    }
}
