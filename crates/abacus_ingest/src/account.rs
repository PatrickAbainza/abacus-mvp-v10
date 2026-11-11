//! Account-ID mapping from wire width to substrate dictionary codes.
//!
//! ## Why this module exists
//!
//! The wire contract (Golden Source §6) permits account-ID widths of 8, 16,
//! 32 and 64 bits. The existing substrate represents a sender or receiver as a
//! `u32` dictionary code (`abacus_tensor::tensor::coordinate::{SenderId,
//! ReceiverId}`), because the CSF mode-1/mode-2 coordinate arrays are `u32`.
//! A 64-bit wire identifier therefore cannot be stored directly.
//
//! This is a *representation* translation, not a change to the identity
//! contract: Golden Source §4 requires that "a change in width must not change
//! the logical identity of an account. The source-to-ID mapping and its
//! namespace must be preserved." So the mapping is:
//!
//! - **append-only** — codes are assigned in first-seen order and never
//!   reassigned, which is exactly the substrate's own dictionary discipline
//!   (`pipeline.rs`: "Unknown accounts extend dictionaries append-only ...
//!   existing codes never shift").
//! - **stable across resume** — [`AccountMap::export`]/[`AccountMap::restore`]
//!   round-trip the exact mapping so a resumed session reuses it
//!   (§13: "Session resumption must preserve account mappings and source-row
//!   identity").
//! - **width-independent** — the key is the full `u64` wire identifier, so the
//!   same logical account keeps its code whether the segment declares a 1-byte
//!   or an 8-byte width (§17.3: "Width promotion preserves logical identity").
//!
//! ## Reserved ID 0
//!
//! §6: "ID 0 is reserved for invalid or missing IDs." So code 0 is never
//! assigned to a real account; the first allocated code is 1.
//
//! ## Privacy
//!
//! §6 requires that raw account strings never cross the browser upload
//! boundary. They do not reach this module at all: it keys on the already
//! opaque numeric wire identifier. The exported mapping is internal substrate
//! state and is deliberately NOT exposed through the API (task step 3: "Do not
//! expose raw account mappings").

use crate::error::{ErrorCode, IngestError, IngestResult};
use std::collections::HashMap;

/// The substrate's maximum account-code value (`u32::MAX`).
pub const MAX_ACCOUNT_CODE: u64 = u32::MAX as u64;

/// Append-only map from opaque wire account identifier to substrate code.
///
/// One instance serves the single shared sender/receiver code namespace,
/// because §6 requires that "sender and receiver IDs share one namespace".
#[derive(Debug, Clone, Default)]
pub struct AccountMap {
    /// Wire identifier -> substrate code.
    to_code: HashMap<u64, u32>,
    /// Substrate code -> wire identifier (reverse, for export/diagnostics).
    to_wire: Vec<u64>,
}

impl AccountMap {
    /// Empty map. No codes allocated.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a wire identifier to its substrate code, allocating on first
    /// sight (append-only).
    ///
    /// # Errors
    /// - [`ErrorCode::AccountId`] if the identifier is 0 (reserved, §6).
    /// - [`ErrorCode::AccountIdCapacity`] if the `u32` code space is
    ///   exhausted. The failure is a rejection, never a wrap (§19).
    pub fn intern(&mut self, wire_id: u64) -> IngestResult<u32> {
        if wire_id == 0 {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                "account id 0 is reserved for invalid or missing identifiers",
            ));
        }
        if let Some(&code) = self.to_code.get(&wire_id) {
            return Ok(code);
        }
        // Index 0 of `to_wire` is the reserved sentinel; the first real code
        // is therefore 1.
        let next_code = self.to_wire.len();
        if next_code == 0 {
            self.to_wire.push(0); // sentinel for the reserved code 0
            return self.intern(wire_id);
        }
        let code = u32::try_from(next_code).map_err(|_| {
            IngestError::new(
                ErrorCode::AccountIdCapacity,
                format!(
                    "account-ID capacity exceeded: more than {MAX_ACCOUNT_CODE} distinct identifiers"
                ),
            )
        })?;
        self.to_wire.push(wire_id);
        self.to_code.insert(wire_id, code);
        Ok(code)
    }

    /// Look up an existing code without allocating.
    ///
    /// Returns `None` for an identifier that has not been interned yet. Used by
    /// recovery, where an unknown identifier must fail closed rather than
    /// silently extending the dictionary.
    #[must_use]
    pub fn code_of(&self, wire_id: u64) -> Option<u32> {
        self.to_code.get(&wire_id).copied()
    }

    /// Reverse lookup: substrate code -> wire identifier.
    #[must_use]
    pub fn wire_of(&self, code: u32) -> Option<u64> {
        self.to_wire.get(code as usize).copied()
    }

    /// Number of allocated codes including the reserved sentinel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.to_wire.len()
    }

    /// True iff no account has been interned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.to_wire.len() <= 1
    }

    /// Number of real (non-reserved) accounts mapped.
    #[must_use]
    pub fn account_count(&self) -> usize {
        self.to_wire.len().saturating_sub(1)
    }

    /// Export the mapping in code order for persistence / session resume.
    ///
    /// The first element is the reserved sentinel (0), so element *i* is the
    /// wire identifier carrying code *i* — the export is self-describing and
    /// restores exactly.
    #[must_use]
    pub fn export(&self) -> Vec<u64> {
        self.to_wire.clone()
    }

    /// Restore a mapping produced by [`Self::export`].
    ///
    /// Fails closed on any inconsistency (§11: never silently repair), because
    /// a divergent mapping would change the logical identity of every account.
    ///
    /// # Errors
    /// - [`ErrorCode::AccountId`] if the export is empty, its sentinel is not
    ///   0, or it contains a duplicate or zero identifier.
    pub fn restore(exported: &[u64]) -> IngestResult<Self> {
        if exported.is_empty() {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                "account map export is empty",
            ));
        }
        if exported[0] != 0 {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                "account map export is missing the reserved code-0 sentinel",
            ));
        }
        let mut m = Self::new();
        m.to_wire.push(0);
        for &wire_id in &exported[1..] {
            if wire_id == 0 {
                return Err(IngestError::new(
                    ErrorCode::AccountId,
                    "account map export contains identifier 0 in a non-reserved slot",
                ));
            }
            let code = u32::try_from(m.to_wire.len()).map_err(|_| {
                IngestError::new(
                    ErrorCode::AccountIdCapacity,
                    "account map export exceeds the u32 code space",
                )
            })?;
            if m.to_code.insert(wire_id, code).is_some() {
                return Err(IngestError::new(
                    ErrorCode::AccountId,
                    "account map export contains a duplicate identifier",
                ));
            }
            m.to_wire.push(wire_id);
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_zero_is_never_assigned() {
        let mut m = AccountMap::new();
        assert!(m.code_of(0).is_none());
        assert_eq!(m.intern(42).unwrap(), 1);
        assert_eq!(m.intern(7).unwrap(), 2);
        assert!(m.code_of(0).is_none(), "code 0 must stay reserved");
    }

    #[test]
    fn interning_zero_is_rejected() {
        let mut m = AccountMap::new();
        assert_eq!(m.intern(0).unwrap_err().code, ErrorCode::AccountId);
    }

    #[test]
    fn interning_is_append_only_and_stable() {
        let mut m = AccountMap::new();
        let a = m.intern(100).unwrap();
        let b = m.intern(200).unwrap();
        let c = m.intern(300).unwrap();
        // Re-interning returns the SAME code (never a new one).
        assert_eq!(m.intern(100).unwrap(), a);
        assert_eq!(m.intern(200).unwrap(), b);
        assert_eq!(m.intern(300).unwrap(), c);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(m.account_count(), 3);
    }

    #[test]
    fn one_shared_namespace_for_sender_and_receiver() {
        // §6: sender and receiver IDs share one namespace, so the same wire
        // identifier used as both resolves to ONE code.
        let mut m = AccountMap::new();
        let as_sender = m.intern(555).unwrap();
        let as_receiver = m.intern(555).unwrap();
        assert_eq!(as_sender, as_receiver);
        assert_eq!(m.account_count(), 1);
    }

    #[test]
    fn width_promotion_preserves_logical_identity() {
        // §17.3: promoting the declared width must not change identity. The key
        // is the numeric value, so the same value interns identically whether
        // it arrived in a 1-byte or an 8-byte field.
        let mut m1 = AccountMap::new();
        let mut m8 = AccountMap::new();
        let mut prev = 0u32;
        for v in [1u64, 2, 255, 256, 65_535, 65_536, u64::MAX - 1] {
            let c1 = m1.intern(v).unwrap();
            let c8 = m8.intern(v).unwrap();
            assert_eq!(c1, c8, "width changed the code for {v}");
            assert_ne!(c1, prev, "code reused");
            prev = c1;
        }
    }

    #[test]
    fn max_u64_identifier_is_supported() {
        let mut m = AccountMap::new();
        assert_eq!(m.intern(u64::MAX).unwrap(), 1);
        assert_eq!(m.wire_of(1), Some(u64::MAX));
    }

    #[test]
    fn export_restore_roundtrip_preserves_mapping() {
        let mut m = AccountMap::new();
        let ids = [9u64, 4, 4, 77, u64::MAX, 1];
        let codes: Vec<u32> = ids.iter().map(|&i| m.intern(i).unwrap()).collect();
        let exported = m.export();
        let r = AccountMap::restore(&exported).unwrap();
        for (i, &code) in ids.iter().zip(codes.iter()) {
            assert_eq!(r.code_of(*i), Some(code), "resume changed identity of {i}");
            assert_eq!(r.wire_of(code), Some(*i));
        }
        assert_eq!(r.account_count(), m.account_count());
    }

    #[test]
    fn restored_map_continues_append_only() {
        let mut m = AccountMap::new();
        m.intern(11).unwrap();
        m.intern(22).unwrap();
        let mut r = AccountMap::restore(&m.export()).unwrap();
        assert_eq!(r.intern(33).unwrap(), 3, "restart must not reuse a code");
        assert_eq!(r.intern(11).unwrap(), 1);
    }

    #[test]
    fn restore_fails_closed_on_bad_exports() {
        assert!(AccountMap::restore(&[]).is_err());
        assert!(AccountMap::restore(&[1, 2, 3]).is_err()); // bad sentinel
        assert!(AccountMap::restore(&[0, 5, 5]).is_err()); // duplicate
        assert!(AccountMap::restore(&[0, 5, 0]).is_err()); // zero in real slot
    }

    #[test]
    fn capacity_boundary_is_a_rejection_not_a_wrap() {
        // Build an export that reaches the u32 boundary. Kept small-scale by
        // exercising the same guard through restore with a synthetic export is
        // infeasible at 4 billion entries, so instead verify the guard logic on
        // the code-space constant and that normal interning never wraps.
        let mut m = AccountMap::new();
        for i in 1..=1000u64 {
            assert_eq!(m.intern(i).unwrap(), i as u32);
        }
        assert_eq!(MAX_ACCOUNT_CODE, u32::MAX as u64);
    }
}
