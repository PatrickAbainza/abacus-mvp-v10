//! Account identifiers: strings in, opaque numeric codes out.
//!
//! Golden Source §6 is the whole spec for this module, and every clause is
//! load-bearing:
//!
//! - "Parse source identifiers as strings. Never parse arbitrary account
//!   identifiers through JavaScript Number."
//! - "Preserve leading zeros and case unless the source explicitly defines
//!   different identifier semantics."
//! - "Empty identifiers and prohibited control characters are rejected."
//! - "Trimming and normalization rules must be explicit. Detect possible
//!   collisions before accepting the mapping."
//! - "IDs must remain stable throughout the applicable dataset or import
//!   namespace."
//! - "Select the width per dataset or immutable segment, never per event."
//! - "If a segment exceeds its selected capacity, stop that segment and report
//!   the failure. Do not wrap IDs or silently change the width mid-segment."
//! - "Resumable imports must retain or reproducibly reconstruct their account
//!   mapping."
//! - "Do not upload the raw account-to-numeric-ID mapping from browser CSV
//!   ingestion."
//!
//! ## Why trimming is a policy, not a default
//!
//! `" 123"` and `"123"` are either the same account or two different accounts,
//! and nothing in the file says which. Choosing one silently would merge or
//! split accounts. So [`IdentifierPolicy::trim`] must be stated, and when
//! trimming is on, two raw strings that fold to the same key are reported as a
//! **collision** rather than quietly merged — that is §6's "Detect possible
//! collisions before accepting the mapping."
//!
//! ## Why the map is never exported
//!
//! The map is the privacy boundary (§6: "Do not upload the raw
//! account-to-numeric-ID mapping"). Codes are assigned in first-seen order, so
//! the mapping is *reproducible* from the same file — which is how a resume
//! reconstructs it without persisting or transmitting it (§6: "Resumable
//! imports must retain or reproducibly reconstruct their account mapping").

use std::collections::HashMap;

use abacus_ingest::config::ACCOUNT_ID_WIDTHS;
use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};

/// Maximum distinct non-zero identifiers each declared width can address (§6).
///
/// ID 0 is reserved for invalid or missing IDs, so a 1-byte width addresses 255
/// accounts, not 256.
#[must_use]
pub const fn capacity_for_width(width: u8) -> Option<u64> {
    match width {
        1 => Some(255),
        2 => Some(65_535),
        4 => Some(4_294_967_295),
        8 => Some(u64::MAX),
        _ => None,
    }
}

/// The explicit identifier policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentifierPolicy {
    /// Whether surrounding ASCII whitespace is stripped before interning.
    ///
    /// §6: "Trimming and normalization rules must be explicit." `false`
    /// preserves the identifier byte-for-byte, leading/trailing spaces included.
    pub trim: bool,
    /// Whether identifiers are compared case-sensitively.
    ///
    /// §6: "Preserve leading zeros and case unless the source explicitly defines
    /// different identifier semantics." So the default and the honest value is
    /// `true`; folding case is an explicit opt-in.
    pub case_sensitive: bool,
    /// Account-ID width in bytes for this dataset (§6: "Select the width per
    /// dataset or immutable segment, never per event").
    pub account_id_width: u8,
}

impl IdentifierPolicy {
    /// Validate the width against the four permitted values (§6, wire §2).
    ///
    /// # Errors
    /// [`ErrorCode::AccountId`] for a width outside `{1, 2, 4, 8}`. The 128-bit
    /// option is explicitly excluded by §6.
    pub fn validate(&self) -> IngestResult<()> {
        if !ACCOUNT_ID_WIDTHS.contains(&self.account_id_width) {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!(
                    "unsupported account_id_width {}; permitted: 1, 2, 4, 8 (128-bit is excluded)",
                    self.account_id_width
                ),
            ));
        }
        Ok(())
    }

    /// Fold a raw identifier to its intern key under this policy.
    ///
    /// Exposed so the preview can detect collisions *before* an upload is
    /// confirmed (§12 step 5: "report ... identifier collisions"), without
    /// mutating the interner.
    #[must_use]
    pub fn key(&self, raw: &str) -> String {
        let s = if self.trim { raw.trim() } else { raw };
        if self.case_sensitive {
            s.to_string()
        } else {
            s.to_ascii_lowercase()
        }
    }

    /// Capacity of the declared width.
    ///
    /// # Errors
    /// [`ErrorCode::AccountId`] for an unsupported width.
    pub fn capacity(&self) -> IngestResult<u64> {
        capacity_for_width(self.account_id_width).ok_or_else(|| {
            IngestError::new(
                ErrorCode::AccountId,
                format!("unsupported account_id_width {}", self.account_id_width),
            )
        })
    }
}

/// The append-only interner: raw identifier string -> opaque wire code.
///
/// Codes start at 1 because §6 reserves ID 0 for invalid or missing IDs. They
/// are assigned in first-seen order and never reassigned, so the same file
/// always produces the same codes — which is what makes a resume (or a retry
/// after a restart) reconstruct the identical mapping without persisting it.
///
/// One interner serves senders **and** receivers, because §6 requires that
/// "sender and receiver IDs share one namespace".
#[derive(Debug, Clone)]
pub struct IdentifierInterner {
    /// Intern key -> wire code.
    codes: HashMap<String, u64>,
    /// The raw spellings behind each code, in first-seen order.
    ///
    /// Held only so a collision can be reported with both spellings and both
    /// source lines; it is never encoded, uploaded or logged.
    raw: Vec<(String, usize)>,
    /// Raw spellings that folded onto an already-seen key, for the preview to
    /// report as a possible collision (§6: "Detect possible collisions before
    /// accepting the mapping").
    folded: Vec<(String, String, usize)>,
    /// The next code to allocate.
    next: u64,
    /// The policy in force.
    policy: IdentifierPolicy,
}

impl IdentifierInterner {
    /// Create an interner under `policy`.
    ///
    /// # Errors
    /// [`ErrorCode::AccountId`] for an unsupported width.
    pub fn new(policy: IdentifierPolicy) -> IngestResult<Self> {
        policy.validate()?;
        Ok(Self {
            codes: HashMap::new(),
            raw: Vec::new(),
            folded: Vec::new(),
            next: 1,
            policy,
        })
    }

    /// Intern one raw identifier, returning its stable opaque code.
    ///
    /// # Errors
    /// - [`ErrorCode::AccountId`] for an empty identifier, or one containing a
    ///   prohibited control character (§6).
    /// - [`ErrorCode::AccountIdCapacity`] when the declared width is full. §6:
    ///   "If a segment exceeds its selected capacity, stop that segment and
    ///   report the failure. Do not wrap IDs or silently change the width
    ///   mid-segment."
    pub fn intern(&mut self, raw: &str, line: usize) -> IngestResult<u64> {
        let key = self.policy.key(raw);
        if key.is_empty() {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!("source line {line} has an empty account identifier"),
            ));
        }
        // §6: "prohibited control characters are rejected."
        if key.chars().any(char::is_control) {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!(
                    "source line {line} has an account identifier containing a control character"
                ),
            ));
        }
        if let Some(&code) = self.codes.get(&key) {
            // §6: "Trimming and normalization rules must be explicit. Detect
            // possible collisions before accepting the mapping." Two raw
            // spellings that fold to one key are *recorded* so the preview can
            // report them, but they are NOT rejected: the fold is the declared
            // policy, so merging them is correct behaviour, not an accident.
            // The caller surfaces the note before the user confirms the upload
            // (§12 step 5), which is what "detect" asks for.
            let first = self
                .raw
                .get(code as usize - 1)
                .cloned()
                .unwrap_or_else(|| (key.clone(), line));
            if first.0 != raw {
                self.folded.push((key.clone(), raw.to_string(), line));
            }
            return Ok(code);
        }
        let cap = self.policy.capacity()?;
        if self.next.saturating_sub(1) >= cap {
            return Err(IngestError::new(
                ErrorCode::AccountIdCapacity,
                format!(
                    "the declared {}-byte account-ID width addresses at most {cap} identifiers; \
                     stop the segment rather than wrapping or widening it",
                    self.policy.account_id_width
                ),
            ));
        }
        let code = self.next;
        self.next += 1;
        self.codes.insert(key.clone(), code);
        self.raw.push((raw.to_string(), line));
        Ok(code)
    }

    /// Number of distinct identifiers interned so far.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.next - 1
    }

    /// True iff nothing has been interned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.next == 1
    }

    /// Raw spellings that folded onto an already-seen key, as
    /// `(key, spelling, source line)`.
    ///
    /// **Privacy:** these are raw source identifier strings, so this exists only
    /// for the local preview and is never serialized into an upload, a report or
    /// a log (§6: "Do not upload the raw account-to-numeric-ID mapping").
    #[must_use]
    pub fn folded_spellings(&self) -> &[(String, String, usize)] {
        &self.folded
    }

    /// The codes assigned, in first-seen order, as `(code, key)` pairs.
    ///
    /// **Privacy:** these are the opaque keys and codes only. This exists so a
    /// caller can assert reproducibility (two runs over the same file yield the
    /// same sequence); it is never serialized into a record, an upload, a report
    /// or a log (§6: "Do not upload the raw account-to-numeric-ID mapping").
    #[must_use]
    pub fn assignment_order(&self) -> Vec<(u64, String)> {
        self.raw
            .iter()
            .enumerate()
            .map(|(i, (k, _))| (i as u64 + 1, k.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(width: u8) -> IdentifierPolicy {
        IdentifierPolicy {
            trim: false,
            case_sensitive: true,
            account_id_width: width,
        }
    }

    #[test]
    fn codes_start_at_one_and_are_stable() {
        let mut m = IdentifierInterner::new(policy(4)).unwrap();
        assert_eq!(m.intern("A", 2).unwrap(), 1);
        assert_eq!(m.intern("B", 3).unwrap(), 2);
        assert_eq!(m.intern("A", 9).unwrap(), 1, "a repeat must reuse its code");
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn leading_zeros_are_preserved_not_numeric_coerced() {
        let mut m = IdentifierInterner::new(policy(4)).unwrap();
        let a = m.intern("007", 2).unwrap();
        let b = m.intern("7", 2).unwrap();
        assert_ne!(a, b, "leading zeros change the logical identity");
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn case_is_preserved_unless_the_policy_folds_it() {
        let mut m = IdentifierInterner::new(policy(4)).unwrap();
        assert_ne!(m.intern("Abc", 2).unwrap(), m.intern("abc", 3).unwrap());
        let mut folded = IdentifierInterner::new(IdentifierPolicy {
            trim: false,
            case_sensitive: false,
            account_id_width: 4,
        })
        .unwrap();
        assert_eq!(
            folded.intern("Abc", 2).unwrap(),
            folded.intern("ABC", 3).unwrap()
        );
    }

    #[test]
    fn trimming_is_explicit_not_implicit() {
        let mut t = IdentifierInterner::new(IdentifierPolicy {
            trim: true,
            case_sensitive: true,
            account_id_width: 4,
        })
        .unwrap();
        assert_eq!(t.intern(" 1", 2).unwrap(), t.intern("1", 3).unwrap());
        let mut n = IdentifierInterner::new(policy(4)).unwrap();
        assert_ne!(n.intern(" 1", 2).unwrap(), n.intern("1", 3).unwrap());
    }

    #[test]
    fn empty_identifier_is_rejected_with_its_source_line() {
        let mut m = IdentifierInterner::new(policy(4)).unwrap();
        let e = m.intern("", 7).unwrap_err();
        assert_eq!(e.code, ErrorCode::AccountId);
        assert!(e.detail.contains("source line 7"));
    }

    #[test]
    fn whitespace_only_identifier_is_rejected_under_trim() {
        let mut m = IdentifierInterner::new(IdentifierPolicy {
            trim: true,
            case_sensitive: true,
            account_id_width: 4,
        })
        .unwrap();
        assert_eq!(m.intern("   ", 5).unwrap_err().code, ErrorCode::AccountId);
    }

    #[test]
    fn control_characters_are_rejected() {
        let mut m = IdentifierInterner::new(policy(4)).unwrap();
        for bad in ["a\nb", "a\tb", "a\u{0}b"] {
            assert_eq!(m.intern(bad, 4).unwrap_err().code, ErrorCode::AccountId);
        }
    }

    #[test]
    fn capacity_is_a_rejection_never_a_wrap() {
        let mut m = IdentifierInterner::new(policy(1)).unwrap();
        for i in 1..=255u64 {
            m.intern(&i.to_string(), 2).unwrap();
        }
        let e = m.intern("over", 300).unwrap_err();
        assert_eq!(e.code, ErrorCode::AccountIdCapacity);
        assert_eq!(
            m.len(),
            255,
            "no code may be allocated on a capacity failure"
        );
    }

    #[test]
    fn capacity_matches_the_declared_width_table() {
        assert_eq!(capacity_for_width(1), Some(255));
        assert_eq!(capacity_for_width(2), Some(65_535));
        assert_eq!(capacity_for_width(4), Some(4_294_967_295));
        assert_eq!(capacity_for_width(8), Some(u64::MAX));
        assert_eq!(capacity_for_width(3), None);
        assert_eq!(capacity_for_width(16), None, "128-bit is excluded by §6");
    }

    #[test]
    fn width_128_is_rejected() {
        let e = IdentifierInterner::new(policy(16)).unwrap_err();
        assert_eq!(e.code, ErrorCode::AccountId);
    }

    #[test]
    fn assignment_order_is_reproducible_for_the_same_file() {
        let mut a = IdentifierInterner::new(policy(4)).unwrap();
        let mut b = IdentifierInterner::new(policy(4)).unwrap();
        for (i, id) in ["z", "y", "x", "z"].iter().enumerate() {
            a.intern(id, i + 2).unwrap();
            b.intern(id, i + 2).unwrap();
        }
        assert_eq!(a.assignment_order(), b.assignment_order());
        assert_eq!(
            a.assignment_order(),
            vec![
                (1, "z".to_string()),
                (2, "y".to_string()),
                (3, "x".to_string())
            ]
        );
    }

    #[test]
    fn folded_spellings_are_detected_and_reported_not_rejected() {
        // §6: "Detect possible collisions before accepting the mapping." The
        // fold is declared policy, so the row is accepted — but the collision
        // is recorded so the preview can warn the user before confirmation.
        let mut m = IdentifierInterner::new(IdentifierPolicy {
            trim: true,
            case_sensitive: false,
            account_id_width: 4,
        })
        .unwrap();
        assert_eq!(m.intern(" Abc ", 2).unwrap(), 1);
        assert_eq!(m.intern("ABC", 5).unwrap(), 1, "the fold merges them");
        assert_eq!(m.len(), 1, "one logical account");
        let folded = m.folded_spellings();
        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].0, "abc", "the shared key");
        assert_eq!(folded[0].1, "ABC", "the later spelling");
        assert_eq!(folded[0].2, 5, "its source line");
    }

    #[test]
    fn an_exact_repeat_records_no_collision() {
        // Re-delivering the identical spelling is normal (a duplicated row), not
        // a collision: nothing is reported.
        let mut m = IdentifierInterner::new(policy(4)).unwrap();
        m.intern("A", 2).unwrap();
        m.intern("A", 9).unwrap();
        assert!(m.folded_spellings().is_empty());
    }
}
