//! The resume token: the minimum durable state a retry needs.
//!
//! Golden Source §13: "Interrupted sessions have explicit resume or abort
//! semantics" and "Session resumption must preserve account mappings and
//! source-row identity." §3.2 adds: "Reuse it when resuming or retrying that
//! same logical import or stream. Never regenerate it during a retry."
//!
//! ## What is in it, and what deliberately is not
//!
//! The token carries the session id, the event namespace, the configuration
//! fingerprint and the chunk sequence already acknowledged. That is enough for a
//! retry to re-attach to the *same* session and the *same* identity domain, so
//! the backend's CAS recognises re-delivered events as duplicates (§5.1) instead
//! of applying them twice.
//!
//! It deliberately does **not** carry:
//!
//! - **The account mapping.** §6: "Resumable imports must retain or
//!   reproducibly reconstruct their account mapping" — and reconstructing it is
//!   the safe half of that, because codes are assigned in first-seen order from
//!   the same file. Storing it would mean persisting the raw-to-code map, which
//!   §6 forbids uploading and which nothing here needs.
//! - **Any amount.** Amounts are re-derived from the same file on retry and are
//!   never written to durable state.
//! - **The normalized dataset.** The file is re-read; nothing is cached.
//!
//! So a resume is "re-derive everything deterministically, then re-attach",
//! which is why it is correct after a crash and why it cannot leak.

use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};
use abacus_ingest::namespace::EventNamespace;

/// The durable state a retry needs to re-attach to an existing session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeToken {
    /// The backend session id (`sess-N`).
    pub session_id: String,
    /// The event namespace bound to that session (§3.2: never regenerate).
    pub namespace: EventNamespace,
    /// The accepted configuration fingerprint, hex.
    ///
    /// A retry whose plan hashes differently must not join this session (§13:
    /// "Conflicting configuration fingerprints must not be accepted into the
    /// same logical session without an explicit migration policy").
    pub config_fingerprint: String,
    /// Highest chunk sequence the backend has acknowledged.
    ///
    /// The retry re-uploads from here; identical bytes make each re-delivery
    /// idempotent (§13).
    pub ack_chunks: u64,
    /// Total accepted records the backend acknowledged, for reconciliation.
    pub accepted_records: u64,
}

impl ResumeToken {
    /// Render as a single line of `key=value` fields.
    ///
    /// Deliberately not JSON and not a serde format: it is a tiny, fixed-shape
    /// record a browser can keep in `localStorage`, and a hand format keeps this
    /// crate dependency-free. No field can hold an amount or an account string,
    /// so there is nothing to redact.
    #[must_use]
    pub fn encode(&self) -> String {
        format!(
            "abacus_resume_v1 session_id={} namespace={} fingerprint={} ack_chunks={} accepted={}",
            self.session_id,
            self.namespace.to_hyphenated(),
            self.config_fingerprint,
            self.ack_chunks,
            self.accepted_records
        )
    }

    /// Parse a token produced by [`Self::encode`].
    ///
    /// # Errors
    /// [`ErrorCode::Session`] if the magic, any required field, or a field's
    /// value is malformed. Parsing fails closed: an unrecognised version is an
    /// error, never a partial interpretation.
    pub fn decode(text: &str) -> IngestResult<Self> {
        let mut it = text.split_whitespace();
        if it.next() != Some("abacus_resume_v1") {
            return Err(IngestError::new(
                ErrorCode::Session,
                "the resume token does not carry the abacus_resume_v1 magic",
            ));
        }
        let mut session_id = None;
        let mut namespace = None;
        let mut fingerprint = None;
        let mut ack_chunks = None;
        let mut accepted = None;
        for field in it {
            let (k, v) = field.split_once('=').ok_or_else(|| {
                IngestError::new(
                    ErrorCode::Session,
                    "a resume token field has no '=' separator",
                )
            })?;
            match k {
                "session_id" => session_id = Some(v.to_string()),
                "namespace" => namespace = Some(EventNamespace::parse_hyphenated(v)?),
                "fingerprint" => fingerprint = Some(v.to_string()),
                "ack_chunks" => {
                    ack_chunks = Some(v.parse::<u64>().map_err(|_| {
                        IngestError::new(ErrorCode::Session, "ack_chunks is not a u64")
                    })?)
                }
                "accepted" => {
                    accepted = Some(v.parse::<u64>().map_err(|_| {
                        IngestError::new(ErrorCode::Session, "accepted is not a u64")
                    })?)
                }
                other => {
                    return Err(IngestError::new(
                        ErrorCode::Session,
                        format!("the resume token has an unknown field '{other}'"),
                    ))
                }
            }
        }
        Ok(Self {
            session_id: session_id.ok_or_else(|| {
                IngestError::new(ErrorCode::Session, "the resume token has no session_id")
            })?,
            namespace: namespace.ok_or_else(|| {
                IngestError::new(ErrorCode::Session, "the resume token has no namespace")
            })?,
            config_fingerprint: fingerprint.ok_or_else(|| {
                IngestError::new(ErrorCode::Session, "the resume token has no fingerprint")
            })?,
            ack_chunks: ack_chunks.ok_or_else(|| {
                IngestError::new(ErrorCode::Session, "the resume token has no ack_chunks")
            })?,
            accepted_records: accepted.ok_or_else(|| {
                IngestError::new(ErrorCode::Session, "the resume token has no accepted count")
            })?,
        })
    }

    /// True iff this token may be reused for a plan hashing to `fingerprint`.
    ///
    /// A different fingerprint is a different configuration, and §13 forbids
    /// mixing them into one logical session. Reusing the token then would attach
    /// events to a session whose declared configuration does not describe them.
    #[must_use]
    pub fn matches_fingerprint(&self, fingerprint: &str) -> bool {
        self.config_fingerprint == fingerprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> ResumeToken {
        ResumeToken {
            session_id: "sess-7".to_string(),
            namespace: EventNamespace::from_bytes([0x9a; 16]),
            config_fingerprint: "abcd".to_string(),
            ack_chunks: 3,
            accepted_records: 12,
        }
    }

    #[test]
    fn roundtrips_exactly() {
        let t = token();
        assert_eq!(ResumeToken::decode(&t.encode()).unwrap(), t);
    }

    #[test]
    fn carries_no_amount_and_no_account_string() {
        // The token is the only durable client state, so it must be provably
        // free of the two things A1/§6 forbid persisting.
        let text = token().encode();
        assert!(
            !text.contains("amount"),
            "the token must not mention amounts"
        );
        // magic + session_id + namespace + fingerprint + ack_chunks + accepted.
        assert_eq!(text.split_whitespace().count(), 6);
    }

    #[test]
    fn decode_fails_closed_on_bad_input() {
        for bad in [
            "",
            "not_a_token a=1",
            "abacus_resume_v1 session_id=s",
            "abacus_resume_v2 session_id=s namespace=00000000-0000-0000-0000-000000000000 fingerprint=f ack_chunks=1 accepted=1",
            "abacus_resume_v1 session_id=s namespace=zz fingerprint=f ack_chunks=1 accepted=1",
            "abacus_resume_v1 session_id=s namespace=00000000-0000-0000-0000-000000000000 fingerprint=f ack_chunks=x accepted=1",
            "abacus_resume_v1 session_id=s namespace=00000000-0000-0000-0000-000000000000 fingerprint=f ack_chunks=1 accepted=1 extra=1",
        ] {
            assert!(ResumeToken::decode(bad).is_err(), "'{bad}' must be rejected");
        }
    }

    #[test]
    fn a_different_fingerprint_cannot_reuse_the_token() {
        let t = token();
        assert!(t.matches_fingerprint("abcd"));
        assert!(
            !t.matches_fingerprint("abce"),
            "a changed configuration must not join the same session (§13)"
        );
    }
}
