//! A complete validated segment: 176-byte header + fixed-width payload.
//!
//! This module owns wire §6 steps 6 through 9, which need the payload bytes
//! and therefore cannot live in the header:
//!
//! 6. "Validate record length and payload length using checked arithmetic."
//! 7. "Verify the actual payload length and SHA-256 checksum."
//! 8. "Validate every record's IDs, feature fields, and reserved bits."
//! 9. "Apply event identity and duplicate/conflict checks."
//!
//! Step 10 is substrate application, which is the session/apply layer's job.
//! The ordering is load-bearing: §6 ends with "No record may be applied before
//! the required segment validation succeeds", and `decode` returns records only
//! after every structural check has passed.

use crate::config::{MAX_PAYLOAD_BYTES, MAX_RECORDS};
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::hash::{digest_eq, sha256};
use crate::record::{record_length_for_width, CanonicalRecord};
use crate::wire::header::SegmentHeader;

/// One decoded segment: its validated header plus every validated record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// The validated 176-byte header.
    pub header: SegmentHeader,
    /// Every record, in wire order, each carrying its compound identity.
    pub records: Vec<CanonicalRecord>,
}

impl Segment {
    /// The single namespace shared by every record in this segment (§2).
    #[must_use]
    pub fn namespace(&self) -> crate::namespace::EventNamespace {
        self.header.event_namespace
    }

    /// Decode a whole segment (`header || payload`) with full validation.
    ///
    /// Validation order follows wire §6: header first (its own checksum), then
    /// the checked length invariants, then the payload checksum, then every
    /// record. Nothing is returned unless all of it passes.
    ///
    /// # Errors
    /// - Any header error from [`SegmentHeader::decode`].
    /// - [`ErrorCode::Binary`] for a length/checksum/record violation.
    /// - [`ErrorCode::AccountId`] if `record_length` disagrees with the
    ///   declared account-ID width.
    pub fn decode(bytes: &[u8]) -> IngestResult<Self> {
        let header = SegmentHeader::decode(bytes)?;
        let payload = &bytes[crate::config::HEADER_LEN..];

        // Step 6: record length must match the declared width.
        let expected_len = record_length_for_width(header.account_id_width)?;
        if header.record_length as usize != expected_len {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!(
                    "record_length is {} but a {}-byte account-ID width requires {expected_len}",
                    header.record_length, header.account_id_width
                ),
            ));
        }

        // Step 6: checked arithmetic for the length invariants (wire §3).
        if header.record_count > MAX_RECORDS {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("record_count {} exceeds the accepted bound {MAX_RECORDS}", header.record_count),
            ));
        }
        if header.payload_length > MAX_PAYLOAD_BYTES {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("payload_length {} exceeds the accepted bound {MAX_PAYLOAD_BYTES}", header.payload_length),
            ));
        }
        let n = usize::try_from(header.record_count).map_err(|_| {
            IngestError::new(ErrorCode::Binary, "record_count is not representable")
        })?;
        let rl = header.record_length as usize;
        let expected_payload = n
            .checked_mul(rl)
            .ok_or_else(|| IngestError::new(
                ErrorCode::Binary,
                "record_count * record_length overflowed (checked arithmetic)",
            ))?;
        if u64::try_from(expected_payload) != Ok(header.payload_length) {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!(
                    "payload_length is {} but record_count * record_length = {expected_payload}",
                    header.payload_length
                ),
            ));
        }

        // Step 7: actual payload length and SHA-256 (wire §4: payload only).
        if payload.len() as u64 != header.payload_length {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!(
                    "actual payload is {} bytes but payload_length declares {}",
                    payload.len(),
                    header.payload_length
                ),
            ));
        }
        let payload_digest = sha256(payload);
        if !digest_eq(&payload_digest, &header.payload_checksum) {
            return Err(IngestError::new(
                ErrorCode::Binary,
                "payload_checksum mismatch",
            ));
        }

        // Step 8: every record. Wire §3: "No partial trailing record is allowed"
        // — guaranteed by the exact payload_length equality above, and asserted
        // again per record by `CanonicalRecord::decode`'s length check.
        let ns = header.event_namespace;
        let mut records = Vec::with_capacity(n.min(1 << 16));
        let mut o = 0usize;
        for i in 0..n {
            let end = o.checked_add(rl).ok_or_else(|| {
                IngestError::new(ErrorCode::Binary, "record offset overflow")
            })?;
            let slice = payload.get(o..end).ok_or_else(|| {
                IngestError::new(
                    ErrorCode::Binary,
                    format!("record {i} is truncated (partial trailing record)"),
                )
            })?;
            records.push(CanonicalRecord::decode(slice, ns, header.account_id_width)?);
            o = end;
        }

        Ok(Self { header, records })
    }

    /// Encode this segment back to `header || payload` bytes.
    ///
    /// Recomputes the payload checksum and header checksum from the emitted
    /// bytes, so an encode/decode round-trip is exact.
    ///
    /// # Errors
    /// [`ErrorCode::AccountId`] if a record does not fit the declared width.
    pub fn encode(&self) -> IngestResult<Vec<u8>> {
        let mut payload = Vec::with_capacity(self.records.len() * self.header.record_length as usize);
        for (i, r) in self.records.iter().enumerate() {
            let b = r.encode(self.header.account_id_width)?;
            if b.len() != self.header.record_length as usize {
                return Err(IngestError::new(
                    ErrorCode::Binary,
                    format!("record {i} encoded to the wrong length"),
                ));
            }
            payload.extend_from_slice(&b);
        }
        let mut header = self.header.clone();
        header.payload_length = payload.len() as u64;
        header.record_count = self.records.len() as u64;
        header.payload_checksum = sha256(&payload);
        let hb = header.encode();
        let mut out = Vec::with_capacity(hb.len() + payload.len());
        out.extend_from_slice(&hb);
        out.extend_from_slice(&payload);
        Ok(out)
    }
}
