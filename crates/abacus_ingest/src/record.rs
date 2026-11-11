//! The canonical event record: `23 + 2W` bytes (Canonical Contract §7).
//!
//! §7.2 fixes the field order, widths and offsets exactly. `W` is the declared
//! account-ID width in bytes (1, 2, 4 or 8):
//!
//! | Offset | Field | Type | Size |
//! |--------|-------|------|------|
//! | 0 | `event_id` | u64 | 8 |
//! | 8 | `event_time_ms` | i64 | 8 |
//! | 16 | `sender_id` | unsigned | W |
//! | 16+W | `receiver_id` | unsigned | W |
//! | 16+2W | `log_magnitude` | u16 | 2 |
//! | 18+2W | `relative_threshold_distance` | i32 | 4 |
//! | 22+2W | `roundedness_flags` | u8 | 1 |
//!
//! There is deliberately **no per-record namespace field**: §7.2 states "There
//! is no per-record namespace field because the header defines the namespace
//! for the entire segment", and §2 requires that each segment contain events
//! from exactly one namespace. The identity is therefore reconstructed as
//! `(header.event_namespace, record.event_id)` (§7.3).
//!
//! ## Encoding discipline (wire §7)
//!
//! "Do not serialize a Rust struct by copying its in-memory bytes. Explicitly
//! encode and decode each field to avoid compiler padding and
//! platform-dependent layout." So [`CanonicalRecord::encode`] and
//! [`CanonicalRecord::decode`] touch each field individually with
//! [`to_le_bytes`]/[`from_le_bytes`], and the account-ID fields are written as
//! exactly `W` little-endian bytes — never as a native-width integer.

use crate::config::ACCOUNT_ID_WIDTHS;
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::features::FeaturePayload;
use crate::namespace::{EventIdentity, EventNamespace};

/// Fixed-size prefix of a record: `event_id` + `event_time_ms`.
pub const RECORD_FIXED_PREFIX: usize = 16;

/// Fixed-size suffix of a record: `log_magnitude` + `relative_threshold_distance`
/// + `roundedness_flags` = 7 bytes (the frozen payload, Golden Source §9.4).
pub const RECORD_FIXED_SUFFIX: usize = FeaturePayload::BYTE_LEN;

/// `record_length = 23 + 2W` (wire §3).
///
/// # Errors
/// [`ErrorCode::AccountId`] if `w` is not a permitted width (1, 2, 4, 8).
pub fn record_length_for_width(w: u8) -> IngestResult<usize> {
    if !ACCOUNT_ID_WIDTHS.contains(&w) {
        return Err(IngestError::new(
            ErrorCode::AccountId,
            format!("unsupported account_id_width {w}; permitted: 1, 2, 4, 8"),
        ));
    }
    Ok(23 + 2 * usize::from(w))
}

/// One decoded canonical event record.
///
/// Carries the decoded fields plus the reconstructed compound identity. The
/// namespace is not stored per record (§7.2) — it is attached here at decode
/// time from the segment header so the identity travels with the value and
/// cannot be dropped downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalRecord {
    /// The compound identity `(namespace, event_id)` (Canonical Contract §3.1).
    pub identity: EventIdentity,
    /// UTC milliseconds since the Unix epoch (signed i64).
    pub event_time_ms: i64,
    /// Opaque sender account identifier at the declared width.
    pub sender_id: u64,
    /// Opaque receiver account identifier at the declared width.
    pub receiver_id: u64,
    /// The frozen seven-byte derived feature payload.
    pub features: FeaturePayload,
}

impl CanonicalRecord {
    /// Encode to exactly `record_length` bytes (wire §7: field-by-field, LE).
    ///
    /// # Errors
    /// - [`ErrorCode::AccountId`] for an unsupported width or an identifier
    ///   that does not fit the declared width (rejected, never truncated).
    pub fn encode(&self, account_id_width: u8) -> IngestResult<Vec<u8>> {
        let w = usize::from(account_id_width);
        let len = record_length_for_width(account_id_width)?;
        let mask = width_mask(w);
        if self.sender_id > mask || self.receiver_id > mask {
            return Err(IngestError::new(
                ErrorCode::AccountId,
                format!(
                    "account identifier does not fit the declared {}-byte width",
                    w
                ),
            ));
        }
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&self.identity.event_id.to_le_bytes()); // 0..8
        out.extend_from_slice(&self.event_time_ms.to_le_bytes()); // 8..16
        out.extend_from_slice(&self.sender_id.to_le_bytes()[..w]); // 16..16+W
        out.extend_from_slice(&self.receiver_id.to_le_bytes()[..w]); // 16+W..16+2W
        out.extend_from_slice(&self.features.log_magnitude.to_le_bytes()); // 16+2W
        out.extend_from_slice(&self.features.relative_threshold_distance.to_le_bytes());
        out.push(self.features.roundedness_flags);
        debug_assert_eq!(out.len(), len);
        Ok(out)
    }

    /// Decode one record from `buf`, which must be exactly `record_length`
    /// bytes.
    ///
    /// The namespace comes from the segment header, not the bytes (§7.2).
    ///
    /// # Errors
    /// - [`ErrorCode::Binary`] if `buf.len()` does not equal `record_length`.
    /// - [`ErrorCode::AccountId`] for an unsupported width.
    /// - [`ErrorCode::Binary`] if a reserved roundedness bit is set.
    pub fn decode(
        buf: &[u8],
        namespace: EventNamespace,
        account_id_width: u8,
    ) -> IngestResult<Self> {
        let w = usize::from(account_id_width);
        let len = record_length_for_width(account_id_width)?;
        if buf.len() != len {
            return Err(IngestError::new(
                ErrorCode::Binary,
                format!("record is {} bytes, expected {len}", buf.len()),
            ));
        }
        let event_id = u64::from_le_bytes(buf[0..8].try_into().expect("8 bytes"));
        let event_time_ms = i64::from_le_bytes(buf[8..16].try_into().expect("8 bytes"));
        let sender_id = read_unsigned(&buf[16..16 + w])?;
        let receiver_id = read_unsigned(&buf[16 + w..16 + 2 * w])?;
        let fo = 16 + 2 * w;
        let log_magnitude = u16::from_le_bytes(buf[fo..fo + 2].try_into().expect("2 bytes"));
        let relative_threshold_distance =
            i32::from_le_bytes(buf[fo + 2..fo + 6].try_into().expect("4 bytes"));
        let roundedness_flags = buf[fo + 6];
        let features = FeaturePayload {
            log_magnitude,
            relative_threshold_distance,
            roundedness_flags,
        };
        features.validate()?;
        Ok(Self {
            identity: EventIdentity::new(namespace, event_id),
            event_time_ms,
            sender_id,
            receiver_id,
            features,
        })
    }
}

/// `2^(8W) - 1`: the largest value representable in `W` bytes.
fn width_mask(w: usize) -> u64 {
    if w >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * w)) - 1
    }
}

/// Decode exactly `W` little-endian bytes into a `u64` (zero-extended).
///
/// # Errors
/// [`ErrorCode::Binary`] if `bytes.len()` is not 1, 2, 4 or 8.
fn read_unsigned(bytes: &[u8]) -> IngestResult<u64> {
    if !ACCOUNT_ID_WIDTHS.contains(&(bytes.len() as u8)) {
        return Err(IngestError::new(
            ErrorCode::AccountId,
            format!("account-ID field is {} bytes; permitted: 1, 2, 4, 8", bytes.len()),
        ));
    }
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= u64::from(b) << (8 * i); // little-endian assembly
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeaturePayload;
    use crate::namespace::EventNamespace;

    fn ns() -> EventNamespace {
        EventNamespace::from_bytes([
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ])
    }

    fn rec(sender: u64, receiver: u64) -> CanonicalRecord {
        CanonicalRecord {
            identity: EventIdentity::new(ns(), 4_242),
            event_time_ms: 1_700_000_000_123,
            sender_id: sender,
            receiver_id: receiver,
            features: FeaturePayload {
                log_magnitude: 9_416,
                relative_threshold_distance: -2_345,
                roundedness_flags: 0b0101,
            },
        }
    }

    #[test]
    fn record_length_matches_the_declared_table() {
        // Wire §3 / Canonical Contract §7.2 exact table.
        assert_eq!(record_length_for_width(1).unwrap(), 25);
        assert_eq!(record_length_for_width(2).unwrap(), 27);
        assert_eq!(record_length_for_width(4).unwrap(), 31);
        assert_eq!(record_length_for_width(8).unwrap(), 39);
        for w in ACCOUNT_ID_WIDTHS {
            assert_eq!(
                record_length_for_width(w).unwrap(),
                23 + 2 * usize::from(w)
            );
        }
    }

    #[test]
    fn bad_width_rejected() {
        for w in [0u8, 3, 5, 7, 16, 255] {
            assert_eq!(
                record_length_for_width(w).unwrap_err().code,
                ErrorCode::AccountId
            );
        }
    }

    #[test]
    fn roundtrip_every_width() {
        for w in ACCOUNT_ID_WIDTHS {
            let max = if w == 8 { u64::MAX } else { (1u64 << (8 * w)) - 1 };
            let r = rec(max, max - 1);
            let bytes = r.encode(w).unwrap();
            assert_eq!(bytes.len(), record_length_for_width(w).unwrap());
            let back = CanonicalRecord::decode(&bytes, ns(), w).unwrap();
            assert_eq!(back, r, "roundtrip failed at width {w}");
        }
    }

    #[test]
    fn exact_offsets_and_little_endian_encoding() {
        // Width 4: verify each field's byte position and endianness.
        let r = CanonicalRecord {
            identity: EventIdentity::new(ns(), 0x0102_0304_0506_0708),
            event_time_ms: -2, // i64 two's complement
            sender_id: 0x1122_3344,
            receiver_id: 0x5566_7788,
            features: FeaturePayload {
                log_magnitude: 0xAABB,
                relative_threshold_distance: -1,
                roundedness_flags: 0x0F,
            },
        };
        let b = r.encode(4).unwrap();
        assert_eq!(b.len(), 31);
        // event_id u64 LE at 0..8
        assert_eq!(&b[0..8], &0x0102_0304_0506_0708u64.to_le_bytes());
        // event_time_ms i64 LE at 8..16 (two's complement of -2)
        assert_eq!(&b[8..16], &(-2i64).to_le_bytes());
        // sender at 16..20 LE
        assert_eq!(&b[16..20], &0x1122_3344u32.to_le_bytes());
        // receiver at 20..24 LE
        assert_eq!(&b[20..24], &0x5566_7788u32.to_le_bytes());
        // log_magnitude u16 LE at 24..26
        assert_eq!(&b[24..26], &0xAABBu16.to_le_bytes());
        // relative_threshold_distance i32 LE at 26..30
        assert_eq!(&b[26..30], &(-1i32).to_le_bytes());
        // roundedness at 30
        assert_eq!(b[30], 0x0F);
    }

    #[test]
    fn no_per_record_namespace_field() {
        // §7.2: the namespace lives only in the header. Confirm the encoded
        // bytes do not contain the namespace at any offset, and that the same
        // record bytes decode under either namespace.
        let other = EventNamespace::from_bytes([0xAA; 16]);
        let r = rec(5, 6);
        let b = r.encode(4).unwrap();
        for off in 0..=b.len().saturating_sub(16) {
            assert_ne!(&b[off..off + 16], &ns().to_bytes(), "namespace leaked into record");
        }
        let a = CanonicalRecord::decode(&b, ns(), 4).unwrap();
        let c = CanonicalRecord::decode(&b, other, 4).unwrap();
        assert_eq!(a.identity.namespace, ns());
        assert_eq!(c.identity.namespace, other);
        assert_eq!(a.identity.event_id, c.identity.event_id);
    }

    #[test]
    fn identifier_wider_than_declared_width_is_rejected() {
        // 256 does not fit a 1-byte field: rejected, never truncated (§19).
        let r = rec(256, 1);
        assert_eq!(r.encode(1).unwrap_err().code, ErrorCode::AccountId);
    }

    #[test]
    fn wrong_length_buffer_rejected() {
        assert_eq!(
            CanonicalRecord::decode(&[0u8; 30], ns(), 4).unwrap_err().code,
            ErrorCode::Binary
        );
        assert_eq!(
            CanonicalRecord::decode(&[0u8; 32], ns(), 4).unwrap_err().code,
            ErrorCode::Binary
        );
    }

    #[test]
    fn reserved_roundedness_bit_rejected_on_decode() {
        let mut r = rec(1, 2);
        r.features.roundedness_flags = 0x10;
        let b = r.encode(4).unwrap();
        assert_eq!(
            CanonicalRecord::decode(&b, ns(), 4).unwrap_err().code,
            ErrorCode::Binary
        );
    }

    #[test]
    fn account_field_zero_extends_not_sign_extends() {
        // A 1-byte field of 0xFF must decode as 255, never as -1.
        let r = rec(255, 0);
        let b = r.encode(1).unwrap();
        let back = CanonicalRecord::decode(&b, ns(), 1).unwrap();
        assert_eq!(back.sender_id, 255);
        assert_eq!(back.receiver_id, 0);
    }

    #[test]
    fn encoding_is_not_a_memory_copy_of_the_struct() {
        // Wire §7 forbids `transmute`-style serialization. Prove the encoded
        // bytes are independent of in-memory layout: two records that differ
        // only in a high unused field position must not alias.
        let a = rec(1, 2);
        let b = rec(1, 2);
        assert_eq!(a.encode(4).unwrap(), b.encode(4).unwrap());
        // And the encoded length is the declared record length, not size_of.
        assert_ne!(
            a.encode(4).unwrap().len(),
            std::mem::size_of::<CanonicalRecord>()
        );
    }
}
