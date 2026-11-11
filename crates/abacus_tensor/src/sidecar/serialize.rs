//! Row 25 (§B.2) + P11: deterministic binary image of the sidecar.
//!
//! ```text
//! header:  magic[8] = "SIDECAR1" | version u32 = 1 | record_count u64 | checksum u64
//! records (record_count ×, fixed field order):
//!   timestamp u64 | source_row u64 | amount_micro i64 | laundering u8
//!   then 11 length-prefixed UTF-8 strings, in this order:
//!     sender_account, receiver_account, amount_raw, time_raw, date_raw,
//!     payment_currency, received_currency, sender_bank_location,
//!     receiver_bank_location, payment_type, laundering_type
//! ```
//!
//! The checksum is `d_fnv` over the RECORDS section only (header excluded) —
//! the same single realization used by the tensor artifact
//! (`composites::fnv1a64`), so no second checksum implementation exists.
//! Provenance bytes are not tensor representation (P11); little-endian fixed
//! fields mirror the artifact codec (`a_{13}`).

use crate::prims::{write_i64, write_len_str, write_u32, write_u64};
use crate::sidecar::types::LaunderingSidecar;

/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"SIDECAR1";
/// Format version.
pub const FORMAT_VERSION: u32 = 1;
/// Header length: `8 + 4 + 8 + 8`.
pub const HEADER_LEN: usize = 28;

/// `d_fnv` — FNV-1a/64 over sidecar bytes (single realization; §B.1).
pub use crate::composites::fnv1a64 as checksum;

/// Row 25 — serialize a sidecar to its immutable binary image.
///
/// Deterministic: identical records yield byte-identical output.
#[must_use]
pub fn serialize_sidecar(sidecar: &LaunderingSidecar) -> Vec<u8> {
    let mut records: Vec<u8> = Vec::new();
    for r in &sidecar.records {
        write_u64(&mut records, r.timestamp);
        write_u64(&mut records, r.source_row);
        write_i64(&mut records, r.amount_micro);
        records.push(r.laundering);
        for s in [
            &r.sender_account,
            &r.receiver_account,
            &r.amount_raw,
            &r.time_raw,
            &r.date_raw,
            &r.payment_currency,
            &r.received_currency,
            &r.sender_bank_location,
            &r.receiver_bank_location,
            &r.payment_type,
            &r.laundering_type,
        ] {
            write_len_str(&mut records, s);
        }
    }
    let sum = checksum(&records);
    let mut out = Vec::with_capacity(HEADER_LEN + records.len());
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, FORMAT_VERSION);
    write_u64(&mut out, sidecar.records.len() as u64);
    write_u64(&mut out, sum);
    out.extend_from_slice(&records);
    out
}
