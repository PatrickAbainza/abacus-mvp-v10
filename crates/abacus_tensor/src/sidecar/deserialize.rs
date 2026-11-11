//! Row 25 (§B.2): deserialization of the sidecar image.
//!
//! Validation order (parsing safety first, then checksum, then the canonical
//! order invariant — every failure rejects): header length → magic → version →
//! record-count bound → checksum over the records section → record decode
//! (UTF-8 validated here, per `K7`) → `prec_4` strict-order check (`d_inv`).

use crate::composites::{fnv1a64 as checksum, strictly_increasing};
use crate::prims::{read_i64, read_len_str, read_u32, read_u64};
use crate::sidecar::serialize::{FORMAT_VERSION, HEADER_LEN, MAGIC};
use crate::sidecar::types::{LaunderingRecord, LaunderingSidecar};
use crate::TensorError;

/// Maximum accepted record count (bounds a hostile count field).
const MAX_RECORDS: u64 = 100_000_000;
/// Maximum accepted length of one string field.
const MAX_STR: u32 = 10_000_000;

/// Row 25 — deserialize and validate a sidecar image.
///
/// # Errors
/// [`TensorError::Decode`] on any structural, checksum, or order violation.
pub fn deserialize_sidecar(data: &[u8]) -> Result<LaunderingSidecar, TensorError> {
    if data.len() < HEADER_LEN {
        return Err(TensorError::Decode("truncated sidecar header".into()));
    }
    if &data[..8] != MAGIC {
        return Err(TensorError::Decode("bad sidecar magic".into()));
    }
    let mut o = 8usize;
    let version = read_u32(data, &mut o, "sidecar")?;
    if version != FORMAT_VERSION {
        return Err(TensorError::Decode(format!(
            "unsupported sidecar version {version}"
        )));
    }
    let count = read_u64(data, &mut o, "sidecar")?;
    let expect_sum = read_u64(data, &mut o, "sidecar")?;
    if count > MAX_RECORDS {
        return Err(TensorError::Decode("sidecar record count too large".into()));
    }
    if checksum(&data[HEADER_LEN..]) != expect_sum {
        return Err(TensorError::Decode(
            "checksum mismatch [sidecar records]".into(),
        ));
    }
    let mut o = HEADER_LEN;
    let mut records = Vec::with_capacity(count.min(1 << 20) as usize);
    for _ in 0..count {
        let timestamp = read_u64(data, &mut o, "sidecar")?;
        let source_row = read_u64(data, &mut o, "sidecar")?;
        let amount_micro = read_i64(data, &mut o, "sidecar")?;
        if o + 1 > data.len() {
            return Err(TensorError::Decode("truncated sidecar".into()));
        }
        let laundering = data[o];
        o += 1;
        if laundering > 1 {
            return Err(TensorError::Decode(
                format!("sidecar record with non-binary laundering {laundering}")
            ));
        }
        let mut field = || read_len_str(data, &mut o, MAX_STR, "sidecar");
        records.push(LaunderingRecord {
            timestamp,
            source_row,
            amount_micro,
            sender_account: field()?,
            receiver_account: field()?,
            amount_raw: field()?,
            time_raw: field()?,
            date_raw: field()?,
            payment_currency: field()?,
            received_currency: field()?,
            sender_bank_location: field()?,
            receiver_bank_location: field()?,
            payment_type: field()?,
            laundering,
            laundering_type: field()?,
        });
    }
    if o != data.len() {
        return Err(TensorError::Decode("trailing bytes in sidecar".into()));
    }
    // d_inv: prec_4 strict increase (source_row is unique, so duplicates are
    // impossible in a well-formed image).
    if !strictly_increasing(&records) {
        return Err(TensorError::Decode(
            "sidecar records not in canonical order".into(),
        ));
    }
    Ok(LaunderingSidecar::new(records))
}
