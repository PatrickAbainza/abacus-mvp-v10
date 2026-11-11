//! Eval-sidecar codec (row-25 shape, P11).

use super::eval_sidecar::*;
use crate::composites::fnv1a64;
use crate::prims::{read_i64, read_len_str, read_u32, read_u64, write_i64, write_len_str, write_u32, write_u64};
use crate::TensorError;

/// Serialize eval records (deterministic `R_7`, `d_fnv` over RECORDS).
#[must_use]
pub fn serialize_eval(records: &[EvalRecord]) -> Vec<u8> {
    let mut rec: Vec<u8> = Vec::new();
    for r in records {
        write_u64(&mut rec, r.tx_id);
        write_i64(&mut rec, r.amount_micro);
        rec.push(r.laundering);
        rec.extend_from_slice(&[0u8; 7]);
        for s in [
            &r.amount_raw, &r.laundering_type, &r.payment_currency,
            &r.received_currency, &r.sender_bank_location,
            &r.receiver_bank_location, &r.payment_type,
        ] {
            write_len_str(&mut rec, s);
        }
    }
    let sum = fnv1a64(&rec);
    let mut out = Vec::with_capacity(HEADER_LEN + rec.len());
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, VERSION);
    write_u64(&mut out, records.len() as u64);
    write_u64(&mut out, sum);
    out.extend_from_slice(&rec);
    out
}

/// Deserialize + validate (fail-closed `R_5`).
///
/// # Errors
/// [`TensorError::Decode`] on any structural, checksum, or order violation.
pub fn deserialize_eval(data: &[u8]) -> Result<Vec<EvalRecord>, TensorError> {
    if data.len() < HEADER_LEN {
        return Err(TensorError::Decode("truncated eval header".into()));
    }
    if &data[..8] != MAGIC {
        return Err(TensorError::Decode("bad eval magic".into()));
    }
    let mut o = 8usize;
    let version = read_u32(data, &mut o, "eval version")?;
    if version != VERSION {
        return Err(TensorError::Decode(format!("unsupported eval version {version}")));
    }
    let count = read_u64(data, &mut o, "eval count")?;
    let expect_sum = read_u64(data, &mut o, "eval checksum")?;
    if count > MAX_RECORDS {
        return Err(TensorError::Decode("eval count too large".into()));
    }
    if fnv1a64(&data[HEADER_LEN..]) != expect_sum {
        return Err(TensorError::Decode("checksum mismatch [eval records]".into()));
    }
    let mut o = HEADER_LEN;
    let mut out = Vec::with_capacity((count.min(1 << 20)) as usize);
    for _ in 0..count {
        let tx_id = read_u64(data, &mut o, "eval tx")?;
        let amount_micro = read_i64(data, &mut o, "eval amount")?;
        if o + 8 > data.len() {
            return Err(TensorError::Decode("truncated eval record".into()));
        }
        let laundering = data[o];
        o += 1;
        if data[o..o + 7] != [0u8; 7] {
            return Err(TensorError::Decode("nonzero eval pad".into()));
        }
        o += 7;
        if laundering > 1 {
            return Err(TensorError::Decode(format!(
                "non-binary laundering {laundering}"
            )));
        }
        let mut field = || read_len_str(data, &mut o, MAX_STR, "eval");
        let r = EvalRecord {
            tx_id, amount_micro, laundering,
            amount_raw: field()?,
            laundering_type: field()?,
            payment_currency: field()?,
            received_currency: field()?,
            sender_bank_location: field()?,
            receiver_bank_location: field()?,
            payment_type: field()?,
        };
        // No intra-file order check: `tx_id` is not an ordering key (§S4).
        // The sidecar is emitted in core `(t,s,r)` order and joined to the
        // core positionally (index bijection) + exact index-set equality.
        out.push(r);
    }
    if o != data.len() {
        return Err(TensorError::Decode("trailing bytes in eval".into()));
    }
    Ok(out)
}
