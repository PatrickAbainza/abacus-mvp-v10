//! Compressed eval-sidecar codec v3 (`SAMEVL03`, addendum §R5f).
//!
//! Derived, lossless P11 provenance encoding — never read by tensor logic.
//! Extends `SAMEVL02` with delta-zigzag `tx_id` (`tx_id` is a permutation of
//! `{0..n-1}` emitted in core order) and zigzag `amount_micro`. Every field is
//! recovered exactly: `Decode_v3(Encode_v3(S)) = S`. No value is aggregated.
//!
//! ```text
//! header[32]: magic[8]="SAMEVL03" | version u32=3 | record_count u64
//!             | checksum u64 | col_count u32=6
//!             checksum = d_fnv over (MAGIC|version|count|col_count) ++ BODY.
//!             The stored checksum field itself is the checked quantity and is
//!             never part of the cover.
//! body (fixed order):
//!   tx_id        : n * zigzag-delta LEB128
//!   amount_micro : n * zigzag LEB128
//!   labels       : ceil(n/8) bytes  (bit i, LSB-first = laundering of i)
//!   amount_raw   : n * (u32 len + UTF-8)
//!   for each of the 6 dict columns (order below):
//!     dict u64 count + (u32 len + UTF-8)*count ; width u8 in {1,2,4} ; codes width*n
//! ```

use super::eval_sidecar::EvalRecord;
use crate::prims::{read_len_str, read_u32, read_u64, write_len_str, write_u32, write_u64};
use crate::varint::{fnv_cover, read_i64_zigzag, write_i64_zigzag};
use crate::TensorError;
use std::collections::BTreeSet;

/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"SAMEVL03";
/// Format version.
pub const VERSION: u32 = 3;
/// Header length: `8 + 4 + 8 + 8 + 4`.
pub const HEADER_LEN: usize = 32;
/// Number of dictionary-encoded columns (fixed order).
pub const DICT_COLS: usize = 6;
/// Max accepted records.
pub const MAX_RECORDS: u64 = 100_000_000;
/// Max accepted string length.
pub const MAX_STR: u32 = 10_000_000;
/// Max accepted dictionary entries.
pub const MAX_DICT: u64 = 50_000_000;

fn dict_columns(r: &EvalRecord) -> [&str; DICT_COLS] {
    [
        &r.laundering_type,
        &r.payment_currency,
        &r.received_currency,
        &r.sender_bank_location,
        &r.receiver_bank_location,
        &r.payment_type,
    ]
}

fn code_width(len: usize) -> u8 {
    if len <= 256 {
        1
    } else if len <= 65_536 {
        2
    } else {
        4
    }
}

/// Serialize eval records to a `SAMEVL03` image (lossless).
///
/// # Errors
/// [`TensorError::Range`] on an internal code-lookup invariant breach.
pub fn serialize_eval_v3(records: &[EvalRecord]) -> Result<Vec<u8>, TensorError> {
    let n = records.len();
    let mut dicts: Vec<Vec<String>> = Vec::with_capacity(DICT_COLS);
    {
        let mut sets: Vec<BTreeSet<&str>> = (0..DICT_COLS).map(|_| BTreeSet::new()).collect();
        for r in records {
            for (i, s) in dict_columns(r).iter().enumerate() {
                sets[i].insert(s);
            }
        }
        for st in sets {
            dicts.push(st.into_iter().map(|s| s.to_string()).collect());
        }
    }
    let widths: Vec<u8> = dicts.iter().map(|d| code_width(d.len())).collect();
    let mut body: Vec<u8> = Vec::new();
    let mut prev = 0i64;
    for r in records {
        let cur = r.tx_id as i64;
        write_i64_zigzag(&mut body, cur - prev);
        prev = cur;
    }
    for r in records {
        write_i64_zigzag(&mut body, r.amount_micro);
    }
    let mut labels = vec![0u8; n.div_ceil(8)];
    for (i, r) in records.iter().enumerate() {
        if r.laundering != 0 {
            labels[i / 8] |= 1u8 << (i % 8);
        }
    }
    body.extend_from_slice(&labels);
    for r in records {
        write_len_str(&mut body, &r.amount_raw);
    }
    for (ci, dict) in dicts.iter().enumerate() {
        write_u64(&mut body, dict.len() as u64);
        for s in dict {
            write_len_str(&mut body, s);
        }
        body.push(widths[ci]);
        for r in records {
            let col = dict_columns(r)[ci];
            let code = match dict.binary_search_by(|d| d.as_str().cmp(col)) {
                Ok(c) => c as u32,
                Err(_) => {
                    return Err(TensorError::Range(
                        "v3 code lookup failed (internal invariant)".into(),
                    ))
                }
            };
            match widths[ci] {
                1 => body.push(code as u8),
                2 => body.extend_from_slice(&(code as u16).to_le_bytes()),
                _ => body.extend_from_slice(&code.to_le_bytes()),
            }
        }
    }
    let sum = {
        let mut buf: Vec<u8> = Vec::with_capacity(24);
        buf.extend_from_slice(MAGIC);
        write_u32(&mut buf, VERSION);
        write_u64(&mut buf, n as u64);
        write_u32(&mut buf, DICT_COLS as u32);
        fnv_cover(&buf, &body)
    };
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, VERSION);
    write_u64(&mut out, n as u64);
    write_u64(&mut out, sum);
    write_u32(&mut out, DICT_COLS as u32);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Deserialize + validate a compressed `SAMEVL03` image (fail-closed, `R_5`).
///
/// # Errors
/// [`TensorError::Decode`] on structure, checksum, code-range, or trailing bytes.
pub fn deserialize_eval_v3(data: &[u8]) -> Result<Vec<EvalRecord>, TensorError> {
    if data.len() < HEADER_LEN {
        return Err(TensorError::Decode("truncated v3 header".into()));
    }
    if &data[..8] != MAGIC {
        return Err(TensorError::Decode("bad v3 magic".into()));
    }
    let mut o = 8usize;
    let version = read_u32(data, &mut o, "v3 version")?;
    if version != VERSION {
        return Err(TensorError::Decode(format!("unsupported v3 version {version}")));
    }
    let count = read_u64(data, &mut o, "v3 count")?;
    let expect = read_u64(data, &mut o, "v3 checksum")?;
    let cols = read_u32(data, &mut o, "v3 col_count")?;
    if count > MAX_RECORDS {
        return Err(TensorError::Decode("v3 count too large".into()));
    }
    if cols != DICT_COLS as u32 {
        return Err(TensorError::Decode("v3 unexpected column count".into()));
    }
    // Reconstruct the covered prefix exactly: MAGIC|version|count (bytes 0..20)
    // ++ col_count (bytes 28..32). The stored checksum field itself (20..28)
    // is the checked quantity, never part of the cover.
    let mut cov: Vec<u8> = Vec::with_capacity(24);
    cov.extend_from_slice(&data[..20]);
    cov.extend_from_slice(&data[28..HEADER_LEN]);
    let body = &data[HEADER_LEN..];
    if fnv_cover(&cov, body) != expect {
        return Err(TensorError::Decode(
            "checksum mismatch [v3 header+body]".into(),
        ));
    }
    let n = count as usize;
    let mut p = HEADER_LEN;
    let mut tx = Vec::with_capacity(n.min(1 << 20));
    let mut prev = 0i64;
    for _ in 0..n {
        let d = read_i64_zigzag(data, &mut p, "v3 tx_id")?;
        let cur = prev
            .checked_add(d)
            .ok_or_else(|| TensorError::Decode("v3 tx_id overflow".into()))?;
        tx.push(cur as u64);
        prev = cur;
    }
    let mut amt = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        amt.push(read_i64_zigzag(data, &mut p, "v3 amount")?);
    }
    let lbl_len = n.div_ceil(8);
    if p + lbl_len > data.len() {
        return Err(TensorError::Decode("truncated v3 labels".into()));
    }
    let labels = &data[p..p + lbl_len];
    p += lbl_len;
    let mut raw = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        raw.push(read_len_str(data, &mut p, MAX_STR, "v3 amount_raw")?);
    }
    let mut col_vals: Vec<Vec<String>> = Vec::with_capacity(DICT_COLS);
    for _ in 0..DICT_COLS {
        let m = read_u64(data, &mut p, "v3 dict_len")?;
        if m > MAX_DICT {
            return Err(TensorError::Decode("v3 dict too large".into()));
        }
        let mut dict = Vec::with_capacity((m.min(1 << 20)) as usize);
        for _ in 0..m {
            dict.push(read_len_str(data, &mut p, MAX_STR, "v3 dict entry")?);
        }
        if p >= data.len() {
            return Err(TensorError::Decode("truncated v3 code width".into()));
        }
        let w = data[p] as usize;
        p += 1;
        if !matches!(w, 1 | 2 | 4) {
            return Err(TensorError::Decode(format!("v3 bad code width {w}")));
        }
        let need = n
            .checked_mul(w)
            .ok_or_else(|| TensorError::Decode("v3 codes length overflow".into()))?;
        if p.checked_add(need).is_none_or(|end| end > data.len()) {
            return Err(TensorError::Decode("truncated v3 codes".into()));
        }
        let mut vals = Vec::with_capacity(n.min(1 << 20));
        for i in 0..n {
            let base = p + i * w;
            let code = match w {
                1 => data[base] as usize,
                2 => {
                    let mut b = [0u8; 2];
                    b.copy_from_slice(&data[base..base + 2]);
                    u16::from_le_bytes(b) as usize
                }
                _ => {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[base..base + 4]);
                    u32::from_le_bytes(b) as usize
                }
            };
            let s = dict
                .get(code)
                .ok_or_else(|| TensorError::Decode("v3 code out of range".into()))?;
            vals.push(s.clone());
        }
        p += need;
        col_vals.push(vals);
    }
    if p != data.len() {
        return Err(TensorError::Decode("trailing bytes in v3".into()));
    }
    let mut out = Vec::with_capacity(n.min(1 << 20));
    for i in 0..n {
        let laundering = u8::from((labels[i / 8] >> (i % 8)) & 1 == 1);
        out.push(EvalRecord {
            tx_id: tx[i],
            amount_micro: amt[i],
            laundering,
            amount_raw: raw[i].clone(),
            laundering_type: col_vals[0][i].clone(),
            payment_currency: col_vals[1][i].clone(),
            received_currency: col_vals[2][i].clone(),
            sender_bank_location: col_vals[3][i].clone(),
            receiver_bank_location: col_vals[4][i].clone(),
            payment_type: col_vals[5][i].clone(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samld::{convert_reader, deserialize_eval};

    fn fixture_records() -> Vec<EvalRecord> {
        // NOTE: see `core_pack` fixture comment — liter trail kept out of the
        // library source body by loading through `std::fs::read`.
        let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/small.csv");
        let csv = std::fs::read(csv_path).expect("fixture csv");
        let conv = convert_reader(csv.as_slice()).unwrap();
        deserialize_eval(&conv.eval_bytes).unwrap()
    }

    #[test]
    fn eval3_roundtrips_losslessly_and_deterministically() {
        let recs = fixture_records();
        let v3 = serialize_eval_v3(&recs).unwrap();
        assert_eq!(deserialize_eval_v3(&v3).unwrap(), recs);
        assert_eq!(serialize_eval_v3(&recs).unwrap(), v3); // determinism (R7)
    }

    #[test]
    fn eval3_fails_closed_on_corruption() {
        let recs = fixture_records();
        let mut v3 = serialize_eval_v3(&recs).unwrap();
        let mid = v3.len() / 2;
        v3[mid] ^= 0xFF;
        assert!(deserialize_eval_v3(&v3).is_err());
    }
}

