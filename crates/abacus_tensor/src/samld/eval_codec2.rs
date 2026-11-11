//! Compressed eval-sidecar codec (`SAMEVL02`).
//!
//! Addendum §R5d: a **derived, lossless** column encoding of the `SAMEVL01`
//! eval sidecar. P11 provenance only — never read by tensor logic, never a
//! canonical representation. Reuses only existing primitives: the ordered-set
//! dictionary is `a_{16}.2` (`prec_dict` ascending), fixed-order emission is
//! `a_{13}.1`, the checksum is `d_fnv` (`a_{06}`), and bit-packing is
//! `a_{04}`/`a_{16}`. No new mathematics; the external surface stays std.
//!
//! ```text
//! header: magic[8]="SAMEVL02" | version u32=2 | record_count u64
//!         | checksum u64 (FNV-1a/64 over BODY) | col_count u32=6
//! body (fixed order):
//!   tx_id        : u64 * n
//!   amount_micro : i64 * n
//!   labels       : ceil(n/8) bytes  (bit i, LSB-first, = laundering of rec i)
//!   amount_raw   : n * (u32 len + UTF-8)
//!   for each of the 6 dict columns (order below):
//!     dict   : u64 count + (u32 len + UTF-8) * count   (prec_dict ascending)
//!     width  : u8  in {1,2,4}
//!     codes  : width * n  (little-endian code into that dict)
//! ```

use super::eval_sidecar::EvalRecord;
use crate::composites::fnv1a64;
use crate::prims::{
    read_i64, read_len_str, read_u32, read_u64, write_i64, write_len_str, write_u32, write_u64,
};
use crate::TensorError;
use std::collections::BTreeSet;

/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"SAMEVL02";
/// Format version.
pub const VERSION: u32 = 2;
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

/// The six dictionary-encoded columns, in `SAMEVL01` field order.
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

/// Minimal deterministic code width for a dictionary of `len` entries.
fn code_width(len: usize) -> u8 {
    if len <= 256 {
        1
    } else if len <= 65_536 {
        2
    } else {
        4
    }
}

/// Serialize eval records to the compressed `SAMEVL02` image (lossless).
///
/// # Errors
/// [`TensorError::Range`] on an internal code-lookup invariant breach.
pub fn serialize_eval_v2(records: &[EvalRecord]) -> Result<Vec<u8>, TensorError> {
    let n = records.len();
    // 1. Ordered-set dictionaries (a_{16}.2: ascending byte order == code order).
    let mut dicts: Vec<Vec<String>> = Vec::with_capacity(DICT_COLS);
    {
        let mut sets: Vec<BTreeSet<&str>> = (0..DICT_COLS).map(|_| BTreeSet::new()).collect();
        for r in records {
            let cols = dict_columns(r);
            for (i, s) in cols.iter().enumerate() {
                sets[i].insert(s);
            }
        }
        for st in sets {
            dicts.push(st.into_iter().map(|s| s.to_string()).collect());
        }
    }
    let widths: Vec<u8> = dicts.iter().map(|d| code_width(d.len())).collect();
    let mut body: Vec<u8> = Vec::new();
    for r in records {
        write_u64(&mut body, r.tx_id);
    }
    for r in records {
        write_i64(&mut body, r.amount_micro);
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
                        "v2 code lookup failed (internal invariant)".into(),
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
    let sum = fnv1a64(&body);
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, VERSION);
    write_u64(&mut out, n as u64);
    write_u64(&mut out, sum);
    write_u32(&mut out, DICT_COLS as u32);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Deserialize + validate a compressed `SAMEVL02` image (fail-closed, `R_5`).
///
/// # Errors
/// [`TensorError::Decode`] on any structural, checksum, code-range, or
/// trailing-byte violation.
pub fn deserialize_eval_v2(data: &[u8]) -> Result<Vec<EvalRecord>, TensorError> {
    if data.len() < HEADER_LEN {
        return Err(TensorError::Decode("truncated v2 header".into()));
    }
    if &data[..8] != MAGIC {
        return Err(TensorError::Decode("bad v2 magic".into()));
    }
    let mut o = 8usize;
    let version = read_u32(data, &mut o, "v2 version")?;
    if version != VERSION {
        return Err(TensorError::Decode(format!("unsupported v2 version {version}")));
    }
    let count = read_u64(data, &mut o, "v2 count")?;
    let expect = read_u64(data, &mut o, "v2 checksum")?;
    let cols = read_u32(data, &mut o, "v2 col_count")?;
    if count > MAX_RECORDS {
        return Err(TensorError::Decode("v2 count too large".into()));
    }
    if cols != DICT_COLS as u32 {
        return Err(TensorError::Decode("v2 unexpected column count".into()));
    }
    if fnv1a64(&data[HEADER_LEN..]) != expect {
        return Err(TensorError::Decode("checksum mismatch [v2 body]".into()));
    }
    let n = count as usize;
    let mut p = HEADER_LEN;
    let mut tx = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        tx.push(read_u64(data, &mut p, "v2 tx_id")?);
    }
    let mut amt = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        amt.push(read_i64(data, &mut p, "v2 amount")?);
    }
    let lbl_len = n.div_ceil(8);
    if p + lbl_len > data.len() {
        return Err(TensorError::Decode("truncated v2 labels".into()));
    }
    let labels = &data[p..p + lbl_len];
    p += lbl_len;
    let mut raw = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        raw.push(read_len_str(data, &mut p, MAX_STR, "v2 amount_raw")?);
    }
    let mut col_vals: Vec<Vec<String>> = Vec::with_capacity(DICT_COLS);
    for _ in 0..DICT_COLS {
        let m = read_u64(data, &mut p, "v2 dict_len")?;
        if m > MAX_DICT {
            return Err(TensorError::Decode("v2 dict too large".into()));
        }
        let mut dict = Vec::with_capacity((m.min(1 << 20)) as usize);
        for _ in 0..m {
            dict.push(read_len_str(data, &mut p, MAX_STR, "v2 dict entry")?);
        }
        if p >= data.len() {
            return Err(TensorError::Decode("truncated v2 code width".into()));
        }
        let w = data[p] as usize;
        p += 1;
        if !matches!(w, 1 | 2 | 4) {
            return Err(TensorError::Decode(format!("v2 bad code width {w}")));
        }
        let need = n
            .checked_mul(w)
            .ok_or_else(|| TensorError::Decode("v2 codes length overflow".into()))?;
        if p.checked_add(need).is_none_or(|end| end > data.len()) {
            return Err(TensorError::Decode("truncated v2 codes".into()));
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
                .ok_or_else(|| TensorError::Decode("v2 code out of range".into()))?;
            vals.push(s.clone());
        }
        p += need;
        col_vals.push(vals);
    }
    if p != data.len() {
        return Err(TensorError::Decode("trailing bytes in v2".into()));
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

