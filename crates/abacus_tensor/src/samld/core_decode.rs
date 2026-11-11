//! Core binary deserialization (fail-closed `R_5`).

use super::core_bin::{
    read_f64, CoreImage, CoreRecord, MAGIC, FORMAT_VERSION, HEADER_LEN,
    MAX_DICT_ENTRIES, MAX_DICT_STR, MAX_RECORDS, RECORD_LEN,
};
use crate::composites::fnv1a64;
use crate::prims::{read_i64, read_len_str, read_u32, read_u64};
use crate::TensorError;

/// Deserialize + validate a core image.
/// Order: header length → magic → version → count bound → checksum over
/// RECORDS → record decode → non-decreasing `(t,s,r)` order (`d_inv`)
/// → dictionaries (bounds/UTF-8).
///
/// # Errors
/// [`TensorError::Decode`] on any structural, checksum, or order violation.
pub fn deserialize_core(data: &[u8]) -> Result<CoreImage, TensorError> {
    if data.len() < HEADER_LEN {
        return Err(TensorError::Decode("truncated core header".into()));
    }
    if &data[..8] != MAGIC {
        return Err(TensorError::Decode("bad core magic".into()));
    }
    let mut o = 8usize;
    let version = read_u32(data, &mut o, "core version")?;
    if version != FORMAT_VERSION {
        return Err(TensorError::Decode(format!(
            "unsupported core version {version}"
        )));
    }
    let count = read_u64(data, &mut o, "core count")?;
    let t_min = read_u64(data, &mut o, "core t_min")?;
    let t_max = read_u64(data, &mut o, "core t_max")?;
    let mean_major = read_f64(data, &mut o, "core mean")?;
    let std_major = read_f64(data, &mut o, "core std")?;
    let expect_sum = read_u64(data, &mut o, "core checksum")?;
    if count > MAX_RECORDS {
        return Err(TensorError::Decode("core record count too large".into()));
    }
    let n = count as usize;
    let rec_len = n
        .checked_mul(RECORD_LEN)
        .ok_or_else(|| TensorError::Decode("corrupt core count (length overflow)".into()))?;
    if o.checked_add(rec_len).is_none_or(|end| end > data.len()) {
        return Err(TensorError::Decode("truncated core records".into()));
    }
    let rec_bytes = &data[o..o + rec_len];
    if fnv1a64(rec_bytes) != expect_sum {
        return Err(TensorError::Decode(
            "checksum mismatch [core records]".into(),
        ));
    }
    let mut p = 0usize;
    let mut records = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        let t = read_u64(rec_bytes, &mut p, "core t")?;
        let s = read_u32(rec_bytes, &mut p, "core s")?;
        let r = read_u32(rec_bytes, &mut p, "core r")?;
        let amount_micro = read_i64(rec_bytes, &mut p, "core amount")?;
        let log1p_amt = read_f64(rec_bytes, &mut p, "core log1p")?;
        let z_amt = read_f64(rec_bytes, &mut p, "core z")?;
        if p + 8 > rec_bytes.len() {
            return Err(TensorError::Decode("truncated core record".into()));
        }
        let roundedness = rec_bytes[p];
        p += 1;
        if rec_bytes[p..p + 7] != [0u8; 7] {
            return Err(TensorError::Decode("nonzero core pad".into()));
        }
        p += 7;
        if roundedness > 1 {
            return Err(TensorError::Decode(format!(
                "non-binary roundedness {roundedness}"
            )));
        }
        records.push(CoreRecord {
            t, s, r, amount_micro, log1p_amt, z_amt, roundedness,
        });
    }
    o += rec_len;
    if records.windows(2).any(|w| w[0] > w[1]) {
        return Err(TensorError::Decode(
            "core records not in canonical (t,s,r) order".into(),
        ));
    }
    let mut dicts: Vec<Vec<String>> = Vec::with_capacity(2);
    for which in ["sender", "receiver"] {
        let m = read_u64(data, &mut o, "core dict count")? as usize;
        if m > MAX_DICT_ENTRIES {
            return Err(TensorError::Decode("core dict too large".into()));
        }
        let mut tab = Vec::with_capacity(m.min(1 << 16));
        for _ in 0..m {
            tab.push(read_len_str(data, &mut o, MAX_DICT_STR, which)?);
        }
        dicts.push(tab);
    }
    if o != data.len() {
        return Err(TensorError::Decode("trailing bytes in core".into()));
    }
    let receivers = dicts.pop().unwrap_or_default();
    let senders = dicts.pop().unwrap_or_default();
    Ok(CoreImage {
        records, senders, receivers, t_min, t_max,
        mean_major, std_major, record_count: count,
    })
}
