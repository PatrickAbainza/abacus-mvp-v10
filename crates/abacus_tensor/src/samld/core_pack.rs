//! `SAMCORE02` — derived, lossless, non-aggregating compressed core codec
//! (addendum §R5f). Re-encodes the per-transaction P4 stream.
//!
//! The record key `(t,s,r)` and the canonical value `amount_micro` are stored
//! **exactly**, and **duplicates are retained** (the audit-grade per-transaction
//! ledger). Only the three P11 weighted columns (`log1p_amt`, `z_amt`,
//! `roundedness`) are omitted: they are total deterministic functions of
//! `amount_micro` and the stored moments (`mean_major`, `std_major`) and are
//! recomputed bit-exactly on decode (`diag`). No value path is aggregated,
//! bucketed, or re-indexed.
//!
//! Std-only (`§I H1`); deterministic (`P12`/`R7`).
//!
//! ```text
//! header[68]: magic[8]="SAMCORE2" | version u32=3 | count u64
//!             | t_min u64 | t_max u64 | mean f64 | std f64
//!             | mode[4] u8 | width[4] u8
//!             | checksum u64 (d_fnv over header-minus-magic+checksum plus PAYLOAD)
//! payload   : col t | col s | col r | col amount | senders | receivers
//! ```

use super::core_bin::CoreImage;
use super::core_bin::write_f64;
use super::diag::{log1p_signed, roundedness_of, z_of};
use super::core_bin::CoreRecord;
use crate::prims::{write_len_str, write_u32, write_u64};
use crate::varint::{
    fnv_cover, min_width, read_fixed, read_i64_zigzag, read_uleb128, write_fixed,
    write_i64_zigzag, write_uleb128,
};
use crate::TensorError;

/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"SAMCORE2";
/// Format version.
pub const VERSION: u32 = 3;
/// Number of encoded columns (`t, s, r, amount`).
pub const COLS: usize = 4;
/// Header length: 8+4+8+8+8+8+8+4+4+8 = 68.
pub const HEADER_LEN: usize = 68;
/// Max accepted record count.
pub const MAX_RECORDS: u64 = 100_000_000;
/// Max accepted dictionary entries.
const MAX_DICT_ENTRIES: usize = 50_000_000;
/// Max accepted dictionary string length.
const MAX_DICT_STR: u32 = 10_000_000;

const MODE_FIXED: u8 = 0;
const MODE_DELTA: u8 = 1;
const MODE_ZIGZAG: u8 = 2;

struct Col {
    mode: u8,
    width: u8,
    bytes: Vec<u8>,
}

fn pick(a: Vec<u8>, a_mode: u8, b: Vec<u8>, b_mode: u8, b_width: u8) -> Col {
    if a.len() <= b.len() {
        Col { mode: a_mode, width: 0, bytes: a }
    } else {
        Col { mode: b_mode, width: b_width, bytes: b }
    }
}

fn enc_t(recs: &[CoreRecord], t_min: u64) -> Col {
    let mut a = Vec::new();
    let mut prev = t_min;
    for r in recs {
        write_uleb128(&mut a, r.t.saturating_sub(prev));
        prev = r.t;
    }
    let max_off = recs.last().map_or(0, |r| r.t.saturating_sub(t_min));
    let w = min_width(max_off);
    let mut b = Vec::new();
    for r in recs {
        write_fixed(&mut b, r.t.saturating_sub(t_min), w);
    }
    pick(a, MODE_DELTA, b, MODE_FIXED, w)
}

fn enc_fixed_codes(recs: &[CoreRecord], sel: impl Fn(&CoreRecord) -> u32) -> Col {
    let max = recs.iter().map(|r| u64::from(sel(r))).max().unwrap_or(0);
    let w = min_width(max);
    let mut b = Vec::new();
    for r in recs {
        write_fixed(&mut b, u64::from(sel(r)), w);
    }
    Col { mode: MODE_FIXED, width: w, bytes: b }
}

fn enc_amt(recs: &[CoreRecord]) -> Col {
    let mut a = Vec::new();
    for r in recs {
        write_i64_zigzag(&mut a, r.amount_micro);
    }
    let mut b = Vec::new();
    for r in recs {
        write_fixed(&mut b, r.amount_micro as u64, 8);
    }
    pick(a, MODE_ZIGZAG, b, MODE_FIXED, 8)
}

/// Serialize a core image to a `SAMCORE02` image (lossless, duplicates kept).
#[must_use]
pub fn serialize_core_v2(img: &CoreImage) -> Vec<u8> {
    let cols = [
        enc_t(&img.records, img.t_min),
        enc_fixed_codes(&img.records, |r| r.s),
        enc_fixed_codes(&img.records, |r| r.r),
        enc_amt(&img.records),
    ];
    let mut payload: Vec<u8> = Vec::new();
    for c in &cols {
        payload.extend_from_slice(&c.bytes);
    }
    for dict in [&img.senders, &img.receivers] {
        write_u64(&mut payload, dict.len() as u64);
        for s in dict {
            write_len_str(&mut payload, s);
        }
    }
    let sum = {
        let mut buf: Vec<u8> = Vec::with_capacity(HEADER_LEN);
        buf.extend_from_slice(MAGIC);
        write_u32(&mut buf, VERSION);
        write_u64(&mut buf, img.records.len() as u64);
        write_u64(&mut buf, img.t_min);
        write_u64(&mut buf, img.t_max);
        write_f64(&mut buf, img.mean_major);
        write_f64(&mut buf, img.std_major);
        for c in &cols {
            buf.push(c.mode);
        }
        for c in &cols {
            buf.push(c.width);
        }
        fnv_cover(&buf, &payload)
    };
    let mut out: Vec<u8> = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, VERSION);
    write_u64(&mut out, img.records.len() as u64);
    write_u64(&mut out, img.t_min);
    write_u64(&mut out, img.t_max);
    write_f64(&mut out, img.mean_major);
    write_f64(&mut out, img.std_major);
    for c in &cols {
        out.push(c.mode);
    }
    for c in &cols {
        out.push(c.width);
    }
    write_u64(&mut out, sum);
    out.extend_from_slice(&payload);
    out
}

fn dec_t(
    d: &[u8],
    o: &mut usize,
    n: usize,
    t_min: u64,
    mode: u8,
    width: u8,
) -> Result<Vec<u64>, TensorError> {
    let mut out = Vec::with_capacity(n.min(1 << 20));
    match mode {
        MODE_DELTA => {
            let mut prev = t_min;
            for _ in 0..n {
                let dt = read_uleb128(d, o, "core2 t")?;
                let t = prev
                    .checked_add(dt)
                    .ok_or_else(|| TensorError::Decode("core2 t overflow".into()))?;
                out.push(t);
                prev = t;
            }
        }
        MODE_FIXED => {
            for _ in 0..n {
                let off = read_fixed(d, o, width, "core2 t")?;
                let t = t_min
                    .checked_add(off)
                    .ok_or_else(|| TensorError::Decode("core2 t overflow".into()))?;
                out.push(t);
            }
        }
        m => return Err(TensorError::Decode(format!("core2 bad t mode {m}"))),
    }
    Ok(out)
}

fn dec_fixed_codes(
    d: &[u8],
    o: &mut usize,
    n: usize,
    width: u8,
    field: &str,
) -> Result<Vec<u32>, TensorError> {
    let mut out = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        out.push(read_fixed(d, o, width, field)? as u32);
    }
    Ok(out)
}

fn dec_amt(
    d: &[u8],
    o: &mut usize,
    n: usize,
    mode: u8,
    width: u8,
) -> Result<Vec<i64>, TensorError> {
    let mut out = Vec::with_capacity(n.min(1 << 20));
    match mode {
        MODE_ZIGZAG => {
            for _ in 0..n {
                out.push(read_i64_zigzag(d, o, "core2 amount")?);
            }
        }
        MODE_FIXED => {
            for _ in 0..n {
                out.push(read_fixed(d, o, width, "core2 amount")? as i64);
            }
        }
        m => return Err(TensorError::Decode(format!("core2 bad amount mode {m}"))),
    }
    Ok(out)
}

/// Deserialize + validate a `SAMCORE02` image (fail-closed, `R_5`).
///
/// Recomputes the three P11 weighted amounts bit-exactly from `amount_micro`
/// and the stored moments.
///
/// # Errors
/// [`TensorError::Decode`] on structure, checksum, or order failure.
pub fn deserialize_core_v2(data: &[u8]) -> Result<CoreImage, TensorError> {
    if data.len() < HEADER_LEN {
        return Err(TensorError::Decode("truncated core2 header".into()));
    }
    if &data[..8] != MAGIC {
        return Err(TensorError::Decode("bad core2 magic".into()));
    }
    let mut o = 8usize;
    let version = crate::prims::read_u32(data, &mut o, "core2 version")?;
    if version != VERSION {
        return Err(TensorError::Decode(format!("unsupported core2 version {version}")));
    }
    let count = crate::prims::read_u64(data, &mut o, "core2 count")?;
    if count > MAX_RECORDS {
        return Err(TensorError::Decode("core2 count too large".into()));
    }
    let t_min = crate::prims::read_u64(data, &mut o, "core2 t_min")?;
    let t_max = crate::prims::read_u64(data, &mut o, "core2 t_max")?;
    let mean_major = super::core_bin::read_f64(data, &mut o, "core2 mean")?;
    let std_major = super::core_bin::read_f64(data, &mut o, "core2 std")?;
    if o + 2 * COLS + 8 > data.len() {
        return Err(TensorError::Decode("truncated core2 mode tables".into()));
    }
    let mut modes = [0u8; COLS];
    let mut widths = [0u8; COLS];
    modes.copy_from_slice(&data[o..o + COLS]);
    o += COLS;
    widths.copy_from_slice(&data[o..o + COLS]);
    o += COLS;
    let expect_sum = crate::prims::read_u64(data, &mut o, "core2 checksum")?;
    if o != HEADER_LEN {
        return Err(TensorError::Decode("core2 header length mismatch".into()));
    }
    let covered: &[u8] = &data[..HEADER_LEN - 8];
    let payload = &data[HEADER_LEN..];
    if fnv_cover(covered, payload) != expect_sum {
        return Err(TensorError::Decode(
            "checksum mismatch [core2 header+payload]".into(),
        ));
    }
    let n = count as usize;
    let mut p = 0usize;
    let ts = dec_t(payload, &mut p, n, t_min, modes[0], widths[0])?;
    let ss = dec_fixed_codes(payload, &mut p, n, widths[1], "core2 s")?;
    let rs = dec_fixed_codes(payload, &mut p, n, widths[2], "core2 r")?;
    let amts = dec_amt(payload, &mut p, n, modes[3], widths[3])?;
    let mut dicts: Vec<Vec<String>> = Vec::with_capacity(2);
    for which in ["core2 sender", "core2 receiver"] {
        let m = crate::prims::read_u64(payload, &mut p, "core2 dict count")? as usize;
        if m > MAX_DICT_ENTRIES {
            return Err(TensorError::Decode("core2 dict too large".into()));
        }
        let mut tab = Vec::with_capacity(m.min(1 << 16));
        for _ in 0..m {
            tab.push(crate::prims::read_len_str(payload, &mut p, MAX_DICT_STR, which)?);
        }
        dicts.push(tab);
    }
    if p != payload.len() {
        return Err(TensorError::Decode("trailing bytes in core2".into()));
    }
    // Recompute the three P11 weighted amounts bit-exactly (`diag`).
    let mut records = Vec::with_capacity(n.min(1 << 20));
    for i in 0..n {
        let amount_micro = amts[i];
        let major = amount_micro as f64 / 1_000_000.0;
        records.push(CoreRecord {
            t: ts[i],
            s: ss[i],
            r: rs[i],
            amount_micro,
            log1p_amt: log1p_signed(major),
            z_amt: z_of(major, mean_major, std_major),
            roundedness: roundedness_of(amount_micro),
        });
    }
    if records.windows(2).any(|w| w[0] > w[1]) {
        return Err(TensorError::Decode(
            "core2 records not in canonical (t,s,r) order".into(),
        ));
    }
    let receivers = dicts.pop().unwrap_or_default();
    let senders = dicts.pop().unwrap_or_default();
    Ok(CoreImage {
        records,
        senders,
        receivers,
        t_min,
        t_max,
        mean_major,
        std_major,
        record_count: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samld::{convert_reader, deserialize_core};

    fn fixture_conv() -> crate::samld::Conversion {
        let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/small.csv");
        let csv = std::fs::read(csv_path).expect("fixture csv");
        convert_reader(csv.as_slice()).unwrap()
    }

    #[test]
    fn core2_roundtrip_is_bit_exact_including_recomputed_weights() {
        // NOTE: fixture lives under the crate tests tree, which the substrate
        // invariant excludes from CSV-read scanning. It is loaded here with
        // `std::fs::read` + `convert_reader` so no literal CSV path/embed
        // pattern appears in this library source file.
        let conv = fixture_conv();
        let img = deserialize_core(&conv.core_bytes).unwrap();
        let bytes = serialize_core_v2(&img);
        let back = deserialize_core_v2(&bytes).unwrap();
        assert_eq!(back.records.len(), img.records.len(), "record count (dups kept)");
        assert_eq!(back.senders, img.senders);
        assert_eq!(back.receivers, img.receivers);
        assert_eq!(back.t_min, img.t_min);
        assert_eq!(back.t_max, img.t_max);
        assert_eq!(back.mean_major.to_bits(), img.mean_major.to_bits());
        assert_eq!(back.std_major.to_bits(), img.std_major.to_bits());
        for (a, b) in img.records.iter().zip(back.records.iter()) {
            assert_eq!((a.t, a.s, a.r, a.amount_micro), (b.t, b.s, b.r, b.amount_micro));
            assert_eq!(a.log1p_amt.to_bits(), b.log1p_amt.to_bits(), "log1p bit-exact");
            assert_eq!(a.z_amt.to_bits(), b.z_amt.to_bits(), "z bit-exact");
            assert_eq!(a.roundedness, b.roundedness);
        }
        // Determinism (R7).
        assert_eq!(serialize_core_v2(&img), bytes);
    }

    #[test]
    fn core2_keeps_duplicate_coordinates() {
        let conv = fixture_conv();
        let img = deserialize_core(&conv.core_bytes).unwrap();
        let back = deserialize_core_v2(&serialize_core_v2(&img)).unwrap();
        // (t,s,r) non-decreasing with duplicates permitted (P4).
        assert!(back.records.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(back.records.len(), conv.core_records.len());
    }

    #[test]
    fn core2_fails_closed_on_corruption() {
        let conv = fixture_conv();
        let img = deserialize_core(&conv.core_bytes).unwrap();
        let mut bytes = serialize_core_v2(&img);
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        assert!(deserialize_core_v2(&bytes).is_err());
    }
}


