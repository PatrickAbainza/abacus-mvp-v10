//! `SAMCSF02` — derived, lossless, structure-preserving compressed CSF codec
//! (addendum §R5f; gap `G-PROFILE-3` reported, not invented).
//!
//! Re-encodes the SAME six fiber arrays of a canonical `CsfTensor` at minimal
//! width / delta-varint. It **never** sums, buckets, re-indexes, or drops a
//! coordinate or value slot: `Decode_v2(Encode_v2(T)) == T` and `nnz` is
//! unchanged. The stored value path is byte-exact (`amount_micro`), so the
//! per-transaction analysis substrate is fully retained.
//!
//! Std-only (`§I H1`); deterministic (`P12`/`R7`); never canonical — the
//! row-17 `ABACUS01` image remains the canonical compressed form (§R5b).
//!
//! ```text
//! header[84]: magic[8]="SAMCSF02" | version u32=2 | scale_exp u32
//!             | n0 u64 | n1 u64 | nnz u64 | t_min u64 | t_max u64 | raw_rows u64
//!             | mode[6] u8 | width[6] u8
//!             | checksum u64 (d_fnv over header-minus-magic+checksum plus PAYLOAD)
//! payload   : I0 | P0 | I1 | P1 | I2 | V   (each column mode-encoded)
//!             | senders | receivers | (row-17 dict shape)
//! ```

use super::compress::CsfTensor;
use super::serialize::{Header, Image};
use crate::prims::{read_u64, write_len_str, write_u32, write_u64};
use crate::tensor::value::MINOR_SCALE_EXP;
use crate::varint::{
    fnv_cover, min_width, read_fixed, read_i64_zigzag, read_uleb128, write_fixed,
    write_i64_zigzag, write_uleb128,
};
use crate::TensorError;

/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"SAMCSF02";
/// Format version.
pub const VERSION: u32 = 2;
/// Number of encoded columns.
pub const COLS: usize = 6;
/// Header length: 8+4+4+ (8*6) + 6 + 6 + 8 = 84.
pub const HEADER_LEN: usize = 84;
/// Max accepted dictionary entries.
const MAX_DICT_ENTRIES: usize = 50_000_000;
/// Max accepted dictionary string length.
const MAX_DICT_STR: u32 = 10_000_000;

const MODE_FIXED: u8 = 0;
const MODE_DELTA: u8 = 1;
const MODE_ZIGZAG: u8 = 2;

/// One encoded column: chosen mode, fixed width (mode 0), and bytes.
struct Col {
    mode: u8,
    width: u8,
    bytes: Vec<u8>,
}

/// Deterministic pick: delta-variant on a tie, else the smaller.
fn pick(a: Vec<u8>, a_mode: u8, b: Vec<u8>, b_mode: u8, b_width: u8) -> Col {
    if a.len() <= b.len() {
        Col { mode: a_mode, width: 0, bytes: a }
    } else {
        Col { mode: b_mode, width: b_width, bytes: b }
    }
}

fn enc_i0(csf: &CsfTensor, t_min: u64) -> Col {
    let mut a = Vec::new();
    let mut prev = t_min;
    for &t in &csf.mode0_coords {
        write_uleb128(&mut a, t.saturating_sub(prev));
        prev = t;
    }
    let max_off = csf.mode0_coords.last().map_or(0, |&t| t.saturating_sub(t_min));
    let w = min_width(max_off);
    let mut b = Vec::new();
    for &t in &csf.mode0_coords {
        write_fixed(&mut b, t.saturating_sub(t_min), w);
    }
    pick(a, MODE_DELTA, b, MODE_FIXED, w)
}

/// Non-decreasing pointer column → fiber sizes (mode 1) or raw pointers (mode 0).
fn enc_ptr(fptr: &[u64]) -> Col {
    let n = fptr.len().saturating_sub(1);
    let mut a = Vec::new();
    for i in 0..n {
        write_uleb128(&mut a, fptr[i + 1] - fptr[i]);
    }
    let max = fptr.last().copied().unwrap_or(0);
    let w = min_width(max);
    let mut b = Vec::new();
    for &p in fptr {
        write_fixed(&mut b, p, w);
    }
    pick(a, MODE_DELTA, b, MODE_FIXED, w)
}

/// Per-fiber strictly-increasing code column → per-fiber delta (mode 1) or
/// minimal fixed codes (mode 0).
fn enc_codes(coords: &[u32], fptr: &[u64]) -> Col {
    let nf = fptr.len().saturating_sub(1);
    let mut a = Vec::new();
    for fi in 0..nf {
        let (s, e) = (fptr[fi] as usize, fptr[fi + 1] as usize);
        let mut prev = 0u32;
        for (k, &v) in coords[s..e].iter().enumerate() {
            if k == 0 {
                write_uleb128(&mut a, u64::from(v));
            } else {
                write_uleb128(&mut a, u64::from(v - prev));
            }
            prev = v;
        }
    }
    let max = u64::from(coords.iter().copied().max().unwrap_or(0));
    let w = min_width(max);
    let mut b = Vec::new();
    for &v in coords {
        write_fixed(&mut b, u64::from(v), w);
    }
    pick(a, MODE_DELTA, b, MODE_FIXED, w)
}

fn enc_v(values: &[i64]) -> Col {
    let mut a = Vec::new();
    for &v in values {
        write_i64_zigzag(&mut a, v);
    }
    let mut b = Vec::new();
    for &v in values {
        write_fixed(&mut b, v as u64, 8);
    }
    pick(a, MODE_ZIGZAG, b, MODE_FIXED, 8)
}

/// Serialize a canonical `CsfTensor` to a `SAMCSF02` image (lossless).
#[must_use]
pub fn serialize_csf_v2(
    csf: &CsfTensor,
    senders: &[String],
    receivers: &[String],
    t_min: u64,
    t_max: u64,
    raw_rows: u64,
) -> Vec<u8> {
    let cols = [
        enc_i0(csf, t_min),
        enc_ptr(&csf.mode0_fptr),
        enc_codes(&csf.mode1_coords, &csf.mode0_fptr),
        enc_ptr(&csf.mode1_fptr),
        enc_codes(&csf.mode2_coords, &csf.mode1_fptr),
        enc_v(&csf.values),
    ];
    let mut payload: Vec<u8> = Vec::new();
    for c in &cols {
        payload.extend_from_slice(&c.bytes);
    }
    for dict in [senders, receivers] {
        write_u64(&mut payload, dict.len() as u64);
        for s in dict {
            write_len_str(&mut payload, s);
        }
    }
    // The covered prefix starts right after the magic: version through raw_rows
    // (every field that steers decode), then the mode/width tables, then the
    // payload bytes. A mutation anywhere in the covered bytes breaks FNV/R5.
    let mut buf: Vec<u8> = Vec::with_capacity(HEADER_LEN + payload.len());
    buf.extend_from_slice(MAGIC);
    write_u32(&mut buf, VERSION);
    write_u32(&mut buf, MINOR_SCALE_EXP);
    write_u64(&mut buf, csf.mode0_coords.len() as u64);
    write_u64(&mut buf, csf.mode1_coords.len() as u64);
    write_u64(&mut buf, csf.values.len() as u64);
    write_u64(&mut buf, t_min);
    write_u64(&mut buf, t_max);
    write_u64(&mut buf, raw_rows);
    for c in &cols {
        buf.push(c.mode);
    }
    for c in &cols {
        buf.push(c.width);
    }
    let mut out: Vec<u8> = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&buf);
    write_u64(&mut out, fnv_cover(&buf, &payload));
    out.extend_from_slice(&payload);
    out
}

fn dec_i0(
    d: &[u8],
    o: &mut usize,
    n0: usize,
    t_min: u64,
    mode: u8,
    width: u8,
) -> Result<Vec<u64>, TensorError> {
    let mut out = Vec::with_capacity(n0.min(1 << 20));
    match mode {
        MODE_DELTA => {
            let mut prev = t_min;
            for _ in 0..n0 {
                let dt = read_uleb128(d, o, "csf2 I0")?;
                let t = prev
                    .checked_add(dt)
                    .ok_or_else(|| TensorError::Decode("csf2 I0 overflow".into()))?;
                out.push(t);
                prev = t;
            }
        }
        MODE_FIXED => {
            for _ in 0..n0 {
                let off = read_fixed(d, o, width, "csf2 I0")?;
                let t = t_min
                    .checked_add(off)
                    .ok_or_else(|| TensorError::Decode("csf2 I0 overflow".into()))?;
                out.push(t);
            }
        }
        m => return Err(TensorError::Decode(format!("csf2 bad I0 mode {m}"))),
    }
    Ok(out)
}

/// Rebuild a sentinel-terminated pointer array of `n+1` entries.
fn dec_ptr(
    d: &[u8],
    o: &mut usize,
    n: usize,
    mode: u8,
    width: u8,
    field: &str,
) -> Result<Vec<u64>, TensorError> {
    let mut out = Vec::with_capacity(n + 1);
    match mode {
        MODE_DELTA => {
            out.push(0);
            let mut acc = 0u64;
            for _ in 0..n {
                acc = acc
                    .checked_add(read_uleb128(d, o, field)?)
                    .ok_or_else(|| TensorError::Decode(format!("csf2 {field} overflow")))?;
                out.push(acc);
            }
        }
        MODE_FIXED => {
            for _ in 0..n + 1 {
                out.push(read_fixed(d, o, width, field)?);
            }
        }
        m => return Err(TensorError::Decode(format!("csf2 bad {field} mode {m}"))),
    }
    Ok(out)
}

/// Rebuild a per-fiber strictly-increasing code column.
fn dec_codes(
    d: &[u8],
    o: &mut usize,
    fptr: &[u64],
    mode: u8,
    width: u8,
    field: &str,
) -> Result<Vec<u32>, TensorError> {
    let nf = fptr.len().saturating_sub(1);
    let total = fptr.last().copied().unwrap_or(0) as usize;
    let mut out = Vec::with_capacity(total.min(1 << 20));
    match mode {
        MODE_DELTA => {
            for fi in 0..nf {
                let len = (fptr[fi + 1] - fptr[fi]) as usize;
                let mut prev = 0u32;
                for k in 0..len {
                    let dd = read_uleb128(d, o, field)?;
                    let du = u32::try_from(dd)
                        .map_err(|_| TensorError::Decode(format!("csf2 {field} range")))?;
                    let v = if k == 0 {
                        du
                    } else {
                        prev.checked_add(du)
                            .ok_or_else(|| TensorError::Decode(format!("csf2 {field} overflow")))?
                    };
                    out.push(v);
                    prev = v;
                }
            }
        }
        MODE_FIXED => {
            for _ in 0..total {
                out.push(read_fixed(d, o, width, field)? as u32);
            }
        }
        m => return Err(TensorError::Decode(format!("csf2 bad {field} mode {m}"))),
    }
    Ok(out)
}

fn dec_v(
    d: &[u8],
    o: &mut usize,
    nnz: usize,
    mode: u8,
    width: u8,
) -> Result<Vec<i64>, TensorError> {
    let mut out = Vec::with_capacity(nnz.min(1 << 20));
    match mode {
        MODE_ZIGZAG => {
            for _ in 0..nnz {
                out.push(read_i64_zigzag(d, o, "csf2 V")?);
            }
        }
        MODE_FIXED => {
            for _ in 0..nnz {
                out.push(read_fixed(d, o, width, "csf2 V")? as i64);
            }
        }
        m => return Err(TensorError::Decode(format!("csf2 bad V mode {m}"))),
    }
    Ok(out)
}

/// Deserialize + validate a `SAMCSF02` image (fail-closed, `R_5`).
///
/// # Errors
/// [`TensorError::Decode`] on structure, checksum, range, or invariant failure.
pub fn deserialize_csf_v2(data: &[u8]) -> Result<Image, TensorError> {
    if data.len() < HEADER_LEN {
        return Err(TensorError::Decode("truncated csf2 header".into()));
    }
    if &data[..8] != MAGIC {
        return Err(TensorError::Decode("bad csf2 magic".into()));
    }
    let mut o = 8usize;
    let version = crate::prims::read_u32(data, &mut o, "csf2 version")?;
    if version != VERSION {
        return Err(TensorError::Decode(format!("unsupported csf2 version {version}")));
    }
    let scale_exp = crate::prims::read_u32(data, &mut o, "csf2 scale")?;
    if scale_exp != MINOR_SCALE_EXP {
        return Err(TensorError::Decode("csf2 bad scale_exp".into()));
    }
    let n0 = read_u64(data, &mut o, "csf2 n0")?;
    let n1 = read_u64(data, &mut o, "csf2 n1")?;
    let nnz = read_u64(data, &mut o, "csf2 nnz")?;
    let t_min = read_u64(data, &mut o, "csf2 t_min")?;
    let t_max = read_u64(data, &mut o, "csf2 t_max")?;
    let raw_rows = read_u64(data, &mut o, "csf2 raw_rows")?;
    if o + 2 * COLS + 8 > data.len() {
        return Err(TensorError::Decode("truncated csf2 mode tables".into()));
    }
    let mut modes = [0u8; COLS];
    let mut widths = [0u8; COLS];
    modes.copy_from_slice(&data[o..o + COLS]);
    o += COLS;
    widths.copy_from_slice(&data[o..o + COLS]);
    o += COLS;
    let expect_sum = read_u64(data, &mut o, "csf2 checksum")?;
    if o != HEADER_LEN {
        return Err(TensorError::Decode("csf2 header length mismatch".into()));
    }
    let covered: &[u8] = &data[..HEADER_LEN - 8];
    let payload = &data[HEADER_LEN..];
    if fnv_cover(covered, payload) != expect_sum {
        return Err(TensorError::Decode(
            "checksum mismatch [csf2 header+payload]".into(),
        ));
    }
    let (nn0, nn1, nnnz) = (n0 as usize, n1 as usize, nnz as usize);
    let mut p = 0usize;
    let i0 = dec_i0(payload, &mut p, nn0, t_min, modes[0], widths[0])?;
    let p0 = dec_ptr(payload, &mut p, nn0, modes[1], widths[1], "P0")?;
    let i1 = dec_codes(payload, &mut p, &p0, modes[2], widths[2], "I1")?;
    let p1 = dec_ptr(payload, &mut p, nn1, modes[3], widths[3], "P1")?;
    let i2 = dec_codes(payload, &mut p, &p1, modes[4], widths[4], "I2")?;
    let vv = dec_v(payload, &mut p, nnnz, modes[5], widths[5])?;
    let mut dicts: Vec<Vec<String>> = Vec::with_capacity(2);
    for which in ["csf2 sender", "csf2 receiver"] {
        let m = read_u64(payload, &mut p, "csf2 dict count")? as usize;
        if m > MAX_DICT_ENTRIES {
            return Err(TensorError::Decode("csf2 dict too large".into()));
        }
        let mut tab = Vec::with_capacity(m.min(1 << 16));
        for _ in 0..m {
            tab.push(crate::prims::read_len_str(payload, &mut p, MAX_DICT_STR, which)?);
        }
        dicts.push(tab);
    }
    if p != payload.len() {
        return Err(TensorError::Decode("trailing bytes in csf2".into()));
    }
    let csf = CsfTensor {
        mode0_coords: i0,
        mode0_fptr: p0,
        mode1_coords: i1,
        mode1_fptr: p1,
        mode2_coords: i2,
        values: vv,
    };
    if !csf.check_invariants() {
        return Err(TensorError::Decode("CSF invariant violation [csf2]".into()));
    }
    let receivers = dicts.pop().unwrap_or_default();
    let senders = dicts.pop().unwrap_or_default();
    Ok(Image {
        csf,
        senders,
        receivers,
        t_min,
        t_max,
        raw_rows,
        header: Header {
            version,
            t_width: if modes[0] == MODE_FIXED { widths[0] } else { 0 },
            s_width: if modes[2] == MODE_FIXED { widths[2] } else { 0 },
            r_width: if modes[4] == MODE_FIXED { widths[4] } else { 0 },
            v_width: if modes[5] == MODE_FIXED { widths[5] } else { 0 },
            scale_exp,
            n0,
            n1,
            nnz,
            checksum: expect_sum,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::entry::TxEntry;
    use crate::csf::compress::compress;

    fn roundtrip(coo: &[TxEntry], senders: &[String], receivers: &[String]) {
        let csf = compress(coo);
        let bytes = serialize_csf_v2(&csf, senders, receivers, 0, 0, coo.len() as u64);
        let img = deserialize_csf_v2(&bytes).expect("decode csf2");
        assert_eq!(img.csf, csf, "lossless: Decode_v2(Encode_v2(T)) == T");
        assert_eq!(img.csf.nnz(), csf.nnz(), "nnz unchanged (no aggregation)");
        assert!(img.csf.check_invariants());
        assert_eq!(img.senders, senders);
        assert_eq!(img.receivers, receivers);
        // Determinism (R7).
        assert_eq!(
            serialize_csf_v2(&csf, senders, receivers, 0, 0, coo.len() as u64),
            bytes
        );
    }

    #[test]
    fn csf2_roundtrip_small_and_structural() {
        let coo = vec![
            TxEntry::new(1, 2, 3, 7),
            TxEntry::new(1, 2, 4, -2),
            TxEntry::new(1, 5, 0, 1),
            TxEntry::new(4, 0, 0, 100),
            TxEntry::new(9, 9, 9, i64::from(-1)),
        ];
        roundtrip(&coo, &["a".into(), "b".into()], &["c".into()]);
    }

    #[test]
    fn csf2_roundtrip_empty_is_zero_tensor() {
        roundtrip(&[], &[], &[]);
    }

    #[test]
    fn csf2_roundtrip_large_codes_and_values() {
        // Wide coordinates + a very large magnitude value (wide varint).
        let coo = vec![
            TxEntry::new(1_700_000_000, 600_000, 300_000, i64::from(2_000_000_000)),
            TxEntry::new(1_700_000_500, 600_001, 300_001, i64::from(-3_000_000)),
            TxEntry::new(1_700_123_456, 700_000, 400_000, 1),
        ];
        roundtrip(&coo, &["z".into()], &["y".into()]);
    }

    #[test]
    fn csf2_fails_closed_on_corruption() {
        let coo = vec![TxEntry::new(1, 2, 3, 7), TxEntry::new(2, 0, 0, 9)];
        let csf = compress(&coo);
        let mut bytes = serialize_csf_v2(&csf, &[], &[], 0, 0, 2);
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        assert!(deserialize_csf_v2(&bytes).is_err());
    }
}



