//! Row 17 (§B.2): serialization of the canonical tensor artifact.
//!
//! Immutable binary image, fixed field order, little-endian (`a_{13}.1`),
//! FNV-1a/64 provenance checksum (`d_fnv`, one realization shared with the
//! sidecar — `composites::fnv1a64`).
//!
//! ```text
//! magic[8]  = "ABACUS01"
//! version   : u32   = 1
//! widths    : u8[4] = [8, 4, 4, 8]        (t, s, r, value byte widths)
//! scale_exp : u32   = 6                  (P2 canonical micro-unit scale)
//! n0        : u64                        (|I0| distinct t)
//! n1        : u64                        (|I1| distinct (t, s))
//! nnz       : u64                        (|supp(T)|)
//! payload_checksum : u64                 (FNV-1a/64 over PAYLOAD bytes only)
//! payload   : I0 | P0 | I1 | P1 | I2 | V
//! dicts     : senders, receivers  (u64 count + [u32 len + UTF-8 bytes]*)
//! footer    : t_min u64 | t_max u64 | raw_rows u64
//! ```
//!
//! Two **distinct** checksum quantities, never conflated (P11): the embedded
//! `payload_checksum` covers the payload bytes only, while the artifact
//! checksum written by `build_samld` into `stats.json` covers the whole file.
//!
//! Deserialization validation order (parsing safety first, then checksum
//! specificity; every failure rejects — `R_5`): magic → version → widths →
//! scale_exp → counts → payload-length bounds (`truncated payload`) → payload
//! checksum (`checksum mismatch [payload]` or
//! `[payload/corrupt-header-counts]`) → payload decode → dictionaries
//! (bounds/UTF-8) → provenance footer → CSF invariants.

use crate::coo::builder::CanonicalBuild;
use crate::csf::compress::CsfTensor;
use crate::prims::{read_len_str, read_u32, read_u64, write_len_str, write_u32, write_u64};
use crate::tensor::value::MINOR_SCALE_EXP;
use crate::TensorError;

/// Serialized format version.
pub const FORMAT_VERSION: u32 = 1;
/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"ABACUS01";
/// Fixed field widths `[t, s, r, value]`.
pub const WIDTHS: [u8; 4] = [8, 4, 4, 8];

/// `d_fnv` — FNV-1a/64 over serialized bytes (single realization; §B.1).
pub use crate::composites::fnv1a64 as checksum;

/// Decoded header: counts, widths, scale and the embedded payload checksum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Format version.
    pub version: u32,
    /// Byte width of `t` coordinates.
    pub t_width: u8,
    /// Byte width of `s` coordinates.
    pub s_width: u8,
    /// Byte width of `r` coordinates.
    pub r_width: u8,
    /// Byte width of values.
    pub v_width: u8,
    /// Decimal scale exponent (canonical `6`, P2).
    pub scale_exp: u32,
    /// Number of distinct `t` fibers.
    pub n0: u64,
    /// Number of distinct `(t, s)` fibers.
    pub n1: u64,
    /// Number of nonzero coordinates.
    pub nnz: u64,
    /// FNV-1a/64 over the PAYLOAD section bytes only (`I0|P0|I1|P1|I2|V`).
    pub checksum: u64,
}

/// Parsed image: CSF + dictionaries + provenance footer.
#[derive(Debug, Clone)]
pub struct Image {
    /// The canonical CSF tensor.
    pub csf: CsfTensor,
    /// Sender dictionary (ascending `prec_dict` order == code order).
    pub senders: Vec<String>,
    /// Receiver dictionary.
    pub receivers: Vec<String>,
    /// Minimum timestamp recorded at build.
    pub t_min: u64,
    /// Maximum timestamp recorded at build.
    pub t_max: u64,
    /// Raw rows observed at build.
    pub raw_rows: u64,
    /// Decoded header.
    pub header: Header,
}

/// Row 17 — serialize `CSF(T)` plus dictionaries and provenance footer.
///
/// Deterministic: identical input yields byte-identical output (no map
/// iteration, fixed field order; `R_7`).
#[must_use]
pub fn serialize(csf: &CsfTensor, build: &CanonicalBuild) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    for &v in &csf.mode0_coords {
        write_u64(&mut payload, v);
    }
    for &v in &csf.mode0_fptr {
        write_u64(&mut payload, v);
    }
    for &v in &csf.mode1_coords {
        write_u32(&mut payload, v);
    }
    for &v in &csf.mode1_fptr {
        write_u64(&mut payload, v);
    }
    for &v in &csf.mode2_coords {
        write_u32(&mut payload, v);
    }
    for &v in &csf.values {
        crate::prims::write_i64(&mut payload, v);
    }
    let sum = checksum(&payload);
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, FORMAT_VERSION);
    out.extend_from_slice(&WIDTHS);
    write_u32(&mut out, MINOR_SCALE_EXP);
    write_u64(&mut out, csf.mode0_coords.len() as u64);
    write_u64(&mut out, csf.mode1_coords.len() as u64);
    write_u64(&mut out, csf.values.len() as u64);
    write_u64(&mut out, sum);
    out.extend_from_slice(&payload);
    for dict in [&build.senders.table, &build.receivers.table] {
        write_u64(&mut out, dict.len() as u64);
        for s in dict {
            write_len_str(&mut out, s);
        }
    }
    write_u64(&mut out, build.t_min);
    write_u64(&mut out, build.t_max);
    write_u64(&mut out, build.raw_rows);
    out
}

/// Maximum dictionary size accepted at decode (bounds a hostile count).
const MAX_DICT_ENTRIES: usize = 50_000_000;
/// Maximum length of one dictionary identifier.
const MAX_DICT_STR: u32 = 10_000_000;
/// Minimum bytes that must follow the payload: two dict counts (8+8) plus the
/// three provenance footer fields (8+8+8).
const MIN_TAIL: usize = 8 + 8 + 8 + 8 + 8;

/// Row 17 — deserialize and verify a canonical tensor artifact.
///
/// Rejects with `E_dec` on any structural violation, bounds violation, or
/// checksum mismatch; `K8` keeps `E_io` on the file-I/O boundary only.
///
/// # Errors
/// [`TensorError::Decode`] on malformation or checksum mismatch.
pub fn deserialize(data: &[u8]) -> Result<Image, TensorError> {
    if data.len() < 8 || &data[0..8] != MAGIC {
        return Err(TensorError::Decode("bad magic".into()));
    }
    let mut o = 8usize;
    let version = read_u32(data, &mut o, "version")?;
    if version != FORMAT_VERSION {
        return Err(TensorError::Decode(format!(
            "unsupported version {version}"
        )));
    }
    if o + 4 > data.len() {
        return Err(TensorError::Decode("truncated widths".into()));
    }
    let (t_w, s_w, r_w, v_w) = (data[o], data[o + 1], data[o + 2], data[o + 3]);
    o += 4;
    if [t_w, s_w, r_w, v_w] != WIDTHS {
        return Err(TensorError::Decode("unsupported widths".into()));
    }
    let scale_exp = read_u32(data, &mut o, "scale_exp")?;
    if scale_exp != MINOR_SCALE_EXP {
        return Err(TensorError::Decode(format!(
            "unsupported scale_exp {scale_exp} (canonical {MINOR_SCALE_EXP})"
        )));
    }
    let n0 = read_u64(data, &mut o, "n0")?;
    let n1 = read_u64(data, &mut o, "n1")?;
    let nnz = read_u64(data, &mut o, "nnz")?;
    let expect_sum = read_u64(data, &mut o, "payload_checksum")?;
    let n0u = usize::try_from(n0).map_err(|_| TensorError::Decode("n0 overflow".into()))?;
    let n1u = usize::try_from(n1).map_err(|_| TensorError::Decode("n1 overflow".into()))?;
    let nzu = usize::try_from(nnz).map_err(|_| TensorError::Decode("nnz overflow".into()))?;
    // Checked length arithmetic: hostile counts reject, they never wrap (a_04).
    let pay_len = n0u
        .checked_mul(8)
        .and_then(|a| a.checked_add(n0u.checked_add(1)?.checked_mul(8)?))
        .and_then(|a| a.checked_add(n1u.checked_mul(4)?))
        .and_then(|a| a.checked_add(n1u.checked_add(1)?.checked_mul(8)?))
        .and_then(|a| a.checked_add(nzu.checked_mul(4)?))
        .and_then(|a| a.checked_add(nzu.checked_mul(8)?))
        .ok_or_else(|| TensorError::Decode("corrupt counts (length overflow)".into()))?;
    if o.checked_add(pay_len).is_none_or(|end| end > data.len()) {
        return Err(TensorError::Decode("truncated payload".into()));
    }
    let payload = &data[o..o + pay_len];
    if checksum(payload) != expect_sum {
        // Bounds are already satisfied, so the corruption is either in the
        // header counts (a wrong-but-in-bounds length, recognized by a
        // too-short tail) or in the payload bytes. Both reject; the message
        // only classifies which.
        let trailing = data.len() - o;
        if pay_len > trailing || trailing - pay_len < MIN_TAIL {
            return Err(TensorError::Decode(
                "checksum mismatch [payload/corrupt-header-counts]".into(),
            ));
        }
        return Err(TensorError::Decode("checksum mismatch [payload]".into()));
    }
    let mut p = 0usize;
    let mut i0 = Vec::with_capacity(n0u);
    for _ in 0..n0u {
        i0.push(read_u64(payload, &mut p, "payload I0")?);
    }
    let mut p0 = Vec::with_capacity(n0u + 1);
    for _ in 0..n0u + 1 {
        p0.push(read_u64(payload, &mut p, "payload P0")?);
    }
    let mut i1 = Vec::with_capacity(n1u);
    for _ in 0..n1u {
        i1.push(read_u32(payload, &mut p, "payload I1")?);
    }
    let mut p1 = Vec::with_capacity(n1u + 1);
    for _ in 0..n1u + 1 {
        p1.push(read_u64(payload, &mut p, "payload P1")?);
    }
    let mut i2 = Vec::with_capacity(nzu);
    for _ in 0..nzu {
        i2.push(read_u32(payload, &mut p, "payload I2")?);
    }
    let mut vv = Vec::with_capacity(nzu);
    for _ in 0..nzu {
        vv.push(crate::prims::read_i64(payload, &mut p, "payload V")?);
    }
    o += pay_len;
    let mut dicts: Vec<Vec<String>> = Vec::with_capacity(2);
    for which in ["sender", "receiver"] {
        let n = read_u64(data, &mut o, "dict count")? as usize;
        if n > MAX_DICT_ENTRIES {
            return Err(TensorError::Decode("dict too large".into()));
        }
        let mut tab = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            tab.push(read_len_str(data, &mut o, MAX_DICT_STR, which)?);
        }
        dicts.push(tab);
    }
    let t_min = read_u64(data, &mut o, "t_min")?;
    let t_max = read_u64(data, &mut o, "t_max")?;
    let raw_rows = read_u64(data, &mut o, "raw_rows")?;
    let csf = CsfTensor {
        mode0_coords: i0,
        mode0_fptr: p0,
        mode1_coords: i1,
        mode1_fptr: p1,
        mode2_coords: i2,
        values: vv,
    };
    // Row 11 d_inv: structural invariants of the decoded object (P3/P6).
    if !csf.check_invariants() {
        return Err(TensorError::Decode("CSF invariant violation".into()));
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
            t_width: t_w,
            s_width: s_w,
            r_width: r_w,
            v_width: v_w,
            scale_exp,
            n0,
            n1,
            nnz,
            checksum: expect_sum,
        },
    })
}
