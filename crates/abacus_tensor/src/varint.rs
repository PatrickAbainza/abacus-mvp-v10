//! LEB128 / zigzag / minimal-width helpers for the derived, lossless
//! compressed codecs admitted by `golden_source_csf_profile_addendum` §R5f
//! (`SAMCSF02`, `SAMCORE02`, `SAMEVL03`).
//!
//! Pure byte-level encoding of already-computed integers: no value is
//! created, altered, summed, or dropped (§R5f: derived, lossless,
//! structure-preserving). Standard library only (`§I H1`). Fail-closed, `R_5`.

use crate::TensorError;

/// Maximum bytes a `u64` LEB128 may occupy (ceil(64/7) = 10).
pub const MAX_ULEB128_BYTES: usize = 10;

/// Zigzag map `i64 -> u64` (small magnitude -> small code, sign in bit 0).
#[inline]
#[must_use]
pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Inverse zigzag map `u64 -> i64`.
#[inline]
#[must_use]
pub fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Append an unsigned LEB128 varint.
#[inline]
pub fn write_uleb128(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

/// Append a zigzag LEB128 varint for a signed value.
#[inline]
pub fn write_i64_zigzag(out: &mut Vec<u8>, v: i64) {
    write_uleb128(out, zigzag(v));
}

/// Read an unsigned LEB128 varint (fail-closed on truncation / overflow).
///
/// # Errors
/// [`TensorError::Decode`] on a truncated or over-long varint; `field` names it.
pub fn read_uleb128(d: &[u8], o: &mut usize, field: &str) -> Result<u64, TensorError> {
    let mut result: u64 = 0;
    let mut i = 0usize;
    loop {
        if i >= MAX_ULEB128_BYTES {
            return Err(TensorError::Decode(format!("varint overflow [{field}]")));
        }
        let byte = *d
            .get(*o)
            .ok_or_else(|| TensorError::Decode(format!("truncated varint [{field}]")))?;
        *o += 1;
        result |= u64::from(byte & 0x7f) << (7 * i);
        i += 1;
        if byte & 0x80 == 0 {
            break;
        }
    }
    Ok(result)
}

/// Read a zigzag LEB128 varint as a signed value.
///
/// # Errors
/// [`TensorError::Decode`] on a truncated / over-long varint.
pub fn read_i64_zigzag(d: &[u8], o: &mut usize, field: &str) -> Result<i64, TensorError> {
    Ok(unzigzag(read_uleb128(d, o, field)?))
}

/// FNV-1a/64 over a header prefix followed by the payload in one stream.
///
/// The stored checksum of the §R5f codecs always covers the header fields
/// that steer decode (counts, modes, widths, bounds) plus the payload — a
/// mutation anywhere in the covered bytes breaks the checksum. Fail-closed.
#[must_use]
pub fn fnv_cover(prefix: &[u8], body: &[u8]) -> u64 {
    let mut h = crate::composites::FNV_OFFSET_BASIS;
    for &b in prefix.iter().chain(body.iter()) {
        h = crate::prims::fnv_mix(h, b);
    }
    h
}

/// Bytes `{1..=8}` needed to hold `max` (at least one).
#[must_use]
pub fn min_width(max: u64) -> u8 {
    let mut w = 1u8;
    while w < 8 && (max >> (8 * w as u32)) != 0 {
        w += 1;
    }
    w
}

/// Append `v` as `w` little-endian bytes (low `8*w` bits).
#[inline]
pub fn write_fixed(out: &mut Vec<u8>, v: u64, w: u8) {
    for i in 0..w {
        out.push(((v >> (8 * u32::from(i))) & 0xff) as u8);
    }
}

/// Read `w` little-endian bytes into a `u64`.
///
/// # Errors
/// [`TensorError::Decode`] on a short read.
pub fn read_fixed(d: &[u8], o: &mut usize, w: u8, field: &str) -> Result<u64, TensorError> {
    let w = w as usize;
    if *o + w > d.len() {
        return Err(TensorError::Decode(format!("truncated fixed [{field}]")));
    }
    let mut v = 0u64;
    for i in 0..w {
        v |= u64::from(d[*o + i]) << (8 * i);
    }
    *o += w;
    Ok(v)
}
