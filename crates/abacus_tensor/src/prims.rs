//! Boundary primitives `𝒫_boundary` actually realized by this crate (§A.1).
//!
//! Only the members the derivations of §B.2 use are realized here; everything
//! else in `_boundary` is used as a language/machine operation directly (no
//! code is required to "realize" `a_{01}`, `a_{10}`, `a_{11}`, `a_{12}`, `a_{15}`,
//! `a_{16}`, `a_{17}`, `a_{18}.1`–`.3`, `a_{19}`, `a_{20}`).
//!
//! ```text
//! member        realized as                     used by (§B.2 rows)
//! a_02          add()      checked i64 +        rows 2,5,7,10,17,21 (+ d_settle)
//! a_03          sub()      checked i64 -        row 20 (d_delta, P8/Thm 8.1)
//! a_04          mul()      checked k*x          row 2 (k=10, k=10^6), row 17 bounds
//! a_05          neg()      checked i64 -x       row 2 (sign), row 8 (tensor negation)
//! a_06.1/.2     fnv_mix()  wrapping mul + xor   rows 17, 25 (d_fnv)
//! a_13.1        write_u32/write_u64/write_i64   rows 17, 25 fixed-order emission
//! a_13.2        read_u32/read_u64/read_i64      rows 17, 25 fixed-order decode
//! a_18.4        read_len_str() UTF-8 validation rows 17 (dicts), 25 (strings)
//! ```
//!
//! `R5` (fail-closed error propagation) is realized **once**, here: every
//! partial primitive returns `Result<_, TensorError>` and no production caller
//! unwraps. `a_{06}` wrapping/XOR and `a_{09}` unchecked word arithmetic never
//! touch the value path (P2, K2); `a_{10}`/`a_{11}`/`a_{12}` are the
//! standard-library slice operations used directly at their call sites.

use crate::TensorError;

/// `a_{02}` — checked addition. Defined iff the exact sum lies in
/// `[-2^63, 2^63 - 1]`; never wraps, never saturates (P2).
///
/// # Errors
/// [`TensorError::Range`] (`E_rng`) when the exact sum is not representable.
pub fn add(a: i64, b: i64, what: &str) -> Result<i64, TensorError> {
    a.checked_add(b)
        .ok_or_else(|| TensorError::Range(format!("{what}: i64 addition overflow (a02/P2)")))
}

/// `a_{03}` — checked subtraction. Value-path use is the elementary delta
/// `Δ = a_new − a_old` (P8, row 20).
///
/// # Errors
/// [`TensorError::Range`] when the exact difference is not representable.
pub fn sub(a: i64, b: i64, what: &str) -> Result<i64, TensorError> {
    a.checked_sub(b)
        .ok_or_else(|| TensorError::Range(format!("{what}: i64 subtraction overflow (a03/P2)")))
}

/// `a_{04}` — checked multiplication with an explicit multiplier `k`
/// (instantiations: `k = 10` digit shift, `k = 10^6` scale embedding, payload
/// length bounds on `usize`).
///
/// # Errors
/// [`TensorError::Range`] when the exact product is not representable.
pub fn mul(x: i64, k: i64, what: &str) -> Result<i64, TensorError> {
    x.checked_mul(k)
        .ok_or_else(|| TensorError::Range(format!("{what}: i64 multiplication overflow (a04/P2)")))
}

/// `a_{05}` — checked negation. Undefined exactly on `-2^63` (minimum i64).
///
/// # Errors
/// [`TensorError::Range`] when `v == i64::MIN`.
pub fn neg(v: i64, what: &str) -> Result<i64, TensorError> {
    v.checked_neg()
        .ok_or_else(|| TensorError::Range(format!("{what}: negation of i64::MIN (a05/P2)")))
}

/// `a_{06}.1` + `a_{06}.2` — one FNV round of the checksum core: XOR then
/// wrapping multiplication mod `2^64`. Forbidden on the value path (P2, P11);
/// used only inside `d_fnv` (rows 17, 25).
#[inline]
#[must_use]
pub fn fnv_mix(h: u64, byte: u8) -> u64 {
    (h ^ u64::from(byte)).wrapping_mul(0x100000001b3)
}

/// `a_{13}.1` — encode `u32` little-endian into a byte sequence.
#[inline]
pub fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// `a_{13}.1` — encode `u64` little-endian into a byte sequence.
#[inline]
pub fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// `a_{13}.1` — encode `i64` little-endian (two's complement) into a sequence.
#[inline]
pub fn write_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// `a_{13}.2` — decode a `u32` from an exact-width little-endian field.
///
/// # Errors
/// [`TensorError::Decode`] (`E_dec`) on a short read; `field` classifies it.
pub fn read_u32(d: &[u8], o: &mut usize, field: &str) -> Result<u32, TensorError> {
    if *o + 4 > d.len() {
        return Err(TensorError::Decode(format!("truncated {field}")));
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&d[*o..*o + 4]);
    *o += 4;
    Ok(u32::from_le_bytes(b))
}

/// `a_{13}.2` — decode a `u64` from an exact-width little-endian field.
///
/// # Errors
/// [`TensorError::Decode`] (`E_dec`) on a short read; `field` classifies it.
pub fn read_u64(d: &[u8], o: &mut usize, field: &str) -> Result<u64, TensorError> {
    if *o + 8 > d.len() {
        return Err(TensorError::Decode(format!("truncated {field}")));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[*o..*o + 8]);
    *o += 8;
    Ok(u64::from_le_bytes(b))
}

/// `a_{13}.2` — decode an `i64` from an exact-width little-endian field.
///
/// # Errors
/// [`TensorError::Decode`] (`E_dec`) on a short read.
pub fn read_i64(d: &[u8], o: &mut usize, field: &str) -> Result<i64, TensorError> {
    Ok(read_u64(d, o, field)? as i64)
}

/// `a_{18}.2` + `a_{13}.1` — emit a length-prefixed byte string in fixed order.
#[inline]
pub fn write_len_str(out: &mut Vec<u8>, s: &str) {
    write_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

/// `a_{18}.4` — read a length-prefixed string and validate UTF-8
/// (fail-closed, `R5`; K7 records that CSV ingest treats bytes as characters
/// and that UTF-8 is validated only at deserialization).
///
/// # Errors
/// [`TensorError::Decode`] on a short read, over-cap length, or bad UTF-8.
pub fn read_len_str(d: &[u8], o: &mut usize, cap: u32, field: &str) -> Result<String, TensorError> {
    let len = read_u32(d, o, field)?;
    if len > cap {
        return Err(TensorError::Decode(format!("oversized {field} string")));
    }
    let len = len as usize;
    if *o + len > d.len() {
        return Err(TensorError::Decode(format!("truncated {field} string")));
    }
    let s = std::str::from_utf8(&d[*o..*o + len])
        .map_err(|_| TensorError::Decode(format!("bad {field} utf8")))?
        .to_string();
    *o += len;
    Ok(s)
}
