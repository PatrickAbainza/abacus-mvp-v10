//! §R5f codec-integrity suite for `abacus_mvp_v7`: **F_pipeline(X) = F_ref(X)**
//! for each derived compressed codec, plus **D(E(X)) ≡bit X** at every lossless
//! encode/decode boundary.
//!
//! For each codec an **independent reference encoder `F_ref`** — written with a
//! different algorithm from the pipeline (hand-rolled byte-level varint/zigzag
//! writers, manual width packing instead of `varint::write_fixed`) — recomputes
//! the same image; each test requires byte-identical output and panics at the
//! FIRST differing byte. The pipelines (`csf::compress_v2`, `samld::core_pack`,
//! `samld::eval_codec3`) are never trusted as their own oracle.
//!
//! At every lossless boundary the suite additionally requires
//! `E(D(bytes)) == bytes` byte-for-byte, including the recomputed f64 weights
//! (`log1p_amt`, `z_amt`, moments) compared with `f64::to_bits`. No comparison is
//! substituted by a checksum or an aggregate sum.
//!
//! Data: only the existing fixture `tests/fixtures/small.csv` (reused; this test
//! writes no files and stores no duplicate transaction data).
//!
//! Golden-Source citations:
//! * `docs/golden_source_mathematical_spec_constitution.md` §B.2 rows 2 (exact
//!   decimal -> micro), 7/11 (`d_settle`, CSF compress), 14 (CSV build), 17
//!   (artifact I/O), 24/25 (sidecar build/codec), 28 (CLI); laws `R_5`
//!   (fail-closed), `R_7` (determinism), `P2` (`i64` micro-units, scale `10^6`),
//!   `P4` (duplicates kept), `P6` (lossless CSF), `P11` (provenance/weights),
//!   `P12` (determinism), §I H1 (standard library only).
//! * `docs/golden_source_samld_binary_addendum.md` §S4 (`tx_id` sole carrier),
//!   §S5 (`log1p`/`z`/`roundedness` from `amount_micro` + moments), §S6
//!   (sidecar provenance), §S7 (direct core -> CSF).
//! * `docs/golden_source_csf_profile_addendum.md` §R5e (CSV as sole truth) and
//!   §R5f (derived lossless codecs `SAMCSF02`/`SAMCORE02`/`SAMEVL03`, gap
//!   `G-PROFILE-3` reported not invented).
//!
//! Determinism: fixed input, no environment, no RNG, no writes.

use abacus_tensor::coo::{build_from_reader, TxEntry};
use abacus_tensor::csf::{deserialize_csf_v2, serialize_csf_v2, CsfTensor};
use abacus_tensor::samld::{
    convert_reader, deserialize_core, deserialize_core_v2, deserialize_eval,
    deserialize_eval_v2, deserialize_eval_v3, direct_csf, serialize_core, serialize_core_v2,
    serialize_eval, serialize_eval_v2, serialize_eval_v3, CoreRecord, EvalRecord,
};
use std::collections::BTreeMap;
use std::fmt::Debug;

const CSV: &str = include_str!("fixtures/small.csv");

// ---------------------------------------------------------------------------
// Fail-fast comparison helpers (first mismatch panics; no checksums).
// ---------------------------------------------------------------------------

fn seq<T: PartialEq + Debug>(label: &str, got: &[T], want: &[T]) {
    assert_eq!(
        got.len(),
        want.len(),
        "[{label}] length {} != {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            panic!("[{label}] first diff at [{i}]:\n  got  {g:?}\n  want {w:?}");
        }
    }
}

fn bytes_eq(label: &str, got: &[u8], want: &[u8]) {
    if got.len() != want.len() {
        panic!(
            "[{label}] byte length {} != {}",
            got.len(),
            want.len()
        );
    }
    if let Some(i) = got.iter().zip(want.iter()).position(|(g, w)| g != w) {
        panic!(
            "[{label}] first byte diff at {i}: got {:#04x}, want {:#04x}",
            got[i], want[i]
        );
    }
}

fn f64b(label: &str, got: f64, want: f64) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "[{label}] f64 bits differ: got {got}, want {want}"
    );
}
/// Core records: keys/values exact; f64 weights compared bit-for-bit.
fn core_bits(label: &str, got: &[CoreRecord], want: &[CoreRecord]) {
    assert_eq!(
        got.len(),
        want.len(),
        "[{label}] core record count {} != {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if (g.t, g.s, g.r, g.amount_micro, g.roundedness)
            != (w.t, w.s, w.r, w.amount_micro, w.roundedness)
        {
            panic!("[{label}] core[{i}] key/value diff:\n  got  {g:?}\n  want {w:?}");
        }
        if g.log1p_amt.to_bits() != w.log1p_amt.to_bits() {
            panic!(
                "[{label}] core[{i}] log1p_amt bits differ: {} != {}",
                g.log1p_amt.to_bits(),
                w.log1p_amt.to_bits()
            );
        }
        if g.z_amt.to_bits() != w.z_amt.to_bits() {
            panic!(
                "[{label}] core[{i}] z_amt bits differ: {} != {}",
                g.z_amt.to_bits(),
                w.z_amt.to_bits()
            );
        }
    }
}

fn eval_eq(label: &str, got: &[EvalRecord], want: &[EvalRecord]) {
    assert_eq!(
        got.len(),
        want.len(),
        "[{label}] eval record count {} != {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.tx_id != w.tx_id || g.amount_micro != w.amount_micro || g.laundering != w.laundering {
            panic!("[{label}] eval[{i}] numeric diff:\n  got  {g:?}\n  want {w:?}");
        }
        for (name, gs, ws) in [
            ("amount_raw", &g.amount_raw, &w.amount_raw),
            ("laundering_type", &g.laundering_type, &w.laundering_type),
            ("payment_currency", &g.payment_currency, &w.payment_currency),
            (
                "received_currency",
                &g.received_currency,
                &w.received_currency,
            ),
            (
                "sender_bank_location",
                &g.sender_bank_location,
                &w.sender_bank_location,
            ),
            (
                "receiver_bank_location",
                &g.receiver_bank_location,
                &w.receiver_bank_location,
            ),
            ("payment_type", &g.payment_type, &w.payment_type),
        ] {
            if gs != ws {
                panic!("[{label}] eval[{i}].{name}: got {gs:?}, want {ws:?}");
            }
        }
    }
}

fn csf_eq(label: &str, got: &CsfTensor, want: &CsfTensor) {
    seq(&format!("{label}.I0"), &got.mode0_coords, &want.mode0_coords);
    seq(&format!("{label}.P0"), &got.mode0_fptr, &want.mode0_fptr);
    seq(&format!("{label}.I1"), &got.mode1_coords, &want.mode1_coords);
    seq(&format!("{label}.P1"), &got.mode1_fptr, &want.mode1_fptr);
    seq(&format!("{label}.I2"), &got.mode2_coords, &want.mode2_coords);
    seq(&format!("{label}.V"), &got.values, &want.values);
}
// ---------------------------------------------------------------------------
// Independent reference primitives F_ref (§R5f). These deliberately do NOT call
// `abacus_tensor::varint` — the point is a second, independent realization of
// the same declared §R5f wire format.
// ---------------------------------------------------------------------------

/// Independent unsigned LEB128 writer (shift/`while` form).
fn ref_uleb(mut v: u64, out: &mut Vec<u8>) {
    while v > 0x7f {
        out.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Independent zigzag via explicit unsigned casts.
fn ref_zig(v: i64) -> u64 {
    ((v as u64) << 1) ^ ((v >> 63) as u64)
}

fn ref_zig_leb(v: i64, out: &mut Vec<u8>) {
    ref_uleb(ref_zig(v), out);
}

/// Independent minimal-width: byte count holding `max`, by repeated shift-down.
fn ref_width(max: u64) -> u8 {
    let mut n = 1u64;
    let mut m = max;
    while m > 0xff {
        m >>= 8;
        n += 1;
    }
    n as u8
}

/// Independent minimal fixed-width writer (chunked accumulator form).
fn ref_fixed(out: &mut Vec<u8>, v: u64, w: u8) {
    let mut acc = v;
    for _ in 0..w {
        out.push((acc & 0xff) as u8);
        acc >>= 8;
    }
}

fn ref_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn ref_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn ref_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}

fn ref_len_str(out: &mut Vec<u8>, s: &str) {
    ref_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

/// Independent FNV-1a/64 over `prefix ++ body`, written directly from the
/// declared `d_fnv` recurrence (does not call `prims::fnv_mix`).
fn ref_fnv(prefix: &[u8], body: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in prefix.iter().chain(body.iter()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Independent `BTreeMap` regrouping of settled COO (row 11 reference).
fn ref_compress(coo: &[TxEntry]) -> CsfTensor {
    let mut grouped: BTreeMap<u64, BTreeMap<u32, Vec<(u32, i64)>>> = BTreeMap::new();
    for e in coo {
        grouped
            .entry(e.t)
            .or_default()
            .entry(e.s)
            .or_default()
            .push((e.r, e.value));
    }
    let mut i0 = Vec::new();
    let mut p0 = vec![0u64];
    let mut i1 = Vec::new();
    let mut p1 = vec![0u64];
    let mut i2 = Vec::new();
    let mut vv = Vec::new();
    for (t, by_s) in &grouped {
        i0.push(*t);
        for (s, rvs) in by_s {
            i1.push(*s);
            for (r, v) in rvs {
                i2.push(*r);
                vv.push(*v);
            }
            p1.push(i2.len() as u64);
        }
        p0.push(i1.len() as u64);
    }
    CsfTensor {
        mode0_coords: i0,
        mode0_fptr: p0,
        mode1_coords: i1,
        mode1_fptr: p1,
        mode2_coords: i2,
        values: vv,
    }
}

/// §R5f `SAMCORE02` reference encoder. Independently derives each column's
/// encoding, the deterministic `pick` (first candidate wins ties), the header,
/// and the header+payload cover.
fn ref_core_v2(img: &abacus_tensor::samld::CoreImage) -> Vec<u8> {
    let recs = &img.records;
    let n = recs.len();

    // column t: delta-LEB128 vs fixed-min-width.
    let mut t_delta = Vec::new();
    let mut prev = img.t_min;
    for r in recs {
        ref_uleb(r.t - prev, &mut t_delta);
        prev = r.t;
    }
    let t_off: Vec<u64> = recs.iter().map(|r| r.t - img.t_min).collect();
    let t_w = ref_width(t_off.last().copied().unwrap_or(0));
    let mut t_fixed = Vec::new();
    for &v in &t_off {
        ref_fixed(&mut t_fixed, v, t_w);
    }

    // columns s, r: minimal fixed-width codes.
    let s_max = recs.iter().map(|r| u64::from(r.s)).max().unwrap_or(0);
    let r_max = recs.iter().map(|r| u64::from(r.r)).max().unwrap_or(0);
    let s_w = ref_width(s_max);
    let r_w = ref_width(r_max);
    let mut s_fixed = Vec::new();
    let mut r_fixed = Vec::new();
    for r in recs {
        ref_fixed(&mut s_fixed, u64::from(r.s), s_w);
        ref_fixed(&mut r_fixed, u64::from(r.r), r_w);
    }

    // column amount: zigzag-LEB128 vs fixed 8.
    let mut a_zig = Vec::new();
    for r in recs {
        ref_zig_leb(r.amount_micro, &mut a_zig);
    }
    let mut a_fix = Vec::new();
    for r in recs {
        ref_fixed(&mut a_fix, r.amount_micro as u64, 8);
    }

    // `pick`: first candidate wins ties, else the strictly smaller.
    let (t_mode, t_stored_w, t_bytes) = if t_delta.len() <= t_fixed.len() {
        (1u8, 0u8, t_delta)
    } else {
        (0u8, t_w, t_fixed)
    };
    let (a_mode, a_stored_w, a_bytes) = if a_zig.len() <= a_fix.len() {
        (2u8, 0u8, a_zig)
    } else {
        (0u8, 8u8, a_fix)
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&t_bytes);
    payload.extend_from_slice(&s_fixed);
    payload.extend_from_slice(&r_fixed);
    payload.extend_from_slice(&a_bytes);
    for dict in [&img.senders, &img.receivers] {
        ref_u64(&mut payload, dict.len() as u64);
        for s in dict {
            ref_len_str(&mut payload, s);
        }
    }

    let mut head: Vec<u8> = Vec::with_capacity(68);
    head.extend_from_slice(b"SAMCORE2");
    ref_u32(&mut head, 3);
    ref_u64(&mut head, n as u64);
    ref_u64(&mut head, img.t_min);
    ref_u64(&mut head, img.t_max);
    ref_f64(&mut head, img.mean_major);
    ref_f64(&mut head, img.std_major);
    head.extend_from_slice(&[t_mode, 0, 0, a_mode]);
    head.extend_from_slice(&[t_stored_w, s_w, r_w, a_stored_w]);
    let sum = ref_fnv(&head, &payload);

    let mut out = head;
    ref_u64(&mut out, sum);
    out.extend_from_slice(&payload);
    out
}

/// §R5f `SAMCSF02` reference encoder over a `CsfTensor` (same six arrays).
#[allow(clippy::too_many_arguments)]
fn ref_csf_v2(
    csf: &CsfTensor,
    senders: &[String],
    receivers: &[String],
    t_min: u64,
    t_max: u64,
    raw_rows: u64,
) -> Vec<u8> {
    // I0: per-`t` offsets, delta vs fixed.
    let mut i0_delta = Vec::new();
    let mut prev = t_min;
    for &t in &csf.mode0_coords {
        ref_uleb(t - prev, &mut i0_delta);
        prev = t;
    }
    let i0_off: Vec<u64> = csf
        .mode0_coords
        .iter()
        .map(|&t| t.saturating_sub(t_min))
        .collect();
    let i0_w = ref_width(i0_off.last().copied().unwrap_or(0));
    let mut i0_fixed = Vec::new();
    for &v in &i0_off {
        ref_fixed(&mut i0_fixed, v, i0_w);
    }
    let (m0, w0, b0) = if i0_delta.len() <= i0_fixed.len() {
        (1u8, 0u8, i0_delta)
    } else {
        (0u8, i0_w, i0_fixed)
    };

    // P0 / P1: fiber-pointer sizes (delta) vs raw pointers (fixed).
    let enc_ptr = |fptr: &[u64]| -> (u8, u8, Vec<u8>) {
        let n = fptr.len().saturating_sub(1);
        let mut d = Vec::new();
        for i in 0..n {
            ref_uleb(fptr[i + 1] - fptr[i], &mut d);
        }
        let w = ref_width(fptr.last().copied().unwrap_or(0));
        let mut f = Vec::new();
        for &p in fptr {
            ref_fixed(&mut f, p, w);
        }
        if d.len() <= f.len() {
            (1, 0, d)
        } else {
            (0, w, f)
        }
    };
    let (m1, w1, b1) = enc_ptr(&csf.mode0_fptr);
    let (m3, w3, b3) = enc_ptr(&csf.mode1_fptr);

    // I1 / I2: per-fiber strictly-increasing codes; first-of-fiber raw, then delta.
    let enc_codes = |coords: &[u32], fptr: &[u64]| -> (u8, u8, Vec<u8>) {
        let nf = fptr.len().saturating_sub(1);
        let mut d = Vec::new();
        for fi in 0..nf {
            let (s, e) = (fptr[fi] as usize, fptr[fi + 1] as usize);
            let mut p = 0u32;
            for (k, &v) in coords[s..e].iter().enumerate() {
                if k == 0 {
                    ref_uleb(u64::from(v), &mut d);
                } else {
                    ref_uleb(u64::from(v - p), &mut d);
                }
                p = v;
            }
        }
        let w = ref_width(coords.iter().copied().max().unwrap_or(0) as u64);
        let mut f = Vec::new();
        for &v in coords {
            ref_fixed(&mut f, u64::from(v), w);
        }
        if d.len() <= f.len() {
            (1, 0, d)
        } else {
            (0, w, f)
        }
    };
    let (m2, w2, b2) = enc_codes(&csf.mode1_coords, &csf.mode0_fptr);
    let (m4, w4, b4) = enc_codes(&csf.mode2_coords, &csf.mode1_fptr);

    // V: zigzag vs fixed 8.
    let mut v_zig = Vec::new();
    for &v in &csf.values {
        ref_zig_leb(v, &mut v_zig);
    }
    let mut v_fix = Vec::new();
    for &v in &csf.values {
        ref_fixed(&mut v_fix, v as u64, 8);
    }
    let (m5, w5, b5) = if v_zig.len() <= v_fix.len() {
        (2, 0, v_zig)
    } else {
        (0, 8, v_fix)
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&b0);
    payload.extend_from_slice(&b1);
    payload.extend_from_slice(&b2);
    payload.extend_from_slice(&b3);
    payload.extend_from_slice(&b4);
    payload.extend_from_slice(&b5);
    for dict in [senders, receivers] {
        ref_u64(&mut payload, dict.len() as u64);
        for s in dict {
            ref_len_str(&mut payload, s);
        }
    }

    let mut head: Vec<u8> = Vec::with_capacity(84);
    head.extend_from_slice(b"SAMCSF02");
    ref_u32(&mut head, 2);
    ref_u32(&mut head, 6); // P2 canonical scale
    ref_u64(&mut head, csf.mode0_coords.len() as u64);
    ref_u64(&mut head, csf.mode1_coords.len() as u64);
    ref_u64(&mut head, csf.values.len() as u64);
    ref_u64(&mut head, t_min);
    ref_u64(&mut head, t_max);
    ref_u64(&mut head, raw_rows);
    head.extend_from_slice(&[m0, m1, m2, m3, m4, m5]);
    head.extend_from_slice(&[w0, w1, w2, w3, w4, w5]);
    let sum = ref_fnv(&head, &payload);

    let mut out = head;
    ref_u64(&mut out, sum);
    out.extend_from_slice(&payload);
    out
}

/// §R5f `SAMEVL03` reference encoder over `SAMEVL01` records.
fn ref_eval_v3(records: &[EvalRecord]) -> Vec<u8> {
    let n = records.len();
    let cols: Vec<fn(&EvalRecord) -> &str> = vec![
        |r| &r.laundering_type,
        |r| &r.payment_currency,
        |r| &r.received_currency,
        |r| &r.sender_bank_location,
        |r| &r.receiver_bank_location,
        |r| &r.payment_type,
    ];
    let mut body: Vec<u8> = Vec::new();
    // tx_id as delta-zigzag.
    let mut prev = 0i64;
    for r in records {
        ref_zig_leb(r.tx_id as i64 - prev, &mut body);
        prev = r.tx_id as i64;
    }
    for r in records {
        ref_zig_leb(r.amount_micro, &mut body);
    }
    let mut labels = vec![0u8; n.div_ceil(8)];
    for (i, r) in records.iter().enumerate() {
        if r.laundering != 0 {
            labels[i / 8] |= 1 << (i % 8);
        }
    }
    body.extend_from_slice(&labels);
    for r in records {
        ref_len_str(&mut body, &r.amount_raw);
    }
    for get in &cols {
        // Ordered-set dictionary, built by sorting + dedup (exact set operation).
        let mut uniq: Vec<&str> = records.iter().map(|r| get(r)).collect();
        uniq.sort_unstable();
        uniq.dedup();
        ref_u64(&mut body, uniq.len() as u64);
        for s in &uniq {
            ref_len_str(&mut body, s);
        }
        let w: u8 = if uniq.len() <= 256 {
            1
        } else if uniq.len() <= 65_536 {
            2
        } else {
            4
        };
        body.push(w);
        for r in records {
            let c = uniq.iter().position(|&d| d == get(r)).unwrap() as u32;
            match w {
                1 => body.push(c as u8),
                2 => body.extend_from_slice(&(c as u16).to_le_bytes()),
                _ => body.extend_from_slice(&c.to_le_bytes()),
            }
        }
    }
    let mut cov: Vec<u8> = Vec::with_capacity(24);
    cov.extend_from_slice(b"SAMEVL03");
    ref_u32(&mut cov, 3);
    ref_u64(&mut cov, n as u64);
    ref_u32(&mut cov, 6);
    let sum = ref_fnv(&cov, &body);
    let mut out = Vec::with_capacity(32 + body.len());
    out.extend_from_slice(b"SAMEVL03");
    ref_u32(&mut out, 3);
    ref_u64(&mut out, n as u64);
    ref_u64(&mut out, sum);
    ref_u32(&mut out, 6);
    out.extend_from_slice(&body);
    out
}

// ---------------------------------------------------------------------------
// §R5f tests
// ---------------------------------------------------------------------------

/// §R5f `SAMCORE02`: the pipeline image must equal the independent reference
/// encoder byte-for-byte, and decode∘encode must reproduce every record with
/// **bit-exact** recomputed P11 weights (duplicates kept, `R_7` re-emission).
#[test]
fn samcore02_matches_independent_reference_and_roundtrips_bit_exact() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img1 = deserialize_core(&conv.core_bytes).unwrap();

    // SAMLDBIN v1 baseline (row 25) — decode∘encode byte re-emission.
    core_bits("SAMLDBIN v1", &img1.records, &conv.core_records);
    bytes_eq(
        "SAMLDBIN v1 re-emit",
        &serialize_core(
            &img1.records,
            &img1.senders,
            &img1.receivers,
            img1.t_min,
            img1.t_max,
            img1.mean_major,
            img1.std_major,
        ),
        &conv.core_bytes,
    );

    // F_pipeline(X) = F_ref(X)
    let pipe = serialize_core_v2(&img1);
    bytes_eq("SAMCORE02 pipeline vs reference", &pipe, &ref_core_v2(&img1));

    // D(E(X)) ≡bit X
    let img2 = deserialize_core_v2(&pipe).unwrap();
    core_bits("SAMCORE02 roundtrip", &img2.records, &img1.records);
    seq("SAMCORE02 senders", &img2.senders, &img1.senders);
    seq("SAMCORE02 receivers", &img2.receivers, &img1.receivers);
    assert_eq!(img2.t_min, img1.t_min, "SAMCORE02 t_min");
    assert_eq!(img2.t_max, img1.t_max, "SAMCORE02 t_max");
    assert_eq!(img2.record_count, img1.record_count, "SAMCORE02 count");
    f64b("SAMCORE02 mean", img2.mean_major, img1.mean_major);
    f64b("SAMCORE02 std", img2.std_major, img1.std_major);
    bytes_eq("SAMCORE02 re-emit", &serialize_core_v2(&img2), &pipe);

    // P4: duplicates retained (no aggregation, no dedup).
    assert_eq!(img2.records.len(), conv.core_records.len(), "dups kept");
    assert!(
        img2.records.windows(2).all(|w| w[0] <= w[1]),
        "(t,s,r) canonical non-decreasing order retained"
    );
}

/// §R5f `SAMCSF02`: pipeline image equals the independent reference encoder
/// byte-for-byte; decode∘encode reproduces all six fiber arrays with `nnz`
/// unchanged (no aggregation); `R_7` re-emission is byte-identical.
#[test]
fn samcsf02_matches_independent_reference_and_preserves_structure() {
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let (csf, settled) = direct_csf(&img).unwrap();

    // The CSF the codec re-encodes is itself canonical (row 11 reference).
    // Note P4: `direct_csf` settles the projected rows first, so the canonical
    // tensor is the compression of the SETTLED entries, not of the raw
    // projection — the two differ whenever a coordinate repeats.
    let raw: Vec<TxEntry> = img
        .records
        .iter()
        .map(|r| TxEntry::new(r.t, r.s, r.r, r.amount_micro))
        .collect();
    assert_eq!(raw.len(), img.records.len(), "projection is 1:1");
    csf_eq("§S7 direct csf", &csf, &ref_compress(&settled));
    assert_eq!(csf.nnz(), settled.len(), "§S7 nnz");
    assert!(csf.check_invariants(), "P6 fiber invariants");
    // Canonical CSF coordinates are unique (P3/P6); repeats were settled away.
    let mut keys: Vec<(u64, u32, u32)> =
        abacus_tensor::csf::decode_csf(&csf).iter().map(|e| (e.t, e.s, e.r)).collect();
    let before = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), before, "canonical CSF coordinates are unique");

    // F_pipeline(X) = F_ref(X)
    let pipe = serialize_csf_v2(
        &csf,
        &conv.senders,
        &conv.receivers,
        conv.t_min,
        conv.t_max,
        build.raw_rows,
    );
    bytes_eq(
        "SAMCSF02 pipeline vs reference",
        &pipe,
        &ref_csf_v2(
            &csf,
            &conv.senders,
            &conv.receivers,
            conv.t_min,
            conv.t_max,
            build.raw_rows,
        ),
    );

    // D(E(X)) ≡ X on all six arrays + dictionaries + provenance footer.
    let back = deserialize_csf_v2(&pipe).unwrap();
    csf_eq("SAMCSF02 roundtrip", &back.csf, &csf);
    seq("SAMCSF02 senders", &back.senders, &conv.senders);
    seq("SAMCSF02 receivers", &back.receivers, &conv.receivers);
    assert_eq!(back.t_min, conv.t_min, "SAMCSF02 t_min");
    assert_eq!(back.t_max, conv.t_max, "SAMCSF02 t_max");
    assert_eq!(back.raw_rows, build.raw_rows, "SAMCSF02 raw_rows");
    assert_eq!(back.header.scale_exp, 6, "SAMCSF02 P2 scale");
    assert_eq!(back.csf.nnz(), csf.nnz(), "nnz unchanged (no aggregation)");
    assert!(back.csf.check_invariants(), "P6 fiber invariants after decode");
    bytes_eq(
        "SAMCSF02 re-emit",
        &serialize_csf_v2(
            &back.csf,
            &back.senders,
            &back.receivers,
            back.t_min,
            back.t_max,
            back.raw_rows,
        ),
        &pipe,
    );
}

/// §R5f `SAMEVL03`: pipeline image equals the independent reference encoder
/// byte-for-byte; `Decode_v3(Encode_v3(S)) = S` over all ten P11 attributes;
/// `R_7` re-emission byte-identical. Also covers the `SAMEVL01`/`SAMEVL02`
/// boundaries that `SAMEVL03` extends.
#[test]
fn samevl03_matches_independent_reference_and_roundtrips_losslessly() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();

    // SAMEVL01 v1 (row 25).
    let ev1 = deserialize_eval(&conv.eval_bytes).unwrap();
    eval_eq("SAMEVL01 v1", &ev1, &conv.eval_records);
    bytes_eq("SAMEVL01 v1 re-emit", &serialize_eval(&ev1), &conv.eval_bytes);

    // SAMEVL02 v2 (§R5d).
    let b2 = serialize_eval_v2(&ev1).unwrap();
    let ev2 = deserialize_eval_v2(&b2).unwrap();
    eval_eq("SAMEVL02 v2", &ev2, &ev1);
    bytes_eq(
        "SAMEVL02 v2 re-emit",
        &serialize_eval_v2(&ev2).unwrap(),
        &b2,
    );

    // SAMEVL03 v3 (§R5f): F_pipeline = F_ref, then D(E(S)) = S.
    let pipe = serialize_eval_v3(&ev1).unwrap();
    bytes_eq("SAMEVL03 pipeline vs reference", &pipe, &ref_eval_v3(&ev1));
    let ev3 = deserialize_eval_v3(&pipe).unwrap();
    eval_eq("SAMEVL03 v3", &ev3, &ev1);
    bytes_eq(
        "SAMEVL03 v3 re-emit",
        &serialize_eval_v3(&ev3).unwrap(),
        &pipe,
    );

    // §S4: `tx_id` is the sole carrier and is a bijection onto `{0..n-1}`
    // (set-operation rule: exact index-set equality).
    let mut ids: Vec<u64> = ev3.iter().map(|e| e.tx_id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids,
        (0..ev1.len() as u64).collect::<Vec<u64>>(),
        "tx_id exact index-set equality"
    );
    // Positional core <-> sidecar amount agreement.
    for (i, e) in ev3.iter().enumerate() {
        assert_eq!(
            e.amount_micro, conv.core_records[i].amount_micro,
            "sidecar/core amount at {i}"
        );
    }
}

/// §R5f non-aggregation guarantee: the codecs must never drop a coordinate or a
/// value slot. `nnz`, record counts, and the total are invariant across the
/// boundary (no summing into fewer slots, no dedup).
#[test]
fn derived_codecs_are_non_aggregating_and_structure_preserving() {
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let (csf, _) = direct_csf(&img).unwrap();

    let csf2 = deserialize_csf_v2(&serialize_csf_v2(
        &csf,
        &conv.senders,
        &conv.receivers,
        conv.t_min,
        conv.t_max,
        build.raw_rows,
    ))
    .unwrap();
    assert_eq!(csf2.csf.nnz(), csf.nnz(), "CSF nnz invariant");
    assert_eq!(
        csf2.csf.values.len(),
        csf.values.len(),
        "value slots invariant"
    );

    let core2 = deserialize_core_v2(&serialize_core_v2(&img)).unwrap();
    assert_eq!(
        core2.records.len(),
        img.records.len(),
        "core records invariant (P4 duplicates kept)"
    );
    assert_eq!(
        core2.records.iter().map(|r| r.amount_micro).sum::<i64>(),
        img.records.iter().map(|r| r.amount_micro).sum::<i64>(),
        "amount total invariant"
    );

    let ev1 = deserialize_eval(&conv.eval_bytes).unwrap();
    let ev3 = deserialize_eval_v3(&serialize_eval_v3(&ev1).unwrap()).unwrap();
    assert_eq!(ev3.len(), ev1.len(), "sidecar record count invariant");
}

/// §R5f `R_5` fail-closed: every codec must reject a mutated byte in the header
/// or in the payload (the checksum covers the steering fields plus payload).
#[test]
fn derived_codecs_fail_closed_on_corruption() {
    let conv = convert_reader(CSV.as_bytes()).unwrap();
    let img = deserialize_core(&conv.core_bytes).unwrap();
    let ev1 = deserialize_eval(&conv.eval_bytes).unwrap();
    let (csf, _) = direct_csf(&img).unwrap();

    struct Case {
        name: &'static str,
        bytes: Vec<u8>,
    }
    let cases = vec![
        Case {
            name: "SAMCSF02",
            bytes: serialize_csf_v2(
                &csf,
                &conv.senders,
                &conv.receivers,
                conv.t_min,
                conv.t_max,
                img.records.len() as u64,
            ),
        },
        Case {
            name: "SAMCORE02",
            bytes: serialize_core_v2(&img),
        },
        Case {
            name: "SAMEVL03",
            bytes: serialize_eval_v3(&ev1).unwrap(),
        },
    ];

    for Case { name, bytes } in &cases {
        let decodes = |b: &[u8]| match *name {
            "SAMCSF02" => deserialize_csf_v2(b).is_ok(),
            "SAMCORE02" => deserialize_core_v2(b).is_ok(),
            _ => deserialize_eval_v3(b).is_ok(),
        };
        assert!(decodes(bytes), "[{name}] clean image must decode");
        // Mutate one byte in the header (steering field) and in the payload.
        for pos in [2usize, bytes.len() / 2, bytes.len() - 1] {
            let mut bad = bytes.clone();
            bad[pos] ^= 0xFF;
            assert!(
                !decodes(&bad),
                "[{name}] must fail closed after mutating byte {pos}"
            );
        }
    }
}
