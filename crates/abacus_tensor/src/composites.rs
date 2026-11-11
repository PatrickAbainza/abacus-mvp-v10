//! Named intermediate composites — the layer `𝒟` of §B.1.
//!
//! Each composite is realized **exactly once** here (or, for the one whose
//! input is ledger state, once at its carrier site), replacing the several
//! independent realizations recorded in §H.2/K6.
//!
//! ```text
//! composite    realized by                            §B.1 decomposition
//! d_lex        Ord for Coord / TxEntry            R4(a01(t),a01(s),a01(r))  (P1)
//! d_horner     horner_digit()                     a02(a04(x,10), d)
//! d_round      round_half_up()                    a01(d7 >= '5') + R2 increment
//! d_settle     settle()                           a10 + a01 + a02 + a15 + R3 + R6
//! d_merge2     merge2() / support_diff()          R3 + a01 + a02 + a15
//! d_fnv        fnv1a64()                          a06.1 + a06.2 over bytes (R3)
//! d_hinnant    days_since_epoch()/epoch_seconds() a07 + a09 + a18 + R2/R3
//! d_cas        Ledger::ledger_op()                a16.1 + a01 + a17 + R2 (ingest::ledger)
//! d_delta      delta()                            a03 + a14.1              (P8)
//! d_inv        strictly_increasing()/is_canonical()/check_invariants()
//! ```
//!
//! `d_cas` is realized in [`crate::ingest::ledger`]: its carrier is ledger
//! state, which is not a tensor value (§7.2). It is still realized once.
//!
//! The CSV line splitter of rows 14 and 24 is one composite with one
//! realization here ([`split_csv_line`], §H.2/K6) — not two.

use crate::coo::entry::TxEntry;
use crate::prims::{add, mul, sub};
use crate::TensorError;

/// FNV-1a/64 offset basis (§B.1 `d_fnv`, specification data).
pub const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;

/// `d_fnv` — deterministic provenance checksum over a finite byte sequence.
#[must_use]
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for &b in data {
        h = crate::prims::fnv_mix(h, b);
    }
    h
}

/// `d_horner` — digit accumulation `x  10x + d` (checked, `R5`).
///
/// # Errors
/// [`TensorError::Range`] if the exact value is not representable.
pub fn horner_digit(acc: i64, digit: u8, what: &str) -> Result<i64, TensorError> {
    add(mul(acc, 10, what)?, i64::from(digit - b'0'), what)
}

/// `d_round` — half-up at the 7th fractional digit (P2): `a_{01}(d7 ≥ '5')`
/// then an `R2`-dispatched increment of the accumulated micro-unit fraction.
/// Inputs with ≤6 fractional digits are exact and `seventh` is `None`.
///
/// # Errors
/// [`TensorError::Range`] if the increment is not representable.
pub fn round_half_up(micro: i64, seventh: Option<u8>, what: &str) -> Result<i64, TensorError> {
    match seventh {
        Some(d7) if d7 >= b'5' => add(micro, 1, what),
        _ => Ok(micro),
    }
}

/// `d_settle` — settling (P4): sort by `prec_3` (`d_lex`), sum values at equal
/// coordinates (`a_{02}`), drop zero sums (`R_6`). Output is the canonical
/// entry sequence of P3 (sorted, unique coordinates, all values nonzero).
///
/// Equal-key accumulation follows the order the sort leaves entries in and no
/// value tie-breaker is added; `K13` is preserved exactly as recorded — the
/// determinism of `R_7` is qualified there, never repaired here.
///
/// # Errors
/// [`TensorError::Range`] if any coordinate sum is not representable.
pub fn settle(mut v: Vec<TxEntry>) -> Result<Vec<TxEntry>, TensorError> {
    if v.is_empty() {
        return Ok(v);
    }
    v.sort(); // a10 under d_lex (P1)
    let mut out: Vec<TxEntry> = Vec::with_capacity(v.len());
    let mut it = v.into_iter();
    let mut cur = match it.next() {
        Some(first) => first,
        None => return Ok(out),
    };
    for e in it {
        if e.t == cur.t && e.s == cur.s && e.r == cur.r {
            cur.value = add(cur.value, e.value, "settle aggregation (P4/Def 4.2)")?;
        } else {
            if cur.value != 0 {
                out.push(cur); // R6 zero elimination
            }
            cur = e;
        }
    }
    if cur.value != 0 {
        out.push(cur);
    }
    Ok(out)
}

/// `d_merge2` — pointwise addition of two canonical sequences (P5): one
/// ordered two-pointer merge, equal coordinates combined by `a_{02}`, zero
/// sums dropped (`R_6`).
///
/// # Errors
/// [`TensorError::Range`] if an equal-coordinate sum is not representable.
pub fn merge2(a: &[TxEntry], b: &[TxEntry]) -> Result<Vec<TxEntry>, TensorError> {
    let mut out: Vec<TxEntry> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match coord_key(a[i]).cmp(&coord_key(b[j])) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let s = add(a[i].value, b[j].value, "pointwise sum (P5/Def 9.3)")?;
                if s != 0 {
                    out.push(TxEntry {
                        t: a[i].t,
                        s: a[i].s,
                        r: a[i].r,
                        value: s,
                    });
                }
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    Ok(out)
}

/// Coordinate projection used by the merge/diff comparisons (`a_{14}.2`).
#[inline]
fn coord_key(e: TxEntry) -> (u64, u32, u32) {
    (e.t, e.s, e.r)
}

/// `d_merge2` (comparison-only variant, row 22) — support difference between
/// two canonical sequences: `(changed, added, removed)` coordinate counts.
#[must_use]
pub fn support_diff(old: &[TxEntry], new: &[TxEntry]) -> (usize, usize, usize) {
    let (mut i, mut j) = (0usize, 0usize);
    let (mut changed, mut added, mut removed) = (0usize, 0usize, 0usize);
    while i < old.len() && j < new.len() {
        match coord_key(old[i]).cmp(&coord_key(new[j])) {
            std::cmp::Ordering::Less => {
                removed += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                added += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if old[i].value != new[j].value {
                    changed += 1;
                }
                i += 1;
                j += 1;
            }
        }
    }
    removed += old.len() - i;
    added += new.len() - j;
    (changed, added, removed)
}

/// `d_delta` — elementary delta `Δ = a_new − a_old` (P8, Thm 8.1), then
/// `a_{14}.1` record construction at the caller's coordinate.
///
/// # Errors
/// [`TensorError::Range`] if the exact difference is not representable.
pub fn delta(a_new: i64, a_old: i64) -> Result<i64, TensorError> {
    sub(a_new, a_old, "delta a_new − a_old (P8)")
}

/// `d_hinnant` step 1 — days since 1970-01-01 for a civil date (era /
/// day-of-era arithmetic: `a_{07}` Euclidean division and remainder over
/// `a_{09}` word arithmetic and specification constants; pre-epoch rejected).
///
/// `K2` is preserved as recorded: whether the validated date-field domain
/// excludes every intermediate overflow is not established here. This is the
/// exact integer computation on the domain where the machine realization
/// agrees with it.
///
/// # Errors
/// [`TensorError::Parse`] on an out-of-range month/day.
pub fn days_since_epoch(y: i64, m: u32, d: u32) -> Result<i64, TensorError> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(TensorError::Parse(format!("bad date {y}-{m}-{d}")));
    }
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2.rem_euclid(400) as u64;
    let mp = u64::from(m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + u64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Ok(era * 146097 + doe as i64 - 719468)
}

/// `d_hinnant` step 2 — `"HH:MM:SS"` + `"YYYY-MM-DD"` to seconds since epoch
/// (`a_{18}.1`/`a_{18}.3` field parsing, `R2` validation, `a_{09}` constants).
///
/// # Errors
/// [`TensorError::Parse`] (`E_parse`, K11) on malformed or pre-epoch input.
pub fn epoch_seconds(time: &str, date: &str) -> Result<u64, TensorError> {
    let tb: Vec<&str> = time.trim().split(':').collect();
    let db: Vec<&str> = date.trim().split('-').collect();
    if tb.len() != 3 || db.len() != 3 {
        return Err(TensorError::Parse(format!(
            "bad datetime {time:?} {date:?}"
        )));
    }
    let bad = |_| TensorError::Parse(format!("bad datetime {time:?} {date:?}"));
    let hh: u64 = tb[0].parse().map_err(bad)?;
    let mm: u64 = tb[1].parse().map_err(bad)?;
    let ss: u64 = tb[2].parse().map_err(bad)?;
    let yy: i64 = db[0].parse().map_err(bad)?;
    let mo: u32 = db[1].parse().map_err(bad)?;
    let dd: u32 = db[2].parse().map_err(bad)?;
    if hh > 23 || mm > 59 || ss > 59 {
        return Err(TensorError::Parse(format!("bad time {time:?}")));
    }
    let days = days_since_epoch(yy, mo, dd)?;
    if days < 0 {
        return Err(TensorError::Parse(format!("pre-epoch date {date:?}")));
    }
    Ok(days as u64 * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// CSV line splitter (rows 14 and 24) — one realization; quoted fields with
/// embedded commas are handled. Each input byte is treated as a character,
/// which is exactly the recorded `K7` behavior for ASCII dataset fields.
#[must_use]
pub fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut it = line.as_bytes().iter().peekable();
    while let Some(&b) = it.next() {
        match b {
            b'"' => {
                if in_q && it.peek() == Some(&&b'"') {
                    cur.push('"');
                    it.next();
                } else {
                    in_q = !in_q;
                }
            }
            b',' if !in_q => fields.push(std::mem::take(&mut cur)),
            _ => cur.push(b as char),
        }
    }
    fields.push(cur);
    fields
}

/// `d_inv` — strict increase under one of the declared orders (P1/P3/P6): one
/// `R_3` adjacent-window scan over `a_{01}`. Used by the canonical-COO check,
/// the CSF fiber-sortedness check (Thm 5.3 consequence) and the sidecar
/// canonical-order check. Realized once.
#[must_use]
pub fn strictly_increasing<T: Ord>(xs: &[T]) -> bool {
    xs.windows(2).all(|w| w[0] < w[1])
}
