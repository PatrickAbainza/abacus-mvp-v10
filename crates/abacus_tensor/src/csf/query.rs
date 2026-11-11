//! Row 16 (§B.2) + P1/P6: read-only queries over CSF.
//!
//! `a_{12}` (partition point) locates the mode-0 time-slice boundary,
//! `a_{11}` (binary search) locates fibers within the `prec_3` order, `a_{15}`
//! zips and scans. No algebra: every read is a regroup of stored values
//! (P6), and an absent coordinate is reported absent, never fabricated (P3).

use crate::coo::entry::TxEntry;
use crate::csf::compress::CsfTensor;

/// Borrowed time-slice view: all entries with `t ∈ [t_start, t_end)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSlice {
    /// Slice start (inclusive).
    pub t_start: u64,
    /// Slice end (exclusive).
    pub t_end: u64,
    /// Matching canonical entries in `prec_3` order.
    pub entries: Vec<TxEntry>,
}

/// Row 16 — mode-0 restriction `T|_[t_start, t_end)` (half-open, P1 order).
#[must_use]
pub fn time_slice(csf: &CsfTensor, t_start: u64, t_end: u64) -> TimeSlice {
    let mut entries = Vec::new();
    if t_start < t_end && !csf.mode0_coords.is_empty() {
        let lo = csf.mode0_coords.partition_point(|&t| t < t_start);
        let hi = csf.mode0_coords.partition_point(|&t| t < t_end);
        for i in lo..hi {
            let t = csf.mode0_coords[i];
            let (js, je) = (csf.mode0_fptr[i] as usize, csf.mode0_fptr[i + 1] as usize);
            for j in js..je {
                let s = csf.mode1_coords[j];
                let (ks, ke) = (csf.mode1_fptr[j] as usize, csf.mode1_fptr[j + 1] as usize);
                for k in ks..ke {
                    entries.push(TxEntry {
                        t,
                        s,
                        r: csf.mode2_coords[k],
                        value: csf.values[k],
                    });
                }
            }
        }
    }
    TimeSlice {
        t_start,
        t_end,
        entries,
    }
}

/// Row 16 — exact point read `T(t, s, r)`; `0` iff the coordinate is absent.
#[must_use]
pub fn lookup(csf: &CsfTensor, t: u64, s: u32, r: u32) -> i64 {
    let Ok(i) = csf.mode0_coords.binary_search(&t) else {
        return 0;
    };
    let (js, je) = (csf.mode0_fptr[i] as usize, csf.mode0_fptr[i + 1] as usize);
    let Ok(rel) = csf.mode1_coords[js..je].binary_search(&s) else {
        return 0;
    };
    let j = js + rel;
    let (ks, ke) = (csf.mode1_fptr[j] as usize, csf.mode1_fptr[j + 1] as usize);
    match csf.mode2_coords[ks..ke].binary_search(&r) {
        Ok(rel2) => csf.values[ks + rel2],
        Err(_) => 0,
    }
}

/// Row 16 — sender-at-time view `(t, s) → [(r, v)]` in `prec_3` order.
#[must_use]
pub fn sender_at_time(csf: &CsfTensor, t: u64, s: u32) -> Vec<(u32, i64)> {
    let Ok(i) = csf.mode0_coords.binary_search(&t) else {
        return Vec::new();
    };
    let (js, je) = (csf.mode0_fptr[i] as usize, csf.mode0_fptr[i + 1] as usize);
    let Ok(rel) = csf.mode1_coords[js..je].binary_search(&s) else {
        return Vec::new();
    };
    let j = js + rel;
    let (ks, ke) = (csf.mode1_fptr[j] as usize, csf.mode1_fptr[j + 1] as usize);
    csf.mode2_coords[ks..ke]
        .iter()
        .zip(csf.values[ks..ke].iter())
        .map(|(&r, &v)| (r, v))
        .collect()
}

/// Row 16 — receiver-at-time view `(t, r) → [(s, v)]`: `a_{11}` on the
/// `t`-fiber then an `R_3` filter over sibling `s`-fibers (CSF is t-major).
#[must_use]
pub fn receiver_at_time(csf: &CsfTensor, t: u64, r: u32) -> Vec<(u32, i64)> {
    let Ok(i) = csf.mode0_coords.binary_search(&t) else {
        return Vec::new();
    };
    let (js, je) = (csf.mode0_fptr[i] as usize, csf.mode0_fptr[i + 1] as usize);
    let mut out = Vec::new();
    for j in js..je {
        let s = csf.mode1_coords[j];
        let (ks, ke) = (csf.mode1_fptr[j] as usize, csf.mode1_fptr[j + 1] as usize);
        for k in ks..ke {
            if csf.mode2_coords[k] == r {
                out.push((s, csf.values[k]));
            }
        }
    }
    out
}
