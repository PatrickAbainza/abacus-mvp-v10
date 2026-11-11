//! Row 1 and P1/P3: raw (`Def 4.1`-style) and canonical entry sequences.
//!
//! `TxEntry` is the canonical record `{t: u64, s: u32, r: u32, value: i64}`
//! (row 1 coordinate construction + `a_{14}.1` record construction). Its `Ord`
//! is `d_lex` (P1): `R_4` over `t`, then `s`, then `r` — the *only* canonical
//! coordinate order in the system. `RawEntry` is the pre-canonical form of
//! P4: duplicates and zeros permitted.

use crate::tensor::coordinate::Coord;

/// Canonical COO entry: one `(t, s, r, value)` tuple, value in micro-units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TxEntry {
    /// `t`: seconds since epoch (primary mode).
    pub t: u64,
    /// `s`: sender dictionary code (secondary mode).
    pub s: u32,
    /// `r`: receiver dictionary code (tertiary mode).
    pub r: u32,
    /// Exact amount in micro-units (A2; never float).
    pub value: i64,
}

impl TxEntry {
    /// Construct an entry (`a_{14}.1`).
    #[inline]
    #[must_use]
    pub const fn new(t: u64, s: u32, r: u32, value: i64) -> Self {
        Self { t, s, r, value }
    }

    /// Row 1 — entry-to-coordinate projection (`a_{14}.2`).
    #[inline]
    #[must_use]
    pub const fn coord(&self) -> Coord {
        Coord {
            t: self.t,
            s: self.s,
            r: self.r,
        }
    }
}

impl PartialOrd for TxEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TxEntry {
    /// P1 / row 1: `prec_3` — `t` first, then `s`, then `r`.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.coord().cmp(&other.coord())
    }
}

/// Raw ingestion entry (P4 input): duplicates and zero values permitted,
/// arrival order meaningful (§9.1 Def 9.1 frontier).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawEntry {
    /// `t`: seconds since epoch.
    pub t: u64,
    /// `s`: sender dictionary code.
    pub s: u32,
    /// `r`: receiver dictionary code.
    pub r: u32,
    /// Observed amount in micro-units.
    pub value: i64,
}

impl From<RawEntry> for TxEntry {
    fn from(e: RawEntry) -> Self {
        Self {
            t: e.t,
            s: e.s,
            r: e.r,
            value: e.value,
        }
    }
}
