//! Row 1 (§B.2) + P1: totally ordered coordinate triples, `d_lex` = `R_4` over
//! `a_{01}` on `(t, s, r)`.
//!
//! `TimeSec` (u64 seconds since epoch), `SenderId`/`ReceiverId` (u32 dictionary
//! codes). The distinct coordinate sets `T`, `S`, `R` of A1 are retained as
//! distinct types even though two share a machine representation — collapsing
//! `S` and `R` would collapse a declared mathematical distinction.

/// Time coordinate: seconds since Unix epoch (A1 set `T`, u64).
pub type TimeSec = u64;
/// Sender dictionary code (A1 set `S`, u32).
pub type SenderId = u32;
/// Receiver dictionary code (A1 set `R`, u32).
pub type ReceiverId = u32;

/// Canonical coordinate triple, ordered `(t, s, r)` (P1, row 1).
///
/// The derived `Ord` is exactly `d_lex`: `R_4` lexicographic priority assembly
/// over three `a_{01}` machine comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Coord {
    /// `t`: primary sparse mode (seconds since epoch).
    pub t: TimeSec,
    /// `s`: secondary mode (sender dictionary code).
    pub s: SenderId,
    /// `r`: tertiary mode (receiver dictionary code).
    pub r: ReceiverId,
}

impl Coord {
    /// Construct a coordinate triple (`a_{14}.1`).
    #[inline]
    #[must_use]
    pub const fn new(t: TimeSec, s: SenderId, r: ReceiverId) -> Self {
        Self { t, s, r }
    }

    /// Projection to the comparison key used by the merge composites
    /// (`a_{14}.2`, `a_{01}`).
    #[inline]
    #[must_use]
    pub const fn key(self) -> (TimeSec, SenderId, ReceiverId) {
        (self.t, self.s, self.r)
    }
}
