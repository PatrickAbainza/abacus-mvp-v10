//! Entry model for the atomic operators (generic terms only).
//!
//! Coordinate: mode0 `t` (u64), mode1 `s` (u32), mode2 `r` (u32).
//! Value: exact i64. Locator set: sorted distinct u32 ordinals into the
//! per-event stores (empty for pure-tensor entries). Absent = bottom.

/// Named tensor modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Leading time mode.
    Mode0,
    /// First key mode.
    Mode1,
    /// Second key mode.
    Mode2,
}

/// A 3-mode coordinate triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Coord3 {
    /// Mode-0 coordinate.
    pub t: u64,
    /// Mode-1 coordinate.
    pub s: u32,
    /// Mode-2 coordinate.
    pub r: u32,
}

impl Coord3 {
    /// Build a coordinate triple.
    #[must_use]
    pub const fn new(t: u64, s: u32, r: u32) -> Self {
        Self { t, s, r }
    }
}

/// Sorted distinct locator set (empty for pure-tensor entries).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocSet {
    /// Sorted distinct locators.
    pub locators: Vec<u32>,
}

impl LocSet {
    /// Empty locator set.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Wrap an already-sorted distinct list (debug-checked).
    #[must_use]
    pub fn from_sorted(mut v: Vec<u32>) -> Self {
        v.sort_unstable();
        v.dedup();
        Self { locators: v }
    }

    /// Union of two locator sets (sorted distinct).
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut out = Vec::with_capacity(self.locators.len() + other.locators.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.locators.len() && j < other.locators.len() {
            match self.locators[i].cmp(&other.locators[j]) {
                std::cmp::Ordering::Less => {
                    out.push(self.locators[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(other.locators[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    out.push(self.locators[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&self.locators[i..]);
        out.extend_from_slice(&other.locators[j..]);
        Self { locators: out }
    }

    /// Lowest locator (deterministic tie-break witness).
    #[must_use]
    pub fn lowest(&self) -> Option<u32> {
        self.locators.first().copied()
    }
}

/// One operator entry: coordinate + value + locator evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Coordinate triple.
    pub coord: Coord3,
    /// Exact integer value.
    pub value: i64,
    /// Locator evidence (empty for pure-tensor entries).
    pub locators: LocSet,
}

impl Entry {
    /// Build an entry.
    #[must_use]
    pub fn new(t: u64, s: u32, r: u32, value: i64, locators: LocSet) -> Self {
        Self {
            coord: Coord3::new(t, s, r),
            value,
            locators,
        }
    }

    /// Total entry order: coordinate, then lowest locator.
    fn sort_key(&self) -> (Coord3, Option<u32>) {
        (self.coord, self.locators.lowest())
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

/// Canonical entry support: sorted by (coord, lowest locator), unique
/// coordinates, no zero values (tensor presence = net-nonzero).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Support {
    /// Canonical entries.
    pub entries: Vec<Entry>,
}

impl Support {
    /// Empty support.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Wrap canonical entries (debug-checked order/uniqueness/nonzero).
    #[must_use]
    pub fn from_canonical(entries: Vec<Entry>) -> Self {
        debug_assert!(entries.windows(2).all(|w| w[0] < w[1]));
        debug_assert!(entries.iter().all(|e| e.value != 0));
        Self { entries }
    }
}

/// Reduction monoids. Empty-input identities: sum = 0, count = 0,
/// min / max / any = bottom (absent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monoid {
    /// Checked integer sum (identity 0).
    Sum,
    /// Minimum (identity bottom).
    Min,
    /// Maximum (identity bottom).
    Max,
    /// Existence (identity bottom; witness = lowest locator).
    Any,
    /// Entry count (identity 0).
    Count,
}
