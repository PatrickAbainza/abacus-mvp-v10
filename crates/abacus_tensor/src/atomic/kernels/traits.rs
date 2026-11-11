//! Single kernel interface for all (operator, structure) pairs.
//!
//! Work counters (entries scanned, seeks) ride every call so benchmarks
//! report measured work, never claims.

use super::super::reference::{Coverage, MapFn, MergeFn, Predicate, ReducedRow, Theta};
use super::super::types::{Mode, Monoid, Support};
use super::super::Outcome;

/// Work counters for one kernel call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelStats {
    /// Entries physically scanned.
    pub scanned: u64,
    /// Seek operations performed (`partition_point` / binary search).
    pub seeks: u64,
}

impl KernelStats {
    /// Add two counters (saturating).
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self {
            scanned: self.scanned.saturating_add(other.scanned),
            seeks: self.seeks.saturating_add(other.seeks),
        }
    }
}

/// One interface for every (operator, structure) kernel.
pub trait OperatorKernel {
    /// Kernel name (structure + operator).
    fn name(&self) -> &'static str;

    /// Window restrict over the kernel's structure.
    fn restrict(&self, source: &Support, p: &Predicate) -> (Support, KernelStats);

    /// Pointwise map over the kernel's structure.
    fn map(&self, source: &Support, f: MapFn, theta: &Theta) -> (Support, KernelStats);

    /// Merge two supports.
    fn merge(
        &self,
        left: &Support,
        right: &Support,
        combine: MergeFn,
        coverage: Coverage,
    ) -> (Outcome<Support>, KernelStats);

    /// Reduce folded modes with a monoid.
    fn reduce(
        &self,
        source: &Support,
        modes: &[Mode],
        monoid: Monoid,
    ) -> (Outcome<Vec<ReducedRow>>, KernelStats);
}
