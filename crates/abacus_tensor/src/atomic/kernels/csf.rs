//! CSF kernels: window restrict + reduce-over-time by mode-0 scan.
//!
//! Reads the canonical `CsfTensor` fiber arrays directly (time-leading
//! order); entry order out is `prec_3`. Work counters count mode-0 seeks
//! plus entries visited.

use super::traits::{KernelStats, OperatorKernel};
use crate::atomic::reference::{self, Coverage, MapFn, MergeFn, Predicate, ReducedRow, Theta};
use crate::atomic::types::{Entry, LocSet, Mode, Monoid, Support};
use crate::atomic::Outcome;
use crate::csf::compress::CsfTensor;

/// Kernel state: borrows the canonical CSF tensor.
pub struct CsfKernel<'a> {
    /// Canonical tensor (time-leading CSF).
    pub csf: &'a CsfTensor,
}

impl<'a> CsfKernel<'a> {
    /// Borrow the canonical tensor.
    #[must_use]
    pub const fn new(csf: &'a CsfTensor) -> Self {
        Self { csf }
    }

    /// Decode one mode-0 fiber range to entries (locator sets empty:
    /// pure-tensor presence = net-nonzero).
    fn scan_range(&self, lo: usize, hi: usize, stats: &mut KernelStats) -> Vec<Entry> {
        let csf = self.csf;
        let mut out = Vec::new();
        for i in lo..hi {
            let t = csf.mode0_coords[i];
            let (js, je) = (csf.mode0_fptr[i] as usize, csf.mode0_fptr[i + 1] as usize);
            for j in js..je {
                let s = csf.mode1_coords[j];
                let (ks, ke) = (csf.mode1_fptr[j] as usize, csf.mode1_fptr[j + 1] as usize);
                for k in ks..ke {
                    stats.scanned = stats.scanned.saturating_add(1);
                    out.push(Entry::new(t, s, csf.mode2_coords[k], csf.values[k], LocSet::empty()));
                }
            }
        }
        out
    }

    /// Resolve a half-open window to mode-0 positions (two seeks).
    fn window_positions(&self, lo_t: u64, hi_t: u64, stats: &mut KernelStats) -> (usize, usize) {
        stats.seeks = stats.seeks.saturating_add(2);
        let lo = self.csf.mode0_coords.partition_point(|&t| t < lo_t);
        let hi = self.csf.mode0_coords.partition_point(|&t| t < hi_t);
        (lo, hi)
    }
}

impl OperatorKernel for CsfKernel<'_> {
    fn name(&self) -> &'static str {
        "csf-scan"
    }

    fn restrict(&self, _source: &Support, p: &Predicate) -> (Support, KernelStats) {
        let mut stats = KernelStats::default();
        // Narrow by window on mode-0 first (real access path), then filter.
        let (lo, hi) = match p.window {
            Some((lo_t, hi_t)) if lo_t < hi_t => self.window_positions(lo_t, hi_t, &mut stats),
            Some(_) => return (Support::empty(), stats),
            None => {
                stats.seeks = stats.seeks.saturating_add(2);
                (0, self.csf.mode0_coords.len())
            }
        };
        let mut entries = self.scan_range(lo, hi, &mut stats);
        entries.retain(|e| p.admits(e));
        (Support::from_canonical(entries), stats)
    }

    fn map(&self, source: &Support, f: MapFn, theta: &Theta) -> (Support, KernelStats) {
        let stats = KernelStats { scanned: source.entries.len() as u64, seeks: 0 };
        (reference::map(source, f, theta), stats)
    }

    fn merge(
        &self,
        left: &Support,
        right: &Support,
        combine: MergeFn,
        coverage: Coverage,
    ) -> (Outcome<Support>, KernelStats) {
        let stats = KernelStats {
            scanned: (left.entries.len() + right.entries.len()) as u64,
            seeks: 0,
        };
        (reference::merge(left, right, combine, coverage), stats)
    }

    fn reduce(
        &self,
        source: &Support,
        modes: &[Mode],
        monoid: Monoid,
    ) -> (Outcome<Vec<ReducedRow>>, KernelStats) {
        let stats = KernelStats { scanned: source.entries.len() as u64, seeks: 0 };
        (reference::reduce(source, modes, monoid), stats)
    }
}
