//! Row 11 (§B.2) + P6: `Compress` and the canonical CSF invariants.
//!
//! PRECONDITION (K3): `compress` requires `prec_3`-sorted canonical COO and
//! does **not** sort internally — the canonical pipeline always establishes
//! sortedness first (row 5/11 settling). In debug/test builds the precondition
//! is machine-checked by `debug_assert!`; in release it is contractual, exactly
//! as `K3` records. The non-canonical `compress_unsorted` convenience of the
//! prior realization is **not** part of the canonical chain (K5) and is
//! deliberately absent: it is not a second canonical path, so its removal
//! removes no capability (see the completion report, §E).
//!
//! Fiber-sortedness (Thm 5.3) is *derived* from the sorted input, never
//! asserted as an extra assumption.

use crate::composites::strictly_increasing;
use crate::coo::entry::TxEntry;
use super::decode::decode_csf;

/// Immutable CSF tensor `(I0, P0, I1, P1, I2, V)` (P6 / Def 5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsfTensor {
    /// `I0`: distinct `t` values in ascending order.
    pub mode0_coords: Vec<u64>,
    /// `P0`: boundary offsets into mode 1; `len = I0.len() + 1`, sentinel
    /// terminated (`P0[0] = 0`, last = `|I1|`).
    pub mode0_fptr: Vec<u64>,
    /// `I1`: distinct `s` values per `t`-fiber (ascending within the fiber).
    pub mode1_coords: Vec<u32>,
    /// `P1`: boundary offsets into mode 2; `len = I1.len() + 1`, sentinel
    /// terminated (`P1[0] = 0`, last = `|I2|`).
    pub mode1_fptr: Vec<u64>,
    /// `I2`: `r` values (ascending within each `(t, s)`-fiber).
    pub mode2_coords: Vec<u32>,
    /// `V`: values aligned with `I2` (exact micro-units).
    pub values: Vec<i64>,
}

impl CsfTensor {
    /// The zero tensor as an empty CSF (P3 + P6).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            mode0_coords: Vec::new(),
            mode0_fptr: vec![0],
            mode1_coords: Vec::new(),
            mode1_fptr: vec![0],
            mode2_coords: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Nonzero count `|supp(T)|`.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// True iff the zero tensor.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Row 11 — `d_inv`: validate the P6/Def 5.1 structural invariants
    /// (pointer lengths, sentinel termination, monotonicity, strict fiber
    /// ordering per Thm 5.3).
    #[must_use]
    pub fn check_invariants(&self) -> bool {
        if self.mode0_fptr.len() != self.mode0_coords.len() + 1
            || self.mode1_fptr.len() != self.mode1_coords.len() + 1
            || self.mode2_coords.len() != self.values.len()
        {
            return false;
        }
        if self.mode0_fptr.first() != Some(&0) || self.mode1_fptr.first() != Some(&0) {
            return false;
        }
        if self.mode0_fptr.last().copied().unwrap_or(0) as usize != self.mode1_coords.len()
            || self.mode1_fptr.last().copied().unwrap_or(0) as usize != self.values.len()
        {
            return false;
        }
        if !strictly_increasing(&self.mode0_coords) {
            return false;
        }
        if self.mode0_fptr.windows(2).any(|w| w[0] > w[1])
            || self.mode1_fptr.windows(2).any(|w| w[0] > w[1])
        {
            return false;
        }
        for i in 0..self.mode0_coords.len() {
            let (js, je) = (self.mode0_fptr[i] as usize, self.mode0_fptr[i + 1] as usize);
            if je > self.mode1_coords.len() {
                return false;
            }
            if !strictly_increasing(&self.mode1_coords[js..je]) {
                return false;
            }
        }
        for j in 0..self.mode1_coords.len() {
            let (ks, ke) = (self.mode1_fptr[j] as usize, self.mode1_fptr[j + 1] as usize);
            if ke > self.values.len() {
                return false;
            }
            if !strictly_increasing(&self.mode2_coords[ks..ke]) {
                return false;
            }
        }
        true
    }
}

impl Default for CsfTensor {
    /// The zero tensor (empty CSF). Default exists so containers of a tensor
    /// start at the additive identity without a special case.
    fn default() -> Self {
        Self::empty()
    }
}

/// Row 11 — `Compress`: canonical COO → CSF in one nested pass over the
/// `prec_3` order (`R_3`), emitting the six fiber arrays with sentinel
/// pointers. No coordinate is dropped, added, or modified.
///
/// PRECONDITION (K3): `coo` is `prec_3`-sorted (canonical, P3).
#[must_use]
pub fn compress(coo: &[TxEntry]) -> CsfTensor {
    debug_assert!(
        coo.windows(2).all(|w| w[0] <= w[1]),
        "compress precondition violated (K3): input COO must be prec_3-sorted; \
         the canonical pipeline settles before compressing"
    );
    let mut i0: Vec<u64> = Vec::new();
    let mut p0: Vec<u64> = vec![0];
    let mut i1: Vec<u32> = Vec::new();
    let mut p1: Vec<u64> = vec![0];
    let mut i2: Vec<u32> = Vec::new();
    let mut vv: Vec<i64> = Vec::new();

    let mut idx = 0usize;
    while idx < coo.len() {
        let t = coo[idx].t;
        while idx < coo.len() && coo[idx].t == t {
            let s = coo[idx].s;
            i1.push(s);
            while idx < coo.len() && coo[idx].t == t && coo[idx].s == s {
                i2.push(coo[idx].r);
                vv.push(coo[idx].value);
                idx += 1;
            }
            p1.push(i2.len() as u64);
        }
        i0.push(t);
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

/// No-self-aggregation invariant (P6 + P3 + P4; §6 equivalence): the CSF is a
/// pure *representation* of the canonical COO — compression never sums, merges
/// or drops anything. The only place duplicates at identical `(t, s, r)`
/// coordinates are aggregated is `d_settle` (P4, R_6), which runs *before*
/// compression and yields the canonical form (P3: unique coordinates, nonzero
/// values). Therefore, for a CSF compressed from canonical COO `coo`:
///
/// 1. `nnz(CSF) == |coo|` — one value slot per canonical entry;
/// 2. `decode_csf(CSF) == coo` entry-for-entry (`⟦CSF(T)⟧ = COO(T)`).
///
/// Algebraic aggregation over the stored structure (e.g. GraphBLAS
/// reduce/ewise primitives) belongs to a separate algebraic layer that
/// *consumes* the CSF; it must never be folded into `compress` itself.
///
/// # Arguments
/// * `coo` — the canonical COO the CSF was compressed from (P3 form).
/// * `csf` — the CSF under test.
#[must_use]
pub fn no_self_aggregation(coo: &[TxEntry], csf: &CsfTensor) -> bool {
    csf.nnz() == coo.len() && decode_csf(csf) == coo
}
