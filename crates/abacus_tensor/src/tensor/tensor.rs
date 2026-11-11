//! Rows 4–9 (§B.2) + A3: the finite-support tensor and its admissible algebra.
//!
//! `Tensor` is a canonical entry sequence (P3): sorted by `prec_3`, unique
//! coordinates, all values nonzero. It is a newtype over that sequence so that
//! every constructor goes through `d_settle` (row 5) and every combination
//! through `d_merge2` (row 7) — the two composites are realized once, in
//! [`crate::composites`].

use crate::composites::{merge2, settle};
use crate::coo::entry::TxEntry;
use crate::tensor::coordinate::Coord;
use crate::tensor::value::MinorUnits;
use crate::TensorError;

/// Immutable finite-support tensor `T : T × S × R → V` (A3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    entries: Vec<TxEntry>,
}

impl Tensor {
    /// Row 4 — the zero tensor (additive identity; empty support, P3).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Row 5 — tensor construction from raw entries: `d_settle` (P4) yields the
    /// normal form `COO(T)` of P3 (sorted, unique, nonzero).
    ///
    /// # Errors
    /// [`TensorError::Range`] if any coordinate sum is not representable.
    pub fn try_from_entries(v: Vec<TxEntry>) -> Result<Self, TensorError> {
        Ok(Self {
            entries: settle(v)?,
        })
    }

    /// Wrap an already-canonical entry sequence (produced by `d_settle`, by
    /// `d_merge2`, or by CSF decoding) without re-settling: all internal call
    /// sites of this constructor are canonical by construction.
    #[must_use]
    pub(crate) fn from_canonical(entries: Vec<TxEntry>) -> Self {
        Self { entries }
    }

    /// Row 4 — canonical entries `COO(T)` (P3).
    #[inline]
    #[must_use]
    pub fn entries(&self) -> &[TxEntry] {
        &self.entries
    }

    /// Row 4 — nonzero count `|supp(T)|`.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Row 4 — zero test (`a_{01}` zero test).
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.entries.is_empty()
    }

    /// Row 6 — point read `T(x)` (`a_{11}` under the `d_lex` comparator, `0`
    /// when `x ∉ supp(T)`).
    #[must_use]
    pub fn get(&self, coord: Coord) -> MinorUnits {
        match self.entries.binary_search_by(|e| e.coord().cmp(&coord)) {
            Ok(i) => self.entries[i].value,
            Err(_) => 0,
        }
    }

    /// Row 7 — checked pointwise addition (`d_merge2`, P5).
    ///
    /// # Errors
    /// [`TensorError::Range`] if an equal-coordinate sum is not representable.
    pub fn try_add(&self, other: &Self) -> Result<Self, TensorError> {
        Ok(Self::from_canonical(merge2(&self.entries, &other.entries)?))
    }

    /// Row 8 — checked pointwise negation (`R_3` map + `a_{05}`, `R_5`).
    ///
    /// # Errors
    /// [`TensorError::Range`] if any value is `i64::MIN` (P2: `V` is not a
    /// group; the minimum has no representable negation).
    pub fn try_neg(&self) -> Result<Self, TensorError> {
        let mut out = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            out.push(TxEntry {
                t: e.t,
                s: e.s,
                r: e.r,
                value: crate::prims::neg(e.value, "tensor negation (row 8)")?,
            });
        }
        Ok(Self::from_canonical(out))
    }

    /// Row 9 — `T → CSF(T)` (`Compress(COO(T))`, P6).
    #[must_use]
    pub fn to_csf(&self) -> crate::csf::compress::CsfTensor {
        crate::csf::compress::compress(&self.entries)
    }

    /// Row 9 — export as COO (`a_{15}` clone of the canonical sequence).
    #[must_use]
    pub fn encode_coo(&self) -> Vec<TxEntry> {
        self.entries.clone()
    }
}

/// Row 10 (§B.2) — direct-sum decode of a raw entry sequence:
/// `⟦C⟧(x) = Σ { v : (x, v) ∈ C }` followed by the P3 normal form.
///
/// This is the *only* decode of a raw sequence; it is a settling step (P4),
/// many-to-one by construction, never an invertible transformation.
///
/// # Errors
/// [`TensorError::Range`] if any coordinate sum is not representable.
pub fn decode_sum(entries: &[TxEntry]) -> Result<Tensor, TensorError> {
    Ok(Tensor::from_canonical(settle(entries.to_vec())?))
}
