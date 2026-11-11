//! Row 15 (§B.2) + P6: CSF decoding — regroup only, drop nothing.
//!
//! `⟦CSF(T)⟧ = T`: index expansion under `R_3` with `a_{14}` record
//! construction and `a_{15}` emission. Decoding to a `Tensor` needs no algebra
//! because the decoded sequence is already canonical (unique coordinates), so
//! it never sums and never fails.

use crate::coo::entry::TxEntry;
use crate::csf::compress::CsfTensor;
use crate::tensor::Tensor;

/// Row 15 — `⟦CSF(T)⟧`: expand the nested fibers back to canonical COO.
#[must_use]
pub fn decode_csf(csf: &CsfTensor) -> Vec<TxEntry> {
    let mut out = Vec::with_capacity(csf.values.len());
    for (i, &t) in csf.mode0_coords.iter().enumerate() {
        let (js, je) = (csf.mode0_fptr[i] as usize, csf.mode0_fptr[i + 1] as usize);
        for j in js..je {
            let s = csf.mode1_coords[j];
            let (ks, ke) = (csf.mode1_fptr[j] as usize, csf.mode1_fptr[j + 1] as usize);
            for k in ks..ke {
                out.push(TxEntry {
                    t,
                    s,
                    r: csf.mode2_coords[k],
                    value: csf.values[k],
                });
            }
        }
    }
    out
}

/// Row 15 — `⟦CSF(T)⟧` as a `Tensor` (equals `T` by the §6 equivalence).
///
/// Infallible: the decoded sequence has unique coordinates and no zero values
/// (P6 + P3), so construction performs no summation and cannot overflow.
#[must_use]
pub fn decode_to_tensor(csf: &CsfTensor) -> Tensor {
    Tensor::from_canonical(decode_csf(csf))
}
