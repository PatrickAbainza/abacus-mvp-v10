//! Ordered-binary → direct CSF (addendum §S7).
//! Read core.bin → order check (d_inv) → project (t,s,r,amount_micro)
//! → d_settle in memory (P4) → canonical check → compress (row 11)
//! → invariants + no-self-aggregation equivalence. No COO file/step.

use super::core_bin::{CoreImage, CoreRecord};
use crate::composites::settle;
use crate::coo::entry::TxEntry;
use crate::csf::compress::{compress, no_self_aggregation, CsfTensor};
use crate::TensorError;

/// Project core records to canonical entries (amount_micro ONLY).
#[must_use]
pub fn project_entries(records: &[CoreRecord]) -> Vec<TxEntry> {
    records
        .iter()
        .map(|r| TxEntry::new(r.t, r.s, r.r, r.amount_micro))
        .collect()
}

/// Direct CSF build from an ordered core image.
/// Fails closed if records are not in non-decreasing `(t,s,r)` order.
/// Returns `(csf, settled_canonical_entries)`.
pub fn direct_csf(image: &CoreImage) -> Result<(CsfTensor, Vec<TxEntry>), TensorError> {
    if image.records.windows(2).any(|w| w[0] > w[1]) {
        return Err(TensorError::Decode(
            "core records not in canonical order (direct CSF precondition)".into(),
        ));
    }
    let raw = project_entries(&image.records);
    let settled = settle(raw)?;
    // d_inv canonical check (P3): strictly increasing + nonzero.
    if settled.iter().any(|e| e.value == 0)
        || settled.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(TensorError::Range(
            "internal: settled output not canonical".into(),
        ));
    }
    let csf = compress(&settled);
    if !csf.check_invariants() {
        return Err(TensorError::Range("CSF invariant check failed".into()));
    }
    if !no_self_aggregation(&settled, &csf) {
        return Err(TensorError::Range(
            "no-self-aggregation invariant failed".into(),
        ));
    }
    Ok((csf, settled))
}
