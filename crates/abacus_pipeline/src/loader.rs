//! Initial loading over the surviving R5f artifact generation.
//!
//! Core SAMCORE02 (`samld/core_pack.rs`), CSF SAMCSF02
//! (`csf/compress_v2.rs`), sidecar SAMEVL03 (`samld/eval_codec3.rs`).
//! Sidecar joins core by positional bijection i<->i with tx_id index-set
//! equality over {0..n-1} (`samld/sets.rs`). On disagreement: fail closed.

use crate::error::PipelineError;
use abacus_tensor::csf::{deserialize_csf_v2, Image as CsfImage};
use abacus_tensor::samld::{
    deserialize_core_v2, deserialize_eval_v3, index_set, tx_correspondence,
    CoreImage, EvalRecord,
};
use std::collections::BTreeSet;

/// Configurable artifact paths (never hardcoded machine locations).
#[derive(Debug, Clone)]
pub struct ArtifactPaths {
    /// `samld_core.bin` (SAMCORE02).
    pub core_bin: String,
    /// `samld_core.csf` (SAMCSF02).
    pub core_csf: String,
    /// `samld_eval.sidecar` (SAMEVL03).
    pub eval_sidecar: String,
}

/// All decoded artifacts, positionally aligned.
#[derive(Debug, Clone)]
pub struct LoadedArtifacts {
    /// Ordered core records (P4 input form, duplicates permitted).
    pub core: CoreImage,
    /// Canonical CSF image + dictionaries + provenance footer.
    pub csf: CsfImage,
    /// Eval sidecar records in core (t,s,r) order.
    pub eval: Vec<EvalRecord>,
    /// Raw bytes per artifact (instrumentation).
    pub core_bytes: usize,
    /// Raw bytes per artifact (instrumentation).
    pub csf_bytes: usize,
    /// Raw bytes per artifact (instrumentation).
    pub eval_bytes: usize,
}

/// Loader diagnostics.
#[derive(Debug, Clone)]
pub struct LoadReport {
    /// Core record count (== sidecar count == raw rows).
    pub records: usize,
    /// Settled nonzero count (nnz).
    pub nnz: usize,
    /// (t_min, t_max) agreed by core and CSF footer.
    pub t_range: (u64, u64),
    /// Sender/receiver dictionary sizes.
    pub dict_sizes: (usize, usize),
}
/// Load + cross-validate the three artifacts through canonical codecs.
///
/// Fail-closed checks: count equality, tx_id index-set equality (exact set
/// ops), positional amount agreement, t_range agreement, dictionary
/// agreement, settled-core == decoded-CSF, CSF check_invariants (P6).
///
/// # Errors
/// `Io` on unreadable paths, `TensorError::Decode` on codec failure,
/// `Mismatch` on any cross-artifact disagreement.
pub fn load_artifacts(paths: &ArtifactPaths) -> Result<(LoadedArtifacts, LoadReport), PipelineError> {
    let core_raw = std::fs::read(&paths.core_bin)?;
    let csf_raw = std::fs::read(&paths.core_csf)?;
    let eval_raw = std::fs::read(&paths.eval_sidecar)?;
    let core = deserialize_core_v2(&core_raw)?;
    let csf = deserialize_csf_v2(&csf_raw)?;
    let eval = deserialize_eval_v3(&eval_raw)?;
    let n = core.records.len();
    if eval.len() != n {
        return Err(PipelineError::Mismatch(format!(
            "core {n} != sidecar {}",
            eval.len()
        )));
    }
    if csf.raw_rows as usize != n {
        return Err(PipelineError::Mismatch(format!(
            "core {n} != csf raw_rows {}",
            csf.raw_rows
        )));
    }
    let tx: BTreeSet<u64> = eval.iter().map(EvalRecord::storage_key).collect();
    let (missing, extra, _) = tx_correspondence(&index_set(n), &tx);
    if !missing.is_empty() || !extra.is_empty() {
        return Err(PipelineError::Mismatch(format!(
            "tx_id set mismatch: missing {} extra {}",
            missing.len(),
            extra.len()
        )));
    }
    for (i, e) in eval.iter().enumerate() {
        if e.amount_micro != core.records[i].amount_micro {
            return Err(PipelineError::Mismatch(format!("amount mismatch at {i}")));
        }
    }
    if (core.t_min, core.t_max) != (csf.t_min, csf.t_max) {
        return Err(PipelineError::Mismatch("t_range mismatch".into()));
    }
    if core.senders != csf.senders || core.receivers != csf.receivers {
        return Err(PipelineError::Mismatch("dictionary mismatch".into()));
    }
    if !csf.csf.check_invariants() {
        return Err(PipelineError::Mismatch("CSF P6 invariant failure".into()));
    }
    let settled = abacus_tensor::composites::settle(
        core.records
            .iter()
            .map(|r| abacus_tensor::coo::entry::TxEntry::new(r.t, r.s, r.r, r.amount_micro))
            .collect(),
    )?;
    let decoded = abacus_tensor::csf::decode_csf(&csf.csf);
    if settled != decoded {
        return Err(PipelineError::Mismatch(format!(
            "settled {} != decoded {}",
            settled.len(),
            decoded.len()
        )));
    }
    let report = LoadReport {
        records: n,
        nnz: decoded.len(),
        t_range: (core.t_min, core.t_max),
        dict_sizes: (core.senders.len(), core.receivers.len()),
    };
    Ok((
        LoadedArtifacts { core, csf, eval, core_bytes: core_raw.len(), csf_bytes: csf_raw.len(), eval_bytes: eval_raw.len() },
        report,
    ))
}
