//! Full CSV → core.bin + eval conversion (task §3).
//! Two passes: stage (row-2 amounts, epoch times) → dictionaries →
//! weighted-amount moments → sort by `(t,s,r)` (internal `tx_id`
//! tie-break) → emit both files. `tx_id` is written to the sidecar only.

use super::core_bin::{serialize_core, CoreRecord};
use super::diag::{log1p_signed, mean_std_major, roundedness_of, z_of};
use super::eval_codec::serialize_eval;
use super::eval_sidecar::EvalRecord;
use super::stage::{build_dicts, code_of, stage_csv};
use crate::TensorError;
use std::io::BufRead;

/// Conversion output: both binary images + provenance facts.
pub struct Conversion {
    /// Ordered core binary bytes.
    pub core_bytes: Vec<u8>,
    /// Eval sidecar bytes (positionally aligned with `core_bytes`).
    pub eval_bytes: Vec<u8>,
    /// Core records (sorted) for dataset counts.
    pub core_records: Vec<CoreRecord>,
    /// Eval records (same order).
    pub eval_records: Vec<EvalRecord>,
    /// Sender dictionary.
    pub senders: Vec<String>,
    /// Receiver dictionary.
    pub receivers: Vec<String>,
    /// Min/max event time.
    pub t_min: u64,
    /// Max event time.
    pub t_max: u64,
    /// Mean major-unit amount.
    pub mean_major: f64,
    /// Population std.
    pub std_major: f64,
}

/// Convert staged CSV into both binary images (deterministic `R_7`).
///
/// # Errors
/// [`TensorError`] on malformed input or unrepresentable amounts.
pub fn convert_reader<R: BufRead>(reader: R) -> Result<Conversion, TensorError> {
    let staged = stage_csv(reader)?;
    if staged.is_empty() {
        let core_bytes = serialize_core(&[], &[], &[], 0, 0, 0.0, 0.0);
        let eval_bytes = serialize_eval(&[]);
        return Ok(Conversion {
            core_bytes, eval_bytes,
            core_records: Vec::new(), eval_records: Vec::new(),
            senders: Vec::new(), receivers: Vec::new(),
            t_min: 0, t_max: 0, mean_major: 0.0, std_major: 0.0,
        });
    }
    let (senders, receivers) = build_dicts(&staged);
    // Pass 1: moments over major-unit amounts (arrival order; sum is
    // order-independent up to f64 presentation precision, K10 class).
    let majors: Vec<f64> = staged.iter()
        .map(|r| r.amount_micro as f64 / 1_000_000.0)
        .collect();
    let (mean_major, std_major) = mean_std_major(&majors);
    let mut t_min = u64::MAX;
    let mut t_max = 0u64;
    // Pass 2: code + diagnostics per row.
    let mut pairs: Vec<(CoreRecord, EvalRecord)> = Vec::with_capacity(staged.len());
    for (i, row) in staged.iter().enumerate() {
        let major = majors[i];
        let s = code_of(&senders, &row.s_raw)
            .ok_or_else(|| TensorError::Parse("sender code lookup failed".into()))?;
        let r = code_of(&receivers, &row.r_raw)
            .ok_or_else(|| TensorError::Parse("receiver code lookup failed".into()))?;
        t_min = t_min.min(row.t);
        t_max = t_max.max(row.t);
        pairs.push((
            CoreRecord {
                t: row.t, s, r,
                amount_micro: row.amount_micro,
                log1p_amt: log1p_signed(major),
                z_amt: z_of(major, mean_major, std_major),
                roundedness: roundedness_of(row.amount_micro),
            },
            EvalRecord {
                tx_id: row.tx_id,
                amount_micro: row.amount_micro,
                laundering: row.laundering,
                amount_raw: row.amount_raw.clone(),
                laundering_type: row.laundering_type.clone(),
                payment_currency: row.payment_currency.clone(),
                received_currency: row.received_currency.clone(),
                sender_bank_location: row.sender_bank_location.clone(),
                receiver_bank_location: row.receiver_bank_location.clone(),
                payment_type: row.payment_type.clone(),
            },
        ));
    }
    // Sort by (t,s,r) (P1 `d_lex`; duplicates permitted, P4). `tx_id` is an
    // internal, un-stored tie-breaker so that duplicate-coordinate records
    // emit deterministically and the eval sidecar stays positionally
    // aligned with the core (index `i` ↔ record `i`).
    pairs.sort_by(|a, b| {
        (a.0.t, a.0.s, a.0.r, a.1.tx_id).cmp(&(b.0.t, b.0.s, b.0.r, b.1.tx_id))
    });
    let core_records: Vec<CoreRecord> = pairs.iter().map(|p| p.0).collect();
    let eval_records: Vec<EvalRecord> = pairs.iter().map(|p| p.1.clone()).collect();
    let core_bytes = serialize_core(
        &core_records, &senders, &receivers, t_min, t_max, mean_major, std_major,
    );
    let eval_bytes = serialize_eval(&eval_records);
    Ok(Conversion {
        core_bytes, eval_bytes, core_records, eval_records,
        senders, receivers, t_min, t_max, mean_major, std_major,
    })
}

/// Convert from a CSV file path (`a_19.1`).
///
/// # Errors
/// [`TensorError`] on I/O failure or malformed content.
pub fn convert_csv(path: &str) -> Result<Conversion, TensorError> {
    let f = std::fs::File::open(path)?;
    convert_reader(std::io::BufReader::new(f))
}
