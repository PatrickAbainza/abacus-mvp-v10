//! Row 24 (§B.2): CSV → `LaunderingSidecar`.
//!
//! Complete binary labels: both `Is_laundering == "0"` and `"1"`. No aggregation
//! and no dedup — one kept row is one record (provenance is a multiset). Output is
//! sorted by `prec_4`; `source_row` is the 0-based data-row ordinal and is the
//! unique tie-breaker.
//!
//! This path never touches the ledger, the frontier, CSF storage, or the API
//! layer (P11). It shares the CSV splitter and `d_hinnant` composites with the
//! dataset build because they are one composite each (§H.2/K6); the sidecar's
//! *semantic* independence is preserved — it constructs no tensor data.

use crate::composites::{epoch_seconds, split_csv_line};
use crate::sidecar::types::{LaunderingRecord, LaunderingSidecar};
use crate::tensor::value::parse_decimal_to_minor;
use crate::TensorError;
use std::io::{BufRead, BufReader};

/// Required provenance columns of the sidecar build.
const COLUMNS: [&str; 11] = [
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Laundering_type",
];

/// Row 24 — build a sidecar from any buffered CSV reader.
///
/// # Errors
/// [`TensorError`] on malformed header/rows, I/O failure, or an amount that is
/// not representable under P2.
pub fn build_sidecar<R: BufRead>(mut reader: R) -> Result<LaunderingSidecar, TensorError> {
    let mut header = String::new();
    if reader.read_line(&mut header)? == 0 {
        return Err(TensorError::Parse("empty csv".into()));
    }
    let h = split_csv_line(header.trim_end_matches(['\n', '\r']));
    let col = |name: &str| -> Result<usize, TensorError> {
        h.iter()
            .position(|c| c.trim() == name)
            .ok_or_else(|| TensorError::Parse(format!("missing column {name}")))
    };
    let mut idx_of: Vec<usize> = Vec::with_capacity(COLUMNS.len() + 1);
    for name in COLUMNS {
        idx_of.push(col(name)?);
    }
    let c_is = col("Is_laundering")?;

    let mut records: Vec<LaunderingRecord> = Vec::new();
    let mut line = String::new();
    let mut row: u64 = 0;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let text = line.trim_end_matches(['\n', '\r']);
        if text.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(text);
        let line_no = row + 2;
        let get = |i: usize| -> Result<&str, TensorError> {
            f.get(i)
                .map(|s| s.trim())
                .ok_or_else(|| TensorError::Parse(format!("line {line_no}: short row")))
        };
        let source_row = row;
        row += 1;
        let is_laun_raw = get(c_is)?;
        let laundering = match is_laun_raw {
            "0" => 0u8,
            "1" => 1u8,
            _ => {
                return Err(TensorError::Parse(format!(
                    "line {line_no}: invalid laundering label {is_laun_raw:?} (must be 0 or 1)"
                )))
            }
        };
        let time_raw = get(idx_of[0])?.to_string();
        let date_raw = get(idx_of[1])?.to_string();
        let timestamp = epoch_seconds(&time_raw, &date_raw)?;
        let amount_raw = get(idx_of[4])?.to_string();
        let amount_micro = parse_decimal_to_minor(&amount_raw)?;
        records.push(LaunderingRecord {
            timestamp,
            source_row,
            sender_account: get(idx_of[2])?.to_string(),
            receiver_account: get(idx_of[3])?.to_string(),
            amount_micro,
            amount_raw,
            time_raw,
            date_raw,
            payment_currency: get(idx_of[5])?.to_string(),
            received_currency: get(idx_of[6])?.to_string(),
            sender_bank_location: get(idx_of[7])?.to_string(),
            receiver_bank_location: get(idx_of[8])?.to_string(),
            payment_type: get(idx_of[9])?.to_string(),
            laundering,
            laundering_type: get(idx_of[10])?.to_string(),
        });
    }
    records.sort(); // prec_4: R_4 over (timestamp, sender, receiver, source_row)
    Ok(LaunderingSidecar::new(records))
}

/// Row 24 — build a sidecar from a CSV file path (`a_{19}.1`).
///
/// # Errors
/// [`TensorError`] on I/O failure or malformed content.
pub fn build_sidecar_from_csv(path: &str) -> Result<LaunderingSidecar, TensorError> {
    let f = std::fs::File::open(path)?;
    build_sidecar(BufReader::new(f))
}
