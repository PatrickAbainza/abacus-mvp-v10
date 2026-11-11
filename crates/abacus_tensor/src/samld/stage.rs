//! CSV staging: row-14/24 shape (ingestion-boundary only).

use crate::composites::{epoch_seconds, split_csv_line};
use crate::tensor::value::parse_decimal_to_minor;
use crate::TensorError;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};

/// Staged CSV row before dictionary coding + diagnostics.
pub struct StagedRow {
    /// Event time (epoch seconds).
    pub t: u64,
    /// Sender identifier verbatim.
    pub s_raw: String,
    /// Receiver identifier verbatim.
    pub r_raw: String,
    /// Canonical amount (P2).
    pub amount_micro: i64,
    /// Source-row ordinal == tx_id.
    pub tx_id: u64,
    /// Binary label.
    pub laundering: u8,
    /// Verbatim provenance fields.
    pub amount_raw: String,
    /// Time verbatim.
    pub time_raw: String,
    /// Date verbatim.
    pub date_raw: String,
    /// Currencies/locations/type verbatim.
    pub payment_currency: String,
    /// Received currency verbatim.
    pub received_currency: String,
    /// Sender bank location verbatim.
    pub sender_bank_location: String,
    /// Receiver bank location verbatim.
    pub receiver_bank_location: String,
    /// Payment type verbatim.
    pub payment_type: String,
    /// Typology verbatim.
    pub laundering_type: String,
}

/// Read + stage all CSV rows (arrival order preserved in `tx_id`).
pub fn stage_csv<R: BufRead>(reader: R) -> Result<Vec<StagedRow>, TensorError> {
    let mut reader = BufReader::new(reader);
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
    let (c_time, c_date, c_send, c_recv, c_amt) = (
        col("Time")?, col("Date")?, col("Sender_account")?,
        col("Receiver_account")?, col("Amount")?,
    );
    let (c_pc, c_rc, c_sbl, c_rbl, c_pt, c_lt) = (
        col("Payment_currency")?, col("Received_currency")?,
        col("Sender_bank_location")?, col("Receiver_bank_location")?,
        col("Payment_type")?, col("Laundering_type")?,
    );
    let c_is = col("Is_laundering")?;
    let mut rows: Vec<StagedRow> = Vec::new();
    let mut line = String::new();
    let mut tx: u64 = 0;
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
        let line_no = tx + 2;
        let get = |i: usize| -> Result<&str, TensorError> {
            f.get(i).map(|s| s.trim())
                .ok_or_else(|| TensorError::Parse(format!("line {line_no}: short row")))
        };
        let time_raw = get(c_time)?.to_string();
        let date_raw = get(c_date)?.to_string();
        let t = epoch_seconds(&time_raw, &date_raw)?;
        let amount_raw = get(c_amt)?.to_string();
        let amount_micro = parse_decimal_to_minor(&amount_raw)?;
        let laun_raw = get(c_is)?;
        let laundering = match laun_raw {
            "0" => 0u8,
            "1" => 1u8,
            _ => return Err(TensorError::Parse(format!(
                "line {line_no}: invalid label {laun_raw:?}"
            ))),
        };
        rows.push(StagedRow {
            t,
            s_raw: get(c_send)?.to_string(),
            r_raw: get(c_recv)?.to_string(),
            amount_micro,
            tx_id: tx,
            laundering,
            amount_raw,
            time_raw,
            date_raw,
            payment_currency: get(c_pc)?.to_string(),
            received_currency: get(c_rc)?.to_string(),
            sender_bank_location: get(c_sbl)?.to_string(),
            receiver_bank_location: get(c_rbl)?.to_string(),
            payment_type: get(c_pt)?.to_string(),
            laundering_type: get(c_lt)?.to_string(),
        });
        tx += 1;
    }
    Ok(rows)
}

/// Build dictionaries (`prec_dict` ascending; code order == table order).
pub fn build_dicts(rows: &[StagedRow]) -> (Vec<String>, Vec<String>) {
    let mut ss = BTreeSet::new();
    let mut rs = BTreeSet::new();
    for r in rows {
        ss.insert(r.s_raw.clone());
        rs.insert(r.r_raw.clone());
    }
    (ss.into_iter().collect(), rs.into_iter().collect())
}

/// Code lookup (`a_11` over `prec_dict` order).
pub fn code_of(tab: &[String], raw: &str) -> Option<u32> {
    u32::try_from(tab.binary_search_by(|s| s.as_str().cmp(raw)).ok()?).ok()
}
