//! Row 12 and row 14 (§B.2): the account dictionary and the dataset build.
//!
//! Row 12: the dictionary is an ordered set of raw identifiers
//! (`a_{16}.2`, ascending iteration only) whose code order equals
//! `prec_dict` — byte-string lexicographic order on identifiers (`a_{01}.2`,
//! P1). Lookup is `a_{11}` over that order.
//!
//! Row 14: `a_{19}.1` lines → CSV split (shared composite) → `d_hinnant` time
//! normalization → row 2 exact decimal → dictionary codes (`a_{16}.2`) →
//! `d_settle` → provenance fold (`a_{01}` min/max, `a_{09}` counters).

use crate::composites::{epoch_seconds, settle, split_csv_line};
use crate::coo::encode::is_sorted;
use crate::coo::entry::TxEntry;
use crate::tensor::value::parse_decimal_to_minor;
use crate::TensorError;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};

/// Sorted raw-identifier → code table (row 12). Code order equals
/// `prec_dict` order by construction.
#[derive(Debug, Clone, Default)]
pub struct AccountDict {
    /// Raw identifiers in ascending byte-lexicographic order; index == code.
    pub table: Vec<String>,
}

impl AccountDict {
    /// Row 12 — code lookup by `a_{11}` binary search under `prec_dict`.
    #[must_use]
    pub fn code_of(&self, raw: &str) -> Option<u32> {
        u32::try_from(self.table.binary_search_by(|s| s.as_str().cmp(raw)).ok()?).ok()
    }
}

/// Output of the canonical dataset build (rows 12–14).
#[derive(Debug, Clone, Default)]
pub struct CanonicalBuild {
    /// Canonical COO (P3: sorted, unique, nonzero).
    pub coo: Vec<TxEntry>,
    /// Sender dictionary.
    pub senders: AccountDict,
    /// Receiver dictionary.
    pub receivers: AccountDict,
    /// Raw data rows observed (before aggregation).
    pub raw_rows: u64,
    /// Minimum timestamp (seconds since epoch), 0 when no rows.
    pub t_min: u64,
    /// Maximum timestamp (seconds since epoch), 0 when no rows.
    pub t_max: u64,
    /// Settlement split (G8): how `raw_rows` were absorbed into `coo`.
    ///
    /// Computed from the **pre-settlement** entries, because settling destroys
    /// per-coordinate multiplicities. Build-time provenance only — never read
    /// by any tensor path, and never a lookup structure (P11, `a_{21}`).
    pub settlement: crate::stats::SettlementDiagnostic,
}

/// Row 14 — build the canonical COO from a buffered CSV reader.
///
/// Required columns: `Time`, `Date`, `Sender_account`, `Receiver_account`,
/// `Amount`. Column positions are discovered from the header (`a_{18}.1`
/// trim + `a_{01}` equality), so column order is not assumed.
///
/// # Errors
/// [`TensorError`] on I/O failure, malformed header/row, or arithmetic that is
/// not representable (fail-closed, `R_5`).
pub fn build_from_reader<R: BufRead>(mut reader: R) -> Result<CanonicalBuild, TensorError> {
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
        col("Time")?,
        col("Date")?,
        col("Sender_account")?,
        col("Receiver_account")?,
        col("Amount")?,
    );

    let mut raw: Vec<TxEntry> = Vec::new();
    let mut senders = BTreeSet::<String>::new();
    let mut receivers = BTreeSet::<String>::new();
    let mut send_rows: Vec<(u64, String, String, i64)> = Vec::new();
    let mut t_min = u64::MAX;
    let mut t_max = 0u64;
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
        row += 1;
        let f = split_csv_line(text);
        let get = |i: usize| -> Result<&str, TensorError> {
            f.get(i)
                .map(|s| s.trim())
                .ok_or_else(|| TensorError::Parse(format!("line {row}: short row")))
        };
        let t = epoch_seconds(get(c_time)?, get(c_date)?)?;
        let s_raw = get(c_send)?;
        let r_raw = get(c_recv)?;
        let v = parse_decimal_to_minor(get(c_amt)?)?;
        t_min = t_min.min(t);
        t_max = t_max.max(t);
        senders.insert(s_raw.to_string());
        receivers.insert(r_raw.to_string());
        send_rows.push((t, s_raw.to_string(), r_raw.to_string(), v));
    }

    if send_rows.is_empty() {
        return Ok(CanonicalBuild::default());
    }
    let send_tab: Vec<String> = senders.into_iter().collect();
    let recv_tab: Vec<String> = receivers.into_iter().collect();
    let send_dict = AccountDict {
        table: send_tab.clone(),
    };
    let recv_dict = AccountDict {
        table: recv_tab.clone(),
    };
    raw.reserve(send_rows.len());
    for (t, s_raw, r_raw, v) in send_rows {
        let s = send_dict
            .code_of(&s_raw)
            .ok_or_else(|| TensorError::Parse("sender code lookup failed (row 12)".into()))?;
        let r = recv_dict
            .code_of(&r_raw)
            .ok_or_else(|| TensorError::Parse("receiver code lookup failed (row 12)".into()))?;
        raw.push(TxEntry::new(t, s, r, v));
    }
    let raw_rows = raw.len() as u64;
    // G8: capture the settlement split from the PRE-settlement entries, since
    // `settle` destroys per-coordinate multiplicities (and removes zero-sum
    // coordinates under `R_6`). Diagnostics only; the value path is unchanged.
    let settlement = crate::stats::settlement_diagnostic(&raw)?;
    let coo = settle(raw)?;
    if !is_sorted(&coo) {
        return Err(TensorError::Range(
            "internal: settle output not sorted".into(),
        ));
    }
    Ok(CanonicalBuild {
        coo,
        senders: send_dict,
        receivers: recv_dict,
        raw_rows,
        t_min,
        t_max,
        settlement,
    })
}

/// Row 14 — build the canonical COO from a CSV file path (`a_{19}.1`).
///
/// # Errors
/// [`TensorError`] on I/O failure or malformed content.
pub fn build_from_csv(path: &str) -> Result<CanonicalBuild, TensorError> {
    let f = std::fs::File::open(path)?;
    build_from_reader(BufReader::new(f))
}
