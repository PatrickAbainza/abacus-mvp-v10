//! Column mapping: which source column carries which canonical field.
//!
//! §12 step 3 is "Map source columns to canonical fields", and §8 lists
//! "Source amount column mapping" as *required configuration*. A mapping is
//! therefore an explicit, complete statement — never inferred from header names.
//!
//! Why not infer: header names are not contractual. `From`/`To`, `Src`/`Dst`,
//! `Debit`/`Credit` all appear in real exports, and picking one silently would
//! put a sender in a receiver column (§9 Stage 1: "Do not guess missing
//! configuration when the guess changes event meaning"). Inference would also
//! make two different files with different headers produce different identities
//! for the same logical transfer, which breaks streaming_ingestion_demo determinism (§17.6).
//!
//! This module *does* offer [`ColumnMapping::suggest`], but only as a UI
//! convenience the user must confirm: it returns a **partial** mapping plus the
//! ambiguity it found, and a partial mapping is rejected by
//! [`ColumnMapping::validate`]. A suggestion can never be used unconfirmed.

use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};

/// The canonical fields a browser CSV row must supply.
///
/// The namespace is not here: it is a session-level decision (§2: one namespace
/// per segment), declared once in the plan, not per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalField {
    /// Event timestamp column.
    Timestamp,
    /// Sender account identifier column.
    Sender,
    /// Receiver account identifier column.
    Receiver,
    /// Amount column.
    Amount,
}

impl CanonicalField {
    /// Stable display name for UI labels and error details.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Timestamp => "timestamp",
            Self::Sender => "sender",
            Self::Receiver => "receiver",
            Self::Amount => "amount",
        }
    }

    /// Every canonical field, in a fixed order.
    pub const ALL: [CanonicalField; 4] = [
        CanonicalField::Timestamp,
        CanonicalField::Sender,
        CanonicalField::Receiver,
        CanonicalField::Amount,
    ];
}

/// An explicit mapping from source column index to canonical field.
///
/// Column indices are 0-based positions in the parsed record. Every canonical
/// field must be mapped, and no source column may serve two fields, because a
impl ColumnMapping {
    /// Build a mapping from `(field, column index)` pairs.
    ///
    /// # Errors
    /// [`ErrorCode::ColumnMapping`] if a canonical field is missing, if a field
    /// is mapped twice, or if two fields share one column.
    pub fn new(pairs: impl IntoIterator<Item = (CanonicalField, usize)>) -> IngestResult<Self> {
        let columns: Vec<(CanonicalField, usize)> = pairs.into_iter().collect();
        let m = Self { columns };
        m.validate()?;
        Ok(m)
    }

    /// The source column index for `field`.
    #[must_use]
    pub fn column_of(&self, field: CanonicalField) -> Option<usize> {
        self.columns
            .iter()
            .find(|(f, _)| *f == field)
            .map(|(_, c)| *c)
    }

    /// All mapped pairs, in the canonical field order.
    #[must_use]
    pub fn pairs(&self) -> Vec<(CanonicalField, usize)> {
        CanonicalField::ALL
            .iter()
            .filter_map(|f| self.column_of(*f).map(|c| (*f, c)))
            .collect()
    }

    /// Highest referenced column index, or `None` for an empty mapping.
    #[must_use]
    pub fn max_column(&self) -> Option<usize> {
        self.columns.iter().map(|(_, c)| *c).max()
    }

    /// Validate completeness and non-overlap.
    ///
    /// # Errors
    /// - [`ErrorCode::ColumnMapping`] for a missing field, a duplicated field,
    ///   or a column claimed by two fields.
    pub fn validate(&self) -> IngestResult<()> {
        for f in CanonicalField::ALL {
            let hits = self.columns.iter().filter(|(g, _)| *g == f).count();
            match hits {
                0 => {
                    return Err(IngestError::new(
                        ErrorCode::ColumnMapping,
                        format!(
                            "no source column is mapped to the canonical field '{}'",
                            f.as_str()
                        ),
                    ))
                }
                1 => {}
                n => {
                    return Err(IngestError::new(
                        ErrorCode::ColumnMapping,
                        format!(
                            "the canonical field '{}' is mapped {n} times; a field must map to exactly one column",
                            f.as_str()
                        ),
                    ))
                }
            }
        }
        // One column cannot serve two canonical fields.
        let mut used: Vec<usize> = self.columns.iter().map(|(_, c)| *c).collect();
        used.sort_unstable();
        for w in used.windows(2) {
            if w[0] == w[1] {
                return Err(IngestError::new(
                    ErrorCode::ColumnMapping,
                    format!("source column {} is mapped to two canonical fields", w[0]),
                ));
            }
        }
        Ok(())
    }
}

/// An explicit mapping from source column index to canonical field.
///
/// Column indices are 0-based positions in the parsed record. Every canonical
/// field must be mapped, and no source column may serve two fields, because a
/// single column cannot be both a sender and an amount.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMapping {
    /// Source column index for each canonical field.
    columns: Vec<(CanonicalField, usize)>,
}

/// Propose a mapping from header names, for the UI to confirm.
///
/// This is deliberately **not** automatic: it returns a partial mapping plus
/// the ambiguity it found, so the confirmation step (§12 step 6: "Require
/// explicit user confirmation before uploading") is a real decision rather
/// than a formality. Matching is case-insensitive and ignores surrounding
/// whitespace, which is all that is safe to normalize.
///
/// # Errors
/// This constructor cannot fail; use [`ColumnMapping::validate`] on the
/// result, which rejects a partial or overlapping mapping.
#[must_use]
pub fn suggest_mapping(header: &[String]) -> (Vec<(CanonicalField, usize)>, Vec<String>) {
    let norm = |s: &str| s.trim().to_ascii_lowercase();
    let mut picked: Vec<(CanonicalField, usize)> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    for (i, name) in header.iter().enumerate() {
        let n = norm(name);
        let field = match n.as_str() {
            "timestamp" | "time" | "datetime" | "date_time" | "date" | "event_time"
            | "eventtime" | "occurred_at" | "transaction_time" | "txn_time" => {
                Some(CanonicalField::Timestamp)
            }
            "sender" | "from" | "from_account" | "sender_account" | "source" | "payer"
            | "originator" | "debit_account" | "src" => Some(CanonicalField::Sender),
            "receiver" | "to" | "to_account" | "receiver_account" | "destination"
            | "beneficiary" | "payee" | "credit_account" | "dst" | "dest" => {
                Some(CanonicalField::Receiver)
            }
            "amount" | "value" | "amt" | "sum" | "total" | "transaction_amount" | "txn_amount" => {
                Some(CanonicalField::Amount)
            }
            _ => None,
        };
        match field {
            Some(f) if picked.iter().any(|(g, _)| *g == f) => {
                notes.push(format!(
                    "{name}: ambiguous, another column already maps to {}",
                    f.as_str()
                ));
            }
            Some(f) => picked.push((f, i)),
            None => notes.push(name.clone()),
        }
    }
    (picked, notes)
}

/// Extract one mapped field from a parsed row, reporting the source line.
///
/// # Errors
/// [`ErrorCode::ColumnMapping`] if the row is shorter than the mapping requires
/// (a ragged row), naming the source line and the missing field.
pub fn field_at<'a>(
    row: &'a crate::csv::CsvRow,
    mapping: &ColumnMapping,
    field: CanonicalField,
) -> IngestResult<&'a str> {
    let col = mapping.column_of(field).ok_or_else(|| {
        IngestError::new(
            ErrorCode::ColumnMapping,
            format!("no column mapped to '{}'", field.as_str()),
        )
    })?;
    row.fields.get(col).map(String::as_str).ok_or_else(|| {
        IngestError::new(
            ErrorCode::ColumnMapping,
            format!(
                "source line {} has {} fields, so column {col} ('{}') does not exist",
                row.line,
                row.fields.len(),
                field.as_str()
            ),
        )
    })
}
