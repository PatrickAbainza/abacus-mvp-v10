//! Row 24 (§B.2) + P1/P11: sidecar record types and provenance-side statistics.
//!
//! Ordering contract: `prec_4` — the coordinate order `prec_3` extended by the
//! unique source-row tie-breaker. `prec_4` is used **only** for sidecar
//! records; it is never mixed into tensor coordinate order (`prec_3`), and the
//! tie-breaker here is a declared order component, not a repair of the `K13`
//! question about equal-coordinate accumulation during settling (`K13`
//! concerns `prec_3` sorting of duplicate coordinates and stays unresolved
//! there).

/// One laundering-labelled CSV row: normalized keys plus the row verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunderingRecord {
    /// Seconds since epoch parsed from (`time_raw`, `date_raw`).
    pub timestamp: u64,
    /// 0-based data-row ordinal in the source CSV (tie-breaker + provenance).
    pub source_row: u64,
    /// Sender account identifier, verbatim from CSV.
    pub sender_account: String,
    /// Receiver account identifier, verbatim from CSV.
    pub receiver_account: String,
    /// Amount in micro-units, parsed under P2 (cross-check value).
    pub amount_micro: i64,
    /// Amount text, verbatim from CSV.
    pub amount_raw: String,
    /// Time text (`Time` column), verbatim.
    pub time_raw: String,
    /// Date text (`Date` column), verbatim.
    pub date_raw: String,
    /// Payment currency, verbatim.
    pub payment_currency: String,
    /// Received currency, verbatim.
    pub received_currency: String,
    /// Sender bank location, verbatim.
    pub sender_bank_location: String,
    /// Receiver bank location, verbatim.
    pub receiver_bank_location: String,
    /// Payment type, verbatim.
    pub payment_type: String,
    /// Binary label: 0 for legitimate, 1 for laundering (strictly binary).
    pub laundering: u8,
    /// Laundering typology label, verbatim.
    pub laundering_type: String,
}

impl LaunderingRecord {
    /// `prec_4` sort key: `(timestamp, sender, receiver, source_row)`.
    #[must_use]
    pub fn sort_key(&self) -> (&u64, &str, &str, &u64) {
        (
            &self.timestamp,
            self.sender_account.as_str(),
            self.receiver_account.as_str(),
            &self.source_row,
        )
    }
}

impl PartialOrd for LaunderingRecord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LaunderingRecord {
    /// `R_4` over the four components, the last one unique.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

/// The sidecar: every CSV data row, each carrying its binary laundering
/// label, in `prec_4` order.
///
/// Not a tensor; it never enters tensor algebra, CSF storage, or any tensor
/// read path (P11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunderingSidecar {
    /// Records in canonical `prec_4` order (enforced by the builder).
    pub records: Vec<LaunderingRecord>,
}

impl LaunderingSidecar {
    /// Create from records already in `prec_4` order.
    #[must_use]
    pub fn new(records: Vec<LaunderingRecord>) -> Self {
        Self { records }
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True iff empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Row 24 — typology distribution: `(type, count)` pairs in ascending type
    /// byte order (`a_{16}.2` ordered counting; total iteration order only,
    /// `R_7`).
    #[must_use]
    pub fn type_distribution(&self) -> Vec<(String, usize)> {
        let mut map = std::collections::BTreeMap::<&str, usize>::new();
        for r in &self.records {
            *map.entry(r.laundering_type.as_str()).or_insert(0) += 1;
        }
        map.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }
}
