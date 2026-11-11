//! SAMLD eval sidecar (`samld_eval.sidecar`).
//! Addendum §S6: P11 provenance only, never read by tensor logic.
//! Stores NO time/sender/receiver. Sole carrier of `tx_id` (§S4/§S6): the
//! core key is `(t,s,r)` alone, so the sidecar joins to the core by record
//! position (bijection) with `tx_id` as the carried transaction identity.

//! Record type only; codec lives in `eval_codec`.

/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"SAMEVL01";
/// Format version.
pub const VERSION: u32 = 1;
/// Header length: 8+4+8+8 = 28.
pub const HEADER_LEN: usize = 28;
/// Max accepted records.
pub const MAX_RECORDS: u64 = 100_000_000;
/// Max accepted string length.
pub const MAX_STR: u32 = 10_000_000;

/// One eval-sidecar record: identification/evaluation only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalRecord {
    /// Transaction identity (0-based CSV data-row ordinal). Sole carrier:
    /// the core key is `(t,s,r)`, so this joins to core positionally.
    pub tx_id: u64,
    /// Cross-check amount (must equal core `amount_micro` at same `tx_id`).
    pub amount_micro: i64,
    /// Binary label (strictly 0/1).
    pub laundering: u8,
    /// Amount text verbatim.
    pub amount_raw: String,
    /// Typology label verbatim.
    pub laundering_type: String,
    /// Payment currency verbatim.
    pub payment_currency: String,
    /// Received currency verbatim.
    pub received_currency: String,
    /// Sender bank location verbatim.
    pub sender_bank_location: String,
    /// Receiver bank location verbatim.
    pub receiver_bank_location: String,
    /// Payment type verbatim.
    pub payment_type: String,
}

impl EvalRecord {
    /// Storage order key: the sidecar file order matches core record order
    /// exactly, so index `i` of the sidecar pairs with core record `i`.
    /// `tx_id` is the carried identity, not an ordering key (the core key
    /// is `(t,s,r)`), so this is not sorted numerically in general.
    #[must_use]
    pub fn storage_key(&self) -> u64 {
        self.tx_id
    }
}

impl PartialOrd for EvalRecord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvalRecord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.tx_id.cmp(&other.tx_id)
    }
}
