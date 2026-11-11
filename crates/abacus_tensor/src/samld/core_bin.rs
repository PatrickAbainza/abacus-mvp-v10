//! SAMLD ordered-binary core (`samld_core.bin`).
//! Addendum §§S2/S4/S5: P1 storage order `(t,s,r)`, P2 `amount_micro`
//! value, the three weighted amounts (`log1p`, `z`, `roundedness`),
//! `a_13.1/.2` LE codec, `d_fnv` checksum. `tx_id` lives in the eval
//! sidecar, not here (§S4/§S6, gap `U-SAMLD-2`).

use crate::composites::fnv1a64;
use crate::prims::{write_i64, write_len_str, write_u32, write_u64};
use crate::TensorError;

/// Magic bytes.
pub const MAGIC: &[u8; 8] = b"SAMLDBIN";
/// Format version (`2`: key `(t,s,r)`; `tx_id` moved to the eval sidecar).
pub const FORMAT_VERSION: u32 = 2;
/// Fixed record length: 8+4+4+8+8+8+1+7 = 48.
pub const RECORD_LEN: usize = 48;
/// Header length: 8+4+8+8+8+8+8+8 = 60.
pub const HEADER_LEN: usize = 60;
/// Max accepted record count (bounds hostile count).
pub const MAX_RECORDS: u64 = 100_000_000;
/// Max accepted dictionary string length.
pub const MAX_DICT_STR: u32 = 10_000_000;
/// Max accepted dictionary entries.
pub const MAX_DICT_ENTRIES: usize = 50_000_000;

/// One ordered core record (P4 input form: sorted, duplicates allowed).
///
/// Key = `(t,s,r)` (`prec_3`, P1); payload = the four weighted amounts on
/// that coordinate. `amount_micro` is the P2 canonical tensor value; the
/// other three (`log1p_amt`, `z_amt`, `roundedness`) are stored per-event
/// weights (P11, gap `U-SAMLD-2`). `tx_id` is NOT stored here — it now
/// lives in the eval sidecar (§S4/§S6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoreRecord {
    /// Event time `t` (epoch seconds).
    pub t: u64,
    /// Sender code `s` (row-12 `prec_dict` code).
    pub s: u32,
    /// Receiver code `r`.
    pub r: u32,
    /// Canonical amount in micro-units (P2). THE tensor value.
    pub amount_micro: i64,
    /// Weighted amount: sign-preserving log1p of major-unit amount (P11).
    pub log1p_amt: f64,
    /// Weighted amount: z-score of major-unit amount (P11).
    pub z_amt: f64,
    /// Weighted amount: 1 iff amount is whole major units (P11).
    pub roundedness: u8,
}

impl CoreRecord {
    /// Storage order key `(t,s,r)` = `d_lex` (P1). Duplicate coordinates
    /// compare equal (P4 permits duplicates); no weight ever enters the
    /// key (§S8: weights/amount are values, never key bits).
    #[must_use]
    pub fn storage_key(&self) -> (u64, u32, u32) {
        (self.t, self.s, self.r)
    }
}

impl PartialOrd for CoreRecord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoreRecord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.storage_key().cmp(&other.storage_key())
    }
}

impl Eq for CoreRecord {}

/// Decoded core image: records + dictionaries + header provenance.
#[derive(Debug, Clone)]
pub struct CoreImage {
    /// Records in `(t,s,r)` order (duplicates permitted, P4).
    pub records: Vec<CoreRecord>,
    /// Sender dictionary (`prec_dict` ascending == code order).
    pub senders: Vec<String>,
    /// Receiver dictionary.
    pub receivers: Vec<String>,
    /// Minimum timestamp.
    pub t_min: u64,
    /// Maximum timestamp.
    pub t_max: u64,
    /// Mean of major-unit amounts (P11 presentation).
    pub mean_major: f64,
    /// Population std of major-unit amounts (P11 presentation).
    pub std_major: f64,
    /// Record count from header.
    pub record_count: u64,
}

/// Write f64 LE (`a_13.1` fixed-order emission; P11 diagnostics only).
pub fn write_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Read f64 LE (`a_13.2`).
pub fn read_f64(d: &[u8], o: &mut usize, field: &str) -> Result<f64, TensorError> {
    if *o + 8 > d.len() {
        return Err(TensorError::Decode(format!("truncated {field}")));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[*o..*o + 8]);
    *o += 8;
    Ok(f64::from_le_bytes(b))
}

/// Serialize records + dictionaries (deterministic `R_7`).
#[must_use]
pub fn serialize_core(
    records: &[CoreRecord],
    senders: &[String],
    receivers: &[String],
    t_min: u64,
    t_max: u64,
    mean_major: f64,
    std_major: f64,
) -> Vec<u8> {
    let mut rec: Vec<u8> = Vec::with_capacity(records.len() * RECORD_LEN);
    for r in records {
        write_u64(&mut rec, r.t);
        write_u32(&mut rec, r.s);
        write_u32(&mut rec, r.r);
        write_i64(&mut rec, r.amount_micro);
        write_f64(&mut rec, r.log1p_amt);
        write_f64(&mut rec, r.z_amt);
        rec.push(r.roundedness);
        rec.extend_from_slice(&[0u8; 7]);
    }
    let sum = fnv1a64(&rec);
    let mut out = Vec::with_capacity(HEADER_LEN + rec.len() + 1024);
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, FORMAT_VERSION);
    write_u64(&mut out, records.len() as u64);
    write_u64(&mut out, t_min);
    write_u64(&mut out, t_max);
    write_f64(&mut out, mean_major);
    write_f64(&mut out, std_major);
    write_u64(&mut out, sum);
    out.extend_from_slice(&rec);
    for dict in [senders, receivers] {
        write_u64(&mut out, dict.len() as u64);
        for s in dict {
            write_len_str(&mut out, s);
        }
    }
    out
}
