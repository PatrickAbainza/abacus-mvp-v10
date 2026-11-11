//! Bounded streaming reader over a `SAMCORE02` ordered-core artifact.
//!
//! Golden-source grounding: this is NOT a second format. It reuses the exact
//! canonical column layout and codec primitives of
//! `abacus_tensor::samld::core_pack` (`serialize_core_v2` /
//! `deserialize_core_v2`, addendum §R5f) and of `abacus_tensor::varint` /
//! `abacus_tensor::prims`. A `StreamRecord` produced here is byte-for-byte
//! the `(t, s, r, amount_micro)` tuple `deserialize_core_v2` decodes at the
//! same ordinal — verified by `tests/phase4.rs` against a full decode.
//!
//! ## `SAMCORE02` layout (mirrored from `core_pack.rs`)
//! ```text
//! header[68]: magic[8]="SAMCORE2" | version u32=3 | count u64
//!             | t_min u64 | t_max u64 | mean f64 | std f64
//!             | mode[4] u8 | width[4] u8 | checksum u64
//! payload   : col t | col s | col r | col amount | senders | receivers
//! ```
//! Column modes: `t` is `DELTA(1)` (uleb128 delta from `t_min`) or
//! `FIXED(0)`; `s`/`r` are always `FIXED` codes of `width` bytes; `amount`
//! is `ZIGZAG(2)` or `FIXED`. The stored checksum is `fnv_cover` over the
//! header-minus-magic-and-checksum followed by the whole payload
//! (§R5f; fail-closed — a mutation anywhere breaks it).
//!
//! ## Bounded memory (task §2B)
//! The compressed payload bytes are held once (checksum verification and the
//! column-block seek require random access into them; the canonical loader
//! reads the same bytes via `std::fs::read`). Decoding is incremental: each
//! column keeps only its current read cursor, and `next_record` decodes exactly
//! one record per call. The full multi-million-record `Vec<CoreRecord>`
//! that `deserialize_core_v2` builds is never materialized.

use abacus_tensor::prims::{read_len_str, read_u32, read_u64};
use abacus_tensor::varint::{fnv_cover, read_fixed, read_i64_zigzag, read_uleb128};
use abacus_tensor::TensorError;

/// `SAMCORE02` magic (`core_pack.rs::MAGIC`).
const MAGIC: &[u8; 8] = b"SAMCORE2";
/// Supported format version (`core_pack.rs::VERSION`).
const VERSION: u32 = 3;
/// Fixed header length (`core_pack.rs::HEADER_LEN`).
const HEADER_LEN: usize = 68;
/// Number of encoded columns (`t, s, r, amount`).
const COLS: usize = 4;
/// Max accepted record count (bounds a hostile header; `core_pack.rs`).
const MAX_RECORDS: u64 = 100_000_000;
/// Max accepted dictionary entries.
const MAX_DICT_ENTRIES: usize = 50_000_000;
/// Max accepted dictionary string length.
const MAX_DICT_STR: u32 = 10_000_000;

const MODE_FIXED: u8 = 0;
const MODE_DELTA: u8 = 1;
const MODE_ZIGZAG: u8 = 2;

/// One decoded transaction record: the exact canonical `(t, s, r, amount)`
/// key+value that `deserialize_core_v2` yields (P1/P2). The three P11
/// weighted columns are omitted — they are total functions of `amount_micro`
/// and the stored moments and are never needed for structural streaming_ingestion_demo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamRecord {
    /// Event time `t` (epoch seconds).
    pub t: u64,
    /// Sender dictionary code.
    pub s: u32,
    /// Receiver dictionary code.
    pub r: u32,
    /// Canonical amount in micro-units (P2). Ingested, never displayed.
    pub amount_micro: i64,
}

/// Per-column decode cursor: absolute byte offset into the payload plus the
/// running state a delta column needs (`prev` for `t`).
#[derive(Debug, Clone, Copy)]
struct ColCursor {
    /// Absolute offset into the payload slice.
    off: usize,
    /// Last decoded `t` (delta mode only); unused for other columns.
    prev: u64,
    /// Column mode (`core_pack.rs`).
    mode: u8,
    /// Column fixed width in bytes (`core_pack.rs`).
    width: u8,
}

/// Streaming `SAMCORE02` reader. Holds the payload once, decodes in bounded
/// batches, and exposes the sender/receiver dictionaries needed to resolve
/// codes to verbatim account identifiers for the `Pipeline::apply` path.
pub struct SamldStreamReader {
    /// Payload bytes (everything after the 68-byte header).
    payload: Vec<u8>,
    /// `t_min` from the header (delta base / fixed offset base).
    t_min: u64,
    /// Total record count (header).
    count: usize,
    /// Records decoded and emitted so far.
    produced: usize,
    /// Column cursors `[t, s, r, amount]`.
    cols: [ColCursor; COLS],
    /// Sender dictionary (`prec_dict` ascending == code order).
    senders: Vec<String>,
    /// Receiver dictionary.
    receivers: Vec<String>,
}

impl SamldStreamReader {
    /// Read + validate a `SAMCORE02` image from `path` and prepare streaming.
    ///
    /// Fail-closed (`R_5`): bad magic, unsupported version, oversized count,
    /// header/checksum mismatch, or a malformed column block all reject before
    /// any record is produced — mirroring `deserialize_core_v2`.
    ///
    /// # Errors
    /// [`TensorError::Io`] on unreadable path; [`TensorError::Decode`] on any
    /// structural, ordering, or checksum failure.
    pub fn open(path: &str) -> Result<Self, TensorError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Validate + prepare streaming from raw `SAMCORE02` bytes (test seam).
    ///
    /// # Errors
    /// [`TensorError::Decode`] on any structural or checksum failure.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TensorError> {
        if data.len() < HEADER_LEN {
            return Err(TensorError::Decode("truncated core2 header".into()));
        }
        if &data[..8] != MAGIC {
            return Err(TensorError::Decode("bad core2 magic".into()));
        }
        let mut o = 8usize;
        let version = read_u32(data, &mut o, "core2 version")?;
        if version != VERSION {
            return Err(TensorError::Decode(format!("unsupported core2 version {version}")));
        }
        let count = read_u64(data, &mut o, "core2 count")?;
        if count > MAX_RECORDS {
            return Err(TensorError::Decode("core2 count too large".into()));
        }
        let t_min = read_u64(data, &mut o, "core2 t_min")?;
        let _t_max = read_u64(data, &mut o, "core2 t_max")?;
        // mean / std (P11 moments) are not needed for structural streaming_ingestion_demo.
        let _mean = read_f64_skip(data, &mut o, "core2 mean")?;
        let _std = read_f64_skip(data, &mut o, "core2 std")?;
        if o + 2 * COLS + 8 > data.len() {
            return Err(TensorError::Decode("truncated core2 mode tables".into()));
        }
        let mut modes = [0u8; COLS];
        let mut widths = [0u8; COLS];
        modes.copy_from_slice(&data[o..o + COLS]);
        o += COLS;
        widths.copy_from_slice(&data[o..o + COLS]);
        o += COLS;
        let expect_sum = read_u64(data, &mut o, "core2 checksum")?;
        if o != HEADER_LEN {
            return Err(TensorError::Decode("core2 header length mismatch".into()));
        }
        let payload = data[HEADER_LEN..].to_vec();
        // §R5f: checksum covers header-minus-magic-and-checksum (data[..60])
        // followed by the whole payload — fail-closed.
        let covered: &[u8] = &data[..HEADER_LEN - 8];
        if fnv_cover(covered, &payload) != expect_sum {
            return Err(TensorError::Decode(
                "checksum mismatch [core2 header+payload]".into(),
            ));
        }
        // Column cursors: `t` is DELTA or FIXED; `s`/`r` always FIXED; `amount`
        // is ZIGZAG or FIXED. The four column blocks are contiguous, followed
        // by the two dictionary blocks.
        let cols = [
            ColCursor { off: 0, prev: t_min, mode: modes[0], width: widths[0] },
            ColCursor { off: 0, prev: 0, mode: modes[1], width: widths[1] },
            ColCursor { off: 0, prev: 0, mode: modes[2], width: widths[2] },
            ColCursor { off: 0, prev: 0, mode: modes[3], width: widths[3] },
        ];
        let n = count as usize;
        let mut reader = Self {
            payload,
            t_min,
            count: n,
            produced: 0,
            cols,
            senders: Vec::new(),
            receivers: Vec::new(),
        };
        // Locate the four column-block boundaries by a throwaway full decode of
        // the two variable-length columns (`t`, `amount`); the fixed-width `s`/
        // `r` blocks are `n * width` bytes. O(n) time, O(1) extra memory — the
        // per-record `Vec` `deserialize_core_v2` builds is never materialized.
        let t_end = reader.measure_t(n)?;
        let s_w = reader.cols[1].width as usize;
        let s_end =
            t_end.checked_add(n.checked_mul(s_w).ok_or_else(overflow)?).ok_or_else(overflow)?;
        let r_w = reader.cols[2].width as usize;
        let r_end =
            s_end.checked_add(n.checked_mul(r_w).ok_or_else(overflow)?).ok_or_else(overflow)?;
        let amt_end = reader.measure_amount(n, r_end)?;
        // Dictionaries now sit at `amt_end`.
        let mut d = amt_end;
        reader.senders = read_dict(&reader.payload, &mut d, "core2 sender")?;
        reader.receivers = read_dict(&reader.payload, &mut d, "core2 receiver")?;
        if d != reader.payload.len() {
            return Err(TensorError::Decode("trailing bytes in core2".into()));
        }
        // Point each column cursor at the head of its block for streaming.
        reader.cols[0].off = 0;
        reader.cols[0].prev = t_min;
        reader.cols[1].off = t_end;
        reader.cols[2].off = s_end;
        reader.cols[3].off = r_end;
        reader.produced = 0;
        Ok(reader)
    }

    /// Total record count in the artifact (header `count`).
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Records already emitted by [`Self::next_record`].
    #[must_use]
    pub fn produced(&self) -> usize {
        self.produced
    }

    /// Borrow the sender dictionary (code -> verbatim identifier).
    #[must_use]
    pub fn senders(&self) -> &[String] {
        &self.senders
    }

    /// Borrow the receiver dictionary (code -> verbatim identifier).
    #[must_use]
    pub fn receivers(&self) -> &[String] {
        &self.receivers
    }

    /// Decode the next up-to-`limit` records in canonical `(t,s,r)` order,
    /// advancing the internal cursors. Returns an empty vector once exhausted.
    /// Never allocates more than the returned batch (bounded, task §2B).
    ///
    /// Decode exactly one record in canonical `(t,s,r)` order, advancing the
    /// internal cursors by a single entry. Returns `Ok(None)` once the source
    /// is exhausted. This is the per-record streaming primitive: each call
    /// reads exactly one record and advances the cursor by one — there is no
    /// read-batch-then-apply-batch step.
    ///
    /// # Errors
    /// [`TensorError::Decode`] on a truncated or malformed column block.
    pub fn next_record(&mut self) -> Result<Option<StreamRecord>, TensorError> {
        if self.produced >= self.count {
            return Ok(None);
        }
        let rec = self.decode_one()?;
        self.produced += 1;
        Ok(Some(rec))
    }

    /// Decode exactly one record at the current cursor position, advancing
    /// every column cursor by one entry.
    fn decode_one(&mut self) -> Result<StreamRecord, TensorError> {
        let t = self.next_t()?;
        let s = self.next_code(1, "core2 s")?;
        let r = self.next_code(2, "core2 r")?;
        let amount = self.next_amount()?;
        Ok(StreamRecord { t, s, r, amount_micro: amount })
    }

    /// Advance the `t` column by one (DELTA or FIXED).
    fn next_t(&mut self) -> Result<u64, TensorError> {
        let c = self.cols[0];
        match c.mode {
            MODE_DELTA => {
                let dt = read_uleb128(&self.payload, &mut self.cols[0].off, "core2 t")?;
                let t = c
                    .prev
                    .checked_add(dt)
                    .ok_or_else(|| TensorError::Decode("core2 t overflow".into()))?;
                self.cols[0].prev = t;
                Ok(t)
            }
            MODE_FIXED => {
                let off = read_fixed(&self.payload, &mut self.cols[0].off, c.width, "core2 t")?;
                self.t_min
                    .checked_add(off)
                    .ok_or_else(|| TensorError::Decode("core2 t overflow".into()))
            }
            m => Err(TensorError::Decode(format!("core2 bad t mode {m}"))),
        }
    }

    /// Advance a fixed-width code column (`s` or `r`) by one.
    fn next_code(&mut self, idx: usize, field: &str) -> Result<u32, TensorError> {
        let c = self.cols[idx];
        let v = read_fixed(&self.payload, &mut self.cols[idx].off, c.width, field)?;
        Ok(v as u32)
    }

    /// Advance the `amount` column by one (ZIGZAG or FIXED).
    fn next_amount(&mut self) -> Result<i64, TensorError> {
        let c = self.cols[3];
        match c.mode {
            MODE_ZIGZAG => read_i64_zigzag(&self.payload, &mut self.cols[3].off, "core2 amount"),
            MODE_FIXED => Ok(read_fixed(
                &self.payload,
                &mut self.cols[3].off,
                c.width,
                "core2 amount",
            )? as i64),
            m => Err(TensorError::Decode(format!("core2 bad amount mode {m}"))),
        }
    }

    /// Decode the full `t` column in a throwaway cursor to find where column 1
    /// begins (the `t` block's end offset). O(n) time, O(1) extra memory.
    fn measure_t(&self, n: usize) -> Result<usize, TensorError> {
        let mut off = 0usize;
        let mut prev = self.t_min;
        match self.cols[0].mode {
            MODE_DELTA => {
                for _ in 0..n {
                    let dt = read_uleb128(&self.payload, &mut off, "core2 t")?;
                    prev = prev
                        .checked_add(dt)
                        .ok_or_else(|| TensorError::Decode("core2 t overflow".into()))?;
                }
            }
            MODE_FIXED => {
                off = n.checked_mul(self.cols[0].width as usize).ok_or_else(overflow)?;
            }
            m => return Err(TensorError::Decode(format!("core2 bad t mode {m}"))),
        }
        Ok(off)
    }

    /// Decode the full `amount` column from `start` to find the dictionary
    /// boundary. O(n) time, O(1) extra memory.
    fn measure_amount(&self, n: usize, start: usize) -> Result<usize, TensorError> {
        let mut off = start;
        match self.cols[3].mode {
            MODE_ZIGZAG => {
                for _ in 0..n {
                    let _ = read_i64_zigzag(&self.payload, &mut off, "core2 amount")?;
                }
            }
            MODE_FIXED => {
                off = start
                    .checked_add(
                        n.checked_mul(self.cols[3].width as usize).ok_or_else(overflow)?,
                    )
                    .ok_or_else(overflow)?;
            }
            m => return Err(TensorError::Decode(format!("core2 bad amount mode {m}"))),
        }
        Ok(off)
    }
}

/// Allocate an overflow error (shared by the `checked_*` closures above).
fn overflow() -> TensorError {
    TensorError::Decode("core2 column length overflow".into())
}

/// Skip 8 bytes as an f64 (`core_bin::read_f64` reads LE f64; the value is a
/// P11 moment never used in structural streaming_ingestion_demo, so only bounds are checked).
fn read_f64_skip(d: &[u8], o: &mut usize, field: &str) -> Result<f64, TensorError> {
    if *o + 8 > d.len() {
        return Err(TensorError::Decode(format!("truncated {field}")));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[*o..*o + 8]);
    *o += 8;
    Ok(f64::from_le_bytes(b))
}

/// Read one length-prefixed dictionary block (`prims::read_len_str`), bounded
/// by the same caps as `deserialize_core_v2`.
fn read_dict(d: &[u8], o: &mut usize, field: &str) -> Result<Vec<String>, TensorError> {
    let m = read_u64(d, o, "core2 dict count")? as usize;
    if m > MAX_DICT_ENTRIES {
        return Err(TensorError::Decode("core2 dict too large".into()));
    }
    let mut tab = Vec::with_capacity(m.min(1 << 16));
    for _ in 0..m {
        tab.push(read_len_str(d, o, MAX_DICT_STR, field)?);
    }
    Ok(tab)
}


#[cfg(test)]
mod tests {
    use super::*;
    use abacus_tensor::samld::{deserialize_core_v2, serialize_core_v2, CoreImage, CoreRecord};

    /// Build a canonical `SAMCORE02` image from hand-written records.
    fn build(records: &[(u64, u32, u32, i64)], senders: &[&str], receivers: &[&str]) -> Vec<u8> {
        let recs: Vec<CoreRecord> = records
            .iter()
            .map(|&(t, s, r, a)| CoreRecord {
                t,
                s,
                r,
                amount_micro: a,
                log1p_amt: 0.0,
                z_amt: 0.0,
                roundedness: 1,
            })
            .collect();
        let t_min = records.iter().map(|r| r.0).min().unwrap_or(0);
        let t_max = records.iter().map(|r| r.0).max().unwrap_or(0);
        let img = CoreImage {
            records: recs,
            senders: senders.iter().map(|s| (*s).to_string()).collect(),
            receivers: receivers.iter().map(|s| (*s).to_string()).collect(),
            t_min,
            t_max,
            mean_major: 1.0,
            std_major: 1.0,
            record_count: records.len() as u64,
        };
        serialize_core_v2(&img)
    }

    #[test]
    fn stream_matches_full_decode() {
        let recs = [
            (100u64, 0u32, 0u32, 1_000_000i64),
            (100, 0, 1, 2_500_000),
            (101, 1, 0, -3_000_000),
            (250, 1, 1, 7_000_000),
            (250, 1, 1, 7_000_000), // duplicate coordinate (P4 permits)
        ];
        let bytes = build(&recs, &["alice", "bob"], &["carol", "dave"]);
        let full = deserialize_core_v2(&bytes).unwrap();
        let mut rd = SamldStreamReader::from_bytes(&bytes).unwrap();
        assert_eq!(rd.count(), full.records.len());
        // Stream record-by-record to exercise cursor continuity.
        let mut got = Vec::new();
        while let Some(rec) = rd.next_record().unwrap() {
            got.push(rec);
        }
        assert_eq!(got.len(), full.records.len());
        for (g, f) in got.iter().zip(full.records.iter()) {
            assert_eq!((g.t, g.s, g.r, g.amount_micro), (f.t, f.s, f.r, f.amount_micro));
        }
        assert_eq!(rd.senders(), &full.senders[..]);
        assert_eq!(rd.receivers(), &full.receivers[..]);
        // Exhausted: further reads return None.
        assert!(rd.next_record().unwrap().is_none());
    }

    #[test]
    fn stream_fails_closed_on_corruption() {
        let mut bytes = build(&[(1, 0, 0, 1_000_000)], &["a"], &["b"]);
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        assert!(SamldStreamReader::from_bytes(&bytes).is_err());
    }
}

