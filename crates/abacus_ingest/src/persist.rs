//! Durable append-only ingestion journal (Golden Source Amendment A2).
//!
//! The journal is the single write-ahead record of every state change the
//! ingestion service acknowledges: session creation, chunk acceptance,
//! record acceptance, identity conflicts, lifecycle transitions, and each
//! substrate event applied through the CAS path. Recovery streaming_ingestion_demos it through
//! the *same* public interfaces the live path uses (`IngestionSession`
//! restore methods, `SubstrateApplier::streaming_ingestion_demo_applied` -> `IngestionEngine`
//! CAS), so there is exactly one update path and no parallel state machine.
//!
//! ## Format `ABACJRN1` version 1
//!
//! ```text
//! file header: magic[8]="ABACJRN1" | u16 version=1 | u16 reserved=0
//! record:      u32 payload_len (LE) | payload | [u8;32] SHA-256(payload)
//! payload:     u8 tag | field bytes (fixed-width LE, length-prefixed strings)
//! ```
//!
//! Field-by-field encoding only — never a struct memory-dump (wire §7
//! discipline). Every record is checksummed with the SHA-256 already mandated
//! by the wire contract §4. `payload_len` is bounded before allocation so a
//! corrupt length cannot trigger unbounded allocation.
//!
//! ## Durability and atomicity
//!
//! Each `append` writes one framed record, flushes, and `sync_data`s before
//! returning, and the service only acknowledges an operation after its
//! journal records are durable. A crash therefore leaves a journal prefix of
//! acknowledged operations; a partially written *tail* record (bad length,
//! short read, or checksum failure with no bytes after it) is truncated to
//! the last good record on recovery. Corruption in the *middle* of the file
//! fails closed — it is never silently repaired.
//!
//! ## Consistency model — what is and is NOT atomic
//!
//! The journal and the substrate CAS are **not one atomic filesystem
//! transaction**, and nothing here claims they are. The substrate
//! (`abacus_tensor::ingest::IngestionEngine`) has **no durable state of its
//! own**: on restart it is rebuilt entirely by streaming_ingestion_demoing this journal through
//! `SubstrateApplier::streaming_ingestion_demo_applied` -> `IngestionEngine::ingest`. The
//! substrate is therefore a *deterministic function of the journal*, and the
//! journal is the single durable authority.
//!
//! That fact is what makes the system crash-safe despite the non-atomic
//! ordering. The per-event order in `commit` is: substrate CAS commit, then
//! journal frame `fsync`, before the next event. The windows this opens are
//! all closed by that determinism plus CAS idempotency:
//!
//! - **Power loss after a substrate CAS but before its journal frame is
//!   durable:** the in-memory substrate held the event, but it is lost with
//!   the process; recovery rebuilds the substrate from the journal, which does
//!   not contain the event. The client retries; the substrate CAS sees the id
//!   absent and applies it exactly once. No durable divergence is possible
//!   because there is no durable substrate to diverge.
//! - **Journal append I/O error (disk full, etc.) mid-commit:** the sink
//!   returns `ErrorCode::Persistence`, which poisons the service until restart
//!   (fail-closed), so in-memory state that is ahead of the journal is never
//!   served. Recovery restores the journal-authoritative prefix.
//! - **Crash after the last `EventApplied` frame but before `SessionCommitted`:**
//!   the session is restored not-yet-committed with all events already applied;
//!   a retry reports each event already-applied and applies nothing new.
//!
//! **Residual limitation (unchanged, documented, not eliminated):** the window
//! between an event's CAS commit and its journal frame's `fsync` is a real
//! window. It is *mitigated* — by journal-before-next ordering, by CAS
//! idempotency making any retry safe, and by fail-closed poisoning on sink
//! failure — but a true power loss inside it is not fully excluded. Because
//! the substrate is non-durable, the worst outcome is a single lost event that
//! the client's retry re-applies, never a substrate that silently disagrees
//! with the durable journal.
//!
//! ## Privacy (Amendment A1)
//!
//! The journal carries no raw attested amounts in transport form. The only
//! value persisted is `EVENT_APPLIED.value_micro` — the exact substrate
//! micro-unit entry, i.e. the *result* of the required transformation, whose
//! durable presence is the substrate's own canonical state (identical in kind
//! to fixture-derived amounts living in the tensor). Transient amount copies
//! (commit request bodies, attestation vectors) are never journaled; they are
//! held in zero-on-drop guards (see [`crate::secret`]) and cleared on every
//! reachable exit of the commit path.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::config::{CurrencyCode, IdentityScheme, SourceConfig, SourceType};
use crate::error::{ErrorCode, IngestError, IngestResult};
use crate::features::FeaturePayload;
use crate::hash::sha256;
use crate::namespace::EventNamespace;
use crate::record::CanonicalRecord;
use crate::session::{AppliedEvent, SessionLimits};

/// Journal file name inside the data directory.
pub const JOURNAL_FILE: &str = "ingestion_journal.bin";

const MAGIC: &[u8; 8] = b"ABACJRN1";
const VERSION: u16 = 1;
const HEADER_LEN: u64 = 12;
/// Hard bound on one payload, enforced before allocation.
const MAX_PAYLOAD: u32 = 1 << 20;
/// Bound on a restored string field (session ids are short by construction).
const MAX_STRING: usize = 512;

const TAG_SESSION_OPEN: u8 = 2;
const TAG_CHUNK_ACCEPTED: u8 = 3;
const TAG_RECORD_ACCEPTED: u8 = 4;
const TAG_RECORD_CONFLICT: u8 = 5;
const TAG_SESSION_SEALED: u8 = 6;
const TAG_SESSION_CANCELLED: u8 = 7;
const TAG_EVENT_APPLIED: u8 = 8;
const TAG_SESSION_COMMITTED: u8 = 9;

fn persistence_err(detail: impl Into<String>) -> IngestError {
    IngestError::new(ErrorCode::Persistence, detail)
}

/// One decoded journal record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalRecord {
    /// A session was created and bound to a namespace and configuration.
    SessionOpen {
        /// Session id.
        id: String,
        /// Bound namespace.
        namespace: EventNamespace,
        /// Bound configuration.
        config: SourceConfig,
        /// Server-enforced limits in force at creation.
        limits: SessionLimits,
    },
    /// A chunk was accepted (idempotency bookkeeping).
    ChunkAccepted {
        /// Session id.
        id: String,
        /// Chunk sequence.
        sequence: u64,
        /// SHA-256 of the exact chunk bytes.
        digest: [u8; 32],
        /// Records decoded from the chunk.
        record_count: u64,
        /// Payload bytes.
        payload_bytes: u64,
        /// New events in this chunk.
        new_events: u64,
        /// Duplicate events in this chunk.
        duplicate_events: u64,
    },
    /// One record was accepted into the session buffer.
    RecordAccepted {
        /// Session id.
        id: String,
        /// The accepted canonical record (carries no amount).
        record: CanonicalRecord,
    },
    /// One conflicting identity was observed and reported.
    RecordConflict {
        /// Session id.
        id: String,
    },
    /// The session was sealed.
    SessionSealed {
        /// Session id.
        id: String,
    },
    /// The session was cancelled.
    SessionCancelled {
        /// Session id.
        id: String,
    },
    /// One event was applied to the substrate through the CAS path.
    EventApplied {
        /// Session id.
        id: String,
        /// The applied event (substrate coordinates, value, wire ids).
        event: AppliedEvent,
    },
    /// The commit completed; the session is now committed.
    SessionCommitted {
        /// Session id.
        id: String,
        /// Events committed in this commit call.
        applied: u64,
        /// Events the CAS reported as already applied.
        already_applied: u64,
    },
}

// ---------------------------------------------------------------------------
// Field-wise encode helpers (little-endian, no struct dumps).
// ---------------------------------------------------------------------------

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, s: &str) -> IngestResult<()> {
    let b = s.as_bytes();
    if b.len() > MAX_STRING {
        return Err(persistence_err("string field too long to journal"));
    }
    put_u16(out, b.len() as u16);
    out.extend_from_slice(b);
    Ok(())
}

impl JournalRecord {
    fn encode(&self) -> IngestResult<Vec<u8>> {
        let mut p = Vec::new();
        match self {
            Self::SessionOpen { id, namespace, config, limits } => {
                p.push(TAG_SESSION_OPEN);
                put_str(&mut p, id)?;
                p.extend_from_slice(&namespace.to_bytes());
                p.push(config.source_type.to_u8());
                p.push(config.identity_scheme.to_u8());
                p.push(config.account_id_width);
                p.extend_from_slice(&config.currency_code.bytes());
                p.push(config.currency_precision);
                put_u64(&mut p, config.threshold_minor_units);
                put_u32(&mut p, config.threshold_effective_day as u32);
                put_u32(&mut p, config.threshold_config_version);
                put_u16(&mut p, config.feature_schema_version);
                put_u16(&mut p, config.feature_algorithm_version);
                put_u32(&mut p, config.parsing_policy_version);
                put_u64(&mut p, config.max_amount_minor_units);
                put_u64(&mut p, limits.max_chunk_bytes);
                put_u64(&mut p, limits.max_chunk_records);
                put_u64(&mut p, limits.max_session_bytes);
                put_u64(&mut p, limits.max_session_records);
                put_u64(&mut p, limits.max_chunks);
            }
            Self::ChunkAccepted {
                id,
                sequence,
                digest,
                record_count,
                payload_bytes,
                new_events,
                duplicate_events,
            } => {
                p.push(TAG_CHUNK_ACCEPTED);
                put_str(&mut p, id)?;
                put_u64(&mut p, *sequence);
                p.extend_from_slice(digest);
                put_u64(&mut p, *record_count);
                put_u64(&mut p, *payload_bytes);
                put_u64(&mut p, *new_events);
                put_u64(&mut p, *duplicate_events);
            }
            Self::RecordAccepted { id, record } => {
                p.push(TAG_RECORD_ACCEPTED);
                put_str(&mut p, id)?;
                put_u64(&mut p, record.identity.event_id);
                put_i64(&mut p, record.event_time_ms);
                put_u64(&mut p, record.sender_id);
                put_u64(&mut p, record.receiver_id);
                put_u16(&mut p, record.features.log_magnitude);
                put_u32(&mut p, record.features.relative_threshold_distance as u32);
                p.push(record.features.roundedness_flags);
            }
            Self::RecordConflict { id } => {
                p.push(TAG_RECORD_CONFLICT);
                put_str(&mut p, id)?;
            }
            Self::SessionSealed { id } => {
                p.push(TAG_SESSION_SEALED);
                put_str(&mut p, id)?;
            }
            Self::SessionCancelled { id } => {
                p.push(TAG_SESSION_CANCELLED);
                put_str(&mut p, id)?;
            }
            Self::EventApplied { id, event } => {
                p.push(TAG_EVENT_APPLIED);
                put_str(&mut p, id)?;
                put_u64(&mut p, event.substrate_event_id);
                put_u64(&mut p, event.t);
                put_u32(&mut p, event.s);
                put_u32(&mut p, event.r);
                put_i64(&mut p, event.value_micro);
                put_u64(&mut p, event.sender_wire_id);
                put_u64(&mut p, event.receiver_wire_id);
            }
            Self::SessionCommitted { id, applied, already_applied } => {
                p.push(TAG_SESSION_COMMITTED);
                put_str(&mut p, id)?;
                put_u64(&mut p, *applied);
                put_u64(&mut p, *already_applied);
            }
        }
        Ok(p)
    }
}


// ---------------------------------------------------------------------------
// Field-wise decode (bounded cursor; every failure is a categorized error).
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> IngestResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| persistence_err("cursor overflow"))?;
        if end > self.buf.len() {
            return Err(persistence_err("truncated journal payload"));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> IngestResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> IngestResult<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("len 2")))
    }
    fn u32(&mut self) -> IngestResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("len 4")))
    }
    fn u64(&mut self) -> IngestResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("len 8")))
    }
    fn i32(&mut self) -> IngestResult<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().expect("len 4")))
    }
    fn i64(&mut self) -> IngestResult<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().expect("len 8")))
    }
    fn str(&mut self) -> IngestResult<String> {
        let n = usize::from(self.u16()?);
        if n > MAX_STRING {
            return Err(persistence_err("journal string field too long"));
        }
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| persistence_err("journal string is not UTF-8"))
    }
    fn done(&self) -> bool {
        self.pos == self.buf.len()
    }
}

impl JournalRecord {
    /// Decode one payload. `namespace_hint` rebinds the per-record identity to
    /// the session's single namespace (§7.2: the namespace lives once, in the
    /// session/segment binding — the journal mirrors that discipline).
    fn decode(payload: &[u8], namespace_hint: Option<EventNamespace>) -> IngestResult<Self> {
        let mut c = Cursor::new(payload);
        let tag = c.u8()?;
        let rec = match tag {
            TAG_SESSION_OPEN => {
                let id = c.str()?;
                let ns = EventNamespace::from_bytes(c.take(16)?.try_into().expect("len 16"));
                let source_type = SourceType::from_u8(c.u8()?)?;
                let identity_scheme = IdentityScheme::from_u8(c.u8()?)?;
                let account_id_width = c.u8()?;
                let cur = CurrencyCode::parse(
                    std::str::from_utf8(c.take(3)?)
                        .map_err(|_| persistence_err("journal currency code is not ASCII"))?,
                )
                .map_err(|e| persistence_err(format!("journal currency code invalid: {e}")))?;
                let currency_precision = c.u8()?;
                let threshold_minor_units = c.u64()?;
                let threshold_effective_day = c.i32()?;
                let threshold_config_version = c.u32()?;
                let feature_schema_version = c.u16()?;
                let feature_algorithm_version = c.u16()?;
                let parsing_policy_version = c.u32()?;
                let max_amount_minor_units = c.u64()?;
                let limits = SessionLimits {
                    max_chunk_bytes: c.u64()?,
                    max_chunk_records: c.u64()?,
                    max_session_bytes: c.u64()?,
                    max_session_records: c.u64()?,
                    max_chunks: c.u64()?,
                };
                Self::SessionOpen {
                    id,
                    namespace: ns,
                    config: SourceConfig {
                        source_type,
                        identity_scheme,
                        account_id_width,
                        currency_code: cur,
                        currency_precision,
                        threshold_minor_units,
                        threshold_effective_day,
                        threshold_config_version,
                        feature_schema_version,
                        feature_algorithm_version,
                        parsing_policy_version,
                        max_amount_minor_units,
                    },
                    limits,
                }
            }
            TAG_CHUNK_ACCEPTED => Self::ChunkAccepted {
                id: c.str()?,
                sequence: c.u64()?,
                digest: c.take(32)?.try_into().expect("len 32"),
                record_count: c.u64()?,
                payload_bytes: c.u64()?,
                new_events: c.u64()?,
                duplicate_events: c.u64()?,
            },
            TAG_RECORD_ACCEPTED => {
                let id = c.str()?;
                let event_id = c.u64()?;
                let event_time_ms = c.i64()?;
                let sender_id = c.u64()?;
                let receiver_id = c.u64()?;
                let log_magnitude = c.u16()?;
                let relative_threshold_distance = c.i32()?;
                let roundedness_flags = c.u8()?;
                let ns = namespace_hint
                    .ok_or_else(|| persistence_err("record before SESSION_OPEN in journal"))?;
                Self::RecordAccepted {
                    id,
                    record: CanonicalRecord {
                        identity: crate::namespace::EventIdentity::new(ns, event_id),
                        event_time_ms,
                        sender_id,
                        receiver_id,
                        features: FeaturePayload {
                            log_magnitude,
                            relative_threshold_distance,
                            roundedness_flags,
                        },
                    },
                }
            }
            TAG_RECORD_CONFLICT => Self::RecordConflict { id: c.str()? },
            TAG_SESSION_SEALED => Self::SessionSealed { id: c.str()? },
            TAG_SESSION_CANCELLED => Self::SessionCancelled { id: c.str()? },
            TAG_EVENT_APPLIED => Self::EventApplied {
                id: c.str()?,
                event: AppliedEvent {
                    substrate_event_id: c.u64()?,
                    t: c.u64()?,
                    s: c.u32()?,
                    r: c.u32()?,
                    value_micro: c.i64()?,
                    sender_wire_id: c.u64()?,
                    receiver_wire_id: c.u64()?,
                },
            },
            TAG_SESSION_COMMITTED => Self::SessionCommitted {
                id: c.str()?,
                applied: c.u64()?,
                already_applied: c.u64()?,
            },
            other => {
                return Err(persistence_err(format!("unknown journal record tag {other}")));
            }
        };
        if !c.done() {
            return Err(persistence_err("journal payload has trailing bytes"));
        }
        Ok(rec)
    }
}

// ---------------------------------------------------------------------------
// The journal: durable append + recovery streaming_ingestion_demo.
// ---------------------------------------------------------------------------

/// Outcome of a recovery streaming_ingestion_demo.
#[derive(Debug)]
pub struct StreamingIngestionDemo {
    /// The intact, acknowledged records in order.
    pub records: Vec<JournalRecord>,
    /// True when an incomplete tail frame was truncated during streaming_ingestion_demo.
    pub truncated_tail: bool,
}

/// The durable write-ahead journal for one data directory.
#[derive(Debug)]
pub struct Journal {
    file: Mutex<File>,
    path: PathBuf,
}

fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len() + 32);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&sha256(payload));
    out
}

impl Journal {
    /// Open (creating if needed) the journal inside `dir`, writing the file
    /// header on first creation and verifying it otherwise.
    ///
    /// # Errors
    /// [`ErrorCode::Persistence`] on I/O failure or a header mismatch.
    pub fn open(dir: &Path) -> IngestResult<Self> {
        std::fs::create_dir_all(dir)
            .map_err(|e| persistence_err(format!("cannot create journal directory: {e}")))?;
        let path = dir.join(JOURNAL_FILE);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            // Never truncate on open: an existing journal is the durable,
            // streaming_ingestion_demo-authoritative state. Truncating here would destroy the
            // very records recovery depends on. Made explicit so the append
            // semantics are unambiguous.
            .truncate(false)
            .open(&path)
            .map_err(|e| persistence_err(format!("cannot open journal: {e}")))?;
        let len = file
            .metadata()
            .map_err(|e| persistence_err(format!("journal stat: {e}")))?
            .len();
        if len == 0 {
            let mut header = Vec::with_capacity(HEADER_LEN as usize);
            header.extend_from_slice(MAGIC);
            header.extend_from_slice(&VERSION.to_le_bytes());
            header.extend_from_slice(&0u16.to_le_bytes());
            file.write_all(&header)
                .and_then(|_| file.flush())
                .and_then(|_| file.sync_data())
                .map_err(|e| persistence_err(format!("cannot write journal header: {e}")))?;
        } else {
            let mut header = [0u8; HEADER_LEN as usize];
            file.seek(SeekFrom::Start(0))
                .and_then(|_| file.read_exact(&mut header))
                .map_err(|e| persistence_err(format!("cannot read journal header: {e}")))?;
            if &header[..8] != MAGIC {
                return Err(persistence_err("journal magic mismatch"));
            }
            if u16::from_le_bytes([header[8], header[9]]) != VERSION {
                return Err(persistence_err("journal format version not supported"));
            }
            if header[10] != 0 || header[11] != 0 {
                return Err(persistence_err("journal reserved header bytes must be zero"));
            }
            file.seek(SeekFrom::End(0))
                .map_err(|e| persistence_err(format!("journal seek: {e}")))?;
        }
        Ok(Self { file: Mutex::new(file), path })
    }

    /// The journal file path (diagnostics).
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record, flush, and `sync_data` before returning. The caller
    /// must not acknowledge the corresponding operation until this succeeds.
    ///
    /// # Errors
    /// [`ErrorCode::Persistence`] on I/O failure.
    pub fn append(&self, rec: &JournalRecord) -> IngestResult<()> {
        self.append_batch(std::slice::from_ref(rec))
    }

    /// Append a batch of records under one lock hold with a single trailing
    /// `sync_data`, so a multi-record operation (one uploaded chunk, one
    /// commit) pays one fsync instead of one per record.
    ///
    /// The bytes on disk are exactly what repeated [`Self::append`] calls
    /// would have written — the same frames in the same order, with the same
    /// per-frame checksums — so recovery is unchanged. Durability is
    /// unchanged too: nothing is acknowledged until this returns, and by then
    /// every frame in the batch is durable. A mid-batch I/O failure aborts
    /// fail-closed exactly like a single-append failure.
    ///
    /// # Errors
    /// [`ErrorCode::Persistence`] on I/O failure.
    pub fn append_batch(&self, recs: &[JournalRecord]) -> IngestResult<()> {
        let mut batch = Vec::new();
        for rec in recs {
            let payload = rec.encode()?;
            if payload.len() as u64 > u64::from(MAX_PAYLOAD) {
                return Err(persistence_err("journal payload exceeds the framing bound"));
            }
            batch.extend_from_slice(&frame(&payload));
        }
        let mut f = self
            .file
            .lock()
            .map_err(|_| persistence_err("journal lock poisoned"))?;
        f.seek(SeekFrom::End(0))
            .and_then(|_| f.write_all(&batch))
            .and_then(|_| f.flush())
            .and_then(|_| f.sync_data())
            .map_err(|e| persistence_err(format!("journal append failed: {e}")))
    }
}



impl Journal {
    /// StreamingIngestionDemo every intact record in `dir`'s journal.
    ///
    /// An incomplete final frame (bad length, short payload, missing or wrong
    /// checksum, undecodable body, with nothing valid after it) is the torn
    /// tail of a write that was never acknowledged: it is truncated to the
    /// last good record and `truncated_tail` is set. Corruption with valid
    /// bytes after the failure fails closed and is never silently repaired.
    ///
    /// # Errors
    /// [`ErrorCode::Persistence`] on I/O failure, header mismatch, or
    /// mid-file corruption.
    pub fn streaming_ingestion_demo(dir: &Path) -> IngestResult<StreamingIngestionDemo> {
        let path = dir.join(JOURNAL_FILE);
        if !path.exists() {
            return Ok(StreamingIngestionDemo { records: Vec::new(), truncated_tail: false });
        }
        let file = File::open(&path)
            .map_err(|e| persistence_err(format!("cannot open journal for streaming_ingestion_demo: {e}")))?;
        let total = file
            .metadata()
            .map_err(|e| persistence_err(format!("cannot stat journal: {e}")))?
            .len();
        let mut reader = BufReader::new(file);
        let mut header = [0u8; HEADER_LEN as usize];
        reader
            .read_exact(&mut header)
            .map_err(|e| persistence_err(format!("cannot read journal header: {e}")))?;
        if &header[..8] != MAGIC || u16::from_le_bytes([header[8], header[9]]) != VERSION {
            return Err(persistence_err("journal header not supported"));
        }
        let mut records: Vec<JournalRecord> = Vec::new();
        let mut hints: std::collections::HashMap<String, EventNamespace> =
            std::collections::HashMap::new();
        let mut good_end: u64 = HEADER_LEN;
        let mut truncated_tail = false;
        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Partial length prefix at EOF: torn tail.
                    if good_end < total {
                        truncate_to(&path, good_end)?;
                        truncated_tail = true;
                    }
                    break;
                }
                Err(e) => return Err(persistence_err(format!("journal read failed: {e}"))),
            }
            let plen = u32::from_le_bytes(len_buf);
            let frame_end = match 4u64.checked_add(u64::from(plen)).and_then(|n| n.checked_add(32)) {
                Some(n) => good_end + n,
                None => total + 1,
            };
            let bail = |detail: &str| -> IngestResult<StreamingIngestionDemo> {
                if frame_end >= total {
                    // The failed frame is (part of) the final bytes: an
                    // unacknowledged torn write. Truncate and stop.
                    truncate_to(&path, good_end)?;
                    Ok(StreamingIngestionDemo { records: records.clone(), truncated_tail: true })
                } else {
                    Err(persistence_err(format!(
                        "mid-file journal corruption at offset {good_end}: {detail}"
                    )))
                }
            };
            if plen > MAX_PAYLOAD || frame_end > total {
                return bail("invalid or unbounded payload length");
            }
            let mut payload = vec![0u8; plen as usize];
            if reader.read_exact(&mut payload).is_err() {
                return bail("truncated journal payload");
            }
            let mut digest = [0u8; 32];
            if reader.read_exact(&mut digest).is_err() {
                return bail("truncated journal checksum");
            }
            if sha256(&payload) != digest {
                return bail("journal checksum mismatch");
            }
            let hint = if payload.first() == Some(&TAG_RECORD_ACCEPTED) && payload.len() >= 3 {
                let n = usize::from(u16::from_le_bytes([payload[1], payload[2]]));
                if n > MAX_STRING || 3 + n > payload.len() {
                    return bail("malformed journal id");
                }
                let id = std::str::from_utf8(&payload[3..3 + n])
                    .map_err(|_| persistence_err("journal id is not UTF-8"))?
                    .to_string();
                hints.get(&id).copied()
            } else {
                None
            };
            match JournalRecord::decode(&payload, hint) {
                Ok(rec) => {
                    if let JournalRecord::SessionOpen { id, namespace, .. } = &rec {
                        hints.insert(id.clone(), *namespace);
                    }
                    good_end = frame_end;
                    records.push(rec);
                }
                Err(e) => return bail(&e.detail),
            }
        }
        let _ = reader.into_inner();
        Ok(StreamingIngestionDemo { records, truncated_tail })
    }
}

fn truncate_to(path: &Path, len: u64) -> IngestResult<()> {
    let f = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| persistence_err(format!("cannot open journal for truncate: {e}")))?;
    f.set_len(len)
        .and_then(|_| f.sync_data())
        .map_err(|e| persistence_err(format!("cannot truncate journal tail: {e}")))
}

