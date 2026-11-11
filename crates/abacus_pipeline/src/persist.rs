//! Durable restart recovery: WAL journal + atomic checkpoint over canonical codecs.
//!
//! Golden-source grounding (no invented tensor behavior):
//! - Sealed CSF authority: checkpoint embeds a canonical `SAMCSF02` image via
//!   `csf::{serialize_csf_v2, deserialize_csf_v2}` (checksummed, fail-closed).
//!   No replacement tensor format is introduced.
//! - Identity contracts preserved: event id = next tx_id; duplicate map
//!   `(t,s,r,amount) -> id` rebuilds on streaming_ingestion_demo; locator stability via
//!   `AccountIndex` rebuild + incremental insert.
//! - Reused primitives: `composites::fnv1a64` (d_fnv), `prims` LE codec
//!   (`a_13.1/.2`), `read_len_str` UTF-8 validation (`a_18.4`). Std only.
//!
//! Crash model: a crash must never expose a partially committed state.
//! Commit order per event is journal-first: the journal record is appended +
//! fsynced BEFORE the in-memory commit. Recovery streaming_ingestion_demos only checksum-valid
//! records; a torn tail is truncated, never applied. The checkpoint is
//! tmp-file + fsync + atomic rename, so a crash leaves the old or the new
//! checkpoint — never a half file (a torn checkpoint is ignored in favor of
//! full journal streaming_ingestion_demo).

use crate::error::PipelineError;
use crate::loader::LoadedArtifacts;
use crate::pipeline::{CheckpointImage, NormalizedEvent, Pipeline, PipelineConfig};
use abacus_tensor::composites::fnv1a64;
use abacus_tensor::prims::{read_len_str, read_u64, write_len_str, write_u64};

/// Journal magic + version (frames the WAL only, never the tensor format).
pub const JOURNAL_MAGIC: &[u8; 8] = b"ABACJNL1";
/// Journal version.
pub const JOURNAL_VERSION: u32 = 1;
/// Checkpoint magic + version.
pub const CHECKPOINT_MAGIC: &[u8; 8] = b"ABACCKPT";
/// Checkpoint version.
pub const CHECKPOINT_VERSION: u32 = 1;
const MAX_STR: u32 = 10_000_000;
const MAX_DICT_ENTRIES: usize = 50_000_000;

/// Encode one journal payload: `t | sender | receiver | amount_micro`.
fn encode_event_payload(ev: &NormalizedEvent) -> Vec<u8> {
    let mut b = Vec::new();
    write_u64(&mut b, ev.t);
    write_len_str(&mut b, &ev.sender);
    write_len_str(&mut b, &ev.receiver);
    b.extend_from_slice(&ev.amount_micro.to_le_bytes());
    b
}

fn decode_event_payload(data: &[u8]) -> Result<NormalizedEvent, PipelineError> {
    let mut o = 0usize;
    let t = read_u64(data, &mut o, "journal t").map_err(PipelineError::Tensor)?;
    let sender =
        read_len_str(data, &mut o, MAX_STR, "journal sender").map_err(PipelineError::Tensor)?;
    let receiver =
        read_len_str(data, &mut o, MAX_STR, "journal receiver").map_err(PipelineError::Tensor)?;
    if o + 8 > data.len() {
        return Err(PipelineError::Mismatch("journal truncated amount".into()));
    }
    let mut ab = [0u8; 8];
    ab.copy_from_slice(&data[o..o + 8]);
    o += 8;
    if o != data.len() {
        return Err(PipelineError::Mismatch("journal trailing bytes".into()));
    }
    Ok(NormalizedEvent { t, sender, receiver, amount_micro: i64::from_le_bytes(ab) })
}

/// Frame a payload: `MAGIC | version u32 | len u64 | payload | checksum u64`.
/// The checksum is `d_fnv` over the payload only.
fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(JOURNAL_MAGIC);
    out.extend_from_slice(&JOURNAL_VERSION.to_le_bytes());
    write_u64(&mut out, payload.len() as u64);
    out.extend_from_slice(payload);
    write_u64(&mut out, fnv1a64(payload));
    out
}

fn sync_parent(path: &str) -> Result<(), PipelineError> {
    let parent = std::path::Path::new(path)
        .parent()
        .map(|p| p.as_os_str().to_owned())
        .unwrap_or_default();
    let dir = if parent.is_empty() {
        std::fs::File::open(".")?
    } else {
        std::fs::File::open(std::path::Path::new(&parent))?
    };
    dir.sync_all()?;
    Ok(())
}
/// Append one committed event to the journal: open-append, write, fsync.
/// Called BEFORE the in-memory commit; failure leaves state unchanged.
pub fn journal_append(journal_path: &str, ev: &NormalizedEvent) -> Result<(), PipelineError> {
    use std::fs::OpenOptions;
    use std::io::Write;
    let rec = frame(&encode_event_payload(ev));
    let mut f = OpenOptions::new().create(true).append(true).open(journal_path)?;
    f.write_all(&rec)?;
    f.sync_all()?;
    Ok(())
}

/// Read all valid journal records. A torn tail (short read / checksum
/// mismatch on the LAST record only) truncates: returns `(events, torn)`.
/// A checksum mismatch with more bytes after it is mid-stream corruption →
/// hard error (never skip: that would silently reorder history).
pub fn journal_read(journal_path: &str) -> Result<(Vec<NormalizedEvent>, bool), PipelineError> {
    let data = match std::fs::read(journal_path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), false)),
        Err(e) => return Err(PipelineError::from(e)),
    };
    let mut events = Vec::new();
    let mut o = 0usize;
    while o < data.len() {
        let rec_start = o;
        if data.len() - o < 8 + 4 + 8 {
            // Torn record header → truncate whole tail.
            return Ok((events, true));
        }
        if &data[o..o + 8] != JOURNAL_MAGIC {
            return Err(PipelineError::Mismatch("journal bad magic".into()));
        }
        o += 8;
        let ver =
            u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        if ver != JOURNAL_VERSION {
            return Err(PipelineError::Mismatch(format!("journal bad version {ver}")));
        }
        o += 4;
        let mut oo = o;
        let len = match read_u64(&data, &mut oo, "journal len") {
            Ok(v) => v as usize,
            Err(_) => return Ok((events, true)),
        };
        o = oo;
        if len > data.len() {
            return Err(PipelineError::Mismatch("journal absurd length".into()));
        }
        if data.len() - o < len + 8 {
            // Torn payload/checksum → truncate this record and everything after.
            let _ = rec_start;
            return Ok((events, true));
        }
        let payload = &data[o..o + len];
        o += len;
        let mut co = o;
        let expect = match read_u64(&data, &mut co, "journal checksum") {
            Ok(v) => v,
            Err(_) => return Ok((events, true)),
        };
        o = co;
        if fnv1a64(payload) != expect {
            if o == data.len() {
                // Last record with bad checksum: torn write → truncate it.
                return Ok((events, true));
            }
            return Err(PipelineError::Mismatch("journal checksum mismatch mid-stream".into()));
        }
        events.push(decode_event_payload(payload)?);
    }
    Ok((events, false))
}

/// Rewrite the journal, dropping any torn tail (compaction after recovery).
pub fn journal_rewrite_clean(
    journal_path: &str,
    events: &[NormalizedEvent],
) -> Result<(), PipelineError> {
    use std::io::Write;
    let tmp = format!("{journal_path}.tmp");
    let mut buf = Vec::new();
    for ev in events {
        buf.extend_from_slice(&frame(&encode_event_payload(ev)));
    }
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&buf)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, journal_path)?;
    sync_parent(journal_path)?;
    Ok(())
}
/// Checkpoint body layout (LE codec, `a_13.1/.2`):
/// `version u64 | committed u64 | t_min u64 | t_max u64 | journal_len u64 |
///  csf_len u64 | csf_bytes | nsenders u64 | senders… | nreceivers u64 |
///  receivers…`. Written tmp + fsync + rename (atomic).
pub fn checkpoint_write(
    checkpoint_path: &str,
    p: &Pipeline,
    journal_compacted_len: usize,
) -> Result<(), PipelineError> {
    use std::io::Write;
    let s = p.snapshot();
    let csf_bytes = abacus_tensor::csf::serialize_csf_v2(
        &s.sealed,
        &s.senders,
        &s.receivers,
        s.t_min,
        s.t_max,
        s.committed_records as u64,
    );
    let mut body = Vec::new();
    write_u64(&mut body, s.version);
    write_u64(&mut body, s.committed_records as u64);
    write_u64(&mut body, s.t_min);
    write_u64(&mut body, s.t_max);
    write_u64(&mut body, journal_compacted_len as u64);
    write_u64(&mut body, csf_bytes.len() as u64);
    body.extend_from_slice(&csf_bytes);
    if s.senders.len() > MAX_DICT_ENTRIES || s.receivers.len() > MAX_DICT_ENTRIES {
        return Err(PipelineError::Mismatch("checkpoint dict too large".into()));
    }
    write_u64(&mut body, s.senders.len() as u64);
    for x in &s.senders {
        write_len_str(&mut body, x);
    }
    write_u64(&mut body, s.receivers.len() as u64);
    for x in &s.receivers {
        write_len_str(&mut body, x);
    }
    let mut out = Vec::new();
    out.extend_from_slice(CHECKPOINT_MAGIC);
    out.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
    write_u64(&mut out, body.len() as u64);
    out.extend_from_slice(&body);
    write_u64(&mut out, fnv1a64(&body));
    let tmp = format!("{checkpoint_path}.tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&out)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, checkpoint_path)?;
    sync_parent(checkpoint_path)?;
    Ok(())
}
/// Checkpoint recovery state: sealed canonical image + scalars + dicts + offset.
#[derive(Debug)]
pub struct CheckpointState {
    /// Canonical sealed CSF image bytes (`SAMCSF02`).
    pub csf_bytes: Vec<u8>,
    /// Committed version at checkpoint.
    pub version: u64,
    /// Committed record count at checkpoint.
    pub committed_records: usize,
    /// Temporal bounds at checkpoint.
    pub t_min: u64,
    /// Temporal bounds at checkpoint.
    pub t_max: u64,
    /// Journal records compacted into this checkpoint.
    pub journal_len: usize,
    /// Full dictionaries at checkpoint.
    pub senders: Vec<String>,
    /// Full dictionaries at checkpoint.
    pub receivers: Vec<String>,
}

/// Load a checkpoint. `Ok(None)` = absent (fresh start) or torn (fail-safe:
/// ignore and streaming_ingestion_demo the full journal). A length-consistent but
/// checksum-invalid image is corruption → hard error.
pub fn checkpoint_read(checkpoint_path: &str) -> Result<Option<CheckpointState>, PipelineError> {
    let data = match std::fs::read(checkpoint_path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(PipelineError::from(e)),
    };
    if data.len() < 8 + 4 + 8 + 8 {
        return Ok(None); // torn header → ignore, streaming_ingestion_demo journal
    }
    if &data[..8] != CHECKPOINT_MAGIC {
        return Err(PipelineError::Mismatch("checkpoint bad magic".into()));
    }
    let ver = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    if ver != CHECKPOINT_VERSION {
        return Err(PipelineError::Mismatch(format!("checkpoint bad version {ver}")));
    }
    let mut o = 12usize;
    let body_len = match read_u64(&data, &mut o, "checkpoint len") {
        Ok(v) => v as usize,
        Err(_) => return Ok(None),
    };
    if data.len() - o < body_len + 8 {
        return Ok(None); // torn body → ignore, streaming_ingestion_demo journal
    }
    let body = &data[o..o + body_len];
    o += body_len;
    let mut co = o;
    let expect = match read_u64(&data, &mut co, "checkpoint checksum") {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    if co != data.len() {
        return Err(PipelineError::Mismatch("checkpoint trailing bytes".into()));
    }
    if fnv1a64(body) != expect {
        return Err(PipelineError::Mismatch("checkpoint checksum mismatch".into()));
    }
    let mut bo = 0usize;
    let version = read_u64(body, &mut bo, "ckpt version").map_err(PipelineError::Tensor)?;
    let committed =
        read_u64(body, &mut bo, "ckpt committed").map_err(PipelineError::Tensor)? as usize;
    let t_min = read_u64(body, &mut bo, "ckpt tmin").map_err(PipelineError::Tensor)?;
    let t_max = read_u64(body, &mut bo, "ckpt tmax").map_err(PipelineError::Tensor)?;
    let journal_len =
        read_u64(body, &mut bo, "ckpt jlen").map_err(PipelineError::Tensor)? as usize;
    let csf_len =
        read_u64(body, &mut bo, "ckpt csflen").map_err(PipelineError::Tensor)? as usize;
    if body.len() - bo < csf_len {
        return Err(PipelineError::Mismatch("checkpoint truncated csf".into()));
    }
    let csf_bytes = body[bo..bo + csf_len].to_vec();
    bo += csf_len;
    let ns = read_u64(body, &mut bo, "ckpt ns").map_err(PipelineError::Tensor)? as usize;
    if ns > MAX_DICT_ENTRIES {
        return Err(PipelineError::Mismatch("checkpoint dict too large".into()));
    }
    let mut senders = Vec::with_capacity(ns.min(1 << 16));
    for _ in 0..ns {
        senders.push(read_len_str(body, &mut bo, MAX_STR, "ckpt sender").map_err(PipelineError::Tensor)?);
    }
    let nr = read_u64(body, &mut bo, "ckpt nr").map_err(PipelineError::Tensor)? as usize;
    if nr > MAX_DICT_ENTRIES {
        return Err(PipelineError::Mismatch("checkpoint dict too large".into()));
    }
    let mut receivers = Vec::with_capacity(nr.min(1 << 16));
    for _ in 0..nr {
        receivers.push(
            read_len_str(body, &mut bo, MAX_STR, "ckpt receiver").map_err(PipelineError::Tensor)?,
        );
    }
    if bo != body.len() {
        return Err(PipelineError::Mismatch("checkpoint body trailing bytes".into()));
    }
    // Validate the embedded CSF image now (canonical codec, fail-closed).
    let img =
        abacus_tensor::csf::deserialize_csf_v2(&csf_bytes).map_err(PipelineError::Tensor)?;
    if !img.csf.check_invariants() {
        return Err(PipelineError::Mismatch("checkpoint CSF invariant failure".into()));
    }
    Ok(Some(CheckpointState {
        csf_bytes,
        version,
        committed_records: committed,
        t_min,
        t_max,
        journal_len,
        senders,
        receivers,
    }))
}
/// Recovery diagnostics.
#[derive(Debug, Clone, Default)]
pub struct RecoverReport {
    /// A valid checkpoint was found and installed.
    pub checkpoint_found: bool,
    /// Journal records skipped (compacted into the checkpoint).
    pub journal_skip: usize,
    /// Total valid journal records.
    pub journal_total: usize,
    /// Records streaming_ingestion_demoed after the checkpoint.
    pub streaming_ingestion_demoed: usize,
    /// A torn tail was detected and truncated.
    pub torn_tail: bool,
    /// The journal file was rewritten without the torn tail.
    pub journal_cleaned: bool,
}

/// Recover: rebuild from artifacts + checkpoint + journal streaming_ingestion_demo.
///
/// 1. `from_loaded` over the artifacts (fail-closed cross-validation).
/// 2. If a valid checkpoint exists, install its sealed image/dicts/scalars and
///    restore the duplicate-detection keys of the compacted journal prefix.
/// 3. StreamingIngestionDemo journal records after `journal_len` through the normal `apply`
///    path (identity contracts + duplicate prevention preserved).
/// 4. A torn journal tail is truncated (and the journal file cleaned); the
///    recovered state contains only fully committed events.
pub fn recover(
    art: LoadedArtifacts,
    config: PipelineConfig,
    load_us: u128,
    checkpoint_path: &str,
    journal_path: &str,
) -> Result<(Pipeline, RecoverReport), PipelineError> {
    let mut p = Pipeline::from_loaded(art, config, load_us);
    let mut report = RecoverReport::default();
    let skip = if let Some(ck) = checkpoint_read(checkpoint_path)? {
        report.checkpoint_found = true;
        let img =
            abacus_tensor::csf::deserialize_csf_v2(&ck.csf_bytes).map_err(PipelineError::Tensor)?;
        p.install_checkpoint(CheckpointImage {
            sealed: img.csf,
            senders: ck.senders,
            receivers: ck.receivers,
            version: ck.version,
            committed_records: ck.committed_records,
            t_min: ck.t_min,
            t_max: ck.t_max,
        })?;
        ck.journal_len
    } else {
        0
    };
    let (events, torn) = journal_read(journal_path)?;
    report.journal_total = events.len();
    report.torn_tail = torn;
    if skip > events.len() {
        return Err(PipelineError::Mismatch(format!(
            "checkpoint journal_len {skip} beyond journal {}",
            events.len()
        )));
    }
    report.journal_skip = skip;
    if report.checkpoint_found {
        // Rebuild the content-keyed duplicate map for the compacted prefix so
        // pre-checkpoint redeliveries are still detected after recovery.
        p.restore_duplicate_keys(&events[..skip])?;
        // Rebuild the derived temporal index entries for the same prefix so
        // the locator directory and both directional indexes cover exactly
        // the checkpointed event set (phase-3 task spec §5).
        p.restore_index_state(&events[..skip])?;
    }
    for ev in &events[skip..] {
        p.apply(ev)?;
        report.streaming_ingestion_demoed += 1;
    }
    // Closing fail-closed alignment check: the tensor, locator directory,
    // global time path and both directional indexes must agree on the full
    // committed set before the recovered pipeline is exposed.
    p.check_index_consistency()?;
    if torn {
        journal_rewrite_clean(journal_path, &events)?;
        report.journal_cleaned = true;
    }
    Ok((p, report))
}
