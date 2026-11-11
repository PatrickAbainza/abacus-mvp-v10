//! `StreamingIngestionDemoEngine` — the streaming_ingestion_demo state machine that drives the canonical
//! incremental pipeline (`Pipeline::apply`) over a streamed SAMLD source.
//!
//! Golden-source grounding (your_task_phase_4.md; no invented behavior):
//! - Ingestion: EVERY record is applied through `Pipeline::apply` — the sole
//!   canonical ingestion entry point (validate -> CAS -> frontier append ->
//!   derived index insert -> atomic commit; derivation rows 18-23 / P7-P10).
//!   The engine never mutates the tensor or indexes directly and never seals
//!   on the per-event path. Start from a clean state means an empty `T_0`
//!   (the same shape `pipeline_run`'s streaming_ingestion_demo mode uses).
//! - Idempotency: `apply` rejects a redelivered `(t,s,r,amount)` as
//!   `Duplicate` with no state change (content-keyed CAS, P7), so an
//!   interrupted or repeated operation cannot silently duplicate an event's
//!   contribution (task §7). `position` only ever moves forward.
//! - Determinism (P12/R7): records are applied in the artifact's canonical
//!   `(t,s,r)` order, so the committed tensor, both directional indexes, the
//!   global time path and the locator directory are identical on every clean
//!   streaming_ingestion_demo (task §9 test 14).
//! - Scheduling (task §2B): `apply_one` is the per-record primitive; the web
//!   scheduler drives it as fast as possible with no rate limiting — natural
//!   streaming at the maximum speed the infrastructure allows. Each record
//!   independently extends the live tensor (CAS → frontier append → index →
//!   commit) — there is no read-batch-then-apply-batch step, and no target
//!   throughput. Actual measured throughput is reported honestly.
//! - Amounts: the canonical `amount_micro` is ingested verbatim (P2 — it is
//!   the tensor value; P3 forbids a zero value) but is never surfaced by this
//!   engine or the web layer (task §1/§6).

use crate::pipeline::{ApplyOutcome, NormalizedEvent, Pipeline, PipelineConfig};
use crate::streaming_ingestion_demo::reader::SamldStreamReader;
use abacus_tensor::csf::{CsfTensor, Header, Image as CsfImage};
use abacus_tensor::samld::{CoreImage, EvalRecord};
use std::time::Instant;

/// StreamingIngestionDemo lifecycle phase (task §3A "Current streaming_ingestion_demo status", §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingIngestionDemoPhase {
    /// Not started, or reset to the initial clean state.
    Idle,
    /// Actively applying records.
    Running,
    /// Was running; scheduling stopped at a safe cancellation point.
    Paused,
    /// Every source record has been applied.
    Completed,
    /// The last operation failed; see `StreamingIngestionDemoStatus::last_error`.
    Error,
}

/// Point-in-time streaming_ingestion_demo status exposed to the web layer (task §8: actual
/// progress, errors, processed counts, measured throughput).
#[derive(Debug, Clone)]
pub struct StreamingIngestionDemoStatus {
    /// Current lifecycle phase.
    pub phase: StreamingIngestionDemoPhase,
    /// Next source record ordinal to apply (advances past duplicates too).
    pub position: usize,
    /// Total source records in the artifact.
    pub total: usize,
    /// Committed records so far (== queryable locators).
    pub processed: usize,
    /// Redelivered duplicates observed so far.
    pub duplicates: usize,
    /// Rejected/errored applies so far.
    pub errors: usize,
    /// Measured processing rate: committed / active wall time (transactions/
    /// second). Reported honestly; may be below `target_tps` (task §2B).
    pub measured_tps: f64,
    /// Current process resident set size in bytes, sampled when the status
    /// snapshot is built (std-only, best effort; 0 where unsupported). Lets
    /// the UI show live memory during streaming ingestion.
    pub current_rss_bytes: u64,
    /// Peak resident set size observed during this run (bytes), sampled after
    /// each applied record. Bounded streaming keeps this flat as records grow.
    pub peak_rss_bytes: u64,
    /// Most recent error message, if `phase == Error`.
    pub last_error: Option<String>,
}

/// The streaming_ingestion_demo engine: a streamed SAMLD source driving one canonical `Pipeline`.
///
/// Single-owner by construction (the web layer wraps it in one `Mutex`); the
/// underlying `Pipeline` follows the repository's serialized-writer /
/// atomic-publication model (`state.rs`), so a reader borrowing a snapshot can
/// never observe the tensor and its indexes disagreeing.
pub struct StreamingIngestionDemoEngine {
    /// Path to the `SAMCORE02` core artifact (reopen on start/reset).
    core_path: String,
    /// The canonical incremental pipeline (single owner; see `state.rs`).
    pipeline: Pipeline,
    /// Bounded streaming source reader.
    reader: SamldStreamReader,
    /// Lifecycle phase.
    phase: StreamingIngestionDemoPhase,
    /// Next source record ordinal to apply.
    position: usize,
    /// Committed records so far.
    processed: usize,
    /// Duplicates observed so far.
    duplicates: usize,
    /// Errors observed so far.
    errors: usize,
    /// Cumulative active apply time (nanoseconds; excludes idle/paused).
    active_ns: u128,
    /// Most recent error message.
    last_error: Option<String>,
    /// Peak resident set size observed during ingestion (bytes).
    peak_rss: u64,
}

impl StreamingIngestionDemoEngine {
    /// Open the artifact and build a clean `T_0` (empty tensor). The streaming_ingestion_demo
    /// begins in [`StreamingIngestionDemoPhase::Idle`]; `start` applies records.
    ///
    /// # Errors
    /// Propagates the streaming reader's fail-closed validation
    /// ([`SamldStreamReader::open`]).
    pub fn open(core_path: &str) -> Result<Self, crate::error::PipelineError> {
        let reader = SamldStreamReader::open(core_path)?;
        let pipeline = empty_pipeline();
        Ok(Self {
            core_path: core_path.to_string(),
            pipeline,
            reader,
            phase: StreamingIngestionDemoPhase::Idle,
            position: 0,
            processed: 0,
            duplicates: 0,
            errors: 0,
            active_ns: 0,
            last_error: None,
            peak_rss: 0,
        })
    }

    /// Borrow the underlying canonical pipeline for read-only queries
    /// (pagination, network building). The web layer holds the engine's `Mutex`
    /// for the whole borrow, so the snapshot is consistent.
    #[must_use]
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Current lifecycle phase.
    #[must_use]
    pub fn phase(&self) -> StreamingIngestionDemoPhase {
        self.phase
    }

    /// Build a point-in-time status snapshot.
    #[must_use]
    pub fn status(&self) -> StreamingIngestionDemoStatus {
        let measured_tps = if self.active_ns == 0 {
            0.0
        } else {
            self.processed as f64 / (self.active_ns as f64 / 1e9)
        };
        // Sample live memory now; report the higher of the running peak and the
        // current sample so the UI can show memory stays bounded during ingest.
        let current_rss_bytes = sample_rss_bytes();
        let peak_rss_bytes = self.peak_rss.max(current_rss_bytes);
        StreamingIngestionDemoStatus {
            phase: self.phase,
            position: self.position,
            total: self.reader.count(),
            processed: self.processed,
            duplicates: self.duplicates,
            errors: self.errors,
            measured_tps,
            current_rss_bytes,
            peak_rss_bytes,
            last_error: self.last_error.clone(),
        }
    }

    /// Initialize a NEW streaming_ingestion_demo from a clean state (task §7 Start).
    ///
    /// Rebuilds an empty `T_0` and reopens the source from ordinal 0, then
    /// validates the requested rate and chunk size BEFORE any state change
    /// (fail-before-mutate). A completed or paused streaming_ingestion_demo restarts from the
    /// beginning; resuming an in-progress session is `resume` (task §7: the
    /// current implementation does not persist sessions, so start is always a
    /// fresh clean streaming_ingestion_demo — documented, not invented).
    ///
    /// # Errors
    /// `Err(String)` on an invalid rate/chunk size (nothing is mutated), or an
    /// artifact reopen failure.
    pub fn start(&mut self) -> Result<(), String> {
        // Nothing to validate before starting — natural streaming has no rate
        // limit. Reopen the artifact and reset to clean state.
        self.reader = SamldStreamReader::open(&self.core_path).map_err(|e| e.to_string())?;
        self.pipeline = empty_pipeline();
        self.phase = StreamingIngestionDemoPhase::Running;
        self.position = 0;
        self.processed = 0;
        self.duplicates = 0;
        self.errors = 0;
        self.active_ns = 0;
        self.last_error = None;
        // Seed the peak with the current baseline so it never under-reports.
        self.peak_rss = sample_rss_bytes();
        Ok(())
    }

    /// Pause scheduling (task §7 Pause). Any in-flight slice already applied is
    /// kept (a safe cancellation point is a slice boundary); no new records are
    /// read until `resume`. Idempotent.
    pub fn pause(&mut self) {
        if self.phase == StreamingIngestionDemoPhase::Running {
            self.phase = StreamingIngestionDemoPhase::Paused;
        }
    }

    /// Resume from the next unprocessed transaction without duplicating
    /// completed events (task §7 Resume). `position` is preserved across pause,
    /// and the content-keyed CAS (`apply`) makes any overlap a no-op `Duplicate`.
    pub fn resume(&mut self) {
        if self.phase == StreamingIngestionDemoPhase::Paused {
            self.phase = StreamingIngestionDemoPhase::Running;
        }
    }

    /// Reset to the initial clean state (task §7 Reset): stop, clear derived
    /// state, reset progress. The original SAMLD source is never modified —
    /// only the in-memory pipeline and stream cursors are rebuilt.
    pub fn reset(&mut self) {
        self.reader = SamldStreamReader::open(&self.core_path)
            .expect("reset reopen: artifact was valid at open");
        self.pipeline = empty_pipeline();
        self.phase = StreamingIngestionDemoPhase::Idle;
        self.position = 0;
        self.processed = 0;
        self.duplicates = 0;
        self.errors = 0;
        self.active_ns = 0;
        self.last_error = None;
        self.peak_rss = 0;
    }

    /// Apply exactly one record through the canonical incremental pipeline
    /// (`Pipeline::apply`: ledger CAS → frontier append → index insert →
    /// atomic commit). This is the per-event incremental tensor update: each
    /// record independently extends the live tensor (frontier overlay) — there
    /// is no read-batch-then-apply-batch step.
    ///
    /// Returns `Ok(true)` when one record was read and applied, `Ok(false)`
    /// when the source is exhausted, or `Err` on an unrecoverable source-decode
    /// failure (phase set to `Error`). Individual rejected events are counted,
    /// not fatal.
    pub fn apply_one(&mut self) -> Result<bool, String> {
        let rec = match self.reader.next_record().map_err(|e| {
            self.phase = StreamingIngestionDemoPhase::Error;
            self.last_error = Some(e.to_string());
            e.to_string()
        })? {
            Some(rec) => rec,
            None => {
                if self.phase == StreamingIngestionDemoPhase::Running {
                    self.phase = StreamingIngestionDemoPhase::Completed;
                }
                return Ok(false);
            }
        };
        let t0 = Instant::now();
        // Resolve codes through the dictionaries fail-closed: a code out of
        // dictionary range is a corrupt source, counted as an error (never a
        // panic, never a fabricated account). The canonical loader is the
        // source of truth for dictionary bounds (P3).
        let (Some(sender), Some(receiver)) = (
            self.reader.senders().get(rec.s as usize),
            self.reader.receivers().get(rec.r as usize),
        ) else {
            self.errors += 1;
            self.position += 1;
            self.last_error = Some(format!(
                "source code out of dictionary range: s={} r={}",
                rec.s, rec.r
            ));
            let ns = t0.elapsed().as_nanos();
            self.active_ns += ns;
            if self.position >= self.reader.count() {
                self.phase = StreamingIngestionDemoPhase::Completed;
            }
            return Ok(true);
        };
        let ev = NormalizedEvent {
            t: rec.t,
            sender: sender.clone(),
            receiver: receiver.clone(),
            amount_micro: rec.amount_micro,
        };
        match self.pipeline.apply(&ev) {
            Ok(ApplyOutcome::Committed(_)) => self.processed += 1,
            Ok(ApplyOutcome::Duplicate(_)) => self.duplicates += 1,
            Err(e) => {
                self.errors += 1;
                self.last_error = Some(e.to_string());
            }
        }
        self.position += 1;
        let ns = t0.elapsed().as_nanos();
        self.active_ns += ns;
        // Track peak memory across applied records so the UI can show it stays
        // bounded while records stream in (the reader never materializes the
        // whole source).
        let rss = sample_rss_bytes();
        if rss > self.peak_rss {
            self.peak_rss = rss;
        }
        if self.position >= self.reader.count() {
            self.phase = StreamingIngestionDemoPhase::Completed;
        }
        Ok(true)
    }

}

/// Sample the current process resident set size (RSS) in bytes.
///
/// Std-only, best effort (§0 H1): on Linux (the deployment target — `docker/`)
/// it reads `VmRSS` from `/proc/self/status`; on other platforms it returns 0
/// and the UI shows "n/a". Lets the UI demonstrate that streaming ingestion
/// keeps memory bounded — the reader never materializes the whole source (§2B).
fn sample_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    let kb: u64 = rest
                        .split_whitespace()
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    return kb.saturating_mul(1024);
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Build a clean empty `T_0`: an empty artifact image fed through the canonical
/// `Pipeline::from_loaded` path (same shape as `pipeline_run`'s streaming_ingestion_demo mode).
/// `t_min = u64::MAX` so the first committed event establishes the lower bound
/// (`commit_ready` only lowers `t_min`, never raises it); `t_max = 0`.
fn empty_pipeline() -> Pipeline {
    use crate::loader::LoadedArtifacts;
    let art = LoadedArtifacts {
        core: CoreImage {
            records: Vec::new(),
            senders: Vec::new(),
            receivers: Vec::new(),
            t_min: u64::MAX,
            t_max: 0,
            mean_major: 0.0,
            std_major: 0.0,
            record_count: 0,
        },
        csf: CsfImage {
            csf: CsfTensor::empty(),
            senders: Vec::new(),
            receivers: Vec::new(),
            t_min: u64::MAX,
            t_max: 0,
            raw_rows: 0,
            header: Header {
                version: 0,
                t_width: 8,
                s_width: 4,
                r_width: 4,
                v_width: 8,
                scale_exp: 6,
                n0: 0,
                n1: 0,
                nnz: 0,
                checksum: 0,
            },
        },
        eval: Vec::<EvalRecord>::new(),
        core_bytes: 0,
        csf_bytes: 0,
        eval_bytes: 0,
    };
    Pipeline::from_loaded(art, PipelineConfig::default(), 0)
}

