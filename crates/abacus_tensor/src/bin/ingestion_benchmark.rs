//! Row 29 (§B.2) — `ingestion_benchmark`: measurement harness over rows 1–23.
//!
//! CLI surface preserved from the prior realization:
//!
//! ```text
//! ingestion_benchmark [csv_path] [flush_interval] [max_rows]
//! ```
//!
//! Row 29 is **outside the mathematical core** (P11): it measures and
//! cross-checks the ingestion path and never defines tensor semantics. The only
//! tensor objects it constructs are compositions of the admissible set
//! (`⟦⟧`, pointwise add/negate, CSF) through `Tensor::try_from_entries` and
//! `Tensor::try_add`.
//!
//! ```text
//! a_{20}  stage timings        -> ingest_timed accumulators
//! a_{16}  seen-fold structures -> dictionary codes, flagged-row fold
//! a_{09}  counters             -> stats, counts, one-pass tally
//! a_{19}  file read            -> row stream + optional artifact cross-check
//! K9      wall clock           -> never influences results
//! K10     f64 presentation     -> ratios/per-event latency in the report only
//! ```
//!
//! Oracles are derived from the scenario definitions and the independent batch
//! build alone, never from engine outputs. Startup costs (row parse, batch
//! build, sidecar reference, artifact load) are reported separately from the
//! ingestion stage. Debug builds are refused: timing a debug build measures the
//! build, not the system.

use abacus_tensor::api;
use abacus_tensor::composites::{epoch_seconds, split_csv_line};
use abacus_tensor::coo::builder::{build_from_csv, AccountDict};
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::ingest::{
    CasOutcome, Event, IngestionEngine, LedgerEntry, LedgerOp, SealReport, StageTimings,
};
use abacus_tensor::sidecar::build_sidecar_from_csv;
use abacus_tensor::tensor::value::parse_decimal_to_minor;
use abacus_tensor::tensor::{Coord, Tensor};
use abacus_tensor::TensorError;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Shipped full-dataset row count (SAML-D 90 days, header excluded).
const SAML_D_ROWS: usize = 2_624_636;
/// Event-scoped correction bump (micro-units) for the Update scenario.
const CORR_MICRO: i64 = 1_000_001;

// ---- K9: peak-memory instrumentation (measurement infrastructure) ----

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(p, l);
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = System.realloc(p, l, new_size);
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            let now = LIVE.fetch_add(new_size, Ordering::Relaxed) + new_size;
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// One parsed CSV data row (test-side mirror of Def 7.1 identity + coordinate).
struct Row {
    id: u64,
    t: u64,
    sender: String,
    receiver: String,
    amount: i64,
    laundering: bool,
}

/// Read every data row of the CSV (`a_{19}.1` + the row-14 column contract).
///
/// Column positions follow the dataset header contract used at ingest
/// (Time, Date, Sender_account, Receiver_account, Amount, …, Is_laundering at
/// index 10); the batch oracle independently rediscovers them from the header.
fn read_all_rows(csv_path: &str, parse_accum: &mut Duration) -> Result<Vec<Row>, TensorError> {
    let f = fs::File::open(csv_path).map_err(|e| TensorError::Io(e.to_string()))?;
    let mut lines = BufReader::new(f).lines();
    let _header = lines
        .next()
        .ok_or_else(|| TensorError::Parse("empty csv".into()))?
        .map_err(|e| TensorError::Io(e.to_string()))?;
    let mut rows = Vec::new();
    let c = |i: usize, f: &[String]| -> Result<String, TensorError> {
        f.get(i)
            .map(|s| s.trim().to_string())
            .ok_or_else(|| TensorError::Parse(format!("short row (column {i})")))
    };
    for (idx, line) in lines.enumerate() {
        let line = line.map_err(|e| TensorError::Io(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let t0 = Instant::now();
        let f = split_csv_line(&line);
        let id = idx as u64;
        let t = epoch_seconds(&c(0, &f)?, &c(1, &f)?)?;
        let sender = c(2, &f)?;
        let receiver = c(3, &f)?;
        let amount = parse_decimal_to_minor(&c(4, &f)?)?;
        let laundering = c(10, &f)? == "1";
        *parse_accum += t0.elapsed();
        rows.push(Row {
            id,
            t,
            sender,
            receiver,
            amount,
            laundering,
        });
    }
    Ok(rows)
}
/// Scenario predicates — pure functions of the 0-based data-row ordinal, so
/// the independent oracle follows from the definitions alone.
///
/// * `held`      rows are delivered after the watermark boundary (late arrival):
///   ordinary Creates; the watermark is never consulted (P10/K12).
/// * `corrected` rows receive one Update (+`CORR_MICRO`), pinning the exact
///   prior record (P7 CAS): `Δ = a_new − a_old` (P8).
/// * `reversed`  rows are Closed, negating the CURRENT ledger value (never a
///   cached creation amount).
/// * `streaming_ingestion_demoed`  rows receive one duplicate Create delivery → CAS retry, no Δ.
///   Every held row is also streaming_ingestion_demoed on arrival.
fn held(id: u64) -> bool {
    id % 1009 == 1008 && id < 2_200_000
}
fn corrected(id: u64) -> bool {
    !held(id) && id % 47 == 23
}
fn reversed(id: u64) -> bool {
    !held(id) && id % 101 == 37
}
fn streaming_ingestion_demoed(id: u64) -> bool {
    id.is_multiple_of(97)
}

/// Independent oracle state and counts (never read from engine outputs).
struct ScenarioPlan {
    t_min: u64,
    t_max: u64,
    /// `T = T_creates + Δ_corr + Δ_rev + T_held` at the end.
    t_expected: Tensor,
    /// Expected live tensor at each scenario-pass boundary.
    after_creates: Tensor,
    after_corrections: Tensor,
    after_reversals: Tensor,
    /// Laundering-flagged rows exactly as the stream should observe them.
    sidecar_reference: Vec<(u64, String, String, u64, i64)>,
    n_held: u64,
    n_corrected: u64,
    n_reversed: u64,
    n_streaming_ingestion_demoed: u64,
    artifact_crosscheck_ok: bool,
}

/// Derive the complete independent oracle from the scenario definitions.
#[allow(clippy::too_many_arguments)]
fn build_scenario_plan(
    rows: &[Row],
    senders: &AccountDict,
    receivers: &AccountDict,
    t_batch: &Tensor,
    t_min: u64,
    t_max: u64,
    sidecar_reference: Vec<(u64, String, String, u64, i64)>,
    artifact_crosscheck_ok: bool,
) -> Result<ScenarioPlan, TensorError> {
    let mut held_e: Vec<TxEntry> = Vec::new();
    let mut corr_e: Vec<TxEntry> = Vec::new();
    let mut rev_e: Vec<TxEntry> = Vec::new();
    let (mut n_held, mut n_cor, mut n_rev) = (0u64, 0u64, 0u64);
    for r in rows {
        let s = senders
            .code_of(&r.sender)
            .ok_or_else(|| TensorError::Parse(format!("row {} sender missing", r.id)))?;
        let rr = receivers
            .code_of(&r.receiver)
            .ok_or_else(|| TensorError::Parse(format!("row {} receiver missing", r.id)))?;
        if held(r.id) {
            held_e.push(TxEntry::new(r.t, s, rr, r.amount));
            n_held += 1;
        }
        if corrected(r.id) {
            corr_e.push(TxEntry::new(r.t, s, rr, CORR_MICRO));
            n_cor += 1;
        }
        if reversed(r.id) {
            // Close negates the CURRENT value: creation amount + correction.
            let corr = if corrected(r.id) { CORR_MICRO } else { 0 };
            rev_e.push(TxEntry::new(r.t, s, rr, -(r.amount + corr)));
            n_rev += 1;
        }
    }
    let n_streaming_ingestion_demoed = rows.iter().filter(|r| streaming_ingestion_demoed(r.id) || held(r.id)).count() as u64;
    let held_t = Tensor::try_from_entries(held_e)?;
    let after_creates = t_batch.try_add(&held_t.try_neg()?)?;
    let after_corrections = after_creates.try_add(&Tensor::try_from_entries(corr_e)?)?;
    let after_reversals = after_corrections.try_add(&Tensor::try_from_entries(rev_e)?)?;
    let t_expected = after_reversals.try_add(&held_t)?;
    Ok(ScenarioPlan {
        t_min,
        t_max,
        t_expected,
        after_creates,
        after_corrections,
        after_reversals,
        sidecar_reference,
        n_held,
        n_corrected: n_cor,
        n_reversed: n_rev,
        n_streaming_ingestion_demoed,
        artifact_crosscheck_ok,
    })
}

/// One materialization's measured cost (row 21 + row 22 report).
struct FlushMeas {
    elapsed: Duration,
    report: SealReport,
}

/// One scenario pass's summary.
struct PassMeas {
    name: &'static str,
    events: u64,
    retries: u64,
    elapsed: Duration,
}

/// One-time startup costs, reported separately from the ingestion stage.
#[derive(Debug, Clone, Copy, Default)]
struct StartupSecs {
    read_rows: Duration,
    batch_build: Duration,
    artifact_load: Duration,
    sidecar_reference: Duration,
    oracle_construction: Duration,
}

/// Aggregated result of one benchmark run.
struct RunResult {
    mode_full: bool,
    flush_interval: usize,
    raw_rows: u64,
    t_min: u64,
    t_max: u64,
    n_senders: usize,
    n_receivers: usize,
    batch_nnz: usize,
    duplicates_absorbed: u64,
    n_held: u64,
    n_corrected: u64,
    n_reversed: u64,
    n_streaming_ingestion_demoed: u64,
    startup_secs: StartupSecs,
    successful_ops: u64,
    retries: u64,
    event_latency_ns: Vec<u32>,
    stage_parse: Duration,
    stage_coord: Duration,
    stage_cas: Duration,
    stage_delta: Duration,
    stage_frontier: Duration,
    ingest_total: Duration,
    total: Duration,
    passes: Vec<PassMeas>,
    flushes: Vec<FlushMeas>,
    stats: abacus_tensor::ingest::EngineStats,
    oracle_ok: bool,
    csf_invariants_ok: bool,
    canonical_ok: bool,
    sidecar_ok: bool,
    live_invariants_ok: bool,
    artifact_crosscheck_ok: bool,
    frontier_growth: usize,
    sealed_nnz: usize,
}

impl RunResult {
    fn events_per_sec(&self) -> f64 {
        self.successful_ops as f64 / self.total.as_secs_f64().max(1e-9)
    }
    fn ingest_events_per_sec(&self) -> f64 {
        self.successful_ops as f64 / self.ingest_total.as_secs_f64().max(1e-9)
    }
    fn mean_ns(&self) -> f64 {
        if self.event_latency_ns.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.event_latency_ns.iter().map(|&v| u64::from(v)).sum();
        sum as f64 / self.event_latency_ns.len() as f64
    }
    fn median_ns(&self) -> u32 {
        percentile(&self.event_latency_ns, 50.0)
    }
    fn p95_ns(&self) -> u32 {
        percentile(&self.event_latency_ns, 95.0)
    }
    fn p99_ns(&self) -> u32 {
        percentile(&self.event_latency_ns, 99.0)
    }
    fn materialization_secs(&self) -> f64 {
        self.flushes.iter().map(|f| f.elapsed.as_secs_f64()).sum()
    }
}

/// Percentile over an unsorted sample vector (`a_{10}` on the sample vector).
fn percentile(samples: &[u32], p: f64) -> u32 {
    if samples.is_empty() {
        return 0;
    }
    let mut v = samples.to_vec();
    v.sort_unstable();
    let k = ((v.len() as f64 - 1.0) * p / 100.0).round() as usize;
    v[k.clamp(0, v.len() - 1)]
}

/// Saturating nanosecond bucket for the latency samples (`a_{06}.3`).
fn lat_to_u32(d: Duration) -> u32 {
    (d.as_nanos() as u64).min(u64::from(u32::MAX)) as u32
}
/// Bookkeeping for one delivered op: successful ops drive the flush cadence and
/// the frontier high-water mark; retries emit no delta (P7/P8).
#[allow(clippy::too_many_arguments)]
fn track_committed(
    out: CasOutcome,
    eng: &mut IngestionEngine,
    flushes: &mut Vec<FlushMeas>,
    since_flush: &mut usize,
    frontier_growth: &mut usize,
    flush_interval: usize,
    pass_events: &mut u64,
    pass_retries: &mut u64,
) -> Result<(), TensorError> {
    match out {
        CasOutcome::Committed { .. } => {
            *pass_events += 1;
            *since_flush += 1;
            *frontier_growth = (*frontier_growth).max(eng.frontier().len());
            if *since_flush >= flush_interval {
                do_flush(eng, flushes)?;
                *since_flush = 0;
            }
        }
        CasOutcome::Retry { .. } => *pass_retries += 1,
    }
    Ok(())
}

/// Row 21 materialization + row 22 report, recorded per flush.
fn do_flush(eng: &mut IngestionEngine, flushes: &mut Vec<FlushMeas>) -> Result<(), TensorError> {
    let t0 = Instant::now();
    let report = eng.seal()?;
    flushes.push(FlushMeas {
        elapsed: t0.elapsed(),
        report,
    });
    Ok(())
}

/// Live-view invariant at a pass boundary: `T_live = T_sealed + ⟦C_frontier⟧`
/// compared against the independently derived expectation (P9).
fn live_boundary_check(
    eng: &IngestionEngine,
    expected: &Tensor,
    ok: &mut bool,
) -> Result<(), TensorError> {
    if eng.decode_live()?.entries() != expected.entries() {
        *ok = false;
    }
    Ok(())
}

/// Row → `(s, r)` dictionary codes (`a_{11}` over `prec_dict`), timed into the
/// coordinate-construction stage.
fn coord_of(
    r: &Row,
    senders: &AccountDict,
    receivers: &AccountDict,
    coord_time: &mut Duration,
) -> Result<(u32, u32), TensorError> {
    let t0 = Instant::now();
    let s = senders
        .code_of(&r.sender)
        .ok_or_else(|| TensorError::Parse(format!("row {} sender missing", r.id)))?;
    let rc = receivers
        .code_of(&r.receiver)
        .ok_or_else(|| TensorError::Parse(format!("row {} receiver missing", r.id)))?;
    *coord_time += t0.elapsed();
    Ok((s, rc))
}

/// Copy the header + first `max` data lines of `src` verbatim into `dest`, so
/// every oracle operates on exactly the capped dataset.
fn write_capped_csv(src: &str, dest: &str, max: usize) -> Result<(), TensorError> {
    let f = fs::File::open(src).map_err(|e| TensorError::Io(e.to_string()))?;
    let mut lines = BufReader::new(f).lines();
    let header = lines
        .next()
        .ok_or_else(|| TensorError::Parse("empty csv".into()))?
        .map_err(|e| TensorError::Io(e.to_string()))?;
    let mut out = String::with_capacity(max.saturating_mul(32));
    out.push_str(&header);
    out.push('\n');
    for (i, line) in lines.enumerate() {
        if i >= max {
            break;
        }
        out.push_str(&line.map_err(|e| TensorError::Io(e.to_string()))?);
        out.push('\n');
    }
    fs::write(dest, out).map_err(|e| TensorError::Io(e.to_string()))
}
/// Execute the scenario suite over `rows` through the native ingestion engine,
/// materializing every `flush_interval` successful ops, then return the
/// aggregate [`RunResult`].
#[allow(clippy::too_many_arguments)]
fn run_stream(
    rows: &[Row],
    senders: &AccountDict,
    receivers: &AccountDict,
    t_batch: &Tensor,
    plan: &ScenarioPlan,
    flush_interval: usize,
    mode_full: bool,
    parse_time: Duration,
    startup: &StartupSecs,
) -> Result<RunResult, TensorError> {
    let flush_interval = flush_interval.max(1);
    let wall_start = Instant::now();
    let mut eng = IngestionEngine::new();
    let mut st = StageTimings::default();
    let mut flushes: Vec<FlushMeas> = Vec::new();
    let mut latency: Vec<u32> = Vec::with_capacity(rows.len().saturating_add(1 << 16));
    let mut stage_coord = Duration::ZERO;
    let mut since_flush = 0usize;
    let mut frontier_growth = 0usize;
    let mut seen_flagged: Vec<(u64, String, String, u64, i64)> = Vec::new();
    let mut passes: Vec<PassMeas> = Vec::new();
    let mut live_ok = true;
    let mut total_success = 0u64;
    let mut total_retries = 0u64;
    let ingest_start = Instant::now();

    // ---- Pass A: sequential Create for every non-held row, plus the duplicate
    //      delivery of the streaming_ingestion_demoed ones (CAS retry, no delta). Co-located
    //      distinct events share a coordinate but keep distinct identities, so
    //      each contributes its own Δ (P8). ----
    let t_pass = Instant::now();
    let (mut pe_a, mut pr_a) = (0u64, 0u64);
    for r in rows {
        if held(r.id) {
            continue; // withheld until after the watermark boundary (P10)
        }
        let t0 = Instant::now();
        let (s, rc) = coord_of(r, senders, receivers, &mut stage_coord)?;
        let e = Event {
            id: r.id,
            coord: Coord::new(r.t, s, rc),
        };
        let out = eng.ingest_timed(&e, LedgerOp::Create(r.amount), &mut st)?;
        latency.push(lat_to_u32(t0.elapsed()));
        track_committed(
            out,
            &mut eng,
            &mut flushes,
            &mut since_flush,
            &mut frontier_growth,
            flush_interval,
            &mut pe_a,
            &mut pr_a,
        )?;
        if r.laundering {
            seen_flagged.push((r.t, r.sender.clone(), r.receiver.clone(), r.id, r.amount));
        }
        if streaming_ingestion_demoed(r.id) {
            let t1 = Instant::now();
            let out2 = eng.ingest_timed(&e, LedgerOp::Create(r.amount), &mut st)?;
            latency.push(lat_to_u32(t1.elapsed()));
            track_committed(
                out2,
                &mut eng,
                &mut flushes,
                &mut since_flush,
                &mut frontier_growth,
                flush_interval,
                &mut pe_a,
                &mut pr_a,
            )?;
        }
    }
    passes.push(PassMeas {
        name: "A sequential create (+C co-located events, +F streaming_ingestion_demoed delivery)",
        events: pe_a,
        retries: pr_a,
        elapsed: t_pass.elapsed(),
    });
    total_success += pe_a;
    total_retries += pr_a;
    // P10/K12: the watermark is a stored sealing boundary, never a coordinate,
    // and ingest never consults it — every event below is an ordinary CAS.
    eng.set_watermark(plan.t_max);
    // P9 live-view invariant at the pass-A boundary.
    live_boundary_check(&eng, &plan.after_creates, &mut live_ok)?;

    // ---- Pass D: event-scoped corrections (P7 Update: pin the exact prior
    //      record; P8 Δ = a_new − a_old). Co-located unrelated events stay
    //      untouched. ----
    let t_pass = Instant::now();
    let (mut pe_d, mut pr_d) = (0u64, 0u64);
    for r in rows {
        if !corrected(r.id) {
            continue;
        }
        let (s, rc) = coord_of(r, senders, receivers, &mut stage_coord)?;
        let e = Event {
            id: r.id,
            coord: Coord::new(r.t, s, rc),
        };
        let t0 = Instant::now();
        let out = eng.ingest_timed(
            &e,
            LedgerOp::Update {
                expected: LedgerEntry {
                    value: r.amount,
                    version: 1,
                },
                new_value: r.amount + CORR_MICRO,
            },
            &mut st,
        )?;
        latency.push(lat_to_u32(t0.elapsed()));
        track_committed(
            out,
            &mut eng,
            &mut flushes,
            &mut since_flush,
            &mut frontier_growth,
            flush_interval,
            &mut pe_d,
            &mut pr_d,
        )?;
    }
    passes.push(PassMeas {
        name: "D event-scoped corrections",
        events: pe_d,
        retries: pr_d,
        elapsed: t_pass.elapsed(),
    });
    total_success += pe_d;
    total_retries += pr_d;

    // P9 live-view invariant at the pass-D boundary.
    live_boundary_check(&eng, &plan.after_corrections, &mut live_ok)?;

    // ---- Pass E: reversals. Close negates the CURRENT ledger value, which
    //      the CAS guard pins exactly (corrected-and-reversed rows covered). ----
    let t_pass = Instant::now();
    let (mut pe_e, mut pr_e) = (0u64, 0u64);
    for r in rows {
        if !reversed(r.id) {
            continue;
        }
        let (s, rc) = coord_of(r, senders, receivers, &mut stage_coord)?;
        let e = Event {
            id: r.id,
            coord: Coord::new(r.t, s, rc),
        };
        let corr = if corrected(r.id) { CORR_MICRO } else { 0 };
        let t0 = Instant::now();
        let out = eng.ingest_timed(
            &e,
            LedgerOp::Close {
                expected: LedgerEntry {
                    value: r.amount + corr,
                    version: if corrected(r.id) { 2 } else { 1 },
                },
            },
            &mut st,
        )?;
        latency.push(lat_to_u32(t0.elapsed()));
        track_committed(
            out,
            &mut eng,
            &mut flushes,
            &mut since_flush,
            &mut frontier_growth,
            flush_interval,
            &mut pe_e,
            &mut pr_e,
        )?;
    }
    passes.push(PassMeas {
        name: "E reversals (Close negates the current value)",
        events: pe_e,
        retries: pr_e,
        elapsed: t_pass.elapsed(),
    });
    total_success += pe_e;
    total_retries += pr_e;
    live_boundary_check(&eng, &plan.after_reversals, &mut live_ok)?;

    // ---- Pass B: late (held) arrivals, after the watermark boundary.
    //      Ordinary Creates plus one duplicate delivery each. ----
    let t_pass = Instant::now();
    let (mut pe_b, mut pr_b) = (0u64, 0u64);
    for r in rows {
        if !held(r.id) {
            continue;
        }
        let (s, rc) = coord_of(r, senders, receivers, &mut stage_coord)?;
        let e = Event {
            id: r.id,
            coord: Coord::new(r.t, s, rc),
        };
        let t0 = Instant::now();
        let out = eng.ingest_timed(&e, LedgerOp::Create(r.amount), &mut st)?;
        latency.push(lat_to_u32(t0.elapsed()));
        track_committed(
            out,
            &mut eng,
            &mut flushes,
            &mut since_flush,
            &mut frontier_growth,
            flush_interval,
            &mut pe_b,
            &mut pr_b,
        )?;
        if r.laundering {
            seen_flagged.push((r.t, r.sender.clone(), r.receiver.clone(), r.id, r.amount));
        }
        let t1 = Instant::now();
        let out2 = eng.ingest_timed(&e, LedgerOp::Create(r.amount), &mut st)?;
        latency.push(lat_to_u32(t1.elapsed()));
        track_committed(
            out2,
            &mut eng,
            &mut flushes,
            &mut since_flush,
            &mut frontier_growth,
            flush_interval,
            &mut pe_b,
            &mut pr_b,
        )?;
    }
    passes.push(PassMeas {
        name: "B late arrivals after the watermark (+F streaming_ingestion_demoed delivery)",
        events: pe_b,
        retries: pr_b,
        elapsed: t_pass.elapsed(),
    });
    total_success += pe_b;
    total_retries += pr_b;

    // Final materialization so the live view is fully sealed, then the P9
    // invariant check `T_live = T_sealed + ⟦C_frontier⟧ = T_expected`.
    do_flush(&mut eng, &mut flushes)?;
    live_ok &= eng.frontier().is_empty();
    live_boundary_check(&eng, &plan.t_expected, &mut live_ok)?;
    let ingest_total = ingest_start.elapsed();

    // ---- Independent correctness checks (rows 15 + 11; P11 isolation) ----
    let oracle_ok = eng.decode_live()?.entries() == plan.t_expected.entries();
    let csf_invariants_ok = eng.sealed().check_invariants();
    let sealed_entries = abacus_tensor::ingest::sealed_decode(eng.sealed());
    let canonical_ok = abacus_tensor::coo::encode::is_canonical(&sealed_entries);
    // Provenance cross-check: the flagged rows the stream observed equal the
    // independently rebuilt sidecar reference.
    seen_flagged.sort();
    let mut reference = plan.sidecar_reference.clone();
    reference.sort();
    let sidecar_ok = seen_flagged == reference;

    Ok(RunResult {
        mode_full,
        flush_interval,
        raw_rows: rows.len() as u64,
        t_min: plan.t_min,
        t_max: plan.t_max,
        n_senders: senders.table.len(),
        n_receivers: receivers.table.len(),
        batch_nnz: t_batch.nnz(),
        duplicates_absorbed: (rows.len() as u64).saturating_sub(t_batch.nnz() as u64),
        n_held: plan.n_held,
        n_corrected: plan.n_corrected,
        n_reversed: plan.n_reversed,
        n_streaming_ingestion_demoed: plan.n_streaming_ingestion_demoed,
        startup_secs: *startup,
        successful_ops: total_success,
        retries: total_retries,
        event_latency_ns: latency,
        stage_parse: parse_time,
        stage_coord,
        stage_cas: st.cas,
        stage_delta: st.delta,
        stage_frontier: st.frontier,
        ingest_total,
        total: wall_start.elapsed(),
        passes,
        flushes,
        stats: eng.stats(),
        oracle_ok,
        csf_invariants_ok,
        canonical_ok,
        sidecar_ok,
        live_invariants_ok: live_ok,
        artifact_crosscheck_ok: plan.artifact_crosscheck_ok,
        frontier_growth,
        sealed_nnz: eng.sealed().nnz(),
    })
}
/// Render the run as deterministic JSON (`a_{18}.2` text assembly, fixed key
/// order; no timestamps, hostnames, or other environment fields — `R_7`).
fn render_report(r: &RunResult) -> String {
    fn q(s: &str) -> String {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    }
    let mut j = String::new();
    j.push_str("{\n");
    j.push_str("  \"harness\": \"ingestion_benchmark\",\n");
    j.push_str("  \"row\": 29,\n");
    j.push_str(&format!("  \"mode_full_dataset\": {},\n", r.mode_full));
    j.push_str(&format!(
        "  \"flush_interval\": {},\n  \"raw_rows\": {},\n  \"t_min\": {},\n  \"t_max\": {},\n",
        r.flush_interval, r.raw_rows, r.t_min, r.t_max
    ));
    j.push_str(&format!(
        "  \"n_senders\": {},\n  \"n_receivers\": {},\n  \"batch_nnz\": {},\n  \
         \"duplicate_coordinates_absorbed\": {},\n",
        r.n_senders, r.n_receivers, r.batch_nnz, r.duplicates_absorbed
    ));
    j.push_str(&format!(
        "  \"scenario_counts\": {{\n    \"held_late\": {},\n    \"corrected\": {},\n    \
         \"reversed\": {},\n    \"streaming_ingestion_demoed\": {}\n  }},\n",
        r.n_held, r.n_corrected, r.n_reversed, r.n_streaming_ingestion_demoed
    ));
    j.push_str(&format!(
        "  \"successful_ops\": {},\n  \"retries\": {},\n  \"deltas_per_success\": {:.6},\n",
        r.successful_ops,
        r.retries,
        if r.successful_ops == 0 {
            0.0
        } else {
            r.stats.deltas_emitted as f64 / r.successful_ops as f64
        }
    ));
    j.push_str(&format!(
        "  \"throughput_events_per_sec_end_to_end\": {:.3},\n  \
         \"throughput_events_per_sec_ingest_stage\": {:.3},\n",
        r.events_per_sec(),
        r.ingest_events_per_sec()
    ));
    j.push_str(&format!(
        "  \"event_latency_ns\": {{\n    \"samples\": {},\n    \"mean\": {:.3},\n    \
         \"p50\": {},\n    \"p95\": {},\n    \"p99\": {}\n  }},\n",
        r.event_latency_ns.len(),
        r.mean_ns(),
        r.median_ns(),
        r.p95_ns(),
        r.p99_ns()
    ));
    j.push_str(&format!(
        "  \"ingest_stage_secs\": {:.6},\n  \"materialization_secs\": {:.6},\n  \
         \"run_total_secs\": {:.6},\n",
        r.ingest_total.as_secs_f64(),
        r.materialization_secs(),
        r.total.as_secs_f64()
    ));
    j.push_str("  \"stage_secs\": {\n");
    j.push_str(&format!(
        "    \"row_parse\": {:.6},\n    \"coordinate_construction\": {:.6},\n    \
         \"ledger_cas\": {:.6},\n    \"delta_construction\": {:.6},\n    \
         \"frontier_append\": {:.6}\n  }},\n",
        r.stage_parse.as_secs_f64(),
        r.stage_coord.as_secs_f64(),
        r.stage_cas.as_secs_f64(),
        r.stage_delta.as_secs_f64(),
        r.stage_frontier.as_secs_f64()
    ));
    j.push_str(&format!(
        "  \"startup_secs\": {{\n    \"read_parse_rows\": {:.6},\n    \"batch_build\": {:.6},\n    \
         \"artifact_load\": {:.6},\n    \"sidecar_reference\": {:.6},\n    \
         \"oracle_construction\": {:.6}\n  }},\n",
        r.startup_secs.read_rows.as_secs_f64(),
        r.startup_secs.batch_build.as_secs_f64(),
        r.startup_secs.artifact_load.as_secs_f64(),
        r.startup_secs.sidecar_reference.as_secs_f64(),
        r.startup_secs.oracle_construction.as_secs_f64()
    ));
    j.push_str("  \"scenario_passes\": [\n");
    for (i, p) in r.passes.iter().enumerate() {
        j.push_str(&format!(
            "    {{\"name\": {}, \"events\": {}, \"retries\": {}, \"secs\": {:.6}}}{}\n",
            q(p.name),
            p.events,
            p.retries,
            p.elapsed.as_secs_f64(),
            if i + 1 == r.passes.len() { "" } else { "," }
        ));
    }
    j.push_str("  ],\n");
    j.push_str(&format!(
        "  \"materializations\": {},\n  \"frontier_high_water\": {},\n  \
         \"sealed_nnz\": {},\n  \"sealed_csf_bytes\": {},\n",
        r.flushes.len(),
        r.frontier_growth,
        r.sealed_nnz,
        r.flushes.last().map_or(0, |f| f.report.csf_bytes_after)
    ));
    j.push_str(&format!(
        "  \"full_rebuild_per_materialization\": {},\n",
        r.flushes.iter().all(|f| f.report.full_rebuild)
    ));
    j.push_str("  \"correctness\": {\n");
    j.push_str(&format!(
        "    \"oracle_live_equals_expected\": {},\n    \"csf_invariants_ok\": {},\n    \
         \"canonical_sealed_ok\": {},\n    \"live_invariants_at_pass_boundaries\": {},\n    \
         \"sidecar_provenance_ok\": {},\n    \"artifact_crosscheck\": {}\n  }},\n",
        r.oracle_ok,
        r.csf_invariants_ok,
        r.canonical_ok,
        r.live_invariants_ok,
        r.sidecar_ok,
        r.artifact_crosscheck_ok
    ));
    j.push_str("  \"engine_stats\": {\n");
    j.push_str(&format!(
        "    \"creates\": {},\n    \"updates\": {},\n    \"closes\": {},\n    \
         \"retries\": {},\n    \"deltas_emitted\": {},\n    \"seals\": {}\n  }},\n",
        r.stats.creates,
        r.stats.updates,
        r.stats.closes,
        r.stats.retries,
        r.stats.deltas_emitted,
        r.stats.seals
    ));
    j.push_str(&format!(
        "  \"stats_retries_match_counted_retries\": {},\n  \"flushes_match_seals\": {},\n",
        r.stats.retries == r.retries,
        r.flushes.len() as u64 == r.stats.seals
    ));
    j.push_str(&format!("  \"peak_memory_bytes\": {},\n", peak_bytes()));
    j.push_str(&format!(
        "  \"comparison\": {}\n",
        q(
            "The only honest shared-work comparison is the independent batch build \
           (startup_secs.batch_build) against the streaming ingest stage \
           (ingest_stage_secs) for the identical dataset: both construct the same \
           canonical tensor through independent paths. Startup costs are one-time \
           load costs and are never folded into per-event latency."
        )
    ));
    j.push_str("}\n");
    j
}

/// CLI parsing: `ingestion_benchmark [csv_path] [flush_interval] [max_rows]`.
fn parse_args(args: &[String]) -> (String, usize, Option<usize>) {
    let default_csv = "raw_data_set/SAML-D-90days_1st_DONT_READ_DIRECTLY.csv".to_string();
    let default_flush: usize = 100_000;
    let bad = |m: &str| -> ! {
        eprintln!("{m}");
        eprintln!("usage: ingestion_benchmark [csv_path] [flush_interval] [max_rows]");
        eprintln!("  csv_path        SAML-D CSV (default: raw_data_set/...)");
        eprintln!("  flush_interval  seal every N successful ops (default 100000)");
        eprintln!("  max_rows        optional smoke cap: verbatim temp copy of first N data rows");
        std::process::exit(2);
    };
    match args.len() {
        1 => (default_csv, default_flush, None),
        2 => (args[1].clone(), default_flush, None),
        3 => (
            args[1].clone(),
            args[2]
                .parse()
                .unwrap_or_else(|_| bad("flush_interval must be a usize")),
            None,
        ),
        4 => (
            args[1].clone(),
            args[2]
                .parse()
                .unwrap_or_else(|_| bad("flush_interval must be a usize")),
            Some(
                args[3]
                    .parse()
                    .unwrap_or_else(|_| bad("max_rows must be a usize")),
            ),
        ),
        _ => bad("too many arguments"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if cfg!(debug_assertions) {
        eprintln!(
            "ingestion_benchmark refuses to benchmark a debug build \
             (timing a debug build measures the build, not the system); use --release"
        );
        std::process::exit(2);
    }
    let (mut csv_path, flush_interval, max_rows) = parse_args(&args);
    if let Some(n) = max_rows {
        let tmp = std::env::temp_dir().join("abacus_ingestion_benchmark_capped.csv");
        let tmp_s = tmp.to_string_lossy().into_owned();
        write_capped_csv(&csv_path, &tmp_s, n).expect("write capped csv");
        csv_path = tmp_s;
        println!("  capped smoke run: {n} data rows -> {csv_path}");
    }
    run_at(&csv_path, flush_interval);
}

/// All of [`main`] after CLI handling: returns `()` and fails the process on a
/// failed correctness check.
fn run_at(csv_path: &str, flush_interval: usize) {
    let mut startup = StartupSecs::default();

    // 1. Row stream (parse is a load-time cost, reported separately).
    let t0 = Instant::now();
    let mut parse_time = Duration::ZERO;
    let rows = read_all_rows(csv_path, &mut parse_time).expect("read/parse rows");
    startup.read_rows = t0.elapsed();

    // 2. Independent batch oracle (row 14 path).
    let t0 = Instant::now();
    let build = build_from_csv(csv_path).expect("batch build");
    startup.batch_build = t0.elapsed();
    let mode_full = rows.len() == SAML_D_ROWS;
    println!(
        "  dataset: {} rows | full-dataset cross-checks: {mode_full}",
        rows.len()
    );

    // 3. Independent sidecar provenance reference (row 24 path).
    let t0 = Instant::now();
    let ref_sidecar = build_sidecar_from_csv(csv_path).expect("sidecar reference");
    // The stream can only observe *flagged* rows, so the provenance
    // cross-check is over the flagged subset of the (complete-binary) sidecar:
    // the exact set the sidecar carried before the 2026-09-26 revision.
    let sidecar_reference: Vec<(u64, String, String, u64, i64)> = ref_sidecar
        .records
        .iter()
        .filter(|rec| rec.laundering == 1)
        .map(|rec| {
            (
                rec.timestamp,
                rec.sender_account.clone(),
                rec.receiver_account.clone(),
                rec.source_row,
                rec.amount_micro,
            )
        })
        .collect();
    startup.sidecar_reference = t0.elapsed();
    let _t0 = Instant::now(); // artifact cross-check timed into artifact_load
    let artifact_load_span = _t0;

    // 4. Artifact cross-check (full-dataset runs only; artifacts are optional).
    let mut artifact_crosscheck_ok = false;
    if mode_full {
        artifact_crosscheck_ok = true;
        if let Ok(bytes) = fs::read("artifacts/tensor.csf") {
            match api::deserialize(&bytes) {
                Ok(img) => {
                    artifact_crosscheck_ok &= api::decode(&img.csf) == build.coo
                        && img.senders == build.senders.table
                        && img.receivers == build.receivers.table
                        && img.raw_rows == build.raw_rows;
                }
                Err(_) => artifact_crosscheck_ok = false,
            }
        } else {
            artifact_crosscheck_ok = false;
        }
        if let Ok(bytes) = fs::read("artifacts/laundering.sidecar") {
            match abacus_tensor::sidecar::deserialize_sidecar(&bytes) {
                Ok(s) => artifact_crosscheck_ok &= s == ref_sidecar,
                Err(_) => artifact_crosscheck_ok = false,
            }
        } else {
            artifact_crosscheck_ok = false;
        }
    }
    startup.artifact_load = artifact_load_span.elapsed();
    // 5. Independent oracle tensor construction (rows 5 + 7 + 8).
    let t0 = Instant::now();
    let t_batch = Tensor::try_from_entries(build.coo.clone()).expect("oracle tensor");
    let mut t_min = u64::MAX;
    let mut t_max = 0u64;
    for r in &rows {
        t_min = t_min.min(r.t);
        t_max = t_max.max(r.t);
    }
    if rows.is_empty() {
        t_min = 0;
    }
    let plan = build_scenario_plan(
        &rows,
        &build.senders,
        &build.receivers,
        &t_batch,
        t_min,
        t_max,
        sidecar_reference,
        artifact_crosscheck_ok,
    )
    .expect("scenario plan");
    startup.oracle_construction = t0.elapsed();

    let result = run_stream(
        &rows,
        &build.senders,
        &build.receivers,
        &t_batch,
        &plan,
        flush_interval,
        mode_full,
        parse_time,
        &startup,
    )
    .expect("run stream");

    println!(
        "  ops={} retries={} seals={} oracle_ok={} live_ok={} sidecar_ok={} csf_ok={}",
        result.successful_ops,
        result.retries,
        result.flushes.len(),
        result.oracle_ok,
        result.live_invariants_ok,
        result.sidecar_ok,
        result.csf_invariants_ok
    );
    println!(
        "  ingest_stage={:.3}s materialization={:.3}s end_to_end={:.3}s",
        result.ingest_total.as_secs_f64(),
        result.materialization_secs(),
        result.total.as_secs_f64()
    );

    // Report output (row 28 `a_{19}` write of a row-29 rendering).
    if let Err(e) = fs::create_dir_all("artifacts") {
        eprintln!("warning: could not create artifacts/: {e}");
    }
    let out_path = "artifacts/ingestion_benchmark_report.json";
    if let Err(e) = fs::write(out_path, render_report(&result)) {
        eprintln!("warning: could not write {out_path}: {e}");
    } else {
        println!("  report: {out_path}");
    }
    if !(result.oracle_ok
        && result.live_invariants_ok
        && result.csf_invariants_ok
        && result.canonical_ok
        && result.sidecar_ok)
    {
        eprintln!("ingestion_benchmark: correctness check FAILED");
        std::process::exit(1);
    }
}
