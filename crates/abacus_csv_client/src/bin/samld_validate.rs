//! `samld_validate` — Phase 7 full-dataset SAML-D ingestion validation driver.
//!
//! A thin, headless harness in the same tradition as `csv_ingest`: it runs the
//! **single existing ingestion path** (CSV normalization → `ABACSEG1` chunks →
//! upload → seal → attested commit → reconciled report) against the canonical
//! SAML-D 90-day dataset and checks every observable against the existing
//! `samld_90d_3rd` baseline.
//!
//! ## Source-layout adaptation (explicit, value-preserving)
//!
//! The SAML-D export carries its timestamp in TWO columns (`Time`, `Date`);
//! the canonical record (Contract §5/§7) has exactly ONE timestamp field. The
//! harness therefore performs one explicit, value-preserving merge in memory —
//! `Date + "T" + Time + "Z"` — before normalization. No value is dropped,
//! reordered or rounded, the raw file on disk is never modified, and the
//! ground-truth label columns are simply never mapped (labels stay out of
//! feature derivation by construction).
//!
//! ## Why a patient transport
//!
//! `TcpTransport`'s hard-coded 10 s read timeout cannot survive a durable
//! (Amendment A2) commit of millions of records: the server fsyncs one journal
//! frame per applied record before acknowledging. The public [`Transport`]
//! seam exists exactly for platform-specific transports, so this harness
//! supplies its own patient native transport through
//! `CsvIngestClient::with_transport`. The ingestion path itself (driver →
//! client → `/api/v1` → service → substrate CAS) is untouched.
//!
//! ## Modes
//!
//! ```text
//! samld_validate run <csv-path> <host:port>        full validation run
//! samld_validate report <host:port> <session-id>   fetch one session report
//! samld_validate recommit <host:port> <session-id> re-POST commit (expect 409)
//! samld_validate journal <journal-path>            verify ABACJRN1 frames
//! ```

use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{BufReader, Read as _, Write as _};
use std::time::{Duration, Instant};

use abacus_csv_client::driver::run_import;
use abacus_csv_client::mapping::{CanonicalField, ColumnMapping};
use abacus_csv_client::policy::{
    AmountFormat, EpochUnit, IngestPlan, NumberPolicy, SubSecondPolicy, TimestampPolicy,
    TimestampStyle,
};
use abacus_csv_client::{
    normalize, ApiResponse, CsvIngestClient, CsvOptions, EventNamespace, IdentifierPolicy,
    Transport,
};
use abacus_ingest::api::json::parse as parse_json;
use abacus_ingest::hash::sha256;
use abacus_ingest::wire::Segment;

// ---------------------------------------------------------------------------
// Baseline expectations.
//
// Provenance: `abacus_mvp_v10/artifacts/profiles/samld_90d_3rd/samld_stats.json`
// and `samld_csf_stats.json`, the pre-existing artifacts built from
// `SAML-D-90days_3rd.csv` by the SAMCORE02 v3 fixture path. Nothing here is
// invented: every constant is either copied from those artifacts or derived
// from the dataset itself by an independent scan (documented inline).
// ---------------------------------------------------------------------------

/// Baseline `num_rows` / `num_transactions`.
const BASELINE_NUM_ROWS: u64 = 2_681_662;
/// Baseline `num_unique_coordinates`.
const BASELINE_NUM_UNIQUE_COORDINATES: u64 = 2_681_567;
/// Baseline `num_duplicates_absorbed` (rows − unique coordinates).
const BASELINE_NUM_DUPLICATES_ABSORBED: u64 = 95;
/// Baseline `num_senders`.
const BASELINE_NUM_SENDERS: u64 = 136_326;
/// Baseline `num_receivers`.
const BASELINE_NUM_RECEIVERS: u64 = 409_291;
/// Baseline `t_min` (epoch seconds, UTC).
const BASELINE_T_MIN_S: i64 = 1_680_652_807;
/// Baseline `t_max` (epoch seconds, UTC).
const BASELINE_T_MAX_S: i64 = 1_688_428_798;

/// Rows whose `Amount` field has more than 2 fractional digits (measured: the
/// dataset's fractional-digit counts are exactly 1, 2, 10, 11, 12 and 13).
/// The pipeline rejects excess precision and never rounds (§8/§19); the
/// baseline rounded half-up at the 7th digit, so this is the one documented
/// divergence between the two paths.
const EXPECTED_REJECTED_ROWS: u64 = 275;
/// `BASELINE_NUM_ROWS − EXPECTED_REJECTED_ROWS`.
const EXPECTED_ACCEPTED_ROWS: u64 = 2_681_387;
/// Unique `(t, sender, receiver)` coordinates among the ACCEPTED rows
/// (measured; every rejected row's coordinate is unique, so this is
/// `BASELINE_NUM_UNIQUE_COORDINATES − EXPECTED_REJECTED_ROWS`).
const EXPECTED_UNIQUE_COORDS_ACCEPTED: u64 = 2_681_292;
/// Distinct senders among accepted rows (measured: 22 baseline senders appear
/// only in rejected rows).
const EXPECTED_SENDERS_ACCEPTED: u64 = 136_304;
/// Distinct receivers among accepted rows (measured: 16 baseline receivers
/// appear only in rejected rows).
const EXPECTED_RECEIVERS_ACCEPTED: u64 = 409_275;
/// Distinct accounts (senders ∪ receivers) among accepted rows (measured; the
/// baseline's two set counts are reproduced exactly by this scan).
const EXPECTED_UNION_ACCOUNTS: u64 = 500_972;

/// Records per uploaded chunk (server default limit: 100,000).
const RECORDS_PER_CHUNK: usize = 100_000;
/// Exact `ABACSEG1` header length (wire §1).
const HEADER_LEN: u64 = 176;
/// Exact canonical record length for width 4: `23 + 2W` (Contract §7.2).
const RECORD_LEN: u64 = 31;
/// Hard bound on one journal frame payload (mirrors `persist::MAX_PAYLOAD`).
const JOURNAL_MAX_PAYLOAD: u32 = 1 << 20;

/// The fixed event namespace for this validation run (§3.2: a retry reuses
/// the namespace; fixing it here makes the encoded chunks reproducible).
const SAMLD_NAMESPACE: EventNamespace = EventNamespace::from_bytes(*b"SAMLD90D3RD\0\0\0\0\0");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("run") => cmd_run(&args[2..]),
        Some("report") => cmd_report(&args[2..]),
        Some("recommit") => cmd_recommit(&args[2..]),
        Some("journal") => cmd_journal(&args[2..]),
        _ => {
            eprintln!(
                "usage: samld_validate run <csv-path> <host:port> | \
                 report <host:port> <session-id> | \
                 recommit <host:port> <session-id> | \
                 journal <journal-path>"
            );
            2
        }
    };
    std::process::exit(code);
}

// ---------------------------------------------------------------------------
// The explicit SAML-D ingest plan.
//
// Every field is declared; nothing is guessed (§9 Stage 1). The numeric
// configuration mirrors the existing demo plan where the dataset does not
// force a change; the dataset forces only: GBP/precision 2 (UK pounds is the
// dataset's majority payment currency; amounts carry at most 2 fractional
// digits except the 275 never-round rejections), no group separator,
// ISO-8601 timestamps with an offset in every value, and the mapping of the
// merged timestamp column.
// ---------------------------------------------------------------------------

fn samld_plan() -> IngestPlan {
    let mapping = ColumnMapping::new([
        (CanonicalField::Timestamp, 0), // merged `DateTTimeZ` column
        (CanonicalField::Sender, 2),    // Sender_account
        (CanonicalField::Receiver, 3),  // Receiver_account
        (CanonicalField::Amount, 4),    // Amount
    ])
    .expect("the SAML-D column mapping is complete and non-overlapping");
    IngestPlan {
        mapping,
        csv: CsvOptions::default(),
        timestamp: TimestampPolicy {
            style: TimestampStyle::Iso8601WithOffset,
            source_offset_minutes: None,
        },
        amount: AmountFormat {
            decimal_separator: b'.',
            group_separator: None,
            accept_negative: false,
        },
        numbers: NumberPolicy {
            epoch_unit: EpochUnit::Millis,
            subsecond: SubSecondPolicy::RejectSubMilli,
        },
        identifiers: IdentifierPolicy {
            trim: true,
            case_sensitive: true,
            account_id_width: 4,
        },
        currency_code: "GBP".to_string(),
        currency_precision: 2,
        threshold_minor_units: 10_000,
        threshold_effective_day: 19_000,
        threshold_config_version: 3,
        // 1e11 minor units = 1e9 major: above the largest accepted amount
        // (12,618,498.40 → 1,261,849,840 minor units, measured).
        max_amount_minor_units: 100_000_000_000,
        parsing_policy_version: 1,
        records_per_chunk: RECORDS_PER_CHUNK,
        namespace: Some(SAMLD_NAMESPACE),
    }
}

// ---------------------------------------------------------------------------
// The timestamp-layout adapter: one explicit, value-preserving merge.
// ---------------------------------------------------------------------------

/// Merge the SAML-D `Time`+`Date` columns into one ISO-8601 UTC column.
///
/// Rewrites each data row's first field as `Date + "T" + Time + "Z"` and keeps
/// every other byte of the row verbatim; the header's first field becomes
/// `timestamp_utc`. This is the trusted normalization boundary that adapts the
/// source's two-column timestamp layout to the canonical single timestamp
/// field (§5/§7) — exactly the shape the baseline's `epoch_seconds(Time, Date)`
/// produces, so the event times are identical to the fixture path's.
///
/// The merge fails closed: a row without two leading commas, a `Time` field
/// that is not `HH:MM:SS`, or a `Date` field that is not `YYYY-MM-DD` aborts
/// with the source line number. Nothing is silently skipped.
///
/// Returns the merged bytes and the data-row count.
///
/// # Errors
/// A structural violation, naming the 1-based source line.
fn merge_samld_timestamps(raw: &[u8]) -> Result<(Vec<u8>, u64), String> {
    let mut out = Vec::with_capacity(raw.len() + 4_194_304);
    let mut rows = 0u64;
    let mut header = true;
    let mut start = 0usize;
    let mut line_no = 0usize;
    let mut emit = |line: &[u8], out: &mut Vec<u8>| -> Result<(), String> {
        line_no += 1;
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if header {
            let c1 = line
                .iter()
                .position(|&b| b == b',')
                .ok_or_else(|| format!("line {line_no}: header has no comma"))?;
            out.extend_from_slice(b"timestamp_utc,");
            out.extend_from_slice(&line[c1 + 1..]);
            out.push(b'\n');
            header = false;
            return Ok(());
        }
        let c1 = line
            .iter()
            .position(|&b| b == b',')
            .ok_or_else(|| format!("line {line_no}: row has no comma"))?;
        let c2 = line[c1 + 1..]
            .iter()
            .position(|&b| b == b',')
            .map(|i| i + c1 + 1)
            .ok_or_else(|| format!("line {line_no}: row has fewer than two commas"))?;
        let time = &line[..c1];
        let date = &line[c1 + 1..c2];
        if time.len() != 8 || time[2] != b':' || time[5] != b':' {
            return Err(format!("line {line_no}: Time field is not HH:MM:SS"));
        }
        if date.len() != 10 || date[4] != b'-' || date[7] != b'-' {
            return Err(format!("line {line_no}: Date field is not YYYY-MM-DD"));
        }
        if !time.iter().all(|b| b.is_ascii_digit() || *b == b':')
            || !date.iter().all(|b| b.is_ascii_digit() || *b == b'-')
        {
            return Err(format!("line {line_no}: Time/Date field has a non-digit"));
        }
        out.extend_from_slice(date);
        out.push(b'T');
        out.extend_from_slice(time);
        out.extend_from_slice(b"Z,");
        // Everything from the original second field on stays verbatim, so the
        // source column positions of Sender/Receiver/Amount are unchanged.
        out.extend_from_slice(&line[c1 + 1..]);
        out.push(b'\n');
        rows += 1;
        Ok(())
    };
    for (i, &b) in raw.iter().enumerate() {
        if b == b'\n' {
            emit(&raw[start..i], &mut out)?;
            start = i + 1;
        }
    }
    if start < raw.len() {
        emit(&raw[start..], &mut out)?;
    }
    Ok((out, rows))
}

// ---------------------------------------------------------------------------
// A patient native transport for the public `Transport` seam.
// ---------------------------------------------------------------------------

/// A `Connection: close` HTTP/1.1 transport identical in wire behaviour to
/// `TcpTransport`, but with a configurable timeout. A durable commit of
/// millions of records takes minutes of per-record fsyncs before the server
/// answers, which the browser-oriented 10 s default cannot span.
struct PatientTransport {
    addr: String,
    timeout: Duration,
}

impl Transport for PatientTransport {
    fn exchange(
        &self,
        method: &str,
        path: &str,
        body: &[u8],
        content_type: Option<&str>,
    ) -> Result<ApiResponse, String> {
        use std::net::TcpStream;
        let mut stream = TcpStream::connect(&self.addr).map_err(|e| format!("connect: {e}"))?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|e| format!("set_read_timeout: {e}"))?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|e| format!("set_write_timeout: {e}"))?;
        let mut head = String::new();
        head.push_str(&format!("{method} {path} HTTP/1.1\r\n"));
        head.push_str(&format!("Host: {}\r\n", self.addr));
        if let Some(ct) = content_type {
            head.push_str(&format!("Content-Type: {ct}\r\n"));
        }
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
        head.push_str("Connection: close\r\n\r\n");
        stream
            .write_all(head.as_bytes())
            .map_err(|e| format!("write head: {e}"))?;
        if !body.is_empty() {
            stream
                .write_all(body)
                .map_err(|e| format!("write body: {e}"))?;
        }
        stream.flush().map_err(|e| format!("flush: {e}"))?;
        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .map_err(|e| format!("read: {e}"))?;
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|i| i + 4)
            .ok_or_else(|| "the response has no header terminator".to_string())?;
        let head_text = String::from_utf8_lossy(&raw[..split]);
        let status: u16 = head_text
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| "the response has no status line".to_string())?;
        let body_bytes = raw.split_off(split);
        let body_text = String::from_utf8_lossy(&body_bytes);
        let json = if body_text.trim().is_empty() {
            None
        } else {
            parse_json(&body_text).ok()
        };
        Ok(ApiResponse {
            status,
            json,
            raw: body_bytes,
        })
    }
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

/// Lowercase-hex rendering of a digest.
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// One recorded validation check.
struct Check {
    name: String,
    measured: String,
    expected: String,
    ok: bool,
}

/// Accumulates checks, printing each one as it is recorded.
struct Checker {
    checks: Vec<Check>,
}

impl Checker {
    fn new() -> Self {
        Self { checks: Vec::new() }
    }

    fn expect_u64(&mut self, name: &str, measured: u64, expected: u64) {
        self.push(
            name,
            measured.to_string(),
            expected.to_string(),
            measured == expected,
        );
    }

    fn expect_i64(&mut self, name: &str, measured: i64, expected: i64) {
        self.push(
            name,
            measured.to_string(),
            expected.to_string(),
            measured == expected,
        );
    }

    fn expect_bool(&mut self, name: &str, measured: bool, expected: bool) {
        self.push(
            name,
            measured.to_string(),
            expected.to_string(),
            measured == expected,
        );
    }

    fn expect_str(&mut self, name: &str, measured: &str, expected: &str) {
        self.push(
            name,
            measured.to_string(),
            expected.to_string(),
            measured == expected,
        );
    }

    fn push(&mut self, name: &str, measured: String, expected: String, ok: bool) {
        println!(
            "CHECK {name}: {} (measured={measured}, expected={expected})",
            if ok { "PASS" } else { "FAIL" }
        );
        self.checks.push(Check {
            name: name.to_string(),
            measured,
            expected,
            ok,
        });
    }

    fn failures(&self) -> usize {
        self.checks.iter().filter(|c| !c.ok).count()
    }
}

// ---------------------------------------------------------------------------
// Mode: the full validation run.
// ---------------------------------------------------------------------------

fn cmd_run(args: &[String]) -> i32 {
    let (path, addr) = match (args.first(), args.get(1)) {
        (Some(p), Some(a)) => (p.clone(), a.clone()),
        _ => {
            eprintln!("usage: samld_validate run <csv-path> <host:port>");
            return 2;
        }
    };

    // 1. Read the raw dataset (input bytes only; the file is never modified).
    let t = Instant::now();
    let raw = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return 1;
        }
    };
    let raw_bytes = raw.len() as u64;
    eprintln!(
        "[1/7] read {raw_bytes} bytes in {:.3}s",
        t.elapsed().as_secs_f64()
    );

    // 2. The explicit timestamp-layout merge (value-preserving; see module docs).
    let t = Instant::now();
    let (merged, merge_rows) = match merge_samld_timestamps(&raw) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("merge failed: {e}");
            return 1;
        }
    };
    drop(raw);
    let merged_bytes = merged.len() as u64;
    eprintln!(
        "[2/7] merged timestamps: {merge_rows} rows, {merged_bytes} bytes in {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let plan = samld_plan();
    let mut checker = Checker::new();

    // 3. Normalization pass 1 (CSV parse, validation, feature derivation,
    //    interning, encoding) — the exact code the upload path uses.
    let t = Instant::now();
    let norm1 = match normalize(&merged, &plan, SAMLD_NAMESPACE) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("normalization failed: {e}");
            return 1;
        }
    };
    let normalize1_s = t.elapsed().as_secs_f64();
    eprintln!(
        "[3/7] normalize pass 1: {} accepted / {} rejected in {normalize1_s:.3}s",
        norm1.preview.accepted_rows, norm1.preview.rejected_rows
    );
    for issue in norm1.preview.issues.iter().take(3) {
        eprintln!(
            "[diag] line={} code={} detail={}",
            issue.line,
            issue.code.as_str(),
            issue.detail
        );
    }
    let digests1: Vec<[u8; 32]> = norm1.chunks.iter().map(|c| sha256(c)).collect();

    // 4. Normalization pass 2 — determinism of interning, features and encoding.
    let t = Instant::now();
    let norm2 = match normalize(&merged, &plan, SAMLD_NAMESPACE) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("normalization pass 2 failed: {e}");
            return 1;
        }
    };
    let normalize2_s = t.elapsed().as_secs_f64();
    let digests2: Vec<[u8; 32]> = norm2.chunks.iter().map(|c| sha256(c)).collect();
    checker.expect_bool(
        "normalization_is_deterministic (preview equality)",
        norm1.preview == norm2.preview,
        true,
    );
    checker.expect_bool(
        "chunk_digests_are_deterministic",
        digests1 == digests2 && norm1.chunks.len() == norm2.chunks.len(),
        true,
    );
    checker.expect_u64(
        "attested_ledger_length_matches_accepted",
        norm2.attested_amounts.len() as u64,
        norm1.preview.accepted_rows,
    );
    drop(norm2);
    eprintln!(
        "[4/7] normalize pass 2 (determinism) in {normalize2_s:.3}s; digests identical: {}",
        digests1 == digests2
    );

    // 5. Decode every chunk with the pipeline's own wire decoder (checksums,
    //    header fields, per-record validation), prove the encode round-trip,
    //    and compute the substrate-facing diagnostics: unique coordinates,
    //    distinct account codes, event-time bounds.
    let t = Instant::now();
    let plan_cfg = plan.to_source_config().expect("plan validates");
    let expected_fingerprint = plan_cfg.fingerprint_hex();
    let mut total_decoded = 0u64;
    let mut total_payload = 0u64;
    let mut sender_codes: HashSet<u64> = HashSet::new();
    let mut receiver_codes: HashSet<u64> = HashSet::new();
    let mut coords: HashSet<u128> = HashSet::new();
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    let mut roundtrip_ok = true;
    let mut header_ok = true;
    let mut chunk_record_counts: Vec<u64> = Vec::new();
    for bytes in &norm1.chunks {
        let seg = match Segment::decode(bytes) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("chunk decode failed: {e}");
                return 1;
            }
        };
        total_payload += bytes.len() as u64;
        chunk_record_counts.push(seg.header.record_count);
        if seg.encode().map(|r| r == *bytes) != Ok(true) {
            roundtrip_ok = false;
        }
        let h = &seg.header;
        if h.account_id_width != 4
            || h.record_length as u64 != RECORD_LEN
            || h.record_count as usize != seg.records.len()
            || h.event_namespace != SAMLD_NAMESPACE
            || hex(&h.config_fingerprint) != expected_fingerprint
            || h.currency_precision != 2
        {
            header_ok = false;
        }
        for rec in &seg.records {
            total_decoded += 1;
            sender_codes.insert(rec.sender_id);
            receiver_codes.insert(rec.receiver_id);
            let t_secs = abacus_ingest::timestamp::event_time_ms_to_seconds(rec.event_time_ms)
                .expect("accepted event times are post-epoch");
            // Width-4 codes and post-2001 epoch seconds fit the 32+32+32 pack.
            let key = ((t_secs as u128) << 64)
                | ((rec.sender_id as u128) << 32)
                | rec.receiver_id as u128;
            coords.insert(key);
            lo = Some(lo.map_or(rec.event_time_ms, |m: i64| m.min(rec.event_time_ms)));
            hi = Some(hi.map_or(rec.event_time_ms, |m: i64| m.max(rec.event_time_ms)));
        }
    }
    let unique_coords = coords.len() as u64;
    let coord_dupes = total_decoded.saturating_sub(unique_coords);
    eprintln!(
        "[5/7] decoded {} chunks, {total_decoded} records, {unique_coords} unique coordinates \
         in {:.3}s",
        norm1.chunks.len(),
        t.elapsed().as_secs_f64()
    );

    // ---- Preview checks. Structural invariants always run; baseline
    //      comparisons run only in full-dataset mode (default). ----
    let baseline_mode = !args.iter().any(|a| a == "--no-baseline");
    let p = &norm1.preview;
    checker.expect_bool(
        "preview_partition_holds (input = accepted + rejected)",
        p.input_rows == p.accepted_rows + p.rejected_rows,
        true,
    );
    checker.expect_bool("no_subsecond_discard", p.discards_subsecond, false);

    // ---- Wire/diagnostics checks: checksums, counts, coordinates, bounds. ----
    let expected_chunks = (p.accepted_rows as usize).div_ceil(RECORDS_PER_CHUNK) as u64;
    let expected_payload = expected_chunks * HEADER_LEN + p.accepted_rows * RECORD_LEN;
    checker.expect_bool("segment_roundtrip_encode_identical", roundtrip_ok, true);
    checker.expect_bool("segment_headers_consistent", header_ok, true);
    checker.expect_u64(
        "decoded_records == accepted_rows",
        total_decoded,
        p.accepted_rows,
    );
    checker.expect_u64("chunk_count", norm1.chunks.len() as u64, expected_chunks);
    checker.expect_u64("total_payload_bytes", total_payload, expected_payload);
    let last_chunk_records = p.accepted_rows % u64::try_from(RECORDS_PER_CHUNK).unwrap_or(1);
    let full_chunk_records = u64::try_from(RECORDS_PER_CHUNK).unwrap_or(0);
    let per_chunk_shape_ok = chunk_record_counts.iter().enumerate().all(|(i, &n)| {
        if i + 1 == chunk_record_counts.len() {
            n == last_chunk_records
        } else {
            n == full_chunk_records
        }
    });
    checker.expect_bool(
        "per_chunk_record_counts_bounded_and_complete",
        per_chunk_shape_ok,
        true,
    );
    let rejects_all_amounts = p
        .issues
        .iter()
        .all(|i| i.code == abacus_ingest::ErrorCode::AmountOrCurrency);
    checker.expect_bool(
        "every_rejection_is_amount_category",
        rejects_all_amounts,
        true,
    );

    if baseline_mode {
        // Recorded comparison context (the baseline's own sender/receiver set
        // counts; the accepted-row equivalents are checked below).
        println!(
            "BASELINE num_senders={BASELINE_NUM_SENDERS} num_receivers={BASELINE_NUM_RECEIVERS} \
             (accepted rows lose 22 senders and 16 receivers that appear only in rejected rows)"
        );
        checker.expect_u64("merge_row_count", merge_rows, BASELINE_NUM_ROWS);
        checker.expect_u64(
            "input_rows (baseline num_rows)",
            p.input_rows,
            BASELINE_NUM_ROWS,
        );
        checker.expect_u64(
            "rejected_rows (excess precision, never rounded)",
            p.rejected_rows,
            EXPECTED_REJECTED_ROWS,
        );
        checker.expect_u64(
            "accepted_rows (baseline − rejected)",
            p.accepted_rows,
            EXPECTED_ACCEPTED_ROWS,
        );
        checker.expect_u64(
            "distinct_accounts (union)",
            p.distinct_accounts,
            EXPECTED_UNION_ACCOUNTS,
        );
        checker.expect_u64(
            "distinct_sender_codes (accepted rows)",
            sender_codes.len() as u64,
            EXPECTED_SENDERS_ACCEPTED,
        );
        checker.expect_u64(
            "distinct_receiver_codes (accepted rows)",
            receiver_codes.len() as u64,
            EXPECTED_RECEIVERS_ACCEPTED,
        );
        checker.expect_u64(
            "unique_coordinates (accepted rows)",
            unique_coords,
            EXPECTED_UNIQUE_COORDS_ACCEPTED,
        );
        checker.expect_u64(
            "coordinate_duplicates (== baseline duplicates_absorbed)",
            coord_dupes,
            BASELINE_NUM_DUPLICATES_ABSORBED,
        );
        checker.expect_u64(
            "unique_coords_plus_rejected (baseline cross-check)",
            unique_coords + EXPECTED_REJECTED_ROWS,
            BASELINE_NUM_UNIQUE_COORDINATES,
        );
        checker.expect_i64(
            "min_event_time_s (baseline t_min)",
            lo.unwrap_or(0) / 1000,
            BASELINE_T_MIN_S,
        );
        checker.expect_i64(
            "max_event_time_s (baseline t_max)",
            hi.unwrap_or(0) / 1000,
            BASELINE_T_MAX_S,
        );
    }

    // 6. The real run: driver → client → /api/v1 → service → substrate CAS,
    //    with the A1 attested-amount commit, over the patient transport.
    let client = CsvIngestClient::new(addr.clone()).with_transport(Box::new(PatientTransport {
        addr: addr.clone(),
        // The A2 durable commit fsyncs one journal frame per applied record
        // (~3ms/frame measured on this machine), so a full-dataset commit is a
        // single multi-hour request. 12h bounds a hung server without cutting
        // a healthy one short.
        timeout: Duration::from_secs(43_200),
    }));
    let t = Instant::now();
    let outcome = match run_import(&merged, &plan, &client, true) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[6/7] import refused: {e}");
            return 1;
        }
    };
    let run_s = t.elapsed().as_secs_f64();
    eprintln!(
        "[6/7] run_import: session {} applied={} already_applied={} in {run_s:.3}s",
        outcome.session_id, outcome.commit.applied, outcome.commit.already_applied
    );

    // 7. The reconciled report (§10) — machine-checked, not merely displayed.
    let r = &outcome.report;
    let counts = r.get("counts");
    let get_u64 = |j: &abacus_ingest::api::json::Json, k: &str| -> u64 {
        j.get(k).and_then(|v| v.as_u64()).unwrap_or(u64::MAX)
    };
    let (input, accepted, duplicate, conflicting, rejected) = match counts {
        Some(c) => (
            get_u64(c, "input"),
            get_u64(c, "accepted"),
            get_u64(c, "duplicate"),
            get_u64(c, "conflicting"),
            get_u64(c, "rejected"),
        ),
        None => (u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX),
    };
    checker.expect_bool(
        "report.reconciles",
        r.get("reconciles")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        true,
    );
    checker.expect_u64("report.counts.input", input, p.accepted_rows);
    checker.expect_u64("report.counts.accepted", accepted, p.accepted_rows);
    checker.expect_u64("report.counts.duplicate", duplicate, 0);
    checker.expect_u64("report.counts.conflicting", conflicting, 0);
    checker.expect_u64("report.counts.rejected (server-side)", rejected, 0);
    checker.expect_u64(
        "report.chunk_count",
        get_u64(r, "chunk_count"),
        expected_chunks,
    );
    checker.expect_u64(
        "report.payload_bytes",
        get_u64(r, "payload_bytes"),
        total_payload,
    );
    let commit = r.get("commit");
    checker.expect_u64(
        "report.commit.applied",
        commit.map(|c| get_u64(c, "applied")).unwrap_or(u64::MAX),
        p.accepted_rows,
    );
    checker.expect_u64(
        "report.commit.already_applied",
        commit
            .map(|c| get_u64(c, "already_applied"))
            .unwrap_or(u64::MAX),
        0,
    );
    checker.expect_str(
        "report.final_state",
        r.get("final_state").and_then(|v| v.as_str()).unwrap_or(""),
        "committed",
    );
    checker.expect_str(
        "report.event_namespace",
        r.get("event_namespace")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        &SAMLD_NAMESPACE.to_hyphenated(),
    );
    checker.expect_str(
        "report.config_fingerprint",
        r.get("config_fingerprint")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        &expected_fingerprint,
    );
    if baseline_mode {
        checker.expect_i64(
            "report.min_event_time_ms (baseline t_min)",
            r.get("min_event_time_ms")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            BASELINE_T_MIN_S * 1000,
        );
        checker.expect_i64(
            "report.max_event_time_ms (baseline t_max)",
            r.get("max_event_time_ms")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            BASELINE_T_MAX_S * 1000,
        );
    }

    // ---- Summary (machine-readable) ----
    let failures = checker.failures();
    let summary = format!(
        "{{\"dataset\":\"{path}\",\"raw_bytes\":{raw_bytes},\"merged_bytes\":{merged_bytes},\
         \"input_rows\":{},\"accepted_rows\":{},\"rejected_rows\":{},\
         \"distinct_accounts\":{},\"unique_coordinates\":{unique_coords},\
         \"coordinate_duplicates\":{coord_dupes},\
         \"chunks\":{},\"payload_bytes\":{total_payload},\
         \"session_id\":\"{}\",\"applied\":{},\"already_applied\":{},\
         \"normalize1_s\":{normalize1_s:.3},\"normalize2_s\":{normalize2_s:.3},\
         \"run_import_s\":{run_s:.3},\"checks\":{},\"failures\":{failures},\
         \"chunk_digest0\":\"{}\"}}",
        p.input_rows,
        p.accepted_rows,
        p.rejected_rows,
        p.distinct_accounts,
        norm1.chunks.len(),
        outcome.session_id,
        outcome.commit.applied,
        outcome.commit.already_applied,
        checker.checks.len(),
        digests1.first().map(|d| hex(d)).unwrap_or_default(),
    );
    println!("SUMMARY {summary}");
    for c in &checker.checks {
        if !c.ok {
            println!(
                "FAILED_CHECK {} measured={} expected={}",
                c.name, c.measured, c.expected
            );
        }
    }
    println!("RESULT: {}", if failures == 0 { "PASS" } else { "FAIL" });
    // The exact report JSON, for the restart-recovery comparison.
    println!("REPORT_JSON {}", r.to_json_string());
    if failures == 0 {
        0
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// Mode: fetch one session's report (used after a server restart).
// ---------------------------------------------------------------------------

fn cmd_report(args: &[String]) -> i32 {
    let (addr, session) = match (args.first(), args.get(1)) {
        (Some(a), Some(s)) => (a.clone(), s.clone()),
        _ => {
            eprintln!("usage: samld_validate report <host:port> <session-id>");
            return 2;
        }
    };
    let client = CsvIngestClient::new(addr).with_transport(Box::new(PatientTransport {
        addr: args.first().cloned().unwrap_or_default(),
        timeout: Duration::from_secs(60),
    }));
    match client.report(&session) {
        Ok(r) => {
            println!("STATUS {}", r.status);
            println!("{}", String::from_utf8_lossy(&r.raw));
            if r.status == 200 {
                0
            } else {
                1
            }
        }
        Err(e) => {
            eprintln!("report fetch failed: {e}");
            1
        }
    }
}

// ---------------------------------------------------------------------------
// Mode: re-POST commit for an already-committed session (expect 409).
// ---------------------------------------------------------------------------

fn cmd_recommit(args: &[String]) -> i32 {
    let (addr, session) = match (args.first(), args.get(1)) {
        (Some(a), Some(s)) => (a.clone(), s.clone()),
        _ => {
            eprintln!("usage: samld_validate recommit <host:port> <session-id>");
            return 2;
        }
    };
    let client = CsvIngestClient::new(addr.clone()).with_transport(Box::new(PatientTransport {
        addr,
        timeout: Duration::from_secs(60),
    }));
    // An empty amount vector: a committed session must refuse on STATE before
    // it ever inspects the attestation body (§13 / `Committed` is terminal).
    match client.commit(&session, "{\"amounts\":[]}") {
        Ok(r) => {
            println!("STATUS {}", r.status);
            println!("{}", String::from_utf8_lossy(&r.raw));
            if r.status == 409 {
                println!("RECOMMIT_REFUSED_AS_EXPECTED");
                0
            } else {
                println!("RECOMMIT_UNEXPECTED_STATUS");
                1
            }
        }
        Err(e) => {
            eprintln!("recommit failed: {e}");
            1
        }
    }
}

// ---------------------------------------------------------------------------
// Mode: verify the ABACJRN1 journal's frame structure, counts and checksums.
// ---------------------------------------------------------------------------

fn cmd_journal(args: &[String]) -> i32 {
    let path = match args.first() {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: samld_validate journal <journal-path>");
            return 2;
        }
    };
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            return 1;
        }
    };
    let file_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 12];
    if r.read_exact(&mut magic).is_err() || &magic[..8] != b"ABACJRN1" {
        eprintln!("not an ABACJRN1 journal");
        return 1;
    }
    let mut tags: HashMap<u8, u64> = HashMap::new();
    let mut frames = 0u64;
    let mut bad_checksums = 0u64;
    let mut torn_tail = false;
    let mut len_buf = [0u8; 4];
    loop {
        // Fill the 4-byte frame length: a short read is NOT end-of-file, only
        // zero bytes at a frame boundary is. Anything else short is a torn tail.
        let mut filled = 0usize;
        let len = loop {
            match r.read(&mut len_buf[filled..]) {
                Ok(0) => {
                    if filled == 0 {
                        break None; // clean EOF on a frame boundary
                    }
                    torn_tail = true;
                    break Some(u32::MAX); // sentinel: length prefix itself is torn
                }
                Ok(n) => {
                    filled += n;
                    if filled == 4 {
                        break Some(u32::from_le_bytes(len_buf));
                    }
                }
                Err(e) => {
                    eprintln!("journal read failed: {e}");
                    return 1;
                }
            }
        };
        let len = match len {
            None => break,
            Some(u32::MAX) => break,
            Some(l) => l,
        };
        if len == 0 || len > JOURNAL_MAX_PAYLOAD {
            eprintln!("journal frame {frames}: implausible payload length {len}");
            return 1;
        }
        let mut payload = vec![0u8; len as usize + 32];
        if r.read_exact(&mut payload).is_err() {
            torn_tail = true;
            break;
        }
        let (body, checksum) = payload.split_at(len as usize);
        if sha256(body).as_slice() != checksum {
            bad_checksums += 1;
        }
        *tags.entry(body[0]).or_insert(0) += 1;
        frames += 1;
    }
    println!("JOURNAL path={path} bytes={file_bytes} frames={frames}");
    let mut names: Vec<(u8, &str)> = vec![
        (2, "SESSION_OPEN"),
        (3, "CHUNK_ACCEPTED"),
        (4, "RECORD_ACCEPTED"),
        (5, "RECORD_CONFLICT"),
        (6, "SESSION_SEALED"),
        (7, "SESSION_CANCELLED"),
        (8, "EVENT_APPLIED"),
        (9, "SESSION_COMMITTED"),
    ];
    names.sort_unstable();
    for (tag, name) in names {
        if let Some(&n) = tags.get(&tag) {
            println!("JOURNAL_TAG {name}={n}");
        }
    }
    println!("JOURNAL bad_checksums={bad_checksums} torn_tail={torn_tail}");
    if bad_checksums == 0 && !torn_tail {
        println!("JOURNAL_RESULT: PASS");
        0
    } else {
        println!("JOURNAL_RESULT: FAIL");
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_rewrites_only_the_timestamp_column() {
        let raw = b"Time,Date,Sender_account,Receiver_account,Amount,X\n\
                    00:00:07,2023-04-05,2102474519,5041444792,4371.33,keep\n\
                    23:59:58,2023-07-03,1,2,0.05,keep2\n";
        let (merged, rows) = merge_samld_timestamps(raw).unwrap();
        assert_eq!(rows, 2);
        let text = String::from_utf8(merged).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "timestamp_utc,Date,Sender_account,Receiver_account,Amount,X"
        );
        assert_eq!(
            lines[1],
            "2023-04-05T00:00:07Z,2023-04-05,2102474519,5041444792,4371.33,keep"
        );
        assert_eq!(lines[2], "2023-07-03T23:59:58Z,2023-07-03,1,2,0.05,keep2");
    }

    #[test]
    fn merge_fails_closed_on_a_bad_time_or_date() {
        // Shape violations only: range validation (25:00:00, 2023-13-45) is
        // the timestamp parser's job inside `normalize`, not the adapter's.
        assert!(merge_samld_timestamps(b"h,t\n2:00:00,2023-04-05,a,b,c\n").is_err());
        assert!(merge_samld_timestamps(b"h,t\n00:00:07,2023-4-5,a,b,c\n").is_err());
        assert!(merge_samld_timestamps(b"h,t\n00:00:07\n").is_err());
    }

    #[test]
    fn the_plan_is_valid_and_binds_the_expected_config() {
        let plan = samld_plan();
        plan.validate().unwrap();
        let cfg = plan.to_source_config().unwrap();
        assert_eq!(cfg.account_id_width, 4);
        assert_eq!(cfg.currency_precision, 2);
        assert_eq!(cfg.max_amount_minor_units, 100_000_000_000);
        assert_eq!(
            cfg.fingerprint_hex(),
            plan.to_source_config().unwrap().fingerprint_hex()
        );
    }

    #[test]
    fn hex_renders_lowercase_two_digit_bytes() {
        assert_eq!(hex(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }
}
