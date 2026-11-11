//! `abacus_mvp_ingest` — the CSV ingestion integration surface for
//! `abacus_mvp_v10`.
//!
//! This crate does **not** reimplement any canonical ingestion logic. It reuses
//! the `/abacus_ingestion` oracle end-to-end and only adds the wiring that
//! proves CSV events reach *this* repository's `abacus_tensor` substrate:
//!
//! | Concern | Owner (reused, not rebuilt) |
//! |---------|------------------------------|
//! | Frozen 7-byte feature derivation | `abacus_ingest::features` |
//! | 23+2W canonical record + 176B `ABACSEG1` | `abacus_ingest::record` / `wire` |
//! | Compound `(namespace, event_id)` identity | `abacus_ingest::namespace` |
//! | Attested-minor-unit commit path (A1) | `abacus_ingest::apply` / `session` |
//! | CSV column mapping + normalization | `abacus_csv_client` |
//! | The substrate mutated | `abacus_tensor::ingest::IngestionEngine` |
//!
//! The data model followed here is the ingestion canonical model — compound
//! identity, one shared append-only account namespace, millisecond event time,
//! and the frozen threshold-relative features — never the legacy `samld` /
//! `abacus_pipeline` model.

use abacus_csv_client::mapping::{CanonicalField, ColumnMapping};
use abacus_csv_client::policy::{
    AmountFormat, EpochUnit, IngestPlan, NumberPolicy, SubSecondPolicy, TimestampPolicy,
    TimestampStyle,
};
use abacus_csv_client::{CsvOptions, EventNamespace, IdentifierPolicy};
use abacus_ingest::api::json::Json;
use abacus_ingest::api::service::CreateSessionRequest;
use abacus_ingest::api::IngestionService;
use abacus_ingest::record::CanonicalRecord;
use abacus_ingest::report::IngestionReport;
use abacus_ingest::session::{IngestionSession, SessionLimits};
use abacus_ingest::{SubstrateApplier, SubstrateValuePolicy};

/// The deterministic CSV fixture.
///
/// A semicolon-delimited de-DE style export: `;` delimiter, `,` grouping
/// separator, `.` decimal separator (the delimiter/grouping ambiguity a
/// comma-delimited comma-grouped file would raise is avoided by stating the
/// delimiter explicitly). The seven data rows cover every row-level rejection
/// category the client can produce — two amounts that parse, an unparseable
/// amount, a non-UTC-offset timestamp, an empty sender, and excess fractional
/// precision — so the accepted set is exactly the three clean transfers.
pub const CSV: &str = "\
timestamp;sender;receiver;amount
2026-10-12T09:30:00Z;acct-001;acct-002;123.45
2026-10-12T10:00:00Z;acct-002;acct-003;1,234.56
2026-10-12T10:30:00Z;acct-003;acct-004;not-a-number
2026-10-12T11:00:00;acct-004;acct-005;50.00
2026-10-12T11:30:00Z;;acct-006;75.00
2026-10-12T12:00:00Z;acct-006;acct-007;10.005
2026-10-12T12:30:00Z;acct-007;acct-001;999.99
";

/// The exact minor-unit amounts of the three accepted rows, in accepted order
/// (chunk-then-row). These are the A1 attested amounts the commit path requires
/// and re-derives the frozen features from; they never enter a canonical record
/// or the wire.
const CLEAN_AMOUNTS: [i64; 3] = [12_345, 123_456, 99_999];

/// The pinned event namespace, so a retry re-derives identical compound
/// identities (Canonical Contract §3.2: generated once, reused, never
/// regenerated on retry).
#[must_use]
pub fn namespace() -> EventNamespace {
    EventNamespace::from_bytes([
        0x5c, 0x11, 0xa7, 0x40, 0x2e, 0x9d, 0x4b, 0x81, 0x93, 0x05, 0xc7, 0x22, 0x48, 0xbd, 0x6e,
        0x10,
    ])
}

/// The complete explicit [`IngestPlan`] for the fixture (Golden Source §4/§12:
/// every normalization field is declared; there are no silent defaults).
#[must_use]
pub fn plan(records_per_chunk: usize) -> IngestPlan {
    IngestPlan {
        mapping: ColumnMapping::new([
            (CanonicalField::Timestamp, 0),
            (CanonicalField::Sender, 1),
            (CanonicalField::Receiver, 2),
            (CanonicalField::Amount, 3),
        ])
        .expect("the fixture's column mapping is valid"),
        csv: CsvOptions {
            delimiter: b';',
            ..CsvOptions::default()
        },
        timestamp: TimestampPolicy {
            style: TimestampStyle::Iso8601WithOffset,
            source_offset_minutes: None,
        },
        amount: AmountFormat {
            decimal_separator: b'.',
            group_separator: Some(b','),
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
        currency_code: "USD".to_string(),
        currency_precision: 2,
        threshold_minor_units: 10_000,
        threshold_effective_day: 20_738,
        threshold_config_version: 3,
        max_amount_minor_units: 100_000_000_000,
        parsing_policy_version: 1,
        records_per_chunk,
        namespace: Some(namespace()),
    }
}

/// Run the full CSV pipeline **in-process** and return the live substrate
/// applier.
///
/// The path exercised is exactly the canonical one: [`abacus_csv_client::normalize`]
/// turns the CSV into bounded `ABACSEG1` chunks and the attested amounts; an
/// [`IngestionSession`] accepts those chunks, seals, and commits them through a
/// [`SubstrateApplier`] whose engine is the `abacus_mvp_v10` [`abacus_tensor`]
/// substrate. Returning the applier lets the caller read the committed frontier
/// and ledger directly — the acceptance proof that CSV events reached the
/// intended substrate.
///
/// # Errors
/// Any plan, normalization, session or commit rejection, propagated unchanged
/// from the oracle pipeline (no error is invented or swallowed here).
pub fn apply_csv_to_substrate(
    records_per_chunk: usize,
) -> Result<SubstrateApplier, abacus_ingest::IngestError> {
    let plan = plan(records_per_chunk);
    plan.validate()?;

    let norm = abacus_csv_client::normalize(CSV.as_bytes(), &plan, namespace())?;
    if norm.preview.accepted_rows == 0 {
        return Err(abacus_ingest::IngestError::new(
            abacus_ingest::ErrorCode::FileOrCsvParse,
            "the fixture produced no accepted rows",
        ));
    }
    let config = plan.to_source_config()?;

    let mut session = IngestionSession::create(
        "sess-substrate",
        config,
        namespace(),
        SessionLimits::default(),
    )?;
    for (i, chunk) in norm.chunks.iter().enumerate() {
        session.submit_chunk(i as u64 + 1, chunk)?;
    }
    session.seal()?;

    // A1: the attestation supplies the exact minor-unit amount per accepted
    // record ordinal; the session re-derives the frozen features and rejects any
    // mismatch before a single substrate mutation.
    let mut applier = SubstrateApplier::new(SubstrateValuePolicy::AttestedMinorUnits);
    session.commit(&mut applier, |idx: usize, _rec: &CanonicalRecord| {
        CLEAN_AMOUNTS.get(idx).copied()
    })?;
    Ok(applier)
}

/// Bind the real ingestion service over loopback and serve `/api/v1`.
///
/// Returns the client, the bound address and the server thread's join handle.
/// The handler is the actual [`abacus_ingest::api::server::handle_one`], so the
/// completed headless CSV driver drives the genuine HTTP surface — and every
/// commit mutates this repository's substrate.
pub fn start_ingestion_server() -> (
    abacus_csv_client::CsvIngestClient,
    std::net::SocketAddr,
    std::thread::JoinHandle<()>,
) {
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let service = Arc::new(Mutex::new(abacus_ingest::api::IngestionService::new()));
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let svc = Arc::clone(&service);
            thread::spawn(move || abacus_ingest::api::server::handle_one(&svc, stream));
        }
    });
    (
        abacus_csv_client::CsvIngestClient::new(addr.to_string()),
        addr,
        handle,
    )
}

/// Normalize `csv` and drive the **full canonical ingestion path in-process**
/// through a shared [`IngestionService`], returning privacy-safe preview +
/// reconciled report JSON.
///
/// This is the server side of the browser CSV test harness (`POST
/// /api/ingest/csv`). It performs exactly what the headless driver and the
/// `/api/v1` HTTP path perform — `abacus_csv_client::normalize` →
/// `create_session` → `submit_chunk` (per bounded `ABACSEG1` chunk) → `seal` →
/// `commit` (A1 attested amounts) → `report` — using the service's own public
/// API, so the browser exercises the same substrate mutation as curl or the
/// WASM shell. No canonical rule is reimplemented and no legacy `samld` path is
/// used.
///
/// # Errors
/// Any plan, normalization, session or commit rejection, propagated unchanged.
pub fn ingest_csv_bytes(
    service: &mut IngestionService,
    csv: &[u8],
) -> Result<String, abacus_ingest::IngestError> {
    let plan = plan(10_000);
    plan.validate()?;

    let norm = abacus_csv_client::normalize(csv, &plan, namespace())?;
    if norm.preview.accepted_rows == 0 {
        return Err(abacus_ingest::IngestError::new(
            abacus_ingest::ErrorCode::FileOrCsvParse,
            "the CSV produced no accepted rows",
        ));
    }
    let cfg = plan.to_source_config()?;

    let req = CreateSessionRequest {
        source_type: cfg.source_type,
        identity_scheme: cfg.identity_scheme,
        account_id_width: cfg.account_id_width,
        currency_code: cfg.currency_code.as_str().to_string(),
        currency_precision: cfg.currency_precision,
        threshold_minor_units: cfg.threshold_minor_units,
        threshold_effective_day: cfg.threshold_effective_day,
        threshold_config_version: cfg.threshold_config_version,
        parsing_policy_version: cfg.parsing_policy_version,
        max_amount_minor_units: cfg.max_amount_minor_units,
        namespace: Some(namespace()),
    };
    let created = service.create_session(&req)?;
    let sid = created.session_id;

    for (i, chunk) in norm.chunks.iter().enumerate() {
        service.submit_chunk(&sid, i as u64 + 1, chunk)?;
    }
    service.seal(&sid)?;

    // A1 attested amounts, by accepted-record ordinal, read out of the
    // client's commit body (`{"amounts":[...]}`) — the same channel the HTTP
    // commit uses. The session re-derives the frozen features from each and
    // rejects any mismatch before a substrate mutation.
    let amounts = parse_amounts(&norm.attested_amounts.commit_body());
    let _commit = service.commit(&sid, |i: usize, _rec: &CanonicalRecord| {
        amounts.get(i).copied()
    })?;

    let report = service.report(&sid)?;
    Ok(render_ingest_json(&norm.preview, &report))
}

// ===========================================================================
// SAML-D dataset ingestion (mvp_v10 policy, following the abacus_ingestion
// oracle's `samld_validate` + `samld_plan` exactly — nothing invented).
//
// The SAML-D export carries its timestamp in TWO columns (`Time`, `Date`);
// the canonical record (Contract §5/§7) has exactly ONE timestamp field. The
// oracle's solution — replicated below — is one explicit, value-preserving
// in-memory merge `Date + "T" + Time + "Z"` before normalization. No value is
// dropped, reordered or rounded; the raw file on disk is never modified; and
// the ground-truth label columns (`Is_laundering`, `Laundering_type`) are
// simply never mapped, so labels stay out of feature derivation by
// construction (Golden Source §6/§9).
// ===========================================================================

/// The exact SAML-D header signature (first physical line), used only to
/// select the one complete, explicit SAML-D plan — never to partially infer a
/// mapping (§9 Stage 1: a partial/ambiguous mapping is rejected, not guessed).
#[must_use]
pub fn is_samld(csv: &[u8]) -> bool {
    let end = csv.iter().position(|&b| b == b'\n').unwrap_or(csv.len());
    let line = &csv[..end];
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    line.starts_with(b"Time,Date,Sender_account,Receiver_account,Amount,")
}

/// The value-preserving SAML-D timestamp merge, ported verbatim from the
/// oracle's `merge_samld_timestamps`.
///
/// Rewrites each `HH:MM:SS,YYYY-MM-DD,<rest>` row into
/// `YYYY-MM-DDTHH:MM:SSZ,<rest>` and the header's `Time` column into
/// `timestamp_utc`. Column positions of Sender/Receiver/Amount are unchanged.
/// Shape violations (wrong length/separators, non-digits, missing commas) fail
/// closed with the source line; semantic range checks (e.g. `25:00:00`,
/// `2023-13-45`) are left to the timestamp parser inside `normalize`, exactly
/// as in the oracle.
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

/// The explicit SAML-D [`IngestPlan`], following the oracle's `samld_plan`.
///
/// `namespace` is the content-addressed namespace derived from the file so a
/// retry of the same bytes re-derives identical compound identities (§3.2).
#[must_use]
pub fn samld_plan(namespace: EventNamespace, records_per_chunk: usize) -> IngestPlan {
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
        // 1e11 minor units = 1e9 major: above the dataset's largest accepted
        // amount (12,618,498.40 → 1,261,849,840 minor units, measured by the
        // oracle).
        max_amount_minor_units: 100_000_000_000,
        parsing_policy_version: 1,
        records_per_chunk,
        namespace: Some(namespace),
    }
}

/// The content-addressed SAML-D event namespace: the first 16 bytes of the
/// SHA-256 of the raw uploaded bytes. Deterministic and stable, so re-posting
/// the same file re-derives identical identities and the backend CAS treats it
/// as a duplicate delivery (§3.2, §13); two different files get different
/// namespaces.
#[must_use]
pub fn samld_namespace(csv: &[u8]) -> EventNamespace {
    let h = abacus_ingest::hash::sha256(csv);
    let mut b = [0u8; 16];
    b.copy_from_slice(&h[..16]);
    EventNamespace::from_bytes(b)
}

/// Normalize a SAML-D CSV and drive the **full canonical ingestion path
/// in-process** through a shared [`IngestionService`], returning privacy-safe
/// preview + reconciled report JSON.
///
/// The SAML-D two-column timestamp is merged first (value-preserving), then the
/// exact canonical path runs: `normalize` → `create_session` → `submit_chunk`
/// → `seal` → `commit` (A1 attested amounts) → `report`. No canonical rule is
/// reimplemented and no legacy `samld` conversion path is used.
///
/// # Errors
/// A merge shape violation, or any plan/normalization/session/commit
/// rejection, propagated unchanged.
pub fn ingest_samld_bytes(
    service: &mut IngestionService,
    csv: &[u8],
) -> Result<String, abacus_ingest::IngestError> {
    let (merged, _rows) = merge_samld_timestamps(csv).map_err(|e| {
        abacus_ingest::IngestError::new(abacus_ingest::ErrorCode::FileOrCsvParse, e)
    })?;
    let ns = samld_namespace(csv);
    let plan = samld_plan(ns, 100_000);
    plan.validate()?;

    let norm = abacus_csv_client::normalize(&merged, &plan, ns)?;
    if norm.preview.accepted_rows == 0 {
        return Err(abacus_ingest::IngestError::new(
            abacus_ingest::ErrorCode::FileOrCsvParse,
            "the SAML-D CSV produced no accepted rows",
        ));
    }
    let cfg = plan.to_source_config()?;

    let req = CreateSessionRequest {
        source_type: cfg.source_type,
        identity_scheme: cfg.identity_scheme,
        account_id_width: cfg.account_id_width,
        currency_code: cfg.currency_code.as_str().to_string(),
        currency_precision: cfg.currency_precision,
        threshold_minor_units: cfg.threshold_minor_units,
        threshold_effective_day: cfg.threshold_effective_day,
        threshold_config_version: cfg.threshold_config_version,
        parsing_policy_version: cfg.parsing_policy_version,
        max_amount_minor_units: cfg.max_amount_minor_units,
        namespace: Some(ns),
    };
    let created = service.create_session(&req)?;
    let sid = created.session_id;

    for (i, chunk) in norm.chunks.iter().enumerate() {
        service.submit_chunk(&sid, i as u64 + 1, chunk)?;
    }
    service.seal(&sid)?;

    let amounts = parse_amounts(&norm.attested_amounts.commit_body());
    let _commit = service.commit(&sid, |i: usize, _rec: &CanonicalRecord| {
        amounts.get(i).copied()
    })?;

    let report = service.report(&sid)?;
    Ok(render_ingest_json(&norm.preview, &report))
}

/// Parse `{"amounts":[i64,...]}` into a `Vec<i64>` (the A1 attestation vector).
fn parse_amounts(body: &str) -> Vec<i64> {
    match abacus_ingest::api::json::parse(body) {
        Ok(j) => match j.get("amounts") {
            Some(Json::Array(a)) => a.iter().filter_map(Json::as_i64).collect(),
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// One validated chunk, ready for the caller to apply to its own substrate or
/// pipeline: the decoded segment header plus every decoded canonical record in
/// wire order, each paired with its A1 attested minor-unit amount.
///
/// The amounts are chunk-local (sliced from the client's attestation vector in
/// chunk-then-row order), so the caller can attest record-by-record without
/// tracking a global ordinal.
#[derive(Debug, Clone)]
pub struct PreparedChunk {
    /// The validated `ABACSEG1` segment header (`Segment::decode` passed its
    /// full wire §6 validation before these records were released).
    pub header: abacus_ingest::wire::SegmentHeader,
    /// `(canonical record, attested minor-unit amount)`, in wire order.
    pub records: Vec<(CanonicalRecord, i64)>,
}

/// A CSV upload normalized, chunked and session-validated through the exact
/// canonical path (`abacus_csv_client::normalize` → `IngestionSession` create →
/// `submit_chunk` → `seal`), ready for the caller to apply to its own
/// substrate or pipeline.
///
/// Returned by [`prepare_csv_bytes`] / [`prepare_samld_bytes`]: the preview
/// for the UI, the session report the canonical path produced, and — per
/// chunk — the validated records alongside their A1 attested amounts. This is
/// the **same** validation `ingest_csv_bytes` / `ingest_samld_bytes` run; the
/// only difference is that no substrate commit happens here, so a caller with
/// its own substrate (e.g. the `IngestPipeline` view server) applies the
/// returned records itself instead of committing a second, divergent copy.
#[derive(Debug, Clone)]
pub struct PreparedCsv {
    /// Privacy-safe preview for the UI (counts + per-line issues).
    pub preview: abacus_csv_client::Preview,
    /// The canonical session report (counts, fingerprint, reconciliation).
    pub report: IngestionReport,
    /// Validated chunks in upload order.
    pub chunks: Vec<PreparedChunk>,
}

/// Render the privacy-safe preview as JSON (counts + per-line issues only —
/// never a raw account string or amount).
fn preview_json(p: &abacus_csv_client::Preview) -> Json {
    let mut o = Json::object();
    o.set("input_rows", Json::uint(p.input_rows));
    o.set("accepted_rows", Json::uint(p.accepted_rows));
    o.set("rejected_rows", Json::uint(p.rejected_rows));
    o.set("distinct_accounts", Json::uint(p.distinct_accounts));
    o.set(
        "account_id_width",
        Json::uint(u64::from(p.account_id_width)),
    );
    o.set("discards_subsecond", Json::bool(p.discards_subsecond));
    let issues: Vec<Json> = p
        .issues
        .iter()
        .map(|i| {
            let mut j = Json::object();
            j.set("line", Json::uint(i.line as u64));
            j.set("code", Json::str(i.code.as_str()));
            j.set("detail", Json::str(&i.detail));
            j
        })
        .collect();
    o.set("issues", Json::Array(issues));
    o
}

/// Shared core of [`prepare_csv_bytes`] / [`prepare_samld_bytes`]: normalize
/// `csv` under `plan` in `namespace`, session-validate every bounded chunk,
/// seal, and slice the A1 attestation vector per chunk.
///
/// # Errors
/// Any plan, normalization or session rejection, propagated unchanged.
fn prepare_with_plan(
    csv: &[u8],
    plan: &abacus_csv_client::policy::IngestPlan,
    namespace: EventNamespace,
    empty_msg: &'static str,
) -> Result<PreparedCsv, abacus_ingest::IngestError> {
    plan.validate()?;
    let norm = abacus_csv_client::normalize(csv, plan, namespace)?;
    if norm.preview.accepted_rows == 0 {
        return Err(abacus_ingest::IngestError::new(
            abacus_ingest::ErrorCode::FileOrCsvParse,
            empty_msg,
        ));
    }
    let cfg = plan.to_source_config()?;
    let mut session = IngestionSession::create(
        format!("prepare-{}", namespace.to_hyphenated()),
        cfg,
        namespace,
        SessionLimits::default(),
    )?;
    // Decode each chunk independently as well: the session validates and
    // classifies, while `Segment::decode` re-validates the wire bytes and
    // releases the header the caller needs (threshold/major scale). Both go
    // through the same oracle codec — no rule is reimplemented.
    let mut decoded: Vec<abacus_ingest::wire::Segment> = Vec::with_capacity(norm.chunks.len());
    for (i, chunk) in norm.chunks.iter().enumerate() {
        session.submit_chunk(i as u64 + 1, chunk)?;
        decoded.push(abacus_ingest::wire::Segment::decode(chunk)?);
    }
    session.seal()?;
    let attested = parse_amounts(&norm.attested_amounts.commit_body());
    let mut offset = 0usize;
    let mut chunks = Vec::with_capacity(decoded.len());
    for seg in &decoded {
        let n = seg.records.len();
        let amounts = attested.get(offset..offset + n).ok_or_else(|| {
            abacus_ingest::IngestError::new(
                abacus_ingest::ErrorCode::SpecificationGap,
                "attestation vector is shorter than the decoded records",
            )
        })?;
        chunks.push(PreparedChunk {
            header: seg.header.clone(),
            records: seg.records.iter().copied().zip(amounts.iter().copied()).collect(),
        });
        offset += n;
    }
    let report = IngestionReport::from_session(&session);
    Ok(PreparedCsv { preview: norm.preview, report, chunks })
}

/// Normalize a generic CSV and session-validate it, returning the validated
/// records + A1 attested amounts for the caller to apply to its own substrate.
///
/// Same plan/namespace/validation as [`ingest_csv_bytes`], but commits
/// nothing: the caller applies `prepared.chunks` (record, attested amount)
/// pairs through its own apply path.
///
/// # Errors
/// Any plan, normalization or session rejection, propagated unchanged.
pub fn prepare_csv_bytes(csv: &[u8]) -> Result<PreparedCsv, abacus_ingest::IngestError> {
    let plan = plan(10_000);
    prepare_with_plan(csv, &plan, namespace(), "the CSV produced no accepted rows")
}

/// Normalize a SAML-D CSV and session-validate it, returning the validated
/// records + A1 attested amounts for the caller to apply to its own substrate.
///
/// Same merge/plan/namespace/validation as [`ingest_samld_bytes`], but commits
/// nothing.
///
/// # Errors
/// A merge shape violation, or any plan/normalization/session rejection,
/// propagated unchanged.
pub fn prepare_samld_bytes(csv: &[u8]) -> Result<PreparedCsv, abacus_ingest::IngestError> {
    let (merged, _rows) = merge_samld_timestamps(csv).map_err(|e| {
        abacus_ingest::IngestError::new(abacus_ingest::ErrorCode::FileOrCsvParse, e)
    })?;
    let ns = samld_namespace(csv);
    let plan = samld_plan(ns, 100_000);
    prepare_with_plan(&merged, &plan, ns, "the SAML-D CSV produced no accepted rows")
}

/// Combine the oracle's canonical report JSON with the client preview.
fn render_ingest_json(preview: &abacus_csv_client::Preview, report: &IngestionReport) -> String {
    let mut o = report.to_json();
    o.set("preview", preview_json(preview));
    o.to_json_string()
}
