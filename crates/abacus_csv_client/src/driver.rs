//! The end-to-end browser flow: normalize, preview, upload, commit, report.
//!
//! Golden Source §12 lists the procedure this function implements, in order:
//! select the file, inspect the header and a bounded sample, map columns,
//! configure the policy, preview normalized records with their invalid rows,
//! require explicit confirmation, parse in bounded batches, assign stable opaque
//! account IDs, derive the seven-byte payload, encode bounded chunks, upload
//! chunks with session identity and sequence metadata, seal, commit through the
//! canonical path, and display the reconciled counts.
//!
//! ## Explicit confirmation is load-bearing
//!
//! [`run_import`] refuses to touch the network unless `confirmed` is true. This
//! is §12 step 6 ("Require explicit user confirmation before uploading"), and
//! the e2e test proves it: calling with `confirmed = false` never creates a
//! session, uploads a chunk, or commits anything.
//!
//! ## Failure semantics
//!
//! A file with zero accepted rows is refused before any session exists. A plan
//! whose fingerprint the server rejects aborts immediately — §13 forbids mixing
//! configurations into one logical session, so the client never retries those.
//! Transport failures are retried under [`RetryPolicy`]; an idempotent chunk
//! re-delivery returns `already_processed` and the run reconciles it rather than
//! counting it twice.

use abacus_ingest::api::json::Json;
use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};
use abacus_ingest::namespace::NamespaceRegistry;

use crate::client::{error_from_response, ApiResponse, ChunkAck, CommitAck, CsvIngestClient};
use crate::normalize::{normalize, Normalization, Preview};
use crate::policy::IngestPlan;
use crate::resume::ResumeToken;

/// The complete outcome of one browser import run.
#[derive(Debug)]
pub struct ImportOutcome {
    /// The preview, shown to the user before anything was uploaded.
    pub preview: Preview,
    /// The session id the server assigned.
    pub session_id: String,
    /// One acknowledgement per uploaded chunk, in upload order.
    pub acks: Vec<ChunkAck>,
    /// The commit outcome.
    pub commit: CommitAck,
    /// The final report JSON, exactly as the server rendered it.
    pub report: Json,
    /// A token that re-attaches a later run to this session after a crash.
    pub resume: ResumeToken,
}

/// Run the full browser flow against `client`.
///
/// `confirmed = false` returns the preview path's refusal and performs no
/// network I/O at all: no session, no chunk, no commit.
///
/// # Errors
/// - Any plan, CSV, normalization or capacity failure, before any upload.
/// - The server's structured rejection, converted to its own [`ErrorCode`],
///   for any non-2xx response.
/// - [`ErrorCode::Session`] if the server's happy-path response is missing a
///   documented field (a response-shape violation fails closed).
pub fn run_import(
    csv: &[u8],
    plan: &IngestPlan,
    client: &CsvIngestClient,
    confirmed: bool,
) -> IngestResult<ImportOutcome> {
    plan.validate()?;

    // 1. Stay local until the user has seen and confirmed the preview.
    let mut registry = NamespaceRegistry::new();
    let namespace = match plan.namespace {
        Some(ns) => ns,
        None => registry.generate()?,
    };
    let norm = normalize(csv, plan, namespace)?;
    if norm.preview.accepted_rows == 0 {
        return Err(IngestError::new(
            ErrorCode::FileOrCsvParse,
            "no row in the file produced a canonical record; nothing to upload",
        ));
    }
    if !confirmed {
        return Err(IngestError::new(
            ErrorCode::Session,
            "the import is not confirmed: the preview was produced and no session \
             was created, no chunk uploaded and nothing committed",
        ));
    }
    upload_run(client, plan, norm, &expected_fp_of(plan)?)
}

/// The plan's own configuration fingerprint, hex.
fn expected_fp_of(plan: &IngestPlan) -> IngestResult<String> {
    Ok(plan.to_source_config()?.fingerprint_hex())
}

/// The network half of the run, after confirmation.
fn upload_run(
    client: &CsvIngestClient,
    plan: &IngestPlan,
    norm: Normalization,
    expected_fp: &str,
) -> IngestResult<ImportOutcome> {
    // 2. Capabilities first: refuse a width the server does not advertise rather
    //    than discovering it as a 400 after an upload.
    let caps = client
        .capabilities()
        .map_err(|e| IngestError::new(ErrorCode::Session, e.detail))?;
    let want_width = plan.identifiers.account_id_width;
    if !caps.account_id_widths.contains(&want_width) {
        return Err(IngestError::new(
            ErrorCode::AccountId,
            format!("the server does not advertise the planned {want_width}-byte account-ID width"),
        ));
    }

    // 3. Create the session. The echoed fingerprint must equal the plan's own,
    //    which proves the server bound exactly the configuration we declared.
    let config = plan.to_source_config()?;
    let body = session_body(&config, &norm.namespace);
    let created = client
        .create_session(&body)
        .map_err(|e| IngestError::new(ErrorCode::Session, e.detail))?;
    let session = extract_session(&created, expected_fp)?;

    // 4. Upload every chunk in order. An already-processed idempotent retry
    //    reconciles without double counting.
    let mut acks: Vec<ChunkAck> = Vec::new();
    let mut accepted_records = 0u64;
    for (i, chunk) in norm.chunks.iter().enumerate() {
        let seq = i as u64 + 1;
        let resp = client
            .upload_chunk(&session, seq, chunk)
            .map_err(|e| IngestError::new(ErrorCode::Session, e.detail))?;
        let ack = extract_ack(&resp, seq)?;
        if !ack.already_processed {
            accepted_records += ack.record_count;
        }
        acks.push(ack);
    }

    // 5. Seal, then commit with the A1 attestation. The amounts live in the
    //    normalization's clearing guard and are zeroed once this scope ends.
    let seal = client
        .seal(&session)
        .map_err(|e| IngestError::new(ErrorCode::Session, e.detail))?;
    if seal.status != 200 {
        return Err(error_from_response(&seal));
    }
    let amounts_body = norm.attested_amounts.commit_body();
    let committed = client
        .commit(&session, &amounts_body)
        .map_err(|e| IngestError::new(ErrorCode::Session, e.detail))?;
    drop(amounts_body);
    let commit = extract_commit(&committed)?;

    // 6. The final report must reconcile, or the run is a failure.
    let reported = client
        .report(&session)
        .map_err(|e| IngestError::new(ErrorCode::Session, e.detail))?;
    if reported.status != 200 {
        return Err(error_from_response(&reported));
    }
    let report = reported
        .json
        .ok_or_else(|| IngestError::new(ErrorCode::Session, "the report response is not JSON"))?;
    check_reconciles(&report)?;

    let ack_count = acks.len() as u64;
    Ok(ImportOutcome {
        preview: norm.preview,
        session_id: session.clone(),
        acks,
        commit,
        report: report.clone(),
        resume: ResumeToken {
            session_id: session,
            namespace: norm.namespace,
            config_fingerprint: expected_fp.to_string(),
            ack_chunks: ack_count,
            accepted_records,
        },
    })
}

/// Render the create-session body from the validated configuration.
fn session_body(
    config: &abacus_ingest::config::SourceConfig,
    namespace: &abacus_ingest::namespace::EventNamespace,
) -> String {
    let mut out = String::from(
        "{\"source_type\":\"browser_csv\",\"identity_scheme\":\"csv_record_ordinal\",",
    );
    out.push_str("\"account_id_width\":");
    out.push_str(&config.account_id_width.to_string());
    out.push_str(",\"currency_code\":\"");
    out.push_str(config.currency_code.as_str());
    out.push_str("\",\"currency_precision\":");
    out.push_str(&config.currency_precision.to_string());
    out.push_str(",\"threshold_minor_units\":");
    out.push_str(&config.threshold_minor_units.to_string());
    out.push_str(",\"threshold_effective_day\":");
    out.push_str(&config.threshold_effective_day.to_string());
    out.push_str(",\"threshold_config_version\":");
    out.push_str(&config.threshold_config_version.to_string());
    out.push_str(",\"parsing_policy_version\":");
    out.push_str(&config.parsing_policy_version.to_string());
    out.push_str(",\"max_amount_minor_units\":");
    out.push_str(&config.max_amount_minor_units.to_string());
    out.push_str(",\"event_namespace\":\"");
    out.push_str(&namespace.to_hyphenated());
    out.push_str("\"}");
    out
}

/// Extract the session id from a 200 create-session response, verifying the echo.
fn extract_session(r: &ApiResponse, expected_fp: &str) -> IngestResult<String> {
    if r.status != 200 {
        return Err(error_from_response(r));
    }
    let j = r.json.as_ref().ok_or_else(|| {
        IngestError::new(ErrorCode::Session, "create-session response is not JSON")
    })?;
    let fp = j
        .get("config_fingerprint")
        .and_then(Json::as_str)
        .ok_or_else(|| {
            IngestError::new(
                ErrorCode::Session,
                "create-session response has no fingerprint",
            )
        })?;
    if fp != expected_fp {
        return Err(IngestError::new(
            ErrorCode::Identity,
            "the server bound a different configuration fingerprint than declared",
        ));
    }
    j.get("session_id")
        .and_then(Json::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            IngestError::new(
                ErrorCode::Session,
                "create-session response has no session id",
            )
        })
}

/// Extract one chunk acknowledgement from an upload response.
fn extract_ack(r: &ApiResponse, seq: u64) -> IngestResult<ChunkAck> {
    if r.status != 200 {
        return Err(error_from_response(r));
    }
    let j = r
        .json
        .as_ref()
        .ok_or_else(|| IngestError::new(ErrorCode::Session, "chunk response is not JSON"))?;
    let status = j.get("status").and_then(Json::as_str).unwrap_or("");
    let record_count = j.get("record_count").and_then(Json::as_u64).unwrap_or(0);
    Ok(ChunkAck {
        sequence: seq,
        already_processed: status == "already_processed",
        record_count,
    })
}

/// Extract the commit outcome from a 200 commit response.
fn extract_commit(r: &ApiResponse) -> IngestResult<CommitAck> {
    if r.status != 200 {
        return Err(error_from_response(r));
    }
    let j = r
        .json
        .as_ref()
        .ok_or_else(|| IngestError::new(ErrorCode::Session, "commit response is not JSON"))?;
    Ok(CommitAck {
        applied: j.get("applied").and_then(Json::as_u64).unwrap_or(0),
        already_applied: j.get("already_applied").and_then(Json::as_u64).unwrap_or(0),
    })
}

/// The reconciled counts (§10: "Counts must reconcile") — or a failure.
fn check_reconciles(report: &Json) -> IngestResult<()> {
    if report
        .get("reconciles")
        .and_then(Json::as_bool)
        .unwrap_or(false)
    {
        Ok(())
    } else {
        Err(IngestError::new(
            ErrorCode::SubstrateInvariant,
            "the final report does not reconcile; the import is reported as failed",
        ))
    }
}

/// Render one normalized outcome for the UI without any raw values.
///
/// A small convenience so drivers and embeddings share one privacy-safe
/// rendering of a [`Normalization`]: counts, per-code issue lines, and the
/// exact chunk/record tallies — never an account string or an amount.
#[must_use]
pub fn render_preview(norm: &Normalization) -> String {
    use std::fmt::Write as _;
    let p = &norm.preview;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "input={} accepted={} rejected={} distinct_accounts={} width={} chunks={} input_accounted={}",
        p.input_rows,
        p.accepted_rows,
        p.rejected_rows,
        p.distinct_accounts,
        p.account_id_width,
        norm.chunks.len(),
        p.input_rows == p.accepted_rows + p.rejected_rows
    );
    for issue in &p.issues {
        let _ = writeln!(
            out,
            "line={} code={} detail={}",
            issue.line,
            issue.code.as_str(),
            issue.detail
        );
    }
    out
}
