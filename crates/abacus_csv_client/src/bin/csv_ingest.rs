//! The `csv_ingest` headless driver for the browser client core.
//!
//! This binary runs the exact browser flow (§12) without a DOM: it reads the
//! selected file's bytes, builds an explicit [`IngestPlan`], prints the
//! privacy-safe preview, requires `--confirm` before any upload, then runs the
//! upload/commit/report flow against the existing `/api/v1` server.
//!
//! Usage: `csv_ingest <csv-path> <host:port> [--confirm] [--out <report-path>]`.
//! The preview is always printed to stdout first, and the exit status is
//! nonzero unless the final report reconciles. The resume token is printed to
//! stderr on success so a later run can re-attach after a crash.

use std::io::Write as _;

use abacus_csv_client::driver::{render_preview, run_import};
use abacus_csv_client::mapping::{CanonicalField, ColumnMapping};
use abacus_csv_client::policy::{
    AmountFormat, EpochUnit, IngestPlan, NumberPolicy, SubSecondPolicy, TimestampPolicy,
    TimestampStyle,
};
use abacus_csv_client::{CsvOptions, IdentifierPolicy};

fn main() {
    if let Err(e) = run() {
        eprintln!("csv_ingest: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: csv_ingest <csv-path> <host:port> [--confirm]";
    let path = args.get(1).ok_or_else(|| usage.to_string())?;
    let addr = args
        .get(2)
        .map_or("127.0.0.1:8099", String::as_str)
        .to_string();
    let confirmed = args.iter().any(|a| a == "--confirm");

    // The selected file's bytes: in a browser these arrive from the File API.
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;

    // The explicit policy for the demo import. Every field is required; there
    // are no silent defaults. Columns follow the fixture export header.
    let mapping = ColumnMapping::new([
        (CanonicalField::Timestamp, 0),
        (CanonicalField::Sender, 1),
        (CanonicalField::Receiver, 2),
        (CanonicalField::Amount, 3),
    ])
    .map_err(|e| format!("column mapping is invalid: {e}"))?;
    let plan = IngestPlan {
        mapping,
        csv: CsvOptions::default(),
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
        threshold_effective_day: 19_000,
        threshold_config_version: 3,
        max_amount_minor_units: 100_000_000_000,
        parsing_policy_version: 1,
        records_per_chunk: 2,
        namespace: None,
    };

    // Show the privacy-safe preview first — even when confirmed, the preview is
    // the record of what was normalized. A preview namespace is used so this
    // local-only rendering binds nothing.
    let preview_ns = abacus_csv_client::EventNamespace::from_bytes([0xC1; 16]);
    let preview_only = abacus_csv_client::normalize(&bytes, &plan, preview_ns)
        .map_err(|e| format!("normalization failed: {e}"))?;
    print!("{}", render_preview(&preview_only));
    std::io::stdout()
        .flush()
        .map_err(|e| format!("stdout: {e}"))?;

    let client = abacus_csv_client::CsvIngestClient::new(addr);
    match run_import(&bytes, &plan, &client, confirmed) {
        Ok(out) => {
            eprintln!(
                "committed session {}: applied={} already_applied={}",
                out.session_id, out.commit.applied, out.commit.already_applied
            );
            eprintln!("resume: {}", out.resume.encode());
            Ok(())
        }
        Err(e) => Err(format!("import refused: {e}")),
    }
}
