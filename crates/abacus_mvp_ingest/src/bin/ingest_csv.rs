//! The `ingest_csv` headless driver: CSV → `ABACSEG1` → the `abacus_mvp_v10`
//! tensor substrate.
//!
//! This is the integration entry point required by the task. It reuses the
//! completed browser flow (`abacus_csv_client::driver::run_import`) unchanged and
//! drives it against an **in-process** ingestion service
//! ([`abacus_mvp_ingest::start_ingestion_server`]) whose substrate is this
//! repository's `abacus_tensor`. No canonical logic is reimplemented and no
//! legacy `samld`/`abacus_pipeline` path is used.
//!
//! Usage: `ingest_csv <csv-path>`.
//!
//! The privacy-safe preview is printed first; the exit status is nonzero unless
//! the final report reconciles. Explicit confirmation is load-bearing: the
//! driver passes `confirmed = true` because invocation on the command line *is*
//! the explicit confirmation (Golden Source §12 step 6).

use std::io::Write as _;

use abacus_csv_client::driver::run_import;

fn main() {
    if let Err(e) = run() {
        eprintln!("ingest_csv: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    // `--emit <outdir>`: write the bounded ABACSEG1 chunk(s) and the attested
    // amounts to `<outdir>` and exit, without uploading. This is the seam for a
    // curl test of the canonical `/api/v1` API — a browser cannot hand-craft
    // ABACSEG1 bytes, so this emits them from the same normalizer the pipeline
    // uses. Produces `chunk-<i>.abacseg1` and `amounts.json`.
    if let Some(pos) = args.iter().position(|a| a == "--emit") {
        let outdir = args
            .get(pos + 1)
            .ok_or_else(|| "--emit requires an output directory".to_string())?;
        let path = args
            .get(1)
            .ok_or_else(|| "usage: ingest_csv <csv-path> --emit <outdir>".to_string())?;
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let plan = abacus_mvp_ingest::plan(10_000);
        plan.validate().map_err(|e| format!("invalid plan: {e}"))?;
        let norm = abacus_csv_client::normalize(&bytes, &plan, abacus_mvp_ingest::namespace())
            .map_err(|e| format!("normalization failed: {e}"))?;
        std::fs::create_dir_all(outdir).map_err(|e| format!("cannot create {outdir}: {e}"))?;
        for (i, chunk) in norm.chunks.iter().enumerate() {
            let f = format!("{outdir}/chunk-{i}.abacseg1");
            std::fs::write(&f, chunk).map_err(|e| format!("cannot write {f}: {e}"))?;
            println!("wrote {f} ({} bytes)", chunk.len());
        }
        let af = format!("{outdir}/amounts.json");
        std::fs::write(&af, norm.attested_amounts.commit_body())
            .map_err(|e| format!("cannot write {af}: {e}"))?;
        println!("wrote {af}");
        println!(
            "preview: input={} accepted={} rejected={}",
            norm.preview.input_rows, norm.preview.accepted_rows, norm.preview.rejected_rows
        );
        return Ok(());
    }

    let path = args
        .get(1)
        .ok_or_else(|| "usage: ingest_csv <csv-path>".to_string())?;

    // The selected file's bytes (in a browser these arrive from the File API).
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;

    // An in-process ingestion service whose substrate is abacus_mvp_v10's tensor.
    let (client, addr, _server) = abacus_mvp_ingest::start_ingestion_server();
    eprintln!("ingest_csv: in-process /api/v1 at {addr}");

    // A single bounded chunk; the plan is the explicit fixture policy.
    let plan = abacus_mvp_ingest::plan(10_000);

    let out =
        run_import(&bytes, &plan, &client, true).map_err(|e| format!("import failed: {e}"))?;

    // The privacy-safe preview: counts and per-line issue codes only.
    let pv = &out.preview;
    println!(
        "preview: input_rows={} accepted_rows={} rejected_rows={}",
        pv.input_rows, pv.accepted_rows, pv.rejected_rows
    );
    for issue in &pv.issues {
        println!("  line {}: {}", issue.line, issue.code.as_str());
    }

    // The reconciled report: the stop condition is a report that reconciles.
    let reconciles = out.report.get("reconciles").and_then(|v| v.as_bool());
    let applied = out
        .report
        .get("commit")
        .and_then(|c| c.get("applied"))
        .and_then(|v| v.as_u64());
    println!(
        "session={} commit.applied={:?} reconciles={reconciles:?}",
        out.session_id, applied
    );

    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "ingest_csv: session {}", out.session_id);

    if reconciles != Some(true) {
        return Err("the final report did not reconcile".to_string());
    }
    Ok(())
}
