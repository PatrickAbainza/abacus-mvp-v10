"use client";

import { useRef, useState } from "react";
import { ingestCsv } from "@/lib/api";
import type { IngestReport } from "@/lib/api";

/**
 * CSV ingestion browser test panel.
 *
 * Streams a selected CSV to the pipeline view server's `/api/ingest/csv`
 * endpoint, which normalizes it and applies it to the SAME `IngestPipeline`
 * the dashboard reads — so the uploaded rows show up in the status,
 * transactions, and network views below. Shows the privacy-safe preview
 * (counts + per-line issues — never amounts) and the reconciled report.
 *
 * Expected CSV: `;`-delimited, header `timestamp;sender;receiver;amount`, with
 * UTC ISO-8601 timestamps (`...Z`) and `.` decimal amounts. This is the
 * server-side test harness; the privacy-preserving path normalizes in-browser
 * (WASM) via the canonical `/api/v1` API.
 */
export default function CsvIngestPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<"uploading" | "processing">("uploading");
  const [uploadPct, setUploadPct] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<IngestReport | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const onIngest = async () => {
    const file = inputRef.current?.files?.[0];
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    setBusy(true);
    setPhase("uploading");
    setUploadPct(0);
    setElapsed(0);
    setError(null);
    setReport(null);
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed((Date.now() - started) / 1000), 200);
    try {
      setFileName(file.name);
      setReport(
        await ingestCsv(file, (fraction) => {
          setUploadPct(Math.round(fraction * 100));
          if (fraction >= 1) setPhase("processing");
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      window.clearInterval(timer);
      setBusy(false);
    }
  };

  return (
    <section className="card card-wide">
      <h2>E · CSV Ingestion (browser test)</h2>
      <p className="muted">
        Upload a CSV to normalize &amp; commit it into the tensor substrate via the
        canonical <code>ABACSEG1</code> path. Two schemas are auto-detected:{" "}
        <strong>SAML-D</strong> exports (<code>Time,Date,Sender_account,Receiver_account,Amount,…</code>
        — the <code>Date</code>+<code>Time</code> columns are merged value-preservingly),
        or a simple <code>;</code>-delimited file with header{" "}
        <code>timestamp;sender;receiver;amount</code> (UTC ISO-8601 timestamps,{" "}
        <code>.</code> decimals). Amounts are never displayed.
      </p>
      <div className="row">
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.txt,text/csv"
          aria-label="CSV file"
        />
        <button className="btn btn-primary" onClick={onIngest} disabled={busy}>
          {busy ? "Ingesting…" : "Ingest CSV"}
        </button>
      </div>

      {error && <div className="banner banner-error">{error}</div>}

      {busy && (
        <div className="ingest-progress" role="status" aria-live="polite">
          {phase === "uploading" ? (
            <>
              <div className="row between">
                <span>
                  Uploading <strong>{uploadPct}%</strong>
                </span>
                <span className="muted">{elapsed.toFixed(1)}s</span>
              </div>
              <div className="progress">
                <div className="progress-bar" style={{ width: `${uploadPct}%` }} />
              </div>
            </>
          ) : (
            <>
              <div className="row between">
                <span>
                  <span className="spinner" aria-hidden="true" /> Processing —
                  normalizing &amp; committing to the substrate…
                </span>
                <span className="muted">{elapsed.toFixed(1)}s</span>
              </div>
              <div className="progress progress-indeterminate" />
            </>
          )}
        </div>
      )}

      {report && (
        <div className="ingest-result">
          <div className="row between">
            <span>
              {fileName ? <strong>{fileName}</strong> : "CSV"} · session{" "}
              <code>{report.session_id}</code>
            </span>
            <span className={`pill ${report.reconciles ? "pill-ok" : "pill-bad"}`}>
              {report.reconciles ? "reconciled" : "NOT reconciled"}
            </span>
          </div>

          <div className="ingest-counts">
            <span>
              input <strong>{report.preview.input_rows}</strong>
            </span>
            <span>
              accepted <strong>{report.preview.accepted_rows}</strong>
            </span>
            <span>
              rejected <strong>{report.preview.rejected_rows}</strong>
            </span>
            <span>
              distinct accounts <strong>{report.preview.distinct_accounts}</strong>
            </span>
            <span>
              committed <strong>{report.commit?.applied ?? 0}</strong>
            </span>
            <span>
              already applied <strong>{report.commit?.already_applied ?? 0}</strong>
            </span>
          </div>

          <div className="muted">
            source <code>{report.source_type}</code> · identity{" "}
            <code>{report.identity_scheme}</code> · width {report.account_id_width}B ·
            fingerprint <code>{report.config_fingerprint.slice(0, 12)}…</code>
          </div>

          {report.preview.issues.length > 0 && (
            <div className="table-wrap">
              <table className="tx-table">
                <thead>
                  <tr>
                    <th>Line</th>
                    <th>Category</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {report.preview.issues.map((iss) => (
                    <tr key={`${iss.line}-${iss.code}`}>
                      <td>{iss.line}</td>
                      <td>{iss.code}</td>
                      <td>{iss.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
