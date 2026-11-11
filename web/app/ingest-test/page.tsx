"use client";

import CsvIngestPanel from "@/components/CsvIngestPanel";

/**
 * Minimal standalone CSV-upload test page (`/ingest-test`).
 *
 * Unlike the full dashboard (`/`), this page has NO status polling, NO
 * transaction table, and NO network modals. It mounts only the CSV ingestion
 * panel, so it works with JUST the pipeline view server running:
 *
 *   cargo run -p abacus_pipeline --bin ingest_pipeline_server  # 0.0.0.0:8091
 *   cd web && INGEST_ORIGIN=http://127.0.0.1:8091 npm run dev  # :3000
 *
 * then open http://localhost:3000/ingest-test. Note: uploads land in the view
 * server's in-memory pipeline (no persistence), so open `/` against the SAME
 * server to see the uploaded rows in the dashboard.
 */
export default function IngestTestPage() {
  return (
    <main className="page">
      <header className="page-head">
        <h1>Abacus CSV Upload Test</h1>
        <p className="muted">
          Minimal harness: pick a CSV, ingest it into the tensor substrate via the
          canonical <code>ABACSEG1</code> path, read the reconciled report. Amounts are
          never displayed.
        </p>
      </header>
      <div className="grid">
        <CsvIngestPanel />
      </div>
    </main>
  );
}
