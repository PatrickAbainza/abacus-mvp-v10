"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ingestApi,
  type IngestPipelineStatus,
  type IngestTransactionsPage,
} from "@/lib/api";

/**
 * Ingestion-backed pipeline view (`/ingest-view`).
 *
 * Renders the CSV/Kafka ingestion data through the shared Account Map and the
 * privacy-preserving derived feature metrics — the migrated `abacus_pipeline`
 * representation (reconciliation directive). Unlike the legacy SAMCORE02
 * streaming_ingestion_demo dashboard (`/`), this view:
 *
 *   - reads from `/api/ingest/*` (the `ingest_pipeline_server`, no SAMCORE02
 *     artifact required),
 *   - shows ONE shared account namespace (an account's label is identical
 *     whether it sent or received — the retired role-separated dictionaries
 *     are not used),
 *   - surfaces the frozen 7-byte derived metrics (log magnitude, threshold
 *     distance, roundedness) INSTEAD of the raw amount.
 *
 * Amounts are never requested or displayed (GS §9 / A1).
 */
const PAGE_SIZE = 50;

export default function IngestViewPage() {
  const [status, setStatus] = useState<IngestPipelineStatus | null>(null);
  const [page, setPage] = useState<IngestTransactionsPage | null>(null);
  const [current, setCurrent] = useState(1);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (p: number) => {
    try {
      const [s, tx] = await Promise.all([
        ingestApi.status(),
        ingestApi.transactions(p, PAGE_SIZE),
      ]);
      setStatus(s);
      setPage(tx);
      setCurrent(tx.page);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load(1);
  }, [load]);

  return (
    <main className="page">
      <header className="page-head">
        <h1>Abacus Ingestion View</h1>
        <p className="muted">
          CSV/Kafka ingestion data over the shared Account Map. Amounts are never
          shown — only the derived feature metrics (log magnitude, threshold
          distance, roundedness). An account&apos;s label is the same whether it
          sent or received.
        </p>
        <p className="muted">
          Serve with:{" "}
          <code>cargo run -p abacus_pipeline --bin ingest_pipeline_server</code>{" "}
          (:8091), then set <code>INGEST_ORIGIN</code> so{" "}
          <code>/api/ingest/*</code> routes here. CSV uploads
          (<code>POST /api/ingest/csv</code>) apply to this same server, so the
          dashboard shows them.
        </p>
      </header>

      {error && <div className="banner banner-error">{error}</div>}

      {status && (
        <section className="card card-wide">
          <h2>Status</h2>
          <div className="ingest-counts">
            <span>
              source <strong>{status.source}</strong>
            </span>
            <span>
              committed <strong>{status.committed}</strong>
            </span>
            <span>
              accounts <strong>{status.accounts}</strong>{" "}
              <span className="muted">(shared namespace)</span>
            </span>
          </div>
        </section>
      )}

      {page && (
        <section className="card card-wide">
          <div className="row between">
            <h2>Transactions</h2>
            <span className="muted">
              page {page.page} / {page.total_pages} · {page.total} total
            </span>
          </div>
          <div className="table-wrap">
            <table className="tx-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Sender</th>
                  <th>Receiver</th>
                  <th>Time (s)</th>
                  <th>log·mag</th>
                  <th>thr·dist</th>
                  <th>round</th>
                </tr>
              </thead>
              <tbody>
                {page.rows.map((r) => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{r.sender}</td>
                    <td>{r.receiver}</td>
                    <td>{r.t}</td>
                    <td>{r.features.log_magnitude}</td>
                    <td>{r.features.relative_threshold_distance}</td>
                    <td>{r.features.roundedness_flags}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row">
            <button
              className="btn"
              onClick={() => void load(Math.max(1, current - 1))}
              disabled={current <= 1}
            >
              ‹ Prev
            </button>
            <button
              className="btn"
              onClick={() => void load(current + 1)}
              disabled={page.total_pages === 0 || current >= page.total_pages}
            >
              Next ›
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
