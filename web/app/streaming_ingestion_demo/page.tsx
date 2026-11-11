"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Network, StreamingIngestionDemoStatus, StreamingIngestionDemoTransactionsPage } from "@/lib/api";
import StreamingIngestionDemoNetworkModal, { type StreamingIngestionDemoNetworkKind } from "@/components/StreamingIngestionDemoNetworkModal";

const DEFAULT_PAGE_SIZE = 1000;

/**
 * Abacus SAMLD streaming_ingestion_demo dashboard — the cleaner "second /web version".
 *
 * StreamingIngestionDemos `artifacts/profiles/samld_full` (SAMCORE02, ~9.5M records) through the
 * Rust `streaming_ingestion_demo_server` backend and shows:
 *   A. StreamingIngestionDemo controls (start / pause / resume / reset)
 *   B. Throughput metrics — measured tx/s (honest: actual rate at which the
 *      infrastructure can stream, no target throughput set), committed / total,
 *      position, RSS
 *   C. Transaction pagination over the committed, locator-ordered rows
 *   D. Structural network views — transaction / account / multi-hop at hop k
 *      (k = 1..8), rendered by the dependency-free SVG GraphView.
 *
 * All API calls go through the same-origin `/api/*` proxy (see
 * `app/api/[...path]/route.ts`): `/api/streaming_ingestion_demo/*` and `/api/transactions/*`
 * route to the Rust streaming_ingestion_demo server (`BACKEND_ORIGIN`, e.g. :8080). Amounts are
 * never requested or displayed.
 */
export default function StreamingIngestionDemoPage() {
  // ---- A/B. Engine status + throughput ----
  const [status, setStatus] = useState<StreamingIngestionDemoStatus | null>(null);
  // Rolling (timestamp, committed) samples for a client-side TPS cross-check.
  const [txSamples, setTxSamples] = useState<{ t: number; c: number }[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  // StreamingIngestionDemo configuration — no target throughput, natural streaming only.
  const [startError, setStartError] = useState<string | null>(null);

  // ---- C. Pagination ----
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [txPage, setTxPage] = useState<StreamingIngestionDemoTransactionsPage | null>(null);
  const [pageError, setPageError] = useState<string | null>(null);

  // ---- D. Selection + network modal ----
  const [selected, setSelected] = useState<{ id: number } | null>(null);
  const [networkKind, setNetworkKind] = useState<StreamingIngestionDemoNetworkKind | null>(null);
  const [hopK, setHopK] = useState(3);

  const poll = useCallback(async () => {
    try {
      const s = await api.status();
      setStatus(s);
      setLoadError(null);
      // Client-side throughput cross-check over a 5-second rolling window.
      const now = Date.now();
      setTxSamples((prev) => {
        const samples = [...prev, { t: now, c: s.processed }];
        const cutoff = now - 5000;
        return samples.filter((x) => x.t >= cutoff);
      });
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Poll status continuously (drives area B + live pagination).
  useEffect(() => {
    void poll();
    const id = setInterval(() => void poll(), 800);
    return () => clearInterval(id);
  }, [poll]);

  const loadPage = useCallback(async (p: number) => {
    try {
      const tx = await api.transactions(p, pageSize);
      setTxPage(tx);
      setPage(tx.page);
      setPageError(null);
    } catch (e) {
      setPageError(e instanceof Error ? e.message : String(e));
    }
  }, [pageSize]);

  useEffect(() => {
    void loadPage(1);
  }, [loadPage]);

  const measuredTps = status?.measured_tps ?? 0;
  const clientTps =
    txSamples.length > 1
      ? ((txSamples[txSamples.length - 1].c - txSamples[0].c) /
          ((txSamples[txSamples.length - 1].t - txSamples[0].t) / 1000))
      : 0;
  const progress = status ? (status.processed / Math.max(status.total, 1)) * 100 : 0;
  const hasCommitted = status ? status.processed > 0 : false;

  const startStreamingIngestionDemo = async () => {
    setStartError(null);
    try {
      await api.start();
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <>
    <main>
      {/* ---------- A. StreamingIngestionDemo controls ---------- */}
      <section className="card card-wide">
        <h2>A · StreamingIngestionDemo controls</h2>
        <div className="row between">
          <div>
            {/* No target throughput — natural streaming only */}
          </div>
          <div className="row" style={{ gap: 8 }}>
            <button
              className="btn btn-primary"
              onClick={startStreamingIngestionDemo}
              disabled={status?.status === "running"}
            >
              Start streaming
            </button>
            <button
              className="btn"
              onClick={() => void api.pause()}
              disabled={status?.status !== "running"}
            >
              Pause
            </button>
            <button
              className="btn"
              onClick={() => void api.resume()}
              disabled={status?.status !== "paused"}
            >
              Resume
            </button>
            <button className="btn btn-ghost" onClick={() => void api.reset()}>
              Reset
            </button>
          </div>
        </div>
        {startError && <div className="banner banner-error">{startError}</div>}
        <div className="row between" style={{ marginTop: 10 }}>
          <span>
            Source: <strong>artifacts/profiles/samld_full</strong> · backend{" "}
            <code>/api/streaming_ingestion_demo/*</code> → streaming_ingestion_demo server
          </span>
          <span className={`status-pill status-${status?.status ?? "idle"}`}>
            {status?.status ?? "idle"}
          </span>
        </div>
      </section>

      {/* ---------- B. Throughput metrics ---------- */}
      <section className="card card-wide">
        <h2>B · Throughput metrics</h2>
        <div className="stats">
          <div>
            <span className="muted">Measured tx/s</span>
            <span className="stat-num">
              {measuredTps.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
          <div>
            <span className="muted">Client tx/s (5s window)</span>
            <span className="stat-num">
              {clientTps.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
          <div>
            <span className="muted">Committed</span>
            <span className="stat-num">{status?.processed?.toLocaleString() ?? "0"}</span>
          </div>
          <div>
            <span className="muted">Total source</span>
            <span className="stat-num">{status?.total?.toLocaleString() ?? "9,504,852"}</span>
          </div>
        </div>
        <div className="row between" style={{ marginTop: 10 }}>
          <span>
            Position{" "}
            <strong>{status?.position?.toLocaleString() ?? "0"}</strong> · duplicates{" "}
            <strong>{status?.duplicates ?? 0}</strong> · errors{" "}
            <strong>{status?.errors ?? 0}</strong>
          </span>
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.min(100, progress)}%` }} />
          </div>
          <div>
            <span className="muted">RSS now / peak</span>
            <br />
            <span className="stat-num">
              {((status?.current_rss_bytes ?? 0) / 1024 / 1024).toFixed(0)} /
              {((status?.peak_rss_bytes ?? 0) / 1024 / 1024).toFixed(0)} MB
            </span>
          </div>
        </div>
        {hasCommitted && (
          <p className="muted">
            The streaming_ingestion_demo engine commits every record through the canonical pipeline
            (`Pipeline::apply`); measured TPS reflects the real achieved rate.
          </p>
        )}
      </section>

      {/* ---------- C. Transaction pagination ---------- */}
      <section className="card card-wide">
        <h2>C · Committed transactions</h2>
        <div className="row between">
          <span>
            page {txPage?.page ?? 1} / {txPage?.total_pages ?? 0} ·{" "}
            {txPage?.total?.toLocaleString() ?? "0"} total
          </span>
        </div>
        {pageError && <div className="banner banner-error">{pageError}</div>}
        <div className="table-wrap">
          <table className="tx-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Sender</th>
                <th>Receiver</th>
                <th>Time (s)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {txPage?.rows.map((r) => (
                <tr
                  key={r.id}
                  className={selected?.id === r.id ? "row-selected" : ""}
                  onClick={() => setSelected({ id: r.id })}
                >
                  <td>{r.id}</td>
                  <td>{r.sender}</td>
                  <td>{r.receiver}</td>
                  <td>{r.t}</td>
                  <td>{r.status}</td>
                </tr>
              ))}
              {txPage?.rows?.length === 0 && (
                <tr>
                  <td colSpan={5} className="muted">
                    Start the streaming_ingestion_demo to populate committed transactions.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="row">
          <button
            className="btn"
            onClick={() => void loadPage(Math.max(1, page - 1))}
            disabled={page <= 1}
          >
            ‹ Prev
          </button>
          <button
            className="btn"
            onClick={() => void loadPage(page + 1)}
            disabled={!txPage || page >= txPage.total_pages}
          >
            Next ›
          </button>
          <label
            className="muted"
            style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}
          >
            page size
            <input
              type="number"
              min="1"
              value={pageSize}
              onChange={(e) => {
                const v = Math.max(1, Number(e.target.value) || 1);
                setPageSize(v);
                void loadPage(1);
              }}
            />
          </label>
        </div>
      </section>

      {/* ---------- D. Network selection ---------- */}
      <section className="card card-wide">
        <h2>D · Structural network</h2>
        {selected ? (
          <div className="row between">
            <span>
              Selected transaction <strong>#{selected.id}</strong> — choose a network view:
            </span>
            <div className="row">
              <button
                className={`btn btn-primary ${networkKind === "transactions" ? "btn-active" : ""}`}
                onClick={() => setNetworkKind("transactions")}
              >
                Transaction Network
              </button>
              <button
                className={`btn btn-primary ${networkKind === "accounts" ? "btn-active" : ""}`}
                onClick={() => setNetworkKind("accounts")}
              >
                Account Network
              </button>
              {networkKind === "multi-hop" && (
                <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  Hop bound k:
                  <input
                    type="number"
                    min="1"
                    max="8"
                    value={hopK}
                    onChange={(e) =>
                      setHopK(Math.max(1, Math.min(8, Number(e.target.value) || 1)))
                    }
                  />
                </label>
              )}
              <button
                className={`btn btn-primary ${networkKind === "multi-hop" ? "btn-active" : ""}`}
                onClick={() => setNetworkKind("multi-hop")}
              >
                Multi-Hop Network
              </button>
            </div>
          </div>
        ) : (
          <span className="muted">
            Click a transaction row above to choose a network view (hop k = 1..8).
          </span>
        )}
      </section>

      {selected && networkKind && (
        <StreamingIngestionDemoNetworkModal
          txId={selected.id}
          kind={networkKind}
          k={networkKind === "multi-hop" ? hopK : undefined}
          streaming_ingestion_demoRunning={hasCommitted}
          onClose={() => setNetworkKind(null)}
        />
      )}
      </main>
      </>
  );
}

