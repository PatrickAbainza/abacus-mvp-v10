"use client";

import { useCallback, useEffect, useState } from "react";
import { ingestApi } from "@/lib/api";
import type { IngestPipelineStatus, IngestTransactionsPage } from "@/lib/api";
import NetworkModal from "@/components/NetworkModal";
import CsvIngestPanel from "@/components/CsvIngestPanel";

const DEFAULT_PAGE_SIZE = 1000;

/**
 * Abacus ingestion dashboard — one page, four areas:
 *   A. Engine status (ingestion source + committed/accounts counts + tx/s throughput)
 *   B. Transaction processing (live committed count)
 *   C. Transaction pagination (plain table, 1,000/page default)
 *   D. Transaction selection -> Transaction / Account network views
 *
 * Reads exclusively from the ingestion-backed pipeline (`/api/ingest/*` via
 * `ingestApi`): transactions keyed by locator with shared-namespace account
 * labels, transaction/account networks over the shared Account Map namespace,
 * and the privacy-preserving derived feature metrics. Amounts are never
 * requested or displayed (task §6 / GS §9 / A1). CSV/Kafka ingestion data is
 * the only source; the former SAMLD streaming_ingestion_demo backend is retired.
 */
export default function Home() {
  // ---- A/B. Engine status (ingestion source + committed/accounts) ----
  const [status, setStatus] = useState<IngestPipelineStatus | null>(null);
  // Throughput tracking: samples of (timestamp, committed) for TPS computation
  const [txSamples, setTxSamples] = useState<{ t: number; c: number }[]>([]);

  // ---- C. Pagination ----
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [txPage, setTxPage] = useState<IngestTransactionsPage | null>(null);
  const [pageError, setPageError] = useState<string | null>(null);
// Chain filter: when active, only shows transactions where sender or
// receiver is also a sender/receiver elsewhere in the visible rows
// (i.e. part of a chain structure A→B→C→D).
const [chainFilter, setChainFilter] = useState(false);
  // Compute chain-filtered rows from the current page's data.
  // A chain account appears as both sender AND receiver in the visible rows.
  let displayRows = txPage?.rows ?? [];
  if (chainFilter && displayRows.length > 0) {
    const senders = new Set(displayRows.map((r) => r.sender));
    const receivers = new Set(displayRows.map((r) => r.receiver));
    const chainAccts = new Set(
      [...senders].filter((a) => receivers.has(a))
    );
    displayRows = displayRows.filter(
      (r) => chainAccts.has(r.sender) || chainAccts.has(r.receiver)
    );
  }

  // ---- D. Selection + network modal ----
  type NetworkKind = "transactions" | "accounts" | "multi-hop";
  const [selected, setSelected] = useState<{ id: number } | null>(null);
  const [networkKind, setNetworkKind] = useState<NetworkKind | null>(null);
  const [hopK, setHopK] = useState(3);

  const poll = useCallback(async () => {
    try {
      const s = await ingestApi.status();
      setStatus(s);
      // Record a throughput sample (timestamp, committed count)
      const now = Date.now();
      setTxSamples((prev) => {
        const samples = [...prev, { t: now, c: s.committed }];
        const cutoff = now - 5000; // 5-second rolling window
        return samples.filter((x) => x.t >= cutoff);
      });
    } catch {
      /* backend not ready yet */
    }
  }, []);

  // Poll ingestion status continuously (drives area B + live pagination).
  useEffect(() => {
    void poll();
    const id = setInterval(() => void poll(), 800);
    return () => clearInterval(id);
  }, [poll]);

  const loadPage = useCallback(async (p: number, ps: number) => {
    try {
      setTxPage(await ingestApi.transactions(p, ps));
      setPageError(null);
    } catch (e) {
      setPageError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Load the current page whenever page/pageSize change, and refresh it on the
  // same cadence as status while any records are committed so newly committed
  // records appear (pagination stays live, area C).
  const committed = status?.committed ?? 0;
  const hasCommitted = committed > 0;
  useEffect(() => {
    void loadPage(page, pageSize);
  }, [loadPage, page, pageSize]);
  useEffect(() => {
    if (!hasCommitted) return;
    const id = setInterval(() => void loadPage(page, pageSize), 1500);
    return () => clearInterval(id);
  }, [hasCommitted, loadPage, page, pageSize]);

  const accounts = status?.accounts ?? 0;
  const rows = txPage?.rows ?? [];
  const totalPages = txPage?.total_pages ?? 1;

  // Throughput: transactions per second over the last 5 seconds
  const throughput = (() => {
    if (txSamples.length < 2) return 0;
    const first = txSamples[0];
    const last = txSamples[txSamples.length - 1];
    const dt = (last.t - first.t) / 1000;
    if (dt <= 0) return 0;
    return (last.c - first.c) / dt;
  })();

  // Numbered page window (task §3C): show up to 7 page buttons around the
  // current page with ellipses, always including first/last where relevant.
  const pageWindow = (() => {
    const win: (number | "…")[] = [];
    const span = 1; // pages either side of current
    const push = (n: number | "…") => {
      if (win[win.length - 1] !== n) win.push(n);
    };
    for (let n = 1; n <= totalPages; n++) {
      if (n === 1 || n === totalPages || (n >= page - span && n <= page + span)) {
        push(n);
      } else if (win[win.length - 1] !== "…") {
        push("…");
      }
    }
    return win;
  })();
  return (
    <main className="page">
      <header className="page-head">
        <h1>Abacus Ingestion</h1>
        <p className="muted">
          CSV/Kafka ingestion data over the shared Account Map: ingestion,
          indexing, pagination and structural network views only — no amounts.
          Accounts are keyed by the single shared namespace (an account&apos;s
          label is identical whether it sent or received). Per-transaction
          derived feature metrics (not the raw amount) are shown.
        </p>
      </header>

      <div className="grid">
        {/* ---------- A/B. Engine status ---------- */}
        <section className="card">
          <h2>A · Ingestion status</h2>
          <div className="stats">
            <div><span className="stat-num">{status?.source ?? "—"}</span><span className="muted">source</span></div>
            <div><span className="stat-num">{committed.toLocaleString()}</span><span className="muted">committed</span></div>
            <div><span className="stat-num">{accounts.toLocaleString()}</span><span className="muted">accounts (shared namespace)</span></div>
            <div><span className="stat-num">{throughput.toFixed(1)}</span><span className="muted">tx/s</span></div>
            <div><span className="stat-num">{status?.status ?? "idle"}</span><span className="muted">state</span></div>
          </div>
        </section>

        {/* ---------- C. Transaction pagination ---------- */}
        <section className="card card-wide">
          <h2>C · Transactions</h2>
          <div className="row between">
            <div className="row pager">
              <button className="btn" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1}>
                ‹ Prev
              </button>
              {pageWindow.map((n, i) =>
                n === "…" ? (
                  <span key={`gap-${i}`} className="muted">…</span>
                ) : (
                  <button
                    key={n}
                    className={`btn page-btn ${n === page ? "page-btn-active" : ""}`}
                    onClick={() => setPage(n)}
                    aria-current={n === page ? "page" : undefined}
                  >
                    {n}
                  </button>
                ),
              )}
              <button
                className="btn"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
              >
                Next ›
              </button>
              <label className="inline jump">
                Go to
                <input
                  type="number"
                  min={1}
                  max={totalPages}
                  value={page}
                  onChange={(e) => {
                    const v = Math.min(totalPages, Math.max(1, Number(e.target.value) || 1));
                    setPage(v);
                  }}
                />
              </label>
            </div>
            <label className="inline">
              Page size
              <input
                type="number"
                min={1}
                max={100000}
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Math.max(1, Number(e.target.value) || DEFAULT_PAGE_SIZE));
                  setPage(1);
                }}
              />
            </label>
          </div>
          <div className="muted page-meta">
            Page {txPage?.page ?? page} of {totalPages} · showing {(page - 1) * pageSize + 1}–
            {Math.min(page * pageSize, txPage?.total ?? 0)} of {(txPage?.total ?? 0).toLocaleString()}{" "}
            committed
          </div>
          <button
            className={`btn btn-outline ${chainFilter ? "btn-active" : ""}`}
            onClick={() => setChainFilter((f) => !f)}
            title="Show only transactions where sender/receiver is also a sender/receiver elsewhere (chain structure)"
          >
            {chainFilter ? "Hide Chain Filter" : "Filter Chain Txns"}
            {chainFilter && " ●"}
          </button>
          {pageError && <div className="banner banner-error">{pageError}</div>}
          <div className="table-wrap">
            <table className="tx-table">
              <thead>
                <tr>
                  <th>Transaction ID</th>
                  <th>Sender</th>
                  <th>Receiver</th>
                  <th>Timestamp</th>
                  <th>log·mag</th>
                  <th>thr·dist</th>
                  <th>round</th>
                </tr>
              </thead>
              <tbody>
                {displayRows.map((r) => (
                  <tr
                    key={r.id}
                    className={selected?.id === r.id ? "row-selected" : ""}
                    onClick={() => setSelected({ id: r.id })}
                  >
                    <td>{r.id}</td>
                    <td>{r.sender}</td>
                    <td>{r.receiver}</td>
                    <td>{r.t}</td>
                    <td>{r.features.log_magnitude}</td>
                    <td>{r.features.relative_threshold_distance}</td>
                    <td>{r.features.roundedness_flags}</td>
                  </tr>
                ))}
                {displayRows.length === 0 && (
                  <tr>
                    <td colSpan={7} className="muted">
                      No committed transactions yet — ingest CSV/Kafka records into the pipeline.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* ---------- D. Network selection ---------- */}
        <section className="card card-wide">
          <h2>D · Network</h2>
          {selected ? (
            <div className="row between">
              <span>
                Selected transaction <strong>#{selected.id}</strong> — choose a network view:
              </span>
              <div className="row">
                <button className="btn btn-primary" onClick={() => setNetworkKind("transactions")}>
                  Transaction Network
                </button>
                <button className="btn btn-primary" onClick={() => setNetworkKind("accounts")}>
                  Account Network
                </button>
                {networkKind === "multi-hop" && (
                  <div className="row">
                    <label>
                      Hop bound k:
                      <input
                        type="number"
                        min={1}
                        max={7}
                        value={hopK}
                        onChange={(e) => setHopK(Math.max(1, Math.min(7, Number(e.target.value) || 2)))}
                      />
                    </label>
                  </div>
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
            <span className="muted">Click a transaction row above to choose a network view.</span>
          )}
        </section>

        {/* ---------- E. CSV ingestion (browser test harness) ---------- */}
        <CsvIngestPanel />
      </div>

      {selected && networkKind && (
        <NetworkModal
          txId={selected.id}
          kind={networkKind}
          k={networkKind === "multi-hop" ? hopK : undefined}
          streaming_ingestion_demoRunning={hasCommitted}
          onClose={() => setNetworkKind(null)}
        />
      )}
    </main>
  );
}

