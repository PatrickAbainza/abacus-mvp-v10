"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ingestApi } from "@/lib/api";
import type { Network } from "@/lib/api";
import GraphView from "./GraphView";

export type NetworkKind = "transactions" | "accounts" | "multi-hop";

/**
 * Modal that shows one structural network for a selected transaction and keeps
 * it current as ingestion continues.
 *
 * Freshness model: the ingestion backend's network endpoints read the LIVE
 * committed state on every request (the shared Account Map + temporal indexes),
 * so a re-fetch always reflects any newly ingested transactions that now fall
 * inside the network's neighborhood. The modal:
 *   1. fetches once when opened,
 *   2. exposes a manual **Refresh** button, and
 *   3. auto-polls while open (only after the first fetch reflected committed
 *      state, i.e. there is committed state to watch — `streaming_ingestion_demoRunning` is the
 *      legacy prop name for "ingestion has committed state"; the current
 *      implementation refetches only when `committed` grows past the last
 *      fetch).
 *
 * No amounts are ever shown (task §6). Accounts are keyed by the single shared
 * namespace, so an account is the same node whether it sent or received.
 */
export default function NetworkModal({
  txId,
  kind,
  k = 3,
  onClose,
  streaming_ingestion_demoRunning,
}: {
  txId: number;
  kind: NetworkKind;
  k?: number;
  onClose: () => void;
  streaming_ingestion_demoRunning: boolean;
}) {
  const [network, setNetwork] = useState<Network | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stale, setStale] = useState(false);
  // Committed count captured at the last successful fetch; used to detect that
  // new transactions have been ingested since (a refresh opportunity).
  const lastCommitted = useRef<number | null>(null);
  const POLL_MS = 1500;

  const title =
    kind === "transactions"
      ? "Transaction Network"
      : kind === "accounts"
        ? "Account Network"
        : `Multi-Hop Network (k=${k})`;

  const load = useCallback(
    async (showSpinner: boolean) => {
      if (showSpinner) setLoading(true);
      try {
        const n =
          kind === "transactions"
            ? await ingestApi.txNetwork(txId)
            : kind === "accounts"
              ? await ingestApi.accountNetwork(txId)
              : await ingestApi.multiHopNetwork(txId, k);
        setNetwork(n);
        lastCommitted.current = n.committed;
        setStale(false);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (showSpinner) setLoading(false);
      }
    },
    [kind, txId, k],
  );

  // Fetch on open / when the selected transaction or network kind changes.
  useEffect(() => {
    setNetwork(null);
    setError(null);
    setStale(false);
    lastCommitted.current = null;
    void load(true);
  }, [load]);

  // Auto-poll while the modal is open and ingestion has committed state: when
  // the backend `committed` count grows past what this view was built from,
  // new transactions may be inside the neighborhood — refetch and re-render.
  useEffect(() => {
    if (!streaming_ingestion_demoRunning) return;
    const id = setInterval(async () => {
      try {
        const s = await ingestApi.status();
        const cur = s.committed;
        if (lastCommitted.current !== null && cur > lastCommitted.current) {
          await load(false);
        }
      } catch {
        /* transient: try again next tick */
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [streaming_ingestion_demoRunning, load]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>
            {title} <span className="muted">· tx #{txId}</span>
          </h3>
          <div className="modal-actions">
            {network && (
              <span className="committed-count" title="Committed transactions this view reflects">
                {network.nodes.length} nodes · {network.edges.length} edges ·{" "}
                {network.committed.toLocaleString()} committed
              </span>
            )}
            <button className="btn" onClick={() => void load(true)} disabled={loading}>
              {loading ? "Refreshing…" : "Refresh"}
            </button>
            <button className="btn btn-ghost" onClick={onClose} aria-label="Close">
              ✕
            </button>
          </div>
        </div>

        {stale && (
          <div className="banner banner-info">
            New transactions have been ingested — refreshing the network…
          </div>
        )}
        {network?.truncated && (
          <div className="banner banner-warn">
            This neighborhood hit the per-lookup cap (2,500) and may be incomplete.
          </div>
        )}
        {error && <div className="banner banner-error">Failed to load network: {error}</div>}

        <div className="modal-body">
          {!network && !error && <div className="muted">Loading network…</div>}
          {network && network.nodes.length === 0 && (
            <div className="muted">No committed neighbors for this transaction yet.</div>
          )}
          {network && network.nodes.length > 0 && <GraphView network={network} />}
        </div>

        <div className="modal-foot muted">
          Structural view only — transaction amounts are excluded. Refetches the live
          tensor; use Refresh or reopen after new ingestion.
        </div>
      </div>
    </div>
  );
}
