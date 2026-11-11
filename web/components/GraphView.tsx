"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GraphEdge, GraphNode, Network } from "@/lib/api";

/**
 * Lightweight, dependency-free structural graph renderer (SVG).
 *
 * Layout: a simple deterministic layered/radial placement — no physics, no
 * third-party graph library (task §2: "a simple interactive graph library or a
 * lightweight server-generated graph image"). Supports basic zoom (wheel) and
 * pan (drag). Clicking an edge reveals the supporting transaction locators
 * (task §6). The selected transaction node is highlighted.
 *
 * Determinism: node order comes from the backend (fixed), so the same network
 * always lays out identically (task §9 test 14).
 */
export default function GraphView({ network }: { network: Network }) {
  const WIDTH = 860;
  const HEIGHT = 560;
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const [activeEdge, setActiveEdge] = useState<number | null>(null);

  // Deterministic placement: accounts on an outer ring, transactions on an
  // inner ring, each ordered by its index in `network.nodes` (backend order is
  // stable), so identical input lays out identically (task §9 test 14).
  const positions = useMemo(() => {
    const accounts = network.nodes.filter((n) => n.kind === "account");
    const txs = network.nodes.filter((n) => n.kind === "transaction");
    const pos = new Map<string, { x: number; y: number }>();
    const cx = WIDTH / 2;
    const cy = HEIGHT / 2;
    const acctR = Math.min(WIDTH, HEIGHT) * 0.4;
    accounts.forEach((n, i) => {
      const a =
        accounts.length <= 1 ? -Math.PI / 2 : (i / accounts.length) * Math.PI * 2 - Math.PI / 2;
      pos.set(n.id, { x: cx + acctR * Math.cos(a), y: cy + acctR * Math.sin(a) });
    });
    const txR = acctR * 0.55;
    txs.forEach((n, i) => {
      const a = txs.length <= 1 ? -Math.PI / 2 : (i / txs.length) * Math.PI * 2 - Math.PI / 2;
      pos.set(n.id, { x: cx + txR * Math.cos(a), y: cy + txR * Math.sin(a) });
    });
    return pos;
  }, [network.nodes, WIDTH, HEIGHT]);

  const nodeById = useMemo(() => {
    const m = new Map<string, GraphNode>();
    network.nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [network.nodes]);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setView((v) => {
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      const k = Math.min(4, Math.max(0.2, v.k * factor));
      return { ...v, k };
    });
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    setView((v) => ({ ...v, x: drag.current!.vx + dx, y: drag.current!.vy + dy }));
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  // Reset view + selection when the network identity changes.
  useEffect(() => {
    setView({ x: 0, y: 0, k: 1 });
    setActiveEdge(null);
  }, [network]);

  const edgePath = (e: GraphEdge) => {
    const a = positions.get(e.from);
    const b = positions.get(e.to);
    if (!a || !b) return "";
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const pad = 16;
    return `M ${a.x + ux * pad} ${a.y + uy * pad} L ${b.x - ux * pad} ${b.y - uy * pad}`;
  };
  return (
    <div className="graph-wrap">
      <svg
        width="100%"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="graph-svg"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#7c8aa5" />
          </marker>
          <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#2f6df6" />
          </marker>
        </defs>
        <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
          {network.edges.map((e, i) => (
            <path
              key={i}
              d={edgePath(e)}
              className={`edge ${activeEdge === i ? "edge-active" : ""}`}
              markerEnd={activeEdge === i ? "url(#arrow-active)" : "url(#arrow)"}
              onClick={() => setActiveEdge(activeEdge === i ? null : i)}
            />
          ))}
          {network.nodes.map((n) => {
            const p = positions.get(n.id)!;
            const isAcct = n.kind === "account";
            return (
              <g key={n.id} transform={`translate(${p.x} ${p.y})`}>
                {isAcct ? (
                  <circle
                    r={n.selected ? 22 : 16}
                    className={`node-acct ${n.selected ? "node-selected" : ""}`}
                  />
                ) : (
                  <rect
                    x={-20}
                    y={-12}
                    width={40}
                    height={24}
                    rx={5}
                    className={`node-tx ${n.selected ? "node-selected" : ""} ${n.hop !== undefined ? `hop-${n.hop}` : ""}`}
                  />
                )}
                {!isAcct && n.hop !== undefined && n.hop !== null && (
                  <text
                    className={`hop-badge hop-${n.hop}`}
                    y={-16}
                    textAnchor="middle"
                  >
                    h{n.hop}
                  </text>
                )}
                <text className="node-label" y={isAcct ? 34 : 26} textAnchor="middle">
                  {n.label.length > 14 ? `${n.label.slice(0, 13)}…` : n.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      <div className="graph-legend">
        <span className="legend-item"><span className="swatch swatch-acct" /> account</span>
        <span className="legend-item"><span className="swatch swatch-tx" /> transaction</span>
        <span className="legend-hint">wheel = zoom · drag = pan · click an edge = supporting transactions</span>
      </div>

      {activeEdge !== null && network.edges[activeEdge] && (
        <div className="edge-detail">
          <strong>
            {nodeById.get(network.edges[activeEdge].from)?.label ??
              network.edges[activeEdge].from}{" "}
            →{" "}
            {nodeById.get(network.edges[activeEdge].to)?.label ??
              network.edges[activeEdge].to}
          </strong>
          <div className="edge-txs">
            {network.edges[activeEdge].txs.length} supporting transaction
            {network.edges[activeEdge].txs.length === 1 ? "" : "s"}:{" "}
            {network.edges[activeEdge].txs.slice(0, 40).join(", ")}
            {network.edges[activeEdge].txs.length > 40 ? ", …" : ""}
          </div>
        </div>
      )}
    </div>
  );
}

