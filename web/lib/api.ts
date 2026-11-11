// Shared API types + a tiny fetch client for the Abacus streaming_ingestion_demo backend.
// All calls go to same-origin `/api/*` (proxied to the Rust server by
// `next.config.js`). Structural fields only — amounts are never requested or
// displayed (task §6).

// ---------------------------------------------------------------------------
// Legacy streaming_ingestion_demo UI surface (retired with the CSV/Kafka ingestion cutover).
//
// These types are retained only as documentation of the former `/api/streaming_ingestion_demo/*`
// + `/api/transactions/*` surface. No component calls them anymore; the live
// dashboard (`app/page.tsx`), `NetworkModal` and the `/ingest-view` page all
// read through `ingestApi` (`/api/ingest/*`, shared Account Map namespace +
// derived feature metrics). Kept so the removed-route drift is visible in one
// place rather than silently dropped.
// ---------------------------------------------------------------------------

export type StreamingIngestionDemoStatusName = "idle" | "running" | "paused" | "completed" | "error";

/** Retired `GET /api/streaming_ingestion_demo/status` (replaced by `IngestPipelineStatus`). */
export interface StreamingIngestionDemoStatus {
  status: StreamingIngestionDemoStatusName;
  position: number;
  total: number;
  processed: number;
  duplicates: number;
  errors: number;
  measured_tps: number;
  /** Live process RSS (bytes), sampled on each status snapshot. */
  current_rss_bytes: number;
  /** Peak RSS observed during the run (bytes). */
  peak_rss_bytes: number;
  last_error: string | null;
}

/** One committed transaction row keyed by its stable locator (event id). */
export interface StreamingIngestionDemoTransactionDetail {
  id: number;
  sender: string;
  receiver: string;
  t: number;
  status: string;
}

/** Paginated committed transactions (locator-ascending, dense slices). */
export interface StreamingIngestionDemoTransactionsPage {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  rows: StreamingIngestionDemoTransactionDetail[];
}

/** Retired row of `GET /api/transactions` (role labels + status, no features). */
export interface TransactionRow {
  id: number;
  sender: string;
  receiver: string;
  t: number;
  status: string;
}

/** Retired `GET /api/transactions?page=&page_size=`. */
export interface TransactionsPage {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  rows: TransactionRow[];
}

/** Retired `GET /api/transactions/{id}`. */
export interface TransactionDetail {
  id: number;
  sender: string;
  receiver: string;
  t: number;
  status: string;
}

/** Retired `GET /api/transactions/{id}`. */
export interface TransactionDetail {
  id: number;
  sender: string;
  receiver: string;
  t: number;
  status: string;
}

export type NodeKind = "transaction" | "account";

/** A node in either network view. */
export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  t?: number;
  selected: boolean;
  hop?: number;
}

/** A directed edge; `txs` lists supporting transaction locators. */
export interface GraphEdge {
  from: string;
  to: string;
  txs: number[];
}

/** `GET /api/transactions/{id}/network/{transactions|accounts}` */
export interface Network {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
  committed: number;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body?.error) msg = body.error;
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const parsed = (await res.json()) as { error?: string };
      if (parsed?.error) msg = parsed.error;
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

/** Retired `GET /api/streaming_ingestion_demo/*` + `/api/transactions/*` client (no live callers). */
export const api = {
  status: () => getJson<StreamingIngestionDemoStatus>("/api/streaming_ingestion_demo/status"),
  /** Start a clean streaming_ingestion_demo with a target rate (tps) and chunk size. */
  start: () => postJson<StreamingIngestionDemoStatus>("/api/streaming_ingestion_demo/start", {}),
  pause: () => postJson<StreamingIngestionDemoStatus>("/api/streaming_ingestion_demo/pause"),
  resume: () => postJson<StreamingIngestionDemoStatus>("/api/streaming_ingestion_demo/resume"),
  reset: () => postJson<StreamingIngestionDemoStatus>("/api/streaming_ingestion_demo/reset"),
  transactions: (page: number, pageSize: number) =>
    getJson<StreamingIngestionDemoTransactionsPage>(`/api/transactions?page=${page}&page_size=${pageSize}`),
  transaction: (id: number) => getJson<StreamingIngestionDemoTransactionDetail>(`/api/transactions/${id}`),
  txNetwork: (id: number) =>
    getJson<Network>(`/api/transactions/${id}/network/transactions`),
  accountNetwork: (id: number) =>
    getJson<Network>(`/api/transactions/${id}/network/accounts`),
  /** Multi-hop structural network with configurable hop bound `k` (1..8). */
  tensorNetwork: (id: number, k = 3) =>
    getJson<Network>(`/api/transactions/${id}/network/tensor?k=${k}`),
};

// ---------------------------------------------------------------------------
// CSV ingestion (browser test harness)
// ---------------------------------------------------------------------------

/** One rejected CSV row, privacy-safe (source line + category + detail). */
export interface IngestIssue {
  line: number;
  code: string;
  detail: string;
}

/** Privacy-safe normalization preview (counts + per-line issues; no amounts). */
export interface IngestPreview {
  input_rows: number;
  accepted_rows: number;
  rejected_rows: number;
  distinct_accounts: number;
  account_id_width: number;
  discards_subsecond: boolean;
  issues: IngestIssue[];
}

/** Reconciled ingestion report from `POST /api/ingest/csv`. */
export interface IngestReport {
  session_id: string;
  source_type: string;
  identity_scheme: string;
  config_fingerprint: string;
  account_id_width: number;
  chunk_count: number;
  counts: {
    input: number;
    accepted: number;
    duplicate: number;
    conflicting: number;
    rejected: number;
  };
  commit: { applied: number; already_applied: number } | null;
  final_state: string;
  reconciles: boolean;
  preview: IngestPreview;
  error?: string;
}

/**
 * Upload a CSV file to the ingestion test-harness endpoint and return the
 * privacy-safe preview + reconciled report. Sent as `text/csv`; the backend
 * normalizes and commits it into the tensor substrate.
 *
 * Uses `XMLHttpRequest` (not `fetch`) so the caller gets real upload-progress
 * events via `onUploadProgress` (fraction 0..1 of request bytes sent). Once
 * the upload completes the server normalizes + commits before responding;
 * that phase has no server-sent progress (single request/response), so the
 * caller should show an indeterminate "processing" state until resolve.
 */
export function ingestCsv(
  file: File,
  onUploadProgress?: (fraction: number) => void,
): Promise<IngestReport> {
  return new Promise((resolve, reject) => {
    const fail = (msg: string) => reject(new Error(msg));
    let body: ArrayBuffer;
    file
      .arrayBuffer()
      .then((buf) => {
        body = buf;
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/ingest/csv");
        xhr.setRequestHeader("Content-Type", "text/csv");
        xhr.setRequestHeader("Accept", "application/json");
        xhr.responseType = "json";
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable && ev.total > 0 && onUploadProgress) {
            onUploadProgress(Math.min(1, ev.loaded / ev.total));
          }
        };
        xhr.onload = () => {
          const parsed = xhr.response as IngestReport | null;
          if (xhr.status >= 200 && xhr.status < 300 && parsed) {
            resolve(parsed);
          } else {
            fail(parsed?.error || `${xhr.status} ${xhr.statusText}`);
          }
        };
        xhr.onerror = () => fail("upload failed: network error");
        xhr.ontimeout = () => fail("upload failed: timed out");
        xhr.send(body);
      })
      .catch(() => fail("could not read the selected file"));
  });
}

// ---------------------------------------------------------------------------
// Ingestion-backed pipeline view (reconciliation: shared Account Map +
// derived feature metrics). Served by `ingest_pipeline_server` under
// `/api/ingest/*`. No raw amount is ever exposed — only the frozen 7-byte
// derived feature payload (GS §9 / A1).
// ---------------------------------------------------------------------------

/** `GET /api/ingest/status` */
export interface IngestPipelineStatus {
  source: "csv_kafka_ingestion";
  status: string;
  committed: number;
  accounts: number;
}

/** The frozen seven-byte amount-derived feature metrics (privacy-safe). */
export interface FeatureMetrics {
  log_magnitude: number;
  relative_threshold_distance: number;
  roundedness_flags: number;
}

/** One row of `GET /api/ingest/transactions`. Shared-namespace labels. */
export interface IngestTransactionRow {
  id: number;
  sender: string;
  receiver: string;
  t: number;
  features: FeatureMetrics;
}

/** `GET /api/ingest/transactions?page=&page_size=` */
export interface IngestTransactionsPage {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  rows: IngestTransactionRow[];
}

/** `GET /api/ingest/transactions/{id}` */
export interface IngestTransactionDetail extends IngestTransactionRow {}

export const ingestApi = {
  status: () => getJson<IngestPipelineStatus>("/api/ingest/status"),
  transactions: (page: number, pageSize: number) =>
    getJson<IngestTransactionsPage>(
      `/api/ingest/transactions?page=${page}&page_size=${pageSize}`,
    ),
  transaction: (id: number) =>
    getJson<IngestTransactionDetail>(`/api/ingest/transactions/${id}`),
  txNetwork: (id: number) =>
    getJson<Network>(`/api/ingest/transactions/${id}/network/transactions`),
  accountNetwork: (id: number) =>
    getJson<Network>(`/api/ingest/transactions/${id}/network/accounts`),
  multiHopNetwork: (id: number, k: number) =>
    getJson<Network>(
      `/api/ingest/transactions/${id}/network/multi-hop?k=${k}`,
    ),
};

// ---------------------------------------------------------------------------
// CSF tensor read boundary (GET /api/ingest/csf/*).
//
// Proxied same-origin to the `IngestPipeline` view server (`INGEST_ORIGIN`,
// :8091) by `web/app/api/[...path]/route.ts` + `lib/route.ts` — the
// `/api/ingest/` prefix is owned by the ingest origin, so no proxy change is
// required here. These endpoints expose the canonical `abacus_tensor::api` read
// boundary (time_slice / lookup / decode / stats) to computation consumers over
// HTTP.
// ---------------------------------------------------------------------------

/** Exact point read response: `found: false` iff the coordinate is absent. */
export interface CsfLookup {
  t: number;
  s: number;
  r: number;
  found: boolean;
  value: number | null;
}

/** One canonical `(t, s, r, value)` entry in `prec_3` order. */
export interface CsfEntry {
  t: number;
  s: number;
  r: number;
  value: number;
}

/** Half-open time-window response `T|_[t_start, t_end)`. */
export interface CsfTimeSlice {
  t_start: number;
  t_end: number;
  entries: CsfEntry[];
}

/** Full live decode response. */
export interface CsfDecode {
  entries: CsfEntry[];
}

/** Derived amount-based feature metrics for one event at a CSF coordinate. */
export interface CsfFeatureEntry {
  /** `round(1000 * ln(1 + amount_minor_units))` (log1p magnitude, GS §9). */
  log_magnitude: number;
  /** `round(1000 * (amount - T) / T)` (z-threshold / relative distance, GS §9). */
  relative_threshold_distance: number;
  /** Exact-divisibility bit field (GS §9). */
  roundedness_flags: number;
}

/** Derived feature metrics for every event at coordinate `(t, s, r)`. */
export interface CsfFeaturesView {
  t: number;
  s: number;
  r: number;
  entries: CsfFeatureEntry[];
}

/** Live tensor statistics. */
export interface CsfStats {
  nnz: number;
}

/** Live sender-at-time view `(t, s) -> [(r, value)]`. */
export interface CsfSenderView {
  t: number;
  s: number;
  entries: CsfDimensionEntry[];
}

/** Live receiver-at-time view `(t, r) -> [(s, value)]`. */
export interface CsfReceiverView {
  t: number;
  r: number;
  entries: CsfDimensionEntry[];
}

/** One entry of a sender/receiver-at-time view: the opposite-end entity + value. */
export interface CsfDimensionEntry {
  value: number;
  /** Receiver id for a sender view; sender id for a receiver view. */
  peer: number;
}

export const csfApi = {
  /** `GET /api/ingest/csf/lookup?t=&s=&r=` */
  lookup: (t: number, s: number, r: number) =>
    getJson<CsfLookup>(`/api/ingest/csf/lookup?t=${t}&s=${s}&r=${r}`),
  /** `GET /api/ingest/csf/time_slice?t_start=&t_end=` */
  timeSlice: (tStart: number, tEnd: number) =>
    getJson<CsfTimeSlice>(
      `/api/ingest/csf/time_slice?t_start=${tStart}&t_end=${tEnd}`,
    ),
  /** `GET /api/ingest/csf/decode` */
  decode: () => getJson<CsfDecode>("/api/ingest/csf/decode"),
  /** `GET /api/ingest/csf/stats` */
  stats: () => getJson<CsfStats>("/api/ingest/csf/stats"),
  /** `GET /api/ingest/csf/sender?t=&s=` */
  sender: (t: number, s: number) =>
    getJson<CsfSenderView>(`/api/ingest/csf/sender?t=${t}&s=${s}`),
  /** `GET /api/ingest/csf/receiver?t=&r=` */
  receiver: (t: number, r: number) =>
    getJson<CsfReceiverView>(`/api/ingest/csf/receiver?t=${t}&r=${r}`),
  /** `GET /api/ingest/csf/features?t=&s=&r=` */
  features: (t: number, s: number, r: number) =>
    getJson<CsfFeaturesView>(`/api/ingest/csf/features?t=${t}&s=${s}&r=${r}`),
};

// ---------------------------------------------------------------------------
// GraphBLAS algebraic layer (`/api/ingest/graphblas/*`, served by the
// `IngestPipeline` view server). These expose the sanctioned
// `abacus_tensor::api::graphblas` boundary (Addendum O rows 30–51) over HTTP:
// tensor-bound `GET` routes operate on the sealed canonical tensor
// (`T_sealed`; call `seal()` first to compact pending commits), while
// `primitive/*` `POST` routes apply stateless GraphBLAS primitives (#1–#13)
// to caller-supplied matrices/vectors.
//
// `value` fields are exact micro-unit tensor values — the computation-consumer
// read boundary, matching `csfApi`. They are intentionally separate from the
// amount-free `ingestApi` dashboard surface.
// ---------------------------------------------------------------------------

/** One `(s, r, value)` sparse-matrix entry. */
export interface GraphblasMatrixEntry {
  s: number;
  r: number;
  value: number;
}

/** A sparse `PairMatrix` payload: cardinality + entries. */
export interface GraphblasMatrix {
  nnz: number;
  entries: GraphblasMatrixEntry[];
}

/** A sparse `Vector` pattern: cardinality + member codes. */
export interface GraphblasVector {
  nnz: number;
  members: number[];
}

/** One `(t, s, r, value)` canonical tensor entry. */
export interface GraphblasTensorEntry {
  t: number;
  s: number;
  r: number;
  value: number;
}

/** Focal parameter bundle shared by focal_context/fan_in. */
export interface GraphblasFocalParams {
  tStart: number;
  tEnd: number;
  k: number;
  epoch: number;
  s: number;
  r: number;
  selfLoop?: "admit" | "exclude";
  pairs?: Array<[number, number]>;
}

/** `GET /api/ingest/graphblas/contract_mode0?t_start=&t_end=` (row 30). */
export interface GraphblasContractMode0 extends GraphblasMatrix {
  t_start: number;
  t_end: number;
}

/** `GET /api/ingest/graphblas/temporal_multiplicity` (row 31). */
export interface GraphblasTemporalMultiplicity extends GraphblasMatrix {
  t_start: number;
  t_end: number;
}

/** `GET /api/ingest/graphblas/support` (#19, row 48). */
export interface GraphblasSupport {
  t_start: number;
  t_end: number;
  a_one: GraphblasMatrix;
  a_cnt: GraphblasMatrix;
  a_vol: GraphblasMatrix;
}

/** `GET /api/ingest/graphblas/prune_time` (#14, row 43). */
export interface GraphblasWindowEntries {
  t_start: number;
  t_end: number;
  entries: GraphblasTensorEntry[];
}

/** `GET /api/ingest/graphblas/hop_ball` (#17, row 46). */
export interface GraphblasHopBall extends GraphblasVector {
  t_start: number;
  t_end: number;
  u: number;
  v: number;
  k: number;
}

/** `GET /api/ingest/graphblas/focal_context` (#20, row 49). */
export interface GraphblasFocalContext {
  epoch: number;
  s: number;
  r: number;
  t_start: number;
  t_end: number;
  k: number;
  admitted: GraphblasTensorEntry[];
  vertices: GraphblasVector;
  a_one: GraphblasMatrix;
  a_cnt: GraphblasMatrix;
  a_vol: GraphblasMatrix;
}

/** `GET /api/ingest/graphblas/fan_in` (#21/#22, row 50). */
export interface GraphblasFanIn {
  epoch: number;
  s: number;
  r: number;
  t_start: number;
  t_end: number;
  k: number;
  fan_in: number;
  support_counterparties: number[];
}

/** `POST /api/ingest/graphblas/seal` (row 22). */
export interface GraphblasSeal {
  sealed_nnz: number;
}

function focalQuery(p: GraphblasFocalParams): string {
  let q =
    `t_start=${p.tStart}&t_end=${p.tEnd}&k=${p.k}` +
    `&epoch=${p.epoch}&s=${p.s}&r=${p.r}`;
  if (p.selfLoop) q += `&self_loop=${p.selfLoop}`;
  for (const [u, r] of p.pairs ?? []) q += `&pair=${u}:${r}`;
  return q;
}

// ---------------------------------------------------------------------------
// Semantic HTTP contract (`/api/ingest/semantic/*`).
//
// External request/response schemas for AI agents and remote consumers.
// These types are intentionally independent of the internal Rust types — the
// adapter converts between them at the HTTP boundary.
// ---------------------------------------------------------------------------

/** `GET /api/ingest/semantic/lookup?t=&s=&r=` or `start_time=` response. */
export interface SemanticLookupResponse {
  op: "lookup";
  t: number;
  /** Human-readable UTC representation of `t`. */
  timestamp_utc: string;
  s: number;
  r: number;
  found: boolean;
  value: number | null;
}

/** `GET /api/ingest/semantic/time_slice?t_start=&t_end=` or `start_time=`/`end_time=` response. */
export interface SemanticTimeSliceResponse {
  op: "time_slice";
  t_start: number;
  /** Human-readable UTC representation of `t_start`. */
  t_start_utc: string;
  t_end: number;
  /** Human-readable UTC representation of `t_end`. */
  t_end_utc: string;
  entries: SemanticTimeSliceEntry[];
}

/** One entry of a semantic time slice. */
export interface SemanticTimeSliceEntry {
  t: number;
  /** Human-readable UTC representation of `t`. */
  timestamp_utc: string;
  s: number;
  r: number;
  value: number;
}

/** `GET /api/ingest/semantic/sender?t=&s=` or `start_time=` response. */
export interface SemanticSenderResponse {
  op: "sender";
  t: number;
  /** Human-readable UTC representation of `t`. */
  timestamp_utc: string;
  s: number;
  entries: SemanticDimensionEntry[];
}

/** `GET /api/ingest/semantic/receiver?t=&r=` or `start_time=` response. */
export interface SemanticReceiverResponse {
  op: "receiver";
  t: number;
  /** Human-readable UTC representation of `t`. */
  timestamp_utc: string;
  r: number;
  entries: SemanticDimensionEntry[];
}

/** One entry of a sender/receiver dimension view. */
export interface SemanticDimensionEntry {
  peer: number;
  /** Human-readable UTC representation of the entry's time coordinate. */
  timestamp_utc: string;
  value: number;
}

/** `GET /api/ingest/semantic/features?t=&s=&r=` or `start_time=` response. */
export interface SemanticFeaturesResponse {
  op: "features";
  t: number;
  /** Human-readable UTC representation of `t`. */
  timestamp_utc: string;
  s: number;
  r: number;
  entries: SemanticFeatureEntry[];
}

/** One derived feature entry. */
export interface SemanticFeatureEntry {
  log_magnitude: number;
  relative_threshold_distance: number;
  roundedness_flags: number;
}

/** `GET /api/ingest/semantic/stats` response. */
export interface SemanticStatsResponse {
  op: "stats";
  nnz: number;
}

/** `GET /api/ingest/semantic/decode` response. */
export interface SemanticDecodeResponse {
  op: "decode";
  entries: SemanticDecodeEntry[];
}

/** One entry of a full decode. */
export interface SemanticDecodeEntry {
  t: number;
  /** Human-readable UTC representation of `t`. */
  timestamp_utc: string;
  s: number;
  r: number;
  value: number;
}

/** Semantic API client — typed calls over `/api/ingest/semantic/*`. */
export const semanticApi = {
  /**
   * `GET /api/ingest/semantic/lookup?t=&s=&r=` — exact coordinate lookup.
   * Accepts either numeric `t` or human-readable `start_time` (ISO 8601).
   */
  lookup: (
    t: number,
    s: number,
    r: number,
    startTime?: string,
  ) => {
    const params = new URLSearchParams({ s: String(s), r: String(r) });
    if (startTime) params.set("start_time", startTime);
    else params.set("t", String(t));
    return getJson<SemanticLookupResponse>(
      `/api/ingest/semantic/lookup?${params.toString()}`,
    );
  },

  /**
   * `GET /api/ingest/semantic/time_slice?t_start=&t_end=` — half-open time window.
   * Accepts either numeric `t_start`/`t_end` or human-readable `start_time`/`end_time` (ISO 8601).
   */
  timeSlice: (
    tStart: number,
    tEnd: number,
    startTime?: string,
    endTime?: string,
  ) => {
    const params = new URLSearchParams();
    if (startTime) params.set("start_time", startTime);
    else params.set("t_start", String(tStart));
    if (endTime) params.set("end_time", endTime);
    else params.set("t_end", String(tEnd));
    return getJson<SemanticTimeSliceResponse>(
      `/api/ingest/semantic/time_slice?${params.toString()}`,
    );
  },

  /**
   * `GET /api/ingest/semantic/sender?t=&s=` — outgoing relationships.
   * Accepts either numeric `t` or human-readable `start_time` (ISO 8601).
   */
  sender: (
    t: number,
    s: number,
    startTime?: string,
  ) => {
    const params = new URLSearchParams({ s: String(s) });
    if (startTime) params.set("start_time", startTime);
    else params.set("t", String(t));
    return getJson<SemanticSenderResponse>(
      `/api/ingest/semantic/sender?${params.toString()}`,
    );
  },

  /**
   * `GET /api/ingest/semantic/receiver?t=&r=` — incoming relationships.
   * Accepts either numeric `t` or human-readable `start_time` (ISO 8601).
   */
  receiver: (
    t: number,
    r: number,
    startTime?: string,
  ) => {
    const params = new URLSearchParams({ r: String(r) });
    if (startTime) params.set("start_time", startTime);
    else params.set("t", String(t));
    return getJson<SemanticReceiverResponse>(
      `/api/ingest/semantic/receiver?${params.toString()}`,
    );
  },

  /**
   * `GET /api/ingest/semantic/features?t=&s=&r=` — derived feature metrics.
   * Accepts either numeric `t` or human-readable `start_time` (ISO 8601).
   */
  features: (
    t: number,
    s: number,
    r: number,
    startTime?: string,
  ) => {
    const params = new URLSearchParams({ s: String(s), r: String(r) });
    if (startTime) params.set("start_time", startTime);
    else params.set("t", String(t));
    return getJson<SemanticFeaturesResponse>(
      `/api/ingest/semantic/features?${params.toString()}`,
    );
  },

  /** `GET /api/ingest/semantic/stats` — live nonzero count. */
  stats: () => getJson<SemanticStatsResponse>("/api/ingest/semantic/stats"),

  /** `GET /api/ingest/semantic/decode` — full live decode (explicitly authorized). */
  decode: () => getJson<SemanticDecodeResponse>("/api/ingest/semantic/decode"),
};

// ---------------------------------------------------------------------------
// Phase-3 temporal retrieval contract (`/api/ingest/retrieval/*`).
//
// Restores the CSR (sender-oriented) / CSC (receiver-oriented) look-back and
// look-ahead queries removed in cutover commit `88fac5c`, fully parameterized:
//
// - `entity` — account identifier string (wire ID as a decimal string).
// - `anchor` — anchor timestamp (epoch seconds).
// - `delta` — window length in seconds; defaults to 7 days (604800)
//   when omitted. Not a hard-coded semantic rule — fully configurable.
// - `maxEdges` — maximum locators per directional lookup; hard cap is 2,500.
// ---------------------------------------------------------------------------

/** Seven-day default window in seconds (`DEFAULT_DELTA_SECS`). */
export const DEFAULT_DELTA_SECS = 7 * 24 * 60 * 60;

/** Hard cap on locators returned by any single directional lookup. */
export const MAX_EDGES_PER_LOOKUP = 2_500;

/** `GET /api/ingest/retrieval/incoming|outgoing|combined` response. */
export interface RetrievalDirectionalResponse {
  op: "incoming" | "outgoing" | "combined";
  /** Matching locators, deterministic order (incoming: newest-first). */
  locators: number[];
  /** True iff qualifying events exist beyond the returned locators. */
  is_truncated: boolean;
  /** Exact qualifying count, or null when intentionally not computed. */
  total_matching_events: number | null;
}

/** `GET /api/ingest/retrieval/range?t_start=&t_end=` response. */
export interface RetrievalRangeResponse {
  op: "range";
  /** Every event locator in `[t_start, t_end)`, uncapped, `(t, locator)` ascending. */
  locators: number[];
  count: number;
}

/** `GET /api/ingest/retrieval/resolve?locator=` response. */
export interface RetrievalResolveResponse {
  op: "resolve";
  locator: number;
  t: number;
  /** Wire IDs (external vocabulary), not internal numeric codes. */
  sender: number;
  receiver: number;
}

/** `GET /api/ingest/retrieval/check` response. */
export interface RetrievalCheckResponse {
  op: "check";
  status: "consistent" | "inconsistent";
  message?: string;
}

/** Optional retrieval query parameters shared by all directional routes. */
export interface RetrievalQueryOptions {
  /** Window length in seconds; defaults to 7 days. */
  delta?: number;
  /** Maximum locators per lookup; hard cap is 2,500. */
  maxEdges?: number;
}

function retrievalParams(
  entity: string,
  anchor: number,
  options?: RetrievalQueryOptions,
): string {
  const params = new URLSearchParams({ entity, anchor: String(anchor) });
  if (options?.delta !== undefined) params.set("delta", String(options.delta));
  if (options?.maxEdges !== undefined)
    params.set("max_edges", String(options.maxEdges));
  return params.toString();
}

/** Retrieval API client — typed calls over `/api/ingest/retrieval/*`. */
export const retrievalApi = {
  /**
   * `GET /api/ingest/retrieval/incoming?entity=&anchor=&delta=&max_edges=` —
   * upstream look-back: incoming events to `entity` in `[anchor - delta, anchor)`,
   * newest-first, most-recent wins on truncation.
   */
  incoming: (
    entity: string,
    anchor: number,
    options?: RetrievalQueryOptions,
  ) =>
    getJson<RetrievalDirectionalResponse>(
      `/api/ingest/retrieval/incoming?${retrievalParams(entity, anchor, options)}`,
    ),

  /**
   * `GET /api/ingest/retrieval/outgoing?entity=&anchor=&delta=&max_edges=` —
   * downstream look-ahead: outgoing events from `entity` in
   * `[anchor, anchor + delta)`, earliest-first.
   */
  outgoing: (
    entity: string,
    anchor: number,
    options?: RetrievalQueryOptions,
  ) =>
    getJson<RetrievalDirectionalResponse>(
      `/api/ingest/retrieval/outgoing?${retrievalParams(entity, anchor, options)}`,
    ),

  /**
   * `GET /api/ingest/retrieval/combined?entity=&anchor=&delta=&max_edges=` —
   * bidirectional: each original event in `[anchor - delta, anchor + delta)`
   * exactly once, `(t, locator)` ascending.
   */
  combined: (
    entity: string,
    anchor: number,
    options?: RetrievalQueryOptions,
  ) =>
    getJson<RetrievalDirectionalResponse>(
      `/api/ingest/retrieval/combined?${retrievalParams(entity, anchor, options)}`,
    ),

  /**
   * `GET /api/ingest/retrieval/range?t_start=&t_end=` — uncapped global
   * temporal retrieval in `[t_start, t_end)`.
   */
  range: (tStart: number, tEnd: number) =>
    getJson<RetrievalRangeResponse>(
      `/api/ingest/retrieval/range?t_start=${tStart}&t_end=${tEnd}`,
    ),

  /** `GET /api/ingest/retrieval/resolve?locator=` — locator to `(t, sender, receiver)`. */
  resolve: (locator: number) =>
    getJson<RetrievalResolveResponse>(
      `/api/ingest/retrieval/resolve?locator=${locator}`,
    ),

  /** `GET /api/ingest/retrieval/check` — structural tensor/index consistency. */
  check: () => getJson<RetrievalCheckResponse>("/api/ingest/retrieval/check"),
};

export const graphblasApi = {
  /** `GET /api/ingest/graphblas/contract_mode0?t_start=&t_end=` (row 30). */
  contractMode0: (tStart: number, tEnd: number) =>
    getJson<GraphblasContractMode0>(
      `/api/ingest/graphblas/contract_mode0?t_start=${tStart}&t_end=${tEnd}`,
    ),
  /** `GET /api/ingest/graphblas/temporal_multiplicity` (row 31). */
  temporalMultiplicity: (tStart: number, tEnd: number) =>
    getJson<GraphblasTemporalMultiplicity>(
      `/api/ingest/graphblas/temporal_multiplicity?t_start=${tStart}&t_end=${tEnd}`,
    ),
  /** `GET /api/ingest/graphblas/support` (#19, row 48). */
  support: (tStart: number, tEnd: number) =>
    getJson<GraphblasSupport>(
      `/api/ingest/graphblas/support?t_start=${tStart}&t_end=${tEnd}`,
    ),
  /** `GET /api/ingest/graphblas/prune_time` (#14, row 43). */
  pruneTime: (tStart: number, tEnd: number) =>
    getJson<GraphblasWindowEntries>(
      `/api/ingest/graphblas/prune_time?t_start=${tStart}&t_end=${tEnd}`,
    ),
  /** `GET /api/ingest/graphblas/hop_ball` (#17, row 46). */
  hopBall: (tStart: number, tEnd: number, u: number, v: number, k: number) =>
    getJson<GraphblasHopBall>(
      `/api/ingest/graphblas/hop_ball?t_start=${tStart}&t_end=${tEnd}&u=${u}&v=${v}&k=${k}`,
    ),
  /** `GET /api/ingest/graphblas/prune_vertices` (#15, row 44). */
  pruneVertices: (
    tStart: number,
    tEnd: number,
    u: number,
    v: number,
    k: number,
  ) =>
    getJson<GraphblasWindowEntries>(
      `/api/ingest/graphblas/prune_vertices?t_start=${tStart}&t_end=${tEnd}&u=${u}&v=${v}&k=${k}`,
    ),
  /** `GET /api/ingest/graphblas/prune_edges` (#16, row 45). */
  pruneEdges: (tStart: number, tEnd: number, pairs: Array<[number, number]>) =>
    getJson<GraphblasWindowEntries>(
      `/api/ingest/graphblas/prune_edges?t_start=${tStart}&t_end=${tEnd}` +
        pairs.map(([u, r]) => `&pair=${u}:${r}`).join(""),
    ),
  /** `GET /api/ingest/graphblas/prune_hops` (#18, row 47). */
  pruneHops: (tStart: number, tEnd: number, u: number, v: number, k: number) =>
    getJson<GraphblasWindowEntries>(
      `/api/ingest/graphblas/prune_hops?t_start=${tStart}&t_end=${tEnd}&u=${u}&v=${v}&k=${k}`,
    ),
  /** `GET /api/ingest/graphblas/focal_context` (#20, row 49). */
  focalContext: (p: GraphblasFocalParams) =>
    getJson<GraphblasFocalContext>(
      `/api/ingest/graphblas/focal_context?${focalQuery(p)}`,
    ),
  /** `GET /api/ingest/graphblas/fan_in` (#21/#22, row 50). */
  fanIn: (p: GraphblasFocalParams) =>
    getJson<GraphblasFanIn>(`/api/ingest/graphblas/fan_in?${focalQuery(p)}`),
  /** `POST /api/ingest/graphblas/seal` (row 22). */
  seal: () => postJson<GraphblasSeal>("/api/ingest/graphblas/seal"),
  /** `POST /api/ingest/graphblas/primitive/extract` (#1, row 32). */
  extract: (entries: GraphblasMatrixEntry[], s: number, r: number) =>
    postJson<{ found: boolean; value: number | null }>(
      "/api/ingest/graphblas/primitive/extract",
      { entries, s, r },
    ),
  /** `POST /api/ingest/graphblas/primitive/reduce_col_plus` (#2, row 33). */
  reduceColPlus: (entries: GraphblasMatrixEntry[], v: number) =>
    postJson<{ sum: number }>(
      "/api/ingest/graphblas/primitive/reduce_col_plus",
      { entries, v },
    ),
  /** `POST /api/ingest/graphblas/primitive/reduce_row_plus` (#3, row 34). */
  reduceRowPlus: (entries: GraphblasMatrixEntry[], u: number) =>
    postJson<{ sum: number }>(
      "/api/ingest/graphblas/primitive/reduce_row_plus",
      { entries, u },
    ),
  /** `POST /api/ingest/graphblas/primitive/mxv` (#4, row 35). */
  mxv: (entries: GraphblasMatrixEntry[], frontier: number[]) =>
    postJson<GraphblasVector>("/api/ingest/graphblas/primitive/mxv", {
      entries,
      frontier,
    }),
  /** `POST /api/ingest/graphblas/primitive/vxm` (#5, row 36). */
  vxm: (entries: GraphblasMatrixEntry[], frontier: number[]) =>
    postJson<GraphblasVector>("/api/ingest/graphblas/primitive/vxm", {
      entries,
      frontier,
    }),
  /** `POST /api/ingest/graphblas/primitive/ewise_add_lor` (#6, row 37). */
  ewiseAddLor: (a: number[], b: number[]) =>
    postJson<GraphblasVector>(
      "/api/ingest/graphblas/primitive/ewise_add_lor",
      { a, b },
    ),
  /** `POST /api/ingest/graphblas/primitive/support` (#7, row 38). */
  supportMatrix: (entries: GraphblasMatrixEntry[]) =>
    postJson<GraphblasMatrix>("/api/ingest/graphblas/primitive/support", {
      entries,
    }),
  /** `POST /api/ingest/graphblas/primitive/multiplicity` (#8, row 39). */
  multiplicity: (pairs: Array<{ s: number; r: number }>) =>
    postJson<GraphblasMatrix>(
      "/api/ingest/graphblas/primitive/multiplicity",
      { pairs },
    ),
  /** `POST /api/ingest/graphblas/primitive/volume` (#9, row 40). */
  volume: (occurrences: GraphblasMatrixEntry[]) =>
    postJson<GraphblasMatrix>("/api/ingest/graphblas/primitive/volume", {
      occurrences,
    }),
  /** `POST /api/ingest/graphblas/primitive/reduce_count` (#10, row 41). */
  reduceCount: (members: number[]) =>
    postJson<{ count: number }>(
      "/api/ingest/graphblas/primitive/reduce_count",
      { members },
    ),
  /** `POST /api/ingest/graphblas/primitive/reduce_count_set` (#11, row 41). */
  reduceCountSet: (members: number[]) =>
    postJson<{ count: number }>(
      "/api/ingest/graphblas/primitive/reduce_count_set",
      { members },
    ),
  /** `POST /api/ingest/graphblas/primitive/reduce_sum` (#12, row 42). */
  reduceSum: (values: number[]) =>
    postJson<{ sum: number }>(
      "/api/ingest/graphblas/primitive/reduce_sum",
      { values },
    ),
  /** `POST /api/ingest/graphblas/primitive/reduce_max` (#13, row 42). */
  reduceMax: (values: number[]) =>
    postJson<{ max: number }>(
      "/api/ingest/graphblas/primitive/reduce_max",
      { values },
    ),
};

