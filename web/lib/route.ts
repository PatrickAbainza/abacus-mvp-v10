// Origin routing for the same-origin `/api/*` proxy (recommendation Layer 3 /
// option 3a — reconciliation §E.4.2). After the CSV/Kafka ingestion cutover the
// web origin reaches up to three backends by path prefix:
//
//   /api/v1/*                                        -> UPLOAD_ORIGIN (:8090,
//       the oracle `IngestionService` upload server — session/chunk/commit API)
//   /api/ingest/csv, /api/ingest/health, /api/ingest/*
//       (status, transactions, networks + the CSV upload)
//                                                     -> INGEST_ORIGIN (:8091,
//       the `IngestPipeline` view server — shared Account Map + derived
//       feature metrics; owns the single write+read path, in-memory only)
//   everything else                                -> BACKEND_ORIGIN (:8080,
//       legacy fallback; no route is served by the current backend)
//
// `POST /api/ingest/csv` routes to the view server (not the upload server) so
// the uploaded rows land in the SAME `IngestPipeline` the dashboard reads —
// otherwise the dashboard stays empty forever (upload/dashboard split). The
// upload server still serves an identical `/api/ingest/csv` for direct
// curl/localhost use; the browser simply never addresses it.
//
// Kept in `lib/` (not the route file) because Next.js only permits route
// handlers/config as named exports of a `route.ts`.

/** Path prefixes owned by the upload server (IngestionService). */
const UPLOAD_PREFIXES = ["/api/v1/"];

/** Which backend owns a given API path (pure — unit-testable). */
export function routeOrigin(pathname: string): "upload" | "ingest" | "backend" {
  if (UPLOAD_PREFIXES.some((p) => pathname.startsWith(p))) return "upload";
  if (pathname.startsWith("/api/ingest/")) return "ingest";
  return "backend";
}

