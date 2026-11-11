import { NextRequest, NextResponse } from "next/server";
import { routeOrigin } from "../../../lib/route";

/**
 * Same-origin API proxy: forwards `/api/*` requests to the Rust backends.
 *
 * Origin split (recommendation Layer 3 / option 3a — reconciliation §E.4.2,
 * updated by the CSV/Kafka ingestion cutover). After the cutover two
 * ingestion services back the UI, on separate ports, plus a legacy fallback:
 *
 *   /api/v1/*                                     -> UPLOAD_ORIGIN
 *       (IngestionService upload server, e.g. :8090 — session/chunk/commit)
 *   /api/ingest/*  (csv upload, health, status, transactions, networks)
 *                                      -> INGEST_ORIGIN  (IngestPipeline view
 *       server, e.g. :8091 — shared Account Map + derived feature metrics;
 *       owns the single write+read path, in-memory only)
 *   everything else                   -> BACKEND_ORIGIN (legacy, e.g. :8080)
 *
 * `POST /api/ingest/csv` MUST route to the view server: it applies the upload
 * to the same `IngestPipeline` the dashboard reads. Routing it to the upload
 * server instead reproduces the upload/dashboard split (dashboard empty).
 *
 * Fallback layering keeps every earlier deployment mode working:
 *
 *   UPLOAD_ORIGIN  || BACKEND_ORIGIN || http://127.0.0.1:8090
 *   INGEST_ORIGIN  || BACKEND_ORIGIN || http://127.0.0.1:8091
 *   BACKEND_ORIGIN || http://127.0.0.1:8080
 *
 * so a deployment that sets only `BACKEND_ORIGIN` (the old single-origin
 * mode, e.g. BACKEND_ORIGIN=:8090 against the upload server) routes exactly
 * as before, while a deployment that sets all three reaches both servers.
 *
 * Method, query string, request body and the upstream status/headers/body are
 * all relayed verbatim, keeping the browser strictly same-origin (no CORS)
 * while letting backend addresses be configured outside the build.
 */
export const dynamic = "force-dynamic";

function originOf(envVar: string, fallback: string): string {
  return (process.env[envVar] || fallback).replace(/\/$/, "");
}

/** Upload server (`IngestionService`): `/api/v1/*` only. `/api/ingest/*` (csv
 * upload, health, status, transactions, networks) belongs to the view server. */
function uploadOrigin(): string {
  return originOf("UPLOAD_ORIGIN", originOf("BACKEND_ORIGIN", "http://127.0.0.1:8090"));
}

/** Pipeline view server (`IngestPipeline`): `/api/ingest/*` (read surface). */
function ingestOrigin(): string {
  return originOf("INGEST_ORIGIN", originOf("BACKEND_ORIGIN", "http://127.0.0.1:8091"));
}

/** Legacy fallback for non-ingestion paths (no backend currently serves them). */
function backendOrigin(): string {
  return originOf("BACKEND_ORIGIN", "http://127.0.0.1:8080");
}

async function proxy(req: NextRequest, params: { path?: string[] }) {
  const segs = params.path ?? [];
  const apiPath = `/api/${segs.join("/")}`;
  const base = ((): string => {
    switch (routeOrigin(apiPath)) {
      case "upload":
        return uploadOrigin();
      case "ingest":
        return ingestOrigin();
      default:
        return backendOrigin();
    }
  })();
  const target = `${base}/api/${segs.join("/")}${req.nextUrl.search}`;
  const method = req.method;

  // Relay the body for methods that carry one; otherwise send an empty request.
  const hasBody = method !== "GET" && method !== "HEAD";
  const init: RequestInit & { duplex?: string } = { method, headers: passthrough(req) };
  if (hasBody) {
    init.body = Buffer.from(await req.arrayBuffer());
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch (e) {
    return NextResponse.json(
      { error: `upstream unreachable: ${e instanceof Error ? e.message : String(e)}` },
      { status: 502 },
    );
  }

  const headers = new Headers();
  const ct = upstream.headers.get("content-type");
  if (ct) headers.set("content-type", ct);
  // Pass through the upstream status so 400/404 surface faithfully to the UI.
  return new NextResponse(upstream.body, { status: upstream.status, headers });
}

/** Copy relevant request headers, dropping hop-by-hop / Next-internal ones. */
function passthrough(req: NextRequest): Headers {
  const out = new Headers();
  req.headers.forEach((value, key) => {
    const k = key.toLowerCase();
    if (k === "host" || k === "connection" || k.startsWith("x-next") || k === "content-length") {
      return;
    }
    out.set(key, value);
  });
  return out;
}

export async function GET(req: NextRequest, ctx: { params: { path?: string[] } }) {
  return proxy(req, ctx.params);
}
export async function POST(req: NextRequest, ctx: { params: { path?: string[] } }) {
  return proxy(req, ctx.params);
}
export async function PUT(req: NextRequest, ctx: { params: { path?: string[] } }) {
  return proxy(req, ctx.params);
}
export async function DELETE(req: NextRequest, ctx: { params: { path?: string[] } }) {
  return proxy(req, ctx.params);
}
export async function PATCH(req: NextRequest, ctx: { params: { path?: string[] } }) {
  return proxy(req, ctx.params);
}
