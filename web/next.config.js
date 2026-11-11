// @ts-check
/**
 * Next.js config for the Abacus SAMLD StreamingIngestionDemo UI.
 *
 * The frontend talks ONLY to same-origin `/api/*`. Those requests are proxied
 * to the Rust streaming_ingestion_demo server by a dynamic Route Handler
 * (`app/api/[...path]/route.ts`) which reads `BACKEND_ORIGIN` at request time
 * (NOT a static `rewrites()` — those are baked at build time and ignore
 * runtime env). Docker Compose sets `BACKEND_ORIGIN=http://backend:8080`.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;

