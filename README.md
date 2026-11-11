# Abacus

Abacus is a Rust-based ingestion and sparse-tensor analytics prototype for transaction data. Its browser dashboard accepts CSV uploads and lets you inspect processed transactions, account activity, and transaction/account network views.

The repository includes Rust crates for the tensor substrate, ingestion pipeline, CSV handling, and Kafka integration. The Docker Compose demo runs the HTTP ingestion API and dashboard; it does not start a Kafka broker.

## Run the demo

Requirements: [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine with the Compose plugin.

```bash
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000). The demo includes:

- The ingestion dashboard at `/`
- The ingestion view at `/ingest-view`
- The CSV upload test panel at `/ingest-test`

The backend is also available on port `8091`:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/ingest/health` | Backend health |
| `GET /api/ingest/status` | Ingestion status and counts |
| `POST /api/ingest/csv` | Upload a CSV file |
| `GET /api/ingest/transactions` | Read processed transactions |
| `GET /api/ingest/networks` | Read transaction/account network data |

Stop the demo with `Ctrl-C`, or run `docker compose down` from another terminal.

## Optional demo dataset

The repository includes a gzip-compressed 90-day CSV sample at `demo-data/SAML-D-90days_1st.csv.gz` (about 52 MiB). Docker Compose does not load it automatically. To unpack it locally while keeping the archive:

```bash
gzip -dk demo-data/SAML-D-90days_1st.csv.gz
```

Upload the resulting CSV from the dashboard. The uncompressed file is about 262 MiB, so allow enough memory for the browser and backend while importing it.

The archive exceeds GitHub's 50 MiB warning threshold but is below its 100 MiB per-file limit. Before publishing, verify that you have permission to redistribute this dataset and that it contains no data that should remain private. The repository's MIT license applies to the project code, not automatically to this dataset.

## How it works

- **Rust backend:** validates and processes CSV uploads and serves ingestion status, transaction, and network APIs on port `8091`.
- **Next.js frontend:** serves the dashboard on port `3000` and proxies browser API requests to the backend over the Compose network.
- **Rust workspace:** contains the tensor substrate and related ingestion, CSV, and Kafka client crates.

## Current limitations

- **Volatile data:** the Compose backend keeps state in memory. Restarting or recreating the backend clears uploaded records.
- **Local prototype:** the demo has no authentication or durable storage. Keep it on a trusted local machine; do not expose it directly to the public internet or upload sensitive data.
- **Separate upload API:** `/api/v1/*` belongs to a separate upload service and is not included in this Compose stack.

## Development

Run the Rust workspace tests:

```bash
cargo test --workspace
```

Build the frontend locally:

```bash
npm ci --prefix web
npm run build --prefix web
```

## License

This project is licensed under the MIT License; see [LICENSE](./LICENSE).
