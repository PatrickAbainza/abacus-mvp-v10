//! `abacus_csv_client` — the browser-side CSV normalization client.
//!
//! Golden Source v2.0 §4.1/§12 define what the browser half of the pipeline must
//! do: read the selected CSV locally, inspect its header and a bounded sample,
//! map source columns to canonical fields, collect the required normalization
//! settings, parse/validate/normalize rows, assign opaque numeric account IDs in
//! one shared sender/receiver namespace, derive the frozen seven-byte payload,
//! encode bounded binary chunks, upload them with session/sequence/length/
//! checksum metadata, and display reconciled acceptance and rejection counts.
//!
//! ## What this crate is, and what it deliberately is not
//!
//! Everything that must be **identical** between the browser and native Rust is
//! *reused* from [`abacus_ingest`], never reimplemented here:
//!
//! | Concern | Owner | Section |
//! |---------|-------|---------|
//! | Frozen 7-byte feature derivation | `abacus_ingest::features::FeaturePayload::derive` | §9 |
//! | 23+2W canonical record layout | `abacus_ingest::record::CanonicalRecord` | Contract §7 |
//! | 176-byte `ABACSEG1` envelope + checksums | `abacus_ingest::wire` / `encode_segment` | wire §1/§4 |
//! | Event identity `(namespace, u64)` | `abacus_ingest::namespace` | Contract §3 |
//! | Structured error taxonomy | `abacus_ingest::error::ErrorCode` | §16 |
//! | Attested-amount clearing guards | `abacus_ingest::secret` | A1 §21.1 |
//!
//! §9.1 requires that "JavaScript `Math.log` and Rust `ln` must not be assumed to
//! produce identical quantized results on every platform" and that the
//! implementation "prefer a shared deterministic implementation". That shared
//! implementation already exists in the pipeline crate and evaluates the
//! logarithm in exact integer arithmetic, so this client calls it instead of
//! writing a second derivation. Golden vectors therefore match byte-for-byte by
//! construction rather than by agreement between two implementations.
//!
//! ## The browser binding lives next door, not here
//!
//! The repository rule for this workspace is "never use Python, always use Rust
//! or C". This crate is the **pure Rust core** a browser compiles: no I/O of
//! its own, no clock, no randomness, no filesystem access — the selected file's
//! bytes and an explicit policy in, a privacy-safe preview and bounded encoded
//! chunks out. The browser/WASM shell that binds this core to the DOM (file
//! picker, configuration form, `fetch`) is the separate `abacus_shell` crate,
//! which owns the synchronous host-fetch import; this crate only exposes the
//! [`Transport`] seam that binding implements. Normalization semantics cannot
//! drift across the boundary because the same code produces the bytes on both
//! targets, and [`crate::bin`]'s `csv_ingest` driver still runs the identical
//! core headlessly against the real `/api/v1` server.
//!
//! ## Privacy boundary
//!
//! Raw account strings never leave this process: identifiers are folded to
//! opaque `u64` codes before anything is encoded, and the raw-to-code map is
//! never serialized, never uploaded and never logged (§6: "Do not upload the raw
//! account-to-numeric-ID mapping from browser CSV ingestion"). Raw amounts are
//! held only for the A1 commit-time attestation channel, inside
//! [`abacus_ingest::secret`] guards that zero on drop, and appear in no record,
//! no report, no log and no error detail. User-facing errors identify a problem
//! row by its source line number only (§12: "expose actionable errors without
//! displaying raw account strings or raw amounts").

#![warn(missing_docs)]

pub mod amount;
pub mod attest;
pub mod client;
pub mod csv;
pub mod identifiers;
pub mod mapping;
pub mod normalize;
pub mod policy;
pub mod resume;
pub mod timestamp;

/// The headless driver for the browser client core.
///
/// Split out so `src/bin/csv_ingest.rs` stays a thin `main`, matching the
/// repository's `ingest_server` convention.
pub mod driver;

pub use amount::parse_amount;
pub use client::{
    ApiResponse, Capabilities, ChunkAck, CommitAck, CsvIngestClient, HttpError, Transport,
};
pub use csv::{CsvOptions, CsvRow, CsvTable};
pub use identifiers::{IdentifierInterner, IdentifierPolicy};
pub use mapping::{field_at, suggest_mapping, CanonicalField, ColumnMapping};
pub use normalize::{normalize, Normalization, Preview, RowIssue};
pub use policy::{
    AmountFormat, EpochUnit, IngestPlan, NumberPolicy, SubSecondPolicy, TimestampPolicy,
    TimestampStyle,
};
pub use resume::ResumeToken;

/// The 128-bit event namespace, re-exported so a browser binding never has to
/// reach past this crate's public surface to build a plan.
pub use abacus_ingest::namespace::EventNamespace;
