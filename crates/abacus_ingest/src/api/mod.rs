//! The versioned ingestion API (`/api/v1`).
//!
//! - [`service`] — the reusable Rust service. All rules live here (task step 3:
//!   "Keep transport handlers thin").
//! - [`server`] — the std-only HTTP/1.1 transport, following the repository's
//!   existing `web::server` conventions. Thin by construction.
//! - [`json`] — the minimal JSON model. No serialization dependency.
//!
//! Privacy boundary (task step 3 + Golden Source §16): no route exposes raw
//! account mappings, raw transaction amounts, or ground-truth labels. Reports
//! carry counts, versions, fingerprints, opaque session/namespace identifiers
//! and per-record ordinals only.

pub mod json;
pub mod server;
pub mod service;

pub use server::{handle_one, route, serve, Request, Response};
pub use service::{Capabilities, CreateSessionRequest, CreateSessionResponse, IngestionService};
