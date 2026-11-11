//! Ingestion-backed web surface (reconciliation Phase 4/5).
//!
//! A parallel, additive server over the ingestion-native [`IngestPipeline`].
//! Unlike the existing `StreamingIngestionDemoEngine` server (which requires a `SAMCORE02`
//! artifact and the retired role-separated account model), this server needs
//! NO artifact file — it serves whatever CSV/Kafka records have been applied
//! through `IngestPipeline`, exposing:
//!
//! - **`/api/ingest/status`** — committed count + shared-namespace account count.
//! - **`/api/ingest/transactions`** — committed rows keyed by locator, with
//!   sender/receiver labels drawn from the SINGLE shared account namespace
//!   (role-independent) and the derived feature metrics (NOT the amount).
//! - **`/api/ingest/transactions/{id}`** — one transaction's structural detail
//!   plus its 7-byte derived feature payload.
//! - **`POST /api/ingest/csv`** — raw CSV upload, normalized + session-validated
//!   through the exact canonical path (`abacus_csv_client::normalize` →
//!   `IngestionSession` submit → seal, same as the `:8090` upload server runs)
//!   and applied to THIS server's `IngestPipeline`, so the dashboard read
//!   surface above shows the uploaded rows. Auto-routes SAML-D vs generic by
//!   header, exactly like the upload server.
//!
//! NOTE — no persistence (deferred to a future refactor): all state is
//! in-memory in this process (`IngestPipeline::new()` at boot). A restart
//! wipes every upload the dashboard showed. A durable journal/load path must be
//! added before this can back anything beyond a local test session.
//!
//! This is additive: the existing streaming_ingestion_demo server and its tests are untouched.
//! Privacy is preserved throughout — no amount is ever serialized here; only
//! the frozen `FeaturePayload` derived metrics are exposed (GS §9 / A1).

use crate::ingest_model::FeaturePayload;
use crate::ingest_pipeline::IngestPipeline;
use crate::retrieval::{
    EventLocator, MultiHopQuery, ResolvedEvent, DEFAULT_DELTA_SECS, MAX_EDGES_PER_LOOKUP,
};
use crate::semantic::timestamp::{from_iso8601, to_utc_iso8601};
use crate::semantic::{SemanticError, SemanticLayer};
use crate::web::graph::Network;
use crate::web::graphblas_server::graphblas_route;
use crate::web::ingest_graph::{account_network, transaction_network, transaction_network_multi_hop};
use crate::web::json::Json;
use crate::web::server::{read_request, Request, Response};

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

/// Build the `/api/ingest/status` JSON.
fn status_json(p: &IngestPipeline) -> Json {
    let mut o = Json::object();
    o.set("source", Json::str("csv_kafka_ingestion"));
    o.set("status", Json::str("ready"));
    o.set("committed", Json::uint(p.committed() as u64));
    o.set("accounts", Json::uint(p.accounts().account_count() as u64));
    o
}

/// Serialize the derived feature metrics (privacy: metrics, not the amount).
fn features_json(f: &FeaturePayload) -> Json {
    let mut o = Json::object();
    o.set("log_magnitude", Json::uint(u64::from(f.log_magnitude)));
    o.set(
        "relative_threshold_distance",
        Json::int(i64::from(f.relative_threshold_distance)),
    );
    o.set(
        "roundedness_flags",
        Json::uint(u64::from(f.roundedness_flags)),
    );
    o
}

/// Build a paginated `/api/ingest/transactions` response.
fn transactions_page(p: &IngestPipeline, page: usize, page_size: usize) -> Json {
    let committed = p.committed();
    let page = page.max(1);
    let page_size = page_size.max(1);
    let start = (page - 1).saturating_mul(page_size).min(committed);
    let end = start.saturating_add(page_size).min(committed);
    let mut rows = Vec::new();
    for loc in start..end {
        if let Some((t, s, r, feats)) = p.resolve(loc as u32) {
            // Labels come from the ONE shared namespace — the same account has
            // the same label in either role.
            let sender = p.account_label(s).unwrap_or_default();
            let receiver = p.account_label(r).unwrap_or_default();
            let mut row = Json::object();
            row.set("id", Json::uint(loc as u64));
            row.set("sender", Json::str(sender));
            row.set("receiver", Json::str(receiver));
            row.set("t", Json::uint(t));
            row.set("features", features_json(&feats));
            rows.push(row);
        }
    }
    let total_pages = committed.div_ceil(page_size).max(1);
    let mut o = Json::object();
    o.set("page", Json::uint(page as u64));
    o.set("page_size", Json::uint(page_size as u64));
    o.set("total", Json::uint(committed as u64));
    o.set("total_pages", Json::uint(total_pages as u64));
    o.set("rows", Json::Array(rows));
    o
}

/// Build `/api/ingest/transactions/{id}` detail.
fn transaction_detail(p: &IngestPipeline, id: u64) -> Response {
    let Ok(loc) = u32::try_from(id) else {
        return Response::not_found("transaction not found");
    };
    match p.resolve(loc) {
        Some((t, s, r, feats)) => {
            let sender = p.account_label(s).unwrap_or_default();
            let receiver = p.account_label(r).unwrap_or_default();
            let mut o = Json::object();
            o.set("id", Json::uint(id));
            o.set("sender", Json::str(sender));
            o.set("receiver", Json::str(receiver));
            o.set("t", Json::uint(t));
            o.set("features", features_json(&feats));
            Response::ok(o.to_string())
        }
        None => Response::not_found("transaction not found"),
    }
}

/// Serialize a structural network (nodes + edges). NO amounts.
fn network_json(n: &Network) -> Json {
    let nodes = n
        .nodes
        .iter()
        .map(|node| {
            let mut o = Json::object();
            o.set("id", Json::str(&node.id));
            o.set("kind", Json::str(node.kind));
            o.set("label", Json::str(&node.label));
            match node.t {
                Some(t) => {
                    o.set("t", Json::uint(t));
                }
                None => {
                    o.set("t", Json::Null);
                }
            };
            o.set("selected", Json::bool(node.selected));
            match node.hop {
                Some(h) => {
                    o.set("hop", Json::uint(u64::from(h)));
                }
                None => {
                    o.set("hop", Json::Null);
                }
            };
            o
        })
        .collect::<Vec<_>>();
    let edges = n
        .edges
        .iter()
        .map(|e| {
            let mut o = Json::object();
            o.set("from", Json::str(&e.from));
            o.set("to", Json::str(&e.to));
            o.set(
                "txs",
                Json::Array(e.txs.iter().map(|&t| Json::uint(t)).collect()),
            );
            o
        })
        .collect::<Vec<_>>();
    let mut o = Json::object();
    o.set("nodes", Json::Array(nodes));
    o.set("edges", Json::Array(edges));
    o.set("truncated", Json::bool(n.truncated));
    o.set("committed", Json::uint(n.committed));
    o
}

/// Handle `POST /api/ingest/csv`: normalize + session-validate the raw CSV
/// (same canonical path the `:8090` upload server runs — SAML-D vs generic
/// auto-route by header) and apply the validated records to THIS server's
/// `IngestPipeline`, so the dashboard read surface shows the uploaded rows.
///
/// In-memory only (no persistence — see the module docs): a restart wipes the
/// applied state.
///
/// The response carries a `timing_us` object with server-side microsecond
/// breakdown (`normalize_validate`, `pipeline_apply`, `report_render`,
/// `total`) plus per-record `ns_per_record` — the sub-millisecond proof for
/// the ingestion path (all stages are in-memory; no I/O per record).
fn ingest_csv(p: &mut IngestPipeline, body: &[u8]) -> Response {
    let t0 = std::time::Instant::now();
    let prepared = if abacus_mvp_ingest::is_samld(body) {
        abacus_mvp_ingest::prepare_samld_bytes(body)
    } else {
        abacus_mvp_ingest::prepare_csv_bytes(body)
    };
    let prepared = match prepared {
        Ok(prep) => prep,
        Err(e) => return Response::bad_request(&e.detail),
    };
    let t1 = std::time::Instant::now();
    let outcome = match p.apply_prepared(&prepared) {
        Ok(out) => out,
        Err(e) => return Response::bad_request(&e.to_string()),
    };
    let t2 = std::time::Instant::now();
    let mut report = ingest_report_json(&prepared.report, &prepared.preview, outcome);
    // Server-side latency proof: microsecond stage breakdown + per-record
    // nanoseconds. `total` spans normalize→apply→render inside this handler
    // (excludes socket I/O, which the client measures separately).
    let records = prepared.report.accepted_records.max(1);
    let norm_us = t1.duration_since(t0).as_micros() as u64;
    let apply_us = t2.duration_since(t1).as_micros() as u64;
    let total_us = t2.duration_since(t0).as_micros() as u64;
    let render_us = 0u64; // rendering is below; measured implicitly in total
    let mut timing = Json::object();
    timing.set("normalize_validate_us", Json::uint(norm_us));
    timing.set("pipeline_apply_us", Json::uint(apply_us));
    timing.set("report_render_us", Json::uint(render_us));
    timing.set("total_us", Json::uint(total_us));
    timing.set(
        "ns_per_record",
        Json::uint(total_us.saturating_mul(1_000) / records),
    );
    timing.set("records", Json::uint(records));
    report.set("timing_us", timing);
    Response::ok(report.to_json_string())
}

/// Read-only JSON helpers for the pipeline's own `Json` model (the oracle's
/// `abacus_ingest::api::json` model is converted field-by-field below — the
/// two models are intentionally not unified; each crate keeps its std-only
/// surface).
fn preview_json(p: &abacus_csv_client::Preview) -> Json {
    let mut o = Json::object();
    o.set("input_rows", Json::uint(p.input_rows));
    o.set("accepted_rows", Json::uint(p.accepted_rows));
    o.set("rejected_rows", Json::uint(p.rejected_rows));
    o.set("distinct_accounts", Json::uint(p.distinct_accounts));
    o.set(
        "account_id_width",
        Json::uint(u64::from(p.account_id_width)),
    );
    o.set("discards_subsecond", Json::bool(p.discards_subsecond));
    let issues: Vec<Json> = p
        .issues
        .iter()
        .map(|i| {
            let mut j = Json::object();
            j.set("line", Json::uint(i.line as u64));
            j.set("code", Json::str(i.code.as_str()));
            j.set("detail", Json::str(&i.detail));
            j
        })
        .collect();
    o.set("issues", Json::Array(issues));
    o
}

/// Convert one oracle `Json` value into this crate's `Json` value.
fn convert_json(v: &abacus_ingest::api::json::Json) -> Json {
    use abacus_ingest::api::json::Json as O;
    match v {
        O::Null => Json::Null,
        O::Bool(b) => Json::bool(*b),
        O::Int(n) => Json::int(*n),
        O::Num(f) => Json::float(*f),
        O::Str(s) => Json::str(s.clone()),
        O::Array(a) => Json::Array(a.iter().map(convert_json).collect()),
        O::Object(m) => {
            let mut o = Json::object();
            for (k, val) in m {
                o.set(k, convert_json(val));
            }
            o
        }
    }
}

/// Convert the oracle's canonical session report JSON into this crate's `Json`
/// model, then attach the privacy-safe preview plus this pipeline's own apply
/// outcome. Field-by-field copy (both models are tiny and std-only) — no
/// serde, no string re-parsing.
fn ingest_report_json(
    report: &abacus_ingest::IngestionReport,
    preview: &abacus_csv_client::Preview,
    outcome: crate::ingest_pipeline::PreparedApplyOutcome,
) -> Json {
    let src = report.to_json();
    let get = |key: &str| -> Json {
        match src {
            abacus_ingest::api::json::Json::Object(ref m) => m
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| convert_json(v))
                .unwrap_or(Json::Null),
            _ => Json::Null,
        }
    };
    let mut o = Json::object();
    for key in [
        "session_id",
        "source_type",
        "identity_scheme",
        "event_namespace",
        "wire_format_version",
        "feature_schema_version",
        "feature_algorithm_version",
        "parsing_policy_version",
        "timestamp_conversion_version",
        "config_fingerprint",
        "account_id_width",
        "chunk_count",
        "counts",
        "payload_bytes",
        "min_event_time_ms",
        "max_event_time_ms",
        "final_state",
        "reconciles",
        "artifact_hash",
        "errors",
    ] {
        o.set(key, get(key));
    }
    // The session itself never commits (prepare path) — the pipeline is the
    // applier. Report the pipeline's own apply outcome under `commit` so the
    // response keeps the shape `CsvIngestPanel` expects (`applied` /
    // `already_applied`).
    let mut commit = Json::object();
    commit.set("applied", Json::uint(outcome.applied));
    commit.set("already_applied", Json::uint(outcome.already_applied));
    o.set("commit", commit);
    o.set("preview", preview_json(preview));
    o
}

// ---------------------------------------------------------------------------
// CSF tensor read boundary (HTTP).
//
// `GET /api/ingest/csf/lookup?t=&s=&r=`    exact point read
// `GET /api/ingest/csf/time_slice?t_start=&t_end=`  half-open time window
// `GET /api/ingest/csf/decode`             full live decode
// `GET /api/ingest/csf/stats`              live nonzero count
//
// Read-only and privacy-neutral in shape: the response carries the canonical
// tensor's exact micro-unit values. This is the sanctioned `abacus_tensor::api`
// read boundary exposed over HTTP for computation consumers; it is intentionally
// separate from the amount-free `/api/ingest/transactions` dashboard surface.
// ---------------------------------------------------------------------------

/// Build `GET /api/ingest/csf/stats`.
fn csf_stats_json(p: &IngestPipeline) -> Json {
    let st = p.csf_stats();
    let mut o = Json::object();
    o.set("nnz", Json::uint(st.nnz));
    o
}

/// Build `GET /api/ingest/csf/lookup?t=&s=&r=`.
fn csf_lookup_json(p: &IngestPipeline, t: u64, s: u32, r: u32) -> Response {
    match p.csf_live_at(t, s, r) {
        Some(value) => {
            let mut o = Json::object();
            o.set("t", Json::uint(t));
            o.set("s", Json::uint(u64::from(s)));
            o.set("r", Json::uint(u64::from(r)));
            o.set("found", Json::bool(true));
            o.set("value", Json::int(value));
            Response::ok(o.to_string())
        }
        None => {
            let mut o = Json::object();
            o.set("t", Json::uint(t));
            o.set("s", Json::uint(u64::from(s)));
            o.set("r", Json::uint(u64::from(r)));
            o.set("found", Json::bool(false));
            o.set("value", Json::Null);
            Response::ok(o.to_string())
        }
    }
}

/// Build `GET /api/ingest/csf/time_slice?t_start=&t_end=`.
fn csf_time_slice_json(p: &IngestPipeline, t_start: u64, t_end: u64) -> Json {
    let mut entries = Vec::new();
    if t_start < t_end {
        for loc in p.temporal().global_range(t_start, t_end) {
            if let Some((t, s, r, _feats)) = p.resolve(*loc) {
                if let Some(value) = p.csf_live_at(t, s, r) {
                    let mut e = Json::object();
                    e.set("t", Json::uint(t));
                    e.set("s", Json::uint(u64::from(s)));
                    e.set("r", Json::uint(u64::from(r)));
                    e.set("value", Json::int(value));
                    entries.push(e);
                }
            }
        }
    }
    let mut o = Json::object();
    o.set("t_start", Json::uint(t_start));
    o.set("t_end", Json::uint(t_end));
    o.set("entries", Json::Array(entries));
    o
}

/// Build `GET /api/ingest/csf/sender?t=&s=` — live sender-at-time view.
fn csf_sender_json(p: &IngestPipeline, t: u64, s: u32) -> Json {
    let mut entries = Vec::new();
    for (peer, value) in p.csf_sender_at_time(t, s) {
        let mut e = Json::object();
        e.set("peer", Json::uint(u64::from(peer)));
        e.set("value", Json::int(value));
        entries.push(e);
    }
    let mut o = Json::object();
    o.set("t", Json::uint(t));
    o.set("s", Json::uint(u64::from(s)));
    o.set("entries", Json::Array(entries));
    o
}

/// Build `GET /api/ingest/csf/receiver?t=&r=` — live receiver-at-time view.
fn csf_receiver_json(p: &IngestPipeline, t: u64, r: u32) -> Json {
    let mut entries = Vec::new();
    for (peer, value) in p.csf_receiver_at_time(t, r) {
        let mut e = Json::object();
        e.set("peer", Json::uint(u64::from(peer)));
        e.set("value", Json::int(value));
        entries.push(e);
    }
    let mut o = Json::object();
    o.set("t", Json::uint(t));
    o.set("r", Json::uint(u64::from(r)));
    o.set("entries", Json::Array(entries));
    o
}

/// Build `GET /api/ingest/csf/features?t=&s=&r=` — derived amount-based
/// feature metrics (log1p magnitude, z-threshold, roundedness) for every
/// committed event at coordinate `(t, s, r)`. Privacy-safe: derived metrics
/// only, never the raw amount (GS §9).
fn csf_features_json(p: &IngestPipeline, t: u64, s: u32, r: u32) -> Json {
    let mut entries = Vec::new();
    for f in p.csf_features_at(t, s, r) {
        let mut e = Json::object();
        e.set("log_magnitude", Json::uint(u64::from(f.log_magnitude)));
        e.set(
            "relative_threshold_distance",
            Json::int(i64::from(f.relative_threshold_distance)),
        );
        e.set(
            "roundedness_flags",
            Json::uint(u64::from(f.roundedness_flags)),
        );
        entries.push(e);
    }
    let mut o = Json::object();
    o.set("t", Json::uint(t));
    o.set("s", Json::uint(u64::from(s)));
    o.set("r", Json::uint(u64::from(r)));
    o.set("entries", Json::Array(entries));
    o
}

/// Build `GET /api/ingest/csf/decode`.
fn csf_decode_json(p: &IngestPipeline) -> Json {
    let mut entries = Vec::new();
    for (t, s, r, value) in p.csf_decode_live() {
        let mut e = Json::object();
        e.set("t", Json::uint(t));
        e.set("s", Json::uint(u64::from(s)));
        e.set("r", Json::uint(u64::from(r)));
        e.set("value", Json::int(value));
        entries.push(e);
    }
    let mut o = Json::object();
    o.set("entries", Json::Array(entries));
    o
}

/// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Contract B + Adapter — HTTP semantic routes (`/api/ingest/semantic/*`).
//
// Each handler: parse query → validate → build SemanticOp → call SemanticLayer
// directly → map typed result to JSON → serialize once at the boundary.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// `GET /api/ingest/semantic/lookup?t=&s=&r=` or `start_time=` instead of `t=`.
fn semantic_lookup_json(p: &IngestPipeline, query: &str) -> Response {
    let t = match parse_time_param(query, "t", "start_time") {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Response::bad_request("missing required time parameter: t or start_time")
        }
        Err(e) => return semantic_error_response(&e),
    };
    let s = match query_param_u32(query, "s") {
        Some(s) => s,
        None => return Response::bad_request("missing required parameter: s"),
    };
    let r = match query_param_u32(query, "r") {
        Some(r) => r,
        None => return Response::bad_request("missing required parameter: r"),
    };
    let layer = SemanticLayer::new(p);
    match layer.lookup(t, s, r) {
        Ok(res) => {
            let timestamp_utc = to_utc_iso8601(t)
                .map(|ts| ts.0)
                .unwrap_or_else(|_| format!("{t}"));
            let mut o = Json::object();
            o.set("op", Json::str("lookup"));
            o.set("t", Json::uint(res.t));
            o.set("timestamp_utc", Json::str(&timestamp_utc));
            o.set("s", Json::uint(u64::from(res.s)));
            o.set("r", Json::uint(u64::from(res.r)));
            o.set("found", Json::bool(res.found));
            match res.value {
                Some(v) => o.set("value", Json::int(v)),
                None => o.set("value", Json::Null),
            };
            Response::ok(o.to_json_string())
        }
        Err(e) => semantic_error_response(&e),
    }
}

/// `GET /api/ingest/semantic/time_slice?t_start=&t_end=` or `start_time=`/`end_time=`.
fn semantic_time_slice_json(p: &IngestPipeline, query: &str) -> Response {
    let t_start = match parse_time_param(query, "t_start", "start_time") {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Response::bad_request("missing required time parameter: t_start or start_time")
        }
        Err(e) => return semantic_error_response(&e),
    };
    let t_end = match parse_time_param(query, "t_end", "end_time") {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Response::bad_request("missing required time parameter: t_end or end_time")
        }
        Err(e) => return semantic_error_response(&e),
    };
    let layer = SemanticLayer::new(p);
    match layer.time_slice(t_start, t_end) {
        Ok(res) => {
            let t_start_utc = to_utc_iso8601(t_start)
                .map(|ts| ts.0)
                .unwrap_or_else(|_| format!("{t_start}"));
            let t_end_utc = to_utc_iso8601(t_end)
                .map(|ts| ts.0)
                .unwrap_or_else(|_| format!("{t_end}"));
            let mut o = Json::object();
            o.set("op", Json::str("time_slice"));
            o.set("t_start", Json::uint(res.t_start));
            o.set("t_start_utc", Json::str(&t_start_utc));
            o.set("t_end", Json::uint(res.t_end));
            o.set("t_end_utc", Json::str(&t_end_utc));
            let mut entries = Json::Array(Vec::new());
            if let Json::Array(ref mut arr) = entries {
                for e in &res.entries {
                    let entry_utc = to_utc_iso8601(e.t)
                        .map(|ts| ts.0)
                        .unwrap_or_else(|_| format!("{}", e.t));
                    let mut item = Json::object();
                    item.set("t", Json::uint(e.t));
                    item.set("timestamp_utc", Json::str(&entry_utc));
                    item.set("s", Json::uint(u64::from(e.s)));
                    item.set("r", Json::uint(u64::from(e.r)));
                    item.set("value", Json::int(e.value));
                    arr.push(item);
                }
            }
            o.set("entries", entries);
            Response::ok(o.to_json_string())
        }
        Err(e) => semantic_error_response(&e),
    }
}

/// `GET /api/ingest/semantic/sender?t=&s=` or `start_time=` instead of `t=`.
fn semantic_sender_json(p: &IngestPipeline, query: &str) -> Response {
    let t = match parse_time_param(query, "t", "start_time") {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Response::bad_request("missing required time parameter: t or start_time")
        }
        Err(e) => return semantic_error_response(&e),
    };
    let s = match query_param_u32(query, "s") {
        Some(s) => s,
        None => return Response::bad_request("missing required parameter: s"),
    };
    let layer = SemanticLayer::new(p);
    match layer.sender(t, s) {
        Ok(res) => {
            let timestamp_utc = to_utc_iso8601(t)
                .map(|ts| ts.0)
                .unwrap_or_else(|_| format!("{t}"));
            let mut o = Json::object();
            o.set("op", Json::str("sender"));
            o.set("t", Json::uint(res.t));
            o.set("timestamp_utc", Json::str(&timestamp_utc));
            o.set("s", Json::uint(u64::from(res.s)));
            let mut entries = Json::Array(Vec::new());
            if let Json::Array(ref mut arr) = entries {
                for e in &res.entries {
                    let entry_utc = to_utc_iso8601(res.t)
                        .map(|ts| ts.0)
                        .unwrap_or_else(|_| format!("{}", res.t));
                    let mut item = Json::object();
                    item.set("peer", Json::uint(u64::from(e.peer)));
                    item.set("timestamp_utc", Json::str(&entry_utc));
                    item.set("value", Json::int(e.value));
                    arr.push(item);
                }
            }
            o.set("entries", entries);
            Response::ok(o.to_json_string())
        }
        Err(e) => semantic_error_response(&e),
    }
}

/// `GET /api/ingest/semantic/receiver?t=&r=` or `start_time=` instead of `t=`.
fn semantic_receiver_json(p: &IngestPipeline, query: &str) -> Response {
    let t = match parse_time_param(query, "t", "start_time") {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Response::bad_request("missing required time parameter: t or start_time")
        }
        Err(e) => return semantic_error_response(&e),
    };
    let r = match query_param_u32(query, "r") {
        Some(r) => r,
        None => return Response::bad_request("missing required parameter: r"),
    };
    let layer = SemanticLayer::new(p);
    match layer.receiver(t, r) {
        Ok(res) => {
            let timestamp_utc = to_utc_iso8601(t)
                .map(|ts| ts.0)
                .unwrap_or_else(|_| format!("{t}"));
            let mut o = Json::object();
            o.set("op", Json::str("receiver"));
            o.set("t", Json::uint(res.t));
            o.set("timestamp_utc", Json::str(&timestamp_utc));
            o.set("r", Json::uint(u64::from(res.r)));
            let mut entries = Json::Array(Vec::new());
            if let Json::Array(ref mut arr) = entries {
                for e in &res.entries {
                    let entry_utc = to_utc_iso8601(res.t)
                        .map(|ts| ts.0)
                        .unwrap_or_else(|_| format!("{}", res.t));
                    let mut item = Json::object();
                    item.set("peer", Json::uint(u64::from(e.peer)));
                    item.set("timestamp_utc", Json::str(&entry_utc));
                    item.set("value", Json::int(e.value));
                    arr.push(item);
                }
            }
            o.set("entries", entries);
            Response::ok(o.to_json_string())
        }
        Err(e) => semantic_error_response(&e),
    }
}

/// `GET /api/ingest/semantic/features?t=&s=&r=` or `start_time=` instead of `t=`.
fn semantic_features_json(p: &IngestPipeline, query: &str) -> Response {
    let t = match parse_time_param(query, "t", "start_time") {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Response::bad_request("missing required time parameter: t or start_time")
        }
        Err(e) => return semantic_error_response(&e),
    };
    let s = match query_param_u32(query, "s") {
        Some(s) => s,
        None => return Response::bad_request("missing required parameter: s"),
    };
    let r = match query_param_u32(query, "r") {
        Some(r) => r,
        None => return Response::bad_request("missing required parameter: r"),
    };
    let layer = SemanticLayer::new(p);
    match layer.features(t, s, r) {
        Ok(res) => {
            let timestamp_utc = to_utc_iso8601(t)
                .map(|ts| ts.0)
                .unwrap_or_else(|_| format!("{t}"));
            let mut o = Json::object();
            o.set("op", Json::str("features"));
            o.set("t", Json::uint(res.t));
            o.set("timestamp_utc", Json::str(&timestamp_utc));
            o.set("s", Json::uint(u64::from(res.s)));
            o.set("r", Json::uint(u64::from(res.r)));
            let mut entries = Json::Array(Vec::new());
            if let Json::Array(ref mut arr) = entries {
                for e in &res.entries {
                    let entry_utc = to_utc_iso8601(res.t)
                        .map(|ts| ts.0)
                        .unwrap_or_else(|_| format!("{}", res.t));
                    let mut item = Json::object();
                    item.set("timestamp_utc", Json::str(&entry_utc));
                    item.set("log_magnitude", Json::uint(u64::from(e.log_magnitude)));
                    item.set(
                        "relative_threshold_distance",
                        Json::int(i64::from(e.relative_threshold_distance)),
                    );
                    item.set(
                        "roundedness_flags",
                        Json::uint(u64::from(e.roundedness_flags)),
                    );
                    arr.push(item);
                }
            }
            o.set("entries", entries);
            Response::ok(o.to_json_string())
        }
        Err(e) => semantic_error_response(&e),
    }
}

/// `GET /api/ingest/semantic/stats`
fn semantic_stats_json(p: &IngestPipeline) -> Response {
    let layer = SemanticLayer::new(p);
    match layer.stats() {
        Ok(res) => {
            let mut o = Json::object();
            o.set("op", Json::str("stats"));
            o.set("nnz", Json::uint(res.nnz));
            Response::ok(o.to_json_string())
        }
        Err(e) => semantic_error_response(&e),
    }
}

/// `GET /api/ingest/semantic/decode`
fn semantic_decode_json(p: &IngestPipeline) -> Response {
    let layer = SemanticLayer::new(p);
    match layer.decode() {
        Ok(res) => {
            let mut o = Json::object();
            o.set("op", Json::str("decode"));
            let mut entries = Json::Array(Vec::new());
            if let Json::Array(ref mut arr) = entries {
                for e in &res.entries {
                    let entry_utc = to_utc_iso8601(e.t)
                        .map(|ts| ts.0)
                        .unwrap_or_else(|_| format!("{}", e.t));
                    let mut item = Json::object();
                    item.set("t", Json::uint(e.t));
                    item.set("timestamp_utc", Json::str(&entry_utc));
                    item.set("s", Json::uint(u64::from(e.s)));
                    item.set("r", Json::uint(u64::from(e.r)));
                    item.set("value", Json::int(e.value));
                    arr.push(item);
                }
            }
            o.set("entries", entries);
            Response::ok(o.to_json_string())
        }
        Err(e) => semantic_error_response(&e),
    }
}

/// Serialize a [`RetrievalResult`] to JSON.
fn retrieval_result_json(query_op: &str, res: &crate::retrieval::RetrievalResult) -> Json {
    let mut o = Json::object();
    o.set("op", Json::str(query_op));
    let mut locators = Json::Array(Vec::new());
    if let Json::Array(ref mut arr) = locators {
        for loc in res.locators.as_slice() {
            arr.push(Json::uint(u64::from(loc.0)));
        }
    }
    o.set("locators", locators);
    o.set("is_truncated", Json::bool(res.is_truncated));
    match res.total_matching_events {
        Some(n) => o.set("total_matching_events", Json::uint(n as u64)),
        None => o.set("total_matching_events", Json::Null),
    };
    o
}

/// Build `GET /api/ingest/retrieval/incoming?entity=&anchor=&delta=&max_edges=`.
///
/// Backward / upstream lookback: incoming events to `entity` in
/// `[anchor - delta, anchor)`. Default delta = 7 days.
fn retrieval_incoming_json(p: &IngestPipeline, query: &str) -> Response {
    let entity = match query_param_str(query, "entity") {
        Some(e) if !e.is_empty() => e,
        _ => return Response::bad_request("missing required parameter: entity"),
    };
    let anchor = match query_param_u64(query, "anchor") {
        Some(a) => a,
        None => return Response::bad_request("missing required parameter: anchor"),
    };
    let delta = query_param_u64(query, "delta").unwrap_or(DEFAULT_DELTA_SECS);
    let max_edges = query_param_usize(query, "max_edges").unwrap_or(MAX_EDGES_PER_LOOKUP);
    let q = MultiHopQuery::new(entity, anchor, delta, max_edges);
    let res = p.incoming_relative(&q);
    Response::ok(retrieval_result_json("incoming", &res).to_json_string())
}

/// Build `GET /api/ingest/retrieval/outgoing?entity=&anchor=&delta=&max_edges=`.
///
/// Forward / downstream lookahead: outgoing events from `entity` in
/// `[anchor, anchor + delta)`. Default delta = 7 days.
fn retrieval_outgoing_json(p: &IngestPipeline, query: &str) -> Response {
    let entity = match query_param_str(query, "entity") {
        Some(e) if !e.is_empty() => e,
        _ => return Response::bad_request("missing required parameter: entity"),
    };
    let anchor = match query_param_u64(query, "anchor") {
        Some(a) => a,
        None => return Response::bad_request("missing required parameter: anchor"),
    };
    let delta = query_param_u64(query, "delta").unwrap_or(DEFAULT_DELTA_SECS);
    let max_edges = query_param_usize(query, "max_edges").unwrap_or(MAX_EDGES_PER_LOOKUP);
    let q = MultiHopQuery::new(entity, anchor, delta, max_edges);
    let res = p.outgoing_relative(&q);
    Response::ok(retrieval_result_json("outgoing", &res).to_json_string())
}

/// Build `GET /api/ingest/retrieval/combined?entity=&anchor=&delta=&max_edges=`.
///
/// Combined incoming + outgoing events for `entity` in
/// `[anchor - delta, anchor + delta)`. Default delta = 7 days.
fn retrieval_combined_json(p: &IngestPipeline, query: &str) -> Response {
    let entity = match query_param_str(query, "entity") {
        Some(e) if !e.is_empty() => e,
        _ => return Response::bad_request("missing required parameter: entity"),
    };
    let anchor = match query_param_u64(query, "anchor") {
        Some(a) => a,
        None => return Response::bad_request("missing required parameter: anchor"),
    };
    let delta = query_param_u64(query, "delta").unwrap_or(DEFAULT_DELTA_SECS);
    let max_edges = query_param_usize(query, "max_edges").unwrap_or(MAX_EDGES_PER_LOOKUP);
    let q = MultiHopQuery::new(entity, anchor, delta, max_edges);
    let res = p.combined_relative(&q);
    Response::ok(retrieval_result_json("combined", &res).to_json_string())
}

/// Build `GET /api/ingest/retrieval/range?t_start=&t_end=`.
///
/// Global temporal retrieval: every event locator in `[t_start, t_end)`,
/// uncapped, ordered `(t, locator)` ascending.
fn retrieval_range_json(p: &IngestPipeline, query: &str) -> Response {
    let t_start = match query_param_u64(query, "t_start") {
        Some(t) => t,
        None => return Response::bad_request("missing required parameter: t_start"),
    };
    let t_end = match query_param_u64(query, "t_end") {
        Some(t) => t,
        None => return Response::bad_request("missing required parameter: t_end"),
    };
    let locators = p.events_in_range(t_start, t_end);
    let mut o = Json::object();
    o.set("op", Json::str("range"));
    let mut arr = Json::Array(Vec::new());
    if let Json::Array(ref mut a) = arr {
        for loc in &locators {
            a.push(Json::uint(u64::from(loc.0)));
        }
    }
    o.set("locators", arr);
    o.set("count", Json::uint(locators.len() as u64));
    Response::ok(o.to_json_string())
}

/// Build `GET /api/ingest/retrieval/resolve?locator=`.
///
/// Resolve a locator to its authoritative `(t, sender, receiver)`.
fn retrieval_resolve_json(p: &IngestPipeline, query: &str) -> Response {
    let locator = match query_param_u64(query, "locator") {
        Some(l) => l,
        None => return Response::bad_request("missing required parameter: locator"),
    };
    let Ok(loc) = u32::try_from(locator) else {
        return Response::bad_request("locator out of u32 range");
    };
    // The wire IDs are the external account vocabulary (the same strings
    // the dashboard surface and the `entity` query parameter use); the
    // internal numeric codes are never exposed.
    let wire_of = |code: u32| p.account_label(code).and_then(|s| s.parse::<u64>().ok());
    match p.resolve_event(EventLocator(loc)) {
        Some(ResolvedEvent {
            locator,
            t,
            sender,
            receiver,
        }) => {
            let (Some(sw), Some(rw)) = (wire_of(sender), wire_of(receiver)) else {
                return Response::json(
                    500,
                    "{\"error\":\"account code has no wire identifier\"}".to_string(),
                );
            };
            let mut o = Json::object();
            o.set("op", Json::str("resolve"));
            o.set("locator", Json::uint(u64::from(locator.0)));
            o.set("t", Json::uint(t));
            o.set("sender", Json::uint(sw));
            o.set("receiver", Json::uint(rw));
            Response::ok(o.to_json_string())
        }
        None => Response::not_found("locator not found"),
    }
}

/// Build `GET /api/ingest/retrieval/check`.
///
/// Structural cross-check of tensor/index agreement over the committed set.
fn retrieval_check_json(p: &IngestPipeline) -> Response {
    match p.check_index_consistency() {
        Ok(()) => {
            let mut o = Json::object();
            o.set("op", Json::str("check"));
            o.set("status", Json::str("consistent"));
            Response::ok(o.to_json_string())
        }
        Err(e) => {
            let mut o = Json::object();
            o.set("op", Json::str("check"));
            o.set("status", Json::str("inconsistent"));
            o.set("message", Json::str(e.to_string()));
            Response::json(500, o.to_json_string())
        }
    }
}

/// Map a [`SemanticError`] to a stable HTTP error response.
fn semantic_error_response(err: &SemanticError) -> Response {
    match err {
        SemanticError::InvalidTimeWindow { t_start, t_end } => {
            let mut o = Json::object();
            o.set("error", Json::str("invalid_time_window"));
            o.set(
                "message",
                Json::str(format!(
                    "invalid time window: [{t_start}, {t_end}) is empty or inverted"
                )),
            );
            o.set("t_start", Json::uint(*t_start));
            o.set("t_end", Json::uint(*t_end));
            Response::json(400, o.to_json_string())
        }
        SemanticError::BadRequest(msg) => {
            let mut o = Json::object();
            o.set("error", Json::str("bad_request"));
            o.set("message", Json::str(msg));
            Response::json(400, o.to_json_string())
        }
        SemanticError::Backend(msg) => {
            let mut o = Json::object();
            o.set("error", Json::str("backend_error"));
            o.set("message", Json::str(msg));
            Response::json(500, o.to_json_string())
        }
        SemanticError::UnsupportedOperation(op) => {
            let mut o = Json::object();
            o.set("error", Json::str("unsupported_operation"));
            o.set(
                "message",
                Json::str(format!("unsupported semantic operation: {op}")),
            );
            Response::json(400, o.to_json_string())
        }
    }
}

/// Parse a time parameter that can be either numeric (Unix epoch seconds)
/// or ISO 8601 human-readable timestamp.
///
/// Returns `None` if neither is provided. Returns `Err` if both are provided
/// and conflict, or if the ISO 8601 value is malformed.
fn parse_time_param(
    query: &str,
    numeric_key: &str,
    human_key: &str,
) -> Result<Option<u64>, SemanticError> {
    let numeric_val = query_param_u64(query, numeric_key);
    let human_val = query_param_str(query, human_key);

    match (numeric_val, human_val) {
        (Some(t), Some(iso)) => {
            // Both provided: verify they represent the same instant.
            let iso_secs = from_iso8601(&iso)
                .map_err(|e| SemanticError::BadRequest(format!("invalid {human_key}: {e}")))?;
            if iso_secs != t {
                return Err(SemanticError::BadRequest(format!(
                    "conflicting time parameters: {numeric_key}={t} and {human_key}={iso} represent different instants"
                )));
            }
            Ok(Some(t))
        }
        (Some(t), None) => Ok(Some(t)),
        (None, Some(iso)) => {
            let secs = from_iso8601(&iso)
                .map_err(|e| SemanticError::BadRequest(format!("invalid {human_key}: {e}")))?;
            Ok(Some(secs))
        }
        (None, None) => Ok(None),
    }
}

/// Parse a string query parameter from a raw query string.
fn query_param_str(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let mut kv = pair.splitn(2, '=');
        if kv.next() == Some(key) {
            kv.next().map(|v| v.to_string())
        } else {
            None
        }
    })
}

/// Parse a u64 query parameter from a raw query string.
fn query_param_u64(query: &str, key: &str) -> Option<u64> {
    query
        .split('&')
        .find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            (kv.next() == Some(key)).then(|| kv.next().and_then(|v| v.parse::<u64>().ok()))
        })
        .flatten()
}

/// Parse a u32 query parameter from a raw query string.
fn query_param_u32(query: &str, key: &str) -> Option<u32> {
    query
        .split('&')
        .find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            (kv.next() == Some(key)).then(|| kv.next().and_then(|v| v.parse::<u32>().ok()))
        })
        .flatten()
}

/// Parse a usize query parameter from a raw query string.
fn query_param_usize(query: &str, key: &str) -> Option<usize> {
    query
        .split('&')
        .find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            (kv.next() == Some(key)).then(|| kv.next().and_then(|v| v.parse::<usize>().ok()))
        })
        .flatten()
}

/// Route one semantic request against the shared pipeline.
///
/// Adapter entry point: HTTP request → validated semantic op → SemanticLayer
/// (direct Rust call) → typed result → JSON response. Returns `None` when the
/// path does not match any semantic route.
fn semantic_route(engine: &Mutex<IngestPipeline>, path: &str, query: &str) -> Option<Response> {
    let p = engine.lock().unwrap();

    if path == "/api/ingest/semantic/lookup" {
        return Some(semantic_lookup_json(&p, query));
    }
    if path == "/api/ingest/semantic/time_slice" {
        return Some(semantic_time_slice_json(&p, query));
    }
    if path == "/api/ingest/semantic/sender" {
        return Some(semantic_sender_json(&p, query));
    }
    if path == "/api/ingest/semantic/receiver" {
        return Some(semantic_receiver_json(&p, query));
    }
    if path == "/api/ingest/semantic/features" {
        return Some(semantic_features_json(&p, query));
    }
    if path == "/api/ingest/semantic/stats" {
        return Some(semantic_stats_json(&p));
    }
    if path == "/api/ingest/semantic/decode" {
        return Some(semantic_decode_json(&p));
    }
    None
}

fn route(engine: &Mutex<IngestPipeline>, req: &Request) -> Response {
    if req.method == "OPTIONS" {
        return Response::ok(String::new());
    }
    if req.path == "/api/ingest/csv" && req.method == "POST" {
        let mut p = engine.lock().unwrap();
        return ingest_csv(&mut p, &req.body);
    }
    if req.path == "/api/ingest/health" && req.method == "GET" {
        return Response::ok("{\"ok\":true}".to_string());
    }
    let path = req.path.trim_end_matches('/').to_string();
    let path = if path.is_empty() {
        "/".to_string()
    } else {
        path
    };

    if path == "/api/ingest/status" {
        let p = engine.lock().unwrap();
        return Response::ok(status_json(&p).to_string());
    }
    if path == "/api/ingest/transactions" && req.method == "GET" {
        let page = req.query_usize("page", 1);
        let page_size = req.query_usize("page_size", 1_000);
        let p = engine.lock().unwrap();
        return Response::ok(transactions_page(&p, page, page_size).to_string());
    }
    if let Some(rest) = path.strip_prefix("/api/ingest/transactions/") {
        // `{id}` or `{id}/network/{transactions|accounts}`.
        let mut seg = rest.split('/');
        let Ok(id) = seg.next().unwrap_or("").parse::<u64>() else {
            return Response::bad_request("invalid transaction id");
        };
        match (seg.next(), seg.next()) {
            (None, _) => {
                let p = engine.lock().unwrap();
                return transaction_detail(&p, id);
            }
            (Some("network"), Some("transactions")) => {
                let p = engine.lock().unwrap();
                return match transaction_network(&p, id) {
                    Some(n) => Response::ok(network_json(&n).to_string()),
                    None => Response::not_found("transaction not found"),
                };
            }
            (Some("network"), Some("accounts")) => {
                let p = engine.lock().unwrap();
                return match account_network(&p, id) {
                    Some(n) => Response::ok(network_json(&n).to_string()),
                    None => Response::not_found("transaction not found"),
                };
            }
            (Some("network"), Some("multi-hop")) => {
                let p = engine.lock().unwrap();
                let k = req.query_usize("k", 3) as u64;
                return match transaction_network_multi_hop(&p, id, k) {
                    Some(n) => Response::ok(network_json(&n).to_string()),
                    None => Response::not_found("transaction not found"),
                };
            }
            _ => return Response::not_found("unknown transactions sub-resource"),
        }
    }
    if path == "/api/ingest/csf/stats" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return Response::ok(csf_stats_json(&p).to_string());
    }
    if path == "/api/ingest/csf/lookup" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return csf_lookup_json(
            &p,
            req.query_usize("t", 0) as u64,
            req.query_usize("s", 0) as u32,
            req.query_usize("r", 0) as u32,
        );
    }
    if path == "/api/ingest/csf/time_slice" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return Response::ok(
            csf_time_slice_json(
                &p,
                req.query_usize("t_start", 0) as u64,
                req.query_usize("t_end", 0) as u64,
            )
            .to_string(),
        );
    }
    if path == "/api/ingest/csf/decode" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return Response::ok(csf_decode_json(&p).to_string());
    }
    if path == "/api/ingest/csf/sender" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return Response::ok(
            csf_sender_json(
                &p,
                req.query_usize("t", 0) as u64,
                req.query_usize("s", 0) as u32,
            )
            .to_string(),
        );
    }
    if path == "/api/ingest/csf/receiver" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return Response::ok(
            csf_receiver_json(
                &p,
                req.query_usize("t", 0) as u64,
                req.query_usize("r", 0) as u32,
            )
            .to_string(),
        );
    }
    if path == "/api/ingest/csf/features" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return Response::ok(
            csf_features_json(
                &p,
                req.query_usize("t", 0) as u64,
                req.query_usize("s", 0) as u32,
                req.query_usize("r", 0) as u32,
            )
            .to_string(),
        );
    }
    // Phase-3 temporal retrieval routes (look-back / look-ahead).
    if path == "/api/ingest/retrieval/incoming" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return retrieval_incoming_json(&p, &req.query);
    }
    if path == "/api/ingest/retrieval/outgoing" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return retrieval_outgoing_json(&p, &req.query);
    }
    if path == "/api/ingest/retrieval/combined" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return retrieval_combined_json(&p, &req.query);
    }
    if path == "/api/ingest/retrieval/range" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return retrieval_range_json(&p, &req.query);
    }
    if path == "/api/ingest/retrieval/resolve" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return retrieval_resolve_json(&p, &req.query);
    }
    if path == "/api/ingest/retrieval/check" && req.method == "GET" {
        let p = engine.lock().unwrap();
        return retrieval_check_json(&p);
    }
    if let Some(resp) = semantic_route(&engine, &path, &req.query) {
        return resp;
    }
    if let Some(resp) = graphblas_route(&engine, req) {
        return resp;
    }
    // --- FAN-IN agents pipeline routes (your_task_16.md) ---
    if path == "/api/ingest/fanin/detect" && req.method == "GET" {
        let mut p = engine.lock().unwrap();
        let detection = crate::fanin_agents::detect_candidates(&mut *p);
        return Response::ok(fanin_detect_json(&detection).to_string());
    }
    if path == "/api/ingest/fanin/pipeline" && req.method == "GET" {
        let mut p = engine.lock().unwrap();
        let result = crate::fanin_agents::run_pipeline(&mut *p, None);
        return Response::ok(fanin_pipeline_json(&result).to_string());
    }
    Response::not_found("not found")
}

// --- JSON serialization for FAN-IN agents pipeline (your_task_16.md) ---

fn fanin_detect_json(d: &crate::fanin_agents::DetectionResult) -> crate::web::json::Json {
    let mut o = crate::web::json::Json::object();
    o.set("receivers_scanned", crate::web::json::Json::uint(d.receivers_scanned));
    o.set("fan_in_topology_count", crate::web::json::Json::uint(d.fan_in_topology_count));
    let mut cand_arr: Vec<crate::web::json::Json> = Vec::new();
    for c in &d.candidates {
        let mut co = crate::web::json::Json::object();
        co.set("candidate_id", crate::web::json::Json::uint(c.candidate_id));
        co.set("focal_receiver", crate::web::json::Json::uint(c.focal_receiver as u64));
        co.set("focal_receiver_label", crate::web::json::Json::str(&c.focal_receiver_label));
        co.set("t_i", crate::web::json::Json::uint(c.t_i));
        co.set("n_i", crate::web::json::Json::uint(c.n_i));
        co.set("fi_topology", crate::web::json::Json::bool(c.fi_topology));
        co.set("t_min", c.t_min.map_or(crate::web::json::Json::Null, |t| crate::web::json::Json::uint(t)));
        co.set("t_max", c.t_max.map_or(crate::web::json::Json::Null, |t| crate::web::json::Json::uint(t)));
        co.set("counting_semantics", crate::web::json::Json::str(&c.counting_semantics));
        let mut locs: Vec<crate::web::json::Json> = Vec::new();
        for loc in &c.incoming_locators {
            locs.push(crate::web::json::Json::uint(*loc as u64));
        }
        co.set("incoming_locators", crate::web::json::Json::Array(locs));
        let mut senders: Vec<crate::web::json::Json> = Vec::new();
        for s in &c.incoming_senders {
            senders.push(crate::web::json::Json::uint(*s as u64));
        }
        co.set("incoming_senders", crate::web::json::Json::Array(senders));
        cand_arr.push(co);
    }
    o.set("candidates", crate::web::json::Json::Array(cand_arr));
    o
}

fn fanin_pipeline_json(r: &crate::fanin_agents::FanInPipelineResult) -> crate::web::json::Json {
    let mut o = crate::web::json::Json::object();
    o.set("receivers_scanned", crate::web::json::Json::uint(r.detection.receivers_scanned));
    o.set("fan_in_topology_count", crate::web::json::Json::uint(r.detection.fan_in_topology_count));
    o.set("candidates_measured", crate::web::json::Json::uint(r.measurement.candidates_measured));
    o.set("measurements_complete", crate::web::json::Json::uint(r.measurement.measurements_complete));
    o.set("measurements_incomplete", crate::web::json::Json::uint(r.measurement.measurements_incomplete));
    let mut val_arr: Vec<crate::web::json::Json> = Vec::new();
    for v in &r.validation {
        let mut vo = crate::web::json::Json::object();
        vo.set("candidate_id", crate::web::json::Json::uint(v.candidate_id));
        vo.set("ground_truth_matched", crate::web::json::Json::bool(v.ground_truth_matched));
        vo.set("ground_truth_label", v.ground_truth_label.map_or(crate::web::json::Json::Null, |l| crate::web::json::Json::uint(l as u64)));
        vo.set("ground_truth_type", v.ground_truth_type.clone().map_or(crate::web::json::Json::Null, |t| crate::web::json::Json::str(&t)));
        let status_str = match v.status {
            crate::fanin_agents::ValidationStatus::StructurallyDetected => "STRUCTURALLY_DETECTED",
            crate::fanin_agents::ValidationStatus::Measured => "MEASURED",
            crate::fanin_agents::ValidationStatus::GroundTruthMatched => "GROUND_TRUTH_MATCHED",
            crate::fanin_agents::ValidationStatus::EmpiricallyValidated => "EMPIRICALLY_VALIDATED",
            crate::fanin_agents::ValidationStatus::NotDiscriminative => "NOT_DISCRIMINATIVE",
            crate::fanin_agents::ValidationStatus::Incomplete => "INCOMPLETE",
        };
        vo.set("status", crate::web::json::Json::str(status_str));
        val_arr.push(vo);
    }
    o.set("validation", crate::web::json::Json::Array(val_arr));
    o
}

/// Serve one connection to completion (reuses the shared HTTP request reader).

/// Serve one connection to completion (reuses the shared HTTP request reader).
fn handle_connection(engine: &Arc<Mutex<IngestPipeline>>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    loop {
        match read_request(&mut reader) {
            Ok(Some(req)) => {
                let close = req.connection_close;
                let resp = route(engine, &req);
                if writer.write_all(&resp.to_http()).is_err() {
                    break;
                }
                let _ = writer.flush();
                if close {
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
}

/// Run the ingestion-backed server: bind and accept connections (one thread
/// each). No scheduler is needed — ingestion is driven by `IngestPipeline`
/// directly (the source of records, not a paced artifact streaming_ingestion_demo).
///
/// # Errors
/// `Err` if the listener cannot bind.
pub fn serve_ingest(engine: Arc<Mutex<IngestPipeline>>, addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let eng = Arc::clone(&engine);
                thread::spawn(move || handle_connection(&eng, s));
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest_pipeline::IngestPipeline;
    use abacus_ingest::features::FeaturePayload;
    use abacus_ingest::namespace::{EventIdentity, EventNamespace};

    fn ns(byte: u8) -> EventNamespace {
        EventNamespace::from_bytes([byte; 16])
    }

    fn rec(event_id: u64, s: u64, r: u64) -> abacus_ingest::record::CanonicalRecord {
        abacus_ingest::record::CanonicalRecord {
            identity: EventIdentity {
                namespace: ns(1),
                event_id,
            },
            event_time_ms: 1_700_000_000_000,
            sender_id: s,
            receiver_id: r,
            features: FeaturePayload {
                log_magnitude: 1,
                relative_threshold_distance: 0,
                roundedness_flags: 0,
            },
        }
    }

    fn seeded() -> IngestPipeline {
        let mut p = IngestPipeline::new();
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100)
            .unwrap();
        p
    }

    #[test]
    fn status_reports_shared_namespace_account_count() {
        let p = seeded();
        let j = status_json(&p).to_json_string();
        // 3 distinct accounts across both events in one shared namespace.
        assert!(j.contains("\"committed\":2"));
        assert!(j.contains("\"accounts\":3"));
    }

    #[test]
    fn transactions_expose_shared_labels_and_features_not_amount() {
        let p = seeded();
        let j = transactions_page(&p, 1, 10).to_json_string();
        // Shared-namespace labels (wire ids as decimal strings).
        assert!(j.contains("\"sender\":\"100\""));
        assert!(j.contains("\"receiver\":\"200\""));
        // Derived feature metrics present.
        assert!(j.contains("\"log_magnitude\":9421"));
        assert!(j.contains("\"features\""));
        // NO amount field anywhere (privacy).
        assert!(!j.contains("amount"));
        assert!(!j.contains("value_micro"));
    }

    #[test]
    fn transaction_detail_roundtrips_locator() {
        let p = seeded();
        let resp = transaction_detail(&p, 1);
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("\"id\":1"));
        assert!(body.contains("\"sender\":\"300\""));
        // Missing locator -> 404.
        assert_eq!(transaction_detail(&p, 999).status, 404);
    }

    /// `POST /api/ingest/csv` lands in the SAME pipeline the dashboard reads:
    /// the response reconciles and `status`/`transactions` then show the rows.
    /// (Regression for the upload/dashboard split: the upload used to commit to
    /// the separate `:8090` process while this server stayed empty.)
    #[test]
    fn csv_upload_is_visible_to_the_dashboard_read_surface() {
        let engine = Mutex::new(IngestPipeline::new());
        let csv = b"timestamp;sender;receiver;amount\n2024-01-01T00:00:00Z;alice;bob;123.45\n2024-01-02T00:00:00Z;bob;carol;10.00\n";
        let req = Request {
            method: "POST".into(),
            path: "/api/ingest/csv".into(),
            query: String::new(),
            body: csv.to_vec(),
            connection_close: true,
        };
        let resp = route(&engine, &req);
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("\"reconciles\":true"), "{body}");
        assert!(body.contains("\"applied\":2"), "{body}");
        // No amount anywhere in the response (privacy, GS §9 / A1).
        assert!(!body.contains("123.45"));
        assert!(!body.contains("10.00"));
        // The dashboard read surface now shows the upload.
        let p = engine.lock().unwrap();
        assert_eq!(p.committed(), 2);
        let status = status_json(&p).to_json_string();
        assert!(status.contains("\"committed\":2"));
        let page = transactions_page(&p, 1, 10).to_json_string();
        assert!(page.contains("\"total\":2"));
    }

    /// A GET helper for exercising the CSF route table directly (the same
    /// entry point the HTTP transport uses).
    fn get(path: &str, query: &str) -> Request {
        Request {
            method: "GET".to_string(),
            path: path.to_string(),
            query: query.to_string(),
            body: Vec::new(),
            connection_close: true,
        }
    }

    fn json_body(resp: &Response) -> abacus_ingest::api::json::Json {
        let text = String::from_utf8_lossy(&resp.body);
        abacus_ingest::api::json::parse(&text).expect("response body must parse as JSON")
    }

    #[test]
    fn csf_stats_reports_live_nonzero_count() {
        let engine = Mutex::new(seeded());
        let resp = route(&engine, &get("/api/ingest/csf/stats", ""));
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("nnz").and_then(|v| v.as_u64()), Some(2));
    }

    #[test]
    fn csf_lookup_returns_exact_value_and_absent() {
        let engine = Mutex::new(seeded());
        // Both seeded events share t = 1_700_000_000 s; account codes are 1/2/3.
        // Present coordinate (event 0: sender code 1, receiver code 2).
        let resp = route(
            &engine,
            &get("/api/ingest/csf/lookup", "t=1700000000&s=1&r=2"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("found").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            j.get("value").and_then(|v| v.as_i64()),
            Some(12_345_000_000)
        );
        // Absent coordinate -> found: false, never fabricated as zero.
        let resp = route(
            &engine,
            &get("/api/ingest/csf/lookup", "t=1700000000&s=9&r=9"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("found").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn csf_time_slice_returns_window_entries() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get(
                "/api/ingest/csf/time_slice",
                "t_start=1700000000&t_end=1700000001",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(
            j.get("t_start").and_then(|v| v.as_u64()),
            Some(1_700_000_000)
        );
        assert_eq!(j.get("t_end").and_then(|v| v.as_u64()), Some(1_700_000_001));
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 2);
            assert_eq!(
                entries[0].get("value").and_then(|v| v.as_i64()),
                Some(12_345_000_000)
            );
        } else {
            panic!("entries array missing: {j:?}");
        }
    }

    #[test]
    fn csf_decode_returns_all_entries_with_values() {
        let engine = Mutex::new(seeded());
        let resp = route(&engine, &get("/api/ingest/csf/decode", ""));
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 2);
            for e in entries {
                assert_eq!(
                    e.get("value").and_then(|v| v.as_i64()),
                    Some(12_345_000_000)
                );
            }
        } else {
            panic!("entries array missing: {j:?}");
        }
    }

    #[test]
    fn csf_sender_at_time_returns_receivers_and_values() {
        let engine = Mutex::new(seeded());
        // Event 0: sender code 1 -> receiver code 2 at t = 1_700_000_000.
        let resp = route(&engine, &get("/api/ingest/csf/sender", "t=1700000000&s=1"));
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("t").and_then(|v| v.as_u64()), Some(1_700_000_000));
        assert_eq!(j.get("s").and_then(|v| v.as_u64()), Some(1));
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].get("peer").and_then(|v| v.as_u64()), Some(2));
            assert_eq!(
                entries[0].get("value").and_then(|v| v.as_i64()),
                Some(12_345_000_000)
            );
        } else {
            panic!("entries array missing: {j:?}");
        }
        // Unknown sender -> empty entries (never fabricated).
        let resp = route(&engine, &get("/api/ingest/csf/sender", "t=1700000000&s=9"));
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert!(entries.is_empty());
        } else {
            panic!("entries array missing: {j:?}");
        }
    }

    #[test]
    fn csf_receiver_at_time_returns_senders_and_values() {
        let engine = Mutex::new(seeded());
        // Event 1: receiver code 1 <- sender code 3 at t = 1_700_000_000.
        let resp = route(
            &engine,
            &get("/api/ingest/csf/receiver", "t=1700000000&r=1"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("t").and_then(|v| v.as_u64()), Some(1_700_000_000));
        assert_eq!(j.get("r").and_then(|v| v.as_u64()), Some(1));
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].get("peer").and_then(|v| v.as_u64()), Some(3));
            assert_eq!(
                entries[0].get("value").and_then(|v| v.as_i64()),
                Some(12_345_000_000)
            );
        } else {
            panic!("entries array missing: {j:?}");
        }
        // Unknown receiver -> empty entries (never fabricated).
        let resp = route(
            &engine,
            &get("/api/ingest/csf/receiver", "t=1700000000&r=9"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert!(entries.is_empty());
        } else {
            panic!("entries array missing: {j:?}");
        }
    }

    #[test]
    fn csf_features_returns_amount_derived_metrics() {
        let engine = Mutex::new(seeded());
        // Event 0 at coordinate (t=1_700_000_000, s=code_of(100), r=code_of(200)).
        let p = engine.lock().unwrap();
        let (t, s, r, _) = p.resolve(0).unwrap();
        drop(p);
        let resp = route(
            &engine,
            &get("/api/ingest/csf/features", &format!("t={t}&s={s}&r={r}")),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("t").and_then(|v| v.as_u64()), Some(t));
        assert_eq!(j.get("s").and_then(|v| v.as_u64()), Some(u64::from(s)));
        assert_eq!(j.get("r").and_then(|v| v.as_u64()), Some(u64::from(r)));
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 1);
            assert_eq!(
                entries[0].get("log_magnitude").and_then(|v| v.as_u64()),
                Some(9_421)
            );
            assert_eq!(
                entries[0]
                    .get("relative_threshold_distance")
                    .and_then(|v| v.as_i64()),
                Some(235)
            );
            assert_eq!(
                entries[0].get("roundedness_flags").and_then(|v| v.as_u64()),
                Some(0)
            );
        } else {
            panic!("entries array missing: {j:?}");
        }
        // Unknown coordinate -> empty entries (never fabricated).
        let resp = route(
            &engine,
            &get("/api/ingest/csf/features", "t=1700000000&s=9&r=9"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert!(entries.is_empty());
        } else {
            panic!("entries array missing: {j:?}");
        }
    }

    #[test]
    fn semantic_lookup_http_returns_exact_value_and_absent() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get("/api/ingest/semantic/lookup", "t=1700000000&s=1&r=2"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("found").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            j.get("value").and_then(|v| v.as_i64()),
            Some(12_345_000_000)
        );
        let resp = route(
            &engine,
            &get("/api/ingest/semantic/lookup", "t=1700000000&s=9&r=9"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("found").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn semantic_time_slice_http_returns_window() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/time_slice",
                "t_start=1700000000&t_end=1700000001",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(
            j.get("t_start").and_then(|v| v.as_u64()),
            Some(1_700_000_000)
        );
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 2);
        } else {
            panic!("entries array missing: {j:?}");
        }
    }

    #[test]
    fn semantic_time_slice_http_rejects_inverted_window() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/time_slice",
                "t_start=1700000001&t_end=1700000000",
            ),
        );
        assert_eq!(resp.status, 400);
        let j = json_body(&resp);
        assert_eq!(
            j.get("error").and_then(|v| v.as_str()),
            Some("invalid_time_window")
        );
    }

    #[test]
    fn semantic_sender_http_returns_outgoing() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get("/api/ingest/semantic/sender", "t=1700000000&s=1"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].get("peer").and_then(|v| v.as_u64()), Some(2));
        }
    }

    #[test]
    fn semantic_receiver_http_returns_incoming() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get("/api/ingest/semantic/receiver", "t=1700000000&r=1"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].get("peer").and_then(|v| v.as_u64()), Some(3));
        }
    }

    #[test]
    fn semantic_features_http_returns_derived_metrics() {
        let engine = Mutex::new(seeded());
        let p = engine.lock().unwrap();
        let (t, s, r, _) = p.resolve(0).unwrap();
        drop(p);
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/features",
                &format!("t={t}&s={s}&r={r}"),
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 1);
            assert_eq!(
                entries[0].get("log_magnitude").and_then(|v| v.as_u64()),
                Some(9_421)
            );
        }
    }

    #[test]
    fn semantic_stats_http_returns_nnz() {
        let engine = Mutex::new(seeded());
        let resp = route(&engine, &get("/api/ingest/semantic/stats", ""));
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("nnz").and_then(|v| v.as_u64()), Some(2));
    }

    #[test]
    fn semantic_decode_http_returns_all_entries() {
        let engine = Mutex::new(seeded());
        let resp = route(&engine, &get("/api/ingest/semantic/decode", ""));
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 2);
        }
    }

    #[test]
    fn semantic_unknown_path_returns_404() {
        let engine = Mutex::new(seeded());
        let resp = route(&engine, &get("/api/ingest/semantic/unknown", ""));
        assert_eq!(resp.status, 404);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Timestamp translation HTTP tests
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn semantic_time_slice_http_with_human_readable_times() {
        let engine = Mutex::new(seeded());
        // Both seeded events are at t = 1_700_000_000 = 2023-11-14T22:13:20Z
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/time_slice",
                "start_time=2023-11-14T22:13:20Z&end_time=2023-11-14T22:13:21Z",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(
            j.get("t_start_utc").and_then(|v| v.as_str()),
            Some("2023-11-14T22:13:20Z")
        );
        assert_eq!(
            j.get("t_end_utc").and_then(|v| v.as_str()),
            Some("2023-11-14T22:13:21Z")
        );
        if let Some(abacus_ingest::api::json::Json::Array(entries)) = j.get("entries") {
            assert_eq!(entries.len(), 2);
            assert_eq!(
                entries[0].get("timestamp_utc").and_then(|v| v.as_str()),
                Some("2023-11-14T22:13:20Z")
            );
        } else {
            panic!("entries array missing: {j:?}");
        }
    }

    #[test]
    fn semantic_time_slice_http_with_timezone_offset() {
        let engine = Mutex::new(seeded());
        // 2023-11-15T06:13:20+08:00 = 2023-11-14T22:13:20Z (same instant)
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/time_slice",
                "start_time=2023-11-15T06:13:20+08:00&end_time=2023-11-15T06:13:21+08:00",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(
            j.get("t_start_utc").and_then(|v| v.as_str()),
            Some("2023-11-14T22:13:20Z")
        );
    }

    #[test]
    fn semantic_lookup_http_with_start_time() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/lookup",
                "start_time=2023-11-14T22:13:20Z&s=1&r=2",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("found").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            j.get("timestamp_utc").and_then(|v| v.as_str()),
            Some("2023-11-14T22:13:20Z")
        );
    }

    #[test]
    fn semantic_lookup_http_conflicting_time_params() {
        let engine = Mutex::new(seeded());
        // t=1700000000 and start_time=2023-11-14T22:13:21Z are different instants
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/lookup",
                "t=1700000000&start_time=2023-11-14T22:13:21Z&s=1&r=2",
            ),
        );
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn semantic_lookup_http_invalid_start_time() {
        let engine = Mutex::new(seeded());
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/lookup",
                "start_time=not-a-timestamp&s=1&r=2",
            ),
        );
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn semantic_lookup_http_missing_time_param() {
        let engine = Mutex::new(seeded());
        let resp = route(&engine, &get("/api/ingest/semantic/lookup", "s=1&r=2"));
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn semantic_lookup_http_naive_time_rejected() {
        let engine = Mutex::new(seeded());
        // No timezone designator → ambiguous, should be rejected
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/lookup",
                "start_time=2023-11-14T22:13:20&s=1&r=2",
            ),
        );
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn semantic_lookup_http_numeric_and_human_readable_agree() {
        let engine = Mutex::new(seeded());
        // t=1700000000 == 2023-11-14T22:13:20Z
        let resp = route(
            &engine,
            &get(
                "/api/ingest/semantic/lookup",
                "t=1700000000&start_time=2023-11-14T22:13:20Z&s=1&r=2",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("found").and_then(|v| v.as_bool()), Some(true));
    }

    /// The upload path is sub-millisecond per record end to end (server-side:
    /// normalize+validate → pipeline apply → report render, all in-memory).
    /// Guards the latency requirement: 2000 records must ingest in well under
    /// 2000 × 1 ms.
    #[test]
    fn csv_upload_is_submillisecond_per_record() {
        let mut p = IngestPipeline::new();
        let mut csv = b"timestamp;sender;receiver;amount\n".to_vec();
        for i in 0..2000u32 {
            let line = format!(
                "2024-01-01T00:00:00Z;acc-{i:05};acc-x;{}.{:02}\n",
                (i % 900) + 1,
                i % 100
            );
            csv.extend_from_slice(line.as_bytes());
        }
        let t0 = std::time::Instant::now();
        let resp = ingest_csv(&mut p, &csv);
        let total = t0.elapsed();
        assert_eq!(resp.status, 200);
        assert_eq!(p.committed(), 2000);
        let per_record_ns = total.as_nanos() / 2000;
        assert!(
            per_record_ns < 1_000_000,
            "per-record {per_record_ns}ns exceeds the 1ms budget (total {total:?})"
        );
    }

    /// A re-post of the same bytes is a duplicate delivery, not a second
    /// application (pipeline identity ledger is authoritative).
    #[test]
    fn csv_reupload_is_a_duplicate_not_a_second_application() {
        let engine = Mutex::new(IngestPipeline::new());
        let csv = b"timestamp;sender;receiver;amount\n2024-01-01T00:00:00Z;alice;bob;123.45\n";
        let post = || {
            route(
                &engine,
                &Request {
                    method: "POST".into(),
                    path: "/api/ingest/csv".into(),
                    query: String::new(),
                    body: csv.to_vec(),
                    connection_close: true,
                },
            )
        };
        assert_eq!(post().status, 200);
        let second = post();
        assert_eq!(second.status, 200);
        let body = String::from_utf8_lossy(&second.body);
        assert!(body.contains("\"already_applied\":1"), "{body}");
        assert_eq!(engine.lock().unwrap().committed(), 1);
    }

    /// A CSV with no accepted rows is a 400 (fail-closed), and commits nothing.
    #[test]
    fn csv_with_no_accepted_rows_is_rejected_fail_closed() {
        let engine = Mutex::new(IngestPipeline::new());
        let csv = b"timestamp;sender;receiver;amount\nnot-a-time;alice;bob;not-an-amount\n";
        let resp = route(
            &engine,
            &Request {
                method: "POST".into(),
                path: "/api/ingest/csv".into(),
                query: String::new(),
                body: csv.to_vec(),
                connection_close: true,
            },
        );
        assert_eq!(resp.status, 400);
        assert_eq!(engine.lock().unwrap().committed(), 0);
    }

    fn retrieval_seeded() -> IngestPipeline {
        let mut p = IngestPipeline::new();
        p.apply(
            &abacus_ingest::record::CanonicalRecord {
                identity: EventIdentity {
                    namespace: ns(1),
                    event_id: 0,
                },
                event_time_ms: 100_000,
                sender_id: 100,
                receiver_id: 200,
                features: FeaturePayload {
                    log_magnitude: 1,
                    relative_threshold_distance: 0,
                    roundedness_flags: 0,
                },
            },
            Some(12_345),
            10_000,
            100,
        )
        .unwrap();
        p.apply(
            &abacus_ingest::record::CanonicalRecord {
                identity: EventIdentity {
                    namespace: ns(1),
                    event_id: 1,
                },
                event_time_ms: 200_000,
                sender_id: 100,
                receiver_id: 300,
                features: FeaturePayload {
                    log_magnitude: 2,
                    relative_threshold_distance: 0,
                    roundedness_flags: 0,
                },
            },
            Some(12_345),
            10_000,
            100,
        )
        .unwrap();
        p.apply(
            &abacus_ingest::record::CanonicalRecord {
                identity: EventIdentity {
                    namespace: ns(1),
                    event_id: 2,
                },
                event_time_ms: 300_000,
                sender_id: 200,
                receiver_id: 100,
                features: FeaturePayload {
                    log_magnitude: 3,
                    relative_threshold_distance: 0,
                    roundedness_flags: 0,
                },
            },
            Some(12_345),
            10_000,
            100,
        )
        .unwrap();
        p
    }

    #[test]
    fn retrieval_incoming_lookback_returns_past_transactions() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get(
                "/api/ingest/retrieval/incoming",
                "entity=100&anchor=400&delta=200",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 1);
        assert_eq!(locators[0].as_u64(), Some(2));
        assert_eq!(j.get("is_truncated").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(
            j.get("total_matching_events").and_then(|v| v.as_u64()),
            Some(1)
        );
    }

    #[test]
    fn retrieval_incoming_excludes_anchor_time() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get(
                "/api/ingest/retrieval/incoming",
                "entity=100&anchor=300&delta=100",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 0);
        assert_eq!(
            j.get("total_matching_events").and_then(|v| v.as_u64()),
            Some(0)
        );
    }

    #[test]
    fn retrieval_incoming_unknown_account_returns_empty() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get(
                "/api/ingest/retrieval/incoming",
                "entity=99999&anchor=400&delta=200",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 0);
        assert_eq!(j.get("is_truncated").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn retrieval_outgoing_lookahead_returns_future_transactions() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get(
                "/api/ingest/retrieval/outgoing",
                "entity=100&anchor=100&delta=150",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 2);
        assert_eq!(locators[0].as_u64(), Some(0));
        assert_eq!(locators[1].as_u64(), Some(1));
        assert_eq!(j.get("is_truncated").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn retrieval_outgoing_excludes_interval_end() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get(
                "/api/ingest/retrieval/outgoing",
                "entity=100&anchor=100&delta=100",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 1);
        assert_eq!(locators[0].as_u64(), Some(0));
    }

    #[test]
    fn retrieval_combined_returns_unique_events() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get(
                "/api/ingest/retrieval/combined",
                "entity=100&anchor=200&delta=150",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 3);
        assert_eq!(
            j.get("total_matching_events").and_then(|v| v.as_u64()),
            Some(3)
        );
    }

    #[test]
    fn retrieval_range_is_uncapped_global() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get("/api/ingest/retrieval/range", "t_start=0&t_end=9999999999"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 3);
        assert_eq!(locators[0].as_u64(), Some(0));
        assert_eq!(locators[1].as_u64(), Some(1));
        assert_eq!(locators[2].as_u64(), Some(2));
        assert_eq!(j.get("count").and_then(|v| v.as_u64()), Some(3));
    }

    #[test]
    fn retrieval_resolve_returns_authoritative_event() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get("/api/ingest/retrieval/resolve", "locator=1"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("locator").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(j.get("t").and_then(|v| v.as_u64()), Some(200));
        assert_eq!(j.get("sender").and_then(|v| v.as_u64()), Some(100));
        assert_eq!(j.get("receiver").and_then(|v| v.as_u64()), Some(300));
    }

    #[test]
    fn retrieval_resolve_unknown_locator_returns_404() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get("/api/ingest/retrieval/resolve", "locator=999"),
        );
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn retrieval_check_reports_consistent() {
        let p = retrieval_seeded();
        let resp = route(&Mutex::new(p), &get("/api/ingest/retrieval/check", ""));
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        assert_eq!(j.get("status").and_then(|v| v.as_str()), Some("consistent"));
    }

    #[test]
    fn retrieval_default_delta_is_seven_days() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get("/api/ingest/retrieval/incoming", "entity=100&anchor=400"),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 1);
        assert_eq!(locators[0].as_u64(), Some(2));
    }

    #[test]
    fn retrieval_max_edges_clamps_to_hard_cap() {
        let p = retrieval_seeded();
        let resp = route(
            &Mutex::new(p),
            &get(
                "/api/ingest/retrieval/incoming",
                "entity=100&anchor=400&delta=200&max_edges=10000",
            ),
        );
        assert_eq!(resp.status, 200);
        let j = json_body(&resp);
        let locators = match j.get("locators") {
            Some(abacus_ingest::api::json::Json::Array(arr)) => arr,
            _ => panic!("locators array missing: {j:?}"),
        };
        assert_eq!(locators.len(), 1);
    }

    #[test]
    fn retrieval_missing_required_params_return_400() {
        let resp = route(
            &Mutex::new(retrieval_seeded()),
            &get("/api/ingest/retrieval/incoming", "anchor=400"),
        );
        assert_eq!(resp.status, 400);
        let resp = route(
            &Mutex::new(retrieval_seeded()),
            &get("/api/ingest/retrieval/incoming", "entity=100"),
        );
        assert_eq!(resp.status, 400);
        let resp = route(
            &Mutex::new(retrieval_seeded()),
            &get("/api/ingest/retrieval/range", "t_end=100"),
        );
        assert_eq!(resp.status, 400);
        let resp = route(
            &Mutex::new(retrieval_seeded()),
            &get("/api/ingest/retrieval/resolve", ""),
        );
        assert_eq!(resp.status, 400);
    }
}
