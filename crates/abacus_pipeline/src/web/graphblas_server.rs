//! GraphBLAS algebraic-layer HTTP endpoints (`/api/ingest/graphblas/*`).
//!
//! Exposes the sanctioned GraphBLAS API-layer module
//! (`abacus_tensor::api::graphblas`, Addendum O rows 30–51) over the same
//! std-only thread-per-connection HTTP/1.1 surface that carries the read
//! boundary (`/api/ingest/csf/*`, row 27 / §B.2). Every route is a thin shell
//! — it extracts parameters, calls the API-layer function directly, and
//! shapes the result as JSON. No tensor behavior is reimplemented here
//! (Golden Source: Addendum O rows 30–51 for the algebra, row 27 / §B.2
//! for the read boundary, §0 H1 for the std-only surface).
//!
//! ## State and scope
//!
//! Tensor-bound routes read the pipeline's **sealed canonical tensor**
//! (`IngestPipeline::csf_sealed`, `T_sealed`); the committed-but-unsealed
//! frontier overlay is excluded. Call `POST /api/ingest/graphblas/seal` first
//! to compact pending commits into the sealed tensor (row 22). The sealed
//! object is borrowed read-only: no route rebuilds, re-materializes, or
//! mutates the canonical tensor.
//!
//! ## Value semantics (phase-5 API rule 9)
//!
//! `value` fields are the canonical tensor's exact micro-unit values — the
//! intended, sanctioned API read boundary for computation consumers, matching
//! `/api/ingest/csf/*`. They are intentionally separate from the amount-free
//! `/api/ingest/transactions` dashboard surface (GS §9). Errors never carry
//! raw account identifiers or amounts (Golden Source §16): reasons come from
//! a closed structural alphabet.
//!
//! ## Routes
//!
//! Tensor-bound (all `GET`; Addendum O rows in parentheses):
//!
//! | Path | API-layer call |
//! |------|---------------|
//! | `/contract_mode0?t_start=&t_end=` | `contract_mode0_window` (row 30) |
//! | `/temporal_multiplicity?t_start=&t_end=` | `temporal_multiplicity_window` (row 31) |
//! | `/support?t_start=&t_end=` | `build_support_window` (#19, row 48) |
//! | `/prune_time?t_start=&t_end=` | `prune_time_from_tensor` (#14, row 43) |
//! | `/hop_ball?t_start=&t_end=&u=&v=&k=` | `hop_ball_from_tensor` (#17, row 46) |
//! | `/prune_vertices?t_start=&t_end=&u=&v=&k=` | hop ball + `prune_vertices_window` (#15, row 44) |
//! | `/prune_edges?t_start=&t_end=&pair=&pair=` | `prune_edges_window` (#16, row 45); `pair=u:r`, repeatable |
//! | `/prune_hops?t_start=&t_end=&u=&v=&k=` | `prune_hops_from_tensor` (#18, row 47) |
//! | `/focal_context?t_start=&t_end=&k=&epoch=&s=&r=[&self_loop=][&pair=]` | `build_focal_context_from_tensor` (#20, row 49) |
//! | `/fan_in?t_start=&t_end=&k=&epoch=&s=&r=[&self_loop=][&pair=]` | `compute_fan_in_from_tensor` (#21/#22, row 50) |
//! | `POST /seal` | `IngestPipeline::seal_substrate` (row 22; no body) |
//!
//! Stateless primitives (all `POST` with a JSON object body; Addendum O #/rows
//! in parentheses): `/primitive/extract` (#1, row 32),
//! `/primitive/reduce_col_plus` (#2, row 33), `/primitive/reduce_row_plus`
//! (#3, row 34), `/primitive/mxv` (#4, row 35), `/primitive/vxm` (#5, row 36),
//! `/primitive/ewise_add_lor` (#6, row 37), `/primitive/support` (#7, row 38),
//! `/primitive/multiplicity` (#8, row 39), `/primitive/volume` (#9, row 40),
//! `/primitive/reduce_count` (#10, row 41), `/primitive/reduce_count_set`
//! (#11, row 41), `/primitive/reduce_sum` (#12, row 42), `/primitive/reduce_max`
//! (#13, row 42).
//!
//! The GraphBLAS read-boundary helpers already live at `/api/ingest/csf/*`
//! (`lookup`, `time_slice` — the live-tensor analogues of `extract_3d` and
//! `prune_time`) and are reused where appropriate — no duplicate endpoints
//! rename them (phase-5 API rule: introduce new routes only for newly required
//! capability). `#23 compare_with_oracle` (row 51) is diagnostic-only and is
//! **not** exposed through this surface.
//!
//! ## Status codes
//!
//! - `200` success (absent coordinates report `found: false`, P3).
//! - `400` invalid request (missing/unparseable parameter, malformed body,
//!   duplicate or zero entry, focal outside window, value-domain overflow).
//! - `404` unknown route under `/api/ingest/graphblas/*`, or a focal
//!   coordinate outside `supp(T)` (section A3).
//! - `405` path matched but the method is wrong (tensor-bound routes are GET;
//!   primitive and seal routes are POST).

use crate::ingest_pipeline::IngestPipeline;
use crate::web::json::{parse as parse_json, Json};
use crate::web::server::{Request, Response};

use std::sync::Mutex;

use abacus_tensor::api::graphblas as gb;
use abacus_tensor::graphblas::matrix::PairMatrix;
use abacus_tensor::graphblas::types::{
    EligiblePairs, FocalTransaction, HopBound, SelfLoopRule, TimeWindow, VertexId,
};
use abacus_tensor::graphblas::vector::Vector;

// ---------------------------------------------------------------------------
// Query-string helpers (same shapes as the adjacent `csf/*` / `semantic/*`
// handlers; fail-closed: unparseable values are `None`).
// ---------------------------------------------------------------------------

/// Required `u64` query parameter; `None` when absent or unparseable.
fn need_u64(query: &str, key: &str) -> Option<u64> {
    query
        .split('&')
        .find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            (kv.next() == Some(key)).then(|| kv.next().and_then(|v| v.parse::<u64>().ok()))
        })
        .flatten()
}

/// Required `u32` query parameter; `None` when absent or unparseable.
fn need_u32(query: &str, key: &str) -> Option<u32> {
    query
        .split('&')
        .find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            (kv.next() == Some(key)).then(|| kv.next().and_then(|v| v.parse::<u32>().ok()))
        })
        .flatten()
}

/// Optional `self_loop=admit|exclude`; defaults to exclude. `None` on any
/// other spelling (an invented rule would violate O's explicit-domain rule).
fn parse_self_loop(query: &str) -> Option<SelfLoopRule> {
    let mut found: Option<SelfLoopRule> = None;
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        if kv.next() == Some("self_loop") {
            found = match kv.next() {
                Some("admit") => Some(SelfLoopRule::Admit),
                Some("exclude") => Some(SelfLoopRule::Exclude),
                _ => return None,
            };
        }
    }
    Some(found.unwrap_or(SelfLoopRule::Exclude))
}

/// Collect every `pair=u:r` occurrence as an eligible-pair set (sorted,
/// distinct — the `EligiblePairs` membership contract is set semantics).
/// `None` when any occurrence is malformed.
fn parse_pairs(query: &str) -> Option<EligiblePairs> {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for part in query.split('&') {
        let mut kv = part.splitn(2, '=');
        if kv.next() == Some("pair") {
            let val = kv.next()?;
            let mut uv = val.splitn(2, ':');
            let u: u32 = uv.next()?.parse().ok()?;
            let r: u32 = uv.next()?.parse().ok()?;
            pairs.push((u, r));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    Some(EligiblePairs::from_sorted(pairs))
}

// ---------------------------------------------------------------------------
// JSON shaping (phase-5 rules 8/9: distinguish invalid/missing/unsupported;
// document value meanings — see module docs).
// ---------------------------------------------------------------------------

/// One `(s, r, value)` matrix entry.
fn matrix_entry_json(s: u32, r: u32, value: i64) -> Json {
    let mut e = Json::object();
    e.set("s", Json::uint(u64::from(s)));
    e.set("r", Json::uint(u64::from(r)));
    e.set("value", Json::int(value));
    e
}

/// One `(t, s, r, value)` canonical tensor entry.
fn tensor_entry_json(t: u64, s: u32, r: u32, value: i64) -> Json {
    let mut e = Json::object();
    e.set("t", Json::uint(t));
    e.set("s", Json::uint(u64::from(s)));
    e.set("r", Json::uint(u64::from(r)));
    e.set("value", Json::int(value));
    e
}

/// A `PairMatrix` as `{"nnz":N,"entries":[{"s","r","value"}]}`.
fn matrix_json(m: &PairMatrix) -> Json {
    let entries: Vec<Json> = m
        .nonzeros()
        .map(|(s, r, v)| matrix_entry_json(s, r, v))
        .collect();
    let mut o = Json::object();
    o.set("nnz", Json::uint(entries.len() as u64));
    o.set("entries", Json::Array(entries));
    o
}

/// A `Vector` pattern as `{"nnz":N,"members":[...]}`.
fn vector_json(v: &Vector) -> Json {
    let members: Vec<Json> = v
        .indices()
        .iter()
        .map(|&i| Json::uint(u64::from(i)))
        .collect();
    let mut o = Json::object();
    o.set("nnz", Json::uint(members.len() as u64));
    o.set("members", Json::Array(members));
    o
}

// ---------------------------------------------------------------------------
// Tensor-bound handlers (GET). Each reads `p.csf_sealed()` (T_sealed) and
// calls the `abacus_tensor::api::graphblas` boundary function directly.
// ---------------------------------------------------------------------------

/// Require the half-open window `[t_start, t_end)` (phase-5 rule 8: required
/// parameters are explicit — no silent whole-tensor scan).
fn need_window(query: &str) -> Result<(u64, u64), Response> {
    let t_start = need_u64(query, "t_start")
        .ok_or_else(|| Response::bad_request("missing required parameter: t_start"))?;
    let t_end = need_u64(query, "t_end")
        .ok_or_else(|| Response::bad_request("missing required parameter: t_end"))?;
    Ok((t_start, t_end))
}

/// Require `(u, v)` vertex codes plus the hop bound `k`.
fn need_ball(query: &str) -> Result<(VertexId, VertexId, HopBound), Response> {
    let u = need_u32(query, "u")
        .ok_or_else(|| Response::bad_request("missing required parameter: u"))?;
    let v = need_u32(query, "v")
        .ok_or_else(|| Response::bad_request("missing required parameter: v"))?;
    let k = need_u64(query, "k")
        .ok_or_else(|| Response::bad_request("missing required parameter: k"))?;
    Ok((VertexId(u), VertexId(v), HopBound(k)))
}

/// `GET …/contract_mode0?t_start=&t_end=` — `A_w(s,r)` (Addendum O row 30).
fn contract_mode0_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    match gb::contract_mode0_window(p.csf_sealed(), t_start, t_end) {
        Ok(m) => {
            let mut o = matrix_json(&m);
            o.set("t_start", Json::uint(t_start));
            o.set("t_end", Json::uint(t_end));
            Response::ok(o.to_string())
        }
        Err(_) => Response::bad_request("value-domain overflow: window sum unrepresentable"),
    }
}

/// `GET …/temporal_multiplicity?t_start=&t_end=` — `A_cnt` (row 31).
fn temporal_multiplicity_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let m = gb::temporal_multiplicity_window(p.csf_sealed(), t_start, t_end);
    let mut o = matrix_json(&m);
    o.set("t_start", Json::uint(t_start));
    o.set("t_end", Json::uint(t_end));
    Response::ok(o.to_string())
}

/// `GET …/support?t_start=&t_end=` — `A_one`/`A_cnt`/`A_vol` (#19, row 48).
fn support_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    match gb::build_support_window(p.csf_sealed(), t_start, t_end) {
        Ok(s) => {
            let mut o = Json::object();
            o.set("t_start", Json::uint(t_start));
            o.set("t_end", Json::uint(t_end));
            o.set("a_one", matrix_json(&s.a_one));
            o.set("a_cnt", matrix_json(&s.a_cnt));
            o.set("a_vol", matrix_json(&s.a_vol));
            Response::ok(o.to_string())
        }
        Err(_) => Response::bad_request("value-domain overflow: window sum unrepresentable"),
    }
}

/// `GET …/prune_time?t_start=&t_end=` — `T|_w` (#14, row 43).
fn prune_time_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let entries: Vec<Json> = gb::prune_time_from_tensor(p.csf_sealed(), t_start, t_end)
        .iter()
        .map(|e| tensor_entry_json(e.t, e.s, e.r, e.value))
        .collect();
    let mut o = Json::object();
    o.set("t_start", Json::uint(t_start));
    o.set("t_end", Json::uint(t_end));
    o.set("entries", Json::Array(entries));
    Response::ok(o.to_string())
}

/// `GET …/hop_ball?t_start=&t_end=&u=&v=&k=` — k-hop ball (#17, row 46).
fn hop_ball_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let (u, v, k) = match need_ball(query) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let ball = gb::hop_ball_from_tensor(p.csf_sealed(), t_start, t_end, u, v, k);
    let mut o = vector_json(&ball);
    o.set("t_start", Json::uint(t_start));
    o.set("t_end", Json::uint(t_end));
    o.set("u", Json::uint(u64::from(u.0)));
    o.set("v", Json::uint(u64::from(v.0)));
    o.set("k", Json::uint(k.0));
    Response::ok(o.to_string())
}

/// `GET …/prune_vertices?t_start=&t_end=&u=&v=&k=` — records whose endpoints
/// lie in the k-hop ball (#15, row 44).
fn prune_vertices_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let (u, v, k) = match need_ball(query) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let csf = p.csf_sealed();
    let ball = gb::hop_ball_from_tensor(csf, t_start, t_end, u, v, k);
    let windowed = gb::prune_time_from_tensor(csf, t_start, t_end);
    let out = gb::prune_vertices_window(&windowed, &ball);
    let entries: Vec<Json> = out
        .iter()
        .map(|e| tensor_entry_json(e.t, e.s, e.r, e.value))
        .collect();
    let mut o = Json::object();
    o.set("t_start", Json::uint(t_start));
    o.set("t_end", Json::uint(t_end));
    o.set("entries", Json::Array(entries));
    Response::ok(o.to_string())
}

/// `GET …/prune_edges?t_start=&t_end=&pair=&pair=` — records on eligible
/// pairs (#16, row 45).
fn prune_edges_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let allowed = match (query.contains("pair="), parse_pairs(query)) {
        (_, None) => return Response::bad_request("malformed pair parameter: expected pair=u:r"),
        (false, _) => return Response::bad_request("missing required parameter: pair"),
        (true, Some(allowed)) => allowed,
    };
    let csf = p.csf_sealed();
    let windowed = gb::prune_time_from_tensor(csf, t_start, t_end);
    let out = gb::prune_edges_window(&windowed, &allowed);
    let entries: Vec<Json> = out
        .iter()
        .map(|e| tensor_entry_json(e.t, e.s, e.r, e.value))
        .collect();
    let mut o = Json::object();
    o.set("t_start", Json::uint(t_start));
    o.set("t_end", Json::uint(t_end));
    o.set("entries", Json::Array(entries));
    Response::ok(o.to_string())
}

/// `GET …/prune_hops?t_start=&t_end=&u=&v=&k=` — records within k hops
/// (#18, row 47).
fn prune_hops_json(p: &IngestPipeline, query: &str) -> Response {
    let (t_start, t_end) = match need_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let (u, v, k) = match need_ball(query) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let out = gb::prune_hops_from_tensor(p.csf_sealed(), t_start, t_end, u, v, k);
    let entries: Vec<Json> = out
        .iter()
        .map(|e| tensor_entry_json(e.t, e.s, e.r, e.value))
        .collect();
    let mut o = Json::object();
    o.set("t_start", Json::uint(t_start));
    o.set("t_end", Json::uint(t_end));
    o.set("entries", Json::Array(entries));
    Response::ok(o.to_string())
}

/// Shared focal parameter bundle: `FocalTransaction` + `ContextParameters`
/// (#20/#21/#22; `ContextParameters.eligible` is `Some` exactly when at least
/// one `pair=` occurrence is present).
struct FocalCall {
    focal: FocalTransaction,
    params: abacus_tensor::graphblas::context::ContextParameters,
}

/// Require `epoch,s,r` plus the window and hop bound.
fn need_focal(query: &str) -> Result<FocalCall, Response> {
    let (t_start, t_end) = need_window(query)?;
    let k = need_u64(query, "k")
        .ok_or_else(|| Response::bad_request("missing required parameter: k"))?;
    let epoch = need_u64(query, "epoch")
        .ok_or_else(|| Response::bad_request("missing required parameter: epoch"))?;
    let s = need_u32(query, "s")
        .ok_or_else(|| Response::bad_request("missing required parameter: s"))?;
    let r = need_u32(query, "r")
        .ok_or_else(|| Response::bad_request("missing required parameter: r"))?;
    let self_loop = parse_self_loop(query)
        .ok_or_else(|| Response::bad_request("self_loop must be admit or exclude"))?;
    let eligible =
        if query.contains("pair=") {
            Some(parse_pairs(query).ok_or_else(|| {
                Response::bad_request("malformed pair parameter: expected pair=u:r")
            })?)
        } else {
            None
        };
    Ok(FocalCall {
        focal: FocalTransaction {
            epoch,
            sender: VertexId(s),
            receiver: VertexId(r),
        },
        params: abacus_tensor::graphblas::context::ContextParameters {
            window: TimeWindow {
                start: t_start,
                end: t_end,
            },
            hop_bound: HopBound(k),
            eligible,
            self_loop,
        },
    })
}

/// Echo the focal parameters back into a response object.
fn focal_echo(o: &mut Json, call: &FocalCall) {
    o.set("epoch", Json::uint(call.focal.epoch));
    o.set("s", Json::uint(u64::from(call.focal.sender.0)));
    o.set("r", Json::uint(u64::from(call.focal.receiver.0)));
    o.set("t_start", Json::uint(call.params.window.start));
    o.set("t_end", Json::uint(call.params.window.end));
    o.set("k", Json::uint(call.params.hop_bound.0));
}

/// Map an API-layer `ExecutionError` to a response. The classification is
/// structural only (Golden Source §16): absent focal goes to 404, invalid
/// window to 400, and value-domain overflow of the JSON-carried result to 400
/// with a closed-alphabet reason (never the offending amount).
fn focal_error(e: &abacus_tensor::graphblas::context::ExecutionError) -> Response {
    use abacus_tensor::graphblas::context::ExecutionError as E;
    match e {
        E::FocalAbsent => Response::not_found("focal coordinate is outside supp(T)"),
        E::FocalOutsideWindow => Response::bad_request("focal epoch outside window"),
        E::VolumeOverflow(_) => {
            Response::bad_request("value-domain overflow: result unrepresentable")
        }
    }
}

/// `GET …/focal_context?…` — the canonical focal context `N_f` (#20, row 49).
fn focal_context_json(p: &IngestPipeline, query: &str) -> Response {
    let call = match need_focal(query) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match gb::build_focal_context_from_tensor(p.csf_sealed(), call.focal, &call.params) {
        Ok(ctx) => {
            let admitted: Vec<Json> = ctx
                .admitted
                .iter()
                .map(|e| tensor_entry_json(e.t, e.s, e.r, e.value))
                .collect();
            let mut o = Json::object();
            focal_echo(&mut o, &call);
            o.set("admitted", Json::Array(admitted));
            o.set("vertices", vector_json(&ctx.vertices));
            o.set("a_one", matrix_json(&ctx.tensors.a_one));
            o.set("a_cnt", matrix_json(&ctx.tensors.a_cnt));
            o.set("a_vol", matrix_json(&ctx.tensors.a_vol));
            Response::ok(o.to_string())
        }
        Err(e) => focal_error(&e),
    }
}

/// `GET …/fan_in?…` — fan-in scalar + witness support (#21/#22, row 50).
fn fan_in_json(p: &IngestPipeline, query: &str) -> Response {
    let call = match need_focal(query) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match gb::compute_fan_in_from_tensor(p.csf_sealed(), call.focal, &call.params) {
        Ok(res) => {
            let mut o = Json::object();
            focal_echo(&mut o, &call);
            o.set("fan_in", Json::uint(res.fan_in.count));
            let counterparties: Vec<Json> = res
                .support_counterparties
                .members
                .iter()
                .map(|&m| Json::uint(u64::from(m)))
                .collect();
            o.set("support_counterparties", Json::Array(counterparties));
            Response::ok(o.to_string())
        }
        Err(e) => focal_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Primitive request-body decoders (POST). Fail-closed with `Err(&'static str)`
// structural reasons; never panics, never invents values.
// ---------------------------------------------------------------------------

/// Decode `{"entries":[{"s","r","value"}]}` into a `PairMatrix`. Rejects
/// duplicate coordinates (a matrix is a function `(s,r) -> v`; conflicting
/// duplicates are an invalid request, not a silent last-write-wins) and
/// rejects explicit zero values (zero is not in the support; P3).
fn decode_matrix(body: &Json) -> Result<PairMatrix, &'static str> {
    let arr = match body.get("entries") {
        Some(Json::Array(a)) => a,
        _ => return Err("missing required field: entries"),
    };
    let mut coords: Vec<(u32, u32, i64)> = Vec::with_capacity(arr.len());
    for item in arr {
        let s = item
            .get("s")
            .and_then(Json::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("entry field s must be a u32")?;
        let r = item
            .get("r")
            .and_then(Json::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("entry field r must be a u32")?;
        let value = item
            .get("value")
            .and_then(Json::as_i64)
            .ok_or("entry field value must be an i64")?;
        if value == 0 {
            return Err("entry value must be nonzero (zero is not in the support)");
        }
        coords.push((s, r, value));
    }
    coords.sort_unstable();
    let mut prev: Option<(u32, u32)> = None;
    for (s, r, _) in &coords {
        if prev == Some((*s, *r)) {
            return Err("duplicate coordinate in entries");
        }
        prev = Some((*s, *r));
    }
    let mut m = PairMatrix::zero();
    for (s, r, v) in coords {
        m.insert(s, r, v);
    }
    Ok(m)
}

/// Decode a `[u32]` member list into a sorted-distinct `Vector` (set
/// semantics: a vector pattern is a set; duplicates carry no information and
/// are dropped by definition — there is no multiplicity to preserve).
fn decode_vector(body: &Json, key: &str) -> Result<Vector, &'static str> {
    let arr = match body.get(key) {
        Some(Json::Array(a)) => a,
        _ => return Err("missing required array field"),
    };
    let mut members: Vec<u32> = Vec::with_capacity(arr.len());
    for item in arr {
        members.push(
            item.as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or("member must be a u32")?,
        );
    }
    members.sort_unstable();
    members.dedup();
    Ok(Vector::from_sorted(members))
}

/// Decode `{"pairs":[{"s","r"}]}` into sorted-distinct pairs (set semantics).
fn decode_pairs(body: &Json) -> Result<Vec<(u32, u32)>, &'static str> {
    let arr = match body.get("pairs") {
        Some(Json::Array(a)) => a,
        _ => return Err("missing required field: pairs"),
    };
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(arr.len());
    for item in arr {
        let s = item
            .get("s")
            .and_then(Json::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("pair field s must be a u32")?;
        let r = item
            .get("r")
            .and_then(Json::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("pair field r must be a u32")?;
        pairs.push((s, r));
    }
    pairs.sort_unstable();
    pairs.dedup();
    Ok(pairs)
}

/// Decode `{"values":[i64]}`.
fn decode_values(body: &Json) -> Result<Vec<i64>, &'static str> {
    let arr = match body.get("values") {
        Some(Json::Array(a)) => a,
        _ => return Err("missing required field: values"),
    };
    arr.iter()
        .map(|v| v.as_i64().ok_or("value must be an i64"))
        .collect()
}

// ---------------------------------------------------------------------------
// Stateless primitive handlers (POST #1–#13). The matrix/vector payloads come
// from the JSON body; the API-layer primitive is applied directly.
// ---------------------------------------------------------------------------

/// Parse the JSON request body; fail-closed 400 on malformed input.
fn parse_body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::bad_request("request body must be UTF-8 JSON"))?;
    parse_json(text).map_err(|_| Response::bad_request("malformed JSON request body"))
}

/// `POST …/primitive/extract` — `A(s, r)` (#1, row 32).
fn primitive_extract(body: &Json) -> Response {
    let m = match decode_matrix(body) {
        Ok(m) => m,
        Err(reason) => return Response::bad_request(reason),
    };
    let s = match body
        .get("s")
        .and_then(Json::as_u64)
        .and_then(|v| u32::try_from(v).ok())
    {
        Some(s) => s,
        None => return Response::bad_request("missing required parameter: s"),
    };
    let r = match body
        .get("r")
        .and_then(Json::as_u64)
        .and_then(|v| u32::try_from(v).ok())
    {
        Some(r) => r,
        None => return Response::bad_request("missing required parameter: r"),
    };
    let mut o = Json::object();
    match gb::extract(&m, s, r) {
        Some(value) => {
            o.set("found", Json::bool(true));
            o.set("value", Json::int(value));
        }
        None => {
            o.set("found", Json::bool(false));
            o.set("value", Json::Null);
        }
    }
    Response::ok(o.to_string())
}

/// `POST …/primitive/reduce_col_plus` — checked column sum (#2, row 33).
fn primitive_reduce_col_plus(body: &Json) -> Response {
    let m = match decode_matrix(body) {
        Ok(m) => m,
        Err(reason) => return Response::bad_request(reason),
    };
    let v = match body
        .get("v")
        .and_then(Json::as_u64)
        .and_then(|x| u32::try_from(x).ok())
    {
        Some(v) => v,
        None => return Response::bad_request("missing required parameter: v"),
    };
    match gb::reduce_col_plus(&m, v) {
        Ok(sum) => {
            let mut o = Json::object();
            o.set("sum", Json::int(sum));
            Response::ok(o.to_string())
        }
        Err(_) => Response::bad_request("value-domain overflow: column sum unrepresentable"),
    }
}

/// `POST …/primitive/reduce_row_plus` — checked row sum (#3, row 34).
fn primitive_reduce_row_plus(body: &Json) -> Response {
    let m = match decode_matrix(body) {
        Ok(m) => m,
        Err(reason) => return Response::bad_request(reason),
    };
    let u = match body
        .get("u")
        .and_then(Json::as_u64)
        .and_then(|x| u32::try_from(x).ok())
    {
        Some(u) => u,
        None => return Response::bad_request("missing required parameter: u"),
    };
    match gb::reduce_row_plus(&m, u) {
        Ok(sum) => {
            let mut o = Json::object();
            o.set("sum", Json::int(sum));
            Response::ok(o.to_string())
        }
        Err(_) => Response::bad_request("value-domain overflow: row sum unrepresentable"),
    }
}

/// `POST …/primitive/mxv` — senders of frontier receivers (#4, row 35).
fn primitive_mxv(body: &Json) -> Response {
    let m = match decode_matrix(body) {
        Ok(m) => m,
        Err(reason) => return Response::bad_request(reason),
    };
    let frontier = match decode_vector(body, "frontier") {
        Ok(f) => f,
        Err(_) => return Response::bad_request("missing required field: frontier"),
    };
    Response::ok(vector_json(&gb::mxv(&m, &frontier)).to_string())
}

/// `POST …/primitive/vxm` — receivers of frontier senders (#5, row 36).
fn primitive_vxm(body: &Json) -> Response {
    let m = match decode_matrix(body) {
        Ok(m) => m,
        Err(reason) => return Response::bad_request(reason),
    };
    let frontier = match decode_vector(body, "frontier") {
        Ok(f) => f,
        Err(_) => return Response::bad_request("missing required field: frontier"),
    };
    Response::ok(vector_json(&gb::vxm(&m, &frontier)).to_string())
}

/// `POST …/primitive/ewise_add_lor` — pattern union (#6, row 37).
fn primitive_ewise_add_lor(body: &Json) -> Response {
    let a = match decode_vector(body, "a") {
        Ok(a) => a,
        Err(_) => return Response::bad_request("missing required field: a"),
    };
    let b = match decode_vector(body, "b") {
        Ok(b) => b,
        Err(_) => return Response::bad_request("missing required field: b"),
    };
    Response::ok(vector_json(&gb::ewise_add_lor(&a, &b)).to_string())
}

/// `POST …/primitive/support` — `A_one = [A_cnt != 0]` (#7, row 38).
fn primitive_support(body: &Json) -> Response {
    let m = match decode_matrix(body) {
        Ok(m) => m,
        Err(reason) => return Response::bad_request(reason),
    };
    Response::ok(matrix_json(&gb::support_matrix(&m)).to_string())
}

/// `POST …/primitive/multiplicity` — `A_cnt` from pair occurrences
/// (#8, row 39).
fn primitive_multiplicity(body: &Json) -> Response {
    let pairs = match decode_pairs(body) {
        Ok(pairs) => pairs,
        Err(reason) => return Response::bad_request(reason),
    };
    Response::ok(matrix_json(&gb::multiplicity(&pairs)).to_string())
}

/// `POST …/primitive/volume` — `A_vol` from `(pair, amount)` occurrences
/// (#9, row 40).
fn primitive_volume(body: &Json) -> Response {
    let occ = match body.get("occurrences") {
        Some(Json::Array(a)) => a,
        _ => return Response::bad_request("missing required field: occurrences"),
    };
    let mut occurrences: Vec<((u32, u32), i64)> = Vec::with_capacity(occ.len());
    for item in occ {
        if !matches!(item, Json::Object(_)) {
            return Response::bad_request("occurrence must be an object");
        }
        let s = match item
            .get("s")
            .and_then(Json::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        {
            Some(s) => s,
            None => return Response::bad_request("occurrence field s must be a u32"),
        };
        let r = match item
            .get("r")
            .and_then(Json::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        {
            Some(r) => r,
            None => return Response::bad_request("occurrence field r must be a u32"),
        };
        let value = match item.get("value").and_then(Json::as_i64) {
            Some(value) => value,
            None => return Response::bad_request("occurrence field value must be an i64"),
        };
        occurrences.push(((s, r), value));
    }
    match gb::volume(&occurrences) {
        Ok(m) => Response::ok(matrix_json(&m).to_string()),
        Err(_) => Response::bad_request("value-domain overflow: volume sum unrepresentable"),
    }
}

/// `{"members":[u32]}` shared by `#10` (list form).
fn decode_values_for_count(body: &Json) -> Result<Vec<u32>, &'static str> {
    let arr = match body.get("members") {
        Some(Json::Array(a)) => a,
        _ => return Err("missing required field: members"),
    };
    arr.iter()
        .map(|v| {
            v.as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or("member must be a u32")
        })
        .collect()
}

/// `POST …/primitive/reduce_count` — cardinality (#10, row 41).
fn primitive_reduce_count(body: &Json) -> Response {
    let values = match decode_values_for_count(body) {
        Ok(values) => values,
        Err(reason) => return Response::bad_request(reason),
    };
    let mut o = Json::object();
    o.set("count", Json::uint(gb::reduce_count(&values)));
    Response::ok(o.to_string())
}

/// `POST …/primitive/reduce_count_set` — vertex-pattern cardinality
/// (#11, row 41).
fn primitive_reduce_count_set(body: &Json) -> Response {
    let s = match decode_vector(body, "members") {
        Ok(s) => s,
        Err(_) => return Response::bad_request("missing required field: members"),
    };
    let mut o = Json::object();
    o.set("count", Json::uint(gb::reduce_count_set(&s)));
    Response::ok(o.to_string())
}

/// `POST …/primitive/reduce_sum` — checked canonical sum (#12, row 42).
fn primitive_reduce_sum(body: &Json) -> Response {
    let values = match decode_values(body) {
        Ok(values) => values,
        Err(reason) => return Response::bad_request(reason),
    };
    match gb::reduce_sum(&values) {
        Ok(sum) => {
            let mut o = Json::object();
            o.set("sum", Json::int(sum));
            Response::ok(o.to_string())
        }
        Err(_) => Response::bad_request("value-domain overflow: sum unrepresentable"),
    }
}

/// `POST …/primitive/reduce_max` — canonical maximum (#13, row 42).
fn primitive_reduce_max(body: &Json) -> Response {
    let values = match decode_values(body) {
        Ok(values) => values,
        Err(reason) => return Response::bad_request(reason),
    };
    let mut o = Json::object();
    o.set("max", Json::int(gb::reduce_max(&values)));
    Response::ok(o.to_string())
}

// ---------------------------------------------------------------------------
// Sub-router entry point. Returns `None` when the path is outside the
// `/api/ingest/graphblas/` family (the caller falls through to its 404).
// ---------------------------------------------------------------------------

/// `POST /api/ingest/graphblas/seal` — compact the committed frontier into
/// the sealed canonical tensor (row 22). The only mutating route in this
/// family: it seals substrate state, never tensor algebra.
fn seal_json(engine: &Mutex<IngestPipeline>) -> Response {
    let mut p = engine.lock().unwrap();
    match p.seal_substrate() {
        Ok(sealed_nnz) => {
            let mut o = Json::object();
            o.set("sealed_nnz", Json::uint(sealed_nnz));
            Response::ok(o.to_string())
        }
        Err(e) => Response::bad_request(&e.to_string()),
    }
}

/// Dispatch one `/api/ingest/graphblas/*` request. Tensor-bound routes take
/// query parameters on GET; primitive routes take a JSON body on POST.
/// `405` when the path matches but the method does not; `404` when nothing
/// under this prefix matches.
pub(crate) fn graphblas_route(engine: &Mutex<IngestPipeline>, req: &Request) -> Option<Response> {
    let path = req.path.trim_end_matches('/');
    let rest = path.strip_prefix("/api/ingest/graphblas/")?;

    // Tensor-bound (GET).
    if req.method == "GET" {
        let p = engine.lock().unwrap();
        return Some(match rest {
            "contract_mode0" => contract_mode0_json(&p, &req.query),
            "temporal_multiplicity" => temporal_multiplicity_json(&p, &req.query),
            "support" => support_json(&p, &req.query),
            "prune_time" => prune_time_json(&p, &req.query),
            "hop_ball" => hop_ball_json(&p, &req.query),
            "prune_vertices" => prune_vertices_json(&p, &req.query),
            "prune_edges" => prune_edges_json(&p, &req.query),
            "prune_hops" => prune_hops_json(&p, &req.query),
            "focal_context" => focal_context_json(&p, &req.query),
            "fan_in" => fan_in_json(&p, &req.query),
            _ => Response::not_found("no such graphblas route"),
        });
    }

    // Stateless primitives + seal (POST, JSON body).
    if req.method == "POST" {
        if rest == "seal" {
            return Some(seal_json(engine));
        }
        let body = match parse_body(req) {
            Ok(body) => body,
            Err(resp) => return Some(resp),
        };
        return Some(match rest {
            "primitive/extract" => primitive_extract(&body),
            "primitive/reduce_col_plus" => primitive_reduce_col_plus(&body),
            "primitive/reduce_row_plus" => primitive_reduce_row_plus(&body),
            "primitive/mxv" => primitive_mxv(&body),
            "primitive/vxm" => primitive_vxm(&body),
            "primitive/ewise_add_lor" => primitive_ewise_add_lor(&body),
            "primitive/support" => primitive_support(&body),
            "primitive/multiplicity" => primitive_multiplicity(&body),
            "primitive/volume" => primitive_volume(&body),
            "primitive/reduce_count" => primitive_reduce_count(&body),
            "primitive/reduce_count_set" => primitive_reduce_count_set(&body),
            "primitive/reduce_sum" => primitive_reduce_sum(&body),
            "primitive/reduce_max" => primitive_reduce_max(&body),
            _ => Response::not_found("no such graphblas route"),
        });
    }

    Some(Response::json(
        405,
        "{\"error\":\"method not allowed\"}".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// In-module contract tests. They drive the same `route()` entry point the
// HTTP transport uses, so every status code and JSON shape asserted here is
// the one a client will observe. Golden Source: Addendum O rows 30–51 for
// the algebra, row 22 for sealing, row 27 for the read boundary.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest_pipeline::IngestPipeline;
    use abacus_ingest::features::FeaturePayload;
    use abacus_ingest::namespace::{EventIdentity, EventNamespace};
    use abacus_ingest::record::CanonicalRecord;

    fn ns(byte: u8) -> EventNamespace {
        EventNamespace::from_bytes([byte; 16])
    }

    fn rec(event_id: u64, s: u64, r: u64) -> CanonicalRecord {
        CanonicalRecord {
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

    /// Two committed events, then sealed so the tensor-bound routes serve them:
    /// `(t=1_700_000_000, s_code=1, r_code=2)` and `(t, s_code=3, r_code=1)`,
    /// each worth `12_345_000_000` micro-units (same values the `csf/*`
    /// endpoint tests pin).
    fn sealed_two() -> IngestPipeline {
        let mut p = IngestPipeline::new();
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100)
            .unwrap();
        p.seal_substrate().unwrap();
        p
    }

    fn wide() -> String {
        "t_start=0&t_end=2000000000".to_string()
    }

    fn get(path: &str, query: &str) -> Request {
        Request {
            method: "GET".to_string(),
            path: path.to_string(),
            query: query.to_string(),
            body: Vec::new(),
            connection_close: true,
        }
    }

    fn post(path: &str, body: &str) -> Request {
        Request {
            method: "POST".to_string(),
            path: path.to_string(),
            query: String::new(),
            body: body.as_bytes().to_vec(),
            connection_close: true,
        }
    }

    fn call(engine: &Mutex<IngestPipeline>, req: &Request) -> Response {
        super::graphblas_route(engine, req).expect("graphblas paths must dispatch")
    }

    fn body_text(resp: &Response) -> String {
        String::from_utf8(resp.body.clone()).expect("response bodies are UTF-8 JSON")
    }

    fn body_json(resp: &Response) -> Json {
        crate::web::json::parse(&body_text(resp)).expect("response body must parse as JSON")
    }

    fn as_u(obj: &Json, key: &str) -> Option<u64> {
        obj.get(key).and_then(Json::as_u64)
    }

    fn as_i(obj: &Json, key: &str) -> Option<i64> {
        obj.get(key).and_then(Json::as_i64)
    }

    fn entries_of(obj: &Json) -> Vec<&Json> {
        match obj.get("entries") {
            Some(Json::Array(a)) => a.iter().collect(),
            other => panic!("entries array missing: {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Tensor-bound routes (rows 30/31/43–50) over the sealed tensor.
    // ------------------------------------------------------------------

    #[test]
    fn seal_compacts_frontier_then_graphblas_serves_sealed() {
        let mut p = IngestPipeline::new();
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100)
            .unwrap();
        // Unsealed: the tensor-bound family sees nothing (frontier excluded).
        let engine = Mutex::new(p);
        let resp = call(&engine, &get("/api/ingest/graphblas/prune_time", &wide()));
        assert_eq!(resp.status, 200);
        assert_eq!(entries_of(&body_json(&resp)).len(), 0);
        // Seal through the endpoint itself.
        let resp = call(&engine, &post("/api/ingest/graphblas/seal", ""));
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_u(&body_json(&resp), "sealed_nnz"), Some(1));
        // Now the sealed tensor serves the event.
        let resp = call(&engine, &get("/api/ingest/graphblas/prune_time", &wide()));
        assert_eq!(resp.status, 200);
        let decoded = body_json(&resp);
        let entries = entries_of(&decoded);
        assert_eq!(entries.len(), 1);
        assert_eq!(as_i(entries[0], "value"), Some(12_345_000_000));
    }

    #[test]
    fn contract_mode0_returns_window_sums() {
        let engine = Mutex::new(sealed_two());
        let resp = call(
            &engine,
            &get("/api/ingest/graphblas/contract_mode0", &wide()),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        let j = body_json(&resp);
        assert_eq!(as_u(&j, "nnz"), Some(2));
        let mut got: Vec<(u64, u64, i64)> = entries_of(&j)
            .iter()
            .map(|e| {
                (
                    as_u(e, "s").unwrap(),
                    as_u(e, "r").unwrap(),
                    as_i(e, "value").unwrap(),
                )
            })
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![(1, 2, 12_345_000_000), (3, 1, 12_345_000_000)]);
    }

    #[test]
    fn temporal_multiplicity_counts_pairs() {
        let engine = Mutex::new(sealed_two());
        let resp = call(
            &engine,
            &get("/api/ingest/graphblas/temporal_multiplicity", &wide()),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        let j = body_json(&resp);
        assert_eq!(as_u(&j, "nnz"), Some(2));
        for e in entries_of(&j) {
            assert_eq!(as_i(e, "value"), Some(1));
        }
    }

    #[test]
    fn support_returns_triple() {
        let engine = Mutex::new(sealed_two());
        let resp = call(&engine, &get("/api/ingest/graphblas/support", &wide()));
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        let j = body_json(&resp);
        for key in ["a_one", "a_cnt", "a_vol"] {
            let m = j.get(key).unwrap_or_else(|| panic!("{key} missing"));
            assert_eq!(as_u(m, "nnz"), Some(2), "{key}");
        }
        let vol = j.get("a_vol").unwrap();
        let mut vals: Vec<i64> = entries_of(vol)
            .iter()
            .map(|e| as_i(e, "value").unwrap())
            .collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![12_345_000_000, 12_345_000_000]);
    }

    fn ball_query(u: u32, v: u32, k: u64) -> String {
        format!("{}&u={u}&v={v}&k={k}", wide(), u = u, v = v, k = k)
    }

    // Account codes in `sealed_two()`: wire 100 -> code 1, 200 -> code 2,
    // 300 -> code 3 (shared namespace interns in first-seen order).
    const S1: u32 = 1;
    const R2: u32 = 2;
    const S3: u32 = 3;
    const R1: u32 = 1;
    const T: u64 = 1_700_000_000;

    fn focal_query() -> String {
        format!("{}&k=2&epoch={T}&s={S1}&r={R2}", wide())
    }

    #[test]
    fn hop_ball_contains_focal_endpoints() {
        let engine = Mutex::new(sealed_two());
        let resp = call(
            &engine,
            &get("/api/ingest/graphblas/hop_ball", &ball_query(S1, R2, 2)),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        let j = body_json(&resp);
        let members: Vec<u64> = match j.get("members") {
            Some(Json::Array(a)) => a.iter().filter_map(Json::as_u64).collect(),
            other => panic!("members missing: {other:?}"),
        };
        assert!(members.contains(&u64::from(S1)), "{members:?}");
        assert!(members.contains(&u64::from(R2)), "{members:?}");
    }

    #[test]
    fn prune_vertices_edges_hops_agree_on_window() {
        let engine = Mutex::new(sealed_two());
        let q = ball_query(S1, R2, 2);
        let v = call(&engine, &get("/api/ingest/graphblas/prune_vertices", &q));
        let h = call(&engine, &get("/api/ingest/graphblas/prune_hops", &q));
        assert_eq!(v.status, 200, "{}", body_text(&v));
        assert_eq!(h.status, 200, "{}", body_text(&h));
        // Both events touch the 2-hop ball around (1,2): same two records.
        assert_eq!(entries_of(&body_json(&v)).len(), 2);
        assert_eq!(entries_of(&body_json(&h)).len(), 2);
        // Eligible-pair pruning keeps only the admitted pair.
        let e = call(
            &engine,
            &get(
                "/api/ingest/graphblas/prune_edges",
                &format!("{}&pair=1:2", wide()),
            ),
        );
        assert_eq!(e.status, 200, "{}", body_text(&e));
        let decoded = body_json(&e);
        let kept = entries_of(&decoded);
        assert_eq!(kept.len(), 1);
        assert_eq!(as_u(kept[0], "s"), Some(1));
        assert_eq!(as_u(kept[0], "r"), Some(2));
    }

    #[test]
    fn focal_context_and_fan_in_echo_and_count() {
        let engine = Mutex::new(sealed_two());
        let resp = call(
            &engine,
            &get("/api/ingest/graphblas/focal_context", &focal_query()),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        let j = body_json(&resp);
        assert_eq!(as_u(&j, "s"), Some(u64::from(S1)));
        assert_eq!(as_u(&j, "r"), Some(u64::from(R2)));
        let admitted = match j.get("admitted") {
            Some(Json::Array(a)) => a.len(),
            other => panic!("admitted missing: {other:?}"),
        };
        assert_eq!(admitted, 2);
        let resp = call(
            &engine,
            &get("/api/ingest/graphblas/fan_in", &focal_query()),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        let j = body_json(&resp);
        // #21 incoming_support: senders into the focal receiver 2 = {1}.
        assert_eq!(as_u(&j, "fan_in"), Some(1));
        let cp: Vec<u64> = match j.get("support_counterparties") {
            Some(Json::Array(a)) => a.iter().filter_map(Json::as_u64).collect(),
            other => panic!("support_counterparties missing: {other:?}"),
        };
        assert_eq!(cp, vec![1]);
    }

    #[test]
    fn focal_errors_distinguish_absent_and_window() {
        let engine = Mutex::new(sealed_two());
        // Coordinate (9,9) is outside supp(T) -> 404 (A3).
        let resp = call(
            &engine,
            &get(
                "/api/ingest/graphblas/focal_context",
                &format!("{}&k=2&epoch={T}&s=9&r=9", wide()),
            ),
        );
        assert_eq!(resp.status, 404, "{}", body_text(&resp));
        let resp = call(
            &engine,
            &get(
                "/api/ingest/graphblas/fan_in",
                &format!("{}&k=2&epoch={T}&s=9&r=9", wide()),
            ),
        );
        assert_eq!(resp.status, 404, "{}", body_text(&resp));
        // Focal epoch outside the window -> 400.
        let resp = call(
            &engine,
            &get(
                "/api/ingest/graphblas/focal_context",
                &format!("t_start=0&t_end=5&k=2&epoch={T}&s={S1}&r={R2}"),
            ),
        );
        assert_eq!(resp.status, 400, "{}", body_text(&resp));
        // Missing parameters -> 400 (never a silent default).
        let resp = call(
            &engine,
            &get("/api/ingest/graphblas/contract_mode0", "t_start=0"),
        );
        assert_eq!(resp.status, 400);
        let resp = call(&engine, &get("/api/ingest/graphblas/hop_ball", &wide()));
        assert_eq!(resp.status, 400);
        let resp = call(&engine, &get("/api/ingest/graphblas/prune_edges", &wide()));
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn unknown_graphblas_paths_and_methods() {
        let engine = Mutex::new(sealed_two());
        let resp = call(&engine, &get("/api/ingest/graphblas/nope", ""));
        assert_eq!(resp.status, 404);
        let resp = call(&engine, &get("/api/ingest/graphblas/primitive/extract", ""));
        assert_eq!(resp.status, 404);
        // Primitive route on GET is a method mismatch -> 405.
        let resp = Request {
            method: "PUT".to_string(),
            path: "/api/ingest/graphblas/support".to_string(),
            query: wide(),
            body: Vec::new(),
            connection_close: true,
        };
        assert_eq!(call(&engine, &resp).status, 405);
    }

    // ------------------------------------------------------------------
    // Stateless primitive routes (#1–#13).
    // ------------------------------------------------------------------

    const MAT: &str = r#"{"entries":[{"s":1,"r":2,"value":100},{"s":1,"r":3,"value":200},{"s":2,"r":3,"value":50}]}"#;

    #[test]
    fn primitive_extract_reports_absent_never_zero() {
        let engine = Mutex::new(sealed_two());
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/extract",
                r#"{"entries":[{"s":1,"r":2,"value":100}],"s":1,"r":2}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert!(body_text(&resp).contains("\"found\":true"));
        assert_eq!(as_i(&body_json(&resp), "value"), Some(100));
        // Absent coordinate reports found:false (P3 — never fabricated as 0).
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/extract",
                r#"{"entries":[{"s":1,"r":2,"value":100}],"s":9,"r":9}"#,
            ),
        );
        assert_eq!(resp.status, 200);
        assert!(body_text(&resp).contains("\"found\":false"));
    }

    #[test]
    fn primitive_reductions_sum_columns_rows_values() {
        let engine = Mutex::new(sealed_two());
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_col_plus",
                r#"{"entries":[{"s":1,"r":2,"value":100},{"s":2,"r":2,"value":50}],"v":2}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_i(&body_json(&resp), "sum"), Some(150));
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_row_plus",
                r#"{"entries":[{"s":1,"r":2,"value":100},{"s":1,"r":3,"value":200}],"u":1}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_i(&body_json(&resp), "sum"), Some(300));
        // Absent column/row goes to Ok(0).
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_col_plus",
                r#"{"entries":[{"s":1,"r":2,"value":100}],"v":9}"#,
            ),
        );
        assert_eq!(resp.status, 200);
        assert_eq!(as_i(&body_json(&resp), "sum"), Some(0));
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_sum",
                r#"{"values":[10,20,-5]}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_i(&body_json(&resp), "sum"), Some(25));
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_max",
                r#"{"values":[10,20,-5]}"#,
            ),
        );
        assert_eq!(resp.status, 200);
        assert_eq!(as_i(&body_json(&resp), "max"), Some(20));
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_max",
                r#"{"values":[]}"#,
            ),
        );
        assert_eq!(resp.status, 200);
        assert_eq!(as_i(&body_json(&resp), "max"), Some(0));
    }

    #[test]
    fn primitive_mxv_vxm_ewise_traverse_patterns() {
        let engine = Mutex::new(sealed_two());
        // mxv: senders of frontier receiver {2} -> {1}.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/mxv",
                r#"{"entries":[{"s":1,"r":2,"value":100},{"s":1,"r":3,"value":200}],"frontier":[2]}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_u(&body_json(&resp), "nnz"), Some(1));
        // vxm: receivers of frontier sender {1} -> {2,3}.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/vxm",
                r#"{"entries":[{"s":1,"r":2,"value":100},{"s":1,"r":3,"value":200}],"frontier":[1]}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_u(&body_json(&resp), "nnz"), Some(2));
        // ewise_add_lor: union of {1,3} and {2,3} -> {1,2,3}.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/ewise_add_lor",
                r#"{"a":[1,3],"b":[2,3]}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_u(&body_json(&resp), "nnz"), Some(3));
    }

    #[test]
    fn primitive_matrix_builders_count_and_volume() {
        let engine = Mutex::new(sealed_two());
        // multiplicity: (1,2)x2, (2,3)x1 -> nnz 2.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/multiplicity",
                r#"{"pairs":[{"s":1,"r":2},{"s":1,"r":2},{"s":2,"r":3}]}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_u(&body_json(&resp), "nnz"), Some(2));
        // volume: (1,2)x(100+50) -> nnz 1.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/volume",
                r#"{"occurrences":[{"s":1,"r":2,"value":100},{"s":1,"r":2,"value":50}]}"#,
            ),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_u(&body_json(&resp), "nnz"), Some(1));
        // support: existence map of MAT -> nnz 3.
        let resp = call(
            &engine,
            &post("/api/ingest/graphblas/primitive/support", MAT),
        );
        assert_eq!(resp.status, 200, "{}", body_text(&resp));
        assert_eq!(as_u(&body_json(&resp), "nnz"), Some(3));
        // reduce_count / reduce_count_set -> 3.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_count",
                r#"{"members":[1,2,3]}"#,
            ),
        );
        assert_eq!(resp.status, 200);
        assert_eq!(as_u(&body_json(&resp), "count"), Some(3));
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_count_set",
                r#"{"members":[1,2,3]}"#,
            ),
        );
        assert_eq!(resp.status, 200);
        assert_eq!(as_u(&body_json(&resp), "count"), Some(3));
    }

    #[test]
    fn primitive_fail_closed_on_overflow_duplicates_zero_malformed() {
        let engine = Mutex::new(sealed_two());
        // i64::MAX + 1 overflows the exact value domain (P2) -> 400.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_sum",
                r#"{"values":[9223372036854775807,1]}"#,
            ),
        );
        assert_eq!(resp.status, 400, "{}", body_text(&resp));
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/volume",
                r#"{"occurrences":[{"s":1,"r":2,"value":9223372036854775807},{"s":1,"r":2,"value":1}]}"#,
            ),
        );
        assert_eq!(resp.status, 400, "{}", body_text(&resp));
        // Duplicate coordinate is an invalid request, not silent last-write.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/extract",
                r#"{"entries":[{"s":1,"r":2,"value":100},{"s":1,"r":2,"value":50}],"s":1,"r":2}"#,
            ),
        );
        assert_eq!(resp.status, 400, "{}", body_text(&resp));
        // Explicit zero value is not in the support (P3).
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/extract",
                r#"{"entries":[{"s":1,"r":2,"value":0}],"s":1,"r":2}"#,
            ),
        );
        assert_eq!(resp.status, 400, "{}", body_text(&resp));
        // Malformed JSON body -> 400.
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_sum",
                r#"{"values":[}"#,
            ),
        );
        assert_eq!(resp.status, 400);
        // Missing field -> 400.
        let resp = call(
            &engine,
            &post("/api/ingest/graphblas/primitive/reduce_sum", r#"{}"#),
        );
        assert_eq!(resp.status, 400);
        // Errors carry no amounts or account identifiers (§16 — structural).
        let resp = call(
            &engine,
            &post(
                "/api/ingest/graphblas/primitive/reduce_sum",
                r#"{"values":[9223372036854775807,1]}"#,
            ),
        );
        assert!(!body_text(&resp).contains("9223372036854775807"));
    }
}
