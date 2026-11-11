//! Contract: K=2 focal transaction context (Phase 11, Amendment A1;
//! owner authority 2026-10-19 admits the Φ FeatureVector parallel, F13).
//!
//! Implementation-independent statement of the input, output, temporal
//! semantics, hop semantics, size limits, and correctness guarantees that
//! BOTH the reference (`build_focal_context`, K=2) and the bounded
//! endpoint-driven (`build_bounded_k2`) paths satisfy. Tests in
//! `tests/bounded_context.rs` derive from each numbered clause.

/// C1 FOCAL: (locator, t_f, s_f, r_f). Committed before query; duplicate
/// delivery returns Duplicate with no state change (identity ledger,
/// Canonical Contract 3); idempotent delta via ledger CAS (d_cas, P7).
/// C2 WINDOW: W = [t_f - 604800, t_f + 1) half-open (P1). t < start and
/// t > t_f excluded; t == start included; focal included (end = t_f + 1).
/// C3 HOPS (K=2, Amendment A1): V0 = {s_f, r_f}; V_{k+1} = N(V_k) where N
/// follows BOTH directions (senders of frontier receivers + receivers of
/// frontier senders, LOR_LAND). Hop-1 = N(V0) \\ V0; hop-2 = N(V1) \\ ball.
/// Cycles/self-loops/repeats: visited-set semantics, each vertex once;
/// self-loop edges admitted only under Admit; no temporal ordering along
/// paths (window membership only). Vertex set (ball) vs edge set (admitted
/// tx) distinguished: ball constrains, admission filters.
/// C4 OUTPUT (minimum reusable by all 7 typology evaluators): admitted
/// (t,s,r,value) in prec_3 order, deduped; directed pairs preserved; ball
/// vertex set sorted-distinct; support tensors (A_one, A_cnt, A_vol) over
/// admitted; FeatureVector per admitted record, index-aligned with `admitted`
/// (log_magnitude, relative_threshold_distance, roundedness_flags — carried
/// verbatim from the commit-time derivation, GS §9, never recomputed in the
/// context; tensor-only reference constructors leave the list empty since
/// the canonical tensor stores exact micro-units, not derived metrics);
/// stable order + dedup guaranteed; records resolved from the
/// tensor/indexes (never fabricated, P3), values exact micro-units (P2).
/// C5 BOUNDS: active hot set <= 32 KiB (64 KiB L1D at 50%); vertices <=
/// 1024; admitted <= 1024. Overflow -> deterministic Overflow error
/// (VertexOverflow/AdmittedOverflow), never silent truncation; fallback =
/// unbounded reference path with completeness reported by its output.
/// C6 ARCH: reuse CSR forward / CSC backward / CSF / set ops; no duplicate
/// persistent index (HCSF 2); typology-independent; one build serves all 7.
pub const OMEGA_SECS: u64 = 604800;
pub const HOP_BOUND: u64 = 2;
