//! `web::graph` — the shared, data-only network value types used by the
//! ingestion-backed graph views (`web::ingest_graph`).
//!
//! The former `Pipeline`-based network builders (over the retired SAMCORE02 /
//! role-separated model) were removed in the CSV/Kafka ingestion cutover; the
//! ingestion-native builders live in `web::ingest_graph` and key accounts
//! through the single shared Account Map namespace. These types are retained so
//! the frontend `GraphView` consumes one shape for both surfaces.
//!
//! NO amounts appear in any node or edge — identifiers, timestamps and hop depth
//! only.

/// One graph node. `id` is a stable string key the frontend uses to place and
/// connect nodes. `kind` distinguishes transaction vs account nodes.
#[derive(Debug, Clone, Default)]
pub struct GraphNode {
    /// Stable node key (e.g. `"tx:123"` or `"acct:ACC1"`).
    pub id: String,
    /// `"transaction"` or `"account"`.
    pub kind: &'static str,
    /// Human label (locator id or account identifier).
    pub label: String,
    /// Event time `t` for transaction nodes; `None` for account nodes.
    pub t: Option<u64>,
    /// True for the selected transaction's node (highlight, task §6).
    pub selected: bool,
    /// Hop depth (0-3) for transaction nodes; `None` for account nodes.
    pub hop: Option<u8>,
}

/// One directed graph edge. For the Transaction Network each edge is one
/// transaction (account -> tx or tx -> account). For the Account Network each
/// edge is one account->account relationship aggregated across its supporting
/// transactions, so `txs` lists every transaction locator that evidences it
/// (task §6: "clicking a relationship may reveal the transactions supporting
/// it"). Carries only identifiers/timestamps — NO amounts.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Supporting transaction locators (>= 1). Empty is never emitted.
    pub txs: Vec<u64>,
}

/// A complete structural network: nodes + directed edges + a truncation flag
/// plus the committed-record count the view was built from (provenance, so the
/// UI can detect that new transactions have been ingested since the last
/// fetch and offer a refresh — task §6 follow-up).
#[derive(Debug, Clone)]
pub struct Network {
    /// Nodes (deterministic order).
    pub nodes: Vec<GraphNode>,
    /// Directed edges (deterministic order).
    pub edges: Vec<GraphEdge>,
    /// True when a directional lookup hit the `MAX_EDGES_PER_LOOKUP` cap and
    /// the neighborhood is therefore incomplete (task §6: report, don't hide).
    pub truncated: bool,
    /// Committed record count at build time. A larger value on a later fetch
    /// means new transactions may now be inside this network's neighborhood.
    pub committed: u64,
}
