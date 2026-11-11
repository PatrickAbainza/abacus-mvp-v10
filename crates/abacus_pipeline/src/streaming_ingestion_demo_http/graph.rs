//! `web::graph` — structural network views derived from the canonical tensor
//! and its indexes (your_task_phase_4.md §6). NO graph database, NO amount
//! exposure, NO re-materialization of the tensor.
//!
//! Golden-source grounding:
//! - Both views are DERIVED from the authoritative sources via the existing
//!   `abacus_pipeline` retrieval APIs — the sealed CSF + frontier overlay and
//!   the phase-3 forward/backward/directory indexes (`retrieval.rs`). They are
//!   read-only access paths, never a second canonical store (hierarchical
//!   invariant §8/§10: no second canonical index).
//! - Temporal/traversal limits are the existing ones: relative windows use
//!   `DEFAULT_DELTA_SECS` (seven days) and every directional lookup is capped
//!   by `MAX_EDGES_PER_LOOKUP` (2,500) — the same contract the phase-3
//!   retrieval APIs enforce. Truncation is reported honestly, never silent.
//! - Amounts are NEVER read into or emitted from these views (task §6).
//! - Determinism (P12/R7): nodes and edges are produced in a fixed order
//!   (locator/`(t,locator)` ascending), so the same committed state always
//!   yields the same graph.

use crate::pipeline::Pipeline;
use crate::retrieval::{DEFAULT_DELTA_SECS, MAX_EDGES_PER_LOOKUP, EventLocator};
use std::collections::{BTreeMap, BTreeSet};

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

/// Resolve a locator to its authoritative `(t, sender_name, receiver_name)`
/// through the phase-3 locator directory + dictionaries. `None` if the locator
/// is unknown (never fabricated, P3).
fn resolve_names(p: &Pipeline, loc: EventLocator) -> Option<(u64, String, String)> {
    let r = p.snapshot().temporal.resolve(loc)?;
    let s = p.snapshot().senders.get(r.sender as usize)?.clone();
    let v = p.snapshot().receivers.get(r.receiver as usize)?.clone();
    Some((r.t, s, v))
}

/// Transaction Network for a selected transaction (task §6, §9 tests 11/13):
/// the selected transaction highlighted, plus its sender- and receiver-side
/// neighbors within the seven-day relative window, displayed as transaction
/// nodes with directed account connections. NO amounts.
///
/// The neighborhood is the union of the selected transaction, the sender's
/// other outgoing events in `[t, t+delta)` (forward index) and the receiver's
/// other incoming events in `[t-delta, t)` (backward index), each capped by
/// `MAX_EDGES_PER_LOOKUP` (existing traversal limit). This keeps the view
/// bounded and centered on the selection without loading the whole dataset's
/// network into the browser (task §6).
#[must_use]
pub fn transaction_network(p: &Pipeline, id: u64) -> Option<Network> {
    let sel = EventLocator(u32::try_from(id).ok()?);
    let (t, s_name, r_name) = resolve_names(p, sel)?;

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut truncated = false;

    // Account nodes for the two endpoints of the selected transaction.
    let s_acct = format!("acct:{s_name}");
    let r_acct = format!("acct:{r_name}");
    nodes.push(GraphNode {
        id: s_acct.clone(),
        kind: "account",
        label: s_name.clone(),
        t: None,
        selected: false,
        ..Default::default()
    });
    nodes.push(GraphNode {
        id: r_acct.clone(),
        kind: "account",
        label: r_name.clone(),
        t: None,
        selected: false,
        ..Default::default()
    });
    // The selected transaction node, highlighted.
    let sel_tx = format!("tx:{id}");
    nodes.push(GraphNode {
        id: sel_tx.clone(),
        kind: "transaction",
        label: format!("tx #{id}"),
        t: Some(t),
        selected: true,
        ..Default::default()
    });
    edges.push(GraphEdge { from: s_acct.clone(), to: sel_tx.clone(), txs: vec![id] });

    // Sender's outgoing neighborhood in [t, t+delta).
    let out =
        p.get_outgoing_in_range(&s_name, t, t.saturating_add(DEFAULT_DELTA_SECS), MAX_EDGES_PER_LOOKUP);
    truncated |= out.is_truncated;
    for loc in out.locators.iter() {
        if loc.0 == u32::try_from(id).unwrap_or(u32::MAX) {
            continue; // already the selected node
        }
        if let Some((nt, ns, _nr)) = resolve_names(p, *loc) {
            let nid = format!("tx:{}", loc.0);
            nodes.push(GraphNode {
                id: nid.clone(),
                kind: "transaction",
                label: format!("tx #{}", loc.0),
                t: Some(nt),
                selected: false,
                hop: None,
            });
            edges.push(GraphEdge {
                from: format!("acct:{ns}"),
                to: nid,
                txs: vec![u64::from(loc.0)],
            });
        }
    }

    // Receiver's incoming neighborhood in [t-delta, t).
    let lo = t.saturating_sub(DEFAULT_DELTA_SECS);
    let inc = p.get_incoming_in_range(&r_name, lo, t, MAX_EDGES_PER_LOOKUP);
    truncated |= inc.is_truncated;
    for loc in inc.locators.iter() {
        if loc.0 == u32::try_from(id).unwrap_or(u32::MAX) {
            continue;
        }
        if let Some((nt, ns, _nr)) = resolve_names(p, *loc) {
            let nid = format!("tx:{}", loc.0);
            nodes.push(GraphNode {
                id: nid.clone(),
                kind: "transaction",
                label: format!("tx #{}", loc.0),
                t: Some(nt),
                selected: false,
                hop: None,
            });
            edges.push(GraphEdge {
                from: nid,
                to: format!("acct:{ns}"),
                txs: vec![u64::from(loc.0)],
            });
        }
    }

    let committed = p.snapshot().committed_records as u64;
    Some(Network { nodes, edges, truncated, committed })
}


/// Account Network for a selected transaction (task §6, §9 tests 12/13): the
/// sender and receiver accounts of the selected transaction as highlighted
/// account nodes, plus directed account->account relationships derived from the
/// supporting transactions in the seven-day window. Each relationship carries
/// the supporting transaction locators so a click can reveal them (task §6).
/// NO amounts.
#[must_use]
pub fn account_network(p: &Pipeline, id: u64) -> Option<Network> {
    let sel = EventLocator(u32::try_from(id).ok()?);
    let (t, s_name, r_name) = resolve_names(p, sel)?;

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    // Deduplicate account nodes by id, first-seen order.
    fn push_acct(name: &str, selected: bool, nodes: &mut Vec<GraphNode>, seen: &mut BTreeSet<String>) {
        let key = format!("acct:{name}");
        if seen.insert(key.clone()) {
            nodes.push(GraphNode {
                id: key,
                kind: "account",
                label: name.to_string(),
                t: None,
                selected,
                ..Default::default()
            });
        }
    }
    push_acct(&s_name, true, &mut nodes, &mut seen);
    push_acct(&r_name, true, &mut nodes, &mut seen);

    // Aggregate account->account relationships: key (from,to) -> supporting
    // transaction locators, in first-seen order (deterministic, task §6).
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut edge_idx: BTreeMap<(String, String), usize> = BTreeMap::new();
    let add_edge = |from: &str, to: &str, tx: u64, edges: &mut Vec<GraphEdge>, idx: &mut BTreeMap<(String, String), usize>| {
        let key = (from.to_string(), to.to_string());
        if let Some(&i) = idx.get(&key) {
            edges[i].txs.push(tx);
        } else {
            idx.insert(key, edges.len());
            edges.push(GraphEdge { from: from.to_string(), to: to.to_string(), txs: vec![tx] });
        }
    };
    let mut truncated = false;

    // Outgoing account relationships from the sender.
    let out =
        p.get_outgoing_in_range(&s_name, t, t.saturating_add(DEFAULT_DELTA_SECS), MAX_EDGES_PER_LOOKUP);
    truncated |= out.is_truncated;
    for loc in out.locators.iter() {
        if let Some((_nt, _ns, nr)) = resolve_names(p, *loc) {
            push_acct(&nr, false, &mut nodes, &mut seen);
            add_edge(&s_name, &nr, u64::from(loc.0), &mut edges, &mut edge_idx);
        }
    }

    // Incoming account relationships to the receiver.
    let lo = t.saturating_sub(DEFAULT_DELTA_SECS);
    let inc = p.get_incoming_in_range(&r_name, lo, t, MAX_EDGES_PER_LOOKUP);
    truncated |= inc.is_truncated;
    for loc in inc.locators.iter() {
        if let Some((_nt, ns, _nr)) = resolve_names(p, *loc) {
            push_acct(&ns, false, &mut nodes, &mut seen);
            add_edge(&ns, &r_name, u64::from(loc.0), &mut edges, &mut edge_idx);
        }
    }

    let committed = p.snapshot().committed_records as u64;
    Some(Network { nodes, edges, truncated, committed })
}

/// Minimum builder so the hop expansion does not capture `nodes` mutably from
/// two closure bodies (which the borrow checker rejects).
struct Builder<'a> {
    nodes: &'a mut Vec<GraphNode>,
    seen_acct: &'a mut BTreeSet<String>,
    seen_tx: &'a mut BTreeSet<EventLocator>,
}

impl<'a> Builder<'a> {
    fn push_account(&mut self, name: &str, selected: bool) -> bool {
        let key = format!("acct:{name}");
        if self.seen_acct.insert(key.clone()) {
            self.nodes.push(GraphNode {
                id: key,
                kind: "account",
                label: name.to_string(),
                t: None,
                selected,
                ..Default::default()
            });
            true
        } else {
            false
        }
    }

    fn push_transaction(
        &mut self,
        loc: EventLocator,
        selected: bool,
        hop: u8,
        p: &Pipeline,
    ) -> bool {
        if self.seen_tx.insert(loc) {
            if let Some(res) = p.snapshot().temporal.resolve(loc) {
                self.nodes.push(GraphNode {
                    id: format!("tx:{}", res.locator.0),
                    kind: "transaction",
                    label: format!("tx #{}", res.locator.0),
                    t: Some(res.t),
                    selected,
                    hop: Some(hop),
                });
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}


/// Tensor network for a selected transaction (task §6 extension): shows the
/// selected transaction placed on the CSF tensor fiber `t -> s -> r` (its
/// sender row + receiver column) and the multi-hop relationships up to
/// `MAX_HOPS` (3) through shared accounts.
///
/// Hop 0 is the selected transaction itself. Hop 1 is the sender's forward
/// neighborhood plus the receiver's backward neighborhood. Hops 2 and 3 repeat
/// the same expansion from the previous hop's transactions. Every directional
/// lookup is bounded to the seven-day relative window (`DEFAULT_DELTA_SECS`)
/// and capped by `MAX_EDGES_PER_LOOKUP`.
///
/// Accounts carry `hop: None`; transaction nodes carry their hop depth.
#[must_use]
pub fn tensor_network(p: &Pipeline, id: u64, k: u8) -> Option<Network> {
    let sel = EventLocator(u32::try_from(id).ok()?);
    let (t, s_name, r_name) = resolve_names(p, sel)?;

    let delta = DEFAULT_DELTA_SECS;
    let lo = t.wrapping_sub(delta);
    let hi = t.wrapping_add(delta);

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut truncated = false;

    // Deduplication: account keys and event locators.
    let mut seen_acct: BTreeSet<String> = BTreeSet::new();
    let mut seen_tx: BTreeSet<EventLocator> = BTreeSet::new();

    let mut builder = Builder {
        nodes: &mut nodes,
        seen_acct: &mut seen_acct,
        seen_tx: &mut seen_tx,
    };

    // Hop 0: selected transaction + its CSF fiber (sender/receiver) accounts.
    builder.push_account(&s_name, false);
    builder.push_account(&r_name, false);
    builder.push_transaction(sel, true, 0, p);

    edges.push(GraphEdge {
        from: format!("acct:{s_name}"),
        to: format!("tx:{}", sel.0),
        txs: vec![u64::from(sel.0)],
    });
    edges.push(GraphEdge {
        from: format!("tx:{}", sel.0),
        to: format!("acct:{r_name}"),
        txs: vec![u64::from(sel.0)],
    });

    // Hops 1..=k: expand through shared sender/receiver accounts.
    let mut frontier = vec![sel];
    for hop in 1..=k {
        let mut next_frontier: Vec<EventLocator> = Vec::new();
        for &loc in &frontier {
            let (_frm, ns, nr) = match resolve_names(p, loc) {
                Some(x) => x,
                None => continue,
            };

            // Forward: sender's outgoing events in [t, t + delta).
            let out =
                p.get_outgoing_in_range(&ns, t, hi, MAX_EDGES_PER_LOOKUP);
            if out.is_truncated {
                truncated = true;
            }
            for &ql in out.locators.iter() {
                if builder.push_transaction(ql, false, hop, p) {
                    next_frontier.push(ql);
                    if let Some((_qt, ns2, nr2)) = resolve_names(p, ql) {
                        builder.push_account(&ns2, false);
                        builder.push_account(&nr2, false);
                        edges.push(GraphEdge {
                            from: format!("acct:{ns2}"),
                            to: format!("tx:{}", ql.0),
                            txs: vec![u64::from(ql.0)],
                        });
                        edges.push(GraphEdge {
                            from: format!("tx:{}", ql.0),
                            to: format!("acct:{nr2}"),
                            txs: vec![u64::from(ql.0)],
                        });
                    }
                }
            }

            // Backward: receiver's incoming events in [t - delta, t).
            let inc =
                p.get_incoming_in_range(&nr, lo, t, MAX_EDGES_PER_LOOKUP);
            if inc.is_truncated {
                truncated = true;
            }
            for &ql in inc.locators.iter() {
                if builder.push_transaction(ql, false, hop, p) {
                    next_frontier.push(ql);
                    if let Some((_qt, ns2, nr2)) = resolve_names(p, ql) {
                        builder.push_account(&ns2, false);
                        builder.push_account(&nr2, false);
                        edges.push(GraphEdge {
                            from: format!("acct:{ns2}"),
                            to: format!("tx:{}", ql.0),
                            txs: vec![u64::from(ql.0)],
                        });
                        edges.push(GraphEdge {
                            from: format!("tx:{}", ql.0),
                            to: format!("acct:{nr2}"),
                            txs: vec![u64::from(ql.0)],
                        });
                    }
                }
            }
        }
        frontier = next_frontier;
    }

    let committed = p.snapshot().committed_records as u64;
    Some(Network {
        nodes,
        edges,
        truncated,
        committed,
    })
}


