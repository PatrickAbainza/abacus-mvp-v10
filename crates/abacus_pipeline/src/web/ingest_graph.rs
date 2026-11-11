//! Ingestion-backed network/graph views (reconciliation Phase 6 graph parity).
//!
//! Ports the structural network views to the ingestion-native [`IngestPipeline`].
//! Crucially, these views key accounts through the **single shared Account Map**
//! namespace, so an account is the SAME node whether it sent or received — the
//! role-separated dictionaries of the retired model are not used. Amounts are
//! never read into or emitted; only identifiers, timestamps and the derived
//! feature metrics appear.
//!
//! Reuses the data-only `Network`/`GraphNode`/`GraphEdge` types from
//! [`crate::web::graph`] so the frontend `GraphView` consumes one shape.

use abacus_tensor::graphblas::{
    from_occurrences, pruning::hop_ball, types::{HopBound, VertexId},
};
use crate::ingest_model::account_node_id;
use crate::ingest_pipeline::IngestPipeline;
use crate::web::graph::{GraphEdge, GraphNode, Network};

use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// Relative window for neighborhood expansion (mirrors the streaming_ingestion_demo view's
/// seven-day default so the two surfaces behave consistently).
const DEFAULT_DELTA_SECS: u64 = 7 * 24 * 60 * 60;

/// Hard cap on one directional lookup (mirrors the retrieval contract).
const MAX_EDGES_PER_LOOKUP: usize = 2_500;

/// Push an account node keyed by its shared-namespace code (dedup via `seen`).
fn push_acct(
    code: u32,
    label: &str,
    selected: bool,
    nodes: &mut Vec<GraphNode>,
    seen: &mut BTreeSet<String>,
) {
    let id = account_node_id(code);
    if seen.insert(id.clone()) {
        nodes.push(GraphNode {
            id,
            kind: "account",
            label: label.to_string(),
            t: None,
            selected,
            hop: None,
        });
    }
}

/// Transaction Network for a selected transaction, expanded through the shared
/// namespace (account -> tx -> account). NO amounts.
///
/// Hop expansion is **unbounded**: each hop queries outgoing + incoming from the
/// current frontier accounts within the temporal window, and continues until no
/// new account is discovered. Bounded only by `MAX_EDGES_PER_LOOKUP` per
/// directional lookup and the temporal window `[t-delta, t+delta]`.
#[must_use]
pub fn transaction_network(p: &IngestPipeline, id: u64) -> Option<Network> {
    let sel = u32::try_from(id).ok()?;
    let (t, s, r, _feats) = p.resolve(sel)?;
    let s_label = p.account_label(s)?;
    let r_label = p.account_label(r)?;

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut truncated = false;
    let mut seen_tx: BTreeSet<String> = BTreeSet::new();
    let mut seen_acct: BTreeSet<String> = BTreeSet::new();

    let s_acct = account_node_id(s);
    let sel_tx = format!("tx:{id}");
    let ws = t.saturating_sub(DEFAULT_DELTA_SECS);
    let we = t.saturating_add(DEFAULT_DELTA_SECS);

    push_acct(s, &s_label, false, &mut nodes, &mut seen_acct);
    push_acct(r, &r_label, false, &mut nodes, &mut seen_acct);
    seen_tx.insert(sel_tx.clone());
    nodes.push(GraphNode {
        id: sel_tx.clone(),
        kind: "transaction",
        label: format!("tx #{id}"),
        t: Some(t),
        selected: true,
        hop: Some(0),
    });
    edges.push(GraphEdge { from: s_acct, to: sel_tx.clone(), txs: vec![id] });
    edges.push(GraphEdge { from: sel_tx.clone(), to: account_node_id(r), txs: vec![id] });

    // Tensor algebraic ball + frontier (Vector pattern: insert/contains/difference).
    // ball = accounts already discovered; frontier = accounts at current hop depth.
    let mut ball: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    ball.insert(s);
    ball.insert(r);
    let mut frontier: Vec<u32> = vec![s, r];

    let mut hop: u32 = 0;
    while !frontier.is_empty() {
        hop += 1;
        let mut next_locators: Vec<u32> = Vec::new();

        // Tensor step: outgoing + incoming from every frontier account within [ws, we).
        for &c in &frontier {
            let out = p.outgoing_in_range(c, ws, we, MAX_EDGES_PER_LOOKUP);
            if out.len() >= MAX_EDGES_PER_LOOKUP {
                truncated = true;
            }
            next_locators.extend(out);

            let inc = p.incoming_in_range(c, ws, we, MAX_EDGES_PER_LOOKUP);
            if inc.len() >= MAX_EDGES_PER_LOOKUP {
                truncated = true;
            }
            next_locators.extend(inc);
        }

        // Dedup locators (algebraic set semantics).
        next_locators.sort_unstable();
        next_locators.dedup();

        let mut next_frontier: Vec<u32> = Vec::new();
        for loc in next_locators {
            if loc == sel {
                continue;
            }
            if let Some((nt, ns, nr, _)) = p.resolve(loc) {
                let nid = format!("tx:{loc}");
                if let (Some(ns_l), Some(nr_l)) = (p.account_label(ns), p.account_label(nr)) {
                    push_acct(ns, &ns_l, false, &mut nodes, &mut seen_acct);
                    push_acct(nr, &nr_l, false, &mut nodes, &mut seen_acct);
                    if seen_tx.insert(nid.clone()) {
                        nodes.push(GraphNode {
                            id: nid.clone(),
                            kind: "transaction",
                            label: format!("tx #{loc}"),
                            t: Some(nt),
                            selected: false,
                            hop: Some(hop as u8),
                        });
                    }
                    edges.push(GraphEdge {
                        from: account_node_id(ns),
                        to: nid.clone(),
                        txs: vec![u64::from(loc)],
                    });
                    edges.push(GraphEdge {
                        from: nid,
                        to: account_node_id(nr),
                        txs: vec![u64::from(loc)],
                    });
                }
                // Ball growth: accounts not yet discovered become next frontier
                // (unvisited mask = complement of ball).
                if ball.insert(ns) {
                    next_frontier.push(ns);
                }
                if ball.insert(nr) {
                    next_frontier.push(nr);
                }
            }
        }

        next_frontier.sort_unstable();
        next_frontier.dedup();
        frontier = next_frontier;
    }

    Some(Network { nodes, edges, truncated, committed: p.committed() as u64 })
}

/// Account Network for a selected transaction: the two endpoint accounts plus
/// directed account->account relationships aggregated across supporting
/// transactions in the window. NO amounts.
///
/// Hop expansion is **unbounded**: each hop queries outgoing + incoming from the
/// current frontier accounts within the temporal window, and continues until no
/// new account is discovered. Bounded only by `MAX_EDGES_PER_LOOKUP` per
/// directional lookup and the temporal window `[t-delta, t+delta]`.
#[must_use]
pub fn account_network(p: &IngestPipeline, id: u64) -> Option<Network> {
    let sel = u32::try_from(id).ok()?;
    let (t, s, r, _feats) = p.resolve(sel)?;
    let s_label = p.account_label(s)?;
    let r_label = p.account_label(r)?;

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut truncated = false;
    let mut seen_acct: BTreeSet<String> = BTreeSet::new();
    let mut seen_tx: BTreeSet<String> = BTreeSet::new();
    // (from_code, to_code) -> supporting locators.
    let mut rel: std::collections::BTreeMap<(u32, u32), Vec<u64>> = std::collections::BTreeMap::new();

    let ws = t.saturating_sub(DEFAULT_DELTA_SECS);
    let we = t.saturating_add(DEFAULT_DELTA_SECS);

    push_acct(s, &s_label, true, &mut nodes, &mut seen_acct);
    push_acct(r, &r_label, true, &mut nodes, &mut seen_acct);
    seen_tx.insert(format!("tx:{id}"));
    rel.entry((s, r)).or_default().push(id);

    // Tensor algebraic ball + frontier.
    let mut ball: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    ball.insert(s);
    ball.insert(r);
    let mut frontier: Vec<u32> = vec![s, r];

    while !frontier.is_empty() {
        let mut next_locators: Vec<u32> = Vec::new();

        // Tensor step: outgoing + incoming from every frontier account within [ws, we).
        for &c in &frontier {
            let out = p.outgoing_in_range(c, ws, we, MAX_EDGES_PER_LOOKUP);
            if out.len() >= MAX_EDGES_PER_LOOKUP {
                truncated = true;
            }
            next_locators.extend(out);

            let inc = p.incoming_in_range(c, ws, we, MAX_EDGES_PER_LOOKUP);
            if inc.len() >= MAX_EDGES_PER_LOOKUP {
                truncated = true;
            }
            next_locators.extend(inc);
        }

        next_locators.sort_unstable();
        next_locators.dedup();

        let mut next_frontier: Vec<u32> = Vec::new();
        for loc in next_locators {
            if seen_tx.insert(format!("tx:{loc}")) {
                if let Some((_nt, ns, nr, _)) = p.resolve(loc) {
                    rel.entry((ns, nr)).or_default().push(u64::from(loc));
                    if ball.insert(ns) {
                        next_frontier.push(ns);
                    }
                    if ball.insert(nr) {
                        next_frontier.push(nr);
                    }
                }
            }
        }

        next_frontier.sort_unstable();
        next_frontier.dedup();
        frontier = next_frontier;
    }

    for ((from, to), txs) in rel {
        if let (Some(fl), Some(tl)) = (p.account_label(from), p.account_label(to)) {
            push_acct(from, &fl, from == s || from == r, &mut nodes, &mut seen_acct);
            push_acct(to, &tl, to == s || to == r, &mut nodes, &mut seen_acct);
            edges.push(GraphEdge { from: account_node_id(from), to: account_node_id(to), txs });
        }
    }

    Some(Network { nodes, edges, truncated, committed: p.committed() as u64 })
}

/// Transaction Network extended to `k` hops using tensor algorithms.
///
/// Performs **iterative frontier expansion with batched CSR/CSC lookups**:
/// starting from the focal sender/receiver, each hop performs batched
/// `outgoing_in_range` + `incoming_in_range` over the frontier accounts
/// within the 7-day window, discovers new accounts and transactions, and
/// continues for `k` hops. Transaction nodes carry their hop level
/// (`h0` = focal, `h1` = direct neighbor, etc.). All collected events are
/// assembled into a [`PairMatrix`] adjacency matrix and [`hop_ball`]
/// (mxv + vxm + ewise_add_lor + difference, Boolean LOR_LAND semiring)
/// confirms the `<= k`-hop ball around {s, r}. Edges are bipartite
/// (account -> tx -> account) and capped at [`MAX_EDGES_PER_LOOKUP`] (2,500).
///
/// # Arguments
/// * `p` - the ingestion pipeline
/// * `id` - focal transaction locator
/// * `k` - hop bound (e.g. `3` = focal tx + 3 levels of 7-day transaction
///   history expansion)
///
/// # Returns
/// `Some(Network)` with nodes/edges/truncated/committed, or `None` if the
/// focal transaction is absent.
#[must_use]
pub fn transaction_network_multi_hop(p: &IngestPipeline, id: u64, k: u64) -> Option<Network> {
    let sel = u32::try_from(id).ok()?;
    let (t, s, r, _feats) = p.resolve(sel)?;
    let s_label = p.account_label(s)?;
    let r_label = p.account_label(r)?;

    // 7-day window around the focal transaction.
    let delta = DEFAULT_DELTA_SECS;
    let t_lo = t.saturating_sub(delta);
    let t_hi = t.saturating_add(delta);

    // Track accounts and transactions with hop levels.
    let mut acct_labels: BTreeMap<u32, String> = BTreeMap::new();
    let mut tx_info: BTreeMap<u32, (u64, u32, u32, u8)> = BTreeMap::new(); // locator -> (t, s, r, hop)
    let mut acct_seen: BTreeSet<u32> = BTreeSet::new();

    acct_labels.insert(s, s_label.clone());
    acct_labels.insert(r, r_label.clone());
    tx_info.insert(sel, (t, s, r, 0));
    acct_seen.insert(s);
    acct_seen.insert(r);

    // Frontier of accounts at the current hop.
    let mut frontier_accts: BTreeSet<u32> = BTreeSet::from([s, r]);

    for hop in 0..k {
        if frontier_accts.is_empty() {
            break;
        }

        // --- Batched CSR/CSC lookups over all frontier accounts ---
        let acct_codes: Vec<u32> = frontier_accts.iter().copied().collect();

        // Outgoing (sender-side): all events where frontier accounts sent.
        let outgoing = p.batch_outgoing_in_range(&acct_codes, t_lo, t_hi, MAX_EDGES_PER_LOOKUP);
        // Incoming (receiver-side): all events where frontier accounts received.
        let incoming = p.batch_incoming_in_range(&acct_codes, t_lo, t_hi, MAX_EDGES_PER_LOOKUP);

        let mut next_frontier_accts: BTreeSet<u32> = BTreeSet::new();

        // Process outgoing results: acct -> counterparty(receiver)
        for (_code, locators) in &outgoing {
            for &loc in locators {
                if tx_info.contains_key(&loc) {
                    continue;
                }
                if let Some((nt, ns, nr, _)) = p.resolve(loc) {
                    tx_info.insert(loc, (nt, ns, nr, hop as u8 + 1));
                    if acct_seen.insert(nr) {
                        next_frontier_accts.insert(nr);
                        if let Some(label) = p.account_label(nr) {
                            acct_labels.insert(nr, label);
                        }
                    }
                    // Ensure sender is also tracked (should already be seen)
                    if acct_seen.insert(ns) {
                        next_frontier_accts.insert(ns);
                        if let Some(label) = p.account_label(ns) {
                            acct_labels.insert(ns, label);
                        }
                    }
                }
            }
        }

        // Process incoming results: counterparty(sender) -> acct
        for (_code, locators) in &incoming {
            for &loc in locators {
                if tx_info.contains_key(&loc) {
                    continue;
                }
                if let Some((nt, ns, nr, _)) = p.resolve(loc) {
                    tx_info.insert(loc, (nt, ns, nr, hop as u8 + 1));
                    if acct_seen.insert(ns) {
                        next_frontier_accts.insert(ns);
                        if let Some(label) = p.account_label(ns) {
                            acct_labels.insert(ns, label);
                        }
                    }
                    if acct_seen.insert(nr) {
                        next_frontier_accts.insert(nr);
                        if let Some(label) = p.account_label(nr) {
                            acct_labels.insert(nr, label);
                        }
                    }
                }
            }
        }

        frontier_accts = next_frontier_accts;
    }

    let mut occurrences: Vec<(u32, u32, i64)> = Vec::new();
    for (_loc, (_t, es, er, _hop)) in &tx_info {
        occurrences.push((*es, *er, 1));
    }

    let matrix = from_occurrences(&occurrences).ok()?;

    let hop_ball_set = hop_ball(&matrix, VertexId(s), VertexId(r), HopBound(k));
    let mut ball_codes: BTreeSet<u32> = BTreeSet::new();
    for v in hop_ball_set.indices() {
        ball_codes.insert(*v);
    }

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut truncated = false;
    let mut seen_acct: BTreeSet<String> = BTreeSet::new();
    let mut seen_tx: BTreeSet<String> = BTreeSet::new();
    let mut edge_count = 0;

    push_acct(s, &s_label, true, &mut nodes, &mut seen_acct);
    push_acct(r, &r_label, true, &mut nodes, &mut seen_acct);

    for (acct, label) in &acct_labels {
        if *acct != s && *acct != r && ball_codes.contains(acct) {
            push_acct(*acct, label, false, &mut nodes, &mut seen_acct);
        }
    }

    for (loc, (_et, es, er, hop)) in &tx_info {
        if !ball_codes.contains(es) || !ball_codes.contains(er) {
            continue;
        }
        let tx_id = format!("tx:{loc}");
        if seen_tx.insert(tx_id.clone()) {
            nodes.push(GraphNode {
                id: tx_id.clone(),
                kind: "transaction",
                label: format!("tx #{loc}"),
                t: Some(*_et),
                selected: *loc == sel,
                hop: Some(*hop),
            });
        }
        if edge_count >= MAX_EDGES_PER_LOOKUP {
            truncated = true;
            break;
        }
        edges.push(GraphEdge {
            from: account_node_id(*es),
            to: tx_id.clone(),
            txs: vec![(*loc).into()],
        });
        edge_count += 1;
        if edge_count >= MAX_EDGES_PER_LOOKUP {
            truncated = true;
            break;
        }
        edges.push(GraphEdge {
            from: tx_id,
            to: account_node_id(*er),
            txs: vec![(*loc).into()],
        });
        edge_count += 1;
    }

    Some(Network {
        nodes,
        edges,
        truncated,
        committed: p.committed() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest_pipeline::IngestPipeline;
    use abacus_ingest::features::FeaturePayload;
    use abacus_ingest::namespace::{EventIdentity, EventNamespace};

    fn ns(b: u8) -> EventNamespace {
        EventNamespace::from_bytes([b; 16])
    }
    fn rec(eid: u64, s: u64, r: u64) -> abacus_ingest::record::CanonicalRecord {
        abacus_ingest::record::CanonicalRecord {
            identity: EventIdentity { namespace: ns(1), event_id: eid },
            event_time_ms: 1_700_000_000_000,
            sender_id: s,
            receiver_id: r,
            features: FeaturePayload { log_magnitude: 1, relative_threshold_distance: 0, roundedness_flags: 0 },
        }
    }

    #[test]
    fn transaction_network_uses_shared_namespace() {
        let mut p = IngestPipeline::new();
        // 100 -> 200 ; 300 -> 100 (100 is both sender and receiver)
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100).unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100).unwrap();
        let net = transaction_network(&p, 0).expect("network");
        // Account 100 appears once as a node (shared namespace), not twice.
        let acct100 = account_node_id(p.resolve(0).unwrap().1);
        let count = net.nodes.iter().filter(|n| n.id == acct100).count();
        assert_eq!(count, 1, "shared namespace: one node per account");
        assert!(net.nodes.iter().any(|n| n.kind == "transaction"));
    }

    #[test]
    fn account_network_aggregates_relationships() {
        let mut p = IngestPipeline::new();
        p.apply(&rec(0, 100, 200), Some(12_345), 10_000, 100).unwrap();
        p.apply(&rec(1, 300, 100), Some(12_345), 10_000, 100).unwrap();
        let net = account_network(&p, 0).expect("network");
        // Endpoint accounts highlighted.
        assert!(net.nodes.iter().any(|n| n.kind == "account" && n.selected));
        assert!(!net.edges.is_empty());
    }

    #[test]
    fn transaction_network_multi_hop_expands_to_k_hops() {
        let mut p = IngestPipeline::new();
        let base_time = 1_700_000_000_000u64;

        // tx 0: 100 -> 200 at base_time (focal, hop 0)
        p.apply(
            &abacus_ingest::record::CanonicalRecord {
                identity: EventIdentity { namespace: ns(1), event_id: 0 },
                event_time_ms: base_time as i64,
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

        // tx 1: 200 -> 300 at base_time + 1s (hop 1)
        p.apply(
            &abacus_ingest::record::CanonicalRecord {
                identity: EventIdentity { namespace: ns(1), event_id: 1 },
                event_time_ms: (base_time + 1000) as i64,
                sender_id: 200,
                receiver_id: 300,
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

        // tx 2: 300 -> 400 at base_time + 2s (hop 2)
        p.apply(
            &abacus_ingest::record::CanonicalRecord {
                identity: EventIdentity { namespace: ns(1), event_id: 2 },
                event_time_ms: (base_time + 2000) as i64,
                sender_id: 300,
                receiver_id: 400,
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

        // k=3 should find all 3 hops: focal + 2 levels of expansion
        let net = transaction_network_multi_hop(&p, 0, 3).expect("network");

        let tx_nodes: Vec<_> = net
            .nodes
            .iter()
            .filter(|n| n.kind == "transaction")
            .collect();
        assert_eq!(tx_nodes.len(), 3, "Expected 3 transaction nodes");

        let hop0: Vec<_> = tx_nodes.iter().filter(|n| n.hop == Some(0)).collect();
        let hop1: Vec<_> = tx_nodes.iter().filter(|n| n.hop == Some(1)).collect();
        let hop2: Vec<_> = tx_nodes.iter().filter(|n| n.hop == Some(2)).collect();

        assert_eq!(hop0.len(), 1, "Expected 1 hop-0 transaction");
        assert_eq!(hop1.len(), 1, "Expected 1 hop-1 transaction");
        assert_eq!(hop2.len(), 1, "Expected 1 hop-2 transaction");

        // Bipartite edges: 2 per transaction (account->tx and tx->account)
        assert_eq!(net.edges.len(), 6, "Expected 6 edges");

        // Focal transaction is selected at hop 0
        let focal: Vec<_> = tx_nodes.iter().filter(|n| n.selected).collect();
        assert_eq!(focal.len(), 1);
        assert_eq!(focal[0].hop, Some(0));
    }
}

