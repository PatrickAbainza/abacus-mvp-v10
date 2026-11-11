//! Hop-bound pruning (#17–#18, rows 46–47): bounded neighborhood via
//! Boolean semiring sparse linear algebra.
//!
//! Ported from v7 `graphblas/pruning/hops.rs` (itself from v5). The `<= k`
//! hop ball from both focal endpoints over `A_one` unions `mxv` (senders of
//! frontier receivers) and `vxm` (receivers of frontier senders) per hop
//! with `ewise_add_lor`, masking against the ball via the complement
//! (`difference`). Frontier reuse: the prefix balls `B_0 <= ... <= B_k`
//! come from one incremental traversal that carries the derived frontier
//! and ball forward. No queue, no visited set, no handwritten BFS.

use super::super::matrix::PairMatrix;
use super::super::primitives::{ewise::ewise_add_lor, mxv::mxv, vxm::vxm};
use super::super::types::{HopBound, TransactionRecord, VertexId};
use super::super::vector::Vector;

/// `V_0 = {u_f, v_f}`: the already-resolved focal coordinates.
fn seed(focal_u: VertexId, focal_v: VertexId) -> Vector {
    let mut ball = Vector::zero();
    ball.insert(focal_u.0);
    ball.insert(focal_v.0);
    ball
}

/// One hop: neighbours of the derived `frontier` (masked `mxv` union `vxm`)
/// that are not already in the derived `ball`.
fn step(a_one: &PairMatrix, ball: &Vector, frontier: &Vector) -> Vector {
    let bwd = mxv(a_one, frontier);
    let fwd = vxm(a_one, frontier);
    let stepped = ewise_add_lor(&bwd, &fwd);
    stepped.difference(ball)
}

/// The `<= k`-hop vertex ball around `{u_f, v_f}` over support `a_one`.
#[must_use]
pub fn hop_ball(a_one: &PairMatrix, focal_u: VertexId, focal_v: VertexId, k: HopBound) -> Vector {
    let mut ball = seed(focal_u, focal_v);
    let mut frontier = ball.clone();
    for _ in 0..k.0 {
        let next = step(a_one, &ball, &frontier);
        if next.is_empty() {
            break;
        }
        ball = ewise_add_lor(&ball, &next);
        frontier = next;
    }
    ball
}

/// Prefix balls `B_0, ..., B_j` (`j = balls.len() - 1 <= k`) from a
/// **single** incremental traversal: each hop consumes the frontier and ball
/// already derived by the preceding hop, so the traversal is never restarted
/// and no resolved frontier vertex is rediscovered.
///
/// Exactness: `balls[i] == hop_ball(a_one, u, v, HopBound(i))` for every
/// `i < balls.len()`; the traversal saturates once a hop adds no new vertex,
/// so for `i >= balls.len() - 1` `hop_ball(..., HopBound(i))` is the same
/// vector as `balls.last()`. Callers index with
/// `balls.get(i).unwrap_or(last)` and therefore never restart the traversal.
///
/// Crate-internal execution helper only: no new public primitive, and no
/// change to hop semantics.
#[allow(dead_code)]
#[must_use]
pub(crate) fn hop_balls_prefix(
    a_one: &PairMatrix,
    focal_u: VertexId,
    focal_v: VertexId,
    k: HopBound,
) -> Vec<Vector> {
    let mut ball = seed(focal_u, focal_v);
    let mut frontier = ball.clone();
    let mut balls = vec![ball.clone()];
    let mut hops_done = 0u64;
    while hops_done < k.0 {
        let next = step(a_one, &ball, &frontier);
        if next.is_empty() {
            break;
        }
        ball = ewise_add_lor(&ball, &next);
        frontier = next;
        balls.push(ball.clone());
        hops_done += 1;
    }
    balls
}

/// `prune_hops`: retain entries whose endpoints both lie in the `<= k`-hop
/// ball.
#[must_use]
pub fn prune_hops(
    records: &[TransactionRecord],
    a_one: &PairMatrix,
    focal_u: VertexId,
    focal_v: VertexId,
    k: HopBound,
) -> Vec<TransactionRecord> {
    let ball = hop_ball(a_one, focal_u, focal_v, k);
    super::vertex::prune_vertices(records, &ball)
}
