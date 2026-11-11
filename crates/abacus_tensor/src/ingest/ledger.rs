//! Row 18 (§B.2) + P7: the ledger and its single compare-and-swap composite.
//!
//! `d_cas` (one of the ten named composites of §B.1) is realized here exactly
//! once, as [`Ledger::ledger_op`]: `a_{16}.1` point read + structural equality
//! (`a_{01}` per field + `a_{17}`) + `R_2` mode dispatch + `a_{16}.1`
//! insert/remove. Success holds iff `L(id(e)) = expected`; that equality check
//! is the entire *model-level* exactly-once mechanism. **Scope (P7/K14):** it
//! is not a claim about external delivery, distribution, or failure between
//! external effects; atomicity, id identity, and ledger persistence are
//! assumed, not derived, and are not implemented here.
//!
//! Storage justification: the ledger is a point-addressed function on event
//! identities; iteration order is never used, so no ordered structure is
//! introduced (`a_{16}.1`).

use crate::tensor::coordinate::Coord;
use crate::tensor::value::MinorUnits;
use std::collections::HashMap;

/// Event identity (Def 7.1). Dataset mapping: 0-based CSV data-row ordinal.
pub type EventId = u64;

/// One event: identity + coordinate (Def 7.1). Amounts ride on [`LedgerOp`]
/// arguments, never on the identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    /// `id(e) ∈ E`.
    pub id: EventId,
    /// `x(e) = (t, s, r)` under `prec_3`.
    pub coord: Coord,
}

/// A live ledger record: `(current recorded value, version)` (Def 7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerEntry {
    /// Current recorded value `a ∈ V` (exact micro-units).
    pub value: MinorUnits,
    /// Monotone version `v ∈ ℕ` (1 after Create, +1 per Update).
    pub version: u64,
}

/// The CAS `expected` argument: an element of `(V × ℕ) ∪ {⊥}` (Def 7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expected {
    /// `⊥` — no live record (never created, or closed).
    Absent,
    /// `(a_old, v)` — pins the EXACT prior ledger state.
    Live(LedgerEntry),
}

/// Row 18 — the unified ledger operation `LedgerOp(e, expected, (a_new, v_new))`
/// (Def 7.3): **one** CAS primitive whose three modes differ only in their
/// arguments. `expected` pins the exact prior state; success iff
/// `L(id(e)) = expected`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerOp {
    /// Create: `expected = ⊥`; writes `(a, 1)`. A redelivered Create fails
    /// because `L(id) ≠ ⊥` (the idempotency mechanism).
    Create(MinorUnits),
    /// Update (correction): `expected = (a_old, v)`; writes `(a_new, v+1)`.
    Update {
        /// Exact prior record the caller expects (CAS guard).
        expected: LedgerEntry,
        /// Corrected value `a_new`.
        new_value: MinorUnits,
    },
    /// Close (reversal): `expected = (a_old, v)`; writes `⊥`. The negated value
    /// is the CURRENT ledger value pinned by `expected`, never a cached amount
    /// from creation time.
    Close {
        /// Exact prior record the caller expects (CAS guard).
        expected: LedgerEntry,
    },
}

/// Outcome of one `LedgerOp` (Def 7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasOutcome {
    /// Success. Carries `(a_old, a_new)`: `a_old` is the value component of
    /// `expected` (0 when `expected = ⊥`); `a_new` is 0 when the op closed.
    Committed {
        /// Prior recorded value (0 when expected was `⊥`).
        a_old: MinorUnits,
        /// New recorded value (0 when the op closed the event).
        a_new: MinorUnits,
        /// Version written by this op (0 when the op closed the event).
        version: u64,
    },
    /// Retry: `L(id(e)) ≠ expected`. No delta is emitted.
    Retry {
        /// The actual current state (diagnostic only).
        actual: Expected,
    },
}

/// The ledger `L : E → (V × ℕ) ∪ {⊥}` (Def 7.2), keyed by event identity.
///
/// `⊥` is represented by key absence; Close removes the key, so after a
/// reversal `L(id) = ⊥` exactly as the definition requires.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    map: HashMap<EventId, LedgerEntry>,
}

impl Ledger {
    /// Empty ledger (every event `⊥`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Read `L(id(e))` (Def 7.2).
    #[must_use]
    pub fn get(&self, id: EventId) -> Expected {
        self.map
            .get(&id)
            .map_or(Expected::Absent, |&e| Expected::Live(e))
    }

    /// `d_cas` — one compare-and-swap. Succeeds iff `L(id(e)) = expected`,
    /// then applies the mode's write in the same step. The delta itself is
    /// constructed by the caller (P8); this function stores exact values only.
    pub fn ledger_op(&mut self, id: EventId, op: LedgerOp) -> CasOutcome {
        let actual = self.get(id);
        let cas_holds = match (&op, actual) {
            (LedgerOp::Create(_), Expected::Absent) => true,
            (
                LedgerOp::Update { expected, .. } | LedgerOp::Close { expected },
                Expected::Live(a),
            ) => *expected == a,
            _ => false,
        };
        if !cas_holds {
            return CasOutcome::Retry { actual };
        }
        match op {
            LedgerOp::Create(a) => {
                self.map.insert(
                    id,
                    LedgerEntry {
                        value: a,
                        version: 1,
                    },
                );
                CasOutcome::Committed {
                    a_old: 0,
                    a_new: a,
                    version: 1,
                }
            }
            LedgerOp::Update {
                expected,
                new_value,
            } => {
                let v = expected.version + 1; // a_09: version counter, not a value
                self.map.insert(
                    id,
                    LedgerEntry {
                        value: new_value,
                        version: v,
                    },
                );
                CasOutcome::Committed {
                    a_old: expected.value,
                    a_new: new_value,
                    version: v,
                }
            }
            LedgerOp::Close { expected } => {
                self.map.remove(&id);
                CasOutcome::Committed {
                    a_old: expected.value,
                    a_new: 0,
                    version: 0,
                }
            }
        }
    }

    /// Number of live records (diagnostic only; never used for algebra).
    #[must_use]
    pub fn live_len(&self) -> usize {
        self.map.len()
    }
}
