//! Rows 18–23 (§B.2): ledger CAS, frontier, ingest, seal, live view, watermark.

use abacus_tensor::ingest::{
    CasOutcome, Event, Expected, IngestionEngine, Ledger, LedgerEntry, LedgerOp,
};
use abacus_tensor::tensor::coordinate::Coord;

fn ev(id: u64, t: u64, s: u32, r: u32) -> Event {
    Event {
        id,
        coord: Coord::new(t, s, r),
    }
}

#[test]
fn cas_create_update_close_follow_def_7_3() {
    let mut l = Ledger::new();
    // Create: expected = ⊥.
    let out = l.ledger_op(7, LedgerOp::Create(100));
    assert!(matches!(
        out,
        CasOutcome::Committed {
            a_old: 0,
            a_new: 100,
            version: 1
        }
    ));
    // Redelivered Create fails: L(id) ≠ ⊥ (the model-level idempotency guard).
    let CasOutcome::Retry { actual } = l.ledger_op(7, LedgerOp::Create(100)) else {
        panic!("expected retry");
    };
    assert_eq!(
        actual,
        Expected::Live(LedgerEntry {
            value: 100,
            version: 1
        })
    );
    // Update with a stale expectation fails; with the exact prior state it
    // writes (a_new, v+1).
    assert!(matches!(
        l.ledger_op(
            7,
            LedgerOp::Update {
                expected: LedgerEntry {
                    value: 1,
                    version: 1
                },
                new_value: 200
            }
        ),
        CasOutcome::Retry { .. }
    ));
    let out = l.ledger_op(
        7,
        LedgerOp::Update {
            expected: LedgerEntry {
                value: 100,
                version: 1,
            },
            new_value: 150,
        },
    );
    assert!(matches!(
        out,
        CasOutcome::Committed {
            a_old: 100,
            a_new: 150,
            version: 2
        }
    ));
    // Close writes ⊥; a later read reports Absent exactly.
    let out = l.ledger_op(
        7,
        LedgerOp::Close {
            expected: LedgerEntry {
                value: 150,
                version: 2,
            },
        },
    );
    assert!(matches!(
        out,
        CasOutcome::Committed {
            a_old: 150,
            a_new: 0,
            version: 0
        }
    ));
    assert_eq!(l.get(7), Expected::Absent);
    // A stale Close against ⊥ is a retry, never a second delta.
    assert!(matches!(
        l.ledger_op(
            7,
            LedgerOp::Close {
                expected: LedgerEntry {
                    value: 150,
                    version: 2
                },
            }
        ),
        CasOutcome::Retry { .. }
    ));
}
#[test]
fn ingest_emits_one_delta_per_committed_op_and_seals_to_oracle() {
    let mut eng = IngestionEngine::new();
    // Two co-located distinct events contribute two Δs (no applied-set merge).
    eng.ingest(&ev(0, 10, 1, 2), LedgerOp::Create(5)).unwrap();
    eng.ingest(&ev(1, 10, 1, 2), LedgerOp::Create(7)).unwrap();
    assert_eq!(eng.frontier().len(), 2);
    assert_eq!(eng.live_at(10, 1, 2).unwrap(), 12);
    assert_eq!(eng.live_at(10, 1, 3).unwrap(), 0); // absent
                                                   // Duplicate delivery retries: no delta, frontier unchanged.
    assert!(matches!(
        eng.ingest(&ev(0, 10, 1, 2), LedgerOp::Create(5)).unwrap(),
        CasOutcome::Retry { .. }
    ));
    assert_eq!(eng.frontier().len(), 2);
    // Correction replaces value 7 with 10 at the same coordinate (Δ = +3).
    eng.ingest(
        &ev(1, 10, 1, 2),
        LedgerOp::Update {
            expected: LedgerEntry {
                value: 7,
                version: 1,
            },
            new_value: 10,
        },
    )
    .unwrap();
    assert_eq!(eng.live_at(10, 1, 2).unwrap(), 15);
    // Seal: T_sealed = merge, frontier cleared, report is a full rebuild.
    let rep = eng.seal().unwrap();
    assert!(rep.full_rebuild && eng.frontier().is_empty());
    assert_eq!(rep.new_coordinates, 1);
    assert_eq!(eng.live_at(10, 1, 2).unwrap(), 15);
    assert!(eng.sealed().check_invariants());
    // Reversals zero the coordinate; the diff reports a removal.
    for (id, v, ver) in [(0u64, 5i64, 1u64), (1u64, 10i64, 2u64)] {
        eng.ingest(
            &ev(id, 10, 1, 2),
            LedgerOp::Close {
                expected: LedgerEntry {
                    value: v,
                    version: ver,
                },
            },
        )
        .unwrap();
    }
    let rep = eng.seal().unwrap();
    assert_eq!(rep.removed_coordinates, 1);
    assert_eq!(eng.live_at(10, 1, 2).unwrap(), 0);
}

#[test]
fn watermark_is_stored_not_consulted() {
    let mut eng = IngestionEngine::new();
    eng.set_watermark(1_000_000);
    assert_eq!(eng.watermark(), Some(1_000_000));
    // A late arrival with t ≤ W is an ordinary Create — not rejected, not
    // special-cased (P10).
    eng.ingest(&ev(0, 5, 0, 0), LedgerOp::Create(1)).unwrap();
    assert_eq!(eng.stats().creates, 1);
    assert_eq!(eng.live_at(5, 0, 0).unwrap(), 1);
}

#[test]
fn support_diff_and_byte_size_are_the_row_22_analytics() {
    use abacus_tensor::coo::entry::TxEntry;
    let a = vec![TxEntry::new(1, 0, 0, 5)];
    let b = vec![TxEntry::new(1, 0, 0, 6), TxEntry::new(2, 0, 0, 1)];
    assert_eq!(abacus_tensor::ingest::diff_coordinates(&a, &b), (1, 1, 0));
    assert_eq!(abacus_tensor::ingest::diff_coordinates(&a, &a), (0, 0, 0));
    let csf = abacus_tensor::csf::compress(&b);
    // Two t-fibers: I0(2×8) + P0(3×8) + I1(2×4) + P1(3×8) + I2(2×4) + V(2×8) = 96.
    assert_eq!(abacus_tensor::ingest::csf_byte_size(&csf), 96);
}

#[test]
fn ingest_range_error_leaves_the_frontier_unchanged() {
    let mut eng = IngestionEngine::new();
    // Create(i64::MIN): Δ = MIN − 0 is representable, so it commits.
    eng.ingest(&ev(0, 1, 0, 0), LedgerOp::Create(i64::MIN))
        .unwrap();
    let before = eng.frontier().len();
    // Update to 1: Δ = 1 − MIN overflows i64 — the error path is fail-closed.
    let err = eng
        .ingest(
            &ev(0, 1, 0, 0),
            LedgerOp::Update {
                expected: LedgerEntry {
                    value: i64::MIN,
                    version: 1,
                },
                new_value: 1,
            },
        )
        .unwrap_err();
    assert!(err.to_string().contains("range"));
    assert_eq!(eng.frontier().len(), before); // unchanged
}

/// Row 18 — ledger op applied twice: the SECOND application must be a
/// Retry (the CAS equality check is the entire model-level idempotency
/// mechanism — P7). No second delta is possible because no second
/// Committed can occur for a redelivered op pinned on the same state.
#[test]
fn ledger_redelivery_is_idempotent_by_cas() {
    let mut l = Ledger::new();
    assert!(matches!(
        l.ledger_op(1, LedgerOp::Create(100)),
        CasOutcome::Committed { .. }
    ));
    assert!(matches!(
        l.ledger_op(1, LedgerOp::Create(100)),
        CasOutcome::Retry { .. }
    ));
    // Update redelivered with the SAME (already stale) expected pin.
    let stale = LedgerEntry {
        value: 100,
        version: 1,
    };
    assert!(matches!(
        l.ledger_op(
            1,
            LedgerOp::Update {
                expected: stale,
                new_value: 150
            }
        ),
        CasOutcome::Committed { .. }
    ));
    assert!(matches!(
        l.ledger_op(
            1,
            LedgerOp::Update {
                expected: stale,
                new_value: 150
            }
        ),
        CasOutcome::Retry { .. }
    ));
}

/// Row 20 — delta construction happens BEFORE any frontier push, so a
/// range-failing correction leaves the frontier unchanged (documented
/// fail-closed behaviour; the caller aborts the stream — K14 scope).
#[test]
fn delta_computed_before_frontier_mutation() {
    // Verify the ORDERING by exercising the engine: the frontier must not
    // gain an entry for an op whose delta failed to construct.
    // Create(i64::MIN): Δ = MIN − 0 is representable, so it commits.
    let mut eng = IngestionEngine::new();
    eng.ingest(&ev(0, 1, 0, 0), LedgerOp::Create(i64::MIN))
        .unwrap();
    // Update to 1: Δ = 1 − MIN overflows i64 → range error, no delta.
    let err = eng.ingest(
        &ev(0, 1, 0, 0),
        LedgerOp::Update {
            expected: LedgerEntry {
                value: i64::MIN,
                version: 1,
            },
            new_value: 1,
        },
    );
    assert!(err.is_err());
    assert_eq!(eng.frontier().len(), 1); // only the create's delta
}
