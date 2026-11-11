//! Baseline-provenance caller tests: every missing/invalid condition
//! yields `Indeterminate{BaselineUnavailable}`; valid provenance yields an
//! exact share.

use abacus_pipeline::atomic::evaluate::evaluate_flagged_share;
use abacus_tensor::atomic::outcome::{OpReason, Outcome};
use abacus_tensor::atomic::reference::{Baseline, Theta};
use abacus_tensor::atomic::types::{Entry, LocSet, Support};

fn support() -> Support {
    // values straddle gate mean(1000)+scale(100)=1100: 900, 1100, 1300
    Support::from_canonical(vec![
        Entry::new(1, 1, 1, 900, LocSet::from_sorted(vec![0])),
        Entry::new(2, 1, 1, 1100, LocSet::from_sorted(vec![1])),
        Entry::new(3, 1, 1, 1300, LocSet::from_sorted(vec![2])),
        Entry::new(50, 1, 1, 5000, LocSet::from_sorted(vec![3])),
    ])
}
fn good_theta() -> Theta {
    Theta {
        threshold: None,
        baseline: Some(Baseline { mean: 1000, scale: 100, version: "bl-2026-10-18".into(), as_of: 1729200000 }),
    }
}

#[test]
fn valid_provenance_yields_exact_share() {
    let Outcome::Determinate(share) = evaluate_flagged_share(&support(), 0, 10, &good_theta()) else {
        panic!("expected determinate");
    };
    assert_eq!((share.flagged, share.total), (2, 3), "1100 and 1300 at/above gate; t=50 outside window");
    assert_eq!(share.baseline_version.version, "bl-2026-10-18");
    assert_eq!(share.baseline_version.as_of, 1729200000);
}

#[test]
fn missing_baseline_is_indeterminate() {
    let theta = Theta::default();
    match evaluate_flagged_share(&support(), 0, 10, &theta) {
        Outcome::Indeterminate(ind) => {
            assert_eq!(ind.reason, OpReason::BaselineUnavailable);
            assert!(ind.cause.contains("absent"));
        }
        Outcome::Determinate(_) => panic!("missing baseline must not yield a share"),
    }
}

#[test]
fn empty_version_is_indeterminate() {
    let theta = Theta {
        threshold: None,
        baseline: Some(Baseline { mean: 1000, scale: 100, version: String::new(), as_of: 1729200000 }),
    };
    match evaluate_flagged_share(&support(), 0, 10, &theta) {
        Outcome::Indeterminate(ind) => assert_eq!(ind.reason, OpReason::BaselineUnavailable),
        Outcome::Determinate(_) => panic!("empty version must not yield a share"),
    }
}

#[test]
fn zero_as_of_is_indeterminate() {
    let theta = Theta {
        threshold: None,
        baseline: Some(Baseline { mean: 1000, scale: 100, version: "v1".into(), as_of: 0 }),
    };
    match evaluate_flagged_share(&support(), 0, 10, &theta) {
        Outcome::Indeterminate(ind) => assert_eq!(ind.reason, OpReason::BaselineUnavailable),
        Outcome::Determinate(_) => panic!("zero as-of must not yield a share"),
    }
}

#[test]
fn gate_is_exact_at_boundary_and_never_float() {
    // value exactly at gate flags; one below does not. Pure integer compare.
    let s = Support::from_canonical(vec![
        Entry::new(1, 1, 1, 1099, LocSet::empty()),
        Entry::new(2, 1, 1, 1100, LocSet::empty()),
    ]);
    let Outcome::Determinate(share) = evaluate_flagged_share(&s, 0, 10, &good_theta()) else {
        panic!("expected determinate");
    };
    assert_eq!((share.flagged, share.total), (1, 2));
}
