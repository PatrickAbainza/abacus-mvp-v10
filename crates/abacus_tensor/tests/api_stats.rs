//! Rows 26–27 + K-sections as tests: statistics, API boundary, error alphabet.

use abacus_tensor::api;
use abacus_tensor::coo::builder::build_from_reader;
use abacus_tensor::stats;

const CSV: &str = include_str!("fixtures/small.csv");

#[test]
fn stats_are_deterministic_diagnostics_outside_value_semantics() {
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    let csf = abacus_tensor::csf::compress(&build.coo);
    let st = stats::compute(&build, &csf);
    assert_eq!(st.num_transactions, 6);
    assert_eq!(st.num_unique_coordinates, 3);
    assert_eq!(st.num_duplicates_absorbed, 3);
    assert_eq!(st.num_senders, 3); // Alice, Bob, Carol
    assert_eq!(st.num_receivers, 3); // Alice, Bob, Dave
    assert!(st.sparsity > 0.0 && st.sparsity <= 1.0);
    assert!(st.compression_ratio > 0.0);
    let js = st.to_json();
    assert!(js.contains("\"num_transactions\": 6"));
    // Whole-file checksum is reported by the artifact render, not invented here.
    let bytes = abacus_tensor::csf::serialize(&csf, &build);
    let js2 = st.to_json_with_artifact(&bytes, "tensor.csf");
    assert!(js2.contains("\"tensor_file\": \"tensor.csf\""));
    assert!(js2.contains(&format!(
        "\"checksum\": {}",
        abacus_tensor::csf::checksum(&bytes)
    )));
    // Deterministic: identical inputs, identical JSON.
    assert_eq!(st.to_json(), stats::compute(&build, &csf).to_json());
}

#[test]
fn settlement_split_separates_aggregation_from_zero_elimination() {
    // G8. `fixtures/small.csv` contains both absorption mechanisms:
    //   (Alice,Bob,10:00)  100.00 + 50.00   = 150.00  -> AGGREGATED (m=2)
    //   (Bob,Alice,09:00)   7.25 + (-7.25) = 0       -> ELIMINATED (R6), absent
    // The eliminated coordinate is NOT in supp(T): it is absent, not zero-valued.
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    let csf = abacus_tensor::csf::compress(&build.coo);
    let st = stats::compute(&build, &csf);

    // One surviving coordinate absorbed >1 row.
    assert_eq!(st.num_aggregated_coordinates, 1, "C_settled");
    // Exactly one row was absorbed into it (the second 50.00).
    assert_eq!(st.absorbed_by_aggregation, 1, "D_agg");
    // Both rows of the cancelling pair were eliminated.
    assert_eq!(st.eliminated_by_cancellation, 2, "D_elim");

    // Conservation: D_agg + D_elim == raw_rows - |supp(T)| == 6 - 3.
    assert_eq!(st.num_duplicates_absorbed, 3, "D_absorbed (unchanged total)");
    assert_eq!(
        st.absorbed_by_aggregation + st.eliminated_by_cancellation,
        st.num_duplicates_absorbed,
        "G8 conservation identity"
    );
    assert_eq!(
        st.absorbed_by_aggregation + st.eliminated_by_cancellation,
        st.num_transactions - st.num_unique_coordinates as u64
    );

    // The eliminated coordinate is absent from the canonical support entirely.
    let t_bob_alice = build.coo.iter().find(|e| e.s == 1 && e.r == 0).map(|e| e.t);
    assert_eq!(
        t_bob_alice, None,
        "the +/-7.25 cancelling pair must not appear in supp(T)"
    );

    // Deterministic: identical inputs, identical diagnostic (R_7).
    let again = build_from_reader(CSV.as_bytes()).unwrap();
    assert_eq!(build.settlement, again.settlement);
}

#[test]
fn api_is_a_read_only_reexposure_without_new_algebra() {
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    let csf = abacus_tensor::csf::compress(&build.coo);
    // time_slice (row 16), half-open.
    let t0 = build.t_min;
    assert_eq!(api::time_slice(&csf, t0, t0 + 1).len(), 1);
    assert!(api::time_slice(&csf, t0 + 1, t0).is_empty());
    // lookup: absent is None, never fabricated (P3 + row 27 R_2 convention).
    let first = &build.coo[0];
    assert_eq!(
        api::lookup(&csf, first.t, first.s, first.r),
        Some(first.value)
    );
    assert_eq!(api::lookup(&csf, 0, 0, 0), None);
    // decode: ⟦CSF(T)⟧ regroup only.
    assert_eq!(api::decode(&csf), build.coo);
    // deserialize + checksum delegate to row 17.
    let bytes = abacus_tensor::csf::serialize(&csf, &build);
    assert_eq!(api::checksum(&bytes), abacus_tensor::csf::checksum(&bytes));
    assert_eq!(api::deserialize(&bytes).unwrap().csf, csf);
}

#[test]
fn error_alphabet_is_fourfold_and_display_total() {
    use abacus_tensor::TensorError;
    let e: TensorError = std::io::Error::other("disk").into();
    assert!(matches!(e, TensorError::Io(_)));
    assert!(TensorError::Parse("x".into()).to_string().contains("parse"));
    assert!(TensorError::Range("x".into()).to_string().contains("range"));
    assert!(TensorError::Decode("x".into())
        .to_string()
        .contains("decode"));
}
