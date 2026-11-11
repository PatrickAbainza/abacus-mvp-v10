//! End-to-end artifact verification of a materialised SAMLD profile against
//! its **source CSV** (addendum §R5e).
//!
//! Opt-in: requires two environment variables, so the default test run stays
//! fast and artifact-independent.
//!
//! ```text
//! ABACUS_PROFILE_CSV=/path/SAML-D-90days_3rd.csv
//! ABACUS_PROFILE_DIR=/path/artifacts/profiles/samld_90d_3rd
//! cargo test -p abacus_tensor --test samld_profile_verify -- --ignored
//! ```
//!
//! Verifies, from the CSV as the only source of truth, the **surviving §R5f
//! generation** (the superseded `SAMLDBIN`/`ABACUS01`/`SAMEVL01`/`SAMEVL02`
//! generations were removed by the 2026-10-03 cleanup):
//! 1. `samld_core.bin`   (SAMCORE02) — decode round-trip vs the CSV rows,
//!    counts/moments bit-exact, `(t,s,r)` order, deterministic re-emission.
//! 2. `samld_eval.sidecar` (SAMEVL03) — decode round-trip over all ten P11
//!    attributes, `tx_id` index-set equality, positional core<->sidecar
//!    correspondence, deterministic re-emission.
//! 3. `samld_core.csf`   (SAMCSF02) — equals the CSV's settled tensor
//!    coordinate-for-coordinate and value-for-value, plus P6 fiber invariants,
//!    dictionaries and provenance footer; `nnz` unchanged (no aggregation).

use abacus_tensor::api;
use abacus_tensor::csf::{deserialize_csf_v2, serialize_csf_v2};
use abacus_tensor::samld::{
    convert_csv, deserialize_core_v2, deserialize_eval_v3, index_set, serialize_core_v2,
    serialize_eval_v3,
};
use std::collections::{BTreeMap, BTreeSet};

fn profile_paths() -> Option<(String, String)> {
    let csv = std::env::var("ABACUS_PROFILE_CSV").ok()?;
    let dir = std::env::var("ABACUS_PROFILE_DIR").ok()?;
    Some((csv, dir))
}

#[test]
#[ignore = "opt-in: set ABACUS_PROFILE_CSV and ABACUS_PROFILE_DIR"]
fn profile_artifacts_are_mathematically_aligned_to_source_csv() {
    let Some((csv_path, dir)) = profile_paths() else {
        panic!("set ABACUS_PROFILE_CSV and ABACUS_PROFILE_DIR");
    };
    let read = |n: &str| {
        std::fs::read(format!("{dir}/{n}"))
            .unwrap_or_else(|e| panic!("read {dir}/{n}: {e}"))
    };
    // The surviving generation (§R5f): SAMCSF02 / SAMCORE02 / SAMEVL03.
    let core_bytes = read("samld_core.bin");
    let eval_bytes = read("samld_eval.sidecar");
    let csf_bytes = read("samld_core.csf");

    // ---- the CSV is the only source of truth ----
    let conv = convert_csv(&csv_path).expect("convert csv");
    let n = conv.core_records.len();
    assert_eq!(n, conv.eval_records.len(), "one core record per CSV row");
    assert!(n > 0, "non-empty profile");

    // ---- 1. core (SAMCORE02) — every CSV row, weights recomputed bit-exactly ----
    let img = deserialize_core_v2(&core_bytes).expect("decode core");
    assert_eq!(serialize_core_v2(&img), core_bytes, "core re-emission deterministic (R7)");
    assert_eq!(img.record_count as usize, n);
    assert_eq!(img.records, conv.core_records, "core decode == CSV rows (dups kept)");
    assert_eq!(img.senders, conv.senders, "sender dictionary");
    assert_eq!(img.receivers, conv.receivers, "receiver dictionary");
    assert_eq!(img.t_min, conv.t_min);
    assert_eq!(img.t_max, conv.t_max);
    assert_eq!(img.mean_major.to_bits(), conv.mean_major.to_bits(), "mean bit-exact");
    assert_eq!(img.std_major.to_bits(), conv.std_major.to_bits(), "std bit-exact");
    // P1 storage order: non-decreasing (t,s,r); P4 duplicates permitted.
    for w in img.records.windows(2) {
        assert!(w[0].storage_key() <= w[1].storage_key(), "core not in d_lex order");
    }
    // Every core coordinate must come from a real CSV row (it does by
    // construction) and every amount must be the P2 canonical micro value.
    let csv_sum: i64 = conv.core_records.iter().map(|r| r.amount_micro).sum();
    assert!(csv_sum > 0, "nonzero total amount");

    // ---- 2. eval sidecar (SAMEVL03): round-trip + index-set correspondence ----
    let ev = deserialize_eval_v3(&eval_bytes).expect("decode sidecar");
    assert_eq!(ev, conv.eval_records, "sidecar decode round-trip (all 10 P11 attrs)");
    let tx: BTreeSet<u64> = ev.iter().map(|e| e.tx_id).collect();
    let (missing, extra, both) = {
        let expected = index_set(n);
        let (a, b, c) = abacus_tensor::samld::tx_correspondence(&expected, &tx);
        (a, b, c)
    };
    assert!(missing.is_empty() && extra.is_empty() && both.len() == n,
        "tx_id set equality with {{0..n-1}}");
    // positional core <-> sidecar correspondence + amount cross-check
    for (i, e) in ev.iter().enumerate() {
        assert_eq!(e.amount_micro, conv.core_records[i].amount_micro,
            "sidecar/core amount mismatch at {i}");
    }
    assert_eq!(
        serialize_eval_v3(&ev).unwrap(), eval_bytes,
        "sidecar re-emission deterministic (R7)"
    );

    // ---- 3. CSF (SAMCSF02) == CSV settled tensor ----
    let csf_img = deserialize_csf_v2(&csf_bytes).expect("decode csf");
    assert!(csf_img.csf.check_invariants(), "P6 fiber invariants");
    let decoded = api::decode(&csf_img.csf);
    // Settle the CSV rows by coordinate (d_settle): sum amounts per (t,s,r).
    let mut settled: BTreeMap<(u64, u32, u32), i64> = BTreeMap::new();
    for r in &conv.core_records {
        *settled.entry(r.storage_key()).or_insert(0) += r.amount_micro;
    }
    assert_eq!(decoded.len(), settled.len(), "CSF nnz == settled coordinate count");
    for e in &decoded {
        let k = (e.t, e.s, e.r);
        assert_eq!(Some(&e.value), settled.get(&k), "CSF value at {k:?}");
    }
    assert_eq!(
        decoded.iter().map(|e| e.value).sum::<i64>(),
        csv_sum,
        "CSF value total == CSV amount total"
    );
    // dictionaries and provenance footer agree with the CSV-derived facts
    assert_eq!(csf_img.senders, conv.senders);
    assert_eq!(csf_img.receivers, conv.receivers);
    assert_eq!(csf_img.t_min, conv.t_min);
    assert_eq!(csf_img.t_max, conv.t_max);
    assert_eq!(csf_img.raw_rows as usize, n);
    assert_eq!(csf_img.header.scale_exp, 6, "P2 canonical scale");

    // ---- 4. §R5f: the surviving generation is structure-preserving (no
    // aggregation) and deterministically re-emittable (R7) ----
    assert_eq!(
        csf_img.csf.nnz(),
        decoded.len(),
        "SAMCSF02 nnz unchanged (no bucket, no re-index)"
    );
    assert_eq!(
        serialize_csf_v2(
            &csf_img.csf,
            &csf_img.senders,
            &csf_img.receivers,
            csf_img.t_min,
            csf_img.t_max,
            csf_img.raw_rows,
        ),
        csf_bytes,
        "SAMCSF02 re-emission deterministic (R7)"
    );
    // P4: the core retains every CSV row, so nothing was merged away.
    assert_eq!(
        img.records.len(),
        conv.core_records.len(),
        "SAMCORE02 record count == CSV rows"
    );
    assert_eq!(
        img.records.iter().map(|r| r.amount_micro).sum::<i64>(),
        csv_sum,
        "SAMCORE02 amount total == CSV total"
    );
}

