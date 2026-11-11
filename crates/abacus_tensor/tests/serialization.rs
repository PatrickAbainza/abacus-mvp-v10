//! Rows 17/25 (§B.2): serialization round-trips, checksums, fail-closed decode.
//!
//! The two checksum quantities are never conflated (P11): the `payload_checksum`
//! / `checksum` fields cover stored bytes, while whole-file quantities are
//! computed at the call site.

use abacus_tensor::coo::builder::build_from_reader;
use abacus_tensor::csf::{checksum, compress, deserialize, serialize};
use abacus_tensor::sidecar::{build_sidecar, deserialize_sidecar, serialize_sidecar};

const CSV: &str = include_str!("fixtures/small.csv");

#[test]
fn tensor_artifact_round_trips_losslessly() {
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    let csf = compress(&build.coo);
    let bytes = serialize(&csf, &build);
    let img = deserialize(&bytes).unwrap();
    assert_eq!(img.csf, csf);
    assert_eq!(img.senders, build.senders.table);
    assert_eq!(img.receivers, build.receivers.table);
    assert_eq!(img.raw_rows, build.raw_rows);
    assert_eq!((img.t_min, img.t_max), (build.t_min, build.t_max));
    assert_eq!(img.header.version, 1);
    assert_eq!(img.header.scale_exp, 6);
    // Serialization is deterministic (R_7): byte-identical on rebuild.
    assert_eq!(serialize(&csf, &build), bytes);
    // Whole-file checksum is a pure function of the bytes (d_fnv).
    assert_eq!(checksum(&bytes), abacus_tensor::composites::fnv1a64(&bytes));
}

#[test]
fn tensor_deserialize_rejects_corruption_fail_closed() {
    let build = build_from_reader(CSV.as_bytes()).unwrap();
    let bytes = serialize(&compress(&build.coo), &build);
    assert!(deserialize(&[]).is_err()); // bad magic
    let mut bad = bytes.clone();
    bad[0] = b'X';
    assert!(deserialize(&bad).is_err()); // bad magic
                                         // Corrupt the version field: unsupported version rejects.
    let mut bad = bytes.clone();
    bad[8] = 99; // unsupported version
    assert!(deserialize(&bad).is_err());
    // The provenance footer is NOT covered by the embedded payload checksum;
    // whole-file integrity is the separate artifact checksum (P11). Flipping a
    // footer byte therefore still decodes, and the whole-file checksum flags it.
    let mut foot = bytes.clone();
    let last = foot.len() - 1;
    foot[last] ^= 0xFF;
    assert!(deserialize(&foot).is_ok());
    assert_ne!(checksum(&foot), checksum(&bytes));
    let mut bad = bytes.clone();
    let mid = bad.len() / 2; // corrupt a payload byte → checksum mismatch
    bad[mid] ^= 0xFF;
    let err = deserialize(&bad).unwrap_err().to_string();
    assert!(err.contains("checksum"), "unexpected: {err}");
    let mut bad = bytes.clone();
    bad.truncate(40); // truncated payload
    assert!(deserialize(&bad).is_err());
}

#[test]
fn sidecar_round_trips_and_preserves_prec4_order() {
    let sidecar = build_sidecar(CSV.as_bytes()).unwrap();
    assert_eq!(sidecar.len(), 6); // full dataset rows in the fixture (both 0 and 1)
    assert!(sidecar.records.windows(2).all(|w| w[0] <= w[1]));
    let bytes = serialize_sidecar(&sidecar);
    let rt = deserialize_sidecar(&bytes).unwrap();
    assert_eq!(rt, sidecar);
    assert_eq!(serialize_sidecar(&sidecar), bytes); // deterministic
}

#[test]
fn sidecar_deserialize_rejects_reordered_or_corrupt_images() {
    let sidecar = build_sidecar(CSV.as_bytes()).unwrap();
    let bytes = serialize_sidecar(&sidecar);
    let mut bad = bytes.clone();
    bad[0] = b'X';
    assert!(deserialize_sidecar(&bad).is_err());
    // Flip bytes inside the records section: checksum mismatch.
    let mut bad = bytes.clone();
    let mid = bad.len() - 3;
    bad[mid] ^= 0xFF;
    assert!(deserialize_sidecar(&bad).is_err());
    // A swapped record pair violates the d_inv strict-order check (when the
    // fixture has ≥2 records with distinct keys) — construct by hand.
    let mut swapped = sidecar.records.clone();
    if swapped.len() >= 2 && swapped[0] != swapped[1] {
        swapped.swap(0, 1);
        let bad = serialize_sidecar(&abacus_tensor::sidecar::LaunderingSidecar::new(swapped));
        assert!(deserialize_sidecar(&bad).is_err());
    }
}
