//! STEP 4: canonical schema and binary-layout conformance.
//!
//! Covers task §4 "Canonical schema and binary-layout conformance", "Every
//! supported account-ID width", "Valid and malformed headers and records",
//! "Checksum failures and truncated payloads", and the wire §8 required
//! golden-header tests.

mod common;

use abacus_ingest::config::{ACCOUNT_ID_WIDTHS, HEADER_LEN, MAGIC};
use abacus_ingest::error::ErrorCode;
use abacus_ingest::record::record_length_for_width;
use abacus_ingest::wire::{build_segment, Segment};
use common::{config, ns, record, resign_header, resign_payload_then_header};

#[test]
fn header_is_exactly_176_bytes() {
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    let bytes = seg.encode().unwrap();
    assert_eq!(bytes.len(), HEADER_LEN + record_length_for_width(4).unwrap());
    // The header is the first 176 bytes; the payload follows immediately.
    assert_eq!(&bytes[..8], MAGIC);
}

#[test]
fn header_length_field_declares_176() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    // header_length is u16 LE at offset 10.
    let declared = u16::from_le_bytes([bytes[10], bytes[11]]);
    assert_eq!(declared as usize, HEADER_LEN);
    // format_version u16 LE at offset 8.
    assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 1);
    // flags u32 at 12 must be zero in v1.
    assert_eq!(&bytes[12..16], &[0, 0, 0, 0]);
    // reserved byte at 79 must be zero.
    assert_eq!(bytes[79], 0);
}

#[test]
fn record_length_table_for_every_width() {
    // Wire §3 exact table.
    assert_eq!(record_length_for_width(1).unwrap(), 25);
    assert_eq!(record_length_for_width(2).unwrap(), 27);
    assert_eq!(record_length_for_width(4).unwrap(), 31);
    assert_eq!(record_length_for_width(8).unwrap(), 39);
}

#[test]
fn golden_bytes_roundtrip_every_width() {
    // Wire §8: "Every supported account-ID width and its required record
    // length" plus "Successful decoding of a known header and payload into the
    // expected canonical events."
    for w in ACCOUNT_ID_WIDTHS {
        let max = if w == 8 { u64::MAX } else { (1u64 << (8 * u32::from(w))) - 1 };
        let recs = vec![
            record(1, 1_700_000_000_000, 1, 2, 50_000),
            record(2, 1_700_000_060_000, max, max - 1, 99_999),
        ];
        let seg = build_segment(&config(w), ns(), &recs).unwrap();
        let bytes = seg.encode().unwrap();
        // Byte-exact length.
        assert_eq!(
            bytes.len(),
            HEADER_LEN + 2 * record_length_for_width(w).unwrap(),
            "width {w}"
        );
        // Deterministic: encoding twice gives identical bytes.
        assert_eq!(bytes, seg.encode().unwrap(), "width {w} not deterministic");
        // Decodes back to the identical logical events.
        let back = Segment::decode(&bytes).unwrap();
        assert_eq!(back.records, recs, "width {w} roundtrip");
        assert_eq!(back.namespace(), ns());
    }
}

#[test]
fn namespace_is_stored_in_canonical_uuid_network_order() {
    // Wire §8: "Canonical UUID byte representation."
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    // event_namespace occupies offsets 40..56 verbatim.
    assert_eq!(&bytes[40..56], &ns().to_bytes());
}

#[test]
fn payload_checksum_covers_only_the_payload() {
    // Wire §4: "It excludes the header."
    let recs = vec![record(1, 1_700_000_000_000, 1, 2, 50_000)];
    let seg = build_segment(&config(4), ns(), &recs).unwrap();
    let bytes = seg.encode().unwrap();
    // The stored payload_checksum (offsets 80..112) must equal SHA-256 of the
    // payload bytes only, i.e. everything after the 176-byte header.
    let payload = &bytes[HEADER_LEN..];
    let computed = abacus_ingest::hash::sha256(payload);
    assert_eq!(&bytes[80..112], &computed[..]);
    // And it must NOT be the digest of the whole buffer.
    assert_ne!(&bytes[80..112], &abacus_ingest::hash::sha256(&bytes)[..]);
}

#[test]
fn header_checksum_covers_bytes_0_to_143() {
    // Wire §4: "SHA-256 of bytes 0-143, inclusive. It includes the payload
    // checksum and configuration fingerprint. It excludes the header-checksum
    // field itself."
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    assert_eq!(&bytes[144..176], &abacus_ingest::hash::sha256(&bytes[..144])[..]);
    // The stored header checksum validates, which decode proves.
    Segment::decode(&bytes).unwrap();
}

#[test]
fn rejects_modified_header_bytes() {
    // Wire §8: "Rejection of modified header bytes." A bit flip anywhere in
    // 0..144 breaks the header checksum, which wire §6 step 3 catches first.
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    let bytes = seg.encode().unwrap();
    for offset in [16usize, 20, 24, 32, 40, 56, 59, 64, 72, 76, 77, 112] {
        let mut bad = bytes.clone();
        bad[offset] ^= 0x01;
        assert_eq!(
            Segment::decode(&bad).unwrap_err().code,
            ErrorCode::Binary,
            "offset {offset} was not rejected"
        );
    }
}

/// Mutate a header field and re-sign, so validation reaches the field check.
///
/// Wire §6 orders the header checksum (step 3) *before* the field checks
/// (steps 4-5), so a raw mutation is always caught by the checksum. Re-signing
/// models a producer that legitimately declared the mutated value.
fn mutate_header_field<F: FnOnce(&mut [u8])>(bytes: &[u8], f: F) -> Vec<u8> {
    let mut bad = bytes.to_vec();
    f(&mut bad);
    resign_header(&mut bad);
    bad
}

#[test]
fn rejects_invalid_magic() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    // Magic is checked before the checksum, so no re-sign is needed.
    let mut bad = seg.encode().unwrap();
    bad[0] = b'X';
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_unsupported_versions() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    // format_version = 2 (offset 8).
    let bad = mutate_header_field(&bytes, |b| b[8] = 2);
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::SchemaVersion);
    // feature_schema_version = 99 (offset 60).
    let bad2 = mutate_header_field(&bytes, |b| {
        b[60] = 99;
        b[61] = 0;
    });
    assert_eq!(Segment::decode(&bad2).unwrap_err().code, ErrorCode::SchemaVersion);
    // feature_algorithm_version = 99 (offset 62).
    let bad3 = mutate_header_field(&bytes, |b| {
        b[62] = 99;
        b[63] = 0;
    });
    assert_eq!(Segment::decode(&bad3).unwrap_err().code, ErrorCode::SchemaVersion);
}

#[test]
fn rejects_wrong_header_length_field() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bad = mutate_header_field(&seg.encode().unwrap(), |b| b[10] = 180);
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_nonzero_reserved_and_flags() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    let bad = mutate_header_field(&bytes, |b| b[79] = 1); // reserved
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::Binary);
    let bad2 = mutate_header_field(&bytes, |b| b[12] = 1); // flags
    assert_eq!(Segment::decode(&bad2).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_unsupported_enum_values() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    // timestamp_encoding, byte_order, checksum_algorithm, source_type,
    // identity_scheme — each must reject an out-of-range enum value.
    for (name, offset) in [
        ("timestamp_encoding", 18usize),
        ("byte_order", 19),
        ("checksum_algorithm", 78),
        ("source_type", 76),
        ("identity_scheme", 77),
    ] {
        let bad = mutate_header_field(&bytes, |b| b[offset] = 99);
        assert!(
            Segment::decode(&bad).is_err(),
            "{name} accepted an unsupported enum value"
        );
    }
}

#[test]
fn rejects_unsupported_account_id_width_including_128_bit() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    for w in [0u8, 3, 5, 16] {
        let bad = mutate_header_field(&bytes, |b| b[16] = w);
        assert_eq!(
            Segment::decode(&bad).unwrap_err().code,
            ErrorCode::AccountId,
            "width {w}"
        );
    }
}

#[test]
fn rejects_wrong_event_id_width() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bad = mutate_header_field(&seg.encode().unwrap(), |b| b[17] = 4);
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_truncated_header() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    for len in [0usize, 8, 100, HEADER_LEN - 1] {
        assert_eq!(
            Segment::decode(&bytes[..len]).unwrap_err().code,
            ErrorCode::Binary,
            "len {len}"
        );
    }
}

#[test]
fn rejects_truncated_payload() {
    // Wire §8: "Rejection of inconsistent record counts and payload lengths"
    // and §3 "No partial trailing record is allowed."
    let seg = build_segment(
        &config(4),
        ns(),
        &[
            record(1, 0, 1, 2, 5_000),
            record(2, 0, 3, 4, 6_000),
        ],
    )
    .unwrap();
    let bytes = seg.encode().unwrap();
    // Drop the last byte: a partial trailing record.
    let truncated = &bytes[..bytes.len() - 1];
    assert_eq!(Segment::decode(truncated).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_oversized_payload() {
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    let mut bytes = seg.encode().unwrap();
    bytes.extend_from_slice(&[0u8; 4]); // trailing garbage
    assert_eq!(Segment::decode(&bytes).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_corrupted_payload_checksum() {
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    let mut bytes = seg.encode().unwrap();
    // Flip a bit inside the payload.
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    assert_eq!(Segment::decode(&bytes).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_inconsistent_record_count_and_payload_length() {
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    // Declare record_count = 2 (offset 24, u64 LE) while the payload holds 1.
    let bad = mutate_header_field(&seg.encode().unwrap(), |b| b[24] = 2);
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_record_length_that_disagrees_with_the_width() {
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    // record_length (offset 20, u32 LE) = 39, which belongs to width 8.
    let bad = mutate_header_field(&seg.encode().unwrap(), |b| b[20] = 39);
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::AccountId);
}

#[test]
fn rejects_payload_length_that_disagrees_with_the_records() {
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    // payload_length (offset 32, u64 LE) = 999 while the real payload is 31.
    let bad = mutate_header_field(&seg.encode().unwrap(), |b| b[32] = 0xe7);
    assert_eq!(Segment::decode(&bad).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_invalid_currency_and_threshold() {
    let seg = build_segment(&config(4), ns(), &[]).unwrap();
    let bytes = seg.encode().unwrap();
    // Lowercase currency code (offset 56..59).
    let bad = mutate_header_field(&bytes, |b| b[56] = b'u');
    assert_eq!(
        Segment::decode(&bad).unwrap_err().code,
        ErrorCode::AmountOrCurrency
    );
    // Zero threshold (offset 64, u64 LE).
    let bad2 = mutate_header_field(&bytes, |b| {
        for b in &mut b[64..72] {
            *b = 0;
        }
    });
    assert_eq!(
        Segment::decode(&bad2).unwrap_err().code,
        ErrorCode::AmountOrCurrency
    );
    // Unsupported precision (offset 59).
    let bad3 = mutate_header_field(&bytes, |b| b[59] = 250);
    assert_eq!(
        Segment::decode(&bad3).unwrap_err().code,
        ErrorCode::AmountOrCurrency
    );
}

#[test]
fn rejects_reserved_roundedness_bit_in_a_record() {
    // Build a valid segment, then set a reserved feature bit in the payload and
    // re-sign both checksums, so the rejection comes from the RECORD check
    // (wire §6 step 8) rather than from a checksum mismatch.
    let seg = build_segment(&config(4), ns(), &[record(1, 0, 1, 2, 5_000)]).unwrap();
    let mut bytes = seg.encode().unwrap();
    // roundedness_flags is the last byte of the record, at offset 22+2W = 30.
    let last = bytes.len() - 1;
    bytes[last] = 0x10;
    resign_payload_then_header(&mut bytes);
    assert_eq!(Segment::decode(&bytes).unwrap_err().code, ErrorCode::Binary);
}

#[test]
fn rejects_pre_epoch_event_time_in_a_record() {
    // The substrate coordinate is u64 seconds since epoch, so a negative
    // millisecond timestamp can never be applied. Build the record, then patch
    // its time field and re-sign.
    let mut r = record(1, 0, 1, 2, 5_000);
    r.event_time_ms = -1;
    let seg = build_segment(&config(4), ns(), &[r]).unwrap();
    let mut bytes = seg.encode().unwrap();
    // The record's event_time_ms occupies bytes 8..16 of the payload.
    bytes[HEADER_LEN + 8..HEADER_LEN + 16].copy_from_slice(&(-1i64).to_le_bytes());
    resign_payload_then_header(&mut bytes);
    // The segment decodes (the record layout is valid); the pre-epoch
    // rejection happens at the application boundary.
    let decoded = Segment::decode(&bytes).unwrap();
    assert_eq!(decoded.records[0].event_time_ms, -1);
    assert!(abacus_ingest::event_time_ms_to_seconds(-1).is_err());
}

#[test]
fn zero_namespace_in_header_is_rejected() {
    // Wire §5: the namespace is part of the identity; an all-zero namespace is
    // not a valid identity domain.
    let seg = build_segment(&config(4), abacus_ingest::EventNamespace::from_bytes([0u8; 16]), &[]);
    // The builder does not itself reject, but decode must.
    let mut bytes = seg.unwrap().encode().unwrap();
    for b in &mut bytes[40..56] {
        *b = 0;
    }
    assert_eq!(Segment::decode(&bytes).unwrap_err().code, ErrorCode::Identity);
}
