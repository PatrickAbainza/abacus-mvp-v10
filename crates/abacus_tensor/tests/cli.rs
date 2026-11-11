//! Row 28 (§B.2): CLI surface — build, verify, sidecar, usage errors.
//!
//! Each case runs a real binary against the shared fixture in a private
//! temporary directory; assertions check the externally observable behavior
//! (exit status, files written, stdout text) without inventing mathematics.

use std::path::PathBuf;
use std::process::Command;

const CSV: &str = include_str!("fixtures/small.csv");

fn tmpdir(name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("abacus_v6_cli_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn bin(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_BIN_EXE_build_samld"));
    p.set_file_name(name);
    assert!(p.exists(), "missing test binary {name}: {p:?}");
    p
}

fn write_csv(d: &std::path::Path) -> PathBuf {
    let csv = d.join("small.csv");
    std::fs::write(&csv, CSV).unwrap();
    csv
}

#[test]
fn build_samld_writes_artifact_and_stats() {
    let d = tmpdir("build");
    let csv = write_csv(&d);
    let csf = d.join("tensor.csf");
    let stats = d.join("stats.json");
    let out = Command::new(bin("build_samld"))
        .args([&csv, &csf, &stats])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&csf).unwrap();
    assert!(!bytes.is_empty());
    let js = std::fs::read_to_string(&stats).unwrap();
    assert!(js.contains("\"num_transactions\": 6"));
    assert!(js.contains("\"artifact_checksum\""));
    // Byte-identical rebuild: the same inputs give the same bytes (R_7).
    let csf2 = d.join("tensor2.csf");
    let stats2 = d.join("stats2.json");
    let out = Command::new(bin("build_samld"))
        .args([&csv, &csf2, &stats2])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(std::fs::read(&csf2).unwrap(), bytes);
}

#[test]
fn build_samld_usage_error_is_fail_closed() {
    let out = Command::new(bin("build_samld")).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("usage:"));
}

#[test]
fn verify_artifact_accepts_a_good_artifact() {
    let d = tmpdir("verify");
    let csv = write_csv(&d);
    let csf = d.join("tensor.csf");
    let stats = d.join("stats.json");
    assert!(Command::new(bin("build_samld"))
        .args([&csv, &csf, &stats])
        .status()
        .unwrap()
        .success());
    let out = Command::new(bin("verify_artifact"))
        .arg(&csf)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(text.contains("nnz=3") && text.contains("raw_rows=6"));
    // Corrupt a body byte (not the provenance footer): verification must fail
    // (R_5), not print a report. Corrupting the last byte only would hit the
    // t_max field with an in-range value and the checksum still rejects it in
    // this fixture — but flipping a payload byte is the robust case.
    let mut bytes = std::fs::read(&csf).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&csf, bytes).unwrap();
    assert!(!Command::new(bin("verify_artifact"))
        .arg(&csf)
        .status()
        .unwrap()
        .success());
}

#[test]
fn build_sidecar_reports_records_and_distribution() {
    let d = tmpdir("sidecar");
    let csv = write_csv(&d);
    let sc = d.join("laundering.sidecar");
    let out = Command::new(bin("build_sidecar"))
        .args([&csv, &sc])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(text.contains("records=6"), "stdout: {text}");
    assert!(
        text.contains("Fan Out=1") && text.contains("Scatter Gather=1") && text.contains("Normal=4"),
        "stdout: {text}"
    );
}
