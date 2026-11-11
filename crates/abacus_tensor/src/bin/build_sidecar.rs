//! Row 28 (§B.2) — `build_sidecar`: CSV → `laundering.sidecar`.
//!
//! CLI surface preserved from the prior realization:
//!
//! ```text
//! build_sidecar <input.csv> <laundering.sidecar>
//! ```
//!
//! Reads only the source CSV (`Is_laundering` ∈ {0,1} — the complete binary
//! ground truth) and never touches `tensor.csf` or any tensor value path (P11).
//! No new mathematics: row 24 build + row 25 serialize with the `d_inv`
//! canonical-order and strictly-binary-label assertions, plus the lossless
//! round-trip self-check.

use abacus_tensor::sidecar::{build_sidecar_from_csv, deserialize_sidecar, serialize_sidecar};
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("build_sidecar failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!(
            "usage: {} <input.csv> <laundering.sidecar>",
            args.first().map(String::as_str).unwrap_or("build_sidecar")
        ));
    }
    let sidecar = build_sidecar_from_csv(&args[1]).map_err(|e| e.to_string())?;
    // d_inv (row 25): canonical prec_4 order (builder already sorts).
    if sidecar.records.windows(2).any(|w| w[0] > w[1]) {
        return Err("sidecar records not in canonical order (internal error)".to_string());
    }
    // Strictly binary laundering labels: {0, 1}.
    if sidecar.records.iter().any(|r| r.laundering > 1) {
        return Err("sidecar contains non-binary laundering record (internal error)".to_string());
    }
    let bytes = serialize_sidecar(&sidecar);
    // Row 25 lossless guarantee: Deserialize(Serialize(S)) = S.
    if deserialize_sidecar(&bytes).map_err(|e| e.to_string())? != sidecar {
        return Err("round-trip Deserialize(Serialize(S)) != S (internal error)".to_string());
    }
    let mut f = std::fs::File::create(&args[2]).map_err(|e| e.to_string())?;
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    let dist = sidecar.type_distribution();
    let dist_s: Vec<String> = dist.iter().map(|(k, v)| format!("{k}={v}")).collect();
    let mut sum_b = [0u8; 8];
    sum_b.copy_from_slice(&bytes[20..28]); // header: magic8+version4+count8+checksum8
    println!(
        "records={} types={} bytes={} checksum={:016x} dist:[{}]",
        sidecar.len(),
        dist.len(),
        bytes.len(),
        u64::from_le_bytes(sum_b),
        dist_s.join(",")
    );
    Ok(())
}
