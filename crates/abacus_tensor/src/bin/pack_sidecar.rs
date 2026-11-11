//! Row 28 (§B.2) + addendum §R5d — `pack_sidecar`:
//! `SAMEVL01` eval sidecar → compressed `SAMEVL02` eval sidecar.
//!
//! ```text
//! pack_sidecar <in.sidecar SAMEVL01> <out.sidecar SAMEVL02>
//! ```
//!
//! Derived, lossless column encoding (P11 provenance; never read by tensor
//! logic). Self-checks the round-trip before writing (`R_5` fail-closed).

use abacus_tensor::samld::{deserialize_eval, deserialize_eval_v2, serialize_eval_v2};
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("pack_sidecar failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!(
            "usage: {} <in.sidecar SAMEVL01> <out.sidecar SAMEVL02>",
            args.first().map(String::as_str).unwrap_or("pack_sidecar")
        ));
    }
    let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let records = deserialize_eval(&bytes).map_err(|e| e.to_string())?;
    let v2 = serialize_eval_v2(&records).map_err(|e| e.to_string())?;
    // Lossless guarantee: Decode_v2(Encode_v2(S)) == S.
    if deserialize_eval_v2(&v2).map_err(|e| e.to_string())? != records {
        return Err("round-trip Decode_v2(Encode_v2(S)) != S (internal error)".to_string());
    }
    let mut f = std::fs::File::create(&args[2]).map_err(|e| e.to_string())?;
    f.write_all(&v2).map_err(|e| e.to_string())?;
    let ratio = v2.len() as f64 / bytes.len() as f64;
    let mut s = [0u8; 8];
    s.copy_from_slice(&v2[20..28]);
    println!(
        "records={} v1_bytes={} v2_bytes={} ratio={:.3} checksum={:016x}",
        records.len(),
        bytes.len(),
        v2.len(),
        ratio,
        u64::from_le_bytes(s)
    );
    Ok(())
}
