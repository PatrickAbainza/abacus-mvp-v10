//! Row 28 (§B.2) + addendum §R5f — `pack_eval_v3`:
//! `SAMEVL01` eval sidecar → derived, lossless `SAMEVL03` eval sidecar.
//!
//! ```text
//! pack_eval_v3 <in.sidecar SAMEVL01> <out.sidecar.v3 SAMEVL03>
//! ```
//!
//! Derived, lossless P11 provenance encoding (never read by tensor logic).
//! Self-checks the round-trip before writing (`R_5`).

use abacus_tensor::composites::fnv1a64;
use abacus_tensor::samld::{deserialize_eval, deserialize_eval_v3, serialize_eval_v3};
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("pack_eval_v3 failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!(
            "usage: {} <in.sidecar SAMEVL01> <out.sidecar.v3 SAMEVL03>",
            args.first().map(String::as_str).unwrap_or("pack_eval_v3")
        ));
    }
    let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let records = deserialize_eval(&bytes).map_err(|e| e.to_string())?;
    let v3 = serialize_eval_v3(&records).map_err(|e| e.to_string())?;
    if deserialize_eval_v3(&v3).map_err(|e| e.to_string())? != records {
        return Err("round-trip Decode_v3(Encode_v3(S)) != S (internal error)".to_string());
    }
    let mut f = std::fs::File::create(&args[2]).map_err(|e| e.to_string())?;
    f.write_all(&v3).map_err(|e| e.to_string())?;
    let ratio = v3.len() as f64 / bytes.len() as f64;
    println!(
        "records={} v1_bytes={} v3_bytes={} ratio={:.3} checksum={:016x}",
        records.len(),
        bytes.len(),
        v3.len(),
        ratio,
        fnv1a64(&v3)
    );
    Ok(())
}
