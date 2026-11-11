//! Row 28 (§B.2) + addendum §R5f — `pack_csf`:
//! canonical `ABACUS01` CSF → derived, lossless `SAMCSF02` CSF.
//!
//! ```text
//! pack_csf <in.csf ABACUS01> <out.csf.v2 SAMCSF02>
//! ```
//!
//! Non-aggregating representation codec (one slot per canonical entry).
//! Self-checks `Decode_v2(Encode_v2(T)) == T` before writing (`R_5`).

use abacus_tensor::composites::fnv1a64;
use abacus_tensor::csf::{deserialize, deserialize_csf_v2, serialize_csf_v2};
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("pack_csf failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!(
            "usage: {} <in.csf ABACUS01> <out.csf.v2 SAMCSF02>",
            args.first().map(String::as_str).unwrap_or("pack_csf")
        ));
    }
    let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let img = deserialize(&bytes).map_err(|e| e.to_string())?;
    let v2 = serialize_csf_v2(
        &img.csf, &img.senders, &img.receivers, img.t_min, img.t_max, img.raw_rows,
    );
    // Lossless, structure-preserving guarantee (no aggregation).
    let back = deserialize_csf_v2(&v2).map_err(|e| e.to_string())?;
    if back.csf != img.csf || back.senders != img.senders || back.receivers != img.receivers {
        return Err("round-trip Decode_v2(Encode_v2(T)) != T (internal error)".to_string());
    }
    let mut f = std::fs::File::create(&args[2]).map_err(|e| e.to_string())?;
    f.write_all(&v2).map_err(|e| e.to_string())?;
    let ratio = v2.len() as f64 / bytes.len() as f64;
    println!(
        "nnz={} v1_bytes={} v2_bytes={} ratio={:.3} checksum={:016x}",
        img.csf.nnz(),
        bytes.len(),
        v2.len(),
        ratio,
        fnv1a64(&v2)
    );
    Ok(())
}
