//! Row 28 (§B.2) + addendum §R5f — `pack_core`:
//! `SAMLDBIN` v2 core → derived, lossless `SAMCORE02` core.
//!
//! ```text
//! pack_core <in.core.bin SAMLDBIN> <out.core.bin.v2 SAMCORE02>
//! ```
//!
//! Per-transaction P4 stream is retained (duplicates kept); the three P11
//! weighted columns are recomputed bit-exactly on decode. Self-checks the
//! round-trip before writing (`R_5`).

use abacus_tensor::composites::fnv1a64;
use abacus_tensor::samld::{deserialize_core, deserialize_core_v2, serialize_core_v2};
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("pack_core failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!(
            "usage: {} <in.core.bin SAMLDBIN> <out.core.bin.v2 SAMCORE02>",
            args.first().map(String::as_str).unwrap_or("pack_core")
        ));
    }
    let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let img = deserialize_core(&bytes).map_err(|e| e.to_string())?;
    let v2 = serialize_core_v2(&img);
    let back = deserialize_core_v2(&v2).map_err(|e| e.to_string())?;
    if back.records.len() != img.records.len()
        || back.senders != img.senders
        || back.receivers != img.receivers
        || back.mean_major.to_bits() != img.mean_major.to_bits()
        || back.std_major.to_bits() != img.std_major.to_bits()
    {
        return Err("round-trip Decode_v2(Encode_v2(C)) != C (internal error)".to_string());
    }
    for (a, b) in img.records.iter().zip(back.records.iter()) {
        if (a.t, a.s, a.r, a.amount_micro) != (b.t, b.s, b.r, b.amount_micro)
            || a.log1p_amt.to_bits() != b.log1p_amt.to_bits()
            || a.z_amt.to_bits() != b.z_amt.to_bits()
            || a.roundedness != b.roundedness
        {
            return Err("core2 record mismatch (internal error)".to_string());
        }
    }
    let mut f = std::fs::File::create(&args[2]).map_err(|e| e.to_string())?;
    f.write_all(&v2).map_err(|e| e.to_string())?;
    let ratio = v2.len() as f64 / bytes.len() as f64;
    println!(
        "records={} v1_bytes={} v2_bytes={} ratio={:.3} checksum={:016x}",
        img.records.len(),
        bytes.len(),
        v2.len(),
        ratio,
        fnv1a64(&v2)
    );
    Ok(())
}
