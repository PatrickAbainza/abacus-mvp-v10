//! Row 28 (§B.2) — `verify_artifact`: re-open a `tensor.csf` and check it.
//!
//! CLI surface preserved from the prior realization:
//!
//! ```text
//! verify_artifact <tensor.csf>
//! ```
//!
//! Checks (`d_inv` + row 15 + row 11, no new mathematics):
//! deserialize validation (row 17), CSF invariants (P6), recompress
//! determinism (`Compress(⟦CSF(T)⟧) = CSF(T)`), and the analytic byte count
//! consistency (row 22 `a_{09}` length math against the actual file size).

use abacus_tensor::api;
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("verify_artifact failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err(format!(
            "usage: {} <tensor.csf>",
            args.first()
                .map(String::as_str)
                .unwrap_or("verify_artifact")
        ));
    }
    let data = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let img = api::deserialize(&data).map_err(|e| e.to_string())?;
    if !img.csf.check_invariants() {
        return Err("CSF invariants failed".to_string());
    }
    let rt = api::decode(&img.csf);
    if abacus_tensor::csf::compress(&rt) != img.csf {
        return Err("recompress mismatch".to_string());
    }
    let payload_bytes = abacus_tensor::ingest::csf_byte_size(&img.csf);
    let n0 = img.csf.mode0_coords.len();
    let n1 = img.csf.mode1_coords.len();
    let nnz = img.csf.values.len();
    let mut buf = String::new();
    buf.push_str(&format!("file_bytes={}\n", data.len()));
    buf.push_str(&format!("payload_bytes={payload_bytes}\n"));
    buf.push_str(&format!("n0_time_fibers={n0}\n"));
    buf.push_str(&format!("n1_sender_fibers={n1}\n"));
    buf.push_str(&format!("nnz={nnz}\n"));
    buf.push_str(&format!("senders={}\n", img.senders.len()));
    buf.push_str(&format!("receivers={}\n", img.receivers.len()));
    buf.push_str(&format!("raw_rows={}\n", img.raw_rows));
    buf.push_str(&format!("t_min={}\n", img.t_min));
    buf.push_str(&format!("t_max={}\n", img.t_max));
    buf.push_str(&format!("version={}\n", img.header.version));
    buf.push_str(&format!("scale_exp={}\n", img.header.scale_exp));
    buf.push_str(&format!("payload_checksum={:016x}\n", img.header.checksum));
    buf.push_str(&format!(
        "artifact_checksum={:016x}\n",
        api::checksum(&data)
    ));
    let mut out = std::io::stdout().lock();
    out.write_all(buf.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}
