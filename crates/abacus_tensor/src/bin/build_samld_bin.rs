//! Row 28 (§B.2) + addendum §§S4/S6 — `build_samld_bin`:
//! CSV → `samld_core.bin` + `samld_eval.sidecar` (+ `samld_stats.json`).
//!
//! CLI surface:
//!
//! ```text
//! build_samld_bin <input.csv> <samld_core.bin> <samld_eval.sidecar> <samld_stats.json>
//! ```
//!
//! Ingestion-boundary only (substrate invariant §2). No new mathematics:
//! row-14/24-shape convert + `d_fnv` checksums + `d_inv` order assertions.
//! Tensor construction reads `amount_micro` only (P2); the weighted amounts
//! (`log1p`, `z`, `roundedness`) are stored values; `tx_id` lives in the
//! eval sidecar (core key is `(t,s,r)`).

use abacus_tensor::samld::{convert_csv, deserialize_core, deserialize_eval, direct_csf};
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("build_samld_bin failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err(format!(
            "usage: {} <input.csv> <samld_core.bin> <samld_eval.sidecar> <samld_stats.json>",
            args.first().map(String::as_str).unwrap_or("build_samld_bin")
        ));
    }
    let conv = convert_csv(&args[1]).map_err(|e| e.to_string())?;
    // d_inv self-checks before writing: core order + direct-CSF equivalence.
    let img = deserialize_core(&conv.core_bytes).map_err(|e| e.to_string())?;
    let (csf, settled) = direct_csf(&img).map_err(|e| e.to_string())?;
    let eval = deserialize_eval(&conv.eval_bytes).map_err(|e| e.to_string())?;
    let n = img.records.len();
    if eval.len() != n {
        return Err("core/eval count mismatch (internal error)".to_string());
    }
    // Core↔eval correspondence: the sidecar is the sole `tx_id` carrier and
    // is emitted in core `(t,s,r)` order, so the join is a positional
    // bijection (index `i` ↔ record `i`) with exact index-set equality
    // (`sets::index_set`, set-operation rule) + a positional amount check.
    {
        use abacus_tensor::samld::sets;
        use std::collections::BTreeSet;
        let expected = sets::index_set(n);
        let eval_tx: BTreeSet<u64> = eval.iter().map(|r| r.tx_id).collect();
        let (missing, extra, _) = sets::tx_correspondence(&expected, &eval_tx);
        if !missing.is_empty() || !extra.is_empty() {
            return Err("core/eval tx_id correspondence failed".to_string());
        }
        for (r, e) in img.records.iter().zip(eval.iter()) {
            if r.amount_micro != e.amount_micro {
                return Err("core/eval amount cross-check failed".to_string());
            }
        }
    }
    std::fs::write(&args[2], &conv.core_bytes).map_err(|e| e.to_string())?;
    std::fs::write(&args[3], &conv.eval_bytes).map_err(|e| e.to_string())?;
    // Deterministic stats JSON (fixed key order, R_7).
    let js = format!(
        "{{\n  \"num_rows\": {},\n  \"num_unique_coordinates\": {},\n  \"num_duplicates_absorbed\": {},\n  \"num_senders\": {},\n  \"num_receivers\": {},\n  \"t_min\": {},\n  \"t_max\": {},\n  \"mean_major\": {:.10},\n  \"std_major\": {:.10},\n  \"nnz\": {},\n  \"t_fibers\": {},\n  \"s_fibers\": {},\n  \"core_bytes\": {},\n  \"eval_bytes\": {},\n  \"core_checksum\": {},\n  \"bit_report\": \"{}\"\n}}\n",
        conv.core_records.len(),
        settled.len(),
        conv.core_records.len().saturating_sub(settled.len()),
        conv.senders.len(),
        conv.receivers.len(),
        conv.t_min,
        conv.t_max,
        conv.mean_major,
        conv.std_major,
        csf.nnz(),
        csf.mode0_coords.len(),
        csf.mode1_coords.len(),
        conv.core_bytes.len(),
        conv.eval_bytes.len(),
        abacus_tensor::composites::fnv1a64(&conv.core_bytes),
        abacus_tensor::samld::bit_report(),
    );
    let mut f = std::fs::File::create(&args[4]).map_err(|e| e.to_string())?;
    f.write_all(js.as_bytes()).map_err(|e| e.to_string())?;
    println!(
        "rows={} nnz={} t_fibers={} s_fibers={} core_bytes={} eval_bytes={} {}",
        conv.core_records.len(),
        csf.nnz(),
        csf.mode0_coords.len(),
        csf.mode1_coords.len(),
        conv.core_bytes.len(),
        conv.eval_bytes.len(),
        abacus_tensor::samld::bit_report(),
    );
    Ok(())
}
