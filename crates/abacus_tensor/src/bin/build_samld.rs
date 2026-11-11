//! Row 28 (§B.2) — `build_samld`: CSV → `tensor.csf` + `stats.json`.
//!
//! CLI surface preserved from the prior realization:
//!
//! ```text
//! build_samld <input.csv> <tensor.csf> <stats.json>
//! ```
//!
//! No new mathematics (row 28 is `a_{19}` file write over rows 11/14/17/26).
//! The internal consistency checks are `d_inv` obligations of row 11 plus the
//! §6 representation equivalence `⟦CSF(T)⟧ = COO(T)`, stated as the named
//! no-self-aggregation invariant (P6 + P3 + P4); a failure aborts with a
//! non-zero exit rather than writing a defective artifact (`R_5`).
//!
//! Provenance policy (P11): `tensor.csf` is the canonical reproducible
//! artifact; `stats.json` carries the whole-file `artifact_checksum`, kept
//! distinct from the embedded payload checksum, and records the invocation
//! output basename as provenance only.

use abacus_tensor::coo::builder::build_from_csv;
use abacus_tensor::csf::{compress, no_self_aggregation, serialize};
use abacus_tensor::stats;
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("build_samld failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(format!(
            "usage: {} <input.csv> <tensor.csf> <stats.json>",
            args.first().map(String::as_str).unwrap_or("build_samld")
        ));
    }
    let build = build_from_csv(&args[1]).map_err(|e| e.to_string())?;
    // Row 11 d_inv: the canonical COO must already be prec_3-sorted (K3).
    if build.coo.windows(2).any(|w| w[0] > w[1]) {
        return Err("canonical COO not sorted (internal error)".to_string());
    }
    let csf = compress(&build.coo);
    // Row 11 d_inv: CSF structural invariants (P6).
    if !csf.check_invariants() {
        return Err("CSF invariant check failed (internal error)".to_string());
    }
    // §6 representation equivalence, stated as the named no-self-aggregation
    // invariant (P6 + P3 + P4): CSF never aggregates — duplicates at identical
    // (t,s,r) were summed exactly once, by d_settle, before compression.
    if !no_self_aggregation(&build.coo, &csf) {
        return Err(
            "no-self-aggregation invariant failed: ⟦CSF(T)⟧ != COO(T) (internal error)"
                .to_string(),
        );
    }
    let bytes = serialize(&csf, &build);
    let mut f = std::fs::File::create(&args[2]).map_err(|e| e.to_string())?;
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    let st = stats::compute(&build, &csf);
    // Provenance-only invocation path, reduced to a deterministic basename.
    let prov = args[2].rsplit(['/', '\\']).next().unwrap_or(&args[2]);
    std::fs::write(&args[3], st.to_json_with_artifact(&bytes, prov)).map_err(|e| e.to_string())?;
    println!(
        "rows={} nnz={} t_fibers={} s_fibers={} bytes={} checksum={:016x}",
        st.num_transactions,
        st.num_unique_coordinates,
        csf.mode0_coords.len(),
        csf.mode1_coords.len(),
        bytes.len(),
        abacus_tensor::csf::checksum(&bytes)
    );
    Ok(())
}
