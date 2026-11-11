//! Build the samld_full profile: CSV → all profile artifacts using the SAMLD
//! compacted generation (SAMCORE02 / SAMCSF02 / SAMEVL03), matching the
//! samld_90d_3rd profile structure.
//!
//! CLI surface:
//! ```text
//! build_samld_full <input.csv> <samld_core.bin> <samld_core.csf> <samld_eval.sidecar> <samld_stats.json> <samld_csf_stats.json>
//! ```
//!
//! Reuses existing functions only — no new mathematics:
//! - `samld::convert_csv` → staged rows, dictionaries, weighted moments
//! - `samld::deserialize_core_v2` → CoreImage from staged bytes
//! - `samld::direct_csf` → CSF tensor (settle in memory, compress)
//! - `samld::serialize_core_v2` → SAMCORE02 core.bin
//! - `samld::serialize_eval_v3` → SAMEVL03 eval sidecar
//! - `csf::compress_v2::serialize_csf_v2` → SAMCSF02 core.csf
//! - `stats::compute` + `stats::to_json_with_artifact` → CSF stats JSON
//! - `stats::DatasetStats::to_json` → samld_stats.json (same layout as
//!   `build_samld_bin`).

use abacus_tensor::csf::compress_v2::serialize_csf_v2;
use abacus_tensor::coo::builder::{AccountDict, CanonicalBuild};
use abacus_tensor::samld::{
    convert_csv, deserialize_core_v2, direct_csf, project_entries, serialize_core_v2,
    serialize_eval_v3, CoreImage,
};
use abacus_tensor::stats::{compute, settlement_diagnostic};

fn main() {
    if let Err(e) = run() {
        eprintln!("build_samld_full failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(format!(
            "usage: {} <input.csv> <samld_core.bin> <samld_core.csf> <samld_eval.sidecar> <samld_stats.json> <samld_csf_stats.json>",
            args.first().map(String::as_str).unwrap_or("build_samld_full")
        ));
    }

    let csv_path = &args[1];
    let core_bin_path = &args[2];
    let csf_path = &args[3];
    let eval_sidecar_path = &args[4];
    let stats_json_path = &args[5];
    let csf_stats_path = &args[6];

    // ---- Phase 1: CSV → staged core/eval bytes (same as build_samld_bin) ----
    let conv = convert_csv(csv_path).map_err(|e| e.to_string())?;

    // Project core records to canonical entries (needed for settlement stats).
    let raw_entries = project_entries(&conv.core_records);
    let raw_settlement = settlement_diagnostic(&raw_entries).map_err(|e| e.to_string())?;

    // Construct a CoreImage from the staged records + dictionaries so we can
    // feed it to serialize_core_v2 (SAMCORE02 v3) and direct_csf. Note:
    // conv.core_bytes was produced by the OLD SAMLDBIN v2 writer and is not
    // usable with SAMCORE02; we rebuild the image from canonical records.
    // core_records is consumed into the image below.
    let num_rows = conv.core_records.len();
    let img = CoreImage {
        records: conv.core_records,
        senders: conv.senders.clone(),
        receivers: conv.receivers.clone(),
        t_min: conv.t_min,
        t_max: conv.t_max,
        mean_major: conv.mean_major,
        std_major: conv.std_major,
        record_count: num_rows as u64,
    };

    // Serialize CoreImage to SAMCORE02 v3 bytes, validate by re-deserializing,
    // then feed the validated image to direct_csf.
    let img_bytes = serialize_core_v2(&img);
    let img_decoded = match deserialize_core_v2(&img_bytes) {
        Ok(i) => i,
        Err(e) => return Err(e.to_string()),
    };

    // ---- Phase 2: direct CSF (settle in memory, compress) ----
    let (csf, settled) = match direct_csf(&img_decoded) {
        Ok(d) => d,
        Err(e) => return Err(e.to_string()),
    };

    // ---- Phase 3: write artifacts ----

    // samld_core.bin → SAMCORE02 (new compacted format)
    let core_bytes_v2 = serialize_core_v2(&img);
    std::fs::write(core_bin_path, &core_bytes_v2).map_err(|e| e.to_string())?;

    // samld_core.csf → SAMCSF02 (new compacted format)
    let csf_bytes = serialize_csf_v2(
        &csf,
        &conv.senders,
        &conv.receivers,
        conv.t_min,
        conv.t_max,
        num_rows as u64,
    );
    std::fs::write(csf_path, &csf_bytes).map_err(|e| e.to_string())?;

    // samld_eval.sidecar → SAMEVL03 (new compacted format)
    let eval_bytes_v3 = serialize_eval_v3(&conv.eval_records).map_err(|e| e.to_string())?;
    std::fs::write(eval_sidecar_path, &eval_bytes_v3).map_err(|e| e.to_string())?;

    // ---- Phase 4: samld_stats.json (same format string as build_samld_bin) ----
    let js = format!(
        "{{\n  \"num_rows\": {}, \"num_unique_coordinates\": {}, \"num_duplicates_absorbed\": {}, \"num_senders\": {}, \"num_receivers\": {}, \"t_min\": {}, \"t_max\": {}, \"mean_major\": {:.10}, \"std_major\": {:.10}, \"nnz\": {}, \"t_fibers\": {}, \"s_fibers\": {}, \"core_bytes\": {}, \"eval_bytes\": {}, \"core_checksum\": {}, \"bit_report\": \"{}\"\n}}\n",
        num_rows,
        settled.len(),
        num_rows.saturating_sub(settled.len()),
        conv.senders.len(),
        conv.receivers.len(),
        conv.t_min,
        conv.t_max,
        conv.mean_major,
        conv.std_major,
        csf.nnz(),
        csf.mode0_coords.len(),
        csf.mode1_coords.len(),
        core_bytes_v2.len(),
        eval_bytes_v3.len(),
        abacus_tensor::composites::fnv1a64(&core_bytes_v2),
        abacus_tensor::samld::bit_report(),
    );
    std::fs::write(stats_json_path, js).map_err(|e| e.to_string())?;

    // ---- Phase 5: samld_csf_stats.json ----
    // Build the CanonicalBuild (same shape as build_from_csv's output) for
    // stats::compute.
    let build = CanonicalBuild {
        coo: settled,
        senders: AccountDict {
            table: conv.senders,
        },
        receivers: AccountDict {
            table: conv.receivers,
        },
        raw_rows: num_rows as u64,
        t_min: conv.t_min,
        t_max: conv.t_max,
        settlement: raw_settlement,
    };
    let stats = compute(&build, &csf);
    let csf_stats_json = stats.to_json_with_artifact(&csf_bytes, "samld_core.csf");
    std::fs::write(csf_stats_path, csf_stats_json).map_err(|e| e.to_string())?;

    println!(
        "rows={} nnz={} t_fibers={} s_fibers={} core_bytes={} eval_bytes={} csf_bytes={}",
        num_rows,
        csf.nnz(),
        csf.mode0_coords.len(),
        csf.mode1_coords.len(),
        core_bytes_v2.len(),
        eval_bytes_v3.len(),
        csf_bytes.len(),
    );
    Ok(())
}
