//! Benchmark: ordered core.bin → direct CSF + tensor state + delta +
//! set selection + algebra + behavioral relations (task §5, measured).
//! Reads core.bin ONLY (never CSV — operational path is binary).
//! Usage: samld_bin_bench <samld_core.bin> [late_count]

use abacus_tensor::ingest::{Event, IngestionEngine, LedgerOp};
use abacus_tensor::samld::{deserialize_core, direct_csf, sets};
use abacus_tensor::tensor::coordinate::Coord;
use std::collections::BTreeSet;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("samld_bin_bench failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        return Err(format!(
            "usage: {} <samld_core.bin> [late_count]",
            args.first().map(String::as_str).unwrap_or("samld_bin_bench")
        ));
    }
    let late_n: usize = args.get(2).map(|s| s.parse().unwrap_or(1000)).unwrap_or(1000);
    let t0 = Instant::now();
    let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let read_s = t0.elapsed();
    let t1 = Instant::now();
    let img = deserialize_core(&bytes).map_err(|e| e.to_string())?;
    let decode_s = t1.elapsed();
    let t2 = Instant::now();
    let (csf, settled) = direct_csf(&img).map_err(|e| e.to_string())?;
    let csf_s = t2.elapsed();
    // Tensor state: canonical object from settled entries.
    let t3 = Instant::now();
    let tensor =
        abacus_tensor::tensor::Tensor::try_from_entries(settled.clone()).map_err(|e| e.to_string())?;
    let state_s = t3.elapsed();
    // Point-read sample (deterministic stride, no decode).
    let t4 = Instant::now();
    let mut acc: i64 = 0;
    let stride = (settled.len() / 10_000).max(1);
    let mut reads = 0usize;
    let mut i = 0;
    while i < settled.len() {
        let e = &settled[i];
        acc = acc.wrapping_add(abacus_tensor::csf::lookup(&csf, e.t, e.s, e.r));
        reads += 1;
        i += stride;
    }
    let read_csf_s = t4.elapsed();
    // Delta updates: late arrivals (fresh EventIds) → seal.
    let t5 = Instant::now();
    let mut eng = IngestionEngine::new();
    // `tx_id` now lives in the eval sidecar, not the core; the ledger
    // EventId is the core record index (unique, deterministic, §S6).
    for (i, r) in img.records.iter().enumerate() {
        eng.ingest(
            &Event { id: i as u64, coord: Coord::new(r.t, r.s, r.r) },
            LedgerOp::Create(r.amount_micro),
        )
        .map_err(|e| e.to_string())?;
    }
    eng.seal().map_err(|e| e.to_string())?;
    let e0 = &settled[0];
    for k in 0..late_n {
        eng.ingest(
            &Event {
                id: 9_000_000_000 + k as u64,
                coord: Coord::new(e0.t, e0.s, e0.r),
            },
            LedgerOp::Create(1),
        )
        .map_err(|e| e.to_string())?;
    }
    let live = eng.live_at(e0.t, e0.s, e0.r).map_err(|e| e.to_string())?;
    let rep = eng.seal().map_err(|e| e.to_string())?;
    let delta_s = t5.elapsed();
    // Set-based selection: support set ops + time-slice restriction.
    let t6 = Instant::now();
    let coords: Vec<(u64, u32, u32)> = settled.iter().map(|e| (e.t, e.s, e.r)).collect();
    let support = sets::support_set(&coords);
    let mid = (img.t_min + img.t_max) / 2;
    let window: BTreeSet<(u64, u32, u32)> = support
        .iter()
        .copied()
        .filter(|(t, _, _)| *t >= img.t_min && *t < mid)
        .collect();
    let inter = sets::set_inter(&support, &window);
    let diff = sets::set_diff(&support, &window);
    let fan = sets::fan_in_senders(&coords, e0.r, img.t_min, img.t_max);
    let co = sets::co_receivers(&coords, e0.s, img.t_min, img.t_max);
    let select_s = t6.elapsed();
    // Tensor algebra (rows 7-8).
    let t7 = Instant::now();
    let neg = tensor.try_neg().map_err(|e| e.to_string())?;
    let zero = tensor.try_add(&neg).map_err(|e| e.to_string())?;
    let alg_s = t7.elapsed();
    println!(
        concat!(
            "records={} nnz={} core_bytes={}\n",
            "read_s={:.3} decode_s={:.3} direct_csf_s={:.3} tensor_state_s={:.3}\n",
            "csf_point_reads={} sample_sum={} read_csf_s={:.3}\n",
            "late_n={} live_after_late={} sealed_nnz_after={} changed={} delta_s={:.3}\n",
            "window_inter={} window_diff={} fan_in_senders={} co_receivers={} select_s={:.3}\n",
            "algebra_zero={} alg_s={:.3}"
        ),
        img.records.len(), csf.nnz(), bytes.len(),
        read_s.as_secs_f64(), decode_s.as_secs_f64(),
        csf_s.as_secs_f64(), state_s.as_secs_f64(),
        reads, acc, read_csf_s.as_secs_f64(),
        late_n, live, eng.sealed().nnz(), rep.changed_coordinates,
        delta_s.as_secs_f64(),
        inter.len(), diff.len(), fan.len(), co.len(), select_s.as_secs_f64(),
        zero.is_zero(), alg_s.as_secs_f64(),
    );
    Ok(())
}
