//! Read-only inspector: prints real decoded samples from a materialised
//! SAMLD profile, using the project's own §R5f decoders (no hand-rolled
//! parsing, no writes, no CSV read).
//!
//! ```sh
//! cargo run -p abacus_tensor --example inspect_profile --release -- <profile_dir>
//! ```
//!
//! Prints, per artifact: the on-disk magic/version header, then the first `N`
//! decoded records/entries with both the canonical `amount_micro` and its
//! human-readable major-unit rendering (P2, `scale_exp = 6`).

use abacus_tensor::csf::deserialize_csf_v2;
use abacus_tensor::samld::{deserialize_core_v2, deserialize_eval_v3};
use abacus_tensor::tensor::value::format_minor;

const N: usize = 8;

/// `"YYYY-MM-DD HH:MM:SS UTC"` from epoch seconds (civil-from-days, inverse of
/// `d_hinnant`). Display only.
fn civil(ts: u64) -> String {
    let days = (ts / 86_400) as i64;
    let secs = ts % 86_400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        y, m, d, secs / 3_600, (secs % 3_600) / 60, secs % 60
    )
}

fn head(bytes: &[u8]) -> String {
    let magic = String::from_utf8_lossy(&bytes[..8]).to_string();
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    format!("magic={magic} version={version} file_bytes={}", bytes.len())
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: inspect_profile <profile_dir>");
        std::process::exit(2);
    });
    let rd = |n: &str| {
        std::fs::read(format!("{dir}/{n}")).unwrap_or_else(|e| panic!("read {dir}/{n}: {e}"))
    };

    println!("\n============ samld_core.bin (SAMCORE02) ============");
    let core = rd("samld_core.bin");
    println!("header: {}", head(&core));
    let ci = deserialize_core_v2(&core).expect("decode core");
    println!(
        "records={}  t_min={} ({})  t_max={} ({})\n\
         mean_major={:.6}  std_major={:.6}\n\
         senders={}  receivers={}",
        ci.records.len(),
        ci.t_min,
        civil(ci.t_min),
        ci.t_max,
        civil(ci.t_max),
        ci.mean_major,
        ci.std_major,
        ci.senders.len(),
        ci.receivers.len()
    );
    println!(
        "\n{:<4} {:<20} {:>7} {:>7} {:>16} {:>18} {:>9} {:>6} {:>4}",
        "row", "t (UTC)", "s", "r", "amount_micro", "amount (major)", "log1p", "z", "rnd"
    );
    for (i, r) in ci.records.iter().take(N).enumerate() {
        println!(
            "{:<4} {:<20} {:>7} {:>7} {:>16} {:>18} {:>9.4} {:>6.2} {:>4}",
            i,
            civil(r.t),
            r.s,
            r.r,
            r.amount_micro,
            format_minor(r.amount_micro),
            r.log1p_amt,
            r.z_amt,
            r.roundedness
        );
    }
    println!(
        "\nsample sender[0]={:?}  sender[1]={:?}",
        ci.senders[0], ci.senders[1]
    );
    println!(
        "sample receiver[0]={:?}  receiver[1]={:?}",
        ci.receivers[0], ci.receivers[1]
    );
    let sum: i64 = ci.records.iter().map(|r| r.amount_micro).sum();
    let pos = ci.records.iter().filter(|r| r.amount_micro > 0).count();
    let neg = ci.records.iter().filter(|r| r.amount_micro < 0).count();
    let zero = ci.records.len() - pos - neg;
    println!(
        "totals: sum(amount_micro)={}  positive={} negative={} zero={}",
        sum, pos, neg, zero
    );

    println!("\n============ samld_core.csf (SAMCSF02) ============");
    let csf_b = rd("samld_core.csf");
    println!("header: {}", head(&csf_b));
    let csf = deserialize_csf_v2(&csf_b).expect("decode csf");
    println!(
        "n0(t fibers)={}  n1((t,s) fibers)={}  nnz={}  scale_exp={}  raw_rows={}\n\
         t_min={} ({})  t_max={} ({})",
        csf.csf.mode0_coords.len(),
        csf.csf.mode1_coords.len(),
        csf.csf.nnz(),
        csf.header.scale_exp,
        csf.raw_rows,
        csf.t_min,
        civil(csf.t_min),
        csf.t_max,
        civil(csf.t_max)
    );
    println!("\nnonzero entries (t, s, r) -> value  [first {N}]:");
    let ent = abacus_tensor::csf::decode_csf(&csf.csf);
    for (i, e) in ent.iter().take(N).enumerate() {
        println!(
            "  [{i}] t={} ({}) s={} r={} value={:>15}  ({})",
            e.t,
            civil(e.t),
            e.s,
            e.r,
            e.value,
            format_minor(e.value)
        );
    }
    let csum: i64 = ent.iter().map(|e| e.value).sum();
    println!(
        "totals: nnz={}  sum(value)={}  == core sum(amount_micro): {}",
        ent.len(),
        csum,
        csum == sum
    );

    println!("\n============ samld_eval.sidecar (SAMEVL03) ============");
    let ev_b = rd("samld_eval.sidecar");
    println!("header: {}", head(&ev_b));
    let ev = deserialize_eval_v3(&ev_b).expect("decode sidecar");
    let lg = ev.iter().filter(|e| e.laundering != 0).count();
    println!(
        "records={}  laundering_flagged={} ({:.2}%)",
        ev.len(),
        lg,
        100.0 * lg as f64 / ev.len() as f64
    );
    println!(
        "\n{:<6} {:>16} {:>18} {:>4}  {:<14} {:<20} {:<8} {:<20} {:<22} {:<24} {:<10}",
        "tx_id", "amount_micro", "amount (major)", "ldg", "amount_raw", "laundering_type",
        "pay_ccy", "recv_ccy", "sender_bank_loc", "receiver_bank_loc", "pay_type"
    );
    for e in ev.iter().take(N) {
        println!(
            "{:<6} {:>16} {:>18} {:>4}  {:<14} {:<20} {:<8} {:<20} {:<22} {:<24} {:<10}",
            e.tx_id,
            e.amount_micro,
            format_minor(e.amount_micro),
            e.laundering,
            e.amount_raw,
            e.laundering_type,
            e.payment_currency,
            e.received_currency,
            e.sender_bank_location,
            e.receiver_bank_location,
            e.payment_type
        );
    }
    for (name, f) in [
        ("laundering_type", 0usize),
        ("payment_currency", 1),
        ("received_currency", 2),
        ("sender_bank_location", 3),
        ("receiver_bank_location", 4),
        ("payment_type", 5),
    ] {
        let col: std::collections::BTreeSet<&str> = ev
            .iter()
            .map(|e| match f {
                0 => e.laundering_type.as_str(),
                1 => e.payment_currency.as_str(),
                2 => e.received_currency.as_str(),
                3 => e.sender_bank_location.as_str(),
                4 => e.receiver_bank_location.as_str(),
                _ => e.payment_type.as_str(),
            })
            .collect();
        println!("dict {name}: {} distinct", col.len());
    }
    let mut ids: Vec<u64> = ev.iter().map(|e| e.tx_id).collect();
    ids.sort_unstable();
    ids.dedup();
    println!(
        "tx_id: distinct={} min={:?} max={:?} == {{0..{}}}: {}",
        ids.len(),
        ids.first(),
        ids.last(),
        ev.len(),
        ids == (0..ev.len() as u64).collect::<Vec<u64>>()
    );
}
