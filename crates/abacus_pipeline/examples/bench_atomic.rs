//! Phase-14 benchmark matrix: 10K / 100K / 500K / 1M events.
//! Per size: total events, pair-history length, matching-event count,
//! window selectivity, median / p95 / p99 latency, heap allocation bytes.
//! Distributions: sparse (pair rare), typical (uniform), hub (pair dominant).
//! Deterministic fixtures (fixed xorshift seeds); identical inputs per rep.
//! Method: release build, 21 reps, 1 warmup, sorted-index quantiles.
//! Allocation: counting global allocator (bytes allocated during the rep).

use abacus_pipeline::atomic::row::{fiber_window, pair_seek, PairSeek};
use abacus_pipeline::retrieval::{EventLocator, TemporalRef};
use abacus_tensor::atomic::kernels::csf::CsfKernel;
use abacus_tensor::atomic::kernels::traits::OperatorKernel;
use abacus_tensor::atomic::reference::Predicate;
use abacus_tensor::atomic::types::Support;
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::tensor::Tensor;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct CountingAlloc;
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOC_BYTES.fetch_add(l.size(), Ordering::Relaxed);
        }
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
    }
}
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

fn pct(mut v: Vec<u128>, p: usize) -> u128 {
    v.sort_unstable();
    v[((p * v.len()) / 100).min(v.len() - 1)]
}
fn med(v: &[u128]) -> u128 {
    let mut v = v.to_vec();
    v.sort_unstable();
    v[v.len() / 2]
}

#[derive(Clone, Copy)]
enum Distr { Sparse, Typical, Hub }
impl Distr {
    fn name(self) -> &'static str {
        match self { Distr::Sparse => "sparse", Distr::Typical => "typical", Distr::Hub => "hub" }
    }
}

fn build_coo(n: usize, d: Distr) -> Vec<TxEntry> {
    let mut rng = Rng(match d { Distr::Sparse => 0x11, Distr::Typical => 0x22, Distr::Hub => 0x33 });
    let mut raw = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = (rng.below(2001) as i64) - 1000;
        if v == 0 { v = 1; }
        // focal pair is (7, 9). sparse: 0.1% of events; typical: uniform/500 keys (~0.0004% exact pair);
        // hub: sender 7 dominates (50%).
        let (s, r) = match d {
            Distr::Sparse => if rng.below(1000) == 0 { (7, 9) } else { (rng.below(500) as u32, rng.below(500) as u32) },
            Distr::Typical => (rng.below(500) as u32, rng.below(500) as u32),
            Distr::Hub => if rng.below(2) == 0 { (7, rng.below(50) as u32) } else { (rng.below(500) as u32, rng.below(500) as u32) },
        };
        raw.push(TxEntry::new(rng.below(100_000), s, r, v));
    }
    Tensor::try_from_entries(raw).expect("settle").encode_coo()
}

fn build_fwd_row(n: usize, d: Distr) -> Vec<TemporalRef> {
    // forward row of key 7: peers per distribution
    let mut rng = Rng(match d { Distr::Sparse => 0xAA, Distr::Typical => 0xBB, Distr::Hub => 0xCC });
    let mut row: Vec<TemporalRef> = Vec::with_capacity(n);
    for i in 0..n as u32 {
        let peer = match d {
            Distr::Sparse => if rng.below(100) == 0 { 9 } else { rng.below(500) as u32 },
            Distr::Typical => rng.below(500) as u32,
            Distr::Hub => rng.below(50) as u32,
        };
        row.push(TemporalRef { t: rng.below(100_000), locator: EventLocator(i), peer });
    }
    row.sort_by_key(|e| (e.t, e.locator.0));
    row
}

fn measure(mut f: impl FnMut()) -> (u128, usize) {
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(1, Ordering::Relaxed);
    let t0 = Instant::now();
    f();
    let ns = t0.elapsed().as_nanos();
    COUNTING.store(0, Ordering::Relaxed);
    (ns, ALLOC_BYTES.load(Ordering::Relaxed))
}

fn main() {
    const REPS: usize = 21;
    for &n in &[10_000usize, 100_000, 500_000, 1_000_000] {
        for d in [Distr::Sparse, Distr::Typical, Distr::Hub] {
            // --- e2e: CSF window restrict ---
            let coo = build_coo(n, d);
            let settled = coo.len();
            let csf = abacus_tensor::csf::compress::compress(&coo);
            let kernel = CsfKernel::new(&csf);
            let mut ev: Vec<abacus_tensor::atomic::types::Entry> = Vec::with_capacity(coo.len());
            for e in &coo {
                ev.push(abacus_tensor::atomic::types::Entry::new(e.t, e.s, e.r, e.value, abacus_tensor::atomic::types::LocSet::empty()));
            }
            ev.sort();
            let support = Support::from_canonical(ev);
            let p = Predicate::window(10_000, 90_000);
            let _ = kernel.restrict(&support, &p);
            let mut lat: Vec<u128> = Vec::with_capacity(REPS);
            let (mut scanned, mut seeks, mut out_n, mut bytes) = (0u64, 0u64, 0usize, 0usize);
            for _ in 0..REPS {
                let (ns, b) = measure(|| { let _ = kernel.restrict(&support, &p); });
                lat.push(ns);
                bytes = b;
            }
            let (got, st) = kernel.restrict(&support, &p);
            scanned = st.scanned;
            seeks = st.seeks;
            out_n = got.entries.len();
            let sel = if settled == 0 { 0.0 } else { out_n as f64 / settled as f64 };
            println!("e2e distr={} events={} settled={} win_out={} sel={:.4} scanned={} seeks={} alloc_bytes={} median_ns={} p95_ns={} p99_ns={}",
                d.name(), n, settled, out_n, sel, scanned, seeks, bytes, med(&lat), pct(lat.clone(), 95), pct(lat, 99));
            // --- pair seek over forward row of key 7 (fixed pair 7->9) ---
            let row = build_fwd_row(n, d);
            let hist = row.len();
            let matches = row.iter().filter(|e| e.peer == 9).count();
            let mut st2: abacus_tensor::atomic::kernels::traits::KernelStats = Default::default();
            let _ = pair_seek(&row, 9, PairSeek::Earliest, &mut st2);
            let mut lat2: Vec<u128> = Vec::with_capacity(REPS);
            let mut bytes2 = 0usize;
            for _ in 0..REPS {
                let (ns, b) = measure(|| { let _ = pair_seek(&row, 9, PairSeek::Earliest, &mut st2); });
                lat2.push(ns);
                bytes2 = b;
            }
            let mut stw = Default::default();
            let w = fiber_window(&row, 10_000, 90_000, &mut stw);
            println!("pair distr={} row_hist={} pair_matches={} match_frac={:.5} median_ns={} p95_ns={} p99_ns={} alloc_bytes={} | win_out={} seeks={}",
                d.name(), hist, matches, if hist == 0 { 0.0 } else { matches as f64 / hist as f64 }, med(&lat2), pct(lat2.clone(), 95), pct(lat2, 99), bytes2, w.len(), stw.seeks);
        }
    }
}
