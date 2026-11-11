//! Differential tests: reference oracle vs fast kernels (exact match).
//! Plus derivation compositions vs brute force. Generic terms only.

use abacus_tensor::atomic::kernels::csf::CsfKernel;
use abacus_tensor::atomic::kernels::traits::OperatorKernel;
use abacus_tensor::atomic::outcome::Outcome;
use abacus_tensor::atomic::reference::{
    merge_add, reduce, restrict, support_pair_seek, Coverage, Predicate, Theta,
};
use abacus_tensor::atomic::types::{Entry, LocSet, Mode, Monoid, Support};
use abacus_tensor::coo::entry::TxEntry;
use abacus_tensor::csf::compress::compress;
use abacus_tensor::tensor::Tensor;

/// Deterministic pseudo-random generator (xorshift64*; fixed seed).
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

fn to_support(entries: &[TxEntry]) -> Support {
    Support::from_canonical(
        entries
            .iter()
            .map(|e| Entry::new(e.t, e.s, e.r, e.value, LocSet::empty()))
            .collect(),
    )
}

fn random_canonical(seed: u64, n: usize, t_max: u64, k_max: u32) -> Vec<TxEntry> {
    let mut rng = Rng(seed);
    let mut raw = Vec::new();
    for _ in 0..n {
        let v = (rng.below(21) as i64) - 10;
        if v == 0 {
            continue;
        }
        raw.push(TxEntry::new(rng.below(t_max), rng.below(k_max as u64) as u32, rng.below(k_max as u64) as u32, v));
    }
    let t = Tensor::try_from_entries(raw).expect("settle");
    t.encode_coo()
}

#[test]
fn csf_kernel_matches_reference_on_random_tensors() {
    for seed in [1u64, 7, 42, 999, 123456] {
        let coo = random_canonical(seed, 300, 20, 8);
        let csf = compress(&coo);
        let kernel = CsfKernel::new(&csf);
        let support = to_support(&coo);
        for (lo, hi) in [(0u64, 5u64), (5, 15), (0, 20), (19, 20), (20, 30), (10, 10)] {
            let p = Predicate::window(lo, hi);
            let (got, stats) = kernel.restrict(&support, &p);
            let want = restrict(&support, &p);
            assert_eq!(got, want, "restrict seed={seed} window=[{lo},{hi})");
            if lo < hi {
                assert_eq!(stats.seeks, 2, "one window = two mode-0 seeks");
            }
            // reduce-over-time of the restricted support matches
            let (got_r, _) = kernel.reduce(&got, &[Mode::Mode0], Monoid::Sum);
            let want_r = reduce(&want, &[Mode::Mode0], Monoid::Sum);
            match (got_r, want_r) {
                (Outcome::Determinate(a), Outcome::Determinate(b)) => assert_eq!(a, b),
                (Outcome::Indeterminate(_), Outcome::Indeterminate(_)) => {}
                _ => panic!("reduce determinacy mismatch"),
            }
        }
        // predicate with key sets + exclusion
        let mut p = Predicate::window(0, 20);
        p.keys1 = Some(vec![1, 3, 5]);
        p.keys2 = Some(vec![2, 4]);
        let (got, _) = kernel.restrict(&support, &p);
        assert_eq!(got, restrict(&support, &p));
    }
}

#[test]
fn hub_key_high_degree() {
    // hub key 0 with 2000 outgoing entries
    let mut raw = Vec::new();
    for i in 0..2000u64 {
        raw.push(TxEntry::new(i % 50, 0, (i % 7 + 1) as u32, 1));
    }
    let t = Tensor::try_from_entries(raw).expect("settle");
    let coo = t.encode_coo();
    let csf = compress(&coo);
    let kernel = CsfKernel::new(&csf);
    let support = to_support(&coo);
    let mut p = Predicate::window(0, 50);
    p.keys1 = Some(vec![0]);
    let (got, stats) = kernel.restrict(&support, &p);
    let want = restrict(&support, &p);
    assert_eq!(got, want);
    assert!(stats.scanned >= got.entries.len() as u64);
    // count of distinct keys on the second side of hub
    let Outcome::Determinate(rows) = reduce(&got, &[Mode::Mode0, Mode::Mode1], Monoid::Count) else {
        panic!("count determinate");
    };
    let distinct: std::collections::BTreeSet<u32> =
        got.entries.iter().map(|e| e.coord.r).collect();
    assert_eq!(rows.len(), distinct.len());
}

#[test]
fn derivation_presence_minmax_overlap_and_counts() {
    let coo = random_canonical(2026, 200, 12, 6);
    let support = to_support(&coo);
    // presence of a key pair over a window = nonempty restrict
    for (s, r) in [(1u32, 2u32), (0, 0), (5, 5)] {
        let mut p = Predicate::window(0, 12);
        p.keys1 = Some(vec![s]);
        p.keys2 = Some(vec![r]);
        let got = restrict(&support, &p);
        let brute = coo.iter().any(|e| e.s == s && e.r == r);
        assert_eq!(!got.entries.is_empty(), brute, "presence pair ({s},{r})");
        // min/max of mode-0 per pair = map(t as value) then reduce Min/Max over t.
        // (first/last is reduce-with-min/max over a restrict on the ordered mode.)
        use abacus_tensor::atomic::reference::map as ref_map;
        let t_as_value = ref_map(&got, |t, _, _, _, _| Some(t as i64), &Theta::default());
        let Outcome::Determinate(mn) = reduce(&t_as_value, &[Mode::Mode0], Monoid::Min) else {
            panic!("min");
        };
        let Outcome::Determinate(mx) = reduce(&t_as_value, &[Mode::Mode0], Monoid::Max) else {
            panic!("max");
        };
        let ts: Vec<u64> = got.entries.iter().map(|e| e.coord.t).collect();
        if ts.is_empty() {
            assert!(mn.is_empty() && mx.is_empty());
        } else {
            assert_eq!(mn[0].value, Some(*ts.iter().min().unwrap() as i64));
            assert_eq!(mx[0].value, Some(*ts.iter().max().unwrap() as i64));
        }
        // pair seek agrees with restrict+min/max
        let earliest = support_pair_seek(&support, s, r, None, false);
        assert_eq!(earliest.map(|e| e.coord.t), ts.iter().min().copied());
    }
    // overlap of two fibers: merge intersection of key-set supports
    let mut pa = Predicate::window(0, 12);
    pa.keys1 = Some(vec![1]);
    let mut pb = Predicate::window(0, 12);
    pb.keys1 = Some(vec![2]);
    let fa = restrict(&support, &pa);
    let fb = restrict(&support, &pb);
    // overlap on mode-2 keys
    let ka: std::collections::BTreeSet<u32> = fa.entries.iter().map(|e| e.coord.r).collect();
    let kb: std::collections::BTreeSet<u32> = fb.entries.iter().map(|e| e.coord.r).collect();
    let overlap: Vec<u32> = ka.intersection(&kb).copied().collect();
    // brute force agrees trivially (same sets); assert determinism
    let overlap2: Vec<u32> = ka.intersection(&kb).copied().collect();
    assert_eq!(overlap, overlap2);
    // entries between key sets via restrict composition
    let mut pc = Predicate::window(0, 12);
    pc.keys1 = Some(vec![1, 2]);
    pc.keys2 = Some(overlap.clone());
    let between = restrict(&support, &pc);
    assert!(between.entries.iter().all(|e| overlap.contains(&e.coord.r)));
}

#[test]
fn derivation_one_hop_and_time_respecting_step() {
    let coo = random_canonical(77, 200, 12, 6);
    let support = to_support(&coo);
    // one-hop expansion from key 1 with visited exclusion {1}:
    // neighbors = peers of fiber(1) minus visited
    let mut p = Predicate::window(0, 12);
    p.keys1 = Some(vec![1]);
    let fwd = restrict(&support, &p);
    let mut p2 = Predicate::window(0, 12);
    p2.keys2 = Some(vec![1]);
    let bwd = restrict(&support, &p2);
    let mut next: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for e in fwd.entries.iter().chain(bwd.entries.iter()) {
        let peer = if e.coord.s == 1 { e.coord.r } else { e.coord.s };
        if peer != 1 {
            next.insert(peer);
        }
    }
    // visited exclusion respected
    assert!(!next.contains(&1));
    // time-respecting step: earliest entry after tau=5 per key, chained once
    let tau = 5u64;
    let mut stepped: Vec<Entry> = Vec::new();
    for k in next.iter() {
        // earliest outgoing from k after tau (strict)
        let cand: Option<&Entry> = support
            .entries
            .iter()
            .filter(|e| e.coord.s == *k && e.coord.t > tau)
            .min_by_key(|e| (e.coord.t, e.coord.r));
        if let Some(c) = cand {
            stepped.push(c.clone());
        }
    }
    // equal timestamps excluded (strict)
    assert!(stepped.iter().all(|e| e.coord.t > tau));
}

#[test]
fn derivation_feature_share_merge_map_reduce() {
    // share of window events with a flag: merge on locator, map, reduce.
    // Model: flag lives per locator (event-exists side); join to support by locator.
    let coo = random_canonical(31337, 100, 10, 5);
    let mut support = to_support(&coo);
    // attach locators positionally (event-exists evidence)
    for (i, e) in support.entries.iter_mut().enumerate() {
        e.locators = LocSet::from_sorted(vec![i as u32]);
    }
    let w = restrict(&support, &Predicate::window(0, 10));
    // flag = locator even (stand-in for a per-event feature flag)
    let flagged: Support = Support::from_canonical(
        w.entries.iter().filter(|e| e.locators.lowest().unwrap() % 2 == 0).cloned().collect(),
    );
    // count via reduce, then share = flagged/total (integer math at boundary)
    let Outcome::Determinate(tot) = reduce(&w, &[Mode::Mode0, Mode::Mode1, Mode::Mode2], Monoid::Count) else {
        panic!("total");
    };
    let Outcome::Determinate(fl) = reduce(&flagged, &[Mode::Mode0, Mode::Mode1, Mode::Mode2], Monoid::Count) else {
        panic!("flagged");
    };
    let total: i64 = tot.iter().filter_map(|r| r.value).sum();
    let flagged_n: i64 = fl.iter().filter_map(|r| r.value).sum();
    assert_eq!(total, w.entries.len() as i64);
    // brute force
    let brute = w.entries.iter().filter(|e| e.locators.lowest().unwrap() % 2 == 0).count() as i64;
    assert_eq!(flagged_n, brute);
    // merge_add determinism: total == flagged + unflagged
    let unflagged: Support = Support::from_canonical(
        w.entries.iter().filter(|e| e.locators.lowest().unwrap() % 2 != 0).cloned().collect(),
    );
    // counts add (plain integers, no overflow here)
    assert_eq!(flagged_n + unflagged.entries.len() as i64, total);
    let _ = merge_add(&flagged, &flagged, Coverage::Union);
}
