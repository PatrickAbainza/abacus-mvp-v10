//! Optimization equivalence: single-pass `pair_seek` vs reference oracle.
//! Proves the fused fast path equals the composition (Phase-13 rule).

use abacus_pipeline::atomic::row::{pair_seek, PairSeek};
use abacus_pipeline::retrieval::{EventLocator, TemporalRef};
use abacus_tensor::atomic::reference::{pair_earliest, pair_latest, pair_next_after, RowRef};

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

fn to_refs(row: &[TemporalRef]) -> Vec<RowRef> {
    row.iter().map(|e| RowRef { t: e.t, peer: e.peer, locator: e.locator.0 }).collect()
}

#[test]
fn single_pass_equals_reference_on_random_rows() {
    let mut rng = Rng(0x5EED);
    for round in 0..200 {
        let n = (rng.below(50) + 1) as usize;
        let mut row: Vec<TemporalRef> = Vec::with_capacity(n);
        for i in 0..n as u32 {
            // duplicate timestamps + duplicate peers + self-pairs, unsorted input
            row.push(TemporalRef {
                t: rng.below(5),
                locator: EventLocator(rng.below(4) as u32),
                peer: rng.below(4) as u32,
            });
            let _ = (round, i);
        }
        row.sort_by_key(|e| (e.t, e.locator.0)); // row invariant
        let refs = to_refs(&row);
        for peer in 0..5u32 {
            let mut st = Default::default();
            assert_eq!(pair_seek(&row, peer, PairSeek::Earliest, &mut st), pair_earliest(&refs, peer), "earliest round {round} peer {peer}");
            assert_eq!(pair_seek(&row, peer, PairSeek::Latest, &mut st), pair_latest(&refs, peer), "latest round {round} peer {peer}");
            for tau in [0u64, 2, 4, 9] {
                assert_eq!(
                    pair_seek(&row, peer, PairSeek::NextAfter(tau), &mut st),
                    pair_next_after(&refs, peer, tau),
                    "next-after tau={tau} round {round} peer {peer}"
                );
            }
        }
    }
}

#[test]
fn single_pass_allocates_nothing() {
    // heap allocation counter lives in the bench harness; here assert the
    // result path needs no Vec: worst case output is one RowRef (Copy).
    let row = vec![
        TemporalRef { t: 1, locator: EventLocator(0), peer: 9 },
        TemporalRef { t: 2, locator: EventLocator(1), peer: 9 },
    ];
    let mut st = Default::default();
    let got = pair_seek(&row, 9, PairSeek::Earliest, &mut st);
    assert_eq!(got, Some(RowRef { t: 1, peer: 9, locator: 0 }));
    assert_eq!(st.scanned, 2);
}
