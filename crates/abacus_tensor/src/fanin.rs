//! Structural Fan-In over the canonical CSF (`docs/spec/fan_in.md`; plan §3.4).
//!
//! ```text
//! FocalTransaction (t_f, s_f, r_f)
//!   ↓ closed window W_f = [t_f − Δ, t_f]          (fan_in.md §5)
//!   ↓ CSF mode-0 restriction t ∈ [t_f−Δ, t_f+1)  (row 16, P1 half-open)
//!   ↓ r = r_f                                    (§6)
//!   ↓ ONE sparse traversal, NO intermediate tensor (§13, INV-11)
//!   ↓ fused exact reductions:  T_i, S_i, t_min, t_max
//! (N_i = |S_i|, RFI = N_i/T_i, FI_topology = 1[N_i>1])   (§7, §45)
//! FanInResult
//! ```
//!
//! # Authorization boundary — what this module does NOT do
//!
//! `FanInState` holds **only authorized exact state**:
//! `T_i: u64`, `S_i: BTreeSet<u32>`, `t_min`/`t_max: Option<u64>`.
//!
//! The canonical 7-field product (`fan_in_addendum.md` §24) also names
//! `M_log`, `C_Z` and `C_R`. They are **deliberately absent**, not disabled:
//! `samld_binary_addendum.md` §S3 / gap `U-SAMLD-1` does not authorize their
//! tensor use ("a future tensor use requires a Spec extension first"), so they
//! MUST NOT appear in the reduction state or the kernel. Consequently
//!
//! * [`FanInState::observe`] takes `(s, t)` — **never an amount**. The
//!   behavioral transforms are therefore not merely gated, they are
//!   *unreachable* from this code path;
//! * there is **no `f64` anywhere** in the reduction state, so `⊕` is
//!   associative by construction and `FanIn_P(f) = FanIn_1(f)` holds bitwise
//!   for every `P` (`fan_in_addendum.md` §36, INV-12) — nothing to disclose;
//! * `RFI_f` is carried as the exact pair `(N_i, T_i)`; numerical rendering
//!   happens only at the output boundary.
//!
//! # Invariants honoured
//!
//! * **Canonical tensor.** Reads `CsfTensor` fiber arrays directly; no decode,
//!   no COO materialization, no CSC/CSR, no adjacency, no graph (INV-01…05).
//! * **Coordinate-level.** `I_f ⊆ supp(T)`, so `T_i` counts **surviving
//!   canonical coordinates**. A coordinate whose source rows cancelled is
//!   absent (G8 / `R_6`) and contributes nothing — it is not a zero-valued
//!   observation.
//! * **Exact sender uniqueness.** `S_i` is a `BTreeSet<u32>`; `N_i` is derived
//!   as `senders.len()` so it can never drift from the set (INV-09, §16).
//! * **`⊥` extrema.** `t_min`/`t_max` are `Option<u64>`, never a sentinel
//!   timestamp (`fan_in_addendum.md` §26).
//! * **Traversal accounting.** `V_f` counts CSF entries physically visited,
//!   `K_f` counts matching entries; `K_f ≤ V_f` is asserted (fan_in.md §19).
//! * **Fail-closed.** Every fallible step returns `Result<_, TensorError>`
//!   (`R_5`); no `unwrap`, no panic, no wrapping arithmetic (P2).
//!
//! Parallel execution (mode-0 position-space partitioning) lives in
//! `abacus_execution::fanin_parallel`, per `mvp7_invariant.md` §47.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::csf::compress::CsfTensor;
use crate::TensorError;

/// The declared experimental window default, `Δ = 7` days (`fan_in.md` §5).
///
/// This is an **Abacus experimental parameter**, not a universal or a source-paper
/// requirement; §5 states it MUST NOT be represented as one.
pub const DEFAULT_DELTA_SECS: u64 = 7 * 86_400;

/// The focal transaction every Fan-In computation is anchored to
/// (`fan_in.md` §4, INV-06).
///
/// Carries the *coordinates only*. `a_f` is deliberately **not** a field: the
/// authorized structural kernel never transforms an amount, so carrying one
/// would imply an authorized use that `U-SAMLD-1` withholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FocalTransaction {
    /// Focal time `t_f` (seconds since epoch) — exact coordinate.
    pub epoch: u64,
    /// Focal sender `s_f` — dictionary code.
    pub sender: u32,
    /// Focal receiver `r_f` — dictionary code; selects the incoming stream (§4).
    pub receiver: u32,
}

impl FocalTransaction {
    /// Construct a focal transaction (`a_{14.1}`).
    #[inline]
    #[must_use]
    pub const fn new(epoch: u64, sender: u32, receiver: u32) -> Self {
        Self {
            epoch,
            sender,
            receiver,
        }
    }
}

/// The temporal context `W_f(Δ) = [t_f − Δ, t_f]` (`fan_in.md` §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FanInWindow {
    /// Window half-width `Δ` in seconds, inclusive on both ends.
    pub delta: u64,
}

impl FanInWindow {
    /// A window with an explicit `Δ`.
    #[inline]
    #[must_use]
    pub const fn with_delta(delta: u64) -> Self {
        Self { delta }
    }

    /// The declared experimental default, `Δ = 7` days.
    #[inline]
    #[must_use]
    pub const fn default_experimental() -> Self {
        Self {
            delta: DEFAULT_DELTA_SECS,
        }
    }
}

/// The mergeable reduction state — **authorized exact fields only**
/// (`fan_in_addendum.md` §24, filtered by G2a).
///
/// `⊕ = (+, ∪, min, max)` over integers and exact sets, so every component is
/// associative and commutative: `merge` is therefore associative and
/// order-independent, which is what makes `FanIn_P = FanIn_1` (§36, INV-12).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FanInState {
    count:   u64,               // T_i   ⊕ +   (checked; never wrapping)
    senders: BTreeSet<u32>,     // S_i   ⊕ ∪   exact (INV-09)
    t_min:   Option<u64>,       // ⊥-safe      (§26)
    t_max:   Option<u64>,       // ⊥-safe
}

impl FanInState {
    /// The identity state `⊥`: empty set, zero count, no extrema (§26).
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            count: 0,
            senders: BTreeSet::new(),
            t_min: None,
            t_max: None,
        }
    }

    /// Fold one **matched** canonical coordinate `(s, r_f, t)` into the state.
    ///
    /// Takes `(s, t)` only — **no amount**. This is the structural observation
    /// path; nothing derived from `values[k]` is reachable here (see the module
    /// header).
    ///
    /// # Errors
    /// [`TensorError::Range`] (`E_rng`) if `T_i` would exceed `u64::MAX`
    /// (`P_2` checked arithmetic, `R_5` fail-closed). Unreachable for any
    /// realizable tensor, since `T_i ≤ |supp(T)| ≤ nnz`.
    pub fn observe(&mut self, s: u32, t: u64) -> Result<(), TensorError> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| TensorError::Range("fan-in T_i overflow (u64)".into()))?;
        self.senders.insert(s);              // exact set insertion `a_{16.2}`
        self.t_min = Some(match self.t_min {
            None => t,
            Some(m) => m.min(t),
        });
        self.t_max = Some(match self.t_max {
            None => t,
            Some(m) => m.max(t),
        });
        Ok(())
    }

    /// Merge another state with `⊕ = (+, ∪, min, max)` (`fan_in_addendum.md` §25).
    ///
    /// Associative and commutative: exact set union is associative (§16) and
    /// `min`/`max` over `Option<u64>` are order-independent, so the reduction
    /// tree may be binary, hierarchical, or flat.
    ///
    /// # Errors
    /// [`TensorError::Range`] on `T_i` overflow (`R_5`).
    pub fn merge(&mut self, other: &Self) -> Result<(), TensorError> {
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or_else(|| TensorError::Range("fan-in T_i overflow (u64)".into()))?;
        self.senders.extend(other.senders.iter().copied());   // exact `∪`
        self.t_min = min_opt(self.t_min, other.t_min);
        self.t_max = max_opt(self.t_max, other.t_max);
        Ok(())
    }

    /// Current `T_i` — total matched canonical coordinates.
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Current `S_i` — the exact unique incoming sender set.
    #[inline]
    #[must_use]
    pub const fn senders(&self) -> &BTreeSet<u32> {
        &self.senders
    }

    /// `⊥`-safe minimum matching time.
    #[inline]
    #[must_use]
    pub const fn t_min(&self) -> Option<u64> {
        self.t_min
    }

    /// `⊥`-safe maximum matching time.
    #[inline]
    #[must_use]
    pub const fn t_max(&self) -> Option<u64> {
        self.t_max
    }
}

/// `⊥`-absorbing `min` (`fan_in_addendum.md` §26): `⊥ ∨ x = x`, so an empty
/// worker partition cannot corrupt the global temporal minimum.
#[inline]
#[must_use]
const fn min_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(if x <= y { x } else { y }),
    }
}

/// `⊥`-absorbing `max` (`fan_in_addendum.md` §26).
#[inline]
#[must_use]
const fn max_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(if x >= y { x } else { y }),
    }
}

/// Traversal accounting: `V_f` visited vs `K_f` matched (`fan_in.md` §19).
///
/// These MUST be distinguished. `K_f ≤ V_f` always; claiming `O(K_f)` traversal
/// is only valid if the concrete CSF implementation gives a receiver
/// lookup/order guarantee, which `t → s → r` does **not** (see [`fold`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FanInAccounting {
    /// `V_f` — CSF entries physically visited in the window.
    pub visited: u64,
    /// `K_f` — entries satisfying `t ∈ W_f, r = r_f`.
    pub matched: u64,
}

impl FanInAccounting {
    /// Merge two partitions' accounting by addition (`V`, `K` are `⊕ +`).
    pub fn merge(&mut self, other: &Self) -> Result<(), TensorError> {
        self.visited = self.visited.checked_add(other.visited).ok_or_else(|| {
            TensorError::Range("fan-in V_f overflow (u64)".into())
        })?;
        self.matched = self.matched.checked_add(other.matched).ok_or_else(|| {
            TensorError::Range("fan-in K_f overflow (u64)".into())
        })?;
        Ok(())
    }
}

/// The canonical Fan-In result (`fan_in.md` §26), authorized fields only.
///
/// `RFI_f` is **not** stored as a float. The exact pair `(N_i, T_i)` is carried
/// in `n_i`/`t_i`; comparisons use `rule_discovery::cmp_rational` and any
/// numerical rendering happens only at the output boundary (presentation-only,
/// `K_10`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FanInResult {
    /// The focal transaction `f`.
    pub focal: FocalTransaction,
    /// `T_i` — total incoming (matched canonical coordinates).
    pub t_i: u64,
    /// `S_i` — exact unique incoming sender set (witness).
    pub senders: BTreeSet<u32>,
    /// `N_i` — `|S_i|`, derived from `senders` so it cannot drift (INV-09).
    pub n_i: u64,
    /// `RFI_f` as the exact integer pair `(N_i, T_i)`; `None` iff `T_i == 0`,
    /// where `fan_in.md` §37 declares `RFI` undefined.
    pub rfi: Option<(u64, u64)>,
    /// `FI_topology` — `1[N_i > 1]`.
    pub fi_topology: bool,
    /// `⊥`-safe minimum matching time.
    pub t_min: Option<u64>,
    /// `⊥`-safe maximum matching time.
    pub t_max: Option<u64>,
    /// `V_f` — CSF entries visited.
    pub visited: u64,
    /// `K_f` — matching entries.
    pub matched: u64,
}

impl FanInResult {
    /// Finalize a reduction state into the canonical result.
    ///
    /// # Errors
    /// [`TensorError::Range`] if `V_f < K_f`, which would contradict `fan_in.md`
    /// §19. Defensive: `observe` only runs on matching entries, so this cannot
    /// occur; asserting it keeps the invariant machine-checked (`R_5`).
    pub fn finalize(
        focal: FocalTransaction,
        state: FanInState,
        accounting: FanInAccounting,
    ) -> Result<Self, TensorError> {
        if accounting.matched > accounting.visited {
            return Err(TensorError::Range(format!(
                "fan-in K_f ({}) exceeds V_f ({})",
                accounting.matched, accounting.visited
            )));
        }
        let n_i = state.senders.len() as u64;   // derived; never a separate counter
        Ok(Self {
            focal,
            t_i: state.count,
            senders: state.senders,
            n_i,
            rfi: (state.count > 0).then_some((n_i, state.count)),
            fi_topology: n_i > 1,
            t_min: state.t_min,
            t_max: state.t_max,
            visited: accounting.visited,
            matched: accounting.matched,
        })
    }
}

/// Half-open CSF mode-0 index bounds for the **closed** window
/// `W_f = [t_f − Δ, t_f]` (`fan_in.md` §5).
///
/// ⚠ **The window in the spec is closed on both ends** — tests #9 and #10
/// require a transaction *exactly at* `t_f − Δ` and *exactly at* `t_f` to be
/// included. The canonical `time_slice` is **half-open** `[start, end)` (P1),
/// so the upper bound is encoded as `t_f + 1`. Passing `t_f` would silently drop
/// the focal transaction itself.
///
/// When `t_f < Δ` the window precedes the epoch: the result is an **empty**
/// window (no error), because the set `I_f` is then legitimately empty (L4
/// totality).
///
/// # Errors
/// [`TensorError::Range`] if `t_f + 1` is not representable (`P_2`, `R_5`).
pub fn window_bounds(
    csf: &CsfTensor,
    focal: &FocalTransaction,
    window: &FanInWindow,
) -> Result<(usize, usize), TensorError> {
    let Some(t_lo) = focal.epoch.checked_sub(window.delta) else {
        return Ok((0, 0)); // window precedes the epoch: empty, not an error
    };
    let t_hi = focal
        .epoch
        .checked_add(1)
        .ok_or_else(|| TensorError::Range("fan-in window end t_f + 1 overflow".into()))?;
    // a_{12} partition_point over the ascending mode-0 coordinates (P1).
    let lo = csf.mode0_coords.partition_point(|&t| t < t_lo);
    let hi = csf.mode0_coords.partition_point(|&t| t < t_hi);
    Ok((lo, hi))
}

/// Fold CSF mode-0 fibers `[lo, hi)` into the reduction state for one focal.
///
/// This is `fan_in.md` §12's logical iteration and §37's reference algorithm,
/// written directly against the canonical fiber arrays:
///
/// ```text
/// for t in fibers[lo..hi]:              # t ∈ W_f
///     for (t,s) in t_fiber(t):          # sender fibers
///         for (s,r) in (t,s)_fiber:     # receiver entries
///             if r == r_f:              # receiver restriction
///                 observe(s, t)
/// ```
///
/// It is a *conceptual* specification, not a demand for these exact source
/// loops; the mathematical result is what is invariant. No intermediate tensor
/// is created — `I_f` is a logical stream, not a materialized tensor (§13,
/// INV-11).
///
/// **Complexity is `O(V_f)`**, not `O(K_f)` (§30). The receiver is tested by
/// comparison within each `(t,s)` fiber; `t → s → r` gives no receiver
/// lookup/order guarantee after filtering (§39), so boundary-only sender
/// counting is **not** used — `S_i` is an exact set regardless.
///
/// # Errors
/// [`TensorError::Range`] on `T_i`/`V_f`/`K_f` overflow (`R_5`).
pub fn fold(
    csf: &CsfTensor,
    lo: usize,
    hi: usize,
    focal: &FocalTransaction,
    state: &mut FanInState,
    accounting: &mut FanInAccounting,
) -> Result<(), TensorError> {
    for i in lo..hi {
        let t = csf.mode0_coords[i];
        let (js, je) = (
            csf.mode0_fptr[i] as usize,
            csf.mode0_fptr[i + 1] as usize,
        );
        for j in js..je {
            let s = csf.mode1_coords[j];
            let (ks, ke) = (
                csf.mode1_fptr[j] as usize,
                csf.mode1_fptr[j + 1] as usize,
            );
            for k in ks..ke {
                accounting.visited = accounting.visited.checked_add(1).ok_or_else(|| {
                    TensorError::Range("fan-in V_f overflow (u64)".into())
                })?;
                if csf.mode2_coords[k] == focal.receiver {
                    accounting.matched = accounting.matched.checked_add(1).ok_or_else(|| {
                        TensorError::Range("fan-in K_f overflow (u64)".into())
                    })?;
                    state.observe(s, t)?;
                }
            }
        }
    }
    Ok(())
}

/// The reference Fan-In kernel: one CSF traversal, all structural reductions.
///
/// `fan_in.md` §18's pipeline, §37's algorithm. Serial (`P = 1`); parallel
/// execution partitions the same mode-0 range and merges with the same `⊕`,
/// which is why `FanIn_P(f) = FanIn_1(f)` (§36).
///
/// # Errors
/// [`TensorError`] on window or counter overflow (`R_5` fail-closed).
pub fn fanin(
    csf: &CsfTensor,
    focal: &FocalTransaction,
    window: &FanInWindow,
) -> Result<FanInResult, TensorError> {
    let (lo, hi) = window_bounds(csf, focal, window)?;
    let mut state = FanInState::identity();
    let mut accounting = FanInAccounting::default();
    fold(csf, lo, hi, focal, &mut state, &mut accounting)?;
    FanInResult::finalize(*focal, state, accounting)
}

/// Bulk Fan-In over **all** receivers in one direct CSF fiber traversal.
///
/// Walks every canonical coordinate `(t, s, r)` with `t ∈ [t_lo, t_hi)` and
/// accumulates `S_i(r) = { s : ∃(t, s, r) ∈ supp(T), t ∈ [t_lo, t_hi) }` —
/// the exact distinct incoming sender set per receiver.
///
/// Single pass: `O(V_f)` where `V_f` = CSF entries physically visited in the
/// window. No intermediate tensor, no PairMatrix, no CSC/CSR construction —
/// reuses the same fiber-walk pattern as [`fold`].
///
/// # Errors
/// [`TensorError::Range`] if the receiver counter would overflow (`R_5`).
#[must_use]
pub fn bulk_fan_in(
    csf: &CsfTensor,
    t_lo: u64,
    t_hi: u64,
) -> Result<BTreeMap<u32, BTreeSet<u32>>, TensorError> {
    let mut fan_in: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();

    let lo = csf.mode0_coords.partition_point(|&t| t < t_lo);
    let hi = csf.mode0_coords.partition_point(|&t| t < t_hi);

    for i in lo..hi {
        let _t = csf.mode0_coords[i]; // t ∈ [t_lo, t_hi) by construction
        let (js, je) = (
            csf.mode0_fptr[i] as usize,
            csf.mode0_fptr[i + 1] as usize,
        );
        for j in js..je {
            let s = csf.mode1_coords[j];
            let (ks, ke) = (
                csf.mode1_fptr[j] as usize,
                csf.mode1_fptr[j + 1] as usize,
            );
            for k in ks..ke {
                let r = csf.mode2_coords[k];
                fan_in.entry(r).or_default().insert(s);
            }
        }
    }

    Ok(fan_in)
}

/// Convenience: receivers with Fan-In topology (distinct incoming sender
/// count `> 1`) in the 7-day experimental window ending at `now_secs`.
///
/// `FI_topology(r) = 1[N_i(r) > 1]` where `N_i(r) = |S_i(r)|`.
///
/// # Errors
/// [`TensorError::Range`] on counter overflow (`R_5`).
#[must_use]
pub fn detect_bulk_fan_in_topology(
    csf: &CsfTensor,
    now_secs: u64,
) -> Result<Vec<(u32, u64)>, TensorError> {
    let delta: u64 = DEFAULT_DELTA_SECS; // 7 days
    let t_lo = now_secs.saturating_sub(delta);
    let t_hi = now_secs.saturating_add(1); // half-open [t_lo, t_hi)

    let fan_in = bulk_fan_in(csf, t_lo, t_hi)?;
    let topology: Vec<(u32, u64)> = fan_in
        .into_iter()
        .filter(|(_, senders)| senders.len() > 1)
        .map(|(r, senders)| (r, senders.len() as u64))
        .collect();

    Ok(topology)
}