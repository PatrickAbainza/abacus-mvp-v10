//! Abacus-native TACO-style sparse execution layer for Fan-In.
//!
//! `your_2nd_task.md` §1, §5, §28. The TACO **conceptual model** is retained —
//! *expression ⊗ format ⊗ schedule → specialized sparse kernel* — implemented
//! natively over the canonical Abacus CSF. TACO is **not** a dependency.
//!
//! # Why this exists
//!
//! The TACO formulation spike (`docs/spec/g5_taco_boundary_decision.md` §5.5bis)
//! returned **NO**: TACO's expression language has no value-level comparison
//! (so `r == r_f` cannot be expressed), coordinate slicing cannot reduce tensor
//! arity, `Dense ← reduction` is rejected by its assembler precondition, and
//! sparse output triggers a codegen defect. Those are properties of TACO's
//! generality, not of the Fan-In workload. This module implements the subset
//! Abacus actually needs.
//!
//! # The four TACO concepts, concretely
//!
//! | Concept | Here |
//! |---|---|
//! | **Expression** | [`ReductionSet`] — the fused reduction operators |
//! | **Format** | [`CanonicalFormat`] — the canonical `t → s → r` hierarchy |
//! | **Schedule** | [`TraversalOrder`] + [`SparseExecutionPlan`] — window/receiver restriction and mode-0 position space |
//! | **Kernel** | [`fold`] — one specialized traversal, all reductions fused |
//!
//! No parser, IR, optimizer, scheduler, or codegen exists here, because none is
//! required (§19).
//!
//! # Canonical substrate
//!
//! The kernel reads [`CsfTensor`] fiber arrays **directly**:
//! `mode0_fptr`/`mode0_coords` → `mode1_fptr`/`mode1_coords` →
//! `mode2_coords`/`values`. There is **no** COO, CSR, CSC, edge list, adjacency
//! matrix, dense mask, second index, or temporary tensor — none is constructed
//! anywhere in this module (§4, §13, §14, INV-13, INV-14).
//!
//! # Fusion
//!
//! One visit to a qualifying coordinate updates **every** authorized reduction
//! (§12): count, sender insertion, `log1p` quantization, roundedness predicate,
//! and both extrema. There is exactly one traversal and no intermediate state.
//!
//! # Reduction state
//!
//! [`FusedFanInState`] = `(T_i, S_i, M_log, C_R, t_min, t_max)`:
//!
//! * `T_i: u64` — `⊕ +` checked
//! * `S_i: BTreeSet<u32>` — `⊕ ∪` exact (INV-09; never approximate, §17)
//! * `M_log: i128` — `⊕ +` **checked** (§8)
//! * `C_R: u64` — `⊕ +` checked (§9)
//! * `t_min`/`t_max: Option<u64>` — `⊥`-absorbing `min`/`max`
//!
//! **No `f64` is stored.** `f64` appears only *inside* [`quantize_log1p`], to
//! evaluate `log1p_amt` and round `2^64 · log1p_amt` into `i128` — the `K_10`
//! presentation class, per the §S5 definition of the feature.
//!
//! `C_Z` is **absent** and is not implemented anywhere: `G3` remains unresolved,
//! so no threshold θ exists and none may be invented (§22, §25).
//!
//! # Exactness
//!
//! `⊕ = (+, ∪, +, +, min, max)` over integers and exact sets is associative and
//! commutative, so [`FusedFanInState::merge`] is order-independent and
//! `FanIn_P(f) = FanIn_1(f)` holds **bitwise** for every partition (§7, §16).

use std::collections::BTreeSet;

use crate::csf::compress::CsfTensor;
use crate::fanin::{window_bounds, FanInAccounting, FocalTransaction, FanInWindow};
use crate::samld::diag;
use crate::TensorError;

/// The canonical storage hierarchy (`your_2nd_task.md` §4, §15).
///
/// A marker, not a runtime option: the canonical tensor has exactly one format
/// and this layer may not introduce another (§3, §24). It exists so the format
/// is **explicit in the plan**, as TACO's model requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalFormat {
    /// Always `true`: the layers are `t → s → r` and may not be permuted.
    pub levels_are_t_s_r: bool,
}

impl CanonicalFormat {
    /// The one canonical format.
    pub const CSF_T_S_R: Self = Self { levels_are_t_s_r: true };
}

/// The canonical iteration order (`your_2nd_task.md` §15).
///
/// Again a marker: the specialized kernel is *only* correct for this order, which
/// is precisely the "loop specialization" benefit (§18.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraversalOrder {
    /// `true` — mode 0, then mode 1, then mode 2.
    pub outermost_is_time: bool,
}

impl TraversalOrder {
    /// `t → s → r`, the only supported order.
    pub const MODE0_THEN1_THEN2: Self = Self {
        outermost_is_time: true,
    };
}

/// The fused reduction set of the expression (`your_2nd_task.md` §6, §12, §15).
///
/// All authorized reductions are always fused; the flags exist to make the
/// expression **explicit and inspectable** in the plan, and to let the kernel
/// refuse a partially-specified expression rather than silently dropping a
/// reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReductionSet {
    /// `T_i = Σ 1`
    pub count: bool,
    /// `S_i = ⋃ {s_i}` — exact set union
    pub senders: bool,
    /// `M_log = Σ q_i`, `q_i = round(2^64 · log1p_amt(A_i))`
    pub log_magnitude: bool,
    /// `C_R = Σ 1[a_i mod 10^6 = 0]`
    pub roundedness: bool,
    /// `t_min = min t_i`
    pub t_min: bool,
    /// `t_max = max t_i`
    pub t_max: bool,
}

impl ReductionSet {
    /// The structural reductions of `fanin.rs` (`T_i`, `S_i`, extrema).
    pub const STRUCTURAL: Self = Self {
        count: true,
        senders: true,
        log_magnitude: false,
        roundedness: false,
        t_min: true,
        t_max: true,
    };

    /// The **authorized fused** set: structural + `M_log` + `C_R`
    /// (`G2a` authorization; `C_Z` deliberately excluded — `G3` unresolved).
    pub const AUTHORIZED_FUSED: Self = Self {
        count: true,
        senders: true,
        log_magnitude: true,
        roundedness: true,
        t_min: true,
        t_max: true,
    };

    /// True iff every field is present.
    #[inline]
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count
            && self.senders
            && self.log_magnitude
            && self.roundedness
            && self.t_min
            && self.t_max
    }
}

/// The execution plan: **expression ⊗ format ⊗ schedule** (`your_2nd_task.md` §5, §15, §20).
///
/// This is a *description*, not an interpreter. It records what the specialized
/// kernel will do so that the schedule is explicit rather than implicit in a
/// hand-written loop nest, and it is validated once per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseExecutionPlan {
    /// Physical storage hierarchy of the canonical tensor.
    pub format: CanonicalFormat,
    /// Iteration order the kernel is specialized to.
    pub traversal: TraversalOrder,
    /// The focal transaction `f = (t_f, s_f, r_f)` — anchor of the whole plan.
    ///
    /// `r_f` (the *receiver restriction* of §15) is read from here via
    /// [`SparseExecutionPlan::focal_receiver`] rather than duplicated in a
    /// second field, so the two can never disagree.
    pub focal: FocalTransaction,
    /// Temporal restriction `[t_f − Δ, t_f]`, closed on both ends.
    pub temporal_window: FanInWindow,
    /// The fused reduction set.
    pub reductions: ReductionSet,
}

impl SparseExecutionPlan {
    /// The authorized Fan-In plan for `focal` and `window`.
    #[must_use]
    pub fn authorized(focal: FocalTransaction, window: FanInWindow) -> Self {
        Self {
            format: CanonicalFormat::CSF_T_S_R,
            traversal: TraversalOrder::MODE0_THEN1_THEN2,
            focal,
            temporal_window: window,
            reductions: ReductionSet::AUTHORIZED_FUSED,
        }
    }

    /// The receiver restriction `r == r_f` (§13).
    #[inline]
    #[must_use]
    pub const fn focal_receiver(&self) -> u32 {
        self.focal.receiver
    }

    /// The temporal restriction `[t_f − Δ, t_f]` (§15).
    ///
    /// Returns `None` (an empty, legitimate window — `L4` totality) when
    /// `t_f < Δ`, matching `fanin::window_bounds`.
    #[must_use]
    pub const fn temporal_window_bounds(&self) -> Option<(u64, u64)> {
        match self.focal.epoch.checked_sub(self.temporal_window.delta) {
            Some(lo) => Some((lo, self.focal.epoch)),
            None => None,
        }
    }

    /// Validate the plan against the canonical tensor.
    ///
    /// **Checks the PLAN only** — format, traversal, reduction set. It does
    /// **not** re-verify the tensor's CSF invariants: that is an `O(nnz)`
    /// whole-tensor scan, and doing it per focal would cost ~13x the actual
    /// `O(V_f)` traversal work. Tensor integrity is the caller's
    /// responsibility and is asserted once at load time.
    ///
    /// # Errors
    /// [`TensorError::Range`] if the plan is not the canonical format/traversal,
    /// or the reduction set is not the authorized fused one (§7 requires all
    /// reductions; a partial set would silently under-report Fan-In).
    pub fn validate(&self, _csf: &CsfTensor) -> Result<(), TensorError> {
        if self.format != CanonicalFormat::CSF_T_S_R {
            return Err(TensorError::Range(
                "sparse kernel: only the canonical t->s->r format is permitted".into(),
            ));
        }
        if self.traversal != TraversalOrder::MODE0_THEN1_THEN2 {
            return Err(TensorError::Range(
                "sparse kernel: only the canonical t->s->r traversal is permitted".into(),
            ));
        }
        if !self.reductions.is_full() {
            return Err(TensorError::Range(
                "sparse kernel: reduction set is not the authorized fused set".into(),
            ));
        }
        Ok(())
    }
}

/// The fixed-point scale for `M_log`: one `log1p` unit = `2^64` (§6).
///
/// Chosen so the largest representable contribution,
/// `round(2^64 · ln(1 + i64::MAX/1e6)) ≈ 29.853 · 2^64 ≈ 5.507e20`,
/// leaves ≈ 3.09e17 headroom before an `i128` accumulator could overflow.
/// Overflow is therefore unreachable for any realizable workload, and the
/// checked merge (§8) is a fail-closed guard rather than a live path.
pub const M_LOG_SCALE: f64 = 18_446_744_073_709_551_616.0; // 2^64

/// Quantize one canonical amount into the `M_log` fixed-point contribution:
/// `q = round(2^64 · log1p_amt(A))`, where `A = amount_micro / 1e6` and
/// `log1p_amt` is §S5:111's sign-preserving `log1p`
/// (`samld::diag::log1p_signed`, reused — not reimplemented).
///
/// **This is quantization, not exact `log1p`, and that is part of the semantic
/// definition** (task §6, decision record §4.1). Rounding is
/// **half-away-from-zero** (`f64::round`), stated explicitly because §S5 names
/// no tie rule.
///
/// `f64` appears only to evaluate the §S5 feature — the `K_10` presentation
/// class. The **returned value is `i128`**; no float enters the reduction state.
///
/// # Errors
/// [`TensorError::Range`] if the scaled value is not finite or leaves `i128`.
/// An `f64 → i128` `as` cast **saturates** in Rust and would silently change the
/// mathematical result, so the bound is checked explicitly first (`R_5`).
pub fn quantize_log1p(amount_micro: i64) -> Result<i128, TensorError> {
    // K10 presentation: micro-units -> major units.
    let major = amount_micro as f64 / 1e6;
    let feature = diag::log1p_signed(major);
    let scaled = feature * M_LOG_SCALE;
    if !scaled.is_finite() || scaled > i128::MAX as f64 || scaled < i128::MIN as f64 {
        return Err(TensorError::Range(format!(
            "M_log quantization out of range for amount_micro={amount_micro}"
        )));
    }
    // Round half away from zero; the bound check above makes the cast exact.
    Ok(scaled.round() as i128)
}

/// The fused Fan-In reduction state (`your_2nd_task.md` §6, §7).
///
/// `A = (T_i, S_i, M_log, C_R, t_min, t_max)`, `⊕ = (+, ∪, +, +, min, max)`.
///
/// `M_log` is `i128` and accumulated with **`checked_add`** (§8). Ordinary `+` is
/// prohibited (panics), `wrapping_add` is prohibited (silently breaks
/// exactness), and `saturating_add` is prohibited because it is **not
/// associative over the signed domain** — saturation would make `M_log`
/// partition-dependent and break `FanIn_P = FanIn_1`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FusedFanInState {
    count:   u64,               // T_i   +  checked
    senders: BTreeSet<u32>,     // S_i   U  exact
    m_log:   i128,              // M_log +  checked (i128)
    c_r:     u64,               // C_R   +  checked
    t_min:   Option<u64>,       // absorbing min
    t_max:   Option<u64>,       // absorbing max
}

impl FusedFanInState {
    /// The identity `⊥` (addendum §26).
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            count: 0,
            senders: BTreeSet::new(),
            m_log: 0,
            c_r: 0,
            t_min: None,
            t_max: None,
        }
    }

    /// Build a state whose `M_log` accumulator is preset to `m_log`.
    ///
    /// Exists so overflow can be tested **through the real `merge` API**: the
    /// authorized domain has ~3e17 headroom (§4.3bis), so a traversal can never
    /// reach the `i128` bound and the checked path would otherwise be dead code.
    /// This is a test seam, not a production path.
    #[inline]
    #[must_use]
    pub fn with_m_log_for_test(m_log: i128) -> Self {
        Self {
            m_log,
            ..Self::identity()
        }
    }

    /// Fold one **matched** canonical coordinate `(s, r_f, t, a)` into the
    /// state, updating **all** reductions at once (fusion, §12).
    ///
    /// # Errors
    /// [`TensorError::Range`] on `T_i`, `M_log`, or `C_R` overflow, or on a
    /// `M_log` quantization that is not representable (§8, §9, `R_5`).
    pub fn observe(&mut self, s: u32, t: u64, amount_micro: i64) -> Result<(), TensorError> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| TensorError::Range("fused T_i overflow (u64)".into()))?;
        self.senders.insert(s);                       // exact set insertion
        // §S5:117 whole-major-unit test: exact integer, no float.
        self.c_r = self
            .c_r
            .checked_add(u64::from(diag::roundedness_of(amount_micro)))
            .ok_or_else(|| TensorError::Range("fused C_R overflow (u64)".into()))?;
        let q = quantize_log1p(amount_micro)?;
        self.m_log = self
            .m_log
            .checked_add(q)
            .ok_or_else(|| TensorError::Range("fused M_log overflow (i128)".into()))?;
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

    /// Merge with `⊕` (§7). Associative and commutative over integers and
    /// exact sets, so the result is independent of partition boundaries and
    /// worker count (§16).
    ///
    /// # Errors
    /// [`TensorError::Range`] on `T_i`, `M_log`, or `C_R` overflow.
    pub fn merge(&mut self, other: &Self) -> Result<(), TensorError> {
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or_else(|| TensorError::Range("fused T_i overflow (u64)".into()))?;
        self.senders.extend(other.senders.iter().copied());   // exact union
        self.m_log = self
            .m_log
            .checked_add(other.m_log)
            .ok_or_else(|| TensorError::Range("fused M_log overflow (i128)".into()))?;
        self.c_r = self
            .c_r
            .checked_add(other.c_r)
            .ok_or_else(|| TensorError::Range("fused C_R overflow (u64)".into()))?;
        self.t_min = min_opt(self.t_min, other.t_min);
        self.t_max = max_opt(self.t_max, other.t_max);
        Ok(())
    }

    /// `T_i`.
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 { self.count }
    /// `S_i` — the exact unique incoming sender set.
    #[inline]
    #[must_use]
    pub const fn senders(&self) -> &BTreeSet<u32> { &self.senders }
    /// `M_log` — the quantized `log1p` accumulator (`i128`).
    #[inline]
    #[must_use]
    pub const fn m_log(&self) -> i128 { self.m_log }
    /// `C_R` — the roundedness count.
    #[inline]
    #[must_use]
    pub const fn c_r(&self) -> u64 { self.c_r }
    /// `⊥`-safe `t_min`.
    #[inline]
    #[must_use]
    pub const fn t_min(&self) -> Option<u64> { self.t_min }
    /// `⊥`-safe `t_max`.
    #[inline]
    #[must_use]
    pub const fn t_max(&self) -> Option<u64> { self.t_max }
}

/// `⊥`-absorbing `min` (addendum §26).
#[inline]
#[must_use]
const fn min_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(if x <= y { x } else { y }),
    }
}

/// `⊥`-absorbing `max` (addendum §26).
#[inline]
#[must_use]
const fn max_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(if x >= y { x } else { y }),
    }
}

/// The fused Fan-In result (`your_2nd_task.md` §14, §21).
///
/// A **reduction state, not a materialized output tensor** — this is precisely
/// where the native implementation is simpler than TACO (§14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedFanInResult {
    /// The focal transaction `f`.
    pub focal: FocalTransaction,
    /// `T_i` — matched canonical coordinates.
    pub t_i: u64,
    /// `S_i` — exact unique incoming sender set.
    pub senders: BTreeSet<u32>,
    /// `N_i = |S_i|`, derived so it cannot drift from the set (INV-09).
    pub n_i: u64,
    /// `RFI_f` as the exact pair `(N_i, T_i)`; `None` iff `T_i == 0`.
    pub rfi: Option<(u64, u64)>,
    /// `FI_topology = 1[N_i > 1]`.
    pub fi_topology: bool,
    /// `M_log` — quantized `log1p` accumulator (`i128`; never a float).
    pub m_log: i128,
    /// `C_R` — roundedness count.
    pub c_r: u64,
    /// `⊥`-safe `t_min`.
    pub t_min: Option<u64>,
    /// `⊥`-safe `t_max`.
    pub t_max: Option<u64>,
    /// `V_f` — CSF entries visited (`your_2nd_task.md` §30 traversal accounting).
    pub visited: u64,
    /// `K_f` — entries matching `t ∈ W_f ∧ r == r_f`.
    pub matched: u64,
}

impl FusedFanInResult {
    /// Finalize a fused reduction state into the canonical result.
    ///
    /// # Errors
    /// [`TensorError::Range`] if `K_f > V_f`, contradicting `fan_in.md` §19.
    pub fn finalize(
        focal: FocalTransaction,
        state: FusedFanInState,
        accounting: FanInAccounting,
    ) -> Result<Self, TensorError> {
        if accounting.matched > accounting.visited {
            return Err(TensorError::Range(format!(
                "fused fan-in K_f ({}) exceeds V_f ({})",
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
            m_log: state.m_log,
            c_r: state.c_r,
            t_min: state.t_min,
            t_max: state.t_max,
            visited: accounting.visited,
            matched: accounting.matched,
        })
    }
}

/// The **specialized sparse kernel**: one canonical CSF traversal, every
/// authorized reduction fused (`your_2nd_task.md` §10, §12).
///
/// ```text
/// for t in fibers[lo..hi]:              # t ∈ W_f            (temporal restriction)
///     for (t,s) in t_fiber(t):          # canonical s fibers (t→s)
///         for (s,r) in (t,s)_fiber:     # canonical r entries(t→s→r)
///             V_f += 1
///             if r == r_f:              # receiver restriction  (§13, scalar compare)
///                 K_f += 1
///                 observe(s, t, values[k])   # count + sender + M_log + C_R + extrema
/// ```
///
/// The receiver restriction is an **ordinary scalar comparison inside the
/// canonical traversal** (§13). No dense mask, no receiver index, no alternative
/// representation (§4, §10).
///
/// `lo..hi` is the mode-0 index window: partitioning that range is the
/// position-space schedule, which is why `FanIn_P = FanIn_1` (§16).
///
/// # Errors
/// [`TensorError::Range`] on any checked accumulator (`R_5`).
pub fn fold(
    csf: &CsfTensor,
    lo: usize,
    hi: usize,
    plan: &SparseExecutionPlan,
    state: &mut FusedFanInState,
    accounting: &mut FanInAccounting,
) -> Result<(), TensorError> {
    let r_f = plan.focal_receiver();
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
                accounting.visited = accounting
                    .visited
                    .checked_add(1)
                    .ok_or_else(|| TensorError::Range("fused V_f overflow (u64)".into()))?;
                // Receiver restriction: scalar compare on the canonical leaf.
                if csf.mode2_coords[k] == r_f {
                    accounting.matched = accounting
                        .matched
                        .checked_add(1)
                        .ok_or_else(|| TensorError::Range("fused K_f overflow (u64)".into()))?;
                    // One visit, all fused reductions. `values[k]` is aligned to
                    // mode 2 (P6), and is the only amount read.
                    state.observe(s, t, csf.values[k])?;
                }
            }
        }
    }
    Ok(())
}

/// Execute the plan: resolve the window, run the specialized kernel, finalize.
///
/// Serial (`P = 1`). Parallel execution partitions the same mode-0 range and
/// merges with the same `⊕`, which is why `FanIn_P = FanIn_1` (§16).
///
/// # Errors
/// [`TensorError`] on an invalid plan, window overflow, or any checked
/// accumulator overflow (`R_5` fail-closed).
pub fn execute(
    csf: &CsfTensor,
    plan: &SparseExecutionPlan,
) -> Result<FusedFanInResult, TensorError> {
    plan.validate(csf)?;
    let (lo, hi) = window_bounds(csf, &plan.focal, &plan.temporal_window)?;
    let mut state = FusedFanInState::identity();
    let mut accounting = FanInAccounting::default();
    fold(csf, lo, hi, plan, &mut state, &mut accounting)?;
    FusedFanInResult::finalize(plan.focal, state, accounting)
}

/// Convenience: build the authorized plan and execute it.
///
/// # Errors
/// As [`execute`].
pub fn fused_fanin(
    csf: &CsfTensor,
    focal: FocalTransaction,
    window: FanInWindow,
) -> Result<FusedFanInResult, TensorError> {
    execute(csf, &SparseExecutionPlan::authorized(focal, window))
}
