//! Row 26 (§B.2) + P11: dataset statistics (deterministic metadata).
//!
//! The admissible operation set does not include presentation data, and
//! floating point appears only here. `f64` is therefore confined to the
//! derived ratios `sparsity` and `compression_ratio`; it never defines, stores,
//! aggregates, or reconstructs tensor values, and nothing here is ever read
//! back into tensor logic (`a_{21}` isolation).
//!
//! Ratios and byte counts are diagnostics; the seed for the non-random,
//! non-tensor counters is entirely the build metadata plus the analytic CSF
//! payload layout (`a_{09}` length arithmetic).

use crate::coo::builder::CanonicalBuild;
use crate::csf::compress::CsfTensor;
use crate::csf::serialize::checksum;
use crate::tensor::value::MINOR_SCALE_EXP;
use crate::TensorError;
use crate::coo::entry::TxEntry;
use crate::prims::add;

/// Settlement diagnostic (G8) — how raw source rows were absorbed by the
/// canonical `d_settle` pass, split by **mechanism**.
///
/// `d_settle` (`composites::settle`) runs **before** the CSF exists:
/// `source rows -> project -> settle -> compress -> CSF`. For each source
/// coordinate `c = (t,s,r)` with `m_c` contributing rows it computes
/// `A_c = Σ_{coord(x)=c} a_x` and then applies `R_6` zero-elimination. That
/// yields exactly two absorbing outcomes, and they are **not** the same event:
///
/// * **Aggregation** — `A_c ≠ 0`: `c` survives into `supp(T)` carrying the
///   settled sum, and `m_c − 1` rows are absorbed.
/// * **Elimination** — `A_c = 0`: `R_6` drops `c`, so it is **absent from
///   `supp(T)` entirely** and all `m_c` rows are absorbed. Fan-In never sees
///   it — not as a zero-valued observation, but as absent.
///
/// `m_c = 1, A_c = 0` is the degenerate case of the same rule (a single
/// zero-valued source row is eliminated, contributing `m_c = 1`).
///
/// This is a **build-time provenance counter only**. It is never a Fan-In
/// lookup structure, never an index, and never read by any tensor path
/// (`a_{21}` isolation, P11). It exists so the settlement behaviour is
/// auditable without reconstructing pre-settlement transactions.
///
/// **Conservation identity** (checked by `total_absorbed`):
/// `absorbed_by_aggregation + eliminated_by_cancellation = raw_rows − |supp(T)|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SettlementDiagnostic {
    /// `C_settled` — surviving coordinates that absorbed `m_c > 1` rows.
    pub aggregated_coordinates: u64,
    /// `D_agg` — source rows absorbed into surviving (aggregated) coordinates.
    pub absorbed_by_aggregation: u64,
    /// `D_elim` — source rows whose coordinate settled to zero and was removed.
    pub eliminated_by_cancellation: u64,
}

impl SettlementDiagnostic {
    /// `D_absorbed` — total absorbed rows, the sum of both mechanisms.
    ///
    /// This is exactly `num_duplicates_absorbed` as previously reported by a
    /// single difference-of-totals; it is now decomposed so the two
    /// mechanisms are never conflated.
    #[must_use]
    pub const fn total_absorbed(&self) -> u64 {
        self.absorbed_by_aggregation + self.eliminated_by_cancellation
    }
}

/// Compute the settlement diagnostic from the **pre-settlement** entry list.
///
/// Must be called with the same slice that is handed to
/// [`crate::composites::settle`], i.e. **before** settling destroys the
/// per-coordinate multiplicities. The grouping mirrors `settle` exactly
/// (`Ord` is `d_lex` on the coordinate, so equal coordinates become adjacent)
/// and uses the same checked `a_{02}` summation, so a coordinate eliminated by
/// `R_6` here is exactly one eliminated there.
///
/// `raw_rows` is the pre-settlement row count and `nnz` the post-settlement
/// support size; the identity `D_agg + D_elim = raw_rows − nnz` is asserted by
/// the caller and by the test suite.
///
/// # Errors
/// [`TensorError::Range`] (`E_rng`) when a coordinate sum is unrepresentable —
/// the same failure `settle` itself raises on the same input (`R_5`).
pub fn settlement_diagnostic(raw: &[TxEntry]) -> Result<SettlementDiagnostic, TensorError> {
    let mut v = raw.to_vec();
    v.sort(); // a_10 under d_lex (P1), identical to settle
    let mut d = SettlementDiagnostic::default();
    let mut i = 0usize;
    while i < v.len() {
        let (t, s, r) = (v[i].t, v[i].s, v[i].r);
        let mut acc = v[i].value;
        let mut m: u64 = 1;
        let mut j = i + 1;
        while j < v.len() && v[j].t == t && v[j].s == s && v[j].r == r {
            acc = add(acc, v[j].value, "settlement diagnostic sum (G8)")?;
            m += 1;
            j += 1;
        }
        if acc == 0 {
            // R6: the coordinate is removed, so every one of its rows is
            // eliminated rather than aggregated.
            d.eliminated_by_cancellation += m;
        } else if m > 1 {
            d.aggregated_coordinates += 1;
            d.absorbed_by_aggregation += m - 1;
        }
        i = j;
    }
    Ok(d)
}

/// Deterministic dataset/tensor statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetStats {
    /// Raw CSV data rows observed.
    pub num_transactions: u64,
    /// Distinct `t` coordinates.
    pub num_unique_timestamps: usize,
    /// Distinct `t` coordinates at second grain.
    pub num_active_seconds: usize,
    /// Distinct senders.
    pub num_senders: usize,
    /// Distinct receivers.
    pub num_receivers: usize,
    /// Nonzero coordinates `|supp(T)|`.
    pub num_unique_coordinates: usize,
    /// `D_absorbed` — source rows absent from `supp(T)`, i.e.
    /// `raw_rows − |supp(T)|`.
    ///
    /// **This is the sum of two distinct mechanisms**, previously conflated by
    /// a single difference-of-totals (G8):
    /// `absorbed_by_aggregation + eliminated_by_cancellation`. A coordinate that
    /// aggregated `m_c > 1` rows keeps its settled sum; a coordinate whose sum
    /// is `0` is removed outright by `R_6`. Use the decomposed fields — not
    /// this total — to reason about settlement behaviour.
    pub num_duplicates_absorbed: u64,
    /// `C_settled` — surviving coordinates that absorbed `m_c > 1` rows.
    pub num_aggregated_coordinates: u64,
    /// `D_agg` — source rows absorbed into surviving aggregated coordinates.
    pub absorbed_by_aggregation: u64,
    /// `D_elim` — source rows whose coordinate settled to zero and was removed
    /// from `supp(T)` by `R_6`.
    pub eliminated_by_cancellation: u64,
    /// `|supp(T)| / (|T|·|S|·|R|)`; `0.0` when any mode is empty.
    /// Presentation-only `f64`; never feeds tensor arithmetic.
    pub sparsity: f64,
    /// Analytic COO bytes: `nnz · (8 + 4 + 4 + 8)`.
    pub coo_bytes: usize,
    /// Analytic CSF payload bytes: `I0 + P0 + I1 + P1 + I2 + V`.
    pub csf_bytes: usize,
    /// `coo_bytes / csf_bytes`; `0.0` when the CSF is empty.
    /// Presentation-only `f64`; never feeds tensor arithmetic.
    pub compression_ratio: f64,
    /// Minimum timestamp.
    pub t_min: u64,
    /// Maximum timestamp.
    pub t_max: u64,
}

/// Row 26 — compute the statistics of a canonical build and its CSF.
///
/// # Errors
/// Never fails; the signature is total so callers need no error path for a
/// diagnostic that cannot reject (byte counts use `a_{09}`/`a_{04}` length
/// arithmetic whose operands are bounded by the stored data).
#[must_use]
pub fn compute(build: &CanonicalBuild, csf: &CsfTensor) -> DatasetStats {
    let nnz = build.coo.len();
    let nt = csf.mode0_coords.len();
    let ns = build.senders.table.len();
    let nr = build.receivers.table.len();
    let domain: u128 = (nt as u128) * (ns as u128) * (nr as u128);
    let coo_bytes = nnz * 24;
    let csf_bytes = crate::ingest::engine::csf_byte_size(csf);
    DatasetStats {
        num_transactions: build.raw_rows,
        num_unique_timestamps: nt,
        num_active_seconds: nt,
        num_senders: ns,
        num_receivers: nr,
        num_unique_coordinates: nnz,
        // G8: the total is unchanged, but it is now derived from the split so
        // the two absorption mechanisms can never be silently conflated again.
        // Conservation: D_agg + D_elim == raw_rows - nnz.
        num_duplicates_absorbed: build.settlement.total_absorbed(),
        num_aggregated_coordinates: build.settlement.aggregated_coordinates,
        absorbed_by_aggregation: build.settlement.absorbed_by_aggregation,
        eliminated_by_cancellation: build.settlement.eliminated_by_cancellation,
        sparsity: if domain == 0 {
            0.0
        } else {
            (nnz as f64) / (domain as f64)
        },
        coo_bytes,
        csf_bytes,
        compression_ratio: if csf_bytes == 0 {
            0.0
        } else {
            (coo_bytes as f64) / (csf_bytes as f64)
        },
        t_min: build.t_min,
        t_max: build.t_max,
    }
}

impl DatasetStats {
    /// Row 26 — deterministic JSON rendering: fixed key order, no map
    /// iteration, no environment fields (`R_7`).
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"num_transactions\": {},\n  \"num_unique_timestamps\": {},\n  \
             \"num_active_seconds\": {},\n  \"num_senders\": {},\n  \"num_receivers\": {},\n  \
             \"num_unique_coordinates\": {},\n  \"num_duplicates_absorbed\": {},\n  \
             \"num_aggregated_coordinates\": {},\n  \
             \"absorbed_by_aggregation\": {},\n  \
             \"eliminated_by_cancellation\": {},\n  \
             \"sparsity\": {:.12e},\n  \"coo_bytes\": {},\n  \"csf_bytes\": {},\n  \
             \"compression_ratio\": {:.6},\n  \"t_min\": {},\n  \"t_max\": {}\n}}",
            self.num_transactions,
            self.num_unique_timestamps,
            self.num_active_seconds,
            self.num_senders,
            self.num_receivers,
            self.num_unique_coordinates,
            self.num_duplicates_absorbed,
            self.num_aggregated_coordinates,
            self.absorbed_by_aggregation,
            self.eliminated_by_cancellation,
            self.sparsity,
            self.coo_bytes,
            self.csf_bytes,
            self.compression_ratio,
            self.t_min,
            self.t_max
        )
    }

    /// Row 26 — statistics plus artifact provenance block.
    ///
    /// `artifact_bytes` is the serialized `tensor.csf` image; its whole-file
    /// checksum is emitted as `artifact_checksum` and, as the same value under
    /// the `checksum` key, for the previously published key name. The two
    /// checksum quantities stay distinct (P11): the whole-file value here is
    /// not the embedded payload checksum of the binary format.
    ///
    /// `tensor_file` records the INVOCATION output basename as provenance
    /// only — it is not reproducibility metadata, and the artifact bytes do not
    /// depend on it.
    #[must_use]
    pub fn to_json_with_artifact(&self, artifact_bytes: &[u8], tensor_file: &str) -> String {
        let mut js = self.to_json();
        js.pop(); // trailing '}'
        let whole = checksum(artifact_bytes);
        let escaped = tensor_file.replace('\\', "\\\\").replace('"', "\\\"");
        js.push_str(&format!(
            ",\n  \"scale_exp\": {},\n  \"artifact_checksum\": {},\n  \
             \"checksum\": {},\n  \"tensor_file\": \"{}\"\n}}\n",
            MINOR_SCALE_EXP, whole, whole, escaped
        ));
        js
    }
}
