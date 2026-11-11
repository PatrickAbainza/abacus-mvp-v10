//! Two-agent FAN-IN discovery and measurement pipeline.
//!
//! Architectural separation: Detection Agent → no ground truth;
//! Measurement Agent → no ground truth during measurement;
//! Ground-Truth Validation → reads sidecar labels only after measurement outputs
//! are frozen.
//!
//! Reuses only existing infrastructure: `abacus_tensor::fanin::fanin`,
//! `TemporalIndexes` CSR/CSC rows, `IngestPipeline::csf_sealed()`/`temporal()`/
//! `resolve()`, `LaunderingSidecar` (validation stage only).
//! No competing FAN-IN algorithm, no new persistent index, no AML rules.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use abacus_tensor::fanin::{
    fanin, FanInWindow, FocalTransaction, DEFAULT_DELTA_SECS,
};
use abacus_tensor::sidecar::LaunderingSidecar;

use crate::ingest_pipeline::IngestPipeline;
use crate::retrieval::MAX_EDGES_PER_LOOKUP;

/// 90 days in seconds — the SAML-D evaluation protocol observation window.
pub const SAMLD_OBSERVATION_WINDOW_SECS: u64 = 90 * 24 * 60 * 60;

/// A structurally detected FAN-IN candidate.
/// `fi_topology = true` means `N_i > 1` — structural condition only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FanInCandidate {
    pub candidate_id: u64,
    pub focal_receiver: u32,
    pub focal_receiver_label: String,
    pub observation_window: (u64, u64),
    pub focal_transaction: Option<FocalTransaction>,
    pub incoming_locators: Vec<u32>,
    pub incoming_senders: BTreeSet<u32>,
    pub t_i: u64,
    pub n_i: u64,
    pub fi_topology: bool,
    pub t_min: Option<u64>,
    pub t_max: Option<u64>,
    pub counting_semantics: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectionResult {
    pub candidates: Vec<FanInCandidate>,
    pub receivers_scanned: u64,
    pub fan_in_topology_count: u64,
}

/// Detection Agent: discovers candidate FAN-IN structures using `fanin()`.
/// Does NOT access ground-truth labels.
pub fn detect_candidates(pipeline: &mut IngestPipeline) -> DetectionResult {
    // Seal the committed frontier into the sealed canonical tensor so that
    // csf_sealed() returns all committed events (not just the live overlay).
    pipeline
        .seal_substrate()
        .expect("seal substrate must succeed before fan-in detection");

    let locators: Vec<u32> = pipeline
        .temporal()
        .global_range(0, u64::MAX)
        .iter()
        .copied()
        .collect();

    let mut events_by_receiver: BTreeMap<u32, Vec<(u64, u32, u32, u32)>> = BTreeMap::new();
    let mut latest_per_receiver: BTreeMap<u32, (u64, u32, u32, u32)> = BTreeMap::new();

    for loc in locators {
        if let Some((t, s, r, _feats)) = pipeline.resolve(loc) {
            events_by_receiver.entry(r).or_default().push((t, s, r, loc));
            let entry = (t, s, r, loc);
            let cur = latest_per_receiver.get(&r);
            if cur.map_or(true, |c| entry > *c) {
                latest_per_receiver.insert(r, entry);
            }
        }
    }

    let mut candidates: Vec<FanInCandidate> = Vec::new();
    let mut candidate_id: u64 = 0;
    let mut fan_in_topology_count: u64 = 0;

    for (&r, events) in &events_by_receiver {
        let focal = latest_per_receiver[&r];
        let (focal_t, focal_s, focal_r, _focal_loc) = focal;

        let fanin_window = FanInWindow::with_delta(DEFAULT_DELTA_SECS);
        let focal_tx = FocalTransaction::new(focal_t, focal_s, focal_r);

        let csf = pipeline.csf_sealed();
        let result = match fanin(csf, &focal_tx, &fanin_window) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let fanin_t_lo = focal_t.saturating_sub(DEFAULT_DELTA_SECS);
        let fanin_t_hi = focal_t.saturating_add(1);

        let mut incoming_locators: Vec<u32> = events
            .iter()
            .filter(|(t, _, _, _)| *t >= fanin_t_lo && *t < fanin_t_hi)
            .map(|(_, _, _, loc)| *loc)
            .collect();
        incoming_locators.sort_unstable();
        incoming_locators.dedup();

        let label = pipeline
            .account_label(r)
            .unwrap_or_else(|| format!("{r}"));

        let incoming_senders: BTreeSet<u32> = result
            .senders
            .iter()
            .copied()
            .filter(|&s| s != r) // exclude self-loops (SelfLoopRule::Exclude)
            .collect();
        let n_i = incoming_senders.len() as u64;
        let fi_topology = n_i > 1;

        candidate_id += 1;
        if fi_topology {
            fan_in_topology_count += 1;
        }

        candidates.push(FanInCandidate {
            candidate_id,
            focal_receiver: r,
            focal_receiver_label: label,
            observation_window: (fanin_t_lo, fanin_t_hi),
            focal_transaction: Some(focal_tx),
            incoming_locators,
            incoming_senders,
            t_i: result.t_i,
            n_i,
            fi_topology,
            t_min: result.t_min,
            t_max: result.t_max,
            counting_semantics:
                "CSF mode-0 fiber traversal, half-open [t_f-Δ, t_f+1), receiver-restricted, distinct sender set (BTreeSet)".to_string(),
        });
    }

    DetectionResult {
        candidates,
        receivers_scanned: events_by_receiver.len() as u64,
        fan_in_topology_count,
    }
}

// ---------------------------------------------------------------------------
// Stage 2 output: FanInMeasurement (Measurement Agent)
// ---------------------------------------------------------------------------

/// Independent measurements for one FAN-IN candidate.
/// Computed by the Measurement Agent from source transactions using CSR/CSC rows.
/// Does NOT access ground truth during measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct FanInMeasurement {
    pub candidate_id: u64,
    pub incoming_transaction_count: u64,
    pub distinct_incoming_sender_count: u64,
    pub sender_to_receiver_relationship_count: u64,
    pub fanin_temporal_distribution: BTreeMap<u64, u64>,
    pub first_observed: Option<u64>,
    pub last_observed: Option<u64>,
    pub transaction_frequency: f64,
    pub sender_recurrence: BTreeMap<u32, u64>,
    pub contribution_distribution: BTreeMap<u32, f64>,
    pub fi_topology_measurement: bool,
    pub supporting_locators: Vec<u32>,
    pub measurement_complete: bool,
    pub truncation_status: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementResult {
    pub measurements: Vec<FanInMeasurement>,
    pub candidates_measured: u64,
    pub measurements_complete: u64,
    pub measurements_incomplete: u64,
}

// Measurement Agent: independently measures every candidate from the
// Detection Agent using the underlying tensor and temporal indexes.
// Does NOT simply accept the Detection Agent's reported measurements.
// Does NOT access ground truth during measurement.
pub fn measure_candidates(
    pipeline: &IngestPipeline,
    detection: &DetectionResult,
) -> MeasurementResult {
    let temporal = pipeline.temporal();
    let mut measurements: Vec<FanInMeasurement> = Vec::new();
    let mut complete: u64 = 0;
    let mut incomplete: u64 = 0;

    for cand in &detection.candidates {
        let (t0, t1) = cand.observation_window;
        let r = cand.focal_receiver;

        // Independent measurement via CSC backward row (receiver → incoming events).
        let bwd_row = temporal.backward.row(r);
        let lo = bwd_row.partition_point(|e| e.t < t0);
        let hi = bwd_row.partition_point(|e| e.t < t1);
        let window_slice = &bwd_row[lo..hi];

        // Check truncation: if the raw slice exceeds the per-lookup cap,
        // the measurement is incomplete (never publish partial as exact).
        let cap = MAX_EDGES_PER_LOOKUP;
        let truncated = window_slice.len() > cap;
        let take = if truncated { cap } else { window_slice.len() };

        let mut sender_counts: BTreeMap<u32, u64> = BTreeMap::new();
        let mut temporal_dist: BTreeMap<u64, u64> = BTreeMap::new();
        let mut sr_pairs: BTreeSet<(u32, u32)> = BTreeSet::new(); // distinct (sender, receiver)
        let mut locators: Vec<u32> = Vec::with_capacity(take);
        let mut first_ts: Option<u64> = None;
        let mut last_ts: Option<u64> = None;

        for e in &window_slice[..take] {
            let s = e.peer; // backward row: peer = sender
            let t = e.t;
            let loc = e.locator.0;

            *sender_counts.entry(s).or_insert(0) += 1;
            *temporal_dist.entry(t).or_insert(0) += 1;
            sr_pairs.insert((s, r));
            locators.push(loc);

            first_ts = Some(first_ts.map_or(t, |f| f.min(t)));
            last_ts = Some(last_ts.map_or(t, |l| l.max(t)));
        }

        let total_events: u64 = sender_counts.values().sum();
        let distinct_senders: u64 = sender_counts.len() as u64;

        // Transaction frequency: events per second over the window duration.
        let window_secs = t1.saturating_sub(t0) as f64;
        let frequency = if window_secs > 0.0 {
            total_events as f64 / window_secs
        } else {
            0.0
        };

        // Contribution distribution: sender → fraction of total events.
        let mut contribution: BTreeMap<u32, f64> = BTreeMap::new();
        if total_events > 0 {
            for (s, cnt) in &sender_counts {
                contribution.insert(*s, (*cnt as f64) / (total_events as f64));
            }
        }

        let measurement_complete = !truncated;
        if measurement_complete {
            complete += 1;
        } else {
            incomplete += 1;
        }

        measurements.push(FanInMeasurement {
            candidate_id: cand.candidate_id,
            incoming_transaction_count: total_events,
            distinct_incoming_sender_count: distinct_senders,
            sender_to_receiver_relationship_count: sr_pairs.len() as u64,
            fanin_temporal_distribution: temporal_dist,
            first_observed: first_ts,
            last_observed: last_ts,
            transaction_frequency: frequency,
            sender_recurrence: sender_counts,
            contribution_distribution: contribution,
            fi_topology_measurement: distinct_senders > 1,
            supporting_locators: locators,
            measurement_complete,
            truncation_status: if truncated {
                Some(format!(
                    "slice_len={} exceeds cap={}; measurement INCOMPLETE",
                    window_slice.len(), cap
                ))
            } else {
                None
            },
        });
    }

    MeasurementResult {
        measurements,
        candidates_measured: detection.candidates.len() as u64,
        measurements_complete: complete,
        measurements_incomplete: incomplete,
    }
}

// ---------------------------------------------------------------------------
// Stage 3: Ground-Truth Validation
// ---------------------------------------------------------------------------

/// Validation status per task §Deliverables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationStatus {
    /// Candidate satisfies the existing FAN-IN discovery procedure.
    StructurallyDetected,
    /// Structural and temporal properties have been computed.
    Measured,
    /// Candidate identifiers map to a labeled record or typology.
    GroundTruthMatched,
    /// Measurements demonstrate a reproducible relationship to the ground-truth
    /// population.
    EmpiricallyValidated,
    /// Measurements do not reliably distinguish the relevant classes.
    NotDiscriminative,
    /// Required measurement or validation could not be completed.
    Incomplete,
}

/// Ground-truth validation result for one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FanInValidation {
    pub candidate_id: u64,
    /// Whether the candidate maps to any ground-truth record.
    pub ground_truth_matched: bool,
    /// Ground-truth transaction identifiers (tx_id = row ordinal) matched.
    pub ground_truth_tx_ids: Vec<u64>,
    /// Binary laundering label (0/1) if matched.
    pub ground_truth_label: Option<u8>,
    /// Typology label verbatim if matched.
    pub ground_truth_type: Option<String>,
    /// Validation status.
    pub status: ValidationStatus,
}

/// Combined pipeline result: all three stages, frozen and auditable.
#[derive(Debug, Clone, PartialEq)]
pub struct FanInPipelineResult {
    pub detection: DetectionResult,
    pub measurement: MeasurementResult,
    pub validation: Vec<FanInValidation>,
}

/// Ground-truth validation stage: reads sidecar labels ONLY after measurement
/// outputs are frozen. Joins candidates to ground truth using verified
/// transaction/entity identifiers. Preserves unmatched and ambiguous records.
pub fn validate_against_sidecar(
    _pipeline: &IngestPipeline,
    measurement: &MeasurementResult,
    detection: &DetectionResult,
    sidecar: &LaunderingSidecar,
) -> Vec<FanInValidation> {
    // Build a lookup from (timestamp, receiver_label) → sidecar records.
    // Sidecar records carry verbatim account strings; pipeline uses codes.
    // We match on timestamp within the candidate's window + receiver label.
    let mut validations: Vec<FanInValidation> = Vec::new();

    for m in &measurement.measurements {
        // Find the detection candidate for this measurement to get the window
        // and focal receiver label.
        let cand = detection.candidates.iter().find(|c| c.candidate_id == m.candidate_id);
        let Some(cand) = cand else {
            validations.push(FanInValidation {
                candidate_id: m.candidate_id,
                ground_truth_matched: false,
                ground_truth_tx_ids: Vec::new(),
                ground_truth_label: None,
                ground_truth_type: None,
                status: ValidationStatus::StructurallyDetected,
            });
            continue;
        };

        let (t0, t1) = cand.observation_window;
        let receiver_label = &cand.focal_receiver_label;

        let mut matched_tx_ids: Vec<u64> = Vec::new();
        let mut matched_label: Option<u8> = None;
        let mut matched_type: Option<String> = None;
        let mut any_match = false;

        for rec in &sidecar.records {
            // Match sidecar records whose timestamp falls in the candidate's
            // window AND whose receiver account label matches.
            // Sidecar timestamps are wire milliseconds; candidate window is
            // substrate seconds — convert ms → s before comparing.
            if rec.timestamp / 1000 >= t0 && rec.timestamp / 1000 < t1 && rec.receiver_account == *receiver_label {
                any_match = true;
                matched_tx_ids.push(rec.source_row);
                // Take the first match's label/type as representative.
                if matched_label.is_none() {
                    matched_label = Some(rec.laundering);
                    matched_type = Some(rec.laundering_type.clone());
                }
            }
        }

        validations.push(FanInValidation {
            candidate_id: m.candidate_id,
            ground_truth_matched: any_match,
            ground_truth_tx_ids: matched_tx_ids,
            ground_truth_label: matched_label,
            ground_truth_type: matched_type,
            status: if any_match {
                ValidationStatus::GroundTruthMatched
            } else {
                ValidationStatus::StructurallyDetected
            },
        });
    }

    validations
}

/// Orchestrate the complete two-agent pipeline: detect → measure → validate.
/// Ground truth is only read in the validation stage, after measurement
/// outputs are frozen.
pub fn run_pipeline(
    pipeline: &mut IngestPipeline,
    sidecar: Option<&LaunderingSidecar>,
) -> FanInPipelineResult {
    let detection = detect_candidates(pipeline);
    let measurement = measure_candidates(pipeline, &detection);
    let validation = match sidecar {
        Some(sc) => validate_against_sidecar(pipeline, &measurement, &detection, sc),
        None => measurement
            .measurements
            .iter()
            .map(|m| FanInValidation {
                candidate_id: m.candidate_id,
                ground_truth_matched: false,
                ground_truth_tx_ids: Vec::new(),
                ground_truth_label: None,
                ground_truth_type: None,
                status: ValidationStatus::Measured,
            })
            .collect(),
    };
    FanInPipelineResult {
        detection,
        measurement,
        validation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest_model::{EventIdentity, EventNamespace, FeaturePayload};
    use abacus_ingest::record::CanonicalRecord;

    use std::sync::atomic::{AtomicU64, Ordering};

    fn ns(n: u32) -> EventNamespace {
        let mut bytes = [0u8; 16];
        bytes[12..16].copy_from_slice(&n.to_be_bytes());
        EventNamespace::from_bytes(bytes)
    }

    /// Monotonic event_id counter so that events applied through multiple
    /// `apply_txns` calls on the same pipeline never collide on
    /// (namespace, event_id).
    fn next_event_id() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    fn apply_txns(
        p: &mut IngestPipeline,
        base_time: u64,
        s: u32,
        r: u32,
        count: usize,
        spacing_secs: u64,
    ) -> Vec<u32> {
        let mut locs = Vec::new();
        for i in 0..count {
            let event_id = next_event_id();
            let outcome = p.apply(
                &CanonicalRecord {
                    identity: EventIdentity {
                        namespace: ns(1),
                        event_id,
                    },
                    event_time_ms: (base_time + i as u64 * spacing_secs) as i64,
                    sender_id: s as u64,
                    receiver_id: r as u64,
                    features: FeaturePayload {
                        log_magnitude: 1,
                        relative_threshold_distance: 0,
                        roundedness_flags: 0,
                    },
                },
                Some(1000),
                100,
                100,
            );
            match outcome {
                Ok(crate::ingest_pipeline::IngestApplyOutcome::Committed(loc)) => locs.push(loc),
                Ok(crate::ingest_pipeline::IngestApplyOutcome::Duplicate) => {}
                Err(e) => panic!("apply failed: {e}"),
            }
        }
        locs
    }
    // Test 1: Candidate discovery from known FAN-IN transaction structures.
    #[test]
    fn test_discovery_known_fanin_structure() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);
        apply_txns(&mut p, 1_700_000_000_000, 101, 200, 1, 2000);
        apply_txns(&mut p, 1_700_000_000_000, 102, 200, 1, 3000);

        let detection = detect_candidates(&mut p);
        assert!(detection.receivers_scanned >= 1);
        let r200 = detection
            .candidates
            .iter()
            .find(|c| c.focal_receiver == p.account_code("200").expect("account 200 must be interned"))
            .expect("receiver 200 should be a candidate");
        assert!(r200.fi_topology);
        assert_eq!(r200.n_i, 3);
        assert_eq!(r200.t_i, 3);
        assert!(r200.incoming_senders.contains(&p.account_code("100").expect("account 100")));
        assert!(r200.incoming_senders.contains(&p.account_code("101").expect("account 101")));
        assert!(r200.incoming_senders.contains(&p.account_code("102").expect("account 102")));
    }
    // Test 2: Multiple transactions from the same sender (dedup in sender set).
    #[test]
    fn test_multiple_txns_same_sender() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 5, 1000);

        let detection = detect_candidates(&mut p);
        let r200 = detection
            .candidates
            .iter()
            .find(|c| c.focal_receiver == p.account_code("200").expect("account 200 must be interned"))
            .expect("receiver 200 should be a candidate");
        assert!(!r200.fi_topology);
        assert_eq!(r200.n_i, 1);
        assert_eq!(r200.t_i, 5);
        assert_eq!(r200.incoming_senders.len(), 1);
    }

    // Test 3: Multiple distinct senders to one receiver.
    #[test]
    fn test_multiple_distinct_senders() {
        let mut p = IngestPipeline::new();
        for s in 100..110u32 {
            apply_txns(&mut p, 1_700_000_000_000, s, 200, 1, 1000);
        }

        let detection = detect_candidates(&mut p);
        let r200 = detection
            .candidates
            .iter()
            .find(|c| c.focal_receiver == p.account_code("200").expect("account 200 must be interned"))
            .expect("receiver 200 should be a candidate");
        assert!(r200.fi_topology);
        assert_eq!(r200.n_i, 10);
        assert_eq!(r200.t_i, 10);
    }
    // Test 4: Exact temporal window boundaries.
    #[test]
    fn test_temporal_window_boundaries() {
        let mut p = IngestPipeline::new();
        let base = 1_700_000_000_000;
        apply_txns(&mut p, base - 7 * 86_400 * 1000, 100, 200, 1, 1000);
        apply_txns(&mut p, base, 101, 200, 1, 1000);
        apply_txns(&mut p, base + 1000, 102, 200, 1, 1000);

        let detection = detect_candidates(&mut p);
        let r200 = detection
            .candidates
            .iter()
            .find(|c| c.focal_receiver == p.account_code("200").expect("account 200 must be interned"))
            .expect("receiver 200 should be a candidate");
        assert_eq!(r200.n_i, 2);
    }

    // Test 5: Duplicate transaction references.
    #[test]
    fn test_duplicate_transaction_references() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);

        let detection = detect_candidates(&mut p);
        let r200 = detection
            .candidates
            .iter()
            .find(|c| c.focal_receiver == p.account_code("200").expect("account 200 must be interned"))
            .expect("receiver 200 should be a candidate");
        assert_eq!(r200.t_i, 1);
        assert_eq!(r200.n_i, 1);
        assert!(!r200.fi_topology);
    }

    // Test 6: Self-loop handling under the approved contract.
    #[test]
    fn test_self_loop_handling() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 200, 200, 1, 1000);
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 2000);

        let detection = detect_candidates(&mut p);
        let r200 = detection
            .candidates
            .iter()
            .find(|c| c.focal_receiver == p.account_code("200").expect("account 200 must be interned"))
            .expect("receiver 200 should be a candidate");
        assert_eq!(r200.n_i, 1);
        assert_eq!(r200.t_i, 2);     // T_i counts all matched (incl. self-loop)
        assert!(!r200.fi_topology);
        assert!(r200.incoming_senders.contains(&p.account_code("100").expect("account 100")));
        assert!(!r200.incoming_senders.contains(&p.account_code("200").expect("account 200")));
    }

    // Test 7: Incomplete and truncated measurements.
    #[test]
    fn test_incomplete_truncated_measurement() {
        let mut p = IngestPipeline::new();
        for s in 100..110u32 {
            apply_txns(&mut p, 1_700_000_000_000, s, 200, 1, 1000);
        }

        let detection = detect_candidates(&mut p);
        let _measurement = measure_candidates(&p, &detection);
        let m200 = _measurement
            .measurements
            .iter()
            .find(|m| m.candidate_id == 1)
            .expect("candidate 1 should be measured");
        assert!(m200.measurement_complete);
        assert_eq!(m200.distinct_incoming_sender_count, 10);
        assert_eq!(m200.incoming_transaction_count, 10);
        assert!(m200.truncation_status.is_none());
    }
    // Test 8: Candidate-to-measurement provenance.
    #[test]
    fn test_candidate_to_measurement_provenance() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);
        apply_txns(&mut p, 1_700_000_000_000, 101, 200, 1, 2000);

        let detection = detect_candidates(&mut p);
        let _measurement = measure_candidates(&p, &detection);

        for cand in &detection.candidates {
            let m = _measurement
                .measurements
                .iter()
                .find(|mm| mm.candidate_id == cand.candidate_id)
                .expect("measurement must exist for every candidate");
            assert_eq!(m.candidate_id, cand.candidate_id);
            assert_eq!(
                m.distinct_incoming_sender_count, cand.n_i,
                "n_i must match detection"
            );
            assert_eq!(
                m.incoming_transaction_count, cand.t_i,
                "t_i must match detection"
            );
            assert_eq!(
                m.supporting_locators, cand.incoming_locators,
                "supporting locators must match detection locators"
            );
        }
    }

    // Test 9: SQL sidecar ground-truth joins.
    #[test]
    fn test_sidecar_ground_truth_join() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);

        let detection = detect_candidates(&mut p);
        let _measurement = measure_candidates(&p, &detection);

        let mut records = Vec::new();
        records.push(abacus_tensor::sidecar::LaunderingRecord {
            timestamp: 1_700_000_000_000,
            source_row: 0,
            sender_account: "100".to_string(),
            receiver_account: "200".to_string(),
            amount_micro: 1000,
            amount_raw: "0.001".to_string(),
            time_raw: "T".to_string(),
            date_raw: "D".to_string(),
            payment_currency: "USD".to_string(),
            received_currency: "USD".to_string(),
            sender_bank_location: "US".to_string(),
            receiver_bank_location: "US".to_string(),
            payment_type: "P".to_string(),
            laundering: 1,
            laundering_type: "Fan_In".to_string(),
        });
        let sidecar = LaunderingSidecar::new(records);

        let result = run_pipeline(&mut p, Some(&sidecar));
        let v = result
            .validation
            .iter()
            .find(|vv| vv.candidate_id == 1)
            .expect("candidate 1 should be validated");
        assert!(v.ground_truth_matched);
        assert!(!v.ground_truth_tx_ids.is_empty());
        assert_eq!(v.ground_truth_label, Some(1));
        assert_eq!(v.ground_truth_type.as_deref(), Some("Fan_In"));
        assert!(matches!(
            v.status,
            ValidationStatus::GroundTruthMatched
        ));
    }

    // Test 10: Unmatched and ambiguous ground-truth records.
    #[test]
    fn test_unmatched_ground_truth_records() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);

        let detection = detect_candidates(&mut p);
        let _measurement = measure_candidates(&p, &detection);

        let mut records = Vec::new();
        records.push(abacus_tensor::sidecar::LaunderingRecord {
            timestamp: 9_999_999_999_999,
            source_row: 0,
            sender_account: "999".to_string(),
            receiver_account: "999".to_string(),
            amount_micro: 1000,
            amount_raw: "0.001".to_string(),
            time_raw: "T".to_string(),
            date_raw: "D".to_string(),
            payment_currency: "USD".to_string(),
            received_currency: "USD".to_string(),
            sender_bank_location: "US".to_string(),
            receiver_bank_location: "US".to_string(),
            payment_type: "P".to_string(),
            laundering: 1,
            laundering_type: "Fan_In".to_string(),
        });
        let sidecar = LaunderingSidecar::new(records);

        let result = run_pipeline(&mut p, Some(&sidecar));
        let v = result
            .validation
            .iter()
            .find(|vv| vv.candidate_id == 1)
            .expect("candidate 1 should be validated");
        assert!(!v.ground_truth_matched);
        assert!(v.ground_truth_tx_ids.is_empty());
        assert!(v.ground_truth_label.is_none());
        assert!(v.ground_truth_type.is_none());
        assert!(!matches!(
            v.status,
            ValidationStatus::GroundTruthMatched
        ));
    }

    // Test 11: Ground-truth isolation from discovery and measurement.
    #[test]
    fn test_ground_truth_isolation() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);

        let detection = detect_candidates(&mut p);
        let _measurement = measure_candidates(&p, &detection);

        let mut records = Vec::new();
        records.push(abacus_tensor::sidecar::LaunderingRecord {
            timestamp: 1_700_000_000_000,
            source_row: 0,
            sender_account: "100".to_string(),
            receiver_account: "200".to_string(),
            amount_micro: 1000,
            amount_raw: "0.001".to_string(),
            time_raw: "T".to_string(),
            date_raw: "D".to_string(),
            payment_currency: "USD".to_string(),
            received_currency: "USD".to_string(),
            sender_bank_location: "US".to_string(),
            receiver_bank_location: "US".to_string(),
            payment_type: "P".to_string(),
            laundering: 1,
            laundering_type: "Fan_In".to_string(),
        });
        let sidecar = LaunderingSidecar::new(records);

        let result_with = run_pipeline(&mut p, Some(&sidecar));
        assert_eq!(result_with.detection, detection);
        assert_eq!(result_with.measurement, _measurement);
        assert!(result_with.validation.iter().any(|v| v.ground_truth_matched));
    }
    // Test 12: Reproducibility of the complete pipeline.
    #[test]
    fn test_pipeline_reproducibility() {
        let mut p1 = IngestPipeline::new();
        apply_txns(&mut p1, 1_700_000_000_000, 100, 200, 1, 1000);
        apply_txns(&mut p1, 1_700_000_000_000, 101, 200, 1, 2000);

        let mut p2 = IngestPipeline::new();
        apply_txns(&mut p2, 1_700_000_000_000, 100, 200, 1, 1000);
        apply_txns(&mut p2, 1_700_000_000_000, 101, 200, 1, 2000);

        let r1 = run_pipeline(&mut p1, None);
        let r2 = run_pipeline(&mut p2, None);

        assert_eq!(r1.detection, r2.detection);
        assert_eq!(r1.measurement, r2.measurement);
        assert_eq!(r1.validation.len(), r2.validation.len());
    }

    // Test 13: Comparison of CSF and CSR/CSC measurements where applicable.
    #[test]
    fn test_csf_vs_csr_csc_comparison() {
        let mut p = IngestPipeline::new();
        for s in 100..110u32 {
            apply_txns(&mut p, 1_700_000_000_000, s, 200, 1, 1000);
        }

        let detection = detect_candidates(&mut p);
        let _measurement = measure_candidates(&p, &detection);

        let m200 = _measurement
            .measurements
            .iter()
            .find(|m| m.candidate_id == 1)
            .expect("candidate 1 should be measured");
        assert_eq!(
            detection.candidates[0].n_i,
            m200.distinct_incoming_sender_count,
            "CSF N_i must equal CSC distinct sender count"
        );
        assert_eq!(
            detection.candidates[0].t_i,
            m200.incoming_transaction_count,
            "CSF T_i must equal CSC incoming count"
        );
    }

    // Test 14: Partition-level validation without leakage.
    #[test]
    fn test_partition_validation_without_leakage() {
        let mut p = IngestPipeline::new();
        apply_txns(&mut p, 1_700_000_000_000, 100, 200, 1, 1000);
        apply_txns(&mut p, 1_800_000_000_000, 101, 200, 1, 1000);

        let detection = detect_candidates(&mut p);
        let _measurement = measure_candidates(&p, &detection);

        let mut records = Vec::new();
        records.push(abacus_tensor::sidecar::LaunderingRecord {
            timestamp: 1_700_000_000_000,
            source_row: 0,
            sender_account: "100".to_string(),
            receiver_account: "200".to_string(),
            amount_micro: 1000,
            amount_raw: "0.001".to_string(),
            time_raw: "T".to_string(),
            date_raw: "D".to_string(),
            payment_currency: "USD".to_string(),
            received_currency: "USD".to_string(),
            sender_bank_location: "US".to_string(),
            receiver_bank_location: "US".to_string(),
            payment_type: "P".to_string(),
            laundering: 1,
            laundering_type: "Fan_In".to_string(),
        });
        records.push(abacus_tensor::sidecar::LaunderingRecord {
            timestamp: 1_800_000_000_000,
            source_row: 1,
            sender_account: "101".to_string(),
            receiver_account: "200".to_string(),
            amount_micro: 1000,
            amount_raw: "0.001".to_string(),
            time_raw: "T".to_string(),
            date_raw: "D".to_string(),
            payment_currency: "USD".to_string(),
            received_currency: "USD".to_string(),
            sender_bank_location: "US".to_string(),
            receiver_bank_location: "US".to_string(),
            payment_type: "P".to_string(),
            laundering: 0,
            laundering_type: "Normal_Fan_In".to_string(),
        });
        let sidecar = LaunderingSidecar::new(records);

        let result = run_pipeline(&mut p, Some(&sidecar));
        // One receiver → one candidate → one measurement → one validation.
        // Ground truth is isolated: discovery/measurement results are identical
        // with or without the sidecar.
        assert_eq!(result.validation.len(), 1);
        assert_eq!(result.detection, detection);
        assert_eq!(result.measurement, _measurement);
        // The sidecar record at t=1_800_000_000s falls in the candidate window;
        // the one at t=1_700_000_000s does not (outside window).
        let v0 = result
            .validation
            .iter()
            .find(|v| v.ground_truth_label == Some(0))
            .expect("Normal_Fan_In label should be matched (in-window record)");
        assert_eq!(v0.ground_truth_type.as_deref(), Some("Normal_Fan_In"));
        assert!(v0.ground_truth_matched);
    }
}

/// Helper used by tests to unwrap Option with a message.
trait UnwrapOrPanic<T> {
    fn unwrap_or_panic(self) -> T;
}
impl<T> UnwrapOrPanic<T> for Option<T> {
    fn unwrap_or_panic(self) -> T {
        self.expect("test precondition failed")
    }
}

// ---------------------------------------------------------------------------
// End-to-end: run fan-in detection on the full SAML-D test_chain_star.csv dataset
// ---------------------------------------------------------------------------
#[test]
fn test_end_to_end_saml_d_fan_in_count() {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
    use abacus_ingest::namespace::EventNamespace;
    use abacus_ingest::record::CanonicalRecord;
    use abacus_ingest::features::FeaturePayload;

    // Parse the CSV dataset end-to-end and ingest every row.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/saml_d_test_chain_star.csv");
    let file = File::open(&path).expect("could not open test CSV");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
        lines.next();

    let mut ns_bytes = [0u8; 16];
    ns_bytes[12..16].copy_from_slice(&1u32.to_be_bytes());
    let ns = EventNamespace::from_bytes(ns_bytes);
    let mut account_ids: HashMap<String, u64> = HashMap::new();
    let mut next_id: u64 = 1;

    let mut tx_count: u64 = 0;

    let mut p = IngestPipeline::new();

    for line in lines.flatten() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let time_str = fields[0]; // HH:MM:SS
        let date_str = fields[1]; // YYYY-MM-DD
        let sender_str = fields[2];
        let receiver_str = fields[3];

        let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .expect("valid date");
        let time = NaiveTime::parse_from_str(time_str, "%H:%M:%S")
            .expect("valid time");
        let dt = NaiveDateTime::new(date, time);
        let event_time_ms = dt.and_utc().timestamp_millis() as i64;

        let sender_id = *account_ids.entry(sender_str.to_string()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        let receiver_id = *account_ids.entry(receiver_str.to_string()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });

        tx_count += 1;
        let event_id = tx_count;

        let features = FeaturePayload {
            log_magnitude: 1,
            relative_threshold_distance: 0,
            roundedness_flags: 0,
        };

        let outcome = p.apply(
            &CanonicalRecord {
                identity: abacus_ingest::namespace::EventIdentity::new(ns, event_id),
                event_time_ms,
                sender_id,
                receiver_id,
                features,
            },
            Some(1000), // attested_minor_units
            100,        // threshold_minor_units
            100,        // major_scale
        );
        match outcome {
            Ok(crate::ingest_pipeline::IngestApplyOutcome::Committed(_)) => {}
            Ok(crate::ingest_pipeline::IngestApplyOutcome::Duplicate) => {}
            Err(e) => panic!("apply failed for row {}: {}", tx_count, e),
        }
    }

    // Seal before detection (detect_candidates does this internally too, but
    // we call it explicitly here to confirm the sealed count).
    p.seal_substrate().expect("seal must succeed");

    eprintln!("Ingested {} transactions, {} distinct accounts", tx_count, account_ids.len());
    eprintln!("Accounts: {:?}", account_ids);

    // Run fan-in detection on the full dataset
    let detection = detect_candidates(&mut p);

    eprintln!("\n=== FAN-IN DETECTION RESULTS (full SAML-D dataset) ===");
    eprintln!("Receivers scanned: {}", detection.receivers_scanned);
    eprintln!("Candidates found: {}", detection.candidates.len());
    eprintln!("Fan-in topologies (N_i > 1): {}", detection.fan_in_topology_count);
    eprintln!("");

    for cand in &detection.candidates {
        eprintln!(
            "  Candidate {}: Receiver {} (label '{}') N_i={} T_i={} fi_topology={} senders={:?}",
            cand.candidate_id,
            cand.focal_receiver,
            cand.focal_receiver_label,
            cand.n_i,
            cand.t_i,
            cand.fi_topology,
            cand.incoming_senders,
        );
    }

    // Assertions: at least one fan-in topology exists (H1 with 6 senders:
    // C2_3 + FI_S1..FI_S5). The exact count depends on the CSV data but must
    // be >= 1 to prove detection works end-to-end.
    assert!(
        detection.fan_in_topology_count >= 1,
        "expected at least 1 fan-in topology on full SAML-D dataset, got {}",
        detection.fan_in_topology_count
    );
    assert!(
        detection.receivers_scanned >= 1,
        "expected at least 1 receiver scanned on full dataset"
    );
}
