//! Normalization: CSV bytes in, validated canonical records and bounded
//! `ABACSEG1` chunks out.
//!
//! This is the browser client's core, and it implements §12's procedure
//! end-to-end: map columns, apply the explicit policy, validate every row,
//! intern identifiers, derive the frozen features, encode bounded chunks.
//!
//! ## Row-level validation, not file-level failure
//!
//! A single bad row does not abort the import. Each row is classified
//! independently and a [`RowIssue`] is recorded with its **source line number**
//! and a structured [`abacus_ingest::error::ErrorCode`] — never the offending
//! value (§12: "The UI must expose actionable errors without displaying raw
//! account strings or raw amounts"). The accepted rows still produce a complete,
//! valid segment.
//!
//! What *does* abort is a structural problem that would make the whole upload
//! meaningless: an unparseable file, an invalid plan, an identifier-capacity
//! overflow. §6 is explicit about the capacity case: "If a segment exceeds its
//! selected capacity, stop that segment and report the failure. Do not wrap IDs
//! or silently change the width mid-segment."
//!
//! ## Event identity
//!
//! `event_id` is the **stable parsed CSV record ordinal** (wire §2
//! `identity_scheme = 1`), counted over *accepted* rows in file order and
//! starting at 1. That is what makes a retry of the same file re-derive
//! identical identities, so the backend's CAS treats it as a duplicate delivery
//! instead of a second application (§5.1, §13).

use abacus_ingest::error::{ErrorCode, IngestError, IngestResult};
use abacus_ingest::features::FeaturePayload;
use abacus_ingest::namespace::EventIdentity;
use abacus_ingest::record::CanonicalRecord;

use crate::csv::{iter_rows, CsvRow};
use crate::identifiers::IdentifierInterner;
use crate::mapping::{field_at, CanonicalField};
use crate::policy::IngestPlan;
use crate::{amount, timestamp};

/// One rejected row, with everything the UI needs and nothing it must not show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowIssue {
    /// 1-based source line number of the record (§12).
    pub line: usize,
    /// Stable error category (Golden Source §16).
    pub code: ErrorCode,
    /// A structural explanation that never contains a raw account or amount.
    pub detail: String,
}

/// The normalization preview: counts and every row-level problem.
///
/// §12 step 5: "Preview normalized records and report invalid rows, ambiguous
/// values, identifier collisions, and unsupported amounts." Every count here is
/// privacy-safe — no account string, no amount, no label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preview {
    /// Data rows seen in the file.
    pub input_rows: u64,
    /// Rows that produced a canonical record.
    pub accepted_rows: u64,
    /// Rows rejected, each with a reason.
    pub rejected_rows: u64,
    /// Distinct account identifiers interned across senders and receivers.
    pub distinct_accounts: u64,
    /// The account-ID width actually in force.
    pub account_id_width: u8,
    /// Earliest and latest accepted event time in milliseconds.
    pub event_time_bounds: Option<(i64, i64)>,
    /// Every rejected row, in source order.
    pub issues: Vec<RowIssue>,
    /// True iff any accepted timestamp has sub-second precision, which the
    /// substrate's whole-second coordinate discards (declared, not hidden).
    pub discards_subsecond: bool,
}

impl Preview {
    /// True iff the file produced at least one accepted row and no issues.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.rejected_rows == 0 && self.accepted_rows > 0
    }
}

/// One accepted, fully validated row in canonical form.
struct Accepted {
    /// The canonical record, ready to encode.
    record: CanonicalRecord,
    /// The exact minor-unit amount, for the A1 attestation channel only.
    amount_minor: i64,
    /// Whether the source timestamp carried sub-second precision.
    subsecond: bool,
}

/// Normalize CSV bytes under an explicit plan into a preview + bounded chunks.
///
/// # Parameters
/// - `bytes` — the selected file's bytes, read locally (never uploaded raw).
/// - `plan` — the complete explicit policy.
/// - `namespace` — the session namespace; every chunk is encoded under it.
///
/// # Errors
/// - Any [`IngestPlan::validate`] rejection, before a single row is read.
/// - [`ErrorCode::AccountIdCapacity`] when the declared width is exhausted.
///   Per §6 the whole import stops: IDs must never wrap or widen mid-segment.
/// - Any [`crate::csv`] parse failure (unterminated quote, oversized field,
///   non-UTF-8 bytes).
///
/// Individual bad rows never abort; they become [`RowIssue`] entries.
pub fn normalize(
    bytes: &[u8],
    plan: &IngestPlan,
    namespace: abacus_ingest::namespace::EventNamespace,
) -> IngestResult<Normalization> {
    plan.validate()?;
    let config = plan.to_source_config()?;
    let major_scale = config.major_scale()?;
    let precision = config.currency_precision;
    let threshold = config.threshold_minor_units;
    let ceiling = config.max_amount_minor_units;

    let mut interner = IdentifierInterner::new(plan.identifiers)?;
    let mut issues: Vec<RowIssue> = Vec::new();
    let mut accepted: Vec<Accepted> = Vec::new();
    let mut input_rows = 0u64;
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    let mut subsecond = false;

    iter_rows(bytes, &plan.csv, |row| {
        input_rows += 1;
        // The event id is the stable CSV record ordinal: the position among
        // ACCEPTED rows in file order, 1-based (wire §2 identity_scheme 1).
        let ordinal = accepted.len() as u64 + 1;
        match accept_row(
            &row,
            ordinal,
            plan,
            precision,
            threshold,
            major_scale,
            ceiling,
            namespace,
            &mut interner,
        ) {
            Ok(a) => {
                lo = Some(lo.map_or(a.record.event_time_ms, |m| m.min(a.record.event_time_ms)));
                hi = Some(hi.map_or(a.record.event_time_ms, |m| m.max(a.record.event_time_ms)));
                subsecond |= a.subsecond;
                accepted.push(a);
            }
            Err(e) => issues.push(RowIssue {
                line: row.line,
                code: e.code,
                detail: e.detail,
            }),
        }
        true
    })?;

    // A capacity overflow must abort the whole import, not just reject the row
    // that hit it: the segment's identity space is already exhausted (§6).
    if issues
        .iter()
        .any(|i| i.code == ErrorCode::AccountIdCapacity)
    {
        return Err(IngestError::new(
            ErrorCode::AccountIdCapacity,
            format!(
                "the declared {}-byte account-ID width is exhausted; the segment stops here. \
                 Do not wrap IDs or widen the width mid-segment (Golden Source §6).",
                plan.identifiers.account_id_width
            ),
        ));
    }

    let distinct = interner.len();
    // Encode bounded chunks. Every chunk is a complete segment, so a retry is
    // idempotent and a partial upload is still independently validatable.
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut amounts = crate::attest::AmountLedger::with_capacity(accepted.len());
    for batch in accepted.chunks(plan.records_per_chunk.max(1)) {
        let records: Vec<CanonicalRecord> = batch.iter().map(|a| a.record).collect();
        for a in batch {
            amounts.push(a.amount_minor);
        }
        chunks.push(abacus_ingest::encode_segment(&config, namespace, &records)?);
    }

    Ok(Normalization {
        preview: Preview {
            input_rows,
            accepted_rows: accepted.len() as u64,
            rejected_rows: issues.len() as u64,
            distinct_accounts: distinct,
            account_id_width: plan.identifiers.account_id_width,
            event_time_bounds: lo.zip(hi),
            issues,
            discards_subsecond: subsecond,
        },
        chunks,
        attested_amounts: amounts,
        namespace,
    })
}

/// Validate and convert one row. Any failure is that row's rejection.
#[allow(clippy::too_many_arguments)]
fn accept_row(
    row: &CsvRow,
    ordinal: u64,
    plan: &IngestPlan,
    precision: u8,
    threshold: u64,
    major_scale: u64,
    ceiling: u64,
    namespace: abacus_ingest::namespace::EventNamespace,
    interner: &mut IdentifierInterner,
) -> IngestResult<Accepted> {
    let mapping = &plan.mapping;

    // 1. Timestamp (§7).
    let ts_text = field_at(row, mapping, CanonicalField::Timestamp)?;
    let event_time_ms = timestamp::parse_timestamp(ts_text, plan.timestamp, plan.numbers)?;

    // 2. Identifiers as strings, folded to opaque codes (§6).
    let sender_text = field_at(row, mapping, CanonicalField::Sender)?;
    let receiver_text = field_at(row, mapping, CanonicalField::Receiver)?;
    let sender = interner.intern(sender_text, row.line)?;
    let receiver = interner.intern(receiver_text, row.line)?;

    // 3. Exact amount (§8).
    let amount_text = field_at(row, mapping, CanonicalField::Amount)?;
    let amount_minor = amount::parse_amount(amount_text, plan.amount, precision, ceiling)?;

    // 4. The frozen seven-byte payload (§9) — derived by the pipeline's own
    //    deterministic implementation, never reimplemented here. This is the
    //    shared implementation §9.1 asks for, so the browser and native Rust
    //    agree byte-for-byte by construction.
    let features = FeaturePayload::derive(amount_minor, threshold, major_scale)?;

    // 5. The canonical record. The namespace is attached from the segment-level
    //    decision (§2/§7.2: no per-record namespace field).
    Ok(Accepted {
        record: CanonicalRecord {
            identity: EventIdentity::new(namespace, ordinal),
            event_time_ms,
            sender_id: sender,
            receiver_id: receiver,
            features,
        },
        amount_minor: i64::try_from(amount_minor).map_err(|_| {
            IngestError::new(
                ErrorCode::AmountOrCurrency,
                "the amount exceeds the i64 minor-unit range",
            )
        })?,
        subsecond: abacus_ingest::timestamp::discards_subsecond(event_time_ms),
    })
}

/// The complete normalization result: the preview plus the encoded chunks.
#[derive(Debug)]
pub struct Normalization {
    /// The privacy-safe preview for the UI.
    pub preview: Preview,
    /// One encoded `ABACSEG1` segment per chunk, in upload order.
    ///
    /// These are the exact bytes uploaded; each is a complete, independently
    /// validatable segment, so a retry of any one of them is idempotent (§13).
    pub chunks: Vec<Vec<u8>>,
    /// The exact minor-unit amount of each accepted record, in chunk-then-row
    /// order.
    ///
    /// **Amendment A1 transport channel.** Held only until commit, wrapped in a
    /// clearing guard, and never placed in a record, a chunk, a report or a log.
    pub attested_amounts: crate::attest::AmountLedger,
    /// The namespace every chunk was encoded under (§2: one per segment).
    pub namespace: abacus_ingest::namespace::EventNamespace,
}
