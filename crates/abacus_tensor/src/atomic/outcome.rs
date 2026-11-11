//! Typed operator outcome (Phase-13 decision b).
//!
//! `Determinate` = complete valid result. `Indeterminate` = the computation
//! cannot establish a valid answer under its declared bounds or required
//! inputs. The reason is preserved; the cause reuses the existing error
//! text. An incomplete result is never converted into a valid negative.

/// Why an operator evaluation is indeterminate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpReason {
    /// Checked i64/i128/u64 arithmetic overflowed.
    Overflow,
    /// A capped retrieval truncated the input (`is_truncated`).
    Truncated,
    /// Required baseline / provenance unavailable.
    BaselineUnavailable,
    /// Semantics requested are not supported by the operator.
    UnsupportedSemantics,
    /// A declared bound (vertex / admitted / locator / dict) was hit.
    BoundHit,
}

/// Typed outcome at the operator / evaluation boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome<T> {
    /// Complete, valid result.
    Determinate(T),
    /// No valid answer under declared bounds / inputs.
    Indeterminate(Indeterminate),
}

/// An indeterminate result with reason and reused cause text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Indeterminate {
    /// Machine-readable reason.
    pub reason: OpReason,
    /// Cause text reused from the existing error (never fabricated).
    pub cause: String,
}

impl Indeterminate {
    /// Build a reasoned indeterminate result.
    #[must_use]
    pub fn new(reason: OpReason, cause: impl Into<String>) -> Self {
        Self {
            reason,
            cause: cause.into(),
        }
    }
}

impl<T> Outcome<T> {
    /// Unwrap the determinate value, if present.
    #[must_use]
    pub fn determinate(self) -> Option<T> {
        match self {
            Self::Determinate(v) => Some(v),
            Self::Indeterminate(_) => None,
        }
    }

    /// True iff indeterminate.
    #[must_use]
    pub fn is_indeterminate(&self) -> bool {
        matches!(self, Self::Indeterminate(_))
    }
}
