//! Pipeline error type: fail-closed `R_5` mapping over the canonical
//! `TensorError` alphabet (`E_rng / E_parse / E_dec / E_io`).

use abacus_tensor::TensorError;

/// Phase-1 pipeline failures. Every variant rejects the event (fail-closed);
/// none partially commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// Canonical tensor/codec failure, propagated unchanged.
    Tensor(TensorError),
    /// Loader cross-artifact mismatch (fail closed, never auto-repair).
    Mismatch(String),
    /// Event validation failure (malformed normalized event).
    Invalid(String),
    /// StreamingIngestionDemo/duplicate rejected by the identity contract.
    Duplicate(String),
    /// I/O failure reading configured artifact paths.
    Io(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(e) => write!(f, "{e}"),
            Self::Mismatch(s) => write!(f, "artifact mismatch: {s}"),
            Self::Invalid(s) => write!(f, "invalid event: {s}"),
            Self::Duplicate(s) => write!(f, "duplicate event: {s}"),
            Self::Io(s) => write!(f, "io error: {s}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<TensorError> for PipelineError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
