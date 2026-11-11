//! Error alphabet `E = E_rng ⊔ E_parse ⊔ E_dec ⊔ E_io` (§0.3) — the lift
//! target of `R5` (fail-closed error propagation, P12).
//!
//! Infrastructure only: the error alphabet is declared by the derivation
//! (§0.3, §A.1 `a_{02}`–`a_{05}` lifted codomains) and carries no tensor
//! semantics. This is the single realization of `E` for every subsystem.

use std::fmt;

/// Errors for deterministic builder / codec paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// `E_parse`: CSV / text input malformed (§0.3, K7/K11).
    Parse(String),
    /// `E_rng`: exact result outside `V`, or undefined `a_{02}`–`a_{05}` (P2).
    Range(String),
    /// `E_dec`: binary/UTF-8 image malformed (§A.1 `a_{13}.2`, `a_{18}.4`).
    Decode(String),
    /// `E_io`: `a_{19}` external-world failure (K8).
    Io(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(s) => write!(f, "parse error: {s}"),
            Self::Range(s) => write!(f, "range error: {s}"),
            Self::Decode(s) => write!(f, "decode error: {s}"),
            Self::Io(s) => write!(f, "io error: {s}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<std::io::Error> for TensorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
