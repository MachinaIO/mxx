use crate::circuit::gate::GateId;
use std::fmt::{Display, Formatter};

/// Error type returned by AR16 evaluation helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ar16Error {
    /// Advice encodings required for the current level or gate are missing.
    MissingAdvice { level: usize, gate: Option<GateId> },
    /// Circuit metadata (e.g., modulus tower or gate layering) does not meet the assumptions.
    LayerMismatch { message: &'static str },
    /// The circuit contains operations that are not supported by the current evaluator.
    UnsupportedGate { gate: GateId },
    /// Placeholder variant while the full algorithms are being implemented.
    Unimplemented(&'static str),
}

impl Display for Ar16Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Ar16Error::MissingAdvice { level, gate } => {
                write!(f, "missing advice for level {level}")?;
                if let Some(id) = gate {
                    write!(f, " (gate {id})")?;
                }
                Ok(())
            }
            Ar16Error::LayerMismatch { message } => write!(f, "layer mismatch: {message}"),
            Ar16Error::UnsupportedGate { gate } => write!(f, "unsupported gate {gate}"),
            Ar16Error::Unimplemented(msg) => write!(f, "unimplemented: {msg}"),
        }
    }
}

impl std::error::Error for Ar16Error {}
