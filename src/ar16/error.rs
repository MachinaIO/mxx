use crate::circuit::gate::GateId;
use thiserror::Error;

use super::Level;

/// Errors that can arise while running the AR16 evaluation algorithms.
#[derive(Debug, Error)]
pub enum Ar16Error {
    #[error("missing AR16 advice for level {level:?} (gate {gate:?})")]
    MissingAdvice { level: Level, gate: Option<GateId> },

    #[error("AR16 input for gate {gate:?} was not provided")]
    MissingInput { gate: GateId },

    #[error("unsupported gate {gate:?} in AR16 evaluation")]
    UnsupportedGate { gate: GateId },

    #[error("level mismatch: expected {expected:?}, got {actual:?}")]
    LevelMismatch { expected: Level, actual: Level },
}

impl Ar16Error {
    pub fn missing_advice(level: Level, gate: Option<GateId>) -> Self {
        Ar16Error::MissingAdvice { level, gate }
    }

    pub fn missing_input(gate: GateId) -> Self {
        Ar16Error::MissingInput { gate }
    }

    pub fn level_mismatch(expected: Level, actual: Level) -> Self {
        Ar16Error::LevelMismatch { expected, actual }
    }

    pub fn unsupported_gate(gate: GateId) -> Self {
        Ar16Error::UnsupportedGate { gate }
    }
}
