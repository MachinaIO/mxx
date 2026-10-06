//! Noise-simulation settings shared by Program audits and the decoder.
//!
//! The concrete noise terms are owned by the backend noise evaluators; this
//! module only selects the noise model and fixes the average-case tail
//! factor.

use crate::noise_utils::NonnegativeRational;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseModelKind {
    WorstCase,
    AverageCase,
}

impl NoiseModelKind {
    /// Application-wide simulation selection. Average acceptance remains
    /// conditional on the explicitly selected coefficient and tail models.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("MXX_EXPONENT_LUT_DECODER_NOISE_MODEL")
            .as_deref()
            .unwrap_or("average-case")
        {
            "worst-case" => Ok(Self::WorstCase),
            "average-case" => Ok(Self::AverageCase),
            _ => {
                Err("MXX_EXPONENT_LUT_DECODER_NOISE_MODEL must be worst-case or average-case"
                    .into())
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageCaseConfig {
    pub failure_exponent: u32,
    pub tail_correction_bits: u32,
    pub allow_average_acceptance: bool,
}

impl Default for AverageCaseConfig {
    fn default() -> Self {
        Self { failure_exponent: 100, tail_correction_bits: 0, allow_average_acceptance: false }
    }
}

impl AverageCaseConfig {
    /// Exact `z^2 = 2 * (693148 / 1000000) * (lambda + ceil(log2(D)) + 1)`
    /// for `D` joint events. Callers derive the event count from the executed
    /// workload, including its structural multiplicities.
    pub fn z_squared_for_events(
        &self,
        event_count: &BigUint,
    ) -> Result<NonnegativeRational, NoiseSimulationError> {
        let events = self
            .failure_exponent
            .checked_add(ceil_log2(event_count)?)
            .and_then(|value| value.checked_add(1))
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        Ok(NonnegativeRational::new(
            BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(events),
            BigUint::from(1_000_000u32),
        )?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum NoiseSimulationError {
    #[error(transparent)]
    Arithmetic(#[from] crate::noise_utils::NoiseArithmeticError),
    #[error("the joint event count is empty")]
    EmptyEvents,
    #[error("integer conversion overflow")]
    IntegerConversion,
}

fn ceil_log2(value: &BigUint) -> Result<u32, NoiseSimulationError> {
    if value.is_zero() {
        return Err(NoiseSimulationError::EmptyEvents);
    }
    let bits = value.bits();
    let exact_power = value == &(BigUint::one() << (bits - 1));
    u32::try_from(if exact_power { bits - 1 } else { bits })
        .map_err(|_| NoiseSimulationError::IntegerConversion)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_tail_factor_uses_the_ceiling_log_of_the_event_count() {
        let config = AverageCaseConfig::default();
        let at = |events: u32| config.z_squared_for_events(&BigUint::from(events)).unwrap();
        // lambda + ceil(log2 D) + 1 for D = 1, 4 and 5.
        for (events, bits) in [(1u32, 101u32), (4, 103), (5, 104)] {
            assert_eq!(
                at(events),
                NonnegativeRational::new(
                    BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(bits),
                    BigUint::from(1_000_000u32),
                )
                .unwrap()
            );
        }
        assert!(config.z_squared_for_events(&BigUint::zero()).is_err());
    }
}
