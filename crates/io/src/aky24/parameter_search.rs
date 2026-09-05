//! Automatic correctness and lattice-security parameter selection for AKY24 iO.

use super::{
    cascade::{Aky24CascadeCompiler, Aky24CascadeGraphError},
    config::{Aky24ConfigError, Aky24IoConfig},
    noise::{Aky24IoNoiseError, Aky24IoNoiseSimulation, simulate_aky24_io_noise},
};
use mxx_ir_core::ParamEnv;
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, process::Command, sync::Arc};
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct Aky24IoParameterSearch {
    pub template: Aky24IoConfig,
    pub min_crt_depth: usize,
    pub initial_max_crt_depth: usize,
    pub min_log_ring_dimension: usize,
    pub max_log_ring_dimension: usize,
    pub crt_modulus_bits: usize,
    pub security_bits: usize,
}

pub struct Aky24IoSelectedParameters {
    pub parameters: DCRTPolyParams,
    pub compiler: Aky24CascadeCompiler,
    pub crt_depth: usize,
    pub log_ring_dimension: usize,
    pub ring_dimension: u32,
    pub modulus: BigUint,
    pub modulus_bits: usize,
    pub achieved_security_bits: u64,
    pub simulation: Aky24IoNoiseSimulation,
}

#[derive(Debug, Error)]
pub enum Aky24IoParameterSearchError {
    #[error("the AKY24 iO parameter-search range is invalid")]
    InvalidRange,
    #[error("AKY24 iO CRT-depth growth overflowed")]
    DepthOverflow,
    #[error("no ring dimension in the configured range meets the security target")]
    NoSecureRingDimension,
    #[error("no correctness-valid AKY24 iO parameters were found")]
    NoCorrectParameters,
    #[error(transparent)]
    Config(#[from] Aky24ConfigError),
    #[error(transparent)]
    Compile(#[from] Aky24CascadeGraphError),
    #[error(transparent)]
    Noise(#[from] Aky24IoNoiseError),
    #[error("lattice-estimator-cli could not be started: {0}")]
    EstimatorIo(#[from] std::io::Error),
    #[error("lattice-estimator-cli failed: {0}")]
    EstimatorFailure(String),
    #[error("lattice-estimator-cli returned an invalid security estimate: {0}")]
    EstimatorOutput(String),
}

struct Candidate {
    selected: Aky24IoSelectedParameters,
    correct: bool,
}

impl Aky24IoParameterSearch {
    pub fn search(&self) -> Result<Aky24IoSelectedParameters, Aky24IoParameterSearchError> {
        self.search_with_security_estimator(lattice_security_bits)
    }

    pub fn search_with_security_estimator<F>(
        &self,
        mut estimate_security: F,
    ) -> Result<Aky24IoSelectedParameters, Aky24IoParameterSearchError>
    where
        F: FnMut(&DCRTPolyParams, f64, f64) -> Result<u64, Aky24IoParameterSearchError>,
    {
        self.validate()?;
        let mut cache = BTreeMap::<usize, Candidate>::new();
        let mut evaluate = |depth: usize| -> Result<bool, Aky24IoParameterSearchError> {
            if !cache.contains_key(&depth) {
                let (log_ring_dimension, achieved_security_bits) =
                    self.select_ring_dimension(depth, &mut estimate_security)?;
                cache.insert(
                    depth,
                    self.evaluate_candidate(depth, log_ring_dimension, achieved_security_bits)?,
                );
            }
            Ok(cache.get(&depth).expect("cached AKY24 candidate").correct)
        };

        let mut upper = self.initial_max_crt_depth;
        while !evaluate(upper)? {
            upper = upper.checked_mul(2).ok_or(Aky24IoParameterSearchError::DepthOverflow)?;
        }
        let mut low = self.min_crt_depth;
        let mut high = upper;
        let mut best = upper;
        while low <= high {
            let depth = low + (high - low) / 2;
            if evaluate(depth)? {
                best = depth;
                if depth == self.min_crt_depth {
                    break;
                }
                high = depth - 1;
            } else {
                low = depth + 1;
            }
        }
        cache
            .remove(&best)
            .map(|candidate| candidate.selected)
            .ok_or(Aky24IoParameterSearchError::NoCorrectParameters)
    }

    fn select_ring_dimension<F>(
        &self,
        depth: usize,
        estimate_security: &mut F,
    ) -> Result<(usize, u64), Aky24IoParameterSearchError>
    where
        F: FnMut(&DCRTPolyParams, f64, f64) -> Result<u64, Aky24IoParameterSearchError>,
    {
        let (secret_sigma, error_sigmas) = self.security_sigmas()?;
        let mut low = self.min_log_ring_dimension;
        let mut high = self.max_log_ring_dimension;
        let mut selected = None;
        while low <= high {
            let log_ring_dimension = low + (high - low) / 2;
            let ring_dimension = 1u32
                .checked_shl(log_ring_dimension as u32)
                .ok_or(Aky24IoParameterSearchError::InvalidRange)?;
            let parameters = DCRTPolyParams::new(ring_dimension, depth, self.crt_modulus_bits, 1);
            let achieved = error_sigmas.iter().try_fold(u64::MAX, |minimum, error_sigma| {
                estimate_security(&parameters, secret_sigma, *error_sigma)
                    .map(|estimate| minimum.min(estimate))
            })?;
            if achieved >= self.security_bits as u64 {
                selected = Some((log_ring_dimension, achieved));
                if log_ring_dimension == 0 {
                    break;
                }
                high = log_ring_dimension - 1;
            } else {
                low = log_ring_dimension + 1;
            }
        }
        selected.ok_or(Aky24IoParameterSearchError::NoSecureRingDimension)
    }

    fn evaluate_candidate(
        &self,
        depth: usize,
        log_ring_dimension: usize,
        achieved_security_bits: u64,
    ) -> Result<Candidate, Aky24IoParameterSearchError> {
        let ring_dimension = 1u32
            .checked_shl(log_ring_dimension as u32)
            .ok_or(Aky24IoParameterSearchError::InvalidRange)?;
        // AKY24's attribute encoding is binary, so the DCRT gadget base is
        // fixed to two rather than exposed as an independent search knob.
        let parameters = DCRTPolyParams::new(ring_dimension, depth, self.crt_modulus_bits, 1);
        let modulus: Arc<BigUint> = parameters.modulus();
        let mut config = self.template.clone();
        config.modulus = BigInt::from(modulus.as_ref().clone());
        config.ring_dimension = ring_dimension as usize;
        config.gadget_base = BigInt::from(2);
        config.digit_count = parameters.modulus_digits();
        let compiler = Aky24CascadeCompiler::new(config)?;
        // Both alternatives at a position are generated by the same circuit
        // with identical source metadata. The all-zero vector is therefore a
        // canonical exact graph representative, not a hand-written formula.
        let canonical_input = vec![false; compiler.config().input_size];
        let simulation = simulate_aky24_io_noise(&compiler, &canonical_input)?;
        let correct = simulation.within_threshold;
        Ok(Candidate {
            selected: Aky24IoSelectedParameters {
                parameters,
                compiler,
                crt_depth: depth,
                log_ring_dimension,
                ring_dimension,
                modulus: modulus.as_ref().clone(),
                modulus_bits: modulus.bits() as usize,
                achieved_security_bits,
                simulation,
            },
            correct,
        })
    }

    fn validate(&self) -> Result<(), Aky24IoParameterSearchError> {
        self.template.validate()?;
        if self.min_crt_depth == 0 ||
            self.min_crt_depth > self.initial_max_crt_depth ||
            self.min_log_ring_dimension > self.max_log_ring_dimension ||
            self.max_log_ring_dimension >= u32::BITS as usize ||
            self.crt_modulus_bits == 0 ||
            self.security_bits == 0
        {
            return Err(Aky24IoParameterSearchError::InvalidRange);
        }
        Ok(())
    }

    fn security_sigmas(&self) -> Result<(f64, Vec<f64>), Aky24IoParameterSearchError> {
        let bindings = ParamEnv::default();
        let evaluate = |sigma: &mxx_ir_core::RealExpr| {
            sigma.evaluate_f64(&bindings).map_err(|_| Aky24IoParameterSearchError::InvalidRange)
        };
        let secret_sigma = evaluate(&self.template.secret_sigma)?;
        let mut error_sigmas = [
            evaluate(&self.template.b_error_sigma)?,
            evaluate(&self.template.fhe_error_sigma)?,
            evaluate(&self.template.attribute_error_sigma)?,
        ]
        .into_iter()
        .collect::<Vec<_>>();
        error_sigmas.sort_by(f64::total_cmp);
        error_sigmas.dedup_by(|left, right| left.to_bits() == right.to_bits());
        Ok((secret_sigma, error_sigmas))
    }
}

fn lattice_security_bits(
    parameters: &DCRTPolyParams,
    secret_sigma: f64,
    error_sigma: f64,
) -> Result<u64, Aky24IoParameterSearchError> {
    let modulus: Arc<BigUint> = parameters.modulus();
    let secret = discrete_gaussian_distribution_json(secret_sigma);
    let error = discrete_gaussian_distribution_json(error_sigma);
    let output = Command::new("lattice-estimator-cli")
        .arg(parameters.ring_dimension().to_string())
        .arg(modulus.to_string())
        .arg("--s-dist")
        .arg(secret)
        .arg("--e-dist")
        .arg(error)
        .output()?;
    if !output.status.success() {
        return Err(Aky24IoParameterSearchError::EstimatorFailure(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value = stdout.lines().rev().find(|line| !line.trim().is_empty()).unwrap_or("").trim();
    value.parse().map_err(|_| Aky24IoParameterSearchError::EstimatorOutput(stdout.into_owned()))
}

fn discrete_gaussian_distribution_json(sigma: f64) -> String {
    format!(r#"{{"name":"DiscreteGaussian","stddev":{sigma}}}"#)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aky24::Aky24GoldreichPrf;
    use mxx_ir_core::RealExpr;

    fn search() -> Aky24IoParameterSearch {
        Aky24IoParameterSearch {
            template: Aky24IoConfig {
                modulus: BigInt::from(257),
                ring_dimension: 8,
                input_size: 8,
                gadget_base: BigInt::from(2),
                digit_count: 9,
                preimage_max_coefficient_bound: 1_000_000.into(),
                modulus_split: BigInt::from(1),
                trapdoor_sigma: RealExpr::from_integer(5),
                secret_sigma: RealExpr::from_integer(4),
                b_error_sigma: RealExpr::from_integer(3),
                fhe_error_sigma: RealExpr::from_integer(1),
                attribute_error_sigma: RealExpr::from_integer(2),
                security_parameter_bits: 128,
                cascade_randomness_bits: 128,
                gaussian_sample_bits: 16,
                uniform_statistical_bits: 16,
                function: Aky24GoldreichPrf { output_bits: 2, graph_seed: [9; 32] },
            },
            min_crt_depth: 1,
            initial_max_crt_depth: 1,
            min_log_ring_dimension: 3,
            max_log_ring_dimension: 3,
            crt_modulus_bits: 20,
            security_bits: 50,
        }
    }

    #[test]
    fn estimator_distribution_argument_is_valid_unescaped_json() {
        assert_eq!(
            discrete_gaussian_distribution_json(4.5),
            r#"{"name":"DiscreteGaussian","stddev":4.5}"#
        );
    }

    #[test]
    fn security_estimator_uses_the_actual_gaussian_secret_and_every_error_distribution() {
        let search = search();
        let mut observed = Vec::new();
        let (log_ring_dimension, achieved) = search
            .select_ring_dimension(1, &mut |_, secret_sigma, error_sigma| {
                observed.push((secret_sigma, error_sigma));
                Ok(match error_sigma as u64 {
                    1 => 90,
                    2 => 70,
                    3 => 60,
                    _ => unreachable!("unexpected protocol error sigma"),
                })
            })
            .unwrap();
        assert_eq!(log_ring_dimension, 3);
        assert_eq!(achieved, 60);
        assert_eq!(observed, vec![(4.0, 1.0), (4.0, 2.0), (4.0, 3.0)]);
    }
}
