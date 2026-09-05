//! Automatic correctness/security parameter selection for Diamond iO.

use super::{
    DiamondIoCompiler, DiamondIoConfig, DiamondIoFunction, DiamondIoNoiseError,
    DiamondIoNoiseSimulation, simulate_diamond_io_noise,
};
use mxx_gadgets::{
    circuit::PolyCircuit,
    circuit_gadgets::{
        arith::NestedRnsPolyContext, fhe::ring_gsw_nested_rns::NestedRnsRingGswContext,
    },
};
use mxx_ir_core::RealExpr;
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, process::Command, sync::Arc};
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct DiamondIoParameterSearch {
    pub template: DiamondIoConfig,
    pub min_crt_depth: usize,
    pub initial_max_crt_depth: usize,
    pub min_log_ring_dimension: usize,
    pub max_log_ring_dimension: usize,
    pub crt_modulus_bits: usize,
    pub gadget_base_bits: u32,
    pub security_bits: usize,
    pub error_sigma: f64,
    pub native_ring_gsw_error_sigma: f64,
    pub nested_p_moduli_bits: usize,
    pub nested_max_unreduced_muls: usize,
    pub nested_scale: u64,
}

#[derive(Clone)]
pub struct DiamondIoSelectedParameters {
    pub parameters: DCRTPolyParams,
    pub compiler: DiamondIoCompiler<DCRTPoly>,
    pub crt_depth: usize,
    pub log_ring_dimension: usize,
    pub ring_dimension: u32,
    pub modulus: BigUint,
    pub modulus_bits: usize,
    pub achieved_security_bits: u64,
    pub simulation: DiamondIoNoiseSimulation,
}

#[derive(Debug, Error)]
pub enum DiamondIoParameterSearchError {
    #[error("the Diamond iO parameter-search range is invalid")]
    InvalidRange,
    #[error("Diamond iO CRT-depth growth overflowed")]
    DepthOverflow,
    #[error("no ring dimension in the configured range meets the security target")]
    NoSecureRingDimension,
    #[error("no correctness-valid Diamond iO parameters were found")]
    NoCorrectParameters,
    #[error("Diamond iO configuration failed: {0}")]
    Config(String),
    #[error(transparent)]
    Noise(#[from] DiamondIoNoiseError),
    #[error("lattice-estimator-cli could not be started: {0}")]
    EstimatorIo(#[from] std::io::Error),
    #[error("lattice-estimator-cli failed: {0}")]
    EstimatorFailure(String),
    #[error("lattice-estimator-cli returned an invalid security estimate: {0}")]
    EstimatorOutput(String),
}

struct Candidate {
    selected: DiamondIoSelectedParameters,
    correct: bool,
}

impl DiamondIoParameterSearch {
    pub fn search(
        &self,
        function: &DiamondIoFunction,
    ) -> Result<DiamondIoSelectedParameters, DiamondIoParameterSearchError> {
        self.search_with_security_estimator(function, lattice_security_bits)
    }

    pub fn search_with_security_estimator<F>(
        &self,
        function: &DiamondIoFunction,
        mut estimate_security: F,
    ) -> Result<DiamondIoSelectedParameters, DiamondIoParameterSearchError>
    where
        F: FnMut(&DCRTPolyParams, f64) -> Result<u64, DiamondIoParameterSearchError>,
    {
        self.validate()?;
        let mut cache = BTreeMap::<usize, Candidate>::new();
        let mut evaluate = |depth: usize| -> Result<bool, DiamondIoParameterSearchError> {
            if !cache.contains_key(&depth) {
                let (log_ring_dimension, achieved_security_bits) =
                    self.select_ring_dimension(depth, &mut estimate_security)?;
                let candidate = self.evaluate_candidate(
                    function,
                    depth,
                    log_ring_dimension,
                    achieved_security_bits,
                )?;
                cache.insert(depth, candidate);
            }
            Ok(cache.get(&depth).expect("cached candidate").correct)
        };

        let mut upper = self.initial_max_crt_depth;
        while !evaluate(upper)? {
            upper = upper.checked_mul(2).ok_or(DiamondIoParameterSearchError::DepthOverflow)?;
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
            .ok_or(DiamondIoParameterSearchError::NoCorrectParameters)
    }

    fn select_ring_dimension<F>(
        &self,
        depth: usize,
        estimate_security: &mut F,
    ) -> Result<(usize, u64), DiamondIoParameterSearchError>
    where
        F: FnMut(&DCRTPolyParams, f64) -> Result<u64, DiamondIoParameterSearchError>,
    {
        let mut low = self.min_log_ring_dimension;
        let mut high = self.max_log_ring_dimension;
        let mut selected = None;
        while low <= high {
            let log_ring_dimension = low + (high - low) / 2;
            let ring_dimension = 1u32
                .checked_shl(log_ring_dimension as u32)
                .ok_or(DiamondIoParameterSearchError::InvalidRange)?;
            let parameters = DCRTPolyParams::new(
                ring_dimension,
                depth,
                self.crt_modulus_bits,
                self.gadget_base_bits,
            );
            let achieved = estimate_security(&parameters, self.error_sigma)?
                .min(estimate_security(&parameters, self.native_ring_gsw_error_sigma)?);
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
        selected.ok_or(DiamondIoParameterSearchError::NoSecureRingDimension)
    }

    fn evaluate_candidate(
        &self,
        function: &DiamondIoFunction,
        depth: usize,
        log_ring_dimension: usize,
        achieved_security_bits: u64,
    ) -> Result<Candidate, DiamondIoParameterSearchError> {
        let ring_dimension = 1u32
            .checked_shl(log_ring_dimension as u32)
            .ok_or(DiamondIoParameterSearchError::InvalidRange)?;
        let parameters = DCRTPolyParams::new(
            ring_dimension,
            depth,
            self.crt_modulus_bits,
            self.gadget_base_bits,
        );
        let compiler = self.compiler_for(&parameters, function)?;
        let simulation = simulate_diamond_io_noise(&compiler, function)?;
        let correct = simulation.within_threshold;
        let modulus: Arc<BigUint> = parameters.modulus();
        Ok(Candidate {
            selected: DiamondIoSelectedParameters {
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

    fn compiler_for(
        &self,
        parameters: &DCRTPolyParams,
        function: &DiamondIoFunction,
    ) -> Result<DiamondIoCompiler<DCRTPoly>, DiamondIoParameterSearchError> {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let depth = parameters.to_crt().2;
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            parameters,
            self.nested_p_moduli_bits,
            self.nested_max_unreduced_muls,
            self.nested_scale,
            false,
            Some(depth),
        ));
        let ring_gsw = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            parameters,
            parameters.ring_dimension() as usize,
            nested_rns,
            Some(depth),
            Some(0),
        ));
        let modulus: Arc<BigUint> = parameters.modulus();
        let (crt_moduli, _, _) = parameters.to_crt();
        let mut config = self.template.clone();
        config.modulus = BigInt::from(modulus.as_ref().clone());
        config.ring_dimension = parameters.ring_dimension() as usize;
        config.gadget_base = BigInt::from(1u64 << parameters.base_bits());
        config.digit_count = parameters.modulus_digits();
        config.error_sigma = RealExpr::from_f64_exact(self.error_sigma)
            .map_err(|error| DiamondIoParameterSearchError::Config(error.to_string()))?;
        config.ring_gsw_public_key_error_sigma = Some(
            RealExpr::from_f64_exact(self.native_ring_gsw_error_sigma)
                .map_err(|error| DiamondIoParameterSearchError::Config(error.to_string()))?,
        );
        config.ring_gsw_width = ring_gsw.width();
        config.refresh_crt_scale_factors = crt_moduli
            .iter()
            .map(|modulus_i| BigInt::from(modulus.as_ref() / *modulus_i).into())
            .collect();
        config.refresh_crt_plaintext_moduli =
            crt_moduli.iter().map(|value| BigInt::from(*value).into()).collect();
        config.refresh_reconstruction_coefficients =
            parameters.reconst_coeffs().into_iter().map(BigInt::from).map(Into::into).collect();
        config.refresh_decoder_public_columns = 2 * (parameters.modulus_digits() + 2);
        config.seed_bits = config.seed_bits.max(
            config
                .minimum_goldreich_seed_bits(function)
                .map_err(|error| DiamondIoParameterSearchError::Config(error.to_string()))?,
        );
        DiamondIoCompiler::new(config, ring_gsw)
            .map_err(|error| DiamondIoParameterSearchError::Config(error.to_string()))
    }

    fn validate(&self) -> Result<(), DiamondIoParameterSearchError> {
        if self.min_crt_depth == 0 ||
            self.min_crt_depth > self.initial_max_crt_depth ||
            self.min_log_ring_dimension > self.max_log_ring_dimension ||
            self.max_log_ring_dimension >= u32::BITS as usize ||
            self.crt_modulus_bits == 0 ||
            self.gadget_base_bits >= u64::BITS ||
            self.security_bits == 0 ||
            !self.error_sigma.is_finite() ||
            self.error_sigma <= 0.0 ||
            !self.native_ring_gsw_error_sigma.is_finite() ||
            self.native_ring_gsw_error_sigma <= 0.0 ||
            self.nested_p_moduli_bits == 0 ||
            self.nested_max_unreduced_muls == 0 ||
            self.nested_scale == 0
        {
            return Err(DiamondIoParameterSearchError::InvalidRange);
        }
        Ok(())
    }
}

fn lattice_security_bits(
    parameters: &DCRTPolyParams,
    error_sigma: f64,
) -> Result<u64, DiamondIoParameterSearchError> {
    let modulus: Arc<BigUint> = parameters.modulus();
    let secret = ternary_distribution_json();
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
        return Err(DiamondIoParameterSearchError::EstimatorFailure(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value = stdout.lines().rev().find(|line| !line.trim().is_empty()).unwrap_or("").trim();
    value.parse().map_err(|_| DiamondIoParameterSearchError::EstimatorOutput(stdout.into_owned()))
}

fn ternary_distribution_json() -> &'static str {
    r#"{"name":"Ternary"}"#
}

fn discrete_gaussian_distribution_json(sigma: f64) -> String {
    format!(r#"{{"name":"DiscreteGaussian","stddev":{sigma}}}"#)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_distribution_arguments_are_valid_unescaped_json() {
        assert_eq!(ternary_distribution_json(), r#"{"name":"Ternary"}"#);
        assert_eq!(
            discrete_gaussian_distribution_json(4.5),
            r#"{"name":"DiscreteGaussian","stddev":4.5}"#
        );
    }
}
