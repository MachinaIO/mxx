use super::{
    DiamondWeCompiler, DiamondWeConfig, default_error_max_coefficient_bound,
    default_preimage_max_coefficient_bound,
};
use mxx_gadgets::circuit::{BooleanCircuitError, BooleanCircuitShape};
use mxx_ir_core::{ParamEnv, RealExpr};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, process::Command, sync::Arc};
use thiserror::Error;
use tracing::info;

#[derive(Clone, Debug)]
pub struct DiamondParameterSearch {
    pub shape: BooleanCircuitShape,
    pub min_crt_depth: usize,
    pub initial_max_crt_depth: usize,
    pub min_log_ring_dimension: usize,
    pub max_log_ring_dimension: usize,
    pub crt_modulus_bits: usize,
    pub gadget_base_bits: u32,
    pub security_bits: usize,
    pub input_count: usize,
    pub digit_base: usize,
    pub batch_bits: usize,
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub bgg_tag: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DiamondSelectedParameters {
    pub parameters: DCRTPolyParams,
    pub compiler: DiamondWeCompiler,
    pub crt_depth: usize,
    pub log_ring_dimension: usize,
    pub ring_dimension: u32,
    pub modulus: BigUint,
    pub modulus_bits: usize,
    pub achieved_security_bits: u64,
}

#[derive(Debug, Error)]
pub enum DiamondParameterSearchError {
    #[error("the Diamond parameter-search range is invalid")]
    InvalidRange,
    #[error(transparent)]
    Shape(#[from] BooleanCircuitError),
    #[error("Diamond CRT-depth growth overflowed")]
    DepthOverflow,
    #[error("no ring dimension in the configured range meets the security target")]
    NoSecureRingDimension,
    #[error("no correctness-valid Diamond parameters were found")]
    NoCorrectParameters,
    #[error("lattice-estimator-cli could not be started: {0}")]
    EstimatorIo(#[from] std::io::Error),
    #[error("lattice-estimator-cli failed: {0}")]
    EstimatorFailure(String),
    #[error("lattice-estimator-cli returned an invalid security estimate: {0}")]
    EstimatorOutput(String),
    #[error("a Diamond search parameter cannot be represented exactly")]
    Expression,
    #[error("the selected Diamond compiler configuration is invalid: {0}")]
    Config(String),
    #[error("the Lean Diamond checker failed: {0}")]
    CheckerInfrastructure(String),
}

struct Candidate {
    selected: DiamondSelectedParameters,
    correct: bool,
}

impl DiamondParameterSearch {
    pub fn search(&self) -> Result<DiamondSelectedParameters, DiamondParameterSearchError> {
        self.search_with_security_estimator(lattice_security_bits)
    }

    pub fn search_with_security_estimator<F>(
        &self,
        mut estimate_security: F,
    ) -> Result<DiamondSelectedParameters, DiamondParameterSearchError>
    where
        F: FnMut(&DCRTPolyParams, f64) -> Result<u64, DiamondParameterSearchError>,
    {
        self.validate()?;
        let mut cache = BTreeMap::<usize, Candidate>::new();
        let mut evaluate = |crt_depth: usize| -> Result<bool, DiamondParameterSearchError> {
            if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(crt_depth) {
                let (log_ring_dimension, achieved) =
                    self.select_ring_dimension(crt_depth, &mut estimate_security)?;
                entry.insert(self.evaluate_candidate(crt_depth, log_ring_dimension, achieved)?);
            }
            Ok(cache.get(&crt_depth).expect("inserted candidate").correct)
        };
        let mut upper = self.initial_max_crt_depth;
        while !evaluate(upper)? {
            upper = upper.checked_mul(2).ok_or(DiamondParameterSearchError::DepthOverflow)?;
            info!(crt_depth = upper, "expanding Diamond WE CRT-depth search upper bound");
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
            .ok_or(DiamondParameterSearchError::NoCorrectParameters)
    }

    fn select_ring_dimension<F>(
        &self,
        crt_depth: usize,
        estimate_security: &mut F,
    ) -> Result<(usize, u64), DiamondParameterSearchError>
    where
        F: FnMut(&DCRTPolyParams, f64) -> Result<u64, DiamondParameterSearchError>,
    {
        let mut low = self.min_log_ring_dimension;
        let mut high = self.max_log_ring_dimension;
        let mut selected = None;
        while low <= high {
            let log = low + (high - low) / 2;
            let ring_dimension =
                1u32.checked_shl(log as u32).ok_or(DiamondParameterSearchError::InvalidRange)?;
            let parameters = DCRTPolyParams::new(
                ring_dimension,
                crt_depth,
                self.crt_modulus_bits,
                self.gadget_base_bits,
            );
            let achieved = estimate_security(&parameters, self.error_sigma)?;
            if achieved >= self.security_bits as u64 {
                selected = Some((log, achieved));
                if log == 0 {
                    break;
                }
                high = log - 1;
            } else {
                low = log + 1;
            }
        }
        selected.ok_or(DiamondParameterSearchError::NoSecureRingDimension)
    }

    fn evaluate_candidate(
        &self,
        crt_depth: usize,
        log_ring_dimension: usize,
        achieved_security_bits: u64,
    ) -> Result<Candidate, DiamondParameterSearchError> {
        let ring_dimension = 1u32
            .checked_shl(log_ring_dimension as u32)
            .ok_or(DiamondParameterSearchError::InvalidRange)?;
        let parameters = DCRTPolyParams::new(
            ring_dimension,
            crt_depth,
            self.crt_modulus_bits,
            self.gadget_base_bits,
        );
        let modulus: Arc<BigUint> = parameters.modulus();
        let error_sigma = RealExpr::from_f64_exact(self.error_sigma)
            .map_err(|_| DiamondParameterSearchError::Expression)?;
        let trapdoor_sigma = RealExpr::from_f64_exact(self.trapdoor_sigma)
            .map_err(|_| DiamondParameterSearchError::Expression)?;
        let error_max_coefficient_bound = default_error_max_coefficient_bound(&error_sigma)
            .map_err(|_| DiamondParameterSearchError::Expression)?;
        let preimage_max_coefficient_bound = default_preimage_max_coefficient_bound(
            &trapdoor_sigma,
            ring_dimension as usize,
            parameters.modulus_digits(),
            &BigInt::from(1u64 << self.gadget_base_bits),
        )
        .map_err(|_| DiamondParameterSearchError::Expression)?;
        let compiler = DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: BigInt::from(modulus.as_ref().clone()),
                ring_dimension: ring_dimension as usize,
                input_count: self.input_count,
                digit_base: self.digit_base,
                batch_bits: self.batch_bits,
                gadget_base: BigInt::from(1u64 << self.gadget_base_bits),
                digit_count: parameters.modulus_digits(),
                trapdoor_sigma,
                error_sigma,
                error_max_coefficient_bound,
                preimage_max_coefficient_bound,
                bgg_tag: self.bgg_tag.clone(),
            },
            self.shape.clone(),
        )
        .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?;
        let correct = lean_checker_accepts(&compiler)?;
        Ok(Candidate {
            selected: DiamondSelectedParameters {
                parameters,
                compiler,
                crt_depth,
                log_ring_dimension,
                ring_dimension,
                modulus: modulus.as_ref().clone(),
                modulus_bits: modulus.bits() as usize,
                achieved_security_bits,
            },
            correct,
        })
    }

    fn validate(&self) -> Result<(), DiamondParameterSearchError> {
        self.shape.validate()?;
        let witness_width = self
            .input_count
            .checked_mul(self.batch_bits)
            .ok_or(DiamondParameterSearchError::InvalidRange)?;
        if self.shape.witness_width != witness_width ||
            self.min_crt_depth == 0 ||
            self.min_crt_depth > self.initial_max_crt_depth ||
            self.min_log_ring_dimension > self.max_log_ring_dimension ||
            self.max_log_ring_dimension >= u32::BITS as usize ||
            self.crt_modulus_bits == 0 ||
            self.gadget_base_bits >= u64::BITS ||
            self.security_bits == 0 ||
            !self.trapdoor_sigma.is_finite() ||
            !self.error_sigma.is_finite() ||
            self.trapdoor_sigma <= 0.0 ||
            self.error_sigma <= 0.0
        {
            return Err(DiamondParameterSearchError::InvalidRange);
        }
        Ok(())
    }
}

fn lean_checker_accepts(compiler: &DiamondWeCompiler) -> Result<bool, DiamondParameterSearchError> {
    let config = &compiler.config;
    let trapdoor_sigma = config
        .trapdoor_sigma
        .evaluate_rational(&ParamEnv::default())
        .map_err(|_| DiamondParameterSearchError::Expression)?;
    let error_sigma = config
        .error_sigma
        .evaluate_rational(&ParamEnv::default())
        .map_err(|_| DiamondParameterSearchError::Expression)?;
    let args = [
        compiler.shape.instance_width.to_string(),
        compiler.shape.witness_width.to_string(),
        compiler.shape.depth.to_string(),
        compiler.shape.max_layer_width.to_string(),
        config.ring_dimension.to_string(),
        config.input_count.to_string(),
        config.digit_base.to_string(),
        config.batch_bits.to_string(),
        config.digit_count.to_string(),
        config.modulus.to_string(),
        config.gadget_base.to_string(),
        config.error_max_coefficient_bound.to_string(),
        config.preimage_max_coefficient_bound.to_string(),
        trapdoor_sigma.numerator().to_string(),
        trapdoor_sigma.denominator().to_string(),
        error_sigma.numerator().to_string(),
        error_sigma.denominator().to_string(),
    ];
    let output = Command::new(env!("MXX_DIAMOND_CHECKER"))
        .args(args)
        .output()
        .map_err(|error| DiamondParameterSearchError::CheckerInfrastructure(error.to_string()))?;
    if !output.status.success() {
        return Err(DiamondParameterSearchError::CheckerInfrastructure(format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    match String::from_utf8_lossy(&output.stdout).trim() {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(DiamondParameterSearchError::CheckerInfrastructure(format!(
            "malformed checker response: {other}"
        ))),
    }
}

fn lattice_security_bits(
    parameters: &DCRTPolyParams,
    error_sigma: f64,
) -> Result<u64, DiamondParameterSearchError> {
    let modulus: Arc<BigUint> = parameters.modulus();
    let output = Command::new("lattice-estimator-cli")
        .arg(parameters.ring_dimension().to_string())
        .arg(modulus.to_string())
        .arg("--s-dist")
        .arg(ternary_distribution_json())
        .arg("--e-dist")
        .arg(discrete_gaussian_distribution_json(error_sigma))
        .output()?;
    if !output.status.success() {
        return Err(DiamondParameterSearchError::EstimatorFailure(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value = stdout.lines().rev().find(|line| !line.trim().is_empty()).unwrap_or("").trim();
    value.parse().map_err(|_| DiamondParameterSearchError::EstimatorOutput(stdout.into_owned()))
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
    fn exact_cutoffs_and_lean_checker_match_rust() {
        assert_eq!(
            default_error_max_coefficient_bound(&RealExpr::from_integer(4)).unwrap(),
            BigInt::from(26)
        );
        let search = DiamondParameterSearch {
            shape: BooleanCircuitShape {
                instance_width: 0,
                witness_width: 1,
                depth: 1,
                max_layer_width: 1,
            },
            min_crt_depth: 1,
            initial_max_crt_depth: 1,
            min_log_ring_dimension: 3,
            max_log_ring_dimension: 3,
            crt_modulus_bits: 60,
            gadget_base_bits: 2,
            security_bits: 1,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            trapdoor_sigma: 4.0,
            error_sigma: 1.0,
            bgg_tag: Vec::new(),
        };
        let selected = search.search_with_security_estimator(|_, _| Ok(1)).unwrap();
        assert!(lean_checker_accepts(&selected.compiler).unwrap());

        let mut invalid_shape_relation = selected.compiler.clone();
        invalid_shape_relation.shape.witness_width += 1;
        assert!(!lean_checker_accepts(&invalid_shape_relation).unwrap());
    }
}
