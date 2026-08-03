use super::{
    DiamondNoiseError, DiamondNoiseSimulation, DiamondWeCompiler, DiamondWeConfig,
    simulate_diamond_noise,
};
use mxx_gadgets::circuit::PolyCircuit;
use mxx_ir_core::RealExpr;
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, process::Command, sync::Arc};
use thiserror::Error;
use tracing::info;

#[derive(Clone, Debug)]
pub struct DiamondParameterSearch {
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
    pub crt_depth: usize,
    pub log_ring_dimension: usize,
    pub ring_dimension: u32,
    pub modulus: BigUint,
    pub modulus_bits: usize,
    pub achieved_security_bits: u64,
    pub simulation: DiamondNoiseSimulation,
}

#[derive(Debug, Error)]
pub enum DiamondParameterSearchError {
    #[error("the Diamond parameter-search range is invalid")]
    InvalidRange,
    #[error("Diamond CRT-depth growth overflowed")]
    DepthOverflow,
    #[error("no ring dimension in the configured range meets the security target")]
    NoSecureRingDimension,
    #[error("no correctness-valid Diamond parameters were found")]
    NoCorrectParameters,
    #[error(transparent)]
    Noise(#[from] DiamondNoiseError),
    #[error("lattice-estimator-cli could not be started: {0}")]
    EstimatorIo(#[from] std::io::Error),
    #[error("lattice-estimator-cli failed: {0}")]
    EstimatorFailure(String),
    #[error("lattice-estimator-cli returned an invalid security estimate: {0}")]
    EstimatorOutput(String),
    #[error("a Diamond search parameter cannot be represented by the IR")]
    Expression,
    #[error("the selected Diamond compiler configuration is invalid: {0}")]
    Config(String),
}

struct Candidate {
    selected: DiamondSelectedParameters,
    correct: bool,
}

impl DiamondParameterSearch {
    pub fn search(
        &self,
        circuit: &PolyCircuit<DCRTPoly>,
        instance: &[bool],
    ) -> Result<DiamondSelectedParameters, DiamondParameterSearchError> {
        self.search_with_security_estimator(circuit, instance, |parameters, sigma| {
            lattice_security_bits(parameters, sigma)
        })
    }

    pub fn search_with_security_estimator<F>(
        &self,
        circuit: &PolyCircuit<DCRTPoly>,
        instance: &[bool],
        mut estimate_security: F,
    ) -> Result<DiamondSelectedParameters, DiamondParameterSearchError>
    where
        F: FnMut(&DCRTPolyParams, f64) -> Result<u64, DiamondParameterSearchError>,
    {
        self.validate()?;
        let mut cache = BTreeMap::<usize, Candidate>::new();
        let mut evaluate = |crt_depth: usize| -> Result<bool, DiamondParameterSearchError> {
            if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(crt_depth) {
                let selected_ring =
                    self.select_ring_dimension(crt_depth, &mut estimate_security)?;
                let candidate = self.evaluate_candidate(
                    circuit,
                    instance,
                    crt_depth,
                    selected_ring.0,
                    selected_ring.1,
                )?;
                entry.insert(candidate);
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
        let mut best_depth = upper;
        while low <= high {
            let depth = low + (high - low) / 2;
            if evaluate(depth)? {
                best_depth = depth;
                if depth == self.min_crt_depth {
                    break;
                }
                high = depth - 1;
            } else {
                low = depth + 1;
            }
        }
        cache
            .remove(&best_depth)
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
            let log_ring_dimension = low + (high - low) / 2;
            let ring_dimension = 1u32
                .checked_shl(log_ring_dimension as u32)
                .ok_or(DiamondParameterSearchError::InvalidRange)?;
            let parameters = DCRTPolyParams::new(
                ring_dimension,
                crt_depth,
                self.crt_modulus_bits,
                self.gadget_base_bits,
            );
            let achieved = estimate_security(&parameters, self.error_sigma)?;
            info!(
                crt_depth,
                log_ring_dimension,
                ring_dimension,
                achieved_security_bits = achieved,
                required_security_bits = self.security_bits,
                "evaluated Diamond WE lattice-security candidate"
            );
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
        selected.ok_or(DiamondParameterSearchError::NoSecureRingDimension)
    }

    fn evaluate_candidate(
        &self,
        circuit: &PolyCircuit<DCRTPoly>,
        instance: &[bool],
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
        let compiler = DiamondWeCompiler::new(DiamondWeConfig {
            modulus: BigInt::from(modulus.as_ref().clone()),
            ring_dimension: ring_dimension as usize,
            input_count: self.input_count,
            digit_base: self.digit_base,
            batch_bits: self.batch_bits,
            gadget_base: BigInt::from(1u64 << self.gadget_base_bits),
            digit_count: parameters.modulus_digits(),
            trapdoor_sigma: RealExpr::from_f64_exact(self.trapdoor_sigma)
                .map_err(|_| DiamondParameterSearchError::Expression)?,
            error_sigma: RealExpr::from_f64_exact(self.error_sigma)
                .map_err(|_| DiamondParameterSearchError::Expression)?,
            bgg_tag: self.bgg_tag.clone(),
        })
        .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?;
        let simulation = simulate_diamond_noise(&compiler, circuit, instance)?;
        let correct = simulation.final_decode.within_threshold;
        info!(
            crt_depth,
            log_ring_dimension,
            ring_dimension,
            modulus_bits = parameters.modulus_bits(),
            noise_bound = %simulation
                .final_decode
                .estimate
                .noise
                .as_ref()
                .map(|noise| noise.bound.clone())
                .unwrap_or_default(),
            decode_threshold = %simulation.final_decode.threshold,
            correct,
            "evaluated Diamond WE noise candidate"
        );
        Ok(Candidate {
            selected: DiamondSelectedParameters {
                parameters,
                crt_depth,
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

    fn validate(&self) -> Result<(), DiamondParameterSearchError> {
        if self.min_crt_depth == 0 ||
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

fn lattice_security_bits(
    parameters: &DCRTPolyParams,
    error_sigma: f64,
) -> Result<u64, DiamondParameterSearchError> {
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
    fn estimator_distribution_arguments_are_valid_unescaped_json() {
        assert_eq!(ternary_distribution_json(), r#"{"name":"Ternary"}"#);
        assert_eq!(
            discrete_gaussian_distribution_json(4.5),
            r#"{"name":"DiscreteGaussian","stddev":4.5}"#
        );
    }

    #[test]
    fn search_selects_the_smallest_correct_crt_depth_and_ring_dimension() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1);
        circuit.output([input]);
        let search = DiamondParameterSearch {
            min_crt_depth: 1,
            initial_max_crt_depth: 2,
            min_log_ring_dimension: 3,
            max_log_ring_dimension: 4,
            crt_modulus_bits: 20,
            gadget_base_bits: 4,
            security_bits: 1,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            trapdoor_sigma: 4.578,
            error_sigma: 1.0,
            bgg_tag: b"diamond-search-test".to_vec(),
        };
        let selected = search
            .search_with_security_estimator(&circuit, &[], |parameters, _| {
                Ok(if parameters.ring_dimension() >= 8 { 1 } else { 0 })
            })
            .unwrap();
        assert_eq!(selected.crt_depth, 3);
        assert_eq!(selected.ring_dimension, 8);
        assert!(selected.simulation.final_decode.estimate.has_signal);
        assert!(selected.simulation.final_decode.within_threshold);

        let mut invalid = search;
        invalid.error_sigma = 0.0;
        assert!(matches!(
            invalid.search_with_security_estimator(&circuit, &[], |_, _| Ok(1)),
            Err(DiamondParameterSearchError::InvalidRange)
        ));
    }
}
