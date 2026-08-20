use super::{
    DiamondWeCompiler, DiamondWeConfig, default_error_max_coefficient_bound,
    default_preimage_max_coefficient_bound,
};
use mxx_correctness::operational_noise::{
    OperationalCheckRequest, OperationalGadgetLayout, OperationalParameterValue,
    OperationalSimulationReport, check_operational_noise_candidate,
};
use mxx_gadgets::circuit::{BooleanCircuitError, BooleanCircuitShape};
use mxx_ir_core::RealExpr;
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, process::Command, sync::Arc, time::Instant};
use thiserror::Error;
use tracing::{debug, info};

#[derive(Clone, Debug)]
pub struct DiamondParameterSearch {
    pub shape: BooleanCircuitShape,
    pub min_crt_depth: usize,
    pub initial_max_crt_depth: usize,
    pub max_crt_depth: usize,
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
    #[error("no correctness-valid Diamond parameters were found up to CRT depth {max_crt_depth}")]
    SearchExhausted { max_crt_depth: usize },
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
    #[error("the Rust operational checker could not evaluate the Diamond candidate: {0}")]
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
        let search_started = Instant::now();
        self.validate()?;
        info!(
            min_crt_depth = self.min_crt_depth,
            initial_max_crt_depth = self.initial_max_crt_depth,
            max_crt_depth = self.max_crt_depth,
            min_log_ring_dimension = self.min_log_ring_dimension,
            max_log_ring_dimension = self.max_log_ring_dimension,
            security_bits = self.security_bits,
            crt_modulus_bits = self.crt_modulus_bits,
            gadget_base_bits = self.gadget_base_bits,
            trapdoor_sigma = self.trapdoor_sigma,
            error_sigma = self.error_sigma,
            "starting Diamond WE parameter search"
        );
        let mut cache = BTreeMap::<usize, Candidate>::new();
        let mut evaluate = |crt_depth: usize| -> Result<bool, DiamondParameterSearchError> {
            let started = Instant::now();
            if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(crt_depth) {
                info!(crt_depth, "evaluating Diamond WE parameter candidate");
                let (log_ring_dimension, achieved) =
                    self.select_ring_dimension(crt_depth, &mut estimate_security)?;
                entry.insert(self.evaluate_candidate(crt_depth, log_ring_dimension, achieved)?);
                let candidate = cache.get(&crt_depth).expect("inserted candidate");
                info!(
                    crt_depth,
                    ring_dimension = candidate.selected.ring_dimension,
                    modulus_bits = candidate.selected.modulus_bits,
                    achieved_security_bits = candidate.selected.achieved_security_bits,
                    accepted = candidate.correct,
                    elapsed_seconds = started.elapsed().as_secs_f64(),
                    "finished Diamond WE parameter candidate"
                );
            } else {
                debug!(crt_depth, "reusing cached Diamond WE parameter candidate");
            }
            Ok(cache.get(&crt_depth).expect("inserted candidate").correct)
        };
        let mut upper = self.initial_max_crt_depth;
        while !evaluate(upper)? {
            if upper == self.max_crt_depth {
                return Err(DiamondParameterSearchError::SearchExhausted {
                    max_crt_depth: self.max_crt_depth,
                });
            }
            upper = upper
                .checked_mul(2)
                .ok_or(DiamondParameterSearchError::DepthOverflow)?
                .min(self.max_crt_depth);
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
        let selected = cache
            .remove(&best)
            .map(|candidate| candidate.selected)
            .ok_or(DiamondParameterSearchError::NoCorrectParameters)?;
        info!(
            crt_depth = selected.crt_depth,
            ring_dimension = selected.ring_dimension,
            modulus_bits = selected.modulus_bits,
            achieved_security_bits = selected.achieved_security_bits,
            elapsed_seconds = search_started.elapsed().as_secs_f64(),
            "selected Diamond WE parameters"
        );
        Ok(selected)
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
            let started = Instant::now();
            debug!(
                crt_depth,
                ring_dimension,
                modulus_bits = parameters.modulus().bits(),
                "starting Diamond WE lattice security estimate"
            );
            let achieved = estimate_security(&parameters, self.error_sigma)?;
            info!(
                crt_depth,
                ring_dimension,
                achieved_security_bits = achieved,
                elapsed_seconds = started.elapsed().as_secs_f64(),
                "finished Diamond WE lattice security estimate"
            );
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
        let started = Instant::now();
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
        debug!(
            crt_depth,
            ring_dimension,
            error_max_coefficient_bound = %compiler.config.error_max_coefficient_bound,
            preimage_max_coefficient_bound = %compiler.config.preimage_max_coefficient_bound,
            setup_elapsed_seconds = started.elapsed().as_secs_f64(),
            "constructed Diamond WE checker candidate"
        );
        let declaration = compiler
            .protocol_decl()
            .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?;
        let request = operational_request(&parameters, &compiler)?;
        let checker_started = Instant::now();
        let report = check_operational_noise_candidate(declaration.protocol(), &request).map_err(
            |error| DiamondParameterSearchError::CheckerInfrastructure(error.to_string()),
        )?;
        let correct = operational_decision(&report);
        info!(
            crt_depth,
            ring_dimension,
            accepted = correct,
            noise_bound = %report.noise_bound,
            ciphertext_modulus = %report.ciphertext_modulus,
            lowered_term_count = report.diagnostics.lowered_term_count,
            normalization_node_count = report.diagnostics.normalization_node_count,
            normalization_node_total = report.diagnostics.normalization_node_total,
            normalization_exact_term_count = report.diagnostics.normalization_exact_term_count,
            normalization_relation_count = report.diagnostics.normalization_relation_count,
            normalization_relation_applied = report.diagnostics.normalization_relation_applied,
            normalization_relation_remaining = report.diagnostics.normalization_relation_remaining,
            normalization_bounded_fold_count = report.diagnostics.normalization_bounded_fold_count,
            elapsed_seconds = checker_started.elapsed().as_secs_f64(),
            "finished Diamond WE Rust operational-noise check"
        );
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
            self.initial_max_crt_depth > self.max_crt_depth ||
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

/// A successfully evaluated report is a candidate decision; only checker errors are search
/// infrastructure failures.  Keeping this boundary explicit prevents ordinary rejected bounds
/// from aborting CRT-depth search.
fn operational_decision(report: &OperationalSimulationReport) -> bool {
    report.accepted
}

fn operational_request(
    parameters: &DCRTPolyParams,
    compiler: &DiamondWeCompiler,
) -> Result<OperationalCheckRequest, DiamondParameterSearchError> {
    let config = &compiler.config;
    let (crt_moduli, crt_bits, _) = parameters.to_crt();
    let base_bits = parameters.base_bits();
    let small_digit_count = crt_bits.div_ceil(base_bits as usize);
    let smallest_crt_modulus =
        crt_moduli.iter().copied().min().ok_or(DiamondParameterSearchError::InvalidRange)?;
    let bindings = compiler
        .circuit_bindings()
        .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?;
    let binding_count = bindings.integers.len() + bindings.reals.len();
    let mut environment = bindings
        .integers
        .into_iter()
        .map(|(name, value)| (name, OperationalParameterValue::Integer(value)))
        .collect::<Vec<_>>();
    for (name, value) in bindings.reals {
        if value.denominator() != &BigInt::from(1) {
            return Err(DiamondParameterSearchError::Expression);
        }
        environment.push((name, OperationalParameterValue::Integer(value.numerator().clone())));
    }
    debug_assert_eq!(environment.len(), binding_count);
    Ok(OperationalCheckRequest {
        environment,
        layouts: vec![OperationalGadgetLayout {
            params_id: "diamond-dcrt".to_owned(),
            ring_dimension: parameters.ring_dimension() as usize,
            crt_moduli,
            crt_bits,
            base_bits: base_bits as usize,
            base: config.gadget_base.clone(),
            regular_digit_count: config.digit_count,
            small_digit_count,
            smallest_crt_modulus,
        }],
        target_id: "diamond-boolean-interval".to_owned(),
    })
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

    fn small_search() -> DiamondParameterSearch {
        DiamondParameterSearch {
            shape: BooleanCircuitShape {
                instance_width: 0,
                witness_width: 1,
                depth: 1,
                max_layer_width: 1,
            },
            min_crt_depth: 1,
            initial_max_crt_depth: 2,
            max_crt_depth: 2,
            min_log_ring_dimension: 5,
            max_log_ring_dimension: 5,
            crt_modulus_bits: 60,
            gadget_base_bits: 2,
            security_bits: 1,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            trapdoor_sigma: 4.0,
            error_sigma: 1.0,
            bgg_tag: Vec::new(),
        }
    }

    fn smallest_checker_candidate() -> (DCRTPolyParams, DiamondWeCompiler) {
        let search = small_search();
        let parameters = DCRTPolyParams::new(32, 2, 60, 2);
        let error_sigma = RealExpr::from_f64_exact(search.error_sigma).unwrap();
        let trapdoor_sigma = RealExpr::from_f64_exact(search.trapdoor_sigma).unwrap();
        let compiler = DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: BigInt::from(parameters.modulus().as_ref().clone()),
                ring_dimension: parameters.ring_dimension() as usize,
                input_count: search.input_count,
                digit_base: search.digit_base,
                batch_bits: search.batch_bits,
                gadget_base: BigInt::from(1_u8) << search.gadget_base_bits,
                digit_count: parameters.modulus_digits(),
                trapdoor_sigma,
                error_sigma,
                error_max_coefficient_bound: BigInt::from(26),
                preimage_max_coefficient_bound: BigInt::from(26),
                bgg_tag: search.bgg_tag,
            },
            search.shape,
        )
        .unwrap();
        (parameters, compiler)
    }

    #[test]
    fn operational_request_uses_the_concrete_dcrt_layout_and_all_graph_parameters() {
        assert_eq!(
            default_error_max_coefficient_bound(&RealExpr::from_integer(4)).unwrap(),
            BigInt::from(26)
        );
        let (parameters, compiler) = smallest_checker_candidate();
        let request = operational_request(&parameters, &compiler).unwrap();
        let bindings = compiler.circuit_bindings().unwrap();
        assert_eq!(request.environment.len(), bindings.integers.len() + bindings.reals.len());
        assert_eq!(request.target_id, "diamond-boolean-interval");
        assert_eq!(request.layouts.len(), 1);
        let layout = &request.layouts[0];
        let (crt_moduli, crt_bits, _) = parameters.to_crt();
        assert_eq!(layout.ring_dimension, parameters.ring_dimension() as usize);
        assert_eq!(layout.crt_moduli, crt_moduli);
        assert_eq!(layout.crt_bits, crt_bits);
        assert_eq!(layout.base_bits, parameters.base_bits() as usize);
        assert_eq!(layout.base, compiler.config.gadget_base);
    }

    #[test]
    fn smallest_diamond_candidate_reaches_typed_checker_rejection() {
        let (parameters, compiler) = smallest_checker_candidate();
        let declaration = compiler.protocol_decl().unwrap();
        let request = operational_request(&parameters, &compiler).unwrap();
        let error = check_operational_noise_candidate(declaration.protocol(), &request)
            .expect_err("the current Diamond residual is rejected during typed lowering");
        assert!(matches!(
            error,
            mxx_correctness::operational_noise::OperationalSimulationError::Production(_)
        ));
    }

    #[test]
    #[ignore = "requires lattice-estimator-cli and Sage on PATH"]
    fn small_search_with_lattice_estimator() {
        let selected = small_search().search().unwrap();
        assert!(selected.achieved_security_bits >= 1);
    }
}
