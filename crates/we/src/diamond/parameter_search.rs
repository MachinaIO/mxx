use super::{
    DiamondWeCompiler, DiamondWeConfig, default_error_max_coefficient_bound,
    default_preimage_max_coefficient_bound,
    graph::{HASH_KEY_INPUT, MESSAGE_INPUT, NOISY_PLAINTEXT_OUTPUT},
};
use mxx_gadgets::circuit::{BooleanCircuitError, BooleanCircuitShape};
use mxx_ir_core::RealExpr;
use mxx_noise_simulator::{
    ExternalInputFact, ExternalInputValue, SimulationLimits, SimulationProgram, SimulationRequest,
    SimulationRoot, SimulationStage, StageId, simulate,
};
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
    /// Simulator upper bound for the absolute coefficient error of the
    /// decryption graph's noisy plaintext output.
    pub noise_bound: BigUint,
}

#[derive(Debug, Error)]
pub enum DiamondParameterSearchError {
    #[error("the Diamond parameter-search range is invalid")]
    InvalidRange,
    #[error(transparent)]
    Shape(#[from] BooleanCircuitError),
    #[error("Diamond CRT-depth growth overflowed")]
    DepthOverflow,
    #[error("no noise-safe Diamond parameters were found up to CRT depth {max_crt_depth}")]
    SearchExhausted { max_crt_depth: usize },
    #[error("no ring dimension in the configured range meets the security target")]
    NoSecureRingDimension,
    #[error("no noise-safe Diamond parameters were found")]
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
    #[error("the Diamond noise simulator could not evaluate the candidate: {0}")]
    SimulatorInfrastructure(String),
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
            "constructed Diamond WE simulator candidate"
        );
        let request = simulation_request(&compiler)?;
        let simulator_started = Instant::now();
        let report = simulate(&request).map_err(|error| {
            DiamondParameterSearchError::SimulatorInfrastructure(error.to_string())
        })?;
        let noise_bound = report
            .roots
            .iter()
            .find(|root| {
                root.root.stage == StageId("decrypt".to_owned()) &&
                    root.root.output == NOISY_PLAINTEXT_OUTPUT
            })
            .map(|root| &root.maximum_absolute_coefficient_error)
            .ok_or_else(|| {
                DiamondParameterSearchError::SimulatorInfrastructure(
                    "simulator report omitted the Diamond residual root".to_owned(),
                )
            })?;
        let correct = boolean_interval_accepts(&compiler.config.modulus, noise_bound);
        info!(
            crt_depth,
            ring_dimension,
            accepted = correct,
            noise_bound = %noise_bound,
            planned_wires = report.diagnostics.planned_wires,
            transfer_steps = report.diagnostics.transfer_steps,
            dropped_carriers = report.diagnostics.dropped_carriers.len(),
            elapsed_seconds = simulator_started.elapsed().as_secs_f64(),
            "finished Diamond WE noise simulation"
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
                noise_bound: noise_bound.clone(),
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

fn simulation_request(
    compiler: &DiamondWeCompiler,
) -> Result<SimulationRequest, DiamondParameterSearchError> {
    let environment = compiler
        .circuit_bindings()
        .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?;
    let encryption = compiler
        .build_encryption()
        .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?
        .graph;
    let encryption_production = mxx_ir_core::artifact::ProductionId {
        spec_hash: mxx_ir_core::encoding::spec_hash(&encryption.graph, &environment)
            .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?,
        execution_nonce: [0; 32],
    };
    let decryption = compiler
        .build_decryption(encryption_production.clone())
        .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?
        .graph;
    let decryption_production = mxx_ir_core::artifact::ProductionId {
        spec_hash: mxx_ir_core::encoding::spec_hash(&decryption.graph, &environment)
            .map_err(|error| DiamondParameterSearchError::Config(error.to_string()))?,
        execution_nonce: [0; 32],
    };
    let encrypt = StageId("encrypt".to_owned());
    let decrypt = StageId("decrypt".to_owned());
    let mut external_inputs = Vec::new();
    add_circuit_facts(
        &mut external_inputs,
        &encrypt,
        compiler.shape.depth,
        compiler.shape.max_layer_width,
    );
    add_circuit_facts(
        &mut external_inputs,
        &decrypt,
        compiler.shape.depth,
        compiler.shape.max_layer_width,
    );
    external_inputs.extend([
        ExternalInputFact {
            stage: encrypt.clone(),
            input: HASH_KEY_INPUT.to_owned(),
            value: ExternalInputValue::Bytes,
        },
        ExternalInputFact {
            stage: encrypt.clone(),
            input: MESSAGE_INPUT.to_owned(),
            value: ExternalInputValue::Boolean,
        },
        integer_family_fact(
            &encrypt,
            mxx_gadgets::circuit::BOOLEAN_INSTANCE_INPUT,
            vec![compiler.shape.max_layer_width],
            0,
            1,
        ),
        integer_family_fact(
            &decrypt,
            mxx_gadgets::circuit::BOOLEAN_INSTANCE_INPUT,
            vec![compiler.shape.max_layer_width],
            0,
            1,
        ),
        integer_family_fact(
            &decrypt,
            mxx_gadgets::circuit::BOOLEAN_WITNESS_INPUT,
            vec![compiler.shape.max_layer_width],
            0,
            1,
        ),
    ]);
    Ok(SimulationRequest {
        program: SimulationProgram {
            stages: vec![
                SimulationStage {
                    id: encrypt,
                    production_id: encryption_production,
                    graph: encryption.graph,
                },
                SimulationStage {
                    id: decrypt.clone(),
                    production_id: decryption_production,
                    graph: decryption.graph,
                },
            ],
        },
        environment,
        roots: vec![SimulationRoot { stage: decrypt, output: NOISY_PLAINTEXT_OUTPUT.to_owned() }],
        external_inputs,
        limits: SimulationLimits::default(),
    })
}

fn add_circuit_facts(
    facts: &mut Vec<ExternalInputFact>,
    stage: &StageId,
    depth: usize,
    max_layer_width: usize,
) {
    let flattened = depth.saturating_mul(max_layer_width);
    facts.extend([
        integer_family_fact(stage, "circuit-active-gate-count", vec![depth], 0, max_layer_width),
        integer_family_fact(stage, "circuit-gate-kind", vec![flattened], 0, 5),
        integer_family_fact(
            stage,
            "circuit-left-source",
            vec![flattened],
            0,
            max_layer_width.saturating_sub(1),
        ),
        integer_family_fact(
            stage,
            "circuit-right-source",
            vec![flattened],
            0,
            max_layer_width.saturating_sub(1),
        ),
        integer_family_fact(
            stage,
            "circuit-output-source",
            vec![1],
            0,
            max_layer_width.saturating_sub(1),
        ),
    ]);
}

fn integer_family_fact(
    stage: &StageId,
    input: &str,
    shape: Vec<usize>,
    minimum: i64,
    maximum_inclusive: usize,
) -> ExternalInputFact {
    ExternalInputFact {
        stage: stage.clone(),
        input: input.to_owned(),
        value: ExternalInputValue::Family {
            shape,
            element: Box::new(ExternalInputValue::IntegerRange {
                minimum: minimum.into(),
                maximum_inclusive: (maximum_inclusive as i64).into(),
            }),
        },
    }
}

fn boolean_interval_accepts(modulus: &BigInt, noise: &BigUint) -> bool {
    if modulus < &BigInt::from(4_u8) {
        return false;
    }
    let quarter = mxx_ir_core::IntExpr::RoundDiv(
        Box::new(mxx_ir_core::IntExpr::constant(modulus.clone() - BigInt::from(2_u8))),
        Box::new(mxx_ir_core::IntExpr::constant(4_u8)),
    )
    .evaluate(&mxx_ir_core::ParamEnv::default())
    .expect("constant positive RoundDiv denominator");
    let half = modulus / BigInt::from(2_u8);
    let noise = BigInt::from(noise.clone());
    // The decoder's true interval is inclusive.  A zero plaintext with an
    // error of exactly `quarter` therefore crosses its lower boundary, so
    // the false-message condition is strict even though the true-message
    // boundaries below are inclusive.
    quarter > noise.clone() &&
        modulus - (BigInt::from(3_u8) * &quarter + &noise) > BigInt::ZERO &&
        half >= quarter.clone() + &noise &&
        BigInt::from(3_u8) * quarter >= half + noise
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
                instance_width: 1,
                witness_width: 1,
                depth: 1,
                max_layer_width: 2,
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

    #[test]
    fn boolean_interval_acceptance_is_owned_by_parameter_search() {
        assert!(boolean_interval_accepts(&BigInt::from(257_u16), &BigUint::from(1_u8)));
        assert!(!boolean_interval_accepts(&BigInt::from(257_u16), &BigUint::from(100_u8)));
        assert!(!boolean_interval_accepts(&BigInt::from(3_u8), &BigUint::ZERO));
    }

    #[test]
    fn boolean_interval_uses_decoder_round_division_at_modulus_boundaries() {
        for modulus in [4_u8, 5, 6, 7] {
            assert!(
                boolean_interval_accepts(&BigInt::from(modulus), &BigUint::ZERO),
                "q={modulus} should accept zero noise"
            );
        }
        assert!(boolean_interval_accepts(&BigInt::from(257_u16), &BigUint::from(63_u8)));
        // q=257 has Q=RoundDiv(255,4)=64.  Although 128 +/- 64 is still
        // inside the inclusive true interval [64,192], 0 + 64 is decoded as
        // true, so uniform correctness requires rejecting this boundary.
        assert!(!boolean_interval_accepts(&BigInt::from(257_u16), &BigUint::from(64_u8)));
    }

    #[test]
    #[ignore = "requires lattice-estimator-cli and Sage on PATH"]
    fn small_search_with_lattice_estimator() {
        let selected = small_search().search().unwrap();
        assert!(selected.achieved_security_bits >= 1);
    }
}
