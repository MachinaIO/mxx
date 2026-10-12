#![cfg(feature = "gpu")]

//! The application-facing RNS Sparse-LWR PRF lifecycle.
//!
//! With `MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID`, the workflow runs the recorded parameter set of
//! that ID in `lean/lwr_prf/parameters.csv` instead of searching, and exports the Lean correctness
//! claim of the executed graphs to `lean/lwr_prf/<parameter id>/generated`. With
//! `MXX_EXPONENT_LUT_LWR_PRF_EXPORT_ONLY=1` as well, it builds and validates the same stage graphs
//! but executes only the secret sampling (on the CPU), and exports the claim without a round trip;
//! this exports parameter sets too large for the local GPU.

mod common;

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime, RuntimeValue,
    authority::CpuExecution,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend_on},
    matrix::dcrt_poly::DCRTPolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_dsl::{DslContext, Family, FamilyType, IdealSpec, Int, IntType};
use mxx_exponent_lut::{
    ExpLayout, api,
    decoder::simulation::AverageCaseConfig,
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, FamilyRange,
        InputBinding, LutOutputRepresentation, Operand, WireId,
    },
    program_gadgets::{
        pbc::{
            PbcCell, PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, clear_pbc_inner_product,
            generate_key_layout,
        },
        prf::{PrfOutput, SparseLwrPrf, SparseLwrPrfProfile},
    },
    rns_bgg::{
        RhsMode, RnsBggDerivedWorkload, RnsBggDynamicDistributions, RnsBggDynamicNoise,
        RnsBggDynamicSetup, RnsBggGaussian, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseRhs,
        RnsBggNoiseTrace, RnsBggNoiseValue, RnsBggParameters, RnsBggPlanRequest, RnsEncryptBuild,
        RnsNoiseBuild, RnsOnlineEvalBuild, RnsOutputUse, RnsPreprocessBuild,
        covering_automorphism_layout, decode_output, verify_output,
    },
    utils::{
        audit_program_at_crt_depth, env_usize, lattice_security_bits, noise_model, rhs_mode,
        ring_dimension_candidates,
    },
};
use mxx_ir_core::{
    Graph, IntExpr, ParamEnv, ValidatedGraph,
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputValueContract, OutputRef, ProtocolDecl, ProtocolInputDestination,
        ProtocolPreconditionSpec, PurePredicateSpec, StageId, Workflow,
    },
};
use num_bigint::{BigInt, BigUint};
use rayon::prelude::*;
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, env};

const HASH_KEY: &[u8; 32] = b"lwr-prf-rns-public-api-hash-key1";

type ParameterRow = BTreeMap<String, String>;

/// The recorded parameter set selected by `MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID`, if any.
fn recorded_parameters() -> Option<(String, ParameterRow)> {
    env::var("MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID")
        .ok()
        .map(|id| (id.clone(), common::parameter_row("lwr_prf", &id)))
}

/// Column `column` of the recorded row, or else the environment variable `name` with `default`.
fn setting(recorded: Option<&ParameterRow>, column: &str, name: &str, default: usize) -> usize {
    match recorded {
        Some(row) => common::column(row, column),
        None => env_usize(name, default).unwrap_or_else(|error| panic!("{name}: {error}")),
    }
}

/// `MXX_EXPONENT_LUT_LWR_PRF_EXPORT_ONLY=1`: build and export the claim without a round trip.
fn export_only() -> bool {
    env::var("MXX_EXPONENT_LUT_LWR_PRF_EXPORT_ONLY").is_ok_and(|value| value == "1")
}

/// A stage execution that executes only the secret sampling, on the CPU. Every other stage is
/// validated and recorded by its caller, returns the production its execution would have (which
/// depends only on the validated graph), and yields no outputs.
struct GraphOnly(CpuExecution);

impl api::StageExecution<api::StageStore> for GraphOnly {
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        _: BTreeMap<String, RuntimeValue>,
        _: &mut api::StageStore,
        execution_nonce: [u8; 32],
    ) -> Result<mxx_ir_core::artifact::ProductionId, api::ApiError> {
        let specification =
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .expect("stage specification hash");
        Ok(mxx_ir_core::artifact::production_id(specification, execution_nonce))
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut api::StageStore,
        execution_nonce: [u8; 32],
    ) -> Result<api::StageOutputs, api::ApiError> {
        if validated.source.name() == "rns-bgg-secret" {
            self.0.evaluate(validated, inputs, store, execution_nonce)
        } else {
            Ok(api::StageOutputs { outputs: BTreeMap::new() })
        }
    }

    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, api::ApiError> {
        api::StageExecution::<api::StageStore>::download_matrix(&self.0, value)
    }

    fn download_integers(&self, value: &RuntimeValue) -> Result<Vec<BigInt>, api::ApiError> {
        api::StageExecution::<api::StageStore>::download_integers(&self.0, value)
    }
}

/// Graphs of the executed stages, recorded for the exported claim.
type StageGraphs = RefCell<BTreeMap<&'static str, Graph>>;

/// A stage observer that records the validated graph of `stage`.
fn record<'a>(
    graphs: &'a StageGraphs,
    stage: &'static str,
) -> impl FnOnce(&ValidatedGraph) -> Result<(), api::ApiError> + 'a {
    move |validated| {
        graphs.borrow_mut().insert(stage, validated.source.clone());
        Ok(())
    }
}

/// The PRF Program: the sparse-LWR PRF gadget of every key and its outputs,
/// one per key.
#[derive(Clone)]
struct LwrPrf {
    program: ExponentLutProgram,
    prf: SparseLwrPrf,
}

impl LwrPrf {
    fn program(&self) -> &ExponentLutProgram {
        &self.program
    }
}

struct Fixture {
    parameters: RnsBggParameters,
    profile: SparseLwrPrfProfile,
    program: LwrPrf,
    /// The Static secrets, sampled before preprocessing. The single Dynamic
    /// secret is sampled only after preprocessing.
    payload: Option<mxx_exponent_lut::rns_bgg::RnsBggPayloadSecret>,
    mask: Option<mxx_exponent_lut::rns_bgg::RnsBggMaskSecret>,
    rhs: BTreeMap<mxx_exponent_lut::program::RhsId, InputBinding<DCRTPoly, Vec<DCRTPoly>>>,
    lhs: BTreeMap<WireId, InputBinding<DCRTPoly, Vec<DCRTPoly>>>,
    trace: RnsBggNoiseTrace,
    keys: Vec<PbcGeneratedKeyLayout>,
    a_exponents: Vec<u64>,
}

fn error_sigma(recorded: Option<&ParameterRow>) -> BigDecimal {
    if let Some(row) = recorded {
        return common::column(row, "error_sigma");
    }
    let sigma = env::var("MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA")
        .unwrap_or_else(|_| "4".to_owned())
        .parse()
        .expect("MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA must be a decimal");
    assert!(sigma > BigDecimal::from(0u8), "MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA must be positive");
    sigma
}

fn profile_and_program(
    ring_dimension: u32,
    recorded: Option<&ParameterRow>,
) -> (SparseLwrPrfProfile, LwrPrf, Vec<PbcGeneratedKeyLayout>) {
    // The LWR input modulus Q fixes the exponent layout inside the ring:
    // X^(2n / Q * z) adds exponents modulo Q natively, so width_log2 =
    // log2(Q) - 1 and stride_log2 = log2(n) - width_log2. The output modulus
    // P and the LWR dimension nu are independent of n.
    let input_modulus =
        setting(recorded, "input_modulus", "MXX_EXPONENT_LUT_LWR_PRF_INPUT_MODULUS", 32);
    assert!(
        input_modulus >= 4 && input_modulus.is_power_of_two(),
        "MXX_EXPONENT_LUT_LWR_PRF_INPUT_MODULUS must be a power of two at least 4"
    );
    let output_modulus =
        setting(recorded, "output_modulus", "MXX_EXPONENT_LUT_LWR_PRF_OUTPUT_MODULUS", 4);
    assert!(
        output_modulus >= 2 && output_modulus % 2 == 0 && input_modulus % output_modulus == 0,
        "MXX_EXPONENT_LUT_LWR_PRF_OUTPUT_MODULUS must be even and divide the input modulus"
    );
    let width_log2 = input_modulus.ilog2() - 1;
    let stride_log2 = ring_dimension.ilog2().checked_sub(width_log2).unwrap_or_else(|| {
        panic!(
            "LWR input modulus {input_modulus} exceeds twice the ring dimension {ring_dimension}"
        )
    });
    let lwr_dimension =
        setting(recorded, "lwr_dimension", "MXX_EXPONENT_LUT_LWR_PRF_LWR_DIMENSION", 16);
    let support_weight =
        setting(recorded, "support_weight", "MXX_EXPONENT_LUT_LWR_PRF_SUPPORT_WEIGHT", 2);
    let output_count = setting(recorded, "outputs", "MXX_EXPONENT_LUT_LWR_PRF_OUTPUTS", 1);
    assert!(
        lwr_dimension > 0 && support_weight > 0 && support_weight <= lwr_dimension,
        "the sparse key weight must be positive and at most the LWR dimension"
    );
    assert!(output_count > 0, "MXX_EXPONENT_LUT_LWR_PRF_OUTPUTS must be positive");
    let profile = SparseLwrPrfProfile::new(
        ExpLayout { stride_log2, width_log2 },
        output_modulus,
        lwr_dimension,
    )
    .expect("valid sparse-LWR profile");
    // Every output evaluates the same public vector `a` under its own sparse
    // key of weight h, placed by its own PBC layout with h' = h + 3 blocks.
    let parameters = PbcParameters::paper_evaluation(lwr_dimension, support_weight);
    // The keys are sampled from the recorded seed, `MXX_EXPONENT_LUT_LWR_PRF_KEY_SEED`, or a
    // fresh seed that is reported, so that any run can be repeated.
    let key_seed = match (recorded, env::var("MXX_EXPONENT_LUT_LWR_PRF_KEY_SEED")) {
        (Some(row), _) => common::column::<u64>(row, "key_seed"),
        (None, Ok(value)) => {
            value.parse::<u64>().expect("MXX_EXPONENT_LUT_LWR_PRF_KEY_SEED must be a u64")
        }
        (None, Err(_)) => rand::random(),
    };
    eprintln!("Sparse-LWR key seed: MXX_EXPONENT_LUT_LWR_PRF_KEY_SEED={key_seed}");
    let mut key_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(key_seed);
    let keys = (0..output_count)
        .map(|_| {
            let support =
                rand::seq::index::sample(&mut key_rng, lwr_dimension, support_weight).into_vec();
            let root = rand::Rng::random(&mut key_rng);
            generate_key_layout(&parameters, PbcRootSeed(root), &support)
                .expect("PBC layout for the sparse LWR key")
        })
        .collect::<Vec<_>>();
    let mut builder = ExponentLutProgramBuilder::new();
    let prf = SparseLwrPrf::build(
        &mut builder,
        profile,
        keys.iter().map(|key| key.public_layout.clone()).collect(),
        LutOutputRepresentation::Coefficient,
    )
    .expect("sparse-LWR PRF");
    let outputs = match prf.output() {
        PrfOutput::Scalar(output) => vec![Operand::Scalar(output)],
        PrfOutput::Family(family) => (0..output_count)
            .map(|index| Operand::Member { family, index: IntExpr::constant(index) })
            .collect(),
    };
    let program = LwrPrf { program: builder.finish(outputs).expect("sparse-LWR PRF program"), prf };
    eprintln!(
        "LWR PRF program: n={ring_dimension} Q={input_modulus} P={output_modulus} nu={lwr_dimension} h={support_weight} outputs={output_count} blocks={} block_width={}",
        parameters.bucket_count, keys[0].public_layout.bucket_width
    );
    (profile, program, keys)
}

fn noise_bindings(
    program: &LwrPrf,
    sigma: &BigDecimal,
    ring_dimension: u32,
) -> RnsBggNoiseBindings {
    let noise_inputs = program
        .program()
        .encryption_inputs()
        .inputs()
        .iter()
        .enumerate()
        .map(|(index, (_, declaration))| {
            match (declaration.knowledge(), declaration.shape()) {
                (AttributeKnowledge::PrivatePayload, InputBinding::Scalar(())) |
                (AttributeKnowledge::PublicPayload, InputBinding::Scalar(())) |
                (AttributeKnowledge::PublicPayload, InputBinding::Family(_)) => {}
                (knowledge, shape) => {
                    panic!("unexpected noise declaration: {knowledge:?} {shape:?}")
                }
            }
            RnsBggNoiseValue::sampled_encoding_sigma(
                sigma,
                format!("lwr-input-{index}").as_bytes(),
                &[BigInt::from(1u8)],
            )
            .expect("valid LWR encoding sigma")
        })
        .collect::<Vec<_>>();
    let mut input_bindings = BTreeMap::new();
    for ((wire, declaration), value) in
        program.program().encryption_inputs().inputs().iter().zip(noise_inputs)
    {
        let binding = match declaration.shape() {
            InputBinding::Scalar(()) => InputBinding::Scalar(value),
            // Every member of an input family is a separate encryption.
            InputBinding::Family(count) => InputBinding::Family(RnsBggNoiseFamily {
                element: value.independent_family_members(),
                range: FamilyRange::full(count.clone()).expect("valid input family range"),
                max_nonzero_payloads: 1u8.into(),
                row_length: None,
            }),
        };
        input_bindings.insert(*wire, binding);
    }
    // A one-key bucket family is one-hot; a key-indexed family holds one
    // bit of every key.
    let rhs_bindings = program
        .prf
        .selector_families()
        .into_iter()
        .map(|family| {
            (
                family.rhs,
                InputBinding::Family(RnsBggNoiseFamily {
                    element: RnsBggNoiseRhs {
                        payload_l1: BigUint::from(1u8),
                        payload_l2_squared: BigUint::from(1u8),
                    },
                    range: FamilyRange::full(IntExpr::constant(family.members))
                        .expect("valid selector family range"),
                    max_nonzero_payloads: family.max_nonzero.into(),
                    row_length: None,
                }),
            )
        })
        .collect();
    let bindings = RnsBggNoiseBindings::new(input_bindings, rhs_bindings)
        .expect("declaration-order noise bindings");
    // Dynamic packages are sampled under the encoding sigma, read through
    // the group covering the PRF's LUTs.
    match rhs_mode("MXX_EXPONENT_LUT_LWR_PRF_").expect("RHS mode configuration") {
        RhsMode::Static => bindings,
        RhsMode::Dynamic => bindings.with_dynamic(
            RnsBggDynamicNoise::for_program(
                [0; 32],
                program.program(),
                RnsBggDynamicDistributions::uniform(
                    RnsBggGaussian::with_default_cutoff(sigma.clone()).expect("valid sigma"),
                ),
                covering_automorphism_layout(program.program()),
                ring_dimension as usize,
            )
            .expect("Dynamic noise contracts"),
        ),
    }
}

fn terminal_value_bounds(profile: &SparseLwrPrfProfile, program: &LwrPrf) -> Vec<BigUint> {
    // The half-integer readout `2 f - 1` is at most `P - 1`.
    let bound = profile
        .canonical_readout()
        .expect("canonical readout")
        .into_iter()
        .max()
        .map(|f| BigUint::from(2 * f - 1))
        .expect("nonempty canonical readout");
    vec![bound; program.program().outputs().len()]
}

type SearchedFixture = (
    RnsBggParameters,
    DCRTPolyParams,
    SparseLwrPrfProfile,
    LwrPrf,
    Vec<PbcGeneratedKeyLayout>,
    RnsBggNoiseTrace,
);

/// Search the configured candidates, or check the recorded parameter set `recorded`.
fn search_fixture(sigma: &BigDecimal, recorded: Option<&ParameterRow>) -> SearchedFixture {
    let required_security = env_usize("MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS", 1)
        .expect("security bits configuration");
    assert!(required_security > 0, "MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS must be positive");
    let required_security = u64::try_from(required_security)
        .expect("MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS must fit in u64");
    let crt_bits = setting(recorded, "crt_bits", "MXX_EXPONENT_LUT_LWR_PRF_CRT_BITS", 40);
    assert!(crt_bits > 0, "MXX_EXPONENT_LUT_LWR_PRF_CRT_BITS must be positive");
    let expected_base_bits =
        setting(recorded, "base_bits", "MXX_EXPONENT_LUT_LWR_PRF_BASE_BITS", crt_bits.div_ceil(2));
    assert!(
        (1..=31).contains(&expected_base_bits),
        "MXX_EXPONENT_LUT_LWR_PRF_BASE_BITS must be in 1..=31"
    );
    let max_crt_depth = env_usize("MXX_EXPONENT_LUT_LWR_PRF_MAX_CRT_DEPTH", 32)
        .expect("maximum CRT depth configuration");
    assert!(max_crt_depth > 0, "MXX_EXPONENT_LUT_LWR_PRF_MAX_CRT_DEPTH must be positive");
    let operations_per_drop =
        setting(recorded, "operations_per_drop", "MXX_EXPONENT_LUT_LWR_PRF_OPERATIONS_PER_DROP", 1);
    // Each drop removes up to this many CRT primes.
    let max_primes_per_drop = env_usize("MXX_EXPONENT_LUT_LWR_PRF_MAX_PRIMES_PER_DROP", 4)
        .expect("maximum primes-per-drop configuration");
    assert!(
        max_primes_per_drop > 0,
        "MXX_EXPONENT_LUT_LWR_PRF_MAX_PRIMES_PER_DROP must be positive"
    );
    assert!(
        operations_per_drop > 0,
        "MXX_EXPONENT_LUT_LWR_PRF_OPERATIONS_PER_DROP must be positive"
    );
    let average = match recorded {
        Some(row) => match row["noise_model"].as_str() {
            "worst" => None,
            "average" => Some(AverageCaseConfig {
                failure_exponent: common::column(row, "failure_exponent"),
                allow_average_acceptance: true,
                ..AverageCaseConfig::default()
            }),
            model => panic!("recorded noise model {model} is neither worst nor average"),
        },
        None => noise_model("MXX_EXPONENT_LUT_LWR_PRF_").expect("noise-model configuration"),
    };
    // `MXX_EXPONENT_LUT_LWR_PRF_RHS_MODE=dynamic` binds the selectors only at
    // encryption. Recorded parameter sets, whose claims are exported, are
    // Static.
    let rhs_mode = rhs_mode("MXX_EXPONENT_LUT_LWR_PRF_").expect("RHS mode configuration");
    assert!(
        recorded.is_none() || rhs_mode == RhsMode::Static,
        "recorded Sparse-LWR parameter sets run the Static RHS lifecycle"
    );
    // `n,crt_depth,primes_per_drop` of a candidate already selected by a
    // full search. It replaces the search, and its lattice security estimate
    // is not repeated; the noise audit still builds its parameter chain.
    let recorded_selection = recorded.map(|row| {
        (
            common::column::<u32>(row, "ring_dimension"),
            common::column::<usize>(row, "crt_depth"),
            common::column::<usize>(row, "primes_per_drop"),
        )
    });
    let selected = recorded_selection.or_else(|| {
        env::var("MXX_EXPONENT_LUT_LWR_PRF_SELECTED_PARAMETERS").ok().map(|value| {
        let fields = value
            .split(',')
            .map(|field| field.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .expect(
                "MXX_EXPONENT_LUT_LWR_PRF_SELECTED_PARAMETERS must be n,crt_depth,primes_per_drop",
            );
        let [ring_dimension, crt_depth, primes_per_drop] = fields[..] else {
            panic!(
                "MXX_EXPONENT_LUT_LWR_PRF_SELECTED_PARAMETERS must be n,crt_depth,primes_per_drop"
            );
        };
        let ring_dimension =
            u32::try_from(ring_dimension).expect("selected LWR ring dimension does not fit u32");
        (ring_dimension, crt_depth, primes_per_drop)
    })
    });
    let candidates = match selected {
        Some((ring_dimension, _, _)) => vec![ring_dimension],
        None => ring_dimension_candidates(
            "MXX_EXPONENT_LUT_LWR_PRF_MIN_LOG_RING_DIMENSION",
            "MXX_EXPONENT_LUT_LWR_PRF_MAX_LOG_RING_DIMENSION",
            5,
            16,
        )
        .expect("ring-dimension configuration"),
    };
    let mut failures = Vec::new();
    for ring_dimension in candidates {
        let (profile, program, keys) = profile_and_program(ring_dimension, recorded);
        let noise = noise_bindings(&program, sigma, ring_dimension);
        let terminal_bounds = terminal_value_bounds(&profile, &program);
        let depth_candidates = match selected {
            Some((_, crt_depth, primes_per_drop)) => vec![(crt_depth, primes_per_drop)],
            None => (1..=max_crt_depth)
                .flat_map(|depth| {
                    (1..=max_primes_per_drop).map(move |primes_per_drop| (depth, primes_per_drop))
                })
                .collect::<Vec<_>>(),
        };
        // Candidates are independent, so they are audited in parallel; the
        // first accepted one in (CRT depth, primes per drop) order wins.
        let rejected = std::sync::Mutex::new(Vec::new());
        let accepted = depth_candidates.par_iter().find_map_first(|&(crt_depth, primes_per_drop)| {
            let reject = |reason: String| {
                rejected.lock().expect("rejection list").push(format!(
                    "ring dimension {ring_dimension}, CRT depth {crt_depth}, primes per drop {primes_per_drop}: {reason}"
                ));
                None
            };
            let request = RnsBggPlanRequest {
                ring_dimension,
                encoding_error_sigma: sigma.clone(),
                crt_bits,
                base_bits: expected_base_bits as u32,
                // A Dynamic chain keeps every level on one basis.
                terminal_depth: if rhs_mode == RhsMode::Dynamic { crt_depth } else { 1 },
                max_crt_depth: crt_depth,
                terminal_value_bound: terminal_bounds
                    .iter()
                    .max()
                    .cloned()
                    .expect("terminal bounds are nonempty"),
                operations_per_drop,
                primes_per_drop,
                rhs_mode,
            };
            let (parameters, audit_trace) = match audit_program_at_crt_depth(
                &request,
                program.program(),
                &noise,
                &terminal_bounds,
                average.as_ref(),
                0,
            ) {
                Ok(candidate) => candidate,
                Err(error) => return reject(error),
            };
            for (index, level) in parameters.levels().iter().enumerate() {
                assert_eq!(
                    level.base_bits(),
                    expected_base_bits as u32,
                    "selected LWR level {index} must use the configured base bits"
                );
            }
            let workload = match RnsBggDerivedWorkload::from_program_with_parameters(
                &parameters,
                program.program(),
            ) {
                Ok(workload) => workload,
                Err(error) => return reject(format!("workload validation failed: {error}")),
            };
            let trace = match api::noise_simulate(
                program.program(),
                RnsNoiseBuild::new(&parameters, program.program(), &workload, &noise)
                    .expect("validated candidate noise lowering state"),
                |_, outputs| Ok(outputs),
            ) {
                Ok(trace) => trace,
                Err(error) => return reject(format!("noise simulation failed: {error}")),
            };
            if trace.outputs.len() != program.program().outputs().len() {
                return reject(format!(
                    "noise simulation finalized {} outputs, expected {}",
                    trace.outputs.len(),
                    program.program().outputs().len()
                ));
            }
            let source = parameters.level(0).expect("source level").clone();
            let achieved = if selected.is_some() && recorded.is_none() {
                eprintln!(
                    "selected LWR parameters were supplied: the lattice security estimate is not repeated"
                );
                required_security
            } else {
                lattice_security_bits(&source, sigma)
                    .expect("lattice security estimator must be available")
            };
            if achieved < required_security {
                return reject(format!("security {achieved} < {required_security}"));
            }
            if let Some(row) = recorded {
                assert_eq!(
                    achieved,
                    common::column::<u64>(row, "security_bits"),
                    "the recorded security estimate must be reproduced"
                );
            }
            eprintln!(
                "LWR parameters accepted: ring_dimension={ring_dimension} crt_depth={crt_depth} primes_per_drop={primes_per_drop} security_bits={achieved} required_security_bits={required_security}"
            );
            Some((parameters, source, audit_trace))
        });
        failures.extend(rejected.into_inner().expect("rejection list"));
        if let Some((parameters, source, audit_trace)) = accepted {
            return (parameters, source, profile, program, keys, audit_trace);
        }
    }
    panic!(
        "no dynamic LWR parameter candidate was accepted ({}): {}",
        failures.len(),
        failures.join("; ")
    );
}

fn fixture(
    parameters: RnsBggParameters,
    poly_parameters: DCRTPolyParams,
    profile: SparseLwrPrfProfile,
    program: LwrPrf,
    keys: Vec<PbcGeneratedKeyLayout>,
    trace: RnsBggNoiseTrace,
    runtime: &mut impl api::StageExecution<api::StageStore>,
    store: &mut api::StageStore,
    graphs: &StageGraphs,
) -> Fixture {
    // The Static secrets are sampled on the stage runtime and stay resident
    // there.
    let dynamic = parameters.rhs_mode() == RhsMode::Dynamic;
    let payload = (!dynamic).then(|| {
        RnsEncryptBuild::new(&parameters, [7; 32])
            .sample_payload_secret(runtime, store, record(graphs, "payload"))
            .expect("payload secret")
    });
    let mask = (!dynamic).then(|| {
        RnsEncryptBuild::new(&parameters, [7; 32])
            .sample_mask_secret(runtime, store, record(graphs, "mask"))
            .expect("mask secret")
    });
    let one = DCRTPoly::const_rotate_poly(&poly_parameters, 0);
    let q = profile.q().expect("Q exponent") as u64;
    let a_exponents = (0..profile.lwr_dimension as u64)
        .map(|coordinate| (7 * coordinate + 3) % q)
        .collect::<Vec<_>>();
    // `a` is a public LHS family supplied at encrypt time. Convert each
    // exponent to its stride-aware signed monomial, including the
    // X^n = -1 negacyclic sign when the rotation crosses the boundary.
    // Each monomial is built from all n coefficients, so build them in
    // parallel.
    let a_values = a_exponents
        .par_iter()
        .map(|exponent| {
            let dimension = poly_parameters.ring_dimension() as usize;
            let rotation = (*exponent as usize * profile.layout.stride()) % (2 * dimension);
            let monomial = DCRTPoly::const_rotate_poly(&poly_parameters, rotation % dimension);
            if rotation >= dimension { -monomial } else { monomial }
        })
        .chain(rayon::iter::once(one.clone()))
        .collect::<Vec<_>>();

    // The accumulator starts at mono(0) = 1 and the family is `a` followed by
    // the encoding of zero read by dummy and padding cells. No private
    // plaintext is copied into the online evaluation call.
    let mut lhs = BTreeMap::new();
    for (wire, declaration) in program.program().encryption_inputs().inputs() {
        let value = match (declaration.knowledge(), declaration.shape()) {
            (AttributeKnowledge::PrivatePayload, InputBinding::Scalar(())) => {
                InputBinding::Scalar(one.clone())
            }
            (AttributeKnowledge::PublicPayload, InputBinding::Family(_)) => {
                InputBinding::Family(a_values.clone())
            }
            (knowledge, shape) => panic!("unexpected input declaration: {knowledge:?} {shape:?}"),
        };
        lhs.insert(*wire, value);
    }

    // Selectors are fixed private RHS plaintexts: each output's key selects
    // one cell per block of its layout.
    let rhs = program
        .prf
        .selector_plaintexts::<DCRTPoly>(&poly_parameters, &keys)
        .expect("selector plaintexts");

    Fixture { parameters, profile, program, payload, mask, rhs, lhs, trace, keys, a_exponents }
}

fn gpu_backend(parameters: &RnsBggParameters, devices: Vec<i32>) -> GpuDcrtBackend {
    let levels = parameters.distinct_levels().into_iter().map(|parameters| {
        GpuDCRTPolyParams::new(
            parameters.ring_dimension(),
            parameters.to_crt().0,
            parameters.base_bits(),
            None,
        )
    });
    gpu_backend_on(levels, devices)
}

/// Independent cleartext oracle for output `output`: `z = <a, k> mod Q`
/// from the key's PBC schedule, then the signed half-integer readout
/// `delta = (-1)^b (2 tau_r - 1)` for `z = r + b * Q/2`. It never reads the
/// production program graph.
fn lwr_oracle(profile: &SparseLwrPrfProfile, key: &PbcGeneratedKeyLayout, a: &[u64]) -> i64 {
    let q = profile.q().expect("Q exponent") as u64;
    let z = clear_pbc_inner_product(&key.public_layout, key.private_schedule(), a, q)
        .expect("clear PBC inner product") as usize;
    let half = q as usize / 2;
    let readout = profile.canonical_readout().expect("canonical readout");
    let magnitude = 2 * readout[z % half] as i64 - 1;
    if z < half { magnitude } else { -magnitude }
}

/// Stage storage selected by `MXX_EXPONENT_LUT_LWR_PRF_STORAGE`: `device`,
/// `host`, `file` (the default, in `directory`), or `file:<directory>`.
fn lwr_prf_storage(directory: &std::path::Path) -> api::StageStorage {
    match env::var("MXX_EXPONENT_LUT_LWR_PRF_STORAGE").as_deref() {
        Err(_) | Ok("file") => api::StageStorage::File(directory.into()),
        Ok("device") => api::StageStorage::Device,
        Ok("host") => api::StageStorage::Host,
        Ok(value) => match value.strip_prefix("file:") {
            Some(directory) => api::StageStorage::File(directory.into()),
            None => panic!("unknown Sparse-LWR stage storage {value}"),
        },
    }
}

#[test]
#[sequential(dcrt_runtime)]
fn lwr_prf_end_to_end() {
    // Progress of the parameter search and the stage plans is reported
    // through tracing; `RUST_LOG` overrides the default `info` filter.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let recorded = recorded_parameters();
    let row = recorded.as_ref().map(|(_, row)| row);
    let sigma = error_sigma(row);
    if env::var("MXX_EXPONENT_LUT_LWR_PRF_SIMULATION_ONLY").is_ok_and(|value| value == "1") {
        // Parameter search, noise simulation, and the security gate only.
        search_fixture(&sigma, row);
        return;
    }
    if export_only() {
        assert!(row.is_some(), "export-only mode needs MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID");
        let (parameters, poly_parameters, profile, program, keys, trace) =
            search_fixture(&sigma, row);
        let directory = tempfile::tempdir().expect("artifact directory");
        let store = api::StageStorage::File(directory.path().into()).open().expect("stage storage");
        let mut execution = GraphOnly(CpuExecution::new(mxx_backends::backend::poly::cpu_backend(
            parameters.levels().iter().cloned(),
        )));
        round_trip(
            parameters,
            poly_parameters,
            profile,
            program,
            keys,
            trace,
            &mut execution,
            store,
            recorded,
        );
        return;
    }
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        // This target is GPU-only; the sequential runtime lock is applied by
        // the test attribute when a device is available.
        eprintln!("skipping Sparse-LWR round trip: no CUDA device detected");
        return;
    }
    let (parameters, poly_parameters, profile, program, keys, trace) = search_fixture(&sigma, row);
    let directory = tempfile::tempdir().expect("artifact directory");
    let store = lwr_prf_storage(directory.path()).open().expect("stage storage");
    // `MXX_EXPONENT_LUT_LWR_PRF_EXECUTOR=cpu` runs the same stages on the CPU
    // executor, a reference for isolating a GPU runtime fault.
    if env::var("MXX_EXPONENT_LUT_LWR_PRF_EXECUTOR").is_ok_and(|value| value == "cpu") {
        let mut execution = mxx_backends::authority::CpuExecution::new(
            mxx_backends::backend::poly::cpu_backend(parameters.levels().iter().cloned()),
        );
        round_trip(
            parameters,
            poly_parameters,
            profile,
            program,
            keys,
            trace,
            &mut execution,
            store,
            recorded,
        );
    } else {
        let mut runtime =
            GpuRuntime::new(gpu_backend(&parameters, devices.clone())).expect("GPU runtime");
        round_trip(
            parameters,
            poly_parameters,
            profile,
            program,
            keys,
            trace,
            &mut runtime,
            store,
            recorded,
        );
    }
}

/// Preprocess, encrypt, and run online evaluation on `runtime`, then verify every
/// output against the clear PRF.
#[allow(clippy::too_many_arguments)]
fn round_trip(
    parameters: RnsBggParameters,
    poly_parameters: DCRTPolyParams,
    profile: SparseLwrPrfProfile,
    program: LwrPrf,
    keys: Vec<PbcGeneratedKeyLayout>,
    trace: RnsBggNoiseTrace,
    runtime: &mut impl api::StageExecution<api::StageStore>,
    mut store: api::StageStore,
    recorded: Option<(String, ParameterRow)>,
) {
    let graphs = StageGraphs::default();
    let Fixture { parameters, profile, program, payload, mask, rhs, lhs, trace, keys, a_exponents } =
        fixture(
            parameters,
            poly_parameters,
            profile,
            program,
            keys,
            trace,
            runtime,
            &mut store,
            &graphs,
        );
    assert!(!trace.wires.is_empty(), "noise simulation must visit the PRF gates");
    let observed = RefCell::new(Vec::<&'static str>::new());
    let observe = |name: &'static str| observed.borrow_mut().push(name);
    // Preprocess receives the private payload secret and fixed RHS selectors,
    // while encrypt receives public D attributes, a fresh mask, and only the
    // preparation artifacts.  Online evaluation receives artifacts/configuration only;
    // RHS plaintexts never cross this boundary.
    let dynamic = parameters.rhs_mode() == RhsMode::Dynamic;
    // A Dynamic Program is compiled against a setup, here its own with the
    // identity mapping; encryption then needs the setup only.
    let setup = dynamic.then(|| {
        RnsBggDynamicSetup::for_program(
            &parameters,
            [7; 32],
            HASH_KEY,
            program.program(),
            covering_automorphism_layout(program.program()),
            RnsBggDynamicDistributions::uniform(parameters.encoding_contract()),
        )
        .expect("Dynamic setup")
    });
    let no_rhs = BTreeMap::new();
    let mut preprocess_build = RnsPreprocessBuild::new(&parameters, [7; 32]);
    let mut encrypt_build = RnsEncryptBuild::new(&parameters, [7; 32]);
    let mut online_build = RnsOnlineEvalBuild::new(&parameters, [7; 32]);
    if let Some((setup, binding)) = &setup {
        preprocess_build =
            preprocess_build.with_dynamic_setup(setup, binding.clone()).expect("setup binding");
        encrypt_build = encrypt_build.with_dynamic_setup(setup).expect("setup binding");
        online_build =
            online_build.with_dynamic_setup(setup, binding.clone()).expect("setup binding");
    }
    let (preprocess, online_eval) = api::preprocess(
        program.program(),
        &ParamEnv::default(),
        if dynamic { &no_rhs } else { &rhs },
        payload.as_ref(),
        HASH_KEY,
        preprocess_build,
        runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |validated| {
            observe("preprocess");
            record(&graphs, "preprocess")(validated)
        },
    )
    .expect("RNS preprocessing");
    // The single Dynamic secret is chosen only after preprocessing, and the
    // selectors are bound with a fresh batch at encryption.
    let payload = payload.unwrap_or_else(|| {
        RnsEncryptBuild::new(&parameters, [7; 32])
            .sample_payload_secret(runtime, &mut store, record(&graphs, "payload"))
            .expect("payload secret")
    });
    let encrypted = api::encrypt(
        &setup.as_ref().map_or_else(
            || program.program().encryption_inputs(),
            |(setup, _)| setup.encryption_inputs(),
        ),
        &ParamEnv::default(),
        &lhs,
        (!dynamic).then_some(&preprocess),
        &payload,
        mask.as_ref(),
        HASH_KEY,
        encrypt_build,
        runtime,
        &mut store,
        |build| {
            if let Some((_, binding)) = &setup {
                let rhs = rhs
                    .iter()
                    .map(|(id, value)| {
                        let coordinate = binding
                            .rhs(*id)
                            .map_err(|error| api::ApiError::Input { message: error.to_string() })?;
                        Ok((coordinate, value.clone()))
                    })
                    .collect::<Result<BTreeMap<_, _>, api::ApiError>>()?;
                build.bind_rhs_inputs(&ParamEnv::default(), &rhs, rand::random())?;
            }
            Ok(())
        },
        |validated| {
            observe("encrypt");
            record(&graphs, "encrypt")(validated)
        },
    )
    .expect("RNS encryption");
    // A Static online evaluation receives the Program without its LUT
    // values; a Dynamic one applies the tables as public actions.
    let evaluated =
        if dynamic { program.program().clone() } else { program.program().without_lut_values() };
    let result = api::online_eval(
        &evaluated,
        &ParamEnv::default(),
        &online_eval,
        &encrypted,
        HASH_KEY,
        online_build,
        runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |validated| {
            observe("online_eval");
            record(&graphs, "online_eval")(validated)
        },
    )
    .expect("RNS online evaluation");
    // Each of the three actual production graphs was planned (and its
    // candidates measured) by the GPU runtime immediately before execution,
    // so the workflow remains simulation -> preprocess -> encrypt -> online evaluation
    // evaluation rather than a test-only reconstruction.
    assert_eq!(
        observed.borrow().as_slice(),
        ["preprocess", "encrypt", "online_eval"],
        "observers must see the exact three production graphs"
    );
    // In estimate mode (`MXX_EXPONENT_LUT_ESTIMATE_ONLY=1`) the GPU stages
    // were only planned, so there are no outputs to verify.
    if env::var("MXX_EXPONENT_LUT_ESTIMATE_ONLY").is_ok_and(|value| value == "1") {
        eprintln!("estimate mode: every stage was planned; outputs are not verified");
        return;
    }
    if export_only() {
        // No stage was evaluated: build the decryption graph on placeholder online evaluation
        // outputs, record it, and export the claim.
        let placeholder = api::StageOutputs {
            outputs: [1, 3, 5]
                .into_iter()
                .map(|port| (format!("output-0.{port}"), RuntimeValue::Int(BigInt::from(0))))
                .collect(),
        };
        let decoded = decode_output(
            program.program(),
            &placeholder,
            0,
            &parameters,
            &payload,
            mask.as_ref(),
            RnsOutputUse::DirectDecode,
            &[0],
            runtime,
            &mut store,
            record(&graphs, "decode"),
        );
        assert!(decoded.is_err(), "the export-only decryption graph is not executed");
        let (id, row) = recorded.expect("export-only mode exports a recorded parameter set");
        export_claim(
            &id,
            &row,
            &profile,
            &program,
            graphs,
            &payload,
            mask.as_ref().expect("recorded parameter sets are Static"),
            (preprocess.production().clone(), encrypted.production().clone()),
        );
        return;
    }
    // Compute every output's expected readout independently from `a` and
    // its key. The coefficient LUT returns the signed half-integer readout
    // `delta` as the constant polynomial, finalized to the scale `k` of its
    // decryption modulus `2P`, so verification decides `delta` modulo `2P`;
    // the expected plaintext is built in the source ring with `delta` in
    // coefficient 0 only, not decoded from the returned carrier.
    let source_parameters = parameters.level(0).expect("LWR source parameters").clone();
    for (output, key) in keys.iter().enumerate() {
        let delta = lwr_oracle(&profile, key, &a_exponents);
        let readout = u32::try_from(delta.unsigned_abs()).expect("readout fits a DCRT coefficient");
        let mut coefficients = vec![0; source_parameters.ring_dimension() as usize];
        coefficients[0] = readout;
        let expected = DCRTPoly::from_u32s(&source_parameters, &coefficients);
        let expected = if delta < 0 { -expected } else { expected };
        verify_output(
            program.program(),
            &result,
            output,
            &expected,
            &parameters,
            &payload,
            mask.as_ref(),
            RnsOutputUse::DirectDecode,
            runtime,
            &mut store,
        )
        .expect("LWR application verification");
    }
    assert!(!trace.outputs.is_empty(), "noise trace must finalize declared output");
    assert!(!result.outputs.is_empty(), "online evaluation must produce output");

    // Decryption of the first output: coefficient 0 of the terminal plaintext `delta`,
    // threshold-decoded modulo 2P, is `delta mod 2P`.
    let delta = lwr_oracle(&profile, &keys[0], &a_exponents);
    let decoded = decode_output(
        program.program(),
        &result,
        0,
        &parameters,
        &payload,
        mask.as_ref(),
        RnsOutputUse::DirectDecode,
        &[0],
        runtime,
        &mut store,
        record(&graphs, "decode"),
    )
    .expect("LWR output decryption");
    assert_eq!(
        decoded,
        vec![BigInt::from(delta.rem_euclid(2 * profile.p() as i64))],
        "decrypted LWR output modulo 2P"
    );

    if let Some((id, row)) = recorded {
        export_claim(
            &id,
            &row,
            &profile,
            &program,
            graphs,
            &payload,
            mask.as_ref().expect("recorded parameter sets are Static"),
            (preprocess.production().clone(), encrypted.production().clone()),
        );
    }
}

/// Export the claim of the recorded stage graphs to `lean/lwr_prf/<id>/generated`.
#[allow(clippy::too_many_arguments)]
fn export_claim(
    id: &str,
    row: &ParameterRow,
    profile: &SparseLwrPrfProfile,
    program: &LwrPrf,
    graphs: StageGraphs,
    payload: &mxx_exponent_lut::rns_bgg::RnsBggPayloadSecret,
    mask: &mxx_exponent_lut::rns_bgg::RnsBggMaskSecret,
    productions: (mxx_ir_core::artifact::ProductionId, mxx_ir_core::artifact::ProductionId),
) {
    let directory = common::generated_directory("lwr_prf", id);
    let protocol =
        lwr_protocol(profile, program, graphs.into_inner(), payload, mask, productions, row);
    mxx_ir_core::lean::protocol::export(&protocol, &directory, false).expect("LWR claim export");
    eprintln!("exported the LWR correctness claim to {}", directory.display());
}

/// The executed LWR workflow of output 0 as a closed protocol: payload and mask sampling,
/// preprocessing of the one-hot selectors, encryption of the state `X^0` and the public vector
/// `a`, online evaluation, and decryption of coefficient 0 of the output modulo `2P`, compared
/// with the clear half-integer readout `delta(<a, k> mod Q) mod 2P` of the selected cells.
/// External inputs are the
/// hash key, `a`, its plaintext monomials `±X^e` (a requirement ties `±X^e = X^(stride a)`), the
/// selector bits of every block (a requirement makes each block one-hot), and the fixed unit
/// coefficient and zero exponent of the state and of the selectors.
fn lwr_protocol(
    profile: &SparseLwrPrfProfile,
    program: &LwrPrf,
    graphs: BTreeMap<&'static str, Graph>,
    payload: &mxx_exponent_lut::rns_bgg::RnsBggPayloadSecret,
    mask: &mxx_exponent_lut::rns_bgg::RnsBggMaskSecret,
    (preprocess, encrypt): (
        mxx_ir_core::artifact::ProductionId,
        mxx_ir_core::artifact::ProductionId,
    ),
    row: &ParameterRow,
) -> ProtocolDecl {
    let id = |name: &str| StageId(name.to_owned());
    let productions = BTreeMap::from([(preprocess, id("preprocess")), (encrypt, id("encrypt"))]);
    let payload_name = payload.input_name().expect("sampled payload").to_owned();
    let mask_name = mask.input_name().expect("sampled mask").to_owned();
    let secret = |stage: &str| (id(stage), "secret".to_owned());
    let mut links =
        BTreeMap::from([(payload_name, secret("payload")), (mask_name, secret("mask"))]);
    for (input, port) in [("decode-key-01", 1), ("decode-key-11", 3), ("decode-row-1", 5)] {
        links.insert(input.to_owned(), (id("online_eval"), format!("output-0.{port}")));
    }
    let graph = |name: &str| &graphs[name];
    let stages = ["payload", "mask", "preprocess", "encrypt", "online_eval", "decode"]
        .into_iter()
        .map(|name| common::stage(name, graph(name), &productions, &links))
        .collect::<Vec<_>>();

    let ring_dimension = profile.ring_dimension().expect("LWR ring dimension");
    let q = profile.q().expect("LWR input modulus");
    let width = profile.native_width();
    let stride = profile.layout.stride();
    let members = profile.lwr_dimension + 1;
    let family = |count: usize| FamilyType {
        element: IntType,
        count: IntExpr::constant(count),
        member_slot: None,
    };
    // Block `j` of output 0 reads coordinate `cells[j][c]` of `a`; dummy and padding cells read
    // the last member.
    assert_eq!(program.prf.members(), 1, "the exported claim covers one key");
    let blocks = program.prf.layouts()[0]
        .cells
        .iter()
        .map(|cells| {
            cells
                .iter()
                .map(|cell| match cell {
                    PbcCell::Real { coordinate, .. } => *coordinate,
                    PbcCell::Dummy | PbcCell::Padding => profile.lwr_dimension,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let selector_families = program.prf.selector_families();
    assert_eq!(selector_families.len(), blocks.len(), "one selector family per block");

    // The ideal output: `delta mod 2P` for the half-integer readout
    // `delta = (-1)^b (2 tau(r) - 1)`, `tau(r) = floor(p' r / (Q/2)) + 1`, of
    // `z mod Q = r + b Q/2` for `z = sum_j a[cells[j][selected_j]]`. Block `j` adds `selector_j[c]
    // a[cells[j][c]]` over its cells `c` in a loop; the cell coordinate is selected from the
    // constant table of the block.
    let block_sum = |selector: &Family<Int>, a: &Family<Int>, coordinates: &[usize]| {
        mxx_dsl::iterate(coordinates.len(), Int::constant(0), |cell, total| {
            let coordinate = mxx_dsl::select(
                cell.clone(),
                coordinates.iter().map(|coordinate| Int::constant(*coordinate as i64)).collect(),
            )?;
            Ok(total.add(selector.at(cell).mul(a.at(coordinate))))
        })
        .expect("block sum loop")
    };
    let ideal = DslContext::new("lwr-ideal");
    let a = ideal.input::<Family<Int>>("a", family(members)).expect("ideal a");
    let mut z = Int::constant(0);
    for (block, coordinates) in blocks.iter().enumerate() {
        let selector = ideal
            .input::<Family<Int>>(format!("selector_{block}"), family(coordinates.len()))
            .expect("ideal selector");
        z = z.add(block_sum(&selector, &a, coordinates));
    }
    // `delta mod 2P` is `m` for `b = 0` and `2P - m` for `b = 1`, with the
    // magnitude `m = 2 tau(r) - 1` in `[1, P - 1]`.
    let p = profile.p() as i64;
    let magnitude =
        z.clone().rem(width).mul(profile.p_prime() as i64).div(width).add(1).mul(2).sub(1);
    let negative = z.rem(2 * width).div(width);
    let value = magnitude.clone().add(negative.mul(Int::constant(2 * p).sub(magnitude.mul(2))));
    let ideal =
        IdealSpec::new(ideal.output("value", value).unwrap().build().unwrap().graph).unwrap();

    // The requirement: `±X^e = X^(stride a)` member by member, and one-hot selectors, each
    // checked in a loop.
    let requirement = DslContext::new("lwr-inputs");
    let a = requirement.input::<Family<Int>>("a", family(members)).unwrap();
    let coefficient = requirement.input::<Family<Int>>("a_coefficient", family(members)).unwrap();
    let exponent = requirement.input::<Family<Int>>("a_exponent", family(members)).unwrap();
    let mut valid = mxx_dsl::iterate(members, Int::constant(0).equal(0), |member, valid| {
        let sign = coefficient.at(member.clone());
        // `2 e + n (1 - c) = 2 stride a` with `c = ±1`.
        Ok(valid &
            sign.clone().mul(sign.clone()).equal(1) &
            exponent
                .at(member.clone())
                .mul(2)
                .add(Int::constant(1).sub(sign).mul(ring_dimension))
                .equal(a.at(member).mul(2 * stride)))
    })
    .unwrap();
    for (block, coordinates) in blocks.iter().enumerate() {
        let selector = requirement
            .input::<Family<Int>>(format!("selector_{block}"), family(coordinates.len()))
            .unwrap();
        let total = mxx_dsl::iterate(coordinates.len(), Int::constant(0), |cell, total| {
            Ok(total.add(selector.at(cell)))
        })
        .unwrap();
        valid = valid & total.equal(1);
    }
    let requirement =
        PurePredicateSpec::new(requirement.output("valid", valid).unwrap().build().unwrap().graph)
            .unwrap();

    let only = |stage: &str, suffix: String| -> String {
        let [name] = <[String; 1]>::try_from(common::inputs_ending_with(graph(stage), &suffix))
            .unwrap_or_else(|_| panic!("{stage} must declare one input ending with {suffix}"));
        name
    };
    let range = |lower: i64, upper: i64| InputValueContract::IntegerRange {
        lower: lower.into(),
        upper: upper.into(),
    };
    let family_range = |count: usize, lower: i64, upper: i64| InputValueContract::Family {
        count: IntExpr::constant(count),
        element: Box::new(range(lower, upper)),
    };
    let encrypt_input = |wire, field: &str| {
        common::to_stage("encrypt", &only("encrypt", format!("/input/{wire:?}.{field}")))
    };
    let requirement_input =
        |input: &str| ProtocolInputDestination::Requirement { requirement: 0, input: input.into() };
    let hash_key = ["preprocess", "encrypt", "online_eval"]
        .into_iter()
        .filter(|stage| !common::inputs_ending_with(graph(stage), "rns-bgg/hash-key").is_empty())
        .map(|stage| common::to_stage(stage, "rns-bgg/hash-key"))
        .collect();
    let state = program.prf.initial_input();
    let vector = program.prf.input_vector();
    let mut externals = vec![
        common::external("hash_key", InputValueContract::Bytes { length: 32.into() }, hash_key),
        common::external(
            "unit_coefficient",
            range(1, 1),
            vec![encrypt_input(state, "coefficient")],
        ),
        common::external("state_exponent", range(0, 0), vec![encrypt_input(state, "exponent")]),
        common::external(
            "a",
            family_range(members, 0, q as i64 - 1),
            vec![ProtocolInputDestination::Ideal { input: "a".into() }, requirement_input("a")],
        ),
        common::external(
            "a_coefficient",
            family_range(members, -1, 1),
            vec![encrypt_input(vector, "coefficient"), requirement_input("a_coefficient")],
        ),
        common::external(
            "a_exponent",
            family_range(members, 0, ring_dimension as i64 - 1),
            vec![encrypt_input(vector, "exponent"), requirement_input("a_exponent")],
        ),
    ];
    for (block, family) in selector_families.iter().enumerate() {
        let rhs = |field: &str| {
            common::to_stage(
                "preprocess",
                &only("preprocess", format!("/rhs/{:?}.{field}", family.rhs)),
            )
        };
        let name = format!("selector_{block}");
        externals.push(common::external(
            &name,
            family_range(family.members, 0, 1),
            vec![
                rhs("coefficient"),
                ProtocolInputDestination::Ideal { input: name.clone() },
                requirement_input(&name),
            ],
        ));
        externals.push(common::external(
            &format!("selector_exponent_{block}"),
            family_range(family.members, 0, 0),
            vec![rhs("exponent")],
        ));
    }
    let (contracts, bindings): (Vec<_>, Vec<_>) = externals.into_iter().unzip();

    // The decrypted coefficient 0 of the output, `decoded-0`, must equal the ideal value.
    let endpoint = EndpointSpecId::Exact;
    let failure_probability_log2 = match row["noise_model"].as_str() {
        "worst" => None,
        _ => Some(common::column(row, "failure_exponent")),
    };
    ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bindings: Default::default(),
        failure_probability_log2,
        bundle: ClosedProtocolBundle {
            workflow: Workflow { stages, entrypoint: id("decode") },
            ideal,
            requirements: vec![requirement],
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decoded-0".to_owned(),
                    ideal_input: "value".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef {
                        stage: id("decode"),
                        output: "decoded-0".to_owned(),
                    },
                    ideal_output: "value".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings: bindings,
            precondition_spec: ProtocolPreconditionSpec {
                requirement_outputs: vec!["valid".to_owned()],
            },
        },
    })
    .expect("valid LWR protocol")
}
