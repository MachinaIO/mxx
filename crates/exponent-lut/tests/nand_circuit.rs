#![cfg(feature = "gpu")]

//! Public RNS-BGG NAND application workflow.
//!
//! With `MXX_EXPONENT_LUT_NAND_PARAMETER_ID`, the workflow runs the recorded parameter set of
//! `lean/nand_circuit/parameters.csv` and exports the Lean correctness claim of the executed
//! graphs to `lean/nand_circuit/<parameter id>/generated`.

mod common;

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime,
    artifact::MemoryArtifactStore,
    backend::poly_gpu::gpu_backend_on,
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_dsl::{DslContext, IdealSpec, Int, IntType};
use mxx_exponent_lut::{
    ExpLayout, NandCircuit, api,
    noise_refresh::simulation::{AverageCaseConfig, NoiseModelKind},
    program::Operand,
    rns_bgg::{
        RhsMode, RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseRhs, RnsBggNoiseValue,
        RnsBggParameters, RnsBggPayloadSecret, RnsBggPlanRequest, RnsEncryptBuild, RnsNoiseBuild,
        RnsOnlineEvalBuild, RnsPreprocessBuild, decode_output, verify_output,
    },
    utils::{
        audit_program_at_crt_depth, env_usize, lattice_security_bits, noise_model, rhs_mode,
        ring_dimension_candidates,
    },
};
use mxx_ir_core::{
    Graph, ParamEnv,
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputValueContract, OutputRef, ProtocolDecl, ProtocolInputDestination,
        ProtocolPreconditionSpec, PurePredicateSpec, StageId, Workflow,
    },
};
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, env};

type ParameterRow = BTreeMap<String, String>;

const HASH_KEY: &[u8; 32] = b"nand-rns-public-api-hash-key-v01";
const NAND_DEPTH_ENV: &str = "MXX_EXPONENT_LUT_NAND_DEPTH";

/// The recorded parameter set selected by `MXX_EXPONENT_LUT_NAND_PARAMETER_ID`, if any.
fn recorded_parameters() -> Option<(String, ParameterRow)> {
    env::var("MXX_EXPONENT_LUT_NAND_PARAMETER_ID")
        .ok()
        .map(|id| (id.clone(), common::parameter_row("nand_circuit", &id)))
}

fn nand_fixture(recorded: Option<&ParameterRow>) -> (NandCircuit, usize, Vec<bool>) {
    let depth = match recorded {
        Some(row) => common::column(row, "nand_depth"),
        None => env::var(NAND_DEPTH_ENV)
            .unwrap_or_else(|_| "1".to_owned())
            .parse::<u32>()
            .unwrap_or_else(|error| {
                panic!("{NAND_DEPTH_ENV} must be a non-negative integer: {error}")
            }),
    };
    let leaves = 1usize
        .checked_shl(depth)
        .unwrap_or_else(|| panic!("{NAND_DEPTH_ENV} is too large for this platform: {depth}"));
    let gate_count =
        leaves.checked_sub(1).expect("a full binary NAND tree always has at least one leaf");
    leaves
        .checked_add(gate_count)
        .unwrap_or_else(|| panic!("{NAND_DEPTH_ENV} overflows the circuit node count: {depth}"));
    let max_lut_rounds = 4usize
        .checked_pow(depth)
        .unwrap_or_else(|| panic!("{NAND_DEPTH_ENV} overflows the LUT-round count: {depth}"));
    let max_lut_budget = env_usize("MXX_EXPONENT_LUT_NAND_MAX_LUT_ROUNDS", 4096)
        .unwrap_or_else(|error| panic!("invalid NAND LUT budget: {error}"));
    assert!(
        max_lut_rounds <= max_lut_budget,
        "{NAND_DEPTH_ENV} requires {max_lut_rounds} LUT rounds, exceeding MXX_EXPONENT_LUT_NAND_MAX_LUT_ROUNDS={max_lut_budget}"
    );

    // Build each level from the previous level's node indices.  Depth zero is
    // the coherent identity case: one leaf, no gates, and that leaf as output.
    let mut frontier = (0..leaves).collect::<Vec<_>>();
    let mut gates = Vec::with_capacity(gate_count);
    for _ in 0..depth {
        let mut next = Vec::with_capacity(frontier.len() / 2);
        for pair in frontier.chunks_exact(2) {
            gates.push([pair[0], pair[1]]);
            let node =
                leaves.checked_add(gates.len() - 1).expect("full NAND tree node index overflow");
            next.push(node);
        }
        frontier = next;
    }
    assert_eq!(frontier.len(), 1, "full NAND tree must have one output");
    let circuit = NandCircuit::new(leaves, gates, frontier[0]).expect("valid NAND tree");
    assert_eq!(circuit.input_count(), leaves);
    assert_eq!(circuit.gates().len(), gate_count);

    // One round trip takes minutes at production parameters, so each run
    // checks a single assignment whose every input is drawn at random.
    let assignment = (0..leaves).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    eprintln!("NAND input assignment: {assignment:?}");
    (circuit, max_lut_rounds, assignment)
}

fn gpu_parameters(parameters: &RnsBggParameters) -> Vec<GpuDCRTPolyParams> {
    parameters
        .distinct_levels()
        .into_iter()
        .map(|level| {
            GpuDCRTPolyParams::new(
                level.ring_dimension(),
                level.to_crt().0,
                level.base_bits(),
                None,
            )
        })
        .collect()
}

fn noise_bindings(
    compiled: &mxx_exponent_lut::CompiledNandCircuit,
    sigma: &BigDecimal,
) -> RnsBggNoiseBindings {
    let input_value =
        RnsBggNoiseValue::sampled_encoding_sigma(sigma, b"nand-input", &[BigInt::from(1u8)])
            .expect("valid NAND encoding sigma");
    let rhs_value = RnsBggNoiseRhs { payload_l1: 1u8.into(), payload_l2_squared: 1u8.into() };
    RnsBggNoiseBindings::from_inputs(
        compiled.program(),
        &[input_value],
        &vec![rhs_value; compiled.rhs_ids().len()],
    )
    .expect("NAND noise bindings match the public declarations")
}

/// Search the configured candidates, or check the recorded parameter set `recorded`, and
/// return the accepted parameters with the audited output bound.
fn select_candidate(
    circuit: &NandCircuit,
    layout: ExpLayout,
    max_lut_rounds: usize,
    recorded: Option<&ParameterRow>,
) -> (DCRTPolyParams, RnsBggParameters, mxx_exponent_lut::CompiledNandCircuit, BigUint) {
    let sigma = match recorded {
        Some(row) => common::column(row, "error_sigma"),
        None => env::var("MXX_EXPONENT_LUT_NAND_ERROR_SIGMA")
            .unwrap_or_else(|_| "4".to_owned())
            .parse::<BigDecimal>()
            .unwrap_or_else(|error| {
                panic!("MXX_EXPONENT_LUT_NAND_ERROR_SIGMA must be a nonnegative decimal: {error}")
            }),
    };
    assert!(
        sigma >= BigDecimal::from(0u8),
        "MXX_EXPONENT_LUT_NAND_ERROR_SIGMA must be nonnegative"
    );
    let crt_bits = match recorded {
        Some(row) => common::column(row, "crt_bits"),
        None => env_usize("MXX_EXPONENT_LUT_NAND_CRT_BITS", 40)
            .unwrap_or_else(|error| panic!("invalid NAND CRT bits: {error}")),
    };
    let max_crt_depth = env_usize("MXX_EXPONENT_LUT_NAND_MAX_CRT_DEPTH", 32)
        .unwrap_or_else(|error| panic!("invalid NAND maximum CRT depth: {error}"));
    let operations_per_drop = match recorded {
        Some(row) => common::column(row, "operations_per_drop"),
        None => env_usize("MXX_EXPONENT_LUT_NAND_OPERATIONS_PER_DROP", 2)
            .unwrap_or_else(|error| panic!("invalid NAND operations-per-drop: {error}")),
    };
    // Each drop removes up to this many CRT primes.
    let max_primes_per_drop = env_usize("MXX_EXPONENT_LUT_NAND_MAX_PRIMES_PER_DROP", 4)
        .unwrap_or_else(|error| panic!("invalid NAND maximum primes-per-drop: {error}"));
    assert!(max_primes_per_drop > 0, "MXX_EXPONENT_LUT_NAND_MAX_PRIMES_PER_DROP must be positive");
    let security_target = env_usize("MXX_EXPONENT_LUT_NAND_SECURITY_BITS", 1)
        .unwrap_or_else(|error| panic!("invalid NAND security target: {error}"));
    assert!(security_target > 0, "MXX_EXPONENT_LUT_NAND_SECURITY_BITS must be positive");
    let security_target = u64::try_from(security_target)
        .unwrap_or_else(|_| panic!("NAND security target does not fit u64: {security_target}"));
    assert!(crt_bits > 0, "MXX_EXPONENT_LUT_NAND_CRT_BITS must be positive");
    assert!(max_crt_depth > 0, "MXX_EXPONENT_LUT_NAND_MAX_CRT_DEPTH must be positive");
    assert!(operations_per_drop > 0, "MXX_EXPONENT_LUT_NAND_OPERATIONS_PER_DROP must be positive");
    let expected_base_bits = crt_bits.div_ceil(2);
    assert!(
        expected_base_bits <= 31,
        "MXX_EXPONENT_LUT_NAND_CRT_BITS is unsupported above 62 bits"
    );
    let expected_base_bits = expected_base_bits as u32;
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
        None => noise_model("MXX_EXPONENT_LUT_NAND_")
            .unwrap_or_else(|error| panic!("invalid NAND noise model: {error}")),
    };
    if let Some(row) = recorded {
        assert_eq!(common::column::<u32>(row, "base_bits"), expected_base_bits);
    }
    // `MXX_EXPONENT_LUT_NAND_RHS_MODE=dynamic`, or a recorded row's
    // `rhs_mode`, chooses the RHS values only at encryption.
    let rhs_mode = match recorded {
        Some(row) => match row["rhs_mode"].as_str() {
            "static" => RhsMode::Static,
            "dynamic" => RhsMode::Dynamic,
            mode => panic!("recorded RHS mode {mode} is neither static nor dynamic"),
        },
        None => rhs_mode("MXX_EXPONENT_LUT_NAND_")
            .unwrap_or_else(|error| panic!("invalid NAND RHS mode: {error}")),
    };
    let expected_acceptance = match average.as_ref() {
        Some(_) => NoiseModelKind::AverageCase,
        None => NoiseModelKind::WorstCase,
    };
    // `n,crt_depth,primes_per_drop` of a candidate already selected by a
    // full search. It replaces the search, and its lattice security estimate
    // is not repeated; the noise audit still builds its parameter chain.
    let selected = match recorded {
        Some(row) => Some((
            common::column::<u32>(row, "ring_dimension"),
            common::column::<usize>(row, "crt_depth"),
            common::column::<usize>(row, "primes_per_drop"),
        )),
        None => env::var("MXX_EXPONENT_LUT_NAND_SELECTED_PARAMETERS").ok().map(|value| {
        let fields = value
            .split(',')
            .map(|field| field.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|error| {
                panic!("MXX_EXPONENT_LUT_NAND_SELECTED_PARAMETERS must be n,crt_depth,primes_per_drop: {error}")
            });
        let [ring_dimension, crt_depth, primes_per_drop] = fields[..] else {
            panic!("MXX_EXPONENT_LUT_NAND_SELECTED_PARAMETERS must be n,crt_depth,primes_per_drop");
        };
        let ring_dimension = u32::try_from(ring_dimension)
            .unwrap_or_else(|_| panic!("selected NAND ring dimension does not fit u32"));
        (ring_dimension, crt_depth, primes_per_drop)
        }),
    };
    let ring_dimensions = match selected {
        Some((ring_dimension, _, _)) => vec![ring_dimension],
        None => ring_dimension_candidates(
            "MXX_EXPONENT_LUT_NAND_MIN_LOG_RING_DIMENSION",
            "MXX_EXPONENT_LUT_NAND_MAX_LOG_RING_DIMENSION",
            5,
            16,
        )
        .unwrap_or_else(|error| panic!("invalid NAND ring-dimension range: {error}")),
    };

    for ring_dimension in ring_dimensions {
        let compiled =
            circuit.compile(layout, ring_dimension as usize, max_lut_rounds).unwrap_or_else(
                |error| panic!("NAND compilation failed for n={ring_dimension}: {error}"),
            );
        let bindings = noise_bindings(&compiled, &sigma);
        let terminal_bounds = vec![BigUint::from(1u8); compiled.program().outputs().len()];
        let depth_candidates: Vec<(usize, usize)> = match selected {
            Some((_, crt_depth, primes_per_drop)) => vec![(crt_depth, primes_per_drop)],
            None => (1..=max_crt_depth)
                .flat_map(|depth| {
                    (1..=max_primes_per_drop).map(move |primes_per_drop| (depth, primes_per_drop))
                })
                .collect(),
        };
        for (crt_depth, primes_per_drop) in depth_candidates {
            let request = RnsBggPlanRequest {
                ring_dimension,
                encoding_error_sigma: sigma.clone(),
                crt_bits,
                base_bits: expected_base_bits,
                terminal_depth: 1,
                max_crt_depth: crt_depth,
                terminal_value_bound: BigUint::from(1u8),
                operations_per_drop,
                primes_per_drop,
                rhs_mode,
            };
            let Ok((parameters, audit_trace)) = audit_program_at_crt_depth(
                &request,
                compiled.program(),
                &bindings,
                &terminal_bounds,
                average.as_ref(),
                0,
            ) else {
                continue;
            };
            assert_eq!(audit_trace.acceptance, Some(expected_acceptance));
            if parameters.levels().iter().any(|level| level.base_bits() != expected_base_bits) {
                continue;
            }
            let workload = RnsBggDerivedWorkload::from_program_with_parameters(
                &parameters,
                compiled.program(),
            )
            .unwrap_or_else(|error| {
                panic!("NAND workload validation failed for n={ring_dimension}, CRT depth {crt_depth}: {error}")
            });
            let trace = api::noise_simulate(
                compiled.program(),
                RnsNoiseBuild::new(&parameters, compiled.program(), &workload, &bindings)
                    .unwrap_or_else(|error| {
                        panic!("NAND noise setup failed for n={ring_dimension}, CRT depth {crt_depth}: {error}")
                    }),
                |_, outputs| Ok(outputs),
            )
            .unwrap_or_else(|error| {
                panic!("NAND noise simulation failed for n={ring_dimension}, CRT depth {crt_depth}: {error}")
            });
            if trace.outputs.is_empty() || trace.outputs.len() != compiled.program().outputs().len()
            {
                continue;
            }
            if average.is_none() {
                let outputs_fit = trace.outputs.iter().enumerate().all(|(index, value)| {
                    let wire = match &compiled.program().outputs()[index] {
                        Operand::Scalar(wire) | Operand::Member { family: wire, .. } => *wire,
                    };
                    let level = workload
                        .level(wire)
                        .unwrap_or_else(|error| panic!("NAND output level lookup failed: {error}"));
                    parameters
                        .noise_fits(level, &value.bound)
                        .unwrap_or_else(|error| panic!("NAND output noise check failed: {error}"))
                });
                if !outputs_fit {
                    continue;
                }
            }
            let source = parameters.level(0).expect("selected source level");
            // A zero sigma is a noise-free correctness diagnostic: without
            // error there is no lattice security to estimate.
            let achieved = if sigma == BigDecimal::from(0u8) {
                eprintln!("noise-free NAND diagnostic: the lattice security gate is skipped");
                security_target
            } else if selected.is_some() && recorded.is_none() {
                eprintln!(
                    "selected NAND parameters were supplied: the lattice security estimate is not repeated"
                );
                security_target
            } else {
                lattice_security_bits(source, &sigma).unwrap_or_else(|error| {
                    panic!("NAND security estimation failed for n={ring_dimension}, CRT depth {crt_depth}: {error}")
                })
            };
            if achieved < security_target {
                continue;
            }
            if let Some(row) = recorded {
                assert_eq!(
                    achieved,
                    common::column::<u64>(row, "security_bits"),
                    "the recorded security estimate must be reproduced"
                );
            }
            eprintln!(
                "selected NAND parameters: n={ring_dimension}, crt_depth={crt_depth}, primes_per_drop={primes_per_drop}, security_bits={achieved}, required_security_bits={security_target}"
            );
            return (
                source.clone(),
                parameters,
                compiled,
                audit_trace.outputs[0].bound.error.clone(),
            );
        }
    }
    panic!(
        "no NAND candidate satisfies the noise and security contracts (required security bits: {security_target})"
    );
}

/// Stage storage selected by `MXX_EXPONENT_LUT_NAND_STORAGE`: `device`,
/// `host` (the default), or `file:<directory>`.
fn nand_storage() -> api::StageStorage {
    match env::var("MXX_EXPONENT_LUT_NAND_STORAGE").as_deref() {
        Err(_) | Ok("host") => api::StageStorage::Host,
        Ok("device") => api::StageStorage::Device,
        Ok(value) => match value.strip_prefix("file:") {
            Some(directory) => api::StageStorage::File(directory.into()),
            None => panic!("unknown NAND stage storage {value}"),
        },
    }
}

#[test]
#[sequential(dcrt_runtime)]
fn nand_circuit_end_to_end() {
    // Stage plans report their measured W/C selection (the benchmark
    // estimate) through tracing; show it with `RUST_LOG=info`.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    // This is a GPU-only application fixture.  Returning here records that
    // the workflow is unverified when the required execution environment is
    // unavailable; it does not silently fall back to a CPU protocol.
    let devices = mxx_backends::poly::dcrt::gpu::detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping NAND round trip: no CUDA device detected");
        return;
    }
    let recorded = recorded_parameters();
    // Select a complete binary tree from the fixture environment and compile
    // it once. `NandCircuit::evaluate` below is the independent Boolean oracle;
    // it does not inspect encoded outputs or reuse the production lowering.
    let (circuit, max_lut_rounds, assignment) = nand_fixture(recorded.as_ref().map(|(_, row)| row));
    let layout = ExpLayout { stride_log2: 0, width_log2: 4 };
    let (poly_parameters, parameters, compiled, output_bound) =
        select_candidate(&circuit, layout, max_lut_rounds, recorded.as_ref().map(|(_, row)| row));
    // Every stage is planned on this GPU runtime before it executes; the
    // planner measures its W/C candidates on the real production graph and
    // logs the selected estimate. The secrets are sampled on it and stay
    // resident there for every stage that binds them. Each stage's graph is
    // recorded for the exported claim.
    let graphs = RefCell::new(BTreeMap::<&'static str, Graph>::new());
    let record = |stage: &'static str| {
        let graphs = &graphs;
        move |validated: &mxx_ir_core::ValidatedGraph| {
            graphs.borrow_mut().insert(stage, validated.source.clone());
            Ok(())
        }
    };
    let mut runtime = GpuRuntime::new(gpu_backend_on(gpu_parameters(&parameters), devices.clone()))
        .expect("GPU runtime");
    let dynamic = parameters.rhs_mode() == RhsMode::Dynamic;
    let sample_payload = |runtime: &mut GpuRuntime| -> RnsBggPayloadSecret {
        RnsEncryptBuild::new(&parameters, [0x4e; 32])
            .sample_payload_secret(runtime, &mut MemoryArtifactStore::default(), record("payload"))
            .expect("payload secret")
    };
    // A Static payload secret exists before preprocessing; the single
    // Dynamic secret is sampled only after it.
    let static_payload = (!dynamic).then(|| sample_payload(&mut runtime));
    let mut store = nand_storage().open().expect("stage storage");
    // Convert natural Boolean inputs into the declared LHS and RHS plaintext
    // maps.  Static RHS plaintexts are preprocessing inputs, Dynamic ones
    // encryption inputs; neither is supplied to online evaluation.
    let (lhs, rhs) = compiled
        .plaintext_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(&poly_parameters, &assignment)
        .expect("NAND plaintext bindings");
    let no_rhs = BTreeMap::new();
    // Static preprocessing receives the private RHS plaintexts and caller
    // payload; Dynamic preprocessing receives neither. It returns two typed
    // named-session artifacts consumed by later stages.
    let (preprocess, online_eval) = api::preprocess(
        compiled.program(),
        &ParamEnv::default(),
        if dynamic { &no_rhs } else { &rhs },
        static_payload.as_ref(),
        HASH_KEY,
        RnsPreprocessBuild::new(&parameters, [0x4e; 32]),
        &mut runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        record("preprocess"),
    )
    .expect("public preprocessing");
    let payload = static_payload.unwrap_or_else(|| sample_payload(&mut runtime));
    // Static encryption owns a fresh caller-supplied mask for each
    // invocation, retained locally for the final public verification only;
    // the Dynamic secret is also the mask.
    let mask = (!dynamic).then(|| {
        RnsEncryptBuild::new(&parameters, rand::random())
            .sample_mask_secret(&mut runtime, &mut store, record("mask"))
            .expect("fresh mask secret")
    });
    // Encrypt receives public LHS plaintexts, the secrets, the preprocess
    // artifact, and in Dynamic mode the RHS plaintexts of a fresh batch.  Its
    // estimate and execution share one graph construction path.
    let encrypted = api::encrypt(
        &compiled.program().encryption_inputs(),
        &ParamEnv::default(),
        &lhs,
        &preprocess,
        &payload,
        mask.as_ref(),
        HASH_KEY,
        RnsEncryptBuild::new(&parameters, [0x4e; 32]),
        &mut runtime,
        &mut store,
        |build| {
            if dynamic {
                build.bind_rhs_inputs(&ParamEnv::default(), &rhs, rand::random())?;
            }
            Ok(())
        },
        record("encrypt"),
    )
    .expect("public encryption");
    // Online evaluation receives only preparation artifacts and public
    // configuration; preprocessing RHS plaintexts and the mask stay out
    // of this stage.
    let result = api::online_eval(
        compiled.program(),
        &ParamEnv::default(),
        &online_eval,
        &encrypted,
        HASH_KEY,
        RnsOnlineEvalBuild::new(&parameters, [0x4e; 32]),
        &mut runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        record("online_eval"),
    )
    .expect("public online evaluation");

    // Compute the expected bit from the ordinary Boolean circuit, then
    // verify the returned encoding through the application-facing API
    // using the caller-retained payload and mask.
    let expected_bit = circuit.evaluate(&assignment).expect("NAND Boolean evaluation");
    assert!(!result.outputs.is_empty(), "online evaluation must produce output");
    let expected = DCRTPoly::const_rotate_poly(&poly_parameters, usize::from(expected_bit));
    verify_output(
        compiled.program(),
        &result,
        0,
        &expected,
        &parameters,
        &payload,
        mask.as_ref(),
        &mut runtime,
        &mut store,
    )
    .expect("NAND application verification");
    // The same output must not verify as the other bit.
    let wrong = DCRTPoly::const_rotate_poly(&poly_parameters, usize::from(!expected_bit));
    assert!(
        verify_output(
            compiled.program(),
            &result,
            0,
            &wrong,
            &parameters,
            &payload,
            mask.as_ref(),
            &mut runtime,
            &mut store
        )
        .is_err(),
        "NAND output must not verify as the wrong bit"
    );
    // The output plaintext is `X^bit`, so its decrypted coefficient one is
    // the bit.
    let decoded = decode_output(
        compiled.program(),
        &result,
        0,
        &parameters,
        &payload,
        mask.as_ref(),
        &[1],
        &mut runtime,
        &mut store,
        record("decode"),
    )
    .expect("NAND output decryption");
    assert_eq!(decoded, vec![BigInt::from(u8::from(expected_bit))], "decrypted NAND output");
    assert!(output_bound > 0u8.into());

    if let Some((id, row)) = recorded {
        let directory = common::generated_directory("nand_circuit", &id);
        let protocol = nand_protocol(
            &circuit,
            &compiled,
            graphs.into_inner(),
            &payload,
            mask.as_ref(),
            (preprocess.production().clone(), encrypted.production().clone()),
            &row,
        );
        mxx_ir_core::lean::protocol::export(&protocol, &directory).expect("NAND claim export");
        eprintln!("exported the NAND correctness claim to {}", directory.display());
    }
}

/// The executed NAND workflow as a closed protocol: payload and mask sampling, preprocessing,
/// encryption, online evaluation, and decryption of the output, compared with the Boolean
/// circuit. External inputs are the hash key, the circuit's input bits, the RHS exponents
/// `8 * bit` of the binary LUT contract (a requirement ties them to the bits), and the fixed
/// unit coefficients and initial packed state exponent `2`.
fn nand_protocol(
    circuit: &NandCircuit,
    compiled: &mxx_exponent_lut::CompiledNandCircuit,
    graphs: BTreeMap<&'static str, Graph>,
    payload: &RnsBggPayloadSecret,
    mask: Option<&mxx_exponent_lut::rns_bgg::RnsBggMaskSecret>,
    (preprocess, encrypt): (
        mxx_ir_core::artifact::ProductionId,
        mxx_ir_core::artifact::ProductionId,
    ),
    row: &ParameterRow,
) -> ProtocolDecl {
    let id = |name: &str| StageId(name.to_owned());
    let productions = BTreeMap::from([(preprocess, id("preprocess")), (encrypt, id("encrypt"))]);
    let secret = |stage: &str| (id(stage), "secret".to_owned());
    let mut links = BTreeMap::from([(
        payload.input_name().expect("sampled payload").to_owned(),
        secret("payload"),
    )]);
    // A Dynamic workflow samples no separate mask: its payload secret, sampled after
    // preprocessing, is also the mask, and encryption binds the RHS plaintexts. No stage the
    // decryption depends on reads a preprocessing artifact, so its preprocessing is not part of
    // the claim.
    if let Some(mask) = mask {
        links.insert(mask.input_name().expect("sampled mask").to_owned(), secret("mask"));
    }
    let (stage_names, rhs_stage): (&[&str], _) = match mask {
        Some(_) => {
            (&["payload", "mask", "preprocess", "encrypt", "online_eval", "decode"], "preprocess")
        }
        None => (&["payload", "encrypt", "online_eval", "decode"], "encrypt"),
    };
    for (input, port) in [("decode-key-01", 1), ("decode-key-11", 3), ("decode-row-1", 5)] {
        links.insert(input.to_owned(), (id("online_eval"), format!("output-0.{port}")));
    }
    let graph = |name: &str| &graphs[name];
    let stages = stage_names
        .iter()
        .map(|name| common::stage(name, graph(name), &productions, &links))
        .collect::<Vec<_>>();

    // The ideal NAND circuit and the requirement `exponent_i = 8 * bit_i`.
    let leaves = circuit.input_count();
    let ideal = DslContext::new("nand-ideal");
    let mut nodes = (0..leaves)
        .map(|leaf| ideal.input::<Int>(format!("bit_{leaf}"), IntType).expect("ideal bit"))
        .collect::<Vec<_>>();
    for [left, right] in circuit.gates() {
        nodes.push(Int::constant(1).sub(nodes[*left].clone().mul(nodes[*right].clone())));
    }
    let ideal = IdealSpec::new(
        ideal.output("bit", nodes[circuit.output()].clone()).unwrap().build().unwrap().graph,
    )
    .unwrap();
    let requirement = DslContext::new("nand-rhs-exponents");
    let valid = (0..leaves)
        .map(|leaf| {
            let bit = requirement.input::<Int>(format!("bit_{leaf}"), IntType).unwrap();
            let exponent = requirement.input::<Int>(format!("exponent_{leaf}"), IntType).unwrap();
            exponent.equal(bit.mul(8))
        })
        .reduce(|left, right| left & right)
        .expect("a NAND circuit has an input");
    let requirement =
        PurePredicateSpec::new(requirement.output("valid", valid).unwrap().build().unwrap().graph)
            .unwrap();

    // The plaintext inputs of the RHS stage (RHS `X^(8 bit)`: preprocessing when Static,
    // encryption when Dynamic) and of encryption (state `X^2`).
    let only = |stage: &str, suffix: String| -> String {
        let [name] = <[String; 1]>::try_from(common::inputs_ending_with(graph(stage), &suffix))
            .unwrap_or_else(|_| panic!("{stage} must declare one input ending with {suffix}"));
        name
    };
    let range = |lower: u8, upper: u8| InputValueContract::IntegerRange {
        lower: lower.into(),
        upper: upper.into(),
    };
    let mut unit = vec![common::to_stage(
        "encrypt",
        &only("encrypt", format!("/input/{:?}.coefficient", compiled.state_input())),
    )];
    let mut exponents = vec![Vec::new(); leaves];
    for (rhs, leaf) in compiled.rhs_ids().iter().zip(compiled.circuit_input_indices()) {
        unit.push(common::to_stage(
            rhs_stage,
            &only(rhs_stage, format!("/rhs/{rhs:?}.coefficient")),
        ));
        exponents[*leaf]
            .push(common::to_stage(rhs_stage, &only(rhs_stage, format!("/rhs/{rhs:?}.exponent"))));
    }
    let hash_key = stage_names
        .iter()
        .copied()
        .filter(|stage| !common::inputs_ending_with(graph(stage), "rns-bgg/hash-key").is_empty())
        .map(|stage| common::to_stage(stage, "rns-bgg/hash-key"))
        .collect();
    let mut externals = vec![
        common::external("hash_key", InputValueContract::Bytes { length: 32.into() }, hash_key),
        common::external("unit_coefficient", range(1, 1), unit),
        common::external(
            "state_exponent",
            range(2, 2),
            vec![common::to_stage(
                "encrypt",
                &only("encrypt", format!("/input/{:?}.exponent", compiled.state_input())),
            )],
        ),
    ];
    for (leaf, mut destinations) in exponents.into_iter().enumerate() {
        externals.push(common::external(
            &format!("bit_{leaf}"),
            range(0, 1),
            vec![
                ProtocolInputDestination::Ideal { input: format!("bit_{leaf}") },
                ProtocolInputDestination::Requirement {
                    requirement: 0,
                    input: format!("bit_{leaf}"),
                },
            ],
        ));
        destinations.push(ProtocolInputDestination::Requirement {
            requirement: 0,
            input: format!("exponent_{leaf}"),
        });
        externals.push(common::external(&format!("exponent_{leaf}"), range(0, 8), destinations));
    }
    let (contracts, bindings): (Vec<_>, Vec<_>) = externals.into_iter().unzip();

    // The decrypted coefficient one of the output, `decoded-1`, must equal the ideal bit.
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
                    actual_input: "decoded-1".to_owned(),
                    ideal_input: "bit".to_owned(),
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
                        output: "decoded-1".to_owned(),
                    },
                    ideal_output: "bit".to_owned(),
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
    .expect("valid NAND protocol")
}
