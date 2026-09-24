#![cfg(feature = "gpu")]

//! Public RNS-BGG NAND application workflow.

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
use mxx_exponent_lut::{
    ExpLayout, NandCircuit, api,
    noise_refresh::simulation::NoiseModelKind,
    program::Operand,
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseRhs, RnsBggNoiseValue,
        RnsBggParameters, RnsBggPayloadSecret, RnsBggPlanRequest, RnsEncryptBuild, RnsNoiseBuild,
        RnsOnlineBuild, RnsPreprocessBuild, verify_output,
    },
    utils::{
        audit_program_at_crt_depth, env_usize, lattice_security_bits, noise_model,
        ring_dimension_candidates,
    },
};
use mxx_ir_core::ParamEnv;
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::env;

const HASH_KEY: &[u8; 32] = b"nand-rns-public-api-hash-key-v01";
const NAND_DEPTH_ENV: &str = "MXX_EXPONENT_LUT_NAND_DEPTH";

fn nand_fixture() -> (NandCircuit, usize, Vec<bool>) {
    let depth = env::var(NAND_DEPTH_ENV)
        .unwrap_or_else(|_| "1".to_owned())
        .parse::<u32>()
        .unwrap_or_else(|error| panic!("{NAND_DEPTH_ENV} must be a non-negative integer: {error}"));
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

fn select_candidate(
    circuit: &NandCircuit,
    layout: ExpLayout,
    max_lut_rounds: usize,
) -> (DCRTPolyParams, RnsBggParameters, mxx_exponent_lut::CompiledNandCircuit, BigUint) {
    let sigma = env::var("MXX_EXPONENT_LUT_NAND_ERROR_SIGMA")
        .unwrap_or_else(|_| "4".to_owned())
        .parse::<BigDecimal>()
        .unwrap_or_else(|error| {
            panic!("MXX_EXPONENT_LUT_NAND_ERROR_SIGMA must be a nonnegative decimal: {error}")
        });
    assert!(
        sigma >= BigDecimal::from(0u8),
        "MXX_EXPONENT_LUT_NAND_ERROR_SIGMA must be nonnegative"
    );
    let crt_bits = env_usize("MXX_EXPONENT_LUT_NAND_CRT_BITS", 40)
        .unwrap_or_else(|error| panic!("invalid NAND CRT bits: {error}"));
    let max_crt_depth = env_usize("MXX_EXPONENT_LUT_NAND_MAX_CRT_DEPTH", 32)
        .unwrap_or_else(|error| panic!("invalid NAND maximum CRT depth: {error}"));
    let operations_per_drop = env_usize("MXX_EXPONENT_LUT_NAND_OPERATIONS_PER_DROP", 2)
        .unwrap_or_else(|error| panic!("invalid NAND operations-per-drop: {error}"));
    // Each drop removes up to this many CRT primes; the drop quotient is the
    // message scale that the level noise must stay below half of.
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
    let average = noise_model("MXX_EXPONENT_LUT_NAND_")
        .unwrap_or_else(|error| panic!("invalid NAND noise model: {error}"));
    let expected_acceptance = match average.as_ref() {
        Some(_) => NoiseModelKind::AverageCase,
        None => NoiseModelKind::WorstCase,
    };

    for ring_dimension in ring_dimension_candidates(
        "MXX_EXPONENT_LUT_NAND_MIN_LOG_RING_DIMENSION",
        "MXX_EXPONENT_LUT_NAND_MAX_LOG_RING_DIMENSION",
        5,
        16,
    )
    .unwrap_or_else(|error| panic!("invalid NAND ring-dimension range: {error}"))
    {
        let compiled =
            circuit.compile(layout, ring_dimension as usize, max_lut_rounds).unwrap_or_else(
                |error| panic!("NAND compilation failed for n={ring_dimension}: {error}"),
            );
        let bindings = noise_bindings(&compiled, &sigma);
        let terminal_bounds = vec![BigUint::from(1u8); compiled.program().outputs().len()];
        let depth_candidates = (1..=max_crt_depth).flat_map(|depth| {
            (1..=max_primes_per_drop).map(move |primes_per_drop| (depth, primes_per_drop))
        });
        for (crt_depth, primes_per_drop) in depth_candidates {
            let request = RnsBggPlanRequest {
                ring_dimension,
                encoding_error_sigma: sigma.clone(),
                crt_bits,
                terminal_depth: 1,
                max_crt_depth: crt_depth,
                terminal_value_bound: BigUint::from(1u8),
                operations_per_drop,
                primes_per_drop,
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
            } else {
                lattice_security_bits(source, &sigma).unwrap_or_else(|error| {
                    panic!("NAND security estimation failed for n={ring_dimension}, CRT depth {crt_depth}: {error}")
                })
            };
            if achieved < security_target {
                continue;
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
    // Select a complete binary tree from the fixture environment and compile
    // it once. `NandCircuit::evaluate` below is the independent Boolean oracle;
    // it does not inspect encoded outputs or reuse the production lowering.
    let (circuit, max_lut_rounds, assignment) = nand_fixture();
    let layout = ExpLayout { stride_log2: 0, width_log2: 4 };
    let (poly_parameters, parameters, compiled, output_bound) =
        select_candidate(&circuit, layout, max_lut_rounds);
    // Every stage is planned on this GPU runtime before it executes; the
    // planner measures its W/C candidates on the real production graph and
    // logs the selected estimate. The secrets are sampled on it and stay
    // resident there for every stage that binds them.
    let mut runtime = GpuRuntime::new(gpu_backend_on(gpu_parameters(&parameters), devices.clone()))
        .expect("GPU runtime");
    let payload: RnsBggPayloadSecret = RnsEncryptBuild::new(&parameters, [0x4e; 32])
        .sample_payload_secret(&mut runtime, &mut MemoryArtifactStore::default())
        .expect("payload secret");
    let mut store = MemoryArtifactStore::default();
    // Convert natural Boolean inputs into the declared LHS and fixed-RHS
    // plaintext maps.  The RHS plaintexts remain preprocessing inputs and
    // are never supplied to online evaluation.
    let (lhs, rhs) = compiled
        .plaintext_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(&poly_parameters, &assignment)
        .expect("NAND plaintext bindings");
    // Encryption owns a fresh caller-supplied mask for each invocation;
    // the mask is retained locally for the final public verification only.
    let mask = RnsEncryptBuild::new(&parameters, rand::random())
        .sample_mask_secret(&mut runtime, &mut store)
        .expect("fresh mask secret");
    // Preprocess receives the private RHS plaintexts and caller payload,
    // then returns two typed named-session artifacts consumed by later
    // stages.
    let (preprocess, online) = api::preprocess(
        compiled.program(),
        &ParamEnv::default(),
        &rhs,
        &payload,
        HASH_KEY,
        RnsPreprocessBuild::new(&parameters, [0x4e; 32]),
        &mut runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |_| Ok(()),
    )
    .expect("public preprocessing");
    // Encrypt receives public LHS plaintexts, the fresh private mask, and
    // the preprocess artifact.  Its estimate and execution share one
    // graph construction path.
    let encrypted = api::encrypt(
        &compiled.program().encryption_inputs(),
        &ParamEnv::default(),
        &lhs,
        &preprocess,
        &payload,
        &mask,
        HASH_KEY,
        RnsEncryptBuild::new(&parameters, [0x4e; 32]),
        &mut runtime,
        &mut store,
        |_| Ok(()),
        |_| Ok(()),
    )
    .expect("public encryption");
    // Online evaluation receives only preparation artifacts and public
    // configuration; preprocessing RHS plaintexts and the mask stay out
    // of this stage.
    let result = api::online_eval(
        compiled.program(),
        &ParamEnv::default(),
        &online,
        &encrypted,
        HASH_KEY,
        RnsOnlineBuild::new(&parameters, [0x4e; 32]),
        &mut runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |_| Ok(()),
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
        &mask,
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
            &mask,
            &mut runtime,
            &mut store,
        )
        .is_err(),
        "NAND output must not verify as the wrong bit"
    );
    assert!(output_bound > 0u8.into());
}
