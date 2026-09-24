#![cfg(feature = "gpu")]

//! The application-facing RNS Sparse-LWR PRF lifecycle.

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime,
    artifact::FileArtifactStore,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend_on},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_exponent_lut::{
    ExpLayout, api,
    noise_refresh::prf::{PrfLabel, SparseLwrPrfProfile, SparseLwrPrfProgram},
    program::{AttributeKnowledge, FamilyRange, InputBinding, WireId},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseRhs,
        RnsBggNoiseTrace, RnsBggNoiseValue, RnsBggParameters, RnsBggPlanRequest, RnsEncryptBuild,
        RnsNoiseBuild, RnsOnlineBuild, RnsPreprocessBuild, verify_output,
    },
    utils::{
        audit_program_at_crt_depth, env_usize, lattice_security_bits, noise_model,
        ring_dimension_candidates,
    },
};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, env};

const HASH_KEY: &[u8; 32] = b"lwr-prf-rns-public-api-hash-key1";

struct Fixture {
    parameters: RnsBggParameters,
    profile: SparseLwrPrfProfile,
    program: SparseLwrPrfProgram,
    payload: mxx_exponent_lut::rns_bgg::RnsBggPayloadSecret,
    mask: mxx_exponent_lut::rns_bgg::RnsBggMaskSecret,
    rhs: BTreeMap<mxx_exponent_lut::program::RhsId, InputBinding<DCRTPoly, Vec<DCRTPoly>>>,
    lhs: BTreeMap<WireId, InputBinding<DCRTPoly, Vec<DCRTPoly>>>,
    trace: RnsBggNoiseTrace,
    d_exponents: Vec<usize>,
}

fn error_sigma() -> BigDecimal {
    let sigma = env::var("MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA")
        .unwrap_or_else(|_| "4".to_owned())
        .parse()
        .expect("MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA must be a decimal");
    assert!(sigma > BigDecimal::from(0u8), "MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA must be positive");
    sigma
}

fn profile_and_program(ring_dimension: u32) -> (SparseLwrPrfProfile, SparseLwrPrfProgram) {
    let log_ring_dimension = ring_dimension.ilog2();
    let stride_log2 =
        log_ring_dimension.checked_sub(4).expect("LWR ring dimension must be at least 2^4");
    // Rebuild the profile for every candidate dimension.  The exponent
    // layout keeps width four while stride expands with the ring, so D's
    // monomials remain valid in every searched negacyclic ring.
    let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2, width_log2: 4 }, 2)
        .expect("valid sparse-LWR profile");
    let labels = [PrfLabel::from_components(&[b"lwr-rns-label"]).expect("canonical label")];
    let program = SparseLwrPrfProgram::new_structural(profile, 2, 2, HASH_KEY, &labels)
        .expect("structural sparse-LWR program");
    (profile, program)
}

fn noise_bindings(program: &SparseLwrPrfProgram, sigma: &BigDecimal) -> RnsBggNoiseBindings {
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
    let noise_rhs = program
        .selection_families()
        .first()
        .is_some()
        .then_some(RnsBggNoiseRhs {
            payload_l1: BigUint::from(1u8),
            payload_l2_squared: BigUint::from(1u8),
        })
        .into_iter()
        .collect::<Vec<_>>();
    let mut input_bindings = BTreeMap::new();
    for ((wire, declaration), value) in
        program.program().encryption_inputs().inputs().iter().zip(noise_inputs)
    {
        let binding = match declaration.shape() {
            InputBinding::Scalar(()) => InputBinding::Scalar(value),
            InputBinding::Family(count) => InputBinding::Family(RnsBggNoiseFamily {
                element: value,
                range: FamilyRange::full(count.clone()).expect("valid input family range"),
                max_nonzero_payloads: 1u8.into(),
            }),
        };
        input_bindings.insert(*wire, binding);
    }
    let rhs_binding = program.selection_families().first().map(|family| {
        (
            family.selector,
            InputBinding::Family(RnsBggNoiseFamily {
                element: noise_rhs[0].clone(),
                range: FamilyRange::full(IntExpr::constant(family.capacity))
                    .expect("valid selector family range"),
                max_nonzero_payloads: 1u8.into(),
            }),
        )
    });
    RnsBggNoiseBindings::new(input_bindings, rhs_binding.into_iter().collect())
        .expect("declaration-order noise bindings")
}

fn terminal_value_bounds(
    profile: &SparseLwrPrfProfile,
    program: &SparseLwrPrfProgram,
) -> Vec<BigUint> {
    let bound = profile
        .canonical_readout()
        .expect("canonical readout")
        .into_iter()
        .max()
        .map(BigUint::from)
        .expect("nonempty canonical readout");
    vec![bound; program.program().outputs().len()]
}

fn search_fixture(
    sigma: &BigDecimal,
) -> (RnsBggParameters, DCRTPolyParams, SparseLwrPrfProfile, SparseLwrPrfProgram, RnsBggNoiseTrace)
{
    let required_security = env_usize("MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS", 1)
        .expect("security bits configuration");
    assert!(required_security > 0, "MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS must be positive");
    let required_security = u64::try_from(required_security)
        .expect("MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS must fit in u64");
    let crt_bits =
        env_usize("MXX_EXPONENT_LUT_LWR_PRF_CRT_BITS", 40).expect("CRT bits configuration");
    assert!(crt_bits > 0, "MXX_EXPONENT_LUT_LWR_PRF_CRT_BITS must be positive");
    let expected_base_bits = crt_bits.div_ceil(2);
    assert!(
        expected_base_bits <= 31,
        "MXX_EXPONENT_LUT_LWR_PRF_CRT_BITS is unsupported above 62 bits"
    );
    let max_crt_depth = env_usize("MXX_EXPONENT_LUT_LWR_PRF_MAX_CRT_DEPTH", 32)
        .expect("maximum CRT depth configuration");
    assert!(max_crt_depth > 0, "MXX_EXPONENT_LUT_LWR_PRF_MAX_CRT_DEPTH must be positive");
    let operations_per_drop = env_usize("MXX_EXPONENT_LUT_LWR_PRF_OPERATIONS_PER_DROP", 1)
        .expect("operations-per-drop configuration");
    // Each drop removes up to this many CRT primes; the drop quotient is the
    // message scale that the level noise must stay below half of.
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
    let average = noise_model("MXX_EXPONENT_LUT_LWR_PRF_").expect("noise-model configuration");
    let candidates = ring_dimension_candidates(
        "MXX_EXPONENT_LUT_LWR_PRF_MIN_LOG_RING_DIMENSION",
        "MXX_EXPONENT_LUT_LWR_PRF_MAX_LOG_RING_DIMENSION",
        5,
        16,
    )
    .expect("ring-dimension configuration");
    let mut failures = Vec::new();
    for ring_dimension in candidates {
        let (profile, program) = profile_and_program(ring_dimension);
        let noise = noise_bindings(&program, sigma);
        let terminal_bounds = terminal_value_bounds(&profile, &program);
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
                terminal_value_bound: terminal_bounds
                    .iter()
                    .max()
                    .cloned()
                    .expect("terminal bounds are nonempty"),
                operations_per_drop,
                primes_per_drop,
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
                Err(error) => {
                    failures.push(format!(
                        "ring dimension {ring_dimension}, CRT depth {crt_depth}: {error}"
                    ));
                    continue;
                }
            };
            for (index, level) in parameters.levels().iter().enumerate() {
                assert_eq!(
                    level.base_bits(),
                    expected_base_bits as u32,
                    "selected LWR level {index} must use ceil(CRT_BITS/2) base bits"
                );
            }
            let workload = match RnsBggDerivedWorkload::from_program_with_parameters(
                &parameters,
                program.program(),
            ) {
                Ok(workload) => workload,
                Err(error) => {
                    failures.push(format!(
                        "ring dimension {ring_dimension}, CRT depth {crt_depth}: workload validation failed: {error}"
                    ));
                    continue;
                }
            };
            let trace = match api::noise_simulate(
                program.program(),
                RnsNoiseBuild::new(&parameters, program.program(), &workload, &noise)
                    .expect("validated candidate noise lowering state"),
                |_, outputs| Ok(outputs),
            ) {
                Ok(trace) => trace,
                Err(error) => {
                    failures.push(format!(
                        "ring dimension {ring_dimension}, CRT depth {crt_depth}: noise simulation failed: {error}"
                    ));
                    continue;
                }
            };
            if trace.outputs.len() != program.program().outputs().len() {
                failures.push(format!(
                    "ring dimension {ring_dimension}, CRT depth {crt_depth}: noise simulation finalized {} outputs, expected {}",
                    trace.outputs.len(),
                    program.program().outputs().len()
                ));
                continue;
            }
            let source = parameters.level(0).expect("source level").clone();
            let achieved = lattice_security_bits(&source, sigma)
                .expect("lattice security estimator must be available");
            if achieved >= required_security {
                eprintln!(
                    "LWR parameters accepted: ring_dimension={ring_dimension} crt_depth={crt_depth} security_bits={achieved} required_security_bits={required_security}"
                );
                return (parameters, source, profile, program, audit_trace);
            }
            failures.push(format!(
                "ring dimension {ring_dimension}, CRT depth {crt_depth}: security {achieved} < {required_security}"
            ));
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
    program: SparseLwrPrfProgram,
    trace: RnsBggNoiseTrace,
    runtime: &mut GpuRuntime,
    store: &mut FileArtifactStore,
) -> Fixture {
    // The secrets are sampled on the stage runtime and stay resident there.
    let payload = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_payload_secret(runtime, store)
        .expect("payload secret");
    let mask = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_mask_secret(runtime, store)
        .expect("mask secret");
    let zero = DCRTPoly::const_zero(&poly_parameters);
    let d_exponents = (0..profile.transform_column_count().expect("D width"))
        .map(|index| index % profile.q().expect("Q exponent"))
        .collect::<Vec<_>>();
    // D is a public LHS family supplied at encrypt time.  Convert each
    // exponent to its stride-aware signed monomial, including the
    // X^n = -1 negacyclic sign when the rotation crosses the boundary.
    let d_values = d_exponents
        .iter()
        .map(|exponent| {
            let dimension = poly_parameters.ring_dimension() as usize;
            let rotation = (exponent * profile.layout.stride()) % (2 * dimension);
            let monomial = DCRTPoly::const_rotate_poly(&poly_parameters, rotation % dimension);
            if rotation >= dimension { -monomial } else { monomial }
        })
        .collect::<Vec<_>>();

    let mut lhs = BTreeMap::new();
    let input_projection = program.program().encryption_inputs();
    for (wire, declaration) in input_projection.inputs() {
        // The initial state is private; public scalar/family inputs are
        // supplied as encrypt-time attributes.  No private plaintext is
        // copied into the online call.
        let value = match (declaration.knowledge(), declaration.shape()) {
            (AttributeKnowledge::PrivatePayload, InputBinding::Scalar(())) => {
                InputBinding::Scalar(zero.clone())
            }
            (AttributeKnowledge::PublicPayload, InputBinding::Scalar(())) => {
                InputBinding::Scalar(zero.clone())
            }
            (AttributeKnowledge::PublicPayload, InputBinding::Family(_)) => {
                InputBinding::Family(d_values.clone())
            }
            (knowledge, shape) => panic!("unexpected input declaration: {knowledge:?} {shape:?}"),
        };
        lhs.insert(*wire, value);
    }

    // Selectors are fixed private RHS plaintexts.  They are prepared during
    // preprocessing and later consumed only through compact RHS artifacts.
    let selector = program.selection_families()[0].selector;
    let mut selector_values = vec![zero; program.block_count() * program.bucket_width()];
    for bucket in 0..program.block_count() {
        selector_values[bucket * program.bucket_width()] =
            DCRTPoly::const_rotate_poly(&poly_parameters, 0);
    }
    let rhs = BTreeMap::from([(selector, InputBinding::Family(selector_values))]);

    Fixture { parameters, profile, program, payload, mask, rhs, lhs, trace, d_exponents }
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

fn sequential_dt_oracle(program: &SparseLwrPrfProgram, d_exponents: &[usize]) -> usize {
    // This is an independent cleartext oracle for the sequential bucket
    // recurrence.  It reads only the immutable transform bits and fixture D
    // exponents, never the production key, row, artifact, or online result.
    let q = program.profile().q().expect("Q exponent");
    (0..program.block_count())
        .map(|bucket| {
            let transform = &program.label_transforms()[bucket * program.bucket_width()];
            transform
                .bits()
                .iter()
                .enumerate()
                .filter_map(|(column, bit)| (*bit == 1).then_some(d_exponents[column]))
                .sum::<usize>()
        })
        .sum::<usize>() %
        q
}

#[test]
#[sequential(dcrt_runtime)]
fn lwr_prf_end_to_end() {
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        // This target is GPU-only; the sequential runtime lock is applied by
        // the test attribute when a device is available.
        eprintln!("skipping Sparse-LWR round trip: no CUDA device detected");
        return;
    }
    let sigma = error_sigma();
    let (parameters, poly_parameters, profile, program, trace) = search_fixture(&sigma);
    // Stage plans report their measured W/C selection (the benchmark
    // estimate) through tracing; show it with `RUST_LOG=info`.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let mut runtime =
        GpuRuntime::new(gpu_backend(&parameters, devices.clone())).expect("GPU runtime");
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = FileArtifactStore::new(directory.path()).expect("artifact store");
    let Fixture { parameters, profile, program, payload, mask, rhs, lhs, trace, d_exponents } =
        fixture(parameters, poly_parameters, profile, program, trace, &mut runtime, &mut store);
    assert!(!trace.wires.is_empty(), "noise simulation must visit the PRF gates");
    let observed = RefCell::new(Vec::<&'static str>::new());
    let observe = |name: &'static str| observed.borrow_mut().push(name);
    // Preprocess receives the private payload secret and fixed RHS selectors,
    // while encrypt receives public D attributes, a fresh mask, and only the
    // preparation artifacts.  Online receives artifacts/configuration only;
    // RHS plaintexts never cross this boundary.
    let (preprocess, online) = api::preprocess(
        program.program(),
        &ParamEnv::default(),
        &rhs,
        &payload,
        HASH_KEY,
        RnsPreprocessBuild::new(&parameters, [7; 32]),
        &mut runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |_| {
            observe("preprocess");
            Ok(())
        },
    )
    .expect("RNS preprocessing");
    let encrypted = api::encrypt(
        &program.program().encryption_inputs(),
        &ParamEnv::default(),
        &lhs,
        &preprocess,
        &payload,
        &mask,
        HASH_KEY,
        RnsEncryptBuild::new(&parameters, [7; 32]),
        &mut runtime,
        &mut store,
        |_| Ok(()),
        |_| {
            observe("encrypt");
            Ok(())
        },
    )
    .expect("RNS encryption");
    let result = api::online_eval(
        program.program(),
        &ParamEnv::default(),
        &online,
        &encrypted,
        HASH_KEY,
        RnsOnlineBuild::new(&parameters, [7; 32]),
        &mut runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |_| {
            observe("online_eval");
            Ok(())
        },
    )
    .expect("RNS online evaluation");
    // Each of the three actual production graphs was planned (and its
    // candidates measured) by the GPU runtime immediately before execution,
    // so the workflow remains simulation -> preprocess -> encrypt -> online
    // evaluation rather than a test-only reconstruction.
    assert_eq!(
        observed.borrow().as_slice(),
        ["preprocess", "encrypt", "online_eval"],
        "observers must see the exact three production graphs"
    );
    // Compute the expected exponent/readout independently from D and the
    // structural transform.  The coefficient polynomial is then built in
    // the source ring with the exact canonical readout in every coefficient;
    // it is not decoded from the returned carrier.
    let expected_exponent = sequential_dt_oracle(&program, &d_exponents);
    let expected_readout =
        profile.canonical_readout().expect("canonical readout")[expected_exponent];
    let source_parameters = parameters.level(0).expect("LWR source parameters").clone();
    let expected_readout =
        u32::try_from(expected_readout).expect("canonical readout fits DCRT coefficient");
    let expected = DCRTPoly::from_u32s(
        &source_parameters,
        &vec![expected_readout; source_parameters.ring_dimension() as usize],
    );
    verify_output(
        program.program(),
        &result,
        0,
        &expected,
        &parameters,
        &payload,
        &mask,
        &mut runtime,
        &mut store,
    )
    .expect("LWR application verification");
    assert!(!trace.outputs.is_empty(), "noise trace must finalize declared output");
    assert!(!result.outputs.is_empty(), "online evaluation must produce output");
}
