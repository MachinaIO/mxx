#![cfg(feature = "gpu")]

//! The application-facing RNS Sparse-LWR PRF lifecycle.

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime,
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
    noise_refresh::{
        pbc::{
            PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, clear_pbc_inner_product,
            generate_key_layout,
        },
        prf::{PrfLabel, SparseLwrPrfProfile, SparseLwrPrfProgram},
    },
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
use rayon::prelude::*;
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
    keys: Vec<PbcGeneratedKeyLayout>,
    a_exponents: Vec<u64>,
}

fn error_sigma() -> BigDecimal {
    let sigma = env::var("MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA")
        .unwrap_or_else(|_| "4".to_owned())
        .parse()
        .expect("MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA must be a decimal");
    assert!(sigma > BigDecimal::from(0u8), "MXX_EXPONENT_LUT_LWR_PRF_ERROR_SIGMA must be positive");
    sigma
}

fn profile_and_program(
    ring_dimension: u32,
) -> (SparseLwrPrfProfile, SparseLwrPrfProgram, Vec<PbcGeneratedKeyLayout>) {
    // The LWR input modulus Q fixes the exponent layout inside the ring:
    // X^(2n / Q * z) adds exponents modulo Q natively, so width_log2 =
    // log2(Q) - 1 and stride_log2 = log2(n) - width_log2. The output modulus
    // P = 2 * p_prime and the LWR dimension nu are independent of n.
    let input_modulus = env_usize("MXX_EXPONENT_LUT_LWR_PRF_INPUT_MODULUS", 32)
        .expect("LWR input modulus configuration");
    assert!(
        input_modulus >= 4 && input_modulus.is_power_of_two(),
        "MXX_EXPONENT_LUT_LWR_PRF_INPUT_MODULUS must be a power of two at least 4"
    );
    let output_modulus = env_usize("MXX_EXPONENT_LUT_LWR_PRF_OUTPUT_MODULUS", 4)
        .expect("LWR output modulus configuration");
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
    let lwr_dimension = env_usize("MXX_EXPONENT_LUT_LWR_PRF_LWR_DIMENSION", 16)
        .expect("LWR dimension configuration");
    let support_weight = env_usize("MXX_EXPONENT_LUT_LWR_PRF_SUPPORT_WEIGHT", 2)
        .expect("sparse key weight configuration");
    let output_count =
        env_usize("MXX_EXPONENT_LUT_LWR_PRF_OUTPUTS", 1).expect("PRF output count configuration");
    assert!(
        lwr_dimension > 0 && support_weight > 0 && support_weight <= lwr_dimension,
        "the sparse key weight must be positive and at most the LWR dimension"
    );
    assert!(output_count > 0, "MXX_EXPONENT_LUT_LWR_PRF_OUTPUTS must be positive");
    let profile = SparseLwrPrfProfile::new(
        ExpLayout { stride_log2, width_log2 },
        output_modulus / 2,
        lwr_dimension,
    )
    .expect("valid sparse-LWR profile");
    // Every output evaluates the same public vector `a` under its own sparse
    // key of weight h, placed by its own PBC layout with h' = h + 3 blocks.
    let parameters = PbcParameters::paper_evaluation(lwr_dimension, support_weight);
    // The keys are sampled from `MXX_EXPONENT_LUT_LWR_PRF_KEY_SEED`, or from a
    // fresh seed that is reported, so that any run can be repeated.
    let key_seed = match env::var("MXX_EXPONENT_LUT_LWR_PRF_KEY_SEED") {
        Ok(value) => value.parse::<u64>().expect("MXX_EXPONENT_LUT_LWR_PRF_KEY_SEED must be a u64"),
        Err(_) => rand::random(),
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
    let labels = (0..output_count as u64)
        .map(|index| {
            PrfLabel::from_components(&[b"lwr-rns-output".as_slice(), &index.to_le_bytes()])
                .expect("canonical label")
        })
        .collect();
    let program = SparseLwrPrfProgram::new(
        profile,
        keys.iter().map(|key| key.public_layout.clone()).collect(),
        labels,
    )
    .expect("sparse-LWR PRF program");
    eprintln!(
        "LWR PRF program: n={ring_dimension} Q={input_modulus} P={output_modulus} nu={lwr_dimension} h={support_weight} outputs={output_count} blocks={} block_width={}",
        parameters.bucket_count, keys[0].public_layout.bucket_width
    );
    (profile, program, keys)
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
            }),
        };
        input_bindings.insert(*wire, binding);
    }
    // Each bucket's one-hot selector family has exactly one nonzero member.
    let rhs_bindings = program
        .selection_families()
        .iter()
        .map(|family| {
            (
                family.selector,
                InputBinding::Family(RnsBggNoiseFamily {
                    element: RnsBggNoiseRhs {
                        payload_l1: BigUint::from(1u8),
                        payload_l2_squared: BigUint::from(1u8),
                    },
                    range: FamilyRange::full(IntExpr::constant(family.capacity))
                        .expect("valid selector family range"),
                    max_nonzero_payloads: 1u8.into(),
                }),
            )
        })
        .collect();
    RnsBggNoiseBindings::new(input_bindings, rhs_bindings)
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

type SearchedFixture = (
    RnsBggParameters,
    DCRTPolyParams,
    SparseLwrPrfProfile,
    SparseLwrPrfProgram,
    Vec<PbcGeneratedKeyLayout>,
    RnsBggNoiseTrace,
);

fn search_fixture(sigma: &BigDecimal) -> SearchedFixture {
    let required_security = env_usize("MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS", 1)
        .expect("security bits configuration");
    assert!(required_security > 0, "MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS must be positive");
    let required_security = u64::try_from(required_security)
        .expect("MXX_EXPONENT_LUT_LWR_PRF_SECURITY_BITS must fit in u64");
    let crt_bits =
        env_usize("MXX_EXPONENT_LUT_LWR_PRF_CRT_BITS", 40).expect("CRT bits configuration");
    assert!(crt_bits > 0, "MXX_EXPONENT_LUT_LWR_PRF_CRT_BITS must be positive");
    let expected_base_bits = env_usize("MXX_EXPONENT_LUT_LWR_PRF_BASE_BITS", crt_bits.div_ceil(2))
        .expect("decomposition base bits configuration");
    assert!(
        (1..=31).contains(&expected_base_bits),
        "MXX_EXPONENT_LUT_LWR_PRF_BASE_BITS must be in 1..=31"
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
    // `n,crt_depth,primes_per_drop` of a candidate already selected by a
    // full search. It replaces the search, and its lattice security estimate
    // is not repeated; the noise audit still builds its parameter chain.
    let selected = env::var("MXX_EXPONENT_LUT_LWR_PRF_SELECTED_PARAMETERS").ok().map(|value| {
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
        let (profile, program, keys) = profile_and_program(ring_dimension);
        let noise = noise_bindings(&program, sigma);
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
            let achieved = if selected.is_some() {
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
    program: SparseLwrPrfProgram,
    keys: Vec<PbcGeneratedKeyLayout>,
    trace: RnsBggNoiseTrace,
    runtime: &mut impl api::StageExecution<api::StageStore>,
    store: &mut api::StageStore,
) -> Fixture {
    // The secrets are sampled on the stage runtime and stay resident there.
    let payload = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_payload_secret(runtime, store)
        .expect("payload secret");
    let mask = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_mask_secret(runtime, store)
        .expect("mask secret");
    let zero = DCRTPoly::const_zero(&poly_parameters);
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
    // plaintext is copied into the online call.
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
        .selection_families()
        .iter()
        .map(|family| {
            let bits =
                program.selector_bits(family.output, &keys[family.output]).expect("key bits");
            let values = bits[family.bucket]
                .iter()
                .map(|bit| if *bit == 1 { one.clone() } else { zero.clone() })
                .collect();
            (family.selector, InputBinding::Family(values))
        })
        .collect::<BTreeMap<_, _>>();

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
/// from the key's PBC schedule, then the signed readout `(-1)^b * tau_r` for
/// `z = r + b * Q/2`. It never reads the production program graph.
fn lwr_oracle(
    profile: &SparseLwrPrfProfile,
    key: &PbcGeneratedKeyLayout,
    a: &[u64],
) -> (bool, usize) {
    let q = profile.q().expect("Q exponent") as u64;
    let z = clear_pbc_inner_product(&key.public_layout, key.private_schedule(), a, q)
        .expect("clear PBC inner product") as usize;
    let half = q as usize / 2;
    let readout = profile.canonical_readout().expect("canonical readout");
    if z < half { (false, readout[z]) } else { (true, readout[z - half]) }
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
    let sigma = error_sigma();
    if env::var("MXX_EXPONENT_LUT_LWR_PRF_SIMULATION_ONLY").is_ok_and(|value| value == "1") {
        // Parameter search, noise simulation, and the security gate only.
        search_fixture(&sigma);
        return;
    }
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        // This target is GPU-only; the sequential runtime lock is applied by
        // the test attribute when a device is available.
        eprintln!("skipping Sparse-LWR round trip: no CUDA device detected");
        return;
    }
    let (parameters, poly_parameters, profile, program, keys, trace) = search_fixture(&sigma);
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
        );
    } else {
        let mut runtime =
            GpuRuntime::new(gpu_backend(&parameters, devices.clone())).expect("GPU runtime");
        round_trip(parameters, poly_parameters, profile, program, keys, trace, &mut runtime, store);
    }
}

/// Preprocess, encrypt, and evaluate online on `runtime`, then verify every
/// output against the clear PRF.
#[allow(clippy::too_many_arguments)]
fn round_trip(
    parameters: RnsBggParameters,
    poly_parameters: DCRTPolyParams,
    profile: SparseLwrPrfProfile,
    program: SparseLwrPrfProgram,
    keys: Vec<PbcGeneratedKeyLayout>,
    trace: RnsBggNoiseTrace,
    runtime: &mut impl api::StageExecution<api::StageStore>,
    mut store: api::StageStore,
) {
    let Fixture { parameters, profile, program, payload, mask, rhs, lhs, trace, keys, a_exponents } =
        fixture(parameters, poly_parameters, profile, program, keys, trace, runtime, &mut store);
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
        runtime,
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
        runtime,
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
        runtime,
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
    // In estimate mode (`MXX_EXPONENT_LUT_ESTIMATE_ONLY=1`) the GPU stages
    // were only planned, so there are no outputs to verify.
    if env::var("MXX_EXPONENT_LUT_ESTIMATE_ONLY").is_ok_and(|value| value == "1") {
        eprintln!("estimate mode: every stage was planned; outputs are not verified");
        return;
    }
    // Compute every output's expected readout independently from `a` and
    // its key. The coefficient LUT returns the signed readout as the
    // constant polynomial, so the expected plaintext is built in the source
    // ring with the readout in coefficient 0 only; it is not decoded from the
    // returned carrier. The terminal scale is floor(q / 2), so verification
    // decides the readout modulo 2: only a key whose readout is odd tells a
    // correct output from a zero one.
    let source_parameters = parameters.level(0).expect("LWR source parameters").clone();
    for (output, key) in keys.iter().enumerate() {
        let (negative, readout) = lwr_oracle(&profile, key, &a_exponents);
        let readout = u32::try_from(readout).expect("canonical readout fits DCRT coefficient");
        let mut coefficients = vec![0; source_parameters.ring_dimension() as usize];
        coefficients[0] = readout;
        let expected = DCRTPoly::from_u32s(&source_parameters, &coefficients);
        let expected = if negative { -expected } else { expected };
        verify_output(
            program.program(),
            &result,
            output,
            &expected,
            &parameters,
            &payload,
            &mask,
            runtime,
            &mut store,
        )
        .expect("LWR application verification");
    }
    assert!(!trace.outputs.is_empty(), "noise trace must finalize declared output");
    assert!(!result.outputs.is_empty(), "online evaluation must produce output");
}
