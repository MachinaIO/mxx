#![cfg(feature = "gpu")]

//! One actual normalized LWR PRF output, without any refresh label expansion.
//! Select BGG or BFV with MXX_EXPONENT_LUT_TEST_SCHEME; CPU simulation precedes GPU execution.
fn use_bfv() -> bool {
    match std::env::var("MXX_EXPONENT_LUT_TEST_SCHEME").as_deref().unwrap_or("bgg") {
        "bgg" => false,
        "bfv" => true,
        other => panic!("MXX_EXPONENT_LUT_TEST_SCHEME must be bgg or bfv, got {other}"),
    }
}
use mxx_bgg::BggSamplerLayout;

use mxx_exponent_lut::{
    bgg::{BggRefreshParameters, RefreshCompiler},
    noise_refresh::{
        pbc::{PbcActiveCellIndex, PbcParameters, PbcRootSeed, generate_key_layout},
        prf::{SparseLwrPrfProfile, SparseLwrPrfProgram},
    },
};

use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};

use num_bigint::{BigInt, BigUint};

use num_traits::{One, Zero};
use rand::{Rng, seq::index::sample};

#[derive(Clone, Copy, Debug)]
struct PrfTestParameters {
    log_n: usize,
    depth: usize,
    crt_bits: usize,
    base_bits: u32,
    universe: usize,
    weight: usize,
    q_native: usize,
    p: usize,
    security: usize,
    encoding_sigma: f64,
    decoder_sigma: f64,
}

impl PrfTestParameters {
    fn from_env() -> Self {
        let env = |suffix: &str, default: usize| {
            std::env::var(format!("MXX_EXPONENT_LUT_REFRESH_{suffix}"))
                .map(|v| v.parse::<usize>().expect("integer parameter"))
                .unwrap_or(default)
        };
        let log_n = env("MIN_LOG_RING_DIMENSION", 5);
        let depth = env("MIN_CRT_DEPTH", 7);
        let crt_bits = env("CRT_BITS", 50);
        let base_bits = env("BASE_BITS", 25) as u32;
        let universe = env("SPARSE_LWR_UNIVERSE", 64);
        let weight = env("SPARSE_LWR_WEIGHT", 1);
        let q_native = env("SPARSE_LWR_MODULUS", 8);
        let p = env("SPARSE_LWR_OUTPUT_MODULUS", 4);
        let security = env("SECURITY_BITS", 1);
        // This integration checks the noise model and arithmetic at the supplied
        // parameters. Security-profile validation belongs to parameter search.
        Self {
            log_n,
            depth,
            crt_bits,
            base_bits,
            universe,
            weight,
            q_native,
            p,
            security,
            encoding_sigma: 4.0,
            decoder_sigma: 4.578,
        }
    }

    fn build(&self) -> (DCRTPolyParams, BggRefreshParameters, SparseLwrPrfProfile) {
        let Self { log_n, depth, crt_bits, base_bits, q_native, p, security, .. } = *self;
        let params = DCRTPolyParams::new(1 << log_n, depth, crt_bits, base_bits, None, None);
        let n = params.ring_dimension() as usize;
        let modulus = BigInt::from(params.modulus().as_ref().clone());
        let width_log2 = q_native.trailing_zeros() as usize - 1;
        assert!(q_native.is_power_of_two() && p >= 4 && p.is_power_of_two());
        let profile = SparseLwrPrfProfile::new(
            mxx_exponent_lut::ExpLayout {
                stride_log2: (log_n - width_log2) as u32,
                width_log2: width_log2 as u32,
            },
            p / 2,
        )
        .unwrap();
        let layout = BggSamplerLayout {
            modulus: modulus.clone().into(),
            ring_dimension: n.into(),
            secret_dimension: 2,
            digit_count: params.modulus_digits(),
            gadget_base: (BigInt::one() << base_bits).into(),
        };
        let (moduli, _, _) = params.to_crt();
        let setup = BggRefreshParameters::new(
            [0x52; 32],
            p,
            2,
            n,
            1,
            1,
            security,
            profile.layout.width(),
            layout.clone(),
            RefreshCompiler {
                full_modulus: modulus.clone().into(),
                crt_plaintext_moduli: moduli.iter().map(|v| BigInt::from(*v).into()).collect(),
                reconstruction_coefficients: params
                    .reconst_coeffs()
                    .into_iter()
                    .map(|v| BigInt::from(v).into())
                    .collect(),
            },
            mxx_ir_core::RealExpr::from_f64_exact(self.decoder_sigma).unwrap(),
            mxx_ir_core::RealExpr::from_f64_exact(self.encoding_sigma).unwrap(),
            "one-output-prf",
        );
        (params, setup, profile)
    }
}

struct PrfTestPlan {
    model: mxx_exponent_lut::noise_refresh::simulation::NoiseModelKind,
    root_seed: [u8; 32],
    support: Vec<usize>,
    layout_id: [u8; 32],
    active_count: usize,
    coefficient_events: usize,
    failure_exponent: usize,
    bound: BigUint,
    worst_bound: BigUint,
}

fn simulate_prf(parameters: PrfTestParameters) -> PrfTestPlan {
    let (params, setup, profile) = parameters.build();
    let n = params.ring_dimension() as usize;
    let layout = &setup.layout;
    let security = parameters.security;
    let mut rng = rand::rng();
    let root_seed = rng.random();
    let support = sample(&mut rng, parameters.universe, parameters.weight).into_vec();
    let generated = generate_key_layout(
        &PbcParameters::paper_evaluation(parameters.universe, parameters.weight),
        PbcRootSeed(root_seed),
        &support,
    )
    .unwrap();
    let public_layout = &generated.public_layout;
    let active = PbcActiveCellIndex::build(public_layout).unwrap();
    let program = SparseLwrPrfProgram::new(
        profile,
        public_layout.bucket_width,
        public_layout.parameters.bucket_count,
    )
    .unwrap();
    let snapshot = BggRefreshParameters::build_noise_snapshot_from_public_layout(
        &setup.clone(),
        &program,
        public_layout,
        &active,
        None,
    )
    .unwrap();
    let noise = snapshot.evaluate_pre_refresh().unwrap();
    let worst_bound = noise.readout_bound;
    // One-output adapter of simulation.rs AverageCaseConfig::z_squared and
    // average_favg. V uses doubled coordinates, hence the exact division by 4.
    // No refresh mask/fresh-label topology is fabricated for this PRF test.
    let coefficient_events = n * layout.public_key_columns();
    let policy = mxx_exponent_lut::noise_refresh::simulation::AverageCaseConfig {
        failure_exponent: security as u32,
        allow_average_acceptance: true,
        ..Default::default()
    };
    let z = policy.z_squared_for_events(&BigUint::from(coefficient_events)).unwrap();
    let moments = mxx_exponent_lut::noise_utils::NoiseMoments {
        mean_numerator: 0.into(),
        mean_denominator: 1u8.into(),
        mean_radius: mxx_exponent_lut::noise_utils::NonnegativeRational::zero(),
        variance: mxx_exponent_lut::noise_utils::NonnegativeRational::new(
            noise.readout_variance.numerator,
            noise.readout_variance.denominator * 4u8,
        )
        .unwrap(),
    };
    let bound = moments.tail_bound(&z);
    let model = mxx_exponent_lut::noise_refresh::simulation::NoiseModelKind::from_env().unwrap();
    let bound = if model == mxx_exponent_lut::noise_refresh::simulation::NoiseModelKind::WorstCase {
        worst_bound.clone()
    } else {
        bound
    };
    assert!(
        !bound.is_zero() && bound < params.modulus().as_ref() / BigUint::from(2u8),
        "PRF noise authority must be nonvacuous below q/2"
    );

    let plan = PrfTestPlan {
        model,
        root_seed,
        support,
        layout_id: public_layout.layout_id.0,
        active_count: active.len(),
        coefficient_events,
        failure_exponent: security,
        bound: bound.clone(),
        worst_bound: worst_bound.clone(),
    };
    println!(
        "prf_noise_simulation model={model:?} failure_exponent={security} coefficient_events={coefficient_events} selected_bound={bound} worst_bound={worst_bound}"
    );
    plan
}

// GPU execution follows the simulation in the same integration invocation.
use mxx_bgg::{BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{DslContext, GraphValue, GraphValueSchema};
use mxx_exponent_lut::{
    bgg::{BggPreparedReadout, encoding::ExponentLutEncodingSampler},
    noise_refresh::{
        pbc::{PbcCell, PbcTrustedSelectorBits, clear_pbc_inner_product},
        prf::{PreparedSparseLwrPrf, PrfSelectionInputs},
    },
    program::ProgramValue,
};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    artifact::{ArtifactConfidentiality, ArtifactType, ProductionId},
    encoding::spec_hash,
};
use mxx_primitives::{
    element::PolyElem,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{Poly, dcrt::poly::DCRTPoly},
};
use mxx_runtime::{
    Backend, ExecutionConfig, RuntimeValue,
    artifact::{ArtifactStore, FileArtifactStore},
    backend::poly::gpu::gpu_backend_on,
    execute, execute_with_config,
    transcript::SamplingMode,
};
use rayon::prelude::*;
use std::{collections::BTreeMap, time::Instant};
fn monomial_input<B: Backend>(
    backend: &mut B,
    modulus: &BigInt,
    n: usize,
    exponent: usize,
) -> B::Matrix {
    let mut coefficients = vec![IntExpr::constant(0); n];
    coefficients[exponent % n] = IntExpr::constant(if (exponent / n) % 2 == 0 { 1 } else { -1 });
    backend
        .constant_matrix(
            &mxx_ir_core::types::ConcreteMatrixType::scalar(modulus.clone(), n),
            &mxx_ir_core::node::ConstantMatrix::Polynomial { coefficients },
            &ParamEnv::default(),
        )
        .unwrap_or_else(|error| panic!("native monomial input: {error}"))
}

#[test]
#[serial_test::serial]
fn test_gpu_lwr_prf_one_output_round_trip() {
    if use_bfv() {
        let fixture = bfv_prf();
        println!("prf_noise_simulation={}", fixture.noise_report);
        run_bfv(fixture);
        return;
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let parameters = PrfTestParameters::from_env();
    let plan = simulate_prf(parameters);
    let (params, setup, profile) = parameters.build();
    let PrfTestParameters {
        log_n: _,
        depth,
        crt_bits,
        base_bits,
        universe,
        weight,
        q_native,
        p,
        security,
        ..
    } = parameters;
    let n = params.ring_dimension() as usize;
    let layout = setup.layout.clone();
    let modulus = BigInt::from(params.modulus().as_ref().clone());
    let (moduli, _, _) = params.to_crt();
    let generated = generate_key_layout(
        &PbcParameters::paper_evaluation(universe, weight),
        PbcRootSeed(plan.root_seed),
        &plan.support,
    )
    .unwrap();
    let public_layout = &generated.public_layout;
    assert_eq!(
        public_layout.layout_id.0, plan.layout_id,
        "reuse exact locally simulated PBC layout"
    );
    let active = PbcActiveCellIndex::build(public_layout).unwrap();
    assert_eq!(active.len(), plan.active_count);
    assert_eq!(plan.coefficient_events, n * layout.public_key_columns());
    assert_eq!(plan.failure_exponent, security);
    let model = plan.model;
    let bound = plan.bound;
    let worst_bound = plan.worst_bound;
    assert!(!bound.is_zero() && bound < params.modulus().as_ref() / BigUint::from(2u8));
    println!(
        "prf_one_output_plan N={n} depth={depth} crt_bits={crt_bits} base_bits={base_bits} universe={universe} weight={weight} Q={q_native} P={p} active={} output_count=1 model={model:?} failure_exponent={security} coefficient_events={} selected_bound={bound} worst_bound={worst_bound} parameter_simulation=true plan_origin=current_run",
        active.len(),
        plan.coefficient_events
    );
    let program = SparseLwrPrfProgram::new(
        profile,
        public_layout.bucket_width,
        public_layout.parameters.bucket_count,
    )
    .unwrap();
    let mut rng = rand::rng();
    let ring = layout.ring();
    let columns = layout.public_key_columns();
    let sampler = ExponentLutEncodingSampler {
        layout: layout.clone(),
        exponent_layout: profile.layout,
        gaussian_sigma: Some(setup.encoding_error_sigma.clone()),
        gaussian_max_coefficient_bound: Some(setup.encoding_error_bound.clone()),
    };
    let t = sampler.sample_secret().unwrap();
    let s = sampler.sample_secret().unwrap();
    let hash = ring.bytes_input("prf-hash", 32);
    let key_id = rng.random();
    let bits = PbcTrustedSelectorBits::from_schedule(&generated, &ring, key_id).unwrap();
    let runtime_bits = bits.runtime_binding(&ring).unwrap();
    let selectors = sampler
        .sample_pbc_selectors(
            bits.family().clone(),
            t.clone(),
            t.clone(),
            hash.clone(),
            public_layout,
            key_id,
        )
        .unwrap();
    let initial = mxx_bgg::BggPublicKeySampler { layout: layout.clone() }
        .sample(hash.clone(), b"prf-initial".as_slice(), &[false])
        .remove(1);
    let bgg_public = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: layout.gadget_base.clone(),
        digit_count: layout.digit_count.into(),
    };
    let scheme = mxx_exponent_lut::bgg::BggScheme::new(bgg_public, profile.layout);
    let labels = ring.input_family("prf-monomials", active.len(), (1, 1));
    let fixed_selectors = mxx_exponent_lut::bgg::encoding::SetupRhsFamily::new_for_layout(
        selectors.clone(),
        public_layout.layout_id,
    )
    .unwrap();
    let blocks =
        program.selection_inputs(public_layout, &fixed_selectors, &labels, key_id).unwrap();
    let (public, prepared) = PreparedSparseLwrPrf::new(
        &scheme,
        program.clone(),
        ProgramValue::Scalar(initial.clone()),
        &blocks,
    )
    .unwrap();
    let public_outputs = public;
    let helpers = sampler
        .sample_native_readout_helper_families(
            t.clone(),
            hash.clone(),
            profile.native_width(),
            b"prf-mask".as_slice(),
            b"prf-helper".as_slice(),
        )
        .unwrap();
    let gadget = ring.gadget(2, layout.gadget_base.clone(), layout.digit_count);
    let readout = program
        .prepare_readout(
            &scheme,
            &public_outputs,
            &(helpers.sigmas.clone(), helpers.switch_ciphertexts.clone(), gadget.clone()),
        )
        .unwrap();
    let public_output =
        program.project_readout(&scheme, &public_outputs, &readout, &helpers.public_masks).unwrap();
    let context = DslContext::new("one-output-prf-preprocessing")
        .private_output("t", t)
        .unwrap()
        .public_output("initial", initial.matrix)
        .unwrap()
        .public_output("selectors", selectors)
        .unwrap()
        .public_output("output-public", public_output)
        .unwrap()
        .public_output("public-masks", helpers.public_masks.clone())
        .unwrap()
        .public_output("readout", readout.clone())
        .unwrap();
    let readout_schema: <BggPreparedReadout as GraphValue>::Schema =
        serde_json::from_slice(&serde_json::to_vec(&readout.schema()).unwrap()).unwrap();
    let (context, descriptor) = prepared.export(&scheme, context, "fixed-prf").unwrap();
    // The exact operation schemas must survive the same serialization boundary
    // as the matrices. Online evaluation cannot reuse producer graph handles.
    let descriptor = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
    let preprocessing = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let nonce = rng.random();
    let production = ProductionId {
        spec_hash: spec_hash(&preprocessing.source, &preprocessing.bindings).unwrap(),
        execution_nonce: nonce,
    };
    let manifest =
        mxx_ir_core::artifact::export_validated_manifest(production.clone(), &preprocessing)
            .unwrap();
    let payload = manifest
        .artifacts
        .values()
        .map(|a| match &a.artifact_type {
            ArtifactType::Preimage { matrix, max_coefficient_bound } |
            ArtifactType::SmallMatrix { matrix, max_coefficient_bound } => {
                (matrix.rows *
                    matrix.columns *
                    matrix.ring_dimension *
                    (1 + max_coefficient_bound.to_biguint().unwrap().bits().div_ceil(8).max(1)
                        as usize)) as u128 *
                    a.family_count.unwrap_or(1) as u128
            }
            _ => 0,
        })
        .sum::<u128>();
    println!("prf_one_output_compact_coefficient_payload_bytes={payload}");
    let public_vector =
        (0..universe).map(|_| rng.random_range(0..q_native as u64)).collect::<Vec<_>>();
    let z = clear_pbc_inner_product(
        public_layout,
        generated.private_schedule(),
        &public_vector,
        q_native as u64,
    )
    .unwrap() as usize;
    let f = &program.readout.canonical_coefficients().unwrap()[z % (q_native / 2)];
    let digit = if z < q_native / 2 { f.clone() } else { BigInt::one() - f };
    // Independent arithmetic specification of the repository's signed readout:
    // 2g = sign(z) * (2 floor(P*(z mod (Q/2))/Q) + 1).
    // This is not the unshifted conventional LWR residue floor(P*z/Q).
    let conventional_lwr = p * z / q_native;
    let magnitude = BigInt::from(2 * (p * (z % (q_native / 2)) / q_native) + 1);
    let doubled_g = if z < q_native / 2 { magnitude } else { -magnitude };
    assert_eq!(
        &digit * BigInt::from(2u8) - BigInt::one(),
        doubled_g,
        "canonical readout must match the independently specified signed LWR map"
    );
    let build_online = |production: ProductionId, manifest: mxx_ir_core::artifact::Manifest| {
        let imported_matrix = |name: &str, rows: usize, cols: usize, private: bool| {
            ring.artifact_input(
                production.clone(),
                name,
                (rows, cols),
                if private {
                    ArtifactConfidentiality::Private
                } else {
                    ArtifactConfidentiality::Public
                },
            )
        };
        let import_matrix_family = |name: &str| {
            let artifact = &manifest.artifacts[name];
            let ArtifactType::Matrix(matrix) = &artifact.artifact_type else {
                panic!("matrix schema")
            };
            ring.family_artifact_input(
                production.clone(),
                name,
                artifact.family_count.unwrap(),
                (matrix.rows, matrix.columns),
                ArtifactConfidentiality::Public,
            )
        };
        let imported_t = imported_matrix("t", 1, 2, true);
        let imported_initial = BggPublicKeyWire {
            matrix: imported_matrix("initial", 2, columns, false),
            reveal_plaintext: false,
        };
        let initial_vector = sampler
            .sample_encodings_for_public_matrices(
                s.clone(),
                Some(imported_t.clone()),
                &[imported_initial],
                &[ring.identity(1)],
            )
            .unwrap()
            .remove(0);
        let prepared = PreparedSparseLwrPrf::import(
            &scheme,
            program.clone(),
            &descriptor,
            production.clone(),
            descriptor.identity().unwrap(),
        )
        .unwrap();
        let accumulation =
            prepared.evaluate(&scheme, ProgramValue::Scalar(initial_vector)).unwrap();

        let imported_readout = readout_schema
            .artifact_input(production.clone(), "readout", ArtifactConfidentiality::Public)
            .unwrap();
        let mask_publics = import_matrix_family("public-masks");
        let masks = {
            let source_family = mask_publics;
            mxx_dsl::parallel(source_family.count().clone(), |family_index| {
                let matrix = source_family.at(&family_index);
                let index = family_index;
                Ok({
                    let sigma = profile
                        .layout
                        .automorphism_index_expression(
                            n,
                            IntExpr::Add(
                                Box::new(index.expression()?),
                                Box::new(IntExpr::constant(1)),
                            ),
                        )
                        .unwrap();
                    let mask_sigma = s.clone().ring_automorphism(sigma);
                    mxx_bgg::BggEncodingSampler {
                        layout: layout.clone(),
                        gaussian_sigma: sampler.gaussian_sigma.clone(),
                        gaussian_max_coefficient_bound: sampler
                            .gaussian_max_coefficient_bound
                            .clone(),
                    }
                    .sample(
                        s.clone(),
                        Some(mask_sigma),
                        &[
                            BggPublicKeyWire {
                                matrix: ring.zero((2, columns)),
                                reveal_plaintext: false,
                            },
                            BggPublicKeyWire { matrix, reveal_plaintext: false },
                        ],
                        &[ring.identity(1)],
                    )
                    .unwrap()
                    .remove(1)
                    .vector
                })
            })
        }
        .unwrap();
        let output = program
            .evaluate_readout(&scheme, &accumulation, &imported_readout, &masks)
            .unwrap()
            .at(0);
        let inverse_two = (&modulus + BigInt::one()) / BigInt::from(2u8);
        let g = ring.polynomial([(&doubled_g * inverse_two).into()]);
        let anchor = s.clone() * import_matrix_family("output-public").at(0) -
            g * (imported_t * gadget.clone());
        let estimate_online = DslContext::new("one-output-prf-production-online")
            .private_output("vector", output.clone())
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .unwrap();
        let online = DslContext::new("one-output-prf-online")
            .private_output("vector", output.clone())
            .unwrap()
            .private_output("anchor", anchor.clone())
            .unwrap()
            .private_output("error", output - anchor)
            .unwrap()
            .build()
            .unwrap();
        assert!(
            !online.graph.scopes().values().flat_map(|scope| scope.nodes()).any(|node| matches!(
                node.kind(),
                mxx_ir_core::node::NodeKind::GadgetDecompose { .. }
            )),
            "online PRF must consume persisted decompositions"
        );
        let online = online
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .expect("validate online graph before expensive preprocessing");
        (online, estimate_online)
    };
    let (_, estimate_online) = build_online(production, manifest);
    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    assert!(!devices.is_empty(), "actual CUDA GPU required");
    let gpu_params = mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
        params.ring_dimension(),
        moduli,
        base_bits,
        None,
    );
    {
        use mxx_bench_estimator::{
            estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
        };
        let backends = devices
            .iter()
            .map(|device| (gpu_backend_on([gpu_params.clone()], [*device]), *device))
            .collect();
        let mut estimator = GpuNodeMeasurementBackend::new(
            backends,
            MeasurementHarnessConfig {
                warm_up_iterations: std::env::var("MXX_EXPONENT_LUT_BENCH_WARMUPS")
                    .map(|v| v.parse().unwrap())
                    .unwrap_or(1),
                measured_iterations: std::env::var("MXX_EXPONENT_LUT_BENCH_ITERATIONS")
                    .map(|v| v.parse().unwrap())
                    .unwrap_or(2),
                ..Default::default()
            },
        );
        let stages = [("preprocess", &preprocessing), ("online", &estimate_online)];
        for (_, graph) in stages {
            estimate(graph, &mut estimator).expect("collect exact PRF graph measurements");
        }
        estimator.measure_collected().expect("measure exact PRF operations");
        let information_bits = p.ilog2();
        for (stage, graph) in stages {
            let report = estimate(graph, &mut estimator).expect("estimate exact PRF graph");
            println!(
                "prf_one_output_estimate stage={stage} total_work_seconds={} cumulative_wave_seconds={} ideal_dag_seconds={} information_bits={information_bits} stage_work_per_information_bit={} stage_cumulative_wave_per_information_bit={} oracle_online_work_excluded=true preprocessing_includes_public_output_projection=true",
                report.total_work_seconds,
                report.total_time_seconds,
                report.critical_path_seconds,
                report.total_work_seconds / f64::from(information_bits),
                report.total_time_seconds / f64::from(information_bits)
            );
        }
    }
    let mut backend = gpu_backend_on([gpu_params], devices);
    for exponent in 0..2 * n {
        let value = monomial_input(&mut backend, &modulus, n, exponent);
        let actual = backend.gather_matrix_for_host(&value).unwrap().to_cpu_matrix();
        let expected = DCRTPoly::const_rotate_poly(&params, exponent % n);
        let expected = if exponent >= n { -expected } else { expected };
        assert_eq!(actual, DCRTPolyMatrix::from_poly_vec_row(&params, vec![expected]));
    }
    let mut runtime_inputs = runtime_bits.materialize(&mut backend).unwrap();
    runtime_inputs
        .insert("prf-hash".into(), RuntimeValue::Bytes(rng.random::<[u8; 32]>().to_vec()));
    std::fs::create_dir_all("test_data/prf_one_output_round_trip").unwrap();
    // Retain completed preprocessing even if a later online assertion fails.
    // This is diagnostic storage, not an automatically reusable checkpoint.
    let directory = tempfile::tempdir_in("test_data/prf_one_output_round_trip").unwrap().keep();
    println!(
        "prf_one_output_artifact_directory={} retained=true cleanup=manual",
        directory.display()
    );
    let monomials = active
        .iter()
        .map(|(bucket, slot, _)| {
            let z = match public_layout.cells[bucket][slot] {
                PbcCell::Real { coordinate, .. } => public_vector[coordinate] as usize,
                PbcCell::Dummy => 0,
                PbcCell::Padding => unreachable!(),
            };
            let exponent = z << profile.layout.stride_log2;
            RuntimeValue::Matrix(monomial_input(&mut backend, &modulus, n, exponent).into())
        })
        .collect::<Vec<_>>();
    runtime_inputs.insert("prf-monomials".into(), RuntimeValue::IndexedFamily(monomials));
    let mut store = FileArtifactStore::new(&directory).unwrap();
    let start = Instant::now();
    let produced =
        execute(&preprocessing, &mut backend, runtime_inputs, &mut store, SamplingMode::Fresh)
            .expect("real PRF preprocessing");
    let production = produced.production_id.unwrap();
    println!("prf_one_output_preprocess_seconds={}", start.elapsed().as_secs_f64());
    drop(store);
    let mut store = FileArtifactStore::open(&directory).unwrap();
    let actual_manifest = store.load_manifest(&production).unwrap();
    let (online, _) = build_online(production, actual_manifest);
    for run in 0..2 {
        let start = Instant::now();
        let result = execute_with_config(
            &online,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .expect("real one-output PRF online execution");
        let mut matrix = |name: &str| {
            let RuntimeValue::Matrix(value) = &result.outputs[name] else { panic!("matrix") };
            backend.gather_matrix_for_host(value).unwrap().to_cpu_matrix()
        };
        let error = matrix("error");
        assert_eq!(matrix("vector"), matrix("anchor") + error.clone());
        let maximum = error
            .get_row(0)
            .par_iter()
            .flat_map_iter(|poly| poly.coeffs())
            .map(|c| c.value().min(&(params.modulus().as_ref() - c.value())).clone())
            .max()
            .unwrap();
        println!(
            "prf_one_output_noise_observed run={run} max_error={maximum} strict_bound={bound} model={model:?}"
        );
        assert!(!maximum.is_zero(), "actual PRF error must be nonzero");
        assert!(maximum < bound, "actual PRF noise exceeds existing readout authority");
        println!(
            "prf_one_output_pass run={run} online_seconds={} z={z} conventional_lwr={conventional_lwr} centered_prf_digit={digit} doubled_g={doubled_g} plaintext_representation=d-1/2 relation=c-sA+g(tG) coefficients={} max_error={maximum} strict_bound={bound} worst_bound={worst_bound} model={model:?} failure_exponent={security}",
            start.elapsed().as_secs_f64(),
            n * columns
        );
    }
}

fn run_bfv(fixture: bfv::BfvRoundTrip) {
    use mxx_primitives::{
        matrix::PolyMatrix,
        poly::{
            PolyParams,
            dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
        },
    };
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU round trip requires a detected device");
    bfv::run(
        fixture,
        |p| {
            let rings = p.levels().iter().map(|p| {
                GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None)
            });
            mxx_runtime::backend::poly::gpu::gpu_backend_on(rings, devices)
        },
        |matrix, name, q, bound| {
            for shard in matrix.shards() {
                let host = shard.value.to_cpu_matrix();
                for row in 0..host.row_size() {
                    for col in 0..host.col_size() {
                        for value in host.entry(row, col).coeffs_biguints() {
                            let magnitude = value.clone().min(q - &value);
                            assert!(&magnitude <= bound, "{name}: {magnitude} > {bound}");
                        }
                    }
                }
            }
        },
    );
}

/// One PRF output, including its terminal digit readout, with the actual
/// executed Program as the noise authority and a reopened artifact store.
fn bfv_prf() -> bfv::BfvRoundTrip {
    use bfv::{BfvRoundTrip, trace_prf};
    use mxx_dsl::Family;
    use mxx_exponent_lut::{
        bfv::{
            BfvParameters, BfvPlanRequest, BfvReadoutWeights, BfvRhsPlaintextFamily, BfvRow,
            BfvScheme,
        },
        layout::ExpLayout,
        noise_refresh::pbc::{PbcLayoutId, derive_lwr_vector},
        program::FamilyRange,
        readout::ExponentLUTReadoutScheme,
    };
    use mxx_ir_core::artifact::Manifest;
    let n: u32 = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(4);
    let layout = ExpLayout { stride_log2: 0, width_log2: n.trailing_zeros() };
    let profile = SparseLwrPrfProfile::new(layout, 2).unwrap();
    let program = SparseLwrPrfProgram::new(profile, 2, 1).unwrap();
    let (parameters, _) = BfvParameters::plan(&BfvPlanRequest {
        ring_dimension: n,
        widths: vec![1, n as usize],
        slots: vec![2, 1],
        eta: 2,
        crt_bits: 17,
        terminal_depth: 2,
        max_crt_depth: 32,
        memory_budget_bytes: 1 << 30,
        terminal_value_bound: 4u8.into(),
    })
    .unwrap();
    let scheme = BfvScheme::new(parameters.clone(), rand::random());
    let high = scheme.ring(0).unwrap();
    let middle = scheme.ring(1).unwrap();
    let seed = high.bytes_input("hash-key", 32);
    let r = scheme.sample_payload_secret().unwrap();
    let key =
        scheme.sample_input_key(0, seed.clone(), b"initial".as_slice(), rand::random()).unwrap();
    let material = scheme.prepare_input(&r, &key, high.polynomial([1.into()])).unwrap();
    let selected = usize::from(rand::random::<bool>());
    let vector =
        derive_lwr_vector(PbcLayoutId(rand::random()), b"one-output", 2, profile.q().unwrap())
            .unwrap();
    let selectors = scheme
        .sample_private_rhs_family(
            &r,
            0,
            &BfvRhsPlaintextFamily {
                payloads: Family::pack(
                    (0..2).map(|i| middle.polynomial([u8::from(i == selected).into()])).collect(),
                )
                .unwrap(),
                id: rand::random(),
            },
            seed.clone(),
            b"selectors".as_slice(),
            rand::random(),
        )
        .unwrap();
    let (public, prepared) = PreparedSparseLwrPrf::new(
        &scheme,
        program.clone(),
        ProgramValue::Scalar((key.clone(), None)),
        &[PrfSelectionInputs {
            selectors,
            labels: Family::pack(
                vector.iter().map(|v| layout.mono(&middle, &BigInt::from(*v)).unwrap()).collect(),
            )
            .unwrap(),
            active: FamilyRange::bounded(0, 2, 2).unwrap(),
            id: rand::random(),
        }],
    )
    .unwrap();
    let public = ProgramValue::Scalar(public.at(0));
    let bank = scheme
        .sample_mask_bank(1, layout, seed.clone(), b"readout-bank".as_slice(), rand::random())
        .unwrap();
    let readout_material = scheme
        .sample_readout_material(
            &r,
            &program.readout,
            &bank,
            seed,
            b"readout".as_slice(),
            rand::random(),
            Some(BfvReadoutWeights::Shared([vec![1.into()], vec![]])),
            None,
        )
        .unwrap();
    let readout = scheme.prepare_readout(&program.readout, &public, &readout_material).unwrap();
    let trace = trace_prf(&parameters, prepared.program());
    let mut terminal_noise = parameters
        .terminal_noise_bound(
            1,
            [(&trace.wires[&program.accumulation.outputs()[0]].bound, n as usize, &1u8.into())],
            readout_material.target_bound().clone(),
        )
        .unwrap();
    let model = mxx_exponent_lut::noise_refresh::simulation::NoiseModelKind::from_env().unwrap();
    let worst_bound = terminal_noise.error.clone();
    if model == mxx_exponent_lut::noise_refresh::simulation::NoiseModelKind::AverageCase {
        let config = mxx_exponent_lut::noise_refresh::simulation::AverageCaseConfig {
            failure_exponent: std::env::var("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS")
                .map(|v| v.parse().unwrap())
                .unwrap_or(1),
            allow_average_acceptance: true,
            ..Default::default()
        };
        let moments = parameters
            .terminal_noise_moments(
                1,
                [(&trace.wires[&program.accumulation.outputs()[0]], n as usize)],
            )
            .unwrap();
        let z = config.z_squared_for_events(&BigUint::from(2 * n)).unwrap();
        terminal_noise.error = moments.tail_bound(&z);
    }
    assert!(parameters.terminal_noise_fits(2, &terminal_noise).unwrap());
    let (producer, prf_descriptor) =
        prepared.export(&scheme, DslContext::new("bfv-prf-preprocess"), "prf").unwrap();
    let (producer, input_descriptor) =
        scheme.export_input_material(&key, &material, producer, "initial").unwrap();
    let (producer, readout_descriptor) =
        scheme.export_prepared_readout(&readout, producer, "readout").unwrap();
    let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let online = Box::new(move |production: ProductionId, manifest: Manifest| {
        let prf_descriptor =
            serde_json::from_slice(&serde_json::to_vec(&prf_descriptor).unwrap()).unwrap();
        let prepared = PreparedSparseLwrPrf::import(
            &scheme,
            program.clone(),
            &prf_descriptor,
            production.clone(),
            prf_descriptor.identity().unwrap(),
        )
        .unwrap();
        let readout_descriptor =
            serde_json::from_slice(&serde_json::to_vec(&readout_descriptor).unwrap()).unwrap();
        let readout = scheme
            .import_prepared_readout(
                &readout_descriptor,
                production.clone(),
                &program.readout,
                &readout.binding(),
                *public.scalar().unwrap().0.id(),
                readout.id(),
            )
            .unwrap();
        let public_result =
            scheme.project_readout(&program.readout, &public, &readout, &()).unwrap();
        let (key, material) = scheme
            .import_input_material(&input_descriptor, production.clone(), *key.id(), 0)
            .unwrap();
        let s = scheme.sample_mask_secret().unwrap();
        let input = ProgramValue::Scalar((scheme.encode(&s, &key, &material).unwrap(), None));
        let encoded = ProgramValue::Scalar(prepared.evaluate(&scheme, input).unwrap().at(0));
        let masks = scheme.prepare_online_masks(&s, &readout.mask_bank()).unwrap();
        let value = scheme.evaluate_readout(&program.readout, &encoded, &readout, &masks).unwrap();
        let e = vector[selected];
        let d = &program.readout.canonical_coefficients().unwrap()[e % n as usize];
        let expected = if e < n as usize { d.clone() } else { BigInt::from(1u8) - d };
        let low = scheme.ring(2).unwrap();
        let expected =
            BfvRow::new([[low.polynomial([expected.into()]), low.polynomial([0.into()])]]).unwrap();
        let error = value.values.scalar().unwrap().clone() -
            scheme.mask_at(&s, 2).unwrap() * public_result.values.scalar().unwrap().clone() -
            expected;
        let consumer = DslContext::new("bfv-prf-online")
            .output("error", error)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest)]),
            )
            .unwrap();
        consumer
    });
    let noise_report = serde_json::json!({"scheme": "bfv", "model": model, "worst_bound": worst_bound.to_string(), "output_count": 1, "readout_error_bound": terminal_noise.error.to_string(), "readout_value_bound": terminal_noise.coefficient_bound.to_string()});
    let bounds = (0..2)
        .map(|column| {
            (
                format!("error.{column}"),
                (
                    parameters.level(2).unwrap().modulus().as_ref().clone(),
                    terminal_noise.error.clone(),
                ),
            )
        })
        .collect();
    BfvRoundTrip { parameters, producer, online, bounds, noise_report }
}

mod bfv {
    use mxx_exponent_lut::bfv::{BfvNoiseBound, BfvParameters};
    use mxx_ir_core::{
        ValidatedGraph,
        artifact::{Manifest, ProductionId},
    };
    use mxx_runtime::{
        Backend, RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigUint;
    use std::collections::BTreeMap;
    pub struct BfvRoundTrip {
        pub parameters: BfvParameters,
        pub producer: ValidatedGraph,
        pub online: Box<dyn Fn(ProductionId, Manifest) -> ValidatedGraph>,
        pub bounds: BTreeMap<String, (BigUint, BigUint)>,
        pub noise_report: serde_json::Value,
    }

    pub fn run<B: Backend>(
        fixture: BfvRoundTrip,
        make_backend: impl FnOnce(&BfvParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &str, &BigUint, &BigUint),
    ) {
        std::fs::create_dir_all("test_data/bfv_round_trip").unwrap();
        let directory = tempfile::tempdir_in("test_data/bfv_round_trip").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend = make_backend(&fixture.parameters);
        let produced = execute(
            &fixture.producer,
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let manifest = store.load_manifest(&production).unwrap();
        let consumer = (fixture.online)(production, manifest);
        let result =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        for (name, value) in result.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial output") };
            let (q, bound) = &fixture.bounds[&name];
            check_matrix(&matrix, &name, q, bound);
        }
    }

    pub fn trace_prf(
        parameters: &BfvParameters,
        execution: &mxx_exponent_lut::program::ExponentLutProgram,
    ) -> mxx_exponent_lut::bfv::noise::BfvNoiseTrace {
        let inputs = BTreeMap::from([(
            *execution.inputs().keys().next().unwrap(),
            mxx_exponent_lut::bfv::noise::BfvNoiseValue {
                level: 0,
                bound: BfvNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
                payload_l2_squared: 1u8.into(),
                moments: Some(mxx_exponent_lut::noise_utils::NoiseMoments {
                    mean_numerator: 0.into(),
                    mean_denominator: 1u8.into(),
                    mean_radius: mxx_exponent_lut::noise_utils::NonnegativeRational::zero(),
                    variance: mxx_exponent_lut::noise_utils::NonnegativeRational::new(
                        parameters.eta().into(),
                        2u8.into(),
                    )
                    .unwrap(),
                }),
                public: false,
                range: None,
                one_hot: false,
            },
        )]);
        let mut rhs_bounds = BTreeMap::new();
        let mut materials = BTreeMap::new();
        for (step, gate) in execution.steps().iter().enumerate() {
            let mxx_exponent_lut::program::ProgramStep::Gate(
                mxx_exponent_lut::program::Gate::MulPrivate { rhs, .. },
            ) = gate
            else {
                panic!("composed PRF must contain direct products")
            };
            let (material, bound) =
                mxx_exponent_lut::bfv::noise::BfvGateNoiseMaterial::from_one_hot_labels(
                    step,
                    None,
                    &[
                        mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit(),
                        mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit(),
                    ],
                )
                .unwrap();
            rhs_bounds.insert(*rhs, bound);
            materials
                .insert(mxx_exponent_lut::program::ProgramGateInstance::Gate { step }, material);
        }
        mxx_exponent_lut::bfv::noise::trace_program(
            parameters,
            execution,
            &mxx_exponent_lut::program::ProgramBindings::new(
                &inputs,
                &rhs_bounds,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                Some(&materials),
            ),
            &mxx_exponent_lut::program::ProgramFamilyBindings::new(),
        )
        .unwrap()
    }
}
