#![cfg(feature = "gpu")]

//! One actual normalized LWR PRF output, without any refresh label expansion.
//! Select BGG or RNS-BGG with MXX_EXPONENT_LUT_TEST_SCHEME; CPU simulation precedes GPU execution.
fn use_rns_bgg() -> bool {
    match std::env::var("MXX_EXPONENT_LUT_TEST_SCHEME").as_deref().unwrap_or("bgg") {
        "bgg" => false,
        "rns-bgg" => true,
        other => panic!("MXX_EXPONENT_LUT_TEST_SCHEME must be bgg or rns-bgg, got {other}"),
    }
}
use bigdecimal::BigDecimal;
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
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    if use_rns_bgg() {
        let started = std::time::Instant::now();
        let fixture = rns_bgg_prf();
        println!("prf_graph_seconds={}", started.elapsed().as_secs_f64());
        println!("prf_noise_simulation={}", fixture.noise_report);
        let manifest = mxx_ir_core::artifact::export_validated_manifest(
            ProductionId {
                spec_hash: spec_hash(&fixture.producer.source, &fixture.producer.bindings).unwrap(),
                execution_nonce: rand::random(),
            },
            &fixture.producer,
        )
        .unwrap();
        let artifact_payload_bytes: u128 = manifest
            .artifacts
            .values()
            .map(|artifact| {
                let bytes = match &artifact.artifact_type {
                    ArtifactType::Matrix(matrix) => {
                        let parameters = fixture
                            .parameters
                            .levels()
                            .iter()
                            .find(|p| BigInt::from(p.modulus().as_ref().clone()) == matrix.modulus)
                            .unwrap();
                        matrix.rows as u128 *
                            matrix.columns as u128 *
                            matrix.ring_dimension as u128 *
                            parameters.to_crt().0.len() as u128 *
                            8
                    }
                    ArtifactType::SmallMatrix { matrix, max_coefficient_bound } |
                    ArtifactType::Preimage { matrix, max_coefficient_bound } => {
                        matrix.rows as u128 *
                            matrix.columns as u128 *
                            matrix.ring_dimension as u128 *
                            (1 + max_coefficient_bound
                                .to_biguint()
                                .unwrap()
                                .bits()
                                .div_ceil(8)
                                .max(1)) as u128
                    }
                    ArtifactType::Bytes { length } => *length as u128,
                    other => panic!("unexpected PRF artifact: {other:?}"),
                };
                bytes * artifact.family_count.unwrap_or(1) as u128
            })
            .sum();
        println!("prf_artifact_payload_bytes={artifact_payload_bytes}");
        if std::env::var("MXX_EXPONENT_LUT_BENCH_STAGE").as_deref() == Ok("graph") {
            return;
        }
        if std::env::var("MXX_EXPONENT_LUT_BENCH_STAGE").as_deref() == Ok("estimate") {
            use mxx_bench_estimator::{
                estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
            };
            use mxx_primitives::poly::dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids};
            let online = (fixture.online)(manifest.production_id.clone(), manifest, false);
            let devices = detected_gpu_device_ids();
            assert!(!devices.is_empty(), "benchmark estimate requires CUDA");
            println!("prf_estimate_gpu_devices={devices:?}");
            let backends = devices
                .into_iter()
                .map(|device| {
                    let rings = fixture.parameters.levels().iter().map(|p| {
                        GpuDCRTPolyParams::new(
                            p.ring_dimension(),
                            p.to_crt().0,
                            p.base_bits(),
                            None,
                        )
                    });
                    (gpu_backend_on(rings, [device]), device)
                })
                .collect();
            let mut estimator = GpuNodeMeasurementBackend::new(
                backends,
                MeasurementHarnessConfig {
                    warm_up_iterations: std::env::var("MXX_EXPONENT_LUT_BENCH_WARMUPS")
                        .map(|v| v.parse().unwrap())
                        .unwrap_or(1),
                    measured_iterations: std::env::var("MXX_EXPONENT_LUT_BENCH_ITERATIONS")
                        .map(|v| v.parse().unwrap())
                        .unwrap_or(1),
                    ..Default::default()
                },
            );
            let stages = [("preprocess", &fixture.producer), ("online_eval", &online)];
            for (_, graph) in stages {
                estimate(graph, &mut estimator).expect("collect one-output PRF operations");
            }
            estimator.measure_collected().expect("measure one-output PRF operations");
            let bits = fixture.noise_report["lwr_p"].as_u64().unwrap().ilog2();
            for (stage, graph) in stages {
                let report = estimate(graph, &mut estimator).expect("estimate one-output PRF");
                println!(
                    "prf_one_output_estimate={}",
                    serde_json::json!({
                        "scheme": "rns-bgg", "stage": stage, "output_count": 1,
                        "information_bits": bits,
                        "work_seconds_per_information_bit": report.total_work_seconds / f64::from(bits),
                        "estimated_seconds_per_information_bit": report.total_time_seconds / f64::from(bits),
                        "selector_setup_amortized": false,
                        "verification_oracle_included": false,
                        "runtime_family_staging_timed": true,
                        "artifact_io_timed": true,
                        "executor_dispatch_model": "measured scalar IR node proxy",
                        "report": report,
                    })
                );
            }
            return;
        }
        run_rns_bgg(fixture);
        return;
    }

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

#[test]
fn test_cpu_rns_bgg_lwr_prf_one_output_round_trip() {
    use mxx_primitives::matrix::PolyMatrix;
    rns_bgg::run(
        rns_bgg_prf(),
        |parameters| mxx_runtime::backend::poly::cpu_backend(parameters.levels().iter().cloned()),
        |_| {},
        |matrix, name, q, bound| {
            for row in 0..matrix.row_size() {
                for col in 0..matrix.col_size() {
                    for value in matrix.entry(row, col).coeffs_biguints() {
                        let magnitude = value.clone().min(q - &value);
                        assert!(&magnitude <= bound, "{name}: {magnitude} > {bound}");
                    }
                }
            }
        },
    );
}

fn run_rns_bgg(fixture: rns_bgg::RnsBggRoundTrip) {
    use mxx_primitives::{
        matrix::PolyMatrix,
        poly::{
            PolyParams,
            dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
        },
    };
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU round trip requires a detected device");
    println!("prf_runtime_gpu_devices={devices:?}");
    rns_bgg::run(
        fixture,
        |p| {
            let rings = p.levels().iter().map(|p| {
                GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None)
            });
            mxx_runtime::backend::poly::gpu::gpu_backend_on(rings, devices)
        },
        |matrix| {
            for shard in matrix.shards() {
                shard.value.wait_until_ready();
            }
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
fn rns_bgg_prf() -> rns_bgg::RnsBggRoundTrip {
    use mxx_dsl::Family;
    use mxx_exponent_lut::{
        layout::ExpLayout,
        noise_refresh::pbc::{PbcLayoutId, derive_lwr_vector},
        program::FamilyRange,
        readout::ExponentLUTReadoutScheme,
        rns_bgg::{
            RnsBggParameters, RnsBggPlanRequest, RnsBggReadoutWeights, RnsBggRhsPlaintextFamily,
            RnsBggRow, RnsBggScheme,
        },
    };
    use mxx_ir_core::artifact::Manifest;
    use rns_bgg::{RnsBggRoundTrip, trace_prf};
    let (parameters, layout, counts, selected, labels, p_lwr) = if let Some(path) =
        std::env::var_os("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT")
    {
        use mxx_exponent_lut::noise_refresh::pbc::{PbcLayoutSeed, PbcPublicLayout};
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let (encoding_error_sigma, encoding_error_bound) = rns_bgg::report_encoding_noise(&report);
        let meta = report.get("numerical_run").unwrap_or(&report);
        let n = meta["n"].as_u64().unwrap() as u32;
        let primes: Vec<Vec<u64>> =
            serde_json::from_value(report["selected_candidate"]["level_primes"].clone()).unwrap();
        let parameters = RnsBggParameters::new(
            primes
                .into_iter()
                .map(|primes| {
                    DCRTPolyParams::new(
                        n,
                        primes.len(),
                        meta["crt_bits"].as_u64().unwrap() as usize,
                        meta["base_bits"].as_u64().unwrap() as u32,
                        Some(primes),
                        None,
                    )
                })
                .collect(),
            encoding_error_sigma,
            1,
        )
        .unwrap();
        assert_eq!(parameters.encoding_error_bound(), &encoding_error_bound);
        let q = meta["lwr_q"].as_u64().unwrap() as usize;
        let p = meta["lwr_p"].as_u64().unwrap() as usize;
        let width_log2 = q.trailing_zeros() - 1;
        let layout = ExpLayout { stride_log2: n.trailing_zeros() - width_log2, width_log2 };
        let universe = meta["lwr_dimension"].as_u64().unwrap() as usize;
        let weight = meta["lwr_weight"].as_u64().unwrap() as usize;
        let public = PbcPublicLayout::build(
            &PbcParameters::paper_evaluation(universe, weight),
            PbcLayoutSeed(serde_json::from_value(meta["pbc_seed"].clone()).unwrap()),
            0,
        )
        .unwrap();
        let active = PbcActiveCellIndex::build(&public).unwrap();
        let counts = active.bucket_active_widths().collect::<Vec<_>>();
        assert_eq!(
            counts,
            serde_json::from_value::<Vec<usize>>(meta["selector_counts"].clone()).unwrap()
        );
        let vector =
            derive_lwr_vector(public.layout_id, &rand::random::<[u8; 32]>(), universe, q).unwrap();
        // Sample a valid weight-h test key on the archived public layout.
        // Distinct real coordinates occupy distinct selected buckets; other
        // buckets select their dummy. This fixture does not resample layout parameters.
        let mut rng = rand::rng();
        let occupied = sample(&mut rng, counts.len(), weight).into_vec();
        let mut support = std::collections::BTreeSet::new();
        let mut selected = Vec::new();
        let mut labels = Vec::new();
        for step in 0..counts.len() {
            let cells = active
                .bucket_iter(step)
                .unwrap()
                .map(|(bucket, slot, _)| &public.cells[bucket][slot])
                .collect::<Vec<_>>();
            let index = if occupied.contains(&step) {
                let choices = cells
                    .iter()
                    .enumerate()
                    .filter_map(|(i, cell)| match cell {
                        PbcCell::Real { coordinate, .. } if !support.contains(coordinate) => {
                            Some((i, *coordinate))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let (index, coordinate) = choices[rng.random_range(0..choices.len())];
                support.insert(coordinate);
                index
            } else {
                cells.iter().position(|cell| matches!(cell, PbcCell::Dummy)).unwrap()
            };
            selected.push(index);
            labels.push(
                cells
                    .iter()
                    .map(|cell| match cell {
                        PbcCell::Real { coordinate, .. } => vector[*coordinate],
                        _ => 0,
                    })
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(support.len(), weight);
        let selected_sum: usize = labels.iter().zip(&selected).map(|(values, i)| values[*i]).sum();
        assert_eq!(selected_sum % q, support.iter().map(|i| vector[*i]).sum::<usize>() % q);
        (parameters, layout, counts, selected, labels, p)
    } else {
        let n: u32 = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(4);
        let layout = ExpLayout { stride_log2: 0, width_log2: n.trailing_zeros() };
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1, n as usize],
            slots: vec![2, 1],
            encoding_error_sigma: BigDecimal::from(4u8),
            crt_bits: 32,
            terminal_depth: 2,
            max_crt_depth: 32,
            terminal_value_bound: 4u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let vector =
            derive_lwr_vector(PbcLayoutId(rand::random()), b"one-output", 2, 2 * n as usize)
                .unwrap();
        (parameters, layout, vec![2], vec![usize::from(rand::random::<bool>())], vec![vector], 4)
    };
    let n = parameters.level(0).unwrap().ring_dimension();
    let terminal = counts.len();
    let output_level = terminal + 1;
    let profile = SparseLwrPrfProfile::new(layout, p_lwr / 2).unwrap();
    let program =
        SparseLwrPrfProgram::new(profile, *counts.iter().max().unwrap(), terminal).unwrap();
    let exponent = labels.iter().zip(&selected).map(|(values, i)| values[*i]).sum::<usize>() %
        profile.q().unwrap();
    let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
    let high = scheme.ring(0).unwrap();
    let seed = high.bytes_input("hash-key", 32);
    let r = scheme.sample_payload_secret().unwrap();
    let key =
        scheme.sample_input_key(0, seed.clone(), b"initial".as_slice(), rand::random()).unwrap();
    let material = scheme.prepare_input(&r, &key, high.polynomial([1.into()])).unwrap();
    let blocks = counts
        .iter()
        .enumerate()
        .map(|(step, count)| {
            let ring = scheme.ring(step + 1).unwrap();
            let selectors = scheme
                .sample_private_rhs_family(
                    &r,
                    step,
                    &RnsBggRhsPlaintextFamily {
                        payloads: Family::pack(
                            (0..*count)
                                .map(|i| ring.polynomial([u8::from(i == selected[step]).into()]))
                                .collect(),
                        )
                        .unwrap(),
                        id: rand::random(),
                    },
                    seed.clone(),
                    format!("selectors-{step}").into_bytes(),
                    rand::random(),
                )
                .unwrap();
            PrfSelectionInputs {
                selectors,
                labels: Family::pack(
                    labels[step]
                        .iter()
                        .map(|v| layout.mono(&ring, &BigInt::from(*v)).unwrap())
                        .collect(),
                )
                .unwrap(),
                active: FamilyRange::bounded(0, *count, *count).unwrap(),
                id: rand::random(),
            }
        })
        .collect::<Vec<_>>();
    let (public, prepared) = PreparedSparseLwrPrf::new(
        &scheme,
        program.clone(),
        ProgramValue::Scalar((key.clone(), None)),
        &blocks,
    )
    .unwrap();
    let public = ProgramValue::Scalar(public.at(0));
    let bank = scheme
        .sample_mask_bank(
            terminal,
            layout,
            seed.clone(),
            b"readout-bank".as_slice(),
            rand::random(),
        )
        .unwrap();
    // Separate all p possible digits by far more than the simulated error.
    // Target scaling is applied before fresh encryption and does not amplify noise.
    let digit_scale = parameters.level(output_level).unwrap().modulus().as_ref() / (2 * p_lwr);
    let readout_material = scheme
        .sample_readout_material(
            &r,
            &program.readout,
            &bank,
            seed,
            b"readout".as_slice(),
            rand::random(),
            Some(RnsBggReadoutWeights::Shared([vec![BigInt::from(digit_scale.clone())], vec![]])),
            None,
        )
        .unwrap();
    let readout = scheme.prepare_readout(&program.readout, &public, &readout_material).unwrap();
    let trace = trace_prf(&parameters, prepared.program(), &counts);
    let mut terminal_noise = parameters
        .terminal_noise_bound(
            terminal,
            [(&trace.wires[&program.accumulation.outputs()[0]].bound, layout.width(), &1u8.into())],
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
                terminal,
                [(&trace.wires[&program.accumulation.outputs()[0]], layout.width())],
            )
            .unwrap();
        let z = config.z_squared_for_events(&BigUint::from(2 * n)).unwrap();
        terminal_noise.error = moments.tail_bound(&z);
    }
    assert!(parameters.terminal_noise_fits(output_level, &terminal_noise).unwrap());
    assert!(
        &terminal_noise.error * 2u8 < digit_scale,
        "digit rounding must recover the exact output"
    );
    let reported_scale = digit_scale.to_string();
    let (producer, prf_descriptor) =
        prepared.export(&scheme, DslContext::new("rns_bgg-prf-preprocess"), "prf").unwrap();
    let (producer, input_descriptor) =
        scheme.export_input_material(&key, &material, producer, "initial").unwrap();
    let (producer, readout_descriptor) =
        scheme.export_prepared_readout(&readout, producer, "readout").unwrap();
    let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let online = Box::new(move |production: ProductionId, manifest: Manifest, verify: bool| {
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
        if !verify {
            // Estimate the same production output, excluding the secret-key
            // correctness oracle that the round trip evaluates separately.
            return DslContext::new("rns_bgg-prf-production-online")
                .output("value", value.values.scalar().unwrap().clone())
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production, manifest)]),
                )
                .unwrap();
        }
        let d = &program.readout.canonical_coefficients().unwrap()[exponent % layout.width()];
        let expected = if exponent < layout.width() { d.clone() } else { BigInt::from(1u8) - d };
        let expected = expected * BigInt::from(digit_scale.clone());
        let low = scheme.ring(output_level).unwrap();
        let expected =
            RnsBggRow::new([[low.polynomial([expected.into()]), low.polynomial([0.into()])]])
                .unwrap();
        let error = value.values.scalar().unwrap().clone() -
            scheme.mask_at(&s, output_level).unwrap() *
                public_result.values.scalar().unwrap().clone() -
            expected;
        let consumer = DslContext::new("rns_bgg-prf-online")
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
    let noise_report = serde_json::json!({"scheme": "rns-bgg", "model": model, "worst_bound": worst_bound.to_string(), "output_count": 1, "n": n, "selector_counts": counts, "lwr_p": p_lwr, "output_level": output_level, "digit_scale": reported_scale, "readout_error_bound": terminal_noise.error.to_string(), "readout_value_bound": terminal_noise.coefficient_bound.to_string()});
    let bounds = (0..2)
        .map(|column| {
            (
                format!("error.{column}"),
                (
                    parameters.level(output_level).unwrap().modulus().as_ref().clone(),
                    terminal_noise.error.clone(),
                ),
            )
        })
        .collect();
    RnsBggRoundTrip { parameters, producer, online, bounds, noise_report }
}

mod rns_bgg {
    use bigdecimal::BigDecimal;
    use mxx_exponent_lut::rns_bgg::{RnsBggNoiseBound, RnsBggParameters};
    use mxx_ir_core::{
        ValidatedGraph,
        artifact::{Manifest, ProductionId},
    };
    use mxx_runtime::{
        Backend, ExecutionConfig, RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        execute_with_config,
        transcript::SamplingMode,
    };
    use num_bigint::BigUint;
    use std::{collections::BTreeMap, str::FromStr};

    pub(super) fn report_encoding_noise(report: &serde_json::Value) -> (BigDecimal, BigUint) {
        let phase2 = report.get("phase2").filter(|value| value.is_object());
        let sigma_value = phase2
            .and_then(|value| value.get("error_sigma"))
            .or_else(|| report.get("error_sigma"))
            .and_then(serde_json::Value::as_str)
            .expect("accepted report must contain error_sigma");
        let bound_value = phase2
            .and_then(|value| value.get("encoding_error_bound"))
            .or_else(|| report.get("encoding_error_bound"))
            .and_then(serde_json::Value::as_str)
            .expect("accepted report must contain encoding_error_bound");
        let sigma = BigDecimal::from_str(sigma_value)
            .expect("accepted report must contain a valid error_sigma");
        let bound = BigUint::from_str(bound_value)
            .expect("accepted report must contain a valid encoding_error_bound");
        let expected = mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma);
        assert_eq!(bound, expected, "accepted report sigma/bound mismatch");
        (sigma, bound)
    }

    #[test]
    fn accepted_report_encoding_noise_round_trip_and_tamper_rejection() {
        let report = serde_json::json!({
            "phase2": {"error_sigma": "4", "encoding_error_bound": "26"}
        });
        let restored: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        let (sigma, bound) = report_encoding_noise(&restored);
        assert_eq!(sigma, BigDecimal::from(4u8));
        assert_eq!(bound, BigUint::from(26u8));
        let tampered = serde_json::json!({
            "phase2": {"error_sigma": "4", "encoding_error_bound": "25"}
        });
        assert!(std::panic::catch_unwind(|| report_encoding_noise(&tampered)).is_err());
    }

    pub struct RnsBggRoundTrip {
        pub parameters: RnsBggParameters,
        pub producer: ValidatedGraph,
        pub online: Box<dyn Fn(ProductionId, Manifest, bool) -> ValidatedGraph>,
        pub bounds: BTreeMap<String, (BigUint, BigUint)>,
        pub noise_report: serde_json::Value,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct OnlineCheckpoint {
        source: mxx_ir_core::Graph,
        bindings: mxx_ir_core::ParamEnv,
        levels: Vec<mxx_primitives::poly::dcrt::params::DCRTPolyParams>,
        encoding_error_sigma: String,
        encoding_error_bound: String,
        operations_per_drop: usize,
        production_id: ProductionId,
        bounds: BTreeMap<String, (BigUint, BigUint)>,
        noise_report: serde_json::Value,
        preprocess_seconds: f64,
        reopen_seconds: f64,
        online_graph_seconds: f64,
    }

    fn checkpoint_encoding_noise(checkpoint: &OnlineCheckpoint) -> (BigDecimal, BigUint) {
        let sigma = BigDecimal::from_str(&checkpoint.encoding_error_sigma)
            .expect("checkpoint must contain a valid encoding_error_sigma");
        let bound = BigUint::from_str(&checkpoint.encoding_error_bound)
            .expect("checkpoint must contain a valid encoding_error_bound");
        let expected = mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma);
        assert_eq!(bound, expected, "checkpoint sigma/bound mismatch");
        (sigma, bound)
    }

    #[test]
    #[ignore = "requires an existing online checkpoint in MXX_EXPONENT_LUT_BENCH_CHECKPOINT"]
    fn test_gpu_rns_bgg_saved_online_checkpoint() {
        use mxx_primitives::{
            matrix::PolyMatrix,
            poly::{
                Poly, PolyParams,
                dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            },
        };
        let path = std::path::PathBuf::from(
            std::env::var_os("MXX_EXPONENT_LUT_BENCH_CHECKPOINT")
                .expect("MXX_EXPONENT_LUT_BENCH_CHECKPOINT must name an existing checkpoint"),
        );
        let checkpoint: OnlineCheckpoint =
            serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(&path).unwrap()))
                .unwrap();
        let (sigma, bound) = checkpoint_encoding_noise(&checkpoint);
        let parameters =
            RnsBggParameters::new(checkpoint.levels, sigma, checkpoint.operations_per_drop)
                .unwrap();
        assert_eq!(parameters.encoding_error_bound(), &bound);
        let mut store = FileArtifactStore::open(path.parent().unwrap()).unwrap();
        let manifest = store.load_manifest(&checkpoint.production_id).unwrap();
        let graph = mxx_ir_core::validate_with_manifests(
            &checkpoint.source,
            &checkpoint.bindings,
            &BTreeMap::from([(checkpoint.production_id, manifest)]),
        )
        .unwrap();
        let devices = detected_gpu_device_ids();
        assert!(!devices.is_empty(), "saved online checkpoint requires a detected GPU");
        let rings = parameters.levels().iter().map(|parameters| {
            GpuDCRTPolyParams::new(
                parameters.ring_dimension(),
                parameters.to_crt().0,
                parameters.base_bits(),
                None,
            )
        });
        let mut backend = mxx_runtime::backend::poly::gpu::gpu_backend_on(rings, devices);
        let result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        for (name, value) in result.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial output") };
            let (q, bound) = &checkpoint.bounds[&name];
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
        }
        println!("prf_saved_online_checkpoint_gpu correctness=true path={}", path.display());
    }

    pub fn run<B: Backend>(
        fixture: RnsBggRoundTrip,
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        finish_matrix: impl Fn(&B::Matrix),
        check_matrix: impl Fn(&B::Matrix, &str, &BigUint, &BigUint),
    ) {
        std::fs::create_dir_all("test_data/rns_bgg_round_trip").unwrap();
        let mut backend = make_backend(&fixture.parameters);
        let warmups: usize = std::env::var("MXX_EXPONENT_LUT_BENCH_WARMUPS")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(0);
        // Bound sibling working sets independently of column distribution across GPUs.
        let parallel_instances = std::env::var("MXX_EXPONENT_LUT_MAX_PARALLEL_INSTANCES")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(64);
        let config = ExecutionConfig {
            max_parallel_instances: std::num::NonZeroUsize::new(parallel_instances).unwrap(),
            ..ExecutionConfig::default()
        };
        println!(
            "prf_runtime_parallel_instances={parallel_instances} primitive_warmups={warmups} measured_iterations=1"
        );
        // Warm the real ring/CRT contexts and transfer/primitive paths without
        // producing and saving an entire PRF's helper bank twice.
        use mxx_ir_core::{ParamEnv, types::ConcreteMatrixType};
        use mxx_primitives::poly::PolyParams;
        use num_bigint::BigInt;
        for repetition in 0..warmups {
            let started = std::time::Instant::now();
            for (level, pair) in fixture.parameters.levels().windows(2).enumerate() {
                let high_q = BigInt::from(pair[0].modulus().as_ref().clone());
                let low_q = BigInt::from(pair[1].modulus().as_ref().clone());
                let n = pair[0].ring_dimension() as usize;
                let high = mxx_dsl::Ring::new(high_q.clone(), n);
                let low = mxx_dsl::Ring::new(low_q.clone(), n);
                let seed = high.bytes_input("hash-key", 32);
                let a = high.hash_matrix(seed.clone(), b"warmup-a".as_slice(), (1, 1));
                let error =
                    high.uniform_interval((1, 1), 0, 1) - high.uniform_interval((1, 1), 0, 1);
                let source = low.hash_matrix(seed.clone(), b"warmup-digits".as_slice(), (1, 1));
                let digits = source
                    .clone()
                    .decompose(1u64 << pair[1].base_bits(), pair[1].modulus_digits())
                    .centered_extend(high_q.clone());
                let left = high.hash_matrix(
                    seed,
                    b"warmup-left".as_slice(),
                    (1, pair[1].modulus_digits()),
                );
                let value = ((a.clone() + error) * a +
                    digits.mul_small_rhs(left) +
                    source.centered_extend(high_q.clone()))
                .ring_automorphism(3)
                .block_mod_switch(low_q.clone(), 1)
                .centered_extend(high_q)
                .reduce_modulus(low_q.clone());
                let graph = mxx_dsl::DslContext::new(format!("prf-primitive-warmup-{level}"))
                    .output("value", value)
                    .unwrap()
                    .build()
                    .unwrap()
                    .validate(&ParamEnv::default())
                    .unwrap();
                let result = execute_with_config(
                    &graph,
                    &mut backend,
                    BTreeMap::from([(
                        "hash-key".into(),
                        RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
                    )]),
                    &mut mxx_runtime::artifact::MemoryArtifactStore::default(),
                    SamplingMode::Fresh,
                    config,
                )
                .unwrap();
                let RuntimeValue::Matrix(matrix) = &result.outputs["value"] else {
                    panic!("matrix")
                };
                let bytes = backend.matrix_to_bytes(matrix).unwrap();
                let restored = backend
                    .matrix_from_bytes(
                        &ConcreteMatrixType {
                            modulus: low_q,
                            ring_dimension: n,
                            rows: 1,
                            columns: 1,
                        },
                        &bytes,
                    )
                    .unwrap();
                finish_matrix(&restored);
            }
            backend.fence_released_memory().unwrap();
            println!(
                "prf_primitive_warmup_complete repetition={repetition} seconds={}",
                started.elapsed().as_secs_f64()
            );
        }
        {
            let iteration = 0;
            let measured = true;
            println!("prf_round_trip_start iteration={iteration} measured={measured}");
            let directory = tempfile::tempdir_in("test_data/rns_bgg_round_trip").unwrap();
            let durable =
                std::env::var("MXX_EXPONENT_LUT_BENCH_STAGE").as_deref() == Ok("roundtrip");
            let (directory_path, _temporary_directory) = if durable {
                (directory.keep(), None)
            } else {
                (directory.path().to_path_buf(), Some(directory))
            };
            println!("prf_artifact_directory={}", directory_path.display());
            let mut store = FileArtifactStore::new(&directory_path).unwrap();
            let round_trip_started = std::time::Instant::now();
            let started = std::time::Instant::now();
            let produced = execute_with_config(
                &fixture.producer,
                &mut backend,
                BTreeMap::from([(
                    "hash-key".into(),
                    RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
                )]),
                &mut store,
                SamplingMode::Fresh,
                config,
            )
            .unwrap();
            for value in produced.outputs.values() {
                if let RuntimeValue::Matrix(matrix) = value {
                    finish_matrix(matrix);
                }
            }
            let preprocess_seconds = started.elapsed().as_secs_f64();
            println!(
                "prf_preprocess_complete iteration={iteration} measured={measured} seconds={preprocess_seconds}"
            );
            let production = produced.production_id.unwrap();
            drop(produced.outputs);
            drop(store);
            let started = std::time::Instant::now();
            let mut store = FileArtifactStore::open(&directory_path).unwrap();
            let manifest = store.load_manifest(&production).unwrap();
            let reopen_seconds = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            let consumer = (fixture.online)(production.clone(), manifest.clone(), true);
            let online_graph_seconds = started.elapsed().as_secs_f64();
            let mut checkpoint_seconds = 0.0;
            let consumer = if durable {
                let started = std::time::Instant::now();
                let checkpoint = OnlineCheckpoint {
                    source: consumer.source,
                    bindings: consumer.bindings,
                    levels: fixture.parameters.levels().to_vec(),
                    encoding_error_sigma: fixture.parameters.encoding_error_sigma().to_string(),
                    encoding_error_bound: fixture.parameters.encoding_error_bound().to_string(),
                    operations_per_drop: fixture.parameters.operations_per_drop(),
                    production_id: production.clone(),
                    bounds: fixture.bounds.clone(),
                    noise_report: fixture.noise_report.clone(),
                    preprocess_seconds,
                    reopen_seconds,
                    online_graph_seconds,
                };
                let checkpoint_path = directory_path.join("online-checkpoint.json");
                let mut temporary = tempfile::NamedTempFile::new_in(&directory_path).unwrap();
                {
                    let mut writer = std::io::BufWriter::new(temporary.as_file_mut());
                    serde_json::to_writer(&mut writer, &checkpoint).unwrap();
                    std::io::Write::flush(&mut writer).unwrap();
                }
                temporary.as_file().sync_all().unwrap();
                temporary.persist(&checkpoint_path).unwrap();
                std::fs::File::open(&directory_path).unwrap().sync_all().unwrap();
                drop(checkpoint);
                let restored: OnlineCheckpoint = serde_json::from_reader(std::io::BufReader::new(
                    std::fs::File::open(&checkpoint_path).unwrap(),
                ))
                .unwrap();
                assert_eq!(restored.production_id, production);
                assert_eq!(restored.bounds, fixture.bounds);
                assert_eq!(restored.noise_report, fixture.noise_report);
                let (sigma, bound) = checkpoint_encoding_noise(&restored);
                let parameters = RnsBggParameters::new(
                    restored.levels.clone(),
                    sigma,
                    restored.operations_per_drop,
                )
                .unwrap();
                assert_eq!(parameters.levels(), fixture.parameters.levels());
                assert_eq!(
                    parameters.encoding_error_sigma(),
                    fixture.parameters.encoding_error_sigma()
                );
                assert_eq!(
                    parameters.encoding_error_bound(),
                    fixture.parameters.encoding_error_bound()
                );
                assert_eq!(bound, fixture.parameters.encoding_error_bound().clone());
                let mut tampered = restored.clone();
                tampered.encoding_error_bound = "25".to_owned();
                assert!(std::panic::catch_unwind(|| checkpoint_encoding_noise(&tampered)).is_err());
                let consumer = mxx_ir_core::validate_with_manifests(
                    &restored.source,
                    &restored.bindings,
                    &BTreeMap::from([(production, manifest)]),
                )
                .unwrap();
                checkpoint_seconds = started.elapsed().as_secs_f64();
                println!(
                    "prf_online_checkpoint={} checkpoint_seconds={checkpoint_seconds}",
                    checkpoint_path.display()
                );
                consumer
            } else {
                consumer
            };
            let started = std::time::Instant::now();
            let result = execute_with_config(
                &consumer,
                &mut backend,
                BTreeMap::new(),
                &mut store,
                SamplingMode::Fresh,
                config,
            )
            .unwrap();
            for value in result.outputs.values() {
                if let RuntimeValue::Matrix(matrix) = value {
                    finish_matrix(matrix);
                }
            }
            let online_eval_seconds = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            for (name, value) in result.outputs {
                let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial output") };
                let (q, bound) = &fixture.bounds[&name];
                check_matrix(&matrix, &name, q, bound);
            }
            let verification_seconds = started.elapsed().as_secs_f64();
            let round_trip_seconds = round_trip_started.elapsed().as_secs_f64();
            println!(
                "prf_round_trip_timing iteration={iteration} measured={measured} preprocess_seconds={preprocess_seconds} reopen_seconds={reopen_seconds} online_graph_seconds={online_graph_seconds} checkpoint_seconds={checkpoint_seconds} online_eval_seconds={online_eval_seconds} verification_seconds={verification_seconds} round_trip_seconds={round_trip_seconds} correctness=true"
            );
        }
    }

    pub fn trace_prf(
        parameters: &RnsBggParameters,
        execution: &mxx_exponent_lut::program::ExponentLutProgram,
        counts: &[usize],
    ) -> mxx_exponent_lut::rns_bgg::noise::RnsBggNoiseTrace {
        let inputs = BTreeMap::from([(
            *execution.inputs().keys().next().unwrap(),
            mxx_exponent_lut::rns_bgg::noise::RnsBggNoiseValue {
                level: 0,
                bound: RnsBggNoiseBound {
                    error: parameters.encoding_error_bound().clone(),
                    payload_l1: 1u8.into(),
                },
                payload_l2_squared: 1u8.into(),
                moments: Some(mxx_exponent_lut::noise_utils::NoiseMoments {
                    mean_numerator: 0.into(),
                    mean_denominator: 1u8.into(),
                    mean_radius: mxx_exponent_lut::noise_utils::NonnegativeRational::zero(),
                    variance: mxx_exponent_lut::noise_utils::NonnegativeRational::new(
                        parameters.encoding_error_variance().numerator.clone(),
                        parameters.encoding_error_variance().denominator.clone(),
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
                mxx_exponent_lut::rns_bgg::noise::RnsBggGateNoiseMaterial::from_one_hot_labels(
                    step,
                    None,
                    &vec![
                        mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit();
                        counts[step]
                    ],
                )
                .unwrap();
            rhs_bounds.insert(*rhs, bound);
            materials
                .insert(mxx_exponent_lut::program::ProgramGateInstance::Gate { step }, material);
        }
        mxx_exponent_lut::rns_bgg::noise::trace_program(
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
