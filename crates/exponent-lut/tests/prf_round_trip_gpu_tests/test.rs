//! One actual normalized LWR PRF output, without any refresh label expansion.
//! Included as a private library-test module to use the production cache APIs.
use super::*;
use crate::{
    ReadoutPreimageSchema,
    noise_refresh::pbc::{
        PbcCell, PbcParameters, PbcRootSeed, clear_pbc_inner_product, generate_key_layout,
    },
};
use mxx_primitives::{
    element::PolyElem,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    Backend, ExecutionConfig, RuntimeValue,
    artifact::{ArtifactStore, FileArtifactStore},
    backend::poly::gpu::gpu_backend_on,
    execute_in_session, execute_with_config,
    transcript::SamplingMode,
};
use rand::{Rng, seq::index::sample};
use rayon::prelude::*;
use std::time::Instant;

// Test-only local plan path. The GPU test consumes it without simulation.
const PLAN_ENV: &str = "MXX_EXPONENT_LUT_PRF_TEST_PLAN";

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
#[ignore = "requires CUDA GPUs; validates native input and explicit output conversion"]
fn test_gpu_prf_monomial_input_output_boundary() {
    let config = PrfTestParameters::from_env();
    let (params, _, _) = config.build();
    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    assert!(!devices.is_empty());
    let gpu_params = mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
        params.ring_dimension(),
        params.to_crt().0,
        config.base_bits,
        None,
    );
    let mut backend = gpu_backend_on([gpu_params], devices);
    let n = params.ring_dimension() as usize;
    let modulus = BigInt::from(params.modulus().as_ref().clone());
    for exponent in 0..2 * n {
        let value = monomial_input(&mut backend, &modulus, n, exponent);
        let actual = backend.gather_matrix_for_host(&value).unwrap().to_cpu_matrix();
        let expected = DCRTPoly::const_rotate_poly(&params, exponent % n);
        let expected = if exponent >= n { -expected } else { expected };
        assert_eq!(actual, DCRTPolyMatrix::from_poly_vec_row(&params, vec![expected]));
    }
    println!("prf_monomial_boundary_pass signed_exponents={}", 2 * n);
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
        let report: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::env::var("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT")
                    .expect("accepted security report required"),
            )
            .expect("read accepted report"),
        )
        .expect("parse accepted report");
        let accepted = &report["result"];
        assert!(accepted["bgg_rlwe_security_bits"].as_u64().unwrap() >= security as u64);
        assert!(accepted["sparse_lwr_security_bits"].as_u64().unwrap() >= security as u64);
        assert_eq!(accepted["checker_accepted"], true);
        for (field, value) in [
            ("log_ring_dimension", log_n),
            ("crt_depth", depth),
            ("crt_bits", crt_bits),
            ("base_bits", base_bits as usize),
        ] {
            assert_eq!(
                accepted["candidate"][field].as_u64().unwrap(),
                value as u64,
                "candidate security binding: {field}"
            );
        }
        for (field, value) in [
            ("sparse_lwr_universe", universe),
            ("sparse_lwr_weight", weight),
            ("sparse_lwr_q_l", q_native),
            ("sparse_lwr_p", p),
        ] {
            assert_eq!(
                accepted[field].as_u64().unwrap(),
                value as u64,
                "PRF security binding: {field}"
            );
        }
        println!(
            "prf_one_output_security target_bits={security} bgg_rlwe_bits={} sparse_lwr_bits={} encoding_sigma=4",
            accepted["bgg_rlwe_security_bits"], accepted["sparse_lwr_security_bits"]
        );
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

    fn build(&self) -> (DCRTPolyParams, RefreshSetupParameters, SparseLwrPrfProfile) {
        let Self { log_n, depth, crt_bits, base_bits, q_native, p, security, .. } = *self;
        let params = DCRTPolyParams::new(1 << log_n, depth, crt_bits, base_bits, None, None);
        let n = params.ring_dimension() as usize;
        let modulus = BigInt::from(params.modulus().as_ref().clone());
        let width_log2 = q_native.trailing_zeros() as usize - 1;
        assert!(q_native.is_power_of_two() && p >= 4 && p.is_power_of_two());
        let profile = SparseLwrPrfProfile::new(
            crate::ExpLayout {
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
        let setup = RefreshSetupParameters::new(
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

#[derive(serde::Serialize, serde::Deserialize)]
struct PrfTestPlan {
    parameters: PrfTestParameters,
    root_seed: [u8; 32],
    support: Vec<usize>,
    layout_id: [u8; 32],
    active_count: usize,
    coefficient_events: usize,
    failure_exponent: usize,
    variance: crate::AverageVariance,
    z_squared_numerator: String,
    z_squared_denominator: u64,
    average_bound: String,
    worst_bound: String,
}

#[test]
#[ignore = "local CPU-only one-output parameter simulation"]
fn test_prepare_lwr_prf_one_output_plan() {
    let parameters = PrfTestParameters::from_env();
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
    let snapshot = RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout(
        setup.clone(),
        profile,
        public_layout,
        &active,
    )
    .unwrap();
    let noise = snapshot.evaluate_pre_refresh().unwrap();
    let worst_bound = noise.readout_bound;
    // One-output adapter of simulation.rs AverageCaseConfig::z_squared and
    // average_favg. V uses doubled coordinates, hence the exact division by 4.
    // No refresh mask/fresh-label topology is fabricated for this PRF test.
    let coefficient_events = n * layout.public_key_columns();
    let event_log2 = usize::BITS - (coefficient_events - 1).leading_zeros();
    let tail_events = security as u64 + u64::from(event_log2) + 1;
    let z_squared_numerator =
        BigUint::from(2u8) * BigUint::from(693148u32) * BigUint::from(tail_events);
    let numerator = &z_squared_numerator * &noise.readout_variance.numerator;
    let denominator = BigUint::from(4_000_000u32) * &noise.readout_variance.denominator;
    let squared_bound = (&numerator + &denominator - BigUint::one()) / &denominator;
    let floor = squared_bound.sqrt();
    let bound = if &floor * &floor == squared_bound { floor } else { floor + BigUint::one() };
    assert!(
        !bound.is_zero() && bound < params.modulus().as_ref() / BigUint::from(2u8),
        "PRF noise authority must be nonvacuous below q/2"
    );

    let plan = PrfTestPlan {
        parameters,
        root_seed,
        support,
        layout_id: public_layout.layout_id.0,
        active_count: active.len(),
        coefficient_events,
        failure_exponent: security,
        variance: noise.readout_variance,
        z_squared_numerator: z_squared_numerator.to_string(),
        z_squared_denominator: 1_000_000,
        average_bound: bound.to_string(),
        worst_bound: worst_bound.to_string(),
    };
    let path = std::env::var(PLAN_ENV).expect("local output plan path required");
    std::fs::write(&path, serde_json::to_vec_pretty(&plan).unwrap()).expect("save local test plan");
    println!(
        "prf_local_plan_saved path={path} model=AverageCase failure_exponent={security} coefficient_events={coefficient_events} average_bound={bound} worst_bound={worst_bound} gpu_execution=false"
    );
}

#[test]
#[ignore = "local CPU-only replay of an archived one-output plan"]
fn test_recompute_lwr_prf_readout_helper_gain() {
    let path = std::path::PathBuf::from(std::env::var(PLAN_ENV).expect("archived input plan"));
    let archived_bytes = std::fs::read(&path).unwrap();
    let old: PrfTestPlan = serde_json::from_slice(&archived_bytes).unwrap();
    assert_eq!(old.parameters, PrfTestParameters::from_env());
    let (params, setup, profile) = old.parameters.build();
    let generated = generate_key_layout(
        &PbcParameters::paper_evaluation(old.parameters.universe, old.parameters.weight),
        PbcRootSeed(old.root_seed),
        &old.support,
    )
    .unwrap();
    assert_eq!(generated.public_layout.layout_id.0, old.layout_id);
    let active = PbcActiveCellIndex::build(&generated.public_layout).unwrap();
    assert_eq!(active.len(), old.active_count);
    let snapshot = RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout(
        setup.clone(),
        profile,
        &generated.public_layout,
        &active,
    )
    .unwrap();
    let noise = snapshot.evaluate_pre_refresh().unwrap();
    let gain = profile
        .canonical_readout()
        .unwrap()
        .target_plaintext_gain(params.ring_dimension() as usize)
        .unwrap();
    let branches = BigUint::from(snapshot.readout_gains.nonidentity_trace.len());
    let worst_delta = &snapshot.readout_gains.normalization.l1 *
        &branches *
        (&gain.l1 - BigUint::one()) *
        &snapshot.helper_error_cutoff;
    let variance_delta = snapshot.helper_doubled_variance.scaled(
        &(&snapshot.readout_gains.normalization.l2_squared *
            &branches *
            (&gain.l2_squared - BigUint::one())),
    );
    assert!(!worst_delta.is_zero() && !variance_delta.numerator.is_zero());
    assert_eq!(noise.readout_bound, old.worst_bound.parse::<BigUint>().unwrap() + &worst_delta);
    assert_eq!(noise.readout_variance, old.variance.add(&variance_delta));
    assert_eq!(
        old.coefficient_events,
        params.ring_dimension() as usize * setup.layout.public_key_columns()
    );
    assert_eq!(old.failure_exponent, old.parameters.security);
    let event_log2 = usize::BITS - (old.coefficient_events - 1).leading_zeros();
    let z_squared_numerator = BigUint::from(2u8) *
        BigUint::from(693148u32) *
        BigUint::from(old.failure_exponent as u64 + u64::from(event_log2) + 1);
    assert_eq!(old.z_squared_numerator, z_squared_numerator.to_string());
    assert_eq!(old.z_squared_denominator, 1_000_000);
    let average_bound = |variance: &crate::AverageVariance| {
        let numerator = &z_squared_numerator * &variance.numerator;
        let denominator = BigUint::from(4_000_000u32) * &variance.denominator;
        let square = (&numerator + &denominator - BigUint::one()) / &denominator;
        let floor = square.sqrt();
        if &floor * &floor == square { floor } else { floor + BigUint::one() }
    };
    assert_eq!(average_bound(&old.variance).to_string(), old.average_bound);
    let bound = average_bound(&noise.readout_variance);
    assert!(bound < params.modulus().as_ref() / BigUint::from(2u8));
    let variance_json = |value: &crate::AverageVariance| {
        serde_json::json!({
            "numerator": value.numerator.to_string(), "denominator": value.denominator.to_string(),
        })
    };
    let comparison = serde_json::json!({
        "parameters": old.parameters, "root_seed": old.root_seed, "support": old.support,
        "layout_id": old.layout_id, "active_count": old.active_count,
        "target_plaintext_gain": {"l1": gain.l1.to_string(), "l2_squared": gain.l2_squared.to_string()},
        "helper_count": branches.to_string(),
        "coefficient_events": old.coefficient_events, "failure_exponent": old.failure_exponent,
        "old_variance": variance_json(&old.variance), "variance_delta": variance_json(&variance_delta),
        "new_variance": variance_json(&noise.readout_variance),
        "old_average_bound": old.average_bound, "new_average_bound": bound.to_string(),
        "old_worst_bound": old.worst_bound, "worst_bound_delta": worst_delta.to_string(),
        "new_worst_bound": noise.readout_bound.to_string(),
        "gpu_execution": false, "parameter_search": false,
    });
    let output = path.with_file_name("prf-one-output-readout-gain-comparison.json");
    assert_ne!(output, path);
    std::fs::write(&output, serde_json::to_vec_pretty(&comparison).unwrap()).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), archived_bytes, "historical plan unchanged");
    println!("prf_local_readout_gain_comparison path={} values={comparison}", output.display());
}

#[test]
#[ignore = "actual CUDA PRF round-trip using a locally simulated plan"]
#[serial_test::serial]
fn test_gpu_lwr_prf_one_output_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let parameters = PrfTestParameters::from_env();
    let plan: PrfTestPlan = serde_json::from_slice(
        &std::fs::read(std::env::var(PLAN_ENV).expect("locally simulated plan required"))
            .expect("read local test plan"),
    )
    .expect("parse local test plan");
    assert_eq!(plan.parameters, parameters, "local/remote test parameters must match");
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
    let bound: BigUint = plan.average_bound.parse().unwrap();
    let worst_bound: BigUint = plan.worst_bound.parse().unwrap();
    assert!(!bound.is_zero() && bound < params.modulus().as_ref() / BigUint::from(2u8));
    println!(
        "prf_one_output_plan N={n} depth={depth} crt_bits={crt_bits} base_bits={base_bits} universe={universe} weight={weight} Q={q_native} P={p} active={} output_count=1 model=AverageCase failure_exponent={security} coefficient_events={} average_bound={bound} worst_bound={worst_bound} parameter_simulation=false plan_origin=local",
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
    let selectors = build_structural_selector_families(
        &sampler,
        bits.family().clone(),
        t.clone(),
        t.clone(),
        hash.clone(),
        public_layout,
        key_id,
    )
    .unwrap();
    let (public_inputs, label_rhs) = sample_setup_public_key_families(
        &mxx_bgg::BggPublicKeySampler { layout: layout.clone() },
        hash.clone(),
        active.len(),
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
    let public_compiler =
        ExponentLutPublicKeyCompiler::with_layout(bgg_public.clone(), profile.layout);
    let compiler = ExponentLutEncodingCompiler::with_layout(bgg_public, profile.layout);
    let public_batch = program
        .compile_public_batch_with_preprocessing(
            &public_compiler,
            initial.clone(),
            &PublicRhsFamily::new_for_layout(selectors.gsw.clone(), public_layout.layout_id)
                .unwrap(),
            &NativePrfPublicInputs::new_for_layout(public_inputs, 1, public_layout.layout_id)
                .unwrap(),
            public_layout,
        )
        .unwrap();
    let table_values = program
        .readout
        .canonical_coefficients()
        .unwrap()
        .iter()
        .map(|v| v.to_usize().unwrap())
        .collect();
    let table =
        LutTable::unary(profile.native_width(), profile.native_width(), table_values).unwrap();
    let helpers = sampler
        .sample_native_readout_helper_families(
            t.clone(),
            t.clone(),
            hash.clone(),
            &table,
            b"prf-mask".as_slice(),
            b"prf-helper".as_slice(),
        )
        .unwrap();
    let gadget = ring.gadget(2, layout.gadget_base.clone(), layout.digit_count);
    let readout = program
        .produce_public_readout_family_artifacts(
            &public_compiler,
            public_batch.outputs.clone(),
            &helpers,
            gadget.clone(),
        )
        .unwrap();
    let public_output = readout
        .project_public_batch(
            &program.readout,
            &public_compiler.public_key,
            public_batch.outputs,
            helpers.public_masks.clone(),
        )
        .unwrap();
    let names = canonical_program_cache_names(&program, public_layout, rng.random()).unwrap();
    let mut context = DslContext::new("one-output-prf-preprocessing")
        .private_output("t", t)
        .unwrap()
        .public_output("initial", initial.matrix)
        .unwrap()
        .public_output("selectors", selectors.gsw)
        .unwrap()
        .public_output("label-rhs", label_rhs)
        .unwrap()
        .public_output("output-public", public_output)
        .unwrap()
        .public_output("public-masks", helpers.public_masks.clone())
        .unwrap()
        .public_output("readout-target", readout.target_decompositions().clone())
        .unwrap()
        .public_output("readout-switch", readout.helper_switch_decompositions().clone())
        .unwrap()
        .public_output("readout-public", readout.helper_public_decompositions().clone())
        .unwrap()
        .public_output("readout-normalization", readout.normalization_decomposition().clone())
        .unwrap();
    for entry in &names.families {
        context = context
            .public_output(
                entry.rhs.clone(),
                public_batch.cache.family_entry(entry.step, entry.body_gate).unwrap().2.clone(),
            )
            .unwrap();
    }
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
    let import_preimage_family = |name: &str| {
        let artifact = &manifest.artifacts[name];
        let ArtifactType::Preimage { matrix, max_coefficient_bound } = &artifact.artifact_type
        else {
            panic!("preimage schema")
        };
        ring.preimage_family_artifact_input(
            production.clone(),
            name,
            artifact.family_count.unwrap(),
            (matrix.rows, matrix.columns),
            max_coefficient_bound.clone(),
            ArtifactConfidentiality::Public,
        )
    };
    let import_matrix_family = |name: &str| {
        let artifact = &manifest.artifacts[name];
        let ArtifactType::Matrix(matrix) = &artifact.artifact_type else { panic!("matrix schema") };
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
    let plaintexts = ring.input_family("prf-monomials", active.len(), (1, 1));
    let vectors = (import_preimage_family("label-rhs"), plaintexts.clone())
        .parallel_zip_tuple_result(|_, (rhs, plaintext)| {
            sample_online_label_vector_from_preimage(&sampler, &s, rhs, plaintext)
                .map_err(|_| mxx_dsl::DslError::Schema)
        })
        .unwrap();
    let inputs =
        NativePrfOnlineBatchInputs::new_for_layout(vectors, plaintexts, 1, public_layout.layout_id)
            .unwrap();
    let imported_selectors = EncodingRhsFamily::new_for_layout(
        import_matrix_family("selectors"),
        public_layout.layout_id,
    )
    .unwrap();
    let mut cache = ProgramCacheBatchBindings::new(1).unwrap();
    for entry in &names.families {
        let family = import_preimage_family(&entry.rhs);
        if entry.output_indexed {
            cache.insert_family(entry.step, entry.body_gate, entry.count, family).unwrap();
        } else {
            cache.insert_shared_family(entry.step, entry.body_gate, entry.count, family).unwrap();
        }
    }
    let accumulation = program
        .compile_encoding_batch_with_preprocessed_artifacts(
            &compiler,
            initial_vector,
            &imported_selectors,
            &inputs,
            public_layout,
            &cache,
        )
        .unwrap();
    let normalization = &manifest.artifacts["readout-normalization"].artifact_type;
    let ArtifactType::Preimage { matrix, max_coefficient_bound } = normalization else {
        panic!("normalization preimage")
    };
    let schema = ReadoutPreimageSchema::new(
        layout.gadget_base.evaluate(&ParamEnv::default()).unwrap(),
        layout.digit_count,
        max_coefficient_bound.clone(),
    )
    .unwrap();
    let imported_readout = NativePrfReadoutFamilyArtifacts::new(
        import_preimage_family("readout-target"),
        schema.clone(),
        helpers.sigmas.as_ref().clone(),
        import_preimage_family("readout-switch"),
        import_preimage_family("readout-public"),
        ring.preimage_artifact_input(
            production.clone(),
            "readout-normalization",
            (matrix.rows, matrix.columns),
            max_coefficient_bound.clone(),
            ArtifactConfidentiality::Public,
        ),
        schema,
        1,
    )
    .unwrap();
    let mask_publics = import_matrix_family("public-masks");
    let masks = mask_publics
        .parallel_map(|index, matrix| {
            let sigma = profile
                .layout
                .automorphism_index_expression(
                    n,
                    IntExpr::Add(Box::new(index.expression()), Box::new(IntExpr::constant(1))),
                )
                .unwrap();
            let mask_sigma = s.clone().ring_automorphism(sigma);
            mxx_bgg::BggEncodingSampler {
                layout: layout.clone(),
                gaussian_sigma: sampler.gaussian_sigma.clone(),
                gaussian_max_coefficient_bound: sampler.gaussian_max_coefficient_bound.clone(),
            }
            .sample(
                s.clone(),
                Some(mask_sigma),
                &[
                    BggPublicKeyWire { matrix: ring.zero((2, columns)), reveal_plaintext: false },
                    BggPublicKeyWire { matrix, reveal_plaintext: false },
                ],
                &[ring.identity(1)],
            )
            .unwrap()
            .remove(1)
            .vector
        })
        .unwrap();
    let output = program
        .execute_value_readout_batch_with_mask_family(
            &compiler.bgg,
            &accumulation,
            &imported_readout,
            masks,
        )
        .unwrap()
        .vectors()
        .get_static(0);
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
    let inverse_two = (&modulus + BigInt::one()) / BigInt::from(2u8);
    let g = ring.polynomial([(&doubled_g * inverse_two).into()]);
    let anchor =
        s.clone() * import_matrix_family("output-public").get_static(0) - g * (imported_t * gadget);
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
    validate_cache_required_online_graph(&online).unwrap();
    online
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(production.clone(), manifest.clone())]),
        )
        .expect("validate online graph before expensive preprocessing");
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
    let mut store = FileArtifactStore::new(&directory).unwrap();
    let start = Instant::now();
    execute_in_session(&preprocessing, &mut backend, runtime_inputs, &mut store, nonce)
        .expect("real PRF preprocessing");
    println!("prf_one_output_preprocess_seconds={}", start.elapsed().as_secs_f64());
    drop(store);
    let mut store = FileArtifactStore::open(&directory).unwrap();
    let actual_manifest = store.load_manifest(&production).unwrap();
    let online = online
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(production, actual_manifest)]),
        )
        .unwrap();
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
    for run in 0..3 {
        let start = Instant::now();
        let result = execute_with_config(
            &online,
            &mut backend,
            BTreeMap::from([(
                "prf-monomials".into(),
                RuntimeValue::IndexedFamily(monomials.clone()),
            )]),
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
            "prf_one_output_noise_observed run={run} max_error={maximum} strict_bound={bound} model=AverageCase"
        );
        assert!(!maximum.is_zero(), "actual PRF error must be nonzero");
        assert!(maximum < bound, "actual PRF noise exceeds existing readout authority");
        println!(
            "prf_one_output_pass run={run} online_seconds={} z={z} conventional_lwr={conventional_lwr} centered_prf_digit={digit} doubled_g={doubled_g} plaintext_representation=d-1/2 relation=c-sA+g(tG) coefficients={} max_error={maximum} strict_bound={bound} worst_bound={worst_bound} model=AverageCase failure_exponent={security}",
            start.elapsed().as_secs_f64(),
            n * columns
        );
    }
}
