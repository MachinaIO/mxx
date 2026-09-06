//! Compile-time coverage for the native sparse-LWR refresh setup boundary.
//!
//! The former fixture instantiated removed grouped reduction and terminal LUT
//! helpers. Native evaluator/preimage attachment is exercised by the runtime
//! owner; this module keeps the GPU-facing construction names and native
//! layout contract checked. Setup retains public-key matrices in
//! `BggPublicKeyWire`, while online `BggEncodingWire` carries only its vector
//! and optional plaintext. Multiplication consumes the generic preimage cache;
//! it does not require a public-key field on the encoding wire.

#![cfg(feature = "gpu")]

#[allow(dead_code)]
mod test_utils;

use mxx_exponent_lut::{SparseLwrPrfProfile, layout::ExpLayout};
use mxx_ir_core::{ParamEnv, types::ConcreteMatrixType};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
};
use mxx_runtime::{
    ExecutionConfig, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly::gpu::gpu_backend_on,
    execute_in_session, execute_with_config,
    transcript::{SamplingMode, TranscriptRecorder},
};
use std::{collections::BTreeMap, num::NonZeroUsize};
use test_utils::tiny_native_refresh_bundle;

/// Real noisy execution, deliberately separate from the benchmark estimator.
#[test]
#[ignore = "requires an accepted parameter report and a CUDA GPU"]
#[serial_test::serial]
fn test_gpu_noise_refresh_round_trip() {
    use mxx_exponent_lut::noise_refresh::{
        pbc::{PbcActiveCellIndex, PbcCell, clear_pbc_inner_product},
        refresh::{RefreshPrfLabel, RefreshPrfLabelIndex},
    };
    use mxx_primitives::{element::PolyElem, poly::dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        artifact::{ArtifactStore, FileArtifactStore},
        backend::Backend,
    };
    use num_bigint::{BigInt, BigUint};
    use num_traits::{One, Zero};
    use rand::Rng;
    use rayon::prelude::*;

    let config = test_utils::SearchConfig::from_env().expect("round-trip parameters");
    assert!(config.error_sigma > 0.0 && config.decoder_sigma > 0.0, "real Gaussian noise required");
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            std::env::var("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT")
                .expect("accepted report required"),
        )
        .expect("read accepted report"),
    )
    .expect("parse accepted report");
    let selected = &report["result"];
    assert!(selected["bgg_rlwe_security_bits"].as_u64().unwrap() >= config.security_bits);
    assert!(selected["sparse_lwr_security_bits"].as_u64().unwrap() >= config.security_bits);
    assert_eq!(report["phase1"]["security_target_bits"].as_u64().unwrap(), config.security_bits);
    assert_eq!(report["phase2"]["security_target_bits"].as_u64().unwrap(), config.security_bits);
    assert_eq!(selected["estimator_commit"].as_str().unwrap(), config.phase1_estimator_commit);
    assert_eq!(
        selected["estimator_cost_model"].as_str().unwrap(),
        config.phase1_estimator_cost_model
    );
    assert_eq!(
        selected["estimator_shape_model"].as_str().unwrap(),
        config.phase1_estimator_shape_model
    );
    assert_eq!(selected["checker_accepted"], true);
    assert!(selected["achieved_security_bits"].as_u64().unwrap() >= config.security_bits);
    for (name, expected) in [
        ("sparse_lwr_universe", config.sparse_lwr_universe),
        ("sparse_lwr_weight", config.sparse_lwr_weight),
        ("sparse_lwr_q_l", config.sparse_lwr_modulus),
        ("sparse_lwr_p", config.sparse_lwr_output_modulus),
    ] {
        assert_eq!(selected[name].as_u64().unwrap(), expected as u64, "{name}");
    }
    let candidate: test_utils::Candidate =
        serde_json::from_value(selected["candidate"].clone()).unwrap();
    let evidence = &selected["average_evidence"];
    assert_eq!(evidence["accepted"], true);
    let mask_digits = evidence["mask_digit_count"].as_u64().unwrap() as usize;
    let fresh_digits = evidence["fresh_error_digit_count"].as_u64().unwrap() as usize;
    let params = DCRTPolyParams::new(
        1 << candidate.log_ring_dimension,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
        None,
        None,
    );
    let n = params.ring_dimension() as usize;
    println!(
        "round_trip_security bgg_rlwe_bits={} sparse_lwr_bits={} N={n} q_bits={} sigma={}",
        selected["bgg_rlwe_security_bits"],
        selected["sparse_lwr_security_bits"],
        params.modulus().bits(),
        config.error_sigma
    );
    let mut rng = rand::rng();
    let plaintext_exponent = rng.random_range(1..n);
    let (bundle, generated, profile) = test_utils::round_trip_bundle(
        &config,
        candidate,
        mask_digits,
        fresh_digits,
        plaintext_exponent,
    )
    .expect("build real refresh verification bundle");
    println!(
        "round_trip_config security_bits={} mask_security_bits={} correctness_failure_exponent={} encoding_sigma={} decoder_sigma={} candidate={candidate:?}",
        config.security_bits,
        config.mask_statistical_security_bits,
        config.security_bits,
        config.error_sigma,
        config.decoder_sigma
    );
    let noise = bundle
        .simulate_average_noise(&test_utils::average_case_config(&config).unwrap())
        .expect("randomized layout noise authority");
    assert!(noise.accepted, "randomized runtime layout must independently pass authority");
    let columns = config.secret_dimension * params.modulus_digits();
    let labels = RefreshPrfLabelIndex::new(
        [0; 32],
        candidate.crt_depth,
        columns,
        n,
        vec![mask_digits; candidate.crt_depth],
        fresh_digits,
    )
    .unwrap();
    let active = PbcActiveCellIndex::build(&generated.public_layout).unwrap();
    let table = profile.canonical_readout().unwrap().canonical_coefficients().unwrap().to_vec();
    let q = profile.q().unwrap();
    let scalar_type = ConcreteMatrixType::scalar(params.modulus().as_ref().clone().into(), n);
    // Share immutable staging bytes for equal public monomials. Every label
    // still has its independently sampled exponent; only Q distinct matrices
    // exist, so their storage need not grow with the label count.
    let public_monomials = (0..q)
        .into_par_iter()
        .map(|z| {
            let exponent = z << profile.layout.stride_log2;
            let poly = DCRTPoly::const_rotate_poly(&params, exponent % n);
            let poly = if exponent >= n { -poly } else { poly };
            RuntimeValue::HostMatrix {
                matrix_type: scalar_type.clone(),
                bytes: DCRTPolyMatrix::from_poly_vec_row(&params, vec![poly])
                    .to_cpu_staging_bytes()
                    .into(),
            }
        })
        .collect::<Vec<_>>();
    let mut expected_coefficients = vec![vec![BigInt::zero(); n]; columns];
    let mut monomials = Vec::with_capacity(labels.len() * active.len());
    for index in 0..labels.len() {
        let public = (0..config.sparse_lwr_universe)
            .map(|_| rng.random_range(0..q as u64))
            .collect::<Vec<_>>();
        let z = clear_pbc_inner_product(
            &generated.public_layout,
            generated.private_schedule(),
            &public,
            q as u64,
        )
        .unwrap() as usize;
        if let RefreshPrfLabel::FreshError { component, coefficient, digit, .. } =
            labels.label(index).unwrap()
        {
            let f = &table[z % (q / 2)];
            let d = if z < q / 2 { f.clone() } else { BigInt::one() - f };
            // BGG payload has negative sign: routing -g and adding 1/2 gives 1-d.
            expected_coefficients[component][coefficient] += (BigInt::one() - d) *
                BigInt::from(config.sparse_lwr_output_modulus).pow(digit as u32);
        }
        monomials.extend(active.iter().map(|(bucket, slot, _)| {
            let z = match generated.public_layout.cells[bucket][slot] {
                PbcCell::Real { coordinate, .. } => public[coordinate] as usize,
                PbcCell::Dummy => 0,
                PbcCell::Padding => unreachable!("active index excludes padding"),
            };
            public_monomials[z].clone()
        }));
    }
    let expected = DCRTPolyMatrix::from_poly_vec_row(
        &params,
        expected_coefficients
            .par_iter()
            .map(|values| {
                DCRTPoly::from_coeffs(
                    &params,
                    &values
                        .iter()
                        .map(|v| {
                            mxx_primitives::element::finite_ring::FinRingElem::new(
                                v.clone(),
                                params.modulus(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect(),
    );
    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    assert!(!devices.is_empty(), "real GPU required");
    let (moduli, _, _) = params.to_crt();
    let mut backend = gpu_backend_on(
        [mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
            params.ring_dimension(),
            moduli,
            params.base_bits(),
            None,
        )],
        devices,
    );
    let preprocessing = bundle
        .preprocessing_graph()
        .validate_with_manifests(&ParamEnv::default(), bundle.preprocessing_manifests())
        .unwrap();
    let mut inputs = bundle.selector_runtime_inputs(&mut backend).unwrap();
    inputs.insert(
        "refresh-parameter-search-hash-key".into(),
        RuntimeValue::Bytes(rng.random::<[u8; 32]>().to_vec()),
    );
    std::fs::create_dir_all("test_data/noise_refresh_round_trip").unwrap();
    let directory = tempfile::tempdir_in("test_data/noise_refresh_round_trip").unwrap();
    let mut store = FileArtifactStore::new(directory.path()).unwrap();
    let production =
        execute_in_session(&preprocessing, &mut backend, inputs, &mut store, rng.random())
            .expect("real preprocessing")
            .production_id
            .unwrap();
    drop(store);
    let mut store = FileArtifactStore::open(directory.path()).unwrap();
    let manifest = store.load_manifest(&production).unwrap();
    let online = bundle
        .verification_graph()
        .validate_with_manifests(&ParamEnv::default(), &BTreeMap::from([(production, manifest)]))
        .unwrap();
    let mut previous_vector = None;
    for online_run in 0..3 {
        let result = execute_with_config(
            &online,
            &mut backend,
            BTreeMap::from([(
                "refresh-parameter-search-native-monomials".into(),
                RuntimeValue::IndexedFamily(monomials.clone()),
            )]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .expect("real online refresh");
        let matrix = |name: &str| {
            let RuntimeValue::Matrix(value) = &result.outputs[name] else {
                panic!("matrix output required")
            };
            DCRTPolyMatrix::from_compact_bytes(&params, &backend.matrix_to_bytes(value))
        };
        let residual = matrix("residual");
        assert_eq!(
            residual, expected,
            "every fresh-error coefficient must match the independent clear PRF oracle"
        );
        let vector = matrix("refresh-verification-vector");
        assert_eq!(
            vector,
            matrix("refresh-verification-anchor") + expected.clone(),
            "refreshed vector must equal plaintext anchor plus actual fresh error"
        );
        if let Some(previous) = previous_vector.replace(vector.clone()) {
            assert_ne!(previous, vector, "fresh online sampling must change the encoding mask");
        }
        let maximum = residual
            .get_row(0)
            .par_iter()
            .flat_map_iter(|p| p.coeffs())
            .map(|v| v.value().min(&(params.modulus().as_ref() - v.value())).clone())
            .max()
            .unwrap();
        let bound = BigUint::from(config.sparse_lwr_output_modulus).pow(fresh_digits as u32) -
            BigUint::one();
        assert!(!maximum.is_zero(), "refresh must contain nonzero fresh noise");
        assert!(
            maximum < bound,
            "actual coefficient norm {maximum} must be strictly below B_e={bound}"
        );
        let decoded = result
            .outputs
            .iter()
            .filter(|(name, _)| name.starts_with("decoded_"))
            .collect::<Vec<_>>();
        assert_eq!(decoded.len(), columns);
        assert!(decoded.iter().all(|(_, v)| matches!(v, RuntimeValue::Bool(false))));
        println!(
            "round_trip_passed online_run={online_run} N={n} depth={} plaintext_exponent={plaintext_exponent} max_noise={maximum} strict_fresh_bound={bound} mask_digits={mask_digits} fresh_digits={fresh_digits}",
            candidate.crt_depth
        );
    }
}

#[test]
#[serial_test::serial]
fn gpu_setup_uses_derived_native_q_p_and_readout() {
    let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8)
        .expect("native PRF profile");
    assert_eq!(profile.q().expect("Q"), 64);
    assert_eq!(profile.p().expect("P"), 16);
    let program = mxx_exponent_lut::noise_refresh::prf::SparseLwrPrfProgram::new(profile, 1, 1)
        .expect("native PRF program");
    assert!(program.has_only_native_accumulation());
}

#[test]
#[serial_test::serial]
fn gpu_artifact_backed_refresh_replays_across_wave_settings() {
    let (poly_parameters, bundle, total_labels) = tiny_native_refresh_bundle();
    let noise = bundle.simulate_noise().expect("tiny fixture noise authority");
    assert!(noise.refresh.fresh_error_accepted, "strict fresh-error bound must hold");
    assert!(noise.refresh.mask_coverage_accepted, "tiny mask digits must cover native Q");
    // This deliberately tiny arithmetic fixture is not a security parameter
    // profile. Selected-profile authority acceptance is exercised by the
    // parameter-search integration test; this runtime test covers the same
    // artifact-backed execution path and the strict fresh-error invariant.
    let preprocessing = bundle
        .preprocessing_graph()
        .validate_with_manifests(&ParamEnv::default(), bundle.preprocessing_manifests())
        .expect("validate preprocessing graph");
    let (moduli, _, _) = poly_parameters.to_crt();
    let gpu_parameters = mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
        poly_parameters.ring_dimension(),
        moduli,
        poly_parameters.base_bits(),
        None,
    );
    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU required for artifact-backed smoke test");
    let mut backend =
        gpu_backend_on(bundle.ring_parameters(gpu_parameters).expect("refresh rings"), devices);
    let mut inputs = bundle.selector_runtime_inputs(&mut backend).expect("selector inputs");
    inputs.insert(
        "refresh-parameter-search-hash-key".to_owned(),
        RuntimeValue::Bytes(vec![0x94; 32]),
    );
    let mut store = MemoryArtifactStore::default();
    let result = execute_in_session(&preprocessing, &mut backend, inputs, &mut store, [0x53; 32])
        .expect("GPU preprocessing execution");
    let production = result.production_id.expect("preprocessing production");
    let manifest = store.manifest(&production).expect("preprocessing artifact manifest").clone();
    let online = bundle
        .verification_graph()
        .validate_with_manifests(&ParamEnv::default(), &BTreeMap::from([(production, manifest)]))
        .expect("artifact-backed online graph");
    let monomials = (0..total_labels)
        .map(|index| {
            let exponent = 2 * (index % 32);
            let monomial = DCRTPoly::const_rotate_poly(
                &poly_parameters,
                if exponent >= 32 { exponent - 32 } else { exponent },
            );
            let monomial = if exponent >= 32 { -monomial } else { monomial };
            let matrix = DCRTPolyMatrix::from_poly_vec_row(&poly_parameters, vec![monomial]);
            RuntimeValue::HostMatrix {
                matrix_type: ConcreteMatrixType::scalar(
                    poly_parameters.modulus().as_ref().clone().into(),
                    poly_parameters.ring_dimension() as usize,
                ),
                bytes: matrix.to_cpu_staging_bytes().into(),
            }
        })
        .collect();
    let online_inputs = BTreeMap::from([(
        "refresh-parameter-search-native-monomials".to_owned(),
        RuntimeValue::IndexedFamily(monomials),
    )]);
    let mut transcript = TranscriptRecorder::default();
    let first = execute_with_config(
        &online,
        &mut backend,
        online_inputs.clone(),
        &mut store,
        SamplingMode::Record(&mut transcript),
        ExecutionConfig {
            max_parallel_instances: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
    )
    .expect("GPU online execution at narrow wave");
    let replayer = transcript.into_replayer();
    let second = execute_with_config(
        &online,
        &mut backend,
        online_inputs,
        &mut store,
        SamplingMode::Replay(&replayer),
        ExecutionConfig {
            max_parallel_instances: NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        },
    )
    .expect("GPU online replay at wider wave");
    let decoded = |result: &mxx_runtime::ExecutionResult<_>| {
        result
            .outputs
            .iter()
            .filter(|(name, _)| name.starts_with("decoded_"))
            .map(|(_, value)| matches!(value, RuntimeValue::Bool(false)))
            .collect::<Vec<_>>()
    };
    assert!(!decoded(&first).is_empty());
    assert!(decoded(&first).iter().all(|value| *value));
    assert_eq!(decoded(&first), decoded(&second), "wave setting changed decoded output");
}
