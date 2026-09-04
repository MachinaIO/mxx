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
    );
    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU required for artifact-backed smoke test");
    let mut backend = gpu_backend_on([gpu_parameters], devices);
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
