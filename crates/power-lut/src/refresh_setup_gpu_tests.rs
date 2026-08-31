//! GPU acceptance coverage for the real refresh-setup producer.
//!
//! The CPU refresh-setup tests validate frozen graph structure and manifests.
//! This module owns the sampled runtime path so every concrete value in the
//! acceptance test stays in the GPU backend.

use crate::refresh_setup::{
    ImportedRefreshSetup, RefreshPreprocessingProducer, build_refresh_verification,
};
use mxx_dsl::DslContext;
use mxx_ir_core::ParamEnv;
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    RuntimeValue,
    artifact::{ArtifactStore, MemoryArtifactStore},
    backend::poly::gpu::gpu_backend,
    execute_in_session_with_config, execute_with_config,
    executor::ExecutionConfig,
    transcript::SamplingMode,
};
use serial_test::serial;
use std::{collections::BTreeMap, ffi::OsStr, num::NonZeroUsize, time::Instant};
use tracing::info;
use tracing_subscriber::EnvFilter;

const MAX_PARALLEL_INSTANCES_ENV: &str = "MXX_POWER_LUT_REFRESH_SETUP_MAX_PARALLEL_INSTANCES";
const DEFAULT_MAX_PARALLEL_INSTANCES: usize = 64;

/// Parse the parallelism knob owned by this GPU acceptance test.
///
/// This is deliberately not a runtime-wide setting: it only controls the four
/// executions below. Missing configuration uses the conservative test default;
/// malformed, zero, non-Unicode, or overflowing values fail closed.
fn parse_max_parallel_instances(value: Option<&OsStr>) -> Result<NonZeroUsize, String> {
    let value = match value {
        Some(value) => value.to_str().ok_or_else(|| {
            format!("{MAX_PARALLEL_INSTANCES_ENV} must be valid Unicode decimal digits")
        })?,
        None => return Ok(NonZeroUsize::new(DEFAULT_MAX_PARALLEL_INSTANCES).unwrap()),
    };
    if value.is_empty() {
        return Err(format!("{MAX_PARALLEL_INSTANCES_ENV} must not be empty"));
    }
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("{MAX_PARALLEL_INSTANCES_ENV} must contain only ASCII decimal digits"));
    }
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{MAX_PARALLEL_INSTANCES_ENV} does not fit usize"))?;
    NonZeroUsize::new(parsed)
        .ok_or_else(|| format!("{MAX_PARALLEL_INSTANCES_ENV} must be greater than zero"))
}

fn refresh_setup_execution_config() -> Result<ExecutionConfig, String> {
    let max_parallel_instances =
        parse_max_parallel_instances(std::env::var_os(MAX_PARALLEL_INSTANCES_ENV).as_deref())?;
    Ok(ExecutionConfig {
        max_parallel_instances,
        preimage_progress: None,
        release_fence_interval: None,
    })
}

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_power_lut=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

#[test]
#[serial(dcrt_runtime)]
fn producer_execute_export_import_refresh_and_verify_gpu() {
    install_tracing();
    let execution_config = refresh_setup_execution_config()
        .unwrap_or_else(|error| panic!("invalid {MAX_PARALLEL_INSTANCES_ENV}: {error}"));
    let fixture = crate::refresh_setup::tests::fixture_with_ring_and_crt(4, 2);
    let dcrt = DCRTPolyParams::new(4, 2, 17, 16);
    let (moduli, crt_bits, crt_depth) = dcrt.to_crt();
    assert_eq!(fixture.parameters.layout.ring_dimension, 4.into());
    assert_eq!(fixture.parameters.layout.digit_count, dcrt.modulus_digits());
    assert_eq!(fixture.parameters.component_count, 2);
    assert_eq!(fixture.parameters.coefficient_count, 1);
    assert_eq!(fixture.parameters.refresh.crt_plaintext_moduli.len(), 2);
    let mask_label_count = fixture.parameters.refresh.crt_plaintext_moduli.len() *
        fixture.parameters.component_count *
        fixture.parameters.coefficient_count *
        fixture.parameters.digit_count;
    let fresh_label_count = fixture.parameters.component_count *
        fixture.parameters.coefficient_count *
        fixture.parameters.digit_count;
    info!(
        mask_labels = mask_label_count,
        fresh_labels = fresh_label_count,
        selector_cells = fixture.selector_bit_values.len(),
        "refresh setup GPU structural batches"
    );

    assert_eq!(crt_depth, 2);
    assert_eq!(crt_bits.div_ceil(16), 2);
    let gpu_parameters = GpuDCRTPolyParams::new(4, moduli, 16);
    assert_eq!(gpu_parameters.modulus_digits(), dcrt.modulus_digits());
    let producer = RefreshPreprocessingProducer::build(crate::refresh_setup::tests::request_clone(
        &fixture.request,
    ))
    .unwrap();
    let mut store = MemoryArtifactStore::default();

    let selector = fixture.selector_graph.validate(&ParamEnv::default()).unwrap();
    let selector_parallel_loops = selector
        .source
        .root_scope()
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(_)))
        .count();
    assert_eq!(selector_parallel_loops, 1, "selector must have one active-cell ParallelLoop");
    let selector_started = Instant::now();
    info!(
        stage = "selector_execute",
        count = fixture.selector_bit_values.len(),
        max_parallel_instances = execution_config.max_parallel_instances.get(),
        "refresh setup GPU stage started"
    );
    let selector_result = execute_in_session_with_config(
        &selector,
        &mut gpu_backend([gpu_parameters.clone()]),
        BTreeMap::from([
            ("refresh-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32])),
            (
                fixture.selector_bit_input_name.clone(),
                RuntimeValue::IndexedFamily(
                    fixture
                        .selector_bit_values
                        .iter()
                        .map(|bit| {
                            let value = DCRTPolyMatrix::from_poly_vec(
                                &dcrt,
                                vec![vec![DCRTPoly::from_usize_to_constant(&dcrt, *bit as usize)]],
                            );
                            RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                                &gpu_parameters,
                                &value,
                            ))
                        })
                        .collect(),
                ),
            ),
        ]),
        &mut store,
        fixture.selector_production.execution_nonce,
        execution_config,
    )
    .unwrap();
    info!(
        stage = "selector_execute",
        count = fixture.selector_bit_values.len(),
        elapsed_ms = selector_started.elapsed().as_secs_f64() * 1_000.0,
        "refresh setup GPU stage completed"
    );
    assert_eq!(selector_result.production_id.as_ref(), Some(&fixture.selector_production));
    let selector_manifest = store.manifest(&fixture.selector_production).cloned().unwrap();
    mxx_ir_core::artifact::validate_manifest(&selector_manifest).unwrap();
    let validated = producer
        .built()
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(fixture.selector_production.clone(), selector_manifest.clone())]),
        )
        .unwrap();
    let producer_started = Instant::now();
    info!(
        stage = "producer_execute",
        count = fixture.parameters.refresh.crt_plaintext_moduli.len(),
        max_parallel_instances = execution_config.max_parallel_instances.get(),
        "refresh setup GPU stage started"
    );
    let producer_result = execute_with_config(
        &validated,
        &mut gpu_backend([gpu_parameters.clone()]),
        BTreeMap::from([("refresh-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
        &mut store,
        SamplingMode::Fresh,
        execution_config,
    )
    .unwrap();
    info!(
        stage = "producer_execute",
        count = fixture.parameters.refresh.crt_plaintext_moduli.len(),
        elapsed_ms = producer_started.elapsed().as_secs_f64() * 1_000.0,
        "refresh setup GPU stage completed"
    );
    let production_id = producer_result.production_id.clone().unwrap();
    let mut manifest = store.load_manifest(&production_id).unwrap();
    producer.finalize_export_manifest(&mut manifest).unwrap();
    mxx_ir_core::artifact::validate_manifest(&manifest).unwrap();
    store.store_manifest(manifest.clone()).unwrap();
    assert!(
        store.manifest(&fixture.selector_production).is_some(),
        "selector production manifest must remain available for verification validation"
    );
    assert!(
        store.manifest(&manifest.production_id).is_some(),
        "refresh producer manifest must remain available for verification validation"
    );

    let ring = fixture.parameters.layout.ring();
    let imported = ImportedRefreshSetup::import(
        production_id,
        fixture.parameters.clone(),
        producer.declaration().clone(),
        producer.attestation(),
        &manifest,
    )
    .unwrap();
    assert_eq!(imported.public_b.matrix_type(), &ring.matrix_type((2, 12)));
    assert_eq!(imported.preimages.len(), 2);
    for slot in 0..imported.preimages.len() {
        assert_eq!(imported.preimages[slot].matrix_type(), &ring.matrix_type((12, 8)));
        assert_eq!(imported.decoder_bases[slot].vector.matrix_type(), &ring.matrix_type((1, 12)));
    }
    let setup =
        fixture.parameters.refresh.bind_imported_setup(&fixture.compiler, &imported).unwrap();
    let refreshed = fixture.parameters.refresh.refresh(&fixture.compiler, &setup).unwrap();
    let public_key_columns = fixture.parameters.layout.public_key_columns();
    let expected = ring.zero((1, 1));
    let verification = build_refresh_verification(
        refreshed.encoding(),
        &fixture.request.secret,
        &expected,
        fixture.parameters.layout.gadget_base.clone(),
        fixture.parameters.digit_count,
        fixture.parameters.refresh.full_modulus.clone(),
        1,
    )
    .unwrap();
    let mut relation_context = DslContext::new("refresh-setup-gpu-verification");
    for slot in 0..fixture.parameters.refresh.crt_plaintext_moduli.len() {
        let scale = ring.polynomial([fixture.parameters.refresh.scale_expression(slot).unwrap()]);
        let combined = fixture
            .compiler
            .bgg
            .add(
                &fixture
                    .compiler
                    .bgg
                    .add(
                        &fixture.compiler.bgg.large_scalar_mul(&imported.state, &scale),
                        &imported.masks[slot],
                    )
                    .unwrap(),
                &fixture.compiler.bgg.large_scalar_mul(&imported.fresh, &scale),
            )
            .unwrap();
        let target = combined.pubkey.matrix - scale * imported.a_prime.clone();
        let actual = imported.public_b.clone() * imported.preimages[slot].clone();
        relation_context = relation_context
            .public_output(format!("actual_bk_{slot}"), actual)
            .unwrap()
            .public_output(format!("target_{slot}"), target)
            .unwrap();
    }
    relation_context = relation_context
        .public_output("gpu_expected_zero", ring.zero((1, public_key_columns)))
        .unwrap();
    let graph = verification
        .add_outputs(relation_context, "residual", "decoded")
        .unwrap()
        .build()
        .unwrap()
        // The verification graph consumes the imported refresh encoding, so
        // validation must resolve the exact finalized producer manifest.  A
        // bare structural validation would correctly fail closed with
        // `MissingManifest`; passing this manifest preserves the artifact
        // identity/confidentiality checks before GPU execution.
        ;
    let verification_productions = graph
        .graph
        .scopes()
        .values()
        .flat_map(|scope| scope.nodes())
        .filter_map(|node| match node.kind() {
            mxx_ir_core::node::NodeKind::Input { artifact: Some(artifact), .. } => {
                Some(artifact.production_id.clone())
            }
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let expected_productions = std::collections::BTreeSet::from([
        fixture.selector_production.clone(),
        manifest.production_id.clone(),
    ]);
    assert_eq!(
        verification_productions, expected_productions,
        "verification must depend on exactly selector and refresh producer manifests"
    );
    let verification_manifests = BTreeMap::from([
        (fixture.selector_production.clone(), selector_manifest.clone()),
        (manifest.production_id.clone(), manifest.clone()),
    ]);
    let graph =
        graph.validate_with_manifests(&ParamEnv::default(), &verification_manifests).unwrap();
    let verification_started = Instant::now();
    info!(
        stage = "verification_execute",
        count = fixture.parameters.refresh.crt_plaintext_moduli.len(),
        max_parallel_instances = execution_config.max_parallel_instances.get(),
        "refresh setup GPU stage started"
    );
    let result = execute_with_config(
        &graph,
        &mut gpu_backend([gpu_parameters.clone()]),
        BTreeMap::new(),
        &mut store,
        SamplingMode::Fresh,
        execution_config,
    )
    .unwrap();
    info!(
        stage = "verification_execute",
        count = fixture.parameters.refresh.crt_plaintext_moduli.len(),
        elapsed_ms = verification_started.elapsed().as_secs_f64() * 1_000.0,
        "refresh setup GPU stage completed"
    );
    let RuntimeValue::Matrix(residual) = &result.outputs["residual"] else {
        panic!("verification residual must be a matrix")
    };
    let RuntimeValue::Matrix(expected_zero) = &result.outputs["gpu_expected_zero"] else {
        panic!("GPU zero output must be a matrix")
    };
    assert_eq!(residual, expected_zero, "noiseless GPU residual is not zero");
    for slot in 0..fixture.parameters.refresh.crt_plaintext_moduli.len() {
        let RuntimeValue::Matrix(actual) = &result.outputs[&format!("actual_bk_{slot}")] else {
            panic!("B*K must be a matrix")
        };
        let RuntimeValue::Matrix(target) = &result.outputs[&format!("target_{slot}")] else {
            panic!("target must be a matrix")
        };
        assert_eq!((actual.nrow, actual.ncol), (2, 8));
        assert_eq!((target.nrow, target.ncol), (2, 8));
        assert_eq!(actual, target, "B*K target mismatch at CRT slot {slot}");
    }
    assert_eq!(verification.decoded().len(), public_key_columns);
    for column in 0..public_key_columns {
        assert!(matches!(&result.outputs[&format!("decoded_{column}")], RuntimeValue::Bool(false)));
    }

    let wrong = ring.polynomial([1.into()]);
    let wrong_verification = build_refresh_verification(
        refreshed.encoding(),
        &fixture.request.secret,
        &wrong,
        fixture.parameters.layout.gadget_base.clone(),
        fixture.parameters.digit_count,
        fixture.parameters.refresh.full_modulus.clone(),
        1,
    )
    .unwrap();
    let wrong_graph = wrong_verification
        .add_outputs(DslContext::new("refresh-setup-gpu-wrong-verification"), "residual", "decoded")
        .unwrap()
        .build()
        .unwrap()
        // Keep the deliberately wrong plaintext under the same validated
        // artifact boundary; only the expected plaintext changes here.
        .validate_with_manifests(&ParamEnv::default(), &verification_manifests)
        .unwrap();
    let wrong_verification_started = Instant::now();
    info!(
        stage = "wrong_verification_execute",
        count = 1usize,
        max_parallel_instances = execution_config.max_parallel_instances.get(),
        "refresh setup GPU stage started"
    );
    let wrong_result = execute_with_config(
        &wrong_graph,
        &mut gpu_backend([gpu_parameters]),
        BTreeMap::new(),
        &mut store,
        SamplingMode::Fresh,
        execution_config,
    )
    .unwrap();
    info!(
        stage = "wrong_verification_execute",
        count = 1usize,
        elapsed_ms = wrong_verification_started.elapsed().as_secs_f64() * 1_000.0,
        "refresh setup GPU stage completed"
    );
    let RuntimeValue::Matrix(wrong_residual) = &wrong_result.outputs["residual"] else {
        panic!("wrong verification residual must be a matrix")
    };
    assert_ne!(wrong_residual, expected_zero, "wrong expected plaintext remained zero");
}

#[cfg(test)]
mod parallelism_parser_tests {
    use super::{DEFAULT_MAX_PARALLEL_INSTANCES, parse_max_parallel_instances};
    use std::{ffi::OsStr, num::NonZeroUsize};

    #[test]
    fn missing_value_uses_test_default() {
        assert_eq!(
            parse_max_parallel_instances(None).unwrap(),
            NonZeroUsize::new(DEFAULT_MAX_PARALLEL_INSTANCES).unwrap()
        );
    }

    #[test]
    fn parses_one_and_large_values_without_clamping() {
        assert_eq!(parse_max_parallel_instances(Some(OsStr::new("1"))).unwrap().get(), 1);
        let large = usize::MAX.to_string();
        assert_eq!(
            parse_max_parallel_instances(Some(OsStr::new(&large))).unwrap().get(),
            usize::MAX
        );
    }

    #[test]
    fn rejects_zero() {
        let error = parse_max_parallel_instances(Some(OsStr::new("0"))).unwrap_err();
        assert!(error.contains("greater than zero"));
    }

    #[test]
    fn rejects_malformed_and_empty_values() {
        let malformed = parse_max_parallel_instances(Some(OsStr::new("1.5"))).unwrap_err();
        assert!(malformed.contains("ASCII decimal digits"));
        let empty = parse_max_parallel_instances(Some(OsStr::new(""))).unwrap_err();
        assert!(empty.contains("must not be empty"));
    }

    #[test]
    fn rejects_overflow() {
        let overflow = format!("{}0", usize::MAX);
        let error = parse_max_parallel_instances(Some(OsStr::new(&overflow))).unwrap_err();
        assert!(error.contains("does not fit usize"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_nonunicode_values() {
        use std::os::unix::ffi::OsStrExt;

        let error = parse_max_parallel_instances(Some(OsStr::from_bytes(b"\xff"))).unwrap_err();
        assert!(error.contains("valid Unicode"));
    }
}
