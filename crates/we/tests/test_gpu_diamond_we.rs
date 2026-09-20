#![cfg(feature = "gpu")]

use mxx_gadgets::circuit::{
    BooleanCircuitData, BooleanCircuitShape, BooleanGateData, BooleanGateKind,
};
use mxx_primitives::poly::{
    PolyParams,
    dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
};
use mxx_runtime::{
    artifact::MemoryArtifactStore, authority::GpuExecution,
    gpu_measurement::GpuWarmupMeasurementConfig,
};
use mxx_we::diamond::{DiamondParameterSearch, DiamondWeRuntime};
use std::{env, time::Instant};
use tracing::info;
use tracing_subscriber::EnvFilter;

type GpuDiamondWeRuntime = DiamondWeRuntime<GpuExecution, MemoryArtifactStore>;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_we=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

fn and_circuit() -> BooleanCircuitData {
    BooleanCircuitData {
        layers: vec![vec![BooleanGateData { kind: BooleanGateKind::And, left: 0, right: 1 }]],
        output_source: 0,
    }
}

/// Exercises parameter selection with generated Lean certificates, the GPU cost model, and runtime.
///
/// The small defaults are intentionally only a smoke configuration. All search, measurement, and
/// execution settings are environment-overridable for a larger benchmark invocation.
#[test]
#[ignore = "explicit GPU Diamond WE parameter search and round trip"]
#[serial_test::serial]
fn test_gpu_diamond_we_parameter_search_and_round_trip() {
    install_tracing();
    let total_started = Instant::now();
    let device_ids = detected_gpu_device_ids();
    assert!(!device_ids.is_empty(), "the GPU Diamond WE integration test requires a GPU");
    let effective_parallel_width =
        env_usize("MXX_DIAMOND_WE_GPU_PARALLEL_WIDTH", device_ids.len()).clamp(1, device_ids.len());
    let device_ids = device_ids[..effective_parallel_width].to_vec();
    let shape =
        BooleanCircuitShape { instance_width: 1, witness_width: 1, depth: 1, max_layer_width: 2 };
    let search = DiamondParameterSearch {
        shape,
        min_crt_depth: env_usize("MXX_DIAMOND_WE_GPU_MIN_CRT_DEPTH", 1),
        initial_max_crt_depth: env_usize("MXX_DIAMOND_WE_GPU_INITIAL_MAX_CRT_DEPTH", 1),
        max_crt_depth: env_usize("MXX_DIAMOND_WE_GPU_MAX_CRT_DEPTH", 4),
        min_log_ring_dimension: env_usize("MXX_DIAMOND_WE_GPU_MIN_LOG_RING_DIM", 5),
        max_log_ring_dimension: env_usize("MXX_DIAMOND_WE_GPU_MAX_LOG_RING_DIM", 5),
        crt_modulus_bits: env_usize("MXX_DIAMOND_WE_GPU_CRT_MODULUS_BITS", 60),
        gadget_base_bits: env_usize("MXX_DIAMOND_WE_GPU_GADGET_BASE_BITS", 4) as u32,
        security_bits: env_usize("MXX_DIAMOND_WE_GPU_SECURITY_BITS", 1),
        input_count: 1,
        digit_base: 2,
        batch_bits: 1,
        trapdoor_sigma: 4.0,
        error_sigma: 1.0,
        bgg_tag: b"diamond-we-gpu-integration".to_vec(),
    };

    info!(?device_ids, effective_parallel_width, "starting GPU Diamond WE integration test");
    let search_started = Instant::now();
    let selected = search.search().expect("GPU Diamond WE parameter search");
    info!(
        ring_dimension = selected.ring_dimension,
        crt_depth = selected.crt_depth,
        modulus_bits = selected.modulus_bits,
        achieved_security_bits = selected.achieved_security_bits,
        elapsed_seconds = search_started.elapsed().as_secs_f64(),
        "completed GPU Diamond WE parameter search"
    );

    let (moduli, _, _) = selected.parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new_with_gpu(
        selected.parameters.ring_dimension(),
        moduli,
        selected.parameters.base_bits(),
        device_ids.clone(),
        Some(effective_parallel_width as u32),
        None,
        None,
    );
    let runtime_started = Instant::now();
    let backend = mxx_runtime::backend::poly_gpu::gpu_backend_on(
        [gpu_parameters.clone()],
        device_ids.iter().copied(),
    );
    let mut runtime = GpuDiamondWeRuntime::new(
        selected.compiler,
        GpuExecution::new(
            backend,
            [gpu_parameters.clone()],
            GpuWarmupMeasurementConfig::default(),
            "diamond-production-measured",
        ),
        MemoryArtifactStore::default(),
    )
    .expect("GPU Diamond WE runtime construction");
    let circuit = and_circuit();
    let instance = [true];
    let witness = [true];
    let message = true;
    let encrypt_started = Instant::now();
    let ciphertext = runtime
        .encrypt(&circuit, &instance, message, [0x2a; 32])
        .expect("GPU Diamond WE encryption");
    info!(
        elapsed_seconds = encrypt_started.elapsed().as_secs_f64(),
        "completed GPU Diamond WE encryption"
    );
    let decrypt_started = Instant::now();
    let decoded = runtime
        .decrypt(&circuit, &instance, &witness, &ciphertext)
        .expect("GPU Diamond WE decryption");
    info!(
        elapsed_seconds = decrypt_started.elapsed().as_secs_f64(),
        runtime_elapsed_seconds = runtime_started.elapsed().as_secs_f64(),
        "completed GPU Diamond WE decryption"
    );
    assert_eq!(decoded, message, "GPU Diamond WE round trip must preserve the message");
    let preparations = runtime.execution.preparations();
    assert_eq!(preparations.len(), 2, "Diamond encrypt/decrypt each prepare once");
    let predicted_seconds =
        preparations.iter().map(|prepared| prepared.report().predicted_seconds).sum::<f64>();
    let measurement_counts =
        preparations.iter().map(|prepared| prepared.measurement_count()).collect::<Vec<_>>();
    for prepared in preparations {
        let report = prepared.report();
        assert_eq!(report, prepared.report(), "GPU warmup report reads are pure");
        let evidence = prepared.evidence();
        assert_eq!(evidence, prepared.evidence(), "GPU warmup evidence reads are pure");
        prepared.assert_measurements_unchanged();
    }
    assert_eq!(
        measurement_counts,
        preparations.iter().map(|prepared| prepared.measurement_count()).collect::<Vec<_>>(),
        "GPU measurement counters remain unchanged after production runs"
    );
    info!(predicted_seconds, "Diamond GPU protocol predicted warmup time");
    info!(
        elapsed_seconds = total_started.elapsed().as_secs_f64(),
        "completed GPU Diamond WE integration test"
    );
}
