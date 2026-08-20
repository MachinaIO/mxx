#![cfg(feature = "gpu")]

use keccak_asm::Keccak256;
use mxx_bench_estimator::{EstimateConfig, harness::MeasurementHarnessConfig};
use mxx_gadgets::circuit::{
    BooleanCircuitData, BooleanCircuitShape, BooleanGateData, BooleanGateKind,
};
use mxx_primitives::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
    sampler::{
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
};
use mxx_runtime::{ExecutionConfig, artifact::MemoryArtifactStore};
use mxx_we::diamond::{
    DiamondGpuMeasurementBackend, DiamondParameterSearch, DiamondWeRuntime, estimate_diamond_cost,
};
use std::{env, num::NonZeroUsize, time::Instant};
use tracing::info;
use tracing_subscriber::EnvFilter;

type GpuDiamondWeRuntime = DiamondWeRuntime<
    GpuDCRTPolyMatrix,
    GpuDCRTPolyUniformSampler,
    GpuDCRTPolyHashSampler<Keccak256>,
    GpuDCRTPolyTrapdoorSampler,
    MemoryArtifactStore,
>;

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

/// Exercises the real selection, Rust operational checker, GPU cost model, and GPU runtime path.
///
/// The small defaults are intentionally only a smoke configuration. All search, measurement, and
/// execution settings are environment-overridable for a larger benchmark invocation.
#[test]
#[ignore = "explicit GPU Diamond WE parameter search, cost estimation, and round trip"]
#[serial_test::serial]
fn test_gpu_diamond_we_parameter_search_estimate_and_round_trip() {
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
    );
    let warm_up_iterations = env_usize("MXX_DIAMOND_WE_GPU_MEASUREMENT_WARMUPS", 1);
    let measured_iterations = env_usize("MXX_DIAMOND_WE_GPU_MEASUREMENT_ITERATIONS", 1);
    assert!(measured_iterations > 0, "MXX_DIAMOND_WE_GPU_MEASUREMENT_ITERATIONS must be positive");
    let estimate_started = Instant::now();
    let mut measurement_backend = DiamondGpuMeasurementBackend::new(
        gpu_parameters.clone(),
        &device_ids,
        MeasurementHarnessConfig {
            warm_up_iterations,
            measured_iterations,
            ..MeasurementHarnessConfig::default()
        },
    );
    let estimate = estimate_diamond_cost(
        &selected.compiler,
        &mut measurement_backend,
        &EstimateConfig { device_pool_size: effective_parallel_width, per_instance_occupancy: 1 },
    )
    .expect("GPU Diamond WE graph cost estimation");
    info!(
        encryption_work_seconds = estimate.encryption.total_work_seconds,
        encryption_critical_path_seconds = estimate.encryption.critical_path_seconds,
        decryption_work_seconds = estimate.decryption.total_work_seconds,
        decryption_critical_path_seconds = estimate.decryption.critical_path_seconds,
        payload_peak_bytes =
            estimate.encryption.peak_memory_bytes.max(estimate.decryption.peak_memory_bytes),
        workspace_bytes_unmeasured = true,
        elapsed_seconds = estimate_started.elapsed().as_secs_f64(),
        "completed GPU Diamond WE analytical cost estimation"
    );
    assert!(estimate.encryption.total_work_seconds > 0.0);
    assert!(estimate.decryption.total_work_seconds > 0.0);
    assert!(
        estimate
            .encryption
            .persistent_bytes_over_time
            .iter()
            .chain(estimate.decryption.persistent_bytes_over_time.iter())
            .copied()
            .max()
            .unwrap_or_default() >
            0,
        "the analytical matrix/trapdoor payload estimate must be nonzero"
    );

    let runtime_started = Instant::now();
    let mut runtime =
        GpuDiamondWeRuntime::new(selected.compiler, gpu_parameters, MemoryArtifactStore::default())
            .expect("GPU Diamond WE runtime construction")
            .with_execution_config(ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(effective_parallel_width)
                    .expect("effective parallel width is nonzero"),
                ..ExecutionConfig::default()
            });
    let circuit = and_circuit();
    let instance = [true];
    let witness = [true];
    let message = true;
    let encrypt_started = Instant::now();
    let ciphertext =
        runtime.encrypt(&circuit, &instance, message).expect("GPU Diamond WE encryption");
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
    info!(
        elapsed_seconds = total_started.elapsed().as_secs_f64(),
        "completed GPU Diamond WE integration test"
    );
}
