#[cfg(feature = "gpu")]
use std::{hint::black_box, time::Instant};

#[cfg(feature = "gpu")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|value| value.parse::<usize>().unwrap_or_else(|_| panic!("{name} must be an integer")))
        .unwrap_or(default)
}

#[cfg(feature = "gpu")]
fn bench_gpu_small_rhs() {
    use mxx_primitives::{
        matrix::PolyMatrixSmallRhs,
        poly::{
            PolyParams,
            dcrt::{
                gpu::{
                    GpuDCRTPolyParams, gpu_default_mempool_reset_high_water,
                    gpu_default_mempool_usage,
                },
                params::DCRTPolyParams,
            },
        },
        sampler::{DistType, PolyUniformSampler, gpu::GpuDCRTPolyUniformSampler},
    };

    let ring_dimension = env_usize("MXX_BENCH_GPU_RING_DIMENSION", 1024);
    let crt_depth = env_usize("MXX_BENCH_GPU_CRT_DEPTH", 4);
    let crt_bits = env_usize("MXX_BENCH_GPU_CRT_BITS", 28);
    let base_bits = env_usize("MXX_BENCH_GPU_BASE_BITS", 8);
    let lhs_rows = env_usize("MXX_BENCH_GPU_LHS_ROWS", 2);
    let inner_rows = env_usize("MXX_BENCH_GPU_INNER_ROWS", 4);
    let rhs_columns = env_usize("MXX_BENCH_GPU_RHS_COLUMNS", 32);
    assert!(
        lhs_rows > 0 && inner_rows > 0 && rhs_columns > 0,
        "matrix dimensions must be positive"
    );

    let ring_dimension = u32::try_from(ring_dimension).expect("ring dimension exceeds u32");
    let base_bits = u32::try_from(base_bits).expect("base bits exceeds u32");
    let cpu_params = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
    let (moduli, _, _) = cpu_params.to_crt();
    let params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());
    let digit_count = params.crt_bits().div_ceil(params.base_bits() as usize);
    let lhs_columns = inner_rows.checked_mul(digit_count).expect("lhs column count overflow");
    let sampler = GpuDCRTPolyUniformSampler::new();
    let lhs = sampler.sample_uniform(&params, lhs_rows, lhs_columns, DistType::FinRingDist);
    let target = sampler.sample_uniform(&params, inner_rows, rhs_columns, DistType::FinRingDist);
    lhs.wait_until_ready();
    target.wait_until_ready();
    params.fence_released_memory();

    let device = *params.gpu_ids().first().expect("GPU benchmark requires one device");
    gpu_default_mempool_reset_high_water(device).expect("reset CUDA mempool high-water");
    let baseline =
        gpu_default_mempool_usage(device).expect("query CUDA mempool baseline").used_current;

    let end_to_end_start = Instant::now();
    let decomposition_start = Instant::now();
    let compact = target.gadget_decompose(true).expect("compact gadget decomposition failed");
    compact.wait_until_ready();
    let decomposition_seconds = decomposition_start.elapsed().as_secs_f64();
    let allocation =
        compact.allocation_report(&lhs).expect("query compact multiplication allocation report");

    let multiplication_start = Instant::now();
    let output = lhs.multiply_small_rhs(&compact).expect("compact RHS multiplication failed");
    output.wait_until_ready();
    let multiplication_seconds = multiplication_start.elapsed().as_secs_f64();
    let end_to_end_seconds = end_to_end_start.elapsed().as_secs_f64();

    let usage = gpu_default_mempool_usage(device).expect("query CUDA mempool peak");
    let incremental_peak_bytes = usage
        .used_high
        .checked_sub(baseline)
        .expect("CUDA mempool high-water fell below the measured baseline");
    black_box(output);

    println!(
        "gpu_small_rhs ring_dimension={ring_dimension} crt_depth={crt_depth} crt_bits={crt_bits} \
base_bits={base_bits} lhs_rows={lhs_rows} inner_rows={inner_rows} rhs_columns={rhs_columns} \
digit_count={digit_count}"
    );
    println!(
        "runtime_shard_columns={} inner_dimension={} active_limbs={}",
        rhs_columns, lhs_columns, crt_depth
    );
    println!("compact_rhs_bytes={}", allocation.compact_rhs_bytes);
    println!("full_output_bytes={}", allocation.full_output_bytes);
    println!("expanded_rhs_workspace_bytes={}", allocation.expanded_rhs_workspace_bytes);
    println!("modeled_high_water_bytes={}", allocation.high_water_bytes);
    println!("decomposition_seconds={decomposition_seconds:.9}");
    println!("multiplication_seconds={multiplication_seconds:.9}");
    println!("end_to_end_seconds={end_to_end_seconds:.9}");
    println!("mempool_baseline_bytes={baseline}");
    println!("mempool_peak_bytes={}", usage.used_high);
    println!("mempool_incremental_peak_bytes={incremental_peak_bytes}");
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_gpu_small_rhs();
}
