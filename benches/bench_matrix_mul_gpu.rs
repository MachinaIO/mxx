use std::{hint::black_box, time::Instant};
use tracing::info;

#[cfg(feature = "gpu")]
fn bench_gpu_matrix_mul() {
    use mxx::{
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::{
            PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    gpu_device_sync();
    let _ = tracing_subscriber::fmt::try_init();
    let cpu_params = DCRTPolyParams::new(16384, 15, 24, 12);
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let left_cpu = uniform_sampler.sample_uniform(&cpu_params, 1, 30, DistType::FinRingDist);
    let right_cpu = uniform_sampler.sample_uniform(&cpu_params, 30, 120, DistType::FinRingDist);

    let (moduli, _, _) = cpu_params.to_crt();
    let params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());
    let left = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left_cpu);
    let right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &right_cpu);

    let start = Instant::now();
    let (result, kernel_ms) = left.mul_with_kernel_time(&right);
    let elapsed = start.elapsed();
    black_box(result);

    info!("GPU GpuDCRTPolyMatrix mul: total={:?}, kernel={:.3} ms", elapsed, kernel_ms);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_gpu_matrix_mul();
}
