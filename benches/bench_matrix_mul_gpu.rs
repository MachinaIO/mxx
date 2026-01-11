use std::{hint::black_box, time::Instant};

use mxx::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{DistType, PolyUniformSampler},
};

#[cfg(feature = "gpu")]
fn bench_gpu_matrix_mul() {
    use mxx::{
        poly::dcrt::gpu::{GpuDCRTPolyParams, gpu_device_sync},
        sampler::gpu_uniform::GpuDCRTPolyUniformSampler,
    };

    gpu_device_sync();
    let _ = tracing_subscriber::fmt::try_init();
    let cpu_params = DCRTPolyParams::new(8192, 7, 51, 30);
    let (moduli, _, _) = cpu_params.to_crt();
    let params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let left = uniform_sampler.sample_uniform(&params, 200, 200, DistType::FinRingDist);
    let right = uniform_sampler.sample_uniform(&params, 200, 200, DistType::FinRingDist);

    // gpu_device_sync();
    let start = Instant::now();
    let result = &left * &right;
    // gpu_device_sync();
    let elapsed = start.elapsed();
    black_box(result);

    println!("GPU GpuDCRTPolyMatrix mul: {:?}", elapsed);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_gpu_matrix_mul();
}
