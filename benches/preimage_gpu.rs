use std::{hint::black_box, time::Instant};

use mxx::{
    matrix::PolyMatrix,
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{DistType, PolyTrapdoorSampler, PolyUniformSampler},
};

const SIGMA: f64 = 4.578;

#[cfg(feature = "gpu")]
fn bench_preimage_gpu() {
    use mxx::{
        poly::dcrt::gpu::{GpuDCRTPolyParams, gpu_device_sync},
        sampler::{gpu_uniform::GpuDCRTPolyUniformSampler, trapdoor::GpuDCRTPolyTrapdoorSampler},
    };

    gpu_device_sync();
    let _ = tracing_subscriber::fmt::try_init();
    let cpu_params = DCRTPolyParams::new(2048, 15, 24, 19);
    let (moduli, _, _) = cpu_params.to_crt();
    let params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());
    let size = 2;
    let target_cols = 1000;
    let k = params.modulus_digits();
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
    let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let target = uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

    gpu_device_sync();
    let start = Instant::now();
    let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
    let elapsed = start.elapsed();
    black_box(&preimage);

    let expected_rows = size * (k + 2);
    let expected_cols = target_cols;

    assert_eq!(
        preimage.row_size(),
        expected_rows,
        "Preimage matrix should have the correct number of rows"
    );
    assert_eq!(
        preimage.col_size(),
        expected_cols,
        "Preimage matrix should have the correct number of columns (equal to target columns)"
    );

    let product = public_matrix * &preimage;
    assert_eq!(product, target, "Product of public matrix and preimage should equal target");

    println!("GPU preimage generation: {:?}", elapsed);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_preimage_gpu();
}
