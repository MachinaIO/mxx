#[cfg(feature = "gpu")]
use std::{hint::black_box, time::Instant};
#[cfg(feature = "gpu")]
use tracing::info;

#[cfg(feature = "gpu")]
const SIGMA: f64 = 4.578;
#[cfg(feature = "gpu")]
const TRAPDOOR_SIZE: usize = 1;
#[cfg(feature = "gpu")]
const TARGET_COLS: usize = 50;

#[cfg(feature = "gpu")]
fn bench_gpu_preimage() {
    use mxx::{
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::{
            PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::{
            DistType, PolyTrapdoorSampler, PolyUniformSampler,
            trapdoor::GpuDCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
        },
    };

    gpu_device_sync();
    let _ = tracing_subscriber::fmt::try_init();

    // Keep parameters aligned with the CPU benchmark for a fair comparison.
    let cpu_params = DCRTPolyParams::new(16384, 10, 24, 12);
    let (moduli, _, _) = cpu_params.to_crt();
    let params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());

    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
    let uniform_sampler = DCRTPolyUniformSampler::new();

    let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, TRAPDOOR_SIZE);
    let target_cpu = uniform_sampler.sample_uniform(
        &cpu_params,
        TRAPDOOR_SIZE,
        TARGET_COLS,
        DistType::FinRingDist,
    );
    let target = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &target_cpu);

    gpu_device_sync();
    let start = Instant::now();
    let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
    gpu_device_sync();
    let elapsed = start.elapsed();
    black_box(preimage);

    info!("GPU DCRT preimage: {:?}", elapsed);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_gpu_preimage();
}
