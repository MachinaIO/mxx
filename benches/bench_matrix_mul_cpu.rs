use mxx::{
    poly::dcrt::params::DCRTPolyParams,
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};
use std::{hint::black_box, time::Instant};
use tracing::info;

fn bench_cpu_matrix_mul() {
    let _ = tracing_subscriber::fmt::try_init();
    let params = DCRTPolyParams::new(16384, 15, 24, 12);
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let left = uniform_sampler.sample_uniform(&params, 1, 30, DistType::FinRingDist);
    let right = uniform_sampler.sample_uniform(&params, 30, 120, DistType::FinRingDist);

    let start = Instant::now();
    let result = &left * &right;
    let elapsed = start.elapsed();
    black_box(result);

    info!("CPU DCRTPolyMatrix mul: {:?}", elapsed);
}

fn main() {
    bench_cpu_matrix_mul();
}
