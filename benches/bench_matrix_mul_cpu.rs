use std::{hint::black_box, time::Instant};

use mxx::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};

fn bench_cpu_matrix_mul() {
    let _ = tracing_subscriber::fmt::try_init();
    let params = DCRTPolyParams::new(2048, 15, 24, 19);
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let left = uniform_sampler.sample_uniform(&params, 200, 200, DistType::FinRingDist);
    let right = uniform_sampler.sample_uniform(&params, 200, 200, DistType::FinRingDist);

    let start = Instant::now();
    let result = &left * &right;
    let elapsed = start.elapsed();
    black_box(result);

    println!("CPU DCRTPolyMatrix mul: {:?}", elapsed);
}

fn main() {
    bench_cpu_matrix_mul();
}
