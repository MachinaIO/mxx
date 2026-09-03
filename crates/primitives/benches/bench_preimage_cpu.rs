use mxx_primitives::{
    matrix::{PolyMatrix, ResidentPolyMatrixColumnSource},
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, bounds::default_preimage_cutoff,
        trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
    },
};
use std::{hint::black_box, time::Instant};
use tracing::info;

const SIGMA: f64 = 4.578;
const TRAPDOOR_SIZE: usize = 1;
const TARGET_COLS: usize = 50;

fn bench_cpu_preimage() {
    let _ = tracing_subscriber::fmt::try_init();

    // Keep parameters aligned with the GPU benchmark for a fair comparison.
    let params = DCRTPolyParams::new(16384, 10, 24, 12);
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
    let uniform_sampler = DCRTPolyUniformSampler::new();

    let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, TRAPDOOR_SIZE);
    let target =
        uniform_sampler.sample_uniform(&params, TRAPDOOR_SIZE, TARGET_COLS, DistType::FinRingDist);

    let start = Instant::now();
    let bound = default_preimage_cutoff(
        params.ring_dimension(),
        public_matrix.row_size(),
        params.modulus_digits(),
        1u32 << params.base_bits(),
        SIGMA,
    )
    .expect("preimage cutoff");
    let target_source = ResidentPolyMatrixColumnSource::new(target);
    let preimage = trapdoor_sampler
        .preimage(&params, &trapdoor, &public_matrix, &target_source, bound, rand::random())
        .expect("compact preimage sampling");
    let elapsed = start.elapsed();
    black_box(preimage);

    info!("CPU DCRT preimage: {:?}", elapsed);
}

fn main() {
    bench_cpu_preimage();
}
