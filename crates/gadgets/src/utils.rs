pub use mxx_primitives::utils::*;

use crate::{
    poly::{
        Poly,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};
// Helper function to create a random polynomial using UniformSampler
pub fn create_random_poly(params: &DCRTPolyParams) -> DCRTPoly {
    let sampler = DCRTPolyUniformSampler::new();
    sampler.sample_poly(params, &DistType::FinRingDist)
}

pub fn create_bit_random_poly(params: &DCRTPolyParams) -> DCRTPoly {
    let sampler = DCRTPolyUniformSampler::new();
    sampler.sample_poly(params, &DistType::BitDist)
}

pub fn create_ternary_random_poly(params: &DCRTPolyParams) -> DCRTPoly {
    let sampler = DCRTPolyUniformSampler::new();
    sampler.sample_poly(params, &DistType::TernaryDist)
}

// Helper function to create a bit polynomial (0 or 1)
pub fn create_bit_poly(params: &DCRTPolyParams, bit: bool) -> DCRTPoly {
    if bit { DCRTPoly::const_one(params) } else { DCRTPoly::const_zero(params) }
}
