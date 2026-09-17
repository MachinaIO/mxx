//! GPU-only checks for RNS-BGG parameter and ring wiring.
//!
//! The former fixture suite depended on the removed scheme/material facade.
//! End-to-end GPU coverage belongs to application tests that construct a
//! complete stage; this module keeps the crate-local GPU gate compiling and
//! checks the parameter-to-device ring contract without duplicating a runner.

use super::RnsBggParameters;
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
};
use serial_test::serial as sequential;

fn gpu_parameters(level: &DCRTPolyParams, device: i32) -> GpuDCRTPolyParams {
    GpuDCRTPolyParams::new_with_gpu(
        level.ring_dimension(),
        level.to_crt().0,
        level.base_bits(),
        vec![device],
        None,
        None,
        None,
    )
}

#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_rns_bgg_parameter_levels_preserve_ring_shapes() {
    let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
    let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
    let gpu = gpu_parameters(parameters.level(0).unwrap(), 0);
    assert_eq!(gpu.ring_dimension(), level.ring_dimension());
    assert_eq!(gpu.to_crt().0, level.to_crt().0);
}
