//! GPU-only checks for RNS-BGG parameter and ring wiring.
//!
//! The former fixture suite depended on the removed scheme/material facade.
//! End-to-end GPU coverage belongs to application tests that construct a
//! complete stage; this module keeps the crate-local GPU gate compiling and
//! checks the parameter-to-device ring contract without duplicating a runner.

use super::RnsBggParameters;
use mxx_backends::poly::{
    PolyParams,
    dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
};
use num_bigint::BigUint;
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

#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_rns_bgg_operations_per_drop_three_at_boundaries() {
    let pool = DCRTPolyParams::new(16, 4, 17, 4, None, None);
    let primes = pool.to_crt().0;
    let levels = [4usize, 3, 2, 1]
        .into_iter()
        .map(|depth| {
            let modulus = primes[..depth]
                .iter()
                .fold(BigUint::from(1u8), |product, prime| product * BigUint::from(*prime));
            pool.select_modulus(&modulus).expect("nested RNS modulus level")
        })
        .collect::<Vec<_>>();
    let parameters = RnsBggParameters::new(levels, 1, 3).expect("valid cadence");

    assert_eq!(parameters.operations_per_drop(), 3);
    // Logical levels 0..=2 share the first physical CRT depth.  The drop is
    // selected at the first operation in each subsequent cadence window.
    assert_eq!(super::planned_crt_depth(0, 3, 1, 4, 1).unwrap(), 4);
    assert_eq!(super::planned_crt_depth(2, 3, 1, 4, 1).unwrap(), 4);
    assert_eq!(super::planned_crt_depth(3, 3, 1, 4, 1).unwrap(), 3);
    assert_eq!(super::planned_crt_depth(5, 3, 1, 4, 1).unwrap(), 3);
    assert_eq!(super::planned_crt_depth(6, 3, 1, 4, 1).unwrap(), 2);
    assert_eq!(super::planned_crt_depth(9, 3, 1, 4, 1).unwrap(), 1);
    assert!(super::planned_crt_depth(0, 0, 1, 4, 1).is_err());
}
