//! BGV parameter fixtures shared by the GPU tests and the Lean export.

use crate::{
    BgvHybridParams, BgvParams,
    utils::{common_params, integer, primes},
};
use std::env;

/// The BGV test parameters: profile `FHE_TEST_PROFILE` (`bgv-54` by default), with
/// `FHE_TEST_*` variables overriding each value.
pub fn bgv_params() -> BgvParams {
    let profile = env::var("FHE_TEST_PROFILE").unwrap_or_else(|_| "bgv-54".into());
    let (q, p, n, t, digit_size, sigma) = match profile.as_str() {
        "bgv-54" => (
            vec![18014398507892737, 18014398508138497, 18014398508400641],
            vec![72057594037616641],
            8192,
            1032193,
            1,
            "3.2",
        ),
        "bgv-36" => (
            vec![68717740033, 68718346241, 68718428161, 68719230977],
            vec![137438773249, 137438822401],
            8192,
            1032193,
            2,
            "3.2",
        ),
        _ => panic!("FHE_TEST_PROFILE must be bgv-54 or bgv-36"),
    };
    BgvParams::new(
        common_params(
            integer("FHE_TEST_RING_DIMENSION", n),
            primes("FHE_TEST_Q_PRIMES", q),
            integer("FHE_TEST_BASE_BITS", 8),
            sigma,
            false,
        ),
        integer("FHE_TEST_PLAINTEXT_MODULUS", t) as u64,
        Some(BgvHybridParams {
            digit_size: integer("FHE_TEST_HYBRID_DIGIT_SIZE", digit_size),
            auxiliary_primes: primes("FHE_TEST_P_PRIMES", p),
        }),
    )
    .expect("valid BGV parameters")
}

/// The number of levels the test gate switches down (`FHE_TEST_MODSWITCH_STEPS`, default 1).
pub fn modswitch_steps() -> usize {
    let steps = integer("FHE_TEST_MODSWITCH_STEPS", 1);
    assert!(steps > 0, "the round trip must include modulus switching");
    steps
}

/// One GPU context per distinct ordered ring of `bgv`.
#[cfg(feature = "gpu")]
pub fn gpu_parameters(bgv: &BgvParams) -> Vec<mxx_backends::poly::dcrt::gpu::GpuDCRTPolyParams> {
    crate::utils::gpu::related_gpu_parameters(
        bgv.runtime_parameters().expect("BGV runtime parameters"),
    )
}
