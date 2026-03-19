use crate::poly::{Poly, PolyParams};
use rayon::prelude::*;

pub(super) fn map_nested_rns_outputs_with_params_gpu<P, T, F>(
    params: &P::Params,
    output_count: usize,
    f: F,
) -> Vec<T>
where
    P: Poly,
    T: Send,
    F: Fn(usize, &P::Params) -> T + Send + Sync,
{
    let device_ids = params.device_ids();
    assert!(
        !device_ids.is_empty(),
        "at least one GPU device is required for nested-RNS GPU encoding"
    );

    let chunk_parallelism = device_ids.len();
    let mut outputs = Vec::with_capacity(output_count);
    for chunk_start in (0..output_count).step_by(chunk_parallelism.max(1)) {
        let chunk_len = (chunk_start + chunk_parallelism).min(output_count) - chunk_start;
        let chunk_device_ids = &device_ids[..chunk_len];
        let mut chunk_outputs = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let local_params = params.params_for_device(chunk_device_ids[offset]);
                f(chunk_start + offset, &local_params)
            })
            .collect::<Vec<_>>();
        outputs.append(&mut chunk_outputs);
    }
    outputs
}

#[cfg(test)]
mod tests {
    use super::super::{encode_nested_rns_poly, encode_nested_rns_poly_compact_bytes};
    use crate::{
        __PAIR, __TestState,
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
                params::DCRTPolyParams,
                poly::DCRTPoly,
            },
        },
    };
    use num_bigint::BigUint;

    const P_MODULI_BITS: usize = 6;
    const BASE_BITS: u32 = 6;

    #[test]
    #[sequential_test::sequential]
    fn test_gpu_encode_nested_rns_poly_compact_bytes_matches_cpu_bytes() {
        let _ = tracing_subscriber::fmt::try_init();
        gpu_device_sync();

        let cpu_params = DCRTPolyParams::new(4, 6, 18, BASE_BITS);
        let (moduli, _, _) = cpu_params.to_crt();
        let gpu_ids = detected_gpu_device_ids();
        assert!(
            !gpu_ids.is_empty(),
            "at least one GPU device is required for test_gpu_encode_nested_rns_poly_compact_bytes_matches_cpu_bytes"
        );
        let gpu_params = GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli,
            cpu_params.base_bits(),
            gpu_ids.clone(),
            Some(gpu_ids.len() as u32),
        );
        let q_level = Some(2);

        for input in [12345u64, 23456u64, 34567u64] {
            let input = BigUint::from(input);
            let expected =
                encode_nested_rns_poly::<DCRTPoly>(P_MODULI_BITS, &cpu_params, &input, q_level);
            let actual_bytes = encode_nested_rns_poly_compact_bytes::<GpuDCRTPoly>(
                P_MODULI_BITS,
                &gpu_params,
                &input,
                q_level,
            );

            assert_eq!(actual_bytes.len(), expected.len());
            for (idx, (actual_bytes, expected_poly)) in
                actual_bytes.into_iter().zip(expected.into_iter()).enumerate()
            {
                let local_params = gpu_params.params_for_device(gpu_ids[idx % gpu_ids.len()]);
                let actual_poly = GpuDCRTPoly::from_compact_bytes(&local_params, &actual_bytes);
                assert_eq!(actual_poly.coeffs_biguints(), expected_poly.coeffs_biguints());
            }

            gpu_device_sync();
        }
    }
}
