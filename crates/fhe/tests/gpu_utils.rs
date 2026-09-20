//! GPU execution helpers shared by the round-trip integration targets.
use mxx_fhe::BgvParams;
use mxx_primitives::poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams};
use mxx_runtime::{RuntimeValue, backend::poly_gpu::GpuDcrtBackend};
use num_bigint::{BigInt, BigUint};
use std::collections::BTreeMap;
pub type Inputs = BTreeMap<String, RuntimeValue<GpuDcrtBackend>>;
#[path = "../src/gpu_test_utils.rs"]
mod gpu_test_utils;
pub use gpu_test_utils::configure_widths;

pub fn bgv_gpu_parameters(bgv: &BgvParams) -> Vec<GpuDCRTPolyParams> {
    let rings = bgv.runtime_parameters().unwrap();
    let mut parameters: Vec<GpuDCRTPolyParams> = Vec::with_capacity(rings.len());
    for ring in rings {
        let parameter = if let Some(related) = parameters.first() {
            GpuDCRTPolyParams::new_with_gpu(
                ring.ring_dimension(),
                ring.to_crt().0,
                ring.base_bits(),
                related.gpu_ids().to_vec(),
                Some(1),
                Some(related),
                None,
            )
        } else {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        };
        parameters.push(parameter);
    }
    parameters
}

pub fn input(values: &[i64]) -> RuntimeValue<GpuDcrtBackend> {
    RuntimeValue::IndexedFamily(
        values.iter().map(|v| RuntimeValue::Int(BigInt::from(*v))).collect(),
    )
}

pub fn integers(outputs: &Inputs, name: &str) -> Vec<BigInt> {
    let RuntimeValue::IndexedFamily(values) = &outputs[name] else {
        panic!("integer family {name}")
    };
    values
        .iter()
        .map(|value| {
            let RuntimeValue::Int(value) = value else { panic!("integer") };
            value.clone()
        })
        .collect()
}

pub fn centered(value: &BigInt, modulus: &BigUint) -> BigInt {
    use num_integer::Integer;
    let q = BigInt::from(modulus.clone());
    let value = value.mod_floor(&q);
    if value > &q / 2 { value - q } else { value }
}

/// Canonical ciphertext bytes for deterministic replay checks. Call only outside
/// measured intervals: serialization intentionally transfers the result to host.
pub fn matrix_bytes(value: &RuntimeValue<GpuDcrtBackend>, backend: &GpuDcrtBackend) -> Vec<u8> {
    use mxx_runtime::backend::Backend;
    let RuntimeValue::Matrix(matrix) = value else { panic!("materialized matrix") };
    backend.matrix_to_bytes(matrix)
}
