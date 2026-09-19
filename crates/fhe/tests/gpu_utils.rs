//! GPU execution helpers shared by the round-trip integration targets.
use mxx_dsl::BuiltGraph;
use mxx_fhe::{BgvParams, FheCommonParams};
use mxx_ir_core::{ParamEnv, ValidatedGraph};
use mxx_primitives::poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams};
use mxx_runtime::{
    MemoryArtifactStore, RuntimeValue,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend},
    execute,
    transcript::SamplingMode,
};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, time::Instant};
pub type Inputs = BTreeMap<String, RuntimeValue<GpuDcrtBackend>>;
#[path = "../src/gpu_test_utils.rs"]
mod gpu_test_utils;
pub use gpu_test_utils::configure_widths;

pub fn backend(common: &FheCommonParams, bgv: Option<&BgvParams>) -> GpuDcrtBackend {
    let rings = if let Some(bgv) = bgv {
        bgv.runtime_parameters().unwrap()
    } else {
        let (primes, _, depth) = common.ring.to_crt();
        let mut rings =
            (0..depth).map(|level| common.parameters_at(level).unwrap()).collect::<Vec<_>>();
        rings
            .extend(primes.iter().map(|p| common.ring.select_modulus(&BigUint::from(*p)).unwrap()));
        rings
    };
    gpu_backend(
        rings
            .iter()
            .map(|p| GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None)),
    )
}

pub fn input(values: &[i64]) -> RuntimeValue<GpuDcrtBackend> {
    RuntimeValue::IndexedFamily(
        values.iter().map(|v| RuntimeValue::Int(BigInt::from(*v))).collect(),
    )
}

pub fn compile(graph: BuiltGraph, backend: &mut GpuDcrtBackend) -> ValidatedGraph {
    let graph = graph.validate(&ParamEnv::default()).expect("valid integration graph");
    configure_widths(backend, &graph);
    graph
}

/// Includes production execution, output retrieval and result-event completion.
/// Timed evaluator graphs use untagged outputs to retain GPU residency; tagged
/// keygen/encryption outputs may persist artifacts and are measured separately.
/// No device-wide synchronization, decryption, or correctness diagnostics are timed.
pub fn run(graph: &ValidatedGraph, backend: &mut GpuDcrtBackend, inputs: Inputs) -> (Inputs, f64) {
    let mut store = MemoryArtifactStore::default();
    // Complete releases from the preceding oracle or input preparation before
    // measuring this call. Default execute keeps releases asynchronous; output
    // event completion below still includes all work producing the result.
    mxx_runtime::backend::Backend::fence_released_memory(backend)
        .expect("complete prior-iteration GPU cleanup");
    let start = Instant::now();
    let mut result =
        execute(graph, backend, inputs, &mut store, SamplingMode::Fresh).expect("GPU execution");
    for name in result.outputs.keys().cloned().collect::<Vec<_>>() {
        if let RuntimeValue::Matrix(matrix) =
            result.materialize_output(&name, backend, &mut store).expect("materialize GPU output")
        {
            matrix.wait_until_ready();
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    result.cleanup_staged(&mut store).unwrap();
    (result.outputs, seconds)
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
