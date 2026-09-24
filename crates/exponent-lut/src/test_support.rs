//! Shared unit-test helpers for concrete CRT rings and CPU execution.

use mxx_backends::{
    ExecutionConfig, ExecutionResult, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly::cpu_backend,
    execute,
    matrix::dcrt_poly::DCRTPolyMatrix,
    openfhe_guard::gen_modulus_and_warmup,
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    transcript::SamplingMode,
};
use mxx_dsl::{BuiltGraph, Ring};
use mxx_ir_core::{ConcreteRing, ParamEnv, ValidatedGraph};
use std::collections::BTreeMap;

/// The DSL ring with the exact ordered CRT basis of `parameters`.
pub(crate) fn ring_of(parameters: &DCRTPolyParams) -> Ring {
    explicit_ring(&parameters.to_crt().0, parameters.ring_dimension())
}

/// A DSL ring over an explicit ordered CRT basis.
pub(crate) fn explicit_ring(moduli: &[u64], ring_dimension: u32) -> Ring {
    Ring::from_crt_moduli(moduli.iter().copied().map(Into::into).collect(), ring_dimension)
}

/// The concrete ordered CRT ring of an explicit basis.
pub(crate) fn concrete_ring(moduli: &[u64], ring_dimension: u32) -> ConcreteRing {
    explicit_ring(moduli, ring_dimension)
        .as_ref()
        .resolve(&ParamEnv::default(), gen_modulus_and_warmup)
        .expect("explicit CRT basis resolves")
}

/// The product modulus of a graph matrix type's concrete CRT ring.
pub(crate) fn modulus_of(matrix: &mxx_ir_core::types::MatrixType) -> num_bigint::BigInt {
    matrix
        .ring
        .resolve(&ParamEnv::default(), gen_modulus_and_warmup)
        .expect("matrix ring resolves")
        .modulus()
}

/// `count` NTT-friendly primes of `bits` bits for ring dimension `ring_dimension`.
pub(crate) fn primes(ring_dimension: u32, count: usize, bits: usize) -> Vec<u64> {
    gen_modulus_and_warmup(ring_dimension, count, bits, None).expect("generated CRT basis")
}

pub(crate) fn validate(graph: &BuiltGraph) -> ValidatedGraph {
    graph.validate(&ParamEnv::default(), gen_modulus_and_warmup).expect("graph validates")
}

/// Execute a validated graph on the CPU backend with fresh sampling.
pub(crate) fn execute_cpu(
    graph: &ValidatedGraph,
    parameters: impl IntoIterator<Item = DCRTPolyParams>,
    inputs: BTreeMap<String, RuntimeValue>,
) -> ExecutionResult {
    execute(
        graph,
        &mut cpu_backend(parameters),
        inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
        ExecutionConfig::default(),
    )
    .expect("graph executes")
}

/// The full host matrix held by a CPU runtime value.
pub(crate) fn full(value: &RuntimeValue) -> &DCRTPolyMatrix {
    match value {
        RuntimeValue::Matrix(matrix) => matrix.as_cpu_full().expect("full CPU matrix"),
        _ => panic!("runtime value is not a matrix"),
    }
}
