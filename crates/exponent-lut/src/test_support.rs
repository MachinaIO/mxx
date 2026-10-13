//! Shared unit-test helpers for concrete CRT rings and CPU execution.

use crate::rns_bgg::{RnsBggMaskSecret, RnsBggPayloadSecret, RnsEncryptBuild, RnsOnlineEvalBuild};
use mxx_backends::{
    ExecutionConfig, ExecutionResult, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly::cpu_backend,
    execute,
    matrix::dcrt_poly::DCRTPolyMatrix,
    openfhe_guard::gen_modulus_and_warmup,
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    transcript::SamplingMode,
};
use mxx_dsl::{BuiltGraph, Ring};
use mxx_ir_core::{ParamEnv, ValidatedGraph};
use std::collections::BTreeMap;

/// The DSL ring with the exact ordered CRT basis of `parameters`.
pub(crate) fn ring_of(parameters: &DCRTPolyParams) -> Ring {
    explicit_ring(&parameters.to_crt().0, parameters.ring_dimension())
}

/// A DSL ring over an explicit ordered CRT basis.
pub(crate) fn explicit_ring(moduli: &[u64], ring_dimension: u32) -> Ring {
    Ring::from_crt_moduli(moduli.iter().copied().map(Into::into).collect(), ring_dimension)
}

/// The product modulus of a graph matrix type's concrete CRT ring.
pub(crate) fn modulus_of(matrix: &mxx_ir_core::types::MatrixType) -> num_bigint::BigInt {
    matrix.ring.resolve(&ParamEnv::default()).expect("matrix ring resolves").modulus()
}

/// `count` NTT-friendly primes of `bits` bits for ring dimension `ring_dimension`.
pub(crate) fn primes(ring_dimension: u32, count: usize, bits: usize) -> Vec<u64> {
    gen_modulus_and_warmup(ring_dimension, count, bits, None).expect("generated CRT basis")
}

pub(crate) fn validate(graph: &BuiltGraph) -> ValidatedGraph {
    graph.validate(&ParamEnv::default()).expect("graph validates")
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

/// Preprocess, encrypt and evaluate `program` with the plaintext inputs
/// `lhs` under Static parameters, and return the online result with the
/// payload secret and the stage mask. The online
/// evaluation receives the Program without its LUT values if
/// `withhold_values` (a LUT on a public payload needs them), and
/// `observe` sees its graph. Preprocessing and online evaluation are
/// given the Program data `data`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_without_rhs(
    parameters: &crate::rns_bgg::RnsBggParameters,
    program: &crate::program::ExponentLutProgram,
    data: &crate::program::ProgramData,
    withhold_values: bool,
    lhs: &crate::api::PlaintextInputs<crate::program::WireId, DCRTPoly>,
    execution: &mut mxx_backends::authority::CpuExecution,
    store: &mut MemoryArtifactStore,
    observe: impl FnOnce(&mxx_ir_core::ValidatedGraph),
) -> (crate::api::StageOutputs, RnsBggPayloadSecret, RnsBggMaskSecret) {
    let hash_key: [u8; 32] = rand::random();
    let context: [u8; 32] = rand::random();
    let payload = RnsEncryptBuild::new(parameters, context)
        .sample_payload_secret(execution, store, |_| Ok(()))
        .unwrap();
    let (preprocess, online_eval) = crate::api::preprocess(
        program,
        &ParamEnv::default(),
        &crate::api::PlaintextInputs::<crate::program::RhsId, DCRTPoly>::new(),
        Some(&payload),
        &hash_key,
        crate::rns_bgg::RnsPreprocessBuild::new(parameters, context).with_data(data.clone()),
        execution,
        store,
        |_, outputs| Ok(outputs),
        |_| Ok(()),
    )
    .unwrap();
    let mask = RnsEncryptBuild::new(parameters, rand::random())
        .sample_mask_secret(execution, store, |_| Ok(()))
        .unwrap();
    let encrypted = crate::api::encrypt(
        &program.encryption_inputs(),
        &ParamEnv::default(),
        lhs,
        Some(&preprocess),
        &payload,
        Some(&mask),
        &hash_key,
        RnsEncryptBuild::new(parameters, context),
        execution,
        store,
        |_| Ok(()),
        |_| Ok(()),
    )
    .unwrap();
    let online_program =
        if withhold_values { program.without_lut_values() } else { program.clone() };
    let result = crate::api::online_eval(
        &online_program,
        &ParamEnv::default(),
        &online_eval,
        &encrypted,
        &hash_key,
        RnsOnlineEvalBuild::new(parameters, context).with_data(data.clone()),
        execution,
        store,
        |_, outputs| Ok(outputs),
        |graph| {
            observe(graph);
            Ok(())
        },
    )
    .unwrap();
    (result, payload, mask)
}
