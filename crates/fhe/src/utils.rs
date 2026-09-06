//! Coefficient-boundary graphs. RNS modulus switching must not call these.
use crate::FheError;
use mxx_dsl::{DslError, Family, Int, Mat, Ring, select};
use mxx_ir_core::IntExpr;
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};

pub(crate) fn check_family(values: &Family<Int>, count: usize) -> Result<(), FheError> {
    if values.count() != &IntExpr::constant(count) {
        return Err(FheError::ShapeMismatch);
    }
    Ok(())
}

pub(crate) fn check_matrix(
    parameters: &DCRTPolyParams,
    value: &Mat,
    rows: usize,
    columns: usize,
) -> Result<(), FheError> {
    let ty = value.matrix_type();
    if ty.modulus != IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())) ||
        ty.ring_dimension != IntExpr::constant(parameters.ring_dimension())
    {
        return Err(FheError::LevelMismatch);
    }
    if ty.rows != IntExpr::constant(rows) || ty.columns != IntExpr::constant(columns) {
        return Err(FheError::ShapeMismatch);
    }
    Ok(())
}

pub(crate) fn scalar(parameters: &DCRTPolyParams, value: impl Into<BigInt>) -> Mat {
    Ring::new(
        IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
        parameters.ring_dimension(),
    )
    .polynomial([IntExpr::constant(value.into())])
}

pub(crate) fn centered(value: Int, modulus: &BigUint) -> Result<Int, DslError> {
    let q = BigInt::from(modulus.clone());
    // select uses index 0 for false and 1 for true; the latter keeps residues
    // at or below q/2, while larger residues are interpreted as value - q.
    let above_half = value.clone().mul(2).less_equal(Int::constant(q.clone()));
    select(above_half.to_int(), vec![value.clone().sub(Int::constant(q)), value])
}

pub(crate) fn extract(parameters: &DCRTPolyParams, value: &Mat) -> Result<Family<Int>, FheError> {
    check_matrix(parameters, value, 1, 1)?;
    Ok(value.coefficients())
}

pub(crate) fn pack(parameters: &DCRTPolyParams, values: &Family<Int>) -> Result<Mat, FheError> {
    check_family(values, parameters.ring_dimension() as usize)?;
    Ok(Ring::new(parameters.modulus().as_ref().clone(), parameters.ring_dimension())
        .from_coefficients(values))
}

#[cfg(test)]
use crate::FheCommonParams;
#[cfg(test)]
use mxx_dsl::BuiltGraph;
#[cfg(test)]
use mxx_ir_core::{ParamEnv, node::SampleRange};
#[cfg(test)]
use mxx_runtime::{
    ExecutionResult, MemoryArtifactStore, RuntimeValue,
    backend::poly::{CpuDcrtBackend, cpu_backend},
    execute,
    transcript::SamplingMode,
};
#[cfg(test)]
use std::collections::BTreeMap;

// FHE_TEST_RING_DIMENSION/CRT_DEPTH/CRT_BITS/BASE_BITS control the toy ring;
// FHE_TEST_SIGMA/ERROR_CUTOFF control its bounded noise. Defaults are correctness
// fixtures only. Larger dimensions also require compatible SIMD plaintext moduli.
#[cfg(test)]
pub(crate) fn common() -> FheCommonParams {
    let read = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .map(|value| value.parse().expect("integer test parameter"))
            .unwrap_or(default)
    };
    FheCommonParams {
        ring: DCRTPolyParams::new(
            read("FHE_TEST_RING_DIMENSION", 8) as u32,
            read("FHE_TEST_CRT_DEPTH", 3),
            read("FHE_TEST_CRT_BITS", 30),
            read("FHE_TEST_BASE_BITS", 4) as u32,
            None,
            None,
        ),
        secret_range: SampleRange { minimum: (-1).into(), maximum: 1.into() },
        error_sigma: std::env::var("FHE_TEST_SIGMA")
            .ok()
            .map(|s| s.parse().expect("Gaussian sigma"))
            .unwrap_or(1.0),
        error_cutoff: BigUint::from(read("FHE_TEST_ERROR_CUTOFF", 8)),
    }
}

#[cfg(test)]
pub(crate) fn execute_graph(
    graph: BuiltGraph,
    common: &FheCommonParams,
    inputs: BTreeMap<String, RuntimeValue<CpuDcrtBackend>>,
    extra_parameters: &[DCRTPolyParams],
) -> ExecutionResult<CpuDcrtBackend> {
    // Register prefix rings for ciphertext levels and single-prime rings for
    // modswitch corrections; a modulus alone does not specify CRT tower order.
    let (moduli, _, depth) = common.ring.to_crt();
    let mut parameters =
        (0..depth).map(|level| common.parameters_at(level).unwrap()).collect::<Vec<_>>();
    parameters
        .extend(moduli.iter().map(|p| common.ring.select_modulus(&BigUint::from(*p)).unwrap()));
    let validated = graph.validate(&ParamEnv::default()).expect("valid FHE DSL graph");
    parameters.extend_from_slice(extra_parameters);
    let mut backend = cpu_backend(parameters);
    let mut store = MemoryArtifactStore::default();
    let mut result = execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh)
        .expect("FHE graph execution");
    // Structural families may be lazy staged outputs. Materialize them before
    // dropping the memory store so assertions observe actual runtime values.
    for name in result.outputs.keys().cloned().collect::<Vec<_>>() {
        result.materialize_output(&name, &backend, &mut store).expect("load FHE output");
    }
    result.cleanup_staged(&mut store).expect("clean streamed outputs");
    result
}

#[cfg(test)]
pub(crate) fn int_input(values: &[i64]) -> RuntimeValue<CpuDcrtBackend> {
    RuntimeValue::IndexedFamily(
        values.iter().map(|v| RuntimeValue::Int(BigInt::from(*v))).collect(),
    )
}

#[cfg(test)]
pub(crate) fn integers(result: &ExecutionResult<CpuDcrtBackend>, name: &str) -> Vec<BigInt> {
    let RuntimeValue::IndexedFamily(values) = &result.outputs[name] else {
        panic!("expected integer family {name}")
    };
    values
        .iter()
        .map(|v| {
            let RuntimeValue::Int(value) = v else { panic!("expected integer") };
            value.clone()
        })
        .collect()
}
