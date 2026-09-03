use mxx_dsl::BuiltGraph;
use mxx_ir_core::ParamEnv;
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    ExecutionResult, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly::{CpuDcrtBackend, cpu_backend},
    execute,
    transcript::SamplingMode,
};
use std::collections::BTreeMap;

pub fn row(parameters: &DCRTPolyParams, columns: usize, offset: usize) -> DCRTPolyMatrix {
    DCRTPolyMatrix::from_poly_vec_row(
        parameters,
        (0..columns)
            .map(|index| {
                DCRTPoly::const_rotate_poly(
                    parameters,
                    (index + offset) % parameters.ring_dimension() as usize,
                )
            })
            .collect(),
    )
}

pub fn execute_graph(
    graph: BuiltGraph,
    parameters: DCRTPolyParams,
    inputs: BTreeMap<String, RuntimeValue<CpuDcrtBackend>>,
) -> ExecutionResult<CpuDcrtBackend> {
    let validated = graph.validate(&ParamEnv::default()).expect("valid runtime graph");
    execute(
        &validated,
        &mut cpu_backend([parameters]),
        inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("graph execution")
}

pub fn matrix_output<'a>(
    result: &'a ExecutionResult<CpuDcrtBackend>,
    name: &str,
) -> &'a DCRTPolyMatrix {
    let RuntimeValue::Matrix(value) = &result.outputs[name] else {
        panic!("{name} must be a matrix output")
    };
    value.as_ref()
}

pub fn small_matrix_output(result: &ExecutionResult<CpuDcrtBackend>, name: &str) -> DCRTPolyMatrix {
    let RuntimeValue::SmallMatrix(value) = &result.outputs[name] else {
        panic!("{name} must be a compact small-matrix output")
    };
    value.decode_full().expect("valid compact small-matrix output")
}
