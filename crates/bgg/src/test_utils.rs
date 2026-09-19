use mxx_dsl::BuiltGraph;
use mxx_ir_core::{
    ParamEnv,
    node::NodeKind,
    types::{NodeId, Port, WireRef},
};
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
    for (index, node) in validated.root_scope().execution_order.iter().enumerate() {
        let NodeKind::Input { name, artifact: None, .. } = node.kind() else {
            continue;
        };
        let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
        let concrete = &validated.root_scope().wire_types[&wire];
        let value = inputs.get(name).unwrap_or_else(|| panic!("missing test input {name}"));
        assert!(
            value.matches_wire_type(concrete),
            "test input {name} has runtime kind incompatible with root wire {wire:?}: {concrete:?}"
        );
    }
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
