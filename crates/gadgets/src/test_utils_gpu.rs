use crate::{circuit::PolyCircuit, test_utils::build_circuit_graph};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    RuntimeValue, artifact::MemoryArtifactStore, backend::poly::gpu::gpu_backend, execute,
    transcript::SamplingMode,
};
use std::collections::BTreeMap;

#[test]
fn test_gpu_dsl_ir_runtime_executes_gadget_arithmetic() {
    let parameters = DCRTPolyParams::new(8, 1, 20, 4);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits());
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let lhs = circuit.input(1).as_single_wire();
    let rhs = circuit.input(1).as_single_wire();
    let sum = circuit.add_gate(lhs, rhs);
    circuit.output([sum]);
    let graph = build_circuit_graph("gpu-dsl-ir-runtime", &parameters, &circuit, 2, (1, 1));

    let lhs = DCRTPolyMatrix::from_poly_vec_row(
        &parameters,
        vec![DCRTPoly::from_usize_to_constant(&parameters, 7)],
    );
    let rhs = DCRTPolyMatrix::from_poly_vec_row(
        &parameters,
        vec![DCRTPoly::from_usize_to_constant(&parameters, 11)],
    );
    let expected = lhs.clone() + &rhs;
    let result = execute(
        &graph,
        &mut gpu_backend([gpu_parameters.clone()]),
        BTreeMap::from([
            (
                "input-0".to_owned(),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, &lhs)),
            ),
            (
                "input-1".to_owned(),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, &rhs)),
            ),
        ]),
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("execute gadget graph on the GPU runtime backend");
    let RuntimeValue::Matrix(actual) = &result.outputs["output-0"] else {
        panic!("gadget output must be a matrix")
    };
    assert_eq!(actual.to_cpu_matrix(), expected);
}
