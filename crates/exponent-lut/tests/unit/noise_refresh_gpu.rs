//! GPU unit execution of the production switched decoder graph (library test module).

use mxx_primitives::{
    matrix::PolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};
use mxx_runtime::{
    MemoryArtifactStore, RuntimeValue, backend::poly_gpu::gpu_backend_on, transcript::SamplingMode,
};
use std::collections::BTreeMap;

#[test]
#[serial_test::serial(dcrt_runtime)]
fn test_gpu_switched_decoder_numeric_round_trip_with_shared_fresh_error() {
    let (graph, parameters, bounds) = super::setup::tests::switched_decoder_numeric_fixture();
    let source = &parameters[0];
    let root =
        GpuDCRTPolyParams::new(source.ring_dimension(), source.to_crt().0, source.base_bits());
    let selected = parameters
        .iter()
        .skip(1)
        .map(|p| root.select_modulus(p.modulus().as_ref()).unwrap())
        .collect::<Vec<_>>();
    let mut rings = vec![root];
    rings.extend(selected);
    let mut backend = gpu_backend_on(rings, detected_gpu_device_ids());
    for _execution in 0..2 {
        let result = mxx_runtime::execute(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let mut observations = BTreeMap::new();
        for (name, value) in result.outputs {
            let RuntimeValue::Matrix(value) = value else {
                panic!("matrix output expected");
            };
            // Host copies occur only after the entire GPU graph, at the assertion boundary.
            let matrix = backend.gather_matrix_for_host(&value).unwrap();
            let mut coefficients = Vec::new();
            for row in 0..matrix.row_size() {
                for column in 0..matrix.col_size() {
                    coefficients.extend(matrix.entry(row, column).coeffs_biguints());
                }
            }
            observations.insert(name, (matrix.params().modulus().as_ref().clone(), coefficients));
        }
        super::setup::tests::assert_numeric_refresh_outputs(observations, &bounds);
    }
}
