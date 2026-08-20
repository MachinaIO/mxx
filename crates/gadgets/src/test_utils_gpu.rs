use crate::{
    circuit::PolyCircuit,
    circuit_gadgets::{
        arith::{
            CrtWindow,
            nested_rns::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
        },
        fhe::{
            ring_gsw::RingGswCiphertext,
            ring_gsw_nested_rns::{
                NestedRnsRingGswContext, active_q_modulus, ciphertext_from_outputs,
                ciphertext_inputs_from_native, decrypt_ciphertext, encrypt_plaintext_bit,
                sample_public_key, sample_secret_key,
            },
        },
    },
    test_utils::{build_circuit_graph, diagonal_matrix},
};
use mxx_dsl::{DslContext, Family, Ring};
use mxx_ir_core::{ParamEnv, node::NodeKind};
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
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

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

#[test]
fn test_gpu_parallel_loop_executes_batched_matrix_arithmetic() {
    let parameters = DCRTPolyParams::new(8, 1, 20, 4);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits());
    let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
    let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
    let families = (0..4)
        .map(|_| Family::pack(vec![ring.identity(1), ring.zero((1, 1))]).expect("matrix family"))
        .collect::<Vec<_>>();
    let sums = Family::parallel_zip_many_values(families, |_, inputs| {
        inputs.into_iter().reduce(|left, right| left + right).expect("non-empty batch")
    })
    .expect("parallel matrix batch");
    let built = DslContext::new("gpu-parallel-matrix-batch")
        .output("first", sums.get_static(0))
        .expect("first output")
        .output("second", sums.get_static(1))
        .expect("second output")
        .build()
        .expect("build GPU batch graph");
    let graph = built.validate(&ParamEnv::default()).expect("validate GPU batch graph");
    let execution = execute(
        &graph,
        &mut gpu_backend([gpu_parameters]),
        BTreeMap::new(),
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("execute batched matrix arithmetic on the GPU runtime backend");
    let RuntimeValue::Matrix(first) = &execution.outputs["first"] else {
        panic!("first batch output must be a matrix")
    };
    let RuntimeValue::Matrix(second) = &execution.outputs["second"] else {
        panic!("second batch output must be a matrix")
    };
    let four = DCRTPolyMatrix::from_poly_vec_row(
        &parameters,
        vec![DCRTPoly::from_usize_to_constant(&parameters, 4)],
    );
    assert_eq!(first.to_cpu_matrix(), four);
    assert_eq!(second.to_cpu_matrix(), DCRTPolyMatrix::zero(&parameters, 1, 1));
}

#[test]
#[serial_test::serial]
fn test_gpu_packed_nested_rns_addition_matches_cpu_matrices() {
    let parameters = DCRTPolyParams::new(2, 2, 12, 6);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits());
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let context =
        Arc::new(NestedRnsPolyContext::setup(&mut circuit, &parameters, 6, 2, 16, false, None));
    let coefficient_slots = 2;
    let window = CrtWindow::full(context.q_moduli_depth);
    let left = NestedRnsPoly::input(context.clone(), coefficient_slots, window, &mut circuit);
    let right = NestedRnsPoly::input(context.clone(), coefficient_slots, window, &mut circuit);
    let sum = left.add(&right, &mut circuit);
    circuit.output([sum.inner]);

    let encode = |values: &[BigUint]| {
        encode_nested_rns_poly::<DCRTPoly>(
            context.p_moduli_bits,
            context.max_unreduced_muls,
            &parameters,
            values,
            window,
        )
        .into_iter()
        .map(|lanes| {
            diagonal_matrix(
                &parameters,
                lanes.into_iter().map(|lane| DCRTPoly::from_biguint_to_constant(&parameters, lane)),
            )
        })
        .collect::<Vec<_>>()
    };
    let left_inputs = encode(&[BigUint::from(1u8), BigUint::from(2u8)]);
    let right_inputs = encode(&[BigUint::from(3u8), BigUint::from(4u8)]);
    let expected = left_inputs
        .iter()
        .zip(&right_inputs)
        .map(|(left, right)| left.clone() + right.clone())
        .collect::<Vec<_>>();
    let cpu_inputs = left_inputs.into_iter().chain(right_inputs).collect::<Vec<_>>();
    let wire_size = coefficient_slots * context.q_moduli_depth;
    assert!(wire_size > 1);
    let graph = build_circuit_graph(
        "gpu-packed-nested-rns-addition",
        &parameters,
        &circuit,
        circuit.num_input(),
        (wire_size, wire_size),
    );
    let runtime_inputs = cpu_inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            (
                format!("input-{index}"),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, input)),
            )
        })
        .collect();
    let execution = execute(
        &graph,
        &mut gpu_backend([gpu_parameters]),
        runtime_inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("execute packed nested-RNS addition on the GPU runtime backend");
    for (index, expected) in expected.into_iter().enumerate() {
        let RuntimeValue::Matrix(actual) = &execution.outputs[&format!("output-{index}")] else {
            panic!("packed nested-RNS output must be a matrix")
        };
        assert_eq!(actual.to_cpu_matrix(), expected);
    }
}

#[test]
#[ignore = "full nested-RNS Ring-GSW GPU runtime round trip takes more than six minutes"]
fn test_gpu_ring_gsw_arithmetic_executes_through_dsl_ir_runtime_and_decrypts() {
    let ring_dimension = 2u32;
    let active_levels = 1usize;
    let parameters = DCRTPolyParams::new(ring_dimension, active_levels, 10, 5);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits());
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let nested_rns = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        &parameters,
        5,
        2,
        16,
        false,
        Some(active_levels),
    ));
    let context: Arc<NestedRnsRingGswContext<DCRTPoly>> =
        Arc::new(crate::circuit_gadgets::fhe::ring_gsw::RingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            ring_dimension as usize,
            nested_rns,
            CrtWindow::new(0, active_levels, parameters.to_crt().2),
        ));
    let inputs = (0..2)
        .map(|_| RingGswCiphertext::input(context.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let results = [
        inputs[0].add(&inputs[1], &mut circuit),
        inputs[0].sub(&inputs[1], &mut circuit),
        inputs[0].mul(&inputs[1], &mut circuit),
    ];
    let mut output_wires = Vec::new();
    for result in &results {
        output_wires.extend(result.reconstruct(&mut circuit));
    }
    circuit.output(output_wires);

    let secret_key = sample_secret_key(&parameters);
    let public_key = sample_public_key(
        &parameters,
        context.width(),
        &secret_key,
        rand::random(),
        b"gpu-ring-gsw-dsl-ir-runtime",
        Some(0.0),
    );
    let plaintexts = [false, true];
    let cpu_inputs = plaintexts
        .into_par_iter()
        .flat_map_iter(|plaintext| {
            let ciphertext = encrypt_plaintext_bit(
                &parameters,
                context.nested_rns.as_ref(),
                &public_key,
                plaintext,
            );
            ciphertext_inputs_from_native(
                &parameters,
                &parameters,
                context.nested_rns.as_ref(),
                &ciphertext,
                CrtWindow { offset: context.level_offset, depth: context.active_levels },
            )
        })
        .collect::<Vec<_>>();
    let graph = build_circuit_graph(
        "gpu-ring-gsw-dsl-ir-runtime",
        &parameters,
        &circuit,
        circuit.num_input(),
        (ring_dimension as usize, ring_dimension as usize),
    );
    assert!(
        graph
            .source
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_))),
        "Ring-GSW column calls must lower to an IR parallel loop"
    );
    let runtime_inputs = cpu_inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            (
                format!("input-{index}"),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, input)),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let execution = execute(
        &graph,
        &mut gpu_backend([gpu_parameters]),
        runtime_inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("execute Ring-GSW graph on the GPU runtime backend");
    let outputs = (0..circuit.output_gate_ids().len())
        .map(|index| {
            let RuntimeValue::Matrix(output) = &execution.outputs[&format!("output-{index}")]
            else {
                panic!("Ring-GSW output must be a matrix")
            };
            output.to_cpu_matrix()
        })
        .collect::<Vec<_>>();

    let output_width = 2 * context.width();
    let expectations = [
        (3u64, (u64::from(plaintexts[0]) + u64::from(plaintexts[1])) % 3),
        (3u64, (u64::from(plaintexts[0]) + 3 - u64::from(plaintexts[1])) % 3),
        (2u64, u64::from(plaintexts[0] && plaintexts[1])),
    ];
    let q_modulus = active_q_modulus(context.nested_rns.as_ref());
    outputs.par_chunks(output_width).zip(expectations).for_each(
        |(output, (plaintext_modulus, expected))| {
            let ciphertext = ciphertext_from_outputs(&parameters, output, context.width());
            let decrypted = decrypt_ciphertext::<DCRTPoly, DCRTPolyMatrix>(
                &parameters,
                context.nested_rns.as_ref(),
                &ciphertext,
                &secret_key,
                plaintext_modulus,
            );
            let half_q = &q_modulus / 2u8;
            let rounded = decrypted
                .coeffs_biguints()
                .into_iter()
                .map(|coefficient| {
                    ((num_bigint::BigUint::from(plaintext_modulus) * coefficient + &half_q) /
                        &q_modulus)
                        .to_u64()
                        .expect("rounded coefficient fits u64") %
                        plaintext_modulus
                })
                .collect::<Vec<_>>();
            let mut expected_coefficients = vec![0; ring_dimension as usize];
            expected_coefficients[0] = expected;
            assert_eq!(rounded, expected_coefficients);
        },
    );
}
