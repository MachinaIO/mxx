#![cfg(feature = "gpu")]

//! Observable semantics of direct GPU Graph nodes, compared with the CPU matrix backend.

use mxx_backends::{
    GpuRuntime, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly_gpu::gpu_backend,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
};
use mxx_dsl::{DslContext, Mat, Ring};
use mxx_ir_core::{
    ParamEnv,
    node::{ConcatAxis, IndexRange},
    ring::{RingExpr, RingRef},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, sync::Arc};

fn cpu_input(parameters: &DCRTPolyParams) -> DCRTPolyMatrix {
    let entry = |constant: u64| {
        let mut coefficients = vec![BigUint::from(0u8); parameters.ring_dimension() as usize];
        coefficients[0] = BigUint::from(constant);
        coefficients[1] = BigUint::from(constant + 1);
        coefficients[3] = BigUint::from(2 * constant + 1);
        DCRTPoly::from_biguints(parameters, &coefficients)
    };
    DCRTPolyMatrix::from_poly_vec(
        parameters,
        vec![vec![entry(1), entry(2)], vec![entry(3), entry(4)]],
    )
}

fn assert_same_matrix_values(label: &str, actual: &DCRTPolyMatrix, expected: &DCRTPolyMatrix) {
    assert_eq!(actual.size(), expected.size(), "{label}: shape");
    assert_eq!(
        actual.params().ring_dimension(),
        expected.params().ring_dimension(),
        "{label}: ring dimension"
    );
    assert_eq!(actual.params().moduli(), expected.params().moduli(), "{label}: ordered CRT basis");
    for row in 0..actual.size().0 {
        for column in 0..actual.size().1 {
            assert_eq!(
                actual.entry(row, column).coeffs(),
                expected.entry(row, column).coeffs(),
                "{label}: coefficients at ({row}, {column})",
            );
        }
    }
}

fn run_shape_case(
    label: &str,
    expression: impl FnOnce(Mat) -> Mat,
    expected: impl FnOnce(&DCRTPolyMatrix) -> DCRTPolyMatrix,
) {
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_parameters = GpuDCRTPolyParams::new(
        parameters.ring_dimension(),
        parameters.moduli().to_vec(),
        parameters.base_bits(),
        None,
    );
    let ring = Ring::from_crt_moduli(
        parameters.moduli().iter().copied().map(Into::into).collect(),
        parameters.ring_dimension(),
    );
    let graph = DslContext::new(label)
        .output("result", expression(ring.input("source", (2, 2))))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let input = cpu_input(&parameters);
    let input_ring = RingRef::new(RingExpr::Explicit {
        crt_moduli: parameters.moduli().iter().copied().map(Into::into).collect(),
        ring_dimension: parameters.ring_dimension(),
    })
    .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
    .unwrap();
    let value = RuntimeValue::gpu_matrix(
        ConcreteWireType::Matrix(ConcreteMatrixType { ring: input_ring, rows: 2, columns: 2 }),
        Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, &input)),
    )
    .unwrap();
    let bindings = BTreeMap::from([("source".to_owned(), value)]);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters])).unwrap();
    let mut plan = runtime.plan(graph, &bindings).unwrap();
    let result =
        runtime.execute(&mut plan, bindings, &mut MemoryArtifactStore::default(), [7; 32]).unwrap();
    let actual = runtime.download_matrix(&result.outputs["result"]).unwrap();
    assert_same_matrix_values(label, &actual, &expected(&input));
}

fn cpu_crt_input(parameters: &DCRTPolyParams) -> DCRTPolyMatrix {
    let modulus = parameters.modulus();
    let q = modulus.as_ref();
    let values = [q / 4u8, q / 2u8 - 3u8, q / 2u8 + 3u8, q - 1u8];
    DCRTPolyMatrix::from_poly_vec(
        parameters,
        values
            .chunks(2)
            .map(|row| {
                row.iter()
                    .map(|value| DCRTPoly::from_biguint_to_constant(parameters, value.clone()))
                    .collect()
            })
            .collect(),
    )
}

fn run_crt_case(
    label: &str,
    source_is_full: bool,
    expression: impl FnOnce(Mat, &Ring) -> Mat,
    expected: impl FnOnce(&DCRTPolyMatrix, &DCRTPolyParams) -> DCRTPolyMatrix,
) {
    run_crt_case_with_prime(label, source_is_full, 0, expression, expected);
}

fn run_crt_case_with_prime(
    label: &str,
    source_is_full: bool,
    prime_index: usize,
    expression: impl FnOnce(Mat, &Ring) -> Mat,
    expected: impl FnOnce(&DCRTPolyMatrix, &DCRTPolyParams) -> DCRTPolyMatrix,
) {
    let full = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let selected_prime = full.moduli()[prime_index];
    let single = full.select_modulus(&BigUint::from(selected_prime)).unwrap();
    let gpu_full = GpuDCRTPolyParams::new(
        full.ring_dimension(),
        full.moduli().to_vec(),
        full.base_bits(),
        None,
    );
    let gpu_single = GpuDCRTPolyParams::new_with_gpu(
        single.ring_dimension(),
        single.moduli().to_vec(),
        single.base_bits(),
        gpu_full.gpu_ids().to_vec(),
        Some(1),
        Some(&gpu_full),
        None,
    );
    let (source, destination, gpu_source) =
        if source_is_full { (&full, &single, &gpu_full) } else { (&single, &full, &gpu_single) };
    let source_ring = Ring::from_crt_moduli(
        source.moduli().iter().copied().map(Into::into).collect(),
        source.ring_dimension(),
    );
    let destination_ring = Ring::from_crt_moduli(
        destination.moduli().iter().copied().map(Into::into).collect(),
        destination.ring_dimension(),
    );
    let graph = DslContext::new(label)
        .output("result", expression(source_ring.input("source", (2, 2)), &destination_ring))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let input = cpu_crt_input(source);
    let concrete_ring = RingRef::new(RingExpr::Explicit {
        crt_moduli: source.moduli().iter().copied().map(Into::into).collect(),
        ring_dimension: source.ring_dimension(),
    })
    .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
    .unwrap();
    let value = RuntimeValue::gpu_matrix(
        ConcreteWireType::Matrix(ConcreteMatrixType { ring: concrete_ring, rows: 2, columns: 2 }),
        Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(gpu_source, &input)),
    )
    .unwrap();
    let bindings = BTreeMap::from([("source".to_owned(), value)]);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_full, gpu_single])).unwrap();
    let mut plan = runtime.plan(graph, &bindings).unwrap();
    let result =
        runtime.execute(&mut plan, bindings, &mut MemoryArtifactStore::default(), [9; 32]).unwrap();
    let actual = runtime.download_matrix(&result.outputs["result"]).unwrap();
    assert_same_matrix_values(label, &actual, &expected(&input, destination));
}

fn run_manual_matrix_case(
    label: &str,
    kind: mxx_ir_core::node::NodeKind,
    expected: impl FnOnce(&DCRTPolyMatrix) -> DCRTPolyMatrix,
) {
    use mxx_ir_core::{Graph, GraphOutput, NodeHandle, WireType};
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_parameters = GpuDCRTPolyParams::new(
        parameters.ring_dimension(),
        parameters.moduli().to_vec(),
        parameters.base_bits(),
        None,
    );
    let ring = Ring::from_crt_moduli(
        parameters.moduli().iter().copied().map(Into::into).collect(),
        parameters.ring_dimension(),
    );
    let wire_type = WireType::Matrix(ring.matrix_type((2, 2)));
    let source = NodeHandle::new(
        mxx_ir_core::node::NodeKind::Input {
            name: "source".into(),
            wire_type: wire_type.clone(),
            artifact: None,
        },
        vec![],
        vec![wire_type.clone()],
    );
    let result = NodeHandle::new(kind, vec![source.output(0).unwrap()], vec![wire_type]);
    let graph = Graph::freeze(
        label,
        vec![],
        BTreeMap::from([(
            "result".into(),
            GraphOutput { value: result.output(0).unwrap(), availability: None },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let graph = mxx_ir_core::validate(
        &graph,
        &ParamEnv::default(),
        mxx_backends::openfhe_guard::gen_modulus_and_warmup,
    )
    .unwrap();
    let input = cpu_input(&parameters);
    let concrete_ring = RingRef::new(RingExpr::Explicit {
        crt_moduli: parameters.moduli().iter().copied().map(Into::into).collect(),
        ring_dimension: parameters.ring_dimension(),
    })
    .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
    .unwrap();
    let value = RuntimeValue::gpu_matrix(
        ConcreteWireType::Matrix(ConcreteMatrixType { ring: concrete_ring, rows: 2, columns: 2 }),
        Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, &input)),
    )
    .unwrap();
    let bindings = BTreeMap::from([("source".to_owned(), value)]);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters])).unwrap();
    let mut plan = runtime.plan(graph, &bindings).unwrap();
    let result = runtime
        .execute(&mut plan, bindings, &mut MemoryArtifactStore::default(), [11; 32])
        .unwrap();
    let actual = runtime.download_matrix(&result.outputs["result"]).unwrap();
    assert_same_matrix_values(label, &actual, &expected(&input));
}

#[test]
#[serial_test::serial]
fn matrix_negate_matches_cpu() {
    run_shape_case("direct-negate", |a| -a, |a| a.negate_out_of_place());
}

#[test]
#[serial_test::serial]
fn matrix_transpose_matches_cpu() {
    run_shape_case("direct-transpose", |a| a.transpose(), |a| a.transpose());
}

#[test]
#[serial_test::serial]
fn matrix_slice_matches_cpu() {
    let range =
        |start: usize, end: usize| Some(IndexRange { start: start.into(), end: end.into() });
    run_shape_case("direct-slice", |a| a.slice(range(0, 2), range(1, 2)), |a| a.slice(0, 2, 1, 2));
}

#[test]
#[serial_test::serial]
fn matrix_tensor_matches_cpu() {
    run_shape_case("direct-tensor", |a| a.clone().tensor(a), |a| a.tensor(a));
}

#[test]
#[serial_test::serial]
fn matrix_concat_rows_matches_cpu() {
    run_shape_case(
        "direct-concat-rows",
        |a| Mat::concat(ConcatAxis::Rows, vec![a.clone(), a]),
        |a| a.concat_rows(&[a]),
    );
}

#[test]
#[serial_test::serial]
fn matrix_concat_columns_matches_cpu() {
    run_shape_case(
        "direct-concat-columns",
        |a| Mat::concat(ConcatAxis::Columns, vec![a.clone(), a]),
        |a| a.concat_columns(&[a]),
    );
}

#[test]
#[serial_test::serial]
fn matrix_concat_diagonal_matches_cpu() {
    run_shape_case(
        "direct-concat-diagonal",
        |a| Mat::concat(ConcatAxis::Diagonal, vec![a.clone(), a]),
        |a| a.concat_diag(&[a]),
    );
}

#[test]
#[serial_test::serial]
fn matrix_mul_accumulate_matches_unfused_cpu() {
    run_shape_case(
        "direct-matrix-mul-accumulate",
        |a| {
            Mat::multi_row_gemm_accumulate(
                vec![(2, a.clone(), a.clone()), (-1, a.clone(), a.clone())],
                Some(a),
            )
        },
        |a| a.clone() * a + a,
    );
}

#[test]
#[serial_test::serial]
fn gadget_decompose_and_small_rhs_multiply_reconstruct_input() {
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let digit_count = parameters.modulus_digits();
    run_shape_case(
        "direct-gadget-decompose-small-rhs",
        move |source| {
            let ring = Ring::from_ref(source.matrix_type().ring.clone());
            source.decompose(16, digit_count).mul_small_rhs(ring.gadget(2, 16, digit_count))
        },
        |source| source.clone(),
    );
}

#[test]
#[serial_test::serial]
fn balanced_small_gadget_decompose_reconstructs_full_crt_input() {
    use mxx_ir_core::node::ConstantMatrix;
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let digits = parameters.crt_bits().div_ceil(parameters.base_bits() as usize);
    run_shape_case(
        "direct-balanced-small-gadget-decompose",
        move |source| {
            let ring = Ring::from_ref(source.matrix_type().ring.clone());
            let gadget = ring
                .constant((2, 2 * digits), ConstantMatrix::Gadget { base: 16.into(), small: true });
            source.small_decompose(16, digits).mul_small_rhs(gadget)
        },
        |source| source.clone(),
    );
}

#[test]
#[serial_test::serial]
fn ring_automorphism_matches_cpu() {
    run_shape_case(
        "direct-ring-automorphism",
        |a| a.ring_automorphism(3),
        |a| a.ring_automorphism_out_of_place(3),
    );
}

#[test]
#[serial_test::serial]
fn matrix_scale_matches_cpu() {
    run_manual_matrix_case(
        "direct-matrix-scale",
        mxx_ir_core::node::NodeKind::MatrixScale { scalar: 3.into() },
        |a| a.clone() + a + a,
    );
}

#[test]
#[serial_test::serial]
fn loop_index_matrix_scale_replays_with_fresh_input() {
    run_loop_index_matrix_scale_case(
        "direct-loop-index-matrix-scale",
        mxx_ir_core::IntExpr::LoopIndex(0),
        |index| BigInt::from(index),
    );
}

#[test]
#[serial_test::serial]
fn negative_loop_index_matrix_scale_matches_cpu() {
    use mxx_ir_core::IntExpr;
    run_loop_index_matrix_scale_case(
        "direct-negative-loop-index-matrix-scale",
        IntExpr::Sub(Box::new(IntExpr::constant(0)), Box::new(IntExpr::LoopIndex(0))),
        |index| -BigInt::from(index),
    );
}

#[test]
#[serial_test::serial]
fn multiword_loop_index_matrix_scale_matches_cpu() {
    use mxx_ir_core::IntExpr;
    let offset = BigInt::from(1u8) << 80usize;
    run_loop_index_matrix_scale_case(
        "direct-multiword-loop-index-matrix-scale",
        IntExpr::Add(Box::new(IntExpr::LoopIndex(0)), Box::new(IntExpr::constant(offset.clone()))),
        move |index| &offset + BigInt::from(index),
    );
}

#[test]
#[serial_test::serial]
fn exact_dynamic_division_with_negative_divisor_matches_cpu() {
    use mxx_ir_core::IntExpr;
    run_loop_index_matrix_scale_case(
        "direct-exact-dynamic-divide",
        IntExpr::Div(
            Box::new(IntExpr::Mul(Box::new(IntExpr::LoopIndex(0)), Box::new(IntExpr::constant(6)))),
            Box::new(IntExpr::constant(-3)),
        ),
        |index| -BigInt::from(2 * index),
    );
}

#[test]
#[serial_test::serial]
fn multiword_dynamic_floor_division_with_negative_divisor_matches_cpu() {
    use mxx_ir_core::IntExpr;
    use num_integer::Integer;
    let offset = BigInt::from(1u8) << 80usize;
    run_loop_index_matrix_scale_case(
        "direct-multiword-floor-divide",
        IntExpr::FloorDiv(
            Box::new(IntExpr::Add(
                Box::new(IntExpr::constant(offset.clone())),
                Box::new(IntExpr::LoopIndex(0)),
            )),
            Box::new(IntExpr::constant(-3)),
        ),
        move |index| (&offset + BigInt::from(index)).div_floor(&BigInt::from(-3)),
    );
}

#[test]
#[serial_test::serial]
fn dynamic_remainder_with_negative_divisor_matches_cpu() {
    use mxx_ir_core::IntExpr;
    use num_integer::Integer;
    run_loop_index_matrix_scale_case(
        "direct-negative-dynamic-remainder",
        IntExpr::Rem(
            Box::new(IntExpr::Sub(
                Box::new(IntExpr::constant(-1)),
                Box::new(IntExpr::LoopIndex(0)),
            )),
            Box::new(IntExpr::constant(-2)),
        ),
        |index| (-BigInt::from(index) - BigInt::from(1)).mod_floor(&BigInt::from(-2)),
    );
}

#[test]
#[serial_test::serial]
fn dynamic_log2_ceil_matches_cpu() {
    use mxx_ir_core::IntExpr;
    run_loop_index_matrix_scale_case(
        "direct-dynamic-log2-ceil",
        IntExpr::Log2Ceil(Box::new(IntExpr::Add(
            Box::new(IntExpr::LoopIndex(0)),
            Box::new(IntExpr::constant(1)),
        ))),
        |index| BigInt::from([0, 1, 2][index]),
    );
}

#[test]
#[serial_test::serial]
fn dynamic_exact_division_rejects_inexact_lane() {
    use mxx_ir_core::IntExpr;
    run_loop_index_matrix_scale_error_case(
        "direct-inexact-dynamic-divide",
        IntExpr::Div(
            Box::new(IntExpr::Add(Box::new(IntExpr::LoopIndex(0)), Box::new(IntExpr::constant(1)))),
            Box::new(IntExpr::constant(2)),
        ),
    );
}

#[test]
#[serial_test::serial]
fn dynamic_floor_division_rejects_actual_zero_divisor() {
    use mxx_ir_core::IntExpr;
    run_loop_index_matrix_scale_error_case(
        "direct-zero-dynamic-floor-divisor",
        IntExpr::FloorDiv(
            Box::new(IntExpr::constant(1)),
            Box::new(IntExpr::Sub(Box::new(IntExpr::LoopIndex(0)), Box::new(IntExpr::constant(1)))),
        ),
    );
}

#[test]
#[serial_test::serial]
fn dynamic_log2_ceil_rejects_actual_zero_argument() {
    use mxx_ir_core::IntExpr;
    run_loop_index_matrix_scale_error_case(
        "direct-zero-dynamic-log2-ceil",
        IntExpr::Log2Ceil(Box::new(IntExpr::LoopIndex(0))),
    );
}

#[test]
#[serial_test::serial]
fn unselected_invalid_integer_branch_does_not_fail() {
    use mxx_ir_core::IntExpr;
    run_loop_index_matrix_scale_case(
        "direct-lazy-integer-select",
        IntExpr::Select {
            selector: Box::new(IntExpr::LoopIndex(0)),
            branches: vec![
                IntExpr::constant(1),
                IntExpr::constant(2),
                IntExpr::constant(3),
                IntExpr::Div(Box::new(IntExpr::constant(1)), Box::new(IntExpr::constant(0))),
            ],
        },
        |index| BigInt::from(index + 1),
    );
}

fn run_loop_index_matrix_scale_error_case(label: &str, scalar: mxx_ir_core::IntExpr) {
    run_loop_index_matrix_scale_case_impl(label, scalar, None);
}

fn run_loop_index_matrix_scale_case(
    label: &str,
    scalar: mxx_ir_core::IntExpr,
    expected_scalar: impl Fn(usize) -> BigInt,
) {
    run_loop_index_matrix_scale_case_impl(label, scalar, Some(&expected_scalar));
}

fn run_loop_index_matrix_scale_case_impl(
    label: &str,
    scalar: mxx_ir_core::IntExpr,
    expected_scalar: Option<&dyn Fn(usize) -> BigInt>,
) {
    use mxx_ir_core::{
        Graph, GraphOutput, IntExpr, NodeHandle, WireType,
        graph::{SubgraphHandle, with_new_construction_scope},
        node::{LoopInputMode, NodeKind, ParallelLoop},
    };
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_parameters = GpuDCRTPolyParams::new(
        parameters.ring_dimension(),
        parameters.moduli().to_vec(),
        parameters.base_bits(),
        None,
    );
    let ring = Ring::from_crt_moduli(
        parameters.moduli().iter().copied().map(Into::into).collect(),
        parameters.ring_dimension(),
    );
    let matrix_type = WireType::Matrix(ring.matrix_type((2, 2)));
    let child = with_new_construction_scope(|scope| {
        let input = NodeHandle::new(
            NodeKind::Input {
                name: "body_matrix".into(),
                wire_type: matrix_type.clone(),
                artifact: None,
            },
            vec![],
            vec![matrix_type.clone()],
        )
        .output(0)
        .unwrap();
        let scaled = NodeHandle::new(
            NodeKind::MatrixScale { scalar },
            vec![input.clone()],
            vec![matrix_type.clone()],
        )
        .output(0)
        .unwrap();
        SubgraphHandle::new("direct-loop-scale-body", scope, vec![input], vec![scaled]).unwrap()
    });
    let source = NodeHandle::new(
        NodeKind::Input { name: "source".into(), wire_type: matrix_type.clone(), artifact: None },
        vec![],
        vec![matrix_type.clone()],
    );
    let family_type =
        WireType::IndexedFamily { element: Box::new(matrix_type), count: IntExpr::constant(3) };
    let scaled = NodeHandle::parallel_loop(
        child,
        vec![source.output(0).unwrap()],
        vec![family_type],
        ParallelLoop {
            count: IntExpr::constant(3),
            minimum_count: 0,
            index_slot: 0,
            bindings: vec![],
            input_modes: vec![LoopInputMode::Broadcast],
        },
    );
    let graph = Graph::freeze(
        label,
        vec![],
        BTreeMap::from([(
            "scaled".into(),
            GraphOutput { value: scaled.output(0).unwrap(), availability: None },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let graph = mxx_ir_core::validate(
        &graph,
        &ParamEnv::default(),
        mxx_backends::openfhe_guard::gen_modulus_and_warmup,
    )
    .unwrap();
    let concrete_ring = RingRef::new(RingExpr::Explicit {
        crt_moduli: parameters.moduli().iter().copied().map(Into::into).collect(),
        ring_dimension: parameters.ring_dimension(),
    })
    .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
    .unwrap();
    let wire =
        ConcreteWireType::Matrix(ConcreteMatrixType { ring: concrete_ring, rows: 2, columns: 2 });
    let bind = |matrix: &DCRTPolyMatrix| {
        BTreeMap::from([(
            "source".to_owned(),
            RuntimeValue::gpu_matrix(
                wire.clone(),
                Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, matrix)),
            )
            .unwrap(),
        )])
    };
    let first = cpu_input(&parameters);
    let second = first.clone() + &first;
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    let mut plan = runtime.plan(graph, &bind(&first)).unwrap();
    let replays = if expected_scalar.is_none() { vec![first] } else { vec![first, second] };
    for (replay, matrix) in replays.into_iter().enumerate() {
        let result = runtime.execute(
            &mut plan,
            bind(&matrix),
            &mut MemoryArtifactStore::default(),
            [replay as u8; 32],
        );
        if expected_scalar.is_none() {
            let error = result.err().expect("invalid dynamic integer expression must fail");
            assert!(error.to_string().contains("outputs suppressed"), "{error}");
            continue;
        }
        let result = result.unwrap();
        for index in 0..3 {
            let actual = runtime.download_matrix_member(&result.outputs["scaled"], index).unwrap();
            let modulus = BigInt::from(parameters.modulus().as_ref().clone());
            let scalar = expected_scalar.expect("successful run has a CPU oracle")(index);
            let residue = ((scalar % &modulus) + &modulus) % &modulus;
            let multiplier =
                DCRTPoly::from_biguint_to_constant(&parameters, residue.to_biguint().unwrap());
            let expected = matrix.multiply_poly_out_of_place(&multiplier);
            assert_same_matrix_values(
                &format!("replay {replay}, lane {index}"),
                &actual,
                &expected,
            );
        }
    }
}

#[test]
#[serial_test::serial]
fn modulus_switch_matches_cpu() {
    run_crt_case(
        "direct-modulus-switch",
        true,
        |a, destination| a.modulus_switch(destination),
        |a, destination| a.modulus_switch(destination),
    );
}

#[test]
#[serial_test::serial]
fn modulus_reduce_matches_cpu() {
    run_crt_case(
        "direct-modulus-reduce",
        true,
        |a, destination| a.reduce_modulus(destination),
        |a, destination| a.reduce_modulus(destination),
    );
}

#[test]
#[serial_test::serial]
fn centered_rebase_matches_cpu() {
    run_crt_case(
        "direct-centered-rebase",
        true,
        |a, destination| a.centered_rebase(destination),
        |a, destination| a.centered_rebase(destination).unwrap(),
    );
}

#[test]
#[serial_test::serial]
fn centered_round_divide_matches_cpu() {
    run_crt_case(
        "direct-centered-round-divide",
        true,
        |a, _| a.centered_round_divide(3),
        |a, _| a.centered_round_divide(&BigUint::from(3u8)).unwrap(),
    );
}

#[test]
#[serial_test::serial]
fn block_mod_switch_matches_cpu() {
    run_crt_case(
        "direct-block-mod-switch",
        true,
        |a, destination| a.block_mod_switch(destination, 3),
        |a, destination| a.block_mod_switch(destination, &BigUint::from(3u8)).unwrap(),
    );
}

#[test]
#[serial_test::serial]
fn block_mod_switch_nonprefix_subset_matches_cpu() {
    run_crt_case_with_prime(
        "direct-block-mod-switch-nonprefix",
        true,
        1,
        |a, destination| a.block_mod_switch(destination, 3),
        |a, destination| a.block_mod_switch(destination, &BigUint::from(3u8)).unwrap(),
    );
}

#[test]
#[serial_test::serial]
fn rns_mod_up_matches_cpu() {
    run_crt_case(
        "direct-rns-mod-up",
        false,
        |a, destination| a.rns_mod_up(destination, 1, true),
        |a, destination| a.rns_mod_up(destination, 1, true).unwrap(),
    );
}

#[test]
#[serial_test::serial]
fn rns_mod_down_matches_cpu() {
    run_crt_case(
        "direct-rns-mod-down",
        true,
        |a, destination| a.rns_mod_down(destination, 3),
        |a, destination| a.rns_mod_down(destination, 3).unwrap(),
    );
}

#[test]
#[serial_test::serial]
fn rns_mod_down_nonprefix_wide_plaintext_matches_cpu() {
    let full = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let wide_t = BigInt::from(3u8) + (BigInt::from(full.modulus().as_ref().clone()) << 80usize);
    assert!(wide_t > BigInt::from(u64::MAX));
    run_crt_case_with_prime(
        "direct-rns-mod-down-nonprefix-wide-t",
        true,
        1,
        move |a, destination| a.rns_mod_down(destination, wide_t),
        |a, destination| a.rns_mod_down(destination, 3).unwrap(),
    );
}

#[test]
#[serial_test::serial]
fn crt_recompose_matches_independent_cpu_formula() {
    let full = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let moduli = full.moduli();
    let left_params = full.select_modulus(&BigUint::from(moduli[0])).unwrap();
    let right_params = full.select_modulus(&BigUint::from(moduli[1])).unwrap();
    let gpu_full =
        GpuDCRTPolyParams::new(full.ring_dimension(), moduli.to_vec(), full.base_bits(), None);
    let related = |parameters: &DCRTPolyParams| {
        GpuDCRTPolyParams::new_with_gpu(
            parameters.ring_dimension(),
            parameters.moduli().to_vec(),
            parameters.base_bits(),
            gpu_full.gpu_ids().to_vec(),
            Some(1),
            Some(&gpu_full),
            None,
        )
    };
    let gpu_left = related(&left_params);
    let gpu_right = related(&right_params);
    let make_ring = |parameters: &DCRTPolyParams| {
        Ring::from_crt_moduli(
            parameters.moduli().iter().copied().map(Into::into).collect(),
            parameters.ring_dimension(),
        )
    };
    let output = Mat::crt_recompose(
        vec![
            make_ring(&left_params).input("left", (1, 1)),
            make_ring(&right_params).input("right", (1, 1)),
        ],
        vec![5.into(), 7.into()],
        vec![21.into(), 15.into()],
        &make_ring(&full),
    );
    let graph = DslContext::new("direct-crt-recompose")
        .output("result", output)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let make_coefficients = |modulus: u64| {
        [
            0,
            modulus / 7,
            modulus / 4,
            modulus / 2 - 1,
            modulus / 2 + 1,
            3 * modulus / 4,
            modulus - 2,
            modulus - 1,
        ]
        .map(BigUint::from)
    };
    let left_coefficients = make_coefficients(moduli[0]);
    let right_coefficients = make_coefficients(moduli[1]);
    let left = DCRTPolyMatrix::from_poly_vec_row(
        &left_params,
        vec![DCRTPoly::from_biguints(&left_params, &left_coefficients)],
    );
    let right = DCRTPolyMatrix::from_poly_vec_row(
        &right_params,
        vec![DCRTPoly::from_biguints(&right_params, &right_coefficients)],
    );
    let expected_coefficients = left_coefficients
        .iter()
        .zip(&right_coefficients)
        .map(|(left, right)| {
            let left = left.to_u64_digits().first().copied().unwrap_or(0);
            let right = right.to_u64_digits().first().copied().unwrap_or(0);
            let rounded_left = ((5 * left + moduli[0] / 2) / moduli[0]) % 5;
            let rounded_right = ((7 * right + moduli[1] / 2) / moduli[1]) % 7;
            BigUint::from(21 * rounded_left + 15 * rounded_right)
        })
        .collect::<Vec<_>>();
    let expected = DCRTPolyMatrix::from_poly_vec_row(
        &full,
        vec![DCRTPoly::from_biguints(&full, &expected_coefficients)],
    );
    let wire = |parameters: &DCRTPolyParams| {
        let ring = RingRef::new(RingExpr::Explicit {
            crt_moduli: parameters.moduli().iter().copied().map(Into::into).collect(),
            ring_dimension: parameters.ring_dimension(),
        })
        .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
        ConcreteWireType::Matrix(ConcreteMatrixType { ring, rows: 1, columns: 1 })
    };
    let inputs = BTreeMap::from([
        (
            "left".to_owned(),
            RuntimeValue::gpu_matrix(
                wire(&left_params),
                Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_left, &left)),
            )
            .unwrap(),
        ),
        (
            "right".to_owned(),
            RuntimeValue::gpu_matrix(
                wire(&right_params),
                Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_right, &right)),
            )
            .unwrap(),
        ),
    ]);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_full, gpu_left, gpu_right])).unwrap();
    let mut plan = runtime.plan(graph, &inputs).unwrap();
    let result =
        runtime.execute(&mut plan, inputs, &mut MemoryArtifactStore::default(), [13; 32]).unwrap();
    let actual = runtime.download_matrix(&result.outputs["result"]).unwrap();
    assert_same_matrix_values("CRT recompose", &actual, &expected);
}

#[test]
#[serial_test::serial]
fn hash_sample_matches_cpu_for_fixed_key_and_tag() {
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_parameters = GpuDCRTPolyParams::new(
        parameters.ring_dimension(),
        parameters.moduli().to_vec(),
        parameters.base_bits(),
        None,
    );
    let ring = Ring::from_crt_moduli(
        parameters.moduli().iter().copied().map(Into::into).collect(),
        parameters.ring_dimension(),
    );
    let key = [0x5au8; 32];
    let tag = b"direct-hash-sample:";
    let graph = DslContext::new("direct-hash-sample")
        .output(
            "result",
            ring.hash_matrix(ring.bytes_input("key", key.len()), tag.as_slice(), (2, 2)),
        )
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let expected = DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
        &parameters,
        key,
        tag,
        2,
        2,
        DistType::FinRingDist,
    );
    let inputs = BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(Arc::from(key)))]);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters])).unwrap();
    let mut plan = runtime.plan(graph, &inputs).unwrap();
    let result =
        runtime.execute(&mut plan, inputs, &mut MemoryArtifactStore::default(), [17; 32]).unwrap();
    let actual = runtime.download_matrix(&result.outputs["result"]).unwrap();
    assert_same_matrix_values("hash sample", &actual, &expected);
}
