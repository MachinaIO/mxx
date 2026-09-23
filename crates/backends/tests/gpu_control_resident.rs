#![cfg(feature = "gpu")]

//! Regression graph for the resident GPU control path.
//!
//! The graph intentionally keeps all scalar work in one small parallel body.  In
//! particular, the family index is a loop-dependent value, so lowering it must
//! retain `FamilyGetDynamic` in the resident control vocabulary instead of
//! creating a host-side family protocol operation.

use mxx_backends::{
    GpuExecutionResult, GpuRuntime, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly_gpu::gpu_backend,
    gpu_column_policy::is_resident_control_operation,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, GpuSignedValues},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_dsl::{Bool, DslContext, Family, Int, Ring, iterate, parallel, select};
use mxx_ir_core::{
    ParamEnv,
    node::{IntBinaryOp, IntCompareOp, NodeKind},
    ring::{RingExpr, RingRef},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use num_bigint::BigInt;
use std::{collections::BTreeMap, sync::Arc};

fn test_ring(parameters: &DCRTPolyParams) -> Ring {
    Ring::from_crt_moduli(
        parameters.moduli().iter().copied().map(Into::into).collect(),
        parameters.ring_dimension(),
    )
}

fn matrix_wire(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> ConcreteWireType {
    let ring = RingRef::new(RingExpr::Explicit {
        crt_moduli: parameters.moduli().iter().copied().map(Into::into).collect(),
        ring_dimension: parameters.ring_dimension(),
    })
    .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
    .expect("resolve test CRT basis");
    ConcreteWireType::Matrix(ConcreteMatrixType { ring, rows, columns })
}

fn gpu_matrix_value(parameters: &DCRTPolyParams, matrix: GpuDCRTPolyMatrix) -> RuntimeValue {
    let (rows, columns) = matrix.size();
    RuntimeValue::gpu_matrix(matrix_wire(parameters, rows, columns), Arc::new(matrix))
        .expect("bind exact GPU matrix owner")
}

fn gpu_integer_family(parameters: &GpuDCRTPolyParams, values: &[BigInt]) -> RuntimeValue {
    let owner = Arc::new(
        GpuSignedValues::from_bigints(parameters, 0, values).expect("upload signed integer family"),
    );
    RuntimeValue::gpu_signed_family(
        ConcreteWireType::IndexedFamily {
            element: Box::new(ConcreteWireType::Int),
            count: values.len(),
        },
        owner,
    )
    .expect("bind exact signed integer family")
}

fn gpu_boolean_family(parameters: &GpuDCRTPolyParams, values: &[BigInt]) -> RuntimeValue {
    let bits = values
        .iter()
        .map(|value| {
            if value == &BigInt::from(0) {
                0
            } else if value == &BigInt::from(1) {
                1
            } else {
                2
            }
        })
        .collect::<Vec<_>>();
    let owner =
        Arc::new(GpuSignedValues::upload(parameters, 0, &bits).expect("upload boolean family"));
    RuntimeValue::gpu_signed_family(
        ConcreteWireType::IndexedFamily {
            element: Box::new(ConcreteWireType::Bool),
            count: values.len(),
        },
        owner,
    )
    .expect("bind exact boolean family")
}

fn control_graph(moduli: &[u64], ring_dimension: u32) -> mxx_dsl::BuiltGraph {
    control_graph_with_count(moduli, ring_dimension, 4)
}

#[test]
#[serial_test::serial]
fn matrix_partial_slice_concat_replays_with_fresh_owners() {
    use mxx_backends::poly::dcrt::gpu::GpuDCRTPoly;
    use mxx_ir_core::node::{ConcatAxis, IndexRange};
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(8, moduli, 4, None);
    let context = DslContext::new("partial-matrix-replay");
    let ring = test_ring(&parameters);
    let input = ring.input("matrix", (4, 4));
    let range =
        |start: usize, end: usize| Some(IndexRange { start: start.into(), end: end.into() });
    let left = input.clone().slice(range(0, 2), range(0, 2));
    let right = input.slice(range(0, 2), range(2, 4));
    let combined = mxx_dsl::Mat::concat(ConcatAxis::Columns, vec![left, right]);
    let graph = context
        .output("matrix", combined.clone() + combined)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let make = |scalar: u64| {
        GpuDCRTPolyMatrix::identity(
            &gpu_parameters,
            4,
            Some(GpuDCRTPoly::from_usize_to_constant(&gpu_parameters, scalar as usize)),
        )
    };
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    let mut plan = runtime
        .plan(graph, &BTreeMap::from([("matrix".into(), gpu_matrix_value(&parameters, make(1)))]))
        .unwrap();
    for scalar in [3, 7] {
        let source = make(scalar);
        let expected = make(2 * scalar).slice(0, 2, 0, 4).to_cpu_matrix();
        let result = runtime
            .execute(
                &mut plan,
                BTreeMap::from([("matrix".into(), gpu_matrix_value(&parameters, source))]),
                &mut MemoryArtifactStore::default(),
                [scalar as u8; 32],
            )
            .unwrap();
        let actual = runtime.download_matrix(&result.outputs["matrix"]).unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
#[serial_test::serial]
fn typed_real_boundary_and_trapdoor_public_output_execute() {
    use mxx_ir_core::{Graph, GraphOutput, NodeHandle, RealExpr, WireType, node::RealBinaryOp};
    let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(8, moduli, 4, None);
    let ring = test_ring(&parameters);
    let trapdoor = ring.gadget_trapdoor(1, 16, 10);
    let integer =
        NodeHandle::new(NodeKind::ConstantInt(144.into()), vec![], vec![WireType::ConstantInt]);
    let convert = NodeHandle::new(
        NodeKind::IntToReal,
        vec![integer.output(0).unwrap()],
        vec![WireType::Real],
    );
    let square_root =
        NodeHandle::new(NodeKind::RealSqrt, vec![convert.output(0).unwrap()], vec![WireType::Real]);
    let constant = NodeHandle::new(
        NodeKind::ConstantReal(RealExpr::from_integer(2)),
        vec![],
        vec![WireType::ConstantReal],
    );
    let result = NodeHandle::new(
        NodeKind::RealBinary(RealBinaryOp::Subtract),
        vec![square_root.output(0).unwrap(), constant.output(0).unwrap()],
        vec![WireType::Real],
    );
    let graph = Graph::freeze(
        "typed-real-native-public-output",
        vec![],
        BTreeMap::from([
            ("real".into(), GraphOutput { value: result.output(0).unwrap(), availability: None }),
            (
                "public".into(),
                GraphOutput {
                    value: trapdoor.public_matrix().value_handle().clone(),
                    availability: None,
                },
            ),
            (
                "trapdoor".into(),
                GraphOutput { value: trapdoor.value_handle().clone(), availability: None },
            ),
        ]),
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
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    let mut plan = runtime.plan(graph, &BTreeMap::new()).unwrap();
    for seed in [1, 2] {
        let result = runtime
            .execute(&mut plan, BTreeMap::new(), &mut MemoryArtifactStore::default(), [seed; 32])
            .unwrap();
        assert!(matches!(result.outputs["real"], RuntimeValue::Real(value) if value == 10.0));
        assert!(matches!(result.outputs["public"], RuntimeValue::Matrix(_)));
        assert!(matches!(result.outputs["trapdoor"], RuntimeValue::Resident(_)));
    }
}

#[test]
#[serial_test::serial]
fn resident_pack_invalid_bit_and_out_of_range_suppress_publication() {
    use mxx_ir_core::{Graph, GraphOutput, NodeHandle, WireType};
    let parameters = DCRTPolyParams::new(8, 4, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(8, moduli, 4, None);
    let modulus = BigInt::from(parameters.modulus().as_ref().clone());
    let width = modulus.bits() as usize + 128;
    let ring = test_ring(&parameters);
    let family =
        WireType::IndexedFamily { element: Box::new(WireType::Bool), count: (width * 8).into() };
    let input = NodeHandle::new(
        NodeKind::Input { name: "bits".into(), wire_type: family.clone(), artifact: None },
        vec![],
        vec![family],
    );
    let ty = ring.matrix_type((1, 1));
    let pack = NodeHandle::new(
        NodeKind::PackPolynomialCoefficients {
            matrix_type: ty.clone(),
            coefficient_bits: width.into(),
        },
        vec![input.output(0).unwrap()],
        vec![WireType::Matrix(ty)],
    );
    let graph = Graph::freeze(
        "invalid-native-pack",
        vec![],
        BTreeMap::from([(
            "packed".into(),
            GraphOutput { value: pack.output(0).unwrap(), availability: None },
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
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    let valid = vec![BigInt::from(0); width * 8];
    let mut plan = runtime
        .plan(
            graph,
            &BTreeMap::from([("bits".into(), gpu_boolean_family(&gpu_parameters, &valid))]),
        )
        .unwrap();
    runtime
        .execute(
            &mut plan,
            BTreeMap::from([("bits".into(), gpu_boolean_family(&gpu_parameters, &valid))]),
            &mut MemoryArtifactStore::default(),
            [0; 32],
        )
        .expect("leading zero bits are valid regardless of pack width");
    for invalid_case in 0..3 {
        let mut bits = valid.clone();
        if invalid_case == 0 {
            bits[0] = BigInt::from(2);
        } else if invalid_case == 2 {
            bits[width - 1] = BigInt::from(1);
        } else {
            for (bit, value) in bits.iter_mut().take(width).enumerate() {
                *value = (&modulus >> bit) & BigInt::from(1);
            }
        }
        let error = runtime
            .execute(
                &mut plan,
                BTreeMap::from([("bits".into(), gpu_boolean_family(&gpu_parameters, &bits))]),
                &mut MemoryArtifactStore::default(),
                [1; 32],
            )
            .err()
            .expect("invalid pack must not publish");
        assert!(error.to_string().contains("outputs suppressed"), "{error}");
    }
}

#[test]
#[serial_test::serial]
fn resident_multiword_polynomial_primitives_rebind_and_decode() {
    let parameters = DCRTPolyParams::new(8, 4, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(8, moduli, 4, None);
    let modulus = BigInt::from(parameters.modulus().as_ref().clone());
    let context = DslContext::new("resident-wide-polynomial-primitives");
    let ring = test_ring(&parameters);
    let inputs = context.int_family_input("values", 8);
    let polynomial = ring.from_coefficients(&inputs);
    let coefficient = polynomial.clone().extract_coefficient(1);
    let sum = coefficient.clone() + Int::constant(BigInt::from(1u64) << 80usize);
    let lifted = sum.clone().lift_to_constant_polynomial(ring.matrix_type((1, 1)));
    let decoded = polynomial.clone().threshold_decode_ints(2, 3);
    let plaintext_modulus = (BigInt::from(1) << 128usize) + BigInt::from(3);
    let wide_decoded = polynomial.clone().threshold_decode_ints(plaintext_modulus.clone(), 3);
    let wide_bools = polynomial.clone().threshold_decode_bools(plaintext_modulus.clone(), 3);
    let decoded_bools = polynomial.clone().threshold_decode_bools(2, 8);
    let coefficient_bits = modulus.bits() as usize;
    let zero = Int::constant(0).bit(0).unwrap();
    let bits = decoded_bools
        .into_iter()
        .flat_map(|bit| {
            std::iter::once(bit).chain(std::iter::repeat_n(zero.clone(), coefficient_bits - 1))
        })
        .collect();
    let packed =
        ring.pack_polynomial_coefficients(Family::<Bool>::pack(bits).unwrap(), coefficient_bits);
    let graph = context
        .output("values", polynomial.coefficients())
        .unwrap()
        .output("evaluations", ring.from_evaluations(&inputs).evaluations())
        .unwrap()
        .output("coefficient", coefficient)
        .unwrap()
        .output("sum", sum)
        .unwrap()
        .output("lifted", lifted.coefficients())
        .unwrap()
        .output("decoded", Family::<Int>::pack(decoded).unwrap())
        .unwrap()
        .output("wide_decoded", Family::<Int>::pack(wide_decoded).unwrap())
        .unwrap()
        .output("wide_bools", Family::<Bool>::pack(wide_bools).unwrap())
        .unwrap()
        .output("packed", packed.coefficients())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    let make_values = |shift: usize| {
        (0..8)
            .map(|index| {
                let magnitude = (&modulus << (64usize * shift)) +
                    (&modulus * BigInt::from(index + shift) / BigInt::from(8));
                if index % 2 == 0 { magnitude } else { -magnitude }
            })
            .collect::<Vec<_>>()
    };
    let first = make_values(1);
    let mut plan = runtime
        .plan(
            graph,
            &BTreeMap::from([("values".into(), gpu_integer_family(&gpu_parameters, &first))]),
        )
        .unwrap();
    for shift in [1, 3] {
        let values = make_values(shift);
        let result = runtime
            .execute(
                &mut plan,
                BTreeMap::from([("values".into(), gpu_integer_family(&gpu_parameters, &values))]),
                &mut MemoryArtifactStore::default(),
                [shift as u8; 32],
            )
            .unwrap();
        let canonical: Vec<_> =
            values.iter().map(|value| ((value % &modulus) + &modulus) % &modulus).collect();
        let read = |name: &str| runtime.download_integer_family(&result.outputs[name]).unwrap();
        assert_eq!(read("values"), canonical);
        assert_eq!(read("evaluations"), canonical);
        assert_eq!(read("coefficient"), vec![canonical[1].clone()]);
        let sum = &canonical[1] + (BigInt::from(1u64) << 80usize);
        assert_eq!(read("sum"), vec![sum.clone()]);
        let mut lifted = vec![BigInt::from(0); 8];
        lifted[0] = sum % &modulus;
        assert_eq!(read("lifted"), lifted);
        let decoded: Vec<_> =
            canonical.iter().map(|value| ((value * 4 + &modulus) / (&modulus * 2)) % 2).collect();
        assert_eq!(read("decoded"), decoded[..3]);
        let wide = mxx_backends::backend::poly::threshold_decode_coefficients(
            canonical.iter().map(|value| value.to_biguint().unwrap()).collect(),
            &modulus,
            &plaintext_modulus,
            3,
        );
        assert_eq!(read("wide_decoded"), wide);
        assert_eq!(
            read("wide_bools"),
            wide.iter()
                .map(|value| BigInt::from(u8::from(value != &BigInt::from(0))))
                .collect::<Vec<_>>()
        );
        assert_eq!(read("packed"), decoded);
    }
}

fn control_graph_with_count(
    moduli: &[u64],
    ring_dimension: u32,
    count: usize,
) -> mxx_dsl::BuiltGraph {
    let context = DslContext::new("gpu-resident-control-regression");
    let ring =
        Ring::from_crt_moduli(moduli.iter().copied().map(Into::into).collect(), ring_dimension);
    let values = context.int_family_input("values", count.max(6));
    let packed = Family::<Int>::pack(vec![Int::constant(-3), Int::constant(4), Int::constant(7)])
        .expect("pack integer selector family");

    let result = parallel(count, |index| {
        let family_index = index.clone() % 3;
        let left = values.at(index.clone());
        let right = packed.at(family_index.clone());

        let sum = left.clone() + right.clone();
        let difference = sum.clone() - index.clone();
        let product = difference.clone() * -2;
        let quotient = product.clone() / 3;
        let remainder = product.clone() % 3;

        // Keep both signed comparison forms live through Bool-to-int and Select.
        let negative = quotient.clone().less(-1);
        let bounded = remainder.clone().less_equal(1);
        let compare_bit: Bool = negative & bounded;
        let compare_int = compare_bit.to_int();
        let selector = index % 2;
        select(selector, vec![quotient + compare_int.clone(), remainder + compare_int])
    })
    .expect("construct resident control parallel body");

    // A concrete matrix parameter keeps this graph eligible for the public GPU
    // planning API. It is otherwise independent of the scalar control result.
    let anchor = ring.input("anchor", (1, 1));
    context
        .output("result", result)
        .expect("result output")
        .output("anchor", anchor)
        .expect("anchor output")
        .build()
        .expect("build resident control graph")
}

fn sequential_control_graph(
    moduli: &[u64],
    ring_dimension: u32,
    count: usize,
) -> mxx_dsl::BuiltGraph {
    let context = DslContext::new("gpu-resident-sequential-control-regression");
    let ring =
        Ring::from_crt_moduli(moduli.iter().copied().map(Into::into).collect(), ring_dimension);
    let result = iterate(count, Int::constant(0), |index, state| Ok(state + index + 1))
        .expect("construct sequential resident body");
    let anchor = ring.input("anchor", (1, 1));
    context
        .output("result", result)
        .expect("result output")
        .output("anchor", anchor)
        .expect("anchor output")
        .build()
        .expect("build sequential resident graph")
}

fn all_nodes(graph: &mxx_dsl::BuiltGraph) -> Vec<&mxx_ir_core::NodeHandle> {
    graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect()
}

#[test]
fn resident_control_regression_graph_covers_scalar_and_family_operations() {
    let graph = control_graph(&[17], 8);
    graph
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .expect("validate resident control graph");
    let nodes = all_nodes(&graph);

    assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_))));
    assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyPack { .. })));
    assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic)));
    assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::Select { .. })));
    assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::BoolToInt)));
    assert!(
        nodes.iter().all(|node| {
            matches!(node.kind(), NodeKind::Input { .. }) ||
                is_resident_control_operation(node.kind())
        }),
        "control graph contains a non-resident/non-control operation"
    );

    for operation in [
        IntBinaryOp::Add,
        IntBinaryOp::Subtract,
        IntBinaryOp::Multiply,
        IntBinaryOp::Divide,
        IntBinaryOp::Remainder,
    ] {
        assert!(
            nodes.iter().any(
                |node| matches!(node.kind(), NodeKind::IntBinary(actual) if *actual == operation)
            ),
            "missing signed integer operation {operation:?}"
        );
    }
    for operation in [IntCompareOp::Less, IntCompareOp::LessEqual] {
        assert!(
            nodes.iter().any(
                |node| matches!(node.kind(), NodeKind::IntCompare(actual) if *actual == operation)
            ),
            "missing signed integer comparison {operation:?}"
        );
    }
}

fn expected_values(values: &[i64]) -> Vec<BigInt> {
    let packed = [-3_i64, 4, 7];
    (0..values.len())
        .map(|index| {
            let left = values[index];
            let right = packed[index % packed.len()];
            let product = (left + right - index as i64) * -2;
            let quotient = product.div_euclid(3);
            let remainder = product.rem_euclid(3);
            let bit = i64::from(quotient < -1 && remainder <= 1);
            if index % 2 == 0 { quotient + bit } else { remainder + bit }
        })
        .map(BigInt::from)
        .collect()
}

#[test]
#[serial_test::serial]
fn resident_control_parallel_tail_replays_bounded_wave() {
    unsafe { std::env::set_var("MXX_GPU_MAX_PARALLEL_INSTANCES", "3") };
    let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits(), None);
    let graph = control_graph_with_count(parameters.moduli(), parameters.ring_dimension(), 5)
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .expect("validate tail resident graph");
    let anchor = gpu_anchor(&parameters, &gpu_parameters);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    let input_values = [1, 2, 3, 4, 5, 6];
    let inputs = gpu_control_inputs(&gpu_parameters, &anchor, &input_values);
    let mut plan = runtime.plan(graph, &inputs).expect("plan tail resident graph");
    let result = runtime.execute(&mut plan, inputs, &mut MemoryArtifactStore::default(), [4; 32]);
    unsafe { std::env::remove_var("MXX_GPU_MAX_PARALLEL_INSTANCES") };
    let result = result.expect("execute bounded resident tail");
    assert_eq!(resident_output(&mut runtime, result), expected_values(&input_values[..5]));
}

#[test]
#[serial_test::serial]
fn resident_control_sequential_carried_counts_publish_final_state() {
    let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits(), None);
    let anchor = gpu_anchor(&parameters, &gpu_parameters);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    for (nonce, count) in [0usize, 1, 2, 4].into_iter().enumerate() {
        let graph =
            sequential_control_graph(parameters.moduli(), parameters.ring_dimension(), count)
                .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
                .expect("validate sequential resident graph");
        let inputs = BTreeMap::from([("anchor".to_owned(), anchor.clone())]);
        let mut plan = runtime.plan(graph, &inputs).expect("plan sequential resident graph");
        let result = runtime
            .execute(&mut plan, inputs, &mut MemoryArtifactStore::default(), [10 + nonce as u8; 32])
            .expect("execute sequential resident graph");
        assert!(matches!(result.outputs["result"], RuntimeValue::Resident(_)));
        let actual = runtime
            .download_integer_family(&result.outputs["result"])
            .expect("read sequential result");
        assert_eq!(actual, [BigInt::from((count * (count + 1) / 2) as i64)]);
    }
}

fn gpu_control_inputs(
    parameters: &GpuDCRTPolyParams,
    anchor: &RuntimeValue,
    values: &[i64],
) -> BTreeMap<String, RuntimeValue> {
    let integer_values = values.iter().copied().map(BigInt::from).collect::<Vec<_>>();
    BTreeMap::from([
        ("values".to_owned(), gpu_integer_family(parameters, &integer_values)),
        ("anchor".to_owned(), anchor.clone()),
    ])
}

fn gpu_anchor(parameters: &DCRTPolyParams, gpu_parameters: &GpuDCRTPolyParams) -> RuntimeValue {
    let matrix = DCRTPolyMatrix::from_poly_vec_row(
        parameters,
        vec![DCRTPoly::from_usize_to_constant(parameters, 1)],
    );
    gpu_matrix_value(parameters, GpuDCRTPolyMatrix::from_cpu_matrix(gpu_parameters, &matrix))
}

#[test]
#[serial_test::serial]
fn resident_control_replay_uses_new_family_values_and_signed_semantics() {
    let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits(), None);
    let graph = control_graph(parameters.moduli(), parameters.ring_dimension())
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .expect("validate resident control graph");
    let backend = gpu_backend([gpu_parameters.clone()]);
    let anchor = gpu_anchor(&parameters, &gpu_parameters);
    let mut runtime = GpuRuntime::new(backend).expect("construct resident control runtime");
    let planning_values = [1, 2, 3, 4, 5, 6];
    let planning_inputs = gpu_control_inputs(&gpu_parameters, &anchor, &planning_values);
    let mut plan = runtime.plan(graph, &planning_inputs).expect("prepare resident control plan");

    assert!(plan.compiled_region_count() > 0, "control graph needs a compiled GPU region");

    let first_values = [1, 2, 3, 4, 5, 6];
    let first_inputs = gpu_control_inputs(&gpu_parameters, &anchor, &first_values);
    let first = runtime
        .execute(&mut plan, first_inputs, &mut MemoryArtifactStore::default(), [0; 32])
        .expect("execute first resident control replay");
    assert_eq!(resident_output(&mut runtime, first), expected_values(&first_values[..4]));

    let second_values = [-9, 8, -7, 6, -5, 4];
    let second_inputs = gpu_control_inputs(&gpu_parameters, &anchor, &second_values);
    let second = runtime
        .execute(&mut plan, second_inputs, &mut MemoryArtifactStore::default(), [1; 32])
        .expect("execute second resident control replay");
    assert_eq!(resident_output(&mut runtime, second), expected_values(&second_values[..4]));
    assert!(plan.compiled_launch_count() >= 2, "both control replays must launch GPU work");
}

fn resident_output(runtime: &GpuRuntime, result: GpuExecutionResult) -> Vec<BigInt> {
    assert!(matches!(result.outputs["result"], RuntimeValue::Resident(_)));
    runtime
        .download_integer_family(&result.outputs["result"])
        .expect("gather resident control result")
}

#[test]
#[serial_test::serial]
fn test_gpu_resident_invalid_index_suppresses_outputs() {
    let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
    let (moduli, _, _) = parameters.to_crt();
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits(), None);
    let context = DslContext::new("resident-invalid-index");
    let values = context.int_family_input("values", 6);
    let packed = Family::<Int>::pack(vec![Int::constant(10), Int::constant(20)]).unwrap();
    let result = parallel(4, |index| Ok(packed.at(values.at(index)))).unwrap();
    let anchor_wire = test_ring(&parameters).input("anchor", (1, 1));
    let graph = context
        .output("result", result)
        .unwrap()
        .output("anchor", anchor_wire)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let anchor = gpu_anchor(&parameters, &gpu_parameters);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters.clone()])).unwrap();
    let planning_inputs = gpu_control_inputs(&gpu_parameters, &anchor, &[0, 1, 0, 1, 0, 0]);
    let mut plan = runtime.plan(graph, &planning_inputs).unwrap();
    let invalid_inputs = gpu_control_inputs(&gpu_parameters, &anchor, &[0, 2, 0, 1, 0, 0]);
    let result =
        runtime.execute(&mut plan, invalid_inputs, &mut MemoryArtifactStore::default(), [2; 32]);
    let error = match result {
        Ok(_) => panic!("invalid device index published an ExecutionResult"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("index"), "unexpected execution error: {error}");
    let valid_inputs = gpu_control_inputs(&gpu_parameters, &anchor, &[1, 0, 1, 0, 0, 0]);
    let recovered = runtime
        .execute(&mut plan, valid_inputs, &mut MemoryArtifactStore::default(), [3; 32])
        .unwrap();
    assert_eq!(resident_output(&mut runtime, recovered), [20, 10, 20, 10].map(BigInt::from));
}
