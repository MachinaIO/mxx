//! Emit the generated nonzero-target-error consumption relation used by Stage A.
use mxx_ir_core::{
    Graph, GraphOutput, NodeHandle, ParamEnv, RealExpr, WireType,
    graph::CompileParameter,
    lean::{ExportOptions, export},
    node::{MatrixBinaryOp, NodeKind},
    types::MatrixType,
    validate,
};
use std::{collections::BTreeMap, env, fs};

fn input(name: &str, ty: MatrixType) -> mxx_ir_core::ValueHandle {
    let wire_type = WireType::Matrix(ty);
    NodeHandle::new(
        NodeKind::Input { name: name.into(), wire_type: wire_type.clone(), artifact: None },
        vec![],
        vec![wire_type],
    )
    .output(0)
    .unwrap()
}

fn main() {
    let output = env::args().nth(1).expect("output Lean path");
    let b_ty = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 1.into(),
        columns: 3.into(),
    };
    let one_ty = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 1.into(),
        columns: 1.into(),
    };
    let k_ty = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 3.into(),
        columns: 1.into(),
    };
    let left = input("left", one_ty.clone());
    let public = input("public", b_ty.clone());
    let trapdoor_ty = WireType::Trapdoor {
        matrix: b_ty.clone(),
        sigma: RealExpr::from(1),
        gadget_base: 2.into(),
        digit_count: 1.into(),
        preimage_max_coefficient_bound: 4.into(),
    };
    let trapdoor = NodeHandle::new(
        NodeKind::Input { name: "trapdoor".into(), wire_type: trapdoor_ty.clone(), artifact: None },
        vec![],
        vec![trapdoor_ty],
    )
    .output(0)
    .unwrap();
    let ideal = input("ideal", one_ty.clone());
    let target_error = input("target-error", one_ty.clone());
    let source_error = input("source-error", b_ty.clone());
    let target = NodeHandle::new(
        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
        vec![ideal.clone(), target_error],
        vec![WireType::Matrix(one_ty.clone())],
    )
    .output(0)
    .unwrap();
    let k = NodeHandle::new(
        NodeKind::PreimageSample { matrix_type: k_ty.clone(), max_coefficient_bound: 4.into() },
        vec![public.clone(), trapdoor, target],
        vec![WireType::Preimage { matrix: k_ty, max_coefficient_bound: 4.into() }],
    )
    .output(0)
    .unwrap();
    let lb = NodeHandle::new(
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
        vec![left.clone(), public],
        vec![WireType::Matrix(b_ty.clone())],
    )
    .output(0)
    .unwrap();
    let source = NodeHandle::new(
        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
        vec![lb, source_error],
        vec![WireType::Matrix(b_ty)],
    )
    .output(0)
    .unwrap();
    let result = NodeHandle::new(
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
        vec![source, k],
        vec![WireType::Matrix(one_ty)],
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        "stage-a-consumption",
        Vec::<CompileParameter>::new(),
        BTreeMap::from([(
            String::from("result"),
            GraphOutput { value: result, confidentiality: None },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let checked = validate(&graph, &ParamEnv::default()).unwrap();
    let mut options = ExportOptions::default();
    options.primitives.matrix_add = "(· + ·)".into();
    options.primitives.matrix_mul = "(· * ·)".into();
    let artifact = export(&checked, &options).unwrap();
    fs::write(output, artifact.source).unwrap();
}
