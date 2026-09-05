//! Generate literal-matrix semantics and proofs about their actual named root projections.

use crate::{
    Graph, GraphOutput, NodeHandle, ParamEnv,
    expr::IntExpr,
    lean::{ExportOptions, export},
    node::{ConstantMatrix, NodeKind},
    types::{MatrixType, WireType},
    validate,
};
use std::collections::BTreeMap;

#[test]
fn export_constant_fixture() {
    let matrix = MatrixType {
        modulus: IntExpr::constant(17),
        ring_dimension: IntExpr::constant(2),
        rows: IntExpr::constant(1),
        columns: IntExpr::constant(1),
    };
    let constant = |value| {
        NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap()
    };
    let polynomial = constant(ConstantMatrix::Polynomial {
        coefficients: vec![IntExpr::constant(2), IntExpr::constant(-3)],
    });
    let graph = Graph::freeze(
        "constant-fixture",
        vec![],
        BTreeMap::from([
            (
                "zero".into(),
                GraphOutput { value: constant(ConstantMatrix::Zero), confidentiality: None },
            ),
            (
                "identity".into(),
                GraphOutput { value: constant(ConstantMatrix::Identity), confidentiality: None },
            ),
            ("polynomial".into(), GraphOutput { value: polynomial.clone(), confidentiality: None }),
            ("alias".into(), GraphOutput { value: polynomial, confidentiality: None }),
        ]),
        vec![constant(ConstantMatrix::Polynomial { coefficients: vec![] })],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let checked = validate(&graph, &ParamEnv::default()).unwrap();
    let artifact = export(&checked, &ExportOptions::default()).unwrap();
    let root = &artifact.root;
    let mut proof = String::new();
    for (name, expected, tactic) in [
        ("zero", "0", "rw [h]"),
        ("identity", "1", "rw [h]"),
        (
            "polynomial",
            "fun _ _ ↦ (2 : Mxx.Primitives.ExactPoly 17 2) + AdjoinRoot.root (Mxx.Primitives.negacyclicModulus 2 (ZMod 17)) * (-3)",
            "rw [h]\n  funext i j\n  simp [MxxRuntime.matrixPolynomial]",
        ),
    ] {
        proof.push_str(&format!(
            "\ntheorem generated_{name} {{outputs : {}}}\n    (h : {} {{ unit := () }} () outputs) :\n    {} = ({expected} : {}) := by\n  change outputs = _ at h\n  {tactic}\n",
            root.output_type, root.relation, root.outputs[name].projection,
            root.outputs[name].lean_type,
        ));
    }
    proof.push_str(&format!(
        "\ntheorem generated_alias {{outputs : {}}} : {} = {} := by rfl\n",
        root.output_type, root.outputs["alias"].projection, root.outputs["polynomial"].projection,
    ));
    super::write_fixture("constant", format!("{}\n{proof}", artifact.source));
}
