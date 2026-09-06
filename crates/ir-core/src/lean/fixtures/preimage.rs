//! Generate a real frozen-IR preimage consumer and its kernel-checked Lean equation.
//!
//! The graph contains the nonzero-target-error path `(L * B + e) * K`, with
//! `B * K = P + E`. Matrix operators are supplied as ordinary typeclass operations only for this
//! fixture; the exporter itself remains application agnostic.

use crate::{
    Graph, GraphOutput, NodeHandle, ParamEnv, RealExpr, WireType,
    graph::CompileParameter,
    lean::{BackendLayout, ExportOptions, PrimitiveNames, export},
    node::{MatrixBinaryOp, NodeKind},
    types::MatrixType,
    validate,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Geometry {
    name: &'static str,
    source_rows: u32,
    inner: u32,
    target_columns: u32,
}

fn geometry(name: &str) -> Geometry {
    match name {
        "wide" => Geometry { name: "Wide", source_rows: 2, inner: 3, target_columns: 2 },
        "small" => Geometry { name: "Small", source_rows: 1, inner: 3, target_columns: 1 },
        other => panic!("geometry must be `small` or `wide`, got {other:?}"),
    }
}

fn matrix_type(rows: u32, columns: u32) -> MatrixType {
    MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: (rows as usize).into(),
        columns: (columns as usize).into(),
    }
}

fn input(name: &str, ty: MatrixType) -> crate::graph::ValueHandle {
    NodeHandle::new(
        NodeKind::Input {
            name: name.into(),
            wire_type: WireType::Matrix(ty.clone()),
            artifact: None,
        },
        vec![],
        vec![WireType::Matrix(ty)],
    )
    .output(0)
    .unwrap()
}

fn render(selected: Geometry) -> String {
    let b = input("B", matrix_type(selected.source_rows, selected.inner));
    let trapdoor_matrix = matrix_type(selected.source_rows, selected.inner);
    let trapdoor = NodeHandle::new(
        NodeKind::Input {
            name: "trapdoor".into(),
            wire_type: WireType::Trapdoor {
                matrix: trapdoor_matrix.clone(),
                sigma: RealExpr::from(1),
                gadget_base: 32.into(),
                digit_count: 1.into(),
                preimage_max_coefficient_bound: 4.into(),
            },
            artifact: None,
        },
        vec![],
        vec![WireType::Trapdoor {
            matrix: trapdoor_matrix,
            sigma: RealExpr::from(1),
            gadget_base: 32.into(),
            digit_count: 1.into(),
            preimage_max_coefficient_bound: 4.into(),
        }],
    )
    .output(0)
    .unwrap();
    let left = input("L", matrix_type(1, selected.source_rows));
    let left_error = input("e", matrix_type(1, selected.inner));
    let ideal = input("P", matrix_type(selected.source_rows, selected.target_columns));
    let target_error = input("E", matrix_type(selected.source_rows, selected.target_columns));
    let target_type = matrix_type(selected.source_rows, selected.target_columns);
    let target = NodeHandle::new(
        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
        vec![ideal.clone(), target_error.clone()],
        vec![WireType::Matrix(target_type.clone())],
    )
    .output(0)
    .unwrap();
    let preimage_type = matrix_type(selected.inner, selected.target_columns);
    let preimage = NodeHandle::new(
        NodeKind::PreimageSample {
            matrix_type: preimage_type.clone(),
            max_coefficient_bound: 4.into(),
        },
        vec![b.clone(), trapdoor, target.clone()],
        vec![WireType::Preimage { matrix: preimage_type, max_coefficient_bound: 4.into() }],
    )
    .output(0)
    .unwrap();
    let left_times_public = NodeHandle::new(
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
        vec![left.clone(), b],
        vec![WireType::Matrix(matrix_type(1, selected.inner))],
    )
    .output(0)
    .unwrap();
    let value = NodeHandle::new(
        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
        vec![left_times_public, left_error],
        vec![WireType::Matrix(matrix_type(1, selected.inner))],
    )
    .output(0)
    .unwrap();
    let consumed = NodeHandle::new(
        NodeKind::MatrixMulSmallRhs,
        vec![value, preimage.clone()],
        vec![WireType::Matrix(matrix_type(1, selected.target_columns))],
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        format!("stage-a-consumption-{}", selected.name),
        Vec::<CompileParameter>::new(),
        BTreeMap::from([
            ("c".into(), GraphOutput { value: consumed, confidentiality: None }),
            ("k".into(), GraphOutput { value: preimage, confidentiality: None }),
            ("target".into(), GraphOutput { value: target, confidentiality: None }),
        ]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let checked = validate(&graph, &ParamEnv::default()).unwrap();
    let primitives = PrimitiveNames {
        matrix_add: "(fun x y => x + y)".into(),
        matrix_mul: "(fun x y => x * y)".into(),
        ..PrimitiveNames::default()
    };
    let namespace = format!("Generated{}", selected.name);
    let artifact = export(
        &checked,
        &ExportOptions {
            namespace: namespace.clone(),
            module_name: namespace.clone(),
            primitives,
            backend_layouts: vec![BackendLayout {
                modulus: 17.into(),
                ring_dimension: 2,
                base: 32.into(),
                regular_digits: 1,
            }],
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let sr = selected.source_rows;
    let inner = selected.inner;
    let tc = selected.target_columns;
    let proof = format!(
        r#"
theorem generated_nonzero_target_consumption
    {{backend : MxxRuntime.BackendContext}}
    {{b : Mxx.Primitives.ExactMatrix 17 2 {sr} {inner}}}
    {{td : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 17 2 {sr} {inner}) Unit}}
    {{l : Mxx.Primitives.ExactMatrix 17 2 1 {sr}}}
    {{e : Mxx.Primitives.ExactMatrix 17 2 1 {inner}}}
    {{p : Mxx.Primitives.ExactMatrix 17 2 {sr} {tc}}}
    {{targetError : Mxx.Primitives.ExactMatrix 17 2 {sr} {tc}}}
    {{c : Mxx.Primitives.ExactMatrix 17 2 1 {tc}}}
    {{k : Mxx.Primitives.ExactMatrix 17 2 {inner} {tc}}}
    {{target : Mxx.Primitives.ExactMatrix 17 2 {sr} {tc}}}
    (h : {namespace}.generatedRoot backend {{ unit := () }}
      (l, b, e, td, p, targetError, ()) (c, k, target, ())) :
    c = l * p + (l * targetError + e * k) := by
  rcases h with ⟨sample, _, sampleRuns, outputEq⟩
  have sampleEq : k = sample := by
    simpa using congrArg (fun value => value.2.1) outputEq
  have hc : c = (l * b + e) * k := by
    simpa [sampleEq] using congrArg Prod.fst outputEq
  have ht : target = p + targetError := by
    simpa using congrArg (fun value => value.2.2.1) outputEq
  have hkSample : b * sample = p + targetError := by
    simpa using (MxxRuntime.preimageRunsDispatched_equation (by norm_num) (by norm_num) sampleRuns)
  have hk : b * k = target := by
    rw [sampleEq, ht]
    exact hkSample
  rw [hc, Matrix.add_mul, Matrix.mul_assoc l b k, hk, ht, Matrix.mul_add]
  ac_rfl
"#,
    );
    format!("{}\n{}", artifact.source, proof)
}

#[test]
fn export_preimage_fixtures() {
    for name in ["small", "wide"] {
        super::write_fixture(&format!("preimage_{name}"), render(geometry(name)));
    }
}
