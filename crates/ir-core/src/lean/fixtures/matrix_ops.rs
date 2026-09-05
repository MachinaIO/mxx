//! Generate generic slice and matrix-concatenation relations, including a loop-dependent slice
//! offset.

use crate::{
    Graph, GraphOutput, NodeHandle, ParamEnv,
    expr::IntExpr,
    graph::{CompileParameter, SubgraphHandle, with_new_construction_scope},
    lean::{ExportOptions, export},
    node::{ConcatAxis, IndexRange, LoopInputMode, NodeKind, ParallelLoop},
    types::{MatrixType, WireType},
    validate,
};
use std::collections::BTreeMap;

fn matrix(rows: usize, columns: usize) -> MatrixType {
    MatrixType {
        modulus: IntExpr::constant(17),
        ring_dimension: IntExpr::constant(2),
        rows: IntExpr::constant(rows),
        columns: IntExpr::constant(columns),
    }
}

#[test]
fn export_matrix_ops_fixture() {
    let one_by_two = matrix(1, 2);
    let two_by_one = matrix(2, 1);
    let one_by_one = matrix(1, 1);
    let two_by_two = matrix(2, 2);
    let three_by_three = matrix(3, 3);
    let input = |name: &str, ty: MatrixType| {
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
    };
    let left = input("left", one_by_two.clone());
    let right = input("right", one_by_two.clone());
    let narrow = input("narrow", one_by_one.clone());
    let tall = input("tall", two_by_one.clone());
    let slice_input = input("slice_input", matrix(2, 2));
    let rows = NodeHandle::new(
        NodeKind::Concat { axis: ConcatAxis::Rows },
        vec![left.clone(), right.clone()],
        vec![WireType::Matrix(two_by_two)],
    )
    .output(0)
    .unwrap();
    let columns = NodeHandle::new(
        NodeKind::Concat { axis: ConcatAxis::Columns },
        vec![left.clone(), narrow],
        vec![WireType::Matrix(matrix(1, 3))],
    )
    .output(0)
    .unwrap();
    let diagonal = NodeHandle::new(
        NodeKind::Concat { axis: ConcatAxis::Diagonal },
        vec![left.clone(), tall],
        vec![WireType::Matrix(three_by_three)],
    )
    .output(0)
    .unwrap();
    let slice_body = with_new_construction_scope(|scope| {
        let value = input("formal", matrix(2, 2));
        let start = IntExpr::LoopIndex(5);
        let end = IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(1)));
        let sliced = NodeHandle::new(
            NodeKind::Slice {
                rows: Some(IndexRange { start, end }),
                columns: Some(IndexRange {
                    start: IntExpr::constant(0),
                    end: IntExpr::constant(2),
                }),
            },
            vec![value.clone()],
            vec![WireType::Matrix(matrix(1, 2))],
        )
        .output(0)
        .unwrap();
        SubgraphHandle::new("slice_body", scope, vec![value], vec![sliced]).unwrap()
    });
    let family = NodeHandle::parallel_loop(
        slice_body,
        vec![slice_input],
        vec![WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix(1, 2))),
            count: IntExpr::constant(2),
        }],
        ParallelLoop {
            count: IntExpr::constant(2),
            minimum_count: 1,
            index_slot: 5,
            bindings: vec![],
            input_modes: vec![LoopInputMode::Broadcast],
        },
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        "stage-a-matrix-ops",
        Vec::<CompileParameter>::new(),
        BTreeMap::from([
            ("rows".into(), GraphOutput { value: rows, confidentiality: None }),
            ("columns".into(), GraphOutput { value: columns, confidentiality: None }),
            ("diagonal".into(), GraphOutput { value: diagonal, confidentiality: None }),
            ("slice".into(), GraphOutput { value: family, confidentiality: None }),
        ]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let checked = validate(&graph, &ParamEnv::default()).unwrap();
    let artifact = export(&checked, &ExportOptions::default()).unwrap();
    let proof = r#"
theorem generated_concat_rows_first
    {left : Mxx.Primitives.ExactMatrix 17 2 1 2}
    {right : Mxx.Primitives.ExactMatrix 17 2 1 2}
    {output : Mxx.Primitives.ExactMatrix 17 2 2 2}
    (h : MxxRuntime.concatRows left right output) (column : Fin 2) :
    output ⟨0, by omega⟩ column = left ⟨0, by omega⟩ column := by
  simpa using h ⟨0, by omega⟩ column

theorem generated_concat_columns_first
    {left : Mxx.Primitives.ExactMatrix 17 2 1 2}
    {right : Mxx.Primitives.ExactMatrix 17 2 1 1}
    {output : Mxx.Primitives.ExactMatrix 17 2 1 3}
    (h : MxxRuntime.concatColumns left right output) :
    output ⟨0, by omega⟩ ⟨0, by omega⟩ = left ⟨0, by omega⟩ ⟨0, by omega⟩ := by
  simpa using h ⟨0, by omega⟩ ⟨0, by omega⟩

theorem generated_concat_diagonal_bottom_right
    {left : Mxx.Primitives.ExactMatrix 17 2 1 2}
    {right : Mxx.Primitives.ExactMatrix 17 2 2 1}
    {output : Mxx.Primitives.ExactMatrix 17 2 3 3}
    (h : MxxRuntime.concatDiagonal left right output) :
    output ⟨1, by omega⟩ ⟨2, by omega⟩ = right ⟨0, by omega⟩ ⟨0, by omega⟩ := by
  simpa using h ⟨1, by omega⟩ ⟨2, by omega⟩

theorem generated_root_columns_projection
    {inputs : Mxx.Primitives.ExactMatrix 17 2 1 2 ×
      Mxx.Primitives.ExactMatrix 17 2 1 1 × Mxx.Primitives.ExactMatrix 17 2 2 1 ×
      Mxx.Primitives.ExactMatrix 17 2 1 2 × Mxx.Primitives.ExactMatrix 17 2 2 2 × Unit}
    {outputs : Mxx.Primitives.ExactMatrix 17 2 1 3 × Mxx.Primitives.ExactMatrix 17 2 3 3 ×
      Mxx.Primitives.ExactMatrix 17 2 2 2 ×
      (Fin 2 → Mxx.Primitives.ExactMatrix 17 2 1 2) × Unit}
    (h : Generated.generatedRoot { unit := () } inputs outputs) :
    outputs.1 ⟨0, by omega⟩ ⟨0, by omega⟩ =
      inputs.1 ⟨0, by omega⟩ ⟨0, by omega⟩ := by
  rcases h with ⟨columns, diagonal, rows, family, hColumns, hDiagonal, hRows, hFamily, hout⟩
  rw [hout]
  simpa using hColumns ⟨0, by omega⟩ ⟨0, by omega⟩

theorem generated_slice_coefficients
    {i : Fin 2}
    {input : Mxx.Primitives.ExactMatrix 17 2 2 2}
    {output : Mxx.Primitives.ExactMatrix 17 2 1 2}
    (h : Generated.parallel_generatedRoot_8 { unit := () } i input output) :
    output ⟨0, by omega⟩ ⟨0, by omega⟩ = input ⟨i.val, by omega⟩ ⟨0, by omega⟩ := by
  rcases h with ⟨w, _, _, _, _, _, _, hslice, hout⟩
  subst output
  rcases hslice with ⟨_, _, _, _, _, _, _, _, hcoeff⟩
  simpa using hcoeff ⟨0, by omega⟩ ⟨0, by omega⟩ (by omega) (by omega)
"#;
    super::write_fixture("matrix_ops", format!("{}\n{}", artifact.source, proof));
}
