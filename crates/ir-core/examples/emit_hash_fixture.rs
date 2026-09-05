//! Emit full-tag plain hashing and canonical coefficient extraction from frozen IR.
use mxx_ir_core::{
    Graph, GraphOutput, NodeHandle, ParamEnv, WireType,
    lean::{ExportOptions, export},
    node::{HashVariant, NodeKind},
    types::MatrixType,
    validate,
};
use std::{collections::BTreeMap, env, fs};

fn main() {
    let path = env::args().nth(1).expect("output Lean path");
    let key_type = WireType::Bytes { length: 32.into() };
    let key = NodeHandle::new(
        NodeKind::Input { name: "key".into(), wire_type: key_type.clone(), artifact: None },
        vec![],
        vec![key_type],
    )
    .output(0)
    .unwrap();
    let operand = NodeHandle::new(
        NodeKind::Input { name: "operand".into(), wire_type: WireType::Int, artifact: None },
        vec![],
        vec![WireType::Int],
    )
    .output(0)
    .unwrap();
    let matrix = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 1.into(),
        columns: 1.into(),
    };
    let hash = NodeHandle::new(
        NodeKind::HashSample {
            matrix_type: matrix.clone(),
            variant: HashVariant::Plain,
            tag_prefix: vec![0, 255],
            tag_expressions: vec![(-256).into()],
            tag_decimal_expressions: vec![(-42).into()],
            tag_u64_le_expressions: vec![258.into()],
            base: None,
            digit_count: None,
        },
        vec![key, operand],
        vec![WireType::Matrix(matrix)],
    )
    .output(0)
    .unwrap();
    let coefficient = NodeHandle::new(
        NodeKind::ExtractCoefficient { position: 1.into(), canonical_input_exclusive_upper: None },
        vec![hash.clone()],
        vec![WireType::Int],
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        "hash-fixture",
        vec![],
        BTreeMap::from([
            ("hash".into(), GraphOutput { value: hash, confidentiality: None }),
            ("coefficient".into(), GraphOutput { value: coefficient, confidentiality: None }),
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
example : MxxRuntime.signedIntegerTag (-256) = [1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0] := by decide
example : MxxRuntime.signedIntegerTag 0 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] := by decide
example : MxxRuntime.u64LittleEndian 258 = [2, 1, 0, 0, 0, 0, 0, 0] := by decide
example : MxxRuntime.completeHashTag [] [] [(-42)] [] [] = [45, 52, 50] := by decide

theorem generated_hash_output {model : MxxRuntime.HashModel} {key : ByteArray} {operand : Int}
    {outputs : Int × Mxx.Primitives.ExactMatrix 17 2 1 1 × Unit}
    (h : Generated.generatedRoot model { unit := () } (key, operand, ()) outputs) :
    outputs.2.1 = model.sample 17 2 1 1 key
      (MxxRuntime.completeHashTag [0, 255] [(-256)] [(-42)] [258] [operand]) := by
  rcases h with ⟨sample, coefficient, hashRun, extractRun, outputEq⟩
  rw [outputEq]
  exact hashRun.2.2

theorem generated_canonical_coefficient {model : MxxRuntime.HashModel}
    {key : ByteArray} {operand : Int}
    {outputs : Int × Mxx.Primitives.ExactMatrix 17 2 1 1 × Unit}
    (h : Generated.generatedRoot model { unit := () } (key, operand, ()) outputs) :
    ∃ index : Fin 2, (index.val : Int) = 1 ∧
      outputs.1 = ((outputs.2.1 0 0).coeff index).val := by
  rcases h with ⟨sample, coefficient, hashRun, extractRun, outputEq⟩
  rw [outputEq]
  exact extractRun
"#;
    fs::write(path, format!("{}\n{proof}", artifact.source)).unwrap();
}
