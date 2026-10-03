//! Generate integer hash families, integer matrix-vector products, and runtime monomial
//! multiplication from frozen IR.
use crate::{
    Graph, GraphOutput, NodeHandle, ParamEnv, WireType,
    lean::{ExportOptions, export},
    node::{HashTagComponent, NodeKind},
    types::MatrixType,
};
use std::collections::BTreeMap;

#[test]
fn export_integer_lwe_fixture() {
    let input = |name: &str, wire_type: WireType| {
        NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: wire_type.clone(), artifact: None },
            vec![],
            vec![wire_type],
        )
        .output(0)
        .unwrap()
    };
    let family = |count: usize| WireType::IndexedFamily {
        element: Box::new(WireType::Int),
        count: count.into(),
    };
    let key = input("key", WireType::Bytes { length: 32.into() });
    let secret = input("secret", family(2));
    let exponent = input("exponent", WireType::Int);
    let matrix =
        MatrixType { ring: crate::ring::test_ring(17, 4), rows: 1.into(), columns: 2.into() };
    let accumulator = input("accumulator", WireType::Matrix(matrix.clone()));
    let masks = NodeHandle::new(
        NodeKind::HashIntFamily {
            count: 6.into(),
            modulus: 8.into(),
            tag_prefix: vec![7],
            tag_components: vec![HashTagComponent::U64Le(3.into())],
        },
        vec![key],
        vec![family(6)],
    )
    .output(0)
    .unwrap();
    let product = |transpose: bool| {
        NodeHandle::new(
            NodeKind::IntMatrixVectorProduct { transpose },
            vec![masks.clone(), secret.clone()],
            vec![family(3)],
        )
        .output(0)
        .unwrap()
    };
    let rows = product(false);
    let columns = product(true);
    let rotated = NodeHandle::new(
        NodeKind::MultiplyMonomial,
        vec![accumulator, exponent],
        vec![WireType::Matrix(matrix)],
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        "integer-lwe-fixture",
        vec![],
        BTreeMap::from([
            ("masks".into(), GraphOutput { value: masks, availability: None }),
            ("rows".into(), GraphOutput { value: rows, availability: None }),
            ("columns".into(), GraphOutput { value: columns, availability: None }),
            ("rotated".into(), GraphOutput { value: rotated, availability: None }),
        ]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let checked = crate::ring::test_validate(&graph, &ParamEnv::default()).unwrap();
    let artifact = export(&checked, &ExportOptions::default()).unwrap();
    let proof = r#"
section IntegerLwe
variable {model : MxxRuntime.HashModel} {key : ByteArray} {secret : Fin 2 → Int}
  {accumulator : Mxx.Primitives.ExactMatrix 17 4 1 2} {exponent : Int}
  {outputs : (Fin 3 → Int) × (Fin 6 → Int) × Mxx.Primitives.ExactMatrix 17 4 1 2 ×
    (Fin 3 → Int) × Unit}

theorem generated_masks_in_range
    (h : Generated.generatedRoot model { unit := () } (key, secret, accumulator, exponent, ())
      outputs) (index : Fin 6) :
    outputs.2.1 index = ((model.integers 6 8 key
        (MxxRuntime.completeHashTag [7] [.u64Le 3]) index % 8 : Nat) : Int) ∧
      0 ≤ outputs.2.1 index ∧ outputs.2.1 index < 8 := by
  rcases h with ⟨witness, ⟨_, _, _, hmasks⟩, outputEq⟩
  rw [outputEq]
  simp only
  rw [hmasks]
  dsimp only
  rw [show Int.toNat 8 = 8 from rfl]
  refine ⟨rfl, by positivity, ?_⟩
  have := Nat.mod_lt (model.integers 6 8 key
    (MxxRuntime.completeHashTag [7] [.u64Le 3]) index) (by decide : 0 < 8)
  omega

theorem generated_row_product
    (h : Generated.generatedRoot model { unit := () } (key, secret, accumulator, exponent, ())
      outputs) :
    outputs.2.2.2.1 1 = outputs.2.1 2 * secret 0 + outputs.2.1 3 * secret 1 := by
  rcases h with ⟨witness, _, outputEq⟩
  rw [outputEq]
  simp [MxxRuntime.intMatrixVectorProduct, Fin.sum_univ_two]

theorem generated_column_product
    (h : Generated.generatedRoot model { unit := () } (key, secret, accumulator, exponent, ())
      outputs) :
    outputs.1 0 = outputs.2.1 0 * secret 0 + outputs.2.1 3 * secret 1 := by
  rcases h with ⟨witness, _, outputEq⟩
  rw [outputEq]
  simp [MxxRuntime.intMatrixVectorProduct, Fin.sum_univ_two]

/-- A negative exponent wraps modulo `2n`, and `X^n = -1` negates. -/
theorem generated_monomial_wraps
    (h : Generated.generatedRoot model { unit := () } (key, secret, accumulator, -4, ())
      outputs) :
    outputs.2.2.1 = -accumulator := by
  rcases h with ⟨witness, _, outputEq⟩
  rw [outputEq]
  funext row column
  have hpow : AdjoinRoot.root (Mxx.Primitives.negacyclicModulus 4 (ZMod 17)) ^ 4 = -1 :=
    Mxx.Primitives.Negacyclic.root_pow_n
  simp [MxxRuntime.multiplyMonomial, hpow]
end IntegerLwe
"#;
    super::write_fixture("integer_lwe", format!("{}\n{proof}", artifact.source));
}
