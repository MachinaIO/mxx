//! Generate and check the generic deterministic gadget-decomposition relation.

use crate::{
    Graph, GraphOutput, NodeHandle, ParamEnv, WireType,
    graph::CompileParameter,
    lean::{BackendLayout, ExportOptions, export},
    node::NodeKind,
    types::MatrixType,
    validate,
};
use std::collections::BTreeMap;

#[test]
fn export_gadget_fixture() {
    let matrix = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 1.into(),
        columns: 1.into(),
    };
    let input = NodeHandle::new(
        NodeKind::Input {
            name: "target".into(),
            wire_type: WireType::Matrix(matrix.clone()),
            artifact: None,
        },
        vec![],
        vec![WireType::Matrix(matrix.clone())],
    )
    .output(0)
    .unwrap();
    let decomposition = NodeHandle::new(
        NodeKind::GadgetDecompose { base: 2.into(), small: false, digit_count: 5.into() },
        vec![input],
        vec![WireType::Preimage {
            matrix: MatrixType { rows: 5.into(), ..matrix },
            max_coefficient_bound: 1.into(),
        }],
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        "stage-a-gadget",
        Vec::<CompileParameter>::new(),
        BTreeMap::from([(
            "decomposition".into(),
            GraphOutput { value: decomposition, confidentiality: None },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let checked = validate(&graph, &ParamEnv::default()).unwrap();
    let artifact = export(
        &checked,
        &ExportOptions {
            backend_layouts: vec![BackendLayout {
                modulus: 17.into(),
                ring_dimension: 2,
                base: 2.into(),
                regular_digits: 5,
            }],
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let proof = r#"
theorem generated_gadget_relation
    {backend : MxxRuntime.BackendContext}
    {target : Mxx.Primitives.ExactMatrix 17 2 1 1}
    {decomposition : Mxx.Primitives.ExactMatrix 17 2 5 1}
    (h : Generated.generatedRoot backend { unit := () } target decomposition) :
    MxxRuntime.gadgetDecomposeRuns backend 2 5 target decomposition := by
  rcases h with ⟨sample, sampleRuns, outputEq⟩
  rw [outputEq]
  exact sampleRuns
"#;
    super::write_fixture("gadget", format!("{}\n{}", artifact.source, proof));
}
