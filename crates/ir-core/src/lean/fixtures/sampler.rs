//! Generate the sampled-trapdoor path and consume both coupled outputs in Lean.

use crate::{
    Graph, GraphOutput, NodeHandle, ParamEnv, RealExpr, WireType,
    graph::CompileParameter,
    lean::{BackendLayout, ExportOptions, export},
    node::NodeKind,
    types::MatrixType,
    validate,
};
use std::collections::BTreeMap;

#[test]
fn export_sampler_fixture() {
    let public_type = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 1.into(),
        columns: 3.into(),
    };
    let target_type = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 1.into(),
        columns: 1.into(),
    };
    let preimage_type = MatrixType {
        modulus: 17.into(),
        ring_dimension: 2.into(),
        rows: 3.into(),
        columns: 1.into(),
    };
    let trapdoor = NodeHandle::new(
        NodeKind::TrapdoorSample {
            matrix_type: public_type.clone(),
            sigma: RealExpr::from(1),
            gadget_base: 32.into(),
            digit_count: 1.into(),
            preimage_max_coefficient_bound: 4.into(),
        },
        vec![],
        vec![
            WireType::Matrix(public_type.clone()),
            WireType::Trapdoor {
                matrix: public_type.clone(),
                sigma: RealExpr::from(1),
                gadget_base: 32.into(),
                digit_count: 1.into(),
                preimage_max_coefficient_bound: 4.into(),
            },
        ],
    );
    let public = trapdoor.output(0).unwrap();
    let token = trapdoor.output(1).unwrap();
    let target = NodeHandle::new(
        NodeKind::Input {
            name: "target".into(),
            wire_type: WireType::Matrix(target_type.clone()),
            artifact: None,
        },
        vec![],
        vec![WireType::Matrix(target_type)],
    )
    .output(0)
    .unwrap();
    let preimage = NodeHandle::new(
        NodeKind::PreimageSample {
            matrix_type: preimage_type.clone(),
            max_coefficient_bound: 4.into(),
        },
        vec![public.clone(), token.clone(), target.clone()],
        vec![WireType::Preimage { matrix: preimage_type.clone(), max_coefficient_bound: 4.into() }],
    )
    .output(0)
    .unwrap();
    let second = NodeHandle::new(
        NodeKind::PreimageSample {
            matrix_type: preimage_type.clone(),
            max_coefficient_bound: 4.into(),
        },
        vec![public, token, target],
        vec![WireType::Preimage { matrix: preimage_type, max_coefficient_bound: 4.into() }],
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        "stage-a-sampled-trapdoor",
        Vec::<CompileParameter>::new(),
        BTreeMap::from([
            ("preimage".into(), GraphOutput { value: preimage, confidentiality: None }),
            ("second".into(), GraphOutput { value: second, confidentiality: None }),
        ]),
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
                base: 32.into(),
                regular_digits: 1,
            }],
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let proof = r#"
theorem generated_sampled_trapdoor_path
    {backend : MxxRuntime.BackendContext}
    {target : Mxx.Primitives.ExactMatrix 17 2 1 1}
    {k : Mxx.Primitives.ExactMatrix 17 2 3 1}
    {second : Mxx.Primitives.ExactMatrix 17 2 3 1}
    (h : Generated.generatedRoot backend { unit := () } target (k, second, ())) :
    ∃ (publicMatrix : Mxx.Primitives.ExactMatrix 17 2 1 3)
      (td : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 17 2 1 3) Unit),
      td.kind = .sampledSecret ∧ td.publicMatrix = publicMatrix ∧
        publicMatrix * k = target ∧ publicMatrix * second = target := by
  rcases h with ⟨td, publicMatrix, sample, sample2, sampleRuns, _, preimageRuns, _, preimageRuns2, outputEq⟩
  refine ⟨publicMatrix, td, MxxRuntime.trapdoorSample_sampled sampleRuns,
    MxxRuntime.trapdoorSample_public sampleRuns, ?_, ?_⟩
  · have hk := congrArg Prod.fst outputEq
    change k = sample at hk
    rw [hk]
    exact MxxRuntime.preimageRunsDispatched_equation (by decide) (by decide) preimageRuns
  · have hk := congrArg (fun result => result.2.1) outputEq
    change second = sample2 at hk
    rw [hk]
    exact MxxRuntime.preimageRunsDispatched_equation (by decide) (by decide) preimageRuns2

#print axioms generated_sampled_trapdoor_path
"#;
    let source = format!("{}\n{}", artifact.source, proof);
    println!(
        "source_bytes={} relation_declarations={} static_node_visits={} proof_declarations=1",
        source.len(),
        artifact.source.lines().filter(|line| line.starts_with("def ")).count(),
        artifact.static_node_visits
    );
    super::write_fixture("sampler", source);
}
