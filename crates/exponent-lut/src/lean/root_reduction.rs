//! Actual Program reductions through the shared segmented Family proof.
use super::{
    root_inputs::value_expression, root_program::ProgramStage,
    root_readout_public::segmented_reduction,
};
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::LeanArtifact,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use std::collections::BTreeMap;

pub(super) fn root_reduction(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    stages: &[ProgramStage],
) -> Result<(String, BTreeMap<WireRef, WireRef>), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let body = shape.body_relation.as_ref().ok_or("missing reduction root body")?;
    let record = shape.witness_record.as_ref().ok_or("missing reduction root witness")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut source = format!(
        "import {}\nimport ExponentReadout\nimport ExponentRefreshRoutes\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedReduction\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\n",
        artifact.module_name
    );
    let mut outputs = BTreeMap::new();
    for stage in stages {
        let mut output = stage.output;
        let mut width = stage.members;
        while width > 1 {
            let next = width.div_ceil(2);
            output = root.nodes().iter().enumerate().find_map(|(index, node)| {
                if !matches!(node.kind(), NodeKind::ParallelLoop(_)) || root.arguments(node)? != [output] {
                    return None;
                }
                let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
                matches!(&types[&wire], ConcreteWireType::IndexedFamily { count, .. } if *count == stage.lane_count * next).then_some(wire)
            }).ok_or("missing actual segmented Program reduction")?;
            width = next;
        }
        let (proof, original, members) =
            segmented_reduction(graph, artifact, output, stage.lane_count, "lane")?;
        if original != stage.output || members != stage.members {
            return Err("Program reduction does not reach its actual body output".into());
        }
        source.push_str(&format!("theorem reduction_{} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (lane : Fin {}) :\n    ({}) lane = ∑ member : Fin {}, ({}) ⟨lane.val * {} + member.val, by omega⟩ := by\n{}  exact reduced\n\n",
            stage.index, artifact.root.parameter_type, artifact.root.input_type, artifact.root.output_type,
            stage.lane_count, value(output)?, stage.members, value(stage.output)?, stage.members, proof));
        outputs.insert(stage.output, output);
    }
    source.push_str("end ExponentGeneratedReduction\n");
    Ok((source, outputs))
}
