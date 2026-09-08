//! Producer key recurrences from the actual whole-Program family body.
use super::{
    root_inputs::value_expression,
    root_program_setup::{parallel_arguments, parallel_operand, program_cache_binding},
    root_readout_public::{root_fact, segmented_reduction},
};
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::LeanArtifact,
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use std::collections::BTreeMap;

pub(super) struct ProgramKeyStage {
    pub theorem: String,
    pub output: WireRef,
    pub public_inputs: WireRef,
    pub public_cache: WireRef,
    pub lane_count: usize,
    pub member_count: usize,
    pub offset: usize,
    pub raw_count: usize,
}

pub(super) fn root_program_keys(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<(String, BTreeMap<String, ProgramKeyStage>), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let body = shape.body_relation.as_ref().ok_or("missing key producer body")?;
    let record = shape.witness_record.as_ref().ok_or("missing key producer record")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let family_count = |wire: WireRef| -> Result<usize, String> {
        match &types[&wire] {
            ConcreteWireType::IndexedFamily { count, .. } => Ok(*count),
            _ => Err("key operation not family".into()),
        }
    };
    let mut source = String::from(
        "import RootProgramSetup\nimport ExponentReadout\nimport ExponentRefreshRoutes\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramKeys\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\n",
    );
    let mut stages = BTreeMap::new();
    for (name, cache) in &artifact.root.outputs {
        if !name.contains("-program-cache-family-") || !name.ends_with("-body-1-k-rhs") {
            continue;
        }
        let fail = || format!("unsupported producer key recurrence {name}");
        let private = artifact
            .root
            .outputs
            .get(&name.replace("-body-1-k-rhs", "-body-0-k-rhs"))
            .ok_or_else(fail)?;
        let members = family_count(private.wire)?;
        let lanes = family_count(cache.wire)?;
        if members == 0 || lanes % members != 0 {
            return Err(fail());
        }
        let width = lanes / members;
        let NodeKind::ParallelLoop(private_loop) =
            root.node(private.wire.node).ok_or_else(fail)?.kind()
        else {
            return Err(fail());
        };
        let offset = match private_loop.input_modes.as_slice() {
            [LoopInputMode::Zip] => 0,
            [LoopInputMode::ZipOffset { offset }] => *offset,
            _ => return Err(fail()),
        };
        let (scope_id, decomposition, _, _) = program_cache_binding(graph, cache.wire)?;
        let scope = &graph.source.scopes()[&scope_id];
        let member_shape = &artifact.scope_proofs[&scope_id];
        if scope.outputs().len() != 2 || cache.wire.port != Port(1) {
            return Err(fail());
        }
        let weighted_child = scope.outputs()[0];
        let weighted_node = scope.node(weighted_child.node).ok_or_else(fail)?;
        if !matches!(
            weighted_node.kind(),
            NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) {
            return Err(fail());
        }
        let operands = scope.arguments(weighted_node).ok_or_else(fail)?;
        if operands.len() != 2 || operands[1] != decomposition {
            return Err(fail());
        }
        let label_get = operands[0];
        let (public_inputs, _, get, _) =
            parallel_operand(graph, artifact, cache.wire, label_get, "position")?;
        if get.as_ref().map(|(get, _)| *get) != Some(label_get) {
            return Err(fail());
        }
        let global_count = family_count(public_inputs)?;
        if width == 0 || global_count % width != 0 {
            return Err(fail());
        }
        let raw_count = global_count / width;
        if offset + members > raw_count {
            return Err(fail());
        }
        let product = WireRef { node: cache.wire.node, port: Port(0) };
        let mut current = product;
        let mut remaining = members;
        while remaining > 1 {
            let next = remaining.div_ceil(2);
            current = root
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(index, node)| {
                    if !matches!(node.kind(), NodeKind::ParallelLoop(_)) ||
                        root.arguments(node)? != [current]
                    {
                        return None;
                    }
                    let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
                    (family_count(wire).ok()? == width * next).then_some(wire)
                })
                .ok_or_else(fail)?;
            remaining = next;
        }
        let (reduction, original, reduction_width) =
            segmented_reduction(graph, artifact, current, width, "lane")?;
        if original != product || reduction_width != members {
            return Err(fail());
        }
        let (index_slot, input_tuple, output_tuple) =
            parallel_arguments(graph, artifact, cache.wire, "position")?;
        let arguments = format!(
            "(backend := backend) (hashModel := hashModel) (params := params) (i_{index_slot} := position.val) (inputs := {input_tuple}) (outputs := {output_tuple}) (witness := leaf)"
        );
        let get_fact = member_shape
            .node_facts
            .get(&label_get.node)
            .and_then(|facts| facts.last())
            .ok_or_else(fail)?;
        let output_fact = member_shape.output_fact.as_ref().ok_or_else(fail)?;
        let number = stages.len();
        let theorem = format!("key_after_{number}");
        let actual = value(current)?;
        let inputs = value(public_inputs)?;
        let cached = value(cache.wire)?;
        let weighted = value(product)?;
        source.push_str(&format!("-- Actual producer key following cache {name:?}.\ntheorem {theorem} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedPrivateRoot.RootInput}} {{outputs : ExponentGeneratedPrivateRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (lane : Fin {width}) :\n    ({actual}) lane = ∑ member : Fin {members},\n      ({inputs}) ⟨lane.val * {raw_count} + {offset} + member.val, by clear * - lane member; omega⟩ *\n      ({cached}) ⟨lane.val * {members} + member.val, by clear * - lane member; omega⟩ := by\n{reduction}  rw [reduced]\n  apply Finset.sum_congr rfl\n  intro memberIndex _\n  let position : Fin {lanes} := ⟨lane.val * {members} + memberIndex.val, by omega⟩\n  obtain ⟨leaf, leafFacts⟩ := ({} backend hashModel params position.val {input_tuple} {output_tuple}).mp (({}) position)\n  have getFact := {get_fact} {arguments} leafFacts\n  have outputFact := {output_fact} {arguments} leafFacts\n  have memberMod : (position.val : Int) % {members} = (memberIndex.val : Int) := by dsimp [position]; omega\n  have memberDiv : (position.val : Int) / {members} = (lane.val : Int) := by dsimp [position]; omega\n  have labelEq : leaf.w_{}_{} = ({inputs}) ⟨lane.val * {raw_count} + {offset} + memberIndex.val, by omega⟩ := by\n    apply route_lookup_eq _ ?_ getFact\n    change (position.val : Int) / {members} * {raw_count} + ((position.val : Int) % {members} + {offset}) = ((lane.val * {raw_count} + {offset} + memberIndex.val : Nat) : Int)\n    rw [memberDiv, memberMod]\n    omega\n  have keyEq := congrArg (fun output => output.1) outputFact\n  have cacheEq := congrArg (fun output => output.2.1) outputFact\n  change ({weighted}) position = leaf.w_{}_{} * leaf.w_{}_{} at keyEq\n  change ({cached}) position = leaf.w_{}_{} at cacheEq\n  change ({weighted}) position = _\n  rw [keyEq, cacheEq, labelEq]\n\n", member_shape.runs_iff_body, root_fact(artifact,cache.wire)?, label_get.node.0,label_get.port.0,label_get.node.0,label_get.port.0,decomposition.node.0,decomposition.port.0,decomposition.node.0,decomposition.port.0,params=artifact.root.parameter_type));
        stages.insert(
            name.clone(),
            ProgramKeyStage {
                theorem: format!("ExponentGeneratedProgramKeys.{theorem}"),
                output: current,
                public_inputs,
                public_cache: cache.wire,
                lane_count: width,
                member_count: members,
                offset,
                raw_count,
            },
        );
    }
    if stages.is_empty() {
        return Err("missing producer key stages".into());
    }
    source.push_str("end ExponentGeneratedProgramKeys\n");
    Ok((source, stages))
}
