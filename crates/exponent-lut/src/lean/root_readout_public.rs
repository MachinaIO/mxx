//! Actual producer public readout: segmented helper sum and final normalization.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    graph::GraphScope,
    lean::LeanArtifact,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

/// Bind each actual single-capture reduction layer, from the output to its source.
pub(super) fn segmented_reduction_levels(
    graph: &ValidatedGraph,
    output: WireRef,
    labels: usize,
) -> Result<Vec<(WireRef, WireRef, NodeId, NodeId, usize, usize)>, String> {
    let root = graph.source.root_scope();
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let child = |wire: WireRef| {
        graph.source.scopes().get(&FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: wire.node,
        })
    };
    let mut current = output;
    let mut levels = Vec::new();
    loop {
        let Some(scope) = child(current) else {
            break;
        };
        let Some(parents) = root.arguments(root.node(current.node).ok_or("missing reducer node")?)
        else {
            break;
        };
        if parents.len() != 1 || scope.outputs().len() != 1 {
            break;
        }
        let gets = scope
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.kind(), NodeKind::FamilyGetDynamic))
            .map(|(index, _)| NodeId(index as u64))
            .collect::<Vec<_>>();
        if gets.len() != 2 {
            break;
        }
        let left_source = parents[0];
        let ConcreteWireType::IndexedFamily { count, .. } = &types[&left_source] else {
            break;
        };
        let ConcreteWireType::IndexedFamily { count: next, .. } = &types[&current] else {
            break;
        };
        if labels == 0 ||
            count % labels != 0 ||
            next % labels != 0 ||
            next / labels != (count / labels).div_ceil(2)
        {
            return Err("non-segmented readout reducer".into());
        }
        levels.push((current, left_source, gets[0], gets[1], count / labels, next / labels));
        current = left_source;
    }
    Ok(levels)
}

pub(super) fn segmented_reduction(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    labels: usize,
    segment: &str,
) -> Result<(String, WireRef, usize), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let child =
        |wire: WireRef| -> Option<&GraphScope> { graph.source.scopes().get(&child_id(wire)) };
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| root_fact(artifact, wire);
    let levels = segmented_reduction_levels(graph, output, labels)?;
    let current = levels.last().map_or(output, |level| level.1);
    let ConcreteWireType::IndexedFamily { count: total, .. } = &types[&current] else {
        return Err("invalid helper family".into());
    };
    let helpers = total / labels;
    let original = value(current)?;
    let output_value = value(output)?;
    let mut source = String::new();
    let mut equations = Vec::new();
    for (result, previous, left_get, right_get, width, next_width) in levels.iter().rev() {
        let id = result.node.0;
        let scope_id = child_id(*result);
        let member_shape = &artifact.scope_proofs[&scope_id];
        let NodeKind::ParallelLoop(spec) = root.node(result.node).unwrap().kind() else {
            unreachable!()
        };
        let previous_value = value(*previous)?;
        let result_value = value(*result)?;
        let position = format!(
            "(⟨segment.val * {next_width} + element.val, by omega⟩ : Fin {})",
            labels * next_width
        );
        let arguments = format!(
            "(backend := backend) (hashModel := hashModel) (params := params) (i_{} := position.val) (inputs := ({previous_value})) (outputs := ({result_value}) position) (witness := member)",
            spec.index_slot
        );
        let member_fact = |node: NodeId| -> Result<String, String> {
            let name = member_shape
                .node_facts
                .get(&node)
                .and_then(|facts| facts.last())
                .ok_or("missing reducer node fact")?;
            Ok(format!("{name} {arguments} memberFacts"))
        };
        let output_fact = member_shape.output_fact.as_ref().ok_or("missing reducer output fact")?;
        source.push_str(&format!("  have sum_{id} : (∑ element : Fin {next_width}, ({result_value}) ⟨{segment}.val * {next_width} + element.val, by omega⟩) = ∑ element : Fin {width}, ({previous_value}) ⟨{segment}.val * {width} + element.val, by omega⟩ := by\n    refine segmented_pairwise_sum (segments := {labels}) (width := {width}) (fun segment element => ({previous_value}) ⟨segment.val * {width} + element.val, by omega⟩) (fun segment element => ({result_value}) ⟨segment.val * {next_width} + element.val, by omega⟩) ?_ {segment}\n    intro segment element\n    let position := {position}\n    obtain ⟨member, memberFacts⟩ := ({} backend hashModel params position.val ({previous_value}) (({result_value}) position)).mp (({}) position)\n    have leftFact := {}\n    have rightFact := {}\n    have outputFact := {output_fact} {arguments} memberFacts\n    have elementBound := element.isLt\n    have segmentBound := segment.isLt\n    have memberMod : (position.val : Int) % {next_width} = (element.val : Int) := by\n      dsimp [position]\n      omega\n    have memberDiv : (position.val : Int) / {next_width} = (segment.val : Int) := by\n      dsimp [position]\n      omega\n    have leftEq : member.w_{}_0 = ({previous_value}) ⟨segment.val * {width} + 2 * element.val, by omega⟩ := by\n      apply route_lookup_eq _ ?_ leftFact\n      change (position.val : Int) / {next_width} * {width} + (position.val : Int) % {next_width} * 2 = ((segment.val * {width} + 2 * element.val : Nat) : Int)\n      rw [memberDiv, memberMod]\n      omega\n    by_cases hright : 2 * element.val + 1 < {width}\n    · simp only [dif_pos hright]\n      have rightEq : member.w_{}_0 = ({previous_value}) ⟨segment.val * {width} + (2 * element.val + 1), by omega⟩ := by\n        apply route_lookup_eq _ ?_ rightFact\n        change (position.val : Int) / {next_width} * {width} + ((position.val : Int) % {next_width} * 2 + 1) % {width} = ((segment.val * {width} + (2 * element.val + 1) : Nat) : Int)\n        rw [memberDiv, memberMod, Int.emod_eq_of_lt (by omega) (by omega)]\n        omega\n", member_shape.runs_iff_body, fact(*result)?, member_fact(*left_get)?, member_fact(*right_get)?, left_get.0, right_get.0));
        let member_scope = child(*result).unwrap();
        let output_node = member_scope.node(member_scope.outputs()[0].node).unwrap();
        if matches!(output_node.kind(), NodeKind::Select { .. }) {
            let select_fact = member_fact(member_scope.outputs()[0].node)?;
            source.push_str(&format!("      have selectFact := {select_fact}\n      have notLast : (element.val : Int) ≠ {} := by omega\n      simp only [Int.ofNat_eq_natCast, memberMod, notLast, decide_false, Bool.false_eq_true, ↓reduceIte] at selectFact\n      exact outputFact.trans (by simpa only [MxxRuntime.matrixAdd, leftEq, rightEq] using route_select_zero selectFact)\n    · simp only [dif_neg hright]\n      have selectFact := {select_fact}\n      have last : (element.val : Int) = {} := by omega\n      simp only [Int.ofNat_eq_natCast, memberMod, last, decide_true, ↓reduceIte] at selectFact\n      exact outputFact.trans ((route_select_one selectFact).trans leftEq)\n", next_width - 1, next_width - 1));
        } else {
            source.push_str("      simpa only [MxxRuntime.matrixAdd, leftEq, rightEq] using outputFact\n    · omega\n");
        }
        equations.push(format!("sum_{id}"));
    }
    source.push_str(&format!("  have reduced : ({output_value}) {segment} = ∑ member : Fin {helpers}, ({original}) ⟨{segment}.val * {helpers} + member.val, by omega⟩ := by\n"));
    if let Some(last) = equations.last() {
        source.push_str(&format!("    have combined := {last}\n"));
        for equation in equations.iter().rev().skip(1) {
            source.push_str(&format!("    rw [{equation}] at combined\n"));
        }
        source.push_str("    simpa only [Fin.sum_univ_one, Fin.val_zero, Nat.mul_one, Nat.add_zero, Fin.eta] using combined\n");
    } else {
        if helpers != 1 {
            return Err("missing segmented reduction".into());
        }
        source.push_str(
            "    simp only [Fin.sum_univ_one, Fin.val_zero, Nat.mul_one, Nat.add_zero, Fin.eta]\n",
        );
    }
    Ok((source, current, helpers))
}

pub(super) fn normalized_execution(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    namespace: &str,
    imports: &str,
) -> Result<(String, WireRef, WireRef, WireRef, WireRef, usize, usize), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let body = shape.body_relation.as_ref().ok_or("missing readout root body")?;
    let record = shape.witness_record.as_ref().ok_or("missing readout root witness")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| root_fact(artifact, wire);
    let normalization =
        artifact
            .root
            .outputs
            .iter()
            .filter_map(|(name, output)| name.ends_with("-readout-0.3").then_some(output.wire))
            .chain(
                artifact.root.inputs.iter().filter_map(|(name, input)| {
                    name.ends_with("-readout-0.3").then_some(input.wire)
                }),
            )
            .next()
            .ok_or("missing readout normalization binding")?;
    let (normalization_node, normalization_args) = root
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(index, node)| {
            let args = root.arguments(node)?;
            (matches!(node.kind(), NodeKind::ParallelLoop(_)) && args.contains(&normalization))
                .then_some((index, args))
        })
        .ok_or("missing normalization operation")?;
    let normalized_wire = WireRef { node: NodeId(normalization_node as u64), port: Port(0) };
    let traced = *normalization_args
        .iter()
        .find(|wire| **wire != normalization)
        .ok_or("missing trace input")?;
    let traced_node = root.node(traced.node).ok_or("missing traced family")?;
    let traced_args = root.arguments(traced_node).ok_or("missing traced operands")?;
    if traced_args.len() != 2 {
        return Err("trace must add identity and helpers".into());
    }
    let ConcreteWireType::IndexedFamily { count: labels, .. } = &types[&traced] else {
        return Err("trace is not family".into());
    };
    let transformed = traced_args[0];
    let (reduction, current, helpers) =
        segmented_reduction(graph, artifact, traced_args[1], *labels, "label")?;
    let mut input_types = vec![String::new(); artifact.root.input_count];
    for input in artifact.root.inputs.values() {
        input_types[input.tuple_index] = format!("({})", input.lean_type);
    }
    let input_type = input_types.join(" × ") + " × Unit";
    let mut outputs = vec![String::new(); artifact.root.output_count];
    for output in artifact.root.outputs.values() {
        outputs[output.tuple_index] = format!("({})", output.lean_type);
    }
    let output_type = outputs.join(" × ") + " × Unit";
    let normalized = value(normalized_wire)?;
    let transformed_wire = transformed;
    let transformed = value(transformed)?;
    let original = value(current)?;
    let norm = value(normalization)?;
    let mut source = format!(
        "import {}\nimport ExponentReadout\nimport ExponentRefreshRoutes\n{imports}\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace {namespace}\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\nabbrev RootInput := {input_type}\nabbrev RootOutput := {output_type}\n\ntheorem normalized_execution {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∀ label : Fin {labels}, ({normalized}) label =\n      (({transformed}) label + ∑ helper : Fin {helpers},\n        ({original}) ⟨label.val * {helpers} + helper.val, by omega⟩) * ({norm}) := by\n  intro label\n",
        artifact.module_name,
        params = artifact.root.parameter_type
    );
    source.push_str(&reduction);
    source.push_str(&format!("  have normalizedEquation := ({}) label\n  have tracedEquation := ({}) label\n  rw [normalizedEquation, tracedEquation]\n  simp only [MxxRuntime.matrixMul, MxxRuntime.matrixAdd]\n  rw [reduced]\n\n", fact(normalized_wire)?, fact(traced)?));
    Ok((source, normalized_wire, transformed_wire, current, normalization, *labels, helpers))
}

pub(super) fn root_fact(artifact: &LeanArtifact, wire: WireRef) -> Result<String, String> {
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let name = shape
        .node_facts
        .get(&wire.node)
        .and_then(|facts| facts.last())
        .ok_or("missing root readout fact")?;
    Ok(format!(
        "{name} (backend := backend) (hashModel := hashModel) (params := params) (inputs := inputs) (outputs := outputs) (witness := witness) facts"
    ))
}

pub(super) fn root_readout_public(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing readout root body")?;
    let record = shape.witness_record.as_ref().ok_or("missing readout root witness")?;
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let child =
        |wire: WireRef| -> Option<&GraphScope> { graph.source.scopes().get(&child_id(wire)) };
    let gather = |wire: WireRef| -> Option<(WireRef, WireRef)> {
        let scope = child(wire)?;
        let output = scope.node(scope.outputs()[0].node)?;
        if !matches!(output.kind(), NodeKind::FamilyGetDynamic) {
            return None;
        }
        let args = scope.arguments(output)?;
        let parents = root.arguments(root.node(wire.node)?)?;
        let map = |arg: WireRef| {
            scope.inputs().iter().position(|input| *input == arg).map(|i| parents[i])
        };
        Some((map(args[0])?, map(args[1])?))
    };
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| root_fact(artifact, wire);
    let (mut source, normalized_wire, transformed_wire, current, normalization, labels, helpers) =
        normalized_execution(graph, artifact, "ExponentGeneratedReadoutPublic", "")?;
    let normalized = value(normalized_wire)?;
    let norm = value(normalization)?;
    let total = labels * helpers;
    let raw_node = root.node(current.node).ok_or("missing public helper product")?;
    let raw_args = root.arguments(raw_node).ok_or("missing public helper factors")?;
    if raw_args.len() != 2 {
        return Err("public helper product arity".into());
    }
    let (gather_slot, (mask_source, mask_indices)) = raw_args
        .iter()
        .enumerate()
        .find_map(|(slot, wire)| gather(*wire).map(|parts| (slot, parts)))
        .ok_or("missing public mask gather")?;
    let mask_gather = raw_args[gather_slot];
    let helper_cache = raw_args[1 - gather_slot];
    let helper_boundary = artifact
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-readout-0.2"))
        .ok_or("missing public helper cache")?
        .1
        .wire;
    if helper_cache != helper_boundary {
        return Err("public helper product uses wrong cache".into());
    }
    let transformed_args = root
        .arguments(root.node(transformed_wire.node).ok_or("missing public target action")?)
        .ok_or("missing public target operands")?;
    let target_cache = artifact
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-readout-0.0"))
        .ok_or("missing public target cache")?
        .1
        .wire;
    let program_public = *transformed_args
        .iter()
        .find(|wire| **wire != target_cache)
        .ok_or("missing public program output")?;
    if transformed_args.len() != 2 || !transformed_args.contains(&target_cache) {
        return Err("public target action does not use actual target cache".into());
    }
    let masks = value(mask_source)?;
    let gathered_masks = value(mask_gather)?;
    let indices = value(mask_indices)?;
    let cache = value(helper_cache)?;
    let program = value(program_public)?;
    let target = value(target_cache)?;
    let index_shape = &artifact.scope_proofs[&child_id(mask_indices)];
    let gather_shape = &artifact.scope_proofs[&child_id(mask_gather)];
    let NodeKind::ParallelLoop(index_spec) = root.node(mask_indices.node).unwrap().kind() else {
        unreachable!()
    };
    let NodeKind::ParallelLoop(gather_spec) = root.node(mask_gather.node).unwrap().kind() else {
        unreachable!()
    };
    let gather_node = child(mask_gather).unwrap().outputs()[0].node;
    let gather_fact =
        gather_shape.node_facts[&gather_node].last().ok_or("missing mask lookup fact")?;
    let slot = format!("(⟨label.val * {helpers} + helper.val, by omega⟩ : Fin {total})");
    let index_context = format!(
        "(backend := backend) (hashModel := hashModel) (params := params) (i_{} := {slot}.val) (inputs := ()) (outputs := ({indices}) {slot}) (witness := member)",
        index_spec.index_slot
    );
    let gather_context = format!(
        "(backend := backend) (hashModel := hashModel) (params := params) (i_{} := {slot}.val) (inputs := _) (outputs := ({gathered_masks}) {slot}) (witness := member)",
        gather_spec.index_slot
    );
    source.push_str(&format!("theorem normalized_public {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∀ label : Fin {labels}, ({normalized}) label =\n      (({program}) label * ({target}) label + ∑ helper : Fin {helpers},\n        ({masks}) helper * ({cache}) ⟨label.val * {helpers} + helper.val, by omega⟩) * ({norm}) := by\n  intro label\n  rw [normalized_execution witness facts]\n  have targetEquation := {} label\n  rw [targetEquation]\n  simp only [MxxRuntime.matrixMul]\n  congr 1\n  congr 1\n  apply Finset.sum_congr rfl\n  intro helper hmem\n  have indexEquation : ({indices}) {slot} = (helper.val : Int) := by\n    obtain ⟨member, memberFacts⟩ := ({} backend hashModel params {slot}.val () _).mp (({}) {slot})\n    have h := {} {index_context} memberFacts\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at h ⊢\n    omega\n  have gatherEquation : ({gathered_masks}) {slot} = ({masks}) helper := by\n    obtain ⟨member, memberFacts⟩ := ({} backend hashModel params {slot}.val _ _).mp (({}) {slot})\n    have hg := {gather_fact} {gather_context} memberFacts\n    have hout := {} {gather_context} memberFacts\n    exact hout.trans (route_lookup_eq helper indexEquation hg)\n  have productEquation := {} {slot}\n  simpa only [MxxRuntime.matrixMul, gatherEquation] using productEquation\n\nend ExponentGeneratedReadoutPublic\n",fact(transformed_wire)?,index_shape.runs_iff_body,fact(mask_indices)?,index_shape.output_fact.as_ref().ok_or("missing index output fact")?,gather_shape.runs_iff_body,fact(mask_gather)?,gather_shape.output_fact.as_ref().ok_or("missing gather output fact")?,fact(current)?,params=artifact.root.parameter_type));
    Ok(source)
}
