//! Actual producer public readout: segmented helper sum and final normalization.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    graph::GraphScope,
    lean::LeanArtifact,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

pub(super) fn root_readout_public(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
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
    let fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing readout constraint")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let normalization = artifact
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-readout-normalization-decomposition-0"))
        .ok_or("missing readout normalization input")?
        .1
        .wire;
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
    let mut current = traced_args[1];
    let mut levels = Vec::new();
    loop {
        let Some(scope) = child(current) else {
            break;
        };
        let Some(parents) = root.arguments(root.node(current.node).ok_or("missing reducer node")?)
        else {
            break;
        };
        if parents.len() != 2 || scope.outputs().len() != 1 {
            break;
        }
        let (Some((left_source, left_indices)), Some((right_source, right_indices))) =
            (gather(parents[0]), gather(parents[1]))
        else {
            break;
        };
        if left_source != right_source {
            break;
        }
        let ConcreteWireType::IndexedFamily { count, .. } = &types[&left_source] else {
            break;
        };
        let ConcreteWireType::IndexedFamily { count: next, .. } = &types[&current] else {
            break;
        };
        if *labels == 0 ||
            count % labels != 0 ||
            next % labels != 0 ||
            next / labels != (count / labels).div_ceil(2)
        {
            return Err("non-segmented readout reducer".into());
        }
        levels.push((
            current,
            parents[0],
            parents[1],
            left_indices,
            right_indices,
            left_source,
            count / labels,
            next / labels,
        ));
        current = left_source;
    }
    if levels.is_empty() {
        return Err("missing balanced readout helper sum".into());
    }
    let ConcreteWireType::IndexedFamily { count: total, .. } = &types[&current] else {
        return Err("invalid helper family".into());
    };
    let helpers = total / labels;
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
        "import {}\nimport ExponentReadout\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedReadoutPublic\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\nabbrev RootInput := {input_type}\nabbrev RootOutput := {output_type}\n\nprivate theorem lookup_eq {{α : Type}} {{count : Nat}} {{family : Fin count → α}} {{index : Int}} {{output : α}} (position : Fin count) (hindex : index = (position.val : Int)) (h : MxxRuntime.familyGetDynamic family index output) : output = family position := by\n  obtain ⟨actual, ha, hv⟩ := h\n  have same : actual = position := by apply Fin.ext; omega\n  simpa only [same] using hv\n\ntheorem normalized_execution {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∀ label : Fin {labels}, ({normalized}) label =\n      (({transformed}) label + ∑ helper : Fin {helpers},\n        ({original}) ⟨label.val * {helpers} + helper.val, by omega⟩) * ({norm}) := by\n  intro label\n",
        artifact.module_name,
        params = artifact.root.parameter_type
    );
    let mut equations = Vec::new();
    for (result, left, right, left_indices, right_indices, previous, width, next_width) in
        levels.iter().rev()
    {
        let count = labels * width;
        let next = labels * next_width;
        for lane in 0..*next_width {
            for (side, gather_wire, indices, position) in [
                ("l", *left, *left_indices, lane * 2),
                ("r", *right, *right_indices, (lane * 2 + 1) % width),
            ] {
                let ir = &artifact.scope_relations[&child_id(indices)];
                let gr = &artifact.scope_relations[&child_id(gather_wire)];
                let index_value = value(indices)?;
                let previous_value = value(*previous)?;
                let gather_value = value(gather_wire)?;
                let ipath = fact(indices)?;
                let gpath = fact(gather_wire)?;
                let id = result.node.0;
                source.push_str(&format!("  have idx_{id}_{lane}_{side} : ({index_value}) ⟨label.val * {next_width} + {lane}, by omega⟩ = ((label.val * {width} + {position} : Nat) : Int) := by\n    have h := {ipath} (⟨label.val * {next_width} + {lane}, by omega⟩ : Fin {next})\n    dsimp only [{ir}, {ir}.constraints_0] at h\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at h ⊢\n    omega\n  have get_{id}_{lane}_{side} : ({gather_value}) ⟨label.val * {next_width} + {lane}, by omega⟩ = ({previous_value}) ⟨label.val * {width} + {position}, by omega⟩ := by\n    have h := {gpath} (⟨label.val * {next_width} + {lane}, by omega⟩ : Fin {next})\n    dsimp only [{gr}, {gr}.constraints_0] at h\n    obtain ⟨sample, hlo, hhi, hg, hout⟩ := h\n    exact hout.trans (lookup_eq (⟨label.val * {width} + {position}, by omega⟩ : Fin {count}) idx_{id}_{lane}_{side} hg)\n"));
            }
            if lane == 0 || (2 * lane + 1) % width == 0 {
                source.push_str(&format!(
                    "  simp only [Nat.add_zero] at get_{id}_{lane}_l get_{id}_{lane}_r\n",
                    id = result.node.0
                ));
            }
            let result_value = value(*result)?;
            let previous_value = value(*previous)?;
            let rpath = fact(*result)?;
            let rr = &artifact.scope_relations[&child_id(*result)];
            let id = result.node.0;
            let left_index = lane * 2;
            let right_index = lane * 2 + 1;
            let rhs = if right_index < *width {
                format!(
                    "({previous_value}) ⟨label.val * {width} + {left_index}, by omega⟩ + ({previous_value}) ⟨label.val * {width} + {right_index}, by omega⟩"
                )
            } else {
                format!("({previous_value}) ⟨label.val * {width} + {left_index}, by omega⟩")
            };
            let eq = format!("step_{id}_{lane}");
            source.push_str(&format!("  have {eq} : ({result_value}) ⟨label.val * {next_width} + {lane}, by omega⟩ = {rhs} := by\n    have h := {rpath} (⟨label.val * {next_width} + {lane}, by omega⟩ : Fin {next})\n"));
            let is_select = matches!(
                child(*result)
                    .and_then(|scope| scope.node(scope.outputs()[0].node))
                    .map(|node| node.kind()),
                Some(NodeKind::Select { .. })
            );
            if is_select {
                source.push_str(&format!("    have modEq : ((label.val * {next_width} + {lane} : Nat) : Int) % {next_width} = {lane} := by\n      simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]\n      omega\n    norm_num [{rr}, {rr}.constraints_0, modEq, MxxRuntime.select] at h\n"));
                if right_index < *width {
                    source.push_str(&format!("    simpa only [MxxRuntime.matrixAdd, get_{id}_{lane}_l, get_{id}_{lane}_r, Nat.add_zero] using h\n"));
                } else {
                    source.push_str(&format!("    obtain ⟨position, hp, hv⟩ := h\n    have same : position = (1 : Fin 2) := by apply Fin.ext; exact_mod_cast hp\n    subst position\n    simpa only [List.get_eq_getElem, List.getElem_cons_succ, List.getElem_cons_zero, get_{id}_{lane}_l, Nat.add_zero] using hv\n"));
                }
            } else {
                source.push_str(&format!("    simpa only [{rr}, {rr}.constraints_0, MxxRuntime.matrixAdd, get_{id}_{lane}_l, get_{id}_{lane}_r, Nat.add_zero] using h\n"));
            }
            equations.push(eq);
        }
    }
    source.push_str(&format!("  have normalizedEquation := {} label\n  have tracedEquation := {} label\n  rw [normalizedEquation, tracedEquation]\n  simp only [MxxRuntime.matrixMul, MxxRuntime.matrixAdd]\n  congr 1\n  congr 1\n  have labelEq : label = (⟨label.val * 1 + 0, by omega⟩ : Fin {labels}) := by apply Fin.ext; simp\n  conv_lhs => rw [labelEq]\n  simp only [{}]\n  simp [Fin.sum_univ_succ, add_assoc]\n\n",fact(normalized_wire)?,fact(traced)?,equations.iter().rev().cloned().collect::<Vec<_>>().join(", ")));
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
        .find(|(name, _)| name.ends_with("-readout-helper-public-family-0"))
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
        .find(|(name, _)| name.ends_with("-readout-target-decomposition-0"))
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
    let ir = &artifact.scope_relations[&child_id(mask_indices)];
    let gr = &artifact.scope_relations[&child_id(mask_gather)];
    let slot = format!("(⟨label.val * {helpers} + helper.val, by omega⟩ : Fin {total})");
    source.push_str(&format!("theorem normalized_public {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∀ label : Fin {labels}, ({normalized}) label =\n      (({program}) label * ({target}) label + ∑ helper : Fin {helpers},\n        ({masks}) helper * ({cache}) ⟨label.val * {helpers} + helper.val, by omega⟩) * ({norm}) := by\n  intro label\n  rw [normalized_execution witness facts]\n  have targetEquation := {} label\n  rw [targetEquation]\n  simp only [MxxRuntime.matrixMul]\n  congr 1\n  congr 1\n  apply Finset.sum_congr rfl\n  intro helper hmem\n  have indexEquation : ({indices}) {slot} = (helper.val : Int) := by\n    have h := {} {slot}\n    dsimp only [{ir}, {ir}.constraints_0] at h\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at h ⊢\n    omega\n  have gatherEquation : ({gathered_masks}) {slot} = ({masks}) helper := by\n    have h := {} {slot}\n    dsimp only [{gr}, {gr}.constraints_0] at h\n    obtain ⟨sample, hlo, hhi, hg, hout⟩ := h\n    exact hout.trans (lookup_eq helper indexEquation hg)\n  have productEquation := {} {slot}\n  simpa only [MxxRuntime.matrixMul, gatherEquation] using productEquation\n\nend ExponentGeneratedReadoutPublic\n",fact(transformed_wire)?,fact(mask_indices)?,fact(mask_gather)?,fact(current)?,params=artifact.root.parameter_type));
    Ok(source)
}
