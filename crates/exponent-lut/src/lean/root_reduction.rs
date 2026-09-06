//! Exact balanced-family reductions projected from the actual program witness.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    graph::GraphScope,
    lean::LeanArtifact,
    node::{MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

fn matrix_type(ty: &ConcreteWireType) -> Result<String, String> {
    if let Some(matrix) = ty.matrix_type() {
        return Ok(format!(
            "Mxx.Primitives.ExactMatrix {} {} {} {}",
            matrix.modulus, matrix.ring_dimension, matrix.rows, matrix.columns
        ));
    }
    if let ConcreteWireType::IndexedFamily { element, count } = ty {
        return Ok(format!("Fin {count} → {}", matrix_type(element)?));
    }
    Err("unsupported reduction boundary".into())
}

pub(super) fn root_reduction(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let outer = graph.source.root_scope();
    let cache = artifact
        .root
        .inputs
        .iter()
        .find(|(name, _)| {
            name.contains("-program-cache-family-") && name.ends_with("-body-0-k-rhs")
        })
        .ok_or("missing program cache")?
        .1;
    let owner = outer
        .nodes()
        .iter()
        .position(|node| {
            matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                outer.arguments(node).is_some_and(|args| args.contains(&cache.wire))
        })
        .ok_or("missing program loop")?;
    let scope_id = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: NodeId(owner as u64),
    };
    let root = &graph.source.scopes()[&scope_id];
    let shape = &artifact.scope_proofs[&scope_id];
    let types = &graph.scopes[&scope_id].wire_types;
    let body = shape.body_relation.as_ref().ok_or("missing reduction body")?;
    let record = shape.witness_record.as_ref().ok_or("missing reduction record")?;
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(scope_id.clone()),
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
    let path = |wire: WireRef| -> Result<String, String> {
        Ok(format!(
            "facts{}.1",
            ".2".repeat(
                shape.node_constraints[&wire.node]
                    .clone()
                    .last()
                    .ok_or("missing reduction constraint")?
            )
        ))
    };
    let input_type = root
        .inputs()
        .iter()
        .map(|wire| matrix_type(&types[wire]).map(|ty| format!("({ty})")))
        .collect::<Result<Vec<_>, _>>()?
        .join(" × ") +
        " × Unit";
    let mut source = format!(
        "import {}\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedReduction\nopen Mxx.Primitives\nopen scoped BigOperators\nabbrev RootInput := {input_type}\nabbrev RootOutput := {}\n\nprivate theorem lookup_eq {{α : Type}} {{count : Nat}} {{f : Fin count → α}} {{index : Int}} {{output : α}} (position : Fin count) (hindex : index = (position.val : Int)) (h : MxxRuntime.familyGetDynamic f index output) : output = f position := by\n  obtain ⟨actual, ha, hv⟩ := h\n  have he : actual = position := by apply Fin.ext; omega\n  simpa only [he] using hv\n\nprivate theorem singleton_sum {{α : Type}} [AddCommMonoid α] {{f : Fin 1 → α}} {{output : α}} (h : MxxRuntime.familyGetStatic f 0 output) : output = ∑ i, f i := by\n  simpa using lookup_eq (0 : Fin 1) rfl h\n\n",
        artifact.module_name,
        matrix_type(&types[&root.outputs()[0]])?
    );
    let mut total = 0;
    for (index, node) in root.nodes().iter().enumerate() {
        if !matches!(node.kind(), NodeKind::FamilyGetStatic { .. }) {
            continue;
        }
        let output = WireRef { node: NodeId(index as u64), port: Port(0) };
        let final_family = root.arguments(node).ok_or("missing final gather args")?[0];
        let mut current = final_family;
        let mut levels = Vec::new();
        loop {
            let Some(scope) = child(current) else { break };
            let Some(parents) = root.arguments(root.node(current.node).unwrap()) else { break };
            if parents.len() != 2 || scope.outputs().len() != 1 {
                break;
            }
            let (Some((left_source, left_indices)), Some((right_source, right_indices))) =
                (gather(parents[0]), gather(parents[1]))
            else {
                break
            };
            if left_source != right_source {
                break;
            }
            let ConcreteWireType::IndexedFamily { count, .. } = &types[&left_source] else { break };
            let ConcreteWireType::IndexedFamily { count: next, .. } = &types[&current] else {
                break
            };
            if *next != count.div_ceil(2) {
                break;
            }
            levels.push((
                current,
                parents[0],
                parents[1],
                left_indices,
                right_indices,
                left_source,
                *count,
                *next,
            ));
            current = left_source;
        }
        if levels.is_empty() {
            // A singleton sum is the actual getter of a one-lane added family.
            let singleton_add = matches!(
                &types[&final_family],
                ConcreteWireType::IndexedFamily { count: 1, .. }
            ) && child(final_family)
                .and_then(|scope| scope.node(scope.outputs()[0].node))
                .is_some_and(|node| {
                    matches!(node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add))
                });
            if !singleton_add {
                continue;
            }
        }
        let ConcreteWireType::IndexedFamily { count: lanes, .. } = &types[&current] else {
            unreachable!()
        };
        let actual = value(output)?;
        let original = value(current)?;
        source.push_str(&format!("-- Actual reduction ending at node {index}, with {lanes} source lanes.\ntheorem reduction_{total} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (i_0 : Nat) (witness : {record}) (facts : {body} backend hashModel params i_0 inputs outputs witness) :\n    ({actual}) = ∑ i : Fin {lanes}, ({original}) i := by\n",artifact.root.parameter_type));
        let mut equations = Vec::new();
        for (result, left, right, left_indices, right_indices, previous, count, next) in
            levels.iter().rev()
        {
            for lane in 0..*next {
                for (side, gather_wire, indices, position) in [
                    ("l", *left, *left_indices, 2 * lane),
                    ("r", *right, *right_indices, (2 * lane + 1) % count),
                ] {
                    let ir = &artifact.scope_relations[&child_id(indices)];
                    let gr = &artifact.scope_relations[&child_id(gather_wire)];
                    source.push_str(&format!("  have idx_{id}_{lane}_{side} : ({indices_value}) {lane} = ({position} : Int) := by\n    have h := {ipath} ({lane} : Fin {next})\n    norm_num [{ir}, {ir}.constraints_0] at h\n    exact h\n  have get_{id}_{lane}_{side} : ({gather_value}) {lane} = ({previous_value}) {position} := by\n    have h := {gpath} ({lane} : Fin {next})\n    dsimp only [{gr}, {gr}.constraints_0] at h\n    obtain ⟨sample, _, _, hg, hout⟩ := h\n    exact hout.trans (lookup_eq ({position} : Fin {count}) idx_{id}_{lane}_{side} hg)\n",id=result.node.0,indices_value=value(indices)?,ipath=path(indices)?,gather_value=value(gather_wire)?,previous_value=value(*previous)?,gpath=path(gather_wire)?));
                }
                let rr = &artifact.scope_relations[&child_id(*result)];
                let rhs = if 2 * lane + 1 < *count {
                    format!(
                        "({}) {} + ({}) {}",
                        value(*previous)?,
                        2 * lane,
                        value(*previous)?,
                        2 * lane + 1
                    )
                } else {
                    format!("({}) {}", value(*previous)?, 2 * lane)
                };
                let eq = format!("step_{}_{}", result.node.0, lane);
                let is_select = matches!(
                    child(*result)
                        .unwrap()
                        .node(child(*result).unwrap().outputs()[0].node)
                        .unwrap()
                        .kind(),
                    NodeKind::Select { .. }
                );
                source.push_str(&format!("  have {eq} : ({}) {lane} = {rhs} := by\n    have h := {} ({lane} : Fin {next})\n",value(*result)?,path(*result)?));
                if is_select {
                    source.push_str(&format!(
                        "    norm_num [{rr}, {rr}.constraints_0, MxxRuntime.select] at h\n"
                    ));
                    if 2 * lane + 1 == *count {
                        source.push_str(&format!("    obtain ⟨position, hposition, hvalue⟩ := h\n    have hp : position = (1 : Fin 2) := by\n      apply Fin.ext\n      change position.val = 1\n      exact_mod_cast hposition\n    subst position\n    simpa only [List.get_eq_getElem, List.getElem_cons_succ, List.getElem_cons_zero, get_{id}_{lane}_l] using hvalue\n", id=result.node.0));
                    } else {
                        source.push_str(&format!("    simpa only [get_{id}_{lane}_l, get_{id}_{lane}_r, MxxRuntime.matrixAdd] using h\n", id=result.node.0));
                    }
                } else {
                    source.push_str(&format!("    simpa only [{rr}, {rr}.constraints_0, MxxRuntime.matrixAdd, get_{id}_{lane}_l, get_{id}_{lane}_r] using h\n",id=result.node.0));
                }
                equations.push(eq);
            }
        }
        if equations.is_empty() {
            source.push_str(&format!("  exact singleton_sum {}\n\n", path(output)?));
        } else {
            source.push_str(&format!("  have last := lookup_eq (0 : Fin 1) rfl {}\n  rw [last]\n  simp only [{}]\n  simp [Fin.sum_univ_succ, add_assoc]\n\n", path(output)?, equations.iter().rev().cloned().collect::<Vec<_>>().join(", ")));
        }
        total += 1;
    }
    let expected = artifact
        .root
        .inputs
        .keys()
        .filter(|name| name.contains("-program-cache-family-") && name.ends_with("-body-0-k-rhs"))
        .count();
    if total != expected {
        return Err(format!("expected {expected} program reductions, found {total}"));
    }
    source.push_str("end ExponentGeneratedReduction\n");
    Ok(source)
}
