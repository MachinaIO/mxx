//! Producer key recurrences from actual segmented gather/add reductions.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::LeanArtifact,
    node::{IntBinaryOp, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use std::collections::{BTreeMap, BTreeSet};

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
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let child = |wire: WireRef| graph.source.scopes().get(&child_id(wire));
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| -> Result<String, String> {
        Ok(format!(
            "facts{}.1",
            ".2".repeat(
                shape.node_constraints[&wire.node].clone().last().ok_or("missing key fact")?
            )
        ))
    };
    let family_count = |wire: WireRef| -> Result<usize, String> {
        match &types[&wire] {
            ConcreteWireType::IndexedFamily { count, .. } => Ok(*count),
            _ => Err("key operation not family".into()),
        }
    };
    let gather = |wire: WireRef| -> Option<(WireRef, WireRef)> {
        let scope = child(wire)?;
        let node = scope.node(scope.outputs()[0].node)?;
        if !matches!(node.kind(), NodeKind::FamilyGetDynamic) {
            return None;
        }
        let args = scope.arguments(node)?;
        let captures = root.arguments(root.node(wire.node)?)?;
        let map = |arg: WireRef| scope.inputs().iter().position(|w| *w == arg).map(|i| captures[i]);
        Some((map(args[0])?, map(args[1])?))
    };
    let mut source = String::from(
        "import RootProgramSetup\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramKeys\nopen Mxx.Primitives\nopen scoped BigOperators\n\nprivate theorem lookup_eq {α : Type} {count : Nat} {f : Fin count → α} {index : Int} {output : α} (position : Fin count) (hindex : index = (position.val : Int)) (h : MxxRuntime.familyGetDynamic f index output) : output = f position := by\n  obtain ⟨actual, ha, hv⟩ := h\n  have he : actual = position := by apply Fin.ext; omega\n  simpa only [he] using hv\n\n",
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
        let mut product = None;
        for (i, node) in root.nodes().iter().enumerate() {
            let wire = WireRef { node: NodeId(i as u64), port: Port(0) };
            let Some(scope) = child(wire) else {
                continue;
            };
            let Some(output) = scope.node(scope.outputs()[0].node) else {
                continue;
            };
            if !matches!(
                output.kind(),
                NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
            ) {
                continue;
            }
            let args = scope.arguments(output).ok_or_else(fail)?;
            let captures = root.arguments(node).ok_or_else(fail)?;
            let map =
                |arg: WireRef| scope.inputs().iter().position(|w| *w == arg).map(|i| captures[i]);
            if map(args[1]) == Some(cache.wire) {
                product = Some((wire, map(args[0]).ok_or_else(fail)?));
                break;
            }
        }
        let (product, public_gather) = product.ok_or_else(fail)?;
        let (public_inputs, public_indices) = gather(public_gather).ok_or_else(fail)?;
        let global_count = family_count(public_inputs)?;
        if width == 0 || global_count % width != 0 {
            return Err(fail());
        }
        let raw_count = global_count / width;
        let mut pending = vec![public_indices];
        let mut index_nodes = BTreeSet::new();
        while let Some(wire) = pending.pop() {
            if !index_nodes.insert(wire) {
                continue;
            }
            let node = root.node(wire.node).ok_or_else(fail)?;
            if child(wire).is_none() {
                return Err(fail());
            }
            for arg in root.arguments(node).ok_or_else(fail)? {
                pending.push(arg);
            }
        }
        let offset = index_nodes
            .iter()
            .find_map(|wire| {
                let scope = child(*wire)?;
                let node = scope.node(scope.outputs()[0].node)?;
                if !matches!(node.kind(), NodeKind::IntBinary(IntBinaryOp::Add)) {
                    return None;
                }
                scope.arguments(node)?.iter().find_map(|arg| match scope.node(arg.node)?.kind() {
                    NodeKind::ConstantInt(v) => usize::try_from(v).ok(),
                    _ => None,
                })
            })
            .ok_or_else(fail)?;
        if offset + members > raw_count {
            return Err(fail());
        }
        let mut current = product;
        let mut member_width = members;
        let mut levels = Vec::new();
        while member_width > 1 {
            let next = member_width.div_ceil(2);
            let mut found = None;
            for (i, node) in root.nodes().iter().enumerate() {
                let result = WireRef { node: NodeId(i as u64), port: Port(0) };
                if child(result).is_none() {
                    continue;
                }
                let Some(args) = root.arguments(node) else {
                    continue;
                };
                if args.len() != 2 {
                    continue;
                }
                let (Some((ls, li)), Some((rs, ri))) = (gather(args[0]), gather(args[1])) else {
                    continue;
                };
                if ls == current && rs == current && family_count(result)? == width * next {
                    found = Some((result, args[0], args[1], li, ri, current, member_width, next));
                    break;
                }
            }
            let level = found.ok_or_else(fail)?;
            current = level.0;
            member_width = next;
            levels.push(level);
        }
        let number = stages.len();
        let theorem = format!("key_after_{number}");
        let actual = value(current)?;
        let inputs = value(public_inputs)?;
        let cached = value(cache.wire)?;
        source.push_str(&format!("-- Actual producer key following cache {name:?}.\ntheorem {theorem} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedPrivateRoot.RootInput}} {{outputs : ExponentGeneratedPrivateRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (lane : Fin {width}) :\n    ({actual}) lane = ∑ member : Fin {members},\n      ({inputs}) ⟨lane.val * {raw_count} + {offset} + member.val, by clear * - lane; omega⟩ *\n      ({cached}) ⟨lane.val * {members} + member.val, by clear * - lane; omega⟩ := by\n",params=artifact.root.parameter_type));
        let mut equations = Vec::new();
        for member in 0..members {
            source.push_str(&format!("  let input_{member} : Fin {lanes} := ⟨lane.val * {members} + {member}, by clear * - lane; omega⟩\n"));
            let mut index_eqs = Vec::new();
            for wire in &index_nodes {
                let relation = &artifact.scope_relations[&child_id(*wire)];
                let eq = format!("index_{}_{member}", wire.node.0);
                source.push_str(&format!("  have {eq} := {} input_{member}\n  norm_num [{relation}, {relation}.constraints_0] at {eq}\n",fact(*wire)?));
                index_eqs.push(eq);
            }
            let gr = &artifact.scope_relations[&child_id(public_gather)];
            source.push_str(&format!("  have input_index_{member} : ({}) input_{member} = ((lane.val * {raw_count} + {offset} + {member} : Nat) : Int) := by\n    simp only [{}]\n    dsimp only [input_{member}]\n    clear * - lane\n    omega\n  have input_get_{member} : ({}) input_{member} = ({inputs}) ⟨lane.val * {raw_count} + {offset} + {member}, by clear * - lane; omega⟩ := by\n    have h := {} input_{member}\n    dsimp only [{gr}, {gr}.constraints_0] at h\n    obtain ⟨sample, _, _, hg, hout⟩ := h\n    exact hout.trans (lookup_eq _ input_index_{member} hg)\n",value(public_indices)?,index_eqs.iter().rev().cloned().collect::<Vec<_>>().join(", "),value(public_gather)?,fact(public_gather)?));
            let pr = &artifact.scope_relations[&child_id(product)];
            let eq = format!("product_{member}");
            source.push_str(&format!("  have {eq} : ({}) ⟨lane.val * {members} + {member}, by clear * - lane; omega⟩ = ({inputs}) ⟨lane.val * {raw_count} + {offset} + {member}, by clear * - lane; omega⟩ * ({cached}) input_{member} := by\n    have h := {} input_{member}\n    simpa only [{pr}, {pr}.constraints_0, MxxRuntime.matrixMul, input_get_{member}] using h\n",value(product)?,fact(product)?));
            equations.push(eq);
        }
        for (result, left, right, li, ri, previous, previous_width, next) in &levels {
            for member in 0..*next {
                let id = result.node.0;
                let next_count = width * next;
                let prev_count = width * previous_width;
                source.push_str(&format!("  let slot_{id}_{member} : Fin {next_count} := ⟨lane.val * {next} + {member}, by clear * - lane; omega⟩\n"));
                for (side, gather_wire, index_wire, position) in [
                    ("l", *left, *li, 2 * member),
                    ("r", *right, *ri, (2 * member + 1) % previous_width),
                ] {
                    let ir = &artifact.scope_relations[&child_id(index_wire)];
                    let gr = &artifact.scope_relations[&child_id(gather_wire)];
                    source.push_str(&format!("  have index_{id}_{member}_{side} : ({}) slot_{id}_{member} = ((lane.val * {previous_width} + {position} : Nat) : Int) := by\n    have h := {} slot_{id}_{member}\n    norm_num [{ir}, {ir}.constraints_0] at h\n    rw [h]\n    dsimp only [slot_{id}_{member}]\n    clear * - lane\n    omega\n  have get_{id}_{member}_{side} : ({}) slot_{id}_{member} = ({}) ⟨lane.val * {previous_width} + {position}, by clear * - lane; omega⟩ := by\n    have h := {} slot_{id}_{member}\n    dsimp only [{gr}, {gr}.constraints_0] at h\n    obtain ⟨sample, _, _, hg, hout⟩ := h\n    exact hout.trans (lookup_eq (⟨lane.val * {previous_width} + {position}, by clear * - lane; omega⟩ : Fin {prev_count}) index_{id}_{member}_{side} hg)\n",value(index_wire)?,fact(index_wire)?,value(gather_wire)?,value(*previous)?,fact(gather_wire)?));
                }
                let left_term = format!(
                    "({}) ⟨lane.val * {previous_width} + {}, by clear * - lane; omega⟩",
                    value(*previous)?,
                    2 * member
                );
                let rhs = if 2 * member + 1 < *previous_width {
                    format!(
                        "{left_term} + ({}) ⟨lane.val * {previous_width} + {}, by clear * - lane; omega⟩",
                        value(*previous)?,
                        2 * member + 1
                    )
                } else {
                    left_term
                };
                let eq = format!("step_{id}_{member}");
                let rr = &artifact.scope_relations[&child_id(*result)];
                let scope = child(*result).unwrap();
                let select = matches!(
                    scope.node(scope.outputs()[0].node).unwrap().kind(),
                    NodeKind::Select { .. }
                );
                source.push_str(&format!("  have {eq} : ({}) ⟨lane.val * {next} + {member}, by clear * - lane; omega⟩ = {rhs} := by\n    have h := {} slot_{id}_{member}\n",value(*result)?,fact(*result)?));
                if select {
                    source.push_str(&format!("    have hm : ((slot_{id}_{member}.val : Nat) : Int) % {next} = {member} := by dsimp only [slot_{id}_{member}]; clear * - lane; omega\n    norm_num [{rr}, {rr}.constraints_0, MxxRuntime.select, hm] at h\n"));
                    if 2 * member + 1 == *previous_width {
                        source.push_str(&format!("    obtain ⟨position, hposition, hvalue⟩ := h\n    have hp : position = (1 : Fin 2) := by\n      apply Fin.ext\n      change position.val = 1\n      exact_mod_cast hposition\n    subst position\n    simpa only [List.get_eq_getElem, List.getElem_cons_succ, List.getElem_cons_zero, get_{id}_{member}_l] using hvalue\n"));
                    } else {
                        source.push_str(&format!("    simpa only [get_{id}_{member}_l, get_{id}_{member}_r, MxxRuntime.matrixAdd] using h\n"));
                    }
                } else {
                    source.push_str(&format!("    simpa only [{rr}, {rr}.constraints_0, MxxRuntime.matrixAdd, get_{id}_{member}_l, get_{id}_{member}_r] using h\n"));
                }
                equations.push(eq);
            }
        }
        let final_slot = levels
            .last()
            .map(|level| format!("slot_{}_0", level.0.node.0))
            .unwrap_or("input_0".into());
        source.push_str(&format!("  have last : lane = {final_slot} := by apply Fin.ext; simp [{final_slot}]\n  conv_lhs => rw [last]\n  dsimp only [{final_slot}]\n  simp only [{}]\n  simp [Fin.sum_univ_succ, add_assoc{}]\n\n",equations.iter().rev().cloned().collect::<Vec<_>>().join(", "),(0..members).map(|i|format!(", input_{i}")).collect::<String>()));
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
