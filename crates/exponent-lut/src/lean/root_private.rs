//! Private selector caches projected from the same preprocessing-root witness.

use std::collections::BTreeMap;

use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{BoundaryValue, LeanArtifact},
    node::{ConcatAxis, LoopInputMode, NodeKind},
    types::{ConcreteWireType, WireRef},
};

use super::root_inputs::value_expression;

fn boundary_type(values: &BTreeMap<String, BoundaryValue>, count: usize) -> String {
    let mut types = vec![String::new(); count];
    for value in values.values() {
        types[value.tuple_index] = value.lean_type.clone();
    }
    match count {
        0 => "Unit".into(),
        1 => types.remove(0),
        _ => format!(
            "{} × Unit",
            types.iter().map(|ty| format!("({ty})")).collect::<Vec<_>>().join(" × ")
        ),
    }
}

/// Follow the actual decomposition, gather and sampler operands. Unsupported
/// cache shapes fail closed instead of acquiring an assumed cache equation.
pub(super) fn root_private(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<(String, BTreeMap<String, String>), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing preprocessing root body")?;
    let witness_type = shape.witness_record.as_ref().ok_or("missing preprocessing witness")?;
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|range| range.clone().last())
            .ok_or("missing root node constraint")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let mut source = format!(
        "import {}\nimport SelectorSetupScopes\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedPrivateRoot\nopen Mxx.Primitives MxxRuntime MxxExponentLut ExponentGeneratedSetup\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        artifact.module_name,
        boundary_type(&artifact.root.inputs, artifact.root.input_count),
        boundary_type(&artifact.root.outputs, artifact.root.output_count),
    );
    let mut theorems = BTreeMap::new();
    for (name, output) in &artifact.root.outputs {
        if !name.contains("-program-cache-family-") || !name.ends_with("-body-0-k-rhs") {
            continue;
        }
        let fail = || format!("unsupported actual private cache shape at {name}");
        let decomp_node = root.node(output.wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(decomp_loop) = decomp_node.kind() else { return Err(fail()) };
        if decomp_loop.input_modes != [LoopInputMode::Zip] {
            return Err(fail());
        }
        let decomp_scope_id = child_id(output.wire);
        let decomp_scope = graph.source.scopes().get(&decomp_scope_id).ok_or_else(fail)?;
        if decomp_scope.nodes().len() != 2 ||
            decomp_scope.inputs().len() != 1 ||
            decomp_scope.outputs().len() != 1
        {
            return Err(fail());
        }
        let decomp = decomp_scope.node(decomp_scope.outputs()[0].node).ok_or_else(fail)?;
        let NodeKind::GadgetDecompose { base, digit_count, .. } = decomp.kind() else {
            return Err(fail())
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let gather_wire =
            root.arguments(decomp_node).and_then(|a| a.first().copied()).ok_or_else(fail)?;
        let gather_node = root.node(gather_wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(gather_loop) = gather_node.kind() else { return Err(fail()) };
        let gather_scope_id = child_id(gather_wire);
        let gather_scope = graph.source.scopes().get(&gather_scope_id).ok_or_else(fail)?;
        if gather_scope.nodes().len() != 3 ||
            gather_scope.inputs().len() != 2 ||
            gather_scope.outputs().len() != 1
        {
            return Err(fail());
        }
        let get = gather_scope.node(gather_scope.outputs()[0].node).ok_or_else(fail)?;
        if !matches!(get.kind(), NodeKind::FamilyGetDynamic) {
            return Err(fail());
        }
        let get_args = gather_scope.arguments(get).ok_or_else(fail)?;
        let family_slot =
            gather_scope.inputs().iter().position(|w| *w == get_args[0]).ok_or_else(fail)?;
        let index_slot =
            gather_scope.inputs().iter().position(|w| *w == get_args[1]).ok_or_else(fail)?;
        if gather_loop.input_modes[family_slot] != LoopInputMode::Broadcast ||
            gather_loop.input_modes[index_slot] != LoopInputMode::Zip
        {
            return Err(fail());
        }
        let gather_args = root.arguments(gather_node).ok_or_else(fail)?;
        let sampler_wire = gather_args[family_slot];
        let indices = value(gather_args[index_slot])?;
        let sampler_node = root.node(sampler_wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(sampler_loop) = sampler_node.kind() else { return Err(fail()) };
        let sampler_id = child_id(sampler_wire);
        let sampler = graph.source.scopes().get(&sampler_id).ok_or_else(fail)?;
        if sampler.nodes().len() != 13 ||
            sampler.inputs().len() != 3 ||
            sampler.outputs().len() != 1
        {
            return Err(fail());
        }
        // The setup exporter emits this unique sampler as selector_setup_0.
        // Reject ambiguity rather than guessing another theorem's numbering.
        let sampler_count = graph.source.scopes().iter().filter(|(id, scope)|
            matches!(id, FrozenGraphScopeId::ParallelBody { parent, .. } if matches!(parent.as_ref(), FrozenGraphScopeId::Root)) &&
            scope.nodes().len() == 13 && scope.inputs().len() == 3 && scope.outputs().len() == 1).count();
        if sampler_count != 1 {
            return Err("ambiguous selector sampler numbering".into());
        }
        let sampler_types = &graph.scopes[&sampler_id].wire_types;
        let secret_slot = sampler
            .inputs()
            .iter()
            .position(|wire| {
                sampler_types[wire].matrix_type().is_some_and(|m| m.rows == 1 && m.columns == 2)
            })
            .ok_or_else(fail)?;
        let payload_slot = sampler
            .inputs()
            .iter()
            .position(|wire| {
                sampler_types[wire].matrix_type().is_some_and(|m| m.rows == 1 && m.columns == 1)
            })
            .ok_or_else(fail)?;
        if sampler_loop.input_modes[secret_slot] != LoopInputMode::Broadcast ||
            sampler_loop.input_modes[payload_slot] != LoopInputMode::Zip
        {
            return Err(fail());
        }
        let sampler_args = root.arguments(sampler_node).ok_or_else(fail)?;
        let secret_wire = sampler_args[secret_slot];
        let secret_node = root.node(secret_wire.node).ok_or_else(fail)?;
        if !matches!(secret_node.kind(), NodeKind::Concat { axis: ConcatAxis::Columns }) {
            return Err(fail());
        }
        let secret = value(secret_wire)?;
        let payload = value(sampler_args[payload_slot])?;
        let cutoff = sampler
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                    max_coefficient_bound.evaluate(&ParamEnv::default()).ok()
                }
                _ => None,
            })
            .ok_or_else(fail)?;
        let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
            return Err(fail())
        };
        let matrix = element.matrix_type().ok_or_else(fail)?;
        let ConcreteWireType::IndexedFamily { count: raw_count, .. } = &types[&sampler_wire] else {
            return Err(fail())
        };
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let cached = value(output.wire)?;
        let decomp_fact = fact(output.wire)?;
        let gather_fact = fact(gather_wire)?;
        let sampler_fact = fact(sampler_wire)?;
        let secret_fact = fact(secret_wire)?;
        let number = theorems.len();
        let theorem = format!("private_cache_{number}");
        source.push_str(&format!(
            "-- Actual exported cache {name:?}.\ntheorem {theorem} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type})\n    (facts : {body} backend hashModel params inputs outputs witness)\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, ∃ position : Fin {raw_count}, ∃ error : ErrorMatrix {n} 1 {columns},\n    ∃ action : ErrorMatrix {n} {columns} {columns},\n      (position.val : Int) = ({indices}) i ∧\n      ({cached}) i = reduceMatrix {q} {n} {columns} {columns} action ∧\n      ActionGain action (denseRegularGain layout {n} 2) ∧\n      ({secret}) * gadget * (({cached}) i) = (({payload}) position 0 0) • (({secret}) * gadget) + reduceMatrix {q} {n} 1 {columns} error ∧\n      CoeffBound error ({cutoff} : Nat) := by\n  have augmented := augmented_source_last {secret_fact} (by rfl)\n  intro i\n  obtain ⟨selected, hlo, hhi, hget, hout⟩ := {gather_fact} i\n  obtain ⟨position, hindex, hselected⟩ := hget\n  obtain ⟨decomposition, hd, hdout⟩ := {decomp_fact} i\n  rw [hout, hselected] at hd\n  obtain ⟨other, error, action, hg, ha, hgain, hcache, hb⟩ :=\n    selector_cache_0 layout lookup ({sampler_fact} position) augmented hd\n  have same : other = gadget := gadget_runs_deterministic hg gadgetRuns\n  subst other\n  exact ⟨position, error, action, hindex, hdout.trans ha, hgain, by rw [hdout]; exact hcache, hb⟩\n\n",
            params = artifact.root.parameter_type,
        ));
        let index_wire = gather_args[index_slot];
        let index_node = root.node(index_wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(index_loop) = index_node.kind() else {
            return Err(fail());
        };
        let index_scope_id = child_id(index_wire);
        let index_scope = &graph.source.scopes()[&index_scope_id];
        let sum = index_scope.node(index_scope.outputs()[0].node).ok_or_else(fail)?;
        if !matches!(sum.kind(), NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Add)) {
            return Err(fail());
        }
        let sum_args = index_scope.arguments(sum).ok_or_else(fail)?;
        let offset = sum_args
            .iter()
            .find_map(|arg| match index_scope.node(arg.node)?.kind() {
                NodeKind::ConstantInt(value) => usize::try_from(value).ok(),
                _ => None,
            })
            .ok_or_else(fail)?;
        if !sum_args.iter().any(|arg| matches!(index_scope.node(arg.node).map(|n|n.kind()), Some(NodeKind::EvaluateInt(mxx_ir_core::IntExpr::LoopIndex(slot))) if *slot == index_loop.index_slot)) || offset + lanes > *raw_count { return Err(fail()); }
        let index_fact = fact(index_wire)?;
        source.push_str(&format!("theorem private_bucket_{number} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type}) (facts : {body} backend hashModel params inputs outputs witness)\n    (layout : MxxRuntime.RegularLayout {q}) (lookup : backend.regularLayout {q} {n} = some layout)\n    (gadget : ExactMatrix {q} {n} 2 {columns}) (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, ∃ error : ErrorMatrix {n} 1 {columns}, ∃ action : ErrorMatrix {n} {columns} {columns},\n      ({cached}) i = reduceMatrix {q} {n} {columns} {columns} action ∧\n      ActionGain action (denseRegularGain layout {n} 2) ∧\n      ({secret}) * gadget * (({cached}) i) = (({payload}) ⟨{offset} + i.val, by omega⟩ 0 0) • (({secret}) * gadget) + reduceMatrix {q} {n} 1 {columns} error ∧\n      CoeffBound error ({cutoff} : Nat) := by\n  intro i\n  obtain ⟨position,error,action,hindex,ha,hgain,hcache,hbound⟩ := {theorem} witness facts layout lookup gadget gadgetRuns i\n  have indexEquation : ({indices}) i = (i.val : Int) + {offset} := {index_fact} i\n  have same : position = (⟨{offset} + i.val, by omega⟩ : Fin {raw_count}) := by\n    apply Fin.ext\n    change position.val = {offset} + i.val\n    omega\n  rw [same] at hcache\n  exact ⟨error,action,ha,hgain,hcache,hbound⟩\n\n",params=artifact.root.parameter_type));
        theorems.insert(name.clone(), format!("ExponentGeneratedPrivateRoot.{theorem}"));
    }
    if theorems.is_empty() {
        return Err("no actual private selector caches found".into());
    }
    source.push_str("end ExponentGeneratedPrivateRoot\n");
    Ok((source, theorems))
}
