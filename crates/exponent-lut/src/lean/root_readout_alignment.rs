//! Readout public-helper alignment from the actual preprocessing-root operands.

use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, ValidatedGraph,
    lean::{BoundaryValue, LeanArtifact},
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, WireRef},
};
use std::collections::BTreeMap;

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

// Preserve the graph expression, including Select's actual table order. Other
// expression forms fail closed until their lexical binding is supported here.
fn index_expression(expr: &IntExpr, slot: u32) -> Result<String, String> {
    Ok(match expr {
        IntExpr::Const(value) => value.to_string(),
        IntExpr::LoopIndex(index) if *index == slot => "(Int.ofNat i.val)".into(),
        IntExpr::Add(a, b) => {
            format!("({} + {})", index_expression(a, slot)?, index_expression(b, slot)?)
        }
        IntExpr::Rem(a, b) => {
            format!("Int.fmod ({}) ({})", index_expression(a, slot)?, index_expression(b, slot)?)
        }
        IntExpr::Select { selector, branches } => format!(
            "MxxIR.selectInt ({}) [{}]",
            index_expression(selector, slot)?,
            branches
                .iter()
                .map(|branch| index_expression(branch, slot).map(|s| format!("({s})")))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
        _ => return Err("unsupported readout automorphism index expression".into()),
    })
}

pub(super) fn root_readout_alignment(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<(String, BTreeMap<String, String>), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing preprocessing body")?;
    let witness_type = shape.witness_record.as_ref().ok_or("missing preprocessing witness")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing root constraint")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let mut source = format!(
        "import {}\nimport ExponentSetup\nimport RuntimeDecomposition\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedReadoutAlignment\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        artifact.module_name,
        boundary_type(&artifact.root.inputs, artifact.root.input_count),
        boundary_type(&artifact.root.outputs, artifact.root.output_count)
    );
    let mut theorems = BTreeMap::new();
    for (name, output) in &artifact.root.outputs {
        if !name.ends_with("-readout-helper-public-family-0") {
            continue;
        }
        let fail = || format!("unsupported readout alignment graph at {name}");
        let node = root.node(output.wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(loop_) = node.kind() else {
            return Err(fail());
        };
        if loop_.input_modes != [LoopInputMode::Zip, LoopInputMode::Zip] {
            return Err(fail());
        }
        let id = child_id(output.wire);
        let child = &graph.source.scopes()[&id];
        let proof = &artifact.scope_proofs[&id];
        if child.nodes().len() != 5 || child.inputs().len() != 2 || child.outputs().len() != 1 {
            return Err(fail());
        }
        let decompose = child.node(child.outputs()[0].node).ok_or_else(fail)?;
        let NodeKind::GadgetDecompose { base, digit_count, .. } = decompose.kind() else {
            return Err(fail());
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let product_wire =
            child.arguments(decompose).and_then(|a| a.first().copied()).ok_or_else(fail)?;
        let product = child.node(product_wire.node).ok_or_else(fail)?;
        if !matches!(
            product.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) | NodeKind::MatrixMulSmallRhs
        ) {
            return Err(fail());
        }
        let operands = child.arguments(product).ok_or_else(fail)?;
        let automorphism = child.node(operands[0].node).ok_or_else(fail)?;
        let NodeKind::RingAutomorphism { index } = automorphism.kind() else {
            return Err(fail());
        };
        let auto_input =
            child.arguments(automorphism).and_then(|a| a.first().copied()).ok_or_else(fail)?;
        let target_slot =
            child.inputs().iter().position(|wire| *wire == auto_input).ok_or_else(fail)?;
        let switch_slot =
            child.inputs().iter().position(|wire| *wire == operands[1]).ok_or_else(fail)?;
        let args = root.arguments(node).ok_or_else(fail)?;
        // The automorphism acts after the actual target right action. Follow
        // its gather and multiplication rather than postulating a target key.
        let target_gather_wire = args[target_slot];
        let target_gather_node = root.node(target_gather_wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(target_gather_loop) = target_gather_node.kind() else {
            return Err(fail());
        };
        let target_gather_id = child_id(target_gather_wire);
        let target_gather = &graph.source.scopes()[&target_gather_id];
        if target_gather.nodes().len() != 3 ||
            target_gather.inputs().len() != 2 ||
            target_gather.outputs().len() != 1
        {
            return Err(fail());
        }
        let target_get = target_gather.node(target_gather.outputs()[0].node).ok_or_else(fail)?;
        if !matches!(target_get.kind(), NodeKind::FamilyGetDynamic) {
            return Err(fail());
        }
        let target_get_args = target_gather.arguments(target_get).ok_or_else(fail)?;
        let target_family_slot = target_gather
            .inputs()
            .iter()
            .position(|wire| *wire == target_get_args[0])
            .ok_or_else(fail)?;
        let target_index_slot = target_gather
            .inputs()
            .iter()
            .position(|wire| *wire == target_get_args[1])
            .ok_or_else(fail)?;
        if target_gather_loop.input_modes[target_family_slot] != LoopInputMode::Broadcast ||
            target_gather_loop.input_modes[target_index_slot] != LoopInputMode::Zip
        {
            return Err(fail());
        }
        let target_gather_args = root.arguments(target_gather_node).ok_or_else(fail)?;
        let transformed_wire = target_gather_args[target_family_slot];
        let target_indices = value(target_gather_args[target_index_slot])?;
        let transformed_node = root.node(transformed_wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(transformed_loop) = transformed_node.kind() else {
            return Err(fail());
        };
        if transformed_loop.input_modes != [LoopInputMode::Zip, LoopInputMode::Zip] {
            return Err(fail());
        }
        let transformed_id = child_id(transformed_wire);
        let transformed = &graph.source.scopes()[&transformed_id];
        if transformed.nodes().len() != 3 ||
            transformed.inputs().len() != 2 ||
            transformed.outputs().len() != 1
        {
            return Err(fail());
        }
        let multiplication = transformed.node(transformed.outputs()[0].node).ok_or_else(fail)?;
        if !matches!(
            multiplication.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) | NodeKind::MatrixMulSmallRhs
        ) {
            return Err(fail());
        }
        let multiplication_args = transformed.arguments(multiplication).ok_or_else(fail)?;
        let left_slot = transformed
            .inputs()
            .iter()
            .position(|wire| *wire == multiplication_args[0])
            .ok_or_else(fail)?;
        let right_slot = transformed
            .inputs()
            .iter()
            .position(|wire| *wire == multiplication_args[1])
            .ok_or_else(fail)?;
        let transformed_args = root.arguments(transformed_node).ok_or_else(fail)?;
        let program_public = value(transformed_args[left_slot])?;
        let target_decomposition = value(transformed_args[right_slot])?;
        let ConcreteWireType::IndexedFamily { count: target_count, .. } =
            &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&transformed_wire]
        else {
            return Err(fail());
        };
        let target_gather_fact = fact(target_gather_wire)?;
        let transformed_fact = fact(transformed_wire)?;
        let gather_wire = args[switch_slot];
        let gather_node = root.node(gather_wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(gather_loop) = gather_node.kind() else {
            return Err(fail());
        };
        let gather_id = child_id(gather_wire);
        let gather = &graph.source.scopes()[&gather_id];
        if gather.nodes().len() != 3 || gather.inputs().len() != 2 || gather.outputs().len() != 1 {
            return Err(fail());
        }
        let get = gather.node(gather.outputs()[0].node).ok_or_else(fail)?;
        if !matches!(get.kind(), NodeKind::FamilyGetDynamic) {
            return Err(fail());
        }
        let get_args = gather.arguments(get).ok_or_else(fail)?;
        let family_slot =
            gather.inputs().iter().position(|wire| *wire == get_args[0]).ok_or_else(fail)?;
        let index_slot =
            gather.inputs().iter().position(|wire| *wire == get_args[1]).ok_or_else(fail)?;
        if gather_loop.input_modes[family_slot] != LoopInputMode::Broadcast ||
            gather_loop.input_modes[index_slot] != LoopInputMode::Zip
        {
            return Err(fail());
        }
        let gather_args = root.arguments(gather_node).ok_or_else(fail)?;
        let switch_wire = gather_args[family_slot];
        if !artifact.root.outputs.iter().any(|(name, output)| {
            name.ends_with("-readout-helper-switch-family-0") && output.wire == switch_wire
        }) {
            return Err(fail());
        }
        let switches = value(switch_wire)?;
        let gather_indices = value(gather_args[index_slot])?;
        let ConcreteWireType::IndexedFamily { count: switch_count, .. } =
            &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&switch_wire]
        else {
            return Err(fail());
        };
        let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
            return Err(fail());
        };
        let matrix = element.matrix_type().ok_or_else(fail)?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let index = index_expression(index, loop_.index_slot)?;
        let cached = value(output.wire)?;
        let family_fact = fact(output.wire)?;
        let gather_fact = fact(gather_wire)?;
        let child_fact = |wire: WireRef| -> Result<String, String> {
            let position = proof
                .node_constraints
                .get(&wire.node)
                .and_then(|r| r.clone().last())
                .ok_or_else(fail)?;
            Ok(format!("childFacts{}.1", ".2".repeat(position)))
        };
        let hauto = child_fact(operands[0])?;
        let hd = child_fact(child.outputs()[0])?;
        let witnesses =
            proof.witnesses.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
        let output_path = ".2".repeat(proof.constraint_count);
        let theorem = format!("readout_alignment_{}", theorems.len());
        source.push_str(&format!("-- Actual exported helper alignment {name:?}.\ntheorem {theorem} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type})\n    (facts : {body} backend hashModel params inputs outputs witness)\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, ∃ position : Fin {switch_count}, ∃ targetPosition : Fin {target_count}, ∃ action : ErrorMatrix {n} {columns} {columns},\n      (position.val : Int) = ({gather_indices}) i ∧\n      (targetPosition.val : Int) = ({target_indices}) i ∧\n      ({cached}) i = reduceMatrix {q} {n} {columns} {columns} action ∧\n      ActionGain action (denseRegularGain layout {n} 2) ∧\n      gadget * (({cached}) i) = MxxRuntime.ringAutomorphism ({index}).toNat ((({program_public}) targetPosition) * (({target_decomposition}) targetPosition)) * (({switches}) position) := by\n  intro i\n  obtain ⟨selected, hlo, hhi, hget, hout⟩ := {gather_fact} i\n  obtain ⟨position, hindex, hselected⟩ := hget\n  obtain ⟨selectedTarget, htlo, hthi, htget, htout⟩ := {target_gather_fact} i\n  obtain ⟨targetPosition, htindex, htselected⟩ := htget\n  have targetEquation := {transformed_fact} targetPosition\n  obtain ⟨{witnesses}, childFacts⟩ := {family_fact} i\n  have autoEquation := ({hauto}).2.2.2\n  have decomposed := {hd}\n  have reconstruction := gadget_runs_reconstruct (by decide) (by decide) gadgetRuns decomposed Fact.out\n  obtain ⟨action, ha, hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) decomposed\n  have outputEquation := childFacts{output_path}\n  refine ⟨position, targetPosition, action, hindex, htindex, outputEquation.trans ha, hgain, ?_⟩\n  rw [outputEquation, reconstruction, autoEquation, hout, hselected, htout, htselected]\n  dsimp only\n  rw [targetEquation]\n  rfl\n\n", params = artifact.root.parameter_type));
        theorems.insert(name.clone(), format!("ExponentGeneratedReadoutAlignment.{theorem}"));
    }
    if theorems.is_empty() {
        return Err("no actual readout public-helper alignment cache".into());
    }
    source.push_str("end ExponentGeneratedReadoutAlignment\n");
    Ok((source, theorems))
}
