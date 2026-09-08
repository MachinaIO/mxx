//! Readout switch cache equations from one actual preprocessing-root witness.

use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, ValidatedGraph,
    lean::{BoundaryValue, LeanArtifact},
    node::{ConcatAxis, LoopInputMode, MatrixBinaryOp, NodeKind},
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

pub(super) fn root_readout(
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
    let root_fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing root constraint")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let mut source = format!(
        "import {}\nimport ExponentReadoutSetup\nimport RuntimeDecomposition\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedReadoutRoot\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        artifact.module_name,
        boundary_type(&artifact.root.inputs, artifact.root.input_count),
        boundary_type(&artifact.root.outputs, artifact.root.output_count)
    );
    let mut theorems = BTreeMap::new();
    for (name, output) in &artifact.root.outputs {
        if !name.ends_with("-readout-0.1") {
            continue;
        }
        let fail = || format!("unsupported readout switch graph at {name}");
        let decomp_node = root.node(output.wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(decomp_loop) = decomp_node.kind() else {
            return Err(fail());
        };
        if decomp_loop.input_modes != [LoopInputMode::Zip] {
            return Err(fail());
        }
        let decomp_id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: output.wire.node,
        };
        let decomp_scope = &graph.source.scopes()[&decomp_id];
        if decomp_scope.nodes().len() != 2 || decomp_scope.outputs().len() != 1 {
            return Err(fail());
        }
        let NodeKind::GadgetDecompose { base, digit_count, .. } =
            decomp_scope.node(decomp_scope.outputs()[0].node).ok_or_else(fail)?.kind()
        else {
            return Err(fail());
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let sampler_wire =
            root.arguments(decomp_node).and_then(|args| args.first().copied()).ok_or_else(fail)?;
        let sampler_node = root.node(sampler_wire.node).ok_or_else(fail)?;
        let NodeKind::ParallelLoop(sampler_loop) = sampler_node.kind() else {
            return Err(fail());
        };
        if sampler_loop.input_modes.iter().any(|mode| *mode != LoopInputMode::Broadcast) {
            return Err(fail());
        }
        let sampler_id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: sampler_wire.node,
        };
        let sampler = &graph.source.scopes()[&sampler_id];
        let proof = &artifact.scope_proofs[&sampler_id];
        let sampled_wire = *sampler.outputs().get(sampler_wire.port.0 as usize).ok_or_else(fail)?;
        let concat = sampler.node(sampled_wire.node).ok_or_else(fail)?;
        if !matches!(concat.kind(), NodeKind::Concat { axis: ConcatAxis::Rows }) {
            return Err(fail());
        }
        let concat_args = sampler.arguments(concat).ok_or_else(fail)?;
        let binary = |wire: WireRef, op: MatrixBinaryOp| -> Result<Vec<WireRef>, String> {
            let node = sampler.node(wire.node).ok_or_else(fail)?;
            if !matches!(node.kind(), NodeKind::MatrixBinary(actual) if *actual == op) {
                return Err(fail());
            }
            sampler.arguments(node).ok_or_else(fail)
        };
        let added = binary(concat_args[1], MatrixBinaryOp::Add)?;
        let subtracted = binary(added[0], MatrixBinaryOp::Subtract)?;
        let target_action = binary(subtracted[0], MatrixBinaryOp::Multiply)?;
        let target_product = binary(target_action[1], MatrixBinaryOp::Multiply)?;
        let prefix_action = binary(subtracted[1], MatrixBinaryOp::Multiply)?;
        if prefix_action[1] != concat_args[0] {
            return Err(fail());
        }
        let slice = sampler.node(prefix_action[0].node).ok_or_else(fail)?;
        let source_wire =
            sampler.arguments(slice).and_then(|args| args.first().copied()).ok_or_else(fail)?;
        let automorphism = sampler.node(source_wire.node).ok_or_else(fail)?;
        let NodeKind::RingAutomorphism { index } = automorphism.kind() else {
            return Err(fail());
        };
        let auto_input = sampler
            .arguments(automorphism)
            .and_then(|args| args.first().copied())
            .ok_or_else(fail)?;
        if auto_input != target_product[0] {
            return Err(fail());
        }
        let target_slot =
            sampler.inputs().iter().position(|wire| *wire == auto_input).ok_or_else(fail)?;
        let target_wire = root.arguments(sampler_node).ok_or_else(fail)?[target_slot];
        let target_node = root.node(target_wire.node).ok_or_else(fail)?;
        if !matches!(target_node.kind(), NodeKind::Concat { axis: ConcatAxis::Columns }) {
            return Err(fail());
        }
        let gaussian = sampler.node(added[1].node).ok_or_else(fail)?;
        let NodeKind::GaussianSample { max_coefficient_bound, .. } = gaussian.kind() else {
            return Err(fail());
        };
        let cutoff = max_coefficient_bound.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
            return Err(fail());
        };
        let matrix = element.matrix_type().ok_or_else(fail)?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let index = index_expression(index, sampler_loop.index_slot)?;
        let target = value(target_wire)?;
        let cached = value(output.wire)?;
        let child_fact = |wire: WireRef| -> Result<String, String> {
            let position = proof
                .node_constraints
                .get(&wire.node)
                .and_then(|r| r.clone().last())
                .ok_or_else(fail)?;
            Ok(format!("childFacts{}.1", ".2".repeat(position)))
        };
        let variable = |wire: WireRef| format!("w_{}_{}", wire.node.0, wire.port.0);
        let witnesses =
            proof.witnesses.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
        let sampler_fact = root_fact(sampler_wire)?;
        let decomp_fact = root_fact(output.wire)?;
        let target_fact = root_fact(target_wire)?;
        let hg = child_fact(target_product[1])?;
        let hauto = child_fact(source_wire)?;
        let hslice = child_fact(prefix_action[0])?;
        let hgaussian = child_fact(added[1])?;
        let hconcat = child_fact(sampled_wire)?;
        let other = variable(target_product[1]);
        let sampled_source = variable(source_wire);
        let top = variable(concat_args[0]);
        let prefix = variable(prefix_action[0]);
        let sample = variable(added[1]);
        let output_path = ".2".repeat(proof.constraint_count);
        let output_projection = ".2".repeat(sampler_wire.port.0 as usize) + ".1";
        let theorem = format!("readout_switch_{}", theorems.len());
        source.push_str(&format!("-- Actual exported helper cache {name:?}.\ntheorem {theorem} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type})\n    (facts : {body} backend hashModel params inputs outputs witness)\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, ∃ error : ErrorMatrix {n} 1 {columns}, ∃ action : ErrorMatrix {n} {columns} {columns},\n      ({cached}) i = reduceMatrix {q} {n} {columns} {columns} action ∧\n      ActionGain action (denseRegularGain layout {n} 2) ∧\n      MxxRuntime.ringAutomorphism ({index}).toNat ({target}) * gadget * (({cached}) i) =\n        ({target}) * gadget + reduceMatrix {q} {n} 1 {columns} error ∧ CoeffBound error ({cutoff} : Nat) := by\n  have augmented := augmented_source_last {target_fact} (by rfl)\n  intro i\n  obtain ⟨{witnesses}, childFacts⟩ := {sampler_fact} i\n  have auto := {hauto}\n  obtain ⟨positive, upper, oddIndex, autoEquation⟩ := auto\n  have oddNat : Odd ({index}).toNat := by apply Nat.odd_iff.mpr; omega\n  have sourceLast : {sampled_source} 0 1 = 1 := by\n    rw [autoEquation, readoutAutomorphismHom_runtime (by decide) (by decide) _ oddNat]\n    change readoutAutomorphismHom _ oddNat (({target}) 0 1) = 1\n    rw [augmented, map_one]\n  obtain ⟨decomposition, hd, hdout⟩ := {decomp_fact} i\n  have sampledOutput := congrArg (fun result => result{output_projection}) childFacts{output_path}\n  dsimp only at sampledOutput\n  rw [sampledOutput] at hd\n  have same : {other} = gadget := gadget_runs_deterministic {hg} gadgetRuns\n  obtain ⟨error, he, hb⟩ := sampled_fixed_rhs_cache (by decide) (by decide) sourceLast {hslice}\n    (show MxxRuntime.matrixAdd (MxxRuntime.matrixSub (MxxRuntime.matrixMulScalarLeft (1 : ExactMatrix {q} {n} 1 1) (MxxRuntime.matrixMul ({target}) {other})) (MxxRuntime.matrixMulScalarLeft {prefix} {top})) {sample} = (1 : ExactPoly {q} {n}) • (({target}) * {other}) - {prefix} * {top} + {sample} from by\n      ext row column\n      have hr : row = 0 := Subsingleton.elim _ _\n      subst row\n      simp [MxxRuntime.matrixAdd, MxxRuntime.matrixSub, MxxRuntime.matrixMul, Matrix.mul_apply])\n    {hconcat} {hgaussian} gadgetRuns hd Fact.out\n  obtain ⟨action, ha, hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd\n  refine ⟨error, action, hdout.trans ha, hgain, ?_, hb⟩\n  rw [autoEquation, same, one_smul] at he\n  rw [hdout]\n  exact he\n\n", params = artifact.root.parameter_type));
        theorems.insert(name.clone(), format!("ExponentGeneratedReadoutRoot.{theorem}"));
    }
    if theorems.is_empty() {
        return Err("no actual readout helper switch cache".into());
    }
    source.push_str("end ExponentGeneratedReadoutRoot\n");
    Ok((source, theorems))
}
