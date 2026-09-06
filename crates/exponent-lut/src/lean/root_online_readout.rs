//! Readout execution facts projected from the fixed verification witness.

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

pub(super) fn root_online_readout(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing verification body")?;
    let witness_type = shape.witness_record.as_ref().ok_or("missing verification witness")?;
    let root_types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
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
    let binders = format!(
        "{{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type})\n    (facts : {body} backend hashModel params inputs outputs witness)",
        artifact.root.parameter_type
    );
    let mut source = format!(
        "import {}\nimport ExponentInputSlice\nimport ExponentDecomposition\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedOnlineReadout\nopen Mxx.Primitives MxxExponentLut\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        artifact.module_name,
        boundary_type(&artifact.root.inputs, artifact.root.input_count),
        boundary_type(&artifact.root.outputs, artifact.root.output_count)
    );
    let mut branches = 0;
    let mut masks = 0;
    for (id, child) in graph.source.scopes() {
        let FrozenGraphScopeId::ParallelBody { parent, owner } = id else {
            continue;
        };
        if !matches!(parent.as_ref(), FrozenGraphScopeId::Root) {
            continue;
        }
        let root_node = root.node(*owner).ok_or("missing loop owner")?;
        let NodeKind::ParallelLoop(loop_) = root_node.kind() else {
            continue;
        };
        let root_args = root.arguments(root_node).ok_or("missing loop arguments")?;
        let proof = &artifact.scope_proofs[id];
        let child_types = &graph.scopes[id].wire_types;
        let child_fact = |wire: WireRef| -> Result<String, String> {
            let position = proof
                .node_constraints
                .get(&wire.node)
                .and_then(|r| r.clone().last())
                .ok_or("missing child constraint")?;
            Ok(format!("childFacts{}.1", ".2".repeat(position)))
        };
        let actual = |port| WireRef { node: *owner, port: mxx_ir_core::types::Port(port) };
        let ConcreteWireType::IndexedFamily { count: lanes, .. } = &root_types[&actual(0)] else {
            continue;
        };
        let family_fact = fact(actual(0))?;
        let output = value(actual(0))?;
        let witnesses =
            proof.witnesses.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
        let input_value = |wire: WireRef| -> Result<String, String> {
            let slot = child
                .inputs()
                .iter()
                .position(|input| *input == wire)
                .ok_or("operation operand is not a direct loop input")?;
            let v = value(root_args[slot])?;
            match loop_.input_modes[slot] {
                LoopInputMode::Broadcast => Ok(format!("({v})")),
                LoopInputMode::Zip => Ok(format!("({v}) i")),
                _ => Err("unsupported readout loop offset".into()),
            }
        };
        if child.nodes().len() == 3 && child.inputs().len() == 2 && child.outputs().len() == 1 {
            let node = child.node(child.outputs()[0].node).ok_or("missing simple operation")?;
            let args = child.arguments(node).ok_or("missing simple operands")?;
            let operator = match node.kind() {
                NodeKind::MatrixBinary(MatrixBinaryOp::Add) => Some("+"),
                NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) | NodeKind::MatrixMulSmallRhs => {
                    Some("*")
                }
                _ => None,
            };
            if let Some(operator) = operator {
                let left = input_value(args[0])?;
                let right = input_value(args[1])?;
                source.push_str(&format!("-- Actual family operation node {}.\ntheorem family_operation_{} {binders} :\n    ∀ i : Fin {lanes}, ({output}) i = ({left}) {operator} ({right}) := by\n  intro i\n  exact {family_fact} i\n\n", owner.0, owner.0));
            } else if matches!(node.kind(), NodeKind::FamilyGetDynamic) {
                let family = input_value(args[0])?;
                let index = input_value(args[1])?;
                let ConcreteWireType::IndexedFamily { count, .. } = &child_types[&args[0]] else {
                    return Err("invalid gather family type".into());
                };
                source.push_str(&format!("theorem family_gather_{} {binders} :\n    ∀ i : Fin {lanes}, ∃ position : Fin {count},\n      (position.val : Int) = ({index}) ∧ ({output}) i = ({family}) position := by\n  intro i\n  obtain ⟨selected, hlo, hhi, hget, hout⟩ := {family_fact} i\n  obtain ⟨position, hindex, heq⟩ := hget\n  exact ⟨position, hindex, hout.trans heq⟩\n\n", owner.0));
            }
        }
        // The raw readout branch is aligned + automorphism(transformed)*switch.
        if child.nodes().len() == 6 && child.inputs().len() == 3 && child.outputs().len() == 1 {
            let addition = child.node(child.outputs()[0].node).ok_or("missing branch output")?;
            if !matches!(addition.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add)) {
                continue;
            }
            let add = child.arguments(addition).ok_or("missing branch add operands")?;
            let product = child.node(add[1].node).ok_or("missing branch product")?;
            if !matches!(
                product.kind(),
                NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
            ) {
                continue;
            }
            let mul = child.arguments(product).ok_or("missing branch product operands")?;
            let automorphism = child.node(mul[0].node).ok_or("missing raw automorphism")?;
            let NodeKind::RingAutomorphism { index } = automorphism.kind() else {
                continue;
            };
            let auto_args = child.arguments(automorphism).ok_or("missing raw source")?;
            let transformed = input_value(auto_args[0])?;
            let aligned = input_value(add[0])?;
            let switch = input_value(mul[1])?;
            let index = index_expression(index, loop_.index_slot)?;
            let matrix = child_types[&mul[0]].matrix_type().ok_or("invalid raw matrix")?;
            let q = &matrix.modulus;
            let n = matrix.ring_dimension;
            let rows = matrix.rows;
            let columns = matrix.columns;
            let hr = child_fact(mul[0])?;
            let output_path = ".2".repeat(proof.constraint_count);
            source.push_str(&format!("theorem raw_branch_{} {binders} :\n    ∀ i : Fin {lanes}, ∃ raw : ExactMatrix {q} {n} {rows} {columns},\n      MxxRuntime.ringAutomorphismRuns ({index}) ({transformed}) raw ∧\n      ({output}) i = ({aligned}) + raw * ({switch}) := by\n  intro i\n  obtain ⟨{witnesses}, childFacts⟩ := {family_fact} i\n  exact ⟨_, {hr}, childFacts{output_path}⟩\n\n", owner.0));
            branches += 1;
        }
        // Helper masks are the right block of the actual Gaussian encoding.
        for (port, selected) in child.outputs().iter().enumerate() {
            let selected_node = child.node(selected.node).ok_or("missing helper output")?;
            let slice_args = child.arguments(selected_node).unwrap_or_default();
            if !matches!(selected_node.kind(), NodeKind::Slice { .. }) || slice_args.len() != 1 {
                continue;
            }
            let batch_node = child.node(slice_args[0].node).ok_or("missing helper batch")?;
            if !matches!(batch_node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add)) {
                continue;
            }
            let batch = child.arguments(batch_node).ok_or("missing mask sum")?;
            let gaussian = child.node(batch[1].node).ok_or("missing helper sample")?;
            let NodeKind::GaussianSample { max_coefficient_bound, .. } = gaussian.kind() else {
                continue;
            };
            let ideal_node = child.node(batch[0].node).ok_or("missing helper ideal")?;
            if !matches!(ideal_node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)) {
                continue;
            }
            let ideal = child.arguments(ideal_node).ok_or("missing helper ideal operands")?;
            let tensor_node = child.node(ideal[1].node).ok_or("missing helper tensor")?;
            if !matches!(tensor_node.kind(), NodeKind::Tensor) {
                continue;
            }
            let tensor = child.arguments(tensor_node).ok_or("missing helper tensor operands")?;
            let public_product = child.node(ideal[0].node).ok_or("missing mask key product")?;
            if !matches!(public_product.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) {
                continue;
            }
            let public = child.arguments(public_product).ok_or("missing mask key operands")?;
            let public_concat = child.node(public[1].node).ok_or("missing mask bank")?;
            if !matches!(public_concat.kind(), NodeKind::Concat { axis: ConcatAxis::Columns }) {
                continue;
            }
            let public_bank = child.arguments(public_concat).ok_or("missing mask bank blocks")?;
            let message_concat = child.node(tensor[0].node).ok_or("missing helper messages")?;
            if !matches!(message_concat.kind(), NodeKind::Concat { axis: ConcatAxis::Columns }) {
                continue;
            }
            let payload_product = child.node(tensor[1].node).ok_or("missing payload product")?;
            if !matches!(payload_product.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) {
                continue;
            }
            let payload = child.arguments(payload_product).ok_or("missing payload factors")?;
            let auto_node = child.node(payload[0].node).ok_or("missing mask automorphism")?;
            let NodeKind::RingAutomorphism { index } = auto_node.kind() else {
                continue;
            };
            let auto_args = child.arguments(auto_node).ok_or("missing mask automorphism input")?;
            if auto_args[0] != public[0] {
                return Err("readout mask source/target secret mismatch".into());
            }
            let secret = input_value(public[0])?;
            let mask_key_port = child
                .outputs()
                .iter()
                .position(|wire| *wire == public_bank[1])
                .ok_or("helper mask key is not an actual family output")?;
            let mask_key = value(actual(mask_key_port as u32))?;
            let mask = value(actual(port as u32))?;
            let index = index_expression(index, loop_.index_slot)?;
            let matrix = child_types[selected].matrix_type().ok_or("missing mask type")?;
            let q = &matrix.modulus;
            let n = matrix.ring_dimension;
            let columns = matrix.columns;
            let gadget_node = child.node(payload[1].node).ok_or("missing helper gadget")?;
            let NodeKind::ConstantMatrix {
                value: mxx_ir_core::node::ConstantMatrix::Gadget { base, .. },
                ..
            } = gadget_node.kind()
            else {
                return Err("mask payload gadget not constant".into());
            };
            let base =
                base.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant mask gadget base")?;
            let digits = columns / 2;
            let cutoff = max_coefficient_bound
                .evaluate(&ParamEnv::default())
                .map_err(|_| "nonconstant mask cutoff")?;
            let hpublic = child_fact(public[1])?;
            let hmessage = child_fact(tensor[0])?;
            let htensor = child_fact(ideal[1])?;
            let hslice = child_fact(*selected)?;
            let hgaussian = child_fact(batch[1])?;
            let hgadget = child_fact(payload[1])?;
            let hauto = child_fact(payload[0])?;
            let gadget_name = format!("w_{}_{}", payload[1].node.0, payload[1].port.0);
            let output_path = ".2".repeat(proof.constraint_count);
            let mask_projection = ".2".repeat(port) + ".1";
            let key_projection = ".2".repeat(mask_key_port) + ".1";
            source.push_str(&format!("theorem helper_mask_{} {binders}\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, ∃ error : ErrorMatrix {n} 1 {columns},\n      ({mask}) i = ({secret}) * (({mask_key}) i) -\n        MxxRuntime.ringAutomorphism ({index}).toNat ({secret}) * gadget +\n          reduceMatrix {q} {n} 1 {columns} error ∧ CoeffBound error ({cutoff} : Nat) := by\n  intro i\n  obtain ⟨{witnesses}, childFacts⟩ := {family_fact} i\n  obtain ⟨error, he, hb⟩ := gaussian_error_witness {hgaussian}\n  have bounded := initial_right_block {hpublic} {hmessage} {htensor} {hslice}\n    ⟨error, by rw [he]; rfl, hb⟩\n  have maskOutput := congrArg (fun result => result{mask_projection}) childFacts{output_path}\n  have keyOutput := congrArg (fun result => result{key_projection}) childFacts{output_path}\n  dsimp only at maskOutput keyOutput\n  have autoEquation := ({hauto}).2.2.2\n  have same : {gadget_name} = gadget := gadget_runs_deterministic {hgadget} gadgetRuns\n  obtain ⟨selectedError, selectedEquation, selectedBound⟩ := bounded\n  refine ⟨selectedError, ?_, selectedBound⟩\n  rw [maskOutput, selectedEquation, keyOutput, autoEquation, same]\n  simp [MxxRuntime.matrixMul]\n\n", owner.0));
            masks += 1;
        }
    }
    if branches == 0 || masks == 0 {
        return Err(format!(
            "readout execution extraction incomplete: {branches} branches, {masks} masks"
        ));
    }
    source.push_str("end ExponentGeneratedOnlineReadout\n");
    Ok(source)
}
