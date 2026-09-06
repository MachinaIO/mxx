//! Initial encoding equations projected from one actual verification-root witness.

use std::collections::{BTreeMap, BTreeSet};

use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{BoundaryValue, LeanArtifact},
    node::{ConcatAxis, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, NodeId, Port, WireRef},
};

fn boundary_type(values: &BTreeMap<String, BoundaryValue>, count: usize) -> String {
    let mut types = vec![String::new(); count];
    for value in values.values() {
        types[value.tuple_index] = value.lean_type.clone();
    }
    match types.len() {
        0 => "Unit".into(),
        1 => types.remove(0),
        _ => format!(
            "{} × Unit",
            types.iter().map(|ty| format!("({ty})")).collect::<Vec<_>>().join(" × ")
        ),
    }
}

fn matrix(ty: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
    match ty {
        ConcreteWireType::Matrix(matrix) |
        ConcreteWireType::SmallMatrix { matrix, .. } |
        ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
        _ => None,
    }
}

/// Expand lexical wire references using the exporter's actual right-hand sides.
/// String literals and quoted identifiers are left intact, and record projections
/// are terminal references to this theorem's fixed witness.
pub(super) fn value_expression(
    name: &str,
    values: &BTreeMap<String, String>,
) -> Result<String, String> {
    fn expand(
        name: &str,
        values: &BTreeMap<String, String>,
        active: &mut BTreeSet<String>,
    ) -> Result<String, String> {
        let rhs = values.get(name).ok_or_else(|| format!("missing exported value {name}"))?;
        if !active.insert(name.into()) {
            return Err(format!("cyclic exported value {name}"));
        }
        let mut output = String::new();
        let mut token = String::new();
        let mut quoted = None;
        let mut escaped = false;
        let flush = |token: &mut String,
                     output: &mut String,
                     active: &mut BTreeSet<String>|
         -> Result<(), String> {
            if !token.is_empty() {
                if values.contains_key(token) && !output.ends_with('.') {
                    output.push('(');
                    output.push_str(&expand(token, values, active)?);
                    output.push(')');
                } else {
                    output.push_str(token);
                }
                token.clear();
            }
            Ok(())
        };
        for ch in rhs.chars() {
            if let Some(end) = quoted {
                output.push(ch);
                if escaped {
                    escaped = false;
                } else if end == '"' && ch == '\\' {
                    escaped = true;
                } else if ch == end {
                    quoted = None;
                }
            } else if ch.is_ascii_alphanumeric() || ch == '_' {
                token.push(ch);
            } else {
                flush(&mut token, &mut output, active)?;
                output.push(ch);
                if ch == '"' || ch == '«' {
                    quoted = Some(if ch == '«' { '»' } else { '"' });
                }
            }
        }
        flush(&mut token, &mut output, active)?;
        active.remove(name);
        Ok(output)
    }
    expand(name, values, &mut BTreeSet::new())
}

/// Produce conditional lemmas about the same root witness that later online
/// program proofs consume. There is no second run or independent sample here.
pub(super) fn root_inputs(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing root body metadata")?;
    let witness_type = shape.witness_record.as_ref().ok_or("missing root witness metadata")?;
    let root = graph.source.root_scope();
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let binary = |wire: WireRef, operation| {
        let node = root.node(wire.node)?;
        if wire.port != Port(0) ||
            !matches!(node.kind(), NodeKind::MatrixBinary(op) if *op == operation)
        {
            return None;
        }
        let args = root.arguments(node)?;
        (args.len() == 2).then(|| [args[0], args[1]])
    };
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut source = format!(
        "import {}\nimport ExponentInputSlice\nimport ExponentProgram\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedInputsRoot\nopen Mxx.Primitives MxxExponentLut\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        artifact.module_name,
        boundary_type(&artifact.root.inputs, artifact.root.input_count),
        boundary_type(&artifact.root.outputs, artifact.root.output_count)
    );
    let mut count = 0;
    for index in 0..root.nodes().len() {
        let actual_wire = WireRef { node: NodeId(index as u64), port: Port(0) };
        let Some([ideal_wire, error_wire]) = binary(actual_wire, MatrixBinaryOp::Add) else {
            continue
        };
        let Some(error_node) = root.node(error_wire.node) else { continue };
        let NodeKind::GaussianSample { max_coefficient_bound, .. } = error_node.kind() else {
            continue
        };
        let Some(output) = types.get(&actual_wire).and_then(matrix) else { continue };
        if output.rows != 1 {
            continue;
        }
        let cutoff = max_coefficient_bound.evaluate(&ParamEnv::default()).map_err(|_| {
            format!("root initial encoding node {index} requires a closed Gaussian cutoff")
        })?;
        if cutoff < 0.into() {
            return Err(format!("negative Gaussian cutoff at node {index}"));
        }
        let constraint = shape
            .node_constraints
            .get(&error_wire.node)
            .ok_or("missing Gaussian constraint metadata")?
            .end
            .checked_sub(1)
            .ok_or("empty Gaussian constraints")?;
        let path = ".2".repeat(constraint);
        let actual = value(actual_wire)?;
        let ideal = value(ideal_wire)?;
        let q = &output.modulus;
        let n = output.ring_dimension;
        let columns = output.columns;
        let backend_binder = if artifact.root.requires_backend {
            " {backend : MxxRuntime.BackendContext}"
        } else {
            ""
        };
        let backend_arg = if artifact.root.requires_backend { "backend " } else { "" };
        let hash_binder = if artifact.root.requires_hash_model {
            " {hashModel : MxxRuntime.HashModel}"
        } else {
            ""
        };
        let hash_arg = if artifact.root.requires_hash_model { "hashModel " } else { "" };
        source.push_str(&format!("-- Actual root encoding node {index}; Gaussian node {}.\ntheorem initial_encoding_{count}{backend_binder}{hash_binder}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type})\n    (facts : {body} {backend_arg}{hash_arg}params inputs outputs witness) :\n    ∃ error : ErrorMatrix {n} 1 {columns},\n      ({actual}) = ({ideal}) +\n        reduceMatrix {q} {n} 1 {columns} error ∧ CoeffBound error ({cutoff} : Nat) := by\n  have gaussian := facts{path}.1\n  obtain ⟨error, equation, bounded⟩ := gaussian_error_witness gaussian\n  refine ⟨error, ?_, bounded⟩\n  exact congrArg (fun sample : ExactMatrix {q} {n} 1 {columns} => ({ideal}) + sample) equation\n\n", error_wire.node.0, artifact.root.parameter_type));
        // Normalize only the actual two-block BGG sampler layout. The Lean
        // theorem verifies all dimensions and the exact selected slice bounds.
        if let Some([product_wire, tensor_wire]) = binary(ideal_wire, MatrixBinaryOp::Subtract) {
            if let Some([secret_wire, public_wire]) = binary(product_wire, MatrixBinaryOp::Multiply)
            {
                let tensor_node = root.node(tensor_wire.node).ok_or("missing tensor node")?;
                let public_node = root.node(public_wire.node).ok_or("missing public concat")?;
                if matches!(tensor_node.kind(), NodeKind::Tensor) &&
                    matches!(public_node.kind(), NodeKind::Concat { axis: ConcatAxis::Columns })
                {
                    let tensor_args = root.arguments(tensor_node).ok_or("missing tensor args")?;
                    let public_args = root.arguments(public_node).ok_or("missing public args")?;
                    if tensor_args.len() != 2 || public_args.len() != 2 {
                        return Err("initial sampler requires exactly two blocks".into());
                    }
                    let messages_node = root.node(tensor_args[0].node).ok_or("missing messages")?;
                    if !matches!(
                        messages_node.kind(),
                        NodeKind::Concat { axis: ConcatAxis::Columns }
                    ) {
                        return Err("initial sampler messages are not concatenated".into());
                    }
                    let messages_args =
                        root.arguments(messages_node).ok_or("missing message args")?;
                    if messages_args.len() != 2 {
                        return Err("initial sampler requires two plaintext blocks".into());
                    }
                    let projection = |node: NodeId| -> Result<String, String> {
                        let position = shape
                            .node_constraints
                            .get(&node)
                            .and_then(|range| range.end.checked_sub(1))
                            .ok_or("missing initial layout constraint")?;
                        Ok(format!("facts{}.1", ".2".repeat(position)))
                    };
                    for (slice_index, slice) in root.nodes().iter().enumerate() {
                        if !matches!(slice.kind(), NodeKind::Slice { .. }) ||
                            root.arguments(slice).is_none_or(|args| args != [actual_wire])
                        {
                            continue;
                        }
                        let selected = WireRef { node: NodeId(slice_index as u64), port: Port(0) };
                        let selected_type = types
                            .get(&selected)
                            .and_then(matrix)
                            .ok_or("initial slice is not a matrix")?;
                        if selected_type.columns * 2 != columns {
                            return Err("initial slice is not one of two equal blocks".into());
                        }
                        let selected_value = value(selected)?;
                        let secret = value(secret_wire)?;
                        let right_public = value(public_args[1])?;
                        let message = value(messages_args[1])?;
                        let payload = value(tensor_args[1])?;
                        source.push_str(&format!("-- Actual selected encoding slice node {slice_index}.\ntheorem initial_selected_{count}{backend_binder}{hash_binder}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type})\n    (facts : {body} {backend_arg}{hash_arg}params inputs outputs witness) :\n    Approx ({selected_value})\n      (({secret}) * ({right_public}) - (({message}) 0 0) • ({payload})) ({cutoff} : Nat) := by\n  exact initial_right_block {} {} {} {}\n    (initial_encoding_{count} witness facts)\n\n",
                            artifact.root.parameter_type, projection(public_wire.node)?,
                            projection(tensor_args[0].node)?, projection(tensor_wire.node)?,
                            projection(selected.node)?));
                        let [payload_secret_wire, gadget_wire] =
                            binary(tensor_args[1], MatrixBinaryOp::Multiply)
                                .ok_or("initial payload is not the actual secret-gadget product")?;
                        let payload_secret = value(payload_secret_wire)?;
                        let gadget = value(gadget_wire)?;
                        let selected_columns = selected_type.columns;
                        source.push_str(&format!("theorem initial_program_state_{count}{backend_binder}{hash_binder}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {witness_type})\n    (facts : {body} {backend_arg}{hash_arg}params inputs outputs witness) :\n    ProgramDerivation ({gadget})\n      (⟨({selected_value}), ({right_public}), ({message}) 0 0, ({secret}), ({payload_secret})⟩ : ProgramState {q} {n} 2 {selected_columns}) {cutoff} := by\n  obtain ⟨error, equation, bounded⟩ := initial_selected_{count} witness facts\n  exact ProgramDerivation.input _ error {cutoff} equation bounded\n\n", artifact.root.parameter_type));
                    }
                }
            }
        }
        count += 1;
    }
    if count == 0 {
        return Err("no root initial encoding Gaussian pattern found".into());
    }
    source.push_str("end ExponentGeneratedInputsRoot\n");
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_expansion_preserves_fixed_witness_and_quoted_names() {
        let values = BTreeMap::from([
            ("w_0_0".into(), "witness.w_0_0".into()),
            ("w_1_0".into(), "inputs.1".into()),
            ("w_2_0".into(), "combine w_0_0 w_1_0 params.«w_0_0» \"w_1_0\"".into()),
        ]);
        assert_eq!(
            value_expression("w_2_0", &values).unwrap(),
            "combine (witness.w_0_0) (inputs.1) params.«w_0_0» \"w_1_0\""
        );
    }

    #[test]
    fn value_expansion_rejects_missing_values_and_cycles() {
        assert!(value_expression("missing", &BTreeMap::new()).is_err());
        let values =
            BTreeMap::from([("w_0_0".into(), "w_1_0".into()), ("w_1_0".into(), "w_0_0".into())]);
        assert!(value_expression("w_0_0", &values).unwrap_err().contains("cyclic"));
    }
}
