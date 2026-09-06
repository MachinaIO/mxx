//! Application proof instantiations for actual primitive action scopes.

use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::LeanArtifact,
    node::{MatrixBinaryOp, NodeKind},
    types::ConcreteWireType,
};

fn boundary_type(
    values: &std::collections::BTreeMap<String, mxx_ir_core::lean::BoundaryValue>,
    count: usize,
) -> String {
    if count == 0 {
        return "Unit".into();
    }
    let mut types = vec![String::new(); count];
    for value in values.values() {
        types[value.tuple_index] = value.lean_type.clone();
    }
    if count == 1 {
        types[0].clone()
    } else {
        format!(
            "{} × Unit",
            types.iter().map(|ty| format!("({ty})")).collect::<Vec<_>>().join(" × ")
        )
    }
}

/// Each generated theorem consumes the exported scope relation itself. Scope
/// selection inspects the frozen nodes; it never parses generated Lean text.
pub(super) fn matrix_action_proofs(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> (String, usize) {
    let mut source = format!(
        "import {}\nimport ExponentTransfers\n\nnamespace ExponentGeneratedScopes\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\n",
        artifact.module_name,
    );
    let mut count = 0;
    for (id, scope) in graph.source.scopes() {
        if scope.inputs().len() != 2 || scope.outputs().len() != 1 || scope.nodes().len() != 3 {
            continue;
        }
        let output_wire = scope.outputs()[0];
        let Some(node) = scope.node(output_wire.node) else { continue };
        if !matches!(
            node.kind(),
            NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) {
            continue;
        }
        let Some(arguments) = scope.arguments(node) else { continue };
        if arguments.len() != 2 || arguments.iter().any(|wire| !scope.inputs().contains(wire)) {
            continue;
        }
        let types = &graph.scopes[id].wire_types;
        let ConcreteWireType::Matrix(left) = &types[&arguments[0]] else { continue };
        let right = match &types[&arguments[1]] {
            ConcreteWireType::Matrix(matrix) |
            ConcreteWireType::Preimage { matrix, .. } |
            ConcreteWireType::SmallMatrix { matrix, .. } => matrix,
            _ => continue,
        };
        // Scalar-specialized multiplication has a different runtime equation;
        // those scopes are proved by the scalar transfer lemmas separately.
        if left.is_scalar() || right.is_scalar() {
            continue;
        }
        let mut indices = 0;
        let mut ancestor = id;
        while let FrozenGraphScopeId::ParallelBody { parent, .. } = ancestor {
            indices += 1;
            ancestor = parent;
        }
        if !matches!(ancestor, FrozenGraphScopeId::Root) || indices == 0 {
            continue;
        }
        let q = &left.modulus;
        let n = left.ring_dimension;
        let rows = left.rows;
        let inner = left.columns;
        let columns = right.columns;
        let relation = &artifact.scope_relations[id];
        let parameter_type = &artifact.root.parameter_type;
        let context_binders = format!(
            "{}{}",
            if artifact.root.requires_backend {
                "{backend : MxxRuntime.BackendContext} "
            } else {
                ""
            },
            if artifact.root.requires_hash_model {
                "{hashModel : MxxRuntime.HashModel} "
            } else {
                ""
            }
        );
        let context_arguments = format!(
            "{}{}",
            if artifact.root.requires_backend { "backend " } else { "" },
            if artifact.root.requires_hash_model { "hashModel " } else { "" }
        );
        let index_names = (0..indices).map(|i| format!("i_{i}")).collect::<Vec<_>>().join(" ");
        let action = format!("reduceMatrix {q} {n} {inner} {columns} action");
        let inputs = if arguments[0] == scope.inputs()[0] {
            format!("(input, {action}, ())")
        } else {
            format!("({action}, input, ())")
        };
        source.push_str(&format!(
            "theorem matrix_action_{count} {context_binders}{{params : {parameter_type}}} {{{index_names} : Nat}}\n    {{input ideal : ExactMatrix {q} {n} {rows} {inner}}}\n    {{action : ErrorMatrix {n} {inner} {columns}}}\n    {{output : ExactMatrix {q} {n} {rows} {columns}}} {{bound gain : Nat}}\n    (runs : {relation} {context_arguments}params {index_names} {inputs} output)\n    (close : Approx input ideal bound) (actionBound : ActionGain action gain) :\n    Approx output (ideal * ({action})) (gain * bound) := by\n  have equation : output = input * ({action}) := runs\n  rw [equation]\n  exact approximate_right_action (by decide) close actionBound\n\n"
        ));
        let outer_indices =
            (0..indices - 1).map(|i| format!("i_{i}")).collect::<Vec<_>>().join(" ");
        let outer_binders = if outer_indices.is_empty() {
            String::new()
        } else {
            format!("{{{outer_indices} : Nat}}")
        };
        let family_action = format!("reduceMatrix {q} {n} {inner} {columns} (action i)");
        let family_inputs = if arguments[0] == scope.inputs()[0] {
            format!("(input i, {family_action}, ())")
        } else {
            format!("({family_action}, input i, ())")
        };
        source.push_str(&format!(
            "theorem matrix_action_family_{count} {context_binders}{{params : {parameter_type}}} {outer_binders} {{lanes : Nat}}\n    {{input ideal : Fin lanes → ExactMatrix {q} {n} {rows} {inner}}}\n    {{action : Fin lanes → ErrorMatrix {n} {inner} {columns}}}\n    {{output : Fin lanes → ExactMatrix {q} {n} {rows} {columns}}} {{bound gain : Fin lanes → Nat}}\n    (runs : ∀ i : Fin lanes, {relation} {context_arguments}params {outer_indices} i.val {family_inputs} (output i))\n    (close : ∀ i, Approx (input i) (ideal i) (bound i))\n    (actionBound : ∀ i, ActionGain (action i) (gain i)) :\n    ∀ i, Approx (output i) (ideal i * ({family_action})) (gain i * bound i) := by\n  intro i\n  exact matrix_action_{count} (runs i) (close i) (actionBound i)\n\n"
        ));
        source.push_str(&format!(
            "theorem matrix_action_family_sum_{count} {context_binders}{{params : {parameter_type}}} {outer_binders} {{lanes : Nat}}\n    {{input ideal : Fin lanes → ExactMatrix {q} {n} {rows} {inner}}}\n    {{action : Fin lanes → ErrorMatrix {n} {inner} {columns}}}\n    {{output : Fin lanes → ExactMatrix {q} {n} {rows} {columns}}} {{bound gain : Fin lanes → Nat}}\n    (runs : ∀ i : Fin lanes, {relation} {context_arguments}params {outer_indices} i.val {family_inputs} (output i))\n    (close : ∀ i, Approx (input i) (ideal i) (bound i))\n    (actionBound : ∀ i, ActionGain (action i) (gain i)) :\n    Approx (Finset.univ.sum output)\n      (Finset.univ.sum (fun i => ideal i * ({family_action})))\n      (Finset.univ.sum (fun i => gain i * bound i)) := by\n  exact family_approximate_sum Finset.univ output _ _ (fun i _ => matrix_action_{count} (runs i) (close i) (actionBound i))\n\n"
        ));
        if rows == 1 && inner == columns {
            source.push_str(&format!(
                "theorem private_encoding_scope_{count} {context_binders}{{params : {parameter_type}}} {{{index_names} : Nat}}\n    {{secret inputBound helperBound gain : Nat}}\n    {{input output : ExactMatrix {q} {n} 1 {columns}}}\n    {{mask : ExactMatrix {q} {n} 1 secret}} {{publicKey : ExactMatrix {q} {n} secret {columns}}}\n    {{payloadGadget : ExactMatrix {q} {n} 1 {columns}}}\n    {{message rhsMessage : ExactPoly {q} {n}}} {{messageInteger : ErrorPoly {n}}}\n    {{inputError helperError : ErrorMatrix {n} 1 {columns}}}\n    {{action : ErrorMatrix {n} {columns} {columns}}}\n    (runs : {relation} {context_arguments}params {index_names} {inputs} output)\n    (encodingEquation : input = mask * publicKey - message • payloadGadget + reduceMatrix {q} {n} 1 {columns} inputError)\n    (actionEquation : payloadGadget * ({action}) = rhsMessage • payloadGadget + reduceMatrix {q} {n} 1 {columns} helperError)\n    (messageLift : message = reducePoly {q} {n} messageInteger)\n    (messageBound : polynomialL1 messageInteger ≤ 1)\n    (inputBounded : CoeffBound inputError inputBound)\n    (helperBounded : CoeffBound helperError helperBound)\n    (actionBounded : ActionGain action gain) :\n    Approx output (mask * (publicKey * ({action})) - (message * rhsMessage) • payloadGadget)\n      (gain * inputBound + helperBound) := by\n  have equation : output = input * ({action}) := runs\n  rw [equation]\n  exact private_encoding_transfer_sampled (by decide) encodingEquation actionEquation messageLift messageBound inputBounded helperBounded actionBounded\n\n"
            ));
        }
        count += 1;
    }
    source.push_str("end ExponentGeneratedScopes\n");
    (source, count)
}

/// The fixed selector producer has one sampled top row, gadget, sliced source,
/// sampled error and concatenated ciphertext. The proof destructures that actual
/// exported scope and applies the handwritten cancellation theorem.
pub(super) fn selector_setup_proofs(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    bindings: &mxx_ir_core::ParamEnv,
) -> (String, usize) {
    use mxx_ir_core::node::{ConcatAxis, ConstantMatrix};
    let mut source = format!(
        "import {}\nimport ExponentSetup\nimport ExponentSecret\nimport RuntimeDecomposition\n\nnamespace ExponentGeneratedSetup\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\n",
        artifact.module_name
    );
    let mut count = 0;
    for (id, scope) in graph.source.scopes() {
        if scope.inputs().len() != 3 || scope.outputs().len() != 1 || scope.nodes().len() != 13 {
            continue;
        }
        if !matches!(id, FrozenGraphScopeId::ParallelBody { parent, .. } if matches!(parent.as_ref(), FrozenGraphScopeId::Root))
        {
            continue;
        }
        let types = &graph.scopes[id].wire_types;
        let ConcreteWireType::Matrix(output) = &types[&scope.outputs()[0]] else { continue };
        if output.rows != 2 {
            continue;
        }
        let mut names = Vec::new();
        for input in scope.inputs() {
            names.push(match &types[input] {
                ConcreteWireType::Bytes { .. } => "key",
                ConcreteWireType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1 => {
                    "payload"
                }
                ConcreteWireType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 2 => {
                    "secret"
                }
                _ => "unsupported",
            });
        }
        if names.contains(&"unsupported") ||
            !names.contains(&"payload") ||
            !names.contains(&"secret") ||
            !names.contains(&"key")
        {
            continue;
        }
        let mut base = None;
        let mut cutoff = None;
        let mut concat = false;
        for node in scope.nodes() {
            match node.kind() {
                NodeKind::ConstantMatrix {
                    value: ConstantMatrix::Gadget { base: value, .. },
                    ..
                } => base = value.evaluate(bindings).ok(),
                NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                    cutoff = max_coefficient_bound.evaluate(bindings).ok()
                }
                NodeKind::Concat { axis: ConcatAxis::Rows } => concat = true,
                _ => {}
            }
        }
        let (Some(base), Some(cutoff)) = (base, cutoff) else { continue };
        if !concat {
            continue;
        }
        let q = &output.modulus;
        let n = output.ring_dimension;
        let columns = output.columns;
        let digits = columns / 2;
        let relation = &artifact.scope_relations[id];
        let inputs = format!("({}, ())", names.join(", "));
        source.push_str(&format!(
            "theorem selector_setup_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}.Params}} {{index : Nat}} {{key : ByteArray}}\n    {{payload : ExactMatrix {q} {n} 1 1}} {{secret : ExactMatrix {q} {n} 1 2}}\n    {{ciphertext : ExactMatrix {q} {n} 2 {columns}}}\n    (runs : {relation} backend hashModel params index {inputs} ciphertext)\n    (augmented : secret 0 1 = 1) :\n    ∃ gadget : ExactMatrix {q} {n} 2 {columns}, ∃ error : ErrorMatrix {n} 1 {columns},\n      MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget ∧\n      secret * ciphertext = (payload 0 0) • (secret * gadget) + reduceMatrix {q} {n} 1 {columns} error ∧\n      CoeffBound error ({cutoff} : Nat) := by\n  rcases runs with ⟨top, gadget, sourcePrefix, sample, result, facts⟩\n  rcases facts with ⟨_, hg, _, _, _, _, _, _, hs, he, hc, ho⟩\n  obtain ⟨error, herror, hbound⟩ := gaussian_error_witness he\n  refine ⟨gadget, error, hg, ?_, hbound⟩\n  rw [ho]\n  have equation := sample_fixed_rhs_equation augmented hs (show MxxRuntime.matrixAdd (MxxRuntime.matrixSub (MxxRuntime.matrixMulScalarLeft payload (MxxRuntime.matrixMul secret gadget)) (MxxRuntime.matrixMulScalarLeft sourcePrefix top)) sample = (payload 0 0) • (secret * gadget) - sourcePrefix * top + sample from ?_) hc\n  · rw [equation, herror]\n  · ext row column\n    have hr : row = 0 := Subsingleton.elim _ _\n    subst row\n    simp [MxxRuntime.matrixAdd, MxxRuntime.matrixSub, MxxRuntime.matrixMul, Matrix.mul_apply]\n\n",
            artifact.module_name,
        ));
        source.push_str(&format!(
            "theorem selector_cache_{count} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}.Params}} {{index : Nat}} {{key : ByteArray}}\n    {{payload : ExactMatrix {q} {n} 1 1}} {{secret : ExactMatrix {q} {n} 1 2}}\n    {{ciphertext : ExactMatrix {q} {n} 2 {columns}}}\n    {{decomposition : ExactMatrix {q} {n} {columns} {columns}}}\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (runs : {relation} backend hashModel params index {inputs} ciphertext)\n    (augmented : secret 0 1 = 1)\n    (decomposed : MxxRuntime.gadgetDecomposeRuns backend {base} {digits} ciphertext decomposition) :\n    ∃ gadget : ExactMatrix {q} {n} 2 {columns}, ∃ error : ErrorMatrix {n} 1 {columns},\n    ∃ action : ErrorMatrix {n} {columns} {columns},\n      MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget ∧\n      decomposition = reduceMatrix {q} {n} {columns} {columns} action ∧\n      ActionGain action (denseRegularGain layout {n} 2) ∧\n      secret * gadget * decomposition = (payload 0 0) • (secret * gadget) + reduceMatrix {q} {n} 1 {columns} error ∧\n      CoeffBound error ({cutoff} : Nat) := by\n  obtain ⟨gadget, error, hg, he, hb⟩ := selector_setup_{count} runs augmented\n  obtain ⟨action, ha, hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) decomposed\n  refine ⟨gadget, error, action, hg, ha, hgain, ?_, hb⟩\n  rw [Matrix.mul_assoc, gadget_runs_reconstruct (by decide) (by decide) hg decomposed Fact.out]\n  exact he\n\n",
            artifact.module_name,
        ));
        source.push_str(&format!(
            "theorem selector_private_step_{count} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}.Params}} {{index : Nat}} {{key : ByteArray}}\n    {{payload : ExactMatrix {q} {n} 1 1}} {{secret mask : ExactMatrix {q} {n} 1 2}}\n    {{ciphertext gadget publicKey : ExactMatrix {q} {n} 2 {columns}}}\n    {{decomposition : ExactMatrix {q} {n} {columns} {columns}}}\n    {{input output : ExactMatrix {q} {n} 1 {columns}}}\n    {{inputError : ErrorMatrix {n} 1 {columns}}} {{inputBound : Nat}}\n    {{message : ExactPoly {q} {n}}} {{messageInteger : ErrorPoly {n}}}\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (runs : {relation} backend hashModel params index {inputs} ciphertext)\n    (augmented : secret 0 1 = 1)\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget)\n    (decomposed : MxxRuntime.gadgetDecomposeRuns backend {base} {digits} ciphertext decomposition)\n    (online : output = input * decomposition)\n    (encoded : input = mask * publicKey - message • (secret * gadget) + reduceMatrix {q} {n} 1 {columns} inputError)\n    (messageLift : message = reducePoly {q} {n} messageInteger)\n    (messageBound : polynomialL1 messageInteger ≤ 1)\n    (inputBounded : CoeffBound inputError inputBound) :\n    Approx output (mask * (publicKey * decomposition) - (message * payload 0 0) • (secret * gadget))\n      (denseRegularGain layout {n} 2 * inputBound + {cutoff}) := by\n  obtain ⟨other, error, action, hg, ha, hgain, hcache, hb⟩ := selector_cache_{count} layout lookup runs augmented decomposed\n  have same : other = gadget := gadget_runs_deterministic hg gadgetRuns\n  subst other\n  rw [ha] at hcache\n  rw [online, ha]\n  exact private_encoding_transfer_sampled (by decide) encoded hcache messageLift messageBound inputBounded hb hgain\n\n",
            artifact.module_name,
        ));

        // Lift the child sampler equation to the actual exported producer root.
        // Both the augmented source and sampled selector family are root outputs.
        if let FrozenGraphScopeId::ParallelBody { owner, .. } = id {
            let root_scope = graph.source.root_scope();
            let root = &artifact.root;
            let root_shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
            let parent_node = root_scope.node(*owner).expect("validated loop owner");
            let arguments = root_scope.arguments(parent_node).expect("validated loop arguments");
            let secret_wire = arguments[names.iter().position(|name| *name == "secret").unwrap()];
            let payload_wire = arguments[names.iter().position(|name| *name == "payload").unwrap()];
            let secret_output = root.outputs.values().find(|value| value.wire == secret_wire);
            let family_output = root.outputs.values().find(|value| value.wire.node == *owner);
            let payload_input = root.inputs.values().find(|value| value.wire == payload_wire);
            if let (Some(secret_output), Some(family_output), Some(payload_input)) =
                (secret_output, family_output, payload_input)
            {
                let projection = |name: &str, index: usize, size: usize| {
                    if size == 1 {
                        name.to_owned()
                    } else {
                        format!("{name}{}.1", ".2".repeat(index))
                    }
                };
                let secret = projection("output", secret_output.tuple_index, root.output_count);
                let family = projection("output", family_output.tuple_index, root.output_count);
                let payload = projection("input", payload_input.tuple_index, root.input_count);
                let concat = root_shape.node_constraints[&secret_wire.node].clone().last().unwrap();
                let concat_node = root_scope.node(secret_wire.node).unwrap();
                let small_wire = root_scope.arguments(concat_node).unwrap()[0];
                let sample_position =
                    root_shape.node_constraints[&small_wire.node].clone().last().unwrap();
                source.push_str(&format!(
                    "theorem root_secret_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {parameter_type}}} {{input : {input_type}}} {{output : {output_type}}}\n    (runs : {root_relation} backend hashModel params input output) :\n    ∃ secret : ErrorMatrix {n} 1 2, reduceMatrix {q} {n} 1 2 secret = ({secret}) ∧\n      (∀ i, ((secret 0 0).coeff i).natAbs ≤ 1) ∧ secret 0 1 = 1 := by\n  obtain ⟨witness, facts⟩ := runs\n  have sample := facts{sample_path}.1\n  have concat := facts{concat_path}.1\n  have outputEquation := facts{output_path}\n  rw [outputEquation]\n  exact augmented_secret_of_sample _ _ _ sample concat (by rfl)\n\n",
                    parameter_type = root.parameter_type,
                    input_type = boundary_type(&root.inputs, root.input_count),
                    output_type = boundary_type(&root.outputs, root.output_count),
                    root_relation = root.relation,
                    sample_path = ".2".repeat(sample_position),
                    concat_path = ".2".repeat(concat),
                    output_path = ".2".repeat(root_shape.constraint_count),
                ));
                let family_fact = root_shape.node_constraints[owner].clone().last().unwrap();
                source.push_str(&format!(
                    "theorem root_selector_setup_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {parameter_type}}} {{input : {input_type}}} {{output : {output_type}}}\n    (runs : {root_relation} backend hashModel params input output) :\n    ({secret}) 0 1 = 1 ∧\n    ∀ i, ∃ gadget : ExactMatrix {q} {n} 2 {columns}, ∃ error : ErrorMatrix {n} 1 {columns},\n      MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget ∧\n      ({secret}) * (({family}) i) = (({payload}) i 0 0) • (({secret}) * gadget) + reduceMatrix {q} {n} 1 {columns} error ∧\n      CoeffBound error ({cutoff} : Nat) := by\n  obtain ⟨witness, facts⟩ := runs\n  have concat := facts{concat_path}.1\n  have family := facts{family_path}.1\n  have outputEquation := facts{output_path}\n  have augmented := augmented_source_last concat (by rfl)\n  rw [outputEquation]\n  refine ⟨augmented, ?_⟩\n  intro i\n  exact selector_setup_{count} (family i) augmented\n\n",
                    parameter_type = root.parameter_type,
                    input_type = boundary_type(&root.inputs, root.input_count),
                    output_type = boundary_type(&root.outputs, root.output_count),
                    root_relation = root.relation,
                    concat_path = ".2".repeat(concat),
                    family_path = ".2".repeat(family_fact),
                    output_path = ".2".repeat(root_shape.constraint_count),
                ));
            }
        }
        count += 1;
    }
    source.push_str("end ExponentGeneratedSetup\n");
    (source, count)
}

/// Public monomial input encodings produced inside a structural family.
pub(super) fn encoding_input_proofs(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    bindings: &mxx_ir_core::ParamEnv,
) -> (String, usize) {
    use mxx_ir_core::node::ConstantMatrix;
    let mut source = format!(
        "import {}\nimport ExponentSetup\n\nnamespace ExponentGeneratedInputs\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\n",
        artifact.module_name
    );
    let mut count = 0;
    for (id, scope) in graph.source.scopes() {
        if scope.inputs().len() != 3 || scope.outputs().len() != 1 || scope.nodes().len() != 11 {
            continue;
        }
        if !matches!(id, FrozenGraphScopeId::ParallelBody { parent, .. } if matches!(parent.as_ref(), FrozenGraphScopeId::Root))
        {
            continue;
        }
        let types = &graph.scopes[id].wire_types;
        let ConcreteWireType::Matrix(output) = &types[&scope.outputs()[0]] else { continue };
        if output.rows != 1 || output.columns <= 2 {
            continue;
        }
        let mut names = Vec::new();
        for input in scope.inputs() {
            let matrix = match &types[input] {
                ConcreteWireType::Matrix(matrix) |
                ConcreteWireType::Preimage { matrix, .. } |
                ConcreteWireType::SmallMatrix { matrix, .. } => matrix,
                _ => {
                    names.push("unsupported");
                    continue;
                }
            };
            names.push(if matrix.rows == 1 && matrix.columns == 1 {
                "message"
            } else if matrix.rows == 1 && matrix.columns == 2 {
                "secret"
            } else if matrix.rows == output.columns && matrix.columns == output.columns {
                "action"
            } else {
                "unsupported"
            });
        }
        if names.contains(&"unsupported") ||
            !names.contains(&"message") ||
            !names.contains(&"secret") ||
            !names.contains(&"action")
        {
            continue;
        }
        let mut base = None;
        let mut cutoff = None;
        for node in scope.nodes() {
            match node.kind() {
                NodeKind::ConstantMatrix {
                    value: ConstantMatrix::Gadget { base: value, .. },
                    ..
                } => base = value.evaluate(bindings).ok(),
                NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                    cutoff = max_coefficient_bound.evaluate(bindings).ok()
                }
                _ => {}
            }
        }
        let (Some(base), Some(cutoff)) = (base, cutoff) else { continue };
        let q = &output.modulus;
        let n = output.ring_dimension;
        let columns = output.columns;
        let digits = columns / 2;
        let relation = &artifact.scope_relations[id];
        let inputs = format!("({}, ())", names.join(", "));
        source.push_str(&format!(
            "theorem encoding_input_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}.Params}} {{index : Nat}}\n    {{message : ExactMatrix {q} {n} 1 1}} {{secret : ExactMatrix {q} {n} 1 2}}\n    {{action : ExactMatrix {q} {n} {columns} {columns}}} {{output : ExactMatrix {q} {n} 1 {columns}}}\n    (runs : {relation} backend hashModel params index {inputs} output) :\n    ∃ gadget : ExactMatrix {q} {n} 2 {columns}, ∃ error : ErrorMatrix {n} 1 {columns},\n      MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget ∧\n      output = secret * (gadget * action) - (message 0 0) • (secret * gadget) + reduceMatrix {q} {n} 1 {columns} error ∧\n      CoeffBound error ({cutoff} : Nat) := by\n  rcases runs with ⟨gadget, sample, hg, he, ho⟩\n  obtain ⟨error, herror, hb⟩ := gaussian_error_witness he\n  refine ⟨gadget, error, hg, ?_, hb⟩\n  rw [ho, ← herror, ← Matrix.mul_assoc]\n  ext row column\n  have hr : row = 0 := Subsingleton.elim _ _\n  subst row\n  rfl\n\n",
            artifact.module_name,
        ));
        count += 1;
    }
    source.push_str("end ExponentGeneratedInputs\n");
    (source, count)
}

/// Prove bounds for direct cached decompositions exported by the entire producer.
/// Witness/constraint positions come from ir-core's proof metadata, including
/// guards, rather than an application-side copy of lowering rules.
pub(super) fn root_decomposition_proofs(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> (String, usize) {
    let mut source = format!(
        "import {}\nimport RuntimeDecomposition\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedRoot\nopen Mxx.Primitives MxxRuntime\n\n",
        artifact.module_name
    );
    let root = &artifact.root;
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let input_type = boundary_type(&root.inputs, root.input_count);
    let output_type = boundary_type(&root.outputs, root.output_count);
    source.push_str(&format!(
        "abbrev RootInput := {input_type}\nabbrev RootOutput := {output_type}\n\n"
    ));
    let input_type = "RootInput";
    let output_type = "RootOutput";
    let witness_tactic = "  obtain ⟨witness, facts⟩ := runs\n";
    let scope = graph.source.root_scope();
    let mut count = 0;
    for (output_name, value) in &root.outputs {
        source.push_str(&format!("-- Exported boundary {output_name:?}.\n"));
        let Some(node) = scope.node(value.wire.node) else { continue };

        if matches!(node.kind(), NodeKind::ParallelLoop(_)) {
            let child_id = FrozenGraphScopeId::ParallelBody {
                parent: Box::new(FrozenGraphScopeId::Root),
                owner: value.wire.node,
            };
            let Some(child) = graph.source.scopes().get(&child_id) else { continue };
            if child.nodes().len() != 2 || child.inputs().len() != 1 || child.outputs().len() != 1 {
                continue;
            }
            let child_output = child.node(child.outputs()[0].node).unwrap();
            if !matches!(child_output.kind(), NodeKind::GadgetDecompose { .. }) {
                continue;
            }
            let ConcreteWireType::IndexedFamily { element, count: lanes } = &value.wire_type else {
                continue
            };
            let Some(matrix) = element.matrix_type() else { continue };
            let Some(target) = graph.scopes[&child_id].wire_types[&child.inputs()[0]].matrix_type()
            else {
                continue
            };
            let position = shape.node_constraints[&value.wire.node].clone().last().unwrap();
            let q = &matrix.modulus;
            let n = matrix.ring_dimension;
            let rows = matrix.rows;
            let columns = matrix.columns;
            let target_rows = target.rows;
            let projection = format!("output{}.1", ".2".repeat(value.tuple_index));
            source.push_str(&format!(
                "theorem root_decomposition_family_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {parameter_type}}} {{input : {input_type}}} {{output : {output_type}}}\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (runs : {relation} backend hashModel params input output) :\n    ∀ i : Fin {lanes}, ∃ action : ErrorMatrix {n} {rows} {columns},\n      ({projection}) i = reduceMatrix {q} {n} {rows} {columns} action ∧\n      ActionGain action (denseRegularGain layout {n} {target_rows}) := by\n  obtain ⟨witness, facts⟩ := runs\n  have family := facts{family_path}.1\n  have outputEquation := facts{output_path}\n  rw [outputEquation]\n  intro i\n  obtain ⟨decomposition, hd, ho⟩ := family i\n  obtain ⟨action, equation, bounded⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd\n  exact ⟨action, ho.trans equation, bounded⟩\n\n",
                parameter_type = root.parameter_type,
                relation = root.relation,
                family_path = ".2".repeat(position),
                output_path = ".2".repeat(shape.constraint_count),
            ));
            count += 1;
            continue;
        }
        if !matches!(node.kind(), NodeKind::GadgetDecompose { .. }) {
            continue;
        }
        let matrix = match &value.wire_type {
            ConcreteWireType::Matrix(matrix) |
            ConcreteWireType::Preimage { matrix, .. } |
            ConcreteWireType::SmallMatrix { matrix, .. } => matrix,
            _ => continue,
        };
        let Some(arguments) = scope.arguments(node) else { continue };
        let ConcreteWireType::Matrix(target) =
            &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&arguments[0]]
        else {
            continue
        };
        let range = &shape.node_constraints[&value.wire.node];
        let Some(last_constraint) = range.clone().last() else { continue };
        let fact_tactic = format!(
            "  have fact_{last_constraint} := facts{}.1\n  have outputEquation := facts{}\n",
            ".2".repeat(last_constraint),
            ".2".repeat(shape.constraint_count)
        );
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let rows = matrix.rows;
        let columns = matrix.columns;
        let target_rows = target.rows;
        let projection = if root.output_count == 1 {
            "output".to_owned()
        } else {
            format!("output{}.1", ".2".repeat(value.tuple_index))
        };
        source.push_str(&format!(
            "theorem root_decomposition_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {parameter_type}}} {{input : {input_type}}} {{output : {output_type}}}\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (runs : {relation} backend hashModel params input output) :\n    ∃ action : ErrorMatrix {n} {rows} {columns},\n      {projection} = reduceMatrix {q} {n} {rows} {columns} action ∧\n      ActionGain action (denseRegularGain layout {n} {target_rows}) := by\n{witness_tactic}{fact_tactic}  obtain ⟨action, equation, bounded⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) fact_{last_constraint}\n  refine ⟨action, ?_, bounded⟩\n  rw [outputEquation]\n  exact equation\n\n",
            parameter_type = root.parameter_type,
            relation = root.relation,
        ));
        count += 1;
    }
    source.push_str("end ExponentGeneratedRoot\n");
    (source, count)
}
