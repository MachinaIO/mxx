//! Actual public-cache reconstruction for native program stages.
use super::{root_inputs::value_expression, root_readout_public::root_fact};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, WireRef},
};

/// Bind the exported cache port to its decomposition and actual product operands.
pub(super) fn program_cache_binding(
    graph: &ValidatedGraph,
    cache: WireRef,
) -> Result<(FrozenGraphScopeId, WireRef, WireRef, WireRef), String> {
    let scope_id = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: cache.node,
    };
    let scope = graph.source.scopes().get(&scope_id).ok_or("missing cache body")?;
    let decomposition =
        *scope.outputs().get(cache.port.0 as usize).ok_or("missing cache output port")?;
    let node = scope.node(decomposition.node).ok_or("missing cache decomposition")?;
    if !matches!(node.kind(), NodeKind::GadgetDecompose { .. }) {
        return Err("cache output is not a decomposition".into());
    }
    let target = scope.arguments(node).ok_or("missing decomposition target")?[0];
    let product = scope.node(target.node).ok_or("missing cache target product")?;
    if !matches!(
        product.kind(),
        NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
    ) {
        return Err("cache target is not the actual product".into());
    }
    let args = scope.arguments(product).ok_or("missing cache product operands")?;
    Ok((scope_id, decomposition, args[0], args[1]))
}

/// Resolve direct captures and captured-family lookups using their real loop bindings.
pub(super) fn parallel_operand(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    loop_wire: WireRef,
    child_wire: WireRef,
    index: &str,
) -> Result<(WireRef, usize, Option<(WireRef, String)>, String), String> {
    let root = graph.source.root_scope();
    let scope_id = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: loop_wire.node,
    };
    let scope = graph.source.scopes().get(&scope_id).ok_or("missing operand scope")?;
    let loop_node = root.node(loop_wire.node).ok_or("missing operand loop")?;
    let NodeKind::ParallelLoop(spec) = loop_node.kind() else {
        return Err("operand owner is not parallel".into())
    };
    let node = scope.node(child_wire.node).ok_or("missing child operand")?;
    let (input, lookup) = if matches!(node.kind(), NodeKind::FamilyGetDynamic) {
        let args = scope.arguments(node).ok_or("missing captured lookup arguments")?;
        let expression = value_expression(
            &format!("w_{}_{}", args[1].node.0, args[1].port.0),
            &artifact.scope_proofs[&scope_id].value_expressions,
        )?;
        (
            args[0],
            Some((
                child_wire,
                expression.replace(&format!("i_{}", spec.index_slot), &format!("({index}).val")),
            )),
        )
    } else {
        (child_wire, None)
    };
    let slot = scope
        .inputs()
        .iter()
        .position(|wire| *wire == input)
        .ok_or("operand source is not captured")?;
    let source = root.arguments(loop_node).ok_or("missing loop operands")?[slot];
    let value = value_expression(
        &format!("w_{}_{}", source.node.0, source.port.0),
        &artifact.scope_proofs[&FrozenGraphScopeId::Root].value_expressions,
    )?;
    let bound = match spec.input_modes[slot] {
        LoopInputMode::Broadcast => value,
        LoopInputMode::Zip => format!("({value}) ⟨({index}).val, by omega⟩"),
        LoopInputMode::ZipOffset { offset } => {
            format!("({value}) ⟨({index}).val + {offset}, by omega⟩")
        }
    };
    let bound = if let Some((_, expression)) = &lookup {
        format!("({bound}) ⟨Int.toNat ({expression}), by simp only [Int.ofNat_eq_natCast]; omega⟩")
    } else {
        bound
    };
    Ok((source, slot, lookup, bound))
}

/// Actual input and output tuples of one loop member, including every output port.
pub(super) fn parallel_arguments(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    loop_wire: WireRef,
    index: &str,
) -> Result<(u32, String, String), String> {
    let scope_id = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: loop_wire.node,
    };
    let scope = graph.source.scopes().get(&scope_id).ok_or("missing member scope")?;
    let NodeKind::ParallelLoop(spec) =
        graph.source.root_scope().node(loop_wire.node).ok_or("missing member loop")?.kind()
    else {
        return Err("member owner is not parallel".into())
    };
    let inputs = scope
        .inputs()
        .iter()
        .map(|wire| {
            parallel_operand(graph, artifact, loop_wire, *wire, index)
                .map(|(_, _, _, expression)| expression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = scope
        .outputs()
        .iter()
        .enumerate()
        .map(|(port, _)| {
            value_expression(
                &format!("w_{}_{}", loop_wire.node.0, port),
                &artifact.scope_proofs[&FrozenGraphScopeId::Root].value_expressions,
            )
            .map(|expression| format!("({expression}) {index}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tuple = |values: Vec<String>| match values.as_slice() {
        [] => "()".to_owned(),
        [one] => format!("({one})"),
        _ => values.iter().rev().fold("()".to_owned(), |tail, value| format!("({value}, {tail})")),
    };
    Ok((spec.index_slot, tuple(inputs), tuple(outputs)))
}

pub(super) fn root_program_setup(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing producer body")?;
    let record = shape.witness_record.as_ref().ok_or("missing producer witness")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut source = String::from(
        "import RootPrivate\nimport ExponentSetup\nimport ExponentRefreshRoutes\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramSetup\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\n",
    );
    let caches = artifact
        .root
        .outputs
        .iter()
        .filter(|(name, _)| {
            name.contains("-program-cache-family-") && name.ends_with("-body-1-k-rhs")
        })
        .collect::<Vec<_>>();
    if caches.is_empty() {
        return Err("missing public program caches".into());
    }
    let input_cache = artifact
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-setup-public-key-k-rhs-0"))
        .ok_or("missing native public-input cache")?;
    let count = caches.len();
    for (number, (name, output)) in
        caches.into_iter().chain(std::iter::once(input_cache)).enumerate()
    {
        let scope_id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: output.wire.node,
        };
        let scope = graph.source.scopes().get(&scope_id).ok_or("missing public cache scope")?;
        let decomposition = scope.outputs()[output.wire.port.0 as usize];
        let node = scope.node(decomposition.node).ok_or("missing cache output operation")?;
        let NodeKind::GadgetDecompose { base, digit_count, .. } = node.kind() else {
            return Err("cache output is not decomposition".into())
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|e| e.to_string())?;
        let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|e| e.to_string())?;
        let operands = if number < count {
            let (_, _, left, right) = program_cache_binding(graph, output.wire)?;
            vec![left, right]
        } else {
            vec![scope.arguments(node).ok_or("missing input cache target")?[0]]
        };
        let bindings = operands
            .iter()
            .map(|operand| parallel_operand(graph, artifact, output.wire, *operand, "i"))
            .collect::<Result<Vec<_>, _>>()?;
        let target = bindings
            .iter()
            .map(|(_, _, _, expression)| format!("({expression})"))
            .collect::<Vec<_>>()
            .join(" * ");
        let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
            return Err("cache is not a family".into())
        };
        let matrix = element.matrix_type().ok_or("cache element is not matrix")?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let cached = value(output.wire)?;
        let theorem = if number < count {
            format!("public_cache_{number}")
        } else {
            "public_inputs_cache".into()
        };
        let (slot, inputs, outputs) = parallel_arguments(graph, artifact, output.wire, "i")?;
        let child_shape = &artifact.scope_proofs[&scope_id];
        let context = format!(
            "(backend := backend) (hashModel := hashModel) (params := params) (i_{slot} := i.val) (inputs := {inputs}) (outputs := {outputs}) (witness := member)"
        );
        let hd = child_shape.node_facts[&decomposition.node]
            .last()
            .ok_or("missing decomposition fact")?;
        let output_fact =
            child_shape.output_fact.as_ref().ok_or("missing cache output equation")?;
        let projection = if scope.outputs().len() == 1 {
            "result".to_owned()
        } else {
            format!("result{}.1", ".2".repeat(output.wire.port.0 as usize))
        };
        source.push_str(&format!("-- Actual cache {name:?}.\ntheorem {theorem} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedPrivateRoot.RootInput}} {{outputs : ExponentGeneratedPrivateRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, gadget * (({cached}) i) = {target} := by\n  intro i\n  obtain ⟨member, memberFacts⟩ := ({} backend hashModel params i.val {inputs} {outputs}).mp (({}) i)\n  have hd := {hd} {context} memberFacts\n  have outputFact := {output_fact} {context} memberFacts\n  have cacheEquation : ({cached}) i = member.w_{}_{} := congrArg (fun result => {projection}) outputFact\n",child_shape.runs_iff_body,root_fact(artifact,output.wire)?,decomposition.node.0,decomposition.port.0,params=artifact.root.parameter_type));
        let mut equalities = Vec::new();
        for (side, (_, _, lookup, expression)) in bindings.iter().enumerate() {
            if let Some((get, index_expression)) = lookup {
                let get_fact = child_shape.node_facts[&get.node]
                    .last()
                    .ok_or("missing cache operand lookup fact")?;
                source.push_str(&format!("  have operand_{side} : member.w_{}_{} = {expression} := by\n    have lookup := {get_fact} {context} memberFacts\n    apply route_lookup_eq _ ?_ lookup\n    change {index_expression} = (Int.toNat ({index_expression}) : Int)\n    omega\n",get.node.0,get.port.0));
                equalities.push(format!("operand_{side}"));
            }
        }
        let rewrites = std::iter::once("cacheEquation".to_owned())
            .chain(equalities)
            .chain(std::iter::once("MxxRuntime.matrixMul".to_owned()))
            .collect::<Vec<_>>()
            .join(", ");
        source.push_str(&format!("  have reconstruction := gadget_runs_reconstruct (by decide) (by decide) gadgetRuns hd Fact.out\n  simpa only [{rewrites}] using reconstruction\n\n"));
    }
    source.push_str("end ExponentGeneratedProgramSetup\n");
    Ok(source)
}
