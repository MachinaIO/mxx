//! Actual pre-switch route sums, retaining per-column support and sampled errors.
use super::{
    root_inputs::value_expression,
    root_program_setup::{parallel_arguments, parallel_operand},
    root_readout_public::{root_fact, segmented_reduction},
};
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    graph::GraphScope,
    lean::LeanArtifact,
    node::NodeKind,
    types::{ConcreteWireType, WireRef},
};

fn child(wire: WireRef) -> FrozenGraphScopeId {
    FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    }
}

fn gather(graph: &ValidatedGraph, wire: WireRef) -> Option<(WireRef, Option<WireRef>)> {
    let root = graph.source.root_scope();
    let scope = graph.source.scopes().get(&child(wire))?;
    let output = *scope.outputs().get(wire.port.0 as usize)?;
    let parent = root.node(wire.node)?;
    let parents = root.arguments(parent)?;
    let map = |arg| scope.inputs().iter().position(|input| *input == arg).map(|i| parents[i]);
    if let Some(slot) = scope.inputs().iter().position(|input| *input == output) {
        let NodeKind::ParallelLoop(spec) = parent.kind() else { return None };
        if matches!(
            spec.input_modes[slot],
            mxx_ir_core::node::LoopInputMode::Zip |
                mxx_ir_core::node::LoopInputMode::ZipOffset { .. }
        ) {
            return Some((parents[slot], None));
        }
    }
    let node = scope.node(output.node)?;
    if !matches!(node.kind(), NodeKind::FamilyGetDynamic) {
        return None;
    }
    let args = scope.arguments(node)?;
    Some((map(args[0])?, Some(map(args[1])?)))
}

/// Bind a route view to its actual source family and integer index expression.
pub(super) fn route_binding(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    index: &str,
) -> Result<Option<(WireRef, String)>, String> {
    let Some((family, indices)) = gather(graph, output) else { return Ok(None) };
    let expression = if let Some(indices) = indices {
        let scope = &graph.source.scopes()[&child(indices)];
        if !scope.inputs().is_empty() {
            return Err("route indices depend on non-index inputs".into());
        }
        let NodeKind::ParallelLoop(spec) =
            graph.source.root_scope().node(indices.node).ok_or("missing route index loop")?.kind()
        else {
            return Err("route indices are not parallel".into())
        };
        let mut values = artifact.scope_proofs[&child(indices)].value_expressions.clone();
        values.insert(format!("i_{}", spec.index_slot), format!("({index}).val"));
        let wire = scope.outputs()[indices.port.0 as usize];
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &values)?
    } else {
        let scope = &graph.source.scopes()[&child(output)];
        let wire = scope.outputs()[output.port.0 as usize];
        let slot = scope
            .inputs()
            .iter()
            .position(|input| *input == wire)
            .ok_or("identity route output is not an input")?;
        let NodeKind::ParallelLoop(spec) = graph
            .source
            .root_scope()
            .node(output.node)
            .ok_or("missing identity route loop")?
            .kind()
        else {
            return Err("identity route is not parallel".into())
        };
        match spec.input_modes[slot] {
            mxx_ir_core::node::LoopInputMode::Zip => format!("(({index}).val : Int)"),
            mxx_ir_core::node::LoopInputMode::ZipOffset { offset } => {
                format!("((({index}).val : Int) + {offset})")
            }
            _ => return Err("identity route is not indexed".into()),
        }
    };
    Ok(Some((family, expression)))
}

fn arguments(scope: &GraphScope, wire: WireRef) -> Result<Vec<WireRef>, String> {
    scope
        .node(wire.node)
        .and_then(|node| scope.arguments(node))
        .ok_or("missing refresh operands".into())
}

/// Shared checked reduction preserves the exact scalar route theorem boundary.
fn reduction(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    segments: usize,
    name: &str,
) -> Result<(String, WireRef, usize), String> {
    let (proof, leaf, width) = segmented_reduction(graph, artifact, output, segments, "segment")?;
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let body = shape.body_relation.as_ref().ok_or("missing route body")?;
    let record = shape.witness_record.as_ref().ok_or("missing route witness")?;
    let source = format!(
        "theorem {name} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (segment : Fin {segments}) : ({}) segment = ∑ i : Fin {width}, ({}) ⟨segment.val * {width} + i.val, by omega⟩ := by\n{proof}  exact reduced\n\n",
        artifact.root.parameter_type,
        artifact.root.input_type,
        artifact.root.output_type,
        value(output)?,
        value(leaf)?
    );
    Ok((source, leaf, width))
}

fn route_gather(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    prefix: &str,
) -> Result<String, String> {
    let (family, indices) = gather(graph, output).ok_or("missing actual route gather")?;
    let scope = &graph.source.scopes()[&child(output)];
    let output_wire = scope.outputs()[output.port.0 as usize];
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let member_shape = &artifact.scope_proofs[&child(output)];
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let count = |wire: WireRef| match &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&wire] {
        ConcreteWireType::IndexedFamily { count, .. } => Ok(*count),
        _ => Err("route gather value not family".to_owned()),
    };
    let (slot, inputs, outputs) = parallel_arguments(graph, artifact, output, "i")?;
    let context = format!(
        "(backend := backend) (hashModel := hashModel) (params := params) (i_{slot} := i.val) (inputs := {inputs}) (outputs := {outputs}) (witness := member)"
    );
    let output_fact = member_shape.output_fact.as_ref().ok_or("missing route output fact")?;
    let projection = if scope.outputs().len() == 1 {
        "result".to_owned()
    } else {
        format!("result{}.1", ".2".repeat(output.port.0 as usize))
    };
    let (_, expression) =
        route_binding(graph, artifact, output, "i")?.ok_or("missing route binding")?;
    let mut preparation = String::new();
    if let Some(indices) = indices {
        let index_scope = &graph.source.scopes()[&child(indices)];
        if !index_scope.inputs().is_empty() {
            return Err("route gather indices depend on non-index inputs".into());
        }
        let index_shape = &artifact.scope_proofs[&child(indices)];
        let (index_slot, index_inputs, index_outputs) =
            parallel_arguments(graph, artifact, indices, "indexPosition")?;
        let index_output_fact =
            index_shape.output_fact.as_ref().ok_or("missing route index fact")?;
        preparation.push_str(&format!("  let indexPosition : Fin {} := ⟨i.val, by have bound := i.isLt; omega⟩\n  obtain ⟨indexWitness, indexFacts⟩ := ({} backend hashModel params indexPosition.val {index_inputs} {index_outputs}).mp (({}) indexPosition)\n  have indexEq : ({}) indexPosition = {expression} := {index_output_fact} (backend := backend) (hashModel := hashModel) (params := params) (i_{index_slot} := indexPosition.val) (inputs := {index_inputs}) (outputs := {index_outputs}) (witness := indexWitness) indexFacts\n",count(indices)?,index_shape.runs_iff_body,root_fact(artifact,indices)?,value(indices)?));
    }
    let record = shape.witness_record.as_ref().ok_or("missing route record")?;
    let body = shape.body_relation.as_ref().ok_or("missing route body")?;
    let mut source = format!(
        "theorem {prefix}route_gather_{} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (i : Fin {}) : ∃ bound : ({expression}).toNat < {},\n    ({}) i = ({}) ⟨({expression}).toNat, bound⟩ := by\n{preparation}  obtain ⟨member, memberFacts⟩ := ({} backend hashModel params i.val {inputs} {outputs}).mp (({}) i)\n  have outputFact := {output_fact} {context} memberFacts\n",
        output.node.0,
        artifact.root.parameter_type,
        artifact.root.input_type,
        artifact.root.output_type,
        count(output)?,
        count(family)?,
        value(output)?,
        value(family)?,
        member_shape.runs_iff_body,
        root_fact(artifact, output)?
    );
    if indices.is_some() {
        let lookup_fact =
            member_shape.node_facts[&output_wire.node].last().ok_or("missing route lookup fact")?;
        source.push_str(&format!("  have lookup := {lookup_fact} {context} memberFacts\n  have outputEq := congrArg (fun result => {projection}) outputFact\n  obtain ⟨actual, hindex, hvalue⟩ := lookup\n  rw [indexEq] at hindex\n  have bound : ({expression}).toNat < {} := by have actualBound := actual.isLt; omega\n  refine ⟨bound, outputEq.trans ?_⟩\n  have same : actual = (⟨({expression}).toNat, bound⟩ : Fin {}) := by apply Fin.ext; dsimp at hindex ⊢; omega\n  simpa only [same] using hvalue\n\n",count(family)?,count(family)?));
    } else {
        let (_, input_slot, _, bound_operand) =
            parallel_operand(graph, artifact, output, output_wire, "i")?;
        let NodeKind::ParallelLoop(spec) =
            graph.source.root_scope().node(output.node).ok_or("missing route view loop")?.kind()
        else {
            unreachable!()
        };
        let position = match spec.input_modes[input_slot] {
            mxx_ir_core::node::LoopInputMode::Zip => "i.val".to_owned(),
            mxx_ir_core::node::LoopInputMode::ZipOffset { offset } => format!("i.val + {offset}"),
            _ => return Err("route view is not indexed".into()),
        };
        source.push_str(&format!("  have outputEq : ({}) i = {bound_operand} := congrArg (fun result => {projection}) outputFact\n  have bound : ({expression}).toNat < {} := by have bound := i.isLt; omega\n  refine ⟨bound, outputEq.trans ?_⟩\n  exact congrArg (fun position : Fin {} => ({}) position) (by apply Fin.ext; change {position} = ({expression}).toNat; omega)\n\n",value(output)?,count(family)?,count(family)?,value(family)?));
    }
    Ok(source)
}

fn route_product(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    name: &str,
) -> Result<String, String> {
    use mxx_ir_core::node::MatrixBinaryOp;
    let scope = &graph.source.scopes()[&child(output)];
    let product_wire = scope.outputs()[output.port.0 as usize];
    let product = scope.node(product_wire.node).ok_or("missing route product")?;
    if !matches!(
        product.kind(),
        NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
    ) {
        return Err("route leaf is not actual matrix product".into());
    }
    let operands = arguments(scope, product_wire)?;
    let bound = operands
        .iter()
        .map(|wire| parallel_operand(graph, artifact, output, *wire, "i"))
        .collect::<Result<Vec<_>, _>>()?;
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let member_shape = &artifact.scope_proofs[&child(output)];
    let value = value_expression(
        &format!("w_{}_{}", output.node.0, output.port.0),
        &shape.value_expressions,
    )?;
    let ConcreteWireType::IndexedFamily { count, .. } =
        &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&output]
    else {
        return Err("route product is not a family".into());
    };
    let record = shape.witness_record.as_ref().ok_or("missing route record")?;
    let body = shape.body_relation.as_ref().ok_or("missing route body")?;
    let (slot, inputs, outputs) = parallel_arguments(graph, artifact, output, "i")?;
    let context = format!(
        "(backend := backend) (hashModel := hashModel) (params := params) (i_{slot} := i.val) (inputs := {inputs}) (outputs := {outputs}) (witness := member)"
    );
    let output_fact = member_shape.output_fact.as_ref().ok_or("missing product output fact")?;
    let projection = if scope.outputs().len() == 1 {
        "result".to_owned()
    } else {
        format!("result{}.1", ".2".repeat(output.port.0 as usize))
    };
    let mut source = format!(
        "theorem {name} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (i : Fin {count}) : ({value}) i = ({}) * ({}) := by\n  obtain ⟨member, memberFacts⟩ := ({} backend hashModel params i.val {inputs} {outputs}).mp (({}) i)\n  have outputFact := {output_fact} {context} memberFacts\n  have productEq := congrArg (fun result => {projection}) outputFact\n",
        artifact.root.parameter_type,
        artifact.root.input_type,
        artifact.root.output_type,
        bound[0].3,
        bound[1].3,
        member_shape.runs_iff_body,
        root_fact(artifact, output)?
    );
    let mut rewrites = vec!["MxxRuntime.matrixMul".to_owned()];
    for (side, (_, _, lookup, expression)) in bound.iter().enumerate() {
        if let Some((get, index)) = lookup {
            let get_fact = member_shape.node_facts[&get.node]
                .last()
                .ok_or("missing product operand lookup")?;
            source.push_str(&format!("  have operand_{side} : member.w_{}_{} = {expression} := by\n    have lookup := {get_fact} {context} memberFacts\n    apply route_lookup_eq _ ?_ lookup\n    change ({index}) = (Int.toNat ({index}) : Int)\n    simp only [Int.ofNat_eq_natCast]\n    omega\n",get.node.0,get.port.0));
            rewrites.push(format!("operand_{side}"));
        }
    }
    source.push_str(&format!("  simpa only [{}] using productEq\n\n", rewrites.join(", ")));
    Ok(source)
}

pub(super) fn root_refresh_source(
    pregraph: &ValidatedGraph,
    preartifact: &LeanArtifact,
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    use mxx_ir_core::{ParamEnv, node::MatrixBinaryOp};
    let root = graph.source.root_scope();
    let crt = root
        .nodes()
        .iter()
        .find(|node| matches!(node.kind(), NodeKind::CrtRecompose { .. }))
        .ok_or("missing actual refresh CRT")?;
    let levels = root.arguments(crt).ok_or("missing refresh CRT levels")?;
    let mut source = format!(
        "import {}\nimport {}\nimport ExponentRefreshRoutes\nimport ExponentReadout\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshSource\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\n",
        artifact.module_name, preartifact.module_name
    );
    let mut emitted = std::collections::BTreeSet::new();
    let mut products = std::collections::BTreeSet::new();
    let mut gathers = std::collections::BTreeSet::new();
    for (slot, level) in levels.iter().enumerate() {
        let getter = root.node(level.node).ok_or("missing CRT level")?;
        let NodeKind::FamilyGetStatic { index } = getter.kind() else {
            return Err("refresh CRT level not grouped getter".into())
        };
        let local = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic refresh slot")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "invalid refresh slot")?;
        let family = arguments(root, *level)?[0];
        let group = &graph.source.scopes()[&child(family)];
        let difference = arguments(group, group.outputs()[0])?;
        let switch = group.node(difference[0].node).ok_or("missing actual ciphertext switch")?;
        if !matches!(switch.kind(), NodeKind::ModulusSwitch { .. }) {
            return Err("refresh level lacks switch".into());
        }
        let sum = group.arguments(switch).ok_or("missing pre-switch sum")?[0];
        let final_terms = arguments(group, sum)?;
        let first_terms = arguments(group, final_terms[0])?;
        let grouped_terms = [first_terms[0], first_terms[1], final_terms[1]];
        let parents = arguments(root, family)?;
        for (role, term) in ["state", "mask", "fresh"].into_iter().zip(grouped_terms) {
            let capture = group
                .inputs()
                .iter()
                .position(|input| *input == term)
                .ok_or("refresh term is not a group input")?;
            let packed = parents[capture];
            let pack = root.node(packed.node).ok_or("missing refresh term pack")?;
            if !matches!(pack.kind(), NodeKind::FamilyPack { .. }) {
                return Err("refresh term lacks exact slot pack".into());
            }
            let actual =
                *arguments(root, packed)?.get(local).ok_or("refresh slot outside group")?;
            if role == "state" {
                continue;
            }
            if !matches!(
                root.node(actual.node).map(|node| node.kind()),
                Some(NodeKind::MatrixBinary(MatrixBinaryOp::Add))
            ) {
                return Err("refresh route lacks public offset".into());
            }
            let reduced = arguments(root, actual)?[0];
            if !matches!(
                root.node(reduced.node).map(|node| node.kind()),
                Some(NodeKind::FamilyGetStatic { .. })
            ) {
                return Err("refresh routed sum is not selected family".into());
            }
            let result = arguments(root, reduced)?[0];
            if !emitted.insert(result) {
                continue;
            }
            let ConcreteWireType::IndexedFamily { count, .. } =
                &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&result]
            else {
                return Err("refresh result not family".into())
            };
            let (proof, mut leaf, _) =
                reduction(graph, artifact, result, *count, &format!("{role}_reduction_{slot}"))?;
            source.push_str(&proof);
            while let Some((family, _)) = gather(graph, leaf) {
                if gathers.insert(leaf) {
                    source.push_str(&route_gather(graph, artifact, leaf, "")?);
                }
                leaf = family;
            }
            if products.insert(leaf) {
                let product_scope = &graph.source.scopes()[&child(leaf)];
                let operands =
                    arguments(product_scope, product_scope.outputs()[leaf.port.0 as usize])?;
                let (mut input, _, _, _) =
                    parallel_operand(graph, artifact, leaf, operands[0], "i")?;
                while let Some((family, _)) = gather(graph, input) {
                    if gathers.insert(input) {
                        source.push_str(&route_gather(graph, artifact, input, "")?);
                    }
                    input = family;
                }
                source.push_str(&route_product(
                    graph,
                    artifact,
                    leaf,
                    &format!("{role}_product_{slot}"),
                )?);
            }
        }
    }
    source.push_str(&public_reductions(pregraph, preartifact)?);
    source.push_str("end ExponentGeneratedRefreshSource\n");
    Ok(source)
}

fn public_reductions(graph: &ValidatedGraph, artifact: &LeanArtifact) -> Result<String, String> {
    use mxx_ir_core::ParamEnv;
    let root = graph.source.root_scope();
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let mut source = String::new();
    let mut emitted = std::collections::BTreeSet::new();
    let mut gathers = std::collections::BTreeSet::new();
    for (name, target) in &artifact.root.outputs {
        let Some((_, slot)) = name.rsplit_once("-refresh-target-") else { continue };
        let slot = slot.parse::<usize>().map_err(|_| "invalid refresh target slot")?;
        let getter = root.node(target.wire.node).ok_or("missing target getter")?;
        let NodeKind::FamilyGetStatic { index } = getter.kind() else {
            return Err("public target not selected group".into())
        };
        let local = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic public target slot")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "negative public target slot")?;
        let family = arguments(root, target.wire)?[0];
        let group = &graph.source.scopes()[&child(family)];
        let difference = arguments(group, group.outputs()[0])?;
        let high = arguments(group, difference[0])?[0];
        let capture = group
            .inputs()
            .iter()
            .position(|input| *input == high)
            .ok_or("public switch source not captured")?;
        let packed = arguments(root, family)?[capture];
        if !matches!(
            root.node(packed.node).map(|node| node.kind()),
            Some(NodeKind::FamilyPack { .. })
        ) {
            return Err("public switch group not exact pack".into());
        }
        let high_get =
            *arguments(root, packed)?.get(local).ok_or("public switch local slot missing")?;
        let sum_family = arguments(root, high_get)?[0];
        let sum_scope = &graph.source.scopes()[&child(sum_family)];
        let sum_args = arguments(sum_scope, sum_scope.outputs()[0])?;
        let first_args = arguments(sum_scope, sum_args[0])?;
        for (role, term) in [("mask", first_args[1]), ("fresh", sum_args[1])] {
            let capture = sum_scope
                .inputs()
                .iter()
                .position(|input| *input == term)
                .ok_or("public route sum operand not captured")?;
            let packed = arguments(root, sum_family)?[capture];
            if !matches!(
                root.node(packed.node).map(|node| node.kind()),
                Some(NodeKind::FamilyPack { .. })
            ) {
                return Err("public routed sums not packed".into());
            }
            let selected =
                *arguments(root, packed)?.get(slot).ok_or("public route slot missing")?;
            if !matches!(
                root.node(selected.node).map(|node| node.kind()),
                Some(NodeKind::FamilyGetStatic { .. })
            ) {
                return Err("public route not selected reduction".into());
            }
            let result = arguments(root, selected)?[0];
            if !emitted.insert(result) {
                continue;
            }
            let ConcreteWireType::IndexedFamily { count, .. } = &types[&result] else {
                return Err("public route result not family".into())
            };
            let (proof, mut leaf, _) = reduction(
                graph,
                artifact,
                result,
                *count,
                &format!("pre_{role}_reduction_{slot}"),
            )?;
            source.push_str(&proof);
            while let Some((family, _)) = gather(graph, leaf) {
                if gathers.insert(leaf) {
                    source.push_str(&route_gather(graph, artifact, leaf, "pre_")?);
                }
                leaf = family;
            }
            let product_scope = &graph.source.scopes()[&child(leaf)];
            let operands = arguments(product_scope, product_scope.outputs()[leaf.port.0 as usize])?;
            let (mut input, ..) = parallel_operand(graph, artifact, leaf, operands[0], "i")?;
            while let Some((family, _)) = gather(graph, input) {
                if gathers.insert(input) {
                    source.push_str(&route_gather(graph, artifact, input, "pre_")?);
                }
                input = family;
            }
        }
    }
    if emitted.is_empty() {
        return Err("no public refresh route reductions".into());
    }
    Ok(source)
}
