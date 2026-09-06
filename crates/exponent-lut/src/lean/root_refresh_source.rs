//! Actual pre-switch route sums, retaining per-column support and sampled errors.
use super::root_inputs::value_expression;
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

fn gather(graph: &ValidatedGraph, wire: WireRef) -> Option<(WireRef, WireRef)> {
    let root = graph.source.root_scope();
    let scope = graph.source.scopes().get(&child(wire))?;
    let output = scope.node(scope.outputs()[0].node)?;
    if !matches!(output.kind(), NodeKind::FamilyGetDynamic) {
        return None;
    }
    let args = scope.arguments(output)?;
    let parents = root.arguments(root.node(wire.node)?)?;
    let map = |arg| scope.inputs().iter().position(|input| *input == arg).map(|i| parents[i]);
    Some((map(args[0])?, map(args[1])?))
}

fn arguments(scope: &GraphScope, wire: WireRef) -> Result<Vec<WireRef>, String> {
    scope
        .node(wire.node)
        .and_then(|node| scope.arguments(node))
        .ok_or("missing refresh operands".into())
}

/// Every layer is proved for a symbolic position, so large mask families do
/// not expand into one proof term per label or coefficient.
fn reduction(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    segments: usize,
    name: &str,
) -> Result<(String, WireRef, usize), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let maximum_fact = std::cell::Cell::new(0usize);
    let fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing route reduction fact")?;
        maximum_fact.set(maximum_fact.get().max(position));
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let count = |wire| -> Result<usize, String> {
        let ConcreteWireType::IndexedFamily { count, .. } = &types[&wire] else {
            return Err("route reduction not a family".into())
        };
        Ok(*count)
    };
    if segments == 0 || count(output)? != segments {
        return Err("route final segment count mismatch".into());
    }
    let mut current = output;
    let mut levels = Vec::new();
    loop {
        let Some(scope) = graph.source.scopes().get(&child(current)) else { break };
        let parents = arguments(root, current)?;
        if parents.len() != 2 || scope.outputs().len() != 1 {
            break;
        }
        let (Some((left, left_index)), Some((right, right_index))) =
            (gather(graph, parents[0]), gather(graph, parents[1]))
        else {
            break
        };
        if left != right {
            break;
        }
        let width = count(left)? / segments;
        let next = count(current)? / segments;
        if count(left)? % segments != 0 ||
            count(current)? % segments != 0 ||
            next != width.div_ceil(2)
        {
            return Err("route reduction has inconsistent segments".into());
        }
        levels.push((current, parents[0], parents[1], left_index, right_index, left, width, next));
        current = left;
    }
    let width = count(current)? / segments;
    if width == 0 {
        return Err("empty route segment".into());
    }
    let record = shape.witness_record.as_ref().ok_or("missing route root record")?;
    let body = shape.body_relation.as_ref().ok_or("missing route root body")?;
    let relation = &artifact.root;
    let binders = format!(
        "{{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (segment : Fin {segments})",
        relation.parameter_type, relation.input_type, relation.output_type,
    );
    let public_source = format!(
        "theorem {name} {binders} :\n    ({}) segment = ∑ i : Fin {width}, ({}) ⟨segment.val * {width} + i.val, by clear * - segment i; omega⟩ := by\n  classical\n",
        value(output)?,
        value(current)?
    );
    let mut layers_source = String::new();
    let mut source = String::new();
    for (result, left, right, left_index, right_index, previous, old, next) in levels.iter().rev() {
        source.clear();
        maximum_fact.set(0);
        let id = result.node.0;
        let old_count = segments * old;
        let next_count = segments * next;
        let previous_value = value(*previous)?;
        let result_value = value(*result)?;
        source.push_str(&format!("private theorem {name}_layer_{id} {binders} : (∑ i : Fin {next}, ({}) ⟨segment.val * {next} + i.val, by clear * - segment i; omega⟩) =\n      ∑ i : Fin {old}, ({}) ⟨segment.val * {old} + i.val, by clear * - segment i; omega⟩ := by\n    apply pairwise_sum (n := {old}) (f := fun i : Fin {old} => ({previous_value}) ⟨segment.val * {old} + i.val, by clear * - segment i; omega⟩) (g := fun i : Fin {next} => ({result_value}) ⟨segment.val * {next} + i.val, by clear * - segment i; omega⟩)\n    intro i\n    have leftBound : 2 * i.val < {old} := by clear * - segment i; omega\n    have indexDiv : ((segment.val : Int) * {next} + i.val) / {next} = segment.val := by\n      clear * - segment i\n      omega\n    have indexMod : ((segment.val : Int) * {next} + i.val) % {next} = i.val := by\n      clear * - segment i\n      omega\n",value(*result)?,value(*previous)?));
        for (side, wire, indices, position) in [
            ("left", *left, *left_index, "2 * i.val".to_string()),
            ("right", *right, *right_index, format!("(2 * i.val + 1) % {old}")),
        ] {
            let ir = &artifact.scope_relations[&child(indices)];
            let gr = &artifact.scope_relations[&child(wire)];
            source.push_str(&format!("    have {side}Index : ({}) ⟨segment.val * {next} + i.val, by clear * - segment i; omega⟩ = ((segment.val * {old} + {position} : Nat) : Int) := by\n      have h := {} (⟨segment.val * {next} + i.val, by clear * - segment i; omega⟩ : Fin {next_count})\n      dsimp only [{ir}, {ir}.constraints_0] at h\n      norm_num only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, Int.natCast_mod, indexDiv, indexMod] at h ⊢\n      clear * - segment i h leftBound indexDiv indexMod\n      omega\n    have {side}Get : ({}) ⟨segment.val * {next} + i.val, by clear * - segment i; omega⟩ = ({}) ⟨segment.val * {old} + {position}, by clear * - segment i; omega⟩ := by\n      have h := {} (⟨segment.val * {next} + i.val, by clear * - segment i; omega⟩ : Fin {next_count})\n      dsimp only [{gr}, {gr}.constraints_0] at h\n      obtain ⟨sample, hlo, hhi, hg, hout⟩ := h\n      exact hout.trans (route_lookup_eq (⟨segment.val * {old} + {position}, by clear * - segment i; omega⟩ : Fin {old_count}) {side}Index hg)\n",value(indices)?,fact(indices)?,value(wire)?,value(*previous)?,fact(wire)?));
        }
        let rr = &artifact.scope_relations[&child(*result)];
        let scope = &graph.source.scopes()[&child(*result)];
        let is_select = matches!(
            scope.node(scope.outputs()[0].node).map(|n| n.kind()),
            Some(NodeKind::Select { .. })
        );
        source.push_str(&format!("    have h := {} (⟨segment.val * {next} + i.val, by clear * - segment i; omega⟩ : Fin {next_count})\n    by_cases rightBound : 2 * i.val + 1 < {old}\n",fact(*result)?));
        if is_select {
            source.push_str(&format!("    · have selected : (i.val : Int) ≠ {} := by clear * - segment i rightBound; omega\n      simp only [{rr}, {rr}.constraints_0, Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, indexMod] at h\n      simp only [selected, decide_false, Bool.false_eq_true, ↓reduceIte] at h\n      obtain ⟨selectedValue, hnonzero, hlo, hhi, hlength, hselect, hout⟩ := h\n      have selectedEq := route_select_zero hselect\n      simpa only [leftBound, rightBound, ↓reduceDIte, MxxRuntime.matrixAdd, leftGet, rightGet, Nat.mod_eq_of_lt rightBound] using hout.trans selectedEq\n    · have selected : (i.val : Int) = {} := by clear * - segment i rightBound; omega\n      simp only [{rr}, {rr}.constraints_0, Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, indexMod] at h\n      simp only [selected, decide_true, ↓reduceIte] at h\n      obtain ⟨selectedValue, hnonzero, hlo, hhi, hlength, hselect, hout⟩ := h\n      have selectedEq := route_select_one hselect\n      simpa only [leftBound, rightBound, ↓reduceDIte, add_zero, leftGet] using hout.trans selectedEq\n",next-1,next-1));
        } else {
            source.push_str(&format!("    · simpa only [{rr}, {rr}.constraints_0, MxxRuntime.matrixAdd, leftBound, rightBound, ↓reduceDIte, leftGet, rightGet, Nat.mod_eq_of_lt rightBound] using h\n    · clear * - i rightBound\n      omega\n"));
        }
        source = source
            .lines()
            .map(|line| line.strip_prefix("  ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        source.push('\n');
        source = source.replacen(" := by\n", " := by\n  classical\n", 1);
        let mut tails = String::new();
        for block in (1..=maximum_fact.get() / 64).rev() {
            let position = block * 64;
            source = source
                .replace(&format!("facts{}", ".2".repeat(position)), &format!("facts_{position}"));
        }
        for position in (64..=maximum_fact.get()).step_by(64) {
            let previous = if position == 64 {
                "facts".to_string()
            } else {
                format!("facts_{}", position - 64)
            };
            tails.push_str(&format!("  have facts_{position} := {previous}{}\n", ".2".repeat(64)));
        }
        source = source.replacen("  classical\n", &format!("  classical\n{tails}"), 1);
        layers_source.push_str(&source);
        layers_source.push('\n');
    }
    let mut source = public_source;
    if levels.is_empty() {
        source.push_str("  simp\n");
    } else {
        source.push_str("  calc\n");
        let final_value = value(output)?;
        source.push_str(&format!("    ({final_value}) segment = ∑ i : Fin 1, ({final_value}) ⟨segment.val * 1 + i.val, by clear * - segment i; omega⟩ := by\n      rw [Fin.sum_univ_one]\n      apply congrArg\n      apply Fin.ext\n      simp only [Nat.mul_one, Nat.add_zero, Fin.val_zero]\n"));
        for (result, _, _, _, _, previous, old, _) in &levels {
            source.push_str(&format!("    _ = ∑ i : Fin {old}, ({}) ⟨segment.val * {old} + i.val, by clear * - segment i; omega⟩ := {name}_layer_{} witness facts segment\n",value(*previous)?,result.node.0));
        }
    }
    source.push('\n');
    layers_source.push_str(&source);
    Ok((layers_source, current, width))
}

fn route_gather(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    prefix: &str,
) -> Result<String, String> {
    let (family, indices) = gather(graph, output).ok_or("missing actual route gather")?;
    let index_scope = &graph.source.scopes()[&child(indices)];
    if !index_scope.inputs().is_empty() {
        return Err("route gather indices depend on non-index inputs".into());
    }
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let index_shape = &artifact.scope_proofs[&child(indices)];
    let mut values = index_shape.value_expressions.clone();
    let NodeKind::ParallelLoop(spec) =
        graph.source.root_scope().node(indices.node).ok_or("missing route indices")?.kind()
    else {
        return Err("route indices not parallel".into());
    };
    values.insert(format!("i_{}", spec.index_slot), "i.val".into());
    let wire = index_scope.outputs()[0];
    let expression = value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &values)?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let count = |wire: WireRef| -> Result<usize, String> {
        match &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&wire] {
            ConcreteWireType::IndexedFamily { count, .. } => Ok(*count),
            _ => Err("route gather value not family".into()),
        }
    };
    let fact = |wire: WireRef| -> Result<String, String> {
        let position =
            shape.node_constraints[&wire.node].clone().last().ok_or("missing route gather fact")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let ir = &artifact.scope_relations[&child(indices)];
    let gr = &artifact.scope_relations[&child(output)];
    let record = shape.witness_record.as_ref().ok_or("missing route record")?;
    let body = shape.body_relation.as_ref().ok_or("missing route body")?;
    Ok(format!(
        "theorem {prefix}route_gather_{} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (i : Fin {}) : ∃ bound : ({expression}).toNat < {},\n    ({}) i = ({}) ⟨({expression}).toNat, bound⟩ := by\n  have indexFacts := {} i\n  dsimp only [{ir}, {ir}.constraints_0] at indexFacts\n  have indexEq := indexFacts{}\n  have h := {} i\n  dsimp only [{gr}, {gr}.constraints_0] at h\n  obtain ⟨sample, hlo, hhi, hg, hout⟩ := h\n  rw [indexEq] at hlo hhi hg\n  have bound : ({expression}).toNat < {} := by clear * - hlo hhi; omega\n  refine ⟨bound, hout.trans ?_⟩\n  apply route_lookup_eq (⟨({expression}).toNat, bound⟩ : Fin {}) _ hg\n  exact (Int.toNat_of_nonneg hlo).symm\n\n",
        output.node.0,
        artifact.root.parameter_type,
        artifact.root.input_type,
        artifact.root.output_type,
        count(output)?,
        count(family)?,
        value(output)?,
        value(family)?,
        fact(indices)?,
        ".2".repeat(index_shape.constraint_count),
        fact(output)?,
        count(family)?,
        count(family)?
    ))
}

fn route_product(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    output: WireRef,
    name: &str,
) -> Result<String, String> {
    use mxx_ir_core::node::{LoopInputMode, MatrixBinaryOp};
    let root = graph.source.root_scope();
    let scope = &graph.source.scopes()[&child(output)];
    let product = scope.node(scope.outputs()[0].node).ok_or("missing route product")?;
    if !matches!(
        product.kind(),
        NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
    ) {
        return Err("route leaf is not actual matrix product".into());
    }
    let parent = root.node(output.node).ok_or("missing route product parent")?;
    let NodeKind::ParallelLoop(loop_spec) = parent.kind() else {
        return Err("route product is not parallel".into());
    };
    let parents = arguments(root, output)?;
    let operands = arguments(scope, scope.outputs()[0])?;
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut captured = Vec::new();
    for operand in operands {
        let index = scope
            .inputs()
            .iter()
            .position(|input| *input == operand)
            .ok_or("route product operand is not captured")?;
        if loop_spec.input_modes[index] != LoopInputMode::Zip {
            return Err("route product operand is not zipped".into());
        }
        captured.push(value(parents[index])?);
    }
    let ConcreteWireType::IndexedFamily { count, .. } =
        &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&output]
    else {
        return Err("route product is not a family".into());
    };
    let position =
        shape.node_constraints[&output.node].clone().last().ok_or("missing route product fact")?;
    let record = shape.witness_record.as_ref().ok_or("missing route record")?;
    let body = shape.body_relation.as_ref().ok_or("missing route body")?;
    let relation = &artifact.scope_relations[&child(output)];
    Ok(format!(
        "theorem {name} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (i : Fin {count}) : ({}) i = ({}) i * ({}) i := by\n  have h := facts{}.1 i\n  simpa only [{relation}, {relation}.constraints_0, MxxRuntime.matrixMul] using h\n\n",
        artifact.root.parameter_type,
        artifact.root.input_type,
        artifact.root.output_type,
        value(output)?,
        captured[0],
        captured[1],
        ".2".repeat(position)
    ))
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
        "import {}\nimport {}\nimport ExponentRefreshRoutes\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshSource\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\n",
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
                let operands = arguments(product_scope, product_scope.outputs()[0])?;
                let parents = arguments(root, leaf)?;
                let mut input = parents[product_scope
                    .inputs()
                    .iter()
                    .position(|wire| *wire == operands[0])
                    .ok_or("route input not captured")?];
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
        }
    }
    if emitted.is_empty() {
        return Err("no public refresh route reductions".into());
    }
    Ok(source)
}
