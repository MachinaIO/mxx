//! Actual PRE public route products, tied to the exported canonical route caches.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::ExponentLutNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    graph::GraphScope,
    lean::{LeanArtifact, ScopeProofShape},
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, NodeId, WireRef},
};

fn child(wire: WireRef) -> FrozenGraphScopeId {
    FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    }
}

fn value(shape: &ScopeProofShape, wire: WireRef) -> Result<String, String> {
    value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
}

fn fact(shape: &ScopeProofShape, wire: WireRef) -> Result<String, String> {
    let index = shape
        .node_constraints
        .get(&wire.node)
        .and_then(|range| range.clone().last())
        .ok_or("missing route node constraint")?;
    Ok(format!("facts{}.1", ".2".repeat(index)))
}

fn unfold(artifact: &LeanArtifact, wire: WireRef) -> Result<String, String> {
    let relation = artifact.scope_relations.get(&child(wire)).ok_or("missing route scope")?;
    let relative = relation.strip_prefix(&format!("{}.", artifact.module_name)).unwrap_or(relation);
    let prefix = format!("abbrev {relative}.constraints_");
    let mut names = vec![relation.clone()];
    for line in artifact.source.lines().filter(|line| line.starts_with(&prefix)) {
        let name = line.split_whitespace().nth(1).ok_or("invalid route constraints declaration")?;
        names.push(format!("{}.{}", artifact.module_name, name));
    }
    if names.len() == 1 {
        return Err("missing route constraint blocks".into());
    }
    Ok(names.join(", "))
}

/// Extract the actual selector term, retaining the exporter's precise syntax.
/// `change` below checks the remaining expression after this term is rewritten.
fn selectors(expression: &str) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut rest = expression;
    while let Some(index) = rest.find("MxxIR.selectInt ") {
        rest = &rest[index + "MxxIR.selectInt ".len()..];
        if !rest.starts_with('(') {
            return Err("route selector is not parenthesized".into());
        }
        let mut depth = 0usize;
        let mut end = None;
        for (index, character) in rest.char_indices() {
            match character {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(index + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.ok_or("unbalanced route selector")?;
        let selector = rest[..end].replace("i_0", "idx.val");
        if !result.contains(&selector) {
            result.push(selector);
        }
        rest = &rest[end..];
    }
    Ok(result)
}

fn arguments(scope: &GraphScope, wire: WireRef) -> Result<Vec<WireRef>, String> {
    scope
        .node(wire.node)
        .and_then(|node| scope.arguments(node))
        .ok_or("missing public route operands".into())
}

fn operand(scope: &GraphScope, wire: WireRef, index: usize) -> Result<WireRef, String> {
    arguments(scope, wire)?.get(index).copied().ok_or("missing public route operand".into())
}

fn output(scope: &GraphScope) -> Result<WireRef, String> {
    let [wire] = scope.outputs() else {
        return Err("public route scope must have one output".into());
    };
    Ok(*wire)
}

fn gather(graph: &ValidatedGraph, wire: WireRef) -> Option<(WireRef, WireRef)> {
    let scope = graph.source.scopes().get(&child(wire))?;
    if scope.outputs().len() != 1 {
        return None;
    }
    let node = scope.node(scope.outputs()[0].node)?;
    if !matches!(node.kind(), NodeKind::FamilyGetDynamic) {
        return None;
    }
    let args = scope.arguments(node)?;
    let NodeKind::ParallelLoop(spec) = graph.source.root_scope().node(wire.node)?.kind() else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    let parents = arguments(graph.source.root_scope(), wire).ok()?;
    let map = |wire, mode| {
        let i = scope.inputs().iter().position(|input| *input == wire)?;
        if spec.input_modes.get(i) != Some(&mode) {
            return None;
        }
        parents.get(i).copied()
    };
    Some((map(args[0], LoopInputMode::Broadcast)?, map(args[1], LoopInputMode::Zip)?))
}

/// Follow actual public outputs through the grouped switch and balanced sums.
fn raw_routes(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    slots: usize,
) -> Result<[WireRef; 2], String> {
    let root = graph.source.root_scope();
    let mut raw = [None, None];
    let mut seen = std::collections::BTreeSet::new();
    for (name, target) in &artifact.root.outputs {
        let Some((_, slot)) = name.rsplit_once("-refresh-target-") else { continue };
        let slot = slot.parse::<usize>().map_err(|_| "invalid public route slot")?;
        if slot >= slots || !seen.insert(slot) {
            return Err("ambiguous public route slot".into());
        }
        let NodeKind::FamilyGetStatic { index } =
            root.node(target.wire.node).ok_or("missing target")?.kind()
        else {
            return Err("public target is not a selected group".into());
        };
        let local = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic target index")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "invalid target index")?;
        let family = operand(root, target.wire, 0)?;
        let group = graph.source.scopes().get(&child(family)).ok_or("missing target group")?;
        let difference = arguments(group, output(group)?)?;
        let high = operand(group, *difference.first().ok_or("missing public difference")?, 0)?;
        let capture = group
            .inputs()
            .iter()
            .position(|input| *input == high)
            .ok_or("uncaptured public source")?;
        let packed = operand(root, family, capture)?;
        if !matches!(root.node(packed.node).map(|n| n.kind()), Some(NodeKind::FamilyPack { .. })) {
            return Err("public group not packed".into());
        }
        let high_get = *arguments(root, packed)?.get(local).ok_or("missing public slot")?;
        let sums = operand(root, high_get, 0)?;
        let scope = graph.source.scopes().get(&child(sums)).ok_or("missing public sums")?;
        let sum_args = arguments(scope, output(scope)?)?;
        let first = arguments(scope, *sum_args.first().ok_or("missing public sum")?)?;
        for (role, term) in
            [*first.get(1).ok_or("missing mask sum")?, *sum_args.get(1).ok_or("missing fresh sum")?]
                .into_iter()
                .enumerate()
        {
            let capture = scope
                .inputs()
                .iter()
                .position(|input| *input == term)
                .ok_or("uncaptured route sum")?;
            let pack = operand(root, sums, capture)?;
            if !matches!(root.node(pack.node).map(|n| n.kind()), Some(NodeKind::FamilyPack { .. }))
            {
                return Err("route sums not packed".into());
            }
            let selected = *arguments(root, pack)?.get(slot).ok_or("missing route sum slot")?;
            if !matches!(
                root.node(selected.node).map(|n| n.kind()),
                Some(NodeKind::FamilyGetStatic { .. })
            ) {
                return Err("route sum not selected".into());
            }
            let mut leaf = operand(root, selected, 0)?;
            loop {
                let Some(scope) = graph.source.scopes().get(&child(leaf)) else {
                    break;
                };
                if !matches!(
                    scope.node(output(scope)?.node).map(|node| node.kind()),
                    Some(NodeKind::MatrixBinary(MatrixBinaryOp::Add) | NodeKind::Select { .. })
                ) {
                    break;
                }
                let parents = arguments(root, leaf)?;
                if parents.len() != 2 {
                    break;
                }
                let (Some((left, _)), Some((right, _))) =
                    (gather(graph, parents[0]), gather(graph, parents[1]))
                else {
                    break;
                };
                if left != right {
                    break;
                }
                leaf = left;
            }
            if role == 0 {
                leaf = gather(graph, leaf)
                    .ok_or_else(|| {
                        format!(
                            "mask route leaf {} not gathered: {:?}",
                            leaf.node.0,
                            root.node(leaf.node).map(|node| node.kind())
                        )
                    })?
                    .0;
            }
            if raw[role].is_some_and(|previous| previous != leaf) {
                return Err("public slots use different raw routes".into());
            }
            raw[role] = Some(leaf);
        }
    }
    if seen.len() != slots {
        return Err("incomplete public route slots".into());
    }
    Ok([raw[0].ok_or("missing mask route")?, raw[1].ok_or("missing fresh route")?])
}

pub(super) fn root_refresh_route_public(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
) -> Result<String, String> {
    let root = pre.source.root_scope();
    let shape = pa.scope_proofs.get(&FrozenGraphScopeId::Root).ok_or("missing PRE proof shape")?;
    let types = &pre.scopes[&FrozenGraphScopeId::Root].wire_types;
    let count = |wire| -> Result<usize, String> {
        match types.get(&wire) {
            Some(ConcreteWireType::IndexedFamily { count, .. }) => Ok(*count),
            _ => Err("public route operand not family".into()),
        }
    };
    let refresh = &snapshot.refresh;
    let slots = refresh.slot_count;
    if slots == 0 ||
        refresh.mask_digit_counts.len() != slots ||
        refresh.switch_moduli.len() != slots ||
        refresh.plaintext_moduli.len() != slots ||
        refresh.mask_digit_counts.contains(&0) ||
        refresh.fresh_digit_count == 0
    {
        return Err("invalid public route slot candidates".into());
    }
    let q = refresh.global_modulus.as_ref().ok_or("missing public route modulus")?;
    let raws = raw_routes(pre, pa, slots)?;
    let mask_inputs = arguments(root, raws[0])?;
    let fresh_inputs = arguments(root, raws[1])?;
    if mask_inputs.len() != 2 || fresh_inputs.len() != 2 {
        return Err("public route loop must zip public keys and scales".into());
    }
    let normalized = gather(pre, mask_inputs[0]).ok_or("mask public input not gathered")?.0;
    let fresh_slice = gather(pre, fresh_inputs[0]).ok_or("fresh public input not repeated")?.0;
    if gather(pre, fresh_slice).ok_or("fresh public slice not gathered")?.0 != normalized {
        return Err("public routes do not share normalized readout".into());
    }
    let record = shape.witness_record.as_ref().ok_or("missing PRE witness")?;
    let body = shape.body_relation.as_ref().ok_or("missing PRE body")?;
    let params = &pa.root.parameter_type;
    let mut source = format!(
        "import {}\nimport Backend\nimport RootRefreshRouteTargets\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshRoutePublic\nopen Mxx.Primitives MxxExponentLut\nopen ExponentGeneratedRefreshRouteTargets\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\nnoncomputable abbrev normalizedReadoutPublic (witness : {record}) := {}\n\n",
        pa.module_name,
        pa.root.input_type,
        pa.root.output_type,
        value(shape, normalized)?
    );
    for (role_index, role) in ["mask", "fresh"].into_iter().enumerate() {
        let raw = raws[role_index];
        let inputs = arguments(root, raw)?;
        let total = count(raw)?;
        let suffix = format!("-{role}-route-k-rhs-0");
        let mut endpoints = pa.root.outputs.iter().filter(|(name, _)| name.ends_with(&suffix));
        let endpoint = endpoints.next().ok_or("missing exported public route D")?.1;
        if endpoints.next().is_some() || count(endpoint.wire)? != total {
            return Err("ambiguous or mismatched exported route D".into());
        }
        let ConcreteWireType::IndexedFamily { element, .. } = &types[&endpoint.wire] else {
            unreachable!()
        };
        let matrix = element.matrix_type().ok_or("route D is not matrix")?;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        if n == 0 ||
            columns == 0 ||
            matrix.rows != columns ||
            matrix.modulus != num_bigint::BigInt::from(q.clone())
        {
            return Err("invalid route D dimensions".into());
        }
        let mut layouts = pa.backend_layouts.iter().enumerate().filter(|(_, l)| {
            l.modulus == matrix.modulus &&
                l.ring_dimension == n &&
                l.regular_digits.checked_mul(2) == Some(columns)
        });
        let (layout_index, layout) = layouts.next().ok_or("missing route backend layout")?;
        if layouts.next().is_some() {
            return Err("ambiguous route backend layout".into());
        }
        let base = &layout.base;
        let regular_digits = layout.regular_digits;
        let coordinates = columns.checked_mul(n).ok_or("route coordinate overflow")?;
        let digits = if role_index == 0 {
            refresh.mask_digit_counts.clone()
        } else {
            vec![refresh.fresh_digit_count; slots]
        };
        let expected = digits
            .iter()
            .try_fold(0usize, |s, d| coordinates.checked_mul(*d).and_then(|v| s.checked_add(v)))
            .ok_or("route count overflow")?;
        let mask_total = count(raws[0])?;
        let fresh_group =
            coordinates.checked_mul(refresh.fresh_digit_count).ok_or("fresh group overflow")?;
        if expected != total ||
            count(normalized)? !=
                mask_total.checked_add(fresh_group).ok_or("readout count overflow")? ||
            count(fresh_slice)? != fresh_group
        {
            return Err("public route candidate count mismatch".into());
        }
        let NodeKind::ParallelLoop(spec) = root.node(raw.node).ok_or("missing raw route")?.kind()
        else {
            return Err("raw route is not a loop".into());
        };
        if spec.input_modes != [LoopInputMode::Zip, LoopInputMode::Zip] {
            return Err("unsupported raw public input modes".into());
        }
        let scope = pre.source.scopes().get(&child(raw)).ok_or("missing raw public loop")?;
        let raw_shape = pa.scope_proofs.get(&child(raw)).ok_or("missing raw public proof shape")?;
        let decompositions = scope
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
            .collect::<Vec<_>>();
        if decompositions.len() != 1 ||
            raw_shape.witnesses.len() != 1 ||
            scope.inputs().len() != 2 ||
            scope.outputs().len() != 1
        {
            return Err(format!(
                "unsupported raw public loop telescope at {}: decompositions {}, witnesses {}, constraints {}, inputs {}, outputs {}",
                raw.node.0,
                decompositions.len(),
                raw_shape.witnesses.len(),
                raw_shape.constraint_count,
                scope.inputs().len(),
                scope.outputs().len()
            ));
        }
        let target =
            scope.arguments(decompositions[0].1).ok_or("missing internal decomposition target")?[0];
        let target_selectors = selectors(&value(raw_shape, target)?)?;
        let scalar_role = format!("{role}Public");
        let (scalar_proofs, digit_table) = super::root_refresh_route_targets::route_scalar_proofs(
            pre,
            pa,
            snapshot,
            inputs[1],
            n,
            columns,
            &digits,
            role_index == 1,
            &scalar_role,
        )?;
        source.push_str(&scalar_proofs);
        source.push_str(&format!("noncomputable abbrev raw{}Public (witness : {record}) := {}\nnoncomputable abbrev {}PublicInput (witness : {record}) := {}\nabbrev exported{}D (outputs : RootOutput) := {}\n\n", if role_index == 0 { "Mask" } else { "Fresh" }, value(shape, raw)?, role, value(shape, inputs[0])?, if role_index == 0 { "Mask" } else { "Fresh" }, endpoint.projection));
        let mut offset = 0usize;
        for (slot, digit_count) in digits.into_iter().enumerate() {
            let divisor = if role_index == 0 {
                &refresh.switch_moduli[slot]
            } else {
                &refresh.plaintext_moduli[slot]
            };
            if *divisor == num_bigint::BigUint::from(0u8) ||
                q % divisor != num_bigint::BigUint::from(0u8)
            {
                return Err("public route scale not exact quotient".into());
            }
            let target_base = if digit_table { "1".into() } else { refresh.base_p.to_string() };
            let flat = format!(
                "{offset} + (component.val * {n} + coefficient.val) * {digit_count} + digit.val"
            );
            let local =
                format!("(component.val * {n} + coefficient.val) * {digit_count} + digit.val");
            let label =
                if role_index == 0 { flat.clone() } else { format!("{mask_total} + {local}") };
            let bound = "by clear * - component coefficient digit; omega";
            let idx = format!("⟨{flat}, {bound}⟩");
            let raw_value = value(shape, raw)?;
            let public_value = value(shape, inputs[0])?;
            let scale_value = value(shape, inputs[1])?;
            let normalized_value = value(shape, normalized)?;
            let output = &endpoint.projection;
            source.push_str(&format!("theorem {role}_public_{slot} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record})\n    (facts : {body} RefreshBackend.backend hashModel params inputs outputs witness)\n    (G : ExactMatrix {q} {n} 2 {columns})\n    (hg : MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {base} {regular_digits} G)\n    (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digit_count}) :\n    ({raw_value}) {idx} =\n      ({normalized_value}) ⟨{label}, {bound}⟩ * ({output}) {idx} := by\n  classical\n  let idx : Fin {total} := {idx}\n"));
            // Each gather index is proved against the actual pure scalar body.
            let stages = if role_index == 0 {
                vec![(inputs[0], "idx".to_string(), flat.clone())]
            } else {
                vec![
                    (inputs[0], "idx".to_string(), local.clone()),
                    (
                        fresh_slice,
                        format!("(⟨{local}, {bound}⟩ : Fin {fresh_group})"),
                        label.clone(),
                    ),
                ]
            };
            for (stage, (wire, position, selected)) in stages.iter().enumerate() {
                let (family, indices) = gather(pre, *wire).ok_or("missing public gather")?;
                let index_shape =
                    pa.scope_proofs.get(&child(indices)).ok_or("missing gather index shape")?;
                if !index_shape.witnesses.is_empty() {
                    return Err("public gather index not pure".into());
                }
                let index_value = value(shape, indices)?;
                let family_count = count(family)?;
                let normalize = if role_index == 1 && stage == 0 {
                    format!(
                        "    simp only [Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ {fresh_group} by decide)] at he\n"
                    )
                } else {
                    String::new()
                };
                source.push_str(&format!("  have publicIndex{stage} : ({index_value}) {position} = (({selected} : Nat) : Int) := by\n    have h := {} {position}\n    dsimp only [{}] at h\n    have he := h{}\n    clear * - component coefficient digit idx he\n{normalize}    dsimp [idx] at he ⊢\n    norm_num only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at he ⊢\n    omega\n  have publicGet{stage} : ({}) {position} = ({}) (⟨{selected}, {bound}⟩ : Fin {family_count}) := by\n    obtain ⟨sample, hlo, hhi, hget, hout⟩ := {} {position}\n    exact hout.trans (route_lookup_eq (⟨{selected}, {bound}⟩ : Fin {family_count}) publicIndex{stage} hget)\n", fact(shape, indices)?, unfold(pa, indices)?, ".2".repeat(index_shape.constraint_count), value(shape, *wire)?, value(shape, family)?, fact(shape, *wire)?));
            }
            source.push_str(&format!("  have publicEq : ({public_value}) idx = ({normalized_value}) ⟨{label}, {bound}⟩ := {}\n", if role_index == 0 { "publicGet0" } else { "publicGet0.trans publicGet1" }));
            source.push_str(&format!("  have scaleEq := {scalar_role}_route_scale_{slot} witness facts component coefficient digit\n"));
            let decomposition_fact = raw_shape
                .node_constraints
                .get(&NodeId(decompositions[0].0 as u64))
                .and_then(|range| range.clone().last())
                .ok_or("missing internal decomposition fact")?;
            source.push_str(&format!("  obtain ⟨decomposition, rawFacts⟩ := {} idx\n  have hd := rawFacts{}.1\n  have hout := rawFacts{}\n", fact(shape, raw)?, ".2".repeat(decomposition_fact), ".2".repeat(raw_shape.constraint_count)));
            for (i, selector) in target_selectors.iter().enumerate() {
                source.push_str(&format!("  have slotEq{i} : {selector} = ({slot} : Int) := by\n    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ {total} by decide)]\n    dsimp [idx]\n    omega\n"));
            }
            if !target_selectors.is_empty() {
                source.push_str(&format!(
                    "  simp only [{}] at hd\n",
                    (0..target_selectors.len())
                        .map(|i| format!("slotEq{i}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            source.push_str(&format!("  change MxxRuntime.gadgetDecomposeRuns RefreshBackend.backend {base} {regular_digits}\n    (MxxRuntime.matrixMulScalarLeft (({scale_value}) idx)\n      (MxxRuntime.matrixMul\n        (MxxRuntime.matrixMulScalarLeft\n          (MxxRuntime.matrixNeg (MxxRuntime.matrixMulScalarLeft\n            (MxxRuntime.liftInteger (({target_base} : Int) ^ (Int.fmod (Int.ofNat idx.val + -1 * {offset}) {digit_count}).toNat))\n            (MxxRuntime.rotationPolynomial (Int.fmod (Int.fdiv (Int.ofNat idx.val + -1 * {offset}) {digit_count}) {n}))))\n          (MxxRuntime.unitColumn 1))\n        (MxxRuntime.unitRow (Int.fdiv (Int.fdiv (Int.ofNat idx.val + -1 * {offset}) {digit_count}) {n})))) decomposition at hd\n  have hdigit : Int.fmod (Int.ofNat idx.val + -1 * {offset}) {digit_count} = (digit.val : Int) := by\n    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]\n    dsimp [idx]\n    omega\n  have hquot : Int.fdiv (Int.ofNat idx.val + -1 * {offset}) {digit_count} = (component.val * {n} + coefficient.val : Int) := by\n    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]\n    dsimp [idx]\n    omega\n  have hcoeff : Int.fmod (component.val * {n} + coefficient.val : Int) {n} = coefficient.val := by\n    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]\n    omega\n  have hcomponent : Int.fdiv (component.val * {n} + coefficient.val : Int) {n} = component.val := by\n    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]\n    omega\n  simp only [hdigit, hquot, hcoeff, hcomponent, Int.toNat_natCast] at hd\n  rw [scaleEq, route_target_expression _ coefficient component, route_target_scale] at hd\n  have canonical := (route_cache_canonical RefreshBackend.layout{layout_index} (by rfl)\n    (by decide) (by decide) (by rfl) hg hd).1\n  have exported := ({role}_cache_{slot} witness facts G hg component coefficient digit).1\n  have sameD : decomposition = ({output}) idx := by\n    apply canonical.trans\n    simpa only [idx, {role}Action_{slot}, mul_neg, neg_mul, one_pow, mul_one] using exported.symm\n  change ({raw_value}) idx = _\n  change ({raw_value}) idx = ({public_value}) idx * decomposition at hout\n  rw [hout, sameD, publicEq]\n\n#print axioms {role}_public_{slot}\n\n"));
            offset += coordinates * digit_count;
        }
    }
    source.push_str("end ExponentGeneratedRefreshRoutePublic\n");
    Ok(source)
}
