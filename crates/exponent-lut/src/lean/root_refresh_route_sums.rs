//! Reindex actual balanced route sums without enumerating coefficients or lanes.
use super::{
    root_inputs::value_expression, root_readout_public::segmented_reduction_levels,
    root_refresh_source::route_binding,
};
use crate::bgg::noise::BggNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::NodeKind,
    types::{ConcreteWireType, WireRef},
};
use std::collections::BTreeMap;

fn child(wire: WireRef) -> FrozenGraphScopeId {
    FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    }
}

fn args(graph: &ValidatedGraph, wire: WireRef) -> Result<Vec<WireRef>, String> {
    let root = graph.source.root_scope();
    root.node(wire.node)
        .and_then(|node| root.arguments(node))
        .ok_or("missing route sum operands".into())
}

fn selected(graph: &ValidatedGraph, wire: WireRef) -> Result<(WireRef, usize), String> {
    let NodeKind::FamilyGetStatic { index } =
        graph.source.root_scope().node(wire.node).ok_or("missing route sum getter")?.kind()
    else {
        return Err("route sum is not selected".into())
    };
    let index = index
        .evaluate(&ParamEnv::default())
        .map_err(|_| "dynamic route segment")?
        .to_string()
        .parse()
        .map_err(|_| "invalid route segment")?;
    Ok((args(graph, wire)?[0], index))
}

fn sums(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    producer: bool,
) -> Result<BTreeMap<(String, usize), (WireRef, usize)>, String> {
    let root = graph.source.root_scope();
    let mut result = BTreeMap::new();
    if producer {
        for (name, boundary) in &artifact.root.outputs {
            let Some((_, slot)) = name.rsplit_once("-refresh-target-") else { continue };
            let slot = slot.parse().map_err(|_| "invalid public route slot")?;
            let (family, local) = selected(graph, boundary.wire)?;
            let scope = &graph.source.scopes()[&child(family)];
            let operands = |wire: WireRef| {
                scope
                    .node(wire.node)
                    .and_then(|node| scope.arguments(node))
                    .ok_or("missing public route group operands")
            };
            let difference = operands(scope.outputs()[0])?;
            let high = operands(difference[0])?[0];
            let capture = scope
                .inputs()
                .iter()
                .position(|wire| *wire == high)
                .ok_or("public high key not captured")?;
            let packed = args(graph, family)?[capture];
            let high_getter =
                *args(graph, packed)?.get(local).ok_or("public high key outside pack")?;
            let (sum_family, _) = selected(graph, high_getter)?;
            let sum_scope = &graph.source.scopes()[&child(sum_family)];
            let sum_args = sum_scope
                .arguments(sum_scope.node(sum_scope.outputs()[0].node).ok_or("missing public sum")?)
                .ok_or("missing public sum args")?;
            let first = sum_scope
                .arguments(sum_scope.node(sum_args[0].node).ok_or("missing public first sum")?)
                .ok_or("missing public first args")?;
            for (role, term) in [("mask", first[1]), ("fresh", sum_args[1])] {
                let capture = sum_scope
                    .inputs()
                    .iter()
                    .position(|wire| *wire == term)
                    .ok_or("public route term not captured")?;
                let packed = args(graph, sum_family)?[capture];
                let getter =
                    *args(graph, packed)?.get(slot).ok_or("public route slot outside pack")?;
                result.insert((role.into(), slot), selected(graph, getter)?);
            }
        }
    } else {
        let crt = root
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::CrtRecompose { .. }))
            .ok_or("missing refresh CRT")?;
        for (slot, level) in root.arguments(crt).ok_or("missing CRT levels")?.iter().enumerate() {
            let (family, local) = selected(graph, *level)?;
            let scope = &graph.source.scopes()[&child(family)];
            let operands = |wire: WireRef| {
                scope
                    .node(wire.node)
                    .and_then(|node| scope.arguments(node))
                    .ok_or("missing online route group operands")
            };
            let difference = operands(scope.outputs()[0])?;
            let high = operands(difference[0])?[0];
            let final_terms = operands(high)?;
            let first = operands(final_terms[0])?;
            for (role, term) in [("mask", first[1]), ("fresh", final_terms[1])] {
                let capture = scope
                    .inputs()
                    .iter()
                    .position(|wire| *wire == term)
                    .ok_or("online route term not captured")?;
                let packed = args(graph, family)?[capture];
                let offset_added =
                    *args(graph, packed)?.get(local).ok_or("online route slot outside pack")?;
                result.insert((role.into(), slot), selected(graph, args(graph, offset_added)?[0])?);
            }
        }
    }
    Ok(result)
}

pub(super) fn root_refresh_route_sums(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    ver: &ValidatedGraph,
    va: &LeanArtifact,
    snapshot: &BggNoiseSnapshot,
) -> Result<String, String> {
    let mut source = String::from(
        "import RootRefreshSource\nimport RootRefreshRoutePublic\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshRouteSums\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\n",
    );
    for (graph, artifact, producer, prefix) in [(pre, pa, true, "pre_"), (ver, va, false, "")] {
        let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
        let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
        let value = |wire: WireRef| {
            value_expression(
                &format!("w_{}_{}", wire.node.0, wire.port.0),
                &shape.value_expressions,
            )
        };
        let record = shape.witness_record.as_ref().ok_or("missing route record")?;
        let body = shape.body_relation.as_ref().ok_or("missing route body")?;
        let found = sums(graph, artifact, producer)?;
        if found.len() != 2 * snapshot.refresh.slot_count {
            return Err("incomplete actual route sums".into())
        }
        for ((role, slot), (output, segment)) in found {
            let ConcreteWireType::IndexedFamily { count: segments, .. } = &types[&output] else {
                return Err("route output not family".into())
            };
            let levels = segmented_reduction_levels(graph, output, *segments)?;
            let leaf = levels.last().map_or(output, |level| level.1);
            let ConcreteWireType::IndexedFamily { count, element } = &types[&leaf] else {
                return Err("route leaf not family".into())
            };
            let matrix = element.matrix_type().ok_or("route leaf not matrix")?;
            let n = matrix.ring_dimension;
            let columns = matrix.columns;
            let digits = if role == "mask" {
                snapshot.refresh.mask_digit_counts[slot]
            } else {
                snapshot.refresh.fresh_digit_count
            };
            let width = columns * n * digits;
            if *count != segments * width {
                return Err("route width disagrees with actual reduction".into())
            }
            let offset = if role == "mask" {
                snapshot.refresh.mask_digit_counts[..slot].iter().sum::<usize>() * columns * n
            } else {
                slot * width
            };
            let relation_slot = if role == "fresh" { 0 } else { slot };
            let binding = route_binding(graph, artifact, leaf, "index")?;
            let raw = binding.as_ref().map_or(leaf, |(family, _)| *family);
            let ConcreteWireType::IndexedFamily { count: total, .. } = &types[&raw] else {
                return Err("raw route not family".into())
            };
            let local = format!("(component.val*{n}+coefficient.val)*{digits}+digit.val");
            let flat =
                format!("⟨{offset}+{local}, by clear * - component coefficient digit; omega⟩");
            source.push_str(&format!("theorem {prefix}{role}_raw_sum_{slot} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ({}) (⟨{segment}, by decide⟩ : Fin {segments}) =\n      ∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits}, ({}) ({flat} : Fin {total}) := by\n  classical\n  calc\n    _ = ∑ i : Fin {width}, ({}) ⟨{segment}*{width}+i.val, by clear * - i; omega⟩ := ExponentGeneratedRefreshSource.{prefix}{role}_reduction_{relation_slot} witness facts ⟨{segment}, by decide⟩\n    _ = ∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits}, ({}) ⟨{segment}*{width}+({local}), by clear * - component coefficient digit; omega⟩ := route_sum_three (columns := {columns}) (n := {n}) (digits := {digits}) _\n    _ = _ := by\n      apply Finset.sum_congr rfl\n      intro component hc\n      apply Finset.sum_congr rfl\n      intro coefficient hk\n      apply Finset.sum_congr rfl\n      intro digit hd\n      let index : Fin {count} := ⟨{segment}*{width}+({local}), by clear * - component coefficient digit; omega⟩\n",artifact.root.parameter_type,artifact.root.input_type,artifact.root.output_type,value(output)?,value(raw)?,value(leaf)?,value(leaf)?));
            if let Some((_, expression)) = &binding {
                source.push_str(&format!("      obtain ⟨bound, equation⟩ := ExponentGeneratedRefreshSource.{prefix}route_gather_{} witness facts index\n      have same : (⟨({expression}).toNat, bound⟩ : Fin {total}) = {flat} := by\n        apply Fin.ext\n        dsimp only [index]\n        norm_num only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, Int.toNat_natCast]\n        clear * - component coefficient digit\n        omega\n      exact equation.trans (congrArg ({}) same)\n\n",leaf.node.0,value(raw)?));
            } else {
                source.push_str(&format!("      change ({}) index = ({}) ({flat} : Fin {total})\n      congr 1\n      apply Fin.ext\n      dsimp only [index]\n      clear * - component coefficient digit\n      omega\n\n",value(raw)?,value(raw)?));
            }
            if producer {
                let exported = if role == "mask" { "exportedMaskD" } else { "exportedFreshD" };
                let label_offset = if role == "mask" {
                    offset
                } else {
                    snapshot.refresh.mask_digit_counts.iter().sum::<usize>() * columns * n
                };
                let label = format!(
                    "⟨{label_offset}+{local}, by clear * - component coefficient digit; omega⟩"
                );
                source.push_str(&format!("theorem {role}_public_sum_{slot} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ({}) (⟨{segment}, by decide⟩ : Fin {segments}) =\n      ∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits},\n        ExponentGeneratedRefreshRoutePublic.normalizedReadoutPublic witness {label} * ExponentGeneratedRefreshRoutePublic.{exported} outputs ({flat} : Fin {total}) := by\n  classical\n  calc\n    _ = ∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits}, ({}) ({flat} : Fin {total}) := pre_{role}_raw_sum_{slot} witness facts\n    _ = _ := by\n      apply Finset.sum_congr rfl\n      intro component hc\n      apply Finset.sum_congr rfl\n      intro coefficient hk\n      apply Finset.sum_congr rfl\n      intro digit hd\n      exact ExponentGeneratedRefreshRoutePublic.{role}_public_{slot} witness facts component coefficient digit\n\n",artifact.root.parameter_type,artifact.root.input_type,artifact.root.output_type,value(output)?,value(raw)?));
            }
        }
    }
    source.push_str("end ExponentGeneratedRefreshRouteSums\n");
    Ok(source)
}
