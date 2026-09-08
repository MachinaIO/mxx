//! Actual PRE public route products, tied to the exported prepared route caches.
use super::{
    root_inputs::value_expression,
    root_program_setup::{parallel_arguments, parallel_operand},
    root_readout_public::{root_fact, segmented_reduction_levels},
    root_refresh_source::route_binding,
};
use crate::bgg::noise::BggNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    graph::GraphScope,
    lean::{LeanArtifact, ScopeProofShape},
    node::NodeKind,
    types::{ConcreteWireType, WireRef},
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
            let reduced = operand(root, selected, 0)?;
            let ConcreteWireType::IndexedFamily { count: segments, .. } =
                &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&reduced]
            else {
                return Err("route reduction is not a family".into())
            };
            let levels = segmented_reduction_levels(graph, reduced, *segments)?;
            let mut leaf = levels.last().map_or(reduced, |level| level.1);
            while let Some((family, _)) = route_binding(graph, artifact, leaf, "i")? {
                leaf = family;
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

/// Compose checked route views without imposing a number of intermediate slices.
fn operand_equation(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    mut wire: WireRef,
    endpoint: WireRef,
    actual: &str,
    expected: &str,
    name: &str,
    index_proof: &str,
) -> Result<String, String> {
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let count = |wire: WireRef| match &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&wire] {
        ConcreteWireType::IndexedFamily { count, .. } => Ok(*count),
        _ => Err("public operand is not a family".to_owned()),
    };
    let mut position = format!("(⟨idx.val, by have h := idx.isLt; omega⟩ : Fin {})", count(wire)?);
    let mut source = String::new();
    let mut views = 0;
    while wire != endpoint {
        let Some((family, index)) = route_binding(graph, artifact, wire, &position)? else {
            return Err(format!("{name} operand does not reference its actual endpoint"));
        };
        source.push_str(&format!("  obtain ⟨{name}Bound{views}, {name}Get{views}⟩ := pre_route_gather_{} witness facts {position}\n",wire.node.0));
        position = format!("(⟨({index}).toNat, {name}Bound{views}⟩ : Fin {})", count(family)?);
        wire = family;
        views += 1;
    }
    source.push_str(&format!("  have {name}Index : ({position}).val = ({expected} : Nat) := by\n{index_proof}  have {name}End : ({}) {position} = ({}) ⟨{expected}, by omega⟩ := congrArg (fun member : Fin {} => ({}) member) (by apply Fin.ext; exact {name}Index)\n",value(shape,endpoint)?,value(shape,endpoint)?,count(endpoint)?,value(shape,endpoint)?));
    let equation = (0..views)
        .rev()
        .fold(format!("{name}End"), |tail, view| format!("{name}Get{view}.trans ({tail})"));
    source.push_str(&format!(
        "  have {name}Eq : ({actual}) = ({}) ⟨{expected}, by omega⟩ := {equation}\n",
        value(shape, endpoint)?
    ));
    Ok(source)
}

pub(super) fn root_refresh_route_public(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    snapshot: &BggNoiseSnapshot,
) -> Result<String, String> {
    let root = pre.source.root_scope();
    let shape = &pa.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &pre.scopes[&FrozenGraphScopeId::Root].wire_types;
    let count = |wire: WireRef| match types.get(&wire) {
        Some(ConcreteWireType::IndexedFamily { count, .. }) => Ok(*count),
        _ => Err("public route operand not family".to_owned()),
    };
    let refresh = &snapshot.refresh;
    let slots = refresh.slot_count;
    let raws = raw_routes(pre, pa, slots)?;
    let mask_inputs = arguments(root, raws[0])?;
    let fresh_inputs = arguments(root, raws[1])?;
    let endpoint_of = |mut wire: WireRef| -> Result<WireRef, String> {
        while let Some((family, _)) = route_binding(pre, pa, wire, "i")? {
            wire = family;
        }
        Ok(wire)
    };
    let normalized = endpoint_of(mask_inputs[0])?;
    if endpoint_of(fresh_inputs[0])? != normalized {
        return Err("public routes do not share normalized readout".into());
    }
    let record = shape.witness_record.as_ref().ok_or("missing PRE witness")?;
    let body = shape.body_relation.as_ref().ok_or("missing PRE body")?;
    let params = &pa.root.parameter_type;
    let mut source = format!(
        "import {}\nimport RootRefreshSource\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshRoutePublic\nopen Mxx.Primitives MxxExponentLut\nopen ExponentGeneratedRefreshSource\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\nnoncomputable abbrev normalizedReadoutPublic (witness : {record}) := {}\n\n",
        pa.module_name,
        pa.root.input_type,
        pa.root.output_type,
        value(shape, normalized)?
    );
    for (role_index, role) in ["mask", "fresh"].into_iter().enumerate() {
        let raw = raws[role_index];
        let scope = &pre.source.scopes()[&child(raw)];
        let raw_shape = &pa.scope_proofs[&child(raw)];
        let product_wire = scope.outputs()[raw.port.0 as usize];
        let product = scope.node(product_wire.node).ok_or("missing raw public product")?;
        if !matches!(
            product.kind(),
            NodeKind::MatrixMulSmallRhs |
                NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply)
        ) {
            return Err("raw public route is not prepared-cache multiplication".into());
        }
        let operands = arguments(scope, product_wire)?;
        let public_binding = parallel_operand(pre, pa, raw, operands[0], "idx")?;
        let cache_binding = parallel_operand(pre, pa, raw, operands[1], "idx")?;
        let suffix = format!("-{role}-route-k-rhs-0");
        let mut endpoints = pa.root.outputs.iter().filter(|(name, _)| name.ends_with(&suffix));
        let endpoint = endpoints.next().ok_or("missing exported prepared route cache")?.1;
        if endpoints.next().is_some() {
            return Err("ambiguous exported route cache".into());
        }
        let mut cache_source = cache_binding.0;
        while cache_source != endpoint.wire {
            let Some((next, _)) = route_binding(pre, pa, cache_source, "i")? else {
                return Err("raw public product does not consume exported prepared cache".into())
            };
            cache_source = next;
        }
        let ConcreteWireType::IndexedFamily { element, .. } = &endpoint.wire_type else {
            return Err("exported route cache is not family".into())
        };
        let matrix = element.matrix_type().ok_or("route cache element is not matrix")?;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let coordinates = columns.checked_mul(n).ok_or("route coordinate overflow")?;
        let digits = if role_index == 0 {
            refresh.mask_digit_counts.clone()
        } else {
            vec![refresh.fresh_digit_count; slots]
        };
        let total = count(raw)?;
        let expected = digits
            .iter()
            .try_fold(0usize, |sum, digit| {
                coordinates.checked_mul(*digit).and_then(|size| sum.checked_add(size))
            })
            .ok_or("route count overflow")?;
        let mask_total = count(raws[0])?;
        let fresh_group = coordinates
            .checked_mul(refresh.fresh_digit_count)
            .ok_or("fresh route count overflow")?;
        if total != expected ||
            count(endpoint.wire)? != total ||
            count(normalized)? != mask_total + fresh_group
        {
            return Err("public route candidate counts disagree".into());
        }
        source.push_str(&format!("noncomputable abbrev raw{}Public (witness : {record}) := {}\nnoncomputable abbrev {}PublicInput (witness : {record}) := {}\nabbrev exported{}D (outputs : RootOutput) := {}\n\n",if role_index==0 {"Mask"} else {"Fresh"},value(shape,raw)?,role,value(shape,public_binding.0)?,if role_index==0 {"Mask"} else {"Fresh"},endpoint.projection));
        let mut offset = 0;
        for (slot, digits) in digits.into_iter().enumerate() {
            let flat = format!(
                "{offset} + (component.val * {n} + coefficient.val) * {digits} + digit.val"
            );
            let label = if role_index == 0 {
                flat.clone()
            } else {
                format!(
                    "{mask_total} + (component.val * {n} + coefficient.val) * {digits} + digit.val"
                )
            };
            let bound = "by clear * - component coefficient digit; omega";
            let raw_value = value(shape, raw)?;
            let normalized_value = value(shape, normalized)?;
            let output = &endpoint.projection;
            source.push_str(&format!("theorem {role}_public_{slot} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digits}) :\n    ({raw_value}) ⟨{flat}, {bound}⟩ = ({normalized_value}) ⟨{label}, {bound}⟩ * ({output}) ⟨{flat}, {bound}⟩ := by\n  let idx : Fin {total} := ⟨{flat}, {bound}⟩\n"));
            let public_index_proof = if public_binding.0 == normalized {
                "    rfl\n".into()
            } else if role_index == 1 {
                "    dsimp [idx]\n    omega\n".into()
            } else {
                format!("    have indexValue : idx.val = {flat} := rfl\n    omega\n")
            };
            source.push_str(&operand_equation(
                pre,
                pa,
                public_binding.0,
                normalized,
                &public_binding.3,
                &label,
                "public",
                &public_index_proof,
            )?);
            let cache_proof = if cache_binding.0 == endpoint.wire {
                "    rfl\n"
            } else {
                "    have indexValue := idx.isLt\n    omega\n"
            };
            source.push_str(&operand_equation(
                pre,
                pa,
                cache_binding.0,
                endpoint.wire,
                &cache_binding.3,
                "idx.val",
                "cache",
                cache_proof,
            )?);
            let (index_slot, member_inputs, member_outputs) =
                parallel_arguments(pre, pa, raw, "idx")?;
            let context = format!(
                "(backend := backend) (hashModel := hashModel) (params := params) (i_{index_slot} := idx.val) (inputs := {member_inputs}) (outputs := {member_outputs}) (witness := member)"
            );
            let execution = root_fact(pa, raw)?;
            let output_fact =
                raw_shape.output_fact.as_ref().ok_or("missing raw product output fact")?;
            let projection = if scope.outputs().len() == 1 {
                "result".to_owned()
            } else {
                format!("result{}.1", ".2".repeat(raw.port.0 as usize))
            };
            let root_output = shape.output_fact.as_ref().ok_or("missing exported output fact")?;
            let exported_projection = endpoint.projection.replace("outputs", "result");
            source.push_str(&format!("  obtain ⟨member, memberFacts⟩ := ({} backend hashModel params idx.val {member_inputs} {member_outputs}).mp (({execution}) idx)\n  have productEq := congrArg (fun result => {projection}) ({output_fact} {context} memberFacts)\n  have exportedEq : ({output}) idx = ({}) idx := congrArg (fun result => ({exported_projection}) idx) ({root_output} (backend := backend) (hashModel := hashModel) (params := params) (inputs := inputs) (outputs := outputs) (witness := witness) facts)\n  change ({raw_value}) idx = ({normalized_value}) ⟨{label}, {bound}⟩ * ({output}) idx\n  rw [exportedEq]\n  simpa only [MxxRuntime.matrixMul, publicEq, cacheEq, Fin.eta] using productEq\n\n#print axioms {role}_public_{slot}\n\n",raw_shape.runs_iff_body,value(shape,endpoint.wire)?));
            offset += coordinates * digits;
        }
    }
    source.push_str("end ExponentGeneratedRefreshRoutePublic\n");
    Ok(source)
}
