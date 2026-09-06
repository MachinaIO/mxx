//! Canonical integer actions for the actual PRE route caches.
//!
//! Snapshot numbers are candidates: the emitted `change` and arithmetic proofs
//! check them against the exported target, index, scale pack, and runtime facts.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::ExponentLutNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
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

pub(super) fn root_refresh_route_targets(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
) -> Result<String, String> {
    let root = pre.source.root_scope();
    let shape = pa.scope_proofs.get(&FrozenGraphScopeId::Root).ok_or("missing PRE proof shape")?;
    let types = &pre.scopes[&FrozenGraphScopeId::Root].wire_types;
    let args = |wire: WireRef| {
        root.node(wire.node)
            .and_then(|node| root.arguments(node))
            .ok_or_else(|| "missing route root operands".to_string())
    };
    let refresh = &snapshot.refresh;
    let slots = refresh.slot_count;
    if slots == 0 ||
        refresh.mask_digit_counts.len() != slots ||
        refresh.switch_moduli.len() != slots ||
        refresh.plaintext_moduli.len() != slots ||
        refresh.fresh_digit_count == 0 ||
        refresh.mask_digit_counts.contains(&0)
    {
        return Err("invalid route slot candidates".into());
    }
    let q = refresh.global_modulus.as_ref().ok_or("missing route full modulus")?;
    let q_integer = num_bigint::BigInt::from(q.clone());
    let record = shape.witness_record.as_ref().ok_or("missing PRE witness record")?;
    let body = shape.body_relation.as_ref().ok_or("missing PRE body")?;
    let params = &pa.root.parameter_type;
    let mut source = format!(
        "import {}\nimport Backend\nimport ExponentRefreshRouteTargets\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshRouteTargets\nopen Mxx.Primitives MxxExponentLut\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        pa.module_name, pa.root.input_type, pa.root.output_type
    );
    for role in ["mask", "fresh"] {
        let suffix = format!("-{role}-route-k-rhs-0");
        let mut outputs = pa.root.outputs.iter().filter(|(name, _)| name.ends_with(&suffix));
        let (_, endpoint) = outputs.next().ok_or_else(|| format!("missing {role} route cache"))?;
        if outputs.next().is_some() {
            return Err(format!("ambiguous {role} route cache"));
        }
        let cache = endpoint.wire;
        let ConcreteWireType::IndexedFamily { count: total, element } = &types[&cache] else {
            return Err("route cache is not a family".into());
        };
        let matrix = element.matrix_type().ok_or("route cache is not a matrix family")?;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        if n == 0 || columns == 0 || matrix.rows != columns || matrix.modulus != q_integer {
            return Err("route cache has incompatible high-ring dimensions".into());
        }
        let mut layouts = pa.backend_layouts.iter().enumerate().filter(|(_, layout)| {
            layout.modulus == matrix.modulus &&
                layout.ring_dimension == n &&
                layout.regular_digits.checked_mul(2) == Some(columns)
        });
        let (layout_index, layout) = layouts.next().ok_or("missing actual high-ring layout")?;
        if layouts.next().is_some() {
            return Err("ambiguous actual high-ring layout".into());
        }
        let base = &layout.base;
        let regular_digits = layout.regular_digits;
        let digits = if role == "mask" {
            refresh.mask_digit_counts.clone()
        } else {
            vec![refresh.fresh_digit_count; slots]
        };
        let coordinates = columns.checked_mul(n).ok_or("route coordinate overflow")?;
        let expected = digits
            .iter()
            .try_fold(0usize, |sum, d| {
                coordinates.checked_mul(*d).and_then(|count| sum.checked_add(count))
            })
            .ok_or("route count overflow")?;
        if expected != *total {
            return Err("snapshot route counts do not match actual cache".into());
        }
        let cache_args = args(cache)?;
        if cache_args.len() != 1 {
            return Err("route decomposition must have one target family".into());
        }
        let scaled = cache_args[0];
        let scaled_args = args(scaled)?;
        if scaled_args.len() != 2 {
            return Err("route target lacks actual scale zip".into());
        }
        // Lexical capture order follows operand use, not the old zip tuple order.
        // Identify the actual gathered scale family by its producer operation.
        let gathered = scaled_args
            .iter()
            .enumerate()
            .filter_map(|(index, wire)| {
                let scope = pre.source.scopes().get(&child(*wire))?;
                let node = scope.node(scope.outputs().first()?.node)?;
                matches!(node.kind(), NodeKind::FamilyGetDynamic).then_some(index)
            })
            .collect::<Vec<_>>();
        if gathered.len() != 1 {
            return Err("route target must capture exactly one gathered scale family".into());
        }
        let gather = scaled_args[gathered[0]];
        let target = scaled_args[1 - gathered[0]];
        let gather_scope = pre.source.scopes().get(&child(gather)).ok_or("missing route gather")?;
        let getter =
            gather_scope.node(gather_scope.outputs()[0].node).ok_or("missing route getter")?;
        if !matches!(getter.kind(), NodeKind::FamilyGetDynamic) {
            return Err("route scales are not gathered".into());
        }
        let getter_args = gather_scope.arguments(getter).ok_or("missing route gather operands")?;
        let gather_args = args(gather)?;
        let capture = |wire| -> Result<WireRef, String> {
            let i = gather_scope
                .inputs()
                .iter()
                .position(|input| *input == wire)
                .ok_or("route gather operand not captured")?;
            gather_args.get(i).copied().ok_or("route gather capture missing".into())
        };
        let pack = capture(getter_args[0])?;
        let indices = capture(getter_args[1])?;
        if !matches!(
            root.node(pack.node).map(|node| node.kind()),
            Some(NodeKind::FamilyPack { .. })
        ) || args(pack)?.len() != slots
        {
            return Err("route scale table is not the actual slot pack".into());
        }
        let target_scope =
            pre.source.scopes().get(&child(target)).ok_or("missing route targets")?;
        let target_shape =
            pa.scope_proofs.get(&child(target)).ok_or("missing target proof shape")?;
        if !target_shape.witnesses.is_empty() ||
            !target_scope.inputs().is_empty() ||
            target_scope.outputs().len() != 1
        {
            return Err("route target is not a pure range body".into());
        }
        let target_selectors = selectors(&value(target_shape, target_scope.outputs()[0])?)?;
        let index_shape =
            pa.scope_proofs.get(&child(indices)).ok_or("missing route index proof")?;
        if !index_shape.witnesses.is_empty() {
            return Err("route index body is not pure".into());
        }
        let cache_scope = pre.source.scopes().get(&child(cache)).ok_or("missing cache body")?;
        if cache_scope
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
            .count() !=
            1
        {
            return Err("route cache lacks actual GadgetDecompose".into());
        }
        let target_value = value(shape, target)?;
        let scaled_value = value(shape, scaled)?;
        let index_value = value(shape, indices)?;
        let gather_value = value(shape, gather)?;
        let cache_value = value(shape, cache)?;
        let target_fact = fact(shape, target)?;
        let index_fact = fact(shape, indices)?;
        let gather_fact = fact(shape, gather)?;
        let scaled_fact = fact(shape, scaled)?;
        let cache_fact = fact(shape, cache)?;
        let pack_fact = fact(shape, pack)?;
        let target_unfold = unfold(pa, target)?;
        let index_unfold = unfold(pa, indices)?;
        let target_path = ".2".repeat(target_shape.constraint_count);
        let index_path = ".2".repeat(index_shape.constraint_count);
        let output_path = ".2".repeat(shape.constraint_count);
        let output = &endpoint.projection;
        let result = output.replacen("outputs", "result", 1);
        let base_p = &refresh.base_p;
        let mut offset = 0usize;
        for (slot, digit_count) in digits.into_iter().enumerate() {
            let divisor = if role == "mask" {
                &refresh.switch_moduli[slot]
            } else {
                &refresh.plaintext_moduli[slot]
            };
            if *divisor == num_bigint::BigUint::from(0u8) ||
                q % divisor != num_bigint::BigUint::from(0u8)
            {
                return Err("route scale candidate is not an exact quotient".into());
            }
            let scale = q / divisor;
            let actual_scale = value(shape, args(pack)?[slot])?;
            let index_normalize = if role == "mask" {
                format!(
                    "    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ {total} by decide)] at he\n"
                )
            } else {
                String::new()
            };
            let action = format!("{role}Action_{slot}");
            let theorem = format!("{role}_cache_{slot}");
            let flat = format!(
                "{offset} + (component.val * {n} + coefficient.val) * {digit_count} + digit.val"
            );
            let binders = format!(
                "(component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digit_count})"
            );
            let bound = "by clear * - component coefficient digit; omega";
            let idx = format!("⟨{flat}, {bound}⟩");
            source.push_str(&format!("noncomputable def {action} {binders} : ErrorMatrix {n} {columns} {columns} :=\n  regularIntegerDigits RefreshBackend.layout{layout_index}\n    (routeTarget (-({scale} : Int) * ({base_p} : Int)^digit.val) coefficient component)\n\n"));
            source.push_str(&format!("theorem {theorem} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record})\n    (facts : {body} RefreshBackend.backend hashModel params inputs outputs witness)\n    (G : ExactMatrix {q} {n} 2 {columns})\n    (hg : MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {base} {regular_digits} G)\n    {binders} :\n    ({output}) {idx} = reduceMatrix {q} {n} {columns} {columns} ({action} component coefficient digit) ∧\n    G * ({output}) {idx} = routeTarget (-({scale} : Int) * ({base_p} : Int)^digit.val) coefficient component := by\n  classical\n  let idx : Fin {total} := {idx}\n  have targetEq : ({target_value}) idx = routeTarget (-({base_p} : Int)^digit.val) coefficient component := by\n    have h := {target_fact} idx\n    dsimp only [{target_unfold}] at h\n    have equation := h{target_path}\n    refine equation.trans ?_\n"));
            for (i, selector) in target_selectors.iter().enumerate() {
                source.push_str(&format!("    have slotEq{i} : {selector} = ({slot} : Int) := by\n      simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ {total} by decide)]\n      dsimp [idx]\n      omega\n"));
            }
            if !target_selectors.is_empty() {
                source.push_str(&format!(
                    "    simp only [{}]\n",
                    (0..target_selectors.len())
                        .map(|i| format!("slotEq{i}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            // Definitional equality checks the actual base, selected offset, digit
            // count, row choice and matrix operation order before arithmetic.
            source.push_str(&format!("    change MxxRuntime.matrixMul\n      (MxxRuntime.matrixMulScalarLeft\n        (MxxRuntime.matrixNeg (MxxRuntime.matrixMulScalarLeft\n          (MxxRuntime.liftInteger (({base_p} : Int) ^ (Int.fmod (Int.ofNat idx.val + -1 * {offset}) {digit_count}).toNat))\n          (MxxRuntime.rotationPolynomial (Int.fmod (Int.fdiv (Int.ofNat idx.val + -1 * {offset}) {digit_count}) {n}))))\n        (MxxRuntime.unitColumn 1))\n      (MxxRuntime.unitRow (Int.fdiv (Int.fdiv (Int.ofNat idx.val + -1 * {offset}) {digit_count}) {n})) = _\n    have hd : Int.fmod (Int.ofNat idx.val + -1 * {offset}) {digit_count} = (digit.val : Int) := by\n      rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]\n      dsimp [idx]\n      omega\n    have hq : Int.fdiv (Int.ofNat idx.val + -1 * {offset}) {digit_count} = (component.val * {n} + coefficient.val : Int) := by\n      rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]\n      dsimp [idx]\n      omega\n    have hc : Int.fmod (component.val * {n} + coefficient.val : Int) {n} = coefficient.val := by\n      rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]\n      omega\n    have hm : Int.fdiv (component.val * {n} + coefficient.val : Int) {n} = component.val := by\n      rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]\n      omega\n    simp only [hd, hq, hc, hm, Int.toNat_natCast]\n    exact route_target_expression _ coefficient component\n"));
            source.push_str(&format!("  have indexEq : ({index_value}) idx = ({slot} : Int) := by\n    have h := {index_fact} idx\n    dsimp only [{index_unfold}] at h\n    have he := h{index_path}\n    clear * - component coefficient digit idx he\n{index_normalize}    dsimp [idx] at he ⊢\n    norm_num only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at he\n    omega\n  have packEq := route_pack_at {pack_fact} (⟨{slot}, by decide⟩ : Fin {slots})\n  have scaleEq : ({gather_value}) idx = MxxRuntime.liftInteger ({scale}) := by\n    obtain ⟨sample, hlo, hhi, hget, hout⟩ := {gather_fact} idx\n    have hselected := route_lookup_eq (⟨{slot}, by decide⟩ : Fin {slots}) indexEq hget\n    rw [hout, hselected]\n    dsimp only\n    rw [packEq]\n    change ({actual_scale}) = MxxRuntime.liftInteger ({scale})\n    funext row column\n    norm_num [MxxIR.exactDiv, MxxRuntime.matrixPolynomial, MxxRuntime.liftInteger]\n  have scaledEq := {scaled_fact} idx\n  change ({scaled_value}) idx = MxxRuntime.matrixMulScalarLeft (({gather_value}) idx) (({target_value}) idx) at scaledEq\n  rw [scaleEq, targetEq, route_target_scale] at scaledEq\n  have outputEq := congrArg (fun result : RootOutput => {result}) facts{output_path}\n  change ({output}) = ({cache_value}) at outputEq\n  obtain ⟨decomposition, hd, hout⟩ := {cache_fact} idx\n  change MxxRuntime.gadgetDecomposeRuns RefreshBackend.backend {base} {regular_digits} (({scaled_value}) idx) decomposition at hd\n  rw [scaledEq] at hd\n  have canonical := route_cache_canonical RefreshBackend.layout{layout_index} (by rfl)\n    (by decide) (by decide) (by rfl) hg hd\n  change ({output}) idx = _ ∧ G * ({output}) idx = _\n  rw [outputEq, hout]\n  simpa only [{action}, mul_neg, neg_mul] using canonical\n\n#print axioms {theorem}\n\n",));
            offset += coordinates * digit_count;
        }
    }
    source.push_str("end ExponentGeneratedRefreshRouteTargets\n");
    Ok(source)
}
