//! Actual high refresh source bounds from the linked state and routed readouts.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::{ExponentLutNoiseReport, ExponentLutNoiseSnapshot};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{LeanArtifact, ScopeProofShape},
    node::{ConstantMatrix, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
fn expr(s: &ScopeProofShape, w: WireRef, pre: bool) -> Result<String, String> {
    let mut v = s.value_expressions.clone();
    v.insert("witness".into(), if pre { "pw" } else { "vw" }.into());
    v.insert("inputs".into(), if pre { "pin" } else { "vin" }.into());
    value_expression(&format!("w_{}_{}", w.node.0, w.port.0), &v)
}
fn fact(s: &ScopeProofShape, w: WireRef, pre: bool) -> Result<String, String> {
    let i = s
        .node_constraints
        .get(&w.node)
        .and_then(|r| r.clone().last())
        .ok_or("missing actual high source primitive fact")?;
    Ok(format!("{}{}.1", if pre { "pf" } else { "vf" }, ".2".repeat(i)))
}
const CONTEXT: &str = "  let pw := preprocessingWitness runs\n  let vw := verificationWitness runs\n  let pin := preprocessingInputs runs.2.1\n  let vin := verificationInputs runs.2.2.1\n  have pf := preprocessingFacts runs\n  have vf := verificationFacts runs\n";
pub(super) fn root_refresh_high_source(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    ver: &ValidatedGraph,
    va: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
    report: &ExponentLutNoiseReport,
) -> Result<String, String> {
    if report.refresh.slots.iter().any(|slot| slot.rounding_margin <= 0.into()) {
        return Ok("import RootRefreshFinish\n-- Strict rounding report rejected; no high source theorem is emitted.\n".into())
    }
    let vr = ver.source.root_scope();
    let ps = &pa.scope_proofs[&FrozenGraphScopeId::Root];
    let vs = &va.scope_proofs[&FrozenGraphScopeId::Root];
    let pv = |w| expr(ps, w, true);
    let first_cache = pa
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-mask-route-k-rhs-0"))
        .ok_or("missing actual mask route cache")?
        .1;
    let ConcreteWireType::IndexedFamily { element, .. } =
        &pre.scopes[&FrozenGraphScopeId::Root].wire_types[&first_cache.wire]
    else {
        return Err("mask route cache not family".into())
    };
    let matrix = element.matrix_type().ok_or("route cache element not matrix")?;
    let q = &matrix.modulus;
    let n = matrix.ring_dimension;
    let columns = matrix.columns;
    let base = &snapshot.refresh.base_p;
    let width = 1usize << snapshot.layout.width_log2;
    let gadget = vr
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, node)| {
            if !matches!(
                node.kind(),
                NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { .. }, .. }
            ) {
                return None
            }
            let w = WireRef { node: NodeId(i as u64), port: Port(0) };
            let t = ver.scopes[&FrozenGraphScopeId::Root].wire_types[&w].matrix_type()?;
            (t.modulus == *q && t.ring_dimension == n && t.rows == 2 && t.columns == columns)
                .then_some(w)
        })
        .ok_or("missing actual source gadget")?;
    let layout = pa
        .backend_layouts
        .iter()
        .find(|layout| {
            layout.modulus == *q &&
                layout.ring_dimension == n &&
                layout.regular_digits * 2 == columns
        })
        .ok_or("missing source layout")?;
    let bind = "{hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution} (runs : Runs hashModel external execution)";
    let mut source = String::from(
        "import RootRefreshHighLinks\nimport RootRefreshSourceProof\nimport RootRefreshRouteSums\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshHighSource\nopen Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution ExponentGeneratedRefreshHighLinks\nopen scoped BigOperators\n\n",
    );
    source.push_str(r#"private theorem combine_sources {q n columns stateBound maskBound freshBound : Nat}
    (secret : ExactMatrix q n 1 2) (statePublic maskPublic freshPublic : ExactMatrix q n 2 columns)
    (state mask fresh payload maskNative freshNative maskOffset freshOffset : ExactMatrix q n 1 columns)
    (mu lift : Int)
    (stateClose : Approx state (secret * statePublic - mu • payload) stateBound)
    (maskClose : Approx mask (secret * maskPublic + lift • maskNative) maskBound)
    (freshClose : Approx fresh (secret * freshPublic + mu • freshNative) freshBound) :
    Approx ((state + (mask + maskOffset)) + (fresh + freshOffset))
      (secret * (statePublic + maskPublic + freshPublic) +
        (-mu • payload + lift • maskNative + maskOffset + mu • freshNative + freshOffset))
      (stateBound + maskBound + freshBound) := by
  obtain ⟨se, hs, bs⟩ := stateClose
  obtain ⟨me, hm, bm⟩ := maskClose
  obtain ⟨fe, hf, bf⟩ := freshClose
  refine ⟨se + me + fe, ?_, coeffBound_add (coeffBound_add bs bm) bf⟩
  rw [hs, hm, hf, reduceMatrix_add, reduceMatrix_add, Matrix.mul_add, Matrix.mul_add]
  simp only [neg_smul]
  abel
"#);
    source.push_str(r#"private theorem route_transfer {q n columns width digits bound : Nat}
    (actual : ExactMatrix q n 1 columns)
    (mask payload : ExactMatrix q n 1 2) (unit : payload 0 1 = 1)
    (gadget : ExactMatrix q n 2 columns)
    (publicKey : Fin columns → Fin n → Fin digits → ExactMatrix q n 2 columns)
    (action : Fin columns → Fin n → Fin digits → ExactMatrix q n columns columns)
    (scale : Int) (base : Nat) (native : Fin (2*width) → ExactPoly q n)
    (selection : Fin 1 → Fin columns → Fin n → Fin digits → Fin (2*width))
    (reconstruction : ∀ c k d, gadget * action c k d = routeTarget (-scale * (base : Int)^d.val) k c)
    (close : Approx actual (∑ c, ∑ k, ∑ d, (mask * publicKey c k d - native (selection 0 c k d) • (payload*gadget)) * action c k d) bound) :
    Approx actual (mask * (∑ c, ∑ k, ∑ d, publicKey c k d * action c k d) + scale • refreshNativeRadixMatrix base native selection) bound := by
  have ideal := route_native_ideal (q:=q) (n:=n) (columns:=columns) (width:=width) (digits:=digits) mask payload unit gadget publicKey action scale base native selection reconstruction
  rw [ideal] at close
  convert close using 1
  ext row col
  have hr : row=0 := Subsingleton.elim _ _
  subst row
  simp only [refreshNativeRadixMatrix, Matrix.add_apply, Matrix.smul_apply, zsmul_eq_mul]
"#);
    source.push_str("private theorem replace_actual {q n rows columns bound : Nat} {actual other ideal : ExactMatrix q n rows columns} (equation : actual = other) (close : Approx other ideal bound) : Approx actual ideal bound := by\n  obtain ⟨error, errorEquation, bounded⟩ := close\n  exact ⟨error, equation.trans errorEquation, bounded⟩\n\n");
    source.push_str("private theorem native_value : ExponentGeneratedReadoutProof.nativeValue = ExponentGeneratedRefreshDigits.nativeValue := rfl\n\n");
    let public_getters = route_getters(pre, pa, true)?;
    let cipher_getters = route_getters(ver, va, false)?;
    let mut offset = 0usize;
    for slot in 0..snapshot.refresh.mask_digit_counts.len() {
        let mu = q / num_bigint::BigInt::from(snapshot.refresh.plaintext_moduli[slot].clone());
        let lift = &report.refresh.slots[slot].lift;
        let pre_bound = &report.refresh.slots[slot].pre_switch_bound;
        for (role, digits) in [
            ("mask", snapshot.refresh.mask_digit_counts[slot]),
            ("fresh", snapshot.refresh.fresh_digit_count),
        ] {
            for (kind, graph, artifact, getter, producer) in [
                ("public", pre, pa, public_getters[&(role.into(), slot)], true),
                ("cipher", ver, va, cipher_getters[&(role.into(), slot)], false),
            ] {
                let (family, index) = selected(graph, getter)?;
                let ConcreteWireType::IndexedFamily { count, .. } =
                    graph.scopes[&FrozenGraphScopeId::Root].wire_types[&family]
                else {
                    return Err("selected route family type".into())
                };
                let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
                let field = if producer { "Public" } else { "Cipher" };
                let family_expression = expr(shape, family, producer)?
                    .replace("(pw)", "(preprocessingWitness runs)")
                    .replace("(vw)", "(verificationWitness runs)")
                    .replace("(pin)", "(preprocessingInputs runs.2.1)")
                    .replace("(vin)", "(verificationInputs runs.2.2.1)");
                source.push_str(&format!("private theorem {role}_{kind}_selected_{slot} {bind} :\n  {role}{field}_{slot} runs = ({family_expression}) (⟨{index}, by decide⟩ : Fin {count}) := by\n{CONTEXT}  exact route_lookup_eq (⟨{index}, by decide⟩ : Fin {count}) (by rfl) ({})\n\n",fact(shape,getter,producer)?));
            }
            let suffix = format!("-{role}-route-k-rhs-0");
            let cache = pa
                .root
                .outputs
                .iter()
                .find(|(name, _)| name.ends_with(&suffix))
                .ok_or("missing high source route cache")?
                .1;
            let ConcreteWireType::IndexedFamily { count: total, .. } =
                pre.scopes[&FrozenGraphScopeId::Root].wire_types[&cache.wire]
            else {
                return Err("route cache type".into())
            };
            let scale = if role == "mask" { lift.to_string() } else { mu.to_string() };
            let role_offset = if role == "mask" { offset } else { slot * columns * n * digits };
            let label_offset = if role == "mask" {
                offset
            } else {
                snapshot.refresh.mask_digit_counts.iter().sum::<usize>() * columns * n
            };
            let flat = format!(
                "⟨{role_offset}+(component.val*{n}+coefficient.val)*{digits}+digit.val, by clear * - component coefficient digit; omega⟩"
            );
            let label = format!(
                "⟨{label_offset}+(component.val*{n}+coefficient.val)*{digits}+digit.val, by clear * - component coefficient digit; omega⟩"
            );
            let output = cache.projection.replace("outputs", "execution.preprocessing");
            source.push_str(&format!("theorem {role}_source_{slot} {bind} :\n    Approx ({role}Cipher_{slot} runs)\n      (ExponentGeneratedReadoutProof.mask runs * {role}Public_{slot} runs +\n        ({scale} : Int) • ExponentGeneratedRefreshNormalize.{role}Native_{slot} (ExponentGeneratedRefreshSourceProof.{role}Selection_{slot} runs))\n      (ExponentGeneratedRefreshSourceProof.{role}Gain_{slot} * ExponentGeneratedReadoutProof.readoutBound ExponentGeneratedNativeProgramProof.nativeOutputBound) := by\n  classical\n{CONTEXT}  have hg : MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {} {} (ExponentGeneratedReadoutProof.gadget runs) := {}\n  have outputLink : ({output}) = ({}) := by\n    rw [preprocessingOutput runs]\n    rfl\n  have close := ExponentGeneratedRefreshSourceProof.{role}_sum_{slot} runs\n  dsimp only at close\n  rw [← outputLink, ExponentGeneratedRefreshSourceProof.{role}_gain_{slot}] at close\n  let publicKey := fun (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digits}) => ExponentGeneratedReadoutProof.readoutPublicKey runs {label}\n  let action := fun (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digits}) => ({output}) ({flat} : Fin {total})\n  have reconstruction : ∀ c k d, ExponentGeneratedReadoutProof.gadget runs * action c k d = routeTarget (-({scale} : Int) * ({base} : Int)^d.val) k c := by\n    intro component coefficient digit\n    exact (ExponentGeneratedRefreshRouteTargets.{role}_cache_{slot} pw pf _ hg component coefficient digit).2\n  change Approx _ (∑ c, ∑ k, ∑ d, (ExponentGeneratedReadoutProof.mask runs * publicKey c k d - ExponentGeneratedReadoutProof.nativeValue (ExponentGeneratedRefreshSourceProof.{role}Selection_{slot} runs 0 c k d) • (ExponentGeneratedReadoutProof.payload runs * ExponentGeneratedReadoutProof.gadget runs)) * action c k d) _ at close\n  have converted := route_transfer (q := {q}) (n := {n}) (columns := {columns}) (width := {width}) (digits := {digits}) _ (ExponentGeneratedReadoutProof.mask runs) (ExponentGeneratedReadoutProof.payload runs) (payload_unit runs) (ExponentGeneratedReadoutProof.gadget runs) publicKey action ({scale} : Int) {base} ExponentGeneratedReadoutProof.nativeValue (ExponentGeneratedRefreshSourceProof.{role}Selection_{slot} runs) reconstruction close\n  have publicSum : {role}Public_{slot} runs = ∑ c, ∑ k, ∑ d, publicKey c k d * action c k d := ({role}_public_selected_{slot} runs).trans (ExponentGeneratedRefreshRouteSums.{role}_public_sum_{slot} pw pf _ hg)\n  rw [← publicSum, native_value] at converted\n  have rawSum := ({role}_cipher_selected_{slot} runs).trans (ExponentGeneratedRefreshRouteSums.{role}_raw_sum_{slot} vw vf)\n  exact replace_actual rawSum converted\n\n",layout.base,layout.regular_digits,fact(vs,gadget,false)?,pv(cache.wire)?));
        }
        source.push_str(&format!("theorem source_{slot} {bind} : ExponentGeneratedRefreshFinish.SourceApprox_{slot} runs (ExponentGeneratedRefreshSourceProof.readoutExponent runs) := by\n  classical\n{CONTEXT}  have stateClose := ExponentGeneratedRefreshStateSource.state_source_{slot} runs\n  have maskClose := mask_source_{slot} runs\n  have freshClose := fresh_source_{slot} runs\n  have combined := combine_sources (ExponentGeneratedReadoutProof.mask runs)\n    (ExponentGeneratedRefreshStateSource.stateScaledPublic_{slot} runs) (maskPublic_{slot} runs) (freshPublic_{slot} runs)\n    (ExponentGeneratedRefreshStateSource.stateScaledC_{slot} runs) (maskCipher_{slot} runs) (freshCipher_{slot} runs)\n    (ExponentGeneratedReadoutProof.payload runs * ExponentGeneratedReadoutProof.gadget runs)\n    (ExponentGeneratedRefreshNormalize.maskNative_{slot} (ExponentGeneratedRefreshSourceProof.maskSelection_{slot} runs))\n    (ExponentGeneratedRefreshNormalize.freshNative_{slot} (ExponentGeneratedRefreshSourceProof.freshSelection_{slot} runs))\n    (ExponentGeneratedRefreshNormalize.maskPublicOffset_{slot} vw) (ExponentGeneratedRefreshNormalize.freshPublicOffset_{slot} vw)\n    ({mu} : Int) ({lift} : Int) stateClose maskClose freshClose\n  have factor := ExponentGeneratedRefreshNormalize.high_ideal_factor_{slot} vw vf (ExponentGeneratedRefreshFinish.payloadInteger runs)\n    (ExponentGeneratedRefreshSourceProof.freshSelection_{slot} runs) (ExponentGeneratedRefreshSourceProof.maskSelection_{slot} runs)\n  rw [payload_lift runs] at factor\n  rw [factor, ← high_public_{slot} runs] at combined\n  have boundEquation : (ExponentGeneratedRefreshState.stateGain_{slot} * ExponentGeneratedRefreshStateSource.initialBound +\n      ExponentGeneratedRefreshSourceProof.maskGain_{slot} * ExponentGeneratedReadoutProof.readoutBound ExponentGeneratedNativeProgramProof.nativeOutputBound) +\n      ExponentGeneratedRefreshSourceProof.freshGain_{slot} * ExponentGeneratedReadoutProof.readoutBound ExponentGeneratedNativeProgramProof.nativeOutputBound = {pre_bound} := by\n    have targetGain := ExponentGeneratedReadoutFunction.target_gain_value\n    change (∑ k, (2 * ExponentGeneratedReadoutFunction.targetCoefficients k - 1).natAbs) = _ at targetGain\n    simp only [ExponentGeneratedNativeProgramProof.nativeOutputBound, ExponentGeneratedNativeProgramProof.native_output_bound, ExponentGeneratedReadoutProof.readoutBound, targetGain]\n    decide\n  rw [boundEquation] at combined\n  change Approx (highCipher_{slot} runs) (ExponentGeneratedReadoutProof.mask runs * (highPublic_{slot} runs) + reduceMatrix {q} {n} 1 {columns} (({lift} : ErrorPoly {n}) • ExponentGeneratedRefreshFinish.lowIdeal_{slot} runs (ExponentGeneratedRefreshSourceProof.readoutExponent runs))) {pre_bound}\n  rw [high_cipher_{slot} runs]\n  have scalar : (({lift} : ErrorPoly {n}) • ExponentGeneratedRefreshFinish.lowIdeal_{slot} runs (ExponentGeneratedRefreshSourceProof.readoutExponent runs)) = ({lift} : Int) • ExponentGeneratedRefreshFinish.lowIdeal_{slot} runs (ExponentGeneratedRefreshSourceProof.readoutExponent runs) := by\n    simpa only [Int.cast_ofNat] using Int.cast_smul_eq_zsmul (ErrorPoly {n}) ({lift} : Int) (ExponentGeneratedRefreshFinish.lowIdeal_{slot} runs (ExponentGeneratedRefreshSourceProof.readoutExponent runs))\n  rw [scalar]\n  exact combined\n\n"));
        offset += columns * n * snapshot.refresh.mask_digit_counts[slot];
    }
    source.push_str("end ExponentGeneratedRefreshHighSource\n");
    Ok(source)
}

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

fn route_getters(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    producer: bool,
) -> Result<std::collections::BTreeMap<(String, usize), WireRef>, String> {
    let root = graph.source.root_scope();
    let mut result = std::collections::BTreeMap::new();
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
                result.insert((role.into(), slot), getter);
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
                result.insert((role.into(), slot), args(graph, offset_added)?[0]);
            }
        }
    }
    Ok(result)
}
