//! Actual payload and high refresh recurrences from one linked execution.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{LeanArtifact, ScopeProofShape},
    node::{ConcatAxis, ConstantMatrix, LoopInputMode, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
fn child(w: WireRef) -> FrozenGraphScopeId {
    FrozenGraphScopeId::ParallelBody { parent: Box::new(FrozenGraphScopeId::Root), owner: w.node }
}
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
        .ok_or("missing finish primitive fact")?;
    Ok(format!("{}{}.1", if pre { "pf" } else { "vf" }, ".2".repeat(i)))
}
const CONTEXT: &str = "  let pw := preprocessingWitness runs\n  let vw := verificationWitness runs\n  let pin := preprocessingInputs runs.2.1\n  let vin := verificationInputs runs.2.2.1\n  have pf := preprocessingFacts runs\n  have vf := verificationFacts runs\n";
const GETTERS: &str = "  let pw := preprocessingWitness runs\n  let vw := verificationWitness runs\n  let pin := preprocessingInputs runs.2.1\n  let vin := verificationInputs runs.2.2.1\n";
pub(super) fn root_refresh_high_links(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    ver: &ValidatedGraph,
    va: &LeanArtifact,
) -> Result<String, String> {
    let pr = pre.source.root_scope();
    let vr = ver.source.root_scope();
    let ps = &pa.scope_proofs[&FrozenGraphScopeId::Root];
    let vs = &va.scope_proofs[&FrozenGraphScopeId::Root];
    let pv = |w| expr(ps, w, true);
    let vv = |w| expr(vs, w, false);
    let pargs = |w: WireRef| {
        pr.node(w.node).and_then(|node| pr.arguments(node)).ok_or("missing high source PRE operand")
    };
    let vargs = |w: WireRef| {
        vr.node(w.node).and_then(|node| vr.arguments(node)).ok_or("missing high source VER operand")
    };
    let crt = vr
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| matches!(node.kind(), NodeKind::CrtRecompose { .. }))
        .ok_or("missing high source CRT")?;
    let crt_wire = WireRef { node: NodeId(crt.0 as u64), port: Port(0) };
    let levels = vargs(crt_wire)?;
    let endpoint = va
        .root
        .outputs
        .values()
        .find(|o| vargs(o.wire).is_ok_and(|args| args.first() == Some(&crt_wire)))
        .ok_or("missing high source endpoint")?;
    let anchor = vargs(endpoint.wire)?[1];
    let payload_term = vargs(anchor)?[1];
    let product = vargs(payload_term)?[1];
    let product_args = vargs(product)?;
    let v_payload = product_args[0];
    let v_gadget = product_args[1];
    let payload_name = va
        .root
        .inputs
        .iter()
        .find(|(_, b)| b.wire == v_payload)
        .map(|(name, _)| name)
        .ok_or("payload is not linked PRE boundary")?;
    let payload = pa.root.outputs.get(payload_name).ok_or("missing actual PRE payload")?.wire;
    if !matches!(
        pr.node(payload.node).map(|node| node.kind()),
        Some(NodeKind::Concat { axis: ConcatAxis::Columns, .. })
    ) {
        return Err("payload is not augmented by actual column concat".into())
    }
    let matrix = ver.scopes[&FrozenGraphScopeId::Root].wire_types[&anchor]
        .matrix_type()
        .ok_or("anchor type")?;
    let q = &matrix.modulus;
    let n = matrix.ring_dimension;
    let columns = matrix.columns;
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

    let bind = "{hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution} (runs : Runs hashModel external execution)";
    let mut source = String::from(
        "import RootRefreshFinish\nimport RootRefreshStateSource\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshHighLinks\nopen Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution\nopen scoped BigOperators\n\n",
    );
    source.push_str(&format!("theorem payload_unit {bind} : ExponentGeneratedReadoutProof.payload runs 0 1 = 1 := by\n{CONTEXT}  exact augmented_source_last ({}) (by rfl)\n\n",fact(ps,payload,true)?));
    source.push_str(&format!("theorem payload_lift {bind} : reduceMatrix {q} {n} 1 {columns} (ExponentGeneratedRefreshFinish.payloadInteger runs) = ExponentGeneratedReadoutProof.payload runs * ExponentGeneratedReadoutProof.gadget runs := by\n{CONTEXT}  have payloadLink : ({}) = ExponentGeneratedReadoutProof.payload runs := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n  have hg : ({}) = ExponentGeneratedReadoutProof.gadget runs := gadget_runs_deterministic ({}) ({})\n  have reduced := refresh_anchor_lift (by decide : 1 < {q}) (by decide : 0 < {n}) ({})\n  change reduceMatrix {q} {n} 1 {columns} (ExponentGeneratedRefreshFinish.payloadInteger runs) = _ at reduced\n  rw [reduced]\n  change MxxRuntime.matrixMulScalarLeft (MxxRuntime.matrixPolynomial [1]) (({}) * ({})) = _\n  rw [payloadLink, hg]\n  ext row col\n  simp [MxxRuntime.matrixMulScalarLeft, MxxRuntime.matrixPolynomial]\n\n",vv(v_payload)?,vv(v_gadget)?,fact(vs,v_gadget,false)?,fact(vs,gadget,false)?,vv(payload_term)?,vv(v_payload)?,vv(v_gadget)?));
    for (slot, level) in levels.iter().enumerate() {
        let NodeKind::FamilyGetStatic { index } =
            vr.node(level.node).ok_or("missing source level")?.kind()
        else {
            return Err("source level is not selected".into())
        };
        let local = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic source level")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "bad source level")?;
        let online = vargs(*level)?[0];
        let os = &ver.source.scopes()[&child(online)];
        let oa = |w: WireRef| {
            os.node(w.node)
                .and_then(|node| os.arguments(node))
                .ok_or("missing online source operands")
        };
        let high = oa(oa(os.outputs()[0])?[0])?[0];
        let high_terms = oa(high)?;
        let first = oa(high_terms[0])?;
        let online_parents = vargs(online)?;
        let mut online_actual = Vec::new();
        let mut online_links = String::new();
        for (part, term) in [first[0], first[1], high_terms[1]].into_iter().enumerate() {
            let capture = os
                .inputs()
                .iter()
                .position(|w| *w == term)
                .ok_or("online high source term not captured")?;
            let pack = online_parents[capture];
            let actual = emit_pack(
                &mut online_links,
                ver,
                va,
                pack,
                local,
                false,
                &format!("onlinePack{part}"),
                slot,
            )?;
            online_actual.push(actual);
        }
        let mask_terms = vargs(online_actual[1])?;
        let fresh_terms = vargs(online_actual[2])?;
        let target = pa
            .root
            .outputs
            .iter()
            .find(|(name, _)| name.ends_with(&format!("-refresh-target-{slot}")))
            .ok_or("missing high target boundary")?
            .1
            .wire;
        let NodeKind::FamilyGetStatic { index } =
            pr.node(target.node).ok_or("missing high target")?.kind()
        else {
            return Err("high target not selected".into())
        };
        let target_local = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic target local")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "bad target local")?;
        let target_family = pargs(target)?[0];
        let ts = &pre.source.scopes()[&child(target_family)];
        let ta = |w: WireRef| {
            ts.node(w.node).and_then(|node| ts.arguments(node)).ok_or("missing target operands")
        };
        let target_high = ta(ta(ts.outputs()[0])?[0])?[0];
        let capture =
            ts.inputs().iter().position(|w| *w == target_high).ok_or("target high not captured")?;
        let high_pack = pargs(target_family)?[capture];
        let mut public_links = String::new();
        let high_getter =
            emit_pack(&mut public_links, pre, pa, high_pack, target_local, true, "highPack", slot)?;
        let NodeKind::FamilyGetStatic { index } =
            pr.node(high_getter.node).ok_or("missing high key getter")?.kind()
        else {
            return Err("high key not selected".into())
        };
        let public_local = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic high index")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "bad high index")?;
        let public_family = pargs(high_getter)?[0];
        let hs = &pre.source.scopes()[&child(public_family)];
        let ha = |w: WireRef| {
            hs.node(w.node)
                .and_then(|node| hs.arguments(node))
                .ok_or("missing public addition operands")
        };
        let final_terms = ha(hs.outputs()[0])?;
        let public_first = ha(final_terms[0])?;
        let public_parents = pargs(public_family)?;
        let mut public_actual = Vec::new();
        let mut public_captures = Vec::new();
        for (part, term) in
            [public_first[0], public_first[1], final_terms[1]].into_iter().enumerate()
        {
            let capture = hs
                .inputs()
                .iter()
                .position(|w| *w == term)
                .ok_or("public high term not captured")?;
            public_captures.push(public_parents[capture]);
            public_actual.push(emit_pack(
                &mut public_links,
                pre,
                pa,
                public_parents[capture],
                public_local,
                true,
                &format!("publicPack{part}"),
                slot,
            )?);
        }
        for (name, value) in [
            (format!("maskCipher_{slot}"), vv(mask_terms[0])?),
            (format!("freshCipher_{slot}"), vv(fresh_terms[0])?),
            (format!("maskPublic_{slot}"), pv(public_actual[1])?),
            (format!("freshPublic_{slot}"), pv(public_actual[2])?),
        ] {
            let getters = GETTERS
                .lines()
                .filter(|line| {
                    value.contains(&format!("({})", line.split_whitespace().nth(1).unwrap_or("")))
                })
                .map(|line| format!("{line}\n"))
                .collect::<String>();
            source.push_str(&format!("noncomputable def {name} {bind} :=\n{getters}  {value}\n\n"));
        }
        let NodeKind::ParallelLoop(spec) =
            vr.node(online.node).ok_or("missing online group")?.kind()
        else {
            return Err("online group shape".into())
        };
        let ConcreteWireType::IndexedFamily { count: online_count, .. } =
            ver.scopes[&FrozenGraphScopeId::Root].wire_types[&online]
        else {
            return Err("online group type".into())
        };
        let mut values = va.scope_proofs[&child(online)].value_expressions.clone();
        for (i, w) in os.inputs().iter().enumerate() {
            let value = vv(online_parents[i])?;
            values.insert(
                format!("w_{}_{}", w.node.0, w.port.0),
                match spec.input_modes[i] {
                    LoopInputMode::Broadcast => value,
                    LoopInputMode::Zip => {
                        format!("({value}) (⟨{local}, by decide⟩ : Fin {online_count})")
                    }
                    _ => return Err("shifted source group input".into()),
                },
            );
        }
        let high_cipher = value_expression(&format!("w_{}_{}", high.node.0, high.port.0), &values)?;
        let high_count = pargs(high_pack)?.len();
        let high_public =
            format!("({}) (⟨{target_local}, by decide⟩ : Fin {high_count})", pv(high_pack)?);
        let public_relation = &pa.scope_relations[&child(public_family)];
        let ConcreteWireType::IndexedFamily { count: public_count, .. } =
            pre.scopes[&FrozenGraphScopeId::Root].wire_types[&public_family]
        else {
            return Err("public group type".into())
        };
        let selected_public = public_captures
            .iter()
            .map(|wire| {
                pv(*wire).map(|value| {
                    format!("({value}) (⟨{public_local}, by decide⟩ : Fin {public_count})")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let public_equation = format!(
            "({}) (⟨{public_local}, by decide⟩ : Fin {public_count}) = ({}) + ({}) + ({})",
            pv(public_family)?,
            selected_public[0],
            selected_public[1],
            selected_public[2]
        );
        for (name, value) in [
            (format!("highPublic_{slot}"), high_public.clone()),
            (format!("highCipher_{slot}"), high_cipher.clone()),
        ] {
            source.push_str(&format!("noncomputable def {name} {bind} :=\n{GETTERS}  {value}\n\n"));
        }
        source.push_str(&format!("theorem high_public_{slot} {bind} : highPublic_{slot} runs = ExponentGeneratedRefreshStateSource.stateScaledPublic_{slot} runs + maskPublic_{slot} runs + freshPublic_{slot} runs := by\n{CONTEXT}{public_links}  have publicEquation := ({}) (⟨{public_local}, by decide⟩ : Fin {public_count})\n  dsimp only [{public_relation}, {public_relation}.constraints_0] at publicEquation\n  change {public_equation} at publicEquation\n  rw [publicPack0, publicPack1, publicPack2] at publicEquation\n  have selected : ({}) = ({}) (⟨{public_local}, by decide⟩ : Fin {public_count}) := by\n    obtain ⟨position, indexEquation, equation⟩ := ({})\n    have same : position = ⟨{public_local}, by decide⟩ := by\n      apply Fin.ext\n      change position.val = {public_local}\n      change (position.val : Int) = {public_local} at indexEquation\n      omega\n    simpa only [same] using equation\n  exact highPack.trans (selected.trans publicEquation)\n\n",fact(ps,public_family,true)?,pv(high_getter)?,pv(public_family)?,fact(ps,high_getter,true)?));
        source.push_str(&format!("theorem high_cipher_{slot} {bind} : highCipher_{slot} runs = (ExponentGeneratedRefreshStateSource.stateScaledC_{slot} runs + (maskCipher_{slot} runs + ExponentGeneratedRefreshNormalize.maskPublicOffset_{slot} (verificationWitness runs))) + (freshCipher_{slot} runs + ExponentGeneratedRefreshNormalize.freshPublicOffset_{slot} (verificationWitness runs)) := by\n{CONTEXT}{online_links}  change ({high_cipher}) = _\n  rw [onlinePack0, onlinePack1, onlinePack2]\n  rfl\n\n"));
    }
    source.push_str("end ExponentGeneratedRefreshHighLinks\n");
    Ok(source)
}

fn emit_pack(
    source: &mut String,
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    pack: WireRef,
    index: usize,
    pre: bool,
    name: &str,
    slot: usize,
) -> Result<WireRef, String> {
    let root = graph.source.root_scope();
    let node = root.node(pack.node).ok_or("missing actual source pack")?;
    if !matches!(node.kind(), NodeKind::FamilyPack { .. }) &&
        (name == "publicPack0" || name == "onlinePack0")
    {
        let ConcreteWireType::IndexedFamily { count, .. } =
            graph.scopes[&FrozenGraphScopeId::Root].wire_types[&pack]
        else {
            return Err("direct state capture is not family".into())
        };
        let field = if pre { "stateScaledPublic" } else { "stateScaledC" };
        let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
        source.push_str(&format!("  have {name} : ({}) (⟨{index}, by decide⟩ : Fin {count}) = ExponentGeneratedRefreshStateSource.{field}_{slot} runs := by rfl\n",expr(shape,pack,pre)?));
        return Ok(pack)
    }
    if !matches!(node.kind(), NodeKind::FamilyPack { .. }) {
        return Err(format!(
            "high source {name} capture node {} is not FamilyPack: {:?}",
            pack.node.0,
            node.kind()
        ))
    }
    let values = root.arguments(node).ok_or("missing pack values")?;
    let value = *values.get(index).ok_or("high source pack index out of range")?;
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let count = values.len();
    let successor = if index == 0 { "" } else { ", List.getElem_cons_succ" };
    source.push_str(&format!("  have {name} : ({}) (⟨{index}, by decide⟩ : Fin {count}) = ({}) := by\n    obtain ⟨position, indexEquation, equation⟩ := ({}).2.2 (⟨{index}, by decide⟩ : Fin {count})\n    have same : position = ⟨{index}, by change {index} < {count}; decide⟩ := Fin.ext indexEquation\n    simpa only [same, List.get_eq_getElem, Fin.val_mk, List.getElem_cons_zero{successor}] using equation\n",expr(shape,pack,pre)?,expr(shape,value,pre)?,fact(shape,pack,pre)?));
    Ok(value)
}
