//! Downstream refresh correctness from the exact per-slot source approximations.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::{ExponentLutNoiseReport, ExponentLutNoiseSnapshot};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{LeanArtifact, ScopeProofShape},
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
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
pub(super) fn root_refresh_finish(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    ver: &ValidatedGraph,
    va: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
    report: &ExponentLutNoiseReport,
) -> Result<String, String> {
    if report.refresh.slots.iter().any(|slot| slot.rounding_margin <= 0.into()) {
        return Ok("-- This numeric report fails strict slot rounding; no finish theorem is emitted.\nimport RootRefreshDecode\nnamespace ExponentGeneratedRefreshFinish\nend ExponentGeneratedRefreshFinish\n".into());
    }
    let pr = pre.source.root_scope();
    let vr = ver.source.root_scope();
    let ps = &pa.scope_proofs[&FrozenGraphScopeId::Root];
    let vs = &va.scope_proofs[&FrozenGraphScopeId::Root];
    let pt = &pre.scopes[&FrozenGraphScopeId::Root].wire_types;
    let vt = &ver.scopes[&FrozenGraphScopeId::Root].wire_types;
    let pv = |w| expr(ps, w, true);
    let vv = |w| expr(vs, w, false);
    let pargs = |w: WireRef| {
        pr.node(w.node).and_then(|x| pr.arguments(x)).ok_or("missing finish PRE operands")
    };
    let vargs = |w: WireRef| {
        vr.node(w.node).and_then(|x| vr.arguments(x)).ok_or("missing finish VER operands")
    };
    let (crt_id, crt) = vr
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, x)| matches!(x.kind(), NodeKind::CrtRecompose { .. }))
        .ok_or("missing finish CRT")?;
    let crt_wire = WireRef { node: NodeId(crt_id as u64), port: Port(0) };
    let NodeKind::CrtRecompose { plaintext_moduli, .. } = crt.kind() else { unreachable!() };
    let levels = vr.arguments(crt).ok_or("missing finish levels")?;
    let (endpoint, anchor) = va
        .root
        .outputs
        .values()
        .find_map(|o| {
            let n = vr.node(o.wire.node)?;
            if !matches!(n.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)) {
                return None
            }
            let a = vr.arguments(n)?;
            (a[0] == crt_wire).then_some((o, a[1]))
        })
        .ok_or("missing finish residual endpoint")?;
    let anchor_args = vargs(anchor)?;
    let anchor_product = vargs(anchor_args[0])?;
    let full_secret = anchor_product[0];
    let aprime = anchor_product[1];
    let payload = anchor_args[1];
    let mt = vt[&anchor].matrix_type().ok_or("finish anchor type")?;
    let q = &mt.modulus;
    let n = mt.ring_dimension;
    let columns = mt.columns;
    let width = 1usize << snapshot.layout.width_log2;
    let slots = levels.len();
    let fresh_count = snapshot.refresh.fresh_digit_count;
    let fresh_bound = snapshot
        .refresh
        .base_p
        .pow(u32::try_from(fresh_count).map_err(|_| "finish fresh digit count overflow")?) -
        1u8;
    let mask_total = snapshot.refresh.mask_digit_counts.iter().sum::<usize>() * columns * n;
    let labels = mask_total + columns * n * fresh_count;
    let bind = "{hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution} (runs : Runs hashModel external execution)";
    let selection = format!("Fin {labels} → Fin (2 * {width})");
    let mut s = String::from(
        "import ExecutionWitnesses\nimport RootRefreshNormalize\nimport RootRefreshDecode\nimport RootRefreshCrt\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshFinish\nopen Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution\nopen scoped BigOperators\n\n",
    );
    let getters = |value: &str| {
        GETTERS
            .lines()
            .filter(|line| {
                value.contains(&format!("({})", line.split_whitespace().nth(1).unwrap_or("")))
            })
            .map(|line| format!("{line}\n"))
            .collect::<String>()
    };
    let payload_getters = getters(&vv(payload)?);
    let public_getters = getters(&vv(aprime)?);
    let secret_getters = getters(&vv(full_secret)?);
    s.push_str(&format!("noncomputable def secretInteger {bind} : ErrorMatrix {n} 1 2 :=\n  Classical.choose (ExponentGeneratedDecoderSetup.secret_source (verificationWitness runs) (verificationFacts runs))\n\nnoncomputable def payloadInteger {bind} : ErrorMatrix {n} 1 {columns} :=\n{payload_getters}  refreshFreshLift fun row col k ↦ Int.ofNat ((({}) row col).coeff k).val\n\nnoncomputable def publicInteger {bind} : ErrorMatrix {n} 2 {columns} :=\n{public_getters}  fun row col ↦ ∑ k : Fin {n}, (Int.ofNat ((({}) row col).coeff k).val : ErrorPoly {n}) * AdjoinRoot.root (negacyclicModulus {n} Int) ^ k.val\n\nnoncomputable def anchorInteger {bind} : ErrorMatrix {n} 1 {columns} := secretInteger runs * publicInteger runs - payloadInteger runs\n\nnoncomputable def freshSelection (selection : {selection}) : Fin 1 → Fin {columns} → Fin {n} → Fin ExponentGeneratedRefreshDigits.freshCount → Fin (2 * {width}) :=\n  fun _ col k d ↦ selection ⟨{mask_total} + (col.val * {n} + k.val) * {fresh_count} + d.val, by have := d.isLt; change d.val < {fresh_count} at this; omega⟩\n\n",vv(payload)?,vv(aprime)?));
    s.push_str(&format!("theorem secret_source {bind} :\n{secret_getters}  reduceMatrix {q} {n} 1 2 (secretInteger runs) = ({}) ∧\n    (∀ i, (((secretInteger runs) 0 0).coeff i).natAbs ≤ 1) ∧ (secretInteger runs) 0 1 = 1 :=\n  Classical.choose_spec (ExponentGeneratedDecoderSetup.secret_source (verificationWitness runs) (verificationFacts runs))\n\n",vv(full_secret)?));
    let mut rounded = std::collections::BTreeMap::new();
    for (name, kout) in &pa.root.outputs {
        let Some((_, slot)) = name.rsplit_once("-preimage-") else { continue };
        let slot = slot.parse::<usize>().map_err(|_| "bad finish slot")?;
        let k = kout.wire;
        let vk = va.root.inputs.get(name).ok_or("missing finish K link")?.wire;
        let NodeKind::FamilyGetStatic { index } = pr.node(k.node).ok_or("missing finish K")?.kind()
        else {
            return Err("finish K not gathered".into())
        };
        let index = index.evaluate(&ParamEnv::default()).map_err(|_| "dynamic finish K index")?;
        let family = pargs(k)?[0];
        let sc = &pre.source.scopes()[&child(family)];
        let NodeKind::ParallelLoop(loop_) = pr.node(family.node).ok_or("missing sampler")?.kind()
        else {
            return Err("finish sampler not parallel".into())
        };
        let captures = pargs(family)?;
        let sample_args = sc
            .arguments(sc.node(sc.outputs()[0].node).ok_or("missing sampler output")?)
            .ok_or("missing sampler operands")?;
        let capture = |w: WireRef| {
            sc.inputs()
                .iter()
                .position(|x| *x == w)
                .map(|i| captures[i])
                .ok_or("finish sample capture")
        };
        let pb = capture(sample_args[0])?;
        let target_family = capture(sample_args[2])?;
        if loop_.input_modes
            [sc.inputs().iter().position(|w| *w == sample_args[2]).ok_or("missing target mode")?] !=
            LoopInputMode::Zip
        {
            return Err("finish target not zipped".into())
        }
        let bname =
            pa.root.outputs.iter().find(|(_, o)| o.wire == pb).ok_or("finish B not exported")?.0;
        let vb = va.root.inputs.get(bname).ok_or("finish B not linked")?.wire;
        let ts = &pre.source.scopes()[&child(target_family)];
        let NodeKind::ParallelLoop(tloop) =
            pr.node(target_family.node).ok_or("missing target loop")?.kind()
        else {
            return Err("finish target not parallel".into())
        };
        let tc = pargs(target_family)?;
        let ta = |w: WireRef| {
            ts.node(w.node).and_then(|x| ts.arguments(x)).ok_or("finish target operands")
        };
        let diff = ta(ts.outputs()[0])?;
        let hpwire = ta(diff[0])?[0];
        let factors = ta(diff[1])?;
        let target_wire =
            |w: WireRef| ts.inputs().iter().position(|x| *x == w).ok_or("finish target capture");
        let di = target_wire(factors[0])?;
        let li = target_wire(factors[1])?;
        if tloop.input_modes[di] != LoopInputMode::Zip ||
            tloop.input_modes[li] != LoopInputMode::Broadcast
        {
            return Err("finish scalar/public capture mode".into())
        }
        let delta_family = tc[di];
        let low_public = tc[li];
        let high_public_pre = pargs(low_public)?[0];
        if !matches!(
            pr.node(low_public.node).map(|x| x.kind()),
            Some(NodeKind::ModulusReduce { .. })
        ) {
            return Err("finish low public not reduction".into())
        }
        let ConcreteWireType::IndexedFamily { count, .. } = pt[&family] else {
            return Err("finish family type".into())
        };
        let pos = format!("(⟨{index}, by decide⟩ : Fin {count})");
        let hi = target_wire(hpwire)?;
        let high_public = match tloop.input_modes[hi] {
            LoopInputMode::Broadcast => pv(tc[hi])?,
            LoopInputMode::Zip => format!("({}) {pos}", pv(tc[hi])?),
            _ => return Err("shifted finish high public".into()),
        };
        let level = levels[slot];
        let NodeKind::FamilyGetStatic { index: local } =
            vr.node(level.node).ok_or("finish level")?.kind()
        else {
            return Err("finish level not gather".into())
        };
        let local = local.evaluate(&ParamEnv::default()).map_err(|_| "dynamic local slot")?;
        let online = vargs(level)?[0];
        let os = &ver.source.scopes()[&child(online)];
        let ops = &va.scope_proofs[&child(online)];
        let NodeKind::ParallelLoop(ol) = vr.node(online.node).ok_or("finish online group")?.kind()
        else {
            return Err("finish online group not parallel".into())
        };
        let oc = vargs(online)?;
        let ConcreteWireType::IndexedFamily { count: ocount, .. } = vt[&online] else {
            return Err("finish online type".into())
        };
        let mut values = ops.value_expressions.clone();
        for (i, w) in os.inputs().iter().enumerate() {
            let v = vv(oc[i])?;
            values.insert(
                format!("w_{}_{}", w.node.0, w.port.0),
                match ol.input_modes[i] {
                    LoopInputMode::Broadcast => v,
                    LoopInputMode::Zip => format!("({v}) (⟨{local}, by decide⟩ : Fin {ocount})"),
                    _ => return Err("finish shifted online input".into()),
                },
            );
        }
        let oa =
            |w: WireRef| os.node(w.node).and_then(|x| os.arguments(x)).ok_or("finish online args");
        let source_wire = oa(oa(os.outputs()[0])?[0])?[0];
        let high_cipher =
            value_expression(&format!("w_{}_{}", source_wire.node.0, source_wire.port.0), &values)?;
        let plaintext = plaintext_moduli[slot]
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic finish plaintext")?;
        let exact = report
            .refresh
            .slots
            .iter()
            .find(|r| r.slot == slot)
            .ok_or("missing finish report slot")?;
        let p = &exact.switch_modulus;
        let j = &exact.lift;
        let delta = &exact.spacing;
        let prebound = &exact.pre_switch_bound;
        let md = snapshot.refresh.mask_digit_counts[slot];
        let offset = snapshot.refresh.mask_digit_counts[..slot].iter().sum::<usize>() * n * columns;
        let source_getters = getters(&format!("{high_cipher} {high_public} {}", vv(full_secret)?));
        let rounded_getters = getters(&vv(level)?);
        let list_successor = if index == 0.into() { "" } else { ", List.getElem_cons_succ" };
        s.push_str(&format!("noncomputable def maskSelection_{slot} (selection : {selection}) : Fin 1 → Fin {columns} → Fin {n} → Fin (ExponentGeneratedRefreshDigits.maskCount ⟨{slot}, by decide⟩) → Fin (2 * {width}) :=\n  fun _ col k d ↦ selection ⟨{offset} + (col.val * {n} + k.val) * {md} + d.val, by have := d.isLt; change d.val < {md} at this; omega⟩\n\nnoncomputable def lowIdeal_{slot} {bind} (selection : {selection}) : ErrorMatrix {n} 1 {columns} :=\n  ({delta} : Int) • (-payloadInteger runs + ExponentGeneratedRefreshDigits.freshInteger (freshSelection selection)) +\n    ExponentGeneratedRefreshDigits.maskInteger ⟨{slot}, by decide⟩ (maskSelection_{slot} selection)\n\ndef SourceApprox_{slot} {bind} (selection : {selection}) : Prop :=\n{source_getters}  Approx ({high_cipher}) (({}) * ({high_public}) + reduceMatrix {q} {n} 1 {columns} (({j} : ErrorPoly {n}) • lowIdeal_{slot} runs selection)) {prebound}\n\n",vv(full_secret)?));
        let mut proof = format!(
            "theorem rounded_{slot} {bind} (selection : {selection}) (source : SourceApprox_{slot} runs selection) :\n{rounded_getters}  ∀ row col (k : Fin {n}), ((({} : Int) * Int.ofNat ((({}) row col).coeff k).val + Int.ofNat ({p} / 2)) / Int.ofNat {p}) % ({} : Int) =\n    ((anchorInteger runs + ExponentGeneratedRefreshDigits.freshInteger (freshSelection selection)) row col).coeff k % ({} : Int) := by\n{CONTEXT}  have secretFacts := secret_source runs\n  have publicLink : ({}) = ({}) := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n  have preimageLink : ({}) = ({}) := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n  have highPublicLink : ({}) = ({}) := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n  have lowPublic : ({}) = reduceMatrix {p} {n} 2 {columns} (publicInteger runs) := by\n    have lowEquation : ({}) = MxxRuntime.modulusReduce ({}) := ({}).2.2\n    rw [lowEquation, ← highPublicLink]\n    exact (refresh_public_integer_reduce (by decide) (by decide) _).symm\n",
            plaintext,
            vv(level)?,
            plaintext,
            plaintext,
            vv(vb)?,
            pv(pb)?,
            vv(vk)?,
            pv(k)?,
            vv(aprime)?,
            pv(high_public_pre)?,
            pv(low_public)?,
            pv(low_public)?,
            pv(high_public_pre)?,
            fact(ps, low_public, true)?
        );
        proof.push_str(&format!("  have deltaScalar : (({}) {pos}) 0 0 = ({delta} : ExactPoly {p} {n}) := by\n    obtain ⟨actual, indexEq, equation⟩ := ({}).2.2 {pos}\n    have same : actual = ⟨{index}, by decide⟩ := Fin.ext indexEq\n    rw [same] at equation\n    have scalar := congrArg (fun matrix : ExactMatrix {p} {n} 1 1 ↦ matrix 0 0) equation\n    simp only [List.get_eq_getElem, Fin.val_mk, List.getElem_cons_zero{list_successor}] at scalar\n    norm_num [MxxRuntime.matrixPolynomial, MxxIR.exactDiv] at scalar\n    exact scalar\n  have correction : MxxRuntime.matrixMulScalarLeft (({}) {pos}) ({}) =\n      ({delta} : Int) • reduceMatrix {p} {n} 2 {columns} (publicInteger runs) := by\n    ext row col\n    simp only [MxxRuntime.matrixMulScalarLeft, deltaScalar, lowPublic, Matrix.smul_apply, zsmul_eq_mul]\n    norm_num\n  apply ExponentGeneratedRefreshDecode.slot_rounded_{slot} pw pf vw vf (secretInteger runs) secretFacts.1 secretFacts.2.1 secretFacts.2.2 publicLink preimageLink\n    (lowIdeal_{slot} runs selection) _ rfl source (anchorInteger runs + ExponentGeneratedRefreshDigits.freshInteger (freshSelection selection))\n    (ExponentGeneratedRefreshDigits.maskInteger ⟨{slot}, by decide⟩ (maskSelection_{slot} selection))\n  · intro row col k\n    rw [correction]\n    exact ExponentGeneratedRefreshNormalize.decoder_ideal_coeff_{slot} (secretInteger runs) (publicInteger runs) (payloadInteger runs) (freshSelection selection) (maskSelection_{slot} selection) row col k\n  · exact ExponentGeneratedRefreshDigits.mask_integer_bound ⟨{slot}, by decide⟩ (maskSelection_{slot} selection)\n  · exact ExponentGeneratedRefreshDecode.slot_accepted_{slot}\n\n",pv(delta_family)?,fact(ps,delta_family,true)?,pv(delta_family)?,pv(low_public)?));
        rounded.insert(slot, proof);
    }
    for (_, proof) in rounded {
        s.push_str(&proof);
    }
    s.push_str(&format!("theorem finish {bind} (selection : {selection})\n"));
    for slot in 0..slots {
        s.push_str(&format!("    (source_{slot} : SourceApprox_{slot} runs selection)\n"));
    }
    s.push_str(&format!("    : Approx ({}) execution.«ideal» {fresh_bound} := by\n{CONTEXT}  have idealZero : execution.«ideal» = 0 := by\n    obtain ⟨witness, facts⟩ := runs.2.2.2\n    exact facts\n  rw [idealZero]\n  have anchorSource : ({}) = reduceMatrix {q} {n} 1 {columns} (anchorInteger runs) := by\n    have secretFacts := secret_source runs\n    have payloadSource := refresh_anchor_lift (by decide : 1 < {q}) (by decide : 0 < {n}) ({})\n    have publicSource := refresh_public_integer_source (by decide : 1 < {q}) (by decide : 0 < {n}) ({})\n    change _ = reduceMatrix {q} {n} 1 {columns} (secretInteger runs * publicInteger runs - payloadInteger runs)\n    have subtract : reduceMatrix {q} {n} 1 {columns} (secretInteger runs * publicInteger runs - payloadInteger runs) =\n        reduceMatrix {q} {n} 1 2 (secretInteger runs) * reduceMatrix {q} {n} 2 {columns} (publicInteger runs) - reduceMatrix {q} {n} 1 {columns} (payloadInteger runs) := by\n      ext row col\n      have h := congrArg (fun matrix => matrix row col) (reduceMatrix_mul {q} {n} 1 2 {columns} (secretInteger runs) (publicInteger runs))\n      simpa only [reduceMatrix, Matrix.sub_apply, map_sub] using congrArg (fun x => x - reducePoly {q} {n} (payloadInteger runs row col)) h\n    rw [subtract, secretFacts.1, show reduceMatrix {q} {n} 2 {columns} (publicInteger runs) = ({}) from publicSource, show reduceMatrix {q} {n} 1 {columns} (payloadInteger runs) = ({}) from payloadSource]\n    rfl\n  exact ExponentGeneratedRefreshCrt.residual_of_rounds vw vf (anchorInteger runs)\n    (fun row col k ↦ ExponentGeneratedRefreshDigits.freshValue (freshSelection selection row col k)) anchorSource\n",endpoint.projection.replace("outputs","execution.«verification»"),vv(anchor)?,vv(payload)?,vv(aprime)?,vv(aprime)?,vv(payload)?));
    for slot in 0..slots {
        s.push_str(&format!("    (rounded_{slot} runs selection source_{slot})\n"));
    }
    s.push_str("    (fun row col k ↦ ExponentGeneratedRefreshDigits.fresh_bound (freshSelection selection row col k))\n\nend ExponentGeneratedRefreshFinish\n");
    Ok(s)
}
