//! Actual switched decoder levels and strict canonical residues.
use std::collections::BTreeMap;

use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{BoundaryValue, LeanArtifact, ScopeProofShape},
    node::{ConcatAxis, LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

fn boundary(values: &BTreeMap<String, BoundaryValue>, count: usize) -> String {
    let mut types = vec![String::new(); count];
    for value in values.values() {
        types[value.tuple_index] = format!("({})", value.lean_type);
    }
    match count {
        0 => "Unit".into(),
        1 => types.remove(0),
        _ => types.join(" × ") + " × Unit",
    }
}
fn fact(shape: &ScopeProofShape, wire: WireRef, prefix: &str) -> Result<String, String> {
    let position = shape
        .node_constraints
        .get(&wire.node)
        .and_then(|r| r.clone().last())
        .ok_or("missing decoder operation constraint")?;
    Ok(format!("{prefix}{}.1", ".2".repeat(position)))
}
fn child(owner: WireRef) -> FrozenGraphScopeId {
    FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: owner.node,
    }
}

pub(super) fn root_refresh_decode(
    pregraph: &ValidatedGraph,
    preartifact: &LeanArtifact,
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    report: &crate::noise_refresh::simulation::ExponentLutNoiseReport,
) -> Result<String, String> {
    let preroot = pregraph.source.root_scope();
    let root = graph.source.root_scope();
    let preshape = &preartifact.scope_proofs[&FrozenGraphScopeId::Root];
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let pretypes = &pregraph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let prevalue = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &preshape.value_expressions)
            .map(|v| v.replace("witness.", "preWitness.").replace("inputs.", "preInputs."))
    };
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let prerecord = preshape.witness_record.as_ref().ok_or("missing PRE decoder witness")?;
    let record = shape.witness_record.as_ref().ok_or("missing VER decoder witness")?;
    let prebody = preshape.body_relation.as_ref().ok_or("missing PRE decoder body")?;
    let body = shape.body_relation.as_ref().ok_or("missing VER decoder body")?;
    let binders = format!(
        "{{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{preParams : {}}} {{params : {}}}\n    {{preInputs : PreInput}} {{preOutputs : PreOutput}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (preWitness : {prerecord}) (preFacts : {prebody} backend hashModel preParams preInputs preOutputs preWitness)\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)",
        preartifact.root.parameter_type, artifact.root.parameter_type
    );
    let mut source = format!(
        "import {}\nimport {}\nimport RootDecoderSetup\nimport ExponentRefreshDecode\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedRefreshDecode\nopen Mxx.Primitives MxxExponentLut\nabbrev PreInput := {}\nabbrev PreOutput := {}\nabbrev RootInput := {}\nabbrev RootOutput := {}\n",
        preartifact.module_name,
        artifact.module_name,
        boundary(&preartifact.root.inputs, preartifact.root.input_count),
        boundary(&preartifact.root.outputs, preartifact.root.output_count),
        boundary(&artifact.root.inputs, artifact.root.input_count),
        boundary(&artifact.root.outputs, artifact.root.output_count),
    );
    let mut slots = 0;
    let mut common_secret = None;
    for (name, preimage_output) in &preartifact.root.outputs {
        let Some((_, slot)) = name.rsplit_once("-preimage-") else { continue };
        let slot = slot.parse::<usize>().map_err(|_| "invalid decoder slot suffix")?;
        let ver_k = artifact.root.inputs.get(name).ok_or("missing imported decoder preimage")?.wire;
        let k = preimage_output.wire;
        let getter = preroot.node(k.node).ok_or("missing decoder preimage getter")?;
        let NodeKind::FamilyGetStatic { index } = getter.kind() else {
            return Err("decoder preimage not static group member".into())
        };
        let index = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic decoder group member")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "negative decoder group member")?;
        let family = preroot.arguments(getter).ok_or("missing decoder preimage family")?[0];
        let sample_id = child(family);
        let sample_scope = &pregraph.source.scopes()[&sample_id];

        let sample_wire = sample_scope.outputs()[0];
        let sample_node = sample_scope.node(sample_wire.node).ok_or("missing decoder sampler")?;
        let NodeKind::PreimageSample { max_coefficient_bound, .. } = sample_node.kind() else {
            return Err("decoder is not actual sampled preimage".into())
        };
        let cutoff = max_coefficient_bound
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic decoder bound")?;
        let sample_args =
            sample_scope.arguments(sample_node).ok_or("missing decoder sample operands")?;
        let family_node = preroot.node(family.node).ok_or("missing decoder family")?;
        let NodeKind::ParallelLoop(sample_loop) = family_node.kind() else {
            return Err("decoder sampler not parallel".into())
        };
        let family_args =
            preroot.arguments(family_node).ok_or("missing decoder family captures")?;
        let capture = |wire: WireRef, mode: LoopInputMode| -> Result<WireRef, String> {
            let slot = sample_scope
                .inputs()
                .iter()
                .position(|input| *input == wire)
                .ok_or("decoder operand not input")?;
            if sample_loop.input_modes[slot] != mode {
                return Err("decoder operand binding mode mismatch".into())
            }
            Ok(family_args[slot])
        };
        let pre_b = capture(sample_args[0], LoopInputMode::Broadcast)?;
        let trapdoor = capture(sample_args[1], LoopInputMode::Broadcast)?;
        let target_family = capture(sample_args[2], LoopInputMode::Zip)?;
        if !matches!(
            preroot.node(trapdoor.node).ok_or("missing decoder trapdoor")?.kind(),
            NodeKind::TrapdoorSample { .. }
        ) {
            return Err("decoder trapdoor not sampled".into())
        }
        let b_name = preartifact
            .root
            .outputs
            .iter()
            .find(|(_, v)| v.wire == pre_b)
            .ok_or("decoder B not exported")?
            .0;
        let b = artifact.root.inputs.get(b_name).ok_or("decoder B not imported")?.wire;
        let (anchor, low_secret, gaussian) = root
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(i, node)| {
                if !matches!(node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add)) {
                    return None
                }
                let args = root.arguments(node)?;
                if !matches!(root.node(args[1].node)?.kind(), NodeKind::GaussianSample { .. }) {
                    return None
                }
                let product = root.node(args[0].node)?;
                if !matches!(product.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) {
                    return None
                }
                let operands = root.arguments(product)?;
                if operands[1] != b ||
                    !matches!(
                        root.node(operands[0].node)?.kind(),
                        NodeKind::ModulusReduce { .. }
                    )
                {
                    return None
                }
                Some((WireRef { node: NodeId(i as u64), port: Port(0) }, operands[0], args[1]))
            })
            .ok_or("missing actual online decoder Gaussian anchor")?;
        let full_secret = root
            .arguments(root.node(low_secret.node).ok_or("missing reduced secret")?)
            .ok_or("missing reduction source")?[0];
        if common_secret.replace(full_secret).is_some_and(|old| old != full_secret) {
            return Err("decoder groups use different online secrets".into())
        }
        let secret_node = root.node(full_secret.node).ok_or("missing online secret constructor")?;
        if !matches!(secret_node.kind(), NodeKind::Concat { axis: ConcatAxis::Columns }) {
            return Err("online secret not augmented".into())
        }
        let secret_args = root.arguments(secret_node).ok_or("missing augmented secret terms")?;
        if secret_args.len() != 2 ||
            !matches!(
                root.node(secret_args[0].node).ok_or("missing small secret")?.kind(),
                NodeKind::UniformIntervalSample { .. }
            )
        {
            return Err("online secret not interval sampler plus unit".into())
        }
        let q = &types[&full_secret].matrix_type().ok_or("bad secret type")?.modulus;
        let matrix = pretypes[&k].matrix_type().ok_or("bad decoder type")?;
        let p = &matrix.modulus;
        let n = matrix.ring_dimension;
        let inner = matrix.rows;
        let columns = matrix.columns;
        let NodeKind::GaussianSample { max_coefficient_bound: anchor_cutoff, .. } =
            root.node(gaussian.node).ok_or("missing decoder error")?.kind()
        else {
            unreachable!()
        };
        let anchor_cutoff =
            anchor_cutoff.evaluate(&ParamEnv::default()).map_err(|_| "dynamic anchor cutoff")?;
        let target_id = child(target_family);
        let target_scope = &pregraph.source.scopes()[&target_id];

        let target_node = preroot.node(target_family.node).ok_or("missing target group")?;
        let NodeKind::ParallelLoop(target_loop) = target_node.kind() else {
            return Err("decoder targets not parallel".into())
        };
        let target_args =
            preroot.arguments(target_node).ok_or("missing decoder target captures")?;
        let result_node =
            target_scope.node(target_scope.outputs()[0].node).ok_or("missing target output")?;
        if !matches!(result_node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)) {
            return Err("decoder target not public difference".into())
        }
        let result_args = target_scope.arguments(result_node).ok_or("missing target difference")?;
        let switch = result_args[0];
        let switch_node = target_scope.node(switch.node).ok_or("missing public switch")?;
        if !matches!(switch_node.kind(), NodeKind::ModulusSwitch { .. }) {
            return Err("decoder target lacks actual modulus switch".into())
        }
        let product =
            target_scope.node(result_args[1].node).ok_or("missing target offset product")?;
        if !matches!(product.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) {
            return Err("decoder target offset not scalar product".into())
        }
        let factors = target_scope.arguments(product).ok_or("missing decoder target scalar")?;
        let source_public =
            target_scope.arguments(switch_node).ok_or("missing public switch input")?[0];
        let target_capture = |wire: WireRef| -> Result<String, String> {
            let slot = target_scope
                .inputs()
                .iter()
                .position(|input| *input == wire)
                .ok_or("decoder target operand not captured")?;
            let actual = prevalue(target_args[slot])?;
            Ok(match target_loop.input_modes[slot] {
                LoopInputMode::Broadcast => actual,
                LoopInputMode::Zip => format!("({actual}) position"),
                _ => return Err("decoder target capture uses shifted zip".into()),
            })
        };
        let high_public = target_capture(source_public)?;
        let delta = target_capture(factors[0])?;
        let low_a_prime = target_capture(factors[1])?;
        let ConcreteWireType::IndexedFamily { count, .. } = pretypes[&family] else {
            return Err("decoder family type".into())
        };
        let full = value(full_secret)?;
        let vb = value(b)?;
        let pb = prevalue(pre_b)?;
        let vk = value(ver_k)?;
        let pk = prevalue(k)?;
        let av = value(anchor)?;
        let crt = root
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::CrtRecompose { .. }))
            .ok_or("missing actual CRT recomposition")?;
        let NodeKind::CrtRecompose { plaintext_moduli, .. } = crt.kind() else { unreachable!() };
        let level = *root
            .arguments(crt)
            .ok_or("missing CRT levels")?
            .get(slot)
            .ok_or("missing slot level")?;
        let getter = root.node(level.node).ok_or("missing level getter")?;
        let NodeKind::FamilyGetStatic { index: local } = getter.kind() else {
            return Err("level is not static group member".into())
        };
        let local = local.evaluate(&ParamEnv::default()).map_err(|_| "dynamic level index")?;
        let online_family = root.arguments(getter).ok_or("missing level family")?[0];
        let online_node = root.node(online_family.node).ok_or("missing online switch group")?;
        let NodeKind::ParallelLoop(online_loop) = online_node.kind() else {
            return Err("level group not parallel".into())
        };
        let online_args = root.arguments(online_node).ok_or("missing online captures")?;
        let online_id = child(online_family);
        let online_scope = &graph.source.scopes()[&online_id];
        let online_shape = &artifact.scope_proofs[&online_id];
        let ConcreteWireType::IndexedFamily { count: online_count, .. } = types[&online_family]
        else {
            return Err("bad online group type".into())
        };
        let online_position = format!("(⟨{local}, by decide⟩ : Fin {online_count})");
        let mut online_values = online_shape.value_expressions.clone();
        for (input_index, input) in online_scope.inputs().iter().enumerate() {
            let actual = value(online_args[input_index])?;
            let actual = match online_loop.input_modes[input_index] {
                LoopInputMode::Broadcast => actual,
                LoopInputMode::Zip => format!("({actual}) {online_position}"),
                _ => return Err("shifted online switch capture".into()),
            };
            online_values.insert(format!("w_{}_{}", input.node.0, input.port.0), actual);
        }
        for (name, _) in &online_shape.witnesses {
            online_values.insert(name.clone(), format!("online_{name}"));
        }
        let ov = |wire: WireRef| {
            value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &online_values)
        };
        let result_wire = online_scope.outputs()[0];
        let result_node = online_scope.node(result_wire.node).ok_or("missing online result")?;
        if !matches!(result_node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)) {
            return Err("online level not switched ciphertext minus decoder".into())
        }
        let result_args =
            online_scope.arguments(result_node).ok_or("missing online subtraction")?;
        let switched_node =
            online_scope.node(result_args[0].node).ok_or("missing ciphertext switch")?;
        if !matches!(switched_node.kind(), NodeKind::ModulusSwitch { .. }) {
            return Err("actual ciphertext not switched".into())
        }
        let high_cipher =
            ov(online_scope.arguments(switched_node).ok_or("missing high ciphertext")?[0])?;
        let decoder_value = ov(result_args[1])?;
        let decoder_input = online_scope
            .inputs()
            .iter()
            .position(|w| *w == result_args[1])
            .ok_or("decoder subtraction operand not captured")?;
        if online_loop.input_modes[decoder_input] != LoopInputMode::Zip {
            return Err("decoder capture not zipped".into())
        }
        let decoder_pack = online_args[decoder_input];
        if !matches!(
            root.node(decoder_pack.node).ok_or("missing decoder pack")?.kind(),
            NodeKind::FamilyPack { .. }
        ) {
            return Err("decoder group is not actual family pack".into())
        }
        let decoder_pack_fact = fact(shape, decoder_pack, "facts")?;

        let switched_cipher = ov(result_args[0])?;
        let result_value = ov(result_wire)?;
        let actual_level = value(level)?;
        let preposition = format!("(⟨{index}, by decide⟩ : Fin {count})");
        let high_public = high_public.replace("position", &preposition);
        let correction = format!(
            "MxxRuntime.matrixMulScalarLeft ({}) ({low_a_prime})",
            delta.replace("position", &preposition)
        );
        let exact = report
            .refresh
            .slots
            .iter()
            .find(|r| r.slot == slot)
            .ok_or("missing exact slot noise report")?;
        if p.to_biguint().as_ref() != Some(&exact.switch_modulus) {
            return Err("slot report switching modulus differs from actual decoder".into())
        }
        let lift = &exact.lift;
        let prebound = &exact.pre_switch_bound;
        let gain = format!("({inner} * {n} * ({cutoff} : Int).toNat)");
        let routebound = format!(
            "switchedErrorBound {prebound} {lift} {n} ({gain} * ({anchor_cutoff} : Int).toNat)"
        );
        source.push_str(&format!(
            "theorem slot_report_bound_{slot} : ({routebound}) = {} := by decide\n\n",
            exact.residual_bound
        ));
        let common = format!(
            "{binders}\n    (secret : ErrorMatrix {n} 1 2)\n    (secretSource : reduceMatrix {q} {n} 1 2 secret = ({full}))\n    (small : ∀ i, ((secret 0 0).coeff i).natAbs ≤ 1) (unit : secret 0 1 = 1)\n    (publicLink : ({vb}) = ({pb})) (preimageLink : ({vk}) = ({pk}))\n    (lowIdeal : ErrorMatrix {n} 1 {columns})\n    (highIdeal : ExactMatrix {q} {n} 1 {columns})\n    (idealLift : highIdeal = reduceMatrix {q} {n} 1 {columns} (({lift} : ErrorPoly {n}) • lowIdeal))\n    (sourceApprox : Approx ({high_cipher}) (({full}) * ({high_public}) + highIdeal) {prebound})"
        );
        let ideal = format!(
            "reduceMatrix {p} {n} 1 {columns} lowIdeal + reduceMatrix {p} {n} 1 2 secret * ({correction})"
        );
        source.push_str(&format!("/-- The actual CRT input level after one paired switch and sampled decoder subtraction. -/\ntheorem slot_level_{slot} {common} :\n    Approx ({actual_level}) ({ideal}) ({routebound}) := by\n  obtain ⟨anchorError, action, switchedPublic, lowSecret, anchorEquation, anchorBound, actionEquation, actionBound, actionGain, publicSwitch, reconstruction⟩ :=\n    ExponentGeneratedDecoderSetup.decoder_setup_{slot} preWitness preFacts witness facts secret secretSource publicLink preimageLink\n"));
        let names = online_shape
            .witnesses
            .iter()
            .map(|(name, _)| format!("online_{name}"))
            .collect::<Vec<_>>()
            .join(", ");
        source.push_str(&format!("  obtain ⟨{names}, onlineFacts⟩ := {} {online_position}\n  have decoderEquation : ({decoder_value}) = ({av}) * ({vk}) := by\n    obtain ⟨actual, indexEquation, equation⟩ := ({decoder_pack_fact}).2.2 {online_position}\n    have same : actual = ⟨{local}, by change {local} < {online_count}; decide⟩ := Fin.ext indexEquation\n    simpa only [same] using equation\n  have levelEquation : ({actual_level}) = {result_value} :=\n    (refresh_lookup_eq {online_position} (by decide) {}).trans (onlineFacts{})\n  have sourceEquation : Approx ({high_cipher}) (reduceMatrix {q} {n} 1 2 secret * ({high_public}) +\n      reduceMatrix {q} {n} 1 {columns} (({lift} : ErrorPoly {n}) • lowIdeal)) {prebound} := by\n    rw [secretSource, ← idealLift]\n    exact sourceApprox\n  have bounded := refresh_switched_decoder (by decide) (by decide) (by decide) secret lowIdeal\n    ({high_cipher}) ({high_public}) ({switched_cipher}) switchedPublic ({correction})\n    ({vb}) ({av}) ({vk}) anchorError action small unit sourceEquation {} publicSwitch\n    anchorEquation reconstruction actionEquation actionGain anchorBound\n  rw [levelEquation]\n  simpa only [decoderEquation] using bounded\n\n",fact(shape,online_family,"facts")?,fact(shape,level,"facts")?,".2".repeat(online_shape.constraint_count),fact(online_shape,result_args[0],"onlineFacts")?));
        let plaintext = plaintext_moduli[slot]
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic plaintext modulus")?;
        let spacing = &exact.spacing;
        let mask = &exact.mask_bound;
        source.push_str(&format!("/-- Strict acceptance is an explicit finite numeric condition for rejected candidates. -/\ntheorem slot_rounded_{slot} {common}\n    (message maskError : ErrorMatrix {n} 1 {columns})\n    (idealSplit : ∀ row column coefficient, (({ideal}) row column).coeff coefficient =\n      (({spacing} : Int) * (message row column).coeff coefficient + (maskError row column).coeff coefficient : Int))\n    (maskBound : CoeffBound maskError {mask})\n    (accepted : 2 * ({mask} + ({routebound})) < {spacing}) :\n    ∀ row column coefficient,\n      ((({plaintext} : Int) * Int.ofNat ((({actual_level}) row column).coeff coefficient).val +\n        Int.ofNat ({p} / 2)) / Int.ofNat {p}) % ({plaintext} : Int) =\n        (message row column).coeff coefficient % ({plaintext} : Int) := by\n  exact refresh_rounded_residues (by decide) (by decide) (by decide) (by decide) (by decide)\n    ({actual_level}) ({ideal}) message maskError\n    (slot_level_{slot} preWitness preFacts witness facts secret secretSource small unit publicLink preimageLink lowIdeal highIdeal idealLift sourceApprox)\n    idealSplit maskBound accepted\n\n"));
        if exact.rounding_margin > 0.into() {
            source.push_str(&format!("theorem slot_accepted_{slot} : 2 * ({mask} + ({routebound})) < {spacing} := by decide\n\n"));
        }
        slots += 1;
    }
    if slots == 0 {
        return Err("no actual refresh decoder slots".into())
    }
    source.push_str("end ExponentGeneratedRefreshDecode\n");
    Ok(source)
}
