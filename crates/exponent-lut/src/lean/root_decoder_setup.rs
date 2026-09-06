//! Decoder anchor and sampled-preimage facts from one linked producer/consumer run.
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

pub(super) fn root_decoder_setup(
    pregraph: &ValidatedGraph,
    preartifact: &LeanArtifact,
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
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
        "import {}\nimport {}\nimport ExponentSecret\nimport ExponentSetup\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedDecoderSetup\nopen Mxx.Primitives MxxExponentLut\nabbrev PreInput := {}\nabbrev PreOutput := {}\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n{}\n",
        preartifact.module_name,
        artifact.module_name,
        boundary(&preartifact.root.inputs, preartifact.root.input_count),
        boundary(&preartifact.root.outputs, preartifact.root.output_count),
        boundary(&artifact.root.inputs, artifact.root.input_count),
        boundary(&artifact.root.outputs, artifact.root.output_count),
        HELPERS
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
        let sample_shape = &preartifact.scope_proofs[&sample_id];
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
        let target_shape = &preartifact.scope_proofs[&target_id];
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
        let low = value(low_secret)?;
        let vb = value(b)?;
        let pb = prevalue(pre_b)?;
        let vk = value(ver_k)?;
        let pk = prevalue(k)?;
        let av = value(anchor)?;
        if slots == 0 {
            source.push_str(&format!("theorem secret_source {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∃ secret : ErrorMatrix {n} 1 2, reduceMatrix {q} {n} 1 2 secret = ({full}) ∧\n      (∀ i, ((secret 0 0).coeff i).natAbs ≤ 1) ∧ secret 0 1 = 1 := by\n  exact augmented_secret_of_sample ({}) ({}) ({full}) {} {} (by simp)\n\n",artifact.root.parameter_type,value(secret_args[0])?,value(secret_args[1])?,fact(shape,secret_args[0],"facts")?,fact(shape,full_secret,"facts")?));
        }
        source.push_str(&format!("theorem decoder_setup_{slot} {binders}\n    (secret : ErrorMatrix {n} 1 2) (secretSource : reduceMatrix {q} {n} 1 2 secret = ({full}))\n    (publicLink : ({vb}) = ({pb})) (preimageLink : ({vk}) = ({pk})) :\n    let position : Fin {count} := ⟨{index}, by decide⟩\n    ∃ anchorError : ErrorMatrix {n} 1 {inner}, ∃ action : ErrorMatrix {n} {inner} {columns},\n    ∃ switchedPublic : ExactMatrix {p} {n} 2 {columns},\n      reduceMatrix {p} {n} 1 2 secret = ({low}) ∧\n      ({av}) = reduceMatrix {p} {n} 1 2 secret * ({vb}) + reduceMatrix {p} {n} 1 {inner} anchorError ∧\n      CoeffBound anchorError ({anchor_cutoff} : Int).toNat ∧\n      ({vk}) = reduceMatrix {p} {n} {inner} {columns} action ∧\n      CoeffBound action ({cutoff} : Int).toNat ∧ ActionGain action ({inner} * {n} * ({cutoff} : Int).toNat) ∧\n      MxxRuntime.modulusSwitchRuns ({high_public}) switchedPublic ∧\n      ({vb}) * ({vk}) = switchedPublic - MxxRuntime.matrixMulScalarLeft ({delta}) ({low_a_prime}) := by\n  let position : Fin {count} := ⟨{index}, by decide⟩\n  have lowSecret := reduce_secret (by decide) (by decide) secret ({full}) ({low}) secretSource {}\n  obtain ⟨anchorError, anchorReduction, anchorBound⟩ := gaussian_error_witness {}\n",fact(shape,low_secret,"facts")?,fact(shape,gaussian,"facts")?));
        let names = sample_shape
            .witnesses
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let sample_var = format!("w_{}_{}", sample_wire.node.0, sample_wire.port.0);
        let sample_end = format!("sampleFacts{}", ".2".repeat(sample_shape.constraint_count));
        source.push_str(&format!("  obtain ⟨{names}, sampleFacts⟩ := {} position\n  have sampled := {}\n  have kEquation : ({pk}) = {sample_var} := (lookup_eq position (by decide) {}).trans ({sample_end})\n  have trapdoorKind := MxxRuntime.trapdoorSample_sampled {}\n  obtain ⟨action, actionReduction, actionBound, actionGain⟩ := sampled_action trapdoorKind sampled\n  have reconstruction := MxxRuntime.preimageRunsDispatched_equation (by decide) (by decide) sampled\n",fact(preshape,family,"preFacts")?,fact(sample_shape,sample_wire,"sampleFacts")?,fact(preshape,k,"preFacts")?,fact(preshape,trapdoor,"preFacts")?));
        let switch_names = target_shape
            .witnesses
            .iter()
            .map(|(name, _)| format!("target_{name}"))
            .collect::<Vec<_>>()
            .join(", ");
        let switched = format!("target_w_{}_{}", switch.node.0, switch.port.0);
        let target_end = format!("targetFacts{}", ".2".repeat(target_shape.constraint_count));
        source.push_str(&format!("  obtain ⟨{switch_names}, targetFacts⟩ := {} position\n  refine ⟨anchorError, action, {switched}, lowSecret, ?_, anchorBound, ?_, actionBound, actionGain, {}, ?_⟩\n  · change ({low}) * ({vb}) + _ = _\n    rw [anchorReduction, lowSecret]\n  · exact preimageLink.trans (kEquation.trans actionReduction)\n  · rw [publicLink, preimageLink, kEquation, reconstruction, {target_end}]\n    rfl\n\n",fact(preshape,target_family,"preFacts")?,fact(target_shape,switch,"targetFacts")?));
        slots += 1;
    }
    if slots == 0 {
        return Err("no decoder preimages found".into())
    }
    source.push_str("end ExponentGeneratedDecoderSetup\n");
    Ok(source)
}

const HELPERS: &str = r#"private theorem poly_ext {q n : Nat} (hq : 1 < q) (hn : 0 < n)
    {a b : ExactPoly q n} (h : ∀ i, a.coeff i = b.coeff i) : a = b := by
  letI : Fact (1 < q) := ⟨hq⟩
  rw [Negacyclic.expansion hn a, Negacyclic.expansion hn b]
  apply Finset.sum_congr rfl
  intro i hi
  rw [h i]

private theorem reduce_secret {q p n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n)
    (secret : ErrorMatrix n rows columns)
    (high : ExactMatrix q n rows columns) (low : ExactMatrix p n rows columns)
    (source : reduceMatrix q n rows columns secret = high)
    (runs : MxxRuntime.modulusReduceRuns high low) :
    reduceMatrix p n rows columns secret = low := by
  rcases runs with ⟨hp, hd, rfl⟩
  letI : NeZero q := ⟨by omega⟩
  apply exactMatrix_ext
  intro row column
  apply poly_ext hp hn
  intro i
  simp only [MxxRuntime.modulusReduce, MxxRuntime.polynomialOfCoefficients_coeff hp hn]
  rw [← source]
  simp only [reduceMatrix, reducePoly_coeff hq hn, reducePoly_coeff hp hn]
  have h := congrArg (ZMod.castHom hd (ZMod p))
    (ZMod.natCast_zmod_val (((secret row column).coeff i) : ZMod q))
  simpa only [map_natCast, map_intCast, Int.ofNat_eq_natCast, Int.cast_natCast] using h.symm

private theorem sampled_action {q n rows inner columns cutoff : Nat}
    {backend : MxxRuntime.BackendContext} {Token : Type}
    {b : ExactMatrix q n rows inner}
    {trapdoor : MxxRuntime.TrapdoorValue (ExactMatrix q n rows inner) Token}
    {target : ExactMatrix q n rows columns} {d : ExactMatrix q n inner columns}
    (kind : trapdoor.kind = .sampledSecret)
    (runs : MxxRuntime.preimageRunsDispatched backend b trapdoor target cutoff d) :
    ∃ action : ErrorMatrix n inner columns,
      d = reduceMatrix q n inner columns action ∧ CoeffBound action cutoff ∧
      ActionGain action (inner * n * cutoff) := by
  rcases runs.2 with sampled | publicCase
  · obtain ⟨action, heq, hbound⟩ := sampled.2.2.2
    exact ⟨action, heq, hbound, bounded_matrix_action_gain hbound⟩
  · have contradiction := publicCase.1
    rw [kind] at contradiction
    cases contradiction

private theorem lookup_eq {α : Type} {count : Nat} {family : Fin count → α}
    {index : Int} {output : α} (position : Fin count)
    (hindex : index = (position.val : Int))
    (h : MxxRuntime.familyGetStatic family index output) : output = family position := by
  obtain ⟨actual, ha, hv⟩ := h
  have same : actual = position := by apply Fin.ext; omega
  simpa only [same] using hv
"#;
