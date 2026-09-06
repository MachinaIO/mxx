//! Close each actual state-scaling source on the witnesses selected by `Runs`.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::ExponentLutNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    graph::GraphScope,
    lean::{LeanArtifact, ScopeProofShape},
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{NodeId, Port, WireRef},
};

fn wire(index: usize) -> WireRef {
    WireRef { node: NodeId(index as u64), port: Port(0) }
}

fn args(root: &GraphScope, wire: WireRef) -> Result<Vec<WireRef>, String> {
    root.node(wire.node)
        .and_then(|node| root.arguments(node))
        .ok_or("missing state operands".into())
}

fn value(
    shape: &ScopeProofShape,
    wire: WireRef,
    witness: &str,
    inputs: &str,
) -> Result<String, String> {
    let mut values = shape.value_expressions.clone();
    values.insert("witness".into(), witness.into());
    values.insert("inputs".into(), inputs.into());
    value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &values)
}

fn fact(shape: &ScopeProofShape, wire: WireRef, facts: &str) -> Result<String, String> {
    let position = shape
        .node_constraints
        .get(&wire.node)
        .and_then(|range| range.clone().last())
        .ok_or("missing state operation fact")?;
    Ok(format!("{facts}{}.1", ".2".repeat(position)))
}

pub(super) fn root_refresh_state_source(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    ver: &ValidatedGraph,
    va: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
) -> Result<String, String> {
    let pr = pre.source.root_scope();
    let vr = ver.source.root_scope();
    let ps = &pa.scope_proofs[&FrozenGraphScopeId::Root];
    let vs = &va.scope_proofs[&FrozenGraphScopeId::Root];
    let pv = |w| value(ps, w, "pw", "pin");
    let vv = |w| value(vs, w, "vw", "vin");
    let count = snapshot.refresh.slot_count;
    let caches = (0..count)
        .map(|slot| {
            let suffix = format!("-state-scaling-k-rhs-{slot}");
            let p = pa
                .root
                .outputs
                .iter()
                .find(|(name, _)| name.ends_with(&suffix))
                .map(|(_, v)| v.wire)
                .ok_or("missing PRE state cache")?;
            let v = va
                .root
                .inputs
                .iter()
                .find(|(name, _)| name.ends_with(&suffix))
                .map(|(_, v)| v.wire)
                .ok_or("missing VER state cache")?;
            Ok((p, v))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let packed = pr
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, node)| {
            (matches!(node.kind(), NodeKind::FamilyPack { .. }) &&
                pr.arguments(node) == Some(caches.iter().map(|(p, _)| *p).collect()))
            .then_some(wire(i))
        })
        .ok_or("missing ordered actual state cache pack")?;
    let mut products = pr.nodes().iter().enumerate().filter_map(|(i, node)| {
        let NodeKind::ParallelLoop(spec) = node.kind() else {
            return None;
        };
        let parents = pr.arguments(node)?;
        let cache_position = parents.iter().position(|w| *w == packed)?;
        let id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: wire(i).node,
        };
        let scope = pre.source.scopes().get(&id)?;
        let output = scope.node(scope.outputs()[0].node)?;
        if !matches!(
            output.kind(),
            NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) {
            return None;
        }
        let operands = scope.arguments(output)?;
        let left = scope.inputs().iter().position(|w| *w == operands[0])?;
        let right = scope.inputs().iter().position(|w| *w == operands[1])?;
        (right == cache_position &&
            spec.input_modes[right] == LoopInputMode::Zip &&
            spec.input_modes[left] == LoopInputMode::Broadcast)
            .then_some((wire(i), parents[left], id))
    });
    let (public_product, public_input, public_scope) =
        products.next().ok_or("missing actual PRE state public multiplication")?;
    if products.next().is_some() {
        return Err("ambiguous PRE state multiplication".into());
    }
    let public_relation = &pa.scope_relations[&public_scope];
    let binders = "{hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}\n    (runs : Runs hashModel external execution)";
    let locals = "  let pw := preprocessingWitness runs\n  let vw := verificationWitness runs\n  let pin := preprocessingInputs runs.2.1\n  let vin := verificationInputs runs.2.2.1\n";
    let mut source = String::from(
        "import RootInputs\nimport RootRefreshState\nimport RootReadoutProof\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshStateSource\nopen Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution\n\nprivate theorem state_transfer {q n columns secret bound gain : Nat} (hn : 0 < n)\n    {c : ExactMatrix q n 1 columns} {a g : ExactMatrix q n secret columns}\n    {s payload : ExactMatrix q n 1 secret} {d : ExactMatrix q n columns columns}\n    {action : ErrorMatrix n columns columns} {mu : Int}\n    (close : Approx c (s * a - payload * g) bound)\n    (reduction : d = reduceMatrix q n columns columns action)\n    (bounded : ActionGain action gain) (cache : g * d = mu • g) :\n    Approx (c * d) (s * (a * d) - mu • (payload * g)) (gain * bound) := by\n  have result := approximate_right_action hn close bounded\n  rw [← reduction] at result\n  rw [Matrix.sub_mul, Matrix.mul_assoc, Matrix.mul_assoc, cache, Matrix.mul_smul] at result\n  exact result\n\n",
    );
    source.push_str(&format!("abbrev initialBound : Nat := {}\n\n", snapshot.initial_state_bound));
    let mut initial = None;
    for (slot, (pcache, vcache)) in caches.into_iter().enumerate() {
        let mut actuals = vr.nodes().iter().enumerate().filter_map(|(i, node)| {
            if !matches!(
                node.kind(),
                NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
            ) {
                return None;
            }
            let operands = vr.arguments(node)?;
            (operands.len() == 2 && operands[1] == vcache).then_some((wire(i), operands[0]))
        });
        let (actual, input) = actuals.next().ok_or("missing actual VER state multiplication")?;
        if actuals.next().is_some() {
            return Err("ambiguous VER state multiplication".into());
        }
        if initial.is_some_and(|w| w != input) {
            return Err("state slots use different initial ciphertexts".into());
        }
        initial = Some(input);
        // Follow the actual selected two-block encoding, matching RootInputs' producer.
        if !matches!(vr.node(input.node).map(|n| n.kind()), Some(NodeKind::Slice { .. })) {
            return Err("state source is not initial encoding slice".into());
        }
        let encoded = args(vr, input)?[0];
        let encoded_args = args(vr, encoded)?;
        let gaussian = vr.node(encoded_args[1].node).ok_or("missing initial Gaussian")?;
        let NodeKind::GaussianSample { max_coefficient_bound, .. } = gaussian.kind() else {
            return Err("initial encoding lacks Gaussian".into());
        };
        let cutoff = max_coefficient_bound
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic initial cutoff")?;
        if cutoff.to_biguint().as_ref() != Some(&snapshot.initial_state_bound) {
            return Err("initial report bound differs from actual Gaussian cutoff".into());
        }
        let ideal_args = args(vr, encoded_args[0])?;
        let product_args = args(vr, ideal_args[0])?;
        let public_args = args(vr, product_args[1])?;
        let tensor_args = args(vr, ideal_args[1])?;
        let message = vv(args(vr, tensor_args[0])?[1])?;
        let secret = vv(product_args[0])?;
        let input_type = ver.scopes[&FrozenGraphScopeId::Root].wire_types[&input]
            .matrix_type()
            .ok_or("initial ciphertext is not a matrix")?;
        let q = &input_type.modulus;
        let n = input_type.ring_dimension;
        let payload_args = args(vr, tensor_args[1])?;
        let gadget = payload_args[1];
        let decomposition_target = args(pr, pcache)?[0];
        let pre_gadget = args(pr, decomposition_target)?[1];
        let NodeKind::GadgetDecompose { base, digit_count, .. } =
            pr.node(pcache.node).ok_or("missing cache node")?.kind()
        else {
            return Err("state cache not decomposition".into());
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| "dynamic cache base")?;
        let digits =
            digit_count.evaluate(&ParamEnv::default()).map_err(|_| "dynamic cache digits")?;
        let p_d = pv(pcache)?;
        let v_d = vv(vcache)?;
        let p_g = pv(pre_gadget)?;
        let c = vv(input)?;
        let a = vv(public_args[1])?;
        let payload = vv(payload_args[0])?;
        let g = vv(gadget)?;
        let public = pv(public_input)?;
        let scaled_public =
            format!("({}) (⟨{slot}, by decide⟩ : Fin {count})", pv(public_product)?);
        for (name, expression) in [
            (format!("stateScaledC_{slot}"), vv(actual)?),
            (format!("stateScaledPublic_{slot}"), scaled_public.clone()),
        ] {
            let definition_locals = [
                ("pw", "preprocessingWitness runs"),
                ("vw", "verificationWitness runs"),
                ("pin", "preprocessingInputs runs.2.1"),
                ("vin", "verificationInputs runs.2.2.1"),
            ]
            .into_iter()
            .filter(|(name, _)| expression.contains(&format!("({name}).")))
            .map(|(name, definition)| format!("  let {name} := {definition}\n"))
            .collect::<String>();
            source.push_str(&format!(
                "noncomputable def {name} {binders} :=\n{definition_locals}  {expression}\n\n"
            ));
        }
        let public_input_binding = if public.contains("(pin).") || p_d.contains("(pin).") {
            "let pin := preprocessingInputs runs.2.1; "
        } else {
            ""
        };
        source.push_str(&format!("theorem state_public_{slot} {binders} :\n    stateScaledPublic_{slot} runs =\n      (let pw := preprocessingWitness runs; {public_input_binding}({public}) * ({p_d})) := by\n{locals}  have pf := preprocessingFacts runs\n  have product := {} (⟨{slot}, by decide⟩ : Fin {count})\n  dsimp only [{public_relation}, {public_relation}.constraints_0] at product\n  have packEquation : ({}) (⟨{slot}, by decide⟩ : Fin {count}) = ({p_d}) := by\n    obtain ⟨position, indexEquation, equation⟩ := ({}).2.2 (⟨{slot}, by decide⟩ : Fin {count})\n    have same : position = ⟨{slot}, by change {slot} < {count}; decide⟩ := Fin.ext indexEquation\n    simpa only [same] using equation\n  change ({scaled_public}) = ({public}) * (({}) (⟨{slot}, by decide⟩ : Fin {count})) at product\n  rw [packEquation] at product\n  exact product\n\n", fact(ps, public_product, "pf")?, pv(packed)?, fact(ps, packed, "pf")?, pv(packed)?));
        source.push_str(&format!("theorem state_source_{slot} {binders} :\n    Approx (stateScaledC_{slot} runs)\n      (ExponentGeneratedReadoutProof.mask runs * stateScaledPublic_{slot} runs -\n        ExponentGeneratedRefreshState.stateScale_{slot} •\n          (ExponentGeneratedReadoutProof.payload runs * ExponentGeneratedReadoutProof.gadget runs))\n      (ExponentGeneratedRefreshState.stateGain_{slot} * initialBound) := by\n{locals}  have pf := preprocessingFacts runs\n  have vf := verificationFacts runs\n  have cacheLink : ({v_d}) = ({p_d}) := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n  have publicLink : ({a}) = ({public}) := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n  have payloadLink : ({payload}) = ExponentGeneratedReadoutProof.payload runs := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n  have initialRun := ExponentGeneratedInputsRoot.initial_program_state_0 vw vf\n  have close : Approx ({c})\n      (ExponentGeneratedReadoutProof.mask runs * ({public}) -\n        ExponentGeneratedReadoutProof.payload runs * ExponentGeneratedReadoutProof.gadget runs) initialBound := by\n    have initialClose := initialRun.sound (by decide)\n    have messageEquation : ({message}) 0 0 = (1 : ExactPoly {q} {n}) := by\n      norm_num [MxxRuntime.matrixPolynomial]\n    change Approx ({c}) (({secret}) * ({a}) - (({message}) 0 0) • (({payload}) * ({g}))) initialBound at initialClose\n    rw [publicLink, payloadLink, messageEquation, one_smul] at initialClose\n    exact initialClose\n  obtain ⟨action, reduction, bounded, reconstruction, gadgetRuns⟩ :=\n    ExponentGeneratedRefreshState.state_cache_{slot} pw pf\n  have gadgetEquation : ({p_g}) = ExponentGeneratedReadoutProof.gadget runs :=\n    gadget_runs_deterministic gadgetRuns (show MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {base} {digits} ({g}) from {})\n  have cache : ExponentGeneratedReadoutProof.gadget runs * ({p_d}) =\n      ExponentGeneratedRefreshState.stateScale_{slot} • ExponentGeneratedReadoutProof.gadget runs := by\n    rw [gadgetEquation] at reconstruction\n    exact reconstruction\n  have result := state_transfer (by decide) close reduction bounded cache\n  have actualEquation : stateScaledC_{slot} runs = ({c}) * ({p_d}) := by\n    change ({c}) * ({v_d}) = _\n    rw [cacheLink]\n  rw [← actualEquation] at result\n  rw [state_public_{slot}]\n  exact result\n\n", fact(vs, gadget, "vf")?));
    }
    source.push_str("end ExponentGeneratedRefreshStateSource\n");
    Ok(source)
}
