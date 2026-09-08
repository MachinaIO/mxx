//! Whole-execution native readout correctness from the generated claim witnesses.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{LeanArtifact, ScopeProofShape},
    node::{ConstantMatrix, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

fn expression(
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

pub(super) fn root_readout_proof(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    ver: &ValidatedGraph,
    va: &LeanArtifact,
) -> Result<String, String> {
    let root = ver.source.root_scope();
    let preroot = pre.source.root_scope();
    let ps = &pa.scope_proofs[&FrozenGraphScopeId::Root];
    let vs = &va.scope_proofs[&FrozenGraphScopeId::Root];
    let vt = &ver.scopes[&FrozenGraphScopeId::Root].wire_types;
    let pt = &pre.scopes[&FrozenGraphScopeId::Root].wire_types;
    let pv = |wire| expression(ps, wire, "pw", "pin");
    let vv = |wire| expression(vs, wire, "vw", "vin");
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let preboundary = |suffix: &str| {
        pa.root
            .outputs
            .iter()
            .find(|(name, _)| name.ends_with(suffix))
            .map(|(_, boundary)| boundary.wire)
            .ok_or_else(|| format!("missing readout pre boundary {suffix}"))
    };
    let verboundary = |suffix: &str| {
        va.root
            .inputs
            .iter()
            .find(|(name, _)| name.ends_with(suffix))
            .map(|(_, boundary)| boundary.wire)
            .ok_or_else(|| format!("missing readout ver boundary {suffix}"))
    };
    let norm_suffix = "-readout-0.3";
    let target_suffix = "-readout-0.0";
    let switch_suffix = "-readout-0.1";
    let helper_suffix = "-readout-0.2";
    let norm = verboundary(norm_suffix)?;
    let prenorm = preboundary(norm_suffix)?;
    let normalized = root
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, node)| {
            (matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                root.arguments(node).is_some_and(|args| args.contains(&norm)))
            .then_some(WireRef { node: NodeId(i as u64), port: Port(0) })
        })
        .ok_or("missing actual online normalization")?;
    let normalized_args = root
        .arguments(root.node(normalized.node).ok_or("missing normalization node")?)
        .ok_or("missing normalization operands")?;
    let traced =
        *normalized_args.iter().find(|wire| **wire != norm).ok_or("missing traced input")?;
    let traced_args = root
        .arguments(root.node(traced.node).ok_or("missing traced node")?)
        .ok_or("missing traced operands")?;
    let transformed = traced_args[0];
    let transformed_args = root
        .arguments(root.node(transformed.node).ok_or("missing target operation")?)
        .ok_or("missing target operands")?;
    if transformed_args[1] != verboundary(target_suffix)? {
        return Err("target cache role mismatch".into());
    }
    let source_wire = transformed_args[0];
    let ConcreteWireType::IndexedFamily { count: labels, element } = &vt[&normalized] else {
        return Err("readout output is not a family".into());
    };
    let matrix = element.matrix_type().ok_or("readout output is not a matrix family")?;
    let q = &matrix.modulus;
    let n = matrix.ring_dimension;
    let columns = matrix.columns;
    if matrix.rows != 1 {
        return Err("native readout ciphertext must have one row".into());
    }
    let ConcreteWireType::IndexedFamily { count: helpers, .. } = &vt[&verboundary(switch_suffix)?]
    else {
        return Err("readout switch cache is not indexed".into());
    };
    let width = helpers + 1;
    if !width.is_power_of_two() || n % width != 0 {
        return Err("invalid native trace dimensions".into());
    }
    let stride = n / width;

    // Follow the Gaussian encoding's actual left public-key action to its
    // secret. A setup-key automorphism in the same sampler has a different role.
    let mask_candidates = root
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            if !matches!(node.kind(), NodeKind::ParallelLoop(_)) {
                return None;
            }
            let wire = WireRef { node: NodeId(i as u64), port: Port(2) };
            let ConcreteWireType::IndexedFamily { count, element } = vt.get(&wire)? else {
                return None;
            };
            let ty = element.matrix_type()?;
            if count != helpers || ty != matrix {
                return None;
            }
            let scope = ver.source.scopes().get(&child_id(wire))?;
            scope
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
                .then_some(wire)
        })
        .collect::<Vec<_>>();
    if mask_candidates.len() != 1 {
        return Err("ambiguous native readout mask sampler".into());
    }
    let mask_wire = mask_candidates[0];
    let mask_scope = &ver.source.scopes()[&child_id(mask_wire)];
    let args = |wire: WireRef| {
        mask_scope
            .node(wire.node)
            .and_then(|node| mask_scope.arguments(node))
            .ok_or("missing readout mask operation")
    };
    let batch = args(args(mask_scope.outputs()[2])?[0])?;
    let ideal = args(batch[0])?;
    let public = args(ideal[0])?;
    let secret_slot = mask_scope
        .inputs()
        .iter()
        .position(|wire| *wire == public[0])
        .ok_or("native mask secret is not a capture")?;
    let mask_parents = root
        .arguments(root.node(mask_wire.node).ok_or("missing mask sampler")?)
        .ok_or("missing mask captures")?;
    let secret_wire = mask_parents[secret_slot];
    let NodeKind::GaussianSample { max_coefficient_bound, .. } =
        mask_scope.node(batch[1].node).ok_or("missing native mask Gaussian")?.kind()
    else {
        return Err("missing native mask Gaussian".into());
    };
    let cutoff = max_coefficient_bound
        .evaluate(&ParamEnv::default())
        .map_err(|_| "nonconstant native cutoff")?;

    let prehelper = preboundary(helper_suffix)?;
    let helper_parents = preroot
        .arguments(preroot.node(prehelper.node).ok_or("missing pre helper")?)
        .ok_or("missing pre helper operands")?;
    let gathered = preroot
        .arguments(preroot.node(helper_parents[0].node).ok_or("missing target gather")?)
        .ok_or("missing target gather captures")?;
    let pretransformed = gathered
        .iter()
        .copied()
        .find(|wire| {
            matches!(&pt[wire],
        ConcreteWireType::IndexedFamily { element, .. } if element.matrix_type().is_some())
        })
        .ok_or("missing pre transformed public family")?;
    let pretransformed_args = preroot
        .arguments(preroot.node(pretransformed.node).ok_or("missing pre target action")?)
        .ok_or("missing pre target operands")?;
    if pretransformed_args[1] != preboundary(target_suffix)? {
        return Err("pre target role mismatch".into());
    }
    let public_wire = pretransformed_args[0];
    let prenormalized = preroot
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, node)| {
            (matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                preroot.arguments(node).is_some_and(|args| args.contains(&prenorm)))
            .then_some(WireRef { node: NodeId(i as u64), port: Port(0) })
        })
        .ok_or("missing actual pre normalization")?;

    // The setup key is the target in the actual sampled fixed-RHS equation.
    let preswitch = preboundary(switch_suffix)?;
    let sampler = preroot
        .arguments(preroot.node(preswitch.node).ok_or("missing switch decomposition")?)
        .ok_or("missing switch decomposition input")?[0];
    let sample_scope = &pre.source.scopes()[&child_id(sampler)];
    let sample_args = |wire: WireRef| {
        sample_scope
            .node(wire.node)
            .and_then(|node| sample_scope.arguments(node))
            .ok_or("missing switch sample operation")
    };
    let concat = sample_args(sample_scope.outputs()[sampler.port.0 as usize])?;
    let add = sample_args(concat[1])?;
    let subtract = sample_args(add[0])?;
    let target_action = sample_args(subtract[0])?;
    let target_product = sample_args(target_action[1])?;
    let payload_slot = sample_scope
        .inputs()
        .iter()
        .position(|wire| *wire == target_product[0])
        .ok_or("native readout payload is not a capture")?;
    let sampler_parents = preroot
        .arguments(preroot.node(sampler.node).ok_or("missing switch sampler")?)
        .ok_or("missing switch sampler captures")?;
    let payload_wire = sampler_parents[payload_slot];

    let (gadget_wire, base) = root
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, node)| {
            let wire = WireRef { node: NodeId(i as u64), port: Port(0) };
            let NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { base, .. }, .. } =
                node.kind()
            else {
                return None;
            };
            let ty = vt[&wire].matrix_type()?;
            (ty.modulus == *q && ty.ring_dimension == n && ty.rows == 2 && ty.columns == columns)
                .then_some((wire, base))
        })
        .ok_or("missing actual native gadget")?;
    let base = base.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant gadget radix")?;
    let helper_scope = &pre.source.scopes()[&child_id(prehelper)];
    let NodeKind::GadgetDecompose { digit_count, .. } = helper_scope
        .node(helper_scope.outputs()[0].node)
        .ok_or("missing helper decomposition")?
        .kind()
    else {
        return Err("missing helper decomposition".into());
    };
    let digits =
        digit_count.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant gadget digits")?;
    let gadget_position = vs
        .node_constraints
        .get(&gadget_wire.node)
        .and_then(|range| range.clone().last())
        .ok_or("missing actual gadget fact")?;
    let gadget_fact = format!("vf{}.1", ".2".repeat(gadget_position));
    let source_value = vv(source_wire)?;
    let secret = vv(secret_wire)?;
    let public = pv(public_wire)?;
    let payload = pv(payload_wire)?;
    let gadget = vv(gadget_wire)?;
    let normalized_value = vv(normalized)?;
    let normalized_public = pv(prenormalized)?;
    let mut source = String::from(
        "import ExecutionWitnesses\nimport ExponentNativeProgramProof\nimport RootReadoutStage\nimport RootReadoutPublic\nimport RootSeededKeys\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 4000000\nnamespace ExponentGeneratedReadoutProof\nopen Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution\nopen scoped BigOperators\n\n",
    );
    for (name, value) in [
        ("sourceCiphertext", &source_value),
        ("sourcePublicKey", &public),
        ("mask", &secret),
        ("payload", &payload),
        ("gadget", &gadget),
        ("readoutCiphertext", &normalized_value),
        ("readoutPublicKey", &normalized_public),
    ] {
        let bindings = [
            ("pw", "preprocessingWitness runs"),
            ("vw", "verificationWitness runs"),
            ("pin", "preprocessingInputs runs.2.1"),
            ("vin", "verificationInputs runs.2.2.1"),
        ]
        .into_iter()
        .filter(|(name, _)| {
            value.contains(&format!("{name}.")) || value.contains(&format!("({name})."))
        })
        .map(|(name, definition)| format!("  let {name} := {definition}\n"))
        .collect::<String>();
        source.push_str(&format!("noncomputable def {name} {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) :=\n{bindings}  {value}\n\n"));
    }
    source.push_str(&format!(r#"noncomputable def nativeValue (exponent : Fin (2 * {width})) : ExactPoly {q} {n} :=
  if h : exponent.val < {width} then
    (ExponentGeneratedReadoutFunction.targetCoefficients ⟨exponent.val, h⟩ : ExactPoly {q} {n}) -
      ({width} : ExactPoly {q} {n}) * ExponentGeneratedReadoutFunction.normalizationScalar
  else ({width} : ExactPoly {q} {n}) * ExponentGeneratedReadoutFunction.normalizationScalar -
    (ExponentGeneratedReadoutFunction.targetCoefficients ⟨exponent.val - {width}, by omega⟩ : ExactPoly {q} {n})

noncomputable def readoutBound (inputBound : Nat) : Nat :=
  normalizedTraceBound Finset.univ inputBound {cutoff} ({columns} * {n} * (({base} : Int).toNat / 2)) 1
    ({columns} * {n} * (({base} : Int).toNat / 2))
    (∑ k, (2 * ExponentGeneratedReadoutFunction.targetCoefficients k - 1).natAbs)
    (fun _ : Fin {helpers} => {columns} * {n} * (({base} : Int).toNat / 2))
    (fun _ : Fin {helpers} => {columns} * {n} * (({base} : Int).toNat / 2))

/-- Every execution-dependent readout premise comes from this one claim run.
Only the source program's strided-message approximation remains to compose. -/
theorem native_readout {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}
    (runs : Runs hashModel external execution) (label : Fin {labels})
    (exponent : Fin (2 * {width})) (inputBound : Nat)
    (hsource : Approx (sourceCiphertext runs label)
      (mask runs * sourcePublicKey runs label -
        (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val)) •
          (payload runs * gadget runs)) inputBound) :
    Approx (readoutCiphertext runs label)
      (mask runs * readoutPublicKey runs label - nativeValue exponent • (payload runs * gadget runs))
      (readoutBound inputBound) := by
  classical
  let pw := preprocessingWitness runs
  let vw := verificationWitness runs
  let pin := preprocessingInputs runs.2.1
  let vin := verificationInputs runs.2.2.1
  have pf := preprocessingFacts runs
  have vf := verificationFacts runs
  have hg : MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {base} {digits} ({gadget}) := {gadget_fact}
"#));
    for (role, suffix) in [
        ("target", target_suffix),
        ("switch", switch_suffix),
        ("helper", helper_suffix),
        ("normalization", norm_suffix),
    ] {
        let online = vv(verboundary(suffix)?)?;
        let cached = pv(preboundary(suffix)?)?;
        source.push_str(&format!("  have {role}Link : ({online}) = ({cached}) := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n"));
    }
    source.push_str(&format!(r#"  have sourceApprox : Approx (({source_value}) label)
      (({secret}) * (({public}) label) -
        (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val)) •
          (({payload}) * ({gadget}))) inputBound := hsource
  have result := ExponentGeneratedReadoutStage.native_transfer pw pf vw vf
    targetLink switchLink helperLink normalizationLink ({gadget}) hg label exponent inputBound sourceApprox
  have keys := ExponentGeneratedSeededKeys.readout_mask_keys pw pf vw vf (by rfl)
  have publicEquation := ExponentGeneratedReadoutPublic.normalized_public pw pf label
  simp only [keys] at result
  rw [← publicEquation] at result
  exact result

/-- The actual native program and readout compose without an error or cache premise. -/
theorem native_readout_closed {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}}
    {{execution : Execution}} (runs : Runs hashModel external execution) (label : Fin {labels}) :
    ∃ exponent : Fin (2 * {width}),
      Approx (readoutCiphertext runs label)
        (mask runs * readoutPublicKey runs label - nativeValue exponent • (payload runs * gadget runs))
        (readoutBound ExponentGeneratedNativeProgramProof.nativeOutputBound) := by
  obtain ⟨message, ⟨exponent, messageEq⟩, sourceApprox⟩ :=
    ExponentGeneratedNativeProgramProof.programOutputApprox runs label
  rw [messageEq] at sourceApprox
  rw [← ExponentGeneratedNativeProgramProof.native_output_bound] at sourceApprox
  exact ⟨exponent, native_readout runs label exponent
    ExponentGeneratedNativeProgramProof.nativeOutputBound sourceApprox⟩

end ExponentGeneratedReadoutProof
"#));
    Ok(source)
}
