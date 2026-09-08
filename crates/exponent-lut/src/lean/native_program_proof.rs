//! Application proof composing native program stages from the generated execution contract.
use super::{
    root_inputs::value_expression,
    root_program::ProgramStage,
    root_program_keys::ProgramKeyStage,
    root_program_setup::{parallel_arguments, parallel_operand, program_cache_binding},
    root_readout_public::root_fact,
};
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::{LeanArtifact, ScopeProofShape},
    node::NodeKind,
    types::{NodeId, Port, WireRef},
};
use std::collections::BTreeMap;

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
fn renamed(text: String, witness: &str, inputs: &str) -> Result<String, String> {
    value_expression(
        "expression",
        &BTreeMap::from([
            ("expression".into(), text),
            ("witness".into(), witness.into()),
            ("inputs".into(), inputs.into()),
        ]),
    )
}

fn execution_fact(
    artifact: &LeanArtifact,
    wire: WireRef,
    preprocessing: bool,
) -> Result<String, String> {
    let (params, inputs, outputs, witness, facts) = if preprocessing {
        ("preprocessing_params", "pin", "execution.preprocessing", "pw", "pf")
    } else {
        ("verification_params", "vin", "execution.verification", "vw", "vf")
    };
    Ok(root_fact(artifact, wire)?
        .replace("(backend := backend)", "(backend := RefreshBackend.backend)")
        .replace("(params := params)", &format!("(params := {params})"))
        .replace("(inputs := inputs)", &format!("(inputs := {inputs})"))
        .replace("(outputs := outputs)", &format!("(outputs := {outputs})"))
        .replace("(witness := witness) facts", &format!("(witness := {witness}) {facts}")))
}

fn render(template: &str, values: &BTreeMap<&str, String>) -> Result<String, String> {
    let mut result = template.to_owned();
    // Inserted proof blocks may themselves contain parameter placeholders.
    for _ in 0..3 {
        for (name, value) in values {
            result = result.replace(&format!("@{name}@"), value);
        }
    }
    if result.contains('@') {
        return Err("unresolved native proof template parameter".into());
    }
    Ok(result)
}

fn arguments(
    graph: &ValidatedGraph,
    id: &FrozenGraphScopeId,
    wire: WireRef,
) -> Result<Vec<WireRef>, String> {
    let scope = &graph.source.scopes()[id];
    scope
        .node(wire.node)
        .and_then(|node| scope.arguments(node))
        .map(|args| args.to_vec())
        .ok_or("missing native operands".into())
}

fn child(id: &FrozenGraphScopeId, wire: WireRef) -> FrozenGraphScopeId {
    FrozenGraphScopeId::ParallelBody { parent: Box::new(id.clone()), owner: wire.node }
}

fn cutoff(graph: &ValidatedGraph, id: &FrozenGraphScopeId) -> Result<String, String> {
    let values = graph.source.scopes()[id]
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                Some(max_coefficient_bound.evaluate(&mxx_ir_core::ParamEnv::default()))
            }
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "open Gaussian cutoff")?;
    if values.len() != 1 || values[0] < 0.into() {
        return Err("ambiguous native Gaussian cutoff".into());
    }
    Ok(values[0].to_string())
}

/// Render the closed application certificate from the same graph metadata and
/// external contracts used by the claim. The numeric equality is checked by Lean
/// against the existing simulator's program output bound.
pub(super) fn native_program_proof(
    pre: &ValidatedGraph,
    ver: &ValidatedGraph,
    pa: &LeanArtifact,
    va: &LeanArtifact,
    stages: &[ProgramStage],
    keys: &BTreeMap<String, ProgramKeyStage>,
    reductions: &BTreeMap<WireRef, WireRef>,
    layout_index: usize,
    externals: &[mxx_ir_core::lean::claim::ExternalInput],
    snapshot: &crate::bgg::noise::BggNoiseSnapshot,
) -> Result<String, String> {
    use mxx_ir_core::lean::claim::InputContract;
    let root = FrozenGraphScopeId::Root;
    let first = stages.first().ok_or("empty native program")?;
    let outer = ver.source.root_scope();
    let sid = child(&root, first.output);
    let ps = &pa.scope_proofs[&root];
    let vs = &va.scope_proofs[&root];
    let pv = |wire| expression(ps, wire, "pw", "pin");
    let vv = |wire| expression(vs, wire, "vw", "vin");
    let matrix = ver.scopes[&sid].wire_types[&first.current]
        .matrix_type()
        .ok_or("native state is not a matrix")?;
    let n = matrix.ring_dimension;
    let columns = matrix.columns;
    if matrix.rows != 1 || n == 0 {
        return Err("unsupported native ciphertext dimensions".into());
    }
    let layout = va.backend_layouts.get(layout_index).ok_or("missing native backend layout")?;
    if layout.modulus != matrix.modulus ||
        layout.ring_dimension != n ||
        2 * layout.regular_digits != columns
    {
        return Err("native backend layout disagrees with actual state type".into());
    }
    let q = matrix.modulus.to_string();
    let gadget_wire = outer
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, node)| {
            let wire = WireRef { node: NodeId(i as u64), port: Port(0) };
            (matches!(
                node.kind(),
                NodeKind::ConstantMatrix {
                    value: mxx_ir_core::node::ConstantMatrix::Gadget { .. },
                    ..
                }
            ) && ver.scopes[&root].wire_types[&wire].matrix_type().is_some_and(|m| {
                m.modulus == matrix.modulus &&
                    m.ring_dimension == n &&
                    m.rows == 2 &&
                    m.columns == columns
            }))
            .then_some(wire)
        })
        .ok_or("missing native gadget")?;
    // The public-input exporter requires this unique sampler shape too.
    let public_samplers = ver
        .source
        .scopes()
        .iter()
        .filter(|(id, scope)| {
            matches!(id, FrozenGraphScopeId::ParallelBody { parent, .. } if **parent == root) &&
                scope.inputs().len() == 3 &&
                scope.nodes().len() == 11 &&
                scope.outputs().len() == 1
        })
        .collect::<Vec<_>>();
    if public_samplers.len() != 1 {
        return Err("ambiguous native public sampler".into());
    }
    let (public_id, public_scope) = public_samplers[0];
    let FrozenGraphScopeId::ParallelBody { owner: public_owner, .. } = public_id else {
        unreachable!()
    };
    let mask_slot = public_scope
        .inputs()
        .iter()
        .position(|wire| {
            ver.scopes[public_id].wire_types[wire]
                .matrix_type()
                .is_some_and(|m| m.rows == 1 && m.columns == 2)
        })
        .ok_or("missing native mask")?;
    let mask =
        vv(arguments(ver, &root, WireRef { node: *public_owner, port: Port(0) })?[mask_slot])?;
    let public_cutoff = cutoff(ver, public_id)?;
    let initial_samples = outer
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            let wire = WireRef { node: NodeId(i as u64), port: Port(0) };
            match node.kind() {
                NodeKind::GaussianSample { max_coefficient_bound, .. }
                    if ver.scopes[&root].wire_types[&wire].matrix_type().is_some_and(|m| {
                        m.modulus == matrix.modulus &&
                            m.ring_dimension == n &&
                            m.rows == 1 &&
                            m.columns == 2 * columns
                    }) =>
                {
                    Some(max_coefficient_bound)
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    if initial_samples.len() != 1 {
        return Err("ambiguous native initial sampler".into());
    }
    let initial_cutoff = initial_samples[0]
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|_| "open native initial cutoff")?
        .to_string();
    if initial_cutoff != snapshot.initial_state_bound.to_string() ||
        snapshot.inputs.public_family_errors.is_empty() ||
        snapshot.inputs.public_family_errors.values().any(|family| match family {
            crate::bgg::noise::FamilyNoiseState::Uniform { state, .. } => {
                state.bound.to_string() != public_cutoff
            }
            crate::bgg::noise::FamilyNoiseState::Members(states) => {
                states.iter().any(|state| state.bound.to_string() != public_cutoff)
            }
        })
    {
        return Err("native input cutoff differs from noise snapshot".into());
    }
    let selectors = externals
        .iter()
        .enumerate()
        .filter_map(|(i, external)| match &external.contract {
            InputContract::OneHotPolynomialFamily { block_lengths } => Some((i, block_lengths)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let monomials = externals
        .iter()
        .enumerate()
        .filter_map(|(i, external)| match &external.contract {
            InputContract::Family { element, .. } => match element.as_ref() {
                InputContract::SignedMonomial { exponent_stride } => Some((i, *exponent_stride)),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    if selectors.len() != 1 || monomials.len() != 1 {
        return Err("ambiguous native external contracts".into());
    }
    let (selector_index, blocks) = selectors[0];
    let (monomial_index, stride) = monomials[0];
    if stride == 0 ||
        n % stride != 0 ||
        blocks != &snapshot.family_active_counts ||
        blocks.len() != stages.len()
    {
        return Err("native external blocks or stride disagree with metadata".into());
    }
    let contract_path = |i: usize| {
        format!("{}{}", ".2".repeat(i), if i + 1 == externals.len() { "" } else { ".1" })
    };
    let setup = pa
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-setup-public-key-k-rhs-0"))
        .ok_or("missing native public-input setup export")?
        .1;
    let export_path = |index: usize| format!("execution.preprocessing{}.1", ".2".repeat(index));
    let mut values = BTreeMap::from([
        ("q", q),
        ("n", n.to_string()),
        ("columns", columns.to_string()),
        ("layout", layout_index.to_string()),
        ("stride", stride.to_string()),
        ("width", (n / stride).to_string()),
        ("pre_body", ps.body_relation.clone().ok_or("missing preprocessing body")?),
        ("ver_body", vs.body_relation.clone().ok_or("missing verification body")?),
        ("gadget", vv(gadget_wire)?),
        ("gadget_fact", execution_fact(va, gadget_wire, false)?),
        ("mask", mask),
        ("selector", format!("external.input_{selector_index}")),
        ("monomials", format!("external.input_{monomial_index}")),
        ("monomial_fact", format!("valid{}", contract_path(monomial_index))),
        ("setup_export", export_path(setup.tuple_index)),
        ("public_cutoff", public_cutoff),
    ]);
    if reductions.len() != stages.len() {
        return Err("incomplete native reduction coverage".into());
    }
    let mut source = String::from(
        "import ExecutionWitnesses\nimport RootProgramContext\nimport RootPrivate\nimport RootCacheAlignment\nimport RootProgramKeys\nimport RootProgramSetup\nimport RootReduction\nimport ExponentNativeProgram\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedNativeProgramProof\nopen Mxx.Primitives MxxRuntime MxxExponentLut GeneratedClaim ExponentGeneratedExecution\nopen scoped BigOperators\n\n",
    );
    source.push_str(&format!("abbrev nativeBound0 : Nat := {initial_cutoff}\n"));
    let mut previous_key = None;
    let mut previous_output = None;
    for (j, stage) in stages.iter().enumerate() {
        if stage.index != j {
            return Err("native theorem numbering does not match metadata".into());
        }
        let key = keys
            .get(stage.public_name.strip_prefix("artifact:").unwrap_or(&stage.public_name))
            .ok_or("missing native key metadata")?;
        let output = reductions[&stage.output];
        let current = parallel_operand(ver, va, stage.output, stage.current, "(flat i)")?;
        if previous_output.is_some_and(|wire| current.0 != wire) {
            return Err("native stage dependency does not match the actual reduction wire".into());
        }
        if key.member_count != stage.members ||
            blocks[j] != stage.members ||
            key.offset != blocks[..j].iter().sum::<usize>()
        {
            return Err("native selector block does not match actual key gather".into());
        }
        let private = pa
            .root
            .outputs
            .get(stage.private_name.strip_prefix("artifact:").unwrap_or(&stage.private_name))
            .ok_or("missing private export")?;
        let public = pa
            .root
            .outputs
            .get(stage.public_name.strip_prefix("artifact:").unwrap_or(&stage.public_name))
            .ok_or("missing public export")?;
        let sampler_wire = arguments(pre, &root, private.wire)?[0];
        let sampler_id = child(&root, sampler_wire);
        let sampler = &pre.source.scopes()[&sampler_id];
        let payload_slot = sampler
            .inputs()
            .iter()
            .position(|wire| {
                pre.scopes[&sampler_id].wire_types[wire]
                    .matrix_type()
                    .is_some_and(|m| m.rows == 1 && m.columns == 2)
            })
            .ok_or("missing private payload")?;
        let payload_wire = arguments(pre, &root, sampler_wire)?[payload_slot];
        let payload_export = pa
            .root
            .outputs
            .values()
            .find(|value| value.wire == payload_wire)
            .ok_or("missing native payload export")?;
        let private_cutoff = cutoff(pre, &sampler_id)?;
        if private_cutoff != snapshot.helper_error_cutoff.to_string() {
            return Err("native private cutoff differs from noise snapshot".into());
        }
        let (cache_scope, decomposition, left, _) = program_cache_binding(pre, key.public_cache)?;
        let prior_key = parallel_operand(pre, pa, key.public_cache, left, "(flat i)")?.0;
        if previous_key.is_some_and(|wire| wire != prior_key) {
            return Err("native key recurrence does not use prior stage key".into());
        }
        if stage.lane_count != key.lane_count {
            return Err("native online/preprocessing lane counts differ".into());
        }
        let private_operand =
            parallel_operand(ver, va, stage.output, stage.private_cache, "(flat i)")?;
        let online_cache = parallel_operand(ver, va, stage.output, stage.public_cache, "(flat i)")?;
        if va
            .root
            .inputs
            .get(&stage.private_name)
            .is_none_or(|input| input.wire != private_operand.0) ||
            va.root
                .inputs
                .get(&stage.public_name)
                .is_none_or(|input| input.wire != online_cache.0)
        {
            return Err("native online caches differ from exported cache bindings".into());
        }
        let messages = parallel_operand(ver, va, stage.output, stage.public_message, "(flat i)")?;
        let private_expression = renamed(private_operand.3, "vw", "vin")?;
        let cache_expression = renamed(online_cache.3, "vw", "vin")?;
        let message_index = renamed(
            messages.2.ok_or("native message must use actual indexed input")?.1,
            "vw",
            "vin",
        )?;
        let public_is_family = matches!(
            &pre.scopes[&root].wire_types[&prior_key],
            mxx_ir_core::types::ConcreteWireType::IndexedFamily { .. }
        );
        let prior_key_expression =
            if public_is_family { format!("({}) lane", pv(prior_key)?) } else { pv(prior_key)? };
        let cache_shape = &pa.scope_proofs[&cache_scope];
        let (cache_slot, cache_inputs, cache_outputs) =
            parallel_arguments(pre, pa, key.public_cache, "(flat i)")?;
        let cache_inputs = renamed(cache_inputs, "pw", "pin")?;
        let cache_outputs = renamed(cache_outputs, "pw", "pin")?;
        let cache_context = format!(
            "(backend := RefreshBackend.backend) (hashModel := hashModel) (params := preprocessing_params) (i_{cache_slot} := (flat i).val) (inputs := {cache_inputs}) (outputs := {cache_outputs}) (witness := member)"
        );
        let decomp_fact = cache_shape
            .node_facts
            .get(&decomposition.node)
            .and_then(|facts| facts.last())
            .ok_or("missing actual public decomposition fact")?;
        let cache_output_fact =
            cache_shape.output_fact.as_ref().ok_or("missing actual public cache output fact")?;
        let cache_port = if pre.source.scopes()[&cache_scope].outputs().len() == 1 {
            String::new()
        } else {
            format!("{}.1", ".2".repeat(key.public_cache.port.0 as usize))
        };
        let cache_gain = format!(
            "    obtain ⟨member, memberFacts⟩ := ({} RefreshBackend.backend hashModel preprocessing_params (flat i).val {cache_inputs} {cache_outputs}).mp (({}) (flat i))\n    have hd := {decomp_fact} {cache_context} memberFacts\n    have ho := {cache_output_fact} {cache_context} memberFacts\n    have cacheEq := congrArg (fun output => output{cache_port}) ho\n    change (@public_cache@) (flat i) = member.w_{}_{} at cacheEq\n    obtain ⟨action, ha, hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd\n    exact ⟨action, cacheEq.trans ha, hgain⟩\n",
            cache_shape.runs_iff_body,
            execution_fact(pa, key.public_cache, true)?,
            decomposition.node.0,
            decomposition.port.0
        );
        values.extend([
            ("stage", j.to_string()),
            ("next", (j + 1).to_string()),
            ("members", stage.members.to_string()),
            ("lanes", key.lane_count.to_string()),
            ("raw", key.raw_count.to_string()),
            ("global_count", (key.lane_count * key.raw_count).to_string()),
            ("offset", key.offset.to_string()),
            ("output", format!("({}) lane", vv(output)?)),
            ("key", pv(key.output)?),
            ("key_theorem", key.theorem.clone()),
            ("global_keys", pv(key.public_inputs)?),
            ("payload", pv(payload_wire)?),
            ("private_cutoff", private_cutoff.clone()),
            ("private_cache", private_expression),
            ("public_cache", pv(key.public_cache)?),
            ("online_cache", cache_expression),
            ("message_index", message_index),
            ("private_export", export_path(private.tuple_index)),
            ("public_export", export_path(public.tuple_index)),
            ("cache_gain", cache_gain),
            ("previous_key", prior_key_expression),
            (
                "selector_fact",
                format!(
                    "runs.1{}{}{}",
                    contract_path(selector_index),
                    ".2".repeat(j + 1),
                    if j + 1 == blocks.len() { "" } else { ".1" }
                ),
            ),
        ]);
        let mut bridges = String::new();
        let mut bridge = |name: &str, index: usize, value: String| {
            let path = format!("{}.1", ".2".repeat(index));
            bridges.push_str(&format!("  have {name} : execution.preprocessing{path} = {value} :=\n    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output{path}) (preprocessingOutput runs)\n"));
        };
        bridge("pre6", private.tuple_index, pv(private.wire)?);
        bridge("pre7", public.tuple_index, pv(public.wire)?);
        bridge("pre28", setup.tuple_index, pv(setup.wire)?);
        bridge("pre29", payload_export.tuple_index, pv(payload_wire)?);
        if j == 0 {
            let repeated = prior_key;
            let initial_key = pa
                .root
                .outputs
                .values()
                .find(|value| value.wire == repeated)
                .ok_or("missing initial native key export")?;
            bridge("pre30", initial_key.tuple_index, pv(repeated)?);
            values.insert("initial_key_export", export_path(initial_key.tuple_index));
            values.extend([
                ("initial", INITIAL.to_string()),
                ("initial_key", INITIAL_KEY.to_string()),
                ("initial_strided", INITIAL_STRIDED.to_string()),
                ("current_equation", "(fun _ => rfl)".into()),
                ("payload_equation", "pre29".into()),
                ("mask_equation", "rfl".into()),
                (
                    "shared_payload",
                    "simpa only [st0,ProgramDerivation.state] using publicPayload i".into(),
                ),
                (
                    "shared_mask",
                    "simpa only [st0,ProgramDerivation.state] using publicMask i".into(),
                ),
            ]);
        } else {
            values.extend([("initial", format!("  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_{} runs lane\n  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift\n    (by decide : 0 < {n}) (initialStrided.signed (by decide) (by decide))\n", j - 1)), ("initial_key", String::new()), ("initial_strided", String::new()), ("current_equation", "(by intro i; rw [initialCipher]; congr 1; apply Fin.ext; dsimp [flat]; omega)".into()), ("payload_equation", "initialPayload".into()), ("mask_equation", "initialMask".into()), ("shared_payload", "exact (publicPayload i).trans initialMask.symm".into()), ("shared_mask", "exact (publicMask i).trans initialMask.symm".into())]);
        }
        values.insert("bridges", bridges);
        source.push_str(&format!("abbrev nativeBound{} : Nat := {} * (denseRegularGain RefreshBackend.layout{layout_index} {n} 2 * nativeBound{j} + {private_cutoff} + denseRegularGain RefreshBackend.layout{layout_index} {n} 2 * {})\n", j + 1, stage.members, values["public_cutoff"]));
        source.push_str(&render(STAGE, &values)?);
        if j == 0 {
            source.push_str(&render(FIRST_STAGE_APPROX, &values)?);
        }
        previous_key = Some(key.output);
        previous_output = Some(output);
    }
    let expected = snapshot
        .evaluate_pre_refresh()
        .map_err(|error| error.to_string())?
        .program
        .output_state()
        .bound
        .clone();
    values.insert("expected_bound", expected.to_string());
    values.insert("program_output", vv(previous_output.ok_or("missing final native output")?)?);
    values.insert("last", (stages.len() - 1).to_string());
    source.push_str(&format!("abbrev nativeOutputBound : Nat := nativeBound{}\n", stages.len()));
    source.push_str(&render(OUTPUT, &values)?);
    source.push_str("end ExponentGeneratedNativeProgramProof\n");
    for j in 0..stages.len() {
        source.push_str(&format!("#print axioms ExponentGeneratedNativeProgramProof.stage_{j}\n"));
    }
    source.push_str("#print axioms ExponentGeneratedNativeProgramProof.programOutputApprox\n");
    source.push_str("#print axioms ExponentGeneratedNativeProgramProof.first_stage_approx\n");
    Ok(source)
}
const FIRST_STAGE_APPROX: &str = r#"theorem first_stage_approx
    {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin @lanes@) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    ∃ message : ExactPoly @q@ @n@, StridedMonomial @stride@ @width@ message ∧
      Approx (@output@) (@mask@ * ((@key@) lane) - message • (@payload@ * (@gadget@)))
        nativeBound1 := by
  obtain ⟨state, derived, ciphertext, key, message, mask, payload⟩ := stage_0 runs lane
  refine ⟨state.message, message, ?_⟩
  have invariant := derived.sound (by decide : 0 < @n@)
  simpa only [ProgramState.Invariant, ciphertext, key, mask, payload] using invariant

"#;
const STAGE: &str = r#"theorem stage_@stage@ {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin @lanes@) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    ∃ state : ProgramState @q@ @n@ 2 @columns@,
      ProgramDerivation (@gadget@) state nativeBound@next@ ∧
      state.ciphertext = (@output@) ∧ state.publicKey = (@key@) lane ∧
      StridedMonomial @stride@ @width@ state.message ∧ state.mask = @mask@ ∧ state.payload = @payload@ := by
  classical
  let pw := preprocessingWitness runs
  let vw := verificationWitness runs
  let pin := preprocessingInputs runs.2.1
  let vin := verificationInputs runs.2.2.1
  have pf : @pre_body@ RefreshBackend.backend hashModel preprocessing_params pin execution.preprocessing pw := preprocessingFacts runs
  have vf : @ver_body@ RefreshBackend.backend hashModel verification_params vin execution.verification vw := verificationFacts runs
@bridges@  let g := (@gadget@)
  let layout := RefreshBackend.layout@layout@
  let gain := denseRegularGain layout @n@ 2
  have lookup : RefreshBackend.backend.regularLayout @q@ @n@ = some layout := by rfl
  have hg := @gadget_fact@
@initial@  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_@stage@ pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  let flat (i : Fin @members@) : Fin (@lanes@ * @members@) := ⟨lane.val * @members@ + i.val, by clear * - lane i; omega⟩
  have publicFacts := fun i : Fin @members@ => ExponentGeneratedProgramContext.public_branches_@stage@ vw vf (flat i)
  choose positions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex using publicFacts
  have actions : ∀ i : Fin @members@, ∃ action : ErrorMatrix @n@ @columns@ @columns@, (@public_cache@) (flat i) = reduceMatrix @q@ @n@ @columns@ @columns@ action ∧ ActionGain action gain := by
    intro i
@cache_gain@  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin @members@, (@online_cache@) = (@public_cache@) (flat i) := by
    intro i
    have same : (@online_cache@) = (@public_export@) (flat i) := by
      rfl
    exact same.trans (congrFun pre7 (flat i))
@initial_key@  have privateOnline : ∀ i : Fin @members@, (@private_cache@) = reduceMatrix @q@ @n@ @columns@ @columns@ (privateAction i) := by
    intro i
    have same : (@private_cache@) = (@private_export@) i := by
      apply congrArg (@private_export@)
      apply Fin.ext
      dsimp [flat]
      omega
    exact same.trans ((congrFun pre6 i).trans (privateEq i))
  have publicOnline : ∀ i : Fin @members@, (@online_cache@) = reduceMatrix @q@ @n@ @columns@ @columns@ (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix @q@ @n@ @columns@ @columns@ (privateAction i) = (@selector@ ⟨@offset@+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix @q@ @n@ 1 @columns@ (privateError i) := by
    intro i
    have payload : st0.payload = @payload@ := @payload_equation@
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    @shared_payload@
  have decomposition : ∀ i, g * reduceMatrix @q@ @n@ @columns@ @columns@ (publicAction i) = st0.publicKey * reduceMatrix @q@ @n@ @columns@ @columns@ (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_@stage@ pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial @stride@ @width@ (publicState i).message := by
    intro i
    rw [publicMessage i]
    let position : Fin @global_count@ := ⟨Int.toNat (@message_index@), by simp only [Int.ofNat_eq_natCast]; omega⟩
    change StridedMonomial @stride@ @width@ (@monomials@ position 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := @monomial_fact@
    obtain ⟨exponent,he⟩ := monomials position
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < @n@) (by decide : @n@ = @stride@ * @width@)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < @n@) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_@stage@ lane vw vf g st0 publicState privateAction publicAction
    (fun i => @selector@ ⟨@offset@+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound@stage@ @public_cutoff@ gain gain @private_cutoff@
    initial publicDerived @current_equation@ publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; @shared_mask@
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound@stage@ + @private_cutoff@ + gain * @public_cutoff@) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := @selector_fact@
@initial_strided@  have monomial : StridedMonomial @stride@ @width@ combined.message := by
    change StridedMonomial @stride@ @width@ (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => @selector@ ⟨@offset@ + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,@mask_equation@,?_⟩
  · simpa only [gain,layout,nativeBound@next@] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_@stage@ vw vf lane).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    have selectedKeys : ∀ i, (publicState i).publicKey = (@global_keys@) ⟨lane.val * @raw@ + @offset@ + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have globalFact := globalIndex i
      have positionEq : positions i = (⟨lane.val * @raw@ + @offset@ + i.val, by clear * - lane i; omega⟩ : Fin @global_count@) := by
        apply Fin.ext
        dsimp [flat] at globalFact ⊢
        omega
      rw [publicKey i]
      have aligned := ExponentGeneratedProgramSetup.public_inputs_cache pw pf g hg (positions i)
      rw [positionEq] at aligned
      change g * (@setup_export@) (positions i) = _
      rw [pre28, positionEq]
      exact aligned
    simp_rw [selectedKeys]
    exact (@key_theorem@ pw pf lane).symm

  · exact @payload_equation@


"#;

const INITIAL: &str = r#"  have initial := ExponentGeneratedInputsRoot.initial_program_state_0 vw vf
  let st0 := initial.state
  have initialMessage : st0.message = 1 := by simp [st0,ProgramDerivation.state,MxxRuntime.matrixPolynomial]
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < @n@) (show SignedMonomial st0.message from initialMessage ▸ signedMonomial_one (by decide))
"#;

const INITIAL_KEY: &str = r#"  have initialKey : st0.publicKey = (@previous_key@) := by
    change @initial_key_export@ = @previous_key@
    exact pre30
"#;

const INITIAL_STRIDED: &str = r#"  have initialStrided : StridedMonomial @stride@ @width@ st0.message := by
    refine ⟨⟨0, by decide⟩, ?_⟩
    simpa using initialMessage
"#;
const OUTPUT: &str = r#"
theorem native_output_bound : nativeBound@next@ = @expected_bound@ := by decide

/-- The actual final program output satisfies the existing worst-case report. -/
theorem programOutputApprox {hashModel : MxxRuntime.HashModel} {external : ExternalInputs}
    {execution : Execution} (runs : Runs hashModel external execution) (lane : Fin @lanes@) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    ∃ message : ExactPoly @q@ @n@,
      StridedMonomial @stride@ @width@ message ∧
      Approx ((@program_output@) lane) (@mask@ * ((@key@) lane) - message • (@payload@ * @gadget@)) @expected_bound@ := by
  let pw := preprocessingWitness runs
  let vw := verificationWitness runs
  obtain ⟨state, derived, ciphertext, key, message, mask, payload⟩ := stage_@last@ runs lane
  refine ⟨state.message, message, ?_⟩
  have invariant := derived.sound (by decide : 0 < @n@)
  simpa only [ProgramState.Invariant, ciphertext, key, mask, payload, native_output_bound] using invariant

"#;
