//! Application proof composing native program stages from the generated execution contract.
use super::{
    root_inputs::value_expression, root_program::ProgramStage, root_program_keys::ProgramKeyStage,
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
fn projection(shape: &ScopeProofShape, wire: WireRef, facts: &str) -> Result<String, String> {
    let position = shape
        .node_constraints
        .get(&wire.node)
        .and_then(|r| r.clone().last())
        .ok_or("missing native primitive fact")?;
    Ok(format!("{facts}{}.1", ".2".repeat(position)))
}
fn gather(
    graph: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    wire: WireRef,
) -> Result<(WireRef, WireRef), String> {
    let scope = &graph.source.scopes()[scope_id];
    let node = scope.node(wire.node).ok_or("missing native gather")?;
    if !matches!(node.kind(), NodeKind::ParallelLoop(_)) {
        return Err("native gather is not a loop".into());
    }
    let id =
        FrozenGraphScopeId::ParallelBody { parent: Box::new(scope_id.clone()), owner: wire.node };
    let child = &graph.source.scopes()[&id];
    if child.nodes().len() != 3 || child.inputs().len() != 2 || child.outputs().len() != 1 {
        return Err("unsupported native gather shape".into());
    }
    let get = child.node(child.outputs()[0].node).ok_or("missing native getter")?;
    if !matches!(get.kind(), NodeKind::FamilyGetDynamic) {
        return Err("wrong native getter".into());
    }
    let args = child.arguments(get).ok_or("missing getter operands")?;
    let parents = scope.arguments(node).ok_or("missing gather captures")?;
    Ok((
        parents
            [child.inputs().iter().position(|w| *w == args[0]).ok_or("missing gather family")?],
        parents[child.inputs().iter().position(|w| *w == args[1]).ok_or("missing gather index")?],
    ))
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
    layout_index: usize,
    externals: &[mxx_ir_core::lean::claim::ExternalInput],
    snapshot: &crate::noise_refresh::simulation::ExponentLutNoiseSnapshot,
) -> Result<String, String> {
    use mxx_ir_core::{lean::claim::InputContract, types::ConcreteWireType};
    let root = FrozenGraphScopeId::Root;
    let first = stages.first().ok_or("empty native program")?;
    let cache = va.root.inputs.get(&first.private_name).ok_or("missing native cache")?.wire;
    let outer = ver.source.root_scope();
    let (owner, owner_node) = outer
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| {
            matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                outer.arguments(node).is_some_and(|args| args.contains(&cache))
        })
        .ok_or("missing native loop")?;
    let owner_wire = WireRef { node: NodeId(owner as u64), port: Port(0) };
    let sid = child(&root, owner_wire);
    let scope = &ver.source.scopes()[&sid];
    let cs = &va.scope_proofs[&sid];
    let ps = &pa.scope_proofs[&root];
    let vs = &va.scope_proofs[&root];
    let cv = |wire| expression(cs, wire, "cw", "ci");
    let pv = |wire| expression(ps, wire, "pw", "pin");
    let vv = |wire| expression(vs, wire, "vw", "vin");
    let cf = |wire| projection(cs, wire, "cf");
    let pf = |wire| projection(ps, wire, "pf");
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
    let input_tuple = format!(
        "({}, ())",
        outer
            .arguments(owner_node)
            .ok_or("missing native captures")?
            .iter()
            .map(|wire| vv(*wire))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ")
    );
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
        public_cutoff != snapshot.public_input_error_cutoff.to_string()
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
        ("child_inputs", input_tuple),
        ("pre_body", ps.body_relation.clone().ok_or("missing preprocessing body")?),
        ("ver_body", vs.body_relation.clone().ok_or("missing verification body")?),
        ("child_body", cs.body_relation.clone().ok_or("missing program body")?),
        ("gadget", vv(gadget_wire)?),
        ("gadget_fact", projection(vs, gadget_wire, "vf")?),
        ("mask", mask),
        ("selector", format!("external.input_{selector_index}")),
        ("monomials", format!("external.input_{monomial_index}")),
        ("monomial_fact", format!("valid{}", contract_path(monomial_index))),
        ("setup_export", export_path(setup.tuple_index)),
        ("public_cutoff", public_cutoff),
    ]);
    // Trace the actual balanced reductions. The stage dependency is the reduced
    // wire, not a position in the stage metadata vector.
    let mut reductions = BTreeMap::new();
    for (i, node) in scope.nodes().iter().enumerate() {
        if !matches!(node.kind(), NodeKind::FamilyGetStatic { .. }) {
            continue;
        }
        let wire = WireRef { node: NodeId(i as u64), port: Port(0) };
        let mut current = arguments(ver, &sid, wire)?[0];
        loop {
            let Ok(args) = arguments(ver, &sid, current) else { break };
            if args.len() != 2 {
                break;
            }
            let (Ok((left, _)), Ok((right, _))) =
                (gather(ver, &sid, args[0]), gather(ver, &sid, args[1]))
            else {
                break
            };
            if left != right {
                break;
            }
            let types = &ver.scopes[&sid].wire_types;
            let (
                ConcreteWireType::IndexedFamily { count: a, .. },
                ConcreteWireType::IndexedFamily { count: b, .. },
            ) = (&types[&left], &types[&current])
            else {
                break
            };
            if *b != a.div_ceil(2) {
                break;
            }
            current = left;
        }
        if stages.iter().any(|stage| stage.output == current) {
            if reductions.insert(current, (reductions.len(), wire)).is_some() {
                return Err("ambiguous native reduction".into());
            }
        }
    }
    if reductions.len() != stages.len() {
        return Err("incomplete native reduction coverage".into());
    }
    let mut source = String::from(
        "import ExecutionWitnesses\nimport RootProgramContext\nimport RootPrivate\nimport RootCacheAlignment\nimport RootProgramKeys\nimport RootProgramSetup\nimport RootReduction\nimport ExponentNativeProgram\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedNativeProgramProof\nopen Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution\nopen scoped BigOperators\n\n",
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
        let (reduction_index, output) = reductions[&stage.output];
        if reduction_index != stage.index ||
            previous_output.is_some_and(|wire| stage.current != wire)
        {
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
        let gather_wire = arguments(pre, &root, private.wire)?[0];
        let (sampler_wire, _) = gather(pre, &root, gather_wire)?;
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
        let target = arguments(pre, &root, key.public_cache)?[0];
        let left = arguments(pre, &root, target)?[0];
        let (prior_key, _) = gather(pre, &root, left)?;
        if previous_key.is_some_and(|wire| wire != prior_key) {
            return Err("native key recurrence does not use prior stage key".into());
        }
        let (_, cache_indices) = gather(ver, &sid, stage.public_cache)?;
        let (message_prefix, _) = gather(ver, &sid, stage.public_message)?;
        let (public_prefix, local_indices) = gather(ver, &sid, stage.public_ciphertext)?;
        let (_, global_indices) = gather(ver, &sid, public_prefix)?;
        values.extend([
            ("stage", j.to_string()),
            ("next", (j + 1).to_string()),
            ("members", stage.members.to_string()),
            ("lanes", key.lane_count.to_string()),
            ("raw", key.raw_count.to_string()),
            ("global_count", (key.lane_count * key.raw_count).to_string()),
            ("offset", key.offset.to_string()),
            ("public_node", stage.public_ciphertext.node.0.to_string()),
            ("output", cv(output)?),
            ("key", pv(key.output)?),
            ("key_theorem", key.theorem.clone()),
            ("global_keys", pv(key.public_inputs)?),
            ("payload", pv(payload_wire)?),
            ("private_cutoff", private_cutoff.clone()),
            ("private_cache", cv(stage.private_cache)?),
            ("public_cache", pv(key.public_cache)?),
            ("online_cache", cv(stage.public_cache)?),
            ("private_export", export_path(private.tuple_index)),
            ("public_export", export_path(public.tuple_index)),
            ("cache_fact", cf(stage.public_cache)?),
            ("cache_index_fact", cf(cache_indices)?),
            ("cache_index_relation", va.scope_relations[&child(&sid, cache_indices)].clone()),
            ("decomposition_fact", pf(key.public_cache)?),
            ("message_fact", cf(stage.public_message)?),
            ("message_prefix_fact", cf(message_prefix)?),
            ("message_prefix", cv(message_prefix)?),
            ("local_index_fact", cf(local_indices)?),
            ("global_index_fact", cf(global_indices)?),
            ("local_index_relation", va.scope_relations[&child(&sid, local_indices)].clone()),
            ("global_index_relation", va.scope_relations[&child(&sid, global_indices)].clone()),
            ("previous_key", pv(prior_key)?),
            ("previous_key_fact", pf(prior_key)?),
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
            let repeated = arguments(pre, &root, prior_key)?[0];
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
                ("current_equation", "(by rfl)".into()),
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
            values.extend([("initial", format!("  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_{} runs lane\n  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift\n    (by decide : 0 < {n}) (initialStrided.signed (by decide) (by decide))\n", j - 1)), ("initial_key", String::new()), ("initial_strided", String::new()), ("current_equation", "initialCipher".into()), ("payload_equation", "initialPayload".into()), ("mask_equation", "initialMask".into()), ("shared_payload", "exact (publicPayload i).trans initialMask.symm".into()), ("shared_mask", "exact (publicMask i).trans initialMask.symm".into())]);
        }
        values.insert("bridges", bridges);
        source.push_str(&format!("abbrev nativeBound{} : Nat := {} * (denseRegularGain RefreshBackend.layout{layout_index} {n} 2 * nativeBound{j} + {private_cutoff} + denseRegularGain RefreshBackend.layout{layout_index} {n} 2 * {})\n", j + 1, stage.members, values["public_cutoff"]));
        source.push_str(&render(STAGE, &values)?);
        previous_key = Some(key.output);
        previous_output = Some(output);
    }
    if scope.outputs() != [previous_output.ok_or("missing final native output")?] {
        return Err("native stages do not reach actual program output".into());
    }
    let expected =
        snapshot.evaluate_pre_refresh().map_err(|error| error.to_string())?.program.output_bound;
    values.insert("expected_bound", expected.to_string());
    values.insert("program_output", vv(owner_wire)?);
    values.insert("output_path", ".2".repeat(cs.constraint_count));
    values.insert("last", (stages.len() - 1).to_string());
    source.push_str(&format!("abbrev nativeOutputBound : Nat := nativeBound{}\n", stages.len()));
    source.push_str(&render(OUTPUT, &values)?);
    source.push_str("end ExponentGeneratedNativeProgramProof\n");
    for j in 0..stages.len() {
        source.push_str(&format!("#print axioms ExponentGeneratedNativeProgramProof.stage_{j}\n"));
    }
    source.push_str("#print axioms ExponentGeneratedNativeProgramProof.programOutputApprox\n");
    Ok(source)
}
const STAGE: &str = r#"theorem stage_@stage@ {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin @lanes@) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState @q@ @n@ 2 @columns@,
      ProgramDerivation (@gadget@) state nativeBound@next@ ∧
      state.ciphertext = (@output@) ∧ state.publicKey = (@key@) lane ∧
      StridedMonomial @stride@ @width@ state.message ∧ state.mask = @mask@ ∧ state.payload = @payload@ := by
  classical
  let pw := preprocessingWitness runs
  let vw := verificationWitness runs
  let cw := programWitness runs lane
  let pin := preprocessingInputs runs.2.1
  let vin := verificationInputs runs.2.2.1
  let ci : ExponentGeneratedProgramRoot.RootInput := @child_inputs@
  have pf : @pre_body@ RefreshBackend.backend hashModel preprocessing_params pin execution.preprocessing pw := preprocessingFacts runs
  have vf : @ver_body@ RefreshBackend.backend hashModel verification_params vin execution.verification vw := verificationFacts runs
  have cf : @child_body@ RefreshBackend.backend hashModel verification_params lane.val ci _ cw := programFacts runs lane
@bridges@  let g := (@gadget@)
  let layout := RefreshBackend.layout@layout@
  let gain := denseRegularGain layout @n@ 2
  have lookup : RefreshBackend.backend.regularLayout @q@ @n@ = some layout := by rfl
  have hg := @gadget_fact@
@initial@  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_@stage@ pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_@public_node@ vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin @members@) : Fin (@lanes@ * @members@) := ⟨lane.val * @members@ + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin @members@, ∃ action : ErrorMatrix @n@ @columns@ @columns@, (@public_cache@) (flat i) = reduceMatrix @q@ @n@ @columns@ @columns@ action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := @decomposition_fact@ (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin @members@, (@online_cache@) i = (@public_cache@) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := @cache_fact@ i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := @cache_index_fact@ i
    norm_num [@cache_index_relation@] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change @online_cache@ i = (@public_export@) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
@initial_key@  have privateOnline : ∀ i, (@private_cache@) i = reduceMatrix @q@ @n@ @columns@ @columns@ (privateAction i) := by
    intro i
    change (@private_export@) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (@online_cache@) i = reduceMatrix @q@ @n@ @columns@ @columns@ (publicAction i) := by
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
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := @message_fact@ i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := @message_prefix_fact@ position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial @stride@ @width@ (@message_prefix@ position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial @stride@ @width@ (@monomials@ rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := @monomial_fact@
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < @n@) (by decide : @n@ = @stride@ * @width@)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < @n@) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_@stage@ lane.val cw cf g st0 publicState privateAction publicAction
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
    exact (ExponentGeneratedReduction.reduction_@stage@ lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (@global_keys@) ⟨lane.val * @raw@ + @offset@ + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := @local_index_fact@ i
      have gi := @global_index_fact@ (localPositions i)
      norm_num [@local_index_relation@,
        @local_index_relation@.constraints_0] at li
      norm_num [@global_index_relation@,
        @global_index_relation@.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * @raw@ + @offset@ + i.val, by clear * - lane i; omega⟩ : Fin @global_count@) := by
        apply Fin.ext
        dsimp only
        clear * - lane i localFact globalFact li gi
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

const INITIAL_KEY: &str = r#"  have initialKey : st0.publicKey = (@previous_key@) lane := by
    have repeated := @previous_key_fact@ lane
    change @initial_key_export@ = @previous_key@ lane
    exact pre30.trans repeated.symm
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
  let cw := programWitness runs lane
  obtain ⟨state, derived, ciphertext, key, message, mask, payload⟩ := stage_@last@ runs lane
  refine ⟨state.message, message, ?_⟩
  have invariant := derived.sound (by decide : 0 < @n@)
  have output := (programFacts runs lane)@output_path@
  change ((@program_output@) lane) = (@output@) at output
  change Approx ((@program_output@) lane) _ _
  rw [output]
  simpa only [ProgramState.Invariant, ciphertext, key, mask, payload, native_output_bound] using invariant

"#;
