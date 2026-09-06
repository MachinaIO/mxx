//! Public branch inputs linked to the actual outer execution's sampled family.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{ConstantMatrix, LoopInputMode, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

pub(super) fn root_program_context(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    stages: &[super::root_program::ProgramStage],
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let root_shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let root_body = root_shape.body_relation.as_ref().ok_or("missing online root body")?;
    let root_record = root_shape.witness_record.as_ref().ok_or("missing online root witness")?;
    let mut root_values = root_shape.value_expressions.clone();
    root_values.insert("inputs".into(), "rootInputs".into());
    for (name, _) in &root_shape.witnesses {
        root_values.insert(name.clone(), format!("rootWitness.{name}"));
    }
    for wire in root.inputs() {
        let name = format!("w_{}_{}", wire.node.0, wire.port.0);
        let expression = root_values[&name].replacen("inputs", "rootInputs", 1);
        root_values.insert(name, expression);
    }
    let root_value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &root_values)
    };
    let sampler=graph.source.scopes().iter().find(|(id,s)|matches!(id,FrozenGraphScopeId::ParallelBody{parent,..} if matches!(parent.as_ref(),FrozenGraphScopeId::Root))&&s.nodes().len()==11&&s.inputs().len()==3&&s.outputs().len()==1).ok_or("missing public sampler")?;
    let FrozenGraphScopeId::ParallelBody { owner: sampler_owner, .. } = sampler.0 else {
        unreachable!()
    };
    let sampler_wire = WireRef { node: *sampler_owner, port: Port(0) };
    let sampler_args =
        root.arguments(root.node(*sampler_owner).unwrap()).ok_or("missing sampler operands")?;
    let sampler_types = &graph.scopes[sampler.0].wire_types;
    let output =
        sampler_types[&sampler.1.outputs()[0]].matrix_type().ok_or("bad sampler output")?;
    let q = &output.modulus;
    let n = output.ring_dimension;
    let columns = output.columns;
    let message_slot = sampler
        .1
        .inputs()
        .iter()
        .position(|wire| {
            sampler_types[wire].matrix_type().is_some_and(|m| m.rows == 1 && m.columns == 1)
        })
        .ok_or("missing message slot")?;
    let secret_slot = sampler
        .1
        .inputs()
        .iter()
        .position(|wire| {
            sampler_types[wire].matrix_type().is_some_and(|m| m.rows == 1 && m.columns == 2)
        })
        .ok_or("missing secret slot")?;
    let action_slot = sampler
        .1
        .inputs()
        .iter()
        .position(|wire| {
            sampler_types[wire]
                .matrix_type()
                .is_some_and(|m| m.rows == columns && m.columns == columns)
        })
        .ok_or("missing public action slot")?;
    let global_action = root_value(sampler_args[action_slot])?;
    let global_messages = sampler_args[message_slot];
    let secret = root_value(sampler_args[secret_slot])?;
    let cutoff = sampler
        .1
        .nodes()
        .iter()
        .find_map(|node| match node.kind() {
            NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                max_coefficient_bound.evaluate(&ParamEnv::default()).ok()
            }
            _ => None,
        })
        .ok_or("missing sampler cutoff")?;
    let root_types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let ConcreteWireType::IndexedFamily { count: public_count, .. } = &root_types[&sampler_wire]
    else {
        return Err("wrong sampled public family".into());
    };
    let gadget_wire = root
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, node)| {
            let wire = WireRef { node: NodeId(i as u64), port: Port(0) };
            if matches!(
                node.kind(),
                NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { .. }, .. }
            ) && root_types[&wire].matrix_type().is_some_and(|m| {
                m.modulus == *q && m.ring_dimension == n && m.rows == 2 && m.columns == columns
            }) {
                Some(wire)
            } else {
                None
            }
        })
        .ok_or("missing common gadget")?;
    let gadget = root_value(gadget_wire)?;
    let gadget_path = ".2".repeat(
        root_shape.node_constraints[&gadget_wire.node]
            .clone()
            .last()
            .ok_or("missing gadget fact")?,
    );
    let (owner, parent) = root
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| {
            matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                root.arguments(node).is_some_and(|args| args.contains(&sampler_wire))
        })
        .ok_or("missing native program owner")?;
    let NodeKind::ParallelLoop(parent_loop) = parent.kind() else { unreachable!() };
    let id = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: NodeId(owner as u64),
    };
    let scope = &graph.source.scopes()[&id];
    let shape = &artifact.scope_proofs[&id];
    let record = shape.witness_record.as_ref().ok_or("missing program record")?;
    let body = shape.body_relation.as_ref().ok_or("missing program body")?;
    let parent_args = root.arguments(parent).unwrap();
    let mut values = shape.value_expressions.clone();
    let mut captures = Vec::new();
    for (slot, input) in scope.inputs().iter().enumerate() {
        if parent_loop.input_modes[slot] != LoopInputMode::Broadcast {
            return Err("native public context expects broadcast captures".into());
        }
        let expression = root_value(parent_args[slot])?;
        values.insert(format!("w_{}_{}", input.node.0, input.port.0), expression.clone());
        captures.push(expression);
    }
    let inputs = format!("({}, ())", captures.join(", "));
    let value =
        |wire: WireRef| value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &values);
    let fact = |wire: WireRef| -> Result<String, String> {
        Ok(format!(
            "facts{}.1",
            ".2".repeat(
                shape.node_constraints[&wire.node].clone().last().ok_or("missing gather fact")?
            )
        ))
    };
    let mut gathers = Vec::new();
    for (index, node) in scope.nodes().iter().enumerate() {
        if !matches!(node.kind(), NodeKind::ParallelLoop(_)) {
            continue;
        }
        let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
        let cid =
            FrozenGraphScopeId::ParallelBody { parent: Box::new(id.clone()), owner: wire.node };
        let child = &graph.source.scopes()[&cid];
        if child.nodes().len() != 3 || child.inputs().len() != 2 || child.outputs().len() != 1 {
            continue;
        }
        let get = child.node(child.outputs()[0].node).unwrap();
        if !matches!(get.kind(), NodeKind::FamilyGetDynamic) {
            continue;
        }
        let args = child.arguments(get).unwrap();
        let parents = scope.arguments(node).unwrap();
        let family = parents
            [child.inputs().iter().position(|w| *w == args[0]).ok_or("bad gather family")?];
        let indices = parents
            [child.inputs().iter().position(|w| *w == args[1]).ok_or("bad gather indices")?];
        gathers.push((wire, family, indices));
    }
    let mut source = String::from(
        "import RootProgramInputs\nimport RootProgram\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramContext\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\n",
    );
    let root_family = |wire: WireRef| {
        scope.inputs().iter().position(|w| *w == wire).map(|position| parent_args[position])
    };
    let mut prefixes = Vec::new();
    for (cipher, family, index) in &gathers {
        if root_family(*family) != Some(sampler_wire) {
            continue;
        }
        let (message, _, _) = gathers
            .iter()
            .find(|(_, f, i)| root_family(*f) == Some(global_messages) && i == index)
            .ok_or("cipher/message gathers have different actual indices")?;
        let ConcreteWireType::IndexedFamily { count: lanes, .. } =
            &graph.scopes[&id].wire_types[cipher]
        else {
            return Err("bad branch family".into())
        };
        let actual = value(*cipher)?;
        let messages = value(*message)?;
        let indices = value(*index)?;
        let cf = fact(*cipher)?;
        let mf = fact(*message)?;
        source.push_str(&format!("theorem public_branches_{node} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{rootInputs : ExponentGeneratedInputsRoot.RootInput}} {{rootOutputs : ExponentGeneratedInputsRoot.RootOutput}}\n    (rootWitness : {root_record}) (rootFacts : {root_body} backend hashModel params rootInputs rootOutputs rootWitness)\n    (i_0 : Nat) (witness : {record}) (outputs : ExponentGeneratedProgramRoot.RootOutput)\n    (facts : {body} backend hashModel params i_0 {inputs} outputs witness) :\n    ∀ i : Fin {lanes}, ∃ position : Fin {public_count}, ∃ state : ProgramState {q} {n} 2 {columns},\n      ProgramDerivation ({gadget}) state {cutoff} ∧\n      state.ciphertext = ({actual}) i ∧ state.message = ({messages}) i 0 0 ∧\n      state.mask = ({secret}) ∧ state.payload = ({secret}) ∧\n      state.publicKey = ({gadget}) * (({global_action}) position) ∧ (position.val : Int) = ({indices}) i := by\n  intro i\n  obtain ⟨cipher,hclo,hchi,hget,hcout⟩ := {cf} i\n  obtain ⟨position,hindex,hcipher⟩ := hget\n  obtain ⟨message,hmlo,hmhi,hmget,hmout⟩ := {mf} i\n  obtain ⟨other,hother,hmessage⟩ := hmget\n  have samePosition : other = position := by\n    apply Fin.ext\n    exact_mod_cast hother.trans hindex.symm\n  subst other\n  obtain ⟨localG,state,hg,derived,hc,hkey,hm,hmask,hpayload⟩ := ExponentGeneratedProgramInputs.public_inputs_0 rootWitness rootFacts position\n  have sameG : localG = ({gadget}) := gadget_runs_deterministic hg rootFacts{gadget_path}.1\n  subst localG\n  refine ⟨position,state,derived,?_,?_,hmask,hpayload,hkey,hindex⟩\n  · exact hc.trans (hcout.trans hcipher).symm\n  · exact hm.trans (congrArg (fun x => x 0 0) (hmout.trans hmessage)).symm\n\n",node=cipher.node.0,params=artifact.root.parameter_type));
        prefixes.push((*cipher, *message, *index, *lanes));
    }
    if prefixes.len() != 1 {
        return Err("ambiguous native public input prefix gather".into());
    }
    let (prefix_cipher, prefix_message, prefix_index, prefix_count) = prefixes[0];
    let global_indices = value(prefix_index)?;
    let mut covered = std::collections::BTreeSet::new();
    for stage in stages {
        let (cipher, family, index) = gathers
            .iter()
            .find(|(wire, _, _)| *wire == stage.public_ciphertext)
            .ok_or("missing actual stage ciphertext gather")?;
        if *family != prefix_cipher {
            return Err("unsupported native ciphertext gather depth".into());
        }
        let (message, _, _) = gathers
            .iter()
            .find(|(wire, f, i)| {
                *wire == stage.public_message && *f == prefix_message && i == index
            })
            .ok_or("stage ciphertext/message gather lineage differs")?;
        let actual = value(*cipher)?;
        let messages = value(*message)?;
        let indices = value(*index)?;
        let cf = fact(*cipher)?;
        let mf = fact(*message)?;
        let lanes = stage.members;
        source.push_str(&format!("theorem public_branches_{node} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{rootInputs : ExponentGeneratedInputsRoot.RootInput}} {{rootOutputs : ExponentGeneratedInputsRoot.RootOutput}}\n    (rootWitness : {root_record}) (rootFacts : {root_body} backend hashModel params rootInputs rootOutputs rootWitness)\n    (i_0 : Nat) (witness : {record}) (outputs : ExponentGeneratedProgramRoot.RootOutput)\n    (facts : {body} backend hashModel params i_0 {inputs} outputs witness) :\n    ∀ i : Fin {lanes}, ∃ position : Fin {public_count}, ∃ localPosition : Fin {prefix_count}, ∃ state : ProgramState {q} {n} 2 {columns},\n      ProgramDerivation ({gadget}) state {cutoff} ∧\n      state.ciphertext = ({actual}) i ∧ state.message = ({messages}) i 0 0 ∧\n      state.mask = ({secret}) ∧ state.payload = ({secret}) ∧\n      state.publicKey = ({gadget}) * (({global_action}) position) ∧\n      (position.val : Int) = ({global_indices}) localPosition ∧ (localPosition.val : Int) = ({indices}) i := by\n  intro i\n  obtain ⟨cipher,hclo,hchi,hget,hcout⟩ := {cf} i\n  obtain ⟨localPosition,hindex,hcipher⟩ := hget\n  obtain ⟨message,hmlo,hmhi,hmget,hmout⟩ := {mf} i\n  obtain ⟨other,hother,hmessage⟩ := hmget\n  have samePosition : other = localPosition := by\n    apply Fin.ext\n    exact_mod_cast hother.trans hindex.symm\n  subst other\n  obtain ⟨position,state,derived,hc,hm,hmask,hpayload,hkey,hglobal⟩ := public_branches_{prefix} rootWitness rootFacts i_0 witness outputs facts localPosition\n  refine ⟨position,localPosition,state,derived,?_,?_,hmask,hpayload,hkey,hglobal,hindex⟩\n  · exact hc.trans (hcout.trans hcipher).symm\n  · exact hm.trans (congrArg (fun x => x 0 0) (hmout.trans hmessage)).symm\n\n",node=cipher.node.0,prefix=prefix_cipher.node.0,params=artifact.root.parameter_type));
        if !covered.insert(*cipher) {
            return Err("duplicate native stage ciphertext gather".into());
        }
    }
    if covered.len() != stages.len() || stages.is_empty() {
        return Err("incomplete native stage public input coverage".into());
    }
    source.push_str("end ExponentGeneratedProgramContext\n");
    Ok(source)
}
