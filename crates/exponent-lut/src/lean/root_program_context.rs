//! Public branch inputs linked to the actual outer execution's sampled family.
use super::{
    root_inputs::value_expression,
    root_program_setup::{parallel_arguments, parallel_operand},
    root_readout_public::root_fact,
};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{ConstantMatrix, NodeKind},
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
    let renamed = |text: String| -> Result<String, String> {
        value_expression(
            "expression",
            &std::collections::BTreeMap::from([
                ("expression".into(), text),
                ("witness".into(), "rootWitness".into()),
                ("inputs".into(), "rootInputs".into()),
                ("facts".into(), "rootFacts".into()),
            ]),
        )
    };
    let fact = |wire| -> Result<String, String> {
        Ok(root_fact(artifact, wire)?
            .replace("(inputs := inputs)", "(inputs := rootInputs)")
            .replace("(outputs := outputs)", "(outputs := rootOutputs)")
            .replace("(witness := witness) facts", "(witness := rootWitness) rootFacts"))
    };
    let gadget_fact = fact(gadget_wire)?;
    let mut source = String::from(
        "import RootProgramInputs\nimport RootProgram\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramContext\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\n",
    );
    for stage in stages {
        let (cipher_source, _, cipher_lookup, actual) =
            parallel_operand(graph, artifact, stage.output, stage.public_ciphertext, "index")?;
        let (message_source, _, message_lookup, messages) =
            parallel_operand(graph, artifact, stage.output, stage.public_message, "index")?;
        if cipher_source != sampler_wire || message_source != global_messages {
            return Err("stage public operands do not use actual sampled inputs".into());
        }
        let (cipher_get, indices) =
            cipher_lookup.ok_or("stage ciphertext has no indexed lookup")?;
        let (message_get, message_indices) =
            message_lookup.ok_or("stage message has no indexed lookup")?;
        if indices != message_indices {
            return Err("stage ciphertext/message indices differ".into());
        }
        let ConcreteWireType::IndexedFamily { count, .. } = &root_types[&stage.output] else {
            return Err("stage output is not a family".into());
        };
        let id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: stage.output.node,
        };
        let shape = &artifact.scope_proofs[&id];
        let (slot, inputs, outputs) = parallel_arguments(graph, artifact, stage.output, "index")?;
        let inputs = renamed(inputs)?;
        let outputs = renamed(outputs)?;
        let actual = renamed(actual)?;
        let messages = renamed(messages)?;
        let indices = renamed(indices)?;
        let context = format!(
            "(backend := backend) (hashModel := hashModel) (params := params) (i_{slot} := index.val) (inputs := {inputs}) (outputs := {outputs}) (witness := member)"
        );
        let get_fact = |wire: WireRef| -> Result<String, String> {
            let fact = shape
                .node_facts
                .get(&wire.node)
                .and_then(|facts| facts.last())
                .ok_or("missing public operand lookup fact")?;
            Ok(format!("{fact} {context} memberFacts"))
        };
        let loop_fact = fact(stage.output)?;
        source.push_str(&format!("theorem public_branches_{} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{rootInputs : ExponentGeneratedInputsRoot.RootInput}} {{rootOutputs : ExponentGeneratedInputsRoot.RootOutput}}\n    (rootWitness : {root_record}) (rootFacts : {root_body} backend hashModel params rootInputs rootOutputs rootWitness)\n    (index : Fin {count}) :\n    ∃ position : Fin {public_count}, ∃ state : ProgramState {q} {n} 2 {columns},\n      ProgramDerivation ({gadget}) state {cutoff} ∧\n      state.ciphertext = ({actual}) ∧ state.message = ({messages}) 0 0 ∧\n      state.mask = ({secret}) ∧ state.payload = ({secret}) ∧\n      state.publicKey = ({gadget}) * (({global_action}) position) ∧ (position.val : Int) = ({indices}) := by\n  obtain ⟨member, memberFacts⟩ := ({} backend hashModel params index.val {inputs} {outputs}).mp (({loop_fact}) index)\n  have cipherLookup := {}\n  have messageLookup := {}\n  obtain ⟨position, hindex, hcipher⟩ := cipherLookup\n  obtain ⟨other, hother, hmessage⟩ := messageLookup\n  have samePosition : other = position := by\n    apply Fin.ext\n    exact_mod_cast hother.trans hindex.symm\n  subst other\n  obtain ⟨localG, state, hg, derived, hc, hkey, hm, hmask, hpayload⟩ := ExponentGeneratedProgramInputs.public_inputs_0 rootWitness rootFacts position\n  have sameG : localG = ({gadget}) := gadget_runs_deterministic hg ({gadget_fact})\n  subst localG\n  have indexEquation : position.val = ({indices}).toNat := by omega\n  refine ⟨position, state, derived, ?_, ?_, hmask, hpayload, hkey, hindex⟩\n  · exact hc.trans (congrArg (fun i : Fin {public_count} => ({}) i) (by apply Fin.ext; exact indexEquation))\n  · exact hm.trans (congrArg (fun i : Fin {public_count} => ({}) i 0 0) (by apply Fin.ext; exact indexEquation))\n\n", stage.index, artifact.root.parameter_type, shape.runs_iff_body, get_fact(cipher_get)?, get_fact(message_get)?, root_value(sampler_wire)?, root_value(global_messages)?));
    }
    source.push_str("end ExponentGeneratedProgramContext\n");
    Ok(source)
}
