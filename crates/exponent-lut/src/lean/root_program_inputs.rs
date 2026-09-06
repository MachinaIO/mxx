//! Native public input derivations from the fixed online sampling execution.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{LoopInputMode, NodeKind},
    types::{ConcreteWireType, Port, WireRef},
};

pub(super) fn root_program_inputs(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing online body")?;
    let record = shape.witness_record.as_ref().ok_or("missing online witness")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut source = String::from(
        "import RootInputs\nimport EncodingInputScopes\nimport ExponentProgram\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramInputs\nopen Mxx.Primitives MxxExponentLut\n\n",
    );
    let mut count = 0;
    for (id, child) in graph.source.scopes() {
        let FrozenGraphScopeId::ParallelBody { parent, owner } = id else { continue };
        if !matches!(parent.as_ref(), FrozenGraphScopeId::Root) ||
            child.inputs().len() != 3 ||
            child.nodes().len() != 11 ||
            child.outputs().len() != 1
        {
            continue;
        }
        let wire = WireRef { node: *owner, port: Port(0) };
        let parent_node = root.node(*owner).ok_or("missing input parent")?;
        let NodeKind::ParallelLoop(loop_) = parent_node.kind() else {
            return Err("wrong input owner".into())
        };
        let args = root.arguments(parent_node).ok_or("missing input arguments")?;
        let types = &graph.scopes[id].wire_types;
        let output = types[&child.outputs()[0]].matrix_type().ok_or("wrong input output")?;
        let q = &output.modulus;
        let n = output.ring_dimension;
        let columns = output.columns;
        let mut message = None;
        let mut secret = None;
        let mut action = None;
        for (slot, input) in child.inputs().iter().enumerate() {
            let m = types[input].matrix_type().ok_or("wrong encoding input type")?;
            let expression = value(args[slot])?;
            if m.rows == 1 && m.columns == 1 && loop_.input_modes[slot] == LoopInputMode::Zip {
                message = Some(expression);
            } else if m.rows == 1 &&
                m.columns == 2 &&
                loop_.input_modes[slot] == LoopInputMode::Broadcast
            {
                secret = Some(expression);
            } else if m.rows == columns &&
                m.columns == columns &&
                loop_.input_modes[slot] == LoopInputMode::Zip
            {
                action = Some(expression);
            }
        }
        let message = message.ok_or("missing actual public message")?;
        let secret = secret.ok_or("missing actual public secret")?;
        let action = action.ok_or("missing actual public action")?;
        let base = child
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::ConstantMatrix {
                    value: mxx_ir_core::node::ConstantMatrix::Gadget { base, .. },
                    ..
                } => base.evaluate(&ParamEnv::default()).ok(),
                _ => None,
            })
            .ok_or("missing public gadget")?;
        let digits = columns / 2;
        let cutoff = child
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                    max_coefficient_bound.evaluate(&ParamEnv::default()).ok()
                }
                _ => None,
            })
            .ok_or("missing public sample cutoff")?;
        let ConcreteWireType::IndexedFamily { count: lanes, .. } =
            &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&wire]
        else {
            return Err("wrong public family".into())
        };
        let actual = value(wire)?;
        let position =
            shape.node_constraints[owner].clone().last().ok_or("missing input constraint")?;
        let fact = format!("facts{}.1", ".2".repeat(position));
        source.push_str(&format!("theorem public_inputs_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedInputsRoot.RootInput}} {{outputs : ExponentGeneratedInputsRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∀ i : Fin {lanes}, ∃ gadget : ExactMatrix {q} {n} 2 {columns},\n      ∃ state : ProgramState {q} {n} 2 {columns},\n      MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget ∧\n      ProgramDerivation gadget state {cutoff} ∧\n      state.ciphertext = ({actual}) i ∧\n      state.publicKey = gadget * (({action}) i) ∧\n      state.message = ({message}) i 0 0 ∧\n      state.mask = ({secret}) ∧ state.payload = ({secret}) := by\n  intro i\n  obtain ⟨gadget,error,hg,equation,bounded⟩ := ExponentGeneratedInputs.encoding_input_{count} ({fact} i)\n  let state : ProgramState {q} {n} 2 {columns} := ⟨({actual}) i, gadget * (({action}) i), ({message}) i 0 0, ({secret}), ({secret})⟩\n  exact ⟨gadget,state,hg,ProgramDerivation.input state error {cutoff} equation bounded,rfl,rfl,rfl,rfl,rfl⟩\n\n",params=artifact.root.parameter_type));
        count += 1;
    }
    if count != 1 {
        return Err("ambiguous native input theorem numbering".into());
    }
    source.push_str("end ExponentGeneratedProgramInputs\n");
    Ok(source)
}
