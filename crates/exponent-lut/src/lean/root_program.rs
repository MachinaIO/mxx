//! Program-family transfer proofs tied to a fixed online root execution witness.
use super::{
    root_inputs::value_expression,
    root_program_setup::{parallel_arguments, parallel_operand},
    root_readout_public::root_fact,
};
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::LeanArtifact,
    node::{MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

pub(super) struct ProgramStage {
    pub index: usize,
    pub private_name: String,
    pub public_name: String,
    pub members: usize,
    pub lane_count: usize,
    pub current: WireRef,
    pub private_cache: WireRef,
    pub public_cache: WireRef,
    pub public_ciphertext: WireRef,
    pub public_message: WireRef,
    pub output: WireRef,
}

pub(super) fn root_program(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<(String, Vec<ProgramStage>), String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let record = shape.witness_record.as_ref().ok_or("missing program root witness")?;
    let body = shape.body_relation.as_ref().ok_or("missing program root body")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    fn depends(scope: &mxx_ir_core::graph::GraphScope, wire: WireRef, source: WireRef) -> bool {
        wire == source ||
            scope
                .node(wire.node)
                .and_then(|node| scope.arguments(node))
                .is_some_and(|args| args.iter().any(|arg| depends(scope, *arg, source)))
    }
    let mut source = format!(
        "import {}\nimport ExponentProgram\nimport ExponentRefreshRoutes\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramRoot\nopen Mxx.Primitives MxxExponentLut\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        artifact.module_name, artifact.root.input_type, artifact.root.output_type
    );
    source.push_str(&format!("def inputsOfExecution {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}} {{witness : {record}}}\n    (_facts : {body} backend hashModel params inputs outputs witness) := inputs\n\n",artifact.root.parameter_type));
    let mut stages = Vec::new();
    for (name, cache) in &artifact.root.inputs {
        if !name.contains("-program-cache-family-") || !name.ends_with("-body-0-k-rhs") {
            continue;
        }
        let public_name = name.replace("-body-0-k-rhs", "-body-1-k-rhs");
        let public_cache =
            artifact.root.inputs.get(&public_name).ok_or("missing paired public cache")?;
        let (owner, loop_node) = root
            .nodes()
            .iter()
            .enumerate()
            .find(|(_, node)| {
                matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                    root.arguments(node).is_some_and(|args| {
                        args.contains(&cache.wire) && args.contains(&public_cache.wire)
                    })
            })
            .ok_or("missing actual whole program loop")?;
        let scope_id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: NodeId(owner as u64),
        };
        let scope = &graph.source.scopes()[&scope_id];
        let parents = root.arguments(loop_node).ok_or("missing whole program captures")?;
        let captured = |wire: WireRef| {
            parents
                .iter()
                .position(|arg| *arg == wire)
                .map(|slot| scope.inputs()[slot])
                .ok_or("program cache is not captured")
        };
        let private_input = captured(cache.wire)?;
        let public_input = captured(public_cache.wire)?;
        let operations = scope
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let op = match node.kind() {
                    NodeKind::MatrixMulSmallRhs => MatrixBinaryOp::Multiply,
                    NodeKind::MatrixBinary(op) => *op,
                    _ => return None,
                };
                let args = scope.arguments(node)?;
                (args.len() == 2).then_some((
                    WireRef { node: NodeId(index as u64), port: Port(0) },
                    op,
                    [args[0], args[1]],
                ))
            })
            .collect::<Vec<_>>();
        let (private_output, _, private_args) = operations
            .iter()
            .find(|(_, op, args)| {
                *op == MatrixBinaryOp::Multiply && depends(scope, args[1], private_input)
            })
            .ok_or("missing actual private product")?;
        let (first_output, _, first_args) = operations
            .iter()
            .find(|(_, op, args)| {
                *op == MatrixBinaryOp::Multiply && depends(scope, args[1], public_input)
            })
            .ok_or("missing actual public product")?;
        let (second_output, _, second_args) = operations
            .iter()
            .find(|(_, op, args)| *op == MatrixBinaryOp::Multiply && args[1] == *private_output)
            .ok_or("missing actual message action")?;
        let (sum, _, _) = operations
            .iter()
            .find(|(_, op, args)| {
                *op == MatrixBinaryOp::Add && args == &[*first_output, *second_output]
            })
            .ok_or("missing actual public/private sum")?;
        let port = scope
            .outputs()
            .iter()
            .position(|wire| wire == sum)
            .ok_or("program result is not a loop output")?;
        let output = WireRef { node: NodeId(owner as u64), port: Port(port as u32) };
        let ConcreteWireType::IndexedFamily { count: members, .. } = &cache.wire_type else {
            return Err("private cache is not a member family".into())
        };
        let ConcreteWireType::IndexedFamily { count: lanes, element } =
            &graph.scopes[&FrozenGraphScopeId::Root].wire_types[&output]
        else {
            return Err("program output is not family".into())
        };
        if *members == 0 || lanes % members != 0 {
            return Err("program output does not have complete member segments".into());
        }
        let lane_count = lanes / members;
        let matrix = element.matrix_type().ok_or("program output is not matrix-valued")?;
        if matrix.rows != 1 {
            return Err("program ciphertext has multiple rows".into());
        }
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let count = stages.len();
        let flat = format!("(⟨lane.val * {members} + i.val, by omega⟩ : Fin {lanes})");
        let operand_wires =
            [private_args[0], private_args[1], first_args[0], first_args[1], second_args[0]];
        let operands = operand_wires
            .iter()
            .map(|wire| parallel_operand(graph, artifact, output, *wire, &flat))
            .collect::<Result<Vec<_>, _>>()?;
        let current = &operands[0].3;
        let current_hypothesis = if graph.scopes[&FrozenGraphScopeId::Root].wire_types
            [&operands[0].0]
            .matrix_type()
            .is_some()
        {
            format!("Fin {members} → state.ciphertext = ({current})")
        } else {
            format!("∀ i : Fin {members}, state.ciphertext = ({current})")
        };
        let private_cache = &operands[1].3;
        let public = &operands[2].3;
        let public_cache = &operands[3].3;
        let messages = &operands[4].3;
        let actual = value(output)?;
        let proof_operands = operand_wires
            .iter()
            .map(|wire| parallel_operand(graph, artifact, output, *wire, "position"))
            .collect::<Result<Vec<_>, _>>()?;
        let (index_slot, member_inputs, member_outputs) =
            parallel_arguments(graph, artifact, output, "position")?;
        let child_shape = &artifact.scope_proofs[&scope_id];
        let output_fact = child_shape.output_fact.as_ref().ok_or("missing program output fact")?;
        let context = format!(
            "(backend := backend) (hashModel := hashModel) (params := params) (i_{index_slot} := position.val) (inputs := {member_inputs}) (outputs := {member_outputs}) (witness := member)"
        );
        let projection = if scope.outputs().len() == 1 {
            "result".to_owned()
        } else {
            format!("result{}.1", ".2".repeat(port))
        };
        let mut evaluation = format!(
            "  let position := {flat}\n  obtain ⟨member, memberFacts⟩ := ({} backend hashModel params position.val {member_inputs} {member_outputs}).mp (({}) position)\n  have outputFact := congrArg (fun result => {projection}) ({output_fact} {context} memberFacts)\n",
            child_shape.runs_iff_body,
            root_fact(artifact, output)?
        );
        let mut rewrites = vec![
            "MxxRuntime.matrixMul".to_owned(),
            "MxxRuntime.matrixMulScalarLeft".to_owned(),
            "MxxRuntime.matrixAdd".to_owned(),
        ];
        for (index, (_, _, get, expression)) in proof_operands.iter().enumerate() {
            if let Some((get, index_expression)) = get {
                let get_fact = child_shape.node_facts[&get.node]
                    .last()
                    .ok_or("missing program operand fact")?;
                evaluation.push_str(&format!("  have operand_{index} : member.w_{}_{} = ({expression}) := by\n    have lookup := {get_fact} {context} memberFacts\n    apply route_lookup_eq _ ?_ lookup\n    change ({index_expression}) = (Int.toNat ({index_expression}) : Int)\n    simp only [Int.ofNat_eq_natCast]\n    omega\n",get.node.0,get.port.0));
                rewrites.push(format!("operand_{index}"));
            }
        }
        evaluation.push_str(&format!("  have actualEq : ({actual}) position = ({}) * ({}) + (({}) 0 0) • (({}) * ({})) := by\n    simpa only [{}] using outputFact\n",proof_operands[2].3,proof_operands[3].3,proof_operands[4].3,proof_operands[0].3,proof_operands[1].3,rewrites.join(", ")));
        source.push_str(&format!("-- Actual private/public family pair {name:?}.\ntheorem program_family_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (lane : Fin {lane_count}) (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (state : ProgramState {q} {n} 2 {columns})\n    (publicState : Fin {members} → ProgramState {q} {n} 2 {columns})\n    (privateAction publicAction : Fin {members} → ErrorMatrix {n} {columns} {columns})\n    (rhs : Fin {members} → ExactPoly {q} {n}) (sampledError : Fin {members} → ErrorMatrix {n} 1 {columns})\n    (messageLift : ErrorPoly {n}) (publicLift : Fin {members} → ErrorPoly {n})\n    (bound publicBound privateGain publicGain cutoff : Nat)\n    (previous : ProgramDerivation gadget state bound)\n    (publicPrevious : ∀ i, ProgramDerivation gadget (publicState i) publicBound)\n    (current : {current_hypothesis})\n    (publicCiphertext : ∀ i, (publicState i).ciphertext = ({public}))\n    (publicMessage : ∀ i, (publicState i).message = ({messages}) 0 0)\n    (privateActionEquation : ∀ i, ({private_cache}) = reduceMatrix {q} {n} {columns} {columns} (privateAction i))\n    (publicActionEquation : ∀ i, ({public_cache}) = reduceMatrix {q} {n} {columns} {columns} (publicAction i))\n    (cache : ∀ i, (state.payload * gadget) * reduceMatrix {q} {n} {columns} {columns} (privateAction i) = rhs i • (state.payload * gadget) + reduceMatrix {q} {n} 1 {columns} (sampledError i))\n    (lift : state.message = reducePoly {q} {n} messageLift) (monomial : polynomialL1 messageLift ≤ 1)\n    (sampleBound : ∀ i, CoeffBound (sampledError i) cutoff)\n    (privateActionBound : ∀ i, ActionGain (privateAction i) privateGain)\n    (shared : ∀ i, (publicState i).payload = state.mask)\n    (decomposition : ∀ i, gadget * reduceMatrix {q} {n} {columns} {columns} (publicAction i) = state.publicKey * reduceMatrix {q} {n} {columns} {columns} (privateAction i))\n    (publicMessageLift : ∀ i, (publicState i).message = reducePoly {q} {n} (publicLift i))\n    (publicMonomial : ∀ i, polynomialL1 (publicLift i) ≤ 1)\n    (publicActionBound : ∀ i, ActionGain (publicAction i) publicGain) :\n    ∀ i, ∃ result : ProgramState {q} {n} 2 {columns},\n      ProgramDerivation gadget result (privateGain * bound + cutoff + publicGain * publicBound) ∧\n      result.ciphertext = ({actual}) {flat} ∧\n      result.publicKey = (publicState i).publicKey * reduceMatrix {q} {n} {columns} {columns} (publicAction i) ∧\n      result.message = (publicState i).message * (state.message * rhs i) ∧\n      result.mask = (publicState i).mask ∧ result.payload = state.payload := by\n  intro i\n{evaluation}  exact program_private_public_transfer previous (publicPrevious i)\n    (privateAction i) (publicAction i) (rhs i) (sampledError i) messageLift (publicLift i)\n    (cache i) lift monomial (sampleBound i) (privateActionBound i) (shared i)\n    (decomposition i) (publicMessageLift i) (publicMonomial i) (publicActionBound i)\n    (current i) (publicCiphertext i) (publicMessage i)\n    (privateActionEquation i) (publicActionEquation i) actualEq\n\n",params=artifact.root.parameter_type));
        stages.push(ProgramStage {
            index: count,
            private_name: name.clone(),
            public_name,
            members: *members,
            lane_count,
            current: private_args[0],
            private_cache: private_args[1],
            public_cache: first_args[1],
            public_ciphertext: first_args[0],
            public_message: second_args[0],
            output,
        });
    }
    if stages.is_empty() {
        return Err("no native private/public program family pair".into());
    }
    source.push_str("end ExponentGeneratedProgramRoot\n");
    Ok((source, stages))
}
