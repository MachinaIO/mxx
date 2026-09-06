//! Program-family transfer proofs tied to a fixed online execution witness.

use super::root_inputs::value_expression;
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
    let outer = graph.source.root_scope();
    let first_cache = artifact
        .root
        .inputs
        .iter()
        .find(|(name, _)| {
            name.contains("-program-cache-family-") && name.ends_with("-body-0-k-rhs")
        })
        .ok_or("missing program cache")?
        .1;
    let (owner, parent_node) = outer
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| {
            matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                outer.arguments(node).is_some_and(|args| args.contains(&first_cache.wire))
        })
        .ok_or("missing outer program loop")?;
    let scope_id = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: NodeId(owner as u64),
    };
    let root = &graph.source.scopes()[&scope_id];
    let parent_arguments = outer.arguments(parent_node).unwrap();
    let captured = |wire: WireRef| -> Result<WireRef, String> {
        let position = parent_arguments
            .iter()
            .position(|arg| *arg == wire)
            .ok_or("cache not captured in program loop")?;
        Ok(root.inputs()[position])
    };
    let shape = &artifact.scope_proofs[&scope_id];
    let body = shape.body_relation.as_ref().ok_or("missing program body")?;
    let record = shape.witness_record.as_ref().ok_or("missing program witness")?;
    fn ty(value: &ConcreteWireType) -> Result<String, String> {
        if let Some(matrix) = value.matrix_type() {
            return Ok(format!(
                "Mxx.Primitives.ExactMatrix {} {} {} {}",
                matrix.modulus, matrix.ring_dimension, matrix.rows, matrix.columns
            ));
        }
        match value {
            ConcreteWireType::IndexedFamily { element, count } => {
                Ok(format!("Fin {count} → {}", ty(element)?))
            }
            _ => Err("unsupported program boundary type".into()),
        }
    }
    let types = &graph.scopes[&scope_id].wire_types;
    let input_types =
        root.inputs().iter().map(|wire| ty(&types[wire])).collect::<Result<Vec<_>, _>>()?;
    let input_type = format!(
        "{} × Unit",
        input_types.iter().map(|ty| format!("({ty})")).collect::<Vec<_>>().join(" × ")
    );
    let output_type = ty(&types[&root.outputs()[0]])?;
    fn depends(scope: &mxx_ir_core::graph::GraphScope, wire: WireRef, source: WireRef) -> bool {
        if wire == source {
            return true;
        }
        let Some(node) = scope.node(wire.node) else { return false };
        scope
            .arguments(node)
            .is_some_and(|args| args.iter().any(|arg| depends(scope, *arg, source)))
    }
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut operations = Vec::new();
    for index in 0..root.nodes().len() {
        let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
        let node = root.node(wire.node).unwrap();
        if !matches!(node.kind(), NodeKind::ParallelLoop(_)) {
            continue;
        }
        let id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(scope_id.clone()),
            owner: wire.node,
        };
        let child = &graph.source.scopes()[&id];
        if child.inputs().len() != 2 || child.outputs().len() != 1 || child.nodes().len() != 3 {
            continue;
        }
        let output = child.node(child.outputs()[0].node).unwrap();
        let op = match output.kind() {
            NodeKind::MatrixMulSmallRhs => MatrixBinaryOp::Multiply,
            NodeKind::MatrixBinary(op) => *op,
            _ => continue,
        };
        let args = child.arguments(output).unwrap();
        let parents = root.arguments(node).unwrap();
        let mut actual = Vec::new();
        for arg in args {
            let Some(position) = child.inputs().iter().position(|input| *input == arg) else {
                continue
            };
            actual.push(parents[position]);
        }
        if actual.len() == 2 {
            operations.push((wire, op, [actual[0], actual[1]]));
        }
    }
    let mut source = format!(
        "import {}\nimport ExponentProgram\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedProgramRoot\nopen Mxx.Primitives MxxExponentLut\n\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\n",
        artifact.module_name, input_type, output_type
    );
    source.push_str(&format!("def inputsOfExecution {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{index : Nat}} {{inputs : RootInput}} {{outputs : RootOutput}} {{witness : {record}}}\n    (_facts : {body} backend hashModel params index inputs outputs witness) := inputs\n\n",params=artifact.root.parameter_type));
    let mut count = 0;
    let mut stages = Vec::new();
    for (name, cache) in &artifact.root.inputs {
        if !name.contains("-program-cache-family-") || !name.ends_with("-body-0-k-rhs") {
            continue;
        }
        let public_name = name.replace("-body-0-k-rhs", "-body-1-k-rhs");
        let public_cache =
            artifact.root.inputs.get(&public_name).ok_or("missing paired public cache")?;
        let cache_wire = captured(cache.wire)?;
        let public_cache_wire = captured(public_cache.wire)?;
        let Some((private_output, _, private_args)) = operations.iter().find(|(_, op, args)| {
            *op == MatrixBinaryOp::Multiply && depends(root, args[1], cache_wire)
        }) else {
            return Err(format!("missing actual private family for {name}"));
        };
        let Some((first_output, _, first_args)) = operations.iter().find(|(_, op, args)| {
            *op == MatrixBinaryOp::Multiply && depends(root, args[1], public_cache_wire)
        }) else {
            return Err(format!("missing actual public family for {name}"));
        };
        let Some((second_output, _, second_args)) = operations
            .iter()
            .find(|(_, op, args)| *op == MatrixBinaryOp::Multiply && args[1] == *private_output)
        else {
            return Err(format!("missing actual scalar family for {name}"));
        };
        let Some((output, _, _)) = operations.iter().find(|(_, op, args)| {
            *op == MatrixBinaryOp::Add && args == &[*first_output, *second_output]
        }) else {
            return Err(format!("missing actual sum family for {name}"));
        };
        let ConcreteWireType::IndexedFamily { element, count: lanes } = &types[output] else {
            return Err("program output is not a family".into())
        };
        let matrix = element.matrix_type().ok_or("program output is not matrix-valued")?;
        if matrix.rows != 1 {
            return Err("program ciphertext has multiple rows".into());
        }
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let current = value(private_args[0])?;
        let private = value(*private_output)?;
        let public = value(first_args[0])?;
        let messages = value(second_args[0])?;
        let first = value(*first_output)?;
        let second = value(*second_output)?;
        let actual = value(*output)?;
        let private_cache = value(private_args[1])?;
        let public_cache = value(first_args[1])?;
        let path = |wire: WireRef| -> Result<String, String> {
            Ok(".2".repeat(
                shape.node_constraints[&wire.node]
                    .clone()
                    .last()
                    .ok_or("missing program constraint")?,
            ))
        };
        source.push_str(&format!("-- Actual private/public family pair {name:?}.\ntheorem program_family_{count} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (i_0 : Nat) (witness : {record}) (facts : {body} backend hashModel params i_0 inputs outputs witness)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (state : ProgramState {q} {n} 2 {columns})\n    (publicState : Fin {lanes} → ProgramState {q} {n} 2 {columns})\n    (privateAction publicAction : Fin {lanes} → ErrorMatrix {n} {columns} {columns})\n    (rhs : Fin {lanes} → ExactPoly {q} {n}) (sampledError : Fin {lanes} → ErrorMatrix {n} 1 {columns})\n    (messageLift : ErrorPoly {n}) (publicLift : Fin {lanes} → ErrorPoly {n})\n    (bound publicBound privateGain publicGain cutoff : Nat)\n    (previous : ProgramDerivation gadget state bound)\n    (publicPrevious : ∀ i, ProgramDerivation gadget (publicState i) publicBound)\n    (current : state.ciphertext = ({current}))\n    (publicCiphertext : ∀ i, (publicState i).ciphertext = ({public}) i)\n    (publicMessage : ∀ i, (publicState i).message = ({messages}) i 0 0)\n    (privateActionEquation : ∀ i, ({private_cache}) i = reduceMatrix {q} {n} {columns} {columns} (privateAction i))\n    (publicActionEquation : ∀ i, ({public_cache}) i = reduceMatrix {q} {n} {columns} {columns} (publicAction i))\n    (cache : ∀ i, (state.payload * gadget) * reduceMatrix {q} {n} {columns} {columns} (privateAction i) = rhs i • (state.payload * gadget) + reduceMatrix {q} {n} 1 {columns} (sampledError i))\n    (lift : state.message = reducePoly {q} {n} messageLift) (monomial : polynomialL1 messageLift ≤ 1)\n    (sampleBound : ∀ i, CoeffBound (sampledError i) cutoff)\n    (privateActionBound : ∀ i, ActionGain (privateAction i) privateGain)\n    (shared : ∀ i, (publicState i).payload = state.mask)\n    (decomposition : ∀ i, gadget * reduceMatrix {q} {n} {columns} {columns} (publicAction i) = state.publicKey * reduceMatrix {q} {n} {columns} {columns} (privateAction i))\n    (publicMessageLift : ∀ i, (publicState i).message = reducePoly {q} {n} (publicLift i))\n    (publicMonomial : ∀ i, polynomialL1 (publicLift i) ≤ 1)\n    (publicActionBound : ∀ i, ActionGain (publicAction i) publicGain) :\n    ∀ i, ∃ result : ProgramState {q} {n} 2 {columns},\n      ProgramDerivation gadget result (privateGain * bound + cutoff + publicGain * publicBound) ∧\n      result.ciphertext = ({actual}) i ∧\n      result.publicKey = (publicState i).publicKey * reduceMatrix {q} {n} {columns} {columns} (publicAction i) ∧\n      result.message = (publicState i).message * (state.message * rhs i) ∧\n      result.mask = (publicState i).mask ∧ result.payload = state.payload := by\n  intro i\n  have hp : ({private}) i = ({current}) * ({private_cache}) i := facts{private_path}.1 i\n  have hf : ({first}) i = ({public}) i * ({public_cache}) i := facts{first_path}.1 i\n  have hs : ({second}) i = (({messages}) i 0 0) • ({private}) i := facts{second_path}.1 i\n  have ha : ({actual}) i = ({first}) i + ({second}) i := facts{add_path}.1 i\n  have pd := ProgramDerivation.privateStep previous (privateAction i) (rhs i) (sampledError i) messageLift privateGain cutoff (cache i) lift monomial (sampleBound i) (privateActionBound i)\n  have ud := ProgramDerivation.publicStep pd (publicPrevious i) (publicAction i) (publicLift i) publicGain (shared i) (decomposition i) (publicMessageLift i) (publicMonomial i) (publicActionBound i)\n  refine ⟨_, ud, ?_, rfl, rfl, rfl, rfl⟩\n  dsimp [ProgramState.privateOutput, ProgramState.publicOutput]\n  rw [current, publicCiphertext i, publicMessage i, ← privateActionEquation i, ← publicActionEquation i, ha, hf, hs, hp]\n\n",params=artifact.root.parameter_type,private_path=path(*private_output)?,first_path=path(*first_output)?,second_path=path(*second_output)?,add_path=path(*output)?));
        stages.push(ProgramStage {
            index: count,
            private_name: name.clone(),
            public_name,
            members: *lanes,
            current: private_args[0],
            private_cache: private_args[1],
            public_cache: first_args[1],
            public_ciphertext: first_args[0],
            public_message: second_args[0],
            output: *output,
        });
        count += 1;
    }
    if count == 0 {
        return Err("no native private/public program family pair".into());
    }
    source.push_str("end ExponentGeneratedProgramRoot\n");
    Ok((source, stages))
}
