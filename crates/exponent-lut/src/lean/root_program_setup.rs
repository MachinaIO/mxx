//! Actual public-cache reconstruction for native program stages.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{MatrixBinaryOp, NodeKind},
    types::ConcreteWireType,
};

pub(super) fn root_program_setup(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = shape.body_relation.as_ref().ok_or("missing producer body")?;
    let record = shape.witness_record.as_ref().ok_or("missing producer witness")?;
    let value = |wire: mxx_ir_core::types::WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: mxx_ir_core::types::WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing cache fact")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let mut source = String::from(
        "import RootPrivate\nimport ExponentSetup\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedProgramSetup\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\n",
    );
    let mut count = 0;
    for (name, output) in &artifact.root.outputs {
        if !name.contains("-program-cache-family-") || !name.ends_with("-body-1-k-rhs") {
            continue;
        }
        let fail = || format!("unsupported actual public cache {name}");
        let node = root.node(output.wire.node).ok_or_else(fail)?;
        if !matches!(node.kind(), NodeKind::ParallelLoop(_)) {
            return Err(fail());
        }
        let id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: output.wire.node,
        };
        let child = &graph.source.scopes()[&id];
        if child.nodes().len() != 2 || child.inputs().len() != 1 {
            return Err(fail());
        }
        let decomposition = child.node(child.outputs()[0].node).ok_or_else(fail)?;
        let NodeKind::GadgetDecompose { base, digit_count, .. } = decomposition.kind() else {
            return Err(fail())
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let target_wire = root.arguments(node).ok_or_else(fail)?[0];
        let target_node = root.node(target_wire.node).ok_or_else(fail)?;
        if !matches!(target_node.kind(), NodeKind::ParallelLoop(_)) {
            return Err(fail());
        }
        let tid = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: target_wire.node,
        };
        let target = &graph.source.scopes()[&tid];
        if target.nodes().len() != 3 || target.inputs().len() != 2 {
            return Err(fail());
        }
        let product = target.node(target.outputs()[0].node).ok_or_else(fail)?;
        if !matches!(
            product.kind(),
            NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) {
            return Err(fail());
        }
        let args = target.arguments(product).ok_or_else(fail)?;
        let parents = root.arguments(target_node).ok_or_else(fail)?;
        let left =
            value(parents[target.inputs().iter().position(|w| *w == args[0]).ok_or_else(fail)?])?;
        let right =
            value(parents[target.inputs().iter().position(|w| *w == args[1]).ok_or_else(fail)?])?;
        let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
            return Err(fail())
        };
        let matrix = element.matrix_type().ok_or_else(fail)?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let cached = value(output.wire)?;
        let hd = fact(output.wire)?;
        let ht = fact(target_wire)?;
        source.push_str(&format!("-- Actual cache {name:?}.\ntheorem public_cache_{count} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedPrivateRoot.RootInput}} {{outputs : ExponentGeneratedPrivateRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, gadget * (({cached}) i) = (({left}) i) * (({right}) i) := by\n  intro i\n  obtain ⟨decomposition, hd, hout⟩ := {hd} i\n  have reconstruction := gadget_runs_reconstruct (by decide) (by decide) gadgetRuns hd Fact.out\n  rw [hout, reconstruction]\n  exact {ht} i\n\n",params=artifact.root.parameter_type));
        count += 1;
    }
    if count == 0 {
        return Err("missing public program caches".into());
    }
    let (name, output) = artifact
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-setup-public-key-k-rhs-0"))
        .ok_or("missing native public-input cache")?;
    let node = root.node(output.wire.node).ok_or("missing input cache node")?;
    let id = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: output.wire.node,
    };
    let child = &graph.source.scopes()[&id];
    if child.nodes().len() != 2 || child.inputs().len() != 1 {
        return Err("unsupported public-input cache decomposition".into());
    }
    let NodeKind::GadgetDecompose { base, digit_count, .. } =
        child.node(child.outputs()[0].node).ok_or("missing public input decomposition")?.kind()
    else {
        return Err("wrong public-input cache operation".into());
    };
    let base = base.evaluate(&ParamEnv::default()).map_err(|e| e.to_string())?;
    let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|e| e.to_string())?;
    let target = value(root.arguments(node).ok_or("missing input cache target")?[0])?;
    let cached = value(output.wire)?;
    let hd = fact(output.wire)?;
    let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
        return Err("wrong input cache type".into());
    };
    let matrix = element.matrix_type().ok_or("wrong input cache matrix")?;
    let q = &matrix.modulus;
    let n = matrix.ring_dimension;
    let columns = matrix.columns;
    source.push_str(&format!("-- Actual native public-input cache {name:?}.\ntheorem public_inputs_cache {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedPrivateRoot.RootInput}} {{outputs : ExponentGeneratedPrivateRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (gadget : ExactMatrix {q} {n} 2 {columns}) (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ i : Fin {lanes}, gadget * (({cached}) i) = ({target}) i := by\n  intro i\n  obtain ⟨decomposition, hd, hout⟩ := {hd} i\n  rw [hout]\n  exact gadget_runs_reconstruct (by decide) (by decide) gadgetRuns hd Fact.out\n\n",params=artifact.root.parameter_type));
    source.push_str("end ExponentGeneratedProgramSetup\n");
    Ok(source)
}
