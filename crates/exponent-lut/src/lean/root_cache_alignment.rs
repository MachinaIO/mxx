//! Public-cache flattening indices derived from actual whole-body operands.
use super::{
    root_inputs::value_expression,
    root_program_setup::{parallel_operand, program_cache_binding},
};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::NodeKind,
    types::{ConcreteWireType, WireRef},
};

pub(super) fn root_cache_alignment(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let body = shape.body_relation.as_ref().ok_or("missing cache producer body")?;
    let record = shape.witness_record.as_ref().ok_or("missing cache producer record")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut source = String::from(
        "import RootProgramSetup\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedCacheAlignment\nopen Mxx.Primitives\n\n",
    );
    let mut count = 0;
    for (name, output) in &artifact.root.outputs {
        if !name.contains("-program-cache-family-") || !name.ends_with("-body-1-k-rhs") {
            continue;
        }
        let (scope_id, decomposition, left, right) = program_cache_binding(graph, output.wire)?;
        let scope = &graph.source.scopes()[&scope_id];
        let NodeKind::GadgetDecompose { base, digit_count, .. } =
            scope.node(decomposition.node).ok_or("missing aligned decomposition")?.kind()
        else {
            unreachable!()
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|e| e.to_string())?;
        let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|e| e.to_string())?;
        let left = parallel_operand(graph, artifact, output.wire, left, "flat")?;
        let right = parallel_operand(graph, artifact, output.wire, right, "flat")?;
        let ConcreteWireType::IndexedFamily { count: members, .. } = &types[&right.0] else {
            return Err("shared selector source is not a family".into())
        };
        let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
            return Err("public cache is not a family".into())
        };
        if *members == 0 || lanes % members != 0 {
            return Err("public cache does not have complete member segments".into());
        }
        let width = lanes / members;
        let expected_private = name.replace("-body-1-k-rhs", "-body-0-k-rhs");
        if artifact.root.outputs.get(&expected_private).is_none_or(|output| output.wire != right.0)
        {
            return Err("public cache does not use its paired selector cache".into());
        }
        let original_public = value(left.0)?;
        let original_private = value(right.0)?;
        let public_is_family = matches!(&types[&left.0], ConcreteWireType::IndexedFamily { .. });
        let public = if public_is_family {
            format!("({original_public}) lane")
        } else {
            format!("({original_public})")
        };
        let matrix = element.matrix_type().ok_or("public cache element is not a matrix")?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let cached = value(output.wire)?;
        source.push_str(&format!("-- Actual flattened public cache {name:?}.\ntheorem public_cache_aligned_{count} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedPrivateRoot.RootInput}} {{outputs : ExponentGeneratedPrivateRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget)\n    (lane : Fin {width}) (member : Fin {members}) :\n    gadget * (({cached}) ⟨lane.val * {members} + member.val, by omega⟩) =\n      {public} * ({original_private}) member := by\n  let flat : Fin {lanes} := ⟨lane.val * {members} + member.val, by omega⟩\n",params=artifact.root.parameter_type));
        let mut equations = Vec::new();
        for (side, binding, position, expected) in [
            ("public", &left, "lane", public),
            ("private", &right, "member", format!("({original_private}) member")),
        ] {
            if let Some((_, index)) = &binding.2 {
                source.push_str(&format!("  have {side}_index : Int.toNat ({index}) = {position}.val := by\n    dsimp [flat]\n    omega\n  have {side}_operand : ({actual}) = {expected} := by\n    congr 1\n    apply Fin.ext\n    exact {side}_index\n",actual=binding.3));
                equations.push(format!("{side}_operand"));
            }
        }
        source.push_str(&format!("  have reconstruction := ExponentGeneratedProgramSetup.public_cache_{count} witness facts gadget gadgetRuns flat\n  simpa only [{}] using reconstruction\n\n",equations.join(", ")));
        count += 1;
    }
    if count == 0 {
        return Err("missing public cache alignment".into());
    }
    source.push_str("end ExponentGeneratedCacheAlignment\n");
    Ok(source)
}
