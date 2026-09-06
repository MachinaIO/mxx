//! Public-cache flattening indices derived from the actual producer gathers.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, WireRef},
};

pub(super) fn root_cache_alignment(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let body = shape.body_relation.as_ref().ok_or("missing cache producer body")?;
    let record = shape.witness_record.as_ref().ok_or("missing cache producer record")?;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing cache gather fact")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let gather = |wire: WireRef| -> Result<(WireRef, WireRef), String> {
        let scope =
            graph.source.scopes().get(&child_id(wire)).ok_or("missing cache gather scope")?;
        let node = scope.node(scope.outputs()[0].node).ok_or("missing gather output")?;
        if !matches!(node.kind(), NodeKind::FamilyGetDynamic) {
            return Err("cache operand is not actual gather".into());
        }
        let args = scope.arguments(node).ok_or("missing gather args")?;
        let parents = root
            .arguments(root.node(wire.node).ok_or("missing gather parent")?)
            .ok_or("missing gather captures")?;
        let map = |arg: WireRef| -> Result<WireRef, String> {
            Ok(parents[scope
                .inputs()
                .iter()
                .position(|input| *input == arg)
                .ok_or("gather operand not captured")?])
        };
        Ok((map(args[0])?, map(args[1])?))
    };
    let mut source = String::from(
        "import RootProgramSetup\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedCacheAlignment\nopen Mxx.Primitives\n\nprivate theorem lookup_eq {α : Type} {count : Nat} {f : Fin count → α} {index : Int} {output : α} (position : Fin count) (hindex : index = (position.val : Int)) (h : MxxRuntime.familyGetDynamic f index output) : output = f position := by\n  obtain ⟨actual, ha, hv⟩ := h\n  have he : actual = position := by apply Fin.ext; omega\n  simpa only [he] using hv\n\n",
    );
    let mut count = 0;
    for (name, output) in &artifact.root.outputs {
        if !name.contains("-program-cache-family-") || !name.ends_with("-body-1-k-rhs") {
            continue;
        }
        let fail = || format!("unsupported cache alignment {name}");
        let decomposition = &graph.source.scopes()[&child_id(output.wire)];
        let NodeKind::GadgetDecompose { base, digit_count, .. } =
            decomposition.node(decomposition.outputs()[0].node).ok_or_else(fail)?.kind()
        else {
            return Err(fail());
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|_| fail())?;
        let target_wire =
            root.arguments(root.node(output.wire.node).ok_or_else(fail)?).ok_or_else(fail)?[0];
        let target = &graph.source.scopes()[&child_id(target_wire)];
        let product = target.node(target.outputs()[0].node).ok_or_else(fail)?;
        if !matches!(
            product.kind(),
            NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) {
            return Err(fail());
        }
        let product_args = target.arguments(product).ok_or_else(fail)?;
        let captures =
            root.arguments(root.node(target_wire.node).ok_or_else(fail)?).ok_or_else(fail)?;
        let left = captures
            [target.inputs().iter().position(|w| *w == product_args[0]).ok_or_else(fail)?];
        let right = captures
            [target.inputs().iter().position(|w| *w == product_args[1]).ok_or_else(fail)?];
        let (public, public_indices) = gather(left)?;
        let (private, private_indices) = gather(right)?;
        let ConcreteWireType::IndexedFamily { count: width, .. } = &types[&public] else {
            return Err(fail());
        };
        let ConcreteWireType::IndexedFamily { count: members, .. } = &types[&private] else {
            return Err(fail());
        };
        let ConcreteWireType::IndexedFamily { count: lanes, element } = &output.wire_type else {
            return Err(fail());
        };
        if *members == 0 || width.checked_mul(*members) != Some(*lanes) {
            return Err(fail());
        }
        let expected_private = name.replace("-body-1-k-rhs", "-body-0-k-rhs");
        if artifact.root.outputs.get(&expected_private).is_none_or(|output| output.wire != private)
        {
            return Err("public cache gather does not reference paired private cache".into());
        }
        let matrix = element.matrix_type().ok_or_else(fail)?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let cached = value(output.wire)?;
        let original_public = value(public)?;
        let original_private = value(private)?;
        source.push_str(&format!("-- Actual flattened public cache {name:?}.\ntheorem public_cache_aligned_{count} {{backend : MxxRuntime.BackendContext}} [Fact ((backend.regularLayout {q} {n}).map MxxRuntime.RegularLayout.droppedModuli = some 0)] {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : ExponentGeneratedPrivateRoot.RootInput}} {{outputs : ExponentGeneratedPrivateRoot.RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget)\n    (lane : Fin {width}) (member : Fin {members}) :\n    gadget * (({cached}) ⟨lane.val * {members} + member.val, by omega⟩) =\n      ({original_public}) lane * ({original_private}) member := by\n  let flat : Fin {lanes} := ⟨lane.val * {members} + member.val, by omega⟩\n",params=artifact.root.parameter_type));
        for (side, indices, gather_wire, position) in [
            ("public", public_indices, left, "lane"),
            ("private", private_indices, right, "member"),
        ] {
            let ir = &artifact.scope_relations[&child_id(indices)];
            let gr = &artifact.scope_relations[&child_id(gather_wire)];
            let original = if side == "public" { &original_public } else { &original_private };
            source.push_str(&format!("  have {side}_index : ({index_value}) flat = ({position}.val : Int) := by\n    have h := {index_fact} flat\n    norm_num [{ir}, {ir}.constraints_0] at h\n    rw [h]\n    dsimp only [flat]\n    omega\n  have {side}_get : ({gather_value}) flat = ({original}) {position} := by\n    have h := {gather_fact} flat\n    dsimp only [{gr}, {gr}.constraints_0] at h\n    obtain ⟨sample, _, _, hg, hout⟩ := h\n    exact hout.trans (lookup_eq {position} {side}_index hg)\n",index_value=value(indices)?,index_fact=fact(indices)?,gather_value=value(gather_wire)?,gather_fact=fact(gather_wire)?));
        }
        source.push_str(&format!("  have reconstruction := ExponentGeneratedProgramSetup.public_cache_{count} witness facts gadget gadgetRuns flat\n  simpa only [public_get, private_get] using reconstruction\n\n"));
        count += 1;
    }
    if count == 0 {
        return Err("missing public cache alignment".into());
    }
    source.push_str("end ExponentGeneratedCacheAlignment\n");
    Ok(source)
}
