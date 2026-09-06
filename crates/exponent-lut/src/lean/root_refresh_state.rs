//! Bind each actual state-scaling cache to its exact GadgetScalar noise gain.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::ExponentLutNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::WireRef,
};

pub(super) fn root_refresh_state(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing state cache operation fact")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let record = shape.witness_record.as_ref().ok_or("missing PRE state witness")?;
    let body = shape.body_relation.as_ref().ok_or("missing PRE state body")?;
    let binders = format!(
        "{{hashModel : MxxRuntime.HashModel}} {{params : {}}}\n    {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} RefreshBackend.backend hashModel params inputs outputs witness)",
        artifact.root.parameter_type, artifact.root.input_type, artifact.root.output_type
    );
    let routes = snapshot.refresh.route_gains.as_ref().ok_or("missing state scaling gains")?;
    let mut source = format!(
        "import {}\nimport Backend\nimport ExponentRefreshDecomposition\nimport ExponentSetup\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshState\nopen Mxx.Primitives MxxRuntime MxxExponentLut\nopen scoped BigOperators\n\n",
        artifact.module_name
    );
    let mut slots = 0;
    for (name, output) in &artifact.root.outputs {
        let Some((_, slot)) = name.rsplit_once("-state-scaling-k-rhs-") else { continue };
        let slot = slot.parse::<usize>().map_err(|_| "bad state cache slot")?;
        let d = output.wire;
        let node = root.node(d.node).ok_or("missing state decomposition")?;
        let NodeKind::GadgetDecompose { base, digit_count: digits, .. } = node.kind() else {
            return Err("state scaling cache not direct gadget decomposition".into())
        };
        let target = root.arguments(node).ok_or("missing state decomposition input")?[0];
        let target_node = root.node(target.node).ok_or("missing state scalar product")?;
        if !matches!(target_node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) {
            return Err("state cache target not scalar product".into())
        }
        let factors = root.arguments(target_node).ok_or("missing state scaling operands")?;
        if factors.len() != 2 {
            return Err("state product arity".into())
        }
        let scalar = factors[0];
        let gadget = factors[1];
        let scalar_node = root.node(scalar.node).ok_or("missing state scalar")?;
        let NodeKind::ConstantMatrix { value: ConstantMatrix::Polynomial { coefficients }, .. } =
            scalar_node.kind()
        else {
            return Err("state scaling scalar not actual polynomial literal".into())
        };
        if coefficients.len() != 1 {
            return Err("state scale is not constant coefficient".into())
        }
        let mu = coefficients[0].evaluate(&ParamEnv::default()).map_err(|_| "dynamic state mu")?;
        let gadget_node = root.node(gadget.node).ok_or("missing actual state gadget")?;
        let NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { base: gbase, .. }, .. } =
            gadget_node.kind()
        else {
            return Err("state gadget not runtime gadget constructor".into())
        };
        if base != gbase {
            return Err("state gadget and decomposition radices disagree".into())
        }
        let mt = types[&gadget].matrix_type().ok_or("state gadget type")?;
        let dt = types[&d].matrix_type().ok_or("state decomposition type")?;
        let q = &mt.modulus;
        let n = mt.ring_dimension;
        let rows = mt.rows;
        let columns = mt.columns;
        let inner = dt.rows;
        if dt.modulus != *q || dt.ring_dimension != n || dt.columns != columns {
            return Err("state cache dimensions differ".into())
        }
        let layout = artifact
            .backend_layouts
            .iter()
            .position(|l| {
                l.modulus == *q && l.ring_dimension == n && rows * l.regular_digits == columns
            })
            .ok_or("state backend layout not registered")?;
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| "dynamic state base")?;
        let digits =
            digits.evaluate(&ParamEnv::default()).map_err(|_| "dynamic state digit count")?;
        let plaintext =
            snapshot.refresh.plaintext_moduli.get(slot).ok_or("missing state plaintext modulus")?;
        let global =
            snapshot.refresh.global_modulus.as_ref().ok_or("missing global state modulus")?;
        if q.to_biguint().as_ref() != Some(global) ||
            mu.to_biguint().as_ref() != Some(&(global / plaintext))
        {
            return Err("actual state mu differs from q/plaintext".into())
        }
        let gain = &routes.state_scaling.get(slot).ok_or("missing state scalar gain")?.l1;
        let gv = value(gadget)?;
        let dv = value(d)?;
        let tv = value(target)?;
        let sv = value(scalar)?;
        let gain_expr = format!(
            "Finset.univ.sup (fun digit : Fin RefreshBackend.layout{layout}.digitsPerTower => sparseColumnGain RefreshBackend.layout{layout} (({mu} : Int) * (RefreshBackend.layout{layout}.base : Int) ^ digit.val))"
        );
        source.push_str(&format!("abbrev stateScale_{slot} : Int := {mu}\nabbrev stateGain_{slot} : Nat := {gain}\n\n/-- Exact equality with the snapshot's state_scaling action gain. -/\ntheorem state_gain_{slot} : ({gain_expr}) = stateGain_{slot} := by decide\n\n/-- The actual producer gadget and cache execute the same registered layout. -/\ntheorem state_cache_{slot} {binders} :\n    ∃ action : ErrorMatrix {n} {inner} {columns},\n      ({dv}) = reduceMatrix {q} {n} {inner} {columns} action ∧\n      ActionGain action stateGain_{slot} ∧\n      ({gv}) * ({dv}) = stateScale_{slot} • ({gv}) ∧\n      MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {base} {digits} ({gv}) := by\n  have gadgetRuns : MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {base} {digits} ({gv}) := {}\n  have scalarEquation : ({sv}) 0 0 = (stateScale_{slot} : ExactPoly {q} {n}) := by\n    norm_num [MxxRuntime.matrixPolynomial, MxxIR.exactDiv, stateScale_{slot}]\n  have targetEquation : ({tv}) = stateScale_{slot} • ({gv}) := by\n    change MxxRuntime.matrixMulScalarLeft ({sv}) ({gv}) = _\n    ext row column\n    simp only [MxxRuntime.matrixMulScalarLeft, scalarEquation, Matrix.smul_apply, zsmul_eq_mul]\n  have decomposeRuns : MxxRuntime.gadgetDecomposeRuns RefreshBackend.backend {base} {digits} (stateScale_{slot} • ({gv})) ({dv}) := by\n    rw [← targetEquation]\n    exact {}\n  obtain ⟨action, actionEquation, actionGain⟩ := gadgetRuns_refresh_gain RefreshBackend.layout{layout}\n    (by rfl) (by decide) (by decide) stateScale_{slot} gadgetRuns decomposeRuns (by rfl)\n  refine ⟨action, actionEquation, ?_, gadget_runs_reconstruct (by decide) (by decide) gadgetRuns decomposeRuns Fact.out, gadgetRuns⟩\n  simpa only [stateScale_{slot}, state_gain_{slot}] using actionGain\n\n",fact(gadget)?,fact(d)?));
        slots += 1;
    }
    if slots != snapshot.refresh.slot_count {
        return Err("state cache slot coverage mismatch".into())
    }
    source.push_str("end ExponentGeneratedRefreshState\n");
    Ok(source)
}
