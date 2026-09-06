//! Actual root CRT reconstruction and exported residual, conditional on decoder rounds.
//! Intended fresh coefficients and their support come from the readout/input proof.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{MatrixBinaryOp, NodeKind},
    types::{NodeId, Port, WireRef},
};

pub(super) fn root_refresh_crt(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = artifact
        .scope_proofs
        .get(&FrozenGraphScopeId::Root)
        .ok_or("missing CRT root proof metadata")?;
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let mut candidates = root
        .nodes()
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.kind(), NodeKind::CrtRecompose { .. }));
    let (index, node) = candidates.next().ok_or("missing root CRT")?;
    if candidates.next().is_some() {
        return Err("ambiguous root CRT".into());
    }
    let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
    let NodeKind::CrtRecompose { modulus, plaintext_moduli, reconstruction_coefficients } =
        node.kind()
    else {
        unreachable!()
    };
    let args = root.arguments(node).ok_or("missing CRT operands")?;
    let slots = args.len();
    if slots < 2 || plaintext_moduli.len() != slots || reconstruction_coefficients.len() != slots {
        return Err("invalid CRT operand table".into());
    }
    let env = ParamEnv::default();
    let q = modulus.evaluate(&env).map_err(|_| "dynamic CRT modulus")?;
    let matrix = types[&wire].matrix_type().ok_or("CRT output is not a matrix")?;
    let n = matrix.ring_dimension;
    let columns = matrix.columns;
    if q <= 1.into() || matrix.modulus != q || matrix.rows != 1 || n == 0 || columns == 0 {
        return Err("unsupported CRT output shape".into());
    }
    let mut endpoints = artifact.root.outputs.values().filter_map(|output| {
        let node = root.node(output.wire.node)?;
        if !matches!(node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)) {
            return None;
        }
        let operands = root.arguments(node)?;
        (operands.len() == 2 && operands[0] == wire).then_some((output, operands[1]))
    });
    let (endpoint, anchor_wire) =
        endpoints.next().ok_or("missing exported CRT residual subtraction")?;
    if endpoints.next().is_some() {
        return Err("ambiguous exported CRT residual".into());
    }
    let mut sources = Vec::new();
    let mut plains = Vec::new();
    let mut weights = Vec::new();
    let mut levels = Vec::new();
    for (slot, input) in args.iter().enumerate() {
        let ty = types[input].matrix_type().ok_or("CRT level not a matrix")?;
        if ty.rows != 1 || ty.columns != columns || ty.ring_dimension != n || ty.modulus <= 1.into()
        {
            return Err("unsupported CRT level shape".into());
        }
        let p = plaintext_moduli[slot].evaluate(&env).map_err(|_| "dynamic CRT plaintext")?;
        let c =
            reconstruction_coefficients[slot].evaluate(&env).map_err(|_| "dynamic CRT weight")?;
        if p <= 1.into() || (&c * &p) % &q != 0.into() {
            return Err("CRT weight does not annihilate its plaintext modulus".into());
        }
        sources.push(ty.modulus.to_string());
        plains.push(p.to_string());
        weights.push(c);
        levels.push(value(*input)?);
    }
    if weights.iter().cloned().sum::<num_bigint::BigInt>() % &q != 1.into() {
        return Err("CRT weights do not partition unity".into());
    }
    let weights = weights.iter().map(|v| format!("({v})")).collect::<Vec<_>>();
    let record = shape.witness_record.as_ref().ok_or("missing CRT root witness")?;
    let body = shape.body_relation.as_ref().ok_or("missing CRT root body")?;
    let constraint = shape
        .node_constraints
        .get(&wire.node)
        .and_then(|r| r.clone().last())
        .ok_or("missing CRT equation constraint")?;
    let actual = value(wire)?;
    let anchor_value = value(anchor_wire)?;
    let residual = value(endpoint.wire)?;
    let mut text = format!(
        "import {}\nimport ExponentRefreshCrt\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 4000000\nnamespace ExponentGeneratedRefreshCrt\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\ndef source : Fin {slots} → Nat := ![{}]\ndef plaintext : Fin {slots} → Int := ![{}]\ndef weight : Fin {slots} → Int := ![{}]\n\ntheorem annihilates : ∀ slot, (weight slot : ZMod {q}) * (plaintext slot : ZMod {q}) = 0 := by\n  intro slot\n  fin_cases slot <;> norm_num [weight, plaintext] <;> decide\n\ntheorem partition : (∑ slot : Fin {slots}, (weight slot : ZMod {q})) = 1 := by\n  norm_num [weight, Fin.sum_univ_succ]\n  all_goals decide\n\ntheorem residual_of_rounds {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (anchor : ErrorMatrix {n} 1 {columns})\n    (digits : Fin 1 → Fin {columns} → Fin {n} → Int)\n    (anchorSource : ({anchor_value}) = reduceMatrix {q} {n} 1 {columns} anchor)\n",
        artifact.module_name,
        sources.join(", "),
        plains.join(", "),
        weights.join(", "),
        artifact.root.parameter_type,
        artifact.root.input_type,
        artifact.root.output_type,
    );
    for slot in 0..slots {
        text.push_str(&format!(
            "    (rounded_{slot} : ∀ row column (i : Fin {n}),\n      ((({} : Int) * Int.ofNat ((({}) row column).coeff i).val + Int.ofNat ({} / 2)) / Int.ofNat {}) % ({} : Int) =\n      ((anchor + refreshFreshLift digits) row column).coeff i % ({} : Int))\n",
            plains[slot], levels[slot], sources[slot], sources[slot], plains[slot], plains[slot],
        ));
    }
    text.push_str(&format!(
        "    {{bound : Nat}} (support : ∀ row column i, (digits row column i).natAbs ≤ bound) :\n    Approx ({}) 0 bound := by\n  let levels : (slot : Fin {slots}) → ExactMatrix (source slot) {n} 1 {columns} :=\n",
        endpoint.projection,
    ));
    let mut level_cases = "fun i => Fin.elim0 i".to_string();
    for level in levels.iter().rev() {
        level_cases = format!("Fin.cases ({level}) ({level_cases})");
    }
    text.push_str(&format!("    {level_cases}\n"));
    text.push_str(&format!(
        "  have crtRuns : ({actual}) = ∑ slot : Fin {slots}, MxxRuntime.crtRecomposeLevel (plaintext slot) (weight slot) (levels slot) := by\n    simpa [levels, source, plaintext, weight, Fin.sum_univ_succ, add_assoc] using facts{}.1\n  have endpointEquation : ({residual}) = ({actual}) - reduceMatrix {q} {n} 1 {columns} anchor := by\n    change ({actual}) - ({anchor_value}) = _\n    rw [anchorSource]\n  have result : Approx ({residual}) 0 bound := refresh_crt_residual (by decide) (by decide) source plaintext weight levels anchor digits ({actual}) ({residual}) crtRuns endpointEquation annihilates partition (by\n    intro slot\n    fin_cases slot\n",
        ".2".repeat(constraint),
    ));
    for slot in 0..slots {
        text.push_str(&format!("    · exact rounded_{slot}\n"));
    }
    text.push_str(&format!(
        "    ) support\n  have outputEquation := facts{}\n  rw [outputEquation]\n  exact result\n\nend ExponentGeneratedRefreshCrt\n",
        ".2".repeat(shape.constraint_count),
    ));
    Ok(text)
}
