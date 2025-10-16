use crate::{
    ar16::{
        AR16Encoding, AR16PublicKey, Ar16Error,
        eval_pk::{EvalPkOutput, determine_gate_level},
        quadratic::combine_encodings,
        state::EvalContext,
    },
    circuit::{PolyCircuit, PolyGateType, gate::GateId},
    poly::Poly,
};
use std::collections::HashMap;

pub fn eval_encoding<P: Poly>(
    circuit: &PolyCircuit<P>,
    pk_output: &EvalPkOutput<P>,
    ctx: &mut EvalContext<P>,
    input_encodings: &HashMap<GateId, AR16Encoding<P>>,
) -> Result<Vec<AR16Encoding<P>>, Ar16Error> {
    if pk_output.outputs.len() != circuit.num_output() {
        return Err(Ar16Error::LayerMismatch {
            message: "eval_public_key output count does not match circuit outputs",
        });
    }

    let mut outputs = Vec::with_capacity(circuit.num_output());
    for &gate in circuit.output_ids() {
        let encoding = evaluate_gate_encoding(circuit, pk_output, ctx, input_encodings, gate)?;
        outputs.push(encoding);
    }
    Ok(outputs)
}

fn evaluate_gate_encoding<P: Poly>(
    circuit: &PolyCircuit<P>,
    pk_output: &EvalPkOutput<P>,
    ctx: &mut EvalContext<P>,
    input_encodings: &HashMap<GateId, AR16Encoding<P>>,
    gate: GateId,
) -> Result<AR16Encoding<P>, Ar16Error> {
    if let Some(existing) = ctx.encoding_cache().get(&gate) {
        return Ok(existing.clone());
    }

    if let Some(from_input) = input_encodings.get(&gate) {
        let cloned = from_input.clone();
        ctx.encoding_cache_mut().insert(gate, cloned.clone());
        return Ok(cloned);
    }

    let gate_def = circuit.gate(gate).ok_or(Ar16Error::UnsupportedGate { gate })?;
    let level = determine_gate_level(circuit, ctx, gate)?;

    let computed = match &gate_def.gate_type {
        PolyGateType::Input => {
            return Err(Ar16Error::MissingAdvice { level: 0, gate: Some(gate) });
        }
        PolyGateType::Add => {
            let lhs_id = gate_def.input_gates[0];
            let rhs_id = gate_def.input_gates[1];
            let lhs = evaluate_gate_encoding(circuit, pk_output, ctx, input_encodings, lhs_id)?;
            let rhs = evaluate_gate_encoding(circuit, pk_output, ctx, input_encodings, rhs_id)?;
            let mut sum = lhs.clone();
            sum += &rhs;
            sum
        }
        PolyGateType::Sub => {
            let lhs_id = gate_def.input_gates[0];
            let rhs_id = gate_def.input_gates[1];
            let lhs = evaluate_gate_encoding(circuit, pk_output, ctx, input_encodings, lhs_id)?;
            let rhs = evaluate_gate_encoding(circuit, pk_output, ctx, input_encodings, rhs_id)?;
            let mut diff = lhs.clone();
            diff -= &rhs;
            diff
        }
        PolyGateType::Mul => {
            let lhs_id = gate_def.input_gates[0];
            let rhs_id = gate_def.input_gates[1];
            let lhs_child =
                evaluate_gate_encoding(circuit, pk_output, ctx, input_encodings, lhs_id)?;
            let rhs_child =
                evaluate_gate_encoding(circuit, pk_output, ctx, input_encodings, rhs_id)?;
            let lhs_pk = get_public_key_for_gate(ctx, pk_output, lhs_id)
                .ok_or(Ar16Error::MissingAdvice { level, gate: Some(lhs_id) })?;
            let rhs_pk = get_public_key_for_gate(ctx, pk_output, rhs_id)
                .ok_or(Ar16Error::MissingAdvice { level, gate: Some(rhs_id) })?;
            let advice = ctx
                .advice_for_level(level)
                .ok_or(Ar16Error::MissingAdvice { level, gate: Some(gate) })?;
            combine_encodings(
                level, advice, lhs_id, rhs_id, &lhs_child, &rhs_child, &lhs_pk, &rhs_pk,
            )?
        }
        _ => {
            return Err(Ar16Error::UnsupportedGate { gate });
        }
    };

    ctx.encoding_cache_mut().insert(gate, computed.clone());
    Ok(computed)
}

fn get_public_key_for_gate<P: Poly>(
    ctx: &EvalContext<P>,
    pk_output: &EvalPkOutput<P>,
    gate: GateId,
) -> Option<AR16PublicKey<P>> {
    ctx.pk_cache().get(&gate).cloned().or_else(|| pk_output.cache.get(&gate).cloned())
}
