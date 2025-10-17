use std::collections::{BTreeMap, HashMap};

use crate::{
    ar16::{Ar16Error, EvalContext, pk::determine_gate_level, quadratic::combine_encodings},
    circuit::{PolyCircuit, PolyGateType, gate::GateId},
    poly::Poly,
};

use super::{AR16Encoding, AR16PublicKey, eval_pk_gate};

/// Evaluate AR16 ciphertext encodings for all gates required by the outputs.
pub fn eval_ar16_encodings<P: Poly>(
    circuit: &PolyCircuit<P>,
    ctx: &mut EvalContext<P>,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
    input_encodings: &HashMap<GateId, AR16Encoding<P>>,
) -> Result<BTreeMap<GateId, AR16Encoding<P>>, Ar16Error> {
    for &gate in circuit.output_ids() {
        eval_ct_gate(circuit, gate, ctx, input_keys, input_encodings)?;
    }
    let mut result = BTreeMap::new();
    for (gate, ct) in ctx.encoding_cache().iter() {
        result.insert(*gate, ct.clone());
    }
    Ok(result)
}

/// Gate-level ciphertext evaluation (EvalCT in AR16 §5).
pub fn eval_ct_gate<P: Poly>(
    circuit: &PolyCircuit<P>,
    gate: GateId,
    ctx: &mut EvalContext<P>,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
    input_encodings: &HashMap<GateId, AR16Encoding<P>>,
) -> Result<AR16Encoding<P>, Ar16Error> {
    if let Some(existing) = ctx.encoding_cache().get(&gate) {
        return Ok(existing.clone());
    }

    let gate_def = circuit.gate(gate).ok_or_else(|| Ar16Error::unsupported_gate(gate))?;
    let level = determine_gate_level(circuit, ctx, gate)?;

    let encoding = match gate_def.gate_type {
        PolyGateType::Input => {
            input_encodings.get(&gate).cloned().ok_or_else(|| Ar16Error::missing_input(gate))?
        }
        PolyGateType::Add | PolyGateType::Sub => {
            let lhs = gate_def.input_gates[0];
            let rhs = gate_def.input_gates[1];
            let lhs_ct = eval_ct_gate(circuit, lhs, ctx, input_keys, input_encodings)?;
            let rhs_ct = eval_ct_gate(circuit, rhs, ctx, input_keys, input_encodings)?;
            if lhs_ct.level() != rhs_ct.level() {
                return Err(Ar16Error::level_mismatch(lhs_ct.level(), rhs_ct.level()));
            }
            match gate_def.gate_type {
                PolyGateType::Add => {
                    let mut sum = lhs_ct.clone();
                    sum += &rhs_ct;
                    sum
                }
                PolyGateType::Sub => {
                    let mut diff = lhs_ct.clone();
                    diff -= &rhs_ct;
                    diff
                }
                _ => unreachable!(),
            }
        }
        PolyGateType::Mul => {
            let lhs = gate_def.input_gates[0];
            let rhs = gate_def.input_gates[1];
            let lhs_ct = eval_ct_gate(circuit, lhs, ctx, input_keys, input_encodings)?;
            let rhs_ct = eval_ct_gate(circuit, rhs, ctx, input_keys, input_encodings)?;
            let lhs_pk = eval_pk_gate(circuit, lhs, ctx, input_keys, input_encodings)?;
            let rhs_pk = eval_pk_gate(circuit, rhs, ctx, input_keys, input_encodings)?;
            if level.0 < 2 {
                return Err(Ar16Error::missing_advice(level, Some(gate)));
            }
            combine_encodings(
                circuit,
                level,
                ctx,
                lhs,
                rhs,
                input_keys,
                input_encodings,
                &lhs_ct,
                &rhs_ct,
                &lhs_pk,
                &rhs_pk,
            )?
        }
        _ => return Err(Ar16Error::unsupported_gate(gate)),
    };

    ctx.encoding_cache_mut().insert(gate, encoding.clone());
    Ok(encoding)
}
