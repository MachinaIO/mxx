use std::collections::{BTreeMap, HashMap};

use crate::{
    ar16::{Ar16Error, EvalContext, Level, quadratic::combine_public_keys},
    circuit::{PolyCircuit, PolyGateType, gate::GateId},
    poly::Poly,
};

use super::AR16PublicKey;

/// Evaluate AR16 public keys for all gates required by the outputs.
pub fn eval_ar16_public_keys<P: Poly>(
    circuit: &PolyCircuit<P>,
    ctx: &mut EvalContext<P>,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
) -> Result<BTreeMap<GateId, AR16PublicKey<P>>, Ar16Error> {
    for &gate in circuit.output_ids() {
        eval_pk_gate(circuit, gate, ctx, input_keys)?;
    }
    let mut result = BTreeMap::new();
    for (gate, pk) in ctx.pk_cache().iter() {
        result.insert(*gate, pk.clone());
    }
    Ok(result)
}

/// Gate-level public-key evaluation (Algorithm EvalPK in AR16 §5).
pub fn eval_pk_gate<P: Poly>(
    circuit: &PolyCircuit<P>,
    gate: GateId,
    ctx: &mut EvalContext<P>,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
) -> Result<AR16PublicKey<P>, Ar16Error> {
    if let Some(existing) = ctx.pk_cache().get(&gate) {
        return Ok(existing.clone());
    }

    let gate_def = circuit.gate(gate).ok_or_else(|| Ar16Error::unsupported_gate(gate))?;
    let level = determine_gate_level(circuit, ctx, gate)?;

    let pk = match gate_def.gate_type {
        PolyGateType::Input => {
            input_keys.get(&gate).cloned().ok_or_else(|| Ar16Error::missing_input(gate))?
        }
        PolyGateType::Add | PolyGateType::Sub => {
            let lhs = gate_def.input_gates[0];
            let rhs = gate_def.input_gates[1];
            let lhs_pk = eval_pk_gate(circuit, lhs, ctx, input_keys)?;
            let rhs_pk = eval_pk_gate(circuit, rhs, ctx, input_keys)?;
            if lhs_pk.level() != rhs_pk.level() {
                return Err(Ar16Error::level_mismatch(lhs_pk.level(), rhs_pk.level()));
            }
            match gate_def.gate_type {
                PolyGateType::Add => {
                    let mut sum = lhs_pk.clone();
                    sum += &rhs_pk;
                    sum
                }
                PolyGateType::Sub => {
                    let mut diff = lhs_pk.clone();
                    diff -= &rhs_pk;
                    diff
                }
                _ => unreachable!(),
            }
        }
        PolyGateType::Mul => {
            let lhs = gate_def.input_gates[0];
            let rhs = gate_def.input_gates[1];
            let lhs_pk = eval_pk_gate(circuit, lhs, ctx, input_keys)?;
            let rhs_pk = eval_pk_gate(circuit, rhs, ctx, input_keys)?;
            if level.0 < 2 {
                return Err(Ar16Error::MissingAdvice { level, gate: Some(gate) });
            }
            combine_public_keys(level, ctx, lhs, rhs, &lhs_pk, &rhs_pk)?
        }
        _ => return Err(Ar16Error::unsupported_gate(gate)),
    };

    ctx.pk_cache_mut().insert(gate, pk.clone());
    Ok(pk)
}

/// Determine and cache the level for a gate according to Section 1.
pub fn determine_gate_level<P: Poly>(
    circuit: &PolyCircuit<P>,
    ctx: &mut EvalContext<P>,
    gate: GateId,
) -> Result<Level, Ar16Error> {
    if let Some(level) = ctx.gate_level(gate) {
        return Ok(level);
    }

    let gate_def = circuit.gate(gate).ok_or_else(|| Ar16Error::unsupported_gate(gate))?;
    let level = match gate_def.gate_type {
        PolyGateType::Input => Level::new(1),
        PolyGateType::Add | PolyGateType::Sub => gate_def
            .input_gates
            .iter()
            .map(|&child| determine_gate_level(circuit, ctx, child))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(Level::new(1)),
        PolyGateType::Mul => {
            let max_child = gate_def
                .input_gates
                .iter()
                .map(|&child| determine_gate_level(circuit, ctx, child))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or(Level::new(1));
            max_child.next()
        }
        _ => return Err(Ar16Error::unsupported_gate(gate)),
    };
    ctx.set_gate_level(gate, level);
    Ok(level)
}
