use crate::{
    ar16::{AR16PublicKey, Ar16Error, EvalContext, quadratic::combine_public_keys},
    circuit::{PolyCircuit, PolyGateType, gate::GateId},
    poly::Poly,
};
use std::collections::HashMap;

/// Result bundle returned by `eval_public_key`.
#[derive(Debug, Clone)]
pub struct EvalPkOutput<P: Poly> {
    pub outputs: Vec<AR16PublicKey<P>>,
    pub cache: HashMap<GateId, AR16PublicKey<P>>,
}

impl<P: Poly> EvalPkOutput<P> {
    pub fn new(outputs: Vec<AR16PublicKey<P>>, cache: HashMap<GateId, AR16PublicKey<P>>) -> Self {
        Self { outputs, cache }
    }
}

/// Evaluate the public-key labels for every circuit output wire.
pub fn eval_public_key<P>(
    circuit: &PolyCircuit<P>,
    ctx: &mut EvalContext<P>,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
) -> Result<EvalPkOutput<P>, Ar16Error>
where
    P: Poly,
{
    let mut outputs = Vec::with_capacity(circuit.num_output());

    for &gate in circuit.output_ids() {
        let pk = evaluate_gate_pk(circuit, ctx, input_keys, gate)?;
        outputs.push(pk);
    }

    let cache_snapshot = ctx.pk_cache().clone();
    Ok(EvalPkOutput::new(outputs, cache_snapshot))
}

fn evaluate_gate_pk<P>(
    circuit: &PolyCircuit<P>,
    ctx: &mut EvalContext<P>,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
    gate: GateId,
) -> Result<AR16PublicKey<P>, Ar16Error>
where
    P: Poly,
{
    if let Some(existing) = ctx.pk_cache().get(&gate) {
        return Ok(existing.clone());
    }

    if let Some(from_input) = input_keys.get(&gate) {
        let cloned = from_input.clone();
        ctx.pk_cache_mut().insert(gate, cloned.clone());
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
            let lhs = evaluate_gate_pk(circuit, ctx, input_keys, lhs_id)?;
            let rhs = evaluate_gate_pk(circuit, ctx, input_keys, rhs_id)?;
            let mut sum = lhs.clone();
            sum += &rhs;
            sum
        }
        PolyGateType::Sub => {
            let lhs_id = gate_def.input_gates[0];
            let rhs_id = gate_def.input_gates[1];
            let lhs = evaluate_gate_pk(circuit, ctx, input_keys, lhs_id)?;
            let rhs = evaluate_gate_pk(circuit, ctx, input_keys, rhs_id)?;
            let mut diff = lhs.clone();
            diff -= &rhs;
            diff
        }
        PolyGateType::Mul => {
            let lhs_id = gate_def.input_gates[0];
            let rhs_id = gate_def.input_gates[1];
            let lhs_pk = evaluate_gate_pk(circuit, ctx, input_keys, lhs_id)?;
            let rhs_pk = evaluate_gate_pk(circuit, ctx, input_keys, rhs_id)?;
            let advice = ctx
                .advice_for_level(level)
                .ok_or(Ar16Error::MissingAdvice { level, gate: Some(gate) })?;
            combine_public_keys(level, advice, lhs_id, rhs_id, &lhs_pk, &rhs_pk)?
        }
        _ => {
            return Err(Ar16Error::UnsupportedGate { gate });
        }
    };

    ctx.pk_cache_mut().insert(gate, computed.clone());
    Ok(computed)
}

pub(crate) fn determine_gate_level<P: Poly>(
    circuit: &PolyCircuit<P>,
    ctx: &mut EvalContext<P>,
    gate: GateId,
) -> Result<usize, Ar16Error> {
    if let Some(level) = ctx.gate_level(gate) {
        return Ok(level);
    }

    let gate_def = circuit.gate(gate).ok_or(Ar16Error::UnsupportedGate { gate })?;
    let level = match &gate_def.gate_type {
        PolyGateType::Input => {
            ctx.set_gate_level(gate, 0);
            return Ok(0);
        }
        PolyGateType::Add | PolyGateType::Sub => gate_def
            .input_gates
            .iter()
            .map(|&child| determine_gate_level(circuit, ctx, child))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0),
        PolyGateType::Mul => {
            let max_child = gate_def
                .input_gates
                .iter()
                .map(|&child| determine_gate_level(circuit, ctx, child))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            max_child + 1
        }
        _ => return Err(Ar16Error::UnsupportedGate { gate }),
    };
    ctx.set_gate_level(gate, level);
    Ok(level)
}
