use std::collections::HashMap;

use crate::{
    ar16::{Ar16Error, EvalContext, Level},
    circuit::{PolyCircuit, PolyGateType, gate::GateId},
    poly::Poly,
};

use super::{AR16Encoding, AR16PublicKey, encoding::eval_ct_gate, pk::eval_pk_gate};

/// Combine public keys at a multiplication gate according to equation (5.25).
pub fn combine_public_keys<P: Poly>(
    circuit: &PolyCircuit<P>,
    level: Level,
    ctx: &mut EvalContext<P>,
    lhs_gate: GateId,
    rhs_gate: GateId,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
    input_encodings: &HashMap<GateId, AR16Encoding<P>>,
    lhs_pk: &AR16PublicKey<P>,
    rhs_pk: &AR16PublicKey<P>,
) -> Result<AR16PublicKey<P>, Ar16Error> {
    let enc_s2 =
        ctx.advice().ek_s2(level).ok_or_else(|| Ar16Error::missing_advice(level, None))?.clone();
    let cis = reconstruct_times_s(circuit, ctx, level, lhs_gate, input_keys, input_encodings)?;
    let cjs = reconstruct_times_s(circuit, ctx, level, rhs_gate, input_keys, input_encodings)?;
    let product = lhs_pk.u().clone() * rhs_pk.u().clone();
    let mut result = AR16PublicKey::new(level, P::const_zero(ctx.params()));
    result += &enc_s2.label().scale(&product);
    result -= &cis.label().scale(rhs_pk.u());
    result -= &cjs.label().scale(lhs_pk.u());
    Ok(result)
}

/// Combine ciphertext encodings at a multiplication gate according to equation (5.24).
#[allow(clippy::too_many_arguments)]
pub fn combine_encodings<P: Poly>(
    circuit: &PolyCircuit<P>,
    level: Level,
    ctx: &mut EvalContext<P>,
    lhs_gate: GateId,
    rhs_gate: GateId,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
    input_encodings: &HashMap<GateId, AR16Encoding<P>>,
    lhs_ct: &AR16Encoding<P>,
    rhs_ct: &AR16Encoding<P>,
    lhs_pk: &AR16PublicKey<P>,
    rhs_pk: &AR16PublicKey<P>,
) -> Result<AR16Encoding<P>, Ar16Error> {
    let enc_s2 =
        ctx.advice().ek_s2(level).ok_or_else(|| Ar16Error::missing_advice(level, None))?.clone();
    let cis = reconstruct_times_s(circuit, ctx, level, lhs_gate, input_keys, input_encodings)?;
    let cjs = reconstruct_times_s(circuit, ctx, level, rhs_gate, input_keys, input_encodings)?;

    let mut combined = multiply_encodings(level, ctx, lhs_ct, rhs_ct);
    let product = lhs_pk.u().clone() * rhs_pk.u().clone();

    combined += &enc_s2.scale(&product);
    combined -= &cis.scale(rhs_pk.u());
    combined -= &cjs.scale(lhs_pk.u());
    Ok(combined)
}

/// Retrieve (or reconstruct) the encoding `E_k(c^{k-1}_g · s)`.
pub fn reconstruct_times_s<P: Poly>(
    circuit: &PolyCircuit<P>,
    ctx: &mut EvalContext<P>,
    level: Level,
    gate: GateId,
    input_keys: &HashMap<GateId, AR16PublicKey<P>>,
    input_encodings: &HashMap<GateId, AR16Encoding<P>>,
) -> Result<AR16Encoding<P>, Ar16Error> {
    if let Some(existing) = ctx.times_s_cache().get(&(level, gate)) {
        return Ok(existing.clone());
    }

    if level.0 < 2 {
        return Err(Ar16Error::missing_advice(level, Some(gate)));
    }

    if level.0 == 2 {
        let encoding = ctx
            .advice()
            .ek_times_s_of(level, gate)
            .cloned()
            .ok_or_else(|| Ar16Error::missing_advice(level, Some(gate)))?;
        ctx.times_s_cache_mut().insert((level, gate), encoding.clone());
        return Ok(encoding);
    }

    let gate_def = circuit.gate(gate).ok_or_else(|| Ar16Error::unsupported_gate(gate))?;
    if gate_def.gate_type != PolyGateType::Mul {
        return Err(Ar16Error::unsupported_gate(gate));
    }
    let lhs = gate_def.input_gates[0];
    let rhs = gate_def.input_gates[1];

    let lhs_ct = eval_ct_gate(circuit, lhs, ctx, input_keys, input_encodings)?;

    let lhs_pk = eval_pk_gate(circuit, lhs, ctx, input_keys, input_encodings)?;
    let rhs_pk = eval_pk_gate(circuit, rhs, ctx, input_keys, input_encodings)?;

    let prev_level = Level::new(level.0 - 1);
    let rhs_times_prev =
        reconstruct_times_s(circuit, ctx, prev_level, rhs, input_keys, input_encodings)?;

    let enc_s2 =
        ctx.advice().ek_s2(level).ok_or_else(|| Ar16Error::missing_advice(level, None))?.clone();
    let t_js = ctx
        .advice()
        .ek_times_s_of_times(level, rhs)
        .cloned()
        .ok_or_else(|| Ar16Error::missing_advice(level, Some(rhs)))?;
    let t_i = ctx
        .advice()
        .ek_times_s_of(level, lhs)
        .cloned()
        .ok_or_else(|| Ar16Error::missing_advice(level, Some(lhs)))?;
    let s2_times = ctx
        .advice()
        .ek_times_s_s2(level)
        .cloned()
        .ok_or_else(|| Ar16Error::missing_advice(level, None))?;

    let ui = lhs_pk.u().clone();
    let uj = rhs_pk.u().clone();
    let u_js = rhs_times_prev.label().u().clone();

    let mut q = multiply_encodings(level, ctx, &lhs_ct, &rhs_times_prev);
    q += &enc_s2.clone().scale(&(ui.clone() * u_js.clone()));
    q -= &t_js.clone().scale(&ui);
    q -= &t_i.clone().scale(&u_js);

    let mut result = q;
    result -= &t_js.scale(&uj);
    result -= &t_i.scale(&ui);
    result += &s2_times.scale(&(ui.clone() * uj.clone()));

    ctx.times_s_cache_mut().insert((level, gate), result.clone());
    Ok(result)
}

fn multiply_encodings<P: Poly>(
    level: Level,
    ctx: &EvalContext<P>,
    lhs: &AR16Encoding<P>,
    rhs: &AR16Encoding<P>,
) -> AR16Encoding<P> {
    let product = lhs.body().clone() * rhs.body().clone();
    AR16Encoding::new(AR16PublicKey::new(level, P::const_zero(ctx.params())), product)
}
