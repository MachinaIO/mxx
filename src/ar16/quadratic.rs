use crate::{
    ar16::{Ar16Error, EvalContext, Level},
    circuit::gate::GateId,
    poly::Poly,
};

use super::{AR16Encoding, AR16PublicKey};

/// Combine public keys at a multiplication gate according to equation (5.25).
pub fn combine_public_keys<P: Poly>(
    level: Level,
    ctx: &mut EvalContext<P>,
    lhs_gate: GateId,
    rhs_gate: GateId,
    lhs_pk: &AR16PublicKey<P>,
    rhs_pk: &AR16PublicKey<P>,
) -> Result<AR16PublicKey<P>, Ar16Error> {
    let advice = ctx.advice();
    let enc_s2 = advice.ek_s2(level).ok_or_else(|| Ar16Error::missing_advice(level, None))?;
    let cis = advice
        .ek_times_s_of(level, lhs_gate)
        .ok_or_else(|| Ar16Error::missing_advice(level, Some(lhs_gate)))?;
    let cjs = advice
        .ek_times_s_of(level, rhs_gate)
        .ok_or_else(|| Ar16Error::missing_advice(level, Some(rhs_gate)))?;

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
    level: Level,
    ctx: &mut EvalContext<P>,
    lhs_gate: GateId,
    rhs_gate: GateId,
    lhs_ct: &AR16Encoding<P>,
    rhs_ct: &AR16Encoding<P>,
    lhs_pk: &AR16PublicKey<P>,
    rhs_pk: &AR16PublicKey<P>,
) -> Result<AR16Encoding<P>, Ar16Error> {
    let advice = ctx.advice();
    let enc_s2 = advice.ek_s2(level).ok_or_else(|| Ar16Error::missing_advice(level, None))?.clone();
    let cis = reconstruct_times_s(ctx, level, lhs_gate)?;
    let cjs = reconstruct_times_s(ctx, level, rhs_gate)?;

    let mut combined = multiply_encodings(level, ctx, lhs_ct, rhs_ct);
    let product = lhs_pk.u().clone() * rhs_pk.u().clone();

    combined += &enc_s2.scale(&product);
    combined -= &cis.scale(rhs_pk.u());
    combined -= &cjs.scale(lhs_pk.u());
    Ok(combined)
}

/// Retrieve (or reconstruct) the encoding `E_k(c^{k-1}_g · s)`.
pub fn reconstruct_times_s<P: Poly>(
    ctx: &mut EvalContext<P>,
    level: Level,
    gate: GateId,
) -> Result<AR16Encoding<P>, Ar16Error> {
    if let Some(existing) = ctx.times_s_cache().get(&(level, gate)) {
        return Ok(existing.clone());
    }

    let advice = ctx.advice();
    let encoding = advice
        .ek_times_s_of(level, gate)
        .cloned()
        .ok_or_else(|| Ar16Error::missing_advice(level, Some(gate)))?;
    ctx.times_s_cache_mut().insert((level, gate), encoding.clone());
    Ok(encoding)
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
