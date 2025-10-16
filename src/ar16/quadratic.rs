use crate::{
    ar16::{
        AR16Encoding, AR16PublicKey, Ar16Error,
        advice::{AdviceRole, AdviceSet},
    },
    circuit::gate::GateId,
    poly::Poly,
};

fn multiply_encodings<P: Poly>(lhs: &AR16Encoding<P>, rhs: &AR16Encoding<P>) -> AR16Encoding<P> {
    let a = lhs.a.a.clone() * rhs.b.clone() + rhs.a.a.clone() * lhs.b.clone();
    let b = lhs.b.clone() * rhs.b.clone();
    AR16Encoding::new(AR16PublicKey::new(a), b)
}

pub fn combine_public_keys<P: Poly>(
    level: usize,
    advice: &AdviceSet<P>,
    lhs_gate: GateId,
    rhs_gate: GateId,
    lhs_pk: &AR16PublicKey<P>,
    rhs_pk: &AR16PublicKey<P>,
) -> Result<AR16PublicKey<P>, Ar16Error> {
    let pk_s2 = advice
        .get_public_key(&AdviceRole::SecretSquare)
        .cloned()
        .ok_or(Ar16Error::MissingAdvice { level, gate: None })?;
    let pk_ci_s = advice
        .get_public_key(&AdviceRole::LiftedCipher { gate: lhs_gate })
        .cloned()
        .ok_or(Ar16Error::MissingAdvice { level, gate: Some(lhs_gate) })?;
    let pk_cj_s = advice
        .get_public_key(&AdviceRole::LiftedCipher { gate: rhs_gate })
        .cloned()
        .ok_or(Ar16Error::MissingAdvice { level, gate: Some(rhs_gate) })?;

    let scalar_ui = lhs_pk.a.clone();
    let scalar_uj = rhs_pk.a.clone();
    let product = scalar_ui.clone() * scalar_uj.clone();

    let mut result = pk_s2.scale(&product);
    result -= &pk_ci_s.scale(&scalar_uj);
    result -= &pk_cj_s.scale(&scalar_ui);
    Ok(result)
}

pub fn combine_encodings<P: Poly>(
    level: usize,
    advice: &AdviceSet<P>,
    lhs_gate: GateId,
    rhs_gate: GateId,
    lhs_child: &AR16Encoding<P>,
    rhs_child: &AR16Encoding<P>,
    lhs_pk: &AR16PublicKey<P>,
    rhs_pk: &AR16PublicKey<P>,
) -> Result<AR16Encoding<P>, Ar16Error> {
    let enc_s2 = advice
        .get_encoding(&AdviceRole::SecretSquare)
        .cloned()
        .ok_or(Ar16Error::MissingAdvice { level, gate: None })?;
    let enc_s_plain = advice.get_encoding(&AdviceRole::SecretPlain).cloned();
    let enc_ci_s = advice.get_encoding(&AdviceRole::LiftedCipher { gate: lhs_gate }).cloned();
    let enc_cj_s = advice.get_encoding(&AdviceRole::LiftedCipher { gate: rhs_gate }).cloned();
    let enc_ci_plain = advice
        .get_encoding(&AdviceRole::LiftedPlain { gate: lhs_gate })
        .cloned()
        .ok_or(Ar16Error::MissingAdvice { level, gate: Some(lhs_gate) })?;
    let enc_cj_plain = advice
        .get_encoding(&AdviceRole::LiftedPlain { gate: rhs_gate })
        .cloned()
        .ok_or(Ar16Error::MissingAdvice { level, gate: Some(rhs_gate) })?;

    let base = multiply_encodings(lhs_child, rhs_child);

    let enc_ci_s = match enc_ci_s {
        Some(enc) => enc,
        None => {
            let secret_plain =
                enc_s_plain.clone().ok_or(Ar16Error::MissingAdvice { level, gate: None })?;
            multiply_encodings(&enc_ci_plain, &secret_plain)
        }
    };
    let enc_cj_s = match enc_cj_s {
        Some(enc) => enc,
        None => {
            let secret_plain =
                enc_s_plain.clone().ok_or(Ar16Error::MissingAdvice { level, gate: None })?;
            multiply_encodings(&enc_cj_plain, &secret_plain)
        }
    };

    let scalar_ui = lhs_pk.a.clone();
    let scalar_uj = rhs_pk.a.clone();
    let product = scalar_ui.clone() * scalar_uj.clone();

    let mut result = base;
    result += &enc_s2.scale(&product);
    result -= &enc_ci_s.scale(&scalar_uj);
    result -= &enc_cj_s.scale(&scalar_ui);

    Ok(result)
}
