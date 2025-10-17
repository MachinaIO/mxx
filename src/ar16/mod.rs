//! AR16 evaluation helpers following ePrint 2016/361 §5.
//!
//! The module is organised into small submodules:
//! * [`advice`] – storage for the `C_k` advice encodings supplied by the encryptor.
//! * [`context`] – memoisation context shared across public-key and ciphertext evaluation.
//! * [`pk`] / [`encoding`] – implementations of `EvalPK` and `EvalCT`.
//! * [`quadratic`] – helpers for combining encodings at multiplication gates.
//! * [`error`] – crate-specific error reporting for missing advice or invalid circuits.

mod advice;
mod context;
mod encoding;
mod error;
mod pk;
mod quadratic;
mod types;

pub use advice::{Advice, AdviceKey};
pub use context::EvalContext;
pub use encoding::{eval_ar16_encodings, eval_ct_gate};
pub use error::Ar16Error;
pub use pk::{eval_ar16_public_keys, eval_pk_gate};
pub use quadratic::reconstruct_times_s;
pub use types::{AR16Encoding, AR16PublicKey, Level};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::{PolyCircuit, gate::GateId},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_bigint::BigUint;
    use std::collections::HashMap;

    fn const_poly(params: &DCRTPolyParams, value: u64) -> DCRTPoly {
        let ring_dim = params.ring_dimension() as usize;
        let coeff = BigUint::from(value);
        let slots = vec![coeff; ring_dim];
        DCRTPoly::from_biguints_eval(params, &slots)
    }

    fn zero_poly(params: &DCRTPolyParams) -> DCRTPoly {
        DCRTPoly::const_zero(params)
    }

    fn build_simple_circuit() -> (PolyCircuit<DCRTPoly>, GateId, GateId, GateId, GateId) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);
        let mul_gate = circuit.mul_gate(add_gate, inputs[1]);
        circuit.output(vec![mul_gate]);
        (circuit, inputs[0], inputs[1], add_gate, mul_gate)
    }

    #[test]
    fn test_ar16_eval_ct_pk() {
        let params = DCRTPolyParams::default();
        let secret = const_poly(&params, 3);
        let message_a = const_poly(&params, 5);
        let message_b = const_poly(&params, 7);
        let zero = zero_poly(&params);

        let (circuit, gate_a, gate_b, gate_add, gate_mul) = build_simple_circuit();

        let level_input = Level::new(1);
        let level_mul = Level::new(2);

        let pk_a = AR16PublicKey::new(level_input, zero.clone());
        let pk_b = AR16PublicKey::new(level_input, zero.clone());
        let enc_a = AR16Encoding::new(pk_a.clone(), message_a.clone());
        let enc_b = AR16Encoding::new(pk_b.clone(), message_b.clone());

        let mut input_keys: HashMap<GateId, AR16PublicKey<DCRTPoly>> = HashMap::new();
        input_keys.insert(gate_a, pk_a.clone());
        input_keys.insert(gate_b, pk_b.clone());

        let mut input_encodings: HashMap<GateId, AR16Encoding<DCRTPoly>> = HashMap::new();
        input_encodings.insert(gate_a, enc_a.clone());
        input_encodings.insert(gate_b, enc_b.clone());

        let addition_encoding = enc_a.clone() + enc_b.clone();
        let addition_message = addition_encoding.evaluate(&secret);
        let message_b_plain = enc_b.evaluate(&secret);

        let mut advice = Advice::new();
        let level_k = level_mul;
        let zero_label = AR16PublicKey::new(level_k, zero.clone());

        let s_squared = secret.clone() * secret.clone();
        advice.insert_ek(
            level_k,
            AdviceKey::S2(level_k),
            AR16Encoding::new(zero_label.clone(), s_squared),
        );

        let add_times_secret = addition_message.clone() * secret.clone();
        advice.insert_ek_times_s(
            level_k,
            AdviceKey::Msg(gate_add),
            AR16Encoding::new(zero_label.clone(), add_times_secret),
        );

        let rhs_times_secret = message_b_plain.clone() * secret.clone();
        advice.insert_ek_times_s(
            level_k,
            AdviceKey::Msg(gate_b),
            AR16Encoding::new(zero_label.clone(), rhs_times_secret),
        );

        advice.insert_ek(
            level_k,
            AdviceKey::Msg(gate_add),
            AR16Encoding::new(zero_label.clone(), addition_message.clone()),
        );
        advice.insert_ek(
            level_k,
            AdviceKey::Msg(gate_b),
            AR16Encoding::new(zero_label.clone(), message_b_plain.clone()),
        );

        let mut ctx = EvalContext::new(advice, params.clone());

        let pk_map = eval_ar16_public_keys(&circuit, &mut ctx, &input_keys).expect("eval pk");
        let ct_map = eval_ar16_encodings(&circuit, &mut ctx, &input_keys, &input_encodings)
            .expect("eval ct");

        let final_pk = pk_map.get(&gate_mul).expect("mul pk available");
        let final_ct = ct_map.get(&gate_mul).expect("mul ct available");

        let expected_message = (message_a + message_b.clone()) * message_b.clone();
        let decrypted = final_ct.evaluate(&secret);
        assert_eq!(decrypted, expected_message, "decrypted message should match expected value");

        let lhs = final_ct.body().clone();
        let rhs = final_pk.u().clone() * secret.clone() + expected_message.clone();
        assert_eq!(lhs, rhs, "Equation (5.1) should hold at the output gate");

        let add_pk = pk_map.get(&gate_add).expect("addition pk cached");
        let add_ct = ct_map.get(&gate_add).expect("addition ct cached");
        let add_rhs = add_pk.u().clone() * secret.clone() + addition_message.clone();
        assert_eq!(add_ct.body().clone(), add_rhs, "Equation (5.1) should hold at addition gate");

        for (gate, ct) in ctx.encoding_cache().iter() {
            if let Some(pk) = ctx.pk_cache().get(gate) {
                let gate_rhs = pk.u().clone() * secret.clone() + ct.evaluate(&secret);
                assert_eq!(
                    ct.body().clone(),
                    gate_rhs,
                    "Equation (5.1) must hold for cached gate {gate:?}"
                );
            }
        }

        assert_eq!(final_pk.level(), level_mul);
        assert_eq!(final_ct.level(), level_mul);
    }

    #[test]
    fn test_ar16_eval_pk_fails_when_advice_missing() {
        let params = DCRTPolyParams::default();
        let label_a = const_poly(&params, 2);
        let label_b = const_poly(&params, 3);

        let (circuit, gate_a, gate_b, _, _) = build_simple_circuit();

        let pk_a = AR16PublicKey::new(Level::new(1), label_a);
        let pk_b = AR16PublicKey::new(Level::new(1), label_b);

        let mut input_keys: HashMap<GateId, AR16PublicKey<DCRTPoly>> = HashMap::new();
        input_keys.insert(gate_a, pk_a);
        input_keys.insert(gate_b, pk_b);

        let advice = Advice::new();
        let mut ctx = EvalContext::new(advice, params.clone());

        let err = eval_ar16_public_keys(&circuit, &mut ctx, &input_keys).expect_err("should fail");
        assert!(matches!(err, Ar16Error::MissingAdvice { .. }));
    }
}
