use mxx::{
    ar16::{
        AR16Encoding, AR16PublicKey, Ar16Error,
        advice::{AdviceRole, AdviceSet},
        eval_encoding,
        eval_pk::eval_public_key,
        state::EvalContext,
    },
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

fn build_simple_circuit() -> (PolyCircuit<DCRTPoly>, GateId, GateId, GateId) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(2);
    let add_gate = circuit.add_gate(inputs[0], inputs[1]);
    let mul_gate = circuit.mul_gate(add_gate, inputs[1]);
    circuit.output(vec![mul_gate]);
    (circuit, inputs[0], inputs[1], add_gate)
}

#[test]
fn test_eval_pk_and_encoding_matches_equation() {
    let params = DCRTPolyParams::default();
    let secret = const_poly(&params, 3);
    let message_a = const_poly(&params, 5);
    let message_b = const_poly(&params, 7);
    let zero_poly = DCRTPoly::const_zero(&params);
    let secret_enc = AR16Encoding::new(AR16PublicKey::new(zero_poly.clone()), secret.clone());

    let (circuit, gate_a, gate_b, gate_add) = build_simple_circuit();

    let pk_a = AR16PublicKey::new(zero_poly.clone());
    let pk_b = AR16PublicKey::new(zero_poly.clone());
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

    let s_squared_message = secret.clone() * secret.clone();
    let addition_times_secret_message = addition_message.clone() * secret.clone();
    let b_times_secret_message = message_b_plain.clone() * secret.clone();

    let mut advice_sets = vec![AdviceSet::<DCRTPoly>::new(0), AdviceSet::<DCRTPoly>::new(1)];
    let level1 = advice_sets.get_mut(1).expect("level 1 advice set exists");

    level1.insert_encoding(
        AdviceRole::SecretSquare,
        AR16Encoding::new(AR16PublicKey::new(zero_poly.clone()), s_squared_message.clone()),
    );
    level1.insert_public_key(AdviceRole::SecretSquare, AR16PublicKey::new(zero_poly.clone()));
    level1.insert_encoding(AdviceRole::SecretPlain, secret_enc.clone());
    level1.insert_public_key(AdviceRole::SecretPlain, AR16PublicKey::new(zero_poly.clone()));

    level1.insert_encoding(
        AdviceRole::LiftedCipher { gate: gate_add },
        AR16Encoding::new(
            AR16PublicKey::new(zero_poly.clone()),
            addition_times_secret_message.clone(),
        ),
    );
    level1.insert_public_key(
        AdviceRole::LiftedCipher { gate: gate_add },
        AR16PublicKey::new(zero_poly.clone()),
    );

    level1.insert_encoding(
        AdviceRole::LiftedCipher { gate: gate_b },
        AR16Encoding::new(AR16PublicKey::new(zero_poly.clone()), b_times_secret_message.clone()),
    );
    level1.insert_public_key(
        AdviceRole::LiftedCipher { gate: gate_b },
        AR16PublicKey::new(zero_poly.clone()),
    );

    level1.insert_encoding(
        AdviceRole::LiftedPlain { gate: gate_add },
        AR16Encoding::new(AR16PublicKey::new(zero_poly.clone()), addition_message.clone()),
    );
    level1.insert_public_key(
        AdviceRole::LiftedPlain { gate: gate_add },
        AR16PublicKey::new(zero_poly.clone()),
    );
    level1.insert_encoding(
        AdviceRole::LiftedPlain { gate: gate_b },
        AR16Encoding::new(AR16PublicKey::new(zero_poly.clone()), message_b_plain.clone()),
    );
    level1.insert_public_key(
        AdviceRole::LiftedPlain { gate: gate_b },
        AR16PublicKey::new(zero_poly.clone()),
    );

    let mut ctx = EvalContext::new(advice_sets);

    let pk_output = eval_public_key(&circuit, &mut ctx, &input_keys).expect("eval pk");
    let enc_output =
        eval_encoding(&circuit, &pk_output, &mut ctx, &input_encodings).expect("eval encoding");

    assert_eq!(pk_output.outputs.len(), 1);
    assert_eq!(enc_output.len(), 1);

    let final_pk = &pk_output.outputs[0];
    let final_ct = &enc_output[0];

    let expected_message = (message_a + message_b.clone()) * message_b.clone();
    let decrypted = final_ct.evaluate(&secret);
    assert_eq!(decrypted, expected_message, "decrypted message should match expected value");

    let lhs = final_ct.b.clone();
    let rhs = final_pk.a.clone() * secret.clone() + expected_message.clone();
    assert_eq!(lhs, rhs, "Equation (5.1) should hold in the zero-noise setting");

    assert_eq!(ctx.pk_cache().len(), 4);
    assert_eq!(ctx.encoding_cache().len(), 4);

    let pk_output_again = eval_public_key(&circuit, &mut ctx, &input_keys).expect("second run");
    assert_eq!(pk_output_again.outputs[0].a, final_pk.a);
}

#[test]
fn test_eval_public_key_missing_advice_errors() {
    let params = DCRTPolyParams::default();
    let label_a = const_poly(&params, 2);
    let label_b = const_poly(&params, 3);

    let (circuit, gate_a, gate_b, _) = build_simple_circuit();

    let pk_a = AR16PublicKey::new(label_a);
    let pk_b = AR16PublicKey::new(label_b);

    let mut input_keys: HashMap<GateId, AR16PublicKey<DCRTPoly>> = HashMap::new();
    input_keys.insert(gate_a, pk_a);
    input_keys.insert(gate_b, pk_b);

    let advice_sets = vec![AdviceSet::<DCRTPoly>::new(0), AdviceSet::<DCRTPoly>::new(1)];
    let mut ctx = EvalContext::new(advice_sets);

    let err = eval_public_key(&circuit, &mut ctx, &input_keys).expect_err("should fail");
    assert!(matches!(err, Ar16Error::MissingAdvice { .. }));
}
