//! Ring-GSW decryption circuits for noise-refresh PRG material.
//!
//! This module owns only the ciphertext-to-polynomial decoding phase.  It accepts Ring-GSW
//! ciphertext inputs, decrypts coefficient ciphertexts with `decrypt_batch`, and emits slotwise
//! polynomial wires.  It intentionally does not add error and mask wires together; that merge step
//! lives in `circuit_merge` so tests and benchmarks can feed pre-decoded wires directly.

use crate::{
    circuit::PolyCircuit,
    decoder::mask_circuit::{
        decrypt_centered_bit_decomposed_polynomial, decrypt_error_coefficients_as_polynomial,
        mask_plaintext_moduli_from_full_modulus,
    },
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use std::sync::Arc;

fn decrypt_refreshed_wire_digit_crt<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    errors: &[RingGswCiphertext<P, A>],
    masks: &[RingGswCiphertext<P, A>],
    decryption_key: crate::circuit::gate::GateId,
    plaintext_modulus: BigUint,
    mask_plaintext_moduli: &[BigUint],
) -> (crate::circuit::gate::GateId, crate::circuit::gate::GateId)
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let decrypted_error = decrypt_error_coefficients_as_polynomial::<P, A, M>(
        circuit,
        errors,
        decryption_key,
        plaintext_modulus,
    );
    let decrypted_mask = decrypt_centered_bit_decomposed_polynomial::<P, A, M>(
        circuit,
        masks,
        decryption_key,
        mask_plaintext_moduli,
    );
    (decrypted_error, decrypted_mask)
}

/// Builds one CRT-specific decrypt subcircuit for one refreshed wire.
///
/// Inputs are all error ciphertexts for one refresh wire and only the mask ciphertexts for the
/// selected CRT level.  Outputs are ordered as all decrypted error wires followed by all decrypted
/// mask wires, one pair per gadget digit.
pub fn build_refreshed_wire_crt_decrypt_subcircuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    v_bits: usize,
    plaintext_modulus: BigUint,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(v_bits > 0, "v_bits must be positive");
    assert!(!plaintext_modulus.is_zero(), "CRT plaintext modulus must be positive");

    let mut circuit = ring_gsw.fresh_circuit();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    let log_base_q = ring_gsw.params.modulus_digits();
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let full_modulus = ring_gsw.params.modulus();
    let full_modulus: Arc<BigUint> = full_modulus.into();
    let mask_plaintext_moduli =
        mask_plaintext_moduli_from_full_modulus(full_modulus.as_ref(), v_bits);

    let errors = (0..log_base_q * ring_dim)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let masks = (0..log_base_q * mask_q_chunk_len)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();

    let mut decrypted_errors = Vec::with_capacity(log_base_q);
    let mut decrypted_masks = Vec::with_capacity(log_base_q);
    for digit_idx in 0..log_base_q {
        let error_start = digit_idx * ring_dim;
        let mask_start = digit_idx * mask_q_chunk_len;
        let (decrypted_error, decrypted_mask) = decrypt_refreshed_wire_digit_crt::<P, A, M>(
            &mut circuit,
            &errors[error_start..error_start + ring_dim],
            &masks[mask_start..mask_start + mask_q_chunk_len],
            decryption_key,
            plaintext_modulus.clone(),
            &mask_plaintext_moduli,
        );
        decrypted_errors.push(decrypted_error);
        decrypted_masks.push(decrypted_mask);
    }

    circuit.output(decrypted_errors.iter().chain(decrypted_masks.iter()).copied());
    circuit
}

/// Builds an all-CRT decrypt subcircuit for one gadget digit of one refreshed wire.
///
/// Inputs are `ring_dim` error coefficient ciphertexts followed by
/// `crt_depth * ring_dim * v_bits` mask bit ciphertexts.  Outputs are ordered as all CRT-level
/// decrypted error wires, followed by all CRT-level decrypted mask wires.
pub fn build_refreshed_wire_digit_all_crt_decrypt<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    v_bits: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(v_bits > 0, "v_bits must be positive");
    let (q_moduli, _crt_bits, q_moduli_depth) = ring_gsw.params.to_crt();

    let mut circuit = ring_gsw.fresh_circuit();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let full_modulus = ring_gsw.params.modulus();
    let full_modulus: Arc<BigUint> = full_modulus.into();
    let mask_plaintext_moduli =
        mask_plaintext_moduli_from_full_modulus(full_modulus.as_ref(), v_bits);

    let errors = (0..ring_dim)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let masks = (0..q_moduli_depth * mask_q_chunk_len)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();

    let mut decrypted_errors = Vec::with_capacity(q_moduli_depth);
    let mut decrypted_masks = Vec::with_capacity(q_moduli_depth);
    for (crt_idx, &q_i) in q_moduli.iter().enumerate() {
        let mask_start =
            crt_idx.checked_mul(mask_q_chunk_len).expect("CRT mask slice start overflow");
        let mask_end =
            mask_start.checked_add(mask_q_chunk_len).expect("CRT mask slice end overflow");
        let (decrypted_error, decrypted_mask) = decrypt_refreshed_wire_digit_crt::<P, A, M>(
            &mut circuit,
            &errors,
            &masks[mask_start..mask_end],
            decryption_key,
            BigUint::from(q_i),
            &mask_plaintext_moduli,
        );
        decrypted_errors.push(decrypted_error);
        decrypted_masks.push(decrypted_mask);
    }

    circuit.output(decrypted_errors.iter().chain(decrypted_masks.iter()).copied());
    circuit
}
