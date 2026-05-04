//! Ring-GSW decryption circuits for noise-refresh PRG material.
//!
//! This module owns only the ciphertext-to-polynomial decoding phase.  It accepts Ring-GSW
//! ciphertext inputs, decrypts coefficient ciphertexts with `decrypt_batch`, and emits slotwise
//! polynomial wires.  It intentionally does not add error and mask wires together; that merge step
//! lives in `circuit_merge` so tests and benchmarks can feed pre-decoded wires directly.

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        fhe_prg::goldreich::{decrypt_bit_decomposed_scalar_outputs, sum_gate_ids},
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use std::sync::Arc;

/// Decrypts one coefficient-level error polynomial.
///
/// The input layout is exactly `ring_dim` coefficient ciphertexts for one gadget digit.  A single
/// `decrypt_batch` call decrypts all coefficients and places the recovered coefficient values in
/// the corresponding slots of one output wire.
pub fn decrypt_error_coefficients_as_polynomial<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_coefficients: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    plaintext_modulus: BigUint,
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(!plaintext_modulus.is_zero(), "plaintext_modulus must be positive");
    let first_ctx: Arc<RingGswContext<P, A>> = encrypted_coefficients
        .first()
        .expect("at least one encrypted coefficient is required")
        .ctx
        .clone();
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    assert_eq!(
        encrypted_coefficients.len(),
        ring_dim,
        "encrypted coefficient count {} must match ring_dim {}",
        encrypted_coefficients.len(),
        ring_dim
    );

    let ciphertexts = encrypted_coefficients.iter().collect::<Vec<_>>();
    RingGswCiphertext::decrypt_batch::<M>(
        ciphertexts.as_slice(),
        decryption_key,
        plaintext_modulus,
        circuit,
    )
    .add_in_circuit(circuit)
}

/// Decrypts one bit-decomposed mask polynomial.
///
/// The input layout is `ring_dim * bit_size` ciphertexts ordered by coefficient, then bit index.
/// For each bit index, one `decrypt_batch` call decrypts the `ring_dim` coefficient ciphertexts and
/// places those coefficient values in one slotwise polynomial wire.  The bit-polynomial wires are
/// then summed.
///
/// Mask bits are decoded against the full coefficient modulus `q`, not against a CRT modulus
/// `q_i`.  Bit `j` uses plaintext modulus `q / 2^j`, so Ring-GSW decrypt contributes
/// `2^j * mask_j`.  The merge phase can then add this small unscaled perturbation to a
/// `q/q_i`-scaled error and rely on rounding to remove the mask.
pub fn decrypt_bit_decomposed_polynomial<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bits: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    plaintext_moduli: &[BigUint],
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let bit_size = plaintext_moduli.len();
    assert!(bit_size > 0, "bit_size must be positive");
    assert!(
        plaintext_moduli.iter().all(|modulus| !modulus.is_zero()),
        "all bit plaintext moduli must be positive"
    );
    let first_ctx: Arc<RingGswContext<P, A>> =
        encrypted_bits.first().expect("at least one encrypted bit is required").ctx.clone();
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    let expected_len = ring_dim.checked_mul(bit_size).expect("Ring-GSW bit chunk length overflow");
    assert_eq!(
        encrypted_bits.len(),
        expected_len,
        "encrypted bit chunk must equal ring_dim * bit_size"
    );
    let bit_terms = (0..bit_size)
        .map(|bit_idx| {
            let ciphertexts = (0..ring_dim)
                .map(|coeff_idx| &encrypted_bits[coeff_idx * bit_size + bit_idx])
                .collect::<Vec<_>>();
            RingGswCiphertext::decrypt_batch::<M>(
                ciphertexts.as_slice(),
                decryption_key,
                plaintext_moduli[bit_idx].clone(),
                circuit,
            )
            .add_in_circuit(circuit)
        })
        .collect::<Vec<_>>();
    sum_gate_ids(circuit, &bit_terms)
}

/// Decrypts one bit-decomposed scalar mask.
///
/// This is the coefficient-free counterpart of `decrypt_bit_decomposed_polynomial`: bit `j` is
/// decrypted with plaintext modulus `q / 2^j`, so Ring-GSW contributes `2^j * bit_j`, and the
/// returned wire is the ordinary binary sum of the encrypted mask bits.
pub fn decrypt_bit_decomposed_scalar<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bits: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    plaintext_moduli: &[BigUint],
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    decrypt_bit_decomposed_scalar_outputs::<P, A, M>(
        circuit,
        encrypted_bits,
        decryption_key,
        plaintext_moduli,
    )
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
        decrypted_errors.push(decrypt_error_coefficients_as_polynomial::<P, A, M>(
            &mut circuit,
            &errors[error_start..error_start + ring_dim],
            decryption_key,
            plaintext_modulus.clone(),
        ));

        let mask_start = digit_idx * mask_q_chunk_len;
        decrypted_masks.push(decrypt_bit_decomposed_polynomial::<P, A, M>(
            &mut circuit,
            &masks[mask_start..mask_start + mask_q_chunk_len],
            decryption_key,
            &mask_plaintext_moduli,
        ));
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
        decrypted_errors.push(decrypt_error_coefficients_as_polynomial::<P, A, M>(
            &mut circuit,
            &errors,
            decryption_key,
            BigUint::from(q_i),
        ));

        let mask_start =
            crt_idx.checked_mul(mask_q_chunk_len).expect("CRT mask slice start overflow");
        let mask_end =
            mask_start.checked_add(mask_q_chunk_len).expect("CRT mask slice end overflow");
        decrypted_masks.push(decrypt_bit_decomposed_polynomial::<P, A, M>(
            &mut circuit,
            &masks[mask_start..mask_end],
            decryption_key,
            &mask_plaintext_moduli,
        ));
    }

    circuit.output(decrypted_errors.iter().chain(decrypted_masks.iter()).copied());
    circuit
}

/// Derives mask bit plaintext moduli from the full coefficient modulus.
///
/// Bit indices are zero-based: bit `0` uses plaintext modulus `q`, bit `1` uses `q/2`, and so on.
/// With Ring-GSW decrypt's `q/plaintext_modulus` scaling, the decoded mask polynomial is therefore
/// the ordinary binary sum `sum_j 2^j * mask_j`.
pub fn mask_plaintext_moduli_from_full_modulus(
    full_modulus: &BigUint,
    bit_size: usize,
) -> Vec<BigUint> {
    assert!(bit_size > 0, "bit_size must be positive");
    assert!(!full_modulus.is_zero(), "full modulus must be positive");
    (0..bit_size)
        .map(|bit_idx| {
            let modulus = full_modulus >> bit_idx;
            assert!(
                !modulus.is_zero(),
                "full modulus / 2^{bit_idx} must be positive for Ring-GSW decrypt"
            );
            modulus
        })
        .collect()
}
