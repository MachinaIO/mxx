//! Mask-decrypt circuits shared by masked decoders.

use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::Zero;

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext, RingGswDecryptionParts},
        fhe_prg::goldreich::sum_gate_ids,
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};

/// Derives Ring-GSW plaintext moduli for bit-decomposed mask bits.
///
/// Bit `j` is decrypted with plaintext modulus `q / 2^j`, so the Ring-GSW
/// `q / t` scaling contributes the intended `2^j` factor.
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

/// Decrypts one coefficient-level error polynomial.
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

/// Appends a representative one-ciphertext-bit Ring-GSW decrypt circuit.
///
/// The output is the split decrypt result, `secret_dependent` followed by
/// `public_bottom`. Benchmark estimators use this as the unit contribution for
/// both error coefficients and bit-decomposed mask bits, then scale the number
/// of independent contributions separately.
pub(crate) fn append_one_ciphertext_bit_decrypt<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    ring_gsw: Arc<RingGswContext<P, A>>,
    plaintext_modulus: BigUint,
) -> RingGswDecryptionParts
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(!plaintext_modulus.is_zero(), "plaintext_modulus must be positive");
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let encrypted_bit = RingGswCiphertext::input(ring_gsw, Some(BigUint::from(1u64)), circuit);
    RingGswCiphertext::decrypt_batch::<M>(
        &[&encrypted_bit],
        decryption_key,
        plaintext_modulus,
        circuit,
    )
}

/// Builds a representative one-ciphertext-bit Ring-GSW decrypt circuit.
pub(crate) fn build_one_ciphertext_bit_decrypt_circuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    plaintext_modulus: BigUint,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let mut circuit = ring_gsw.fresh_circuit();
    let decrypted =
        append_one_ciphertext_bit_decrypt::<P, A, M>(&mut circuit, ring_gsw, plaintext_modulus);
    circuit.output(vec![decrypted.secret_dependent, decrypted.public_bottom]);
    circuit
}

/// Decrypts one bit-decomposed polynomial mask into split Ring-GSW branches.
pub fn decrypt_bit_decomposed_polynomial_parts<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bits: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    plaintext_moduli: &[BigUint],
) -> RingGswDecryptionParts
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
        })
        .collect::<Vec<_>>();
    let secret_dependent_terms =
        bit_terms.iter().map(|term| term.secret_dependent).collect::<Vec<_>>();
    let public_bottom_terms = bit_terms.iter().map(|term| term.public_bottom).collect::<Vec<_>>();
    RingGswDecryptionParts {
        secret_dependent: sum_gate_ids(circuit, &secret_dependent_terms),
        public_bottom: sum_gate_ids(circuit, &public_bottom_terms),
    }
}

/// Decrypts one centered bit-decomposed polynomial mask.
pub fn decrypt_centered_bit_decomposed_polynomial<P, A, M>(
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
    let first_ctx: Arc<RingGswContext<P, A>> =
        encrypted_bits.first().expect("at least one encrypted bit is required").ctx.clone();
    let decoded = decrypt_bit_decomposed_polynomial_parts::<P, A, M>(
        circuit,
        encrypted_bits,
        decryption_key,
        plaintext_moduli,
    )
    .add_in_circuit(circuit);
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    let midpoint = BigUint::from(1u64) << (bit_size - 1);
    let midpoint_poly = P::from_biguints(&first_ctx.params, &vec![midpoint; ring_dim]);
    let midpoint_gate = circuit.const_poly(&midpoint_poly);
    circuit.add_gate(decoded, midpoint_gate).as_single_wire()
}

/// Adds the public-bottom centering constant used by final masked decoders.
pub fn center_public_bottom<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    params: &P::Params,
    public_bottom: GateId,
    mask_bits: usize,
) -> GateId {
    assert!(mask_bits > 0, "mask_bits must be positive");
    let midpoint = BigUint::from(1u64) << (mask_bits - 1);
    let midpoint_poly = P::from_biguints(params, &vec![midpoint; params.ring_dimension() as usize]);
    let midpoint_gate = circuit.const_poly(&midpoint_poly);
    circuit.sub_gate(public_bottom, midpoint_gate).as_single_wire()
}
