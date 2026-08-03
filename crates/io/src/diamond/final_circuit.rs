use mxx_gadgets::{
    Poly,
    circuit::PolyCircuit,
    circuit_gadgets::{
        arith::NestedRnsPoly,
        fhe::{
            ring_gsw::{RingGswCiphertext, RingGswDecryptionParts},
            ring_gsw_nested_rns::NestedRnsRingGswContext,
        },
    },
    decoder::mask_circuit::{
        center_public_bottom, decrypt_bit_decomposed_polynomial_parts,
        mask_plaintext_moduli_from_full_modulus,
    },
};
use mxx_primitives::{matrix::PolyMatrix, poly::PolyParams};
use num_bigint::BigUint;
use std::sync::Arc;

pub(crate) fn build_final_mask_decrypt_circuit<P, M>(
    ring_gsw: Arc<NestedRnsRingGswContext<P>>,
    bit_size: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
{
    let mut circuit = ring_gsw.fresh_circuit();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let ring_dimension = ring_gsw.params.ring_dimension() as usize;
    let encrypted_bits = (0..ring_dimension * bit_size)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let modulus: Arc<BigUint> = ring_gsw.params.modulus().into();
    let plaintext_moduli = mask_plaintext_moduli_from_full_modulus(modulus.as_ref(), bit_size);
    let RingGswDecryptionParts { secret_dependent, public_bottom } =
        decrypt_bit_decomposed_polynomial_parts::<P, NestedRnsPoly<P>, M>(
            &mut circuit,
            &encrypted_bits,
            decryption_key,
            &plaintext_moduli,
        );
    let public_bottom =
        center_public_bottom(&mut circuit, &ring_gsw.params, public_bottom, bit_size);
    circuit.output([secret_dependent, public_bottom]);
    circuit
}

pub(crate) fn build_final_function_decrypt_circuit<P, M>(
    ring_gsw: Arc<NestedRnsRingGswContext<P>>,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
{
    let mut circuit = ring_gsw.fresh_circuit();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let encrypted_bit = RingGswCiphertext::input(ring_gsw, None, &mut circuit);
    let decrypted = RingGswCiphertext::decrypt_batch::<M>(
        &[&encrypted_bit],
        decryption_key,
        BigUint::from(2u8),
        &mut circuit,
    );
    circuit.output([decrypted.secret_dependent, decrypted.public_bottom]);
    circuit
}
