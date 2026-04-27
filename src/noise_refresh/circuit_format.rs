//! Formatting and evaluation helpers for noise-refresh PRG material.
//!
//! This module intentionally treats PRG material as ordinary circuit inputs once it reaches the
//! formatting phase.  The decrypt/combine circuits below do not care whether the supplied error
//! and mask ciphertexts were produced by the real Goldreich PRG, by a benchmark shortcut, or by a
//! caller-provided fixture.  They only require the same layout and enough ciphertexts to run the
//! Ring-GSW decrypt and CRT combination logic.

use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
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

/// Flattens Ring-GSW ciphertexts into the wire list expected by a subcircuit call.
///
/// This is the only place in the formatter where ciphertext grouping is intentionally erased.  All
/// layout checks happen before or after this flattening step.
fn ciphertext_wires<P, A>(ciphertexts: &[RingGswCiphertext<P, A>]) -> Vec<BatchedWire>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    ciphertexts.iter().flat_map(|ciphertext| ciphertext.sub_circuit_wires()).collect()
}

/// Adds a nonempty list of scalar wires with the circuit's ordinary addition gate.
fn sum_gate_ids<P: Poly>(circuit: &mut PolyCircuit<P>, values: &[GateId]) -> GateId {
    let (first, rest) = values.split_first().expect("at least one gate is required");
    rest.iter().fold(*first, |acc, value| circuit.add_gate(acc, *value).as_single_wire())
}

/// Decrypts one coefficient-level error polynomial.
///
/// The input layout is exactly `ring_dim` coefficient ciphertexts for one gadget digit.  A single
/// `decrypt_batch` call decrypts all coefficients and places the recovered coefficient values in
/// the corresponding slots of one output wire.  There is no second slot-chunk axis here: this wire
/// is the error polynomial that will be added to one BGG+ encoding column.
fn decrypt_error_coefficients_as_polynomial<P, A, M>(
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
}

/// Decrypts one bit-decomposed mask polynomial.
///
/// The input layout is `ring_dim * bit_size` ciphertexts ordered by coefficient, then bit index.
/// For each bit index, one `decrypt_batch` call decrypts the `ring_dim` coefficient ciphertexts and
/// places those coefficient values in one slotwise polynomial wire.  The bit-polynomial wires are
/// then summed.  This matches the refresh layout: one mask polynomial is needed per gadget digit
/// and CRT level, not one extra polynomial per output slot.
///
/// The plaintext moduli are supplied by the caller instead of recomputed here.  Mask bits are not
/// decoded at the CRT modulus `q_i`: they are decoded against the full coefficient modulus `q`.
/// Bit `j` uses plaintext modulus `q / 2^j`, so Ring-GSW decrypt contributes
/// `2^j * mask_j` to the output coefficient.  This keeps the decoded mask as a small unscaled
/// perturbation.  When the caller later rounds a `q/q_i`-scaled output by `q_i/q`, any mask below
/// `q/(2*q_i)` disappears while the CRT-scaled error survives.
fn decrypt_bit_decomposed_polynomial<P, A, M>(
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
        })
        .collect::<Vec<_>>();
    sum_gate_ids(circuit, &bit_terms)
}

/// Builds one CRT-specific formatter subcircuit for one refreshed wire.
///
/// Inputs are all error ciphertexts for one refresh wire and only the mask ciphertexts for the
/// selected CRT level.  Outputs are `log_base_q` decoded refresh wires, each equal to decrypted
/// error plus decrypted mask.  The wire being refreshed is intentionally not part of this circuit;
/// callers add these refresh wires to whatever encoding column they are refreshing.
pub fn build_refreshed_wire_crt_formatter_subcircuit<P, A, M>(
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
    let decryption_key = GateId(0);
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

    let mut decoded_refresh_outputs = Vec::with_capacity(log_base_q);
    for digit_idx in 0..log_base_q {
        let error_start = digit_idx * ring_dim;
        let digit_errors = &errors[error_start..error_start + ring_dim];
        let decrypted_error = decrypt_error_coefficients_as_polynomial::<P, A, M>(
            &mut circuit,
            digit_errors,
            decryption_key,
            plaintext_modulus.clone(),
        );

        let mask_start = digit_idx * mask_q_chunk_len;
        let digit_masks = &masks[mask_start..mask_start + mask_q_chunk_len];
        let decrypted_mask = decrypt_bit_decomposed_polynomial::<P, A, M>(
            &mut circuit,
            digit_masks,
            decryption_key,
            &mask_plaintext_moduli,
        );

        decoded_refresh_outputs
            .push(circuit.add_gate(decrypted_error, decrypted_mask).as_single_wire());
    }

    circuit.output(decoded_refresh_outputs.iter().copied());
    circuit
}

/// Builds one all-CRT formatter subcircuit for one refreshed wire.
///
/// `PolyParams::to_crt()` owns the cached CRT modulus list for the polynomial parameters.  This
/// helper reads that list once, registers one CRT-specific chunk formatter per `q_i`, calls each
/// formatter with the shared error inputs and the CRT-specific mask slice, and exposes the
/// concatenated child outputs as its own outputs.
pub fn build_refreshed_wire_all_crt_formatter<P, A, M>(
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
    assert_eq!(
        q_moduli_depth,
        ring_gsw.params.to_crt().2,
        "Ring-GSW CRT depth must match params.to_crt()"
    );

    let mut circuit = ring_gsw.fresh_circuit();
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    let log_base_q = ring_gsw.params.modulus_digits();
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let errors = (0..log_base_q * ring_dim)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let masks = (0..q_moduli_depth * log_base_q * mask_q_chunk_len)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();

    let formatter_subcircuit_ids = q_moduli
        .iter()
        .map(|&q_i| {
            let formatter = build_refreshed_wire_crt_formatter_subcircuit::<P, A, M>(
                ring_gsw.clone(),
                v_bits,
                BigUint::from(q_i),
            );
            circuit.register_sub_circuit(formatter)
        })
        .collect::<Vec<_>>();

    let error_inputs = ciphertext_wires::<P, A>(&errors);
    let mut outputs =
        Vec::with_capacity(q_moduli_depth.checked_mul(log_base_q).expect("output overflow"));
    for crt_idx in 0..q_moduli_depth {
        let mask_start = crt_idx
            .checked_mul(log_base_q)
            .and_then(|value| value.checked_mul(mask_q_chunk_len))
            .expect("CRT mask slice start overflow");
        let mask_end = mask_start
            .checked_add(log_base_q.checked_mul(mask_q_chunk_len).expect("CRT mask slice overflow"))
            .expect("CRT mask slice end overflow");
        let mut formatter_inputs = Vec::with_capacity(
            error_inputs.len() +
                log_base_q.checked_mul(mask_q_chunk_len).expect("CRT mask input count overflow"),
        );
        formatter_inputs.extend(error_inputs.iter().copied());
        formatter_inputs.extend(ciphertext_wires::<P, A>(&masks[mask_start..mask_end]));
        outputs
            .extend(circuit.call_sub_circuit(formatter_subcircuit_ids[crt_idx], formatter_inputs));
    }
    circuit.output(outputs);
    circuit
}

/// Derives mask bit plaintext moduli from the full coefficient modulus.
///
/// Bit indices are zero-based: bit `0` uses plaintext modulus `q`, bit `1` uses `q/2`, and so on.
/// With Ring-GSW decrypt's `q/plaintext_modulus` scaling, the decoded mask polynomial is therefore
/// the ordinary binary sum `sum_j 2^j * mask_j`.  This is intentionally independent of the CRT
/// level currently being formatted.
fn mask_plaintext_moduli_from_full_modulus(
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
