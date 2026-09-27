//! Encrypted-seed material circuit shared by BGG preprocessing and evaluation.

use crate::{
    circuit::{BatchedWire, PolyCircuit},
    circuit_gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    },
    matrix::PolyMatrix,
    noise_refresh::{
        circuit_decrypt::build_refreshed_wire_digit_all_crt_decrypt,
        circuit_merge::build_refreshed_wire_digit_all_crt_merge,
        circuit_prg::build_goldreich_encrypted_seed_material_ranges,
    },
    poly::{Poly, PolyParams},
};
use std::sync::Arc;

/// Builds the exact material circuit used by both sides of one refresh.
///
/// Inputs are all encrypted seed ciphertext wires followed by the Ring-GSW
/// decryption key. Outputs are ordered by slot, CRT level, then gadget digit.
/// Each output is the decoded CBD error plus its decoded bit-decomposed mask.
pub fn build_noise_refresh_material_circuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    num_slots: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(seed_bits >= 5, "noise refresh requires at least five seed bits");
    assert!(v_bits > 0, "noise-refresh mask width must be positive");
    assert!(cbd_n > 0, "noise-refresh CBD parameter must be positive");
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    assert_eq!(num_slots, ring_dim, "noise-refresh slots must match the ring dimension");

    let mut circuit = ring_gsw.fresh_circuit();
    let seed_inputs = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let seed_wires =
        seed_inputs.iter().flat_map(RingGswCiphertext::sub_circuit_wires).collect::<Vec<_>>();
    let ciphertext_template = {
        let mut template_circuit = ring_gsw.fresh_circuit();
        RingGswCiphertext::input(ring_gsw.clone(), None, &mut template_circuit)
    };
    let ciphertext_wire_count =
        ciphertext_template.sub_circuit_wires().iter().map(|wire| wire.len()).sum::<usize>();
    let decrypt_sub_id = circuit.register_sub_circuit(
        build_refreshed_wire_digit_all_crt_decrypt::<P, A, M>(ring_gsw.clone(), v_bits),
    );
    let merge_sub_id = circuit
        .register_sub_circuit(build_refreshed_wire_digit_all_crt_merge::<P>(&ring_gsw.params));
    let log_base_q = ring_gsw.params.modulus_digits();
    let crt_depth = ring_gsw.params.to_crt().2;
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let mut outputs = Vec::with_capacity(num_slots * crt_depth * log_base_q);

    for slot in 0..num_slots {
        let mut by_crt_digit =
            vec![vec![BatchedWire::single(decryption_key); log_base_q]; crt_depth];
        for digit in 0..log_base_q {
            let error_start = digit.checked_mul(ring_dim).expect("error range overflow");
            let mask_ranges = (0..crt_depth)
                .map(|crt| {
                    let start = crt
                        .checked_mul(log_base_q)
                        .and_then(|value| value.checked_mul(mask_q_chunk_len))
                        .and_then(|value| value.checked_add(digit * mask_q_chunk_len))
                        .expect("mask range overflow");
                    (start, mask_q_chunk_len)
                })
                .collect::<Vec<_>>();
            let prg_sub_id = circuit.register_sub_circuit(
                build_goldreich_encrypted_seed_material_ranges::<P, A>(
                    ring_gsw.clone(),
                    seed_bits,
                    v_bits,
                    graph_seed,
                    cbd_n,
                    slot,
                    error_start,
                    ring_dim,
                    &mask_ranges,
                ),
            );
            let flattened = circuit
                .call_sub_circuit(prg_sub_id, seed_wires.clone())
                .into_iter()
                .flat_map(|wire| (0..wire.len()).map(move |index| wire.at(index)))
                .collect::<Vec<_>>();
            let material = flattened
                .chunks_exact(ciphertext_wire_count)
                .map(|chunk| {
                    RingGswCiphertext::from_sub_circuit_outputs(&ciphertext_template, chunk)
                })
                .collect::<Vec<_>>();
            assert_eq!(material.len(), ring_dim + crt_depth * mask_q_chunk_len);
            let (errors, masks) = material.split_at(ring_dim);
            let mut decrypt_inputs = vec![BatchedWire::single(decryption_key)];
            decrypt_inputs.extend(errors.iter().flat_map(RingGswCiphertext::sub_circuit_wires));
            decrypt_inputs.extend(masks.iter().flat_map(RingGswCiphertext::sub_circuit_wires));
            let decrypted = circuit.call_sub_circuit(decrypt_sub_id, decrypt_inputs);
            let merged = circuit.call_sub_circuit(merge_sub_id, decrypted);
            assert_eq!(merged.len(), crt_depth);
            for (crt, output) in merged.into_iter().enumerate() {
                by_crt_digit[crt][digit] = output;
            }
        }
        outputs.extend(by_crt_digit.into_iter().flatten());
    }
    circuit.output(outputs);
    circuit
}
