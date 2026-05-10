//! Merge circuits for decoded noise-refresh material.
//!
//! The merge phase starts after Ring-GSW decryption has already produced slotwise polynomial wires.
//! It combines decoded error and mask wires by ordinary circuit addition.  Keeping this phase
//! separate lets tests and benchmarks reuse one decoded mask wire many times without repeatedly
//! evaluating the expensive Ring-GSW decrypt path.

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    poly::{Poly, PolyParams},
};

/// Builds a merge subcircuit for `value_count` decoded error/mask pairs.
///
/// Inputs are ordered as all decoded error wires followed by all decoded mask wires.  Output `i`
/// is `decoded_errors[i] + decoded_masks[i]`.
pub fn build_refreshed_wire_merge_subcircuit<P: Poly>(value_count: usize) -> PolyCircuit<P> {
    assert!(value_count > 0, "value_count must be positive");
    let mut circuit = PolyCircuit::<P>::new();
    // Every decoded error or mask is passed as its own one-wire input. The
    // `at(0)` below selects the sole wire of each input object; `value_count`
    // counts how many such inputs exist, not slots inside a batched input.
    let decoded_errors =
        (0..value_count).map(|_| circuit.input(1).at(0).as_single_wire()).collect::<Vec<GateId>>();
    let decoded_masks =
        (0..value_count).map(|_| circuit.input(1).at(0).as_single_wire()).collect::<Vec<GateId>>();
    let outputs = decoded_errors
        .iter()
        .zip(decoded_masks.iter())
        .map(|(&error, &mask)| circuit.add_gate(error, mask).as_single_wire())
        .collect::<Vec<_>>();
    circuit.output(outputs);
    circuit
}

/// Builds the all-CRT merge circuit for one gadget digit of one refreshed wire.
///
/// The CRT depth is read from `params.to_crt()`.  The resulting circuit accepts one decoded error
/// wire per CRT level and one decoded mask wire per CRT level, then emits the element-wise sums.
pub fn build_refreshed_wire_digit_all_crt_merge<P>(params: &P::Params) -> PolyCircuit<P>
where
    P: Poly,
{
    let (_q_moduli, _crt_bits, crt_depth) = params.to_crt();
    build_refreshed_wire_merge_subcircuit::<P>(crt_depth)
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::build_refreshed_wire_digit_all_crt_merge;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::fhe_prg::goldreich::{GoldreichGraph, GoldreichGraphGeneration},
        lookup::poly_vec::PolyVecPltEvaluator,
        noise_refresh::circuit_prg::derive_noise_refresh_graph_seed,
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
                params::DCRTPolyParams,
            },
        },
    };
    use num_bigint::BigUint;
    use num_traits::Zero;

    const RING_DIM: u32 = 2;
    const NUM_SLOTS: usize = RING_DIM as usize;
    const CRT_DEPTH: usize = 2;
    const CRT_BITS: usize = 10;
    const BASE_BITS: u32 = 5;

    fn evaluate_plaintext_graph(graph: &GoldreichGraph, input_bits: &[u64]) -> Vec<u64> {
        graph
            .edges
            .iter()
            .map(|edge| {
                let xor_part = edge
                    .xor_inputs
                    .iter()
                    .fold(0u64, |acc, &input_idx| acc ^ input_bits[input_idx]);
                let and_part = input_bits[edge.and_inputs[0]] & input_bits[edge.and_inputs[1]];
                xor_part ^ and_part
            })
            .collect()
    }

    fn evaluate_plaintext_cbd_prf(
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
        input_bits: &[u64],
        cbd_n: usize,
    ) -> Vec<i64> {
        assert!(cbd_n > 0, "cbd_n must be positive");
        (0..2 * cbd_n)
            .map(|sample_idx| {
                let mut seed = derive_noise_refresh_graph_seed(
                    graph_seed,
                    b"NoiseRefreshMergeTestCBD/v1",
                    sample_idx as u64,
                );
                seed[0] ^= sample_idx as u8;
                GoldreichGraph::generate(
                    input_size,
                    output_size,
                    seed,
                    GoldreichGraphGeneration::default(),
                )
            })
            .collect::<Vec<_>>()
            .chunks(2 * cbd_n)
            .flat_map(|graphs| {
                let uniform_samples = graphs
                    .iter()
                    .map(|graph| evaluate_plaintext_graph(graph, input_bits))
                    .collect::<Vec<_>>();
                (0..output_size)
                    .map(|output_idx| {
                        let positive = uniform_samples[..cbd_n]
                            .iter()
                            .map(|sample| sample[output_idx] as i64)
                            .sum::<i64>();
                        let negative = uniform_samples[cbd_n..]
                            .iter()
                            .map(|sample| sample[output_idx] as i64)
                            .sum::<i64>();
                        positive - negative
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn slot_constant_poly(params: &GpuDCRTPolyParams, value: BigUint) -> GpuDCRTPoly {
        let ring_dim = params.ring_dimension() as usize;
        let mut coeffs = vec![BigUint::zero(); ring_dim];
        coeffs[0] = value;
        GpuDCRTPoly::from_biguints(params, &coeffs)
    }

    fn slotwise_poly_vec(params: &GpuDCRTPolyParams, coeffs: &[BigUint]) -> PolyVec<GpuDCRTPoly> {
        assert_eq!(coeffs.len(), params.ring_dimension() as usize);
        PolyVec::new(
            coeffs
                .iter()
                .cloned()
                .map(|coeff| slot_constant_poly(params, coeff))
                .collect::<Vec<_>>(),
        )
    }

    fn output_poly_from_slots(
        params: &GpuDCRTPolyParams,
        output: &PolyVec<GpuDCRTPoly>,
    ) -> GpuDCRTPoly {
        assert_eq!(
            output.len(),
            params.ring_dimension() as usize,
            "merge output must pack one coefficient per slot"
        );
        GpuDCRTPoly::from_biguints(
            params,
            &output
                .as_slice()
                .iter()
                .map(|slot_poly| {
                    slot_poly
                        .coeffs_biguints()
                        .into_iter()
                        .next()
                        .expect("slot polynomial must have a constant coefficient")
                })
                .collect::<Vec<_>>(),
        )
    }

    fn round_scaled_crt_residue(coeff: &BigUint, q_i: u64, q: &BigUint) -> BigUint {
        let half_q = q / BigUint::from(2u64);
        let rounded = (BigUint::from(q_i) * coeff + half_q) / q;
        rounded % BigUint::from(q_i)
    }

    fn centered_to_mod_q(value: i64, q: &BigUint) -> BigUint {
        if value >= 0 {
            BigUint::from(value as u64) % q
        } else {
            let magnitude = BigUint::from(value.unsigned_abs()) % q;
            if magnitude.is_zero() { BigUint::zero() } else { q - magnitude }
        }
    }

    fn build_shared_mask_merge_test_circuit(
        params: &GpuDCRTPolyParams,
    ) -> PolyCircuit<GpuDCRTPoly> {
        let (_q_moduli, _crt_bits, crt_depth) = params.to_crt();
        let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
        let decoded_errors =
            (0..crt_depth).map(|_| circuit.input(1).at(0).as_single_wire()).collect::<Vec<_>>();
        let shared_mask = circuit.input(1).at(0).as_single_wire();

        let merge = build_refreshed_wire_digit_all_crt_merge::<GpuDCRTPoly>(params);
        let merge_id = circuit.register_sub_circuit(merge);
        let mut merge_inputs = decoded_errors
            .iter()
            .map(|&wire| crate::circuit::BatchedWire::single(wire))
            .collect::<Vec<_>>();
        merge_inputs.extend(std::iter::repeat_n(
            crate::circuit::BatchedWire::single(shared_mask),
            crt_depth,
        ));
        let outputs = circuit.call_sub_circuit(merge_id, merge_inputs);
        circuit.output(outputs);
        circuit
    }

    #[sequential_test::sequential]
    #[test]
    fn merge_gpu_recovers_native_prg_errors_after_shared_mask_rounding() {
        let cpu_params = DCRTPolyParams::new(RING_DIM, CRT_DEPTH, CRT_BITS, BASE_BITS);
        let (q_moduli, _crt_bits, crt_depth) = cpu_params.to_crt();
        assert_eq!(crt_depth, CRT_DEPTH);
        let gpu_params = GpuDCRTPolyParams::new(RING_DIM, q_moduli.clone(), BASE_BITS);
        let q = gpu_params.modulus().as_ref().clone();
        let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
        let mask_bound = &q / BigUint::from(2u64) / BigUint::from(q_max);
        let shared_mask = BigUint::from(1u64);
        assert!(
            shared_mask < mask_bound,
            "shared test mask must be small enough to disappear after CRT rounding"
        );

        let seed_plaintexts = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let expected_errors = evaluate_plaintext_cbd_prf(
            seed_plaintexts.len(),
            gpu_params.ring_dimension() as usize,
            [9u8; 32],
            &seed_plaintexts,
            2,
        );
        assert_eq!(expected_errors.len(), gpu_params.ring_dimension() as usize);

        let circuit_inputs = q_moduli
            .iter()
            .map(|&q_i| {
                let scale = &q / BigUint::from(q_i);
                let coeffs = expected_errors
                    .iter()
                    .map(|&error| (centered_to_mod_q(error, &q) * &scale) % &q)
                    .collect::<Vec<_>>();
                slotwise_poly_vec(&gpu_params, &coeffs)
            })
            .chain(std::iter::once(slotwise_poly_vec(
                &gpu_params,
                &vec![shared_mask.clone(); NUM_SLOTS],
            )))
            .collect::<Vec<_>>();

        let merge_circuit = build_shared_mask_merge_test_circuit(&gpu_params);
        let one = PolyVec::new(vec![GpuDCRTPoly::const_one(&gpu_params); NUM_SLOTS]);
        let outputs = merge_circuit.eval(
            &gpu_params,
            one,
            circuit_inputs,
            None::<&PolyVecPltEvaluator>,
            None,
            Some(1),
        );
        assert_eq!(outputs.len(), crt_depth);

        let decoded_polys = outputs
            .iter()
            .map(|output| output_poly_from_slots(&gpu_params, output))
            .collect::<Vec<_>>();
        let reconst_coeffs = gpu_params.reconst_coeffs();
        let ring_dim = gpu_params.ring_dimension() as usize;
        for coeff_idx in 0..ring_dim {
            let mut reconstructed = BigUint::zero();
            for (crt_idx, &q_i) in q_moduli.iter().enumerate() {
                let output_coeffs = decoded_polys[crt_idx].coeffs_biguints();
                let residue = round_scaled_crt_residue(&output_coeffs[coeff_idx], q_i, &q);
                reconstructed += residue * &reconst_coeffs[crt_idx];
            }
            reconstructed %= &q;
            assert_eq!(
                reconstructed,
                centered_to_mod_q(expected_errors[coeff_idx], &q),
                "CRT-reconstructed merge output should recover native CBD error at coeff {coeff_idx}"
            );
        }
    }
}
