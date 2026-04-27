pub mod circuit_format;
pub mod circuit_prg;

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{
        circuit_format::build_refreshed_wire_all_crt_formatter,
        circuit_prg::{
            build_goldreich_encrypted_seeds_with_output, derive_noise_refresh_graph_seed,
            goldreich_noise_refresh_output_sizes,
        },
    };
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::{
            arith::{DEFAULT_MAX_UNREDUCED_MULS, NestedRnsPoly, NestedRnsPolyContext},
            fhe::{
                ring_gsw::RingGswCiphertext,
                ring_gsw_nested_rns::{
                    NestedRnsRingGswContext, ciphertext_inputs_from_native, encrypt_plaintext_bit,
                    sample_public_key,
                },
            },
            fhe_prg::goldreich::{GoldreichFheCbdPrg, GoldreichGraph},
        },
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
                params::DCRTPolyParams,
                poly::DCRTPoly,
            },
        },
        slot_transfer::PolyVecSlotTransferEvaluator,
    };
    use num_bigint::BigUint;
    use num_traits::Zero;
    use std::sync::Arc;

    const RING_DIM: u32 = 2;
    const NUM_SLOTS: usize = RING_DIM as usize;
    const ACTIVE_LEVELS: usize = 2;
    const CRT_BITS: usize = 20;
    const BASE_BITS: u32 = 10;
    const P_MODULI_BITS: usize = 7;
    const SCALE: u64 = 1 << 6;

    fn create_test_context(
        circuit: &mut PolyCircuit<GpuDCRTPoly>,
    ) -> (DCRTPolyParams, GpuDCRTPolyParams, Arc<NestedRnsRingGswContext<GpuDCRTPoly>>) {
        let cpu_params = DCRTPolyParams::new(RING_DIM, ACTIVE_LEVELS, CRT_BITS, BASE_BITS);
        let (q_moduli, _, _) = cpu_params.to_crt();
        let gpu_params = GpuDCRTPolyParams::new(RING_DIM, q_moduli, BASE_BITS);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &gpu_params,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            false,
            Some(ACTIVE_LEVELS),
        ));
        let ring_gsw = Arc::new(NestedRnsRingGswContext::from_arith_context(
            circuit,
            &gpu_params,
            NUM_SLOTS,
            nested_rns,
            Some(ACTIVE_LEVELS),
            None,
        ));
        (cpu_params, gpu_params, ring_gsw)
    }

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
        uniform_graphs: &[GoldreichGraph],
        input_bits: &[u64],
        cbd_n: usize,
    ) -> Vec<i64> {
        assert_eq!(uniform_graphs.len(), 2 * cbd_n);
        let uniform_samples = uniform_graphs
            .iter()
            .map(|graph| evaluate_plaintext_graph(graph, input_bits))
            .collect::<Vec<_>>();
        let output_size = uniform_samples[0].len();
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
            .collect()
    }

    fn output_poly_from_slots(
        params: &GpuDCRTPolyParams,
        output: &PolyVec<GpuDCRTPoly>,
    ) -> GpuDCRTPoly {
        assert_eq!(
            output.len(),
            params.ring_dimension() as usize,
            "formatter output must pack one coefficient per slot"
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

    #[sequential_test::sequential]
    #[test]
    fn goldreich_prg_material_formatter_recovers_native_errors_after_mask_rounding() {
        let mut setup_circuit = PolyCircuit::<GpuDCRTPoly>::new();
        let (cpu_params, gpu_params, ring_gsw) = create_test_context(&mut setup_circuit);
        let (q_moduli, _crt_bits, crt_depth) = gpu_params.to_crt();
        assert_eq!(crt_depth, ACTIVE_LEVELS);
        let q: Arc<BigUint> = gpu_params.modulus().into();
        let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
        let mask_bound = q.as_ref() / BigUint::from(2u64) / BigUint::from(q_max);
        let max_safe_v_bits = usize::try_from(mask_bound.bits().saturating_sub(1))
            .expect("mask bit length must fit in usize")
            .max(1);
        let v_bits = max_safe_v_bits.min(2);
        assert!(
            (BigUint::from(1u64) << v_bits) < mask_bound,
            "chosen v_bits must make every binary mask value smaller than q/(2*q_max)"
        );

        let seed_bits = 6usize;
        let cbd_n = 2usize;
        let output_scope_idx = 0usize;
        let graph_seed = [9u8; 32];
        let output_sizes = goldreich_noise_refresh_output_sizes(
            gpu_params.ring_dimension() as usize,
            gpu_params.modulus_digits(),
            crt_depth,
            v_bits,
        );

        let prg_circuit =
            build_goldreich_encrypted_seeds_with_output::<GpuDCRTPoly, NestedRnsPoly<GpuDCRTPoly>>(
                ring_gsw.clone(),
                seed_bits,
                v_bits,
                graph_seed,
                cbd_n,
                output_scope_idx,
            );
        let formatter_circuit = build_refreshed_wire_all_crt_formatter::<
            GpuDCRTPoly,
            NestedRnsPoly<GpuDCRTPoly>,
            GpuDCRTPolyMatrix,
        >(ring_gsw.clone(), v_bits);

        let mut circuit = ring_gsw.fresh_circuit();
        let encrypted_seed_inputs = (0..seed_bits)
            .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
            .collect::<Vec<_>>();
        let prg_subcircuit_id = circuit.register_sub_circuit(prg_circuit);
        let formatter_subcircuit_id = circuit.register_sub_circuit(formatter_circuit);
        let prg_inputs = encrypted_seed_inputs
            .iter()
            .flat_map(|ciphertext| ciphertext.sub_circuit_wires())
            .collect::<Vec<_>>();
        let prg_outputs = circuit.call_sub_circuit(prg_subcircuit_id, prg_inputs);
        let decoded_refresh_outputs =
            circuit.call_sub_circuit(formatter_subcircuit_id, prg_outputs);
        circuit.output(decoded_refresh_outputs);

        let secret_key = DCRTPoly::const_one(&cpu_params);
        let public_key = sample_public_key(
            &cpu_params,
            ring_gsw.width(),
            &secret_key,
            [3u8; 32],
            b"noise_refresh_formatter_public_key",
            None,
        );
        let seed_plaintexts = vec![1, 0, 1, 1, 0, 1];
        let native_seed_ciphertexts = seed_plaintexts
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, plaintext)| {
                let tag = format!("noise_refresh_formatter_seed_{idx}_{plaintext}");
                encrypt_plaintext_bit(
                    &cpu_params,
                    ring_gsw.nested_rns.as_ref(),
                    &public_key,
                    plaintext,
                    [7u8; 32],
                    tag.as_bytes(),
                )
            })
            .collect::<Vec<_>>();
        let circuit_inputs = native_seed_ciphertexts
            .iter()
            .flat_map(|ciphertext| {
                ciphertext_inputs_from_native(
                    &gpu_params,
                    ring_gsw.nested_rns.as_ref(),
                    ciphertext,
                    0,
                    Some(ring_gsw.active_levels),
                )
            })
            .collect::<Vec<_>>();
        let one = PolyVec::new(vec![GpuDCRTPoly::const_one(&gpu_params); NUM_SLOTS]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        let outputs = circuit.eval(
            &gpu_params,
            one,
            circuit_inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            Some(1),
        );
        assert_eq!(outputs.len(), crt_depth * gpu_params.modulus_digits());

        let cbd_seed = derive_noise_refresh_graph_seed(
            graph_seed,
            b"NoiseRefreshCBD/v1",
            output_scope_idx as u64,
        );
        let mut expected_circuit = ring_gsw.fresh_circuit();
        let cbd_prf = GoldreichFheCbdPrg::<
            GpuDCRTPoly,
            RingGswCiphertext<GpuDCRTPoly, NestedRnsPoly<GpuDCRTPoly>>,
        >::setup_range(
            &mut expected_circuit,
            ring_gsw.clone(),
            seed_bits,
            output_sizes.cbd_values,
            0,
            output_sizes.cbd_values,
            cbd_seed,
            cbd_n,
        );
        let expected_errors =
            evaluate_plaintext_cbd_prf(cbd_prf.uniform_graphs(), &seed_plaintexts, cbd_n);
        assert_eq!(
            expected_errors.len(),
            gpu_params.modulus_digits() * gpu_params.ring_dimension() as usize
        );

        let decoded_polys = outputs
            .iter()
            .map(|output| output_poly_from_slots(&gpu_params, output))
            .collect::<Vec<_>>();
        let reconst_coeffs = gpu_params.reconst_coeffs();
        let ring_dim = gpu_params.ring_dimension() as usize;
        for digit_idx in 0..gpu_params.modulus_digits() {
            for coeff_idx in 0..ring_dim {
                let mut reconstructed = BigUint::zero();
                for (crt_idx, &q_i) in q_moduli.iter().enumerate() {
                    let output_idx = crt_idx * gpu_params.modulus_digits() + digit_idx;
                    let output_coeffs = decoded_polys[output_idx].coeffs_biguints();
                    let residue = round_scaled_crt_residue(&output_coeffs[coeff_idx], q_i, &q);
                    reconstructed += residue * &reconst_coeffs[crt_idx];
                }
                reconstructed %= q.as_ref();
                let expected_idx = digit_idx * ring_dim + coeff_idx;
                assert_eq!(
                    reconstructed,
                    centered_to_mod_q(expected_errors[expected_idx], &q),
                    "CRT-reconstructed decoded refresh value should recover native CBD error at digit {digit_idx}, coeff {coeff_idx}"
                );
            }
        }
    }
}
