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

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::build_refreshed_wire_all_crt_formatter;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::{
            arith::{DEFAULT_MAX_UNREDUCED_MULS, NestedRnsPoly, NestedRnsPolyContext},
            fhe::{
                ring_gsw::RingGswCiphertext,
                ring_gsw_nested_rns::{
                    NativeRingGswCiphertext, NestedRnsRingGswContext,
                    ciphertext_inputs_from_native, encrypt_plaintext_bit, sample_public_key,
                },
            },
            fhe_prg::goldreich::{GoldreichFheCbdPrg, GoldreichFhePrg, GoldreichGraph},
        },
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        noise_refresh::circuit_prg::{
            derive_noise_refresh_graph_seed, goldreich_noise_refresh_output_sizes,
        },
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
    use num_traits::{ToPrimitive, Zero};
    use std::sync::Arc;

    const RING_DIM: u32 = 1;
    const NUM_SLOTS: usize = RING_DIM as usize;
    const ACTIVE_LEVELS: usize = 2;
    const CRT_BITS: usize = 10;
    const BASE_BITS: u32 = 5;
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 4;

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

    fn encrypt_constant_for_formatter(
        cpu_params: &DCRTPolyParams,
        ring_gsw: &NestedRnsRingGswContext<GpuDCRTPoly>,
        public_key: &NativeRingGswCiphertext,
        plaintext: &BigUint,
        tag: &[u8],
    ) -> NativeRingGswCiphertext {
        let plaintext = plaintext.to_u64().expect("small formatter test plaintext must fit in u64");
        encrypt_plaintext_bit(
            cpu_params,
            ring_gsw.nested_rns.as_ref(),
            public_key,
            plaintext,
            [7u8; 32],
            tag,
        )
    }

    fn native_formatter_material(
        cpu_params: &DCRTPolyParams,
        gpu_params: &GpuDCRTPolyParams,
        ring_gsw: &Arc<NestedRnsRingGswContext<GpuDCRTPoly>>,
        public_key: &NativeRingGswCiphertext,
        seed_plaintexts: &[u64],
        graph_seed: [u8; 32],
        output_scope_idx: usize,
        v_bits: usize,
        cbd_n: usize,
    ) -> (Vec<NativeRingGswCiphertext>, Vec<i64>) {
        let (_q_moduli, _crt_bits, crt_depth) = gpu_params.to_crt();
        let output_sizes = goldreich_noise_refresh_output_sizes(
            gpu_params.ring_dimension() as usize,
            gpu_params.modulus_digits(),
            crt_depth,
            v_bits,
        );
        let mut graph_circuit = ring_gsw.fresh_circuit();
        let cbd_seed = derive_noise_refresh_graph_seed(
            graph_seed,
            b"NoiseRefreshCBD/v1",
            output_scope_idx as u64,
        );
        let cbd_prf = GoldreichFheCbdPrg::<
            GpuDCRTPoly,
            RingGswCiphertext<GpuDCRTPoly, NestedRnsPoly<GpuDCRTPoly>>,
        >::setup_range(
            &mut graph_circuit,
            ring_gsw.clone(),
            seed_plaintexts.len(),
            output_sizes.cbd_values,
            0,
            output_sizes.cbd_values,
            cbd_seed,
            cbd_n,
        );
        let expected_errors =
            evaluate_plaintext_cbd_prf(cbd_prf.uniform_graphs(), seed_plaintexts, cbd_n);
        assert_eq!(expected_errors.len(), output_sizes.cbd_values);

        let q: Arc<BigUint> = gpu_params.modulus().into();
        let mut material = expected_errors
            .iter()
            .enumerate()
            .map(|(idx, &error)| {
                let plaintext = centered_to_mod_q(error, &q);
                let tag = format!("noise_refresh_formatter_native_error_{idx}_{error}");
                encrypt_constant_for_formatter(
                    cpu_params,
                    ring_gsw.as_ref(),
                    public_key,
                    &plaintext,
                    tag.as_bytes(),
                )
            })
            .collect::<Vec<_>>();

        let mask_seed = derive_noise_refresh_graph_seed(
            graph_seed,
            b"NoiseRefreshMask/v1",
            output_scope_idx as u64,
        );
        let mask_prg = GoldreichFhePrg::<
            GpuDCRTPoly,
            RingGswCiphertext<GpuDCRTPoly, NestedRnsPoly<GpuDCRTPoly>>,
        >::setup_range(
            &mut graph_circuit,
            ring_gsw.clone(),
            seed_plaintexts.len(),
            output_sizes.mask_bits,
            0,
            output_sizes.mask_bits,
            mask_seed,
        );
        let mask_bits = evaluate_plaintext_graph(mask_prg.graph(), seed_plaintexts);
        assert_eq!(mask_bits.len(), output_sizes.mask_bits);
        material.extend(mask_bits.iter().enumerate().map(|(idx, &mask_bit)| {
            let tag = format!("noise_refresh_formatter_native_mask_{idx}_{mask_bit}");
            encrypt_constant_for_formatter(
                cpu_params,
                ring_gsw.as_ref(),
                public_key,
                &BigUint::from(mask_bit),
                tag.as_bytes(),
            )
        }));

        (material, expected_errors)
    }

    #[sequential_test::sequential]
    #[test]
    fn formatter_gpu_recovers_native_prg_errors_after_mask_rounding() {
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
        let v_bits = 1usize;
        assert!(
            v_bits <= max_safe_v_bits,
            "single-bit mask test must stay within the maximum safe mask bit length"
        );
        assert!(
            (BigUint::from(1u64) << v_bits) < mask_bound,
            "chosen v_bits must make every binary mask value smaller than q/(2*q_max)"
        );

        let formatter_circuit = build_refreshed_wire_all_crt_formatter::<
            GpuDCRTPoly,
            NestedRnsPoly<GpuDCRTPoly>,
            GpuDCRTPolyMatrix,
        >(ring_gsw.clone(), v_bits);
        let secret_key = DCRTPoly::const_one(&cpu_params);
        let public_key = sample_public_key(
            &cpu_params,
            ring_gsw.width(),
            &secret_key,
            [3u8; 32],
            b"noise_refresh_formatter_public_key",
            None,
        );

        let seed_plaintexts = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let graph_seed = [9u8; 32];
        let output_scope_idx = 0usize;
        let cbd_n = 2usize;
        let (native_material, expected_errors) = native_formatter_material(
            &cpu_params,
            &gpu_params,
            &ring_gsw,
            &public_key,
            &seed_plaintexts,
            graph_seed,
            output_scope_idx,
            v_bits,
            cbd_n,
        );
        let circuit_inputs = native_material
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
        let outputs = formatter_circuit.eval(
            &gpu_params,
            one,
            circuit_inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            Some(1),
        );
        assert_eq!(outputs.len(), crt_depth * gpu_params.modulus_digits());

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
                    "CRT-reconstructed formatter output should recover native CBD error at digit {digit_idx}, coeff {coeff_idx}"
                );
            }
        }
    }
}
