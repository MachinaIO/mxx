use super::*;
use crate::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use std::str::FromStr;

const NOISY_ACTIVE_LEVELS: usize = ACTIVE_LEVELS;
const NOISY_CRT_BITS: usize = 24;
const NOISY_MUL_CRT_BITS: usize = 40;
const NOISY_MUL_P_MODULI_BITS: usize = 7;
const GPU_TEST_NUM_SLOTS: usize = 2;
const GPU_TEST_PARALLEL_GATES: Option<usize> = Some(400);

fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
    let (moduli, _, _) = params.to_crt();
    GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
}

fn create_gpu_test_context_with(
    circuit: &mut PolyCircuit<GpuDCRTPoly>,
    ring_dim: u32,
    num_slots: usize,
    active_levels: usize,
    crt_bits: usize,
    p_moduli_bits: usize,
    max_unused_muls: usize,
) -> (DCRTPolyParams, GpuDCRTPolyParams, Arc<RingGswContext<GpuDCRTPoly>>) {
    let cpu_params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, BASE_BITS);
    let gpu_params = gpu_params_from_cpu(&cpu_params);
    let ctx = Arc::new(RingGswContext::setup(
        circuit,
        &gpu_params,
        num_slots,
        p_moduli_bits,
        max_unused_muls,
        SCALE,
        Some(active_levels),
        None,
    ));
    (cpu_params, gpu_params, ctx)
}

fn gpu_poly_from_cpu(params: &GpuDCRTPolyParams, poly: &DCRTPoly) -> GpuDCRTPoly {
    GpuDCRTPoly::from_biguints(params, &poly.coeffs_biguints())
}

fn rounded_coeffs_mod_plaintext<P: Poly>(
    decrypted: &P,
    plaintext_modulus: u64,
    q_modulus: &BigUint,
) -> Vec<u64> {
    rounded_coeffs(decrypted, plaintext_modulus, q_modulus)
        .into_iter()
        .map(|coeff| coeff % plaintext_modulus)
        .collect()
}

fn sample_public_key_error(
    params: &DCRTPolyParams,
    width: usize,
    error_sigma: f64,
) -> Vec<DCRTPoly> {
    let uniform_sampler = DCRTPolyUniformSampler::new();
    uniform_sampler
        .sample_uniform(params, 1, width, DistType::GaussDist { sigma: error_sigma })
        .get_row(0)
}

fn centered_mod_distance(value: &BigUint, expected: &BigUint, modulus: &BigUint) -> BigUint {
    let diff = if value >= expected { value - expected } else { modulus - (expected - value) };
    let wrapped = modulus - &diff;
    diff.min(wrapped)
}

fn max_centered_decryption_error<P: Poly>(
    decrypted: &P,
    expected_plaintext: u64,
    plaintext_modulus: u64,
    q_modulus: &BigUint,
) -> BigUint {
    let scaled_expected_constant =
        (q_modulus / BigUint::from(plaintext_modulus)) * BigUint::from(expected_plaintext);
    decrypted
        .coeffs_biguints()
        .into_iter()
        .enumerate()
        .map(|(coeff_idx, coeff)| {
            let expected =
                if coeff_idx == 0 { scaled_expected_constant.clone() } else { BigUint::from(0u64) };
            centered_mod_distance(&coeff, &expected, q_modulus)
        })
        .max()
        .expect("decrypted polynomial must contain at least one coefficient")
}

fn q_over_two_p_threshold(q_modulus: &BigUint, plaintext_modulus: u64) -> BigUint {
    q_modulus / BigUint::from(2u64 * plaintext_modulus)
}

fn q_over_two_p_threshold_bigdecimal(q_modulus: &BigUint, plaintext_modulus: u64) -> BigDecimal {
    BigDecimal::from_str(&q_over_two_p_threshold(q_modulus, plaintext_modulus).to_string())
        .expect("q/(2p) threshold must parse as BigDecimal")
}

fn gpu_expected_coeffs(expected: u64) -> Vec<u64> {
    let mut coeffs = vec![0u64; GPU_TEST_NUM_SLOTS];
    coeffs[0] = expected;
    coeffs
}

#[sequential_test::sequential]
#[test]
fn test_ring_gsw_add_circuit_decrypts_to_expected_integer_sum_with_noisy_public_key() {
    let error_sigma = 4.0;
    let plaintext_modulus = 3u64;
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let (cpu_params, gpu_params, ctx) = create_gpu_test_context_with(
        &mut circuit,
        GPU_TEST_NUM_SLOTS as u32,
        GPU_TEST_NUM_SLOTS,
        NOISY_ACTIVE_LEVELS,
        NOISY_CRT_BITS,
        P_MODULI_BITS,
        DEFAULT_MAX_UNREDUCED_MULS,
    );
    let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let sum = lhs.add(&rhs, &mut circuit);
    let estimated_error = sum.estimate_decryption_error_norm(error_sigma);
    let q_modulus = active_q_modulus(ctx.nested_rns.as_ref());
    let threshold = q_over_two_p_threshold(&q_modulus, plaintext_modulus);
    assert!(
        estimated_error < q_over_two_p_threshold_bigdecimal(&q_modulus, plaintext_modulus),
        "estimated add decryption error {} must stay below q/(2p) {}",
        estimated_error,
        threshold
    );
    let wire_secret_key = circuit.input(1).at(0).as_single_wire();
    let decrypted_sum = sum.decrypt::<GpuDCRTPolyMatrix>(
        wire_secret_key,
        BigUint::from(plaintext_modulus),
        &mut circuit,
    );
    circuit.output(vec![decrypted_sum]);

    let secret_key = sample_secret_key(&cpu_params);
    let gpu_secret_key = gpu_poly_from_cpu(&gpu_params, &secret_key);
    let public_key_hash_key = sample_hash_key();
    let randomizer_hash_key = sample_hash_key();
    let public_key_error = sample_public_key_error(&cpu_params, lhs.width(), error_sigma);
    let public_key = sample_public_key(
        &cpu_params,
        lhs.width(),
        &secret_key,
        public_key_hash_key,
        b"ring_gsw_public_key_with_error",
        Some(&public_key_error),
    );

    let (x1, x2) = sample_binary_input_pair();
    let expected = (x1 + x2) % plaintext_modulus;
    let lhs_tag = format!("add_circuit_noisy_lhs_{x1}_{x2}");
    let rhs_tag = format!("add_circuit_noisy_rhs_{x1}_{x2}");
    let lhs_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x1,
        randomizer_hash_key,
        lhs_tag.as_bytes(),
    );
    let rhs_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x2,
        randomizer_hash_key,
        rhs_tag.as_bytes(),
    );

    let inputs = [
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &lhs_native,
            0,
            Some(ctx.active_levels),
        ),
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &rhs_native,
            0,
            Some(ctx.active_levels),
        ),
        vec![PolyVec::new(vec![gpu_secret_key.clone()])],
    ]
    .concat();
    let outputs = eval_outputs_with_parallel_gates(
        &gpu_params,
        GPU_TEST_NUM_SLOTS,
        &circuit,
        inputs,
        GPU_TEST_PARALLEL_GATES,
    );
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].len(), 1);
    let actual_error = max_centered_decryption_error(
        &outputs[0].as_slice()[0],
        expected,
        plaintext_modulus,
        &q_modulus,
    );
    assert!(
        actual_error < threshold,
        "actual add decryption error {} must stay below q/(2p) {}",
        actual_error,
        threshold
    );
    assert_eq!(
        rounded_coeffs_mod_plaintext(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
        gpu_expected_coeffs(expected),
        "Ring-GSW addition with noisy public key should decrypt in-circuit to the plaintext-modulus sum for sampled x1={x1}, x2={x2}"
    );
}

#[sequential_test::sequential]
#[test]
fn test_ring_gsw_sub_circuit_decrypts_to_expected_integer_difference_with_noisy_public_key() {
    let error_sigma = 4.0;
    let plaintext_modulus = 3u64;
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let (cpu_params, gpu_params, ctx) = create_gpu_test_context_with(
        &mut circuit,
        GPU_TEST_NUM_SLOTS as u32,
        GPU_TEST_NUM_SLOTS,
        NOISY_ACTIVE_LEVELS,
        NOISY_CRT_BITS,
        P_MODULI_BITS,
        DEFAULT_MAX_UNREDUCED_MULS,
    );
    let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let difference = lhs.sub(&rhs, &mut circuit);
    let estimated_error = difference.estimate_decryption_error_norm(error_sigma);
    let q_modulus = active_q_modulus(ctx.nested_rns.as_ref());
    let threshold = q_over_two_p_threshold(&q_modulus, plaintext_modulus);
    assert!(
        estimated_error < q_over_two_p_threshold_bigdecimal(&q_modulus, plaintext_modulus),
        "estimated sub decryption error {} must stay below q/(2p) {}",
        estimated_error,
        threshold
    );
    let wire_secret_key = circuit.input(1).at(0).as_single_wire();
    let decrypted_difference = difference.decrypt::<GpuDCRTPolyMatrix>(
        wire_secret_key,
        BigUint::from(plaintext_modulus),
        &mut circuit,
    );
    circuit.output(vec![decrypted_difference]);

    let secret_key = sample_secret_key(&cpu_params);
    let gpu_secret_key = gpu_poly_from_cpu(&gpu_params, &secret_key);
    let public_key_hash_key = sample_hash_key();
    let randomizer_hash_key = sample_hash_key();
    let public_key_error = sample_public_key_error(&cpu_params, lhs.width(), error_sigma);
    let public_key = sample_public_key(
        &cpu_params,
        lhs.width(),
        &secret_key,
        public_key_hash_key,
        b"ring_gsw_public_key_with_error",
        Some(&public_key_error),
    );

    let (x1, x2) = sample_binary_input_pair();
    let expected = (x1 + plaintext_modulus - x2) % plaintext_modulus;
    let lhs_tag = format!("sub_circuit_noisy_lhs_{x1}_{x2}");
    let rhs_tag = format!("sub_circuit_noisy_rhs_{x1}_{x2}");
    let lhs_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x1,
        randomizer_hash_key,
        lhs_tag.as_bytes(),
    );
    let rhs_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x2,
        randomizer_hash_key,
        rhs_tag.as_bytes(),
    );

    let inputs = [
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &lhs_native,
            0,
            Some(ctx.active_levels),
        ),
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &rhs_native,
            0,
            Some(ctx.active_levels),
        ),
        vec![PolyVec::new(vec![gpu_secret_key.clone()])],
    ]
    .concat();
    let outputs = eval_outputs_with_parallel_gates(
        &gpu_params,
        GPU_TEST_NUM_SLOTS,
        &circuit,
        inputs,
        GPU_TEST_PARALLEL_GATES,
    );
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].len(), 1);
    let actual_error = max_centered_decryption_error(
        &outputs[0].as_slice()[0],
        expected,
        plaintext_modulus,
        &q_modulus,
    );
    assert!(
        actual_error < threshold,
        "actual sub decryption error {} must stay below q/(2p) {}",
        actual_error,
        threshold
    );
    assert_eq!(
        rounded_coeffs_mod_plaintext(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
        gpu_expected_coeffs(expected),
        "Ring-GSW subtraction with noisy public key should decrypt in-circuit to the plaintext-modulus difference for sampled x1={x1}, x2={x2}"
    );
}

#[sequential_test::sequential]
#[test]
#[ignore = "This test is currently ignored because it is too slow"]
fn test_ring_gsw_mul_circuit_decrypts_to_expected_integer_product_with_noisy_public_key() {
    let error_sigma = 4.0;
    let plaintext_modulus = 2u64;
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let (cpu_params, gpu_params, ctx) = create_gpu_test_context_with(
        &mut circuit,
        GPU_TEST_NUM_SLOTS as u32,
        GPU_TEST_NUM_SLOTS,
        NOISY_ACTIVE_LEVELS,
        NOISY_MUL_CRT_BITS,
        NOISY_MUL_P_MODULI_BITS,
        DEFAULT_MAX_UNREDUCED_MULS,
    );
    let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let product = lhs.mul(&rhs, &mut circuit);
    let estimated_error = product.estimate_decryption_error_norm(error_sigma);
    let q_modulus = active_q_modulus(ctx.nested_rns.as_ref());
    let threshold = q_over_two_p_threshold(&q_modulus, plaintext_modulus);
    assert!(
        estimated_error < q_over_two_p_threshold_bigdecimal(&q_modulus, plaintext_modulus),
        "estimated mul decryption error {} must stay below q/(2p) {}",
        estimated_error,
        threshold
    );
    let wire_secret_key = circuit.input(1).at(0).as_single_wire();
    let decrypted_product = product.decrypt::<GpuDCRTPolyMatrix>(
        wire_secret_key,
        BigUint::from(plaintext_modulus),
        &mut circuit,
    );
    circuit.output(vec![decrypted_product]);

    let secret_key = sample_secret_key(&cpu_params);
    let gpu_secret_key = gpu_poly_from_cpu(&gpu_params, &secret_key);
    let public_key_hash_key = sample_hash_key();
    let randomizer_hash_key = sample_hash_key();
    let public_key_error = sample_public_key_error(&cpu_params, lhs.width(), error_sigma);
    let public_key = sample_public_key(
        &cpu_params,
        lhs.width(),
        &secret_key,
        public_key_hash_key,
        b"ring_gsw_public_key_with_error",
        Some(&public_key_error),
    );

    let (x1, x2) = sample_binary_input_pair();
    let expected = (x1 * x2) % plaintext_modulus;
    let lhs_tag = format!("mul_circuit_noisy_lhs_{x1}_{x2}");
    let rhs_tag = format!("mul_circuit_noisy_rhs_{x1}_{x2}");
    let lhs_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x1,
        randomizer_hash_key,
        lhs_tag.as_bytes(),
    );
    let rhs_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x2,
        randomizer_hash_key,
        rhs_tag.as_bytes(),
    );

    let inputs = [
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &lhs_native,
            0,
            Some(ctx.active_levels),
        ),
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &rhs_native,
            0,
            Some(ctx.active_levels),
        ),
        vec![PolyVec::new(vec![gpu_secret_key.clone()])],
    ]
    .concat();
    let outputs = eval_outputs_with_parallel_gates(
        &gpu_params,
        GPU_TEST_NUM_SLOTS,
        &circuit,
        inputs,
        GPU_TEST_PARALLEL_GATES,
    );
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].len(), 1);
    let actual_error = max_centered_decryption_error(
        &outputs[0].as_slice()[0],
        expected,
        plaintext_modulus,
        &q_modulus,
    );
    assert!(
        actual_error < threshold,
        "actual mul decryption error {} must stay below q/(2p) {}",
        actual_error,
        threshold
    );
    assert_eq!(
        rounded_coeffs_mod_plaintext(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
        gpu_expected_coeffs(expected),
        "Ring-GSW multiplication with noisy public key should decrypt in-circuit to the plaintext-modulus product for sampled x1={x1}, x2={x2}"
    );
}

#[sequential_test::sequential]
#[test]
#[ignore = "This test is currently ignored because it is too slow"]
fn test_ring_gsw_chained_mul_circuit_decrypts_to_expected_integer_product_with_noisy_public_key() {
    let error_sigma = 4.0;
    let plaintext_modulus = 2u64;
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let (cpu_params, gpu_params, ctx) = create_gpu_test_context_with(
        &mut circuit,
        GPU_TEST_NUM_SLOTS as u32,
        GPU_TEST_NUM_SLOTS,
        NOISY_ACTIVE_LEVELS,
        NOISY_MUL_CRT_BITS,
        NOISY_MUL_P_MODULI_BITS,
        DEFAULT_MAX_UNREDUCED_MULS,
    );
    let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let rhs1 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let rhs2 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let chained_product = lhs.mul(&rhs1, &mut circuit).mul(&rhs2, &mut circuit);
    let estimated_error = chained_product.estimate_decryption_error_norm(error_sigma);
    let q_modulus = active_q_modulus(ctx.nested_rns.as_ref());
    let threshold = q_over_two_p_threshold(&q_modulus, plaintext_modulus);
    assert!(
        estimated_error < q_over_two_p_threshold_bigdecimal(&q_modulus, plaintext_modulus),
        "estimated chained-mul decryption error {} must stay below q/(2p) {}",
        estimated_error,
        threshold
    );
    let wire_secret_key = circuit.input(1).at(0).as_single_wire();
    let decrypted_product = chained_product.decrypt::<GpuDCRTPolyMatrix>(
        wire_secret_key,
        BigUint::from(plaintext_modulus),
        &mut circuit,
    );
    circuit.output(vec![decrypted_product]);

    let secret_key = sample_secret_key(&cpu_params);
    let gpu_secret_key = gpu_poly_from_cpu(&gpu_params, &secret_key);
    let public_key_hash_key = sample_hash_key();
    let randomizer_hash_key = sample_hash_key();
    let public_key_error = sample_public_key_error(&cpu_params, lhs.width(), error_sigma);
    let public_key = sample_public_key(
        &cpu_params,
        lhs.width(),
        &secret_key,
        public_key_hash_key,
        b"ring_gsw_public_key_with_error",
        Some(&public_key_error),
    );

    let (x1, x2) = sample_binary_input_pair();
    let x3 = rand::rng().random_range(0..2u64);
    let expected = (x1 * x2 * x3) % plaintext_modulus;
    let lhs_tag = format!("chain_mul_circuit_noisy_lhs_{x1}_{x2}_{x3}");
    let rhs1_tag = format!("chain_mul_circuit_noisy_rhs1_{x1}_{x2}_{x3}");
    let rhs2_tag = format!("chain_mul_circuit_noisy_rhs2_{x1}_{x2}_{x3}");
    let lhs_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x1,
        randomizer_hash_key,
        lhs_tag.as_bytes(),
    );
    let rhs1_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x2,
        randomizer_hash_key,
        rhs1_tag.as_bytes(),
    );
    let rhs2_native = encrypt_plaintext_bit(
        &cpu_params,
        ctx.nested_rns.as_ref(),
        &public_key,
        x3,
        randomizer_hash_key,
        rhs2_tag.as_bytes(),
    );

    let inputs = [
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &lhs_native,
            0,
            Some(ctx.active_levels),
        ),
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &rhs1_native,
            0,
            Some(ctx.active_levels),
        ),
        ciphertext_inputs_from_native::<GpuDCRTPoly>(
            &gpu_params,
            ctx.nested_rns.as_ref(),
            &rhs2_native,
            0,
            Some(ctx.active_levels),
        ),
        vec![PolyVec::new(vec![gpu_secret_key.clone()])],
    ]
    .concat();
    let outputs = eval_outputs_with_parallel_gates(
        &gpu_params,
        GPU_TEST_NUM_SLOTS,
        &circuit,
        inputs,
        GPU_TEST_PARALLEL_GATES,
    );
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].len(), 1);
    let actual_error = max_centered_decryption_error(
        &outputs[0].as_slice()[0],
        expected,
        plaintext_modulus,
        &q_modulus,
    );
    assert!(
        actual_error < threshold,
        "actual chained-mul decryption error {} must stay below q/(2p) {}",
        actual_error,
        threshold
    );
    assert_eq!(
        rounded_coeffs_mod_plaintext(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
        gpu_expected_coeffs(expected),
        "Chained Ring-GSW multiplication with noisy public key should decrypt in-circuit to the plaintext-modulus product for sampled x1={x1}, x2={x2}, x3={x3}"
    );
}
