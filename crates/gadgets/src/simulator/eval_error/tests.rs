use super::*;
use crate::{
    circuit::PolyCircuit,
    lookup::PublicLut,
    poly::{
        Poly,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    simulator::{SimulatorContext, dependency_set::DependencySet},
    slot_transfer::SlotTransferEvaluator,
    utils::bigdecimal_bits_ceil,
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;

fn make_ctx() -> Arc<SimulatorContext> {
    // secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=(128/32)*7 = 28
    Arc::new(SimulatorContext::new(
        BigDecimal::from(1024u64), // ring_dim_sqrt
        BigDecimal::from(32u64),   // base
        2,
        28, // log_base_q
        3,  // log_base_q_small
    ))
}

fn assert_matrix_bound_eq(actual: &PolyMatrixNorm, expected: &PolyMatrixNorm) {
    assert_eq!(actual.nrow, expected.nrow);
    assert_eq!(actual.ncol, expected.ncol);
    assert_eq!(actual.ncol_sqrt, expected.ncol_sqrt);
    assert_eq!(actual.poly_norm, expected.poly_norm);
    assert_eq!(actual.zero_rows, expected.zero_rows);
}

fn assert_error_bound_eq(actual: &ErrorNorm, expected: &ErrorNorm) {
    assert_eq!(actual.plaintext_norm, expected.plaintext_norm);
    assert_matrix_bound_eq(&actual.matrix_norm, &expected.matrix_norm);
}

fn assert_error_bounds_eq(actual: &[ErrorNorm], expected: &[ErrorNorm]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_error_bound_eq(actual, expected);
    }
}

fn assert_error_bound_covers(actual: &ErrorNorm, expected: &ErrorNorm) {
    assert_eq!(actual.plaintext_norm, expected.plaintext_norm);
    assert_eq!(actual.matrix_norm.nrow, expected.matrix_norm.nrow);
    assert_eq!(actual.matrix_norm.ncol, expected.matrix_norm.ncol);
    assert_eq!(actual.matrix_norm.ncol_sqrt, expected.matrix_norm.ncol_sqrt);
    assert!(
        actual.matrix_norm.poly_norm.norm >= expected.matrix_norm.poly_norm.norm,
        "actual matrix norm {} does not cover expected {}",
        actual.matrix_norm.poly_norm.norm,
        expected.matrix_norm.poly_norm.norm
    );
}

fn assert_error_bounds_cover(actual: &[ErrorNorm], expected: &[ErrorNorm]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_error_bound_covers(actual, expected);
    }
}

fn simulate_max_error_norm_via_generic_eval_reference<P: AffinePltEvaluator>(
    circuit: &PolyCircuit<DCRTPoly>,
    ctx: Arc<SimulatorContext>,
    input_norm_bound: BigDecimal,
    input_size: usize,
    e_init_norm: &BigDecimal,
    plt_evaluator: Option<&P>,
) -> Vec<ErrorNorm> {
    let one_error = ErrorNorm::new(
        PolyNorm::one(ctx.clone()),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
    );
    let input_error = ErrorNorm::fresh_input(
        PolyNorm::constant(ctx.clone(), input_norm_bound),
        PolyMatrixNorm::fresh_random_with_norm(
            ctx,
            1,
            one_error.ctx().m_g,
            e_init_norm.clone(),
            None,
        ),
    );
    circuit.eval(&(), one_error, vec![input_error; input_size], plt_evaluator, None, None)
}

const E_B_SIGMA: f64 = 4.0;
const E_INIT_NORM: u32 = 26;

#[test]
fn test_poly_norm_sample_gauss_records_envelope_norm() {
    let ctx = make_ctx();
    let sigma = BigDecimal::from(4u64);
    let sampled = PolyNorm::sample_gauss(ctx, sigma.clone());

    assert_eq!(sampled.norm, BigDecimal::from(26u64));
    assert_eq!(sampled.maximum_coefficient_bound(), BigDecimal::from(26u64));
}

#[test]
fn test_compute_preimage_sigma_uses_optional_sigma() {
    let ctx = make_ctx();
    let default_norm =
        compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None, None);
    let explicit_default_norm =
        compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None, Some(4.578));
    let larger_sigma_norm =
        compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None, Some(6.0));

    assert_eq!(default_norm, explicit_default_norm);
    assert!(larger_sigma_norm > default_norm);
}

#[test]
fn test_constant_poly_norm_mul_skips_ring_dim_sqrt() {
    let ctx = make_ctx();
    let lhs = PolyNorm::constant(ctx.clone(), BigDecimal::from(3u64));
    let rhs = PolyNorm::constant(ctx.clone(), BigDecimal::from(5u64));
    let product = &lhs * &rhs;

    assert_eq!(product.sigma, BigDecimal::from(15u64));
    assert!(product.is_const_poly);

    let general = PolyNorm::new(ctx.clone(), BigDecimal::from(5u64));
    let mixed = &lhs * &general;
    assert_eq!(mixed.sigma, BigDecimal::from(15u64));
    assert!(!mixed.is_const_poly);

    let general_product = &general * &general;
    assert_eq!(general_product.sigma, BigDecimal::from(25u64) * &ctx.ring_dim_sqrt);
    assert!(!general_product.is_const_poly);
}

#[test]
fn test_sub_circuit_plaintext_range_accepts_constant_at_declared_max() {
    let ctx = make_ctx();
    let ranges = vec![SubCircuitInputMaxPlaintextNormRange::new(0, 1, BigUint::from(7u64))];
    let actual = vec![PolyNorm::constant(ctx.clone(), BigDecimal::from(7u64))];

    let normalized = validate_input_plaintext_norms_against_ranges(
        &ranges,
        &actual,
        &ctx,
        "constant plaintext range test",
    );

    assert_eq!(normalized.len(), 1);
    assert_eq!(normalized[0].sigma, BigDecimal::from(7u64));
    assert!(normalized[0].is_const_poly);
}

#[test]
#[should_panic(expected = "exceeds declared max")]
fn test_sub_circuit_plaintext_range_rejects_nonconstant_public_bound_above_declared_max() {
    let ctx = make_ctx();
    let ranges = vec![SubCircuitInputMaxPlaintextNormRange::new(0, 1, BigUint::from(12u64))];
    let actual = vec![PolyNorm::new(ctx.clone(), BigDecimal::from(13u64))];

    validate_input_plaintext_norms_against_ranges(
        &ranges,
        &actual,
        &ctx,
        "nonconstant plaintext range test",
    );
}

#[test]
fn test_wire_norm_addition() {
    let ctx = make_ctx();
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ins = circuit.input(2).to_vec();
    let out_gid = circuit.add_gate(ins[0], ins[1]);
    circuit.output(vec![out_gid]);
    let input_bound = BigDecimal::from(5u64);

    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        2,
        &BigDecimal::from(E_INIT_NORM),
        None::<&NormPltLWEEvaluator>,
        None,
    );
    assert_eq!(out.len(), 1);
    // Build expected from input wires and add them
    let in_wire = ErrorNorm::fresh_input(
        PolyNorm::constant(ctx.clone(), input_bound),
        PolyMatrixNorm::fresh_random_with_norm(
            ctx.clone(),
            1,
            ctx.m_g,
            BigDecimal::from(E_INIT_NORM),
            None,
        ),
    );
    let expected = &in_wire + &in_wire;
    assert_error_bound_eq(&out[0], &expected);
}

#[test]
fn test_wire_norm_subtraction() {
    let ctx = make_ctx();
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ins = circuit.input(2).to_vec();
    let out_gid = circuit.sub_gate(ins[0], ins[1]);
    circuit.output(vec![out_gid]);
    let input_bound = BigDecimal::from(5u64);
    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        2,
        &BigDecimal::from(E_INIT_NORM),
        None::<&NormPltLWEEvaluator>,
        None,
    );
    assert_eq!(out.len(), 1);
    let in_wire = ErrorNorm::fresh_input(
        PolyNorm::constant(ctx.clone(), input_bound),
        PolyMatrixNorm::fresh_random_with_norm(
            ctx.clone(),
            1,
            ctx.m_g,
            BigDecimal::from(E_INIT_NORM),
            None,
        ),
    );
    let expected = &in_wire - &in_wire; // subtraction bound equals addition bound
    assert_error_bound_eq(&out[0], &expected);
}

#[test]
fn test_wire_norm_multiplication() {
    // ctx: secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=28
    let ctx = make_ctx();
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ins = circuit.input(2).to_vec();
    let out_gid = circuit.mul_gate(ins[0], ins[1]);
    circuit.output(vec![out_gid]);
    let input_bound = BigDecimal::from(5u64);
    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        2,
        &BigDecimal::from(E_INIT_NORM),
        None::<&NormPltLWEEvaluator>,
        None,
    );
    assert_eq!(out.len(), 1);

    // Build expected = in_wire * in_wire
    let in_wire = ErrorNorm::fresh_input(
        PolyNorm::constant(ctx.clone(), input_bound),
        PolyMatrixNorm::fresh_random_with_norm(
            ctx.clone(),
            1,
            ctx.m_g,
            BigDecimal::from(E_INIT_NORM),
            None,
        ),
    );
    let expected = &in_wire * &in_wire;
    assert_error_bound_eq(&out[0], &expected);
}

#[test]
fn test_wire_norm_simulator_multiplication_matches_generic_eval() {
    let ctx = make_ctx();
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ins = circuit.input(2).to_vec();
    let out_gid = circuit.mul_gate(ins[0], ins[1]);
    circuit.output(vec![out_gid]);
    let input_bound = BigDecimal::from(5u64);
    let e_init_norm = BigDecimal::from(E_INIT_NORM);
    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        2,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
        None,
    );
    let generic = simulate_max_error_norm_via_generic_eval_reference(
        &circuit,
        ctx,
        input_bound,
        2,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
    );
    assert_error_bounds_eq(&out, &generic);
}

#[test]
fn test_wire_norm_simulator_mul_binary_tree_plaintext_one_matches_generic_eval() {
    let tree_height = 12usize;
    let input_size = 1usize << tree_height;
    let input_plaintext_norm = BigDecimal::one();
    let ring_dim_sqrt = BigDecimal::from(1u64 << 8);
    let base = BigDecimal::from(14u64);
    let secret_size = 1usize;
    let log_base_q = 2usize * 30;
    let log_base_q_small = 2usize;
    let e_init_norm = BigDecimal::from(E_INIT_NORM);
    let ctx = Arc::new(SimulatorContext::new(
        ring_dim_sqrt,
        base,
        secret_size,
        log_base_q,
        log_base_q_small,
    ));

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let mut current_level = circuit.input(input_size).to_vec();

    for _ in 0..tree_height {
        let mut next_level = Vec::with_capacity(current_level.len() / 2);
        for pair in current_level.chunks_exact(2) {
            next_level.push(circuit.mul_gate(pair[0], pair[1]).as_single_wire());
        }
        current_level = next_level;
    }

    assert_eq!(current_level.len(), 1);
    circuit.output(vec![current_level[0]]);

    let simulated = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_plaintext_norm.clone(),
        input_size,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
        None,
    );
    let generic = simulate_max_error_norm_via_generic_eval_reference(
        &circuit,
        ctx,
        input_plaintext_norm,
        input_size,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
    );

    assert_eq!(simulated.len(), 1);
    assert_error_bounds_eq(&simulated, &generic);
    println!(
        "mul_binary_tree_output_error_bits={}",
        bigdecimal_bits_ceil(&simulated[0].matrix_norm.poly_norm.sigma)
    );
}

#[test]
fn test_wire_norm_slot_transfer_matches_bgg_poly_encoding_bound() {
    let ctx = make_ctx();
    let e_b0_sigma = 11.0;
    let c_b0_error_norm = PolyMatrixNorm::sample_gauss(
        ctx.clone(),
        1,
        ctx.m_b,
        BigDecimal::from_f64(e_b0_sigma).unwrap(),
    );
    let evaluator = NormBggPolyEncodingSTEvaluator::new(
        ctx.clone(),
        e_b0_sigma,
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        None,
    );
    let input = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), BigDecimal::from(5u64)),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(7u64), None),
    );
    let src_slots = [(2, None), (0, Some(3)), (1, Some(2))];

    let out = evaluator.slot_transfer(&(), &input, &src_slots, GateId(0));

    let b0_preimage_sigma =
        compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(1), None);
    let s_vec = PolyMatrixNorm::new(ctx.clone(), 1, ctx.secret_size, BigDecimal::one(), None);
    let gate_preimage = PolyMatrixNorm::fresh_preimage(
        ctx.clone(),
        ctx.m_b,
        ctx.m_g,
        b0_preimage_sigma.clone(),
        None,
    );
    let gate_target_error = PolyMatrixNorm::sample_gauss(
        ctx.clone(),
        ctx.secret_size,
        ctx.m_g,
        BigDecimal::from_f64(E_B_SIGMA).unwrap(),
    );
    let slot_preimage_b0 = PolyMatrixNorm::fresh_preimage(
        ctx.clone(),
        ctx.m_b,
        2 * ctx.m_b,
        b0_preimage_sigma.clone(),
        None,
    );
    let b1_preimage_sigma =
        compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(2), None);
    let slot_preimage_b1 = PolyMatrixNorm::fresh_preimage(
        ctx.clone(),
        ctx.m_b * 2,
        ctx.m_g,
        b1_preimage_sigma.clone(),
        None,
    );
    let slot_preimage_b0_target_error = PolyMatrixNorm::sample_gauss(
        ctx.clone(),
        ctx.secret_size,
        ctx.m_b * 2,
        BigDecimal::from_f64(E_B_SIGMA).unwrap(),
    );
    let slot_preimage_b1_target_error = PolyMatrixNorm::sample_gauss(
        ctx.clone(),
        ctx.secret_size * 2,
        ctx.m_g,
        BigDecimal::from_f64(E_B_SIGMA).unwrap(),
    );
    let slot_secret_and_identity = PolyMatrixNorm::new(
        ctx.clone(),
        ctx.secret_size,
        ctx.secret_size * 2,
        BigDecimal::one(),
        None,
    );
    let scalar_bd = BigDecimal::from(3u64);
    let input_vector_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
    let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
    let const_term = s_vec.clone() * &gate_target_error + c_b0_error_norm.clone() * &gate_preimage;
    let transfer_plaintext_multiplier =
        s_vec.clone() * slot_secret_and_identity * slot_preimage_b1_target_error +
            s_vec.clone() * slot_preimage_b0_target_error * slot_preimage_b1.clone() +
            c_b0_error_norm * slot_preimage_b0 * slot_preimage_b1;
    let matrix_norm = const_term +
        (input.matrix_norm.clone() * &input_vector_multiplier) * &scalar_bd +
        transfer_plaintext_multiplier * &plaintext_norm;

    let expected =
        ErrorNorm::from_parts(plaintext_norm, matrix_norm, input.pubkey_deps.clone(), false);
    assert_error_bound_eq(&out, &expected);
}

#[test]
fn test_wire_norm_slot_transfer_bound_is_independent_of_slot_count() {
    let ctx = make_ctx();
    let e_b0_sigma = 9.0;
    let evaluator = NormBggPolyEncodingSTEvaluator::new(
        ctx.clone(),
        e_b0_sigma,
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        None,
    );
    let input = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), BigDecimal::from(4u64)),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(6u64), None),
    );

    let out_single = evaluator.slot_transfer(&(), &input, &[(0, Some(2))], GateId(0));
    let out_many = evaluator.slot_transfer(
        &(),
        &input,
        &[(0, Some(2)), (1, Some(2)), (2, Some(2))],
        GateId(1),
    );

    assert_error_bound_eq(&out_single, &out_many);
}

#[test]
fn test_wire_norm_naive_bgg_encoding_vec_slot_transfer_is_free() {
    let ctx = make_ctx();
    let evaluator = NormNaiveBggEncodingVecSTEvaluator::new();
    let input = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), BigDecimal::from(5u64)),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(7u64), None),
    );
    let src_slots = [(2, None), (0, Some(3)), (1, Some(2))];

    let out = evaluator.slot_transfer(&(), &input, &src_slots, GateId(0));

    assert_error_bound_eq(&out, &input.small_scalar_mul(&(), &[3]));
}

#[test]
fn test_wire_norm_naive_bgg_encoding_vec_slot_transfer_affine_path_is_free() {
    let ctx = make_ctx();
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let out_gate = circuit.slot_transfer_gate(inputs[0], &[(2, None), (0, Some(4)), (1, Some(2))]);
    circuit.output(vec![out_gate.as_single_wire()]);

    let evaluator = NormNaiveBggEncodingVecSTEvaluator::new();
    let input_bound = BigDecimal::from(5u64);
    let e_init_norm = BigDecimal::from(E_INIT_NORM);
    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        1,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
        Some(&evaluator),
    );
    let input = ErrorNorm::fresh_input(
        PolyNorm::constant(ctx.clone(), input_bound),
        PolyMatrixNorm::fresh_random_with_norm(ctx.clone(), 1, ctx.m_g, e_init_norm, None),
    );
    let expected = vec![input.small_scalar_mul(&(), &[4])];

    assert_error_bounds_eq(&out, &expected);
}

#[test]
fn test_wire_norm_lwe_plt_bounds() {
    // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
    let params = DCRTPolyParams::default();
    let plt = PublicLut::<DCRTPoly>::new(
        &params,
        2,
        |params, idx| match idx {
            0 => Some((
                0,
                DCRTPoly::from_usize_to_constant(params, 5)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            1 => Some((
                1,
                DCRTPoly::from_usize_to_constant(params, 7)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            _ => unreachable!("index out of range for test LUT"),
        },
        None,
    );

    // Circuit: out = PLT(in)
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let plt_id = circuit.register_public_lookup(plt);
    let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
    circuit.output(vec![out_gate]);

    let ctx = make_ctx();
    let input_bound = BigDecimal::from(5u64);
    let plt_evaluator =
        NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(E_B_SIGMA).unwrap());
    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        1,
        &BigDecimal::from(E_INIT_NORM),
        Some(&plt_evaluator),
        None,
    );
    assert_eq!(out.len(), 1);
    // Bound must be max output coeff across LUT entries (7)
    assert_eq!(out[0].plaintext_norm.sigma, BigDecimal::from(7u64));
}

#[test]
fn test_wire_norm_ggh15_plt_bounds() {
    // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
    let params = DCRTPolyParams::default();
    let plt = PublicLut::<DCRTPoly>::new(
        &params,
        2,
        |params, idx| match idx {
            0 => Some((
                0,
                DCRTPoly::from_usize_to_constant(params, 5)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            1 => Some((
                1,
                DCRTPoly::from_usize_to_constant(params, 7)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            _ => unreachable!("index out of range for test LUT"),
        },
        None,
    );

    // Circuit: out = PLT(in)
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let plt_id = circuit.register_public_lookup(plt);
    let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
    circuit.output(vec![out_gate]);

    let ctx = make_ctx();
    let input_bound = BigDecimal::from(5u64);
    let plt_evaluator = NormPltGGH15Evaluator::new(
        ctx.clone(),
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        None,
    );
    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        1,
        &BigDecimal::from(E_INIT_NORM),
        Some(&plt_evaluator),
        None,
    );
    assert_eq!(out.len(), 1);
    // Bound must be max output coeff across LUT entries (7)
    assert_eq!(out[0].plaintext_norm.sigma, BigDecimal::from(7u64));
}

#[test]
fn test_wire_norm_simulator_ggh15_plt_uses_lut_plaintext_bound() {
    let params = DCRTPolyParams::default();
    let plt = PublicLut::<DCRTPoly>::new(
        &params,
        2,
        |params, idx| match idx {
            0 => Some((
                0,
                DCRTPoly::from_usize_to_constant(params, 5)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            1 => Some((
                1,
                DCRTPoly::from_usize_to_constant(params, 7)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            _ => unreachable!("index out of range for test LUT"),
        },
        None,
    );

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let plt_id = circuit.register_public_lookup(plt);
    let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
    circuit.output(vec![out_gate]);

    let ctx = make_ctx();
    let input_bound = BigDecimal::from(5u64);
    let plt_evaluator = NormPltGGH15Evaluator::new(
        ctx.clone(),
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        None,
    );
    let out = circuit.simulate_max_error_norm(
        ctx,
        input_bound,
        1,
        &BigDecimal::from(E_INIT_NORM),
        Some(&plt_evaluator),
        None,
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].plaintext_norm.sigma, BigDecimal::from(7u64));
}

#[test]
fn test_wire_norm_simulator_sub_circuit_matches_generic_eval() {
    let params = DCRTPolyParams::default();
    let plt = PublicLut::<DCRTPoly>::new(
        &params,
        2,
        |params, idx| match idx {
            0 => Some((
                0,
                DCRTPoly::from_usize_to_constant(params, 3)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            1 => Some((
                1,
                DCRTPoly::from_usize_to_constant(params, 5)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            _ => unreachable!("index out of range for test LUT"),
        },
        None,
    );

    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(1).to_vec();
    let squared = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
    let sub_out = sub_circuit.add_gate(squared, sub_inputs[0]);
    sub_circuit.output(vec![sub_out]);

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(2).to_vec();
    let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
    let left = circuit.call_sub_circuit(sub_circuit_id, [inputs[0]]);
    let right = circuit.call_sub_circuit(sub_circuit_id, [inputs[1]]);
    let summed = circuit.add_gate(left[0], right[0]);
    let plt_id = circuit.register_public_lookup(plt);
    let out = circuit.public_lookup_gate(summed, plt_id);
    circuit.output(vec![out]);

    let ctx = make_ctx();
    let input_bound = BigDecimal::from(13u64);
    let e_init_norm = BigDecimal::from(E_INIT_NORM);
    let plt_evaluator = NormPltGGH15Evaluator::new(
        ctx.clone(),
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        None,
    );

    let simulated = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        2,
        &e_init_norm,
        Some(&plt_evaluator),
        None,
    );
    let generic = simulate_max_error_norm_via_generic_eval_reference(
        &circuit,
        ctx,
        input_bound,
        2,
        &e_init_norm,
        Some(&plt_evaluator),
    );

    assert_error_bounds_cover(&simulated, &generic);
}

#[test]
fn test_wire_norm_simulator_sub_circuit_recomputes_for_new_plaintext_profile() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(1).to_vec();
    let squared = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
    let sub_out = sub_circuit.add_gate(squared, sub_inputs[0]);
    sub_circuit.output(vec![sub_out]);

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let doubled = circuit.add_gate(inputs[0], inputs[0]);
    let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
    let left = circuit.call_sub_circuit(sub_circuit_id, [inputs[0]]);
    let right = circuit.call_sub_circuit(sub_circuit_id, [doubled]);
    let out = circuit.add_gate(left[0], right[0]);
    circuit.output(vec![out]);

    let ctx = make_ctx();
    let input_bound = BigDecimal::from(13u64);
    let e_init_norm = BigDecimal::from(E_INIT_NORM);

    let simulated = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        1,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
        None,
    );
    let generic = simulate_max_error_norm_via_generic_eval_reference(
        &circuit,
        ctx,
        input_bound,
        1,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
    );

    assert_error_bounds_cover(&simulated, &generic);
}

#[test]
fn test_wire_norm_simulator_nested_sub_circuit_matches_generic_eval() {
    let mut inner_sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let inner_inputs = inner_sub_circuit.input(1).to_vec();
    let inner_out = inner_sub_circuit.add_gate(inner_inputs[0], inner_inputs[0]);
    inner_sub_circuit.output(vec![inner_out]);

    let mut outer_sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let outer_inputs = outer_sub_circuit.input(2).to_vec();
    let inner_sub_circuit_id = outer_sub_circuit.register_sub_circuit(inner_sub_circuit);
    let inner_from_second =
        outer_sub_circuit.call_sub_circuit(inner_sub_circuit_id, [outer_inputs[1]]);
    let outer_out = outer_sub_circuit.add_gate(outer_inputs[0], inner_from_second[0]);
    outer_sub_circuit.output(vec![outer_out]);

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(2).to_vec();
    let outer_sub_circuit_id = circuit.register_sub_circuit(outer_sub_circuit);
    let left = circuit.call_sub_circuit(outer_sub_circuit_id, [inputs[0], inputs[1]]);
    let right = circuit.call_sub_circuit(outer_sub_circuit_id, [inputs[1], inputs[0]]);
    let out = circuit.add_gate(left[0], right[0]);
    circuit.output(vec![out]);

    let ctx = make_ctx();
    let input_bound = BigDecimal::from(11u64);
    let e_init_norm = BigDecimal::from(E_INIT_NORM);

    let simulated = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        2,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
        None,
    );
    let generic = simulate_max_error_norm_via_generic_eval_reference(
        &circuit,
        ctx,
        input_bound,
        2,
        &e_init_norm,
        None::<&NormPltLWEEvaluator>,
    );

    assert_error_bounds_eq(&simulated, &generic);
}

#[test]
fn test_wire_norm_commit_plt_bounds() {
    // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
    let params = DCRTPolyParams::default();
    let plt = PublicLut::<DCRTPoly>::new(
        &params,
        2,
        |params, idx| match idx {
            0 => Some((
                0,
                DCRTPoly::from_usize_to_constant(params, 5)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            1 => Some((
                1,
                DCRTPoly::from_usize_to_constant(params, 7)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist"),
            )),
            _ => unreachable!("index out of range for test LUT"),
        },
        None,
    );

    // Circuit: out = PLT(in)
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let plt_id = circuit.register_public_lookup(plt);
    let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
    circuit.output(vec![out_gate]);

    let ctx = make_ctx();
    let input_bound = BigDecimal::from(5u64);
    let tree_base = 2;
    let plt_evaluator = NormPltCommitEvaluator::new(
        ctx.clone(),
        &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
        tree_base,
        &circuit,
    );
    let out = circuit.simulate_max_error_norm(
        ctx.clone(),
        input_bound.clone(),
        1,
        &BigDecimal::from(E_INIT_NORM),
        Some(&plt_evaluator),
        None,
    );
    assert_eq!(out.len(), 1);
    // Bound must be max output coeff across LUT entries (7)
    assert_eq!(out[0].plaintext_norm.sigma, BigDecimal::from(7u64));
}

#[test]
fn test_error_norm_rhs_pubkey_gadget_uses_rhs_metadata() {
    let ctx = make_ctx();
    let lhs_error = PolyMatrixNorm::sample_gauss(ctx.clone(), 1, ctx.m_g, BigDecimal::from(2u64));
    let lhs = ErrorNorm::from_parts(
        PolyNorm::one(ctx.clone()),
        lhs_error,
        DependencySet::singleton(ctx.fresh_source_id()),
        false,
    );
    let rhs_pubkey_deps = DependencySet::singleton(ctx.fresh_source_id());
    let rhs = ErrorNorm::from_parts(
        PolyNorm::one(ctx.clone()),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(3u64), None),
        rhs_pubkey_deps.clone(),
        true,
    );
    let rhs_gadget = rhs
        .rhs_pubkey_gadget_norm(PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g).poly_norm);

    assert_eq!(rhs_gadget.deps, rhs_pubkey_deps);
    assert!(rhs_gadget.clt_ready);

    let product = lhs.matrix_norm * &rhs_gadget;
    assert!(product.clt_ready);
}

#[test]
fn test_error_norm_rhs_pubkey_gadget_overlap_forces_worst_case() {
    let ctx = make_ctx();
    let shared = DependencySet::singleton(ctx.fresh_source_id());
    let lhs_matrix = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(2u64), None)
        .with_deps(shared.clone(), true);
    let rhs = ErrorNorm::from_parts(
        PolyNorm::one(ctx.clone()),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(3u64), None),
        shared,
        true,
    );
    let rhs_gadget = rhs
        .rhs_pubkey_gadget_norm(PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g).poly_norm);
    let product = lhs_matrix * &rhs_gadget;

    assert!(rhs_gadget.clt_ready);
    assert!(!product.clt_ready);
}

#[test]
fn test_error_norm_lut_and_ordinary_gate_pubkey_randomness_flags() {
    let ctx = make_ctx();
    let input = ErrorNorm::fresh_input(
        PolyNorm::one(ctx.clone()),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(2u64), None),
    );
    assert!(input.is_pubkey_random);

    let lut = ErrorNorm::fresh_lut_output(
        PolyNorm::one(ctx.clone()),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(3u64), None),
    );
    assert!(lut.is_pubkey_random);
    assert!(input.pubkey_deps.is_disjoint(&lut.pubkey_deps));

    let small_scaled_input = input.small_scalar_mul(&(), &[3]);
    assert!(small_scaled_input.is_pubkey_random);
    assert_eq!(small_scaled_input.pubkey_deps, input.pubkey_deps);

    let zero_scaled_input = input.small_scalar_mul(&(), &[0]);
    assert!(!zero_scaled_input.is_pubkey_random);
    assert_eq!(zero_scaled_input.pubkey_deps, DependencySet::empty());

    let ordinary = input + &lut;
    assert!(!ordinary.is_pubkey_random);
}

#[test]
fn test_error_norm_sub_circuit_summary_preserves_forwarded_input_pubkey_metadata() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(1).to_vec();
    sub_circuit.output(vec![sub_inputs[0]]);

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
    let out = circuit.call_sub_circuit(sub_circuit_id, [inputs[0]]);
    circuit.output(vec![out[0]]);

    let ctx = make_ctx();
    let out = circuit.simulate_max_error_norm(
        ctx,
        BigDecimal::from(3u64),
        1,
        &BigDecimal::from(5u64),
        None::<&NormPltLWEEvaluator>,
        None,
    );

    assert!(out[0].is_pubkey_random);
    assert!(out[0].pubkey_deps != DependencySet::empty());
}
