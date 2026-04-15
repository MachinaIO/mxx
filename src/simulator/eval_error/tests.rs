use super::*;
use crate::{
    circuit::PolyCircuit,
    lookup::PublicLut,
    poly::{
        Poly,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    simulator::SimulatorContext,
    slot_transfer::SlotTransferEvaluator,
};
use bigdecimal::BigDecimal;

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
    let input_error = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), input_norm_bound),
        PolyMatrixNorm::new(ctx, 1, one_error.ctx().m_g, e_init_norm.clone(), None),
    );
    circuit.eval(&(), one_error, vec![input_error; input_size], plt_evaluator, None, None)
}

const E_B_SIGMA: f64 = 4.0;
const E_INIT_NORM: u32 = 1 << 14;

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
    let in_wire = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), input_bound),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
    );
    let expected = &in_wire + &in_wire;
    assert_eq!(out[0], expected);
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
    let in_wire = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), input_bound),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
    );
    let expected = &in_wire - &in_wire; // subtraction bound equals addition bound
    assert_eq!(out[0], expected);
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
    let in_wire = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), input_bound),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
    );
    let expected = &in_wire * &in_wire;
    assert_eq!(out[0], expected);
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
    assert_eq!(out, generic);
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

    let b0_preimage_norm =
        compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(1));
    let s_vec = PolyMatrixNorm::new(ctx.clone(), 1, ctx.secret_size, BigDecimal::one(), None);
    let gate_preimage =
        PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, b0_preimage_norm.clone(), None);
    let gate_target_error = PolyMatrixNorm::new(
        ctx.clone(),
        ctx.secret_size,
        ctx.m_g,
        BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
        None,
    );
    let slot_preimage_b0 =
        PolyMatrixNorm::new(ctx.clone(), ctx.m_b, 2 * ctx.m_b, b0_preimage_norm.clone(), None);
    let b1_preimage_norm =
        compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(2));
    let slot_preimage_b1 =
        PolyMatrixNorm::new(ctx.clone(), ctx.m_b * 2, ctx.m_g, b1_preimage_norm.clone(), None);
    let slot_preimage_b0_target_error = PolyMatrixNorm::new(
        ctx.clone(),
        ctx.secret_size,
        ctx.m_b * 2,
        BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
        None,
    );
    let slot_preimage_b1_target_error = PolyMatrixNorm::new(
        ctx.clone(),
        ctx.secret_size * 2,
        ctx.m_g,
        BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
        None,
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

    assert_eq!(out, ErrorNorm { plaintext_norm, matrix_norm });
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

    assert_eq!(out_single, out_many);
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
    assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
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
    assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
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
    assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
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
    let left = circuit.call_sub_circuit(sub_circuit_id, &[inputs[0]]);
    let right = circuit.call_sub_circuit(sub_circuit_id, &[inputs[1]]);
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

    assert_eq!(simulated, generic);
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
    let left = circuit.call_sub_circuit(sub_circuit_id, &[inputs[0]]);
    let right = circuit.call_sub_circuit(sub_circuit_id, &[doubled]);
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

    assert_eq!(simulated, generic);
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
        outer_sub_circuit.call_sub_circuit(inner_sub_circuit_id, &[outer_inputs[1]]);
    let outer_out = outer_sub_circuit.add_gate(outer_inputs[0], inner_from_second[0]);
    outer_sub_circuit.output(vec![outer_out]);

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(2).to_vec();
    let outer_sub_circuit_id = circuit.register_sub_circuit(outer_sub_circuit);
    let left = circuit.call_sub_circuit(outer_sub_circuit_id, &[inputs[0], inputs[1]]);
    let right = circuit.call_sub_circuit(outer_sub_circuit_id, &[inputs[1], inputs[0]]);
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

    assert_eq!(simulated, generic);
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
    assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
}
