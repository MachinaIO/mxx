use super::*;
use crate::{
    __PAIR, __TestState,
    circuit::{PolyGateKind, evaluable::PolyVec},
    gadgets::ntt::encode_nested_rns_poly_vec_with_offset,
    lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    slot_transfer::PolyVecSlotTransferEvaluator,
    utils::{ceil_biguint_nth_root, pow_biguint_usize},
};
use num_traits::One;
use super::encoding::sample_crt_primes_mul_budget_bound;

const P_MODULI_BITS: usize = 6;
const MAX_UNREDUCED_MULS: usize = DEFAULT_MAX_UNREDUCED_MULS;
const SCALE: u64 = 1 << 8;
const BASE_BITS: u32 = 6;

fn create_test_context(
    circuit: &mut PolyCircuit<DCRTPoly>,
) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
    create_test_context_with_config(circuit, None, P_MODULI_BITS, MAX_UNREDUCED_MULS)
}

fn create_test_context_with_q_level(
    circuit: &mut PolyCircuit<DCRTPoly>,
    q_level: Option<usize>,
) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
    create_test_context_with_config(circuit, q_level, P_MODULI_BITS, MAX_UNREDUCED_MULS)
}

fn create_test_context_with_config(
    circuit: &mut PolyCircuit<DCRTPoly>,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
    let _ = tracing_subscriber::fmt::try_init();
    let params = DCRTPolyParams::new(4, 6, 18, BASE_BITS);
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        circuit,
        &params,
        p_moduli_bits,
        max_unreduced_muls,
        SCALE,
        false,
        q_level,
    ));
    println!("p moduli: {:?}", &ctx.p_moduli);
    let p = ctx.p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
    println!("p: {}", p);
    (params, ctx)
}

fn create_test_context_for_params(
    circuit: &mut PolyCircuit<DCRTPoly>,
    params: &DCRTPolyParams,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
) -> Arc<NestedRnsPolyContext> {
    let _ = tracing_subscriber::fmt::try_init();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        circuit,
        params,
        p_moduli_bits,
        max_unreduced_muls,
        SCALE,
        false,
        q_level,
    ));
    println!("p moduli: {:?}", &ctx.p_moduli);
    let p = ctx.p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
    println!("p: {}", p);
    ctx
}

fn sparse_gadget_entry(
    ctx: Arc<NestedRnsPolyContext>,
    target_q_idx: usize,
    circuit: &mut PolyCircuit<DCRTPoly>,
) -> NestedRnsPoly<DCRTPoly> {
    let active_levels = ctx.q_moduli_depth;
    assert!(
        target_q_idx < active_levels,
        "test sparse gadget q_idx {} exceeds active levels {}",
        target_q_idx,
        active_levels
    );
    let value = ctx.gadget_values[target_q_idx]
        .first()
        .cloned()
        .expect("gadget values must contain at least one digit per q level");
    NestedRnsPoly::<DCRTPoly>::sparse_constant_level_poly(
        ctx,
        active_levels,
        None,
        0,
        target_q_idx,
        &value,
        circuit,
    )
}

fn build_gadget_decomposition_reconstruction_circuit(
    circuit: &mut PolyCircuit<DCRTPoly>,
    ctx: Arc<NestedRnsPolyContext>,
) {
    let input = NestedRnsPoly::input(ctx.clone(), None, None, circuit);
    let gadget = NestedRnsPoly::<DCRTPoly>::gadget_vector(ctx.clone(), None, None, circuit);
    let decomposition = input.gadget_decompose(circuit);

    assert_eq!(gadget.len(), decomposition.len());
    assert_eq!(gadget.len(), ctx.q_moduli_depth * (ctx.p_moduli.len() + 1));

    let mut term_iter = gadget.iter().zip(decomposition.iter());
    let (first_gadget, first_decomposition) =
        term_iter.next().expect("gadget decomposition must contain at least one chunk");
    let mut reconstructed_sum = first_gadget.mul(first_decomposition, circuit);
    for (gadget_entry, decomposition_entry) in term_iter {
        let term = gadget_entry.mul(decomposition_entry, circuit);
        reconstructed_sum = reconstructed_sum.add(&term, circuit);
    }
    let out = reconstructed_sum.reconstruct(circuit);
    circuit.output(vec![out]);
}

fn build_gadget_vector_reconstruction_circuit(
    circuit: &mut PolyCircuit<DCRTPoly>,
    ctx: Arc<NestedRnsPolyContext>,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> usize {
    NestedRnsPoly::input(ctx.clone(), enable_levels, level_offset, circuit);
    let gadget =
        NestedRnsPoly::<DCRTPoly>::gadget_vector(ctx, enable_levels, level_offset, circuit);
    let output_gates = gadget.iter().map(|entry| entry.reconstruct(circuit)).collect::<Vec<_>>();
    let gadget_len = output_gates.len();
    circuit.output(output_gates);
    gadget_len
}

fn build_matrix_gadget_decomposition_reconstruction_circuit(
    circuit: &mut PolyCircuit<DCRTPoly>,
    ctx: Arc<NestedRnsPolyContext>,
    row_size: usize,
    col_size: usize,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> usize {
    assert!(row_size > 0, "row_size must be positive");
    assert!(col_size > 0, "col_size must be positive");
    let inputs = (0..row_size)
        .map(|_| {
            (0..col_size)
                .map(|_| NestedRnsPoly::input(ctx.clone(), enable_levels, level_offset, circuit))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let gadget_len =
        NestedRnsPoly::<DCRTPoly>::gadget_vector(ctx, enable_levels, level_offset, circuit).len();
    let decomposed = inputs
        .iter()
        .map(|row| row.iter().map(|entry| entry.gadget_decompose(circuit)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let mut output_gates = Vec::with_capacity(row_size * col_size * gadget_len);
    for row in &decomposed {
        for digit_idx in 0..gadget_len {
            for col in row {
                output_gates.push(col[digit_idx].reconstruct(circuit));
            }
        }
    }
    circuit.output(output_gates);
    gadget_len
}

fn eval_poly_vec_outputs(
    params: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    inputs: Vec<PolyVec<DCRTPoly>>,
) -> Vec<PolyVec<DCRTPoly>> {
    let one = PolyVec::new(vec![DCRTPoly::const_one(params); params.ring_dimension() as usize]);
    let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
    let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
    circuit.eval(params, one, inputs, Some(&plt_evaluator), Some(&slot_transfer_evaluator), None)
}

fn poly_from_poly_vec_output(params: &DCRTPolyParams, output: &PolyVec<DCRTPoly>) -> DCRTPoly {
    let slots = output
        .as_slice()
        .iter()
        .map(|slot_poly| {
            slot_poly
                .coeffs_biguints()
                .into_iter()
                .next()
                .expect("output slot polynomial must contain a constant coefficient")
        })
        .collect::<Vec<_>>();
    DCRTPoly::from_biguints(params, &slots)
}

fn matrix_from_poly_vec_outputs(
    params: &DCRTPolyParams,
    outputs: &[PolyVec<DCRTPoly>],
    row_size: usize,
    col_size: usize,
) -> DCRTPolyMatrix {
    assert_eq!(
        outputs.len(),
        row_size * col_size,
        "output count must match the requested matrix shape"
    );
    let matrix = (0..row_size)
        .map(|row_idx| {
            (0..col_size)
                .map(|col_idx| {
                    poly_from_poly_vec_output(params, &outputs[row_idx * col_size + col_idx])
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    DCRTPolyMatrix::from_poly_vec(params, matrix)
}

fn encode_matrix_inputs_with_offset(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    values: &DCRTPolyMatrix,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> Vec<PolyVec<DCRTPoly>> {
    let q_level_offset = level_offset.unwrap_or(0);
    let mut inputs = Vec::new();
    for row_idx in 0..values.row_size() {
        for col_idx in 0..values.col_size() {
            let poly = values.entry(row_idx, col_idx);
            inputs.extend(encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                params,
                ctx,
                &poly.coeffs_biguints(),
                q_level_offset,
                enable_levels,
            ));
        }
    }
    inputs
}

fn div_ceil_biguint_by_u64(value: BigUint, divisor: u64) -> BigUint {
    let adjustment = BigUint::from(divisor.saturating_sub(1));
    (value + adjustment) / BigUint::from(divisor)
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_add_auto_reduce_maxes() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let a_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
    let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
    test_nested_rns_poly_add_generic(circuit, params, ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_add_auto_reduce_random() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let mut rng = rand::rng();
    let a_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let b_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    test_nested_rns_poly_add_generic(circuit, params, ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_sub_auto_reduce_maxes() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let a_value: BigUint = BigUint::ZERO;
    let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
    test_nested_rns_poly_sub_generic(circuit, params, ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_sub_auto_reduce_random() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let mut rng = rand::rng();
    let a_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let b_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    test_nested_rns_poly_sub_generic(circuit, params, ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_mul_auto_reduce_maxes() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let a_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
    let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
    test_nested_rns_poly_mul_generic(circuit, params, ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_mul_auto_reduce_random() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let mut rng = rand::rng();
    let a_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let b_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    test_nested_rns_poly_mul_generic(circuit, params, ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_mul_auto_reduce_reconstruct_maxes() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let a_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
    let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
    test_nested_rns_poly_mul_reconstruct_generic(circuit, params, ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_mul_right_sparse_matches_generic_mul() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let right_q_idx = if ctx.q_moduli_depth > 1 { 1 } else { 0 };
    let lhs = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let rhs_sparse = sparse_gadget_entry(ctx, right_q_idx, &mut circuit);
    let sparse_product = lhs.mul_right_sparse(&rhs_sparse, right_q_idx, &mut circuit);
    let generic_product = lhs.mul(&rhs_sparse, &mut circuit);
    let sparse_out = sparse_product.reconstruct(&mut circuit);
    let generic_out = generic_product.reconstruct(&mut circuit);
    circuit.output(vec![sparse_out, generic_out]);

    let modulus = params.modulus();
    let input_values =
        vec![BigUint::ZERO, BigUint::from(7u64), modulus.as_ref().clone() - BigUint::from(5u64)];
    let one = DCRTPoly::const_one(&params);
    let plt_evaluator = PolyPltEvaluator::new();
    for lhs_value in input_values {
        let lhs_inputs =
            encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &lhs_value, None);
        let eval_results =
            circuit.eval(&params, one.clone(), lhs_inputs, Some(&plt_evaluator), None, None);
        assert_eq!(eval_results.len(), 2);
        assert_eq!(
            eval_results[0].coeffs_biguints()[0],
            eval_results[1].coeffs_biguints()[0],
            "mul_right_sparse should match generic mul for lhs_value={lhs_value}"
        );
    }
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_mul_right_sparse_uses_fewer_gates_than_generic_mul() {
    let right_q_idx = 0usize;
    let (generic_counts, sparse_counts) = rayon::join(
        || {
            let mut generic_circuit = PolyCircuit::<DCRTPoly>::new();
            let (_generic_params, generic_ctx) = create_test_context(&mut generic_circuit);
            let generic_lhs =
                NestedRnsPoly::input(generic_ctx.clone(), None, None, &mut generic_circuit);
            let generic_rhs = sparse_gadget_entry(generic_ctx, right_q_idx, &mut generic_circuit);
            let generic_product = generic_lhs.mul(&generic_rhs, &mut generic_circuit);
            let generic_out = generic_product.reconstruct(&mut generic_circuit);
            generic_circuit.output(vec![generic_out]);
            generic_circuit.count_gates_by_type_vec()
        },
        || {
            let mut sparse_circuit = PolyCircuit::<DCRTPoly>::new();
            let (_sparse_params, sparse_ctx) = create_test_context(&mut sparse_circuit);
            let sparse_lhs =
                NestedRnsPoly::input(sparse_ctx.clone(), None, None, &mut sparse_circuit);
            let sparse_rhs = sparse_gadget_entry(sparse_ctx, right_q_idx, &mut sparse_circuit);
            let sparse_product =
                sparse_lhs.mul_right_sparse(&sparse_rhs, right_q_idx, &mut sparse_circuit);
            let sparse_out = sparse_product.reconstruct(&mut sparse_circuit);
            sparse_circuit.output(vec![sparse_out]);
            sparse_circuit.count_gates_by_type_vec()
        },
    );

    let generic_mul_gates = generic_counts.get(&PolyGateKind::Mul).copied().unwrap_or_default();
    let sparse_mul_gates = sparse_counts.get(&PolyGateKind::Mul).copied().unwrap_or_default();
    assert!(
        sparse_mul_gates < generic_mul_gates,
        "mul_right_sparse should use fewer Mul gates than generic mul: sparse={}, generic={}",
        sparse_mul_gates,
        generic_mul_gates
    );

    let generic_total: usize = generic_counts.values().copied().sum();
    let sparse_total: usize = sparse_counts.values().copied().sum();
    assert!(
        sparse_total < generic_total,
        "mul_right_sparse should use fewer expanded gates than generic mul: sparse={}, generic={}",
        sparse_total,
        generic_total
    );
}

#[sequential_test::sequential]
#[test]
#[should_panic(expected = "mul_right_sparse requires the right operand to be zero outside q_idx")]
fn test_nested_rns_poly_mul_right_sparse_requires_sparse_rhs() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, ctx) = create_test_context(&mut circuit);
    let lhs = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let rhs = NestedRnsPoly::input(ctx, None, None, &mut circuit);
    let _ = lhs.mul_right_sparse(&rhs, 0, &mut circuit);
}

fn build_manual_conv_mul_right_decomposed_many(
    params: &DCRTPolyParams,
    left_rows: &[Vec<NestedRnsPoly<DCRTPoly>>],
    right: &NestedRnsPoly<DCRTPoly>,
    num_slots: usize,
    circuit: &mut PolyCircuit<DCRTPoly>,
) -> Vec<NestedRnsPoly<DCRTPoly>> {
    let p_moduli_depth = right.ctx.p_moduli.len();
    let chunk_width = p_moduli_depth + 1;
    let decomposed = right.gadget_decompose(circuit);
    left_rows
        .iter()
        .map(|row| {
            assert_eq!(
                row.len(),
                decomposed.len(),
                "manual decomposed-conv row length {} must match gadget decomposition length {}",
                row.len(),
                decomposed.len()
            );
            let terms =
                row.iter().zip(decomposed.iter()).enumerate().map(|(entry_idx, (left, decomp))| {
                    crate::gadgets::conv_mul::negacyclic_conv_mul_right_sparse(
                        params,
                        circuit,
                        left,
                        decomp,
                        entry_idx / chunk_width,
                        num_slots,
                    )
                });
            let mut terms = terms.collect::<Vec<_>>().into_iter();
            let first = terms
                .next()
                .expect("manual decomposed convolution requires at least one gadget term");
            terms.fold(first, |acc, term| acc.add(&term, circuit))
        })
        .collect()
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_conv_mul_right_decomposed_many_matches_manual_pipeline() {
    let q_level = Some(1usize);
    let num_slots = 4usize;

    let mut fused_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, fused_ctx) = create_test_context_with_q_level(&mut fused_circuit, q_level);
    let chunk_width = fused_ctx.p_moduli.len() + 1;
    let gadget_len = fused_ctx.q_moduli_depth * chunk_width;
    let fused_left_rows = (0..2)
        .map(|_| {
            (0..gadget_len)
                .map(|_| NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let fused_right = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_outputs = fused_right.conv_mul_right_decomposed_many(
        &params,
        &[fused_left_rows[0].as_slice(), fused_left_rows[1].as_slice()],
        num_slots,
        &mut fused_circuit,
    );
    let fused_reconstructed = fused_outputs
        .iter()
        .map(|output| output.reconstruct(&mut fused_circuit))
        .collect::<Vec<_>>();
    fused_circuit.output(fused_reconstructed);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_manual_params, manual_ctx) =
        create_test_context_with_q_level(&mut manual_circuit, q_level);
    let manual_left_rows = (0..2)
        .map(|_| {
            (0..gadget_len)
                .map(|_| {
                    NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let manual_right = NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_outputs = build_manual_conv_mul_right_decomposed_many(
        &params,
        &manual_left_rows,
        &manual_right,
        num_slots,
        &mut manual_circuit,
    );
    let manual_reconstructed = manual_outputs
        .iter()
        .map(|output| output.reconstruct(&mut manual_circuit))
        .collect::<Vec<_>>();
    manual_circuit.output(manual_reconstructed);

    let left_input_values = (0..(2 * gadget_len))
        .map(|idx| BigUint::from(u64::try_from(idx + 2).expect("test input index fits u64")))
        .collect::<Vec<_>>();
    let right_value = BigUint::from(13u64);
    let fused_inputs = left_input_values
        .iter()
        .flat_map(|value| {
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                fused_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        })
        .chain({
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = right_value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                fused_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        })
        .collect::<Vec<_>>();
    let manual_inputs = left_input_values
        .iter()
        .flat_map(|value| {
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                manual_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        })
        .chain({
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = right_value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                manual_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        })
        .collect::<Vec<_>>();
    let fused_eval = eval_poly_vec_outputs(&params, &fused_circuit, fused_inputs);
    let manual_eval = eval_poly_vec_outputs(&params, &manual_circuit, manual_inputs);
    assert_eq!(fused_eval, manual_eval);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_conv_mul_right_decomposed_many_does_not_increase_non_free_depth() {
    let q_level = Some(1usize);
    let num_slots = 4usize;

    let mut fused_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, fused_ctx) = create_test_context_with_q_level(&mut fused_circuit, q_level);
    let chunk_width = fused_ctx.p_moduli.len() + 1;
    let gadget_len = fused_ctx.q_moduli_depth * chunk_width;
    let fused_left_rows = (0..2)
        .map(|_| {
            (0..gadget_len)
                .map(|_| NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let fused_right = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_outputs = fused_right.conv_mul_right_decomposed_many(
        &params,
        &[fused_left_rows[0].as_slice(), fused_left_rows[1].as_slice()],
        num_slots,
        &mut fused_circuit,
    );
    let fused_reconstructed = fused_outputs
        .iter()
        .map(|output| output.reconstruct(&mut fused_circuit))
        .collect::<Vec<_>>();
    fused_circuit.output(fused_reconstructed);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_manual_params, manual_ctx) =
        create_test_context_with_q_level(&mut manual_circuit, q_level);
    let manual_left_rows = (0..2)
        .map(|_| {
            (0..gadget_len)
                .map(|_| {
                    NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let manual_right = NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_outputs = build_manual_conv_mul_right_decomposed_many(
        &params,
        &manual_left_rows,
        &manual_right,
        num_slots,
        &mut manual_circuit,
    );
    let manual_reconstructed = manual_outputs
        .iter()
        .map(|output| output.reconstruct(&mut manual_circuit))
        .collect::<Vec<_>>();
    manual_circuit.output(manual_reconstructed);

    assert!(
        fused_circuit.non_free_depth() <= manual_circuit.non_free_depth(),
        "fused right-decomposed convolution should not increase non-free depth: fused={}, manual={}",
        fused_circuit.non_free_depth(),
        manual_circuit.non_free_depth()
    );
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_gadget_decompose_unreduced_matches_explicit_lazy_reduce() {
    let q_level = Some(1usize);
    let num_slots = 4usize;

    let mut fused_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, fused_ctx) = create_test_context_with_q_level(&mut fused_circuit, q_level);
    let fused_left = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_right = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_unreduced = fused_left.add(&fused_right, &mut fused_circuit);
    let fused_outputs = fused_unreduced
        .gadget_decompose(&mut fused_circuit)
        .into_iter()
        .map(|term| term.reconstruct(&mut fused_circuit))
        .collect::<Vec<_>>();
    fused_circuit.output(fused_outputs);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_manual_params, manual_ctx) =
        create_test_context_with_q_level(&mut manual_circuit, q_level);
    let manual_left = NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_right = NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_unreduced = manual_left.add(&manual_right, &mut manual_circuit);
    let manual_reduced = manual_unreduced.lazy_reduce_if_unreduced(&mut manual_circuit);
    let manual_outputs = manual_reduced
        .gadget_decompose(&mut manual_circuit)
        .into_iter()
        .map(|term| term.reconstruct(&mut manual_circuit))
        .collect::<Vec<_>>();
    manual_circuit.output(manual_outputs);

    let left_coeffs =
        vec![BigUint::from(3u64), BigUint::from(5u64), BigUint::from(7u64), BigUint::from(11u64)];
    let right_coeffs = vec![
        BigUint::from(13u64),
        BigUint::from(17u64),
        BigUint::from(19u64),
        BigUint::from(23u64),
    ];
    let fused_inputs = [
        encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &params,
            fused_ctx.as_ref(),
            &left_coeffs,
            0,
            q_level,
        ),
        encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &params,
            fused_ctx.as_ref(),
            &right_coeffs,
            0,
            q_level,
        ),
    ]
    .concat();
    let manual_inputs = [
        encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &params,
            manual_ctx.as_ref(),
            &left_coeffs,
            0,
            q_level,
        ),
        encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &params,
            manual_ctx.as_ref(),
            &right_coeffs,
            0,
            q_level,
        ),
    ]
    .concat();
    let fused_eval = eval_poly_vec_outputs(&params, &fused_circuit, fused_inputs);
    let manual_eval = eval_poly_vec_outputs(&params, &manual_circuit, manual_inputs);
    assert_eq!(fused_eval, manual_eval);
    assert_eq!(fused_eval.len(), fused_ctx.p_moduli.len() + 1);
    assert!(num_slots > 0);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_gadget_decompose_unreduced_does_not_increase_non_free_depth() {
    let q_level = Some(1usize);

    let mut fused_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_params, fused_ctx) = create_test_context_with_q_level(&mut fused_circuit, q_level);
    let fused_left = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_right = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_unreduced = fused_left.add(&fused_right, &mut fused_circuit);
    let fused_outputs = fused_unreduced
        .gadget_decompose(&mut fused_circuit)
        .into_iter()
        .map(|term| term.reconstruct(&mut fused_circuit))
        .collect::<Vec<_>>();
    fused_circuit.output(fused_outputs);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_params, manual_ctx) = create_test_context_with_q_level(&mut manual_circuit, q_level);
    let manual_left = NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_right = NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_unreduced = manual_left.add(&manual_right, &mut manual_circuit);
    let manual_reduced = manual_unreduced.lazy_reduce_if_unreduced(&mut manual_circuit);
    let manual_outputs = manual_reduced
        .gadget_decompose(&mut manual_circuit)
        .into_iter()
        .map(|term| term.reconstruct(&mut manual_circuit))
        .collect::<Vec<_>>();
    manual_circuit.output(manual_outputs);

    assert!(
        fused_circuit.non_free_depth() < manual_circuit.non_free_depth(),
        "unreduced gadget_decompose should reduce non-free depth: fused={}, manual={}",
        fused_circuit.non_free_depth(),
        manual_circuit.non_free_depth()
    );
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_conv_mul_right_decomposed_many_unreduced_matches_explicit_lazy_reduce() {
    let q_level = Some(1usize);
    let num_slots = 4usize;

    let mut fused_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, fused_ctx) = create_test_context_with_q_level(&mut fused_circuit, q_level);
    let chunk_width = fused_ctx.p_moduli.len() + 1;
    let gadget_len = fused_ctx.q_moduli_depth * chunk_width;
    let fused_left_row = (0..gadget_len)
        .map(|_| {
            let left0 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
            let left1 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
            left0.add(&left1, &mut fused_circuit)
        })
        .collect::<Vec<_>>();
    let fused_right0 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_right1 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_right = fused_right0.add(&fused_right1, &mut fused_circuit);
    let fused_outputs = fused_right.conv_mul_right_decomposed_many(
        &params,
        &[fused_left_row.as_slice()],
        num_slots,
        &mut fused_circuit,
    );
    let fused_reconstructed = fused_outputs
        .iter()
        .map(|output| output.reconstruct(&mut fused_circuit))
        .collect::<Vec<_>>();
    fused_circuit.output(fused_reconstructed);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_manual_params, manual_ctx) =
        create_test_context_with_q_level(&mut manual_circuit, q_level);
    let manual_left_row = (0..gadget_len)
        .map(|_| {
            let left0 =
                NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
            let left1 =
                NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
            left0.add(&left1, &mut manual_circuit).lazy_reduce_if_unreduced(&mut manual_circuit)
        })
        .collect::<Vec<_>>();
    let manual_right0 =
        NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_right1 =
        NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_right = manual_right0
        .add(&manual_right1, &mut manual_circuit)
        .lazy_reduce_if_unreduced(&mut manual_circuit);
    let manual_outputs = manual_right.conv_mul_right_decomposed_many(
        &params,
        &[manual_left_row.as_slice()],
        num_slots,
        &mut manual_circuit,
    );
    let manual_reconstructed = manual_outputs
        .iter()
        .map(|output| output.reconstruct(&mut manual_circuit))
        .collect::<Vec<_>>();
    manual_circuit.output(manual_reconstructed);

    let left_input_values = (0..(2 * gadget_len))
        .map(|idx| BigUint::from(u64::try_from(idx + 2).expect("test input index fits u64")))
        .collect::<Vec<_>>();
    let right_input_values = [BigUint::from(101u64), BigUint::from(211u64)];
    let fused_inputs = left_input_values
        .iter()
        .flat_map(|value| {
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                fused_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        })
        .chain(right_input_values.iter().flat_map(|value| {
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                fused_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        }))
        .collect::<Vec<_>>();
    let manual_inputs = left_input_values
        .iter()
        .flat_map(|value| {
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                manual_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        })
        .chain(right_input_values.iter().flat_map(|value| {
            let mut coeffs = vec![BigUint::ZERO; num_slots];
            coeffs[0] = value.clone();
            encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                &params,
                manual_ctx.as_ref(),
                &coeffs,
                0,
                q_level,
            )
        }))
        .collect::<Vec<_>>();
    let fused_eval = eval_poly_vec_outputs(&params, &fused_circuit, fused_inputs);
    let manual_eval = eval_poly_vec_outputs(&params, &manual_circuit, manual_inputs);
    assert_eq!(fused_eval, manual_eval);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_conv_mul_right_decomposed_many_unreduced_does_not_increase_non_free_depth()
{
    let q_level = Some(1usize);
    let num_slots = 4usize;

    let mut fused_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, fused_ctx) = create_test_context_with_q_level(&mut fused_circuit, q_level);
    let chunk_width = fused_ctx.p_moduli.len() + 1;
    let gadget_len = fused_ctx.q_moduli_depth * chunk_width;
    let fused_left_row = (0..gadget_len)
        .map(|_| {
            let left0 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
            let left1 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
            left0.add(&left1, &mut fused_circuit)
        })
        .collect::<Vec<_>>();
    let fused_right0 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_right1 = NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
    let fused_right = fused_right0.add(&fused_right1, &mut fused_circuit);
    let fused_outputs = fused_right.conv_mul_right_decomposed_many(
        &params,
        &[fused_left_row.as_slice()],
        num_slots,
        &mut fused_circuit,
    );
    let fused_reconstructed = fused_outputs
        .iter()
        .map(|output| output.reconstruct(&mut fused_circuit))
        .collect::<Vec<_>>();
    fused_circuit.output(fused_reconstructed);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_params, manual_ctx) = create_test_context_with_q_level(&mut manual_circuit, q_level);
    let manual_left_row = (0..gadget_len)
        .map(|_| {
            let left0 =
                NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
            let left1 =
                NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
            left0.add(&left1, &mut manual_circuit).lazy_reduce_if_unreduced(&mut manual_circuit)
        })
        .collect::<Vec<_>>();
    let manual_right0 =
        NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_right1 =
        NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
    let manual_right = manual_right0
        .add(&manual_right1, &mut manual_circuit)
        .lazy_reduce_if_unreduced(&mut manual_circuit);
    let manual_outputs = manual_right.conv_mul_right_decomposed_many(
        &params,
        &[manual_left_row.as_slice()],
        num_slots,
        &mut manual_circuit,
    );
    let manual_reconstructed = manual_outputs
        .iter()
        .map(|output| output.reconstruct(&mut manual_circuit))
        .collect::<Vec<_>>();
    manual_circuit.output(manual_reconstructed);

    assert!(
        fused_circuit.non_free_depth() < manual_circuit.non_free_depth(),
        "unreduced conv_mul_right_decomposed_many should reduce non-free depth: fused={}, manual={}",
        fused_circuit.non_free_depth(),
        manual_circuit.non_free_depth()
    );
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_const_mul_auto_reduce_random() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let modulus = params.modulus();
    let (q_moduli, _, _) = params.to_crt();
    let mut rng = rand::rng();
    let a_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let tower_constants = q_moduli
        .iter()
        .map(|&q_i| {
            crate::utils::gen_biguint_for_modulus(&mut rng, &BigUint::from(q_i))
                .to_u64()
                .expect("tower constant must fit in u64")
        })
        .collect::<Vec<_>>();
    test_nested_rns_poly_const_mul_generic(circuit, params, ctx, a_value, tower_constants);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_respects_q_level() {
    let q_level = 2usize;
    let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context_with_q_level(&mut setup_circuit, Some(q_level));

    assert_eq!(ctx.q_moduli_depth, q_level);
    assert_eq!(ctx.max_unreduced_muls, MAX_UNREDUCED_MULS);
    assert_eq!(ctx.full_reduce_bindings.len(), q_level);
    assert_eq!(ctx.q_moduli.len(), q_level);
    assert_eq!(ctx.full_reduce_max_plaintexts.len(), q_level);

    let (q_moduli, _, _) = params.to_crt();
    let q_level_modulus =
        q_moduli.iter().take(q_level).fold(BigUint::from(1u64), |acc, &qi| acc * BigUint::from(qi));

    // Max-value multiplication under the limited q_level.
    let mut max_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, max_ctx) = create_test_context_with_q_level(&mut max_circuit, Some(q_level));
    let max_value = &q_level_modulus - BigUint::from(1u64);
    test_nested_rns_poly_mul_generic(
        max_circuit,
        params.clone(),
        max_ctx,
        max_value.clone(),
        max_value,
    );

    // Random multiplication under the limited q_level.
    let mut rng = rand::rng();
    let a_value = crate::utils::gen_biguint_for_modulus(&mut rng, &q_level_modulus);
    let b_value = crate::utils::gen_biguint_for_modulus(&mut rng, &q_level_modulus);
    let mut random_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, random_ctx) = create_test_context_with_q_level(&mut random_circuit, Some(q_level));
    test_nested_rns_poly_mul_generic(random_circuit, params, random_ctx, a_value, b_value);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_input_and_full_reduce_track_max_plaintexts() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, ctx) = create_test_context_with_q_level(&mut circuit, Some(2));
    let input = NestedRnsPoly::input(ctx.clone(), Some(2), None, &mut circuit);
    let expected_input_bounds =
        ctx.q_moduli.iter().map(|&q_i| BigUint::from(q_i - 1)).collect::<Vec<_>>();
    assert_eq!(input.max_plaintexts, expected_input_bounds);

    let reduced = input.full_reduce(&mut circuit);
    assert_eq!(reduced.max_plaintexts, ctx.full_reduce_max_plaintexts);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_gadget_decomposition_reconstructs_random_inputs() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    build_gadget_decomposition_reconstruction_circuit(&mut circuit, ctx);

    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let modulus = params.modulus();
    let mut rng = rand::rng();
    for _ in 0..5 {
        let input_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let encoded_input =
            encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &input_value, None);
        let eval_results =
            circuit.eval(&params, one.clone(), encoded_input, Some(&plt_evaluator), None, None);
        assert_eq!(eval_results.len(), 1);
        assert_eq!(eval_results[0].coeffs_biguints()[0], input_value % modulus.as_ref());
    }
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_gadget_vector_matrix_matches_reconstructed_poly_vec_outputs() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let enable_levels = Some(2usize);
    let level_offset = Some(1usize);
    let gadget_len = build_gadget_vector_reconstruction_circuit(
        &mut circuit,
        ctx.clone(),
        enable_levels,
        level_offset,
    );

    let zero_poly = DCRTPoly::const_zero(&params);
    let eval_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
        &params,
        ctx.as_ref(),
        &zero_poly.coeffs_biguints(),
        level_offset.expect("test uses a fixed level offset"),
        enable_levels,
    );
    let eval_outputs = eval_poly_vec_outputs(&params, &circuit, eval_inputs);
    let reconstructed = matrix_from_poly_vec_outputs(&params, &eval_outputs, 1, gadget_len);
    let expected = nested_rns_gadget_vector::<DCRTPoly, DCRTPolyMatrix>(
        &params,
        ctx.as_ref(),
        enable_levels,
        level_offset,
    );
    assert_eq!(reconstructed, expected);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_gadget_decomposed_matrix_matches_reconstructed_poly_vec_outputs() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let enable_levels = Some(2usize);
    let level_offset = Some(1usize);
    let row_size = 2usize;
    let col_size = 2usize;
    let gadget_len = build_matrix_gadget_decomposition_reconstruction_circuit(
        &mut circuit,
        ctx.clone(),
        row_size,
        col_size,
        enable_levels,
        level_offset,
    );

    let q0 = BigUint::from(ctx.q_moduli[level_offset.expect("test uses a fixed level offset")]);
    let q1 = BigUint::from(ctx.q_moduli[level_offset.expect("test uses a fixed level offset") + 1]);
    let input_matrix = DCRTPolyMatrix::from_poly_vec(
        &params,
        vec![
            vec![
                DCRTPoly::from_biguints(
                    &params,
                    &[
                        BigUint::from(1u64),
                        &q0 + BigUint::from(3u64),
                        &q1 + BigUint::from(5u64),
                        &q0 + &q1 + BigUint::from(7u64),
                    ],
                ),
                DCRTPoly::from_biguints(
                    &params,
                    &[
                        &q0 + BigUint::from(11u64),
                        BigUint::from(13u64),
                        &q1 + BigUint::from(17u64),
                        &q0 + &q1 + BigUint::from(19u64),
                    ],
                ),
            ],
            vec![
                DCRTPoly::from_biguints(
                    &params,
                    &[
                        &q1 + BigUint::from(23u64),
                        &q0 + &q1 + BigUint::from(29u64),
                        BigUint::from(31u64),
                        &q0 + BigUint::from(37u64),
                    ],
                ),
                DCRTPoly::from_biguints(
                    &params,
                    &[
                        &q0 + &q1 + BigUint::from(41u64),
                        &q1 + BigUint::from(43u64),
                        &q0 + BigUint::from(47u64),
                        BigUint::from(53u64),
                    ],
                ),
            ],
        ],
    );

    let eval_inputs = encode_matrix_inputs_with_offset(
        &params,
        ctx.as_ref(),
        &input_matrix,
        enable_levels,
        level_offset,
    );
    let eval_outputs = eval_poly_vec_outputs(&params, &circuit, eval_inputs);
    let reconstructed =
        matrix_from_poly_vec_outputs(&params, &eval_outputs, row_size * gadget_len, col_size);
    let expected = nested_rns_gadget_decomposed::<DCRTPoly, DCRTPolyMatrix>(
        &params,
        ctx.as_ref(),
        &input_matrix,
        enable_levels,
        level_offset,
    );
    assert_eq!(reconstructed, expected);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_gadget_decomposed_random_fin_ring_matrix_matches_reconstructed_poly_vec_outputs()
{
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let enable_levels = Some(2usize);
    let level_offset = Some(1usize);
    let row_size = 2usize;
    let col_size = 2usize;
    let gadget_len = build_matrix_gadget_decomposition_reconstruction_circuit(
        &mut circuit,
        ctx.clone(),
        row_size,
        col_size,
        enable_levels,
        level_offset,
    );

    let uniform_sampler = DCRTPolyUniformSampler::new();
    let input_matrix =
        uniform_sampler.sample_uniform(&params, row_size, col_size, DistType::FinRingDist);

    let eval_inputs = encode_matrix_inputs_with_offset(
        &params,
        ctx.as_ref(),
        &input_matrix,
        enable_levels,
        level_offset,
    );
    let eval_outputs = eval_poly_vec_outputs(&params, &circuit, eval_inputs);
    let reconstructed =
        matrix_from_poly_vec_outputs(&params, &eval_outputs, row_size * gadget_len, col_size);
    let expected = nested_rns_gadget_decomposed::<DCRTPoly, DCRTPolyMatrix>(
        &params,
        ctx.as_ref(),
        &input_matrix,
        enable_levels,
        level_offset,
    );
    assert_eq!(reconstructed, expected);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_gadget_decomposition_large_circuit_metrics() {
    let crt_bits = 24usize;
    let crt_depth = 1usize;
    let ring_dim = 1u32 << 10;
    let num_slots = 1usize << 10;
    let params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, BASE_BITS);
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = create_test_context_for_params(
        &mut circuit,
        &params,
        None,
        P_MODULI_BITS,
        MAX_UNREDUCED_MULS,
    );
    build_gadget_decomposition_reconstruction_circuit(&mut circuit, ctx);

    println!(
        "nested_rns gadget decomposition metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}"
    );
    println!("non-free depth {}", circuit.non_free_depth());
    println!("gate counts {:?}", circuit.count_gates_by_type_vec());
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_reconstruct_auto_reduce_matches_manual_full_reduce() {
    let q_level = 1usize;
    let input_value = BigUint::from(123u64);

    let mut auto_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, auto_ctx) = create_test_context_with_q_level(&mut auto_circuit, Some(q_level));
    let mut auto_input =
        NestedRnsPoly::input(auto_ctx.clone(), Some(q_level), None, &mut auto_circuit);
    auto_input.max_plaintexts = vec![auto_ctx.p_full.clone()];
    let auto_out = auto_input.reconstruct(&mut auto_circuit);
    auto_circuit.output(vec![auto_out]);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, manual_ctx) = create_test_context_with_q_level(&mut manual_circuit, Some(q_level));
    let mut manual_input =
        NestedRnsPoly::input(manual_ctx.clone(), Some(q_level), None, &mut manual_circuit);
    manual_input.max_plaintexts = vec![manual_ctx.p_full.clone()];
    let manual_out = manual_input.full_reduce(&mut manual_circuit).reconstruct(&mut manual_circuit);
    manual_circuit.output(vec![manual_out]);

    let encoded_input = encode_nested_rns_poly(
        P_MODULI_BITS,
        MAX_UNREDUCED_MULS,
        &params,
        &input_value,
        Some(q_level),
    );
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let auto_eval = auto_circuit.eval(
        &params,
        one.clone(),
        encoded_input.clone(),
        Some(&plt_evaluator),
        None,
        None,
    );
    let manual_eval =
        manual_circuit.eval(&params, one, encoded_input, Some(&plt_evaluator), None, None);

    assert_eq!(auto_eval, manual_eval);
    assert_eq!(auto_circuit.non_free_depth(), manual_circuit.non_free_depth());
    assert_eq!(auto_circuit.count_gates_by_type_vec(), manual_circuit.count_gates_by_type_vec());
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_slot_transfer_tracks_distinct_q_level_bounds() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, ctx) = create_test_context_with_q_level(&mut circuit, Some(2));
    let input = NestedRnsPoly::input(ctx.clone(), Some(2), None, &mut circuit);
    let src_slots = &[(0, Some(vec![3, 5])), (1, Some(vec![4, 2])), (2, None)];
    let transferred = input.slot_transfer(src_slots, &mut circuit);

    let expected = vec![
        BigUint::from(ctx.q_moduli[0] - 1) * BigUint::from(4u64),
        BigUint::from(ctx.q_moduli[1] - 1) * BigUint::from(5u64),
    ];
    assert_eq!(transferred.max_plaintexts, expected);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_slot_transfer_auto_reduce_matches_manual_full_reduce() {
    let q_level = 1usize;

    let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, ctx) = create_test_context_with_q_level(&mut setup_circuit, Some(q_level));
    let slot_transfer_scale = BigUint::from(2u64);
    let operand_bound = div_ceil_biguint_by_u64(ctx.p_full.clone(), 2);
    let reduced_transfer_bound = &ctx.full_reduce_max_plaintexts[0] * &slot_transfer_scale;
    assert!(&operand_bound * &slot_transfer_scale >= ctx.p_full);
    assert!(reduced_transfer_bound < ctx.p_full);

    let mut auto_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, auto_ctx) = create_test_context_with_q_level(&mut auto_circuit, Some(q_level));
    let mut auto_input =
        NestedRnsPoly::input(auto_ctx.clone(), Some(q_level), None, &mut auto_circuit);
    auto_input.max_plaintexts = vec![operand_bound.clone()];
    let auto_transferred = auto_input
        .slot_transfer(&[(0, Some(vec![1])), (1, Some(vec![2])), (2, None)], &mut auto_circuit);
    assert_eq!(auto_transferred.max_plaintexts, vec![reduced_transfer_bound.clone()]);
    auto_circuit.output(vec![auto_transferred.inner[0]]);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, manual_ctx) = create_test_context_with_q_level(&mut manual_circuit, Some(q_level));
    let mut manual_input =
        NestedRnsPoly::input(manual_ctx.clone(), Some(q_level), None, &mut manual_circuit);
    manual_input.max_plaintexts = vec![operand_bound];
    let manual_transferred = manual_input
        .full_reduce(&mut manual_circuit)
        .slot_transfer(&[(0, Some(vec![1])), (1, Some(vec![2])), (2, None)], &mut manual_circuit);
    assert_eq!(manual_transferred.max_plaintexts, vec![reduced_transfer_bound]);
    manual_circuit.output(vec![manual_transferred.inner[0]]);

    assert_eq!(auto_circuit.non_free_depth(), manual_circuit.non_free_depth());
    assert_eq!(auto_circuit.count_gates_by_type_vec(), manual_circuit.count_gates_by_type_vec());
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_sequential_add_auto_reduce_runs_full_reduce_once() {
    let q_level = 1usize;
    let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context_with_q_level(&mut setup_circuit, Some(q_level));
    let reduce_bound = ctx.full_reduce_max_plaintexts[0].clone();
    let (operand_count, operand_bound) = (2usize..=8usize)
        .find_map(|candidate_count| {
            let count_big = BigUint::from(candidate_count as u64);
            let bound = (&ctx.p_full + &count_big - BigUint::one()) / &count_big;
            let pre_last = &bound * BigUint::from(u64::try_from(candidate_count - 1).unwrap_or(0));
            let final_unreduced = &bound * &count_big;
            if pre_last < ctx.p_full &&
                final_unreduced >= ctx.p_full &&
                &reduce_bound + &bound < ctx.p_full
            {
                Some((candidate_count, bound))
            } else {
                None
            }
        })
        .expect("expected to find a small add-chain that triggers exactly one full_reduce");

    let pre_last_bound =
        &operand_bound * BigUint::from(u64::try_from(operand_count - 1).unwrap_or(0));
    let final_unreduced_bound = &operand_bound * BigUint::from(operand_count as u64);
    assert!(pre_last_bound < ctx.p_full);
    assert!(final_unreduced_bound >= ctx.p_full);
    assert!(&reduce_bound + &operand_bound < ctx.p_full);

    let input_value = BigUint::one();
    let q_level_modulus = BigUint::from(ctx.q_moduli[0]);
    let expected_output =
        (&input_value * BigUint::from(operand_count as u64)) % q_level_modulus.clone();

    let mut auto_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, auto_ctx) = create_test_context_with_q_level(&mut auto_circuit, Some(q_level));
    let auto_inputs = (0..operand_count)
        .map(|_| {
            let input =
                NestedRnsPoly::input(auto_ctx.clone(), Some(q_level), None, &mut auto_circuit);
            NestedRnsPoly::new(
                input.ctx.clone(),
                input.inner.clone(),
                None,
                input.enable_levels,
                vec![operand_bound.clone()],
            )
        })
        .collect::<Vec<_>>();
    let mut auto_sum = auto_inputs[0].clone();
    for poly in auto_inputs.iter().skip(1) {
        auto_sum = auto_sum.add(poly, &mut auto_circuit);
    }
    let auto_out = auto_sum.reconstruct(&mut auto_circuit);
    auto_circuit.output(vec![auto_out]);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, manual_ctx) = create_test_context_with_q_level(&mut manual_circuit, Some(q_level));
    let manual_inputs = (0..operand_count)
        .map(|_| {
            let input =
                NestedRnsPoly::input(manual_ctx.clone(), Some(q_level), None, &mut manual_circuit);
            NestedRnsPoly::new(
                input.ctx.clone(),
                input.inner.clone(),
                None,
                input.enable_levels,
                vec![operand_bound.clone()],
            )
        })
        .collect::<Vec<_>>();
    let mut manual_sum = manual_inputs[0].clone();
    for poly in manual_inputs.iter().skip(1).take(operand_count - 2) {
        manual_sum = manual_sum.add(poly, &mut manual_circuit);
    }
    manual_sum = manual_sum.full_reduce(&mut manual_circuit);
    let last_input = manual_inputs
        .last()
        .expect("manual input chain must contain a final operand")
        .full_reduce(&mut manual_circuit);
    manual_sum = manual_sum.add(&last_input, &mut manual_circuit);
    let manual_out = manual_sum.reconstruct(&mut manual_circuit);
    manual_circuit.output(vec![manual_out]);

    let encoded_input = encode_nested_rns_poly(
        P_MODULI_BITS,
        MAX_UNREDUCED_MULS,
        &params,
        &input_value,
        Some(q_level),
    );
    let eval_inputs = (0..operand_count).flat_map(|_| encoded_input.clone()).collect::<Vec<_>>();
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);

    let auto_eval = auto_circuit.eval(
        &params,
        one.clone(),
        eval_inputs.clone(),
        Some(&plt_evaluator),
        None,
        None,
    );
    let manual_eval =
        manual_circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);

    assert_eq!(auto_eval.len(), 1);
    assert_eq!(manual_eval.len(), 1);
    let auto_output = auto_eval[0].coeffs_biguints()[0].clone();
    let manual_output = manual_eval[0].coeffs_biguints()[0].clone();
    assert_eq!(auto_output, manual_output);
    assert_eq!(auto_output % &q_level_modulus, expected_output);

    let expected_depth = manual_circuit.non_free_depth();
    assert_eq!(auto_circuit.non_free_depth(), expected_depth);
    assert_eq!(auto_circuit.count_gates_by_type_vec(), manual_circuit.count_gates_by_type_vec());
}

#[sequential_test::sequential]
#[test]
fn test_encode_nested_rns_poly_compact_bytes_matches_polys() {
    let params = DCRTPolyParams::new(4, 6, 18, BASE_BITS);
    let input = BigUint::from(12345u64);
    let expected = encode_nested_rns_poly::<DCRTPoly>(
        P_MODULI_BITS,
        MAX_UNREDUCED_MULS,
        &params,
        &input,
        Some(2),
    )
    .into_iter()
    .map(|poly| poly.to_compact_bytes())
    .collect::<Vec<_>>();
    let actual = encode_nested_rns_poly_compact_bytes::<DCRTPoly>(
        P_MODULI_BITS,
        MAX_UNREDUCED_MULS,
        &params,
        &input,
        Some(2),
    );
    assert_eq!(actual, expected);
}

#[sequential_test::sequential]
#[test]
fn test_sample_crt_primes_respects_configured_mul_budget() {
    let q_max = 43u64;
    let max_unreduced_muls = 4usize;
    let p_moduli = sample_crt_primes(P_MODULI_BITS, q_max, max_unreduced_muls);
    let prod = p_moduli.iter().fold(BigUint::one(), |acc, &pi| acc * BigUint::from(pi));
    let sum = p_moduli.iter().copied().sum::<u64>();
    let mul_budget_bound = sample_crt_primes_mul_budget_bound(sum, p_moduli.len(), q_max);

    assert!(pow_biguint_usize(&mul_budget_bound, max_unreduced_muls) < prod);

    let smaller_budget_prod = sample_crt_primes(P_MODULI_BITS, q_max, MAX_UNREDUCED_MULS)
        .iter()
        .fold(BigUint::one(), |acc, &pi| acc * BigUint::from(pi));
    assert!(smaller_budget_prod < prod);
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_sequential_mul_auto_reduce_uses_extended_mul_budget() {
    let q_level = 1usize;
    let p_moduli_bits = 10usize;
    let max_unreduced_muls = 4usize;
    let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context_with_config(
        &mut setup_circuit,
        Some(q_level),
        p_moduli_bits,
        max_unreduced_muls,
    );
    let reduce_bound = ctx.full_reduce_max_plaintexts[0].clone();
    let operand_count = max_unreduced_muls + 1;
    let operand_bound = ceil_biguint_nth_root(&ctx.p_full, operand_count);

    let pre_last_bound = pow_biguint_usize(&operand_bound, operand_count - 1);
    let final_unreduced_bound = &pre_last_bound * &operand_bound;
    assert!(pre_last_bound < ctx.p_full);
    assert!(final_unreduced_bound >= ctx.p_full);
    assert!(pow_biguint_usize(&reduce_bound, 2) < ctx.p_full);

    let input_value = BigUint::one();
    let q_level_modulus = BigUint::from(ctx.q_moduli[0]);
    let expected_output = input_value.clone() % q_level_modulus.clone();

    let mut auto_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, auto_ctx) = create_test_context_with_config(
        &mut auto_circuit,
        Some(q_level),
        p_moduli_bits,
        max_unreduced_muls,
    );
    let auto_inputs = (0..operand_count)
        .map(|_| {
            let input =
                NestedRnsPoly::input(auto_ctx.clone(), Some(q_level), None, &mut auto_circuit);
            NestedRnsPoly::new(
                input.ctx.clone(),
                input.inner.clone(),
                None,
                input.enable_levels,
                vec![operand_bound.clone()],
            )
        })
        .collect::<Vec<_>>();
    let mut auto_product = auto_inputs[0].clone();
    for poly in auto_inputs.iter().skip(1) {
        auto_product = auto_product.mul(poly, &mut auto_circuit);
    }
    let auto_out = auto_product.reconstruct(&mut auto_circuit);
    auto_circuit.output(vec![auto_out]);

    let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, manual_ctx) = create_test_context_with_config(
        &mut manual_circuit,
        Some(q_level),
        p_moduli_bits,
        max_unreduced_muls,
    );
    let manual_inputs = (0..operand_count)
        .map(|_| {
            let input =
                NestedRnsPoly::input(manual_ctx.clone(), Some(q_level), None, &mut manual_circuit);
            NestedRnsPoly::new(
                input.ctx.clone(),
                input.inner.clone(),
                None,
                input.enable_levels,
                vec![operand_bound.clone()],
            )
        })
        .collect::<Vec<_>>();
    let mut manual_product = manual_inputs[0].clone();
    for poly in manual_inputs.iter().skip(1).take(operand_count - 2) {
        manual_product = manual_product.mul(poly, &mut manual_circuit);
    }
    manual_product = manual_product.full_reduce(&mut manual_circuit);
    let last_input = manual_inputs
        .last()
        .expect("manual input chain must contain a final operand")
        .full_reduce(&mut manual_circuit);
    manual_product = manual_product.mul(&last_input, &mut manual_circuit);
    let manual_out = manual_product.reconstruct(&mut manual_circuit);
    manual_circuit.output(vec![manual_out]);

    let encoded_input = encode_nested_rns_poly(
        p_moduli_bits,
        max_unreduced_muls,
        &params,
        &input_value,
        Some(q_level),
    );
    let eval_inputs = (0..operand_count).flat_map(|_| encoded_input.clone()).collect::<Vec<_>>();
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);

    let auto_eval = auto_circuit.eval(
        &params,
        one.clone(),
        eval_inputs.clone(),
        Some(&plt_evaluator),
        None,
        None,
    );
    let manual_eval =
        manual_circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);

    assert_eq!(auto_eval.len(), 1);
    assert_eq!(manual_eval.len(), 1);
    let auto_output = auto_eval[0].coeffs_biguints()[0].clone();
    let manual_output = manual_eval[0].coeffs_biguints()[0].clone();
    assert_eq!(auto_output, manual_output);
    assert_eq!(auto_output % &q_level_modulus, expected_output);

    let expected_depth = manual_circuit.non_free_depth();
    assert_eq!(auto_circuit.non_free_depth(), expected_depth);
    assert_eq!(auto_circuit.count_gates_by_type_vec(), manual_circuit.count_gates_by_type_vec());
}

fn test_nested_rns_poly_add_generic(
    mut circuit: PolyCircuit<DCRTPoly>,
    params: DCRTPolyParams,
    ctx: Arc<NestedRnsPolyContext>,
    a_value: BigUint,
    b_value: BigUint,
) {
    let poly_a = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let sum = poly_a.add(&poly_b, &mut circuit);
    let out = sum.reconstruct(&mut circuit);
    circuit.output(vec![out]);
    println!("non-free depth {}", circuit.non_free_depth());
    println!("circuit size {:?}", circuit.count_gates_by_type_vec());

    let modulus = params.modulus();
    // println!("modulus {:?}", &modulus);
    // println!("a_value {:?}", &a_value);
    // println!("b_value {:?}", &b_value);
    let a_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &a_value, None);
    let b_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &b_value, None);
    let expected_out = (&a_value + &b_value) % modulus.as_ref();
    // println!("expected_out {:?}", &expected_out);
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let eval_inputs = [a_inputs, b_inputs].concat();
    let eval_results = circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
    // println!("eval_results {:?}", eval_results);
    assert_eq!(eval_results.len(), 1);
    assert_eq!(eval_results[0].coeffs_biguints()[0], expected_out);
}

fn test_nested_rns_poly_sub_generic(
    mut circuit: PolyCircuit<DCRTPoly>,
    params: DCRTPolyParams,
    ctx: Arc<NestedRnsPolyContext>,
    a_value: BigUint,
    b_value: BigUint,
) {
    let poly_a = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let sum = poly_a.sub(&poly_b, &mut circuit);
    let out = sum.reconstruct(&mut circuit);
    circuit.output(vec![out]);
    println!("non-free depth {}", circuit.non_free_depth());
    println!("circuit size {:?}", circuit.count_gates_by_type_vec());

    let modulus = params.modulus();
    // println!("modulus {:?}", &modulus);
    // println!("a_value {:?}", &a_value);
    // println!("b_value {:?}", &b_value);
    let a_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &a_value, None);
    let b_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &b_value, None);
    let expected_out = {
        let mut value = &a_value + modulus.as_ref();
        value -= &b_value;
        value %= modulus.as_ref();
        value
    };
    // println!("expected_out {:?}", &expected_out);
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let eval_inputs = [a_inputs, b_inputs].concat();
    let eval_results = circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
    // println!("eval_results {:?}", eval_results);
    assert_eq!(eval_results.len(), 1);
    assert_eq!(eval_results[0].coeffs_biguints()[0], expected_out);
}

fn test_nested_rns_poly_mul_generic(
    mut circuit: PolyCircuit<DCRTPoly>,
    params: DCRTPolyParams,
    ctx: Arc<NestedRnsPolyContext>,
    a_value: BigUint,
    b_value: BigUint,
) {
    let poly_a = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let sum = poly_a.mul(&poly_b, &mut circuit);
    let out = sum.reconstruct(&mut circuit);
    circuit.output(vec![out]);
    println!("non-free depth {}", circuit.non_free_depth());
    println!("circuit size {:?}", circuit.count_gates_by_type_vec());

    let modulus = params.modulus();
    let (q_moduli, _, _) = params.to_crt();
    let active_q_level = ctx.q_moduli_depth;
    let q_level_modulus = q_moduli
        .iter()
        .take(active_q_level)
        .fold(BigUint::from(1u64), |acc, &qi| acc * BigUint::from(qi));
    let a_inputs = encode_nested_rns_poly(
        P_MODULI_BITS,
        MAX_UNREDUCED_MULS,
        &params,
        &a_value,
        Some(active_q_level),
    );
    let b_inputs = encode_nested_rns_poly(
        P_MODULI_BITS,
        MAX_UNREDUCED_MULS,
        &params,
        &b_value,
        Some(active_q_level),
    );
    let expected_mod_q_level =
        ((&a_value % &q_level_modulus) * (&b_value % &q_level_modulus)) % &q_level_modulus;
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let eval_inputs = [a_inputs, b_inputs].concat();
    let eval_results = circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
    assert_eq!(eval_results.len(), 1);
    let output = eval_results[0].coeffs_biguints()[0].clone();
    if active_q_level == q_moduli.len() {
        let expected_out = (&a_value * &b_value) % modulus.as_ref();
        assert_eq!(output, expected_out);
    } else {
        assert_eq!(output.clone() % &q_level_modulus, expected_mod_q_level);
    }
}

fn test_nested_rns_poly_mul_reconstruct_generic(
    mut circuit: PolyCircuit<DCRTPoly>,
    params: DCRTPolyParams,
    ctx: Arc<NestedRnsPolyContext>,
    a_value: BigUint,
    b_value: BigUint,
) {
    let poly_a = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let product = poly_a.mul(&poly_b, &mut circuit);
    let out = product.reconstruct(&mut circuit);
    circuit.output(vec![out]);

    let modulus = params.modulus();
    let a_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &a_value, None);
    let b_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &b_value, None);
    let expected_out = (&a_value * &b_value) % modulus.as_ref();
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let eval_inputs = [a_inputs, b_inputs].concat();
    let eval_results = circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
    assert_eq!(eval_results.len(), 1);
    assert_eq!(eval_results[0].coeffs_biguints()[0], expected_out);
}

fn test_nested_rns_poly_const_mul_generic(
    mut circuit: PolyCircuit<DCRTPoly>,
    params: DCRTPolyParams,
    ctx: Arc<NestedRnsPolyContext>,
    a_value: BigUint,
    tower_constants: Vec<u64>,
) {
    let poly = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
    let product = poly.const_mul(&tower_constants, &mut circuit);
    let out = product.reconstruct(&mut circuit);
    circuit.output(vec![out]);

    let a_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &a_value, None);
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let eval_results = circuit.eval(&params, one, a_inputs, Some(&plt_evaluator), None, None);
    assert_eq!(eval_results.len(), 1);

    let (q_moduli, _, _) = params.to_crt();
    let output = eval_results[0].coeffs_biguints()[0].clone();
    for (q_idx, &q_i) in q_moduli.iter().take(ctx.q_moduli_depth).enumerate() {
        let q_big = BigUint::from(q_i);
        let expected = ((&a_value % &q_big) * BigUint::from(tower_constants[q_idx])) % &q_big;
        assert_eq!(output.clone() % &q_big, expected, "tower {q_idx} mismatch");
    }
}

#[sequential_test::sequential]
#[test]
fn test_nested_rns_poly_mul_with_enable_levels_field() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (params, ctx) = create_test_context(&mut circuit);
    let enable_levels = Some(2usize);
    let poly_a = NestedRnsPoly::input(ctx.clone(), enable_levels, None, &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx, enable_levels, None, &mut circuit);
    let product = poly_a.mul(&poly_b, &mut circuit);
    let out = product.reconstruct(&mut circuit);
    circuit.output(vec![out]);

    let (q_moduli, _, _) = params.to_crt();
    let q_level_modulus = q_moduli
        .iter()
        .take(enable_levels.expect("test uses a fixed q level"))
        .fold(BigUint::from(1u64), |acc, &qi| acc * BigUint::from(qi));
    let a_value = &q_level_modulus - BigUint::from(2u64);
    let b_value = &q_level_modulus - BigUint::from(3u64);
    let expected =
        ((&a_value % &q_level_modulus) * (&b_value % &q_level_modulus)) % &q_level_modulus;
    let a_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &a_value, enable_levels);
    let b_inputs =
        encode_nested_rns_poly(P_MODULI_BITS, MAX_UNREDUCED_MULS, &params, &b_value, enable_levels);
    let plt_evaluator = PolyPltEvaluator::new();
    let one = DCRTPoly::const_one(&params);
    let eval_inputs = [a_inputs, b_inputs].concat();
    let eval_results = circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
    assert_eq!(eval_results.len(), 1);
    let output_coeffs = eval_results[0].coeffs_biguints();
    assert_eq!(output_coeffs[0].clone() % &q_level_modulus, expected);
}

#[sequential_test::sequential]
#[test]
#[should_panic(expected = "mismatched enable_levels")]
fn test_nested_rns_poly_binary_ops_require_matching_enable_levels() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let (_, ctx) = create_test_context(&mut circuit);
    let poly_a = NestedRnsPoly::input(ctx.clone(), Some(1), None, &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx, Some(2), None, &mut circuit);
    let _ = poly_a.add(&poly_b, &mut circuit);
}
