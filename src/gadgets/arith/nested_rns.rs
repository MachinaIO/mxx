#[cfg(feature = "gpu")]
#[path = "nested_rns_gpu.rs"]
mod gpu;

use crate::{
    circuit::{BatchedWire, PolyCircuit, SubCircuitParamKind, SubCircuitParamValue, gate::GateId},
    gadgets::conv_mul::{
        negacyclic_conv_mul_right_decomposed_term_many_subcircuit, negacyclic_conv_mul_right_sparse,
    },
    lookup::PublicLut,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    utils::{mod_inverse, pow_biguint_usize, round_div},
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};
use tracing::{debug, info};

pub const DEFAULT_MAX_UNREDUCED_MULS: usize = 2;

fn single_wire_gate_ids(wires: Vec<BatchedWire>) -> Vec<GateId> {
    wires.into_iter().map(BatchedWire::as_single_wire).collect()
}

#[derive(Debug, Clone)]
pub struct NestedRnsPolyContext {
    pub p_moduli_bits: usize,
    pub max_unreduced_muls: usize,
    pub scale: u64,
    pub p_moduli: Vec<u64>,
    q_moduli: Vec<u64>,
    pub q_moduli_depth: usize,
    p_max: u64,
    lut_mod_p_max_map_size: BigUint,
    p_full: BigUint,
    p_over_pis: Vec<BigUint>,
    gadget_values: Vec<Vec<BigUint>>,
    pub full_reduce_max_plaintexts: Vec<BigUint>,
    lut_mod_p_ids: Vec<usize>,
    lut_x_to_y_ids: Vec<usize>,
    lut_x_to_real_ids: Vec<usize>,
    lut_real_to_v_id: usize,
    add_without_reduce_id: usize,
    sub_with_trace_offsets_id: usize,
    lazy_reduce_id: usize,
    decomposition_terms_id: usize,
    gadget_decompose_id: usize,
    full_reduce_id: usize,
    full_reduce_bindings: Vec<Vec<SubCircuitParamValue>>,
    mul_lazy_reduce_id: usize,
    mul_right_sparse_id: usize,
}

fn dummy_lut<P: Poly + 'static>(params: &P::Params) -> PublicLut<P> {
    PublicLut::new(
        params,
        1,
        |params, x| {
            if x != 0 {
                return None;
            }
            let y_elem = P::from_usize_to_constant(params, 0)
                .coeffs()
                .into_iter()
                .next()
                .expect("constant-term coefficient must exist");
            Some((0, y_elem))
        },
        None,
    )
}

fn max_output_row_from_biguint<P: Poly>(
    params: &P::Params,
    idx: usize,
    value: BigUint,
) -> (u64, P::Elem) {
    let poly = P::from_biguint_to_constant(params, value);
    let coeff =
        poly.coeffs().into_iter().max().expect("max_output_row requires at least one coefficient");
    (u64::try_from(idx).expect("row index must fit in u64"), coeff)
}

// Conservative output bound for the integer represented by one q-level after full_reduce.
//
// The current implementation uses canonical nonnegative residues throughout:
// - each y_i produced by lut_x_to_y lies in [0, p_i),
// - each coefficient [p_hat_i]_q and [p]_q is represented in [0, q),
// - real_i = round(y_i * scale / p_i) is nonnegative, and because y_i <= p_i - 1 we have real_i <=
//   scale, so v = round(sum_i real_i / scale) satisfies 0 <= v <= k where k = p_moduli.len().
//
// For one q-level with modulus q, full_reduce evaluates the integer
//   x' = sum_i y_i * [p_hat_i]_q - v * [p]_q.
// Using the bounds above,
//   |x'|
//   <= sum_i |y_i| * |[p_hat_i]_q| + |v| * |[p]_q|
//   <  sum_i p_i * q + k * q
//   =  (sum_i p_i + k) * q.
//
// This is intentionally looser than the centered-residue bound from the paper: it matches the
// repository's [0, q) / [0, p_i) representation contract rather than the paper's symmetric one.
fn full_reduce_output_max_plaintext_bound(p_moduli: &[u64], q_modulus: u64) -> BigUint {
    let sum_p_moduli = p_moduli.iter().fold(BigUint::ZERO, |acc, &p_i| acc + BigUint::from(p_i));
    let modulus_count =
        u64::try_from(p_moduli.len()).expect("p_moduli length must fit in u64 for bound tracking");
    (sum_p_moduli + BigUint::from(modulus_count)) * BigUint::from(q_modulus)
}

fn sample_crt_primes_mul_budget_bound(
    sum_p_moduli: u64,
    modulus_count: usize,
    q_max: u64,
) -> BigUint {
    let modulus_count =
        u64::try_from(modulus_count).expect("p_moduli length must fit in u64 for bound tracking");
    BigUint::from(sum_p_moduli + modulus_count) * BigUint::from(q_max) / BigUint::from(2u64)
}

fn lut_mod_p_map_size(p_i: u64, max_p_modulus: u64, modulus_count: usize) -> u128 {
    (p_i as u128 * max_p_modulus as u128).max(p_i as u128 * (2 * modulus_count) as u128)
}

fn precompute_nested_rns_gadget_values(
    q_moduli: &[u64],
    p_full: &BigUint,
    p_over_pis: &[BigUint],
) -> Vec<Vec<BigUint>> {
    q_moduli
        .iter()
        .map(|&q_i| {
            let q_i_big = BigUint::from(q_i);
            let p_mod_qi = p_full % &q_i_big;
            let mut level_values =
                p_over_pis.iter().map(|p_hat_i| p_hat_i % &q_i_big).collect::<Vec<_>>();
            level_values.push(if p_mod_qi == BigUint::ZERO {
                BigUint::ZERO
            } else {
                &q_i_big - &p_mod_qi
            });
            level_values
        })
        .collect()
}

fn full_reduce_param_bindings(
    scalars_y: &[Vec<u32>],
    scalars_v: &[u32],
) -> Vec<SubCircuitParamValue> {
    let (mut y_bindings, v_bindings) = rayon::join(
        || {
            scalars_y
                .par_iter()
                .flat_map_iter(|row| {
                    row.iter().map(|scalar| SubCircuitParamValue::SmallScalarMul(vec![*scalar]))
                })
                .collect::<Vec<_>>()
        },
        || {
            scalars_v
                .par_iter()
                .map(|scalar| SubCircuitParamValue::SmallScalarMul(vec![*scalar]))
                .collect::<Vec<_>>()
        },
    );
    let total_len = y_bindings.len() + v_bindings.len();
    y_bindings.reserve(v_bindings.len());
    y_bindings.extend(v_bindings);
    debug_assert_eq!(y_bindings.len(), total_len);
    let bindings = y_bindings;
    bindings
}

fn sub_with_trace_offset_param_bindings(
    offset_multiplier: &BigUint,
    p_moduli: &[u64],
) -> Vec<SubCircuitParamValue> {
    p_moduli
        .par_iter()
        .map(|&p_i| {
            SubCircuitParamValue::LargeScalarMul(vec![offset_multiplier * BigUint::from(p_i)])
        })
        .collect()
}

impl NestedRnsPolyContext {
    pub(crate) fn q_moduli(&self) -> &[u64] {
        &self.q_moduli
    }

    pub(crate) fn full_reduce_output_metadata(
        &self,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> (Vec<BigUint>, Vec<BigUint>) {
        let level_offset = level_offset.unwrap_or(0);
        let input_count = enable_levels.unwrap_or(self.q_moduli_depth);
        assert!(
            level_offset + input_count <= self.q_moduli_depth,
            "active range exceeds q_moduli_depth: level_offset={level_offset}, enable_levels={input_count}, q_moduli_depth={}",
            self.q_moduli_depth
        );
        let max_plaintexts = self.full_reduce_max_plaintexts
            [level_offset..level_offset + input_count]
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let p_max_traces = vec![self.reduced_p_max_trace(); input_count];
        (max_plaintexts, p_max_traces)
    }

    pub(crate) fn reduced_p_max_trace(&self) -> BigUint {
        BigUint::from(self.p_max - 1)
    }

    pub(crate) fn p_full_ref(&self) -> &BigUint {
        &self.p_full
    }

    pub(crate) fn lut_mod_p_max_map_size_ref(&self) -> &BigUint {
        &self.lut_mod_p_max_map_size
    }

    fn unreduced_trace_threshold(&self) -> BigUint {
        BigUint::from(self.p_max)
    }

    pub fn setup<P: Poly + 'static>(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        p_moduli_bits: usize,
        max_unreduced_muls: usize,
        scale: u64,
        dummy_scalar: bool,
        q_level: Option<usize>,
    ) -> Self {
        let (q_moduli, _q_moduli_bits, max_q_moduli_depth) = params.to_crt();
        let q_moduli_depth = q_level.unwrap_or(max_q_moduli_depth);
        assert!(
            q_moduli_depth <= max_q_moduli_depth,
            "q_level exceeds q_moduli_depth: q_level={}, q_moduli_depth={}",
            q_moduli_depth,
            max_q_moduli_depth
        );
        let q_moduli_min = q_moduli.iter().min().expect("there should be at least one q modulus");
        let q_moduli_max = q_moduli.iter().max().expect("there should be at least one q modulus");
        let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max, max_unreduced_muls);
        debug!(
            "NestedRnsPolyContext setup: p_moduli = {:?}, q_moduli = {:?}, scale = {}, max_unreduced_muls = {}",
            p_moduli, q_moduli, scale, max_unreduced_muls
        );
        let p_moduli_depth = p_moduli.len();
        let max_p_modulus = *p_moduli.iter().max().expect("p_moduli must not be empty");
        let p_moduli_depth_u64 =
            u64::try_from(p_moduli_depth).expect("p_moduli length must fit in u64");
        assert!(
            p_moduli_depth_u64 < max_p_modulus,
            "NestedRnsPolyContext requires p_moduli.len() < p_max, got s={} and p_max={}",
            p_moduli_depth,
            max_p_modulus
        );
        let lut_mod_p_max_map_size =
            BigUint::from(lut_mod_p_map_size(max_p_modulus, max_p_modulus, p_moduli_depth));
        let active_q_moduli = q_moduli.iter().take(q_moduli_depth).copied().collect::<Vec<_>>();
        if dummy_scalar {
            let dummy_lut = dummy_lut::<P>(params);
            let dummy_lut_id = circuit.register_public_lookup(dummy_lut);
            let lut_mod_p_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_y_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_real_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_real_to_v_id = dummy_lut_id;
            let p_full =
                p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
            let full_reduce_max_plaintexts = active_q_moduli
                .iter()
                .map(|&q_i| full_reduce_output_max_plaintext_bound(&p_moduli, q_i))
                .collect::<Vec<_>>();
            let p_over_pis =
                p_moduli.iter().map(|&p_i| &p_full / BigUint::from(p_i)).collect::<Vec<_>>();
            let gadget_values =
                precompute_nested_rns_gadget_values(&active_q_moduli, &p_full, &p_over_pis);
            let mut scalars_y = vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth];
            let mut scalars_v = vec![vec![0; p_moduli_depth]; q_moduli_depth];
            for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
                for (q_idx, q_k) in q_moduli.iter().take(q_moduli_depth).enumerate() {
                    for (p_j_idx, p_over_pj) in p_over_pis.iter().enumerate() {
                        let p_over_pj_mod_qk = (p_over_pj % BigUint::from(*q_k))
                            .to_u64()
                            .expect("CRT residue must fit in u64");
                        let p_over_pj_mod_qk_mod_pi = p_over_pj_mod_qk % p_i;
                        scalars_y[q_idx][p_i_idx][p_j_idx] = p_over_pj_mod_qk_mod_pi as u32;
                    }
                    let p_mod_qk = (&p_full % BigUint::from(*q_k))
                        .to_u64()
                        .expect("CRT residue must fit in u64");
                    let p_mod_qk_mod_pi = p_mod_qk % p_i;
                    scalars_v[q_idx][p_i_idx] = p_mod_qk_mod_pi as u32;
                }
            }
            let add_without_reduce_id =
                circuit.register_sub_circuit(Self::add_without_reduce_subcircuit::<P>(&p_moduli));
            let sub_with_trace_offsets_id = circuit
                .register_sub_circuit(Self::sub_with_trace_offsets_subcircuit::<P>(p_moduli_depth));
            let mul_lazy_reduce_id = circuit.register_sub_circuit(
                Self::mul_lazy_reduce_subcircuit::<P>(&p_moduli, &lut_mod_p_ids),
            );
            let mul_right_sparse_id = circuit.register_sub_circuit(
                Self::mul_right_sparse_subcircuit::<P>(&p_moduli, &lut_mod_p_ids),
            );
            let lazy_reduce_id = circuit
                .register_sub_circuit(Self::lazy_reduce_subcircuit::<P>(&p_moduli, &lut_mod_p_ids));
            let decomposition_terms_id =
                circuit.register_sub_circuit(Self::decomposition_terms_subcircuit::<P>(
                    &lut_x_to_y_ids,
                    &lut_x_to_real_ids,
                    lut_real_to_v_id,
                ));
            let gadget_decompose_id =
                circuit.register_sub_circuit(Self::gadget_decompose_subcircuit::<P>(
                    &p_moduli,
                    &lut_mod_p_ids,
                    &lut_x_to_y_ids,
                    &lut_x_to_real_ids,
                    lut_real_to_v_id,
                ));
            let full_reduce_id = circuit.register_sub_circuit(Self::full_reduce_subcircuit::<P>(
                &p_moduli,
                &lut_mod_p_ids,
                &lut_x_to_y_ids,
                &lut_x_to_real_ids,
                lut_real_to_v_id,
            ));
            let full_reduce_bindings = (0..q_moduli_depth)
                .into_par_iter()
                .map(|q_idx| full_reduce_param_bindings(&scalars_y[q_idx], &scalars_v[q_idx]))
                .collect::<Vec<_>>();
            return Self {
                p_moduli_bits,
                max_unreduced_muls,
                scale,
                p_moduli,
                q_moduli: active_q_moduli,
                q_moduli_depth,
                p_max: max_p_modulus,
                lut_mod_p_max_map_size,
                p_full,
                p_over_pis,
                gadget_values,
                full_reduce_max_plaintexts,
                lut_mod_p_ids,
                lut_x_to_y_ids,
                lut_x_to_real_ids,
                lut_real_to_v_id,
                add_without_reduce_id,
                sub_with_trace_offsets_id,
                lazy_reduce_id,
                decomposition_terms_id,
                gadget_decompose_id,
                full_reduce_id,
                full_reduce_bindings,
                mul_lazy_reduce_id,
                mul_right_sparse_id,
            };
        }

        let p_full = p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        let full_reduce_max_plaintexts = active_q_moduli
            .iter()
            .map(|&q_i| full_reduce_output_max_plaintext_bound(&p_moduli, q_i))
            .collect::<Vec<_>>();
        let p_over_pis =
            p_moduli.iter().map(|&p_i| &p_full / BigUint::from(p_i)).collect::<Vec<_>>();
        let gadget_values =
            precompute_nested_rns_gadget_values(&active_q_moduli, &p_full, &p_over_pis);

        let mut lut_mod_p = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_y = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_real = Vec::with_capacity(p_moduli_depth);
        let mut scalars_y = vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth];
        let mut scalars_v = vec![vec![0; p_moduli_depth]; q_moduli_depth];

        for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
            let lut_mod_p_map_size = lut_mod_p_map_size(p_i, max_p_modulus, p_moduli.len());
            // .checked_mul(max_p_modulus as u128)
            // .expect("lut_mod_p_map_size overflow");
            debug_assert!(
                lut_mod_p_map_size < *q_moduli_min as u128,
                "LUT size exceeds q modulus size; increase q_moduli_bits or decrease p_moduli_bits"
            );
            let lut_mod_p_len = lut_mod_p_map_size as usize;
            let max_mod_p_row = max_output_row_from_biguint::<P>(
                params,
                (p_i - 1) as usize,
                BigUint::from(p_i - 1),
            );
            let lut_mod_p_lut = PublicLut::<P>::new(
                params,
                lut_mod_p_len as u64,
                move |params, t| {
                    if t >= lut_mod_p_len as u64 {
                        return None;
                    }
                    let output = BigUint::from(t % p_i);
                    let y_elem = P::from_biguint_to_constant(params, output)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist");
                    Some((t, y_elem))
                },
                Some(max_mod_p_row),
            );
            info!("Constructed lut_mod_p for p_{} = {} with size {}", p_i_idx, p_i, lut_mod_p_len);
            lut_mod_p.push(lut_mod_p_lut);

            let p_moduli_big = BigUint::from(p_i);
            let p_over_pi_mod_pi = (&p_over_pis[p_i_idx] % &p_moduli_big)
                .to_u64()
                .expect("CRT residue must fit in u64");
            let p_over_pi_inv = {
                let inv = mod_inverse(p_over_pi_mod_pi, p_i).expect("CRT moduli must be coprime");
                // info!("{}", format!("pi = {}, p_over_pi = {}, inv = {}", pi, p_over_pi, inv));
                BigUint::from(inv)
            };

            let max_idx_mod_pi =
                (((p_i - 1) as u128 * p_over_pi_mod_pi as u128) % p_i as u128) as usize;
            let max_x_to_y_row =
                max_output_row_from_biguint::<P>(params, max_idx_mod_pi, BigUint::from(p_i - 1));
            let max_x_to_real_value = round_div((p_i - 1) * scale, p_i);
            let max_x_to_real_row = max_output_row_from_biguint::<P>(
                params,
                max_idx_mod_pi,
                BigUint::from(max_x_to_real_value),
            );

            let p_over_pi_inv = Arc::new(p_over_pi_inv);
            let p_moduli_big = Arc::new(p_moduli_big);
            let lut_x_to_y_len = lut_mod_p_map_size as usize;
            let lut_x_to_y_lut = PublicLut::<P>::new(
                params,
                lut_x_to_y_len as u64,
                {
                    let p_over_pi_inv = Arc::clone(&p_over_pi_inv);
                    let p_moduli_big = Arc::clone(&p_moduli_big);
                    move |params, t| {
                        if t >= lut_x_to_y_len as u64 {
                            return None;
                        }
                        let input = BigUint::from(t);
                        let output = (&input * p_over_pi_inv.as_ref()) % p_moduli_big.as_ref();
                        let y_elem = P::from_biguint_to_constant(params, output)
                            .coeffs()
                            .into_iter()
                            .next()
                            .expect("constant-term coefficient must exist");
                        Some((t, y_elem))
                    }
                },
                Some(max_x_to_y_row),
            );
            info!(
                "Constructed lut_x_to_y for p_{} = {} with size {}",
                p_i_idx, p_i, lut_x_to_y_len
            );
            lut_x_to_y.push(lut_x_to_y_lut);

            let lut_x_to_real_len = lut_mod_p_map_size as usize;
            let lut_x_to_real_lut = PublicLut::<P>::new(
                params,
                lut_x_to_real_len as u64,
                {
                    let p_over_pi_inv = Arc::clone(&p_over_pi_inv);
                    let p_moduli_big = Arc::clone(&p_moduli_big);
                    move |params, t| {
                        if t >= lut_x_to_real_len as u64 {
                            return None;
                        }
                        let input = BigUint::from(t);
                        let y = ((&input * p_over_pi_inv.as_ref()) % p_moduli_big.as_ref())
                            .to_u64()
                            .expect("y must fit in u64");
                        let output = BigUint::from(round_div(y * scale, p_i));
                        let y_elem = P::from_biguint_to_constant(params, output)
                            .coeffs()
                            .into_iter()
                            .next()
                            .expect("constant-term coefficient must exist");
                        Some((t, y_elem))
                    }
                },
                Some(max_x_to_real_row),
            );
            info!(
                "Constructed lut_x_to_real for p_{} = {} with size {}",
                p_i_idx, p_i, lut_x_to_real_len
            );
            lut_x_to_real.push(lut_x_to_real_lut);

            debug!("Computed LUTs for p_{} = {}", p_i_idx, p_i);
            for (q_idx, q_k) in q_moduli.iter().take(q_moduli_depth).enumerate() {
                for (p_j_idx, p_over_pj) in p_over_pis.iter().enumerate() {
                    let p_over_pj_mod_qk = (p_over_pj % BigUint::from(*q_k))
                        .to_u64()
                        .expect("CRT residue must fit in u64");
                    let p_over_pj_mod_qk_mod_pi = p_over_pj_mod_qk % p_i;
                    scalars_y[q_idx][p_i_idx][p_j_idx] = p_over_pj_mod_qk_mod_pi as u32;
                }
                let p_mod_qk =
                    (&p_full % BigUint::from(*q_k)).to_u64().expect("CRT residue must fit in u64");
                let p_mod_qk_mod_pi = p_mod_qk % p_i;
                scalars_v[q_idx][p_i_idx] = p_mod_qk_mod_pi as u32;
                debug!("Computed scalars for q_{} = {} and p_{} = {}", q_idx, q_k, p_i_idx, p_i);
            }
        }

        let max_real = scale * p_moduli_depth as u64;
        let lut_real_to_v_len = max_real as usize + 1;
        let max_real_to_v_row = max_output_row_from_biguint::<P>(
            params,
            max_real as usize,
            BigUint::from(round_div(max_real, scale)),
        );
        let lut_real_to_v_lut = PublicLut::<P>::new(
            params,
            lut_real_to_v_len as u64,
            move |params, t| {
                if t >= lut_real_to_v_len as u64 {
                    return None;
                }
                let output = BigUint::from(round_div(t, scale));
                let y_elem = P::from_biguint_to_constant(params, output)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist");
                Some((t, y_elem))
            },
            Some(max_real_to_v_row),
        );
        info!("Constructed lut_real_to_v with size {}", lut_real_to_v_len);
        let lut_real_to_v = lut_real_to_v_lut;

        let lut_mod_p_ids = lut_mod_p
            .iter()
            .map(|lut| circuit.register_public_lookup(lut.clone()))
            .collect::<Vec<_>>();
        let lut_x_to_y_ids = lut_x_to_y
            .iter()
            .map(|lut| circuit.register_public_lookup(lut.clone()))
            .collect::<Vec<_>>();
        let lut_x_to_real_ids = lut_x_to_real
            .iter()
            .map(|lut| circuit.register_public_lookup(lut.clone()))
            .collect::<Vec<_>>();
        let lut_real_to_v_id = circuit.register_public_lookup(lut_real_to_v.clone());

        let add_without_reduce_id =
            circuit.register_sub_circuit(Self::add_without_reduce_subcircuit::<P>(&p_moduli));
        let sub_with_trace_offsets_id = circuit
            .register_sub_circuit(Self::sub_with_trace_offsets_subcircuit::<P>(p_moduli_depth));
        let mul_lazy_reduce_id = circuit
            .register_sub_circuit(Self::mul_lazy_reduce_subcircuit::<P>(&p_moduli, &lut_mod_p_ids));
        let mul_right_sparse_id = circuit.register_sub_circuit(
            Self::mul_right_sparse_subcircuit::<P>(&p_moduli, &lut_mod_p_ids),
        );
        let lazy_reduce_id = circuit
            .register_sub_circuit(Self::lazy_reduce_subcircuit::<P>(&p_moduli, &lut_mod_p_ids));
        let decomposition_terms_id =
            circuit.register_sub_circuit(Self::decomposition_terms_subcircuit::<P>(
                &lut_x_to_y_ids,
                &lut_x_to_real_ids,
                lut_real_to_v_id,
            ));
        let gadget_decompose_id =
            circuit.register_sub_circuit(Self::gadget_decompose_subcircuit::<P>(
                &p_moduli,
                &lut_mod_p_ids,
                &lut_x_to_y_ids,
                &lut_x_to_real_ids,
                lut_real_to_v_id,
            ));
        let full_reduce_id = circuit.register_sub_circuit(Self::full_reduce_subcircuit::<P>(
            &p_moduli,
            &lut_mod_p_ids,
            &lut_x_to_y_ids,
            &lut_x_to_real_ids,
            lut_real_to_v_id,
        ));
        let full_reduce_bindings = (0..q_moduli_depth)
            .into_par_iter()
            .map(|q_idx| full_reduce_param_bindings(&scalars_y[q_idx], &scalars_v[q_idx]))
            .collect::<Vec<_>>();
        Self {
            p_moduli_bits,
            max_unreduced_muls,
            scale,
            p_moduli,
            q_moduli: active_q_moduli,
            q_moduli_depth,
            p_max: max_p_modulus,
            lut_mod_p_max_map_size,
            p_full,
            p_over_pis,
            gadget_values,
            full_reduce_max_plaintexts,
            lut_mod_p_ids,
            lut_x_to_y_ids,
            lut_x_to_real_ids,
            lut_real_to_v_id,
            add_without_reduce_id,
            sub_with_trace_offsets_id,
            lazy_reduce_id,
            decomposition_terms_id,
            gadget_decompose_id,
            full_reduce_id,
            full_reduce_bindings,
            mul_lazy_reduce_id,
            mul_right_sparse_id,
        }
    }

    pub(crate) fn register_subcircuits_in<P: Poly + 'static>(
        &self,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let p_moduli = self.p_moduli.clone();
        let p_moduli_depth = p_moduli.len();
        let q_moduli_depth = self.q_moduli_depth;
        let mut scalars_y = vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth];
        let mut scalars_v = vec![vec![0; p_moduli_depth]; q_moduli_depth];

        for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
            for (q_idx, &q_k) in self.q_moduli.iter().take(q_moduli_depth).enumerate() {
                for (p_j_idx, p_over_pj) in self.p_over_pis.iter().enumerate() {
                    let p_over_pj_mod_qk = (p_over_pj % BigUint::from(q_k))
                        .to_u64()
                        .expect("CRT residue must fit in u64");
                    let p_over_pj_mod_qk_mod_pi = p_over_pj_mod_qk % p_i;
                    scalars_y[q_idx][p_i_idx][p_j_idx] = p_over_pj_mod_qk_mod_pi as u32;
                }
                let p_mod_qk = (&self.p_full % BigUint::from(q_k))
                    .to_u64()
                    .expect("CRT residue must fit in u64");
                let p_mod_qk_mod_pi = p_mod_qk % p_i;
                scalars_v[q_idx][p_i_idx] = p_mod_qk_mod_pi as u32;
            }
        }

        let add_without_reduce_id =
            circuit.register_sub_circuit(Self::add_without_reduce_subcircuit::<P>(&p_moduli));
        let sub_with_trace_offsets_id = circuit
            .register_sub_circuit(Self::sub_with_trace_offsets_subcircuit::<P>(p_moduli_depth));
        let mul_lazy_reduce_id = circuit.register_sub_circuit(
            Self::mul_lazy_reduce_subcircuit::<P>(&p_moduli, &self.lut_mod_p_ids),
        );
        let mul_right_sparse_id = circuit.register_sub_circuit(
            Self::mul_right_sparse_subcircuit::<P>(&p_moduli, &self.lut_mod_p_ids),
        );
        let lazy_reduce_id = circuit.register_sub_circuit(Self::lazy_reduce_subcircuit::<P>(
            &p_moduli,
            &self.lut_mod_p_ids,
        ));
        let decomposition_terms_id =
            circuit.register_sub_circuit(Self::decomposition_terms_subcircuit::<P>(
                &self.lut_x_to_y_ids,
                &self.lut_x_to_real_ids,
                self.lut_real_to_v_id,
            ));
        let gadget_decompose_id =
            circuit.register_sub_circuit(Self::gadget_decompose_subcircuit::<P>(
                &p_moduli,
                &self.lut_mod_p_ids,
                &self.lut_x_to_y_ids,
                &self.lut_x_to_real_ids,
                self.lut_real_to_v_id,
            ));
        let full_reduce_id = circuit.register_sub_circuit(Self::full_reduce_subcircuit::<P>(
            &p_moduli,
            &self.lut_mod_p_ids,
            &self.lut_x_to_y_ids,
            &self.lut_x_to_real_ids,
            self.lut_real_to_v_id,
        ));
        let full_reduce_bindings = (0..q_moduli_depth)
            .into_par_iter()
            .map(|q_idx| full_reduce_param_bindings(&scalars_y[q_idx], &scalars_v[q_idx]))
            .collect::<Vec<_>>();
        Self {
            p_moduli_bits: self.p_moduli_bits,
            max_unreduced_muls: self.max_unreduced_muls,
            scale: self.scale,
            p_moduli,
            q_moduli: self.q_moduli.clone(),
            q_moduli_depth,
            p_max: self.p_max,
            lut_mod_p_max_map_size: self.lut_mod_p_max_map_size.clone(),
            p_full: self.p_full.clone(),
            p_over_pis: self.p_over_pis.clone(),
            gadget_values: self.gadget_values.clone(),
            full_reduce_max_plaintexts: self.full_reduce_max_plaintexts.clone(),
            lut_mod_p_ids: self.lut_mod_p_ids.clone(),
            lut_x_to_y_ids: self.lut_x_to_y_ids.clone(),
            lut_x_to_real_ids: self.lut_x_to_real_ids.clone(),
            lut_real_to_v_id: self.lut_real_to_v_id,
            add_without_reduce_id,
            sub_with_trace_offsets_id,
            lazy_reduce_id,
            decomposition_terms_id,
            gadget_decompose_id,
            full_reduce_id,
            full_reduce_bindings,
            mul_lazy_reduce_id,
            mul_right_sparse_id,
        }
    }

    pub(crate) fn register_shared_subcircuits_in<P: Poly + 'static>(
        &self,
        source_circuit: &PolyCircuit<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        circuit.inherit_registries_from_parent(source_circuit);
        let p_moduli = self.p_moduli.clone();
        let p_moduli_depth = p_moduli.len();
        let q_moduli_depth = self.q_moduli_depth;
        let mut scalars_y = vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth];
        let mut scalars_v = vec![vec![0; p_moduli_depth]; q_moduli_depth];

        for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
            for (q_idx, &q_k) in self.q_moduli.iter().take(q_moduli_depth).enumerate() {
                for (p_j_idx, p_over_pj) in self.p_over_pis.iter().enumerate() {
                    let p_over_pj_mod_qk = (p_over_pj % BigUint::from(q_k))
                        .to_u64()
                        .expect("CRT residue must fit in u64");
                    let p_over_pj_mod_qk_mod_pi = p_over_pj_mod_qk % p_i;
                    scalars_y[q_idx][p_i_idx][p_j_idx] = p_over_pj_mod_qk_mod_pi as u32;
                }
                let p_mod_qk = (&self.p_full % BigUint::from(q_k))
                    .to_u64()
                    .expect("CRT residue must fit in u64");
                let p_mod_qk_mod_pi = p_mod_qk % p_i;
                scalars_v[q_idx][p_i_idx] = p_mod_qk_mod_pi as u32;
            }
        }

        let add_without_reduce_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.add_without_reduce_id),
        );
        let sub_with_trace_offsets_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.sub_with_trace_offsets_id),
        );
        let mul_lazy_reduce_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.mul_lazy_reduce_id),
        );
        let mul_right_sparse_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.mul_right_sparse_id),
        );
        let lazy_reduce_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.lazy_reduce_id),
        );
        let decomposition_terms_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.decomposition_terms_id),
        );
        let gadget_decompose_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.gadget_decompose_id),
        );
        let full_reduce_id = circuit.register_shared_sub_circuit(
            source_circuit.registered_sub_circuit_ref(self.full_reduce_id),
        );
        let full_reduce_bindings = (0..q_moduli_depth)
            .into_par_iter()
            .map(|q_idx| full_reduce_param_bindings(&scalars_y[q_idx], &scalars_v[q_idx]))
            .collect::<Vec<_>>();

        Self {
            p_moduli_bits: self.p_moduli_bits,
            max_unreduced_muls: self.max_unreduced_muls,
            scale: self.scale,
            p_moduli,
            q_moduli: self.q_moduli.clone(),
            q_moduli_depth,
            p_max: self.p_max,
            lut_mod_p_max_map_size: self.lut_mod_p_max_map_size.clone(),
            p_full: self.p_full.clone(),
            p_over_pis: self.p_over_pis.clone(),
            gadget_values: self.gadget_values.clone(),
            full_reduce_max_plaintexts: self.full_reduce_max_plaintexts.clone(),
            lut_mod_p_ids: self.lut_mod_p_ids.clone(),
            lut_x_to_y_ids: self.lut_x_to_y_ids.clone(),
            lut_x_to_real_ids: self.lut_x_to_real_ids.clone(),
            lut_real_to_v_id: self.lut_real_to_v_id,
            add_without_reduce_id,
            sub_with_trace_offsets_id,
            lazy_reduce_id,
            decomposition_terms_id,
            gadget_decompose_id,
            full_reduce_id,
            full_reduce_bindings,
            mul_lazy_reduce_id,
            mul_right_sparse_id,
        }
    }

    pub(crate) fn reduce_q_level_row<P: Poly>(
        &self,
        row: &[GateId],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        assert_eq!(
            row.len(),
            self.p_moduli.len(),
            "q-level row depth {} must match p_moduli depth {}",
            row.len(),
            self.p_moduli.len()
        );
        circuit
            .call_sub_circuit(self.lazy_reduce_id, row.iter().copied())
            .into_iter()
            .map(BatchedWire::as_single_wire)
            .collect()
    }

    pub(crate) fn mul_q_level_rows<P: Poly>(
        &self,
        left: &[GateId],
        right: &[GateId],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        assert_eq!(
            left.len(),
            self.p_moduli.len(),
            "left q-level row depth {} must match p_moduli depth {}",
            left.len(),
            self.p_moduli.len()
        );
        assert_eq!(
            right.len(),
            self.p_moduli.len(),
            "right q-level row depth {} must match p_moduli depth {}",
            right.len(),
            self.p_moduli.len()
        );
        circuit
            .call_sub_circuit(
                self.mul_lazy_reduce_id,
                left.iter().copied().chain(right.iter().copied()),
            )
            .into_iter()
            .map(BatchedWire::as_single_wire)
            .collect()
    }

    fn mul_lazy_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let mul_circuit = Self::mul_without_reduce_subcircuit::<P>(p_moduli);
        let reduce_circuit = Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids);
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mul_circuit_id = circuit.register_sub_circuit(mul_circuit);
        let prod = circuit.call_sub_circuit(mul_circuit_id, inputs.gate_ids());
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &prod);
        circuit.output(reduced);
        circuit
    }

    fn mul_right_sparse_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        Self::mul_lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids)
    }

    fn add_without_reduce_subcircuit<P: Poly>(p_moduli: &[u64]) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mut result_p_moduli = Vec::with_capacity(p_moduli_depth);
        let left = inputs.slice(0..p_moduli_depth).to_vec();
        let right = inputs.slice(p_moduli_depth..inputs.len()).to_vec();
        for p_idx in 0..p_moduli_depth {
            let sum_gate = circuit.add_gate(left[p_idx], right[p_idx]);
            result_p_moduli.push(sum_gate);
        }
        circuit.output(result_p_moduli);
        circuit
    }

    fn mul_without_reduce_subcircuit<P: Poly>(p_moduli: &[u64]) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mut result_p_moduli = Vec::with_capacity(p_moduli_depth);
        let left = inputs.slice(0..p_moduli_depth).to_vec();
        let right = inputs.slice(p_moduli_depth..inputs.len()).to_vec();
        for p_idx in 0..p_moduli_depth {
            let mul_gate = circuit.mul_gate(left[p_idx], right[p_idx]);
            result_p_moduli.push(mul_gate);
        }
        circuit.output(result_p_moduli);
        circuit
    }

    fn lazy_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);

        let mut result_p_moduli = Vec::with_capacity(p_moduli_depth);
        for p_idx in 0..p_moduli_depth {
            let reduced_gate = circuit.public_lookup_gate(inputs.at(p_idx), lut_mod_p_ids[p_idx]);
            result_p_moduli.push(reduced_gate);
        }
        circuit.output(result_p_moduli);
        circuit
    }

    fn decomposition_terms_subcircuit<P: Poly>(
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> PolyCircuit<P> {
        assert_eq!(
            lut_x_to_y_ids.len(),
            lut_x_to_real_ids.len(),
            "decomposition-terms LUT depths must match"
        );
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = lut_x_to_y_ids.len();
        let inputs = circuit.input(p_moduli_depth);
        let mut outputs = Vec::with_capacity(p_moduli_depth + 1);
        let mut real_sum = circuit.const_zero_gate();
        for p_idx in 0..p_moduli_depth {
            let y_i = circuit.public_lookup_gate(inputs.at(p_idx), lut_x_to_y_ids[p_idx]);
            outputs.push(y_i);
            let real_i = circuit.public_lookup_gate(inputs.at(p_idx), lut_x_to_real_ids[p_idx]);
            real_sum = circuit.add_gate(real_sum, real_i);
        }
        outputs.push(circuit.public_lookup_gate(real_sum, lut_real_to_v_id));
        circuit.output(outputs);
        circuit
    }

    fn gadget_decompose_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);
        let decomposition_terms_id =
            circuit.register_sub_circuit(Self::decomposition_terms_subcircuit::<P>(
                lut_x_to_y_ids,
                lut_x_to_real_ids,
                lut_real_to_v_id,
            ));
        let decomposition_terms =
            circuit.call_sub_circuit(decomposition_terms_id, inputs.gate_ids());
        let ys = decomposition_terms[..p_moduli_depth].to_vec();
        let w = decomposition_terms[p_moduli_depth];
        let lazy_reduce_id = circuit
            .register_sub_circuit(Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids));

        let mut outputs = Vec::with_capacity((p_moduli_depth + 1) * p_moduli_depth);
        for y_i in ys {
            let repeated = vec![y_i; p_moduli_depth];
            outputs.extend(circuit.call_sub_circuit(lazy_reduce_id, &repeated));
        }
        outputs.extend(circuit.call_sub_circuit(lazy_reduce_id, &vec![w; p_moduli_depth]));
        circuit.output(outputs);
        circuit
    }

    fn full_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);
        let x = inputs;
        let scalar_y_param_ids = (0..p_moduli_depth)
            .map(|_| {
                (0..p_moduli_depth)
                    .map(|_| {
                        circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scalar_v_param_ids = (0..p_moduli_depth)
            .map(|_| circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul))
            .collect::<Vec<_>>();

        // 1. y_i = [[x]_{p_i} * (p/p_oi)^(-1} mod p_i] mod p_i
        let ys = (0..p_moduli_depth)
            .map(|p_idx| circuit.public_lookup_gate(x.at(p_idx), lut_x_to_y_ids[p_idx]))
            .collect::<Vec<_>>();
        // 2. real_i = round_div(y_i * scale, p_i), but the input is x_i rather than y_i
        let reals = (0..p_moduli_depth)
            .map(|p_idx| circuit.public_lookup_gate(x.at(p_idx), lut_x_to_real_ids[p_idx]))
            .collect::<Vec<_>>();
        // 3. sum_i real_i
        let mut real_sum = circuit.const_zero_gate();
        for &real in reals.iter() {
            real_sum = circuit.add_gate(real_sum, real);
        }
        // 4. v = round_div(real_sum, scale)
        let v = circuit.public_lookup_gate(real_sum, lut_real_to_v_id);
        // 5. p_i_sum = (sum_j (y_j * [[\hat{p_j}]_{q_k}]_{p_i}) mod p_i) - v * [[p]_{q_k}]_{p_i}
        //    Keep the running sum reduced to avoid negative ranges before the final mod.
        let mut p_i_sums = Vec::with_capacity(p_moduli_depth);
        for p_idx in 0..p_moduli_depth {
            let mut p_i_sum = circuit.const_zero_gate();
            for p_j_idx in 0..p_moduli_depth {
                let y_j = ys[p_j_idx];
                let term = circuit.small_scalar_mul_param(y_j, scalar_y_param_ids[p_idx][p_j_idx]);
                let term_mod_p = circuit.public_lookup_gate(term, lut_mod_p_ids[p_idx]);
                p_i_sum = circuit.add_gate(p_i_sum, term_mod_p);
            }
            let term = circuit.small_scalar_mul_param(v, scalar_v_param_ids[p_idx]);
            let p_i_const = circuit.const_digits(&[p_moduli.len() as u32 * p_moduli[p_idx] as u32]);
            let sum = circuit.add_gate(p_i_sum, p_i_const);
            let p_i_sum = circuit.sub_gate(sum, term);
            let p_i_sum_mod_p = circuit.public_lookup_gate(p_i_sum, lut_mod_p_ids[p_idx]);
            p_i_sums.push(p_i_sum_mod_p);
        }
        circuit.output(p_i_sums);
        circuit
    }

    fn sub_with_trace_offsets_subcircuit<P: Poly>(p_moduli_depth: usize) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let inputs = circuit.input(2 * p_moduli_depth);
        let (left, right) = inputs.split_at(p_moduli_depth);
        let one = circuit.const_one_gate();
        let offset_param_ids = (0..p_moduli_depth)
            .map(|_| circuit.register_sub_circuit_param(SubCircuitParamKind::LargeScalarMul))
            .collect::<Vec<_>>();
        let outputs = (0..p_moduli_depth)
            .map(|p_idx| {
                let offset_gate = circuit.large_scalar_mul_param(one, offset_param_ids[p_idx]);
                let shifted_left = circuit.add_gate(left.at(p_idx), offset_gate);
                circuit.sub_gate(shifted_left, right.at(p_idx))
            })
            .collect::<Vec<_>>();
        circuit.output(outputs);
        circuit
    }
}

#[derive(Debug, Clone)]
pub struct NestedRnsPoly<P: Poly> {
    pub ctx: Arc<NestedRnsPolyContext>,
    pub inner: Vec<Vec<GateId>>, // inner[q_moduli_idx][p_moduli_idx]
    pub level_offset: usize,
    pub enable_levels: Option<usize>,
    pub max_plaintexts: Vec<BigUint>,
    pub(crate) p_max_traces: Vec<BigUint>,
    _p: PhantomData<P>,
}

impl<P: Poly> NestedRnsPoly<P> {
    pub fn new(
        ctx: Arc<NestedRnsPolyContext>,
        inner: Vec<Vec<GateId>>,
        level_offset: Option<usize>,
        enable_levels: Option<usize>,
        max_plaintexts: Vec<BigUint>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        let p_max_traces = vec![ctx.reduced_p_max_trace(); inner.len()];
        let poly = Self {
            ctx,
            inner,
            level_offset,
            enable_levels,
            max_plaintexts,
            p_max_traces,
            _p: PhantomData,
        };
        poly.validate_enable_levels(poly.enable_levels);
        poly
    }

    pub(crate) fn with_p_max_traces(mut self, p_max_traces: Vec<BigUint>) -> Self {
        self.p_max_traces = p_max_traces;
        self.validate_enable_levels(self.enable_levels);
        self
    }

    pub fn input(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        let input_count = enable_levels.unwrap_or(ctx.q_moduli_depth);
        assert!(
            level_offset + input_count <= ctx.q_moduli_depth,
            "active range exceeds q_moduli_depth: level_offset={level_offset}, enable_levels={input_count}, q_moduli_depth={}",
            ctx.q_moduli_depth
        );
        let inner = (0..input_count).map(|_| circuit.input(ctx.p_moduli.len()).to_vec()).collect();
        let max_plaintexts = ctx
            .q_moduli
            .iter()
            .skip(level_offset)
            .take(input_count)
            .map(|&q_i| BigUint::from(q_i - 1))
            .collect();
        Self::new(ctx, inner, Some(level_offset), enable_levels, max_plaintexts)
    }

    pub(crate) fn input_with_metadata(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        let input_count = enable_levels.unwrap_or(ctx.q_moduli_depth);
        assert!(
            level_offset + input_count <= ctx.q_moduli_depth,
            "active range exceeds q_moduli_depth: level_offset={level_offset}, enable_levels={input_count}, q_moduli_depth={}",
            ctx.q_moduli_depth
        );
        assert_eq!(
            max_plaintexts.len(),
            input_count,
            "max_plaintexts length {} must match active levels {}",
            max_plaintexts.len(),
            input_count
        );
        assert_eq!(
            p_max_traces.len(),
            input_count,
            "p_max_traces length {} must match active levels {}",
            p_max_traces.len(),
            input_count
        );
        let inner = (0..input_count).map(|_| circuit.input(ctx.p_moduli.len()).to_vec()).collect();
        Self::new(ctx, inner, Some(level_offset), enable_levels, max_plaintexts)
            .with_p_max_traces(p_max_traces)
    }

    pub(crate) fn input_like_with_ctx(
        template: &Self,
        ctx: Arc<NestedRnsPolyContext>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::input_with_metadata(
            ctx,
            template.enable_levels,
            Some(template.level_offset),
            template.max_plaintexts.clone(),
            template.p_max_traces.clone(),
            circuit,
        )
    }

    fn lazy_reduce_selected_levels(
        &self,
        reduce_levels: &[bool],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        assert_eq!(
            reduce_levels.len(),
            levels,
            "lazy reduce mask length {} must match active levels {}",
            reduce_levels.len(),
            levels
        );
        if !reduce_levels.iter().any(|&flag| flag) {
            return self.clone();
        }

        let mut inner = self.inner.clone();
        let mut p_max_traces = self.p_max_traces.clone();
        let reduced_trace = self.ctx.reduced_p_max_trace();
        for q_idx in 0..levels {
            if reduce_levels[q_idx] {
                inner[q_idx] = single_wire_gate_ids(
                    circuit.call_sub_circuit(self.ctx.lazy_reduce_id, &self.inner[q_idx]),
                );
                p_max_traces[q_idx] = reduced_trace.clone();
            }
        }
        Self::new(
            self.ctx.clone(),
            inner,
            Some(self.level_offset),
            self.enable_levels,
            self.max_plaintexts.clone(),
        )
        .with_p_max_traces(p_max_traces)
    }

    fn lazy_reduce_if_unreduced(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let threshold = self.ctx.unreduced_trace_threshold();
        let reduce_levels = self.p_max_traces[..self.resolve_enable_levels()]
            .par_iter()
            .map(|trace| trace >= &threshold)
            .collect::<Vec<_>>();
        self.lazy_reduce_selected_levels(&reduce_levels, circuit)
    }

    fn reduced_p_max_traces(&self) -> Vec<BigUint> {
        vec![self.ctx.reduced_p_max_trace(); self.resolve_enable_levels()]
    }

    fn compute_add_output_p_max_traces(&self, other: &Self) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        self.p_max_traces[..levels]
            .par_iter()
            .zip(other.p_max_traces[..levels].par_iter())
            .map(|(left_trace, right_trace)| left_trace + right_trace)
            .collect()
    }

    fn trace_multiplier(&self, trace: &BigUint) -> BigUint {
        (trace + BigUint::from(self.ctx.p_max - 1)) / BigUint::from(self.ctx.p_max)
    }

    fn compute_sub_output_p_max_traces(&self, other: &Self) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        let p_max = BigUint::from(self.ctx.p_max);
        self.p_max_traces[..levels]
            .par_iter()
            .zip(other.p_max_traces[..levels].par_iter())
            .map(|(left_trace, right_trace)| {
                left_trace + self.trace_multiplier(right_trace) * &p_max
            })
            .collect()
    }

    fn assert_p_max_traces_within_lut_map_size(&self, traces: &[BigUint], message: &str) {
        assert!(
            traces.iter().all(|trace| trace < &self.ctx.lut_mod_p_max_map_size),
            "{}: p_max_traces={:?}, lut_mod_p_max_map_size={}",
            message,
            traces,
            self.ctx.lut_mod_p_max_map_size
        );
    }

    pub fn slot_transfer(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let mut operand = self.clone();
        let predicted_bounds = self.compute_slot_transfer_output_bounds(src_slots);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            operand = self.full_reduce(circuit);
        }
        operand = operand.lazy_reduce_if_unreduced(circuit);
        let final_bounds = operand.compute_slot_transfer_output_bounds(src_slots);
        operand.assert_bounds_within_p_full(
            &final_bounds,
            "slot_transfer output exceeds p_full even after automatic full_reduce",
        );

        let levels = operand.resolve_enable_levels();
        let mut inner = Vec::with_capacity(levels);
        for q_moduli_idx in 0..levels {
            let q_level = &operand.inner[q_moduli_idx];
            assert_eq!(
                q_level.len(),
                operand.ctx.p_moduli.len(),
                "mismatched p_moduli depth for q_moduli_idx {}",
                q_moduli_idx
            );
            let transferred = q_level
                .iter()
                .zip(operand.ctx.p_moduli.iter())
                .map(|(&gate_id, &p_j)| {
                    let lowered_src_slots = src_slots
                        .iter()
                        .enumerate()
                        .map(|(slot_idx, (src_slot, slot_scalars))| {
                            let scalar = slot_scalars.as_ref().map(|slot_scalars| {
                                let residue = *slot_scalars.get(q_moduli_idx).unwrap_or_else(|| {
                                    panic!(
                                        "slot {} scalar depth {} does not cover q_moduli_idx {}",
                                        slot_idx,
                                        slot_scalars.len(),
                                        q_moduli_idx
                                    )
                                });
                                u32::try_from(residue % p_j)
                                    .expect("slot-transfer scalar must fit in u32")
                            });
                            (*src_slot, scalar)
                        })
                        .collect::<Vec<_>>();
                    circuit.slot_transfer_gate(gate_id, &lowered_src_slots)
                })
                .collect::<Vec<_>>();
            inner.push(single_wire_gate_ids(
                circuit.call_sub_circuit(operand.ctx.lazy_reduce_id, &transferred),
            ));
        }
        Self::new(
            operand.ctx.clone(),
            inner,
            Some(operand.level_offset),
            operand.enable_levels,
            final_bounds,
        )
        .with_p_max_traces(operand.reduced_p_max_traces())
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_matching_enable_levels(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds =
            self.compute_binary_output_bounds(other, &|left, right, _| left + right);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        let predicted_traces = left.compute_add_output_p_max_traces(&right);
        let reduce_levels = predicted_traces
            .iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        left = left.lazy_reduce_selected_levels(&reduce_levels, circuit);
        right = right.lazy_reduce_selected_levels(&reduce_levels, circuit);

        let final_bounds =
            left.compute_binary_output_bounds(&right, &|left, right, _| left + right);
        left.assert_bounds_within_p_full(
            &final_bounds,
            "additive operation output exceeds p_full even after automatic full_reduce",
        );
        let final_traces = left.compute_add_output_p_max_traces(&right);
        left.assert_p_max_traces_within_lut_map_size(
            &final_traces,
            "additive operation output exceeds lut_mod_p_map_size even after pre-reduction",
        );
        left.call_uniform_binary_subcircuit(
            &right,
            circuit,
            self.ctx.add_without_reduce_id,
            final_bounds,
            final_traces,
        )
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_matching_enable_levels(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds = self.compute_binary_output_bounds(other, &|left, _right, q_i| {
            left + BigUint::from(q_i - 1)
        });
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        let predicted_traces = left.compute_sub_output_p_max_traces(&right);
        let reduce_levels = predicted_traces
            .iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        left = left.lazy_reduce_selected_levels(&reduce_levels, circuit);
        right = right.lazy_reduce_selected_levels(&reduce_levels, circuit);

        let final_bounds = left.compute_binary_output_bounds(&right, &|left, _right, q_i| {
            left + BigUint::from(q_i - 1)
        });
        left.assert_bounds_within_p_full(
            &final_bounds,
            "subtractive operation output exceeds p_full even after automatic full_reduce",
        );
        let final_traces = left.compute_sub_output_p_max_traces(&right);
        left.assert_p_max_traces_within_lut_map_size(
            &final_traces,
            "subtractive operation output exceeds lut_mod_p_map_size even after pre-reduction",
        );
        left.call_sub_with_trace_offsets(&right, circuit, final_bounds, final_traces)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let left = self.lazy_reduce_if_unreduced(circuit);
        let right = other.lazy_reduce_if_unreduced(circuit);
        left.apply_binary_operation(
            &right,
            circuit,
            self.ctx.mul_lazy_reduce_id,
            |left, right, _| left * right,
        )
    }

    pub fn mul_right_sparse(
        &self,
        other: &Self,
        right_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.assert_matching_enable_levels(other);
        let levels = self.resolve_enable_levels();
        other.assert_sparse_at_q_idx(right_q_idx);

        let mut left = self.clone();
        let mut right = other.clone();
        let mut predicted_bounds = vec![BigUint::ZERO; levels];
        predicted_bounds[right_q_idx] =
            &self.max_plaintexts[right_q_idx] * &other.max_plaintexts[right_q_idx];
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        left = left.lazy_reduce_if_unreduced(circuit);
        right = right.lazy_reduce_if_unreduced(circuit);

        let mut final_bounds = vec![BigUint::ZERO; levels];
        final_bounds[right_q_idx] =
            &left.max_plaintexts[right_q_idx] * &right.max_plaintexts[right_q_idx];
        left.assert_bounds_within_p_full(
            &final_bounds,
            "mul_right_sparse output exceeds p_full even after automatic full_reduce",
        );

        let mut final_traces = vec![BigUint::ZERO; levels];
        final_traces[right_q_idx] = left.ctx.reduced_p_max_trace();
        left.call_sparse_right_subcircuit(
            &right,
            right_q_idx,
            circuit,
            self.ctx.mul_right_sparse_id,
            final_bounds,
            final_traces,
        )
    }

    pub fn full_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let operand = self.lazy_reduce_if_unreduced(circuit);
        let levels = self.resolve_enable_levels();
        assert!(
            levels <= self.ctx.full_reduce_bindings.len(),
            "enable_levels exceeds full_reduce subcircuit depth"
        );
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let outputs = circuit.call_sub_circuit_with_bindings(
                self.ctx.full_reduce_id,
                &operand.inner[q_idx],
                &self.ctx.full_reduce_bindings[self.level_offset + q_idx],
            );
            result_inner.push(single_wire_gate_ids(outputs));
        }
        let max_plaintexts = (0..levels)
            .map(|local_idx| {
                self.ctx.full_reduce_max_plaintexts[self.level_offset + local_idx].clone()
            })
            .collect::<Vec<_>>();
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(operand.reduced_p_max_traces())
    }

    pub fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<P>) -> Self {
        let levels = self.resolve_enable_levels();
        assert_eq!(
            tower_constants.len(),
            levels,
            "tower constant depth {} must match active levels {}",
            tower_constants.len(),
            levels
        );
        let mut operand = self.clone();
        let predicted_bounds = self.compute_const_mul_output_bounds(tower_constants);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            operand = self.full_reduce(circuit);
        }
        operand = operand.lazy_reduce_if_unreduced(circuit);
        let final_bounds = operand.compute_const_mul_output_bounds(tower_constants);
        operand.assert_bounds_within_p_full(
            &final_bounds,
            "const_mul output exceeds p_full even after automatic full_reduce",
        );
        let mut result_inner = Vec::with_capacity(levels);
        for (q_idx, &tower_constant) in tower_constants.iter().enumerate() {
            let scaled = operand.inner[q_idx]
                .iter()
                .zip(self.ctx.p_moduli.iter())
                .map(|(&gate_id, &p_i)| {
                    let scalar_digits = u64_to_u32_digits(tower_constant % p_i);
                    circuit.small_scalar_mul(gate_id, &scalar_digits)
                })
                .collect::<Vec<_>>();
            result_inner.push(single_wire_gate_ids(
                circuit.call_sub_circuit(self.ctx.lazy_reduce_id, &scaled),
            ));
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            final_bounds,
        )
        .with_p_max_traces(operand.reduced_p_max_traces())
    }

    pub fn gadget_vector(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        let (level_offset, active_q_moduli) =
            resolve_nested_rns_active_window(ctx.as_ref(), enable_levels, level_offset);
        let active_levels = active_q_moduli.len();
        let chunk_width = ctx.p_moduli.len() + 1;
        let gadget_values = ctx.gadget_values[level_offset..level_offset + active_levels]
            .iter()
            .flat_map(|level_values| level_values.iter().cloned())
            .collect::<Vec<_>>();
        let mut constant_cache = BTreeMap::new();
        gadget_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                Self::sparse_constant_level_poly(
                    ctx.clone(),
                    active_levels,
                    enable_levels,
                    level_offset,
                    idx / chunk_width,
                    &value,
                    circuit,
                    &mut constant_cache,
                )
            })
            .collect()
    }

    pub fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        let operand = if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.clone()
        };
        operand.assert_p_max_traces_within_lut_map_size(
            &operand.p_max_traces[..operand.resolve_enable_levels()],
            "gadget_decompose input exceeds lut_mod_p_map_size",
        );
        let levels = operand.resolve_enable_levels();
        let p_moduli_depth = operand.ctx.p_moduli.len();
        let w_bound =
            BigUint::from(u64::try_from(p_moduli_depth).expect("p_moduli length must fit in u64"));
        let mut decomposition = Vec::with_capacity(levels * (p_moduli_depth + 1));

        for q_idx in 0..levels {
            let outputs =
                circuit.call_sub_circuit(operand.ctx.gadget_decompose_id, &operand.inner[q_idx]);
            assert_eq!(
                outputs.len(),
                (p_moduli_depth + 1) * p_moduli_depth,
                "gadget_decompose subcircuit output count must match ((|p| + 1) * |p|)"
            );
            for p_idx in 0..p_moduli_depth {
                let y_bound = BigUint::from(operand.ctx.p_moduli[p_idx] - 1);
                let start = p_idx * p_moduli_depth;
                let y_row = outputs[start..start + p_moduli_depth]
                    .iter()
                    .copied()
                    .map(BatchedWire::as_single_wire)
                    .collect();
                decomposition.push(operand.sparse_level_poly_from_row(
                    q_idx,
                    y_row,
                    y_bound.clone(),
                    y_bound,
                    circuit,
                ));
            }

            let w_start = p_moduli_depth * p_moduli_depth;
            let w_row = outputs[w_start..w_start + p_moduli_depth]
                .iter()
                .copied()
                .map(BatchedWire::as_single_wire)
                .collect();
            decomposition.push(operand.sparse_level_poly_from_row(
                q_idx,
                w_row,
                w_bound.clone(),
                w_bound.clone(),
                circuit,
            ));
        }

        decomposition
    }

    pub fn conv_mul_right_decomposed_many(
        &self,
        params: &P::Params,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self>
    where
        P: 'static,
    {
        if left_rows.is_empty() {
            return vec![];
        }

        assert!(num_slots > 0, "conv_mul_right_decomposed_many requires positive num_slots");
        assert!(
            num_slots <= params.ring_dimension() as usize,
            "num_slots {} exceeds ring dimension {}",
            num_slots,
            params.ring_dimension()
        );

        let levels = self.resolve_enable_levels();
        let p_moduli_depth = self.ctx.p_moduli.len();
        let chunk_width = p_moduli_depth + 1;
        let gadget_len = levels * chunk_width;
        for (row_idx, row) in left_rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                gadget_len,
                "left row {} length {} must match gadget_len {}",
                row_idx,
                row.len(),
                gadget_len
            );
            for (entry_idx, entry) in row.iter().enumerate() {
                entry.assert_matching_enable_levels(self);
                assert!(
                    Arc::ptr_eq(&entry.ctx, &self.ctx),
                    "conv_mul_right_decomposed_many requires left row {} entry {} to share the NestedRnsPolyContext with right",
                    row_idx,
                    entry_idx
                );
            }
        }

        let right = self.prepare_for_decomposed_conv(circuit);
        let prepared_left_rows = left_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| entry.prepare_for_decomposed_conv(circuit))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let row_count = prepared_left_rows.len();

        let term_subcircuit = negacyclic_conv_mul_right_decomposed_term_many_subcircuit::<P>(
            right.ctx.as_ref(),
            row_count,
            num_slots,
        );
        let term_subcircuit_id = circuit.register_sub_circuit(term_subcircuit);

        let flat_term_output_templates =
            map_nested_rns_values(row_count * gadget_len, |flat_idx| {
                let row_idx = flat_idx / gadget_len;
                let global_idx = flat_idx % gadget_len;
                let left = &prepared_left_rows[row_idx][global_idx];
                Self::conv_mul_right_decomposed_output_template(
                    params,
                    left,
                    global_idx / chunk_width,
                    global_idx % chunk_width,
                    num_slots,
                )
            });
        let term_output_templates = flat_term_output_templates
            .chunks(gadget_len)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();

        let mut row_terms = vec![Vec::with_capacity(gadget_len); row_count];
        for q_idx in 0..levels {
            let (ys, w) = right.decomposition_terms_for_level(q_idx, circuit);
            for term_idx in 0..chunk_width {
                let global_idx = q_idx * chunk_width + term_idx;
                let term_gate = if term_idx < p_moduli_depth { ys[term_idx] } else { w };
                let term_row = vec![term_gate; p_moduli_depth];
                let mut inputs = Vec::with_capacity((row_count + 1) * p_moduli_depth);
                for row in &prepared_left_rows {
                    inputs.extend_from_slice(&row[global_idx].inner[q_idx]);
                }
                inputs.extend_from_slice(&term_row);
                let outputs = circuit.call_sub_circuit(term_subcircuit_id, &inputs);
                assert_eq!(
                    outputs.len(),
                    row_count * p_moduli_depth,
                    "conv_mul_right_decomposed_many term helper output size must match row_count * p_moduli_depth"
                );
                for row_idx in 0..row_count {
                    let start = row_idx * p_moduli_depth;
                    let output_template = &term_output_templates[row_idx][global_idx];
                    row_terms[row_idx].push(Self::sparse_level_poly_from_row_with_metadata(
                        self.ctx.clone(),
                        levels,
                        self.enable_levels,
                        self.level_offset,
                        q_idx,
                        outputs[start..start + p_moduli_depth]
                            .iter()
                            .copied()
                            .map(BatchedWire::as_single_wire)
                            .collect(),
                        output_template.max_plaintexts[q_idx].clone(),
                        output_template.p_max_traces[q_idx].clone(),
                        circuit,
                    ));
                }
            }
        }

        row_terms
            .into_iter()
            .map(|mut terms| {
                let mut acc = terms
                    .pop()
                    .expect("conv_mul_right_decomposed_many requires at least one gadget term");
                for term in terms {
                    acc = acc.add(&term, circuit);
                }
                acc
            })
            .collect()
    }

    pub fn conv_mul_right_decomposed(
        &self,
        params: &P::Params,
        left_row: &[Self],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self
    where
        P: 'static,
    {
        self.conv_mul_right_decomposed_many(params, &[left_row], num_slots, circuit)
            .into_iter()
            .next()
            .expect("conv_mul_right_decomposed must produce one output row")
    }

    fn prepare_for_decomposed_conv(&self, circuit: &mut PolyCircuit<P>) -> Self {
        if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.assert_p_max_traces_within_lut_map_size(
                &self.p_max_traces[..self.resolve_enable_levels()],
                "decomposed convolution input exceeds lut_mod_p_map_size",
            );
            self.clone()
        }
    }

    fn sparse_decomposed_term_input_template(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        term_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let active_levels = enable_levels.unwrap_or(ctx.q_moduli_depth - level_offset);
        let target_row = circuit.input(ctx.p_moduli.len()).to_vec();
        let (max_plaintext, p_max_trace) = if term_idx < ctx.p_moduli.len() {
            let bound = BigUint::from(ctx.p_moduli[term_idx] - 1);
            (bound.clone(), bound)
        } else {
            let bound =
                BigUint::from(u64::try_from(ctx.p_moduli.len()).expect("p_moduli length fits u64"));
            (bound.clone(), bound)
        };
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            active_levels,
            enable_levels,
            level_offset,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn conv_mul_right_decomposed_output_template(
        params: &P::Params,
        left: &Self,
        target_q_idx: usize,
        term_idx: usize,
        num_slots: usize,
    ) -> Self
    where
        P: 'static,
    {
        let mut template_circuit = PolyCircuit::<P>::new();
        let template_ctx = Arc::new(left.ctx.register_subcircuits_in(&mut template_circuit));
        let lhs = Self::input_like_with_ctx(left, template_ctx.clone(), &mut template_circuit);
        let rhs = Self::sparse_decomposed_term_input_template(
            template_ctx,
            lhs.enable_levels,
            lhs.level_offset,
            target_q_idx,
            term_idx,
            &mut template_circuit,
        );
        negacyclic_conv_mul_right_sparse(
            params,
            &mut template_circuit,
            &lhs,
            &rhs,
            target_q_idx,
            num_slots,
        )
    }

    pub(crate) fn prepare_for_reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Self {
        if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.lazy_reduce_if_unreduced(circuit)
        }
    }

    pub fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let operand = self.prepare_for_reconstruct(circuit);
        let levels = operand.resolve_enable_levels();
        let mut sum_mod_q = circuit.const_zero_gate();
        let active_moduli = operand.active_q_moduli();
        let active_modulus =
            active_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
        for q_idx in 0..levels {
            let q_i_big = BigUint::from(active_moduli[q_idx]);
            let q_over_qi = &active_modulus / &q_i_big;
            let q_over_qi_mod = &q_over_qi % &q_i_big;
            let inv = mod_inverse(
                q_over_qi_mod.to_u64().expect("CRT residue must fit in u64"),
                active_moduli[q_idx],
            )
            .expect("CRT modulus must be invertible within the active range");
            let reconst_coeff = (&q_over_qi * BigUint::from(inv)) % &active_modulus;
            let mut sum_without_reduce = circuit.const_zero_gate();
            let (ys, w) = operand.decomposition_terms_for_level(q_idx, circuit);
            for (p_idx, y_i) in ys.into_iter().enumerate() {
                let y_i_p_j_hat =
                    circuit.large_scalar_mul(y_i, &[operand.ctx.p_over_pis[p_idx].clone()]);
                sum_without_reduce = circuit.add_gate(sum_without_reduce, y_i_p_j_hat);
            }
            let pv = circuit.large_scalar_mul(w, &[operand.ctx.p_full.clone()]);
            let sum_q_k = circuit.sub_gate(sum_without_reduce, pv);
            let sum_q_k_scaled = circuit.large_scalar_mul(sum_q_k, &[reconst_coeff]);
            sum_mod_q = circuit.add_gate(sum_mod_q, sum_q_k_scaled);
        }
        sum_mod_q.as_single_wire()
    }

    pub fn benchmark_multiplication_tree(
        ctx: Arc<NestedRnsPolyContext>,
        circuit: &mut PolyCircuit<P>,
        height: usize,
        enable_levels: Option<usize>,
    ) {
        let num_inputs =
            1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");
        let mut current_layer: Vec<NestedRnsPoly<P>> = (0..num_inputs)
            .map(|_| NestedRnsPoly::input(ctx.clone(), enable_levels, None, circuit))
            .collect();
        while current_layer.len() > 1 {
            debug_assert!(current_layer.len().is_multiple_of(2), "layer size must stay even");
            let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
            for pair in current_layer.chunks(2) {
                let parent = pair[0].mul(&pair[1], circuit);
                next_layer.push(parent);
            }
            current_layer = next_layer;
        }
        let root = current_layer.pop().expect("multiplication tree must contain at least one node");
        let out = root.reconstruct(circuit);
        circuit.output(vec![out]);
    }

    fn call_binary_subcircuit(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        assert!(
            levels <= other.inner.len(),
            "operand q_moduli depth {} does not cover active levels {}",
            other.inner.len(),
            levels
        );
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let left = &self.inner[q_idx];
            let right = &other.inner[q_idx];
            assert_eq!(left.len(), right.len(), "mismatched p_moduli depth");
            let mut inputs = Vec::with_capacity(left.len() + right.len());
            inputs.extend_from_slice(left);
            inputs.extend_from_slice(right);
            let outputs = circuit.call_sub_circuit(subcircuit_id, &inputs);
            result_inner.push(single_wire_gate_ids(outputs));
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn call_sparse_right_subcircuit(
        &self,
        other: &Self,
        target_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        assert!(
            target_q_idx < levels,
            "sparse target q_idx {} exceeds active levels {}",
            target_q_idx,
            levels
        );
        assert!(
            levels <= other.inner.len(),
            "operand q_moduli depth {} does not cover active levels {}",
            other.inner.len(),
            levels
        );
        let zero_gate = circuit.const_zero_gate().as_single_wire();
        let p_moduli_depth = self.ctx.p_moduli.len();
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            if q_idx == target_q_idx {
                let left = &self.inner[q_idx];
                let right = &other.inner[q_idx];
                assert_eq!(left.len(), right.len(), "mismatched p_moduli depth");
                let mut inputs = Vec::with_capacity(left.len() + right.len());
                inputs.extend_from_slice(left);
                inputs.extend_from_slice(right);
                let outputs = circuit.call_sub_circuit(subcircuit_id, &inputs);
                result_inner.push(single_wire_gate_ids(outputs));
            } else {
                result_inner.push(vec![zero_gate; p_moduli_depth]);
            }
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn call_uniform_binary_subcircuit(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        assert!(
            levels <= other.inner.len(),
            "operand q_moduli depth {} does not cover active levels {}",
            other.inner.len(),
            levels
        );
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let left = &self.inner[q_idx];
            let right = &other.inner[q_idx];
            assert_eq!(left.len(), right.len(), "mismatched p_moduli depth");
            let mut inputs = Vec::with_capacity(left.len() + right.len());
            inputs.extend_from_slice(left);
            inputs.extend_from_slice(right);
            let outputs = circuit.call_sub_circuit(subcircuit_id, &inputs);
            result_inner.push(single_wire_gate_ids(outputs));
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn call_sub_with_trace_offsets(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        assert!(
            levels <= other.inner.len(),
            "operand q_moduli depth {} does not cover active levels {}",
            other.inner.len(),
            levels
        );
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let left = &self.inner[q_idx];
            let right = &other.inner[q_idx];
            assert_eq!(left.len(), right.len(), "mismatched p_moduli depth");
            let offset_multiplier = self.trace_multiplier(&other.p_max_traces[q_idx]);
            let bindings =
                sub_with_trace_offset_param_bindings(&offset_multiplier, &self.ctx.p_moduli);
            let mut inputs = Vec::with_capacity(left.len() + right.len());
            inputs.extend_from_slice(left);
            inputs.extend_from_slice(right);
            let outputs = circuit.call_sub_circuit_with_bindings(
                self.ctx.sub_with_trace_offsets_id,
                &inputs,
                &bindings,
            );
            result_inner.push(single_wire_gate_ids(outputs));
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn apply_binary_operation<FB>(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        output_bound: FB,
    ) -> Self
    where
        FB: Fn(&BigUint, &BigUint, u64) -> BigUint,
    {
        self.assert_matching_enable_levels(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds = self.compute_binary_output_bounds(other, &output_bound);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }
        let final_bounds = left.compute_binary_output_bounds(&right, &output_bound);
        left.assert_bounds_within_p_full(
            &final_bounds,
            "binary operation output exceeds p_full even after automatic full_reduce",
        );
        left.call_binary_subcircuit(
            &right,
            circuit,
            subcircuit_id,
            final_bounds,
            left.reduced_p_max_traces(),
        )
    }

    pub(crate) fn decomposition_terms_for_level(
        &self,
        q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId) {
        let levels = self.resolve_enable_levels();
        assert!(q_idx < levels, "q_idx {} exceeds active levels {}", q_idx, levels);
        let outputs = circuit.call_sub_circuit(self.ctx.decomposition_terms_id, &self.inner[q_idx]);
        assert_eq!(
            outputs.len(),
            self.ctx.p_moduli.len() + 1,
            "decomposition_terms subcircuit output size must match |p| + 1"
        );
        let p_moduli_depth = self.ctx.p_moduli.len();
        (
            outputs[..p_moduli_depth].iter().copied().map(BatchedWire::as_single_wire).collect(),
            outputs[p_moduli_depth].as_single_wire(),
        )
    }

    fn sparse_constant_level_poly(
        ctx: Arc<NestedRnsPolyContext>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        value: &BigUint,
        circuit: &mut PolyCircuit<P>,
        constant_cache: &mut BTreeMap<BigUint, GateId>,
    ) -> Self {
        let p_moduli = ctx.p_moduli.clone();
        let max_plaintext = value.clone();
        let value_for_residues = value.clone();
        let residues = map_nested_rns_values(p_moduli.len(), move |idx| {
            &value_for_residues % BigUint::from(p_moduli[idx])
        });
        let p_max_trace = residues.iter().cloned().max().unwrap_or(BigUint::ZERO);
        let row = residues
            .into_iter()
            .map(|residue| const_biguint_gate_cached(circuit, &residue, constant_cache))
            .collect::<Vec<_>>();
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            active_levels,
            enable_levels,
            level_offset,
            target_q_idx,
            row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn sparse_level_poly_from_row_with_metadata(
        ctx: Arc<NestedRnsPolyContext>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        target_row: Vec<GateId>,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(
            target_q_idx < active_levels,
            "target_q_idx {} exceeds active levels {}",
            target_q_idx,
            active_levels
        );
        assert_eq!(
            target_row.len(),
            ctx.p_moduli.len(),
            "target row depth {} must match p_moduli depth {}",
            target_row.len(),
            ctx.p_moduli.len()
        );
        assert!(
            p_max_trace < ctx.lut_mod_p_max_map_size,
            "sparse row trace {} must stay below lut_mod_p_max_map_size {}",
            p_max_trace,
            ctx.lut_mod_p_max_map_size
        );

        let zero_gate = circuit.const_zero_gate();
        let mut inner = Vec::with_capacity(active_levels);
        let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
        let mut p_max_traces = vec![BigUint::ZERO; active_levels];
        let mut target_row = Some(target_row);
        max_plaintexts[target_q_idx] = max_plaintext;
        p_max_traces[target_q_idx] = p_max_trace;

        for q_idx in 0..active_levels {
            if q_idx == target_q_idx {
                inner.push(target_row.take().expect("target row must be present exactly once"));
            } else {
                inner.push(vec![zero_gate.as_single_wire(); ctx.p_moduli.len()]);
            }
        }

        Self::new(ctx, inner, Some(level_offset), enable_levels, max_plaintexts)
            .with_p_max_traces(p_max_traces)
    }

    fn sparse_level_poly_from_row(
        &self,
        target_q_idx: usize,
        target_row: Vec<GateId>,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::sparse_level_poly_from_row_with_metadata(
            self.ctx.clone(),
            self.resolve_enable_levels(),
            self.enable_levels,
            self.level_offset,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn compute_binary_output_bounds<F>(&self, other: &Self, output_bound: &F) -> Vec<BigUint>
    where
        F: Fn(&BigUint, &BigUint, u64) -> BigUint,
    {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                output_bound(
                    &self.max_plaintexts[q_idx],
                    &other.max_plaintexts[q_idx],
                    self.ctx.q_moduli[self.level_offset + q_idx],
                )
            })
            .collect()
    }

    fn compute_const_mul_output_bounds(&self, tower_constants: &[u64]) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                &self.max_plaintexts[q_idx] *
                    BigUint::from(
                        tower_constants[q_idx] % self.ctx.q_moduli[self.level_offset + q_idx],
                    )
            })
            .collect()
    }

    fn compute_slot_transfer_output_bounds(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
    ) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        let tower_scales = self.compute_slot_transfer_tower_scales(src_slots);
        (0..levels).map(|q_idx| &self.max_plaintexts[q_idx] * &tower_scales[q_idx]).collect()
    }

    fn compute_slot_transfer_tower_scales(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
    ) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                src_slots
                    .iter()
                    .map(|(_src_slot, slot_scalars)| {
                        let scalar = slot_scalars.as_ref().map_or(1u64, |slot_scalars| {
                            let residue = *slot_scalars.get(q_idx).unwrap_or_else(|| {
                                panic!(
                                    "slot scalar depth {} does not cover q_moduli_idx {}",
                                    slot_scalars.len(),
                                    q_idx
                                )
                            });
                            residue % self.ctx.q_moduli[self.level_offset + q_idx]
                        });
                        BigUint::from(scalar)
                    })
                    .max()
                    .unwrap_or(BigUint::ZERO)
            })
            .collect()
    }

    fn bounds_exceed_p_full(&self, bounds: &[BigUint]) -> bool {
        bounds.iter().any(|bound| bound >= &self.ctx.p_full)
    }

    fn assert_bounds_within_p_full(&self, bounds: &[BigUint], message: &str) {
        assert!(
            !self.bounds_exceed_p_full(bounds),
            "{}: max_plaintexts={:?}, p_full={}",
            message,
            bounds,
            self.ctx.p_full
        );
    }

    fn assert_matching_enable_levels(&self, other: &Self) {
        assert_eq!(
            self.enable_levels, other.enable_levels,
            "mismatched enable_levels: left={:?}, right={:?}",
            self.enable_levels, other.enable_levels
        );
        assert_eq!(
            self.level_offset, other.level_offset,
            "mismatched level_offset: left={}, right={}",
            self.level_offset, other.level_offset
        );
    }

    fn assert_sparse_at_q_idx(&self, target_q_idx: usize) {
        let levels = self.resolve_enable_levels();
        assert!(
            target_q_idx < levels,
            "mul_right_sparse target q_idx {} exceeds active levels {}",
            target_q_idx,
            levels
        );
        for q_idx in 0..levels {
            let should_be_zero = q_idx != target_q_idx;
            if should_be_zero {
                assert!(
                    self.max_plaintexts[q_idx] == BigUint::ZERO,
                    "mul_right_sparse requires the right operand to be zero outside q_idx {}",
                    target_q_idx
                );
            }
        }
        assert!(
            self.max_plaintexts[target_q_idx] != BigUint::ZERO,
            "mul_right_sparse requires a non-zero bound at q_idx {}",
            target_q_idx
        );
    }

    fn resolve_enable_levels(&self) -> usize {
        let max_levels = self.inner.len();
        match self.enable_levels {
            Some(levels) => {
                assert!(levels <= max_levels, "enable_levels exceeds available levels");
                levels
            }
            None => max_levels,
        }
    }

    fn validate_enable_levels(&self, enable_levels: Option<usize>) {
        if let Some(levels) = enable_levels {
            assert!(levels <= self.inner.len(), "enable_levels exceeds available levels");
            assert!(
                self.level_offset + levels <= self.ctx.q_moduli_depth,
                "active range exceeds available q-moduli: level_offset={}, enable_levels={}, q_moduli_depth={}",
                self.level_offset,
                levels,
                self.ctx.q_moduli_depth
            );
        }
        assert_eq!(
            self.inner.len(),
            self.max_plaintexts.len(),
            "max_plaintexts length {} must match inner q_moduli depth {}",
            self.max_plaintexts.len(),
            self.inner.len()
        );
        assert_eq!(
            self.inner.len(),
            self.p_max_traces.len(),
            "p_max_traces length {} must match inner q_moduli depth {}",
            self.p_max_traces.len(),
            self.inner.len()
        );
        assert!(
            self.level_offset <= self.ctx.q_moduli_depth,
            "level_offset {} exceeds q_moduli_depth {}",
            self.level_offset,
            self.ctx.q_moduli_depth
        );
    }

    pub fn active_q_moduli(&self) -> Vec<u64> {
        let levels = self.resolve_enable_levels();
        self.ctx.q_moduli.iter().skip(self.level_offset).take(levels).copied().collect()
    }
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn const_biguint_gate<P: Poly>(circuit: &mut PolyCircuit<P>, value: &BigUint) -> GateId {
    if let Some(value_u32) = value.to_u32() {
        circuit.const_digits(&[value_u32]).as_single_wire()
    } else {
        let one = circuit.const_one_gate();
        circuit.large_scalar_mul(one, std::slice::from_ref(value)).as_single_wire()
    }
}

fn const_biguint_gate_cached<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    value: &BigUint,
    cache: &mut BTreeMap<BigUint, GateId>,
) -> GateId {
    if let Some(&gate) = cache.get(value) {
        return gate;
    }

    let gate = if value == &BigUint::ZERO {
        circuit.const_zero_gate().as_single_wire()
    } else {
        const_biguint_gate(circuit, value)
    };
    cache.insert(value.clone(), gate);
    gate
}

fn u64_to_u32_digits(mut value: u64) -> Vec<u32> {
    if value == 0 {
        return vec![0];
    }
    let mut digits = Vec::new();
    while value > 0 {
        digits.push((value & u32::MAX as u64) as u32);
        value >>= 32;
    }
    digits
}

/// Return the first `count` pairwise coprime integers within the requested `bit_width`.
///
/// Deterministic: the output depends only on `bit_width` and `count` (no randomness).
pub(crate) fn sample_crt_primes(
    max_bit_width: usize,
    q_max: u64,
    max_unreduced_muls: usize,
) -> Vec<u64> {
    assert!(max_bit_width > 1, "bit_width must be at least 2 bits");
    assert!(max_bit_width < 32, "bit_width must be less than 32 bits");
    assert!(
        max_bit_width <= usize::BITS as usize,
        "bit_width {max_bit_width} exceeds target pointer width {}",
        usize::BITS
    );
    assert!(max_unreduced_muls > 0, "max_unreduced_muls must be at least 1");
    // assert!(count > 0, "count must be greater than 0");

    let lower = 3u64;
    let upper = 1u64 << max_bit_width;
    let mut results: Vec<u64> = Vec::new();
    let mut sum = 0u64;
    let mut prod = BigUint::one();
    let mut prod_reached = false;

    // Prefer larger moduli (bigger `p = ∏ p_i` for the same depth), but keep selection
    // deterministic.
    for candidate in lower..upper {
        if results.iter().all(|&chosen| gcd_u64(candidate, chosen) == 1) {
            results.push(candidate);
            sum += candidate;
            prod *= BigUint::from(candidate);
        }
        let mul_budget_bound = sample_crt_primes_mul_budget_bound(sum, results.len(), q_max);
        if pow_biguint_usize(&mul_budget_bound, max_unreduced_muls) < prod {
            prod_reached = true;
            break;
        }
    }

    if !prod_reached {
        panic!(
            "failed to find enough pairwise coprime integers with bit width {max_bit_width} to \
             satisfy q_max {q_max} and max_unreduced_muls {max_unreduced_muls}; try increasing bit width"
        );
    }

    results
}

fn resolve_nested_rns_encoding_layout<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    q_level: Option<usize>,
) -> (Vec<u64>, usize, Vec<u64>) {
    let (q_moduli, _, q_moduli_depth) = params.to_crt();
    let active_q_level = q_level.unwrap_or(q_moduli_depth);
    assert!(
        active_q_level <= q_moduli_depth,
        "q_level exceeds q_moduli_depth: q_level={}, q_moduli_depth={}",
        active_q_level,
        q_moduli_depth
    );
    let q_moduli_max = q_moduli.iter().max().expect("there should be at least one q modulus");
    let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max, max_unreduced_muls);
    (q_moduli, active_q_level, p_moduli)
}

fn map_nested_rns_outputs_with_params<P, T, F>(
    params: &P::Params,
    output_count: usize,
    f: F,
) -> Vec<T>
where
    P: Poly,
    T: Send,
    F: Fn(usize, &P::Params) -> T + Send + Sync,
{
    if output_count == 0 {
        return Vec::new();
    }

    #[cfg(feature = "gpu")]
    {
        return gpu::map_nested_rns_outputs_with_params_gpu::<P, T, F>(params, output_count, f);
    }

    #[cfg(not(feature = "gpu"))]
    {
        (0..output_count).into_par_iter().map(|idx| f(idx, params)).collect()
    }
}

fn map_nested_rns_values<T, F>(count: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if count == 0 {
        return Vec::new();
    }

    #[cfg(feature = "gpu")]
    {
        return (0..count).map(f).collect();
    }

    #[cfg(not(feature = "gpu"))]
    {
        (0..count).into_par_iter().map(f).collect()
    }
}

fn resolve_nested_rns_active_window(
    ctx: &NestedRnsPolyContext,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> (usize, Vec<u64>) {
    let level_offset = level_offset.unwrap_or(0);
    let active_levels = enable_levels.unwrap_or_else(|| {
        ctx.q_moduli_depth
            .checked_sub(level_offset)
            .expect("level_offset must not exceed q_moduli_depth")
    });
    assert!(
        level_offset + active_levels <= ctx.q_moduli_depth,
        "active q range exceeds q_moduli_depth: level_offset={level_offset}, enable_levels={active_levels}, q_moduli_depth={}",
        ctx.q_moduli_depth
    );
    let active_q_moduli =
        ctx.q_moduli.iter().skip(level_offset).take(active_levels).copied().collect::<Vec<_>>();
    (level_offset, active_q_moduli)
}

fn nested_rns_level_reconstruction_coeffs(active_q_moduli: &[u64]) -> Vec<BigUint> {
    let active_modulus =
        active_q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    active_q_moduli
        .iter()
        .map(|&q_i| {
            let q_i_big = BigUint::from(q_i);
            let q_over_qi = &active_modulus / &q_i_big;
            let q_over_qi_mod =
                (&q_over_qi % &q_i_big).to_u64().expect("CRT residue must fit in u64");
            let inv = mod_inverse(q_over_qi_mod, q_i).expect("CRT moduli must be coprime");
            (&q_over_qi * BigUint::from(inv)) % &active_modulus
        })
        .collect::<Vec<_>>()
}

fn nested_rns_sub_mod(lhs: BigUint, rhs: &BigUint, modulus: &BigUint) -> BigUint {
    let rhs = rhs % modulus;
    if lhs >= rhs { lhs - rhs } else { lhs + modulus - rhs }
}

fn nested_rns_decomposition_terms_from_row(
    ctx: &NestedRnsPolyContext,
    row: &[u64],
) -> (Vec<BigUint>, BigUint) {
    assert_eq!(row.len(), ctx.p_moduli.len(), "row depth must match p_moduli depth");
    let mut ys = Vec::with_capacity(ctx.p_moduli.len());
    let mut real_sum = 0u64;
    for (p_idx, &p_i) in ctx.p_moduli.iter().enumerate() {
        let x_i = row[p_idx] % p_i;
        let p_over_pi_mod_pi = (&ctx.p_over_pis[p_idx] % BigUint::from(p_i))
            .to_u64()
            .expect("CRT residue must fit in u64");
        let p_over_pi_inv = mod_inverse(p_over_pi_mod_pi, p_i).expect("CRT moduli must be coprime");
        let y_i = ((x_i as u128 * p_over_pi_inv as u128) % p_i as u128) as u64;
        real_sum += round_div(y_i * ctx.scale, p_i);
        ys.push(BigUint::from(y_i));
    }
    let w = BigUint::from(round_div(real_sum, ctx.scale));
    (ys, w)
}

fn nested_rns_sparse_level_slot_value<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    reconst_coeff: &BigUint,
    row: &[u64],
) -> BigUint {
    let modulus: std::sync::Arc<BigUint> = params.modulus().into();
    let modulus = modulus.as_ref();
    let (ys, w) = nested_rns_decomposition_terms_from_row(ctx, row);
    let sum_without_reduce = ys
        .iter()
        .zip(ctx.p_over_pis.iter())
        .fold(BigUint::ZERO, |acc, (y_i, p_hat_i)| (acc + y_i * p_hat_i) % modulus);
    let pv = (&w * &ctx.p_full) % modulus;
    let sum_q_k = nested_rns_sub_mod(sum_without_reduce, &pv, modulus);
    (sum_q_k * reconst_coeff) % modulus
}

pub fn nested_rns_gadget_vector<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> M
where
    P: Poly,
    M: PolyMatrix<P = P>,
{
    let (level_offset, active_q_moduli) =
        resolve_nested_rns_active_window(ctx, enable_levels, level_offset);
    let chunk_width = ctx.p_moduli.len() + 1;
    let reconst_coeffs = nested_rns_level_reconstruction_coeffs(&active_q_moduli);
    let coeff_count = params.ring_dimension() as usize;
    let mut gadget_row = Vec::with_capacity(active_q_moduli.len() * chunk_width);
    for (q_idx, level_values) in
        ctx.gadget_values[level_offset..level_offset + active_q_moduli.len()].iter().enumerate()
    {
        for residue in level_values {
            let row = ctx
                .p_moduli
                .iter()
                .map(|&p_i| (residue % BigUint::from(p_i)).to_u64().expect("row residue must fit"))
                .collect::<Vec<_>>();
            let coeff =
                nested_rns_sparse_level_slot_value::<P>(params, ctx, &reconst_coeffs[q_idx], &row);
            gadget_row.push(P::from_biguints(params, &vec![coeff; coeff_count]));
        }
    }
    M::from_poly_vec_row(params, gadget_row)
}

pub fn nested_rns_gadget_decomposed<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    value: &M,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> M
where
    P: Poly,
    M: PolyMatrix<P = P>,
{
    let (_, active_q_moduli) = resolve_nested_rns_active_window(ctx, enable_levels, level_offset);
    let chunk_width = ctx.p_moduli.len() + 1;
    let gadget_len = active_q_moduli.len() * chunk_width;
    let (row_size, col_size) = value.size();
    let reconst_coeffs = nested_rns_level_reconstruction_coeffs(&active_q_moduli);

    let entry_polys = map_nested_rns_values(row_size * col_size, |entry_idx| {
        let row_idx = entry_idx / col_size;
        let col_idx = entry_idx % col_size;
        let coeffs = value.entry(row_idx, col_idx).coeffs_biguints();
        let mut output_coeffs = vec![vec![BigUint::ZERO; coeffs.len()]; gadget_len];
        for (q_idx, &q_i) in active_q_moduli.iter().enumerate() {
            let q_i_big = BigUint::from(q_i);
            for (coeff_idx, coeff) in coeffs.iter().enumerate() {
                let input_residue =
                    (coeff % &q_i_big).to_u64().expect("q-level residue must fit in u64");
                let input_row =
                    ctx.p_moduli.iter().map(|&p_i| input_residue % p_i).collect::<Vec<_>>();
                let (ys, w) = nested_rns_decomposition_terms_from_row(ctx, &input_row);
                for digit_idx in 0..chunk_width {
                    let scalar = if digit_idx < ctx.p_moduli.len() {
                        ys[digit_idx].to_u64().expect("decomposition digit must fit in u64")
                    } else {
                        w.to_u64().expect("rounding digit must fit in u64")
                    };
                    let encoded_row =
                        ctx.p_moduli.iter().map(|&p_i| scalar % p_i).collect::<Vec<_>>();
                    let target_row = q_idx * chunk_width + digit_idx;
                    output_coeffs[target_row][coeff_idx] = nested_rns_sparse_level_slot_value::<P>(
                        params,
                        ctx,
                        &reconst_coeffs[q_idx],
                        &encoded_row,
                    );
                }
            }
        }
        let output_polys = output_coeffs
            .into_iter()
            .map(|coeffs| P::from_biguints(params, &coeffs))
            .collect::<Vec<_>>();
        (row_idx, col_idx, output_polys)
    });

    let mut decomposed = (0..row_size * gadget_len)
        .map(|_| vec![P::const_zero(params); col_size])
        .collect::<Vec<_>>();
    for (row_idx, col_idx, output_polys) in entry_polys {
        for (target_row, poly) in output_polys.into_iter().enumerate() {
            decomposed[row_idx * gadget_len + target_row][col_idx] = poly;
        }
    }

    M::from_poly_vec(params, decomposed)
}

pub fn encode_nested_rns_poly_compact_bytes<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level: Option<usize>,
) -> Vec<Vec<u8>> {
    encode_nested_rns_poly_compact_bytes_with_offset::<P>(
        p_moduli_bits,
        max_unreduced_muls,
        params,
        input,
        0,
        q_level,
    )
}

pub fn encode_nested_rns_poly_compact_bytes_with_offset<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level_offset: usize,
    q_level: Option<usize>,
) -> Vec<Vec<u8>> {
    let (q_moduli, active_q_level, p_moduli) =
        resolve_nested_rns_encoding_layout::<P>(p_moduli_bits, max_unreduced_muls, params, q_level);
    assert!(
        q_level_offset + active_q_level <= q_moduli.len(),
        "active q range exceeds modulus depth: q_level_offset={}, active_q_level={}, q_moduli_depth={}",
        q_level_offset,
        active_q_level,
        q_moduli.len()
    );
    let input_mod_q = q_moduli
        .iter()
        .skip(q_level_offset)
        .take(active_q_level)
        .map(|&q_i| input % BigUint::from(q_i))
        .collect::<Vec<_>>();
    let p_moduli_depth = p_moduli.len();
    let output_count = active_q_level * p_moduli_depth;

    map_nested_rns_outputs_with_params::<P, _, _>(params, output_count, |idx, local_params| {
        let q_idx = idx / p_moduli_depth;
        let p_idx = idx % p_moduli_depth;
        let residue = &input_mod_q[q_idx] % p_moduli[p_idx];
        P::from_biguint_to_constant(local_params, residue).to_compact_bytes()
    })
}

pub fn encode_nested_rns_poly<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level: Option<usize>,
) -> Vec<P> {
    encode_nested_rns_poly_with_offset::<P>(
        p_moduli_bits,
        max_unreduced_muls,
        params,
        input,
        0,
        q_level,
    )
}

pub fn encode_nested_rns_poly_with_offset<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level_offset: usize,
    q_level: Option<usize>,
) -> Vec<P> {
    let (q_moduli, active_q_level, p_moduli) =
        resolve_nested_rns_encoding_layout::<P>(p_moduli_bits, max_unreduced_muls, params, q_level);
    assert!(
        q_level_offset + active_q_level <= q_moduli.len(),
        "active q range exceeds modulus depth: q_level_offset={}, active_q_level={}, q_moduli_depth={}",
        q_level_offset,
        active_q_level,
        q_moduli.len()
    );
    let p_moduli_depth = p_moduli.len();
    let mut polys = vec![Vec::with_capacity(p_moduli_depth); active_q_level];
    for (q_idx, &q_i) in q_moduli.iter().skip(q_level_offset).take(active_q_level).enumerate() {
        let input_qi = input % BigUint::from(q_i);
        for &p_i in p_moduli.iter() {
            polys[q_idx].push(P::from_biguint_to_constant(params, &input_qi % p_i));
        }
    }
    polys.into_iter().flatten().collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
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
        utils::ceil_biguint_nth_root,
    };

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
        let mut constant_cache = BTreeMap::new();
        NestedRnsPoly::<DCRTPoly>::sparse_constant_level_poly(
            ctx,
            active_levels,
            None,
            0,
            target_q_idx,
            &value,
            circuit,
            &mut constant_cache,
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
        let output_gates =
            gadget.iter().map(|entry| entry.reconstruct(circuit)).collect::<Vec<_>>();
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
                    .map(|_| {
                        NestedRnsPoly::input(ctx.clone(), enable_levels, level_offset, circuit)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let gadget_len =
            NestedRnsPoly::<DCRTPoly>::gadget_vector(ctx, enable_levels, level_offset, circuit)
                .len();
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
        circuit.eval(
            params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            None,
        )
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
        let input_values = vec![
            BigUint::ZERO,
            BigUint::from(7u64),
            modulus.as_ref().clone() - BigUint::from(5u64),
        ];
        let one = DCRTPoly::const_one(&params);
        let plt_evaluator = PolyPltEvaluator::new();
        for lhs_value in input_values {
            let lhs_inputs = encode_nested_rns_poly(
                P_MODULI_BITS,
                MAX_UNREDUCED_MULS,
                &params,
                &lhs_value,
                None,
            );
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
                let generic_rhs =
                    sparse_gadget_entry(generic_ctx, right_q_idx, &mut generic_circuit);
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
    #[should_panic(
        expected = "mul_right_sparse requires the right operand to be zero outside q_idx"
    )]
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
                let terms = row
                    .iter()
                    .zip(decomposed.iter())
                    .enumerate()
                    .map(|(entry_idx, (left, decomp))| {
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
                    .map(|_| {
                        NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let fused_right =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
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
        let manual_right =
            NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
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
                    .map(|_| {
                        NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let fused_right =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
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
        let manual_right =
            NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
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
        let fused_right =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
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
        let manual_left =
            NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
        let manual_right =
            NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
        let manual_unreduced = manual_left.add(&manual_right, &mut manual_circuit);
        let manual_reduced = manual_unreduced.lazy_reduce_if_unreduced(&mut manual_circuit);
        let manual_outputs = manual_reduced
            .gadget_decompose(&mut manual_circuit)
            .into_iter()
            .map(|term| term.reconstruct(&mut manual_circuit))
            .collect::<Vec<_>>();
        manual_circuit.output(manual_outputs);

        let left_coeffs = vec![
            BigUint::from(3u64),
            BigUint::from(5u64),
            BigUint::from(7u64),
            BigUint::from(11u64),
        ];
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
        let fused_right =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
        let fused_unreduced = fused_left.add(&fused_right, &mut fused_circuit);
        let fused_outputs = fused_unreduced
            .gadget_decompose(&mut fused_circuit)
            .into_iter()
            .map(|term| term.reconstruct(&mut fused_circuit))
            .collect::<Vec<_>>();
        fused_circuit.output(fused_outputs);

        let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, manual_ctx) = create_test_context_with_q_level(&mut manual_circuit, q_level);
        let manual_left =
            NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
        let manual_right =
            NestedRnsPoly::input(manual_ctx.clone(), q_level, None, &mut manual_circuit);
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
    fn test_nested_rns_poly_conv_mul_right_decomposed_many_unreduced_matches_explicit_lazy_reduce()
    {
        let q_level = Some(1usize);
        let num_slots = 4usize;

        let mut fused_circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, fused_ctx) = create_test_context_with_q_level(&mut fused_circuit, q_level);
        let chunk_width = fused_ctx.p_moduli.len() + 1;
        let gadget_len = fused_ctx.q_moduli_depth * chunk_width;
        let fused_left_row = (0..gadget_len)
            .map(|_| {
                let left0 =
                    NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
                let left1 =
                    NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
                left0.add(&left1, &mut fused_circuit)
            })
            .collect::<Vec<_>>();
        let fused_right0 =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
        let fused_right1 =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
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
                let left0 =
                    NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
                let left1 =
                    NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
                left0.add(&left1, &mut fused_circuit)
            })
            .collect::<Vec<_>>();
        let fused_right0 =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
        let fused_right1 =
            NestedRnsPoly::input(fused_ctx.clone(), q_level, None, &mut fused_circuit);
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
        let q_level_modulus = q_moduli
            .iter()
            .take(q_level)
            .fold(BigUint::from(1u64), |acc, &qi| acc * BigUint::from(qi));

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
            let encoded_input = encode_nested_rns_poly(
                P_MODULI_BITS,
                MAX_UNREDUCED_MULS,
                &params,
                &input_value,
                None,
            );
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
        let q1 =
            BigUint::from(ctx.q_moduli[level_offset.expect("test uses a fixed level offset") + 1]);
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
        let manual_out =
            manual_input.full_reduce(&mut manual_circuit).reconstruct(&mut manual_circuit);
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
        assert_eq!(
            auto_circuit.count_gates_by_type_vec(),
            manual_circuit.count_gates_by_type_vec()
        );
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
        auto_circuit.output(auto_transferred.inner[0].clone());

        let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, manual_ctx) = create_test_context_with_q_level(&mut manual_circuit, Some(q_level));
        let mut manual_input =
            NestedRnsPoly::input(manual_ctx.clone(), Some(q_level), None, &mut manual_circuit);
        manual_input.max_plaintexts = vec![operand_bound];
        let manual_transferred = manual_input.full_reduce(&mut manual_circuit).slot_transfer(
            &[(0, Some(vec![1])), (1, Some(vec![2])), (2, None)],
            &mut manual_circuit,
        );
        assert_eq!(manual_transferred.max_plaintexts, vec![reduced_transfer_bound]);
        manual_circuit.output(manual_transferred.inner[0].clone());

        assert_eq!(auto_circuit.non_free_depth(), manual_circuit.non_free_depth());
        assert_eq!(
            auto_circuit.count_gates_by_type_vec(),
            manual_circuit.count_gates_by_type_vec()
        );
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
                let pre_last =
                    &bound * BigUint::from(u64::try_from(candidate_count - 1).unwrap_or(0));
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
                let input = NestedRnsPoly::input(
                    manual_ctx.clone(),
                    Some(q_level),
                    None,
                    &mut manual_circuit,
                );
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
        let eval_inputs =
            (0..operand_count).flat_map(|_| encoded_input.clone()).collect::<Vec<_>>();
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
        assert_eq!(
            auto_circuit.count_gates_by_type_vec(),
            manual_circuit.count_gates_by_type_vec()
        );
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
                let input = NestedRnsPoly::input(
                    manual_ctx.clone(),
                    Some(q_level),
                    None,
                    &mut manual_circuit,
                );
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
        let eval_inputs =
            (0..operand_count).flat_map(|_| encoded_input.clone()).collect::<Vec<_>>();
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
        assert_eq!(
            auto_circuit.count_gates_by_type_vec(),
            manual_circuit.count_gates_by_type_vec()
        );
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
        let eval_results =
            circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
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
        let eval_results =
            circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
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
        let eval_results =
            circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
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
        let eval_results =
            circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
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
        let a_inputs = encode_nested_rns_poly(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &a_value,
            enable_levels,
        );
        let b_inputs = encode_nested_rns_poly(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &b_value,
            enable_levels,
        );
        let plt_evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let eval_inputs = [a_inputs, b_inputs].concat();
        let eval_results =
            circuit.eval(&params, one, eval_inputs, Some(&plt_evaluator), None, None);
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
}
