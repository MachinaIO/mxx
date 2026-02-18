use crate::{
    circuit::{PolyCircuit, gate::GateId},
    lookup::PublicLut,
    poly::{Poly, PolyParams},
    utils::{mod_inverse, round_div},
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use std::{marker::PhantomData, sync::Arc};
use tracing::debug;

#[derive(Debug, Clone)]
pub struct NestedRnsPolyContext {
    pub p_moduli: Vec<u64>,
    pub q_moduli_depth: usize,
    add_lazy_reduce_ids: Vec<usize>,
    sub_lazy_reduce_ids: Vec<usize>,
    mul_lazy_reduce_ids: Vec<usize>,
    add_full_reduce_ids: Vec<usize>,
    sub_full_reduce_ids: Vec<usize>,
    mul_full_reduce_ids: Vec<usize>,
    reconstruct_ids: Vec<usize>,
}

fn dummy_lut<P: Poly + 'static>(params: &P::Params) -> PublicLut<P> {
    PublicLut::new_from_usize_range(
        params,
        1,
        |params, _| (0, P::from_usize_to_constant(params, 0)),
        None,
    )
}

fn max_output_row_from_biguint<P: Poly>(
    params: &P::Params,
    idx: usize,
    value: BigUint,
) -> (usize, P::Elem) {
    let poly = P::from_biguint_to_constant(params, value);
    let coeff =
        poly.coeffs().into_iter().max().expect("max_output_row requires at least one coefficient");
    (idx, coeff)
}

impl NestedRnsPolyContext {
    pub fn setup<P: Poly + 'static>(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        p_moduli_bits: usize,
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
        let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max);
        debug!(
            "NestedRnsPolyContext setup: p_moduli = {:?}, q_moduli = {:?}, scale = {}",
            p_moduli, q_moduli, scale
        );
        let p_moduli_depth = p_moduli.len();
        if dummy_scalar {
            let dummy_lut = dummy_lut::<P>(params);
            let dummy_lut_id = circuit.register_public_lookup(dummy_lut);
            let lut_mod_p_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_y_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_real_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_real_to_v_id = dummy_lut_id;
            let p = p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
            let p_over_pis =
                p_moduli.iter().map(|&p_i| &p / BigUint::from(p_i)).collect::<Vec<_>>();
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
                    let p_mod_qk =
                        (&p % BigUint::from(*q_k)).to_u64().expect("CRT residue must fit in u64");
                    let p_mod_qk_mod_pi = p_mod_qk % p_i;
                    scalars_v[q_idx][p_i_idx] = p_mod_qk_mod_pi as u32;
                }
            }
            let add_lazy_reduce_ids = (0..q_moduli_depth)
                .map(|_| {
                    circuit.register_sub_circuit(Self::add_lazy_reduce_subcircuit::<P>(
                        &p_moduli,
                        &lut_mod_p_ids,
                    ))
                })
                .collect::<Vec<_>>();
            let sub_lazy_reduce_ids = (0..q_moduli_depth)
                .map(|_| {
                    circuit.register_sub_circuit(Self::sub_lazy_reduce_subcircuit::<P>(
                        &p_moduli,
                        &lut_mod_p_ids,
                    ))
                })
                .collect::<Vec<_>>();
            let mul_lazy_reduce_ids = (0..q_moduli_depth)
                .map(|_| {
                    circuit.register_sub_circuit(Self::mul_lazy_reduce_subcircuit::<P>(
                        &p_moduli,
                        &lut_mod_p_ids,
                    ))
                })
                .collect::<Vec<_>>();
            let add_full_reduce_ids = (0..q_moduli_depth)
                .map(|q_idx| {
                    circuit.register_sub_circuit(Self::add_full_reduce_subcircuit::<P>(
                        &p_moduli,
                        &lut_mod_p_ids,
                        &lut_x_to_y_ids,
                        &lut_x_to_real_ids,
                        lut_real_to_v_id,
                        &scalars_y[q_idx],
                        &scalars_v[q_idx],
                    ))
                })
                .collect::<Vec<_>>();
            let sub_full_reduce_ids = (0..q_moduli_depth)
                .map(|q_idx| {
                    circuit.register_sub_circuit(Self::sub_full_reduce_subcircuit::<P>(
                        &p_moduli,
                        &lut_mod_p_ids,
                        &lut_x_to_y_ids,
                        &lut_x_to_real_ids,
                        lut_real_to_v_id,
                        &scalars_y[q_idx],
                        &scalars_v[q_idx],
                    ))
                })
                .collect::<Vec<_>>();
            let mul_full_reduce_ids = (0..q_moduli_depth)
                .map(|q_idx| {
                    circuit.register_sub_circuit(Self::mul_full_reduce_subcircuit::<P>(
                        &p_moduli,
                        &lut_mod_p_ids,
                        &lut_x_to_y_ids,
                        &lut_x_to_real_ids,
                        lut_real_to_v_id,
                        &scalars_y[q_idx],
                        &scalars_v[q_idx],
                    ))
                })
                .collect::<Vec<_>>();
            let reconstruct_ids = (0..q_moduli_depth)
                .map(|q_idx| {
                    let (_, reconst_coeff) = params.to_crt_coeffs(q_idx);
                    circuit.register_sub_circuit(Self::reconstruct_subcircuit::<P>(
                        &p_moduli,
                        &lut_x_to_y_ids,
                        &lut_x_to_real_ids,
                        lut_real_to_v_id,
                        &p_over_pis,
                        &p,
                        &reconst_coeff,
                    ))
                })
                .collect::<Vec<_>>();
            return Self {
                p_moduli,
                q_moduli_depth,
                add_lazy_reduce_ids,
                sub_lazy_reduce_ids,
                mul_lazy_reduce_ids,
                add_full_reduce_ids,
                sub_full_reduce_ids,
                mul_full_reduce_ids,
                reconstruct_ids,
            };
        }

        let p = p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        let p_over_pis = p_moduli.iter().map(|&p_i| &p / BigUint::from(p_i)).collect::<Vec<_>>();
        let max_p_modulus = *p_moduli.iter().max().expect("p_moduli must not be empty");

        let mut lut_mod_p = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_y = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_real = Vec::with_capacity(p_moduli_depth);
        let mut scalars_y = vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth];
        let mut scalars_v = vec![vec![0; p_moduli_depth]; q_moduli_depth];

        for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
            let lut_mod_p_map_size = (p_i as u128 * max_p_modulus as u128)
                .max(p_i as u128 * (2 * p_moduli.len()) as u128);
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
            let lut_mod_p_lut = PublicLut::<P>::new_from_usize_range(
                params,
                lut_mod_p_len,
                move |params, t| {
                    let output = BigUint::from((t as u64) % p_i);
                    (t, P::from_biguint_to_constant(params, output))
                },
                Some(max_mod_p_row),
            );
            debug!("Constructed lut_mod_p for p_{} = {} with size {}", p_i_idx, p_i, lut_mod_p_len);
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
            let lut_x_to_y_len = p_i as usize;
            let lut_x_to_y_lut = PublicLut::<P>::new_from_usize_range(
                params,
                lut_x_to_y_len,
                {
                    let p_over_pi_inv = Arc::clone(&p_over_pi_inv);
                    let p_moduli_big = Arc::clone(&p_moduli_big);
                    move |params, t| {
                        let input = BigUint::from(t as u64);
                        let output = (&input * p_over_pi_inv.as_ref()) % p_moduli_big.as_ref();
                        (t, P::from_biguint_to_constant(params, output))
                    }
                },
                Some(max_x_to_y_row),
            );
            debug!(
                "Constructed lut_x_to_y for p_{} = {} with size {}",
                p_i_idx, p_i, lut_x_to_y_len
            );
            lut_x_to_y.push(lut_x_to_y_lut);

            let lut_x_to_real_len = p_i as usize;
            let lut_x_to_real_lut = PublicLut::<P>::new_from_usize_range(
                params,
                lut_x_to_real_len,
                {
                    let p_over_pi_inv = Arc::clone(&p_over_pi_inv);
                    let p_moduli_big = Arc::clone(&p_moduli_big);
                    move |params, t| {
                        let input = BigUint::from(t as u64);
                        let y = ((&input * p_over_pi_inv.as_ref()) % p_moduli_big.as_ref())
                            .to_u64()
                            .expect("y must fit in u64");
                        let output = BigUint::from(round_div(y * scale, p_i));
                        (t, P::from_biguint_to_constant(params, output))
                    }
                },
                Some(max_x_to_real_row),
            );
            debug!(
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
                    (&p % BigUint::from(*q_k)).to_u64().expect("CRT residue must fit in u64");
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
        let lut_real_to_v_lut = PublicLut::<P>::new_from_usize_range(
            params,
            lut_real_to_v_len,
            move |params, t| {
                let output = BigUint::from(round_div(t as u64, scale));
                (t, P::from_biguint_to_constant(params, output))
            },
            Some(max_real_to_v_row),
        );
        debug!("Constructed lut_real_to_v with size {}", lut_real_to_v_len);
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

        let add_lazy_reduce_ids = (0..q_moduli_depth)
            .map(|_| {
                circuit.register_sub_circuit(Self::add_lazy_reduce_subcircuit::<P>(
                    &p_moduli,
                    &lut_mod_p_ids,
                ))
            })
            .collect::<Vec<_>>();
        let sub_lazy_reduce_ids = (0..q_moduli_depth)
            .map(|_| {
                circuit.register_sub_circuit(Self::sub_lazy_reduce_subcircuit::<P>(
                    &p_moduli,
                    &lut_mod_p_ids,
                ))
            })
            .collect::<Vec<_>>();
        let mul_lazy_reduce_ids = (0..q_moduli_depth)
            .map(|_| {
                circuit.register_sub_circuit(Self::mul_lazy_reduce_subcircuit::<P>(
                    &p_moduli,
                    &lut_mod_p_ids,
                ))
            })
            .collect::<Vec<_>>();
        let add_full_reduce_ids = (0..q_moduli_depth)
            .map(|q_idx| {
                circuit.register_sub_circuit(Self::add_full_reduce_subcircuit::<P>(
                    &p_moduli,
                    &lut_mod_p_ids,
                    &lut_x_to_y_ids,
                    &lut_x_to_real_ids,
                    lut_real_to_v_id,
                    &scalars_y[q_idx],
                    &scalars_v[q_idx],
                ))
            })
            .collect::<Vec<_>>();
        let sub_full_reduce_ids = (0..q_moduli_depth)
            .map(|q_idx| {
                circuit.register_sub_circuit(Self::sub_full_reduce_subcircuit::<P>(
                    &p_moduli,
                    &lut_mod_p_ids,
                    &lut_x_to_y_ids,
                    &lut_x_to_real_ids,
                    lut_real_to_v_id,
                    &scalars_y[q_idx],
                    &scalars_v[q_idx],
                ))
            })
            .collect::<Vec<_>>();
        let mul_full_reduce_ids = (0..q_moduli_depth)
            .map(|q_idx| {
                circuit.register_sub_circuit(Self::mul_full_reduce_subcircuit::<P>(
                    &p_moduli,
                    &lut_mod_p_ids,
                    &lut_x_to_y_ids,
                    &lut_x_to_real_ids,
                    lut_real_to_v_id,
                    &scalars_y[q_idx],
                    &scalars_v[q_idx],
                ))
            })
            .collect::<Vec<_>>();
        let reconstruct_ids = (0..q_moduli_depth)
            .map(|q_idx| {
                let (_, reconst_coeff) = params.to_crt_coeffs(q_idx);
                circuit.register_sub_circuit(Self::reconstruct_subcircuit::<P>(
                    &p_moduli,
                    &lut_x_to_y_ids,
                    &lut_x_to_real_ids,
                    lut_real_to_v_id,
                    &p_over_pis,
                    &p,
                    &reconst_coeff,
                ))
            })
            .collect::<Vec<_>>();

        Self {
            p_moduli,
            q_moduli_depth,
            add_lazy_reduce_ids,
            sub_lazy_reduce_ids,
            mul_lazy_reduce_ids,
            add_full_reduce_ids,
            sub_full_reduce_ids,
            mul_full_reduce_ids,
            reconstruct_ids,
        }
    }

    fn add_lazy_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let add_circuit = Self::add_without_reduce_subcircuit::<P>(p_moduli);
        let reduce_circuit = Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids);
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let add_circuit_id = circuit.register_sub_circuit(add_circuit);
        let sum = circuit.call_sub_circuit(add_circuit_id, &inputs);
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &sum);
        circuit.output(reduced);
        circuit
    }

    fn sub_lazy_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let sub_circuit = Self::sub_without_reduce_subcircuit::<P>(p_moduli);
        let reduce_circuit = Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids);
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
        let diff = circuit.call_sub_circuit(sub_circuit_id, &inputs);
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &diff);
        circuit.output(reduced);
        circuit
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
        let prod = circuit.call_sub_circuit(mul_circuit_id, &inputs);
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &prod);
        circuit.output(reduced);
        circuit
    }

    fn add_full_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
        scalars_y: &[Vec<u32>],
        scalars_v: &[u32],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let add_circuit = Self::add_lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids);
        let reduce_circuit = Self::full_reduce_subcircuit::<P>(
            p_moduli,
            lut_mod_p_ids,
            lut_x_to_y_ids,
            lut_x_to_real_ids,
            lut_real_to_v_id,
            scalars_y,
            scalars_v,
        );
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let add_circuit_id = circuit.register_sub_circuit(add_circuit);
        let sum = circuit.call_sub_circuit(add_circuit_id, &inputs);
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &sum);
        circuit.output(reduced);
        circuit
    }

    fn sub_full_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
        scalars_y: &[Vec<u32>],
        scalars_v: &[u32],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let sub_circuit = Self::sub_lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids);
        let reduce_circuit = Self::full_reduce_subcircuit::<P>(
            p_moduli,
            lut_mod_p_ids,
            lut_x_to_y_ids,
            lut_x_to_real_ids,
            lut_real_to_v_id,
            scalars_y,
            scalars_v,
        );
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
        let diff = circuit.call_sub_circuit(sub_circuit_id, &inputs);
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &diff);
        circuit.output(reduced);
        circuit
    }

    fn mul_full_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
        scalars_y: &[Vec<u32>],
        scalars_v: &[u32],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let mul_circuit = Self::mul_lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids);
        let reduce_circuit = Self::full_reduce_subcircuit::<P>(
            p_moduli,
            lut_mod_p_ids,
            lut_x_to_y_ids,
            lut_x_to_real_ids,
            lut_real_to_v_id,
            scalars_y,
            scalars_v,
        );
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mul_circuit_id = circuit.register_sub_circuit(mul_circuit);
        let prod = circuit.call_sub_circuit(mul_circuit_id, &inputs);
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &prod);
        circuit.output(reduced);
        circuit
    }

    fn add_without_reduce_subcircuit<P: Poly>(p_moduli: &[u64]) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mut result_p_moduli = Vec::with_capacity(p_moduli_depth);
        let left = inputs[..p_moduli_depth].to_vec();
        let right = inputs[p_moduli_depth..].to_vec();
        for p_idx in 0..p_moduli_depth {
            let sum_gate = circuit.add_gate(left[p_idx], right[p_idx]);
            result_p_moduli.push(sum_gate);
        }
        circuit.output(result_p_moduli);
        circuit
    }

    fn sub_without_reduce_subcircuit<P: Poly>(p_moduli: &[u64]) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mut result_p_moduli = Vec::with_capacity(p_moduli_depth);
        let left = inputs[..p_moduli_depth].to_vec();
        let right = inputs[p_moduli_depth..].to_vec();
        for p_idx in 0..p_moduli_depth {
            let p_i = circuit.const_digits(&[p_moduli[p_idx] as u32]);
            let sum_gate = circuit.add_gate(left[p_idx], p_i);
            let sub_gate = circuit.sub_gate(sum_gate, right[p_idx]);
            result_p_moduli.push(sub_gate);
        }
        circuit.output(result_p_moduli);
        circuit
    }

    fn mul_without_reduce_subcircuit<P: Poly>(p_moduli: &[u64]) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mut result_p_moduli = Vec::with_capacity(p_moduli_depth);
        let left = inputs[..p_moduli_depth].to_vec();
        let right = inputs[p_moduli_depth..].to_vec();
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
            let reduced_gate = circuit.public_lookup_gate(inputs[p_idx], lut_mod_p_ids[p_idx]);
            result_p_moduli.push(reduced_gate);
        }
        circuit.output(result_p_moduli);
        circuit
    }

    fn full_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
        scalars_y: &[Vec<u32>],
        scalars_v: &[u32],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);
        let x = &inputs[0..p_moduli_depth];

        // 1. y_i = [[x]_{p_i} * (p/p_oi)^(-1} mod p_i] mod p_i
        let ys = (0..p_moduli_depth)
            .map(|p_idx| circuit.public_lookup_gate(x[p_idx], lut_x_to_y_ids[p_idx]))
            .collect::<Vec<_>>();
        // 2. real_i = round_div(y_i * scale, p_i), but the input is x_i rather than y_i
        let reals = (0..p_moduli_depth)
            .map(|p_idx| circuit.public_lookup_gate(x[p_idx], lut_x_to_real_ids[p_idx]))
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
                let term = circuit.small_scalar_mul(y_j, &[scalars_y[p_idx][p_j_idx]]);
                let term_mod_p = circuit.public_lookup_gate(term, lut_mod_p_ids[p_idx]);
                p_i_sum = circuit.add_gate(p_i_sum, term_mod_p);
            }
            let term = circuit.small_scalar_mul(v, &[scalars_v[p_idx]]);
            let p_i_const = circuit.const_digits(&[p_moduli.len() as u32 * p_moduli[p_idx] as u32]);
            let sum = circuit.add_gate(p_i_sum, p_i_const);
            let p_i_sum = circuit.sub_gate(sum, term);
            let p_i_sum_mod_p = circuit.public_lookup_gate(p_i_sum, lut_mod_p_ids[p_idx]);
            p_i_sums.push(p_i_sum_mod_p);
        }
        circuit.output(p_i_sums);
        circuit
    }

    fn reconstruct_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
        p_over_pis: &[BigUint],
        p: &BigUint,
        reconst_coeff: &BigUint,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);

        let mut sum_without_reduce = circuit.const_zero_gate();
        let mut real_sum = circuit.const_zero_gate();
        for p_idx in 0..p_moduli_depth {
            let y_i = circuit.public_lookup_gate(inputs[p_idx], lut_x_to_y_ids[p_idx]);
            let y_i_p_j_hat = circuit.large_scalar_mul(y_i, &[p_over_pis[p_idx].clone()]);
            sum_without_reduce = circuit.add_gate(sum_without_reduce, y_i_p_j_hat);
            let real_i = circuit.public_lookup_gate(inputs[p_idx], lut_x_to_real_ids[p_idx]);
            real_sum = circuit.add_gate(real_sum, real_i);
        }
        let v = circuit.public_lookup_gate(real_sum, lut_real_to_v_id);
        let pv = circuit.large_scalar_mul(v, &[p.clone()]);
        let sum_q_k = circuit.sub_gate(sum_without_reduce, pv);
        let sum_q_k_scaled = circuit.large_scalar_mul(sum_q_k, &[reconst_coeff.clone()]);
        circuit.output(vec![sum_q_k_scaled]);
        circuit
    }
}

#[derive(Debug, Clone)]
pub struct NestedRnsPoly<P: Poly> {
    pub ctx: Arc<NestedRnsPolyContext>,
    pub inner: Vec<Vec<GateId>>, // inner[q_moduli_idx][p_moduli_idx]
    _p: PhantomData<P>,
}

impl<P: Poly> NestedRnsPoly<P> {
    pub fn new(ctx: Arc<NestedRnsPolyContext>, inner: Vec<Vec<GateId>>) -> Self {
        Self { ctx, inner, _p: PhantomData }
    }

    pub fn input(ctx: Arc<NestedRnsPolyContext>, circuit: &mut PolyCircuit<P>) -> Self {
        let inner = (0..ctx.q_moduli_depth).map(|_| circuit.input(ctx.p_moduli.len())).collect();
        Self { ctx, inner, _p: PhantomData }
    }

    pub fn add_lazy_reduce(
        &self,
        other: &Self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.call_binary_subcircuit(other, enable_levels, circuit, &self.ctx.add_lazy_reduce_ids)
    }

    pub fn sub_lazy_reduce(
        &self,
        other: &Self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.call_binary_subcircuit(other, enable_levels, circuit, &self.ctx.sub_lazy_reduce_ids)
    }

    pub fn mul_lazy_reduce(
        &self,
        other: &Self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.call_binary_subcircuit(other, enable_levels, circuit, &self.ctx.mul_lazy_reduce_ids)
    }

    pub fn add_full_reduce(
        &self,
        other: &Self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.call_binary_subcircuit(other, enable_levels, circuit, &self.ctx.add_full_reduce_ids)
    }

    pub fn sub_full_reduce(
        &self,
        other: &Self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.call_binary_subcircuit(other, enable_levels, circuit, &self.ctx.sub_full_reduce_ids)
    }

    pub fn mul_full_reduce(
        &self,
        other: &Self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.call_binary_subcircuit(other, enable_levels, circuit, &self.ctx.mul_full_reduce_ids)
    }

    pub fn reconstruct(
        &self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> GateId {
        let levels = self.resolve_enable_levels(enable_levels);
        let mut sum_mod_q = circuit.const_zero_gate();
        assert!(
            levels <= self.ctx.reconstruct_ids.len(),
            "enable_levels exceeds reconstruct subcircuits"
        );
        for q_idx in 0..levels {
            let sum_q_k_scaled =
                circuit.call_sub_circuit(self.ctx.reconstruct_ids[q_idx], &self.inner[q_idx]);
            debug_assert_eq!(sum_q_k_scaled.len(), 1);
            sum_mod_q = circuit.add_gate(sum_mod_q, sum_q_k_scaled[0]);
        }
        sum_mod_q
    }

    pub fn benchmark_multiplication_tree(
        ctx: Arc<NestedRnsPolyContext>,
        circuit: &mut PolyCircuit<P>,
        height: usize,
        enable_levels: Option<usize>,
    ) {
        let num_inputs =
            1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");
        let mut current_layer: Vec<NestedRnsPoly<P>> =
            (0..num_inputs).map(|_| NestedRnsPoly::input(ctx.clone(), circuit)).collect();
        while current_layer.len() > 1 {
            debug_assert!(current_layer.len().is_multiple_of(2), "layer size must stay even");
            let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
            for pair in current_layer.chunks(2) {
                let parent = pair[0].mul_full_reduce(&pair[1], enable_levels, circuit);
                next_layer.push(parent);
            }
            current_layer = next_layer;
        }
        let root = current_layer.pop().expect("multiplication tree must contain at least one node");
        let out = root.reconstruct(enable_levels, circuit);
        circuit.output(vec![out]);
    }

    fn call_binary_subcircuit(
        &self,
        other: &Self,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
        subcircuit_ids: &[usize],
    ) -> Self {
        assert_eq!(self.inner.len(), other.inner.len(), "mismatched q_moduli depth");
        let levels = self.resolve_enable_levels(enable_levels);
        assert!(levels <= subcircuit_ids.len(), "enable_levels exceeds subcircuit depth");
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let left = &self.inner[q_idx];
            let right = &other.inner[q_idx];
            assert_eq!(left.len(), right.len(), "mismatched p_moduli depth");
            let mut inputs = Vec::with_capacity(left.len() + right.len());
            inputs.extend_from_slice(left);
            inputs.extend_from_slice(right);
            let outputs = circuit.call_sub_circuit(subcircuit_ids[q_idx], &inputs);
            result_inner.push(outputs);
        }
        Self { ctx: self.ctx.clone(), inner: result_inner, _p: PhantomData }
    }

    fn resolve_enable_levels(&self, enable_levels: Option<usize>) -> usize {
        let max_levels = self.inner.len();
        match enable_levels {
            Some(levels) => {
                assert!(levels <= max_levels, "enable_levels exceeds available levels");
                levels
            }
            None => max_levels,
        }
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

/// Return the first `count` pairwise coprime integers within the requested `bit_width`.
///
/// Deterministic: the output depends only on `bit_width` and `count` (no randomness).
pub(crate) fn sample_crt_primes(max_bit_width: usize, q_max: u64) -> Vec<u64> {
    assert!(max_bit_width > 1, "bit_width must be at least 2 bits");
    assert!(max_bit_width < 32, "bit_width must be less than 32 bits");
    assert!(
        max_bit_width <= usize::BITS as usize,
        "bit_width {max_bit_width} exceeds target pointer width {}",
        usize::BITS
    );
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
        let bound_sqrt = ((sum + results.len() as u64) * q_max) / 2;
        if BigUint::from(bound_sqrt).pow(2) < prod {
            prod_reached = true;
            break;
        }
    }

    if !prod_reached {
        panic!(
            "failed to find enough pairwise coprime integers with bit width {max_bit_width} to \
             satisfy q_max {q_max}; try increasing bit width"
        );
    }

    results
}

pub fn encode_nested_rns_poly<P: Poly>(
    p_moduli_bits: usize,
    params: &P::Params,
    input: &BigUint,
    q_level: Option<usize>,
) -> Vec<P> {
    let (q_moduli, _, q_moduli_depth) = params.to_crt();
    let active_q_level = q_level.unwrap_or(q_moduli_depth);
    assert!(
        active_q_level <= q_moduli_depth,
        "q_level exceeds q_moduli_depth: q_level={}, q_moduli_depth={}",
        active_q_level,
        q_moduli_depth
    );
    let q_moduli_max = q_moduli.iter().max().expect("there should be at least one q modulus");
    let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max);
    let p_moduli_depth = p_moduli.len();
    let mut polys = vec![Vec::with_capacity(p_moduli_depth); active_q_level];
    for (q_idx, &q_i) in q_moduli.iter().take(active_q_level).enumerate() {
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
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;
    const BASE_BITS: u32 = 6;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
        create_test_context_with_q_level(circuit, None)
    }

    fn create_test_context_with_q_level(
        circuit: &mut PolyCircuit<DCRTPoly>,
        q_level: Option<usize>,
    ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 6, 18, BASE_BITS);
        let ctx = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &params,
            P_MODULI_BITS,
            SCALE,
            false,
            q_level,
        ));
        println!("p moduli: {:?}", &ctx.p_moduli);
        let p = ctx.p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        println!("p: {}", p);
        (params, ctx)
    }

    #[sequential_test::sequential]
    #[test]
    fn test_nested_rns_poly_add_full_reduce_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let a_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        test_nested_rns_poly_add_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_nested_rns_poly_add_full_reduce_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let b_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_nested_rns_poly_add_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_nested_rns_poly_sub_full_reduce_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let a_value: BigUint = BigUint::ZERO;
        let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        test_nested_rns_poly_sub_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_nested_rns_poly_sub_full_reduce_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let b_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_nested_rns_poly_sub_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_nested_rns_poly_mul_full_reduce_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let a_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        test_nested_rns_poly_mul_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_nested_rns_poly_mul_full_reduce_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let b_value: BigUint = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_nested_rns_poly_mul_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_nested_rns_poly_respects_q_level() {
        let q_level = 2usize;
        let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_q_level(&mut setup_circuit, Some(q_level));

        assert_eq!(ctx.q_moduli_depth, q_level);
        assert_eq!(ctx.add_lazy_reduce_ids.len(), q_level);
        assert_eq!(ctx.sub_lazy_reduce_ids.len(), q_level);
        assert_eq!(ctx.mul_lazy_reduce_ids.len(), q_level);
        assert_eq!(ctx.add_full_reduce_ids.len(), q_level);
        assert_eq!(ctx.sub_full_reduce_ids.len(), q_level);
        assert_eq!(ctx.mul_full_reduce_ids.len(), q_level);
        assert_eq!(ctx.reconstruct_ids.len(), q_level);

        let (q_moduli, _, _) = params.to_crt();
        let q_level_modulus = q_moduli
            .iter()
            .take(q_level)
            .fold(BigUint::from(1u64), |acc, &qi| acc * BigUint::from(qi));

        // Max-value multiplication under the limited q_level.
        let mut max_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, max_ctx) = create_test_context_with_q_level(&mut max_circuit, Some(q_level));
        let max_value = &q_level_modulus - BigUint::from(1u64);
        test_nested_rns_poly_mul_full_reduce_generic(
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
        test_nested_rns_poly_mul_full_reduce_generic(
            random_circuit,
            params,
            random_ctx,
            a_value,
            b_value,
        );
    }

    fn test_nested_rns_poly_add_full_reduce_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedRnsPolyContext>,
        a_value: BigUint,
        b_value: BigUint,
    ) {
        let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let sum = poly_a.add_full_reduce(&poly_b, None, &mut circuit);
        let out = sum.reconstruct(None, &mut circuit);
        circuit.output(vec![out]);
        println!("non-free depth {}", circuit.non_free_depth());
        println!("circuit size {:?}", circuit.count_gates_by_type_vec());

        let modulus = params.modulus();
        // println!("modulus {:?}", &modulus);
        // println!("a_value {:?}", &a_value);
        // println!("b_value {:?}", &b_value);
        let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value, None);
        let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value, None);
        let expected_out = (&a_value + &b_value) % modulus.as_ref();
        // println!("expected_out {:?}", &expected_out);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(&plt_evaluator),
        );
        // println!("eval_results {:?}", eval_results);
        assert_eq!(eval_results.len(), 1);
        assert_eq!(eval_results[0].coeffs_biguints()[0], expected_out);
    }

    fn test_nested_rns_poly_sub_full_reduce_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedRnsPolyContext>,
        a_value: BigUint,
        b_value: BigUint,
    ) {
        let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let sum = poly_a.sub_full_reduce(&poly_b, None, &mut circuit);
        let out = sum.reconstruct(None, &mut circuit);
        circuit.output(vec![out]);
        println!("non-free depth {}", circuit.non_free_depth());
        println!("circuit size {:?}", circuit.count_gates_by_type_vec());

        let modulus = params.modulus();
        // println!("modulus {:?}", &modulus);
        // println!("a_value {:?}", &a_value);
        // println!("b_value {:?}", &b_value);
        let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value, None);
        let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value, None);
        let expected_out = {
            let mut value = &a_value + modulus.as_ref();
            value -= &b_value;
            value %= modulus.as_ref();
            value
        };
        // println!("expected_out {:?}", &expected_out);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(&plt_evaluator),
        );
        // println!("eval_results {:?}", eval_results);
        assert_eq!(eval_results.len(), 1);
        assert_eq!(eval_results[0].coeffs_biguints()[0], expected_out);
    }

    fn test_nested_rns_poly_mul_full_reduce_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedRnsPolyContext>,
        a_value: BigUint,
        b_value: BigUint,
    ) {
        let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let sum = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
        let out = sum.reconstruct(None, &mut circuit);
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
        let a_inputs =
            encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value, Some(active_q_level));
        let b_inputs =
            encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value, Some(active_q_level));
        let expected_mod_q_level =
            ((&a_value % &q_level_modulus) * (&b_value % &q_level_modulus)) % &q_level_modulus;
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(&plt_evaluator),
        );
        assert_eq!(eval_results.len(), 1);
        let output = eval_results[0].coeffs_biguints()[0].clone();
        if active_q_level == q_moduli.len() {
            let expected_out = (&a_value * &b_value) % modulus.as_ref();
            assert_eq!(output, expected_out);
        } else {
            assert_eq!(output.clone() % &q_level_modulus, expected_mod_q_level);
            for &q_i in q_moduli.iter().skip(active_q_level) {
                assert_eq!(output.clone() % BigUint::from(q_i), BigUint::ZERO);
            }
        }
    }
}
