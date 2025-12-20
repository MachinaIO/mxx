use crate::{
    circuit::{PolyCircuit, gate::GateId},
    lookup::PublicLut,
    poly::{Poly, PolyParams},
    utils::{mod_inverse, round_div},
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

#[derive(Debug, Clone)]
pub struct NestedRnsPolyContext {
    pub p_moduli: Vec<u64>,
    pub p_moduli_bits: usize,
    pub q_moduli_bits: usize,
    pub q_moduli_depth: usize,
    pub scale: u64,
    pub lut_mod_p: Vec<usize>,
    pub lut_x_to_y: Vec<usize>,
    pub lut_x_to_real: Vec<usize>,
    pub lut_real_to_v: usize,
    pub wires_p_moduli: Vec<GateId>,
    pub scalars_y: Vec<Vec<Vec<u32>>>, /* scalars_y[q_k][p_i][p_j] =
                                        * [[\hat{p_j}]_{q_k}]_{p_i} */
    pub scalars_v: Vec<Vec<u32>>, // scalars_v[q_k][p_i] = [[p]_{q_k}]_{p_i}
}

fn dummy_lut_map() -> HashMap<BigUint, (usize, BigUint)> {
    let mut map = HashMap::new();
    map.insert(BigUint::zero(), (0, BigUint::zero()));
    map
}

impl NestedRnsPolyContext {
    pub fn setup<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        p_moduli_bits: usize,
        scale: u64,
        dummy_scalar: bool,
    ) -> Self {
        let (q_moduli, q_moduli_bits, q_moduli_depth) = params.to_crt();
        let q_moduli_min = q_moduli.iter().min().expect("there should be at least one q modulus");
        let q_moduli_max = q_moduli.iter().max().expect("there should be at least one q modulus");
        let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max);
        let p_moduli_depth = p_moduli.len();
        if dummy_scalar {
            let dummy_map = dummy_lut_map();
            let dummy_lut = PublicLut::<P>::new_biguint(params, dummy_map);
            let lut_mod_p = circuit.register_public_lookup(dummy_lut.clone());
            let lut_x_to_y = circuit.register_public_lookup(dummy_lut.clone());
            let lut_x_to_real = circuit.register_public_lookup(dummy_lut.clone());
            let lut_real_to_v = circuit.register_public_lookup(dummy_lut);
            return Self {
                p_moduli,
                p_moduli_bits,
                q_moduli_bits,
                q_moduli_depth,
                scale,
                lut_mod_p: vec![lut_mod_p; p_moduli_depth],
                lut_x_to_y: vec![lut_x_to_y; p_moduli_depth],
                lut_x_to_real: vec![lut_x_to_real; p_moduli_depth],
                lut_real_to_v,
                wires_p_moduli: vec![circuit.const_zero_gate(); p_moduli_depth],
                scalars_y: vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth],
                scalars_v: vec![vec![0; p_moduli_depth]; q_moduli_depth],
            };
        }

        let p = p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        let p_over_pis = p_moduli.iter().map(|&p_i| &p / BigUint::from(p_i)).collect::<Vec<_>>();
        let sum_p_moduli: u64 = p_moduli.iter().copied().sum();

        let mut lut_mod_p = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_y = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_real = Vec::with_capacity(p_moduli_depth);
        let mut wires_p_moduli = Vec::with_capacity(p_moduli_depth);
        let mut scalars_y = vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth];
        let mut scalars_v = vec![vec![0; p_moduli_depth]; q_moduli_depth];

        for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
            let p_moduli_square = p_i as u64 * p_i as u64;
            let lut_mod_p_map_size = (p_i as u128)
                .checked_mul(sum_p_moduli as u128)
                .expect("lut_mod_p_map_size overflow");
            debug_assert!(
                lut_mod_p_map_size < *q_moduli_min as u128,
                "LUT size exceeds q modulus size; increase q_moduli_bits or decrease p_moduli_bits"
            );
            let lut_mod_p_map = HashMap::from_iter((0..(lut_mod_p_map_size as usize)).map(|t| {
                let input = BigUint::from(t as u64);
                let output = BigUint::from(t as u64 % p_i);
                (input, (t as usize, output))
            }));
            lut_mod_p.push(
                circuit.register_public_lookup(PublicLut::<P>::new_biguint(params, lut_mod_p_map)),
            );

            let p_moduli_big = BigUint::from(p_i);
            let p_over_pi_inv = {
                let residue = (&p_over_pis[p_i_idx] % &p_moduli_big)
                    .to_u64()
                    .expect("CRT residue must fit in u64");
                let inv = mod_inverse(residue, p_i).expect("CRT moduli must be coprime");
                // log_mem(format!("pi = {}, p_over_pi = {}, inv = {}", pi, p_over_pi, inv));
                BigUint::from(inv)
            };

            let lut_x_to_y_map = HashMap::from_iter((0..(p_moduli_square as usize)).map(|t| {
                let input = BigUint::from(t as u64);
                let output = (&input * &p_over_pi_inv) % &p_moduli_big;
                (input, (t as usize, output))
            }));
            lut_x_to_y.push(
                circuit.register_public_lookup(PublicLut::<P>::new_biguint(params, lut_x_to_y_map)),
            );

            let lut_x_to_real_map = HashMap::from_iter((0..(p_moduli_square as usize)).map(|t| {
                let input = BigUint::from(t as u64);
                let y = ((&input * &p_over_pi_inv) % &p_moduli_big)
                    .to_u64()
                    .expect("y must fit in u64");
                let output = BigUint::from(round_div(y * scale, p_i));
                (input, (t as usize, output))
            }));
            lut_x_to_real.push(
                circuit
                    .register_public_lookup(PublicLut::<P>::new_biguint(params, lut_x_to_real_map)),
            );

            wires_p_moduli.push(circuit.const_digits(&[p_i as u32]));

            for (q_idx, q_k) in q_moduli.iter().enumerate() {
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

        let max_real = scale * p_moduli_depth as u64;
        let lut_real_to_v_map = HashMap::from_iter((0..=max_real as usize).map(|t| {
            let input = BigUint::from(t as u64);
            let output = BigUint::from(round_div(t as u64, scale));
            (input, (t as usize, output))
        }));
        let lut_real_to_v =
            circuit.register_public_lookup(PublicLut::<P>::new_biguint(params, lut_real_to_v_map));
        Self {
            p_moduli,
            p_moduli_bits,
            q_moduli_bits,
            q_moduli_depth,
            scale,
            lut_mod_p,
            lut_x_to_y,
            lut_x_to_real,
            lut_real_to_v,
            wires_p_moduli,
            scalars_y,
            scalars_v,
        }
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

    pub fn add_lazy_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let sum = self.add_without_reduce(other, circuit);
        sum.lazy_reduce(circuit)
    }

    pub fn sub_lazy_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let diff = self.sub_without_reduce(other, circuit);
        diff.lazy_reduce(circuit)
    }

    pub fn mul_lazy_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let prod = self.mul_without_reduce(other, circuit);
        prod.lazy_reduce(circuit)
    }

    pub fn add_full_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let sum = self.add_without_reduce(other, circuit);
        sum.full_reduce(circuit)
    }

    pub fn sub_full_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let diff = self.sub_without_reduce(other, circuit);
        diff.full_reduce(circuit)
    }

    pub fn mul_full_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let prod = self.mul_without_reduce(other, circuit);
        prod.full_reduce(circuit)
    }

    pub fn reconstruct(&self, params: &P::Params, circuit: &mut PolyCircuit<P>) -> GateId {
        let mut sum_mod_q = circuit.const_zero_gate();
        let p =
            self.ctx.p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        let p_over_pis =
            self.ctx.p_moduli.iter().map(|&p_i| &p / BigUint::from(p_i)).collect::<Vec<_>>();

        for q_idx in 0..self.ctx.q_moduli_depth {
            let mut sum_without_reduce = circuit.const_zero_gate();
            let mut real_sum = circuit.const_zero_gate();
            let inner_mod_q_k = &self.inner[q_idx];
            for p_idx in 0..self.ctx.p_moduli.len() {
                let y_i =
                    circuit.public_lookup_gate(inner_mod_q_k[p_idx], self.ctx.lut_x_to_y[p_idx]);
                let y_i_p_j_hat = circuit.large_scalar_mul(y_i, &[p_over_pis[p_idx].clone()]);
                sum_without_reduce = circuit.add_gate(sum_without_reduce, y_i_p_j_hat);
                let real_i =
                    circuit.public_lookup_gate(inner_mod_q_k[p_idx], self.ctx.lut_x_to_real[p_idx]);
                real_sum = circuit.add_gate(real_sum, real_i);
            }
            let v = circuit.public_lookup_gate(real_sum, self.ctx.lut_real_to_v);
            let pv = circuit.large_scalar_mul(v, &[p.clone()]);
            let sum_q_k = circuit.sub_gate(sum_without_reduce, pv);
            let (_, reconst_coeff) = params.to_crt_coeffs(q_idx);
            let sum_q_k_scaled = circuit.large_scalar_mul(sum_q_k, &[reconst_coeff]);
            sum_mod_q = circuit.add_gate(sum_mod_q, sum_q_k_scaled);
        }
        sum_mod_q
    }

    pub fn benchmark_multiplication_tree(
        ctx: Arc<NestedRnsPolyContext>,
        params: &P::Params,
        circuit: &mut PolyCircuit<P>,
        height: usize,
    ) {
        let num_inputs =
            1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");
        let mut current_layer: Vec<NestedRnsPoly<P>> =
            (0..num_inputs).map(|_| NestedRnsPoly::input(ctx.clone(), circuit)).collect();
        while current_layer.len() > 1 {
            debug_assert!(current_layer.len().is_multiple_of(2), "layer size must stay even");
            let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
            for pair in current_layer.chunks(2) {
                let parent = pair[0].mul_full_reduce(&pair[1], circuit);
                next_layer.push(parent);
            }
            current_layer = next_layer;
        }
        let root = current_layer.pop().expect("multiplication tree must contain at least one node");
        let out = root.reconstruct(params, circuit);
        circuit.output(vec![out]);
    }

    pub(crate) fn add_without_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            self.ctx.p_moduli, other.ctx.p_moduli,
            "cannot add NestedRnsPoly with different p_moduli"
        );
        assert_eq!(
            self.ctx.q_moduli_depth, other.ctx.q_moduli_depth,
            "cannot add NestedRnsPoly with different q_moduli_depth"
        );
        let mut result_inner = Vec::with_capacity(self.ctx.q_moduli_depth);
        for q_idx in 0..self.ctx.q_moduli_depth {
            let mut result_p_moduli = Vec::with_capacity(self.ctx.p_moduli.len());
            for p_idx in 0..self.ctx.p_moduli.len() {
                let sum_gate =
                    circuit.add_gate(self.inner[q_idx][p_idx], other.inner[q_idx][p_idx]);
                result_p_moduli.push(sum_gate);
            }
            result_inner.push(result_p_moduli);
        }
        Self { ctx: self.ctx.clone(), inner: result_inner, _p: PhantomData }
    }

    pub(crate) fn sub_without_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            self.ctx.p_moduli, other.ctx.p_moduli,
            "cannot add NestedRnsPoly with different p_moduli"
        );
        assert_eq!(
            self.ctx.q_moduli_depth, other.ctx.q_moduli_depth,
            "cannot add NestedRnsPoly with different q_moduli_depth"
        );
        let mut result_inner = Vec::with_capacity(self.ctx.q_moduli_depth);
        for q_idx in 0..self.ctx.q_moduli_depth {
            let mut result_p_moduli = Vec::with_capacity(self.ctx.p_moduli.len());
            for p_idx in 0..self.ctx.p_moduli.len() {
                let sum_gate =
                    circuit.add_gate(self.inner[q_idx][p_idx], self.ctx.wires_p_moduli[p_idx]);
                let sub_gate = circuit.sub_gate(sum_gate, other.inner[q_idx][p_idx]);
                result_p_moduli.push(sub_gate);
            }
            result_inner.push(result_p_moduli);
        }
        Self { ctx: self.ctx.clone(), inner: result_inner, _p: PhantomData }
    }

    pub(crate) fn mul_without_reduce(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            self.ctx.p_moduli, other.ctx.p_moduli,
            "cannot mul NestedRnsPoly with different p_moduli"
        );
        assert_eq!(
            self.ctx.q_moduli_depth, other.ctx.q_moduli_depth,
            "cannot mul NestedRnsPoly with different q_moduli_depth"
        );
        let mut result_inner = Vec::with_capacity(self.ctx.q_moduli_depth);
        for q_idx in 0..self.ctx.q_moduli_depth {
            let mut result_p_moduli = Vec::with_capacity(self.ctx.p_moduli.len());
            for p_idx in 0..self.ctx.p_moduli.len() {
                let mul_gate =
                    circuit.mul_gate(self.inner[q_idx][p_idx], other.inner[q_idx][p_idx]);
                // let mul_mod = circuit.public_lookup_gate(mul_gate, self.ctx.lut_mod_p[p_idx]);
                result_p_moduli.push(mul_gate);
            }
            result_inner.push(result_p_moduli);
        }
        Self { ctx: self.ctx.clone(), inner: result_inner, _p: PhantomData }
    }

    pub(crate) fn lazy_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let mut result_inner = Vec::with_capacity(self.ctx.q_moduli_depth);
        for q_idx in 0..self.ctx.q_moduli_depth {
            let mut result_p_moduli = Vec::with_capacity(self.ctx.p_moduli.len());
            for p_idx in 0..self.ctx.p_moduli.len() {
                let reduced_gate =
                    circuit.public_lookup_gate(self.inner[q_idx][p_idx], self.ctx.lut_mod_p[p_idx]);
                result_p_moduli.push(reduced_gate);
            }
            result_inner.push(result_p_moduli);
        }
        Self { ctx: self.ctx.clone(), inner: result_inner, _p: PhantomData }
    }

    pub(crate) fn full_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let mut result_inner = Vec::with_capacity(self.ctx.q_moduli_depth);
        for q_idx in 0..self.ctx.q_moduli_depth {
            // 1. y_i = [[x]_{p_i} * (p/p_oi)^(-1} mod p_i] mod p_i
            let ys = (0..self.ctx.p_moduli.len())
                .map(|p_idx| {
                    circuit.public_lookup_gate(self.inner[q_idx][p_idx], self.ctx.lut_x_to_y[p_idx])
                })
                .collect::<Vec<_>>();
            // 2. real_i = round_div(y_i * scale, p_i), but the input is x_i rather than y_i
            let reals = (0..self.ctx.p_moduli.len())
                .map(|p_idx| {
                    circuit
                        .public_lookup_gate(self.inner[q_idx][p_idx], self.ctx.lut_x_to_real[p_idx])
                })
                .collect::<Vec<_>>();
            // 3. sum_i real_i
            let mut real_sum = circuit.const_zero_gate();
            for &real in reals.iter() {
                real_sum = circuit.add_gate(real_sum, real);
            }
            // 4. v = round_div(real_sum, scale)
            let v = circuit.public_lookup_gate(real_sum, self.ctx.lut_real_to_v);
            // 5. p_i_sum = (sum_j y_j * [[\hat{p_j}]_{q_k}]_{p_i}) - v * [[p]_{q_k}]_{p_i}
            let mut p_i_sums = Vec::with_capacity(self.ctx.p_moduli.len());
            for p_idx in 0..self.ctx.p_moduli.len() {
                let mut p_i_sum = circuit.const_zero_gate();
                for p_j_idx in 0..self.ctx.p_moduli.len() {
                    let y_j = ys[p_j_idx];
                    let term =
                        circuit.small_scalar_mul(y_j, &[self.ctx.scalars_y[q_idx][p_idx][p_j_idx]]);
                    p_i_sum = circuit.add_gate(p_i_sum, term);
                }
                let term = circuit.small_scalar_mul(v, &[self.ctx.scalars_v[q_idx][p_idx]]);
                p_i_sum = circuit.sub_gate(p_i_sum, term);
                let p_i_sum_mod_p = circuit.public_lookup_gate(p_i_sum, self.ctx.lut_mod_p[p_idx]);
                p_i_sums.push(p_i_sum_mod_p);
            }
            result_inner.push(p_i_sums);
        }
        Self { ctx: self.ctx.clone(), inner: result_inner, _p: PhantomData }
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
    // let mut sum = 0u64;
    let mut prod = BigUint::one();
    let mut prod_reached = false;
    let q_max_squared = BigUint::from(q_max).pow(2);

    // Prefer larger moduli (bigger `p = ∏ p_i` for the same depth), but keep selection
    // deterministic.
    for candidate in lower..upper {
        if results.iter().all(|&chosen| gcd_u64(candidate, chosen) == 1) {
            results.push(candidate);
            // sum += candidate;
            prod *= BigUint::from(candidate);
        } else {
            continue;
        }
        // let bound = ((sum + results.len() as u64) * q_max) / 2;
        if &q_max_squared * BigUint::from(candidate) < prod {
            prod_reached = true;
            break;
        }
    }

    if !prod_reached {
        panic!(
            "failed to find enough pairwise coprime integers with bit width {max_bit_width} to \
             satisfy q_max {q_max}; try increasing bit width",
        );
    }

    // if results.len() != count {
    //     panic!(
    //         "failed to find {count} pairwise coprime integers with bit width {bit_width}; only {}
    // found",         results.len()
    //     );
    // }

    results
}

pub fn encode_nested_rns_poly<P: Poly>(
    p_moduli_bits: usize,
    params: &P::Params,
    input: &BigUint,
) -> Vec<P> {
    let (q_moduli, _, _) = params.to_crt();
    let q_moduli_max = q_moduli.iter().max().expect("there should be at least one q modulus");
    let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max);
    let p_moduli_depth = p_moduli.len();
    let mut polys = vec![Vec::with_capacity(p_moduli_depth); q_moduli.len()];
    for (q_idx, &q_i) in q_moduli.iter().enumerate() {
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
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        simulator::SimulatorContext,
        utils::bigdecimal_bits_ceil,
    };
    use bigdecimal::BigDecimal;

    const P_MODULI_BITS: usize = 5;
    const SCALE: u64 = 1 << 7;
    const BASE_BITS: u32 = 8;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
        let params = DCRTPolyParams::new(4, 3, 15, BASE_BITS);
        let ctx =
            Arc::new(NestedRnsPolyContext::setup(circuit, &params, P_MODULI_BITS, SCALE, false));
        println!("p moduli: {:?}", &ctx.p_moduli);
        let p = ctx.p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        println!("p: {}", p);
        (params, ctx)
    }

    #[test]
    fn test_nested_rns_poly_add_full_reduce_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let a_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        test_nested_rns_poly_add_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

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

    #[test]
    fn test_nested_rns_poly_sub_full_reduce_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let a_value: BigUint = BigUint::zero();
        let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        test_nested_rns_poly_sub_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

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

    #[test]
    fn test_nested_rns_poly_mul_full_reduce_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let a_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        let b_value: BigUint = modulus.as_ref() - BigUint::from(1u64);
        test_nested_rns_poly_mul_full_reduce_generic(circuit, params, ctx, a_value, b_value);
    }

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

    #[test]
    fn test_nested_rns_benchmark_multiplication_tree() {
        let height = 5usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let num_inputs = 1usize << height;
        NestedRnsPoly::benchmark_multiplication_tree(ctx.clone(), &params, &mut circuit, height);
        println!("non-free depth {}", circuit.non_free_depth());
        println!("circuit size {:?}", circuit.count_gates_by_type_vec());

        let sim_ctx = Arc::new(SimulatorContext::new(
            BigDecimal::from(100u64).sqrt().expect("sqrt(100) should succeed"),
            BigDecimal::from(params.ring_dimension() as u64)
                .sqrt()
                .expect("sqrt(ring_dimension) should succeed"),
            BigDecimal::from(1u64 << BASE_BITS),
            params.modulus_digits(),
        ));
        let out = circuit.simulate_max_h_norm(
            sim_ctx,
            BigDecimal::from(
                1u64.checked_shl(P_MODULI_BITS as u32)
                    .expect("P_MODULI_BITS is too large to shift"),
            ),
            circuit.num_input(),
        );
        println!(
            "out[0].h_norm.poly_norm.norm bits = {}",
            bigdecimal_bits_ceil(&out[0].h_norm.poly_norm.norm)
        );

        let mut eval_inputs: Vec<DCRTPoly> =
            Vec::with_capacity(num_inputs * ctx.q_moduli_depth * ctx.p_moduli.len());
        for _ in 0..num_inputs {
            eval_inputs.extend(encode_nested_rns_poly(P_MODULI_BITS, &params, &BigUint::zero()));
        }
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &eval_inputs, Some(plt_evaluator));

        assert_eq!(eval_results.len(), 1);
        assert_eq!(eval_results[0], DCRTPoly::const_zero(&params));
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
        let sum = poly_a.add_full_reduce(&poly_b, &mut circuit);
        let out = sum.reconstruct(&params, &mut circuit);
        circuit.output(vec![out]);
        println!("non-free depth {}", circuit.non_free_depth());
        println!("circuit size {:?}", circuit.count_gates_by_type_vec());

        let modulus = params.modulus();
        // println!("modulus {:?}", &modulus);
        // println!("a_value {:?}", &a_value);
        // println!("b_value {:?}", &b_value);
        let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value);
        let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value);
        let expected_out = (&a_value + &b_value) % modulus.as_ref();
        // println!("expected_out {:?}", &expected_out);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
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
        let sum = poly_a.sub_full_reduce(&poly_b, &mut circuit);
        let out = sum.reconstruct(&params, &mut circuit);
        circuit.output(vec![out]);
        println!("non-free depth {}", circuit.non_free_depth());
        println!("circuit size {:?}", circuit.count_gates_by_type_vec());

        let modulus = params.modulus();
        // println!("modulus {:?}", &modulus);
        // println!("a_value {:?}", &a_value);
        // println!("b_value {:?}", &b_value);
        let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value);
        let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value);
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
            Some(plt_evaluator),
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
        let sum = poly_a.mul_full_reduce(&poly_b, &mut circuit);
        let out = sum.reconstruct(&params, &mut circuit);
        circuit.output(vec![out]);
        println!("non-free depth {}", circuit.non_free_depth());
        println!("circuit size {:?}", circuit.count_gates_by_type_vec());

        let modulus = params.modulus();
        // println!("modulus {:?}", &modulus);
        // println!("a_value {:?}", &a_value);
        // println!("b_value {:?}", &b_value);
        let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value);
        let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value);
        let expected_out = (&a_value * &b_value) % modulus.as_ref();
        // println!("expected_out {:?}", &expected_out);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );
        // println!("eval_results {:?}", eval_results);
        assert_eq!(eval_results.len(), 1);
        assert_eq!(eval_results[0].coeffs_biguints()[0], expected_out);
    }
}
