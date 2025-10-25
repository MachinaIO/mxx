use std::{collections::HashMap, sync::Arc};

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::packed_plt::PackedPlt,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use primal::Primes;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1PolyContext<P: Poly> {
    pub l1_moduli: Vec<u64>,
    pub l1_moduli_bits: usize,
    pub max_degree: usize,
    pub luts: (Vec<PackedPlt<P>>, Vec<PackedPlt<P>>), // (add, mul)
    pub l1_moduli_wires: Vec<GateId>,
}

impl<P: Poly> L1PolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        l1_moduli_bits: usize,
        max_degree: usize,
        dummy_scalar: bool,
    ) -> Self {
        let (_, crt_bits, crt_depth) = params.to_crt();
        let l1_moduli_depth = (2 * crt_bits).div_ceil(l1_moduli_bits);
        let l1_moduli = sample_crt_primes(l1_moduli_bits, l1_moduli_depth);
        let mut add_luts = Vec::with_capacity(l1_moduli_depth);
        let mut mul_luts = Vec::with_capacity(l1_moduli_depth);
        let mut l1_moduli_wires = Vec::with_capacity(l1_moduli_depth);
        let reconst_coeffs = (0..crt_depth).map(|i| params.to_crt_coeffs(i).1).collect::<Vec<_>>();
        let q = params.modulus().into();
        for &modulus in l1_moduli.iter() {
            let add_max = (1 << (l1_moduli_bits + 1)) as u64 - 1;
            let add_map_slot: HashMap<BigUint, (usize, BigUint)> =
                HashMap::from_iter((0..=add_max).map(|t| {
                    let input = BigUint::from(t);
                    let output = BigUint::from(t % modulus);
                    (input, (t as usize, output))
                }));
            add_luts.push(PackedPlt::setup(
                circuit,
                params,
                max_degree,
                add_map_slot,
                dummy_scalar,
            ));
            let mul_max = (1 << (2 * l1_moduli_bits)) as u64 - 1;
            let base = 1u64 << l1_moduli_bits;
            let mul_map_slot: HashMap<BigUint, (usize, BigUint)> =
                HashMap::from_iter((0..=mul_max).map(|t| {
                    let input = BigUint::from(t);
                    let t0 = t % base;
                    let t1 = t / base;
                    let output = BigUint::from((t0 * t1) % modulus);
                    (input, (t as usize, output))
                }));
            mul_luts.push(PackedPlt::setup(
                circuit,
                params,
                max_degree,
                mul_map_slot,
                dummy_scalar,
            ));
            let mut modulus_const = BigUint::zero();
            for reconst_coeff in reconst_coeffs.iter() {
                modulus_const = (&modulus_const + reconst_coeff * modulus) % q.as_ref();
            }
            let wire_id = circuit.const_poly(&P::from_biguint_to_constant(params, modulus_const));
            l1_moduli_wires.push(wire_id);
        }
        let luts = (add_luts, mul_luts);
        Self { l1_moduli, l1_moduli_bits, max_degree, luts, l1_moduli_wires }
    }

    pub fn l1_moduli_depth(&self) -> usize {
        self.l1_moduli.len()
    }
}

/// Return the first `count` primes that fall within the requested `bit_width`.
pub(crate) fn sample_crt_primes(bit_width: usize, count: usize) -> Vec<u64> {
    assert!(bit_width > 1, "bit_width must be at least 2 bits");
    assert!(bit_width < 32, "bit_width must be less than 32 bits");
    assert!(
        bit_width <= usize::BITS as usize,
        "bit_width {bit_width} exceeds target pointer width {}",
        usize::BITS
    );
    assert!(count > 0, "count must be greater than 0");

    let lower = 1u64 << (bit_width - 1);
    let upper = 1u64 << bit_width;
    let mut results: Vec<u64> = Vec::with_capacity(count);

    for prime in
        Primes::all().skip_while(|&p| (p as u64) < lower).take_while(|&p| (p as u64) < upper)
    {
        results.push(prime as u64);
        if results.len() == count {
            break;
        }
    }

    if results.len() != count {
        panic!(
            "failed to find {count} primes with bit width {bit_width}; only {} found",
            results.len()
        );
    }

    results
}

#[derive(Debug, Clone)]
pub struct L1Poly<P: Poly> {
    pub ctx: Arc<L1PolyContext<P>>,
    pub inner: Vec<GateId>,
}

impl<P: Poly> L1Poly<P> {
    pub fn new(ctx: Arc<L1PolyContext<P>>, inner: Vec<GateId>) -> Self {
        Self { ctx, inner }
    }

    pub fn input(ctx: Arc<L1PolyContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let inner = circuit.input(ctx.l1_moduli_depth());
        Self { ctx, inner }
    }

    pub fn constant(ctx: Arc<L1PolyContext<P>>, circuit: &mut PolyCircuit<P>, polys: &[P]) -> Self {
        let inner = polys.into_iter().map(|poly| circuit.const_poly(poly)).collect::<Vec<_>>();
        Self { ctx, inner }
    }

    pub fn zero(ctx: Arc<L1PolyContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let inner = vec![circuit.const_zero_gate(); ctx.l1_moduli_depth()];
        Self { ctx, inner }
    }

    pub fn rotate(&self, shift: usize) -> Self {
        let mut new_inner = Vec::with_capacity(self.ctx.l1_moduli_depth());
        for i in 0..self.ctx.l1_moduli_depth() {
            let idx = (i + shift) % self.ctx.l1_moduli_depth();
            new_inner.push(self.inner[idx]);
        }
        Self { ctx: self.ctx.clone(), inner: new_inner }
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let mut new_inner = Vec::with_capacity(self.ctx.l1_moduli_depth());
        for (i, (&l, &r)) in self.inner.iter().zip(other.inner.iter()).enumerate() {
            let t = circuit.add_gate(l, r);
            new_inner.push(self.ctx.luts.0[i].lookup_all(circuit, t));
        }
        Self { ctx: self.ctx.clone(), inner: new_inner }
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let mut new_inner = Vec::with_capacity(self.ctx.l1_moduli_depth());
        for (i, (&l, &r)) in self.inner.iter().zip(other.inner.iter()).enumerate() {
            let t = circuit.add_gate(l, self.ctx.l1_moduli_wires[i]);
            // circuit.print(t, format!("L1 sub intermediate t at mod {}", self.ctx.l1_moduli[i]));
            // circuit.output(vec![t]);
            let t = circuit.sub_gate(t, r);
            new_inner.push(self.ctx.luts.0[i].lookup_all(circuit, t));
        }
        Self { ctx: self.ctx.clone(), inner: new_inner }
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let mut new_inner = Vec::with_capacity(self.ctx.l1_moduli_depth());
        for (i, (&l, &r)) in self.inner.iter().zip(other.inner.iter()).enumerate() {
            let scaled = circuit.small_scalar_mul(r, &[1u32 << self.ctx.l1_moduli_bits]);
            let t = circuit.add_gate(l, scaled);
            new_inner.push(self.ctx.luts.1[i].lookup_all(circuit, t));
        }
        Self { ctx: self.ctx.clone(), inner: new_inner }
    }
}

// pub fn encode_l1_poly<P: Poly>(
//     l1_moduli_bits: usize,
//     params: &P::Params,
//     inputs: &[Vec<u64>],
// ) -> Vec<P> {
//     let ring_n = params.ring_dimension() as usize;
//     let (_, crt_bits, _) = params.to_crt();
//     let l1_moduli_depth = crt_bits.div_ceil(l1_moduli_bits);
//     let l1_moduli = sample_crt_primes(l1_moduli_bits, l1_moduli_depth);
//     let mut limb_slots: Vec<Vec<BigUint>> = vec![vec![BigUint::zero(); ring_n]; l1_moduli_depth];
//     for (crt_idx, inputs_per_crt) in inputs.iter().enumerate() {
//         // CRT reconstruction coefficients c_i = (q/qi) * (q/qi)^{-1} (mod qi)
//         let (_, reconst_coeff) = params.to_crt_coeffs(crt_idx);
//         for (eval_idx, &input_raw) in inputs_per_crt.iter().enumerate() {
//             for (l1_idx, &modulus) in l1_moduli.iter().enumerate() {
//                 let input = BigUint::from(input_raw % modulus);
//                 limb_slots[l1_idx][eval_idx] = (&limb_slots[crt_idx][eval_idx] +
//                     (&reconst_coeff * input)) %
//                     params.modulus().into().as_ref();
//             }
//         }
//     }
//     limb_slots.iter().map(|slots| P::from_biguints_eval(params, slots)).collect::<Vec<_>>()
// }
