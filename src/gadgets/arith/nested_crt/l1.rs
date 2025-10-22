use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::packed_plt::PackedPlt,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use primal::Primes;
use rayon::iter::{FromParallelIterator, IntoParallelIterator};

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
        let (_, crt_bits, _) = params.to_crt();
        let l1_moduli_depth = crt_bits.div_ceil(l1_moduli_bits);
        let l1_moduli = Self::sample_crt_primes(l1_moduli_bits, l1_moduli_depth);
        let mut add_luts = Vec::with_capacity(l1_moduli_depth);
        let mut mul_luts = Vec::with_capacity(l1_moduli_depth);
        let mut l1_moduli_wires = Vec::with_capacity(l1_moduli_depth);
        for &modulus in l1_moduli.iter() {
            let add_max = 2 * modulus as usize;
            let add_map_slot: HashMap<BigUint, (usize, BigUint)> =
                HashMap::from_iter((0..=add_max).map(|t| {
                    let input = BigUint::from(t as u64);
                    let output = BigUint::from(t as u64 % modulus);
                    (input, (t as usize, output))
                }));
            add_luts.push(PackedPlt::setup(
                circuit,
                params,
                max_degree,
                add_map_slot,
                dummy_scalar,
            ));
            let mul_max = modulus * (modulus - 1);
            let mul_map_slot: HashMap<BigUint, (usize, BigUint)> =
                HashMap::from_iter((0..=mul_max).map(|t| {
                    let input = BigUint::from(t as u64);
                    let t0 = t % modulus;
                    let t1 = t / modulus;
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
            let wire_id = circuit.const_digits(&[modulus as u32]);
            l1_moduli_wires.push(wire_id);
        }
        let luts = (add_luts, mul_luts);
        Self { l1_moduli, l1_moduli_bits, max_degree, luts, l1_moduli_wires }
    }

    pub fn l1_moduli_depth(&self) -> usize {
        self.l1_moduli.len()
    }

    /// Return the first `count` primes that fall within the requested `bit_width`.
    fn sample_crt_primes(bit_width: usize, count: usize) -> Vec<u64> {
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
            let t = circuit.sub_gate(t, r);
            new_inner.push(self.ctx.luts.0[i].lookup_all(circuit, t));
        }
        Self { ctx: self.ctx.clone(), inner: new_inner }
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let mut new_inner = Vec::with_capacity(self.ctx.l1_moduli_depth());
        for (i, (&l, &r)) in self.inner.iter().zip(other.inner.iter()).enumerate() {
            let scaled = circuit.small_scalar_mul(r, &[self.ctx.l1_moduli[i] as u32]);
            let t = circuit.add_gate(l, scaled);
            new_inner.push(self.ctx.luts.1[i].lookup_all(circuit, t));
        }
        Self { ctx: self.ctx.clone(), inner: new_inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primal::is_prime;
}
