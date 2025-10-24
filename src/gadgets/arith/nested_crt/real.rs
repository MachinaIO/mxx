use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::{
        arith::nested_crt::l1::{L1Poly, L1PolyContext},
        packed_plt::PackedPlt,
    },
    poly::Poly,
};
use num_bigint::BigUint;
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealPolyContext<P: Poly> {
    pub scale: u64,
    pub l1_ctx: Arc<L1PolyContext<P>>,
    pub l1_to_real_luts: Vec<PackedPlt<P>>,
    pub real_to_int_lut: PackedPlt<P>,
}

impl<P: Poly> RealPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        l1_moduli_bits: usize,
        scale: u64,
        max_degree: usize,
        dummy_scalar: bool,
    ) -> Self {
        let l1_ctx =
            L1PolyContext::setup(circuit, params, l1_moduli_bits, max_degree, dummy_scalar);
        let l1_moduli = &l1_ctx.l1_moduli;
        let max_add_count = l1_moduli.len() as u64;
        let mut l1_to_real_luts = Vec::with_capacity(l1_moduli.len());
        for &modulus in l1_moduli.iter() {
            let l1_to_real_map_slot: HashMap<BigUint, (usize, BigUint)> =
                HashMap::from_iter((0..(modulus as usize)).map(|t| {
                    let input = BigUint::from(t as u64);
                    let output = BigUint::from(t as u64 * scale / modulus);
                    (input, (t as usize, output))
                }));
            l1_to_real_luts.push(PackedPlt::setup(
                circuit,
                params,
                max_degree,
                l1_to_real_map_slot,
                dummy_scalar,
            ));
        }
        let max_real = scale * max_add_count;
        let real_to_int_map_slot: HashMap<BigUint, (usize, BigUint)> =
            HashMap::from_iter((0..=max_real as usize).map(|t| {
                let input = BigUint::from(t as u64);
                let output = BigUint::from(t as u64 / scale);
                (input, (t as usize, output))
            }));
        let real_to_int_lut =
            PackedPlt::setup(circuit, params, max_degree, real_to_int_map_slot, dummy_scalar);
        Self { scale, l1_ctx: Arc::new(l1_ctx), l1_to_real_luts, real_to_int_lut }
    }
}

#[derive(Debug, Clone)]
pub struct RealPoly<P: Poly> {
    pub ctx: Arc<RealPolyContext<P>>,
    pub inner: Vec<GateId>,
}

impl<P: Poly> RealPoly<P> {
    pub fn new(ctx: Arc<RealPolyContext<P>>, inner: Vec<GateId>) -> Self {
        Self { ctx, inner }
    }

    pub fn from_l1_poly(
        ctx: Arc<RealPolyContext<P>>,
        l1_poly: &L1Poly<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        debug_assert_eq!(ctx.l1_ctx.as_ref(), l1_poly.ctx.as_ref());
        let mut new_inner = Vec::with_capacity(l1_poly.inner.len());
        for (idx, (lut, &input)) in ctx.l1_to_real_luts.iter().zip(l1_poly.inner.iter()).enumerate()
        {
            circuit.print(input, format!("Input to L1->Real LUT idx {idx}"));
            let output = lut.lookup_all(circuit, input);
            circuit.print(output, format!("Real LUT output idx {idx}"));
            new_inner.push(output);
        }
        Self { ctx, inner: new_inner }
    }

    pub fn sum_to_l1_poly(&self, circuit: &mut PolyCircuit<P>) -> L1Poly<P> {
        let mut sum = circuit.const_zero_gate();
        // println!("Summing real polys: {:?}", self.inner);
        for poly in self.inner.iter() {
            sum = circuit.add_gate(sum, *poly);
        }
        circuit.print(sum, format!("Real sum before LUT"));
        let int = self.ctx.real_to_int_lut.lookup_all(circuit, sum);
        // println!("After Real to Int LUT: {:?}", int);
        let new_inner = vec![int; self.ctx.l1_ctx.l1_moduli.len()];
        L1Poly::new(self.ctx.l1_ctx.clone(), new_inner)
    }
}
