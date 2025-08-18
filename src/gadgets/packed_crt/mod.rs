use crate::{
    circuit::{PolyCircuit, evaluable::Evaluable, gate::GateId},
    gadgets::{
        crt::{bigunit::BigUintPoly, montgomery::MontgomeryPoly, *},
        isolate::*,
    },
    poly::Poly,
};
use num_bigint::BigUint;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedCrtContext<P: Poly> {
    pub crt_ctx: Arc<CrtContext<P>>,
    pub isolation_gadget: Arc<IsolationGadget<P>>,
    pub num_limbs_per_pack: usize,
    pub total_limbs_per_crt_poly: usize,
}

impl<P: Poly> PackedCrtContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        pack_bit_size: usize,
    ) -> Self {
        debug_assert_eq!(pack_bit_size % limb_bit_size, 0);
        let crt_ctx = Arc::new(CrtContext::setup(circuit, params, limb_bit_size));
        let max_degree = pack_bit_size.div_ceil(limb_bit_size) as u16;
        let max_norm = 1 << limb_bit_size;
        let isolation_gadget =
            Arc::new(IsolationGadget::setup(circuit, params, max_degree, max_norm));
        let num_limbs_per_pack = pack_bit_size / limb_bit_size;
        let num_crt_slots = crt_ctx.mont_ctxes.len();
        let num_limbs_per_slot = crt_ctx.mont_ctxes[0].num_limbs;
        let total_limbs_per_crt_poly = num_crt_slots * num_limbs_per_slot;
        Self { crt_ctx, isolation_gadget, num_limbs_per_pack, total_limbs_per_crt_poly }
    }
}

#[derive(Debug, Clone)]
pub struct PackedCrtPoly<P: Poly> {
    pub ctx: Arc<PackedCrtContext<P>>,
    pub packed_polys: Vec<GateId>,
}

impl<P: Poly> PackedCrtPoly<P> {
    pub fn input(
        ctx: Arc<PackedCrtContext<P>>,
        circuit: &mut PolyCircuit<P>,
        num_crt_polys: usize,
    ) -> Self {
        let total_limbs_needed = num_crt_polys * ctx.total_limbs_per_crt_poly;
        let num_packed_poly = total_limbs_needed.div_ceil(ctx.num_limbs_per_pack);
        let packed_polys = circuit.input(num_packed_poly);
        Self { ctx, packed_polys }
    }

    pub fn unpack(&self, circuit: &mut PolyCircuit<P>) -> Vec<CrtPoly<P>> {
        let mut const_polys = vec![];
        for poly in self.packed_polys.iter() {
            const_polys.extend(self.ctx.isolation_gadget.isolate_terms(circuit, *poly));
        }
        let total_limbs_needed = self.packed_polys.len() * self.ctx.total_limbs_per_crt_poly;
        let num_limbs_per_slot = self.ctx.crt_ctx.mont_ctxes[0].num_limbs;
        let mut crt_polys = vec![];

        // Group const_polys by complete CRT polynomial sets
        let mut poly_index = 0;
        while poly_index < total_limbs_needed {
            let mut slots = vec![];

            // For each CRT slot, collect the required number of limbs
            for mont_ctx in &self.ctx.crt_ctx.mont_ctxes {
                let mut limbs = vec![];
                for _ in 0..num_limbs_per_slot {
                    limbs.push(const_polys[poly_index]);
                    poly_index += 1;
                }

                let value = BigUintPoly::new(mont_ctx.big_uint_ctx.clone(), limbs);
                let mont_poly = MontgomeryPoly::new(mont_ctx.clone(), value);
                slots.push(mont_poly);
            }
            let crt_poly = CrtPoly::new(self.ctx.crt_ctx.clone(), slots);
            crt_polys.push(crt_poly);
        }

        crt_polys
    }
}

pub fn biguint_to_packed_crt_polys<P: Poly>(
    limb_bit_size: usize,
    pack_bit_size: usize,
    params: &P::Params,
    inputs: &[BigUint],
) -> Vec<P> {
    let all_const_polys = inputs
        .into_iter()
        .flat_map(|input| biguint_to_crt_poly::<P>(limb_bit_size, params, input))
        .collect::<Vec<_>>();
    debug_assert_eq!(pack_bit_size % limb_bit_size, 0);
    let num_limbs_per_pack = pack_bit_size / limb_bit_size;
    let mut packed_polys = vec![];
    let mut new_packed_poly = P::const_zero(params);
    let mut new_packed_poly_deg = 0i32;
    for const_poly in all_const_polys.into_iter() {
        let rotated = const_poly.rotate(params, new_packed_poly_deg);
        new_packed_poly = new_packed_poly + rotated;
        if new_packed_poly_deg as usize == num_limbs_per_pack - 1 {
            new_packed_poly_deg = 0;
            packed_polys.push(new_packed_poly);
            new_packed_poly = P::const_zero(params);
        } else {
            new_packed_poly_deg += 1;
        }
    }
    if new_packed_poly_deg > 0 {
        packed_polys.push(new_packed_poly);
    }
    packed_polys
}
