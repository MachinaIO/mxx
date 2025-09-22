use crate::{
    circuit::{Evaluable, PolyCircuit, gate::GateId},
    element::PolyElem,
    impl_binop_with_refs,
    lookup::{PltEvaluator, PublicLut},
    poly::dcrt::poly::DCRTPoly,
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

impl PolyCircuit<DCRTPoly> {
    pub fn simulate_max_h_norm(
        &self,
        ctx: Arc<SimulatorContext>,
        input_norm_bound: f64,
        input_size: usize,
    ) -> Vec<WireNorm>
    where
        WireNorm: Evaluable<P = DCRTPoly>,
    {
        let nrow = get_nrow_for_input(&ctx, input_size);
        let one_wire = WireNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        let input_wire = WireNorm::new(
            PolyNorm::new(ctx.clone(), input_norm_bound),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        let plt_evaluator = NormPltLweEvaluator::new(ctx.clone(), input_size);
        self.eval(&(), &one_wire, &vec![input_wire; input_size], Some(plt_evaluator))
    }
}

// Note: h_norm and plaintext_norm computed here can be larger than the modulus `q`.
// In such a case, the error after circuit evaluation could be too large.
#[derive(Debug, Clone, PartialEq)]
pub struct WireNorm {
    pub plaintext_norm: PolyNorm,
    pub h_norm: PolyMatrixNorm,
}

impl WireNorm {
    pub fn new(plaintext_norm: PolyNorm, h_norm: PolyMatrixNorm) -> Self {
        debug_assert_eq!(plaintext_norm.ctx, h_norm.clone_ctx());
        Self { plaintext_norm, h_norm }
    }

    #[inline]
    pub fn ctx(&self) -> &SimulatorContext {
        &self.plaintext_norm.ctx
    }
    #[inline]
    pub fn clone_ctx(&self) -> Arc<SimulatorContext> {
        self.plaintext_norm.ctx.clone()
    }
}

impl_binop_with_refs!(WireNorm => Add::add(self, rhs: &WireNorm) -> WireNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    WireNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        h_norm: &self.h_norm + &rhs.h_norm
    }
});

// Note: norm of the subtraction result is bounded by a sum of the norms of the input matrices,
// i.e., |A-B| < |A| + |B|
impl_binop_with_refs!(WireNorm => Sub::sub(self, rhs: &WireNorm) -> WireNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    WireNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        h_norm: &self.h_norm + &rhs.h_norm
    }
});

impl_binop_with_refs!(WireNorm => Mul::mul(self, rhs: &WireNorm) -> WireNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    WireNorm {
        plaintext_norm: &self.plaintext_norm * &rhs.plaintext_norm,
        h_norm: &self.h_norm * PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().log_base_q) + rhs.h_norm.clone() * &self.plaintext_norm
    }
});

impl Evaluable for WireNorm {
    type Params = ();
    type P = DCRTPoly;

    fn rotate(&self, _: &Self::Params, _: i32) -> Self {
        self.clone()
    }

    fn small_scalar_mul(&self, _: &Self::Params, scalar: &[u32]) -> Self {
        let scalar_max = *scalar.iter().max().unwrap() as f64;
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_max);
        WireNorm {
            h_norm: self.h_norm.clone() * &scalar_poly,
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }

    fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_f64 = scalar_max.to_f64().expect("scalar fits into f64");
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_f64);
        WireNorm {
            h_norm: self.h_norm.clone() *
                PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().log_base_q),
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltLweEvaluator {
    pub preimage1_norm: PolyMatrixNorm,
    pub preimage2_norm: PolyMatrixNorm,
}

impl NormPltLweEvaluator {
    pub fn new(ctx: Arc<SimulatorContext>, input_size: usize) -> Self {
        let c_0 = 1.8f64;
        let c_1 = 4.7f64;
        let sigma = 4.578f64;
        let two_sqrt = (2.0f64).sqrt();
        let log_base_q_sqrt = (ctx.log_base_q as f64).sqrt();
        let term = ctx.ring_dim_sqrt * log_base_q_sqrt + two_sqrt * ctx.ring_dim_sqrt + c_1;
        let norm = c_0 * 6.0 * sigma * ((ctx.base + 1.0) * sigma) * term;
        let ncol = ctx.log_base_q;
        let preimage1_norm = PolyMatrixNorm::new(
            ctx.clone(),
            get_nrow_for_input(&ctx, input_size),
            ncol,
            norm,
            None,
        );
        let preimage2_norm = PolyMatrixNorm::new(ctx.clone(), ctx.log_base_q, ncol, norm, None);
        Self { preimage1_norm, preimage2_norm }
    }
}

impl PltEvaluator<WireNorm> for NormPltLweEvaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        input: WireNorm,
        id: GateId,
    ) -> WireNorm {
        let h_norm = &self.preimage1_norm + (&input.h_norm * &self.preimage2_norm);
        let (_, elem) = plt.max_output_row();
        let plaintext_f64 =
            elem.value().to_f64().expect("lookup plaintext coefficient fits into f64");
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_f64);
        WireNorm { h_norm, plaintext_norm }
    }
}

fn get_nrow_for_input(ctx: &SimulatorContext, input_size: usize) -> usize {
    (ctx.log_base_q + 2) + input_size * ctx.log_base_q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        lookup::PublicLut,
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        simulator::SimulatorContext,
    };
    use std::collections::HashMap;

    fn make_ctx() -> Arc<SimulatorContext> {
        // secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=(128/32)*7 = 28
        Arc::new(SimulatorContext::new(
            50.0,   // secpar_sqrt
            1024.0, // ring_dim_sqrt
            32.0,   // base
            28,     // log_base_q
        ))
    }

    #[test]
    fn test_simulate_max_h_norm_identity_input() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        // Output both inputs unchanged
        circuit.output(vec![inputs[0], inputs[1]]);

        let input_bound = 1.0f64;
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound, 2);
        assert_eq!(out.len(), 2);
        let nrow = (ctx.log_base_q + 2) + 2 * ctx.log_base_q;
        let expected = WireNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        assert_eq!(out[0], expected);
        assert_eq!(out[1], expected);
    }

    #[test]
    fn test_wire_norm_addition() {
        // Circuit: out = in0 + in1
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.add_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = 5.0f64;

        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound, 2);
        assert_eq!(out.len(), 1);
        // Build expected from input wires and add them
        let nrow = (ctx.log_base_q + 2) + 2 * ctx.log_base_q;
        let in_wire = WireNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        let expected = &in_wire + &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_subtraction() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.sub_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = 5.0f64;
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound, 2);
        assert_eq!(out.len(), 1);
        let nrow = (ctx.log_base_q + 2) + 2 * ctx.log_base_q;
        let in_wire = WireNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        let expected = &in_wire - &in_wire; // subtraction bound equals addition bound
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_multiplication() {
        // ctx: secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=28
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.mul_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = 5.0f64;
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound, 2);
        assert_eq!(out.len(), 1);

        // Build expected = in_wire * in_wire
        let nrow = (ctx.log_base_q + 2) + 2 * ctx.log_base_q;
        let in_wire = WireNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        let expected = &in_wire * &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_public_lookup_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let mut f = HashMap::new();
        f.insert(
            DCRTPoly::from_usize_to_constant(&params, 0),
            (0usize, DCRTPoly::from_usize_to_constant(&params, 5)),
        );
        f.insert(
            DCRTPoly::from_usize_to_constant(&params, 1),
            (1usize, DCRTPoly::from_usize_to_constant(&params, 7)),
        );
        let plt = PublicLut::<DCRTPoly>::new(f);

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = 5.0f64;
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound, 1);
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, 7.0);
    }
}
