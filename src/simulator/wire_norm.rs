use crate::{
    circuit::{Evaluable, PolyCircuit, gate::GateId},
    element::PolyElem,
    impl_binop_with_refs,
    lookup::{PltEvaluator, PublicLut},
    poly::dcrt::poly::DCRTPoly,
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use num_traits::FromPrimitive;
use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

impl PolyCircuit<DCRTPoly> {
    pub fn simulate_max_h_norm(
        &self,
        ctx: Arc<SimulatorContext>,
        input_norm_bound: BigDecimal,
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
        let scalar_max = BigDecimal::from(*scalar.iter().max().unwrap());
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_max);
        WireNorm {
            h_norm: self.h_norm.clone() * &scalar_poly,
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }

    fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_bd);
        WireNorm {
            h_norm: self.h_norm.clone() *
                PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().log_base_q),
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltLweEvaluator {
    preimage1_norm: PolyMatrixNorm,
    preimage2_norm: PolyMatrixNorm,
}

impl NormPltLweEvaluator {
    pub fn new(ctx: Arc<SimulatorContext>, input_size: usize) -> Self {
        let c_0 = BigDecimal::from_f64(1.8).unwrap();
        let c_1 = BigDecimal::from_f64(4.7).unwrap();
        let sigma = BigDecimal::from_f64(4.578).unwrap();
        let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
        let log_base_q_sqrt =
            BigDecimal::from(ctx.log_base_q as u64).sqrt().expect("sqrt(log_base_q) failed");
        let term = ctx.ring_dim_sqrt.clone() * log_base_q_sqrt +
            two_sqrt * ctx.ring_dim_sqrt.clone() +
            c_1.clone();
        let norm: BigDecimal = c_0 * 6 * sigma.clone() * ((ctx.base.clone() + 1) * sigma) * term;
        let ncol = ctx.log_base_q;
        let preimage1_norm = PolyMatrixNorm::new(
            ctx.clone(),
            get_nrow_for_input(&ctx, input_size),
            ncol,
            norm.clone(),
            None,
        );
        let preimage2_norm =
            PolyMatrixNorm::new(ctx.clone(), ctx.log_base_q, ncol, norm.clone(), None);
        Self { preimage1_norm, preimage2_norm }
    }
}

impl PltEvaluator<WireNorm> for NormPltLweEvaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        input: WireNorm,
        _: GateId,
    ) -> WireNorm {
        let h_norm = &self.preimage1_norm + (&input.h_norm * &self.preimage2_norm);
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
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
    use bigdecimal::BigDecimal;
    use std::collections::HashMap;

    fn make_ctx() -> Arc<SimulatorContext> {
        // secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=(128/32)*7 = 28
        Arc::new(SimulatorContext::new(
            BigDecimal::from(50u64),   // secpar_sqrt
            BigDecimal::from(1024u64), // ring_dim_sqrt
            BigDecimal::from(32u64),   // base
            28,                        // log_base_q
        ))
    }

    #[test]
    fn test_simulate_max_h_norm_identity_input() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        // Output both inputs unchanged
        circuit.output(vec![inputs[0], inputs[1]]);

        let input_bound = BigDecimal::from(1u64);
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound.clone(), 2);
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
        let input_bound = BigDecimal::from(5u64);

        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound.clone(), 2);
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
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound.clone(), 2);
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
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound.clone(), 2);
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
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_h_norm(ctx.clone(), input_bound.clone(), 1);
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use num_bigint::BigUint;

//     fn create_test_error_simulator(
//         ring_dim: u32,
//         h_norm_values: Vec<u32>,
//         plaintext_norm: u32,
//     ) -> WireNorm {
//         let h_norm =
// MSqrtPolyCoeffs::new(h_norm_values.into_iter().map(BigUint::from).collect());         let
// plaintext_norm = BigUint::from(plaintext_norm);         WireNorm::new(h_norm,
// plaintext_norm, ring_dim, 1)     }

//     #[test]
//     fn test_error_simulator_addition() {
//         // Create two ErrorSimulator instances
//         let sim1 = create_test_error_simulator(16, vec![10u32], 5);
//         let sim2 = create_test_error_simulator(16, vec![20u32], 7);

//         // Test addition
//         let result = sim1 + sim2;

//         // Verify the result
//         assert_eq!(result.h_norm.0[0], BigUint::from(30u32)); // 10 + 20
//         assert_eq!(result.plaintext_norm, BigUint::from(12u32)); // 5 + 7
//         assert_eq!(result.dim_sqrt.pow(2), 16);
//     }

//     #[test]
//     fn test_error_simulator_subtraction() {
//         // Create two ErrorSimulator instances
//         let sim1 = create_test_error_simulator(16, vec![10u32], 5);
//         let sim2 = create_test_error_simulator(16, vec![20u32], 7);

//         // Test subtraction (which is actually addition in this implementation)
//         let result = sim1 - sim2;

//         // Verify the result (should be the same as addition)
//         assert_eq!(result.h_norm.0[0], BigUint::from(30u32)); // 10 + 20
//         assert_eq!(result.plaintext_norm, BigUint::from(12u32)); // 5 + 7
//         assert_eq!(result.dim_sqrt.pow(2), 16);
//     }

//     #[test]
//     fn test_error_simulator_multiplication() {
//         // Create two ErrorSimulator instances
//         let sim1 = create_test_error_simulator(16, vec![10u32], 5);
//         let sim2 = create_test_error_simulator(16, vec![20u32], 7);

//         // Test multiplication
//         let result = sim1 * sim2;

//         // Verify the result
//         // h_norm = sim1.h_norm.right_rotate(4) + sim2.h_norm * (sim1.plaintext_norm * 4)

//         // Check the length of the h_norm vector
//         assert_eq!(result.h_norm.0.len(), 2);

//         // First element should be 20 * 5 * 4 (from sim2.h_norm * sim1.plaintext_norm * dim_sqrt)
//         assert_eq!(result.h_norm.0[0], BigUint::from(400u32)); // 20 * 5

//         // Second element should be 10 * 4 (from right_rotate)
//         assert_eq!(result.h_norm.0[1], BigUint::from(40u32));

//         assert_eq!(result.plaintext_norm, BigUint::from(140u32)); // 5 * 7 * 4
//         assert_eq!(result.dim_sqrt.pow(2), 16);
//     }

//     #[test]
//     fn test_simulate_bgg_norm() {
//         // Create a simple circuit: (input1 + input2) * input3
//         let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
//         let inputs = circuit.input(3);
//         let add_gate = circuit.add_gate(inputs[0], inputs[1]);
//         let mul_gate = circuit.mul_gate(add_gate, inputs[2]);
//         circuit.output(vec![mul_gate]);

//         // Simulate norm using the circuit
//         let plaintext_norms = vec![BigUint::from(1u32); 3];
//         let norms = circuit.simulate_bgg_norm(16, 1, plaintext_norms);

//         // Manually calculate the expected norm
//         // Create WireNorm instances for inputs
//         let input1 =
//             WireNorm::new(MSqrtPolyCoeffs::new(vec![BigUint::one()]), BigUint::one(), 16,
// 1);         let input2 = input1.clone();
//         let input3 = input1.clone();

//         // Perform the operations manually
//         let add_result = input1 + input2;
//         let mul_result = add_result * input3;
//         let expected = NormBounds::from_norm_simulators(&[mul_result]);
//         // Verify the result
//         assert_eq!(norms.h_norms.len(), 1);
//         assert_eq!(norms, expected);
//     }
// }
