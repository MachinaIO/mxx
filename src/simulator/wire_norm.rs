use crate::{
    circuit::{Evaluable, PolyCircuit, gate::GateId},
    element::PolyElem,
    impl_binop_with_refs,
    lookup::{PltEvaluator, PublicLut},
    poly::dcrt::poly::DCRTPoly,
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
};
use bigdecimal::BigDecimal;
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use serde::{Deserialize, Serialize};
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
    ) -> Vec<BigDecimal>
    where
        WireNorm: Evaluable<P = DCRTPoly>,
    {
        let nrow = (ctx.log_base_q + 2) + input_size * ctx.log_base_q;
        let one_wire = WireNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        let input_wire = WireNorm::new(
            PolyNorm::sample_bound(ctx.clone(), input_norm_bound),
            PolyMatrixNorm::one(ctx.clone(), nrow, ctx.log_base_q),
        );
        let plt_evaluator = NormPltLweEvaluator::new(ctx.clone());
        let outputs = self.eval(&(), &one_wire, &vec![input_wire; input_size], Some(plt_evaluator));
        outputs.into_iter().map(|out| out.h_norm.poly_norm.norm).collect::<Vec<_>>()
    }
}

// Note: h_norm and plaintext_norm computed here can be larger than the modulus `q`.
// In such a case, the error after circuit evaluation could be too large.
#[derive(Debug, Clone)]
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
        let scalar_poly = PolyNorm::sample_bound(self.clone_ctx(), scalar_max);
        WireNorm {
            h_norm: self.h_norm.clone() * &scalar_poly,
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }

    fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::sample_bound(self.clone_ctx(), scalar_bd);
        WireNorm {
            h_norm: self.h_norm.clone() *
                PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().log_base_q),
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltLweEvaluator {
    preimage_norm: PolyMatrixNorm,
}

impl NormPltLweEvaluator {
    pub fn new(ctx: Arc<SimulatorContext>) -> Self {
        let preimage_norm = PolyMatrixNorm::sample_preimage(ctx);
        Self { preimage_norm }
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
        let h_norm = &self.preimage_norm + (&input.h_norm * &self.preimage_norm);
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::sample_bound(input.clone_ctx(), plaintext_bd);
        WireNorm { h_norm, plaintext_norm }
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
