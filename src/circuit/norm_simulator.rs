// use crate::{
//     circuit::{Evaluable, PolyCircuit, gate::GateId},
//     element::PolyElem,
//     impl_binop_with_refs,
//     lookup::{PltEvaluator, PublicLut},
//     poly::dcrt::poly::DCRTPoly,
// };
// use itertools::Itertools;
// use num_bigint::BigUint;
// use num_traits::One;
// use serde::{Deserialize, Serialize};
// use std::ops::{Add, Mul, Sub};

// impl PolyCircuit<DCRTPoly> {
//     pub fn simulate_bgg_norm(
//         &self,
//         dim: u32,
//         base_bits: u32,
//         packed_input_norms: Vec<BigUint>,
//     ) -> NormBounds
//     where
//         NormSimulator: Evaluable<P = DCRTPoly>,
//     {
//         let one = NormSimulator::new(MSqrtPolyCoeffs::one(), BigUint::one(), dim, base_bits);
//         let inputs = packed_input_norms
//             .into_iter()
//             .map(|plaintext_norm| {
//                 NormSimulator::new(MSqrtPolyCoeffs::one(), plaintext_norm, dim, base_bits)
//             })
//             .collect::<Vec<_>>();
//         let plt_evaluator = NormPltLweEvaluator::new(dim, 1 << base_bits);
//         let outputs = self.eval(&(), &one, &inputs, Some(plt_evaluator));
//         NormBounds::from_norm_simulators(&outputs)
//     }
// }

// #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// pub struct NormBounds {
//     h_norms: Vec<Vec<String>>,
// }

// impl NormBounds {
//     pub fn from_norm_simulators(simulators: &[NormSimulator]) -> Self {
//         let h_norms = simulators
//             .iter()
//             .map(|simulator| simulator.h_norm.0.iter().map(|coeff| coeff.to_string()).collect())
//             .collect();
//         Self { h_norms }
//     }
// }

// // Note: h_norm and plaintext_norm computed here can be larger than the modulus `q`.
// // In such a case, the error after circuit evaluation could be too large.
// #[derive(Debug, Clone)]
// pub struct NormSimulator {
//     pub dim_sqrt: f64,
//     pub base: u32,
//     pub m_sqrt: f64,
//     pub plaintext_norm: f64,
//     pub h_norm: f64,
// }

// impl NormSimulator {
//     pub fn new(plaintext_norm: BigUint, dim: u32, base_bits: u32) -> Self {
//         let base = 1 << base_bits;
//         let dim_sqrt = (dim as f64).sqrt().ceil() as u32;
//         Self { h_norm, plaintext_norm, dim_sqrt, base }
//     }
// }

// impl_binop_with_refs!(NormSimulator => Add::add(self, rhs: &NormSimulator) -> NormSimulator {
//     NormSimulator {
//         h_norm: &self.h_norm + &rhs.h_norm,
//         plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
//         dim_sqrt: self.dim_sqrt,
//         base: self.base,
//     }
// });

// // Note: norm of the subtraction result is bounded by a sum of the norms of the input matrices,
// // i.e., |A-B| < |A| + |B|
// impl_binop_with_refs!(NormSimulator => Sub::sub(self, rhs: &NormSimulator) -> NormSimulator {
//     NormSimulator {
//         h_norm: &self.h_norm + &rhs.h_norm,
//         plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
//         dim_sqrt: self.dim_sqrt,
//         base: self.base,
//     }
// });

// impl_binop_with_refs!(NormSimulator => Mul::mul(self, rhs: &NormSimulator) -> NormSimulator {
//     NormSimulator {
//         h_norm: self.h_norm.right_rotate(self.dim_sqrt as u64 * (self.base as u64 - 1)) +
// &rhs.h_norm * &(&self.plaintext_norm * BigUint::from(self.dim_sqrt)),         plaintext_norm:
// &self.plaintext_norm * &rhs.plaintext_norm * self.dim_sqrt,         dim_sqrt: self.dim_sqrt,
//         base: self.base,
//     }
// });

// impl Evaluable for NormSimulator {
//     type Params = ();
//     type P = DCRTPoly;

//     fn rotate(&self, _: &Self::Params, _: i32) -> Self {
//         self.clone()
//     }

//     fn small_scalar_mul(&self, _: &Self::Params, scalar: &[u32]) -> Self {
//         let scalar_max = BigUint::from(*scalar.iter().max().unwrap());
//         NormSimulator {
//             h_norm: self.h_norm.clone() * &scalar_max,
//             plaintext_norm: &self.plaintext_norm * &scalar_max,
//             dim_sqrt: self.dim_sqrt,
//             base: self.base,
//         }
//     }

//     fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
//         let scalar_max = scalar.iter().max().unwrap();
//         NormSimulator {
//             h_norm: self.h_norm.right_rotate(self.dim_sqrt as u64 * (self.base as u64 - 1)),
//             plaintext_norm: &self.plaintext_norm * scalar_max * BigUint::from(self.dim_sqrt as
// u64),             dim_sqrt: self.dim_sqrt,
//             base: self.base,
//         }
//     }
// }

// pub struct NormPltLweEvaluator {
//     norm_preimage: MSqrtPolyCoeffs,
// }

// impl NormPltLweEvaluator {
//     pub fn new(dim: u32, base: u32) -> Self {
//         let c_0 = 6.0;
//         let sigma = 4.578;
//         let scale = c_0 * sigma * (base as f64 + 1.0) * sigma;
//         let dim_sqrt = (dim as f64).sqrt();
//         let m_sqrt_coeff = BigUint::from((scale * dim_sqrt).ceil() as u128);
//         let const_coeff = BigUint::from((scale * (1.414 * dim_sqrt + 4.7)).ceil() as u128);
//         let norm_preimage = MSqrtPolyCoeffs::new(vec![const_coeff, m_sqrt_coeff]);
//         Self { norm_preimage }
//     }
// }

// impl PltEvaluator<NormSimulator> for NormPltLweEvaluator {
//     fn public_lookup(
//         &self,
//         _: &(),
//         plt: &PublicLut<DCRTPoly>,
//         input: NormSimulator,
//         _: GateId,
//     ) -> NormSimulator {
//         let left_h_norm = self.norm_preimage.clone();
//         let right_h_norm = input.h_norm.right_rotate(input.dim_sqrt as u64) *
// &self.norm_preimage;         let h_norm = left_h_norm + right_h_norm;
//         NormSimulator {
//             h_norm,
//             plaintext_norm: plt.max_output_row().1.value().clone(),
//             dim_sqrt: input.dim_sqrt,
//             base: input.base,
//         }
//     }
// }

// // // each variable X = \sqrt{m} = \sqrt{d log_{base} q}
// // #[derive(Debug, Clone, PartialEq, Eq)]
// // pub struct MSqrtPolyCoeffs(Vec<BigUint>);

// // impl MSqrtPolyCoeffs {
// //     pub fn new(coeffs: Vec<BigUint>) -> Self {
// //         Self(coeffs)
// //     }

// //     #[inline]
// //     pub fn one() -> Self {
// //         Self::new(vec![BigUint::one()])
// //     }

// //     // corresponding to multiplying \sqrt{m}
// //     #[inline]
// //     pub fn right_rotate(&self, scale: u64) -> Self {
// //         let mut coeffs = vec![BigUint::ZERO];
// //         coeffs.extend(self.0.iter().map(|coeff| coeff * scale).collect_vec());
// //         Self(coeffs)
// //     }
// // }

// // impl_binop_with_refs!(MSqrtPolyCoeffs => Add::add(self, rhs: &MSqrtPolyCoeffs) ->
// MSqrtPolyCoeffs // {     let self_len = self.0.len();
// //     let rhs_len = rhs.0.len();
// //     let max_len = self_len.max(rhs_len);

// //     let mut result = Vec::with_capacity(max_len);

// //     for i in 0..max_len {
// //         let a = if i < self_len { self.0[i].clone() } else { BigUint::ZERO };
// //         let b = if i < rhs_len { rhs.0[i].clone() } else { BigUint::ZERO };
// //         result.push(a + b);
// //     }

// //     MSqrtPolyCoeffs(result)
// // });

// // impl_binop_with_refs!(MSqrtPolyCoeffs => Mul::mul(self, rhs: &MSqrtPolyCoeffs) ->
// MSqrtPolyCoeffs // {     let self_len = self.0.len();
// //     let rhs_len = rhs.0.len();

// //     if self_len == 0 || rhs_len == 0 {
// //         return MSqrtPolyCoeffs(Vec::new());
// //     }

// //     let mut result = vec![BigUint::ZERO; self_len + rhs_len - 1];

// //     for (i, a) in self.0.iter().enumerate() {
// //         for (j, b) in rhs.0.iter().enumerate() {
// //             let prod = a * b;
// //             result[i + j] += &prod;
// //         }
// //     }

// //     MSqrtPolyCoeffs(result)
// // });

// // impl Mul<BigUint> for MSqrtPolyCoeffs {
// //     type Output = Self;
// //     fn mul(self, rhs: BigUint) -> Self::Output {
// //         self * &rhs
// //     }
// // }

// // impl Mul<&BigUint> for MSqrtPolyCoeffs {
// //     type Output = Self;
// //     fn mul(self, rhs: &BigUint) -> Self {
// //         &self * rhs
// //     }
// // }

// // impl Mul<&BigUint> for &MSqrtPolyCoeffs {
// //     type Output = MSqrtPolyCoeffs;
// //     fn mul(self, rhs: &BigUint) -> MSqrtPolyCoeffs {
// //         MSqrtPolyCoeffs(self.0.iter().map(|a| a * rhs).collect())
// //     }
// // }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use num_bigint::BigUint;

//     fn create_test_error_simulator(
//         ring_dim: u32,
//         h_norm_values: Vec<u32>,
//         plaintext_norm: u32,
//     ) -> NormSimulator {
//         let h_norm =
// MSqrtPolyCoeffs::new(h_norm_values.into_iter().map(BigUint::from).collect());         let
// plaintext_norm = BigUint::from(plaintext_norm);         NormSimulator::new(h_norm,
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
//         // Create NormSimulator instances for inputs
//         let input1 =
//             NormSimulator::new(MSqrtPolyCoeffs::new(vec![BigUint::one()]), BigUint::one(), 16,
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
