// use num_bigint::BigUint;
// use num_traits::Num;
// use openfhe::{
//     cxx::UniquePtr,
//     ffi::{DCRTPoly as DCRTPolyCxx, DCRTPolyGenFromConst, DCRTPolyGenFromVec, GenModulus},
//     parse_coefficients_bytes,
// };
// use rayon::prelude::*;

// use std::sync::Arc;

// use crate::ring::{FinRingElem, FiniteRing};

// #[derive(Clone, PartialEq, Eq)]
// pub struct DCRTPolyParams {
//     /// polynomial ring dimension
//     ring_dimension: u32,
//     /// size of the tower
//     crt_depth: usize,
//     /// number of bits of each tower's modulus
//     crt_bits: usize,
//     /// bit size of the base for the gadget vector and decomposition
//     base_bits: u32,
// }

// impl DCRTPolyParams {
//     pub fn new(ring_dimension: u32, crt_depth: usize, crt_bits: usize, base_bits: u32) -> Self {
//         assert!(
//             ring_dimension.is_power_of_two(),
//             "ring_dimension must be a power of 2"
//         );
//         Self {
//             ring_dimension,
//             crt_depth,
//             crt_bits,
//             base_bits,
//         }
//     }

//     pub fn get_ring(&self) -> FiniteRing {
//         let modulus = BigUint::from_str_radix(
//             &GenModulus(self.ring_dimension, self.crt_depth, self.crt_bits),
//             10,
//         )
//         .expect("invalid string");
//         FiniteRing::new(modulus)
//     }

//     pub fn crt_depth(&self) -> usize {
//         self.crt_depth
//     }

//     pub fn crt_bits(&self) -> usize {
//         self.crt_bits
//     }

//     pub fn ring_dimension(&self) -> u32 {
//         self.ring_dimension
//     }
// }

// pub struct DCRTPoly {
//     ptr_poly: Arc<UniquePtr<DCRTPolyCxx>>,
// }

// impl DCRTPoly {
//     pub fn new(ptr_poly: UniquePtr<DCRTPolyCxx>) -> Self {
//         Self {
//             ptr_poly: ptr_poly.into(),
//         }
//     }

//     pub fn get_ptr_poly(&self) -> &UniquePtr<DCRTPolyCxx> {
//         &self.ptr_poly
//     }

//     // fn poly_gen_from_vec(params: &DCRTPolyParams, values: Vec<String>) -> Self {
//     //     DCRTPoly::new(DCRTPolyGenFromVec(
//     //         params.ring_dimension(),
//     //         params.crt_depth(),
//     //         params.crt_bits(),
//     //         &values,
//     //     ))
//     // }

//     // fn poly_gen_from_const(params: &DCRTPolyParams, value: String) -> Self {
//     //     DCRTPoly::new(DCRTPolyGenFromConst(
//     //         params.ring_dimension(),
//     //         params.crt_depth(),
//     //         params.crt_bits(),
//     //         &value,
//     //     ))
//     // }

//     // fn coeffs(&self) -> Vec<FinRingElem> {
//     //     let poly_encoding = self.ptr_poly.GetCoefficientsBytes();
//     //     let parsed_values = parse_coefficients_bytes(&poly_encoding);
//     //     let coeffs = parsed_values.coefficients;
//     //     let modulus = parsed_values.modulus;
//     //     let ring = FiniteRing::new(modulus);
//     //     let elems: Vec<&FinRingElem> = coeffs.into_par_iter().map(|c| &ring.elem(c)).collect();
//     //     elems
//     // }
// }
