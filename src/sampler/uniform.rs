use crate::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    openfhe_guard::ensure_openfhe_warmup,
    parallel_iter,
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
    sampler::{DistType, PolyUniformSampler},
};
use openfhe::ffi;
use rayon::prelude::*;
#[cfg(feature = "disk")]
use std::ops::Range;

use crate::poly::dcrt::params::DCRTPolyParams;

pub struct DCRTPolyUniformSampler {}

impl DCRTPolyUniformSampler {
    fn sample_poly_unchecked(&self, params: &DCRTPolyParams, dist: &DistType) -> DCRTPoly {
        let sampled_poly = match dist {
            DistType::FinRingDist => ffi::DCRTPolyGenFromDug(
                params.ring_dimension(),
                params.crt_depth(),
                params.crt_bits(),
            ),
            DistType::GaussDist { sigma } => ffi::DCRTPolyGenFromDgg(
                params.ring_dimension(),
                params.crt_depth(),
                params.crt_bits(),
                *sigma,
            ),
            DistType::BitDist => ffi::DCRTPolyGenFromBug(
                params.ring_dimension(),
                params.crt_depth(),
                params.crt_bits(),
            ),
            DistType::TernaryDist => ffi::DCRTPolyGenFromTug(
                params.ring_dimension(),
                params.crt_depth(),
                params.crt_bits(),
            ),
        };
        if sampled_poly.is_null() {
            panic!("Attempted to dereference a null pointer");
        }
        DCRTPoly::new(sampled_poly)
    }
}

impl Default for DCRTPolyUniformSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl PolyUniformSampler for DCRTPolyUniformSampler {
    type M = DCRTPolyMatrix;

    fn new() -> Self {
        Self {}
    }

    fn sample_poly(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        dist: &DistType,
    ) -> <Self::M as PolyMatrix>::P {
        ensure_openfhe_warmup(params);
        self.sample_poly_unchecked(params, dist)
    }

    fn sample_uniform(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        // Ensure OpenFHE's NTT tables for these parameters are initialized before we enter the
        // parallel sampling loop.
        ensure_openfhe_warmup(params);

        #[cfg(feature = "disk")]
        {
            let mut new_matrix = DCRTPolyMatrix::new_empty(params, nrow, ncol);
            let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<DCRTPoly>> {
                parallel_iter!(row_offsets)
                    .map(|_| {
                        parallel_iter!(col_offsets.clone())
                            .map(|_| self.sample_poly_unchecked(params, &dist))
                            .collect()
                    })
                    .collect()
            };
            new_matrix.replace_entries(0..nrow, 0..ncol, f);
            new_matrix
        }
        #[cfg(not(feature = "disk"))]
        {
            let c: Vec<Vec<DCRTPoly>> = parallel_iter!(0..nrow)
                .map(|_| {
                    parallel_iter!(0..ncol)
                        .map(|_| {
                            self.sample_poly_unchecked(params, &dist)
                        })
                        .collect()
                })
                .collect();

            DCRTPolyMatrix::from_poly_vec(params, c)
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::{__PAIR, __TestState};
    use num_bigint::BigUint;

    use crate::poly::dcrt::params::DCRTPolyParams;

    use super::*;

    #[test]
    #[sequential_test::sequential]
    fn test_ternary_dist_values() {
        // Test that TernaryDist actually produces values in {-1, 0, 1}
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();

        // Sample a small matrix to check values
        let matrix = sampler.sample_uniform(&params, 1, 1, DistType::TernaryDist);
        let poly = matrix.entry(0, 0);
        let coeffs = poly.coeffs();

        // Verify each coefficient is in {-1, 0, 1}
        for coeff in coeffs.iter() {
            let value = coeff.value.clone();
            assert!(
                value == BigUint::ZERO ||
                    value == BigUint::from(1u32) ||
                    value == params.modulus().as_ref() - BigUint::from(1u32),
                "Coefficient value {:?} is not in {{-1, 0, 1}}",
                value
            );
        }
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ring_dist() {
        let params = DCRTPolyParams::default();

        // Test FinRingDist
        let sampler = DCRTPolyUniformSampler::new();
        let matrix1 = sampler.sample_uniform(&params, 20, 5, DistType::FinRingDist);
        assert_eq!(matrix1.row_size(), 20);
        assert_eq!(matrix1.col_size(), 5);

        let matrix2 = sampler.sample_uniform(&params, 20, 5, DistType::FinRingDist);

        let sampler2 = DCRTPolyUniformSampler::new();
        let matrix3 = sampler2.sample_uniform(&params, 5, 12, DistType::FinRingDist);
        assert_eq!(matrix3.row_size(), 5);
        assert_eq!(matrix3.col_size(), 12);

        // Test matrix arithmetic
        let added_matrix = matrix1.clone() + matrix2;
        assert_eq!(added_matrix.row_size(), 20);
        assert_eq!(added_matrix.col_size(), 5);
        let mult_matrix = matrix1 * matrix3;
        assert_eq!(mult_matrix.row_size(), 20);
        assert_eq!(mult_matrix.col_size(), 12);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_gaussian_dist() {
        let params = DCRTPolyParams::default();

        // Test GaussianDist
        let sampler = DCRTPolyUniformSampler::new();
        let matrix1 =
            sampler.sample_uniform(&params, 20, 5, DistType::GaussDist { sigma: 4.57825 });
        assert_eq!(matrix1.row_size(), 20);
        assert_eq!(matrix1.col_size(), 5);

        let matrix2 =
            sampler.sample_uniform(&params, 20, 5, DistType::GaussDist { sigma: 4.57825 });

        let sampler2 = DCRTPolyUniformSampler::new();
        let matrix3 = sampler2.sample_uniform(&params, 5, 12, DistType::FinRingDist);
        assert_eq!(matrix3.row_size(), 5);
        assert_eq!(matrix3.col_size(), 12);

        // Test matrix arithmetic
        let added_matrix = matrix1.clone() + matrix2;
        assert_eq!(added_matrix.row_size(), 20);
        assert_eq!(added_matrix.col_size(), 5);
        let mult_matrix = matrix1 * matrix3;
        assert_eq!(mult_matrix.row_size(), 20);
        assert_eq!(mult_matrix.col_size(), 12);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_bit_dist() {
        let params = DCRTPolyParams::default();

        // Test BitDist
        let sampler = DCRTPolyUniformSampler::new();
        let matrix1 = sampler.sample_uniform(&params, 20, 5, DistType::BitDist);
        assert_eq!(matrix1.row_size(), 20);
        assert_eq!(matrix1.col_size(), 5);
        // [TODO] Test the norm of each coefficient of polynomials in the matrix.

        let matrix2 = sampler.sample_uniform(&params, 20, 5, DistType::BitDist);

        let sampler2 = DCRTPolyUniformSampler::new();
        let matrix3 = sampler2.sample_uniform(&params, 5, 12, DistType::FinRingDist);
        assert_eq!(matrix3.row_size(), 5);
        assert_eq!(matrix3.col_size(), 12);

        // Test matrix arithmetic
        let added_matrix = matrix1.clone() + matrix2;
        assert_eq!(added_matrix.row_size(), 20);
        assert_eq!(added_matrix.col_size(), 5);
        let mult_matrix = matrix1 * matrix3;
        assert_eq!(mult_matrix.row_size(), 20);
        assert_eq!(mult_matrix.col_size(), 12);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_matrix_mul_tensor_identity_simple() {
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();

        // Create matrix S (2x20)
        let s = sampler.sample_uniform(&params, 2, 20, DistType::FinRingDist);
        // Create 'other' matrix (5x7)
        let other = sampler.sample_uniform(&params, 5, 7, DistType::FinRingDist);
        // Perform S * (I_4 ⊗ other)
        let result = s.mul_tensor_identity(&other, 4);

        // Check dimensions
        assert_eq!(result.size().0, 2);
        assert_eq!(result.size().1, 28);

        let identity = DCRTPolyMatrix::identity(&params, 4, None);
        // Check result
        let expected_result = s * (identity.tensor(&other));

        assert_eq!(expected_result.size().0, 2);
        assert_eq!(expected_result.size().1, 28);
        assert_eq!(result, expected_result)
    }

    #[test]
    #[sequential_test::sequential]
    fn test_matrix_mul_tensor_identity_decompose_naive() {
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();

        // Create matrix S (2x2516)
        let s = sampler.sample_uniform(&params, 2, 2516, DistType::FinRingDist);

        // Create 'other' matrix (2x13)
        let other = sampler.sample_uniform(&params, 2, 13, DistType::FinRingDist);

        // Decompose 'other' matrix
        let other_decompose = other.decompose();
        // Perform S * (I_37 ⊗ G^-1(other))
        let result: DCRTPolyMatrix = s.mul_tensor_identity(&other_decompose, 37);
        // Check dimensions
        assert_eq!(result.size().0, 2);
        assert_eq!(result.size().1, 481);

        // Check result
        let tensor = identity_tensor_matrix(37, &other_decompose);
        let expected_result = s * tensor;

        assert_eq!(expected_result.size().0, 2);
        assert_eq!(expected_result.size().1, 481);
        assert_eq!(result, expected_result)
    }

    #[test]
    #[sequential_test::sequential]
    fn test_matrix_mul_tensor_identity_decompose_optimal() {
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();

        // Create matrix S (2x2516)
        let s = sampler.sample_uniform(&params, 2, 2516, DistType::FinRingDist);

        // Create 'other' matrix (2x13)
        let other = sampler.sample_uniform(&params, 2, 13, DistType::FinRingDist);

        // Perform S * (I_37 ⊗ G^-1(other))
        let result: DCRTPolyMatrix = s.mul_tensor_identity_decompose(&other, 37);

        // Check dimensions
        assert_eq!(result.size().0, 2);
        assert_eq!(result.size().1, 481);

        // Check result
        let decomposed = other.decompose();
        let tensor = identity_tensor_matrix(37, &decomposed);
        let expected_result_1 = s.clone() * tensor;
        let expected_result_2 = s.mul_tensor_identity(&decomposed, 37);
        assert_eq!(expected_result_1, expected_result_2);

        assert_eq!(expected_result_1.size().0, 2);
        assert_eq!(expected_result_1.size().1, 481);

        assert_eq!(expected_result_2.size().0, 2);
        assert_eq!(expected_result_2.size().1, 481);

        assert_eq!(result, expected_result_1);
        assert_eq!(result, expected_result_2);
    }

    fn identity_tensor_matrix(identity_size: usize, matrix: &DCRTPolyMatrix) -> DCRTPolyMatrix {
        let mut others = vec![];
        for _ in 1..identity_size {
            others.push(matrix);
        }
        matrix.concat_diag(&others[..])
    }

    #[test]
    #[sequential_test::sequential]
    fn test_matrix_compact_bytes() {
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();

        let dists = [DistType::BitDist, DistType::FinRingDist];
        for dist in dists {
            // todo: interesting finding. if its more square shape (e.g (50,50)more than 2m> (100,1)
            // - total 37s) slower
            let ncol = 15;
            let nrow = 15;

            // Create a random matrix
            let matrix = sampler.sample_uniform(&params, nrow, ncol, dist);

            // Convert to compact bytes
            let start_serialize = std::time::Instant::now();
            let compact_bytes = matrix.to_compact_bytes();
            let serialize_time = start_serialize.elapsed();
            println!(
                "to_compact_bytes took: {:?}, bytes_length={}",
                serialize_time,
                compact_bytes.len()
            );

            // Reconstruct from compact bytes
            let start_deserialize = std::time::Instant::now();
            let reconstructed_matrix = DCRTPolyMatrix::from_compact_bytes(&params, &compact_bytes);
            let deserialize_time = start_deserialize.elapsed();
            println!("from_compact_bytes took: {deserialize_time:?}");

            // Verify the matrices are equal
            assert_eq!(matrix, reconstructed_matrix);
        }
    }
}
