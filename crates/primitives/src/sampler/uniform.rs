use crate::{
    element::PolyElem,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    openfhe_guard::ensure_openfhe_warmup,
    parallel_iter,
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
    sampler::{DistType, PolyUniformSampler, bounds::centered_coefficient_abs},
};
use num_bigint::BigUint;
use openfhe::ffi;
use rayon::prelude::*;

use crate::poly::dcrt::params::DCRTPolyParams;

fn rejection_resample_coefficients<T>(
    count: usize,
    mut candidate: impl FnMut() -> Vec<T>,
    mut accepts: impl FnMut(&T) -> bool,
) -> Vec<T> {
    let mut accepted = (0..count).map(|_| None).collect::<Vec<_>>();
    while accepted.iter().any(Option::is_none) {
        for (slot, coefficient) in accepted.iter_mut().zip(candidate()) {
            if slot.is_none() && accepts(&coefficient) {
                *slot = Some(coefficient);
            }
        }
    }
    accepted
        .into_iter()
        .map(|coefficient| coefficient.expect("all coefficients accepted"))
        .collect()
}

pub struct DCRTPolyUniformSampler {}

impl DCRTPolyUniformSampler {
    fn sample_gaussian_unchecked(&self, params: &DCRTPolyParams, sigma: f64) -> DCRTPoly {
        let sampled_poly = ffi::DCRTPolyGenFromDgg(
            params.ring_dimension(),
            params.crt_depth(),
            params.crt_bits(),
            sigma,
        );
        if sampled_poly.is_null() {
            panic!("Attempted to dereference a null pointer");
        }
        DCRTPoly::new(sampled_poly)
    }

    fn sample_truncated_gaussian_poly(
        &self,
        params: &DCRTPolyParams,
        sigma: f64,
        max_coefficient_bound: &BigUint,
    ) -> DCRTPoly {
        let modulus = params.modulus();
        let coefficients = rejection_resample_coefficients(
            params.ring_dimension() as usize,
            || self.sample_gaussian_unchecked(params, sigma).coeffs(),
            |coefficient| {
                centered_coefficient_abs(coefficient.value(), modulus.as_ref()) <=
                    *max_coefficient_bound
            },
        );
        DCRTPoly::from_coeffs(params, &coefficients)
    }

    fn sample_poly_unchecked(&self, params: &DCRTPolyParams, dist: &DistType) -> DCRTPoly {
        let sampled_poly = match dist {
            DistType::FinRingDist => ffi::DCRTPolyGenFromDug(
                params.ring_dimension(),
                params.crt_depth(),
                params.crt_bits(),
            ),
            DistType::GaussDist { sigma, max_coefficient_bound: Some(bound) } => {
                return self.sample_truncated_gaussian_poly(params, *sigma, bound);
            }
            DistType::GaussDist { sigma, max_coefficient_bound: None } => {
                return self.sample_gaussian_unchecked(params, *sigma);
            }
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

        let c: Vec<Vec<DCRTPoly>> = parallel_iter!(0..nrow)
            .map(|_| {
                parallel_iter!(0..ncol).map(|_| self.sample_poly_unchecked(params, &dist)).collect()
            })
            .collect();

        DCRTPolyMatrix::from_poly_vec(params, c)
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use crate::poly::dcrt::params::DCRTPolyParams;

    use super::*;

    #[test]
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
    fn truncated_gaussian_never_exceeds_the_integer_cutoff() {
        let params = DCRTPolyParams::new(32, 1, 20, 4);
        let cutoff = BigUint::from(2u8);
        let matrix = DCRTPolyUniformSampler::new().sample_uniform(
            &params,
            3,
            2,
            DistType::GaussDist { sigma: 3.0, max_coefficient_bound: Some(cutoff.clone()) },
        );
        assert!(crate::sampler::bounds::matrix_within_coefficient_bound(&matrix, &cutoff));
    }

    #[test]
    fn truncated_coefficients_are_resampled_instead_of_clipped() {
        let candidates = [vec![99, 2, 99], vec![1, 99, 3]];
        let mut next = 0;
        let accepted = rejection_resample_coefficients(
            3,
            || {
                let candidate = candidates[next].clone();
                next += 1;
                candidate
            },
            |coefficient| *coefficient <= 3,
        );
        assert_eq!(accepted, vec![1, 2, 3]);
        assert_eq!(next, 2);
    }

    #[test]
    fn test_gaussian_dist() {
        let params = DCRTPolyParams::default();

        // Test GaussianDist
        let sampler = DCRTPolyUniformSampler::new();
        let matrix1 = sampler.sample_uniform(
            &params,
            20,
            5,
            DistType::GaussDist { sigma: 4.57825, max_coefficient_bound: None },
        );
        assert_eq!(matrix1.row_size(), 20);
        assert_eq!(matrix1.col_size(), 5);

        let matrix2 = sampler.sample_uniform(
            &params,
            20,
            5,
            DistType::GaussDist { sigma: 4.57825, max_coefficient_bound: None },
        );

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
