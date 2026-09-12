use openfhe::{
    cxx::UniquePtr,
    ffi::{GetMatrixCols, GetMatrixRows, Matrix},
};

use crate::poly::dcrt::poly::DCRTPoly;

pub(crate) struct CppMatrix {
    pub(crate) inner: UniquePtr<Matrix>,
}

unsafe impl Send for CppMatrix {}
unsafe impl Sync for CppMatrix {}

impl CppMatrix {
    pub fn new(inner: UniquePtr<Matrix>) -> Self {
        CppMatrix { inner }
    }

    pub(crate) fn nrow(&self) -> usize {
        GetMatrixRows(&self.inner)
    }

    pub(crate) fn ncol(&self) -> usize {
        GetMatrixCols(&self.inner)
    }

    pub(crate) fn entry(&self, i: usize, j: usize) -> DCRTPoly {
        DCRTPoly::new(
            crate::poly::dcrt::native::ffi::exact_basis_matrix_entry(&self.inner, i, j)
                .expect("exact CRT matrix entry copy failed"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        Poly, PolyParams,
        dcrt::{native::ffi as exact, params::DCRTPolyParams},
    };

    #[test]
    fn test_cpp_matrix_entry_preserves_basis_and_format() {
        let (dimension, depth, bits, base) = crate::env::modulus_conversion_test_parameters();
        let generated = DCRTPolyParams::new(dimension, depth, bits, base, None, None);
        let mut primes = generated.to_crt().0;
        primes.reverse();
        let params = DCRTPolyParams::new(dimension, depth, bits, base, Some(primes.clone()), None);
        let original =
            DCRTPoly::new(exact::exact_basis_sample(dimension, &primes, 0, 0.0).unwrap());
        let mut matrix =
            CppMatrix::new(exact::exact_basis_matrix(dimension, &primes, 1, 1, 0).unwrap());
        openfhe::ffi::SetMatrixElement(matrix.inner.pin_mut(), 0, 0, original.get_poly());
        assert_eq!(matrix.entry(0, 0), original);
        exact::exact_basis_matrix_coefficients(matrix.inner.pin_mut()).unwrap();
        let coefficients = matrix.entry(0, 0);
        // OpenFHE equality includes the representation format, so this also
        // checks that coefficient-format entries are not silently transformed.
        assert_ne!(coefficients, original);
        assert_eq!(coefficients.coeffs_biguints(), original.coeffs_biguints());
        assert_eq!(DCRTPoly::from_biguints(&params, &coefficients.coeffs_biguints()), original);
    }
}
