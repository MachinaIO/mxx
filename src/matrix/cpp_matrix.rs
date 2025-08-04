use openfhe::{
    cxx::UniquePtr,
    ffi::{GetMatrixCols, GetMatrixElement, GetMatrixRows, Matrix},
};

use crate::poly::{
    Poly,
    dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
};

pub(crate) struct CppMatrix {
    pub(crate) params: DCRTPolyParams,
    pub(crate) inner: UniquePtr<Matrix>,
}

unsafe impl Send for CppMatrix {}
unsafe impl Sync for CppMatrix {}

impl CppMatrix {
    pub fn new(params: &DCRTPolyParams, inner: UniquePtr<Matrix>) -> Self {
        CppMatrix {
            params: params.clone(),
            inner,
        }
    }

    pub(crate) fn nrow(&self) -> usize {
        GetMatrixRows(&self.inner)
    }

    pub(crate) fn ncol(&self) -> usize {
        GetMatrixCols(&self.inner)
    }

    pub(crate) fn entry(&self, i: usize, j: usize) -> DCRTPoly {
        let poly = DCRTPoly::new(GetMatrixElement(&self.inner, i, j));
        // This ensures that coefficients are rounded to Z_q.
        DCRTPoly::from_coeffs(&self.params, &poly.coeffs())
    }
}
