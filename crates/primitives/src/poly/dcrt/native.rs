#[cxx::bridge(namespace = "mxx")]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("ExactBasis.h");
        fn exact_basis_validate(dimension: u32, moduli: &[u64]) -> Result<()>;
        #[namespace = "openfhe"]
        type DCRTPoly = openfhe::ffi::DCRTPoly;
        fn exact_basis_bgv_mod_reduce(
            input: &DCRTPoly,
            plaintext_modulus: u64,
        ) -> Result<UniquePtr<DCRTPoly>>;
        fn exact_basis_convert(
            input: &DCRTPoly,
            moduli: &[u64],
            centered: bool,
        ) -> Result<UniquePtr<DCRTPoly>>;
        fn exact_basis_coefficients(input: &DCRTPoly) -> Result<Vec<u8>>;
        #[namespace = "openfhe"]
        type Matrix = openfhe::ffi::Matrix;
        fn exact_basis_matrix_coefficients(matrix: Pin<&mut Matrix>) -> Result<()>;
        fn exact_basis_p1(
            a: &Matrix,
            b: &Matrix,
            d: &Matrix,
            tp2: &Matrix,
            columns: usize,
            sigma: f64,
            s: f64,
            dgg_sigma: f64,
        ) -> Result<UniquePtr<Matrix>>;
        fn exact_basis_matrix(
            dimension: u32,
            moduli: &[u64],
            rows: usize,
            columns: usize,
            gadget_base: u64,
        ) -> Result<UniquePtr<Matrix>>;
        fn exact_basis_sample(
            dimension: u32,
            moduli: &[u64],
            distribution: u32,
            sigma: f64,
        ) -> Result<UniquePtr<DCRTPoly>>;
        fn exact_basis_gauss_gq(
            syndrome: &DCRTPoly,
            c: f64,
            digits_count: usize,
            base: i64,
            sigma: f64,
            tower: usize,
        ) -> Result<Vec<i64>>;
        fn exact_basis_poly(
            dimension: u32,
            moduli: &[u64],
            values: &[u64],
            limbs_per_integer: usize,
            evaluation: bool,
        ) -> Result<UniquePtr<DCRTPoly>>;
    }
}
