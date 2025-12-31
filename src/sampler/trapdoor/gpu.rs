use super::{DCRTTrapdoor, sampler::DCRTPolyTrapdoorSampler};
use crate::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, gpu_uniform::GpuDCRTPolyUniformSampler,
    },
};

const SIGMA: f64 = 4.578;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDCRTTrapdoor {
    pub r: GpuDCRTPolyMatrix,
    pub e: GpuDCRTPolyMatrix,
    pub a_mat: GpuDCRTPolyMatrix,
    pub b_mat: GpuDCRTPolyMatrix,
    pub d_mat: GpuDCRTPolyMatrix,
    pub re: GpuDCRTPolyMatrix,
}

impl GpuDCRTTrapdoor {
    pub fn new(params: &GpuDCRTPolyParams, size: usize, sigma: f64) -> Self {
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let log_base_q = params.modulus_digits();
        let dist = DistType::GaussDist { sigma };
        let r = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        let e = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        let a_mat = r.clone() * r.transpose();
        let e_transpose = e.transpose();
        let b_mat = r.clone() * &e_transpose;
        let d_mat = e.clone() * &e_transpose;
        let re = r.concat_rows(&[&e]);
        Self { r, e, a_mat, b_mat, d_mat, re }
    }

    pub fn sample_pert_square_mat(
        &self,
        s: f64,
        c: f64,
        dgg: f64,
        dgg_large_params: (Option<f64>, f64, Option<&[f64]>),
        peikert: bool,
        total_ncol: usize,
    ) -> GpuDCRTPolyMatrix {
        let cpu_params = cpu_params_from_gpu(&self.r.params);
        let cpu_trapdoor = self.to_cpu(&cpu_params);
        let cpu_matrix = cpu_trapdoor.sample_pert_square_mat(
            s,
            c,
            dgg,
            dgg_large_params,
            peikert,
            total_ncol,
        );
        GpuDCRTPolyMatrix::from_cpu_matrix(&self.r.params, &cpu_matrix)
    }

    fn to_cpu(&self, params: &DCRTPolyParams) -> DCRTTrapdoor {
        DCRTTrapdoor {
            r: gpu_matrix_to_cpu(params, &self.r),
            e: gpu_matrix_to_cpu(params, &self.e),
            a_mat: gpu_matrix_to_cpu(params, &self.a_mat),
            b_mat: gpu_matrix_to_cpu(params, &self.b_mat),
            d_mat: gpu_matrix_to_cpu(params, &self.d_mat),
            re: gpu_matrix_to_cpu(params, &self.re),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuDCRTPolyTrapdoorSampler {
    sigma: f64,
    base: u32,
    c: f64,
}

impl PolyTrapdoorSampler for GpuDCRTPolyTrapdoorSampler {
    type M = GpuDCRTPolyMatrix;
    type Trapdoor = GpuDCRTTrapdoor;

    fn new(params: &<<Self::M as PolyMatrix>::P as Poly>::Params, sigma: f64) -> Self {
        let base = 1 << params.base_bits();
        let c = (base as f64 + 1.0) * SIGMA;
        Self { sigma, base, c }
    }

    fn trapdoor(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        size: usize,
    ) -> (Self::Trapdoor, Self::M) {
        let trapdoor = GpuDCRTTrapdoor::new(params, size, self.sigma);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let a_bar = uniform_sampler.sample_uniform(params, size, size, DistType::FinRingDist);
        let g = GpuDCRTPolyMatrix::gadget_matrix(params, size);
        let a0 = a_bar.concat_columns(&[&GpuDCRTPolyMatrix::identity(params, size, None)]);
        let a1 = g - (a_bar * &trapdoor.r + &trapdoor.e);
        let a = a0.concat_columns(&[&a1]);
        (trapdoor, a)
    }

    fn preimage(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M {
        let cpu_params = cpu_params_from_gpu(params);
        let cpu_sampler = DCRTPolyTrapdoorSampler::new(&cpu_params, self.sigma);
        let cpu_trapdoor = trapdoor.to_cpu(&cpu_params);
        let cpu_public = gpu_matrix_to_cpu(&cpu_params, public_matrix);
        let cpu_target = gpu_matrix_to_cpu(&cpu_params, target);
        let cpu_result = cpu_sampler.preimage(&cpu_params, &cpu_trapdoor, &cpu_public, &cpu_target);
        GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu_result)
    }

    fn preimage_extend(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        ext_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M {
        let cpu_params = cpu_params_from_gpu(params);
        let cpu_sampler = DCRTPolyTrapdoorSampler::new(&cpu_params, self.sigma);
        let cpu_trapdoor = trapdoor.to_cpu(&cpu_params);
        let cpu_public = gpu_matrix_to_cpu(&cpu_params, public_matrix);
        let cpu_ext = gpu_matrix_to_cpu(&cpu_params, ext_matrix);
        let cpu_target = gpu_matrix_to_cpu(&cpu_params, target);
        let cpu_result = cpu_sampler.preimage_extend(
            &cpu_params,
            &cpu_trapdoor,
            &cpu_public,
            &cpu_ext,
            &cpu_target,
        );
        GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu_result)
    }
}

fn cpu_params_from_gpu(params: &GpuDCRTPolyParams) -> DCRTPolyParams {
    let cpu_params = DCRTPolyParams::new(
        params.ring_dimension(),
        params.crt_depth(),
        params.crt_bits(),
        params.base_bits(),
    );
    debug_assert_eq!(cpu_params.to_crt().0.as_slice(), params.moduli());
    cpu_params
}

fn gpu_matrix_to_cpu(params: &DCRTPolyParams, matrix: &GpuDCRTPolyMatrix) -> DCRTPolyMatrix {
    DCRTPolyMatrix::from_compact_bytes(params, &matrix.to_compact_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dcrt::gpu::gpu_test_lock;

    fn cpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 17, 1)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _crt_bits, _crt_depth) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    fn test_gpu_trapdoor_generation() {
        let _guard = gpu_test_lock();
        let size: usize = 3;
        let cpu_params = cpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);

        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let expected_rows = size;
        let expected_cols = (params.modulus_digits() + 2) * size;

        assert_eq!(
            public_matrix.row_size(),
            expected_rows,
            "Public matrix should have the correct number of rows"
        );
        assert_eq!(
            public_matrix.col_size(),
            expected_cols,
            "Public matrix should have the correct number of columns"
        );

        let muled = {
            let k = params.modulus_digits();
            let identity = GpuDCRTPolyMatrix::identity(&params, size * k, None);
            let trapdoor_matrix = trapdoor.r.concat_rows(&[&trapdoor.e, &identity]);
            public_matrix * trapdoor_matrix
        };
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, size);
        assert_eq!(muled, gadget_matrix);
    }

    #[test]
    fn test_gpu_preimage_generation_square() {
        let _guard = gpu_test_lock();
        let cpu_params = cpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 3;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);

        let expected_rows = size * (k + 2);
        let expected_cols = size;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );
        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns"
        );

        let product = public_matrix * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    fn test_gpu_preimage_generation_non_square_target_lt() {
        let _guard = gpu_test_lock();
        let cpu_params = cpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 4;
        let target_cols = 2;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );
        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (sliced to match target)"
        );

        let product = public_matrix * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    fn test_gpu_preimage_generation_non_square_target_gt_multiple() {
        let _guard = gpu_test_lock();
        let cpu_params = cpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 4;
        let multiple = 2;
        let target_cols = size * multiple;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );
        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        let product = public_matrix * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    fn test_gpu_preimage_generation_non_square_target_gt_non_multiple() {
        let _guard = gpu_test_lock();
        let cpu_params = cpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 4;
        let target_cols = 6;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );
        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        let product = public_matrix * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    fn test_gpu_preimage_generation_base_8() {
        let _guard = gpu_test_lock();
        let cpu_params = cpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 4;
        let target_cols = 6;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );
        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        let product = public_matrix * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    fn test_gpu_preimage_generation_base_1024() {
        let _guard = gpu_test_lock();
        let cpu_params =  DCRTPolyParams::new(128, 2, 17, 10);
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 4;
        let target_cols = 6;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target =
            uniform_sampler.sample_uniform(&params, size, target_cols, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);

        let expected_rows = size * (k + 2);
        let expected_cols = target_cols;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );
        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns (equal to target columns)"
        );

        let product = public_matrix * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }

    #[test]
    fn test_gpu_preimage_generation_extend() {
        let _guard = gpu_test_lock();
        let cpu_params = cpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 3;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, 1, DistType::FinRingDist);
        let m = size * params.modulus_digits();
        let extend = uniform_sampler.sample_uniform(&params, size, m, DistType::FinRingDist);

        let preimage =
            trapdoor_sampler.preimage_extend(&params, &trapdoor, &public_matrix, &extend, &target);

        let expected_rows = size * (k + 2) + m;
        let expected_cols = 1;

        assert_eq!(
            preimage.row_size(),
            expected_rows,
            "Preimage matrix should have the correct number of rows"
        );
        assert_eq!(
            preimage.col_size(),
            expected_cols,
            "Preimage matrix should have the correct number of columns"
        );

        let product = public_matrix.concat_columns(&[&extend]) * &preimage;
        assert_eq!(product, target, "Product of public matrix and preimage should equal target");
    }
}
