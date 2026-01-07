use super::{
    KARNEY_THRESHOLD, sample_p1_for_pert_mat,
    sampler::decompose_dcrt_gadget,
    utils::{gen_dgg_int_vec, gen_int_karney, split_int64_mat_to_elems},
};
use crate::{
    matrix::{
        PolyMatrix,
        dcrt_poly::DCRTPolyMatrix,
        gpu_dcrt_poly::GpuDCRTPolyMatrix,
        i64::{I64Matrix, I64MatrixParams},
    },
    parallel_iter,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
            params::DCRTPolyParams,
        },
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, gpu_uniform::GpuDCRTPolyUniformSampler,
    },
    utils::log_mem,
};
use rayon::prelude::*;
#[cfg(test)]
use sequential_test::sequential;
use std::{ops::Range, time::Instant};

const SIGMA: f64 = 4.578;
const SPECTRAL_CONSTANT: f64 = 1.8;

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
        // log_mem("start new GpuDCRTTrapdoor");
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let log_base_q = params.modulus_digits();
        let dist = DistType::GaussDist { sigma };
        let r = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        // log_mem("sample r");
        let e = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        // log_mem("sample e");
        let a_mat = r.clone() * r.transpose();
        // log_mem("compute a_mat");
        let e_transpose = e.transpose();
        let b_mat = r.clone() * &e_transpose;
        // log_mem("compute b_mat");
        let d_mat = e.clone() * &e_transpose;
        // log_mem("compute d_mat");
        let re = r.concat_rows(&[&e]);
        // log_mem("compute re");
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
        let overall_start = Instant::now();
        let r = &self.r;
        let params = &r.params;
        let cpu_params = cpu_params_from_gpu(params);
        let n = params.ring_dimension() as usize;
        let (d, dk) = r.size();
        let sigma_large = dgg_large_params.1;
        let num_blocks = total_ncol.div_ceil(d);
        let padded_ncol = num_blocks * d;
        let padding_ncol = padded_ncol - total_ncol;

        let p2z_vec = if sigma_large > KARNEY_THRESHOLD {
            let mut matrix = I64Matrix::new_empty(&I64MatrixParams, n * dk, padded_ncol);
            let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<i64>> {
                parallel_iter!(row_offsets)
                    .map(|_| {
                        parallel_iter!(col_offsets.clone())
                            .map(|_| gen_int_karney(0.0, sigma_large))
                            .collect()
                    })
                    .collect()
            };
            matrix.replace_entries(0..n * dk, 0..padded_ncol, f);
            matrix
        } else {
            let dgg_vectors = gen_dgg_int_vec(
                n * dk * padded_ncol,
                peikert,
                dgg_large_params.0.expect("Missing mean for Peikert sampler"),
                dgg_large_params.1,
                dgg_large_params.2.expect("Missing table for Peikert sampler"),
            );
            let vecs = parallel_iter!(0..n * dk)
                .map(|i| {
                    dgg_vectors.slice(i * padded_ncol, (i + 1) * padded_ncol, 0, 1).transpose()
                })
                .collect::<Vec<_>>();
            vecs[0].concat_rows(&vecs[1..].iter().collect::<Vec<_>>())
        };

        let p2_cpu = split_int64_mat_to_elems(&p2z_vec, &cpu_params);
        let p2_gpu_start = Instant::now();
        let p2_gpu = GpuDCRTPolyMatrix::from_cpu_matrix(params, &p2_cpu);
        log_mem(format!("sample_pert_square_mat p2_gpu in {:?}", p2_gpu_start.elapsed()));

        let tp2_gpu_start = Instant::now();
        let tp2_gpu = self.re.clone() * &p2_gpu;
        log_mem(format!("sample_pert_square_mat tp2_gpu mul in {:?}", tp2_gpu_start.elapsed()));
        let tp2_cpu_start = Instant::now();
        let tp2_cpu = gpu_matrix_to_cpu(&cpu_params, &tp2_gpu);
        log_mem(format!("sample_pert_square_mat tp2_cpu in {:?}", tp2_cpu_start.elapsed()));

        let a_mat_cpu = gpu_matrix_to_cpu(&cpu_params, &self.a_mat);
        let b_mat_cpu = gpu_matrix_to_cpu(&cpu_params, &self.b_mat);
        let d_mat_cpu = gpu_matrix_to_cpu(&cpu_params, &self.d_mat);
        let p1_cpu = sample_p1_for_pert_mat(
            a_mat_cpu,
            b_mat_cpu,
            d_mat_cpu,
            tp2_cpu,
            &cpu_params,
            c,
            s,
            dgg,
            padded_ncol,
        );
        let p1_gpu_start = Instant::now();
        let p1_gpu = GpuDCRTPolyMatrix::from_cpu_matrix(params, &p1_cpu);
        log_mem(format!("sample_pert_square_mat p1_gpu in {:?}", p1_gpu_start.elapsed()));

        let concat_start = Instant::now();
        let mut p = p1_gpu.concat_rows_owned(vec![p2_gpu]);
        log_mem(format!("sample_pert_square_mat concat in {:?}", concat_start.elapsed()));
        if padding_ncol > 0 {
            let slice_start = Instant::now();
            p = p.slice_columns(0, total_ncol);
            log_mem(format!("sample_pert_square_mat slice in {:?}", slice_start.elapsed()));
        }
        log_mem(format!("sample_pert_square_mat total in {:?}", overall_start.elapsed()));
        p
    }

    // fn to_cpu(&self, params: &DCRTPolyParams) -> DCRTTrapdoor {
    //     DCRTTrapdoor {
    //         r: gpu_matrix_to_cpu(params, &self.r),
    //         e: gpu_matrix_to_cpu(params, &self.e),
    //         a_mat: gpu_matrix_to_cpu(params, &self.a_mat),
    //         b_mat: gpu_matrix_to_cpu(params, &self.b_mat),
    //         d_mat: gpu_matrix_to_cpu(params, &self.d_mat),
    //         re: gpu_matrix_to_cpu(params, &self.re),
    //     }
    // }
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
        // log_mem("trapdoor sampled");
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let a_bar = uniform_sampler.sample_uniform(params, size, size, DistType::FinRingDist);
        // log_mem("a_bar sampled");
        let g = GpuDCRTPolyMatrix::gadget_matrix(params, size);
        // log_mem("gadget matrix computed");
        let a0 = a_bar.concat_columns(&[&GpuDCRTPolyMatrix::identity(params, size, None)]);
        // log_mem("a0 computed");
        let a1 = g - (a_bar * &trapdoor.r + &trapdoor.e);
        // log_mem("a1 computed");
        let a = a0.concat_columns_owned(vec![a1]);
        // log_mem("a computed");
        (trapdoor, a)
    }

    fn preimage(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M {
        let overall_start = Instant::now();
        let d = public_matrix.row_size();
        let target_cols = target.col_size();
        log_mem(format!(
            "preimage start d={}, target_cols={}, n={}",
            d,
            target_cols,
            params.ring_dimension()
        ));
        assert_eq!(
            target.row_size(),
            d,
            "Target matrix should have the same number of rows as the public matrix"
        );

        let dgg_start = Instant::now();
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let s = SPECTRAL_CONSTANT *
            (self.base as f64 + 1.0) *
            SIGMA *
            SIGMA *
            (((d * n * k) as f64).sqrt() + ((2 * n) as f64).sqrt() + 4.7);
        let dgg_large_std = (s * s - self.c * self.c).sqrt();
        let peikert = dgg_large_std < KARNEY_THRESHOLD;
        let (dgg_large_mean, dgg_large_table) = if dgg_large_std > KARNEY_THRESHOLD {
            (None, None)
        } else {
            let acc: f64 = 5e-32;
            let m = (-2.0 * acc.ln()).sqrt();
            let fin = (dgg_large_std * m).ceil() as usize;

            let mut m_vals = Vec::with_capacity(fin);
            let variance = 2.0 * dgg_large_std * dgg_large_std;
            let mut cusum = 0.0f64;
            for i in 1..=fin {
                cusum += (-(i as f64 * i as f64) / variance).exp();
                m_vals.push(cusum);
            }
            let m_a = 1.0 / (2.0 * cusum + 1.0);
            for i in 0..fin {
                m_vals[i] *= m_a;
            }
            (Some(m_a), Some(m_vals))
        };
        let dgg_large_params =
            (dgg_large_mean, dgg_large_std, dgg_large_table.as_ref().map(|v| &v[..]));
        log_mem(format!("preimage dgg params in {:?}", dgg_start.elapsed()));

        let p_hat_start = Instant::now();
        let p_hat = trapdoor.sample_pert_square_mat(
            s,
            self.c,
            self.sigma,
            dgg_large_params,
            peikert,
            target_cols,
        );
        log_mem(format!("preimage p_hat in {:?}", p_hat_start.elapsed()));
        let perturbed_start = Instant::now();
        let perturbed_syndrome = target - &(public_matrix * &p_hat);
        log_mem(format!("preimage perturbed_syndrome in {:?}", perturbed_start.elapsed()));

        let decomp_start = Instant::now();
        let cpu_params = cpu_params_from_gpu(params);
        let cpu_perturbed = gpu_matrix_to_cpu(&cpu_params, &perturbed_syndrome);
        let decomposed_rows = parallel_iter!(0..d)
            .map(|i| {
                parallel_iter!(0..target_cols)
                    .map(|j| {
                        let cpu_poly = cpu_perturbed.entry(i, j);
                        let decomposed = decompose_dcrt_gadget(
                            &cpu_poly,
                            self.c,
                            &cpu_params,
                            self.base,
                            self.sigma,
                        );
                        (i, j, decomposed)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let decomposed_results = decomposed_rows.into_iter().flatten().collect::<Vec<_>>();
        log_mem(format!("preimage decomposition in {:?}", decomp_start.elapsed()));

        let zhat_start = Instant::now();
        let z_hat_cpu_start = Instant::now();
        let mut z_hat_cpu = DCRTPolyMatrix::zero(&cpu_params, d * k, target_cols);
        for (i, j, decomposed) in decomposed_results {
            debug_assert_eq!(decomposed.len(), k);
            for (decomposed_idx, mut vec) in decomposed.into_iter().enumerate() {
                debug_assert_eq!(vec.len(), 1);
                let poly = vec.pop().expect("decomposed entry must contain one poly");
                z_hat_cpu.set_entry(i * k + decomposed_idx, j, poly);
            }
        }
        log_mem(format!(
            "preimage z_hat_mat cpu build in {:?}",
            z_hat_cpu_start.elapsed()
        ));
        let z_hat_mat_start = Instant::now();
        let z_hat_mat = GpuDCRTPolyMatrix::from_cpu_matrix(params, &z_hat_cpu);
        log_mem(format!("preimage z_hat_mat construction in {:?}", z_hat_mat_start.elapsed()));
        log_mem(format!("preimage z_hat_mat in {:?}", zhat_start.elapsed()));

        let r_e_z_hat_start = Instant::now();
        let r_z_hat = &trapdoor.r * &z_hat_mat;
        let e_z_hat = &trapdoor.e * &z_hat_mat;
        log_mem(format!("preimage r_z_hat and e_z_hat in {:?}", r_e_z_hat_start.elapsed()));
        let combine_start = Instant::now();
        let r_e_z_hat = r_z_hat.concat_rows_owned(vec![e_z_hat, z_hat_mat]);
        log_mem(format!("preimage concat r_e_z_hat in {:?}", combine_start.elapsed()));
        let result = p_hat + r_e_z_hat;
        log_mem(format!("preimage total in {:?}", overall_start.elapsed()));
        result
    }

    fn preimage_extend(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        ext_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M {
        let d = public_matrix.row_size();
        let ext_ncol = ext_matrix.col_size();
        let target_ncol = target.col_size();
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let s = SPECTRAL_CONSTANT *
            (self.base as f64 + 1.0) *
            SIGMA *
            SIGMA *
            (((d * n * k) as f64).sqrt() + ((2 * n) as f64).sqrt() + 4.7);
        let dist = DistType::GaussDist { sigma: s };
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let preimage_right = uniform_sampler.sample_uniform(params, ext_ncol, target_ncol, dist);
        let t = target - &(ext_matrix * &preimage_right);
        let preimage_left = self.preimage(params, trapdoor, public_matrix, &t);
        preimage_left.concat_rows_owned(vec![preimage_right])
    }
}

fn cpu_params_from_gpu(params: &GpuDCRTPolyParams) -> DCRTPolyParams {
    let cpu_params = DCRTPolyParams::new(
        params.ring_dimension(),
        params.crt_depth(),
        params.crt_bits(),
        params.base_bits(),
    );
    assert_eq!(cpu_params.to_crt().0.as_slice(), params.moduli());
    cpu_params
}

fn gpu_matrix_to_cpu(params: &DCRTPolyParams, matrix: &GpuDCRTPolyMatrix) -> DCRTPolyMatrix {
    let cpu_matrix = matrix.to_cpu_matrix();
    debug_assert_eq!(cpu_matrix.params, *params, "CPU params mismatch in gpu_matrix_to_cpu");
    cpu_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{__PAIR, __TestState, poly::dcrt::gpu::gpu_device_sync};

    fn cpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 17, 1)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let _ = tracing_subscriber::fmt::try_init();
        let (moduli, _, _) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    #[sequential]
    fn test_gpu_trapdoor_generation() {
        gpu_device_sync();
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

        let k = params.modulus_digits();
        let identity = GpuDCRTPolyMatrix::identity(&params, size * k, None);
        log_mem("identity sampled");
        let trapdoor_matrix = trapdoor.r.concat_rows(&[&trapdoor.e, &identity]);
        log_mem("trapdoor_matrix sampled");
        let muled = &public_matrix * &trapdoor_matrix;
        log_mem("muled computed");
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, size);
        log_mem("gadget_matrix computed");
        assert_eq!(muled, gadget_matrix, "Product should equal gadget matrix");
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_generation_square() {
        gpu_device_sync();
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
    #[sequential]
    fn test_gpu_preimage_generation_non_square_target_lt() {
        gpu_device_sync();
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
    #[sequential]
    fn test_gpu_preimage_generation_non_square_target_gt_multiple() {
        gpu_device_sync();
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
    #[sequential]
    fn test_gpu_preimage_generation_non_square_target_gt_non_multiple() {
        gpu_device_sync();
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
    #[sequential]
    fn test_gpu_preimage_generation_base_8() {
        gpu_device_sync();
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
    #[sequential]
    fn test_gpu_preimage_generation_base_1024() {
        gpu_device_sync();
        let cpu_params = DCRTPolyParams::new(128, 2, 17, 10);
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
    #[sequential]
    fn test_gpu_preimage_generation_extend() {
        gpu_device_sync();
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

    #[test]
    #[sequential]
    fn test_gpu_preimage_generation_large() {
        gpu_device_sync();
        let _ = tracing_subscriber::fmt::try_init();
        let cpu_params = DCRTPolyParams::new(1024, 15, 24, 19);
        let params = gpu_params_from_cpu(&cpu_params);
        let size = 2;
        let k = params.modulus_digits();
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, 1000, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);

        let expected_rows = size * (k + 2);
        let expected_cols = 1000;

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
}
