use super::{
    DCRTTrapdoor, KARNEY_THRESHOLD, sample_p1_for_pert_mat,
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
            poly::DCRTPoly,
        },
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, gpu_uniform::GpuDCRTPolyUniformSampler,
    },
};
use rayon::prelude::*;
#[cfg(test)]
use sequential_test::sequential;
use std::ops::Range;

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
        let p2_gpu = GpuDCRTPolyMatrix::from_cpu_matrix(params, &p2_cpu);

        let tp2_gpu = self.re.clone() * &p2_gpu;
        let tp2_cpu = gpu_matrix_to_cpu(&cpu_params, &tp2_gpu);

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
        let p1_gpu = GpuDCRTPolyMatrix::from_cpu_matrix(params, &p1_cpu);

        let mut p = p1_gpu.concat_rows(&[&p2_gpu]);
        if padding_ncol > 0 {
            p = p.slice_columns(0, total_ncol);
        }
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
        let d = public_matrix.row_size();
        let target_cols = target.col_size();
        assert_eq!(
            target.row_size(),
            d,
            "Target matrix should have the same number of rows as the public matrix"
        );

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

        let p_hat = trapdoor.sample_pert_square_mat(
            s,
            self.c,
            self.sigma,
            dgg_large_params,
            peikert,
            target_cols,
        );
        let perturbed_syndrome = target - &(public_matrix * &p_hat);

        let cpu_params = cpu_params_from_gpu(params);
        let cpu_perturbed = gpu_matrix_to_cpu(&cpu_params, &perturbed_syndrome);
        let mut decomposed_results = Vec::with_capacity(d * target_cols);
        for i in 0..d {
            for j in 0..target_cols {
                let cpu_poly = cpu_perturbed.entry(i, j);
                let decomposed =
                    decompose_dcrt_gadget(&cpu_poly, self.c, &cpu_params, self.base, self.sigma);
                decomposed_results.push((i, j, decomposed));
            }
        }

        let mut z_hat_mat = GpuDCRTPolyMatrix::zero(params, d * k, target_cols);
        for (i, j, decomposed) in decomposed_results {
            debug_assert_eq!(decomposed.len(), k);
            for (decomposed_idx, vec) in decomposed.iter().enumerate() {
                debug_assert_eq!(vec.len(), 1);
                let bytes = vec[0].to_compact_bytes();
                let gpu_poly = GpuDCRTPoly::from_compact_bytes(params, &bytes);
                z_hat_mat.set_entry(i * k + decomposed_idx, j, gpu_poly);
            }
        }

        let r_z_hat = &trapdoor.r * &z_hat_mat;
        let e_z_hat = &trapdoor.e * &z_hat_mat;
        let z_hat_former = (p_hat.slice_rows(0, d) + r_z_hat)
            .concat_rows(&[&(p_hat.slice_rows(d, 2 * d) + e_z_hat)]);
        let z_hat_latter = p_hat.slice_rows(2 * d, d * (k + 2)) + z_hat_mat;
        let result = z_hat_former.concat_rows(&[&z_hat_latter]);

        if std::env::var("DEBUG_GPU_PREIMAGE_CPU_CHECK").is_ok() {
            let cpu_params = cpu_params_from_gpu(params);
            let public_cpu = gpu_matrix_to_cpu(&cpu_params, public_matrix);
            let target_cpu = gpu_matrix_to_cpu(&cpu_params, target);
            let preimage_cpu = gpu_matrix_to_cpu(&cpu_params, &result);
            let product_cpu = &public_cpu * &preimage_cpu;
            if product_cpu != target_cpu {
                eprintln!("CPU product mismatch in GPU preimage debug");
            } else {
                eprintln!("CPU product matches target in GPU preimage debug");
            }

            let p_hat_cpu = gpu_matrix_to_cpu(&cpu_params, &p_hat);
            let perturbed_expected = &target_cpu - &(&public_cpu * &p_hat_cpu);
            if cpu_perturbed != perturbed_expected {
                eprintln!("CPU perturbed_syndrome mismatch in GPU preimage debug");
            } else {
                eprintln!("CPU perturbed_syndrome matches in GPU preimage debug");
            }

            let product_gpu = public_matrix * &result;
            let product_gpu_cpu = gpu_matrix_to_cpu(&cpu_params, &product_gpu);
            if product_gpu_cpu != target_cpu {
                eprintln!("GPU product mismatch in GPU preimage debug");
            } else {
                eprintln!("GPU product matches target in GPU preimage debug");
            }
        }

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
        preimage_left.concat_rows(&[&preimage_right])
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
    use crate::{__PAIR, __TestState, poly::dcrt::gpu::gpu_device_sync};

    fn cpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 17, 1)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _crt_bits, _crt_depth) = params.to_crt();
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
}
