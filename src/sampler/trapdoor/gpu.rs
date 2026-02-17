use crate::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{Poly, PolyParams, dcrt::gpu::GpuDCRTPolyParams},
    sampler::{DistType, PolyTrapdoorSampler, PolyUniformSampler, gpu::GpuDCRTPolyUniformSampler},
};
use rand::{Rng, rng};
use std::time::Instant;

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
        let a_mat = &r * &r.transpose(); // d x d
        let b_mat = &r * &e.transpose(); // d x d
        let d_mat = &e * &e.transpose(); // d x d
        let re = r.concat_rows(&[&e]);
        Self { r, e, a_mat, b_mat, d_mat, re }
    }

    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mats = [&self.r, &self.e, &self.a_mat, &self.b_mat, &self.d_mat, &self.re];
        let mut parts = Vec::with_capacity(mats.len());
        let mut total_len = 0usize;
        for mat in mats {
            let bytes = mat.to_compact_bytes();
            total_len += 8 + bytes.len();
            parts.push(bytes);
        }
        let mut out = Vec::with_capacity(total_len);
        for bytes in parts {
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    pub fn from_compact_bytes(params: &GpuDCRTPolyParams, bytes: &[u8]) -> Option<Self> {
        let mut offset = 0usize;
        let next = |buf: &[u8], offset: &mut usize| -> Option<Vec<u8>> {
            if *offset + 8 > buf.len() {
                return None;
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&buf[*offset..*offset + 8]);
            let len = u64::from_le_bytes(len_bytes) as usize;
            *offset += 8;
            if *offset + len > buf.len() {
                return None;
            }
            let out = buf[*offset..*offset + len].to_vec();
            *offset += len;
            Some(out)
        };
        let r_bytes = next(bytes, &mut offset)?;
        let e_bytes = next(bytes, &mut offset)?;
        let a_bytes = next(bytes, &mut offset)?;
        let b_bytes = next(bytes, &mut offset)?;
        let d_bytes = next(bytes, &mut offset)?;
        let re_bytes = next(bytes, &mut offset)?;
        if offset != bytes.len() {
            return None;
        }

        Some(Self {
            r: GpuDCRTPolyMatrix::from_compact_bytes(params, &r_bytes),
            e: GpuDCRTPolyMatrix::from_compact_bytes(params, &e_bytes),
            a_mat: GpuDCRTPolyMatrix::from_compact_bytes(params, &a_bytes),
            b_mat: GpuDCRTPolyMatrix::from_compact_bytes(params, &b_bytes),
            d_mat: GpuDCRTPolyMatrix::from_compact_bytes(params, &d_bytes),
            re: GpuDCRTPolyMatrix::from_compact_bytes(params, &re_bytes),
        })
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
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let trapdoor = GpuDCRTTrapdoor::new(params, size, self.sigma);
        let a_bar = uniform_sampler.sample_uniform(params, size, size, DistType::FinRingDist);
        let g = GpuDCRTPolyMatrix::gadget_matrix(params, size);
        let a0 = a_bar.concat_columns(&[&GpuDCRTPolyMatrix::identity(params, size, None)]);
        let a1 = &g - &(&a_bar * &trapdoor.r + &trapdoor.e);
        let a = a0.concat_columns(&[&a1]);
        (trapdoor, a)
    }

    fn trapdoor_to_bytes(trapdoor: &Self::Trapdoor) -> Vec<u8> {
        trapdoor.to_compact_bytes()
    }

    fn trapdoor_from_bytes(
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        bytes: &[u8],
    ) -> Option<Self::Trapdoor> {
        GpuDCRTTrapdoor::from_compact_bytes(params, bytes)
    }

    fn preimage(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M {
        let preimage_start = Instant::now();
        let d = public_matrix.row_size();
        let target_cols = target.col_size();
        debug_assert_eq!(
            target.row_size(),
            d,
            "Target matrix should have the same number of rows as the public matrix",
        );
        tracing::debug!(d = d, target_cols = target_cols, "gpu preimage: start");

        let param_start = Instant::now();
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let s = SPECTRAL_CONSTANT *
            (self.base as f64 + 1.0) *
            SIGMA *
            SIGMA *
            (((d * n * k) as f64).sqrt() + ((2 * n) as f64).sqrt() + 4.7);
        let dgg_large_std = (s * s - self.c * self.c).sqrt();
        tracing::debug!(
            elapsed_ms = param_start.elapsed().as_secs_f64() * 1_000.0,
            d = d,
            n = n,
            k = k,
            s = s,
            dgg_large_std = dgg_large_std,
            "gpu preimage: parameters derived"
        );

        let p_hat_start = Instant::now();
        let p_hat = sample_pert_square_mat_gpu_native(
            params,
            trapdoor,
            s,
            self.c,
            self.sigma,
            dgg_large_std,
            target_cols,
        );
        tracing::debug!(
            elapsed_ms = p_hat_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: sampled p_hat"
        );

        let perturb_start = Instant::now();
        let perturbed_syndrome = target - &(public_matrix * &p_hat);
        tracing::debug!(
            elapsed_ms = perturb_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: computed perturbed_syndrome"
        );

        let mut rng = rng();
        let z_seed: u64 = rng.random();
        let gauss_start = Instant::now();
        let z_hat_mat = perturbed_syndrome.gauss_samp_gq_arb_base(self.c, self.sigma, z_seed);
        drop(perturbed_syndrome);
        tracing::debug!(
            elapsed_ms = gauss_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: sampled z_hat_mat with gauss_samp_gq_arb_base"
        );

        let r_mul_start = Instant::now();
        let r_z_hat = &trapdoor.r * &z_hat_mat;
        tracing::debug!(
            elapsed_ms = r_mul_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: computed r * z_hat"
        );
        let e_mul_start = Instant::now();
        let e_z_hat = &trapdoor.e * &z_hat_mat;
        tracing::debug!(
            elapsed_ms = e_mul_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: computed e * z_hat"
        );

        let assemble_start = Instant::now();
        let correction_start = Instant::now();
        let correction = r_z_hat.concat_rows_owned(vec![e_z_hat, z_hat_mat]);
        tracing::debug!(
            elapsed_ms = correction_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: assembled correction matrix"
        );
        let mut out = p_hat;
        out.add_in_place(&correction);
        drop(correction);
        tracing::debug!(
            elapsed_ms = assemble_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: assembled output matrix"
        );
        tracing::debug!(
            elapsed_ms = preimage_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: finished"
        );
        out
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

fn sample_pert_square_mat_gpu_native(
    params: &GpuDCRTPolyParams,
    trapdoor: &GpuDCRTTrapdoor,
    s: f64,
    c: f64,
    dgg_stddev: f64,
    sigma_large: f64,
    total_ncol: usize,
) -> GpuDCRTPolyMatrix {
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let d = trapdoor.r.row_size();
    let dk = trapdoor.r.col_size();
    let num_blocks = total_ncol.div_ceil(d);
    let padded_ncol = num_blocks * d;
    let padding_ncol = padded_ncol - total_ncol;
    tracing::debug!(
        d = d,
        dk = dk,
        total_ncol = total_ncol,
        padded_ncol = padded_ncol,
        padding_ncol = padding_ncol,
        "gpu preimage sample_pert: start"
    );

    // p2 is sampled directly on GPU as in the Karney branch of OpenFHE.
    let p2 = uniform_sampler.sample_uniform(
        params,
        dk,
        padded_ncol,
        DistType::GaussDist { sigma: sigma_large },
    );
    tracing::debug!("gpu preimage sample_pert: sampled p2");
    let tp2 = &trapdoor.re * &p2;
    tracing::debug!("gpu preimage sample_pert: computed tp2");

    // Keep perturbation generation on device: this sampler uses the full
    // 2d x 2d covariance induced by (A, B, D) and Tp2.
    let mut prng = rng();
    let p1_seed: u64 = prng.random();
    let p1 = GpuDCRTPolyMatrix::sample_p1_full(
        &trapdoor.a_mat,
        &trapdoor.b_mat,
        &trapdoor.d_mat,
        &tp2,
        c,
        s,
        dgg_stddev,
        p1_seed,
    );
    tracing::debug!("gpu preimage sample_pert: sampled p1");
    drop(tp2);

    let mut p_hat = p1.concat_rows_owned(vec![p2]);
    tracing::debug!("gpu preimage sample_pert: concatenated p1/p2");
    if padding_ncol > 0 {
        p_hat = p_hat.slice_columns(0, total_ncol);
        tracing::debug!("gpu preimage sample_pert: sliced padding columns");
    }
    p_hat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        element::PolyElem,
        matrix::PolyMatrix,
        poly::{
            PolyParams,
            dcrt::{gpu::gpu_device_sync, params::DCRTPolyParams},
        },
        simulator::error_norm::compute_preimage_norm,
    };
    use bigdecimal::{BigDecimal, FromPrimitive};
    use num_bigint::{BigInt, BigUint};
    use sequential_test::sequential;

    const SIGMA: f64 = 4.578;

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 16, 8)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _, _) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    #[sequential]
    fn test_gpu_trapdoor_generation() {
        gpu_device_sync();
        let size: usize = 3;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);

        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let expected_rows = size;
        let expected_cols = (params.modulus_digits() + 2) * size;
        assert_eq!(public_matrix.row_size(), expected_rows);
        assert_eq!(public_matrix.col_size(), expected_cols);

        let k = params.modulus_digits();
        let identity = GpuDCRTPolyMatrix::identity(&params, size * k, None);
        let trapdoor_matrix = trapdoor.r.concat_rows(&[&trapdoor.e, &identity]);
        let muled = public_matrix * trapdoor_matrix;
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, size);
        assert_eq!(muled, gadget_matrix);
    }

    #[test]
    #[sequential]
    fn test_gpu_trapdoor_round_trip_bytes() {
        gpu_device_sync();
        let size: usize = 3;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, _public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let bytes =
            <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_to_bytes(&trapdoor);
        let decoded = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_from_bytes(
            &params, &bytes,
        )
        .expect("trapdoor bytes should decode");
        let reencoded =
            <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_to_bytes(&decoded);
        assert_eq!(
            bytes, reencoded,
            "trapdoor compact bytes should be stable across decode/encode"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_generation_square() {
        gpu_device_sync();
        let size = 3usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
        let product = &public_matrix * &preimage;
        assert_eq!(product, target);
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_generation_square_not_plain_gadget_solution() {
        gpu_device_sync();
        let size = 3usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);

        // Deterministic gadget preimage baseline:
        // z_plain = [R*z; E*z; z], where z = decompose(target).
        let z_plain = target.decompose();
        let z_plain_former = (&trapdoor.r * &z_plain).concat_rows(&[&(&trapdoor.e * &z_plain)]);
        let z_plain_full = z_plain_former.concat_rows(&[&z_plain]);
        assert_eq!(&public_matrix * &z_plain_full, target);

        let sampled = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
        assert_eq!(&public_matrix * &sampled, target);
        assert_ne!(
            sampled, z_plain_full,
            "preimage sampler should not collapse to the plain deterministic gadget preimage"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_coefficients_below_compute_preimage_norm() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = DCRTPolyParams::new(1 << 12, 4, 51, 17);
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();

        let ring_dim_sqrt = BigDecimal::from_u32(params.ring_dimension())
            .expect("ring dimension should convert to BigDecimal")
            .sqrt()
            .expect("ring dimension sqrt should exist");
        let base = BigDecimal::from_biguint(BigUint::from(1u32) << params.base_bits(), 0);
        let m_g = (size * params.modulus_digits()) as u64;
        let preimage_norm_bound = compute_preimage_norm(&ring_dim_sqrt, m_g, &base);
        let modulus = params.modulus();

        for sample_idx in 0..4usize {
            let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);
            let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
            assert_eq!(&public_matrix * &preimage, target);

            for i in 0..preimage.row_size() {
                for j in 0..preimage.col_size() {
                    let poly = preimage.entry(i, j);
                    for (k, coeff) in poly.coeffs().into_iter().enumerate() {
                        let value = coeff.value().clone();
                        let neg = modulus.as_ref() - &value;
                        let centered_abs = if value < neg { value } else { neg };
                        let centered_bd = BigDecimal::from(BigInt::from(centered_abs.clone()));
                        assert!(
                            centered_bd < preimage_norm_bound,
                            "preimage coeff exceeds compute_preimage_norm bound at sample={}, row={}, col={}, coeff_idx={}, centered_abs={}, bound={}",
                            sample_idx,
                            i,
                            j,
                            k,
                            centered_abs,
                            preimage_norm_bound
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_p_hat_coefficients_below_compute_preimage_norm() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = DCRTPolyParams::new(1 << 12, 4, 51, 17);
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, _public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let ring_dim_sqrt = BigDecimal::from_u32(params.ring_dimension())
            .expect("ring dimension should convert to BigDecimal")
            .sqrt()
            .expect("ring dimension sqrt should exist");
        let base = BigDecimal::from_biguint(BigUint::from(1u32) << params.base_bits(), 0);
        let m_g = (size * params.modulus_digits()) as u64;
        let preimage_norm_bound = compute_preimage_norm(&ring_dim_sqrt, m_g, &base);
        let modulus = params.modulus();
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let s = SPECTRAL_CONSTANT *
            ((1u32 << params.base_bits()) as f64 + 1.0) *
            SIGMA *
            SIGMA *
            (((size * n * k) as f64).sqrt() + ((2 * n) as f64).sqrt() + 4.7);
        let dgg_large_std =
            (s * s - (((1u32 << params.base_bits()) as f64 + 1.0) * SIGMA).powi(2)).sqrt();

        for sample_idx in 0..4usize {
            let p_hat = sample_pert_square_mat_gpu_native(
                &params,
                &trapdoor,
                s,
                ((1u32 << params.base_bits()) as f64 + 1.0) * SIGMA,
                SIGMA,
                dgg_large_std,
                size,
            );
            for i in 0..p_hat.row_size() {
                for j in 0..p_hat.col_size() {
                    let poly = p_hat.entry(i, j);
                    for (coeff_idx, coeff) in poly.coeffs().into_iter().enumerate() {
                        let value = coeff.value().clone();
                        let neg = modulus.as_ref() - &value;
                        let centered_abs = if value < neg { value } else { neg };
                        let centered_bd = BigDecimal::from(BigInt::from(centered_abs.clone()));
                        assert!(
                            centered_bd < preimage_norm_bound,
                            "p_hat coeff exceeds compute_preimage_norm bound at sample={}, row={}, col={}, coeff_idx={}, centered_abs={}, bound={}",
                            sample_idx,
                            i,
                            j,
                            coeff_idx,
                            centered_abs,
                            preimage_norm_bound
                        );
                    }
                }
            }
        }
    }
}
