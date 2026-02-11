use super::DCRTTrapdoor;
use crate::{
    matrix::{
        PolyMatrix,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuMatrixSampleDist},
    },
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
    },
    sampler::{DistType, PolyTrapdoorSampler},
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
        let log_base_q = params.modulus_digits();
        let dist = DistType::GaussDist { sigma };
        let r = sample_gpu_matrix_native(params, size, size * log_base_q, dist);
        let e = sample_gpu_matrix_native(params, size, size * log_base_q, dist);
        let a_mat = &r * &r.transpose(); // d x d
        let b_mat = &r * &e.transpose(); // d x d
        let d_mat = &e * &e.transpose(); // d x d
        let re = r.concat_rows(&[&e]);
        Self { r, e, a_mat, b_mat, d_mat, re }
    }

    pub fn to_compact_bytes(&self) -> Vec<u8> {
        self.to_cpu_trapdoor().to_compact_bytes()
    }

    pub fn from_compact_bytes(params: &GpuDCRTPolyParams, bytes: &[u8]) -> Option<Self> {
        let cpu_params = cpu_params_from_gpu(params);
        let cpu = DCRTTrapdoor::from_compact_bytes(&cpu_params, bytes)?;
        Some(Self {
            r: GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu.r),
            e: GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu.e),
            a_mat: GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu.a_mat),
            b_mat: GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu.b_mat),
            d_mat: GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu.d_mat),
            re: GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu.re),
        })
    }

    pub(crate) fn to_cpu_trapdoor(&self) -> DCRTTrapdoor {
        DCRTTrapdoor {
            r: self.r.to_cpu_matrix(),
            e: self.e.to_cpu_matrix(),
            a_mat: self.a_mat.to_cpu_matrix(),
            b_mat: self.b_mat.to_cpu_matrix(),
            d_mat: self.d_mat.to_cpu_matrix(),
            re: self.re.to_cpu_matrix(),
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
        let a_bar = sample_gpu_matrix_native(params, size, size, DistType::FinRingDist);
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

        // OpenFHE-equivalent GaussSampGqArbBase path on GPU:
        // this keeps the perturbation + gadget preimage step randomized.
        let mut rng = rng();
        let z_seed: u64 = rng.random();
        let gauss_start = Instant::now();
        let z_hat_mat = perturbed_syndrome.gauss_samp_gq_arb_base(self.c, self.sigma, z_seed);
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
        let z_hat_former = (p_hat.slice_rows(0, d) + r_z_hat)
            .concat_rows(&[&(p_hat.slice_rows(d, 2 * d) + e_z_hat)]);
        let z_hat_latter = p_hat.slice_rows(2 * d, d * (k + 2)) + z_hat_mat;
        let out = z_hat_former.concat_rows(&[&z_hat_latter]);
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
        let preimage_right = sample_gpu_matrix_native(params, ext_ncol, target_ncol, dist);
        let t = target - &(ext_matrix * &preimage_right);
        let preimage_left = self.preimage(params, trapdoor, public_matrix, &t);
        preimage_left.concat_rows(&[&preimage_right])
    }
}

fn cpu_params_from_gpu(params: &GpuDCRTPolyParams) -> DCRTPolyParams {
    DCRTPolyParams::new(
        params.ring_dimension(),
        params.crt_depth(),
        params.crt_bits(),
        params.base_bits(),
    )
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
    let d = trapdoor.r.row_size();
    let dk = trapdoor.r.col_size();
    let num_blocks = total_ncol.div_ceil(d);
    let padded_ncol = num_blocks * d;
    let padding_ncol = padded_ncol - total_ncol;

    // p2 is sampled directly on GPU as in the Karney branch of OpenFHE.
    let p2 = sample_gpu_matrix_native(
        params,
        dk,
        padded_ncol,
        DistType::GaussDist { sigma: sigma_large },
    );
    let tp2 = &trapdoor.re * &p2;

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

    let mut p_hat = p1.concat_rows(&[&p2]);
    if padding_ncol > 0 {
        p_hat = p_hat.slice_columns(0, total_ncol);
    }
    p_hat
}

fn sample_gpu_matrix_native(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    ncol: usize,
    dist: DistType,
) -> GpuDCRTPolyMatrix {
    if nrow == 0 || ncol == 0 {
        return GpuDCRTPolyMatrix::zero(params, nrow, ncol);
    }
    let mut prng = rng();
    let seed: u64 = prng.random();
    match dist {
        DistType::FinRingDist => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Uniform,
            0.0,
            seed,
        ),
        DistType::GaussDist { sigma } => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Gauss,
            sigma,
            seed,
        ),
        DistType::BitDist => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Bit,
            0.0,
            seed,
        ),
        DistType::TernaryDist => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Ternary,
            0.0,
            seed,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        matrix::PolyMatrix,
        poly::{
            PolyParams,
            dcrt::{gpu::gpu_device_sync, params::DCRTPolyParams},
        },
    };
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
        let target = sample_gpu_matrix_native(&params, size, size, DistType::FinRingDist);

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
        let target = sample_gpu_matrix_native(&params, size, size, DistType::FinRingDist);

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
}
