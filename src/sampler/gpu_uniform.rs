use crate::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::{GpuDCRTPoly, GpuDCRTPolyParams}, params::DCRTPolyParams},
    },
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};

pub struct GpuDCRTPolyUniformSampler {
    cpu_sampler: DCRTPolyUniformSampler,
}

impl GpuDCRTPolyUniformSampler {
    fn cpu_params(params: &GpuDCRTPolyParams) -> DCRTPolyParams {
        let cpu_params = DCRTPolyParams::new(
            params.ring_dimension(),
            params.crt_depth(),
            params.crt_bits(),
            params.base_bits(),
        );
        debug_assert_eq!(cpu_params.to_crt().0.as_slice(), params.moduli());
        cpu_params
    }
}

impl Default for GpuDCRTPolyUniformSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl PolyUniformSampler for GpuDCRTPolyUniformSampler {
    type M = GpuDCRTPolyMatrix;

    fn new() -> Self {
        Self { cpu_sampler: DCRTPolyUniformSampler::new() }
    }

    fn sample_poly(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        dist: &DistType,
    ) -> <Self::M as PolyMatrix>::P {
        let cpu_params = Self::cpu_params(params);
        let cpu_poly = self.cpu_sampler.sample_poly(&cpu_params, dist);
        GpuDCRTPoly::from_compact_bytes(params, &cpu_poly.to_compact_bytes())
    }

    fn sample_uniform(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        let cpu_params = Self::cpu_params(params);
        let cpu_matrix = self.cpu_sampler.sample_uniform(&cpu_params, nrow, ncol, dist);
        GpuDCRTPolyMatrix::from_cpu_matrix(params, &cpu_matrix)
    }
}
