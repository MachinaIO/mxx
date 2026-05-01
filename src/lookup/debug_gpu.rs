use crate::{
    lookup::debug::DebugTrapdoorPreimage,
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::Poly,
    sampler::trapdoor::GpuDCRTTrapdoor,
};

impl DebugTrapdoorPreimage<GpuDCRTPolyMatrix> for GpuDCRTTrapdoor {
    fn debug_preimage(
        &self,
        _params: &<<GpuDCRTPolyMatrix as PolyMatrix>::P as Poly>::Params,
        target: &GpuDCRTPolyMatrix,
    ) -> GpuDCRTPolyMatrix {
        let decomposed = target.decompose();
        let r_part = &self.r * &decomposed;
        let e_part = &self.e * &decomposed;
        r_part.concat_rows(&[&e_part, &decomposed])
    }
}
