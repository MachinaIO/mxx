//! CUDA TFHE native kernel registration; HIP deliberately offers no native TFHE kernel.

#[cfg(mxx_gpu_backend = "cuda")]
use crate::tfhe::BLIND_ROTATION_SUBGRAPH;
use crate::tfhe::TfheParams;

impl TfheParams {
    /// The GPU kernel that executes the blind rotation subgraph of these
    /// parameters in one cooperative launch
    /// (`crates/fhe/cuda/tfhe_blind_rotation.cu`), for registration in
    /// `GpuRuntimeOptions::subgraph_kernels`. `None` when the kernel does not
    /// cover the parameters: it needs a ring dimension of 16 to 2048, at most
    /// four CRT limbs below 2^31, and a base of at most 30 bits.
    /// Returns `None` on HIP: AMD TFHE support is deferred.
    #[cfg(feature = "gpu")]
    pub fn gpu_blind_rotation_kernel(
        &self,
    ) -> Option<mxx_backends::gpu_subgraph_kernel::GpuSubgraphKernel> {
        #[cfg(mxx_gpu_backend = "hip")]
        {
            None
        }
        #[cfg(mxx_gpu_backend = "cuda")]
        {
            use mxx_backends::{
                gpu_subgraph_kernel::{GpuKernelOperandKind, GpuSubgraphKernel},
                poly::PolyParams,
            };
            unsafe extern "C" {
                fn mxx_fhe_tfhe_blind_rotation(launch: *const std::ffi::c_void) -> std::ffi::c_int;
            }
            let ring = &self.common.ring;
            let (moduli, crt_bits, limbs) = ring.to_crt();
            let ring_dimension = ring.ring_dimension() as u64;
            if !(16..=2048).contains(&ring_dimension) ||
                limbs > 4 ||
                moduli.iter().any(|&modulus| modulus >= 1 << 31) ||
                ring.base_bits() > 30
            {
                return None;
            }
            let digits_per_tower = crt_bits.div_ceil(ring.base_bits() as usize);
            let mut build_identity =
                mxx_backends::gpu_subgraph_kernel::GpuKernelBuildIdentity::current();
            build_identity.kernel_revision = env!("MXX_FHE_GPU_NATIVE_REVISION").into();
            Some(GpuSubgraphKernel {
                name: BLIND_ROTATION_SUBGRAPH.into(),
                build_identity,
                inputs: vec![
                    GpuKernelOperandKind::Matrix,
                    GpuKernelOperandKind::Matrix,
                    GpuKernelOperandKind::IntegerFamily,
                    GpuKernelOperandKind::MatrixFamily,
                    GpuKernelOperandKind::MatrixFamily,
                ],
                outputs: vec![GpuKernelOperandKind::Matrix, GpuKernelOperandKind::Matrix],
                parameters: vec![
                    self.lwe_dimension as u64,
                    self.lwe_modulus.bits() - 1,
                    u64::from(ring.base_bits()),
                    digits_per_tower as u64,
                    (ring.modulus_digits() / digits_per_tower) as u64,
                ],
                // The difference coefficients of every (limb, row) as 32-bit
                // words (padded to 8 bytes), then a 64-bit product accumulator
                // of every (row, limb).
                scratch_bytes: (2 * limbs as u64 * ring_dimension).div_ceil(2) * 8 +
                    2 * limbs as u64 * ring_dimension * 8,
                entry: mxx_fhe_tfhe_blind_rotation,
            })
        }
    }
}
