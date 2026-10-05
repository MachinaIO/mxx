//! Subgraph kernels: a named subgraph the GPU runtime executes with one
//! caller-supplied native entry point instead of lowering its body.
//!
//! The subgraph stays an ordinary `SubgraphCall` in the IR, so validation,
//! liveness, the CPU executor, and Lean export read its body as usual. Only
//! GPU planning consults the kernels registered in
//! `GpuRuntimeOptions::subgraph_kernels`: a call whose definition name is
//! registered lowers to one operation that calls `entry` while the GPU graph
//! is built. The entry implements the subgraph's semantics exactly against
//! `crates/backends/gpu/include/SubgraphKernel.h`.
//!
//! Matrix operands are passed in evaluation form, integer operands in their signed encoding, and
//! matrix families as a per-member limb table refreshed before every launch. The runtime allocates
//! the results, a scratch buffer of `scratch_bytes`, and a status word. The entry adds its kernels
//! with `mxx_gpu_launch_kernel` or `mxx_gpu_launch_cooperative_kernel`, declaring every resident
//! address as a patch of its operand's binding so replays rebind it. `mxx-fhe` registers the TFHE
//! blind rotation this way, running the whole CMUX loop as one cooperative launch.

use std::ffi::{c_int, c_void};

/// Native entry point: receives a `const MxxSubgraphLaunch *` and returns 0,
/// or a nonzero status after recording a message with `gpu_set_last_error`.
pub type GpuSubgraphKernelEntry = unsafe extern "C" fn(launch: *const c_void) -> c_int;

/// The physical form of one subgraph argument or result the entry receives
/// (`MxxSubgraphOperandKind`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum GpuKernelOperandKind {
    /// An evaluation-domain matrix.
    Matrix,
    /// A family of evaluation-domain matrices, as a table of member views.
    MatrixFamily,
    /// One integer.
    Integer,
    /// A family of integers.
    IntegerFamily,
}

/// Backend and target contract for a separately compiled native subgraph.
/// A registration must describe the build of its entry point; copying this
/// value from an unrelated build does not make that entry compatible.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct GpuKernelBuildIdentity {
    pub backend: String,
    pub architecture: String,
    pub native_revision: String,
    /// Source and build revision of the registered entry point itself.
    pub kernel_revision: String,
}

impl GpuKernelBuildIdentity {
    /// Identity published by the backend build for kernels built against its
    /// exported header, target and compiler configuration.
    #[cfg(feature = "gpu")]
    pub fn current() -> Self {
        Self {
            backend: env!("MXX_GPU_BACKEND").into(),
            architecture: env!("MXX_GPU_ARCH").into(),
            native_revision: env!("MXX_NATIVE_KERNEL_BUILD_REVISION").into(),
            kernel_revision: env!("MXX_NATIVE_KERNEL_BUILD_REVISION").into(),
        }
    }

    /// Rejects registration from a different backend, target or native build
    /// before any entry point is invoked.
    #[cfg(feature = "gpu")]
    pub fn validate_current(&self) -> Result<(), String> {
        let current = Self::current();
        if self.backend != current.backend ||
            self.architecture != current.architecture ||
            self.native_revision != current.native_revision ||
            self.kernel_revision.is_empty()
        {
            return Err(
                "subgraph kernel backend, target or native revision differs from the GPU build"
                    .into(),
            );
        }
        Ok(())
    }
}

/// A registered subgraph kernel. A call matches it when its definition name
/// equals `name` and its arguments (captures last) and results have the
/// listed kinds; a named call whose operands differ is a planning error.
#[derive(Clone, Debug)]
pub struct GpuSubgraphKernel {
    pub name: String,
    /// Build contract of the native entry point.
    pub build_identity: GpuKernelBuildIdentity,
    pub inputs: Vec<GpuKernelOperandKind>,
    /// Results; each is a `Matrix` the runtime allocates.
    pub outputs: Vec<GpuKernelOperandKind>,
    /// Constants passed to every launch.
    pub parameters: Vec<u64>,
    /// Size of the device scratch buffer each call receives.
    pub scratch_bytes: u64,
    pub entry: GpuSubgraphKernelEntry,
}

impl PartialEq for GpuSubgraphKernel {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name &&
            self.build_identity == other.build_identity &&
            self.inputs == other.inputs &&
            self.outputs == other.outputs &&
            self.parameters == other.parameters &&
            self.scratch_bytes == other.scratch_bytes &&
            self.entry as usize == other.entry as usize
    }
}

impl Eq for GpuSubgraphKernel {}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::GpuKernelBuildIdentity;

    #[test]
    fn test_gpu_subgraph_build_identity_rejects_stale_registration() {
        let current = GpuKernelBuildIdentity::current();
        current.validate_current().unwrap();
        for field in 0..3 {
            let mut stale = current.clone();
            match field {
                0 => stale.backend.push_str("-different"),
                1 => stale.architecture.push_str("-different"),
                _ => stale.native_revision.push_str("-different"),
            }
            assert!(stale.validate_current().is_err());
        }
        let mut missing_entry_revision = current.clone();
        missing_entry_revision.kernel_revision.clear();
        assert!(missing_entry_revision.validate_current().is_err());
        let mut revised_entry = current.clone();
        revised_entry.kernel_revision.push_str("-entry-revision");
        revised_entry.validate_current().unwrap();
        assert_ne!(current, revised_entry);
    }
}
