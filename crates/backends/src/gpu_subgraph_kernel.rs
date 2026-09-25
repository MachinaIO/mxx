//! Subgraph kernels: a named subgraph the GPU runtime executes with one
//! caller-supplied native entry point instead of lowering its body.
//!
//! The subgraph stays an ordinary `SubgraphCall` in the IR, so validation,
//! liveness, the CPU executor, and Lean export read its body as usual. Only
//! GPU planning consults the kernels registered in
//! `GpuRuntimeOptions::subgraph_kernels`: a call whose definition name is
//! registered lowers to one operation that calls `entry` while the CUDA graph
//! is built. The entry implements the subgraph's semantics exactly against
//! `crates/backends/cuda/include/SubgraphKernel.cuh`.

use std::ffi::{c_int, c_void};

/// Native entry point: receives a `const MxxSubgraphLaunch *` and returns 0,
/// or a nonzero status after recording a message with `gpu_set_last_error`.
pub type GpuSubgraphKernelEntry = unsafe extern "C" fn(launch: *const c_void) -> c_int;

/// The physical form of one subgraph argument or result the entry receives
/// (`MxxSubgraphOperandKind`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

/// A registered subgraph kernel. A call matches it when its definition name
/// equals `name` and its arguments (captures last) and results have the
/// listed kinds; a named call whose operands differ is a planning error.
#[derive(Clone, Debug)]
pub struct GpuSubgraphKernel {
    pub name: String,
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
            self.inputs == other.inputs &&
            self.outputs == other.outputs &&
            self.parameters == other.parameters &&
            self.scratch_bytes == other.scratch_bytes &&
            self.entry as usize == other.entry as usize
    }
}

impl Eq for GpuSubgraphKernel {}
