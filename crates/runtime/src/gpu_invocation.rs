//! Concrete arguments at the runtime's GPU preflight boundary.
//!
//! Sampling requests deliberately omit production randomness and transcript
//! identifiers. Preimage pilots receive a target layout, never its loader.

use crate::backend::{IndexRange, MatrixMulAccumulateRequest, SampleRange};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix, HashVariant, MatrixBinaryOp},
    types::ConcreteMatrixType,
};
use num_bigint::BigInt;

/// Metadata required to construct an isolated target with the production layout.
#[derive(Clone, Debug)]
pub enum GpuColumnSourceLayout {
    /// The default for backends whose staging representation is ring-independent.
    Logical { matrix_type: ConcreteMatrixType, global_column_start: usize },
    RnsStaging {
        matrix_type: ConcreteMatrixType,
        global_column_start: usize,
        level: usize,
        is_ntt: bool,
        bytes_per_poly: usize,
    },
}

/// One actual primitive invocation, after its operands have been materialized.
/// Borrowed payloads are inputs to planning; pilots must allocate private outputs.
#[derive(Clone, Debug)]
pub enum GpuInvocation<'a, M, S, T> {
    Constant {
        ty: &'a ConcreteMatrixType,
        value: &'a ConstantMatrix,
        env: &'a ParamEnv,
    },
    Binary {
        operation: MatrixBinaryOp,
        left: &'a M,
        right: &'a M,
    },
    Accumulate {
        request: &'a MatrixMulAccumulateRequest<M>,
    },
    Negate {
        value: &'a M,
    },
    ScaleInteger {
        value: &'a M,
        scalar: &'a BigInt,
    },
    RingAutomorphism {
        value: &'a M,
        index: usize,
    },
    Transpose {
        value: &'a M,
    },
    Slice {
        value: &'a M,
        rows: Option<&'a IndexRange>,
        columns: Option<&'a IndexRange>,
    },
    SumRows {
        value: &'a M,
        rows: &'a [Vec<usize>],
    },
    Tensor {
        left: &'a M,
        right: &'a M,
    },
    TensorSumRows {
        left: &'a M,
        right: &'a M,
        rows: &'a [Vec<usize>],
    },
    Concat {
        inputs: &'a [&'a M],
        axis: ConcatAxis,
    },
    AddRowBlocks {
        blocks: &'a [&'a M],
        right: &'a M,
    },
    MultiplySmallRhs {
        left: &'a M,
        right: &'a S,
    },
    MultiplySmallRhsRowBlocks {
        blocks: &'a [&'a M],
        right: &'a S,
    },
    GadgetDecompose {
        value: &'a M,
        small: bool,
        digit_count: Option<usize>,
    },
    GadgetDecomposeRowBlocks {
        blocks: &'a [&'a M],
        small: bool,
        digit_count: Option<usize>,
    },
    ModulusSwitch {
        value: &'a M,
        destination: &'a ConcreteMatrixType,
    },
    ReduceModulus {
        value: &'a M,
        destination: &'a ConcreteMatrixType,
    },
    CenteredRebase {
        value: &'a M,
        destination: &'a ConcreteMatrixType,
    },
    CenteredExtend {
        value: &'a M,
        destination: &'a ConcreteMatrixType,
    },
    CenteredExtendSmall {
        value: &'a S,
        destination: &'a ConcreteMatrixType,
    },
    RnsModUp {
        value: &'a M,
        destination: &'a ConcreteMatrixType,
        source_moduli: &'a [u64],
        digit_size: usize,
        normalize: bool,
    },
    RnsModDown {
        value: &'a M,
        destination: &'a ConcreteMatrixType,
        source_moduli: &'a [u64],
        plaintext_modulus: u64,
    },
    BlockModSwitch {
        value: &'a M,
        destination: &'a ConcreteMatrixType,
        plaintext_modulus: u64,
    },
    CrtRecompose {
        levels: &'a [M],
        plaintext_moduli: &'a [BigInt],
        reconstruction_coefficients: &'a [BigInt],
        destination: &'a ConcreteMatrixType,
    },
    SampleUniform {
        ty: &'a ConcreteMatrixType,
        range: &'a SampleRange,
    },
    SampleGaussian {
        ty: &'a ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: &'a BigInt,
    },
    SampleHash {
        ty: &'a ConcreteMatrixType,
        variant: HashVariant,
        tag_bytes: usize,
        gadget_base: Option<&'a BigInt>,
        digit_count: Option<usize>,
    },
    SampleTrapdoor {
        ty: &'a ConcreteMatrixType,
        sigma: f64,
        gadget_base: &'a BigInt,
        digit_count: usize,
    },
    SamplePreimage {
        schema: &'a ConcreteBoundedMatrixSchema,
        sigma: f64,
        gadget_base: &'a BigInt,
        digit_count: usize,
        trapdoor: &'a T,
        public: &'a M,
        target: &'a GpuColumnSourceLayout,
    },
    ImportMatrix {
        ty: &'a ConcreteMatrixType,
        bytes: &'a [u8],
    },
    ImportSmallMatrix {
        schema: &'a ConcreteBoundedMatrixSchema,
        bytes: &'a [u8],
        semantic_kind: SmallMatrixSemanticKind,
    },
    ImportTrapdoor {
        ty: &'a ConcreteMatrixType,
        bytes: &'a [u8],
    },
    ImportCpuStaging {
        ty: &'a ConcreteMatrixType,
        bytes: &'a [u8],
    },
    PolynomialFromValues {
        ty: &'a ConcreteMatrixType,
        values: &'a [BigInt],
        evaluation: bool,
    },
    PackPolynomialCoefficients {
        ty: &'a ConcreteMatrixType,
        bits: &'a [bool],
        coefficient_bits: usize,
    },
}

pub fn preflight<B: crate::backend::Backend>(
    backend: &mut B,
    invocation: GpuInvocation<'_, B::Matrix, B::SmallMatrix, B::Trapdoor>,
) -> Result<(), B::Error> {
    let placement = backend.active_placement();
    backend.preflight_gpu_operations(&[(placement, invocation)])
}

/// Prepare imports at the public artifact decoding boundary, shared by standalone
/// callers and executor materialization.
pub fn preflight_artifact<B: crate::backend::Backend>(
    backend: &mut B,
    artifact_type: &mxx_ir_core::artifact::ArtifactType,
    payload: &crate::artifact::ArtifactPayload,
) -> Result<(), B::Error> {
    use crate::artifact::ArtifactPayload;
    use mxx_ir_core::artifact::ArtifactType;
    match (artifact_type, payload) {
        (ArtifactType::Matrix(ty), ArtifactPayload::Matrix(bytes)) => {
            preflight(backend, GpuInvocation::ImportMatrix { ty, bytes })
        }
        (artifact_type, ArtifactPayload::SmallMatrix(bytes))
            if artifact_type.bounded_matrix_schema().is_some() =>
        {
            let (schema, semantic_kind) = artifact_type.bounded_matrix_schema().unwrap();
            preflight(
                backend,
                GpuInvocation::ImportSmallMatrix { schema: &schema, bytes, semantic_kind },
            )
        }
        (
            ArtifactType::Trapdoor { matrix, .. },
            ArtifactPayload::Trapdoor { public_bytes, secret_bytes },
        ) => {
            let placement = backend.active_placement();
            backend.preflight_gpu_operations(&[
                (placement, GpuInvocation::ImportMatrix { ty: matrix, bytes: public_bytes }),
                (placement, GpuInvocation::ImportTrapdoor { ty: matrix, bytes: secret_bytes }),
            ])
        }
        _ => Ok(()),
    }
}
