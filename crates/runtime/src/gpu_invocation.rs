//! Concrete arguments at the runtime's GPU preflight boundary.
//!
//! Sampling requests deliberately omit production randomness and transcript
//! identifiers. Preimage pilots receive a target layout, never its loader.

use crate::backend::{IndexRange, MatrixMulAccumulateRequest, SampleRange};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix, HashVariant, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, NodeId, Port, WireRef},
};
use num_bigint::BigInt;

/// Validated IR metadata of one production matrix invocation: the node kind,
/// its concrete argument and output wire types in port order, and the parameter
/// bindings of the instance that runs it.
///
/// It is exactly the metadata the single operation lowering
/// (`PreparedMatrixOperation::from_ir`) consumes, so admission and execution
/// share one derivation from the validated IR. Actual operand owners and
/// payloads are bound separately at execution.
#[derive(Clone, Debug)]
pub struct GpuNodeOperation {
    scope: FrozenGraphScopeId,
    node: NodeId,
    kind: NodeKind,
    argument_wires: Vec<WireRef>,
    arguments: Vec<ConcreteWireType>,
    outputs: Vec<ConcreteWireType>,
    bindings: ParamEnv,
}

impl GpuNodeOperation {
    /// Resolve the validated node using the actual candidate/production
    /// bindings. Root execution can reuse validation's concrete types; child
    /// instances must concretize declared shapes and bounds in their own env.
    /// This is shared by resource admission and actual submission.
    pub fn new(
        validated: &ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        node_id: NodeId,
        bindings: &ParamEnv,
    ) -> Result<Self, String> {
        let scope = validated.source.scope(scope_id).expect("validated scope");
        let node = scope.node(node_id).expect("validated node");
        let concrete = |wire| {
            validated
                .concrete_wire_type(scope_id, wire, bindings)
                .map_err(|error| error.to_string())
        };
        let argument_wires = scope.arguments(node).expect("validated arguments");
        let arguments =
            argument_wires.iter().copied().map(concrete).collect::<Result<Vec<_>, _>>()?;
        let outputs = (0..node.output_types().len())
            .map(|port| concrete(WireRef { node: node_id, port: Port(port as u32) }))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            scope: scope_id.clone(),
            node: node_id,
            kind: node.kind().clone(),
            argument_wires,
            arguments,
            outputs,
            bindings: bindings.clone(),
        })
    }

    pub(crate) fn scope(&self) -> &FrozenGraphScopeId {
        &self.scope
    }

    pub(crate) fn node(&self) -> NodeId {
        self.node
    }

    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    pub(crate) fn argument_wires(&self) -> &[WireRef] {
        &self.argument_wires
    }

    pub fn arguments(&self) -> &[ConcreteWireType] {
        &self.arguments
    }

    pub fn outputs(&self) -> &[ConcreteWireType] {
        &self.outputs
    }

    pub fn bindings(&self) -> &ParamEnv {
        &self.bindings
    }
}

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

/// Preflight one invocation of the node that owns it. `node` is the validated
/// IR metadata when the invocation is that node's operation; callers whose
/// invocation is not an IR node (fusion results, sampler algorithm steps and
/// artifact or transcript imports) pass `None` and lower from the invocation
/// itself.
pub fn preflight<B: crate::backend::Backend>(
    backend: &mut B,
    node: Option<GpuNodeOperation>,
    invocation: GpuInvocation<'_, B::Matrix, B::SmallMatrix, B::Trapdoor>,
) -> Result<(), B::Error> {
    let placement = backend.active_placement();
    backend.preflight_gpu_operations(&[(placement, node, invocation)])
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
            preflight(backend, None, GpuInvocation::ImportMatrix { ty, bytes })
        }
        (artifact_type, ArtifactPayload::SmallMatrix(bytes))
            if artifact_type.bounded_matrix_schema().is_some() =>
        {
            let (schema, semantic_kind) = artifact_type.bounded_matrix_schema().unwrap();
            preflight(
                backend,
                None,
                GpuInvocation::ImportSmallMatrix { schema: &schema, bytes, semantic_kind },
            )
        }
        (
            ArtifactType::Trapdoor { matrix, .. },
            ArtifactPayload::Trapdoor { public_bytes, secret_bytes },
        ) => {
            let placement = backend.active_placement();
            backend.preflight_gpu_operations(&[
                (placement, None, GpuInvocation::ImportMatrix { ty: matrix, bytes: public_bytes }),
                (
                    placement,
                    None,
                    GpuInvocation::ImportTrapdoor { ty: matrix, bytes: secret_bytes },
                ),
            ])
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::IntExpr;

    #[test]
    fn test_node_operation_uses_actual_shape_and_bound_bindings() {
        let ring = Ring::new(257, 16);
        let input = ring.small_matrix_input(
            "input",
            (IntExpr::Var("rows".into()), 4),
            IntExpr::Var("bound".into()),
        );
        let env = |rows, bound| ParamEnv {
            integers: [("rows".into(), BigInt::from(rows)), ("bound".into(), BigInt::from(bound))]
                .into_iter()
                .collect(),
            ..ParamEnv::default()
        };
        let validated = DslContext::new("actual-gpu-node-bindings")
            .int_parameter("rows")
            .int_parameter("bound")
            .output("output", input)
            .unwrap()
            .build()
            .unwrap()
            .validate(&env(2, 7))
            .unwrap();
        let scope = FrozenGraphScopeId::Root;
        let output = validated.source.scope(&scope).unwrap().outputs()[0];
        let cached = GpuNodeOperation::new(&validated, &scope, output.node, &env(2, 7)).unwrap();
        assert_eq!(
            &cached.outputs()[output.port.0 as usize],
            &validated.root_scope().wire_types[&output]
        );
        let actual = GpuNodeOperation::new(&validated, &scope, output.node, &env(3, 9)).unwrap();
        let ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } =
            &actual.outputs()[output.port.0 as usize]
        else {
            panic!("compact matrix type")
        };
        assert_eq!((matrix.rows, matrix.columns), (3, 4));
        assert_eq!(*max_coefficient_bound, BigInt::from(9));
    }
}
