//! Declarative typed construction API for mxx graphs.
//!
//! Executable operations create immutable `mxx-ir-core` nodes immediately.

use mxx_ir_core::{
    CapturePolicy, CompileParameter, CompileParameterKind, FreezeError, Graph, GraphOutput,
    IntExpr, NodeHandle, ParamEnv, RealExpr, SealMap, SealedSubgraph, SubgraphHandle, ValueHandle,
    artifact::{ArtifactConfidentiality, ProductionId},
    graph::with_new_construction_scope,
    node::{
        ArtifactInput, ConstantMatrix, HashVariant, IndexRange, LoopInputMode, MatrixBinaryOp,
        NodeKind, ParallelLoop, SampleRange, SequentialLoop,
    },
    types::{MatrixType, WireType},
};
use std::{
    cell::Cell,
    collections::BTreeMap,
    ops::{Add, Mul, Neg, Sub},
};
use thiserror::Error;

pub use mxx_ir_core::{Rational, artifact::ArtifactConfidentiality as Confidentiality};

thread_local! {
    /// Lexical loop depth while closure bodies are constructed. Using the depth as the binder
    /// slot is deterministic across builds and keeps nested loop indices distinct.
    static LOOP_BINDER_DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn with_loop_index<T>(body: impl FnOnce(LoopIndex) -> T) -> (u32, T) {
    struct RestoreDepth(u32);
    impl Drop for RestoreDepth {
        fn drop(&mut self) {
            LOOP_BINDER_DEPTH.with(|depth| depth.set(self.0));
        }
    }

    LOOP_BINDER_DEPTH.with(|depth| {
        let slot = depth.get();
        depth.set(slot.checked_add(1).expect("loop nesting depth exceeds u32"));
        let restore = RestoreDepth(slot);
        let output = body(LoopIndex { expression: IntExpr::LoopIndex(slot) });
        drop(restore);
        (slot, output)
    })
}

#[derive(Debug, Error)]
pub enum DslError {
    #[error(transparent)]
    Freeze(#[from] FreezeError),
    #[error("duplicate output name: {0}")]
    DuplicateOutput(String),
    #[error("subgraph body captures an executable value")]
    SubgraphCapture,
    #[error("graph value schema does not match its flattened values")]
    Schema,
    #[error("parallel families have different counts")]
    FamilyCountMismatch,
    #[error(transparent)]
    StructuralValidation(#[from] mxx_ir_core::ValidationError),
    #[error("ideal and predicate specifications must be sampler-free")]
    NonPureSpecification,
    #[error("a pure predicate must have exactly one boolean output")]
    PredicateOutput,
    #[error(
        "a construction trace value does not belong to the body being sealed or one of its captures"
    )]
    TraceRemap,
}

/// Resolves handles retained while a closure body was being constructed to
/// the handles that occur in the sealed child scope.
#[doc(hidden)]
pub struct BodyTraceRemapper<'a> {
    local: &'a SealMap,
    captures: &'a [mxx_ir_core::CapturedValue],
}

impl<'a> BodyTraceRemapper<'a> {
    fn new(sealed: &'a SealedSubgraph) -> Self {
        Self { local: &sealed.remap, captures: &sealed.captures }
    }

    pub fn remap_value(&self, value: &ValueHandle) -> Result<ValueHandle, DslError> {
        if let Some(remapped) = self.local.resolve(value) {
            return Ok(remapped.clone());
        }
        self.captures
            .iter()
            .find(|capture| capture.outer == *value)
            .map(|capture| capture.placeholder.clone())
            .ok_or(DslError::TraceRemap)
    }
}

/// Construction-only remapping for fields created in the current closure
/// body. Implementations must remap every raw handle exactly once.
#[doc(hidden)]
pub trait RemapConstructionTrace: Sized {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError>;
}

impl RemapConstructionTrace for () {
    fn remap_current_body(self, _map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(self)
    }
}

impl RemapConstructionTrace for ValueHandle {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        map.remap_value(&self)
    }
}

impl<T: RemapConstructionTrace> RemapConstructionTrace for Option<T> {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.map(|value| value.remap_current_body(map)).transpose()
    }
}

impl<T: RemapConstructionTrace> RemapConstructionTrace for Vec<T> {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.into_iter().map(|value| value.remap_current_body(map)).collect()
    }
}

impl<T: RemapConstructionTrace> RemapConstructionTrace for Box<T> {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Box::new((*self).remap_current_body(map)?))
    }
}

impl<T: RemapConstructionTrace, const N: usize> RemapConstructionTrace for [T; N] {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        let values = self
            .into_iter()
            .map(|value| value.remap_current_body(map))
            .collect::<Result<Vec<_>, _>>()?;
        match values.try_into() {
            Ok(values) => Ok(values),
            Err(_) => unreachable!("an array iterator preserves its length"),
        }
    }
}

macro_rules! construction_trace_tuple {
    ($(($($type:ident $field:tt),+)),+ $(,)?) => {
        $(
            impl<$($type: RemapConstructionTrace),+> RemapConstructionTrace for ($($type,)+) {
                fn remap_current_body(
                    self,
                    map: &BodyTraceRemapper<'_>,
                ) -> Result<Self, DslError> {
                    Ok(($(self.$field.remap_current_body(map)?,)+))
                }
            }
        )+
    };
}

construction_trace_tuple! {
    (A 0, B 1),
    (A 0, B 1, C 2),
    (A 0, B 1, C 2, D 3),
    (A 0, B 1, C 2, D 3, E 4),
    (A 0, B 1, C 2, D 3, E 4, F 5),
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct SealedCaptureTrace {
    pub parent_source: ValueHandle,
    pub child_placeholder: ValueHandle,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct SealedScopeTrace<T> {
    pub body: T,
    pub captures: Vec<SealedCaptureTrace>,
}

impl<T> SealedScopeTrace<T> {
    fn remap_parent_sources(mut self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        for capture in &mut self.captures {
            capture.parent_source = map.remap_value(&capture.parent_source)?;
        }
        Ok(self)
    }
}

impl<T> RemapConstructionTrace for SealedScopeTrace<T> {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.remap_parent_sources(map)
    }
}

/// Exact construction handles for one structural loop and its already-sealed
/// child scope.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct LoopConstructionTrace<T> {
    pub outputs: Vec<ValueHandle>,
    pub scope: SealedScopeTrace<T>,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct GatherConstructionTrace {
    pub index: ValueHandle,
    pub sources: Vec<ValueHandle>,
    pub outputs: Vec<ValueHandle>,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct EvaluateIntConstructionTrace {
    /// The raw `EvaluateInt` output before ordinary family-compatible materialization.
    pub expression: ValueHandle,
    /// The existing zero operand used by `DslContext::evaluate_int`.
    pub zero: ValueHandle,
    /// The existing `expression + zero` result returned by `DslContext::evaluate_int`.
    pub output: ValueHandle,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct SelectConstructionTrace {
    pub selector: ValueHandle,
    pub branches: Vec<ValueHandle>,
    pub output: ValueHandle,
}

/// Construction trace for one outer segment of little-endian bit packing.
/// The nested scan carries `(sum, weight)` and the outer output is its final
/// sum component.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct PackedBitsConstructionTrace {
    pub scan: LoopConstructionTrace<Vec<ValueHandle>>,
    pub output: ValueHandle,
}

impl RemapConstructionTrace for PackedBitsConstructionTrace {
    fn remap_current_body(mut self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.scan = self.scan.remap_current_body(map)?;
        self.output = map.remap_value(&self.output)?;
        Ok(self)
    }
}

impl RemapConstructionTrace for SelectConstructionTrace {
    fn remap_current_body(mut self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.selector = map.remap_value(&self.selector)?;
        self.branches = self.branches.remap_current_body(map)?;
        self.output = map.remap_value(&self.output)?;
        Ok(self)
    }
}

impl RemapConstructionTrace for EvaluateIntConstructionTrace {
    fn remap_current_body(mut self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.expression = map.remap_value(&self.expression)?;
        self.zero = map.remap_value(&self.zero)?;
        self.output = map.remap_value(&self.output)?;
        Ok(self)
    }
}

impl RemapConstructionTrace for GatherConstructionTrace {
    fn remap_current_body(mut self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.index = map.remap_value(&self.index)?;
        self.sources = self.sources.remap_current_body(map)?;
        self.outputs = self.outputs.remap_current_body(map)?;
        Ok(self)
    }
}

impl<T> RemapConstructionTrace for LoopConstructionTrace<T> {
    fn remap_current_body(mut self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        self.outputs = self.outputs.remap_current_body(map)?;
        self.scope = self.scope.remap_parent_sources(map)?;
        Ok(self)
    }
}

fn finish_scope_trace<T: RemapConstructionTrace>(
    sealed: &SealedSubgraph,
    raw: T,
) -> Result<SealedScopeTrace<T>, DslError> {
    let body = raw.remap_current_body(&BodyTraceRemapper::new(sealed))?;
    let captures = sealed
        .captures
        .iter()
        .map(|capture| SealedCaptureTrace {
            parent_source: capture.outer.clone(),
            child_placeholder: capture.placeholder.clone(),
        })
        .collect();
    Ok(SealedScopeTrace { body, captures })
}

fn loop_construction_trace<T: RemapConstructionTrace>(
    node: &NodeHandle,
    sealed: &SealedSubgraph,
    raw: T,
) -> Result<LoopConstructionTrace<T>, DslError> {
    let outputs = (0..node.output_types().len())
        .map(|port| node.output(port as u32).expect("known loop output port"))
        .collect();
    Ok(LoopConstructionTrace { outputs, scope: finish_scope_trace(sealed, raw)? })
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Shape {
    pub rows: IntExpr,
    pub columns: IntExpr,
}

pub trait IntoShape {
    fn into_shape(self) -> Shape;
}

impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}

impl<R: Into<IntExpr>, C: Into<IntExpr>> IntoShape for (R, C) {
    fn into_shape(self) -> Shape {
        Shape { rows: self.0.into(), columns: self.1.into() }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatType(pub MatrixType);

impl MatType {
    pub fn new(matrix: MatrixType) -> Self {
        Self(matrix)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BytesType {
    pub length: IntExpr,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IntType;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BoolType;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PreimageType(pub MatrixType);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TrapdoorType {
    pub matrix: MatrixType,
    pub sigma: RealExpr,
    pub gadget_base: IntExpr,
    pub digit_count: IntExpr,
    pub preimage_max_coefficient_bound: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TrapdoorFamilyType {
    pub element: TrapdoorType,
    pub count: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatFamilyType {
    pub element: MatrixType,
    pub count: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IntFamilyType {
    pub count: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BoolFamilyType {
    pub count: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Ring {
    modulus: IntExpr,
    ring_dimension: IntExpr,
}

impl Ring {
    /// Creates a protocol boolean input.
    ///
    /// This is placed on `Ring` for consistency with the other typed input builders; the value
    /// itself is ring-independent.
    #[track_caller]
    pub fn bool_input(&self, name: impl Into<String>) -> Bool {
        let node = NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: WireType::Bool, artifact: None },
            Vec::new(),
            vec![WireType::Bool],
        );
        Bool { value: node.output(0).expect("boolean input"), pending: Pending::default() }
    }

    pub fn new(modulus: impl Into<IntExpr>, ring_dimension: impl Into<IntExpr>) -> Self {
        Self { modulus: modulus.into(), ring_dimension: ring_dimension.into() }
    }

    pub fn matrix_type(&self, shape: impl IntoShape) -> MatrixType {
        let shape = shape.into_shape();
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: shape.rows,
            columns: shape.columns,
        }
    }

    #[track_caller]
    pub fn input(&self, name: impl Into<String>, shape: impl IntoShape) -> Mat {
        Mat::source_input(name.into(), self.matrix_type(shape), None)
    }

    #[track_caller]
    pub fn artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        shape: impl IntoShape,
        confidentiality: ArtifactConfidentiality,
    ) -> Mat {
        let artifact_name = artifact_name.into();
        Mat::source_input(
            artifact_name.clone(),
            self.matrix_type(shape),
            Some(ArtifactInput { production_id, artifact_name, confidentiality }),
        )
    }

    #[track_caller]
    pub fn preimage_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        shape: impl IntoShape,
        confidentiality: ArtifactConfidentiality,
    ) -> Preimage {
        let artifact_name = artifact_name.into();
        let matrix_type = self.matrix_type(shape);
        let wire_type = WireType::Preimage(matrix_type.clone());
        let node = NodeHandle::new(
            NodeKind::Input {
                name: artifact_name.clone(),
                wire_type: wire_type.clone(),
                artifact: Some(ArtifactInput { production_id, artifact_name, confidentiality }),
            },
            Vec::new(),
            vec![wire_type],
        );
        Preimage {
            value: node.output(0).expect("preimage artifact input"),
            matrix_type,
            pending: Pending::default(),
        }
    }

    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn trapdoor_artifact_input(
        &self,
        production_id: ProductionId,
        public_artifact_name: impl Into<String>,
        trapdoor_artifact_name: impl Into<String>,
        rows: impl Into<IntExpr>,
        sigma: impl Into<RealExpr>,
        gadget_base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
        preimage_max_coefficient_bound: impl Into<IntExpr>,
    ) -> Trapdoor {
        let rows = rows.into();
        let sigma = sigma.into();
        let gadget_base = gadget_base.into();
        let digit_count = digit_count.into();
        let preimage_max_coefficient_bound = preimage_max_coefficient_bound.into();
        let matrix_type = self.matrix_type(Shape {
            rows: rows.clone(),
            columns: IntExpr::Mul(
                Box::new(rows),
                Box::new(IntExpr::Add(
                    Box::new(digit_count.clone()),
                    Box::new(IntExpr::constant(2)),
                )),
            )
            .canonicalize(),
        });
        let public_artifact_name = public_artifact_name.into();
        let public = Mat::source_input(
            public_artifact_name.clone(),
            matrix_type.clone(),
            Some(ArtifactInput {
                production_id: production_id.clone(),
                artifact_name: public_artifact_name,
                confidentiality: ArtifactConfidentiality::Public,
            }),
        );
        let trapdoor_artifact_name = trapdoor_artifact_name.into();
        let wire_type = WireType::Trapdoor {
            matrix: matrix_type.clone(),
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: trapdoor_artifact_name.clone(),
                wire_type: wire_type.clone(),
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: trapdoor_artifact_name,
                    confidentiality: ArtifactConfidentiality::Private,
                }),
            },
            Vec::new(),
            vec![wire_type],
        );
        Trapdoor {
            public,
            value: node.output(0).expect("trapdoor artifact input"),
            matrix_type,
            preimage_max_coefficient_bound,
            pending: Pending::default(),
        }
    }

    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn trapdoor_family_artifact_input(
        &self,
        production_id: ProductionId,
        public_artifact_name: impl Into<String>,
        trapdoor_artifact_name: impl Into<String>,
        count: impl Into<IntExpr>,
        rows: impl Into<IntExpr>,
        sigma: impl Into<RealExpr>,
        gadget_base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
        preimage_max_coefficient_bound: impl Into<IntExpr>,
    ) -> TrapdoorFamily {
        let count = count.into();
        let rows = rows.into();
        let sigma = sigma.into();
        let gadget_base = gadget_base.into();
        let digit_count = digit_count.into();
        let preimage_max_coefficient_bound = preimage_max_coefficient_bound.into();
        let matrix_type = self.matrix_type(Shape {
            rows: rows.clone(),
            columns: IntExpr::Mul(
                Box::new(rows),
                Box::new(IntExpr::Add(
                    Box::new(digit_count.clone()),
                    Box::new(IntExpr::constant(2)),
                )),
            )
            .canonicalize(),
        });
        let public_artifact_name = public_artifact_name.into();
        let public = Family::source_input(
            format!("artifact:{public_artifact_name}"),
            matrix_type.clone(),
            count.clone(),
            Some(ArtifactInput {
                production_id: production_id.clone(),
                artifact_name: public_artifact_name,
                confidentiality: ArtifactConfidentiality::Public,
            }),
        );
        let trapdoor_artifact_name = trapdoor_artifact_name.into();
        let element = TrapdoorType {
            matrix: matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        };
        TrapdoorFamily::source_input(
            format!("artifact:{trapdoor_artifact_name}"),
            public,
            element,
            count,
            Some(ArtifactInput {
                production_id,
                artifact_name: trapdoor_artifact_name,
                confidentiality: ArtifactConfidentiality::Private,
            }),
        )
    }

    #[track_caller]
    pub fn input_family(
        &self,
        name: impl Into<String>,
        count: impl Into<IntExpr>,
        shape: impl IntoShape,
    ) -> Family<Mat> {
        let element = self.matrix_type(shape);
        Family::source_input(name.into(), element, count.into(), None)
    }

    #[track_caller]
    pub fn family_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        count: impl Into<IntExpr>,
        shape: impl IntoShape,
        confidentiality: ArtifactConfidentiality,
    ) -> Family<Mat> {
        let artifact_name = artifact_name.into();
        Family::source_input(
            format!("artifact:{artifact_name}"),
            self.matrix_type(shape),
            count.into(),
            Some(ArtifactInput { production_id, artifact_name, confidentiality }),
        )
    }

    #[track_caller]
    pub fn zero(&self, shape: impl IntoShape) -> Mat {
        self.constant(shape, ConstantMatrix::Zero)
    }

    #[track_caller]
    pub fn identity(&self, size: impl Into<IntExpr>) -> Mat {
        let size = size.into();
        self.constant(Shape { rows: size.clone(), columns: size }, ConstantMatrix::Identity)
    }

    #[track_caller]
    pub fn gadget(
        &self,
        rows: impl Into<IntExpr>,
        base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
    ) -> Mat {
        let rows = rows.into();
        let base = base.into();
        let digit_count = digit_count.into();
        self.constant(
            Shape {
                rows: rows.clone(),
                columns: IntExpr::Mul(Box::new(rows), Box::new(digit_count)).canonicalize(),
            },
            ConstantMatrix::Gadget { base, small: false },
        )
    }

    #[track_caller]
    pub fn constant(&self, shape: impl IntoShape, value: ConstantMatrix) -> Mat {
        let ty = self.matrix_type(shape);
        Mat::from_node(NodeKind::ConstantMatrix { matrix_type: ty.clone(), value }, Vec::new(), ty)
    }

    #[track_caller]
    pub fn polynomial(&self, coefficients: impl IntoIterator<Item = IntExpr>) -> Mat {
        self.constant(
            (1, 1),
            ConstantMatrix::Polynomial { coefficients: coefficients.into_iter().collect() },
        )
    }

    /// Reconstructs one polynomial from canonical coefficient bits.
    ///
    /// Bits are coefficient-major and little-endian within each coefficient.
    #[track_caller]
    pub fn pack_polynomial_coefficients(&self, bits: Family<Bool>, coefficient_bits: usize) -> Mat {
        let matrix_type = self.matrix_type((1, 1));
        let pending = bits.pending;
        let node = NodeHandle::new(
            NodeKind::PackPolynomialCoefficients {
                matrix_type: matrix_type.clone(),
                coefficient_bits: IntExpr::constant(coefficient_bits),
            },
            vec![bits.value],
            vec![WireType::Matrix(matrix_type.clone())],
        );
        Mat { value: node.output(0).expect("packed polynomial"), matrix_type, pending }
    }

    #[track_caller]
    pub fn uniform(&self, shape: impl IntoShape) -> Mat {
        self.uniform_in(
            shape,
            IntExpr::constant(0),
            IntExpr::Sub(Box::new(self.modulus.clone()), Box::new(IntExpr::constant(1))),
        )
    }

    #[track_caller]
    pub fn uniform_in(
        &self,
        shape: impl IntoShape,
        minimum: impl Into<IntExpr>,
        maximum: impl Into<IntExpr>,
    ) -> Mat {
        let ty = self.matrix_type(shape);
        Mat::from_node(
            NodeKind::UniformSample {
                matrix_type: ty.clone(),
                range: SampleRange { minimum: minimum.into(), maximum: maximum.into() },
            },
            Vec::new(),
            ty,
        )
    }

    #[track_caller]
    pub fn gaussian(
        &self,
        shape: impl IntoShape,
        sigma: impl Into<RealExpr>,
        max_coefficient_bound: impl Into<IntExpr>,
    ) -> Mat {
        let ty = self.matrix_type(shape);
        Mat::from_node(
            NodeKind::GaussianSample {
                matrix_type: ty.clone(),
                sigma: sigma.into(),
                max_coefficient_bound: max_coefficient_bound.into(),
            },
            Vec::new(),
            ty,
        )
    }

    #[track_caller]
    pub fn hash_matrix(&self, key: Bytes, tag: impl Into<HashTag>, shape: impl IntoShape) -> Mat {
        self.hash(key, tag, shape, HashVariant::Plain, None, None)
    }

    #[track_caller]
    pub fn hash_decomposed(
        &self,
        key: Bytes,
        tag: impl Into<HashTag>,
        shape: impl IntoShape,
        base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
    ) -> Mat {
        self.hash(
            key,
            tag,
            shape,
            HashVariant::Decomposed,
            Some(base.into()),
            Some(digit_count.into()),
        )
    }

    #[track_caller]
    fn hash(
        &self,
        key: Bytes,
        tag: impl Into<HashTag>,
        shape: impl IntoShape,
        variant: HashVariant,
        base: Option<IntExpr>,
        digit_count: Option<IntExpr>,
    ) -> Mat {
        let ty = self.matrix_type(shape);
        let tag = tag.into();
        let pending = key.pending.clone();
        let node = NodeHandle::new(
            NodeKind::HashSample {
                matrix_type: ty.clone(),
                variant,
                tag_prefix: tag.prefix,
                tag_expressions: tag.binary,
                tag_decimal_expressions: tag.decimal,
                tag_u64_le_expressions: tag.u64_le,
                base,
                digit_count,
            },
            vec![key.value],
            vec![WireType::Matrix(ty.clone())],
        );
        Mat { value: node.output(0).expect("hash output"), matrix_type: ty, pending }
    }

    #[track_caller]
    pub fn sample_trapdoor(
        &self,
        rows: impl Into<IntExpr>,
        sigma: impl Into<RealExpr>,
        gadget_base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
        preimage_max_coefficient_bound: impl Into<IntExpr>,
    ) -> Trapdoor {
        let rows = rows.into();
        let sigma = sigma.into();
        let gadget_base = gadget_base.into();
        let digit_count = digit_count.into();
        let preimage_max_coefficient_bound = preimage_max_coefficient_bound.into();
        let matrix_type = self.matrix_type(Shape {
            rows: rows.clone(),
            columns: IntExpr::Mul(
                Box::new(rows),
                Box::new(IntExpr::Add(
                    Box::new(digit_count.clone()),
                    Box::new(IntExpr::constant(2)),
                )),
            )
            .canonicalize(),
        });
        let node = NodeHandle::new(
            NodeKind::TrapdoorSample {
                matrix_type: matrix_type.clone(),
                sigma: sigma.clone(),
                gadget_base: gadget_base.clone(),
                digit_count: digit_count.clone(),
                preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
            },
            Vec::new(),
            vec![
                WireType::Matrix(matrix_type.clone()),
                WireType::Trapdoor {
                    matrix: matrix_type.clone(),
                    sigma,
                    gadget_base,
                    digit_count,
                    preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
                },
            ],
        );
        Trapdoor {
            public: Mat {
                value: node.output(0).expect("public output"),
                matrix_type: matrix_type.clone(),
                pending: Pending::default(),
            },
            value: node.output(1).expect("trapdoor output"),
            matrix_type,
            preimage_max_coefficient_bound,
            pending: Pending::default(),
        }
    }

    pub fn bytes_input(&self, name: impl Into<String>, length: impl Into<IntExpr>) -> Bytes {
        let name = name.into();
        let ty = WireType::Bytes { length: length.into() };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: ty.clone(), artifact: None },
            Vec::new(),
            vec![ty],
        );
        Bytes { value: node.output(0).expect("bytes input"), pending: Pending::default() }
    }

    #[track_caller]
    pub fn bytes_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        length: impl Into<IntExpr>,
        confidentiality: ArtifactConfidentiality,
    ) -> Bytes {
        let artifact_name = artifact_name.into();
        let ty = WireType::Bytes { length: length.into() };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: artifact_name.clone(),
                wire_type: ty.clone(),
                artifact: Some(ArtifactInput { production_id, artifact_name, confidentiality }),
            },
            Vec::new(),
            vec![ty],
        );
        Bytes { value: node.output(0).expect("bytes artifact input"), pending: Pending::default() }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HashTag {
    prefix: Vec<u8>,
    binary: Vec<IntExpr>,
    decimal: Vec<IntExpr>,
    u64_le: Vec<IntExpr>,
}

impl HashTag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push<T: HashTagPart>(&mut self, part: T) {
        part.append_to(self);
    }

    pub fn push_decimal(&mut self, index: LoopIndex) {
        self.decimal.push(index.expression);
    }
}

impl From<Vec<u8>> for HashTag {
    fn from(prefix: Vec<u8>) -> Self {
        Self { prefix, ..Self::default() }
    }
}

impl From<&[u8]> for HashTag {
    fn from(prefix: &[u8]) -> Self {
        prefix.to_vec().into()
    }
}

pub trait HashTagPart {
    fn append_to(self, tag: &mut HashTag);
}

impl HashTagPart for &str {
    fn append_to(self, tag: &mut HashTag) {
        tag.prefix.extend_from_slice(self.as_bytes());
        tag.prefix.push(0);
    }
}

impl HashTagPart for String {
    fn append_to(self, tag: &mut HashTag) {
        self.as_str().append_to(tag);
    }
}

impl HashTagPart for LoopIndex {
    fn append_to(self, tag: &mut HashTag) {
        tag.u64_le.push(self.expression);
    }
}

impl HashTagPart for IntExpr {
    fn append_to(self, tag: &mut HashTag) {
        tag.u64_le.push(self);
    }
}

#[derive(Clone)]
pub struct Mat {
    value: ValueHandle,
    matrix_type: MatrixType,
    pending: Pending,
}

impl Mat {
    fn source_input(
        name: String,
        matrix_type: MatrixType,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let wire_type = WireType::Matrix(matrix_type.clone());
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: wire_type.clone(), artifact },
            Vec::new(),
            vec![wire_type],
        );
        Self {
            value: node.output(0).expect("matrix input"),
            matrix_type,
            pending: Pending::default(),
        }
    }

    fn from_node(kind: NodeKind, arguments: Vec<Mat>, matrix_type: MatrixType) -> Self {
        let pending = Pending::merge(arguments.iter().map(|value| value.pending.clone()));
        let arguments = arguments.into_iter().map(|value| value.value).collect();
        let node = NodeHandle::new(kind, arguments, vec![WireType::Matrix(matrix_type.clone())]);
        Self { value: node.output(0).expect("matrix output"), matrix_type, pending }
    }

    pub fn matrix_type(&self) -> &MatrixType {
        &self.matrix_type
    }

    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    #[track_caller]
    pub fn transpose(self) -> Self {
        let ty = MatrixType {
            rows: self.matrix_type.columns.clone(),
            columns: self.matrix_type.rows.clone(),
            ..self.matrix_type.clone()
        };
        Self::from_node(NodeKind::Transpose, vec![self], ty)
    }

    pub fn t(self) -> Self {
        self.transpose()
    }

    #[track_caller]
    pub fn slice(self, rows: Option<IndexRange>, columns: Option<IndexRange>) -> Self {
        let ty = MatrixType {
            rows: rows.as_ref().map_or_else(
                || self.matrix_type.rows.clone(),
                |range| {
                    IntExpr::Sub(Box::new(range.end.clone()), Box::new(range.start.clone()))
                        .canonicalize()
                },
            ),
            columns: columns.as_ref().map_or_else(
                || self.matrix_type.columns.clone(),
                |range| {
                    IntExpr::Sub(Box::new(range.end.clone()), Box::new(range.start.clone()))
                        .canonicalize()
                },
            ),
            ..self.matrix_type.clone()
        };
        Self::from_node(NodeKind::Slice { rows, columns }, vec![self], ty)
    }

    #[track_caller]
    pub fn reshape(self, rows: impl Into<IntExpr>, columns: impl Into<IntExpr>) -> Self {
        let rows = rows.into();
        let columns = columns.into();
        let ty =
            MatrixType { rows: rows.clone(), columns: columns.clone(), ..self.matrix_type.clone() };
        Self::from_node(NodeKind::Reshape { rows, columns }, vec![self], ty)
    }

    #[track_caller]
    pub fn tensor(self, rhs: Mat) -> Self {
        let ty = MatrixType {
            rows: IntExpr::Mul(
                Box::new(self.matrix_type.rows.clone()),
                Box::new(rhs.matrix_type.rows.clone()),
            ),
            columns: IntExpr::Mul(
                Box::new(self.matrix_type.columns.clone()),
                Box::new(rhs.matrix_type.columns.clone()),
            ),
            ..self.matrix_type.clone()
        };
        Self::from_node(NodeKind::Tensor, vec![self, rhs], ty)
    }

    #[track_caller]
    pub fn decompose(self, base: impl Into<IntExpr>, digit_count: impl Into<IntExpr>) -> Preimage {
        let digit_count = digit_count.into();
        let ty = MatrixType {
            rows: IntExpr::Mul(
                Box::new(self.matrix_type.rows.clone()),
                Box::new(digit_count.clone()),
            ),
            ..self.matrix_type.clone()
        };
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::GadgetDecompose {
                base: base.into(),
                small: false,
                digit_count: Some(digit_count),
            },
            vec![self.value],
            vec![WireType::Preimage(ty.clone())],
        );
        Preimage { value: node.output(0).expect("decomposition"), matrix_type: ty, pending }
    }

    #[track_caller]
    pub fn extract_coefficient(self, position: impl Into<IntExpr>) -> Int {
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::ExtractCoefficient { position: position.into() },
            vec![self.value],
            vec![WireType::Int],
        );
        Int { value: node.output(0).expect("coefficient"), pending }
    }

    /// Serializes one polynomial into canonical coefficient bits.
    ///
    /// Bits are coefficient-major and little-endian within each coefficient.
    /// Validation ensures that this matrix is scalar and every requested
    /// coefficient position is in range.
    #[track_caller]
    pub fn canonical_coefficient_bits(
        self,
        ring_dimension: usize,
        coefficient_bits: usize,
    ) -> Result<Family<Bool>, DslError> {
        let mut bits = Vec::with_capacity(ring_dimension.saturating_mul(coefficient_bits));
        for coefficient in 0..ring_dimension {
            let value = self.clone().extract_coefficient(coefficient);
            bits.extend((0..coefficient_bits).map(|bit| value.clone().bit(bit)));
        }
        Family::<Bool>::pack_bools(bits)
    }

    #[track_caller]
    pub fn constant_coefficient(self, position: impl Into<IntExpr>) -> Mat {
        let ty = self.matrix_type.clone();
        Self::from_node(NodeKind::ConstantCoefficient { position: position.into() }, vec![self], ty)
    }

    #[track_caller]
    pub fn threshold_decode_ints(
        self,
        plaintext_modulus: impl Into<IntExpr>,
        length: usize,
    ) -> Vec<Int> {
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::ThresholdDecode {
                plaintext_modulus: plaintext_modulus.into(),
                length: IntExpr::constant(length),
                output_bool: false,
            },
            vec![self.value],
            vec![WireType::Int; length],
        );
        (0..length)
            .map(|port| Int {
                value: node.output(port as u32).expect("decoded integer"),
                pending: pending.clone(),
            })
            .collect()
    }

    #[track_caller]
    pub fn threshold_decode_bools(
        self,
        plaintext_modulus: impl Into<IntExpr>,
        length: usize,
    ) -> Vec<Bool> {
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::ThresholdDecode {
                plaintext_modulus: plaintext_modulus.into(),
                length: IntExpr::constant(length),
                output_bool: true,
            },
            vec![self.value],
            vec![WireType::Bool; length],
        );
        (0..length)
            .map(|port| Bool {
                value: node.output(port as u32).expect("decoded boolean"),
                pending: pending.clone(),
            })
            .collect()
    }

    pub fn concat(axis: ConcatAxis, values: Vec<Mat>) -> Mat {
        let first = values.first().expect("concat requires at least one input").matrix_type.clone();
        let rows = match axis {
            ConcatAxis::Rows | ConcatAxis::Diagonal => values
                .iter()
                .map(|value| value.matrix_type.rows.clone())
                .reduce(|left, right| IntExpr::Add(Box::new(left), Box::new(right)))
                .expect("nonempty"),
            ConcatAxis::Columns => first.rows.clone(),
        };
        let columns = match axis {
            ConcatAxis::Columns | ConcatAxis::Diagonal => values
                .iter()
                .map(|value| value.matrix_type.columns.clone())
                .reduce(|left, right| IntExpr::Add(Box::new(left), Box::new(right)))
                .expect("nonempty"),
            ConcatAxis::Rows => first.columns.clone(),
        };
        Mat::from_node(NodeKind::Concat { axis }, values, MatrixType { rows, columns, ..first })
    }

    #[track_caller]
    pub fn crt_recompose(
        values: Vec<Mat>,
        plaintext_moduli: Vec<IntExpr>,
        reconstruction_coefficients: Vec<IntExpr>,
    ) -> Mat {
        let ty = values.first().expect("CRT recomposition requires inputs").matrix_type.clone();
        Mat::from_node(
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients },
            values,
            ty,
        )
    }
}

impl Add for Mat {
    type Output = Mat;
    #[track_caller]
    fn add(self, rhs: Self) -> Self::Output {
        let ty = self.matrix_type.clone();
        Mat::from_node(NodeKind::MatrixBinary(MatrixBinaryOp::Add), vec![self, rhs], ty)
    }
}

impl Sub for Mat {
    type Output = Mat;
    #[track_caller]
    fn sub(self, rhs: Self) -> Self::Output {
        let ty = self.matrix_type.clone();
        Mat::from_node(NodeKind::MatrixBinary(MatrixBinaryOp::Subtract), vec![self, rhs], ty)
    }
}

impl Mul for Mat {
    type Output = Mat;
    #[track_caller]
    fn mul(self, rhs: Self) -> Self::Output {
        let ty = product_type(&self.matrix_type, &rhs.matrix_type);
        Mat::from_node(NodeKind::MatrixBinary(MatrixBinaryOp::Multiply), vec![self, rhs], ty)
    }
}

impl Neg for Mat {
    type Output = Mat;
    #[track_caller]
    fn neg(self) -> Self::Output {
        let ty = self.matrix_type.clone();
        Mat::from_node(NodeKind::MatrixNegate, vec![self], ty)
    }
}

#[derive(Clone)]
pub struct Preimage {
    value: ValueHandle,
    matrix_type: MatrixType,
    pending: Pending,
}

impl Preimage {
    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn as_mat(self) -> Mat {
        let matrix_type = self.matrix_type;
        Mat::from_node(
            NodeKind::MatrixScale { scalar: IntExpr::constant(1) },
            vec![Mat {
                value: self.value,
                matrix_type: matrix_type.clone(),
                pending: self.pending,
            }],
            matrix_type,
        )
    }
}

#[derive(Clone)]
pub struct Trapdoor {
    public: Mat,
    value: ValueHandle,
    matrix_type: MatrixType,
    preimage_max_coefficient_bound: IntExpr,
    pending: Pending,
}

impl Trapdoor {
    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn public_matrix(&self) -> Mat {
        self.public.clone()
    }

    #[track_caller]
    pub fn sample_preimage(&self, target: Mat, shape: impl IntoShape) -> Preimage {
        let shape = shape.into_shape();
        let ty =
            MatrixType { rows: shape.rows, columns: shape.columns, ..self.matrix_type.clone() };
        let pending = Pending::merge([self.pending.clone(), target.pending.clone()]);
        let node = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: ty.clone(),
                max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
            },
            vec![self.public.value.clone(), self.value.clone(), target.value],
            vec![WireType::Preimage(ty.clone())],
        );
        Preimage { value: node.output(0).expect("preimage"), matrix_type: ty, pending }
    }
}

/// A dynamically sized family of trapdoors and their corresponding public matrices.
///
/// A trapdoor is represented by two core wires, so this wrapper intentionally stores two
/// parallel families rather than pretending that `Family<T>` can contain a multi-wire value.
#[derive(Clone)]
pub struct TrapdoorFamily {
    public: Family<Mat>,
    values: ValueHandle,
    element_schema: TrapdoorType,
    count: IntExpr,
    pending: Pending,
}

impl TrapdoorFamily {
    #[doc(hidden)]
    pub fn secret_value_handle(&self) -> &ValueHandle {
        &self.values
    }

    fn source_input(
        name: String,
        public: Family<Mat>,
        element_schema: TrapdoorType,
        count: IntExpr,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let element_type = WireType::Trapdoor {
            matrix: element_schema.matrix.clone(),
            sigma: element_schema.sigma.clone(),
            gadget_base: element_schema.gadget_base.clone(),
            digit_count: element_schema.digit_count.clone(),
            preimage_max_coefficient_bound: element_schema.preimage_max_coefficient_bound.clone(),
        };
        let family_type =
            WireType::IndexedFamily { element: Box::new(element_type), count: count.clone() };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: family_type.clone(), artifact },
            Vec::new(),
            vec![family_type],
        );
        Self {
            public,
            values: node.output(0).expect("trapdoor family"),
            element_schema,
            count,
            pending: Pending::default(),
        }
    }

    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    pub fn public_matrices(&self) -> Family<Mat> {
        self.public.clone()
    }

    pub fn get_static(&self, index: impl Into<IntExpr>) -> Trapdoor {
        let index = index.into();
        let public = self.public.get_static(index.clone());
        let pending = Pending::merge([self.pending.clone(), public.pending.clone()]);
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { index },
            vec![self.values.clone()],
            vec![self.element_schema.wire_types()[1].clone()],
        );
        Trapdoor::from_values(
            &self.element_schema,
            &[public.value, node.output(0).expect("trapdoor family element")],
            pending,
        )
        .expect("trapdoor family schema")
    }

    pub fn get(&self, index: Int) -> Trapdoor {
        let public = self.public.get(index.clone());
        let pending =
            Pending::merge([self.pending.clone(), public.pending.clone(), index.pending.clone()]);
        let node = NodeHandle::new(
            NodeKind::FamilyGetDynamic,
            vec![self.values.clone(), index.value],
            vec![self.element_schema.wire_types()[1].clone()],
        );
        Trapdoor::from_values(
            &self.element_schema,
            &[public.value, node.output(0).expect("dynamic trapdoor family element")],
            pending,
        )
        .expect("trapdoor family schema")
    }

    pub fn parallel_gather(self, indices: Family<Int>) -> Result<Self, DslError> {
        self.parallel_gather_traced(indices).map(|(family, _)| family)
    }

    #[doc(hidden)]
    pub fn parallel_gather_traced(
        self,
        indices: Family<Int>,
    ) -> Result<(Self, LoopConstructionTrace<GatherConstructionTrace>), DslError> {
        let source_count = self.count.clone();
        let output_count = indices.count.clone();
        let schema = self.element_schema.clone();
        let (index_slot, (body_value, raw_trace, explicit_inputs, scope)) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let index = IntType.placeholders();
                let public = Family::<Mat>::source_input(
                    "gather-trapdoor-public".to_owned(),
                    schema.matrix.clone(),
                    source_count.clone(),
                    None,
                );
                let source = TrapdoorFamily::source_input(
                    "gather-trapdoor-secret".to_owned(),
                    public,
                    schema.clone(),
                    source_count.clone(),
                    None,
                );
                let mut explicit_inputs = vec![index.value.clone()];
                explicit_inputs.extend(source.flatten());
                let output = source.get(index.clone());
                let raw_trace = GatherConstructionTrace {
                    index: index.value,
                    sources: source.flatten(),
                    outputs: output.flatten(),
                };
                (output, raw_trace, explicit_inputs, scope)
            })
        });
        let sealed = SubgraphHandle::seal(
            "parallel-gather-trapdoor-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let node = NodeHandle::parallel_loop(
            sealed.handle.clone(),
            vec![indices.value, self.public.value, self.values],
            body_value.parallel_family_types(&output_count)?,
            ParallelLoop {
                count: output_count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: vec![
                    LoopInputMode::Zip,
                    LoopInputMode::Broadcast,
                    LoopInputMode::Broadcast,
                ],
            },
        );
        let pending = Pending::merge([
            indices.pending,
            self.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        let family = body_value.parallel_families(&node, &mut 0, &output_count, pending)?;
        let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
        Ok((family, trace))
    }

    pub fn parallel_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Trapdoor) -> R,
    ) -> Result<R::Families, DslError> {
        let count = self.count.clone();
        let schema = self.element_schema.clone();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let input = schema.placeholders();
                let explicit_inputs = input.flatten();
                (body(index, input), explicit_inputs, scope)
            })
        });
        let sealed = SubgraphHandle::seal(
            "parallel-map-trapdoor-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.public.value, self.values];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::Zip];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let node = NodeHandle::parallel_loop(
            sealed.handle,
            arguments,
            body_value.parallel_family_types(&count)?,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([self.pending, body_value.pending().remap(&sealed.remap)]);
        body_value.parallel_families(&node, &mut 0, &count, pending)
    }

    pub fn parallel_zip_mat_values<R: ParallelOutput>(
        self,
        matrices: Family<Mat>,
        body: impl FnOnce(LoopIndex, Trapdoor, Mat) -> R,
    ) -> Result<R::Families, DslError> {
        self.parallel_zip_mat_values_traced(matrices, |index, trapdoor, matrix| {
            (body(index, trapdoor, matrix), ())
        })
        .map(|(families, _)| families)
    }

    #[doc(hidden)]
    pub fn parallel_zip_mat_values_traced<R: ParallelOutput, T: RemapConstructionTrace>(
        self,
        matrices: Family<Mat>,
        body: impl FnOnce(LoopIndex, Trapdoor, Mat) -> (R, T),
    ) -> Result<(R::Families, LoopConstructionTrace<T>), DslError> {
        if self.count != matrices.count {
            return Err(DslError::FamilyCountMismatch);
        }
        let count = self.count.clone();
        let trapdoor_schema = self.element_schema.clone();
        let matrix_schema = MatType(matrices.element_schema.matrix_type.clone());
        let (index_slot, (body_value, raw_trace, explicit_inputs, scope)) =
            with_loop_index(|index| {
                with_new_construction_scope(|scope| {
                    let trapdoor = trapdoor_schema.placeholders();
                    let matrix = matrix_schema.placeholders();
                    let mut explicit_inputs = trapdoor.flatten();
                    explicit_inputs.extend(matrix.flatten());
                    let (body_value, raw_trace) = body(index, trapdoor, matrix);
                    (body_value, raw_trace, explicit_inputs, scope)
                })
            });
        let sealed = SubgraphHandle::seal(
            "parallel-zip-trapdoor-mat-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.public.value, self.values, matrices.value];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::Zip, LoopInputMode::Zip];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let node = NodeHandle::parallel_loop(
            sealed.handle.clone(),
            arguments,
            body_value.parallel_family_types(&count)?,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([
            self.pending,
            matrices.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        let families = body_value.parallel_families(&node, &mut 0, &count, pending)?;
        let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
        Ok((families, trace))
    }
}

#[derive(Clone)]
pub struct Int {
    value: ValueHandle,
    pending: Pending,
}

#[derive(Clone)]
pub struct Bool {
    value: ValueHandle,
    pending: Pending,
}

impl Int {
    pub fn constant(value: impl Into<num_bigint::BigInt>) -> Self {
        let node = NodeHandle::new(
            NodeKind::ConstantInt(value.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        );
        Self { value: node.output(0).expect("constant integer"), pending: Pending::default() }
    }

    pub fn evaluate(expression: impl Into<IntExpr>) -> Self {
        let node = NodeHandle::new(
            NodeKind::EvaluateInt(expression.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        );
        Self { value: node.output(0).expect("evaluated integer"), pending: Pending::default() }
    }

    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn pending_assumptions(&self) -> bool {
        false
    }

    pub fn add(self, rhs: Self) -> Self {
        self.binary(rhs, mxx_ir_core::node::IntBinaryOp::Add, "integer sum")
    }

    pub fn sub(self, rhs: Self) -> Self {
        self.binary(rhs, mxx_ir_core::node::IntBinaryOp::Subtract, "integer difference")
    }

    pub fn mul(self, rhs: Self) -> Self {
        self.binary(rhs, mxx_ir_core::node::IntBinaryOp::Multiply, "integer product")
    }

    pub fn div(self, rhs: Self) -> Self {
        self.binary(rhs, mxx_ir_core::node::IntBinaryOp::Divide, "integer quotient")
    }

    pub fn rem(self, rhs: Self) -> Self {
        self.binary(rhs, mxx_ir_core::node::IntBinaryOp::Remainder, "integer remainder")
    }

    fn binary(
        self,
        rhs: Self,
        operation: mxx_ir_core::node::IntBinaryOp,
        output_name: &'static str,
    ) -> Self {
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::IntBinary(operation),
            vec![self.value, rhs.value],
            vec![WireType::Int],
        );
        Self { value: node.output(0).expect(output_name), pending }
    }

    pub fn equal(self, rhs: Self) -> Bool {
        self.compare(rhs, mxx_ir_core::node::IntCompareOp::Equal)
    }

    pub fn less_equal(self, rhs: Self) -> Bool {
        self.compare(rhs, mxx_ir_core::node::IntCompareOp::LessEqual)
    }

    pub fn bit(self, position: impl Into<IntExpr>) -> Bool {
        let node = NodeHandle::new(
            NodeKind::BitExtract { bit: position.into() },
            vec![self.value],
            vec![WireType::Bool],
        );
        Bool { value: node.output(0).expect("integer bit"), pending: self.pending }
    }

    fn compare(self, rhs: Self, operation: mxx_ir_core::node::IntCompareOp) -> Bool {
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::IntCompare(operation),
            vec![self.value, rhs.value],
            vec![WireType::Bool],
        );
        Bool { value: node.output(0).expect("integer comparison"), pending }
    }

    pub fn select(self, branches: Vec<Mat>) -> Result<Mat, DslError> {
        let Some(first) = branches.first() else {
            return Err(DslError::Schema);
        };
        if branches.iter().any(|branch| branch.matrix_type != first.matrix_type) {
            return Err(DslError::Schema);
        }
        let output_type = first.matrix_type.clone();
        let pending = Pending::merge(
            std::iter::once(self.pending)
                .chain(branches.iter().map(|branch| branch.pending.clone())),
        );
        let mut arguments = vec![self.value];
        arguments.extend(branches.iter().map(|branch| branch.value.clone()));
        let node = NodeHandle::new(
            NodeKind::Select { count: IntExpr::constant(branches.len()) },
            arguments,
            vec![WireType::Matrix(output_type.clone())],
        );
        Ok(Mat { value: node.output(0).expect("select output"), matrix_type: output_type, pending })
    }

    pub fn select_int(self, branches: Vec<Int>) -> Result<Int, DslError> {
        let branches = branches
            .into_iter()
            .map(|branch| {
                if matches!(branch.value.wire_type(), WireType::ConstantInt) {
                    branch.add(Int::constant(0))
                } else {
                    branch
                }
            })
            .collect::<Vec<_>>();
        let (value, pending) = select_scalar(
            self,
            branches.iter().map(|branch| (&branch.value, &branch.pending)),
            WireType::Int,
        )?;
        Ok(Int { value, pending })
    }

    pub fn select_bool(self, branches: Vec<Bool>) -> Result<Bool, DslError> {
        let branches = branches
            .into_iter()
            .map(|branch| {
                if matches!(branch.value.wire_type(), WireType::ConstantBool) {
                    branch.to_int().equal(Int::constant(1))
                } else {
                    branch
                }
            })
            .collect::<Vec<_>>();
        let (value, pending) = select_scalar(
            self,
            branches.iter().map(|branch| (&branch.value, &branch.pending)),
            WireType::Bool,
        )?;
        Ok(Bool { value, pending })
    }
}

impl Bool {
    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn constant(value: bool) -> Self {
        let node = NodeHandle::new(
            NodeKind::ConstantBool(value),
            Vec::new(),
            vec![WireType::ConstantBool],
        );
        Self { value: node.output(0).expect("constant boolean"), pending: Pending::default() }
    }

    pub fn to_int(self) -> Int {
        let node = NodeHandle::new(NodeKind::BoolToInt, vec![self.value], vec![WireType::Int]);
        Int { value: node.output(0).expect("boolean integer"), pending: self.pending }
    }
}

impl Family<Bool> {
    pub fn pack_bools(values: Vec<Bool>) -> Result<Self, DslError> {
        Self::pack(values)
    }

    pub fn get_static(&self, index: impl Into<IntExpr>) -> Bool {
        let element_type = family_element_wire_type(&self.value).unwrap_or(WireType::Bool);
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: index.into() },
            vec![self.value.clone()],
            vec![element_type],
        );
        Bool {
            value: node.output(0).expect("boolean family element"),
            pending: self.pending.clone(),
        }
    }

    pub fn get(&self, index: Int) -> Bool {
        scalar_family_get(self, index, WireType::Bool, |value, pending| Bool { value, pending })
    }

    pub fn parallel_map(
        self,
        body: impl FnOnce(LoopIndex, Bool) -> Bool,
    ) -> Result<Self, DslError> {
        scalar_parallel_map(self, "parallel-map-bool-body", WireType::Bool, body)
    }

    #[doc(hidden)]
    pub fn parallel_map_traced<T: RemapConstructionTrace>(
        self,
        body: impl FnOnce(LoopIndex, Bool) -> (Bool, T),
    ) -> Result<(Self, LoopConstructionTrace<T>), DslError> {
        scalar_parallel_map_traced(self, "parallel-map-bool-body", WireType::Bool, body)
    }

    pub fn parallel_gather(self, indices: Family<Int>) -> Result<Self, DslError> {
        scalar_parallel_gather(
            self,
            indices,
            "parallel-gather-bool-body",
            WireType::Bool,
            |value, pending| Bool { value, pending },
        )
    }

    #[doc(hidden)]
    pub fn parallel_gather_traced(
        self,
        indices: Family<Int>,
    ) -> Result<(Self, LoopConstructionTrace<GatherConstructionTrace>), DslError> {
        scalar_parallel_gather_traced(
            self,
            indices,
            "parallel-gather-bool-body",
            WireType::Bool,
            |value, pending| Bool { value, pending },
        )
    }
}

impl Family<Int> {
    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    pub fn get_static(&self, index: impl Into<IntExpr>) -> Int {
        scalar_family_get_static(self, index.into(), WireType::Int, |value, pending| Int {
            value,
            pending,
        })
    }

    pub fn get(&self, index: Int) -> Int {
        scalar_family_get(self, index, WireType::Int, |value, pending| Int { value, pending })
    }

    pub fn parallel_map(self, body: impl FnOnce(LoopIndex, Int) -> Int) -> Result<Self, DslError> {
        scalar_parallel_map(self, "parallel-map-int-body", WireType::Int, body)
    }

    #[doc(hidden)]
    pub fn parallel_map_traced<T: RemapConstructionTrace>(
        self,
        body: impl FnOnce(LoopIndex, Int) -> (Int, T),
    ) -> Result<(Self, LoopConstructionTrace<T>), DslError> {
        scalar_parallel_map_traced(self, "parallel-map-int-body", WireType::Int, body)
    }

    pub fn parallel_gather(self, indices: Family<Int>) -> Result<Self, DslError> {
        scalar_parallel_gather(
            self,
            indices,
            "parallel-gather-int-body",
            WireType::Int,
            |value, pending| Int { value, pending },
        )
    }

    #[doc(hidden)]
    pub fn parallel_gather_traced(
        self,
        indices: Family<Int>,
    ) -> Result<(Self, LoopConstructionTrace<GatherConstructionTrace>), DslError> {
        scalar_parallel_gather_traced(
            self,
            indices,
            "parallel-gather-int-body",
            WireType::Int,
            |value, pending| Int { value, pending },
        )
    }

    /// Packs consecutive little-endian bit segments into integers.
    ///
    /// Segments are evaluated independently by a parallel loop. Within one segment, a sequential
    /// scan carries `(sum, weight)` and updates it as `(sum + bit * weight, 2 * weight)`, avoiding
    /// host expansion and any dynamic exponentiation primitive.
    pub fn parallel_pack_little_endian_bits(
        self,
        segment_count: impl Into<IntExpr>,
        bits_per_segment: impl Into<IntExpr>,
    ) -> Result<Self, DslError> {
        self.parallel_pack_little_endian_bits_traced(segment_count, bits_per_segment)
            .map(|(family, _)| family)
    }

    #[doc(hidden)]
    pub fn parallel_pack_little_endian_bits_traced(
        self,
        segment_count: impl Into<IntExpr>,
        bits_per_segment: impl Into<IntExpr>,
    ) -> Result<(Self, LoopConstructionTrace<PackedBitsConstructionTrace>), DslError> {
        let segment_count = segment_count.into();
        let bits_per_segment = bits_per_segment.into();
        let expected_count =
            IntExpr::Mul(Box::new(segment_count.clone()), Box::new(bits_per_segment.clone()))
                .canonicalize();
        if self.count.canonicalize() != expected_count {
            return Err(DslError::FamilyCountMismatch);
        }
        let source_count = self.count.clone();
        let (index_slot, body_result) = with_loop_index(|segment| {
            with_new_construction_scope(|scope| {
                let family_type = WireType::IndexedFamily {
                    element: Box::new(WireType::Int),
                    count: source_count.clone(),
                };
                let family_node = NodeHandle::new(
                    NodeKind::Input {
                        name: "pack-bit-source".to_owned(),
                        wire_type: family_type.clone(),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![family_type],
                );
                let source = Family {
                    value: family_node.output(0).expect("bit source family"),
                    element_schema: Int::constant(0).add(Int::constant(0)),
                    count: source_count,
                    pending: Pending::default(),
                };
                let segment = segment.as_int();
                let segment_width = Int {
                    value: NodeHandle::new(
                        NodeKind::EvaluateInt(bits_per_segment.clone()),
                        Vec::new(),
                        vec![WireType::ConstantInt],
                    )
                    .output(0)
                    .expect("segment width"),
                    pending: Pending::default(),
                };
                Sequential::range(bits_per_segment.clone())
                    .scan_traced(
                        (Int::constant(0), Int::constant(1)),
                        source.clone(),
                        move |bit, (sum, weight), source| {
                            let source_index = segment.clone().mul(segment_width).add(bit.as_int());
                            let value = source.get(source_index);
                            let next_sum = sum.add(value.mul(weight.clone()));
                            let next_weight = weight.mul(Int::constant(2));
                            let body_outputs = vec![
                                next_sum.value_handle().clone(),
                                next_weight.value_handle().clone(),
                            ];
                            Ok(((next_sum, next_weight), body_outputs))
                        },
                    )
                    .map(|((sum, _), scan)| {
                        let output = sum.value_handle().clone();
                        (
                            sum,
                            PackedBitsConstructionTrace { scan, output },
                            vec![source.value],
                            scope,
                        )
                    })
            })
        });
        let (body_value, body_trace, explicit_inputs, scope) = body_result?;
        let sealed = SubgraphHandle::seal(
            "parallel-pack-little-endian-bits-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value];
        let mut modes = vec![LoopInputMode::Broadcast];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let node = NodeHandle::parallel_loop(
            sealed.handle.clone(),
            arguments,
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Int),
                count: segment_count.clone(),
            }],
            ParallelLoop {
                count: segment_count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([self.pending, body_value.pending().remap(&sealed.remap)]);
        let trace = loop_construction_trace(&node, &sealed, body_trace)?;
        let family = body_value.parallel_families(&node, &mut 0, &segment_count, pending)?;
        Ok((family, trace))
    }

    pub fn parallel_select_mats(
        self,
        candidates: Vec<Family<Mat>>,
    ) -> Result<Family<Mat>, DslError> {
        self.parallel_select_mats_traced(candidates).map(|(family, _)| family)
    }

    #[doc(hidden)]
    pub fn parallel_select_mats_traced(
        self,
        candidates: Vec<Family<Mat>>,
    ) -> Result<(Family<Mat>, LoopConstructionTrace<SelectConstructionTrace>), DslError> {
        let Some(first) = candidates.first() else {
            return Err(DslError::Schema);
        };
        let count = self.count.clone();
        if candidates.iter().any(|candidate| {
            candidate.count != count ||
                candidate.element_schema.matrix_type != first.element_schema.matrix_type
        }) {
            return Err(DslError::FamilyCountMismatch);
        }
        let matrix_type = first.element_schema.matrix_type.clone();
        let candidate_count = candidates.len();
        let (index_slot, body_result) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let mut next = 0;
                let selector = IntType.placeholders_from(&mut next);
                let branches = (0..candidate_count)
                    .map(|_| MatType(matrix_type.clone()).placeholders_from(&mut next))
                    .collect::<Vec<_>>();
                let mut inputs = selector.flatten();
                inputs.extend(branches.iter().flat_map(GraphValue::flatten));
                let selector_handle = selector.value.clone();
                let branch_handles = branches.iter().flat_map(GraphValue::flatten).collect();
                selector.select(branches).map(|output| {
                    let trace = SelectConstructionTrace {
                        selector: selector_handle,
                        branches: branch_handles,
                        output: output.value.clone(),
                    };
                    (output, trace, inputs, scope)
                })
            })
        });
        let (body_value, raw_trace, explicit_inputs, scope) = body_result?;
        let sealed = SubgraphHandle::seal(
            "parallel-select-mats-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value];
        arguments.extend(candidates.iter().map(|candidate| candidate.value.clone()));
        let mut modes = vec![LoopInputMode::Zip; arguments.len()];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let node = NodeHandle::parallel_loop(
            sealed.handle.clone(),
            arguments,
            body_value.parallel_family_types(&count)?,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge(
            std::iter::once(self.pending)
                .chain(candidates.into_iter().map(|candidate| candidate.pending))
                .chain(std::iter::once(body_value.pending().remap(&sealed.remap))),
        );
        let family = body_value.parallel_families(&node, &mut 0, &count, pending)?;
        let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
        Ok((family, trace))
    }
}

#[derive(Clone)]
pub struct Bytes {
    value: ValueHandle,
    pending: Pending,
}

impl Bytes {
    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }
}

#[derive(Clone)]
pub struct Family<T> {
    value: ValueHandle,
    element_schema: T,
    count: IntExpr,
    pending: Pending,
}

#[doc(hidden)]
pub trait FamilyElement: GraphValue {
    fn normalize_for_family(self) -> Self;
}

impl FamilyElement for Mat {
    fn normalize_for_family(self) -> Self {
        self
    }
}

impl FamilyElement for Int {
    fn normalize_for_family(self) -> Self {
        if matches!(self.value.wire_type(), WireType::ConstantInt) {
            self.add(Int::constant(0))
        } else {
            self
        }
    }
}

impl FamilyElement for Bool {
    fn normalize_for_family(self) -> Self {
        if matches!(self.value.wire_type(), WireType::ConstantBool) {
            self.to_int().equal(Int::constant(1))
        } else {
            self
        }
    }
}

impl<T: FamilyElement> Family<T> {
    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn pack(values: Vec<T>) -> Result<Self, DslError> {
        let values =
            values.into_iter().map(FamilyElement::normalize_for_family).collect::<Vec<_>>();
        let Some(first) = values.first() else {
            return Err(DslError::Schema);
        };
        let first_values = first.flatten();
        let [first_value] = first_values.as_slice() else {
            return Err(DslError::Schema);
        };
        if values.iter().any(|value| {
            let flattened = value.flatten();
            flattened.len() != 1 || flattened[0].wire_type() != first_value.wire_type()
        }) {
            return Err(DslError::Schema);
        }
        let count = IntExpr::constant(values.len());
        let pending = Pending::merge(values.iter().map(GraphValue::pending));
        let arguments = values.iter().flat_map(GraphValue::flatten).collect();
        let node = NodeHandle::new(
            NodeKind::FamilyPack { count: count.clone() },
            arguments,
            vec![WireType::IndexedFamily {
                element: Box::new(first_value.wire_type().clone()),
                count: count.clone(),
            }],
        );
        Ok(Self {
            value: node.output(0).expect("packed family"),
            element_schema: first.clone(),
            count,
            pending,
        })
    }
}

fn select_scalar<'a>(
    selector: Int,
    branches: impl IntoIterator<Item = (&'a ValueHandle, &'a Pending)>,
    output_type: WireType,
) -> Result<(ValueHandle, Pending), DslError> {
    let branches = branches.into_iter().collect::<Vec<_>>();
    if branches.is_empty() {
        return Err(DslError::Schema);
    }
    let pending = Pending::merge(
        std::iter::once(selector.pending)
            .chain(branches.iter().map(|(_, pending)| (*pending).clone())),
    );
    let mut arguments = vec![selector.value];
    arguments.extend(branches.iter().map(|(value, _)| (*value).clone()));
    let node = NodeHandle::new(
        NodeKind::Select { count: IntExpr::constant(branches.len()) },
        arguments,
        vec![output_type],
    );
    Ok((node.output(0).expect("scalar select output"), pending))
}

fn scalar_family_get_static<T>(
    family: &Family<T>,
    index: IntExpr,
    fallback_wire_type: WireType,
    parts: impl Fn(ValueHandle, Pending) -> T,
) -> T {
    let node = NodeHandle::new(
        NodeKind::FamilyGetStatic { index },
        vec![family.value.clone()],
        vec![family_element_wire_type(&family.value).unwrap_or(fallback_wire_type)],
    );
    parts(node.output(0).expect("scalar family element"), family.pending.clone())
}

fn scalar_family_get<T>(
    family: &Family<T>,
    index: Int,
    fallback_wire_type: WireType,
    parts: impl Fn(ValueHandle, Pending) -> T,
) -> T {
    let pending = Pending::merge([family.pending.clone(), index.pending]);
    let node = NodeHandle::new(
        NodeKind::FamilyGetDynamic,
        vec![family.value.clone(), index.value],
        vec![family_element_wire_type(&family.value).unwrap_or(fallback_wire_type)],
    );
    parts(node.output(0).expect("dynamic scalar family element"), pending)
}

fn scalar_parallel_map<T>(
    family: Family<T>,
    body_name: &'static str,
    wire_type: WireType,
    body: impl FnOnce(LoopIndex, T) -> T,
) -> Result<Family<T>, DslError>
where
    T: ParallelOutput<Families = Family<T>> + FamilyElement + GraphValue + Clone,
    T::Schema: GraphValueSchema<Value = T>,
{
    scalar_parallel_map_traced(family, body_name, wire_type, |index, value| {
        (body(index, value), ())
    })
    .map(|(family, _)| family)
}

fn scalar_parallel_map_traced<T, U>(
    family: Family<T>,
    body_name: &'static str,
    wire_type: WireType,
    body: impl FnOnce(LoopIndex, T) -> (T, U),
) -> Result<(Family<T>, LoopConstructionTrace<U>), DslError>
where
    T: ParallelOutput<Families = Family<T>> + FamilyElement + GraphValue + Clone,
    T::Schema: GraphValueSchema<Value = T>,
    U: RemapConstructionTrace,
{
    let count = family.count.clone();
    let schema = family.element_schema.schema();
    let (index_slot, (body_value, raw_trace, explicit_input, scope)) = with_loop_index(|index| {
        with_new_construction_scope(|scope| {
            let input = schema.placeholders();
            let input_values = input.flatten();
            let [input_value] = input_values.as_slice() else {
                panic!("scalar family item schema must contain one value")
            };
            let (output, raw_trace) = body(index, input);
            (output.normalize_for_family(), raw_trace, input_value.clone(), scope)
        })
    });
    let sealed = SubgraphHandle::seal(
        body_name,
        scope,
        vec![explicit_input],
        body_value.flatten(),
        CapturePolicy::BroadcastScalarsAndArtifactFamilies,
    )?;
    let mut arguments = vec![family.value];
    let mut modes = vec![LoopInputMode::Zip];
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
    let node = NodeHandle::parallel_loop(
        sealed.handle.clone(),
        arguments,
        vec![WireType::IndexedFamily { element: Box::new(wire_type), count: count.clone() }],
        ParallelLoop {
            count: count.clone(),
            minimum_count: 0,
            index_slot,
            bindings: Vec::new(),
            input_modes: modes,
        },
    );
    let pending = Pending::merge([family.pending, body_value.pending().remap(&sealed.remap)]);
    let family = body_value.parallel_families(&node, &mut 0, &count, pending)?;
    let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
    Ok((family, trace))
}

fn scalar_parallel_gather<T>(
    source: Family<T>,
    indices: Family<Int>,
    body_name: &'static str,
    wire_type: WireType,
    parts: impl Fn(ValueHandle, Pending) -> T + Copy,
) -> Result<Family<T>, DslError>
where
    T: ParallelOutput<Families = Family<T>> + GraphValue + Clone,
{
    scalar_parallel_gather_traced(source, indices, body_name, wire_type, parts)
        .map(|(family, _)| family)
}

fn scalar_parallel_gather_traced<T>(
    source: Family<T>,
    indices: Family<Int>,
    body_name: &'static str,
    wire_type: WireType,
    parts: impl Fn(ValueHandle, Pending) -> T + Copy,
) -> Result<(Family<T>, LoopConstructionTrace<GatherConstructionTrace>), DslError>
where
    T: ParallelOutput<Families = Family<T>> + GraphValue + Clone,
{
    let source_count = source.count.clone();
    let output_count = indices.count.clone();
    let source_element_type = family_element_wire_type(&source.value).unwrap_or(wire_type.clone());
    let (index_slot, (body_value, raw_trace, explicit_inputs, scope)) = with_loop_index(|_| {
        with_new_construction_scope(|scope| {
            let index = IntType.placeholders();
            let family_wire_type = WireType::IndexedFamily {
                element: Box::new(source_element_type.clone()),
                count: source_count.clone(),
            };
            let family_node = NodeHandle::new(
                NodeKind::Input {
                    name: "gather-source".to_owned(),
                    wire_type: family_wire_type.clone(),
                    artifact: None,
                },
                Vec::new(),
                vec![family_wire_type],
            );
            let placeholder_value = NodeHandle::new(
                NodeKind::Input {
                    name: "gather-element".to_owned(),
                    wire_type: source_element_type.clone(),
                    artifact: None,
                },
                Vec::new(),
                vec![source_element_type.clone()],
            )
            .output(0)
            .expect("gather element");
            let family = Family {
                value: family_node.output(0).expect("gather family"),
                element_schema: parts(placeholder_value, Pending::default()),
                count: source_count,
                pending: Pending::default(),
            };
            let output =
                scalar_family_get(&family, index.clone(), source_element_type.clone(), parts);
            let raw_trace = GatherConstructionTrace {
                index: index.value.clone(),
                sources: vec![family.value.clone()],
                outputs: output.flatten(),
            };
            (output, raw_trace, vec![index.value, family.value], scope)
        })
    });
    let sealed = SubgraphHandle::seal(
        body_name,
        scope,
        explicit_inputs,
        body_value.flatten(),
        CapturePolicy::BroadcastScalarsAndArtifactFamilies,
    )?;
    let node = NodeHandle::parallel_loop(
        sealed.handle.clone(),
        vec![indices.value, source.value],
        vec![WireType::IndexedFamily {
            element: Box::new(source_element_type),
            count: output_count.clone(),
        }],
        ParallelLoop {
            count: output_count.clone(),
            minimum_count: 0,
            index_slot,
            bindings: Vec::new(),
            input_modes: vec![LoopInputMode::Zip, LoopInputMode::Broadcast],
        },
    );
    let pending = Pending::merge([
        indices.pending,
        source.pending,
        body_value.pending().remap(&sealed.remap),
    ]);
    let family = body_value.parallel_families(&node, &mut 0, &output_count, pending)?;
    let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
    Ok((family, trace))
}

fn family_element_wire_type(value: &ValueHandle) -> Option<WireType> {
    let WireType::IndexedFamily { element, .. } = value.wire_type() else {
        return None;
    };
    Some((**element).clone())
}

impl Family<Mat> {
    fn source_input(
        name: String,
        matrix_type: MatrixType,
        count: IntExpr,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let element_type = WireType::Matrix(matrix_type.clone());
        let family_type =
            WireType::IndexedFamily { element: Box::new(element_type), count: count.clone() };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: family_type.clone(), artifact },
            Vec::new(),
            vec![family_type],
        );
        let placeholder = Mat::source_input("__family_element".to_owned(), matrix_type, None);
        Self {
            value: node.output(0).expect("family"),
            element_schema: placeholder,
            count,
            pending: Pending::default(),
        }
    }

    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    pub fn element_type(&self) -> &MatrixType {
        &self.element_schema.matrix_type
    }

    pub fn get_static(&self, index: impl Into<IntExpr>) -> Mat {
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: index.into() },
            vec![self.value.clone()],
            vec![WireType::Matrix(self.element_schema.matrix_type.clone())],
        );
        Mat {
            value: node.output(0).expect("family element"),
            matrix_type: self.element_schema.matrix_type.clone(),
            pending: self.pending.clone(),
        }
    }

    pub fn get(&self, index: Int) -> Mat {
        let pending = Pending::merge([self.pending.clone(), index.pending]);
        let node = NodeHandle::new(
            NodeKind::FamilyGetDynamic,
            vec![self.value.clone(), index.value],
            vec![WireType::Matrix(self.element_schema.matrix_type.clone())],
        );
        Mat {
            value: node.output(0).expect("dynamic family element"),
            matrix_type: self.element_schema.matrix_type.clone(),
            pending,
        }
    }

    pub fn parallel_map(self, body: impl FnOnce(LoopIndex, Mat) -> Mat) -> Result<Self, DslError> {
        self.parallel_map_values(body)
    }

    pub fn parallel_gather(self, indices: Family<Int>) -> Result<Self, DslError> {
        self.parallel_gather_traced(indices).map(|(family, _)| family)
    }

    #[doc(hidden)]
    pub fn parallel_gather_traced(
        self,
        indices: Family<Int>,
    ) -> Result<(Self, LoopConstructionTrace<GatherConstructionTrace>), DslError> {
        let source_count = self.count.clone();
        let output_count = indices.count.clone();
        let matrix_type = self.element_schema.matrix_type.clone();
        let (index_slot, (body_value, raw_trace, explicit_inputs, scope)) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let index = IntType.placeholders();
                let source = Family::<Mat>::source_input(
                    "gather-source".to_owned(),
                    matrix_type,
                    source_count,
                    None,
                );
                let output = source.get(index.clone());
                let raw_trace = GatherConstructionTrace {
                    index: index.value.clone(),
                    sources: vec![source.value.clone()],
                    outputs: vec![output.value.clone()],
                };
                (output, raw_trace, vec![index.value, source.value], scope)
            })
        });
        let sealed = SubgraphHandle::seal(
            "parallel-gather-mat-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let node = NodeHandle::parallel_loop(
            sealed.handle.clone(),
            vec![indices.value, self.value],
            body_value.parallel_family_types(&output_count)?,
            ParallelLoop {
                count: output_count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: vec![LoopInputMode::Zip, LoopInputMode::Broadcast],
            },
        );
        let pending = Pending::merge([
            indices.pending,
            self.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        let family = body_value.parallel_families(&node, &mut 0, &output_count, pending)?;
        let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
        Ok((family, trace))
    }

    pub fn parallel_zip_many_values<R: ParallelOutput>(
        families: Vec<Self>,
        body: impl FnOnce(LoopIndex, Vec<Mat>) -> R,
    ) -> Result<R::Families, DslError> {
        let Some(first) = families.first() else {
            return Err(DslError::Schema);
        };
        let count = first.count.clone();
        if families.iter().any(|family| family.count != count) {
            return Err(DslError::FamilyCountMismatch);
        }
        let element_types = families
            .iter()
            .map(|family| family.element_schema.matrix_type.clone())
            .collect::<Vec<_>>();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let inputs = element_types
                    .into_iter()
                    .enumerate()
                    .map(|(index, matrix_type)| {
                        Mat::source_input(format!("item-{index}"), matrix_type, None)
                    })
                    .collect::<Vec<_>>();
                let explicit_inputs = inputs.iter().map(|input| input.value.clone()).collect();
                let output = body(index, inputs);
                (output, explicit_inputs, scope)
            })
        });
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-zip-many-body",
            scope,
            explicit_inputs,
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = families.iter().map(|family| family.value.clone()).collect::<Vec<_>>();
        let mut modes = vec![LoopInputMode::Zip; families.len()];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_loop(
            sealed.handle,
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge(
            families
                .into_iter()
                .map(|family| family.pending)
                .chain(std::iter::once(body_value.pending().remap(&sealed.remap))),
        );
        let mut next_port = 0;
        body_value.parallel_families(&node, &mut next_port, &count, pending)
    }

    pub fn parallel_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Mat) -> R,
    ) -> Result<R::Families, DslError> {
        self.parallel_map_values_traced(|index, value| (body(index, value), ()))
            .map(|(families, _)| families)
    }

    #[doc(hidden)]
    pub fn parallel_map_values_traced<R: ParallelOutput, T: RemapConstructionTrace>(
        self,
        body: impl FnOnce(LoopIndex, Mat) -> (R, T),
    ) -> Result<(R::Families, LoopConstructionTrace<T>), DslError> {
        let outer_family = self.value.clone();
        let count = self.count.clone();
        let element_type = self.element_schema.matrix_type.clone();
        let (index_slot, (body_value, raw_trace, explicit_input, scope)) =
            with_loop_index(|index| {
                with_new_construction_scope(|scope| {
                    let input = Mat::source_input("item".to_owned(), element_type, None);
                    let (output, raw_trace) = body(index, input.clone());
                    (output, raw_trace, input.value, scope)
                })
            });
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-map-body",
            scope,
            vec![explicit_input],
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![outer_family];
        let mut modes = vec![LoopInputMode::Zip];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_loop(
            sealed.handle.clone(),
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([self.pending, body_value.pending().remap(&sealed.remap)]);
        let mut next_port = 0;
        let families = body_value.parallel_families(&node, &mut next_port, &count, pending)?;
        let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
        Ok((families, trace))
    }

    pub fn parallel_threshold_decode_ints(
        self,
        plaintext_modulus: impl Into<IntExpr>,
        length: usize,
    ) -> Result<Vec<Family<Int>>, DslError> {
        self.parallel_threshold_decode(plaintext_modulus.into(), length, false).map(|values| {
            values
                .into_iter()
                .map(|family| Family {
                    value: family.value,
                    element_schema: Int {
                        value: family.element_schema.value,
                        pending: Pending::default(),
                    },
                    count: family.count,
                    pending: family.pending,
                })
                .collect()
        })
    }

    pub fn parallel_threshold_decode_bools(
        self,
        plaintext_modulus: impl Into<IntExpr>,
        length: usize,
    ) -> Result<Vec<Family<Bool>>, DslError> {
        let count = self.count.clone();
        let element_type = self.element_schema.matrix_type.clone();
        let modulus = plaintext_modulus.into();
        let (index_slot, (outputs, input, scope)) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let input = Mat::source_input("item".to_owned(), element_type, None);
                let outputs = input.clone().threshold_decode_bools(modulus, length);
                (outputs, input.value, scope)
            })
        });
        let output_values = outputs.iter().flat_map(GraphValue::flatten).collect();
        let sealed = SubgraphHandle::seal(
            "parallel-decode-bools-body",
            scope,
            vec![input],
            output_values,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let node = parallel_unary_node(
            &self,
            &sealed,
            index_slot,
            vec![
                WireType::IndexedFamily { element: Box::new(WireType::Bool), count: count.clone() };
                length
            ],
        );
        let pending = Pending::merge([
            self.pending,
            Pending::merge(outputs.iter().map(|x| x.pending.clone())).remap(&sealed.remap),
        ]);
        Ok(outputs
            .into_iter()
            .enumerate()
            .map(|(port, output)| Family {
                value: node.output(port as u32).expect("boolean family"),
                element_schema: output,
                count: count.clone(),
                pending: pending.clone(),
            })
            .collect())
    }

    fn parallel_threshold_decode(
        self,
        plaintext_modulus: IntExpr,
        length: usize,
        _output_bool: bool,
    ) -> Result<Vec<Family<Int>>, DslError> {
        let count = self.count.clone();
        let element_type = self.element_schema.matrix_type.clone();
        let (index_slot, (outputs, input, scope)) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let input = Mat::source_input("item".to_owned(), element_type, None);
                let outputs = input.clone().threshold_decode_ints(plaintext_modulus, length);
                (outputs, input.value, scope)
            })
        });
        let output_values = outputs.iter().flat_map(GraphValue::flatten).collect();
        let sealed = SubgraphHandle::seal(
            "parallel-decode-ints-body",
            scope,
            vec![input],
            output_values,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let node = parallel_unary_node(
            &self,
            &sealed,
            index_slot,
            vec![
                WireType::IndexedFamily { element: Box::new(WireType::Int), count: count.clone() };
                length
            ],
        );
        let pending = Pending::merge([
            self.pending,
            Pending::merge(outputs.iter().map(|x| x.pending.clone())).remap(&sealed.remap),
        ]);
        Ok(outputs
            .into_iter()
            .enumerate()
            .map(|(port, output)| Family {
                value: node.output(port as u32).expect("integer family"),
                element_schema: output,
                count: count.clone(),
                pending: pending.clone(),
            })
            .collect())
    }

    pub fn parallel_zip(
        self,
        other: Family<Mat>,
        body: impl FnOnce(LoopIndex, Mat, Mat) -> Mat,
    ) -> Result<Self, DslError> {
        self.parallel_zip_values(other, body)
    }

    pub fn parallel_zip_values<R: ParallelOutput>(
        self,
        other: Family<Mat>,
        body: impl FnOnce(LoopIndex, Mat, Mat) -> R,
    ) -> Result<R::Families, DslError> {
        if self.count != other.count {
            return Err(DslError::FamilyCountMismatch);
        }
        let count = self.count.clone();
        let left_type = self.element_schema.matrix_type.clone();
        let right_type = other.element_schema.matrix_type.clone();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let left = Mat::source_input("left".to_owned(), left_type, None);
                let right = Mat::source_input("right".to_owned(), right_type, None);
                let output = body(index, left.clone(), right.clone());
                (output, vec![left.value, right.value], scope)
            })
        });
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-zip-body",
            scope,
            explicit_inputs,
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value, other.value];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::Zip];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_loop(
            sealed.handle,
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([
            self.pending,
            other.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        let mut next_port = 0;
        body_value.parallel_families(&node, &mut next_port, &count, pending)
    }

    pub fn parallel_zip_offset(
        self,
        other: Family<Mat>,
        offset: usize,
        body: impl FnOnce(LoopIndex, Mat, Mat) -> Mat,
    ) -> Result<Self, DslError> {
        self.parallel_zip_offset_values(other, offset, body)
    }

    pub fn parallel_zip_offset_values<R: ParallelOutput>(
        self,
        other: Family<Mat>,
        offset: usize,
        body: impl FnOnce(LoopIndex, Mat, Mat) -> R,
    ) -> Result<R::Families, DslError> {
        let count = self.count.clone();
        let left_type = self.element_schema.matrix_type.clone();
        let right_type = other.element_schema.matrix_type.clone();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let left = Mat::source_input("left".to_owned(), left_type, None);
                let right = Mat::source_input("right".to_owned(), right_type, None);
                let output = body(index, left.clone(), right.clone());
                (output, vec![left.value, right.value], scope)
            })
        });
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-zip-offset-body",
            scope,
            explicit_inputs,
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value, other.value];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::ZipOffset { offset }];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_loop(
            sealed.handle,
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([
            self.pending,
            other.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        let mut next_port = 0;
        body_value.parallel_families(&node, &mut next_port, &count, pending)
    }

    pub fn parallel_zip3(
        self,
        second: Family<Mat>,
        third: Family<Mat>,
        body: impl FnOnce(LoopIndex, Mat, Mat, Mat) -> Mat,
    ) -> Result<Self, DslError> {
        self.parallel_zip3_values(second, third, body)
    }

    pub fn parallel_zip3_values<R: ParallelOutput>(
        self,
        second: Family<Mat>,
        third: Family<Mat>,
        body: impl FnOnce(LoopIndex, Mat, Mat, Mat) -> R,
    ) -> Result<R::Families, DslError> {
        if self.count != second.count || self.count != third.count {
            return Err(DslError::FamilyCountMismatch);
        }
        let count = self.count.clone();
        let first_type = self.element_schema.matrix_type.clone();
        let second_type = second.element_schema.matrix_type.clone();
        let third_type = third.element_schema.matrix_type.clone();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let first = Mat::source_input("first".to_owned(), first_type, None);
                let second = Mat::source_input("second".to_owned(), second_type, None);
                let third = Mat::source_input("third".to_owned(), third_type, None);
                let output = body(index, first.clone(), second.clone(), third.clone());
                (output, vec![first.value, second.value, third.value], scope)
            })
        });
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-zip3-body",
            scope,
            explicit_inputs,
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value, second.value, third.value];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::Zip, LoopInputMode::Zip];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_loop(
            sealed.handle,
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([
            self.pending,
            second.pending,
            third.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        let mut next_port = 0;
        body_value.parallel_families(&node, &mut next_port, &count, pending)
    }
}

fn parallel_unary_node(
    input: &Family<Mat>,
    sealed: &SealedSubgraph,
    index_slot: u32,
    output_types: Vec<WireType>,
) -> NodeHandle {
    let mut arguments = vec![input.value.clone()];
    let mut modes = vec![LoopInputMode::Zip];
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
    NodeHandle::parallel_loop(
        sealed.handle.clone(),
        arguments,
        output_types,
        ParallelLoop {
            count: input.count.clone(),
            minimum_count: 0,
            index_slot,
            bindings: Vec::new(),
            input_modes: modes,
        },
    )
}

#[derive(Clone, Debug)]
pub struct LoopIndex {
    expression: IntExpr,
}

pub struct ParallelRange {
    count: IntExpr,
}

pub struct Parallel;

pub struct SequentialRange {
    count: IntExpr,
}

pub struct Sequential;

impl Parallel {
    pub fn range(count: impl Into<IntExpr>) -> ParallelRange {
        ParallelRange { count: count.into() }
    }
}

impl Sequential {
    pub fn range(count: impl Into<IntExpr>) -> SequentialRange {
        SequentialRange { count: count.into() }
    }
}

impl SequentialRange {
    /// Builds a sequential carried-state loop.
    ///
    /// `invariants` are explicit body inputs so ordinary executable families can be read at a
    /// dynamic layer index without relying on closure capture. The body must return exactly the
    /// same flattened wire types as `initial`; only the final carried state is returned.
    pub fn scan<S, I>(
        self,
        initial: S,
        invariants: I,
        body: impl FnOnce(LoopIndex, S, I) -> Result<S, DslError>,
    ) -> Result<S, DslError>
    where
        S: GraphValue,
        I: GraphValue,
    {
        self.scan_traced(initial, invariants, |index, state, invariants| {
            body(index, state, invariants).map(|state| (state, ()))
        })
        .map(|(state, _)| state)
    }

    #[doc(hidden)]
    pub fn scan_traced<S, I, T>(
        self,
        initial: S,
        invariants: I,
        body: impl FnOnce(LoopIndex, S, I) -> Result<(S, T), DslError>,
    ) -> Result<(S, LoopConstructionTrace<T>), DslError>
    where
        S: GraphValue,
        I: GraphValue,
        T: RemapConstructionTrace,
    {
        let count = self.count;
        let state_schema = initial.schema();
        let invariant_schema = invariants.schema();
        let state_types = state_schema.wire_types();
        if state_types.is_empty() {
            return Err(DslError::Schema);
        }

        let (index_slot, body_result) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let mut next_argument = 0;
                let state = state_schema.placeholders_from(&mut next_argument);
                let invariant_values = invariant_schema.placeholders_from(&mut next_argument);
                let mut explicit_inputs = state.flatten();
                explicit_inputs.extend(invariant_values.flatten());
                body(index, state, invariant_values)
                    .map(|(next_state, trace)| (next_state, trace, explicit_inputs, scope))
            })
        });
        let (next_state, raw_trace, explicit_inputs, scope) = body_result?;
        if next_state.schema().wire_types() != state_types {
            return Err(DslError::Schema);
        }
        let sealed = SubgraphHandle::seal(
            "sequential-scan-body",
            scope,
            explicit_inputs,
            next_state.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;

        let mut arguments = initial.flatten();
        arguments.extend(invariants.flatten());
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let node = NodeHandle::sequential_loop(
            sealed.handle.clone(),
            arguments,
            state_types.clone(),
            SequentialLoop {
                count,
                index_slot,
                bindings: Vec::new(),
                carried_count: state_types.len(),
            },
        );
        let pending = Pending::merge([
            initial.pending(),
            invariants.pending(),
            next_state.pending().remap(&sealed.remap),
        ]);
        let values = (0..state_types.len())
            .map(|port| node.output(port as u32).ok_or(DslError::Schema))
            .collect::<Result<Vec<_>, _>>()?;
        let state = S::from_values(&state_schema, &values, pending)?;
        let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
        Ok((state, trace))
    }
}

impl ParallelRange {
    pub fn map(self, body: impl FnOnce(LoopIndex) -> Mat) -> Result<Family<Mat>, DslError> {
        self.map_values(body)
    }

    pub fn map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex) -> R,
    ) -> Result<R::Families, DslError> {
        self.map_values_traced(|index| (body(index), ())).map(|(families, _)| families)
    }

    #[doc(hidden)]
    pub fn map_values_traced<R: ParallelOutput, T: RemapConstructionTrace>(
        self,
        body: impl FnOnce(LoopIndex) -> (R, T),
    ) -> Result<(R::Families, LoopConstructionTrace<T>), DslError> {
        let count = self.count;
        let (index_slot, (body_value, raw_trace, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let (body_value, raw_trace) = body(index);
                (body_value, raw_trace, scope)
            })
        });
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-range-body",
            scope,
            Vec::new(),
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let arguments = sealed.captures.iter().map(|capture| capture.outer.clone()).collect();
        let modes = (0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast).collect();
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_loop(
            sealed.handle.clone(),
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = body_value.pending().remap(&sealed.remap);
        let families = body_value.parallel_families(&node, &mut 0, &count, pending)?;
        let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
        Ok((families, trace))
    }
}

pub trait ParallelZipTuple {
    fn parallel_zip_tuple<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> R,
    ) -> Result<R::Families, DslError>;
    fn parallel_zip_tuple_result<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> Result<R, DslError>,
    ) -> Result<R::Families, DslError>;
    #[doc(hidden)]
    fn parallel_zip_tuple_result_traced<R: ParallelOutput, T: RemapConstructionTrace>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> Result<(R, T), DslError>,
    ) -> Result<(R::Families, LoopConstructionTrace<T>), DslError>;
    type Items;
}

impl<A, B> ParallelZipTuple for (Family<A>, Family<B>)
where
    A: GraphValue,
    B: GraphValue,
{
    type Items = (A, B);

    fn parallel_zip_tuple<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> R,
    ) -> Result<R::Families, DslError> {
        self.parallel_zip_tuple_result(|index, items| Ok(body(index, items)))
    }

    fn parallel_zip_tuple_result<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> Result<R, DslError>,
    ) -> Result<R::Families, DslError> {
        self.parallel_zip_tuple_result_traced(|index, items| {
            body(index, items).map(|output| (output, ()))
        })
        .map(|(families, _)| families)
    }

    fn parallel_zip_tuple_result_traced<R: ParallelOutput, T: RemapConstructionTrace>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> Result<(R, T), DslError>,
    ) -> Result<(R::Families, LoopConstructionTrace<T>), DslError> {
        if self.0.count != self.1.count {
            return Err(DslError::FamilyCountMismatch);
        }
        let count = self.0.count.clone();
        let first_schema = self.0.element_schema.schema();
        let second_schema = self.1.element_schema.schema();
        let (index_slot, body_result) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let mut next = 0;
                let first = first_schema.placeholders_from(&mut next);
                let second = second_schema.placeholders_from(&mut next);
                let mut explicit_inputs = first.flatten();
                explicit_inputs.extend(second.flatten());
                let (output, trace) = body(index, (first, second))?;
                Ok::<_, DslError>((output, trace, explicit_inputs, scope))
            })
        });
        let (body_value, trace, explicit_inputs, scope) = body_result?;
        if explicit_inputs.len() != 2 {
            return Err(DslError::Schema);
        }
        finish_parallel_zip_traced(
            count,
            vec![self.0.value, self.1.value],
            vec![self.0.pending, self.1.pending],
            body_value,
            explicit_inputs,
            scope,
            index_slot,
            "parallel-zip-bundle2-body",
            trace,
        )
    }
}

impl<A, B, C> ParallelZipTuple for (Family<A>, Family<B>, Family<C>)
where
    A: GraphValue,
    B: GraphValue,
    C: GraphValue,
{
    type Items = (A, B, C);

    fn parallel_zip_tuple<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> R,
    ) -> Result<R::Families, DslError> {
        self.parallel_zip_tuple_result(|index, items| Ok(body(index, items)))
    }

    fn parallel_zip_tuple_result<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> Result<R, DslError>,
    ) -> Result<R::Families, DslError> {
        self.parallel_zip_tuple_result_traced(|index, items| {
            body(index, items).map(|output| (output, ()))
        })
        .map(|(families, _)| families)
    }

    fn parallel_zip_tuple_result_traced<R: ParallelOutput, T: RemapConstructionTrace>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> Result<(R, T), DslError>,
    ) -> Result<(R::Families, LoopConstructionTrace<T>), DslError> {
        if self.0.count != self.1.count || self.0.count != self.2.count {
            return Err(DslError::FamilyCountMismatch);
        }
        let count = self.0.count.clone();
        let first_schema = self.0.element_schema.schema();
        let second_schema = self.1.element_schema.schema();
        let third_schema = self.2.element_schema.schema();
        let (index_slot, body_result) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let mut next = 0;
                let first = first_schema.placeholders_from(&mut next);
                let second = second_schema.placeholders_from(&mut next);
                let third = third_schema.placeholders_from(&mut next);
                let mut explicit_inputs = first.flatten();
                explicit_inputs.extend(second.flatten());
                explicit_inputs.extend(third.flatten());
                let (output, trace) = body(index, (first, second, third))?;
                Ok::<_, DslError>((output, trace, explicit_inputs, scope))
            })
        });
        let (body_value, trace, explicit_inputs, scope) = body_result?;
        if explicit_inputs.len() != 3 {
            return Err(DslError::Schema);
        }
        finish_parallel_zip_traced(
            count,
            vec![self.0.value, self.1.value, self.2.value],
            vec![self.0.pending, self.1.pending, self.2.pending],
            body_value,
            explicit_inputs,
            scope,
            index_slot,
            "parallel-zip-bundle3-body",
            trace,
        )
    }
}

fn finish_parallel_zip_traced<R: ParallelOutput, T: RemapConstructionTrace>(
    count: IntExpr,
    mut arguments: Vec<ValueHandle>,
    pendings: Vec<Pending>,
    body_value: R,
    explicit_inputs: Vec<ValueHandle>,
    scope: mxx_ir_core::ConstructionScopeId,
    index_slot: u32,
    body_name: &'static str,
    raw_trace: T,
) -> Result<(R::Families, LoopConstructionTrace<T>), DslError> {
    let body_outputs = body_value.flatten();
    let sealed = SubgraphHandle::seal(
        body_name,
        scope,
        explicit_inputs,
        body_outputs,
        CapturePolicy::BroadcastScalarsAndArtifactFamilies,
    )?;
    let zipped_count = arguments.len();
    let mut modes = vec![LoopInputMode::Zip; zipped_count];
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    modes.extend((0..sealed.captures.len()).map(|_| LoopInputMode::Broadcast));
    let node = NodeHandle::parallel_loop(
        sealed.handle.clone(),
        arguments,
        body_value.parallel_family_types(&count)?,
        ParallelLoop {
            count: count.clone(),
            minimum_count: 0,
            index_slot,
            bindings: Vec::new(),
            input_modes: modes,
        },
    );
    let pending = Pending::merge(
        pendings.into_iter().chain(std::iter::once(body_value.pending().remap(&sealed.remap))),
    );
    let families = body_value.parallel_families(&node, &mut 0, &count, pending)?;
    let trace = loop_construction_trace(&node, &sealed, raw_trace)?;
    Ok((families, trace))
}

pub fn parallel_zip<T: ParallelZipTuple, R: ParallelOutput>(
    families: T,
    body: impl FnOnce(LoopIndex, T::Items) -> R,
) -> Result<R::Families, DslError> {
    families.parallel_zip_tuple(body)
}

pub fn parallel_zip_bundle<T: ParallelZipTuple, R: ParallelOutput>(
    families: T,
    body: impl FnOnce(LoopIndex, T::Items) -> R,
) -> Result<R::Families, DslError> {
    families.parallel_zip_tuple(body)
}

pub fn parallel_zip_bundle_result<T: ParallelZipTuple, R: ParallelOutput>(
    families: T,
    body: impl FnOnce(LoopIndex, T::Items) -> Result<R, DslError>,
) -> Result<R::Families, DslError> {
    families.parallel_zip_tuple_result(body)
}

#[doc(hidden)]
pub fn parallel_zip_bundle_result_traced<
    Z: ParallelZipTuple,
    R: ParallelOutput,
    T: RemapConstructionTrace,
>(
    families: Z,
    body: impl FnOnce(LoopIndex, Z::Items) -> Result<(R, T), DslError>,
) -> Result<(R::Families, LoopConstructionTrace<T>), DslError> {
    families.parallel_zip_tuple_result_traced(body)
}

impl LoopIndex {
    pub fn expression(&self) -> IntExpr {
        self.expression.clone()
    }

    pub fn as_int(&self) -> Int {
        let node = NodeHandle::new(
            NodeKind::EvaluateInt(self.expression.clone()),
            Vec::new(),
            vec![WireType::ConstantInt],
        );
        Int { value: node.output(0).expect("evaluated loop index"), pending: Pending::default() }
    }
}

#[derive(Clone, Default)]
#[doc(hidden)]
pub struct Pending;

impl Pending {
    #[doc(hidden)]
    pub fn merge(_values: impl IntoIterator<Item = Pending>) -> Self {
        Self
    }

    fn remap(&self, _map: &SealMap) -> Self {
        Self
    }
}
pub struct DslContext {
    name: String,
    parameters: Vec<CompileParameter>,
    outputs: BTreeMap<String, PendingOutput>,
    real_constants: BTreeMap<String, RealExpr>,
}

struct PendingOutput {
    value: ValueHandle,
    confidentiality: Option<ArtifactConfidentiality>,
}

impl DslContext {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: Vec::new(),
            outputs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        }
    }

    pub fn int_parameter(mut self, name: impl Into<String>) -> Self {
        self.parameters
            .push(CompileParameter { name: name.into(), kind: CompileParameterKind::Integer });
        self
    }

    pub fn real_parameter(mut self, name: impl Into<String>) -> Self {
        self.parameters
            .push(CompileParameter { name: name.into(), kind: CompileParameterKind::Real });
        self
    }

    /// Materializes a compile-time integer expression as an integer wire.
    ///
    /// This is primarily useful inside structural loop bodies when a flattened family index
    /// combines loop indices with symbolic compile parameters.
    pub fn evaluate_int(&self, expression: impl Into<IntExpr>) -> Int {
        self.evaluate_int_traced(expression).0
    }

    #[doc(hidden)]
    pub fn evaluate_int_traced(
        &self,
        expression: impl Into<IntExpr>,
    ) -> (Int, EvaluateIntConstructionTrace) {
        let node = NodeHandle::new(
            NodeKind::EvaluateInt(expression.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        );
        let expression = node.output(0).expect("evaluated integer expression");
        let zero = Int::constant(0);
        let output =
            Int { value: expression.clone(), pending: Pending::default() }.add(zero.clone());
        let trace = EvaluateIntConstructionTrace {
            expression,
            zero: zero.value,
            output: output.value.clone(),
        };
        (output, trace)
    }

    #[track_caller]
    pub fn int_family_input(
        &self,
        name: impl Into<String>,
        fixed_count: impl Into<IntExpr>,
    ) -> Family<Int> {
        let count = fixed_count.into();
        let wire_type =
            WireType::IndexedFamily { element: Box::new(WireType::Int), count: count.clone() };
        let node = NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: wire_type.clone(), artifact: None },
            Vec::new(),
            vec![wire_type],
        );
        Family {
            value: node.output(0).expect("integer family input"),
            element_schema: IntType.placeholders(),
            count,
            pending: Pending::default(),
        }
    }

    pub fn output(mut self, name: impl Into<String>, mat: Mat) -> Result<Self, DslError> {
        self.insert_output(name.into(), mat, None)?;
        Ok(self)
    }

    pub fn bool_output(mut self, name: impl Into<String>, value: Bool) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), value.value, value.pending, None)?;
        Ok(self)
    }

    pub fn int_output(mut self, name: impl Into<String>, value: Int) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), value.value, value.pending, None)?;
        Ok(self)
    }

    pub fn bytes_output(mut self, name: impl Into<String>, value: Bytes) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), value.value, value.pending, None)?;
        Ok(self)
    }

    pub fn public_output(mut self, name: impl Into<String>, mat: Mat) -> Result<Self, DslError> {
        self.insert_output(name.into(), mat, Some(ArtifactConfidentiality::Public))?;
        Ok(self)
    }

    pub fn public_bytes_output(
        mut self,
        name: impl Into<String>,
        value: Bytes,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            value.value,
            value.pending,
            Some(ArtifactConfidentiality::Public),
        )?;
        Ok(self)
    }

    pub fn private_output(mut self, name: impl Into<String>, mat: Mat) -> Result<Self, DslError> {
        self.insert_output(name.into(), mat, Some(ArtifactConfidentiality::Private))?;
        Ok(self)
    }

    pub fn private_trapdoor_output(
        mut self,
        name: impl Into<String>,
        trapdoor: Trapdoor,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            trapdoor.value,
            trapdoor.pending,
            Some(ArtifactConfidentiality::Private),
        )?;
        Ok(self)
    }

    pub fn private_trapdoor_family_output(
        mut self,
        name: impl Into<String>,
        trapdoors: TrapdoorFamily,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            trapdoors.values,
            trapdoors.pending,
            Some(ArtifactConfidentiality::Private),
        )?;
        Ok(self)
    }

    fn insert_output(
        &mut self,
        name: String,
        mat: Mat,
        confidentiality: Option<ArtifactConfidentiality>,
    ) -> Result<(), DslError> {
        if self
            .outputs
            .insert(name.clone(), PendingOutput { value: mat.value, confidentiality })
            .is_some()
        {
            return Err(DslError::DuplicateOutput(name));
        }
        Ok(())
    }

    pub fn family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Mat>,
    ) -> Result<Self, DslError> {
        self.insert_family_output(name.into(), family, None)?;
        Ok(self)
    }

    pub fn int_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Int>,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), family.value, family.pending, None)?;
        Ok(self)
    }

    pub fn bool_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Bool>,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), family.value, family.pending, None)?;
        Ok(self)
    }

    pub fn public_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Mat>,
    ) -> Result<Self, DslError> {
        self.insert_family_output(name.into(), family, Some(ArtifactConfidentiality::Public))?;
        Ok(self)
    }

    pub fn private_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Mat>,
    ) -> Result<Self, DslError> {
        self.insert_family_output(name.into(), family, Some(ArtifactConfidentiality::Private))?;
        Ok(self)
    }

    fn insert_family_output(
        &mut self,
        name: String,
        family: Family<Mat>,
        confidentiality: Option<ArtifactConfidentiality>,
    ) -> Result<(), DslError> {
        self.insert_pending_output(name, family.value, family.pending, confidentiality)
    }

    fn insert_pending_output(
        &mut self,
        name: String,
        value: ValueHandle,
        _pending: Pending,
        confidentiality: Option<ArtifactConfidentiality>,
    ) -> Result<(), DslError> {
        if self.outputs.insert(name.clone(), PendingOutput { value, confidentiality }).is_some() {
            return Err(DslError::DuplicateOutput(name));
        }
        Ok(())
    }

    pub fn build(self) -> Result<BuiltGraph, DslError> {
        self.build_with_freeze_map().map(|(graph, _)| graph)
    }

    #[doc(hidden)]
    pub fn build_with_freeze_map(self) -> Result<(BuiltGraph, mxx_ir_core::FreezeMap), DslError> {
        let outputs = self
            .outputs
            .into_iter()
            .map(|(name, output)| {
                (name, GraphOutput { value: output.value, confidentiality: output.confidentiality })
            })
            .collect();
        let (graph, freeze_map) = Graph::freeze(
            self.name,
            self.parameters,
            outputs,
            Vec::new(),
            Vec::new(),
            self.real_constants,
        )?;
        mxx_ir_core::validate_structure(&graph)?;
        Ok((BuiltGraph { graph }, freeze_map))
    }
}

pub struct BuiltGraph {
    pub graph: Graph,
}

#[derive(Clone)]
pub struct IdealSpec {
    pub graph: Graph,
}

#[derive(Clone)]
pub struct PurePredicateSpec {
    pub graph: Graph,
}

fn require_sampler_free(graph: &Graph) -> Result<(), DslError> {
    let contains_sampler = graph.scopes().values().any(|scope| {
        scope.nodes().iter().any(|node| {
            matches!(
                node.kind(),
                NodeKind::UniformSample { .. } |
                    NodeKind::GaussianSample { .. } |
                    NodeKind::HashSample { .. } |
                    NodeKind::TrapdoorSample { .. } |
                    NodeKind::PreimageSample { .. }
            )
        })
    });
    if contains_sampler {
        return Err(DslError::NonPureSpecification);
    }
    Ok(())
}

impl IdealSpec {
    pub fn new(graph: BuiltGraph) -> Result<Self, DslError> {
        require_sampler_free(&graph.graph)?;
        Ok(Self { graph: graph.graph })
    }
}

impl PurePredicateSpec {
    pub fn new(graph: BuiltGraph) -> Result<Self, DslError> {
        require_sampler_free(&graph.graph)?;
        if graph.graph.outputs().len() != 1 {
            return Err(DslError::PredicateOutput);
        }
        let output = graph.graph.outputs().values().next().expect("one predicate output").value;
        let output_type = graph
            .graph
            .root_scope()
            .node(output.node)
            .and_then(|node| node.output_types().get(output.port.0 as usize));
        if output_type != Some(&WireType::Bool) && output_type != Some(&WireType::ConstantBool) {
            return Err(DslError::PredicateOutput);
        }
        Ok(Self { graph: graph.graph })
    }
}

impl BuiltGraph {
    pub fn validate(
        &self,
        bindings: &ParamEnv,
    ) -> Result<mxx_ir_core::validate::ValidatedGraph, ValidationBuildError> {
        Ok(mxx_ir_core::validate(&self.graph, bindings)?)
    }

    pub fn validate_with_manifests(
        &self,
        bindings: &ParamEnv,
        manifests: &BTreeMap<ProductionId, mxx_ir_core::artifact::Manifest>,
    ) -> Result<mxx_ir_core::validate::ValidatedGraph, ValidationBuildError> {
        Ok(mxx_ir_core::validate_with_manifests(&self.graph, bindings, manifests)?)
    }
}

#[derive(Debug, Error)]
pub enum ValidationBuildError {
    #[error(transparent)]
    Core(#[from] mxx_ir_core::ValidationError),
}

pub trait GraphValue: Clone {
    type Schema: GraphValueSchema<Value = Self>;
    fn flatten(&self) -> Vec<ValueHandle>;
    fn pending(&self) -> Pending;
    fn schema(&self) -> Self::Schema;
    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError>;
}

pub trait ParallelOutput: GraphValue {
    type Families;

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError>;

    fn parallel_families(
        self,
        node: &NodeHandle,
        next_port: &mut u32,
        count: &IntExpr,
        pending: Pending,
    ) -> Result<Self::Families, DslError>;
}

impl ParallelOutput for Mat {
    type Families = Family<Mat>;

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
        Ok(vec![WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(self.matrix_type.clone())),
            count: count.clone(),
        }])
    }

    fn parallel_families(
        self,
        node: &NodeHandle,
        next_port: &mut u32,
        count: &IntExpr,
        pending: Pending,
    ) -> Result<Self::Families, DslError> {
        let value = node.output(*next_port).ok_or(DslError::Schema)?;
        *next_port += 1;
        Ok(Family {
            value,
            element_schema: Mat {
                value: self.value,
                matrix_type: self.matrix_type,
                pending: Pending::default(),
            },
            count: count.clone(),
            pending,
        })
    }
}

impl ParallelOutput for Trapdoor {
    type Families = TrapdoorFamily;

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
        Ok(self
            .schema()
            .wire_types()
            .into_iter()
            .map(|element| WireType::IndexedFamily {
                element: Box::new(element),
                count: count.clone(),
            })
            .collect())
    }

    fn parallel_families(
        self,
        node: &NodeHandle,
        next_port: &mut u32,
        count: &IntExpr,
        pending: Pending,
    ) -> Result<Self::Families, DslError> {
        let public_value = node.output(*next_port).ok_or(DslError::Schema)?;
        *next_port += 1;
        let trapdoor_value = node.output(*next_port).ok_or(DslError::Schema)?;
        *next_port += 1;
        let schema = self.schema();
        Ok(TrapdoorFamily {
            public: Family {
                value: public_value,
                element_schema: self.public,
                count: count.clone(),
                pending: pending.clone(),
            },
            values: trapdoor_value,
            element_schema: schema,
            count: count.clone(),
            pending,
        })
    }
}

macro_rules! scalar_parallel_output {
    ($value:ty, $wire_type:expr) => {
        impl ParallelOutput for $value {
            type Families = Family<$value>;

            fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
                Ok(vec![WireType::IndexedFamily {
                    element: Box::new($wire_type),
                    count: count.clone(),
                }])
            }

            fn parallel_families(
                self,
                node: &NodeHandle,
                next_port: &mut u32,
                count: &IntExpr,
                pending: Pending,
            ) -> Result<Self::Families, DslError> {
                let value = node.output(*next_port).ok_or(DslError::Schema)?;
                *next_port += 1;
                Ok(Family { value, element_schema: self, count: count.clone(), pending })
            }
        }
    };
}

scalar_parallel_output!(Int, WireType::Int);
scalar_parallel_output!(Bool, WireType::Bool);

impl<A: ParallelOutput, B: ParallelOutput> ParallelOutput for (A, B) {
    type Families = (A::Families, B::Families);

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
        let mut types = self.0.parallel_family_types(count)?;
        types.extend(self.1.parallel_family_types(count)?);
        Ok(types)
    }

    fn parallel_families(
        self,
        node: &NodeHandle,
        next_port: &mut u32,
        count: &IntExpr,
        pending: Pending,
    ) -> Result<Self::Families, DslError> {
        let left = self.0.parallel_families(node, next_port, count, pending.clone())?;
        let right = self.1.parallel_families(node, next_port, count, pending)?;
        Ok((left, right))
    }
}

impl<A: ParallelOutput, B: ParallelOutput, C: ParallelOutput> ParallelOutput for (A, B, C) {
    type Families = (A::Families, B::Families, C::Families);

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
        let mut types = self.0.parallel_family_types(count)?;
        types.extend(self.1.parallel_family_types(count)?);
        types.extend(self.2.parallel_family_types(count)?);
        Ok(types)
    }

    fn parallel_families(
        self,
        node: &NodeHandle,
        next_port: &mut u32,
        count: &IntExpr,
        pending: Pending,
    ) -> Result<Self::Families, DslError> {
        let first = self.0.parallel_families(node, next_port, count, pending.clone())?;
        let second = self.1.parallel_families(node, next_port, count, pending.clone())?;
        let third = self.2.parallel_families(node, next_port, count, pending)?;
        Ok((first, second, third))
    }
}

impl<T: ParallelOutput> ParallelOutput for Vec<T> {
    type Families = Vec<T::Families>;

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
        self.iter().try_fold(Vec::new(), |mut types, value| {
            types.extend(value.parallel_family_types(count)?);
            Ok(types)
        })
    }

    fn parallel_families(
        self,
        node: &NodeHandle,
        next_port: &mut u32,
        count: &IntExpr,
        pending: Pending,
    ) -> Result<Self::Families, DslError> {
        self.into_iter()
            .map(|value| value.parallel_families(node, next_port, count, pending.clone()))
            .collect()
    }
}

pub trait GraphValueSchema: Clone {
    type Value: GraphValue<Schema = Self>;
    fn placeholders(&self) -> Self::Value {
        self.placeholders_from(&mut 0)
    }
    #[doc(hidden)]
    fn placeholders_from(&self, next: &mut usize) -> Self::Value;
    fn wire_types(&self) -> Vec<WireType>;
}

fn argument_name(next: &mut usize, role: &str) -> String {
    let index = *next;
    *next += 1;
    format!("arg-{index}-{role}")
}

impl GraphValue for Mat {
    type Schema = MatType;
    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }
    fn pending(&self) -> Pending {
        self.pending.clone()
    }
    fn schema(&self) -> Self::Schema {
        MatType(self.matrix_type.clone())
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Mat { value: value.clone(), matrix_type: schema.0.clone(), pending })
    }
}

impl GraphValueSchema for MatType {
    type Value = Mat;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        Mat::source_input(argument_name(next, "matrix"), self.0.clone(), None)
    }
    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Matrix(self.0.clone())]
    }
}

impl GraphValue for Bytes {
    type Schema = BytesType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        let WireType::Bytes { length } = self.value.wire_type() else {
            unreachable!("Bytes always wraps a bytes wire")
        };
        BytesType { length: length.clone() }
    }

    fn from_values(
        _schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone(), pending })
    }
}

impl GraphValue for Int {
    type Schema = IntType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        IntType
    }

    fn from_values(
        _schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone(), pending })
    }
}

impl GraphValue for Bool {
    type Schema = BoolType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        BoolType
    }

    fn from_values(
        _schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone(), pending })
    }
}

impl GraphValueSchema for BoolType {
    type Value = Bool;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "boolean"),
                wire_type: WireType::Bool,
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Bool],
        );
        Bool { value: node.output(0).expect("boolean argument"), pending: Pending::default() }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Bool]
    }
}

impl GraphValueSchema for IntType {
    type Value = Int;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "integer"),
                wire_type: WireType::Int,
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Int],
        );
        Int { value: node.output(0).expect("integer argument"), pending: Pending::default() }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Int]
    }
}

impl GraphValueSchema for BytesType {
    type Value = Bytes;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let wire_type = WireType::Bytes { length: self.length.clone() };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "bytes"),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type],
        );
        Bytes { value: node.output(0).expect("bytes argument"), pending: Pending::default() }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Bytes { length: self.length.clone() }]
    }
}

impl GraphValue for Preimage {
    type Schema = PreimageType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        PreimageType(self.matrix_type.clone())
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone(), matrix_type: schema.0.clone(), pending })
    }
}

impl GraphValueSchema for PreimageType {
    type Value = Preimage;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let wire_type = WireType::Preimage(self.0.clone());
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "preimage"),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type],
        );
        Preimage {
            value: node.output(0).expect("preimage argument"),
            matrix_type: self.0.clone(),
            pending: Pending::default(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Preimage(self.0.clone())]
    }
}

impl GraphValue for Trapdoor {
    type Schema = TrapdoorType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.public.value.clone(), self.value.clone()]
    }

    fn pending(&self) -> Pending {
        Pending::merge([self.public.pending.clone(), self.pending.clone()])
    }

    fn schema(&self) -> Self::Schema {
        let WireType::Trapdoor {
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            ..
        } = self.value.wire_type()
        else {
            unreachable!("Trapdoor always wraps a trapdoor wire")
        };
        TrapdoorType {
            matrix: self.matrix_type.clone(),
            sigma: sigma.clone(),
            gadget_base: gadget_base.clone(),
            digit_count: digit_count.clone(),
            preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [public, value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            public: Mat {
                value: public.clone(),
                matrix_type: schema.matrix.clone(),
                pending: pending.clone(),
            },
            value: value.clone(),
            matrix_type: schema.matrix.clone(),
            preimage_max_coefficient_bound: schema.preimage_max_coefficient_bound.clone(),
            pending,
        })
    }
}

impl GraphValueSchema for TrapdoorType {
    type Value = Trapdoor;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let public =
            Mat::source_input(argument_name(next, "trapdoor-public"), self.matrix.clone(), None);
        let wire_type = WireType::Trapdoor {
            matrix: self.matrix.clone(),
            sigma: self.sigma.clone(),
            gadget_base: self.gadget_base.clone(),
            digit_count: self.digit_count.clone(),
            preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "trapdoor-secret"),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type],
        );
        Trapdoor {
            public,
            value: node.output(0).expect("trapdoor argument"),
            matrix_type: self.matrix.clone(),
            preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
            pending: Pending::default(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![
            WireType::Matrix(self.matrix.clone()),
            WireType::Trapdoor {
                matrix: self.matrix.clone(),
                sigma: self.sigma.clone(),
                gadget_base: self.gadget_base.clone(),
                digit_count: self.digit_count.clone(),
                preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
            },
        ]
    }
}

impl GraphValue for TrapdoorFamily {
    type Schema = TrapdoorFamilyType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.public.value.clone(), self.values.clone()]
    }

    fn pending(&self) -> Pending {
        Pending::merge([self.public.pending.clone(), self.pending.clone()])
    }

    fn schema(&self) -> Self::Schema {
        TrapdoorFamilyType { element: self.element_schema.clone(), count: self.count.clone() }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [public, trapdoors] = values else { return Err(DslError::Schema) };
        Ok(Self {
            public: Family {
                value: public.clone(),
                element_schema: Mat::source_input(
                    "__trapdoor-family-public-schema".to_owned(),
                    schema.element.matrix.clone(),
                    None,
                ),
                count: schema.count.clone(),
                pending: pending.clone(),
            },
            values: trapdoors.clone(),
            element_schema: schema.element.clone(),
            count: schema.count.clone(),
            pending,
        })
    }
}

impl GraphValueSchema for TrapdoorFamilyType {
    type Value = TrapdoorFamily;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let public = Family::source_input(
            argument_name(next, "trapdoor-public-family"),
            self.element.matrix.clone(),
            self.count.clone(),
            None,
        );
        TrapdoorFamily::source_input(
            argument_name(next, "trapdoor-secret-family"),
            public,
            self.element.clone(),
            self.count.clone(),
            None,
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.element
            .wire_types()
            .into_iter()
            .map(|element| WireType::IndexedFamily {
                element: Box::new(element),
                count: self.count.clone(),
            })
            .collect()
    }
}

impl GraphValue for Family<Mat> {
    type Schema = MatFamilyType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        MatFamilyType {
            element: self.element_schema.matrix_type.clone(),
            count: self.count.clone(),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            value: value.clone(),
            element_schema: Mat::source_input(
                "__family-element-schema".to_owned(),
                schema.element.clone(),
                None,
            ),
            count: schema.count.clone(),
            pending,
        })
    }
}

impl GraphValueSchema for MatFamilyType {
    type Value = Family<Mat>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        Family::source_input(
            argument_name(next, "family"),
            self.element.clone(),
            self.count.clone(),
            None,
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(self.element.clone())),
            count: self.count.clone(),
        }]
    }
}

macro_rules! scalar_family_graph_value {
    ($value:ty, $schema:ident, $wire_type:expr) => {
        impl GraphValue for Family<$value> {
            type Schema = $schema;

            fn flatten(&self) -> Vec<ValueHandle> {
                vec![self.value.clone()]
            }
            fn pending(&self) -> Pending {
                self.pending.clone()
            }
            fn schema(&self) -> Self::Schema {
                $schema { count: self.count.clone() }
            }
            fn from_values(
                schema: &Self::Schema,
                values: &[ValueHandle],
                pending: Pending,
            ) -> Result<Self, DslError> {
                let [value] = values else { return Err(DslError::Schema) };
                let element_schema = <$value as GraphValueSchemaValue>::placeholder();
                Ok(Self {
                    value: value.clone(),
                    element_schema,
                    count: schema.count.clone(),
                    pending,
                })
            }
        }

        impl GraphValueSchema for $schema {
            type Value = Family<$value>;
            fn placeholders_from(&self, next: &mut usize) -> Self::Value {
                let wire_type = WireType::IndexedFamily {
                    element: Box::new($wire_type),
                    count: self.count.clone(),
                };
                let node = NodeHandle::new(
                    NodeKind::Input {
                        name: argument_name(next, "family"),
                        wire_type: wire_type.clone(),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![wire_type],
                );
                let element_schema = <$value as GraphValueSchemaValue>::placeholder();
                Family {
                    value: node.output(0).expect("family argument"),
                    element_schema,
                    count: self.count.clone(),
                    pending: Pending::default(),
                }
            }
            fn wire_types(&self) -> Vec<WireType> {
                vec![WireType::IndexedFamily {
                    element: Box::new($wire_type),
                    count: self.count.clone(),
                }]
            }
        }
    };
}

trait GraphValueSchemaValue: GraphValue {
    fn placeholder() -> Self;
}

impl GraphValueSchemaValue for Int {
    fn placeholder() -> Self {
        IntType.placeholders()
    }
}

impl GraphValueSchemaValue for Bool {
    fn placeholder() -> Self {
        BoolType.placeholders()
    }
}

scalar_family_graph_value!(Int, IntFamilyType, WireType::Int);
scalar_family_graph_value!(Bool, BoolFamilyType, WireType::Bool);

impl<A: GraphValue, B: GraphValue> GraphValue for (A, B) {
    type Schema = (A::Schema, B::Schema);
    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.0.flatten();
        values.extend(self.1.flatten());
        values
    }
    fn pending(&self) -> Pending {
        Pending::merge([self.0.pending(), self.1.pending()])
    }
    fn schema(&self) -> Self::Schema {
        (self.0.schema(), self.1.schema())
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let left_count = schema.0.wire_types().len();
        Ok((
            A::from_values(&schema.0, &values[..left_count], pending.clone())?,
            B::from_values(&schema.1, &values[left_count..], pending)?,
        ))
    }
}

impl<A: GraphValueSchema, B: GraphValueSchema> GraphValueSchema for (A, B) {
    type Value = (A::Value, B::Value);
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        (self.0.placeholders_from(next), self.1.placeholders_from(next))
    }
    fn wire_types(&self) -> Vec<WireType> {
        let mut values = self.0.wire_types();
        values.extend(self.1.wire_types());
        values
    }
}

impl<A: GraphValue, B: GraphValue, C: GraphValue> GraphValue for (A, B, C) {
    type Schema = (A::Schema, B::Schema, C::Schema);

    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.0.flatten();
        values.extend(self.1.flatten());
        values.extend(self.2.flatten());
        values
    }

    fn pending(&self) -> Pending {
        Pending::merge([self.0.pending(), self.1.pending(), self.2.pending()])
    }

    fn schema(&self) -> Self::Schema {
        (self.0.schema(), self.1.schema(), self.2.schema())
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let first_count = schema.0.wire_types().len();
        let second_count = schema.1.wire_types().len();
        Ok((
            A::from_values(&schema.0, &values[..first_count], pending.clone())?,
            B::from_values(
                &schema.1,
                &values[first_count..first_count + second_count],
                pending.clone(),
            )?,
            C::from_values(&schema.2, &values[first_count + second_count..], pending)?,
        ))
    }
}

impl<A: GraphValueSchema, B: GraphValueSchema, C: GraphValueSchema> GraphValueSchema for (A, B, C) {
    type Value = (A::Value, B::Value, C::Value);

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        (
            self.0.placeholders_from(next),
            self.1.placeholders_from(next),
            self.2.placeholders_from(next),
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        let mut values = self.0.wire_types();
        values.extend(self.1.wire_types());
        values.extend(self.2.wire_types());
        values
    }
}

impl<T: GraphValue> GraphValue for Vec<T> {
    type Schema = Vec<T::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.iter().flat_map(GraphValue::flatten).collect()
    }

    fn pending(&self) -> Pending {
        Pending::merge(self.iter().map(GraphValue::pending))
    }

    fn schema(&self) -> Self::Schema {
        self.iter().map(GraphValue::schema).collect()
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let mut offset = 0;
        schema
            .iter()
            .map(|item| {
                let count = item.wire_types().len();
                let result = T::from_values(
                    item,
                    values.get(offset..offset + count).ok_or(DslError::Schema)?,
                    pending.clone(),
                )?;
                offset += count;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()
            .and_then(|result| (offset == values.len()).then_some(result).ok_or(DslError::Schema))
    }
}

impl<T: GraphValueSchema> GraphValueSchema for Vec<T> {
    type Value = Vec<T::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        self.iter().map(|schema| schema.placeholders_from(next)).collect()
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.iter().flat_map(GraphValueSchema::wire_types).collect()
    }
}

pub struct Subgraph<I: GraphValue, O: GraphValue> {
    handle: SubgraphHandle,
    input_schema: I::Schema,
    output_schema: O::Schema,
    pending: Pending,
}

impl<I: GraphValue, O: GraphValue> Clone for Subgraph<I, O> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            pending: self.pending.clone(),
        }
    }
}

impl<I: GraphValue, O: GraphValue> Subgraph<I, O> {
    pub fn define(
        name: impl Into<String>,
        input_schema: I::Schema,
        body: impl FnOnce(I) -> O,
    ) -> Result<Self, DslError> {
        let name = name.into();
        let (inputs, output, scope) = with_new_construction_scope(|scope| {
            let inputs = input_schema.placeholders();
            let output = body(inputs.clone());
            (inputs, output, scope)
        });
        let sealed = SubgraphHandle::seal(
            name,
            scope,
            inputs.flatten(),
            output.flatten(),
            CapturePolicy::Reject,
        )
        .map_err(|error| match error {
            FreezeError::ForeignScope { .. } => DslError::SubgraphCapture,
            other => DslError::Freeze(other),
        })?;
        Ok(Self {
            handle: sealed.handle,
            input_schema,
            output_schema: output.schema(),
            pending: output.pending().remap(&sealed.remap),
        })
    }

    pub fn call(&self, input: I) -> Result<O, DslError> {
        let node = NodeHandle::subgraph_call(self.handle.clone(), input.flatten(), Vec::new());
        let values = (0..self.output_schema.wire_types().len())
            .map(|port| node.output(port as u32).expect("subgraph output"))
            .collect::<Vec<_>>();
        O::from_values(
            &self.output_schema,
            &values,
            Pending::merge([input.pending(), self.pending.clone()]),
        )
    }
}

fn product_type(left: &MatrixType, right: &MatrixType) -> MatrixType {
    let left_scalar = is_scalar_type(left);
    let right_scalar = is_scalar_type(right);
    let (rows, columns) = if left_scalar {
        (right.rows.clone(), right.columns.clone())
    } else if right_scalar {
        (left.rows.clone(), left.columns.clone())
    } else {
        (left.rows.clone(), right.columns.clone())
    };
    MatrixType { rows, columns, ..left.clone() }
}

fn is_scalar_type(matrix: &MatrixType) -> bool {
    matrix.rows == IntExpr::constant(1) && matrix.columns == IntExpr::constant(1)
}

#[macro_export]
macro_rules! concat_rows {
    ($($value:expr),+ $(,)?) => {
        $crate::Mat::concat($crate::ConcatAxis::Rows, vec![$($value),+])
    };
}

#[macro_export]
macro_rules! concat_cols {
    ($($value:expr),+ $(,)?) => {
        $crate::Mat::concat($crate::ConcatAxis::Columns, vec![$($value),+])
    };
}

#[macro_export]
macro_rules! concat_diag {
    ($($value:expr),+ $(,)?) => {
        $crate::Mat::concat($crate::ConcatAxis::Diagonal, vec![$($value),+])
    };
}

#[macro_export]
macro_rules! tag {
    ($($part:expr),* $(,)?) => {{
        let mut tag = $crate::HashTag::new();
        $(tag.push($part);)*
        tag
    }};
}

pub use mxx_ir_core::node::ConcatAxis;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_arithmetic_builds_and_validates() {
        let ring = Ring::new(17, 8);
        let input = ring.input("input", (2, 2));
        let output = input.clone() + input;
        let built = DslContext::new("sum").output("sum", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn sampler_cutoff_is_serialized_and_validated() {
        let ring = Ring::new(257, 8);
        let sample = ring.gaussian((1, 1), 3, 19);
        let built =
            DslContext::new("bounded-gaussian").output("sample", sample).unwrap().build().unwrap();
        let serialized = serde_json::to_string(&built.graph).unwrap();
        assert!(serialized.contains("max_coefficient_bound"));
        built.validate(&ParamEnv::default()).unwrap();

        let parameterized = DslContext::new("parameterized-bounded-gaussian")
            .int_parameter("cutoff")
            .output("sample", ring.gaussian((1, 1), 3, IntExpr::Var("cutoff".to_owned())))
            .unwrap()
            .build()
            .unwrap();
        let negative = ParamEnv {
            integers: BTreeMap::from([("cutoff".to_owned(), (-1).into())]),
            ..ParamEnv::default()
        };
        let constraints = mxx_ir_core::derive_param_constraints(&parameterized.graph).unwrap();
        assert!(constraints.iter().any(|constraint| !constraint.evaluate(&negative).unwrap()));
        assert!(parameterized.validate(&negative).is_err());
    }

    #[test]
    fn pure_specs_reject_sampling() {
        let ring = Ring::new(257, 8);
        let sampled = DslContext::new("not-pure")
            .output("sample", ring.gaussian((1, 1), 3, 19))
            .unwrap()
            .build()
            .unwrap();
        assert!(matches!(IdealSpec::new(sampled), Err(DslError::NonPureSpecification)));
    }

    #[test]
    fn scalar_families_gather_through_existing_parallel_loop_nodes() {
        let context = DslContext::new("scalar-family-gather");
        let values = context.int_family_input("values", 3);
        let indices = Family::<Int>::pack(vec![Int::constant(2), Int::constant(0)]).unwrap();
        let gathered = values.parallel_gather(indices).unwrap();
        let built = context.int_family_output("gathered", gathered).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        assert!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
        );
        assert!(
            built
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic))
        );
    }

    #[test]
    fn integer_families_pack_parameterized_bit_segments_with_nested_loops() {
        let segments = IntExpr::Var("segments".to_owned());
        let bits = IntExpr::Var("bits".to_owned());
        let count = IntExpr::Mul(Box::new(segments.clone()), Box::new(bits.clone()));
        let context = DslContext::new("parameterized-bit-segments")
            .int_parameter("segments")
            .int_parameter("bits");
        let input = context.int_family_input("input", count);
        let packed = input
            .parallel_pack_little_endian_bits(segments, bits)
            .expect("parameterized bit packing");
        let built = context.int_family_output("packed", packed).unwrap().build().unwrap();
        let bindings = ParamEnv {
            integers: BTreeMap::from([
                ("segments".to_owned(), 2.into()),
                ("bits".to_owned(), 3.into()),
            ]),
            ..ParamEnv::default()
        };
        built.validate(&bindings).unwrap();
        assert!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
        );
        assert!(
            built
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::SequentialLoop(_)))
        );
    }

    #[test]
    fn parameterized_trapdoor_families_use_parallel_loop_outputs() {
        let count = IntExpr::Var("count".to_owned());
        let ring = Ring::new(257, 8);
        let trapdoors = Parallel::range(count.clone())
            .map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))
            .unwrap();
        let targets = Parallel::range(count.clone()).map(|_| ring.zero((1, 1))).unwrap();
        let preimages = trapdoors
            .clone()
            .parallel_zip_mat_values(targets, |_, trapdoor, target| {
                trapdoor
                    .sample_preimage(target, (trapdoor.public_matrix().matrix_type.columns, 1))
                    .as_mat()
            })
            .unwrap();
        let built = DslContext::new("parameterized-trapdoor-families")
            .int_parameter("count")
            .public_family_output("public", trapdoors.public_matrices())
            .unwrap()
            .private_trapdoor_family_output("trapdoors", trapdoors)
            .unwrap()
            .private_family_output("preimages", preimages)
            .unwrap()
            .build()
            .unwrap();
        let bindings = ParamEnv {
            integers: BTreeMap::from([("count".to_owned(), 3.into())]),
            ..ParamEnv::default()
        };
        built.validate(&bindings).unwrap();

        let encoded = serde_json::to_vec(&built.graph).unwrap();
        let decoded: Graph = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, built.graph);
        mxx_ir_core::validate(&decoded, &bindings).unwrap();
        assert_eq!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
                .count(),
            3
        );
    }

    #[test]
    fn trapdoor_families_gather_public_and_secret_wires_together() {
        let ring = Ring::new(257, 8);
        let trapdoors =
            Parallel::range(3).map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000)).unwrap();
        let indices = Family::<Int>::pack(vec![Int::constant(2), Int::constant(0)]).unwrap();
        let gathered = trapdoors.parallel_gather(indices).unwrap();
        let built = DslContext::new("trapdoor-family-gather")
            .public_family_output("public", gathered.public_matrices())
            .unwrap()
            .private_trapdoor_family_output("secret", gathered)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert!(built.graph.root_scope().nodes().iter().any(|node| {
            matches!(node.kind(), NodeKind::ParallelLoop(loop_node) if loop_node.input_modes == vec![
                LoopInputMode::Zip,
                LoopInputMode::Broadcast,
                LoopInputMode::Broadcast,
            ])
        }));
    }

    #[test]
    fn trapdoor_and_matrix_zip_rejects_different_family_counts() {
        let ring = Ring::new(257, 8);
        let trapdoors =
            Parallel::range(2).map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000)).unwrap();
        let targets = Parallel::range(3).map(|_| ring.zero((1, 1))).unwrap();
        assert!(matches!(
            trapdoors.parallel_zip_mat_values(targets, |_, trapdoor, target| {
                trapdoor.sample_preimage(target, (6, 1)).as_mat()
            }),
            Err(DslError::FamilyCountMismatch)
        ));
    }

    #[test]
    fn heterogeneous_parallel_zip_uses_one_loop() {
        let context = DslContext::new("heterogeneous-zip");
        let kinds = context.int_family_input("kinds", 2);
        let left = Family::<Bool>::pack(vec![Bool::constant(false), Bool::constant(true)]).unwrap();
        let right =
            Family::<Bool>::pack(vec![Bool::constant(true), Bool::constant(false)]).unwrap();
        let outputs = parallel_zip_bundle((kinds, left, right), |_, (kind, left, right)| {
            kind.select_bool(vec![left, right]).expect("matching boolean candidates")
        })
        .unwrap();
        let built = context.bool_family_output("outputs", outputs).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert_eq!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
                .count(),
            1
        );
    }

    #[test]
    fn sequential_scan_keeps_nested_loop_binders_distinct() {
        let context = DslContext::new("nested-sequential-parallel");
        let initial = Family::<Int>::pack(vec![Int::constant(0), Int::constant(0)]).unwrap();
        let final_state = Sequential::range(3)
            .scan(initial, Bool::constant(true), |layer, state, _| {
                state.parallel_map(|_, value| value.add(layer.as_int()))
            })
            .unwrap();
        let built = context.int_family_output("state", final_state).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let sequential = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::SequentialLoop(spec) => Some(spec),
                _ => None,
            })
            .expect("root sequential loop");
        assert_eq!(sequential.index_slot, 0);
        let nested = built
            .graph
            .scopes()
            .iter()
            .find_map(|(scope_id, scope)| {
                matches!(scope_id, mxx_ir_core::FrozenGraphScopeId::SequentialBody { .. })
                    .then(|| {
                        scope.nodes().iter().find_map(|node| match node.kind() {
                            NodeKind::ParallelLoop(spec) => Some(spec),
                            _ => None,
                        })
                    })
                    .flatten()
            })
            .expect("nested parallel loop");
        assert_eq!(nested.index_slot, 1);

        let encoded = serde_json::to_vec(&built.graph).unwrap();
        let decoded: Graph = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(built.graph, decoded);
        mxx_ir_core::validate(&decoded, &ParamEnv::default()).unwrap();
    }

    #[test]
    fn traced_nested_loops_remap_local_and_captured_endpoints_exactly() {
        let context = DslContext::new("traced-nested-captures");
        let root_capture = Int::constant(7).add(Int::constant(0));
        let initial = Int::constant(1).add(Int::constant(0));
        let (final_state, trace) = Sequential::range(2)
            .scan_traced(initial, root_capture.clone(), |_, state, invariant| {
                let local = state.add(invariant);
                let local_handle = local.value_handle().clone();
                let (mapped, nested) = Parallel::range(2).map_values_traced(|_| {
                    let output = local.clone().add(root_capture.clone());
                    let raw = vec![
                        local.value_handle().clone(),
                        root_capture.value_handle().clone(),
                        output.value_handle().clone(),
                    ];
                    (output, raw)
                })?;
                Ok((mapped.get_static(0), (local_handle, nested)))
            })
            .unwrap();
        let (built, freeze_map) =
            context.int_output("state", final_state).unwrap().build_with_freeze_map().unwrap();

        let sequential_output = freeze_map.resolve_unique(&trace.outputs[0]).unwrap();
        assert_eq!(sequential_output.scope, mxx_ir_core::FrozenGraphScopeId::Root);
        let (local, nested) = &trace.scope.body;
        assert!(matches!(
            freeze_map.resolve_unique(local).unwrap().scope,
            mxx_ir_core::FrozenGraphScopeId::SequentialBody { .. }
        ));
        assert!(matches!(
            freeze_map.resolve_unique(&nested.outputs[0]).unwrap().scope,
            mxx_ir_core::FrozenGraphScopeId::SequentialBody { .. }
        ));
        assert_eq!(nested.scope.body.len(), 3);
        for value in &nested.scope.body {
            assert!(matches!(
                freeze_map.resolve_unique(value).unwrap().scope,
                mxx_ir_core::FrozenGraphScopeId::ParallelBody { .. }
            ));
        }
        assert_eq!(nested.scope.captures.len(), 2);
        for capture in &nested.scope.captures {
            assert!(matches!(
                freeze_map.resolve_unique(&capture.parent_source).unwrap().scope,
                mxx_ir_core::FrozenGraphScopeId::SequentialBody { .. }
            ));
            assert!(matches!(
                freeze_map.resolve_unique(&capture.child_placeholder).unwrap().scope,
                mxx_ir_core::FrozenGraphScopeId::ParallelBody { .. }
            ));
        }
        assert_eq!(trace.scope.captures.len(), 1);
        assert_eq!(
            freeze_map.resolve_unique(&trace.scope.captures[0].parent_source).unwrap().scope,
            mxx_ir_core::FrozenGraphScopeId::Root
        );
        assert!(matches!(
            freeze_map.resolve_unique(&trace.scope.captures[0].child_placeholder).unwrap().scope,
            mxx_ir_core::FrozenGraphScopeId::SequentialBody { .. }
        ));

        let outer_capture = &trace.scope.captures[0];
        let outer_source = freeze_map.resolve_unique(&outer_capture.parent_source).unwrap();
        let sequential_placeholder =
            freeze_map.resolve_unique(&outer_capture.child_placeholder).unwrap();
        let sequential_node = built
            .graph
            .root_scope()
            .node(sequential_output.wire.node)
            .expect("sequential loop node");
        let sequential_arguments =
            built.graph.root_scope().arguments(sequential_node).expect("root arguments");
        assert_eq!(sequential_arguments[2], outer_source.wire);
        let sequential_scope =
            built.graph.scope(&sequential_placeholder.scope).expect("sequential child scope");
        assert_eq!(sequential_scope.inputs()[2], sequential_placeholder.wire);

        let nested_output = freeze_map.resolve_unique(&nested.outputs[0]).unwrap();
        let nested_node =
            sequential_scope.node(nested_output.wire.node).expect("nested parallel loop node");
        let nested_arguments = sequential_scope.arguments(nested_node).expect("nested arguments");
        assert_eq!(nested_arguments.len(), nested.scope.captures.len());
        for (argument, capture) in nested_arguments.iter().zip(&nested.scope.captures) {
            assert_eq!(*argument, freeze_map.resolve_unique(&capture.parent_source).unwrap().wire);
        }
        let root_capture_in_nested = nested
            .scope
            .captures
            .iter()
            .find(|capture| {
                freeze_map.resolve_unique(&capture.parent_source).unwrap() == sequential_placeholder
            })
            .expect("root capture must flow through the sequential placeholder");
        let parallel_placeholder =
            freeze_map.resolve_unique(&root_capture_in_nested.child_placeholder).unwrap();
        let parallel_scope =
            built.graph.scope(&parallel_placeholder.scope).expect("parallel child scope");
        let capture_index = nested
            .scope
            .captures
            .iter()
            .position(|capture| std::ptr::eq(capture, root_capture_in_nested))
            .unwrap();
        assert_eq!(nested_arguments[capture_index], sequential_placeholder.wire);
        assert_eq!(parallel_scope.inputs()[capture_index], parallel_placeholder.wire);
    }

    #[test]
    fn body_trace_remap_covers_identity_preserving_and_forced_clone_paths() {
        use std::{cell::RefCell, rc::Rc};

        let captured = Int::constant(5).add(Int::constant(0));
        let raw_preserved = Rc::new(RefCell::new(None));
        let raw_cloned = Rc::new(RefCell::new(None));
        let (outputs, trace) = Parallel::range(1)
            .map_values_traced({
                let raw_preserved = raw_preserved.clone();
                let raw_cloned = raw_cloned.clone();
                move |_| {
                    let preserved = Int::constant(1).add(Int::constant(0));
                    let cloned = preserved.clone().add(captured.clone());
                    *raw_preserved.borrow_mut() = Some(preserved.value_handle().clone());
                    *raw_cloned.borrow_mut() = Some(cloned.value_handle().clone());
                    (
                        cloned.clone(),
                        (preserved.value_handle().clone(), cloned.value_handle().clone()),
                    )
                }
            })
            .unwrap();
        let raw_preserved = raw_preserved.borrow().clone().unwrap();
        let raw_cloned = raw_cloned.borrow().clone().unwrap();
        assert_eq!(trace.scope.body.0, raw_preserved);
        assert_ne!(trace.scope.body.1, raw_cloned);

        let (_, freeze_map) = DslContext::new("remap-clone-paths")
            .int_family_output("outputs", outputs)
            .unwrap()
            .build_with_freeze_map()
            .unwrap();
        assert!(freeze_map.resolve_unique(&trace.scope.body.0).is_ok());
        assert!(freeze_map.resolve_unique(&trace.scope.body.1).is_ok());
    }

    #[test]
    fn traced_loop_rejects_unreachable_raw_body_handle() {
        let result = Parallel::range(1).map_values_traced(|_| {
            let output = Int::constant(1).add(Int::constant(0));
            let unreachable = Int::constant(9);
            (output, unreachable.value_handle().clone())
        });
        assert!(matches!(result, Err(DslError::TraceRemap)));
    }

    #[test]
    fn ordinary_and_traced_nested_loop_builders_freeze_identically() {
        fn ordinary() -> Graph {
            let context = DslContext::new("nested-loop-equivalence");
            let initial = context.int_family_input("initial", 3);
            let state = Sequential::range(2)
                .scan(initial, Bool::constant(true), |_, state, _| {
                    let indices = Parallel::range(3)
                        .map_values(|index| index.as_int().add(Int::constant(0)))?;
                    let gathered = state.clone().parallel_gather(indices)?;
                    parallel_zip_bundle_result((state, gathered), |_, (left, right)| {
                        Ok(left.add(right))
                    })
                })
                .unwrap();
            context.int_family_output("state", state).unwrap().build().unwrap().graph
        }

        fn traced() -> Graph {
            let context = DslContext::new("nested-loop-equivalence");
            let initial = context.int_family_input("initial", 3);
            let (state, _) = Sequential::range(2)
                .scan_traced(initial, Bool::constant(true), |_, state, _| {
                    let (indices, map_trace) = Parallel::range(3).map_values_traced(|index| {
                        let value = index.as_int().add(Int::constant(0));
                        let handle = value.value_handle().clone();
                        (value, handle)
                    })?;
                    let (gathered, gather_trace) = state.clone().parallel_gather_traced(indices)?;
                    let (next, zip_trace) = parallel_zip_bundle_result_traced(
                        (state, gathered),
                        |_, (left, right)| {
                            let output = left.add(right);
                            let handle = output.value_handle().clone();
                            Ok((output, handle))
                        },
                    )?;
                    Ok((next, (map_trace, gather_trace, zip_trace)))
                })
                .unwrap();
            context.int_family_output("state", state).unwrap().build().unwrap().graph
        }

        assert_eq!(ordinary(), traced());
    }

    #[test]
    fn ordinary_and_traced_trapdoor_matrix_zip_freeze_identically() {
        fn build(traced: bool) -> Graph {
            let ring = Ring::new(257, 8);
            let trapdoors = Parallel::range(2)
                .map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))
                .unwrap();
            let targets = Parallel::range(2).map(|_| ring.zero((1, 1))).unwrap();
            let outputs = if traced {
                trapdoors
                    .parallel_zip_mat_values_traced(targets, |_, trapdoor, target| {
                        let public = trapdoor.public_matrix();
                        let trapdoor_handle = trapdoor.value_handle().clone();
                        let target_handle = target.value_handle().clone();
                        let preimage = trapdoor.sample_preimage(target, (6, 1));
                        let preimage_handle = preimage.value_handle().clone();
                        let output = preimage.as_mat();
                        let trace = vec![
                            public.value_handle().clone(),
                            trapdoor_handle,
                            target_handle,
                            preimage_handle,
                            output.value_handle().clone(),
                        ];
                        (output, trace)
                    })
                    .unwrap()
                    .0
            } else {
                trapdoors
                    .parallel_zip_mat_values(targets, |_, trapdoor, target| {
                        trapdoor.sample_preimage(target, (6, 1)).as_mat()
                    })
                    .unwrap()
            };
            DslContext::new("trapdoor-zip-equivalence")
                .private_family_output("outputs", outputs)
                .unwrap()
                .build()
                .unwrap()
                .graph
        }

        assert_eq!(build(false), build(true));
    }

    #[test]
    fn context_materializes_composite_integer_expressions() {
        let context = DslContext::new("evaluate-composite-int").int_parameter("width");
        let values = Parallel::range(2)
            .map_values(|index| {
                context.evaluate_int(IntExpr::Add(
                    Box::new(IntExpr::Mul(
                        Box::new(index.expression()),
                        Box::new(IntExpr::Var("width".to_owned())),
                    )),
                    Box::new(IntExpr::constant(1)),
                ))
            })
            .unwrap();
        let built = context.int_family_output("values", values).unwrap().build().unwrap();
        built
            .validate(&ParamEnv {
                integers: BTreeMap::from([("width".to_owned(), 3.into())]),
                ..ParamEnv::default()
            })
            .unwrap();
        assert!(
            built
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| { matches!(node.kind(), NodeKind::EvaluateInt(IntExpr::Add(_, _))) })
        );
    }

    #[test]
    fn scalar_selects_and_maps_promote_constant_outputs() {
        let context = DslContext::new("normalized-scalar-outputs");
        let selectors = context.int_family_input("selectors", 1);
        let selector = selectors.get_static(0);
        let dynamic_int = context.int_family_input("dynamic", 1).get_static(0);
        let dynamic_bool = dynamic_int.clone().equal(Int::constant(0));

        let all_constant_int =
            selector.clone().select_int(vec![Int::constant(3), Int::constant(5)]).unwrap();
        let mixed_int =
            selector.clone().select_int(vec![dynamic_int.clone(), Int::constant(7)]).unwrap();
        let all_constant_bool = selector
            .clone()
            .select_bool(vec![Bool::constant(false), Bool::constant(true)])
            .unwrap();
        let mixed_bool = selector.select_bool(vec![dynamic_bool, Bool::constant(false)]).unwrap();
        let constant_ints =
            context.int_family_input("map-ints", 2).parallel_map(|_, _| Int::constant(11)).unwrap();
        let constant_bools =
            Family::<Bool>::pack(vec![Bool::constant(false), Bool::constant(true)])
                .unwrap()
                .parallel_map(|_, _| Bool::constant(true))
                .unwrap();

        let built = context
            .int_output("all-constant-int", all_constant_int)
            .unwrap()
            .int_output("mixed-int", mixed_int)
            .unwrap()
            .bool_output("all-constant-bool", all_constant_bool)
            .unwrap()
            .bool_output("mixed-bool", mixed_bool)
            .unwrap()
            .int_family_output("constant-ints", constant_ints)
            .unwrap()
            .bool_family_output("constant-bools", constant_bools)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
    }
}
