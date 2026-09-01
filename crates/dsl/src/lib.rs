//! Declarative typed construction API for mxx graphs.
//!
//! Executable operations create immutable `mxx-ir-core` nodes immediately.

use mxx_ir_core::{
    CapturePolicy, CompileParameter, CompileParameterKind, FreezeError, Graph, GraphOutput,
    IndexExpr, IndexMap, IntExpr, NodeHandle, ParamEnv, RealExpr, SealMap, SealedSubgraph,
    SubgraphHandle, ValueHandle,
    artifact::{ArtifactConfidentiality, ProductionId},
    graph::with_new_construction_scope,
    node::{
        ArtifactInput, ConstantMatrix, IndexRange, MatrixBinaryOp, NodeKind,
        ParallelGrid as IrParallelGrid, SampleRange, SequentialLoop,
    },
    types::{MatrixType, WireType},
};
use num_bigint::BigUint;
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
    #[error("canonical input exclusive upper bound count does not match flattened subgraph inputs")]
    CanonicalInputUpperCount,
    #[error("canonical input exclusive upper bounds must be positive")]
    CanonicalInputUpperZero,
    #[error("canonical input exclusive upper bounds require matrix subgraph inputs")]
    CanonicalInputUpperNonMatrix,
    #[error("parallel families have different counts")]
    FamilyCountMismatch,
    #[error("parallel_map_values requires a rank-one input family")]
    ParallelMapRank,
    #[error("parallel family operation requires rank-one input families")]
    ParallelFamilyRank,
    #[error("parallel_gather requires rank-one source and index families")]
    ParallelGatherRank,
    #[error("parallel zip requires rank-one input families")]
    ParallelZipRank,
    #[error(transparent)]
    StructuralValidation(#[from] mxx_ir_core::ValidationError),
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
/// Type of a witness matrix `K` whose public relation is `B*K=T`.
///
/// This wrapper is intentionally distinct from `Mat`: a matrix value alone does not authorize
/// relation consumption, while a `Preimage` value does.
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
/// Shape and element type of a family of trapdoor relations indexed by a rank-N coordinate.
pub struct TrapdoorFamilyType {
    pub element: TrapdoorType,
    pub shape: Vec<IntExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// Shape and matrix element type of a rank-N family `X[u]`.
pub struct MatFamilyType {
    pub element: MatrixType,
    pub shape: Vec<IntExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// Shape and typed witness element of a rank-N preimage family `K[u]`.
pub struct PreimageFamilyType {
    pub element: MatrixType,
    pub shape: Vec<IntExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IntFamilyType {
    pub shape: Vec<IntExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BoolFamilyType {
    pub shape: Vec<IntExpr>,
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

    /// Creates a typed preimage protocol input representing a witness `K` for `B*K=T`.
    ///
    /// Noise simulation remains fail-closed unless a relation for this value is supplied by the
    /// surrounding stage or artifact flow.
    #[track_caller]
    pub fn preimage_input(&self, name: impl Into<String>, shape: impl IntoShape) -> Preimage {
        let matrix_type = self.matrix_type(shape);
        let wire_type = WireType::Preimage(matrix_type.clone());
        let node = NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: wire_type.clone(), artifact: None },
            Vec::new(),
            vec![wire_type],
        );
        Preimage {
            value: node.output(0).expect("preimage input"),
            matrix_type,
            pending: Pending::default(),
        }
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
        let public = Family::<Mat>::source_input(
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
            vec![count],
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
        Family::<Mat>::source_input(name.into(), element, count.into(), None)
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
        Family::<Mat>::source_input(
            format!("artifact:{artifact_name}"),
            self.matrix_type(shape),
            count.into(),
            Some(ArtifactInput { production_id, artifact_name, confidentiality }),
        )
    }

    /// Loads a public or private artifact family whose elements are preimages.
    ///
    /// Preimages retain their relation marker after crossing a stage boundary;
    /// this is what lets a later `mul_decomposed`/`apply_preimage` operation
    /// consume the artifact without re-discovering the relation.
    #[track_caller]
    pub fn preimage_family_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        family_shape: Vec<IntExpr>,
        matrix_shape: impl IntoShape,
        confidentiality: ArtifactConfidentiality,
    ) -> Family<Preimage> {
        let artifact_name = artifact_name.into();
        let matrix_type = self.matrix_type(matrix_shape);
        Family::<Preimage>::source_input(
            format!("artifact:{artifact_name}"),
            matrix_type,
            family_shape,
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
    /// Samples a matrix uniformly from the full coefficient residue ring `R_q`.
    pub fn uniform_residue(&self, shape: impl IntoShape) -> Mat {
        let ty = self.matrix_type(shape);
        Mat::from_node(NodeKind::UniformResidueSample { matrix_type: ty.clone() }, Vec::new(), ty)
    }

    #[track_caller]
    /// Samples from one of the supported small integer intervals: `[-1, 1]` or `[0, 1]`.
    pub fn uniform_interval(
        &self,
        shape: impl IntoShape,
        minimum: impl Into<IntExpr>,
        maximum: impl Into<IntExpr>,
    ) -> Mat {
        let ty = self.matrix_type(shape);
        Mat::from_node(
            NodeKind::UniformIntervalSample {
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
        self.hash(key, tag, shape)
    }

    #[track_caller]
    pub fn hash_decomposed(
        &self,
        key: Bytes,
        tag: impl Into<HashTag>,
        shape: impl IntoShape,
        base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
    ) -> Decomposition {
        let shape = shape.into_shape();
        let base = base.into();
        let digit_count = digit_count.into();
        let plain_shape = Shape {
            rows: IntExpr::Div(Box::new(shape.rows.clone()), Box::new(digit_count.clone())),
            columns: shape.columns.clone(),
        };
        self.hash(key, tag, plain_shape).decompose(base, digit_count)
    }

    #[track_caller]
    fn hash(&self, key: Bytes, tag: impl Into<HashTag>, shape: impl IntoShape) -> Mat {
        let ty = self.matrix_type(shape);
        let tag = tag.into();
        let pending = Pending::merge([key.pending.clone(), tag.pending]);
        let mut arguments = vec![key.value];
        arguments.extend(tag.dynamic);
        let node = NodeHandle::new(
            NodeKind::HashSample {
                matrix_type: ty.clone(),
                tag_prefix: tag.prefix,
                tag_expressions: tag.binary,
                tag_decimal_expressions: tag.decimal,
                tag_u64_le_expressions: tag.u64_le,
            },
            arguments,
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

#[derive(Clone, Default)]
pub struct HashTag {
    prefix: Vec<u8>,
    binary: Vec<IntExpr>,
    decimal: Vec<IntExpr>,
    u64_le: Vec<IntExpr>,
    dynamic: Vec<ValueHandle>,
    pending: Pending,
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

impl HashTagPart for Int {
    fn append_to(self, tag: &mut HashTag) {
        tag.dynamic.push(self.value);
        tag.pending = Pending::merge([std::mem::take(&mut tag.pending), self.pending]);
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

    /// Fuses a sum of coefficient-weighted matrix products for execution.
    /// Semantic consumers expand this to ordinary multiply, scale, and add.
    #[track_caller]
    pub fn multi_row_gemm_accumulate<C: Into<IntExpr>>(
        products: Vec<(C, Mat, Mat)>,
        bias: Option<Mat>,
    ) -> Self {
        assert!(!products.is_empty(), "multi-row GEMM requires at least one product");
        let output_type = MatrixType {
            columns: products[0].2.matrix_type.columns.clone(),
            ..products[0].1.matrix_type.clone()
        };
        let has_bias = bias.is_some();
        let mut coefficients = Vec::with_capacity(products.len());
        let mut arguments = Vec::with_capacity(products.len() * 2 + usize::from(has_bias));
        for (coefficient, left, right) in products {
            coefficients.push(coefficient.into());
            arguments.push(left);
            arguments.push(right);
        }
        if let Some(bias) = bias {
            arguments.push(bias);
        }
        Self::from_node(
            NodeKind::MatrixMulAccumulate { coefficients, has_bias },
            arguments,
            output_type,
        )
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
    pub fn decompose(
        self,
        base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
    ) -> Decomposition {
        self.decompose_with_mode(base.into(), digit_count.into(), false)
    }

    /// Applies a typed preimage relation explicitly; this is the only operation that transports
    /// relation semantics across a matrix product.
    ///
    /// Given ordinary `A` and witness `K` with `B*K=T`, this emits `A*K`. The arithmetic matches
    /// `Mat * Mat`, but only this method is allowed to consume the `Preimage` marker; ordinary
    /// multiplication remains relation-unaware.
    #[track_caller]
    pub fn apply_preimage(self, preimage: Preimage) -> Self {
        let matrix_type = product_type(&self.matrix_type, &preimage.matrix_type);
        let pending = Pending::merge([self.pending, preimage.pending]);
        let node = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![self.value, preimage.value],
            vec![WireType::Matrix(matrix_type.clone())],
        );
        Self { value: node.output(0).expect("preimage application"), matrix_type, pending }
    }

    #[track_caller]
    pub fn mul_decomposed(self, decomposition: Decomposition) -> Self {
        self.apply_preimage(decomposition.into_preimage_relation())
    }

    #[track_caller]
    pub fn small_decompose(
        self,
        base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
    ) -> Decomposition {
        self.decompose_with_mode(base.into(), digit_count.into(), true)
    }

    fn decompose_with_mode(
        self,
        base: IntExpr,
        digit_count: IntExpr,
        small: bool,
    ) -> Decomposition {
        // The witness has `digit_count` times the target row count, making `G*K=T` well typed for
        // the gadget matrix G selected by `base` and `small`.
        let ty = MatrixType {
            rows: IntExpr::Mul(
                Box::new(self.matrix_type.rows.clone()),
                Box::new(digit_count.clone()),
            )
            .canonicalize(),
            ..self.matrix_type.clone()
        };
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::GadgetDecompose { base, small, digit_count },
            vec![self.value],
            vec![WireType::Preimage(ty.clone())],
        );
        Decomposition {
            preimage: Preimage {
                value: node.output(0).expect("decomposition"),
                matrix_type: ty,
                pending,
            },
        }
    }

    #[track_caller]
    pub fn extract_coefficient(self, position: impl Into<IntExpr>) -> Int {
        self.extract_coefficient_with_canonical_input_exclusive_upper(position, None)
    }

    /// Extracts a coefficient and optionally records a compile-time-only
    /// exclusive upper bound for a canonical input integer.
    #[track_caller]
    pub fn extract_coefficient_with_canonical_input_exclusive_upper(
        self,
        position: impl Into<IntExpr>,
        canonical_input_exclusive_upper: Option<num_bigint::BigUint>,
    ) -> Int {
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::ExtractCoefficient {
                position: position.into(),
                canonical_input_exclusive_upper,
            },
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

    /// Projects a witness `K` to an ordinary matrix only when its registered target is exact.
    ///
    /// The runtime representation is unchanged (`K -> K`), but the `B*K=T` marker is dropped.
    /// The noise simulator rejects this operation when `T` has nonzero error, so it cannot erase
    /// noise transport.
    pub fn materialize_exact(self) -> Mat {
        let matrix_type = self.matrix_type.clone();
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::MaterializePreimageExact,
            vec![self.value],
            vec![WireType::Matrix(matrix_type.clone())],
        );
        Mat { value: node.output(0).expect("exact preimage materialization"), matrix_type, pending }
    }

    /// Adds two witnesses for the same public source and adds their relation targets.
    ///
    /// If `B*K_1=T_1` and `B*K_2=T_2`, the result is `K'=K_1+K_2` with
    /// `B*K'=T_1+T_2` by distributivity.
    pub fn add_same_source(self, rhs: Self) -> Self {
        let matrix_type = self.matrix_type.clone();
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::PreimageBinary(mxx_ir_core::node::PreimageBinaryOp::Add),
            vec![self.value, rhs.value],
            vec![WireType::Preimage(matrix_type.clone())],
        );
        Self { value: node.output(0).expect("preimage sum"), matrix_type, pending }
    }

    /// Right-multiplies a witness by an exact ordinary scalar matrix `A`.
    ///
    /// The resulting relation follows from `B*(K*A)=(B*K)*A=T*A`.
    pub fn right_multiply_exact(self, rhs: Mat) -> Self {
        let matrix_type = product_type(&self.matrix_type, &rhs.matrix_type);
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::PreimageBinary(mxx_ir_core::node::PreimageBinaryOp::RightMultiplyExact),
            vec![self.value, rhs.value],
            vec![WireType::Preimage(matrix_type.clone())],
        );
        Self { value: node.output(0).expect("preimage right product"), matrix_type, pending }
    }

    /// Right-composes a witness with an exact gadget witness `L`.
    ///
    /// With `B*K=T` and `G*L=U`, this emits `K*L`; the common-source relation is preserved as
    /// `B*(K*L)=T*L`.
    pub fn compose_exact_decomposition(self, rhs: Decomposition) -> Self {
        let rhs = rhs.preimage;
        let matrix_type = product_type(&self.matrix_type, &rhs.matrix_type);
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::PreimageBinary(
                mxx_ir_core::node::PreimageBinaryOp::ComposeExactDecomposition,
            ),
            vec![self.value, rhs.value],
            vec![WireType::Preimage(matrix_type.clone())],
        );
        Self { value: node.output(0).expect("composed preimage"), matrix_type, pending }
    }

    /// Concatenates columns of witnesses sharing one public source.
    ///
    /// For each input `B*K_j=T_j`, the output `K=[K_1|...|K_n]` has target
    /// `T=[T_1|...|T_n]`, so `B*K=T` columnwise.
    pub fn concat_columns(values: Vec<Self>) -> Self {
        assert!(!values.is_empty(), "preimage concat requires at least one value");
        let first = &values[0];
        let matrix_type = MatrixType {
            columns: values
                .iter()
                .map(|value| value.matrix_type.columns.clone())
                .reduce(|left, right| IntExpr::Add(Box::new(left), Box::new(right)).canonicalize())
                .expect("nonempty preimage concat"),
            ..first.matrix_type.clone()
        };
        let pending = Pending::merge(values.iter().map(|value| value.pending.clone()));
        let node = NodeHandle::new(
            NodeKind::PreimageConcatColumns,
            values.into_iter().map(|value| value.value).collect(),
            vec![WireType::Preimage(matrix_type.clone())],
        );
        Self { value: node.output(0).expect("preimage column concat"), matrix_type, pending }
    }
}

/// A gadget decomposition that can either be consumed as its typed preimage relation or queried
/// entry-by-entry when its relation target is exact.
#[derive(Clone)]
pub struct Decomposition {
    preimage: Preimage,
}

impl Decomposition {
    /// Retains the universal gadget relation `G*K=T` while exposing the common witness interface.
    pub fn into_preimage_relation(self) -> Preimage {
        self.preimage
    }

    /// Selects one scalar digit `K[row,column]` without exposing the whole decomposition as a
    /// relation-free matrix; the underlying witness still denotes `G*K=T`.
    pub fn entry(&self, row: impl Into<IntExpr>, column: impl Into<IntExpr>) -> Mat {
        let matrix_type = MatrixType {
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
            ..self.preimage.matrix_type.clone()
        };
        let node = NodeHandle::new(
            NodeKind::DecompositionEntry { row: row.into(), column: column.into() },
            vec![self.preimage.value.clone()],
            vec![WireType::Matrix(matrix_type.clone())],
        );
        Mat {
            value: node.output(0).expect("decomposition entry"),
            matrix_type,
            pending: self.preimage.pending.clone(),
        }
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
    /// Samples a witness `K` for the registered public matrix `B` and target `T` so that `B*K=T`.
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
/// parallel families rather than pretending that `Family<T>` can contain a multi-wire value. At
/// coordinate `u`, the pair is `(B[u], trapdoor[u])`, defining the relation used for preimages.
#[derive(Clone)]
pub struct TrapdoorFamily {
    public: Family<Mat>,
    values: ValueHandle,
    element_schema: TrapdoorType,
    count: IntExpr,
    shape: Vec<IntExpr>,
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
        shape: Vec<IntExpr>,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let element_type = WireType::Trapdoor {
            matrix: element_schema.matrix.clone(),
            sigma: element_schema.sigma.clone(),
            gadget_base: element_schema.gadget_base.clone(),
            digit_count: element_schema.digit_count.clone(),
            preimage_max_coefficient_bound: element_schema.preimage_max_coefficient_bound.clone(),
        };
        let count = shape_count(&shape);
        let family_type =
            WireType::Family { element: Box::new(element_type), shape: shape.clone() };
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
            shape,
            pending: Pending::default(),
        }
    }

    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    pub fn public_matrices(&self) -> Family<Mat> {
        self.public.clone()
    }

    pub fn shape(&self) -> &[IntExpr] {
        &self.shape
    }

    /// Reindexes public matrices and secret trapdoors with one identical map.
    ///
    /// For output coordinate `u`, both sides use the same `f(u)`: `B'[u]=B[f(u)]` and
    /// `trapdoor'[u]=trapdoor[f(u)]`. This preserves the source/trapdoor pairing exactly.
    pub fn reindex(self, output_shape: Vec<IntExpr>, map: IndexMap) -> Result<Self, DslError> {
        let public = self.public.reindex(output_shape.clone(), map.clone())?;
        let element_type = WireType::Trapdoor {
            matrix: self.element_schema.matrix.clone(),
            sigma: self.element_schema.sigma.clone(),
            gadget_base: self.element_schema.gadget_base.clone(),
            digit_count: self.element_schema.digit_count.clone(),
            preimage_max_coefficient_bound: self
                .element_schema
                .preimage_max_coefficient_bound
                .clone(),
        };
        let family_type =
            WireType::Family { element: Box::new(element_type), shape: output_shape.clone() };
        let node = NodeHandle::new(
            NodeKind::FamilyReindex { output_shape: output_shape.clone(), map },
            vec![self.values],
            vec![family_type],
        );
        Ok(Self {
            public,
            values: node.output(0).expect("reindexed trapdoor family"),
            element_schema: self.element_schema,
            count: shape_count(&output_shape),
            shape: output_shape,
            pending: self.pending,
        })
    }

    /// Samples a shared-source preimage table. The final logical family axis is
    /// the branch axis; the public and trapdoor families are broadcast over it.
    /// Runtime and simulator validation check the concrete group/branch shapes. For source
    /// coordinate `i` and target branch `j`, the result is `K[i,j]` satisfying
    /// `B[i]*K[i,j]=T[i,j]`.
    #[track_caller]
    pub fn sample_preimage_branches(
        &self,
        targets: Family<Mat>,
        shape: impl IntoShape,
    ) -> Result<Family<Preimage>, DslError> {
        let shape = shape.into_shape();
        let matrix_type =
            MatrixType { rows: shape.rows, columns: shape.columns, ..self.matrix_type() };
        let pending = Pending::merge([self.pending.clone(), targets.pending.clone()]);
        // The node represents Y[u]=X[f(u)]; only coordinate interpretation changes, not the
        // matrix element schema.
        let family_type = WireType::Family {
            element: Box::new(WireType::Preimage(matrix_type.clone())),
            shape: targets.shape.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::FamilyPreimageSample {
                matrix_type: matrix_type.clone(),
                max_coefficient_bound: self.element_schema.preimage_max_coefficient_bound.clone(),
            },
            vec![self.public.value.clone(), self.values.clone(), targets.value],
            vec![family_type],
        );
        Ok(Family {
            value: node.output(0).expect("family preimage output"),
            element_schema: Preimage {
                value: NodeHandle::new(
                    NodeKind::Input {
                        name: "family-preimage-element".to_owned(),
                        wire_type: WireType::Preimage(matrix_type.clone()),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![WireType::Preimage(matrix_type.clone())],
                )
                .output(0)
                .expect("family preimage schema"),
                matrix_type: matrix_type.clone(),
                pending: Pending::default(),
            },
            count: targets.count.clone(),
            shape: targets.shape.clone(),
            pending,
        })
    }

    fn matrix_type(&self) -> MatrixType {
        self.element_schema.matrix.clone()
    }

    pub fn get_static(&self, indices: impl IntoFamilyStaticIndices) -> Trapdoor {
        let indices = indices.into_family_indices();
        let public = self.public.get_static(indices.clone());
        let pending = Pending::merge([self.pending.clone(), public.pending.clone()]);
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices },
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

    pub fn get(&self, indices: impl IntoFamilyDynamicIndices) -> Trapdoor {
        let indices = indices.into_family_indices();
        let public = self.public.get(indices.clone());
        let pending = Pending::merge(
            std::iter::once(self.pending.clone())
                .chain(indices.iter().map(|index| index.pending.clone())),
        );
        let mut arguments = vec![self.values.clone()];
        arguments.extend(indices.iter().map(|index| index.value.clone()));
        let node = NodeHandle::new(
            NodeKind::FamilyGetDynamic { rank: indices.len() },
            arguments,
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
        if self.shape.len() != 1 || indices.shape.len() != 1 {
            return Err(DslError::ParallelGatherRank);
        }
        let source_count = self.count.clone();
        let output_count = indices.count.clone();
        let schema = self.element_schema.clone();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|_| {
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
                    vec![source_count.clone()],
                    None,
                );
                let mut explicit_inputs = vec![index.value.clone()];
                explicit_inputs.extend(source.flatten());
                let output = source.get(index.clone());
                (output, explicit_inputs, scope)
            })
        });
        let sealed = SubgraphHandle::seal(
            "parallel-gather-trapdoor-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            vec![indices.value, self.public.value, self.values],
            body_value.parallel_family_types(&output_count)?,
            IrParallelGrid {
                shape: vec![output_count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: vec![
                    mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    },
                    mxx_ir_core::node::GridInputMode::Broadcast,
                    mxx_ir_core::node::GridInputMode::Broadcast,
                ],
            },
        );
        let pending = Pending::merge([
            indices.pending,
            self.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        body_value.parallel_families(&node, &mut 0, &output_count, pending)
    }

    pub fn parallel_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Trapdoor) -> R,
    ) -> Result<R::Families, DslError> {
        if self.shape.len() != 1 {
            // TrapdoorFamily carries paired public and secret family views.
            // Flattening either view here would discard their common
            // Cartesian coordinate and invalidate the pairing contract.
            return Err(DslError::ParallelMapRank);
        }
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
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let node = NodeHandle::parallel_grid(
            sealed.handle,
            arguments,
            body_value.parallel_family_types(&count)?,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: std::iter::once(mxx_ir_core::node::GridInputMode::Reindex {
                    map: IndexMap::new([IndexExpr::Axis(0)]),
                })
                .chain(std::iter::once(mxx_ir_core::node::GridInputMode::Reindex {
                    map: IndexMap::new([IndexExpr::Axis(0)]),
                }))
                .chain(
                    (0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                )
                .collect(),
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
        if self.shape.len() != 1 || matrices.shape.len() != 1 {
            return Err(DslError::ParallelZipRank);
        }
        if self.count != matrices.count {
            return Err(DslError::FamilyCountMismatch);
        }
        let count = self.count.clone();
        let trapdoor_schema = self.element_schema.clone();
        let matrix_schema = MatType(matrices.element_schema.matrix_type.clone());
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let trapdoor = trapdoor_schema.placeholders();
                let matrix = matrix_schema.placeholders();
                let mut explicit_inputs = trapdoor.flatten();
                explicit_inputs.extend(matrix.flatten());
                let body_value = body(index, trapdoor, matrix);
                (body_value, explicit_inputs, scope)
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
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            arguments,
            body_value.parallel_family_types(&count)?,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: std::iter::repeat_with(|| mxx_ir_core::node::GridInputMode::Reindex {
                    map: IndexMap::new([IndexExpr::Axis(0)]),
                })
                .take(3)
                .chain(
                    (0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                )
                .collect(),
            },
        );
        let pending = Pending::merge([
            self.pending,
            matrices.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        body_value.parallel_families(&node, &mut 0, &count, pending)
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

    #[track_caller]
    pub fn lift_to_constant_polynomial(self, matrix_type: MatrixType) -> Mat {
        assert_eq!(matrix_type.rows, IntExpr::constant(1), "constant-polynomial lift is scalar");
        assert_eq!(matrix_type.columns, IntExpr::constant(1), "constant-polynomial lift is scalar");
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type: matrix_type.clone() },
            vec![self.value],
            vec![WireType::Matrix(matrix_type.clone())],
        );
        Mat { value: node.output(0).expect("constant-polynomial lift"), matrix_type, pending }
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

    pub fn get_static(&self, indices: impl IntoFamilyStaticIndices) -> Bool {
        let element_type = family_element_wire_type(&self.value).unwrap_or(WireType::Bool);
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices: indices.into_family_indices() },
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

    pub fn parallel_gather(self, indices: Family<Int>) -> Result<Self, DslError> {
        scalar_parallel_gather(
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

    pub fn get_static(&self, indices: impl IntoFamilyStaticIndices) -> Int {
        scalar_family_get_static(
            self,
            indices.into_family_indices(),
            WireType::Int,
            |value, pending| Int { value, pending },
        )
    }

    pub fn get(&self, index: Int) -> Int {
        scalar_family_get(self, index, WireType::Int, |value, pending| Int { value, pending })
    }

    pub fn parallel_map(self, body: impl FnOnce(LoopIndex, Int) -> Int) -> Result<Self, DslError> {
        scalar_parallel_map(self, "parallel-map-int-body", WireType::Int, body)
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
        if self.shape.len() != 1 {
            return Err(DslError::ParallelFamilyRank);
        }
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
                let family_type = WireType::Family {
                    element: Box::new(WireType::Int),
                    shape: vec![source_count.clone()],
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
                    count: source_count.clone(),
                    shape: vec![source_count.clone()],
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
                    .scan(
                        (Int::constant(0), Int::constant(1)),
                        source.clone(),
                        move |bit, (sum, weight), source| {
                            let source_index = segment.clone().mul(segment_width).add(bit.as_int());
                            let value = source.get(source_index);
                            let next_sum = sum.add(value.mul(weight.clone()));
                            let next_weight = weight.mul(Int::constant(2));
                            Ok((next_sum, next_weight))
                        },
                    )
                    .map(|(sum, _)| (sum, vec![source.value], scope))
            })
        });
        let (body_value, explicit_inputs, scope) = body_result?;
        let sealed = SubgraphHandle::seal(
            "parallel-pack-little-endian-bits-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let argument_count = arguments.len();
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            arguments,
            vec![WireType::Family {
                element: Box::new(WireType::Int),
                shape: vec![segment_count.clone()],
            }],
            IrParallelGrid {
                shape: vec![segment_count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..argument_count)
                    .map(|_| mxx_ir_core::node::GridInputMode::Broadcast)
                    .collect(),
            },
        );
        let pending = Pending::merge([self.pending, body_value.pending().remap(&sealed.remap)]);
        body_value.parallel_families(&node, &mut 0, &segment_count, pending)
    }

    pub fn parallel_select_mats(
        self,
        candidates: Vec<Family<Mat>>,
    ) -> Result<Family<Mat>, DslError> {
        if self.shape.len() != 1 || candidates.iter().any(|candidate| candidate.shape.len() != 1) {
            return Err(DslError::ParallelFamilyRank);
        }
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
                selector.select(branches).map(|output| (output, inputs, scope))
            })
        });
        let (body_value, explicit_inputs, scope) = body_result?;
        let sealed = SubgraphHandle::seal(
            "parallel-select-mats-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value];
        arguments.extend(candidates.iter().map(|candidate| candidate.value.clone()));
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let argument_count = arguments.len() - sealed.captures.len();
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            arguments,
            body_value.parallel_family_types(&count)?,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..argument_count)
                    .map(|_| mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    })
                    .chain(
                        (0..sealed.captures.len())
                            .map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                    )
                    .collect(),
            },
        );
        let pending = Pending::merge(
            std::iter::once(self.pending)
                .chain(candidates.into_iter().map(|candidate| candidate.pending))
                .chain(std::iter::once(body_value.pending().remap(&sealed.remap))),
        );
        body_value.parallel_families(&node, &mut 0, &count, pending)
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
/// A rank-N family whose logical value at coordinate `u` is `X[u]`.
///
/// `count` is the flat cardinality `∏_a shape[a]`; `shape` retains the axes needed to interpret
/// reindexing, selection, and gathering without collapsing a coordinate to an opaque integer.
pub struct Family<T> {
    value: ValueHandle,
    element_schema: T,
    count: IntExpr,
    shape: Vec<IntExpr>,
    pending: Pending,
}

pub enum FamilyAxisSelection<T> {
    Scalar(T),
    Family(Family<T>),
}

fn shape_count(shape: &[IntExpr]) -> IntExpr {
    // A Cartesian family has one element for every coordinate, hence |I|=∏_a n_a.
    shape
        .iter()
        .cloned()
        .reduce(|left, right| IntExpr::Mul(Box::new(left), Box::new(right)))
        .unwrap_or_else(|| IntExpr::constant(0))
}

pub trait IntoFamilyStaticIndices {
    fn into_family_indices(self) -> Vec<IndexExpr>;
}

impl IntoFamilyStaticIndices for usize {
    fn into_family_indices(self) -> Vec<IndexExpr> {
        vec![IndexExpr::constant(self)]
    }
}

impl IntoFamilyStaticIndices for i32 {
    fn into_family_indices(self) -> Vec<IndexExpr> {
        vec![IndexExpr::constant(self)]
    }
}

impl IntoFamilyStaticIndices for IndexExpr {
    fn into_family_indices(self) -> Vec<IndexExpr> {
        vec![self]
    }
}

impl IntoFamilyStaticIndices for Vec<IndexExpr> {
    fn into_family_indices(self) -> Vec<IndexExpr> {
        self
    }
}

pub trait IntoFamilyDynamicIndices {
    fn into_family_indices(self) -> Vec<Int>;
}

pub trait IntoFamilyAxisSelector {
    fn selector_parts(self) -> (ValueHandle, Pending, Option<Vec<IntExpr>>);
}

impl IntoFamilyAxisSelector for Int {
    fn selector_parts(self) -> (ValueHandle, Pending, Option<Vec<IntExpr>>) {
        (self.value, self.pending, None)
    }
}

impl IntoFamilyAxisSelector for Family<Int> {
    fn selector_parts(self) -> (ValueHandle, Pending, Option<Vec<IntExpr>>) {
        (self.value, self.pending, Some(self.shape))
    }
}

impl IntoFamilyDynamicIndices for Int {
    fn into_family_indices(self) -> Vec<Int> {
        vec![self]
    }
}

impl IntoFamilyDynamicIndices for Vec<Int> {
    fn into_family_indices(self) -> Vec<Int> {
        self
    }
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

impl FamilyElement for Preimage {
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

    /// Packs values into a one-dimensional family `X[i]=values[i]`.
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
            NodeKind::FamilyPack { shape: vec![count.clone()] },
            arguments,
            vec![WireType::Family {
                element: Box::new(first_value.wire_type().clone()),
                shape: vec![count.clone()],
            }],
        );
        Ok(Self {
            value: node.output(0).expect("packed family"),
            element_schema: first.clone(),
            count: count.clone(),
            shape: vec![count.clone()],
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
    indices: Vec<IndexExpr>,
    fallback_wire_type: WireType,
    parts: impl Fn(ValueHandle, Pending) -> T,
) -> T {
    let node = NodeHandle::new(
        NodeKind::FamilyGetStatic { indices },
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
        NodeKind::FamilyGetDynamic { rank: 1 },
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
    if family.shape.len() != 1 {
        return Err(DslError::ParallelMapRank);
    }
    // The sealed body is evaluated independently at each coordinate i, producing Y[i]=F(X[i]);
    // the input family is therefore reindexed by the grid axis and captures are broadcast.
    let count = family.count.clone();
    let schema = family.element_schema.schema();
    let (index_slot, (body_value, explicit_input, scope)) = with_loop_index(|index| {
        with_new_construction_scope(|scope| {
            let input = schema.placeholders();
            let input_values = input.flatten();
            let [input_value] = input_values.as_slice() else {
                panic!("scalar family item schema must contain one value")
            };
            let output = body(index, input);
            (output.normalize_for_family(), input_value.clone(), scope)
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
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    let node = NodeHandle::parallel_grid(
        sealed.handle.clone(),
        arguments,
        vec![WireType::Family { element: Box::new(wire_type), shape: vec![count.clone()] }],
        IrParallelGrid {
            // A rank-one grid has coordinates i in [0,count), so its output cardinality is count.
            shape: vec![count.clone()],
            index_slots: vec![index_slot],
            bindings: Vec::new(),
            input_modes: std::iter::once(mxx_ir_core::node::GridInputMode::Reindex {
                map: IndexMap::new([IndexExpr::Axis(0)]),
            })
            .chain((0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast))
            .collect(),
        },
    );
    let pending = Pending::merge([family.pending, body_value.pending().remap(&sealed.remap)]);
    body_value.parallel_families(&node, &mut 0, &count, pending)
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
    if source.shape.len() != 1 || indices.shape.len() != 1 {
        return Err(DslError::ParallelGatherRank);
    }
    // For each output coordinate i, the selector family supplies a source index s[i], yielding
    // Y[i]=X[s[i]] while the source family itself is broadcast into the body.
    let source_count = source.count.clone();
    let output_count = indices.count.clone();
    let source_element_type = family_element_wire_type(&source.value).unwrap_or(wire_type.clone());
    let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|_| {
        with_new_construction_scope(|scope| {
            let index = IntType.placeholders();
            let family_wire_type = WireType::Family {
                element: Box::new(source_element_type.clone()),
                shape: vec![source_count.clone()],
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
                count: source_count.clone(),
                shape: vec![source_count.clone()],
                pending: Pending::default(),
            };
            let output =
                scalar_family_get(&family, index.clone(), source_element_type.clone(), parts);
            (output, vec![index.value, family.value], scope)
        })
    });
    let sealed = SubgraphHandle::seal(
        body_name,
        scope,
        explicit_inputs,
        body_value.flatten(),
        CapturePolicy::BroadcastScalarsAndArtifactFamilies,
    )?;
    let node = NodeHandle::parallel_grid(
        sealed.handle.clone(),
        vec![indices.value, source.value],
        vec![WireType::Family {
            element: Box::new(source_element_type),
            shape: vec![output_count.clone()],
        }],
        IrParallelGrid {
            shape: vec![output_count.clone()],
            index_slots: vec![index_slot],
            bindings: Vec::new(),
            input_modes: vec![
                mxx_ir_core::node::GridInputMode::Reindex {
                    map: IndexMap::new([IndexExpr::Axis(0)]),
                },
                mxx_ir_core::node::GridInputMode::Broadcast,
            ],
        },
    );
    let pending = Pending::merge([
        indices.pending,
        source.pending,
        body_value.pending().remap(&sealed.remap),
    ]);
    body_value.parallel_families(&node, &mut 0, &output_count, pending)
}

fn family_element_wire_type(value: &ValueHandle) -> Option<WireType> {
    let element = match value.wire_type() {
        WireType::Family { element, .. } => element,
        _ => return None,
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
            WireType::Family { element: Box::new(element_type), shape: vec![count.clone()] };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: family_type.clone(), artifact },
            Vec::new(),
            vec![family_type],
        );
        let placeholder = Mat::source_input("__family_element".to_owned(), matrix_type, None);
        Self {
            value: node.output(0).expect("family"),
            element_schema: placeholder,
            count: count.clone(),
            shape: vec![count.clone()],
            pending: Pending::default(),
        }
    }

    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    pub fn shape(&self) -> &[IntExpr] {
        &self.shape
    }

    /// Applies a deterministic rank-N coordinate map while preserving the
    /// family element summary.
    pub fn reindex(self, output_shape: Vec<IntExpr>, map: IndexMap) -> Result<Self, DslError> {
        // The node represents Y[u]=X[f(u)]; only coordinate interpretation changes, not the
        // matrix element schema.
        if output_shape.is_empty() {
            return Err(DslError::Schema);
        }
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(self.element_schema.matrix_type.clone())),
            shape: output_shape.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::FamilyReindex { output_shape: output_shape.clone(), map },
            vec![self.value],
            vec![family_type],
        );
        let count = output_shape
            .clone()
            .into_iter()
            .reduce(|left, right| IntExpr::Mul(Box::new(left), Box::new(right)))
            .expect("nonempty shape");
        Ok(Self {
            value: node.output(0).expect("reindexed family"),
            element_schema: self.element_schema,
            count: count.clone(),
            shape: output_shape.clone(),
            pending: self.pending,
        })
    }

    /// Selects the named family axis using one runtime integer selector.
    pub fn select_axis(
        self,
        axis: usize,
        selector: impl IntoFamilyAxisSelector,
    ) -> Result<FamilyAxisSelection<Mat>, DslError> {
        // Removing axis a maps each remaining coordinate u to X[u with axis a=selector(u)].
        let (selector_value, selector_pending, selector_shape) = selector.selector_parts();
        if axis >= self.shape.len() {
            return Err(DslError::Schema);
        }
        let mut output_shape = self.shape.clone();
        output_shape.remove(axis);
        if let Some(selector_shape) = &selector_shape {
            if *selector_shape != output_shape {
                return Err(DslError::Schema);
            }
        }
        let pending = Pending::merge([self.pending, selector_pending]);
        let element_type = WireType::Matrix(self.element_schema.matrix_type.clone());
        let output_type = if output_shape.is_empty() {
            element_type.clone()
        } else {
            WireType::Family {
                element: Box::new(element_type.clone()),
                shape: output_shape.clone(),
            }
        };
        let node = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis },
            vec![self.value, selector_value],
            vec![output_type],
        );
        let value = node.output(0).expect("selected family axis");
        if output_shape.is_empty() {
            Ok(FamilyAxisSelection::Scalar(Mat {
                value,
                matrix_type: self.element_schema.matrix_type,
                pending,
            }))
        } else {
            Ok(FamilyAxisSelection::Family(Self {
                value,
                element_schema: self.element_schema,
                count: shape_count(&output_shape),
                shape: output_shape,
                pending,
            }))
        }
    }

    /// Performs a runtime-dependent gather of full source coordinates.
    pub fn gather(
        self,
        output_shape: Vec<IntExpr>,
        selectors: Vec<Family<Int>>,
    ) -> Result<Self, DslError> {
        // One selector family per source axis defines f(u)=(s_0[u],...,s_{r-1}[u]) and
        // Y[u]=X[f(u)].
        if output_shape.is_empty() || selectors.is_empty() {
            return Err(DslError::Schema);
        }
        let mut arguments = vec![self.value];
        arguments.extend(selectors.iter().map(|selector| selector.value.clone()));
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(self.element_schema.matrix_type.clone())),
            shape: output_shape.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::FamilyGather {
                output_shape: output_shape.clone(),
                input_rank: selectors.len(),
            },
            arguments,
            vec![family_type],
        );
        let count = output_shape
            .clone()
            .into_iter()
            .reduce(|left, right| IntExpr::Mul(Box::new(left), Box::new(right)))
            .expect("nonempty shape");
        let pending = Pending::merge(
            std::iter::once(self.pending)
                .chain(selectors.into_iter().map(|selector| selector.pending)),
        );
        Ok(Self {
            value: node.output(0).expect("gathered family"),
            element_schema: self.element_schema,
            count,
            shape: output_shape,
            pending,
        })
    }

    pub fn element_type(&self) -> &MatrixType {
        &self.element_schema.matrix_type
    }

    pub fn get_static(&self, indices: impl IntoFamilyStaticIndices) -> Mat {
        // Static coordinates directly select X[u] and preserve the element matrix type.
        let indices = indices.into_family_indices();
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices },
            vec![self.value.clone()],
            vec![WireType::Matrix(self.element_schema.matrix_type.clone())],
        );
        Mat {
            value: node.output(0).expect("family element"),
            matrix_type: self.element_schema.matrix_type.clone(),
            pending: self.pending.clone(),
        }
    }

    pub fn get(&self, indices: impl IntoFamilyDynamicIndices) -> Mat {
        // Dynamic coordinates select X[u] at runtime; pending dependencies include both X and u.
        let indices = indices.into_family_indices();
        let pending = Pending::merge(
            std::iter::once(self.pending.clone())
                .chain(indices.iter().map(|index| index.pending.clone())),
        );
        let mut arguments = vec![self.value.clone()];
        arguments.extend(indices.iter().map(|index| index.value.clone()));
        let node = NodeHandle::new(
            NodeKind::FamilyGetDynamic { rank: indices.len() },
            arguments,
            vec![WireType::Matrix(self.element_schema.matrix_type.clone())],
        );
        Mat {
            value: node.output(0).expect("dynamic family element"),
            matrix_type: self.element_schema.matrix_type.clone(),
            pending,
        }
    }

    /// Selects one same-shaped matrix family without materializing the other branches.
    pub fn select(selector: Int, branches: Vec<Self>) -> Result<Self, DslError> {
        // Branch selection is Y[u]=X_{selector}[u]. Shape and element checks ensure every branch
        // denotes the same coordinate domain and matrix schema.
        let Some(first) = branches.first() else {
            return Err(DslError::Schema);
        };
        if branches.iter().any(|branch| {
            branch.shape != first.shape ||
                branch.element_schema.matrix_type != first.element_schema.matrix_type
        }) {
            return Err(DslError::FamilyCountMismatch);
        }
        let pending = Pending::merge(
            std::iter::once(selector.pending.clone())
                .chain(branches.iter().map(|branch| branch.pending.clone())),
        );
        let mut arguments = vec![selector.value];
        arguments.extend(branches.iter().map(|branch| branch.value.clone()));
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(first.element_schema.matrix_type.clone())),
            shape: first.shape.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Select { count: IntExpr::constant(branches.len()) },
            arguments,
            vec![family_type],
        );
        Ok(Self {
            value: node.output(0).expect("selected matrix family"),
            element_schema: first.element_schema.clone(),
            count: first.count.clone(),
            shape: first.shape.clone(),
            pending,
        })
    }

    pub fn parallel_map(self, body: impl FnOnce(LoopIndex, Mat) -> Mat) -> Result<Self, DslError> {
        self.parallel_map_values(body)
    }

    pub fn parallel_gather(self, indices: Family<Int>) -> Result<Self, DslError> {
        if self.shape.len() != 1 || indices.shape.len() != 1 {
            return Err(DslError::ParallelGatherRank);
        }
        let source_count = self.count.clone();
        let output_count = indices.count.clone();
        let matrix_type = self.element_schema.matrix_type.clone();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let index = IntType.placeholders();
                let source = Family::<Mat>::source_input(
                    "gather-source".to_owned(),
                    matrix_type,
                    source_count,
                    None,
                );
                let output = source.get(index.clone());
                (output, vec![index.value, source.value], scope)
            })
        });
        let sealed = SubgraphHandle::seal(
            "parallel-gather-mat-body",
            scope,
            explicit_inputs,
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            vec![indices.value, self.value],
            body_value.parallel_family_types(&output_count)?,
            IrParallelGrid {
                shape: vec![output_count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: vec![
                    mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    },
                    mxx_ir_core::node::GridInputMode::Broadcast,
                ],
            },
        );
        let pending = Pending::merge([
            indices.pending,
            self.pending,
            body_value.pending().remap(&sealed.remap),
        ]);
        body_value.parallel_families(&node, &mut 0, &output_count, pending)
    }

    pub fn parallel_zip_many_values<R: ParallelOutput>(
        families: Vec<Self>,
        body: impl FnOnce(LoopIndex, Vec<Mat>) -> R,
    ) -> Result<R::Families, DslError> {
        let Some(first) = families.first() else {
            return Err(DslError::Schema);
        };
        if families.iter().any(|family| family.shape.len() != 1) {
            return Err(DslError::ParallelZipRank);
        }
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
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle,
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..families.len())
                    .map(|_| mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    })
                    .chain(
                        (0..sealed.captures.len())
                            .map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                    )
                    .collect(),
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

    pub fn parallel_zip_many_with_broadcast_values<R: ParallelOutput>(
        zipped: Vec<Self>,
        broadcast: Vec<Self>,
        body: impl FnOnce(LoopIndex, Vec<Mat>, Vec<Self>) -> Result<R, DslError>,
    ) -> Result<R::Families, DslError> {
        let Some(first) = zipped.first() else {
            return Err(DslError::Schema);
        };
        if zipped.iter().chain(&broadcast).any(|family| family.shape.len() != 1) {
            return Err(DslError::ParallelZipRank);
        }
        let count = first.count.clone();
        if zipped.iter().any(|family| family.count != count) {
            return Err(DslError::FamilyCountMismatch);
        }
        let zipped_types = zipped
            .iter()
            .map(|family| family.element_schema.matrix_type.clone())
            .collect::<Vec<_>>();
        let broadcast_types = broadcast
            .iter()
            .map(|family| (family.element_schema.matrix_type.clone(), family.count.clone()))
            .collect::<Vec<_>>();
        let (index_slot, body_result) = with_loop_index(|index| -> Result<_, DslError> {
            with_new_construction_scope(|scope| -> Result<_, DslError> {
                let zipped_inputs = zipped_types
                    .into_iter()
                    .enumerate()
                    .map(|(index, matrix_type)| {
                        Mat::source_input(format!("zip-item-{index}"), matrix_type, None)
                    })
                    .collect::<Vec<_>>();
                let broadcast_inputs = broadcast_types
                    .into_iter()
                    .enumerate()
                    .map(|(index, (matrix_type, count))| {
                        Family::<Mat>::source_input(
                            format!("broadcast-family-{index}"),
                            matrix_type,
                            count,
                            None,
                        )
                    })
                    .collect::<Vec<_>>();
                let explicit_inputs = zipped_inputs
                    .iter()
                    .map(|input| input.value.clone())
                    .chain(broadcast_inputs.iter().map(|input| input.value.clone()))
                    .collect();
                let output = body(index, zipped_inputs, broadcast_inputs)?;
                Ok((output, explicit_inputs, scope))
            })
        });
        let (body_value, explicit_inputs, scope) = body_result?;
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-zip-many-with-broadcast-body",
            scope,
            explicit_inputs,
            body_outputs,
            CapturePolicy::Reject,
        )?;
        let mut arguments = zipped.iter().map(|family| family.value.clone()).collect::<Vec<_>>();
        arguments.extend(broadcast.iter().map(|family| family.value.clone()));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle,
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..zipped.len())
                    .map(|_| mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    })
                    .chain(
                        (0..broadcast.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                    )
                    .collect(),
            },
        );
        let pending = Pending::merge(
            zipped
                .into_iter()
                .map(|family| family.pending)
                .chain(broadcast.into_iter().map(|family| family.pending))
                .chain(std::iter::once(body_value.pending().remap(&sealed.remap))),
        );
        let mut next_port = 0;
        body_value.parallel_families(&node, &mut next_port, &count, pending)
    }

    pub fn parallel_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Mat) -> R,
    ) -> Result<R::Families, DslError> {
        if self.shape.len() != 1 {
            // This API exposes one flat LoopIndex and ParallelOutput builds a
            // rank-one result. Reject a Cartesian input instead of silently
            // replacing its logical shape with [product(shape)].
            return Err(DslError::ParallelMapRank);
        }
        let outer_family = self.value.clone();
        let count = self.count.clone();
        let element_type = self.element_schema.matrix_type.clone();
        let (index_slot, (body_value, explicit_input, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let input = Mat::source_input("item".to_owned(), element_type, None);
                let output = body(index, input.clone());
                (output, input.value, scope)
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
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: std::iter::once(mxx_ir_core::node::GridInputMode::Reindex {
                    map: IndexMap::new([IndexExpr::Axis(0)]),
                })
                .chain(
                    (0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                )
                .collect(),
            },
        );
        let pending = Pending::merge([self.pending, body_value.pending().remap(&sealed.remap)]);
        let mut next_port = 0;
        body_value.parallel_families(&node, &mut next_port, &count, pending)
    }

    pub fn parallel_threshold_decode_ints(
        self,
        plaintext_modulus: impl Into<IntExpr>,
        length: usize,
    ) -> Result<Vec<Family<Int>>, DslError> {
        if self.shape.len() != 1 {
            return Err(DslError::ParallelFamilyRank);
        }
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
                    shape: family.shape,
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
        if self.shape.len() != 1 {
            return Err(DslError::ParallelFamilyRank);
        }
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
                WireType::Family { element: Box::new(WireType::Bool), shape: vec![count.clone()] };
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
                shape: vec![count.clone()],
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
                WireType::Family { element: Box::new(WireType::Int), shape: vec![count.clone()] };
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
                shape: vec![count.clone()],
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
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(DslError::ParallelZipRank);
        }
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
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle,
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..2)
                    .map(|_| mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    })
                    .chain(
                        (0..sealed.captures.len())
                            .map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                    )
                    .collect(),
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
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(DslError::ParallelZipRank);
        }
        // The second reindex reads other[i + offset] for every i in self. A
        // nonnegative canonical difference is the construction-time proof
        // that the final read remains inside the other family's domain.
        let remaining_capacity = IntExpr::Sub(
            Box::new(other.count.clone()),
            Box::new(IntExpr::Add(
                Box::new(self.count.clone()),
                Box::new(IntExpr::constant(offset)),
            )),
        )
        .canonicalize();
        if !matches!(remaining_capacity, IntExpr::Const(remaining) if remaining >= 0.into()) {
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
            "parallel-zip-offset-body",
            scope,
            explicit_inputs,
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![self.value, other.value];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle,
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: vec![
                    mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    },
                    mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Add(
                            Box::new(IndexExpr::Axis(0)),
                            Box::new(IndexExpr::Constant(offset.into())),
                        )]),
                    },
                ]
                .into_iter()
                .chain(
                    (0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                )
                .collect(),
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
        if self.shape.len() != 1 || second.shape.len() != 1 || third.shape.len() != 1 {
            return Err(DslError::ParallelZipRank);
        }
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
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle,
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..3)
                    .map(|_| mxx_ir_core::node::GridInputMode::Reindex {
                        map: IndexMap::new([IndexExpr::Axis(0)]),
                    })
                    .chain(
                        (0..sealed.captures.len())
                            .map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                    )
                    .collect(),
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
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    NodeHandle::parallel_grid(
        sealed.handle.clone(),
        arguments,
        output_types,
        IrParallelGrid {
            shape: vec![input.count.clone()],
            index_slots: vec![index_slot],
            bindings: Vec::new(),
            input_modes: std::iter::once(mxx_ir_core::node::GridInputMode::Reindex {
                map: IndexMap::new([IndexExpr::Axis(0)]),
            })
            .chain((0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast))
            .collect(),
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

/// A rank-N Cartesian grid. Runtime storage is flat and row-major; axes remain
/// explicit in the frozen IR through `NodeKind::ParallelGrid`.
pub struct GridRange {
    shape: Vec<IntExpr>,
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

    pub fn grid(shape: impl Into<Vec<IntExpr>>) -> GridRange {
        GridRange { shape: shape.into() }
    }
}

impl GridRange {
    /// Evaluates `body` at every rank-N coordinate `u` and stores the result as `Y[u]`.
    pub fn map(self, body: impl FnOnce(Vec<LoopIndex>) -> Mat) -> Result<Family<Mat>, DslError> {
        if self.shape.is_empty() {
            return Err(DslError::Schema);
        }
        let shape = self.shape;
        let (index_slots, (body_value, scope)) = with_grid_indices(shape.len(), |indices| {
            with_new_construction_scope(|scope| (body(indices), scope))
        });
        let element_type = body_value.matrix_type.clone();
        let sealed = SubgraphHandle::seal(
            "parallel-grid-body",
            scope,
            Vec::new(),
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let arguments = sealed.captures.iter().map(|capture| capture.outer.clone()).collect();
        let input_modes = (0..sealed.captures.len())
            .map(|_| mxx_ir_core::node::GridInputMode::Broadcast)
            .collect();
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(element_type.clone())),
            shape: shape.clone(),
        };
        // The grid node represents the Cartesian domain ∏_a [0,shape[a]); each body output is
        // therefore one element of the resulting family at its corresponding coordinate.
        let node = NodeHandle::parallel_grid(
            sealed.handle,
            arguments,
            vec![family_type],
            IrParallelGrid { shape: shape.clone(), index_slots, bindings: Vec::new(), input_modes },
        );
        let count = shape
            .clone()
            .into_iter()
            .reduce(|left, right| IntExpr::Mul(Box::new(left), Box::new(right)))
            .expect("nonempty grid shape");
        Ok(Family {
            value: node.output(0).expect("parallel grid output"),
            element_schema: body_value,
            count: count.clone(),
            shape: shape.clone(),
            pending: Pending::default(),
        })
    }
}

fn with_grid_indices<T>(rank: usize, body: impl FnOnce(Vec<LoopIndex>) -> T) -> (Vec<u32>, T) {
    if rank == 0 {
        return (Vec::new(), body(Vec::new()));
    }
    let mut slots = Vec::with_capacity(rank);
    let mut indices = Vec::with_capacity(rank);
    for _ in 0..rank {
        let slot = LOOP_BINDER_DEPTH.with(|depth| {
            let slot = depth.get();
            depth.set(slot + 1);
            slot
        });
        slots.push(slot);
        indices.push(LoopIndex { expression: IntExpr::LoopIndex(slot) });
    }
    let output = body(indices);
    for _ in 0..rank {
        LOOP_BINDER_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
    (slots, output)
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
                    .map(|next_state| (next_state, explicit_inputs, scope))
            })
        });
        let (next_state, explicit_inputs, scope) = body_result?;
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
        S::from_values(&state_schema, &values, pending)
    }
}

impl ParallelRange {
    pub fn map(self, body: impl FnOnce(LoopIndex) -> Mat) -> Result<Family<Mat>, DslError> {
        self.map_values(body)
    }

    pub fn try_map(
        self,
        body: impl FnOnce(LoopIndex) -> Result<Mat, DslError>,
    ) -> Result<Family<Mat>, DslError> {
        self.try_map_values(body)
    }

    pub fn map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex) -> R,
    ) -> Result<R::Families, DslError> {
        // A range is the rank-one case: for i in [0,count), evaluate Y[i]=F(i), then wrap each
        // body output in a family of shape [count].
        let count = self.count;
        let (index_slot, (body_value, scope)) =
            with_loop_index(|index| with_new_construction_scope(|scope| (body(index), scope)));
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-range-body",
            scope,
            Vec::new(),
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let arguments = sealed.captures.iter().map(|capture| capture.outer.clone()).collect();
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..sealed.captures.len())
                    .map(|_| mxx_ir_core::node::GridInputMode::Broadcast)
                    .collect(),
            },
        );
        let pending = body_value.pending().remap(&sealed.remap);
        body_value.parallel_families(&node, &mut 0, &count, pending)
    }

    pub fn try_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex) -> Result<R, DslError>,
    ) -> Result<R::Families, DslError> {
        // The fallible form has the same equation Y[i]=F(i); construction errors only abort graph
        // creation before the rank-one family is emitted.
        let count = self.count;
        let (index_slot, body_result) = with_loop_index(|index| {
            with_new_construction_scope(|scope| body(index).map(|body_value| (body_value, scope)))
        });
        let (body_value, scope) = body_result?;
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-range-body",
            scope,
            Vec::new(),
            body_outputs,
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let arguments = sealed.captures.iter().map(|capture| capture.outer.clone()).collect();
        let family_outputs = body_value.parallel_family_types(&count)?;
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            arguments,
            family_outputs,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: (0..sealed.captures.len())
                    .map(|_| mxx_ir_core::node::GridInputMode::Broadcast)
                    .collect(),
            },
        );
        let pending = body_value.pending().remap(&sealed.remap);
        body_value.parallel_families(&node, &mut 0, &count, pending)
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
        if self.0.shape.len() != 1 || self.1.shape.len() != 1 {
            return Err(DslError::ParallelZipRank);
        }
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
                let output = body(index, (first, second))?;
                Ok::<_, DslError>((output, explicit_inputs, scope))
            })
        });
        let (body_value, explicit_inputs, scope) = body_result?;
        if explicit_inputs.len() != 2 {
            return Err(DslError::Schema);
        }
        finish_parallel_zip(
            count,
            vec![self.0.value, self.1.value],
            vec![self.0.pending, self.1.pending],
            body_value,
            explicit_inputs,
            scope,
            index_slot,
            "parallel-zip-bundle2-body",
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
        if self.0.shape.len() != 1 || self.1.shape.len() != 1 || self.2.shape.len() != 1 {
            return Err(DslError::ParallelZipRank);
        }
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
                let output = body(index, (first, second, third))?;
                Ok::<_, DslError>((output, explicit_inputs, scope))
            })
        });
        let (body_value, explicit_inputs, scope) = body_result?;
        if explicit_inputs.len() != 3 {
            return Err(DslError::Schema);
        }
        finish_parallel_zip(
            count,
            vec![self.0.value, self.1.value, self.2.value],
            vec![self.0.pending, self.1.pending, self.2.pending],
            body_value,
            explicit_inputs,
            scope,
            index_slot,
            "parallel-zip-bundle3-body",
        )
    }
}

fn finish_parallel_zip<R: ParallelOutput>(
    count: IntExpr,
    mut arguments: Vec<ValueHandle>,
    pendings: Vec<Pending>,
    body_value: R,
    explicit_inputs: Vec<ValueHandle>,
    scope: mxx_ir_core::ConstructionScopeId,
    index_slot: u32,
    body_name: &'static str,
) -> Result<R::Families, DslError> {
    let body_outputs = body_value.flatten();
    let sealed = SubgraphHandle::seal(
        body_name,
        scope,
        explicit_inputs,
        body_outputs,
        CapturePolicy::BroadcastScalarsAndArtifactFamilies,
    )?;
    let zipped_count = arguments.len();
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    let node = NodeHandle::parallel_grid(
        sealed.handle.clone(),
        arguments,
        body_value.parallel_family_types(&count)?,
        IrParallelGrid {
            shape: vec![count.clone()],
            index_slots: vec![index_slot],
            bindings: Vec::new(),
            input_modes: (0..zipped_count)
                .map(|_| mxx_ir_core::node::GridInputMode::Reindex {
                    map: IndexMap::new([IndexExpr::Axis(0)]),
                })
                .chain(
                    (0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                )
                .collect(),
        },
    );
    let pending = Pending::merge(
        pendings.into_iter().chain(std::iter::once(body_value.pending().remap(&sealed.remap))),
    );
    body_value.parallel_families(&node, &mut 0, &count, pending)
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
pub struct Pending;

impl Pending {
    pub fn merge(values: impl IntoIterator<Item = Pending>) -> Self {
        drop(values);
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
        let node = NodeHandle::new(
            NodeKind::EvaluateInt(expression.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        );
        let expression = node.output(0).expect("evaluated integer expression");
        Int { value: expression, pending: Pending::default() }.add(Int::constant(0))
    }

    #[track_caller]
    pub fn int_family_input(
        &self,
        name: impl Into<String>,
        fixed_count: impl Into<IntExpr>,
    ) -> Family<Int> {
        let count = fixed_count.into();
        let wire_type =
            WireType::Family { element: Box::new(WireType::Int), shape: vec![count.clone()] };
        let node = NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: wire_type.clone(), artifact: None },
            Vec::new(),
            vec![wire_type],
        );
        Family {
            value: node.output(0).expect("integer family input"),
            element_schema: IntType.placeholders(),
            count: count.clone(),
            shape: vec![count.clone()],
            pending: Pending::default(),
        }
    }

    pub fn output(mut self, name: impl Into<String>, mat: Mat) -> Result<Self, DslError> {
        self.insert_output(name.into(), mat, None)?;
        Ok(self)
    }

    pub fn preimage_output(
        mut self,
        name: impl Into<String>,
        preimage: Preimage,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), preimage.value, preimage.pending, None)?;
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

    pub fn public_preimage_output(
        mut self,
        name: impl Into<String>,
        preimage: Preimage,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            preimage.value,
            preimage.pending,
            Some(ArtifactConfidentiality::Public),
        )?;
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

    pub fn private_preimage_output(
        mut self,
        name: impl Into<String>,
        preimage: Preimage,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            preimage.value,
            preimage.pending,
            Some(ArtifactConfidentiality::Private),
        )?;
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

    pub fn preimage_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Preimage>,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), family.value, family.pending, None)?;
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

    pub fn public_preimage_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Preimage>,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            family.value,
            family.pending,
            Some(ArtifactConfidentiality::Public),
        )?;
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

    pub fn private_preimage_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Preimage>,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            family.value,
            family.pending,
            Some(ArtifactConfidentiality::Private),
        )?;
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
        Ok(vec![WireType::Family {
            element: Box::new(WireType::Matrix(self.matrix_type.clone())),
            shape: vec![count.clone()],
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
            shape: vec![count.clone()],
            pending,
        })
    }
}

impl ParallelOutput for Preimage {
    type Families = Family<Preimage>;

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
        // Mapping a witness over i produces a family K[i] with the same witness matrix schema;
        // the relation marker remains part of each family element type.
        Ok(vec![WireType::Family {
            element: Box::new(WireType::Preimage(self.matrix_type.clone())),
            shape: vec![count.clone()],
        }])
    }

    fn parallel_families(
        self,
        node: &NodeHandle,
        next_port: &mut u32,
        count: &IntExpr,
        pending: Pending,
    ) -> Result<Self::Families, DslError> {
        // The body output port is lifted to the family coordinate i without materializing K[i].
        let value = node.output(*next_port).ok_or(DslError::Schema)?;
        *next_port += 1;
        Ok(Family {
            value,
            element_schema: Preimage {
                value: self.value,
                matrix_type: self.matrix_type,
                pending: Pending::default(),
            },
            count: count.clone(),
            shape: vec![count.clone()],
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
            .map(|element| WireType::Family {
                element: Box::new(element),
                shape: vec![count.clone()],
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
                shape: vec![count.clone()],
                pending: pending.clone(),
            },
            values: trapdoor_value,
            element_schema: schema,
            count: count.clone(),
            shape: vec![count.clone()],
            pending,
        })
    }
}

macro_rules! scalar_parallel_output {
    ($value:ty, $wire_type:expr) => {
        impl ParallelOutput for $value {
            type Families = Family<$value>;

            fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
                Ok(vec![WireType::Family {
                    element: Box::new($wire_type),
                    shape: vec![count.clone()],
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
                    element_schema: self,
                    count: count.clone(),
                    shape: vec![count.clone()],
                    pending,
                })
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
        // A schema round-trip must retain `Preimage(MatrixType)`, not weaken it to `MatrixType`;
        // otherwise a subgraph boundary could silently authorize arbitrary multiplication.
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
        // Placeholder construction carries the typed witness wire so a subgraph can consume
        // `B*K=T` only through an explicit relation-aware operation.
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
        TrapdoorFamilyType { element: self.element_schema.clone(), shape: self.shape.clone() }
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
                count: shape_count(&schema.shape),
                shape: schema.shape.clone(),
                pending: pending.clone(),
            },
            values: trapdoors.clone(),
            element_schema: schema.element.clone(),
            count: shape_count(&schema.shape),
            shape: schema.shape.clone(),
            pending,
        })
    }
}

impl GraphValueSchema for TrapdoorFamilyType {
    type Value = TrapdoorFamily;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let public = Family::<Mat>::source_input(
            argument_name(next, "trapdoor-public-family"),
            self.element.matrix.clone(),
            shape_count(&self.shape),
            None,
        );
        TrapdoorFamily::source_input(
            argument_name(next, "trapdoor-secret-family"),
            public,
            self.element.clone(),
            self.shape.clone(),
            None,
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.element
            .wire_types()
            .into_iter()
            .map(|element| WireType::Family {
                element: Box::new(element),
                shape: self.shape.clone(),
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
            shape: self.shape.clone(),
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
            count: shape_count(&schema.shape),
            shape: schema.shape.clone(),
            pending,
        })
    }
}

impl GraphValue for Family<Preimage> {
    type Schema = PreimageFamilyType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        // Family schema preserves both the witness element type and every coordinate extent.
        PreimageFamilyType {
            element: self.element_schema.matrix_type.clone(),
            shape: self.shape.clone(),
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
            element_schema: Preimage {
                value: value.clone(),
                matrix_type: schema.element.clone(),
                pending: Pending::default(),
            },
            count: schema
                .shape
                .iter()
                .cloned()
                .reduce(|left, right| IntExpr::Mul(Box::new(left), Box::new(right)))
                .ok_or(DslError::Schema)?,
            shape: schema.shape.clone(),
            pending,
        })
    }
}

impl GraphValueSchema for PreimageFamilyType {
    type Value = Family<Preimage>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        // A family placeholder denotes K[u] for every u in the declared Cartesian shape, with
        // each element retaining the `B*K[u]=T[u]` witness marker.
        let wire_type = WireType::Family {
            element: Box::new(WireType::Preimage(self.element.clone())),
            shape: self.shape.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "preimage-family"),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type],
        );
        Family {
            value: node.output(0).expect("preimage family argument"),
            element_schema: Preimage {
                value: node.output(0).expect("preimage element schema"),
                matrix_type: self.element.clone(),
                pending: Pending::default(),
            },
            count: self
                .shape
                .iter()
                .cloned()
                .reduce(|left, right| IntExpr::Mul(Box::new(left), Box::new(right)))
                .expect("nonempty family shape"),
            shape: self.shape.clone(),
            pending: Pending::default(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Family {
            element: Box::new(WireType::Preimage(self.element.clone())),
            shape: self.shape.clone(),
        }]
    }
}

impl Family<Preimage> {
    /// Creates a family of typed witnesses `K[u]`; each element remains associated with its
    /// public relation `B[u]*K[u]=T[u]` across graph and artifact boundaries.
    fn source_input(
        name: String,
        matrix_type: MatrixType,
        shape: Vec<IntExpr>,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let element_type = WireType::Preimage(matrix_type.clone());
        let family_type =
            WireType::Family { element: Box::new(element_type), shape: shape.clone() };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: family_type.clone(), artifact },
            Vec::new(),
            vec![family_type],
        );
        let placeholder = Preimage {
            value: node.output(0).expect("preimage family"),
            matrix_type: matrix_type.clone(),
            pending: Pending::default(),
        };
        Self {
            value: node.output(0).expect("preimage family"),
            element_schema: placeholder,
            count: shape_count(&shape),
            shape,
            pending: Pending::default(),
        }
    }

    pub fn shape(&self) -> &[IntExpr] {
        &self.shape
    }

    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    pub fn element_type(&self) -> &MatrixType {
        &self.element_schema.matrix_type
    }

    /// Selects one same-shaped preimage family while retaining its relation type.
    pub fn select(selector: Int, branches: Vec<Self>) -> Result<Self, DslError> {
        // Branch selection is K[u]=K_selector[u]. Equal shapes and witness schemas ensure that
        // the selected family remains a typed preimage family at every coordinate.
        let Some(first) = branches.first() else {
            return Err(DslError::Schema);
        };
        if branches.iter().any(|branch| {
            branch.shape != first.shape ||
                branch.element_schema.matrix_type != first.element_schema.matrix_type
        }) {
            return Err(DslError::FamilyCountMismatch);
        }
        let pending = Pending::merge(
            std::iter::once(selector.pending.clone())
                .chain(branches.iter().map(|branch| branch.pending.clone())),
        );
        let mut arguments = vec![selector.value];
        arguments.extend(branches.iter().map(|branch| branch.value.clone()));
        let family_type = WireType::Family {
            element: Box::new(WireType::Preimage(first.element_schema.matrix_type.clone())),
            shape: first.shape.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Select { count: IntExpr::constant(branches.len()) },
            arguments,
            vec![family_type],
        );
        Ok(Self {
            value: node.output(0).expect("selected preimage family"),
            element_schema: first.element_schema.clone(),
            count: first.count.clone(),
            shape: first.shape.clone(),
            pending,
        })
    }

    pub fn parallel_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Preimage) -> R,
    ) -> Result<R::Families, DslError> {
        if self.shape.len() != 1 {
            // A typed preimage relation is indexed by the full family
            // coordinate. Flattening that coordinate would also invalidate
            // the source/target relation views, so fail at the DSL boundary.
            return Err(DslError::ParallelMapRank);
        }
        // For each branch coordinate i, the body maps a witness K[i] to F(K[i]); relation typing
        // is retained by the output family schema and no witness is materialized implicitly.
        let outer_family = self.value.clone();
        let count = self.count.clone();
        let matrix_type = self.element_schema.matrix_type.clone();
        let (index_slot, (body_value, explicit_input, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let wire_type = WireType::Preimage(matrix_type.clone());
                let node = NodeHandle::new(
                    NodeKind::Input {
                        name: "preimage-item".to_owned(),
                        wire_type: wire_type.clone(),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![wire_type],
                );
                let input = Preimage {
                    value: node.output(0).expect("preimage family item"),
                    matrix_type,
                    pending: Pending::default(),
                };
                let output = body(index, input.clone());
                (output, input.value, scope)
            })
        });
        let sealed = SubgraphHandle::seal(
            "parallel-map-preimage-body",
            scope,
            vec![explicit_input],
            body_value.flatten(),
            CapturePolicy::BroadcastScalarsAndArtifactFamilies,
        )?;
        let mut arguments = vec![outer_family];
        arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
        let node = NodeHandle::parallel_grid(
            sealed.handle.clone(),
            arguments,
            body_value.parallel_family_types(&count)?,
            IrParallelGrid {
                shape: vec![count.clone()],
                index_slots: vec![index_slot],
                bindings: Vec::new(),
                input_modes: std::iter::once(mxx_ir_core::node::GridInputMode::Reindex {
                    map: IndexMap::new([IndexExpr::Axis(0)]),
                })
                .chain(
                    (0..sealed.captures.len()).map(|_| mxx_ir_core::node::GridInputMode::Broadcast),
                )
                .collect(),
            },
        );
        let pending = Pending::merge([self.pending, body_value.pending().remap(&sealed.remap)]);
        body_value.parallel_families(&node, &mut 0, &count, pending)
    }

    pub fn get_static(&self, indices: impl IntoFamilyStaticIndices) -> Preimage {
        // Static access returns the witness K[u] itself, preserving its B*K=T relation marker.
        let indices = indices.into_family_indices();
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices },
            vec![self.value.clone()],
            vec![WireType::Preimage(self.element_schema.matrix_type.clone())],
        );
        Preimage {
            value: node.output(0).expect("preimage family element"),
            matrix_type: self.element_schema.matrix_type.clone(),
            pending: self.pending.clone(),
        }
    }

    pub fn get(&self, indices: impl IntoFamilyDynamicIndices) -> Preimage {
        // Dynamic access returns K[u] at runtime and carries index dependencies alongside the
        // family dependency; it does not convert the selected witness to an ordinary matrix.
        let indices = indices.into_family_indices();
        let pending = Pending::merge(
            std::iter::once(self.pending.clone())
                .chain(indices.iter().map(|index| index.pending.clone())),
        );
        let mut arguments = vec![self.value.clone()];
        arguments.extend(indices.iter().map(|index| index.value.clone()));
        let node = NodeHandle::new(
            NodeKind::FamilyGetDynamic { rank: indices.len() },
            arguments,
            vec![WireType::Preimage(self.element_schema.matrix_type.clone())],
        );
        Preimage {
            value: node.output(0).expect("dynamic preimage family element"),
            matrix_type: self.element_schema.matrix_type.clone(),
            pending,
        }
    }

    pub fn reindex(self, output_shape: Vec<IntExpr>, map: IndexMap) -> Result<Self, DslError> {
        // Reindexing preserves the witness element: K'[u]=K[f(u)], so every selected coordinate
        // still has the same typed source relation.
        if output_shape.is_empty() {
            return Err(DslError::Schema);
        }
        let family_type = WireType::Family {
            element: Box::new(WireType::Preimage(self.element_schema.matrix_type.clone())),
            shape: output_shape.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::FamilyReindex { output_shape: output_shape.clone(), map },
            vec![self.value],
            vec![family_type],
        );
        let count = output_shape
            .iter()
            .cloned()
            .reduce(|left, right| IntExpr::Mul(Box::new(left), Box::new(right)))
            .expect("nonempty shape");
        Ok(Self {
            value: node.output(0).expect("reindexed preimage family"),
            element_schema: self.element_schema,
            count,
            shape: output_shape,
            pending: self.pending,
        })
    }

    pub fn select_axis(
        self,
        axis: usize,
        selector: impl IntoFamilyAxisSelector,
    ) -> Result<FamilyAxisSelection<Preimage>, DslError> {
        // Axis selection changes only coordinates: K'[u]=K[u with axis a=selector(u)]. The
        // selected element remains a Preimage when all axes are reduced.
        if axis >= self.shape.len() {
            return Err(DslError::Schema);
        }
        let (selector_value, selector_pending, selector_shape) = selector.selector_parts();
        let mut output_shape = self.shape.clone();
        output_shape.remove(axis);
        if selector_shape.as_ref().is_some_and(|shape| *shape != output_shape) {
            return Err(DslError::Schema);
        }
        let element_type = WireType::Preimage(self.element_schema.matrix_type.clone());
        let output_type = if output_shape.is_empty() {
            element_type.clone()
        } else {
            WireType::Family { element: Box::new(element_type), shape: output_shape.clone() }
        };
        let node = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis },
            vec![self.value, selector_value],
            vec![output_type],
        );
        let pending = Pending::merge([self.pending, selector_pending]);
        let value = node.output(0).expect("selected preimage family");
        if output_shape.is_empty() {
            Ok(FamilyAxisSelection::Scalar(Preimage {
                value,
                matrix_type: self.element_schema.matrix_type,
                pending,
            }))
        } else {
            Ok(FamilyAxisSelection::Family(Self {
                value,
                element_schema: self.element_schema,
                count: shape_count(&output_shape),
                shape: output_shape,
                pending,
            }))
        }
    }

    pub fn gather(
        self,
        output_shape: Vec<IntExpr>,
        selectors: Vec<Family<Int>>,
    ) -> Result<Self, DslError> {
        // Gather applies f(u)=(s_0[u],...,s_{r-1}[u]) to witnesses, yielding K'[u]=K[f(u)] and
        // preserving each witness's relation identity.
        if output_shape.is_empty() || selectors.len() != self.shape.len() {
            return Err(DslError::Schema);
        }
        let mut arguments = vec![self.value];
        arguments.extend(selectors.iter().map(|selector| selector.value.clone()));
        let node = NodeHandle::new(
            NodeKind::FamilyGather {
                output_shape: output_shape.clone(),
                input_rank: selectors.len(),
            },
            arguments,
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(self.element_schema.matrix_type.clone())),
                shape: output_shape.clone(),
            }],
        );
        let pending = Pending::merge(
            std::iter::once(self.pending)
                .chain(selectors.into_iter().map(|selector| selector.pending)),
        );
        Ok(Self {
            value: node.output(0).expect("gathered preimage family"),
            element_schema: self.element_schema,
            count: shape_count(&output_shape),
            shape: output_shape,
            pending,
        })
    }
}

impl GraphValueSchema for MatFamilyType {
    type Value = Family<Mat>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        Family::<Mat>::source_input(
            argument_name(next, "family"),
            self.element.clone(),
            shape_count(&self.shape),
            None,
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Family {
            element: Box::new(WireType::Matrix(self.element.clone())),
            shape: self.shape.clone(),
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
                $schema { shape: self.shape.clone() }
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
                    count: shape_count(&schema.shape),
                    shape: schema.shape.clone(),
                    pending,
                })
            }
        }

        impl GraphValueSchema for $schema {
            type Value = Family<$value>;
            fn placeholders_from(&self, next: &mut usize) -> Self::Value {
                let wire_type =
                    WireType::Family { element: Box::new($wire_type), shape: self.shape.clone() };
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
                    count: shape_count(&self.shape),
                    shape: self.shape.clone(),
                    pending: Pending::default(),
                }
            }
            fn wire_types(&self) -> Vec<WireType> {
                vec![WireType::Family { element: Box::new($wire_type), shape: self.shape.clone() }]
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
        Self::try_define(name, input_schema, |inputs| Ok(body(inputs)))
    }

    pub fn try_define(
        name: impl Into<String>,
        input_schema: I::Schema,
        body: impl FnOnce(I) -> Result<O, DslError>,
    ) -> Result<Self, DslError> {
        let name = name.into();
        let (inputs, output, scope) =
            with_new_construction_scope(|scope| -> Result<_, DslError> {
                let inputs = input_schema.placeholders();
                let output = body(inputs.clone())?;
                Ok((inputs, output, scope))
            })?;
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
        let flattened = input.flatten();
        let input_count = flattened.len();
        self.call_flattened(flattened, input.pending(), vec![None; input_count])
    }

    /// Calls this subgraph with authoritative canonical coefficient bounds for
    /// its flattened arguments.  `Some(U)` means a constant-polynomial
    /// argument has canonical coefficients in `0..U`; `None` supplies no
    /// such contract.  The vector includes every argument, including a
    /// synthetic constant-one argument when the caller supplies one.
    pub fn call_with_canonical_input_exclusive_uppers(
        &self,
        input: I,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<O, DslError> {
        let flattened = input.flatten();
        self.call_flattened(flattened, input.pending(), canonical_input_exclusive_uppers)
    }

    fn call_flattened(
        &self,
        flattened: Vec<ValueHandle>,
        input_pending: Pending,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<O, DslError> {
        if canonical_input_exclusive_uppers.len() != flattened.len() {
            return Err(DslError::CanonicalInputUpperCount);
        }
        if canonical_input_exclusive_uppers
            .iter()
            .any(|upper| upper.as_ref().is_some_and(|upper| upper == &BigUint::from(0u8)))
        {
            return Err(DslError::CanonicalInputUpperZero);
        }
        if canonical_input_exclusive_uppers.iter().zip(&flattened).any(|(upper, input)| {
            upper.is_some() && !matches!(input.wire_type(), WireType::Matrix(_))
        }) {
            return Err(DslError::CanonicalInputUpperNonMatrix);
        }
        let node = NodeHandle::subgraph_call(
            self.handle.clone(),
            flattened,
            Vec::new(),
            canonical_input_exclusive_uppers,
        );
        let values = (0..self.output_schema.wire_types().len())
            .map(|port| node.output(port as u32).expect("subgraph output"))
            .collect::<Vec<_>>();
        O::from_values(
            &self.output_schema,
            &values,
            Pending::merge([input_pending, self.pending.clone()]),
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
    fn preimage_parallel_outputs_and_artifacts_preserve_their_wire_type() {
        let ring = Ring::new(257, 8);
        let trapdoor = ring.sample_trapdoor(1, 5, 4, 4, 1_000_000);
        let scalar = trapdoor.sample_preimage(ring.zero((1, 1)), (6, 1));
        let family = Parallel::range(2)
            .map_values({
                let trapdoor = trapdoor.clone();
                let ring = ring.clone();
                move |_| trapdoor.sample_preimage(ring.zero((1, 1)), (6, 1))
            })
            .unwrap();
        let built = DslContext::new("typed-preimage-outputs")
            .public_preimage_output("scalar", scalar)
            .unwrap()
            .private_preimage_family_output("family", family)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let scalar = built.graph.outputs()["scalar"].value;
        let scalar_type = &built.graph.root_scope().node(scalar.node).unwrap().output_types()
            [scalar.port.0 as usize];
        assert!(matches!(scalar_type, WireType::Preimage(_)));
        let family = built.graph.outputs()["family"].value;
        let family_type = &built.graph.root_scope().node(family.node).unwrap().output_types()
            [family.port.0 as usize];
        assert!(matches!(
            family_type,
            WireType::Family { element, shape }
                if matches!(element.as_ref(), WireType::Preimage(_)) &&
                    shape == &vec![IntExpr::constant(2)]
        ));

        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([1; 32]),
            execution_nonce: [2; 32],
        };
        let scalar = ring.preimage_artifact_input(
            production_id.clone(),
            "scalar",
            (6, 1),
            ArtifactConfidentiality::Public,
        );
        let family = ring.preimage_family_artifact_input(
            production_id,
            "family",
            vec![IntExpr::constant(2)],
            (6, 1),
            ArtifactConfidentiality::Private,
        );
        let family_left = ring.input("family-left", (1, 6));
        let family = Family::<Preimage>::select(
            Int::constant(0).add(Int::constant(0)),
            vec![family.clone(), family],
        )
        .unwrap();
        let applied_family = family
            .parallel_map_values(move |_, preimage| family_left.clone().apply_preimage(preimage))
            .unwrap();
        DslContext::new("typed-preimage-inputs")
            .output("scalar", ring.input("left", (1, 6)).apply_preimage(scalar))
            .unwrap()
            .public_family_output("family", applied_family)
            .unwrap()
            .build()
            .unwrap();
    }

    #[test]
    fn dynamic_integer_hash_tag_is_an_explicit_argument() {
        let ring = Ring::new(17, 8);
        let row = Int::constant(7).add(Int::constant(0));
        let mut tag = HashTag::from(b"dynamic-hash/v1:".as_slice());
        tag.push(row);
        let sample = ring.hash_matrix(ring.bytes_input("key", 32), tag, (1, 1));
        let built =
            DslContext::new("dynamic-hash-tag").output("sample", sample).unwrap().build().unwrap();

        let hash = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::HashSample { .. }))
            .expect("hash sample");
        assert_eq!(hash.arguments().len(), 2);
        assert!(matches!(hash.arguments()[1].wire_type(), WireType::Int));
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
    fn decomposition_requires_explicit_positive_metadata_and_preserves_mode() {
        let ring = Ring::new(257, 8);
        let input = ring.input("input", (1, 1));
        let regular = DslContext::new("regular-decomposition")
            .preimage_output("value", input.clone().decompose(4, 4).into_preimage_relation())
            .unwrap()
            .build()
            .unwrap();
        regular.validate(&ParamEnv::default()).unwrap();
        let serialized = serde_json::to_string(&regular.graph).unwrap();
        assert!(serialized.contains("digit_count"));
        assert!(serialized.contains("\"small\":false"));

        let small = DslContext::new("small-decomposition")
            .preimage_output("value", input.clone().small_decompose(4, 4).into_preimage_relation())
            .unwrap()
            .build()
            .unwrap();
        small.validate(&ParamEnv::default()).unwrap();
        assert!(serde_json::to_string(&small.graph).unwrap().contains("\"small\":true"));

        let invalid = DslContext::new("negative-decomposition-base")
            .preimage_output("value", input.decompose(-4, 4).into_preimage_relation())
            .unwrap()
            .build()
            .unwrap();
        assert!(invalid.validate(&ParamEnv::default()).is_err());
    }

    #[test]
    fn scalar_families_gather_through_parallel_grid_nodes() {
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
                .any(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
        );
        assert!(
            built
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic { .. }))
        );
    }

    #[test]
    fn generated_index_family_gather_has_no_explicit_family_pack() {
        let context = DslContext::new("generated-index-family-gather");
        let values = context.int_family_input("values", 8);
        let indices =
            Parallel::range(2).map_values(|index| index.as_int().mul(Int::constant(3))).unwrap();
        let gathered = values.parallel_gather(indices).unwrap();
        let built = context.int_family_output("gathered", gathered).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let all_nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert!(all_nodes.iter().any(|node| matches!(node.kind(), NodeKind::ParallelGrid(_))));
        assert!(
            all_nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic { .. }))
        );
        assert!(!all_nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyPack { .. })));
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
                .any(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
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
    fn parameterized_trapdoor_families_use_parallel_grid_outputs() {
        let count = IntExpr::Var("count".to_owned());
        let ring = Ring::new(257, 8);
        let trapdoors = Parallel::range(count.clone())
            .map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))
            .unwrap();
        let targets = Parallel::range(count.clone()).map(|_| ring.zero((1, 1))).unwrap();
        let preimages = trapdoors
            .clone()
            .parallel_zip_mat_values(targets, |_, trapdoor, target| {
                trapdoor.sample_preimage(target, (trapdoor.public_matrix().matrix_type.columns, 1))
            })
            .unwrap();
        let built = DslContext::new("parameterized-trapdoor-families")
            .int_parameter("count")
            .public_family_output("public", trapdoors.public_matrices())
            .unwrap()
            .private_trapdoor_family_output("trapdoors", trapdoors)
            .unwrap()
            .private_preimage_family_output("preimages", preimages)
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
        assert!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
        );
    }

    #[test]
    fn parallel_zip_offset_accepts_exact_boundary_and_rejects_short_source_before_body() {
        let ring = Ring::new(257, 8);
        let matrix = || ring.zero((1, 1));
        let left = Family::pack(vec![matrix(), matrix()]).expect("left family");
        let exact =
            Family::pack(vec![matrix(), matrix(), matrix(), matrix()]).expect("offset family");
        let valid_body_called = Cell::new(false);
        let zipped = left
            .parallel_zip_offset(exact, 2, |_, left, right| {
                valid_body_called.set(true);
                left + right
            })
            .expect("exact offset boundary");
        assert!(valid_body_called.get());
        assert_eq!(zipped.count, IntExpr::constant(2));
        DslContext::new("parallel-zip-offset-boundary")
            .family_output("zipped", zipped)
            .expect("family output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");

        let left = Family::pack(vec![matrix(), matrix()]).expect("left family");
        let too_short =
            Family::pack(vec![matrix(), matrix(), matrix()]).expect("short offset family");
        let rejected_body_called = Cell::new(false);
        assert!(matches!(
            left.parallel_zip_offset_values(too_short, 2, |_, left, right| {
                rejected_body_called.set(true);
                left + right
            }),
            Err(DslError::FamilyCountMismatch)
        ));
        assert!(!rejected_body_called.get());
    }

    #[test]
    fn trapdoor_family_branch_preimages_validate() {
        let ring = Ring::new(257, 8);
        let trapdoors =
            Parallel::range(2).map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000)).unwrap();
        let targets = Parallel::grid(vec![IntExpr::constant(2), IntExpr::constant(3)])
            .map(|_| ring.zero((1, 1)))
            .unwrap();
        let preimages = trapdoors.sample_preimage_branches(targets, (6, 1)).unwrap();
        let selected = preimages.get_static(vec![IndexExpr::constant(0), IndexExpr::constant(0)]);
        let selected_other =
            preimages.get_static(vec![IndexExpr::constant(1), IndexExpr::constant(0)]);
        let applied = trapdoors
            .public_matrices()
            .get_static(vec![IndexExpr::constant(0)])
            .apply_preimage(selected.clone());
        let applied_other = trapdoors
            .public_matrices()
            .get_static(vec![IndexExpr::constant(1)])
            .apply_preimage(selected_other);
        let built = DslContext::new("trapdoor-family-branch-preimages")
            .private_preimage_output("preimage", selected)
            .unwrap()
            .public_output("applied", applied)
            .unwrap()
            .public_output("applied-other", applied_other)
            .unwrap()
            .build()
            .unwrap();

        built.validate(&ParamEnv::default()).unwrap();
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
            matches!(node.kind(), NodeKind::ParallelGrid(grid) if grid.shape == vec![IntExpr::constant(2)] && grid.input_modes.len() == 3)
        }));
    }

    #[test]
    fn trapdoor_parallel_helpers_reject_rank_n_coordinate_domains() {
        let ring = Ring::new(257, 8);
        let shape = vec![IntExpr::constant(2), IntExpr::constant(2)];
        let flatten = IndexMap::new([IndexExpr::Add(
            Box::new(IndexExpr::Multiply(
                Box::new(IndexExpr::Axis(0)),
                Box::new(IndexExpr::constant(2)),
            )),
            Box::new(IndexExpr::Axis(1)),
        )]);
        let trapdoors = || {
            Parallel::range(4)
                .map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))
                .expect("trapdoor family")
        };
        let matrices = || {
            Family::pack(vec![
                ring.zero((1, 1)),
                ring.zero((1, 1)),
                ring.zero((1, 1)),
                ring.zero((1, 1)),
            ])
            .expect("matrix family")
        };
        let rank_two_indices = || {
            let flat = Family::<Int>::pack(vec![
                Int::constant(0).add(Int::constant(0)),
                Int::constant(1).add(Int::constant(0)),
                Int::constant(2).add(Int::constant(0)),
                Int::constant(3).add(Int::constant(0)),
            ])
            .expect("flat indices");
            let value = NodeHandle::new(
                NodeKind::FamilyReindex { output_shape: shape.clone(), map: flatten.clone() },
                vec![flat.value],
                vec![WireType::Family { element: Box::new(WireType::Int), shape: shape.clone() }],
            )
            .output(0)
            .expect("rank-two indices");
            Family {
                value,
                element_schema: Int::constant(0).add(Int::constant(0)),
                count: IntExpr::constant(4),
                shape: shape.clone(),
                pending: Pending::default(),
            }
        };

        let rank_two_source =
            trapdoors().reindex(shape.clone(), flatten.clone()).expect("rank-two trapdoors");
        let rank_one_indices =
            Family::<Int>::pack(vec![Int::constant(0)]).expect("rank-one indices");
        assert!(matches!(
            rank_two_source.parallel_gather(rank_one_indices),
            Err(DslError::ParallelGatherRank)
        ));
        assert!(matches!(
            trapdoors().parallel_gather(rank_two_indices()),
            Err(DslError::ParallelGatherRank)
        ));

        let trapdoor_body_called = Cell::new(false);
        assert!(matches!(
            trapdoors()
                .reindex(shape.clone(), flatten.clone())
                .expect("rank-two trapdoors")
                .parallel_zip_mat_values(matrices(), |_, _trapdoor, matrix| {
                    trapdoor_body_called.set(true);
                    matrix
                }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!trapdoor_body_called.get());

        let matrix_body_called = Cell::new(false);
        let rank_two_matrices = matrices().reindex(shape, flatten).expect("rank-two matrices");
        assert!(matches!(
            trapdoors().parallel_zip_mat_values(rank_two_matrices, |_, _trapdoor, matrix| {
                matrix_body_called.set(true);
                matrix
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!matrix_body_called.get());

        let valid_body_called = Cell::new(false);
        let valid = trapdoors()
            .parallel_zip_mat_values(matrices(), |_, _trapdoor, matrix| {
                valid_body_called.set(true);
                matrix
            })
            .expect("rank-one trapdoor zip");
        assert!(valid_body_called.get());
        DslContext::new("rank-one-trapdoor-matrix-zip")
            .family_output("values", valid)
            .expect("family output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
    }

    #[test]
    fn trapdoor_and_matrix_zip_rejects_different_family_counts() {
        let ring = Ring::new(257, 8);
        let trapdoors =
            Parallel::range(2).map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000)).unwrap();
        let targets = Parallel::range(3).map(|_| ring.zero((1, 1))).unwrap();
        assert!(matches!(
            trapdoors.parallel_zip_mat_values(targets, |_, trapdoor, target| {
                trapdoor.sample_preimage(target, (6, 1))
            }),
            Err(DslError::FamilyCountMismatch)
        ));
    }

    #[test]
    fn matrix_parallel_zip_rejects_rank_n_inputs_before_body() {
        let ring = Ring::new(257, 8);
        let matrix = || ring.zero((1, 1));
        let flatten = IndexMap::new([IndexExpr::Add(
            Box::new(IndexExpr::Multiply(
                Box::new(IndexExpr::Axis(0)),
                Box::new(IndexExpr::constant(2)),
            )),
            Box::new(IndexExpr::Axis(1)),
        )]);
        let rank_two = || {
            Family::pack(vec![matrix(), matrix(), matrix(), matrix()])
                .expect("flat family")
                .reindex(vec![IntExpr::constant(2), IntExpr::constant(2)], flatten.clone())
                .expect("rank-two family")
        };
        let rank_one =
            || Family::pack(vec![matrix(), matrix(), matrix(), matrix()]).expect("rank-one family");

        let valid_body_called = Cell::new(false);
        rank_one()
            .parallel_zip_values(rank_one(), |_, left, right| {
                valid_body_called.set(true);
                left + right
            })
            .expect("rank-one zip");
        assert!(valid_body_called.get());

        let left_body_called = Cell::new(false);
        assert!(matches!(
            rank_two().parallel_zip_values(rank_one(), |_, left, right| {
                left_body_called.set(true);
                left + right
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!left_body_called.get());

        let right_body_called = Cell::new(false);
        assert!(matches!(
            rank_one().parallel_zip_values(rank_two(), |_, left, right| {
                right_body_called.set(true);
                left + right
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!right_body_called.get());

        let offset_body_called = Cell::new(false);
        assert!(matches!(
            rank_one().parallel_zip_offset_values(rank_two(), 0, |_, left, right| {
                offset_body_called.set(true);
                left + right
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!offset_body_called.get());
    }

    #[test]
    fn generic_parallel_zip_rejects_every_rank_n_tuple_position_before_body() {
        let ring = Ring::new(257, 8);
        let matrix = || ring.zero((1, 1));
        let flatten = IndexMap::new([IndexExpr::Add(
            Box::new(IndexExpr::Multiply(
                Box::new(IndexExpr::Axis(0)),
                Box::new(IndexExpr::constant(2)),
            )),
            Box::new(IndexExpr::Axis(1)),
        )]);
        let rank_two = || {
            Family::pack(vec![matrix(), matrix(), matrix(), matrix()])
                .expect("flat family")
                .reindex(vec![IntExpr::constant(2), IntExpr::constant(2)], flatten.clone())
                .expect("rank-two family")
        };
        let rank_one =
            || Family::pack(vec![matrix(), matrix(), matrix(), matrix()]).expect("rank-one family");

        let valid_body_calls = Cell::new(0);
        let valid = parallel_zip((rank_one(), rank_one()), |_, (left, right)| {
            valid_body_calls.set(valid_body_calls.get() + 1);
            left + right
        })
        .expect("rank-one generic zip");
        assert_eq!(valid_body_calls.get(), 1);
        DslContext::new("generic-parallel-zip-rank-one")
            .family_output("output", valid)
            .expect("family output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");

        let rejected_body_calls = Cell::new(0);
        assert!(matches!(
            parallel_zip((rank_two(), rank_one()), |_, (left, right)| {
                rejected_body_calls.set(rejected_body_calls.get() + 1);
                left + right
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(matches!(
            parallel_zip((rank_one(), rank_two()), |_, (left, right)| {
                rejected_body_calls.set(rejected_body_calls.get() + 1);
                left + right
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(matches!(
            parallel_zip_bundle(
                (rank_two(), rank_one(), rank_one()),
                |_, (left, middle, right)| {
                    rejected_body_calls.set(rejected_body_calls.get() + 1);
                    left + middle + right
                },
            ),
            Err(DslError::ParallelZipRank)
        ));
        assert!(matches!(
            parallel_zip_bundle(
                (rank_one(), rank_two(), rank_one()),
                |_, (left, middle, right)| {
                    rejected_body_calls.set(rejected_body_calls.get() + 1);
                    left + middle + right
                },
            ),
            Err(DslError::ParallelZipRank)
        ));
        assert!(matches!(
            parallel_zip_bundle_result(
                (rank_one(), rank_one(), rank_two()),
                |_, (left, middle, right)| {
                    rejected_body_calls.set(rejected_body_calls.get() + 1);
                    Ok::<_, DslError>(left + middle + right)
                },
            ),
            Err(DslError::ParallelZipRank)
        ));
        assert_eq!(rejected_body_calls.get(), 0);
    }

    #[test]
    fn matrix_rank_one_parallel_helpers_reject_rank_n_and_preserve_rank_one_shape() {
        let ring = Ring::new(257, 8);
        let rank_one = || Parallel::range(4).map(|_| ring.zero((1, 1))).unwrap();
        let rank_two = || {
            Parallel::grid(vec![IntExpr::constant(2), IntExpr::constant(2)])
                .map(|_| ring.zero((1, 1)))
                .unwrap()
        };
        let rank_two_indices = || {
            let flat = Family::<Int>::pack(vec![
                Int::constant(0).add(Int::constant(0)),
                Int::constant(1).add(Int::constant(0)),
                Int::constant(2).add(Int::constant(0)),
                Int::constant(3).add(Int::constant(0)),
            ])
            .unwrap();
            let shape = vec![IntExpr::constant(2), IntExpr::constant(2)];
            let value = NodeHandle::new(
                NodeKind::FamilyReindex {
                    output_shape: shape.clone(),
                    map: IndexMap::new([IndexExpr::Add(
                        Box::new(IndexExpr::Multiply(
                            Box::new(IndexExpr::Axis(0)),
                            Box::new(IndexExpr::constant(2)),
                        )),
                        Box::new(IndexExpr::Axis(1)),
                    )]),
                },
                vec![flat.value],
                vec![WireType::Family { element: Box::new(WireType::Int), shape: shape.clone() }],
            )
            .output(0)
            .unwrap();
            Family {
                value,
                element_schema: flat.element_schema,
                count: flat.count,
                shape,
                pending: flat.pending,
            }
        };

        let gathered = rank_one()
            .parallel_gather(Family::<Int>::pack(vec![Int::constant(3), Int::constant(0)]).unwrap())
            .expect("rank-one gather");
        assert_eq!(gathered.shape(), &[IntExpr::constant(2)]);
        assert!(matches!(
            rank_two().parallel_gather(
                Family::<Int>::pack(vec![Int::constant(0), Int::constant(1)]).unwrap()
            ),
            Err(DslError::ParallelGatherRank)
        ));
        assert!(matches!(
            rank_one().parallel_gather(rank_two_indices()),
            Err(DslError::ParallelGatherRank)
        ));

        let zipped =
            Family::<Mat>::parallel_zip_many_values(vec![rank_one(), rank_one()], |_, matrices| {
                matrices.into_iter().next().unwrap()
            })
            .expect("rank-one zip-many");
        assert_eq!(zipped.shape(), &[IntExpr::constant(4)]);
        let zip_many_body_called = Cell::new(false);
        assert!(matches!(
            Family::<Mat>::parallel_zip_many_values(vec![rank_one(), rank_two()], |_, matrices| {
                zip_many_body_called.set(true);
                matrices.into_iter().next().unwrap()
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!zip_many_body_called.get());

        let broadcast_body_called = Cell::new(false);
        assert!(matches!(
            Family::<Mat>::parallel_zip_many_with_broadcast_values(
                vec![rank_one()],
                vec![rank_two()],
                |_, matrices, _| {
                    broadcast_body_called.set(true);
                    Ok(matrices.into_iter().next().unwrap())
                },
            ),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!broadcast_body_called.get());

        let decoded_ints = rank_one().parallel_threshold_decode_ints(2, 1).unwrap();
        let decoded_bools = rank_one().parallel_threshold_decode_bools(2, 1).unwrap();
        assert_eq!(decoded_ints[0].shape, vec![IntExpr::constant(4)]);
        assert_eq!(decoded_bools[0].shape, vec![IntExpr::constant(4)]);
        assert!(matches!(
            rank_two().parallel_threshold_decode_ints(2, 1),
            Err(DslError::ParallelFamilyRank)
        ));
        assert!(matches!(
            rank_two().parallel_threshold_decode_bools(2, 1),
            Err(DslError::ParallelFamilyRank)
        ));

        let zipped_three = rank_one()
            .parallel_zip3_values(rank_one(), rank_one(), |_, first, _, _| first)
            .expect("rank-one zip3");
        assert_eq!(zipped_three.shape(), &[IntExpr::constant(4)]);
        let zip3_body_called = Cell::new(false);
        assert!(matches!(
            rank_one().parallel_zip3_values(rank_one(), rank_two(), |_, first, _, _| {
                zip3_body_called.set(true);
                first
            }),
            Err(DslError::ParallelZipRank)
        ));
        assert!(!zip3_body_called.get());
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
                .filter(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
                .count(),
            1
        );
    }

    #[test]
    fn parallel_zip_many_with_broadcast_keeps_formal_family_inputs() {
        let ring = Ring::new(17, 8);
        let zipped =
            Family::pack(vec![ring.input("zipped-0", (1, 1)), ring.input("zipped-1", (1, 1))])
                .unwrap();
        let broadcast = Family::pack(vec![
            ring.input("broadcast-0", (1, 1)),
            ring.input("broadcast-1", (1, 1)),
            ring.input("broadcast-2", (1, 1)),
        ])
        .unwrap();
        let output = Family::<Mat>::parallel_zip_many_with_broadcast_values(
            vec![zipped],
            vec![broadcast],
            |index, zipped, broadcast| {
                Ok(zipped.into_iter().next().unwrap() + broadcast[0].get(index.as_int()))
            },
        )
        .unwrap();
        let context = DslContext::new("parallel-zip-many-with-broadcast");
        let built = context.public_family_output("output", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let grid_spec = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::ParallelGrid(spec) => Some(spec),
                _ => None,
            })
            .expect("parallel grid");
        assert_eq!(grid_spec.input_modes.len(), 2);
        let encoded = serde_json::to_vec(&built.graph).unwrap();
        let decoded: Graph = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, built.graph);
        mxx_ir_core::validate(&decoded, &ParamEnv::default()).unwrap();
    }

    #[test]
    fn parallel_zip_many_with_broadcast_rejects_zipped_count_mismatch() {
        let ring = Ring::new(17, 8);
        let left =
            Family::pack(vec![ring.input("left-0", (1, 1)), ring.input("left-1", (1, 1))]).unwrap();
        let right = Family::pack(vec![
            ring.input("right-0", (1, 1)),
            ring.input("right-1", (1, 1)),
            ring.input("right-2", (1, 1)),
        ])
        .unwrap();
        assert!(matches!(
            Family::<Mat>::parallel_zip_many_with_broadcast_values(
                vec![left, right],
                Vec::new(),
                |_, zipped, _| Ok(zipped.into_iter().next().unwrap()),
            ),
            Err(DslError::FamilyCountMismatch)
        ));
    }

    #[test]
    fn try_define_accepts_a_formal_nonartifact_family() {
        let ring = Ring::new(17, 8);
        let matrix_type = MatType(ring.matrix_type((1, 1)));
        let family_type =
            MatFamilyType { element: ring.matrix_type((1, 1)), shape: vec![2.into()] };
        let subgraph = Subgraph::<(Mat, Family<Mat>), Mat>::try_define(
            "formal-matrix-family",
            (matrix_type.clone(), family_type.clone()),
            |(matrix, family)| Ok(matrix + family.get_static(0)),
        )
        .unwrap();
        let context = DslContext::new("formal-matrix-family-call");
        let input_family =
            Family::pack(vec![ring.input("family-0", (1, 1)), ring.input("family-1", (1, 1))])
                .unwrap();
        let output = subgraph.call((ring.input("matrix", (1, 1)), input_family)).unwrap();
        let built = context.output("output", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn parallel_map_values_rejects_rank_n_matrix_preimage_and_trapdoor_families() {
        let ring = Ring::new(17, 8);
        let matrix_type = ring.matrix_type((1, 1));
        let shape = vec![IntExpr::constant(2), IntExpr::constant(2)];
        let flatten = IndexMap::new([IndexExpr::Add(
            Box::new(IndexExpr::Multiply(
                Box::new(IndexExpr::Axis(0)),
                Box::new(IndexExpr::constant(2)),
            )),
            Box::new(IndexExpr::Axis(1)),
        )]);
        let matrices = Family::<Mat>::source_input(
            "rank-two-matrices".into(),
            matrix_type.clone(),
            IntExpr::constant(4),
            None,
        )
        .reindex(shape.clone(), flatten.clone())
        .unwrap();
        assert!(matches!(
            matrices.parallel_map_values(|_, matrix| matrix),
            Err(DslError::ParallelMapRank)
        ));

        let preimages = Family::<Preimage>::source_input(
            "rank-two-preimages".into(),
            matrix_type,
            shape.clone(),
            None,
        );
        assert!(matches!(
            preimages.parallel_map_values(|_, preimage| preimage),
            Err(DslError::ParallelMapRank)
        ));

        let trapdoors = Parallel::range(4)
            .map_values(|_| ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))
            .unwrap()
            .reindex(shape, flatten)
            .unwrap();
        assert_eq!(trapdoors.public_matrices().shape(), &[2.into(), 2.into()]);
        assert!(matches!(
            trapdoors.parallel_map_values(|_, trapdoor| trapdoor.public_matrix()),
            Err(DslError::ParallelMapRank)
        ));
    }

    #[test]
    fn scalar_parallel_helpers_reject_rank_n_and_keep_rank_one_inputs() {
        let rank_two_shape = vec![IntExpr::constant(2), IntExpr::constant(2)];
        let rank_one_shape = vec![IntExpr::constant(4)];
        let rank_two_int = IntFamilyType { shape: rank_two_shape.clone() }.placeholders();
        let rank_two_bool = BoolFamilyType { shape: rank_two_shape.clone() }.placeholders();
        assert!(matches!(
            rank_two_int.clone().parallel_map(|_, _| panic!("rank guard must precede callback")),
            Err(DslError::ParallelMapRank)
        ));
        assert!(matches!(
            rank_two_bool.parallel_map(|_, _| panic!("rank guard must precede callback")),
            Err(DslError::ParallelMapRank)
        ));

        let rank_one_int = IntFamilyType { shape: rank_one_shape.clone() }.placeholders();
        let rank_two_indices = IntFamilyType { shape: rank_two_shape.clone() }.placeholders();
        assert!(matches!(
            rank_two_int.clone().parallel_gather(rank_one_int.clone()),
            Err(DslError::ParallelGatherRank)
        ));
        assert!(matches!(
            rank_one_int.clone().parallel_gather(rank_two_indices),
            Err(DslError::ParallelGatherRank)
        ));
        assert!(matches!(
            rank_two_int.clone().parallel_pack_little_endian_bits(2, 2),
            Err(DslError::ParallelFamilyRank)
        ));

        let ring = Ring::new(17, 8);
        let rank_one_mats =
            MatFamilyType { element: ring.matrix_type((1, 1)), shape: rank_one_shape.clone() }
                .placeholders();
        let rank_two_mats = Family::<Mat>::source_input(
            "rank-two-select-candidates".into(),
            ring.matrix_type((1, 1)),
            IntExpr::constant(4),
            None,
        )
        .reindex(
            rank_two_shape,
            IndexMap::new([IndexExpr::Add(
                Box::new(IndexExpr::Multiply(
                    Box::new(IndexExpr::Axis(0)),
                    Box::new(IndexExpr::constant(2)),
                )),
                Box::new(IndexExpr::Axis(1)),
            )]),
        )
        .unwrap();
        assert!(matches!(
            rank_two_int.parallel_select_mats(vec![rank_one_mats.clone()]),
            Err(DslError::ParallelFamilyRank)
        ));
        assert!(matches!(
            rank_one_int.clone().parallel_select_mats(vec![rank_two_mats]),
            Err(DslError::ParallelFamilyRank)
        ));

        let rank_one_bool = BoolFamilyType { shape: rank_one_shape.clone() }.placeholders();
        rank_one_bool.parallel_map(|_, value| value).unwrap();
        rank_one_int.clone().parallel_map(|_, value| value).unwrap();
        rank_one_int.clone().parallel_gather(rank_one_int.clone()).unwrap();
        rank_one_int.clone().parallel_pack_little_endian_bits(2, 2).unwrap();
        rank_one_int.parallel_select_mats(vec![rank_one_mats]).unwrap();
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
                            NodeKind::ParallelGrid(spec) => Some(spec),
                            _ => None,
                        })
                    })
                    .flatten()
            })
            .expect("nested parallel loop");
        assert_eq!(nested.index_slots, vec![1]);

        let encoded = serde_json::to_vec(&built.graph).unwrap();
        let decoded: Graph = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(built.graph, decoded);
        mxx_ir_core::validate(&decoded, &ParamEnv::default()).unwrap();
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

    #[test]
    fn matrix_family_select_preserves_the_family_wire_type() {
        let ring = Ring::new(17, 8);
        let context = DslContext::new("select-matrix-family");
        let selector = context.int_family_input("selector", 1).get_static(0);
        let one = ring.polynomial([IntExpr::constant(1)]);
        let left = Family::pack(vec![ring.zero((1, 1)), one.clone()]).unwrap();
        let right = Family::pack(vec![one, ring.zero((1, 1))]).unwrap();
        let selected = Family::<Mat>::select(selector, vec![left, right]).unwrap();
        let built = context.public_family_output("selected", selected).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        // Both branches contain four matrices, but [4] and [2,2] are
        // different coordinate domains. Their equal flat cardinality must
        // not authorize constructing a Select with one branch's wire shape.
        let four = IntExpr::Mul(Box::new(IntExpr::constant(2)), Box::new(IntExpr::constant(2)));
        let flat = Family::<Mat>::source_input(
            "flat-select-branch".into(),
            ring.matrix_type((1, 1)),
            four.clone(),
            None,
        );
        let rank_two = Family::<Mat>::source_input(
            "rank-two-select-source".into(),
            ring.matrix_type((1, 1)),
            four,
            None,
        )
        .reindex(
            vec![IntExpr::constant(2), IntExpr::constant(2)],
            IndexMap::new([IndexExpr::Add(
                Box::new(IndexExpr::Multiply(
                    Box::new(IndexExpr::Axis(0)),
                    Box::new(IndexExpr::constant(2)),
                )),
                Box::new(IndexExpr::Axis(1)),
            )]),
        )
        .unwrap();
        assert_eq!(flat.count(), rank_two.count());
        assert!(matches!(
            Family::<Mat>::select(Int::constant(0), vec![flat, rank_two]),
            Err(DslError::FamilyCountMismatch)
        ));
    }

    #[test]
    fn subgraph_call_carries_canonical_input_exclusive_uppers() {
        let ring = Ring::new(17, 8);
        let matrix = MatType(ring.matrix_type((1, 1)));
        let subgraph = Subgraph::<Mat, Mat>::define("bounded-matrix", matrix, |value| value)
            .expect("subgraph definition");
        let context = DslContext::new("bounded-subgraph");
        let output = subgraph
            .call_with_canonical_input_exclusive_uppers(
                ring.input("input", (1, 1)),
                vec![Some(BigUint::from(4u8))],
            )
            .expect("bounded subgraph call");
        let built = context.output("output", output).expect("output").build().expect("graph");
        let call = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::SubgraphCall(call) => Some(call),
                _ => None,
            })
            .expect("subgraph call node");
        assert_eq!(call.canonical_input_exclusive_uppers, vec![Some(BigUint::from(4u8))]);
        let encoded = serde_json::to_vec(&built.graph).expect("serialize graph");
        let decoded: Graph = serde_json::from_slice(&encoded).expect("deserialize graph");
        assert_eq!(built.graph, decoded);
        mxx_ir_core::validate(&decoded, &ParamEnv::default()).expect("valid graph");
    }

    #[test]
    fn subgraph_call_rejects_invalid_canonical_input_exclusive_uppers() {
        let subgraph = Subgraph::<Int, Int>::define("bounded-int-errors", IntType, |value| value)
            .expect("subgraph definition");
        assert!(matches!(
            subgraph.call_with_canonical_input_exclusive_uppers(Int::constant(0), Vec::new()),
            Err(DslError::CanonicalInputUpperCount)
        ));
        assert!(matches!(
            subgraph.call_with_canonical_input_exclusive_uppers(
                Int::constant(0),
                vec![Some(BigUint::from(0u8))]
            ),
            Err(DslError::CanonicalInputUpperZero)
        ));
        assert!(matches!(
            subgraph.call_with_canonical_input_exclusive_uppers(
                Int::constant(0),
                vec![Some(BigUint::from(1u8))]
            ),
            Err(DslError::CanonicalInputUpperNonMatrix)
        ));
    }
}
