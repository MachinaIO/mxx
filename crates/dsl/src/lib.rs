//! Declarative typed construction API for mxx graphs.
//!
//! Executable operations create immutable `mxx-ir-core` nodes immediately.
//! The only non-executable expression wrapper is [`SymbolicExpr`], used by
//! [`Mat::assume`] to attach an axiom-like symbolic annotation.

use mxx_ir_core::{
    CapturePolicy, CompileParameter, CompileParameterKind, FreezeError, FreezeMap, Graph,
    GraphOutput, IntExpr, NodeHandle, ParamEnv, RealExpr, SealMap, SealedSubgraph, SubgraphHandle,
    ValueHandle,
    artifact::{ArtifactConfidentiality, ProductionId},
    graph::with_new_construction_scope,
    node::{
        ArtifactInput, ConstantMatrix, HashVariant, IndexRange, LoopInputMode, MatrixBinaryOp,
        NodeKind, ParallelLoop, SampleRange,
    },
    types::{MatrixType, WireType},
};
use mxx_ir_symbolic::overlay::{
    DeclaredDependencyLabels, PendingSymbolicExpr, PendingSymbolicExprNode, StableVirtualAtomId,
    SymbolicOverlay, SymbolicValueRef, VirtualAtomDecl, VirtualKind,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::{Add, Mul, Neg, Sub},
    sync::Arc,
};
use thiserror::Error;

pub use mxx_ir_core::{Rational, artifact::ArtifactConfidentiality as Confidentiality};
pub use mxx_ir_symbolic::atom::AssumedMetadata;

#[derive(Debug, Error)]
pub enum DslError {
    #[error(transparent)]
    Freeze(#[from] FreezeError),
    #[error("duplicate output name: {0}")]
    DuplicateOutput(String),
    #[error("assume target already has an assumption")]
    DuplicateAssumption,
    #[error("assume expression type does not match its target")]
    AssumptionType,
    #[error("assume references an executable value that is unreachable or in another scope")]
    UnreachableSymbolicFactor,
    #[error("subgraph body captures an executable value")]
    SubgraphCapture,
    #[error("graph value schema does not match its flattened values")]
    Schema,
    #[error("parallel families have different counts")]
    FamilyCountMismatch,
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
pub struct PreimageType(pub MatrixType);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TrapdoorType {
    pub matrix: MatrixType,
    pub sigma: RealExpr,
    pub gadget_base: IntExpr,
    pub digit_count: IntExpr,
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
    ) -> Trapdoor {
        let rows = rows.into();
        let sigma = sigma.into();
        let gadget_base = gadget_base.into();
        let digit_count = digit_count.into();
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
        let wire_type =
            WireType::Trapdoor { matrix: matrix_type.clone(), sigma, gadget_base, digit_count };
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
            pending: Pending::default(),
        }
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
    pub fn gaussian(&self, shape: impl IntoShape, sigma: impl Into<RealExpr>) -> Mat {
        let ty = self.matrix_type(shape);
        Mat::from_node(
            NodeKind::GaussianSample { matrix_type: ty.clone(), sigma: sigma.into() },
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
    ) -> Trapdoor {
        let rows = rows.into();
        let sigma = sigma.into();
        let gadget_base = gadget_base.into();
        let digit_count = digit_count.into();
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
            },
            Vec::new(),
            vec![
                WireType::Matrix(matrix_type.clone()),
                WireType::Trapdoor { matrix: matrix_type.clone(), sigma, gadget_base, digit_count },
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

    pub fn assume(self, expression: impl IntoSymbolicExpr) -> Result<Self, DslError> {
        let expression = expression.into_symbolic_expr();
        if !expression.well_typed || self.matrix_type != expression.matrix_type {
            return Err(DslError::AssumptionType);
        }
        let pending = Pending::merge([self.pending.clone(), expression.pending.clone()]);
        let annotation = PendingAssumption { target: self.value.clone(), expression };
        Ok(Self { pending: Pending::annotation(pending, annotation), ..self })
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

macro_rules! mat_symbolic_rhs {
    ($rhs:ty) => {
        impl Add<$rhs> for Mat {
            type Output = SymbolicExpr;
            fn add(self, rhs: $rhs) -> Self::Output {
                self.into_symbolic_expr() + rhs
            }
        }
        impl Sub<$rhs> for Mat {
            type Output = SymbolicExpr;
            fn sub(self, rhs: $rhs) -> Self::Output {
                self.into_symbolic_expr() - rhs
            }
        }
        impl Mul<$rhs> for Mat {
            type Output = SymbolicExpr;
            fn mul(self, rhs: $rhs) -> Self::Output {
                self.into_symbolic_expr() * rhs
            }
        }
    };
}

mat_symbolic_rhs!(VirtualMat);
mat_symbolic_rhs!(SymbolicExpr);

#[derive(Clone)]
pub struct Preimage {
    value: ValueHandle,
    matrix_type: MatrixType,
    pending: Pending,
}

impl Preimage {
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
    pending: Pending,
}

impl Trapdoor {
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
            NodeKind::PreimageSample { matrix_type: ty.clone() },
            vec![self.public.value.clone(), self.value.clone(), target.value],
            vec![WireType::Preimage(ty.clone())],
        );
        Preimage { value: node.output(0).expect("preimage"), matrix_type: ty, pending }
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

    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn pending_assumptions(&self) -> bool {
        self.pending.0.is_some()
    }

    pub fn add(self, rhs: Self) -> Self {
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Add),
            vec![self.value, rhs.value],
            vec![WireType::Int],
        );
        Self { value: node.output(0).expect("integer sum"), pending }
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
}

impl Bool {
    pub fn to_int(self) -> Int {
        let node = NodeHandle::new(NodeKind::BoolToInt, vec![self.value], vec![WireType::Int]);
        Int { value: node.output(0).expect("boolean integer"), pending: self.pending }
    }
}

#[derive(Clone)]
pub struct Bytes {
    value: ValueHandle,
    pending: Pending,
}

#[derive(Clone)]
pub struct Family<T> {
    value: ValueHandle,
    element_schema: T,
    count: IntExpr,
    pending: Pending,
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

    pub fn pack(values: Vec<Mat>) -> Result<Self, DslError> {
        let Some(first) = values.first() else {
            return Err(DslError::Schema);
        };
        if values.iter().any(|value| value.matrix_type != first.matrix_type) {
            return Err(DslError::Schema);
        }
        let count = IntExpr::constant(values.len());
        let element_type = first.matrix_type.clone();
        let pending = Pending::merge(values.iter().map(|value| value.pending.clone()));
        let arguments = values.into_iter().map(|value| value.value).collect();
        let family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(element_type.clone())),
            count: count.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::FamilyPack { count: count.clone() },
            arguments,
            vec![family_type],
        );
        Ok(Self {
            value: node.output(0).expect("packed family"),
            element_schema: Mat::source_input("__family_element".to_owned(), element_type, None),
            count,
            pending,
        })
    }

    pub fn parallel_map(self, body: impl FnOnce(LoopIndex, Mat) -> Mat) -> Result<Self, DslError> {
        self.parallel_map_values(body)
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
        let (body_value, explicit_inputs, scope) = with_new_construction_scope(|scope| {
            let inputs = element_types
                .into_iter()
                .enumerate()
                .map(|(index, matrix_type)| {
                    Mat::source_input(format!("item-{index}"), matrix_type, None)
                })
                .collect::<Vec<_>>();
            let explicit_inputs = inputs.iter().map(|input| input.value.clone()).collect();
            let output = body(LoopIndex { expression: IntExpr::LoopIndex(0) }, inputs);
            (output, explicit_inputs, scope)
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
                index_slot: 0,
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
        let outer_family = self.value.clone();
        let count = self.count.clone();
        let element_type = self.element_schema.matrix_type.clone();
        let (body_value, explicit_input, scope) = with_new_construction_scope(|scope| {
            let input = Mat::source_input("item".to_owned(), element_type, None);
            let output = body(LoopIndex { expression: IntExpr::LoopIndex(0) }, input.clone());
            (output, input.value, scope)
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
            sealed.handle,
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: modes,
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
        let (outputs, input, scope) = with_new_construction_scope(|scope| {
            let input = Mat::source_input("item".to_owned(), element_type, None);
            let outputs = input.clone().threshold_decode_bools(modulus, length);
            (outputs, input.value, scope)
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
        let (outputs, input, scope) = with_new_construction_scope(|scope| {
            let input = Mat::source_input("item".to_owned(), element_type, None);
            let outputs = input.clone().threshold_decode_ints(plaintext_modulus, length);
            (outputs, input.value, scope)
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
        let (body_value, explicit_inputs, scope) = with_new_construction_scope(|scope| {
            let left = Mat::source_input("left".to_owned(), left_type, None);
            let right = Mat::source_input("right".to_owned(), right_type, None);
            let output =
                body(LoopIndex { expression: IntExpr::LoopIndex(0) }, left.clone(), right.clone());
            (output, vec![left.value, right.value], scope)
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
                index_slot: 0,
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
        let (body_value, explicit_inputs, scope) = with_new_construction_scope(|scope| {
            let left = Mat::source_input("left".to_owned(), left_type, None);
            let right = Mat::source_input("right".to_owned(), right_type, None);
            let output =
                body(LoopIndex { expression: IntExpr::LoopIndex(0) }, left.clone(), right.clone());
            (output, vec![left.value, right.value], scope)
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
                index_slot: 0,
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
        let (body_value, explicit_inputs, scope) = with_new_construction_scope(|scope| {
            let first = Mat::source_input("first".to_owned(), first_type, None);
            let second = Mat::source_input("second".to_owned(), second_type, None);
            let third = Mat::source_input("third".to_owned(), third_type, None);
            let output = body(
                LoopIndex { expression: IntExpr::LoopIndex(0) },
                first.clone(),
                second.clone(),
                third.clone(),
            );
            (output, vec![first.value, second.value, third.value], scope)
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
                index_slot: 0,
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
            index_slot: 0,
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

impl Parallel {
    pub fn range(count: impl Into<IntExpr>) -> ParallelRange {
        ParallelRange { count: count.into() }
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
        let count = self.count;
        let (body_value, scope) = with_new_construction_scope(|scope| {
            (body(LoopIndex { expression: IntExpr::LoopIndex(0) }), scope)
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
            sealed.handle,
            arguments,
            family_outputs,
            ParallelLoop {
                count: count.clone(),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: modes,
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
    type Items;
}

impl ParallelZipTuple for (Family<Mat>, Family<Mat>) {
    type Items = (Mat, Mat);

    fn parallel_zip_tuple<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> R,
    ) -> Result<R::Families, DslError> {
        self.0.parallel_zip_values(self.1, |index, left, right| body(index, (left, right)))
    }
}

impl ParallelZipTuple for (Family<Mat>, Family<Mat>, Family<Mat>) {
    type Items = (Mat, Mat, Mat);

    fn parallel_zip_tuple<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, Self::Items) -> R,
    ) -> Result<R::Families, DslError> {
        self.0.parallel_zip3_values(self.1, self.2, |index, first, second, third| {
            body(index, (first, second, third))
        })
    }
}

pub fn parallel_zip<T: ParallelZipTuple, R: ParallelOutput>(
    families: T,
    body: impl FnOnce(LoopIndex, T::Items) -> R,
) -> Result<R::Families, DslError> {
    families.parallel_zip_tuple(body)
}

impl LoopIndex {
    pub fn expression(&self) -> IntExpr {
        self.expression.clone()
    }
}

#[derive(Clone)]
pub struct VirtualMat(Arc<VirtualMatInner>);

#[derive(Clone)]
struct VirtualMatInner {
    diagnostic_name: String,
    matrix_type: MatrixType,
    kind: VirtualKind,
}

impl VirtualMat {
    pub fn large(name: impl Into<String>, matrix_type: MatrixType) -> Self {
        Self(Arc::new(VirtualMatInner {
            diagnostic_name: name.into(),
            matrix_type,
            kind: VirtualKind::Large,
        }))
    }

    pub fn bounded(
        name: impl Into<String>,
        matrix_type: MatrixType,
        metadata: BoundedMetadata,
    ) -> Self {
        Self(Arc::new(VirtualMatInner {
            diagnostic_name: name.into(),
            matrix_type,
            kind: VirtualKind::Bounded {
                norm: metadata.norm,
                is_const_poly: metadata.is_const_poly,
                zero_rows: metadata.zero_rows,
                dependencies: metadata.dependencies,
                clt_ready: metadata.clt_ready,
            },
        }))
    }

    fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

#[derive(Clone, Debug)]
pub struct BoundedMetadata {
    pub norm: RealExpr,
    pub is_const_poly: bool,
    pub zero_rows: Option<IntExpr>,
    pub dependencies: DeclaredDependencyLabels,
    pub clt_ready: bool,
}

impl BoundedMetadata {
    pub fn conservative(norm: impl Into<RealExpr>) -> Self {
        Self {
            norm: norm.into(),
            is_const_poly: false,
            zero_rows: None,
            dependencies: DeclaredDependencyLabels::Unknown,
            clt_ready: false,
        }
    }
}

#[derive(Clone)]
pub struct SymbolicExpr {
    matrix_type: MatrixType,
    node: PendingSymbolicNode,
    pending: Pending,
    well_typed: bool,
}

#[derive(Clone)]
enum PendingSymbolicNode {
    Value(PendingSymbolicValue),
    Add(Vec<SymbolicExpr>),
    Scale { coefficient: IntExpr, value: Box<SymbolicExpr> },
    Mul(Vec<SymbolicExpr>),
}

#[derive(Clone)]
enum PendingSymbolicValue {
    Concrete(ValueHandle),
    Virtual(VirtualMat),
}

pub trait IntoSymbolicExpr {
    fn into_symbolic_expr(self) -> SymbolicExpr;
}

impl IntoSymbolicExpr for SymbolicExpr {
    fn into_symbolic_expr(self) -> SymbolicExpr {
        self
    }
}

impl IntoSymbolicExpr for VirtualMat {
    fn into_symbolic_expr(self) -> SymbolicExpr {
        SymbolicExpr {
            matrix_type: self.0.matrix_type.clone(),
            node: PendingSymbolicNode::Value(PendingSymbolicValue::Virtual(self)),
            pending: Pending::default(),
            well_typed: true,
        }
    }
}

impl IntoSymbolicExpr for Mat {
    fn into_symbolic_expr(self) -> SymbolicExpr {
        SymbolicExpr {
            matrix_type: self.matrix_type,
            node: PendingSymbolicNode::Value(PendingSymbolicValue::Concrete(self.value)),
            pending: self.pending,
            well_typed: true,
        }
    }
}

impl<T: IntoSymbolicExpr> Add<T> for SymbolicExpr {
    type Output = SymbolicExpr;
    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into_symbolic_expr();
        let well_typed = self.well_typed && rhs.well_typed && self.matrix_type == rhs.matrix_type;
        SymbolicExpr {
            matrix_type: self.matrix_type.clone(),
            node: PendingSymbolicNode::Add(vec![self.clone(), rhs.clone()]),
            pending: Pending::merge([self.pending, rhs.pending]),
            well_typed,
        }
    }
}

impl<T: IntoSymbolicExpr> Sub<T> for SymbolicExpr {
    type Output = SymbolicExpr;
    fn sub(self, rhs: T) -> Self::Output {
        self + (-rhs.into_symbolic_expr())
    }
}

impl<T: IntoSymbolicExpr> Mul<T> for SymbolicExpr {
    type Output = SymbolicExpr;
    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into_symbolic_expr();
        let matrix_type = product_type(&self.matrix_type, &rhs.matrix_type);
        let well_typed = self.well_typed &&
            rhs.well_typed &&
            self.matrix_type.modulus == rhs.matrix_type.modulus &&
            self.matrix_type.ring_dimension == rhs.matrix_type.ring_dimension &&
            (self.matrix_type.columns == rhs.matrix_type.rows ||
                is_scalar_type(&self.matrix_type) ||
                is_scalar_type(&rhs.matrix_type));
        SymbolicExpr {
            matrix_type,
            node: PendingSymbolicNode::Mul(vec![self.clone(), rhs.clone()]),
            pending: Pending::merge([self.pending, rhs.pending]),
            well_typed,
        }
    }
}

impl Neg for SymbolicExpr {
    type Output = SymbolicExpr;

    fn neg(self) -> Self::Output {
        SymbolicExpr {
            matrix_type: self.matrix_type.clone(),
            node: PendingSymbolicNode::Scale {
                coefficient: IntExpr::constant(-1),
                value: Box::new(self.clone()),
            },
            pending: self.pending,
            well_typed: self.well_typed,
        }
    }
}

impl Neg for VirtualMat {
    type Output = SymbolicExpr;

    fn neg(self) -> Self::Output {
        -self.into_symbolic_expr()
    }
}

macro_rules! symbolic_binary_lhs {
    ($lhs:ty) => {
        impl<T: IntoSymbolicExpr> Add<T> for $lhs {
            type Output = SymbolicExpr;
            fn add(self, rhs: T) -> Self::Output {
                self.into_symbolic_expr() + rhs
            }
        }
        impl<T: IntoSymbolicExpr> Sub<T> for $lhs {
            type Output = SymbolicExpr;
            fn sub(self, rhs: T) -> Self::Output {
                self.into_symbolic_expr() - rhs
            }
        }
        impl<T: IntoSymbolicExpr> Mul<T> for $lhs {
            type Output = SymbolicExpr;
            fn mul(self, rhs: T) -> Self::Output {
                self.into_symbolic_expr() * rhs
            }
        }
    };
}

symbolic_binary_lhs!(VirtualMat);

#[derive(Clone, Default)]
#[doc(hidden)]
pub struct Pending(Option<Arc<PendingNode>>);

struct PendingNode {
    parents: Vec<Pending>,
    assumption: Option<PendingAssumption>,
}

#[derive(Clone)]
struct PendingAssumption {
    target: ValueHandle,
    expression: SymbolicExpr,
}

impl Pending {
    #[doc(hidden)]
    pub fn merge(values: impl IntoIterator<Item = Pending>) -> Self {
        let parents = values.into_iter().filter(|value| value.0.is_some()).collect::<Vec<_>>();
        match parents.len() {
            0 => Self::default(),
            1 => parents.into_iter().next().expect("one parent"),
            _ => Self(Some(Arc::new(PendingNode { parents, assumption: None }))),
        }
    }

    fn annotation(parent: Pending, assumption: PendingAssumption) -> Self {
        Self(Some(Arc::new(PendingNode { parents: vec![parent], assumption: Some(assumption) })))
    }

    fn remap(&self, map: &SealMap) -> Self {
        let Some(node) = &self.0 else { return Self::default() };
        let parents = node.parents.iter().map(|parent| parent.remap(map)).collect();
        let assumption = node.assumption.as_ref().map(|assumption| PendingAssumption {
            target: map
                .resolve(&assumption.target)
                .cloned()
                .unwrap_or_else(|| assumption.target.clone()),
            expression: assumption.expression.remap(map),
        });
        Self(Some(Arc::new(PendingNode { parents, assumption })))
    }
}

impl SymbolicExpr {
    fn remap(&self, map: &SealMap) -> Self {
        let mut result = self.clone();
        result.node = match &self.node {
            PendingSymbolicNode::Value(PendingSymbolicValue::Concrete(value)) => {
                PendingSymbolicNode::Value(PendingSymbolicValue::Concrete(
                    map.resolve(value).cloned().unwrap_or_else(|| value.clone()),
                ))
            }
            PendingSymbolicNode::Value(PendingSymbolicValue::Virtual(value)) => {
                PendingSymbolicNode::Value(PendingSymbolicValue::Virtual(value.clone()))
            }
            PendingSymbolicNode::Add(children) => {
                PendingSymbolicNode::Add(children.iter().map(|child| child.remap(map)).collect())
            }
            PendingSymbolicNode::Scale { coefficient, value } => PendingSymbolicNode::Scale {
                coefficient: coefficient.clone(),
                value: Box::new(value.remap(map)),
            },
            PendingSymbolicNode::Mul(children) => {
                PendingSymbolicNode::Mul(children.iter().map(|child| child.remap(map)).collect())
            }
        };
        result.pending = result.pending.remap(map);
        result
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
    pending: Pending,
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

    pub fn public_output(mut self, name: impl Into<String>, mat: Mat) -> Result<Self, DslError> {
        self.insert_output(name.into(), mat, Some(ArtifactConfidentiality::Public))?;
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

    fn insert_output(
        &mut self,
        name: String,
        mat: Mat,
        confidentiality: Option<ArtifactConfidentiality>,
    ) -> Result<(), DslError> {
        if self
            .outputs
            .insert(
                name.clone(),
                PendingOutput { value: mat.value, pending: mat.pending, confidentiality },
            )
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
        pending: Pending,
        confidentiality: Option<ArtifactConfidentiality>,
    ) -> Result<(), DslError> {
        if self
            .outputs
            .insert(name.clone(), PendingOutput { value, pending, confidentiality })
            .is_some()
        {
            return Err(DslError::DuplicateOutput(name));
        }
        Ok(())
    }

    pub fn build(self) -> Result<BuiltGraph, DslError> {
        let pending = Pending::merge(self.outputs.values().map(|output| output.pending.clone()));
        let retained_roots = pending_assumptions(&pending)
            .iter()
            .flat_map(|assumption| concrete_symbolic_values(&assumption.expression))
            .collect();
        let outputs = self
            .outputs
            .into_iter()
            .map(|(name, output)| {
                (name, GraphOutput { value: output.value, confidentiality: output.confidentiality })
            })
            .collect();
        let (graph, map) = Graph::freeze(
            self.name,
            self.parameters,
            outputs,
            retained_roots,
            Vec::new(),
            self.real_constants,
        )?;
        let symbolic_overlay = freeze_pending(pending, &map)?;
        mxx_ir_core::validate_structure(&graph)?;
        Ok(BuiltGraph { graph, symbolic_overlay })
    }
}

pub struct BuiltGraph {
    pub graph: Graph,
    pub symbolic_overlay: SymbolicOverlay,
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

    pub fn elaborate(
        &self,
        bindings: &ParamEnv,
    ) -> Result<mxx_ir_symbolic::ElaboratedGraph, mxx_ir_symbolic::ElaborationError> {
        mxx_ir_symbolic::elaborate_with_overlay(&self.graph, bindings, &[], &self.symbolic_overlay)
    }

    pub fn elaborate_with_manifests(
        &self,
        bindings: &ParamEnv,
        artifact_manifests: &BTreeMap<ProductionId, mxx_ir_core::artifact::Manifest>,
        symbolic_manifests: &[mxx_ir_symbolic::manifest::Manifest],
    ) -> Result<mxx_ir_symbolic::ElaboratedGraph, mxx_ir_symbolic::ElaborationError> {
        let validated =
            mxx_ir_core::validate_with_manifests(&self.graph, bindings, artifact_manifests)?;
        mxx_ir_symbolic::elaborate_validated(&validated, symbolic_manifests, &self.symbolic_overlay)
    }
}

#[derive(Debug, Error)]
pub enum ValidationBuildError {
    #[error(transparent)]
    Core(#[from] mxx_ir_core::ValidationError),
}

fn freeze_pending(pending: Pending, map: &FreezeMap) -> Result<SymbolicOverlay, DslError> {
    let mut assumptions = pending_assumptions(&pending);
    assumptions.sort_by_key(|assumption| map.resolve(&assumption.target).cloned());

    let mut overlay = SymbolicOverlay::default();
    let mut virtual_ids = HashMap::<usize, StableVirtualAtomId>::new();
    for assumption in assumptions {
        let target =
            map.resolve(&assumption.target).cloned().ok_or(DslError::UnreachableSymbolicFactor)?;
        if overlay.assumptions.contains_key(&target) {
            return Err(DslError::DuplicateAssumption);
        }
        let expression = freeze_symbolic_expression(
            assumption.expression,
            &assumption.target,
            map,
            &mut overlay.virtual_atoms,
            &mut virtual_ids,
        )?;
        overlay.assumptions.insert(target, expression);
    }
    Ok(overlay)
}

fn pending_assumptions(pending: &Pending) -> Vec<PendingAssumption> {
    let mut assumptions = Vec::new();
    let mut seen = HashSet::new();
    fn visit(
        pending: &Pending,
        seen: &mut HashSet<usize>,
        assumptions: &mut Vec<PendingAssumption>,
    ) {
        let Some(node) = &pending.0 else { return };
        let identity = Arc::as_ptr(node) as usize;
        if !seen.insert(identity) {
            return;
        }
        for parent in &node.parents {
            visit(parent, seen, assumptions);
        }
        if let Some(assumption) = &node.assumption {
            assumptions.push(assumption.clone());
        }
    }
    visit(&pending, &mut seen, &mut assumptions);
    assumptions
}

fn concrete_symbolic_values(expression: &SymbolicExpr) -> Vec<ValueHandle> {
    fn visit(expression: &SymbolicExpr, values: &mut Vec<ValueHandle>) {
        match &expression.node {
            PendingSymbolicNode::Value(PendingSymbolicValue::Concrete(value)) => {
                values.push(value.clone());
            }
            PendingSymbolicNode::Value(PendingSymbolicValue::Virtual(_)) => {}
            PendingSymbolicNode::Add(children) | PendingSymbolicNode::Mul(children) => {
                for child in children {
                    visit(child, values);
                }
            }
            PendingSymbolicNode::Scale { value, .. } => visit(value, values),
        }
    }
    let mut values = Vec::new();
    visit(expression, &mut values);
    values
}

fn freeze_symbolic_expression(
    expression: SymbolicExpr,
    target: &ValueHandle,
    map: &FreezeMap,
    virtual_atoms: &mut BTreeMap<StableVirtualAtomId, VirtualAtomDecl>,
    virtual_ids: &mut HashMap<usize, StableVirtualAtomId>,
) -> Result<PendingSymbolicExpr, DslError> {
    let node = match expression.node {
        PendingSymbolicNode::Value(PendingSymbolicValue::Concrete(value)) => {
            if value.construction_scope() != target.construction_scope() {
                return Err(DslError::UnreachableSymbolicFactor);
            }
            PendingSymbolicExprNode::Value(SymbolicValueRef::Local(
                map.resolve(&value).cloned().ok_or(DslError::UnreachableSymbolicFactor)?,
            ))
        }
        PendingSymbolicNode::Value(PendingSymbolicValue::Virtual(value)) => {
            let identity = value.identity();
            let next = StableVirtualAtomId(virtual_ids.len() as u64);
            let id = *virtual_ids.entry(identity).or_insert_with(|| {
                virtual_atoms.insert(
                    next,
                    VirtualAtomDecl {
                        diagnostic_name: value.0.diagnostic_name.clone(),
                        matrix_type: value.0.matrix_type.clone(),
                        kind: value.0.kind.clone(),
                    },
                );
                next
            });
            PendingSymbolicExprNode::Value(SymbolicValueRef::Virtual(id))
        }
        PendingSymbolicNode::Add(children) => PendingSymbolicExprNode::Add(
            children
                .into_iter()
                .map(|child| {
                    freeze_symbolic_expression(child, target, map, virtual_atoms, virtual_ids)
                })
                .collect::<Result<_, _>>()?,
        ),
        PendingSymbolicNode::Scale { coefficient, value } => PendingSymbolicExprNode::Scale {
            coefficient,
            value: Box::new(freeze_symbolic_expression(
                *value,
                target,
                map,
                virtual_atoms,
                virtual_ids,
            )?),
        },
        PendingSymbolicNode::Mul(children) => PendingSymbolicExprNode::Mul(
            children
                .into_iter()
                .map(|child| {
                    freeze_symbolic_expression(child, target, map, virtual_atoms, virtual_ids)
                })
                .collect::<Result<_, _>>()?,
        ),
    };
    Ok(PendingSymbolicExpr { matrix_type: expression.matrix_type, node })
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
        let WireType::Trapdoor { sigma, gadget_base, digit_count, .. } = self.value.wire_type()
        else {
            unreachable!("Trapdoor always wraps a trapdoor wire")
        };
        TrapdoorType {
            matrix: self.matrix_type.clone(),
            sigma: sigma.clone(),
            gadget_base: gadget_base.clone(),
            digit_count: digit_count.clone(),
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
            },
        ]
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

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    #[test]
    fn executable_arithmetic_is_an_immutable_shared_dag() {
        let ring = Ring::new(17, 8);
        let input = ring.input("input", (2, 2));
        let sum = input.clone() + input;
        let built = DslContext::new("sum").public_output("sum", sum).unwrap().build().unwrap();
        assert_eq!(built.graph.root_scope().nodes().len(), 2);
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn build_rejects_undeclared_compile_variables() {
        let ring = Ring::new(IntExpr::Var("missing".to_owned()), 8);
        let result = DslContext::new("undeclared")
            .output("value", ring.input("value", (1, 1)))
            .expect("output")
            .build();
        assert!(matches!(
            result,
            Err(DslError::StructuralValidation(
                mxx_ir_core::ValidationError::UndeclaredCompileVariable(name)
            )) if name == "missing"
        ));
    }

    #[test]
    fn build_rejects_loop_indices_in_structural_types() {
        let ring = Ring::new(17, 8);
        let family = Parallel::range(2)
            .map(|index| {
                ring.gaussian(
                    (IntExpr::Add(Box::new(index.expression()), Box::new(IntExpr::constant(1))), 1),
                    2,
                )
            })
            .expect("parallel body construction");
        let result = DslContext::new("structural-loop-index")
            .family_output("family", family)
            .expect("output")
            .build();
        assert!(matches!(
            result,
            Err(DslError::StructuralValidation(
                mxx_ir_core::ValidationError::StructuralLoopIndex { slot: 0, .. }
            ))
        ));
    }

    #[test]
    fn assume_uses_arithmetic_and_preserves_virtual_handle_identity() {
        let ring = Ring::new(17, 8);
        let a = ring.input("a", (2, 2));
        let c = ring.input("c", (2, 2));
        let s = VirtualMat::bounded(
            "s",
            matrix_type(2, 2),
            BoundedMetadata::conservative(RealExpr::from_integer(2)),
        );
        let e = VirtualMat::bounded(
            "e",
            matrix_type(2, 2),
            BoundedMetadata::conservative(RealExpr::from_integer(3)),
        );
        let c = c.assume(s * a.clone() + e).unwrap();
        let built = DslContext::new("assume").output("c", c).unwrap().build().unwrap();
        assert_eq!(built.graph.root_scope().nodes().len(), 2);
        assert_eq!(built.symbolic_overlay.virtual_atoms.len(), 2);
        assert_eq!(built.symbolic_overlay.assumptions.len(), 1);
        let assumption = built.symbolic_overlay.assumptions.values().next().expect("assumption");
        assert!(matches!(
            &assumption.node,
            PendingSymbolicExprNode::Add(children)
                if children.len() == 2 &&
                    children.iter().any(|child| matches!(child.node, PendingSymbolicExprNode::Mul(_)))
        ));
    }

    #[test]
    fn virtual_clone_reuses_identity_but_same_named_construction_does_not() {
        let ring = Ring::new(17, 8);
        let metadata = BoundedMetadata::conservative(1);
        let first = VirtualMat::bounded("v", matrix_type(1, 1), metadata.clone());
        let distinct = VirtualMat::bounded("v", matrix_type(1, 1), metadata);
        let target = ring.input("target", (1, 1));
        let assumed = target.assume(first.clone() + first + distinct).expect("typed assumption");
        let built = DslContext::new("virtual-identities")
            .output("output", assumed)
            .expect("output")
            .build()
            .expect("build");
        assert_eq!(built.symbolic_overlay.virtual_atoms.len(), 2);
    }

    #[test]
    fn assume_rejects_type_mismatches_and_duplicate_targets() {
        let ring = Ring::new(17, 8);
        let target = ring.input("target", (1, 1));
        let wrong_shape = VirtualMat::large("wrong", matrix_type(1, 2));
        assert!(matches!(target.clone().assume(wrong_shape), Err(DslError::AssumptionType)));

        let first = target
            .clone()
            .assume(VirtualMat::large("first", matrix_type(1, 1)))
            .expect("first assumption");
        let second = target
            .assume(VirtualMat::large("second", matrix_type(1, 1)))
            .expect("second assumption handle");
        let result = DslContext::new("duplicate-assumption")
            .output("first", first)
            .expect("first output")
            .output("second", second)
            .expect("second output")
            .build();
        assert!(matches!(result, Err(DslError::DuplicateAssumption)));
    }

    #[test]
    fn executable_zero_roots_and_products_elaborate_to_typed_zero() {
        let ring = Ring::new(17, 8);
        let root = ring.zero((2, 3));
        let product = ring.zero((2, 2)) * ring.gaussian((2, 3), 2);
        let built = DslContext::new("typed-zero-elaboration")
            .output("root", root)
            .unwrap()
            .output("product", product)
            .unwrap()
            .build()
            .unwrap();
        let elaborated = built.elaborate(&ParamEnv::default()).unwrap();
        for name in ["root", "product"] {
            let expression = elaborated
                .wire(&elaborated.outputs[name])
                .and_then(|wire| wire.expression)
                .expect("matrix expression");
            assert!(matches!(
                elaborated.expressions.get(expression).unwrap().node,
                mxx_ir_symbolic::SymbolicExprNode::Zero
            ));
        }
    }

    #[test]
    fn subgraph_rejects_implicit_runtime_capture() {
        let ring = Ring::new(17, 8);
        let key = ring.input("key", (2, 2));
        let result =
            Subgraph::<Mat, Mat>::define("capturing", MatType(matrix_type(2, 2)), |input| {
                input * key
            });
        assert!(matches!(result, Err(DslError::SubgraphCapture)));
    }

    #[test]
    fn subgraph_assumption_rejects_symbolic_only_outer_capture() {
        let ring = Ring::new(17, 8);
        let outer = ring.input("outer", (1, 1));
        let virtual_value =
            VirtualMat::bounded("virtual", matrix_type(1, 1), BoundedMetadata::conservative(1));
        let assumed = Subgraph::<Mat, Mat>::define(
            "symbolic-capture",
            MatType(matrix_type(1, 1)),
            move |input: Mat| {
                input.clone().assume(virtual_value * outer).expect("typed assumption")
            },
        )
        .expect("symbolic-only capture is detected after stable scope assignment");
        let output = assumed.call(ring.input("input", (1, 1))).expect("subgraph call");
        let result =
            DslContext::new("symbolic-capture").output("output", output).expect("output").build();
        assert!(matches!(result, Err(DslError::UnreachableSymbolicFactor)));
    }

    #[test]
    fn repeated_matrix_schema_uses_distinct_argument_names() {
        let ty = MatType(matrix_type(1, 1));
        let reversed = Subgraph::define("reverse", (ty.clone(), ty), |(left, right)| (right, left))
            .expect("two matrix arguments");
        let ring = Ring::new(17, 8);
        let (right, left) = reversed
            .call((ring.input("left", (1, 1)), ring.input("right", (1, 1))))
            .expect("subgraph call");
        DslContext::new("two-matrix-subgraph")
            .output("left", left)
            .expect("left output")
            .output("right", right)
            .expect("right output")
            .build()
            .expect("distinct placeholder names");
    }

    #[test]
    fn range_zip_and_offset_build_structural_parallel_loops() {
        let ring = Ring::new(17, 8);
        let key = ring.bytes_input("hash-key", 32);
        let generated = Parallel::range(3)
            .map(|index| ring.hash_matrix(key.clone(), tag!["public-key", index], (1, 1)))
            .expect("range loop");
        let left = ring.input_family("left", 2, (1, 1));
        let right = ring.input_family("right", 3, (1, 1));
        let offset =
            left.parallel_zip_offset(right, 1, |_, left, right| left + right).expect("offset loop");
        let built = DslContext::new("parallel-forms")
            .public_family_output("generated", generated)
            .expect("generated output")
            .public_family_output("offset", offset)
            .expect("offset output")
            .build()
            .expect("build");
        built.validate(&ParamEnv::default()).expect("valid loops");
        built.elaborate(&ParamEnv::default()).expect("symbolically elaborated loops");
        assert_eq!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
                .count(),
            2
        );
    }

    #[test]
    fn parallel_range_can_return_multiple_homogeneous_families() {
        let ring = Ring::new(17, 8);
        let (left, right) = Parallel::range(3)
            .map_values(|_| (ring.gaussian((1, 1), 2), ring.gaussian((2, 1), 3)))
            .expect("two family outputs");
        let built = DslContext::new("parallel-multi-output")
            .family_output("left", left)
            .expect("left")
            .family_output("right", right)
            .expect("right")
            .build()
            .expect("build");
        built.validate(&ParamEnv::default()).expect("validation");
        let loop_node = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .expect("parallel loop");
        assert_eq!(loop_node.output_types().len(), 2);
    }

    #[test]
    fn parallel_zip_many_keeps_each_matrix_as_a_separate_zipped_family() {
        let ring = Ring::new(17, 8);
        let families = (0..4)
            .map(|_| {
                Family::pack(vec![ring.identity(1), ring.zero((1, 1))]).expect("matrix family")
            })
            .collect::<Vec<_>>();
        let outputs = Family::parallel_zip_many_values(families, |_, inputs| {
            inputs.into_iter().reduce(|left, right| left + right).expect("non-empty zipped inputs")
        })
        .expect("parallel zip many");
        let built = DslContext::new("parallel-zip-many")
            .output("first", outputs.get_static(0))
            .expect("first")
            .output("second", outputs.get_static(1))
            .expect("second")
            .build()
            .expect("build");
        built.validate(&ParamEnv::default()).expect("validation");
        let loop_spec = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::ParallelLoop(spec) => Some(spec),
                _ => None,
            })
            .expect("parallel loop");
        assert_eq!(loop_spec.input_modes, vec![LoopInputMode::Zip; 4]);
    }

    #[test]
    fn static_parallel_members_have_distinct_symbolic_iteration_identities() {
        let ring = Ring::new(17, 8);
        let family = Parallel::range(2).map(|_| ring.gaussian((1, 1), 2)).expect("parallel family");
        let built = DslContext::new("parallel-identities")
            .output("first", family.get_static(0))
            .expect("first")
            .output("second", family.get_static(1))
            .expect("second")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        let atom = |name: &str| {
            let expression = elaborated
                .wire(&elaborated.outputs[name])
                .and_then(|wire| wire.expression)
                .expect("one Gaussian expression");
            let mxx_ir_symbolic::SymbolicExprNode::Atom(atom) =
                &elaborated.expressions.get(expression).expect("expression").node
            else {
                panic!("one Gaussian atom")
            };
            atom.clone()
        };
        let first = atom("first");
        let second = atom("second");
        assert_ne!(first, second);
        assert!(matches!(
            first,
            mxx_ir_symbolic::atom::AtomId::Instantiated {
                instantiation_path,
                ..
            } if matches!(
                instantiation_path.last(),
                Some(mxx_ir_symbolic::atom::SymbolicInstantiationFrame::ParallelIteration {
                    index: mxx_ir_symbolic::atom::ParallelIndex::Static(0),
                    ..
                })
            )
        ));
        assert!(matches!(
            second,
            mxx_ir_symbolic::atom::AtomId::Instantiated {
                instantiation_path,
                ..
            } if matches!(
                instantiation_path.last(),
                Some(mxx_ir_symbolic::atom::SymbolicInstantiationFrame::ParallelIteration {
                    index: mxx_ir_symbolic::atom::ParallelIndex::Static(1),
                    ..
                })
            )
        ));
    }

    #[test]
    fn zipped_exact_family_specializes_to_the_matching_static_member() {
        let ring = Ring::new(257, 8);
        let input = Family::pack(vec![ring.gaussian((1, 1), 2), ring.gaussian((1, 1), 3)]).unwrap();
        let mapped = input.parallel_map(|_, value| value).expect("parallel identity");
        let built = DslContext::new("zipped-family-specialization")
            .output("first", mapped.get_static(0))
            .unwrap()
            .output("second", mapped.get_static(1))
            .unwrap()
            .build()
            .unwrap();
        let elaborated = built.elaborate(&ParamEnv::default()).unwrap();
        let report = mxx_noise_simulator::simulate(&elaborated).unwrap();
        assert_eq!(
            report.outputs["first"].noise.as_ref().unwrap().bound,
            bigdecimal::BigDecimal::from(13u8),
        );
        assert_eq!(
            report.outputs["second"].noise.as_ref().unwrap().bound,
            bigdecimal::BigDecimal::from(39u8) / bigdecimal::BigDecimal::from(2u8),
        );
    }

    #[test]
    fn zipped_structural_family_offset_specializes_to_shifted_source_members() {
        let ring = Ring::new(257, 8);
        let left = Parallel::range(2).map(|_| ring.zero((1, 1))).expect("left family");
        let right = Parallel::range(3).map(|_| ring.gaussian((1, 1), 2)).expect("right family");
        let shifted =
            left.parallel_zip_offset(right, 1, |_, _, right| right).expect("offset family");
        let dynamic_index = ring.input("index", (1, 1)).extract_coefficient(0);
        let built = DslContext::new("structural-offset-specialization")
            .output("first", shifted.get_static(0))
            .unwrap()
            .output("second", shifted.get_static(1))
            .unwrap()
            .output("dynamic", shifted.get(dynamic_index))
            .unwrap()
            .build()
            .unwrap();
        let elaborated = built.elaborate(&ParamEnv::default()).unwrap();
        let iteration = |name: &str| {
            let expression = elaborated
                .wire(&elaborated.outputs[name])
                .and_then(|wire| wire.expression)
                .expect("one Gaussian expression");
            let mxx_ir_symbolic::SymbolicExprNode::Atom(atom) =
                &elaborated.expressions.get(expression).expect("expression").node
            else {
                panic!("one Gaussian atom")
            };
            let mxx_ir_symbolic::atom::AtomId::Instantiated { instantiation_path, .. } = atom
            else {
                panic!("instantiated Gaussian atom")
            };
            let Some(mxx_ir_symbolic::atom::SymbolicInstantiationFrame::ParallelIteration {
                index,
                index_offset,
                ..
            }) = instantiation_path.last()
            else {
                panic!("parallel iteration frame")
            };
            (index.clone(), *index_offset)
        };
        assert_eq!(iteration("first"), (mxx_ir_symbolic::atom::ParallelIndex::Static(1), 0),);
        assert_eq!(iteration("second"), (mxx_ir_symbolic::atom::ParallelIndex::Static(2), 0),);
        assert!(matches!(
            iteration("dynamic"),
            (mxx_ir_symbolic::atom::ParallelIndex::Dynamic(_), 1)
        ));
    }

    #[test]
    fn structural_parallel_template_arena_size_is_independent_of_iteration_count() {
        let arena_size = |count| {
            let ring = Ring::new(257, 8);
            let family =
                Parallel::range(count).map(|_| ring.gaussian((1, 1), 2)).expect("parallel family");
            let built = DslContext::new("compact-parallel")
                .family_output("family", family)
                .unwrap()
                .build()
                .unwrap();
            built.elaborate(&ParamEnv::default()).unwrap().expressions.len()
        };
        assert_eq!(arena_size(2), arena_size(200));
    }

    #[test]
    fn template_dynamic_and_subgraph_calls_have_distinct_instantiation_identities() {
        let ring = Ring::new(257, 8);
        let family = Parallel::range(2).map(|_| ring.gaussian((1, 1), 2)).unwrap();
        let dynamic_index = ring.input("index", (1, 1)).extract_coefficient(0);
        let dynamic = family.get(dynamic_index);
        let sampled_ring = ring.clone();
        let sampled =
            Subgraph::<Mat, Mat>::define("sampled", MatType(ring.matrix_type((1, 1))), move |_| {
                sampled_ring.gaussian((1, 1), 2)
            })
            .unwrap();
        let first_call = sampled.call(ring.zero((1, 1))).unwrap();
        let second_call = sampled.call(ring.zero((1, 1))).unwrap();
        let built = DslContext::new("instantiation-identities")
            .family_output("family", family)
            .unwrap()
            .output("dynamic", dynamic)
            .unwrap()
            .output("first-call", first_call)
            .unwrap()
            .output("second-call", second_call)
            .unwrap()
            .build()
            .unwrap();
        let elaborated = built.elaborate(&ParamEnv::default()).unwrap();
        let atom = |expression| {
            let mxx_ir_symbolic::SymbolicExprNode::Atom(atom) =
                &elaborated.expressions.get(expression).unwrap().node
            else {
                panic!("single sampled atom")
            };
            atom.clone()
        };
        let family_wire = elaborated.wire(&elaborated.outputs["family"]).unwrap();
        let mxx_ir_symbolic::elaborate::SymbolicFamily::StructuralTemplate { template, .. } =
            family_wire.family.as_ref().expect("family state")
        else {
            panic!("structural template")
        };
        let template_atom = atom(*template);
        let dynamic_atom = atom(
            elaborated
                .wire(&elaborated.outputs["dynamic"])
                .and_then(|wire| wire.expression)
                .unwrap(),
        );
        let call_atom = |name: &str| {
            atom(
                elaborated
                    .wire(&elaborated.outputs[name])
                    .and_then(|wire| wire.expression)
                    .unwrap(),
            )
        };
        let first_call = call_atom("first-call");
        let second_call = call_atom("second-call");
        assert!(matches!(
            template_atom,
            mxx_ir_symbolic::atom::AtomId::Instantiated { ref instantiation_path, .. }
                if matches!(
                    instantiation_path.last(),
                    Some(mxx_ir_symbolic::atom::SymbolicInstantiationFrame::ParallelIteration {
                        index: mxx_ir_symbolic::atom::ParallelIndex::Template,
                        ..
                    })
                )
        ));
        assert!(matches!(
            dynamic_atom,
            mxx_ir_symbolic::atom::AtomId::Instantiated { ref instantiation_path, .. }
                if matches!(
                    instantiation_path.last(),
                    Some(mxx_ir_symbolic::atom::SymbolicInstantiationFrame::ParallelIteration {
                        index: mxx_ir_symbolic::atom::ParallelIndex::Dynamic(_),
                        ..
                    })
                )
        ));
        assert_ne!(first_call, second_call);
        assert!(matches!(
            first_call,
            mxx_ir_symbolic::atom::AtomId::Instantiated { ref instantiation_path, .. }
                if matches!(
                    instantiation_path.last(),
                    Some(mxx_ir_symbolic::atom::SymbolicInstantiationFrame::Call(_))
                )
        ));
    }

    #[test]
    fn select_elaboration_preserves_the_selection_domain() {
        let ring = Ring::new(17, 8);
        let index = ring.input("index", (1, 1)).extract_coefficient(0);
        let selected =
            index.select(vec![ring.gaussian((1, 1), 2), ring.gaussian((1, 1), 3)]).expect("select");
        let built = DslContext::new("select")
            .output("selected", selected)
            .expect("output")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        let expression = elaborated
            .wire(&elaborated.outputs["selected"])
            .and_then(|wire| wire.expression)
            .expect("selected expression");
        assert!(matches!(
            elaborated.expressions.get(expression).expect("expression").node,
            mxx_ir_symbolic::SymbolicExprNode::Select { .. }
        ));
    }

    #[test]
    fn concat_is_preserved_as_a_typed_symbolic_node() {
        let ring = Ring::new(17, 8);
        let left = ring.input("left", (1, 3));
        let right = ring.input("right", (2, 3)) + ring.gaussian((2, 3), 2);
        let rows = Mat::concat(ConcatAxis::Rows, vec![left, right]);
        let columns = Mat::concat(
            ConcatAxis::Columns,
            vec![ring.gaussian((2, 1), 2), ring.input("column-right", (2, 2))],
        );
        let built = DslContext::new("structural-concat")
            .output("rows", rows)
            .expect("output")
            .output("columns", columns)
            .expect("columns")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("typed concat");
        for (name, expected) in [("rows", ConcatAxis::Rows), ("columns", ConcatAxis::Columns)] {
            let expression = elaborated
                .wire(&elaborated.outputs[name])
                .and_then(|wire| wire.expression)
                .expect("output expression");
            assert!(matches!(
                elaborated.expressions.get(expression).expect("expression").node,
                mxx_ir_symbolic::SymbolicExprNode::Concat { axis, .. } if axis == expected
            ));
        }
    }

    #[test]
    fn diagonal_concat_accepts_mixed_symbolic_inputs() {
        let ring = Ring::new(17, 8);
        let mixed = ring.input("signal", (1, 1)) + ring.gaussian((1, 1), 2);
        let output = Mat::concat(ConcatAxis::Diagonal, vec![mixed, ring.gaussian((2, 2), 3)]);
        let built = DslContext::new("diagonal-concat")
            .output("output", output)
            .expect("output")
            .build()
            .expect("build");
        built.elaborate(&ParamEnv::default()).expect("diagonal concat is unrestricted");
    }

    #[test]
    fn all_builtin_graph_values_cross_a_subgraph_boundary() {
        let ring = Ring::new(257, 8);
        let trapdoor = ring.sample_trapdoor(1, 3, 4, 2);
        let preimage = trapdoor.sample_preimage(ring.input("target", (1, 1)), (4, 1));
        let bytes = ring.bytes_input("bytes", 32);
        let family = ring.input_family("family", 2, (1, 1));
        let schema = (
            BytesType { length: 32.into() },
            (
                MatFamilyType { element: ring.matrix_type((1, 1)), count: 2.into() },
                (trapdoor.schema(), preimage.schema()),
            ),
        );
        let identity = Subgraph::define("typed-identity", schema, |value| value)
            .expect("typed subgraph definition");
        let (_, (family, (trapdoor, preimage))) =
            identity.call((bytes, (family, (trapdoor, preimage)))).expect("typed subgraph call");
        let built = DslContext::new("typed-values")
            .family_output("family", family)
            .expect("family output")
            .output("trapdoor-public", trapdoor.public_matrix())
            .expect("trapdoor output")
            .output("preimage", preimage.as_mat())
            .expect("preimage output")
            .build()
            .expect("build");
        built.validate(&ParamEnv::default()).expect("validation");
    }

    #[test]
    fn trapdoor_subgraph_placeholder_preserves_public_preimage_relation() {
        let ring = Ring::new(257, 8);
        let trapdoor = ring.sample_trapdoor(1, 3, 4, 2);
        let schema = (trapdoor.schema(), MatType(ring.matrix_type((1, 1))));
        let relation =
            Subgraph::define("trapdoor-relation", schema, |(trapdoor, target): (Trapdoor, Mat)| {
                let public = trapdoor.public_matrix();
                let preimage = trapdoor.sample_preimage(target, (4, 1));
                public * preimage.as_mat()
            })
            .expect("trapdoor relation subgraph");
        let target = ring.input("target", (1, 1));
        let product = relation.call((trapdoor, target)).expect("subgraph call");
        let built = DslContext::new("trapdoor-relation")
            .output("product", product)
            .expect("output")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        let product = elaborated
            .wire(&elaborated.outputs["product"])
            .and_then(|wire| wire.expression)
            .expect("product expression");
        assert!(matches!(
            elaborated.expressions.get(product).expect("expression").node,
            mxx_ir_symbolic::SymbolicExprNode::Atom(_)
        ));
    }

    #[test]
    fn symbolic_artifact_import_preserves_preimage_identity() {
        let ring = Ring::new(257, 8);
        let trapdoor = ring.sample_trapdoor(1, 3, 4, 2);
        let public = trapdoor.public_matrix();
        let target = ring.input("target", (1, 1));
        let preimage = trapdoor.sample_preimage(target, (4, 1));
        let producer = DslContext::new("producer")
            .public_output("public", public)
            .expect("public output")
            .public_output("preimage", preimage.as_mat())
            .expect("preimage output")
            .build()
            .expect("producer build");
        let bindings = ParamEnv::default();
        let validated = producer.validate(&bindings).expect("producer validation");
        let elaborated = producer.elaborate(&bindings).expect("producer elaboration");
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([7; 32]),
            execution_nonce: [9; 32],
        };
        let artifact_manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &validated)
                .expect("artifact manifest");
        let symbolic_exports = elaborated
            .outputs
            .iter()
            .map(|(name, reference)| {
                let wire = elaborated.wire(reference).expect("elaborated output");
                (
                    name.clone(),
                    mxx_ir_symbolic::manifest::ExportArtifact {
                        wire_type: wire.wire_type.clone(),
                        expression: wire.expression,
                        family: wire.family.clone(),
                        content_hash: None,
                        layout: None,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let symbolic_manifest = mxx_ir_symbolic::manifest::export_manifest(
            production.clone(),
            &symbolic_exports,
            &elaborated.atoms,
            &elaborated.expressions,
            &elaborated.preimage_relations,
            elaborated.assumption_digest,
        )
        .expect("symbolic manifest");

        let imported_public = ring.artifact_input(
            production.clone(),
            "public",
            (1, 4),
            ArtifactConfidentiality::Public,
        );
        let imported_preimage = ring.artifact_input(
            production.clone(),
            "preimage",
            (4, 1),
            ArtifactConfidentiality::Public,
        );
        let consumer = DslContext::new("consumer")
            .output("product", imported_public * imported_preimage)
            .expect("consumer output")
            .build()
            .expect("consumer build");
        let imported = consumer
            .elaborate_with_manifests(
                &bindings,
                &BTreeMap::from([(production, artifact_manifest)]),
                &[symbolic_manifest],
            )
            .expect("consumer elaboration");
        let product = imported
            .wire(&imported.outputs["product"])
            .and_then(|wire| wire.expression)
            .expect("product expression");
        let mxx_ir_symbolic::SymbolicExprNode::Atom(atom) =
            &imported.expressions.get(product).expect("expression").node
        else {
            panic!("B*K rewrites to the imported target")
        };
        assert!(matches!(atom, mxx_ir_symbolic::atom::AtomId::Imported { .. }));
    }

    #[test]
    fn parallel_loop_can_broadcast_and_select_an_artifact_family() {
        let ring = Ring::new(257, 8);
        let lookup = Family::pack(vec![ring.gaussian((1, 1), 2), ring.gaussian((1, 1), 3)])
            .expect("lookup family");
        let producer = DslContext::new("artifact-family-producer")
            .public_family_output("lookup", lookup)
            .expect("lookup output")
            .build()
            .expect("producer build");
        let bindings = ParamEnv::default();
        let validated = producer.validate(&bindings).expect("producer validation");
        let elaborated = producer.elaborate(&bindings).expect("producer elaboration");
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([11; 32]),
            execution_nonce: [12; 32],
        };
        let artifact_manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &validated)
                .expect("artifact manifest");
        let output = &elaborated.outputs["lookup"];
        let wire = elaborated.wire(output).expect("lookup output wire");
        let symbolic_manifest = mxx_ir_symbolic::manifest::export_manifest(
            production.clone(),
            &BTreeMap::from([(
                "lookup".to_owned(),
                mxx_ir_symbolic::manifest::ExportArtifact {
                    wire_type: wire.wire_type.clone(),
                    expression: wire.expression,
                    family: wire.family.clone(),
                    content_hash: None,
                    layout: None,
                },
            )]),
            &elaborated.atoms,
            &elaborated.expressions,
            &elaborated.preimage_relations,
            elaborated.assumption_digest,
        )
        .expect("symbolic manifest");

        let imported = ring.family_artifact_input(
            production.clone(),
            "lookup",
            2,
            (1, 1),
            ArtifactConfidentiality::Public,
        );
        let indices = ring.input_family("indices", 2, (1, 1));
        let selected = indices
            .parallel_map(move |_, index| imported.get(index.extract_coefficient(0)))
            .expect("parallel artifact selection");
        let consumer = DslContext::new("artifact-family-consumer")
            .family_output("selected", selected)
            .expect("selected output")
            .build()
            .expect("consumer build");
        let imported = consumer
            .elaborate_with_manifests(
                &bindings,
                &BTreeMap::from([(production, artifact_manifest)]),
                &[symbolic_manifest],
            )
            .expect("consumer elaboration");
        let selected = imported
            .wire(&imported.outputs["selected"])
            .and_then(|wire| wire.family.as_ref())
            .expect("selected symbolic family");
        assert!(matches!(
            selected,
            mxx_ir_symbolic::elaborate::SymbolicFamily::StructuralTemplate { .. }
        ));
    }
}
