//! Declarative typed construction API for mxx graphs.
//!
//! Executable operations create immutable `mxx-ir-core` nodes immediately.

use mxx_ir_core::{
    CapturePolicy, CompileParameter, CompileParameterKind, FreezeError, Graph, GraphOutput,
    IntExpr, NodeHandle, ParamEnv, RealExpr, ScopedWireRef, SealMap, SealedSubgraph,
    SubgraphHandle, ValueHandle,
    artifact::{ArtifactConfidentiality, ProductionId},
    graph::with_new_construction_scope,
    node::{
        ArtifactInput, ConstantMatrix, HashVariant, IndexRange, LoopInputMode, MatrixBinaryOp,
        NodeKind, ParallelLoop, SampleRange, SequentialLoop,
    },
    types::{MatrixType, WireType},
};
use num_bigint::BigUint;
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
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
    #[error(transparent)]
    StructuralValidation(#[from] mxx_ir_core::ValidationError),
    #[error("ideal and predicate specifications must be sampler-free")]
    NonPureSpecification,
    #[error("a pure predicate must have exactly one boolean output")]
    PredicateOutput,
    #[error("semantic anchor could not be resolved in the frozen graph: {0}")]
    SemanticAnchorResolution(String),
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
pub struct SmallMatrixType {
    pub matrix: MatrixType,
    pub max_coefficient_bound: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PreimageType {
    pub matrix: MatrixType,
    pub max_coefficient_bound: IntExpr,
}

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
pub struct FamilyType<S> {
    pub element: S,
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
    pub fn small_matrix_input(
        &self,
        name: impl Into<String>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
    ) -> SmallMatrix {
        SmallMatrix::source_input(
            name.into(),
            self.matrix_type(shape),
            max_coefficient_bound.into(),
            None,
        )
    }

    #[track_caller]
    pub fn small_matrix_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
        confidentiality: ArtifactConfidentiality,
    ) -> SmallMatrix {
        let artifact_name = artifact_name.into();
        SmallMatrix::source_input(
            artifact_name.clone(),
            self.matrix_type(shape),
            max_coefficient_bound.into(),
            Some(ArtifactInput { production_id, artifact_name, confidentiality }),
        )
    }

    #[track_caller]
    pub fn preimage_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
        confidentiality: ArtifactConfidentiality,
    ) -> Preimage {
        let artifact_name = artifact_name.into();
        let matrix_type = self.matrix_type(shape);
        let max_coefficient_bound = max_coefficient_bound.into();
        let wire_type = WireType::Preimage {
            matrix: matrix_type.clone(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        };
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
            max_coefficient_bound,
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
            Mat::source_input(
                "__trapdoor-family-public-schema".to_owned(),
                matrix_type.clone(),
                None,
            ),
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
        Family::<Mat>::source_input(
            name.into(),
            Mat::source_input("__family-element".to_owned(), element, None),
            count.into(),
            None,
        )
    }

    #[track_caller]
    pub fn small_matrix_input_family(
        &self,
        name: impl Into<String>,
        count: impl Into<IntExpr>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
    ) -> Family<SmallMatrix> {
        Family::<SmallMatrix>::source_input(
            name.into(),
            SmallMatrix::source_input(
                "__small-matrix-family-element".to_owned(),
                self.matrix_type(shape),
                max_coefficient_bound.into(),
                None,
            ),
            count.into(),
            None,
        )
    }

    #[track_caller]
    pub fn preimage_input_family(
        &self,
        name: impl Into<String>,
        count: impl Into<IntExpr>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
    ) -> Family<Preimage> {
        let matrix_type = self.matrix_type(shape);
        let max_coefficient_bound = max_coefficient_bound.into();
        Family::<Preimage>::source_input(
            name.into(),
            Preimage::source_input(
                "__preimage-family-element".to_owned(),
                matrix_type,
                max_coefficient_bound,
                None,
            ),
            count.into(),
            None,
        )
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
            Mat::source_input("__family-element".to_owned(), self.matrix_type(shape), None),
            count.into(),
            Some(ArtifactInput { production_id, artifact_name, confidentiality }),
        )
    }

    #[track_caller]
    pub fn small_matrix_family_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        count: impl Into<IntExpr>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
        confidentiality: ArtifactConfidentiality,
    ) -> Family<SmallMatrix> {
        let artifact_name = artifact_name.into();
        Family::<SmallMatrix>::source_input(
            format!("artifact:{artifact_name}"),
            SmallMatrix::source_input(
                "__small-matrix-family-element".to_owned(),
                self.matrix_type(shape),
                max_coefficient_bound.into(),
                None,
            ),
            count.into(),
            Some(ArtifactInput { production_id, artifact_name, confidentiality }),
        )
    }

    #[track_caller]
    pub fn preimage_family_artifact_input(
        &self,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        count: impl Into<IntExpr>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
        confidentiality: ArtifactConfidentiality,
    ) -> Family<Preimage> {
        let artifact_name = artifact_name.into();
        let matrix_type = self.matrix_type(shape);
        let max_coefficient_bound = max_coefficient_bound.into();
        Family::<Preimage>::source_input(
            format!("artifact:{artifact_name}"),
            Preimage::source_input(
                "__preimage-family-element".to_owned(),
                matrix_type,
                max_coefficient_bound,
                None,
            ),
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
    ) -> SmallMatrix {
        self.hash_bounded(
            key,
            tag,
            shape,
            HashVariant::Decomposed,
            Some(base.into()),
            Some(digit_count.into()),
        )
    }

    #[track_caller]
    pub fn hash_small_decomposed(
        &self,
        key: Bytes,
        tag: impl Into<HashTag>,
        shape: impl IntoShape,
        base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
    ) -> SmallMatrix {
        self.hash_bounded(
            key,
            tag,
            shape,
            HashVariant::SmallDecomposed,
            Some(base.into()),
            Some(digit_count.into()),
        )
    }

    #[track_caller]
    fn hash_bounded(
        &self,
        key: Bytes,
        tag: impl Into<HashTag>,
        shape: impl IntoShape,
        variant: HashVariant,
        base: Option<IntExpr>,
        digit_count: Option<IntExpr>,
    ) -> SmallMatrix {
        let matrix_type = self.matrix_type(shape);
        let tag = tag.into();
        let base = base.expect("bounded hash requires a gadget base");
        let max_coefficient_bound = if matches!(variant, HashVariant::SmallDecomposed) {
            IntExpr::Sub(Box::new(base.clone()), Box::new(IntExpr::constant(1))).canonicalize()
        } else {
            IntExpr::RoundDiv(Box::new(base.clone()), Box::new(IntExpr::constant(2))).canonicalize()
        };
        let pending = Pending::merge([key.pending.clone(), tag.pending]);
        let mut arguments = vec![key.value];
        arguments.extend(tag.dynamic);
        let wire_type = WireType::SmallMatrix {
            matrix: matrix_type.clone(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::HashSample {
                matrix_type: matrix_type.clone(),
                variant,
                tag_prefix: tag.prefix,
                tag_expressions: tag.binary,
                tag_decimal_expressions: tag.decimal,
                tag_u64_le_expressions: tag.u64_le,
                base: Some(base),
                digit_count,
            },
            arguments,
            vec![wire_type],
        );
        SmallMatrix {
            value: node.output(0).expect("bounded hash output"),
            matrix_type,
            max_coefficient_bound,
            pending,
        }
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
        let pending = Pending::merge([key.pending.clone(), tag.pending]);
        let mut arguments = vec![key.value];
        arguments.extend(tag.dynamic);
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

    #[track_caller]
    pub fn mul_small_rhs(self, rhs: SmallMatrix) -> Self {
        let output_type = MatrixType {
            rows: self.matrix_type.rows.clone(),
            columns: rhs.matrix_type.columns.clone(),
            ..self.matrix_type.clone()
        };
        let pending = Pending::merge([self.pending.clone(), rhs.pending.clone()]);
        let node = NodeHandle::new(
            NodeKind::MatrixMulSmallRhs,
            vec![self.value, rhs.value],
            vec![WireType::Matrix(output_type.clone())],
        );
        Self {
            value: node.output(0).expect("small RHS multiplication"),
            matrix_type: output_type,
            pending,
        }
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
    pub fn decompose(self, base: impl Into<IntExpr>, digit_count: impl Into<IntExpr>) -> Preimage {
        self.decompose_with_mode(base.into(), digit_count.into(), false)
    }

    #[track_caller]
    pub fn small_decompose(
        self,
        base: impl Into<IntExpr>,
        digit_count: impl Into<IntExpr>,
    ) -> Preimage {
        self.decompose_with_mode(base.into(), digit_count.into(), true)
    }

    fn decompose_with_mode(self, base: IntExpr, digit_count: IntExpr, small: bool) -> Preimage {
        let ty = MatrixType {
            rows: IntExpr::Mul(
                Box::new(self.matrix_type.rows.clone()),
                Box::new(digit_count.clone()),
            )
            .canonicalize(),
            ..self.matrix_type.clone()
        };
        let pending = self.pending;
        let max_coefficient_bound = if small {
            IntExpr::Sub(Box::new(base.clone()), Box::new(IntExpr::constant(1))).canonicalize()
        } else {
            IntExpr::RoundDiv(Box::new(base.clone()), Box::new(IntExpr::constant(2))).canonicalize()
        };
        let node = NodeHandle::new(
            NodeKind::GadgetDecompose { base, small, digit_count },
            vec![self.value],
            vec![WireType::Preimage {
                matrix: ty.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
            }],
        );
        let preimage = Preimage {
            value: node.output(0).expect("decomposition"),
            matrix_type: ty,
            max_coefficient_bound,
            pending,
        };
        preimage
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
    max_coefficient_bound: IntExpr,
    pending: Pending,
}

impl Preimage {
    fn source_input(
        name: String,
        matrix_type: MatrixType,
        max_coefficient_bound: IntExpr,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let wire_type = WireType::Preimage {
            matrix: matrix_type.clone(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: wire_type.clone(), artifact },
            Vec::new(),
            vec![wire_type],
        );
        Self {
            value: node.output(0).expect("preimage input"),
            matrix_type,
            max_coefficient_bound,
            pending: Pending::default(),
        }
    }

    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn matrix_type(&self) -> &MatrixType {
        &self.matrix_type
    }

    pub fn max_coefficient_bound(&self) -> &IntExpr {
        &self.max_coefficient_bound
    }

    #[track_caller]
    pub fn mul_small_rhs(self, lhs: Mat) -> Mat {
        let output_type = MatrixType {
            rows: lhs.matrix_type.rows.clone(),
            columns: self.matrix_type.columns.clone(),
            ..lhs.matrix_type.clone()
        };
        let pending = Pending::merge([lhs.pending.clone(), self.pending.clone()]);
        let node = NodeHandle::new(
            NodeKind::MatrixMulSmallRhs,
            vec![lhs.value, self.value],
            vec![WireType::Matrix(output_type.clone())],
        );
        Mat {
            value: node.output(0).expect("preimage multiplication"),
            matrix_type: output_type,
            pending,
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

    pub fn preimage_max_coefficient_bound(&self) -> &IntExpr {
        &self.preimage_max_coefficient_bound
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
            vec![WireType::Preimage {
                matrix: ty.clone(),
                max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
            }],
        );
        let preimage = Preimage {
            value: node.output(0).expect("preimage"),
            matrix_type: ty,
            max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
            pending,
        };
        preimage
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
        let source_count = self.count.clone();
        let output_count = indices.count.clone();
        let schema = self.element_schema.clone();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let index = IntType.placeholders();
                let public = Family::<Mat>::source_input(
                    "gather-trapdoor-public".to_owned(),
                    Mat::source_input("__family-element".to_owned(), schema.matrix.clone(), None),
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
        body_value.parallel_families(&node, &mut 0, &output_count, pending)
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
}

impl Family<Int> {
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
        body_value.parallel_families(&node, &mut 0, &segment_count, pending)
    }

    pub fn parallel_select_mats(
        self,
        candidates: Vec<Family<Mat>>,
    ) -> Result<Family<Mat>, DslError> {
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
pub struct Family<T> {
    value: ValueHandle,
    element_schema: T,
    count: IntExpr,
    pending: Pending,
}

#[doc(hidden)]
pub trait FamilyElement: GraphValue + Clone {
    fn normalize_for_family(self) -> Self;
}

impl FamilyElement for Mat {
    fn normalize_for_family(self) -> Self {
        self
    }
}

impl FamilyElement for SmallMatrix {
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

impl<T: FamilyElement> Family<T> {
    fn source_input(
        name: String,
        element_schema: T,
        count: IntExpr,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let element_wire_types = element_schema.schema().wire_types();
        assert_eq!(element_wire_types.len(), 1, "family elements must have one wire");
        let family_type = WireType::IndexedFamily {
            element: Box::new(element_wire_types.into_iter().next().expect("family wire")),
            count: count.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: family_type.clone(), artifact },
            Vec::new(),
            vec![family_type],
        );
        Self {
            value: node.output(0).expect("family"),
            element_schema,
            count,
            pending: Pending::default(),
        }
    }

    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    /// Selects one same-schema family without materializing the other branches.
    pub fn select(selector: Int, branches: Vec<Self>) -> Result<Self, DslError> {
        let Some(first) = branches.first() else {
            return Err(DslError::Schema);
        };
        let first_wire_types = first.element_schema.schema().wire_types();
        let [first_wire_type] = first_wire_types.as_slice() else {
            return Err(DslError::Schema);
        };
        if branches.iter().any(|branch| {
            branch.count != first.count ||
                branch.element_schema.schema().wire_types() != first_wire_types
        }) {
            return Err(DslError::FamilyCountMismatch);
        }
        let pending = Pending::merge(
            std::iter::once(selector.pending.clone())
                .chain(branches.iter().map(|branch| branch.pending.clone())),
        );
        let mut arguments = vec![selector.value];
        arguments.extend(branches.iter().map(|branch| branch.value.clone()));
        let family_type = WireType::IndexedFamily {
            element: Box::new(first_wire_type.clone()),
            count: first.count.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Select { count: IntExpr::constant(branches.len()) },
            arguments,
            vec![family_type],
        );
        Ok(Self {
            value: node.output(0).expect("selected family"),
            element_schema: first.element_schema.clone(),
            count: first.count.clone(),
            pending,
        })
    }

    pub fn get_static(&self, index: impl Into<IntExpr>) -> T {
        let schema = self.element_schema.schema();
        let wire_types = schema.wire_types();
        assert_eq!(wire_types.len(), 1, "family elements must have one wire");
        let node = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: index.into() },
            vec![self.value.clone()],
            wire_types,
        );
        T::from_values(&schema, &[node.output(0).expect("family element")], self.pending.clone())
            .expect("family element schema")
    }

    pub fn get(&self, index: Int) -> T {
        let schema = self.element_schema.schema();
        let wire_types = schema.wire_types();
        assert_eq!(wire_types.len(), 1, "family elements must have one wire");
        let pending = Pending::merge([self.pending.clone(), index.pending]);
        let node = NodeHandle::new(
            NodeKind::FamilyGetDynamic,
            vec![self.value.clone(), index.value],
            wire_types,
        );
        T::from_values(&schema, &[node.output(0).expect("family element")], pending)
            .expect("family element schema")
    }

    pub fn parallel_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex, T) -> R,
    ) -> Result<R::Families, DslError>
    where
        T::Schema: GraphValueSchema<Value = T>,
    {
        let outer_family = self.value.clone();
        let count = self.count.clone();
        let schema = self.element_schema.schema();
        let (index_slot, (body_value, explicit_input, scope)) = with_loop_index(|index| {
            with_new_construction_scope(|scope| {
                let input = schema.placeholders();
                let output = body(index, input.clone());
                (output, input.flatten(), scope)
            })
        });
        let body_outputs = body_value.flatten();
        let sealed = SubgraphHandle::seal(
            "parallel-map-family-body",
            scope,
            explicit_input,
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
                index_slot,
                bindings: Vec::new(),
                input_modes: modes,
            },
        );
        let pending = Pending::merge([self.pending, body_value.pending().remap(&sealed.remap)]);
        let mut next_port = 0;
        body_value.parallel_families(&node, &mut next_port, &count, pending)
    }

    pub fn parallel_map(self, body: impl FnOnce(LoopIndex, T) -> T) -> Result<Self, DslError>
    where
        T: ParallelOutput<Families = Family<T>>,
        T::Schema: GraphValueSchema<Value = T>,
    {
        self.parallel_map_values(|index, value| body(index, value).normalize_for_family())
    }

    pub fn parallel_gather(self, indices: Family<Int>) -> Result<Self, DslError>
    where
        T: ParallelOutput<Families = Family<T>>,
        T::Schema: GraphValueSchema<Value = T>,
    {
        let source_count = self.count.clone();
        let output_count = indices.count.clone();
        let element_schema = self.element_schema.schema();
        let (index_slot, (body_value, explicit_inputs, scope)) = with_loop_index(|_| {
            with_new_construction_scope(|scope| {
                let index = IntType.placeholders();
                let source = Family::<T>::source_input(
                    "gather-source".to_owned(),
                    element_schema.placeholders(),
                    source_count.clone(),
                    None,
                );
                let output = source.get(index.clone());
                (output, vec![index.value, source.value], scope)
            })
        });
        let sealed = SubgraphHandle::seal(
            "parallel-gather-family-body",
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
        body_value.parallel_families(&node, &mut 0, &output_count, pending)
    }
}

impl<T> ParallelOutput for T
where
    T: FamilyElement,
    T::Schema: GraphValueSchema<Value = T>,
{
    type Families = Family<T>;

    fn parallel_family_types(&self, count: &IntExpr) -> Result<Vec<WireType>, DslError> {
        let mut element_types = self.schema().wire_types();
        if element_types.len() != 1 {
            return Err(DslError::Schema);
        }
        Ok(vec![WireType::IndexedFamily {
            element: Box::new(element_types.pop().expect("one-wire family element")),
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
            element_schema: self.schema().placeholders(),
            count: count.clone(),
            pending,
        })
    }
}

impl Family<Mat> {
    pub fn element_type(&self) -> &MatrixType {
        &self.element_schema.matrix_type
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

    pub fn parallel_zip_many_with_broadcast_values<R: ParallelOutput>(
        zipped: Vec<Self>,
        broadcast: Vec<Self>,
        body: impl FnOnce(LoopIndex, Vec<Mat>, Vec<Self>) -> Result<R, DslError>,
    ) -> Result<R::Families, DslError> {
        let Some(first) = zipped.first() else {
            return Err(DslError::Schema);
        };
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
                            Mat::source_input("__family-element".to_owned(), matrix_type, None),
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
        let mut modes = vec![LoopInputMode::Zip; zipped.len()];
        arguments.extend(broadcast.iter().map(|family| family.value.clone()));
        modes.extend((0..broadcast.len()).map(|_| LoopInputMode::Broadcast));
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
            zipped
                .into_iter()
                .map(|family| family.pending)
                .chain(broadcast.into_iter().map(|family| family.pending))
                .chain(std::iter::once(body_value.pending().remap(&sealed.remap))),
        );
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
        body_value.parallel_families(&node, &mut 0, &count, pending)
    }

    pub fn try_map_values<R: ParallelOutput>(
        self,
        body: impl FnOnce(LoopIndex) -> Result<R, DslError>,
    ) -> Result<R::Families, DslError> {
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

/// Builds a zipped loop whose body receives three zipped elements and three
/// explicitly broadcast family arguments. Broadcast families are formal
/// inputs of the sealed body, so accidental outer-family captures are
/// rejected rather than silently captured.
pub fn parallel_zip_bundle_with_broadcast<A, B, C, D, E, F, R: ParallelOutput>(
    zipped: (Family<A>, Family<B>, Family<C>),
    broadcast: (Family<D>, Family<E>, Family<F>),
    body: impl FnOnce(LoopIndex, (A, B, C), (Family<D>, Family<E>, Family<F>)) -> R,
) -> Result<R::Families, DslError>
where
    A: GraphValue,
    B: GraphValue,
    C: GraphValue,
    D: FamilyElement,
    E: FamilyElement,
    F: FamilyElement,
{
    if zipped.0.count != zipped.1.count || zipped.0.count != zipped.2.count {
        return Err(DslError::FamilyCountMismatch);
    }
    let count = zipped.0.count.clone();
    let broadcast_counts =
        (broadcast.0.count.clone(), broadcast.1.count.clone(), broadcast.2.count.clone());
    let schemas = (
        zipped.0.element_schema.schema(),
        zipped.1.element_schema.schema(),
        zipped.2.element_schema.schema(),
    );
    let broadcast_schemas = (
        broadcast.0.element_schema.clone(),
        broadcast.1.element_schema.clone(),
        broadcast.2.element_schema.clone(),
    );
    let (index_slot, body_result) = with_loop_index(|index| {
        with_new_construction_scope(|scope| {
            let mut next = 0;
            let a = schemas.0.placeholders_from(&mut next);
            let b = schemas.1.placeholders_from(&mut next);
            let c = schemas.2.placeholders_from(&mut next);
            let d = Family::<D>::source_input(
                "broadcast-0".to_owned(),
                broadcast_schemas.0.clone(),
                broadcast_counts.0.clone(),
                None,
            );
            let e = Family::<E>::source_input(
                "broadcast-1".to_owned(),
                broadcast_schemas.1.clone(),
                broadcast_counts.1.clone(),
                None,
            );
            let f = Family::<F>::source_input(
                "broadcast-2".to_owned(),
                broadcast_schemas.2.clone(),
                broadcast_counts.2.clone(),
                None,
            );
            let mut explicit_inputs = a.flatten();
            explicit_inputs.extend(b.flatten());
            explicit_inputs.extend(c.flatten());
            explicit_inputs.extend(d.flatten());
            explicit_inputs.extend(e.flatten());
            explicit_inputs.extend(f.flatten());
            let output = body(index, (a, b, c), (d, e, f));
            Ok::<_, DslError>((output, explicit_inputs, scope))
        })
    });
    let (body_value, explicit_inputs, scope) = body_result?;
    let sealed = SubgraphHandle::seal(
        "parallel-zip-broadcast-body",
        scope,
        explicit_inputs,
        body_value.flatten(),
        CapturePolicy::Reject,
    )?;
    let remapped_pending = body_value.pending().remap(&sealed.remap);
    let mut arguments = vec![
        zipped.0.value,
        zipped.1.value,
        zipped.2.value,
        broadcast.0.value,
        broadcast.1.value,
        broadcast.2.value,
    ];
    let modes = vec![
        LoopInputMode::Zip,
        LoopInputMode::Zip,
        LoopInputMode::Zip,
        LoopInputMode::Broadcast,
        LoopInputMode::Broadcast,
        LoopInputMode::Broadcast,
    ];
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    let mut modes = modes;
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
    let pending = Pending::merge([
        zipped.0.pending,
        zipped.1.pending,
        zipped.2.pending,
        broadcast.0.pending,
        broadcast.1.pending,
        broadcast.2.pending,
        remapped_pending,
    ]);
    body_value.parallel_families(&node, &mut 0, &count, pending)
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

/// Semantic wire sets retained by the DSL and resolved exactly once when the graph is frozen.
///
/// Labels are proof-facing names, not executable nodes. A label may name more than one wire so
/// callers can identify a typed tuple or family interface without reconstructing it by searching
/// the frozen graph.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FrozenSemanticAnchors {
    entries: BTreeMap<String, Vec<ScopedWireRef>>,
}

/// A frozen, owner-crate rule reference retained alongside an executable graph.
///
/// This is generator infrastructure: it identifies the exact wires to which an owning crate's
/// checked operational rule applies.  It contains neither a claimed equation nor a numeric bound.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Deserialize, serde::Serialize)]
#[doc(hidden)]
pub struct FrozenDerivationAttachment {
    pub namespace: String,
    pub rule: String,
    pub roles: Vec<(String, ScopedWireRef)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[doc(hidden)]
pub struct FrozenDerivationAttachments {
    entries: Vec<FrozenDerivationAttachment>,
}

impl FrozenDerivationAttachments {
    #[doc(hidden)]
    pub fn iter(&self) -> impl Iterator<Item = &FrozenDerivationAttachment> {
        self.entries.iter()
    }

    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct DerivationAttachment {
    namespace: String,
    rule: String,
    roles: Vec<(String, ValueHandle)>,
}

impl FrozenSemanticAnchors {
    pub fn get(&self, name: &str) -> Option<&[ScopedWireRef]> {
        self.entries.get(name).map(Vec::as_slice)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &[ScopedWireRef])> {
        self.entries.iter().map(|(name, wires)| (name.as_str(), wires.as_slice()))
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Default)]
#[doc(hidden)]
pub struct Pending {
    semantic_anchors: BTreeMap<String, Vec<ValueHandle>>,
    derivation_attachments: Vec<DerivationAttachment>,
}

impl Pending {
    #[doc(hidden)]
    pub fn merge(values: impl IntoIterator<Item = Pending>) -> Self {
        let mut merged = Self::default();
        for pending in values {
            for (name, wires) in pending.semantic_anchors {
                merged.semantic_anchors.entry(name).or_default().extend(wires);
            }
            merged.derivation_attachments.extend(pending.derivation_attachments);
        }
        merged
    }

    fn remap(&self, map: &SealMap) -> Self {
        let semantic_anchors = self
            .semantic_anchors
            .iter()
            .map(|(name, wires)| {
                let wires = wires
                    .iter()
                    .map(|wire| map.resolve(wire).cloned().unwrap_or_else(|| wire.clone()))
                    .collect();
                (name.clone(), wires)
            })
            .collect();
        let derivation_attachments = self
            .derivation_attachments
            .iter()
            .map(|attachment| DerivationAttachment {
                namespace: attachment.namespace.clone(),
                rule: attachment.rule.clone(),
                roles: attachment
                    .roles
                    .iter()
                    .map(|(role, wire)| {
                        (role.clone(), map.resolve(wire).cloned().unwrap_or_else(|| wire.clone()))
                    })
                    .collect(),
            })
            .collect();
        Self { semantic_anchors, derivation_attachments }
    }

    fn with_semantic_anchor(mut self, name: String, wires: Vec<ValueHandle>) -> Self {
        self.semantic_anchors.entry(name).or_default().extend(wires);
        self
    }

    fn with_derivation_attachment(mut self, attachment: DerivationAttachment) -> Self {
        self.derivation_attachments.push(attachment);
        self
    }
}

/// Adds a proof-facing name to a DSL value without changing the executable graph.
pub trait SemanticAnchor: GraphValue + Sized {
    fn semantic_anchor(self, name: impl Into<String>) -> Result<Self, DslError> {
        let schema = self.schema();
        let wires = self.flatten();
        let pending = self.pending().with_semantic_anchor(name.into(), wires.clone());
        Self::from_values(&schema, &wires, pending)
    }
}

impl<T: GraphValue> SemanticAnchor for T {}

/// Attaches an owning-crate operational-rule reference without changing the executable graph.
///
/// This trait is intentionally hidden from normal DSL documentation.  Reusable gadget and BGG
/// builders use it mechanically; protocol authors do not supply bounds, identities, or rules.
#[doc(hidden)]
pub trait DerivationAttachmentValue: GraphValue + Sized {
    fn derivation_attachment(
        self,
        namespace: impl Into<String>,
        rule: impl Into<String>,
        roles: Vec<(String, ValueHandle)>,
    ) -> Result<Self, DslError> {
        let schema = self.schema();
        let wires = self.flatten();
        let pending = self.pending().with_derivation_attachment(DerivationAttachment {
            namespace: namespace.into(),
            rule: rule.into(),
            roles,
        });
        Self::from_values(&schema, &wires, pending)
    }
}

impl<T: GraphValue> DerivationAttachmentValue for T {}

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

    pub fn output<V: GraphValue>(
        mut self,
        name: impl Into<String>,
        value: V,
    ) -> Result<Self, DslError> {
        self.insert_graph_value(name.into(), value, None)?;
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

    pub fn public_output<V: GraphValue>(
        mut self,
        name: impl Into<String>,
        value: V,
    ) -> Result<Self, DslError> {
        self.insert_graph_value(name.into(), value, Some(ArtifactConfidentiality::Public))?;
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

    pub fn private_output<V: GraphValue>(
        mut self,
        name: impl Into<String>,
        value: V,
    ) -> Result<Self, DslError> {
        self.insert_graph_value(name.into(), value, Some(ArtifactConfidentiality::Private))?;
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

    fn insert_graph_value<V: GraphValue>(
        &mut self,
        name: String,
        value: V,
        confidentiality: Option<ArtifactConfidentiality>,
    ) -> Result<(), DslError> {
        let pending = value.pending();
        let values = value.flatten();
        let [value] = values.as_slice() else { return Err(DslError::Schema) };
        self.insert_pending_output(name, value.clone(), pending, confidentiality)
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
        self.build_with_freeze_map().map(|(graph, _)| graph)
    }

    #[doc(hidden)]
    pub fn build_with_freeze_map(self) -> Result<(BuiltGraph, mxx_ir_core::FreezeMap), DslError> {
        let pending = Pending::merge(self.outputs.values().map(|output| output.pending.clone()));
        let root_scope = mxx_ir_core::current_construction_scope();
        let retained_roots = pending
            .semantic_anchors
            .values()
            .flat_map(|wires| wires.iter().cloned())
            .chain(
                pending
                    .derivation_attachments
                    .iter()
                    .flat_map(|attachment| attachment.roles.iter().map(|(_, wire)| wire.clone())),
            )
            .filter(|wire| wire.construction_scope() == root_scope)
            .collect();
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
            retained_roots,
            Vec::new(),
            self.real_constants,
        )?;
        mxx_ir_core::validate_structure(&graph)?;
        let anchors = pending
            .semantic_anchors
            .into_iter()
            .map(|(name, wires)| {
                let wires = wires
                    .iter()
                    .map(|wire| freeze_map.resolve_unique(wire).cloned())
                    .collect::<Result<BTreeSet<_>, _>>()?
                    .into_iter()
                    .collect();
                Ok((name, wires))
            })
            .collect::<Result<BTreeMap<_, _>, mxx_ir_core::FreezeResolveError>>()
            .map_err(|error| DslError::SemanticAnchorResolution(error.to_string()))?;
        let mut attachments = pending
            .derivation_attachments
            .into_iter()
            .map(|attachment| {
                let roles = attachment
                    .roles
                    .into_iter()
                    .map(|(role, wire)| {
                        freeze_map.resolve_unique(&wire).cloned().map(|wire| (role, wire))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(FrozenDerivationAttachment {
                    namespace: attachment.namespace,
                    rule: attachment.rule,
                    roles,
                })
            })
            .collect::<Result<Vec<_>, mxx_ir_core::FreezeResolveError>>()
            .map_err(|error| DslError::SemanticAnchorResolution(error.to_string()))?;
        attachments.sort();
        attachments.dedup();
        Ok((
            BuiltGraph {
                graph,
                anchors: FrozenSemanticAnchors { entries: anchors },
                derivation_attachments: FrozenDerivationAttachments { entries: attachments },
            },
            freeze_map,
        ))
    }
}

pub struct BuiltGraph {
    pub graph: Graph,
    pub anchors: FrozenSemanticAnchors,
    #[doc(hidden)]
    pub derivation_attachments: FrozenDerivationAttachments,
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
                NodeKind::UniformResidueSample { .. } |
                    NodeKind::UniformIntervalSample { .. } |
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

#[derive(Clone)]
pub struct SmallMatrix {
    value: ValueHandle,
    matrix_type: MatrixType,
    max_coefficient_bound: IntExpr,
    pending: Pending,
}

impl SmallMatrix {
    fn source_input(
        name: String,
        matrix_type: MatrixType,
        max_coefficient_bound: IntExpr,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let wire_type = WireType::SmallMatrix {
            matrix: matrix_type.clone(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input { name, wire_type: wire_type.clone(), artifact },
            Vec::new(),
            vec![wire_type],
        );
        Self {
            value: node.output(0).expect("small matrix input"),
            matrix_type,
            max_coefficient_bound,
            pending: Pending::default(),
        }
    }

    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn matrix_type(&self) -> &MatrixType {
        &self.matrix_type
    }

    pub fn max_coefficient_bound(&self) -> &IntExpr {
        &self.max_coefficient_bound
    }
}

impl GraphValue for SmallMatrix {
    type Schema = SmallMatrixType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        SmallMatrixType {
            matrix: self.matrix_type.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
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
            matrix_type: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
            pending,
        })
    }
}

impl GraphValueSchema for SmallMatrixType {
    type Value = SmallMatrix;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        SmallMatrix::source_input(
            argument_name(next, "small-matrix"),
            self.matrix.clone(),
            self.max_coefficient_bound.clone(),
            None,
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::SmallMatrix {
            matrix: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }]
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
        PreimageType {
            matrix: self.matrix_type.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
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
            matrix_type: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
            pending,
        })
    }
}

impl GraphValueSchema for PreimageType {
    type Value = Preimage;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let wire_type = WireType::Preimage {
            matrix: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        };
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
            matrix_type: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
            pending: Pending::default(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Preimage {
            matrix: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }]
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
        let public = Family::<Mat>::source_input(
            argument_name(next, "trapdoor-public-family"),
            Mat::source_input("__family-element".to_owned(), self.element.matrix.clone(), None),
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

impl<T> GraphValue for Family<T>
where
    T: FamilyElement,
{
    type Schema = FamilyType<T::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        FamilyType { element: self.element_schema.schema(), count: self.count.clone() }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            value: value.clone(),
            element_schema: schema.element.placeholders(),
            count: schema.count.clone(),
            pending,
        })
    }
}

impl<S> GraphValueSchema for FamilyType<S>
where
    S: GraphValueSchema,
    S::Value: FamilyElement,
{
    type Value = Family<S::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let element_schema = self.element.clone();
        Family::<S::Value>::source_input(
            argument_name(next, "family"),
            element_schema.placeholders_from(next),
            self.count.clone(),
            None,
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        let mut element_types = self.element.wire_types();
        assert_eq!(element_types.len(), 1, "family elements must have one wire");
        vec![WireType::IndexedFamily {
            element: Box::new(element_types.pop().expect("family wire")),
            count: self.count.clone(),
        }]
    }
}

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
    use num_bigint::BigInt;

    #[test]
    fn executable_arithmetic_builds_and_validates() {
        let ring = Ring::new(17, 8);
        let input = ring.input("input", (2, 2));
        let output = input.clone() + input;
        let built = DslContext::new("sum").output("sum", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn small_rhs_graph_preserves_bounded_kind_and_has_no_scale_erasure() {
        let ring = Ring::new(17, 8);
        let lhs = ring.input("lhs", (2, 3));
        let rhs = ring.small_matrix_input("rhs", (3, 4), 7);
        let output = lhs.mul_small_rhs(rhs);
        let built =
            DslContext::new("small-rhs").output("product", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let nodes = built.graph.root_scope().nodes();
        assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)));
        assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::MatrixScale { .. })));
        assert!(nodes.iter().any(|node| {
            node.output_types().iter().any(|wire| {
                matches!(
                    wire,
                    WireType::SmallMatrix { max_coefficient_bound, .. }
                        if *max_coefficient_bound == IntExpr::constant(7)
                )
            })
        }));
    }

    #[test]
    fn preimage_rhs_graph_preserves_relation_typed_multiplication() {
        let ring = Ring::new(17, 8);
        let lhs = ring.input("lhs", (2, 3));
        let trapdoor = ring.sample_trapdoor(1, 1, 4, 1, 3);
        let rhs = trapdoor.sample_preimage(ring.zero((1, 4)), (3, 4));
        let output = rhs.mul_small_rhs(lhs);
        let built =
            DslContext::new("preimage-rhs").output("product", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let nodes = built.graph.root_scope().nodes();
        assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)));
        assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::MatrixScale { .. })));
        assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. })));
    }

    #[test]
    fn gadget_modes_are_relation_typed_and_hash_modes_are_generic_bounded() {
        let ring = Ring::new(17, 8);
        let key = ring.bytes_input("key", 32);
        let input = ring.input("input", (1, 1));
        let regular = input.clone().decompose(4, 2);
        let unsigned = input.small_decompose(4, 2);
        assert!(matches!(
            regular.value_handle().wire_type(),
            WireType::Preimage { max_coefficient_bound, .. }
                if max_coefficient_bound.evaluate(&ParamEnv::default()).unwrap() == 2.into()
        ));
        assert!(matches!(
            unsigned.value_handle().wire_type(),
            WireType::Preimage { max_coefficient_bound, .. }
                if max_coefficient_bound.evaluate(&ParamEnv::default()).unwrap() == 3.into()
        ));

        let balanced_hash = ring.hash_decomposed(key.clone(), tag!("balanced"), (2, 2), 4, 2);
        let unsigned_hash = ring.hash_small_decomposed(key, tag!("unsigned"), (2, 2), 4, 2);
        assert!(matches!(
            balanced_hash.value_handle().wire_type(),
            WireType::SmallMatrix { max_coefficient_bound, .. }
                if max_coefficient_bound.evaluate(&ParamEnv::default()).unwrap() == 2.into()
        ));
        assert!(matches!(
            unsigned_hash.value_handle().wire_type(),
            WireType::SmallMatrix { max_coefficient_bound, .. }
                if max_coefficient_bound.evaluate(&ParamEnv::default()).unwrap() == 3.into()
        ));
    }

    #[test]
    fn gadget_bounds_and_validation_edges_are_fixed() {
        let ring = Ring::new(17, 8);
        let input = ring.input("input", (1, 1));
        let regular = input.clone().decompose(3, 2);
        let unsigned = input.clone().small_decompose(3, 2);
        assert_eq!(regular.max_coefficient_bound.evaluate(&ParamEnv::default()).unwrap(), 2.into());
        assert_eq!(
            unsigned.max_coefficient_bound.evaluate(&ParamEnv::default()).unwrap(),
            2.into()
        );

        for base in [1, 0, -1] {
            let built = DslContext::new("invalid-gadget-base")
                .output("value", ring.input("input", (1, 1)).decompose(base, 2))
                .unwrap()
                .build()
                .unwrap();
            assert!(built.validate(&ParamEnv::default()).is_err());
        }
        for digits in [0, -1] {
            let built = DslContext::new("invalid-gadget-digits")
                .output("value", ring.input("input", (1, 1)).decompose(3, digits))
                .unwrap()
                .build()
                .unwrap();
            assert!(built.validate(&ParamEnv::default()).is_err());
        }

        let huge_rows = BigInt::from(usize::MAX);
        let built = DslContext::new("gadget-row-overflow")
            .output("value", ring.input("input", (huge_rows, 1)).decompose(3, 2))
            .unwrap()
            .build()
            .unwrap();
        assert!(built.validate(&ParamEnv::default()).is_err());
    }

    #[test]
    fn bounded_families_keep_their_element_wire_kinds() {
        let ring = Ring::new(17, 8);
        let small = ring.small_matrix_input_family("small", 2, (2, 2), 3);
        let preimage = ring.preimage_input_family("preimage", 2, (2, 2), 3);
        let small_static = small.get_static(0);
        let preimage_dynamic = preimage.get(Int::constant(1));
        let mapped_small = small.clone().parallel_map(|_, value| value).unwrap();
        let mapped_preimage = preimage.clone().parallel_map(|_, value| value).unwrap();
        let built = DslContext::new("bounded-families")
            .output("small", mapped_small)
            .unwrap()
            .output("preimage", mapped_preimage)
            .unwrap()
            .output("small-static", small_static)
            .unwrap()
            .output("preimage-dynamic", preimage_dynamic)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert!(matches!(
            small.get_static(0).value_handle().wire_type(),
            WireType::SmallMatrix { .. }
        ));
        assert!(matches!(
            preimage.get_static(0).value_handle().wire_type(),
            WireType::Preimage { .. }
        ));
        let mut all_nodes = built.graph.scopes().values().flat_map(|scope| scope.nodes());
        assert!(
            all_nodes.clone().any(|node| matches!(node.kind(), NodeKind::FamilyGetStatic { .. }))
        );
        assert!(all_nodes.clone().any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic)));
        assert!(all_nodes.any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_))));
    }

    #[test]
    fn dynamic_integer_hash_tag_is_an_explicit_argument_and_preserves_pending_metadata() {
        let ring = Ring::new(17, 8);
        let row = Int::constant(7).add(Int::constant(0)).semantic_anchor("hash-row").unwrap();
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
        assert_eq!(built.anchors.get("hash-row").expect("dynamic tag anchor").len(), 1);
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn semantic_anchor_resolves_to_the_frozen_output_without_an_ir_node() {
        let ring = Ring::new(17, 8);
        let input = ring.input("input", (2, 2));
        let output = (input.clone() + input).semantic_anchor("result-carrier").unwrap();
        let built = DslContext::new("anchored-sum").output("sum", output).unwrap().build().unwrap();

        let anchor = built.anchors.get("result-carrier").unwrap();
        assert_eq!(anchor.len(), 1);
        assert_eq!(anchor[0].scope, mxx_ir_core::FrozenGraphScopeId::Root);
        assert_eq!(anchor[0].wire, built.graph.outputs()["sum"].value);
    }

    #[test]
    fn semantic_anchor_is_remapped_into_a_sealed_loop_body() {
        let ring = Ring::new(17, 8);
        let captured = ring.input("captured", (1, 1));
        let family = Parallel::range(2)
            .map(move |_| {
                (captured.clone() + captured.clone()).semantic_anchor("loop-body-sum").unwrap()
            })
            .unwrap();
        let built = DslContext::new("anchored-loop")
            .family_output("values", family)
            .unwrap()
            .build()
            .unwrap();

        let [anchor] = built.anchors.get("loop-body-sum").unwrap() else {
            panic!("one body-template wire must be anchored")
        };
        assert!(matches!(anchor.scope, mxx_ir_core::FrozenGraphScopeId::ParallelBody { .. }));
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
            .output("value", input.clone().decompose(4, 4))
            .unwrap()
            .build()
            .unwrap();
        regular.validate(&ParamEnv::default()).unwrap();
        let serialized = serde_json::to_string(&regular.graph).unwrap();
        assert!(serialized.contains("digit_count"));
        assert!(serialized.contains("\"small\":false"));

        let small = DslContext::new("small-decomposition")
            .output("value", input.clone().small_decompose(4, 4))
            .unwrap()
            .build()
            .unwrap();
        small.validate(&ParamEnv::default()).unwrap();
        assert!(serde_json::to_string(&small.graph).unwrap().contains("\"small\":true"));

        let invalid = DslContext::new("negative-decomposition-base")
            .output("value", input.decompose(-4, 4))
            .unwrap()
            .build()
            .unwrap();
        assert!(invalid.validate(&ParamEnv::default()).is_err());
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
        assert!(all_nodes.iter().any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_))));
        assert!(all_nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic)));
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
                    .mul_small_rhs(trapdoor.public_matrix())
            })
            .unwrap();
        let built = DslContext::new("parameterized-trapdoor-families")
            .int_parameter("count")
            .public_output("public", trapdoors.public_matrices())
            .unwrap()
            .private_trapdoor_family_output("trapdoors", trapdoors)
            .unwrap()
            .private_output("preimages", preimages)
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
            .public_output("public", gathered.public_matrices())
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
                trapdoor.sample_preimage(target, (6, 1)).mul_small_rhs(trapdoor.public_matrix())
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
        let built = context.public_output("output", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
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
        assert_eq!(loop_spec.input_modes, vec![LoopInputMode::Zip, LoopInputMode::Broadcast]);
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
    fn heterogeneous_zip_bundle_with_broadcast_is_explicit_and_fail_closed() {
        let ring = Ring::new(17, 8);
        let context = DslContext::new("heterogeneous-zip-broadcast");
        let zipped = (
            ring.input_family("rows", 2, (1, 1)),
            context.int_family_input("indices", 2),
            ring.input_family("targets", 2, (1, 1)),
        );
        let broadcast = (
            ring.small_matrix_input_family("small", 3, (1, 1), 7),
            ring.preimage_input_family("preimages", 4, (1, 1), 11),
            ring.input_family("matrices", 5, (1, 1)),
        );
        let output = parallel_zip_bundle_with_broadcast(
            zipped,
            broadcast,
            |index, (row, _index, target), (small, preimages, matrices)| {
                let _ = small.get(index.as_int());
                let _ = preimages.get(index.as_int());
                let _ = matrices.get(index.as_int());
                row + target
            },
        )
        .unwrap();
        let built = context.public_output("output", output).unwrap().build().unwrap();
        let loop_node = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .expect("parallel loop");
        let NodeKind::ParallelLoop(spec) = loop_node.kind() else { unreachable!() };
        assert_eq!(
            spec.input_modes,
            vec![
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Broadcast,
                LoopInputMode::Broadcast,
                LoopInputMode::Broadcast,
            ]
        );
        assert!(
            matches!(loop_node.arguments()[3].wire_type(), WireType::IndexedFamily { element, count } if *count == 3.into() && matches!(element.as_ref(), WireType::SmallMatrix { max_coefficient_bound, .. } if *max_coefficient_bound == 7.into()))
        );
        assert!(
            matches!(loop_node.arguments()[4].wire_type(), WireType::IndexedFamily { element, count } if *count == 4.into() && matches!(element.as_ref(), WireType::Preimage { max_coefficient_bound, .. } if *max_coefficient_bound == 11.into()))
        );
        assert!(
            matches!(loop_node.arguments()[5].wire_type(), WireType::IndexedFamily { element, count } if *count == 5.into() && matches!(element.as_ref(), WireType::Matrix(_)))
        );
        built.validate(&ParamEnv::default()).unwrap();

        let foreign = ring.input_family("foreign", 2, (1, 1));
        let negative_context = DslContext::new("heterogeneous-zip-broadcast-negative");
        let rejected = parallel_zip_bundle_with_broadcast(
            (
                ring.input_family("rows2", 2, (1, 1)),
                negative_context.int_family_input("indices2", 2),
                ring.input_family("targets2", 2, (1, 1)),
            ),
            (
                ring.small_matrix_input_family("small2", 2, (1, 1), 7),
                ring.preimage_input_family("preimages2", 2, (1, 1), 11),
                ring.input_family("matrices2", 2, (1, 1)),
            ),
            move |_, (row, _, _), _| row + foreign.get_static(0),
        );
        assert!(rejected.is_err(), "ordinary family capture must be rejected");
    }

    #[test]
    fn try_define_accepts_a_formal_nonartifact_family() {
        let ring = Ring::new(17, 8);
        let matrix_type = MatType(ring.matrix_type((1, 1)));
        let family_type =
            FamilyType { element: MatType(ring.matrix_type((1, 1))), count: 2.into() };
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
        let selected = Family::select(selector, vec![left, right]).unwrap();
        let built = context.public_output("selected", selected).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn preimage_family_select_preserves_bound_and_rejects_schema_mismatches() {
        let ring = Ring::new(17, 8);
        let context = DslContext::new("select-preimage-family");
        let selector = context.int_family_input("selector", 1).get_static(0);
        let left = ring.preimage_input_family("left", 2, (2, 3), 7);
        let right = ring.preimage_input_family("right", 2, (2, 3), 7);
        let selected = Family::select(selector.clone(), vec![left, right]).unwrap();
        assert!(matches!(
            selected.value_handle().wire_type(),
            WireType::IndexedFamily { element, count }
                if matches!(
                    element.as_ref(),
                    WireType::Preimage { max_coefficient_bound, .. }
                        if *max_coefficient_bound == IntExpr::constant(7)
                ) && *count == IntExpr::constant(2)
        ));
        let built = context.public_output("selected", selected).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let bound_mismatch = Family::select(
            selector.clone(),
            vec![
                ring.preimage_input_family("bound-left", 2, (2, 3), 7),
                ring.preimage_input_family("bound-right", 2, (2, 3), 8),
            ],
        );
        assert!(matches!(bound_mismatch, Err(DslError::FamilyCountMismatch)));
        let count_mismatch = Family::select(
            selector,
            vec![
                ring.preimage_input_family("count-left", 2, (2, 3), 7),
                ring.preimage_input_family("count-right", 3, (2, 3), 7),
            ],
        );
        assert!(matches!(count_mismatch, Err(DslError::FamilyCountMismatch)));
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
