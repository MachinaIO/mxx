//! Declarative typed construction API for mxx graphs.
//!
//! Executable operations create immutable `mxx-ir-core` nodes immediately.

use mxx_ir_core::{
    CapturePolicy, CompileParameter, CompileParameterKind, FreezeError, Graph, GraphOutput,
    IntExpr, NodeHandle, ParamEnv, RealExpr, SealMap, SubgraphHandle, ValueHandle,
    artifact::{ArtifactConfidentiality, ProductionId},
    graph::with_new_construction_scope,
    node::{
        ArtifactInput, ConstantMatrix, HashTagComponent, HashVariant, IndexRange, MatrixBinaryOp,
        NodeKind, ParallelLoop, SampleRange, SequentialLoop,
    },
    types::{MatrixType, WireType},
};
mod control;
mod family;
mod integer;
mod operators;
pub use integer::{Bool, Bytes, Int};
mod subgraph;
mod value;
pub use control::{iterate, parallel, select};
pub use family::Family;
pub use subgraph::Subgraph;
use value::argument_name;
pub use value::{GraphValue, GraphValueSchema};

use num_bigint::BigUint;
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    ops::{Add, Mul, Neg, Sub},
};
use thiserror::Error;

pub use mxx_ir_core::{
    Rational,
    artifact::ArtifactConfidentiality as Confidentiality,
    protocol::{
        FrozenDerivationAttachment, FrozenDerivationAttachments, FrozenSemanticAnchors, IdealSpec,
        PurePredicateSpec,
    },
};
#[cfg(test)]
mod bundle_tests;
#[cfg(test)]
mod protocol_tests;
#[cfg(test)]
mod test_protocol;

thread_local! {
    /// Lexical loop depth while closure bodies are constructed. Using the depth as the binder
    /// slot is deterministic across builds and keeps nested loop indices distinct.
    static LOOP_BINDER_DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn with_loop_index<T>(body: impl FnOnce(u32) -> T) -> (u32, T) {
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
        let output = body(slot);
        drop(restore);
        (slot, output)
    })
}

#[derive(Debug, Error)]
pub enum DslError {
    #[error("this operation requires an integer index known before execution")]
    CompileTimeIndex,
    #[error(transparent)]
    Freeze(#[from] FreezeError),
    #[error("duplicate output name: {0}")]
    DuplicateOutput(String),
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
    #[error(transparent)]
    Specification(#[from] mxx_ir_core::protocol::SpecificationError),
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
    pub fn preimage_input(
        &self,
        name: impl Into<String>,
        shape: impl IntoShape,
        max_coefficient_bound: impl Into<IntExpr>,
    ) -> Preimage {
        Preimage::source_input(
            name.into(),
            self.matrix_type(shape),
            max_coefficient_bound.into(),
            None,
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
            columns: (rows * (digit_count.clone() + IntExpr::constant(2))).canonicalize(),
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
    ) -> Family<Trapdoor> {
        let count = count.into();
        let rows = rows.into();
        let sigma = sigma.into();
        let gadget_base = gadget_base.into();
        let digit_count = digit_count.into();
        let preimage_max_coefficient_bound = preimage_max_coefficient_bound.into();
        let matrix_type = self.matrix_type(Shape {
            rows: rows.clone(),
            columns: (rows * (digit_count.clone() + IntExpr::constant(2))).canonicalize(),
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
        Family::<Trapdoor>::trapdoor_input(
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
            Shape { rows: rows.clone(), columns: (rows * digit_count).canonicalize() },
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
            vec![bits.values[0].clone()],
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
            (base.clone() - IntExpr::constant(1)).canonicalize()
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
                tag_components: tag.components,
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
                tag_components: tag.components,
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
            columns: (rows * (digit_count.clone() + IntExpr::constant(2))).canonicalize(),
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

/// A fixed caller-chosen namespace prefix followed by ordered, typed components.
/// Keep the raw prefix fixed within a sampling domain; use `push` for variable data.
/// Component framing changes sampled values relative to older grouped tags, so persisted
/// hash-derived artifacts must be rebuilt when adopting this encoding.
#[derive(Clone, Default)]
pub struct HashTag {
    prefix: Vec<u8>,
    components: Vec<HashTagComponent>,
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

    pub fn push_decimal(&mut self, index: impl Into<Int>) -> Result<(), DslError> {
        let index = index.into();
        self.components.push(HashTagComponent::Decimal(
            index.compile_expression().ok_or(DslError::CompileTimeIndex)?,
        ));
        self.pending = Pending::merge([std::mem::take(&mut self.pending), index.pending]);
        Ok(())
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
        tag.components.push(HashTagComponent::Bytes(self.as_bytes().to_vec()));
    }
}

impl HashTagPart for String {
    fn append_to(self, tag: &mut HashTag) {
        self.as_str().append_to(tag);
    }
}

impl HashTagPart for IntExpr {
    fn append_to(self, tag: &mut HashTag) {
        tag.components.push(HashTagComponent::U64Le(self));
    }
}

impl HashTagPart for Int {
    fn append_to(self, tag: &mut HashTag) {
        match self.value.node().kind() {
            NodeKind::EvaluateInt(expression @ IntExpr::LoopIndex(_))
                if self.compile_expression().is_some() =>
            {
                tag.components.push(HashTagComponent::U64Le(expression.clone()));
            }
            _ => {
                tag.components.push(HashTagComponent::Operand(tag.dynamic.len() + 1));
                tag.dynamic.push(self.value);
            }
        }
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

    /// Applies the raw negacyclic automorphism `sigma_k: X -> X^k` entrywise.
    #[track_caller]
    pub fn ring_automorphism(self, index: impl Into<IntExpr>) -> Self {
        let ty = self.matrix_type.clone();
        Self::from_node(NodeKind::RingAutomorphism { index: index.into() }, vec![self], ty)
    }

    /// Rounds each coefficient after scaling by destination/source modulus.
    /// The destination must be an odd divisor of the source modulus.
    #[track_caller]
    pub fn modulus_switch(self, modulus: impl Into<IntExpr>) -> Self {
        let modulus = modulus.into();
        let ty = MatrixType { modulus: modulus.clone(), ..self.matrix_type.clone() };
        Self::from_node(NodeKind::ModulusSwitch { modulus }, vec![self], ty)
    }

    /// Reduces coefficients into a divisor ring without scaling their values.
    #[track_caller]
    pub fn reduce_modulus(self, modulus: impl Into<IntExpr>) -> Self {
        let modulus = modulus.into();
        let ty = MatrixType { modulus: modulus.clone(), ..self.matrix_type.clone() };
        Self::from_node(NodeKind::ModulusReduce { modulus }, vec![self], ty)
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
                |range| (range.end.clone() - range.start.clone()).canonicalize(),
            ),
            columns: columns.as_ref().map_or_else(
                || self.matrix_type.columns.clone(),
                |range| (range.end.clone() - range.start.clone()).canonicalize(),
            ),
            ..self.matrix_type.clone()
        };
        Self::from_node(NodeKind::Slice { rows, columns }, vec![self], ty)
    }

    #[track_caller]
    pub fn tensor(self, rhs: Mat) -> Self {
        let ty = MatrixType {
            rows: (self.matrix_type.rows.clone() * rhs.matrix_type.rows.clone()),
            columns: (self.matrix_type.columns.clone() * rhs.matrix_type.columns.clone()),
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
            rows: (self.matrix_type.rows.clone() * digit_count.clone()).canonicalize(),
            ..self.matrix_type.clone()
        };
        let pending = self.pending;
        let max_coefficient_bound = if small {
            (base.clone() - IntExpr::constant(1)).canonicalize()
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
            bits.extend(
                (0..coefficient_bits)
                    .map(|bit| value.clone().bit(bit))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Family::<Bool>::pack(bits)
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
                .reduce(|left, right| left + right)
                .expect("nonempty"),
            ConcatAxis::Columns => first.rows.clone(),
        };
        let columns = match axis {
            ConcatAxis::Columns | ConcatAxis::Diagonal => values
                .iter()
                .map(|value| value.matrix_type.columns.clone())
                .reduce(|left, right| left + right)
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
        modulus: IntExpr,
    ) -> Mat {
        let mut ty = values.first().expect("CRT recomposition requires inputs").matrix_type.clone();
        ty.modulus = modulus.clone();
        Mat::from_node(
            NodeKind::CrtRecompose { modulus, plaintext_moduli, reconstruction_coefficients },
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

#[derive(Clone)]
#[doc(hidden)]
pub struct DerivationAttachment {
    namespace: String,
    rule: String,
    roles: Vec<(String, ValueHandle)>,
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

    /// Declares a value, including a record or a family of records, as named runtime inputs.
    /// Composite leaves use `name.0`, `name.1`, ... in schema order.
    pub fn input<V: GraphValue>(
        &self,
        name: impl Into<String>,
        schema: V::Schema,
    ) -> Result<V, DslError> {
        let name = name.into();
        let types = schema.wire_types();
        let count = types.len();
        if count == 0 {
            return Err(DslError::Schema);
        }
        let values = types
            .into_iter()
            .enumerate()
            .map(|(port, wire_type)| {
                let name = if count == 1 { name.clone() } else { format!("{name}.{port}") };
                NodeHandle::new(
                    NodeKind::Input { name, wire_type: wire_type.clone(), artifact: None },
                    vec![],
                    vec![wire_type],
                )
                .output(0)
                .expect("input field")
            })
            .collect::<Vec<_>>();
        V::from_values(&schema, &values, Pending::default())
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
            values: vec![node.output(0).expect("integer family input")],
            element_schema: IntType,
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
        trapdoors: Family<Trapdoor>,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(
            name.into(),
            trapdoors.values[1].clone(),
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
        if values.is_empty() {
            return Err(DslError::Schema);
        }
        let names = (0..values.len())
            .map(|port| if values.len() == 1 { name.clone() } else { format!("{name}.{port}") })
            .collect::<Vec<_>>();
        if let Some(name) = names.iter().find(|name| self.outputs.contains_key(*name)) {
            return Err(DslError::DuplicateOutput(name.clone()));
        }
        for (name, value) in names.into_iter().zip(values) {
            self.insert_pending_output(name, value, pending.clone(), confidentiality)?;
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
        self.insert_pending_output(name.into(), family.values[0].clone(), family.pending, None)?;
        Ok(self)
    }

    pub fn bool_family_output(
        mut self,
        name: impl Into<String>,
        family: Family<Bool>,
    ) -> Result<Self, DslError> {
        self.insert_pending_output(name.into(), family.values[0].clone(), family.pending, None)?;
        Ok(self)
    }

    fn insert_family_output(
        &mut self,
        name: String,
        family: Family<Mat>,
        confidentiality: Option<ArtifactConfidentiality>,
    ) -> Result<(), DslError> {
        self.insert_pending_output(name, family.values[0].clone(), family.pending, confidentiality)
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
                anchors: FrozenSemanticAnchors::new(anchors),
                derivation_attachments: FrozenDerivationAttachments::new(attachments),
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
    use mxx_ir_core::node::LoopInputMode;
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
        let small_static = small.at(0);
        let index = DslContext::new("indices").int_family_input("index", 1).at(0);
        let preimage_dynamic = preimage.at(index);
        let mapped_small = parallel(small.count().clone(), |i| Ok(small.at(i))).unwrap();
        let mapped_preimage = parallel(preimage.count().clone(), |i| Ok(preimage.at(i))).unwrap();
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
        assert!(matches!(small.at(0).value_handle().wire_type(), WireType::SmallMatrix { .. }));
        assert!(matches!(preimage.at(0).value_handle().wire_type(), WireType::Preimage { .. }));
        let mut all_nodes = built.graph.scopes().values().flat_map(|scope| scope.nodes());
        assert!(
            all_nodes.clone().any(|node| matches!(node.kind(), NodeKind::FamilyGetStatic { .. }))
        );
        assert!(all_nodes.clone().any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic)));
        assert!(all_nodes.any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_))));
    }

    #[test]
    fn hash_tag_components_preserve_mixed_insertion_order() {
        let ring = Ring::new(17, 8);
        let mut tag = HashTag::from(b"ordered-tag:".as_slice());
        tag.push("before");
        tag.push(IntExpr::constant(3));
        tag.push("middle");
        tag.push_decimal(23).unwrap();
        tag.push(Int::constant(7));
        tag.push("after");
        let sample = ring.hash_matrix(ring.bytes_input("key", 32), tag, (1, 1));
        let built =
            DslContext::new("ordered-tags").output("sample", sample).unwrap().build().unwrap();
        let hash = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::HashSample { .. }))
            .unwrap();
        let NodeKind::HashSample { tag_components, .. } = hash.kind() else { unreachable!() };
        assert_eq!(
            tag_components,
            &vec![
                HashTagComponent::Bytes(b"before".to_vec()),
                HashTagComponent::U64Le(3.into()),
                HashTagComponent::Bytes(b"middle".to_vec()),
                HashTagComponent::Decimal(23.into()),
                HashTagComponent::Operand(1),
                HashTagComponent::Bytes(b"after".to_vec()),
            ]
        );
        built.validate(&ParamEnv::default()).unwrap();
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
        let family = parallel(2, move |_| {
            (captured.clone() + captured.clone()).semantic_anchor("loop-body-sum")
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
        assert!(matches!(
            IdealSpec::new(sampled.graph),
            Err(mxx_ir_core::protocol::SpecificationError::NonPureSpecification)
        ));
    }

    #[test]
    fn scalar_families_gather_through_existing_parallel_loop_nodes() {
        let context = DslContext::new("scalar-family-gather");
        let values = context.int_family_input("values", 3);
        let indices = Family::<Int>::pack(vec![Int::constant(2), Int::constant(0)]).unwrap();
        let gathered = parallel(indices.count().clone(), |i| Ok(values.at(indices.at(i)))).unwrap();
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
    fn indexed_parallel_reads_nonartifact_family_inputs() {
        let ring = Ring::new(97, 4);
        let family = ring.input_family("label-major-input", 6, (1, 2));
        let output = parallel(2, |label| Ok(family.at(label * 3))).unwrap();
        assert_eq!(output.count(), &IntExpr::constant(2));
        let graph = DslContext::new("lexical-family-input")
            .output("labels", output)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        let foreign = with_new_construction_scope(|_| ring.input_family("escaped", 6, (1, 2)));
        assert!(parallel(2, |label| Ok(foreign.at(label * 3))).is_err());
    }

    #[test]
    fn parallel_outputs_normalize_constant_scalar_leaves_before_sealing() {
        let indices = parallel(3, |index| Ok(index)).unwrap();
        let records = parallel(3, |index| {
            Ok((
                Int::constant(7),
                Bool::constant(true),
                (index, vec![Bool::constant(false), Bool::constant(true)]),
            ))
        })
        .unwrap();
        let integers = records.field(|record| record.0).unwrap();
        let booleans = records.field(|record| record.1).unwrap();
        let nested = records.field(|record| record.2).unwrap();
        let mapped = parallel(integers.count().clone(), |_| Ok(Bool::constant(false))).unwrap();
        let zipped = parallel(3, |index| Ok((index, Bool::constant(true)))).unwrap();
        assert_eq!(indices.count(), &IntExpr::constant(3));
        assert_eq!(booleans.count(), &IntExpr::constant(3));
        let built = DslContext::new("parallel-constant-leaves")
            .output("indices", indices)
            .unwrap()
            .output("integers", integers)
            .unwrap()
            .output("booleans", booleans)
            .unwrap()
            .output("nested", nested)
            .unwrap()
            .output("mapped", mapped)
            .unwrap()
            .output("zipped", zipped)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert_eq!(built.graph.outputs().len(), 9);
    }

    #[test]
    fn generated_index_family_gather_has_no_explicit_family_pack() {
        let context = DslContext::new("generated-index-family-gather");
        let values = context.int_family_input("values", 8);
        let indices = parallel(2, |index| Ok(index * 3)).unwrap();
        let gathered = parallel(indices.count().clone(), |i| Ok(values.at(indices.at(i)))).unwrap();
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
        let count = &segments * &bits;
        let context = DslContext::new("parameterized-bit-segments")
            .int_parameter("segments")
            .int_parameter("bits");
        let input = context.int_family_input("input", count);
        let packed = parallel(segments, |segment| {
            let (sum, _) = iterate(
                bits.clone(),
                (Int::constant(0), Int::constant(1)),
                |bit, (sum, weight)| {
                    let value = input.at(segment * bits.clone() + bit);
                    Ok((sum + value * &weight, weight * 2))
                },
            )?;
            Ok(sum)
        })
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
        let trapdoors =
            parallel(count.clone(), |_| Ok(ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))).unwrap();
        let targets = parallel(count.clone(), |_| Ok(ring.zero((1, 1)))).unwrap();
        let preimages = parallel(count, |i| {
            let trapdoor = trapdoors.at(&i);
            Ok(trapdoor
                .sample_preimage(targets.at(i), (trapdoor.public_matrix().matrix_type.columns, 1))
                .mul_small_rhs(trapdoor.public_matrix()))
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
        let trapdoors = parallel(3, |_| Ok(ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))).unwrap();
        let indices = Family::<Int>::pack(vec![Int::constant(2), Int::constant(0)]).unwrap();
        let gathered =
            parallel(indices.count().clone(), |i| Ok(trapdoors.at(indices.at(i)))).unwrap();
        let built = DslContext::new("trapdoor-family-gather")
            .public_output("public", gathered.public_matrices())
            .unwrap()
            .private_trapdoor_family_output("secret", gathered)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let loop_node = built.graph.root_scope().nodes().iter().find(|node| {
            matches!(node.kind(), NodeKind::ParallelLoop(spec) if spec.input_modes.len() == 3)
        }).expect("one gathered trapdoor has two aligned fields and one index");
        let NodeKind::ParallelLoop(spec) = loop_node.kind() else { unreachable!() };
        for (value, mode) in loop_node.arguments().iter().zip(&spec.input_modes) {
            let WireType::IndexedFamily { element, .. } = value.wire_type() else {
                panic!("family argument")
            };
            assert_eq!(
                *mode,
                if matches!(element.as_ref(), WireType::Int) {
                    LoopInputMode::Zip
                } else {
                    LoopInputMode::Broadcast
                }
            );
        }
        let gets = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic))
            .collect::<Vec<_>>();
        assert_eq!(gets.len(), 2);
        assert_eq!(
            gets[0].arguments()[1],
            gets[1].arguments()[1],
            "public and secret use the identical index"
        );
    }

    #[test]
    fn trapdoor_indexing_rejects_a_source_shorter_than_the_iteration_domain() {
        let ring = Ring::new(257, 8);
        let trapdoors = parallel(2, |_| Ok(ring.sample_trapdoor(1, 5, 4, 4, 1_000_000))).unwrap();
        let targets = parallel(3, |_| Ok(ring.zero((1, 1)))).unwrap();
        let output = parallel(targets.count().clone(), |i| {
            let trapdoor = trapdoors.at(&i);
            Ok(trapdoor
                .sample_preimage(targets.at(i), (6, 1))
                .mul_small_rhs(trapdoor.public_matrix()))
        })
        .unwrap();
        let built = DslContext::new("short-trapdoor-source")
            .output("values", output)
            .unwrap()
            .build()
            .unwrap();
        assert!(built.validate(&ParamEnv::default()).is_err());
    }

    #[test]
    fn heterogeneous_parallel_zip_uses_one_loop() {
        let context = DslContext::new("heterogeneous-zip");
        let kinds = context.int_family_input("kinds", 2);
        let left = Family::<Bool>::pack(vec![Bool::constant(false), Bool::constant(true)]).unwrap();
        let right =
            Family::<Bool>::pack(vec![Bool::constant(true), Bool::constant(false)]).unwrap();
        let outputs = parallel(kinds.count().clone(), |i| {
            select(kinds.at(&i), vec![left.at(&i), right.at(i)])
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
    fn indexed_parallel_keeps_shared_sources_and_round_trips() {
        let ring = Ring::new(17, 8);
        let source = ring.input_family("source", 2, (1, 1));
        let shared = ring.input_family("shared", 3, (1, 1));
        let indices = DslContext::new("indices").int_family_input("indices", 2);
        let output = parallel(2, |i| Ok(source.at(&i) + shared.at(indices.at(i)))).unwrap();
        let built = DslContext::new("indexed-with-shared-source")
            .public_output("output", output)
            .unwrap()
            .build()
            .unwrap();
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
            .unwrap();
        assert_eq!(
            loop_spec.input_modes.iter().filter(|mode| **mode == LoopInputMode::Zip).count(),
            2
        );
        assert_eq!(
            loop_spec.input_modes.iter().filter(|mode| **mode == LoopInputMode::Broadcast).count(),
            1
        );
        let encoded = serde_json::to_vec(&built.graph).unwrap();
        let decoded: Graph = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, built.graph);
        mxx_ir_core::validate(&decoded, &ParamEnv::default()).unwrap();
    }

    #[test]
    fn composite_family_pack_rejects_field_count_mismatch() {
        let ring = Ring::new(17, 8);
        let left = ring.input_family("left", 2, (1, 1));
        let right = ring.input_family("right", 3, (1, 1));
        assert!(matches!(Family::pack(vec![left, right]), Err(DslError::Schema)));
    }

    #[test]
    fn heterogeneous_indexing_preserves_bounded_family_types_and_rejects_escaped_sources() {
        let ring = Ring::new(17, 8);
        let context = DslContext::new("heterogeneous-indexing");
        let rows = ring.input_family("rows", 2, (1, 1));
        let indices = context.int_family_input("indices", 2);
        let small = ring.small_matrix_input_family("small", 3, (1, 1), 7);
        let preimages = ring.preimage_input_family("preimages", 4, (1, 1), 11);
        let matrices = ring.input_family("matrices", 5, (1, 1));
        let output = parallel(2, |i| {
            let index = indices.at(&i);
            Ok((
                rows.at(i).mul_small_rhs(small.at(&index)),
                preimages.at(&index),
                matrices.at(index),
            ))
        })
        .unwrap();
        let built = context.public_output("output", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let loop_node = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .unwrap();
        let arguments = loop_node.arguments();
        assert!(arguments.iter().any(|value| matches!(value.wire_type(), WireType::IndexedFamily { element, count } if *count == 3.into() && matches!(element.as_ref(), WireType::SmallMatrix { max_coefficient_bound, .. } if *max_coefficient_bound == 7.into()))));
        assert!(arguments.iter().any(|value| matches!(value.wire_type(), WireType::IndexedFamily { element, count } if *count == 4.into() && matches!(element.as_ref(), WireType::Preimage { max_coefficient_bound, .. } if *max_coefficient_bound == 11.into()))));
        assert!(arguments.iter().any(|value| matches!(value.wire_type(), WireType::IndexedFamily { element, count } if *count == 5.into() && matches!(element.as_ref(), WireType::Matrix(_)))));
        let escaped = with_new_construction_scope(|_| ring.input_family("escaped", 2, (1, 1)));
        assert!(parallel(2, |i| Ok(escaped.at(i))).is_err());
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
            |(matrix, family)| Ok(matrix + family.at(0)),
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
        let final_state = iterate(3, initial, |layer, state| {
            parallel(state.count().clone(), |i| Ok(state.at(i) + layer))
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
        let values = parallel(2, |index| {
            Ok(context.evaluate_int(index.expression()? * IntExpr::Var("width".to_owned()) + 1))
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
        let selector = selectors.at(0);
        let dynamic_int = context.int_family_input("dynamic", 1).at(0);
        let dynamic_bool = dynamic_int.clone().equal(Int::constant(0));

        let all_constant_int =
            select(selector.clone(), vec![Int::constant(3), Int::constant(5)]).unwrap();
        let mixed_int =
            select(selector.clone(), vec![dynamic_int.clone(), Int::constant(7)]).unwrap();
        let all_constant_bool =
            select(selector.clone(), vec![Bool::constant(false), Bool::constant(true)]).unwrap();
        let mixed_bool = select(selector, vec![dynamic_bool, Bool::constant(false)]).unwrap();
        let constant_ints = parallel(2, |_| Ok(Int::constant(11))).unwrap();
        let constant_bools = parallel(2, |_| Ok(Bool::constant(true))).unwrap();

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
        let selector = context.int_family_input("selector", 1).at(0);
        let one = ring.polynomial([IntExpr::constant(1)]);
        let left = Family::pack(vec![ring.zero((1, 1)), one.clone()]).unwrap();
        let right = Family::pack(vec![one, ring.zero((1, 1))]).unwrap();
        let selected = select(selector, vec![left, right]).unwrap();
        let built = context.public_output("selected", selected).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn preimage_family_select_preserves_bound_and_rejects_schema_mismatches() {
        let ring = Ring::new(17, 8);
        let context = DslContext::new("select-preimage-family");
        let selector = context.int_family_input("selector", 1).at(0);
        let left = ring.preimage_input_family("left", 2, (2, 3), 7);
        let right = ring.preimage_input_family("right", 2, (2, 3), 7);
        let selected = select(selector.clone(), vec![left, right]).unwrap();
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

        let bound_mismatch = select(
            selector.clone(),
            vec![
                ring.preimage_input_family("bound-left", 2, (2, 3), 7),
                ring.preimage_input_family("bound-right", 2, (2, 3), 8),
            ],
        );
        assert!(matches!(bound_mismatch, Err(DslError::Schema)));
        let count_mismatch = select(
            selector,
            vec![
                ring.preimage_input_family("count-left", 2, (2, 3), 7),
                ring.preimage_input_family("count-right", 3, (2, 3), 7),
            ],
        );
        assert!(matches!(count_mismatch, Err(DslError::Schema)));
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
