use crate::{
    artifact::{ArtifactConfidentiality, ProductionId},
    expr::{IntExpr, RealExpr},
    types::WireType,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

/// Executable operation represented by a declarative graph node.
///
/// Node identity, arguments, output types, and structural child definitions
/// live on `GraphNode`; this enum contains operation semantics only.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum NodeKind {
    Input {
        name: String,
        wire_type: WireType,
        artifact: Option<ArtifactInput>,
    },
    ConstantInt(#[serde(with = "crate::serde_support::bigint")] BigInt),
    EvaluateInt(IntExpr),
    ConstantReal(RealExpr),
    ConstantBool(bool),
    ConstantMatrix {
        matrix_type: crate::types::MatrixType,
        value: ConstantMatrix,
    },
    GadgetTrapdoor {
        matrix_type: crate::types::MatrixType,
        base: IntExpr,
    },
    TrapdoorPublic,
    IntBinary(IntBinaryOp),
    IntCompare(IntCompareOp),
    BitExtract {
        bit: IntExpr,
    },
    IntToReal,
    BoolToInt,
    RealBinary(RealBinaryOp),
    RealSqrt,
    MatrixBinary(MatrixBinaryOp),
    MatrixNegate,
    MatrixScale {
        scalar: IntExpr,
    },
    Transpose,
    Slice {
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
    },
    Tensor,
    Concat {
        axis: ConcatAxis,
    },
    Reshape {
        rows: IntExpr,
        columns: IntExpr,
    },
    UniformSample {
        matrix_type: crate::types::MatrixType,
        range: SampleRange,
    },
    GaussianSample {
        matrix_type: crate::types::MatrixType,
        sigma: RealExpr,
    },
    HashSample {
        matrix_type: crate::types::MatrixType,
        variant: HashVariant,
        tag_prefix: Vec<u8>,
        #[serde(default)]
        tag_expressions: Vec<IntExpr>,
        #[serde(default)]
        tag_decimal_expressions: Vec<IntExpr>,
        #[serde(default)]
        tag_u64_le_expressions: Vec<IntExpr>,
        base: Option<IntExpr>,
        #[serde(default)]
        digit_count: Option<IntExpr>,
    },
    TrapdoorSample {
        matrix_type: crate::types::MatrixType,
        sigma: RealExpr,
        gadget_base: IntExpr,
        digit_count: IntExpr,
    },
    PreimageSample {
        matrix_type: crate::types::MatrixType,
    },
    GadgetDecompose {
        base: IntExpr,
        small: bool,
        #[serde(default)]
        digit_count: Option<IntExpr>,
    },
    ExtractCoefficient {
        position: IntExpr,
    },
    ConstantCoefficient {
        position: IntExpr,
    },
    ThresholdDecode {
        plaintext_modulus: IntExpr,
        length: IntExpr,
        output_bool: bool,
    },
    CrtRecompose {
        plaintext_moduli: Vec<IntExpr>,
        reconstruction_coefficients: Vec<IntExpr>,
    },
    /// Reconstructs one polynomial from canonical coefficient bits.
    ///
    /// The input is a fixed-length boolean family ordered coefficient-major
    /// and little-endian within each coefficient.
    PackPolynomialCoefficients {
        matrix_type: crate::types::MatrixType,
        coefficient_bits: IntExpr,
    },
    SubgraphCall(SubgraphCall),
    ParallelLoop(ParallelLoop),
    FamilyPack {
        count: IntExpr,
    },
    FamilyGetStatic {
        index: IntExpr,
    },
    FamilyGetDynamic,
    Select {
        count: IntExpr,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactInput {
    pub production_id: ProductionId,
    pub artifact_name: String,
    pub confidentiality: ArtifactConfidentiality,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ConstantMatrix {
    Zero,
    Identity,
    UnitRow { index: IntExpr },
    UnitColumn { index: IntExpr },
    Gadget { base: IntExpr, small: bool },
    PowerOfBase { base: IntExpr, exponent: IntExpr },
    Rotation { exponent: IntExpr },
    Polynomial { coefficients: Vec<IntExpr> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IntBinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IntCompareOp {
    Equal,
    Less,
    LessEqual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RealBinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MatrixBinaryOp {
    Add,
    Subtract,
    Multiply,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SampleRange {
    pub minimum: IntExpr,
    pub maximum: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct IndexRange {
    pub start: IntExpr,
    pub end: IntExpr,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ConcatAxis {
    Rows,
    Columns,
    Diagonal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HashVariant {
    Plain,
    Decomposed,
    SmallDecomposed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubgraphCall {
    pub definition: String,
    pub bindings: Vec<(String, IntExpr)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParallelLoop {
    pub count: IntExpr,
    #[serde(default)]
    pub minimum_count: usize,
    pub index_slot: u32,
    pub bindings: Vec<(String, IntExpr)>,
    pub input_modes: Vec<LoopInputMode>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LoopInputMode {
    Broadcast,
    Zip,
    ZipOffset { offset: usize },
}
