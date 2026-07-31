use crate::{
    artifact::{ArtifactConfidentiality, ProductionId},
    expr::{IntExpr, RealExpr},
    serde_support,
    types::{NodeId, WireRef, WireType},
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub args: Vec<WireRef>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum NodeKind {
    Input {
        name: String,
        wire_type: WireType,
        artifact: Option<ArtifactInput>,
    },
    Output {
        name: String,
        artifact_confidentiality: Option<ArtifactConfidentiality>,
    },
    ConstantInt(#[serde(with = "serde_support::bigint")] BigInt),
    /// Materializes an integer expression after parameter and loop bindings
    /// have been applied to the current graph instance.
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
        /// Integer tag components appended as their ASCII decimal spelling.
        /// This preserves legacy tags built with `format!("...{index}")` in
        /// bounded loops without unrolling the loop.
        #[serde(default)]
        tag_decimal_expressions: Vec<IntExpr>,
        /// Integer tag components encoded as fixed-width little-endian u64,
        /// matching legacy slot/output namespaces.
        #[serde(default)]
        tag_u64_le_expressions: Vec<IntExpr>,
        base: Option<IntExpr>,
        /// Explicit decomposition width for backend-specific gadget layouts
        /// such as the DCRT small gadget.
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
        /// Explicit decomposition width when it cannot be derived from the
        /// aggregate modulus and base alone.
        #[serde(default)]
        digit_count: Option<IntExpr>,
    },
    ModDown {
        target_modulus: IntExpr,
    },
    ModUp {
        target_modulus: IntExpr,
    },
    ExtractCoefficient {
        position: IntExpr,
    },
    /// Keeps one coefficient of a scalar polynomial as its constant term and
    /// clears every other coefficient.
    ConstantCoefficient {
        position: IntExpr,
    },
    ThresholdDecode {
        plaintext_modulus: IntExpr,
        length: IntExpr,
        output_bool: bool,
    },
    /// Centered-rounds congruent full-modulus representatives at each CRT
    /// level and recombines them with explicit reconstruction coefficients.
    CrtRecompose {
        plaintext_moduli: Vec<IntExpr>,
        reconstruction_coefficients: Vec<IntExpr>,
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
    #[serde(with = "serde_support::bigint")]
    pub minimum: BigInt,
    #[serde(with = "serde_support::bigint")]
    pub maximum: BigInt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct IndexRange {
    pub start: usize,
    pub end: usize,
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
    pub graph: String,
    pub bindings: Vec<(String, IntExpr)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParallelLoop {
    pub graph: String,
    pub count: IntExpr,
    #[serde(default)]
    pub minimum_count: usize,
    pub index_variable: String,
    pub bindings: Vec<(String, IntExpr)>,
    pub input_modes: Vec<LoopInputMode>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LoopInputMode {
    Broadcast,
    Zip,
    ZipOffset { offset: usize },
}
