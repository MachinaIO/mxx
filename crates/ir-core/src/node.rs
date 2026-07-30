use crate::{
    artifact::ProductionId,
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
    },
    ConstantInt(#[serde(with = "serde_support::bigint")] BigInt),
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
        base: Option<IntExpr>,
        /// Explicit decomposition width for backend-specific gadget layouts
        /// such as the DCRT small gadget.
        #[serde(default)]
        digit_count: Option<IntExpr>,
    },
    TrapdoorSample {
        matrix_type: crate::types::MatrixType,
        sigma: RealExpr,
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
    ThresholdDecode {
        plaintext_modulus: IntExpr,
        length: IntExpr,
        output_bool: bool,
    },
    SubgraphCall(SubgraphCall),
    ParallelLoop(ParallelLoop),
    Select {
        count: IntExpr,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactInput {
    pub production_id: ProductionId,
    pub artifact_name: String,
    pub family_count: Option<IntExpr>,
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
    pub index_variable: String,
    pub bindings: Vec<(String, IntExpr)>,
}
