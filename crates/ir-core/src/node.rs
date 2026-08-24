use crate::{
    artifact::{ArtifactConfidentiality, ProductionId},
    expr::{IntExpr, RealExpr},
    types::WireType,
};
use num_bigint::{BigInt, BigUint};
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
    /// Computes `bias + sum(coefficients[t] * left[t] * right[t])`.
    /// This is an execution fusion; its semantics are ordinary multiply,
    /// integer scale, and add operations.
    MatrixMulAccumulate {
        coefficients: Vec<IntExpr>,
        has_bias: bool,
    },
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
    /// Samples every coefficient uniformly from the full residue ring `R_q`.
    ///
    /// The modulus belongs to `matrix_type`, so this operation remains meaningful
    /// before a concrete parameter environment is selected.
    UniformResidueSample {
        matrix_type: crate::types::MatrixType,
    },
    /// Samples every coefficient from an explicit integer interval.
    UniformIntervalSample {
        matrix_type: crate::types::MatrixType,
        range: SampleRange,
    },
    GaussianSample {
        matrix_type: crate::types::MatrixType,
        sigma: RealExpr,
        max_coefficient_bound: IntExpr,
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
        preimage_max_coefficient_bound: IntExpr,
    },
    PreimageSample {
        matrix_type: crate::types::MatrixType,
        max_coefficient_bound: IntExpr,
    },
    GadgetDecompose {
        base: IntExpr,
        small: bool,
        digit_count: IntExpr,
    },
    ExtractCoefficient {
        position: IntExpr,
        /// Compile-time-only exclusive upper bound for a canonical input.
        canonical_input_exclusive_upper: Option<BigUint>,
    },
    /// Lifts an integer into the constant coefficient of a scalar polynomial.
    LiftIntegerToConstantPolynomial {
        matrix_type: crate::types::MatrixType,
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
    SequentialLoop(SequentialLoop),
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
    /// Per-argument canonical coefficient exclusive upper bounds.
    ///
    /// A `Some(U)` states that the corresponding argument is a constant
    /// polynomial whose canonical coefficient is in `0..U`.  It is an
    /// authoritative producer contract, rather than a value observed while
    /// executing the graph.  Every call argument, including synthetic
    /// constants, has one entry; an argument without this contract is `None`.
    pub canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
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

/// A structural loop whose body consumes and returns a carried state.
///
/// Arguments are ordered as the initial carried values followed by loop-invariant values. The
/// body receives values in the same order and returns exactly `carried_count` values. Iteration
/// outputs replace the carried inputs for the next iteration; the node exposes the final state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SequentialLoop {
    pub count: IntExpr,
    pub index_slot: u32,
    pub bindings: Vec<(String, IntExpr)>,
    pub carried_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LoopInputMode {
    Broadcast,
    Zip,
    ZipOffset { offset: usize },
}
