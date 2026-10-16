//! The executable node vocabulary.
//!
//! [`NodeKind`] is the complete set of operations an executor must implement:
//!
//! | Category | Variants |
//! | --- | --- |
//! | Inputs and constants | `Input`, `ConstantInt`, `EvaluateInt`, `Sha256BitColumns`, `ConstantReal`, `ConstantBool`, `ConstantMatrix` |
//! | Trapdoor structure | `GadgetTrapdoor`, `TrapdoorPublic` |
//! | Scalar arithmetic | `IntBinary`, `IntCompare`, `BitExtract`, `IntToReal`, `BoolToInt`, `RealBinary`, `RealSqrt`, `IntMatrixVectorProduct` |
//! | Matrix arithmetic | `MatrixBinary`, `MatrixMulAccumulate`, `MatrixMulSmallRhs`, `MatrixNegate`, `MatrixScale`, `RingAutomorphism`, `MultiplyMonomial` |
//! | Ring and modulus conversion | `ModulusSwitch`, `ModulusReduce`, `CenteredRebase`, `CenteredRoundDivide`, `RnsModUp`, `RnsModDown`, `BlockModSwitch` |
//! | Shape | `Transpose`, `Slice`, `Tensor`, `BlockDiagonal`, `Concat` |
//! | Samplers | `UniformResidueSample`, `UniformIntervalSample`, `GaussianSample`, `HashSample`, `HashIntFamily`, `TrapdoorSample`, `PreimageSample` |
//! | Decomposition and coefficients | `GadgetDecompose`, `ExtractCoefficient`, `LiftIntegerToConstantPolynomial`, `PackPolynomialCoefficients`, `PolynomialFromValues`, `PolynomialValues` |
//! | Decoding and CRT | `ThresholdDecode`, `CrtRecompose` |
//! | Control | `SubgraphCall`, `ParallelLoop`, `SequentialLoop`, `Select` |
//! | Families | `FamilyPack`, `FamilyGetStatic`, `FamilyGetDynamic` |
//!
//! Hash-tag components are typed ([`HashTagComponent`]), so different framings cannot collide.
//! `HashIntFamily` returns `count` integers uniform on `[0, modulus)` for a power-of-two modulus;
//! integer `i` is coefficient `i` of entry `(0, 0)` of the `HashSample` transcript, truncated to
//! `log2(modulus)` bits, so no candidate is rejected.
//!
//! The ring-conversion nodes are fused CRT operations with explicit destination rings, not a
//! generic implicit modulus-changing mechanism. `MultiplyMonomial` multiplies every entry by `X^k`
//! for a runtime `k` of any sign taken modulo `2n`.
//!
//! Control nodes carry their own metadata. A [`ParallelLoop`] runs independent instances over
//! `0..count`, each argument with a [`LoopInputMode`] (`Broadcast`, `Zip`, `ZipOffset`,
//! `ZipQuotient`, or `Gather` by another argument's instance value), and returns indexed families.
//! In a [`SequentialLoop`] the first `carried_count` arguments are the carried state and the rest
//! are captures. `Select` eagerly selects one candidate by a runtime selector; both candidates are
//! part of the graph, so selection is not lazy branching.

use crate::{
    artifact::{ArtifactAvailability, ProductionId},
    expr::{IntExpr, RealExpr},
    ring::RingRef,
    types::WireType,
};
use num_bigint::{BigInt, BigUint};
use serde::{Deserialize, Serialize};

/// Ordered components after the caller's explicit hash-tag namespace prefix.
/// Byte strings and decimal integers are length-framed; integer encodings have type markers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HashTagComponent {
    Bytes(Vec<u8>),
    Integer(IntExpr),
    Decimal(IntExpr),
    U64Le(IntExpr),
    /// Index into hash sampler arguments, including the key at index zero.
    Operand(usize),
    /// Index into hash sampler arguments of a runtime integer scalar, encoded like
    /// [`HashTagComponent::U64Le`] of its value (`[3]` followed by the value as eight
    /// little-endian bytes); the value must lie in `[0, 2^64)` at run time.
    OperandU64Le(usize),
}

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
    /// Public SHA-256 bit-matrix columns, packed ascending row indices and counts.
    /// See [`crate::hash_columns::sha256_bit_columns`] for the exact stream convention.
    Sha256BitColumns {
        seed: [u8; 32],
        rows: usize,
        columns: usize,
        index_bits: usize,
    },
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
    /// Ordinary matrix multiplication with a bounded compact right-hand side.
    MatrixMulSmallRhs,
    MatrixNegate,
    MatrixScale {
        scalar: IntExpr,
    },
    /// Applies the raw negacyclic ring automorphism `sigma_k: X -> X^k` entrywise.
    /// Concrete validation requires `0 < k < 2n` and odd `k`.
    RingAutomorphism {
        index: IntExpr,
    },
    /// Multiplies every entry by `X^k` in the negacyclic ring, for the
    /// runtime integer `k` of argument 1 (any sign, taken modulo `2n`).
    MultiplyMonomial,
    /// Coefficientwise exact nearest scaling into a divisor ring.
    ModulusSwitch {
        destination: RingRef,
    },
    /// Ordinary coefficient-ring reduction, preserving small integer values.
    ModulusReduce {
        destination: RingRef,
    },
    /// Re-encodes the centered coefficients of a source CRT basis in another ring.
    CenteredRebase {
        destination: RingRef,
    },
    /// Divides centered coefficients by a positive compile-time divisor and rounds
    /// to the nearest integer, preserving the matrix ring and shape.
    CenteredRoundDivide {
        divisor: IntExpr,
    },
    /// Fused centered CRT digit extension into a multiple modulus.
    RnsModUp {
        destination: RingRef,
        digit_size: usize,
        normalize: bool,
    },
    /// Fused BGV plaintext-preserving removal of the auxiliary CRT basis.
    RnsModDown {
        destination: RingRef,
        plaintext_modulus: IntExpr,
    },
    /// Exact block CRT modulus switching. `source_moduli` is the complete
    /// source basis; `modulus` is the product of a strict non-empty subset.
    /// The plaintext modulus is the positive correction factor `t`.
    BlockModSwitch {
        destination: RingRef,
        plaintext_modulus: IntExpr,
    },
    Transpose,
    Slice {
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
    },
    Tensor,
    /// The Kronecker product `I_count ⊗ input`, with a positive repetition count.
    /// One source and one result represent all blocks without concat intermediates.
    BlockDiagonal {
        count: usize,
    },
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
    /// Samples a discrete Gaussian of width `sigma` truncated at `max_coefficient_bound`. With
    /// `compact`, the output is a globally bounded `SmallMatrix` whose bound is the cutoff, so an
    /// export stores it as compact signed coefficients; otherwise it is a `Matrix`.
    GaussianSample {
        matrix_type: crate::types::MatrixType,
        sigma: RealExpr,
        max_coefficient_bound: IntExpr,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        compact: bool,
    },
    HashSample {
        matrix_type: crate::types::MatrixType,
        variant: HashVariant,
        tag_prefix: Vec<u8>,
        tag_components: Vec<HashTagComponent>,
        base: Option<IntExpr>,
        #[serde(default)]
        digit_count: Option<IntExpr>,
    },
    /// `count` integers uniform on `[0, modulus)` for a power-of-two
    /// `modulus`, keyed like `HashSample`: integer `i` is the first
    /// `log2(modulus)` bits of the digest stream of coefficient `i` of entry
    /// `(0, 0)`, so no candidate is ever rejected.
    HashIntFamily {
        count: IntExpr,
        modulus: IntExpr,
        tag_prefix: Vec<u8>,
        tag_components: Vec<HashTagComponent>,
    },
    TrapdoorSample {
        matrix_type: crate::types::MatrixType,
        sigma: RealExpr,
        gadget_base: IntExpr,
        digit_count: IntExpr,
        preimage_max_coefficient_bound: IntExpr,
        #[serde(
            default,
            skip_serializing_if = "crate::types::TrapdoorInstantiation::is_computational"
        )]
        instantiation: crate::types::TrapdoorInstantiation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preimage_sigma: Option<RealExpr>,
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
    /// Decodes coefficients `0..length` of a scalar polynomial to
    /// `round(t x / q) mod t`: `length` ports, or with `family` one integer
    /// family of `length` members.
    ThresholdDecode {
        plaintext_modulus: IntExpr,
        length: IntExpr,
        output_bool: bool,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        family: bool,
    },
    CrtRecompose {
        modulus: IntExpr,
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
    /// Imports runtime coefficients or native evaluation slots into a scalar polynomial.
    PolynomialFromValues {
        matrix_type: crate::types::MatrixType,
        evaluation: bool,
    },
    /// Exports a scalar polynomial as canonical coefficients or native evaluation slots.
    PolynomialValues {
        evaluation: bool,
    },
    /// The integer product of a row-major matrix family (argument 0) with
    /// a vector family (argument 1). The vector length fixes the inner
    /// dimension: `M v` with `out[i] = sum_j M[i, j] v[j]`, or with
    /// `transpose` `v^T M` with `out[j] = sum_i v[i] M[i, j]`.
    IntMatrixVectorProduct {
        transpose: bool,
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
    pub availability: ArtifactAvailability,
}

/// A constant matrix payload. `AllCoefficients` makes every entry the
/// polynomial whose `ring_dimension` coefficients all equal `value`, so the
/// graph stays independent of the ring dimension.
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
    AllCoefficients { value: IntExpr },
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum LoopInputMode {
    Broadcast,
    Zip,
    ZipOffset {
        offset: usize,
    },
    /// Instance `i` receives the member of this family whose index is the
    /// instance's value of argument `index_argument`, a `Zip` or `ZipOffset`
    /// integer family. The member keys then depend only on values from outside
    /// the loop and the instance index, so a backend can read members ahead
    /// of the instances that consume them.
    Gather {
        index_argument: usize,
    },
    /// Instance `i` receives member `floor(i / divisor)` (`divisor >= 1`): consecutive groups of
    /// `divisor` instances share one member, e.g. the parts of one chain reading the chain's
    /// value.
    ZipQuotient {
        divisor: usize,
    },
}

impl LoopInputMode {
    /// The member a zipped mode (`Zip`, `ZipOffset`, `ZipQuotient`) gives instance `instance`;
    /// `None` for `Broadcast` and `Gather` (or a zero divisor). The index type can be `usize`
    /// for execution or `u128` for symbolic cardinality validation; arithmetic stays checked.
    pub fn zipped_member<Index>(self, instance: Index) -> Option<Index>
    where
        Index: num_traits::CheckedAdd + num_traits::CheckedDiv + TryFrom<usize>,
    {
        match self {
            Self::Zip => Some(instance),
            Self::ZipOffset { offset } => instance.checked_add(&Index::try_from(offset).ok()?),
            Self::ZipQuotient { divisor } => instance.checked_div(&Index::try_from(divisor).ok()?),
            Self::Broadcast | Self::Gather { .. } => None,
        }
    }
}
