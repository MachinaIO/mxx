use crate::{
    artifact::{ArtifactConfidentiality, ProductionId},
    expr::{IndexExpr, IndexMap, IntExpr, RealExpr},
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
        tag_prefix: Vec<u8>,
        #[serde(default)]
        tag_expressions: Vec<IntExpr>,
        #[serde(default)]
        tag_decimal_expressions: Vec<IntExpr>,
        #[serde(default)]
        tag_u64_le_expressions: Vec<IntExpr>,
    },
    TrapdoorSample {
        matrix_type: crate::types::MatrixType,
        sigma: RealExpr,
        gadget_base: IntExpr,
        digit_count: IntExpr,
        preimage_max_coefficient_bound: IntExpr,
    },
    /// Samples a typed matrix `K` for the registered public source `B` and target `T`.
    ///
    /// The node's output is not merely a matrix of the declared shape: its type records the
    /// relation `B * K = T`, with multiplication performed in the ring represented by the
    /// matrix type.  The public source, trapdoor, and target are the three input arguments.
    PreimageSample {
        matrix_type: crate::types::MatrixType,
        max_coefficient_bound: IntExpr,
    },
    /// Consumes a typed relation and computes its ordinary matrix product with the left input.
    ///
    /// Relation-aware analysis is authorized only when the left value carries the matching
    /// source `B` and the right value carries `B * K = T`. Under that condition, the runtime
    /// product is still the ordinary matrix multiplication `B * K`, while analysis may replace it
    /// by `T` and transport the target's noise/carrier. `MatrixBinary::Multiply` has no such
    /// matching-source authorization and therefore cannot consume or transport the relation.
    ApplyPreimage,
    /// Reinterprets a typed preimage as an ordinary matrix only for an exact relation target.
    ///
    /// The runtime value is unchanged (`K -> K`), while the semantic relation is intentionally
    /// discarded.  A noisy target cannot use this escape: validation of the operational result
    /// must reject it, so this node cannot erase the transport of target error.
    MaterializePreimageExact,
    /// Performs relation-preserving algebra on preimages sharing a common left source `B`.
    ///
    /// Each operation computes a new witness `K'` and target `T'` while preserving the equation
    /// `B * K' = T'`; the operation-specific equations are documented by `PreimageBinaryOp`.
    PreimageBinary(PreimageBinaryOp),
    /// Concatenates witnesses with one common left source along matrix columns.
    ///
    /// For witnesses `K_j` satisfying `B * K_j = T_j`, the output is
    /// `K = [K_1 | ... | K_n]` and its target is `T = [T_1 | ... | T_n]`, hence `B * K = T`.
    PreimageConcatColumns,
    /// Samples branch-indexed witnesses for a shared source/trapdoor family.
    ///
    /// For each source branch `i` and final target branch `j`, the output witness `K_i,j` is typed
    /// by `B_i * K_i,j = T_i,j`; the source branch axes are preserved and the target contributes
    /// one final branch axis.
    FamilyPreimageSample {
        matrix_type: crate::types::MatrixType,
        max_coefficient_bound: IntExpr,
    },
    /// Builds the universal gadget relation for an ordinary matrix `T`.
    ///
    /// With digit count `ℓ`, the output witness `K` has `ℓ` times as many rows as `T` and is typed
    /// by `G * K = T`, where `G` is the gadget matrix determined by `base` and `small`.
    GadgetDecompose {
        base: IntExpr,
        small: bool,
        digit_count: IntExpr,
    },
    /// Selects one scalar matrix entry from a gadget witness `K`.
    ///
    /// The selected value is `K[row, column]`; the surrounding decomposition still denotes
    /// `G * K = T`, and the exact-target rule is enforced before this relation is forgotten.
    DecompositionEntry {
        row: IntExpr,
        column: IntExpr,
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
    SequentialLoop(SequentialLoop),
    /// Packs `∏_a n_a` same-typed values into a rank-`r` family with shape `(n_0, ..., n_{r-1})`.
    ///
    /// The flat argument order is row-major: a coordinate `i` maps to flat offset
    /// `sum_a i_a * ∏_{b>a} n_b`.
    FamilyPack {
        shape: Vec<IntExpr>,
    },
    /// Reads the family element at a statically known coordinate `i`, producing `X[i]`.
    FamilyGetStatic {
        indices: Vec<IndexExpr>,
    },
    /// Reads the family element at runtime coordinates `i`, producing `X[i]`.
    FamilyGetDynamic {
        rank: usize,
    },
    /// Selects one axis using a scalar or coordinate-wise selector.
    ///
    /// If `X` has shape `(n_0, ..., n_{r-1})` and axis `a` is selected, the output shape removes
    /// `n_a`; for each remaining coordinate `u`, the result is `X[u with axis a = selector(u)]`.
    FamilySelectAxis {
        axis: usize,
    },
    /// Reindexes a family by a deterministic coordinate map `f`.
    ///
    /// For output coordinate `u`, the resulting element is `Y[u] = X[f(u)]`; the map preserves
    /// the element type while changing the declared family shape.
    FamilyReindex {
        output_shape: Vec<IntExpr>,
        map: IndexMap,
    },
    /// Gathers a family using one integer selector family per input axis.
    ///
    /// For output coordinate `u`, selectors `s_a` define the source coordinate
    /// `f(u) = (s_0[u], ..., s_{r-1}[u])`, so the result is `Y[u] = X[f(u)]`.
    FamilyGather {
        output_shape: Vec<IntExpr>,
        input_rank: usize,
    },
    /// Executes one body per coordinate of a rank-`r` Cartesian grid.
    ///
    /// The body is evaluated at `u ∈ ∏_a [0, n_a)`, and its output is stored as the family
    /// element `Y[u]`; `shape` therefore determines exactly the output index set.
    ParallelGrid(ParallelGrid),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreimageBinaryOp {
    /// Adds witnesses and targets componentwise: `B*K' = (B*K_1) + (B*K_2) = T_1 + T_2`.
    Add,
    /// Right-multiplies by exact `A`: `B*(K*A) = (B*K)*A = T*A`.
    RightMultiplyExact,
    /// Composes with `G*L = U`: `B*(K*L) = (B*K)*L = T*L`.
    ComposeExactDecomposition,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GridInputMode {
    /// Uses the same input value at every grid coordinate: `X_u = X`.
    Broadcast,
    /// Uses `X_u = X[f(u)]`, where `f` maps output coordinates to input coordinates.
    Reindex { map: IndexMap },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParallelGrid {
    /// Extents `(n_0, ..., n_{r-1})` of the output coordinate set `∏_a [0, n_a)`.
    pub shape: Vec<IntExpr>,
    /// Loop slots carrying the current coordinate components `u_a` into the body.
    pub index_slots: Vec<u32>,
    /// Compile-time bindings used to instantiate expressions inside the body.
    pub bindings: Vec<(String, IntExpr)>,
    /// Per-input equation selecting broadcast or coordinate-reindexed transport.
    pub input_modes: Vec<GridInputMode>,
}
