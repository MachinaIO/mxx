//! Node identities, wires, and wire types.
//!
//! [`WireType`] is the symbolic type of a wire: scalars (`Int`, `Real`, `Bool` and their constant
//! forms), `Bytes`, `TypedBlob`, `Matrix`, `Trapdoor`, bounded `SmallMatrix` and `Preimage`
//! matrices, and `IndexedFamily`. [`ConcreteWireType`] mirrors it with resolved sizes after
//! validation. A [`MatrixType`] refers to a ring, not a single modulus. [`CoefficientBoundDomain`]
//! says whether a bounded coefficient is a single `Global` integer or a signed residue per CRT limb
//! (`PerCrtLimb`).
//!
//! # Index-dependent families
//!
//! Inside an `IndexedFamily` element type, `IntExpr::LoopIndex(MEMBER_INDEX)` is the member's
//! index, so members may have different types (for example matrices over different rings); a
//! parallel loop whose body types depend on its index produces such a family. A family whose
//! members resolve to one type is an ordinary `ConcreteWireType::IndexedFamily`; one whose
//! members differ resolves to `ConcreteWireType::DependentFamily`, member by member.

use crate::{
    expr::{IntExpr, RealExpr},
    ring::{ConcreteRing, RingRef},
    serde_support,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

/// The loop-index slot that denotes a family member's index inside a family element type.
pub const MEMBER_INDEX: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Port(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct WireRef {
    pub node: NodeId,
    pub port: Port,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct InstantiationFrame {
    pub call: NodeId,
    pub loop_index: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct WireId {
    pub instantiation_path: Vec<InstantiationFrame>,
    pub wire: WireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct MatrixType<R = RingRef, D = IntExpr> {
    pub ring: R,
    pub rows: D,
    pub columns: D,
}

pub type ConcreteMatrixType = MatrixType<ConcreteRing, usize>;

/// The domain in which a bounded compact coefficient is interpreted.
/// Per-CRT-limb digits retain one signed residue per ordered CRT tower.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum CoefficientBoundDomain {
    Global,
    PerCrtLimb,
}

/// How a sampled trapdoor relates its public matrix to the gadget matrix.
///
/// For `rows` public rows and `digits` gadget digits per row:
///
/// - `Computational`: `A = [ā | I | G - (ā R + E)]` with `ā` uniform `rows x rows` and `R`, `E`
///   Gaussian `rows x rows·digits`, so `A` has `rows·(digits + 2)` columns and the trapdoor is `[R;
///   E; I]`. Its uniformity rests on the hardness of RLWE with secret `R`.
/// - `Statistical { m_bar }`: `A = [ā | g - ā T]` with `ā` uniform `1 x m_bar` and `T` Gaussian
///   `m_bar x digits`, so `A` has `m_bar + digits` columns and the trapdoor is `[T; I]`. With
///   enough columns `m_bar`, `A` is statistically close to uniform by the regularity lemma. Only
///   one public row is supported.
#[derive(
    Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
pub enum TrapdoorInstantiation {
    #[default]
    Computational,
    Statistical {
        m_bar: usize,
    },
}

impl TrapdoorInstantiation {
    pub fn is_computational(&self) -> bool {
        matches!(self, Self::Computational)
    }

    /// Number of public-matrix columns of a trapdoor with `rows` rows and `digits` gadget digits
    /// per row, or `None` on overflow, on an empty statistical `m_bar`, or for a statistical
    /// trapdoor with other than one row.
    pub fn columns(self, rows: usize, digits: usize) -> Option<usize> {
        match self {
            Self::Computational => rows.checked_mul(digits.checked_add(2)?),
            Self::Statistical { m_bar } if rows == 1 && m_bar > 0 => m_bar.checked_add(digits),
            Self::Statistical { .. } => None,
        }
    }

    /// Symbolic [`Self::columns`] for graph construction.
    pub fn columns_expr(self, rows: IntExpr, digits: IntExpr) -> IntExpr {
        match self {
            Self::Computational => (rows * (digits + IntExpr::constant(2))).canonicalize(),
            Self::Statistical { m_bar } => (IntExpr::constant(m_bar) + digits).canonicalize(),
        }
    }

    /// Number of rows of the top block `T'` of the trapdoor `[T'; I]`: `2·rows` (`[R; E]`) for
    /// `Computational`, `m_bar` (`T`) for `Statistical`.
    pub fn top_rows(self, rows: usize) -> usize {
        match self {
            Self::Computational => 2 * rows,
            Self::Statistical { m_bar } => m_bar,
        }
    }

    /// Recovers the public row count from a trapdoor-width column count.
    pub fn rows_from_columns(self, columns: usize, digits: usize) -> Option<usize> {
        match self {
            Self::Computational => {
                let rows = columns.checked_div(digits.checked_add(2)?)?;
                (rows > 0 && rows * (digits + 2) == columns).then_some(rows)
            }
            Self::Statistical { m_bar } => (columns == m_bar.checked_add(digits)?).then_some(1),
        }
    }
}

impl MatrixType<ConcreteRing, usize> {
    pub fn scalar(ring: ConcreteRing) -> Self {
        Self { ring, rows: 1, columns: 1 }
    }

    pub fn is_scalar(&self) -> bool {
        self.rows == 1 && self.columns == 1
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum WireType {
    ConstantInt,
    ConstantReal,
    ConstantBool,
    Int,
    Real,
    Bool,
    Bytes {
        length: IntExpr,
    },
    TypedBlob {
        type_name: String,
        schema_hash: [u8; 32],
    },
    Matrix(MatrixType),
    /// A sampled or gadget trapdoor. `sigma` is the TrapGen width, `preimage_sigma` the width
    /// every preimage under the trapdoor is sampled with (`None`: the smallest width the trapdoor
    /// sampler accepts), and `preimage_max_coefficient_bound` their common hard cutoff.
    Trapdoor {
        matrix: MatrixType,
        sigma: crate::expr::RealExpr,
        gadget_base: IntExpr,
        digit_count: IntExpr,
        preimage_max_coefficient_bound: IntExpr,
        #[serde(default, skip_serializing_if = "TrapdoorInstantiation::is_computational")]
        instantiation: TrapdoorInstantiation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preimage_sigma: Option<crate::expr::RealExpr>,
    },
    SmallMatrix {
        matrix: MatrixType,
        max_coefficient_bound: IntExpr,
        bound_domain: CoefficientBoundDomain,
    },
    Preimage {
        matrix: MatrixType,
        max_coefficient_bound: IntExpr,
        bound_domain: CoefficientBoundDomain,
    },
    IndexedFamily {
        element: Box<WireType>,
        count: IntExpr,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ConcreteWireType {
    ConstantInt,
    ConstantReal,
    ConstantBool,
    Int,
    Real,
    Bool,
    Bytes {
        length: usize,
    },
    TypedBlob {
        type_name: String,
        schema_hash: [u8; 32],
    },
    Matrix(ConcreteMatrixType),
    Trapdoor {
        matrix: ConcreteMatrixType,
        sigma: RealExpr,
        #[serde(with = "serde_support::bigint")]
        gadget_base: BigInt,
        digit_count: usize,
        #[serde(with = "serde_support::bigint")]
        preimage_max_coefficient_bound: BigInt,
        #[serde(default, skip_serializing_if = "TrapdoorInstantiation::is_computational")]
        instantiation: TrapdoorInstantiation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preimage_sigma: Option<RealExpr>,
    },
    SmallMatrix {
        matrix: ConcreteMatrixType,
        #[serde(with = "serde_support::bigint")]
        max_coefficient_bound: BigInt,
        bound_domain: CoefficientBoundDomain,
    },
    Preimage {
        matrix: ConcreteMatrixType,
        #[serde(with = "serde_support::bigint")]
        max_coefficient_bound: BigInt,
        bound_domain: CoefficientBoundDomain,
    },
    IndexedFamily {
        element: Box<ConcreteWireType>,
        /// Symbolic cardinality; execution additionally requires host-addressable storage.
        /// Validation and Lean export do not allocate one value per member.
        count: u128,
    },
    /// A family whose members have different types (an index-dependent element type).
    DependentFamily {
        members: Vec<ConcreteWireType>,
    },
}

impl MatrixType {
    fn substitute_loop_index(&self, slot: u32, value: &IntExpr) -> Self {
        Self {
            ring: self.ring.substitute_loop_index(slot, value),
            rows: self.rows.substitute_loop_index(slot, value),
            columns: self.columns.substitute_loop_index(slot, value),
        }
    }
}

impl WireType {
    /// This type with every `LoopIndex(slot)` replaced by `value`.
    pub fn substitute_loop_index(&self, slot: u32, value: &IntExpr) -> Self {
        let sub = |expression: &IntExpr| expression.substitute_loop_index(slot, value);
        match self {
            Self::ConstantInt |
            Self::ConstantReal |
            Self::ConstantBool |
            Self::Int |
            Self::Real |
            Self::Bool |
            Self::TypedBlob { .. } => self.clone(),
            Self::Bytes { length } => Self::Bytes { length: sub(length) },
            Self::Matrix(matrix) => Self::Matrix(matrix.substitute_loop_index(slot, value)),
            Self::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                instantiation,
                preimage_sigma,
            } => Self::Trapdoor {
                matrix: matrix.substitute_loop_index(slot, value),
                sigma: sigma.substitute_loop_index(slot, value),
                gadget_base: sub(gadget_base),
                digit_count: sub(digit_count),
                preimage_max_coefficient_bound: sub(preimage_max_coefficient_bound),
                instantiation: *instantiation,
                preimage_sigma: preimage_sigma
                    .as_ref()
                    .map(|sigma| sigma.substitute_loop_index(slot, value)),
            },
            Self::SmallMatrix { matrix, max_coefficient_bound, bound_domain } => {
                Self::SmallMatrix {
                    matrix: matrix.substitute_loop_index(slot, value),
                    max_coefficient_bound: sub(max_coefficient_bound),
                    bound_domain: *bound_domain,
                }
            }
            Self::Preimage { matrix, max_coefficient_bound, bound_domain } => Self::Preimage {
                matrix: matrix.substitute_loop_index(slot, value),
                max_coefficient_bound: sub(max_coefficient_bound),
                bound_domain: *bound_domain,
            },
            Self::IndexedFamily { element, count } => Self::IndexedFamily {
                element: Box::new(element.substitute_loop_index(slot, value)),
                count: sub(count),
            },
        }
    }

    /// The type of member `index` of a family with element type `element`.
    pub fn family_member(element: &Self, index: &IntExpr) -> Self {
        element.substitute_loop_index(MEMBER_INDEX, index)
    }

    /// Whether this type's shape, ring, or bounds depend on a loop index.
    pub fn contains_loop_index(&self) -> bool {
        let matrix = |matrix: &MatrixType| {
            matrix.ring.contains_loop_index() ||
                matrix.rows.contains_loop_index() ||
                matrix.columns.contains_loop_index()
        };
        match self {
            Self::ConstantInt |
            Self::ConstantReal |
            Self::ConstantBool |
            Self::Int |
            Self::Real |
            Self::Bool |
            Self::TypedBlob { .. } => false,
            Self::Bytes { length } => length.contains_loop_index(),
            Self::Matrix(type_) => matrix(type_),
            Self::Trapdoor {
                matrix: type_,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                preimage_sigma,
                ..
            } => {
                matrix(type_) ||
                    sigma.contains_loop_index() ||
                    gadget_base.contains_loop_index() ||
                    digit_count.contains_loop_index() ||
                    preimage_max_coefficient_bound.contains_loop_index() ||
                    preimage_sigma.as_ref().is_some_and(RealExpr::contains_loop_index)
            }
            Self::SmallMatrix { matrix: type_, max_coefficient_bound, .. } |
            Self::Preimage { matrix: type_, max_coefficient_bound, .. } => {
                matrix(type_) || max_coefficient_bound.contains_loop_index()
            }
            Self::IndexedFamily { element, count } => {
                element.contains_loop_index() || count.contains_loop_index()
            }
        }
    }
}

impl ConcreteWireType {
    pub fn matrix_type(&self) -> Option<&ConcreteMatrixType> {
        match self {
            Self::Matrix(matrix) |
            Self::SmallMatrix { matrix, .. } |
            Self::Preimage { matrix, .. } => Some(matrix),
            Self::Trapdoor { matrix, .. } => Some(matrix),
            _ => None,
        }
    }
}
