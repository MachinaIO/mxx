//! The compact expression language used by the operational-noise checker.
//!
//! `MxxLang` is deliberately an analysis language, rather than another Graph
//! IR. Canonical coefficient metadata belongs directly to the checker
//! `ExtractCoefficient` term; no checker-only matrix operation is introduced.

use crate::operational_noise::identity::{
    AtomicSourceId, Axis, BinderId, CrtSpecId, HashQuerySpecId, MatrixConstantSpecId,
    ResolvedIntExpr, ResolvedMatrixType, SliceSpecId,
};
use egg::{FromOp, Id, Language};
use num_bigint::{BigInt, BigUint};
use std::fmt;

/// Compact, typed expression nodes for a single operational-checker job.
///
/// Attribute values have already been owner-resolved before they enter this
/// language.  Runtime values are children, so egg congruence compares their
/// canonical e-classes instead of incidental Graph IR node identities.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MxxLang {
    Atom {
        source: AtomicSourceId,
        indices: Box<[Id]>,
    },

    IntConst(BigInt),
    IntParameter(String),
    IntBinder(BinderId),

    IntAdd([Id; 2]),
    IntSub([Id; 2]),
    IntMul([Id; 2]),
    IntExactDiv([Id; 2]),
    IntEuclideanDiv([Id; 2]),
    IntEuclideanRemainder([Id; 2]),
    IntRoundDiv([Id; 2]),
    IntLog2Ceil([Id; 1]),

    BoolConst(bool),
    IntEqual([Id; 2]),
    IntLess([Id; 2]),
    IntLessEqual([Id; 2]),
    BitExtract {
        bit: ResolvedIntExpr,
        input: [Id; 1],
    },
    BoolToInt([Id; 1]),

    /// The exact `f64::to_bits()` representation of a real constant.
    RealConst(u64),
    IntToReal([Id; 1]),
    RealAdd([Id; 2]),
    RealSub([Id; 2]),
    RealMul([Id; 2]),
    RealDiv([Id; 2]),
    RealSqrt([Id; 1]),

    MatrixConstant(MatrixConstantSpecId),

    HashPlain {
        query: HashQuerySpecId,
        arguments: Box<[Id]>,
    },

    MatrixAdd(Box<[Id]>),
    MatrixMultiply(Box<[Id]>),
    MatrixNegate([Id; 1]),
    MatrixScale([Id; 2]),
    MatrixTranspose([Id; 1]),

    MatrixSlice {
        spec: SliceSpecId,
        input: [Id; 1],
    },
    MatrixTensor([Id; 2]),
    MatrixConcat {
        axis: Axis,
        inputs: Box<[Id]>,
    },

    /// Ordered as `selector, case0, case1, ...`.
    Switch(Box<[Id]>),
    ExtractCoefficient {
        canonical_exclusive_upper: Option<BigUint>,
        input: [Id; 2],
    },
    LiftConstantPolynomial {
        matrix_type: ResolvedMatrixType,
        input: [Id; 1],
    },
    CrtRecompose {
        spec: CrtSpecId,
        inputs: Box<[Id]>,
    },
    PackPolynomialCoefficients {
        matrix_type: ResolvedMatrixType,
        coefficient_bits: ResolvedIntExpr,
        bits: Box<[Id]>,
    },
}

impl MxxLang {
    /// Returns the operation spelling used only for diagnostics and egg dumps.
    pub const fn operator_name(&self) -> &'static str {
        match self {
            Self::Atom { .. } => "atom",
            Self::IntConst(_) => "int-const",
            Self::IntParameter(_) => "int-parameter",
            Self::IntBinder(_) => "int-binder",
            Self::IntAdd(_) => "int-add",
            Self::IntSub(_) => "int-sub",
            Self::IntMul(_) => "int-mul",
            Self::IntExactDiv(_) => "int-exact-div",
            Self::IntEuclideanDiv(_) => "int-euclidean-div",
            Self::IntEuclideanRemainder(_) => "int-euclidean-remainder",
            Self::IntRoundDiv(_) => "int-round-div",
            Self::IntLog2Ceil(_) => "int-log2-ceil",
            Self::BoolConst(_) => "bool-const",
            Self::IntEqual(_) => "int-equal",
            Self::IntLess(_) => "int-less",
            Self::IntLessEqual(_) => "int-less-equal",
            Self::BitExtract { .. } => "bit-extract",
            Self::BoolToInt(_) => "bool-to-int",
            Self::RealConst(_) => "real-const",
            Self::IntToReal(_) => "int-to-real",
            Self::RealAdd(_) => "real-add",
            Self::RealSub(_) => "real-sub",
            Self::RealMul(_) => "real-mul",
            Self::RealDiv(_) => "real-div",
            Self::RealSqrt(_) => "real-sqrt",
            Self::MatrixConstant(_) => "matrix-constant",
            Self::HashPlain { .. } => "hash-plain",
            Self::MatrixAdd(_) => "matrix-add",
            Self::MatrixMultiply(_) => "matrix-multiply",
            Self::MatrixNegate(_) => "matrix-negate",
            Self::MatrixScale(_) => "matrix-scale",
            Self::MatrixTranspose(_) => "matrix-transpose",
            Self::MatrixSlice { .. } => "matrix-slice",
            Self::MatrixTensor(_) => "matrix-tensor",
            Self::MatrixConcat { .. } => "matrix-concat",
            Self::Switch(_) => "switch",
            Self::ExtractCoefficient { .. } => "extract-coefficient",
            Self::LiftConstantPolynomial { .. } => "lift-constant-polynomial",
            Self::CrtRecompose { .. } => "crt-recompose",
            Self::PackPolynomialCoefficients { .. } => "pack-polynomial-coefficients",
        }
    }
}

impl Language for MxxLang {
    type Discriminant = std::mem::Discriminant<Self>;

    fn discriminant(&self) -> Self::Discriminant {
        std::mem::discriminant(self)
    }

    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Atom { source: left_source, indices: left_indices },
                Self::Atom { source: right_source, indices: right_indices },
            ) => left_source == right_source && left_indices.len() == right_indices.len(),
            (Self::IntConst(left), Self::IntConst(right)) => left == right,
            (Self::IntParameter(left), Self::IntParameter(right)) => left == right,
            (Self::IntBinder(left), Self::IntBinder(right)) => left == right,
            (Self::BoolConst(left), Self::BoolConst(right)) => left == right,
            (Self::RealConst(left), Self::RealConst(right)) => left == right,
            (Self::MatrixConstant(left), Self::MatrixConstant(right)) => left == right,
            (
                Self::ExtractCoefficient { canonical_exclusive_upper: left_upper, .. },
                Self::ExtractCoefficient { canonical_exclusive_upper: right_upper, .. },
            ) => left_upper == right_upper,
            (Self::BitExtract { bit: left_bit, .. }, Self::BitExtract { bit: right_bit, .. }) => {
                left_bit == right_bit
            }
            (
                Self::HashPlain { query: left_query, arguments: left_arguments },
                Self::HashPlain { query: right_query, arguments: right_arguments },
            ) => left_query == right_query && left_arguments.len() == right_arguments.len(),
            (Self::MatrixAdd(left), Self::MatrixAdd(right)) |
            (Self::MatrixMultiply(left), Self::MatrixMultiply(right)) |
            (Self::Switch(left), Self::Switch(right)) => left.len() == right.len(),
            (
                Self::MatrixSlice { spec: left_spec, .. },
                Self::MatrixSlice { spec: right_spec, .. },
            ) => left_spec == right_spec,
            (
                Self::MatrixConcat { axis: left_axis, inputs: left_inputs },
                Self::MatrixConcat { axis: right_axis, inputs: right_inputs },
            ) => left_axis == right_axis && left_inputs.len() == right_inputs.len(),
            (
                Self::LiftConstantPolynomial { matrix_type: left_matrix_type, .. },
                Self::LiftConstantPolynomial { matrix_type: right_matrix_type, .. },
            ) => left_matrix_type == right_matrix_type,
            (
                Self::CrtRecompose { spec: left_spec, inputs: left_inputs },
                Self::CrtRecompose { spec: right_spec, inputs: right_inputs },
            ) => left_spec == right_spec && left_inputs.len() == right_inputs.len(),
            (
                Self::PackPolynomialCoefficients {
                    matrix_type: left_matrix_type,
                    coefficient_bits: left_coefficient_bits,
                    bits: left_bits,
                },
                Self::PackPolynomialCoefficients {
                    matrix_type: right_matrix_type,
                    coefficient_bits: right_coefficient_bits,
                    bits: right_bits,
                },
            ) => {
                left_matrix_type == right_matrix_type &&
                    left_coefficient_bits == right_coefficient_bits &&
                    left_bits.len() == right_bits.len()
            }
            (Self::IntAdd(_), Self::IntAdd(_)) |
            (Self::IntSub(_), Self::IntSub(_)) |
            (Self::IntMul(_), Self::IntMul(_)) |
            (Self::IntExactDiv(_), Self::IntExactDiv(_)) |
            (Self::IntEuclideanDiv(_), Self::IntEuclideanDiv(_)) |
            (Self::IntEuclideanRemainder(_), Self::IntEuclideanRemainder(_)) |
            (Self::IntRoundDiv(_), Self::IntRoundDiv(_)) |
            (Self::IntLog2Ceil(_), Self::IntLog2Ceil(_)) |
            (Self::IntEqual(_), Self::IntEqual(_)) |
            (Self::IntLess(_), Self::IntLess(_)) |
            (Self::IntLessEqual(_), Self::IntLessEqual(_)) |
            (Self::BoolToInt(_), Self::BoolToInt(_)) |
            (Self::IntToReal(_), Self::IntToReal(_)) |
            (Self::RealAdd(_), Self::RealAdd(_)) |
            (Self::RealSub(_), Self::RealSub(_)) |
            (Self::RealMul(_), Self::RealMul(_)) |
            (Self::RealDiv(_), Self::RealDiv(_)) |
            (Self::RealSqrt(_), Self::RealSqrt(_)) |
            (Self::MatrixNegate(_), Self::MatrixNegate(_)) |
            (Self::MatrixScale(_), Self::MatrixScale(_)) |
            (Self::MatrixTranspose(_), Self::MatrixTranspose(_)) |
            (Self::MatrixTensor(_), Self::MatrixTensor(_)) => true,
            _ => false,
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Atom { indices, .. } |
            Self::HashPlain { arguments: indices, .. } |
            Self::MatrixAdd(indices) |
            Self::MatrixMultiply(indices) |
            Self::Switch(indices) |
            Self::CrtRecompose { inputs: indices, .. } |
            Self::PackPolynomialCoefficients { bits: indices, .. } |
            Self::MatrixConcat { inputs: indices, .. } => indices,
            Self::IntAdd(children) |
            Self::IntSub(children) |
            Self::IntMul(children) |
            Self::IntExactDiv(children) |
            Self::IntEuclideanDiv(children) |
            Self::IntEuclideanRemainder(children) |
            Self::IntRoundDiv(children) |
            Self::IntEqual(children) |
            Self::IntLess(children) |
            Self::IntLessEqual(children) |
            Self::RealAdd(children) |
            Self::RealSub(children) |
            Self::RealMul(children) |
            Self::RealDiv(children) |
            Self::MatrixScale(children) |
            Self::MatrixTensor(children) => children,
            Self::ExtractCoefficient { input, .. } => input,
            Self::IntLog2Ceil(children) |
            Self::BoolToInt(children) |
            Self::IntToReal(children) |
            Self::RealSqrt(children) |
            Self::MatrixNegate(children) |
            Self::MatrixTranspose(children) => children,
            Self::BitExtract { input, .. } |
            Self::MatrixSlice { input, .. } |
            Self::LiftConstantPolynomial { input, .. } => input,
            Self::IntConst(_) |
            Self::IntParameter(_) |
            Self::IntBinder(_) |
            Self::BoolConst(_) |
            Self::RealConst(_) |
            Self::MatrixConstant(_) => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Atom { indices, .. } |
            Self::HashPlain { arguments: indices, .. } |
            Self::MatrixAdd(indices) |
            Self::MatrixMultiply(indices) |
            Self::Switch(indices) |
            Self::CrtRecompose { inputs: indices, .. } |
            Self::PackPolynomialCoefficients { bits: indices, .. } |
            Self::MatrixConcat { inputs: indices, .. } => indices,
            Self::IntAdd(children) |
            Self::IntSub(children) |
            Self::IntMul(children) |
            Self::IntExactDiv(children) |
            Self::IntEuclideanDiv(children) |
            Self::IntEuclideanRemainder(children) |
            Self::IntRoundDiv(children) |
            Self::IntEqual(children) |
            Self::IntLess(children) |
            Self::IntLessEqual(children) |
            Self::RealAdd(children) |
            Self::RealSub(children) |
            Self::RealMul(children) |
            Self::RealDiv(children) |
            Self::MatrixScale(children) |
            Self::MatrixTensor(children) => children,
            Self::ExtractCoefficient { input, .. } => input,
            Self::IntLog2Ceil(children) |
            Self::BoolToInt(children) |
            Self::IntToReal(children) |
            Self::RealSqrt(children) |
            Self::MatrixNegate(children) |
            Self::MatrixTranspose(children) => children,
            Self::BitExtract { input, .. } |
            Self::MatrixSlice { input, .. } |
            Self::LiftConstantPolynomial { input, .. } => input,
            Self::IntConst(_) |
            Self::IntParameter(_) |
            Self::IntBinder(_) |
            Self::BoolConst(_) |
            Self::RealConst(_) |
            Self::MatrixConstant(_) => &mut [],
        }
    }
}

impl fmt::Display for MxxLang {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.operator_name())
    }
}

impl FromOp for MxxLang {
    type Error = egg::FromOpError;

    fn from_op(operation: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        // The checked lowerer is the only constructor. Untyped parsing would
        // bypass owner-resolved attributes, so it is deliberately closed.
        Err(egg::FromOpError::new(operation, children))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: usize) -> Id {
        Id::from(value)
    }

    #[test]
    fn dynamic_nodes_keep_every_ordered_child() {
        let hash = MxxLang::HashPlain {
            query: HashQuerySpecId(7),
            arguments: vec![id(1), id(2), id(3)].into_boxed_slice(),
        };
        let switch = MxxLang::Switch(vec![id(4), id(5), id(6), id(7)].into_boxed_slice());
        let product = MxxLang::MatrixMultiply(vec![id(8), id(9), id(10)].into_boxed_slice());

        assert_eq!(hash.children(), &[id(1), id(2), id(3)]);
        assert_eq!(switch.children(), &[id(4), id(5), id(6), id(7)]);
        assert_eq!(product.children(), &[id(8), id(9), id(10)]);
    }

    #[test]
    fn n_ary_matrix_add_keeps_every_ordered_child() {
        let sum = MxxLang::MatrixAdd(vec![id(1), id(2), id(3), id(4)].into_boxed_slice());

        assert_eq!(sum.children(), &[id(1), id(2), id(3), id(4)]);
        assert!(sum.matches(&MxxLang::MatrixAdd(
            vec![id(10), id(11), id(12), id(13)].into_boxed_slice(),
        )));
        assert!(
            !sum.matches(&MxxLang::MatrixAdd(vec![id(10), id(11), id(12)].into_boxed_slice(),))
        );
    }

    #[test]
    fn matrix_constant_identity_is_its_interned_spec_id() {
        let first = MxxLang::MatrixConstant(MatrixConstantSpecId(4));
        let same = MxxLang::MatrixConstant(MatrixConstantSpecId(4));
        let different = MxxLang::MatrixConstant(MatrixConstantSpecId(5));

        assert!(first.matches(&same));
        assert!(!first.matches(&different));
        assert!(first.children().is_empty());
    }

    #[test]
    fn atom_identity_includes_source_and_index_arity_but_not_child_ids() {
        let first = MxxLang::Atom {
            source: AtomicSourceId(4),
            indices: vec![id(1), id(2)].into_boxed_slice(),
        };
        let same_source = MxxLang::Atom {
            source: AtomicSourceId(4),
            indices: vec![id(10), id(11)].into_boxed_slice(),
        };
        let different_arity =
            MxxLang::Atom { source: AtomicSourceId(4), indices: vec![id(1)].into_boxed_slice() };
        let different_source = MxxLang::Atom {
            source: AtomicSourceId(5),
            indices: vec![id(1), id(2)].into_boxed_slice(),
        };

        assert!(first.matches(&same_source));
        assert!(!first.matches(&different_arity));
        assert!(!first.matches(&different_source));
    }

    #[test]
    fn children_mut_rewrites_the_actual_dynamic_storage() {
        let mut node = MxxLang::Switch(vec![id(1), id(2), id(3)].into_boxed_slice());
        node.children_mut()[1] = id(99);
        assert_eq!(node.children(), &[id(1), id(99), id(3)]);
    }

    #[test]
    fn from_op_is_fail_closed() {
        let error = MxxLang::from_op("matrix-add", vec![id(1), id(2)]).unwrap_err();
        assert!(format!("{error:?}").contains("matrix-add"));
    }
}
