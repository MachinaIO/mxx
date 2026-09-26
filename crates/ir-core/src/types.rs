use crate::{
    expr::{IntExpr, RealExpr},
    ring::{ConcreteRing, RingRef},
    serde_support,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

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
    Trapdoor {
        matrix: MatrixType,
        sigma: crate::expr::RealExpr,
        gadget_base: IntExpr,
        digit_count: IntExpr,
        preimage_max_coefficient_bound: IntExpr,
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
        count: usize,
    },
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
