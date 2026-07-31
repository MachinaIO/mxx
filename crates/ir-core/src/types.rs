use crate::{
    expr::{IntExpr, RealExpr},
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
pub struct MatrixType {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub rows: IntExpr,
    pub columns: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ConcreteMatrixType {
    #[serde(with = "serde_support::bigint")]
    pub modulus: BigInt,
    pub ring_dimension: usize,
    pub rows: usize,
    pub columns: usize,
}

impl ConcreteMatrixType {
    pub fn scalar(modulus: BigInt, ring_dimension: usize) -> Self {
        Self { modulus, ring_dimension, rows: 1, columns: 1 }
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
    Matrix(MatrixType),
    Trapdoor {
        matrix: MatrixType,
        sigma: crate::expr::RealExpr,
        gadget_base: IntExpr,
        digit_count: IntExpr,
    },
    Preimage(MatrixType),
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
    Matrix(ConcreteMatrixType),
    Trapdoor {
        matrix: ConcreteMatrixType,
        sigma: RealExpr,
        #[serde(with = "serde_support::bigint")]
        gadget_base: BigInt,
        digit_count: usize,
    },
    Preimage(ConcreteMatrixType),
}

impl ConcreteWireType {
    pub fn matrix_type(&self) -> Option<&ConcreteMatrixType> {
        match self {
            Self::Matrix(matrix) | Self::Preimage(matrix) => Some(matrix),
            Self::Trapdoor { matrix, .. } => Some(matrix),
            _ => None,
        }
    }
}
