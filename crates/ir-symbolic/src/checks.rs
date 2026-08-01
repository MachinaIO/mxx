use mxx_ir_core::types::NodeId;
use serde::{Deserialize, Serialize};

pub use mxx_ir_core::checks::{
    CheckError, check_add_shape, check_same_ring, check_topological, multiplication_type,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum WarningKind {
    RuntimeSelectBoundsCheck,
    UnusedVirtualAtom,
    UnusedAssumption,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ElaborationWarning {
    pub node: NodeId,
    pub kind: WarningKind,
    pub message: String,
}
