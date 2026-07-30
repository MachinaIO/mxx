use crate::{
    atom::{AtomKind, AtomTable},
    term::{TermError, TermList},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use mxx_graph_ir::types::NodeId;
use mxx_graph_ir::{graph::Graph, types::ConcreteMatrixType};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum WarningKind {
    DroppedPreimageReferences,
    RuntimeSelectBoundsCheck,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ElaborationWarning {
    pub node: NodeId,
    pub kind: WarningKind,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum CheckError {
    #[error(transparent)]
    Core(#[from] mxx_graph_ir::checks::CheckError),
    #[error("operation requires a reduced input")]
    RequiresReducedInput,
    #[error("mod-down input is not in signal-tail normal form")]
    InvalidModDownNormalForm,
    #[error(transparent)]
    Terms(#[from] TermError),
}

pub fn check_topological(graph: &Graph) -> Result<(), CheckError> {
    mxx_graph_ir::checks::check_topological(graph).map_err(CheckError::Core)
}

pub fn check_same_ring(
    left: &ConcreteMatrixType,
    right: &ConcreteMatrixType,
) -> Result<(), CheckError> {
    mxx_graph_ir::checks::check_same_ring(left, right).map_err(CheckError::Core)
}

pub fn check_add_shape(
    left: &ConcreteMatrixType,
    right: &ConcreteMatrixType,
) -> Result<(), CheckError> {
    mxx_graph_ir::checks::check_add_shape(left, right).map_err(CheckError::Core)
}

pub fn multiplication_type(
    left: &ConcreteMatrixType,
    right: &ConcreteMatrixType,
) -> Result<ConcreteMatrixType, CheckError> {
    mxx_graph_ir::checks::multiplication_type(left, right).map_err(CheckError::Core)
}

pub fn is_reduced(terms: &TermList, atoms: &AtomTable) -> Result<bool, TermError> {
    Ok(terms.terms.iter().all(|term| {
        term.factors
            .iter()
            .filter(|factor| {
                atoms.get(&factor.atom).is_some_and(|atom| matches!(atom.kind, AtomKind::Large))
            })
            .count() <=
            1
    }))
}

pub fn check_mod_down_normal_form(terms: &TermList, atoms: &AtomTable) -> Result<(), CheckError> {
    for term in &terms.terms {
        let non_scalar = term
            .factors
            .iter()
            .filter(|factor| {
                atoms.get(&factor.atom).is_some_and(|atom| !atom.matrix_type.is_scalar())
            })
            .collect::<Vec<_>>();
        let large = term
            .factors
            .iter()
            .filter(|factor| {
                atoms.get(&factor.atom).is_some_and(|atom| matches!(atom.kind, AtomKind::Large))
            })
            .collect::<Vec<_>>();
        if large.is_empty() {
            continue;
        }
        if large.len() != 1 || non_scalar.last().map(|factor| &factor.atom) != Some(&large[0].atom)
        {
            return Err(CheckError::InvalidModDownNormalForm);
        }
    }
    Ok(())
}
