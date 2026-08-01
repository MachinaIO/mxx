use crate::atom::{ManifestAtomId, ProductionId};
use mxx_ir_core::{
    ScopedWireRef,
    expr::{IntExpr, RealExpr},
    types::MatrixType,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StableVirtualAtomId(pub u64);

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolicOverlay {
    pub virtual_atoms: BTreeMap<StableVirtualAtomId, VirtualAtomDecl>,
    pub assumptions: BTreeMap<ScopedWireRef, PendingSymbolicExpr>,
    pub preimage_relations: Vec<PreimageRelation>,
}

impl SymbolicOverlay {
    pub fn is_empty(&self) -> bool {
        self.virtual_atoms.is_empty() &&
            self.assumptions.is_empty() &&
            self.preimage_relations.is_empty()
    }

    pub fn validate(&self) -> Result<(), String> {
        for expression in self.assumptions.values() {
            validate_expression(expression, self)?;
        }
        for relation in &self.preimage_relations {
            validate_atom_ref(&relation.left_matrix, self)?;
            validate_atom_ref(&relation.preimage, self)?;
            validate_expression(&relation.product, self)?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Option<[u8; 32]>, String> {
        if self.is_empty() {
            return Ok(None);
        }
        #[derive(Serialize)]
        struct DigestView<'a> {
            virtual_atoms: Vec<(&'a StableVirtualAtomId, &'a VirtualAtomDecl)>,
            assumptions: Vec<(&'a ScopedWireRef, &'a PendingSymbolicExpr)>,
            preimage_relations: &'a [PreimageRelation],
        }
        let bytes = serde_json::to_vec(&DigestView {
            virtual_atoms: self.virtual_atoms.iter().collect(),
            assumptions: self.assumptions.iter().collect(),
            preimage_relations: &self.preimage_relations,
        })
        .map_err(|error| error.to_string())?;
        Ok(Some(Sha256::digest(bytes).into()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VirtualAtomDecl {
    pub diagnostic_name: String,
    pub matrix_type: MatrixType,
    pub kind: VirtualKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum VirtualKind {
    Large,
    Bounded {
        norm: RealExpr,
        is_const_poly: bool,
        zero_rows: Option<IntExpr>,
        dependencies: DeclaredDependencyLabels,
        clt_ready: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum DeclaredDependencyLabels {
    Known(BTreeSet<String>),
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ExactAtomRef {
    Local(ScopedWireRef),
    Virtual(StableVirtualAtomId),
    Imported { production_id: ProductionId, manifest_atom_id: ManifestAtomId },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum SymbolicValueRef {
    Local(ScopedWireRef),
    Virtual(StableVirtualAtomId),
    ImportedAtom { production_id: ProductionId, manifest_atom_id: ManifestAtomId },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PendingSymbolicExpr {
    pub matrix_type: MatrixType,
    pub node: PendingSymbolicExprNode,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum PendingSymbolicExprNode {
    Zero,
    Value(SymbolicValueRef),
    Add(Vec<PendingSymbolicExpr>),
    Scale { coefficient: IntExpr, value: Box<PendingSymbolicExpr> },
    Mul(Vec<PendingSymbolicExpr>),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreimageRelation {
    pub left_matrix: ExactAtomRef,
    pub preimage: ExactAtomRef,
    pub product: PendingSymbolicExpr,
}

fn validate_expression(
    expression: &PendingSymbolicExpr,
    overlay: &SymbolicOverlay,
) -> Result<(), String> {
    match &expression.node {
        PendingSymbolicExprNode::Zero => {}
        PendingSymbolicExprNode::Value(SymbolicValueRef::Virtual(id)) => {
            if !overlay.virtual_atoms.contains_key(id) {
                return Err(format!("virtual atom {id:?} is undeclared"));
            }
        }
        PendingSymbolicExprNode::Value(_) => {}
        PendingSymbolicExprNode::Add(children) | PendingSymbolicExprNode::Mul(children) => {
            if children.is_empty() {
                return Err("symbolic Add or Mul must not be empty".to_owned());
            }
            for child in children {
                validate_expression(child, overlay)?;
            }
        }
        PendingSymbolicExprNode::Scale { value, .. } => validate_expression(value, overlay)?,
    }
    Ok(())
}

fn validate_atom_ref(reference: &ExactAtomRef, overlay: &SymbolicOverlay) -> Result<(), String> {
    if let ExactAtomRef::Virtual(id) = reference &&
        !overlay.virtual_atoms.contains_key(id)
    {
        return Err(format!("virtual atom {id:?} is undeclared"));
    }
    Ok(())
}
