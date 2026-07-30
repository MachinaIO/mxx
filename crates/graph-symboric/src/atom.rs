use crate::{
    serde_support,
    term::TermList,
    types::{ConcreteMatrixType, NodeId, WireRef},
    ubound::UBound,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub use mxx_graph_ir::{
    artifact::{ProductionId, SpecHash},
    node::ConcatAxis,
    types::InstantiationFrame,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ManifestAtomId(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct TermListId(pub u64);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SelectionDomain {
    pub index_wire: WireRef,
    pub instantiation_path: Vec<InstantiationFrame>,
    pub count: u64,
    #[serde(with = "serde_support::bigint")]
    pub modulus: BigInt,
    pub ring_dimension: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum AtomId {
    Constant {
        kind: String,
        /// Canonically encoded compile parameters, including matrix parameters.
        params: Vec<String>,
    },
    Local {
        instantiation_path: Vec<InstantiationFrame>,
        node: NodeId,
        port: u32,
    },
    Imported {
        production_id: ProductionId,
        manifest_atom_id: ManifestAtomId,
    },
    Indicator {
        domain: SelectionDomain,
        branch: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum AtomKind {
    Large,
    Bounded { norm: UBound },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum DefExpr {
    TermList(TermList),
    Tensor {
        left: AtomId,
        right: AtomId,
    },
    Concat {
        inputs: Vec<AtomId>,
        axis: ConcatAxis,
    },
    Reshape {
        input: AtomId,
        rows: usize,
        columns: usize,
    },
    ModDownImage {
        source: AtomId,
        #[serde(with = "serde_support::bigint")]
        source_modulus: BigInt,
        #[serde(with = "serde_support::bigint")]
        target_modulus: BigInt,
    },
    ModUpLift {
        source: AtomId,
        #[serde(with = "serde_support::bigint")]
        source_modulus: BigInt,
        #[serde(with = "serde_support::bigint")]
        target_modulus: BigInt,
    },
    Indicator {
        index_wire: WireRef,
        branch: u64,
    },
    ModDownError {
        input: WireRef,
        signal: TermList,
        #[serde(with = "serde_support::bigint")]
        source_modulus: BigInt,
        #[serde(with = "serde_support::bigint")]
        target_modulus: BigInt,
    },
    ModUpError {
        input: WireRef,
        lifted: TermList,
        #[serde(with = "serde_support::bigint")]
        source_modulus: BigInt,
        #[serde(with = "serde_support::bigint")]
        target_modulus: BigInt,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum AtomClass {
    Source,
    Derived { definition: DefExpr },
    Ghost { definition: DefExpr },
    OpaqueImported,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum TargetRef {
    Local(WireRef),
    Imported { production_id: ProductionId, term_list_id: TermListId },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreimageRefs {
    pub uniform: AtomId,
    pub target: TargetRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    pub id: AtomId,
    pub class: AtomClass,
    pub kind: AtomKind,
    pub matrix_type: ConcreteMatrixType,
    pub dependencies: BTreeSet<AtomId>,
    pub preimage_refs: Option<PreimageRefs>,
}

impl Atom {
    pub fn norm(&self) -> Option<&UBound> {
        match &self.kind {
            AtomKind::Large => None,
            AtomKind::Bounded { norm } => Some(norm),
        }
    }

    pub fn is_large(&self) -> bool {
        matches!(self.kind, AtomKind::Large)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AtomTable {
    atoms: BTreeMap<AtomId, Atom>,
}

impl AtomTable {
    pub fn insert(&mut self, atom: Atom) -> Option<Atom> {
        self.atoms.insert(atom.id.clone(), atom)
    }

    pub fn get(&self, id: &AtomId) -> Option<&Atom> {
        self.atoms.get(id)
    }

    pub fn get_mut(&mut self, id: &AtomId) -> Option<&mut Atom> {
        self.atoms.get_mut(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&AtomId, &Atom)> {
        self.atoms.iter()
    }

    pub fn values(&self) -> impl Iterator<Item = &Atom> {
        self.atoms.values()
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    pub fn contains_key(&self, id: &AtomId) -> bool {
        self.atoms.contains_key(id)
    }
}
