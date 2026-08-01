use crate::{expression::SymbolicExprId, overlay::StableVirtualAtomId, serde_support};
use mxx_ir_core::{
    ScopedWireRef,
    expr::RealExpr,
    node::{ConstantMatrix, HashVariant},
    types::ConcreteMatrixType,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub use mxx_ir_core::artifact::{ProductionId, SpecHash};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ManifestAtomId(#[serde(with = "serde_support::hex32")] pub [u8; 32]);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SelectionDomain {
    pub index_wire: ScopedWireRef,
    pub instantiation_path: Vec<SymbolicInstantiationFrame>,
    pub count: u64,
    #[serde(with = "serde_support::bigint")]
    pub modulus: BigInt,
    pub ring_dimension: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum SymbolicInstantiationFrame {
    Call(ScopedWireRef),
    ParallelIteration {
        call_site: ScopedWireRef,
        index_slot: u32,
        index: ParallelIndex,
        /// Added to the selected iteration for a structural `ZipOffset` input.
        index_offset: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ParallelIndex {
    Template,
    Static(u64),
    Dynamic(ScopedWireRef),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum AtomId {
    Constant {
        kind: String,
        /// Canonically encoded compile parameters, including matrix parameters.
        params: Vec<String>,
    },
    Local(ScopedWireRef),
    TrapdoorPublic(ScopedWireRef),
    Instantiated {
        template: ScopedWireRef,
        instantiation_path: Vec<SymbolicInstantiationFrame>,
    },
    Imported {
        production_id: ProductionId,
        manifest_atom_id: ManifestAtomId,
    },
    Virtual(StableVirtualAtomId),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum AtomKind {
    Large,
    Bounded,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ExternalSourceKind {
    Matrix,
    Preimage,
    TrapdoorUniform,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum SourceKind {
    ConstantMatrix {
        value: ConstantMatrix,
    },
    UniformSample {
        #[serde(with = "serde_support::bigint")]
        minimum: BigInt,
        #[serde(with = "serde_support::bigint")]
        maximum: BigInt,
    },
    GaussianSample {
        sigma: RealExpr,
    },
    TrapdoorUniform {
        sigma: RealExpr,
        #[serde(with = "serde_support::bigint")]
        gadget_base: BigInt,
        digit_count: usize,
    },
    PreimageSample {
        trapdoor_sigma: RealExpr,
        #[serde(with = "serde_support::bigint")]
        gadget_base: BigInt,
        digit_count: usize,
        public_matrix_rows: usize,
        target_block_rows: usize,
        zero_rows: Option<usize>,
    },
    GadgetDecomposition {
        #[serde(with = "serde_support::bigint")]
        base: BigInt,
        digit_count: usize,
        small: bool,
    },
    HashSample {
        variant: HashVariant,
        #[serde(with = "serde_support::optional_bigint")]
        base: Option<BigInt>,
        digit_count: Option<usize>,
    },
    HashTarget {
        variant: HashVariant,
    },
    External {
        kind: ExternalSourceKind,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum DeclaredDependencyRef {
    Local(String),
    Imported { production_id: ProductionId, label: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum DeclaredDependencies {
    Known(BTreeSet<DeclaredDependencyRef>),
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssumedMetadata {
    pub norm: RealExpr,
    pub is_const_poly: bool,
    pub zero_rows: Option<usize>,
    pub dependencies: DeclaredDependencies,
    pub clt_ready: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum SelectionDomainRef {
    Local(SelectionDomain),
    Imported { production_id: ProductionId, domain: SelectionDomain },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum AtomClass {
    Source { source: SourceKind },
    Assumed { metadata: Option<AssumedMetadata> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreimageRelation {
    pub left_matrix: AtomId,
    pub preimage: AtomId,
    pub product: SymbolicExprId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    pub id: AtomId,
    pub class: AtomClass,
    pub kind: AtomKind,
    pub matrix_type: ConcreteMatrixType,
}

impl Atom {
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
