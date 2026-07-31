use crate::{
    atom::{ManifestAtomId, ProductionId},
    term::ViewDescriptor,
    types::{MatrixType, NodeId},
};
use mxx_ir_core::{
    expr::{IntExpr, RealExpr},
    types::InstantiationFrame,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct AssumedTermListId(pub String);

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolicOverlay {
    pub virtual_atoms: BTreeMap<String, VirtualAtomDecl>,
    pub term_lists: BTreeMap<AssumedTermListId, AssumedTermList>,
    pub entries: Vec<(WireSelector, Reinterpretation)>,
}

impl SymbolicOverlay {
    pub fn is_empty(&self) -> bool {
        self.virtual_atoms.is_empty() && self.term_lists.is_empty() && self.entries.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VirtualAtomDecl {
    pub matrix_type: MatrixType,
    pub kind: VirtualKind,
    pub preimage: Option<AssumedPreimage>,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssumedPreimage {
    pub uniform: AtomRef,
    pub target: AssumedTermListId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum Reinterpretation {
    Fold(FoldSpec),
    Unfold(UnfoldSpec),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FoldSpec {
    pub expected: ExpectedTermList,
    pub groups: Vec<FoldGroup>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum FoldGroup {
    Signal { terms: BTreeSet<usize>, suffix_len: u32 },
    Noise { terms: BTreeSet<usize> },
    Keep { terms: BTreeSet<usize> },
}

impl FoldGroup {
    pub(crate) fn terms(&self) -> &BTreeSet<usize> {
        match self {
            Self::Signal { terms, .. } | Self::Noise { terms } | Self::Keep { terms } => terms,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnfoldSpec {
    pub new_terms: AssumedTermListId,
    pub replace_derived: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct WireSelector {
    pub path: Vec<FrameMatcher>,
    pub node: NodeId,
    pub port: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct FrameMatcher {
    pub call: NodeId,
    pub loop_index: LoopIndexMatcher,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum LoopIndexMatcher {
    Concrete(u64),
    Any,
    Var(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum PortMatcher {
    Concrete(u32),
    Affine { var: String, stride: u32, offset: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum AtomRef {
    Constant { kind: String, params: Vec<String> },
    Node { path: Vec<FrameMatcher>, node: NodeId, port: PortMatcher },
    Virtual { name: String },
    Imported { production_id: ProductionId, manifest_atom_id: ManifestAtomId },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FactorRef {
    pub atom: AtomRef,
    pub view: Option<ViewDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OverlayTerm {
    pub coefficient: IntExpr,
    pub factors: Vec<FactorRef>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssumedTermList {
    pub terms: Vec<OverlayTerm>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ExpectedEntry {
    Term(OverlayTerm),
    IndicatorSum { select: SelectNodeSelector, index_var: String, body: OverlayTerm },
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExpectedTermList {
    pub entries: Vec<ExpectedEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SelectNodeSelector {
    pub path: Vec<FrameMatcher>,
    pub node: NodeId,
}

pub(crate) fn selector_matches(
    selector: &WireSelector,
    path: &[InstantiationFrame],
    node: NodeId,
    port: u32,
) -> Option<BTreeMap<String, u64>> {
    if selector.node != node || selector.port != port || selector.path.len() != path.len() {
        return None;
    }
    let mut bindings = BTreeMap::new();
    for (matcher, frame) in selector.path.iter().zip(path) {
        if matcher.call != frame.call {
            return None;
        }
        match (&matcher.loop_index, frame.loop_index) {
            (LoopIndexMatcher::Concrete(expected), Some(actual)) if *expected == actual => {}
            (LoopIndexMatcher::Any, Some(_)) => {}
            (LoopIndexMatcher::Var(name), Some(actual)) => {
                if bindings.insert(name.clone(), actual).is_some_and(|previous| previous != actual)
                {
                    return None;
                }
            }
            (_, None) if matches!(matcher.loop_index, LoopIndexMatcher::Any) => {}
            _ => return None,
        }
    }
    Some(bindings)
}

pub(crate) fn validate_overlay(overlay: &SymbolicOverlay) -> Result<(), String> {
    for (name, declaration) in &overlay.virtual_atoms {
        if let Some(preimage) = &declaration.preimage {
            if matches!(preimage.uniform, AtomRef::Node { .. }) {
                return Err(format!("virtual atom {name} uses a forbidden global Node reference"));
            }
            if !overlay.term_lists.contains_key(&preimage.target) {
                return Err(format!(
                    "virtual atom {name} refers to undeclared term list {}",
                    preimage.target.0
                ));
            }
            validate_global_atom_ref(&preimage.uniform, overlay)?;
        }
    }
    for (id, terms) in &overlay.term_lists {
        for term in &terms.terms {
            for factor in &term.factors {
                if matches!(factor.atom, AtomRef::Node { .. }) {
                    return Err(format!(
                        "assumed term list {} uses a forbidden global Node reference",
                        id.0
                    ));
                }
                validate_global_atom_ref(&factor.atom, overlay)?;
            }
        }
    }
    for (entry_index, (_, reinterpretation)) in overlay.entries.iter().enumerate() {
        match reinterpretation {
            Reinterpretation::Fold(spec) => {
                let mut positions = BTreeSet::new();
                for group in &spec.groups {
                    for position in group.terms() {
                        if *position >= spec.expected.entries.len() {
                            return Err(format!(
                                "fold entry {entry_index} has an out-of-range group position"
                            ));
                        }
                        if !positions.insert(*position) {
                            return Err(format!("fold entry {entry_index} has overlapping groups"));
                        }
                    }
                }
                if positions.len() != spec.expected.entries.len() {
                    return Err(format!(
                        "fold entry {entry_index} groups do not partition expected positions"
                    ));
                }
                for expected in &spec.expected.entries {
                    match expected {
                        ExpectedEntry::Term(term) => validate_entry_term(term, overlay)?,
                        ExpectedEntry::IndicatorSum { index_var, body, .. } => {
                            validate_indicator_term(index_var, body, overlay)?;
                        }
                    }
                }
            }
            Reinterpretation::Unfold(spec) => {
                if !overlay.term_lists.contains_key(&spec.new_terms) {
                    return Err(format!(
                        "unfold entry {entry_index} refers to undeclared term list {}",
                        spec.new_terms.0
                    ));
                }
            }
        }
    }
    validate_assumed_cycles(overlay)
}

fn validate_global_atom_ref(reference: &AtomRef, overlay: &SymbolicOverlay) -> Result<(), String> {
    match reference {
        AtomRef::Virtual { name } if !overlay.virtual_atoms.contains_key(name) => {
            Err(format!("virtual atom {name} is undeclared"))
        }
        AtomRef::Node { .. } => Err("Node references are forbidden in global declarations".into()),
        _ => Ok(()),
    }
}

fn validate_entry_term(term: &OverlayTerm, overlay: &SymbolicOverlay) -> Result<(), String> {
    for factor in &term.factors {
        if let AtomRef::Virtual { name } = &factor.atom &&
            !overlay.virtual_atoms.contains_key(name)
        {
            return Err(format!("virtual atom {name} is undeclared"));
        }
        if let AtomRef::Node { port: PortMatcher::Affine { .. }, .. } = &factor.atom {
            return Err("affine ports are legal only inside IndicatorSum".into());
        }
    }
    Ok(())
}

fn validate_indicator_term(
    index_var: &str,
    term: &OverlayTerm,
    overlay: &SymbolicOverlay,
) -> Result<(), String> {
    for factor in &term.factors {
        if let AtomRef::Virtual { name } = &factor.atom &&
            !overlay.virtual_atoms.contains_key(name)
        {
            return Err(format!("virtual atom {name} is undeclared"));
        }
        if let AtomRef::Node { port: PortMatcher::Affine { var, .. }, .. } = &factor.atom &&
            var != index_var
        {
            return Err(format!("IndicatorSum port variable {var} differs from binder {index_var}"));
        }
    }
    Ok(())
}

fn validate_assumed_cycles(overlay: &SymbolicOverlay) -> Result<(), String> {
    let mut edges = BTreeMap::<AssumedTermListId, BTreeSet<AssumedTermListId>>::new();
    for (id, terms) in &overlay.term_lists {
        let outgoing = edges.entry(id.clone()).or_default();
        for factor in terms.terms.iter().flat_map(|term| &term.factors) {
            if let AtomRef::Virtual { name } = &factor.atom &&
                let Some(target) = overlay
                    .virtual_atoms
                    .get(name)
                    .and_then(|declaration| declaration.preimage.as_ref())
                    .map(|preimage| preimage.target.clone())
            {
                outgoing.insert(target);
            }
        }
    }
    fn visit(
        id: &AssumedTermListId,
        edges: &BTreeMap<AssumedTermListId, BTreeSet<AssumedTermListId>>,
        active: &mut BTreeSet<AssumedTermListId>,
        complete: &mut BTreeSet<AssumedTermListId>,
    ) -> Result<(), String> {
        if complete.contains(id) {
            return Ok(());
        }
        if !active.insert(id.clone()) {
            return Err(format!("assumed term-list cycle contains {}", id.0));
        }
        if let Some(targets) = edges.get(id) {
            for target in targets {
                visit(target, edges, active, complete)?;
            }
        }
        active.remove(id);
        complete.insert(id.clone());
        Ok(())
    }
    let mut active = BTreeSet::new();
    let mut complete = BTreeSet::new();
    for id in overlay.term_lists.keys() {
        visit(id, &edges, &mut active, &mut complete)?;
    }
    Ok(())
}

pub(crate) type OverlayHashes = (Option<[u8; 32]>, Option<[u8; 32]>);

pub(crate) fn overlay_hashes(overlay: &SymbolicOverlay) -> Result<OverlayHashes, String> {
    const OVERLAY_FORMAT_VERSION: u32 = 2;
    if overlay.is_empty() {
        return Ok((None, None));
    }
    let mut canonical = overlay.clone();
    canonical.entries.sort_by_cached_key(|entry| {
        serde_json::to_vec(entry).expect("overlay entries must serialize")
    });
    #[derive(Serialize)]
    struct VersionedOverlay<'a> {
        format_version: u32,
        overlay: &'a SymbolicOverlay,
    }
    let overlay_hash = Some(hash_serializable(&VersionedOverlay {
        format_version: OVERLAY_FORMAT_VERSION,
        overlay: &canonical,
    })?);
    let mut unfold_entries = canonical
        .entries
        .iter()
        .filter(|(_, reinterpretation)| matches!(reinterpretation, Reinterpretation::Unfold(_)))
        .cloned()
        .collect::<Vec<_>>();
    if unfold_entries.is_empty() {
        return Ok((overlay_hash, None));
    }
    unfold_entries.sort_by_cached_key(|entry| {
        serde_json::to_vec(entry).expect("overlay entries must serialize")
    });
    let mut reachable = BTreeSet::new();
    let mut queue = unfold_entries
        .iter()
        .filter_map(|(_, reinterpretation)| match reinterpretation {
            Reinterpretation::Unfold(spec) => Some(spec.new_terms.clone()),
            Reinterpretation::Fold(_) => None,
        })
        .collect::<Vec<_>>();
    while let Some(id) = queue.pop() {
        if !reachable.insert(id.clone()) {
            continue;
        }
        if let Some(terms) = canonical.term_lists.get(&id) {
            for factor in terms.terms.iter().flat_map(|term| &term.factors) {
                if let AtomRef::Virtual { name } = &factor.atom &&
                    let Some(target) = canonical
                        .virtual_atoms
                        .get(name)
                        .and_then(|declaration| declaration.preimage.as_ref())
                        .map(|preimage| preimage.target.clone())
                {
                    queue.push(target);
                }
            }
        }
    }
    let term_lists = reachable
        .into_iter()
        .filter_map(|id| canonical.term_lists.get(&id).cloned().map(|terms| (id, terms)))
        .collect::<BTreeMap<_, _>>();
    #[derive(Serialize)]
    struct Assumptions<'a> {
        format_version: u32,
        virtual_atoms: &'a BTreeMap<String, VirtualAtomDecl>,
        term_lists: BTreeMap<AssumedTermListId, AssumedTermList>,
        entries: Vec<(WireSelector, Reinterpretation)>,
    }
    let assumptions = Assumptions {
        format_version: OVERLAY_FORMAT_VERSION,
        virtual_atoms: &canonical.virtual_atoms,
        term_lists,
        entries: unfold_entries,
    };
    Ok((overlay_hash, Some(hash_serializable(&assumptions)?)))
}

fn hash_serializable(value: &impl Serialize) -> Result<[u8; 32], String> {
    let encoded = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(Sha256::digest(encoded).into())
}
