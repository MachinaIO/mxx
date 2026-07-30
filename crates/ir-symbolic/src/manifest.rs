use crate::{
    atom::{
        Atom, AtomClass, AtomId, AtomKind, AtomTable, ConcatAxis, DefExpr, ManifestAtomId,
        PreimageRefs, ProductionId, SpecHash, TargetRef, TermListId,
    },
    overlay::AssumedTermListId,
    serde_support,
    term::{Factor, Term, TermList, ViewDescriptor},
    types::{ConcreteMatrixType, WireId, WireRef},
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque, btree_map::Entry};
use thiserror::Error;

pub const SYMBOLIC_MANIFEST_VERSION: u32 = 2;
pub type InterpretationDigest = (Option<[u8; 32]>, Option<[u8; 32]>);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ManifestAtomRef {
    Local(ManifestAtomId),
    Imported { production_id: ProductionId, manifest_atom_id: ManifestAtomId },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ManifestTermListRef {
    Local(TermListId),
    Imported { production_id: ProductionId, term_list_id: TermListId },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
#[serde(tag = "tag", content = "value")]
pub enum LocalTermListOrigin {
    Wire(WireId),
    Assumed(AssumedTermListId),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub production_id: ProductionId,
    pub artifacts: BTreeMap<String, ManifestArtifact>,
    #[serde(with = "atom_record_map")]
    pub atoms: BTreeMap<ManifestAtomRef, ManifestAtom>,
    #[serde(with = "term_list_record_map")]
    pub term_lists: BTreeMap<ManifestTermListRef, ManifestTermList>,
    #[serde(with = "serde_support::optional_hex32")]
    pub overlay_hash: Option<[u8; 32]>,
    #[serde(with = "serde_support::optional_hex32")]
    pub assumption_hash: Option<[u8; 32]>,
    #[serde(with = "serde_support::hex32_set")]
    pub assumption_digests: BTreeSet<[u8; 32]>,
    #[serde(with = "interpretation_digest_map")]
    pub interpretation_digests: BTreeMap<ProductionId, InterpretationDigest>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ManifestMetadata {
    pub overlay_hash: Option<[u8; 32]>,
    pub assumption_hash: Option<[u8; 32]>,
    pub assumption_digests: BTreeSet<[u8; 32]>,
    pub interpretation_digests: BTreeMap<ProductionId, InterpretationDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub wire_type: ConcreteMatrixType,
    pub term_list: ManifestTermListRef,
    pub family: Option<Vec<ManifestTermListRef>>,
    #[serde(with = "serde_support::optional_hex32")]
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestAtom {
    pub class: ManifestAtomClass,
    pub kind: AtomKind,
    pub matrix_type: ConcreteMatrixType,
    pub dependencies: BTreeSet<ManifestAtomRef>,
    pub preimage_refs: Option<ManifestPreimageRefs>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ManifestAtomClass {
    Source,
    Derived { definition: ManifestDefExpr },
    Assumed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestPreimageRefs {
    pub uniform: ManifestAtomRef,
    pub target: ManifestTermListRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ManifestDefExpr {
    TermList(ManifestTermList),
    Tensor {
        left: ManifestAtomRef,
        right: ManifestAtomRef,
    },
    Concat {
        inputs: Vec<ManifestAtomRef>,
        axis: ConcatAxis,
    },
    Reshape {
        input: ManifestAtomRef,
        rows: usize,
        columns: usize,
    },
    ModDownImage {
        source: ManifestAtomRef,
        source_modulus: String,
        target_modulus: String,
    },
    ModUpLift {
        source: ManifestAtomRef,
        source_modulus: String,
        target_modulus: String,
    },
    Indicator {
        index_wire: WireRef,
        branch: u64,
    },
    ModDownError {
        input: WireRef,
        signal: ManifestTermList,
        source_modulus: String,
        target_modulus: String,
    },
    ModUpError {
        input: WireRef,
        lifted: ManifestTermList,
        source_modulus: String,
        target_modulus: String,
    },
    Fold(ManifestTermList),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestFactor {
    pub atom: ManifestAtomRef,
    pub view: Option<ViewDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestTerm {
    pub coefficient: String,
    pub factors: Vec<ManifestFactor>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestTermList {
    pub terms: Vec<ManifestTerm>,
}

#[derive(Clone, Debug)]
pub struct ExportArtifact {
    pub wire: WireId,
    pub wire_type: ConcreteMatrixType,
    pub family: Option<Vec<WireId>>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ImportedManifest {
    pub atoms: AtomTable,
    pub term_lists: BTreeMap<TargetRef, TermList>,
    pub artifacts: BTreeMap<String, ImportedArtifact>,
    pub assumption_digests: BTreeSet<[u8; 32]>,
    pub interpretation_digests: BTreeMap<ProductionId, InterpretationDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedArtifact {
    pub wire_type: ConcreteMatrixType,
    pub terms: TermList,
    pub family: Option<Vec<TermList>>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("symbolic manifest format version {actual} does not match {expected}")]
    FormatVersion { expected: u32, actual: u32 },
    #[error("manifest {production:?} does not carry its own interpretation digest")]
    MissingOwnInterpretation { production: ProductionId },
    #[error("manifest {production:?} carries an inconsistent own interpretation digest")]
    InvalidOwnInterpretation { production: ProductionId },
    #[error("manifest omits assumption digest {digest} from transitive provenance")]
    MissingAssumptionProvenance { digest: String },
    #[error("symbolic interpretations for production {production:?} disagree")]
    InterpretationConflict { production: ProductionId },
    #[error("manifest records for {key} disagree: {first} != {second}")]
    RecordConflict { key: String, first: Box<str>, second: Box<str> },
    #[error(
        "manifest artifact {artifact} for {production:?} disagrees across projections: {first} != {second}"
    )]
    ArtifactConflict {
        production: Box<ProductionId>,
        artifact: String,
        first: Box<str>,
        second: Box<str>,
    },
    #[error("wire {0:?} has no elaborated term list")]
    MissingWire(WireId),
    #[error("atom {0:?} is absent from the atom table")]
    MissingAtom(AtomId),
    #[error("manifest atom {0:?} is absent")]
    MissingManifestAtom(ManifestAtomRef),
    #[error("manifest term list {0:?} is absent")]
    MissingTermList(ManifestTermListRef),
    #[error("preimage target {0:?} has no exported term-list record")]
    MissingTarget(TargetRef),
    #[error("manifest integer is invalid: {0}")]
    InvalidInteger(String),
    #[error("SHA-256 id collision for {0}")]
    HashCollision(String),
    #[error("canonical serialization failed: {0}")]
    Serialization(String),
}

pub(crate) mod interpretation_digest_map {
    use super::{InterpretationDigest, ProductionId, SpecHash};
    use crate::serde_support;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
    use std::collections::BTreeMap;

    #[derive(Serialize, Deserialize)]
    struct DigestPair(
        #[serde(with = "serde_support::optional_hex32")] Option<[u8; 32]>,
        #[serde(with = "serde_support::optional_hex32")] Option<[u8; 32]>,
    );

    pub(super) fn production_key(production: &ProductionId) -> String {
        format!(
            "{}:{}",
            serde_support::hex32::encode(&production.spec_hash.0),
            serde_support::hex32::encode(&production.execution_nonce)
        )
    }

    pub(super) fn parse_production_key<E: Error>(key: &str) -> Result<ProductionId, E> {
        let (spec_hash, execution_nonce) =
            key.split_once(':').ok_or_else(|| E::custom("invalid canonical production id"))?;
        Ok(ProductionId {
            spec_hash: SpecHash(serde_support::hex32::decode::<E>(spec_hash)?),
            execution_nonce: serde_support::hex32::decode::<E>(execution_nonce)?,
        })
    }

    pub fn serialize<S: Serializer>(
        value: &BTreeMap<ProductionId, InterpretationDigest>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .iter()
            .map(|(production, digest)| {
                (production_key(production), DigestPair(digest.0, digest.1))
            })
            .collect::<BTreeMap<_, _>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<ProductionId, InterpretationDigest>, D::Error> {
        BTreeMap::<String, DigestPair>::deserialize(deserializer)?
            .into_iter()
            .map(|(key, digest)| {
                Ok((parse_production_key::<D::Error>(&key)?, (digest.0, digest.1)))
            })
            .collect()
    }
}

mod atom_record_map {
    use super::{ManifestAtom, ManifestAtomId, ManifestAtomRef, interpretation_digest_map};
    use crate::serde_support;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
    use std::collections::BTreeMap;

    fn key(reference: &ManifestAtomRef) -> String {
        match reference {
            ManifestAtomRef::Local(id) => {
                format!("local:{}", serde_support::hex32::encode(&id.0))
            }
            ManifestAtomRef::Imported { production_id, manifest_atom_id } => format!(
                "imported:{}:{}",
                interpretation_digest_map::production_key(production_id),
                serde_support::hex32::encode(&manifest_atom_id.0)
            ),
        }
    }

    fn parse<E: Error>(key: &str) -> Result<ManifestAtomRef, E> {
        if let Some(id) = key.strip_prefix("local:") {
            return Ok(ManifestAtomRef::Local(ManifestAtomId(serde_support::hex32::decode::<E>(
                id,
            )?)));
        }
        let components = key.split(':').collect::<Vec<_>>();
        if components.len() != 4 || components[0] != "imported" {
            return Err(E::custom("invalid canonical manifest atom reference"));
        }
        Ok(ManifestAtomRef::Imported {
            production_id: interpretation_digest_map::parse_production_key::<E>(&format!(
                "{}:{}",
                components[1], components[2]
            ))?,
            manifest_atom_id: ManifestAtomId(serde_support::hex32::decode::<E>(components[3])?),
        })
    }

    pub fn serialize<S: Serializer>(
        value: &BTreeMap<ManifestAtomRef, ManifestAtom>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .iter()
            .map(|(reference, record)| (key(reference), record))
            .collect::<BTreeMap<_, _>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<ManifestAtomRef, ManifestAtom>, D::Error> {
        BTreeMap::<String, ManifestAtom>::deserialize(deserializer)?
            .into_iter()
            .map(|(key, record)| Ok((parse::<D::Error>(&key)?, record)))
            .collect()
    }
}

mod term_list_record_map {
    use super::{ManifestTermList, ManifestTermListRef, TermListId, interpretation_digest_map};
    use crate::serde_support;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
    use std::collections::BTreeMap;

    fn key(reference: &ManifestTermListRef) -> String {
        match reference {
            ManifestTermListRef::Local(id) => {
                format!("local:{}", serde_support::hex32::encode(&id.0))
            }
            ManifestTermListRef::Imported { production_id, term_list_id } => format!(
                "imported:{}:{}",
                interpretation_digest_map::production_key(production_id),
                serde_support::hex32::encode(&term_list_id.0)
            ),
        }
    }

    fn parse<E: Error>(key: &str) -> Result<ManifestTermListRef, E> {
        if let Some(id) = key.strip_prefix("local:") {
            return Ok(ManifestTermListRef::Local(TermListId(serde_support::hex32::decode::<E>(
                id,
            )?)));
        }
        let components = key.split(':').collect::<Vec<_>>();
        if components.len() != 4 || components[0] != "imported" {
            return Err(E::custom("invalid canonical manifest term-list reference"));
        }
        Ok(ManifestTermListRef::Imported {
            production_id: interpretation_digest_map::parse_production_key::<E>(&format!(
                "{}:{}",
                components[1], components[2]
            ))?,
            term_list_id: TermListId(serde_support::hex32::decode::<E>(components[3])?),
        })
    }

    pub fn serialize<S: Serializer>(
        value: &BTreeMap<ManifestTermListRef, ManifestTermList>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .iter()
            .map(|(reference, record)| (key(reference), record))
            .collect::<BTreeMap<_, _>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<ManifestTermListRef, ManifestTermList>, D::Error> {
        BTreeMap::<String, ManifestTermList>::deserialize(deserializer)?
            .into_iter()
            .map(|(key, record)| Ok((parse::<D::Error>(&key)?, record)))
            .collect()
    }
}

pub fn production_id(spec_hash: SpecHash, execution_nonce: [u8; 32]) -> ProductionId {
    ProductionId { spec_hash, execution_nonce }
}

pub fn manifest_atom_id(id: &AtomId) -> Result<ManifestAtomId, ManifestError> {
    canonical_digest(id).map(ManifestAtomId)
}

pub fn term_list_id(origin: &LocalTermListOrigin) -> Result<TermListId, ManifestError> {
    canonical_digest(origin).map(TermListId)
}

fn canonical_digest(value: &impl Serialize) -> Result<[u8; 32], ManifestError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ManifestError::Serialization(error.to_string()))?;
    Ok(Sha256::digest(bytes).into())
}

pub fn export_manifest(
    production_id: ProductionId,
    artifacts: &BTreeMap<String, ExportArtifact>,
    wire_terms: &BTreeMap<WireId, TermList>,
    target_terms: &BTreeMap<TargetRef, TermList>,
    atoms: &AtomTable,
    metadata: &ManifestMetadata,
) -> Result<Manifest, ManifestError> {
    let mut atom_records = BTreeMap::new();
    let mut term_records = BTreeMap::new();
    let mut atom_origins = BTreeMap::<ManifestAtomId, AtomId>::new();
    let mut term_origins = BTreeMap::<TermListId, LocalTermListOrigin>::new();
    let mut atom_queue = VecDeque::new();
    let mut target_queue = VecDeque::new();

    let exported_artifacts = artifacts
        .iter()
        .map(|(name, artifact)| {
            let term_list = local_wire_ref(&artifact.wire, &mut term_origins)?;
            target_queue
                .push_back((TargetRef::Local(artifact.wire.wire), Some(artifact.wire.clone())));
            let family = artifact
                .family
                .as_ref()
                .map(|members| {
                    members
                        .iter()
                        .map(|wire| {
                            let reference = local_wire_ref(wire, &mut term_origins)?;
                            target_queue
                                .push_back((TargetRef::Local(wire.wire), Some(wire.clone())));
                            Ok(reference)
                        })
                        .collect::<Result<Vec<_>, ManifestError>>()
                })
                .transpose()?;
            Ok((
                name.clone(),
                ManifestArtifact {
                    wire_type: artifact.wire_type.clone(),
                    term_list,
                    family,
                    content_hash: artifact.content_hash,
                    layout: artifact.layout.clone(),
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, ManifestError>>()?;

    while !target_queue.is_empty() || !atom_queue.is_empty() {
        while let Some((target, local_wire)) = target_queue.pop_front() {
            let reference = match (&target, &local_wire) {
                (TargetRef::Local(_), Some(wire)) => local_wire_ref(wire, &mut term_origins)?,
                (TargetRef::Assumed(id), _) => {
                    let origin = LocalTermListOrigin::Assumed(id.clone());
                    let id = register_term_origin(origin, &mut term_origins)?;
                    ManifestTermListRef::Local(id)
                }
                (TargetRef::Imported { production_id, term_list_id }, _) => {
                    ManifestTermListRef::Imported {
                        production_id: production_id.clone(),
                        term_list_id: *term_list_id,
                    }
                }
                (TargetRef::Local(_), None) => return Err(ManifestError::MissingTarget(target)),
            };
            if term_records.contains_key(&reference) {
                continue;
            }
            let terms = match (&target, local_wire) {
                (TargetRef::Local(_), Some(wire)) => {
                    wire_terms.get(&wire).ok_or(ManifestError::MissingWire(wire))?
                }
                _ => target_terms
                    .get(&target)
                    .ok_or_else(|| ManifestError::MissingTarget(target.clone()))?,
            };
            let exported = export_terms(terms)?;
            for term in &terms.terms {
                atom_queue.extend(term.factors.iter().map(|factor| factor.atom.clone()));
            }
            term_records.insert(reference, exported);
        }

        while let Some(id) = atom_queue.pop_front() {
            let reference = atom_ref(&id, &mut atom_origins)?;
            if atom_records.contains_key(&reference) {
                continue;
            }
            let atom = atoms.get(&id).ok_or_else(|| ManifestError::MissingAtom(id.clone()))?;
            atom_queue.extend(atom.dependencies.iter().cloned());
            if let AtomClass::Derived { definition } = &atom.class {
                atom_queue.extend(definition_atoms(definition));
            }
            if let Some(preimage) = &atom.preimage_refs {
                atom_queue.push_back(preimage.uniform.clone());
                let local_wire = match (&preimage.target, &id) {
                    (TargetRef::Local(wire), AtomId::Local { instantiation_path, .. }) |
                    (TargetRef::Local(wire), AtomId::Overlay { instantiation_path, .. }) => {
                        Some(WireId { instantiation_path: instantiation_path.clone(), wire: *wire })
                    }
                    _ => None,
                };
                target_queue.push_back((preimage.target.clone(), local_wire));
            }
            atom_records
                .insert(reference, export_atom(atom, &mut atom_origins, &mut term_origins)?);
        }
    }

    let own_digest = (metadata.overlay_hash, metadata.assumption_hash);
    let mut interpretation_digests = metadata.interpretation_digests.clone();
    match interpretation_digests.entry(production_id.clone()) {
        Entry::Vacant(entry) => {
            entry.insert(own_digest);
        }
        Entry::Occupied(entry) if *entry.get() == own_digest => {}
        Entry::Occupied(_) => {
            return Err(ManifestError::InvalidOwnInterpretation { production: production_id });
        }
    }
    Ok(Manifest {
        format_version: SYMBOLIC_MANIFEST_VERSION,
        production_id,
        artifacts: exported_artifacts,
        atoms: atom_records,
        term_lists: term_records,
        overlay_hash: metadata.overlay_hash,
        assumption_hash: metadata.assumption_hash,
        assumption_digests: metadata.assumption_digests.clone(),
        interpretation_digests,
    })
}

fn local_wire_ref(
    wire: &WireId,
    origins: &mut BTreeMap<TermListId, LocalTermListOrigin>,
) -> Result<ManifestTermListRef, ManifestError> {
    let id = register_term_origin(LocalTermListOrigin::Wire(wire.clone()), origins)?;
    Ok(ManifestTermListRef::Local(id))
}

fn register_term_origin(
    origin: LocalTermListOrigin,
    origins: &mut BTreeMap<TermListId, LocalTermListOrigin>,
) -> Result<TermListId, ManifestError> {
    let id = term_list_id(&origin)?;
    match origins.entry(id) {
        Entry::Vacant(entry) => {
            entry.insert(origin);
        }
        Entry::Occupied(entry) if entry.get() == &origin => {}
        Entry::Occupied(_) => return Err(ManifestError::HashCollision(format!("{id:?}"))),
    }
    Ok(id)
}

fn atom_ref(
    id: &AtomId,
    origins: &mut BTreeMap<ManifestAtomId, AtomId>,
) -> Result<ManifestAtomRef, ManifestError> {
    if let AtomId::Imported { production_id, manifest_atom_id } = id {
        return Ok(ManifestAtomRef::Imported {
            production_id: production_id.clone(),
            manifest_atom_id: *manifest_atom_id,
        });
    }
    let manifest_id = manifest_atom_id(id)?;
    match origins.entry(manifest_id) {
        Entry::Vacant(entry) => {
            entry.insert(id.clone());
        }
        Entry::Occupied(entry) if entry.get() == id => {}
        Entry::Occupied(_) => return Err(ManifestError::HashCollision(format!("{manifest_id:?}"))),
    }
    Ok(ManifestAtomRef::Local(manifest_id))
}

fn export_atom(
    atom: &Atom,
    atom_origins: &mut BTreeMap<ManifestAtomId, AtomId>,
    term_origins: &mut BTreeMap<TermListId, LocalTermListOrigin>,
) -> Result<ManifestAtom, ManifestError> {
    let class = match &atom.class {
        AtomClass::Source => ManifestAtomClass::Source,
        AtomClass::Assumed => ManifestAtomClass::Assumed,
        AtomClass::Derived { definition } => {
            ManifestAtomClass::Derived { definition: export_definition(definition, atom_origins)? }
        }
    };
    let dependencies =
        atom.dependencies.iter().map(|id| atom_ref(id, atom_origins)).collect::<Result<_, _>>()?;
    let preimage_refs = atom
        .preimage_refs
        .as_ref()
        .map(|preimage| {
            let target = match &preimage.target {
                TargetRef::Imported { production_id, term_list_id } => {
                    ManifestTermListRef::Imported {
                        production_id: production_id.clone(),
                        term_list_id: *term_list_id,
                    }
                }
                TargetRef::Assumed(id) => ManifestTermListRef::Local(register_term_origin(
                    LocalTermListOrigin::Assumed(id.clone()),
                    term_origins,
                )?),
                TargetRef::Local(wire) => {
                    let path = match &atom.id {
                        AtomId::Local { instantiation_path, .. } |
                        AtomId::Overlay { instantiation_path, .. } => instantiation_path.clone(),
                        _ => Vec::new(),
                    };
                    local_wire_ref(&WireId { instantiation_path: path, wire: *wire }, term_origins)?
                }
            };
            Ok(ManifestPreimageRefs { uniform: atom_ref(&preimage.uniform, atom_origins)?, target })
        })
        .transpose()?;
    Ok(ManifestAtom {
        class,
        kind: atom.kind.clone(),
        matrix_type: atom.matrix_type.clone(),
        dependencies,
        preimage_refs,
    })
}

fn export_definition(
    definition: &DefExpr,
    atom_origins: &mut BTreeMap<ManifestAtomId, AtomId>,
) -> Result<ManifestDefExpr, ManifestError> {
    Ok(match definition {
        DefExpr::TermList(terms) => ManifestDefExpr::TermList(export_terms(terms)?),
        DefExpr::Tensor { left, right } => ManifestDefExpr::Tensor {
            left: atom_ref(left, atom_origins)?,
            right: atom_ref(right, atom_origins)?,
        },
        DefExpr::Concat { inputs, axis } => ManifestDefExpr::Concat {
            inputs: inputs
                .iter()
                .map(|input| atom_ref(input, atom_origins))
                .collect::<Result<_, _>>()?,
            axis: *axis,
        },
        DefExpr::Reshape { input, rows, columns } => ManifestDefExpr::Reshape {
            input: atom_ref(input, atom_origins)?,
            rows: *rows,
            columns: *columns,
        },
        DefExpr::ModDownImage { source, source_modulus, target_modulus } => {
            ManifestDefExpr::ModDownImage {
                source: atom_ref(source, atom_origins)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
        DefExpr::ModUpLift { source, source_modulus, target_modulus } => {
            ManifestDefExpr::ModUpLift {
                source: atom_ref(source, atom_origins)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
        DefExpr::Indicator { index_wire, branch } => {
            ManifestDefExpr::Indicator { index_wire: *index_wire, branch: *branch }
        }
        DefExpr::ModDownError { input, signal, source_modulus, target_modulus } => {
            ManifestDefExpr::ModDownError {
                input: *input,
                signal: export_terms(signal)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
        DefExpr::ModUpError { input, lifted, source_modulus, target_modulus } => {
            ManifestDefExpr::ModUpError {
                input: *input,
                lifted: export_terms(lifted)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
        DefExpr::Fold(terms) => ManifestDefExpr::Fold(export_terms(terms)?),
    })
}

fn definition_atoms(definition: &DefExpr) -> Vec<AtomId> {
    match definition {
        DefExpr::TermList(terms) | DefExpr::Fold(terms) => referenced_atoms(terms),
        DefExpr::Tensor { left, right } => vec![left.clone(), right.clone()],
        DefExpr::Concat { inputs, .. } => inputs.clone(),
        DefExpr::Reshape { input, .. } |
        DefExpr::ModDownImage { source: input, .. } |
        DefExpr::ModUpLift { source: input, .. } => vec![input.clone()],
        DefExpr::ModDownError { signal, .. } => referenced_atoms(signal),
        DefExpr::ModUpError { lifted, .. } => referenced_atoms(lifted),
        DefExpr::Indicator { .. } => Vec::new(),
    }
}

fn referenced_atoms(terms: &TermList) -> Vec<AtomId> {
    terms
        .terms
        .iter()
        .flat_map(|term| term.factors.iter().map(|factor| factor.atom.clone()))
        .collect()
}

fn export_terms(terms: &TermList) -> Result<ManifestTermList, ManifestError> {
    Ok(ManifestTermList {
        terms: terms
            .terms
            .iter()
            .map(|term| {
                Ok(ManifestTerm {
                    coefficient: term.coefficient.to_string(),
                    factors: term
                        .factors
                        .iter()
                        .map(|factor| {
                            Ok(ManifestFactor {
                                atom: match &factor.atom {
                                    AtomId::Imported { production_id, manifest_atom_id } => {
                                        ManifestAtomRef::Imported {
                                            production_id: production_id.clone(),
                                            manifest_atom_id: *manifest_atom_id,
                                        }
                                    }
                                    id => ManifestAtomRef::Local(manifest_atom_id(id)?),
                                },
                                view: factor.view.clone(),
                            })
                        })
                        .collect::<Result<_, ManifestError>>()?,
                })
            })
            .collect::<Result<_, ManifestError>>()?,
    })
}

pub fn merge_manifest_projections(
    manifests: &[Manifest],
) -> Result<BTreeMap<ProductionId, Manifest>, ManifestError> {
    let mut interpretations = BTreeMap::<ProductionId, InterpretationDigest>::new();
    for manifest in manifests {
        validate_manifest_header(manifest)?;
        for (production, digest) in &manifest.interpretation_digests {
            match interpretations.entry(production.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(*digest);
                }
                Entry::Occupied(entry) if entry.get() == digest => {}
                Entry::Occupied(_) => {
                    return Err(ManifestError::InterpretationConflict {
                        production: production.clone(),
                    });
                }
            }
        }
    }
    let mut merged = BTreeMap::<ProductionId, Manifest>::new();
    for manifest in manifests {
        match merged.entry(manifest.production_id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(manifest.clone());
            }
            Entry::Occupied(mut entry) => merge_projection(entry.get_mut(), manifest)?,
        }
    }
    Ok(merged)
}

fn validate_manifest_header(manifest: &Manifest) -> Result<(), ManifestError> {
    if manifest.format_version != SYMBOLIC_MANIFEST_VERSION {
        return Err(ManifestError::FormatVersion {
            expected: SYMBOLIC_MANIFEST_VERSION,
            actual: manifest.format_version,
        });
    }
    let own = manifest.interpretation_digests.get(&manifest.production_id).ok_or_else(|| {
        ManifestError::MissingOwnInterpretation { production: manifest.production_id.clone() }
    })?;
    if own != &(manifest.overlay_hash, manifest.assumption_hash) {
        return Err(ManifestError::InvalidOwnInterpretation {
            production: manifest.production_id.clone(),
        });
    }
    for digest in
        manifest.interpretation_digests.values().filter_map(|(_, assumption_hash)| *assumption_hash)
    {
        if !manifest.assumption_digests.contains(&digest) {
            return Err(ManifestError::MissingAssumptionProvenance {
                digest: serde_support::hex32::encode(&digest),
            });
        }
    }
    Ok(())
}

fn merge_projection(target: &mut Manifest, source: &Manifest) -> Result<(), ManifestError> {
    merge_records(&mut target.atoms, &source.atoms, "atom")?;
    merge_records(&mut target.term_lists, &source.term_lists, "term list")?;
    for (name, artifact) in &source.artifacts {
        match target.artifacts.entry(name.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(artifact.clone());
            }
            Entry::Occupied(entry) if entry.get() == artifact => {}
            Entry::Occupied(entry) => {
                return Err(ManifestError::ArtifactConflict {
                    production: Box::new(target.production_id.clone()),
                    artifact: name.clone(),
                    first: format!("{:?}", entry.get()).into_boxed_str(),
                    second: format!("{artifact:?}").into_boxed_str(),
                });
            }
        }
    }
    target.assumption_digests.extend(source.assumption_digests.iter().copied());
    for (production, digest) in &source.interpretation_digests {
        target.interpretation_digests.insert(production.clone(), *digest);
    }
    Ok(())
}

fn merge_records<K, V>(
    target: &mut BTreeMap<K, V>,
    source: &BTreeMap<K, V>,
    label: &str,
) -> Result<(), ManifestError>
where
    K: Clone + Ord + std::fmt::Debug,
    V: Clone + Eq + std::fmt::Debug,
{
    for (key, value) in source {
        match target.entry(key.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(value.clone());
            }
            Entry::Occupied(entry) if entry.get() == value => {}
            Entry::Occupied(entry) => {
                return Err(ManifestError::RecordConflict {
                    key: format!("{label} {key:?}"),
                    first: format!("{:?}", entry.get()).into_boxed_str(),
                    second: format!("{value:?}").into_boxed_str(),
                });
            }
        }
    }
    Ok(())
}

pub fn import_manifest(manifest: &Manifest) -> Result<ImportedManifest, ManifestError> {
    validate_manifest_header(manifest)?;
    let atom_id = |reference: &ManifestAtomRef| match reference {
        ManifestAtomRef::Local(id) => AtomId::Imported {
            production_id: manifest.production_id.clone(),
            manifest_atom_id: *id,
        },
        ManifestAtomRef::Imported { production_id, manifest_atom_id } => AtomId::Imported {
            production_id: production_id.clone(),
            manifest_atom_id: *manifest_atom_id,
        },
    };
    let target_ref = |reference: &ManifestTermListRef| match reference {
        ManifestTermListRef::Local(id) => {
            TargetRef::Imported { production_id: manifest.production_id.clone(), term_list_id: *id }
        }
        ManifestTermListRef::Imported { production_id, term_list_id } => TargetRef::Imported {
            production_id: production_id.clone(),
            term_list_id: *term_list_id,
        },
    };
    let imported_terms = manifest
        .term_lists
        .iter()
        .map(|(reference, terms)| {
            Ok((target_ref(reference), import_terms(terms, &manifest.atoms, &atom_id)?))
        })
        .collect::<Result<BTreeMap<_, _>, ManifestError>>()?;
    let mut atoms = AtomTable::default();
    for (reference, record) in &manifest.atoms {
        let id = atom_id(reference);
        let class = match &record.class {
            ManifestAtomClass::Source => AtomClass::Source,
            ManifestAtomClass::Assumed => AtomClass::Assumed,
            ManifestAtomClass::Derived { definition } => AtomClass::Derived {
                definition: import_definition(definition, &manifest.atoms, &atom_id)?,
            },
        };
        let dependencies = record
            .dependencies
            .iter()
            .map(|reference| {
                ensure_atom(reference, &manifest.atoms)?;
                Ok(atom_id(reference))
            })
            .collect::<Result<_, ManifestError>>()?;
        let preimage_refs = record
            .preimage_refs
            .as_ref()
            .map(|preimage| {
                ensure_atom(&preimage.uniform, &manifest.atoms)?;
                if !manifest.term_lists.contains_key(&preimage.target) {
                    return Err(ManifestError::MissingTermList(preimage.target.clone()));
                }
                Ok(PreimageRefs {
                    uniform: atom_id(&preimage.uniform),
                    target: target_ref(&preimage.target),
                })
            })
            .transpose()?;
        atoms.insert(Atom {
            id,
            class,
            kind: record.kind.clone(),
            matrix_type: record.matrix_type.clone(),
            dependencies,
            preimage_refs,
        });
    }
    let artifacts = manifest
        .artifacts
        .iter()
        .map(|(name, artifact)| {
            let terms = imported_terms
                .get(&target_ref(&artifact.term_list))
                .cloned()
                .ok_or_else(|| ManifestError::MissingTermList(artifact.term_list.clone()))?;
            let family = artifact
                .family
                .as_ref()
                .map(|members| {
                    members
                        .iter()
                        .map(|reference| {
                            imported_terms
                                .get(&target_ref(reference))
                                .cloned()
                                .ok_or_else(|| ManifestError::MissingTermList(reference.clone()))
                        })
                        .collect()
                })
                .transpose()?;
            Ok((
                name.clone(),
                ImportedArtifact {
                    wire_type: artifact.wire_type.clone(),
                    terms,
                    family,
                    content_hash: artifact.content_hash,
                    layout: artifact.layout.clone(),
                },
            ))
        })
        .collect::<Result<_, ManifestError>>()?;
    Ok(ImportedManifest {
        atoms,
        term_lists: imported_terms,
        artifacts,
        assumption_digests: manifest.assumption_digests.clone(),
        interpretation_digests: manifest.interpretation_digests.clone(),
    })
}

fn ensure_atom(
    reference: &ManifestAtomRef,
    records: &BTreeMap<ManifestAtomRef, ManifestAtom>,
) -> Result<(), ManifestError> {
    records
        .contains_key(reference)
        .then_some(())
        .ok_or_else(|| ManifestError::MissingManifestAtom(reference.clone()))
}

fn import_terms(
    terms: &ManifestTermList,
    records: &BTreeMap<ManifestAtomRef, ManifestAtom>,
    atom_id: &impl Fn(&ManifestAtomRef) -> AtomId,
) -> Result<TermList, ManifestError> {
    Ok(TermList {
        terms: terms
            .terms
            .iter()
            .map(|term| {
                Ok(Term {
                    coefficient: term
                        .coefficient
                        .parse::<BigInt>()
                        .map_err(|_| ManifestError::InvalidInteger(term.coefficient.clone()))?,
                    factors: term
                        .factors
                        .iter()
                        .map(|factor| {
                            ensure_atom(&factor.atom, records)?;
                            Ok(Factor { atom: atom_id(&factor.atom), view: factor.view.clone() })
                        })
                        .collect::<Result<_, ManifestError>>()?,
                })
            })
            .collect::<Result<_, ManifestError>>()?,
    })
}

fn import_definition(
    definition: &ManifestDefExpr,
    records: &BTreeMap<ManifestAtomRef, ManifestAtom>,
    atom_id: &impl Fn(&ManifestAtomRef) -> AtomId,
) -> Result<DefExpr, ManifestError> {
    let resolve = |reference: &ManifestAtomRef| {
        ensure_atom(reference, records)?;
        Ok(atom_id(reference))
    };
    Ok(match definition {
        ManifestDefExpr::TermList(terms) => {
            DefExpr::TermList(import_terms(terms, records, atom_id)?)
        }
        ManifestDefExpr::Tensor { left, right } => {
            DefExpr::Tensor { left: resolve(left)?, right: resolve(right)? }
        }
        ManifestDefExpr::Concat { inputs, axis } => DefExpr::Concat {
            inputs: inputs.iter().map(resolve).collect::<Result<_, _>>()?,
            axis: *axis,
        },
        ManifestDefExpr::Reshape { input, rows, columns } => {
            DefExpr::Reshape { input: resolve(input)?, rows: *rows, columns: *columns }
        }
        ManifestDefExpr::ModDownImage { source, source_modulus, target_modulus } => {
            DefExpr::ModDownImage {
                source: resolve(source)?,
                source_modulus: parse_integer(source_modulus)?,
                target_modulus: parse_integer(target_modulus)?,
            }
        }
        ManifestDefExpr::ModUpLift { source, source_modulus, target_modulus } => {
            DefExpr::ModUpLift {
                source: resolve(source)?,
                source_modulus: parse_integer(source_modulus)?,
                target_modulus: parse_integer(target_modulus)?,
            }
        }
        ManifestDefExpr::Indicator { index_wire, branch } => {
            DefExpr::Indicator { index_wire: *index_wire, branch: *branch }
        }
        ManifestDefExpr::ModDownError { input, signal, source_modulus, target_modulus } => {
            DefExpr::ModDownError {
                input: *input,
                signal: import_terms(signal, records, atom_id)?,
                source_modulus: parse_integer(source_modulus)?,
                target_modulus: parse_integer(target_modulus)?,
            }
        }
        ManifestDefExpr::ModUpError { input, lifted, source_modulus, target_modulus } => {
            DefExpr::ModUpError {
                input: *input,
                lifted: import_terms(lifted, records, atom_id)?,
                source_modulus: parse_integer(source_modulus)?,
                target_modulus: parse_integer(target_modulus)?,
            }
        }
        ManifestDefExpr::Fold(terms) => DefExpr::Fold(import_terms(terms, records, atom_id)?),
    })
}

fn parse_integer(value: &str) -> Result<BigInt, ManifestError> {
    value.parse().map_err(|_| ManifestError::InvalidInteger(value.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        atom::{AtomClass, AtomKind},
        types::{InstantiationFrame, NodeId, Port},
        ubound::UBound,
    };

    fn matrix_type() -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: BigInt::from(97), ring_dimension: 8, rows: 2, columns: 2 }
    }

    #[test]
    fn term_list_ids_include_instantiation_path_and_origin_tag() {
        let wire = WireRef { node: NodeId(1), port: Port(0) };
        let first = LocalTermListOrigin::Wire(WireId {
            instantiation_path: vec![InstantiationFrame { call: NodeId(2), loop_index: Some(0) }],
            wire,
        });
        let second = LocalTermListOrigin::Wire(WireId {
            instantiation_path: vec![InstantiationFrame { call: NodeId(2), loop_index: Some(1) }],
            wire,
        });
        let assumed = LocalTermListOrigin::Assumed(AssumedTermListId("wire".to_owned()));
        assert_ne!(term_list_id(&first).unwrap(), term_list_id(&second).unwrap());
        assert_ne!(term_list_id(&first).unwrap(), term_list_id(&assumed).unwrap());
    }

    #[test]
    fn projection_merge_rejects_artifact_conflicts() {
        let production = production_id(SpecHash([1; 32]), [2; 32]);
        let mut interpretation_digests = BTreeMap::new();
        interpretation_digests.insert(production.clone(), (None, None));
        let term = ManifestTermListRef::Local(TermListId([3; 32]));
        let artifact = ManifestArtifact {
            wire_type: matrix_type(),
            term_list: term.clone(),
            family: None,
            content_hash: None,
            layout: None,
        };
        let manifest = |layout: Option<&str>| Manifest {
            format_version: SYMBOLIC_MANIFEST_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "a".to_owned(),
                ManifestArtifact { layout: layout.map(str::to_owned), ..artifact.clone() },
            )]),
            atoms: BTreeMap::new(),
            term_lists: BTreeMap::from([(term.clone(), ManifestTermList::default())]),
            overlay_hash: None,
            assumption_hash: None,
            assumption_digests: BTreeSet::new(),
            interpretation_digests: interpretation_digests.clone(),
        };
        let error = merge_manifest_projections(&[manifest(None), manifest(Some("other"))])
            .expect_err("conflicting artifacts must fail");
        assert!(matches!(error, ManifestError::ArtifactConflict { .. }));
    }

    #[test]
    fn imported_record_keeps_its_origin() {
        let producer = production_id(SpecHash([3; 32]), [4; 32]);
        let intermediary = production_id(SpecHash([5; 32]), [6; 32]);
        let atom_id = ManifestAtomId([7; 32]);
        let atom_ref = ManifestAtomRef::Imported {
            production_id: producer.clone(),
            manifest_atom_id: atom_id,
        };
        let term_ref = ManifestTermListRef::Local(TermListId([8; 32]));
        let mut interpretation_digests = BTreeMap::new();
        interpretation_digests.insert(intermediary.clone(), (None, None));
        interpretation_digests.insert(producer.clone(), (None, None));
        let manifest = Manifest {
            format_version: SYMBOLIC_MANIFEST_VERSION,
            production_id: intermediary,
            artifacts: BTreeMap::from([(
                "x".to_owned(),
                ManifestArtifact {
                    wire_type: matrix_type(),
                    term_list: term_ref.clone(),
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            )]),
            atoms: BTreeMap::from([(
                atom_ref.clone(),
                ManifestAtom {
                    class: ManifestAtomClass::Source,
                    kind: AtomKind::Bounded { norm: UBound::one() },
                    matrix_type: matrix_type(),
                    dependencies: BTreeSet::new(),
                    preimage_refs: None,
                },
            )]),
            term_lists: BTreeMap::from([(
                term_ref,
                ManifestTermList {
                    terms: vec![ManifestTerm {
                        coefficient: "1".to_owned(),
                        factors: vec![ManifestFactor { atom: atom_ref, view: None }],
                    }],
                },
            )]),
            overlay_hash: None,
            assumption_hash: None,
            assumption_digests: BTreeSet::new(),
            interpretation_digests,
        };
        let imported = import_manifest(&manifest).expect("embedded producer record imports");
        assert!(imported.atoms.contains_key(&AtomId::Imported {
            production_id: producer,
            manifest_atom_id: atom_id,
        }));
    }

    #[test]
    fn complete_definition_round_trip() {
        let production = production_id(SpecHash([9; 32]), [10; 32]);
        let source = AtomId::Local { instantiation_path: Vec::new(), node: NodeId(1), port: 0 };
        let derived = AtomId::Local { instantiation_path: Vec::new(), node: NodeId(2), port: 0 };
        let mut atoms = AtomTable::default();
        atoms.insert(Atom {
            id: source.clone(),
            class: AtomClass::Source,
            kind: AtomKind::Bounded { norm: UBound::one() },
            matrix_type: matrix_type(),
            dependencies: BTreeSet::new(),
            preimage_refs: None,
        });
        atoms.insert(Atom {
            id: derived.clone(),
            class: AtomClass::Derived {
                definition: DefExpr::TermList(TermList::atom(source.clone())),
            },
            kind: AtomKind::Bounded { norm: UBound::one() },
            matrix_type: matrix_type(),
            dependencies: BTreeSet::from([source]),
            preimage_refs: None,
        });
        let wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(2), port: Port(0) },
        };
        let manifest = export_manifest(
            production,
            &BTreeMap::from([(
                "x".to_owned(),
                ExportArtifact {
                    wire: wire.clone(),
                    wire_type: matrix_type(),
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            )]),
            &BTreeMap::from([(wire, TermList::atom(derived))]),
            &BTreeMap::new(),
            &atoms,
            &ManifestMetadata::default(),
        )
        .expect("complete closure exports");
        assert_eq!(manifest.atoms.len(), 2);
        let imported = import_manifest(&manifest).expect("manifest imports");
        assert_eq!(imported.atoms.len(), 2);
    }

    #[test]
    fn manifest_json_uses_hex_digests_and_round_trips_reference_keyed_maps() {
        let production = production_id(SpecHash([1; 32]), [2; 32]);
        let atom_ref = ManifestAtomRef::Local(ManifestAtomId([3; 32]));
        let term_ref = ManifestTermListRef::Local(TermListId([4; 32]));
        let digest = [5; 32];
        let manifest = Manifest {
            format_version: SYMBOLIC_MANIFEST_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "x".to_owned(),
                ManifestArtifact {
                    wire_type: matrix_type(),
                    term_list: term_ref.clone(),
                    family: None,
                    content_hash: Some([6; 32]),
                    layout: None,
                },
            )]),
            atoms: BTreeMap::from([(
                atom_ref.clone(),
                ManifestAtom {
                    class: ManifestAtomClass::Source,
                    kind: AtomKind::Large,
                    matrix_type: matrix_type(),
                    dependencies: BTreeSet::new(),
                    preimage_refs: None,
                },
            )]),
            term_lists: BTreeMap::from([(
                term_ref,
                ManifestTermList {
                    terms: vec![ManifestTerm {
                        coefficient: "1".to_owned(),
                        factors: vec![ManifestFactor { atom: atom_ref, view: None }],
                    }],
                },
            )]),
            overlay_hash: Some(digest),
            assumption_hash: None,
            assumption_digests: BTreeSet::from([digest]),
            interpretation_digests: BTreeMap::from([(production, (Some(digest), None))]),
        };
        let json = serde_json::to_string(&manifest).expect("canonical JSON");
        assert!(json.contains(&format!("\"{}\"", "05".repeat(32))));
        assert!(json.contains(&format!("local:{}", "03".repeat(32))));
        assert!(!json.contains("[5,5,5,5"));
        assert_eq!(serde_json::from_str::<Manifest>(&json).expect("round trip"), manifest);
    }

    #[test]
    fn import_rejects_format_and_interpretation_disagreement() {
        let production = production_id(SpecHash([11; 32]), [12; 32]);
        let base = |overlay_hash| Manifest {
            format_version: SYMBOLIC_MANIFEST_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::new(),
            atoms: BTreeMap::new(),
            term_lists: BTreeMap::new(),
            overlay_hash,
            assumption_hash: None,
            assumption_digests: BTreeSet::new(),
            interpretation_digests: BTreeMap::from([(production.clone(), (overlay_hash, None))]),
        };
        let mut wrong_version = base(None);
        wrong_version.format_version += 1;
        assert!(matches!(
            import_manifest(&wrong_version),
            Err(ManifestError::FormatVersion { .. })
        ));

        let mut first = base(Some([1; 32]));
        first.assumption_hash = Some([3; 32]);
        first.assumption_digests.insert([3; 32]);
        first.interpretation_digests.insert(production.clone(), (Some([1; 32]), Some([3; 32])));
        let mut second = base(Some([2; 32]));
        second.assumption_hash = Some([4; 32]);
        second.assumption_digests.insert([4; 32]);
        second.interpretation_digests.insert(production.clone(), (Some([2; 32]), Some([4; 32])));
        let conflict =
            merge_manifest_projections(&[first, second]).expect_err("interpretations disagree");
        assert!(matches!(conflict, ManifestError::InterpretationConflict { .. }));
    }

    #[test]
    fn assumed_preimage_target_round_trips_and_complete_closure_is_embedded() {
        let production = production_id(SpecHash([13; 32]), [14; 32]);
        let uniform = AtomId::Virtual { name: "A".to_owned() };
        let preimage = AtomId::Virtual { name: "K".to_owned() };
        let error = AtomId::Virtual { name: "e".to_owned() };
        let assumed_id = AssumedTermListId("target".to_owned());
        let mut atoms = AtomTable::default();
        for (id, kind) in [
            (uniform.clone(), AtomKind::Large),
            (preimage.clone(), AtomKind::Bounded { norm: UBound::one() }),
            (error.clone(), AtomKind::Bounded { norm: UBound::one() }),
        ] {
            atoms.insert(Atom {
                id: id.clone(),
                class: AtomClass::Assumed,
                kind,
                matrix_type: matrix_type(),
                dependencies: BTreeSet::new(),
                preimage_refs: (id == preimage).then(|| PreimageRefs {
                    uniform: uniform.clone(),
                    target: TargetRef::Assumed(assumed_id.clone()),
                }),
            });
        }
        let target = TermList {
            terms: vec![
                Term {
                    coefficient: BigInt::from(1),
                    factors: vec![Factor { atom: uniform.clone(), view: None }],
                },
                Term {
                    coefficient: BigInt::from(1),
                    factors: vec![Factor { atom: error.clone(), view: None }],
                },
            ],
        };
        let wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(1), port: Port(0) },
        };
        let manifest = export_manifest(
            production.clone(),
            &BTreeMap::from([(
                "k".to_owned(),
                ExportArtifact {
                    wire: wire.clone(),
                    wire_type: matrix_type(),
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            )]),
            &BTreeMap::from([(wire, TermList::atom(preimage.clone()))]),
            &BTreeMap::from([(TargetRef::Assumed(assumed_id), target.clone())]),
            &atoms,
            &ManifestMetadata::default(),
        )
        .expect("export");
        assert_eq!(manifest.atoms.len(), 3);
        assert_eq!(manifest.term_lists.len(), 2);

        let imported = import_manifest(&manifest).expect("import");
        let imported_preimage = AtomId::Imported {
            production_id: production.clone(),
            manifest_atom_id: manifest_atom_id(&preimage).unwrap(),
        };
        let refs = imported
            .atoms
            .get(&imported_preimage)
            .and_then(|atom| atom.preimage_refs.as_ref())
            .expect("preimage refs");
        let TargetRef::Imported { production_id: target_production, .. } = &refs.target else {
            panic!("assumed target must import into the producer namespace")
        };
        assert_eq!(target_production, &production);
        assert_eq!(
            imported.term_lists.get(&refs.target).expect("target"),
            &TermList {
                terms: target
                    .terms
                    .into_iter()
                    .map(|mut term| {
                        for factor in &mut term.factors {
                            factor.atom = AtomId::Imported {
                                production_id: production.clone(),
                                manifest_atom_id: manifest_atom_id(&factor.atom).unwrap(),
                            };
                        }
                        term
                    })
                    .collect(),
            }
        );
    }

    #[test]
    fn projection_merge_unions_disjoint_artifacts_independently_of_input_order() {
        let production = production_id(SpecHash([15; 32]), [16; 32]);
        let digest = BTreeMap::from([(production.clone(), (None, None))]);
        let manifest = |name: &str, id: u8| {
            let term = ManifestTermListRef::Local(TermListId([id; 32]));
            Manifest {
                format_version: SYMBOLIC_MANIFEST_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([(
                    name.to_owned(),
                    ManifestArtifact {
                        wire_type: matrix_type(),
                        term_list: term.clone(),
                        family: None,
                        content_hash: None,
                        layout: None,
                    },
                )]),
                atoms: BTreeMap::new(),
                term_lists: BTreeMap::from([(term, ManifestTermList::default())]),
                overlay_hash: None,
                assumption_hash: None,
                assumption_digests: BTreeSet::new(),
                interpretation_digests: digest.clone(),
            }
        };
        let left =
            merge_manifest_projections(&[manifest("a", 1), manifest("b", 2)]).expect("merge");
        let right =
            merge_manifest_projections(&[manifest("b", 2), manifest("a", 1)]).expect("merge");
        assert_eq!(left, right);
        assert_eq!(left[&production].artifacts.len(), 2);
    }
}
