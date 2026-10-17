//! Bounded device residency for large PRF selector and coordinate inputs.
//!
//! These are sampled input data, not another cryptographic graph. Large integer families are
//! transported as indexed artifacts, so the GPU imports only the members consumed by a lane.
//! Inputs whose combined size fits the resident budget retain the ordinary runtime-input path.
//! Secret selectors use a separate
//! temporary production, discarded after obfuscation, and never enter the public obfuscation.
use super::{
    Dio2Error, StreamingFunction,
    scheme::{Kind, derive, selector_input, table_input},
    step::ChainProgram,
};
use mxx_backends::{ArtifactKey, ArtifactPayload, ArtifactStore, RuntimeValue, SessionStore};
use mxx_dsl::{DslContext, Family, FamilyType, Int};
use mxx_ir_core::{
    IntExpr,
    artifact::{
        ArtifactAvailability, ArtifactType, Manifest, ManifestArtifact, ProductionId, SpecHash,
    },
    encoding::IR_VERSION,
};
use std::collections::BTreeMap;

/// A signed one-word integer occupies two device words. Keep all ordinary PRF inputs together below
/// 64 MiB.
const RESIDENT_MEMBERS: usize = (64 << 20) / 16;

#[derive(Clone)]
pub(crate) struct InputStorage {
    pub public: Manifest,
    secret: Manifest,
}

fn error(error: impl std::error::Error + Send + Sync + 'static) -> Dio2Error {
    mxx_exponent_lut::api::ApiError::Scheme { source: Box::new(error) }.into()
}

impl InputStorage {
    pub fn new(
        hash: &[u8; 32],
        function: &StreamingFunction,
        programs: &[(Kind, &ChainProgram)],
        shape_only: bool,
    ) -> Self {
        let nonce = if shape_only { derive(hash, "input-shape", &[]) } else { rand::random() };
        let manifest = |role| Manifest {
            ir_version: IR_VERSION,
            production_id: ProductionId {
                spec_hash: SpecHash(derive(hash, role, &[])),
                execution_nonce: nonce,
            },
            artifacts: BTreeMap::new(),
        };
        let mut result =
            Self { public: manifest("public-prf-inputs"), secret: manifest("secret-prf-inputs") };
        // The budget is global: thousands of individually small buckets must not
        // bypass it. Count both the coordinate table and the secret selector.
        let total_members = programs
            .iter()
            .flat_map(|(kind, program)| kind.prfs(program))
            .flat_map(|(_, prf)| prf.selector_families().into_iter())
            .fold(0u128, |total, selector| {
                total.saturating_add(
                    ((selector.members as u128) << function.eval_chunk_bits).saturating_mul(2),
                )
            });
        if total_members <= RESIDENT_MEMBERS as u128 {
            return result;
        }
        for (kind, program) in programs {
            for (role, prf) in kind.prfs(program) {
                for (bucket, selector) in prf.selector_families().iter().enumerate() {
                    let count = (selector.members as u128) << function.eval_chunk_bits;
                    let descriptor = ManifestArtifact {
                        artifact_type: ArtifactType::Int,
                        family_count: Some(count),
                        availability: ArtifactAvailability::Transferred,
                        layout: None,
                    };
                    result.public.artifacts.insert(table_input(role, bucket), descriptor.clone());
                    result.secret.artifacts.insert(selector_input(role, bucket), descriptor);
                }
            }
        }
        result
    }

    pub fn family(&self, name: String, count: IntExpr) -> Result<Family<Int>, Dio2Error> {
        let source =
            [&self.public, &self.secret].into_iter().find(|m| m.artifacts.contains_key(&name));
        match source {
            None => Ok(mxx_dsl::int_family_input(name, count)),
            Some(source) => Ok(DslContext::new("dio2-input-data").artifact_input(
                source.production_id.clone(),
                name,
                FamilyType { element: mxx_dsl::IntType, count, member_slot: None },
                ArtifactAvailability::Transferred,
            )?),
        }
    }

    pub fn manifests(&self, secret: bool) -> BTreeMap<ProductionId, Manifest> {
        std::iter::once(&self.public)
            .chain(secret.then_some(&self.secret))
            .filter(|m| !m.artifacts.is_empty())
            .map(|m| (m.production_id.clone(), m.clone()))
            .collect()
    }

    /// Move the sampled host data to the store without uploading whole families to the GPU.
    /// Input sampling and its distribution are unchanged. The store may stream to disk.
    pub fn publish<S: ArtifactStore>(
        &self,
        store: &mut S,
        public: &mut BTreeMap<String, RuntimeValue>,
        secret: &mut BTreeMap<String, RuntimeValue>,
    ) -> Result<(), Dio2Error> {
        for (manifest, values) in [(&self.public, public), (&self.secret, secret)] {
            if manifest.artifacts.is_empty() {
                continue;
            }
            for (name, descriptor) in &manifest.artifacts {
                let Some(RuntimeValue::IndexedFamily { values: members, .. }) = values.remove(name)
                else {
                    return Err(Dio2Error::Input("missing PRF input family"));
                };
                if Some(members.len() as u128) != descriptor.family_count {
                    return Err(Dio2Error::Input("PRF input family count mismatch"));
                }
                for (index, member) in members.into_iter().enumerate() {
                    let RuntimeValue::Int(value) = member else {
                        return Err(Dio2Error::Input("a PRF input is not an integer"));
                    };
                    store
                        .store(
                            ArtifactKey {
                                production: manifest.production_id.clone(),
                                name: name.clone(),
                                index: Some(index),
                            },
                            &ArtifactType::Int,
                            ArtifactAvailability::Transferred,
                            None,
                            ArtifactPayload::Bytes(value.to_signed_bytes_le()),
                        )
                        .map_err(error)?;
                }
            }
            store.store_manifest(manifest.clone()).map_err(error)?;
        }
        Ok(())
    }

    pub fn discard_secret<S: SessionStore>(&self, store: &mut S) -> Result<(), Dio2Error> {
        if !self.secret.artifacts.is_empty() {
            store.discard_session(&self.secret.production_id).map_err(error)?;
        }
        Ok(())
    }

    /// Public metadata only. Do not retain secret input descriptors in the obfuscation.
    pub fn public_only(mut self) -> Self {
        self.secret.artifacts.clear();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_backends::MemoryArtifactStore;

    #[test]
    fn test_input_budget_counts_all_families_and_both_roles() {
        let (parameters, mut function) = super::super::test_support::toy();
        let plan = super::super::step::StepPlan::new(&parameters, &function).unwrap();
        let program = Kind::Last.program(&parameters, &function, &plan).unwrap();
        let families = program.smudge.selector_families();
        let members: u128 = families.iter().map(|family| family.members).sum();
        function.eval_chunk_bits = 0;
        assert!(2 * members <= RESIDENT_MEMBERS as u128);
        let small = InputStorage::new(&[0; 32], &function, &[(Kind::Last, &program)], true);
        assert!(small.manifests(true).is_empty());

        // Cross the combined budget while every individual coordinate/selector family
        // still fits. This catches the former per-family budget check.
        while (2 * members) << function.eval_chunk_bits <= RESIDENT_MEMBERS as u128 {
            function.eval_chunk_bits += 1;
        }
        assert!(families.iter().all(|family| {
            family.members << function.eval_chunk_bits <= RESIDENT_MEMBERS as u128
        }));
        let large = InputStorage::new(&[0; 32], &function, &[(Kind::Last, &program)], true);
        assert_eq!(large.public.artifacts.len(), families.len());
        assert_eq!(large.secret.artifacts.len(), families.len());
        for (bucket, family) in families.iter().enumerate() {
            let count = Some((family.members as u128) << function.eval_chunk_bits);
            assert_eq!(large.public.artifacts[&table_input("output", bucket)].family_count, count);
            assert_eq!(
                large.secret.artifacts[&selector_input("output", bucket)].family_count,
                count
            );
        }
    }

    #[test]
    fn test_input_artifacts_preserve_integers_and_discard_only_secrets() {
        let manifest = |role| Manifest {
            ir_version: IR_VERSION,
            production_id: ProductionId {
                spec_hash: SpecHash(derive(&[0; 32], role, &[])),
                execution_nonce: rand::random(),
            },
            artifacts: BTreeMap::from([(
                role.to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Int,
                    family_count: Some(3),
                    availability: ArtifactAvailability::Transferred,
                    layout: None,
                },
            )]),
        };
        let storage = InputStorage { public: manifest("public"), secret: manifest("secret") };
        let integers = vec![(-129).into(), 0.into(), 256.into()];
        let mut public =
            BTreeMap::from([("public".into(), RuntimeValue::integer_values(integers.clone()))]);
        let mut secret =
            BTreeMap::from([("secret".into(), RuntimeValue::integer_values(integers.clone()))]);
        let mut store = MemoryArtifactStore::default();
        storage.publish(&mut store, &mut public, &mut secret).unwrap();
        assert!(public.is_empty() && secret.is_empty());
        for manifest in [&storage.public, &storage.secret] {
            let (name, descriptor) = manifest.artifacts.first_key_value().unwrap();
            for (index, expected) in integers.iter().enumerate() {
                let key = ArtifactKey {
                    production: manifest.production_id.clone(),
                    name: name.clone(),
                    index: Some(index),
                };
                let ArtifactPayload::Bytes(bytes) = store.load(&key, descriptor).unwrap() else {
                    panic!("integer payload")
                };
                assert_eq!(&num_bigint::BigInt::from_signed_bytes_le(&bytes), expected);
            }
        }
        storage.discard_secret(&mut store).unwrap();
        assert!(store.load_manifest(&storage.secret.production_id).is_err());
        assert!(store.load_manifest(&storage.public.production_id).is_ok());
        assert!(storage.public_only().secret.artifacts.is_empty());
    }
}
