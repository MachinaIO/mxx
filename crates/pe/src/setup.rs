//! Setup: the anchor `B = [B^(1); B^(2)]` with the trapdoor of `B^(1)`, the
//! message row `d = (d_1, d_2)`, and the class's Dynamic setup.
//!
//! `B^(1)` is the public row of the one-row statistical TrapGen `[ā | g - ā T]`,
//! `1 x m_B` with `m_B = m̄ + D`; every preimage under it is sampled at the key width with
//! the key cutoff, which the trapdoor's type carries. Its trapdoor stays in the secret
//! setup production, which only [`crate::keygen`] imports; a second
//! production republishes `B^(1)` alone, and the public key names only that
//! one. `B^(2)` and `d` are expanded from the public hash key under setup
//! tags, as are the Dynamic setup's `A_0`, `H_k`, and companion matrices.
use crate::{
    PeError, PeParameters,
    branching_program::{ATTRIBUTE_COORDINATE, INITIAL_STATE_COORDINATE},
};
use mxx_backends::{SessionStore, poly::PolyParams};
use mxx_dsl::{Bytes, DslContext, HashTag, Mat, Ring, Trapdoor};
use mxx_exponent_lut::{
    api::{self, StageExecution, StageGraph},
    program::{AttributeKnowledge, InputBinding, ProgramInputDeclaration},
    rns_bgg::RnsBggDynamicSetup,
};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    artifact::{ArtifactAvailability, Manifest, ProductionId},
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Graph input of the public hash key in the scheme's own graphs.
pub(crate) const HASH_KEY_INPUT: &str = "mxx-pe/hash-key";
/// The trapdoor output of the secret setup production: `.0` is `B^(1)`,
/// `.1` its trapdoor.
const ANCHOR_TRAPDOOR: &str = "mxx-pe/anchor";
/// `B^(1)` in the public setup production.
const ANCHOR_PUBLIC: &str = "mxx-pe/anchor-public";

pub(crate) fn digest(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(domain);
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    digest.finalize().into()
}

/// The master public key: the parameters, the public seed, the Dynamic
/// setup, and the production publishing `B^(1)`.
#[derive(Clone, Debug)]
pub struct PePublicKey {
    parameters: PeParameters,
    context: [u8; 32],
    hash_key: [u8; 32],
    setup: RnsBggDynamicSetup,
    anchor: (ProductionId, Manifest),
    identity: [u8; 32],
}

/// The master secret key: the production holding the trapdoor of `B^(1)`.
#[derive(Clone, Debug)]
pub struct PeMasterSecretKey {
    public: [u8; 32],
    trapdoor: (ProductionId, Manifest),
}

impl PeMasterSecretKey {
    /// The identity of the public key this key belongs to.
    pub const fn public_key(&self) -> &[u8; 32] {
        &self.public
    }

    pub(crate) const fn trapdoor_production(&self) -> &(ProductionId, Manifest) {
        &self.trapdoor
    }
}

impl PePublicKey {
    pub const fn parameters(&self) -> &PeParameters {
        &self.parameters
    }

    pub const fn context(&self) -> &[u8; 32] {
        &self.context
    }

    pub const fn hash_key(&self) -> &[u8; 32] {
        &self.hash_key
    }

    /// The Dynamic setup of the class.
    pub const fn setup(&self) -> &RnsBggDynamicSetup {
        &self.setup
    }

    /// The identity of this public key, which every ciphertext and key
    /// commits to.
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    pub(crate) const fn anchor_production(&self) -> &(ProductionId, Manifest) {
        &self.anchor
    }

    /// The setup ring.
    pub(crate) fn ring(&self) -> Result<Ring, PeError> {
        Ok(self.parameters.rns().ring(0)?)
    }

    /// `B^(1)`, imported from the public setup production.
    pub(crate) fn anchor_row(&self) -> Result<Mat, PeError> {
        Ok(self.ring()?.artifact_input(
            self.anchor.0.clone(),
            ANCHOR_PUBLIC,
            (1, self.parameters.anchor_columns()),
            ArtifactAvailability::Transferred,
        ))
    }

    /// The hash tag of the public matrix `domain`. It names the parameters through the context;
    /// the hash key, which keys every expansion, separates two setups of one parameter set.
    fn tag(&self, domain: &str) -> HashTag {
        let mut tag = HashTag::from(b"mxx-pe".to_vec());
        tag.push(self.context.iter().map(|byte| format!("{byte:02x}")).collect::<String>());
        tag.push(domain.to_owned());
        tag
    }

    /// `B^(2)`, expanded from the hash key.
    pub(crate) fn anchor_second_row(&self, hash_key: Bytes) -> Result<Mat, PeError> {
        Ok(self.ring()?.hash_matrix(
            hash_key,
            self.tag("anchor-second-row"),
            (1, self.parameters.anchor_columns()),
        ))
    }

    /// The message row `(d_1, d_2)`, expanded from the hash key.
    pub(crate) fn message_row(&self, hash_key: Bytes) -> Result<(Mat, Mat), PeError> {
        let ring = self.ring()?;
        let entry = |index: u64| {
            let mut tag = self.tag("message-row");
            tag.push(IntExpr::constant(index));
            ring.hash_matrix(hash_key.clone(), tag, (1, 1))
        };
        Ok((entry(0), entry(1)))
    }

    /// The hash key as a graph input and its runtime binding.
    pub(crate) fn hash_key_input(
        &self,
        runtime: &mut BTreeMap<String, mxx_backends::RuntimeValue>,
    ) -> Result<Bytes, PeError> {
        runtime.insert(
            HASH_KEY_INPUT.to_owned(),
            mxx_backends::RuntimeValue::Bytes(self.hash_key.as_slice().into()),
        );
        Ok(self.ring()?.bytes_input(HASH_KEY_INPUT, 32))
    }

    /// The trapdoor of `B^(1)` imported from the secret setup production of
    /// `master`.
    pub(crate) fn trapdoor(&self, master: &PeMasterSecretKey) -> Result<Trapdoor, PeError> {
        if master.public != self.identity {
            return Err(PeError::Mismatch);
        }
        let level = self.parameters.level();
        Ok(self.ring()?.trapdoor_artifact_input_with_width(
            master.trapdoor.0.clone(),
            format!("{ANCHOR_TRAPDOOR}.1"),
            1,
            self.parameters.instantiation(),
            self.parameters.trapdoor_sigma_expr()?,
            1u64 << level.base_bits(),
            level.modulus_digits(),
            self.parameters.preimage_sigma_expr()?,
            num_bigint::BigInt::from(self.parameters.key_cutoff().clone()),
        ))
    }
}

/// `Setup(1^lambda, F)`: sample the trapdoor of `B^(1)` and fix the class's
/// public material under a fresh public seed.
pub fn setup<E, S>(
    parameters: &PeParameters,
    execution: &mut E,
    store: &mut S,
) -> Result<(PePublicKey, PeMasterSecretKey), PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    // The public hash key makes every setup's hash-expanded matrices fresh; the context only
    // names the parameters, so the stage graphs of one parameter set do not depend on the setup.
    let hash_key: [u8; 32] = rand::random();
    let context = digest(b"mxx-pe/setup-context/v2", &[&parameters.identity()]);
    let class = parameters.class();
    let rns = parameters.rns();
    let dynamic = RnsBggDynamicSetup::new(
        rns,
        context,
        &hash_key,
        BTreeMap::from([(
            INITIAL_STATE_COORDINATE,
            ProgramInputDeclaration::new(
                InputBinding::Scalar(()),
                AttributeKnowledge::PrivatePayload,
            ),
        )]),
        // The lookup packages, if a lookup step reads them, whose bottom rows the encryption
        // prelude exports.
        class
            .has_lookup_packages()
            .then(|| {
                (
                    ATTRIBUTE_COORDINATE,
                    InputBinding::Family(IntExpr::constant(class.attribute_length())),
                )
            })
            .into_iter()
            .collect(),
        class.layout(),
        parameters.distributions(),
    )?;
    let dynamic = match parameters.lut_radix() {
        Some(radix) => dynamic.with_lut_radix(radix)?,
        None => dynamic,
    };
    let dynamic = match parameters.compact_extension() {
        Some(extension) => dynamic.with_compact_encodings(rns, extension.to_vec())?,
        None => dynamic,
    };
    let ring = rns.ring(0)?;
    let level = parameters.level();
    let trapdoor = ring.sample_trapdoor_with_width(
        1,
        parameters.instantiation(),
        parameters.trapdoor_sigma_expr()?,
        1u64 << level.base_bits(),
        level.modulus_digits(),
        parameters.preimage_sigma_expr()?,
        num_bigint::BigInt::from(parameters.key_cutoff().clone()),
    );
    let columns = trapdoor.public_matrix().matrix_type().columns.clone();
    if columns != IntExpr::constant(parameters.anchor_columns()) {
        return Err(PeError::Parameters("the TrapGen width is not the parameters' m_B".into()));
    }
    let graph = DslContext::new("mxx-pe/setup-trapdoor")
        .transferred_output(ANCHOR_TRAPDOOR, trapdoor)?
        .build()?;
    let secret = api::execute_artifacts(
        StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
        &ParamEnv::default(),
        (),
        digest(b"mxx-pe/setup-trapdoor/v2", &[&context, &hash_key]),
        execution,
        store,
        |_| Ok(()),
    )?;
    let anchor = ring.artifact_input(
        secret.production().clone(),
        format!("{ANCHOR_TRAPDOOR}.0"),
        (1, parameters.anchor_columns()),
        ArtifactAvailability::Transferred,
    );
    let graph = DslContext::new("mxx-pe/setup-public")
        .transferred_output(ANCHOR_PUBLIC, anchor)?
        .build()?;
    let public = api::execute_artifacts(
        StageGraph::new(
            graph,
            BTreeMap::new(),
            BTreeMap::from([(secret.production().clone(), secret.manifest().clone())]),
        ),
        &ParamEnv::default(),
        (),
        digest(b"mxx-pe/setup-public/v2", &[&context, &hash_key]),
        execution,
        store,
        |_| Ok(()),
    )?;
    let anchor = (public.production().clone(), public.manifest().clone());
    let identity = digest(
        b"mxx-pe/public-key/v1",
        &[
            &parameters.identity(),
            &context,
            &dynamic.identity(),
            &anchor.0.spec_hash.0,
            &anchor.0.execution_nonce,
        ],
    );
    let master = PeMasterSecretKey {
        public: identity,
        trapdoor: (secret.production().clone(), secret.manifest().clone()),
    };
    Ok((
        PePublicKey {
            parameters: parameters.clone(),
            context,
            hash_key,
            setup: dynamic,
            anchor,
            identity,
        },
        master,
    ))
}
