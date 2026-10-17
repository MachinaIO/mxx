//! Key generation: `KeyGen(msk, F)`.
//!
//! The predicate is compiled and its output key `A_F` evaluated from setup
//! matrices alone (the exponent-lut Dynamic preprocessing with the gadget
//! output use), then a short `k` with `M_F k = y = d_1 delta_1` is sampled
//! for the family matrix
//!
//! `M_F = [ (delta_i (x) sigma_i(B^(1)))_i | (delta_i (x) sigma_i(H_k^(1)) -
//! delta_(ik) (x) G^(1))_(i,k) | delta_1 (x) A_F^(1) ]`,
//!
//! rows indexed by `H` in its canonical order (the identity first), blocks
//! ordered as anchors by `i`, helpers by `(i, k)` lexicographically, then
//! the output block; `ik` is the group product modulo `2n`. `M_F` is never
//! formed: SampleLeft on `M_F = [diag(sigma_i(B^(1))) | E]` samples the one
//! extension `v` (every helper and output coordinate) at the key width, then
//! solves each row `i` for its anchor block with the existing trapdoor of
//! `B^(1)`: `B^(1) k~_i = sigma_i^-1(u_i)` for the residual
//! `u_i = y_i - (E v)_i`, and `k_i = sigma_i(k~_i)`. The anchor blocks are
//! stored in these inverse coordinates; decryption contracts them as
//! `sigma_i((b - B^(2)) k~_i)`.
//!
//! With compact encodings (the manuscript's Appendix C), the family matrix is
//! `M~_F = [ L | (delta_i (x) sigma_i(B''^(1)))_i | I_W | delta_1 (x) A_F^(1) ]`
//! for `B'' = Dc(B_c)` (see [`crate::setup`]); both new blocks are extension
//! coordinates, sampled with `v` at the key width: `v''_i` (`m_c D` per row)
//! and `v_iota` (one per row). The block `I_W` serves only the security
//! proof; decryption's row is zero there.
//!
//! Helper samples are streamed in one compact matrix per group row before contraction.
//! Each contraction lane reads one row and contributes a
//! `W x 1` scalar residual column to a fused sum. Only the selected helper
//! column of each product contributes; no expanded extension family is retained. Compressed lanes
//! likewise export compact blocks and retain only their scalar residual contributions.
use crate::{
    BranchingProgram, CompiledPredicate, PeError, PeMasterSecretKey, PePublicKey, setup::digest,
};
use mxx_backends::{
    SessionStore,
    poly::{PolyParams, dcrt::poly::DCRTPoly},
};
use mxx_dsl::{DslContext, Int, Mat, Ring, SmallMatrix};
use mxx_exponent_lut::{
    api::{self, PlaintextInputs, StageArtifacts, StageExecution, StageGraph},
    program::{Operand, ProgramData, RhsId, WireId},
    rns_bgg::{RnsBggPreprocessOnlineEvalDescriptor, RnsOutputUse, RnsPreprocessBuild},
};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    artifact::ArtifactAvailability,
    node::{ConcatAxis, IndexRange},
};
use std::collections::BTreeMap;

/// The anchor samples `k~_i`, stored as a streamed bounded-matrix family
/// indexed by the row of `H` (see [`anchor_keys`]).
pub(crate) const ANCHOR_KEY: &str = "mxx-pe/key/anchor";
/// The helper blocks `v_(i,k)`: one compact `width x |P|` matrix per row `i`.
pub(crate) const HELPER_KEY: &str = "mxx-pe/key/helper";
/// The output block `v_F`, stored as a compact bounded matrix.
pub(crate) const OUTPUT_KEY: &str = "mxx-pe/key/output";
/// With compact encodings, the blocks `v''_i` of `delta_i (x) sigma_i(B''^(1))` (a family of
/// `W` members of `m_c D x 1`).
pub(crate) const COMPRESSED_KEY: &str = "mxx-pe/key/compressed";
/// With compact encodings, the block `v_iota` of `I_W` (a family of `W` members of `1 x 1`).
pub(crate) const IDENTITY_KEY: &str = "mxx-pe/key/identity";

/// Serialization restores internal handles written by this crate; referenced store
/// productions must remain available. It is not an untrusted-input format.
///
/// A key for one predicate: the class Program with the predicate as its data,
/// the public evaluation artifacts, and the key and helper productions.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PePredicateKey {
    public_key: [u8; 32],
    predicate: BranchingProgram,
    compiled: CompiledPredicate,
    data: ProgramData,
    evaluation: StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
    key: StageArtifacts<()>,
    helpers: StageArtifacts<()>,
    commitment: [u8; 32],
}

impl PePredicateKey {
    pub const fn public_key(&self) -> &[u8; 32] {
        &self.public_key
    }

    pub const fn predicate(&self) -> &BranchingProgram {
        &self.predicate
    }

    pub const fn compiled(&self) -> &CompiledPredicate {
        &self.compiled
    }

    /// The predicate as the class Program's data ([`BranchingProgram::data`]).
    pub const fn data(&self) -> &ProgramData {
        &self.data
    }

    /// The compiled predicate's public evaluation artifacts, including `A_F`.
    pub const fn evaluation(&self) -> &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor> {
        &self.evaluation
    }

    /// The anchor, output, and compressed key blocks. Helpers are in [`Self::helpers`].
    pub const fn key(&self) -> &StageArtifacts<()> {
        &self.key
    }

    /// The streamed compact helper matrices, indexed by row, with helper `k` in column `k`.
    pub const fn helpers(&self) -> &StageArtifacts<()> {
        &self.helpers
    }

    /// The commitment to the setup, the predicate, its output layout, the
    /// block order, and the key width and cutoff.
    pub const fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }

    /// The output wire of the compiled predicate.
    pub(crate) fn output(&self) -> Result<WireId, PeError> {
        match self.compiled.program().outputs() {
            [Operand::Scalar(wire)] => Ok(*wire),
            _ => Err(PeError::Mismatch),
        }
    }
}

/// The canonical group `H` of the class, its inverses, the automorphisms
/// `P` with a published helper (`H \ {1}`, or `J_B` under a LUT radix `B`,
/// see [`mxx_exponent_lut::rns_bgg::RnsBggDynamicSetup::helper_automorphisms`]),
/// and, for row `i` and `k` in `P`, the destination row `i k` of the
/// negative gadget term of helper block `(i, k)`.
pub(crate) struct Group {
    pub(crate) elements: Vec<usize>,
    pub(crate) inverses: Vec<usize>,
    pub(crate) helpers: Vec<usize>,
    pub(crate) destinations: Vec<usize>,
}

impl Group {
    pub(crate) fn of(public_key: &PePublicKey) -> Result<Self, PeError> {
        let setup = public_key.setup();
        let elements = setup.automorphisms().to_vec();
        let order = 2 * public_key.parameters().level().ring_dimension() as usize;
        let inverses = elements
            .iter()
            .map(|k| setup.automorphism_inverse(*k))
            .collect::<Result<Vec<_>, _>>()?;
        let position = |value: usize| {
            elements.iter().position(|element| *element == value).ok_or(PeError::Mismatch)
        };
        let helpers = setup.helper_automorphisms();
        let destinations = elements
            .iter()
            .flat_map(|i| helpers.iter().map(move |k| i * k % order))
            .map(position)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { elements, inverses, helpers, destinations })
    }

    pub(crate) fn width(&self) -> usize {
        self.elements.len()
    }

    /// `|P|`: the helper blocks per row.
    pub(crate) fn helper_count(&self) -> usize {
        self.helpers.len()
    }

    pub(crate) fn select(values: &[usize], index: IntExpr) -> IntExpr {
        IntExpr::Select {
            selector: Box::new(index),
            branches: values.iter().map(|value| IntExpr::constant(*value)).collect(),
        }
    }
}

/// The anchor blocks `k~_row` of `key` (one member per row of `H`), from its key production.
pub(crate) fn anchor_keys(
    ring: &Ring,
    parameters: &crate::PeParameters,
    key: &PePredicateKey,
) -> mxx_dsl::Family<SmallMatrix> {
    ring.small_matrix_family_artifact_input(
        key.key().production().clone(),
        ANCHOR_KEY,
        parameters.class().lookup_width(),
        (parameters.anchor_columns(), 1),
        num_bigint::BigInt::from(parameters.key_cutoff().clone()),
        ArtifactAvailability::Transferred,
    )
}

/// Completed key-generation stages from one setup and predicate. Restore only state
/// serialized by this crate, with its original artifact store intact.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct PeKeygenCheckpoint {
    identity: Option<([u8; 32], [u8; 32])>,
    evaluation: Option<StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>>,
    helpers: Option<StageArtifacts<()>>,
}

/// `KeyGen(msk, F)`.
pub fn keygen<E, S>(
    public_key: &PePublicKey,
    master: &PeMasterSecretKey,
    predicate: &BranchingProgram,
    execution: &mut E,
    store: &mut S,
) -> Result<PePredicateKey, PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    keygen_resumable(
        public_key,
        master,
        predicate,
        execution,
        store,
        &mut PeKeygenCheckpoint::default(),
        |_| Ok(()),
    )
}

/// Generate a predicate key, skipping completed preprocessing and helper sampling.
/// `checkpoint` must belong to this setup and predicate. `save` runs after each
/// newly completed stage and must durably save the state before returning success.
/// Runtime plans are rebuilt. On success the stage handles move into the returned key.
pub fn keygen_resumable<E, S>(
    public_key: &PePublicKey,
    master: &PeMasterSecretKey,
    predicate: &BranchingProgram,
    execution: &mut E,
    store: &mut S,
    checkpoint: &mut PeKeygenCheckpoint,
    mut save: impl FnMut(&PeKeygenCheckpoint) -> Result<(), PeError>,
) -> Result<PePredicateKey, PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let parameters = public_key.parameters();
    if predicate.class() != parameters.class() || master.public_key() != public_key.identity() {
        return Err(PeError::Mismatch);
    }
    let compiled = predicate.class().compile()?;
    let data = predicate.data(&compiled);
    let rns = parameters.rns();
    let identity = (*public_key.identity(), data.commitment());
    if checkpoint.identity.is_some_and(|saved| saved != identity) {
        return Err(PeError::Mismatch);
    }
    checkpoint.identity = Some(identity);
    if checkpoint.evaluation.is_none() {
        let (_, evaluation) = api::preprocess(
            compiled.program(),
            &ParamEnv::default(),
            &PlaintextInputs::<RhsId, DCRTPoly>::new(),
            None,
            public_key.hash_key(),
            RnsPreprocessBuild::new(rns, *public_key.context())
                .with_output_use(RnsOutputUse::Gadget)
                .with_dynamic_setup(public_key.setup(), compiled.binding().clone())?
                .with_data(data.clone()),
            execution,
            store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )?;
        checkpoint.evaluation = Some(evaluation);
        save(checkpoint)?;
    } else {
        tracing::info!("restored PE preprocess");
    }
    let evaluation = checkpoint.evaluation.as_ref().unwrap();
    let output = match compiled.program().outputs() {
        [Operand::Scalar(wire)] => *wire,
        _ => return Err(PeError::Mismatch),
    };
    let ring = public_key.ring()?;
    let columns = parameters.digits();
    let width = 2 * columns;
    let m_b = parameters.anchor_columns();
    let group = Group::of(public_key)?;
    let w = group.width();
    let p = group.helper_count();
    let mut runtime = BTreeMap::new();
    let hash_key = public_key.hash_key_input(&mut runtime)?;
    let trapdoor = public_key.trapdoor(master)?;
    let output_key = |port: usize| {
        ring.artifact_input(
            evaluation.production().clone(),
            format!("public-key/{output:?}.{port}"),
            (1, columns),
            ArtifactAvailability::Transferred,
        )
    };
    let output_first_row = Mat::concat(ConcatAxis::Columns, vec![output_key(0), output_key(1)]);
    let (d_1, _) = public_key.message_row(hash_key.clone())?;
    let key_sigma = crate::params::real_expr(parameters.key_width())?;
    let key_cutoff = parameters.key_cutoff().clone();
    // The extension blocks are bounded by the key cutoff, so they are stored compactly.
    let gaussian_of =
        |rows: usize| ring.small_gaussian((rows, 1), key_sigma.clone(), key_cutoff.clone());
    let gaussian = || gaussian_of(width);
    let output_block = gaussian();
    // Each compressed lane exports its compact samples and only a scalar contribution.
    let compressed_width = parameters.compressed_anchor_width();
    let compressed = public_key
        .compressed_anchor_rows(hash_key.clone())?
        .map(|(first_row, _)| {
            mxx_dsl::parallel(w, |row| {
                let block = gaussian_of(compressed_width);
                let identity = gaussian_of(1);
                let i = Group::select(&group.elements, row.expression()?);
                let residual = first_row.clone().ring_automorphism(i).mul_small_rhs(block.clone()) +
                    ring.identity(1).mul_small_rhs(identity.clone());
                Ok((block, identity, residual))
            })
        })
        .transpose()?;
    let gadget_first_row = Mat::concat(
        ConcatAxis::Columns,
        vec![parameters.rns().public_gadget(0)?, ring.zero((1, columns))],
    );
    // Only row `1` carries the target `d_1` and the output block `A_F^(1) v_F`.
    let first = d_1 - output_first_row.mul_small_rhs(output_block.clone());
    let setup = public_key.setup();
    // Sample and export one compact row before contraction. This production
    // boundary avoids retaining W * p digit columns and publishes only W artifacts.
    let sampled = mxx_dsl::parallel(w, |_| {
        Ok(ring.small_gaussian((width, p), key_sigma.clone(), key_cutoff.clone()))
    })?;
    let sampled = DslContext::new("mxx-pe/keygen-helpers")
        .transferred_output(HELPER_KEY, sampled)?
        .build()?;
    let nonce: [u8; 32] = rand::random();
    if checkpoint.helpers.is_none() {
        let helper_artifacts = api::execute_artifacts(
            StageGraph::new(sampled, BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            (),
            digest(b"mxx-pe/keygen-helpers", &[public_key.identity(), &nonce]),
            execution,
            store,
            |_| Ok(()),
        )?;
        checkpoint.helpers = Some(helper_artifacts);
        save(checkpoint)?;
    } else {
        tracing::info!("restored PE keygen-helpers");
    }
    let helper_artifacts = checkpoint.helpers.as_ref().unwrap();
    let blocks = ring.small_matrix_family_artifact_input(
        helper_artifacts.production().clone(),
        HELPER_KEY,
        w,
        (width, p),
        num_bigint::BigInt::from(key_cutoff.clone()),
        ArtifactAvailability::Transferred,
    );
    // Contract a stored row once. Its positive terms go to row i and its
    // negative gadget terms to row i*k, so no partner-row import is needed. Each
    // term is added into one entry of a `1 x W` row through a unit row, which the
    // GPU lowers to a single-column addition.
    let unit = |index: IntExpr| mxx_exponent_lut::rns_bgg::unit_row(&ring, w, index);
    let contributions = mxx_dsl::parallel(w, |row| {
        let own = blocks.at(row.clone());
        let row = row.expression()?;
        let gadget_products = gadget_first_row.clone().mul_small_rhs(own.clone());
        mxx_dsl::iterate(p, ring.zero((1, w)), |k, sum| {
            let k = k.expression()?;
            let helper = setup
                .helper_key(rns, hash_key.clone(), Group::select(&group.helpers, k.clone()))
                .map_err(|_| mxx_dsl::DslError::Schema)?;
            let first_row = Mat::concat(ConcatAxis::Columns, vec![helper.0.0, helper.0.1]);
            let selected =
                Some(IndexRange { start: k.clone(), end: k.clone() + IntExpr::constant(1) });
            let positive = first_row
                .ring_automorphism(Group::select(&group.elements, row.clone()))
                .mul_small_rhs(own.clone())
                .slice(None, selected.clone());
            let negative = ring.zero((1, 1)) - gadget_products.clone().slice(None, selected);
            let destination = Group::select(
                &group.destinations,
                (row.clone() * IntExpr::constant(p) + k).canonicalize(),
            );
            let sum = sum + positive * unit(row.clone());
            Ok(sum + negative * unit(destination))
        })
    })?;
    let residuals = match w {
        1 => contributions.at(Int::constant(0)),
        count => {
            let first = contributions.at(Int::constant(0)) + contributions.at(Int::constant(1));
            mxx_dsl::iterate(count - 2, first, |index, sum| {
                Ok(sum + contributions.at(index.add(2)))
            })?
        }
    };
    // Row targets retain only scalar residuals, never full extension vectors.
    let compressed_residuals =
        compressed.as_ref().map(|blocks| blocks.field(|(_, _, r)| r)).transpose()?;
    let targets = mxx_dsl::parallel(w, |row| {
        let row = row.expression()?;
        let residual = residuals.clone().slice(
            None,
            Some(IndexRange { start: row.clone(), end: row.clone() + IntExpr::constant(1) }),
        );
        let residual = match &compressed_residuals {
            Some(contributions) => residual + contributions.at(Int::evaluate(row.clone())),
            None => residual,
        };
        let indicator = Int::evaluate(IntExpr::Select {
            selector: Box::new(row.clone()),
            branches: (0..w).map(|index| IntExpr::constant(usize::from(index == 0))).collect(),
        })
        .lift_to_constant_polynomial(ring.matrix_type((1, 1)));
        Ok((indicator * first.clone() - residual)
            .ring_automorphism(Group::select(&group.inverses, row)))
    })?;
    let anchors = mxx_dsl::parallel(w, |row| {
        // Parallel exports accept bounded matrices. Rebasing to the same ring
        // preserves the sample and its bound, using the compact forwarding path.
        Ok(trapdoor.sample_preimage(targets.at(row), (m_b, 1)).centered_rebase(&ring))
    })?;
    let graph = DslContext::new("mxx-pe/keygen").transferred_output(ANCHOR_KEY, anchors)?;
    let mut graph = graph.transferred_output(OUTPUT_KEY, output_block)?;
    if let Some(blocks) = compressed {
        graph = graph
            .transferred_output(COMPRESSED_KEY, blocks.field(|(block, _, _)| block)?)?
            .transferred_output(IDENTITY_KEY, blocks.field(|(_, identity, _)| identity)?)?;
    }
    let graph = graph.build()?;
    let (trapdoor_production, trapdoor_manifest) = master.trapdoor_production().clone();
    let nonce: [u8; 32] = rand::random();
    let key = api::execute_artifacts(
        StageGraph::new(
            graph,
            runtime,
            BTreeMap::from([
                (trapdoor_production, trapdoor_manifest),
                (helper_artifacts.production().clone(), helper_artifacts.manifest().clone()),
                (evaluation.production().clone(), evaluation.manifest().clone()),
                // `B_Q^(1)`, whose decompression the compact blocks read.
                public_key.anchor_production().clone(),
            ]),
        ),
        &ParamEnv::default(),
        (),
        digest(
            b"mxx-pe/keygen/v1",
            &[
                public_key.identity(),
                compiled.program().id().as_bytes(),
                &data.commitment(),
                &nonce,
            ],
        ),
        execution,
        store,
        |_| Ok(()),
    )?;
    let commitment = digest(
        b"mxx-pe/predicate-key/v1",
        &[
            public_key.identity(),
            compiled.program().id().as_bytes(),
            &data.commitment(),
            &serde_json::to_vec(compiled.binding())
                .map_err(|error| PeError::Graph(error.to_string()))?,
            b"gadget-output",
            &serde_json::to_vec(&group.elements)
                .map_err(|error| PeError::Graph(error.to_string()))?,
            parameters.key_width().to_string().as_bytes(),
            &parameters.key_cutoff().to_bytes_be(),
            &key.production().spec_hash.0,
            &key.production().execution_nonce,
        ],
    );
    Ok(PePredicateKey {
        public_key: *public_key.identity(),
        predicate: predicate.clone(),
        compiled,
        data,
        evaluation: checkpoint.evaluation.take().unwrap(),
        key,
        helpers: checkpoint.helpers.take().unwrap(),
        commitment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PeClass, setup::setup, test_support::*};
    use mxx_backends::{MemoryArtifactStore, poly::Poly};

    #[test]
    fn handles_restore_and_resume_keygen() {
        fn restore<T: serde::Serialize + serde::de::DeserializeOwned>(value: &T) -> T {
            let bytes = serde_json::to_vec(value).unwrap();
            let restored = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                serde_json::to_value(value).unwrap(),
                serde_json::to_value(&restored).unwrap()
            );
            restored
        }
        let class = PeClass::new(1, 2, 2, 1).unwrap();
        let parameters = restore(&parameters(class));
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
        let public_key = restore(&public_key);
        let master = restore(&master);
        let predicate = equals(class, &[0]);
        let mut progress = PeKeygenCheckpoint::default();
        // Simulate a process stopping immediately after each durable stage write.
        for _ in 0..2 {
            let result = keygen_resumable(
                &public_key,
                &master,
                &predicate,
                &mut execution,
                &mut store,
                &mut progress,
                |_| Err(PeError::Graph("interrupted".into())),
            );
            assert!(matches!(result, Err(PeError::Graph(_))));
            progress = restore(&progress);
        }
        let other = equals(class, &[1]);
        assert!(matches!(
            keygen_resumable(
                &public_key,
                &master,
                &other,
                &mut execution,
                &mut store,
                &mut progress,
                |_| Ok(())
            ),
            Err(PeError::Mismatch)
        ));
        let key = keygen_resumable(
            &public_key,
            &master,
            &predicate,
            &mut execution,
            &mut store,
            &mut progress,
            |_| panic!("all intermediate stages were restored"),
        )
        .unwrap();
        let key = restore(&key);
        let message = rand::random::<bool>();
        let ciphertext =
            crate::encrypt(&public_key, &[0], message, &mut execution, &mut store).unwrap();
        let ciphertext = restore(&ciphertext);
        assert_eq!(
            crate::decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap(),
            message
        );
    }

    /// SampleLeft: the key satisfies `M_F k = d_1 delta_1` exactly. Each row
    /// `i` of `M_F` is rebuilt from its blocks, independently of the
    /// residuals keygen solves: the inverse-coordinate anchor block
    /// `sigma_i(B^(1) k~_i)`, the helper blocks `sigma_i(H_k^(1)) v_(i,k)`
    /// of row `i` and `-G^(1) v_(j,k)` of every pair with `j k = i`, and,
    /// in row `1` only, `A_F^(1) v_F`. The one extension sample `v` enters
    /// every row.
    #[test]
    fn keys_solve_the_implicit_family_matrix() {
        let class = PeClass::new(2, 2, 2, 2).unwrap();
        keys_solve(class, parameters(class));
    }

    /// [`keys_solve_the_implicit_family_matrix`] under the LUT radix 2: the helper blocks are
    /// those of `J_2`, fewer than `W - 1` per row.
    #[test]
    fn radix_keys_solve_the_implicit_family_matrix() {
        let class = PeClass::new(2, 2, 2, 2).unwrap();
        keys_solve(class, parameters(class).with_lut_radix(2).unwrap());
    }

    /// [`keys_solve_the_implicit_family_matrix`] with compact encodings: each row also carries
    /// `sigma_i(B''^(1)) v''_i + v_iota,i`.
    #[test]
    fn compact_keys_solve_the_extended_family_matrix() {
        let class = PeClass::new(2, 2, 2, 2).unwrap();
        keys_solve(class, parameters(class).with_compact_encodings().unwrap());
    }

    fn keys_solve(class: PeClass, parameters: crate::PeParameters) {
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
        let pattern = [0; 2].map(|_| rand::random::<u32>() as usize % 2);
        let predicate = equals(class, &pattern);
        let key = keygen(&public_key, &master, &predicate, &mut execution, &mut store).unwrap();
        let ring = public_key.ring().unwrap();
        let rns = parameters.rns();
        let columns = parameters.digits();
        let width = 2 * columns;
        let group = Group::of(&public_key).unwrap();
        let w = group.width();
        let p = group.helper_count();
        let order = 2 * parameters.level().ring_dimension() as usize;
        let mut runtime = BTreeMap::new();
        let hash_key = public_key.hash_key_input(&mut runtime).unwrap();
        let production = key.key().production().clone();
        let cutoff = num_bigint::BigInt::from(parameters.key_cutoff().clone());
        let helpers = ring.small_matrix_family_artifact_input(
            key.helpers.production().clone(),
            HELPER_KEY,
            w,
            (width, p),
            cutoff.clone(),
            ArtifactAvailability::Transferred,
        );
        let output_block = ring.small_matrix_artifact_input(
            production.clone(),
            OUTPUT_KEY,
            (width, 1),
            cutoff.clone(),
            ArtifactAvailability::Transferred,
        );
        let compressed =
            public_key.compressed_anchor_rows(hash_key.clone()).unwrap().map(|(first_row, _)| {
                let family = |name: &str, rows: usize| {
                    ring.small_matrix_family_artifact_input(
                        production.clone(),
                        name,
                        w,
                        (rows, 1),
                        cutoff.clone(),
                        ArtifactAvailability::Transferred,
                    )
                };
                (
                    first_row,
                    family(COMPRESSED_KEY, parameters.compressed_anchor_width()),
                    family(IDENTITY_KEY, 1),
                )
            });
        let output = key.output().unwrap();
        let evaluation = key.evaluation().production().clone();
        let output_first_row = Mat::concat(
            ConcatAxis::Columns,
            (0..2)
                .map(|port| {
                    ring.artifact_input(
                        evaluation.clone(),
                        format!("public-key/{output:?}.{port}"),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    )
                })
                .collect(),
        );
        let gadget_first_row = Mat::concat(
            ConcatAxis::Columns,
            vec![rns.public_gadget(0).unwrap(), ring.zero((1, columns))],
        );
        let first = public_key.anchor_row().unwrap();
        let (d_1, _) = public_key.message_row(hash_key.clone()).unwrap();
        let helper = |left: Mat, row: usize, k: usize| {
            left.mul_small_rhs(helpers.at(Int::constant(row))).slice(
                None,
                Some(IndexRange { start: IntExpr::constant(k), end: IntExpr::constant(k + 1) }),
            )
        };
        let mut graph = DslContext::new("mxx-pe/test-sample-left");
        for (row, i) in group.elements.iter().enumerate() {
            let mut value = first
                .clone()
                .mul_small_rhs(anchor_keys(&ring, &parameters, &key).at(Int::constant(row)))
                .ring_automorphism(IntExpr::constant(*i));
            for (k_index, k) in group.helpers.iter().enumerate() {
                let public =
                    public_key.setup().helper_key(rns, hash_key.clone(), IntExpr::constant(*k));
                let public = public.unwrap();
                let helper_first_row =
                    Mat::concat(ConcatAxis::Columns, vec![public.0.0, public.0.1])
                        .ring_automorphism(IntExpr::constant(*i));
                value = value + helper(helper_first_row, row, k_index);
                for (source, j) in group.elements.iter().enumerate() {
                    if j * k % order == *i {
                        value = value - helper(gadget_first_row.clone(), source, k_index);
                    }
                }
            }
            if let Some((first_row, blocks, identity)) = &compressed {
                value = value +
                    first_row
                        .clone()
                        .ring_automorphism(IntExpr::constant(*i))
                        .mul_small_rhs(blocks.at(Int::constant(row))) +
                    ring.identity(1).mul_small_rhs(identity.at(Int::constant(row)));
            }
            if row == 0 {
                value = value + output_first_row.clone().mul_small_rhs(output_block.clone()) -
                    d_1.clone();
            }
            graph = graph.output(format!("row-{row}"), value).unwrap();
        }
        let outputs = api::execute_graph(
            StageGraph::new(
                graph.build().unwrap(),
                runtime,
                BTreeMap::from([
                    (key.key().production().clone(), key.key().manifest().clone()),
                    (key.helpers.production().clone(), key.helpers.manifest().clone()),
                    (evaluation, key.evaluation().manifest().clone()),
                    public_key.anchor_production().clone(),
                ]),
            ),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        for row in 0..w {
            let value = <_ as StageExecution<MemoryArtifactStore>>::download_matrix(
                &execution,
                &outputs.outputs[&format!("row-{row}")],
            )
            .unwrap()
            .entry(0, 0);
            assert!(
                value.coeffs_biguints().iter().all(|coefficient| *coefficient == 0u8.into()),
                "row {row} of M_F k - y is nonzero"
            );
        }
    }
}
