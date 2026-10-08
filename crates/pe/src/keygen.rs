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
use crate::{
    BranchingProgram, CompiledPredicate, PeError, PeMasterSecretKey, PePublicKey, setup::digest,
};
use mxx_backends::{
    SessionStore,
    poly::{PolyParams, dcrt::poly::DCRTPoly},
};
use mxx_dsl::{DslContext, Int, Mat, Preimage, Ring};
use mxx_exponent_lut::{
    api::{self, PlaintextInputs, StageArtifacts, StageExecution, StageGraph},
    program::{Operand, RhsId, WireId},
    rns_bgg::{RnsBggPreprocessOnlineEvalDescriptor, RnsOutputUse, RnsPreprocessBuild},
};
use mxx_ir_core::{IntExpr, ParamEnv, artifact::ArtifactAvailability, node::ConcatAxis};
use std::collections::BTreeMap;

/// The anchor blocks `k~_i` in the key production: one preimage artifact
/// `{ANCHOR_KEY}/{row}` per row of `H` (see [`anchor_key`]).
pub(crate) const ANCHOR_KEY: &str = "mxx-pe/key/anchor";
/// The helper blocks `v_(i,k)` (a family, member `i (W - 1) + k`).
pub(crate) const HELPER_KEY: &str = "mxx-pe/key/helper";
/// The output block `v_F`.
pub(crate) const OUTPUT_KEY: &str = "mxx-pe/key/output";

/// A key for one predicate: the compiled predicate with its public
/// evaluation artifacts, and the key production.
pub struct PePredicateKey {
    public_key: [u8; 32],
    predicate: BranchingProgram,
    compiled: CompiledPredicate,
    evaluation: StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
    key: StageArtifacts<()>,
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

    /// The compiled predicate's public evaluation artifacts, including `A_F`.
    pub const fn evaluation(&self) -> &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor> {
        &self.evaluation
    }

    /// The key production.
    pub const fn key(&self) -> &StageArtifacts<()> {
        &self.key
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

/// The canonical group `H` of the class, its inverses, and, for row `i`
/// and nonidentity `k`, the row `i k^-1` whose helper block `(i k^-1, k)`
/// places `-G^(1)` in row `i`.
pub(crate) struct Group {
    pub(crate) elements: Vec<usize>,
    pub(crate) inverses: Vec<usize>,
    pub(crate) sources: Vec<usize>,
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
        let sources = elements
            .iter()
            .flat_map(|i| inverses[1..].iter().map(move |k_inverse| i * k_inverse % order))
            .map(position)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { elements, inverses, sources })
    }

    pub(crate) fn width(&self) -> usize {
        self.elements.len()
    }

    pub(crate) fn select(values: &[usize], index: IntExpr) -> IntExpr {
        IntExpr::Select {
            selector: Box::new(index),
            branches: values.iter().map(|value| IntExpr::constant(*value)).collect(),
        }
    }
}

/// The anchor block `k~_row` of `key`, from its key production.
pub(crate) fn anchor_key(
    ring: &Ring,
    parameters: &crate::PeParameters,
    key: &PePredicateKey,
    row: usize,
) -> Preimage {
    ring.preimage_artifact_input(
        key.key().production().clone(),
        format!("{ANCHOR_KEY}/{row}"),
        (parameters.anchor_columns(), 1),
        num_bigint::BigInt::from(parameters.key_cutoff().clone()),
        ArtifactAvailability::Transferred,
    )
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
    let parameters = public_key.parameters();
    if predicate.class() != parameters.class() || master.public_key() != public_key.identity() {
        return Err(PeError::Mismatch);
    }
    let compiled = predicate.compile()?;
    let rns = parameters.rns();
    let (_, evaluation) = api::preprocess(
        compiled.program(),
        &ParamEnv::default(),
        &PlaintextInputs::<RhsId, DCRTPoly>::new(),
        None,
        public_key.hash_key(),
        RnsPreprocessBuild::new(rns, *public_key.context())
            .with_output_use(RnsOutputUse::Gadget)
            .with_dynamic_setup(public_key.setup(), compiled.binding().clone())?,
        execution,
        store,
        |_, outputs| Ok(outputs),
        |_| Ok(()),
    )?;
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
    let gaussian = || ring.gaussian((width, 1), key_sigma.clone(), key_cutoff.clone());
    // The one extension sample `v`, shared by every row.
    let helpers = mxx_dsl::parallel(w * (w - 1), |_| Ok(gaussian()))?;
    let output_block = gaussian();
    let gadget_first_row = Mat::concat(
        ConcatAxis::Columns,
        vec![parameters.rns().public_gadget(0)?, ring.zero((1, columns))],
    );
    // Only row `1` carries the target `d_1` and the output block `A_F^(1) v_F`.
    let first = d_1 - output_first_row * output_block.clone();
    let setup = public_key.setup();
    // Row `i`'s target `sigma_i^-1(u_i)` in a parallel loop; its preimage
    // under `B^(1)` is one existing sampler call per row.
    let targets = mxx_dsl::parallel(w, |row| {
        let row = row.expression()?;
        let i = Group::select(&group.elements, row.clone());
        let residual = mxx_dsl::iterate(w - 1, ring.zero((1, 1)), |k_index, sum| {
            let k_index = k_index.expression()?;
            let k = Group::select(&group.elements[1..], k_index.clone());
            let helper = setup
                .helper_key(rns, hash_key.clone(), k)
                .map_err(|_| mxx_dsl::DslError::Schema)?;
            let helper_first_row = Mat::concat(ConcatAxis::Columns, vec![helper.0.0, helper.0.1])
                .ring_automorphism(i.clone());
            let own = helpers.at(Int::evaluate(
                (row.clone() * IntExpr::constant(w - 1) + k_index.clone()).canonicalize(),
            ));
            let source = Group::select(
                &group.sources,
                (row.clone() * IntExpr::constant(w - 1) + k_index.clone()).canonicalize(),
            );
            let switched = helpers
                .at(Int::evaluate((source * IntExpr::constant(w - 1) + k_index).canonicalize()));
            Ok(sum + helper_first_row * own - gadget_first_row.clone() * switched)
        })?;
        let indicator = Int::evaluate(IntExpr::Select {
            selector: Box::new(row.clone()),
            branches: (0..w).map(|index| IntExpr::constant(usize::from(index == 0))).collect(),
        })
        .lift_to_constant_polynomial(ring.matrix_type((1, 1)));
        Ok((indicator * first.clone() - residual)
            .ring_automorphism(Group::select(&group.inverses, row)))
    })?;
    let mut graph = DslContext::new("mxx-pe/keygen");
    for row in 0..w {
        let anchor = trapdoor.sample_preimage(targets.at(Int::constant(row)), (m_b, 1));
        graph = graph.transferred_output(format!("{ANCHOR_KEY}/{row}"), anchor)?;
    }
    let graph = graph
        .transferred_output(HELPER_KEY, helpers)?
        .transferred_output(OUTPUT_KEY, output_block)?
        .build()?;
    let (trapdoor_production, trapdoor_manifest) = master.trapdoor_production().clone();
    let nonce: [u8; 32] = rand::random();
    let key = api::execute_artifacts(
        StageGraph::new(
            graph,
            runtime,
            BTreeMap::from([
                (trapdoor_production, trapdoor_manifest),
                (evaluation.production().clone(), evaluation.manifest().clone()),
            ]),
        ),
        &ParamEnv::default(),
        (),
        digest(
            b"mxx-pe/keygen/v1",
            &[public_key.identity(), compiled.program().id().as_bytes(), &nonce],
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
        evaluation,
        key,
        commitment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PeClass, setup::setup, test_support::*};
    use mxx_backends::{MemoryArtifactStore, poly::Poly};

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
        let parameters = parameters(class);
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
        let order = 2 * parameters.level().ring_dimension() as usize;
        let mut runtime = BTreeMap::new();
        let hash_key = public_key.hash_key_input(&mut runtime).unwrap();
        let production = key.key().production().clone();
        let helpers = ring.family_artifact_input(
            production.clone(),
            HELPER_KEY,
            w * (w - 1),
            (width, 1),
            ArtifactAvailability::Transferred,
        );
        let output_block = ring.artifact_input(
            production,
            OUTPUT_KEY,
            (width, 1),
            ArtifactAvailability::Transferred,
        );
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
        let helper = |row: usize, k: usize| helpers.at(Int::constant(row * (w - 1) + k - 1));
        let mut graph = DslContext::new("mxx-pe/test-sample-left");
        for (row, i) in group.elements.iter().enumerate() {
            let mut value = first
                .clone()
                .mul_small_rhs(anchor_key(&ring, &parameters, &key, row))
                .ring_automorphism(IntExpr::constant(*i));
            for (k_index, k) in group.elements.iter().enumerate().skip(1) {
                let public =
                    public_key.setup().helper_key(rns, hash_key.clone(), IntExpr::constant(*k));
                let public = public.unwrap();
                let helper_first_row =
                    Mat::concat(ConcatAxis::Columns, vec![public.0.0, public.0.1])
                        .ring_automorphism(IntExpr::constant(*i));
                value = value + helper_first_row * helper(row, k_index);
                for (source, j) in group.elements.iter().enumerate() {
                    if j * k % order == *i {
                        value = value - gadget_first_row.clone() * helper(source, k_index);
                    }
                }
            }
            if row == 0 {
                value = value + output_first_row.clone() * output_block.clone() - d_1.clone();
            }
            graph = graph.output(format!("row-{row}"), value).unwrap();
        }
        let outputs = api::execute_graph(
            StageGraph::new(
                graph.build().unwrap(),
                runtime,
                BTreeMap::from([
                    (key.key().production().clone(), key.key().manifest().clone()),
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
