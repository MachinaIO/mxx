//! Declaring a refresh from its PRF settings: the PBC parameters and profile of every PRF child,
//! the generated or declared key sets, and the keys' clear samples on a label.

use super::{RefreshDeclaration, RefreshError, RefreshKeys, TopReference};
use crate::{
    program::{
        AttributeKnowledge, ExpLayout, ExponentLutProgram, ExponentLutProgramBuilder,
        LutOutputRepresentation, LutTable, Operand, WireId,
    },
    program_gadgets::{
        pbc::{PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, generate_key_layout},
        prf::{PrfKeyShape, SparseLwrPrfProfile},
    },
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::BTreeMap};

/// The sparse-LWR PRF of every child of a refresh: LWR dimension `nu`, key weight `h`, input
/// modulus `Q`, output modulus `P` and, for a weight above one, the PBC row width `W` (`0`:
/// [`PbcParameters::default_bucket_width_limit`]).
#[derive(Clone, Copy, Debug)]
pub struct PrfSettings {
    pub lwr_dimension: usize,
    pub support_weight: usize,
    pub input_modulus: usize,
    pub output_modulus: usize,
    pub bucket_width: usize,
}

impl PrfSettings {
    /// The PBC parameters of every key: weight one in two buckets (every coordinate in both, so
    /// the row width is `nu + 1`), otherwise the paper's PBC layout of `h + 3` blocks of row width
    /// `bucket_width`.
    pub fn parameters(&self) -> PbcParameters {
        let nu = self.lwr_dimension;
        if self.support_weight == 1 {
            return PbcParameters::custom(nu, 1, 2, 2, 64, Some(nu + 1));
        }
        let parameters = PbcParameters::paper_evaluation(nu, self.support_weight);
        let width = match self.bucket_width {
            0 => parameters.default_bucket_width_limit(),
            width => width,
        };
        PbcParameters { bucket_width_limit: Some(width), ..parameters }
    }

    /// The PRF profile of every child in the ring of dimension `n`: `X^(2n / Q z)` adds exponents
    /// modulo `Q`, so `width_log2 = log2 Q - 1` and `stride_log2 = log2 n - width_log2`. An input
    /// modulus above `2 n` is invalid.
    pub fn profile(&self, n: usize) -> Result<SparseLwrPrfProfile, RefreshError> {
        let width_log2 = self.input_modulus.ilog2() - 1;
        let stride_log2 =
            n.ilog2().checked_sub(width_log2).ok_or(RefreshError::InvalidDeclaration)?;
        Ok(SparseLwrPrfProfile::new(
            ExpLayout { stride_log2, width_log2 },
            self.output_modulus,
            self.lwr_dimension,
        )?)
    }

    /// `count` PRF keys of weight `h` over the `nu` LWR coordinates on [`Self::parameters`],
    /// drawn from `rng` (a layout wider than the row width is retried, up to 64 times).
    pub fn generate_keys(
        &self,
        rng: &mut StdRng,
        count: usize,
    ) -> Result<Vec<PbcGeneratedKeyLayout>, RefreshError> {
        let (nu, parameters) = (self.lwr_dimension, self.parameters());
        (0..count)
            .map(|_| {
                let support = match self.support_weight {
                    1 => vec![rng.random::<u32>() as usize % nu],
                    weight => rand::seq::index::sample(rng, nu, weight).into_vec(),
                };
                let mut last = None;
                for _ in 0..64 {
                    match generate_key_layout(&parameters, PbcRootSeed(rng.random()), &support) {
                        Ok(layout) => return Ok(layout),
                        Err(error) => last = Some(error),
                    }
                }
                Err(last.map_or(RefreshError::InvalidDeclaration, RefreshError::from))
            })
            .collect()
    }

    /// The shape of `members` keys on [`Self::parameters`] that are never generated, for a run
    /// that executes no stage (a parameter search, a noise simulation or an export without
    /// executed keys): its key-set commitment and table digest are derived from the key seed
    /// `seed`, the role `role` of the keys and the shape, not from layouts, so such a run declares
    /// the refresh without materializing one layout per member (`w_0 n d` per smudge child). Its
    /// claim has the graphs of a run with generated keys up to these commitments (the hash
    /// namespaces derived from them, and the order of the hash-key externals named after them).
    /// Key counts remain `usize`, but their symbolic selector and coordinate-table cardinalities
    /// may exceed it: the PRF computes the product of members and row width in 128 bits and
    /// lowers its arithmetic through arbitrary-precision `IntExpr` values.
    pub fn declared_shape(
        &self,
        seed: u64,
        role: &str,
        members: usize,
    ) -> Result<PrfKeyShape, RefreshError> {
        let parameters = self.parameters();
        let encoded =
            serde_json::to_vec(&parameters).map_err(|_| RefreshError::InvalidDeclaration)?;
        let digest = |kind: &[u8]| -> [u8; 32] {
            let mut digest = Sha256::new();
            // The domain tag predates this module; it is part of every declared commitment, so
            // changing it changes every exported claim of declared keys.
            digest.update(b"mxx-exponent-lut/refresh-top-test/declared-keys/v1");
            digest.update(kind);
            digest.update(seed.to_le_bytes());
            digest.update((role.len() as u64).to_le_bytes());
            digest.update(role.as_bytes());
            digest.update((members as u64).to_le_bytes());
            digest.update(&encoded);
            digest.finalize().into()
        };
        Ok(PrfKeyShape::committed(parameters, members, digest(b"key-set"), digest(b"tables"))?)
    }

    /// The refresh of `target_output` of `source_program` to the format `reference` (see
    /// [`RefreshDeclaration::new`]: chain depths `depths`, refreshed slots `slots`, extension
    /// primes `extension` and `digits` smudge digits), with CBD(1) fresh noise on the profile
    /// [`Self::profile`] of the reference's ring, and its keys drawn from the key seed `seed`:
    /// with `generate`, the keys are generated ([`Self::generate_keys`], one layout per PRF
    /// member, in the order fresh noise of components 0 and 1, then every part's smudge keys)
    /// and returned; otherwise they are only declared ([`Self::declared_shape`], roles
    /// `fresh-<j>` and `smudge-<chain>-<j>`) and `None` is returned. The noise and the stage
    /// graphs depend on the key shapes only.
    #[allow(clippy::too_many_arguments)]
    pub fn declare_refresh(
        &self,
        source_program: &ExponentLutProgram,
        target_output: WireId,
        reference: &TopReference,
        depths: Vec<usize>,
        slots: Vec<usize>,
        extension: Vec<u64>,
        digits: usize,
        (seed, generate): (u64, bool),
    ) -> Result<(RefreshDeclaration, Option<RefreshKeyLayouts>), RefreshError> {
        let n = reference.ring_dimension as usize;
        let columns = reference.columns();
        let rng = RefCell::new(StdRng::seed_from_u64(seed));
        let keys = |role: &str, count: usize| -> Result<_, RefreshError> {
            if generate {
                let generated = self.generate_keys(&mut rng.borrow_mut(), count)?;
                let layouts = generated.iter().map(|key| key.public_layout.clone());
                Ok((PrfKeyShape::new(&layouts.collect::<Vec<_>>())?, generated))
            } else {
                Ok((self.declared_shape(seed, role, count)?, Vec::new()))
            }
        };
        let fresh = [keys("fresh-0", columns * n)?, keys("fresh-1", columns * n)?];
        let smudge = RefCell::new(BTreeMap::new());
        // The smudge keys are drawn while the declaration builds the parts; the first failure is
        // kept and reported after it (the callback returns a placeholder shape meanwhile).
        let failure = RefCell::new(None);
        let mut sample = |chain: usize, component: usize| match keys(
            &format!("smudge-{chain}-{component}"),
            columns * n * digits,
        ) {
            Ok((shape, generated)) => {
                smudge.borrow_mut().insert((chain, component), generated);
                shape
            }
            Err(error) => {
                failure.borrow_mut().get_or_insert(error);
                fresh[0].0.clone()
            }
        };
        let top = RefreshDeclaration::new(
            source_program,
            target_output,
            self.profile(n)?,
            1,
            reference.clone(),
            depths,
            slots,
            extension,
            digits,
            RefreshKeys { fresh: [fresh[0].0.clone(), fresh[1].0.clone()], smudge: &mut sample },
        );
        if let Some(error) = failure.into_inner() {
            return Err(error);
        }
        let top = top?;
        let [fresh_0, fresh_1] = fresh;
        let keys = RefreshKeyLayouts { fresh: [fresh_0.1, fresh_1.1], smudge: smudge.into_inner() };
        Ok((top, generate.then_some(keys)))
    }
}

/// The generated PRF keys of a refresh (see [`PrfSettings::declare_refresh`]).
#[derive(Clone, Debug)]
pub struct RefreshKeyLayouts {
    /// Component `j`'s `w_0 n` fresh-noise keys, shared by every chain.
    pub fresh: [Vec<PbcGeneratedKeyLayout>; 2],
    /// Every part's `w_0 n d` smudge keys (column `j` from `j n d`), by `(chain, component)`.
    pub smudge: BTreeMap<(usize, usize), Vec<PbcGeneratedKeyLayout>>,
}

impl RefreshKeyLayouts {
    /// The fresh noise `e'_j` of component `component` on the label of the invocation (hash key
    /// `hash_key`, label tag `label_tag`, see [`RefreshDeclaration::label`]): per column, the `n`
    /// CBD samples of its keys.
    pub fn fresh_noise(
        &self,
        top: &RefreshDeclaration,
        component: usize,
        hash_key: [u8; 32],
        label_tag: &[u8],
    ) -> Result<Vec<Vec<i64>>, RefreshError> {
        let fresh = top.fresh((0, component)).ok_or(RefreshError::InvalidDeclaration)?;
        let keys = self.fresh.get(component).ok_or(RefreshError::InvalidDeclaration)?;
        let label = top.label(hash_key, label_tag)?;
        let n = top.reference().ring_dimension as usize;
        keys.chunks(n).map(|column| fresh.samples(column, &label)).collect()
    }
}

/// The source Program of the refresh claim of [`crate::rns_bgg::RefreshRun::protocol`]: one
/// private input `X^b` (`b < 2`) mapped by the exponent LUT `X^e -> X^(1 - e)`, that input and
/// the output.
pub fn bit_flip_source() -> Result<(ExponentLutProgram, WireId, WireId), RefreshError> {
    let invalid = |_| RefreshError::InvalidDeclaration;
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(AttributeKnowledge::PrivatePayload)?;
    let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
    let table =
        LutTable::unary(2, layout, 2, vec![1, 0], LutOutputRepresentation::Exponent(layout))
            .map_err(invalid)?;
    let table = builder.register_lut(table)?;
    let output = builder.lut(Operand::Scalar(input), table)?;
    Ok((builder.finish(vec![Operand::Scalar(output)])?, input, output))
}
