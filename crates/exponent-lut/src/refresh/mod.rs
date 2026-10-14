//! Noise refresh of an exponent LUT output to the top level.
//!
//! The target is a private Program output `c_x` at level `L` (both component rows, payload
//! `Delta_L r_j x G^st_L`, `r = (r_0, 1)`). The refresh outputs, without any secret, the two rows
//! of a level-0 private encoding of the same `x` under a new mask `sigma = (sigma_0, 1)`:
//!
//! `y_j = sigma A'_(j) - Delta_0 r_j x G^st_0 + e'_j      (j = 0, 1)`
//!
//! in the format of a reference chain ([`TopReference`]: its level-0 primes `Q_0` and level-1
//! primes `Q_1`, `Delta_0 = Q_0 / Q_1`, `G^st_0` the gadget of `Q_1`), with `e'_j` CBD vectors
//! computed by the sparse-LWR PRF ([`FreshNoiseProgram`]). The caller provides both the new mask
//! and the public matrices:
//!
//! - `sigma_0` arrives as an encoding of `sigma_0 G` under the caller's mask secret (see [Caller
//!   inputs](#caller-inputs)); the refresh never sees `sigma_0` itself.
//! - `A' = [A'_(0) | A'_(1)]` is a public matrix of the reference's level-0 ring.
//!
//! # Chains
//!
//! Decoding `y_j mod q` needs `q` in the decoding modulus, but level `L` has lost most primes of
//! `Q_0`. The refresh therefore evaluates the source Program on one chain per refreshed slot, a
//! prime `q` of `Q_0`: level `l` of chain `q` has the first `depths[l]` primes of its pool order
//! `[q, the other primes of Q_0 in canonical order, extension primes]`, so it keeps `q` to level
//! `L + 1` (the plaintext modulus of level `L`). Every ring lists its towers in the canonical
//! order, so a modulus has one tower order on every chain. On chain `q` the part `(q, j)` decodes
//! `y_j mod q` (the per-chain decoding step, see `part`): it combines the one encoding
//! `[g | c_one]`, with `g` the caller's encoding of `sigma_0 G`, the target's component `j` and
//! the fresh-noise carriers into a row that one smudged trapdoor decoder rounds to `y_j mod q`;
//! the target action maps `Delta_L r_j x G^st_L` to `Q_q (Delta_0 r_j x G^st_0 mod q)`.
//! [`crate::rns_bgg`] recombines the residues of every chain modulo the product `P` of the
//! refreshed slots' primes (`Q_0` when every slot is refreshed); refreshing a subset of the
//! slots yields `y_j mod P` only.
//!
//! # Caller inputs
//!
//! - **`A'`.** One matrix `[A'_(0) | A'_(1)]` of shape `2 x 2 w_0` over the reference's level-0
//!   ring `R_(Q_0)` (the primes of [`TopReference::top`] in canonical order, ring dimension
//!   [`TopReference::ring_dimension`]), `A'_(j)` its columns `j w_0 .. (j + 1) w_0`. Every part
//!   reads the whole matrix and reduces it modulo the primes of its decoding level (a CRT sub-basis
//!   of `Q_0` in the same tower order), so all chains use the residues of one `A'`; the decode
//!   stage reads it modulo `Q_0` (modulo `P` for a refresh of some slots). Then `sigma A' = [sigma
//!   A'_(0) | sigma A'_(1)]` is the mask part of the two output rows side by side. The two halves
//!   must be independent and uniform: with `A'_(0) = A'_(1)`, `y_0 - y_1 = -Delta_0 (r_0 - 1) x
//!   G^st_0 + e'_0 - e'_1` reveals the payload.
//! - **The encoding of `sigma_0 G`, per chain.** Chain `q` takes one encoding, shared by its two
//!   parts `(chain, 0)` and `(chain, 1)`, which both read its public key `K_g` and its row `g`: an
//!   encoding under the caller's mask secret `s = (s_0, 1)` (the mask secret of the target's
//!   stages) in the ring of the chain's decoding level `L` (the ring of `Q`, the level's primes in
//!   canonical order, the same for both parts), with `m` columns (the digits of `Q`): `g = s K_g -
//!   v G_Q + e_g`, `K_g in R_Q^(2 x m)`, `G_Q` the public gadget of that ring, and `v = sigma_0
//!   (mod q)` for the chain's output prime `q` (only `v mod q` matters: a PRF output with payload
//!   `(M +
//!   1) sigma_0 G_Q` may be passed as is). `sigma_0` must be binary (the security of the refreshed
//!   output, [`crate::rns_bgg::RnsBggRefresh::security`], assumes a uniform binary mask), the same
//!   on every chain and component, and the error `e_g` must be bounded as the noise simulation is
//!   told ([`crate::rns_bgg::RefreshSigmaNoise`]: `||e_g||_inf` and its average-case moments),
//!   since it enters the decoded row through `(I_2 ⊗ G_Q)^-1`. This is exactly the finished
//!   [`crate::rns_bgg::RnsOutputUse::Gadget`] output of a raw coefficient Program evaluated on
//!   [`crate::rns_bgg::RnsBggParameters::refresh_child_parameters`] (its key and its row component
//!   1), e.g. the binary [`FreshNoiseProgram`] of `n` samples.
//!
//! # Shared and chain-specific material
//!
//! The CBD children of every chain use the same keys (one CBD key set per component) and the
//! same label, so all chains compute the same `e'_j`; every decoder has its own smudge keys, one
//! key set per column (one family over the columns). The stages of chain `q` run under the hash
//! namespace [`RefreshDeclaration::chain_namespace`], so no two chains share a hash-derived
//! public matrix. The label is sampled under the refresh's own namespace, identical for every
//! chain.
//!
//! # Declaring and running a refresh
//!
//! [`PrfSettings::declare_refresh`] declares a refresh from the settings of its sparse-LWR PRF and
//! a key seed, generating the keys ([`RefreshKeyLayouts`]) or only declaring their shapes (the
//! noise and the stage graphs depend on the shapes only). [`crate::rns_bgg::RnsBggRefresh`] runs
//! it on RNS-BGG stages and states the claim of a run for the source [`bit_flip_source`].
//!
//! # Caller responsibilities and preconditions
//!
//! - The target must be the output of an exponent LUT, whose plaintext is one signed table monomial
//!   (`||x||_1 = 1`) under the crate's LUT input contract (every LUT input is one signed monomial);
//!   a Program that sums before its last LUT is outside the claim.
//! - Refreshing two encodings under the same secret and the same `A'` reveals the difference of
//!   their payloads unless each has a fresh `sigma_0`; the refresh does not track it.
//! - Every invocation uses a fresh label tag ([`RefreshDeclaration::label`]); a repeated label with
//!   the same keys repeats `e'` and every smudge.
//! - Label encodings of another label under the right artifact names are outside the contract.
//!
//! The smudges hide the decoders' noise under the heuristic of the existing decoder, which
//! ignores the correlation of the smudge carriers' errors with the smudge values, over the joint
//! view of every chain's hint, preimages and child encodings.

use crate::{
    decoder::decoding::{DecoderDeclaration, DecoderError},
    program::{
        ExponentLutProgram, ExponentLutProgramBuilder, ExponentLutProgramId, FamilyRange, Operand,
        ProgramValidationError, WireId,
    },
    program_gadgets::{
        prf::{PrfKeyShape, PrfKeys, SparseLwrPrfProfile},
        prf_sampler::{PrfDistribution, PrfSampler, PrfSamplerError},
    },
};
use mxx_ir_core::IntExpr;
use serde::Serialize;
use sha2::{Digest, Sha256};

mod part;
mod setup;
pub(crate) use part::RefreshPartDeclaration;
pub use setup::{PrfSettings, RefreshKeyLayouts, bit_flip_source};

/// Errors in a refresh declaration.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum RefreshError {
    #[error("invalid noise refresh declaration")]
    InvalidDeclaration,
    #[error(transparent)]
    Decoder(#[from] DecoderError),
    #[error(transparent)]
    Sampler(#[from] PrfSamplerError),
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
    #[error(transparent)]
    Prf(#[from] crate::program_gadgets::prf::PrfError),
    #[error(transparent)]
    Pbc(#[from] crate::program_gadgets::pbc::PbcError),
}

/// The fresh-noise child Program: one output, the row of `m` column blocks
/// whose block `j` is the polynomial `sum_k e'_(j n + k) X^k` of `n` CBD
/// samples of one PRF member and key each (`m n` members in one
/// [`PrfSampler`] family, the row `Sum` of
/// [`ExponentLutProgramBuilder::sum_columns`] over the `m` columns). Its
/// coefficient output is a raw carrier row with payload blocks `e'_j G`
/// (scale 1); the refresh moves it down to the decoding ring once and never
/// finalizes it for decryption.
#[derive(Clone, Debug)]
pub struct FreshNoiseProgram {
    program: ExponentLutProgram,
    sampler: PrfSampler,
    columns: usize,
    output: WireId,
}

impl FreshNoiseProgram {
    /// The `columns` polynomials of `coefficients` CBD(`eta`) samples on
    /// `profile`, member `j coefficients + k` on key `j coefficients + k` of
    /// the shape `keys`.
    pub fn new(
        profile: SparseLwrPrfProfile,
        eta: u32,
        columns: usize,
        coefficients: usize,
        keys: PrfKeyShape,
    ) -> Result<Self, RefreshError> {
        Self::with_distribution(profile, PrfDistribution::Cbd { eta }, columns, coefficients, keys)
    }

    /// The `columns` polynomials of `coefficients` samples of `distribution`
    /// (see [`Self::new`]); one polynomial of [`PrfDistribution::Binary`] samples modulo a
    /// chain's output prime is an encoding of `sigma_0 G` a refresh accepts (see
    /// [Caller inputs](crate::refresh#caller-inputs)).
    pub fn with_distribution(
        profile: SparseLwrPrfProfile,
        distribution: PrfDistribution,
        columns: usize,
        coefficients: usize,
        keys: PrfKeyShape,
    ) -> Result<Self, RefreshError> {
        let members = columns.checked_mul(coefficients).ok_or(RefreshError::InvalidDeclaration)?;
        if columns == 0 ||
            coefficients == 0 ||
            members < 2 ||
            keys.members() != members ||
            Some(coefficients) != profile.ring_dimension().ok()
        {
            return Err(RefreshError::InvalidDeclaration);
        }
        let mut builder = ExponentLutProgramBuilder::new();
        let sampler = PrfSampler::build(&mut builder, profile, PrfKeys::Shape(keys), distribution)?;
        let crate::program_gadgets::prf::PrfOutput::Family(samples) = sampler.output() else {
            return Err(RefreshError::InvalidDeclaration);
        };
        // Member `j n + k` lands at `X^k` of column `j`.
        let range = FamilyRange::full(IntExpr::constant(members))?;
        let placed = builder.parallel(range, |builder, member| {
            let coefficient =
                IntExpr::Rem(Box::new(member.clone()), Box::new(IntExpr::constant(coefficients)))
                    .canonicalize();
            builder
                .mul_small_constant(
                    Operand::Member { family: samples, index: member },
                    vec![1],
                    coefficient,
                )
                .map(Operand::Scalar)
        })?;
        let output = builder.sum_columns(placed, columns)?;
        let program = builder.finish(vec![Operand::Scalar(output)])?;
        Ok(Self { program, sampler, columns, output })
    }

    pub fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    pub fn id(&self) -> ExponentLutProgramId {
        self.program.id()
    }

    /// The sampler of the `m n` members.
    pub fn sampler(&self) -> &PrfSampler {
        &self.sampler
    }

    /// The key shape of the members, which commits to their key set.
    pub fn keys(&self) -> Result<&PrfKeyShape, RefreshError> {
        self.sampler.prf().key_shape().ok_or(RefreshError::InvalidDeclaration)
    }

    /// The output: the row of the column polynomials.
    pub const fn output(&self) -> WireId {
        self.output
    }

    /// The number of columns of the output row.
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// The samples of the members on the generated keys `keys` (member `k` on key `k`) for the
    /// PRF label `label` ([`RefreshDeclaration::label`]): the values the Program's sampler reads
    /// before it places them (`+-h` for [`PrfDistribution::Binary`], the CBD value for
    /// [`PrfDistribution::Cbd`]).
    pub fn samples(
        &self,
        keys: &[crate::program_gadgets::pbc::PbcGeneratedKeyLayout],
        label: &[usize],
    ) -> Result<Vec<i64>, RefreshError> {
        let q = self.sampler.prf().profile().q()?;
        let label = label.iter().map(|value| *value as u64).collect::<Vec<_>>();
        keys.iter()
            .map(|key| {
                let z = crate::program_gadgets::pbc::clear_pbc_inner_product(
                    &key.public_layout,
                    key.private_schedule(),
                    &label,
                    q as u64,
                )?;
                Ok(self.sampler.sample_at(&self.program, z as usize)?)
            })
            .collect()
    }
}

/// The output format of a refresh to the top level: the level-0 private
/// payload `Delta_0 r_j x G^st_0` of a reference chain whose level 0 has the
/// CRT primes `top` and level 1 the primes `next` (both in canonical order),
/// in the ring of dimension `ring_dimension`, with gadget base
/// `2^base_bits` (`G^st_0` is the gadget of `next`, `Delta_0 = Q_0 / Q_1`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TopReference {
    pub top: Vec<u64>,
    pub next: Vec<u64>,
    pub ring_dimension: u32,
    pub base_bits: u32,
}

impl TopReference {
    /// The columns `w_0` of `G^st_0`: the digits of `Q_1` in base `2^base_bits`.
    pub fn columns(&self) -> usize {
        let bits: u64 = self
            .next
            .iter()
            .map(|prime| u64::from(u64::BITS - prime.leading_zeros()))
            .max()
            .unwrap_or(0);
        bits.div_ceil(u64::from(self.base_bits)) as usize * self.next.len()
    }
}

/// The key shapes of a refresh's PRF keys. Each shape commits to its key set (see
/// [`PrfKeyShape`]); the keys' layouts are the stages' integer data (coordinate tables, see
/// [`crate::program::ProgramData`]).
pub struct RefreshKeys<'a> {
    /// Component `j`'s CBD child keys (`w_0 n`), shared by every chain.
    pub fresh: [PrfKeyShape; 2],
    /// The `w_0 n d` smudge keys of the decoder of part `(chain, component)`, column `j` the keys
    /// from `j n d`; every call must return fresh keys.
    pub smudge: &'a mut dyn FnMut(usize, usize) -> PrfKeyShape,
}

/// Immutable, committed declaration of a refresh to the top level.
///
/// Its parts `(chain, component)`, one per refreshed slot and component, each run their own
/// preprocessing, encryption and online evaluation (the `rns_bgg` stage builders take the part
/// as `(chain, component)`); [`Self::fresh`], [`Self::decoder`] and [`Self::fresh_namespace`]
/// give the children of a part.
#[derive(Clone, Debug)]
pub struct RefreshDeclaration {
    profile: SparseLwrPrfProfile,
    reference: TopReference,
    depths: Vec<usize>,
    extension: Vec<u64>,
    level: usize,
    /// The refreshed slots: chain `chain` decodes prime `reference.top[slots[chain]]`.
    slots: Vec<usize>,
    /// Part `(chain, component)` at `2 chain + component`.
    parts: Vec<RefreshPartDeclaration>,
    spec: [u8; 32],
}

impl RefreshDeclaration {
    /// Check and freeze the refresh of `target_output` of `source_program`, whose level is
    /// `depths.len() - 2`, to the format `reference`: the chains have `depths[l]` primes at level
    /// `l` (`depths[0]` the primes of `reference.top`, `depths[1]` those of `reference.next`),
    /// the children extend them with primes of `extension`, the fresh noise is CBD(`eta`) and
    /// every decoder smudges with `digits` digits, on the keys of `keys`. `slots` lists the
    /// refreshed primes of `reference.top` by increasing index, one chain each. The
    /// specification commits to the CBD key sets, and every part's to its smudge key sets, so
    /// refreshes of distinct key sets share no hash namespace. The new mask and `A'` are the
    /// caller's inputs of the stages (see [Caller inputs](crate::refresh#caller-inputs)).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_program: &ExponentLutProgram,
        target_output: WireId,
        profile: SparseLwrPrfProfile,
        eta: u32,
        reference: TopReference,
        depths: Vec<usize>,
        slots: Vec<usize>,
        extension: Vec<u64>,
        digits: usize,
        keys: RefreshKeys<'_>,
    ) -> Result<Self, RefreshError> {
        let invalid = RefreshError::InvalidDeclaration;
        let level = depths.len().checked_sub(2).ok_or(RefreshError::InvalidDeclaration)?;
        let coefficients =
            profile.ring_dimension().map_err(|_| RefreshError::InvalidDeclaration)?;
        if reference.ring_dimension as usize != coefficients ||
            depths.first() != Some(&reference.top.len()) ||
            depths.get(1) != Some(&reference.next.len()) ||
            !reference.next.iter().all(|prime| reference.top.contains(prime)) ||
            depths.windows(2).any(|pair| pair[1] > pair[0]) ||
            depths[level] < 2 ||
            depths[level + 1] == 0 ||
            extension.iter().any(|prime| reference.top.contains(prime)) ||
            slots.is_empty() ||
            slots.windows(2).any(|pair| pair[1] <= pair[0]) ||
            slots.last().is_some_and(|slot| *slot >= reference.top.len())
        {
            return Err(invalid);
        }
        target_admission(source_program, target_output)?;
        let spec = commitment(&(
            b"mxx-exponent-lut/refresh-top/spec/v3".as_slice(),
            source_program.id(),
            target_output,
            &profile,
            eta,
            &reference,
            &depths,
            &slots,
            &extension,
            digits,
            &keys.fresh,
        ))?;
        let primes = depths[level];
        let outputs = slots.iter().map(|slot| reference.top[*slot]).collect::<Vec<_>>();
        let mut parts = Vec::with_capacity(2 * slots.len());
        for (chain, prime) in slots.iter().map(|slot| &reference.top[*slot]).enumerate() {
            // The decoding level keeps the output prime and the first `primes - 1` other
            // primes of `Q_0`, in canonical tower order.
            let kept = reference.top.iter().filter(|other| *other != prime).take(primes - 1);
            let kept = kept.collect::<Vec<_>>();
            let position = reference
                .top
                .iter()
                .take_while(|other| *other != prime)
                .filter(|other| kept.contains(other))
                .count();
            for component in 0..2 {
                parts.push(RefreshPartDeclaration::new(
                    source_program,
                    target_output,
                    profile,
                    eta,
                    primes,
                    (position, *prime),
                    outputs.clone(),
                    component,
                    digits,
                    reference.clone(),
                    keys.fresh[component].clone(),
                    (keys.smudge)(chain, component),
                    spec,
                )?);
            }
        }
        Ok(Self { profile, reference, depths, extension, level, slots, parts, spec })
    }

    pub const fn spec(&self) -> &[u8; 32] {
        &self.spec
    }

    /// The sparse-LWR PRF profile of every child.
    pub const fn profile(&self) -> &SparseLwrPrfProfile {
        &self.profile
    }

    pub const fn reference(&self) -> &TopReference {
        &self.reference
    }

    /// The target's level `L`.
    pub const fn level(&self) -> usize {
        self.level
    }

    /// The prime count of every level of every chain.
    pub fn depths(&self) -> &[usize] {
        &self.depths
    }

    /// The number of chains: one per refreshed slot.
    pub fn chains(&self) -> usize {
        self.slots.len()
    }

    /// The refreshed slots, indices into the reference's `Q_0` primes.
    pub fn slots(&self) -> &[usize] {
        &self.slots
    }

    /// Chain `chain`'s output prime.
    pub fn output_prime(&self, chain: usize) -> Option<u64> {
        self.slots.get(chain).map(|slot| self.reference.top[*slot])
    }

    /// The primes of the refreshed slots in canonical order; their product
    /// `P` is the modulus of the recombined rows.
    pub fn output_primes(&self) -> Vec<u64> {
        self.slots.iter().map(|slot| self.reference.top[*slot]).collect()
    }

    /// Chain `chain`'s pool order: its output prime, the other primes of
    /// `Q_0` in canonical order, then the extension primes.
    pub fn pool_order(&self, chain: usize) -> Option<Vec<u64>> {
        let prime = self.output_prime(chain)?;
        Some(
            std::iter::once(prime)
                .chain(self.reference.top.iter().copied().filter(|other| *other != prime))
                .chain(self.extension.iter().copied())
                .collect(),
        )
    }

    /// Part `(chain, component)`.
    pub(crate) fn part(
        &self,
        (chain, component): (usize, usize),
    ) -> Option<&RefreshPartDeclaration> {
        (component < 2).then(|| self.parts.get(2 * chain + component)).flatten()
    }

    /// The fresh-noise child of part `(chain, component)`: the CBD noise `e'_j` of the
    /// component.
    pub fn fresh(&self, part: (usize, usize)) -> Option<&FreshNoiseProgram> {
        self.part(part).map(RefreshPartDeclaration::fresh)
    }

    /// The decoder of part `(chain, component)`, which decodes the chain's output prime and
    /// whose smudge child runs under its commitment.
    pub fn decoder(&self, part: (usize, usize)) -> Option<&DecoderDeclaration> {
        self.part(part).map(RefreshPartDeclaration::decoder)
    }

    /// The hash namespace of the fresh-noise child of part `(chain, component)`: its stage
    /// builders take it through `with_hash_namespace`.
    pub fn fresh_namespace(&self, part: (usize, usize)) -> Option<[u8; 32]> {
        self.part(part).map(RefreshPartDeclaration::fresh_namespace)
    }

    /// The hash namespace of chain `chain`'s stages.
    pub fn chain_namespace(&self, chain: usize) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/refresh-top/chain-namespace/v1");
        digest.update(self.spec);
        digest.update((chain as u64).to_le_bytes());
        digest.finalize().into()
    }

    /// The PRF label `a in Z_Q'^v` of an invocation with label tag
    /// `label_tag` under the stage hash key, shared by every part.
    pub fn label(&self, hash_key: [u8; 32], label_tag: &[u8]) -> Result<Vec<usize>, RefreshError> {
        self.parts[0].label(hash_key, label_tag)
    }
}

/// The admission rule of a refresh target: the output of a LUT with an exponent representation
/// (one signed table monomial, `||x||_1 = 1`, under the LUT input contract), so
/// `||r_j x||_inf <= 1 < q_min / 2` for the ternary payload secret and the same integer
/// represents `r_j x` modulo the target's plaintext modulus and modulo `Q_1`.
fn target_admission(program: &ExponentLutProgram, target: WireId) -> Result<(), RefreshError> {
    let representation = program.lut_output_representation(target)?;
    match representation {
        Some(crate::program::LutOutputRepresentation::Exponent(_)) => Ok(()),
        _ => Err(RefreshError::InvalidDeclaration),
    }
}

fn commitment(value: &impl Serialize) -> Result<[u8; 32], RefreshError> {
    let bytes = serde_json::to_vec(value).map_err(|_| RefreshError::InvalidDeclaration)?;
    Ok(Sha256::digest(bytes).into())
}

#[cfg(test)]
pub(crate) mod tests;
