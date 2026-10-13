//! Noise refresh of a private encoding row.
//!
//! The target is component 1 of a private Program output at a non-terminal
//! level `l` with modulus `Q = q_1 ... q_L` (`L >= 2`):
//! `c_x = s K_x - Delta x G^st + e_x`, where `G^st` is the level's state gadget
//! (the gadget of level `l + 1`, modulus `q_(l+1)`) and `Delta = Q / q_(l+1)`.
//! The refresh outputs, without any secret, the row
//!
//! `y = s A' - Delta x G_Q + e'      in R_Q^(1 x m)`
//!
//! for the hash-sampled public `A' in R_Q^(2 x m)`, the full gadget `G_Q` of
//! level `l` (`m` columns) and a fresh noise `e'` of `m` CBD polynomials
//! computed by the sparse-LWR PRF ([`FreshNoiseProgram`]).
//!
//! # Computation
//!
//! With the one encoding `c_1 = s A_1 - s (I_2 ⊗ G_Q) + e_1` (a public-payload
//! encoding of `1` at level `l`) and the fresh-noise carriers
//! `f_j = s K_(f,j) - e'_j G_Q + e_(f,j)` (scale 1), for every prime `q_i`,
//! `Q_i = Q / q_i`:
//!
//! - `u_i = c_1 (I_2 ⊗ G_Q)^-1(Q_i A') = s A_1 X_i - Q_i s A' + e_1 X_i`;
//! - `v_i = c_x G^st^-1(Q_i G_Q mod q_(l+1)) = s K_x Y_i - Q_i Delta x G_Q + e_x Y_i`, exact modulo
//!   `Q` because `Delta q_(l+1) = Q`;
//! - `w_i = [f_j G_Q^-1(Q_i)]_j = s K_f Z_i - Q_i e' + e_f Z_i`;
//! - `d_i = u_i - v_i + w_i = s K_i - Q_i y + E_i`.
//!
//! The row `d_i` is decoded by one smudged trapdoor decoder per prime
//! ([`crate::decoder`], target [`DecoderTarget::Refresh`]) to `y mod q_i`:
//! every column `j` has its own smudge keys and its own preimage column, so
//! the columns are decoded independently, but as one `1 x m` row (one
//! smudge family over the columns, one preimage of `m` columns). A CRT
//! recomposition over the primes returns `y`. Preprocessing computes the
//! decoded keys `K_i` by the same steps on the public keys to sample the
//! decoders' preimages; the output key `A'` is public.
//!
//! # Caller responsibilities
//!
//! - `A'` is fixed by the declaration's matrix tag and reused across encryptions (each with a fresh
//!   mask secret). Refreshing two encodings under the same secret and the same `A'` reveals the
//!   difference of their payloads; the refresh does not track it.
//! - Every invocation uses a fresh label tag: the PRF label `a` is hashed from it (see
//!   [`RefreshDeclaration::label`]), and a repeated label with the same keys repeats `e'` and every
//!   smudge.
//!
//! The smudge hides the decoders' noise under the heuristic of the existing
//! decoder, which ignores the correlation of the smudge carriers' errors with
//! the smudge values.
//!
//! [`top`] builds on this refresh to re-encrypt a level-`L` encoding as a
//! level-0 encoding under a new mask, with one decoding chain per prime of
//! the top modulus.

use crate::{
    decoder::decoding::{DecoderDeclaration, DecoderError, DecoderSmudgeProgram, DecoderTarget},
    program::{
        ExponentLutProgram, ExponentLutProgramBuilder, ExponentLutProgramId, FamilyRange, Operand,
        ProgramValidationError, WireId,
    },
    program_gadgets::{
        prf::{PrfKeyShape, PrfKeys, SparseLwrPrfProfile},
        prf_sampler::{PrfDistribution, PrfSampler, PrfSamplerError},
    },
};
use mxx_backends::sampler::hash::sample_hash_integers;

pub mod top;
use mxx_ir_core::IntExpr;
use num_bigint::BigUint;
use serde::Serialize;
use sha2::{Digest, Sha256};

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
    /// (see [`Self::new`]); the top refresh's mask child is one polynomial of
    /// [`PrfDistribution::Binary`] samples.
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

/// What makes a [`RefreshDeclaration`] one part of a refresh to the top
/// level ([`top::TopRefreshDeclaration`]): it refreshes component
/// `component` of its target into the reference format and decodes only the
/// output prime `output_prime` (an index into the decoding level's primes),
/// with the new mask `(sigma_0, 1)` whose `sigma_0` is the mask child's
/// binary polynomial.
#[derive(Clone, Debug)]
pub struct TopRefreshPart {
    top: [u8; 32],
    component: usize,
    output_prime: usize,
    output_modulus: u64,
    output_moduli: Vec<u64>,
    reference: TopReference,
    mask: FreshNoiseProgram,
}

impl TopRefreshPart {
    /// The commitment of the top refresh: it names the label and `A'`.
    pub const fn top(&self) -> &[u8; 32] {
        &self.top
    }

    pub const fn component(&self) -> usize {
        self.component
    }

    pub const fn output_prime(&self) -> usize {
        self.output_prime
    }

    /// The output prime `q` itself.
    pub const fn output_modulus(&self) -> u64 {
        self.output_modulus
    }

    /// The refreshed primes of the top refresh, in canonical order: their
    /// product `P` is the modulus of the recombined rows, to which every
    /// part contributes its residue's CRT term.
    pub fn output_moduli(&self) -> &[u64] {
        &self.output_moduli
    }

    pub const fn reference(&self) -> &TopReference {
        &self.reference
    }

    pub const fn mask(&self) -> &FreshNoiseProgram {
        &self.mask
    }
}

/// Immutable, committed declaration of one noise refresh.
#[derive(Clone, Debug)]
pub struct RefreshDeclaration {
    source_program: ExponentLutProgramId,
    target_output: WireId,
    profile: SparseLwrPrfProfile,
    eta: u32,
    primes: usize,
    columns: usize,
    coefficients: usize,
    digits: usize,
    matrix_tag: Vec<u8>,
    fresh: FreshNoiseProgram,
    /// One decoder of the whole row per decoded prime
    /// ([`Self::decoded_primes`]), in order.
    decoders: Vec<DecoderDeclaration>,
    spec: [u8; 32],
    top: Option<TopRefreshPart>,
}

impl RefreshDeclaration {
    /// Check and freeze the refresh of `target_output` of `source_program`
    /// over a decoding modulus of `primes` CRT primes (`>= 2`) and a gadget of
    /// `columns` digits, in a ring of dimension `coefficients`: the fresh noise
    /// is CBD(`eta`) on the keys of shape `fresh_keys` (`columns *
    /// coefficients` members), the decoder of prime `i` smudges every column
    /// with `digits` digits on the keys of shape `smudge_keys[i]` (`columns *
    /// coefficients * digits` members, column `j` the members from `j
    /// coefficients digits`), and `A'` is sampled under `matrix_tag`. The
    /// keys must be pairwise distinct; the specification commits to every key
    /// set.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_program: &ExponentLutProgram,
        target_output: WireId,
        profile: SparseLwrPrfProfile,
        eta: u32,
        primes: usize,
        columns: usize,
        digits: usize,
        fresh_keys: PrfKeyShape,
        smudge_keys: Vec<PrfKeyShape>,
        matrix_tag: Vec<u8>,
    ) -> Result<Self, RefreshError> {
        Self::build(
            source_program,
            target_output,
            profile,
            eta,
            primes,
            columns,
            digits,
            fresh_keys,
            smudge_keys,
            matrix_tag,
            None,
        )
    }

    /// Part `(output_prime, component)` of the refresh to the top level
    /// committed by `top` (see [`top::TopRefreshDeclaration`]): the decoding
    /// level has `primes` primes, the refreshed row has the `columns()` of
    /// `reference`, CBD(`eta`) noise on the keys of shape `fresh_keys` (one
    /// polynomial per column), the decoder of `output_prime` on the keys of
    /// shape `smudge_keys[0]`, the mask child's binary samples on the keys of
    /// shape `mask_keys` (`n` members, the bit modulo the output prime
    /// `q_min`), and `A'` under `matrix_tag`; `output_moduli` are the
    /// refreshed primes (the output prime among them).
    #[allow(clippy::too_many_arguments)]
    pub fn top_part(
        source_program: &ExponentLutProgram,
        target_output: WireId,
        profile: SparseLwrPrfProfile,
        eta: u32,
        primes: usize,
        output_prime: (usize, u64),
        output_moduli: Vec<u64>,
        component: usize,
        digits: usize,
        reference: TopReference,
        mask_keys: PrfKeyShape,
        fresh_keys: PrfKeyShape,
        smudge_keys: Vec<PrfKeyShape>,
        matrix_tag: Vec<u8>,
        top: [u8; 32],
    ) -> Result<Self, RefreshError> {
        let coefficients =
            profile.ring_dimension().map_err(|_| RefreshError::InvalidDeclaration)?;
        if component > 1 || output_prime.0 >= primes || !output_moduli.contains(&output_prime.1) {
            return Err(RefreshError::InvalidDeclaration);
        }
        // One polynomial of `n` bits `h (1 +- 1)`, `h = 2^-1 mod q`.
        let mask = FreshNoiseProgram::with_distribution(
            profile,
            PrfDistribution::Binary { q_min: output_prime.1 },
            1,
            coefficients,
            mask_keys,
        )?;
        let columns = reference.columns();
        Self::build(
            source_program,
            target_output,
            profile,
            eta,
            primes,
            columns,
            digits,
            fresh_keys,
            smudge_keys,
            matrix_tag,
            Some(TopRefreshPart {
                top,
                component,
                output_prime: output_prime.0,
                output_modulus: output_prime.1,
                output_moduli,
                reference,
                mask,
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        source_program: &ExponentLutProgram,
        target_output: WireId,
        profile: SparseLwrPrfProfile,
        eta: u32,
        primes: usize,
        columns: usize,
        digits: usize,
        fresh_keys: PrfKeyShape,
        smudge_keys: Vec<PrfKeyShape>,
        matrix_tag: Vec<u8>,
        top: Option<TopRefreshPart>,
    ) -> Result<Self, RefreshError> {
        let coefficients =
            profile.ring_dimension().map_err(|_| RefreshError::InvalidDeclaration)?;
        let decoded = top.as_ref().map_or(primes, |_| 1);
        if primes < 2 || smudge_keys.len() != decoded {
            return Err(RefreshError::InvalidDeclaration);
        }
        let fresh = FreshNoiseProgram::new(profile, eta, columns, coefficients, fresh_keys)?;
        let smudges = smudge_keys
            .into_iter()
            .map(|keys| {
                if Some(keys.members()) != columns.checked_mul(coefficients * digits) {
                    return Err(RefreshError::InvalidDeclaration);
                }
                Ok(DecoderSmudgeProgram::new(&profile, keys, columns)?)
            })
            .collect::<Result<Vec<_>, RefreshError>>()?;
        // The children's Programs depend on their key shapes only; the
        // shapes' key-set commitments bind the specification, and so every
        // hash namespace derived from it, to the key sets (the decoders'
        // commitments cover their smudge keys).
        let mask_keys = top.as_ref().map(|part| part.mask.keys()).transpose()?;
        let spec = commitment(&(
            b"mxx-exponent-lut/refresh/spec/v2".as_slice(),
            source_program.id(),
            target_output,
            &profile,
            eta,
            primes,
            columns,
            coefficients,
            digits,
            &matrix_tag,
            fresh.id(),
            fresh.keys()?,
            smudges.iter().map(DecoderSmudgeProgram::id).collect::<Vec<_>>(),
            smudges
                .iter()
                .map(|smudge| smudge.prf().key_shape())
                .collect::<Option<Vec<_>>>()
                .ok_or(RefreshError::InvalidDeclaration)?,
            mask_keys,
            top.as_ref().map(|part| {
                (
                    part.top,
                    part.component,
                    part.output_prime,
                    part.output_modulus,
                    &part.output_moduli,
                    &part.reference,
                    part.mask.id(),
                )
            }),
        ))?;
        let decoded_primes =
            top.as_ref().map_or_else(|| (0..primes).collect(), |part| vec![part.output_prime]);
        let decoders = smudges
            .into_iter()
            .zip(decoded_primes)
            .map(|(smudge, prime)| {
                Ok(DecoderDeclaration::from_smudge(
                    source_program,
                    target_output,
                    profile,
                    (columns, coefficients, digits),
                    smudge,
                    DecoderTarget::Refresh { spec, prime },
                )?)
            })
            .collect::<Result<Vec<_>, RefreshError>>()?;
        Ok(Self {
            source_program: source_program.id(),
            target_output,
            profile,
            eta,
            primes,
            columns,
            coefficients,
            digits,
            matrix_tag,
            fresh,
            decoders,
            spec,
            top,
        })
    }

    pub const fn source_program(&self) -> ExponentLutProgramId {
        self.source_program
    }

    pub const fn target_output(&self) -> WireId {
        self.target_output
    }

    pub const fn profile(&self) -> &SparseLwrPrfProfile {
        &self.profile
    }

    /// The CBD parameter of the fresh noise: `||e'||_inf <= eta`.
    pub const fn eta(&self) -> u32 {
        self.eta
    }

    /// The number `L` of CRT primes of the decoding modulus.
    pub const fn primes(&self) -> usize {
        self.primes
    }

    /// The number of columns of the refreshed row: the decoding level's `m`
    /// gadget columns, or the reference's `w_0` for a top-refresh part.
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// The top-refresh part this declaration is, if any.
    pub const fn top(&self) -> Option<&TopRefreshPart> {
        self.top.as_ref()
    }

    /// The refreshed component of the target: 1, or the part's component.
    pub fn component(&self) -> usize {
        self.top.as_ref().map_or(1, TopRefreshPart::component)
    }

    /// The primes decoded, as indices into the decoding level's primes: all
    /// of them, or the part's output prime.
    pub fn decoded_primes(&self) -> Vec<usize> {
        self.top.as_ref().map_or_else(|| (0..self.primes).collect(), |part| vec![part.output_prime])
    }

    /// The ring dimension `n`.
    pub const fn coefficients(&self) -> usize {
        self.coefficients
    }

    /// The smudge digit count of every decoder.
    pub const fn digits(&self) -> usize {
        self.digits
    }

    /// The tag of the public matrix `A'`.
    pub fn matrix_tag(&self) -> &[u8] {
        &self.matrix_tag
    }

    pub const fn fresh(&self) -> &FreshNoiseProgram {
        &self.fresh
    }

    /// The decoder of a decoded prime.
    pub fn decoder(&self, prime: usize) -> Option<&DecoderDeclaration> {
        let position = self.decoded_primes().iter().position(|decoded| *decoded == prime)?;
        self.decoders.get(position)
    }

    /// Every decoder, the decoder of `prime` at the position of `prime`
    /// among [`Self::decoded_primes`].
    pub fn decoders(&self) -> &[DecoderDeclaration] {
        &self.decoders
    }

    /// The specification commitment (see [`DecoderTarget::Refresh`]).
    pub const fn spec(&self) -> &[u8; 32] {
        &self.spec
    }

    /// The hash tag of the PRF label of an invocation with label tag
    /// `label_tag`.
    pub fn label_hash_tag(&self, label_tag: &[u8]) -> Vec<u8> {
        match &self.top {
            // Every part of a top refresh shares the top refresh's label.
            Some(part) => top::label_hash_tag(&part.top, label_tag),
            None => [b"refresh/v1/label".as_slice(), self.spec.as_slice(), label_tag].concat(),
        }
    }

    /// The PRF label `a in Z_Q'^v` of an invocation with label tag
    /// `label_tag` under the stage hash key: uniform weak-PRF input of every
    /// fresh-noise and smudge key (random-oracle model).
    pub fn label(&self, hash_key: [u8; 32], label_tag: &[u8]) -> Result<Vec<usize>, RefreshError> {
        let q = self.profile.q().map_err(|_| RefreshError::InvalidDeclaration)?;
        sample_hash_integers::<keccak_asm::Keccak256>(
            hash_key,
            &self.label_hash_tag(label_tag),
            self.profile.lwr_dimension,
            &BigUint::from(q),
        )
        .into_iter()
        .map(|value| usize::try_from(value).map_err(|_| RefreshError::InvalidDeclaration))
        .collect()
    }

    /// The hash tag of the public matrix `A'`: the matrix tag, and for a part of a refresh to the
    /// top level its component as an integer component (parts differ only in integers, so their
    /// stages share one structure).
    pub fn matrix_hash_tag(&self) -> mxx_dsl::HashTag {
        let mut tag = mxx_dsl::HashTag::from(
            [b"refresh/v1/A".as_slice(), self.matrix_tag.as_slice()].concat(),
        );
        if self.top.is_some() {
            tag.push("component");
            tag.push(mxx_ir_core::IntExpr::constant(self.component()));
        }
        tag
    }
}

fn commitment(value: &impl Serialize) -> Result<[u8; 32], RefreshError> {
    let bytes = serde_json::to_vec(value).map_err(|_| RefreshError::InvalidDeclaration)?;
    Ok(Sha256::digest(bytes).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decoder::decoding::fixture::{keys, profile, shape},
        program::AttributeKnowledge,
    };

    fn identity() -> (ExponentLutProgram, WireId) {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        (builder.finish(vec![Operand::Scalar(input)]).unwrap(), input)
    }

    /// The key shape of `count` fresh keys.
    fn key_shape(count: usize) -> PrfKeyShape {
        shape(&keys(count))
    }

    /// Two primes, two columns, one digit: the fresh program has two outputs
    /// of `n` members each, and decoder `i` targets prime `i` under the
    /// specification commitment with `2 n` smudge members over both columns;
    /// labels are reproducible.
    #[test]
    fn declaration_lays_out_fresh_outputs_and_decoders() {
        let (source, input) = identity();
        let n = profile().ring_dimension().unwrap();
        let declaration = RefreshDeclaration::new(
            &source,
            input,
            profile(),
            1,
            2,
            2,
            1,
            key_shape(2 * n),
            (0..2).map(|_| key_shape(2 * n)).collect(),
            b"tag".to_vec(),
        )
        .unwrap();
        assert_eq!(declaration.fresh().columns(), 2);
        assert_eq!(declaration.fresh().sampler().prf().members(), 2 * n);
        for prime in 0..2 {
            let decoder = declaration.decoder(prime).unwrap();
            assert_eq!(
                decoder.target(),
                DecoderTarget::Refresh { spec: *declaration.spec(), prime }
            );
            assert_eq!((decoder.columns(), decoder.smudge().members()), (2, 2 * n));
        }
        assert!(declaration.decoder(2).is_none());
        let label = declaration.label([7; 32], b"nu").unwrap();
        assert_eq!(label, declaration.label([7; 32], b"nu").unwrap());
        assert_ne!(label, declaration.label([7; 32], b"other").unwrap());
        let q = profile().q().unwrap();
        assert!(label.len() == profile().lwr_dimension && label.iter().all(|value| *value < q));
    }

    /// One prime, a missing smudge key set, a short fresh key set or a
    /// smudge set of the wrong size is rejected; so is a coefficient target.
    #[test]
    fn declaration_rejects_malformed_refreshes() {
        let (source, input) = identity();
        let n = profile().ring_dimension().unwrap();
        let build = |primes: usize, fresh: usize, smudges: usize, smudge: usize| {
            RefreshDeclaration::new(
                &source,
                input,
                profile(),
                1,
                primes,
                1,
                1,
                key_shape(fresh),
                (0..smudges).map(|_| key_shape(smudge)).collect(),
                Vec::new(),
            )
        };
        assert!(build(2, n, 2, n).is_ok());
        assert_eq!(build(1, n, 1, n).unwrap_err(), RefreshError::InvalidDeclaration);
        assert_eq!(build(2, n, 1, n).unwrap_err(), RefreshError::InvalidDeclaration);
        assert_eq!(build(2, n - 1, 2, n).unwrap_err(), RefreshError::InvalidDeclaration);
        assert_eq!(build(2, n, 2, n + 1).unwrap_err(), RefreshError::InvalidDeclaration);
    }
}
