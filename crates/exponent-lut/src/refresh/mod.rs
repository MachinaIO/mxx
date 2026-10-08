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
//! Column `j` of `d_i` is decoded by its own smudged trapdoor decoder
//! ([`crate::decoder`], target [`DecoderTarget::Refresh`]) to `y_j mod q_i`,
//! and a CRT recomposition over the primes returns `y_j`. Preprocessing
//! computes the decoded keys `K_i` by the same steps on the public keys to
//! sample the decoders' preimages; the output key `A'` is public.
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

use crate::{
    decoder::decoding::{DecoderDeclaration, DecoderError, DecoderSmudgeProgram, DecoderTarget},
    program::{
        ExponentLutProgram, ExponentLutProgramBuilder, ExponentLutProgramId, FamilyRange, Operand,
        ProgramValidationError, WireId,
    },
    program_gadgets::{
        pbc::PbcPublicLayout,
        prf::SparseLwrPrfProfile,
        prf_sampler::{PrfDistribution, PrfSampler, PrfSamplerError},
    },
};
use mxx_backends::sampler::hash::sample_hash_integers;
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

/// The fresh-noise child Program: `m` outputs, output `j` the polynomial
/// `sum_k e'_(j n + k) X^k` of `n` CBD samples of one PRF member and key each
/// (`m n` members in one [`PrfSampler`] family). Its coefficient outputs are
/// raw carriers with payload `e'_j G` (scale 1); the refresh moves them down
/// to the decoding ring once and never finalizes them for decryption.
#[derive(Clone, Debug)]
pub struct FreshNoiseProgram {
    program: ExponentLutProgram,
    sampler: PrfSampler,
    outputs: Vec<WireId>,
}

impl FreshNoiseProgram {
    /// The `columns` polynomials of `coefficients` CBD(`eta`) samples on
    /// `profile`, member `j coefficients + k` on `layouts[j coefficients + k]`.
    pub fn new(
        profile: SparseLwrPrfProfile,
        eta: u32,
        columns: usize,
        coefficients: usize,
        layouts: Vec<PbcPublicLayout>,
    ) -> Result<Self, RefreshError> {
        let members = columns.checked_mul(coefficients).ok_or(RefreshError::InvalidDeclaration)?;
        if columns == 0 ||
            coefficients == 0 ||
            members < 2 ||
            layouts.len() != members ||
            Some(coefficients) != profile.ring_dimension().ok()
        {
            return Err(RefreshError::InvalidDeclaration);
        }
        let mut builder = ExponentLutProgramBuilder::new();
        let sampler =
            PrfSampler::build(&mut builder, profile, layouts, PrfDistribution::Cbd { eta })?;
        let crate::program_gadgets::prf::PrfOutput::Family(samples) = sampler.output() else {
            return Err(RefreshError::InvalidDeclaration);
        };
        let outputs = (0..columns)
            .map(|column| {
                let range = FamilyRange::full(IntExpr::constant(coefficients))?;
                let placed = builder.parallel(range, |builder, coefficient| {
                    let member = IntExpr::Add(
                        Box::new(IntExpr::constant(column * coefficients)),
                        Box::new(coefficient.clone()),
                    )
                    .canonicalize();
                    builder
                        .mul_small_constant(
                            Operand::Member { family: samples, index: member },
                            vec![1],
                            coefficient,
                        )
                        .map(Operand::Scalar)
                })?;
                builder.sum(placed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let program = builder.finish(outputs.iter().copied().map(Operand::Scalar).collect())?;
        Ok(Self { program, sampler, outputs })
    }

    pub fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    pub fn id(&self) -> ExponentLutProgramId {
        self.program.id()
    }

    /// The CBD sampler of the `m n` members.
    pub fn sampler(&self) -> &PrfSampler {
        &self.sampler
    }

    /// The polynomial outputs, one per column.
    pub fn outputs(&self) -> &[WireId] {
        &self.outputs
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
    /// Decoder `(prime, column)` at `prime * columns + column`.
    decoders: Vec<DecoderDeclaration>,
    spec: [u8; 32],
}

impl RefreshDeclaration {
    /// Check and freeze the refresh of `target_output` of `source_program`
    /// over a decoding modulus of `primes` CRT primes (`>= 2`) and a gadget of
    /// `columns` digits, in a ring of dimension `coefficients`: the fresh noise
    /// is CBD(`eta`) on `fresh_layouts` (`columns * coefficients` members),
    /// decoder `(i, j)` smudges with `digits` digits on
    /// `smudge_layouts[i * columns + j]` (`coefficients * digits` members),
    /// and `A'` is sampled under `matrix_tag`. Every layout carries its own
    /// key; the keys must be pairwise distinct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_program: &ExponentLutProgram,
        target_output: WireId,
        profile: SparseLwrPrfProfile,
        eta: u32,
        primes: usize,
        columns: usize,
        digits: usize,
        fresh_layouts: Vec<PbcPublicLayout>,
        smudge_layouts: Vec<Vec<PbcPublicLayout>>,
        matrix_tag: Vec<u8>,
    ) -> Result<Self, RefreshError> {
        let coefficients =
            profile.ring_dimension().map_err(|_| RefreshError::InvalidDeclaration)?;
        if primes < 2 || Some(smudge_layouts.len()) != primes.checked_mul(columns) {
            return Err(RefreshError::InvalidDeclaration);
        }
        let fresh = FreshNoiseProgram::new(profile, eta, columns, coefficients, fresh_layouts)?;
        let smudges = smudge_layouts
            .into_iter()
            .map(|layouts| {
                if layouts.len() != coefficients * digits {
                    return Err(RefreshError::InvalidDeclaration);
                }
                Ok(DecoderSmudgeProgram::new(&profile, layouts)?)
            })
            .collect::<Result<Vec<_>, RefreshError>>()?;
        let spec = commitment(&(
            b"mxx-exponent-lut/refresh/spec/v1".as_slice(),
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
            smudges.iter().map(DecoderSmudgeProgram::id).collect::<Vec<_>>(),
        ))?;
        let decoders = smudges
            .into_iter()
            .enumerate()
            .map(|(index, smudge)| {
                Ok(DecoderDeclaration::from_smudge(
                    source_program,
                    target_output,
                    profile,
                    coefficients,
                    digits,
                    smudge,
                    DecoderTarget::Refresh {
                        spec,
                        prime: index / columns,
                        column: index % columns,
                    },
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

    /// The number `m` of gadget columns of the refreshed row.
    pub const fn columns(&self) -> usize {
        self.columns
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

    /// Decoder `(prime, column)`.
    pub fn decoder(&self, prime: usize, column: usize) -> Option<&DecoderDeclaration> {
        (column < self.columns).then(|| self.decoders.get(prime * self.columns + column)).flatten()
    }

    /// Every decoder, `(prime, column)` at `prime * columns + column`.
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
        [b"refresh/v1/label".as_slice(), self.spec.as_slice(), label_tag].concat()
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

    /// The hash tag of the public matrix `A'`.
    pub fn matrix_hash_tag(&self) -> Vec<u8> {
        [b"refresh/v1/A".as_slice(), self.matrix_tag.as_slice()].concat()
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
        decoder::decoding::fixture::{keys, profile},
        program::AttributeKnowledge,
    };

    fn identity() -> (ExponentLutProgram, WireId) {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        (builder.finish(vec![Operand::Scalar(input)]).unwrap(), input)
    }

    fn layouts(count: usize) -> Vec<PbcPublicLayout> {
        keys(count).into_iter().map(|key| key.public_layout).collect()
    }

    /// Two primes, two columns, one digit: the fresh program has two outputs
    /// of `n` members each, and decoder `(i, j)` targets prime `i` and column
    /// `j` under the specification commitment; labels are reproducible.
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
            layouts(2 * n),
            (0..4).map(|_| layouts(n)).collect(),
            b"tag".to_vec(),
        )
        .unwrap();
        assert_eq!(declaration.fresh().outputs().len(), 2);
        assert_eq!(declaration.fresh().sampler().prf().members(), 2 * n);
        for prime in 0..2 {
            for column in 0..2 {
                let decoder = declaration.decoder(prime, column).unwrap();
                assert_eq!(
                    decoder.target(),
                    DecoderTarget::Refresh { spec: *declaration.spec(), prime, column }
                );
                assert_eq!(decoder.smudge().members(), n);
            }
        }
        assert!(declaration.decoder(0, 2).is_none() && declaration.decoder(2, 0).is_none());
        let label = declaration.label([7; 32], b"nu").unwrap();
        assert_eq!(label, declaration.label([7; 32], b"nu").unwrap());
        assert_ne!(label, declaration.label([7; 32], b"other").unwrap());
        let q = profile().q().unwrap();
        assert!(label.len() == profile().lwr_dimension && label.iter().all(|value| *value < q));
    }

    /// One prime, a missing smudge layout set, a short fresh layout set or a
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
                layouts(fresh),
                (0..smudges).map(|_| layouts(smudge)).collect(),
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
