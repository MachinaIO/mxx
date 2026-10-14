//! One part of the refresh: the per-chain decoding step.
//!
//! Part `(chain, component)` of a [`RefreshDeclaration`](super::RefreshDeclaration) runs on the
//! chain of its output prime `q` and decodes `y_j mod q` (`j` the component). Its target is the
//! private Program output `c_x` at a non-terminal level `l` with modulus `Q = q_1 ... q_L`
//! (`L >= 2`, `q` among the primes) whose payload is `Delta_l r_j x G^st` (`G^st` the level's state
//! gadget, the gadget of level `l + 1`, `Delta_l = Q / Q_(l+1)`).
//!
//! # Computation
//!
//! With the one encoding `[g | c_1]`, `g = s K_g - v G_Q + e_g` the caller's encoding of
//! `v = sigma_0` modulo `q` (see [Caller inputs](super#caller-inputs)) and `c_1` the second block
//! of a public-payload encoding of `1` at level `l`, and the fresh-noise carriers
//! `f_k = s K_(f,k) - e'_k G_Q + e_(f,k)` (scale 1), for `Q_q = Q / q`:
//!
//! - `u = [g | c_1] (I_2 ⊗ G_Q)^-1(Q_q A'_(j)) = s A_1 X - Q_q sigma A'_(j) + e_1 X` modulo `q`;
//! - `v = c_x,j G^st^-1(R_q) = s K_x Y - Q_q Delta_0 r_j x G^st_0 + e_x Y`, exact modulo `Q` with
//!   the target weight `R_q` (see `RnsBggParameters::top_target_weight`);
//! - `w = [f_k G_Q^-1(Q_q)]_k = s K_f Z - Q_q e'_j + e_f Z`;
//! - `d = u - v + w = s K - Q_q y_j + E`.
//!
//! The row `d` is decoded by one smudged trapdoor decoder ([`crate::decoder`], target
//! [`DecoderTarget::Refresh`]) to `y_j mod q`: every column has its own smudge keys and its own
//! preimage column, so the columns are decoded independently, but as one `1 x w_0` row (one
//! smudge family over the columns, one preimage of `w_0` columns). Preprocessing computes the
//! decoded key `K` by the same steps on the public keys (the caller's `K_g` and `A'`) to sample
//! the decoder's preimage; the output key `A'_(j)` is the caller's public input.

use super::{FreshNoiseProgram, RefreshError, TopReference, commitment};
use crate::{
    decoder::decoding::{DecoderDeclaration, DecoderSmudgeProgram, DecoderTarget},
    program::{ExponentLutProgram, ExponentLutProgramId, WireId},
    program_gadgets::prf::{PrfKeyShape, SparseLwrPrfProfile},
};
use mxx_backends::sampler::hash::sample_hash_integers;
use num_bigint::BigUint;
use sha2::{Digest, Sha256};

/// Immutable, committed declaration of part `(chain, component)` of a refresh.
#[derive(Clone, Debug)]
pub(crate) struct RefreshPartDeclaration {
    source_program: ExponentLutProgramId,
    target_output: WireId,
    profile: SparseLwrPrfProfile,
    eta: u32,
    primes: usize,
    columns: usize,
    coefficients: usize,
    fresh: FreshNoiseProgram,
    decoder: DecoderDeclaration,
    spec: [u8; 32],
    /// The commitment of the whole refresh: it names the label.
    refresh: [u8; 32],
    component: usize,
    output_prime: usize,
    output_modulus: u64,
    output_moduli: Vec<u64>,
    reference: TopReference,
}

impl RefreshPartDeclaration {
    /// Part `(output_prime, component)` of the refresh committed by `refresh`: the decoding level
    /// has `primes` primes (`>= 2`), the refreshed row has the `columns()` of `reference`,
    /// CBD(`eta`) noise on the keys of shape `fresh_keys` (one polynomial per column), the decoder
    /// of the output prime `output_prime = (index, q)` (an index into the decoding level's
    /// primes) smudges with `digits` digits on the keys of shape `smudge_keys` (`w_0 n d` members,
    /// column `k` the members from `k n d`); `output_moduli` are the refreshed primes (`q` among
    /// them).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
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
        fresh_keys: PrfKeyShape,
        smudge_keys: PrfKeyShape,
        refresh: [u8; 32],
    ) -> Result<Self, RefreshError> {
        let coefficients =
            profile.ring_dimension().map_err(|_| RefreshError::InvalidDeclaration)?;
        if component > 1 || output_prime.0 >= primes || !output_moduli.contains(&output_prime.1) {
            return Err(RefreshError::InvalidDeclaration);
        }
        let columns = reference.columns();
        if primes < 2 {
            return Err(RefreshError::InvalidDeclaration);
        }
        let fresh = FreshNoiseProgram::new(profile, eta, columns, coefficients, fresh_keys)?;
        if Some(smudge_keys.members()) != columns.checked_mul(coefficients * digits) {
            return Err(RefreshError::InvalidDeclaration);
        }
        let smudge = DecoderSmudgeProgram::new(&profile, smudge_keys, columns)?;
        // The children's Programs depend on their key shapes only; the shapes' key-set
        // commitments bind the specification, and so every hash namespace derived from it, to
        // the key sets (the decoder's commitment covers its smudge keys). The smudge Programs and
        // their key shapes are lists of one decoder.
        let spec = commitment(&(
            b"mxx-exponent-lut/refresh/spec/v3".as_slice(),
            source_program.id(),
            target_output,
            &profile,
            eta,
            primes,
            columns,
            coefficients,
            digits,
            fresh.id(),
            fresh.keys()?,
            vec![smudge.id()],
            vec![smudge.prf().key_shape().ok_or(RefreshError::InvalidDeclaration)?],
            (refresh, component, output_prime.0, output_prime.1, &output_moduli, &reference),
        ))?;
        let decoder = DecoderDeclaration::from_smudge(
            source_program,
            target_output,
            profile,
            (columns, coefficients, digits),
            smudge,
            DecoderTarget::Refresh { spec, prime: output_prime.0 },
        )?;
        Ok(Self {
            source_program: source_program.id(),
            target_output,
            profile,
            eta,
            primes,
            columns,
            coefficients,
            fresh,
            decoder,
            spec,
            refresh,
            component,
            output_prime: output_prime.0,
            output_modulus: output_prime.1,
            output_moduli,
            reference,
        })
    }

    pub(crate) const fn source_program(&self) -> ExponentLutProgramId {
        self.source_program
    }

    pub(crate) const fn target_output(&self) -> WireId {
        self.target_output
    }

    /// The CBD parameter of the fresh noise: `||e'||_inf <= eta`.
    pub(crate) const fn eta(&self) -> u32 {
        self.eta
    }

    /// The number of CRT primes of the decoding modulus.
    pub(crate) const fn primes(&self) -> usize {
        self.primes
    }

    /// The number of columns of the refreshed row: the reference's `w_0`.
    pub(crate) const fn columns(&self) -> usize {
        self.columns
    }

    /// The ring dimension `n`.
    pub(crate) const fn coefficients(&self) -> usize {
        self.coefficients
    }

    /// The fresh-noise child.
    pub(crate) const fn fresh(&self) -> &FreshNoiseProgram {
        &self.fresh
    }

    /// The decoder of the output prime.
    pub(crate) const fn decoder(&self) -> &DecoderDeclaration {
        &self.decoder
    }

    /// The specification commitment (see [`DecoderTarget::Refresh`]).
    pub(crate) const fn spec(&self) -> &[u8; 32] {
        &self.spec
    }

    /// The refreshed component `j` of the target.
    pub(crate) const fn component(&self) -> usize {
        self.component
    }

    /// The output prime, as an index into the decoding level's primes.
    pub(crate) const fn output_prime(&self) -> usize {
        self.output_prime
    }

    /// The output prime `q` itself.
    pub(crate) const fn output_modulus(&self) -> u64 {
        self.output_modulus
    }

    /// The refreshed primes of the refresh, in canonical order: their product `P` is the modulus
    /// of the recombined rows, to which every part contributes its residue's CRT term.
    pub(crate) fn output_moduli(&self) -> &[u64] {
        &self.output_moduli
    }

    pub(crate) const fn reference(&self) -> &TopReference {
        &self.reference
    }

    /// The PRF label `a in Z_Q'^v` of an invocation with label tag `label_tag` under the stage
    /// hash key: uniform weak-PRF input of every fresh-noise and smudge key (random-oracle
    /// model), shared by every part of the refresh.
    pub(crate) fn label(
        &self,
        hash_key: [u8; 32],
        label_tag: &[u8],
    ) -> Result<Vec<usize>, RefreshError> {
        let q = self.profile.q().map_err(|_| RefreshError::InvalidDeclaration)?;
        sample_hash_integers::<keccak_asm::Keccak256>(
            hash_key,
            &[b"refresh-top/v1/label".as_slice(), self.refresh.as_slice(), label_tag].concat(),
            self.profile.lwr_dimension,
            &BigUint::from(q),
        )
        .into_iter()
        .map(|value| usize::try_from(value).map_err(|_| RefreshError::InvalidDeclaration))
        .collect()
    }

    /// The hash namespace of the fresh-noise child (the decoder's smudge child uses its decoder
    /// commitment).
    pub(crate) fn fresh_namespace(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/refresh/fresh-namespace/v1");
        digest.update(self.spec);
        digest.finalize().into()
    }
}
