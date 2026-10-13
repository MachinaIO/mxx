//! The RNS-BGG arithmetic of the noise refresh (see [`crate::refresh`]).
//!
//! The stages share the public pieces defined here: the hash-sampled `A'`
//! and one-encoding key, the digit actions `X_i, Y_i, z_i` of every prime,
//! the combination `d_i` (the same on keys and rows), the CRT coefficients,
//! and the refresh children's chains, hash namespaces and artifact names.
use super::*;
use crate::{
    program::ExponentLutProgram,
    refresh::{RefreshDeclaration, TopReference, top::TopRefreshDeclaration},
};

/// `refresh/<spec>/<role>`: the namespace of every refresh artifact.
pub(crate) fn refresh_artifact_name(spec: [u8; 32], role: &str) -> String {
    format!("refresh/{}/{role}", decoder_commitment_hex(spec))
}

/// `refresh/<spec>/fresh/<role>`: an artifact of the fresh-noise child.
pub(crate) fn refresh_fresh_artifact_name(spec: [u8; 32], role: &str) -> String {
    refresh_artifact_name(spec, &format!("fresh/{role}"))
}

/// The public-key artifact `port` of fresh-noise child input `wire`.
pub(crate) fn refresh_fresh_input_key_artifact_name(
    spec: [u8; 32],
    wire: WireId,
    port: usize,
) -> String {
    refresh_fresh_artifact_name(spec, &format!("input-key/{wire:?}.{port}"))
}

/// `refresh/<spec>/mask/<role>`: an artifact of a top-refresh part's mask
/// child.
pub(crate) fn refresh_mask_artifact_name(spec: [u8; 32], role: &str) -> String {
    refresh_artifact_name(spec, &format!("mask/{role}"))
}

/// The public-key artifact `port` of mask child input `wire`.
pub(crate) fn refresh_mask_input_key_artifact_name(
    spec: [u8; 32],
    wire: WireId,
    port: usize,
) -> String {
    refresh_mask_artifact_name(spec, &format!("input-key/{wire:?}.{port}"))
}

/// The hash namespace of a top-refresh part's mask child.
pub fn refresh_mask_namespace(declaration: &RefreshDeclaration) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-exponent-lut/refresh/mask-namespace/v1");
    digest.update(declaration.spec());
    digest.finalize().into()
}

/// The hash namespace of `declaration`'s fresh-noise child (each decoder's
/// smudge child uses its decoder commitment): a refresh child's stage
/// builders take it through `with_hash_namespace`.
pub fn refresh_fresh_namespace(declaration: &RefreshDeclaration) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-exponent-lut/refresh/fresh-namespace/v1");
    digest.update(declaration.spec());
    digest.finalize().into()
}

/// The hash key a child with hash namespace `namespace` samples its public
/// matrices from: independent of the stage key's other uses, so children of
/// one shape (same wire, RHS and LUT ids) never share a hash-derived matrix.
pub(crate) fn child_hash_key(hash_key: &[u8], namespace: &[u8; 32]) -> Vec<u8> {
    let mut digest = Sha256::new();
    digest.update(b"mxx-exponent-lut/child-hash-key/v1");
    digest.update(namespace);
    digest.update(hash_key);
    digest.finalize().to_vec()
}

/// The graph input carrying the hash key of `namespace` (the stage key for
/// `None`).
pub(crate) fn hash_key_input_name(namespace: Option<&[u8; 32]>) -> String {
    match namespace {
        None => HASH_KEY_INPUT.to_owned(),
        Some(namespace) => format!("{HASH_KEY_INPUT}/{}", decoder_commitment_hex(*namespace)),
    }
}

/// The digit actions of one prime `q_i` (`Q_i = Q / q_i`): `x` is
/// `(I_2 ⊗ G_Q)^-1(Q_i A')` (`2m x m`), `y` is
/// `G^st^-1(Q_i G_Q mod q_(l+1))` (`w x m`) and `z` is `G_Q^-1(Q_i)` (`m x 1`).
pub(crate) struct RefreshActions {
    pub(crate) x: SmallMatrix,
    pub(crate) y: SmallMatrix,
    pub(crate) z: SmallMatrix,
}

impl RnsBggParameters {
    /// The decoding level `l` of `declaration`'s target in `source_program`,
    /// after checking the refresh against these parameters: Static RHS mode,
    /// a non-terminal level whose modulus has the declared prime count and
    /// gadget column count, every smudge child's ring dimension, and a fresh
    /// noise bound below every prime.
    pub(crate) fn refresh_level(
        &self,
        declaration: &RefreshDeclaration,
        source_program: &ExponentLutProgram,
    ) -> Result<usize, RnsBggError> {
        let invalid = RnsBggError::Parameters("the refresh does not match its parameters");
        if self.rhs_mode != RhsMode::Static || source_program.id() != declaration.source_program() {
            return Err(invalid);
        }
        let level = RnsBggDerivedWorkload::from_program(source_program)?
            .level(declaration.target_output())?;
        let parameters = self.level(level)?;
        let primes = parameters.to_crt().0;
        if level + 1 >= self.levels.len() ||
            primes.len() != declaration.primes() ||
            parameters.ring_dimension() as usize != declaration.coefficients() ||
            primes.iter().any(|prime| *prime <= u64::from(declaration.eta()))
        {
            return Err(invalid);
        }
        let shape = match declaration.top() {
            None => parameters.modulus_digits() == declaration.columns(),
            // The output prime is the decoded prime and survives into the
            // plaintext modulus of the level (the target action inverts the
            // level's scale modulo it).
            Some(part) => {
                primes.get(part.output_prime()) == Some(&part.output_modulus()) &&
                    self.level(level + 1)?.to_crt().0.contains(&part.output_modulus()) &&
                    part.reference().ring_dimension == parameters.ring_dimension() &&
                    part.reference().top.contains(&part.output_modulus())
            }
        };
        if !shape {
            return Err(invalid);
        }
        Ok(level)
    }

    /// The fresh-noise child's chain (see [`Self::child_chain`]): its outputs
    /// are raw coefficient carriers at their LUT input level, moved down once
    /// to the decoding level.
    pub fn refresh_fresh_parameters(
        &self,
        declaration: &RefreshDeclaration,
        source_program: &ExponentLutProgram,
        pool: &DCRTPolyParams,
        primes_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        let level = self.refresh_level(declaration, source_program)?;
        self.child_chain(level, Self::refresh_fresh_level(declaration)?, pool, primes_per_drop)
    }

    /// The common LUT input level of the fresh-noise child's outputs.
    pub(crate) fn refresh_fresh_level(
        declaration: &RefreshDeclaration,
    ) -> Result<usize, RnsBggError> {
        noise_program_level(declaration.fresh())
    }

    /// The LUT input level of a top-refresh part's mask child output.
    pub(crate) fn refresh_mask_level(
        declaration: &RefreshDeclaration,
    ) -> Result<usize, RnsBggError> {
        let part =
            declaration.top().ok_or(RnsBggError::Parameters("the refresh has no mask child"))?;
        noise_program_level(part.mask())
    }

    /// The mask child's chain of a top-refresh part (see
    /// [`Self::refresh_fresh_parameters`]).
    pub fn refresh_mask_parameters(
        &self,
        declaration: &RefreshDeclaration,
        source_program: &ExponentLutProgram,
        pool: &DCRTPolyParams,
        primes_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        let level = self.refresh_level(declaration, source_program)?;
        self.child_chain(level, Self::refresh_mask_level(declaration)?, pool, primes_per_drop)
    }

    /// Check `fresh` and every chain of `smudges` (one per decoder, in
    /// decoder order) against `declaration`.
    pub(crate) fn validate_refresh_children(
        &self,
        declaration: &RefreshDeclaration,
        source_program: &ExponentLutProgram,
        mask: Option<&Self>,
        fresh: &Self,
        smudges: &[&Self],
    ) -> Result<usize, RnsBggError> {
        let level = self.refresh_level(declaration, source_program)?;
        self.validate_child_chain(fresh, level, Self::refresh_fresh_level(declaration)?)?;
        match (declaration.top(), mask) {
            (None, None) => {}
            (Some(_), Some(mask)) => {
                self.validate_child_chain(mask, level, Self::refresh_mask_level(declaration)?)?
            }
            _ => return Err(RnsBggError::Decoder(DecoderError::TargetMismatch)),
        }
        if smudges.len() != declaration.decoders().len() {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        for (smudge, decoder) in smudges.iter().zip(declaration.decoders()) {
            self.validate_decoder_smudge_parameters(smudge, decoder, source_program)?;
        }
        Ok(level)
    }

    /// The public `A' in R_Q^(2 x m)` of `declaration` at level `level`,
    /// sampled from `hash_key` (the stage key; for a top-refresh part the key
    /// of [`TopRefreshDeclaration::matrix_namespace`]). A top-refresh part's
    /// `A'_(j)` is sampled in the reference's level-0 ring, the same for every
    /// chain, and reduced to level `level`.
    pub(crate) fn refresh_matrix(
        &self,
        level: usize,
        hash_key: Bytes,
        declaration: &RefreshDeclaration,
    ) -> Result<Mat, RnsBggError> {
        let ring = self.ring(level)?;
        match declaration.top() {
            None => {
                let columns = self.level(level)?.modulus_digits();
                Ok(ring.hash_matrix(hash_key, declaration.matrix_hash_tag(), (2, columns)))
            }
            Some(part) => {
                Ok(top_matrix(part.reference(), hash_key, declaration).reduce_modulus(&ring))
            }
        }
    }

    /// The public key `A_1` of the one encoding at level `level`: a public
    /// input key schema sampled from the stage hash key under the refresh
    /// namespace.
    pub(crate) fn refresh_one_key(
        &self,
        level: usize,
        hash_key: Bytes,
        declaration: &RefreshDeclaration,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let ring = self.ring(level)?;
        let columns =
            self.key_component_columns(level, crate::program::AttributeKnowledge::PublicPayload)?;
        // The specification is an integer component, so parts differ only in integers.
        let mut tag = mxx_dsl::HashTag::from(b"refresh/v1/one".to_vec());
        super::push_digest(&mut tag, declaration.spec());
        Ok(super::public_key::matrix_from_hash(&ring, hash_key, tag, columns))
    }

    /// The digit actions of prime `prime` at level `level` for the public
    /// `a_prime = A'` of `declaration`.
    pub(crate) fn refresh_actions(
        &self,
        level: usize,
        prime: usize,
        a_prime: Mat,
        declaration: &RefreshDeclaration,
    ) -> Result<RefreshActions, RnsBggError> {
        let parameters = self.level(level)?;
        let modulus: BigUint = parameters.modulus().as_ref().clone();
        let q_i = parameters
            .to_crt()
            .0
            .get(prime)
            .copied()
            .ok_or(RnsBggError::Parameters("refresh prime index out of range"))?;
        let cofactor = &modulus / q_i;
        let cofactor_scalar = self.scalar(level, BigInt::from(cofactor.clone()))?;
        let x = self.gadget_inverse(level, level, cofactor_scalar.clone() * a_prime)?;
        let weight = match declaration.top() {
            None => (cofactor_scalar * self.public_gadget(level)?)
                .reduce_modulus(&self.ring(level + 1)?),
            Some(part) => self.top_target_weight(level, q_i, part.reference())?,
        };
        let y = self.gadget_inverse(level + 1, level, weight)?;
        let z = self.gadget_inverse(level, level, self.scalar(level, BigInt::from(cofactor))?)?;
        Ok(RefreshActions { x, y, z })
    }

    /// The target weight `R_q` of a top-refresh part decoding the prime `q`
    /// of level `level` (`1 x w_0`, ring of level `level + 1`):
    /// `R_q = (Delta_l^-1 Q_q T_q mod q)` modulo `q` and `0` modulo the other
    /// primes of level `l + 1`, with `T_q = Delta_0 G^st_0 mod q` of the
    /// reference. Then `Delta_l G^st_l G^-1(R_q) = Q_q T_q (mod Q)`: modulo
    /// `q` because `q` is a plaintext prime (`Delta_l` is invertible), and
    /// modulo every other prime both sides vanish (see [`crate::refresh::top`]).
    pub(crate) fn top_target_weight(
        &self,
        level: usize,
        q: u64,
        reference: &TopReference,
    ) -> Result<Mat, RnsBggError> {
        let invalid = RnsBggError::Parameters("the top refresh does not match its chain");
        let modulus: BigUint = self.level(level)?.modulus().as_ref().clone();
        let next_parameters = self.level(level + 1)?;
        let next_modulus: BigUint = next_parameters.modulus().as_ref().clone();
        let prime = BigUint::from(q);
        if !(&next_modulus % &prime).is_zero() {
            return Err(invalid);
        }
        let delta = &modulus / &next_modulus;
        let cofactor = &modulus / &prime;
        let kappa = (delta % &prime).modinv(&prime).ok_or(invalid)? * (cofactor % &prime) % &prime;
        let next_cofactor = &next_modulus / &prime;
        let idempotent = (&next_cofactor % &prime)
            .modinv(&prime)
            .ok_or(RnsBggError::Parameters("refresh CRT primes are not coprime"))? *
            &next_cofactor %
            &next_modulus;
        let ring_q = Ring::from_crt_moduli(vec![IntExpr::from(q)], reference.ring_dimension);
        let scalar = |ring: &Ring, value: BigUint| ring.polynomial([BigInt::from(value).into()]);
        let target = top_reference_target(reference, &ring_q, q) * scalar(&ring_q, kappa);
        let next = self.ring(level + 1)?;
        Ok(target.centered_rebase(&next) * scalar(&next, idempotent))
    }

    /// The CRT recomposition of level `level`: its primes and the
    /// coefficients `Q_i (Q_i^-1 mod q_i)`.
    pub(crate) fn refresh_crt(
        &self,
        level: usize,
    ) -> Result<(Vec<IntExpr>, Vec<IntExpr>), RnsBggError> {
        let parameters = self.level(level)?;
        let modulus: BigUint = parameters.modulus().as_ref().clone();
        let primes = parameters.to_crt().0;
        let coefficients = primes
            .iter()
            .map(|prime| {
                let prime = BigUint::from(*prime);
                let cofactor = &modulus / &prime;
                let inverse = (&cofactor % &prime)
                    .modinv(&prime)
                    .ok_or(RnsBggError::Parameters("refresh CRT primes are not coprime"))?;
                Ok(IntExpr::constant(BigInt::from(cofactor * inverse)))
            })
            .collect::<Result<Vec<_>, RnsBggError>>()?;
        Ok((primes.into_iter().map(IntExpr::from).collect(), coefficients))
    }
}

/// The LUT input level of the output of a refresh child of
/// [`crate::refresh::FreshNoiseProgram`] shape.
fn noise_program_level(program: &crate::refresh::FreshNoiseProgram) -> Result<usize, RnsBggError> {
    let workload = RnsBggDerivedWorkload::from_program(program.program())?;
    workload.level(program.output())
}

/// The reference ring of `reference`: its level-0 primes in canonical order.
pub(crate) fn top_reference_ring(reference: &TopReference) -> Ring {
    Ring::from_crt_moduli(
        reference.top.iter().copied().map(IntExpr::from).collect(),
        reference.ring_dimension,
    )
}

/// `T_q = Delta_0 G^st_0 mod q` of `reference` (`1 x w_0`, in `ring_q`):
/// `G^st_0` is the gadget of the level-1 primes, its centered entries
/// re-encoded at level 0 as the state gadget is.
pub(crate) fn top_reference_target(reference: &TopReference, ring_q: &Ring, q: u64) -> Mat {
    let next = Ring::from_crt_moduli(
        reference.next.iter().copied().map(IntExpr::from).collect(),
        reference.ring_dimension,
    );
    let gadget = next
        .gadget(1, 1u64 << reference.base_bits, reference.columns())
        .centered_rebase(&top_reference_ring(reference))
        .reduce_modulus(ring_q);
    let top: BigUint = reference.top.iter().map(|prime| BigUint::from(*prime)).product();
    let next_modulus: BigUint = reference.next.iter().map(|prime| BigUint::from(*prime)).product();
    let delta = (top / next_modulus) % q;
    gadget * ring_q.polynomial([BigInt::from(delta).into()])
}

/// `A'_(j)` of a top-refresh part in the reference ring (`2 x w_0`).
pub(crate) fn top_matrix(
    reference: &TopReference,
    hash_key: Bytes,
    declaration: &RefreshDeclaration,
) -> Mat {
    top_reference_ring(reference).hash_matrix(
        hash_key,
        declaration.matrix_hash_tag(),
        (2, reference.columns()),
    )
}

/// The ring of the product `P` of a top refresh's refreshed primes (`Q_0` when every slot is
/// refreshed), towers in canonical order.
pub(crate) fn top_output_ring(top: &TopRefreshDeclaration) -> Ring {
    output_ring(&top.output_primes(), top.reference().ring_dimension)
}

fn output_ring(primes: &[u64], ring_dimension: u32) -> Ring {
    Ring::from_crt_moduli(primes.iter().copied().map(IntExpr::from).collect(), ring_dimension)
}

/// A top-refresh part's term of `y_j mod P` ([`top_output_ring`]): its
/// residue `y_j mod q` (a `1 x w_0` row in the ring of its output prime
/// `q`), centered and re-encoded in the ring of `P`, times its CRT
/// coefficient `P / q ((P / q)^-1 mod q)`; the terms of every chain sum to
/// `y_j mod P`. With one refreshed prime the term is the residue itself.
/// Every part's term has the same type, so the parts' terms form one family.
pub(crate) fn top_residue_term(
    part: &crate::refresh::TopRefreshPart,
    residue: Mat,
) -> Result<Mat, RnsBggError> {
    let primes = part.output_moduli();
    if primes.len() == 1 {
        return Ok(residue);
    }
    let ring = output_ring(primes, part.reference().ring_dimension);
    let modulus: BigUint = primes.iter().map(|prime| BigUint::from(*prime)).product();
    let prime = BigUint::from(part.output_modulus());
    let cofactor = &modulus / &prime;
    let coefficient = (&cofactor % &prime)
        .modinv(&prime)
        .ok_or(RnsBggError::Parameters("refresh CRT primes are not coprime"))? *
        cofactor %
        &modulus;
    Ok(residue.centered_rebase(&ring) * ring.polynomial([BigInt::from(coefficient).into()]))
}

impl RnsBggParameters {
    /// Chain `chain` of `declaration` (see [`crate::refresh::top`]): level
    /// `l` has the first `depths[l]` primes of the chain's pool order, and
    /// the returned pool, which also hosts the chain's children. `None` for
    /// `chain` gives the reference chain (canonical order). Every chain's
    /// pool lists its towers in the canonical order (the primes of `Q_0`,
    /// then the extension primes), so rings of one modulus have one tower
    /// order (and one gadget layout) across chains; the pool order only
    /// selects each level's primes.
    pub fn top_refresh_chain(
        declaration: &TopRefreshDeclaration,
        chain: Option<usize>,
        encoding_error_sigma: impl Into<BigDecimal>,
        operations_per_drop: usize,
    ) -> Result<(DCRTPolyParams, Self), RnsBggError> {
        let invalid = || RnsBggError::Parameters("invalid top refresh chain");
        let reference = declaration.reference();
        let mut canonical = reference.top.clone();
        canonical.extend(declaration.pool_order(0).ok_or_else(invalid)?.split_off(canonical.len()));
        let order = match chain {
            Some(chain) => declaration.pool_order(chain).ok_or_else(invalid)?,
            None => canonical.clone(),
        };
        let bits = order
            .iter()
            .map(|prime| (u64::BITS - prime.leading_zeros()) as usize)
            .max()
            .ok_or_else(invalid)?;
        let pool = DCRTPolyParams::try_new(
            reference.ring_dimension,
            canonical.len(),
            bits,
            reference.base_bits,
            Some(canonical),
            None,
        )
        .map_err(|_| RnsBggError::Parameters("invalid top refresh prime pool"))?;
        let levels = declaration
            .depths()
            .iter()
            .map(|depth| {
                let modulus = order[..*depth].iter().map(|prime| BigUint::from(*prime)).product();
                pool.select_modulus(&modulus).ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((pool, Self::new(levels, encoding_error_sigma, operations_per_drop)?))
    }
}

/// `d_i = one X_i - target Y_i + [fresh_j z_i]_j` for one component: the two
/// one-encoding blocks `(one_0, one_1)` (`1 x m` each), the target's
/// component-1 block (`1 x w`) and the fresh child's component-1 row of
/// `columns` blocks `fresh_j` (`1 x m` each, the row `Sum` of the child, see
/// [`crate::refresh::FreshNoiseProgram`]). Applied to the key rows it gives
/// the decoded key `K_i`, applied to the rows it gives the decoded row `d_i`
/// (`1 x columns`).
pub(crate) fn refresh_combination(
    one: (Mat, Mat),
    target: Mat,
    (fresh, columns): (Mat, usize),
    actions: &RefreshActions,
) -> Result<Mat, RnsBggError> {
    let one = Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![one.0, one.1])
        .mul_small_rhs(actions.x.clone());
    Ok(one - target.mul_small_rhs(actions.y.clone()) +
        blockwise_mul_small_rhs(fresh, columns, &actions.z)?)
}

/// `[row_j rhs]_j` for a row of `blocks` equal blocks `row_j` and a small
/// `rhs` of one column: one product for one block, else a sequential loop
/// that places each block's product in its column (see
/// [`super::place_in_column`]), so the graph does not grow with `blocks`.
fn blockwise_mul_small_rhs(
    row: Mat,
    blocks: usize,
    rhs: &mxx_dsl::SmallMatrix,
) -> Result<Mat, RnsBggError> {
    if blocks == 1 {
        return Ok(row.mul_small_rhs(rhs.clone()));
    }
    let width = match row.matrix_type().columns.canonicalize() {
        IntExpr::Const(columns) => columns
            .to_usize()
            .filter(|columns| columns % blocks == 0)
            .map(|columns| columns / blocks)
            .ok_or(RnsBggError::Parameters("a block row has a malformed width"))?,
        _ => return Err(RnsBggError::Parameters("a block row needs a constant width")),
    };
    let block = |index: &IntExpr| {
        let start = IntExpr::Mul(Box::new(index.clone()), Box::new(IntExpr::constant(width)))
            .canonicalize();
        let end = IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(width)))
            .canonicalize();
        let product = row
            .clone()
            .slice(None, Some(mxx_ir_core::node::IndexRange { start, end }))
            .mul_small_rhs(rhs.clone());
        super::place_in_column(product, blocks, index)
    };
    mxx_dsl::iterate(blocks - 1, block(&IntExpr::constant(0)), |index, sum| {
        let index = index.add(1usize).expression()?;
        Ok(sum + block(&index))
    })
    .map_err(|error| RnsBggError::Graph(error.to_string()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        program::{AttributeKnowledge, ExponentLutProgramBuilder, Operand},
        program_gadgets::pbc::PbcGeneratedKeyLayout,
        test_support::{execute_cpu, full, validate},
    };
    use mxx_backends::{RuntimeValue, matrix::PolyMatrix, poly::Poly};
    use mxx_dsl::DslContext;

    /// The identity Program of one private input, and that input.
    pub(crate) fn identity_source() -> (ExponentLutProgram, WireId) {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        (builder.finish(vec![Operand::Scalar(input)]).unwrap(), input)
    }

    /// A refresh of `source`'s output `target` on `parameters` (level 0) with
    /// the decoder fixture's profile (`n = 8`, `P = 4`), CBD(1) fresh noise,
    /// `digits` smudge digits and random keys: the declaration, the fresh
    /// keys and every decoder's smudge keys.
    #[allow(clippy::type_complexity)]
    pub(crate) fn declaration(
        parameters: &RnsBggParameters,
        source: &ExponentLutProgram,
        target: WireId,
        digits: usize,
    ) -> (RefreshDeclaration, Vec<PbcGeneratedKeyLayout>, Vec<Vec<PbcGeneratedKeyLayout>>) {
        let level = parameters.level(0).unwrap();
        let (primes, columns) = (level.to_crt().0.len(), level.modulus_digits());
        let profile = crate::decoder::decoding::fixture::profile();
        let n = profile.ring_dimension().unwrap();
        let fresh = crate::decoder::decoding::fixture::keys(columns * n);
        let smudges = (0..primes)
            .map(|_| crate::decoder::decoding::fixture::keys(columns * n * digits))
            .collect::<Vec<_>>();
        let shape = crate::decoder::decoding::fixture::shape;
        let declaration = RefreshDeclaration::new(
            source,
            target,
            profile,
            1,
            primes,
            columns,
            digits,
            shape(&fresh),
            smudges.iter().map(|keys| shape(keys)).collect(),
            b"refresh-test".to_vec(),
        )
        .unwrap();
        (declaration, fresh, smudges)
    }

    /// Synthetic exact carriers (zero errors, no ModDown) under a mask
    /// `s = (s_0, 1)`: the one encoding, the target's component 1 and the
    /// fresh carriers. For every prime, the combination of the rows minus the
    /// mask times the same combination of the keys is exactly
    /// `-Q_i (s A' - Delta x G_Q + e')`: the decompositions `(I_2 ⊗ G_Q)^-1`,
    /// `G^st^-1(. mod q_(l+1))` and `G_Q^-1` reconstruct their weights.
    /// On every chain of a top refresh, the target action of the output prime
    /// `q` maps the level-`L` payload scale to the reference format:
    /// `Delta_L G^st_L G^-1(R_q) = Q_q (Delta_0 G^st_0 mod q)` modulo `Q_L`,
    /// with `T_q` the reference chain's own `Delta_0 G^st_0` modulo `q`;
    /// reductions between rings of different prime orders agree; and the
    /// residues of every chain recombine to the value modulo `Q_0`.
    #[test]
    fn top_refresh_target_weight_and_combination_are_exact() {
        use crate::refresh::top::tests::{declaration, lut_source, primes, reference};
        let (source, output) = lut_source(false);
        let top = declaration(&source, output, vec![3, 2, 2]).unwrap();
        let reference = reference();
        let (reference_pool, reference_chain) =
            RnsBggParameters::top_refresh_chain(&top, None, 0, 1).unwrap();
        let ring_of = |prime: u64| Ring::from_crt_moduli(vec![IntExpr::from(prime)], 8);
        let scalar = |ring: &Ring, value: BigUint| ring.polynomial([BigInt::from(value).into()]);
        let canonical = top_reference_ring(&reference);
        let hash = canonical.bytes_input("hash-key", 32);
        let value = canonical.hash_matrix(hash.clone(), b"value".to_vec(), (1, 4));
        let mut graph = DslContext::new("top-refresh-exact");
        let mut parameters = reference_chain.levels().to_vec();
        let mut residues = Vec::new();
        for chain in 0..top.chains() {
            let q = top.output_prime(chain).unwrap();
            let (pool, chain_parameters) =
                RnsBggParameters::top_refresh_chain(&top, Some(chain), 0, 1).unwrap();
            parameters.extend(chain_parameters.levels().iter().cloned());
            parameters.push(pool.select_modulus(&BigUint::from(q)).unwrap());
            let level = top.level();
            let ring = chain_parameters.ring(level).unwrap();
            let modulus: BigUint =
                chain_parameters.level(level).unwrap().modulus().as_ref().clone();
            let weight = chain_parameters.top_target_weight(level, q, &reference).unwrap();
            let y = chain_parameters.gadget_inverse(level + 1, level, weight).unwrap();
            let delta = chain_parameters
                .scalar(level, chain_parameters.message_scale(level).unwrap())
                .unwrap();
            let lhs = delta * chain_parameters.state_gadget(level).unwrap().mul_small_rhs(y);
            let target = top_reference_target(&reference, &ring_of(q), q);
            let rhs = scalar(&ring, &modulus / q) * target.clone().centered_rebase(&ring);
            let expected =
                (reference_chain.scalar(0, reference_chain.message_scale(0).unwrap()).unwrap() *
                    reference_chain.state_gadget(0).unwrap())
                .reduce_modulus(&ring_of(q));
            let direct = value.clone().reduce_modulus(&ring_of(q));
            let through_chain = value.clone().reduce_modulus(&ring).reduce_modulus(&ring_of(q));
            let part = top.part(chain, 0).unwrap().top().unwrap();
            residues.push(top_residue_term(part, direct.clone()).unwrap());
            graph = graph
                .output(format!("action-{chain}"), lhs - rhs)
                .unwrap()
                .output(format!("target-{chain}"), target - expected)
                .unwrap()
                .output(format!("order-{chain}"), direct - through_chain)
                .unwrap();
        }
        let combined = residues.into_iter().reduce(|sum, term| sum + term).unwrap();
        let graph = graph.output("combine", combined - value).unwrap().build().unwrap();
        parameters.push(reference_pool);
        assert_eq!(primes().len(), 16);
        let result = execute_cpu(
            &validate(&graph),
            parameters,
            std::collections::BTreeMap::from([(
                "hash-key".to_owned(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().as_slice().into()),
            )]),
        );
        for (name, output) in &result.outputs {
            let matrix = full(output);
            let (rows, columns) = matrix.size();
            for row in 0..rows {
                for column in 0..columns {
                    assert!(
                        matrix.entry(row, column).coeffs_biguints().iter().all(Zero::is_zero),
                        "{name} is not zero at ({row}, {column})"
                    );
                }
            }
        }
    }

    /// A top-refresh part is checked against its own chain: another chain
    /// drops its output prime before the plaintext level, so the part is
    /// rejected. Distinct chains derive distinct stage keys.
    #[test]
    fn top_refresh_parts_bind_to_their_chain() {
        use crate::refresh::top::tests::{declaration, lut_source};
        let (source, output) = lut_source(false);
        let top = declaration(&source, output, vec![3, 2, 1]).unwrap();
        let chain =
            |index: usize| RnsBggParameters::top_refresh_chain(&top, Some(index), 0, 1).unwrap().1;
        let (first, second) = (chain(0), chain(1));
        let part = top.part(0, 1).unwrap();
        assert_eq!(first.refresh_level(part, &source).unwrap(), 1);
        assert!(second.refresh_level(part, &source).is_err(), "a swapped chain");
        assert_eq!(second.refresh_level(top.part(1, 0).unwrap(), &source).unwrap(), 1);
        assert_eq!(first.level(2).unwrap().to_crt().0, vec![top.output_prime(0).unwrap()]);
        assert_eq!(second.level(2).unwrap().to_crt().0, vec![top.output_prime(1).unwrap()]);
        // Both chains list the decoding primes in the canonical order.
        assert_eq!(first.level(1).unwrap().to_crt().0, second.level(1).unwrap().to_crt().0);
        let key = [5; 32];
        let keys = (0..top.chains())
            .map(|chain| child_hash_key(&key, &top.chain_namespace(chain)))
            .chain(std::iter::once(child_hash_key(&key, &top.matrix_namespace())))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), top.chains() + 1);
    }

    #[test]
    fn refresh_combination_of_exact_carriers_is_the_scaled_target_row() {
        let (_, parameters) = crate::rns_bgg::tests::decoder_test_chain();
        let (source, input) = identity_source();
        let (declaration, _, _) = declaration(&parameters, &source, input, 1);
        let ring = parameters.ring(0).unwrap();
        let hash = ring.bytes_input("hash-key", 32);
        let sample = |tag: &str, columns: usize| {
            ring.hash_matrix(hash.clone(), tag.as_bytes().to_vec(), (1, columns))
        };
        let columns = parameters.level(0).unwrap().modulus_digits();
        let width = parameters.component_columns(0).unwrap();
        let s0 = sample("s0", 1);
        let a_prime = parameters.refresh_matrix(0, hash.clone(), &declaration).unwrap();
        let one = parameters.refresh_one_key(0, hash.clone(), &declaration).unwrap();
        let gadget = parameters.public_gadget(0).unwrap();
        let state = parameters.state_gadget(0).unwrap();
        let delta = parameters.scalar(0, parameters.message_scale(0).unwrap()).unwrap();
        let x = ring.polynomial([0, 0, 0, 1].map(IntExpr::constant));
        let target = (sample("kx0", width), sample("kx1", width));
        let fresh_keys = (0..columns)
            .map(|j| (sample(&format!("kf0-{j}"), columns), sample(&format!("kf1-{j}"), columns)))
            .collect::<Vec<_>>();
        let fresh_noise = (0..columns)
            .map(|j| ring.polynomial([j as i64 - 1, 1, 0, -1].map(IntExpr::constant)))
            .collect::<Vec<_>>();
        let one_row = (
            s0.clone() * one.0.0.clone() + one.1.0.clone() - s0.clone() * gadget.clone(),
            s0.clone() * one.0.1.clone() + one.1.1.clone() - gadget.clone(),
        );
        let target_row =
            s0.clone() * target.0.clone() + target.1.clone() - delta.clone() * x.clone() * state;
        let fresh_rows = fresh_keys
            .iter()
            .zip(&fresh_noise)
            .map(|(key, noise)| {
                s0.clone() * key.0.clone() + key.1.clone() - noise.clone() * gadget.clone()
            })
            .collect::<Vec<_>>();
        let row = |index: usize| {
            a_prime.clone().slice(
                Some(mxx_ir_core::node::IndexRange {
                    start: IntExpr::constant(index),
                    end: IntExpr::constant(index + 1),
                }),
                None,
            )
        };
        let refreshed = s0.clone() * row(0) + row(1) - delta * x * gadget +
            Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, fresh_noise);
        let modulus: BigUint = parameters.level(0).unwrap().modulus().as_ref().clone();
        let mut graph = DslContext::new("refresh-exact-carriers");
        for (prime, q_i) in parameters.level(0).unwrap().to_crt().0.into_iter().enumerate() {
            let actions =
                parameters.refresh_actions(0, prime, a_prime.clone(), &declaration).unwrap();
            // The fresh carriers as one row of column blocks, as the fresh
            // child's row `Sum` returns them.
            let blocks = |rows: Vec<Mat>| {
                (Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, rows), columns)
            };
            let decoded = refresh_combination(
                one_row.clone(),
                target_row.clone(),
                blocks(fresh_rows.clone()),
                &actions,
            )
            .unwrap();
            let top = refresh_combination(
                (one.0.0.clone(), one.0.1.clone()),
                target.0.clone(),
                blocks(fresh_keys.iter().map(|key| key.0.clone()).collect()),
                &actions,
            )
            .unwrap();
            let bottom = refresh_combination(
                (one.1.0.clone(), one.1.1.clone()),
                target.1.clone(),
                blocks(fresh_keys.iter().map(|key| key.1.clone()).collect()),
                &actions,
            )
            .unwrap();
            let cofactor = parameters.scalar(0, BigInt::from(&modulus / q_i)).unwrap();
            let residual = decoded - (s0.clone() * top + bottom) + cofactor * refreshed.clone();
            graph = graph
                .output(format!("residual-{prime}"), residual)
                .unwrap()
                .output(format!("x-{prime}"), ring.identity(2 * columns).mul_small_rhs(actions.x))
                .unwrap()
                .output(format!("y-{prime}"), ring.identity(width).mul_small_rhs(actions.y))
                .unwrap()
                .output(format!("z-{prime}"), ring.identity(columns).mul_small_rhs(actions.z))
                .unwrap();
        }
        let graph = graph.build().unwrap();
        let result = execute_cpu(
            &validate(&graph),
            parameters.levels().iter().cloned(),
            std::collections::BTreeMap::from([(
                "hash-key".to_owned(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().as_slice().into()),
            )]),
        );
        let centered = |value: &BigUint| {
            let value = BigInt::from(value.clone());
            let modulus = BigInt::from(modulus.clone());
            if &value * 2u8 > modulus { value - modulus } else { value }
        };
        // Column `l1` norms of a constant digit matrix, and its largest digit.
        let digits = |name: &str| {
            let matrix = full(&result.outputs[name]);
            let entries = (0..matrix.row_size())
                .map(|row| {
                    (0..matrix.col_size())
                        .map(|column| {
                            matrix
                                .entry(row, column)
                                .coeffs_biguints()
                                .iter()
                                .map(|value| centered(value).magnitude().clone())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let largest = entries.iter().flatten().flatten().max().cloned().unwrap_or_default();
            let l1 = (0..matrix.col_size())
                .map(|column| entries.iter().map(|row| row[column].iter().sum::<BigUint>()).sum())
                .collect::<Vec<BigUint>>();
            (largest, l1)
        };
        let half_base =
            |level: usize| BigUint::one() << (parameters.level(level).unwrap().base_bits() - 1);
        for (prime, q_i) in parameters.level(0).unwrap().to_crt().0.into_iter().enumerate() {
            // The bounds of the refresh noise report hold for the actual digits.
            let (largest, _) = digits(&format!("x-{prime}"));
            assert!(largest <= half_base(0), "X digits of prime {prime}");
            let (largest, l1) = digits(&format!("y-{prime}"));
            assert!(largest <= half_base(1), "Y digits of prime {prime}");
            let width = BigUint::from(parameters.level(1).unwrap().modulus_digits());
            assert!(
                l1.iter().all(|norm| norm <= &(&width * half_base(1))),
                "Y l1 of prime {prime}"
            );
            let (_, l1) = digits(&format!("z-{prime}"));
            let (expected, _) = parameters.gadget_inverse_norms(0, &(&modulus / q_i)).unwrap();
            assert_eq!(l1, vec![expected], "z l1 of prime {prime}");
        }
        for prime in 0..declaration.primes() {
            let residual = full(&result.outputs[&format!("residual-{prime}")]);
            for column in 0..columns {
                assert!(
                    residual
                        .entry(0, column)
                        .coeffs_biguints()
                        .iter()
                        .all(|value| value == &BigUint::ZERO),
                    "prime {prime}, column {column}"
                );
            }
        }
    }

    /// The refresh is defined only on a Static, non-terminal target level
    /// whose modulus has the declared prime and column counts.
    #[test]
    fn refresh_level_requires_a_static_non_terminal_target_of_the_declared_shape() {
        let (pool, parameters) = crate::rns_bgg::tests::decoder_test_chain();
        let (source, input) = identity_source();
        let (declaration, _, _) = declaration(&parameters, &source, input, 1);
        assert_eq!(parameters.refresh_level(&declaration, &source).unwrap(), 0);
        let terminal = RnsBggParameters::new(vec![parameters.levels()[0].clone()], 1, 1).unwrap();
        assert!(terminal.refresh_level(&declaration, &source).is_err());
        let dynamic = parameters.clone().with_rhs_mode(RhsMode::Dynamic);
        assert!(dynamic.refresh_level(&declaration, &source).is_err());
        let prefix = |depth: usize| {
            let modulus =
                pool.to_crt().0[..depth].iter().map(|prime| BigUint::from(*prime)).product();
            pool.select_modulus(&modulus).unwrap()
        };
        let two_primes = RnsBggParameters::new(vec![prefix(2), prefix(1)], 1, 1).unwrap();
        assert!(two_primes.refresh_level(&declaration, &source).is_err());
        // The children chains end at the target's level with a strict drop.
        let fresh = parameters.refresh_fresh_parameters(&declaration, &source, &pool, 1).unwrap();
        let smudge = parameters
            .decoder_smudge_parameters(&declaration.decoders()[0], &source, &pool, 1)
            .unwrap();
        let smudges = vec![&smudge; declaration.decoders().len()];
        assert_eq!(
            parameters
                .validate_refresh_children(&declaration, &source, None, &fresh, &smudges)
                .unwrap(),
            0
        );
        assert!(
            parameters
                .validate_refresh_children(&declaration, &source, None, &smudge, &smudges)
                .is_err(),
            "the smudge chain is not the fresh chain"
        );
        assert!(
            parameters
                .validate_refresh_children(&declaration, &source, None, &fresh, &smudges[1..])
                .is_err()
        );
    }
}
