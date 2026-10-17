//! The RNS-BGG arithmetic of the noise refresh (see [`crate::refresh`]).
//!
//! The stages of a part share the public pieces defined here: the caller's `A'_(j)` (the graph
//! input [`REFRESH_MATRIX_INPUT`]), the hash-sampled one-encoding key, the digit actions `X, Y, z`
//! of the output prime, the combination `d` (the same on keys and rows), the CRT term of the
//! part's residue, and the refresh's chains, child hash keys and artifact names.
//!
//! [`run`] runs a declared refresh on its chains and records its stage graphs, and [`claim`]
//! states the claim of a recorded run.
use super::*;
use crate::{
    program::ExponentLutProgram,
    refresh::{RefreshDeclaration, RefreshPartDeclaration, TopReference},
};

mod claim;
mod run;
pub use run::{
    RefreshChain, RefreshRun, RefreshSecurity, RefreshSigma, RefreshSigmaRequest, RnsBggRefresh,
    SIGMA_ROW_OUTPUT, refresh_noise_accepted,
};

/// The graph input of the caller's `A' = [A'_(0) | A'_(1)]` (`2 x 2 w_0` over the reference's
/// level-0 ring, see [Caller inputs](crate::refresh#caller-inputs)): every part's preprocessing
/// and online evaluation and the decode stage read it under this name.
pub const REFRESH_MATRIX_INPUT: &str = "top-refresh-a-prime";

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

/// `refresh/<spec>/sigma-encoding`: the online evaluation's graph input of the caller's row `g`
/// of the encoding of `sigma_0 G` (see [Caller inputs](crate::refresh#caller-inputs)), one name
/// per part.
pub(crate) fn refresh_sigma_input_name(spec: [u8; 32]) -> String {
    refresh_artifact_name(spec, "sigma-encoding")
}

/// Part `part` of `declaration`.
pub(crate) fn refresh_part(
    declaration: &RefreshDeclaration,
    part: (usize, usize),
) -> Result<&RefreshPartDeclaration, RnsBggError> {
    declaration.part(part).ok_or(RnsBggError::Parameters("the refresh has no such part"))
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

/// The digit actions of the output prime `q` (`Q_q = Q / q`): `x` is
/// `(I_2 ⊗ G_Q)^-1(Q_q A'_(j))` (`2m x w_0`), `y` is `G^st^-1(R_q)` (`w x w_0`, the target
/// weight of [`RnsBggParameters::top_target_weight`]) and `z` is `G_Q^-1(Q_q)` (`m x 1`).
pub struct RefreshActions {
    pub(crate) x: SmallMatrix,
    pub(crate) y: SmallMatrix,
    pub(crate) z: SmallMatrix,
}

impl RnsBggParameters {
    /// The decoding level `l` of the part `declaration`'s target in `source_program`, after
    /// checking the part against these parameters (its chain): Static RHS mode, a non-terminal
    /// level whose modulus has the declared prime count with the output prime at its position,
    /// which survives into the plaintext modulus of the level (the target action inverts the
    /// level's scale modulo it), the ring dimension, and a fresh noise bound below every prime.
    pub(crate) fn refresh_level(
        &self,
        declaration: &RefreshPartDeclaration,
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
        let reference = declaration.reference();
        if primes.get(declaration.output_prime()) != Some(&declaration.output_modulus()) ||
            !self.level(level + 1)?.to_crt().0.contains(&declaration.output_modulus()) ||
            reference.ring_dimension != parameters.ring_dimension() ||
            !reference.top.contains(&declaration.output_modulus())
        {
            return Err(invalid);
        }
        Ok(level)
    }

    /// The chain on which the raw coefficient Program `child` (one output, a
    /// [`crate::refresh::FreshNoiseProgram`]) of chain `chain` of `declaration` (these parameters)
    /// is evaluated (a child chain of these parameters): its output's LUT input level is followed
    /// by the part's decoding level, so its normalized carrier is a carrier of the decoding
    /// ring. The fresh-noise children run on it, and so may the caller's encoding of `sigma_0
    /// G` (see [Caller inputs](crate::refresh#caller-inputs)), with the output use
    /// [`super::RnsOutputUse::Gadget`].
    #[allow(clippy::too_many_arguments)]
    pub fn refresh_child_parameters(
        &self,
        declaration: &RefreshDeclaration,
        chain: usize,
        source_program: &ExponentLutProgram,
        child: &crate::refresh::FreshNoiseProgram,
        pool: &DCRTPolyParams,
        primes_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        let part = refresh_part(declaration, (chain, 0))?;
        let level = self.refresh_level(part, source_program)?;
        self.child_chain(level, noise_program_level(child)?, pool, primes_per_drop)
    }

    /// The fresh-noise child's chain of chain `chain` of `declaration` (see
    /// [`Self::refresh_child_parameters`]). Both components' fresh children share it.
    pub fn refresh_fresh_parameters(
        &self,
        declaration: &RefreshDeclaration,
        chain: usize,
        source_program: &ExponentLutProgram,
        pool: &DCRTPolyParams,
        primes_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        let fresh = declaration
            .fresh((chain, 0))
            .ok_or(RnsBggError::Parameters("the refresh has no such part"))?;
        self.refresh_child_parameters(
            declaration,
            chain,
            source_program,
            fresh,
            pool,
            primes_per_drop,
        )
    }

    /// The common LUT input level of the fresh-noise child's outputs.
    pub(crate) fn refresh_fresh_level(
        declaration: &RefreshPartDeclaration,
    ) -> Result<usize, RnsBggError> {
        noise_program_level(declaration.fresh())
    }

    /// Check the children's chains `fresh` and `smudge` against the part `declaration`.
    pub(crate) fn validate_refresh_children(
        &self,
        declaration: &RefreshPartDeclaration,
        source_program: &ExponentLutProgram,
        fresh: &Self,
        smudge: &Self,
    ) -> Result<usize, RnsBggError> {
        let level = self.refresh_level(declaration, source_program)?;
        self.validate_child_chain(fresh, level, Self::refresh_fresh_level(declaration)?)?;
        self.validate_decoder_smudge_parameters(smudge, declaration.decoder(), source_program)?;
        Ok(level)
    }

    /// The part `declaration`'s `A'_(j)` at level `level`: columns `j w_0 .. (j + 1) w_0` of the
    /// caller's input [`REFRESH_MATRIX_INPUT`] ([`top_matrix_input`]), reduced modulo the level's
    /// primes (a CRT sub-basis of the reference's `Q_0`), the residues of one `A'` on every chain.
    pub(crate) fn refresh_matrix(
        &self,
        level: usize,
        declaration: &RefreshPartDeclaration,
    ) -> Result<Mat, RnsBggError> {
        let ring = self.ring(level)?;
        Ok(component_matrix(top_matrix_input(declaration.reference()), declaration)
            .reduce_modulus(&ring))
    }

    /// The public key `A_1` of the one encoding at level `level`: a public
    /// input key schema sampled from the stage hash key under the refresh
    /// namespace.
    pub(crate) fn refresh_one_key(
        &self,
        level: usize,
        hash_key: Bytes,
        declaration: &RefreshPartDeclaration,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let ring = self.ring(level)?;
        let columns =
            self.key_component_columns(level, crate::program::AttributeKnowledge::PublicPayload)?;
        // The specification is an integer component, so parts differ only in integers.
        let mut tag = mxx_dsl::HashTag::from(b"refresh/v1/one".to_vec());
        super::push_digest(&mut tag, declaration.spec());
        Ok(super::public_key::matrix_from_hash(&ring, hash_key, tag, columns))
    }

    /// The digit actions of decoding the prime at index `output_prime` of the basis of level
    /// `level` of a row refreshed to the format `reference` under the public `a_prime = A'_(j)`
    /// (`2 x w_0` in the ring of `level`): a part's actions (see [`crate::refresh`]), or a row
    /// target's (see [`crate::rns_bgg::RnsBggParameters::retarget_encoding`]). The index may be
    /// a constant or a symbolic loop index. It must name a prime surviving at `level + 1`.
    /// Constant tables hold the prime-dependent scalars; the action graph has fixed size.
    pub fn refresh_actions(
        &self,
        level: usize,
        a_prime: Mat,
        output_prime: impl Into<mxx_dsl::Int>,
        reference: &TopReference,
    ) -> Result<RefreshActions, RnsBggError> {
        let parameters = self.level(level)?;
        let output_prime = output_prime.into().expression()?;
        let primes = parameters.to_crt().0;
        validate_prime_index(&output_prime, primes.len())?;
        let modulus: BigUint = parameters.modulus().as_ref().clone();
        let cofactor = prime_selection(
            &output_prime,
            primes.iter().map(|prime| IntExpr::from(&modulus / prime)),
        );
        let cofactor_scalar = self.ring(level)?.polynomial([cofactor]);
        let x = self.gadget_inverse(level, level, cofactor_scalar.clone() * a_prime)?;
        let weight = self.top_target_weight(level, &output_prime, reference)?;
        let y = self.gadget_inverse(level + 1, level, weight)?;
        let z = self.gadget_inverse(level, level, cofactor_scalar)?;
        Ok(RefreshActions { x, y, z })
    }

    /// The target weight `R_q` of a part decoding the prime `q`
    /// of level `level` (`1 x w_0`, ring of level `level + 1`):
    /// `R_q = (Delta_l^-1 Q_q T_q mod q)` modulo `q` and `0` modulo the other
    /// primes of level `l + 1`, with `T_q = Delta_0 G^st_0 mod q` of the
    /// reference. `output_prime` indexes the level's basis, including inside a graph loop.
    /// Then `Delta_l G^st_l G^-1(R_q) = Q_q T_q (mod Q)`: modulo
    /// `q` because `q` is a plaintext prime (`Delta_l` is invertible), and
    /// modulo every other prime both sides vanish (see [`crate::refresh`]).
    /// When every prime of the level survives at `level + 1`, the weight is formed in the ring
    /// of `level + 1` alone, as the reference gadget times the scale and the CRT idempotent of
    /// `q` (`1` modulo `q`, `0` modulo the other primes), so its graph has the same types for
    /// every prime: a loop over the primes has instance-independent types. Otherwise it is
    /// formed modulo the selected prime, and a non-surviving symbolic index selects an
    /// out-of-bounds plaintext-ring slice rather than silently producing a valid zero action.
    /// Constant invalid indices fail here.
    pub(crate) fn top_target_weight(
        &self,
        level: usize,
        output_prime: &IntExpr,
        reference: &TopReference,
    ) -> Result<Mat, RnsBggError> {
        let invalid = || RnsBggError::Parameters("the refresh does not match its chain");
        let parameters = self.level(level)?;
        let primes = parameters.to_crt().0;
        validate_prime_index(output_prime, primes.len())?;
        let modulus: BigUint = parameters.modulus().as_ref().clone();
        let next_parameters = self.level(level + 1)?;
        let next_primes = next_parameters.to_crt().0;
        let next_modulus: BigUint = next_parameters.modulus().as_ref().clone();
        if let IntExpr::Const(index) = output_prime.clone().canonicalize() {
            let prime = primes[index.to_usize().ok_or_else(invalid)?];
            if !next_primes.contains(&prime) {
                return Err(invalid());
            }
        }
        let delta = &modulus / &next_modulus;
        let reference_top: BigUint = reference.top.iter().copied().map(BigUint::from).product();
        let reference_next: BigUint = reference.next.iter().copied().map(BigUint::from).product();
        let reference_delta = reference_top / reference_next;
        let survive = primes.iter().all(|prime| next_primes.contains(prime));
        let mut indices = Vec::with_capacity(primes.len());
        let mut scales = Vec::with_capacity(primes.len());
        let mut idempotents = Vec::with_capacity(primes.len());
        for prime_value in primes {
            let Some(index) = next_primes.iter().position(|prime| *prime == prime_value) else {
                indices.push(IntExpr::constant(next_primes.len()));
                scales.push(IntExpr::constant(0));
                idempotents.push(IntExpr::constant(0));
                continue;
            };
            indices.push(IntExpr::constant(index));
            let prime = BigUint::from(prime_value);
            let cofactor = &modulus / &prime;
            let kappa = (&delta % &prime).modinv(&prime).ok_or_else(invalid)? * (cofactor % &prime) %
                &prime;
            let next_cofactor = &next_modulus / &prime;
            let idempotent = (&next_cofactor % &prime)
                .modinv(&prime)
                .ok_or(RnsBggError::Parameters("refresh CRT primes are not coprime"))? *
                &next_cofactor %
                &next_modulus;
            scales.push(IntExpr::from(&reference_delta * kappa % &prime));
            idempotents.push(IntExpr::from(idempotent));
        }
        let next = self.ring(level + 1)?;
        let scale = prime_selection(output_prime, scales);
        let idempotent = next.polynomial([prime_selection(output_prime, idempotents)]);
        if survive {
            return Ok(top_reference_gadget(reference).reduce_modulus(&next) *
                next.polynomial([scale]) *
                idempotent);
        }
        // Some prime does not survive: the weight is formed modulo the selected prime, so that
        // selecting a dropped prime reads an out-of-bounds slice instead of a zero action.
        let index = prime_selection(output_prime, indices);
        let ring_q = next.slice_crt(index.clone(), index + 1);
        let target =
            top_reference_gadget(reference).reduce_modulus(&ring_q) * ring_q.polynomial([scale]);
        Ok(target.centered_rebase(&next) * idempotent)
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
/// re-encoded at level 0 as the state gadget is. This independent reference
/// expression checks the combined scalar selection used by the production weight action.
#[cfg(test)]
pub(crate) fn top_reference_target(reference: &TopReference, ring_q: &Ring, q: u64) -> Mat {
    let gadget = top_reference_gadget(reference).reduce_modulus(ring_q);
    let top: BigUint = reference.top.iter().map(|prime| BigUint::from(*prime)).product();
    let next_modulus: BigUint = reference.next.iter().map(|prime| BigUint::from(*prime)).product();
    let delta = (top / next_modulus) % q;
    gadget * ring_q.polynomial([BigInt::from(delta).into()])
}

/// The reference state gadget, centered in its plaintext ring and lifted to its top ring.
fn top_reference_gadget(reference: &TopReference) -> Mat {
    let next = Ring::from_crt_moduli(
        reference.next.iter().copied().map(IntExpr::from).collect(),
        reference.ring_dimension,
    );
    next.gadget(1, 1u64 << reference.base_bits, reference.columns())
        .centered_rebase(&top_reference_ring(reference))
}

/// The caller's `A' = [A'_(0) | A'_(1)]` as the graph input [`REFRESH_MATRIX_INPUT`] of the
/// reference ring (`2 x 2 w_0`).
pub(crate) fn top_matrix_input(reference: &TopReference) -> Mat {
    top_reference_ring(reference).input(REFRESH_MATRIX_INPUT, (2, 2 * reference.columns()))
}

/// `A'_(j)` of the part `declaration`: columns `j w_0 .. (j + 1) w_0` of `a_prime` (the component
/// is an integer of the slice, so the parts' graphs differ only in integers).
pub(crate) fn component_matrix(a_prime: Mat, declaration: &RefreshPartDeclaration) -> Mat {
    let columns = declaration.reference().columns();
    let start = declaration.component() * columns;
    a_prime.slice(
        None,
        Some(mxx_ir_core::node::IndexRange {
            start: IntExpr::constant(start),
            end: IntExpr::constant(start + columns),
        }),
    )
}

/// The ring of the product `P` of a refresh's refreshed primes (`Q_0` when every slot is
/// refreshed), towers in canonical order.
pub(crate) fn top_output_ring(top: &RefreshDeclaration) -> Ring {
    output_ring(&top.output_primes(), top.reference().ring_dimension)
}

fn output_ring(primes: &[u64], ring_dimension: u32) -> Ring {
    Ring::from_crt_moduli(primes.iter().copied().map(IntExpr::from).collect(), ring_dimension)
}

/// A part's term of `y_j mod P` ([`top_output_ring`]): its
/// residue `y_j mod q` (a `1 x w_0` row in the ring of its output prime
/// `q`), centered and re-encoded in the ring of `P`, times its CRT
/// coefficient `P / q ((P / q)^-1 mod q)`; the terms of every chain sum to
/// `y_j mod P`. With one refreshed prime the term is the residue itself.
/// Every part's term has the same type, so the parts' terms form one family.
pub(crate) fn top_residue_term(
    part: &RefreshPartDeclaration,
    residue: Mat,
) -> Result<Mat, RnsBggError> {
    residue_term(
        part.output_moduli(),
        part.output_moduli()
            .iter()
            .position(|prime| *prime == part.output_modulus())
            .ok_or(RnsBggError::Parameters("refresh prime index out of range"))?,
        part.reference().ring_dimension,
        residue,
    )
}

/// The term of a residue `residue` (`y mod q`, a row in the ring of the prime `q`) of the
/// recombination of `y` modulo the product `P` of `primes` (`q` among them, ring dimension
/// `ring_dimension`): the centered residue in the ring of `P` times its CRT coefficient
/// `P / q ((P / q)^-1 mod q)`. `output_prime` is its zero-based index in `primes`, either
/// constant or symbolic; the residue's ring must be `output_ring.slice_crt(index, index + 1)`.
/// The terms of every prime of `P` sum to `y mod P`. Only scalar tables grow with the basis.
pub fn residue_term(
    primes: &[u64],
    output_prime: impl Into<mxx_dsl::Int>,
    ring_dimension: u32,
    residue: Mat,
) -> Result<Mat, RnsBggError> {
    let output_prime = output_prime.into().expression()?;
    validate_prime_index(&output_prime, primes.len())?;
    let ring = output_ring(primes, ring_dimension);
    let modulus: BigUint = primes.iter().map(|prime| BigUint::from(*prime)).product();
    let coefficients = primes
        .iter()
        .map(|prime| {
            let prime = BigUint::from(*prime);
            let cofactor = &modulus / &prime;
            let inverse = (&cofactor % &prime)
                .modinv(&prime)
                .ok_or(RnsBggError::Parameters("refresh CRT primes are not coprime"))?;
            Ok(IntExpr::from(inverse * cofactor % &modulus))
        })
        .collect::<Result<Vec<_>, RnsBggError>>()?;
    let prime_ring = ring.slice_crt(output_prime.clone(), output_prime.clone() + 1);
    let residue = residue.reduce_modulus(&prime_ring).centered_rebase(&ring);
    Ok(residue * ring.polynomial([prime_selection(&output_prime, coefficients)]))
}

/// The term of the recombination modulo the product `P` of `primes` (ring dimension
/// `ring_dimension`) of the residue modulo the prime `q` (of index `output_prime`, constant or
/// symbolic) of `round(decoded / (P / q))`, for a decoded row `decoded` in the ring of `P`: the
/// centered rounded division by the cofactor `P / q`, times the CRT coefficient
/// `P / q ((P / q)^-1 mod q)`. It equals [`residue_term`] of `decoded` switched to the ring of
/// `q`, without that ring: every prime's term has the same type, so a loop over the primes has
/// instance-independent types. The terms of every prime sum to the recombined row.
pub fn decoded_residue_term(
    primes: &[u64],
    output_prime: impl Into<mxx_dsl::Int>,
    ring_dimension: u32,
    decoded: Mat,
) -> Result<Mat, RnsBggError> {
    let output_prime = output_prime.into().expression()?;
    validate_prime_index(&output_prime, primes.len())?;
    let ring = output_ring(primes, ring_dimension);
    let modulus: BigUint = primes.iter().map(|prime| BigUint::from(*prime)).product();
    let (cofactors, coefficients) = primes
        .iter()
        .map(|prime| {
            let prime = BigUint::from(*prime);
            let cofactor = &modulus / &prime;
            let inverse = (&cofactor % &prime)
                .modinv(&prime)
                .ok_or(RnsBggError::Parameters("refresh CRT primes are not coprime"))?;
            Ok((IntExpr::from(cofactor.clone()), IntExpr::from(inverse * cofactor % &modulus)))
        })
        .collect::<Result<(Vec<_>, Vec<_>), RnsBggError>>()?;
    Ok(decoded.centered_round_divide(prime_selection(&output_prime, cofactors)) *
        ring.polynomial([prime_selection(&output_prime, coefficients)]))
}

/// Selects one integer table entry without unrolling any matrix graph operations.
fn prime_selection(index: &IntExpr, branches: impl IntoIterator<Item = IntExpr>) -> IntExpr {
    IntExpr::Select { selector: Box::new(index.clone()), branches: branches.into_iter().collect() }
        .canonicalize()
}

/// Rejects empty bases and invalid constant indices; symbolic bounds are checked on execution.
fn validate_prime_index(index: &IntExpr, count: usize) -> Result<(), RnsBggError> {
    if count == 0 ||
        matches!(index.clone().canonicalize(), IntExpr::Const(value) if value.to_usize().is_none_or(|index| index >= count))
    {
        return Err(RnsBggError::Parameters("refresh prime index out of range"));
    }
    Ok(())
}

impl RnsBggParameters {
    /// Chain `chain` of `declaration` (see [`crate::refresh`]): level
    /// `l` has the first `depths[l]` primes of the chain's pool order, and
    /// the returned pool, which also hosts the chain's children. `None` for
    /// `chain` gives the reference chain (canonical order). Every chain's
    /// pool lists its towers in the canonical order (the primes of `Q_0`,
    /// then the extension primes), so rings of one modulus have one tower
    /// order (and one gadget layout) across chains; the pool order only
    /// selects each level's primes.
    pub fn refresh_chain(
        declaration: &RefreshDeclaration,
        chain: Option<usize>,
        encoding_error_sigma: impl Into<BigDecimal>,
        operations_per_drop: usize,
    ) -> Result<(DCRTPolyParams, Self), RnsBggError> {
        let invalid = || RnsBggError::Parameters("invalid refresh chain");
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
        .map_err(|_| RnsBggError::Parameters("invalid refresh prime pool"))?;
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

/// `d = one X - target Y + fresh` for one row: the two one-encoding blocks `(one_0, one_1)`
/// (`1 x m` each), the target's component block (`1 x w`) and the fresh child's term `fresh`
/// (`[fresh_k z]_k`, [`refresh_fresh_term`]). Applied to the key rows it gives the decoded key
/// `K`, applied to the rows it gives the decoded row `d` (`1 x columns`).
pub fn refresh_combination(
    one: (Mat, Mat),
    target: Mat,
    fresh: Mat,
    actions: &RefreshActions,
) -> Result<Mat, RnsBggError> {
    let one = Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![one.0, one.1])
        .mul_small_rhs(actions.x.clone());
    Ok(one - target.mul_small_rhs(actions.y.clone()) + fresh)
}

/// The fresh terms `[fresh_k z]_k` of the entries of a fresh child's component-1 row of
/// `blocks` column blocks: `block(k)` gives the entries' block `k` (`1 x m` each, an index
/// expression `k`), each times the one-column action `z` of `actions` and placed in column `k`
/// ([`super::place_in_column`]). Independent blocks are a parallel family consumed by its
/// canonical sum. The backend chooses bounded wave width and streams the reduction without
/// materializing every block. Graph size does not grow with `blocks`.
pub fn refresh_fresh_term(
    blocks: usize,
    actions: &RefreshActions,
    block: impl Fn(&IntExpr) -> Result<Vec<Mat>, RnsBggError>,
) -> Result<Vec<Mat>, RnsBggError> {
    let z = &actions.z;
    let placed = |index: &IntExpr| -> Result<Vec<Mat>, RnsBggError> {
        Ok(block(index)?
            .into_iter()
            .map(|entry| {
                let product = entry.mul_small_rhs(z.clone());
                if blocks == 1 { product } else { super::place_in_column(product, blocks, index) }
            })
            .collect())
    };
    if blocks == 0 {
        return Err(RnsBggError::Binding("fresh term requires at least one block"));
    }
    if blocks == 1 {
        return placed(&IntExpr::constant(0));
    }
    let mut failure = None;
    let terms = mxx_dsl::parallel(blocks, |index| {
        placed(&index.expression()?).map_err(|error| {
            failure = Some(error);
            mxx_dsl::DslError::Schema
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let terms = terms.map_err(|error| RnsBggError::Graph(error.to_string()))?;
    let add = |left: Vec<Mat>, right: Vec<Mat>| {
        left.into_iter().zip(right).map(|(left, right)| left + right).collect::<Vec<_>>()
    };
    let initial = add(terms.at(0usize), terms.at(1usize));
    mxx_dsl::iterate(blocks - 2, initial, |index, sum| Ok(add(sum, terms.at(index.add(2usize)))))
        .map_err(|error| RnsBggError::Graph(error.to_string()))
}

/// The fresh term of a fresh child's component-1 row `row` of `blocks` equal column blocks:
/// [`refresh_fresh_term`] of the row's blocks.
pub(crate) fn row_fresh_term(
    row: Mat,
    blocks: usize,
    actions: &RefreshActions,
) -> Result<Mat, RnsBggError> {
    let width = match row.matrix_type().columns.canonicalize() {
        IntExpr::Const(columns) => columns
            .to_usize()
            .filter(|columns| columns % blocks == 0)
            .map(|columns| columns / blocks)
            .ok_or(RnsBggError::Parameters("a block row has a malformed width"))?,
        _ => return Err(RnsBggError::Parameters("a block row needs a constant width")),
    };
    Ok(refresh_fresh_term(blocks, actions, |index| {
        let start = IntExpr::Mul(Box::new(index.clone()), Box::new(IntExpr::constant(width)));
        let end = IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(width)));
        Ok(vec![row.clone().slice(
            None,
            Some(mxx_ir_core::node::IndexRange {
                start: start.canonicalize(),
                end: end.canonicalize(),
            }),
        )])
    })?
    .remove(0))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::test_support::{execute_cpu, full, validate};
    use mxx_backends::{RuntimeValue, matrix::PolyMatrix, poly::Poly};
    use mxx_dsl::DslContext;
    use std::collections::BTreeSet;

    /// Small two-level chains for symbolic-prime tests, with overridable arithmetic sizes.
    fn symbolic_chain(kept: usize) -> (RnsBggParameters, TopReference) {
        let dimension = std::env::var("MXX_REFRESH_TEST_RING_DIMENSION")
            .map(|value| value.parse().unwrap())
            .unwrap_or(8);
        let bits = std::env::var("MXX_REFRESH_TEST_PRIME_BITS")
            .map(|value| value.parse().unwrap())
            .unwrap_or(17);
        let base_bits = std::env::var("MXX_REFRESH_TEST_BASE_BITS")
            .map(|value| value.parse().unwrap())
            .unwrap_or(4);
        let pool = DCRTPolyParams::new(dimension, kept + 1, bits, base_bits, None, None);
        let primes = pool.to_crt().0;
        let next = primes[..kept].to_vec();
        let modulus = next.iter().copied().map(BigUint::from).product::<BigUint>();
        let low = pool.select_modulus(&modulus).unwrap();
        let reference = TopReference { top: primes, next, ring_dimension: dimension, base_bits };
        (RnsBggParameters::new(vec![pool, low], 0, 1).unwrap(), reference)
    }

    /// Materializes bounded action digits without losing any digit coordinate.
    fn action_matrices(ring: &Ring, actions: RefreshActions) -> (Mat, Mat, Mat) {
        let materialize = |action: SmallMatrix| {
            ring.identity(action.matrix_type().rows.clone()).mul_small_rhs(action)
        };
        (materialize(actions.x), materialize(actions.y), materialize(actions.z))
    }

    /// Executes difference matrices using a fresh random hash input and checks every entry.
    fn assert_symbolic_zero(parameters: &RnsBggParameters, graph: mxx_dsl::BuiltGraph) {
        let pool = parameters.level(0).unwrap();
        let mut rings = parameters.levels().to_vec();
        rings.extend(
            pool.to_crt()
                .0
                .into_iter()
                .map(|prime| pool.select_modulus(&BigUint::from(prime)).unwrap()),
        );
        let result = execute_cpu(
            &validate(&graph),
            rings,
            BTreeMap::from([(
                "symbolic-hash".to_owned(),
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
                        "{name} at ({row}, {column})"
                    );
                }
            }
        }
    }

    /// Every digit of symbolic actions equals the constant-index action on random matrices.
    #[test]
    fn symbolic_prime_actions_match_constant_indices() {
        let kept = std::env::var("MXX_REFRESH_TEST_PRIMES")
            .map(|value| value.parse().unwrap())
            .unwrap_or(3);
        let (parameters, reference) = symbolic_chain(kept);
        let ring = parameters.ring(0).unwrap();
        let matrix = ring.hash_matrix(
            ring.bytes_input("symbolic-hash", 32),
            b"a-prime".to_vec(),
            (2, reference.columns()),
        );
        let actions = mxx_dsl::parallel(kept, |prime| {
            let actions = parameters.refresh_actions(0, matrix.clone(), prime, &reference).unwrap();
            Ok(action_matrices(&ring, actions))
        })
        .unwrap();
        let mut graph = DslContext::new("symbolic-prime-actions");
        for prime in 0..kept {
            let actual = actions.at(prime);
            let expected = action_matrices(
                &ring,
                parameters.refresh_actions(0, matrix.clone(), prime, &reference).unwrap(),
            );
            graph = graph
                .output(
                    format!("prime-{prime}"),
                    (actual.0 - expected.0, actual.1 - expected.1, actual.2 - expected.2),
                )
                .unwrap();
        }
        assert_symbolic_zero(&parameters, graph.build().unwrap());
    }

    /// Symbolic CRT terms match all constant indices and reconstruct a random matrix,
    /// including a singleton basis. The iterate body reads a symbolic CRT slice.
    #[test]
    fn symbolic_prime_residues_match_constants_and_reconstruct() {
        let kept = std::env::var("MXX_REFRESH_TEST_PRIMES")
            .map(|value| value.parse().unwrap())
            .unwrap_or(3);
        let (parameters, reference) = symbolic_chain(kept);
        let mut graph = DslContext::new("symbolic-prime-residues");
        for primes in [&reference.top[..1], reference.top.as_slice()] {
            let ring = output_ring(primes, reference.ring_dimension);
            let value = ring.hash_matrix(
                ring.bytes_input("symbolic-hash", 32),
                b"residue".to_vec(),
                (1, 2),
            );
            let terms = mxx_dsl::parallel(primes.len(), |prime| {
                let index = prime.expression()?;
                let residue =
                    value.clone().reduce_modulus(&ring.slice_crt(index.clone(), index + 1));
                Ok(residue_term(primes, prime, reference.ring_dimension, residue).unwrap())
            })
            .unwrap();
            for prime in 0..primes.len() {
                let residue = value.clone().reduce_modulus(&ring.slice_crt(prime, prime + 1));
                let expected =
                    residue_term(primes, prime, reference.ring_dimension, residue).unwrap();
                graph = graph
                    .output(
                        format!("basis-{}-prime-{prime}", primes.len()),
                        terms.at(prime) - expected,
                    )
                    .unwrap();
            }
            let sum = mxx_dsl::iterate(primes.len(), ring.zero((1, 2)), |prime, sum| {
                Ok(sum + terms.at(prime))
            })
            .unwrap();
            graph = graph.output(format!("basis-{}-roundtrip", primes.len()), sum - value).unwrap();
        }
        assert_symbolic_zero(&parameters, graph.build().unwrap());
    }

    /// On a chain whose next level keeps every prime, the target weight of every prime (in a
    /// loop over the primes) is the reference gadget's weight formed modulo the prime and
    /// re-encoded with the prime's CRT idempotent.
    #[test]
    fn surviving_prime_weights_match_weights_modulo_the_prime() {
        let kept = std::env::var("MXX_REFRESH_TEST_PRIMES")
            .map(|value| value.parse().unwrap())
            .unwrap_or(3);
        let (dropping, _) = symbolic_chain(kept);
        let pool = dropping.level(0).unwrap().clone();
        let parameters = RnsBggParameters::new(vec![pool.clone(), pool.clone()], 0, 1).unwrap();
        let primes = pool.to_crt().0;
        let reference = TopReference {
            top: primes.clone(),
            next: primes.clone(),
            ring_dimension: pool.ring_dimension(),
            base_bits: pool.base_bits(),
        };
        let weights = mxx_dsl::parallel(primes.len(), |prime| {
            Ok(parameters.top_target_weight(0, &prime.expression()?, &reference).unwrap())
        })
        .unwrap();
        let next = parameters.ring(1).unwrap();
        let modulus: BigUint = primes.iter().copied().map(BigUint::from).product();
        let mut graph = DslContext::new("surviving-prime-weights");
        for (index, prime) in primes.iter().enumerate() {
            let prime = BigUint::from(*prime);
            let cofactor = &modulus / &prime;
            let scale = &cofactor % &prime;
            let idempotent = (&cofactor % &prime).modinv(&prime).unwrap() * &cofactor % &modulus;
            let ring_q = next.slice_crt(index, index + 1);
            let expected = (top_reference_gadget(&reference).reduce_modulus(&ring_q) *
                ring_q.polynomial([IntExpr::from(scale)]))
            .centered_rebase(&next) *
                next.polynomial([IntExpr::from(idempotent)]);
            // A hashed scalar multiplies both sides, so the check covers a random multiple.
            let scalar =
                next.hash_matrix(next.bytes_input("symbolic-hash", 32), b"scalar".to_vec(), (1, 1));
            graph = graph
                .output(format!("prime-{index}"), scalar * (weights.at(index) - expected))
                .unwrap();
        }
        assert_symbolic_zero(&parameters, graph.build().unwrap());
    }

    /// The decoded residue term of every prime (in a loop over the primes) equals the residue
    /// term of the decoded row switched to the prime's ring.
    #[test]
    fn decoded_residue_terms_match_modulus_switched_residue_terms() {
        let kept = std::env::var("MXX_REFRESH_TEST_PRIMES")
            .map(|value| value.parse().unwrap())
            .unwrap_or(3);
        let (parameters, reference) = symbolic_chain(kept);
        let primes = reference.top.as_slice();
        let ring = output_ring(primes, reference.ring_dimension);
        let decoded =
            ring.hash_matrix(ring.bytes_input("symbolic-hash", 32), b"decoded".to_vec(), (1, 3));
        let terms = mxx_dsl::parallel(primes.len(), |prime| {
            Ok(decoded_residue_term(primes, prime, reference.ring_dimension, decoded.clone())
                .unwrap())
        })
        .unwrap();
        let mut graph = DslContext::new("decoded-residue-terms");
        for prime in 0..primes.len() {
            let residue = decoded.clone().modulus_switch(&ring.slice_crt(prime, prime + 1));
            let expected = residue_term(primes, prime, reference.ring_dimension, residue).unwrap();
            graph = graph.output(format!("prime-{prime}"), terms.at(prime) - expected).unwrap();
        }
        assert_symbolic_zero(&parameters, graph.build().unwrap());
    }

    /// Builds both prime loops once; only integer metadata tables scale with the basis.
    fn symbolic_loop_graph(kept: usize) -> mxx_dsl::BuiltGraph {
        let (parameters, reference) = symbolic_chain(kept);
        let ring = parameters.ring(0).unwrap();
        let matrix = ring.input("a-prime", (2, reference.columns()));
        let value = ring.input("residue", (1, 2));
        let actions = mxx_dsl::parallel(kept, |prime| {
            Ok(action_matrices(
                &ring,
                parameters.refresh_actions(0, matrix, prime, &reference).unwrap(),
            ))
        })
        .unwrap();
        let residue = mxx_dsl::iterate(reference.top.len(), ring.zero((1, 2)), |prime, sum| {
            let index = prime.expression()?;
            let residue = value.reduce_modulus(&ring.slice_crt(index.clone(), index + 1));
            Ok(sum +
                residue_term(&reference.top, prime, reference.ring_dimension, residue).unwrap())
        })
        .unwrap();
        DslContext::new("symbolic-prime-loops")
            .output("actions", actions)
            .unwrap()
            .output("residue", residue)
            .unwrap()
            .build()
            .unwrap()
    }

    /// Both nested-body and total node counts are independent of the number of primes.
    #[test]
    fn symbolic_prime_loop_node_count_is_constant() {
        let small = std::env::var("MXX_REFRESH_TEST_SMALL_PRIMES")
            .map(|value| value.parse().unwrap())
            .unwrap_or(2);
        let large = std::env::var("MXX_REFRESH_TEST_LARGE_PRIMES")
            .map(|value| value.parse().unwrap())
            .unwrap_or(5);
        assert!(small > 0 && large > small);
        let counts = |graph: mxx_dsl::BuiltGraph| {
            validate(&graph);
            graph.graph.scopes().values().map(|scope| scope.nodes().len()).collect::<Vec<_>>()
        };
        assert_eq!(counts(symbolic_loop_graph(small)), counts(symbolic_loop_graph(large)));
    }

    /// Constant invalid, negative and dropped prime indices fail during graph construction.
    #[test]
    fn symbolic_prime_helpers_reject_invalid_constants() {
        let (parameters, reference) = symbolic_chain(2);
        let ring = parameters.ring(0).unwrap();
        let matrix = ring.zero((2, reference.columns()));
        for prime in [-1, 2, 3] {
            assert!(parameters.refresh_actions(0, matrix.clone(), prime, &reference).is_err());
        }
        for prime in [-1, 3] {
            assert!(
                residue_term(&reference.top, prime, reference.ring_dimension, ring.zero((1, 1)))
                    .is_err()
            );
        }
        assert!(residue_term(&[], 0, reference.ring_dimension, ring.zero((1, 1))).is_err());
    }

    /// On every chain of a refresh, the target action of the output prime
    /// `q` maps the level-`L` payload scale to the reference format:
    /// `Delta_L G^st_L G^-1(R_q) = Q_q (Delta_0 G^st_0 mod q)` modulo `Q_L`,
    /// with `T_q` the reference chain's own `Delta_0 G^st_0` modulo `q`;
    /// reductions between rings of different prime orders agree; and the
    /// residues of every chain recombine to the value modulo `Q_0`.
    #[test]
    fn top_refresh_target_weight_and_combination_are_exact() {
        use crate::refresh::tests::{declaration, lut_source, primes, reference};
        let (source, output) = lut_source(false);
        let top = declaration(&source, output, vec![3, 2, 2]).unwrap();
        let reference = reference();
        let (reference_pool, reference_chain) =
            RnsBggParameters::refresh_chain(&top, None, 0, 1).unwrap();
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
                RnsBggParameters::refresh_chain(&top, Some(chain), 0, 1).unwrap();
            parameters.extend(chain_parameters.levels().iter().cloned());
            parameters.push(pool.select_modulus(&BigUint::from(q)).unwrap());
            let level = top.level();
            let ring = chain_parameters.ring(level).unwrap();
            let modulus: BigUint =
                chain_parameters.level(level).unwrap().modulus().as_ref().clone();
            let index = chain_parameters
                .level(level)
                .unwrap()
                .to_crt()
                .0
                .iter()
                .position(|prime| *prime == q)
                .unwrap();
            let weight = chain_parameters
                .top_target_weight(level, &IntExpr::constant(index), &reference)
                .unwrap();
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
            let part = top.part((chain, 0)).unwrap();
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

    /// A part is checked against its own chain: another chain
    /// drops its output prime before the plaintext level, so the part is
    /// rejected. Distinct chains derive distinct stage keys.
    #[test]
    fn top_refresh_parts_bind_to_their_chain() {
        use crate::refresh::tests::{declaration, lut_source};
        let (source, output) = lut_source(false);
        let top = declaration(&source, output, vec![3, 2, 1]).unwrap();
        let chain =
            |index: usize| RnsBggParameters::refresh_chain(&top, Some(index), 0, 1).unwrap().1;
        let (first, second) = (chain(0), chain(1));
        let part = top.part((0, 1)).unwrap();
        assert_eq!(first.refresh_level(part, &source).unwrap(), 1);
        assert!(second.refresh_level(part, &source).is_err(), "a swapped chain");
        assert_eq!(second.refresh_level(top.part((1, 0)).unwrap(), &source).unwrap(), 1);
        assert_eq!(first.level(2).unwrap().to_crt().0, vec![top.output_prime(0).unwrap()]);
        assert_eq!(second.level(2).unwrap().to_crt().0, vec![top.output_prime(1).unwrap()]);
        // Both chains list the decoding primes in the canonical order.
        assert_eq!(first.level(1).unwrap().to_crt().0, second.level(1).unwrap().to_crt().0);
        let key = [5; 32];
        let keys = (0..top.chains())
            .map(|chain| child_hash_key(&key, &top.chain_namespace(chain)))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), top.chains());
    }

    /// Synthetic exact carriers (zero errors, no ModDown) under a mask `s = (s_0, 1)` on the
    /// chain of part `(0, 1)`: the one encoding, the target's component 1 and the fresh
    /// carriers. The combination of the rows minus the mask times the same combination of the
    /// keys is exactly `-Q_q (s A'_(1) - x T_q + e')` with `T_q` the reference's
    /// `Delta_0 G^st_0 mod q`: the decompositions `(I_2 ⊗ G_Q)^-1`, `G^st^-1(R_q)` and `G_Q^-1`
    /// reconstruct their weights, and the digits stay within the bounds of the noise report.
    #[test]
    fn refresh_combination_of_exact_carriers_is_the_scaled_target_row() {
        use crate::refresh::tests::{declaration, lut_source, reference};
        let (source, output) = lut_source(false);
        let top = declaration(&source, output, vec![3, 2, 2]).unwrap();
        let declaration = top.part((0, 1)).unwrap();
        let reference = reference();
        let (reference_pool, reference_chain) =
            RnsBggParameters::refresh_chain(&top, None, 0, 1).unwrap();
        let (pool, parameters) = RnsBggParameters::refresh_chain(&top, Some(0), 0, 1).unwrap();
        let level = top.level();
        let q = top.output_prime(0).unwrap();
        let ring = parameters.ring(level).unwrap();
        let hash = ring.bytes_input("hash-key", 32);
        let sample = |tag: &str, columns: usize| {
            ring.hash_matrix(hash.clone(), tag.as_bytes().to_vec(), (1, columns))
        };
        let columns = parameters.level(level).unwrap().modulus_digits();
        let outputs = declaration.columns();
        let width = parameters.component_columns(level).unwrap();
        let s0 = sample("s0", 1);
        // A hash-sampled `A'` of the reference ring, reduced as the parts reduce the caller's.
        let a_prime = component_matrix(
            top_reference_ring(&reference).hash_matrix(
                hash.clone(),
                b"a-prime".to_vec(),
                (2, 2 * reference.columns()),
            ),
            declaration,
        )
        .reduce_modulus(&ring);
        let one = parameters.refresh_one_key(level, hash.clone(), declaration).unwrap();
        let gadget = parameters.public_gadget(level).unwrap();
        let state = parameters.state_gadget(level).unwrap();
        let delta = parameters.scalar(level, parameters.message_scale(level).unwrap()).unwrap();
        let x = ring.polynomial([0, 0, 0, 1].map(IntExpr::constant));
        let target = (sample("kx0", width), sample("kx1", width));
        let fresh_keys = (0..outputs)
            .map(|j| (sample(&format!("kf0-{j}"), columns), sample(&format!("kf1-{j}"), columns)))
            .collect::<Vec<_>>();
        let fresh_noise = (0..outputs)
            .map(|j| ring.polynomial([j as i64 - 1, 1, 0, -1].map(IntExpr::constant)))
            .collect::<Vec<_>>();
        let one_row = (
            s0.clone() * one.0.0.clone() + one.1.0.clone() - s0.clone() * gadget.clone(),
            s0.clone() * one.0.1.clone() + one.1.1.clone() - gadget.clone(),
        );
        let target_row =
            s0.clone() * target.0.clone() + target.1.clone() - delta * x.clone() * state;
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
        let ring_q = Ring::from_crt_moduli(vec![IntExpr::from(q)], reference.ring_dimension);
        let target_weight = top_reference_target(&reference, &ring_q, q).centered_rebase(&ring);
        let refreshed = s0.clone() * row(0) + row(1) - x * target_weight +
            Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, fresh_noise);
        let modulus: BigUint = parameters.level(level).unwrap().modulus().as_ref().clone();
        let actions = parameters
            .refresh_actions(
                level,
                a_prime.clone(),
                declaration.output_prime(),
                declaration.reference(),
            )
            .unwrap();
        // The fresh terms of the fresh carriers' column blocks, each block selected from the row
        // of the blocks the fresh child's row `Sum` returns.
        let fresh = |rows: Vec<Mat>| {
            let row = Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, rows);
            row_fresh_term(row, outputs, &actions).unwrap()
        };
        let decoded =
            refresh_combination(one_row, target_row, fresh(fresh_rows), &actions).unwrap();
        let top_key = refresh_combination(
            (one.0.0.clone(), one.0.1.clone()),
            target.0.clone(),
            fresh(fresh_keys.iter().map(|key| key.0.clone()).collect()),
            &actions,
        )
        .unwrap();
        let bottom_key = refresh_combination(
            (one.1.0.clone(), one.1.1.clone()),
            target.1.clone(),
            fresh(fresh_keys.iter().map(|key| key.1.clone()).collect()),
            &actions,
        )
        .unwrap();
        let cofactor = parameters.scalar(level, BigInt::from(&modulus / q)).unwrap();
        let residual = decoded - (s0 * top_key + bottom_key) + cofactor * refreshed;
        let graph = DslContext::new("refresh-exact-carriers")
            .output("residual", residual)
            .unwrap()
            .output("x", ring.identity(2 * columns).mul_small_rhs(actions.x))
            .unwrap()
            .output("y", ring.identity(width).mul_small_rhs(actions.y))
            .unwrap()
            .output("z", ring.identity(columns).mul_small_rhs(actions.z))
            .unwrap()
            .build()
            .unwrap();
        let mut rings = reference_chain.levels().to_vec();
        rings.extend(parameters.levels().iter().cloned());
        rings.push(pool.select_modulus(&BigUint::from(q)).unwrap());
        rings.push(reference_pool);
        let result = execute_cpu(
            &validate(&graph),
            rings,
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
        // The bounds of the refresh noise report hold for the actual digits.
        let (largest, _) = digits("x");
        assert!(largest <= half_base(level), "X digits");
        let (largest, l1) = digits("y");
        assert!(largest <= half_base(level + 1), "Y digits");
        let next_width = BigUint::from(parameters.level(level + 1).unwrap().modulus_digits());
        assert!(l1.iter().all(|norm| norm <= &(&next_width * half_base(level + 1))), "Y l1");
        let (_, l1) = digits("z");
        let (expected, _) = parameters.gadget_inverse_norms(level, &(&modulus / q)).unwrap();
        assert_eq!(l1, vec![expected], "z l1");
        let residual = full(&result.outputs["residual"]);
        for column in 0..outputs {
            assert!(
                residual
                    .entry(0, column)
                    .coeffs_biguints()
                    .iter()
                    .all(|value| value == &BigUint::ZERO),
                "column {column}"
            );
        }
    }

    /// A part is defined only on a Static, non-terminal decoding level of its chain, and its
    /// children's chains end at that level with a strict drop.
    #[test]
    fn refresh_level_requires_a_static_non_terminal_target_and_child_chains() {
        use crate::refresh::tests::{declaration, lut_source};
        let (source, output) = lut_source(false);
        let top = declaration(&source, output, vec![3, 2, 1]).unwrap();
        let (pool, parameters) = RnsBggParameters::refresh_chain(&top, Some(0), 0, 1).unwrap();
        let part = top.part((0, 0)).unwrap();
        assert_eq!(parameters.refresh_level(part, &source).unwrap(), 1);
        let terminal = RnsBggParameters::new(parameters.levels()[..2].to_vec(), 0, 1).unwrap();
        assert!(terminal.refresh_level(part, &source).is_err());
        let dynamic = parameters.clone().with_rhs_mode(RhsMode::Dynamic);
        assert!(dynamic.refresh_level(part, &source).is_err());
        let fresh = parameters.refresh_fresh_parameters(&top, 0, &source, &pool, 1).unwrap();
        let smudge =
            parameters.decoder_smudge_parameters(part.decoder(), &source, &pool, 1).unwrap();
        assert_eq!(
            parameters.validate_refresh_children(part, &source, &fresh, &smudge).unwrap(),
            1
        );
        assert!(
            parameters.validate_refresh_children(part, &source, &parameters, &smudge).is_err(),
            "the source chain is not a child chain"
        );
    }
}
