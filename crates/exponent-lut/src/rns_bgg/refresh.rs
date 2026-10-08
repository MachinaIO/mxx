//! The RNS-BGG arithmetic of the noise refresh (see [`crate::refresh`]).
//!
//! The stages share the public pieces defined here: the hash-sampled `A'`
//! and one-encoding key, the digit actions `X_i, Y_i, z_i` of every prime,
//! the combination `d_i` (the same on keys and rows), the CRT coefficients,
//! and the refresh children's chains, hash namespaces and artifact names.
use super::*;
use crate::{program::ExponentLutProgram, refresh::RefreshDeclaration};

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
            parameters.modulus_digits() != declaration.columns() ||
            parameters.ring_dimension() as usize != declaration.coefficients() ||
            primes.iter().any(|prime| *prime <= u64::from(declaration.eta()))
        {
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
        let workload = RnsBggDerivedWorkload::from_program(declaration.fresh().program())?;
        let levels = declaration
            .fresh()
            .outputs()
            .iter()
            .map(|wire| workload.level(*wire))
            .collect::<Result<BTreeSet<_>, _>>()?;
        match levels.into_iter().collect::<Vec<_>>().as_slice() {
            [level] => Ok(*level),
            _ => Err(RnsBggError::Parameters("fresh-noise outputs on different levels")),
        }
    }

    /// Check `fresh` and every chain of `smudges` (one per decoder, in
    /// decoder order) against `declaration`.
    pub(crate) fn validate_refresh_children(
        &self,
        declaration: &RefreshDeclaration,
        source_program: &ExponentLutProgram,
        fresh: &Self,
        smudges: &[&Self],
    ) -> Result<usize, RnsBggError> {
        let level = self.refresh_level(declaration, source_program)?;
        self.validate_child_chain(fresh, level, Self::refresh_fresh_level(declaration)?)?;
        if smudges.len() != declaration.decoders().len() {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        for (smudge, decoder) in smudges.iter().zip(declaration.decoders()) {
            self.validate_decoder_smudge_parameters(smudge, decoder, source_program)?;
        }
        Ok(level)
    }

    /// The public `A' in R_Q^(2 x m)` of `declaration` at level `level`,
    /// sampled from the stage hash key.
    pub(crate) fn refresh_matrix(
        &self,
        level: usize,
        hash_key: Bytes,
        declaration: &RefreshDeclaration,
    ) -> Result<Mat, RnsBggError> {
        let ring = self.ring(level)?;
        let columns = self.level(level)?.modulus_digits();
        Ok(ring.hash_matrix(hash_key, declaration.matrix_hash_tag(), (2, columns)))
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
        let tag = [b"refresh/v1/one".as_slice(), declaration.spec().as_slice()].concat();
        Ok(super::public_key::matrix_from_hash(&ring, hash_key, tag, columns))
    }

    /// The digit actions of prime `prime` at level `level` for the public
    /// `a_prime = A'`.
    pub(crate) fn refresh_actions(
        &self,
        level: usize,
        prime: usize,
        a_prime: Mat,
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
        let weight =
            (cofactor_scalar * self.public_gadget(level)?).reduce_modulus(&self.ring(level + 1)?);
        let y = self.gadget_inverse(level + 1, level, weight)?;
        let z = self.gadget_inverse(level, level, self.scalar(level, BigInt::from(cofactor))?)?;
        Ok(RefreshActions { x, y, z })
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

/// `d_i = one X_i - target Y_i + [fresh_j z_i]_j` for one component: the two
/// one-encoding blocks `(one_0, one_1)` (`1 x m` each), the target's
/// component-1 block (`1 x w`) and every fresh carrier's component-1 block
/// (`1 x m`). Applied to the key rows it gives the decoded key `K_i`, applied
/// to the rows it gives the decoded row `d_i` (`1 x m`).
pub(crate) fn refresh_combination(
    one: (Mat, Mat),
    target: Mat,
    fresh: &[Mat],
    actions: &RefreshActions,
) -> Mat {
    let one = Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![one.0, one.1])
        .mul_small_rhs(actions.x.clone());
    let fresh = Mat::concat(
        mxx_ir_core::node::ConcatAxis::Columns,
        fresh.iter().map(|carrier| carrier.clone().mul_small_rhs(actions.z.clone())).collect(),
    );
    one - target.mul_small_rhs(actions.y.clone()) + fresh
}

/// Column `column` of a one-row matrix.
pub(crate) fn refresh_column(row: Mat, column: usize) -> Mat {
    row.slice(
        None,
        Some(mxx_ir_core::node::IndexRange {
            start: IntExpr::constant(column),
            end: IntExpr::constant(column + 1),
        }),
    )
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
        let smudges = (0..primes * columns)
            .map(|_| crate::decoder::decoding::fixture::keys(n * digits))
            .collect::<Vec<_>>();
        let layouts = |keys: &[PbcGeneratedKeyLayout]| {
            keys.iter().map(|key| key.public_layout.clone()).collect::<Vec<_>>()
        };
        let declaration = RefreshDeclaration::new(
            source,
            target,
            profile,
            1,
            primes,
            columns,
            digits,
            layouts(&fresh),
            smudges.iter().map(|keys| layouts(keys)).collect(),
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
            let actions = parameters.refresh_actions(0, prime, a_prime.clone()).unwrap();
            let decoded =
                refresh_combination(one_row.clone(), target_row.clone(), &fresh_rows, &actions);
            let top = refresh_combination(
                (one.0.0.clone(), one.0.1.clone()),
                target.0.clone(),
                &fresh_keys.iter().map(|key| key.0.clone()).collect::<Vec<_>>(),
                &actions,
            );
            let bottom = refresh_combination(
                (one.1.0.clone(), one.1.1.clone()),
                target.1.clone(),
                &fresh_keys.iter().map(|key| key.1.clone()).collect::<Vec<_>>(),
                &actions,
            );
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
            parameters.validate_refresh_children(&declaration, &source, &fresh, &smudges).unwrap(),
            0
        );
        assert!(
            parameters.validate_refresh_children(&declaration, &source, &smudge, &smudges).is_err(),
            "the smudge chain is not the fresh chain"
        );
        assert!(
            parameters
                .validate_refresh_children(&declaration, &source, &fresh, &smudges[1..])
                .is_err()
        );
    }
}
