//! Operations on the rows of a Static chain outside a Program: the re-targeting `Gamma` of an
//! encoding to another form and level, and the public multiplication of a general row.
//!
//! A Program's lowered output is an encoding in the form its carrier names: the state form
//! `Delta_l G^st_l` of a private wire at level `l` (a raw coefficient wire is one too), or the
//! public gadget `G_(Q_l)` of a public wire. A protocol that combines outputs of several
//! Programs, or rows that are not Program wires (Diamond iO 2 combines component rows whose
//! targets are not of the payload-secret form, see `mxx-io`), moves them to one form by
//! `Gamma` and multiplies them by public encodings here. [`RnsBggParameters::retarget_noise`]
//! bounds the error a `Gamma` leaves.

use super::*;

impl RnsBggParameters {
    /// The level whose gadget the form `form` of `level` is a multiple of: `level + 1` for the
    /// state form (whose gadget is the public gadget of the next level), `level` for the
    /// public form.
    fn gadget_level(level: usize, form: RnsBggForm) -> usize {
        match form {
            RnsBggForm::State => level + 1,
            RnsBggForm::Public => level,
        }
    }

    /// `Gamma` of an encoding (`row` and its key `key`) in `form` at `level` to the target form
    /// `target` (a `1 x w` row in the ring of `target_level`) at `target_level`:
    ///
    /// `Gamma(c) = ModDown_(Q_l -> Q_t)(c G_N^-1((N / Q_t) target))`
    ///
    /// for the gadget modulus `N` of the source form (`Q_(l+1)` for the state form, `Q_l` for the
    /// public form), whose source form is `(Q_l / N) G_N`. The target `T` of the encoding is kept
    /// and its form becomes `target`: `(Q_l / N) G_N G_N^-1((N / Q_t) target) = (Q_l / Q_t)
    /// target`, which the ModDown divides exactly (`Rel.modDown`). It needs `Q_t | N`, that is
    /// `target_level` at or below the gadget level. The key's blocks are acted on alike, and the
    /// error becomes `(e D - r_c + s_bar r_0 + r_1) / (Q_l / Q_t)` for the digits `D` and the
    /// rounding remainders of the row and the key blocks (no rounding when `Q_t = Q_l`).
    pub fn retarget_encoding(
        &self,
        level: usize,
        form: RnsBggForm,
        target_level: usize,
        target: Mat,
        (row, key): (RnsBggRow, RnsBggMatrix),
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        let gadget_level = Self::gadget_level(level, form);
        let gadget = self.level(gadget_level)?;
        let target_parameters = self.level(target_level)?;
        let gadget_modulus: BigUint = gadget.modulus().as_ref().clone();
        let target_modulus: BigUint = target_parameters.modulus().as_ref().clone();
        if target_level < gadget_level || !(&gadget_modulus % &target_modulus).is_zero() {
            return Err(RnsBggError::Parameters("a re-targeting needs a divisor of its gadget"));
        }
        let gadget_ring = self.ring(gadget_level)?;
        let ratio = &gadget_modulus / &target_modulus;
        let scaled = target.centered_rebase(&gadget_ring) *
            gadget_ring.polynomial([IntExpr::from(BigInt::from(ratio))]);
        let digits = scaled
            .decompose(public_gadget_base(gadget.base_bits())?, gadget.modulus_digits())
            .centered_rebase(&self.ring(level)?);
        let act = |entry: Mat| entry.mul_small_rhs(digits.clone());
        let row = map_row(row, act);
        let key = map_matrix(key, act);
        if self.level(level)?.modulus() == target_parameters.modulus() {
            return Ok((row, key));
        }
        let ring = self.ring(target_level)?;
        Ok((checked_mod_down_encoding(row, &ring)?, checked_mod_down_public(key, &ring)?))
    }

    /// The public multiplication `MulPub(P, c)` at `level` of the public encoding `public`
    /// (row `P`, key `K^P`, in the public form of `level`) of the clear plaintext `alpha` by the
    /// general row `c` with key `K^c` (any form at `level`, `w` columns per component): for each
    /// component `j`, with `D_j = [G^-1(K^c_(0j)); G^-1(K^c_(1j))]` under the public gadget of
    /// `level`,
    ///
    /// `out_j = [P_0 | P_1] D_j + alpha c_j`, `K^out_(kj) = [K^P_(k0) | K^P_(k1)] D_j`,
    ///
    /// an encoding of `alpha T_j` in the form of `c` for every target `T` of `c` (the general
    /// component relation `mul_public` of `ExponentPublic.lean`: the mask term
    /// `alpha (s_bar K^c_(0j) + K^c_(1j))` of the public product cancels against `alpha c_j`).
    pub fn mul_public_rows(
        &self,
        level: usize,
        (public_row, public_key, alpha): (&RnsBggRow, &RnsBggMatrix, &Mat),
        (row, key): (&RnsBggRow, &RnsBggMatrix),
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        let parameters = self.level(level)?;
        let base = public_gadget_base(parameters.base_bits())?;
        let digits = parameters.modulus_digits();
        let product = |column: usize| {
            let (top, bottom) = match column {
                0 => (
                    key.0.0.clone().decompose(base, digits),
                    key.1.0.clone().decompose(base, digits),
                ),
                _ => (
                    key.0.1.clone().decompose(base, digits),
                    key.1.1.clone().decompose(base, digits),
                ),
            };
            let entry = |left: &Mat, right: &Mat| {
                left.clone().mul_small_rhs(top.clone()) +
                    right.clone().mul_small_rhs(bottom.clone())
            };
            let row_entry = entry(&public_row.0, &public_row.1);
            let key_entries =
                (entry(&public_key.0.0, &public_key.0.1), entry(&public_key.1.0, &public_key.1.1));
            (row_entry, key_entries)
        };
        let (row_0, (key_00, key_10)) = product(0);
        let (row_1, (key_01, key_11)) = product(1);
        Ok((
            (row_0 + alpha.clone() * row.0.clone(), row_1 + alpha.clone() * row.1.clone()),
            ((key_00, key_01), (key_10, key_11)),
        ))
    }

    /// The integer entries of the form `form` of `level` ([`Self::form`]) modulo `Q_level`.
    ///
    /// The public gadget of a level is the exact CRT gadget: entry `t d + k` is `B^k` modulo
    /// tower `t` and zero modulo the other towers. The state form is `Delta_l` times the public
    /// gadget of level `l + 1` (centered there), or `Delta_l` alone on the last level.
    pub fn form_weights(
        &self,
        level: usize,
        form: RnsBggForm,
    ) -> Result<Vec<BigUint>, RnsBggError> {
        let modulus: BigUint = self.level(level)?.modulus().as_ref().clone();
        let gadget = |gadget_level: usize| -> Result<Vec<BigUint>, RnsBggError> {
            let parameters = self.level(gadget_level)?;
            let (moduli, _, _) = parameters.to_crt();
            let gadget_modulus: BigUint = parameters.modulus().as_ref().clone();
            let digits = parameters.modulus_digits() / moduli.len();
            let base = BigUint::one() << parameters.base_bits();
            let mut weights = Vec::with_capacity(parameters.modulus_digits());
            for tower in &moduli {
                let tower = BigUint::from(*tower);
                let cofactor = &gadget_modulus / &tower;
                let idempotent = (&cofactor % &tower)
                    .modinv(&tower)
                    .ok_or(RnsBggError::Parameters("CRT primes are not coprime"))? *
                    &cofactor;
                let mut power = BigUint::one();
                for _ in 0..digits {
                    weights.push(&idempotent * &power % &gadget_modulus);
                    power = power * &base % &tower;
                }
            }
            Ok(weights)
        };
        Ok(match form {
            RnsBggForm::Public => gadget(level)?,
            RnsBggForm::State => {
                let delta = BigInt::from(self.message_scale(level)?);
                let modulus = BigInt::from(modulus);
                match self.levels.get(level + 1) {
                    Some(next) => {
                        let next_modulus = BigInt::from(next.modulus().as_ref().clone());
                        gadget(level + 1)?
                            .into_iter()
                            .map(|weight| {
                                let weight = BigInt::from(weight);
                                let centered = if &weight * 2 > next_modulus {
                                    weight - &next_modulus
                                } else {
                                    weight
                                };
                                canonical_residue(&delta * centered, &modulus)
                            })
                            .collect()
                    }
                    None => vec![canonical_residue(delta, &modulus)],
                }
            }
        })
    }

    /// The error a re-targeting ([`Self::retarget_encoding`]) of an encoding in `form` at
    /// `level` with error `noise` to the target row of integer entries `weights` at
    /// `target_level` leaves: the largest `l1` norm `g` of the digit columns
    /// `G_N^-1((N / Q_t) w)` (balanced, as the runtime decomposes them) scales it, and a ModDown
    /// by `p = Q_l / Q_t > 1` divides it and adds the roundings, `ceil((2 g e + (n + 2)(p - 1))
    /// / 2p)` (the bound of [`Self::lut_mod_down_bound`] at ratio `p`). The digits are constants,
    /// so the moments scale coherently.
    pub fn retarget_noise(
        &self,
        (level, form): (usize, RnsBggForm),
        (target_level, weights): (usize, &[BigUint]),
        noise: &RefreshSigmaNoise,
    ) -> Result<RefreshSigmaNoise, RnsBggError> {
        let gadget_level = Self::gadget_level(level, form);
        let gadget_modulus: BigUint = self.level(gadget_level)?.modulus().as_ref().clone();
        let target_modulus: BigUint = self.level(target_level)?.modulus().as_ref().clone();
        if target_level < gadget_level || !(&gadget_modulus % &target_modulus).is_zero() {
            return Err(RnsBggError::Parameters("a re-targeting needs a divisor of its gadget"));
        }
        let ratio = &gadget_modulus / &target_modulus;
        let target = BigInt::from(target_modulus.clone());
        let gadget = BigInt::from(gadget_modulus);
        let gain = weights.iter().try_fold(BigUint::zero(), |largest, weight| {
            let weight = BigInt::from(weight.clone());
            let centered = if &weight * 2 > target { weight - &target } else { weight };
            let lifted = canonical_residue(centered * BigInt::from(ratio.clone()), &gadget);
            let (l1, _) = self.gadget_inverse_norms(gadget_level, &lifted)?;
            Ok::<_, RnsBggError>(largest.max(l1))
        })?;
        let source_modulus: BigUint = self.level(level)?.modulus().as_ref().clone();
        let p = &source_modulus / &target_modulus;
        let error = &noise.error * &gain;
        let moments = noise.moments.as_ref().map(|moments| moments.scaled(&gain));
        if p.is_one() {
            return Ok(RefreshSigmaNoise { error, moments });
        }
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let twice_p = &p * 2u8;
        Ok(RefreshSigmaNoise {
            error: (error * 2u8 + (&n + 2u8) * (&p - 1u8) + &twice_p - 1u8) / twice_p,
            moments: moments
                .map(|moments| super::noise::mod_down_moments_by(&n, &p, &moments))
                .transpose()?,
        })
    }
}

/// `value` modulo `modulus` in `[0, modulus)`.
fn canonical_residue(value: BigInt, modulus: &BigInt) -> BigUint {
    let residue = value % modulus;
    let residue =
        if residue.sign() == num_bigint::Sign::Minus { residue + modulus } else { residue };
    residue.to_biguint().expect("a nonnegative residue")
}
