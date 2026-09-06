//! BGV graph builders with exact CRT gadget switching and modulus reduction.
use crate::{
    FheCommonParams, FheError, FheScheme,
    utils::{self, check_family, check_matrix},
};
use mxx_dsl::{
    DslError, Family, GraphValue, GraphValueSchema, Int, Mat, MatType, Ring, concat_rows, parallel,
};
use mxx_ir_core::{
    IntExpr, ValueHandle,
    node::{ConcatAxis, IndexRange},
    types::WireType,
};
use mxx_primitives::{
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    utils::{mod_inverse, mod_inverse_biguints},
};
use num_bigint::{BigInt, BigUint};

#[derive(Clone)]
pub struct BgvParams {
    pub common: FheCommonParams,
    pub plaintext_modulus: u64,
}

/// Descending coefficients in -s: (a,b), or (a1*a2,a1*b2+b1*a2,b1*b2).
/// The public factor and noise bound belong to the graph schema, not the matrix artifact.
/// `noise_bound` bounds each integer coefficient of e in v = centered(v mod t) + t*e,
/// including plaintext representative carries, before reduction modulo the ciphertext modulus.
#[derive(Clone)]
pub struct BgvCiphertext {
    pub components: Mat,
    /// Public invertible multiplier f modulo the plaintext modulus t.
    /// Under the decryption bound, the centered ciphertext phase v satisfies
    /// v mod t = f*m mod t, where m is the logical plaintext polynomial.
    /// Decryption recovers m by multiplying v by f^(-1) modulo t.
    ///
    /// Fresh encryption sets f = 1. Multiplication sets f = f_lhs*f_rhs mod t;
    /// dropping a CRT prime p sets f = f*p^(-1) mod t because modswitch divides
    /// the phase by p. These updates preserve the intended plaintext semantics.
    /// Addition requires equal factors; use match_correction_factor to align
    /// them first. Relinearization and rotations preserve f.
    ///
    /// This is graph-construction metadata, not the plaintext modulus or an
    /// encryption scale. Pass it alongside components across protocol stages.
    pub correction_factor: u64,
    pub noise_bound: BigUint,
}

#[derive(Clone, PartialEq)]
pub struct BgvCiphertextSchema {
    pub components: MatType,
    /// Retains the plaintext multiplier described by BgvCiphertext::correction_factor.
    pub correction_factor: u64,
    pub noise_bound: BigUint,
}

impl GraphValue for BgvCiphertext {
    type Schema = BgvCiphertextSchema;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.components.flatten()
    }
    fn schema(&self) -> Self::Schema {
        BgvCiphertextSchema {
            components: self.components.schema(),
            correction_factor: self.correction_factor,
            noise_bound: self.noise_bound.clone(),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            components: Mat::from_values(&schema.components, values)?,
            correction_factor: schema.correction_factor,
            noise_bound: schema.noise_bound.clone(),
        })
    }
}
impl GraphValueSchema for BgvCiphertextSchema {
    type Value = BgvCiphertext;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        BgvCiphertext {
            components: self.components.placeholders_from(next),
            correction_factor: self.correction_factor,
            noise_bound: self.noise_bound.clone(),
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        self.components.wire_types()
    }
}

pub(crate) fn row(matrix: &Mat, index: usize) -> Mat {
    matrix.clone().slice(Some(IndexRange { start: index.into(), end: (index + 1).into() }), None)
}

impl BgvParams {
    pub fn new(common: FheCommonParams, plaintext_modulus: u64) -> Result<Self, FheError> {
        common.validate()?;
        if plaintext_modulus < 2 ||
            common
                .ring
                .to_crt()
                .0
                .iter()
                .any(|&p| mod_inverse(plaintext_modulus % p, p).is_none())
        {
            return Err(FheError::InvalidParameters(
                "plaintext modulus must be at least two and coprime to every CRT prime",
            ));
        }
        Ok(Self { common, plaintext_modulus })
    }

    // For v = r + t*e with centered r, |r| <= floor(t/2) and |e| <= noise.
    // This converts an unscaled noise bound into a bound on the whole phase.
    fn phase_from_noise(&self, noise: &BigUint) -> BigUint {
        BigUint::from(self.plaintext_modulus / 2) + BigUint::from(self.plaintext_modulus) * noise
    }

    // Re-centering the plaintext residue can add a carry to e. Bounding the
    // whole phase first includes that carry even when the sampled noise is zero.
    fn noise_from_phase(&self, phase: &BigUint) -> BigUint {
        (phase + BigUint::from(self.plaintext_modulus / 2)) / self.plaintext_modulus
    }

    /// Sufficient correctness condition for the tracked integer phase to avoid wrapping Q.
    pub fn can_decrypt(&self, ct: &BgvCiphertext) -> Result<bool, FheError> {
        self.ciphertext_rows(ct)?;
        let parameters = self.common.parameters_at(self.level_of(&ct.components)?)?;
        // Require |v| < Q/2. phase_from_noise already includes t*noise_bound,
        // so this is 2*(floor(t/2) + t*noise_bound) < Q; do not multiply by t again.
        Ok(self.phase_from_noise(&ct.noise_bound) * 2u8 < *parameters.modulus())
    }

    /// One exact CRT gadget switch adds sum(e_i*d_i), with L polynomials and |d_i| <= B/2.
    fn key_switch_noise(&self, parameters: &DCRTPolyParams) -> BigUint {
        BigUint::from(parameters.ring_dimension()) *
            BigUint::from(parameters.modulus_digits()) *
            (BigUint::from(1u8) << (parameters.base_bits() - 1) as usize) *
            &self.common.error_cutoff
    }

    pub(crate) fn level_of(&self, matrix: &Mat) -> Result<usize, FheError> {
        for level in 0..self.common.ring.to_crt().2 {
            let params = self.common.parameters_at(level)?;
            if matrix.matrix_type().modulus == IntExpr::constant(params.modulus().as_ref().clone())
            {
                return Ok(level);
            }
        }
        Err(FheError::LevelMismatch)
    }
    pub(crate) fn validate_ciphertext(
        &self,
        ct: &BgvCiphertext,
        rows: usize,
    ) -> Result<usize, FheError> {
        if ct.correction_factor >= self.plaintext_modulus ||
            mod_inverse(ct.correction_factor, self.plaintext_modulus).is_none()
        {
            return Err(FheError::InvalidCorrectionFactor);
        }
        let level = self.level_of(&ct.components)?;
        utils::check_matrix(&self.common.parameters_at(level)?, &ct.components, rows, 1)?;
        Ok(level)
    }
    fn ciphertext_rows(&self, ct: &BgvCiphertext) -> Result<usize, FheError> {
        let rows = if ct.components.matrix_type().rows == IntExpr::constant(2) { 2 } else { 3 };
        self.validate_ciphertext(ct, rows)?;
        Ok(rows)
    }
    pub(crate) fn key_switch_key(
        &self,
        secret: &Mat,
        level: usize,
        target: &Mat,
    ) -> Result<Mat, FheError> {
        utils::check_matrix(&self.common.ring, secret, 1, 1)?;
        let params = self.common.parameters_at(level)?;
        utils::check_matrix(&params, target, 1, 1)?;
        let ring = Ring::new(params.modulus().as_ref().clone(), params.ring_dimension());
        let s = secret.clone().reduce_modulus(params.modulus().as_ref().clone());
        let width = params.modulus_digits();
        let a = ring.uniform_residue((1, width));
        let e = self.common.gaussian(&params, 1, width);
        let g = ring.gadget(1, BigUint::from(1u8) << params.base_bits(), width);
        // The key phase is target*G + t*e, so multiplying by exact gadget
        // digits substitutes the target without introducing a rounding error.
        let b = &s * &a + &utils::scalar(&params, self.plaintext_modulus) * &e + target * &g;
        Ok(concat_rows![a, b])
    }
    /// Evaluation keys encrypt a secret-dependent target; callers must account for this assumption.
    pub fn relinearization_key(&self, secret: &Mat, level: usize) -> Result<Mat, FheError> {
        let params = self.common.parameters_at(level)?;
        utils::check_matrix(&self.common.ring, secret, 1, 1)?;
        let s = secret.clone().reduce_modulus(params.modulus().as_ref().clone());
        self.key_switch_key(secret, level, &(&s * &s))
    }
    pub fn mul_unrelinearized(
        &self,
        lhs: &BgvCiphertext,
        rhs: &BgvCiphertext,
    ) -> Result<BgvCiphertext, FheError> {
        let level = self.validate_ciphertext(lhs, 2)?;
        if level != self.validate_ciphertext(rhs, 2)? {
            return Err(FheError::LevelMismatch);
        }
        let a1 = row(&lhs.components, 0);
        let b1 = row(&lhs.components, 1);
        let a2 = row(&rhs.components, 0);
        let b2 = row(&rhs.components, 1);
        // Expand (b1 - s*a1)(b2 - s*a2) in descending powers of -s.
        // A negacyclic product coefficient sums N signed products, hence N*V1*V2.
        Ok(BgvCiphertext {
            components: concat_rows![&a1 * &a2, &a1 * &b2 + &b1 * &a2, &b1 * &b2],
            noise_bound: self.noise_from_phase(
                &(BigUint::from(self.common.ring.ring_dimension()) *
                    self.phase_from_noise(&lhs.noise_bound) *
                    self.phase_from_noise(&rhs.noise_bound)),
            ),
            correction_factor: ((lhs.correction_factor as u128 * rhs.correction_factor as u128) %
                self.plaintext_modulus as u128) as u64,
        })
    }
    pub fn relinearize(&self, ct: &BgvCiphertext, key: &Mat) -> Result<BgvCiphertext, FheError> {
        let level = self.validate_ciphertext(ct, 3)?;
        let params = self.common.parameters_at(level)?;
        utils::check_matrix(&params, key, 2, params.modulus_digits())?;
        let digits = row(&ct.components, 0)
            .decompose(BigUint::from(1u8) << params.base_bits(), params.modulus_digits());
        // The leading coefficient multiplies s^2; its key encrypts +s^2,
        // so the switched pair is added to the remaining linear polynomial.
        let switched = digits.mul_small_rhs(key.clone());
        Ok(BgvCiphertext {
            components: concat_rows![row(&ct.components, 1), row(&ct.components, 2)] + switched,
            correction_factor: ct.correction_factor,
            noise_bound: &ct.noise_bound + self.key_switch_noise(&params),
        })
    }
    /// Explicitly aligns factors without consuming a level; centered scaling grows noise.
    pub fn match_correction_factor(
        &self,
        ct: &BgvCiphertext,
        target: u64,
    ) -> Result<BgvCiphertext, FheError> {
        self.ciphertext_rows(ct)?;
        if target >= self.plaintext_modulus || mod_inverse(target, self.plaintext_modulus).is_none()
        {
            return Err(FheError::InvalidCorrectionFactor);
        }
        let inverse = mod_inverse(ct.correction_factor, self.plaintext_modulus)
            .ok_or(FheError::InvalidCorrectionFactor)?;
        // Scaling components by k = target/f changes v mod t from f*m to
        // target*m. The logical plaintext stays m, but the noise may grow.
        let scalar = ((target as u128 * inverse as u128) % self.plaintext_modulus as u128) as u64;
        let centered = if scalar > self.plaintext_modulus / 2 {
            BigInt::from(scalar) - BigInt::from(self.plaintext_modulus)
        } else {
            BigInt::from(scalar)
        };
        let params = self.common.parameters_at(self.level_of(&ct.components)?)?;
        Ok(BgvCiphertext {
            components: &ct.components * &utils::scalar(&params, centered),
            correction_factor: target,
            noise_bound: self.noise_from_phase(
                &(self.phase_from_noise(&ct.noise_bound) *
                    BigUint::from(scalar.min(self.plaintext_modulus - scalar))),
            ),
        })
    }
    /// Drops trailing primes with unsigned single-limb corrections, never coefficient extraction.
    pub fn mod_switch_to(
        &self,
        ct: &BgvCiphertext,
        target_level: usize,
    ) -> Result<BgvCiphertext, FheError> {
        self.validate_ciphertext(ct, 2)?;
        let source_level = self.level_of(&ct.components)?;
        if target_level > source_level {
            return Err(FheError::LevelMismatch);
        }
        let mut output = ct.clone();
        for level in (target_level + 1..=source_level).rev() {
            let source = self.common.parameters_at(level)?;
            let dest = self.common.parameters_at(level - 1)?;
            let p = *source.to_crt().0.last().expect("validated nonempty CRT basis");
            let dropped =
                source.select_modulus(&BigUint::from(p)).ok_or(FheError::LevelMismatch)?;
            let inverse_t = mod_inverse(self.plaintext_modulus % p, p)
                .ok_or(FheError::InvalidParameters("noninvertible plaintext modulus"))?;
            // Choose U = -C/t mod p so C + t*U is divisible by the dropped
            // prime. Only this one-limb residue is centered and rebased; the
            // full coefficient modulo Q is never reconstructed as a big integer.
            let u = output.components.clone().reduce_modulus(p) *
                utils::scalar(&dropped, p - inverse_t);
            let correction = u.centered_rebase(dest.modulus().as_ref().clone());
            let inverse_p = mod_inverse_biguints(&BigUint::from(p), dest.modulus().as_ref())
                .ok_or(FheError::InvalidParameters("noninvertible dropped prime"))?;
            let prime = BigUint::from(p);
            // Each correction has norm <= floor(p/2). The phase correction
            // is t*(U_b - s*U_a), bounded by t*floor(p/2)*(1 + N) for |s| <= 1.
            let correction_bound = BigUint::from(self.plaintext_modulus) *
                (&prime / 2u8) *
                (BigUint::from(source.ring_dimension()) + 1u8);
            let phase_bound =
                (self.phase_from_noise(&output.noise_bound) + correction_bound + &prime - 1u8) /
                    &prime;
            let noise_bound = self.noise_from_phase(&phase_bound);
            let components = (output.components.reduce_modulus(dest.modulus().as_ref().clone()) +
                correction * utils::scalar(&dest, self.plaintext_modulus)) *
                utils::scalar(&dest, inverse_p);
            // Division by p also scales the plaintext phase modulo t; retain
            // that public factor so decryption can undo it after later operations.
            let factor = ((output.correction_factor as u128 *
                mod_inverse(p % self.plaintext_modulus, self.plaintext_modulus)
                    .ok_or(FheError::InvalidCorrectionFactor)? as u128) %
                self.plaintext_modulus as u128) as u64;
            output = BgvCiphertext { components, correction_factor: factor, noise_bound };
        }
        Ok(output)
    }
}

impl FheScheme for BgvParams {
    type Plaintext = Family<Int>;
    type Ciphertext = BgvCiphertext;
    type MulRhs = BgvCiphertext;
    type EvaluationKey = Mat;
    fn common_params(&self) -> &FheCommonParams {
        &self.common
    }
    fn keygen(&self) -> Result<(Mat, Mat), FheError> {
        self.common.validate()?;
        let s = self.common.sample_secret();
        let a = self.common.ring().uniform_residue((1, 1));
        let b = &s * &a +
            utils::scalar(&self.common.ring, self.plaintext_modulus) *
                self.common.gaussian(&self.common.ring, 1, 1);
        Ok((s, concat_rows![a, b]))
    }
    fn encrypt(&self, key: &Mat, coefficients: &Family<Int>) -> Result<BgvCiphertext, FheError> {
        utils::check_matrix(&self.common.ring, key, 2, 1)?;
        utils::check_family(coefficients, self.common.ring.ring_dimension() as usize)?;
        let centered = parallel(self.common.ring.ring_dimension(), |i| {
            utils::centered(
                coefficients.at(i).rem(Int::constant(self.plaintext_modulus)),
                &BigUint::from(self.plaintext_modulus),
            )
        })?;
        let message = utils::pack(&self.common.ring, &centered)?;
        let u = self.common.sample_secret();
        let t = utils::scalar(&self.common.ring, self.plaintext_modulus);
        // The phase noise is e_pk*u + e_b - s*e_a. Both secret polynomials
        // have coefficient magnitude <= 1, giving the fresh bound (2N + 1)*cutoff.
        let a = row(key, 0) * &u + &t * self.common.gaussian(&self.common.ring, 1, 1);
        let b = row(key, 1) * u + &t * self.common.gaussian(&self.common.ring, 1, 1) + message;
        Ok(BgvCiphertext {
            components: concat_rows![a, b],
            correction_factor: 1,
            noise_bound: (BigUint::from(self.common.ring.ring_dimension()) * 2u8 + 1u8) *
                &self.common.error_cutoff,
        })
    }
    fn decrypt(&self, secret: &Mat, ct: &BgvCiphertext) -> Result<Family<Int>, FheError> {
        let rows = self.ciphertext_rows(ct)?;
        utils::check_matrix(&self.common.ring, secret, 1, 1)?;
        let params = self.common.parameters_at(self.level_of(&ct.components)?)?;
        let minus_s = -secret.clone().reduce_modulus(params.modulus().as_ref().clone());
        // Horner evaluation supports both ordinary pairs and unrelinearized
        // triples with the same descending-in-minus-s component convention.
        let mut phase = row(&ct.components, 0);
        for index in 1..rows {
            phase = phase * &minus_s + row(&ct.components, index);
        }
        let coefficients = utils::extract(&params, &phase)?;
        // The centered phase reduces to f*m modulo t. Undo the public factor
        // to recover m even after modulus switching has made f different from 1.
        let inverse = mod_inverse(ct.correction_factor, self.plaintext_modulus)
            .ok_or(FheError::InvalidCorrectionFactor)?;
        Ok(parallel(params.ring_dimension(), |i| {
            Ok(utils::centered(coefficients.at(i), params.modulus().as_ref())?
                .mul(Int::constant(inverse))
                .rem(Int::constant(self.plaintext_modulus)))
        })?)
    }
    fn add(&self, lhs: &BgvCiphertext, rhs: &BgvCiphertext) -> Result<BgvCiphertext, FheError> {
        let rows = self.ciphertext_rows(lhs)?;
        if self.validate_ciphertext(lhs, rows)? != self.validate_ciphertext(rhs, rows)? {
            return Err(FheError::LevelMismatch);
        }
        if lhs.correction_factor != rhs.correction_factor {
            return Err(FheError::CorrectionFactorMismatch);
        }
        Ok(BgvCiphertext {
            components: &lhs.components + &rhs.components,
            correction_factor: lhs.correction_factor,
            noise_bound: self.noise_from_phase(
                &(self.phase_from_noise(&lhs.noise_bound) +
                    self.phase_from_noise(&rhs.noise_bound)),
            ),
        })
    }
    fn mul(
        &self,
        lhs: &BgvCiphertext,
        rhs: &BgvCiphertext,
        eval_key: &Mat,
    ) -> Result<BgvCiphertext, FheError> {
        self.relinearize(&self.mul_unrelinearized(lhs, rhs)?, eval_key)
    }
}

// Public modular constants only: no plaintext or ciphertext values are inspected here.
fn pow_mod(mut value: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = (u128::from(result) * u128::from(value) % u128::from(modulus)) as u64;
        }
        value = (u128::from(value) * u128::from(value) % u128::from(modulus)) as u64;
        exponent >>= 1;
    }
    result
}

pub(crate) fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if value % prime == 0 {
            return value == prime;
        }
    }
    let shifts = (value - 1).trailing_zeros();
    let odd = (value - 1) >> shifts;
    // Deterministic Miller-Rabin bases covering every unsigned 64-bit integer.
    [2, 325, 9375, 28178, 450775, 9780504, 1795265022].into_iter().all(|base| {
        if base % value == 0 {
            return true;
        }
        let mut x = pow_mod(base % value, odd, value);
        if x == 1 || x == value - 1 {
            return true;
        }
        for _ in 1..shifts {
            x = (u128::from(x) * u128::from(x) % u128::from(value)) as u64;
            if x == value - 1 {
                return true;
            }
        }
        false
    })
}

impl BgvParams {
    /// Returns the validated primitive `2N`-th root defining the batching layout.
    pub(crate) fn batching_root(&self) -> Result<u64, FheError> {
        let n = u64::from(self.common.ring.ring_dimension());
        let t = self.plaintext_modulus;
        if !is_prime(t) || (t - 1) % (2 * n) != 0 {
            return Err(FheError::InvalidParameters("batching requires prime t = 1 mod 2N"));
        }
        // For power-of-two 2N, zeta^N = -1
        // establishes exact order 2N without factoring t-1.
        (2..t)
            .map(|a| pow_mod(a, (t - 1) / (2 * n), t))
            .find(|&zeta| pow_mod(zeta, n, t) == t - 1)
            .ok_or(FheError::InvalidParameters("no primitive batching root"))
    }

    /// Plaintext NTT ring to register alongside the ciphertext CRT chain.
    pub fn batching_parameters(&self) -> Result<DCRTPolyParams, FheError> {
        self.batching_root()?;
        Ok(DCRTPolyParams::new(
            self.common.ring.ring_dimension(),
            1,
            (64 - self.plaintext_modulus.leading_zeros()) as usize,
            1,
            Some(vec![self.plaintext_modulus]),
            None,
        ))
    }

    // Map logical slots to the native root/bit-reversal order. Only public roots
    // are inspected here; plaintext values are transformed inside the runtime.
    fn batching_indices(&self) -> Result<Vec<usize>, FheError> {
        let params = self.batching_parameters()?;
        let n = params.ring_dimension() as usize;
        let zeta = self.batching_root()?;
        // Evaluating X exposes each native slot root directly. Matching roots
        // avoids duplicating the primitive backend's NTT ordering conventions.
        let native = DCRTPoly::const_rotate_poly(&params, 1)
            .evals_biguints()
            .into_iter()
            .enumerate()
            .map(|(i, root)| (root, i))
            .collect::<std::collections::BTreeMap<_, _>>();
        // The two rows use exponents +5^j and -5^j modulo 2N. Multiplying an
        // exponent by 5 rotates within a row; negating it exchanges the rows.
        Ok((0..n)
            .map(|i| {
                let exponent = pow_mod(5, (i % (n / 2)) as u64, (2 * n) as u64);
                let exponent = if i < n / 2 { exponent } else { 2 * n as u64 - exponent };
                native[&BigUint::from(pow_mod(zeta, exponent, self.plaintext_modulus))]
            })
            .collect())
    }

    /// Encodes row-major slots through the primitive inverse NTT at modulus t.
    pub fn encode_slots(&self, slots: &Family<Int>) -> Result<Family<Int>, FheError> {
        let n = self.common.ring.ring_dimension() as usize;
        check_family(slots, n)?;
        let mut inverse = vec![0; n];
        for (slot, native) in self.batching_indices()?.into_iter().enumerate() {
            inverse[native] = slot;
        }
        let indices = Family::pack(inverse.into_iter().map(Int::constant).collect())?;
        let native = parallel(n, |i| Ok(slots.at(indices.at(i))))?;
        Ok(Ring::new(self.plaintext_modulus, n).from_evaluations(&native).coefficients())
    }

    /// Decodes coefficients through the primitive forward NTT at modulus t.
    pub fn decode_slots(&self, coefficients: &Family<Int>) -> Result<Family<Int>, FheError> {
        let n = self.common.ring.ring_dimension() as usize;
        check_family(coefficients, n)?;
        let indices =
            Family::pack(self.batching_indices()?.into_iter().map(Int::constant).collect())?;
        let native =
            Ring::new(self.plaintext_modulus, n).from_coefficients(coefficients).evaluations();
        Ok(parallel(n, |i| Ok(native.at(indices.at(i))))?)
    }

    /// Generates the key for a row rotation at the specified CRT level.
    pub fn rotation_key(&self, secret: &Mat, level: usize, steps: i32) -> Result<Mat, FheError> {
        self.batching_root()?;
        let index = self.rotation_index(steps);
        let parameters = self.common.parameters_at(level)?;
        check_matrix(&self.common.ring, secret, 1, 1)?;
        let target = secret
            .clone()
            .reduce_modulus(parameters.modulus().as_ref().clone())
            .ring_automorphism(index);
        self.key_switch_key(secret, level, &target)
    }

    /// Generates the automorphism key that swaps the two batching rows.
    pub fn row_swap_key(&self, secret: &Mat, level: usize) -> Result<Mat, FheError> {
        self.batching_root()?;
        let parameters = self.common.parameters_at(level)?;
        check_matrix(&self.common.ring, secret, 1, 1)?;
        let index = 2 * u64::from(self.common.ring.ring_dimension()) - 1;
        let target = secret
            .clone()
            .reduce_modulus(parameters.modulus().as_ref().clone())
            .ring_automorphism(index);
        self.key_switch_key(secret, level, &target)
    }

    /// Positive steps move slot j+steps into slot j within each row.
    /// A zero (including wraparound) rotation accepts no evaluation key.
    pub fn rotate_rows(
        &self,
        key: Option<&Mat>,
        ct: &BgvCiphertext,
        steps: i32,
    ) -> Result<BgvCiphertext, FheError> {
        self.batching_root()?;
        self.validate_ciphertext(ct, 2)?;
        let index = self.rotation_index(steps);
        if index == 1 {
            return Ok(ct.clone());
        }
        self.apply_automorphism(key.ok_or(FheError::MissingEvaluationKey)?, ct, index)
    }

    /// Swaps the batching rows without reversing their columns.
    pub fn swap_rows(&self, key: &Mat, ct: &BgvCiphertext) -> Result<BgvCiphertext, FheError> {
        self.batching_root()?;
        self.apply_automorphism(key, ct, 2 * u64::from(self.common.ring.ring_dimension()) - 1)
    }

    fn rotation_index(&self, steps: i32) -> u64 {
        let n = u64::from(self.common.ring.ring_dimension());
        let normalized = i64::from(steps).rem_euclid((n / 2) as i64) as u64;
        pow_mod(5, normalized, 2 * n)
    }

    fn apply_automorphism(
        &self,
        key: &Mat,
        ct: &BgvCiphertext,
        index: u64,
    ) -> Result<BgvCiphertext, FheError> {
        let level = self.validate_ciphertext(ct, 2)?;
        let parameters = self.common.parameters_at(level)?;
        let digits = parameters.modulus_digits();
        check_matrix(&parameters, key, 2, digits)?;
        // The automorphism changes the decryption secret to sigma(s). The
        // switch key encrypts sigma(s) under s; subtract its phase to recover
        // sigma(b) - sigma(s)*sigma(a), explaining both minus signs below.
        let transformed = ct.components.clone().ring_automorphism(index);
        let a = row(&transformed, 0);
        let b = row(&transformed, 1);
        let switched = a
            .decompose(IntExpr::constant(1u64 << parameters.base_bits()), digits)
            .mul_small_rhs(key.clone());
        let switched_a = row(&switched, 0);
        let switched_b = row(&switched, 1);
        Ok(BgvCiphertext {
            components: Mat::concat(ConcatAxis::Rows, vec![-switched_a, b - switched_b]),
            correction_factor: ct.correction_factor,
            noise_bound: &ct.noise_bound + self.key_switch_noise(&parameters),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{common, execute_graph, int_input, integers};
    use mxx_dsl::DslContext;
    use num_integer::Integer;
    use num_traits::Signed;
    use std::collections::BTreeMap;

    #[test]
    fn test_bgv_phase_bounds_cover_noisy_runtime_composition() {
        let common = common();
        let bgv = BgvParams::new(common.clone(), 17).unwrap();
        let top = common.ring.to_crt().2 - 1;
        assert!(top >= 1, "phase bound test needs at least two CRT limbs");
        let n = common.ring.ring_dimension() as usize;
        let context = DslContext::new("fhe-bgv-phase-bounds");
        let input = context.int_family_input("message", n);
        let (secret, key) = bgv.keygen().unwrap();
        let fresh = bgv.encrypt(&key, &input).unwrap();
        let square = bgv.mul_unrelinearized(&fresh, &fresh).unwrap();
        let relin_key = bgv.relinearization_key(&secret, top).unwrap();
        let relin = bgv.relinearize(&square, &relin_key).unwrap();
        let switched = bgv.mod_switch_to(&relin, top - 1).unwrap();
        let aligned = bgv.match_correction_factor(&switched, 1).unwrap();
        let sum = bgv.add(&aligned, &aligned).unwrap();
        let rotation_key = bgv.rotation_key(&secret, top - 1, 1).unwrap();
        let rotated = bgv.rotate_rows(Some(&rotation_key), &sum, 1).unwrap();
        let cases = [
            ("fresh", fresh, top, 2),
            ("quadratic", square, top, 3),
            ("relinearized", relin, top, 2),
            ("modswitched", switched, top - 1, 2),
            ("aligned", aligned, top - 1, 2),
            ("sum", sum, top - 1, 2),
            ("rotated", rotated, top - 1, 2),
        ];
        let mut context = context;
        for (name, ct, level, rows) in &cases {
            assert!(bgv.can_decrypt(ct).unwrap(), "insufficient test modulus for {name}");
            let parameters = common.parameters_at(*level).unwrap();
            let minus_secret =
                -secret.clone().reduce_modulus(parameters.modulus().as_ref().clone());
            let mut phase = row(&ct.components, 0);
            for index in 1..*rows {
                phase = phase * &minus_secret + row(&ct.components, index);
            }
            let coefficients = utils::extract(&parameters, &phase).unwrap();
            let centered =
                parallel(n, |i| utils::centered(coefficients.at(i), parameters.modulus().as_ref()))
                    .unwrap();
            context = context.private_output(*name, centered).unwrap();
            context = context
                .private_output(format!("decoded-{name}"), bgv.decrypt(&secret, ct).unwrap())
                .unwrap();
        }
        // Squaring 3 gives 9, outside the centered interval for t = 17. This
        // exercises plaintext carries in addition to sampled Gaussian errors,
        // including the representative changes accounted for by the bound.
        let mut message = vec![0; n];
        message[0] = 3;
        let result = execute_graph(
            context.build().unwrap(),
            &common,
            BTreeMap::from([("message".into(), int_input(&message))]),
        );
        for (name, ct, _, _) in cases {
            // Recover the actual e from the runtime phase, independently of
            // the bound formulas, then also check the decoded semantic result.
            for coefficient in integers(&result, name) {
                let t = BigInt::from(bgv.plaintext_modulus);
                let residue = coefficient.mod_floor(&t);
                let centered = if residue > &t / 2 { residue - &t } else { residue };
                let noise = (coefficient - centered) / &t;
                assert!(
                    noise.abs().to_biguint().unwrap() <= ct.noise_bound,
                    "coefficient noise exceeds tracked bound for {name}"
                );
            }
            let mut expected = vec![BigInt::from(0); n];
            expected[0] = BigInt::from(if name == "fresh" {
                3
            } else if name == "sum" || name == "rotated" {
                1
            } else {
                9
            });
            assert_eq!(integers(&result, &format!("decoded-{name}")), expected, "{name}");
        }
        let excessive = BgvCiphertext {
            components: common.ring().zero((2, 1)),
            correction_factor: 1,
            noise_bound: common.ring.modulus().as_ref().clone(),
        };
        assert!(!bgv.can_decrypt(&excessive).unwrap());
    }

    #[test]
    fn test_bgv_noisy_arithmetic_and_factor_alignment() {
        let common = common();
        let n = common.ring.ring_dimension() as usize;
        let top = common.ring.to_crt().2 - 1;
        assert!(top >= 2, "BGV test requires at least three CRT limbs");
        let bgv = BgvParams::new(common.clone(), 17).unwrap();
        let context = DslContext::new("fhe-bgv-noisy-arithmetic");
        let input = context.int_family_input("coefficients", n);
        let (secret, pk) = bgv.keygen().unwrap();
        let ct = bgv.encrypt(&pk, &input).unwrap();
        let square = bgv.mul_unrelinearized(&ct, &ct).unwrap();
        let key = bgv.relinearization_key(&secret, top).unwrap();
        let relin = bgv.relinearize(&square, &key).unwrap();
        let once = bgv.mod_switch_to(&relin, top - 1).unwrap();
        let twice = bgv.mod_switch_to(&once, top - 2).unwrap();
        let aligned = bgv.match_correction_factor(&once, 1).unwrap();
        let sum = bgv.add(&aligned, &aligned).unwrap();
        let lower_key = bgv.relinearization_key(&secret, top - 1).unwrap();
        let low_input = bgv.mod_switch_to(&ct, top - 1).unwrap();
        let lower_square = bgv.mul(&low_input, &low_input, &lower_key).unwrap();
        let graph = context
            .private_output("roundtrip", bgv.decrypt(&secret, &ct).unwrap())
            .unwrap()
            .private_output("quadratic", bgv.decrypt(&secret, &square).unwrap())
            .unwrap()
            .private_output("relinearized", bgv.decrypt(&secret, &relin).unwrap())
            .unwrap()
            .private_output("once", bgv.decrypt(&secret, &once).unwrap())
            .unwrap()
            .private_output("twice", bgv.decrypt(&secret, &twice).unwrap())
            .unwrap()
            .private_output("aligned_sum", bgv.decrypt(&secret, &sum).unwrap())
            .unwrap()
            .private_output("lower_square", bgv.decrypt(&secret, &lower_square).unwrap())
            .unwrap()
            .build()
            .unwrap();
        // Constant polynomial multiplication has a trusted scalar expectation;
        // SIMD tests separately exercise every nonconstant coefficient and slot.
        let mut message = vec![0i64; n];
        message[0] = 3;
        let result = execute_graph(
            graph,
            &common,
            BTreeMap::from([("coefficients".to_owned(), int_input(&message))]),
        );
        assert_eq!(
            integers(&result, "roundtrip"),
            message.iter().copied().map(BigInt::from).collect::<Vec<_>>()
        );
        for name in ["quadratic", "relinearized", "once", "twice", "lower_square", "aligned_sum"] {
            let mut expected = vec![BigInt::from(0); n];
            expected[0] = BigInt::from(if name == "aligned_sum" { 1 } else { 9 });
            assert_eq!(integers(&result, name), expected, "{name}");
        }
        assert!(
            common.ring.to_crt().0.iter().any(|p| p % 17 != 1),
            "factor test needs a nontrivial removed-prime inverse"
        );
    }

    #[test]
    fn test_bgv_rejects_invalid_metadata_and_shapes() {
        let common = common();
        let bgv = BgvParams::new(common.clone(), 17).unwrap();
        let ct = BgvCiphertext {
            components: common.ring().zero((2, 1)),
            correction_factor: 1,
            noise_bound: BigUint::from(0u8),
        };
        let malformed = BgvCiphertext {
            components: common.ring().zero((4, 1)),
            correction_factor: 1,
            noise_bound: BigUint::from(0u8),
        };
        assert!(bgv.add(&ct, &malformed).is_err());
        let wrong_factor = BgvCiphertext {
            components: ct.components.clone(),
            correction_factor: 2,
            noise_bound: BigUint::from(0u8),
        };
        assert!(matches!(bgv.add(&ct, &wrong_factor), Err(FheError::CorrectionFactorMismatch)));
        assert!(bgv.match_correction_factor(&ct, 0).is_err());
        assert!(bgv.match_correction_factor(&ct, 17).is_err());
        assert!(bgv.mod_switch_to(&ct, common.ring.to_crt().2).is_err());
        let quadratic = bgv.mul_unrelinearized(&ct, &ct).unwrap();
        assert!(bgv.relinearize(&quadratic, &common.ring().zero((2, 1))).is_err());
        let wrong_modulus = BgvCiphertext {
            components: Ring::new(97, common.ring.ring_dimension()).zero((2, 1)),
            correction_factor: 1,
            noise_bound: BigUint::from(0u8),
        };
        assert!(bgv.add(&ct, &wrong_modulus).is_err());
        assert!(BgvParams::new(common.clone(), common.ring.to_crt().0[0]).is_err());
        assert!(BgvParams::new(common, 1).is_err());
    }
    #[test]
    fn test_bgv_staged_evaluator_without_secret_input() {
        use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality};
        use mxx_runtime::{
            RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
            transcript::SamplingMode,
        };
        use num_traits::ToPrimitive;
        let common = common();
        let n = common.ring.ring_dimension() as usize;
        let bgv = BgvParams::new(common.clone(), 17).unwrap();
        let top = common.ring.to_crt().2 - 1;
        let mut parameters =
            (0..=top).map(|level| common.parameters_at(level).unwrap()).collect::<Vec<_>>();
        parameters.extend(
            common
                .ring
                .to_crt()
                .0
                .iter()
                .map(|&p| common.ring.select_modulus(&BigUint::from(p)).unwrap()),
        );
        let mut backend = cpu_backend(parameters);
        // Distinct graphs exchange manifests and in-memory artifacts, exercising
        // the protocol boundary without serializing secrets or ciphertexts to files.
        let mut store = MemoryArtifactStore::default();
        let env = ParamEnv::default();
        let context = DslContext::new("fhe-bgv-staged-encryption");
        let input = context.int_family_input("coefficients", n);
        let (secret, pk) = bgv.keygen().unwrap();
        let ct = bgv.encrypt(&pk, &input).unwrap();
        let key = bgv.relinearization_key(&secret, top).unwrap();
        // Matrix artifacts contain no Rust schema metadata. Export the public
        // bound explicitly and rebuild the ciphertext record in the next stage.
        let encryption_noise = ct.noise_bound.clone();
        let encryption = context
            .private_output("secret", secret)
            .unwrap()
            .public_output("ciphertext", ct.components)
            .unwrap()
            .public_output("noise", Int::constant(encryption_noise.clone()))
            .unwrap()
            .public_output("evaluation_key", key)
            .unwrap()
            .build()
            .unwrap()
            .validate(&env)
            .unwrap();
        let mut message = vec![0i64; n];
        message[0] = 3;
        let encrypted = execute(
            &encryption,
            &mut backend,
            BTreeMap::from([("coefficients".into(), int_input(&message))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Int(exported_encryption_noise) = &encrypted.outputs["noise"] else {
            panic!("public encryption noise bound integer")
        };
        assert_eq!(exported_encryption_noise.to_biguint(), Some(encryption_noise));
        let encryption_id = encrypted.production_id.unwrap();
        let mut manifests = BTreeMap::from([(
            encryption_id.clone(),
            store.manifest(&encryption_id).unwrap().clone(),
        )]);
        let input_ct = BgvCiphertext {
            components: common.ring().artifact_input(
                encryption_id.clone(),
                "ciphertext",
                (2, 1),
                ArtifactConfidentiality::Public,
            ),
            correction_factor: 1,
            noise_bound: exported_encryption_noise.to_biguint().unwrap(),
        };
        let input_key = common.ring().artifact_input(
            encryption_id.clone(),
            "evaluation_key",
            (2, common.ring.modulus_digits()),
            ArtifactConfidentiality::Public,
        );
        let evaluated = bgv
            .mod_switch_to(&bgv.mul(&input_ct, &input_ct, &input_key).unwrap(), top - 1)
            .unwrap();
        let factor = evaluated.correction_factor;
        let evaluation_noise = evaluated.noise_bound.clone();
        let evaluator = DslContext::new("fhe-bgv-staged-evaluator")
            .public_output("evaluated", evaluated.components)
            .unwrap()
            .public_output("factor", Int::constant(factor))
            .unwrap()
            .public_output("noise", Int::constant(evaluation_noise.clone()))
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(&env, &manifests)
            .unwrap();
        // The evaluator imports only the ciphertext and public evaluation key.
        let evaluation =
            execute(&evaluator, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        let RuntimeValue::Int(exported_factor) = &evaluation.outputs["factor"] else {
            panic!("public factor integer")
        };
        assert_eq!(exported_factor.to_u64(), Some(factor));
        let RuntimeValue::Int(exported_noise) = &evaluation.outputs["noise"] else {
            panic!("public noise bound integer")
        };
        assert_eq!(exported_noise.to_biguint(), Some(evaluation_noise));
        let evaluation_id = evaluation.production_id.unwrap();
        manifests.insert(evaluation_id.clone(), store.manifest(&evaluation_id).unwrap().clone());
        let lower = common.parameters_at(top - 1).unwrap();
        let lower_ring = Ring::new(lower.modulus().as_ref().clone(), n);
        let imported = BgvCiphertext {
            components: lower_ring.artifact_input(
                evaluation_id.clone(),
                "evaluated",
                (2, 1),
                ArtifactConfidentiality::Public,
            ),
            correction_factor: exported_factor.to_u64().unwrap(),
            noise_bound: exported_noise.to_biguint().unwrap(),
        };
        let sk = common.ring().artifact_input(
            encryption_id,
            "secret",
            (1, 1),
            ArtifactConfidentiality::Private,
        );
        let decryption = DslContext::new("fhe-bgv-staged-decryption")
            .private_output("plaintext", bgv.decrypt(&sk, &imported).unwrap())
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(&env, &manifests)
            .unwrap();
        let mut result =
            execute(&decryption, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        result.materialize_output("plaintext", &backend, &mut store).unwrap();
        let mut expected = vec![BigInt::from(0); n];
        expected[0] = BigInt::from(9);
        assert_eq!(integers(&result, "plaintext"), expected);
        let malformed = lower_ring.artifact_input(
            evaluation_id,
            "evaluated",
            (3, 1),
            ArtifactConfidentiality::Public,
        );
        assert!(
            DslContext::new("fhe-bgv-invalid-artifact")
                .output("value", malformed)
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(&env, &manifests)
                .is_err()
        );
    }

    #[test]
    fn test_bgv_wide_rns_modswitch_no_coefficient_nodes() {
        use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
        let mut common = common();
        let depth = std::env::var("FHE_TEST_WIDE_CRT_DEPTH")
            .ok()
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(5);
        let (_, bits, _) = common.ring.to_crt();
        common.ring = DCRTPolyParams::new(
            common.ring.ring_dimension(),
            depth,
            bits,
            common.ring.base_bits(),
            None,
            None,
        );
        assert!(common.ring.modulus().bits() > 128, "wide RNS test requires Q > 128 bits");
        let bgv = BgvParams::new(common.clone(), 17).unwrap();
        let ct = BgvCiphertext {
            components: common.ring().input("ciphertext", (2, 1)),
            correction_factor: 1,
            noise_bound: bgv.common.ring.modulus().as_ref() / (2 * bgv.plaintext_modulus) + 1u8,
        };
        let switched = bgv.mod_switch_to(&ct, 0).unwrap();
        let isolated = DslContext::new("fhe-bgv-rns-graph-audit")
            .output("switched", switched.components)
            .unwrap()
            .build()
            .unwrap();
        let encoded = serde_json::to_string(&isolated.graph).unwrap();
        for forbidden in [
            "ExtractCoefficient",
            "PackPolynomialCoefficients",
            "PolynomialFromValues",
            "PolynomialValues",
            "IntBinary",
            "BitExtract",
            "LiftIntegerToConstantPolynomial",
        ] {
            assert!(!encoded.contains(forbidden), "modswitch graph contains {forbidden}");
        }
        let n = common.ring.ring_dimension() as usize;
        let context = DslContext::new("fhe-bgv-wide-rns-roundtrip");
        let input = context.int_family_input("coefficients", n);
        let (secret, key) = bgv.keygen().unwrap();
        let encrypted = bgv.encrypt(&key, &input).unwrap();
        let switched = bgv.mod_switch_to(&encrypted, 0).unwrap();
        let graph = context
            .private_output("plaintext", bgv.decrypt(&secret, &switched).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let message = (0..n).map(|i| (i % 17) as i64).collect::<Vec<_>>();
        let result = execute_graph(
            graph,
            &common,
            BTreeMap::from([("coefficients".into(), int_input(&message))]),
        );
        assert_eq!(
            integers(&result, "plaintext"),
            message.into_iter().map(BigInt::from).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_bgv_wide_rns_modswitch_matches_native_mod_reduce() {
        use mxx_primitives::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
            poly::{
                Poly,
                dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
            },
        };
        use mxx_runtime::RuntimeValue;
        use std::sync::Arc;

        let mut common = common();
        let depth = std::env::var("FHE_TEST_WIDE_CRT_DEPTH")
            .ok()
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(5);
        let (_, bits, _) = common.ring.to_crt();
        let n = common.ring.ring_dimension();
        assert!(n >= 4 && depth >= 2);
        let mut primes = (0..depth)
            .map(|i| {
                DCRTPolyParams::new(n, depth, bits - i % 2, common.ring.base_bits(), None, None)
                    .to_crt()
                    .0[i]
            })
            .collect::<Vec<_>>();
        // Mixed prime widths and a permuted basis catch assumptions that CRT
        // towers are sorted or share a width; the default Q also exceeds 128 bits.
        for reordered in [false, true] {
            if reordered {
                primes.reverse();
                primes.rotate_left(1);
            }
            common.ring = DCRTPolyParams::new(
                n,
                depth,
                bits,
                common.ring.base_bits(),
                Some(primes.clone()),
                None,
            );
            assert!(common.ring.modulus().bits() > 128, "oracle test requires Q > 128 bits");
            let bgv = BgvParams::new(common.clone(), 17).unwrap();
            let p = *primes.last().unwrap();
            // Force the correction -c/t mod p to lie at both center boundaries,
            // and at zero and p-1. Large lifts exercise the complete CRT value.
            let corrections = [0, p / 2, p / 2 + 1, p - 1];
            let polys = (0..2)
                .map(|row| {
                    let values =
                        (0..n as usize)
                            .map(|i| {
                                let u = corrections[(i + row) % corrections.len()];
                                let residue = ((p - u) as u128 * 17 % p as u128) as u64;
                                let lift = (common.ring.modulus().as_ref() / p) *
                                    rand::random::<u64>() /
                                    u64::MAX;
                                lift * p + residue
                            })
                            .collect::<Vec<_>>();
                    DCRTPoly::from_biguints(&common.ring, &values)
                })
                .collect::<Vec<_>>();
            let input = DCRTPolyMatrix::from_poly_vec(
                &common.ring,
                polys.iter().cloned().map(|p| vec![p]).collect(),
            );
            let ct = BgvCiphertext {
                components: common.ring().input("ciphertext", (2, 1)),
                correction_factor: 1,
                noise_bound: bgv.common.ring.modulus().as_ref() / (2 * bgv.plaintext_modulus) + 1u8,
            };
            let mut context = DslContext::new("fhe-bgv-native-modreduce-oracle");
            for level in (0..depth - 1).rev() {
                context = context
                    .output(
                        format!("level_{level}"),
                        bgv.mod_switch_to(&ct, level).unwrap().components,
                    )
                    .unwrap();
            }
            let result = execute_graph(
                context.build().unwrap(),
                &common,
                BTreeMap::from([("ciphertext".into(), RuntimeValue::Matrix(Arc::new(input)))]),
            );
            // Compare each complete DSL reduction against the native primitive
            // applied one level at a time, rather than reimplementing its formula.
            let mut expected = polys;
            for level in (0..depth - 1).rev() {
                expected = expected.iter().map(|p| p.bgv_mod_reduce(17).unwrap()).collect();
                let RuntimeValue::Matrix(actual) = &result.outputs[&format!("level_{level}")]
                else {
                    panic!("matrix output")
                };
                assert_eq!(actual.params(), &common.parameters_at(level).unwrap());
                for (row, oracle) in expected.iter().enumerate() {
                    // Native polynomial equality checks the format, ordered tower
                    // parameters, and every evaluation residue (no interpolation).
                    assert_eq!(
                        &actual.entry(row, 0),
                        oracle,
                        "reordered={reordered}, level={level}, row={row}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use crate::{
        FheScheme,
        utils::{common, execute_graph, int_input, integers},
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_primitives::poly::{
        Poly,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    fn parameters() -> BgvParams {
        let common = common();
        let order = 2 * u64::from(common.ring.ring_dimension());
        let t = (1..).map(|k| k * order + 1).find(|&t| is_prime(t)).unwrap();
        BgvParams::new(common, t).unwrap()
    }

    #[test]
    fn test_simd_encoding_runtime_and_native_evaluation() {
        let bgv = parameters();
        let n = bgv.common.ring.ring_dimension() as usize;
        let t = bgv.plaintext_modulus;
        let slots = (0..n).map(|i| i as i64).collect::<Vec<_>>();
        let context = DslContext::new("fhe-simd-encoding");
        let input = context.int_family_input("slots", n);
        let coefficients = bgv.encode_slots(&input).unwrap();
        let decoded = bgv.decode_slots(&coefficients).unwrap();
        let graph = context
            .private_output("coefficients", coefficients)
            .unwrap()
            .private_output("slots", decoded)
            .unwrap()
            .build()
            .unwrap();
        let result = execute_graph(
            graph,
            &bgv.common,
            BTreeMap::from([("slots".into(), int_input(&slots))]),
        );
        assert_eq!(
            integers(&result, "slots"),
            slots.iter().copied().map(BigInt::from).collect::<Vec<_>>()
        );

        // OpenFHE exposes its own primitive-root and bit-reversed evaluation
        // order through the polynomial X. This adapter makes no root-order
        // assumption and checks the encoding against its established NTT.
        let plain_ring = DCRTPolyParams::new(
            n as u32,
            1,
            (64 - t.leading_zeros()) as usize,
            1,
            Some(vec![t]),
            None,
        );
        let native_roots = DCRTPoly::const_rotate_poly(&plain_ring, 1).eval_slots();
        let values = integers(&result, "coefficients")
            .into_iter()
            .map(|x| x.to_biguint().unwrap())
            .collect::<Vec<_>>();
        let native = DCRTPoly::from_biguints(&plain_ring, &values).eval_slots();
        let zeta = bgv.batching_root().unwrap();
        for (i, value) in slots.iter().enumerate() {
            let exponent = pow_mod(5, (i % (n / 2)) as u64, (2 * n) as u64);
            let exponent = if i < n / 2 { exponent } else { 2 * n as u64 - exponent };
            let root = BigUint::from(pow_mod(zeta, exponent, t));
            let native_index = native_roots.iter().position(|r| r == &root).unwrap();
            assert_eq!(native[native_index], BigUint::from(*value as u64));
        }
    }

    #[test]
    fn test_simd_runtime_arithmetic_rotations_and_pipeline() {
        let bgv = parameters();
        let n = bgv.common.ring.ring_dimension() as usize;
        let t = bgv.plaintext_modulus as i64;
        let top = bgv.common.ring.crt_depth() - 1;
        assert!(top > 0, "SIMD pipeline test requires at least two CRT limbs");
        let context = DslContext::new("fhe-simd-pipeline");
        let input = context.int_family_input("slots", n);
        let (secret, public) = bgv.keygen().unwrap();
        let ct = bgv.encrypt(&public, &bgv.encode_slots(&input).unwrap()).unwrap();
        let relin = bgv.relinearization_key(&secret, top).unwrap();
        let sum = bgv.add(&ct, &ct).unwrap();
        let product = bgv.mul(&ct, &ct, &relin).unwrap();
        let mut context = context
            .private_output("sum", bgv.decode_slots(&bgv.decrypt(&secret, &sum).unwrap()).unwrap())
            .unwrap()
            .private_output(
                "product",
                bgv.decode_slots(&bgv.decrypt(&secret, &product).unwrap()).unwrap(),
            )
            .unwrap();
        // Include negative and wrapped steps, plus both identity encodings;
        // identity rotations deliberately omit a key to test the no-op contract.
        let steps = [1, -1, 0, n as i32 / 2 + 1, n as i32 / 2];
        for (i, step) in steps.into_iter().enumerate() {
            let key = if step.rem_euclid(n as i32 / 2) == 0 {
                None
            } else {
                Some(bgv.rotation_key(&secret, top, step).unwrap())
            };
            let rotated = bgv.rotate_rows(key.as_ref(), &ct, step).unwrap();
            context = context
                .private_output(
                    format!("rotate{i}"),
                    bgv.decode_slots(&bgv.decrypt(&secret, &rotated).unwrap()).unwrap(),
                )
                .unwrap();
        }
        let swap_key = bgv.row_swap_key(&secret, top).unwrap();
        let swapped = bgv.swap_rows(&swap_key, &ct).unwrap();
        context = context
            .private_output(
                "swap",
                bgv.decode_slots(&bgv.decrypt(&secret, &swapped).unwrap()).unwrap(),
            )
            .unwrap();
        // Rotation after multiplication and level reduction must use a key at
        // the new level while preserving the nontrivial correction factor.
        let reduced = bgv.mod_switch_to(&product, top - 1).unwrap();
        let low_key = bgv.rotation_key(&secret, top - 1, 1).unwrap();
        let pipeline = bgv.rotate_rows(Some(&low_key), &reduced, 1).unwrap();
        context = context
            .private_output(
                "pipeline",
                bgv.decode_slots(&bgv.decrypt(&secret, &pipeline).unwrap()).unwrap(),
            )
            .unwrap();
        let slots = (0..n).map(|i| i as i64).collect::<Vec<_>>();
        let result = execute_graph(
            context.build().unwrap(),
            &bgv.common,
            BTreeMap::from([("slots".into(), int_input(&slots))]),
        );
        let assert_output = |name: &str, expected: Vec<i64>| {
            assert_eq!(
                integers(&result, name),
                expected.into_iter().map(BigInt::from).collect::<Vec<_>>(),
                "{name}"
            )
        };
        assert_output("sum", slots.iter().map(|x| (2 * x) % t).collect());
        assert_output("product", slots.iter().map(|x| (x * x) % t).collect());
        for (i, step) in steps.into_iter().enumerate() {
            let shift = step.rem_euclid(n as i32 / 2) as usize;
            assert_output(
                &format!("rotate{i}"),
                (0..n).map(|j| slots[(j / (n / 2)) * (n / 2) + (j + shift) % (n / 2)]).collect(),
            );
        }
        assert_output("swap", (0..n).map(|j| slots[(j + n / 2) % n]).collect());
        assert_output(
            "pipeline",
            (0..n)
                .map(|j| {
                    let x = slots[(j / (n / 2)) * (n / 2) + (j + 1) % (n / 2)];
                    x * x % t
                })
                .collect(),
        );
    }

    #[test]
    fn test_simd_rejects_invalid_parameters_shapes_and_missing_keys() {
        let bgv = parameters();
        let n = bgv.common.ring.ring_dimension() as usize;
        let context = DslContext::new("fhe-simd-invalid");
        assert!(bgv.encode_slots(&context.int_family_input("short", n - 1)).is_err());
        let bad = BgvParams::new(bgv.common.clone(), 2).unwrap();
        assert!(bad.encode_slots(&context.int_family_input("slots", n)).is_err());
        assert!(!is_prime(341550071728321));
        let t = bgv.plaintext_modulus;
        let bad = BgvParams::new(bgv.common.clone(), t * t).unwrap();
        assert!(bad.batching_root().is_err());
        let ring = Ring::new(bgv.common.ring.modulus().as_ref().clone(), n);
        let ct = BgvCiphertext {
            components: ring.input("ct", (2, 1)),
            correction_factor: 1,
            noise_bound: bgv.common.ring.modulus().as_ref() / (2 * bgv.plaintext_modulus) + 1u8,
        };
        if n > 2 {
            assert!(matches!(bgv.rotate_rows(None, &ct, 1), Err(FheError::MissingEvaluationKey)));
            assert!(bgv.rotate_rows(Some(&ring.input("bad-key", (1, 1))), &ct, 1).is_err());
        }
        assert!(bgv.rotate_rows(None, &ct, 0).is_ok());
    }
}
