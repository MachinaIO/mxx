use crate::{
    FheCommonParams, FheError,
    utils::{check_family, check_matrix},
};
use mxx_dsl::{
    Bytes, DslContext, DslError, Family, FamilyType, GraphValue, GraphValueSchema, HashTag, Int,
    IntType, Mat, MatType, Ring, iterate, parallel, select,
};
use mxx_ir_core::{IntExpr, RealExpr, ValueHandle, node::ConcatAxis, types::WireType};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

/// Polynomial `a`/`b` matrices used for ring-LWE `(1,1)` values and ring-GSW
/// `(1,2L)` evaluation-key entries. Ring-LWE phase is `b - secret * a`.
///
/// Bounds are compile-time diagnostic annotations, not runtime wires. They do
/// not inspect ciphertext or key values; callers must carry suitable metadata
/// when rebinding resident values between separately built graphs.
#[derive(Clone)]
pub struct RingCiphertext {
    pub a: Mat,
    pub b: Mat,
    pub noise_bound: BigUint,
    pub plaintext_bound: BigUint,
}

#[derive(Clone, PartialEq)]
pub struct RingCiphertextSchema {
    pub a: MatType,
    pub b: MatType,
    pub noise_bound: BigUint,
    pub plaintext_bound: BigUint,
}

impl GraphValue for RingCiphertext {
    type Schema = RingCiphertextSchema;

    fn flatten(&self) -> Vec<ValueHandle> {
        (self.a.clone(), self.b.clone()).flatten()
    }

    fn schema(&self) -> Self::Schema {
        RingCiphertextSchema {
            a: self.a.schema(),
            b: self.b.schema(),
            noise_bound: self.noise_bound.clone(),
            plaintext_bound: self.plaintext_bound.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (a, b) = <(Mat, Mat)>::from_values(&(schema.a.clone(), schema.b.clone()), values)?;
        Ok(Self {
            a,
            b,
            noise_bound: schema.noise_bound.clone(),
            plaintext_bound: schema.plaintext_bound.clone(),
        })
    }
}

impl GraphValueSchema for RingCiphertextSchema {
    type Value = RingCiphertext;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RingCiphertext {
            a: self.a.placeholders_from(next),
            b: self.b.placeholders_from(next),
            noise_bound: self.noise_bound.clone(),
            plaintext_bound: self.plaintext_bound.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        (self.a.clone(), self.b.clone()).wire_types()
    }
}

/// Integer LWE ciphertext with phase `b - <a, secret>` modulo `modulus`.
///
/// Boolean messages use `+floor(q/8)` for true and `-floor(q/8)` for false.
/// `noise_bound` is caller-maintained static metadata. It is not attached to or
/// checked against runtime values passed between separately built graphs.
#[derive(Clone)]
pub struct LweCiphertext {
    pub a: Family<Int>,
    pub b: Int,
    pub dimension: usize,
    pub modulus: BigUint,
    pub noise_bound: BigUint,
}

#[derive(Clone, PartialEq)]
pub struct LweCiphertextSchema {
    pub a: FamilyType<IntType>,
    pub dimension: usize,
    pub modulus: BigUint,
    pub noise_bound: BigUint,
}

impl LweCiphertext {
    /// Adds ciphertexts under the same key and modulus.
    pub fn add(&self, rhs: &Self) -> Result<Self, FheError> {
        self.check_compatible(rhs)?;
        let modulus = Int::constant(BigInt::from(self.modulus.clone()));
        let a = parallel(self.dimension, |index| {
            Ok(self.a.at(index.clone()).add(rhs.a.at(index)).rem(modulus.clone()))
        })?;
        Ok(Self {
            a,
            b: self.b.clone().add(rhs.b.clone()).rem(modulus),
            dimension: self.dimension,
            modulus: self.modulus.clone(),
            noise_bound: &self.noise_bound + &rhs.noise_bound,
        })
    }

    /// Subtracts ciphertexts under the same key and modulus.
    pub fn sub(&self, rhs: &Self) -> Result<Self, FheError> {
        self.add(&rhs.negate()?)
    }

    /// Adds a public integer to the phase.
    pub fn add_plain(&self, value: impl Into<BigInt>) -> Result<Self, FheError> {
        let modulus = Int::constant(BigInt::from(self.modulus.clone()));
        Ok(Self {
            a: self.a.clone(),
            b: self.b.clone().add(Int::constant(value.into())).rem(modulus),
            dimension: self.dimension,
            modulus: self.modulus.clone(),
            noise_bound: self.noise_bound.clone(),
        })
    }

    /// Negates the phase and every LWE vector coordinate modulo q.
    pub fn negate(&self) -> Result<Self, FheError> {
        let modulus = Int::constant(BigInt::from(self.modulus.clone()));
        let a = parallel(self.dimension, |index| {
            Ok(Int::constant(0).sub(self.a.at(index)).rem(modulus.clone()))
        })?;
        Ok(Self {
            a,
            b: Int::constant(0).sub(self.b.clone()).rem(modulus),
            dimension: self.dimension,
            modulus: self.modulus.clone(),
            noise_bound: self.noise_bound.clone(),
        })
    }

    fn check_compatible(&self, rhs: &Self) -> Result<(), FheError> {
        if self.dimension != rhs.dimension {
            return Err(FheError::ShapeMismatch);
        }
        if self.modulus != rhs.modulus {
            return Err(FheError::LevelMismatch);
        }
        check_family(&self.a, self.dimension)?;
        check_family(&rhs.a, rhs.dimension)
    }
}

impl GraphValue for LweCiphertext {
    type Schema = LweCiphertextSchema;

    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.a.flatten();
        values.extend(self.b.flatten());
        values
    }

    fn schema(&self) -> Self::Schema {
        LweCiphertextSchema {
            a: self.a.schema(),
            dimension: self.dimension,
            modulus: self.modulus.clone(),
            noise_bound: self.noise_bound.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        if values.len() != 2 {
            return Err(DslError::Schema);
        }
        Ok(Self {
            a: Family::<Int>::from_values(&schema.a, &values[..1])?,
            b: Int::from_values(&IntType, &values[1..])?,
            dimension: schema.dimension,
            modulus: schema.modulus.clone(),
            noise_bound: schema.noise_bound.clone(),
        })
    }
}

impl GraphValueSchema for LweCiphertextSchema {
    type Value = LweCiphertext;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        LweCiphertext {
            a: self.a.placeholders_from(next),
            b: IntType.placeholders_from(next),
            dimension: self.dimension,
            modulus: self.modulus.clone(),
            noise_bound: self.noise_bound.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        let mut types = self.a.wire_types();
        types.extend(IntType.wire_types());
        types
    }
}

/// GSW encryptions of every coefficient of the integer LWE secret.
#[derive(Clone)]
pub struct BootstrappingKey {
    pub entries: Family<RingCiphertext>,
    pub lwe_dimension: usize,
    pub ring_modulus: BigUint,
}

#[derive(Clone, PartialEq)]
pub struct BootstrappingKeySchema {
    pub entries: FamilyType<RingCiphertextSchema>,
    pub lwe_dimension: usize,
    pub ring_modulus: BigUint,
}

impl GraphValue for BootstrappingKey {
    type Schema = BootstrappingKeySchema;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.entries.flatten()
    }

    fn schema(&self) -> Self::Schema {
        BootstrappingKeySchema {
            entries: self.entries.schema(),
            lwe_dimension: self.lwe_dimension,
            ring_modulus: self.ring_modulus.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            entries: Family::<RingCiphertext>::from_values(&schema.entries, values)?,
            lwe_dimension: schema.lwe_dimension,
            ring_modulus: schema.ring_modulus.clone(),
        })
    }
}

impl GraphValueSchema for BootstrappingKeySchema {
    type Value = BootstrappingKey;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        BootstrappingKey {
            entries: self.entries.placeholders_from(next),
            lwe_dimension: self.lwe_dimension,
            ring_modulus: self.ring_modulus.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.entries.wire_types()
    }
}

/// Flat key-switch material. `a_values` contains one contiguous LWE vector for
/// every ring-secret coefficient and base digit; `b_values` stores the matching
/// scalar components.
#[derive(Clone)]
pub struct KeySwitchKey {
    pub a_values: Family<Int>,
    pub b_values: Family<Int>,
    pub ring_dimension: usize,
    pub lwe_dimension: usize,
    pub base_bits: usize,
    pub digit_count: usize,
    pub modulus: BigUint,
    pub noise_bound: BigUint,
}

#[derive(Clone, PartialEq)]
pub struct KeySwitchKeySchema {
    pub a_values: FamilyType<IntType>,
    pub b_values: FamilyType<IntType>,
    pub ring_dimension: usize,
    pub lwe_dimension: usize,
    pub base_bits: usize,
    pub digit_count: usize,
    pub modulus: BigUint,
    pub noise_bound: BigUint,
}

impl GraphValue for KeySwitchKey {
    type Schema = KeySwitchKeySchema;

    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.a_values.flatten();
        values.extend(self.b_values.flatten());
        values
    }

    fn schema(&self) -> Self::Schema {
        KeySwitchKeySchema {
            a_values: self.a_values.schema(),
            b_values: self.b_values.schema(),
            ring_dimension: self.ring_dimension,
            lwe_dimension: self.lwe_dimension,
            base_bits: self.base_bits,
            digit_count: self.digit_count,
            modulus: self.modulus.clone(),
            noise_bound: self.noise_bound.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        if values.len() != 2 {
            return Err(DslError::Schema);
        }
        Ok(Self {
            a_values: Family::<Int>::from_values(&schema.a_values, &values[..1])?,
            b_values: Family::<Int>::from_values(&schema.b_values, &values[1..])?,
            ring_dimension: schema.ring_dimension,
            lwe_dimension: schema.lwe_dimension,
            base_bits: schema.base_bits,
            digit_count: schema.digit_count,
            modulus: schema.modulus.clone(),
            noise_bound: schema.noise_bound.clone(),
        })
    }
}

impl GraphValueSchema for KeySwitchKeySchema {
    type Value = KeySwitchKey;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        KeySwitchKey {
            a_values: self.a_values.placeholders_from(next),
            b_values: self.b_values.placeholders_from(next),
            ring_dimension: self.ring_dimension,
            lwe_dimension: self.lwe_dimension,
            base_bits: self.base_bits,
            digit_count: self.digit_count,
            modulus: self.modulus.clone(),
            noise_bound: self.noise_bound.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        let mut types = self.a_values.wire_types();
        types.extend(self.b_values.wire_types());
        types
    }
}

/// Secret and evaluation material produced together for a TFHE instance.
#[derive(Clone)]
pub struct TfheKeys {
    pub lwe_secret: Family<Int>,
    pub ring_secret: Mat,
    pub bootstrapping_key: BootstrappingKey,
    pub key_switch_key: KeySwitchKey,
}

#[derive(Clone, PartialEq)]
pub struct TfheKeysSchema {
    pub lwe_secret: FamilyType<IntType>,
    pub ring_secret: MatType,
    pub bootstrapping_key: BootstrappingKeySchema,
    pub key_switch_key: KeySwitchKeySchema,
}

impl GraphValue for TfheKeys {
    type Schema = TfheKeysSchema;

    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.lwe_secret.flatten();
        values.extend(self.ring_secret.flatten());
        values.extend(self.bootstrapping_key.flatten());
        values.extend(self.key_switch_key.flatten());
        values
    }

    fn schema(&self) -> Self::Schema {
        TfheKeysSchema {
            lwe_secret: self.lwe_secret.schema(),
            ring_secret: self.ring_secret.schema(),
            bootstrapping_key: self.bootstrapping_key.schema(),
            key_switch_key: self.key_switch_key.schema(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let lwe_count = schema.lwe_secret.wire_types().len();
        let ring_count = schema.ring_secret.wire_types().len();
        let bsk_count = schema.bootstrapping_key.wire_types().len();
        let ksk_count = schema.key_switch_key.wire_types().len();
        if values.len() != lwe_count + ring_count + bsk_count + ksk_count {
            return Err(DslError::Schema);
        }
        let ring_start = lwe_count;
        let bsk_start = ring_start + ring_count;
        let ksk_start = bsk_start + bsk_count;
        Ok(Self {
            lwe_secret: Family::<Int>::from_values(&schema.lwe_secret, &values[..ring_start])?,
            ring_secret: Mat::from_values(&schema.ring_secret, &values[ring_start..bsk_start])?,
            bootstrapping_key: BootstrappingKey::from_values(
                &schema.bootstrapping_key,
                &values[bsk_start..ksk_start],
            )?,
            key_switch_key: KeySwitchKey::from_values(
                &schema.key_switch_key,
                &values[ksk_start..],
            )?,
        })
    }
}

impl GraphValueSchema for TfheKeysSchema {
    type Value = TfheKeys;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        TfheKeys {
            lwe_secret: self.lwe_secret.placeholders_from(next),
            ring_secret: self.ring_secret.placeholders_from(next),
            bootstrapping_key: self.bootstrapping_key.placeholders_from(next),
            key_switch_key: self.key_switch_key.placeholders_from(next),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        let mut types = self.lwe_secret.wire_types();
        types.extend(self.ring_secret.wire_types());
        types.extend(self.bootstrapping_key.wire_types());
        types.extend(self.key_switch_key.wire_types());
        types
    }
}

/// TFHE parameters over integer LWE and the repository's exact CRT ring.
///
/// LWE and ring moduli are independent. The LWE modulus must be a power of two,
/// and the ring dimension must be at least four for the signed NAND LUT;
/// sample extraction performs rounded `Q -> q` modulus switching before the
/// flat key switch.
#[derive(Clone, Debug)]
pub struct TfheParams {
    pub common: FheCommonParams,
    pub lwe_dimension: usize,
    pub lwe_modulus: BigUint,
    pub lwe_error_sigma: f64,
    pub lwe_error_cutoff: BigUint,
    key_switch_base_bits: usize,
    key_switch_digits: usize,
}

impl TfheParams {
    pub fn new(
        common: FheCommonParams,
        lwe_dimension: usize,
        lwe_modulus: BigUint,
        lwe_error_sigma: f64,
        lwe_error_cutoff: BigUint,
    ) -> Result<Self, FheError> {
        common.validate()?;
        if common.secret_range.minimum != IntExpr::constant(0) ||
            common.secret_range.maximum != IntExpr::constant(1)
        {
            return Err(FheError::InvalidParameters(
                "TFHE uses binary ring and integer LWE secrets in [0,1]",
            ));
        }
        let modulus_bit_length = lwe_modulus.bits();
        let power_of_two =
            modulus_bit_length > 0 && (BigUint::one() << (modulus_bit_length - 1)) == lwe_modulus;
        let ring_dimension = common.ring.ring_dimension() as usize;
        if !power_of_two ||
            ring_dimension < 4 ||
            lwe_dimension == 0 ||
            lwe_dimension > ring_dimension ||
            lwe_modulus < BigUint::from(16u8)
        {
            return Err(FheError::InvalidParameters(
                "TFHE needs ring dimension >= 4, 0 < LWE dimension <= ring dimension, and a power-of-two LWE modulus",
            ));
        }
        if !lwe_error_sigma.is_finite() || lwe_error_sigma <= 0.0 || lwe_error_cutoff.is_zero() {
            return Err(FheError::InvalidParameters(
                "TFHE LWE Gaussian sigma and cutoff must be positive",
            ));
        }
        let minimum_ring_cutoff = minimum_sixteen_sigma_cutoff(common.error_sigma)?;
        let minimum_lwe_cutoff = minimum_sixteen_sigma_cutoff(lwe_error_sigma)?;
        if common.error_cutoff < minimum_ring_cutoff || lwe_error_cutoff < minimum_lwe_cutoff {
            return Err(FheError::InvalidParameters(
                "TFHE ring and LWE Gaussian cutoffs must be at least 16 times sigma",
            ));
        }
        let delta = &lwe_modulus >> 3;
        if &lwe_error_cutoff >= &delta || lwe_modulus >= *common.ring.modulus() {
            return Err(FheError::InvalidParameters(
                "LWE noise must fit the signed encoding and q must be below ring Q",
            ));
        }
        // The security profile uses binary key switching independently of the
        // larger ring-GSW gadget base.
        let key_switch_base_bits = 1usize;
        let key_switch_digits = (&lwe_modulus - BigUint::one()).bits() as usize;
        Ok(Self {
            common,
            lwe_dimension,
            lwe_modulus,
            lwe_error_sigma,
            lwe_error_cutoff,
            key_switch_base_bits,
            key_switch_digits,
        })
    }

    pub fn delta(&self) -> BigUint {
        &self.lwe_modulus >> 3
    }

    pub fn key_switch_digit_count(&self) -> usize {
        self.key_switch_digits
    }

    pub fn key_switch_base_bits(&self) -> usize {
        self.key_switch_base_bits
    }

    /// Registers the ordered CRT prefixes and single-prime rings used by the
    /// runtime polynomial backend for external products and extraction.
    pub fn runtime_parameters(&self) -> Vec<DCRTPolyParams> {
        let (primes, _, depth) = self.common.ring.to_crt();
        let mut rings = (0..depth)
            .map(|level| self.common.parameters_at(level).expect("validated CRT prefix"))
            .collect::<Vec<_>>();
        rings.extend(primes.iter().map(|prime| {
            self.common.ring.select_modulus(&BigUint::from(*prime)).expect("validated CRT prime")
        }));
        let mut unique = Vec::with_capacity(rings.len());
        for ring in rings {
            if !unique.contains(&ring) {
                unique.push(ring);
            }
        }
        unique
    }

    /// Generates independent binary integer and ring secrets, GSW encryptions
    /// for blind rotation, and a flat key-switch key. `hash_key` is a fresh
    /// public 32-byte seed supplied by the caller from an OS CSPRNG; it drives
    /// only KSK `a` sampling and must not be reused for another key generation.
    /// The flat hash-sampled family is deterministic from its seed, tag, and
    /// element index (the seeded-LWE convention), not literal independent draws.
    pub fn keygen(&self, hash_key: &Bytes) -> Result<TfheKeys, FheError> {
        self.common.validate()?;
        self.check_hash_key(hash_key)?;
        let context = DslContext::new("tfhe-keygen");
        let ring = self.common.ring();
        let lwe_secret_source = ring.uniform_interval((1, 1), 0, 1).coefficients();
        let lwe_secret = parallel(self.lwe_dimension, |index| Ok(lwe_secret_source.at(index)))?;
        let ring_secret = ring.uniform_interval((1, 1), 0, 1);
        let lwe_secret_dimension = self.lwe_dimension;
        let bootstrapping_entries = parallel(lwe_secret_dimension, |index| {
            Ok(self.encrypt_gsw_unchecked(&ring_secret, lwe_secret.at(index)))
        })?;
        let bootstrapping_key = BootstrappingKey {
            entries: bootstrapping_entries,
            lwe_dimension: self.lwe_dimension,
            ring_modulus: self.common.ring.modulus().as_ref().clone(),
        };

        let ring_dimension = self.common.ring.ring_dimension() as usize;
        let key_count = ring_dimension * self.key_switch_digits;
        let flat_a_count = key_count * self.lwe_dimension;
        let modulus_expr = IntExpr::constant(BigInt::from(self.lwe_modulus.clone()));
        let a_values = context.hash_int_family(
            hash_key.clone(),
            HashTag::from(b"tfhe/keygen/ksk-a/v1".as_slice()),
            flat_a_count,
            modulus_expr,
        );
        let ring_secret_coefficients = ring_secret.coefficients();
        let base = BigUint::from(1u8) << self.key_switch_base_bits;
        let powers = (0..self.key_switch_digits)
            .map(|digit| Int::constant(BigInt::from(base.pow(digit as u32))))
            .collect::<Vec<_>>();
        let b_values = parallel(key_count, |flat_index| {
            let coefficient_index = flat_index.clone().div(self.key_switch_digits);
            let digit_index = flat_index.clone().rem(self.key_switch_digits);
            let gadget = select(digit_index, powers.clone())?;
            let dot = balanced_sum(
                (0..self.lwe_dimension)
                    .map(|coordinate| {
                        let index = flat_index.clone().mul(self.lwe_dimension).add(coordinate);
                        a_values.at(index).mul(lwe_secret.at(coordinate))
                    })
                    .collect(),
            );
            let message = ring_secret_coefficients.at(coefficient_index).mul(gadget);
            Ok(dot
                .add(message)
                .add(self.lwe_error()?)
                .rem(Int::constant(BigInt::from(self.lwe_modulus.clone()))))
        })?;
        let key_switch_key = KeySwitchKey {
            a_values,
            b_values,
            ring_dimension,
            lwe_dimension: self.lwe_dimension,
            base_bits: self.key_switch_base_bits,
            digit_count: self.key_switch_digits,
            modulus: self.lwe_modulus.clone(),
            noise_bound: self.lwe_error_cutoff.clone(),
        };
        Ok(TfheKeys { lwe_secret, ring_secret, bootstrapping_key, key_switch_key })
    }

    /// Encrypts the integer bit 0 or 1 using signed `±floor(q/8)` encoding.
    /// `hash_key` must be fresh public 32-byte CSPRNG output for every
    /// ciphertext; a repeated seed and tag reproduce the same pseudorandom LWE
    /// `a` vector. The `a` family is a seeded-LWE pseudorandom expansion,
    /// indexed and domain-separated by the hash sampler.
    pub fn encrypt(
        &self,
        secret: &Family<Int>,
        message: &Int,
        hash_key: &Bytes,
    ) -> Result<LweCiphertext, FheError> {
        check_family(secret, self.lwe_dimension)?;
        self.check_hash_key(hash_key)?;
        let context = DslContext::new("tfhe-lwe-encrypt");
        let modulus = Int::constant(BigInt::from(self.lwe_modulus.clone()));
        let a = context.hash_int_family(
            hash_key.clone(),
            HashTag::from(b"tfhe/lwe-encrypt/a/v1".as_slice()),
            self.lwe_dimension,
            IntExpr::constant(BigInt::from(self.lwe_modulus.clone())),
        );
        let dot = inner_product(&a, secret, self.lwe_dimension);
        let signed_bit = message.clone().mul(2).sub(1);
        let encoded = signed_bit.mul(Int::constant(BigInt::from(self.delta())));
        let b = dot.add(encoded).add(self.lwe_error()?).rem(modulus);
        Ok(LweCiphertext {
            a,
            b,
            dimension: self.lwe_dimension,
            modulus: self.lwe_modulus.clone(),
            noise_bound: self.lwe_error_cutoff.clone(),
        })
    }

    /// Decodes a signed Boolean phase to the integer 0 or 1.
    pub fn decrypt(
        &self,
        secret: &Family<Int>,
        ciphertext: &LweCiphertext,
    ) -> Result<Int, FheError> {
        self.validate_lwe(ciphertext, self.lwe_dimension, &self.lwe_modulus)?;
        check_family(secret, self.lwe_dimension)?;
        let q = Int::constant(BigInt::from(self.lwe_modulus.clone()));
        let phase = ciphertext
            .b
            .clone()
            .sub(inner_product(&ciphertext.a, secret, self.lwe_dimension))
            .rem(q.clone());
        let centered = center_residue(phase, &self.lwe_modulus)?;
        Ok(Int::constant(1).sub(centered.less(Int::constant(0)).to_int()))
    }

    /// Checks shape and the declared LWE noise bound against the signed-bit
    /// decoding margin. This does not inspect runtime ciphertext values; after
    /// a staged rebind, the caller must ensure the schema carries a valid bound.
    pub fn can_decrypt(&self, ciphertext: &LweCiphertext) -> bool {
        ciphertext.dimension == self.lwe_dimension &&
            ciphertext.modulus == self.lwe_modulus &&
            check_family(&ciphertext.a, self.lwe_dimension).is_ok() &&
            ciphertext.noise_bound < self.delta()
    }

    /// Forms `Δ - ct1 - ct2`, the signed-phase input used by the NAND LUT.
    pub fn nand_input(
        &self,
        lhs: &LweCiphertext,
        rhs: &LweCiphertext,
    ) -> Result<LweCiphertext, FheError> {
        self.validate_lwe(lhs, self.lwe_dimension, &self.lwe_modulus)?;
        self.validate_lwe(rhs, self.lwe_dimension, &self.lwe_modulus)?;
        lhs.add(rhs)?.negate()?.add_plain(BigInt::from(self.delta()))
    }

    /// Constructs the ring accumulator for a signed phase sign LUT. For input
    /// phases `-Δ`, `+Δ`, and `+3Δ`, its extracted constant coefficient is
    /// respectively `-Δ`, `+Δ`, and `+Δ` after Q-to-q rounding.
    pub fn nand_accumulator(&self) -> Mat {
        let ring = self.common.ring();
        let coefficient =
            IntExpr::constant(BigInt::from(self.common.ring.modulus().as_ref() / 8u8));
        ring.polynomial(std::iter::repeat_n(
            coefficient,
            self.common.ring.ring_dimension() as usize,
        ))
    }

    /// Stage 1: rotate the sign accumulator by `X^-b` and wrap it as a trivial
    /// ring LWE ciphertext `(0, X^-b V)`. Input phase noise affects which LUT
    /// region is selected; it is not an additive ring-coefficient error term.
    pub fn pre_blind_rotation(
        &self,
        ciphertext: &LweCiphertext,
        accumulator: &Mat,
    ) -> Result<RingCiphertext, FheError> {
        self.validate_lwe(ciphertext, self.lwe_dimension, &self.lwe_modulus)?;
        check_matrix(&self.common.ring, accumulator, 1, 1)?;
        let exponent = scale_to_ring_exponent(
            ciphertext.b.clone(),
            &self.lwe_modulus,
            self.common.ring.ring_dimension() as usize,
        );
        let negative_exponent =
            Int::constant(0).sub(exponent).rem(2 * self.common.ring.ring_dimension() as usize);
        let ring = self.common.ring();
        Ok(RingCiphertext {
            a: ring.zero((1, 1)),
            b: rotate(
                accumulator,
                negative_exponent,
                &ring,
                self.common.ring.ring_dimension() as usize,
            )?,
            noise_bound: BigUint::zero(),
            // Any canonical accumulator coefficient is at most Q/2 in
            // centered form. Built-in NAND coefficients use only Q/8.
            plaintext_bound: self.common.ring.modulus().as_ref() / 2u8,
        })
    }

    /// Stage 2: apply one encrypted conditional monomial rotation per LWE
    /// secret coordinate using the ring-GSW external product.
    pub fn blind_rotation(
        &self,
        initial: &RingCiphertext,
        ciphertext: &LweCiphertext,
        bootstrapping_key: &BootstrappingKey,
    ) -> Result<RingCiphertext, FheError> {
        self.validate_lwe(ciphertext, self.lwe_dimension, &self.lwe_modulus)?;
        check_matrix(&self.common.ring, &initial.a, 1, 1)?;
        check_matrix(&self.common.ring, &initial.b, 1, 1)?;
        if bootstrapping_key.lwe_dimension != self.lwe_dimension ||
            bootstrapping_key.ring_modulus != *self.common.ring.modulus() ||
            bootstrapping_key.entries.count() != &IntExpr::constant(self.lwe_dimension)
        {
            return Err(FheError::ShapeMismatch);
        }
        let first_entry = bootstrapping_key.entries.at(0);
        check_matrix(&self.common.ring, &first_entry.a, 1, 2 * self.common.ring.modulus_digits())?;
        check_matrix(&self.common.ring, &first_entry.b, 1, 2 * self.common.ring.modulus_digits())?;

        let ring_dimension = self.common.ring.ring_dimension() as usize;
        let ring = self.common.ring();
        let (a, b) = iterate(
            self.lwe_dimension,
            (initial.a.clone(), initial.b.clone()),
            |index, (current_a, current_b)| {
                let exponent = scale_to_ring_exponent(
                    ciphertext.a.at(index.clone()),
                    &self.lwe_modulus,
                    ring_dimension,
                );
                let rotated_a = rotate(&current_a, exponent.clone(), &ring, ring_dimension)?;
                let rotated_b = rotate(&current_b, exponent, &ring, ring_dimension)?;
                let difference_a = rotated_a - current_a.clone();
                let difference_b = rotated_b - current_b.clone();
                let encrypted_bit = bootstrapping_key.entries.at(index);
                let (product_a, product_b) = self.external_product_matrices(
                    &encrypted_bit.a,
                    &encrypted_bit.b,
                    &difference_a,
                    &difference_b,
                );
                Ok((current_a + product_a, current_b + product_b))
            },
        )?;

        // Propagate static bounds on the host while the ciphertext matrices
        // travel through a single sequential-loop node in the graph.
        let mut noise_bound = initial.noise_bound.clone();
        let mut plaintext_bound = initial.plaintext_bound.clone();
        let gsw_schema = bootstrapping_key.entries.schema().element;
        for _ in 0..self.lwe_dimension {
            let difference_noise = &noise_bound * 2u8;
            let difference_plaintext = &plaintext_bound * 2u8;
            let product_noise =
                &BigUint::from(ring_dimension) * &gsw_schema.plaintext_bound * difference_noise +
                    &BigUint::from(ring_dimension * 2 * self.common.ring.modulus_digits()) *
                        (BigUint::one() << (self.common.ring.base_bits() - 1)) *
                        &gsw_schema.noise_bound;
            let product_plaintext =
                BigUint::from(ring_dimension) * difference_plaintext * &gsw_schema.plaintext_bound;
            noise_bound += product_noise;
            plaintext_bound += product_plaintext;
        }
        Ok(RingCiphertext { a, b, noise_bound, plaintext_bound })
    }

    /// Stage 3: sample-extract coefficient zero, then round each extracted
    /// coefficient from the ring modulus Q to the power-of-two LWE modulus q.
    pub fn sample_extract(&self, ciphertext: &RingCiphertext) -> Result<LweCiphertext, FheError> {
        let parameters = &self.common.ring;
        check_matrix(parameters, &ciphertext.a, 1, 1)?;
        check_matrix(parameters, &ciphertext.b, 1, 1)?;
        let ring_dimension = parameters.ring_dimension() as usize;
        let q = parameters.modulus();
        let q = q.as_ref();
        let ring_a = ciphertext.a.coefficients();
        let ring_b = ciphertext.b.coefficients();
        let q_integer = Int::constant(BigInt::from(self.lwe_modulus.clone()));
        let q_ring_integer = BigInt::from(q.clone());
        let a = parallel(ring_dimension, |index| {
            let source_index = Int::constant(ring_dimension).sub(index.clone()).rem(ring_dimension);
            let sign = index.clone().equal(0).to_int().mul(2).sub(1);
            let coefficient =
                ring_a.at(source_index).mul(sign).rem(Int::constant(q_ring_integer.clone()));
            Ok(mod_switch(coefficient, &q_ring_integer, &self.lwe_modulus, q_integer.clone()))
        })?;
        let b_coefficient = ring_b.at(0).rem(Int::constant(q_ring_integer.clone()));
        let b = mod_switch(b_coefficient, &q_ring_integer, &self.lwe_modulus, q_integer);
        let scaled_noise = (&ciphertext.noise_bound * &self.lwe_modulus + q / 2u8) / q;
        Ok(LweCiphertext {
            a,
            b,
            dimension: ring_dimension,
            modulus: self.lwe_modulus.clone(),
            noise_bound: scaled_noise + BigUint::from(ring_dimension + 1),
        })
    }

    /// Stage 4: switch an extracted q-LWE ciphertext to the original integer
    /// LWE key using base digits and the flat key-switch arrays.
    pub fn key_switch(
        &self,
        extracted: &LweCiphertext,
        key: &KeySwitchKey,
    ) -> Result<LweCiphertext, FheError> {
        let ring_dimension = self.common.ring.ring_dimension() as usize;
        self.validate_lwe(extracted, ring_dimension, &self.lwe_modulus)?;
        if key.ring_dimension != ring_dimension ||
            key.lwe_dimension != self.lwe_dimension ||
            key.base_bits != self.key_switch_base_bits ||
            key.digit_count != self.key_switch_digits ||
            key.modulus != self.lwe_modulus ||
            key.a_values.count() !=
                &IntExpr::constant(
                    ring_dimension * self.key_switch_digits * self.lwe_dimension,
                ) ||
            key.b_values.count() != &IntExpr::constant(ring_dimension * self.key_switch_digits)
        {
            return Err(FheError::ShapeMismatch);
        }
        let base = BigUint::from(1u8) << self.key_switch_base_bits;
        let q = Int::constant(BigInt::from(self.lwe_modulus.clone()));
        let initial_a = parallel(self.lwe_dimension, |_| Ok(Int::constant(0)))?;
        let (a_sum, b_sum) = iterate(
            ring_dimension,
            (initial_a, Int::constant(0)),
            |coefficient_index, (a_sum, b_sum)| {
                // The KSK uses base 2. Fold all digit terms for this ring
                // coefficient in parallel with the output coordinates, so
                // the runtime loop has only N sequential iterations.
                let coefficient = extracted.a.at(coefficient_index.clone());
                let digits = (0..self.key_switch_digits)
                    .map(|digit| {
                        coefficient
                            .clone()
                            .div(Int::constant(BigInt::from(base.pow(digit as u32))))
                            .rem(base.clone())
                    })
                    .collect::<Vec<_>>();
                let mut b_contribution = Int::constant(0);
                for (digit_index, digit) in digits.iter().enumerate() {
                    let entry =
                        coefficient_index.clone().mul(self.key_switch_digits).add(digit_index);
                    b_contribution = b_contribution.add(digit.clone().mul(key.b_values.at(entry)));
                }
                let next_a = parallel(self.lwe_dimension, |coordinate| {
                    let mut contribution = Int::constant(0);
                    for (digit_index, digit) in digits.iter().enumerate() {
                        let entry =
                            coefficient_index.clone().mul(self.key_switch_digits).add(digit_index);
                        let flat_index = entry.mul(self.lwe_dimension).add(coordinate.clone());
                        contribution =
                            contribution.add(digit.clone().mul(key.a_values.at(flat_index)));
                    }
                    Ok(a_sum.at(coordinate).add(contribution))
                })?;
                let next_b = b_sum.add(b_contribution);
                Ok((next_a, next_b))
            },
        )?;
        let b = extracted.b.clone().sub(b_sum).rem(q.clone());
        let a = parallel(self.lwe_dimension, |coordinate| {
            Ok(Int::constant(0).sub(a_sum.at(coordinate)).rem(q.clone()))
        })?;
        let key_noise = &key.noise_bound *
            BigUint::from(self.common.ring.ring_dimension() as usize) *
            BigUint::from(self.key_switch_digits) *
            (base - 1u8);
        Ok(LweCiphertext {
            a,
            b,
            dimension: self.lwe_dimension,
            modulus: self.lwe_modulus.clone(),
            noise_bound: &extracted.noise_bound + key_noise,
        })
    }

    /// Runs the four bootstrap stages on an arbitrary signed-bit LWE input.
    pub fn bootstrap(
        &self,
        ciphertext: &LweCiphertext,
        accumulator: &Mat,
        bootstrapping_key: &BootstrappingKey,
        key_switch_key: &KeySwitchKey,
    ) -> Result<LweCiphertext, FheError> {
        let initial = self.pre_blind_rotation(ciphertext, accumulator)?;
        let rotated = self.blind_rotation(&initial, ciphertext, bootstrapping_key)?;
        let extracted = self.sample_extract(&rotated)?;
        self.key_switch(&extracted, key_switch_key)
    }

    /// NANDs two encrypted bits using `Δ - ct1 - ct2` and the sign accumulator.
    pub fn nand(
        &self,
        lhs: &LweCiphertext,
        rhs: &LweCiphertext,
        accumulator: &Mat,
        bootstrapping_key: &BootstrappingKey,
        key_switch_key: &KeySwitchKey,
    ) -> Result<LweCiphertext, FheError> {
        let input = self.nand_input(lhs, rhs)?;
        self.bootstrap(&input, accumulator, bootstrapping_key, key_switch_key)
    }

    fn validate_lwe(
        &self,
        ciphertext: &LweCiphertext,
        dimension: usize,
        modulus: &BigUint,
    ) -> Result<(), FheError> {
        if ciphertext.dimension != dimension {
            return Err(FheError::ShapeMismatch);
        }
        if &ciphertext.modulus != modulus {
            return Err(FheError::LevelMismatch);
        }
        check_family(&ciphertext.a, dimension)
    }

    fn check_hash_key(&self, key: &Bytes) -> Result<(), FheError> {
        if key.schema().length != IntExpr::constant(32) {
            return Err(FheError::ShapeMismatch);
        }
        Ok(())
    }

    fn lwe_error(&self) -> Result<Int, DslError> {
        let sigma = RealExpr::from_f64_exact(self.lwe_error_sigma)
            .expect("validated finite LWE error sigma");
        let ring = self.common.ring();
        let sample = ring.gaussian(
            (1, 1),
            sigma,
            IntExpr::constant(BigInt::from(self.lwe_error_cutoff.clone())),
        );
        center_residue(sample.extract_coefficient(0), self.common.ring.modulus().as_ref())
    }

    fn encrypt_gsw_unchecked(&self, secret: &Mat, message: Int) -> RingCiphertext {
        let parameters = &self.common.ring;
        let digits = parameters.modulus_digits();
        let ring = self.common.ring();
        let base = IntExpr::constant(BigInt::from(1u64 << parameters.base_bits()));
        let polynomial_message =
            message.lift_to_constant_polynomial(ring.zero((1, 1)).matrix_type().clone());
        let h = polynomial_message * ring.gadget(1, base, digits);
        let zero = ring.zero((1, digits));
        let a = ring.uniform_residue((1, 2 * digits));
        let b = secret * &a + self.common.gaussian(parameters, 1, 2 * digits);
        RingCiphertext {
            a: a + Mat::concat(ConcatAxis::Columns, vec![h.clone(), zero.clone()]),
            b: b + Mat::concat(ConcatAxis::Columns, vec![zero, h]),
            noise_bound: self.common.error_cutoff.clone(),
            plaintext_bound: BigUint::one(),
        }
    }

    #[cfg(test)]
    fn encrypt_gsw_noiseless(&self, message: Int) -> RingCiphertext {
        let parameters = &self.common.ring;
        let digits = parameters.modulus_digits();
        let ring = self.common.ring();
        let base = IntExpr::constant(BigInt::from(1u64 << parameters.base_bits()));
        let polynomial_message =
            message.lift_to_constant_polynomial(ring.zero((1, 1)).matrix_type().clone());
        let h = polynomial_message * ring.gadget(1, base, digits);
        let zero = ring.zero((1, digits));
        RingCiphertext {
            a: Mat::concat(ConcatAxis::Columns, vec![h.clone(), zero.clone()]),
            b: Mat::concat(ConcatAxis::Columns, vec![zero, h]),
            noise_bound: BigUint::zero(),
            plaintext_bound: BigUint::one(),
        }
    }

    fn external_product_matrices(
        &self,
        multiplier_a: &Mat,
        multiplier_b: &Mat,
        input_a: &Mat,
        input_b: &Mat,
    ) -> (Mat, Mat) {
        let parameters = &self.common.ring;
        let digits = parameters.modulus_digits();
        let matrix =
            Mat::concat(ConcatAxis::Rows, vec![multiplier_a.clone(), multiplier_b.clone()]);
        let column = Mat::concat(ConcatAxis::Rows, vec![input_a.clone(), input_b.clone()]);
        let decomposition = column
            .decompose(IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())), digits);
        let output = decomposition.mul_small_rhs(matrix);
        (
            output.clone().slice(
                Some(mxx_ir_core::node::IndexRange { start: 0.into(), end: 1.into() }),
                None,
            ),
            output.slice(
                Some(mxx_ir_core::node::IndexRange { start: 1.into(), end: 2.into() }),
                None,
            ),
        )
    }
}

fn inner_product(lhs: &Family<Int>, rhs: &Family<Int>, dimension: usize) -> Int {
    balanced_sum((0..dimension).map(|index| lhs.at(index).mul(rhs.at(index))).collect())
}

/// Builds a pairwise reduction tree so large dot products remain shallow when
/// the DSL graph sealer recursively visits expression arguments.
fn balanced_sum(mut terms: Vec<Int>) -> Int {
    if terms.is_empty() {
        return Int::constant(0);
    }
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut round = terms.into_iter();
        while let Some(left) = round.next() {
            if let Some(right) = round.next() {
                next.push(left.add(right));
            } else {
                next.push(left);
            }
        }
        terms = next;
    }
    terms.pop().expect("nonempty reduction terms")
}

fn center_residue(value: Int, modulus: &BigUint) -> Result<Int, DslError> {
    let q = BigInt::from(modulus.clone());
    let at_or_below_half = value.clone().mul(2).less_equal(Int::constant(q.clone()));
    select(at_or_below_half.to_int(), vec![value.clone().sub(Int::constant(q.clone())), value])
}

fn scale_to_ring_exponent(value: Int, modulus: &BigUint, ring_dimension: usize) -> Int {
    let exponent_modulus = 2 * ring_dimension;
    value
        .mul(exponent_modulus)
        .add(Int::constant(BigInt::from(modulus / 2u8)))
        .div(Int::constant(BigInt::from(modulus.clone())))
        .rem(Int::constant(exponent_modulus))
}

fn mod_switch(value: Int, q_from: &BigInt, q_to: &BigUint, q_to_value: Int) -> Int {
    value
        .mul(Int::constant(BigInt::from(q_to.clone())))
        .add(Int::constant(q_from / 2u8))
        .div(Int::constant(q_from.clone()))
        .rem(q_to_value)
}

fn minimum_sixteen_sigma_cutoff(sigma: f64) -> Result<BigUint, FheError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(FheError::InvalidParameters(
            "TFHE ring and LWE Gaussian sigma values must be positive",
        ));
    }
    let cutoff = (16.0 * sigma).ceil();
    if !cutoff.is_finite() || cutoff >= u64::MAX as f64 {
        return Err(FheError::InvalidParameters("TFHE Gaussian cutoff is too large"));
    }
    Ok(BigUint::from(cutoff as u64))
}

fn rotate(value: &Mat, exponent: Int, ring: &Ring, ring_dimension: usize) -> Result<Mat, DslError> {
    let n = Int::constant(ring_dimension);
    let normalized = exponent.rem(Int::constant(2 * ring_dimension));
    let index = normalized.clone().rem(n.clone());
    let negacyclic_sign = Int::constant(1).sub(normalized.div(n).mul(2));
    let coefficients = parallel(ring_dimension, |coefficient| {
        Ok(index.clone().equal(coefficient).to_int().mul(negacyclic_sign.clone()))
    })?;
    let monomial = ring.from_coefficients(&coefficients);
    Ok(&monomial * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{common, execute_graph, int_input};
    use mxx_dsl::{BuiltGraph, DslContext, IntType};
    use mxx_runtime::RuntimeValue;
    use rand::RngCore;
    use std::collections::BTreeMap;

    fn test_params() -> TfheParams {
        let mut common = common();
        common.secret_range =
            mxx_ir_core::node::SampleRange { minimum: 0.into(), maximum: 1.into() };
        common.error_cutoff =
            common.error_cutoff.max(BigUint::from((common.error_sigma * 16.0).ceil() as u64));
        let target_modulus = BigUint::one() << 32usize;
        let lwe_modulus = if target_modulus < *common.ring.modulus() {
            target_modulus
        } else {
            BigUint::one() << (common.ring.modulus().bits() - 1) as usize
        };
        TfheParams::new(common, 2, lwe_modulus, 1.0, BigUint::from(16u8)).unwrap()
    }

    fn exact_key_switch_key(
        parameters: &TfheParams,
        ring_secret: &Mat,
    ) -> Result<KeySwitchKey, DslError> {
        let ring_dimension = parameters.common.ring.ring_dimension() as usize;
        let key_count = ring_dimension * parameters.key_switch_digit_count();
        let flat_a_count = key_count * parameters.lwe_dimension;
        let a_values = parallel(flat_a_count, |_| Ok(Int::constant(0)))?;
        let coefficients = ring_secret.coefficients();
        let base = BigUint::from(1u8) << parameters.key_switch_base_bits();
        let powers = (0..parameters.key_switch_digit_count())
            .map(|digit| Int::constant(BigInt::from(base.pow(digit as u32))))
            .collect::<Vec<_>>();
        let b_values = parallel(key_count, |index| {
            let coefficient = index.clone().div(parameters.key_switch_digit_count());
            let digit = index.rem(parameters.key_switch_digit_count());
            Ok(coefficients.at(coefficient).mul(select(digit, powers.clone())?))
        })?;
        Ok(KeySwitchKey {
            a_values,
            b_values,
            ring_dimension,
            lwe_dimension: parameters.lwe_dimension,
            base_bits: parameters.key_switch_base_bits(),
            digit_count: parameters.key_switch_digit_count(),
            modulus: parameters.lwe_modulus.clone(),
            noise_bound: BigUint::zero(),
        })
    }

    fn input_ciphertext(context: &DslContext, name: &str, params: &TfheParams) -> LweCiphertext {
        LweCiphertext {
            a: context.int_family_input(format!("{name}-a"), params.lwe_dimension),
            b: context.input(format!("{name}-b"), IntType).unwrap(),
            dimension: params.lwe_dimension,
            modulus: params.lwe_modulus.clone(),
            noise_bound: BigUint::zero(),
        }
    }

    fn encoded_ciphertext_inputs(
        params: &TfheParams,
        name: &str,
        a: [i64; 2],
        secret: [i64; 2],
        bit: bool,
    ) -> [(String, RuntimeValue<mxx_runtime::backend::poly::CpuDcrtBackend>); 2] {
        let dot = a[0] * secret[0] + a[1] * secret[1];
        let signed_delta = if bit { params.delta() } else { &params.lwe_modulus - params.delta() };
        let b = (BigUint::from(dot as u64) + signed_delta) % &params.lwe_modulus;
        [
            (format!("{name}-a"), int_input(&a)),
            (format!("{name}-b"), RuntimeValue::Int(BigInt::from(b))),
        ]
    }

    fn graph_copy(graph: &BuiltGraph) -> BuiltGraph {
        BuiltGraph { graph: graph.graph.clone() }
    }

    fn fresh_test_hash_key() -> Vec<u8> {
        let mut key = vec![0u8; 32];
        rand::rng().fill_bytes(&mut key);
        key
    }

    #[test]
    fn test_tfhe_noiseless_nand_truth_table_and_repeated_gate() {
        let parameters = test_params();
        assert_eq!(parameters.key_switch_base_bits(), 1);
        assert_eq!(
            parameters.key_switch_digit_count(),
            (&parameters.lwe_modulus - BigUint::one()).bits() as usize
        );
        let context = DslContext::new("tfhe-noiseless-nand");
        let lwe_secret = context.int_family_input("lwe-secret", parameters.lwe_dimension);
        let ring_coefficients = context.int_family_input(
            "ring-secret-coefficients",
            parameters.common.ring.ring_dimension() as usize,
        );
        let ring_secret = parameters.common.ring().from_coefficients(&ring_coefficients);
        let bsk_entries = parallel(parameters.lwe_dimension, |index| {
            Ok(parameters.encrypt_gsw_noiseless(lwe_secret.at(index)))
        })
        .unwrap();
        let bootstrapping_key = BootstrappingKey {
            entries: bsk_entries,
            lwe_dimension: parameters.lwe_dimension,
            ring_modulus: parameters.common.ring.modulus().as_ref().clone(),
        };
        let key_switch_key = exact_key_switch_key(&parameters, &ring_secret).unwrap();
        let lhs = input_ciphertext(&context, "lhs", &parameters);
        let rhs = input_ciphertext(&context, "rhs", &parameters);
        let true_value = input_ciphertext(&context, "true", &parameters);
        let accumulator = parameters.nand_accumulator();
        let first =
            parameters.nand(&lhs, &rhs, &accumulator, &bootstrapping_key, &key_switch_key).unwrap();
        let repeated = parameters
            .nand(&first, &true_value, &accumulator, &bootstrapping_key, &key_switch_key)
            .unwrap();
        let graph = context
            .output("nand", parameters.decrypt(&lwe_secret, &first).unwrap())
            .unwrap()
            .output("repeated", parameters.decrypt(&lwe_secret, &repeated).unwrap())
            .unwrap()
            .build()
            .unwrap();

        let lwe_key = [1i64, 1i64];
        let ring_key = (0..parameters.common.ring.ring_dimension() as usize)
            .map(|index| if index % 2 == 0 { 1i64 } else { 0 })
            .collect::<Vec<_>>();
        for left in [false, true] {
            for right in [false, true] {
                let expected_nand = !(left && right);
                let mut inputs = BTreeMap::from([
                    ("lwe-secret".into(), int_input(&lwe_key)),
                    ("ring-secret-coefficients".into(), int_input(&ring_key)),
                ]);
                inputs.extend(encoded_ciphertext_inputs(&parameters, "lhs", [3, 5], lwe_key, left));
                inputs.extend(encoded_ciphertext_inputs(
                    &parameters,
                    "rhs",
                    [7, 11],
                    lwe_key,
                    right,
                ));
                inputs.extend(encoded_ciphertext_inputs(
                    &parameters,
                    "true",
                    [13, 17],
                    lwe_key,
                    true,
                ));
                let result = execute_graph(graph_copy(&graph), &parameters.common, inputs, &[]);
                let RuntimeValue::Int(nand) = &result.outputs["nand"] else {
                    panic!("expected NAND integer output")
                };
                let RuntimeValue::Int(repeated) = &result.outputs["repeated"] else {
                    panic!("expected repeated NAND integer output")
                };
                assert_eq!(nand, &BigInt::from(u8::from(expected_nand)));
                assert_eq!(repeated, &BigInt::from(u8::from(left && right)));
            }
        }
    }

    #[test]
    fn test_tfhe_nearest_exponent_rounding_prevents_accumulated_floor_bias() {
        let ring = DCRTPolyParams::try_new(8, 2, 7, 1, Some(vec![97, 113]), None).unwrap();
        let common = FheCommonParams {
            ring,
            secret_range: mxx_ir_core::node::SampleRange { minimum: 0.into(), maximum: 1.into() },
            error_sigma: 1.0,
            error_cutoff: BigUint::from(16u8),
        };
        let parameters =
            TfheParams::new(common, 8, BigUint::from(256u16), 1.0, BigUint::from(16u8)).unwrap();
        let context = DslContext::new("tfhe-nearest-exponent-rounding");
        let lwe_secret = context.int_family_input("lwe-secret", 8);
        let ring_coefficients = context.int_family_input("ring-secret-coefficients", 8);
        let ring_secret = parameters.common.ring().from_coefficients(&ring_coefficients);
        let bootstrapping_key = BootstrappingKey {
            entries: parallel(
                8,
                |index| Ok(parameters.encrypt_gsw_noiseless(lwe_secret.at(index))),
            )
            .unwrap(),
            lwe_dimension: 8,
            ring_modulus: parameters.common.ring.modulus().as_ref().clone(),
        };
        let key_switch_key = exact_key_switch_key(&parameters, &ring_secret).unwrap();
        let ciphertext = LweCiphertext {
            a: context.int_family_input("ciphertext-a", 8),
            b: context.input("ciphertext-b", IntType).unwrap(),
            dimension: 8,
            modulus: parameters.lwe_modulus.clone(),
            noise_bound: BigUint::zero(),
        };
        let bootstrapped = parameters
            .bootstrap(
                &ciphertext,
                &parameters.nand_accumulator(),
                &bootstrapping_key,
                &key_switch_key,
            )
            .unwrap();
        let graph = context
            .output("decoded", parameters.decrypt(&lwe_secret, &bootstrapped).unwrap())
            .unwrap()
            .build()
            .unwrap();

        // Here N=n=8, q=256, a_i=14, and b=sum(a_i)+q/8=144. Per-coordinate
        // floor scaling gives sum(floor(16*a_i/q))-floor(16*b/q)=-9 ≡ 7
        // (mod 16), crossing the LUT boundary. Nearest rounding gives
        // 8-9=-1 ≡ 15 and preserves the positive signed phase.
        let inputs = BTreeMap::from([
            ("lwe-secret".into(), int_input(&[1; 8])),
            ("ring-secret-coefficients".into(), int_input(&[0; 8])),
            ("ciphertext-a".into(), int_input(&[14; 8])),
            ("ciphertext-b".into(), RuntimeValue::Int(BigInt::from(144u16))),
        ]);
        let result = execute_graph(graph, &parameters.common, inputs, &[]);
        let RuntimeValue::Int(decoded) = &result.outputs["decoded"] else {
            panic!("expected decoded integer bit")
        };
        assert_eq!(decoded, &BigInt::from(1u8));
    }

    #[test]
    fn test_tfhe_q_2_pow_32_uses_32_binary_key_switch_digits() {
        let mut common = common();
        common.secret_range =
            mxx_ir_core::node::SampleRange { minimum: 0.into(), maximum: 1.into() };
        common.error_cutoff =
            common.error_cutoff.max(BigUint::from((common.error_sigma * 16.0).ceil() as u64));
        let q = BigUint::one() << 32usize;
        if q >= *common.ring.modulus() {
            return;
        }
        let parameters = TfheParams::new(common, 2, q, 1.0, BigUint::from(16u8)).unwrap();
        assert_eq!(parameters.key_switch_digit_count(), 32);
    }

    #[test]
    fn test_tfhe_rejects_ring_dimension_two_for_nand_lut() {
        let ring = DCRTPolyParams::try_new(2, 2, 6, 1, Some(vec![53, 61]), None).unwrap();
        let common = FheCommonParams {
            ring,
            secret_range: mxx_ir_core::node::SampleRange { minimum: 0.into(), maximum: 1.into() },
            error_sigma: 0.0625,
            error_cutoff: BigUint::one(),
        };
        let error =
            TfheParams::new(common, 2, BigUint::from(16u8), 0.0625, BigUint::one()).unwrap_err();
        assert!(matches!(
            error,
            FheError::InvalidParameters(
                "TFHE needs ring dimension >= 4, 0 < LWE dimension <= ring dimension, and a power-of-two LWE modulus"
            )
        ));
    }

    #[test]
    fn test_tfhe_noisy_nand_truth_table() {
        let parameters = test_params();
        let context = DslContext::new("tfhe-noisy-nand");
        let ring = parameters.common.ring();
        let keygen_hash_key = ring.bytes_input("keygen-hash-key", 32);
        let keys = parameters.keygen(&keygen_hash_key).unwrap();
        let left_message = context.input("left-message", IntType).unwrap();
        let right_message = context.input("right-message", IntType).unwrap();
        let left_hash_key = ring.bytes_input("left-hash-key", 32);
        let right_hash_key = ring.bytes_input("right-hash-key", 32);
        let left = parameters.encrypt(&keys.lwe_secret, &left_message, &left_hash_key).unwrap();
        let right = parameters.encrypt(&keys.lwe_secret, &right_message, &right_hash_key).unwrap();
        let accumulator = parameters.nand_accumulator();
        let output = parameters
            .nand(&left, &right, &accumulator, &keys.bootstrapping_key, &keys.key_switch_key)
            .unwrap();
        let decrypted = parameters.decrypt(&keys.lwe_secret, &output).unwrap();
        let graph = context.output("nand", decrypted).unwrap().build().unwrap();

        for left in [false, true] {
            for right in [false, true] {
                let inputs = BTreeMap::from([
                    ("keygen-hash-key".into(), RuntimeValue::Bytes(fresh_test_hash_key())),
                    ("left-hash-key".into(), RuntimeValue::Bytes(fresh_test_hash_key())),
                    ("right-hash-key".into(), RuntimeValue::Bytes(fresh_test_hash_key())),
                    ("left-message".into(), RuntimeValue::Int(BigInt::from(u8::from(left)))),
                    ("right-message".into(), RuntimeValue::Int(BigInt::from(u8::from(right)))),
                ]);
                let result = execute_graph(graph_copy(&graph), &parameters.common, inputs, &[]);
                let RuntimeValue::Int(actual) = &result.outputs["nand"] else {
                    panic!("expected noisy NAND bit")
                };
                assert_eq!(actual, &BigInt::from(u8::from(!(left && right))));
            }
        }
    }

    #[test]
    fn test_tfhe_noisy_integer_lwe_roundtrip_uses_fresh_hash_keys() {
        let parameters = test_params();
        let context = DslContext::new("tfhe-noisy-lwe-roundtrip");
        let ring = parameters.common.ring();
        let keygen_hash_key = ring.bytes_input("keygen-hash-key", 32);
        let keys = parameters.keygen(&keygen_hash_key).unwrap();
        let message = context.input("message", IntType).unwrap();
        let encryption_hash_key = ring.bytes_input("encryption-hash-key", 32);
        let ciphertext =
            parameters.encrypt(&keys.lwe_secret, &message, &encryption_hash_key).unwrap();
        let decrypted = parameters.decrypt(&keys.lwe_secret, &ciphertext).unwrap();
        let graph = context.output("decrypted", decrypted).unwrap().build().unwrap();

        for bit in [0u8, 1] {
            let inputs = BTreeMap::from([
                ("keygen-hash-key".into(), RuntimeValue::Bytes(fresh_test_hash_key())),
                ("encryption-hash-key".into(), RuntimeValue::Bytes(fresh_test_hash_key())),
                ("message".into(), RuntimeValue::Int(BigInt::from(bit))),
            ]);
            let result = execute_graph(graph_copy(&graph), &parameters.common, inputs, &[]);
            let RuntimeValue::Int(actual) = &result.outputs["decrypted"] else {
                panic!("expected decoded integer bit")
            };
            assert_eq!(actual, &BigInt::from(bit));
        }
    }
}
