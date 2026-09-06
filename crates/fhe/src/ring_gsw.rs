use crate::{
    FheCommonParams, FheError, FheScheme,
    utils::{self, check_matrix, scalar},
};
use mxx_dsl::{DslError, GraphValue, GraphValueSchema, Int, Mat, MatType, parallel};
use mxx_ir_core::{
    IntExpr, ValueHandle,
    node::{ConcatAxis, IndexRange},
    types::WireType,
};
use mxx_primitives::poly::PolyParams;
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;

/// Shared a/b graph record. Bounds are public declarations propagated by the
/// evaluator, not norms measured from secret values or authenticated metadata.
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
    // Only matrices become runtime wires. Public bounds travel in the schema,
    // so reconstructing a graph value must retain both bounds explicitly.
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
/// Scalar a/b parts, phase b-s*a.
pub type RingRegevCiphertext = RingCiphertext;
/// A/b row parts each have shape (1,2L).
pub type RingGswCiphertext = RingCiphertext;

#[derive(Clone, Debug)]
pub struct RingGswParams {
    pub common: FheCommonParams,
    pub scale: BigUint,
    /// Declared centered coefficient bound for fresh Mat plaintext inputs.
    pub plaintext_bound: BigUint,
}

impl RingGswParams {
    pub fn new(
        common: FheCommonParams,
        scale: BigUint,
        plaintext_bound: BigUint,
    ) -> Result<Self, FheError> {
        common.validate()?;
        if scale.is_zero() ||
            &common.error_cutoff * 2u8 >= scale ||
            (&scale * &plaintext_bound + &common.error_cutoff) * 2u8 >= *common.ring.modulus()
        {
            return Err(FheError::InvalidParameters(
                "Ring Regev fresh message and noise bounds must satisfy strict decoding margins",
            ));
        }
        Ok(Self { common, scale, plaintext_bound })
    }

    /// Encrypts a Mat in the ciphertext ring without scaling its gadget diagonal.
    /// Every centered input coefficient must have magnitude <= plaintext_bound;
    /// this public contract is not checked by inspecting runtime secret values.
    /// Construction follows TFHE's Enc(0)+mu*G, using the repository's exact CRT
    /// gadget and the phase convention b-s*a (CGGI16, ePrint 2016/870).
    pub fn encrypt_gsw(&self, secret: &Mat, message: &Mat) -> Result<RingGswCiphertext, FheError> {
        self.common.validate()?;
        let params = &self.common.ring;
        check_matrix(params, secret, 1, 1)?;
        check_matrix(params, message, 1, 1)?;
        let digits = params.modulus_digits();
        let ring = self.common.ring();
        let base = IntExpr::constant(BigInt::from(1u64 << params.base_bits()));
        // Adding mu*G on the two diagonal blocks makes the phase row
        // (-s, 1)*C equal mu*(-s, 1)*G plus the sampled error row.
        let h = message * ring.gadget(1, base, digits);
        let zero = ring.zero((1, digits));
        let a = ring.uniform_residue((1, 2 * digits));
        let b = secret * &a + self.common.gaussian(params, 1, 2 * digits);
        Ok(RingCiphertext {
            a: a + Mat::concat(ConcatAxis::Columns, vec![h.clone(), zero.clone()]),
            b: b + Mat::concat(ConcatAxis::Columns, vec![zero, h]),
            noise_bound: self.common.error_cutoff.clone(),
            plaintext_bound: self.plaintext_bound.clone(),
        })
    }

    pub fn external_product(
        &self,
        multiplier: &RingGswCiphertext,
        input: &RingRegevCiphertext,
    ) -> Result<RingRegevCiphertext, FheError> {
        let params = &self.common.ring;
        let digits = params.modulus_digits();
        for part in [&multiplier.a, &multiplier.b] {
            check_matrix(params, part, 1, 2 * digits)?;
        }
        for part in [&input.a, &input.b] {
            check_matrix(params, part, 1, 1)?;
        }
        let matrix =
            Mat::concat(ConcatAxis::Rows, vec![multiplier.a.clone(), multiplier.b.clone()]);
        let column = Mat::concat(ConcatAxis::Rows, vec![input.a.clone(), input.b.clone()]);
        let decomposition =
            column.decompose(IntExpr::constant(BigInt::from(1u64 << params.base_bits())), digits);
        // Preimage's receiver is the bounded RHS, but the generated operation
        // is matrix * decomposition. Mat's overload instead takes SmallMatrix.
        let output = decomposition.mul_small_rhs(matrix);
        let n = BigUint::from(params.ring_dimension());
        // Negacyclic convolution: ||uv||∞ <= N ||u||∞ ||v||∞.
        // There are 2L digit polynomials, each bounded by B/2.
        let digit_bound = BigUint::from(1u8) << (params.base_bits() - 1);
        Ok(RingCiphertext {
            a: output.clone().slice(Some(IndexRange { start: 0.into(), end: 1.into() }), None),
            b: output.slice(Some(IndexRange { start: 1.into(), end: 2.into() }), None),
            noise_bound: &n * &multiplier.plaintext_bound * &input.noise_bound +
                &n * (2 * digits) * digit_bound * &multiplier.noise_bound,
            plaintext_bound: n * &input.plaintext_bound * &multiplier.plaintext_bound,
        })
    }

    /// Sufficient decoding condition under the declared and propagated bounds.
    /// A false result is conservative; decryption remains available to callers.
    pub fn can_decrypt(&self, ciphertext: &RingRegevCiphertext) -> bool {
        check_matrix(&self.common.ring, &ciphertext.a, 1, 1).is_ok() &&
            check_matrix(&self.common.ring, &ciphertext.b, 1, 1).is_ok() &&
            &ciphertext.noise_bound * 2u8 < self.scale &&
            (&self.scale * &ciphertext.plaintext_bound + &ciphertext.noise_bound) * 2u8 <
                *self.common.ring.modulus()
    }
}

impl FheScheme for RingGswParams {
    type Plaintext = Mat;
    type Ciphertext = RingRegevCiphertext;
    type MulRhs = RingGswCiphertext;
    type EvaluationKey = ();
    fn common_params(&self) -> &FheCommonParams {
        &self.common
    }
    fn keygen(&self) -> Result<(Mat, Mat), FheError> {
        self.common.validate()?;
        let secret = self.common.sample_secret();
        Ok((secret.clone(), secret))
    }
    fn encrypt(&self, key: &Mat, message: &Mat) -> Result<Self::Ciphertext, FheError> {
        let p = &self.common.ring;
        check_matrix(p, key, 1, 1)?;
        check_matrix(p, message, 1, 1)?;
        let a = self.common.ring().uniform_residue((1, 1));
        let b = key * &a +
            self.common.gaussian(p, 1, 1) +
            message * scalar(p, BigInt::from(self.scale.clone()));
        Ok(RingCiphertext {
            a,
            b,
            noise_bound: self.common.error_cutoff.clone(),
            plaintext_bound: self.plaintext_bound.clone(),
        })
    }
    fn decrypt(&self, secret: &Mat, ciphertext: &Self::Ciphertext) -> Result<Mat, FheError> {
        let p = &self.common.ring;
        for part in [secret, &ciphertext.a, &ciphertext.b] {
            check_matrix(p, part, 1, 1)?;
        }
        // Center before rounding: a residue near q represents a small negative
        // phase. Packing the decoded integers returns canonical R_q residues.
        let phase = &ciphertext.b - secret * &ciphertext.a;
        let coefficients = utils::extract(p, &phase)?;
        let delta = BigInt::from(self.scale.clone());
        let decoded = parallel(p.ring_dimension(), |index| {
            let value = utils::centered(coefficients.at(index), p.modulus().as_ref())?;
            Ok(value.mul(2).add(Int::constant(delta.clone())).div(Int::constant(&delta * 2)))
        })?;
        utils::pack(p, &decoded)
    }
    fn add(
        &self,
        lhs: &Self::Ciphertext,
        rhs: &Self::Ciphertext,
    ) -> Result<Self::Ciphertext, FheError> {
        for part in [&lhs.a, &lhs.b, &rhs.a, &rhs.b] {
            check_matrix(&self.common.ring, part, 1, 1)?;
        }
        Ok(RingCiphertext {
            a: &lhs.a + &rhs.a,
            b: &lhs.b + &rhs.b,
            noise_bound: &lhs.noise_bound + &rhs.noise_bound,
            plaintext_bound: &lhs.plaintext_bound + &rhs.plaintext_bound,
        })
    }
    fn mul(
        &self,
        lhs: &Self::Ciphertext,
        rhs: &Self::MulRhs,
        (): &(),
    ) -> Result<Self::Ciphertext, FheError> {
        self.external_product(rhs, lhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{common, execute_graph, int_input, integers};
    use mxx_dsl::DslContext;
    use rand::Rng;
    use std::collections::BTreeMap;

    #[test]
    fn test_ring_regev_noisy_runtime_and_external_products() {
        let common = common();
        let n = common.ring.ring_dimension() as usize;
        let scale = BigUint::from(1u64 << 22);
        let scheme = RingGswParams::new(common.clone(), scale, BigUint::from(2u8)).unwrap();
        let ctx = DslContext::new("ring-regev-gsw");
        let message = ctx.int_family_input("message", n);
        let (sk, ek) = scheme.keygen().unwrap();
        let ct = scheme.encrypt(&ek, &common.ring().from_coefficients(&message)).unwrap();
        let doubled = scheme.add(&ct, &ct).unwrap();
        let mut ctx = ctx
            .private_output("roundtrip", scheme.decrypt(&sk, &ct).unwrap().coefficients())
            .unwrap()
            .private_output("double", scheme.decrypt(&sk, &doubled).unwrap().coefficients())
            .unwrap();
        let mut inputs = BTreeMap::new();
        let values = (0..n).map(|_| rand::rng().random_range(-2..=2)).collect::<Vec<i64>>();
        inputs.insert("message".into(), int_input(&values));
        // Constants isolate zero/identity/sign behavior; X also exercises the
        // sign change when a coefficient wraps across X^N = -1.
        for (name, index, value) in
            [("zero", 0, 0), ("one", 0, 1), ("negative", 0, -1), ("rotate", 1, 1)]
        {
            let multiplier = ctx.int_family_input(name, n);
            let gsw =
                scheme.encrypt_gsw(&sk, &common.ring().from_coefficients(&multiplier)).unwrap();
            let result = scheme.mul(&ct, &gsw, &()).unwrap();
            ctx = ctx
                .private_output(
                    format!("result-{name}"),
                    scheme.decrypt(&sk, &result).unwrap().coefficients(),
                )
                .unwrap();
            let mut coefficients = vec![0; n];
            coefficients[index] = value;
            inputs.insert(name.into(), int_input(&coefficients));
        }
        let polynomial = ctx.int_family_input("polynomial", n);
        let gsw = scheme.encrypt_gsw(&sk, &common.ring().from_coefficients(&polynomial)).unwrap();
        let product = scheme.mul(&ct, &gsw, &()).unwrap();
        ctx = ctx
            .private_output(
                "polynomial-product",
                scheme.decrypt(&sk, &product).unwrap().coefficients(),
            )
            .unwrap();
        let mut multiplier = vec![0i64; n];
        multiplier[0] = 1;
        multiplier[1] = -1;
        inputs.insert("polynomial".into(), int_input(&multiplier));
        let column = Mat::concat(ConcatAxis::Rows, vec![ct.a.clone(), ct.b.clone()]);
        let digits = common.ring.modulus_digits();
        let base = BigUint::from(1u8) << common.ring.base_bits();
        // Check G*decompose(c) = c independently of encryption/decryption, so
        // a gadget-layout mismatch cannot hide behind a successful round trip.
        let recomposed = column
            .clone()
            .decompose(base.clone(), digits)
            .mul_small_rhs(common.ring().gadget(2, base, digits));
        ctx = ctx.output("original", column).unwrap().output("recomposed", recomposed).unwrap();
        let result = execute_graph(ctx.build().unwrap(), &common, inputs);
        let mxx_runtime::RuntimeValue::Matrix(original) = &result.outputs["original"] else {
            panic!("matrix")
        };
        let mxx_runtime::RuntimeValue::Matrix(recomposed) = &result.outputs["recomposed"] else {
            panic!("matrix")
        };
        assert_eq!(original, recomposed);
        use mxx_primitives::{
            element::PolyElem,
            poly::{Poly, dcrt::poly::DCRTPoly},
        };
        use num_integer::Integer;
        let q = BigInt::from(common.ring.modulus().as_ref().clone());
        let native = |v: &[i64]| {
            DCRTPoly::from_biguints(
                &common.ring,
                &v.iter()
                    .map(|v| BigInt::from(*v).mod_floor(&q).to_biguint().unwrap())
                    .collect::<Vec<_>>(),
            )
        };
        // Use the existing polynomial primitive as the multiplication oracle,
        // including negacyclic wraparound and canonical residues.
        let expected = (native(&values) * native(&multiplier))
            .coeffs()
            .iter()
            .map(|v| BigInt::from(v.value().clone()))
            .collect::<Vec<_>>();
        assert_eq!(integers(&result, "polynomial-product"), expected);
        assert_eq!(
            integers(&result, "roundtrip"),
            values.iter().map(|v| BigInt::from(*v).mod_floor(&q)).collect::<Vec<_>>()
        );
        assert_eq!(
            integers(&result, "double"),
            values.iter().map(|v| BigInt::from(2 * v).mod_floor(&q)).collect::<Vec<_>>()
        );
        assert_eq!(integers(&result, "result-zero"), vec![BigInt::from(0); n]);
        assert_eq!(integers(&result, "result-one"), integers(&result, "roundtrip"));
        assert_eq!(
            integers(&result, "result-negative"),
            values.iter().map(|v| BigInt::from(-v).mod_floor(&q)).collect::<Vec<_>>()
        );
        let mut rotated = values.clone();
        rotated.rotate_right(1);
        rotated[0] = -rotated[0];
        assert_eq!(
            integers(&result, "result-rotate"),
            rotated.into_iter().map(|v| BigInt::from(v).mod_floor(&q)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_ring_gsw_rejects_wrong_shape_and_scale() {
        let common = common();
        assert!(
            RingGswParams::new(common.clone(), BigUint::from(0u8), BigUint::from(2u8)).is_err()
        );
        assert!(
            RingGswParams::new(common.clone(), &common.error_cutoff * 2u8, BigUint::from(0u8))
                .is_err()
        );
        assert!(
            RingGswParams::new(
                common.clone(),
                common.ring.modulus().as_ref().clone(),
                BigUint::from(1u8)
            )
            .is_err()
        );
        let scheme =
            RingGswParams::new(common.clone(), BigUint::from(1024u32), BigUint::from(2u8)).unwrap();
        let ring = common.ring();
        let wrong = RingCiphertext {
            a: ring.zero((1, 1)),
            b: ring.zero((1, 1)),
            noise_bound: BigUint::from(0u8),
            plaintext_bound: BigUint::from(0u8),
        };
        assert!(matches!(scheme.external_product(&wrong, &wrong), Err(FheError::ShapeMismatch)));
        assert!(scheme.encrypt(&ring.zero((1, 1)), &ring.zero((1, 2))).is_err());
        assert!(
            scheme
                .encrypt_gsw(
                    &ring.zero((1, 1)),
                    &mxx_dsl::Ring::new(97, common.ring.ring_dimension()).zero((1, 1))
                )
                .is_err()
        );
        let mut undecodable = wrong.clone();
        undecodable.noise_bound = scheme.scale.clone();
        assert!(!scheme.can_decrypt(&undecodable));
    }
    #[test]
    fn test_ring_gsw_chained_bounds_and_schema() {
        use num_integer::Integer;
        let common = common();
        let n = common.ring.ring_dimension() as usize;
        let scheme =
            RingGswParams::new(common.clone(), BigUint::from(1u64 << 40), BigUint::from(2u8))
                .unwrap();
        let context = DslContext::new("ring-gsw-chained-bounds");
        let message = context.int_family_input("message", n);
        let multiplier = context.int_family_input("multiplier", n);
        let m = common.ring().from_coefficients(&message);
        let mu = common.ring().from_coefficients(&multiplier);
        let (secret, key) = scheme.keygen().unwrap();
        let input = scheme.encrypt(&key, &m).unwrap();
        let mut gsw = scheme.encrypt_gsw(&secret, &mu).unwrap();
        // A caller may carry a looser previously propagated bound. The next
        // operation must consume that bound rather than fresh sampler cutoff.
        gsw.noise_bound *= 2u8;
        let first = scheme.external_product(&gsw, &input).unwrap();
        let second = scheme.external_product(&gsw, &first).unwrap();
        let sum = scheme.add(&second, &first).unwrap();
        let digit_term = BigUint::from(n) *
            (2 * common.ring.modulus_digits()) *
            (BigUint::from(1u8) << (common.ring.base_bits() - 1)) *
            &gsw.noise_bound;
        assert_eq!(
            first.noise_bound,
            BigUint::from(n) * &gsw.plaintext_bound * &input.noise_bound + &digit_term
        );
        assert_eq!(
            second.noise_bound,
            BigUint::from(n) * &gsw.plaintext_bound * &first.noise_bound + digit_term
        );
        assert_eq!(sum.noise_bound, &second.noise_bound + &first.noise_bound);
        assert_eq!(sum.plaintext_bound, &second.plaintext_bound + &first.plaintext_bound);
        assert!(scheme.can_decrypt(&sum));
        // Graph reconstruction and loop placeholders must preserve metadata
        // even though flatten() exposes only the two matrix handles.
        let schema = sum.schema();
        let restored = RingCiphertext::from_values(&schema, &sum.flatten()).unwrap();
        assert!(restored.schema() == schema);
        assert!(schema.placeholders().schema() == schema);
        let expected = &m * &mu * &mu + &m * &mu;
        // Measure the actual accumulated error separately from decoding: a
        // correct plaintext alone would not prove that the bound is sound.
        let residual =
            &sum.b - &secret * &sum.a - &expected * scalar(&common.ring, scheme.scale.clone());
        let graph = context
            .private_output("decoded", scheme.decrypt(&secret, &sum).unwrap().coefficients())
            .unwrap()
            .private_output("expected", expected.coefficients())
            .unwrap()
            .private_output("residual", residual.coefficients())
            .unwrap()
            .build()
            .unwrap();
        let values = (0..n).map(|_| rand::rng().random_range(-2..=2)).collect::<Vec<i64>>();
        let mut mu_values = vec![0; n];
        mu_values[0] = 1;
        mu_values[1] = -1;
        let result = execute_graph(
            graph,
            &common,
            BTreeMap::from([
                ("message".into(), int_input(&values)),
                ("multiplier".into(), int_input(&mu_values)),
            ]),
        );
        assert_eq!(integers(&result, "decoded"), integers(&result, "expected"));
        let q = BigInt::from(common.ring.modulus().as_ref().clone());
        for value in integers(&result, "residual") {
            let value = value.mod_floor(&q);
            let centered = if &value * 2 > q { value - &q } else { value };
            assert!(centered.magnitude() <= &sum.noise_bound);
        }
    }
}
