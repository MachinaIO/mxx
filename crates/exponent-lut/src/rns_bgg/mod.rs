//! RNS-BGG parameters, polynomial blocks, relation carriers, and scheme capabilities.
pub mod encoding;
#[cfg(all(test, feature = "gpu"))]
pub(crate) mod gpu;
pub mod noise;
pub mod public_key;
use crate::{
    layout::ExpLayout,
    noise_refresh::{ExponentLUTRefreshDirectScheme, ExponentLUTRefreshScheme},
    program::{
        ExponentLutProgram, ExponentProgramLowering, FamilyRange, LutId, LutTable,
        ProgramFamilyBindings, ProgramGateInstance, ProgramLoweringTypes, ProgramValue,
    },
    readout::{
        ExponentLUTReadoutScheme, ExponentLUTTerminalScheme, RowLutTable, ValueReadout,
        ValueReadoutIdentity,
    },
    scheme::{ExponentLUTEncodingScheme, ExponentLUTFixedRhsScheme},
    utils::modular_inverse,
};
#[cfg(test)]
use crate::{program::WireId, scheme::PreparedProgram};
use bigdecimal::BigDecimal;
use mxx_dsl::{
    Bytes, DslContext, DslError, Family, FamilyType, GraphValue, GraphValueSchema, HashTag, Int,
    Mat, MatType, Preimage, Ring,
};
use mxx_ir_core::{
    IntExpr, ParamEnv, Rational, RealExpr, ValueHandle,
    artifact::{ArtifactConfidentiality, ProductionId},
    types::{MatrixType, WireType},
};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use noise::RnsBggNoiseValue;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
pub use public_key::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ops::{Add, Mul, Neg, Sub},
    sync::Arc,
};
use thiserror::Error;
impl From<RnsBggError> for crate::noise_refresh::setup::RefreshSetupError {
    fn from(error: RnsBggError) -> Self {
        Self::Backend(Box::new(error))
    }
}

#[derive(Debug, Error)]
pub enum RnsBggError {
    #[error("invalid RNS-BGG parameters: {0}")]
    Parameters(&'static str),
    #[error("RNS-BGG budget exceeded: {0}")]
    Budget(&'static str),
    #[error("RNS-BGG key, level, or matrix context mismatch")]
    Context,
    #[error("invalid RNS-BGG LUT or RHS")]
    Lut,
    #[error("RNS-BGG graph construction failed: {0}")]
    Graph(String),
}

impl From<mxx_dsl::DslError> for RnsBggError {
    fn from(error: mxx_dsl::DslError) -> Self {
        Self::Graph(error.to_string())
    }
}

/// Fixed low-ring ciphertext satisfying r*C = p_next*nu*r*G_next + F.
/// Sampling has L=(); preprocessing supplies Dec(C) in the input ring as L.
/// Its plaintext is owned by preprocessing, never supplied to online evaluation.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPrivateRhs<C = RnsBggMatrix, L = ()> {
    pub(crate) lifted_ciphertext: L,
    pub(crate) rhs_id: [u8; 32],
    pub(crate) ciphertext: C,
    pub(crate) level: usize,
    pub(crate) context: [u8; 32],
    pub(crate) id: [u8; 32],
}

/// A selector generator is invoked once per member inside its consuming batch.
/// Combined RHS values are already materialized graph families.
#[derive(Clone)]
pub enum RnsBggRhsFamily {
    Generated { body: mxx_dsl::Subgraph<Int, RnsBggMatrix>, count: IntExpr },
    Values(Family<RnsBggMatrix>),
}
impl RnsBggRhsFamily {
    fn count(&self) -> &IntExpr {
        match self {
            Self::Generated { count, .. } => count,
            Self::Values(values) => values.count(),
        }
    }
    fn at(&self, index: Int) -> Result<RnsBggMatrix, DslError> {
        match self {
            Self::Generated { body, .. } => body.call(index),
            Self::Values(values) => Ok(values.at(index)),
        }
    }
}

/// Cached direct product. Both ordinary states retain their key/level identity;
/// the bounded RHS action is shared with public preparation, without LUT helpers.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPreparedPrivateMul<K = RnsBggPublicKey, C = RnsBggMatrix, D = RnsBggDigits> {
    pub(crate) input: K,
    pub(crate) output: K,
    pub(crate) rhs: RnsBggPrivateRhs<C, D>,
}

impl<C, L> RnsBggPrivateRhs<C, L> {
    pub fn rhs_id(&self) -> &[u8; 32] {
        &self.rhs_id
    }
    pub fn id(&self) -> &[u8; 32] {
        &self.id
    }
}

impl<K, C, D> RnsBggPreparedPrivateMul<K, C, D> {
    pub fn output_key(&self) -> &K {
        &self.output
    }

    pub fn rhs(&self) -> &RnsBggPrivateRhs<C, D> {
        &self.rhs
    }
}

impl RnsBggPublicKeyFamily {
    fn from_program(
        values: &RnsBggPublicProgramValue,
        count: IntExpr,
    ) -> Result<Self, RnsBggError> {
        match values {
            ProgramValue::Scalar((value, _)) => Ok(Self {
                matrices: ProgramValue::Scalar(value.matrix.clone()),
                level: value.level,
                id: value.id,
                context: value.context,
                range: FamilyRange::full(count).map_err(|_| RnsBggError::Context)?,
            }),
            ProgramValue::Family(values) => {
                let metadata = values.schema().element.0;
                Ok(Self {
                    matrices: ProgramValue::Family(values.field(|(value, _)| value.matrix)?),
                    level: metadata.level,
                    id: metadata.id,
                    context: metadata.context,
                    range: FamilyRange::full(values.count().clone())
                        .map_err(|_| RnsBggError::Context)?,
                })
            }
        }
    }
    fn into_program(self) -> Result<RnsBggPublicProgramValue, RnsBggError> {
        let (element, shared, ports) = match self.matrices {
            ProgramValue::Scalar(value) => {
                let ports = value.flatten();
                (value.schema(), vec![true; ports.len()], ports)
            }
            ProgramValue::Family(values) => {
                let schema = values.schema();
                (schema.element, schema.shared, values.flatten())
            }
        };
        let schema = FamilyType {
            element: (
                RnsBggPublicKey {
                    matrix: element,
                    level: self.level,
                    id: self.id,
                    context: self.context,
                },
                None::<MatType>,
            ),
            count: self.range.count().clone(),
            shared,
        };
        Ok(ProgramValue::Family(Family::from_values(&schema, &ports)?))
    }
}
impl RnsBggEncodingFamily {
    fn from_program(
        values: &RnsBggEncodingProgramValue,
        count: IntExpr,
    ) -> Result<Self, RnsBggError> {
        match values {
            ProgramValue::Scalar((value, _)) => Ok(Self {
                rows: ProgramValue::Scalar(value.row.clone()),
                level: value.level,
                key_id: value.key_id,
                context: value.context,
                range: FamilyRange::full(count).map_err(|_| RnsBggError::Context)?,
            }),
            ProgramValue::Family(values) => {
                let metadata = values.schema().element.0;
                Ok(Self {
                    rows: ProgramValue::Family(values.field(|(value, _)| value.row)?),
                    level: metadata.level,
                    key_id: metadata.key_id,
                    context: metadata.context,
                    range: FamilyRange::full(values.count().clone())
                        .map_err(|_| RnsBggError::Context)?,
                })
            }
        }
    }
    fn into_program(self) -> Result<RnsBggEncodingProgramValue, RnsBggError> {
        let (element, shared, ports) = match self.rows {
            ProgramValue::Scalar(value) => {
                let ports = value.flatten();
                (value.schema(), vec![true; ports.len()], ports)
            }
            ProgramValue::Family(values) => {
                let schema = values.schema();
                (schema.element, schema.shared, values.flatten())
            }
        };
        let schema = FamilyType {
            element: (
                RnsBggEncoding {
                    row: element,
                    level: self.level,
                    key_id: self.key_id,
                    context: self.context,
                },
                None::<MatType>,
            ),
            count: self.range.count().clone(),
            shared,
        };
        Ok(ProgramValue::Family(Family::from_values(&schema, &ports)?))
    }
}
impl ExponentLUTFixedRhsScheme for RnsBggScheme {
    type FixedRhs = RnsBggPrivateRhs<RnsBggRhsFamily>;
    type PreparedFixedProduct = RnsBggPreparedPrivateMul<
        RnsBggPublicKeyFamily,
        ProgramValue<RnsBggMatrix, Family<RnsBggMatrix>>,
        ProgramValue<RnsBggDigits, Family<RnsBggDigits>>,
    >;
    type FixedProductBinding = RnsBggArtifactBinding;

    fn fixed_product_binding(&self) -> Self::FixedProductBinding {
        self.artifact_binding()
    }

    fn combine_fixed_rhs(
        &self,
        rhs: &Self::FixedRhs,
        weights: &Family<Mat>,
        active: &FamilyRange,
        id: [u8; 32],
    ) -> Result<Self::FixedRhs, RnsBggError> {
        let size = active
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(RnsBggError::Context)?;
        let batch = crate::utils::preprocess_batch_size()?;
        let outputs =
            Int::evaluate(weights.count().clone()).div(Int::constant(size)).expression()?;
        let ring = self.ring(rhs.level + 1)?;
        let columns = self.parameters.component_columns(rhs.level + 1)?;
        let initial = mxx_dsl::parallel(outputs.clone(), |_| {
            RnsBggMatrix::new(std::array::from_fn(|_| {
                std::array::from_fn(|_| ring.zero((1, columns)))
            }))
        })?;
        let process = |start: Int, len: usize, accumulated: Family<RnsBggMatrix>| {
            // Finish each selector's consumers before generating the next.
            // Every output shares the same fresh ciphertext/error, while only
            // the accumulated output family crosses the host-RAM boundary.
            mxx_dsl::iterate(len, accumulated, |index, accumulated| {
                let selector = rhs.ciphertext.at(Int::evaluate(active.start().clone())
                    .add(start.clone())
                    .add(index.clone()))?;
                mxx_dsl::parallel(outputs.clone(), |output| {
                    let weight = output
                        .clone()
                        .mul(Int::constant(size))
                        .add(start.clone())
                        .add(index.clone());
                    let previous = accumulated.at(output);
                    let weight = weights.at(weight);
                    RnsBggMatrix::new(std::array::from_fn(|row| {
                        std::array::from_fn(|column| {
                            Mat::multi_row_gemm_accumulate(
                                vec![(1usize, weight.clone(), selector.entry(row, column).clone())],
                                Some(previous.entry(row, column).clone()),
                            )
                        })
                    }))
                })
            })
        };
        let full = size / batch;
        let mut ciphertext = if full > 0 {
            mxx_dsl::iterate(full, initial, |index, accumulated| {
                process(index.mul(Int::constant(batch)), batch, accumulated)
            })?
        } else {
            initial
        };
        if size % batch != 0 {
            ciphertext = process(Int::constant(full * batch), size % batch, ciphertext)?;
        }
        let ciphertext = RnsBggRhsFamily::Values(ciphertext);
        Ok(RnsBggPrivateRhs {
            ciphertext,
            lifted_ciphertext: (),
            rhs_id: id,
            id,
            level: rhs.level,
            context: rhs.context,
        })
    }

    fn prepare_fixed_product(
        &self,
        input: &RnsBggPublicProgramValue,
        rhs: &Self::FixedRhs,
        id: [u8; 32],
    ) -> Result<(RnsBggPublicProgramValue, Self::PreparedFixedProduct), RnsBggError> {
        let key = RnsBggPublicKeyFamily::from_program(input, rhs.ciphertext.count().clone())?;
        if key.range.start() != &0.into() || key.level != rhs.level || key.context != rhs.context {
            return Err(RnsBggError::Context);
        }
        let prepared = self.prepare_mul_private_family(
            &key,
            &ProgramValue::Family(self.prepare_private_rhs_family(rhs, &key.range)?),
            id,
        )?;
        Ok((prepared.output_key().clone().into_program()?, prepared))
    }

    fn evaluate_fixed_product(
        &self,
        input: &RnsBggEncodingProgramValue,
        prepared: &Self::PreparedFixedProduct,
    ) -> Result<RnsBggEncodingProgramValue, RnsBggError> {
        let encoding =
            RnsBggEncodingFamily::from_program(input, prepared.input.range.count().clone())?;
        self.evaluate_mul_private_family(&encoding, prepared)?.into_program()
    }
}

#[derive(Clone, Debug)]
pub struct RnsBggParameters {
    levels: Vec<DCRTPolyParams>,
    drops: Vec<BigUint>,
    encoding_error_sigma: BigDecimal,
    encoding_error_bound: BigUint,
    operations_per_drop: usize,
}

/// Public workload inputs, independent of secret values and sampled noise.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPlanRequest {
    pub ring_dimension: u32,
    pub widths: Vec<usize>,
    pub slots: Vec<usize>,
    #[serde(with = "bigdecimal_string")]
    pub encoding_error_sigma: BigDecimal,
    pub crt_bits: usize,
    pub terminal_depth: usize,
    pub max_crt_depth: usize,
    pub memory_budget_bytes: u64,
    pub terminal_value_bound: BigUint,
    /// Number of logical MulPrivate/LUT operations between physical drops.
    pub operations_per_drop: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggNoiseBound {
    /// Bound on e in c = s*a - p*mu*r + e.
    pub error: BigUint,
    pub payload_l1: BigUint,
}

impl RnsBggNoiseBound {
    /// Addition is polynomial addition, and does not preserve a monomial
    /// domain without a separately established one-hot selection contract.
    pub fn add(&self, rhs: &Self) -> Self {
        Self { error: &self.error + &rhs.error, payload_l1: &self.payload_l1 + &rhs.payload_l1 }
    }

    /// Negacyclic convolution satisfies ||nu*e||_inf <= ||nu||_1 ||e||_inf.
    pub fn mul_public(&self, multiplier_l1: &BigUint) -> Self {
        Self { error: &self.error * multiplier_l1, payload_l1: &self.payload_l1 * multiplier_l1 }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggParameterManifest {
    pub workload: RnsBggPlanRequest,
    pub ring_dimension: u32,
    #[serde(with = "bigdecimal_string")]
    pub encoding_error_sigma: BigDecimal,
    pub encoding_error_bound: BigUint,
    pub operations_per_drop: usize,
    pub levels: Vec<Vec<u64>>,
    pub base_bits: Vec<u32>,
    pub drops: Vec<BigUint>,
    pub payload_scales: Vec<BigUint>,
    pub noise_bounds: Vec<RnsBggNoiseBound>,
    pub estimated_bytes: BigUint,
    /// Arithmetic acceptance alone never certifies related-secret security.
    pub security_certified: bool,
}

mod bigdecimal_string {
    use bigdecimal::BigDecimal;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::str::FromStr;

    pub fn serialize<S: Serializer>(value: &BigDecimal, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.normalized().to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<BigDecimal, D::Error> {
        let value = String::deserialize(deserializer)?;
        BigDecimal::from_str(&value)
            .map(|value| value.normalized())
            .map_err(serde::de::Error::custom)
    }
}

fn canonical_encoding_sigma(sigma: BigDecimal) -> Result<BigDecimal, RnsBggError> {
    let sigma = sigma.normalized();
    let (_, exponent) = sigma.as_bigint_and_exponent();
    if sigma <= BigDecimal::zero() ||
        exponent > i64::from(u32::MAX) ||
        exponent < -i64::from(u32::MAX)
    {
        return Err(RnsBggError::Parameters("encoding_error_sigma must be positive"));
    }
    Ok(sigma)
}

fn encoding_error_bound(sigma: &BigDecimal) -> Result<BigUint, RnsBggError> {
    let bound = std::panic::catch_unwind(|| {
        mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(sigma)
    })
    .map_err(|_| RnsBggError::Parameters("encoding_error_sigma is unrepresentable"))?;
    if bound.is_zero() {
        return Err(RnsBggError::Parameters("encoding_error_sigma derives a zero cutoff"));
    }
    Ok(bound)
}

impl RnsBggParameters {
    pub fn new(
        levels: Vec<DCRTPolyParams>,
        encoding_error_sigma: impl Into<BigDecimal>,
        operations_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        let sigma = canonical_encoding_sigma(encoding_error_sigma.into())?;
        let bound = encoding_error_bound(&sigma)?;
        let parameters = Self::new_unchecked_schedule(levels, sigma, bound, operations_per_drop)?;
        parameters.validate_schedule()?;
        Ok(parameters)
    }

    fn new_unchecked_schedule(
        levels: Vec<DCRTPolyParams>,
        encoding_error_sigma: BigDecimal,
        encoding_error_bound: BigUint,
        operations_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        if operations_per_drop == 0 {
            return Err(RnsBggError::Parameters("operations_per_drop must be positive"));
        }
        let first = levels.first().ok_or(RnsBggError::Parameters("empty modulus chain"))?;
        if levels.len() < 2 {
            return Err(RnsBggError::Parameters("need a modulus transition"));
        }
        for level in &levels {
            if level.ring_dimension() != first.ring_dimension() || level.dropped_moduli() != 0 {
                return Err(RnsBggError::Parameters("incompatible ring or partial basis"));
            }
        }
        let mut drops = Vec::with_capacity(levels.len());
        for pair in levels.windows(2) {
            if pair[0] == pair[1] {
                drops.push(BigUint::one());
                continue;
            }
            let source = pair[0].to_crt().0;
            let target = pair[1].to_crt().0;
            if target.len() >= source.len() || target.iter().any(|p| !source.contains(p)) {
                return Err(RnsBggError::Parameters(
                    "levels must be equal or strictly nested prime bases",
                ));
            }
            drops.push(pair[0].modulus().as_ref() / pair[1].modulus().as_ref());
        }
        drops.push(BigUint::one());
        Ok(Self { levels, drops, encoding_error_sigma, encoding_error_bound, operations_per_drop })
    }

    fn validate_schedule(&self) -> Result<(), RnsBggError> {
        let stages = self.levels.len() - 1;
        for (operation, pair) in self.levels.windows(2).enumerate() {
            let must_drop =
                (operation + 1) % self.operations_per_drop == 0 || operation + 1 == stages;
            if (pair[0] == pair[1]) == must_drop {
                return Err(RnsBggError::Parameters("modulus chain does not match drop cadence"));
            }
        }
        Ok(())
    }

    pub fn levels(&self) -> &[DCRTPolyParams] {
        &self.levels
    }

    /// Register these single-prime bases alongside the evaluation chain when
    /// the runtime will execute CRT refresh at this output level.
    pub fn refresh_residue_parameters(
        &self,
        level: usize,
    ) -> Result<Vec<DCRTPolyParams>, RnsBggError> {
        let parameters = self.level(level)?;
        parameters
            .to_crt()
            .0
            .into_iter()
            .map(|prime| {
                parameters
                    .select_modulus(&BigUint::from(prime))
                    .ok_or(RnsBggError::Parameters("refresh residue is outside the CRT basis"))
            })
            .collect()
    }
    pub fn level(&self, level: usize) -> Result<&DCRTPolyParams, RnsBggError> {
        self.levels.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }
    pub fn scale(&self, level: usize) -> Result<&BigUint, RnsBggError> {
        self.drops.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }
    /// Each secret component has one full exact gadget row for the next ring.
    /// The chain endpoint is a terminal two-column value, not a resumable state.
    pub fn component_columns(&self, level: usize) -> Result<usize, RnsBggError> {
        self.level(level)?;
        Ok(self.levels.get(level + 1).map_or(1, PolyParams::modulus_digits))
    }
    pub fn encoding_error_sigma_expr(&self) -> RealExpr {
        let (integer, exponent) = self.encoding_error_sigma.as_bigint_and_exponent();
        if exponent >= 0 {
            let denominator = BigInt::from(10u8).pow(exponent as u32);
            RealExpr::Rational(Rational::new(integer, denominator).expect("positive denominator"))
        } else {
            RealExpr::Rational(
                Rational::new(
                    integer * BigInt::from(10u8).pow((-exponent) as u32),
                    BigInt::from(1u8),
                )
                .expect("positive denominator"),
            )
        }
    }
    pub fn encoding_error_sigma(&self) -> &BigDecimal {
        &self.encoding_error_sigma
    }
    pub fn encoding_error_bound(&self) -> &BigUint {
        &self.encoding_error_bound
    }
    pub fn operations_per_drop(&self) -> usize {
        self.operations_per_drop
    }
}

impl RnsBggPlanRequest {
    fn validate(&self) -> Result<(), RnsBggError> {
        if self.operations_per_drop == 0 {
            return Err(RnsBggError::Parameters("operations_per_drop must be positive"));
        }
        let stages = self.widths.len();
        if stages == 0 ||
            self.slots.len() != stages ||
            self.slots.contains(&0) ||
            self.widths.iter().any(|w| !w.is_power_of_two() || *w > self.ring_dimension as usize) ||
            self.terminal_depth == 0 ||
            self.max_crt_depth < self.terminal_depth ||
            canonical_encoding_sigma(self.encoding_error_sigma.clone()).is_err()
        {
            return Err(RnsBggError::Parameters("invalid public workload"));
        }
        DCRTPolyParams::validate_capability(
            self.ring_dimension,
            self.max_crt_depth,
            self.crt_bits,
            1,
        )
        .map_err(|_| RnsBggError::Parameters("unsupported prime pool"))?;
        Ok(())
    }
}

impl RnsBggParameterManifest {
    /// Restore the recorded bases, without generating a new prime pool, and
    /// verify every redundant certificate field before accepting the artifact.
    /// This verifies arithmetic consistency, not artifact authenticity.
    pub fn validate(&self) -> Result<RnsBggParameters, RnsBggError> {
        self.workload.validate()?;
        let workload_sigma = canonical_encoding_sigma(self.workload.encoding_error_sigma.clone())?;
        let manifest_sigma = canonical_encoding_sigma(self.encoding_error_sigma.clone())?;
        if self.ring_dimension != self.workload.ring_dimension ||
            self.encoding_error_sigma != manifest_sigma ||
            self.encoding_error_sigma != workload_sigma ||
            self.encoding_error_bound != encoding_error_bound(&workload_sigma)? ||
            self.operations_per_drop != self.workload.operations_per_drop ||
            self.levels.len() != self.workload.widths.len() + 1 ||
            self.base_bits.len() != self.levels.len() ||
            self.levels
                .iter()
                .any(|base| base.is_empty() || base.len() > self.workload.max_crt_depth)
        {
            return Err(RnsBggError::Parameters("invalid manifest basis dimensions"));
        }
        let levels = self
            .levels
            .iter()
            .zip(&self.base_bits)
            .map(|(base, base_bits)| {
                let bits =
                    base.iter().map(|p| (u64::BITS - p.leading_zeros()) as usize).max().unwrap();
                DCRTPolyParams::try_new(
                    self.ring_dimension,
                    base.len(),
                    bits,
                    *base_bits,
                    Some(base.clone()),
                    None,
                )
                .map_err(|_| RnsBggError::Parameters("invalid manifest prime basis"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = RnsBggParameters::new(
            levels,
            self.encoding_error_sigma.clone(),
            self.workload.operations_per_drop,
        )?;
        if parameters.encoding_error_bound != self.encoding_error_bound {
            return Err(RnsBggError::Parameters("manifest encoding error cutoff mismatch"));
        }
        let stages = self.workload.widths.len();
        for (operation, pair) in parameters.levels.windows(2).enumerate() {
            let must_drop =
                (operation + 1) % self.workload.operations_per_drop == 0 || operation + 1 == stages;
            if (pair[0] == pair[1]) == must_drop {
                return Err(RnsBggError::Parameters("manifest levels do not match drop cadence"));
            }
        }
        if parameters.audit(&self.workload)? != *self {
            return Err(RnsBggError::Parameters(
                "manifest certificate does not match workload and bases",
            ));
        }
        Ok(parameters)
    }
}

#[cfg(test)]
mod parameter_tests {
    use super::*;
    use num_traits::Zero;

    fn request() -> RnsBggPlanRequest {
        RnsBggPlanRequest {
            ring_dimension: std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
                .ok()
                .map(|value| value.parse().unwrap())
                .unwrap_or(16),
            widths: vec![1],
            slots: vec![1],
            encoding_error_sigma: BigDecimal::from(1u8),
            crt_bits: 16,
            terminal_depth: 1,
            max_crt_depth: 16,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        }
    }

    #[test]
    fn test_rns_bgg_manifest_roundtrip_and_certificate_validation() {
        let (parameters, manifest) = RnsBggParameters::plan(&request()).unwrap();
        let encoded = serde_json::to_vec(&manifest).unwrap();
        let decoded: RnsBggParameterManifest = serde_json::from_slice(&encoded).unwrap();
        let restored = decoded.validate().unwrap();
        assert_eq!(decoded, manifest);
        assert_eq!(restored.drops, parameters.drops);
        assert_eq!(parameters.encoding_error_bound, BigUint::from(6u8));
        for (actual, expected) in restored.levels.iter().zip(&parameters.levels) {
            assert_eq!(actual.to_crt(), expected.to_crt());
            assert_eq!(actual.modulus(), expected.modulus());
            assert_eq!(actual.base_bits(), expected.base_bits());
        }
        // Each mutation changes an independently checked artifact claim.
        let mut invalid = Vec::new();
        let mut changed = manifest.clone();
        changed.noise_bounds[1].error = BigUint::zero();
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.noise_bounds[1].payload_l1 += 1u8;
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.drops[0] += 1u8;
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.payload_scales[1] += 1u8;
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.estimated_bytes = BigUint::zero();
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.security_certified = true;
        invalid.push(changed);
        let mut changed = manifest.clone();
        let duplicate = changed.levels[0][0];
        changed.levels[0].push(duplicate);
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.levels[1] = changed.levels[0].clone();
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.workload.slots[0] += 1;
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.ring_dimension *= 2;
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.encoding_error_sigma = BigDecimal::from(2u8);
        invalid.push(changed);
        let mut changed = manifest.clone();
        changed.operations_per_drop += 1;
        invalid.push(changed);
        for (index, changed) in invalid.into_iter().enumerate() {
            assert!(changed.validate().is_err(), "accepted altered certificate {index}");
        }
    }

    #[test]
    fn test_rns_bgg_encoding_sigma_contract_and_manifest_tamper_detection() {
        let mut workload = request();
        workload.encoding_error_sigma = "4.0".parse().unwrap();
        let (parameters, manifest) = RnsBggParameters::plan(&workload).unwrap();
        assert_eq!(parameters.encoding_error_sigma.to_string(), "4");
        assert_eq!(parameters.encoding_error_bound, BigUint::from(26u8));
        assert_eq!(manifest.encoding_error_bound, BigUint::from(26u8));

        let mut changed = manifest.clone();
        changed.encoding_error_sigma = "4.1".parse().unwrap();
        assert!(changed.validate().is_err());
        let mut changed = manifest;
        changed.encoding_error_bound = BigUint::from(25u8);
        assert!(changed.validate().is_err());
        assert!(
            RnsBggParameters::new(
                parameters.levels().to_vec(),
                "0.01".parse::<BigDecimal>().unwrap(),
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn test_rns_bgg_parameter_identity_binds_canonical_sigma_and_cutoff() {
        let base = request();
        let levels = RnsBggParameters::plan(&base).unwrap().0.levels().to_vec();
        let four = RnsBggScheme::new(
            RnsBggParameters::new(levels.clone(), BigDecimal::from(4u8), 1).unwrap(),
            [7u8; 32],
        );
        let four_point_zero = RnsBggScheme::new(
            RnsBggParameters::new(levels.clone(), "4.0".parse::<BigDecimal>().unwrap(), 1).unwrap(),
            [7u8; 32],
        );
        let four_point_one = RnsBggScheme::new(
            RnsBggParameters::new(levels, "4.1".parse::<BigDecimal>().unwrap(), 1).unwrap(),
            [7u8; 32],
        );
        assert_eq!(four.parameter_identity(), four_point_zero.parameter_identity());
        assert_ne!(four.parameter_identity(), four_point_one.parameter_identity());
        assert_eq!(four.parameters().encoding_error_bound, BigUint::from(26u8));
        assert_eq!(four_point_one.parameters().encoding_error_bound, BigUint::from(26u8));
    }

    #[test]
    fn test_rns_bgg_planner_rejects_insufficient_public_budgets() {
        let base = request();
        let mut changed = base.clone();
        changed.max_crt_depth = changed.terminal_depth;
        assert!(matches!(RnsBggParameters::plan(&changed), Err(RnsBggError::Budget("CRT depth"))));
        let mut changed = base.clone();
        changed.memory_budget_bytes = 0;
        assert!(matches!(RnsBggParameters::plan(&changed), Err(RnsBggError::Budget("memory"))));
        let mut changed = base.clone();
        changed.terminal_value_bound = BigUint::from(1u8) << 128;
        assert!(matches!(
            RnsBggParameters::plan(&changed),
            Err(RnsBggError::Budget("terminal range"))
        ));
        let mut changed = base;
        changed.slots[0] = usize::MAX;
        assert!(matches!(
            RnsBggParameters::plan(&changed),
            Err(RnsBggError::Budget("evaluation noise"))
        ));
    }

    #[test]
    fn test_rns_bgg_cadence_repeats_bases_and_rejects_invalid_schedule() {
        let mut workload = request();
        workload.widths = vec![1; 3];
        workload.slots = vec![1; 3];
        workload.operations_per_drop = 2;
        let (parameters, manifest) = RnsBggParameters::plan(&workload).unwrap();
        assert_eq!(parameters.operations_per_drop(), 2);
        assert_eq!(parameters.levels()[0], parameters.levels()[1]);
        assert_ne!(parameters.levels()[1], parameters.levels()[2]);
        assert_ne!(parameters.levels()[2], parameters.levels()[3]);
        assert_eq!(parameters.scale(0).unwrap(), &BigUint::one());
        manifest.validate().unwrap();

        let mut invalid = manifest;
        invalid.levels[2] = invalid.levels[1].clone();
        assert!(invalid.validate().is_err());

        workload.operations_per_drop = 0;
        assert!(matches!(
            RnsBggParameters::plan(&workload),
            Err(RnsBggError::Parameters("operations_per_drop must be positive"))
        ));

        let mut short = request();
        short.widths = vec![1; 2];
        short.slots = vec![1; 2];
        short.operations_per_drop = 3;
        let (short_parameters, _) = RnsBggParameters::plan(&short).unwrap();
        assert_eq!(short_parameters.levels()[0], short_parameters.levels()[1]);
        assert_ne!(short_parameters.levels()[1], short_parameters.levels()[2]);

        let n = workload.ring_dimension;
        let source = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let target = DCRTPolyParams::new(n, 2, 17, 3, Some(source.to_crt().0), None);
        assert!(matches!(
            RnsBggParameters::new(vec![source, target], 1, 2),
            Err(RnsBggError::Parameters("levels must be equal or strictly nested prime bases"))
        ));
    }
}

#[derive(Clone)]
pub struct Blocks<const R: usize, const C: usize, T = Mat> {
    entries: [[T; C]; R],
    matrix_type: MatrixType,
}

pub type RnsBggRow = Blocks<1, 2>;

pub type RnsBggMatrix = Blocks<2, 2>;
pub type RnsBggDigits = Blocks<2, 2, Preimage>;

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[[T; C]; R]: Serialize",
    deserialize = "[[T; C]; R]: Deserialize<'de>"
))]
pub struct BlocksType<const R: usize, const C: usize, T = MatType>([[T; C]; R]);

impl<const R: usize, const C: usize, T: GraphValueSchema> BlocksType<R, C, T> {
    pub fn matrix_type(&self) -> MatrixType {
        let wire = self.0[0][0].wire_types().into_iter().next().expect("matrix block");
        let mut matrix = match wire {
            WireType::Matrix(matrix) | WireType::Preimage { matrix, .. } => matrix,
            _ => unreachable!("blocks contain matrix graph values"),
        };
        matrix.rows = (matrix.rows * IntExpr::constant(R)).canonicalize();
        matrix.columns = (matrix.columns * IntExpr::constant(C)).canonicalize();
        matrix
    }
}

/// A structural product of branch fields, not a nested indexed family.
#[derive(Clone)]
pub struct RnsBggBranches(pub Vec<RnsBggMatrix>);

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggBranchesType(Vec<BlocksType<2, 2>>);

impl GraphValue for RnsBggBranches {
    type Schema = RnsBggBranchesType;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.0.iter().flat_map(GraphValue::flatten).collect()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggBranchesType(self.0.iter().map(GraphValue::schema).collect())
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        if schema.0.is_empty() || values.len() != 4 * schema.0.len() {
            return Err(DslError::Schema);
        }
        Ok(Self(
            schema
                .0
                .iter()
                .zip(values.chunks_exact(4))
                .map(|(schema, values)| RnsBggMatrix::from_values(schema, values))
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl GraphValueSchema for RnsBggBranchesType {
    type Value = RnsBggBranches;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggBranches(self.0.iter().map(|schema| schema.placeholders_from(next)).collect())
    }
    fn wire_types(&self) -> Vec<WireType> {
        self.0.iter().flat_map(GraphValueSchema::wire_types).collect()
    }
}

impl<const R: usize, const C: usize, T: GraphValue> Blocks<R, C, T> {
    pub fn new(entries: [[T; C]; R]) -> Result<Self, DslError> {
        if R == 0 || C == 0 {
            return Err(DslError::Schema);
        }
        let schema = entries[0][0].schema().wire_types();
        if entries.iter().flatten().any(|entry| entry.schema().wire_types() != schema) {
            return Err(DslError::Schema);
        }
        let matrix_type = BlocksType::<R, C, T::Schema>(std::array::from_fn(|row| {
            std::array::from_fn(|column| entries[row][column].schema())
        }))
        .matrix_type();
        Ok(Self { entries, matrix_type })
    }

    pub fn entry(&self, row: usize, column: usize) -> &T {
        &self.entries[row][column]
    }
    pub fn matrix_type(&self) -> &MatrixType {
        &self.matrix_type
    }

    fn map<U: GraphValue>(self, mut operation: impl FnMut(T) -> U) -> Blocks<R, C, U> {
        Blocks::new(self.entries.map(|row| row.map(&mut operation)))
            .expect("block operation preserves a common entry schema")
    }
}

impl<const R: usize, const C: usize> Blocks<R, C> {
    pub fn hash(ring: &Ring, key: Bytes, tag: impl Into<HashTag>, columns: usize) -> Self {
        let root = tag.into();
        Self::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                let mut tag = root.clone();
                tag.push("rns_bgg-block");
                tag.push(IntExpr::constant(row));
                tag.push(IntExpr::constant(column));
                ring.hash_matrix(key.clone(), tag, (1, columns))
            })
        }))
        .expect("one common scalar ring")
    }

    pub fn ring_automorphism(self, index: impl Into<IntExpr>) -> Self {
        let index = index.into();
        self.map(|value| value.ring_automorphism(index.clone()))
    }
    pub fn reduce_modulus(self, modulus: impl Into<IntExpr>) -> Self {
        let modulus = modulus.into();
        self.map(|value| value.reduce_modulus(modulus.clone()))
    }
    pub fn centered_extend(self, modulus: impl Into<IntExpr>) -> Self {
        let modulus = modulus.into();
        self.map(|value| value.centered_extend(modulus.clone()))
    }
    pub fn block_mod_switch(self, modulus: impl Into<IntExpr>) -> Self {
        let modulus = modulus.into();
        if self.entries[0][0].matrix_type().modulus == modulus {
            return self;
        }
        self.map(|value| value.block_mod_switch(modulus.clone(), 1))
    }
}

impl RnsBggMatrix {
    pub fn from_rows(top: RnsBggRow, bottom: RnsBggRow) -> Self {
        Self::new([
            top.entries.into_iter().next().expect("one row"),
            bottom.entries.into_iter().next().expect("one row"),
        ])
        .expect("rows in one ring")
    }
}

impl<const R: usize, const C: usize> Add for Blocks<R, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut left = self.entries.into_iter();
        let mut right = rhs.entries.into_iter();
        Self::new(std::array::from_fn(|_| {
            let mut left = left.next().expect("fixed rows").into_iter();
            let mut right = right.next().expect("fixed rows").into_iter();
            std::array::from_fn(|_| {
                left.next().expect("fixed columns") + right.next().expect("fixed columns")
            })
        }))
        .expect("block sum in one ring")
    }
}

impl<const R: usize, const C: usize> Neg for Blocks<R, C> {
    type Output = Self;
    fn neg(self) -> Self {
        self.map(|entry| -entry)
    }
}

impl<const R: usize, const C: usize> Sub for Blocks<R, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl<const R: usize, const C: usize> Mul<Blocks<R, C>> for Mat {
    type Output = Blocks<R, C>;
    fn mul(self, rhs: Blocks<R, C>) -> Self::Output {
        rhs.map(|entry| self.clone() * entry)
    }
}

impl<const R: usize, const C: usize, const K: usize> Mul<Blocks<C, K>> for Blocks<R, C> {
    type Output = Blocks<R, K>;
    fn mul(self, rhs: Blocks<C, K>) -> Self::Output {
        Blocks::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (0..C)
                    .map(|inner| {
                        self.entries[row][inner].clone() * rhs.entries[inner][column].clone()
                    })
                    .reduce(|left, right| left + right)
                    .expect("nonempty inner dimension")
            })
        }))
        .expect("block product in one ring")
    }
}

impl<const R: usize, const C: usize> Blocks<R, C, Preimage> {
    pub fn centered_extend(self, modulus: impl Into<IntExpr>) -> Self {
        let modulus = modulus.into();
        self.map(|value| value.centered_extend(modulus.clone()))
    }
}

impl<const R: usize, const C: usize, const K: usize> Mul<Blocks<C, K, Preimage>> for Blocks<R, C> {
    type Output = Blocks<R, K>;
    fn mul(self, rhs: Blocks<C, K, Preimage>) -> Self::Output {
        Blocks::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (0..C)
                    .map(|inner| {
                        rhs.entries[inner][column]
                            .clone()
                            .mul_small_rhs(self.entries[row][inner].clone())
                    })
                    .reduce(|left, right| left + right)
                    .expect("nonempty inner dimension")
            })
        }))
        .expect("bounded block product in one ring")
    }
}

impl<const R: usize, const C: usize, T: GraphValue> GraphValue for Blocks<R, C, T> {
    type Schema = BlocksType<R, C, T::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.entries.iter().flatten().flat_map(GraphValue::flatten).collect()
    }
    fn schema(&self) -> Self::Schema {
        BlocksType(std::array::from_fn(|row| {
            std::array::from_fn(|column| self.entries[row][column].schema())
        }))
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        if values.len() != R * C || R == 0 || C == 0 {
            return Err(DslError::Schema);
        }
        let rows = (0..R)
            .map(|row| {
                (0..C)
                    .map(|column| {
                        T::from_values(
                            &schema.0[row][column],
                            &values[row * C + column..row * C + column + 1],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .map_err(|_| DslError::Schema)
            })
            .collect::<Result<Vec<[T; C]>, _>>()?;
        Self::new(rows.try_into().map_err(|_| DslError::Schema)?)
    }
}

impl<const R: usize, const C: usize, T: GraphValueSchema> GraphValueSchema for BlocksType<R, C, T> {
    type Value = Blocks<R, C, T::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        Blocks::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| self.0[row][column].placeholders_from(next))
        }))
        .expect("validated block schema")
    }
    fn wire_types(&self) -> Vec<WireType> {
        self.0.iter().flatten().flat_map(GraphValueSchema::wire_types).collect()
    }
}

#[cfg(test)]
mod block_tests {
    use super::*;
    use mxx_dsl::{DslContext, Family};
    use mxx_ir_core::{ParamEnv, node::ConcatAxis};
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_polynomial_blocks_match_matrix_primitives() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let high = DCRTPolyParams::new(n, 3, 17, 1, None, None);
        let low = high.select_modulus(&high.to_crt().0[0].into()).unwrap();
        let high_modulus = BigInt::from(high.modulus().as_ref().clone());
        let low_modulus = BigInt::from(low.modulus().as_ref().clone());
        let ring = Ring::new(high_modulus.clone(), n as usize);
        let left = RnsBggMatrix::new(std::array::from_fn(|_| {
            std::array::from_fn(|_| ring.uniform_residue((1, 1)))
        }))
        .unwrap();
        let right = RnsBggMatrix::new(std::array::from_fn(|_| {
            std::array::from_fn(|_| ring.uniform_residue((1, 1)))
        }))
        .unwrap();
        // Concatenation is confined to this independent test oracle. Production
        // block arithmetic never reconstructs a dense matrix to slice it again.
        let dense = |blocks: &RnsBggMatrix| {
            Mat::concat(
                ConcatAxis::Rows,
                (0..2)
                    .map(|row| {
                        Mat::concat(
                            ConcatAxis::Columns,
                            (0..2).map(|col| blocks.entry(row, col).clone()).collect(),
                        )
                    })
                    .collect(),
            )
        };
        let expected = (dense(&left) * dense(&right))
            .block_mod_switch(low_modulus.clone(), 1)
            .centered_extend(high_modulus.clone());
        let inputs = Family::pack(vec![left.clone(), right.clone()]).unwrap();
        let products = mxx_dsl::parallel(2, |index| {
            Ok((inputs.at(&index) * right.clone())
                .block_mod_switch(low_modulus.clone())
                .centered_extend(high_modulus.clone()))
        })
        .unwrap();
        let actual = products.at(0);
        let mut graph = DslContext::new("rns_bgg-block-matrix-arithmetic")
            .output("expected", expected)
            .unwrap();
        for row in 0..2 {
            for column in 0..2 {
                graph = graph
                    .output(format!("actual-{row}-{column}"), actual.entry(row, column).clone())
                    .unwrap();
            }
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend([high, low]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(expected) = &results.outputs["expected"] else { panic!("matrix") };
        for row in 0..2 {
            for column in 0..2 {
                let RuntimeValue::Matrix(actual) =
                    &results.outputs[&format!("actual-{row}-{column}")]
                else {
                    panic!("polynomial")
                };
                assert_eq!(
                    actual.block_entries(0..1, 0..1)[0][0],
                    expected.block_entries(row..row + 1, column..column + 1)[0][0]
                );
            }
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
enum LinearOperation<M = Mat> {
    Add,
    MulPublic(M),
}

/// Public preparation binds both operand keys and the public multiplier.
/// Online evaluation cannot replace that multiplier or reinterpret an addition.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPreparedLinear<K = RnsBggPublicKey, M = Mat> {
    lhs: K,
    rhs: K,
    output: K,
    operation: LinearOperation<M>,
}

impl<K, M> RnsBggPreparedLinear<K, M> {
    pub fn output_key(&self) -> &K {
        &self.output
    }
}

#[derive(Clone)]
pub struct RnsBggPayloadSecret {
    pub(crate) row: RnsBggRow,
    context: [u8; 32],
}

#[derive(Clone)]
pub struct RnsBggMaskSecret {
    pub(crate) row: RnsBggRow,
    context: [u8; 32],
}

/// Private payload-secret contribution prepared before the online mask secret
/// exists. This is not a public helper: it must never be exported publicly.
#[derive(Clone)]
pub struct RnsBggInputMaterial {
    pub(crate) row: RnsBggRow,
    level: usize,
    context: [u8; 32],
    key_id: [u8; 32],
}

#[cfg(test)]
pub(crate) struct TestSecrets {
    pub mask: RnsBggMaskSecret,
    pub payload: RnsBggPayloadSecret,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPublicKey<M = RnsBggMatrix> {
    pub(crate) matrix: M,
    pub(crate) level: usize,
    pub(crate) id: [u8; 32],
    pub(crate) context: [u8; 32],
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggEncoding<R = RnsBggRow> {
    pub(crate) row: R,
    pub(crate) level: usize,
    pub(crate) key_id: [u8; 32],
    pub(crate) context: [u8; 32],
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct MaskBranch<M = RnsBggMatrix> {
    sigma: usize,
    public: M,
}

#[derive(Clone)]
pub struct RnsBggMaskBank {
    id: [u8; 32],
    level: usize,
    context: [u8; 32],
    branches: Vec<MaskBranch>,
}

/// s-dependent rows prepared once per public mask bank and online secret.
/// The caller uses the same s for these rows and the input encodings.
#[derive(Clone, PartialEq)]
pub struct RnsBggOnlineMasks<R = RnsBggRow> {
    id: [u8; 32],
    level: usize,
    context: [u8; 32],
    rows: Arc<std::collections::BTreeMap<usize, R>>,
}

/// A fresh, fixed RHS and its mask-alignment material. `table=None` denotes
/// the identity function, and requires no registered identity table.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggLutMaterial<M = RnsBggMatrix> {
    pub(crate) mask_bank: [u8; 32],
    pub(crate) id: [u8; 32],
    pub(crate) level: usize,
    pub(crate) context: [u8; 32],
    pub(crate) table: Option<[u8; 32]>,
    pub(crate) rhs_id: Option<[u8; 32]>,
    branches: Vec<(MaskBranch<M>, M)>,
}

/// Encryption-time plaintext paired with a public artifact identity. The
/// identity, never the plaintext, is supplied to the online program binding.
pub struct RnsBggRhsPlaintext {
    pub payload: Mat,
    pub id: [u8; 32],
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct PreparedBranch<M = RnsBggMatrix, D = RnsBggDigits> {
    sigma: usize,
    // U and V are prepared once and reused online as bounded right actions.
    lifted_ciphertext: D,
    lifted_intermediate: D,
    mask_public: M,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPreparedLut<K = RnsBggPublicKey, M = RnsBggMatrix, D = RnsBggDigits> {
    binding: RnsBggLutBinding,
    input: K,
    output: K,
    branches: Vec<PreparedBranch<M, D>>,
}

impl RnsBggPreparedLut {
    pub fn mask_bank(&self) -> RnsBggMaskBank {
        RnsBggMaskBank {
            id: self.binding.mask_bank,
            level: self.input.level,
            context: self.input.context,
            branches: self
                .branches
                .iter()
                .map(|branch| MaskBranch {
                    sigma: branch.sigma,
                    public: branch.mask_public.clone(),
                })
                .collect(),
        }
    }
}

/// Scheme context identity is public metadata, independent of secret values.
/// Persist this identity with setup artifacts to reproduce their context.
#[derive(Clone)]
pub struct RnsBggScheme {
    parameter_identity: std::sync::OnceLock<[u8; 32]>,
    parameters: Arc<RnsBggParameters>,
    pub(crate) context: [u8; 32],
    encoding_errors_noiseless: bool,
}

impl RnsBggPublicKey {
    pub fn matrix(&self) -> &RnsBggMatrix {
        &self.matrix
    }
    pub fn level(&self) -> usize {
        self.level
    }
    pub fn id(&self) -> &[u8; 32] {
        &self.id
    }
}

impl RnsBggEncoding {
    pub fn row(&self) -> &RnsBggRow {
        &self.row
    }
    pub fn level(&self) -> usize {
        self.level
    }
}

impl RnsBggLutMaterial {
    pub fn table_commitment(&self) -> Option<&[u8; 32]> {
        self.table.as_ref()
    }
    pub fn has_rhs(&self) -> bool {
        self.rhs_id.is_some()
    }
}

impl RnsBggPreparedLut {
    pub fn binding(&self) -> &RnsBggLutBinding {
        &self.binding
    }
    pub fn output_key(&self) -> &RnsBggPublicKey {
        &self.output
    }
}

impl RnsBggScheme {
    pub fn new(parameters: RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameter_identity: std::sync::OnceLock::new(),
            parameters: Arc::new(parameters),
            context,
            encoding_errors_noiseless: false,
        }
    }
    /// Construct a scheme with encoding noise disabled for isolated diagnostics.
    /// This does not alter parameter identity or any production constructor.
    #[cfg(feature = "test-utils")]
    pub fn new_noiseless_for_testing(parameters: RnsBggParameters, context: [u8; 32]) -> Self {
        let mut scheme = Self::new(parameters, context);
        scheme.encoding_errors_noiseless = true;
        scheme
    }
    pub fn parameters(&self) -> &RnsBggParameters {
        &self.parameters
    }
    pub fn ring(&self, level: usize) -> Result<Ring, RnsBggError> {
        let parameters = self.parameters.level(level)?;
        Ok(Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        ))
    }
    fn prepare_digits(
        &self,
        level: usize,
        value: RnsBggMatrix,
    ) -> Result<RnsBggDigits, RnsBggError> {
        let high = self.parameters.level(level)?;
        let low = self.parameters.level(level + 1)?;
        Ok(value
            .map(|entry| entry.decompose(1u64 << low.base_bits(), low.modulus_digits()))
            .centered_extend(BigInt::from(high.modulus().as_ref().clone())))
    }
    fn state_gadget(&self, level: usize) -> Result<Mat, RnsBggError> {
        let high = self.parameters.level(level)?;
        match self.parameters.levels.get(level + 1) {
            Some(low) => Ok(self
                .ring(level + 1)?
                .gadget(1, 1u64 << low.base_bits(), low.modulus_digits())
                .centered_extend(BigInt::from(high.modulus().as_ref().clone()))),
            None => Ok(self.ring(level)?.identity(1)),
        }
    }
    pub fn scalar(&self, level: usize, value: impl Into<BigInt>) -> Result<Mat, RnsBggError> {
        Ok(self.ring(level)?.polynomial([mxx_ir_core::IntExpr::constant(value.into())]))
    }
}

pub(self) fn sum<T: mxx_dsl::GraphValue + std::ops::Add<Output = T>>(
    family: Family<T>,
) -> Result<T, RnsBggError> {
    crate::utils::balanced_sum_family(family).map_err(|e| RnsBggError::Graph(e.to_string()))
}

/// A transition to the combined-output base, followed by extraction of one
/// CRT prime. The source is the preceding level; setup owns this contract.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct RnsBggRefreshLevel {
    pub output_level: usize,
    pub prime: BigInt,
}

/// Setup-owned output key and CRT idempotents in its full output ring.
pub struct RnsBggRefreshRecomposition {
    pub output_key: RnsBggPublicKey,
    pub coefficients: Vec<Mat>,
}

impl ExponentLUTRefreshScheme for RnsBggScheme {
    type RefreshRow = RnsBggRow;
    type RefreshDecoder = RnsBggRow;
    type RefreshLevel = RnsBggRefreshLevel;
    type RefreshRecomposition = RnsBggRefreshRecomposition;

    fn add_refresh_rows(&self, lhs: RnsBggRow, rhs: RnsBggRow) -> RnsBggRow {
        lhs + rhs
    }

    fn switch_refresh_row(&self, row: RnsBggRow, level: &RnsBggRefreshLevel) -> RnsBggRow {
        let q = self.parameters.level(level.output_level).expect("prepared output level").modulus();
        row.block_mod_switch(BigInt::from(q.as_ref().clone()))
    }

    fn apply_refresh_decoder(&self, switched: RnsBggRow, decoder: RnsBggRow) -> RnsBggRow {
        switched - decoder
    }

    fn extract_refresh_residue(&self, row: RnsBggRow, level: &RnsBggRefreshLevel) -> RnsBggRow {
        // Drop all primes except pi: P=H=q/pi. This is coefficient rounding,
        // not modular multiplication by H^-1 in the full output ring.
        row.block_mod_switch(level.prime.clone())
    }

    fn recompose_refresh_rows(
        &self,
        rows: Vec<RnsBggRow>,
        preparation: &RnsBggRefreshRecomposition,
    ) -> Self::Encoding {
        let key = &preparation.output_key;
        let q = key.matrix.matrix_type().modulus.clone();
        // Each source is a single CRT tower. Exact extension and multiplication
        // by its CRT idempotent retain its residue and zero every other tower.
        // No full-width coefficient integer is reconstructed.
        let terms = rows
            .into_iter()
            .zip(&preparation.coefficients)
            .map(|(row, coefficient)| coefficient.clone() * row.centered_extend(q.clone()))
            .collect();
        let row = crate::utils::balanced_sum_family(
            Family::pack(terms).expect("prepared CRT output family"),
        )
        .expect("nonempty prepared CRT basis");
        (RnsBggEncoding { row, level: key.level, key_id: key.id, context: self.context }, None)
    }
    type RefreshPublicRow = RnsBggMatrix;
    type RefreshTargetScale = Mat;

    fn add_refresh_public(&self, lhs: RnsBggMatrix, rhs: RnsBggMatrix) -> RnsBggMatrix {
        lhs + rhs
    }

    fn reduce_refresh_public(
        &self,
        value: RnsBggMatrix,
        level: &RnsBggRefreshLevel,
    ) -> RnsBggMatrix {
        let q = self.parameters.level(level.output_level).expect("prepared output level").modulus();
        value.reduce_modulus(BigInt::from(q.as_ref().clone()))
    }

    fn switch_refresh_public(
        &self,
        value: RnsBggMatrix,
        level: &RnsBggRefreshLevel,
    ) -> RnsBggMatrix {
        let q = self.parameters.level(level.output_level).expect("prepared output level").modulus();
        value.block_mod_switch(BigInt::from(q.as_ref().clone()))
    }

    fn scale_refresh_public(&self, value: RnsBggMatrix, scale: Mat) -> RnsBggMatrix {
        scale * value
    }

    fn subtract_refresh_public(&self, lhs: RnsBggMatrix, rhs: RnsBggMatrix) -> RnsBggMatrix {
        lhs - rhs
    }
}

impl ExponentLUTRefreshDirectScheme for RnsBggScheme {
    type RefreshArtifactBinding = RnsBggArtifactBinding;
    type RefreshOutputKey = RnsBggPublicKey;

    fn refresh_artifact_binding(&self) -> RnsBggArtifactBinding {
        self.artifact_binding()
    }

    fn refresh_output_key(preparation: &RnsBggRefreshRecomposition) -> &RnsBggPublicKey {
        &preparation.output_key
    }

    fn import_refresh_recomposition(
        &self,
        key: RnsBggPublicKey,
    ) -> Result<(RnsBggRefreshRecomposition, BlocksType<2, 2>, BlocksType<1, 2>), RnsBggError> {
        let ring = self.ring(key.level)?;
        let columns = self.parameters.component_columns(key.level)?;
        let target = BlocksType(std::array::from_fn(|_| {
            std::array::from_fn(|_| MatType(ring.matrix_type((1, columns))))
        }));
        let offset = BlocksType(std::array::from_fn(|_| {
            std::array::from_fn(|_| MatType(ring.matrix_type((1, columns))))
        }));
        Ok((self.prepare_refresh_recomposition(key)?, target, offset))
    }

    type RefreshSecret = RnsBggMaskSecret;
    type RefreshDecoderParameters = usize;

    fn offset_refresh_decoder(&self, decoder: RnsBggRow, offset: RnsBggRow) -> RnsBggRow {
        decoder - offset
    }

    fn prepare_direct_refresh_decoder(
        &self,
        secret: &RnsBggMaskSecret,
        target: RnsBggMatrix,
        level: &RnsBggRefreshLevel,
        eta: &usize,
    ) -> RnsBggRow {
        let ring = self.ring(level.output_level).expect("prepared output level");
        let mask = self.mask_at(secret, level.output_level).expect("prepared secret context");
        if *eta == 0 {
            return mask * target;
        }
        let columns = target.entry(0, 0).matrix_type().columns.clone();
        let bits = mxx_dsl::parallel(*eta, |_| {
            RnsBggRow::new([std::array::from_fn(|_| {
                ring.uniform_interval((1, columns.clone()), 0, 1) -
                    ring.uniform_interval((1, columns.clone()), 0, 1)
            })])
        })
        .expect("prepared decoder distribution");
        let error = crate::utils::balanced_sum_family(bits).expect("positive decoder eta");
        mask * target + error
    }
    type RefreshDigitSecret = RnsBggPayloadSecret;
    type RefreshDigitMaskBank = RnsBggMaskBank;

    fn sample_refresh_digit_material(
        &self,
        secret: &Self::RefreshDigitSecret,
        readout: &crate::readout::ValueReadout,
        bank: &Self::RefreshDigitMaskBank,
        routing: &crate::noise_refresh::refresh::RefreshDigitRouting,
        hash_key: mxx_dsl::Bytes,
        tag: mxx_dsl::HashTag,
        id: [u8; 32],
    ) -> Result<
        (ProgramValue<Self::TerminalMaterial, Family<Self::TerminalMaterial>>, RnsBggRow),
        RnsBggError,
    > {
        use mxx_ir_core::{IntExpr, node::ConstantMatrix};
        let level = bank.level + 1;
        let ring = self.ring(level)?;
        let n = self.parameters.level(level)?.ring_dimension() as usize;
        let columns = self.parameters.component_columns(level)?;
        let scale = self.scalar(level, routing.scale.clone())?;
        let weights = mxx_dsl::parallel(2 * columns * n * routing.count, |index| {
            let (_, digit, coefficient, component) =
                crate::noise_refresh::refresh::route_coordinates(
                    index.expression()?,
                    2 * columns,
                    n,
                    &[routing.count],
                )
                .map_err(|_| mxx_dsl::DslError::Schema)?;
            let weight = scale *
                ring.constant(
                    (1, 1),
                    ConstantMatrix::PowerOfBase {
                        base: IntExpr::constant(routing.base),
                        exponent: digit,
                    },
                ) *
                ring.constant((1, 1), ConstantMatrix::Rotation { exponent: coefficient });
            let block = component.floor_div(IntExpr::constant(columns));
            let column = component % IntExpr::constant(columns);
            let entry = ring.constant((1, columns), ConstantMatrix::UnitRow { index: column });
            let first = ring.polynomial([IntExpr::constant(1) - block.clone()]);
            let second = ring.polynomial([block]);
            let row = weight * entry;
            RnsBggRow::new([[first * row.clone(), second * row]])
        })
        .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        let maximum_weight =
            (1..routing.count).fold(routing.scale.clone(), |value, _| value * routing.base);
        let material = self.sample_readout_material(
            secret,
            readout,
            bank,
            hash_key,
            tag,
            id,
            Some(RnsBggReadoutWeights::Indexed {
                rows: weights,
                coefficient_bound: maximum_weight,
            }),
            None,
        )?;
        let q = BigInt::from(self.parameters.level(level)?.modulus().as_ref().clone());
        let half = crate::utils::modular_inverse(&2.into(), &q).expect("odd output modulus");
        let offset = routing.offset_numerator() * half;
        let polynomial = ring.polynomial((0..n).map(|_| IntExpr::constant(offset.clone())));
        // Repeat within each wide block; the two secret-component blocks stay
        // separate. This uses the existing matrix concatenation implementation.
        let polynomial =
            Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![polynomial; columns]);
        Ok((
            material.into_terminal(),
            RnsBggRow::new([[polynomial.clone(), polynomial]])
                .map_err(|e| RnsBggError::Graph(e.to_string()))?,
        ))
    }

    fn refresh_terminal_public(
        &self,
        raw: RnsBggTerminalAccumulator<RnsBggMatrix>,
    ) -> RnsBggMatrix {
        raw.value
    }

    fn refresh_terminal_encoding(&self, raw: RnsBggTerminalAccumulator<RnsBggRow>) -> RnsBggRow {
        raw.value
    }
}

/// Terminal stages use the same branch binding as ordinary LUTs, but their
/// output is a raw public accumulator at the input level. No finalized output
/// or s-dependent helper row is persisted here.
#[derive(Clone, Serialize, Deserialize)]
pub struct RnsBggPreparedTerminalDescriptor<
    P = RnsBggPreparedLutDescriptor<<RnsBggPreparedTerminal as GraphValue>::Schema>,
> {
    prepared: P,
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggArtifactBinding {
    parameters: [u8; 32],
    context: [u8; 32],
}

#[cfg(test)]
mod refresh_verification_tests {
    use super::*;
    use crate::noise_refresh::setup::RefreshVerification;
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, node::NodeKind};

    #[test]
    fn refresh_verification_preserves_two_polynomial_ports() {
        let ring = Ring::new(97, 4);
        let row = |name: &str| {
            RnsBggRow::new([[
                ring.input(format!("{name}-0"), (1, 1)),
                ring.input(format!("{name}-1"), (1, 1)),
            ]])
            .unwrap()
        };
        let encoding = row("encoding");
        let anchor = row("anchor");
        let residual = encoding.clone() - anchor.clone();
        let components = (0..2).map(|column| residual.entry(0, column).clone()).collect();
        let verification =
            RefreshVerification::new(encoding, anchor, residual, components, 2, 2).unwrap();
        assert_eq!(verification.decoded().len(), 4);
        verification.validate_decoder_residuals(verification.decoder_residuals()).unwrap();
        let graph = verification
            .add_outputs(DslContext::new("rns_bgg-refresh-verification"), "residual", "decoded")
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ThresholdDecode { .. }))
                .count(),
            2
        );
        assert!(
            !nodes.iter().any(|node| matches!(
                node.kind(),
                NodeKind::Slice { .. } | NodeKind::Concat { .. }
            ))
        );
        for column in 0..2 {
            for prefix in ["refresh-verification-vector", "refresh-verification-anchor", "residual"]
            {
                assert!(graph.graph.outputs().contains_key(&format!("{prefix}.{column}")));
            }
        }
    }
}

pub(self) fn import_blocks<const R: usize, const C: usize>(
    ring: &Ring,
    production: &mxx_ir_core::artifact::ProductionId,
    prefix: &str,
    columns: impl Into<IntExpr>,
) -> Blocks<R, C> {
    let columns = columns.into();
    BlocksType::<R, C>(std::array::from_fn(|_| {
        std::array::from_fn(|_| MatType(ring.matrix_type((1, columns.clone()))))
    }))
    .artifact_input(
        production.clone(),
        prefix,
        mxx_ir_core::artifact::ArtifactConfidentiality::Public,
    )
    .expect("one common artifact block shape")
}

/// Matrices contain only the active range; `range.start` identifies those
/// members in external RHS/public-value families of the full capacity.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPublicKeyFamily<M = ProgramValue<RnsBggMatrix, Family<RnsBggMatrix>>> {
    pub matrices: M,
    pub level: usize,
    pub id: [u8; 32],
    pub context: [u8; 32],
    pub range: FamilyRange,
}

#[derive(Clone)]
pub struct RnsBggEncodingFamily {
    pub rows: ProgramValue<RnsBggRow, Family<RnsBggRow>>,
    pub level: usize,
    pub key_id: [u8; 32],
    pub context: [u8; 32],
    pub range: FamilyRange,
}

pub struct RnsBggRhsPlaintextFamily {
    pub payloads: Family<Mat>,
    pub id: [u8; 32],
}

#[derive(Clone)]
pub struct RnsBggLutMaterialFamily {
    pub(self) binding: RnsBggLutBinding,
    /// Each member carries branch polynomial fields as a structural product.
    pub(self) ciphertexts: Family<RnsBggBranches>,
    pub(self) masks: Vec<MaskBranch>,
}

impl RnsBggLutMaterialFamily {
    /// Resolve one external member without expanding the structural family.
    pub(self) fn at(&self, index: Int) -> RnsBggLutMaterial {
        RnsBggLutMaterial {
            mask_bank: self.binding.mask_bank,
            id: self.binding.material_id,
            level: self.binding.level,
            context: self.binding.context,
            table: self.binding.table,
            rhs_id: self.binding.rhs,
            branches: self.masks.iter().cloned().zip(self.ciphertexts.at(index).0).collect(),
        }
    }

    pub fn binding(&self) -> &RnsBggLutBinding {
        &self.binding
    }
}

#[derive(Clone)]
pub struct RnsBggPreparedLutFamily {
    pub(self) input: RnsBggPublicKeyFamily,
    pub(self) output: RnsBggPublicKeyFamily,
    pub(self) prepared: Family<RnsBggPreparedLut>,
}

impl RnsBggPreparedLutFamily {
    pub fn mask_bank(&self) -> RnsBggMaskBank {
        self.prepared.at(0).mask_bank()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPreparedReduction<I = RnsBggPublicKeyFamily, K = RnsBggPublicKey> {
    pub(self) input: I,
    pub(self) output: K,
}

impl RnsBggPreparedReduction {
    pub fn output_key(&self) -> &RnsBggPublicKey {
        &self.output
    }
}

/// Public metadata used in place of encryption-time RHS plaintexts.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RnsBggLutBinding {
    pub mask_bank: [u8; 32],
    pub material_id: [u8; 32],
    pub level: usize,
    pub context: [u8; 32],
    pub table: Option<[u8; 32]>,
    pub rhs: Option<[u8; 32]>,
}

impl<M> RnsBggLutMaterial<M> {
    pub fn binding(&self) -> RnsBggLutBinding {
        RnsBggLutBinding {
            mask_bank: self.mask_bank,
            material_id: self.id,
            level: self.level,
            context: self.context,
            table: self.table,
            rhs: self.rhs_id,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct RnsBggOnlineLutHelpers<R = RnsBggRow> {
    pub binding: RnsBggLutBinding,
    pub masks: crate::rns_bgg::RnsBggOnlineMasks<R>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum PreparedGate<K = RnsBggPublicKey, M = RnsBggMatrix, P = Mat, D = RnsBggDigits> {
    MulPrivate(RnsBggPreparedPrivateMul<K, M, D>),
    Linear(RnsBggPreparedLinear<K, P>),
    Lut(RnsBggPreparedLut<K, M, D>),
    /// Only the output key is needed to bind an online aggregate.
    Reduction(K),
}

#[derive(Clone, PartialEq)]
pub struct RnsBggProgramCache<G = PreparedGate, P = Mat> {
    pub(self) gates: BTreeMap<ProgramGateInstance, G>,
    pub(self) plaintexts: BTreeMap<ProgramGateInstance, P>,
}

// Gate instances are structured keys; JSON transports ordered entries, not object keys.
impl<G: Serialize, P: Serialize> Serialize for RnsBggProgramCache<G, P> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.gates.iter().collect::<Vec<_>>(), self.plaintexts.iter().collect::<Vec<_>>())
            .serialize(serializer)
    }
}
impl<'de, G: Deserialize<'de>, P: Deserialize<'de>> Deserialize<'de> for RnsBggProgramCache<G, P> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (gates, plaintexts) =
            <(Vec<(ProgramGateInstance, G)>, Vec<(ProgramGateInstance, P)>)>::deserialize(
                deserializer,
            )?;
        Ok(Self {
            gates: gates.into_iter().collect(),
            plaintexts: plaintexts.into_iter().collect(),
        })
    }
}

impl<G, P> Default for RnsBggProgramCache<G, P> {
    fn default() -> Self {
        Self { gates: BTreeMap::new(), plaintexts: BTreeMap::new() }
    }
}

pub struct RnsBggPublicEvaluator<'a> {
    scheme: &'a RnsBggScheme,
    cache: &'a mut RnsBggProgramCache,
}

pub struct RnsBggEncodingEvaluator<'a> {
    scheme: &'a RnsBggScheme,
    cache: &'a RnsBggProgramCache,
}

impl ExponentLUTEncodingScheme for RnsBggScheme {
    fn reduce_public(
        &self,
        values: Family<Self::PublicKey>,
        gate: ProgramGateInstance,
        cache: &mut Self::Cache,
    ) -> Result<Self::PublicKey, RnsBggError> {
        let input = values.at(0).0;
        let matrix = crate::utils::balanced_sum_family(values.field(|(key, _)| key.matrix)?)
            .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        let plaintext = if values.schema().element.1.is_some() {
            Some(
                crate::utils::balanced_sum_family(values.field(|(_, known)| known.unwrap())?)
                    .map_err(|e| RnsBggError::Graph(e.to_string()))?,
            )
        } else {
            None
        };
        let output = RnsBggPublicKey {
            matrix,
            level: input.level,
            id: output_id(self.context, b"family-reduce", gate, &[input.id]),
            context: self.context,
        };
        cache.gates.insert(gate, PreparedGate::Reduction(output.clone()));
        if let Some(value) = &plaintext {
            cache.plaintexts.insert(gate, value.clone());
        }
        Ok((output, plaintext))
    }
    fn reduce_encoding(
        &self,
        values: Family<Self::Encoding>,
        gate: ProgramGateInstance,
        cache: &Self::Cache,
    ) -> Result<Self::Encoding, RnsBggError> {
        let Some(PreparedGate::Reduction(output)) = cache.gates.get(&gate) else {
            return Err(RnsBggError::Context);
        };
        let row = crate::utils::balanced_sum_family(values.field(|(value, _)| value.row)?)
            .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok((
            RnsBggEncoding { row, level: output.level, key_id: output.id, context: self.context },
            cache.plaintexts.get(&gate).cloned(),
        ))
    }
    fn encoding_dimensions(
        &self,
        key: &Self::PublicKey,
    ) -> (mxx_ir_core::IntExpr, mxx_ir_core::IntExpr) {
        let shape = key.0.matrix.matrix_type().clone();
        (shape.columns.clone(), shape.ring_dimension.clone())
    }
    type PublicKey = (RnsBggPublicKey, Option<Mat>);
    type Encoding = (RnsBggEncoding, Option<Mat>);
    type PublicRhs = RnsBggPublicRhs;
    type EncodingRhs = RnsBggPublicRhs;
    type PublicHelpers = RnsBggProgramMaterial;
    type EncodingHelpers = RnsBggOnlineLutHelpers;
    type Cache = RnsBggProgramCache;
    type NoiseState = RnsBggNoiseValue;
    type NoiseModel<'a> = &'a crate::rns_bgg::RnsBggParameters;
    type NoiseEvaluator<'a> = crate::rns_bgg::noise::RnsBggNoiseEvaluator<'a>;
    fn noise_evaluator<'a>(
        _: &'a ExponentLutProgram,
        model: Self::NoiseModel<'a>,
        ranges: &'a ProgramFamilyBindings,
    ) -> Self::NoiseEvaluator<'a>
    where
        Self: 'a,
    {
        crate::rns_bgg::noise::RnsBggNoiseEvaluator {
            parameters: model,
            ranges,
            gates: BTreeMap::new(),
        }
    }
    fn reduce_noise<'a>(
        evaluator: &mut Self::NoiseEvaluator<'a>,
        value: &Self::NoiseState,
        range: &FamilyRange,
        gate: ProgramGateInstance,
    ) -> Result<Self::NoiseState, RnsBggError>
    where
        Self: 'a,
    {
        evaluator.reduce_add(value, range, gate)
    }
    type Error = RnsBggError;
    type PublicEvaluator<'a> = RnsBggPublicEvaluator<'a>;
    type EncodingEvaluator<'a> = RnsBggEncodingEvaluator<'a>;

    fn public_evaluator<'a>(
        &'a self,
        cache: &'a mut Self::Cache,
        _shared_cache: &'a Self::Cache,
        _ranges: &'a ProgramFamilyBindings,
    ) -> Self::PublicEvaluator<'a> {
        RnsBggPublicEvaluator { scheme: self, cache }
    }

    fn encoding_evaluator<'a>(
        &'a self,
        cache: &'a Self::Cache,
        _shared_cache: &'a Self::Cache,
        _ranges: &'a ProgramFamilyBindings,
    ) -> Self::EncodingEvaluator<'a> {
        RnsBggEncodingEvaluator { scheme: self, cache }
    }
}

pub(self) fn output_id(
    context: [u8; 32],
    operation: &[u8],
    gate: ProgramGateInstance,
    inputs: &[[u8; 32]],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rns_bgg-program-output");
    hash.update(context);
    hash.update((operation.len() as u64).to_le_bytes());
    hash.update(operation);
    match gate {
        ProgramGateInstance::Gate { step } => {
            hash.update([0]);
            hash.update((step as u64).to_le_bytes());
        }
        ProgramGateInstance::FamilyGate { step, gate, .. } => {
            hash.update([1]);
            hash.update((step as u64).to_le_bytes());
            hash.update((gate as u64).to_le_bytes());
        }
        ProgramGateInstance::FamilyReduction { step } => {
            hash.update([2]);
            hash.update((step as u64).to_le_bytes());
        }
    }
    for input in inputs {
        hash.update(input);
    }
    hash.finalize().into()
}

pub(self) fn check_lut_binding(
    scheme: &RnsBggScheme,
    program: &ExponentLutProgram,
    lut: LutId,
    level: usize,
    binding: &RnsBggLutBinding,
) -> Result<(), RnsBggError> {
    let table = program.lut(lut).ok_or(RnsBggError::Lut)?;
    if binding.context != scheme.context ||
        binding.level != level ||
        binding.table != Some(table.commitment()) ||
        binding.rhs.is_some() ||
        table.rhs_width().is_some()
    {
        return Err(RnsBggError::Context);
    }
    let layout = program.layout().ok_or(RnsBggError::Lut)?;
    use mxx_primitives::poly::PolyParams;
    if layout.exponent.stride_log2 != 0 ||
        layout.ring_dimension_log2 !=
            scheme.parameters.level(level)?.ring_dimension().trailing_zeros()
    {
        return Err(RnsBggError::Parameters("incompatible RNS-BGG LUT layout"));
    }
    Ok(())
}

impl ProgramLoweringTypes for RnsBggPublicEvaluator<'_> {
    type Wire = (RnsBggPublicKey, Option<Mat>);
    type Rhs = RnsBggPublicRhs;
    type HelperSet = RnsBggProgramMaterial;
    type Error = RnsBggError;
}

impl ProgramLoweringTypes for RnsBggEncodingEvaluator<'_> {
    type Wire = (RnsBggEncoding, Option<Mat>);
    type Rhs = RnsBggPublicRhs;
    type HelperSet = RnsBggOnlineLutHelpers;
    type Error = RnsBggError;
}

pub type RnsBggPublicProgramValue =
    ProgramValue<(RnsBggPublicKey, Option<Mat>), Family<(RnsBggPublicKey, Option<Mat>)>>;
pub type RnsBggEncodingProgramValue =
    ProgramValue<(RnsBggEncoding, Option<Mat>), Family<(RnsBggEncoding, Option<Mat>)>>;

/// Public preparation consumes a fixed RHS ciphertext. Online private
/// multiplication binds its identity to the already prepared ciphertext.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum RnsBggPublicRhs<M = RnsBggMatrix, D = RnsBggDigits> {
    Fixed(RnsBggPrivateRhs<M, D>),
    Identity([u8; 32]),
}
impl<M, D> RnsBggPublicRhs<M, D> {
    pub fn id(&self) -> [u8; 32] {
        match self {
            Self::Fixed(rhs) => rhs.rhs_id,
            Self::Identity(id) => *id,
        }
    }
}
pub type RnsBggProgramMaterial = RnsBggLutMaterial;

pub(self) fn lut_plaintext(
    scheme: &RnsBggScheme,
    program: &ExponentLutProgram,
    lut: LutId,
    level: usize,
    mu: &Mat,
) -> Result<Mat, RnsBggError> {
    use mxx_primitives::poly::PolyParams;
    let modulus =
        num_bigint::BigInt::from(scheme.parameters.level(level)?.modulus().as_ref().clone());
    let mu = mu.clone().reduce_modulus(modulus);
    let table = program.lut(lut).ok_or(RnsBggError::Lut)?;
    let layout = crate::layout::ExpLayout {
        stride_log2: 0,
        width_log2: table.values().len().trailing_zeros(),
    };
    let ring = scheme.ring(level)?;
    let sigmas = layout
        .automorphism_indices(scheme.parameters.level(level)?.ring_dimension() as usize)
        .map_err(|_| RnsBggError::Lut)?;
    let terms = sigmas
        .into_iter()
        .map(|sigma| {
            let coefficient = crate::utils::lut_coefficient(&ring, &layout, sigma, table.values())
                .map_err(|_| RnsBggError::Lut)?;
            Ok(coefficient * mu.clone().ring_automorphism(sigma))
        })
        .collect::<Result<Vec<_>, RnsBggError>>()?;
    crate::utils::balanced_sum_family(
        mxx_dsl::Family::pack(terms).map_err(|e| RnsBggError::Graph(e.to_string()))?,
    )
    .map_err(|e| RnsBggError::Graph(e.to_string()))
}

/// Public row-polynomial weight for coefficient routing and digit scaling.
/// Empty coefficient lists represent zero polynomials.
pub type RnsBggReadoutWeight = [Vec<BigInt>; 2];

/// Public weights at the terminal output level. Indexed bounds are setup
/// contracts; no ciphertext coefficient scan is performed. Runtime values retain
/// the indexed rows; serialized metadata uses `()` because those rows are stored
/// as public graph artifacts.
#[derive(Clone, Serialize, Deserialize)]
pub enum RnsBggReadoutWeights<R = Family<RnsBggRow>> {
    Shared(RnsBggReadoutWeight),
    Indexed { rows: R, coefficient_bound: BigUint },
}

/// Public preprocessing ports, bound to a readout and terminal stage.
#[derive(Clone, Serialize, Deserialize)]
pub struct RnsBggPreparedReadoutDescriptor {
    prefix: String,
    identity: ValueReadoutIdentity,
    terminal:
        ProgramValue<RnsBggPreparedTerminalDescriptor, RnsBggPreparedTerminalFamilyDescriptor>,
    weight: RnsBggReadoutWeights<()>,
}

pub struct RnsBggReadoutMaterial {
    mode: RnsBggTerminalFamilyMode,
    identity: ValueReadoutIdentity,
    terminal: ProgramValue<RnsBggTerminalMaterial, Family<RnsBggTerminalMaterial>>,
    target_bound: BigUint,
    weight: RnsBggReadoutWeights,
}

impl RnsBggReadoutMaterial {
    /// Raw targets already include the inverse-two normalization. A refresh
    /// aggregate supplies its own exact integer-distribution offset.
    pub fn into_terminal(
        self,
    ) -> ProgramValue<RnsBggTerminalMaterial, Family<RnsBggTerminalMaterial>> {
        self.terminal
    }
    /// Per-member integer output bound after the public affine offset. Sum mode still reports a
    /// per-member bound here: aggregate planning must sum member target bounds and include
    /// every member's branches in `terminal_noise_bound`.
    pub fn target_bound(&self) -> &BigUint {
        &self.target_bound
    }
}

/// Prepared public readout cache. Common identity and output metadata are owned
/// once; the private terminal variant preserves scalar and structural layouts.
pub struct RnsBggPreparedReadout {
    identity: ValueReadoutIdentity,
    public: RnsBggReadoutPublicOutput,
    terminal: RnsBggReadoutTerminal,
}

enum RnsBggReadoutTerminal {
    Scalar {
        terminal: RnsBggPreparedTerminal,
        offset: RnsBggRow,
    },
    Family {
        terminal: RnsBggPreparedTerminalFamily,
        offset: ProgramValue<RnsBggRow, Family<RnsBggRow>>,
    },
}

impl RnsBggPreparedReadout {
    pub fn mask_bank(&self) -> RnsBggMaskBank {
        match &self.terminal {
            RnsBggReadoutTerminal::Scalar { terminal, .. } => terminal.mask_bank(),
            RnsBggReadoutTerminal::Family { terminal, .. } => terminal.mask_bank(),
        }
    }
    pub fn binding(&self) -> crate::rns_bgg::RnsBggLutBinding {
        match &self.terminal {
            RnsBggReadoutTerminal::Scalar { terminal, .. } => terminal.binding().clone(),
            RnsBggReadoutTerminal::Family { terminal, .. } => terminal.binding(),
        }
    }
    pub fn id(&self) -> [u8; 32] {
        match &self.terminal {
            RnsBggReadoutTerminal::Scalar { terminal, .. } => terminal.id(),
            RnsBggReadoutTerminal::Family { terminal, .. } => terminal.id(),
        }
    }
}

#[derive(Clone)]
pub struct RnsBggReadoutPublicOutput {
    pub values: ProgramValue<RnsBggMatrix, (Family<RnsBggMatrix>, FamilyRange)>,
    level: usize,
    weight: RnsBggReadoutWeights,
}

impl RnsBggReadoutPublicOutput {
    pub fn weight(&self) -> &RnsBggReadoutWeights {
        &self.weight
    }

    pub fn level(&self) -> usize {
        self.level
    }
}

pub struct RnsBggReadoutEncodingOutput {
    pub values: ProgramValue<RnsBggRow, (Family<RnsBggRow>, FamilyRange)>,
    level: usize,
}

impl RnsBggReadoutEncodingOutput {
    pub fn level(&self) -> usize {
        self.level
    }
}

pub(self) fn readout_offset(
    ring: &mxx_dsl::Ring,
    weight: &RnsBggReadoutWeight,
) -> Result<RnsBggRow, RnsBggError> {
    RnsBggRow::new([weight.each_ref().map(|polynomial| {
        if polynomial.is_empty() {
            ring.zero((1, 1))
        } else {
            ring.polynomial(polynomial.iter().cloned().map(Into::into))
        }
    })])
    .map_err(|e| RnsBggError::Graph(e.to_string()))
}

impl ExponentLUTReadoutScheme for RnsBggScheme {
    type ReadoutArtifacts = RnsBggPreparedReadout;
    type ReadoutPreparation = RnsBggReadoutMaterial;
    type ReadoutPublicMasks = ();
    type ReadoutEncodingMasks = RnsBggOnlineMasks;
    type ReadoutPublicOutput = RnsBggReadoutPublicOutput;
    type ReadoutEncodingOutput = RnsBggReadoutEncodingOutput;
    type ReadoutError = RnsBggError;

    fn prepare_readout(
        &self,
        readout: &ValueReadout,
        inputs: &RnsBggPublicProgramValue,
        preparation: &RnsBggReadoutMaterial,
    ) -> Result<RnsBggPreparedReadout, RnsBggError> {
        if preparation.identity != readout.identity() {
            return Err(RnsBggError::Context);
        }
        let level = match inputs {
            RnsBggPublicProgramValue::Scalar((input, _)) => input.level + 1,
            RnsBggPublicProgramValue::Family(keys) => keys.schema().element.0.level + 1,
        };
        let modulus = BigInt::from(self.parameters.level(level)?.modulus().as_ref().clone());
        let half = self.scalar(
            level,
            crate::utils::modular_inverse(&2.into(), &modulus).ok_or(RnsBggError::Lut)?,
        )?;
        let ring = self.ring(level)?;
        let offset = match &preparation.weight {
            RnsBggReadoutWeights::Shared(weight) => {
                ProgramValue::Scalar(half.clone() * readout_offset(&ring, weight)?)
            }
            RnsBggReadoutWeights::Indexed { rows, .. } => ProgramValue::Family(
                mxx_dsl::parallel(rows.count().clone(), |index| Ok(half.clone() * rows.at(index)))
                    .map_err(|e| RnsBggError::Graph(e.to_string()))?,
            ),
        };
        match inputs {
            RnsBggPublicProgramValue::Scalar((input, _)) => {
                let ProgramValue::Scalar(material) = &preparation.terminal else {
                    return Err(RnsBggError::Context);
                };
                let ProgramValue::Scalar(offset) = offset else {
                    return Err(RnsBggError::Context);
                };
                let terminal = self.prepare_terminal(&input, material)?;
                let terminal_public =
                    self.finalize_terminal_public(terminal.public_accumulator())?;
                let public = RnsBggReadoutPublicOutput {
                    values: ProgramValue::Scalar(terminal_public.matrix().clone()),
                    level,
                    weight: preparation.weight.clone(),
                };
                Ok(RnsBggPreparedReadout {
                    identity: preparation.identity,
                    public,
                    terminal: RnsBggReadoutTerminal::Scalar { terminal, offset },
                })
            }
            RnsBggPublicProgramValue::Family(_) => {
                let key = RnsBggPublicKeyFamily::from_program(inputs, 1.into())?;
                let terminal = self.prepare_terminal_family(
                    &key,
                    &preparation.terminal,
                    preparation.mode,
                    None,
                )?;
                let offset = if preparation.mode == RnsBggTerminalFamilyMode::Sum {
                    ProgramValue::Scalar(match offset {
                        ProgramValue::Scalar(row) => {
                            ring.polynomial([key.range.count().clone()]) * row
                        }
                        ProgramValue::Family(rows) => crate::utils::balanced_sum_family(rows)
                            .map_err(|e| RnsBggError::Graph(e.to_string()))?,
                    })
                } else {
                    offset
                };
                let matrices = terminal.public_matrices().clone();
                let public = RnsBggReadoutPublicOutput {
                    values: ProgramValue::Family((matrices, terminal.output_range())),
                    level,
                    weight: preparation.weight.clone(),
                };
                Ok(RnsBggPreparedReadout {
                    identity: preparation.identity,
                    public,
                    terminal: RnsBggReadoutTerminal::Family { terminal, offset },
                })
            }
        }
    }

    fn project_readout(
        &self,
        readout: &ValueReadout,
        inputs: &RnsBggPublicProgramValue,
        artifacts: &RnsBggPreparedReadout,
        _masks: &(),
    ) -> Result<RnsBggReadoutPublicOutput, RnsBggError> {
        match (inputs, &artifacts.terminal) {
            (
                RnsBggPublicProgramValue::Scalar((input, _)),
                RnsBggReadoutTerminal::Scalar { terminal, .. },
            ) => {
                self.check_key(&input)?;
                if artifacts.identity != readout.identity() ||
                    input.id != terminal.input.id ||
                    input.level != terminal.input.level ||
                    input.context != terminal.input.context
                {
                    return Err(RnsBggError::Context);
                }
                Ok(artifacts.public.clone())
            }
            (
                RnsBggPublicProgramValue::Family(_),
                RnsBggReadoutTerminal::Family { terminal, .. },
            ) => {
                let key = RnsBggPublicKeyFamily::from_program(inputs, 1.into())?;
                self.check_public_family(&key)?;
                let expected = terminal.input_key();
                if artifacts.identity != readout.identity() ||
                    key.id != expected.id ||
                    key.level != expected.level ||
                    key.context != expected.context ||
                    key.range != expected.range
                {
                    return Err(RnsBggError::Context);
                }
                Ok(artifacts.public.clone())
            }
            _ => Err(RnsBggError::Context),
        }
    }

    fn evaluate_readout(
        &self,
        readout: &ValueReadout,
        inputs: &RnsBggEncodingProgramValue,
        artifacts: &RnsBggPreparedReadout,
        masks: &RnsBggOnlineMasks,
    ) -> Result<RnsBggReadoutEncodingOutput, RnsBggError> {
        match (inputs, &artifacts.terminal) {
            (
                RnsBggEncodingProgramValue::Scalar((input, _)),
                RnsBggReadoutTerminal::Scalar { terminal, offset },
            ) => {
                if artifacts.identity != readout.identity() {
                    return Err(RnsBggError::Context);
                }
                let raw = self.evaluate_terminal(input, terminal, masks)?;
                // The finalized matrix is already owned by the public readout output.
                // Reuse that handle with the terminal binding; no public arithmetic
                // or additional artifact is needed during online evaluation.
                let terminal_public = RnsBggTerminalPublicKey {
                    matrix: artifacts.public.values.scalar().ok_or(RnsBggError::Context)?.clone(),
                    level: artifacts.public.level,
                    context: terminal.input.context,
                    stages: vec![terminal.id],
                };
                let terminal = self.finalize_terminal_encoding(raw, &terminal_public)?;
                Ok(RnsBggReadoutEncodingOutput {
                    values: ProgramValue::Scalar(terminal.row().clone() + offset.clone()),
                    level: terminal.level(),
                })
            }
            (
                RnsBggEncodingProgramValue::Family(_),
                RnsBggReadoutTerminal::Family { terminal, offset },
            ) => {
                let encoding = RnsBggEncodingFamily::from_program(inputs, 1.into())?;
                if artifacts.identity != readout.identity() {
                    return Err(RnsBggError::Context);
                }
                let evaluated = self.evaluate_terminal_family(&encoding, terminal, masks)?;
                let rows = mxx_dsl::parallel(evaluated.count().clone(), |index| {
                    Ok(evaluated.at(&index) + offset.at(index))
                })
                .map_err(|e| RnsBggError::Graph(e.to_string()))?;
                Ok(RnsBggReadoutEncodingOutput {
                    values: ProgramValue::Family((rows, terminal.output_range())),
                    level: terminal.level(),
                })
            }
            _ => Err(RnsBggError::Context),
        }
    }
}

/// A terminal target, distinct from ordinary state material. Shared and indexed
/// storage use the same marker; construction remains inside this backend.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggTerminalMaterial<M = RnsBggLutMaterial, L = Vec<RnsBggDigits>> {
    pub(self) material: M,
    pub(self) lifted_ciphertexts: L,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPreparedTerminal<K = RnsBggPublicKey, M = RnsBggMatrix, P = M, D = RnsBggDigits> {
    pub(self) binding: RnsBggLutBinding,
    pub(self) input: K,
    pub(self) id: [u8; 32],
    pub(self) branches: Vec<PreparedBranch<M, D>>,
    pub(self) raw_public: P,
}

/// All terms stay at their common input base. Only setup identities grow with
/// the number of terms; a carrier holds one polynomial-block matrix or row.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggTerminalAccumulator<T> {
    pub(self) value: T,
    pub(self) level: usize,
    pub(self) context: [u8; 32],
    pub(self) stages: Vec<[u8; 32]>,
}

pub struct RnsBggTerminalPublicKey {
    pub(self) matrix: RnsBggMatrix,
    pub(self) level: usize,
    pub(self) context: [u8; 32],
    pub(self) stages: Vec<[u8; 32]>,
}

pub struct RnsBggTerminalEncoding {
    row: RnsBggRow,
    level: usize,
}

impl RnsBggTerminalPublicKey {
    pub fn matrix(&self) -> &RnsBggMatrix {
        &self.matrix
    }
    pub fn level(&self) -> usize {
        self.level
    }
}

impl RnsBggTerminalEncoding {
    pub fn row(&self) -> &RnsBggRow {
        &self.row
    }
    pub fn level(&self) -> usize {
        self.level
    }
}

impl<P> RnsBggPreparedTerminal<RnsBggPublicKey, RnsBggMatrix, P> {
    pub fn binding(&self) -> &RnsBggLutBinding {
        &self.binding
    }
    pub fn id(&self) -> [u8; 32] {
        self.id
    }
    /// Public masks suffice to generate fresh helpers for another online s.
    pub fn mask_bank(&self) -> RnsBggMaskBank {
        RnsBggMaskBank {
            id: self.binding.mask_bank,
            level: self.input.level,
            context: self.input.context,
            branches: self
                .branches
                .iter()
                .map(|branch| crate::rns_bgg::MaskBranch {
                    sigma: branch.sigma,
                    public: branch.mask_public.clone(),
                })
                .collect(),
        }
    }
}

impl RnsBggPreparedTerminal {
    pub fn public_accumulator(&self) -> RnsBggTerminalAccumulator<RnsBggMatrix> {
        RnsBggTerminalAccumulator {
            value: self.raw_public.clone(),
            level: self.input.level,
            context: self.input.context,
            stages: vec![self.id],
        }
    }
}

impl<T: std::ops::Add<Output = T>> RnsBggTerminalAccumulator<T> {
    pub fn combine(mut self, other: Self) -> Result<Self, RnsBggError> {
        if self.level != other.level || self.context != other.context {
            return Err(RnsBggError::Context);
        }
        self.value = self.value + other.value;
        self.stages.extend(other.stages);
        Ok(self)
    }
}

impl ExponentLUTTerminalScheme for RnsBggScheme {
    fn reduce_terminal_public(
        &self,
        values: Family<Self::TerminalPublicAccumulator>,
    ) -> Result<Self::TerminalPublicAccumulator, RnsBggError> {
        let metadata = values.schema().element;
        let value = crate::utils::balanced_sum_family(values.field(|value| value.value)?)
            .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(RnsBggTerminalAccumulator {
            value,
            level: metadata.level,
            context: metadata.context,
            stages: metadata.stages,
        })
    }
    fn reduce_terminal_encoding(
        &self,
        values: Family<Self::TerminalEncodingAccumulator>,
    ) -> Result<Self::TerminalEncodingAccumulator, RnsBggError> {
        let metadata = values.schema().element;
        let value = crate::utils::balanced_sum_family(values.field(|value| value.value)?)
            .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(RnsBggTerminalAccumulator {
            value,
            level: metadata.level,
            context: metadata.context,
            stages: metadata.stages,
        })
    }

    type TerminalMaterial = RnsBggTerminalMaterial;
    type TerminalOnlineHelpers = RnsBggOnlineMasks;
    type PreparedTerminal = RnsBggPreparedTerminal<RnsBggPublicKey, RnsBggMatrix, ()>;
    type TerminalPublicAccumulator = RnsBggTerminalAccumulator<RnsBggMatrix>;
    type TerminalEncodingAccumulator = RnsBggTerminalAccumulator<RnsBggRow>;
    type TerminalPublicOutput = RnsBggTerminalPublicKey;
    type TerminalEncodingOutput = RnsBggTerminalEncoding;
    type TerminalDescriptor =
        RnsBggPreparedTerminalDescriptor<(String, <Self::PreparedTerminal as GraphValue>::Schema)>;
    type TerminalBinding = (RnsBggLutBinding, [u8; 32], [u8; 32]);

    fn export_terminal_term(
        &self,
        term: &Self::PreparedTerminal,
        graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, Self::TerminalDescriptor), RnsBggError> {
        Ok((
            graph.public_output(prefix, term.clone())?,
            RnsBggPreparedTerminalDescriptor { prepared: (prefix.to_owned(), term.schema()) },
        ))
    }
    fn import_terminal_term(
        &self,
        descriptor: &Self::TerminalDescriptor,
        production: ProductionId,
        expected: &Self::TerminalBinding,
    ) -> Result<Self::PreparedTerminal, RnsBggError> {
        let (prefix, schema) = &descriptor.prepared;
        if schema.binding != expected.0 ||
            schema.input.id != expected.1 ||
            schema.id != expected.2 ||
            schema.input.context != self.context
        {
            return Err(RnsBggError::Context);
        }
        let term = schema.artifact_input(production, prefix, ArtifactConfidentiality::Public)?;
        self.check_key(&term.input)?;
        Ok(term)
    }
    fn export_terminal_output(
        &self,
        public: &RnsBggTerminalPublicKey,
        graph: mxx_dsl::DslContext,
        prefix: &str,
    ) -> Result<mxx_dsl::DslContext, RnsBggError> {
        Ok(graph.public_output(prefix, public.matrix.clone())?)
    }
    fn import_terminal_output(
        &self,
        terms: &[Self::PreparedTerminal],
        production: mxx_ir_core::artifact::ProductionId,
        prefix: &str,
    ) -> Result<RnsBggTerminalPublicKey, RnsBggError> {
        let input_level = terms[0].input.level;
        if terms.iter().any(|term| term.input.level != input_level) {
            return Err(RnsBggError::Context);
        }
        let level = input_level + 1;
        Ok(RnsBggTerminalPublicKey {
            matrix: crate::rns_bgg::import_blocks(
                &self.ring(level)?,
                &production,
                prefix,
                terms[0].branches[0].lifted_ciphertext.entry(0, 0).matrix_type().columns.clone(),
            ),
            level,
            context: self.context,
            stages: terms.iter().map(|term| term.id).collect(),
        })
    }

    fn prepare_terminal_term(
        &self,
        input: &Self::PublicKey,
        material: &Self::TerminalMaterial,
    ) -> Result<(Self::PreparedTerminal, Self::TerminalPublicAccumulator), RnsBggError> {
        let prepared = self.prepare_terminal(&input.0, material)?;
        let raw = RnsBggTerminalAccumulator {
            value: prepared.raw_public,
            level: prepared.input.level,
            context: prepared.input.context,
            stages: vec![prepared.id],
        };
        let cache = RnsBggPreparedTerminal {
            binding: prepared.binding,
            input: prepared.input,
            id: prepared.id,
            branches: prepared.branches,
            raw_public: (),
        };
        Ok((cache, raw))
    }

    fn terminal_encoding_term(
        &self,
        input: &Self::Encoding,
        prepared: &Self::PreparedTerminal,
        helpers: &Self::TerminalOnlineHelpers,
    ) -> Result<Self::TerminalEncodingAccumulator, RnsBggError> {
        self.evaluate_terminal(&input.0, prepared, helpers)
    }

    fn add_terminal_public(
        &self,
        lhs: Self::TerminalPublicAccumulator,
        rhs: Self::TerminalPublicAccumulator,
    ) -> Result<Self::TerminalPublicAccumulator, RnsBggError> {
        lhs.combine(rhs)
    }

    fn add_terminal_encoding(
        &self,
        lhs: Self::TerminalEncodingAccumulator,
        rhs: Self::TerminalEncodingAccumulator,
    ) -> Result<Self::TerminalEncodingAccumulator, RnsBggError> {
        lhs.combine(rhs)
    }

    fn finish_terminal_public(
        &self,
        raw: Self::TerminalPublicAccumulator,
    ) -> Result<Self::TerminalPublicOutput, RnsBggError> {
        self.finalize_terminal_public(raw)
    }

    fn finish_terminal_encoding(
        &self,
        raw: Self::TerminalEncodingAccumulator,
        public: &Self::TerminalPublicOutput,
    ) -> Result<Self::TerminalEncodingOutput, RnsBggError> {
        self.finalize_terminal_encoding(raw, public)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RnsBggTerminalFamilyMode {
    /// One separately finalized output per input member.
    Independent,
    /// One output containing the sum of raw members, finalized once.
    Sum,
}

pub struct RnsBggPreparedTerminalFamily {
    mode: RnsBggTerminalFamilyMode,
    finalize: bool,
    input: RnsBggPublicKeyFamily,
    id: [u8; 32],
    cache: Family<RnsBggPreparedTerminal<RnsBggPublicKey, RnsBggMatrix, ()>>,
    public: Family<RnsBggMatrix>,
}

impl RnsBggPreparedTerminalFamily {
    pub fn input_key(&self) -> &RnsBggPublicKeyFamily {
        &self.input
    }
    pub fn binding(&self) -> crate::rns_bgg::RnsBggLutBinding {
        self.cache.schema().element.binding
    }
    pub fn id(&self) -> [u8; 32] {
        self.id
    }

    pub fn public_matrices(&self) -> &Family<RnsBggMatrix> {
        &self.public
    }

    pub fn level(&self) -> usize {
        self.input.level + usize::from(self.finalize)
    }

    pub fn mode(&self) -> RnsBggTerminalFamilyMode {
        self.mode
    }

    /// Input range; aggregate mode has a singleton output family.
    pub fn range(&self) -> &crate::program::FamilyRange {
        &self.input.range
    }

    /// Output range preserves independent members or names the singleton sum.
    pub fn output_range(&self) -> crate::program::FamilyRange {
        match self.mode {
            RnsBggTerminalFamilyMode::Independent => self.input.range.clone(),
            RnsBggTerminalFamilyMode::Sum => {
                crate::program::FamilyRange::bounded(0, 1, 1).expect("singleton range")
            }
        }
    }

    pub fn mask_bank(&self) -> RnsBggMaskBank {
        self.cache.at(0).mask_bank()
    }
}

/// Structural terminal ports and shared/indexed public material, bound at setup.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RnsBggPreparedTerminalFamilyDescriptor {
    prefix: String,
    parameters: [u8; 32],
    mode: RnsBggTerminalFamilyMode,
    finalize: bool,
    id: [u8; 32],
    range: FamilyRange,
    cache:
        <Family<RnsBggPreparedTerminal<RnsBggPublicKey, RnsBggMatrix, ()>> as GraphValue>::Schema,
    public: FamilyType<BlocksType<2, 2>>,
}
